//! The batched raw cut-to-cut matrix `M_B` of a block, for the block-cut lift.

use std::collections::{HashMap, HashSet};

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex};
use rayon::prelude::*;

use super::{
    BlockQueryEngine, LcaOracle, ProjectionCache, projection::SourceSeeds, target::TargetSite,
};
use crate::rooted_tree::TreeAccess;
use crate::{
    min_plus::MinPlus,
    orientation::Orientation,
    path::GfaPathLength,
    skeleton::{DijkstraWorkspace, SkeletonAccess},
    topology::SpqrLocation,
};

impl<'a, IndexType: GraphIndexInteger + 'static> BlockQueryEngine<'a, IndexType> {
    /// Build the dense `(2·cuts)²` row-major cut-to-cut distance matrix
    /// (`[(2i+oi)*dim + (2j+oj)] = start_to_start(cut_i^oi -> cut_j^oj)`), the raw `M_B` the block-cut
    /// lift needs, **byte-identical** to `dim*dim` independent `start_to_start` calls but with the
    /// LCA-skeleton Dijkstra hoisted out of the per-target loop.
    ///
    /// The key observation: for a fixed source `a`, one `workspace.run` over the LCA skeleton already
    /// settles `a`'s distance to *every* node of that skeleton, i.e. to all targets `b` whose
    /// `LCA(x_a, x_b)` is that skeleton. Under the warmed projection cache the read of each target is
    /// side-effect-free (`source_seeds`/`target_distance` are cache hits / pure reads — the sole
    /// `workspace` mutation is that one `run`). So we group targets by their LCA and run once per
    /// `(source, LCA)`, cutting the skeleton searches from `(2·cuts)²` to `Σ_sources |distinct LCA|`
    /// (≈ `2·cuts · k`, `k` small in a block's shallow SPQR tree).
    ///
    /// Rows are filled in parallel (one per directed cut, disjoint writes); each rayon worker owns its
    /// `DijkstraWorkspace` (bound the peak with `RAYON_NUM_THREADS`). The projection warm-up is parallel
    /// too.
    pub fn fill_cut_matrix(&self, cuts: &[NodeIndex<IndexType>]) -> Vec<MinPlus<IndexType>>
    where
        IndexType: Send + Sync,
    {
        let fill = CutFill {
            cuts,
            // Per cut: where it sits in the SPQR tree (`None` for a cut not in the block: its row
            // and column are `∞` but for the diagonal, as `start_to_start` returns `None`).
            cut_meta: cuts.iter().map(|&cut| self.spqr_location(cut)).collect(),
        };
        let cache = self.warm_fill_projections(&fill);
        // A short-lived engine that reuses our borrows plus the local warmed cache (its lifetime is
        // bounded by `&cache`, shorter than `'a`; the shared-ref fields downcast covariantly).
        let engine = BlockQueryEngine {
            topology: self.topology,
            block: self.block,
            graph: self.graph,
            projection_cache: Some(&cache),
        };
        let rows: Vec<Vec<MinPlus<IndexType>>> = (0..fill.dim())
            .into_par_iter()
            .map_init(DijkstraWorkspace::new, |workspace, r| {
                engine.fill_row(&fill, r, workspace)
            })
            .collect();
        rows.into_iter().flatten().collect()
    }

    /// Every pole projection the fill reads, computed across the pool: each cut as a source seed
    /// and as a reverse target seed at its deepest SPQR node, the keys `source_seeds` and
    /// `target_distance` look up. With them cached, the fill's reads are side-effect free.
    fn warm_fill_projections(&self, fill: &CutFill<'_, IndexType>) -> ProjectionCache<IndexType>
    where
        IndexType: Send + Sync,
    {
        let mut key_set: HashSet<(DirectedNodeIndex<IndexType>, usize)> = HashSet::new();
        for (i, &cut) in fill.cuts.iter().enumerate() {
            if let Some(x) = fill.cut_meta[i].map(|loc| loc.x) {
                for orientation in Orientation::all() {
                    let dir = cut.into_directed(orientation.is_forward());
                    for seed in [dir, dir.invert()] {
                        key_set.insert((seed, x));
                    }
                }
            }
        }
        let keys: Vec<_> = key_set.into_iter().collect();
        keys.into_par_iter()
            .map_init(DijkstraWorkspace::new, |ws, (seed, x)| {
                ((seed, x), self.pole_projection(seed, x, ws))
            })
            .collect()
    }

    /// Row `r` of the matrix (source `cuts[r / 2]` in orientation `r % 2`): one group per distinct
    /// LCA of the source with a target cut, read by the LCA's segment tree, labels or one shared
    /// skeleton search.
    fn fill_row(
        &self,
        fill: &CutFill<'_, IndexType>,
        r: usize,
        ws: &mut DijkstraWorkspace<IndexType>,
    ) -> Vec<MinPlus<IndexType>> {
        let tree = self.block.tree();
        let mut row = vec![MinPlus::infinity(); fill.dim()];
        let (i, oi) = (r / 2, r % 2);
        if let Some(a_loc) = fill.cut_meta[i] {
            let source = fill.cuts[i].into_directed(Orientation::from_index(oi).is_forward());
            // Group target cuts by LCA(x_a, x_b) (depends on the cut, not its orientation).
            for (lca, js) in fill.lca_groups(tree, a_loc.x) {
                let seeds =
                    self.source_seeds(source, lca, a_loc.x, a_loc.contains_ancestor(tree, lca), ws);
                self.fill_group(fill, &mut row, lca, &js, &seeds, ws);
            }
        }
        row[r] = MinPlus::zero(); // source == target (overrides the computed self-entry)
        row
    }

    /// The entries of one `(source, LCA)` group, as `start_to_start` reads them: a segment tree or
    /// labels need no shared search, else one multi-source search over the LCA skeleton serves every
    /// target.
    fn fill_group(
        &self,
        fill: &CutFill<'_, IndexType>,
        row: &mut [MinPlus<IndexType>],
        lca: usize,
        js: &[usize],
        seeds: &SourceSeeds<IndexType>,
        ws: &mut DijkstraWorkspace<IndexType>,
    ) {
        let tree = self.block.tree();
        match LcaOracle::of(self.block, lca) {
            LcaOracle::SegTree(seg) => {
                fill.for_targets(tree, lca, js, row, |site| {
                    self.target_distance_segtree(site, &seg, seeds, ws)
                });
            }
            LcaOracle::Labels(labels) => {
                fill.for_targets(tree, lca, js, row, |site| {
                    self.target_distance_labeled(site, &labels, seeds, ws)
                });
            }
            LcaOracle::Skeleton(lca_skeleton) => {
                let dense_seeds: Vec<(u32, GfaPathLength<IndexType>)> = seeds
                    .iter()
                    .filter_map(|&(node, d)| lca_skeleton.node_id(node).map(|id| (id, d)))
                    .collect();
                ws.run(&lca_skeleton, &dense_seeds);
                fill.for_targets(tree, lca, js, row, |site| {
                    self.target_distance(site, &lca_skeleton, ws)
                });
            }
        }
    }
}

/// The cuts of one fill and where each sits in the block's SPQR tree.
struct CutFill<'c, IndexType: GraphIndexInteger> {
    cuts: &'c [NodeIndex<IndexType>],
    cut_meta: Vec<Option<SpqrLocation>>,
}

impl<IndexType: GraphIndexInteger> CutFill<'_, IndexType> {
    fn dim(&self) -> usize {
        2 * self.cuts.len()
    }

    /// The target cuts by their LCA with `x_a`.
    fn lca_groups<T: TreeAccess>(&self, tree: &T, x_a: usize) -> HashMap<usize, Vec<usize>> {
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (j, meta) in self.cut_meta.iter().enumerate() {
            if let Some(x_b) = meta.map(|loc| loc.x) {
                groups.entry(tree.lca(x_a, x_b)).or_default().push(j);
            }
        }
        groups
    }

    /// Write `read(site)` for both orientations of every target cut `js` (all in the block) into
    /// `row`.
    fn for_targets<T: TreeAccess>(
        &self,
        tree: &T,
        lca: usize,
        js: &[usize],
        row: &mut [MinPlus<IndexType>],
        mut read: impl FnMut(TargetSite<IndexType>) -> Option<GfaPathLength<IndexType>>,
    ) {
        for &j in js {
            let b_loc = self.cut_meta[j].unwrap();
            let (x_b, b_in_lca) = (b_loc.x, b_loc.contains_ancestor(tree, lca));
            for oj in 0..2 {
                let target = self.cuts[j].into_directed(Orientation::from_index(oj).is_forward());
                row[2 * j + oj] = read(TargetSite {
                    target,
                    lca,
                    x_b,
                    b_in_lca,
                })
                .map(MinPlus::finite)
                .unwrap_or_else(MinPlus::infinity);
            }
        }
    }
}
