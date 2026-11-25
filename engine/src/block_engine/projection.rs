//! Pole projections: an endpoint's distances to a node's reference poles (a segment tree, a
//! projection table or an inner-skeleton Dijkstra), memoized per thread, and the source seeds the
//! pole-step climb lifts them into.

use std::collections::HashMap;

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use super::{
    BlockQueryEngine, ProjectionOracle,
    memo::{MEMO_MIN_SKELETON, with_proj_memo},
    up_to_four::UpToFour,
};
use crate::rooted_tree::TreeAccess;
use crate::{
    min_plus::{MinPlus, wide::INF},
    orientation::Orientation,
    path::GfaPathLength,
    pole_lift::{Pole, PoleVector},
    s_node_segtree::SegTreeAccess,
    skeleton::{DijkstraWorkspace, Skeleton, SkeletonAccess},
};

/// Cache of `BlockQueryEngine::pole_projection` results, keyed by `(directed seed, SPQR node x)`.
/// `pole_projection` is a pure, LCA-independent function of that key but is the dominant per-query cost
/// (a projection Dijkstra over a possibly-huge inner skeleton). The batched cut-matrix fill precomputes
/// it once per (cut, orientation) — as both a source and a reverse-target seed — and reuses it across the
/// `O(cuts^2)` pairs, collapsing ~2M projections to ~2·`dim`. See [`BlockQueryEngine::fill_cut_matrix`].
pub type ProjectionCache<IndexType> =
    HashMap<(DirectedNodeIndex<IndexType>, usize), PoleVector<IndexType>>;

/// A source's seeds in an LCA skeleton: `(directed node, distance)`, at most four, on the stack.
pub(super) type SourceSeeds<IndexType> =
    UpToFour<(DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>)>;

impl<'a, IndexType: GraphIndexInteger + 'static> BlockQueryEngine<'a, IndexType> {
    /// Precompute the pole projection of every cut node — the LCA-independent, dominant per-query
    /// cost — once, for [`with_projection_cache`](Self::with_projection_cache): the serial form of
    /// the warm-up [`Self::fill_cut_matrix`] runs, for the tests. Each cut is projected as both a
    /// *source* seed (`cut^o`) and a *reverse-target* seed (`(cut^o).invert()`), at the cut's deepest
    /// containing SPQR node `x` — exactly the keys the query path looks up. Each projection
    /// overwrites `workspace`. Absent/empty-local cuts are simply skipped (they project to ∞ anyway).
    #[cfg(test)]
    pub fn warm_cut_projections(
        &self,
        cuts: &[bidirected_adjacency_array::index::NodeIndex<IndexType>],
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> ProjectionCache<IndexType> {
        let mut cache = ProjectionCache::default();
        for &cut in cuts {
            let Some(x) = self.spqr_location(cut).map(|loc| loc.x) else {
                continue;
            };
            for orientation in Orientation::all() {
                let dir = cut.into_directed(orientation.is_forward());
                for seed in [dir, dir.invert()] {
                    cache
                        .entry((seed, x))
                        .or_insert_with(|| self.pole_projection(seed, x, workspace));
                }
            }
        }
        cache
    }

    /// Seeds for the source: its distances to nodes present in the LCA skeleton.
    ///
    /// May run a projection search on `workspace`; the result is fully consumed into the returned
    /// seeds, so the caller is free to reuse `workspace` afterwards.
    pub(super) fn source_seeds(
        &self,
        source: DirectedNodeIndex<IndexType>,
        lca: usize,
        x_a: usize,
        a_in_lca: bool,
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> SourceSeeds<IndexType> {
        // If the source segment is in the LCA skeleton, seed it directly (no projection needed).
        if a_in_lca {
            return UpToFour::one((source, GfaPathLength::from_usize(0)));
        }
        let at_x_a = self.pole_projection(source, x_a, workspace);
        self.seeds_from_source_projection(source, lca, x_a, a_in_lca, at_x_a)
    }

    /// The distances from directed node `seed` to `x`'s reference poles: a projection search within
    /// `x`'s subtree (its inner skeleton, which stays inside the subtree). The all-`∞` vector when
    /// `x` is the root (no poles) or `seed` is not present in the skeleton. Overwrites `workspace`.
    pub(super) fn pole_projection(
        &self,
        seed: DirectedNodeIndex<IndexType>,
        x: usize,
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> PoleVector<IndexType> {
        // Cache hit: `pole_projection` is a pure function of `(seed, x)`, so a warmed value is exact.
        if let Some(cache) = self.projection_cache {
            if let Some(cached) = cache.get(&(seed, x)) {
                return *cached;
            }
            // Build path: the fill owns its cache (warmed upfront, read-only). Don't touch the memo.
            return self.pole_projection_uncached(seed, x, workspace);
        }
        // Query path: memoize per `(image, block, seed, x)`. See `PROJ_MEMO` — the lift re-asks for
        // the SAME (seed, x) ~5x per `cross()` sub-phase, and this is the query's dominant cost.
        if self.block.inner(x).node_count() < MEMO_MIN_SKELETON {
            return self.pole_projection_uncached(seed, x, workspace);
        }
        let key = (
            self.block.image_id(),
            self.block.block().into_usize(),
            seed,
            x,
        );
        if let Some(Some(cached)) = with_proj_memo::<IndexType, _>(|memo| memo.get(&key).copied()) {
            return cached;
        }
        let vector = self.pole_projection_uncached(seed, x, workspace);
        with_proj_memo::<IndexType, _>(|memo| memo.insert(key, vector));
        vector
    }

    /// The uncached body of [`Self::pole_projection`].
    fn pole_projection_uncached(
        &self,
        seed: DirectedNodeIndex<IndexType>,
        x: usize,
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> PoleVector<IndexType> {
        match ProjectionOracle::of(self.block, x) {
            ProjectionOracle::Root => PoleVector::infinity(),
            // Two range products, no search.
            ProjectionOracle::SegTree(seg) => match seg.pole_projection(seed) {
                Some(entries) => PoleVector::from_entries(entries),
                None => PoleVector::infinity(),
            },
            // The four pole distances of every state were computed at build: one row read.
            ProjectionOracle::Table { table, inner } => match inner.node_id(seed) {
                Some(id) => {
                    let row = &table[4 * id as usize..4 * id as usize + 4];
                    PoleVector::from_entries([row[0], row[1], row[2], row[3]])
                }
                None => PoleVector::infinity(),
            },
            ProjectionOracle::Skeleton { inner, poles } => match inner.node_id(seed) {
                Some(id) => {
                    // A full search: stopping once the 4 poles settle gains nothing, since some
                    // pole orientations are unreachable and the heap drains anyway.
                    workspace.run(&inner, &[(id, GfaPathLength::from_usize(0))]);
                    pole_vector_from_workspace(&inner, workspace, poles)
                }
                None => PoleVector::infinity(),
            },
        }
    }

    /// Turn a precomputed source pole projection `at_x_a` (from [`Self::pole_projection`]) into LCA
    /// skeleton seeds, by lifting it up to the LCA's child on the path to `x_a`. Does no search.
    fn seeds_from_source_projection(
        &self,
        source: DirectedNodeIndex<IndexType>,
        lca: usize,
        x_a: usize,
        a_in_lca: bool,
        at_x_a: PoleVector<IndexType>,
    ) -> SourceSeeds<IndexType> {
        if a_in_lca {
            return UpToFour::one((source, GfaPathLength::from_usize(0)));
        }
        let block_tree = self.block;
        let tree = block_tree.tree();
        let child = tree
            .child_on_path_to(lca, x_a)
            .expect("x_a must be a strict descendant of the LCA here");
        let at_child = self.block.steps().lift(tree, x_a, child, at_x_a);

        let (source_pole, target_pole) = block_tree.reference_poles(child).unwrap();
        let mut seeds = UpToFour::new((source, GfaPathLength::from_usize(0)));
        for (pole, segment) in [(Pole::Source, source_pole), (Pole::Target, target_pole)] {
            for orientation in Orientation::all() {
                if let Some(distance) = at_child.get(pole, orientation).into_option() {
                    seeds.push((segment.into_directed(orientation.is_forward()), distance));
                }
            }
        }
        seeds
    }
}

/// The pole-projection table of one inner skeleton: `[4 * id + slot] = D(state id -> pole slot)`
/// over the skeleton, slot order `[s^+, s^-, t^+, t^-]` — exactly what a forward Dijkstra from the
/// state reads at the four pole states ([`pole_vector_from_workspace`]), computed as one backward
/// Dijkstra per pole state over the reversed skeleton. A pole state absent from the skeleton leaves
/// its column `∞`; sums at or above the index type's infinity are clamped one below it.
pub(super) fn projection_table<IndexType: GraphIndexInteger>(
    skeleton: &Skeleton<IndexType>,
    poles: [DirectedNodeIndex<IndexType>; 4],
) -> Vec<MinPlus<IndexType>> {
    use std::{cmp::Reverse, collections::BinaryHeap};
    let n = skeleton.node_count();
    let (offsets, targets, weights) = (
        skeleton.offsets(),
        skeleton.edge_target(),
        skeleton.edge_weight(),
    );
    // Reverse CSR: arc `u -> v (w)` becomes `v -> u (w)`.
    let mut roff = vec![0u32; n + 1];
    for &v in targets {
        roff[v as usize + 1] += 1;
    }
    for i in 0..n {
        roff[i + 1] += roff[i];
    }
    let mut rto = vec![0u32; targets.len()];
    let mut rw = vec![0u64; targets.len()];
    let mut cur = roff.clone();
    for u in 0..n {
        for e in offsets[u] as usize..offsets[u + 1] as usize {
            let v = targets[e] as usize;
            let slot = cur[v] as usize;
            rto[slot] = u as u32;
            rw[slot] = weights[e].into_usize() as u64;
            cur[v] += 1;
        }
    }
    let mut table = vec![MinPlus::infinity(); 4 * n];
    let mut dist = vec![INF; n];
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for (slot, &pole) in poles.iter().enumerate() {
        let Some(source) = skeleton.node_id(pole) else {
            continue;
        };
        dist.iter_mut().for_each(|d| *d = INF);
        heap.clear();
        dist[source as usize] = 0;
        heap.push(Reverse((0, source)));
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            for e in roff[u as usize] as usize..roff[u as usize + 1] as usize {
                let v = rto[e];
                let nd = d.saturating_add(rw[e]);
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    heap.push(Reverse((nd, v)));
                }
            }
        }
        for (id, &d) in dist.iter().enumerate() {
            table[4 * id + slot] = MinPlus::from_wide_clamped(d);
        }
    }
    table
}

/// Build a pole vector by reading the two poles (in both orientations) from a completed workspace
/// search over `skeleton`.
fn pole_vector_from_workspace<
    IndexType: GraphIndexInteger,
    S: SkeletonAccess<IndexType> + ?Sized,
>(
    skeleton: &S,
    workspace: &DijkstraWorkspace<IndexType>,
    poles: (
        bidirected_adjacency_array::index::NodeIndex<IndexType>,
        bidirected_adjacency_array::index::NodeIndex<IndexType>,
    ),
) -> PoleVector<IndexType> {
    let mut vector = PoleVector::infinity();
    for (pole, segment) in [(Pole::Source, poles.0), (Pole::Target, poles.1)] {
        for orientation in Orientation::all() {
            let node = segment.into_directed(orientation.is_forward());
            if let Some(id) = skeleton.node_id(node)
                && let Some(distance) = workspace.get(id)
            {
                vector.set(pole, orientation, MinPlus::finite(distance));
            }
        }
    }
    vector
}
