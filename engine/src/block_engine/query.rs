//! The start-to-start query of one block: locate both endpoints, take their LCA, seed it and read
//! the target (see the module docs of [`super`]).

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use super::{BlockQueryEngine, LcaOracle, target::TargetSite};
use crate::rooted_tree::TreeAccess;
use crate::{
    graph_view::QueryGraph, location::GfaLocation, min_plus::relax, path::GfaPathLength,
    skeleton::DijkstraWorkspace,
};

impl<'a, IndexType: GraphIndexInteger + 'static> BlockQueryEngine<'a, IndexType> {
    /// The shortest-path distance from `source` to `target`, or `None` if unreachable within the
    /// block. Both locations must lie in the block this engine was built for.
    pub fn distance(
        &self,
        source: GfaLocation<IndexType>,
        target: GfaLocation<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let mut workspace = DijkstraWorkspace::new();
        let source_offset = source.offset().into_length();
        let target_offset = target.offset().into_length();
        let mut best: Option<GfaPathLength<IndexType>> = None;

        // Trivial within-node walk (only valid forward within a single directed node).
        if source.node() == target.node() && target.offset() >= source.offset() {
            relax(&mut best, target_offset - source_offset);
        }

        if source.node() != target.node() {
            if let Some(distance) =
                self.start_to_start_with(source.node(), target.node(), &mut workspace)
            {
                relax(&mut best, distance - source_offset + target_offset);
            }
        } else {
            // Same directed node: a non-trivial route must leave and cycle back.
            let source_length = self.graph.node_len(source.node().into_bidirected());
            for to in self.graph.out_neighbors(source.node()) {
                if let Some(distance) = self.start_to_start_with(to, source.node(), &mut workspace)
                {
                    relax(
                        &mut best,
                        source_length + distance - source_offset + target_offset,
                    );
                }
            }
        }

        best
    }

    /// Start-to-start distance from directed node `a` to directed node `b` (offsets `0 -> 0`),
    /// within this block. Used both by [`Self::distance`] and by the cross-block engine.
    pub fn start_to_start(
        &self,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let mut workspace = DijkstraWorkspace::new();
        self.start_to_start_with(a, b, &mut workspace)
    }

    /// Like [`Self::start_to_start`], reusing `workspace` for its searches (its previous contents
    /// are overwritten), so the call allocates nothing.
    pub fn start_to_start_with(
        &self,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let at = self.locate(a, b)?;
        // Seed the search with the source's distances to LCA-skeleton nodes (may itself run a
        // projection search on `workspace`). Independent of the LCA skeleton, so it runs before the
        // LCA's oracle is read.
        let seeds = self.source_seeds(a, at.lca, at.x_a, at.a_in_lca, workspace);
        let site = TargetSite {
            target: b,
            lca: at.lca,
            x_b: at.x_b,
            b_in_lca: at.b_in_lca,
        };
        match LcaOracle::of(self.block, at.lca) {
            // The LCA read is a few `O(log m)` range products, no search.
            LcaOracle::SegTree(seg) => self.target_distance_segtree(site, &seg, &seeds, workspace),
            // Label lookups instead of a skeleton search; `workspace` stays free for the target's
            // reverse projection.
            LcaOracle::Labels(labels) => {
                self.target_distance_labeled(site, &labels, &seeds, workspace)
            }
            // The stored skeleton (inner + `D_out` overlay) and the multi-source Dijkstra.
            LcaOracle::Skeleton(lca_skeleton) => {
                self.finish_lca_search(site, &seeds, &lca_skeleton, workspace)
            }
        }
    }

    /// The deepest SPQR nodes containing `a` and `b`, their LCA and whether each endpoint lies in
    /// the LCA's skeleton; `None` if either is not in this block.
    fn locate(
        &self,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
    ) -> Option<Located> {
        let tree = self.block.tree();
        let (Some(a_loc), Some(b_loc)) = (
            self.spqr_location(a.into_bidirected()),
            self.spqr_location(b.into_bidirected()),
        ) else {
            return None; // not in this block
        };
        let (x_a, x_b) = (a_loc.x, b_loc.x);
        let lca = tree.lca(x_a, x_b);
        Some(Located {
            lca,
            x_a,
            x_b,
            a_in_lca: a_loc.contains_ancestor(tree, lca),
            b_in_lca: b_loc.contains_ancestor(tree, lca),
        })
    }
}

/// Where the endpoints of a start-to-start query sit in the block's SPQR tree.
struct Located {
    lca: usize,
    /// The deepest SPQR nodes containing the source and the target.
    x_a: usize,
    x_b: usize,
    /// Whether the source / target is a vertex of the LCA's skeleton.
    a_in_lca: bool,
    b_in_lca: bool,
}
