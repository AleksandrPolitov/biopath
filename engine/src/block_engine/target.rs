//! The target read: the LCA distances to the poles of the LCA's child towards the target, combined
//! with the target's lifted reverse projection.

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use super::BlockQueryEngine;
use crate::rooted_tree::TreeAccess;
use crate::{
    graph_view::QueryGraph,
    min_plus::relax,
    orientation::Orientation,
    path::GfaPathLength,
    pole_lift::{Pole, PoleVector},
    skeleton::{DijkstraWorkspace, SkeletonAccess},
};

/// Where the target of an LCA read sits in the block's SPQR tree.
#[derive(Clone, Copy)]
pub(super) struct TargetSite<IndexType: GraphIndexInteger> {
    /// The target, a directed node of the block.
    pub(super) target: DirectedNodeIndex<IndexType>,
    /// The LCA of the deepest SPQR nodes containing the source and the target.
    pub(super) lca: usize,
    /// The deepest SPQR node containing the target.
    pub(super) x_b: usize,
    /// Whether the target is a vertex of the LCA's skeleton.
    pub(super) b_in_lca: bool,
}

impl<'a, IndexType: GraphIndexInteger + 'static> BlockQueryEngine<'a, IndexType> {
    /// Read the target's distance from the completed LCA search held in `workspace`.
    ///
    /// When the target projects up from a descendant, the LCA-search distances to the relevant
    /// child's poles are extracted *first* (before the projection search reuses `workspace`).
    pub(super) fn target_distance<S: SkeletonAccess<IndexType> + ?Sized>(
        &self,
        site: TargetSite<IndexType>,
        lca_skeleton: &S,
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let TargetSite {
            target,
            lca,
            x_b,
            b_in_lca,
        } = site;
        if b_in_lca {
            return lca_skeleton
                .node_id(target)
                .and_then(|id| workspace.get(id));
        }

        let block_tree = self.block;
        let tree = block_tree.tree();
        let child = tree
            .child_on_path_to(lca, x_b)
            .expect("x_b must be a strict descendant of the LCA here");

        // Extract the LCA-search distances to the child's poles NOW, before the reverse projection
        // below overwrites `workspace`.
        let from_source = self.extract_from_source(lca_skeleton, child, workspace);

        // `reverse[pole, o'] = D(target' -> child_pole^{o'})` (distances from the reverse-complement
        // of the target to the child's poles); see [`Self::combine_target`].
        let reverse = self.pole_projection(target.invert(), x_b, workspace);
        let at_child = self.block.steps().lift(tree, x_b, child, reverse);
        self.combine_target(target, child, from_source, at_child)
    }

    /// The target read after an oracle-answered LCA read: `lca_to(node)` is the distance from the
    /// seed set to `node` over the LCA-role skeleton (a label merge or a segment-tree read); the rest
    /// (the child towards the target, the reverse projection, the lift and the combine) is the
    /// skeleton path's. The reverse projection uses `workspace`, which no LCA search holds here.
    pub(super) fn target_distance_via(
        &self,
        site: TargetSite<IndexType>,
        lca_to: impl Fn(DirectedNodeIndex<IndexType>) -> Option<GfaPathLength<IndexType>>,
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let TargetSite {
            target,
            lca,
            x_b,
            b_in_lca,
        } = site;
        if b_in_lca {
            return lca_to(target);
        }

        let block_tree = self.block;
        let tree = block_tree.tree();
        let child = tree
            .child_on_path_to(lca, x_b)
            .expect("x_b must be a strict descendant of the LCA here");

        // D(seed -> child_pole^o) for each of the child's poles/orientations, via `lca_to`.
        let (source_pole, target_pole) = block_tree.reference_poles(child).unwrap();
        let poles = [(Pole::Source, source_pole), (Pole::Target, target_pole)];
        let mut from_source: [[Option<GfaPathLength<IndexType>>; 2]; 2] = [[None; 2]; 2];
        for (pole_index, &(_, segment)) in poles.iter().enumerate() {
            for orientation in Orientation::all() {
                from_source[pole_index][orientation.index()] =
                    lca_to(segment.into_directed(orientation.is_forward()));
            }
        }

        let reverse = self.pole_projection(target.invert(), x_b, workspace);
        let at_child = self.block.steps().lift(tree, x_b, child, reverse);
        self.combine_target(target, child, from_source, at_child)
    }

    /// The LCA-search distances (held in `workspace`) to `child`'s two reference poles, indexed
    /// `[pole][orientation]`.
    fn extract_from_source<S: SkeletonAccess<IndexType> + ?Sized>(
        &self,
        lca_skeleton: &S,
        child: usize,
        workspace: &DijkstraWorkspace<IndexType>,
    ) -> [[Option<GfaPathLength<IndexType>>; 2]; 2] {
        let (source_pole, target_pole) = self.block.reference_poles(child).unwrap();
        let poles = [(Pole::Source, source_pole), (Pole::Target, target_pole)];
        let mut from_source: [[Option<GfaPathLength<IndexType>>; 2]; 2] = [[None; 2]; 2];
        for (pole_index, &(_, segment)) in poles.iter().enumerate() {
            for orientation in Orientation::all() {
                let node = segment.into_directed(orientation.is_forward());
                from_source[pole_index][orientation.index()] =
                    lca_skeleton.node_id(node).and_then(|id| workspace.get(id));
            }
        }
        from_source
    }

    /// Combine the LCA-search distances to `child`'s poles (`from_source`) with the target's lifted
    /// reverse distances from those poles (`at_child`). Start-to-start is not reverse-complement
    /// symmetric, so `D(child_pole^o -> target) = at_child[pole, ¬o] + len(child_pole) - len(target)`.
    fn combine_target(
        &self,
        target: DirectedNodeIndex<IndexType>,
        child: usize,
        from_source: [[Option<GfaPathLength<IndexType>>; 2]; 2],
        at_child: PoleVector<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let (source_pole, target_pole) = self.block.reference_poles(child).unwrap();
        let poles = [(Pole::Source, source_pole), (Pole::Target, target_pole)];
        let target_length = self.graph.node_len(target.into_bidirected());
        let mut best: Option<GfaPathLength<IndexType>> = None;
        for (pole_index, &(pole, segment)) in poles.iter().enumerate() {
            let segment_length = self.graph.node_len(segment);
            for orientation in Orientation::all() {
                let from_source = from_source[pole_index][orientation.index()];
                let into_target = at_child
                    .get(pole, orientation.invert())
                    .into_option()
                    .map(|d| d + segment_length - target_length);
                if let (Some(from_source), Some(into_target)) = (from_source, into_target) {
                    relax(&mut best, from_source + into_target);
                }
            }
        }
        best
    }
}
