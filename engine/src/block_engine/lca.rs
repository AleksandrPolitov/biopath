//! The LCA read: the multi-source skeleton Dijkstra, or the label / segment-tree reads that replace
//! it.

use std::cell::RefCell;

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use super::{
    BlockQueryEngine, memo::memoized_distance_id, target::TargetSite, up_to_four::UpToFour,
};
use crate::{
    min_plus::{
        network::Mat4,
        relax,
        word::{inf, length},
    },
    path::GfaPathLength,
    s_node_segtree::{
        Role as SegRole, SegTreeAccess, SegTreeView, orientation_bit as seg_orientation,
    },
    skeleton::{DijkstraWorkspace, SkeletonAccess},
    skeleton_index::LabelsAccess,
};

impl<'a, IndexType: GraphIndexInteger + 'static> BlockQueryEngine<'a, IndexType> {
    /// The multi-source Dijkstra over an assembled LCA skeleton plus the target read: the tail of
    /// [`Self::start_to_start_with`] for an LCA without a segment tree or labels.
    pub(super) fn finish_lca_search<S: SkeletonAccess<IndexType> + ?Sized>(
        &self,
        site: TargetSite<IndexType>,
        seeds: &[(DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>)],
        lca_skeleton: &S,
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let dense_seeds = UpToFour::collect(
            (0, GfaPathLength::from_usize(0)),
            seeds.iter().filter_map(|&(node, distance)| {
                lca_skeleton.node_id(node).map(|id| (id, distance))
            }),
        );

        // The multi-source search over the LCA skeleton.
        workspace.run(lca_skeleton, &dense_seeds);

        // Read the target's distance out of the search.
        self.target_distance(site, lca_skeleton, workspace)
    }

    /// PLL-labeled target read: `lca_to` merges hub labels. Seeds are resolved to label ids once.
    pub(super) fn target_distance_labeled(
        &self,
        site: TargetSite<IndexType>,
        labels: &impl LabelsAccess<IndexType>,
        seeds: &[(DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>)],
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let seed_ids = UpToFour::collect(
            (0, GfaPathLength::from_usize(0)),
            seeds
                .iter()
                .filter_map(|&(node, distance)| labels.node_id(node).map(|id| (id, distance))),
        );
        let lca_to = |node: DirectedNodeIndex<IndexType>| -> Option<GfaPathLength<IndexType>> {
            let tid = labels.node_id(node)?;
            let mut best = None;
            for &(sid, sd) in seed_ids.iter() {
                if let Some(d) = memoized_distance_id(labels, sid, tid) {
                    relax(&mut best, sd + d);
                }
            }
            best
        };
        self.target_distance_via(site, lca_to, workspace)
    }

    /// Segment-tree target read: `lca_to` is `min over seeds of seed_dist + d(seed -> node)` over the
    /// node's LCA-role tree. The seeds sit at the two poles of one child (two cycle vertices) and the
    /// targets at one node or at another child's two poles, so the cut-pair matrices are memoized per
    /// call: at most four `O(log m)` range products answer the whole read.
    pub(super) fn target_distance_segtree(
        &self,
        site: TargetSite<IndexType>,
        seg: &SegTreeView<'_, IndexType>,
        seeds: &[(DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>)],
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let seed_pos = UpToFour::collect(
            (0, 0, GfaPathLength::from_usize(0)),
            seeds.iter().filter_map(|&(node, d)| {
                seg.pos_of(node.into_bidirected())
                    .map(|p| (p, seg_orientation(node), d))
            }),
        );
        let memo = RefCell::new(UpToFour::new((0, 0, [inf(); 16])));
        let pair = |p: usize, q: usize| -> Mat4<IndexType> {
            if let Some(&(_, _, mat)) = memo.borrow().iter().find(|&&(a, b, _)| a == p && b == q) {
                return mat;
            }
            let mat = seg.cut_pair(p, q, SegRole::Lca);
            // A fifth pair (never seen: two seed positions, at most two target ones) goes unmemoized.
            memo.borrow_mut().try_push((p, q, mat));
            mat
        };
        let lca_to = |node: DirectedNodeIndex<IndexType>| -> Option<GfaPathLength<IndexType>> {
            let pt = seg.pos_of(node.into_bidirected())?;
            let ot = seg_orientation(node);
            let mut best = None;
            for &(ps, os, ds) in seed_pos.iter() {
                let raw = if ps == pt {
                    if os == ot {
                        IndexType::zero()
                    } else {
                        seg.flip(ps, os, ot, SegRole::Lca)
                    }
                } else {
                    pair(ps, pt)[4 * os + 2 + ot]
                };
                if let Some(d) = length(raw) {
                    relax(&mut best, ds + d);
                }
            }
            best
        };
        self.target_distance_via(site, lca_to, workspace)
    }
}
