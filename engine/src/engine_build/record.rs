//! One record block: its DP pipeline and its record in the image.
//!
//! The pipeline runs over the block's SPQR tree: the inside DP (bottom-up `4x4` pole matrices),
//! the outside DP (`D_out`, top-down, with hub labels for the big skeletons it searches), the
//! pole-lift transitions, the query skeletons (inner and LCA role), the S-node segment trees and
//! the labels; each node gets the oracles of its tier ([`Tiers`]). The record keeps only what
//! queries read: the block tree, the pole lift, the inner skeletons (without arcs where a label and
//! a projection table answer instead), `D_out` of the nodes whose LCA search runs over a skeleton,
//! the labels, the segment trees and the pole projection tables (see [`crate::block_engine`]).

use bidirected_adjacency_array::{
    index::GraphIndexInteger,
    io::gfa1::{GfaEdgeData, GfaNodeData},
};

use super::{Graph, blocks::FlipClosures};
use crate::{
    block_engine::{Tiers, precompute_labels, precompute_projections, precompute_skeletons},
    block_record::{lca_overlays, write_block},
    block_spqr_tree::BlockSpqrTree,
    image::Image,
    inside_dp::InsideDp,
    min_plus::MinPlus,
    outside_dp::OutsideDp,
    pole_lift::PoleLift,
    s_node_segtree::{SNodeSegTree, precompute_s_node_segtrees},
    skeleton::{Skeleton, SkeletonAccess},
    skeleton_index::SkeletonLabels,
};

/// Per SPQR node of a block: a hub-label oracle, or `None`.
type Labels<IndexType> = Vec<Option<Box<SkeletonLabels<IndexType>>>>;

/// One record block's DP structures: everything its record is written from.
pub(super) struct BlockDp<
    'g,
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
> {
    inside: InsideDp<'g, IndexType, NodeData, EdgeData>,
    outside: OutsideDp<IndexType>,
    pole_lift: PoleLift<IndexType>,
    /// Per SPQR node: its inner-role (pole projection) and LCA-role query skeletons.
    inner: Vec<Skeleton<IndexType>>,
    lca: Vec<Skeleton<IndexType>>,
    segtrees: Vec<Option<Box<SNodeSegTree<IndexType>>>>,
    labels: Labels<IndexType>,
    tiers: Tiers,
}

impl<'g, IndexType, NodeData, EdgeData> BlockDp<'g, IndexType, NodeData, EdgeData>
where
    IndexType: GraphIndexInteger + Send + Sync,
    NodeData: GfaNodeData + Send + Sync,
    EdgeData: GfaEdgeData + Send + Sync,
{
    /// Run the DP pipeline over a block tree, baking `closures` (the flip closures of the block's
    /// child cuts) into its skeletons; each node gets the oracles of its tier (see [`Tiers`]).
    pub(super) fn compute(
        graph: &'g Graph<IndexType, NodeData, EdgeData>,
        tree: BlockSpqrTree<IndexType>,
        closures: &FlipClosures<IndexType>,
        tiers: &Tiers,
    ) -> Self {
        let inside = if closures.is_empty() {
            InsideDp::compute(graph, tree)
        } else {
            InsideDp::compute_with_flips(graph, tree, closures)
        };
        // A big parent's skeleton gets its hub-label oracle here; it fills the children's outside
        // by lookup and later answers the parent's LCA-role searches.
        let (outside, mut outside_labels) = OutsideDp::compute_with_labels(&inside, tiers);
        let pole_lift = PoleLift::compute(&inside);
        let (mut inner, mut lca) = precompute_skeletons(&inside, &outside);
        // A large S-node keeps neither skeletons nor labels: its segment tree answers both its
        // pole projections (inner role) and its LCA searches (with `D_out`).
        let segtrees = precompute_s_node_segtrees(&inside, &outside, &inner, tiers);
        for (local, segtree) in segtrees.iter().enumerate() {
            if segtree.is_some() {
                inner[local] = Skeleton::empty();
                lca[local] = Skeleton::empty();
                if let Some(slot) = outside_labels.get_mut(local) {
                    *slot = None;
                }
            }
        }
        // The outside DP's oracles become LCA-role oracles; any other node at the threshold (a
        // childless leaf) gets its labels here.
        let labels = precompute_labels(&inside, &outside, &lca, tiers, outside_labels);
        Self {
            inside,
            outside,
            pole_lift,
            inner,
            lca,
            segtrees,
            labels,
            tiers: *tiers,
        }
    }

    /// The number of labeled SPQR nodes (the label ordinals the record takes).
    pub(super) fn label_count(&self) -> u32 {
        self.labels.iter().filter(|l| l.is_some()).count() as u32
    }

    /// Append the block's record to the image, its labels numbered from `first_label_ordinal`,
    /// releasing the build-only structures on the way. Returns the record's offset.
    pub(super) fn write(self, image: &Image, first_label_ordinal: u32) -> u64 {
        let Self {
            mut inside,
            outside,
            pole_lift,
            mut inner,
            mut lca,
            segtrees,
            labels,
            tiers,
        } = self;
        // Queries read the block tree but no inside matrix; a labeled node answers its LCA search
        // through its labels, so its LCA-role skeleton is dead.
        inside.clear_matrices();
        for (skeleton, label) in lca.iter_mut().zip(&labels) {
            if label.is_some() {
                *skeleton = Skeleton::empty();
            }
        }
        let tree = inside.block_tree();
        // The record keeps no LCA-role skeleton, only `D_out` of the nodes whose LCA search still
        // runs over one.
        let lca_outside = lca_overlays(
            tree,
            |local| *outside.outside_matrix(local),
            |local| lca[local].node_count() > 0,
        );
        drop(lca);
        let projections = pole_projections(tree, &mut inner, &labels, &segtrees, &tiers);
        let offset = write_block(
            image,
            tree,
            pole_lift,
            inner,
            lca_outside,
            labels,
            segtrees,
            projections,
            first_label_ordinal,
        );
        drop(inside);
        drop(outside);
        offset
    }
}

/// Pole-projection tables for the big inner skeletons: one row read per projection instead of a
/// skeleton Dijkstra (a segment-tree node answers its own). A labeled node then answers its LCA
/// search through its labels and its projection through its table (or has no poles: the block
/// root), so its inner arcs are never read: only its state list stays, for the table lookup.
fn pole_projections<IndexType: GraphIndexInteger + Send + Sync>(
    tree: &BlockSpqrTree<IndexType>,
    inner: &mut [Skeleton<IndexType>],
    labels: &Labels<IndexType>,
    segtrees: &[Option<Box<SNodeSegTree<IndexType>>>],
    tiers: &Tiers,
) -> Vec<Option<Vec<MinPlus<IndexType>>>> {
    let projections = precompute_projections(tree, inner, labels, segtrees, tiers);
    for (local, skeleton) in inner.iter_mut().enumerate() {
        let keeps_arcs = Tiers::keeps_inner_arcs(
            labels[local].is_some(),
            projections[local].is_some(),
            tree.reference_poles(local).is_some(),
        );
        if !keeps_arcs {
            skeleton.drop_arcs();
        }
    }
    projections
}
