//! Test support: build ONE block's record into a scratch image (the way the engine build does) and
//! hand out a [`BlockQueryEngine`] over it, so the per-block tests exercise the exact image-backed
//! query path.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::decomposition::{SPQRDecomposition, indices::BlockIndex};

use super::{
    BlockQueryEngine, Tiers, precompute_labels, precompute_projections, precompute_skeletons,
};
use crate::{
    block_record::{BlockView, write_block},
    block_spqr_tree::BlockSpqrTree,
    graph_view::GraphSections,
    image::Image,
    inside_dp::InsideDp,
    outside_dp::OutsideDp,
    pole_lift::PoleLift,
    s_node_segtree::precompute_s_node_segtrees,
    skeleton::Skeleton,
    topology::TopologySections,
};
use bidirected_adjacency_array::index::GraphIndexInteger;

/// Which label oracles the test block gets.
#[derive(Clone, Copy)]
pub enum TestLabels {
    /// No labels (pure Dijkstra LCA path).
    None,
    /// [`precompute_labels`] with this threshold after a plain outside DP.
    Precompute(usize),
}

pub struct TestBlock<IndexType: GraphIndexInteger> {
    image: Image,
    graph: GraphSections<IndexType>,
    topology: TopologySections<IndexType>,
    /// The record-block nodes' locations (no lift lays rows out here, so they stay build-time).
    build_locs: Vec<crate::topology::BuildLoc>,
    block: BlockIndex<IndexType>,
    offset: u64,
}

impl<IndexType: GraphIndexInteger + Send + Sync + 'static> TestBlock<IndexType> {
    pub fn build<NodeData: GfaNodeData + Send + Sync, EdgeData: GfaEdgeData + Send + Sync>(
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        spqr: &SPQRDecomposition<'_, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>,
        block: BlockIndex<IndexType>,
        labels: TestLabels,
    ) -> Self {
        let segtree_min = Tiers::default().segtree_min;
        Self::build_with_segtree_min(graph, spqr, block, labels, segtree_min)
    }

    /// Like [`Self::build`], with an explicit S-node segment-tree threshold (directed states;
    /// `1` puts a tree on every S-node, `usize::MAX` on none).
    pub fn build_with_segtree_min<
        NodeData: GfaNodeData + Send + Sync,
        EdgeData: GfaEdgeData + Send + Sync,
    >(
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        spqr: &SPQRDecomposition<'_, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>,
        block: BlockIndex<IndexType>,
        labels: TestLabels,
        segtree_min: usize,
    ) -> Self {
        let image = Image::new_arena(1 << 30).unwrap();
        let (topology, build_locs, numbering) =
            TopologySections::write_with_all_trees(&image, spqr);
        let (mut graph_secs, _) = GraphSections::write(&image, graph, topology.lens());
        graph_secs.write_adjacency(&image, graph, |_| true);
        let block_tree = BlockSpqrTree::new(spqr, block);
        // The topology numbers blocks its own way (see `chain_block_numbering`).
        let block = numbering.new_id(block);
        let tiers = Tiers {
            label_min: match labels {
                TestLabels::None => usize::MAX,
                TestLabels::Precompute(t) => t,
            },
            segtree_min,
            ..Tiers::default()
        };
        let inside = InsideDp::compute(graph, block_tree);
        let n = inside.block_tree().node_count();
        let outside = OutsideDp::compute(&inside);
        let pole_lift = PoleLift::compute(&inside);
        let (mut inner, mut lca) = precompute_skeletons(&inside, &outside);
        let segtrees = precompute_s_node_segtrees(&inside, &outside, &inner, &tiers);
        for (local, seg) in segtrees.iter().enumerate() {
            if seg.is_some() {
                inner[local] = Skeleton::empty();
                lca[local] = Skeleton::empty();
            }
        }
        let mut labels = match labels {
            TestLabels::None => (0..n).map(|_| None).collect(),
            TestLabels::Precompute(_) => {
                precompute_labels(&inside, &outside, &lca, &tiers, Vec::new())
            }
        };
        labels.resize_with(n, || None);
        let projections =
            precompute_projections(inside.block_tree(), &inner, &labels, &segtrees, &tiers);
        for (local, seg) in segtrees.iter().enumerate() {
            if seg.is_some() {
                labels[local] = None; // a segtree node carries no labels
            }
        }
        let lca_outside = crate::block_record::lca_overlays(
            inside.block_tree(),
            |local| *outside.outside_matrix(local),
            |local| crate::skeleton::SkeletonAccess::node_count(&lca[local]) > 0,
        );
        let offset = write_block(
            &image,
            inside.block_tree(),
            pole_lift,
            inner,
            lca_outside,
            labels,
            segtrees,
            projections,
            0,
        );
        Self {
            image,
            graph: graph_secs,
            topology,
            build_locs,
            block,
            offset,
        }
    }

    pub fn view(&self) -> BlockView<'_, IndexType> {
        BlockView::read(&self.image, self.block, self.offset as usize)
    }

    pub fn engine<'a>(
        &'a self,
        view: &'a BlockView<'a, IndexType>,
    ) -> BlockQueryEngine<'a, IndexType> {
        BlockQueryEngine::new(
            self.topology.view_build(&self.image, &self.build_locs),
            view,
            self.graph.view(&self.image),
        )
    }
}
