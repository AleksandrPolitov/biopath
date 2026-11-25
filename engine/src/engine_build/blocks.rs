//! The per-block stage: every block, children first over the block-cut forest.
//!
//! Flip baking: a record block's engine carries the flip closures `down[c]` of its child cuts as
//! `c^a -> c^b` edges, so it answers flips through any child cut natively and only its parent
//! cut's flip stays explicit. Before a block, `down[c]` of its child cuts is closed from the
//! finished blocks below ([`FlipBaker::child_closures`]); after it, the block's raw cut-to-cut
//! matrix `M_B` is read through its own engine (a small block: one block-restricted Dijkstra per
//! directed cut) and folded toward its parent cut. The lift starts from the resulting forest: the
//! raw matrices and the bottom-up closures.

use std::sync::Mutex;

use bidirected_adjacency_array::{
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::decomposition::indices::BlockIndex;

use super::{BuildViews, PreparedBlock, record::BlockDp};
use crate::{
    block_cut_lift::{BakedForest, FlipBaker},
    block_dijkstra::BlockDijkstra,
    block_engine::{BlockQueryEngine, Tiers},
    block_record::BlockView,
    block_spqr_tree::BlockSpqrTree,
    fast_hash::FastMap,
    min_plus::{Mat2, MinPlus},
    orientation::Orientation,
};

/// The flip closures of a block's child cuts, keyed by cut node: what its engine bakes in.
pub(super) type FlipClosures<IndexType> = FastMap<NodeIndex<IndexType>, Mat2<IndexType>>;

/// The per-block stage's state.
pub(super) struct BlockRecords<'v, 'b, IndexType: GraphIndexInteger, NodeData, EdgeData> {
    views: &'v BuildViews<'b, IndexType, NodeData, EdgeData>,
    small_search: SmallBlockSearch<IndexType>,
    tiers: Tiers,
    /// Per block, its record's offset in the image; `u64::MAX` marks a small block (no record).
    offsets: Vec<u64>,
    /// The label ordinal of the next record's first label.
    label_ordinal: u32,
}

impl<'v, 'b, IndexType, NodeData, EdgeData> BlockRecords<'v, 'b, IndexType, NodeData, EdgeData>
where
    IndexType: GraphIndexInteger + Send + Sync + 'static,
    NodeData: GfaNodeData + Send + Sync,
    EdgeData: GfaEdgeData + Send + Sync,
{
    pub(super) fn new(
        views: &'v BuildViews<'b, IndexType, NodeData, EdgeData>,
        tiers: Tiers,
    ) -> Self {
        Self {
            views,
            small_search: SmallBlockSearch::new(views.graph.node_count()),
            tiers,
            offsets: vec![u64::MAX; views.topology.block_count()],
            label_ordinal: 0,
        }
    }

    /// Build every block, children first. Returns the record offsets (`u64::MAX`: small block)
    /// and the baked forest for the lift (its raw cut-to-cut matrices and bottom-up closures).
    pub(super) fn build(
        mut self,
        blocks: Vec<PreparedBlock<IndexType>>,
    ) -> (Vec<u64>, BakedForest<IndexType>) {
        let mut baker = FlipBaker::new(&self.views.topology);
        let mut blocks: Vec<Option<PreparedBlock<IndexType>>> =
            blocks.into_iter().map(Some).collect();
        for bid in baker.forest.blocks_postorder() {
            let block = blocks[bid].take().expect("each block is built once");
            debug_assert_eq!(
                block.block_index.into_usize(),
                bid,
                "prepared blocks are in block order"
            );
            // All blocks below are finished: close `down[c]` of the child cuts.
            let closures = baker.child_closures(bid);
            match block.tree {
                None => {
                    let cuts = &baker.forest.block_cuts[bid];
                    let raw = self
                        .small_search
                        .raw_cut_matrix(self.views, block.block_index, cuts);
                    baker.finish_block_raw(bid, false, raw);
                }
                Some(tree) => {
                    let offset = self.write_record(block.block_index, tree, &closures);
                    self.fold(&mut baker, bid, block.block_index, offset);
                }
            }
        }
        (self.offsets, baker.finish())
    }

    /// A record block: its DP pipeline (baking `closures` into its skeletons), then its record in
    /// the image. Returns the record's offset.
    fn write_record(
        &mut self,
        block_index: BlockIndex<IndexType>,
        tree: BlockSpqrTree<IndexType>,
        closures: &FlipClosures<IndexType>,
    ) -> u64 {
        let dp = BlockDp::compute(self.views.graph, tree, closures, &self.tiers);
        let label_count = dp.label_count();
        let offset = dp.write(self.views.image, self.label_ordinal);
        self.label_ordinal += label_count;
        self.offsets[block_index.into_usize()] = offset;
        offset
    }

    /// Flip baking after a record block: its raw `M_B`, read through its own (baked) engine, is
    /// folded toward its parent cut.
    fn fold(
        &self,
        baker: &mut FlipBaker<IndexType>,
        bid: usize,
        block_index: BlockIndex<IndexType>,
        offset: u64,
    ) {
        let views = self.views;
        let view = BlockView::read(views.image, block_index, offset as usize);
        let engine = || BlockQueryEngine::new(views.topology, &view, views.graph_view);
        let intra =
            |from: DirectedNodeIndex<IndexType>,
             to: DirectedNodeIndex<IndexType>,
             _: BlockIndex<IndexType>| engine().start_to_start(from, to);
        let cut_matrix = |_: BlockIndex<IndexType>, cuts: &[NodeIndex<IndexType>]| {
            engine().fill_cut_matrix(cuts)
        };
        baker.finish_block(bid, true, &intra, &cut_matrix);
    }
}

/// A small block's raw cut matrix (it has no record): a block-restricted Dijkstra over the graph.
struct SmallBlockSearch<IndexType: GraphIndexInteger> {
    dijkstra: Mutex<BlockDijkstra<IndexType>>,
}

impl<IndexType: GraphIndexInteger> SmallBlockSearch<IndexType> {
    /// A search over a graph of `node_count` nodes.
    fn new(node_count: usize) -> Self {
        Self {
            dijkstra: Mutex::new(BlockDijkstra::new(node_count * 2)),
        }
    }

    /// The block's raw `M_B` over `cuts` (`(2c)^2` row-major over the directed cuts, zero
    /// diagonal): one search per directed cut, every target read off it.
    fn raw_cut_matrix<NodeData: GfaNodeData, EdgeData>(
        &self,
        views: &BuildViews<'_, IndexType, NodeData, EdgeData>,
        block: BlockIndex<IndexType>,
        cuts: &[NodeIndex<IndexType>],
    ) -> Vec<MinPlus<IndexType>> {
        let dim = 2 * cuts.len();
        let directed =
            |i: usize| cuts[i / 2].into_directed(Orientation::from_index(i % 2).is_forward());
        let mut matrix = vec![MinPlus::infinity(); dim * dim];
        let mut dijkstra = self.dijkstra.lock().unwrap();
        for row in 0..dim {
            let from = directed(row);
            Self::run(&mut dijkstra, views, from, block);
            for col in 0..dim {
                let to = directed(col);
                matrix[row * dim + col] = if from == to {
                    MinPlus::zero()
                } else {
                    dijkstra
                        .get(to)
                        .map(MinPlus::finite)
                        .unwrap_or_else(MinPlus::infinity)
                };
            }
        }
        matrix
    }

    fn run<NodeData: GfaNodeData, EdgeData>(
        dijkstra: &mut BlockDijkstra<IndexType>,
        views: &BuildViews<'_, IndexType, NodeData, EdgeData>,
        from: DirectedNodeIndex<IndexType>,
        block: BlockIndex<IndexType>,
    ) {
        dijkstra.run(views.graph, from, |node| {
            views.topology.node_block_indices(node).any(|b| b == block)
        });
    }
}
