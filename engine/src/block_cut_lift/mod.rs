//! The block-cut lift: the cross-block part of a query, as a min-plus product along the block-cut
//! forest instead of a search.
//!
//! # The forest and the flips
//!
//! Blocks and cut nodes alternate in a forest (`forest.rs`): a walk between two blocks passes the
//! cut nodes on the forest path between them, in order. Only each cut's orientation is free, so a
//! walk is a chain of `2x2` min-plus matrices applied to a `2`-vector of distances to the current
//! cut's two orientations. A walk may also flip a cut's orientation (`c^+ ⇝ c^-`) by an excursion
//! into any part of the graph hanging off it, on the path or off it. The flip closure `full[c]` is
//! the cheapest such flip through the whole graph (`flips.rs`): `down[c]` through the pendant
//! subtree below `c`, computed bottom-up while the blocks are built (each record block bakes its
//! child cuts' `down` into its engine as flip edges), and `up[c]` through the rest, top-down. A cut
//! is flip-capable in a block when it can flip outside that block.
//!
//! # Cut matrices
//!
//! A block's raw matrix `M_B` holds the in-block distances between its directed cuts, read through
//! its engine (`cut_matrix.rs`). Its augmented matrix `M_aug` closes `M_B` with every cut's
//! `full[c]` as a self-loop (`gadget.rs`, `aug.rs`), so it covers every walk between two of its
//! cuts that leaves the block any number of times. The whole-graph closures make this sound (every
//! entry is a real walk) and complete (a shortest walk is always representable): the lift is exact.
//!
//! # The path product
//!
//! On the cut forest (a cut's parent is the parent cut of its parent block) each step is a `2x2` of
//! a block's `M_aug`. The product along a path comes from one of two backends (`path_product.rs`):
//! the closed-form prefix lift ([`crate::prefix_lift`]) when at most 2% of the steps are
//! exceptions to its closed form ([`crate::prefix_lift::MAX_EXCEPTION_FRACTION`]), the chain
//! segment trees ([`crate::chain_segtree`]) otherwise ([`LiftOptions::backend`] can force
//! either).
//!
//! # Node rows
//!
//! Each non-cut node of a block with few cuts ([`LiftOptions::node_row_max_cuts`]) stores its
//! distances to the block's cuts with `M_aug` folded in ([`crate::node_rows`]; `row_plan.rs` picks the kind, `row_table.rs` fills and lays
//! out the table, `row_cells.rs` reads it): **full** rows hold both orientations of every cut;
//! **half** rows hold each cut's away orientation only, for a block that touches each cut on one
//! side (the other cell adds the cut's flip closure); **flip** rows hold the raw distances to the
//! few flip-capable cuts of a block with many cuts; a **closed-form** node, whose arcs all lead to
//! cuts, keeps only its first-hop mask. A block whose nodes are all closed forms keeps no arcs.
//!
//! # The query
//!
//! [`BlockCutLift::cross`] (`query.rs`) anchors both endpoints, a cut node at its cut and any other
//! node at its block. In one block, the route leaves and re-enters through flip-capable cuts, and
//! `M_aug` closes the excursions between them. Otherwise the source is seeded at its on-path cut
//! (`D(a -> c_1^o)`, from its row or through the block's engine), lifted up to the forest LCA,
//! turned through the LCA block's `M_aug` (or the LCA cut's flip closure), lifted down to the
//! target's on-path cut and read into the target (`D(c_m^o -> b)`). Only an endpoint block without
//! full or half rows is ever searched, through its engine.
//!
//! The image layout is in `sections.rs`. Every query of the tests in `distance_index/tests.rs` and of the lift fuzz tests is checked against
//! whole-graph Dijkstra.

mod aug;
mod cut_lists;
mod cut_matrix;
mod flips;
mod forest;
mod gadget;
mod path_product;
mod query;
mod row_cells;
mod row_plan;
mod row_table;
mod sections;

pub use crate::prefix_lift::LiftBackend;
pub use flips::{BakedForest, FlipBaker};
pub use forest::MAX_BLOCK_CUTS;
pub use sections::LiftSections;

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::decomposition::indices::BlockIndex;

use crate::{
    chain_nav::{ChainNav, CutRec, SpecialBlockRec},
    image::Image,
    min_plus::{Mat2, MinPlus},
    node_rows,
    path::GfaPathLength,
    prefix_lift,
    topology::NodeTopology,
};
use aug::block_augs;
use flips::{baked_arcs, block_flip_cuts, can_flip, full_closures};
use forest::BlockCutForest;
use path_product::{BackendView, build_backend};
use row_plan::RowPlan;
use row_table::RowBlocks;
use sections::{AugDenseDir, AugSorted, AugSortedDir, LiftOwned};

/// The build options of the lift
/// ([`BuildOptions::lift`](crate::distance_index::BuildOptions::lift)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiftOptions {
    /// The path-product backend.
    pub backend: LiftBackend,
    /// The nodes of a block with at most this many cuts get cut rows (`0`: no rows, every
    /// cross-block query reads its endpoint blocks through their records).
    pub node_row_max_cuts: usize,
}

impl Default for LiftOptions {
    /// [`LiftBackend::Auto`] and rows for blocks with up to [`node_rows::DEFAULT_MAX_CUTS`] cuts.
    fn default() -> Self {
        Self {
            backend: LiftBackend::Auto,
            node_row_max_cuts: node_rows::DEFAULT_MAX_CUTS,
        }
    }
}

/// The exact in-block start-to-start distance `d_B(from -> to)` of two directed nodes of `block`
/// (`from == to` is `0`), as the lift reads it from the block's engine: `intra(from, to, block)`.
pub trait IntraDistance<IndexType: GraphIndexInteger>:
    Fn(
    DirectedNodeIndex<IndexType>,
    DirectedNodeIndex<IndexType>,
    BlockIndex<IndexType>,
) -> Option<GfaPathLength<IndexType>>
{
}

impl<IndexType: GraphIndexInteger, F> IntraDistance<IndexType> for F where
    F: Fn(
        DirectedNodeIndex<IndexType>,
        DirectedNodeIndex<IndexType>,
        BlockIndex<IndexType>,
    ) -> Option<GfaPathLength<IndexType>>
{
}

/// A block's whole raw cut-to-cut matrix `M_B` over the given cuts in one fill (`(2c)^2` row-major
/// over the directed cuts, `2 * i + o`): `cut_matrix(block, cuts)`.
pub trait CutMatrixFill<IndexType: GraphIndexInteger>:
    Fn(BlockIndex<IndexType>, &[NodeIndex<IndexType>]) -> Vec<MinPlus<IndexType>>
{
}

impl<IndexType: GraphIndexInteger, F> CutMatrixFill<IndexType> for F where
    F: Fn(BlockIndex<IndexType>, &[NodeIndex<IndexType>]) -> Vec<MinPlus<IndexType>>
{
}

/// The orientation index (`0`/`1`) of a directed node.
fn orientation_index<IndexType: GraphIndexInteger>(node: DirectedNodeIndex<IndexType>) -> usize {
    if node.is_forward() { 0 } else { 1 }
}

/// The reversal identity of start-to-start lengths, which count the first node and not the last:
/// `D(w^p -> b) = D(b̄ -> w^{¬p}) + len(w) - len(b)`, from `reversed = D(b̄ -> w^{¬p})`.
fn reversed_length<IndexType: GraphIndexInteger>(
    reversed: GfaPathLength<IndexType>,
    w_len: u64,
    b_len: u64,
) -> GfaPathLength<IndexType> {
    GfaPathLength::from_usize((reversed.into_usize() as u64 + w_len - b_len) as usize)
}

/// The block-cut lift of a whole graph (all components), as an image-backed view (see
/// [`LiftSections`]).
#[derive(Clone, Copy)]
pub struct BlockCutLift<'a, IndexType: GraphIndexInteger> {
    number_of_blocks: usize,
    /// Navigation of the block-cut forest: tree ids `0..nb` are blocks, `nb..nb+nc` are cut nodes
    /// (dense ids = chain order), `nb+nc` is the super-root joining the components. Dense block id
    /// == `BlockIndex` value.
    nav: ChainNav<'a, IndexType>,
    /// Per cut (dense id): its node, parent/child block, prefix, length and flip closure.
    cuts: &'a [CutRec<IndexType>],
    /// Per block: the bits the query reads (row kind, special, cut-list order, away orientations,
    /// derived `M_aug`, flip cuts) and the special index (see `chain_nav::block_bits`).
    block_words: &'a [u32],
    /// Blocks whose cut count is not 2 (or root blocks): their cut lists (dense ids, the `M_aug` /
    /// node-row order), flip cuts and away bits.
    special_recs: &'a [SpecialBlockRec],
    special_cuts: &'a [u32],
    special_flips: &'a [u32],
    /// Sorted `(cut, pos)` lists for the many-cut blocks, and the dense row-major `(2 cuts)^2`
    /// `M_aug` of the blocks that keep one (a derived block has none).
    aug_sorted_dir: &'a [AugSortedDir],
    aug_sorted: &'a [AugSorted<IndexType>],
    aug_dense_dir: &'a [AugDenseDir],
    aug_dense_data: &'a [MinPlus<IndexType>],
    backend: BackendView<'a, IndexType>,
    /// Per-node cut rows in node order (see [`crate::node_rows`]); a node's start sits in its
    /// topology record.
    node_rows: &'a [MinPlus<IndexType>],
}

impl<'a, IndexType: GraphIndexInteger + 'static> BlockCutLift<'a, IndexType> {
    fn is_block(&self, tree_id: usize) -> bool {
        tree_id < self.number_of_blocks
    }

    fn cut_of(&self, tree_id: usize) -> usize {
        tree_id - self.number_of_blocks
    }

    fn tree_id_of_cut(&self, cut: usize) -> usize {
        self.number_of_blocks + cut
    }

    /// The graph node of a cut (dense id).
    #[inline]
    fn cut_node(&self, cut: usize) -> NodeIndex<IndexType> {
        self.cuts[cut].node
    }

    /// The node length of a cut (dense id).
    #[inline]
    fn cut_len(&self, cut: usize) -> GfaPathLength<IndexType> {
        self.cuts[cut].len
    }

    /// The whole-graph flip closure `full[c]` of a cut as a `2x2` (zero diagonal).
    #[inline]
    fn full(&self, cut: usize) -> Mat2<IndexType> {
        let rec = &self.cuts[cut];
        let mut m = Mat2::infinity();
        m.set(0, 0, MinPlus::zero());
        m.set(1, 1, MinPlus::zero());
        m.set(0, 1, rec.f01);
        m.set(1, 0, rec.f10);
        m
    }

    /// One entry of `full[c]`.
    #[inline]
    fn full_entry(&self, cut: usize, i: usize, j: usize) -> MinPlus<IndexType> {
        prefix_lift::full_entry(&self.cuts[cut], i, j)
    }

    /// Build the lift into `image` from the block-cut forest the flip-baking build leaves
    /// (`baked`): every block's raw cut-to-cut matrix `M_B` and the bottom-up flip closures.
    pub fn new<NodeData: GfaNodeData + Sync, EdgeData: GfaEdgeData + Sync>(
        image: &'a Image,
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        topology: &NodeTopology<'_, IndexType>,
        baked: BakedForest<IndexType>,
        options: &LiftOptions,
    ) -> (Self, LiftSections<IndexType>)
    where
        IndexType: Send + Sync,
    {
        let BakedForest {
            forest,
            raw,
            baked,
            down,
            gadget_up,
        } = baked;
        let blocks = forest.number_of_blocks;

        // Flip closures, then each block's `M_aug` and flip-capable cuts.
        let positions = forest.cut_positions();
        let cut_order = forest.cuts_by_depth();
        let full = full_closures(&forest, &raw, &positions, &down, &gadget_up, &cut_order);
        let block_aug = block_augs(&forest, raw, positions, &full);
        let flip_cuts = block_flip_cuts(&forest, &baked, &down, &full);
        let down_flip: Vec<bool> = down.iter().map(can_flip).collect();

        // The path product over the cut forest, then the node rows.
        let (cut_parent, backend) =
            build_backend(options.backend, &forest, &block_aug, &full, &cut_order);
        let rows = RowPlan::new(
            graph,
            topology,
            &forest,
            backend.prefix(),
            &flip_cuts,
            options.node_row_max_cuts,
        );
        let bake_arcs = baked_arcs(&forest, &baked, &down, &down_flip);
        let block_simple = RowBlocks {
            cuts: &forest.block_cuts,
            aug: &block_aug,
            flip_cuts: &flip_cuts,
            bake_arcs: &bake_arcs,
            plan: &rows,
        }
        .write(image, graph, topology);

        let BlockCutForest {
            cut_segment,
            cut_id,
            ..
        } = forest;
        let secs = LiftSections::write(
            image,
            LiftOwned {
                number_of_blocks: blocks,
                cut_segment,
                full,
                down_flip,
                block_aug,
                block_flip_cuts: flip_cuts,
                block_simple,
                cut_id,
                cut_parent,
                rows,
                backend,
                record_secs: topology.chain_record_sections(),
                root_blocks: topology.root_blocks(),
            },
        );
        (secs.view(image), secs)
    }
}
