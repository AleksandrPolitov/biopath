//! The node-row plan: per block, its row kind (see [`crate::node_rows`]), its cuts' away
//! orientations and whether its `M_aug` derives from the prefix lift.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::decomposition::indices::BlockIndex;

use super::forest::BlockCutForest;
use crate::{
    chain_nav::block_bits,
    min_plus::Mat2,
    node_rows::{FLIP_RESERVE_CUTS, RowKind, block_has_rows},
    prefix_lift::PrefixLift,
    topology::NodeTopology,
};

/// The most cuts of a block with half rows: its away orientations are the bits of a `u8`.
const HALF_ROW_MAX_CUTS: usize = u8::BITS as usize;

/// How every block keeps its node rows and its `M_aug`.
pub(super) struct RowPlan {
    /// Per block: its [`RowKind`] (as stored in its block word).
    pub(super) kinds: Vec<u8>,
    /// Per block: bit `j` = the away orientation of its `j`-th cut when the block is one-sided at
    /// every cut (else `0`).
    pub(super) away_bits: Vec<u8>,
    /// Per half-row block: the away orientation of each cut (empty for the other blocks).
    pub(super) away: Vec<Vec<usize>>,
    /// Per block: its `M_aug` derives from the prefix lift (a regular 2-cut block), so no dense
    /// data is written.
    pub(super) aug_derived: Vec<bool>,
}

impl RowPlan {
    /// The plan of every block. A block with `1..=max_cuts` cuts gets half rows when it is
    /// one-sided at every cut ([`away_bits`]), full rows otherwise; a block with more cuts but at
    /// most [`FLIP_RESERVE_CUTS`] flip-capable ones gets flip rows. Under the prefix lift, a block
    /// whose child steps are all regular with exactly two cuts (its parent cut and one child) keeps
    /// no dense `M_aug`: it derives from the prefix and the flip closures.
    pub(super) fn new<IndexType, NodeData, EdgeData>(
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        topology: &NodeTopology<'_, IndexType>,
        forest: &BlockCutForest<IndexType>,
        prefix: Option<&PrefixLift<IndexType>>,
        flip_cuts: &[Vec<(NodeIndex<IndexType>, Mat2<IndexType>)>],
        max_cuts: usize,
    ) -> Self
    where
        IndexType: GraphIndexInteger,
        NodeData: GfaNodeData,
        EdgeData: GfaEdgeData,
    {
        let blocks = 0..forest.number_of_blocks;
        let parent_pos = parent_cut_positions(forest, topology);
        let one_sided: Vec<Option<u8>> = blocks
            .clone()
            .map(|bid| away_bits(graph, topology, bid, &forest.block_cuts[bid]))
            .collect();
        let kinds: Vec<u8> = blocks
            .clone()
            .map(|bid| {
                let cuts = forest.block_cuts[bid].len();
                row_kind(
                    cuts,
                    max_cuts,
                    one_sided[bid].is_some(),
                    flip_cuts[bid].len(),
                ) as u8
            })
            .collect();
        let aug_derived = blocks
            .clone()
            .map(|bid| {
                forest.block_cuts[bid].len() == 2
                    && is_regular_block(forest, prefix, parent_pos[bid], bid)
            })
            .collect();
        let away = blocks
            .map(|bid| {
                if kinds[bid] != RowKind::Half as u8 {
                    return Vec::new();
                }
                let bits = one_sided[bid].expect("a half-row block is one-sided at every cut");
                (0..forest.block_cuts[bid].len())
                    .map(|j| ((bits >> j) & 1) as usize)
                    .collect()
            })
            .collect();
        Self {
            kinds,
            away_bits: one_sided.iter().map(|b| b.unwrap_or(0)).collect(),
            away,
            aug_derived,
        }
    }

    pub(super) fn kind(&self, block: usize) -> RowKind {
        RowKind::from_u8(self.kinds[block])
    }
}

/// The row kind of a block with `cuts` cuts and `flip_cuts` flip-capable ones.
fn row_kind(cuts: usize, max_cuts: usize, one_sided: bool, flip_cuts: usize) -> RowKind {
    if max_cuts == 0 || cuts == 0 {
        RowKind::None
    } else if block_has_rows(cuts, max_cuts) {
        if one_sided {
            RowKind::Half
        } else {
            RowKind::Full
        }
    } else if (1..=FLIP_RESERVE_CUTS).contains(&flip_cuts) {
        RowKind::Flip
    } else {
        RowKind::None
    }
}

/// The position of each block's parent cut in its cut list (`None` at a component root).
fn parent_cut_positions<IndexType: GraphIndexInteger>(
    forest: &BlockCutForest<IndexType>,
    topology: &NodeTopology<'_, IndexType>,
) -> Vec<Option<usize>> {
    let positions: Vec<Option<usize>> = (0..forest.number_of_blocks)
        .map(|bid| {
            forest.parent_cut(bid).map(|cut| {
                forest.block_cuts[bid]
                    .iter()
                    .position(|&s| s == forest.cut_segment[cut])
                    .expect("a block's parent cut is in its cut list")
            })
        })
        .collect();
    if cfg!(debug_assertions) {
        for (bid, &bits) in topology.block_words().iter().enumerate() {
            if bits & block_bits::SPECIAL == 0 {
                debug_assert_eq!(
                    bits & block_bits::PARENT_POS != 0,
                    positions[bid] == Some(1),
                    "the topology's parent-position bit matches the lift's"
                );
            }
        }
    }
    positions
}

/// Whether every child step of block `bid` is regular under the prefix lift (`false` without one,
/// or at a component root).
fn is_regular_block<IndexType: GraphIndexInteger>(
    forest: &BlockCutForest<IndexType>,
    prefix: Option<&PrefixLift<IndexType>>,
    parent_pos: Option<usize>,
    bid: usize,
) -> bool {
    let (Some(prefix), Some(parent_pos)) = (prefix, parent_pos) else {
        return false;
    };
    forest.block_cuts[bid]
        .iter()
        .enumerate()
        .all(|(j, &segment)| j == parent_pos || prefix.is_regular_step(forest.cut_id[&segment]))
}

/// The away orientations of a block that touches each of its (at most [`HALF_ROW_MAX_CUTS`]) cuts
/// on one side, as bits (`j`: the `j`-th cut); `None` if it touches one on both sides. Every route
/// from inside such a block to a cut's other orientation passes its away orientation first (and
/// the cut's whole-graph flip closure is the cheapest way on). The side is read off the graph: the
/// orientation `o` of the arcs `w^o -> y` into the block is the "into" traversal, so the away
/// orientation is `1 - o`.
fn away_bits<IndexType, NodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    topology: &NodeTopology<'_, IndexType>,
    bid: usize,
    cuts: &[NodeIndex<IndexType>],
) -> Option<u8>
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    if cuts.len() > HALF_ROW_MAX_CUTS {
        return None;
    }
    let block_index = BlockIndex::from_usize(bid);
    let mut bits = 0u8;
    for (j, &w) in cuts.iter().enumerate() {
        let mut into: Option<usize> = None;
        for o in 0..2 {
            for edge in graph.iter_outgoing_edges(w.into_directed(o == 0)) {
                let y = edge.to().into_bidirected();
                if y == w || !topology.node_block_indices(y).any(|b| b == block_index) {
                    continue;
                }
                match into {
                    None => into = Some(o),
                    Some(prev) if prev != o => return None,
                    Some(_) => {}
                }
            }
        }
        let into = into?;
        bits |= ((1 - into) as u8) << j;
    }
    Some(bits)
}
