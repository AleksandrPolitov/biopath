//! A single-source shortest-path search restricted to one biconnected block.
//!
//! The cross-block engine needs, per query, the intra-block distances from the source to every cut
//! node of its block (to seed the cut-node overlay) and from every cut node of the target's block to
//! the target (to read it back). Computing these one cut node at a time is `O(#cut nodes)` intra-block
//! queries; instead this runs **one** single-source search per query that yields the distances to all
//! of the block's (directed) nodes at once.
//!
//! Correctness: two biconnected blocks share at most one node (a cut node is a single-node
//! separator), so any graph edge between two distinct nodes of a block is itself a block edge, and a
//! shortest walk between two block nodes never benefits from leaving the block (it would have to
//! re-cross the same cut node, a non-negative detour). Restricting the search to nodes of the block
//! therefore reproduces exactly the intra-block start-to-start distance — the same value the SPQR
//! engine computes, and the same cost model as [`crate::dijkstra::GfaDijkstra`] (leaving a node costs
//! its length), just confined to one block.

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex};
use binary_heap_plus::{BinaryHeap, MinComparator};
use epoch_reset_array::EpochResetArray;

use crate::{
    graph_view::QueryGraph,
    path::{GfaPathLength, OptionalGfaPathLength},
};

/// Reusable scratch for [`BlockDijkstra::run`]; a flat epoch-reset distance array plus a heap, so
/// repeated queries allocate nothing.
pub struct BlockDijkstra<IndexType: GraphIndexInteger> {
    open: BinaryHeap<OpenNode<IndexType>, MinComparator>,
    distance: EpochResetArray<DirectedNodeIndex<IndexType>, OptionalGfaPathLength<IndexType>, u32>,
}

#[derive(Debug, Eq, PartialEq)]
struct OpenNode<IndexType: GraphIndexInteger> {
    node: DirectedNodeIndex<IndexType>,
    cost: GfaPathLength<IndexType>,
}

impl<IndexType: GraphIndexInteger> BlockDijkstra<IndexType> {
    /// Allocate for a graph with `directed_node_count` directed nodes (`2 * node_count`).
    pub fn new(directed_node_count: usize) -> Self {
        Self {
            open: BinaryHeap::new_min(),
            distance: EpochResetArray::new(
                OptionalGfaPathLength::new_none(),
                directed_node_count.into(),
            ),
        }
    }

    /// Run a single-source start-to-start search from directed node `source` (offset `0`),
    /// expanding only into nodes for which `in_block` holds. After this returns, [`Self::get`]
    /// yields the start-to-start distance from `source` to any directed node settled in the block.
    ///
    /// The search seeds the source's successors at cost `len(source)` (leaving the source costs its
    /// length), exactly like [`crate::dijkstra::GfaDijkstra`], so the distances match the oracle.
    pub fn run<G: QueryGraph<IndexType> + ?Sized>(
        &mut self,
        graph: &G,
        source: DirectedNodeIndex<IndexType>,
        in_block: impl Fn(NodeIndex<IndexType>) -> bool,
    ) {
        self.open.clear();
        self.distance.reset();

        let source_length = graph.node_len(source.into_bidirected());
        for to in graph.out_neighbors(source) {
            if in_block(to.into_bidirected()) {
                self.open.push(OpenNode {
                    node: to,
                    cost: source_length,
                });
            }
        }

        while let Some(open) = self.open.pop() {
            if self.distance.get(open.node).into_option().is_some() {
                continue; // already settled
            }
            self.distance.set(open.node, open.cost.into());

            let from_cost = open.cost + graph.node_len(open.node.into_bidirected());
            for to in graph.out_neighbors(open.node) {
                if in_block(to.into_bidirected()) && self.distance.get(to).into_option().is_none() {
                    self.open.push(OpenNode {
                        node: to,
                        cost: from_cost,
                    });
                }
            }
        }
    }

    /// The start-to-start distance from the last run's source to `node`, or `None` if `node` was not
    /// reached within the block.
    pub fn get(&self, node: DirectedNodeIndex<IndexType>) -> Option<GfaPathLength<IndexType>> {
        self.distance.get(node).into_option()
    }
}

impl<IndexType: GraphIndexInteger> Ord for OpenNode<IndexType> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost
            .cmp(&other.cost)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl<IndexType: GraphIndexInteger> PartialOrd for OpenNode<IndexType> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
