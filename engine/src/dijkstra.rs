//! The test oracle: whole-graph Dijkstra on a bidirected GFA graph ([`GfaDijkstra`]), with its
//! source/target sets ([`location_index`]) and the walks it returns ([`gfa_path`]).
//!
//! It is no part of the index. The tests check the index against it, `biopath query --engine
//! dijkstra` runs it as the brute-force baseline and `--compare-whole-graph-dijkstra` cross-checks
//! every answer with it. Its cost model is the index's: a walk costs the rest of the source node
//! after the source offset, the whole length of every node it passes, and the target offset.

use std::collections::HashMap;

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, GraphIndexInteger, OptionalDirectedNodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use binary_heap_plus::{BinaryHeap, MinComparator};
use epoch_reset_array::EpochResetArray;

use self::{
    gfa_path::{GfaPath, PathElement},
    location_index::GfaLocationIndex,
};
use crate::{
    gfa_graph_extensions::GfaNodeDataExt,
    location::{GfaLocation, GfaNodeOffset},
    path::{GfaPathLength, OptionalGfaPathLength},
};

pub mod gfa_path;
pub mod location_index;
#[cfg(test)]
mod tests;

pub struct GfaDijkstra<
    'graph,
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
> {
    graph: &'graph BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    open_list: BinaryHeap<OpenNode<IndexType>, MinComparator>,
    closed_list: EpochResetArray<DirectedNodeIndex<IndexType>, ClosedNode<IndexType>, u32>,
}

#[derive(Debug, Eq, PartialEq)]
struct OpenNode<IndexType: GraphIndexInteger> {
    node: DirectedNodeIndex<IndexType>,
    cost: GfaPathLength<IndexType>,
    predecessor: OptionalDirectedNodeIndex<IndexType>,
}

#[derive(Debug, Clone)]
struct ClosedNode<IndexType: GraphIndexInteger> {
    cost: OptionalGfaPathLength<IndexType>,
    predecessor: OptionalDirectedNodeIndex<IndexType>,
}

impl<'graph, IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData: GfaEdgeData>
    GfaDijkstra<'graph, IndexType, NodeData, EdgeData>
{
    pub fn new(graph: &'graph BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>) -> Self {
        let directed_node_count = graph.node_count() * 2;
        Self {
            graph,
            open_list: BinaryHeap::new_min(),
            closed_list: EpochResetArray::new(
                ClosedNode {
                    cost: OptionalGfaPathLength::new_none(),
                    predecessor: OptionalDirectedNodeIndex::new_none(),
                },
                directed_node_count.into(),
            ),
        }
    }

    /// Compute the shortest paths from the source to all given targets.
    ///
    /// If there is a target that is before the source in the same node, then this target's path may be a cycle.
    ///
    /// Every offset must lie in `0..=len` of its node. The search panics, rather than wrap, when a
    /// distance does not fit `IndexType` (see [`crate::graph_limits`]).
    pub fn shortest_paths(
        &mut self,
        source: GfaLocation<IndexType>,
        targets: &impl GfaLocationIndex<IndexType>,
    ) -> HashMap<GfaLocation<IndexType>, GfaPath<IndexType>> {
        self.open_list.clear();
        self.closed_list.reset();

        // By convention, we insert only the successors of the source node into the open list.
        // Therefore, each path will have one "missing" node at the start which we will handle separately when backtracking.
        // This is to enable circular paths in the case where the shortest path to a target starts from the same node but with a higher offset.
        let source_len = self.graph.node_data(source.node().into_bidirected()).len();
        debug_assert!(
            source.offset().into_length() <= source_len,
            "source offset beyond its node"
        );
        self.expand_node(
            source.node(),
            source_len - source.offset().into_length(),
            true,
        );

        let mut closed_target_counter = 0;

        while let Some(open_node) = self.open_list.pop()
            && closed_target_counter < targets.node_count()
        {
            let previous_cost = self.closed_list.get(open_node.node).cost;
            if let Some(previous_cost) = previous_cost.into_option() {
                debug_assert!(previous_cost <= open_node.cost);

                // Node already closed, skip.
                continue;
            }

            // Close node.
            self.closed_list.set(
                open_node.node,
                ClosedNode {
                    cost: open_node.cost.into(),
                    predecessor: open_node.predecessor,
                },
            );

            if targets.contains(open_node.node) {
                closed_target_counter += 1;
            }

            // Expand node.
            self.expand_node(
                open_node.node,
                checked_sum(
                    open_node.cost,
                    self.graph.node_data(open_node.node.into_bidirected()).len(),
                ),
                false,
            );
        }

        // All targets found, backtrack paths and compute actual cost.
        targets
            .iter_targets()
            .filter_map(|target| {
                let mut path = (target.node() == source.node()
                    && target.offset() >= source.offset())
                .then(|| {
                    GfaPath::new(
                        vec![PathElement::new(
                            source.node(),
                            source.offset(),
                            target.offset(),
                        )],
                        target.offset() - source.offset(),
                    )
                });

                if let Some(cost) = self.closed_list.get(target.node()).cost.into_option() {
                    let cost = checked_sum(cost, target.offset().into_length());
                    let outer_path = self.backtrack_path(source, target, cost);
                    if path
                        .as_ref()
                        .map(|p| outer_path.length() < p.length())
                        .unwrap_or(true)
                    {
                        path = Some(outer_path);
                    }
                }

                path.map(|p| (target, p))
            })
            .collect()
    }

    fn expand_node(
        &mut self,
        from_node: DirectedNodeIndex<IndexType>,
        from_cost: GfaPathLength<IndexType>,
        is_source_node: bool,
    ) {
        for outgoing_edge in self.graph.iter_outgoing_edges(from_node) {
            debug_assert_eq!(
                self.graph
                    .directed_edge_data(outgoing_edge.index())
                    .data()
                    .overlap(),
                0,
                "Only GFA graphs with blunt-ended (i.e. zero overlap) edges are supported. Use for example https://github.com/vgteam/GetBlunted to bluntify your graph.",
            );

            let to_node = outgoing_edge.to();
            let to_cost = from_cost;
            let to_predecessor = if is_source_node {
                OptionalDirectedNodeIndex::new_none()
            } else {
                from_node.into()
            };

            let closed_node = self.closed_list.get(to_node);
            if let Some(closed_cost) = closed_node.cost.into_option() {
                // Node already closed, ensure that we did not find a shorter path.
                assert!(closed_cost <= to_cost);
            } else {
                self.open_list.push(OpenNode {
                    node: to_node,
                    cost: to_cost,
                    predecessor: to_predecessor,
                });
            }
        }
    }

    fn backtrack_path(
        &self,
        source: GfaLocation<IndexType>,
        target: GfaLocation<IndexType>,
        cost: GfaPathLength<IndexType>,
    ) -> GfaPath<IndexType> {
        // Initialise path with target node.
        let mut path = vec![PathElement::new(
            target.node(),
            GfaNodeOffset::from_usize(0),
            target.offset(),
        )];

        // Collect nodes.
        let mut current_node = target.node();
        while let Some(predecessor) = self.closed_list.get(current_node).predecessor.into_option() {
            let offset = GfaNodeOffset::from_usize(0);
            let limit = self
                .graph
                .node_data(predecessor.into_bidirected())
                .len()
                .into_offset();
            path.push(PathElement::new(predecessor, offset, limit));
            current_node = predecessor;
        }

        // Manually add the source node as we started the search from its successors.
        path.push(PathElement::new(
            source.node(),
            source.offset(),
            self.graph
                .node_data(source.node().into_bidirected())
                .len()
                .into_offset(),
        ));

        // Return path.
        path.reverse();
        GfaPath::new(path, cost)
    }
}

/// `a + b`, or a panic when the sum does not fit `IndexType` (whose `MAX` means no path): the
/// oracle must never answer a wrapped distance.
fn checked_sum<IndexType: GraphIndexInteger>(
    a: GfaPathLength<IndexType>,
    b: GfaPathLength<IndexType>,
) -> GfaPathLength<IndexType> {
    a.checked_add(b).unwrap_or_else(|| {
        panic!(
            "whole-graph Dijkstra: a distance exceeds the {}-bit word size; use a larger --word-size",
            IndexType::max_value().count_ones()
        )
    })
}

impl<IndexType: GraphIndexInteger> Ord for OpenNode<IndexType> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost
            .cmp(&other.cost)
            .then_with(|| self.node.cmp(&other.node))
            .then_with(|| self.predecessor.cmp(&other.predecessor))
    }
}

impl<IndexType: GraphIndexInteger> PartialOrd for OpenNode<IndexType> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
