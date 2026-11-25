use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, GraphIndexInteger},
};
use bitvec::{bitvec, vec::BitVec};

use crate::{dijkstra::location_index::GfaLocationIndex, location::GfaLocation};

/// Any number of target positions, several of them possibly on one directed node (or equal).
pub struct MultiGfaLocationIndex<IndexType> {
    /// The directed nodes holding a target.
    is_contained_index: BitVec,
    /// The distinct targets, sorted.
    targets: Vec<GfaLocation<IndexType>>,
    /// The number of distinct directed nodes holding a target.
    node_count: usize,
}

impl<IndexType: GraphIndexInteger> MultiGfaLocationIndex<IndexType> {
    pub fn new_targets<NodeData, EdgeData>(
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        targets: impl IntoIterator<Item = GfaLocation<IndexType>>,
    ) -> Self {
        let directed_node_count = graph.node_count() * 2;
        let mut is_contained_index = bitvec![0; directed_node_count];
        let mut targets: Vec<_> = targets.into_iter().collect();
        targets.sort_unstable();
        targets.dedup();
        let mut node_count = 0;
        for target in &targets {
            if !is_contained_index[target.node().into_usize()] {
                is_contained_index.set(target.node().into_usize(), true);
                node_count += 1;
            }
        }

        Self {
            is_contained_index,
            targets,
            node_count,
        }
    }
}

impl<IndexType: GraphIndexInteger> GfaLocationIndex<IndexType>
    for MultiGfaLocationIndex<IndexType>
{
    fn contains(&self, node: DirectedNodeIndex<IndexType>) -> bool {
        *self
            .is_contained_index
            .get(node.into_usize())
            .expect("Node index out of bounds")
    }

    fn iter_targets(&self) -> impl Iterator<Item = GfaLocation<IndexType>> {
        self.targets.iter().copied()
    }

    fn len(&self) -> usize {
        self.targets.len()
    }

    fn node_count(&self) -> usize {
        self.node_count
    }
}
