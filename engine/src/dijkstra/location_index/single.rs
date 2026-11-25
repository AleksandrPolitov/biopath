use std::iter;

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use crate::{dijkstra::location_index::GfaLocationIndex, location::GfaLocation};

pub struct SingleGfaLocationIndex<IndexType> {
    location: GfaLocation<IndexType>,
}

impl<IndexType: GraphIndexInteger> SingleGfaLocationIndex<IndexType> {
    pub fn new_target(target: GfaLocation<IndexType>) -> Self {
        Self { location: target }
    }
}

impl<IndexType: GraphIndexInteger> GfaLocationIndex<IndexType>
    for SingleGfaLocationIndex<IndexType>
{
    fn contains(&self, node: DirectedNodeIndex<IndexType>) -> bool {
        self.location.node() == node
    }

    fn iter_targets(&self) -> impl Iterator<Item = GfaLocation<IndexType>> {
        iter::once(self.location)
    }

    fn len(&self) -> usize {
        1
    }

    fn node_count(&self) -> usize {
        1
    }
}
