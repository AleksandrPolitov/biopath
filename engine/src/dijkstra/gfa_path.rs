//! A walk in a GFA graph as [`GfaDijkstra`](super::GfaDijkstra) returns it: the visited directed
//! nodes, each with the offset range it covers, and the total length.

use std::fmt::Debug;

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use crate::{location::GfaNodeOffset, path::GfaPathLength};

pub struct GfaPath<IndexType> {
    path: Vec<PathElement<IndexType>>,
    length: GfaPathLength<IndexType>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PathElement<IndexType> {
    node: DirectedNodeIndex<IndexType>,
    offset: GfaNodeOffset<IndexType>,
    limit: GfaNodeOffset<IndexType>,
}

impl<IndexType: GraphIndexInteger> GfaPath<IndexType> {
    pub fn new(path: Vec<PathElement<IndexType>>, length: GfaPathLength<IndexType>) -> Self {
        Self { path, length }
    }

    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = PathElement<IndexType>> {
        self.path.iter().copied()
    }

    pub fn length(&self) -> GfaPathLength<IndexType> {
        self.length
    }
}

impl<IndexType: GraphIndexInteger> PathElement<IndexType> {
    pub fn new(
        node: DirectedNodeIndex<IndexType>,
        offset: GfaNodeOffset<IndexType>,
        limit: GfaNodeOffset<IndexType>,
    ) -> Self {
        Self {
            node,
            offset,
            limit,
        }
    }

    pub fn node(&self) -> DirectedNodeIndex<IndexType> {
        self.node
    }

    pub fn offset(&self) -> GfaNodeOffset<IndexType> {
        self.offset
    }

    pub fn limit(&self) -> GfaNodeOffset<IndexType> {
        self.limit
    }
}

impl<IndexType: GraphIndexInteger> Debug for GfaPath<IndexType> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GfaPath(length: {}, path: [", self.length)?;

        for (i, element) in self.path.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{}[{}..{}]",
                element.node(),
                element.offset(),
                element.limit()
            )?;
        }

        write!(f, "])")
    }
}
