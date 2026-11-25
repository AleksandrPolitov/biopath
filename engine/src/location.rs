//! A position in a GFA graph: a directed node and an offset into it ([`GfaLocation`]), the
//! endpoint of a query.

use std::fmt::Display;

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};
use optional_numeric_index::implement_generic_index;

implement_generic_index!(pub GfaNodeOffset, pub OptionalGfaNodeOffset);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GfaLocation<IndexType> {
    node: DirectedNodeIndex<IndexType>,
    offset: GfaNodeOffset<IndexType>,
}

impl<IndexType: GraphIndexInteger> GfaLocation<IndexType> {
    pub fn new(node: DirectedNodeIndex<IndexType>, offset: GfaNodeOffset<IndexType>) -> Self {
        Self { node, offset }
    }

    pub fn node(&self) -> DirectedNodeIndex<IndexType> {
        self.node
    }

    pub fn offset(&self) -> GfaNodeOffset<IndexType> {
        self.offset
    }
}

impl<IndexType: GraphIndexInteger> GfaNodeOffset<IndexType> {
    pub fn into_length(self) -> crate::path::GfaPathLength<IndexType> {
        crate::path::GfaPathLength::from_raw(self.into_raw())
    }
}

impl<IndexType: GraphIndexInteger> Display for GfaLocation<IndexType> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}[{}]",
            self.node.into_bidirected(),
            if self.node.is_forward() { "+" } else { "-" },
            self.offset,
        )
    }
}
