//! Sets of [`GfaLocation`]s marking the targets of a [`GfaDijkstra`] search: one location
//! ([`single`]) or many ([`multi`]).
//!
//! [`GfaDijkstra`]: super::GfaDijkstra

use bidirected_adjacency_array::index::DirectedNodeIndex;

use crate::location::GfaLocation;

pub mod multi;
pub mod single;

pub trait GfaLocationIndex<IndexType> {
    /// Whether a target lies on `node`.
    fn contains(&self, node: DirectedNodeIndex<IndexType>) -> bool;

    /// The distinct targets.
    fn iter_targets(&self) -> impl Iterator<Item = GfaLocation<IndexType>>;

    /// The number of distinct targets.
    fn len(&self) -> usize;

    /// The number of distinct directed nodes holding a target: the search may stop once it has
    /// settled that many target nodes.
    fn node_count(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
