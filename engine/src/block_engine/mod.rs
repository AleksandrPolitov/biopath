//! The per-block engine: one block's record, and the start-to-start query inside the block.
//!
//! # The record
//!
//! A block that is not small gets a record ([`crate::block_record`]) from the DP pipeline of
//! `engine_build/record.rs`: the inside DP ([`crate::inside_dp`], each node's `4x4` pole-state
//! distances through its subtree), the outside DP ([`crate::outside_dp`], `D_out`, the routes
//! that leave a node's subtree), the pole-lift steps ([`crate::pole_lift`]) and the per-node
//! oracles (`precompute.rs`, [`crate::s_node_segtree`], [`crate::skeleton_index`]). It keeps
//! what queries read: the rooted SPQR tree and every node's reference poles, the pole-lift steps,
//! every node's inner skeleton (without arcs where a table and labels answer instead), `D_out` of
//! the nodes whose LCA search runs over a skeleton, and the oracles of the big nodes: S-node segment
//! trees, hub labels and pole-projection tables. Which node gets which oracle is decided in
//! `tiers.rs` ([`Tiers`]).
//!
//! # A start-to-start query
//!
//! [`BlockQueryEngine::start_to_start`] answers `a -> b` (offsets `0 -> 0`) inside the block:
//!
//! 1. **Locate** (`query.rs`): the deepest SPQR nodes `x_a`, `x_b` containing the endpoints and
//!    their LCA `L`.
//! 2. **Seeds** (`projection.rs`): a source in `L`'s skeleton seeds itself at `0`. Otherwise its
//!    pole projection, its distances to `x_a`'s reference poles ([`ProjectionOracle`]: a segment
//!    tree, a projection table or an inner-skeleton Dijkstra, memoized per thread in `memo.rs`),
//!    climbs the pole-lift steps to the poles of `L`'s child towards `x_a`, which seed `L`.
//! 3. **LCA read** (`lca.rs`, [`LcaOracle`]): the distances from the seeds over `L`'s LCA-role
//!    skeleton, by its segment tree, by label merges (memoized per thread) or by one multi-source
//!    Dijkstra.
//! 4. **Target read** (`target.rs`): a target in `L`'s skeleton is read directly. Otherwise the LCA
//!    distances to the poles of `L`'s child towards `x_b` meet the target's reverse projection,
//!    lifted to that child.
//!
//! [`BlockQueryEngine::distance`] adds the location offsets, the within-node walk and, for one
//! directed node, the cyclic route, following [`GfaDijkstra`](crate::dijkstra)'s convention.
//!
//! The block-cut lift reads a big block's raw cut-to-cut matrix through
//! [`BlockQueryEngine::fill_cut_matrix`] (`cut_matrix.rs`): every cut's projection is warmed once,
//! and one LCA read serves every target cut under the same LCA.

mod cut_matrix;
mod lca;
mod memo;
mod precompute;
mod projection;
mod query;
mod target;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
mod tiers;
mod up_to_four;

pub use precompute::{precompute_labels, precompute_projections, precompute_skeletons};
pub use projection::ProjectionCache;
pub use tiers::{
    LABEL_MIN_DIRECTED_NODES, LcaOracle, PROJ_TABLE_MIN_STATES, ProjectionOracle,
    SNODE_SEGTREE_MIN_STATES, Tiers,
};

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};

use crate::{
    block_record::BlockView,
    graph_view::GraphView,
    topology::{NodeTopology, SpqrLocation},
};

/// The query engine of one block: its image record ([`BlockView`]), the node topology and the
/// image graph.
pub struct BlockQueryEngine<'a, IndexType: GraphIndexInteger> {
    /// The per-node projection of the decomposition, used to resolve a segment's SPQR nodes.
    topology: NodeTopology<'a, IndexType>,
    /// The block's record.
    block: &'a BlockView<'a, IndexType>,
    /// Node lengths and adjacency.
    graph: GraphView<'a, IndexType>,
    /// Optional precomputed pole projections (set by [`Self::fill_cut_matrix`]). When present,
    /// `pole_projection` returns the cached `PoleVector` instead of running the projection search —
    /// the whole point of the batched cut-matrix fill. A miss falls through to the search (so it is
    /// always correct, only faster when warmed). `None` by default.
    projection_cache: Option<&'a ProjectionCache<IndexType>>,
}

// `IndexType: 'static` is required by the `PROJ_MEMO` thread-local's `Any` downcast. It is not a real
// restriction: `GraphIndexInteger` is only implemented for u8/u16/u32/u64.
impl<'a, IndexType: GraphIndexInteger + 'static> BlockQueryEngine<'a, IndexType> {
    /// Bundle a block's record, the topology and the graph into a query engine.
    pub fn new(
        topology: NodeTopology<'a, IndexType>,
        block: &'a BlockView<'a, IndexType>,
        graph: GraphView<'a, IndexType>,
    ) -> Self {
        Self {
            topology,
            block,
            graph,
            projection_cache: None,
        }
    }

    /// Attach a precomputed pole-projection cache (from [`Self::warm_cut_projections`]), as
    /// [`Self::fill_cut_matrix`] does with its own. Purely a speedup: a cache hit skips the projection
    /// search, a miss falls through to it, so results are identical either way. For the tests.
    #[cfg(test)]
    pub fn with_projection_cache(mut self, cache: &'a ProjectionCache<IndexType>) -> Self {
        self.projection_cache = Some(cache);
        self
    }

    /// Where `segment` sits in this block's SPQR tree (`None` = not in this block).
    #[inline]
    fn spqr_location(&self, segment: NodeIndex<IndexType>) -> Option<SpqrLocation> {
        self.topology.spqr_location(segment, self.block.block())
    }
}
