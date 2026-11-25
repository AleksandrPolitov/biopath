//! Exact pre-reduction of the input graph (`biopath index` applies it by default): the engine is
//! built on a smaller graph G′ with the same distances, and an alias maps every original position
//! onto G′.
//!
//! # The reduction
//!
//! Every node has a left and a right *side*, and a link joins two sides (`side_graph.rs`). Two nodes
//! whose facing sides each carry exactly one link, to each other, are always walked straight
//! through: a maximal run of them is a *chain* and behaves like one node. A chain whose two outer
//! sides each carry exactly one link, to sides outside it, is an *allele* between those two partner
//! sides. Only the shortest allele between two partner sides can be on a shortest walk, and a direct
//! link between them dominates every allele there; so a worklist prunes each dominated allele, one
//! at a time (each prune valid in the graph of the moment), joins what the prune leaves walked
//! straight through and re-examines it, until nothing changes (`reduce.rs`). A *tip*, a chain with
//! one outer side unlinked and the other carrying one link to a side outside it, is never walked
//! through, so the worklist prunes it too; its unlinked end has no port. Zero-length nodes are never chained or pruned.
//!
//! G′ keeps every chain left at the end (cycles included) as one node,
//! walked from its lower side id, and the alive links between chain ends (`assemble.rs`).
//!
//! # The alias
//!
//! Every original node lies in a *super-node*: a G′ node or a pruned chain. A pruned chain records
//! its *ports* — the directed original node it exits into and the one it is entered from, both
//! alive when it was pruned (a pruned tip lacks the one at its unlinked end) — and a *level*, one more than the largest level of a pruned chain
//! whose port lies on one of its nodes. A single-node pruned chain (nearly every pruned allele) is a
//! 4-byte record of its own; every other node is a member of a *run*, walk-consecutive members of one
//! super-node whose ids are consecutive among the non-single nodes, so a member's offset is its run
//! head's plus a prefix sum of lengths (`records.rs`, built by `layout.rs`).
//!
//! An original position `(node, orientation, offset)` resolves to its super-node, the direction it
//! walks that super-node in and its offset along that walk (`view.rs`).
//!
//! # A query
//!
//! A query on two original positions resolves both (`distance.rs`):
//!
//! - **One super-node, the target ahead:** the walk inside it is the shortest, so the answer is the
//!   difference of the offsets.
//! - **A position in a pruned chain:** take the one pruned first (the lower level). When it was
//!   pruned the other position was still alive, so the walk is forced out through the chain's end
//!   (a source) or in through its start (a target); it continues from that port, a *hop*. A pruned
//!   tip has no port at its unlinked end: walked towards it, nothing outside is reached, and nothing
//!   outside reaches it walked away from it. Every port
//!   was pruned later than its chain, or never, so the hops end. Two chains of one level are not
//!   attached to each other, and then either order is exact. Two chains at the saturated level have
//!   no known order: both hop sequences are walked once, and the shortest of the walks they allow
//!   (inside a chain both pass, or on G′ where both end) is exact.
//! - **Two G′ positions:** one engine query on G′, or a query *around* the node (leave it and come
//!   back) when both lie on the same G′ node, walked the same way, with the target behind.
//!
//! **Coinciding positions.** Gluing `v` then `w` makes the end of `v` and the start of `w` the same
//! offset of the merged node, but in the original graph the end of `v` is reached from the start of
//! `w` only by going around. Positions are therefore compared on the key `2 * offset - [end of a
//! node]`, so an end-of-node position sorts just before a start-of-node position at the same offset.
//! Zero-length nodes are never chained or pruned, so no two start positions coincide.
//!
//! [`ShrinkAttach`] writes the alias and the original names into the index (`store.rs`).

mod assemble;
mod chains;
mod distance;
#[cfg(test)]
mod fuzz_tests;
mod layout;
mod records;
mod reduce;
mod side_graph;
mod store;
mod view;

pub use distance::{ShrunkOracle, ShrunkPos};
pub use reduce::shrink_graph;
pub use store::{ShrinkAttach, ShrinkSections, ShrinkSink};
pub use view::{OriginalPos, ShrinkView, SuperLocation, SuperNode};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray, index::GraphIndexInteger, io::gfa1::PlainGfaEdgeData,
};

use crate::gfa_graph_extensions::LengthOnlyGfaNodeData;
use layout::RunLayout;

/// The most original nodes the alias encodes (directed ids stay below 2^31).
const MAX_ORIGINAL_NODES: u64 = (1 << 30) - 1;
/// The longest super-node (G′ node or pruned chain) the alias encodes: offsets are `u32`.
const OFF_MASK: u64 = u32::MAX as u64;

/// The reduced graph and the alias of every original node.
pub struct ShrinkPlan<IndexType: GraphIndexInteger> {
    /// G′: the graph the decomposition and the engine are built on.
    pub graph: BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>,
    /// The alias arrays.
    runs: RunLayout,
    pub stats: ShrinkStats,
    original_fingerprint: u64,
}

/// What the reduction did.
#[derive(Clone, Debug, Default)]
pub struct ShrinkStats {
    pub original_nodes: usize,
    pub original_edges: usize,
    pub duplicate_edges: usize,
    /// Pruned allele chains and their nodes.
    pub allele_chains: usize,
    pub allele_nodes: usize,
    /// Allele chains pruned because a direct link dominates their group.
    pub direct_chains: usize,
    /// Dead-end tips pruned and their nodes.
    pub tip_chains: usize,
    pub tip_nodes: usize,
    pub shrunk_nodes: usize,
    pub shrunk_edges: usize,
    pub cyclic_chains: usize,
    pub max_node_len: u64,
    /// The largest level of a pruned chain (see the alias).
    pub max_level: u32,
}
