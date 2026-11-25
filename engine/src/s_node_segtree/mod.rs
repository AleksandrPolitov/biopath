//! S-node distance oracle: a series–parallel **segment tree** over the cycle
//! skeleton, answering any state-to-state skeleton distance in `O(log m)` with `O(m)` space. The
//! `4 × 4` network algebra (series `merge`, `parallel`) lives in [`crate::min_plus::network`].
//!
//! An S-node skeleton with `m` vertices is a ring of `2m` directed states, a poor fit for both a
//! per-query Dijkstra (`O(m log m)`) and hub labels (about three times the bytes of the tree). One
//! oracle answers both roles of the node, the inner skeleton (pole projections) and the LCA-role
//! skeleton (with `D_out`), so a node with a tree keeps neither skeleton nor labels; which S-nodes
//! get one is decided by [`crate::block_engine::Tiers`].
//!
//! The build finds the cycle and its leaf networks (`cycle.rs`), then lays out the ring form
//! (`ring_build.rs`, queried by `ring.rs`) or the segment tree (`tree.rs`); `access.rs` holds the
//! query API shared by the owned oracle and its image view (`record.rs`).
//!
//! # The decomposition
//!
//! Lay the cycle out as `v_0 … v_{m-1}`. Leaf `e` (`0 ≤ e ≤ m-2`) is the 2-terminal network over
//! `(v_e, v_{e+1})`: the pass-through arcs of that cycle edge plus the **rungs** (same-vertex
//! orientation flips) of `v_e`. The **closing edge** `(v_{m-1}, v_0)` is kept out of the tree and
//! stored as two networks: `closing_inner` (its inner arcs plus the rungs of `v_{m-1}`) and
//! `closing_lca` (the same with the node's `D_out` folded in). For a non-root node the cycle order
//! starts at the reference pole `s` and ends at `t`, so the closing edge *is* the reference edge:
//! the inner skeleton is the open path `v_0 … v_{m-1}` and the LCA-role skeleton is the ring closed by
//! `D_out`. One tree therefore serves both roles.
//!
//! A query between positions `p < q` composes the two arcs between them — leaves `[p, q)` from the
//! tree, and leaves `[q, m-1)`, the closing network, leaves `[0, p)` for the wrap-around — then takes
//! their elementwise min and closes the `4 × 4` over the boundary states `[(p,±), (q,±)]`, because a
//! walk may bounce between the two arcs any number of times. Every rung lies in exactly one leaf, so
//! each is available in exactly one arc; `D_out`'s rungs at `v_0` live in the closing network rather
//! than in leaf `0`, which is equivalent (min-plus is idempotent: an arc present in both arcs or in
//! one changes no distance).
//!
//! # Values
//!
//! The matrices hold `d` — the empty walk allowed, zero diagonal — over the engine's
//! start-to-start arcs, so a stored entry is exactly what a skeleton Dijkstra over the same arcs
//! returns. The engine never needs a closed-walk `d⁺` diagonal (a same-state query is answered as
//! `0`, the Dijkstra convention), so none is stored.
//!
//! # The ring form (default)
//!
//! The tree is the general fallback; the **ring form** ([`SEG_KIND_RING`]) is what every S-node
//! gets by default. It cuts the cycle at its *exception* leaves — a leaf passing in two orientation
//! pairs or none (a dead child), a leaf after a lane break, and always the closing — into *runs* of
//! *regular* leaves (one lane each, mirrored back arcs, lane-consistent with the previous leaf).
//! Inside a run a monotone walk never changes lane and every rung is a turnaround, so a run is a
//! prefix sum plus two hairpin minima; an exception is kept as its closed `4 × 4`. A block sequence
//! `U_0 X_0 U_1 … X_{K−2} U_{K−1}` (empty runs omitted) carries a small segment tree, each run stores
//! the `4 × 4` of "the ring minus this run" per role, and each vertex stores its global cheapest
//! flips per role. A query is then: flip-or-not at `u`, the lane walk to the end of `u`'s run, one
//! range product over the blocks between the runs (or the stored around-summary when both lie in the
//! same run), the lane walk from the start of `v`'s run, flip-or-not at `v`; the cw and the ccw
//! route both, min. `O(m + K log K)` build, `O(1)` query inside a run and `O(log K)` across runs —
//! never a decline on a cycle, so no S-node falls back to labels or a per-query Dijkstra.

mod access;
mod build;
mod cycle;
mod record;
mod ring;
mod ring_build;
#[cfg(test)]
mod tests;
mod tree;

pub use access::SegTreeAccess;
pub use build::{SNodeSegTree, precompute_s_node_segtrees};
pub use record::{
    PfxVertex, SEG_KIND_RING, SEG_KIND_TREE, SegRec, SegRunRec, SegTable, SegTableLayout,
    SegTableSizes, SegTreeView, SegVertexRec,
};

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

/// `0` for a forward directed node, `1` for a reverse one (the state index within a vertex).
#[inline]
pub(crate) fn orientation_bit<IndexType: GraphIndexInteger>(
    node: DirectedNodeIndex<IndexType>,
) -> usize {
    if node.is_forward() { 0 } else { 1 }
}

/// Which skeleton a query runs over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The inner skeleton (no `D_out`): pole projections, which must stay inside the subtree.
    Inner = 0,
    /// The LCA-role skeleton (with `D_out`): the LCA search of a query.
    Lca = 1,
}

/// Which representation [`SNodeSegTree::build_with_form`] produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// The ring form, or the tree when the ring form declines (a cost beyond the index word).
    Ring,
    Tree,
}
