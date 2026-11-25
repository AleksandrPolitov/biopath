//! An exact shortest-path distance index for bidirected sequence graphs.
//!
//! The input is a GFA graph: nodes with sequence lengths, and links that join node ends in either
//! orientation. A query names two positions (a directed node and an offset into it) and gets the
//! length of a shortest walk between them, or none. The index answers it exactly and without a
//! search over the whole graph. [`DistanceIndex`] is the entry point: build, save, load and query.
//!
//! # Build
//!
//! 1. **Shrink** ([`shrink`], on by default in `biopath index`): an exact pre-reduction to a
//!    smaller graph G′ with the same distances, and an alias from every original position onto G′.
//! 2. **SPQR decomposition** ([`decomposition`]): the block-cut tree and the SPQR tree of every
//!    biconnected block, each rooted by [`block_spqr_tree`].
//! 3. **Per-block records** ([`block_record`]): the inside and outside DPs ([`inside_dp`],
//!    [`outside_dp`], [`s_node_outside`]) give every SPQR node its pole distances, the pole lift
//!    ([`pole_lift`]) climbs them up the tree, and a big skeleton gets an oracle instead of a
//!    search: hub labels ([`skeleton_index`]), an S-node segment tree ([`s_node_segtree`]) or a
//!    pole-projection table (the tiers of [`block_engine`]). A small block gets no record.
//! 4. **Block-cut lift** ([`block_cut_lift`]): per cut node its flip closure, per block its
//!    cut-to-cut matrix, the path products along the block-cut forest ([`prefix_lift`],
//!    [`chain_segtree`], laid out by [`chain_nav`]) and every node's cut rows ([`node_rows`]).
//! 5. **Image** ([`image`], its sections in [`sections`]): all of it, with the node topology
//!    ([`topology`]) and the graph ([`graph_view`]), as plain-data arrays in one flat arena. It is
//!    written as a `.bpi` file and memory-mapped back (prefaulted whole), with nothing to parse.
//!
//! # Query
//!
//! [`DistanceIndex::distance`] maps original positions onto G′ (on a shrunk index), answers a walk
//! inside one node directly, takes the in-block distance in every block holding both endpoints
//! (the record's [`block_engine`], or [`block_dijkstra`] in a small block) and the cross-block
//! route through cut nodes from the block-cut lift, and returns the minimum. Distances compose in
//! the min-plus semiring ([`min_plus`]) over orientation-indexed matrices ([`orientation`]).
//!
//! [`dijkstra`] is the whole-graph Dijkstra every answer is tested against; it is no part of the
//! index.
//!
//! [`DistanceIndex`]: distance_index::DistanceIndex
//! [`DistanceIndex::distance`]: distance_index::DistanceIndex::distance

// The index image is built in a large `MAP_NORESERVE` reservation and memory-mapped back with
// Linux `mmap`/`madvise` semantics (`MADV_POPULATE_READ`), so the engine targets 64-bit Linux.
#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
compile_error!(
    "biopath supports Linux only (64-bit): the index image is built and memory-mapped with Linux \
     mmap/madvise and needs a 64-bit address space"
);

// The index: build, save, load and query.
pub mod distance_index;

// Build: the in-process SPQR decomposition, the optional exact pre-reduction, the rooted SPQR tree
// of a block and the per-block DPs. The stages run in `engine_build`; `build_report` logs the
// finished image.
pub mod block_spqr_tree;
mod build_report;
pub mod decomposition;
mod engine_build;
pub mod inside_dp;
pub mod outside_dp;
pub mod s_node_outside;
pub mod shrink;

// Query, inside a block: the block's record and its engine, the pole lift, the skeleton searches
// and the oracles that replace them on big skeletons, and the search of a small block.
pub mod block_dijkstra;
pub mod block_engine;
pub mod block_record;
pub mod pole_lift;
pub mod s_node_segtree;
pub mod skeleton;
pub mod skeleton_index;

// Query, across blocks: the block-cut lift, its two path-product backends, the chain layout of the
// block-cut forest and the per-node cut rows.
pub mod block_cut_lift;
pub mod chain_nav;
pub mod chain_segtree;
pub mod node_rows;
pub mod prefix_lift;

// Image: the flat arena and its loader, the section ids and header knobs, the node topology and
// the graph as the index stores them.
pub mod graph_view;
pub mod image;
pub mod sections;
pub mod topology;

// Structures: the min-plus algebra, orientations, rooted trees with LCA, a fast hasher, and the
// GFA-facing types (the GFA reader, positions, path lengths, node data, word-size limits).
pub mod fast_hash;
pub mod gfa_graph_extensions;
pub mod gfa_reader;
pub mod graph_limits;
pub mod location;
pub mod min_plus;
pub mod orientation;
pub mod path;
pub mod rooted_tree;

// Support: a runtime switch of the build arena, and a GFA writer for vg.
pub mod chopped_gfa;
pub mod env;

// Test oracle (no part of the index): whole-graph Dijkstra with its location sets and walks.
pub mod dijkstra;
