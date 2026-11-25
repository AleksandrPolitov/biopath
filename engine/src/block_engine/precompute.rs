//! The build side of a block's query structures: the inner and LCA-role skeletons, the
//! pole-projection tables and the hub labels.

use bidirected_adjacency_array::{
    index::GraphIndexInteger,
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use rayon::prelude::*;

use super::{projection::projection_table, tiers::Tiers};
use crate::{
    block_spqr_tree::BlockSpqrTree,
    inside_dp::InsideDp,
    min_plus::MinPlus,
    outside_dp::OutsideDp,
    skeleton::{Skeleton, SkeletonAccess},
    skeleton_index::SkeletonLabels,
};

/// Precompute the query-ready CSR skeletons of every SPQR node once.
///
/// Returns `(inner, with_out)`: `inner[local]` is [`InsideDp::build_skeleton_adjacency`] (used
/// for source/target projection searches, which stay within the node's subtree); `with_out[local]`
/// additionally folds the node's own `D_out` edge (used when the node is the LCA, where routes may
/// leave its subtree through the outside). Query-independent and linear in total skeleton size, so
/// caching turns per-query skeleton assembly (the dominant query cost) into a one-time build.
pub fn precompute_skeletons<'a, IndexType, NodeData, EdgeData>(
    inside: &InsideDp<'a, IndexType, NodeData, EdgeData>,
    outside: &OutsideDp<IndexType>,
) -> (Vec<Skeleton<IndexType>>, Vec<Skeleton<IndexType>>)
where
    IndexType: GraphIndexInteger + Send + Sync,
    NodeData: GfaNodeData + Send + Sync,
    EdgeData: GfaEdgeData + Send + Sync,
{
    // Each node's two skeletons depend only on read-only inputs (its own child inside-matrices and
    // its outside matrix), so the whole `node_count`-wide loop is embarrassingly parallel — the
    // giant block's ~1.45M nodes are built across the thread pool. `build_skeleton_adjacency` and
    // `Skeleton::from_adjacency` allocate only thread-local scratch.
    let block_tree = inside.block_tree();
    let node_count = block_tree.node_count();
    let build = |local: usize| {
        let adjacency = inside.build_skeleton_adjacency(local);
        // Pole states first, so the stored LCA role (`LcaView`) finds them without hashing.
        let inner = match block_tree.reference_poles(local) {
            Some((s, t)) => {
                Skeleton::from_adjacency_first(&adjacency, &crate::pole_lift::pole_states(s, t))
            }
            None => Skeleton::from_adjacency(&adjacency),
        };
        let mut with = adjacency;
        inside.add_outside_edges(&mut with, local, outside.outside_matrix(local));
        (inner, Skeleton::from_adjacency(&with))
    };
    // A rayon fan-out is only worth its dispatch cost on a big block. On a block-cut graph of
    // ~10^6 tiny blocks this loop is 2-3 nodes wide and would pay a dispatch per block — the
    // measured symptom is idle workers (`wait_until_cold` + `sched_yield` ~11% of build samples).
    // Gate it like `outside_dp`'s PARALLEL_SOURCE_THRESHOLD and `pole_lift`'s OVERSIZED_CHILDREN.
    const PARALLEL_NODE_THRESHOLD: usize = 64;
    if node_count >= PARALLEL_NODE_THRESHOLD {
        (0..node_count).into_par_iter().map(build).unzip()
    } else {
        (0..node_count).map(build).unzip()
    }
}

/// The pole-projection tables of a block: every node with reference poles and no segment tree whose
/// tier asks for one ([`Tiers::table`]). Aligned with the block's local ids; `None` where no table
/// is built.
pub fn precompute_projections<IndexType: GraphIndexInteger + Send + Sync>(
    block_tree: &BlockSpqrTree<IndexType>,
    inner: &[Skeleton<IndexType>],
    labels: &[Option<Box<SkeletonLabels<IndexType>>>],
    segtrees: &[Option<Box<crate::s_node_segtree::SNodeSegTree<IndexType>>>],
    tiers: &Tiers,
) -> Vec<Option<Vec<MinPlus<IndexType>>>> {
    let node_count = block_tree.node_count();
    if tiers.table_min == 0 {
        return (0..node_count).map(|_| None).collect();
    }
    // Every labeled node gets a table whatever its size: with both its LCA search (labels) and its
    // projection (table) precomputed, the build drops its inner arcs from the record.
    let build = |local: usize| -> Option<Vec<MinPlus<IndexType>>> {
        let (s, t) = block_tree.reference_poles(local)?;
        if segtrees[local].is_some()
            || !tiers.table(inner[local].node_count(), labels[local].is_some())
        {
            return None;
        }
        Some(projection_table(
            &inner[local],
            crate::pole_lift::pole_states(s, t),
        ))
    };
    const PARALLEL_NODE_THRESHOLD: usize = 64;
    if node_count >= PARALLEL_NODE_THRESHOLD {
        (0..node_count).into_par_iter().map(build).collect()
    } else {
        (0..node_count).map(build).collect()
    }
}

/// Precompute a hub-labeling distance oracle for every SPQR node whose LCA-role skeleton is at the
/// label tier ([`Tiers::labels`]); smaller nodes get `None` (they keep the Dijkstra path, which is
/// cheaper than building labels for them).
///
/// Labels are built from the SAME adjacency as `with_out` in [`precompute_skeletons`] (real
/// edges + child matrices + the node's own `D_out`), so a labeled LCA search reproduces exactly
/// the Dijkstra LCA search. The build only assembles the adjacency for nodes above the threshold.
///
/// `prebuilt` carries oracles already constructed during the outside DP
/// ([`OutsideDp::compute_with_labels`]): a node's augmented skeleton as a *parent* is identical to
/// its skeleton as the *LCA*, so those are reused as-is and only the (rare) above-threshold nodes
/// that the outside DP did not label — childless leaves — are built here.
pub fn precompute_labels<
    'a,
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
>(
    inside: &InsideDp<'a, IndexType, NodeData, EdgeData>,
    outside: &OutsideDp<IndexType>,
    lca_skeletons: &[Skeleton<IndexType>],
    tiers: &Tiers,
    mut prebuilt: Vec<Option<Box<SkeletonLabels<IndexType>>>>,
) -> Vec<Option<Box<SkeletonLabels<IndexType>>>> {
    let block_tree = inside.block_tree();
    let node_count = block_tree.node_count();
    prebuilt.resize_with(node_count, || None);
    for local in 0..node_count {
        if prebuilt[local].is_some() {
            continue; // reuse the oracle already built during the outside DP.
        }
        if !tiers.labels(lca_skeletons[local].node_count()) {
            continue; // below threshold: keep the Dijkstra path (stays `None`).
        }
        let with = inside.lca_adjacency(local, outside.outside_matrix(local));
        prebuilt[local] = Some(Box::new(inside.skeleton_labels(&with)));
    }
    prebuilt
}
