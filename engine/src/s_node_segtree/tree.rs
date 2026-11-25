//! The segment tree over a cycle's path leaves: the compact `2n − 1` layout, its build and its
//! order-preserving range product.

use bidirected_adjacency_array::index::GraphIndexInteger;

use crate::min_plus::network::{Mat4, merge};

/// Number of stored matrices for a tree over `m` cycle vertices: `2(m−1)−1` tree nodes over the
/// `m−1` path leaves, plus the two closing networks.
#[inline]
pub(super) fn mat_count(m: usize) -> usize {
    2 * (m - 1) - 1 + 2
}

/// The index of the right child of the node covering `[lo, hi)` in the compact layout: the left
/// child owns `2·(mid − lo) − 1` nodes starting at `node + 1`, so the right subtree begins after it.
#[inline]
fn right_child(node: usize, lo: usize, mid: usize) -> usize {
    node + 2 * (mid - lo)
}

/// Recursively build the segment tree over `leaves[lo..hi)` at `node` (compact `2n − 1` layout).
pub(super) fn build_tree<IndexType: GraphIndexInteger>(
    tree: &mut [Mat4<IndexType>],
    leaves: &[Mat4<IndexType>],
    node: usize,
    lo: usize,
    hi: usize,
) {
    if hi - lo == 1 {
        tree[node] = leaves[lo];
        return;
    }
    let mid = (lo + hi) / 2;
    let right = right_child(node, lo, mid);
    build_tree(tree, leaves, node + 1, lo, mid);
    build_tree(tree, leaves, right, mid, hi);
    tree[node] = merge(&tree[node + 1], &tree[right]);
}

/// Order-preserving range product over leaves `[l, r)`; `None` for an empty intersection.
pub(super) fn query_tree<IndexType: GraphIndexInteger>(
    tree: &[Mat4<IndexType>],
    node: usize,
    lo: usize,
    hi: usize,
    l: usize,
    r: usize,
) -> Option<Mat4<IndexType>> {
    if r <= lo || hi <= l {
        return None;
    }
    if l <= lo && hi <= r {
        return Some(tree[node]);
    }
    let mid = (lo + hi) / 2;
    let left = query_tree(tree, node + 1, lo, mid, l, r);
    let right = query_tree(tree, right_child(node, lo, mid), mid, hi, l, r);
    match (left, right) {
        (Some(a), Some(b)) => Some(merge(&a, &b)),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}
