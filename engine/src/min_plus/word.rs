//! Raw min-plus words: a distance as a bare `IndexType` with `IndexType::MAX` as `∞` (the bits of a
//! [`MinPlus`](super::MinPlus)) and saturating addition as `⊗`.
//!
//! `saturating_add` keeps `∞` absorbing without a branch, and it can only trigger on a sum beyond
//! the representable distances, where `∞` is the right answer anyway. The S-node networks
//! ([`super::network`]) are built from these words; this file holds the scalar operations and the
//! all-pairs closure they share.

use bidirected_adjacency_array::index::GraphIndexInteger;

use crate::path::GfaPathLength;

/// The word for `∞` (no walk).
#[inline]
pub fn inf<IndexType: GraphIndexInteger>() -> IndexType {
    IndexType::max_value()
}

/// `a ⊗ b`: the saturating sum.
#[inline]
pub fn sat_add<IndexType: GraphIndexInteger>(a: IndexType, b: IndexType) -> IndexType {
    a.saturating_add(b)
}

/// `cell ⊕= value`: keep the cheaper one.
#[inline]
pub fn relax<IndexType: GraphIndexInteger>(cell: &mut IndexType, value: IndexType) {
    *cell = (*cell).min(value);
}

/// The path length a word holds, `None` for `∞`.
#[inline]
pub fn length<IndexType: GraphIndexInteger>(word: IndexType) -> Option<GfaPathLength<IndexType>> {
    (word != inf()).then(|| GfaPathLength::from_raw(word))
}

/// Elementwise `⊕` of two matrices of the same shape (row-major, `L` entries).
#[inline]
pub fn min_entries<IndexType: GraphIndexInteger, const L: usize>(
    a: &[IndexType; L],
    b: &[IndexType; L],
) -> [IndexType; L] {
    let mut out = *a;
    for (cell, &value) in out.iter_mut().zip(b) {
        relax(cell, value);
    }
    out
}

/// The all-pairs closure of an `N × N` row-major matrix (`L = N²` entries) in place: Floyd–Warshall,
/// intermediates `k` outermost. The diagonal is kept as it is: a zero diagonal (the empty walk)
/// makes the result `d`.
#[inline]
pub fn close_square<IndexType: GraphIndexInteger, const N: usize, const L: usize>(
    m: &mut [IndexType; L],
) {
    const { assert!(N * N == L, "an N x N matrix has N² entries") };
    for k in 0..N {
        for i in 0..N {
            let ik = m[N * i + k];
            if ik == inf() {
                continue;
            }
            for j in 0..N {
                let via = sat_add(ik, m[N * k + j]);
                if via < m[N * i + j] {
                    m[N * i + j] = via;
                }
            }
        }
    }
}
