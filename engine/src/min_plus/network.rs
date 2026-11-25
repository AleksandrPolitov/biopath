//! `4 × 4` min-plus matrices over the boundary states `[(a,+), (a,−), (b,+), (b,−)]` of a
//! two-pole network, in raw words ([`super::word`]), and their series and parallel composition.
//!
//! A segment of an S-node ring (see [`crate::s_node_segtree`]) or a child of an S-node
//! (see [`crate::s_node_outside`]) is summarised by one such matrix holding `d` (empty walk
//! allowed, zero diagonal, [`close_network`]). [`merge`] composes two networks that share a pole
//! (series) by eliminating the shared vertex's two states in closed form; [`parallel`] puts two
//! networks over the same poles side by side; [`close4`] is the plain all-pairs closure.
//!
//! The entries hold the same domain as every other stored distance, `GfaPathLength<IndexType>`, at
//! `IndexType` width rather than a fixed `u64`: that **halves** the segment trees at the default
//! `--word-size 32`, whose bulk is the `4 × 4` matrices.

use bidirected_adjacency_array::index::GraphIndexInteger;

#[cfg(test)]
use super::word::relax;
use super::word::{close_square, inf, min_entries, sat_add};

/// A `4 × 4` min-plus matrix over the boundary states `[(a,+), (a,−), (b,+), (b,−)]`, row-major.
pub type Mat4<IndexType> = [IndexType; 16];

/// A `2 × 2` block of a [`Mat4`] (one pole pair's states), row-major.
type Block2<IndexType> = [IndexType; 4];

/// All-pairs closure of a `4 × 4` (Floyd–Warshall). Zero diagonal ⇒ `d`, empty walk allowed.
pub fn close4<IndexType: GraphIndexInteger>(m: &mut Mat4<IndexType>) {
    close_square::<IndexType, 4, 16>(m);
}

/// Close a network: zero diagonal (the empty walk), then the all-pairs closure.
pub fn close_network<IndexType: GraphIndexInteger>(net: &mut Mat4<IndexType>) {
    for d in 0..4 {
        net[5 * d] = IndexType::zero();
    }
    close4(net);
}

/// Parallel composition: two networks over the same poles side by side (elementwise min), closed,
/// since a walk may alternate between them.
#[inline]
pub fn parallel<IndexType: GraphIndexInteger>(
    a: &Mat4<IndexType>,
    b: &Mat4<IndexType>,
) -> Mat4<IndexType> {
    let mut out = min_entries(a, b);
    close4(&mut out);
    out
}

#[inline]
fn mul2<IndexType: GraphIndexInteger>(
    x: &Block2<IndexType>,
    y: &Block2<IndexType>,
) -> Block2<IndexType> {
    [
        sat_add(x[0], y[0]).min(sat_add(x[1], y[2])),
        sat_add(x[0], y[1]).min(sat_add(x[1], y[3])),
        sat_add(x[2], y[0]).min(sat_add(x[3], y[2])),
        sat_add(x[2], y[1]).min(sat_add(x[3], y[3])),
    ]
}

/// The reflexive–transitive closure `M*` of a `2 × 2` block (walks of length ≥ 0).
#[inline]
fn star2<IndexType: GraphIndexInteger>(m: &Block2<IndexType>) -> Block2<IndexType> {
    let zero = IndexType::zero();
    let mut a = [m[0].min(zero), m[1], m[2], m[3].min(zero)];
    close_square::<IndexType, 2, 4>(&mut a);
    a
}

/// The `2 × 2` block at block-row `r`, block-column `c` of a `Mat4`.
#[inline]
fn blk<IndexType: GraphIndexInteger>(m: &Mat4<IndexType>, r: usize, c: usize) -> Block2<IndexType> {
    let (r0, c0) = (2 * r, 2 * c);
    [
        m[4 * r0 + c0],
        m[4 * r0 + c0 + 1],
        m[4 * (r0 + 1) + c0],
        m[4 * (r0 + 1) + c0 + 1],
    ]
}

/// Series composition: `left` over poles `(a, mid)` and `right` over `(mid, b)` become one network
/// over `(a, b)`, eliminating the shared middle vertex's two states.
///
/// Both inputs are already closed all-pairs matrices, so `L_am` (say) is the true shortest `a → mid`
/// walk inside `left` — excursions within a part need no separate term. What remains is the seam:
/// a walk may cross `mid` any number of times, bouncing between the two parts, which is exactly the
/// closure `M* = (L_mm ⊕ R_mm)*`. Eliminating a 2-state interface in closed form costs ~7 `2 × 2`
/// products instead of the 216 min-plus steps of a `6 × 6` Floyd–Warshall — and this is the hot
/// operation for both build (`m` merges) and query (`O(log m)` merges).
pub fn merge<IndexType: GraphIndexInteger>(
    left: &Mat4<IndexType>,
    right: &Mat4<IndexType>,
) -> Mat4<IndexType> {
    let (l_aa, l_am, l_ma, l_mm) = (
        blk(left, 0, 0),
        blk(left, 0, 1),
        blk(left, 1, 0),
        blk(left, 1, 1),
    );
    let (r_mm, r_mb, r_bm, r_bb) = (
        blk(right, 0, 0),
        blk(right, 0, 1),
        blk(right, 1, 0),
        blk(right, 1, 1),
    );

    let mstar = star2(&min_entries(&l_mm, &r_mm));
    let am_star = mul2(&l_am, &mstar); // a -> mid, then bounce freely
    let bm_star = mul2(&r_bm, &mstar); // b -> mid, then bounce freely

    let aa = min_entries(&l_aa, &mul2(&am_star, &l_ma));
    let ab = mul2(&am_star, &r_mb);
    let ba = mul2(&bm_star, &l_ma);
    let bb = min_entries(&r_bb, &mul2(&bm_star, &r_mb));

    let mut out = [inf::<IndexType>(); 16];
    for (bi, block) in [[aa, ab], [ba, bb]].iter().enumerate() {
        for (bj, m) in block.iter().enumerate() {
            for i in 0..2 {
                for j in 0..2 {
                    out[4 * (2 * bi + i) + 2 * bj + j] = m[2 * i + j];
                }
            }
        }
    }
    out
}

/// The reference series composition: a plain `6 × 6` Floyd–Warshall over `[a, mid, b]`. Obviously
/// correct but ~5× the work of [`merge`]; kept as the differential oracle for it.
#[cfg(test)]
fn merge_reference<IndexType: GraphIndexInteger>(
    left: &Mat4<IndexType>,
    right: &Mat4<IndexType>,
) -> Mat4<IndexType> {
    // States 0,1 = a; 2,3 = mid; 4,5 = b.
    let mut g = [inf::<IndexType>(); 36];
    for i in 0..6 {
        g[6 * i + i] = IndexType::zero();
    }
    for i in 0..4 {
        for j in 0..4 {
            relax(&mut g[6 * i + j], left[4 * i + j]);
            relax(&mut g[6 * (i + 2) + j + 2], right[4 * i + j]);
        }
    }
    close_square::<IndexType, 6, 36>(&mut g);
    let keep = [0usize, 1, 4, 5];
    let mut out = [inf::<IndexType>(); 16];
    for (oi, &i) in keep.iter().enumerate() {
        for (oj, &j) in keep.iter().enumerate() {
            out[4 * oi + oj] = g[6 * i + j];
        }
    }
    out
}

/// Reinterpret a matrix over poles `(a, b)` as one over `(b, a)` (swap the two 2×2 blocks).
pub fn swap_poles<IndexType: GraphIndexInteger>(m: &Mat4<IndexType>) -> Mat4<IndexType> {
    let map = |i: usize| if i < 2 { i + 2 } else { i - 2 };
    let mut out = [inf::<IndexType>(); 16];
    for i in 0..4 {
        for j in 0..4 {
            out[4 * map(i) + map(j)] = m[4 * i + j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed-form 2-state elimination must agree with the plain `6 × 6` Floyd–Warshall on every
    /// input, including `∞` patterns and asymmetric blocks.
    #[test]
    fn fast_merge_matches_reference() {
        // A cheap deterministic LCG: no dependency, and a failure is reproducible.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..4000 {
            let mut draw = || {
                let r = next() % 10;
                // A third of entries are ∞, so unreachable patterns are exercised heavily.
                if r < 3 {
                    inf::<u32>()
                } else {
                    (next() % 50) as u32
                }
            };
            let mut left = [inf::<u32>(); 16];
            let mut right = [inf::<u32>(); 16];
            for i in 0..16 {
                left[i] = draw();
                right[i] = draw();
            }
            // Real segment matrices hold `d` with the empty walk allowed, so the diagonal is 0 and the
            // matrix is closed. Feeding a non-zero diagonal would compare two different conventions
            // (the reference forces `g[i][i] = 0`, the closed form inherits the input's).
            for d in 0..4 {
                left[4 * d + d] = 0;
                right[4 * d + d] = 0;
            }
            close4(&mut left);
            close4(&mut right);
            let fast = merge(&left, &right);
            let reference = merge_reference(&left, &right);
            assert_eq!(fast, reference, "merge mismatch in round {round}");
            // The zero diagonal must be preserved, or a later merge would compare conventions.
            for d in 0..4 {
                assert_eq!(
                    fast[4 * d + d],
                    0,
                    "merge lost the zero diagonal in round {round}"
                );
            }
        }
    }
}
