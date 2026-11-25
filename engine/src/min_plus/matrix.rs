//! Fixed-size square matrices over the min-plus semiring, [`MinPlusMatrix`].
//!
//! These are used at two sizes in the engine:
//! - `N = 2` ([`Mat2`]): the four orientation combinations of a directed distance between two
//!   segments (the block-cut lift's cut matrices, the flip closures).
//! - `N = 4`: a pole-state matrix over two poles x two orientations (the inside and outside DP, the
//!   pole-lift transitions that lift a query node's pole distances up the SPQR tree).
//!
//! Min-plus matrix multiplication composes distances: if `A[i][k]` is the distance from port `i`
//! to port `k` in one region and `B[k][j]` from port `k` to port `j` in the next, then
//! `(A.mul(B))[i][j] = min_k A[i][k] + B[k][j]` is the distance from `i` to `j` through the shared
//! ports `k`.

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::MinPlus;

/// An `N x N` matrix over the min-plus semiring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct MinPlusMatrix<IndexType: GraphIndexInteger, const N: usize> {
    rows: [[MinPlus<IndexType>; N]; N],
}

/// A `2 x 2` matrix indexed by orientation (`0` = forward `+`, `1` = reverse `-`, as
/// [`Orientation::index`](crate::orientation::Orientation::index)): `m.get(a, b)` is the
/// distance leaving the source in orientation `a` and entering the target in orientation `b`.
///
/// [`MinPlusMatrix::mul`] composes `U -> V` with `V -> W` through a consistent orientation of `V`
/// (the walk leaves `V` in the orientation it entered it in). A flip closure is a `Mat2` from a
/// node to itself: its zero diagonal is the empty walk, its off-diagonal entries the cheapest walks
/// from one orientation to the other.
pub type Mat2<IndexType> = MinPlusMatrix<IndexType, 2>;

impl<IndexType: GraphIndexInteger, const N: usize> MinPlusMatrix<IndexType, N> {
    /// The all-`+∞` matrix (the additive identity, and an absorbing element under multiplication).
    pub fn infinity() -> Self {
        Self {
            rows: [[MinPlus::infinity(); N]; N],
        }
    }

    /// The multiplicative identity: `0` on the diagonal, `+∞` elsewhere.
    pub fn identity() -> Self {
        let mut matrix = Self::infinity();
        for i in 0..N {
            matrix.rows[i][i] = MinPlus::zero();
        }
        matrix
    }

    /// Build a matrix from explicit rows.
    #[cfg(test)]
    pub fn from_rows(rows: [[MinPlus<IndexType>; N]; N]) -> Self {
        Self { rows }
    }

    /// The `N x N` block at block-row `i`, block-column `j` of a dense row-major `dim x dim` matrix:
    /// rows `N i..N i + N`, columns `N j..N j + N` (with `N = 2`, the orientations of the `i`-th and
    /// `j`-th node of a matrix over directed nodes `2 * node + orientation`).
    #[inline]
    pub fn block_of(dense: &[MinPlus<IndexType>], dim: usize, i: usize, j: usize) -> Self {
        let mut block = Self::infinity();
        for a in 0..N {
            for b in 0..N {
                block.rows[a][b] = dense[(N * i + a) * dim + (N * j + b)];
            }
        }
        block
    }

    /// The entry in row `i`, column `j`.
    pub fn get(&self, i: usize, j: usize) -> MinPlus<IndexType> {
        self.rows[i][j]
    }

    /// Set the entry in row `i`, column `j`.
    pub fn set(&mut self, i: usize, j: usize, value: MinPlus<IndexType>) {
        self.rows[i][j] = value;
    }

    /// Take the cheaper alternative in every position (semiring addition of matrices).
    pub fn elementwise_min(&self, other: &Self) -> Self {
        let mut result = Self::infinity();
        for i in 0..N {
            for j in 0..N {
                result.rows[i][j] = self.rows[i][j].oplus(other.rows[i][j]);
            }
        }
        result
    }

    /// Min-plus matrix product: `(self.mul(other))[i][j] = min_k self[i][k] + other[k][j]`.
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Self::infinity();
        for i in 0..N {
            for j in 0..N {
                let mut acc = MinPlus::infinity();
                for k in 0..N {
                    acc.min_assign(self.rows[i][k].otimes(other.rows[k][j]));
                }
                result.rows[i][j] = acc;
            }
        }
        result
    }

    /// Relax entry `(i, j)` toward `value` (min-assign): `self[i][j] = min(self[i][j], value)`.
    pub fn relax(&mut self, i: usize, j: usize, value: MinPlus<IndexType>) {
        self.rows[i][j].min_assign(value);
    }

    /// The min-plus transitive closure (all-pairs shortest paths, [`close_dense`]). `self` holds
    /// the direct arc weights (`+∞` where no arc); the result is the all-pairs distance with the
    /// empty walk (`0` diagonal). On a bond (P-node) the pole-state matrix *is* this closure; the
    /// block-cut lift closes its flip closures with it.
    pub fn closure(&self) -> Self {
        let mut m = *self;
        close_dense(m.rows.as_flattened_mut(), N);
        m
    }

    /// `acc ⊕= vector · self`: `acc[j] = min(acc[j], min_i vector[i] + self[i][j])`; an `∞`
    /// entry of `vector` is skipped.
    #[inline]
    pub fn relax_left_product(
        &self,
        acc: &mut [MinPlus<IndexType>; N],
        vector: [MinPlus<IndexType>; N],
    ) {
        for (i, vector_i) in vector.into_iter().enumerate() {
            if vector_i.is_infinite() {
                continue;
            }
            for (j, slot) in acc.iter_mut().enumerate() {
                slot.min_assign(vector_i.otimes(self.rows[i][j]));
            }
        }
    }

    /// `acc ⊕= self · vector`: `acc[i] = min(acc[i], min_j self[i][j] + vector[j])`; an `∞`
    /// entry of `vector` is skipped.
    #[inline]
    pub fn relax_right_product(
        &self,
        acc: &mut [MinPlus<IndexType>; N],
        vector: [MinPlus<IndexType>; N],
    ) {
        for (j, vector_j) in vector.into_iter().enumerate() {
            if vector_j.is_infinite() {
                continue;
            }
            for (i, slot) in acc.iter_mut().enumerate() {
                slot.min_assign(self.rows[i][j].otimes(vector_j));
            }
        }
    }

    /// Left min-plus vector-matrix product: `out[j] = min_i vector[i] + self[i][j]`.
    ///
    /// Lifts a distance vector indexed by this matrix's rows into one indexed by its columns.
    pub fn left_multiply_vector(
        &self,
        vector: &[MinPlus<IndexType>; N],
    ) -> [MinPlus<IndexType>; N] {
        let mut out = [MinPlus::infinity(); N];
        for (j, out_j) in out.iter_mut().enumerate() {
            let mut acc = MinPlus::infinity();
            for (i, &vector_i) in vector.iter().enumerate() {
                acc.min_assign(vector_i.otimes(self.rows[i][j]));
            }
            *out_j = acc;
        }
        out
    }

    /// The transpose (reverses the direction of every stored distance without reversing complement).
    #[cfg(test)]
    pub fn transpose(&self) -> Self {
        let mut result = Self::infinity();
        for i in 0..N {
            for j in 0..N {
                result.rows[j][i] = self.rows[i][j];
            }
        }
        result
    }
}

/// The all-pairs closure of a dense row-major `dim x dim` matrix in place: the diagonal is clamped
/// to `0` (the empty walk), then Floyd–Warshall with the intermediate `k` outermost (an `∞` entry
/// `[i][k]` skips its row: it cannot improve one).
pub fn close_dense<IndexType: GraphIndexInteger>(dist: &mut [MinPlus<IndexType>], dim: usize) {
    debug_assert_eq!(dist.len(), dim * dim);
    for i in 0..dim {
        dist[i * dim + i].min_assign(MinPlus::zero());
    }
    for k in 0..dim {
        for i in 0..dim {
            let ik = dist[i * dim + k];
            if ik.is_infinite() {
                continue;
            }
            for j in 0..dim {
                let via = ik.otimes(dist[k * dim + j]);
                dist[i * dim + j].min_assign(via);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type M = MinPlus<u32>;

    fn f(x: usize) -> M {
        MinPlus::finite_usize(x)
    }

    fn inf() -> M {
        MinPlus::infinity()
    }

    fn mat2(rows: [[usize; 2]; 2]) -> MinPlusMatrix<u32, 2> {
        MinPlusMatrix::from_rows(rows.map(|row| row.map(f)))
    }

    #[test]
    fn identity_and_infinity_shapes() {
        let id: MinPlusMatrix<u32, 3> = MinPlusMatrix::identity();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(id.get(i, j), M::zero());
                } else {
                    assert!(id.get(i, j).is_infinite());
                }
            }
        }

        let z: MinPlusMatrix<u32, 3> = MinPlusMatrix::infinity();
        for i in 0..3 {
            for j in 0..3 {
                assert!(z.get(i, j).is_infinite());
            }
        }
    }

    #[test]
    fn identity_is_multiplicative_neutral() {
        let a = mat2([[1, 4], [7, 2]]);
        let id = MinPlusMatrix::<u32, 2>::identity();
        assert_eq!(id.mul(&a), a);
        assert_eq!(a.mul(&id), a);
    }

    #[test]
    fn infinity_absorbs_under_multiplication() {
        let a = mat2([[1, 4], [7, 2]]);
        let z = MinPlusMatrix::<u32, 2>::infinity();
        assert_eq!(z.mul(&a), z);
        assert_eq!(a.mul(&z), z);
    }

    #[test]
    fn multiply_matches_hand_computation() {
        // A[i][k] + B[k][j], minimised over k.
        let a = mat2([[1, 3], [5, 0]]);
        let b = mat2([[2, 6], [4, 1]]);
        // (0,0): min(1+2, 3+4)=3 ; (0,1): min(1+6, 3+1)=4
        // (1,0): min(5+2, 0+4)=4 ; (1,1): min(5+6, 0+1)=1
        let expected = mat2([[3, 4], [4, 1]]);
        assert_eq!(a.mul(&b), expected);
    }

    #[test]
    fn multiply_handles_infinite_entries() {
        let a = MinPlusMatrix::<u32, 2>::from_rows([[f(1), inf()], [inf(), f(2)]]);
        let b = MinPlusMatrix::<u32, 2>::from_rows([[inf(), f(3)], [f(4), inf()]]);
        // (0,0): min(1+inf, inf+4)=inf ; (0,1): min(1+3, inf+inf)=4
        // (1,0): min(inf+inf, 2+4)=6 ; (1,1): min(inf+3, 2+inf)=inf
        let result = a.mul(&b);
        assert!(result.get(0, 0).is_infinite());
        assert_eq!(result.get(0, 1), f(4));
        assert_eq!(result.get(1, 0), f(6));
        assert!(result.get(1, 1).is_infinite());
    }

    #[test]
    fn multiplication_is_associative() {
        let a = mat2([[1, 3], [5, 0]]);
        let b = mat2([[2, 6], [4, 1]]);
        let c = mat2([[0, 9], [7, 2]]);
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));

        // Also with infinities present.
        let d = MinPlusMatrix::<u32, 2>::from_rows([[f(1), inf()], [inf(), f(0)]]);
        assert_eq!(a.mul(&d).mul(&b), a.mul(&d.mul(&b)));
    }

    #[test]
    fn transpose_is_an_involution_and_swaps_indices() {
        let a = mat2([[1, 4], [7, 2]]);
        assert_eq!(a.transpose().transpose(), a);
        assert_eq!(a.transpose().get(0, 1), a.get(1, 0));
        assert_eq!(a.transpose().get(1, 0), a.get(0, 1));
    }

    #[test]
    fn left_multiply_vector_lifts_distances() {
        let a = mat2([[1, 3], [5, 0]]);
        let v = [f(2), f(10)];
        // out[0] = min(2+1, 10+5) = 3 ; out[1] = min(2+3, 10+0) = 5.
        assert_eq!(a.left_multiply_vector(&v), [f(3), f(5)]);

        // Identity leaves the vector unchanged.
        let id = MinPlusMatrix::<u32, 2>::identity();
        assert_eq!(id.left_multiply_vector(&v), v);

        // v · (A · B) == (v · A) · B (vector-matrix associativity).
        let b = mat2([[2, 6], [4, 1]]);
        let left = a.mul(&b).left_multiply_vector(&v);
        let right = b.left_multiply_vector(&a.left_multiply_vector(&v));
        assert_eq!(left, right);
    }

    #[test]
    fn relax_products_fold_into_the_accumulator() {
        let a = mat2([[1, 3], [5, 0]]);
        let v = [f(2), inf()];
        // From an all-∞ accumulator, `v · A` is the left product.
        let mut acc = [inf(); 2];
        a.relax_left_product(&mut acc, v);
        assert_eq!(acc, a.left_multiply_vector(&v));
        // A cheaper entry already there is kept: [min(0, 3), min(9, 5)].
        let mut acc = [f(0), f(9)];
        a.relax_left_product(&mut acc, v);
        assert_eq!(acc, [f(0), f(5)]);
        // `A · v` is `v · Aᵀ`.
        let mut right = [inf(); 2];
        a.relax_right_product(&mut right, v);
        assert_eq!(right, a.transpose().left_multiply_vector(&v));
    }

    #[test]
    fn block_of_reads_one_node_pair() {
        // A dense 4 x 4 over two nodes' orientations: entry (r, c) holds 10 r + c.
        let dense: Vec<M> = (0..16).map(|k| f(10 * (k / 4) + k % 4)).collect();
        let block = MinPlusMatrix::<u32, 2>::block_of(&dense, 4, 0, 1);
        assert_eq!(block, mat2([[2, 3], [12, 13]]));
        let block = MinPlusMatrix::<u32, 2>::block_of(&dense, 4, 1, 0);
        assert_eq!(block, mat2([[20, 21], [30, 31]]));
    }

    #[test]
    fn elementwise_min_takes_cheaper() {
        let a = mat2([[1, 9], [3, 4]]);
        let b = mat2([[5, 2], [3, 8]]);
        let expected = mat2([[1, 2], [3, 4]]);
        assert_eq!(a.elementwise_min(&b), expected);

        // `+∞` is the neutral element of elementwise min.
        let z = MinPlusMatrix::<u32, 2>::infinity();
        assert_eq!(a.elementwise_min(&z), a);
    }

    #[test]
    fn closure_is_all_pairs_shortest_paths() {
        // Chain 0 ->(2) 1 ->(3) 2: the closure must compose the two hops (0 -> 2 = 5) and clamp the
        // diagonal to 0, while unreachable pairs stay +∞.
        let mut a = MinPlusMatrix::<u32, 3>::infinity();
        a.set(0, 1, f(2));
        a.set(1, 2, f(3));
        let c = a.closure();
        assert_eq!(c.get(0, 0), f(0));
        assert_eq!(c.get(1, 1), f(0));
        assert_eq!(c.get(0, 1), f(2));
        assert_eq!(c.get(0, 2), f(5)); // 0 -> 1 -> 2
        assert_eq!(c.get(2, 0), inf()); // no path back
        assert_eq!(c.get(1, 0), inf());

        // A cheaper direct arc is kept over the two-hop route.
        let mut b = MinPlusMatrix::<u32, 3>::infinity();
        b.set(0, 1, f(2));
        b.set(1, 2, f(3));
        b.set(0, 2, f(4));
        assert_eq!(b.closure().get(0, 2), f(4));
    }
}
