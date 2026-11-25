//! The block gadget: a block's raw cut-to-cut matrix `M_B` over its `2c` directed cuts (its
//! poles) with flip closures as `2x2` self-loops at some of its cuts. Its all-pairs closure
//! ([`gadget_closure`]) is the block's `M_aug`; the flip passes read only one cut's `2x2` diagonal
//! of it ([`GadgetDijkstra::diagonal`]).

use bidirected_adjacency_array::index::GraphIndexInteger;

use crate::min_plus::{Mat2, MinPlus, close_dense};

/// The two-source dense Dijkstra of [`GadgetDijkstra::diagonal`], with its buffers kept between
/// calls (the top-down pass keeps one per worker).
pub(super) struct GadgetDijkstra<IndexType: GraphIndexInteger> {
    /// Tentative distances from the current source pole (`dim` entries).
    dist: Vec<MinPlus<IndexType>>,
    /// Settled flags for the linear-scan dense Dijkstra (`dim` entries).
    settled: Vec<bool>,
    /// `flip_w[p]` = weight of the extra flip edge `p -> (p ^ 1)` contributed by the folded self-
    /// loops (`+∞` where none). A folded `2x2` loop at local cut `l` contributes only its two
    /// off-diagonal entries as edges `2l -> 2l+1` and `2l+1 -> 2l`; the diagonal entries are `>= 0`
    /// self-edges (every loop here is a closed flip closure, `0` on the diagonal) that can never
    /// improve a nonnegative shortest path, so they are inert and dropped.
    flip_w: Vec<MinPlus<IndexType>>,
}

impl<IndexType: GraphIndexInteger> GadgetDijkstra<IndexType> {
    pub(super) fn new() -> Self {
        Self {
            dist: Vec::new(),
            settled: Vec::new(),
            flip_w: Vec::new(),
        }
    }

    /// The `2x2` diagonal block of the gadget `raw` (row-major dense cut-to-cut distances over `dim`
    /// poles) with the self-`loops` folded in, at local cut `source_local`.
    ///
    /// Equal to the `source_local` diagonal of [`gadget_closure`]`(raw, loops)`, computed by two
    /// `O(dim^2)` dense Dijkstra runs from the cut's two poles instead of the `O(dim^3)` closure.
    /// Every gadget edge weight is a nonnegative distance (`raw` is an intra-block cut-to-cut
    /// distance matrix; the folded loops are closed flip closures), so Dijkstra is exact.
    ///
    /// The leave-one-out is the caller's: `loops` must already exclude `source_local`'s own loop.
    pub(super) fn diagonal(
        &mut self,
        raw: &[MinPlus<IndexType>],
        dim: usize,
        loops: &[(usize, Mat2<IndexType>)],
        source_local: usize,
    ) -> Mat2<IndexType> {
        debug_assert_eq!(dim * dim, raw.len());
        self.flip_w.clear();
        self.flip_w.resize(dim, MinPlus::infinity());
        for &(local, matrix) in loops {
            self.flip_w[2 * local].min_assign(matrix.get(0, 1));
            self.flip_w[2 * local + 1].min_assign(matrix.get(1, 0));
        }
        let mut result = Mat2::infinity();
        for oi in 0..2 {
            self.dijkstra(raw, dim, 2 * source_local + oi);
            for oj in 0..2 {
                result.set(oi, oj, self.dist[2 * source_local + oj]);
            }
        }
        result
    }

    /// Dense `O(dim^2)` Dijkstra from pole `source` over `raw` (complete, row-major) plus the sparse
    /// `flip_w` edges; leaves the shortest-path distances in `self.dist`. A linear-scan min beats a
    /// binary heap here because `raw` is a complete graph (every settle relaxes all `dim` poles).
    fn dijkstra(&mut self, raw: &[MinPlus<IndexType>], dim: usize, source: usize) {
        self.dist.clear();
        self.dist.resize(dim, MinPlus::infinity());
        self.settled.clear();
        self.settled.resize(dim, false);
        self.dist[source] = MinPlus::zero();
        for _ in 0..dim {
            let mut u = usize::MAX;
            let mut best = MinPlus::infinity();
            for v in 0..dim {
                if !self.settled[v] && self.dist[v] < best {
                    best = self.dist[v];
                    u = v;
                }
            }
            if u == usize::MAX {
                break; // all remaining poles unreachable
            }
            self.settled[u] = true;
            let du = self.dist[u];
            let row = &raw[u * dim..(u + 1) * dim];
            for (v, &weight) in row.iter().enumerate() {
                self.dist[v].min_assign(du.otimes(weight));
            }
            let fw = self.flip_w[u];
            if !fw.is_infinite() {
                self.dist[u ^ 1].min_assign(du.otimes(fw));
            }
        }
    }
}

/// The all-pairs closure of a block gadget: the raw cut-to-cut distances `raw` (row-major over
/// `dim = 2c` poles) with the `2x2` self-`loops` at their local cuts, closed through every pole as
/// an intermediate (`O(dim^3)`). Returns the full `dim x dim` matrix.
pub(super) fn gadget_closure<IndexType: GraphIndexInteger>(
    raw: &[MinPlus<IndexType>],
    loops: &[(usize, Mat2<IndexType>)],
) -> Vec<MinPlus<IndexType>> {
    let dim = (raw.len() as f64).sqrt().round() as usize;
    debug_assert_eq!(dim * dim, raw.len());
    let mut dist = raw.to_vec();
    for &(local, matrix) in loops {
        for oi in 0..2 {
            for oj in 0..2 {
                dist[(2 * local + oi) * dim + (2 * local + oj)].min_assign(matrix.get(oi, oj));
            }
        }
    }
    close_dense(&mut dist, dim);
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    type M = MinPlus<u32>;

    /// A tiny deterministic LCG so the fuzz is reproducible without an rng dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        /// A random min-plus weight: `+∞` with probability ~1/8, else a small nonnegative length.
        fn weight(&mut self, max: usize) -> M {
            if self.next().is_multiple_of(8) {
                MinPlus::infinity()
            } else {
                MinPlus::finite_usize((self.next() as usize) % (max + 1))
            }
        }
    }

    /// The reference: the diagonal of the full closure.
    fn brute(
        raw: &[M],
        dim: usize,
        loops: &[(usize, Mat2<u32>)],
        source_local: usize,
    ) -> Mat2<u32> {
        Mat2::block_of(&gadget_closure(raw, loops), dim, source_local, source_local)
    }

    /// `GadgetDijkstra::diagonal` must be byte-identical to `gadget_closure` + `extract` for every
    /// source cut, over many random dense gadgets (varying dim, `+∞` entries, and folded flip loops),
    /// each with the source's own loop left out (the leave-one-out the caller performs).
    #[test]
    fn diagonal_matches_gadget_closure_fuzz() {
        let mut rng = Lcg(0x0BAD_C0FFE_u64);
        let mut gadget = GadgetDijkstra::new();
        for cuts in [1usize, 2, 3, 5, 8, 13] {
            let dim = 2 * cuts;
            for _ in 0..40 {
                // Random dense cut-to-cut matrix with a zero diagonal (like the real M_B).
                let mut raw = vec![MinPlus::infinity(); dim * dim];
                for i in 0..dim {
                    for j in 0..dim {
                        raw[i * dim + j] = if i == j {
                            MinPlus::zero()
                        } else {
                            rng.weight(20)
                        };
                    }
                }
                // Random closed flip loops at a subset of cuts.
                let mut all_loops: Vec<(usize, Mat2<u32>)> = Vec::new();
                for local in 0..cuts {
                    if rng.next().is_multiple_of(2) {
                        let m = Mat2::from_rows([
                            [MinPlus::zero(), rng.weight(15)],
                            [rng.weight(15), MinPlus::zero()],
                        ]);
                        all_loops.push((local, m.closure()));
                    }
                }
                for source_local in 0..cuts {
                    // Leave the source's own loop out, exactly as the top-down pass does.
                    let loops: Vec<_> = all_loops
                        .iter()
                        .copied()
                        .filter(|&(local, _)| local != source_local)
                        .collect();
                    let fast = gadget.diagonal(&raw, dim, &loops, source_local);
                    let slow = brute(&raw, dim, &loops, source_local);
                    assert_eq!(
                        fast, slow,
                        "diagonal mismatch: cuts={cuts} source={source_local} loops={loops:?}",
                    );
                }
            }
        }
    }
}
