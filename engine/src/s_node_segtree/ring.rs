//! The query kernels of the ring form of an S-node oracle (see the module docs of [`super`]; its
//! build is in `ring_build.rs`).

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::{PfxVertex, Role, SegRunRec, tree::query_tree};
use crate::min_plus::{
    network::{Mat4, merge},
    wide::{INF, add, add3, narrow, widen},
    word::inf,
};

/// The pieces of a ring-form oracle and its query kernels (see the module docs). Positions and
/// lane states throughout: `x`/`y` are lane-relative orientations (`0` = cw, `1` = ccw).
pub struct Ring<'a, IndexType: GraphIndexInteger> {
    pub(super) m: usize,
    pub(super) pfx: &'a [PfxVertex<IndexType>],
    pub(super) runs: &'a [SegRunRec],
    pub(super) blocks: usize,
    /// The block tree (`2 · blocks − 1` nodes).
    pub(super) tree: &'a [Mat4<IndexType>],
    /// `[closing_inner, closing_lca]`, closed.
    pub(super) closings: &'a [Mat4<IndexType>],
    /// Two per non-empty run (`[Inner, Lca]`): the `4 × 4` of "the ring minus this run" over the
    /// poles `(v_t, v_s)` of the run `[s, t]`.
    pub(super) summaries: &'a [Mat4<IndexType>],
}

impl<IndexType: GraphIndexInteger> Ring<'_, IndexType> {
    #[inline]
    fn run_of(&self, pos: usize) -> usize {
        (self.pfx[pos].flags.get() >> 1) as usize
    }

    /// The lane-relative orientation of raw orientation `o` at `pos`, and back (an involution).
    #[inline]
    pub(super) fn lane(&self, pos: usize, o: usize) -> usize {
        (self.pfx[pos].flags.get() & 1) as usize ^ o
    }

    #[inline]
    fn start(&self, run: usize) -> usize {
        self.runs[run].start as usize
    }

    #[inline]
    fn end(&self, run: usize) -> usize {
        if run + 1 < self.runs.len() {
            self.runs[run + 1].start as usize - 1
        } else {
            self.m - 1
        }
    }

    #[inline]
    fn p(&self, pos: usize) -> u64 {
        widen(self.pfx[pos].p)
    }

    #[inline]
    fn len(&self, pos: usize) -> u64 {
        widen(self.pfx[pos].len)
    }

    /// The global cheapest flip at `pos` (`dir` `0`: cw -> ccw, `1`: ccw -> cw).
    #[inline]
    pub(super) fn flip(&self, role: Role, pos: usize, dir: usize) -> u64 {
        widen(self.pfx[pos].f[2 * role as usize + dir])
    }

    /// The ccw-lane cost from `(hi, ccw)` down to `(lo, ccw)` inside one run (`lo <= hi`): the
    /// mirror of the cw cost (never negative: every regular leaf's back arc is).
    #[inline]
    fn back(&self, hi: usize, lo: usize) -> u64 {
        self.p(hi) - self.p(lo) + self.len(hi) - self.len(lo)
    }

    /// The product of blocks `[lo, hi)`; `None` when empty.
    #[inline]
    fn range(&self, lo: usize, hi: usize) -> Option<Mat4<IndexType>> {
        if lo >= hi {
            None
        } else {
            query_tree(self.tree, 0, 0, self.blocks, lo, hi)
        }
    }

    /// The cw product of the blocks from the end of run `i` to the start of run `j`, over poles
    /// `(v_{t_i}, v_{s_j})` — through the role's closing when `j` lies before `i`; for `i == j` the
    /// stored "ring minus the run" summary (over `(v_{t_i}, v_{s_i})`).
    fn seg(&self, role: Role, i: usize, j: usize) -> Mat4<IndexType> {
        if i == j {
            let slot = self.runs[i].summary as usize;
            return self.summaries[2 * slot + role as usize];
        }
        let xi = self.runs[i].x_block as usize;
        if i < j {
            return self
                .range(xi, self.runs[j - 1].x_block as usize + 1)
                .expect("the blocks between two runs");
        }
        let mut out = self.closings[role as usize];
        if let Some(after) = self.range(xi, self.blocks) {
            out = merge(&after, &out);
        }
        if j > 0
            && let Some(before) = self.range(0, self.runs[j - 1].x_block as usize + 1)
        {
            out = merge(&out, &before);
        }
        out
    }

    /// The cw route from `(u, x)` to `(v, y)`: flip-or-not at `u`, the cw lane to the end `t` of
    /// `u`'s run, `seg` (over `(v_t, v_s)`, `s` the start of `v`'s run), the cw lane to `v`, flip-or-
    /// not at `v`. `seg = None` is the direct same-run route (`u <= v`). The flips are the global
    /// ones, so every candidate is a walk; the optimal simple walk is one of them.
    fn cw(
        &self,
        role: Role,
        u: usize,
        x: usize,
        v: usize,
        y: usize,
        seg: Option<&Mat4<IndexType>>,
    ) -> u64 {
        let f = |pos: usize, dir: usize| self.flip(role, pos, dir);
        let fu = |a: usize| if a == x { 0 } else { f(u, x) };
        let fv = |b: usize| if b == y { 0 } else { f(v, b) };
        let Some(seg) = seg else {
            if u == v {
                return fu(y);
            }
            return add3(fu(0), self.p(v) - self.p(u), fv(0));
        };
        let (t, s) = (self.end(self.run_of(u)), self.start(self.run_of(v)));
        let dep = if u == t {
            [fu(0), fu(1)]
        } else {
            let d = add(fu(0), self.p(t) - self.p(u));
            [d, add(d, f(t, 0))]
        };
        let mut arr = [INF; 2];
        for b in 0..2 {
            for a in 0..2 {
                arr[b] = arr[b].min(add(
                    dep[a],
                    widen(seg[4 * self.lane(t, a) + 2 + self.lane(s, b)]),
                ));
            }
        }
        if v == s {
            return add(arr[0], fv(0)).min(add(arr[1], fv(1)));
        }
        let at_s = arr[0].min(add(arr[1], f(s, 1)));
        add3(at_s, self.p(v) - self.p(s), fv(0))
    }

    /// The ccw route from `(u, x)` to `(v, y)`: the mirror of [`Self::cw`]. `seg` is the cw product
    /// from the end `t` of `v`'s run to the start `s` of `u`'s run (over `(v_t, v_s)`), read in its
    /// `s -> t` block; `None` is the direct same-run route (`u >= v`).
    fn ccw(
        &self,
        role: Role,
        u: usize,
        x: usize,
        v: usize,
        y: usize,
        seg: Option<&Mat4<IndexType>>,
    ) -> u64 {
        let f = |pos: usize, dir: usize| self.flip(role, pos, dir);
        let fu = |a: usize| if a == x { 0 } else { f(u, x) };
        let fv = |b: usize| if b == y { 0 } else { f(v, b) };
        let Some(seg) = seg else {
            if u == v {
                return fu(y);
            }
            return add3(fu(1), self.back(u, v), fv(1));
        };
        let (s, t) = (self.start(self.run_of(u)), self.end(self.run_of(v)));
        let dep = if u == s {
            [fu(0), fu(1)]
        } else {
            let d = add(fu(1), self.back(u, s));
            [add(d, f(s, 1)), d]
        };
        let mut arr = [INF; 2];
        for b in 0..2 {
            for a in 0..2 {
                arr[b] = arr[b].min(add(
                    dep[a],
                    widen(seg[4 * (2 + self.lane(s, a)) + self.lane(t, b)]),
                ));
            }
        }
        if v == t {
            return add(arr[0], fv(0)).min(add(arr[1], fv(1)));
        }
        let at_t = arr[1].min(add(arr[0], f(t, 0)));
        add3(at_t, self.back(t, v), fv(1))
    }

    /// The segment products the two routes from `u` to `v` need: `(cw route's, ccw route's)`;
    /// `None` = a direct same-run route. The pair for `v` to `u` is the same two, swapped.
    pub(super) fn pair_segs(
        &self,
        role: Role,
        u: usize,
        v: usize,
    ) -> (Option<Mat4<IndexType>>, Option<Mat4<IndexType>>) {
        let (i, j) = (self.run_of(u), self.run_of(v));
        let cw = if i == j && u <= v {
            None
        } else {
            Some(self.seg(role, i, j))
        };
        let ccw = if i == j && u >= v {
            None
        } else {
            Some(self.seg(role, j, i))
        };
        (cw, ccw)
    }

    /// `d` from `(u, x)` to `(v, y)` given the pair's segment products: the better of the two routes.
    #[inline]
    pub(super) fn dist_with(
        &self,
        role: Role,
        u: usize,
        x: usize,
        v: usize,
        y: usize,
        segs: &(Option<Mat4<IndexType>>, Option<Mat4<IndexType>>),
    ) -> u64 {
        self.cw(role, u, x, v, y, segs.0.as_ref())
            .min(self.ccw(role, u, x, v, y, segs.1.as_ref()))
    }

    /// `d` from `(u, x)` to `(v, y)` over the role's skeleton.
    #[cfg(test)]
    pub(super) fn distance(&self, role: Role, u: usize, x: usize, v: usize, y: usize) -> u64 {
        let segs = self.pair_segs(role, u, v);
        self.dist_with(role, u, x, v, y, &segs)
    }

    /// The `4 × 4` over `[(p,+), (p,−), (q,+), (q,−)]` (raw orientations): two segment products,
    /// then sixteen `O(1)` route evaluations.
    pub(super) fn cut_pair(&self, p: usize, q: usize, role: Role) -> Mat4<IndexType> {
        debug_assert_ne!(p, q);
        let segs = self.pair_segs(role, p, q);
        let back = (segs.1, segs.0);
        let states = [(p, 0usize), (p, 1), (q, 0), (q, 1)];
        let mut out = [inf::<IndexType>(); 16];
        for (i, &(pu, ou)) in states.iter().enumerate() {
            for (j, &(pv, ov)) in states.iter().enumerate() {
                let d = if pu == pv {
                    if ou == ov {
                        0
                    } else {
                        self.flip(role, pu, self.lane(pu, ou))
                    }
                } else {
                    let s = if pu == p { &segs } else { &back };
                    self.dist_with(role, pu, self.lane(pu, ou), pv, self.lane(pv, ov), s)
                };
                out[4 * i + j] = narrow(d);
            }
        }
        out
    }
}
