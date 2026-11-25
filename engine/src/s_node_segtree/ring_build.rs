//! The build of the ring form (see the module docs of [`super`]): classify the path leaves into lanes,
//! cut the cycle into runs at the exception leaves, then lay out the block sequence, its tree, the
//! run summaries and every position's global cheapest flips.

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::{
    PfxVertex, Role, SegRunRec,
    tree::{build_tree, query_tree},
};
use crate::{
    image::Padded,
    min_plus::{
        network::{Mat4, close_network, merge, parallel, swap_poles},
        wide::{INF, add, add3, narrow, widen},
        word::inf,
    },
};

/// The build-time pieces of a ring-form oracle.
pub(super) struct RingData<IndexType: GraphIndexInteger> {
    pub(super) pfx: Vec<PfxVertex<IndexType>>,
    pub(super) runs: Vec<SegRunRec>,
    pub(super) blocks: usize,
    pub(super) mats: Vec<Mat4<IndexType>>,
}

/// The ring form of a cycle from its RAW (unclosed) path leaves and closing networks (see the
/// module docs). `lens[e]` is the node length of `v_e`. `None` only when a run length or a flip
/// cost exceeds the index word (the caller then keeps the tree).
pub(super) fn ring_form<IndexType: GraphIndexInteger>(
    m: usize,
    leaves: &[Mat4<IndexType>],
    closing_inner: &Mat4<IndexType>,
    closing_lca: &Mat4<IndexType>,
    lens: &[u64],
) -> Option<RingData<IndexType>> {
    let max = word_max::<IndexType>();
    let lanes = Lanes::classify(m, leaves, lens);
    let runs = Runs::of(m, &lanes);
    if runs.p.iter().any(|&x| x > max) {
        return None; // decline (ring): a run length exceeds the index word
    }
    let ring = RingBuild::new(m, lens, lanes.c, runs, leaves, closing_inner);
    let u_blocks: Vec<Mat4<IndexType>> = (0..ring.run_count()).map(|i| ring.run_block(i)).collect();
    let (blocks, runs, summary_count) = ring.block_sequence(&u_blocks, leaves);
    let b = blocks.len();
    let mut mats = vec![[inf::<IndexType>(); 16]; 2 * b - 1 + 2 + 2 * summary_count as usize];
    let (head, summaries) = mats.split_at_mut(2 * b + 1);
    let (tree, closings) = head.split_at_mut(2 * b - 1);
    build_tree(tree, &blocks, 0, 0, b);
    closings[0] = *closing_inner;
    closings[1] = *closing_lca;
    close_network(&mut closings[0]);
    close_network(&mut closings[1]);
    // Per run and role: the summary of "the ring minus this run" and, from it, the global cheapest
    // flip at every position of the run.
    let mut pfx = ring.pfx_vertices();
    for (i, run) in runs.iter().enumerate() {
        for role in [Role::Inner, Role::Lca] {
            let rest = ring_minus_run(tree, b, &closings[role as usize], &runs, i);
            if run.summary != u32::MAX {
                summaries[2 * run.summary as usize + role as usize] = rest;
            }
            ring.run_flips(i, role, &u_blocks[i], &rest, &mut pfx, max)?;
        }
    }
    Some(RingData {
        pfx,
        runs,
        blocks: b,
        mats,
    })
}

/// The largest value a ring-form entry may hold (one below the index word's infinity).
fn word_max<IndexType: GraphIndexInteger>() -> u64 {
    widen(
        inf::<IndexType>()
            .checked_sub(&IndexType::one())
            .unwrap_or(inf()),
    )
}

/// The lane of a regular leaf `net` over `(v_a, v_b)` (node lengths `len_a`, `len_b`): its pass
/// orientations `(a, b)` and cost. `None` (an exception) unless it passes in exactly one
/// orientation pair and its back arcs are exactly that pair's mirror at the mirrored cost (the
/// reversal identity, so its ccw cost is a prefix difference too).
fn lane_of<IndexType: GraphIndexInteger>(
    net: &Mat4<IndexType>,
    len_a: u64,
    len_b: u64,
) -> Option<(usize, usize, u64)> {
    let inf_i = inf::<IndexType>();
    let mut found: Option<(usize, usize, u64)> = None;
    for o in 0..2 {
        for o2 in 0..2 {
            let cell = net[4 * o + 2 + o2];
            if cell != inf_i {
                if found.is_some() {
                    return None;
                }
                found = Some((o, o2, widen(cell)));
            }
        }
    }
    let (a, b, w) = found?;
    for o2 in 0..2 {
        for o in 0..2 {
            let cell = net[4 * (2 + o2) + o];
            if o2 == 1 - b && o == 1 - a {
                if cell == inf_i || widen(cell) + len_a != w + len_b {
                    return None;
                }
            } else if cell != inf_i {
                return None;
            }
        }
    }
    Some((a, b, w))
}

/// The path leaves classified: a leaf is regular when it has a lane ([`lane_of`]) whose departure
/// orientation agrees with the previous regular leaf's arrival; everything else is an exception.
struct Lanes {
    /// The cw orientation of every position (set by the regular leaves around it).
    c: Vec<usize>,
    regular: Vec<bool>,
    /// The lane cost of every regular leaf.
    w: Vec<u64>,
}

impl Lanes {
    fn classify<IndexType: GraphIndexInteger>(
        m: usize,
        leaves: &[Mat4<IndexType>],
        lens: &[u64],
    ) -> Self {
        let mut lanes = Self {
            c: vec![0usize; m],
            regular: vec![false; m - 1],
            w: vec![0u64; m - 1],
        };
        let mut lane_at: Option<usize> = None;
        for e in 0..m - 1 {
            if let Some((a, b, cost)) = lane_of(&leaves[e], lens[e], lens[e + 1])
                && lane_at.is_none_or(|l| l == a)
            {
                lanes.regular[e] = true;
                lanes.c[e] = a;
                lanes.c[e + 1] = b;
                lanes.w[e] = cost;
                lane_at = Some(b);
                continue;
            }
            lane_at = None;
        }
        lanes
    }
}

/// The runs `[starts[i], ends[i]]`, each ended by exception leaf `ends[i]` (the closing, leaf
/// `m−1`, for the last), and the lane prefix sums, restarted at every run.
struct Runs {
    starts: Vec<usize>,
    ends: Vec<usize>,
    run_of: Vec<usize>,
    p: Vec<u64>,
}

impl Runs {
    fn of(m: usize, lanes: &Lanes) -> Self {
        let ends: Vec<usize> = (0..m - 1)
            .filter(|&e| !lanes.regular[e])
            .chain(std::iter::once(m - 1))
            .collect();
        let k = ends.len();
        let starts: Vec<usize> = std::iter::once(0)
            .chain(ends[..k - 1].iter().map(|&e| e + 1))
            .collect();
        let mut run_of = vec![0usize; m];
        let mut p = vec![0u64; m];
        for i in 0..k {
            run_of[starts[i]..=ends[i]].fill(i);
            for e in starts[i] + 1..=ends[i] {
                p[e] = p[e - 1] + lanes.w[e - 1];
            }
        }
        Self {
            starts,
            ends,
            run_of,
            p,
        }
    }
}

/// The per-position quantities of the ring form: the lanes, the runs, the rungs and the in-run
/// hairpin minima.
struct RingBuild<'l> {
    m: usize,
    lens: &'l [u64],
    c: Vec<usize>,
    runs: Runs,
    /// Rungs per position, `[cw -> ccw, ccw -> cw]` over the inner skeleton: leaf `e` holds `v_e`'s,
    /// the inner closing network `v_{m−1}`'s. `D_out`'s rungs reach the flips through the LCA
    /// closing network inside the run summaries.
    r: Vec<[u64; 2]>,
    /// Hairpin minima per run. A cw -> ccw flip at `e` turning inside the run at `k >= e` costs
    /// `2 (p_k − p_e) + len_k − len_e + r_k[0]` = `A_k − (2 p_e + len_e)`; ccw -> cw turning at
    /// `k <= e` costs `B_k + (2 p_e + len_e)` with `B_k = r_k[1] − 2 p_k − len_k`. `suf_a[e]` is the
    /// least `A_k` from `e` to the end of its run, `pre_b[e]` the least `B_k` from its start to `e`.
    suf_a: Vec<u64>,
    pre_b: Vec<Option<i128>>,
}

impl<'l> RingBuild<'l> {
    fn new<IndexType: GraphIndexInteger>(
        m: usize,
        lens: &'l [u64],
        c: Vec<usize>,
        runs: Runs,
        leaves: &[Mat4<IndexType>],
        closing_inner: &Mat4<IndexType>,
    ) -> Self {
        let r: Vec<[u64; 2]> = (0..m)
            .map(|e| {
                let net = if e < m - 1 { &leaves[e] } else { closing_inner };
                let cw = c[e];
                [widen(net[4 * cw + 1 - cw]), widen(net[4 * (1 - cw) + cw])]
            })
            .collect();
        let mut ring = Self {
            m,
            lens,
            c,
            runs,
            r,
            suf_a: vec![INF; m],
            pre_b: vec![None; m],
        };
        ring.hairpin_minima();
        ring
    }

    fn run_count(&self) -> usize {
        self.runs.starts.len()
    }

    fn base(&self, e: usize) -> u64 {
        2 * self.runs.p[e] + self.lens[e]
    }

    fn hairpin_minima(&mut self) {
        let omin = |x: Option<i128>, y: Option<i128>| match (x, y) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, None) => x,
            (None, y) => y,
        };
        for i in 0..self.run_count() {
            let (s, t) = (self.runs.starts[i], self.runs.ends[i]);
            let mut acc = INF;
            for e in (s..=t).rev() {
                acc = acc.min(add(self.base(e), self.r[e][0]));
                self.suf_a[e] = acc;
            }
            let mut acc = None;
            for e in s..=t {
                let b = (self.r[e][1] != INF).then(|| self.r[e][1] as i128 - self.base(e) as i128);
                acc = omin(acc, b);
                self.pre_b[e] = acc;
            }
        }
    }

    /// The cheapest in-run cw -> ccw turnaround from `e` (`INF` = none).
    fn turn_back(&self, e: usize) -> u64 {
        if self.suf_a[e] == INF {
            INF
        } else {
            self.suf_a[e] - self.base(e)
        }
    }

    /// The cheapest in-run ccw -> cw turnaround from `e` (`INF` = none).
    fn turn_forward(&self, e: usize) -> u64 {
        self.pre_b[e].map_or(INF, |b| (b + self.base(e) as i128) as u64)
    }

    /// Run `i`'s `4 × 4` over `(v_s, v_t)` in raw orientations: the lane passes, the in-run hairpins
    /// at both ends and the turnaround rungs, closed.
    fn run_block<IndexType: GraphIndexInteger>(&self, i: usize) -> Mat4<IndexType> {
        let (s, t) = (self.runs.starts[i], self.runs.ends[i]);
        let (p, lens, zero) = (&self.runs.p, self.lens, IndexType::zero());
        let fi = |lane: usize| self.c[s] ^ lane;
        let se = |lane: usize| 2 + (self.c[t] ^ lane);
        let pass = p[t] - p[s];
        let mut net = [inf::<IndexType>(); 16];
        net[4 * fi(0) + se(0)] = narrow(pass);
        net[4 * se(1) + fi(1)] = narrow(pass + lens[t] - lens[s]);
        if s == t {
            // An empty run: both poles are the same vertex, so identify them (both ways).
            net[4 * se(0) + fi(0)] = zero;
            net[4 * fi(1) + se(1)] = zero;
        }
        net[4 * fi(0) + fi(1)] = narrow(self.turn_back(s));
        net[4 * fi(1) + fi(0)] = narrow(self.r[s][1]);
        net[4 * se(1) + se(0)] = narrow(self.turn_forward(t));
        net[4 * se(0) + se(1)] = narrow(self.r[t][0]);
        close_network(&mut net);
        net
    }

    /// The block sequence `U_0 X_0 U_1 … X_{K−2} U_{K−1}`: every non-empty run's block (an empty run
    /// has one too, but it stays out of the sequence) and each path exception's closed leaf; the
    /// run records; and the number of run summaries (one per non-empty run).
    fn block_sequence<IndexType: GraphIndexInteger>(
        &self,
        u_blocks: &[Mat4<IndexType>],
        leaves: &[Mat4<IndexType>],
    ) -> (Vec<Mat4<IndexType>>, Vec<SegRunRec>, u32) {
        let k = self.run_count();
        let mut blocks: Vec<Mat4<IndexType>> = Vec::with_capacity(2 * k);
        let mut runs: Vec<SegRunRec> = Vec::with_capacity(k);
        let mut summary_count = 0u32;
        debug_assert_eq!(u_blocks.len(), k);
        for (i, u_block) in u_blocks.iter().enumerate() {
            let (s, t) = (self.runs.starts[i], self.runs.ends[i]);
            let summary = if t > s {
                blocks.push(*u_block);
                summary_count += 1;
                summary_count - 1
            } else {
                u32::MAX
            };
            let x_block = blocks.len() as u32;
            if i + 1 < k {
                let mut x = leaves[t];
                close_network(&mut x);
                blocks.push(x);
            }
            runs.push(SegRunRec {
                start: s as u32,
                x_block,
                summary,
            });
        }
        (blocks, runs, summary_count)
    }

    /// The position entries, their flips still unset.
    fn pfx_vertices<IndexType: GraphIndexInteger>(&self) -> Vec<PfxVertex<IndexType>> {
        (0..self.m)
            .map(|e| PfxVertex {
                p: narrow(self.runs.p[e]),
                len: narrow(self.lens[e]),
                flags: Padded::new((self.c[e] as u32) | ((self.runs.run_of[e] as u32) << 1)),
                f: [inf::<IndexType>(); 4],
            })
            .collect()
    }

    /// The global cheapest flips at every position of run `i` in `role`, from the run's block and
    /// `rest`, the ring minus the run (over `(v_t, v_s)`). `None` if a flip exceeds the index word.
    fn run_flips<IndexType: GraphIndexInteger>(
        &self,
        i: usize,
        role: Role,
        u_block: &Mat4<IndexType>,
        rest: &Mat4<IndexType>,
        pfx: &mut [PfxVertex<IndexType>],
        max: u64,
    ) -> Option<()> {
        let (s, t) = (self.runs.starts[i], self.runs.ends[i]);
        // All-pairs over the run's four boundary states `(v_s, ±), (v_t, ±)` in the whole ring:
        // walks alternate run pieces (the run block) and cut-path pieces (`rest`, over
        // `(v_t, v_s)`), so the parallel closure of the two is exact. Over `(v_s, v_t)`.
        let w = parallel(u_block, &swap_poles(rest));
        let ws = |lane: usize| self.c[s] ^ lane;
        let wt = |lane: usize| 2 + (self.c[t] ^ lane);
        for (e, entry) in (s..).zip(&mut pfx[s..=t]) {
            // A run end's flips are entries of `w` (its states are boundary states).
            let (f0, f1) = if e == s {
                (widen(w[4 * ws(0) + ws(1)]), widen(w[4 * ws(1) + ws(0)]))
            } else if e == t {
                (widen(w[4 * wt(0) + wt(1)]), widen(w[4 * wt(1) + wt(0)]))
            } else {
                self.interior_flips(e, s, t, &w)
            };
            if (f0 != INF && f0 > max) || (f1 != INF && f1 > max) {
                return None; // decline (ring): a flip cost exceeds the index word
            }
            entry.f[2 * role as usize] = narrow(f0);
            entry.f[2 * role as usize + 1] = narrow(f1);
        }
        Some(())
    }

    /// The cheapest flips `(cw -> ccw, ccw -> cw)` at an interior position `e` of the run `[s, t]`,
    /// given `w` over its boundary states. A simple closed walk either turns inside the run (the
    /// hairpin minima) or leaves `[s, t]`'s interior at a boundary state and comes back through
    /// one: from `(e, cw)` the first boundary state is `(t, cw)` straight ahead or `(s, ccw)` after
    /// an in-run turnaround; the last before `(e, ccw)` is `(t, ccw)` or `(s, cw)` followed by the
    /// in-run hairpin at `e`. Mirrored for `(e, ccw)` to `(e, cw)`.
    fn interior_flips<IndexType: GraphIndexInteger>(
        &self,
        e: usize,
        s: usize,
        t: usize,
        w: &Mat4<IndexType>,
    ) -> (u64, u64) {
        let (p, lens) = (&self.runs.p, self.lens);
        let ws = |lane: usize| self.c[s] ^ lane;
        let wt = |lane: usize| 2 + (self.c[t] ^ lane);
        let (out_t, back_t) = (p[t] - p[e], p[t] - p[e] + lens[t] - lens[e]);
        let (out_s, back_s) = (p[e] - p[s], p[e] - p[s] + lens[e] - lens[s]);
        let hp0 = self.turn_back(e);
        let hp1 = self.turn_forward(e);
        let mut f0 = hp0;
        for &(ai, ac) in &[(wt(0), out_t), (ws(1), add(hp0, back_s))] {
            for &(bi, bc) in &[(wt(1), back_t), (ws(0), add(out_s, hp0))] {
                f0 = f0.min(add3(ac, widen(w[4 * ai + bi]), bc));
            }
        }
        let mut f1 = hp1;
        for &(ai, ac) in &[(ws(1), back_s), (wt(0), add(hp1, out_t))] {
            for &(bi, bc) in &[(ws(0), out_s), (wt(1), add(back_t, hp1))] {
                f1 = f1.min(add3(ac, widen(w[4 * ai + bi]), bc));
            }
        }
        (f0, f1)
    }
}

/// The ring minus run `i` in one role, over `(v_t, v_s)` of the run: the blocks after it (of the `b`
/// under `tree`), the role's `closing`, the blocks before it.
fn ring_minus_run<IndexType: GraphIndexInteger>(
    tree: &[Mat4<IndexType>],
    b: usize,
    closing: &Mat4<IndexType>,
    runs: &[SegRunRec],
    i: usize,
) -> Mat4<IndexType> {
    let range = |lo: usize, hi: usize| -> Option<Mat4<IndexType>> {
        (lo < hi).then(|| query_tree(tree, 0, 0, b, lo, hi).expect("a non-empty block range"))
    };
    let mut rest = *closing;
    if let Some(after) = range(runs[i].x_block as usize, b) {
        rest = merge(&after, &rest);
    }
    if i > 0
        && let Some(before) = range(0, runs[i - 1].x_block as usize + 1)
    {
        rest = merge(&rest, &before);
    }
    rest
}
