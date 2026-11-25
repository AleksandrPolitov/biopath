//! The RI-halved pruned landmark labeling: the reversal-symmetric degree order of the roots and
//! the pruned backward searches.

use std::{cmp::Reverse, collections::BinaryHeap};

use super::INF;

/// RI-halved label build: run only the `n` backward pruned Dijkstras that populate `l_out`,
/// deriving each root's pivot `l_in[r]` from `l_out[rev(r)]` on the fly. Returns `l_out` sorted by
/// hub. See [`SkeletonLabels::build_bidirected`](super::SkeletonLabels::build_bidirected).
///
/// The processing order is **reversal-symmetric** ([`degree_order`]): reversal pairs
/// `{r, rev(r)}` are ranked by the pair's max total degree and their two orientations emitted
/// consecutively. This makes a hub's rank almost equal to its reverse's rank, so the RI-derived
/// pivot `l_in[r]` (from `l_out[rev(r)]` built so far) contains the same hubs the exact pivot
/// would — the pruning stays as strong as a `2n`-search build. An incomplete pivot can only
/// *under*-prune (keep a redundant label), never return a wrong distance, so correctness holds
/// regardless of order.
pub(super) fn build_labels_ri(
    forward: &Csr,
    reverse: &Csr,
    rev_id: &[u32],
    len: &[u64],
) -> Vec<Vec<(u32, u64)>> {
    let n = forward.node_count();
    let mut l_out = vec![Vec::new(); n];
    let mut search = PrunedDijkstra::new(n);
    for r in degree_order(forward, reverse, rev_id) {
        search.run_ri(reverse, r, &mut l_out, rev_id, len);
    }
    for row in &mut l_out {
        row.sort_unstable_by_key(|&(h, _)| h);
    }
    l_out
}

/// The roots in processing order: the reversal pairs `{id, rev(id)}` (by first appearance) ranked
/// by their max total degree (a stable sort), each emitted as `rev(a) = b` first, then `a`, so
/// `a`'s pivot (`l_out[rev(a) = b]`) already includes `b`'s own search.
fn degree_order(forward: &Csr, reverse: &Csr, rev_id: &[u32]) -> Vec<u32> {
    let n = forward.node_count();
    let mut seen = vec![false; n];
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for id in 0..n as u32 {
        if seen[id as usize] {
            continue;
        }
        let r = rev_id[id as usize];
        seen[id as usize] = true;
        seen[r as usize] = true;
        pairs.push((id, r));
    }
    let deg = |u: u32| forward.degree(u) + reverse.degree(u);
    pairs.sort_by_key(|&(a, b)| Reverse(deg(a).max(deg(b))));
    let mut order = Vec::with_capacity(2 * pairs.len());
    for (a, b) in pairs {
        if a == b {
            order.push(a);
        } else {
            order.push(b);
            order.push(a);
        }
    }
    order
}

/// Reusable scratch for the pruned Dijkstra used during label construction.
struct PrunedDijkstra {
    dist: Vec<u64>,
    stamp: Vec<u32>,
    epoch: u32,
    heap: BinaryHeap<Reverse<(u64, u32)>>,
    /// `refd[h]` = distance stored for hub `h` in the pivot's label (set per search, then cleared).
    refd: Vec<u64>,
    pivot_hubs: Vec<u32>,
}

impl PrunedDijkstra {
    fn new(n: usize) -> Self {
        Self {
            dist: vec![INF; n],
            stamp: vec![0; n],
            epoch: 0,
            heap: BinaryHeap::new(),
            refd: vec![INF; n],
            pivot_hubs: Vec::new(),
        }
    }

    /// RI-halved build search (see [`SkeletonLabels::build_bidirected`](super::SkeletonLabels::build_bidirected)). The pivot `l_in[r]` is
    /// **derived on the fly** from `l_out[rev(r)]` via the reversal identity
    /// `D(h -> r) = D(r_bar -> h_bar) + len(h) - len(r)`, so only `l_out` is stored and only the `n`
    /// backward searches run. `l_out` is both the pivot source
    /// (`l_out[rev(r)]`, read into `refd` before the search) and the write target.
    fn run_ri(
        &mut self,
        graph: &Csr,
        r: u32,
        l_out: &mut [Vec<(u32, u64)>],
        rev_id: &[u32],
        len: &[u64],
    ) {
        self.pivot_hubs.clear();
        let lr = len[r as usize];
        let rev_r = rev_id[r as usize] as usize;
        // Read the derived pivot out of l_out[rev(r)] into refd BEFORE search_core mutates l_out.
        for &(hbar, d) in &l_out[rev_r] {
            let h = rev_id[hbar as usize];
            // `d + len[h] - lr` is D(h->r); the RI guarantees it is >= 0 (D(h->r) is a real distance).
            self.refd[h as usize] = d + len[h as usize] - lr;
            self.pivot_hubs.push(h);
        }
        self.search_core(graph, r, l_out);
    }

    /// The shared pruned-Dijkstra core: assumes the pivot is already loaded into `self.refd` /
    /// `self.pivot_hubs`, runs the search from `r` over `graph`, appends `(r, dist)` to
    /// `target_labels[u]` for every un-covered settled `u`, and clears `refd` on exit.
    fn search_core(&mut self, graph: &Csr, r: u32, target_labels: &mut [Vec<(u32, u64)>]) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
        self.heap.clear();
        self.dist[r as usize] = 0;
        self.stamp[r as usize] = self.epoch;
        self.heap.push(Reverse((0, r)));

        while let Some(Reverse((d, u))) = self.heap.pop() {
            if self.stamp[u as usize] == self.epoch && d > self.dist[u as usize] {
                continue;
            }
            // Prune: if the existing 2-hop cover already reaches u within d, r is redundant for u.
            let covered = target_labels[u as usize].iter().any(|&(h, dh)| {
                let rd = self.refd[h as usize];
                rd != INF && rd.saturating_add(dh) <= d
            });
            if covered {
                continue; // do not label u, do not expand past it
            }
            target_labels[u as usize].push((r, d));
            for (v, w) in graph.neighbors(u) {
                let nd = d + w;
                let vu = v as usize;
                if self.stamp[vu] != self.epoch || nd < self.dist[vu] {
                    self.stamp[vu] = self.epoch;
                    self.dist[vu] = nd;
                    self.heap.push(Reverse((nd, v)));
                }
            }
        }

        for &h in &self.pivot_hubs {
            self.refd[h as usize] = INF;
        }
    }
}

/// A compressed-sparse-row directed graph with `u64` weights, over dense ids `0..n`.
pub(super) struct Csr {
    off: Vec<u32>,
    to: Vec<u32>,
    w: Vec<u64>,
}

impl Csr {
    pub(super) fn build(n: usize, edges: impl Iterator<Item = (u32, u32, u64)> + Clone) -> Self {
        let mut off = vec![0u32; n + 1];
        for (u, _, _) in edges.clone() {
            off[u as usize + 1] += 1;
        }
        for i in 0..n {
            off[i + 1] += off[i];
        }
        let total = *off.last().unwrap() as usize;
        let mut to = vec![0u32; total];
        let mut w = vec![0u64; total];
        let mut cur = off.clone();
        for (u, v, ww) in edges {
            let slot = cur[u as usize] as usize;
            to[slot] = v;
            w[slot] = ww;
            cur[u as usize] += 1;
        }
        Self { off, to, w }
    }

    pub(super) fn node_count(&self) -> usize {
        self.off.len() - 1
    }

    pub(super) fn degree(&self, u: u32) -> u32 {
        self.off[u as usize + 1] - self.off[u as usize]
    }

    #[inline]
    pub(super) fn neighbors(&self, u: u32) -> impl Iterator<Item = (u32, u64)> + '_ {
        let a = self.off[u as usize] as usize;
        let b = self.off[u as usize + 1] as usize;
        (a..b).map(move |i| (self.to[i], self.w[i]))
    }
}
