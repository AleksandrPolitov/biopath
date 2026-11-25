//! The chains as a union-find over nodes.

use super::side_graph::{SideGraph, side, side_node};

/// The chains as a union-find over nodes: each root holds the chain's two outer sides (`a`, where a
/// walk from that end enters it, and `b`, where the walk leaves it) and its length.
pub(super) struct Chains {
    parent: Vec<u32>,
    pub(super) a: Vec<u32>,
    pub(super) b: Vec<u32>,
    pub(super) len: Vec<u64>,
    pub(super) cyclic: Vec<bool>,
}

impl Chains {
    /// Every node its own chain, entered at its left side.
    pub(super) fn new(lens: &[u64]) -> Self {
        let n = lens.len();
        Self {
            parent: (0..n as u32).collect(),
            a: (0..n).map(|v| side(v, false) as u32).collect(),
            b: (0..n).map(|v| side(v, true) as u32).collect(),
            len: lens.to_vec(),
            cyclic: vec![false; n],
        }
    }

    pub(super) fn find(&mut self, mut v: usize) -> usize {
        while self.parent[v] as usize != v {
            let grand = self.parent[self.parent[v] as usize];
            self.parent[v] = grand;
            v = grand as usize;
        }
        v
    }

    /// The outer sides `(a, b)` of the chain rooted at `root`.
    pub(super) fn ends(&self, root: usize) -> (usize, usize) {
        (self.a[root] as usize, self.b[root] as usize)
    }

    /// Join the chains ending at sides `s` and `t`, which a link joins; returns the new root. Joining
    /// a chain's two ends closes a cycle; a link already inside one chain (both of its sides were
    /// re-examined after a prune) changes nothing.
    pub(super) fn join(&mut self, s: usize, t: usize) -> usize {
        let (x, y) = (self.find(side_node(s)), self.find(side_node(t)));
        if x == y {
            let ends = self.ends(x);
            if ends == (s, t) || ends == (t, s) {
                self.cyclic[x] = true;
            }
            return x;
        }
        let other = |c: &Chains, r: usize, end: usize| {
            if c.a[r] as usize == end {
                c.b[r]
            } else {
                debug_assert_eq!(c.b[r] as usize, end, "joined side is not a chain end");
                c.a[r]
            }
        };
        let (na, nb) = (other(self, x, s), other(self, y, t));
        let (root, child) = if x < y { (x, y) } else { (y, x) };
        self.parent[child] = root as u32;
        self.a[root] = na;
        self.b[root] = nb;
        self.len[root] = self.len[x] + self.len[y];
        root
    }

    /// Whether `r` is the root of an alive open chain that may be pruned (not cyclic, not of a
    /// zero-length node).
    pub(super) fn is_open_root(&mut self, sg: &SideGraph, r: usize) -> bool {
        sg.alive_node[r] && self.find(r) == r && !self.cyclic[r] && !sg.blocked[r]
    }

    /// The side across the single link at `end` of the chain rooted at `root`, when it lies outside
    /// the chain.
    pub(super) fn partner(&mut self, sg: &SideGraph, root: usize, end: usize) -> Option<usize> {
        if sg.deg[end] != 1 {
            return None;
        }
        let t = sg.only_link(end).1;
        (self.find(side_node(t)) != root).then_some(t)
    }

    /// The sides `(p, q)` an allele rooted at `root` attaches to through its ends `(a, b)`: an open
    /// chain whose outer sides each carry one link, to a side outside it.
    pub(super) fn allele_partners(
        &mut self,
        sg: &SideGraph,
        root: usize,
    ) -> Option<(usize, usize)> {
        if !self.is_open_root(sg, root) {
            return None;
        }
        let (a, b) = self.ends(root);
        let p = self.partner(sg, root, a)?;
        let q = self.partner(sg, root, b)?;
        Some((p, q))
    }

    /// The side a tip rooted at `root` attaches to, as `(entry, exit)` with `None` at its unlinked
    /// end: an open chain with one outer side unlinked and the other carrying one link, to a side
    /// outside it. No walk passes through a tip, so it lies on no shortest walk between two
    /// positions outside it.
    pub(super) fn tip_partner(
        &mut self,
        sg: &SideGraph,
        root: usize,
    ) -> Option<(Option<usize>, Option<usize>)> {
        if !self.is_open_root(sg, root) {
            return None;
        }
        let (a, b) = self.ends(root);
        match (sg.deg[a], sg.deg[b]) {
            (0, 1) => Some((None, Some(self.partner(sg, root, b)?))),
            (1, 0) => Some((Some(self.partner(sg, root, a)?), None)),
            _ => None,
        }
    }
}
