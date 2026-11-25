//! The distance between two original positions of a shrunk index: inside one super-node by
//! subtraction, else through the pruned chains' ports and the engine on G′.

use std::cell::RefCell;

use super::{
    records::LEVEL_SATURATED,
    view::{OriginalPos, ShrinkView, SuperNode, SuperPos},
};

/// A position on a G′ node: `(node, forward, offset)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShrunkPos {
    pub node: usize,
    pub forward: bool,
    pub offset: u64,
}

/// How the engine answers G′ sub-queries for [`ShrinkView::distance`]: `around = false` is an
/// ordinary query between two different directed nodes; `around = true` asks for the shortest walk
/// from a position back to a position on the SAME directed node that must leave the node first.
pub trait ShrunkOracle {
    fn node_len(&mut self, node: usize) -> u64;
    fn distance(&mut self, source: ShrunkPos, target: ShrunkPos, around: bool) -> Option<u64>;
}

/// A pruned chain on the target's hops (see [`ShrinkView::both_saturated`]): the target-side position
/// in it and the walk from there into the target.
#[derive(Clone, Copy)]
struct TargetHop {
    /// The chain ([`chain_key`]).
    chain: u64,
    along: bool,
    /// [`SuperPos::key`] and the offset of the position.
    key: u64,
    offset: u64,
    /// The length of the walk from the position into the target.
    to_target: u64,
}

thread_local! {
    /// The target hops of [`ShrinkView::both_saturated`], reused by every query of the thread.
    static TARGET_HOPS: RefCell<Vec<TargetHop>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
thread_local! {
    /// Tests: a lower level from which two pruned chains count as saturated, so small graphs send
    /// their queries through [`ShrinkView::both_saturated`] (levels start at 1).
    pub(super) static SATURATED_LEVEL_OVERRIDE: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Tests: the work of this thread's queries so far, `[pruned-chain hops, G′ queries]`, so a test
    /// can assert that it stays linear.
    pub(super) static WORK: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
}

/// One hop through a pruned chain (counted in tests, see `WORK`).
#[inline]
fn count_hop() {
    #[cfg(test)]
    WORK.set({
        let [hops, queries] = WORK.get();
        [hops + 1, queries]
    });
}

/// One G′ query (counted in tests, see `WORK`).
#[inline]
fn count_g_query() {
    #[cfg(test)]
    WORK.set({
        let [hops, queries] = WORK.get();
        [hops, queries + 1]
    });
}

/// The level from which the pruning order of two chains is unknown: [`LEVEL_SATURATED`] (lower in
/// tests, see `SATURATED_LEVEL_OVERRIDE`).
#[inline]
fn saturated_level() -> u32 {
    #[cfg(test)]
    if let Some(level) = SATURATED_LEVEL_OVERRIDE.get() {
        return level.min(LEVEL_SATURATED);
    }
    LEVEL_SATURATED
}

/// A pruned chain as a sort key (`of` must not be a G′ node).
#[inline]
fn chain_key(of: SuperNode) -> u64 {
    match of {
        SuperNode::Single(index) => 2 * index as u64,
        SuperNode::Multi(index) => 2 * index as u64 + 1,
        SuperNode::Kept(_) => unreachable!("a G′ node is not a pruned chain"),
    }
}

/// Keep the smaller of `best` and `candidate`.
#[inline]
fn relax(best: &mut Option<u64>, candidate: Option<u64>) {
    if let Some(c) = candidate {
        *best = Some(best.map_or(c, |b| b.min(c)));
    }
}

impl ShrinkView<'_> {
    /// The shortest-walk distance between two ORIGINAL positions (see the module docs).
    pub fn distance(
        &self,
        source: OriginalPos,
        target: OriginalPos,
        oracle: &mut impl ShrunkOracle,
    ) -> Option<u64> {
        let ps = self.resolve(source, oracle);
        let pt = self.resolve(target, oracle);
        // Same super-node in walk order: the in-node walk is the shortest. A G′ node is one node of
        // G′; a pruned chain was one chain of the graph it was pruned from, whose distances equal the
        // original ones for every position still alive there. Any other walk leaves through the end
        // and comes back through the start: at least (len - s) + t >= t - s.
        if ps.same_super(&pt) && pt.key() >= ps.key() {
            return Some(pt.offset - ps.offset);
        }
        match (ps.of, pt.of) {
            (SuperNode::Kept(s), SuperNode::Kept(t)) => self.on_g_prime(&ps, s, &pt, t, oracle),
            (_, SuperNode::Kept(_)) => self.source_first(&ps, target, oracle),
            (SuperNode::Kept(_), _) => self.target_first(source, &pt, oracle),
            _ => self.both_pruned(&ps, source, &pt, target, oracle),
        }
    }

    /// Both positions lie in pruned chains: resolve the side pruned FIRST. When its chain was pruned
    /// the other position was still alive, so the walk is forced through that chain's end (a source)
    /// or start (a target) — later-pruned chains may have attachments pruned earlier, so their own
    /// ends are not forced from the other side. Chains are ordered by level, under which two chains
    /// of one level are not attached to each other and either order is exact. Two chains with
    /// saturated levels have no known order ([`Self::both_saturated`]).
    fn both_pruned(
        &self,
        ps: &SuperPos,
        source: OriginalPos,
        pt: &SuperPos,
        target: OriginalPos,
        oracle: &mut impl ShrunkOracle,
    ) -> Option<u64> {
        let saturated = saturated_level();
        if ps.level >= saturated && pt.level >= saturated {
            self.both_saturated(ps, source, pt, target, oracle)
        } else if pt.level < ps.level {
            self.target_first(source, pt, oracle)
        } else {
            self.source_first(ps, target, oracle)
        }
    }

    /// Both positions lie in pruned chains of saturated levels, so either may have been pruned first.
    ///
    /// The source's hops `S0 = source, S1, …` (`S(i+1)` the exit port of `Si`'s chain) do not depend
    /// on the target, nor the target's hops `T0 = target, T1, …` (`T(j+1)` the entry port of `Tj`'s
    /// chain) on the source, and leaving `Si`'s chain or entering `Tj`'s costs the same whatever the
    /// other side is. So every interleaving of hops that moves the source to `Si` and the target to
    /// `Tj` costs the same `A(i) + B(j)` (`A`, `B` the walks `S0 → Si` and `Tj → T0`), and the walk is
    /// answered where the hops stop: inside one pruned chain (`Si`, `Tj` on it, the target ahead) or
    /// on G′ once both hop sequences reach it. Each such candidate `A(i) + [Si → Tj] + B(j)` is a
    /// walk, and the forced order of hops (which the levels no longer tell) ends at one of them, so
    /// the minimum over every candidate is exact. Each hop sequence is walked once and the pairs on
    /// one chain are matched through the sorted target hops: `O((I + J) log J)` and at most one
    /// engine query, where trying both orders at every hop recurses `binomial(I + J, I)` times.
    fn both_saturated(
        &self,
        ps: &SuperPos,
        source: OriginalPos,
        pt: &SuperPos,
        target: OriginalPos,
        oracle: &mut impl ShrunkOracle,
    ) -> Option<u64> {
        let (mut best, ends) = TARGET_HOPS.with_borrow_mut(|hops| {
            hops.clear();
            let target_end = self.target_hops(pt, target, hops, oracle);
            hops.sort_unstable_by_key(|h| h.chain);
            let mut best = None;
            let source_end = self.source_hops(ps, source, hops, &mut best, oracle);
            (best, source_end.zip(target_end))
        });
        // Both hop sequences reach G′ (neither stops at a pruned tip's unlinked end).
        if let Some(((s, from_source), (t, to_target))) = ends {
            relax(
                &mut best,
                self.distance(s, t, oracle)
                    .map(|d| from_source + d + to_target),
            );
        }
        best
    }

    /// Hop from target position `target` (resolved: `pt`) until a G′ node, pushing every pruned chain
    /// passed onto `hops`: the G′ position reached and the walk from it into `target`, or `None` at a
    /// pruned tip no walk enters.
    fn target_hops(
        &self,
        pt: &SuperPos,
        target: OriginalPos,
        hops: &mut Vec<TargetHop>,
        oracle: &mut impl ShrunkOracle,
    ) -> Option<(OriginalPos, u64)> {
        let (mut at, mut pos, mut to_target) = (*pt, target, 0);
        while !matches!(at.of, SuperNode::Kept(_)) {
            hops.push(TargetHop {
                chain: chain_key(at.of),
                along: at.along,
                key: at.key(),
                offset: at.offset,
                to_target,
            });
            count_hop();
            pos = self.side_port(&at, false)?;
            to_target += at.offset;
            at = self.resolve(pos, oracle);
        }
        Some((pos, to_target))
    }

    /// Hop from source position `source` (resolved: `ps`) until a G′ node, relaxing `best` with every
    /// walk inside a chain the target hops (`hops`, sorted by chain) also pass, the target ahead: the
    /// G′ position reached and the walk from `source` to it, or `None` at a pruned tip no walk leaves.
    fn source_hops(
        &self,
        ps: &SuperPos,
        source: OriginalPos,
        hops: &[TargetHop],
        best: &mut Option<u64>,
        oracle: &mut impl ShrunkOracle,
    ) -> Option<(OriginalPos, u64)> {
        let (mut at, mut pos, mut from_source) = (*ps, source, 0);
        while !matches!(at.of, SuperNode::Kept(_)) {
            let chain = chain_key(at.of);
            let first = hops.partition_point(|h| h.chain < chain);
            for h in hops[first..].iter().take_while(|h| h.chain == chain) {
                if h.along == at.along && h.key >= at.key() {
                    relax(
                        best,
                        Some(from_source + (h.offset - at.offset) + h.to_target),
                    );
                }
            }
            count_hop();
            pos = self.side_port(&at, true)?;
            from_source += at.chain_len - at.offset;
            at = self.resolve(pos, oracle);
        }
        Some((pos, from_source))
    }

    /// `source` lies in a pruned chain: leave it through its end and continue from its exit.
    fn source_first(
        &self,
        ps: &SuperPos,
        target: OriginalPos,
        oracle: &mut impl ShrunkOracle,
    ) -> Option<u64> {
        count_hop();
        // A pruned tip walked towards its unlinked end: no walk leaves it.
        let next = self.side_port(ps, true)?;
        Some(ps.chain_len - ps.offset + self.distance(next, target, oracle)?)
    }

    /// `target` lies in a pruned chain: reach its start through its entry, then walk in.
    fn target_first(
        &self,
        source: OriginalPos,
        pt: &SuperPos,
        oracle: &mut impl ShrunkOracle,
    ) -> Option<u64> {
        count_hop();
        // A pruned tip walked away from its unlinked end: no walk enters it.
        let prev = self.side_port(pt, false)?;
        Some(self.distance(source, prev, oracle)? + pt.offset)
    }

    /// Both positions lie on G′ nodes (`s` and `t`): an engine query, or a walk around the node when
    /// both lie on the same one and the target is not ahead.
    fn on_g_prime(
        &self,
        ps: &SuperPos,
        s: usize,
        pt: &SuperPos,
        t: usize,
        oracle: &mut impl ShrunkOracle,
    ) -> Option<u64> {
        let source = ShrunkPos {
            node: s,
            forward: ps.along,
            offset: ps.offset,
        };
        let target = ShrunkPos {
            node: t,
            forward: pt.along,
            offset: pt.offset,
        };
        count_g_query();
        oracle.distance(source, target, ps.same_super(pt))
    }
}
