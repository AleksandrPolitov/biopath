//! Building the alias records (`records.rs`) while the reducer still has the member lists. A pruned
//! chain is recorded when it is pruned. Runs are consecutive in the id space of the non-single
//! nodes, so the runs of the G′ nodes and of the multi-node pruned chains are cut once every
//! single-node chain is known ([`RunLayout::seal_singles`]); [`RunLayout::finish`] then encodes the
//! lengths and builds the rank sets.

use std::collections::HashMap;

use super::{
    OFF_MASK,
    records::{
        AliasBlock, FarPorts, LEN_CODE_ESCAPE, LEVEL_SATURATED, LenBlock, LongLen, NO_PORT,
        P1_DELTA_ESCAPE, P1_FAR, P1_NO_ENTRY, P1_NO_EXIT, P1Rec, PREFIX_SAMPLE, PrunedChain,
        RUN_DESC, RUN_ID_MASK, RUN_PRUNED, RUN_REV, RunRec,
    },
    side_graph::{ChainMember, reverse_dir},
};

thread_local! {
    /// Test-only override of the largest port delta a single-node record stores inline (smaller
    /// values send more records through `SHRINK_FAR_PORTS`, which small test graphs never reach).
    static P1_DELTA_LIMIT_OVERRIDE: std::cell::Cell<Option<i64>> = const { std::cell::Cell::new(None) };
}

/// Set (or clear) the `P1_DELTA_LIMIT_OVERRIDE` of the current thread.
#[cfg(test)]
pub(super) fn set_p1_delta_limit_override(limit: Option<i64>) {
    P1_DELTA_LIMIT_OVERRIDE.with(|l| l.set(limit));
}

/// The alias arrays, built while the reducer still has the member lists.
#[derive(Default)]
pub(super) struct RunLayout {
    /// `(head, record)` pairs, sorted by head once complete.
    pub(super) runs: Vec<(u32, RunRec)>,
    /// `(node, record)` pairs, sorted by node once complete.
    pub(super) p1: Vec<(u32, P1Rec)>,
    pub(super) chains: Vec<PrunedChain>,
    pub(super) far: Vec<FarPorts>,
    pub(super) blocks: Vec<AliasBlock>,
    /// Lengths: one [`LenBlock`] per 16 ids (one past the last node included), the dictionary the
    /// codes index, and the nodes whose length is not in it (sorted by node).
    pub(super) lens: Vec<LenBlock>,
    pub(super) dict: Vec<u32>,
    pub(super) escapes: Vec<LongLen>,
    /// Member lists of multi-node pruned chains, kept until the single-node set is complete (runs
    /// are cut by rank among non-single nodes): `(chain index, members)`.
    pending: Vec<(usize, Vec<ChainMember>)>,
    /// Rank of every node among the non-single nodes (built by [`Self::seal_singles`]).
    nonsingle_rank: Vec<u32>,
}

impl RunLayout {
    /// The single-node set is complete: rank every node among the non-single nodes, so runs can be
    /// cut (and, at query time, walked) in that id space.
    pub(super) fn seal_singles(&mut self, n: usize) {
        self.p1.sort_unstable_by_key(|r| r.0);
        let mut single = vec![false; n];
        for &(v, _) in &self.p1 {
            single[v as usize] = true;
        }
        let mut rank = 0u32;
        self.nonsingle_rank = (0..n)
            .map(|v| {
                let r = rank;
                rank += u32::from(!single[v]);
                r
            })
            .collect();
    }

    /// Cut `members` (walk order, as [`super::side_graph::SideGraph::walk`] yields them) of super
    /// `id` into runs; after [`Self::seal_singles`].
    pub(super) fn add_runs(
        &mut self,
        id: usize,
        pruned: bool,
        members: &[ChainMember],
        lens: &[u64],
    ) {
        debug_assert!(
            id <= RUN_ID_MASK as usize,
            "super id {id} exceeds the run record"
        );
        let m = members.len();
        let mut i = 0;
        while i < m {
            let mut j = i;
            let mut dir: i64 = 0;
            while j + 1 < m {
                let (here, next) = (members[j], members[j + 1]);
                let step =
                    self.nonsingle_rank[next.node] as i64 - self.nonsingle_rank[here.node] as i64;
                if next.forward != here.forward || step.abs() != 1 || (dir != 0 && step != dir) {
                    break;
                }
                debug_assert_eq!(
                    next.offset,
                    here.offset + lens[here.node],
                    "chain members are contiguous"
                );
                dir = step;
                j += 1;
            }
            let desc = dir == -1;
            let head = if desc { members[j] } else { members[i] };
            debug_assert!(head.offset <= OFF_MASK, "a super-node fits the offset word");
            self.runs.push((
                head.node as u32,
                RunRec {
                    id_flags: id as u32
                        | if pruned { RUN_PRUNED } else { 0 }
                        | if members[i].forward { 0 } else { RUN_REV }
                        | if desc { RUN_DESC } else { 0 },
                    start: head.offset as u32,
                },
            ));
            i = j + 1;
        }
    }

    /// Record a multi-node pruned chain; its `members` are cut into runs by [`Self::finish`].
    pub(super) fn add_chain(&mut self, chain: PrunedChain, members: Vec<ChainMember>) {
        self.pending.push((self.chains.len(), members));
        self.chains.push(chain);
    }

    /// Record a single-node pruned chain (node `v`, walked `forward` from `entry` to `exit`).
    pub(super) fn add_single(
        &mut self,
        v: usize,
        forward: bool,
        entry: u32,
        exit: u32,
        level: u32,
    ) {
        // Normalise to the node's forward direction: walked in reverse, the chain's exit is entered
        // from the end of the node that the forward walk leaves into (reversed), and vice versa. A
        // tip's missing port stays missing.
        let reverse = |d: u32| {
            if d == NO_PORT {
                NO_PORT
            } else {
                reverse_dir(d)
            }
        };
        let (entry, exit) = if forward {
            (entry, exit)
        } else {
            (reverse(exit), reverse(entry))
        };
        let base = 2 * v as i64;
        let limit = P1_DELTA_LIMIT_OVERRIDE
            .with(|l| l.get())
            .unwrap_or(i8::MAX as i64);
        let delta = |d: u32| -> Option<i8> {
            if d == NO_PORT {
                return Some(0);
            }
            let x = d as i64 - base;
            (x > P1_DELTA_ESCAPE as i64 && x.abs() <= limit).then_some(x as i8)
        };
        let mut flags = 0u8;
        if entry == NO_PORT {
            flags |= P1_NO_ENTRY;
        }
        if exit == NO_PORT {
            flags |= P1_NO_EXIT;
        }
        let (d_entry, d_exit) = match (delta(entry), delta(exit)) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                flags |= P1_FAR;
                self.far.push(FarPorts {
                    node: v as u32,
                    exit,
                    entry,
                });
                (P1_DELTA_ESCAPE, P1_DELTA_ESCAPE)
            }
        };
        self.p1.push((
            v as u32,
            P1Rec {
                d_entry,
                d_exit,
                level: level.min(LEVEL_SATURATED) as u8,
                flags,
            },
        ));
    }

    /// Cut the pending pruned chains, sort the records, encode the lengths and build the two rank
    /// sets over `n` original ids. After [`Self::seal_singles`] and every `add_runs`.
    pub(super) fn finish(&mut self, n: usize, lens: &[u64]) {
        let pending = std::mem::take(&mut self.pending);
        for (chain_index, members) in &pending {
            self.add_runs(*chain_index, true, members, lens);
        }
        let code_of = self.length_dictionary(lens);
        self.length_blocks(n, lens, &code_of);
        self.nonsingle_rank = Vec::new();
        self.runs.sort_unstable_by_key(|r| r.0);
        self.far.sort_unstable_by_key(|r| r.node);
        self.blocks = self.rank_sets(n);
    }

    /// The length dictionary: the 254 most frequent lengths (ties: the shorter) get a code, in that
    /// order; returns the code of each.
    fn length_dictionary(&mut self, lens: &[u64]) -> HashMap<u64, u8> {
        let mut freq: HashMap<u64, u32> = HashMap::new();
        for &l in lens {
            *freq.entry(l).or_insert(0) += 1;
        }
        let mut by_freq: Vec<(u64, u32)> = freq.into_iter().collect();
        by_freq.sort_unstable_by_key(|&(l, c)| (std::cmp::Reverse(c), l));
        by_freq.truncate(LEN_CODE_ESCAPE as usize - 1);
        by_freq.retain(|&(l, _)| l <= u32::MAX as u64);
        self.dict = by_freq.iter().map(|&(l, _)| l as u32).collect();
        self.dict
            .iter()
            .enumerate()
            .map(|(i, &l)| (l as u64, i as u8))
            .collect()
    }

    /// The length blocks: one code per node (the rest escape by node) plus the prefix sum of the
    /// non-single lengths (singles are skipped by the query-time scan too); one block past the last
    /// node so `prefix_len(n)` is defined. After [`Self::seal_singles`].
    fn length_blocks(&mut self, n: usize, lens: &[u64], code_of: &HashMap<u64, u8>) {
        self.lens = vec![LenBlock::default(); n / PREFIX_SAMPLE + 1];
        self.escapes.clear();
        let mut total = 0u64;
        let mut next_single = 0usize;
        for (v, &l) in lens.iter().enumerate() {
            let block = &mut self.lens[v / PREFIX_SAMPLE];
            if v.is_multiple_of(PREFIX_SAMPLE) {
                block.prefix = total;
            }
            block.codes[v % PREFIX_SAMPLE] = match code_of.get(&l) {
                Some(&c) => c,
                None => {
                    self.escapes.push(LongLen {
                        node: v as u32,
                        len: l as u32,
                    });
                    LEN_CODE_ESCAPE
                }
            };
            if next_single < self.p1.len() && self.p1[next_single].0 as usize == v {
                next_single += 1;
            } else {
                total += l;
            }
        }
        if n.is_multiple_of(PREFIX_SAMPLE) {
            self.lens[n / PREFIX_SAMPLE].prefix = total;
        }
    }

    /// The two rank sets over `n` original ids, one [`AliasBlock`] per 64: the run heads and the
    /// single-node pruned chains. The head of a run member is the highest run head at or below it
    /// (singles between them are not heads), and the run index is that head's rank. After the runs
    /// are sorted.
    fn rank_sets(&self, n: usize) -> Vec<AliasBlock> {
        let mut blocks = vec![AliasBlock::default(); n.div_ceil(64)];
        for &(h, _) in &self.runs {
            blocks[h as usize / 64].heads |= 1 << (h % 64);
        }
        for &(v, _) in &self.p1 {
            blocks[v as usize / 64].singles |= 1 << (v % 64);
        }
        let (mut heads, mut singles, mut prev) = (0u32, 0u32, u32::MAX);
        for (b, block) in blocks.iter_mut().enumerate() {
            block.head_rank = heads;
            block.single_rank = singles;
            block.prev_head = prev;
            heads += block.heads.count_ones();
            singles += block.singles.count_ones();
            if block.heads != 0 {
                prev = (b * 64) as u32 + 63 - block.heads.leading_zeros();
            }
        }
        blocks
    }
}
