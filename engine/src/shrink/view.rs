//! The query-time view of the shrink sections: where an original node lies in the alias, and an
//! original position resolved onto its super-node.

use std::borrow::Cow;

use super::{
    distance::ShrunkOracle,
    records::{
        AliasBlock, FarPorts, LEN_CODE_ESCAPE, LEVEL_SATURATED, LenBlock, LongLen, NO_PORT, P1_FAR,
        P1_NO_ENTRY, P1_NO_EXIT, P1Rec, PREFIX_SAMPLE, PrunedChain, RUN_DESC, RUN_ID_MASK,
        RUN_PRUNED, RUN_REV, RunRec,
    },
    side_graph::dir_decode,
};
use crate::graph_view::NameView;

/// A position on an original node: `(node, forward, offset)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OriginalPos {
    pub node: usize,
    pub forward: bool,
    pub offset: u64,
}

/// What [`ShrinkView::probe`] reports.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ProbeResult {
    pub kept: Option<usize>,
    /// A pruned chain: its length and its `(exit, entry)` ports (directed original ids), for the walk
    /// direction the position uses.
    pub chain: Option<(u64, (u32, u32))>,
    pub along: bool,
    pub offset: u64,
    pub end: bool,
    pub level: u32,
}

/// The super-node an original node lies in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuperNode {
    /// A G′ node.
    Kept(usize),
    /// A single-node pruned chain: its record index.
    Single(usize),
    /// A multi-node pruned chain: its index.
    Multi(usize),
}

/// A position resolved onto its super-node.
#[derive(Clone, Copy, Debug)]
pub(super) struct SuperPos {
    pub(super) of: SuperNode,
    /// The original node (ports of a single-node chain are relative to it).
    node: usize,
    /// Walking the super-node in its own direction.
    pub(super) along: bool,
    pub(super) offset: u64,
    /// The end of an original node (sorts before a start at the same offset).
    end: bool,
    /// Pruned chains: their length and their level. Kept: unused / `u32::MAX`.
    pub(super) chain_len: u64,
    pub(super) level: u32,
}

/// An original position resolved onto its super-node, as [`ShrinkView::locate`] reports it: which
/// super-node, whether the position's walk goes along the super-node's own direction, and its offset
/// along that walk. For a pruned chain, [`ShrinkView::port_positions`] gives where a walk through it is
/// forced to leave and enter.
#[derive(Clone, Copy, Debug)]
pub struct SuperLocation {
    pos: SuperPos,
}

impl SuperLocation {
    /// The super-node: a G′ node or a pruned chain.
    pub fn super_node(&self) -> SuperNode {
        self.pos.of
    }

    /// Whether the position's walk goes along the super-node's own direction.
    pub fn along(&self) -> bool {
        self.pos.along
    }

    /// The offset along the walk (from the super-node's start when [`Self::along`], else from its
    /// end).
    pub fn offset(&self) -> u64 {
        self.pos.offset
    }

    /// A pruned chain's length (`None` for a G′ node).
    pub fn chain_len(&self) -> Option<u64> {
        (!matches!(self.pos.of, SuperNode::Kept(_))).then_some(self.pos.chain_len)
    }

    /// A pruned chain's level (saturated at 255; `None` for a G′ node).
    pub fn level(&self) -> Option<u32> {
        (!matches!(self.pos.of, SuperNode::Kept(_))).then_some(self.pos.level)
    }

    /// Whether the position is the end of an original node: at an equal offset it sorts just before
    /// a start (see the module docs on coinciding positions).
    pub fn end(&self) -> bool {
        self.pos.end
    }
}

/// [`ShrinkView::resolve`] reads only node lengths from its oracle; this adapter gives it a length
/// function.
struct LengthsOnly<F>(F);

impl<F: FnMut(usize) -> u64> ShrunkOracle for LengthsOnly<F> {
    fn node_len(&mut self, node: usize) -> u64 {
        (self.0)(node)
    }

    fn distance(
        &mut self,
        _: super::distance::ShrunkPos,
        _: super::distance::ShrunkPos,
        _: bool,
    ) -> Option<u64> {
        unreachable!("resolving a position reads no distance")
    }
}

impl SuperPos {
    /// The order of positions on one super-node (see the module docs on coinciding positions).
    #[inline]
    pub(super) fn key(&self) -> u64 {
        2 * self.offset - u64::from(self.end)
    }

    /// Whether both positions lie on the same super-node, walked the same way.
    #[inline]
    pub(super) fn same_super(&self, other: &SuperPos) -> bool {
        self.of == other.of && self.along == other.along
    }
}

/// The resolved shrink sections.
#[derive(Clone, Copy)]
pub struct ShrinkView<'a> {
    pub(super) blocks: &'a [AliasBlock],
    pub(super) runs: &'a [RunRec],
    pub(super) p1: &'a [P1Rec],
    pub(super) chains: &'a [PrunedChain],
    pub(super) far: &'a [FarPorts],
    pub(super) lens: &'a [LenBlock],
    pub(super) dict: &'a [u32],
    pub(super) long_lens: &'a [LongLen],
    pub(super) names: NameView<'a>,
    pub(super) node_count: usize,
}

/// A run member located: its record and its head.
#[derive(Clone, Copy)]
struct RunMember {
    rec: RunRec,
    head: usize,
}

impl<'a> ShrinkView<'a> {
    /// The ORIGINAL node named `name`.
    pub fn node_by_name(&self, name: &str) -> Option<usize> {
        self.names.node_by_name(name)
    }

    /// The name of ORIGINAL node `v`.
    pub fn node_name(&self, v: usize) -> Cow<'a, str> {
        self.names.node_name(v)
    }

    /// The number of original nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Locate original node `v` in its run (`v` must not be a single-node pruned chain).
    #[inline]
    fn run_member(&self, v: usize) -> RunMember {
        let block = &self.blocks[v / 64];
        let bit = v % 64;
        let below = block.heads & (u64::MAX >> (63 - bit));
        let head = if below != 0 {
            (v / 64) * 64 + 63 - below.leading_zeros() as usize
        } else {
            block.prev_head as usize
        };
        let run_index = block.head_rank as usize + below.count_ones() as usize - 1;
        RunMember {
            rec: self.runs[run_index],
            head,
        }
    }

    /// The total length of the non-single original nodes below `v` (`v <= node_count`).
    #[inline]
    fn prefix_len(&self, v: usize) -> u64 {
        let block = &self.lens[v / PREFIX_SAMPLE];
        let mut total = block.prefix;
        let first = (v / PREFIX_SAMPLE) * PREFIX_SAMPLE;
        if v > first {
            let singles = self.blocks[first / 64].singles >> (first % 64);
            for u in first..v {
                if singles >> (u - first) & 1 == 0 {
                    total += match block.codes[u - first] {
                        LEN_CODE_ESCAPE => self.node_len(u),
                        c => self.dict[c as usize] as u64,
                    };
                }
            }
        }
        total
    }

    /// The start offset of run member `v` in its super's walk: the head's, plus the lengths of the
    /// members walked between them (above `v` when ids descend along the walk).
    #[inline]
    fn run_member_start(&self, m: &RunMember, v: usize) -> u64 {
        if v == m.head {
            return m.rec.start as u64;
        }
        let pl = |x: usize| self.prefix_len(x);
        if m.rec.id_flags & RUN_DESC != 0 {
            m.rec.start as u64 - (pl(v + 1) - pl(m.head + 1))
        } else {
            m.rec.start as u64 + (pl(v) - pl(m.head))
        }
    }

    /// The length of ORIGINAL node `v`, from its code.
    #[inline]
    pub fn node_len(&self, v: usize) -> u64 {
        match self.lens[v / PREFIX_SAMPLE].codes[v % PREFIX_SAMPLE] {
            LEN_CODE_ESCAPE => {
                let i = self.long_lens.partition_point(|l| (l.node as usize) < v);
                self.long_lens[i].len as u64
            }
            c => self.dict[c as usize] as u64,
        }
    }

    /// Where ORIGINAL position `p` lies: its super-node and the offset along it (`g_node_len` gives
    /// the length of a G′ node).
    pub fn locate(&self, p: OriginalPos, g_node_len: impl FnMut(usize) -> u64) -> SuperLocation {
        SuperLocation {
            pos: self.resolve(p, &mut LengthsOnly(g_node_len)),
        }
    }

    /// The port positions of a pruned chain for the walk `location` is on: `(exit, entry)`, the start
    /// of the original node the walk leaves into and the end of the one it enters from (`None` at a
    /// pruned tip's unlinked end). `None` for a G′ node.
    #[allow(clippy::type_complexity)]
    pub fn port_positions(
        &self,
        location: &SuperLocation,
    ) -> Option<(Option<OriginalPos>, Option<OriginalPos>)> {
        (!matches!(location.pos.of, SuperNode::Kept(_))).then(|| {
            (
                self.side_port(&location.pos, true),
                self.side_port(&location.pos, false),
            )
        })
    }

    /// Resolve `p` onto its super-node (`oracle` gives the length of a G′ node).
    pub(super) fn resolve(&self, p: OriginalPos, oracle: &mut impl ShrunkOracle) -> SuperPos {
        let len = self.node_len(p.node);
        match self.blocks[p.node / 64].single(p.node) {
            Some(index) => self.resolve_single(p, index, len),
            None => self.resolve_in_run(p, len, oracle),
        }
    }

    /// `p` lies on single-node pruned chain `index` (of length `len`).
    fn resolve_single(&self, p: OriginalPos, index: usize, len: u64) -> SuperPos {
        // A single node walked in its own direction: start 0, so the offset is the position's
        // offset either way.
        SuperPos {
            of: SuperNode::Single(index),
            node: p.node,
            along: p.forward,
            offset: p.offset,
            end: len > 0 && p.offset == len,
            chain_len: len,
            level: self.p1[index].level as u32,
        }
    }

    /// `p` lies on a run member (its node of length `len`).
    fn resolve_in_run(&self, p: OriginalPos, len: u64, oracle: &mut impl ShrunkOracle) -> SuperPos {
        let v = p.node;
        let m = self.run_member(v);
        let flags = m.rec.id_flags;
        let pruned = flags & RUN_PRUNED != 0;
        let id = (flags & RUN_ID_MASK) as usize;
        let start = self.run_member_start(&m, v);
        let along = p.forward == (flags & RUN_REV == 0);
        let chain_len = if pruned {
            self.chains[id].len as u64
        } else {
            0
        };
        // The super-node's length is needed only to mirror an offset walked against it.
        let offset = if along {
            start + p.offset
        } else {
            let super_len = if pruned {
                chain_len
            } else {
                oracle.node_len(id)
            };
            super_len - start - len + p.offset
        };
        SuperPos {
            of: if pruned {
                SuperNode::Multi(id)
            } else {
                SuperNode::Kept(id)
            },
            node: v,
            along,
            offset,
            end: len > 0 && p.offset == len,
            chain_len,
            // Compared in the same saturated space as the single-node records: a chain at or above
            // the cap has an unknown order against another such chain.
            level: if pruned {
                self.chains[id].level.min(LEVEL_SATURATED)
            } else {
                u32::MAX
            },
        }
    }

    /// The `(exit, entry)` ports of a pruned chain, as directed original ids.
    #[inline]
    fn ports(&self, p: &SuperPos) -> (u32, u32) {
        match p.of {
            SuperNode::Multi(c) => {
                let c = self.chains[c];
                (c.exit, c.entry)
            }
            SuperNode::Single(index) => self.single_ports(index, p.node),
            SuperNode::Kept(_) => unreachable!("a G′ node has no ports"),
        }
    }

    /// The `(exit, entry)` ports of single-node pruned chain `index` (node `v`): deltas from `v`'s
    /// forward directed id, or the far table; [`NO_PORT`] at a pruned tip's unlinked end.
    #[inline]
    fn single_ports(&self, index: usize, v: usize) -> (u32, u32) {
        let rec = self.p1[index];
        if rec.flags & P1_FAR != 0 {
            let i = self.far.partition_point(|f| (f.node as usize) < v);
            let f = self.far[i];
            return (f.exit, f.entry);
        }
        let base = 2 * v as i64;
        let port = |flag: u8, delta: i8| {
            if rec.flags & flag != 0 {
                NO_PORT
            } else {
                (base + delta as i64) as u32
            }
        };
        (port(P1_NO_EXIT, rec.d_exit), port(P1_NO_ENTRY, rec.d_entry))
    }

    /// The port a pruned chain walked `along` leaves through (`exit_end`: the start of the directed
    /// original node it leads into) or is entered through (else: the end of the node it is entered
    /// from); `None` at a pruned tip's unlinked end, where no walk leaves or enters.
    pub(super) fn side_port(&self, p: &SuperPos, exit_end: bool) -> Option<OriginalPos> {
        let (exit, entry) = self.ports(p);
        // Walked along, a chain leaves through its stored exit end; walked against, through its stored
        // entry end, and the port there is passed in the opposite orientation.
        let port = if exit_end == p.along { exit } else { entry };
        if port == NO_PORT {
            return None;
        }
        let (node, forward) = dir_decode(port);
        Some(OriginalPos {
            node,
            forward: forward == p.along,
            offset: if exit_end { 0 } else { self.node_len(node) },
        })
    }

    /// How a position resolves (tests): kept G′ node or pruned chain (with its length and
    /// `(exit, entry)` ports as the forward walk sees them), walk direction, offset, end flag and
    /// level.
    #[cfg(test)]
    pub(super) fn probe(&self, p: OriginalPos, oracle: &mut impl ShrunkOracle) -> ProbeResult {
        let sp = self.resolve(p, oracle);
        let kept = match sp.of {
            SuperNode::Kept(g) => Some(g),
            _ => None,
        };
        ProbeResult {
            kept,
            chain: kept.is_none().then(|| (sp.chain_len, self.ports(&sp))),
            along: sp.along,
            offset: sp.offset,
            end: sp.end,
            level: sp.level,
        }
    }
}
