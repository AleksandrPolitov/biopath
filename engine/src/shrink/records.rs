//! The records of the alias as the image stores them; `layout.rs` builds them and `view.rs` reads
//! them.
//!
//! There is no word per original node. A single-node pruned chain is a 4-byte [`P1Rec`] of its own
//! (its ports as deltas of directed ids, its level), found through a rank set over original ids.
//! Every other node is a member of a *run*: walk-consecutive members of one super-node, in one
//! orientation, whose ids are consecutive AMONG THE NON-SINGLE NODES (ascending or descending along
//! the walk) — a backbone piece and the next one are separated in id space only by the alleles
//! between them. A run is addressed by its lowest id (its *head*): a rank set marks the heads, and
//! an 8-byte [`RunRec`] gives the super-node and the head's start offset. A member's offset is that
//! plus the lengths of the non-single nodes between, read from a prefix sum of those lengths (a
//! `u64` sample every [`PREFIX_SAMPLE`] ids plus one length code per node: a byte indexing a
//! per-index dictionary of the 254 most frequent lengths, [`LEN_CODE_ESCAPE`] = see
//! `SHRINK_LONG_LENS`).
//!
//! The level of a pruned chain refines the pruning order exactly where the query's "resolve the
//! chain pruned first" rule needs it: a chain whose port lies in another chain was pruned before
//! it, and two chains of one level are not attached to each other, so either order is exact.

/// Run record flags (high bits of `id_flags`); the low [`RUN_ID_BITS`] bits are the super id (a G′
/// node, or a multi-node pruned chain's index in `SHRINK_PCHAINS`).
pub(super) const RUN_PRUNED: u32 = 1 << 31;
/// The run's members are walked in reverse (entered through their right side).
pub(super) const RUN_REV: u32 = 1 << 30;
/// Ids DEcrease along the walk: the head (lowest id) is the run's LAST member.
pub(super) const RUN_DESC: u32 = 1 << 29;
const RUN_ID_BITS: u32 = 29;
pub(super) const RUN_ID_MASK: u32 = (1 << RUN_ID_BITS) - 1;
/// Single-node pruned chain flag: the ports are in `SHRINK_FAR_PORTS` (a delta did not fit).
pub(super) const P1_FAR: u8 = 1;
/// Single-node pruned chain flags of a pruned tip: no walk enters it (its entry end is unlinked) /
/// no walk leaves it (its exit end is unlinked). The far table stores [`NO_PORT`] instead.
pub(super) const P1_NO_ENTRY: u8 = 2;
pub(super) const P1_NO_EXIT: u8 = 4;
/// The port of a pruned tip's unlinked end, in a [`PrunedChain`], a [`FarPorts`] record or a stored
/// directed id.
pub(super) const NO_PORT: u32 = u32::MAX;
/// The deltas of a single-node pruned chain whose ports are in `SHRINK_FAR_PORTS`.
pub(super) const P1_DELTA_ESCAPE: i8 = i8::MIN;

/// Levels at or above this are stored saturated (then the order of two such chains is unknown, and a
/// query between them takes the shortest of the walks both hop sequences allow, `distance.rs`).
pub(super) const LEVEL_SATURATED: u32 = 255;

/// One run of original nodes (8 bytes): `id_flags` = super id | `RUN_PRUNED` | `RUN_REV` |
/// `RUN_DESC`; `start` = the start offset of the head member in the super's walk. A member's offset
/// is `start` plus the prefix-sum difference (non-single lengths, `LenBlock`) between it and the head.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RunRec {
    pub id_flags: u32,
    pub start: u32,
}
crate::image::image_record!(RunRec = [u32, u32]);

/// One single-node pruned chain (4 bytes): its ports as deltas from its own forward directed id
/// (`2 * node`; a byte each, which fits 98–99.9 % of the records on the graphs measured, the rest
/// escape to `SHRINK_FAR_PORTS`), its level (saturated at `LEVEL_SATURATED`) and flags. Its
/// length is the node's.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct P1Rec {
    pub d_entry: i8,
    pub d_exit: i8,
    pub level: u8,
    pub flags: u8,
}
crate::image::image_record!(P1Rec = [i8, i8, u8, u8]);

/// Length code of a node whose length is not in the dictionary (`SHRINK_LONG_LENS` has it).
pub(super) const LEN_CODE_ESCAPE: u8 = 255;
/// One `u64` prefix-sum sample per this many original ids (the scan between samples reads at most
/// this many length codes).
pub(super) const PREFIX_SAMPLE: usize = 16;

/// One multi-node pruned chain (16 bytes): its length, its `(exit, entry)` ports as directed original
/// ids ([`NO_PORT`] at a pruned tip's unlinked end) and its level.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PrunedChain {
    pub len: u32,
    pub exit: u32,
    pub entry: u32,
    pub level: u32,
}
crate::image::image_record!(PrunedChain = [u32, u32, u32, u32]);

/// 64 original ids of the two rank sets in one line (32 bytes): the run heads and the single-node
/// pruned chains, each with the rank of the block's first id, plus the highest run head below the
/// block (so the head of any run member is found in O(1) without a select structure).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AliasBlock {
    pub heads: u64,
    pub singles: u64,
    pub head_rank: u32,
    pub single_rank: u32,
    pub prev_head: u32,
    pub _pad: u32,
}
crate::image::image_record!(AliasBlock = [u64, u64, u32, u32, u32, u32]);

impl AliasBlock {
    /// Whether `v` (in this block) is a single-node pruned chain, and its rank among them.
    #[inline]
    pub(super) fn single(&self, v: usize) -> Option<usize> {
        let bit = v % 64;
        (self.singles >> bit & 1 == 1).then(|| {
            self.single_rank as usize + (self.singles & ((1u64 << bit) - 1)).count_ones() as usize
        })
    }
}

/// 16 original ids' lengths in one line (24 bytes): the prefix sum of the non-single lengths below
/// the block's first id, and one length code per id (an index into `SHRINK_LEN_DICT`,
/// `LEN_CODE_ESCAPE` = see `SHRINK_LONG_LENS`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LenBlock {
    pub prefix: u64,
    pub codes: [u8; PREFIX_SAMPLE],
}
crate::image::image_record!(LenBlock = [u64, [u8; PREFIX_SAMPLE]]);

/// The ports of a single-node pruned chain whose deltas did not fit (`SHRINK_FAR_PORTS`, sorted by node).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FarPorts {
    pub node: u32,
    pub exit: u32,
    pub entry: u32,
}
crate::image::image_record!(FarPorts = [u32, u32, u32]);

/// The length of an original node that is not in the length dictionary (`SHRINK_LONG_LENS`, sorted
/// by node).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LongLen {
    pub node: u32,
    pub len: u32,
}
crate::image::image_record!(LongLen = [u32, u32]);
