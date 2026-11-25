//! The records of the chain layout: one per cut node, per block (a word), per block that is no
//! cut's heavy child, per chain and per special block.

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};

use crate::{
    image::{Padded, image_record},
    min_plus::MinPlus,
    path::GfaPathLength,
};

/// One cut node, indexed by its dense cut id (= chain position order). 24 bytes at a 32-bit word.
/// The topology writes the structural fields; the lift fills the rest in place. Its blocks are not
/// stored: under the chain block numbering its heavy child is block `roots + c` and its parent block
/// follows from its chain ([`ChainNav::parent_block_of_cut`](super::ChainNav::parent_block_of_cut)).
///
/// No padding at either word size: `flags` is as wide as the index word, so the record ends on a
/// word boundary (a plain `u32` at word size 32).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CutRec<IndexType: GraphIndexInteger> {
    /// The cut's graph node.
    pub node: NodeIndex<IndexType>,
    /// Prefix lift: the sum of the regular steps' direct costs from the cut up to its forest root.
    pub p: GfaPathLength<IndexType>,
    /// The cut node's sequence length.
    pub len: GfaPathLength<IndexType>,
    /// The whole-graph flip closure `full[c]`: its two off-diagonals (the diagonal is `0`).
    pub f01: MinPlus<IndexType>,
    pub f10: MinPlus<IndexType>,
    /// [`cut_flags`].
    pub flags: Padded<u32, IndexType>,
}
image_record!(impl<I> CutRec<I> = [
    NodeIndex<I>,
    GfaPathLength<I>,
    GfaPathLength<I>,
    MinPlus<I>,
    MinPlus<I>,
    Padded<u32, I>,
]);

/// Bits of [`CutRec::flags`]: the low bits are flags, the rest is the cut's chain id.
pub mod cut_flags {
    /// The cut's up-orientation (prefix lift).
    pub const ORIENT: u32 = 1;
    /// The cut's own step (to its parent cut) is irregular / blocked (prefix lift).
    pub const IRREGULAR: u32 = 2;
    pub const BLOCKED: u32 = 4;
    /// The cut can flip through its pendant subtree (`down[c]` has a finite off-diagonal).
    pub const DOWN_FLIP: u32 = 8;
    /// The cut lies in three or more blocks (see [`super::CutExtraDir`]): its light children are
    /// listed in the topology's extra-blocks array.
    pub const HAS_EXTRA: u32 = 16;
    /// The cut is its chain's first cut node (its parent block is not the heavy child of the cut
    /// before it; see [`ChainNav::parent_block_of_cut`](crate::chain_nav::ChainNav::parent_block_of_cut)).
    pub const FIRST_IN_CHAIN: u32 = 32;
    /// The bits below `CHAIN_SHIFT` are flags; `flags >> CHAIN_SHIFT` is the cut's chain id.
    pub const CHAIN_SHIFT: u32 = 6;
}

/// A block that is no cut node's heavy child (a component root, or the head of a chain hanging off a
/// light edge): its tree position, chain and, for a special block, the index of its
/// [`SpecialBlockRec`] (`NONE` otherwise). A heavy child's position and chain are its parent cut's
/// next ones. Indexed by [`ChainNav::head_index`](super::ChainNav::head_index). 12 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadBlockRec {
    pub tpos: u32,
    pub chain: u32,
    pub special: u32,
}
crate::image::image_record!(HeadBlockRec = [u32, u32, u32]);

/// A cut node in three or more blocks: where its child list (the count, then the children, heavy
/// first) starts in the extra-blocks array. Sorted by cut; read only for [`cut_flags::HAS_EXTRA`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CutExtraDir {
    pub cut: u32,
    pub start: u32,
}
crate::image::image_record!(CutExtraDir = [u32, u32]);

/// Bits of a block's word (one `u32` per block): [`block_bits`] in the low bits; above
/// [`block_bits::COMPONENT_SHIFT`] the block's connected component when the index has few enough
/// components to inline them ([`block_bits::COMPONENT_INLINE_LIMIT`]), else zero (the component is
/// the chain's).
pub mod block_bits {
    /// Bits 0-1: the block's node-row kind (`crate::node_rows::RowKind`).
    pub const ROW_KIND_MASK: u32 = 0b11;
    /// The block has a [`super::SpecialBlockRec`] (its cut count is not 2, or it is a root block).
    pub const SPECIAL: u32 = 1 << 2;
    /// 2-cut block: the position (0/1) of its parent cut in its cut list.
    pub const PARENT_POS: u32 = 1 << 3;
    /// 2-cut half-row block: the away orientation of its cut at position 0 / 1.
    pub const AWAY0: u32 = 1 << 4;
    pub const AWAY1: u32 = 1 << 5;
    /// The block's `M_aug` is derived from the prefix lift (no dense data).
    pub const DERIVED: u32 = 1 << 6;
    /// 2-cut block: its parent cut / child cut is flip-capable (the block's flip list).
    pub const FLIP_PARENT: u32 = 1 << 7;
    pub const FLIP_CHILD: u32 = 1 << 8;
    /// Every non-cut node of the block has closed-form rows (and it has no record): its nodes keep no
    /// arcs; in-block distances come from the rows (`BlockCutLift::simple_block_distance`).
    pub const SIMPLE_ROWS: u32 = 1 << 9;
    // Bits 10-11 are unused (`0`).
    /// `bits >> COMPONENT_SHIFT` is the block's component when the index inlines components: the
    /// unreachable-pair test then costs one record read per endpoint, not a chain lookup.
    pub const COMPONENT_SHIFT: u32 = 12;
    /// Components are inlined when the sentinel component id (the component count) fits.
    pub const COMPONENT_INLINE_LIMIT: u32 = 1 << (32 - COMPONENT_SHIFT);
}

/// One heavy-path chain. 56 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ChainRec {
    /// First tree position of the chain and its length (tree nodes).
    pub tpos_start: u32,
    pub len: u32,
    /// Dense id of the chain's first cut node and that cut's offset within the chain (`0` for a
    /// cut-headed chain, `1` for a block-headed one, `2` under the super-root); `cut_off == len`
    /// when the chain has no cut node.
    pub cut_start: u32,
    pub cut_off: u32,
    /// The parent of the chain's head: its position, tree id and chain (the root chain attaches to
    /// itself).
    pub attach_tpos: u32,
    pub attach_id: u32,
    pub attach_chain: u32,
    /// The head's tree id and depth.
    pub head_id: u32,
    pub head_depth: u32,
    /// The first block on the chain (`NONE` if none).
    pub first_block_id: u32,
    /// The connected component of the chain's blocks (the sentinel for a block-less chain).
    pub component: u32,
    /// The chain's exception cuts (irregular / blocked steps), sorted, in the lift's exception
    /// array; `exc_inherit` = the nearest exception above the chain's head (`NONE` if none).
    pub exc_start: u32,
    pub exc_count: u32,
    pub exc_inherit: u32,
}
crate::image::image_record!(
    ChainRec = [
        u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32
    ]
);

/// A block whose cut count is not 2 (or a root block): its cut list in the decomposition's order
/// (dense cut ids, the `M_aug` / node-row order), its flip-capable cuts and its half-row away bits.
/// Sorted by block. The topology writes the cut list; the lift fills the rest in place.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecialBlockRec {
    pub block: u32,
    pub cut_start: u32,
    pub cut_count: u32,
    pub flip_start: u32,
    pub flip_count: u32,
    /// Bit `j` = the away orientation of the block's `j`-th cut (half-row blocks).
    pub away: u32,
    /// Start of the block's dense `M_aug` in the lift's dense data.
    pub aug_start: u32,
}
crate::image::image_record!(SpecialBlockRec = [u32, u32, u32, u32, u32, u32, u32]);

/// The index of special block `b`'s [`SpecialBlockRec`] in the records (sorted by block).
#[inline]
pub fn special_index_by_block(specials: &[SpecialBlockRec], b: usize) -> usize {
    specials
        .binary_search_by_key(&(b as u32), |r| r.block)
        .expect("a special block has a record")
}
