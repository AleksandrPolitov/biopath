//! The 8-byte per-node record: the node's length (or the long-length escape), its kind and the
//! kind's id.

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::rows::ROW_MASK_BITS;

/// The hot per-node fields: 8 bytes. `lo` holds the length (low [`LEN_BITS`] bits,
/// [`LEN_ESCAPE`] = in the long-length table), the kind and, for a non-cut node, the first-hop mask
/// of its closed-form rows; `id` depends on the kind (see the module docs of
/// [`crate::topology`]). A sentinel record follows the last node. Plain data, stored in the
/// image.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NodeRecord<IndexType: GraphIndexInteger> {
    pub(super) lo: u32,
    pub(super) id: u32,
    _index: std::marker::PhantomData<IndexType>,
}
crate::image::image_record!(impl<I> NodeRecord<I> = [u32, u32]);

/// Bits of a record's length; this value means "in the long-length table".
pub const LEN_BITS: u32 = 16;
pub const LEN_ESCAPE: u32 = (1 << LEN_BITS) - 1;
pub(super) const KIND_SHIFT: u32 = LEN_BITS;
pub(super) const MASK_SHIFT: u32 = KIND_SHIFT + 2;
/// A cut node: `id` is its dense cut id.
pub(super) const K_CUT: u32 = 0;
/// A non-cut node outside a record block whose rows are a closed form (mask in `lo`, see
/// [`RowRef::Closed`](super::RowRef::Closed)) or absent (mask `0`): `id` is its block
/// ([`NO_BLOCK`] for an isolated node).
pub(super) const K_SIMPLE: u32 = 1;
/// A non-cut node outside a record block with stored rows: `id` is the position of its header
/// `[block]` in the lift's row table, its cells follow.
pub(super) const K_PLAIN: u32 = 2;
/// A non-cut node of a record block: `id` is the position of its header `[block, SPQR location]` in
/// the lift's row table (its cells follow when [`HAS_CELLS`]); before the lift lays the rows out, its
/// index in the build-time [`BuildLoc`](super::BuildLoc) table instead.
pub(super) const K_RECORD: u32 = 3;
/// `lo` bit: the node's stored row cells follow its header.
pub(super) const HAS_CELLS: u32 = 1 << (MASK_SHIFT + 2 * ROW_MASK_BITS);
/// A [`K_SIMPLE`] node's `id` when it lies in no block.
pub(super) const NO_BLOCK: u32 = u32::MAX;

impl<IndexType: GraphIndexInteger> NodeRecord<IndexType> {
    pub(super) fn new(len: usize, kind: u32, mask: u32, id: u32) -> Self {
        let len = (len as u32).min(LEN_ESCAPE);
        debug_assert!(mask < 1 << (2 * ROW_MASK_BITS));
        Self {
            lo: len | kind << KIND_SHIFT | mask << MASK_SHIFT,
            id,
            _index: std::marker::PhantomData,
        }
    }

    #[inline]
    pub(super) fn kind(&self) -> u32 {
        self.lo >> KIND_SHIFT & 3
    }

    #[inline]
    pub(super) fn mask(&self) -> u32 {
        self.lo >> MASK_SHIFT
    }
}
