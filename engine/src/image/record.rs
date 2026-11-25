//! What an image may store: [`ImageRecord`] marks the element types, and [`Padded`] fills the gaps
//! a generic record would otherwise leave to the compiler.
//!
//! An image is written to disk byte for byte, so a record with implicit padding would write
//! whatever its padding bytes held: uninitialised memory, different on every build. No image record
//! has padding, at either word size. A generic record mixes `u32` fields with index-word fields
//! (`u32` or `u64`); where that would leave a gap at word size 64, a field is stored in a [`Padded`]
//! slot as wide as the gap needs, whose spare bytes are explicit zeros. At word size 32 every slot
//! is exactly its value (`Padded<u32, u32>` is a `u32`), so the default image layout is unchanged.
//!
//! The guard: every element type of the image API is an [`ImageRecord`], which states the sum of
//! its field sizes, and a compile-time assertion ([`ImageRecord::NO_PADDING`]) checks that it
//! equals the type's size: where `image_record!` declares a record (a generic one at both word
//! sizes), and for every type the image API is instantiated with. A padded record does not compile.

use bidirected_adjacency_array::index::{
    DirectedNodeIndex, GraphIndexInteger, NodeIndex, OptionalNodeIndex,
};
use spqr_tree::decomposition::indices::BlockIndex;

use crate::{
    min_plus::{MinPlus, MinPlusMatrix},
    path::GfaPathLength,
};

/// A plain-data type an [`Image`](super::Image) stores: its bytes are written to the `.bpi` file as
/// they are in memory and read back by reinterpreting the file's bytes.
///
/// # Safety
///
/// Implement it only for a primitive integer, or a `#[repr(C)]` / `#[repr(transparent)]` type
/// whose fields are primitive integers, arrays of them or other such types: no pointers or
/// references (the bytes are written to a file), no `bool`, `char` or enum (every bit pattern must
/// be a valid value, since a file is reinterpreted as `&[Self]`). [`Self::FIELD_BYTES`] must be the
/// sum of the field sizes; the image checks that it equals the type's size, so the type has no
/// padding. Use `image_record!` for a struct.
pub unsafe trait ImageRecord: Copy {
    /// The sum of the sizes of the type's fields.
    const FIELD_BYTES: usize;

    /// Compiles only if the type has no padding (`size_of::<Self>() == FIELD_BYTES`). The image
    /// API evaluates it for every element type it is instantiated with.
    const NO_PADDING: () = assert!(
        std::mem::size_of::<Self>() == Self::FIELD_BYTES,
        "an image record type has padding bytes, which would be written to the index \
         uninitialised: make the gap an explicit field (see `image::Padded`)"
    );
}

/// `unsafe impl ImageRecord` for a struct, from the types of its fields in declaration order:
/// `image_record!(Rec = [u32, u64]);`, or for a struct generic over the index word
/// `image_record!(impl<I> Rec<I> = [NodeIndex<I>, u32]);`. The caller asserts the struct is
/// `#[repr(C)]` with exactly these fields, all plain data (see the trait's safety contract). The
/// macro checks [`ImageRecord::NO_PADDING`] right there, for a generic struct at both word sizes
/// (32 and 64), so a padded record, or a field missing from or extra in the list, does not compile.
macro_rules! image_record {
    (impl<$index:ident> $ty:ty = [$($field:ty),* $(,)?]) => {
        // SAFETY: the caller asserts `#[repr(C)]` plain data with exactly these fields;
        // `NO_PADDING` checks the sizes.
        unsafe impl<$index: ::bidirected_adjacency_array::index::GraphIndexInteger>
            $crate::image::ImageRecord for $ty
        {
            const FIELD_BYTES: usize = 0 $(+ ::std::mem::size_of::<$field>())*;
        }
        const _: () = {
            type AtWord<$index> = $ty;
            let () = <AtWord<u32> as $crate::image::ImageRecord>::NO_PADDING;
            let () = <AtWord<u64> as $crate::image::ImageRecord>::NO_PADDING;
        };
    };
    ($ty:ty = [$($field:ty),* $(,)?]) => {
        // SAFETY: as above.
        unsafe impl $crate::image::ImageRecord for $ty {
            const FIELD_BYTES: usize = 0 $(+ ::std::mem::size_of::<$field>())*;
        }
        const _: () = <$ty as $crate::image::ImageRecord>::NO_PADDING;
    };
}
pub(crate) use image_record;

/// A `T` stored in a slot as wide and as aligned as the wider of `T` and `W`, every byte
/// initialised: the bytes past a narrower `T` are zeros. A record wraps one field in it to fill
/// what would be a padding gap at word size 64 (see the module docs). `W` only lends its size and
/// alignment (a primitive integer, typically the index word); it is never read.
#[repr(C)]
#[derive(Clone, Copy)]
pub union Padded<T: Copy, W: Copy> {
    value: T,
    _width: W,
}

impl<T: Copy, W: Copy> Padded<T, W> {
    /// `value`, zero-extended to the slot's width.
    #[inline]
    pub fn new(value: T) -> Self {
        // SAFETY: a union of plain-data fields has no validity requirement, so all-zero bytes are
        // a value; writing `value` then leaves the bytes past it zero.
        let mut slot: Self = unsafe { std::mem::zeroed() };
        slot.value = value;
        slot
    }

    /// The stored value.
    #[inline]
    pub fn get(self) -> T {
        // SAFETY: a slot is made by `new` (which writes a `T`) or read from an image, whose records
        // are plain data, valid for every bit pattern.
        unsafe { self.value }
    }
}

impl<T: Copy + std::fmt::Debug, W: Copy> std::fmt::Debug for Padded<T, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get().fmt(f)
    }
}

impl<T: Copy + PartialEq, W: Copy> PartialEq for Padded<T, W> {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<T: Copy + Eq, W: Copy> Eq for Padded<T, W> {}

// SAFETY: a union of two plain-data fields; its bytes are those of the wider one (`NO_PADDING`
// checks the union adds no tail of its own).
unsafe impl<T: ImageRecord, W: Copy> ImageRecord for Padded<T, W> {
    const FIELD_BYTES: usize = if T::FIELD_BYTES > std::mem::size_of::<W>() {
        T::FIELD_BYTES
    } else {
        std::mem::size_of::<W>()
    };
}

// SAFETY: an array of index words (e.g. `min_plus::network::Mat4`) is plain data with no padding.
unsafe impl<I: GraphIndexInteger, const N: usize> ImageRecord for [I; N] {
    const FIELD_BYTES: usize = N * std::mem::size_of::<I>();
}

macro_rules! primitive_records {
    ($($ty:ty),*) => {$(
        // SAFETY: a primitive integer.
        unsafe impl ImageRecord for $ty {
            const FIELD_BYTES: usize = std::mem::size_of::<$ty>();
        }
    )*};
}
primitive_records!(u8, u16, u32, u64);

// The `repr(transparent)` index-word newtypes.
image_record!(impl<I> NodeIndex<I> = [I]);
image_record!(impl<I> DirectedNodeIndex<I> = [I]);
image_record!(impl<I> OptionalNodeIndex<I> = [I]);
image_record!(impl<I> BlockIndex<I> = [I]);
image_record!(impl<I> GfaPathLength<I> = [I]);
image_record!(impl<I> MinPlus<I> = [I]);

// SAFETY: `repr(transparent)` over `[[MinPlus<I>; N]; N]`.
unsafe impl<I: GraphIndexInteger, const N: usize> ImageRecord for MinPlusMatrix<I, N> {
    const FIELD_BYTES: usize = N * N * std::mem::size_of::<I>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_slots_zero_their_spare_bytes() {
        let slot = Padded::<u32, u64>::new(0xab);
        assert_eq!(std::mem::size_of_val(&slot), 8);
        // SAFETY: every byte of the slot is initialised.
        let bytes: [u8; 8] = unsafe { std::mem::transmute(slot) };
        assert_eq!(u64::from_ne_bytes(bytes), 0xab);
        assert_eq!(slot.get(), 0xab);
        assert_eq!(std::mem::size_of::<Padded<u32, u32>>(), 4);
        assert_eq!(std::mem::size_of::<Padded<u32, u64>>(), 8);
        assert_eq!(std::mem::align_of::<Padded<u32, u64>>(), 8);
        let () = <Padded<u32, u64> as ImageRecord>::NO_PADDING;
        let () = <Padded<u64, u32> as ImageRecord>::NO_PADDING;
    }

    /// At word size 32 the records with `u32` fields keep their plain layout (the golden images pin
    /// it), and at word size 64 every field takes a whole word.
    #[test]
    fn mixed_records_keep_their_layout() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<crate::chain_nav::CutRec<u32>>(), 24);
        assert_eq!(size_of::<crate::topology::SpqrLoc<u32>>(), 12);
        assert_eq!(size_of::<crate::skeleton_index::LabelEntry<u32>>(), 8);
        assert_eq!(size_of::<super::super::hash::NodeSlot<u32>>(), 8);
        assert_eq!(size_of::<crate::s_node_segtree::SegVertexRec<u32>>(), 8);
        assert_eq!(size_of::<crate::s_node_segtree::PfxVertex<u32>>(), 28);
        assert_eq!(size_of::<crate::prefix_lift::IrregularStep<u32>>(), 20);
        assert_eq!(align_of::<crate::chain_nav::CutRec<u32>>(), 4);
        assert_eq!(align_of::<crate::s_node_segtree::PfxVertex<u32>>(), 4);
        assert_eq!(size_of::<crate::chain_nav::CutRec<u64>>(), 48);
        assert_eq!(size_of::<crate::topology::SpqrLoc<u64>>(), 16);
        assert_eq!(size_of::<crate::skeleton_index::LabelEntry<u64>>(), 16);
        assert_eq!(size_of::<super::super::hash::NodeSlot<u64>>(), 16);
        assert_eq!(size_of::<crate::s_node_segtree::SegVertexRec<u64>>(), 16);
        assert_eq!(size_of::<crate::s_node_segtree::PfxVertex<u64>>(), 56);
        assert_eq!(size_of::<crate::prefix_lift::IrregularStep<u64>>(), 40);
    }
}
