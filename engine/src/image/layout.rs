//! Relative sub-layouts inside one record: a sequence of typed arrays packed back to back with
//! natural alignment, where every array's offset is *derived from the counts* in the record header
//! rather than stored. The writer and the reader run the same [`Layout`] arithmetic, so a record
//! carries no per-array offsets (a few dozen bytes saved on each of millions of tiny blocks).

/// Round `value` up to a multiple of `align` (a power of two).
#[inline]
pub const fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// A running byte cursor over one record. Call [`Layout::field`] once per array, in a fixed order,
/// on both the writing and the reading side.
#[derive(Debug, Clone, Copy, Default)]
pub struct Layout {
    pos: usize,
}

impl Layout {
    /// Start at relative offset `0`.
    pub const fn new() -> Self {
        Self { pos: 0 }
    }

    /// Reserve `len` elements of `T` (aligned to `T`, at least 4 bytes); returns the array's relative
    /// byte offset.
    #[inline]
    pub fn field<T>(&mut self, len: usize) -> usize {
        let align = std::mem::align_of::<T>().max(4);
        let start = align_up(self.pos, align);
        self.pos = start + len * std::mem::size_of::<T>();
        start
    }

    /// Skip `bytes` raw bytes (an opaque blob); returns its relative offset (8-aligned).
    #[inline]
    pub fn blob(&mut self, bytes: usize) -> usize {
        let start = align_up(self.pos, 8);
        self.pos = start + bytes;
        start
    }

    /// Total bytes laid out so far, rounded up to 8 so consecutive records stay aligned.
    #[inline]
    pub fn total(&self) -> usize {
        align_up(self.pos, 8)
    }
}
