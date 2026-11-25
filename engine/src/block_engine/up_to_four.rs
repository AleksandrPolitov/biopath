//! A list of at most four values on the stack: the seeds of an LCA read and their per-oracle ids.
//! A source seeds the LCA's skeleton at itself or at one child's two poles in both orientations, so
//! four always suffice, and a query allocates nothing for them.

use std::ops::Deref;

/// At most four `T`s, in insertion order.
#[derive(Clone, Copy)]
pub(super) struct UpToFour<T: Copy> {
    items: [T; 4],
    len: usize,
}

impl<T: Copy> UpToFour<T> {
    /// An empty list; `filler` only fills the unused slots.
    pub(super) fn new(filler: T) -> Self {
        Self {
            items: [filler; 4],
            len: 0,
        }
    }

    /// The list of one item.
    pub(super) fn one(item: T) -> Self {
        let mut list = Self::new(item);
        list.len = 1;
        list
    }

    /// The items of `iter` (at most four); `filler` only fills the unused slots.
    pub(super) fn collect(filler: T, iter: impl IntoIterator<Item = T>) -> Self {
        let mut list = Self::new(filler);
        for item in iter {
            list.push(item);
        }
        list
    }

    /// Append `item`.
    ///
    /// # Panics
    /// Panics if the list already holds four items.
    pub(super) fn push(&mut self, item: T) {
        self.items[self.len] = item;
        self.len += 1;
    }

    /// Append `item` unless the list is full; whether it was appended.
    pub(super) fn try_push(&mut self, item: T) -> bool {
        if self.len == self.items.len() {
            return false;
        }
        self.push(item);
        true
    }
}

impl<T: Copy> Deref for UpToFour<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.items[..self.len]
    }
}
