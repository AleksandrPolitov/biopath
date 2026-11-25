//! Extensions of the GFA node data: a node's length ([`GfaNodeDataExt`]) and node data that keeps
//! only the name and the length ([`LengthOnlyGfaNodeData`]).

use std::borrow::Cow;

use bidirected_adjacency_array::{
    index::GraphIndexInteger,
    io::gfa1::{GfaNodeData, PlainGfaNodeData},
};

use crate::path::GfaPathLength;

pub trait GfaNodeDataExt<IndexType: GraphIndexInteger> {
    fn len(&self) -> GfaPathLength<IndexType>;

    fn is_empty(&self) -> bool {
        self.len().into_usize() == 0
    }
}

impl<T: GfaNodeData, IndexType: GraphIndexInteger> GfaNodeDataExt<IndexType> for T {
    fn len(&self) -> GfaPathLength<IndexType> {
        GfaPathLength::from_usize(self.sequence().len())
    }
}

/// GFA node data that keeps the node **name** and the sequence **length**, but not the sequence.
///
/// The distance index and the whole-graph Dijkstra only ever read a node's length (via
/// [`GfaNodeDataExt::len`]) and name, so holding every sequence byte for the process lifetime is pure
/// overhead: `PlainGfaNodeData` costs 48 inline bytes plus two heap allocations per node. This type
/// is 32 inline bytes plus one allocation (the name).
///
/// [`GfaNodeData::sequence`] still has to return a string of the right length, because the length is
/// derived from it. It returns a slice of a shared, process-wide run of NUL bytes, so **the sequence
/// content is not available** — only its length is correct. Use `PlainGfaNodeData` wherever the
/// bases themselves are needed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LengthOnlyGfaNodeData {
    name: Box<str>,
    /// `len` bytes of the shared placeholder, so `sequence()` is a borrow with no global access.
    placeholder: &'static str,
}

/// A `len`-byte slice of a process-wide run of NUL bytes. The run only grows: a longer request leaks a
/// new buffer of at least twice the size, and the old ones stay alive (leaked), so every slice handed
/// out remains valid. Total leaked memory is at most twice the longest node; the buffer is allocated
/// zeroed and never written, so a large one (a merged node of a shrunk graph, megabases long) stays
/// untouched virtual memory. Called once per node while a graph is built, so the lock is uncontended.
fn length_placeholder(len: usize) -> &'static str {
    static RUN: std::sync::Mutex<&'static str> = std::sync::Mutex::new("");
    let mut run = RUN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if run.len() < len {
        let grown = len.max(2 * run.len()).max(4096);
        let bytes: &'static [u8] = Box::leak(vec![0u8; grown].into_boxed_slice());
        // SAFETY: NUL bytes are valid UTF-8.
        *run = unsafe { std::str::from_utf8_unchecked(bytes) };
    }
    &run[..len]
}

impl LengthOnlyGfaNodeData {
    /// A node with the given name and sequence length (e.g. a merged node of a shrunk graph).
    pub fn with_len(name: &str, len: usize) -> Self {
        Self {
            name: name.into(),
            placeholder: length_placeholder(len),
        }
    }
}

impl From<PlainGfaNodeData> for LengthOnlyGfaNodeData {
    fn from(node: PlainGfaNodeData) -> Self {
        Self {
            name: node.name().into(),
            placeholder: length_placeholder(node.sequence().len()),
        }
    }
}

impl GfaNodeData for LengthOnlyGfaNodeData {
    #[inline]
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    /// A placeholder (NUL bytes) with the node's true length (see the type docs).
    #[inline]
    fn sequence(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.placeholder)
    }
}

#[cfg(test)]
mod length_only_tests {
    use super::*;

    #[test]
    fn length_only_node_keeps_name_and_length() {
        let short = LengthOnlyGfaNodeData::from(PlainGfaNodeData::new("s1", "ACGT"));
        assert_eq!(short.name(), "s1");
        assert_eq!(short.sequence().len(), 4);
        assert_eq!(GfaNodeDataExt::<u32>::len(&short).into_usize(), 4);

        // Longer than any earlier request: the shared run grows and old slices stay valid.
        let long_seq = "A".repeat(100_000);
        let long = LengthOnlyGfaNodeData::from(PlainGfaNodeData::new("s2", &long_seq));
        assert_eq!(long.sequence().len(), 100_000);
        assert_eq!(short.sequence().len(), 4);
        assert_eq!(
            LengthOnlyGfaNodeData::from(PlainGfaNodeData::new("s3", ""))
                .sequence()
                .len(),
            0
        );
        assert!(
            std::mem::size_of::<LengthOnlyGfaNodeData>() < std::mem::size_of::<PlainGfaNodeData>()
        );
    }
}
