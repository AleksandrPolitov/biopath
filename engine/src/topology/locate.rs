//! Where a node sits in a record block's SPQR tree ([`SpqrLocation`]), inline in its record or in
//! the out-of-line list.

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};
use spqr_tree::decomposition::indices::BlockIndex;

use super::{
    NodeTopology, offset,
    record::{K_CUT, K_RECORD},
};
use crate::{image::image_record, rooted_tree::TreeAccess};

/// A record-block node's block and packed SPQR location (inline, or pointing into the list of
/// [`SpqrLoc`]s) while the engine is built: the lift then copies both into the node's header in its
/// row table. Heap only, never in the image.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildLoc {
    pub block: u32,
    pub loc: u32,
}

/// Where a graph node sits in one record block's SPQR tree (block-local ids, the ids
/// [`crate::block_spqr_tree::BlockSpqrTree`] assigns, rooted at local `0`).
///
/// The SPQR nodes containing a vertex form a connected subtree. `x` is its deepest node (the last
/// of maximum depth in the decomposition's order) and `up` how many of `x`'s ancestors it contains,
/// so an ancestor-or-self `a` of `x` contains the node iff `depth(x) - depth(a) <= up`. The build
/// checks that equivalence against the decomposition's full list for every node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpqrLocation {
    pub x: usize,
    pub up: u32,
}

impl SpqrLocation {
    /// Whether the node lies in `ancestor`, an ancestor-or-self of [`Self::x`] (every LCA a query
    /// forms from `x` is one).
    #[inline]
    pub fn contains_ancestor(&self, tree: &impl TreeAccess, ancestor: usize) -> bool {
        debug_assert!(
            tree.is_ancestor(ancestor, self.x),
            "{ancestor} is not an ancestor of x = {}",
            self.x
        );
        tree.depth(self.x) - tree.depth(ancestor) <= self.up as usize
    }
}

/// One out-of-line SPQR location: a cut node has one per record block containing it, a non-cut
/// node one only when its location does not fit inline. A node's entries are contiguous; the top
/// bit of `up_last` marks its last one.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpqrLoc<IndexType: GraphIndexInteger> {
    pub block: BlockIndex<IndexType>,
    pub x: u32,
    pub up_last: u32,
}
image_record!(impl<I> SpqrLoc<I> = [BlockIndex<I>, u32, u32]);

/// Low bits of an inline location holding `x`; the high bits hold `up`.
pub(super) const LOC_X_BITS: u32 = 26;
pub(super) const LOC_X_MASK: u32 = (1 << LOC_X_BITS) - 1;
/// The high-bits value marking the low bits as the index of the node's first [`SpqrLoc`] instead.
pub(super) const LOC_LIST: u32 = u32::MAX >> LOC_X_BITS;
/// [`SpqrLoc::up_last`] bit closing a node's entries.
pub(super) const LOC_LAST: u32 = 1 << 31;

/// The packed location field ([`BuildLoc::loc`]): `up << LOC_X_BITS | x` when both fit
/// (`up < LOC_LIST`), else `LOC_LIST << LOC_X_BITS | first list entry` (`None` = needs the list).
pub(super) fn pack_loc(loc: SpqrLocation) -> Option<u32> {
    #[cfg(test)]
    if FORCE_LOC_LIST.get() {
        return None;
    }
    (loc.x <= LOC_X_MASK as usize && loc.up < LOC_LIST)
        .then_some(loc.up << LOC_X_BITS | loc.x as u32)
}

#[cfg(test)]
thread_local! {
    /// Tests: store every location out of line (exercises the list path for non-cut nodes, which
    /// only blocks beyond `2^LOC_X_BITS` SPQR nodes or reaches of `LOC_LIST` reach otherwise).
    pub(super) static FORCE_LOC_LIST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The location field pointing at list entry `first` (see [`pack_loc`]).
pub(super) fn pack_list(first: usize) -> u32 {
    LOC_LIST << LOC_X_BITS | list_index(first)
}

/// `first` as a list index of a location field (the all-ones index stays free).
pub(super) fn list_index(first: usize) -> u32 {
    assert!(
        first < LOC_X_MASK as usize,
        "the graph needs more than {} out-of-line SPQR-tree locations (one per record block a cut \
         node lies in, and one per node of a record block whose location does not fit inline). \
         This is a hard limit of the index: this graph cannot be indexed",
        LOC_X_MASK - 1
    );
    first as u32
}

/// [`SpqrLocation`] of a node from the `locals` of one block (in the decomposition's order) — the
/// deepest local, and how far the containing subtree reaches above it. Panics unless the locals form
/// ONE connected subtree (exactly one member whose parent is not a member: its top). Then the locals
/// on `x`'s root path are exactly `x` up to the top, so the depth test matches the list.
pub(super) fn location_of(
    tree: &impl TreeAccess,
    locals: &[u32],
    sorted: &mut Vec<u32>,
) -> SpqrLocation {
    let x = locals
        .iter()
        .map(|&l| l as usize)
        .max_by_key(|&l| tree.depth(l))
        .expect("a location needs at least one SPQR node");
    sorted.clear();
    sorted.extend_from_slice(locals);
    sorted.sort_unstable();
    sorted.dedup();
    let mut tops = sorted.iter().map(|&l| l as usize).filter(|&l| {
        tree.parent(l)
            .is_none_or(|p| sorted.binary_search(&(p as u32)).is_err())
    });
    let top = tops.next().expect("a finite set of tree nodes has a top");
    assert!(
        tops.next().is_none(),
        "NodeTopology: the SPQR nodes containing a vertex do not form a connected subtree (x = {x})"
    );
    SpqrLocation {
        x,
        up: offset(tree.depth(x) - tree.depth(top)),
    }
}

impl<'a, IndexType: GraphIndexInteger> NodeTopology<'a, IndexType> {
    /// A record-block node's block and packed SPQR location (see [`BuildLoc`]), from its header once
    /// the lift laid the rows out, else from the build-time table; `None` for any other node.
    #[inline]
    pub fn record_loc(&self, node: NodeIndex<IndexType>) -> Option<BuildLoc> {
        let record = &self.records[node.into_usize()];
        (record.kind() == K_RECORD).then(|| {
            if self.rows.is_empty() {
                self.build_locs[record.id as usize]
            } else {
                let h = record.id as usize;
                BuildLoc {
                    block: self.header_word(h, 0) as u32,
                    loc: self.header_word(h, 1) as u32,
                }
            }
        })
    }

    /// Where `node` sits in `block`'s SPQR tree; `None` when `block` has no record or does not
    /// contain `node`.
    #[inline]
    pub fn spqr_location(
        &self,
        node: NodeIndex<IndexType>,
        block: BlockIndex<IndexType>,
    ) -> Option<SpqrLocation> {
        let record = &self.records[node.into_usize()];
        let low = match record.kind() {
            K_CUT => {
                let c = record.id as usize;
                if !self.cut_loc.contains(c) {
                    return None;
                }
                self.cut_loc_starts[self.cut_loc.rank(c)] as usize
            }
            K_RECORD => {
                let entry = self
                    .record_loc(node)
                    .expect("a record-block node has a location");
                if entry.block as usize != block.into_usize() {
                    return None;
                }
                let field = entry.loc;
                let low = (field & LOC_X_MASK) as usize;
                if field >> LOC_X_BITS != LOC_LIST {
                    return Some(SpqrLocation {
                        x: low,
                        up: field >> LOC_X_BITS,
                    });
                }
                low
            }
            _ => return None,
        };
        for entry in &self.spqr_locs[low..] {
            if entry.block == block {
                return Some(SpqrLocation {
                    x: entry.x as usize,
                    up: entry.up_last & !LOC_LAST,
                });
            }
            if entry.up_last & LOC_LAST != 0 {
                break;
            }
        }
        None
    }
}
