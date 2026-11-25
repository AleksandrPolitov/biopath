//! A block's cut list and flip list as the query reads them: from the block word of a plain block
//! (two cuts, not a component root), from its special record otherwise.

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::BlockCutLift;
use crate::chain_nav::{SpecialBlockRec, block_bits};

/// A block's cut list as dense cut ids, in its `M_aug` / node-row order: two entries for a plain
/// block, the special table's list otherwise.
#[derive(Clone, Copy)]
pub(super) enum CutList<'a> {
    Two([u32; 2]),
    Many(&'a [u32]),
}

impl CutList<'_> {
    #[inline]
    pub(super) fn len(&self) -> usize {
        match self {
            CutList::Two(_) => 2,
            CutList::Many(s) => s.len(),
        }
    }

    #[inline]
    pub(super) fn get(&self, j: usize) -> u32 {
        match self {
            CutList::Two(a) => a[j],
            CutList::Many(s) => s[j],
        }
    }

    #[inline]
    pub(super) fn position(&self, cut: u32) -> Option<usize> {
        (0..self.len()).find(|&j| self.get(j) == cut)
    }
}

/// A block's flip-capable cuts (dense ids) in its cut-list order.
#[derive(Clone, Copy)]
pub(super) enum FlipList<'a> {
    Few([u32; 2], usize),
    Many(&'a [u32]),
}

impl FlipList<'_> {
    #[inline]
    pub(super) fn as_slice(&self) -> &[u32] {
        match self {
            FlipList::Few(a, n) => &a[..*n],
            FlipList::Many(s) => s,
        }
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl<'a, IndexType: GraphIndexInteger + 'static> BlockCutLift<'a, IndexType> {
    /// The special-block record and cut list of `block` (a block whose cut count is not 2, or a
    /// root block).
    #[inline]
    pub(super) fn special(&self, block: usize) -> (&'a SpecialBlockRec, &'a [u32]) {
        let rec = &self.special_recs[self.nav.special_index(block)];
        debug_assert_eq!(
            rec.block as usize, block,
            "a special block's record is its own"
        );
        (
            rec,
            &self.special_cuts[rec.cut_start as usize..(rec.cut_start + rec.cut_count) as usize],
        )
    }

    /// The block's cut list (its `M_aug` row/column order) as dense cut ids.
    #[inline]
    pub(super) fn block_cut_list(&self, block: usize) -> CutList<'a> {
        let bits = self.block_words[block];
        if bits & block_bits::SPECIAL != 0 {
            return CutList::Many(self.special(block).1);
        }
        let parent = self
            .nav
            .parent(block)
            .expect("a plain block has a parent cut")
            - self.number_of_blocks;
        let child = self
            .nav
            .heavy(block)
            .expect("a plain block has a child cut")
            - self.number_of_blocks;
        if bits & block_bits::PARENT_POS == 0 {
            CutList::Two([parent as u32, child as u32])
        } else {
            CutList::Two([child as u32, parent as u32])
        }
    }

    /// The flip-capable cut nodes of a block (dense ids, in its cut-list order): those that can
    /// flip via a route LEAVING the block. Only these can make `same_block_cross` beat the direct
    /// intra-block distance.
    #[inline]
    pub(super) fn flip_list(&self, block: usize) -> FlipList<'a> {
        let bits = self.block_words[block];
        if bits & block_bits::SPECIAL != 0 {
            let rec = self.special(block).0;
            return FlipList::Many(
                &self.special_flips
                    [rec.flip_start as usize..(rec.flip_start + rec.flip_count) as usize],
            );
        }
        let list = self.block_cut_list(block);
        let parent = (self
            .nav
            .parent(block)
            .expect("a plain block has a parent cut")
            - self.number_of_blocks) as u32;
        let mut out = [0u32; 2];
        let mut n = 0usize;
        for j in 0..2 {
            let cut = list.get(j);
            let flag = if cut == parent {
                block_bits::FLIP_PARENT
            } else {
                block_bits::FLIP_CHILD
            };
            if bits & flag != 0 {
                out[n] = cut;
                n += 1;
            }
        }
        FlipList::Few(out, n)
    }
}
