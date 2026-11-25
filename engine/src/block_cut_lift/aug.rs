//! The augmented cut-to-cut matrices `M_aug`: per block, the distances between its directed cuts
//! with every cut's whole-graph flip closure folded in as a self-loop (the closure of its gadget,
//! see [`super::gadget`]). Built for every block; the image keeps the dense matrix only of a block
//! whose `M_aug` does not derive from the prefix lift (a regular 2-cut block's does: its child step
//! and its reversal, and the flip closures on the diagonal).

use std::collections::HashMap;

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};

use super::{
    BlockCutLift, forest::BlockCutForest, gadget::gadget_closure, sections::AUG_SORTED_MIN_CUTS,
};
use crate::{
    chain_nav::{self, block_bits},
    min_plus::{Mat2, MinPlus},
};

/// The `M_aug` of one block (build side).
pub(super) struct BlockAug<IndexType: GraphIndexInteger> {
    /// The block's cuts, in its cut-list order.
    pub(super) cuts: Vec<NodeIndex<IndexType>>,
    /// The position of each cut in `cuts`.
    pub(super) local: HashMap<NodeIndex<IndexType>, usize>,
    /// `aug[(2*i+oi) * dim + (2*j+oj)] = D_aug(cuts[i]^oi -> cuts[j]^oj)`, `dim = 2 * cuts.len()`.
    pub(super) aug: Vec<MinPlus<IndexType>>,
}

/// Every block's `M_aug`: its raw `M_B` (consumed) closed with each cut's whole-graph flip
/// closure `full[c]` as a self-loop. `positions` are the blocks' cut positions
/// ([`BlockCutForest::cut_positions`]), kept for [`BlockAug::matrix`].
pub(super) fn block_augs<IndexType: GraphIndexInteger>(
    forest: &BlockCutForest<IndexType>,
    raw: Vec<Vec<MinPlus<IndexType>>>,
    positions: Vec<HashMap<NodeIndex<IndexType>, usize>>,
    full: &[Mat2<IndexType>],
) -> Vec<BlockAug<IndexType>> {
    raw.into_iter()
        .zip(positions)
        .zip(&forest.block_cuts)
        .map(|((raw, local), cuts)| {
            let loops: Vec<(usize, Mat2<IndexType>)> = cuts
                .iter()
                .enumerate()
                .map(|(i, &segment)| (i, full[forest.cut_id[&segment]]))
                .collect();
            BlockAug {
                cuts: cuts.clone(),
                local,
                aug: gadget_closure(&raw, &loops),
            }
        })
        .collect()
}

impl<IndexType: GraphIndexInteger> BlockAug<IndexType> {
    fn dim(&self) -> usize {
        2 * self.cuts.len()
    }

    /// The `2x2` matrix `D_aug(from^* -> to^*)` between two of the block's cuts.
    pub(super) fn matrix(
        &self,
        from: NodeIndex<IndexType>,
        to: NodeIndex<IndexType>,
    ) -> Mat2<IndexType> {
        Mat2::block_of(&self.aug, self.dim(), self.local[&from], self.local[&to])
    }
}

impl<'a, IndexType: GraphIndexInteger + 'static> BlockCutLift<'a, IndexType> {
    /// Position of a cut (dense id) in the block's cut list: a scan of the list for the usual few
    /// cuts, the block's sorted `(cut, pos)` list above [`AUG_SORTED_MIN_CUTS`].
    #[inline]
    pub(super) fn aug_local(&self, block: usize, cut: usize) -> usize {
        let list = self.block_cut_list(block);
        if list.len() <= AUG_SORTED_MIN_CUTS {
            return list
                .position(cut as u32)
                .expect("cut not in the block's M_aug");
        }
        let d = self
            .aug_sorted_dir
            .binary_search_by_key(&(block as u32), |d| d.block)
            .expect("a many-cut block has a sorted cut list");
        let d = self.aug_sorted_dir[d];
        let sorted = &self.aug_sorted[d.start as usize..(d.start + d.count) as usize];
        let i = sorted
            .binary_search_by_key(&self.cut_node(cut), |e| e.cut)
            .expect("cut segment not in the block's M_aug");
        sorted[i].pos.get() as usize
    }

    /// The block's row-major dense `M_aug` (`dim = 2 * cuts`); only for blocks that keep one: a
    /// special block's record names its data, a plain 2-cut block with an irregular step (the rare
    /// non-derived plain block) is in the dense directory.
    #[inline]
    pub(super) fn aug_dense(&self, block: usize) -> &'a [MinPlus<IndexType>] {
        if self.block_words[block] & block_bits::SPECIAL != 0 {
            let (rec, cuts) = self.special(block);
            debug_assert_ne!(
                rec.aug_start,
                chain_nav::NONE,
                "a special block keeps a dense M_aug"
            );
            let start = rec.aug_start as usize;
            let dim = 2 * cuts.len();
            return &self.aug_dense_data[start..start + dim * dim];
        }
        let i = self
            .aug_dense_dir
            .binary_search_by_key(&(block as u32), |d| d.block)
            .expect("a plain block with a dense M_aug is in the directory");
        let start = self.aug_dense_dir[i].start as usize;
        &self.aug_dense_data[start..start + 16]
    }

    /// The `2x2` matrix `D_aug(from^* -> to^*)` of a block whose `M_aug` is derived (a regular
    /// 2-cut block under the prefix lift): its child step and reversal from the prefix, a cut to
    /// itself from its flip closure (zero diagonal, as the gadget closure has).
    fn derived_aug_matrix(&self, block: usize, from: usize, to: usize) -> Mat2<IndexType> {
        if from == to {
            return self.full(from);
        }
        let child = self
            .nav
            .heavy(block)
            .expect("a derived block has a child cut")
            - self.number_of_blocks;
        let prefix = self.prefix();
        if from == child {
            prefix.step_up_matrix(child)
        } else {
            prefix.step_down_matrix(child)
        }
    }

    /// Whether a block's `M_aug` is derived (no dense data written).
    #[inline]
    pub(super) fn aug_is_derived(&self, block: usize) -> bool {
        self.block_words[block] & block_bits::DERIVED != 0
    }

    /// The dense `4x4` `M_aug` of a derived block, materialized (for the row-less same-block path).
    pub(super) fn derived_aug_dense(&self, block: usize) -> Vec<MinPlus<IndexType>> {
        let list = self.block_cut_list(block);
        let mut data = vec![MinPlus::infinity(); 16];
        for i in 0..list.len() {
            for j in 0..list.len() {
                let m = self.derived_aug_matrix(block, list.get(i) as usize, list.get(j) as usize);
                for oi in 0..2 {
                    for oj in 0..2 {
                        data[(2 * i + oi) * 4 + (2 * j + oj)] = m.get(oi, oj);
                    }
                }
            }
        }
        data
    }

    /// The `2x2` matrix `D_aug(from^* -> to^*)` between two of the block's cuts (dense ids).
    pub(super) fn aug_matrix(&self, block: usize, from: usize, to: usize) -> Mat2<IndexType> {
        if self.aug_is_derived(block) {
            return self.derived_aug_matrix(block, from, to);
        }
        let dim = 2 * self.block_cut_list(block).len();
        let data = self.aug_dense(block);
        let fi = self.aug_local(block, from);
        let ti = self.aug_local(block, to);
        Mat2::block_of(data, dim, fi, ti)
    }
}
