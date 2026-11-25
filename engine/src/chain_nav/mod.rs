//! Navigation of the block-cut tree over its chain layout, and the per-cut, per-block and
//! per-chain records it reads.
//!
//! # Layout
//!
//! The block-cut tree (blocks, cut nodes, one synthetic super-root) is decomposed into heavy-path
//! chains ([`ChainLayout`], `layout.rs`). Chains are numbered parent-before-child (by the depth of
//! their head, ties by id) and every tree node gets a **tree position** `tpos`: the chains laid out
//! one after the other, each from its head down. Along a chain blocks and cut nodes alternate, so a
//! chain's cut nodes sit at every second position from its first cut on. Dense cut ids follow
//! `tpos`, so a cut's record is reached by its id and its position is arithmetic on its chain's
//! [`ChainRec`]. Blocks are in the chain numbering of
//! [`chain_block_numbering`](crate::topology::chain_block_numbering): cut `c`'s heavy child is
//! block `roots + c`, so only the other blocks keep a [`HeadBlockRec`] with their position.
//!
//! # Navigation
//!
//! Two facts make every navigation query arithmetic plus a per-chain lookup ([`ChainNav`]):
//! - a node's ancestors on its own chain are the smaller positions, and
//! - chain ids grow along any root path (a chain attaches to a chain with a smaller id), so `tpos`
//!   does too: for two nodes on one root path, `depth[x] <= depth[y]` iff `tpos(x) <= tpos(y)`.
//!
//! The LCA lifts the node on the larger chain id to its chain's attach node until both are on one
//! chain, then takes the smaller position. Parent, heavy child, depth and the nearest exception
//! above a cut are position arithmetic on the chain record.
//!
//! # Records
//!
//! `records.rs` holds one [`CutRec`] per cut (the topology writes its structural fields, the lift
//! the prefix lift's), one word per block ([`block_bits`]), a [`HeadBlockRec`] per block that is no
//! cut's heavy child, a [`ChainRec`] per chain, the child lists of the cuts in three or more blocks
//! ([`CutExtraDir`]) and a [`SpecialBlockRec`] per block whose cut count is not 2 (or root block).
//! `rank_bits.rs` is the O(1)-rank set ([`RankBits`]) these lists are indexed through.

mod layout;
mod rank_bits;
mod records;
#[cfg(test)]
mod tests;

pub use layout::ChainLayout;
pub use rank_bits::{RankBits, RankBlock, rank_bits};
pub use records::{
    ChainRec, CutExtraDir, CutRec, HeadBlockRec, SpecialBlockRec, block_bits, cut_flags,
    special_index_by_block,
};

use bidirected_adjacency_array::index::GraphIndexInteger;
use spqr_tree::decomposition::indices::BlockIndex;

/// "None" in the `u32` id fields.
pub const NONE: u32 = u32::MAX;

/// Query-time navigation over the persisted records (see the module docs). Tree ids as in the
/// lift: `0..nb` blocks, `nb..nb+nc` cut nodes (dense ids), `nb+nc` the super-root.
#[derive(Clone, Copy)]
pub struct ChainNav<'a, IndexType: GraphIndexInteger> {
    pub nb: usize,
    pub nc: usize,
    /// The chain block numbering (see `crate::topology::chain_block_numbering`): the component
    /// roots are blocks `0..roots` and cut `c`'s heavy child is block `roots + c`.
    pub roots: usize,
    pub cuts: &'a [CutRec<IndexType>],
    /// Per block: its [`block_bits`] word.
    pub block_words: &'a [u32],
    /// The blocks that are no cut's heavy child (see [`Self::head_index`]).
    pub heads: &'a [HeadBlockRec],
    /// Over cut ids: whose heavy child is a special block; the rank is that block's index in the
    /// special records (they are sorted by block, so the heavy children's are one run).
    pub heavy_special: RankBits<'a>,
    pub chains: &'a [ChainRec],
    /// The child lists of the cut nodes in three or more blocks; over cut ids, which cuts have one
    /// (the rank is the cut's index in `extra_dir`), and where each list starts.
    pub extras: &'a [BlockIndex<IndexType>],
    pub extra_rank: RankBits<'a>,
    pub extra_dir: &'a [CutExtraDir],
    /// The lift's exception cuts, grouped per chain (see [`ChainRec::exc_start`]).
    pub exc: &'a [u32],
}

impl<'a, IndexType: GraphIndexInteger> ChainNav<'a, IndexType> {
    #[inline]
    pub fn is_cut(&self, id: usize) -> bool {
        id >= self.nb && id < self.nb + self.nc
    }

    #[inline]
    pub fn super_root(&self) -> usize {
        self.nb + self.nc
    }

    /// The chain containing cut `c` (from its record).
    #[inline]
    pub fn chain_of_cut(&self, c: usize) -> usize {
        (self.cuts[c].flags.get() >> cut_flags::CHAIN_SHIFT) as usize
    }

    /// The cut whose heavy child block `b` is, if it is one.
    #[inline]
    fn heavy_parent_cut(&self, b: usize) -> Option<usize> {
        (b >= self.roots && b < self.roots + self.nc).then(|| b - self.roots)
    }

    /// The index of a block that is no cut's heavy child in [`Self::heads`]: the root blocks
    /// `0..roots` first, then the blocks after the heavy children.
    #[inline]
    pub fn head_index(roots: usize, nc: usize, b: usize) -> usize {
        debug_assert!(
            b < roots || b >= roots + nc,
            "a heavy child has no head record"
        );
        if b < roots { b } else { b - nc }
    }

    /// The chain containing block `b`.
    #[inline]
    pub fn chain_of_block(&self, b: usize) -> usize {
        match self.heavy_parent_cut(b) {
            Some(c) => self.chain_of_cut(c),
            None => self.heads[Self::head_index(self.roots, self.nc, b)].chain as usize,
        }
    }

    /// Tree position of block `b`.
    #[inline]
    pub fn block_tpos(&self, b: usize) -> u32 {
        match self.heavy_parent_cut(b) {
            Some(c) => self.tpos_of_cut(c) + 1,
            None => self.heads[Self::head_index(self.roots, self.nc, b)].tpos,
        }
    }

    /// The index of special block `b`'s [`SpecialBlockRec`]: in its head record, or for a cut's heavy
    /// child the rank among the special heavy children.
    #[inline]
    pub fn special_index(&self, b: usize) -> usize {
        debug_assert!(
            self.block_words[b] & block_bits::SPECIAL != 0,
            "block {b} is not special"
        );
        match self.heavy_parent_cut(b) {
            Some(c) => self.heavy_special.rank(c),
            None => self.heads[Self::head_index(self.roots, self.nc, b)].special as usize,
        }
    }

    /// The parent block of cut `c`: the heavy child of the cut before it on its chain, else its
    /// chain's first block (a block- or root-headed chain) or attach node (a cut-headed chain).
    #[inline]
    pub fn parent_block_of_cut(&self, c: usize) -> usize {
        if self.cuts[c].flags.get() & cut_flags::FIRST_IN_CHAIN == 0 {
            return self.roots + c - 1;
        }
        let k = &self.chains[self.chain_of_cut(c)];
        if k.cut_off == 0 {
            k.attach_id as usize
        } else {
            k.first_block_id as usize
        }
    }

    /// The chain containing a tree id.
    #[inline]
    pub fn chain_of(&self, id: usize) -> usize {
        if id < self.nb {
            self.chain_of_block(id)
        } else if id < self.nb + self.nc {
            self.chain_of_cut(id - self.nb)
        } else {
            0
        }
    }

    /// Tree position of cut `c`.
    #[inline]
    pub fn tpos_of_cut(&self, c: usize) -> u32 {
        let k = &self.chains[self.chain_of_cut(c)];
        k.tpos_start + k.cut_off + 2 * (c as u32 - k.cut_start)
    }

    /// Tree position of a tree id.
    #[inline]
    pub fn tpos(&self, id: usize) -> u32 {
        if id < self.nb {
            self.block_tpos(id)
        } else if id < self.nb + self.nc {
            self.tpos_of_cut(id - self.nb)
        } else {
            0
        }
    }

    /// The dense cut id at tree position `t` of chain `k`, if that position holds a cut node.
    #[inline]
    fn cut_at(&self, k: &ChainRec, t: u32) -> Option<usize> {
        let o = t - k.tpos_start;
        (o >= k.cut_off && (o - k.cut_off).is_multiple_of(2))
            .then(|| (k.cut_start + (o - k.cut_off) / 2) as usize)
    }

    /// Depth in the bipartite tree (super-root `0`).
    #[inline]
    pub fn depth(&self, id: usize) -> u32 {
        let k = &self.chains[self.chain_of(id)];
        k.head_depth + (self.tpos(id) - k.tpos_start)
    }

    /// Depth of cut `c` in the cut forest (a cut under a component root block has depth `0`).
    #[inline]
    pub fn cut_depth(&self, c: usize) -> u32 {
        (self.depth(self.nb + c) - 2) / 2
    }

    /// The parent of a tree id (`None` for the super-root).
    #[inline]
    pub fn parent(&self, id: usize) -> Option<usize> {
        if id >= self.nb + self.nc {
            return None;
        }
        if id >= self.nb {
            return Some(self.parent_block_of_cut(id - self.nb));
        }
        let t = self.block_tpos(id);
        let k = &self.chains[self.chain_of_block(id)];
        if t == k.tpos_start {
            return Some(k.attach_id as usize);
        }
        if t == 1 {
            return Some(self.super_root());
        }
        let c = self
            .cut_at(k, t - 1)
            .expect("a block is preceded by a cut node on its chain");
        Some(self.nb + c)
    }

    /// The heavy child of a tree id (`None` at a leaf).
    #[inline]
    pub fn heavy(&self, id: usize) -> Option<usize> {
        if id < self.nb {
            let t = self.block_tpos(id);
            let k = &self.chains[self.chain_of_block(id)];
            (t + 1 < k.tpos_start + k.len).then(|| {
                self.nb
                    + self
                        .cut_at(k, t + 1)
                        .expect("a block is followed by a cut node on its chain")
            })
        } else if id < self.nb + self.nc {
            self.heavy_child_of_cut(id - self.nb)
        } else {
            let k = &self.chains[0];
            (k.len > 1).then_some(k.first_block_id as usize)
        }
    }

    /// All child blocks of cut `c`, heavy child first.
    #[inline]
    pub fn cut_child_blocks(
        &self,
        c: usize,
    ) -> impl Iterator<Item = BlockIndex<IndexType>> + use<'a, IndexType> {
        let list: &'a [BlockIndex<IndexType>] =
            if self.cuts[c].flags.get() & cut_flags::HAS_EXTRA != 0 {
                // A cut in three or more blocks: its list, through its rank among them.
                let start = self.extra_dir[self.extra_rank.rank(c)].start as usize;
                let n = self.extras[start].into_usize();
                &self.extras[start + 1..start + 1 + n]
            } else {
                &[]
            };
        let heavy = list
            .is_empty()
            .then(|| BlockIndex::from_usize(self.roots + c));
        heavy.into_iter().chain(list.iter().copied())
    }

    /// The heavy child block of cut `c` (the next node on its chain; every cut node has one).
    #[inline]
    pub fn heavy_child_of_cut(&self, c: usize) -> Option<usize> {
        Some(self.roots + c)
    }

    /// The lowest common ancestor of two tree ids.
    #[inline]
    pub fn lca(&self, a: usize, b: usize) -> usize {
        // `(position, tree id, chain)`; the node on the larger chain id is lifted to its chain's
        // attach node until both sit on one chain.
        let (mut a, mut b) = (
            (self.tpos(a), a, self.chain_of(a)),
            (self.tpos(b), b, self.chain_of(b)),
        );
        loop {
            if a.2 == b.2 {
                return if a.0 <= b.0 { a.1 } else { b.1 };
            }
            if a.2 > b.2 {
                let k = &self.chains[a.2];
                a = (k.attach_tpos, k.attach_id as usize, k.attach_chain as usize);
            } else {
                let k = &self.chains[b.2];
                b = (k.attach_tpos, k.attach_id as usize, k.attach_chain as usize);
            }
        }
    }

    /// The child of `ancestor` on the path towards `descendant`, or `None` if `descendant` is not a
    /// strict descendant of `ancestor`.
    pub fn child_on_path_to(&self, ancestor: usize, descendant: usize) -> Option<usize> {
        if ancestor == descendant {
            return None;
        }
        let (ta, ka) = (self.tpos(ancestor), self.chain_of(ancestor));
        let (mut td, mut kd) = (self.tpos(descendant), self.chain_of(descendant));
        loop {
            if kd == ka {
                return (td > ta).then(|| {
                    self.heavy(ancestor)
                        .expect("an ancestor on the chain has a heavy child")
                });
            }
            if kd < ka {
                return None;
            }
            let k = &self.chains[kd];
            if k.attach_tpos == ta {
                return Some(k.head_id as usize);
            }
            td = k.attach_tpos;
            kd = k.attach_chain as usize;
        }
    }

    /// The nearest exception cut at or above cut `c` on its root path (`NONE` if none).
    #[inline]
    pub fn irr_above(&self, c: usize) -> u32 {
        let k = &self.chains[self.chain_of_cut(c)];
        let e = &self.exc[k.exc_start as usize..(k.exc_start + k.exc_count) as usize];
        let i = e.partition_point(|&x| x as usize <= c);
        if i > 0 { e[i - 1] } else { k.exc_inherit }
    }

    /// The parent cut of cut `c` (the parent cut of its parent block), `None` under a root block.
    #[inline]
    pub fn parent_cut(&self, c: usize) -> Option<usize> {
        let block = self.parent_block_of_cut(c);
        let p = self.parent(block)?;
        self.is_cut(p).then(|| p - self.nb)
    }
}
