//! The node topology: what the engine reads about nodes, cut nodes and blocks once the per-block
//! SPQR trees exist, written into the image so that the decomposition can be dropped before the
//! per-block DP.
//!
//! # Node records
//!
//! Every graph node has one 8-byte [`NodeRecord`] (`record.rs`; a sentinel follows the last): its
//! length, a long one escaping to a rank-indexed table (`lens.rs`), its kind and a kind-specific
//! id. A **cut** node holds its dense cut id. A **record-block** node (a non-cut node of a block
//! with an SPQR tree) holds the position of its header `[block, SPQR location]` in the lift's row
//! table, its stored row cells after it. Any other non-cut node is **plain** (header `[block]`,
//! cells after it) or **simple** (its block inline, no cells). A non-cut node's record also holds
//! the first-hop mask of its closed-form rows, if it has them ([`RowRef`], `rows.rs`).
//!
//! The lift settles the row forms after the blocks are built ([`NodeTopology::rewrite_row_starts`]):
//! until then a record-block node's block and location sit in a build-time table ([`BuildLoc`],
//! read by [`TopologySections::view_build`]) and every other non-cut node is simple.
//!
//! # SPQR locations
//!
//! A query enters a record block at the SPQR nodes containing an endpoint. They form a connected
//! subtree of the block's SPQR tree, so two numbers locate the node ([`SpqrLocation`],
//! `locate.rs`): the subtree's deepest node `x`, and how far it reaches above `x`; an ancestor of
//! `x` contains the node iff it lies within that reach. A non-cut node's location is packed into
//! its header; one that does not fit, and a cut node's (one per record block containing it), go to
//! an out-of-line list ([`SpqrLoc`]). `locator.rs` locates every node while the topology is built.
//!
//! # Blocks, cut nodes and chains
//!
//! The block-cut forest under a super-root (`incidence.rs`, `layout.rs`) is laid out in heavy-path
//! chains ([`crate::chain_nav`]). The engine numbers blocks so that the layout is arithmetic
//! ([`chain_block_numbering`], `numbering.rs`): the component roots are blocks `0..R` and cut `k`'s
//! heavy child is block `R + k`; dense cut ids follow the chain order. The topology writes the
//! chain-layout records (`chain_records.rs`); a block's parent cut, a cut's blocks, a block's cut
//! list and a node's component are read through them ([`ChainNav`], this file).
//!
//! # Sections
//!
//! [`TopologySections::write`] (`build.rs`) projects the decomposition and pushes the sections;
//! `sections.rs` resolves them into the [`NodeTopology`] view.

mod build;
mod chain_records;
mod incidence;
mod layout;
mod lens;
mod locate;
mod locator;
mod numbering;
mod record;
mod rows;
mod sections;
#[cfg(test)]
mod tests;

pub use lens::{NodeLenSecs, NodeLens};
#[cfg(test)]
use locate::FORCE_LOC_LIST;
pub use locate::{BuildLoc, SpqrLoc, SpqrLocation};
pub use numbering::{BlockNumbering, chain_block_numbering};
pub use record::{LEN_BITS, LEN_ESCAPE, NodeRecord};
pub use rows::{NodeRows, ROW_CLOSED, ROW_MASK_BITS, RowRef, header_slots, push_header_word};
pub use sections::{TopologyMeta, TopologySections};

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};
use spqr_tree::decomposition::indices::{BlockIndex, ComponentIndex};

use crate::{
    chain_nav::{ChainNav, ChainRec, CutExtraDir, CutRec, HeadBlockRec, RankBits, SpecialBlockRec},
    image::Sec,
    min_plus::MinPlus,
    path::GfaPathLength,
};
use record::{K_CUT, K_PLAIN, K_RECORD, K_SIMPLE, NO_BLOCK};

/// Offsets into a flat array must fit `u32`; a graph beyond that is out of scope for this layout.
fn offset(len: usize) -> u32 {
    u32::try_from(len).expect("NodeTopology: more than u32::MAX entries in one flat array")
}

/// See the module docs. An image-backed view: [`TopologySections::write`] writes the sections,
/// [`TopologySections::view`] resolves them.
#[derive(Clone, Copy)]
pub struct NodeTopology<'a, IndexType: GraphIndexInteger> {
    /// The sections the records and the meta live in (for the build-time row relayout), and the
    /// chain-layout records the lift fills in place.
    records_sec: Sec<NodeRecord<IndexType>>,
    meta_sec: Sec<TopologyMeta>,
    cut_recs_sec: Sec<CutRec<IndexType>>,
    block_words_sec: Sec<u32>,
    chain_recs_sec: Sec<ChainRec>,
    special_block_recs_sec: Sec<SpecialBlockRec>,
    /// `node_count + 1` records (the last a sentinel) and the long-length table.
    lens: NodeLens<'a, IndexType>,
    records: &'a [NodeRecord<IndexType>],
    /// The record-block nodes' locations while the engine is built (empty afterwards), and the
    /// lift's row table holding the [`K_PLAIN`] / [`K_RECORD`] headers (empty before its layout).
    build_locs: &'a [BuildLoc],
    rows: &'a [MinPlus<IndexType>],
    /// The cut nodes with SPQR locations (their rank indexes `cut_loc_starts`, which point into
    /// `spqr_locs`).
    cut_loc: RankBits<'a>,
    cut_loc_starts: &'a [u32],
    /// The SPQR locations that do not fit inline in an entry (every cut node's).
    spqr_locs: &'a [SpqrLoc<IndexType>],
    component_count: usize,
    row_total: usize,
    /// The chain block numbering's root blocks (see [`chain_block_numbering`]).
    root_blocks: usize,
    /// The chain layout (see [`crate::chain_nav`]): one record per cut (dense id = chain
    /// order), a word per block, a record per non-heavy block and per chain, the light child blocks
    /// of multi-block cuts, and the cut lists of the blocks whose cut count is not 2.
    cut_recs: &'a [CutRec<IndexType>],
    block_words: &'a [u32],
    head_blocks: &'a [HeadBlockRec],
    chain_recs: &'a [ChainRec],
    cut_extra_blocks: &'a [BlockIndex<IndexType>],
    cut_extra_dir: &'a [CutExtraDir],
    cut_extra_rank: RankBits<'a>,
    heavy_special: RankBits<'a>,
    special_block_recs: &'a [SpecialBlockRec],
    special_block_cuts: &'a [u32],
}

impl<'a, IndexType: GraphIndexInteger> NodeTopology<'a, IndexType> {
    /// The chain layout's navigation over this topology's records (structure only: the lift's
    /// exception list is not part of it, so [`ChainNav::irr_above`] is not meaningful here).
    #[inline]
    pub fn chain_nav(&self) -> ChainNav<'a, IndexType> {
        ChainNav {
            nb: self.block_words.len(),
            nc: self.cut_recs.len(),
            roots: self.root_blocks,
            cuts: self.cut_recs,
            block_words: self.block_words,
            heads: self.head_blocks,
            heavy_special: self.heavy_special,
            chains: self.chain_recs,
            extras: self.cut_extra_blocks,
            extra_rank: self.cut_extra_rank,
            extra_dir: self.cut_extra_dir,
            exc: &[],
        }
    }

    /// The cut records (dense cut id = chain order).
    #[cfg(test)]
    #[inline]
    pub fn cut_recs(&self) -> &'a [CutRec<IndexType>] {
        self.cut_recs
    }

    /// The block words ([`crate::chain_nav::block_bits`]).
    #[inline]
    pub fn block_words(&self) -> &'a [u32] {
        self.block_words
    }

    /// The chain block numbering's root blocks: cut `c`'s heavy child is block `root_blocks + c`.
    #[inline]
    pub fn root_blocks(&self) -> usize {
        self.root_blocks
    }

    /// The special-block records, sorted by block.
    #[cfg(test)]
    #[inline]
    pub fn special_block_recs_slice(&self) -> &'a [SpecialBlockRec] {
        self.special_block_recs
    }

    /// The special-block record of `block` (a block whose cut count is not 2, or a root block), if
    /// any, and its cut list (dense cut ids in the decomposition's order).
    #[inline]
    pub fn special_block(&self, block: usize) -> Option<(&'a SpecialBlockRec, &'a [u32])> {
        let i = self
            .special_block_recs
            .binary_search_by_key(&(block as u32), |r| r.block)
            .ok()?;
        let rec = &self.special_block_recs[i];
        Some((
            rec,
            &self.special_block_cuts
                [rec.cut_start as usize..(rec.cut_start + rec.cut_count) as usize],
        ))
    }

    /// The light child blocks of cut `c` (none unless the cut lies in three or more blocks).
    #[cfg(test)]
    #[inline]
    pub fn cut_extra_blocks(
        &self,
        c: usize,
    ) -> impl Iterator<Item = BlockIndex<IndexType>> + use<'a, IndexType> {
        self.chain_nav().cut_child_blocks(c).skip(1)
    }

    /// Build-time only: the sections of the chain-layout records the lift fills in place
    /// (`(cut_recs, block_words, chain_recs, special_block_recs)` byte offsets and lengths).
    pub fn chain_record_sections(
        &self,
    ) -> (
        Sec<CutRec<IndexType>>,
        Sec<u32>,
        Sec<ChainRec>,
        Sec<SpecialBlockRec>,
    ) {
        (
            self.cut_recs_sec,
            self.block_words_sec,
            self.chain_recs_sec,
            self.special_block_recs_sec,
        )
    }

    /// The number of blocks (dense ids `0..block_count`).
    pub fn block_count(&self) -> usize {
        self.block_words.len()
    }

    /// The number of graph nodes.
    pub fn node_count(&self) -> usize {
        self.records.len() - 1
    }

    /// The sequence length of `node`.
    #[inline]
    pub fn node_len(&self, node: NodeIndex<IndexType>) -> GfaPathLength<IndexType> {
        self.lens.len(node.into_usize())
    }

    /// The block of a non-cut node (`None` for a cut node or an isolated segment).
    #[inline]
    pub(super) fn non_cut_block(&self, record: &NodeRecord<IndexType>) -> Option<usize> {
        match record.kind() {
            K_SIMPLE => (record.id != NO_BLOCK).then_some(record.id as usize),
            K_PLAIN | K_RECORD if !self.rows.is_empty() => {
                Some(self.header_word(record.id as usize, 0))
            }
            K_RECORD => Some(self.build_locs[record.id as usize].block as usize),
            _ => None,
        }
    }

    /// The connected component of `node`. A node in no block (an isolated segment) reports the
    /// sentinel `component_count`: it shares a component with nothing a query can route to.
    #[inline]
    pub fn node_component_index(&self, node: NodeIndex<IndexType>) -> ComponentIndex<IndexType> {
        use crate::chain_nav::block_bits;
        let record = &self.records[node.into_usize()];
        let block = match record.kind() {
            // Any of the cut's blocks has its component: its heavy child `roots + c` needs no load.
            K_CUT => Some(self.root_blocks + record.id as usize),
            _ => self.non_cut_block(record),
        };
        match block {
            Some(b) => {
                if (self.component_count as u64) < u64::from(block_bits::COMPONENT_INLINE_LIMIT) {
                    ComponentIndex::from_usize(
                        (self.block_words[b] >> block_bits::COMPONENT_SHIFT) as usize,
                    )
                } else {
                    ComponentIndex::from_usize(
                        self.chain_recs[self.chain_nav().chain_of_block(b)].component as usize,
                    )
                }
            }
            None => ComponentIndex::from_usize(self.component_count),
        }
    }

    /// Whether `node` is a cut node (lies in two or more blocks).
    #[inline]
    pub fn is_cut_node(&self, node: NodeIndex<IndexType>) -> bool {
        self.records[node.into_usize()].kind() == K_CUT
    }

    /// The dense cut id of `node` (its position in [`Self::cut_nodes`]), if it is a cut node.
    #[inline]
    pub fn cut_id(&self, node: NodeIndex<IndexType>) -> Option<usize> {
        let record = &self.records[node.into_usize()];
        (record.kind() == K_CUT).then_some(record.id as usize)
    }

    /// The cut nodes in chain order (index = dense cut id; see [`crate::chain_nav`]).
    pub fn cut_nodes(&self) -> Vec<NodeIndex<IndexType>> {
        self.cut_recs.iter().map(|r| r.node).collect()
    }

    /// The number of cut nodes.
    #[cfg(test)]
    pub fn cut_count(&self) -> usize {
        self.cut_recs.len()
    }

    /// The blocks containing `node`: its one block for a non-cut node; for a cut node its parent
    /// block, its heavy child block and any light child blocks (a set: the order is not the
    /// decomposition's).
    pub fn node_block_indices(
        &self,
        node: NodeIndex<IndexType>,
    ) -> impl Iterator<Item = BlockIndex<IndexType>> + 'a {
        let record = &self.records[node.into_usize()];
        let (first, children) = match self.cut_id(node) {
            Some(c) => {
                let nav = self.chain_nav();
                (
                    Some(BlockIndex::from_usize(nav.parent_block_of_cut(c))),
                    Some(nav.cut_child_blocks(c)),
                )
            }
            None => (self.non_cut_block(record).map(BlockIndex::from_usize), None),
        };
        first.into_iter().chain(children.into_iter().flatten())
    }

    /// The dense cut ids of `block`'s cut nodes, in the decomposition's order (the `M_aug` and
    /// node-row order): from the special table, or `[parent cut, child cut]` as the parent-position
    /// bit says for a plain 2-cut block.
    pub fn block_cut_ids(&self, block: BlockIndex<IndexType>) -> Vec<u32> {
        let b = block.into_usize();
        if let Some((_, ids)) = self.special_block(b) {
            return ids.to_vec();
        }
        let nav = self.chain_nav();
        let parent = nav.parent(b).expect("a plain block has a parent cut") - nav.nb;
        let child = nav.heavy(b).expect("a plain block has a child cut") - nav.nb;
        if self.block_words[b] & crate::chain_nav::block_bits::PARENT_POS == 0 {
            vec![parent as u32, child as u32]
        } else {
            vec![child as u32, parent as u32]
        }
    }

    /// The cut nodes of `block`, in the decomposition's order.
    pub fn block_cut_nodes(&self, block: BlockIndex<IndexType>) -> Vec<NodeIndex<IndexType>> {
        self.block_cut_ids(block)
            .into_iter()
            .map(|id| self.cut_recs[id as usize].node)
            .collect()
    }
}
