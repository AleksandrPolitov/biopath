//! The topology's image sections and their resolution into a [`NodeTopology`] view.

use bidirected_adjacency_array::index::GraphIndexInteger;
use spqr_tree::decomposition::indices::BlockIndex;

use super::{BuildLoc, NodeLenSecs, NodeTopology, SpqrLoc};
use crate::{
    chain_nav::{
        ChainRec, CutExtraDir, CutRec, HeadBlockRec, RankBits, RankBlock, SpecialBlockRec,
    },
    image::{Image, Sec},
    sections,
};

/// Scalars of the topology (one-element section).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TopologyMeta {
    /// Components of the decomposition (isolated segments included); also the sentinel component id
    /// a node in no block reports.
    pub component_count: u64,
    /// Total cut-row entries laid out over the nodes (the lift fills exactly this many).
    pub row_total: u64,
    /// The chain block numbering's root blocks `R` (see [`super::chain_block_numbering`]): the
    /// component roots are blocks `0..R` and cut `k`'s heavy child is block `R + k`.
    pub root_blocks: u64,
    /// The lift's row table (it holds the plain and record-block nodes' headers): byte offset and
    /// length, set by the lift's row layout.
    pub rows_off: u64,
    pub rows_len: u64,
}
crate::image::image_record!(TopologyMeta = [u64, u64, u64, u64, u64]);

/// Typed handles to the topology's sections (plain data; resolve with [`Self::view`]).
#[derive(Clone, Copy, Debug)]
pub struct TopologySections<IndexType: GraphIndexInteger> {
    pub(super) lens: NodeLenSecs<IndexType>,
    pub(super) cut_loc_rank: Sec<RankBlock>,
    pub(super) cut_loc_starts: Sec<u32>,
    pub(super) spqr_locs: Sec<SpqrLoc<IndexType>>,
    pub(super) meta: Sec<TopologyMeta>,
    /// The chain layout (see [`crate::chain_nav`]).
    pub(super) cut_recs: Sec<CutRec<IndexType>>,
    pub(super) block_words: Sec<u32>,
    pub(super) head_blocks: Sec<HeadBlockRec>,
    pub(super) chain_recs: Sec<ChainRec>,
    pub(super) cut_extra_blocks: Sec<BlockIndex<IndexType>>,
    pub(super) cut_extra_dir: Sec<CutExtraDir>,
    pub(super) cut_extra_rank: Sec<RankBlock>,
    pub(super) heavy_special_rank: Sec<RankBlock>,
    pub(super) special_block_recs: Sec<SpecialBlockRec>,
    pub(super) special_block_cuts: Sec<u32>,
}

impl<IndexType: GraphIndexInteger> TopologySections<IndexType> {
    /// Resolve against the image the sections live in (after the lift's row layout, or in an image
    /// read back).
    pub fn view<'a>(&self, image: &'a Image) -> NodeTopology<'a, IndexType> {
        self.view_build(image, &[])
    }

    /// Resolve during the build, before the lift's row layout: `build_locs` (from [`Self::write`])
    /// locates the record-block nodes until their headers exist.
    pub fn view_build<'a>(
        &self,
        image: &'a Image,
        build_locs: &'a [BuildLoc],
    ) -> NodeTopology<'a, IndexType> {
        let meta = image.get(self.meta)[0];
        NodeTopology {
            records_sec: self.lens.records,
            meta_sec: self.meta,
            cut_recs_sec: self.cut_recs,
            block_words_sec: self.block_words,
            chain_recs_sec: self.chain_recs,
            special_block_recs_sec: self.special_block_recs,
            lens: self.lens.view(image),
            records: image.get(self.lens.records),
            build_locs,
            rows: if meta.rows_len == 0 {
                &[]
            } else {
                image.get_at(meta.rows_off as usize, meta.rows_len as usize)
            },
            cut_loc: RankBits {
                blocks: image.get(self.cut_loc_rank),
            },
            cut_loc_starts: image.get(self.cut_loc_starts),
            spqr_locs: image.get(self.spqr_locs),
            component_count: meta.component_count as usize,
            row_total: meta.row_total as usize,
            root_blocks: meta.root_blocks as usize,
            cut_recs: image.get(self.cut_recs),
            block_words: image.get(self.block_words),
            head_blocks: image.get(self.head_blocks),
            chain_recs: image.get(self.chain_recs),
            cut_extra_blocks: image.get(self.cut_extra_blocks),
            cut_extra_dir: image.get(self.cut_extra_dir),
            cut_extra_rank: RankBits {
                blocks: image.get(self.cut_extra_rank),
            },
            heavy_special: RankBits {
                blocks: image.get(self.heavy_special_rank),
            },
            special_block_recs: image.get(self.special_block_recs),
            special_block_cuts: image.get(self.special_block_cuts),
        }
    }

    /// Look the sections up in a loaded image.
    pub fn from_image(image: &Image) -> std::io::Result<Self> {
        Ok(Self {
            lens: NodeLenSecs::from_image(image)?,
            cut_loc_rank: image.require(sections::TOPO_CUT_LOC_WORDS, "topology.cut_loc_rank")?,
            cut_loc_starts: image
                .require(sections::TOPO_CUT_LOC_STARTS, "topology.cut_loc_starts")?,
            spqr_locs: image.require(sections::TOPO_SPQR_LOCS, "topology.spqr_locs")?,
            meta: image.require(sections::TOPO_META, "topology.meta")?,
            cut_recs: image.require(sections::TOPO_CUT_RECS, "topology.cut_recs")?,
            block_words: image.require(sections::TOPO_BLOCK_WORDS, "topology.block_words")?,
            head_blocks: image.require(sections::TOPO_HEAD_BLOCKS, "topology.head_blocks")?,
            chain_recs: image.require(sections::TOPO_CHAIN_RECS, "topology.chain_recs")?,
            cut_extra_blocks: image
                .require(sections::TOPO_CUT_EXTRA_BLOCKS, "topology.cut_extra_blocks")?,
            cut_extra_dir: image.require(sections::TOPO_CUT_EXTRA_DIR, "topology.cut_extra_dir")?,
            cut_extra_rank: image
                .require(sections::TOPO_CUT_EXTRA_WORDS, "topology.cut_extra_rank")?,
            heavy_special_rank: image.require(
                sections::TOPO_HEAVY_SPECIAL_WORDS,
                "topology.heavy_special_rank",
            )?,
            special_block_recs: image.require(
                sections::TOPO_SPECIAL_BLOCK_RECS,
                "topology.special_block_recs",
            )?,
            special_block_cuts: image.require(
                sections::TOPO_SPECIAL_BLOCK_CUTS,
                "topology.special_block_cuts",
            )?,
        })
    }

    /// The node-length sections (the graph view reads node lengths from them).
    pub fn lens(&self) -> NodeLenSecs<IndexType> {
        self.lens
    }
}
