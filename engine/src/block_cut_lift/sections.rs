//! The lift's image sections. It fills the topology's chain-layout records in place (cut records,
//! block words, chain records, special-block records) and adds its own: the meta scalars, the
//! prefix lift's per-chain exception lists, the special blocks' flip lists, the `M_aug` tables (the
//! dense matrices and the sorted cut lists of many-cut blocks), the backend's sections and the
//! node-row table.

use std::collections::HashMap;

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};
use spqr_tree::decomposition::indices::BlockIndex;

use super::{
    BlockCutLift,
    aug::BlockAug,
    path_product::{BackendSections, BackendView, OwnedBackend},
    row_plan::RowPlan,
};
use crate::{
    chain_nav::{
        self, ChainNav, ChainRec, CutExtraDir, CutRec, HeadBlockRec, RankBits, RankBlock,
        SpecialBlockRec, block_bits, cut_flags,
    },
    chain_segtree::SegSections,
    image::{Image, Padded, Sec, image_record},
    min_plus::{Mat2, MinPlus},
    path::GfaPathLength,
    prefix_lift, sections,
};

/// `(cut segment, position in the block's cut list)`, sorted by segment, for the `M_aug`
/// row/column lookup of a block with more than [`AUG_SORTED_MIN_CUTS`] cuts (smaller blocks scan
/// their cut list; see [`AugSortedDir`]).
///
/// `pos` is as wide as the index word, so neither word size leaves padding.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AugSorted<IndexType: GraphIndexInteger> {
    pub cut: NodeIndex<IndexType>,
    pub pos: Padded<u32, IndexType>,
}
image_record!(impl<I> AugSorted<I> = [NodeIndex<I>, Padded<u32, I>]);

/// Blocks with at most this many cuts locate a cut's `M_aug` position by scanning their cut list.
pub const AUG_SORTED_MIN_CUTS: usize = 8;

/// Directory entry of a many-cut block's sorted `(cut, pos)` list (`aug_sorted[start..start+count]`);
/// the directory is sorted by block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugSortedDir {
    pub block: u32,
    pub start: u32,
    pub count: u32,
}
crate::image::image_record!(AugSortedDir = [u32, u32, u32]);

/// Directory entry of a plain block that keeps a dense `M_aug` (one not derived from the prefix
/// lift): the start of its row-major `4x4` data; sorted by block. A special block's record holds its
/// own start.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugDenseDir {
    pub block: u32,
    pub start: u32,
}
crate::image::image_record!(AugDenseDir = [u32, u32]);

/// Scalars of the lift (one-element section).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LiftMeta {
    pub number_of_blocks: u64,
    pub number_of_cuts: u64,
    /// [`META_PREFIX_LIFT`]; the other bits are unused (`0`).
    pub flags: u64,
    /// The chain block numbering's root blocks (see `crate::topology::chain_block_numbering`).
    pub root_blocks: u64,
}
crate::image::image_record!(LiftMeta = [u64, u64, u64, u64]);

/// `LiftMeta::flags` bit 0: the path-product backend is the closed-form prefix lift.
const META_PREFIX_LIFT: u64 = 1;

/// Typed handles to the lift's sections. The chain-layout records (`TOPO_CUT_RECS`,
/// `TOPO_BLOCK_WORDS`, `TOPO_CHAIN_RECS`, `TOPO_SPECIAL_BLOCK_*`) are the topology's: the lift fills
/// them in place at build and reads them at query time.
#[derive(Clone, Copy, Debug)]
pub struct LiftSections<IndexType: GraphIndexInteger> {
    meta: Sec<LiftMeta>,
    cut_recs: Sec<CutRec<IndexType>>,
    block_words: Sec<u32>,
    head_blocks: Sec<HeadBlockRec>,
    chain_recs: Sec<ChainRec>,
    cut_extras: Sec<BlockIndex<IndexType>>,
    cut_extra_dir: Sec<CutExtraDir>,
    cut_extra_rank: Sec<RankBlock>,
    heavy_special_rank: Sec<RankBlock>,
    chain_exc: Sec<u32>,
    special_recs: Sec<SpecialBlockRec>,
    special_cuts: Sec<u32>,
    special_flips: Sec<u32>,
    aug_sorted_dir: Sec<AugSortedDir>,
    aug_sorted: Sec<AugSorted<IndexType>>,
    aug_dense_dir: Sec<AugDenseDir>,
    aug_dense_data: Sec<MinPlus<IndexType>>,
    backend: BackendSections<IndexType>,
    node_rows: Sec<MinPlus<IndexType>>,
}

/// Everything the build produces, ready to be written (owned, build-time only).
pub(super) struct LiftOwned<IndexType: GraphIndexInteger> {
    pub(super) number_of_blocks: usize,
    pub(super) cut_segment: Vec<NodeIndex<IndexType>>,
    pub(super) full: Vec<Mat2<IndexType>>,
    /// Per cut: `down[c]` has a finite off-diagonal (the cut can flip through its pendant subtree).
    pub(super) down_flip: Vec<bool>,
    pub(super) block_aug: Vec<BlockAug<IndexType>>,
    pub(super) block_flip_cuts: Vec<Vec<(NodeIndex<IndexType>, Mat2<IndexType>)>>,
    /// Per block: [`block_bits::SIMPLE_ROWS`].
    pub(super) block_simple: Vec<bool>,
    pub(super) cut_id: HashMap<NodeIndex<IndexType>, usize>,
    pub(super) cut_parent: Vec<Option<usize>>,
    /// Per block: its row kind, away orientations and whether its `M_aug` is derived.
    pub(super) rows: RowPlan,
    /// The path-product backend: the prefix lift when chosen, the chain segment trees otherwise.
    pub(super) backend: OwnedBackend<IndexType>,
    /// The topology's chain-layout record sections, filled in place here.
    pub(super) record_secs: (
        Sec<CutRec<IndexType>>,
        Sec<u32>,
        Sec<ChainRec>,
        Sec<SpecialBlockRec>,
    ),
    pub(super) root_blocks: usize,
}

impl<IndexType: GraphIndexInteger> LiftOwned<IndexType> {
    fn meta(&self) -> LiftMeta {
        LiftMeta {
            number_of_blocks: self.number_of_blocks as u64,
            number_of_cuts: self.cut_segment.len() as u64,
            flags: if self.backend.prefix().is_some() {
                META_PREFIX_LIFT
            } else {
                0
            },
            root_blocks: self.root_blocks as u64,
        }
    }
}

/// The topology's chain-layout records, which the lift fills in place.
struct ChainRecords<'i, IndexType: GraphIndexInteger> {
    cuts: &'i mut [CutRec<IndexType>],
    block_words: &'i mut [u32],
    chains: &'i mut [ChainRec],
    special: &'i mut [SpecialBlockRec],
}

impl<'i, IndexType: GraphIndexInteger> ChainRecords<'i, IndexType> {
    fn of(image: &'i Image, owned: &LiftOwned<IndexType>) -> Self {
        let (cut_sec, block_sec, chain_sec, special_sec) = owned.record_secs;
        // SAFETY: the topology pushed these sections into `image` during this build; no view reads
        // them while they are filled (the build re-resolves its views afterwards).
        unsafe {
            Self {
                cuts: image.get_at_mut(cut_sec.off as usize, owned.cut_segment.len()),
                block_words: image.get_at_mut(block_sec.off as usize, owned.number_of_blocks),
                chains: image.get_at_mut(chain_sec.off as usize, chain_sec.len as usize),
                special: image.get_at_mut(special_sec.off as usize, special_sec.len as usize),
            }
        }
    }
}

/// The cut records: the flip closure's off-diagonals, the prefix and its flag bits, and whether
/// the cut can flip through its pendant subtree.
fn fill_cut_records<IndexType: GraphIndexInteger>(
    cut_recs: &mut [CutRec<IndexType>],
    owned: &LiftOwned<IndexType>,
) {
    for (c, rec) in cut_recs.iter_mut().enumerate() {
        let full = &owned.full[c];
        debug_assert_eq!(
            rec.node, owned.cut_segment[c],
            "the lift's cut ids are the topology's"
        );
        debug_assert!(
            full.get(0, 0) == MinPlus::zero() && full.get(1, 1) == MinPlus::zero(),
            "a flip closure has a zero diagonal"
        );
        rec.f01 = full.get(0, 1);
        rec.f10 = full.get(1, 0);
        let mut flags = if owned.down_flip[c] {
            cut_flags::DOWN_FLIP
        } else {
            0
        };
        match owned.backend.prefix() {
            Some(prefix) => {
                rec.p = prefix.p(c);
                flags |= prefix.record_flags(c);
            }
            None => rec.p = GfaPathLength::from_usize(0),
        }
        let cleared = cut_flags::ORIENT | cut_flags::IRREGULAR | cut_flags::BLOCKED;
        rec.flags = Padded::new((rec.flags.get() & !(cleared | cut_flags::DOWN_FLIP)) | flags);
    }
}

/// The prefix lift's per-chain exception lists: each chain's exception cuts in id order (returned,
/// concatenated) and the nearest exception above its first cut.
fn chain_exceptions<IndexType: GraphIndexInteger>(
    chain_recs: &mut [ChainRec],
    prefix: &prefix_lift::PrefixLift<IndexType>,
    cut_parent: &[Option<usize>],
) -> Vec<u32> {
    let mut chain_exc: Vec<u32> = Vec::new();
    for k in chain_recs.iter_mut() {
        let count = if k.cut_off < k.len {
            (k.len - k.cut_off).div_ceil(2) as usize
        } else {
            0
        };
        let first = k.cut_start as usize;
        k.exc_start = chain_exc.len() as u32;
        for c in first..first + count {
            if prefix.is_exception(c) {
                chain_exc.push(c as u32);
            }
        }
        k.exc_count = chain_exc.len() as u32 - k.exc_start;
        k.exc_inherit = match (count > 0).then(|| cut_parent[first]).flatten() {
            Some(parent_cut) => prefix.irr_above(parent_cut),
            None => chain_nav::NONE,
        };
    }
    chain_exc
}

/// The block words (row kind, derived `M_aug`, simple rows; a plain block's flip and away bits) and
/// the special blocks' flip lists and away bits. Returns the concatenated flip lists.
fn fill_block_words<IndexType: GraphIndexInteger>(
    block_words: &mut [u32],
    special_recs: &mut [SpecialBlockRec],
    owned: &LiftOwned<IndexType>,
) -> Vec<u32> {
    let mut special_flips: Vec<u32> = Vec::new();
    let mut special = special_recs.iter_mut();
    for (b, word) in block_words.iter_mut().enumerate() {
        *word = (*word & !block_bits::ROW_KIND_MASK)
            | (owned.rows.kinds[b] as u32 & block_bits::ROW_KIND_MASK);
        if owned.rows.aug_derived[b] {
            *word |= block_bits::DERIVED;
        }
        if owned.block_simple[b] {
            *word |= block_bits::SIMPLE_ROWS;
        }
        let flips: Vec<u32> = owned.block_flip_cuts[b]
            .iter()
            .map(|&(cut, _)| owned.cut_id[&cut] as u32)
            .collect();
        if *word & block_bits::SPECIAL != 0 {
            let rec = special.next().expect("every special block has a record");
            debug_assert_eq!(rec.block as usize, b, "special blocks in block order");
            rec.flip_start = special_flips.len() as u32;
            rec.flip_count = flips.len() as u32;
            special_flips.extend_from_slice(&flips);
            rec.away = owned.rows.away_bits[b] as u32;
        } else {
            *word |= plain_block_bits(*word, &flips, owned, b);
        }
    }
    debug_assert!(special.next().is_none(), "every special block was filled");
    special_flips
}

/// The flip and away bits of plain block `b` (two cuts, its list `[parent, child]` in the order of
/// its parent-position bit in `word`).
fn plain_block_bits<IndexType: GraphIndexInteger>(
    word: u32,
    flips: &[u32],
    owned: &LiftOwned<IndexType>,
    b: usize,
) -> u32 {
    let cuts = &owned.block_aug[b].cuts;
    debug_assert_eq!(cuts.len(), 2, "a plain block has two cuts");
    let parent_pos = usize::from(word & block_bits::PARENT_POS != 0);
    let parent_cut = owned.cut_id[&cuts[parent_pos]] as u32;
    let child_cut = owned.cut_id[&cuts[1 - parent_pos]] as u32;
    let away = owned.rows.away_bits[b];
    let mut bits = 0;
    if flips.contains(&parent_cut) {
        bits |= block_bits::FLIP_PARENT;
    }
    if flips.contains(&child_cut) {
        bits |= block_bits::FLIP_CHILD;
    }
    if away & 1 != 0 {
        bits |= block_bits::AWAY0;
    }
    if away & 2 != 0 {
        bits |= block_bits::AWAY1;
    }
    bits
}

/// The `M_aug` sections: the dense matrix of every block that keeps one (a special block's record
/// names its start, a plain one is in the dense directory) and the sorted `(cut, pos)` lists of
/// the blocks with more than [`AUG_SORTED_MIN_CUTS`] cuts.
struct AugTables<IndexType: GraphIndexInteger> {
    sorted_dir: Vec<AugSortedDir>,
    sorted: Vec<AugSorted<IndexType>>,
    dense_dir: Vec<AugDenseDir>,
    dense_data: Vec<MinPlus<IndexType>>,
}

impl<IndexType: GraphIndexInteger> AugTables<IndexType> {
    /// The tables of every block (setting the special records' `aug_start`).
    fn new(
        owned: &LiftOwned<IndexType>,
        block_words: &[u32],
        special_recs: &mut [SpecialBlockRec],
    ) -> Self {
        let mut tables = Self {
            sorted_dir: Vec::new(),
            sorted: Vec::new(),
            dense_dir: Vec::new(),
            dense_data: Vec::new(),
        };
        for (b, aug) in owned.block_aug.iter().enumerate() {
            if aug.cuts.len() > AUG_SORTED_MIN_CUTS {
                tables.push_sorted(b, &aug.cuts);
            }
            if owned.rows.aug_derived[b] {
                continue;
            }
            let start =
                u32::try_from(tables.dense_data.len()).expect("lift M_aug data exceed u32 entries");
            if block_words[b] & block_bits::SPECIAL != 0 {
                let i = chain_nav::special_index_by_block(special_recs, b);
                special_recs[i].aug_start = start;
            } else {
                tables.dense_dir.push(AugDenseDir {
                    block: b as u32,
                    start,
                });
            }
            tables.dense_data.extend_from_slice(&aug.aug);
        }
        tables
    }

    /// Block `b`'s sorted `(cut, pos)` list.
    fn push_sorted(&mut self, b: usize, cuts: &[NodeIndex<IndexType>]) {
        let mut sorted: Vec<AugSorted<IndexType>> = cuts
            .iter()
            .enumerate()
            .map(|(pos, &cut)| AugSorted {
                cut,
                pos: Padded::new(pos as u32),
            })
            .collect();
        sorted.sort_unstable_by_key(|e| e.cut);
        self.sorted_dir.push(AugSortedDir {
            block: b as u32,
            start: self.sorted.len() as u32,
            count: sorted.len() as u32,
        });
        self.sorted.extend_from_slice(&sorted);
    }

    fn push(&self, image: &Image) {
        image.push_section(sections::LIFT_AUG_SORTED_DIR, &self.sorted_dir);
        image.push_section(sections::LIFT_AUG_SORTED_SPARSE, &self.sorted);
        image.push_section(sections::LIFT_AUG_DENSE_DIR, &self.dense_dir);
        image.push_section(sections::LIFT_AUG_DENSE_DATA, &self.dense_data);
    }
}

impl<IndexType: GraphIndexInteger> LiftSections<IndexType> {
    /// Fill the topology's chain-layout records in place and push the lift's sections (their
    /// order is part of the format), then look the handles up as a loaded image does.
    pub(super) fn write(image: &Image, owned: LiftOwned<IndexType>) -> Self {
        let records = ChainRecords::of(image, &owned);
        fill_cut_records(records.cuts, &owned);
        let chain_exc = match owned.backend.prefix() {
            Some(prefix) => chain_exceptions(records.chains, prefix, &owned.cut_parent),
            None => Vec::new(),
        };
        let special_flips = fill_block_words(records.block_words, records.special, &owned);
        let aug = AugTables::new(&owned, records.block_words, records.special);
        owned.backend.write(image);
        image.push_section(sections::LIFT_META, &[owned.meta()]);
        image.push_section(sections::LIFT_CHAIN_EXC, &chain_exc);
        image.push_section(sections::LIFT_SPECIAL_FLIPS, &special_flips);
        aug.push(image);
        Self::from_image(image).expect("the topology and the lift wrote their sections")
    }

    /// Look the sections up in a loaded image.
    pub fn from_image(image: &Image) -> std::io::Result<Self> {
        let meta_sec: Sec<LiftMeta> = image.require(sections::LIFT_META, "lift.meta")?;
        let meta = image.get(meta_sec)[0];
        let backend = if meta.flags & META_PREFIX_LIFT != 0 {
            BackendSections::Prefix(prefix_lift::PrefixSections::from_image(image)?)
        } else {
            BackendSections::Segtree(SegSections::from_image(image)?)
        };
        Ok(Self {
            meta: meta_sec,
            cut_recs: image.require(sections::TOPO_CUT_RECS, "topology.cut_recs")?,
            block_words: image.require(sections::TOPO_BLOCK_WORDS, "topology.block_words")?,
            head_blocks: image.require(sections::TOPO_HEAD_BLOCKS, "topology.head_blocks")?,
            chain_recs: image.require(sections::TOPO_CHAIN_RECS, "topology.chain_recs")?,
            cut_extras: image
                .require(sections::TOPO_CUT_EXTRA_BLOCKS, "topology.cut_extra_blocks")?,
            cut_extra_dir: image.require(sections::TOPO_CUT_EXTRA_DIR, "topology.cut_extra_dir")?,
            cut_extra_rank: image
                .require(sections::TOPO_CUT_EXTRA_WORDS, "topology.cut_extra_rank")?,
            heavy_special_rank: image.require(
                sections::TOPO_HEAVY_SPECIAL_WORDS,
                "topology.heavy_special_rank",
            )?,
            chain_exc: image.require(sections::LIFT_CHAIN_EXC, "lift.chain_exc")?,
            special_recs: image.require(
                sections::TOPO_SPECIAL_BLOCK_RECS,
                "topology.special_block_recs",
            )?,
            special_cuts: image.require(
                sections::TOPO_SPECIAL_BLOCK_CUTS,
                "topology.special_block_cuts",
            )?,
            special_flips: image.require(sections::LIFT_SPECIAL_FLIPS, "lift.special_flips")?,
            aug_sorted_dir: image.require(sections::LIFT_AUG_SORTED_DIR, "lift.aug_sorted_dir")?,
            aug_sorted: image.require(sections::LIFT_AUG_SORTED_SPARSE, "lift.aug_sorted")?,
            aug_dense_dir: image.require(sections::LIFT_AUG_DENSE_DIR, "lift.aug_dense_dir")?,
            aug_dense_data: image.require(sections::LIFT_AUG_DENSE_DATA, "lift.aug_dense_data")?,
            backend,
            node_rows: image.require(sections::LIFT_NODE_ROWS, "lift.node_rows")?,
        })
    }

    /// Resolve against the image.
    pub fn view<'a>(&self, image: &'a Image) -> BlockCutLift<'a, IndexType> {
        let meta = image.get(self.meta)[0];
        let cuts = image.get(self.cut_recs);
        let block_words = image.get(self.block_words);
        let special_recs = image.get(self.special_recs);
        let nav = ChainNav {
            nb: meta.number_of_blocks as usize,
            nc: meta.number_of_cuts as usize,
            roots: meta.root_blocks as usize,
            cuts,
            block_words,
            heads: image.get(self.head_blocks),
            heavy_special: RankBits {
                blocks: image.get(self.heavy_special_rank),
            },
            chains: image.get(self.chain_recs),
            extras: image.get(self.cut_extras),
            extra_rank: RankBits {
                blocks: image.get(self.cut_extra_rank),
            },
            extra_dir: image.get(self.cut_extra_dir),
            exc: image.get(self.chain_exc),
        };
        let backend = match self.backend {
            BackendSections::Prefix(s) => BackendView::Prefix(s.view(image, cuts, nav)),
            BackendSections::Segtree(s) => BackendView::Segtree(s.view(image)),
        };
        BlockCutLift {
            number_of_blocks: meta.number_of_blocks as usize,
            nav,
            cuts,
            block_words,
            special_recs,
            special_cuts: image.get(self.special_cuts),
            special_flips: image.get(self.special_flips),
            aug_sorted_dir: image.get(self.aug_sorted_dir),
            aug_sorted: image.get(self.aug_sorted),
            aug_dense_dir: image.get(self.aug_dense_dir),
            aug_dense_data: image.get(self.aug_dense_data),
            backend,
            node_rows: image.get(self.node_rows),
        }
    }
}
