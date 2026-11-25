//! The chain-layout records the topology writes (see [`crate::chain_nav`] for how the query
//! navigates them): the structural fields only; the lift fills the rest in place.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::GfaNodeData,
};
use spqr_tree::decomposition::indices::BlockIndex;

use super::{incidence::BlockCutIncidence, offset};
use crate::{
    chain_nav::{
        self, ChainLayout, ChainNav, ChainRec, CutExtraDir, CutRec, HeadBlockRec, RankBlock,
        SpecialBlockRec, block_bits, cut_flags,
    },
    gfa_graph_extensions::GfaNodeDataExt,
    image::Padded,
    min_plus::MinPlus,
    path::GfaPathLength,
};

/// The chain layout's records, blocks in chain numbering with `roots` root blocks.
pub(super) struct ChainRecords<IndexType: GraphIndexInteger> {
    /// One per cut (dense id = chain order).
    pub(super) cut_recs: Vec<CutRec<IndexType>>,
    /// One per block ([`block_bits`]).
    pub(super) block_words: Vec<u32>,
    /// One per block that is no cut's heavy child (see [`ChainNav::head_index`]).
    pub(super) head_blocks: Vec<HeadBlockRec>,
    pub(super) chain_recs: Vec<ChainRec>,
    /// The child lists of the cuts in three or more blocks and their directory.
    pub(super) cut_extra_blocks: Vec<BlockIndex<IndexType>>,
    pub(super) cut_extra_dir: Vec<CutExtraDir>,
    /// Over cut ids: the cuts with a child list, and the cuts whose heavy child is special.
    pub(super) cut_extra_rank: Vec<RankBlock>,
    pub(super) heavy_special_rank: Vec<RankBlock>,
    /// The blocks whose cut count is not 2 (or root blocks), sorted by block, and their cut lists.
    pub(super) special_block_recs: Vec<SpecialBlockRec>,
    pub(super) special_block_cuts: Vec<u32>,
}

impl<IndexType: GraphIndexInteger> ChainRecords<IndexType> {
    /// The records of `layout` (built over `incidence`, whose blocks are in chain numbering with
    /// `roots` root blocks).
    pub(super) fn build<NodeData: GfaNodeData, EdgeData>(
        layout: ChainLayout,
        incidence: &BlockCutIncidence<IndexType>,
        roots: usize,
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    ) -> Self {
        let cut_nodes: Vec<NodeIndex<IndexType>> = layout
            .cut_old_of_new
            .iter()
            .map(|&old| incidence.cut_nodes[old as usize])
            .collect();
        let mut cut_recs = cut_records(&layout, &cut_nodes, graph);
        let block_words = block_words(&layout, incidence, &cut_nodes);
        let mut head_blocks = head_blocks(&layout, roots);
        let (cut_extra_blocks, cut_extra_dir) = extra_blocks(&layout, roots, &mut cut_recs);
        let (special_block_recs, special_block_cuts) =
            special_blocks(&layout, incidence, &block_words, roots, &mut head_blocks);
        let nc = cut_recs.len();
        // The cuts whose heavy child is special: their records are the run of heavy children in
        // the block-sorted special records, after the root blocks'.
        let first_heavy_special =
            special_block_recs.partition_point(|r| (r.block as usize) < roots);
        let heavy_special_rank = chain_nav::rank_bits(
            nc,
            (0..nc).filter(|&c| block_words[roots + c] & block_bits::SPECIAL != 0),
            offset(first_heavy_special),
        );
        let cut_extra_rank =
            chain_nav::rank_bits(nc, cut_extra_dir.iter().map(|d| d.cut as usize), 0);
        Self {
            cut_recs,
            block_words,
            head_blocks,
            chain_recs: layout.chains,
            cut_extra_blocks,
            cut_extra_dir,
            cut_extra_rank,
            heavy_special_rank,
            special_block_recs,
            special_block_cuts,
        }
    }
}

/// The cut records (`cut_nodes` in chain order): node, length, chain and first-in-chain flag.
fn cut_records<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    layout: &ChainLayout,
    cut_nodes: &[NodeIndex<IndexType>],
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
) -> Vec<CutRec<IndexType>> {
    let nb = layout.block_count();
    let cut_recs = cut_nodes
        .iter()
        .enumerate()
        .map(|(new, &node)| {
            let v = nb + layout.cut_old_of_new[new] as usize;
            debug_assert!(
                (layout.parent[v] as usize) < nb,
                "a cut node's parent is a block"
            );
            let chain = layout.chain_of[v];
            let first = new as u32 == layout.chains[chain as usize].cut_start;
            CutRec {
                node,
                p: GfaPathLength::from_usize(0),
                len: graph.node_data(node).len(),
                f01: MinPlus::infinity(),
                f10: MinPlus::infinity(),
                flags: Padded::new(
                    (chain << cut_flags::CHAIN_SHIFT)
                        | if first { cut_flags::FIRST_IN_CHAIN } else { 0 },
                ),
            }
        })
        .collect();
    assert!(
        (layout.chains.len() as u64) < (1u64 << (32 - cut_flags::CHAIN_SHIFT)),
        "NodeTopology: more chains than the cut record's chain field holds"
    );
    cut_recs
}

/// The structural bits of every block word: its component (when inlined), special (not a plain
/// 2-cut non-root block) and, for a plain block, the position of its parent cut in its cut list
/// (the `M_aug` / node-row order).
fn block_words<IndexType: GraphIndexInteger>(
    layout: &ChainLayout,
    incidence: &BlockCutIncidence<IndexType>,
    cut_nodes: &[NodeIndex<IndexType>],
) -> Vec<u32> {
    let nb = layout.block_count();
    let super_root = nb + layout.cut_count();
    let inline_components =
        (incidence.component_count as u64) < u64::from(block_bits::COMPONENT_INLINE_LIMIT);
    (0..nb)
        .map(|b| {
            let cuts = incidence.block_cuts(b);
            let mut bits = if inline_components {
                (incidence.block_component[b].into_usize() as u32) << block_bits::COMPONENT_SHIFT
            } else {
                0
            };
            if cuts.len() != 2 || layout.parent[b] as usize == super_root {
                return bits | block_bits::SPECIAL;
            }
            let parent_cut = layout.parent[b] as usize - nb;
            let parent_node = cut_nodes[layout.cut_new_id[parent_cut] as usize];
            let pos = cuts
                .iter()
                .position(|&c| c == parent_node)
                .expect("a block's parent cut is in its list");
            if pos == 1 {
                bits |= block_bits::PARENT_POS;
            }
            bits
        })
        .collect()
}

/// Position and chain of the blocks that are no cut's heavy child (root blocks and light
/// children); their special index is filled by [`special_blocks`].
fn head_blocks(layout: &ChainLayout, roots: usize) -> Vec<HeadBlockRec> {
    let (nb, nc) = (layout.block_count(), layout.cut_count());
    (0..nb)
        .filter(|&b| b < roots || b >= roots + nc)
        .map(|b| HeadBlockRec {
            tpos: layout.tpos[b],
            chain: layout.chain_of[b],
            special: chain_nav::NONE,
        })
        .collect()
}

/// The child lists of the cut nodes in three or more blocks (count, heavy child, the light
/// children) and their directory, flagging those cuts [`cut_flags::HAS_EXTRA`].
fn extra_blocks<IndexType: GraphIndexInteger>(
    layout: &ChainLayout,
    roots: usize,
    cut_recs: &mut [CutRec<IndexType>],
) -> (Vec<BlockIndex<IndexType>>, Vec<CutExtraDir>) {
    let (nb, nc) = (layout.block_count(), layout.cut_count());
    let mut light: Vec<(u32, BlockIndex<IndexType>)> = Vec::new();
    for b in 0..nb {
        let p = layout.parent[b] as usize;
        if p >= nb && p < nb + nc && layout.heavy[p] as usize != b {
            light.push((layout.cut_new_id[p - nb], BlockIndex::from_usize(b)));
        }
    }
    light.sort_unstable_by_key(|&(c, b)| (c, b.into_usize()));
    let mut blocks: Vec<BlockIndex<IndexType>> = Vec::new();
    let mut dir: Vec<CutExtraDir> = Vec::new();
    for children in light.chunk_by(|x, y| x.0 == y.0) {
        let c = children[0].0 as usize;
        cut_recs[c].flags = Padded::new(cut_recs[c].flags.get() | cut_flags::HAS_EXTRA);
        dir.push(CutExtraDir {
            cut: offset(c),
            start: offset(blocks.len()),
        });
        blocks.push(BlockIndex::from_usize(1 + children.len()));
        blocks.push(BlockIndex::from_usize(roots + c));
        blocks.extend(children.iter().map(|&(_, b)| b));
    }
    (blocks, dir)
}

/// The special blocks (cut count not 2, or a root block, as `block_words` marks them) with their
/// cut lists in the decomposition's order (dense cut ids). A root or chain-head block keeps its
/// special index in its head record; a heavy child (rare) is found by block.
fn special_blocks<IndexType: GraphIndexInteger>(
    layout: &ChainLayout,
    incidence: &BlockCutIncidence<IndexType>,
    block_words: &[u32],
    roots: usize,
    head_blocks: &mut [HeadBlockRec],
) -> (Vec<SpecialBlockRec>, Vec<u32>) {
    let nc = layout.cut_count();
    let mut recs: Vec<SpecialBlockRec> = Vec::new();
    let mut cut_lists: Vec<u32> = Vec::new();
    for (b, &word) in block_words.iter().enumerate() {
        if word & block_bits::SPECIAL == 0 {
            continue;
        }
        let cuts = incidence.block_cuts(b);
        if b < roots || b >= roots + nc {
            head_blocks[ChainNav::<IndexType>::head_index(roots, nc, b)].special =
                offset(recs.len());
        }
        recs.push(SpecialBlockRec {
            block: b as u32,
            cut_start: cut_lists.len() as u32,
            cut_count: cuts.len() as u32,
            flip_start: 0,
            flip_count: 0,
            away: 0,
            aug_start: chain_nav::NONE,
        });
        cut_lists.extend(
            cuts.iter()
                .map(|&cut| layout.cut_new_id[incidence.cut_tmp(cut)]),
        );
    }
    (recs, cut_lists)
}
