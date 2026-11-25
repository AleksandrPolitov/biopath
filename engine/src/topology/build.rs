//! Projecting the decomposition onto the topology sections.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::decomposition::SPQRDecomposition;
#[cfg(test)]
use spqr_tree::decomposition::indices::BlockIndex;

#[cfg(test)]
use super::chain_block_numbering;
use super::{
    BlockNumbering, BuildLoc, NodeLenSecs, SpqrLoc, TopologyMeta, TopologySections,
    chain_records::ChainRecords,
    incidence::BlockCutIncidence,
    layout::block_cut_layout,
    locator::Locator,
    numbering::chain_numbering_roots,
    offset,
    record::{K_CUT, K_RECORD, K_SIMPLE, NO_BLOCK, NodeRecord},
};
#[cfg(test)]
use crate::block_spqr_tree::BlockSpqrTree;
use crate::{
    chain_nav::{self, ChainLayout, RankBlock},
    gfa_graph_extensions::GfaNodeDataExt,
    image::Image,
    path::GfaPathLength,
    rooted_tree::TreeAccess,
    sections,
};

type Graph<IndexType, NodeData, EdgeData> = BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>;

impl<IndexType: GraphIndexInteger> TopologySections<IndexType> {
    /// Project `spqr` onto the topology and write its sections into `image`; returns the handles
    /// and the record-block nodes' build-time locations (see [`Self::view_build`]). Blocks are in
    /// the engine's `numbering` ([`super::chain_block_numbering`]). Nodes are located only in the
    /// blocks for which `tree_of(block)` returns the block's rooted SPQR tree (the tree its record
    /// is built from): a block without a record is never entered through its SPQR nodes. Cut rows
    /// are not laid out here: the lift settles every node's row form
    /// ([`super::NodeTopology::rewrite_row_starts`]).
    pub fn write<'t, NodeData, EdgeData, T>(
        image: &Image,
        spqr: &SPQRDecomposition<'_, Graph<IndexType, NodeData, EdgeData>>,
        numbering: &BlockNumbering,
        tree_of: impl Fn(usize) -> Option<&'t T>,
    ) -> (Self, Vec<BuildLoc>)
    where
        NodeData: GfaNodeData,
        EdgeData: GfaEdgeData,
        T: TreeAccess + 't,
    {
        assert_eq!(
            numbering.len(),
            spqr.block_count(),
            "one engine id per block"
        );
        let incidence = BlockCutIncidence::read(spqr, |b| numbering.old_id(b));
        let mut locator = Locator::new(spqr, numbering, tree_of);
        let mut nodes = NodeRecords::build(spqr, numbering, &incidence, &mut locator);
        let layout = block_cut_layout(&incidence);
        let roots = chain_numbering_roots(&layout, incidence.block_count()).expect(
            "NodeTopology: the blocks are not in chain numbering (see `chain_block_numbering`)",
        );
        let cut_locs = nodes.renumber_cuts(&layout);
        let chain = ChainRecords::build(layout, &incidence, roots, spqr.graph());
        let meta = TopologyMeta {
            component_count: incidence.component_count as u64,
            row_total: 0,
            root_blocks: roots as u64,
            rows_off: 0,
            rows_len: 0,
        };
        drop(incidence);
        let secs = push_sections(
            image,
            spqr.graph(),
            &nodes,
            &cut_locs,
            &locator.list,
            meta,
            &chain,
        );
        (secs, nodes.build_locs)
    }

    /// [`Self::write`] with a tree for every block that has SPQR nodes (built here); also returns
    /// the numbering.
    #[cfg(test)]
    pub(crate) fn write_with_all_trees<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        image: &Image,
        spqr: &SPQRDecomposition<'_, Graph<IndexType, NodeData, EdgeData>>,
    ) -> (Self, Vec<BuildLoc>, BlockNumbering) {
        let numbering = chain_block_numbering(spqr);
        let blocks: Vec<_> = spqr.iter_blocks().map(|(_, block)| block).collect();
        let trees: Vec<Option<BlockSpqrTree<IndexType>>> = (0..numbering.len())
            .map(|b| {
                let old: BlockIndex<IndexType> = numbering.old_id(b);
                (blocks[old.into_usize()].spqr_node_count() > 0)
                    .then(|| BlockSpqrTree::new_without_lookup(spqr, old))
            })
            .collect();
        let (secs, build_locs) = Self::write(image, spqr, &numbering, |b| {
            trees[b].as_ref().map(BlockSpqrTree::tree)
        });
        (secs, build_locs, numbering)
    }
}

/// The per-node part of the topology: every node's record (plus a sentinel), the record-block
/// nodes' build-time locations and the cut nodes' first location entries.
struct NodeRecords<IndexType: GraphIndexInteger> {
    records: Vec<NodeRecord<IndexType>>,
    build_locs: Vec<BuildLoc>,
    /// Per provisional cut id: its first entry in the location list ([`chain_nav::NONE`] when it
    /// lies in no record block).
    cut_loc_tmp: Vec<u32>,
}

/// The cut nodes' locations by dense cut id: the cuts with any (an O(1)-rank set) and where each
/// one's entries start in the location list.
struct CutLocations {
    rank: Vec<RankBlock>,
    starts: Vec<u32>,
}

impl<IndexType: GraphIndexInteger> NodeRecords<IndexType> {
    /// Every node's record, cut nodes under their provisional ids. A non-cut node is a record-block
    /// node when `locator` locates it, else simple until the lift's row layout settles it.
    fn build<'t, NodeData: GfaNodeData, EdgeData: GfaEdgeData, T: TreeAccess>(
        spqr: &SPQRDecomposition<'_, Graph<IndexType, NodeData, EdgeData>>,
        numbering: &BlockNumbering,
        incidence: &BlockCutIncidence<IndexType>,
        locator: &mut Locator<'t, IndexType, T>,
    ) -> Self {
        let graph = spqr.graph();
        let node_count = graph.node_count();
        let mut nodes = Self {
            records: Vec::with_capacity(node_count + 1),
            build_locs: Vec::new(),
            cut_loc_tmp: Vec::new(),
        };
        for v in 0..node_count {
            let node = NodeIndex::from_usize(v);
            let len: GfaPathLength<IndexType> = graph.node_data(node).len();
            let len = len.into_usize();
            let cut_tmp = incidence.cut_tmp_of_node[v];
            let record = if cut_tmp != chain_nav::NONE {
                nodes.cut_loc_tmp.push(locator.locate_cut(spqr, node));
                NodeRecord::new(len, K_CUT, 0, cut_tmp)
            } else {
                let block = spqr
                    .node_block_indices(node)
                    .next()
                    .map(|b| numbering.new_id(b));
                match locator.locate_non_cut(spqr, node, block) {
                    Some(loc) => {
                        let block = block.expect("a located node lies in a block");
                        nodes.build_locs.push(BuildLoc {
                            block: offset(block.into_usize()),
                            loc,
                        });
                        NodeRecord::new(len, K_RECORD, 0, offset(nodes.build_locs.len() - 1))
                    }
                    // Provisional: the lift's row layout settles closed-form / stored / no rows.
                    None => {
                        let block = block.map_or(NO_BLOCK, |b| offset(b.into_usize()));
                        NodeRecord::new(len, K_SIMPLE, 0, block)
                    }
                }
            };
            nodes.records.push(record);
        }
        // Sentinel (its fields are never read as a node's).
        nodes
            .records
            .push(NodeRecord::new(0, K_SIMPLE, 0, NO_BLOCK));
        nodes
    }

    /// Re-key the cut records by dense (chain-order) cut id and return the cut nodes' locations
    /// under those ids.
    fn renumber_cuts(&mut self, layout: &ChainLayout) -> CutLocations {
        let node_count = self.records.len() - 1;
        for rec in self.records[..node_count].iter_mut() {
            if rec.kind() == K_CUT {
                rec.id = layout.cut_new_id[rec.id as usize];
            }
        }
        let start_of = |c: usize| self.cut_loc_tmp[layout.cut_old_of_new[c] as usize];
        let members: Vec<usize> = (0..layout.cut_count())
            .filter(|&c| start_of(c) != chain_nav::NONE)
            .collect();
        CutLocations {
            starts: members.iter().map(|&c| start_of(c)).collect(),
            rank: chain_nav::rank_bits(layout.cut_count(), members, 0),
        }
    }
}

/// Push the topology's sections (this order is part of the image format).
fn push_sections<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    image: &Image,
    graph: &Graph<IndexType, NodeData, EdgeData>,
    nodes: &NodeRecords<IndexType>,
    cut_locs: &CutLocations,
    spqr_locs: &[SpqrLoc<IndexType>],
    meta: TopologyMeta,
    chain: &ChainRecords<IndexType>,
) -> TopologySections<IndexType> {
    let lens = (0..graph.node_count()).map(|v| graph.node_data(NodeIndex::from_usize(v)).len());
    TopologySections {
        lens: NodeLenSecs::write(image, lens, &nodes.records),
        cut_loc_rank: image.push_section(sections::TOPO_CUT_LOC_WORDS, &cut_locs.rank),
        cut_loc_starts: image.push_section(sections::TOPO_CUT_LOC_STARTS, &cut_locs.starts),
        spqr_locs: image.push_section(sections::TOPO_SPQR_LOCS, spqr_locs),
        meta: image.push_section(sections::TOPO_META, &[meta]),
        cut_recs: image.push_section(sections::TOPO_CUT_RECS, &chain.cut_recs),
        block_words: image.push_section(sections::TOPO_BLOCK_WORDS, &chain.block_words),
        head_blocks: image.push_section(sections::TOPO_HEAD_BLOCKS, &chain.head_blocks),
        chain_recs: image.push_section(sections::TOPO_CHAIN_RECS, &chain.chain_recs),
        cut_extra_blocks: image
            .push_section(sections::TOPO_CUT_EXTRA_BLOCKS, &chain.cut_extra_blocks),
        cut_extra_dir: image.push_section(sections::TOPO_CUT_EXTRA_DIR, &chain.cut_extra_dir),
        cut_extra_rank: image.push_section(sections::TOPO_CUT_EXTRA_WORDS, &chain.cut_extra_rank),
        heavy_special_rank: image.push_section(
            sections::TOPO_HEAVY_SPECIAL_WORDS,
            &chain.heavy_special_rank,
        ),
        special_block_recs: image
            .push_section(sections::TOPO_SPECIAL_BLOCK_RECS, &chain.special_block_recs),
        special_block_cuts: image
            .push_section(sections::TOPO_SPECIAL_BLOCK_CUTS, &chain.special_block_cuts),
    }
}
