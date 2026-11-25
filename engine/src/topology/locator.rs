//! Locating every node in the SPQR trees of the record blocks containing it, while the topology is
//! built (see [`super::SpqrLocation`] for what a location holds).

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::decomposition::{SPQRDecomposition, indices::BlockIndex};

use super::{
    BlockNumbering, SpqrLoc, SpqrLocation,
    locate::{LOC_LAST, list_index, location_of, pack_list, pack_loc},
    offset,
};
use crate::{chain_nav, rooted_tree::TreeAccess};

/// The locator over the record blocks' trees: it collects the out-of-line list and counts the
/// inline and listed nodes for the build log.
pub(super) struct Locator<'t, IndexType: GraphIndexInteger, T> {
    /// `(block, local id)` of every SPQR node, in engine block ids; `None` for an SPQR node listed
    /// in no block.
    home: Vec<Option<(BlockIndex<IndexType>, u32)>>,
    /// Per engine block: its SPQR tree, for a record block.
    trees: Vec<Option<&'t T>>,
    /// The out-of-line locations, a node's entries contiguous.
    pub(super) list: Vec<SpqrLoc<IndexType>>,
    /// Per node (reused): its `(block, local)` SPQR nodes in record blocks, the distinct blocks
    /// (first-appearance order), one block's locals and their sorted copy.
    pairs: Vec<(BlockIndex<IndexType>, u32)>,
    blocks: Vec<BlockIndex<IndexType>>,
    locals: Vec<u32>,
    sorted: Vec<u32>,
}

impl<'t, IndexType: GraphIndexInteger, T: TreeAccess> Locator<'t, IndexType, T> {
    /// A locator for the blocks for which `tree_of(block)` (engine ids) returns a tree.
    pub(super) fn new<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        spqr: &SPQRDecomposition<'_, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>,
        numbering: &BlockNumbering,
        tree_of: impl Fn(usize) -> Option<&'t T>,
    ) -> Self {
        let mut home = vec![None; spqr.spqr_node_count()];
        for b in 0..numbering.len() {
            let block_index = BlockIndex::from_usize(b);
            for (local, (spqr_index, _)) in spqr
                .iter_spqr_nodes_in_block(numbering.old_id(b))
                .enumerate()
            {
                home[spqr_index.into_usize()] = Some((block_index, offset(local)));
            }
        }
        Self {
            home,
            trees: (0..numbering.len()).map(tree_of).collect(),
            list: Vec::new(),
            pairs: Vec::new(),
            blocks: Vec::new(),
            locals: Vec::new(),
            sorted: Vec::new(),
        }
    }

    /// The location field of non-cut node `node` (inline, or pointing into the list), `None` when
    /// it lies in no record block. `block` is its block (engine id).
    pub(super) fn locate_non_cut<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &mut self,
        spqr: &SPQRDecomposition<'_, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>,
        node: NodeIndex<IndexType>,
        block: Option<BlockIndex<IndexType>>,
    ) -> Option<u32> {
        self.collect_record_blocks(spqr, node);
        let located = match self.blocks.as_slice() {
            [] => return None,
            &[located] => located,
            _ => panic!(
                "NodeTopology: non-cut node {} lies in several blocks",
                node.into_usize()
            ),
        };
        assert_eq!(
            block,
            Some(located),
            "a non-cut node is located in its block"
        );
        let loc = self.location_in(located);
        if let Some(field) = pack_loc(loc) {
            return Some(field);
        }
        let first = self.list.len();
        self.list.push(SpqrLoc {
            block: located,
            x: offset(loc.x),
            up_last: loc.up | LOC_LAST,
        });
        Some(pack_list(first))
    }

    /// The first list entry of cut node `node`'s locations (one per record block containing it),
    /// [`chain_nav::NONE`] when it lies in no record block.
    pub(super) fn locate_cut<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &mut self,
        spqr: &SPQRDecomposition<'_, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>,
        node: NodeIndex<IndexType>,
    ) -> u32 {
        self.collect_record_blocks(spqr, node);
        if self.blocks.is_empty() {
            return chain_nav::NONE;
        }
        let first = self.list.len();
        for i in 0..self.blocks.len() {
            let block = self.blocks[i];
            let loc = self.location_in(block);
            assert!(
                loc.up < LOC_LAST,
                "NodeTopology: SPQR location reach overflows"
            );
            self.list.push(SpqrLoc {
                block,
                x: offset(loc.x),
                up_last: loc.up,
            });
        }
        self.list.last_mut().unwrap().up_last |= LOC_LAST;
        list_index(first)
    }

    /// `node`'s SPQR nodes in record blocks (`pairs`, in the decomposition's order) and their
    /// distinct blocks (`blocks`, in first-appearance order).
    fn collect_record_blocks<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &mut self,
        spqr: &SPQRDecomposition<'_, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>,
        node: NodeIndex<IndexType>,
    ) {
        self.pairs.clear();
        self.pairs.extend(
            spqr.node_spqr_node_indices(node)
                .filter_map(|spqr_node| self.home[spqr_node.into_usize()])
                .filter(|(block, _)| self.trees[block.into_usize()].is_some()),
        );
        self.blocks.clear();
        for &(block, _) in &self.pairs {
            if !self.blocks.contains(&block) {
                self.blocks.push(block);
            }
        }
    }

    /// The location of the current node in record block `block`, from its locals there.
    fn location_in(&mut self, block: BlockIndex<IndexType>) -> SpqrLocation {
        self.locals.clear();
        self.locals.extend(
            self.pairs
                .iter()
                .filter(|(b, _)| *b == block)
                .map(|&(_, local)| local),
        );
        let tree = self.trees[block.into_usize()].expect("a record block has a tree");
        location_of(tree, &self.locals, &mut self.sorted)
    }
}
