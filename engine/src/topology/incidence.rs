//! The block-cut incidence of a decomposition: what the block-cut tree and the chain records are
//! built from.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::decomposition::{
    SPQRDecomposition,
    indices::{BlockIndex, ComponentIndex},
};

use super::offset;
use crate::chain_nav;

/// Every block's cut nodes and component, blocks in one numbering, and the cut nodes under
/// provisional ids (node order) until the chain layout assigns the dense ones.
pub(super) struct BlockCutIncidence<IndexType: GraphIndexInteger> {
    /// `cuts[offsets[b]..offsets[b + 1]]`: block `b`'s cut nodes in the decomposition's order.
    offsets: Vec<u32>,
    cuts: Vec<NodeIndex<IndexType>>,
    /// The cut nodes by provisional id.
    pub(super) cut_nodes: Vec<NodeIndex<IndexType>>,
    /// Per node: its provisional cut id ([`chain_nav::NONE`] for a non-cut node).
    pub(super) cut_tmp_of_node: Vec<u32>,
    /// Per block: its connected component.
    pub(super) block_component: Vec<ComponentIndex<IndexType>>,
    /// The decomposition's component count.
    pub(super) component_count: usize,
}

impl<IndexType: GraphIndexInteger> BlockCutIncidence<IndexType> {
    /// Read the incidence off `spqr`, block `b` being the decomposition's block `old_id(b)`.
    pub(super) fn read<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        spqr: &SPQRDecomposition<'_, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>,
        old_id: impl Fn(usize) -> BlockIndex<IndexType>,
    ) -> Self {
        let node_count = spqr.graph().node_count();
        let mut cut_nodes = Vec::new();
        let mut cut_tmp_of_node = vec![chain_nav::NONE; node_count];
        for (v, slot) in cut_tmp_of_node.iter_mut().enumerate() {
            let node = NodeIndex::from_usize(v);
            if spqr.is_cut_node(node) {
                *slot = offset(cut_nodes.len());
                cut_nodes.push(node);
            }
        }
        let blocks: Vec<_> = spqr.iter_blocks().map(|(_, block)| block).collect();
        let mut offsets = Vec::with_capacity(blocks.len() + 1);
        let mut cuts = Vec::new();
        let mut block_component = Vec::with_capacity(blocks.len());
        offsets.push(0);
        for b in 0..blocks.len() {
            let block = blocks[old_id(b).into_usize()];
            cuts.extend(
                block
                    .iter_cut_nodes()
                    .map(|cut_node| spqr.cut_node(cut_node).node()),
            );
            offsets.push(offset(cuts.len()));
            let first = block
                .iter_nodes()
                .next()
                .expect("every block contains a node");
            block_component.push(spqr.node_component_index(first));
        }
        Self {
            offsets,
            cuts,
            cut_nodes,
            cut_tmp_of_node,
            block_component,
            component_count: spqr.component_count(),
        }
    }

    pub(super) fn block_count(&self) -> usize {
        self.block_component.len()
    }

    pub(super) fn cut_count(&self) -> usize {
        self.cut_nodes.len()
    }

    /// Block `b`'s cut nodes, in the decomposition's order.
    pub(super) fn block_cuts(&self, b: usize) -> &[NodeIndex<IndexType>] {
        &self.cuts[self.offsets[b] as usize..self.offsets[b + 1] as usize]
    }

    /// The provisional cut id of cut node `cut`.
    pub(super) fn cut_tmp(&self, cut: NodeIndex<IndexType>) -> usize {
        self.cut_tmp_of_node[cut.into_usize()] as usize
    }
}
