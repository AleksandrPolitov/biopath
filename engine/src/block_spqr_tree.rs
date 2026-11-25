//! A rooted SPQR tree for a single biconnected block, with the reference (parent) virtual edge and
//! its poles recorded for every non-root node.
//!
//! `spqr-tree` exposes a block's SPQR tree as a set of SPQR nodes joined by SPQR (virtual) edges,
//! each virtual edge carrying its two pole segments. This type roots that tree (via
//! [`RootedTree`]) and, for every non-root SPQR node, records:
//! - its **reference poles** `(s, t)`: the poles of the virtual edge to its parent, which the
//!   bottom-up DP will treat as the node's two ports; and
//! - its **child virtual edges**: the poles of the virtual edge to each child, which the DP will
//!   weight by the child's inside distance matrix.
//!
//! Real (non-virtual) skeleton edges are recorded as graph edge indices; [`BlockSpqrTree::real_directed_edges`]
//! resolves them to oriented endpoints against the concrete bidirected graph, which is where segment
//! orientation enters.

use crate::rooted_tree::TreeAccess;
use std::collections::HashMap;

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, EdgeIndex, GraphIndexInteger, NodeIndex, OptionalNodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::decomposition::{
    SPQRDecomposition, SPQRNodeType,
    indices::{BlockIndex, SPQREdgeIndex, SPQRNodeIndex},
};

use crate::rooted_tree::RootedTree;

#[cfg(test)]
mod tests;

/// A virtual edge from an SPQR node to one of its children, with the child's local id and poles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildVirtualEdge<IndexType: GraphIndexInteger> {
    /// The local id (within this block tree) of the child SPQR node.
    pub child_local: usize,
    /// The two pole segments of the virtual edge shared with the child.
    pub poles: (NodeIndex<IndexType>, NodeIndex<IndexType>),
}

/// Per-node data of a [`BlockSpqrTree`].
#[derive(Debug, Clone)]
pub struct BlockSpqrNode<IndexType: GraphIndexInteger> {
    node_type: SPQRNodeType,
    real_edges: Box<[EdgeIndex<IndexType>]>,
    /// Both entries are sentinel-none at the root; otherwise the two reference poles. This avoids
    /// the four-byte enum tag/padding of `Option<(NodeIndex, NodeIndex)>` at every SPQR node.
    reference_poles: [OptionalNodeIndex<IndexType>; 2],
    child_virtual_edges: Box<[ChildVirtualEdge<IndexType>]>,
}

impl<IndexType: GraphIndexInteger> BlockSpqrNode<IndexType> {
    /// The SPQR node type (S / P / R).
    pub fn node_type(&self) -> SPQRNodeType {
        self.node_type
    }

    /// The graph edge indices of the real (non-virtual) skeleton edges.
    pub fn real_edges(&self) -> &[EdgeIndex<IndexType>] {
        &self.real_edges
    }

    /// The poles `(s, t)` of the virtual edge to the parent, or `None` for the root.
    pub fn reference_poles(&self) -> Option<(NodeIndex<IndexType>, NodeIndex<IndexType>)> {
        let x = self.reference_poles[0].into_option()?;
        Some((
            x,
            self.reference_poles[1].expect("both reference poles are present"),
        ))
    }

    /// The virtual edges to the children.
    pub fn child_virtual_edges(&self) -> &[ChildVirtualEdge<IndexType>] {
        &self.child_virtual_edges
    }
}

/// A rooted SPQR tree for one block.
#[derive(Debug, Clone)]
pub struct BlockSpqrTree<IndexType: GraphIndexInteger> {
    tree: RootedTree,
    nodes: Box<[BlockSpqrNode<IndexType>]>,
    /// Optional reverse lookup (see [`Self::new_without_lookup`]).
    local_by_spqr: Option<HashMap<SPQRNodeIndex<IndexType>, usize>>,
}

impl<IndexType: GraphIndexInteger> BlockSpqrTree<IndexType> {
    /// Build and root the SPQR tree of `block` at local id `0`.
    ///
    /// The root is the first SPQR node the decomposition lists for the block. For the index
    /// the rooting is arbitrary and does not affect the distances it computes; it only fixes each
    /// node's reference (parent) virtual edge.
    pub fn new<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        spqr_decomposition: &SPQRDecomposition<
            '_,
            BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        >,
        block: BlockIndex<IndexType>,
    ) -> Self {
        Self::build(spqr_decomposition, block, true)
    }

    /// Like [`Self::new`] (same root, same local ids, children retained) but without the
    /// `SPQRNodeIndex -> local` lookup, so [`Self::local_of`] returns `None`. The index resolves
    /// SPQR nodes through [`crate::topology::NodeTopology`] instead, which saves one hash map per
    /// block.
    pub fn new_without_lookup<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        spqr_decomposition: &SPQRDecomposition<
            '_,
            BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        >,
        block: BlockIndex<IndexType>,
    ) -> Self {
        Self::build(spqr_decomposition, block, false)
    }

    fn build<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        spqr_decomposition: &SPQRDecomposition<
            '_,
            BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        >,
        block: BlockIndex<IndexType>,
        keep_lookup: bool,
    ) -> Self {
        // Collect the block's SPQR nodes and assign contiguous local ids.
        let mut spqr_node_indices = Vec::new();
        let mut local_by_spqr = HashMap::new();
        let mut node_types = Vec::new();
        let mut real_edges_per_node = Vec::new();
        for (spqr_index, spqr_node) in spqr_decomposition.iter_spqr_nodes_in_block(block) {
            local_by_spqr.insert(spqr_index, spqr_node_indices.len());
            spqr_node_indices.push(spqr_index);
            node_types.push(spqr_node.spqr_node_type());
            real_edges_per_node.push(spqr_node.iter_edges().collect::<Vec<_>>());
        }
        let node_count = spqr_node_indices.len();

        // Tree edges (as local-id pairs) plus a lookup from an unordered local pair to its virtual
        // edge, so we can recover the poles for any parent/child relationship after rooting.
        let mut tree_edges = Vec::new();
        let mut edge_by_local_pair: HashMap<(usize, usize), SPQREdgeIndex<IndexType>> =
            HashMap::new();
        for (spqr_edge_index, spqr_edge) in spqr_decomposition.iter_spqr_edges_in_block(block) {
            let (u, v) = spqr_edge.endpoints();
            let local_u = local_by_spqr[&u];
            let local_v = local_by_spqr[&v];
            edge_by_local_pair.insert(
                (local_u.min(local_v), local_u.max(local_v)),
                spqr_edge_index,
            );
            tree_edges.push((local_u, local_v));
        }

        let tree = RootedTree::new(node_count, &tree_edges, 0);

        let poles_of = |a: usize, b: usize| {
            let spqr_edge_index = edge_by_local_pair[&(a.min(b), a.max(b))];
            spqr_decomposition.spqr_edge(spqr_edge_index).virtual_edge()
        };

        let nodes: Vec<_> = (0..node_count)
            .map(|local| {
                let reference_poles = tree.parent(local).map_or_else(
                    || [OptionalNodeIndex::new_none(); 2],
                    |parent| {
                        let (x, y) = poles_of(local, parent);
                        [
                            OptionalNodeIndex::from_option(Some(x)),
                            OptionalNodeIndex::from_option(Some(y)),
                        ]
                    },
                );
                let child_virtual_edges: Vec<_> = tree
                    .children(local)
                    .iter()
                    .map(|&child_local| ChildVirtualEdge {
                        child_local,
                        poles: poles_of(local, child_local),
                    })
                    .collect();
                BlockSpqrNode {
                    node_type: node_types[local],
                    real_edges: std::mem::take(&mut real_edges_per_node[local]).into_boxed_slice(),
                    reference_poles,
                    child_virtual_edges: child_virtual_edges.into_boxed_slice(),
                }
            })
            .collect();

        Self {
            tree,
            nodes: nodes.into_boxed_slice(),
            local_by_spqr: keep_lookup.then_some(local_by_spqr),
        }
    }

    /// The rooted tree over the block's SPQR nodes (local ids `0..node_count`).
    pub fn tree(&self) -> &RootedTree {
        &self.tree
    }

    /// The number of SPQR nodes in the block.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The local id of the root SPQR node.
    #[cfg(test)]
    pub fn root_local(&self) -> usize {
        self.tree.root()
    }

    /// The per-node data for the SPQR node with the given local id.
    pub fn node(&self, local: usize) -> &BlockSpqrNode<IndexType> {
        &self.nodes[local]
    }

    /// The parent virtual-edge poles for a local id, available in both layouts.
    pub fn reference_poles(
        &self,
        local: usize,
    ) -> Option<(NodeIndex<IndexType>, NodeIndex<IndexType>)> {
        let poles = self.nodes[local].reference_poles;
        let x = poles[0].into_option()?;
        Some((x, poles[1].expect("both reference poles are present")))
    }

    /// The local id of an SPQR node, if it belongs to this block.
    pub fn local_of(&self, spqr_node: SPQRNodeIndex<IndexType>) -> Option<usize> {
        self.local_by_spqr
            .as_ref()
            .and_then(|map| map.get(&spqr_node).copied())
    }

    /// Resolve the real skeleton edges of a node to oriented endpoints against the graph.
    ///
    /// Each real edge is a GFA link; its `from`/`to` are the oriented endpoints of that link. The
    /// reverse-complement traversal is implicit and added when the directed skeleton is assembled in
    /// a later phase.
    pub fn real_directed_edges<'a, NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &'a self,
        local: usize,
        graph: &'a BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    ) -> impl ExactSizeIterator<Item = RealSkeletonEdge<IndexType>> + 'a {
        self.nodes[local].real_edges.iter().map(|&edge_index| {
            let view = graph.edge(edge_index);
            RealSkeletonEdge {
                edge: edge_index,
                from: view.from(),
                to: view.to(),
            }
        })
    }
}

/// A real skeleton edge resolved to oriented endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealSkeletonEdge<IndexType: GraphIndexInteger> {
    /// The graph edge index of the link.
    pub edge: EdgeIndex<IndexType>,
    /// The oriented endpoint the link leaves.
    pub from: DirectedNodeIndex<IndexType>,
    /// The oriented endpoint the link enters.
    pub to: DirectedNodeIndex<IndexType>,
}
