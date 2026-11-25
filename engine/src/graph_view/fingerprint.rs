//! The graph fingerprint: a hash over the node lengths and every arc, to match a graph to an
//! index.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::GfaNodeData,
};

use crate::{gfa_graph_extensions::GfaNodeDataExt, image::hash, path::GfaPathLength};

/// Compute the fingerprint of an in-memory graph without writing it (for `index`/`query` checks).
pub fn graph_fingerprint<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
) -> u64 {
    let n = graph.node_count();
    let mut fp: u64 = hash::mix(n as u64 ^ 0x9e37_79b9_7f4a_7c15);
    for v in 0..n {
        let len: GfaPathLength<IndexType> = graph.node_data(NodeIndex::from_usize(v)).len();
        fp = hash::mix(fp ^ len.into_usize() as u64);
    }
    for d in 0..2 * n {
        for edge in graph.iter_outgoing_edges(DirectedNodeIndex::from_usize(d)) {
            fp = hash::mix(fp ^ ((d as u64) << 32 | edge.to().into_usize() as u64));
        }
    }
    fp
}
