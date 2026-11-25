//! Node lengths: the records' inline lengths and the long-length table.

use bidirected_adjacency_array::index::GraphIndexInteger;
#[cfg(test)]
use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray, index::NodeIndex, io::gfa1::GfaNodeData,
};

#[cfg(test)]
use super::record::{K_SIMPLE, NO_BLOCK};
use super::record::{LEN_ESCAPE, NodeRecord};
#[cfg(test)]
use crate::gfa_graph_extensions::GfaNodeDataExt;
use crate::{
    chain_nav::{self, RankBits, RankBlock},
    image::{Image, Sec},
    path::GfaPathLength,
    sections,
};

/// The node lengths: the records' inline lengths and the long-length table.
#[derive(Clone, Copy)]
pub struct NodeLens<'a, IndexType: GraphIndexInteger> {
    records: &'a [NodeRecord<IndexType>],
    long: RankBits<'a>,
    long_lens: &'a [GfaPathLength<IndexType>],
}

impl<IndexType: GraphIndexInteger> NodeLens<'_, IndexType> {
    /// The number of graph nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.records.len() - 1
    }

    /// The sequence length of node `v`.
    #[inline]
    pub fn len(&self, v: usize) -> GfaPathLength<IndexType> {
        let len = self.records[v].lo & LEN_ESCAPE;
        if len != LEN_ESCAPE {
            GfaPathLength::from_usize(len as usize)
        } else {
            self.long_lens[self.long.rank(v)]
        }
    }
}

/// Typed handles to the node-length sections (what the graph view reads).
#[derive(Clone, Copy, Debug)]
pub struct NodeLenSecs<IndexType: GraphIndexInteger> {
    pub(super) records: Sec<NodeRecord<IndexType>>,
    pub(super) long_rank: Sec<RankBlock>,
    pub(super) long_lens: Sec<GfaPathLength<IndexType>>,
}

impl<IndexType: GraphIndexInteger> NodeLenSecs<IndexType> {
    /// Resolve against the image.
    pub fn view<'a>(&self, image: &'a Image) -> NodeLens<'a, IndexType> {
        NodeLens {
            records: image.get(self.records),
            long: RankBits {
                blocks: image.get(self.long_rank),
            },
            long_lens: image.get(self.long_lens),
        }
    }

    /// Look the sections up in a loaded image.
    pub fn from_image(image: &Image) -> std::io::Result<Self> {
        Ok(Self {
            records: image.require(sections::TOPO_NODE_RECS, "topology.node_recs")?,
            long_rank: image.require(sections::TOPO_LONG_LEN_WORDS, "topology.long_len_rank")?,
            long_lens: image.require(sections::TOPO_LONG_LENS, "topology.long_lens")?,
        })
    }

    /// Write the length sections of `lens` (one per node) with `records` built by `record(v, len)`
    /// (plus a sentinel).
    pub(super) fn write(
        image: &Image,
        lens: impl Iterator<Item = GfaPathLength<IndexType>> + Clone,
        records: &[NodeRecord<IndexType>],
    ) -> Self {
        let n = records.len() - 1;
        let long: Vec<(usize, GfaPathLength<IndexType>)> = lens
            .enumerate()
            .filter(|(_, len)| len.into_usize() >= LEN_ESCAPE as usize)
            .collect();
        let blocks = if long.is_empty() {
            Vec::new()
        } else {
            chain_nav::rank_bits(n, long.iter().map(|&(v, _)| v), 0)
        };
        let values: Vec<GfaPathLength<IndexType>> = long.into_iter().map(|(_, len)| len).collect();
        Self {
            records: image.push_section(sections::TOPO_NODE_RECS, records),
            long_rank: image.push_section(sections::TOPO_LONG_LEN_WORDS, &blocks),
            long_lens: image.push_section(sections::TOPO_LONG_LENS, &values),
        }
    }

    /// The length sections of a bare graph (records with lengths only: no blocks, cuts or SPQR
    /// nodes), for the graph view's tests.
    #[cfg(test)]
    pub(crate) fn write_plain<NodeData: GfaNodeData, EdgeData>(
        image: &Image,
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    ) -> Self {
        let n = graph.node_count();
        let len = |v: usize| -> GfaPathLength<IndexType> {
            graph.node_data(NodeIndex::from_usize(v)).len()
        };
        let records: Vec<NodeRecord<IndexType>> = (0..=n)
            .map(|v| {
                NodeRecord::new(
                    if v < n { len(v).into_usize() } else { 0 },
                    K_SIMPLE,
                    0,
                    NO_BLOCK,
                )
            })
            .collect();
        Self::write(image, (0..n).map(len), &records)
    }
}
