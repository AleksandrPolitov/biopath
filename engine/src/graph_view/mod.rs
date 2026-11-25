//! The query-time view of the input graph, as stored in the index image: node lengths (read from the
//! topology's node records), the directed adjacency (CSR), and the GFA node names.
//!
//! The query path reads exactly three things from the graph — a node's length, its outgoing
//! directed arcs (for same-node queries and the block-restricted Dijkstra fallbacks) and, at the CLI
//! boundary, a name's node — so this is all the index needs to be self-contained: `query` runs
//! without the GFA.
//!
//! The lengths are the topology's ([`NodeLens`]). Only the nodes whose arcs a query can search keep
//! them: an O(1)-rank set over the node ids selects their CSR rows
//! ([`GraphSections::write_adjacency`]). Names are runs of consecutive integers when every name is
//! one, else a byte table with a hash index (`names.rs`). The fingerprint over the lengths and the
//! arcs ([`graph_fingerprint`]) ties an index to its graph.

mod fingerprint;
mod names;
#[cfg(test)]
mod tests;

pub use fingerprint::graph_fingerprint;
pub(crate) use names::{GRAPH_NAME_IDS, NameIds, NameSections, NameView};
pub use names::{NameRun, integer_name_runs};

use std::borrow::Cow;

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::GfaNodeData,
};

use crate::{
    chain_nav::{RankBits, RankBlock},
    gfa_graph_extensions::GfaNodeDataExt,
    image::{Image, Sec},
    path::GfaPathLength,
    sections,
    topology::{NodeLenSecs, NodeLens},
};

/// What the engine reads from a graph at query time. Implemented by the input adjacency array and
/// by the image-backed [`GraphView`].
pub trait QueryGraph<IndexType: GraphIndexInteger> {
    /// The number of (bidirected) nodes.
    fn node_count(&self) -> usize;
    /// The sequence length of a node.
    fn node_len(&self, node: NodeIndex<IndexType>) -> GfaPathLength<IndexType>;
    /// The directed nodes reached by one arc out of `node`.
    fn out_neighbors(
        &self,
        node: DirectedNodeIndex<IndexType>,
    ) -> impl Iterator<Item = DirectedNodeIndex<IndexType>> + '_;
}

impl<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData> QueryGraph<IndexType>
    for BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>
{
    #[inline]
    fn node_count(&self) -> usize {
        BidirectedAdjacencyArray::node_count(self)
    }
    #[inline]
    fn node_len(&self, node: NodeIndex<IndexType>) -> GfaPathLength<IndexType> {
        self.node_data(node).len()
    }
    #[inline]
    fn out_neighbors(
        &self,
        node: DirectedNodeIndex<IndexType>,
    ) -> impl Iterator<Item = DirectedNodeIndex<IndexType>> + '_ {
        self.iter_outgoing_edges(node).map(|edge| edge.to())
    }
}

/// Typed handles to the graph's sections.
#[derive(Clone, Copy, Debug)]
pub struct GraphSections<IndexType: GraphIndexInteger> {
    /// The topology's node-length sections: node lengths come from them.
    lens: NodeLenSecs<IndexType>,
    /// The adjacency, written once the engine knows which nodes keep their arcs (see
    /// [`Self::write_adjacency`]); `None` while the engine is built.
    adj: Option<AdjSecs<IndexType>>,
    names: NameSections,
}

/// The adjacency sections: the nodes that keep their arcs, and the CSR over their directed states
/// (rank order).
#[derive(Clone, Copy, Debug)]
struct AdjSecs<IndexType: GraphIndexInteger> {
    rank: Sec<RankBlock>,
    offsets: Sec<u32>,
    targets: Sec<DirectedNodeIndex<IndexType>>,
}

impl<IndexType: GraphIndexInteger> GraphSections<IndexType> {
    /// Write the graph's names into `image` and return the handles plus the graph fingerprint (over
    /// the lengths and every arc). `lens` are the already-written node-length sections. The adjacency
    /// follows with [`Self::write_adjacency`].
    pub fn write<NodeData: GfaNodeData, EdgeData>(
        image: &Image,
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        lens: NodeLenSecs<IndexType>,
    ) -> (Self, u64) {
        let n = graph.node_count();
        debug_assert_eq!(lens.view(image).node_count(), n, "a length per node");
        let fingerprint = graph_fingerprint(graph);
        let names = NameSections::write(
            image,
            n,
            |v| graph.node_data(NodeIndex::from_usize(v)).name(),
            GRAPH_NAME_IDS,
        );
        (
            Self {
                lens,
                adj: None,
                names,
            },
            fingerprint,
        )
    }

    /// Write the adjacency of the nodes `keep` accepts (an O(1)-rank set over node ids and the CSR
    /// over their directed states, in rank order).
    pub fn write_adjacency<NodeData: GfaNodeData, EdgeData>(
        &mut self,
        image: &Image,
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        keep: impl Fn(usize) -> bool,
    ) {
        let n = graph.node_count();
        let kept: Vec<usize> = (0..n).filter(|&v| keep(v)).collect();
        let mut offsets = Vec::with_capacity(2 * kept.len() + 1);
        offsets.push(0u32);
        let mut targets: Vec<DirectedNodeIndex<IndexType>> = Vec::new();
        for &v in &kept {
            for forward in [true, false] {
                let dn = NodeIndex::<IndexType>::from_usize(v).into_directed(forward);
                debug_assert_eq!(
                    dn.into_usize(),
                    directed_slot(v, forward),
                    "directed ids are 2v + o"
                );
                targets.extend(graph.iter_outgoing_edges(dn).map(|edge| edge.to()));
                offsets.push(u32::try_from(targets.len()).unwrap_or_else(|_| {
                    panic!(
                        "the graph's nodes outside simple blocks have more than {} directed arcs, a \
                         hard limit of the index: this graph cannot be indexed",
                        u32::MAX
                    )
                }));
            }
        }
        self.adj = Some(AdjSecs {
            rank: image.push_section(
                sections::GRAPH_ADJ_RANK,
                &crate::chain_nav::rank_bits(n, kept, 0),
            ),
            offsets: image.push_section(sections::GRAPH_ADJ_OFFSETS, &offsets),
            targets: image.push_section(sections::GRAPH_ADJ_TARGETS, &targets),
        });
    }

    /// Look the sections up in a loaded image.
    pub fn from_image(image: &Image) -> std::io::Result<Self> {
        Ok(Self {
            lens: NodeLenSecs::from_image(image)?,
            adj: Some(AdjSecs {
                rank: image.require(sections::GRAPH_ADJ_RANK, "graph.adj_rank")?,
                offsets: image.require(sections::GRAPH_ADJ_OFFSETS, "graph.adj_offsets")?,
                targets: image.require(sections::GRAPH_ADJ_TARGETS, "graph.adj_targets")?,
            }),
            names: NameSections::from_image(image, GRAPH_NAME_IDS)?,
        })
    }

    /// Resolve against the image.
    pub fn view<'a>(&self, image: &'a Image) -> GraphView<'a, IndexType> {
        GraphView {
            lens: self.lens.view(image),
            adj_rank: RankBits {
                blocks: self.adj.map_or(&[][..], |a| image.get(a.rank)),
            },
            adj_offsets: self.adj.map_or(&[][..], |a| image.get(a.offsets)),
            adj_targets: self.adj.map_or(&[][..], |a| image.get(a.targets)),
            names: self.names.view(image),
        }
    }
}

/// The image-backed graph (see the module docs).
#[derive(Clone, Copy)]
pub struct GraphView<'a, IndexType: GraphIndexInteger> {
    /// The node lengths.
    lens: NodeLens<'a, IndexType>,
    /// The nodes that keep their arcs and their CSR (see [`GraphSections::write_adjacency`]).
    adj_rank: RankBits<'a>,
    adj_offsets: &'a [u32],
    adj_targets: &'a [DirectedNodeIndex<IndexType>],
    names: NameView<'a>,
}

impl<'a, IndexType: GraphIndexInteger> GraphView<'a, IndexType> {
    /// Whether node `v` keeps its arcs in this index (see [`GraphSections::write_adjacency`]).
    #[inline]
    pub fn has_adjacency(&self, v: usize) -> bool {
        !self.adj_rank.blocks.is_empty() && self.adj_rank.contains(v)
    }

    /// The GFA name of a node.
    pub fn node_name(&self, node: NodeIndex<IndexType>) -> Cow<'a, str> {
        self.names.node_name(node.into_usize())
    }

    /// The node with the given GFA name, if any.
    pub fn node_by_name(&self, name: &str) -> Option<NodeIndex<IndexType>> {
        self.names.node_by_name(name).map(NodeIndex::from_usize)
    }

    /// [`QueryGraph::out_neighbors`], borrowing the image rather than this view. A node without
    /// arcs (see [`Self::has_adjacency`]) is a caller bug.
    #[inline]
    pub fn image_out_neighbors(
        self,
        node: DirectedNodeIndex<IndexType>,
    ) -> impl Iterator<Item = DirectedNodeIndex<IndexType>> + use<'a, IndexType> {
        let v = node.into_bidirected().into_usize();
        assert!(
            self.has_adjacency(v),
            "node {v} keeps no arcs in this index"
        );
        let d = 2 * self.adj_rank.rank(v) + (node.into_usize() - directed_slot(v, true));
        self.adj_targets[self.adj_offsets[d] as usize..self.adj_offsets[d + 1] as usize]
            .iter()
            .copied()
    }
}

impl<IndexType: GraphIndexInteger> QueryGraph<IndexType> for GraphView<'_, IndexType> {
    #[inline]
    fn node_count(&self) -> usize {
        self.lens.node_count()
    }
    #[inline]
    fn node_len(&self, node: NodeIndex<IndexType>) -> GfaPathLength<IndexType> {
        self.lens.len(node.into_usize())
    }
    #[inline]
    /// A node without arcs (see [`GraphView::has_adjacency`]) is a caller bug.
    fn out_neighbors(
        &self,
        node: DirectedNodeIndex<IndexType>,
    ) -> impl Iterator<Item = DirectedNodeIndex<IndexType>> + '_ {
        self.image_out_neighbors(node)
    }
}

/// The directed id of node `v` in orientation `forward` (`2v` forward, `2v + 1` reverse).
#[inline]
fn directed_slot(v: usize, forward: bool) -> usize {
    2 * v + usize::from(!forward)
}
