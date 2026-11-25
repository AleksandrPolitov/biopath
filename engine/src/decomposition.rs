//! Build an SPQR decomposition in-process from a plain GFA graph.
//!
//! The engine is built from *just* a GFA: no external `spqr-rust` CLI, no `.spqr` file, and no text
//! round-trip. [`build_spqr_decomposition`] runs the correct per-block pipeline — connected
//! components, a block-cut tree, then `build_spqr` on each *biconnected block* (the only input
//! `build_spqr` is defined for) — and drives [`SPQRDecompositionBuilder`] directly, passing real
//! graph node/edge indices.
//!
//! Building directly, rather than serialising to a `.spqr` document and re-parsing it with
//! `read_plain_spqr`, matters for correctness *and* speed: the reader resolves each edge by scanning
//! an endpoint's incident-edge list (`edges_between`), which is O(Σ degree²) and explodes on
//! high-degree pangenome hub nodes. Here every edge is assigned by its known index in O(1), and there
//! is no multi-GB intermediate text.
//!
//! The steps: `SpqrInput` maps the graph to spqr-rust ids; the components, then every block of the
//! block-cut tree (`Assembly::add_block`: a bond as one P-node, else `build_spqr` on the block and
//! its tree's SPQR nodes, virtual edges and real edges), then the self-loops and the cut nodes go
//! into the [`SPQRDecompositionBuilder`]. One `info` line logs the size of the block-cut tree.

use std::collections::{HashMap, HashSet};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{EdgeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_rust::{
    BCTree, ConnectedComponents, EdgeId, Graph as SpqrGraph, NodeId, SpqrNodeType, SpqrTree,
    TreeNodeId, build_spqr, connected::connected_components,
};
use spqr_tree::{
    decomposition::{
        SPQRDecomposition, SPQRNodeType,
        builder::SPQRDecompositionBuilder,
        indices::{BlockIndex, ComponentIndex, SPQRNodeIndex},
    },
    graph::StaticGraph,
};

/// Build the multi-component, multi-block [`SPQRDecomposition`] of `graph` directly (no `.spqr` text).
pub fn build_spqr_decomposition<'graph, IndexType, NodeData, EdgeData>(
    graph: &'graph BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
) -> SPQRDecomposition<'graph, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    let input = SpqrInput::of(graph);
    let components = connected_components(&input.graph);
    let bc = BCTree::build(&input.graph);
    log::info!(
        "[decomp] block-cut tree: {} blocks, {} cut nodes",
        bc.num_blocks(),
        bc.cut_vertices().len()
    );

    let mut builder = SPQRDecompositionBuilder::new(graph);
    let component_index_of = add_components(&mut builder, &input, &components);
    let mut assembly = Assembly {
        input: &input,
        components: &components,
        component_index_of,
        block_index_of: vec![None; bc.num_blocks()],
        blocks_of_node: HashMap::new(),
        spqr_node_of_node: HashMap::new(),
        covered_edge: vec![false; graph.edge_count()],
    };
    assembly.add_blocks(&mut builder, &bc);
    assembly.add_self_loops(&mut builder);
    assembly.add_cut_nodes(&mut builder, &bc);
    builder.build()
}

type Builder<'graph, IndexType, NodeData, EdgeData> =
    SPQRDecompositionBuilder<'graph, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>;

/// The graph as spqr-rust takes it: compact node ids and the underlying undirected segment graph.
struct SpqrInput<IndexType> {
    /// spqr-rust node id -> original node index (spqr id == enumeration order).
    node_of_id: Vec<NodeIndex<IndexType>>,
    /// spqr edge `i` is the i-th original edge, so `edge_of_spqr[i]` is that original `EdgeIndex`
    /// and `i` doubles as the original edge ordinal.
    edge_of_spqr: Vec<EdgeIndex<IndexType>>,
    graph: SpqrGraph,
}

impl<IndexType: GraphIndexInteger> SpqrInput<IndexType> {
    fn of<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    ) -> Self {
        let node_count = graph.node_count();
        let mut id_of: HashMap<_, u32> = HashMap::with_capacity(node_count);
        let mut node_of_id = Vec::with_capacity(node_count);
        for node in graph.node_indices() {
            id_of.insert(node, node_of_id.len() as u32);
            node_of_id.push(node);
        }

        let mut spqr_graph = SpqrGraph::with_capacity(node_count, graph.edge_count());
        spqr_graph.add_nodes_fast(node_count);
        let mut edge_of_spqr = Vec::with_capacity(graph.edge_count());
        for edge in graph.edge_indices() {
            let view = graph.edge(edge);
            let u = id_of[&view.from().into_bidirected()];
            let v = id_of[&view.to().into_bidirected()];
            spqr_graph.add_edge(NodeId(u), NodeId(v));
            edge_of_spqr.push(edge);
        }
        Self {
            node_of_id,
            edge_of_spqr,
            graph: spqr_graph,
        }
    }

    /// The original node of a spqr-rust node.
    fn node(&self, id: NodeId) -> NodeIndex<IndexType> {
        self.node_of_id[id.0 as usize]
    }

    /// The original edge of a spqr-rust edge.
    fn edge(&self, id: EdgeId) -> EdgeIndex<IndexType> {
        self.edge_of_spqr[id.0 as usize]
    }
}

/// Add every component (they must precede the blocks; each node lands in exactly one), in
/// component id order; the builder's index of every spqr-rust component id.
fn add_components<IndexType, NodeData, EdgeData>(
    builder: &mut Builder<'_, IndexType, NodeData, EdgeData>,
    input: &SpqrInput<IndexType>,
    components: &ConnectedComponents,
) -> HashMap<u32, ComponentIndex<IndexType>>
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    let mut component_nodes: HashMap<u32, Vec<u32>> = HashMap::new();
    for id in 0..input.node_of_id.len() as u32 {
        component_nodes
            .entry(components.component_of(NodeId(id)))
            .or_default()
            .push(id);
    }
    let mut component_ids: Vec<u32> = component_nodes.keys().copied().collect();
    component_ids.sort_unstable();
    let mut component_index_of = HashMap::with_capacity(component_ids.len());
    for component in &component_ids {
        let nodes = component_nodes[component]
            .iter()
            .map(|&id| input.node(NodeId(id)))
            .collect();
        component_index_of.insert(*component, builder.add_component(nodes));
    }
    component_index_of
}

/// The blocks, self-loops and cut nodes going into the builder, and what they need of each other.
struct Assembly<'a, IndexType> {
    input: &'a SpqrInput<IndexType>,
    components: &'a ConnectedComponents,
    component_index_of: HashMap<u32, ComponentIndex<IndexType>>,
    /// The builder's index of every block-cut tree block.
    block_index_of: Vec<Option<BlockIndex<IndexType>>>,
    /// The blocks containing each spqr-rust node.
    blocks_of_node: HashMap<u32, Vec<usize>>,
    /// Some SPQR node containing each spqr-rust node, used to host that node's self-loops.
    spqr_node_of_node: HashMap<u32, SPQRNodeIndex<IndexType>>,
    /// Blocks partition the non-loop edges; self-loops land in no block, so track coverage.
    covered_edge: Vec<bool>,
}

impl<IndexType: GraphIndexInteger> Assembly<'_, IndexType> {
    /// Every block of `bc` with its SPQR nodes, virtual edges and real edges.
    fn add_blocks<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &mut self,
        builder: &mut Builder<'_, IndexType, NodeData, EdgeData>,
        bc: &BCTree,
    ) {
        for block in 0..bc.num_blocks() {
            self.add_block(builder, bc, block);
        }
    }

    fn add_block<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &mut self,
        builder: &mut Builder<'_, IndexType, NodeData, EdgeData>,
        bc: &BCTree,
        block: usize,
    ) {
        let block_nodes = bc.block_nodes(block);
        let block_edges = bc.block_edges(block);
        let component_index =
            self.component_index_of[&self.components.component_of(block_nodes[0])];

        for &node in block_nodes {
            self.blocks_of_node.entry(node.0).or_default().push(block);
        }
        for &edge in block_edges {
            self.covered_edge[edge.0 as usize] = true;
        }

        let block_index = builder.add_block(
            component_index,
            block_nodes.iter().map(|&n| self.input.node(n)).collect(),
        );
        self.block_index_of[block] = Some(block_index);

        if block_nodes.len() < 3 {
            self.add_bond(builder, block_index, block_nodes, block_edges);
        } else {
            let mut decomposed = BlockDecomposition::build(self.input, block_nodes, block_edges);
            decomposed.add_skeleton_nodes(builder, block_index, &mut self.spqr_node_of_node);
            decomposed.add_virtual_edges(builder, block_index);
            decomposed.add_real_edges(builder, block_edges);
        }
    }

    /// A 2-node block (bridge/bond): the engine needs a block to hold >=1 SPQR node, so carry its
    /// edge(s) on a single P-node whose poles are the block's two nodes.
    fn add_bond<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &mut self,
        builder: &mut Builder<'_, IndexType, NodeData, EdgeData>,
        block_index: BlockIndex<IndexType>,
        block_nodes: &[NodeId],
        block_edges: &[EdgeId],
    ) {
        let pnode = builder.add_spqr_node(
            block_index,
            vec![
                self.input.node(block_nodes[0]),
                self.input.node(block_nodes[1]),
            ],
            SPQRNodeType::PNode,
        );
        for &edge in block_edges {
            builder
                .add_edge_to_spqr_node(self.input.edge(edge), pnode)
                .expect("each block edge is assigned exactly once");
        }
        for &node in block_nodes {
            self.spqr_node_of_node.entry(node.0).or_insert(pnode);
        }
    }

    /// Self-loops (edges in no block). Host on an SPQR node containing the node when there is one
    /// (then the engine can traverse it — e.g. a same-node wraparound query works). For an isolated
    /// segment (only self-loops → a singleton component with no block) the only valid home is the
    /// component level: the builder forbids a 1-node block that carries an edge, and there is no
    /// second node to form a 2-node block. KNOWN LIMITATION: the engine does not traverse a
    /// component-hosted self-loop, so a same-node wraparound query through such an isolated loop
    /// returns `None` instead of the true distance. This affects only isolated self-loop-only
    /// segments; every other path is correct (a self-loop is otherwise a zero-progress cycle).
    fn add_self_loops<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &self,
        builder: &mut Builder<'_, IndexType, NodeData, EdgeData>,
    ) {
        for (ordinal, &covered) in self.covered_edge.iter().enumerate() {
            if covered {
                continue;
            }
            let e = self.input.graph.edge(EdgeId(ordinal as u32));
            let edge = self.input.edge_of_spqr[ordinal];
            if let Some(&spqr_node) = self.spqr_node_of_node.get(&e.src.0) {
                builder
                    .add_edge_to_spqr_node(edge, spqr_node)
                    .expect("each self-loop is hosted exactly once");
            } else {
                let component_index = self.component_index_of[&self.components.component_of(e.src)];
                builder
                    .add_edge_to_component(edge, component_index)
                    .expect("self-loop in a block-less component is hosted at the component");
            }
        }
    }

    /// Cut nodes: the blocks each cut vertex belongs to.
    fn add_cut_nodes<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &self,
        builder: &mut Builder<'_, IndexType, NodeData, EdgeData>,
        bc: &BCTree,
    ) {
        for &cut in bc.cut_vertices() {
            let Some(blocks) = self.blocks_of_node.get(&cut.0) else {
                continue;
            };
            let block_indices = blocks
                .iter()
                .map(|&b| self.block_index_of[b].expect("every block was added"))
                .collect();
            builder.add_cut_node(self.input.node(cut), block_indices);
        }
    }
}

/// One biconnected block of 3+ nodes, decomposed on its own as an induced subgraph with block-local
/// ids, and its SPQR tree's nodes as they are added to the builder.
struct BlockDecomposition<'a, IndexType> {
    input: &'a SpqrInput<IndexType>,
    /// Block-local id -> spqr-rust node id.
    local_node_ids: Vec<u32>,
    tree: SpqrTree,
    /// The builder's SPQR node of every tree node.
    spqr_of_tid: HashMap<u32, SPQRNodeIndex<IndexType>>,
}

impl<'a, IndexType: GraphIndexInteger> BlockDecomposition<'a, IndexType> {
    /// Run `build_spqr` on the block's induced subgraph.
    fn build(
        input: &'a SpqrInput<IndexType>,
        block_nodes: &[NodeId],
        block_edges: &[EdgeId],
    ) -> Self {
        let mut local_of: HashMap<u32, u32> = HashMap::with_capacity(block_nodes.len());
        let mut local_node_ids: Vec<u32> = Vec::with_capacity(block_nodes.len());
        for &node in block_nodes {
            local_of.insert(node.0, local_node_ids.len() as u32);
            local_node_ids.push(node.0);
        }
        let mut block_graph = SpqrGraph::with_capacity(block_nodes.len(), block_edges.len());
        block_graph.add_nodes_fast(block_nodes.len());
        for &edge in block_edges {
            let e = input.graph.edge(edge);
            block_graph.add_edge(NodeId(local_of[&e.src.0]), NodeId(local_of[&e.dst.0]));
        }
        let tree = build_spqr(&block_graph).tree;
        Self {
            input,
            local_node_ids,
            spqr_of_tid: HashMap::with_capacity(tree.len()),
            tree,
        }
    }

    /// The original node of a block-local `NodeId` (in the block graph's id space).
    fn orig_of_local(&self, local: NodeId) -> NodeIndex<IndexType> {
        self.input.node_of_id[self.local_node_ids[local.0 as usize] as usize]
    }

    /// Phase 1: one SPQR node per tree node, on the tree node's (deduplicated) block-local nodes;
    /// records an SPQR node of every block node in `spqr_node_of_node`.
    fn add_skeleton_nodes<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &mut self,
        builder: &mut Builder<'_, IndexType, NodeData, EdgeData>,
        block_index: BlockIndex<IndexType>,
        spqr_node_of_node: &mut HashMap<u32, SPQRNodeIndex<IndexType>>,
    ) {
        // Dedup scratch for a tree node's block-local ids. A reused Vec is safe because `Vec::clear`
        // is O(len); a reused *HashSet* would be a trap — `HashSet::clear` is O(capacity), so a single
        // giant skeleton inflates the capacity and every later (tiny) skeleton then pays O(capacity)
        // to clear it, turning phase 1 into O(n^2). Large skeletons get a fresh, right-sized HashSet.
        let mut seen_small: Vec<u32> = Vec::new();
        for tid in self.tree.iter() {
            let node_type = match self.tree.node_type(tid) {
                SpqrNodeType::S => SPQRNodeType::SNode,
                SpqrNodeType::P => SPQRNodeType::PNode,
                SpqrNodeType::R => SPQRNodeType::RNode,
            };
            let nodes = self.skeleton_nodes(tid, &mut seen_small);
            let spqr_node = builder.add_spqr_node(block_index, nodes, node_type);
            self.spqr_of_tid.insert(tid.0, spqr_node);
            for &local in self.tree.node_mapping_slice(tid) {
                spqr_node_of_node
                    .entry(self.local_node_ids[local.0 as usize])
                    .or_insert(spqr_node);
            }
        }
    }

    /// The original nodes of tree node `tid`, each once, in mapping order.
    fn skeleton_nodes(
        &self,
        tid: TreeNodeId,
        seen_small: &mut Vec<u32>,
    ) -> Vec<NodeIndex<IndexType>> {
        let mapping = self.tree.node_mapping_slice(tid);
        let mut nodes = Vec::with_capacity(mapping.len());
        if mapping.len() <= 32 {
            // The overwhelming majority of skeletons are tiny: a linear scan dedups without
            // hashing or allocating, and `seen_small.clear()` is O(len).
            seen_small.clear();
            for &local in mapping {
                if !seen_small.contains(&local.0) {
                    seen_small.push(local.0);
                    nodes.push(self.orig_of_local(local));
                }
            }
        } else {
            // Rare large skeleton: hash-based dedup in a fresh set sized to this mapping. Dropping
            // it afterwards keeps its capacity from leaking into later, tiny skeletons.
            let mut seen = HashSet::with_capacity(mapping.len());
            for &local in mapping {
                if seen.insert(local.0) {
                    nodes.push(self.orig_of_local(local));
                }
            }
        }
        nodes
    }

    /// Phase 2: the virtual (skeleton) edges, each once (from the lower tree node id).
    fn add_virtual_edges<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &self,
        builder: &mut Builder<'_, IndexType, NodeData, EdgeData>,
        block_index: BlockIndex<IndexType>,
    ) {
        for tid in self.tree.iter() {
            let node_to_original = self.tree.node_mapping_slice(tid);
            for se in self.tree.skeleton_edges_slice(tid) {
                if !se.twin_tree_node.is_valid() || tid.0 >= se.twin_tree_node.0 {
                    continue;
                }
                let pole_a = self.orig_of_local(node_to_original[se.src.idx()]);
                let pole_b = self.orig_of_local(node_to_original[se.dst.idx()]);
                builder.add_spqr_edge(
                    Some(block_index).into(),
                    (
                        self.spqr_of_tid[&tid.0],
                        self.spqr_of_tid[&se.twin_tree_node.0],
                    ),
                    (pole_a, pole_b),
                );
            }
        }
    }

    /// Phase 3: every real edge of the block on the SPQR node of its tree node.
    fn add_real_edges<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &self,
        builder: &mut Builder<'_, IndexType, NodeData, EdgeData>,
        block_edges: &[EdgeId],
    ) {
        for (i, &edge) in block_edges.iter().enumerate() {
            let tid = self.tree.tree_node_of_edge(EdgeId(i as u32));
            if !tid.is_valid() {
                continue;
            }
            builder
                .add_edge_to_spqr_node(self.input.edge(edge), self.spqr_of_tid[&tid.0])
                .expect("each block edge is assigned exactly once");
        }
    }
}
