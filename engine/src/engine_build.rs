//! The engine build: from a graph and its SPQR decomposition to a finished index [`Image`] (see
//! [`DistanceIndex`] for what the image holds and how queries read it).
//!
//! [`build_engine`] runs the stages below in order. The order of the image writes fixes the
//! section offsets and the table of contents, so it is part of the format.
//!
//! 1. **Prepare** ([`prepare`]), the only stage that reads the decomposition: the engine's block
//!    numbering (chain order), the SPQR tree of every block that is not small ([`SmallBlocks`])
//!    and the node topology. An owned decomposition is dropped right after, before the per-block
//!    stage allocates the bulk of the engine.
//! 2. **Graph names**: node names and lengths, and the graph fingerprint.
//! 3. **Block records** ([`blocks`]): every block, children first over the block-cut forest. A
//!    block that is not small gets a record (the DP pipeline of [`record`]); every block is then
//!    folded upward for flip baking.
//! 4. **Lift** ([`build_lift`]): the block-cut lift over the finished blocks.
//! 5. **Adjacency** ([`write_adjacency`]): the arcs of every node except the non-cut nodes of
//!    simple blocks.
//! 6. **Block directory** and the **record scan** for the largest skeleton a query searches.
//! 7. **Shrink sections** (a shrunk graph only: the original graph's names and the alias) and the
//!    **header** ([`image_header`]), which closes the image.
//!
//! The [`BuildOptions`] reach the stages as arguments: the small-block bound the prepare and
//! per-block stages, the tiers the records, the lift options the lift and the
//! shrink sink the shrink sections; the header records the build-affecting values. The build reads
//! no environment variable for them.
//!
//! The build logs one `info` line when the image is finished ([`build_report::log_summary`]).

mod blocks;
mod record;

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use rayon::prelude::*;
use spqr_tree::decomposition::{SPQRDecomposition, indices::BlockIndex};

use crate::{
    block_cut_lift::{BakedForest, BlockCutLift, LiftOptions, LiftSections},
    block_record::{BlockDirRec, BlockView, sparse_block_dir},
    block_spqr_tree::BlockSpqrTree,
    build_report,
    distance_index::{BuildOptions, Decomposition, DistanceIndex, EngineSections},
    graph_view::{GraphSections, GraphView},
    image::{Header, Image},
    sections::{self, knob},
    shrink::ShrinkSink,
    skeleton::SkeletonAccess,
    topology::{BlockNumbering, BuildLoc, NodeTopology, TopologySections, chain_block_numbering},
};
use blocks::BlockRecords;

type Graph<IndexType, NodeData, EdgeData> = BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>;

/// Build the engine for `graph` (see the module docs).
pub(super) fn build_engine<IndexType, NodeData, EdgeData>(
    graph: &Graph<IndexType, NodeData, EdgeData>,
    spqr: Decomposition<'_, '_, Graph<IndexType, NodeData, EdgeData>>,
    options: &BuildOptions<'_>,
) -> DistanceIndex<IndexType>
where
    IndexType: GraphIndexInteger + Send + Sync + 'static,
    NodeData: GfaNodeData + Send + Sync,
    EdgeData: GfaEdgeData + Send + Sync,
{
    let image = Image::new_default_arena().unwrap_or_else(|e| panic!("{e}"));

    let prepared = prepare(spqr.get(), &image, options.small_blocks);
    let (mut graph_secs, fingerprint) =
        GraphSections::write(&image, graph, prepared.topology.lens());
    // Until the lift lays the rows out, the record-block nodes are located through `build_locs`.
    let views = BuildViews {
        graph,
        image: &image,
        topology: prepared.topology.view_build(&image, &prepared.build_locs),
        graph_view: graph_secs.view(&image),
    };
    // An owned decomposition is released here, before the per-block stage.
    drop(spqr);

    let (offsets, baked) = BlockRecords::new(&views, options.tiers).build(prepared.blocks);
    let (lift, lift_secs) = build_lift(&views, baked, &options.lift);
    // The lift re-pointed the node records at its row layout: re-resolve the view.
    let topology = prepared.topology.view(&image);
    write_adjacency(&image, graph, &mut graph_secs, topology, &lift);

    let block_dir = sparse_block_dir(&offsets);
    let block_dir_sec = image.push_section(sections::BLOCK_DIR_SPARSE, &block_dir);
    let max_searched_nodes = max_searched_skeleton::<IndexType>(&image, &block_dir);
    let mut header = image_header(graph, fingerprint, options, max_searched_nodes);
    let shrink_secs = options.shrink.map(|sink| {
        mark_shrunk(&mut header, sink);
        sink.write(&image)
    });
    image.finish(header);
    build_report::log_summary(&image, block_dir.len(), offsets.len() - block_dir.len());

    let handles = EngineSections {
        graph: graph_secs,
        topology: prepared.topology,
        block_dir: block_dir_sec,
        lift: lift_secs,
        shrink: shrink_secs,
    };
    let engine = DistanceIndex::assemble(image, handles, max_searched_nodes);
    settle_allocator();
    engine
}

/// The small-block fast path (the build option
/// [`BuildOptions::small_blocks`](crate::distance_index::BuildOptions::small_blocks)): a block
/// with at most `max_nodes` graph nodes gets no DP pipeline and no record; its in-block distances
/// come from an exact block-restricted Dijkstra over at most `2 * max_nodes` directed states.
/// Cross-block queries never enter such a block: their seeds and reads come from the node cut rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallBlocks {
    /// Whether the fast path is on (off: every block gets a record).
    pub enabled: bool,
    /// The most graph nodes of a small block. The header records it even with the fast path off.
    pub max_nodes: usize,
}

/// The default [`SmallBlocks::max_nodes`].
pub const SMALL_BLOCK_MAX_NODES: usize = 8;

impl Default for SmallBlocks {
    /// On, for blocks of up to [`SMALL_BLOCK_MAX_NODES`] graph nodes.
    fn default() -> Self {
        Self {
            enabled: true,
            max_nodes: SMALL_BLOCK_MAX_NODES,
        }
    }
}

impl SmallBlocks {
    /// Whether a block with `node_count` graph nodes takes the fast path.
    fn is_small(self, node_count: usize) -> bool {
        self.enabled && node_count <= self.max_nodes
    }
}

/// The read-only views the stages after [`prepare`] share.
struct BuildViews<'b, IndexType: GraphIndexInteger, NodeData, EdgeData> {
    graph: &'b Graph<IndexType, NodeData, EdgeData>,
    image: &'b Image,
    /// The node topology, locating record-block nodes through the prepare stage's build
    /// locations (the lift lays out the rows that replace them).
    topology: NodeTopology<'b, IndexType>,
    /// The image graph (names and lengths; no adjacency yet).
    graph_view: GraphView<'b, IndexType>,
}

/// What the build keeps from the decomposition.
struct Prepared<IndexType: GraphIndexInteger> {
    topology: TopologySections<IndexType>,
    /// Where the record-block nodes sit in their block's SPQR tree, until the lift's row layout.
    build_locs: Vec<BuildLoc>,
    /// Every block, in engine block order.
    blocks: Vec<PreparedBlock<IndexType>>,
}

/// One block's inputs to the per-block stage.
struct PreparedBlock<IndexType: GraphIndexInteger> {
    block_index: BlockIndex<IndexType>,
    /// `None` for a small block (it gets no record).
    tree: Option<BlockSpqrTree<IndexType>>,
}

/// The prepare stage: the engine's block numbering (chain order; only here are block ids
/// translated back to the decomposition's), the SPQR tree of every block that is not small, and
/// the node topology. The trees come first: the topology locates nodes only in the blocks with a
/// tree, since only a record block is entered through its SPQR nodes.
fn prepare<IndexType, NodeData, EdgeData>(
    spqr: &SPQRDecomposition<'_, Graph<IndexType, NodeData, EdgeData>>,
    image: &Image,
    small_blocks: SmallBlocks,
) -> Prepared<IndexType>
where
    IndexType: GraphIndexInteger + Send + Sync,
    NodeData: GfaNodeData + Send + Sync,
    EdgeData: GfaEdgeData + Send + Sync,
{
    let numbering = chain_block_numbering(spqr);
    let trees = block_trees(spqr, &numbering, small_blocks);
    let (topology, build_locs) = TopologySections::write(image, spqr, &numbering, |b| {
        trees[b].as_ref().map(BlockSpqrTree::tree)
    });
    let blocks = trees
        .into_iter()
        .enumerate()
        .map(|(b, tree)| PreparedBlock {
            block_index: BlockIndex::from_usize(b),
            tree,
        })
        .collect();
    Prepared {
        topology,
        build_locs,
        blocks,
    }
}

/// The SPQR tree of every block that is not small (`None` for a small one), in engine block
/// order. The blocks are independent, so their trees are built across the pool.
fn block_trees<IndexType, NodeData, EdgeData>(
    spqr: &SPQRDecomposition<'_, Graph<IndexType, NodeData, EdgeData>>,
    numbering: &BlockNumbering,
    small_blocks: SmallBlocks,
) -> Vec<Option<BlockSpqrTree<IndexType>>>
where
    IndexType: GraphIndexInteger + Send + Sync,
    NodeData: GfaNodeData + Send + Sync,
    EdgeData: GfaEdgeData + Send + Sync,
{
    let blocks: Vec<_> = spqr.iter_blocks().map(|(_, block)| block).collect();
    let needs_tree: Vec<bool> = (0..numbering.len())
        .map(|b| {
            let block = blocks[numbering.old_id::<IndexType>(b).into_usize()];
            !small_blocks.is_small(block.node_count())
        })
        .collect();
    (0..needs_tree.len())
        .into_par_iter()
        .with_min_len(256)
        .map(|b| {
            needs_tree[b].then(|| BlockSpqrTree::new_without_lookup(spqr, numbering.old_id(b)))
        })
        .collect()
}

/// The lift stage: the block-cut lift over the finished blocks, from the forest, the raw
/// cut-to-cut matrices and the bottom-up flip closures the flip baking left (`baked`).
fn build_lift<'b, IndexType, NodeData, EdgeData>(
    views: &BuildViews<'b, IndexType, NodeData, EdgeData>,
    baked: BakedForest<IndexType>,
    options: &LiftOptions,
) -> (BlockCutLift<'b, IndexType>, LiftSections<IndexType>)
where
    IndexType: GraphIndexInteger + Send + Sync + 'static,
    NodeData: GfaNodeData + Send + Sync,
    EdgeData: GfaEdgeData + Send + Sync,
{
    BlockCutLift::new(views.image, views.graph, &views.topology, baked, options)
}

/// The adjacency stage, once the lift has marked the simple blocks: their non-cut nodes keep no
/// arcs (the lift's rows answer their in-block distances, their closed-form masks name their
/// neighbours); every other node keeps its arcs.
fn write_adjacency<IndexType, NodeData, EdgeData>(
    image: &Image,
    graph: &Graph<IndexType, NodeData, EdgeData>,
    graph_secs: &mut GraphSections<IndexType>,
    topology: NodeTopology<'_, IndexType>,
    lift: &BlockCutLift<'_, IndexType>,
) where
    IndexType: GraphIndexInteger + 'static,
    NodeData: GfaNodeData,
{
    graph_secs.write_adjacency(image, graph, |v| {
        let node = NodeIndex::from_usize(v);
        topology.is_cut_node(node)
            || !topology
                .node_block_indices(node)
                .next()
                .is_some_and(|b| lift.block_is_simple(b.into_usize()))
    })
}

/// The largest skeleton (inner or LCA role) a query can Dijkstra. Stored in the header, so a
/// loaded index sizes its query scratch without touching every record.
fn max_searched_skeleton<IndexType: GraphIndexInteger>(
    image: &Image,
    block_dir: &[BlockDirRec],
) -> usize {
    let mut max = 0usize;
    for rec in block_dir {
        let (b, off) = (rec.block as usize, rec.off as usize);
        let view = BlockView::<IndexType>::read(image, BlockIndex::from_usize(b), off);
        for local in 0..view.node_count() {
            max = max
                .max(view.inner(local).node_count())
                .max(view.lca(local).node_count());
        }
    }
    max
}

/// The image header: word size, the indexed graph's identity and the build options it records.
fn image_header<IndexType: GraphIndexInteger, NodeData, EdgeData>(
    graph: &Graph<IndexType, NodeData, EdgeData>,
    fingerprint: u64,
    options: &BuildOptions<'_>,
    max_searched_nodes: usize,
) -> Header {
    let small_blocks = options.small_blocks;
    let mut header = Image::header_template(std::mem::size_of::<IndexType>());
    header.graph_node_count = graph.node_count() as u64;
    header.graph_edge_count = graph.edge_count() as u64;
    header.graph_fingerprint = fingerprint;
    let knobs = &mut header.build_knobs;
    knobs[knob::LABEL_THRESHOLD] = options.tiers.label_min as u64;
    knobs[knob::SMALL_BLOCK_MAX] = small_blocks.max_nodes as u64;
    knobs[knob::MAX_SEARCHED_NODES] = max_searched_nodes as u64;
    knobs[knob::SNODE_SEGTREE_MIN] = options.tiers.segtree_min as u64;
    knobs[knob::NODE_ROW_MAX_CUTS] = options.lift.node_row_max_cuts as u64;
    knobs[knob::PROJ_TABLE_MIN] = options.tiers.table_min as u64;
    knobs[knob::FLAGS] = if small_blocks.enabled {
        knob::FLAG_SMALL_BLOCK_FAST
    } else {
        0
    };
    header
}

/// A shrunk index answers on the ORIGINAL graph: the header names that graph and sets the shrunk
/// flag.
fn mark_shrunk(header: &mut Header, sink: &dyn ShrinkSink) {
    (
        header.graph_node_count,
        header.graph_edge_count,
        header.graph_fingerprint,
    ) = sink.original_header();
    header.build_knobs[knob::FLAGS] |= knob::FLAG_SHRUNK;
}

/// Drain the allocator's deferred free-chunk queue during the build instead of the first query.
///
/// Releasing the decomposition and the build's temporaries frees millions of small chunks. glibc
/// parks them in its unsorted bin and sorts at most 10 000 of them per `malloc` that reaches that
/// bin, so the first queries would pay milliseconds of deferred sorting. Large requests always walk
/// the unsorted bin, so a burst of them here settles it; once it is empty each round is a cheap
/// allocate/free pair. Harmless with any other allocator.
fn settle_allocator() {
    for _ in 0..4096 {
        std::hint::black_box(Vec::<u8>::with_capacity(64 << 10));
    }
}
