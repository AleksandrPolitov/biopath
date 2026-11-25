//! The whole-graph shortest-path engine: [`DistanceIndex`] answers the exact distance
//! between two positions (a directed node and an offset into it) of a bidirected GFA graph.
//!
//! # What the engine stores
//!
//! Everything lives in one index [`Image`], built in RAM by [`DistanceIndex::build`] or
//! loaded from a `.bpi` file by [`DistanceIndex::open`]. The engine adds only the section
//! handles and query scratch, so opening an index maps the file and parses nothing. The image
//! holds:
//!
//! - the graph: node lengths and names, and the arcs of every node except the non-cut nodes of
//!   simple blocks (whose rows and closed-form masks replace them);
//! - the node topology: every node's blocks and cut flag, and the SPQR-tree location of a node of a
//!   record block;
//! - one record per biconnected block above the small-block bound, found through the sparse block
//!   directory: the block's SPQR tree with its query skeletons, `D_out` overlays, pole-lift
//!   transitions, hub labels, S-node segment trees and pole-projection tables. A small block has
//!   no record and is searched directly;
//! - the block-cut lift ([`BlockCutLift`]): per cut node its flip closure, per block its cut-to-cut
//!   distances, the lift tables along the block-cut tree, and every node's rows of distances to
//!   its block's cut nodes;
//! - for a shrunk graph G′, the original graph's names and the original -> shrunk alias.
//!
//! The build (`engine_build`) fills the image stage by stage.
//!
//! # Queries
//!
//! [`DistanceIndex::distance`] resolves a query in this order:
//!
//! 1. **Alias** (shrunk index only): the original positions are mapped onto G′ and the answer is
//!    assembled from G′ sub-queries (see [`crate::shrink`]).
//! 2. **Same node**: a target at or after the source on the same directed node is reached
//!    directly; any other walk between two positions of one node leaves the node and returns.
//! 3. **Shared block**: for every block holding both endpoints, the in-block distance from the
//!    block's record engine ([`BlockQueryEngine`]), a simple block's rows, or a small block's
//!    block-restricted Dijkstra.
//! 4. **Block-cut lift**: the route through cut nodes across blocks ([`BlockCutLift::cross`]).
//!
//! The answer is the minimum over the routes; there is no whole-graph search.

use std::cell::RefCell;

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::{
    decomposition::{SPQRDecomposition, indices::BlockIndex},
    graph::StaticGraph,
};

use crate::{
    block_cut_lift::{BlockCutLift, LiftSections},
    block_dijkstra::BlockDijkstra,
    block_engine::BlockQueryEngine,
    block_record::{BlockDirRec, BlockView, block_dir_offset},
    engine_build::build_engine,
    graph_view::{GraphSections, GraphView, QueryGraph},
    image::{Image, Sec},
    location::GfaLocation,
    min_plus::relax,
    path::GfaPathLength,
    sections::{self, knob},
    shrink::{
        OriginalPos, ShrinkSections, ShrinkSink, ShrinkView, ShrunkOracle, ShrunkPos, SuperLocation,
    },
    skeleton::DijkstraWorkspace,
    topology::{NodeTopology, RowRef, TopologySections},
};

#[cfg(test)]
mod lift_fuzz_tests;
#[cfg(test)]
mod memo_tests;
#[cfg(test)]
mod open_tests;
#[cfg(test)]
mod tests;

/// The default hub-label threshold [`Tiers::label_min`].
pub use crate::block_engine::LABEL_MIN_DIRECTED_NODES;
pub use crate::{
    block_cut_lift::{LiftBackend, LiftOptions},
    block_engine::Tiers,
    engine_build::{SMALL_BLOCK_MAX_NODES, SmallBlocks},
};

/// The shortest-path engine over one index [`Image`] (see the module docs): the image holds every
/// retained structure; this struct adds the section handles and the query scratch.
///
/// A `DistanceIndex` is `Send` but not `Sync`: its query scratch lives in `RefCell`s, reused by
/// every query so that a query never allocates. To query from several threads, open one instance
/// per thread; instances memory-mapping the same `.bpi` share its pages in the page cache, so each
/// costs only its scratch.
pub struct DistanceIndex<IndexType: GraphIndexInteger> {
    image: Image,
    graph: GraphSections<IndexType>,
    topology: TopologySections<IndexType>,
    /// The sparse record directory: `(block, byte offset)` of every block that has a record, sorted
    /// by block (a block absent from it is on the small-block fast path).
    block_dir: Sec<BlockDirRec>,
    lift: LiftSections<IndexType>,
    /// Scratch of the record engines' skeleton searches in the shared-block step, reused across
    /// queries.
    workspace: RefCell<DijkstraWorkspace<IndexType>>,
    /// Scratch for the intra-block searches of [`Self::block_start_to_start`], sized at build time for
    /// the largest skeleton a query can Dijkstra, so queries never allocate it.
    intra_workspace: RefCell<DijkstraWorkspace<IndexType>>,
    /// Reusable block-restricted single-source Dijkstra (small-block fallback).
    block_dijkstra: RefCell<BlockDijkstra<IndexType>>,
    /// The source's blocks in [`Self::start_to_start`], reused across queries.
    source_blocks: RefCell<Vec<BlockIndex<IndexType>>>,
    /// A shrunk index: the original graph's names and the original -> shrunk alias. Queries then
    /// name ORIGINAL nodes (see [`crate::shrink`]).
    shrink: Option<ShrinkSections>,
}

/// The options of [`DistanceIndex::build`]: every parameter that shapes the index. The
/// engine reads none of them from the environment; [`Default`] gives the production values.
#[derive(Clone, Copy, Default)]
pub struct BuildOptions<'a> {
    /// Which oracle answers each SPQR node's searches: the hub-label, S-node segment-tree and
    /// pole-projection-table thresholds, in directed skeleton states (see [`Tiers`]).
    pub tiers: Tiers,
    /// The small-block fast path: which blocks get no record and are searched directly.
    pub small_blocks: SmallBlocks,
    /// The block-cut lift: its path-product backend and which blocks get node cut rows.
    pub lift: LiftOptions,
    /// Set when the graph is a shrunk graph G′: writes the original graph's names and the
    /// original -> shrunk alias, so the index answers queries on ORIGINAL positions (see
    /// [`crate::shrink`]).
    pub shrink: Option<&'a dyn ShrinkSink>,
}

impl BuildOptions<'_> {
    /// The default options with an explicit hub-label threshold ([`Tiers::label_min`]: a skeleton
    /// with at least this many directed nodes gets a hub-label oracle; `usize::MAX` disables
    /// labels).
    #[cfg(test)]
    pub fn with_label_threshold(label_threshold: usize) -> Self {
        Self {
            tiers: Tiers {
                label_min: label_threshold,
                ..Tiers::default()
            },
            ..Self::default()
        }
    }
}

/// The SPQR decomposition an engine is built from. An owned one is dropped as soon as the build
/// has extracted the node topology and the block SPQR trees, before the per-block DP where the
/// build's memory peaks; a borrowed one stays with the caller. Both build the same engine.
pub enum Decomposition<'s, 'g, G: StaticGraph> {
    Borrowed(&'s SPQRDecomposition<'g, G>),
    Owned(SPQRDecomposition<'g, G>),
}

impl<'g, G: StaticGraph> Decomposition<'_, 'g, G> {
    /// The decomposition, whether borrowed or owned.
    pub fn get(&self) -> &SPQRDecomposition<'g, G> {
        match self {
            Decomposition::Borrowed(spqr) => spqr,
            Decomposition::Owned(spqr) => spqr,
        }
    }
}

impl<'s, 'g, G: StaticGraph> From<&'s SPQRDecomposition<'g, G>> for Decomposition<'s, 'g, G> {
    fn from(spqr: &'s SPQRDecomposition<'g, G>) -> Self {
        Decomposition::Borrowed(spqr)
    }
}

impl<'g, G: StaticGraph> From<SPQRDecomposition<'g, G>> for Decomposition<'_, 'g, G> {
    fn from(spqr: SPQRDecomposition<'g, G>) -> Self {
        Decomposition::Owned(spqr)
    }
}

/// The section handles of an engine image: what a build writes and a load resolves.
pub(super) struct EngineSections<IndexType: GraphIndexInteger> {
    pub(super) graph: GraphSections<IndexType>,
    pub(super) topology: TopologySections<IndexType>,
    pub(super) block_dir: Sec<BlockDirRec>,
    pub(super) lift: LiftSections<IndexType>,
    pub(super) shrink: Option<ShrinkSections>,
}

impl<IndexType: GraphIndexInteger + 'static> DistanceIndex<IndexType> {
    /// Build the engine for `graph` from its SPQR decomposition, passed as `&spqr` (the caller
    /// keeps it) or `spqr` (released during the build, see [`Decomposition`]).
    ///
    /// The distance arithmetic is not checked for overflow: `graph` must fit `IndexType`, which
    /// [`crate::graph_limits::check_build`] verifies.
    pub fn build<'s, 'g, NodeData, EdgeData>(
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        spqr: impl Into<Decomposition<'s, 'g, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>>,
        options: &BuildOptions<'_>,
    ) -> Self
    where
        'g: 's,
        IndexType: Send + Sync,
        NodeData: GfaNodeData + Send + Sync + 'g,
        EdgeData: GfaEdgeData + Send + Sync + 'g,
    {
        build_engine(graph, spqr.into(), options)
    }

    /// The engine over a finished image and its section handles. `max_searched_nodes` sizes the
    /// intra-block query scratch (the largest skeleton a query can Dijkstra), so queries never
    /// allocate it.
    pub(super) fn assemble(
        image: Image,
        sections: EngineSections<IndexType>,
        max_searched_nodes: usize,
    ) -> Self {
        let EngineSections {
            graph,
            topology,
            block_dir,
            lift,
            shrink,
        } = sections;
        let mut intra_workspace = DijkstraWorkspace::new();
        intra_workspace.reserve(max_searched_nodes);
        let node_count = graph.view(&image).node_count();
        Self {
            image,
            graph,
            topology,
            block_dir,
            lift,
            workspace: RefCell::new(DijkstraWorkspace::new()),
            intra_workspace: RefCell::new(intra_workspace),
            block_dijkstra: RefCell::new(BlockDijkstra::new(node_count * 2)),
            source_blocks: RefCell::new(Vec::new()),
            shrink,
        }
    }

    /// Open a `.bpi` index file written by [`Self::write_to`]: memory-map it and prefault all of it
    /// ([`Image::open`]).
    ///
    /// A `.bpi` is trusted input, like vg's `.dist`: loading refuses a truncated or foreign file,
    /// another format version, byte order or word size, and sections that do not fit the file, but
    /// queries follow the offsets stored inside the records unchecked, so a crafted file can make
    /// them read out of bounds. Load only indexes you built or trust.
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let image = Image::open(path)?;
        Self::from_image(image)
    }

    /// Wrap a loaded (or finished) image.
    pub fn from_image(image: Image) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};
        let header = image.header();
        if header.word_size as usize != std::mem::size_of::<IndexType>() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "index image has word size {} but this engine uses {}",
                    header.word_size * 8,
                    std::mem::size_of::<IndexType>() * 8
                ),
            ));
        }
        let handles = EngineSections {
            graph: GraphSections::from_image(&image)?,
            topology: TopologySections::from_image(&image)?,
            block_dir: image.require(sections::BLOCK_DIR_SPARSE, "blocks.dir_sparse")?,
            lift: LiftSections::from_image(&image)?,
            shrink: if header.build_knobs[knob::FLAGS] & knob::FLAG_SHRUNK != 0 {
                Some(ShrinkSections::from_image(&image)?)
            } else {
                None
            },
        };
        // Sized from the header rather than by scanning every record at load. A searched skeleton
        // holds directed graph nodes (plus at most four virtual pole states), so a larger value is
        // a corrupt header.
        let max_searched_nodes = header.build_knobs[knob::MAX_SEARCHED_NODES];
        let node_count = handles.graph.view(&image).node_count() as u64;
        if max_searched_nodes > 2 * node_count + 4 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "index image header says a query searches {max_searched_nodes} skeleton \
                     states, more than the graph's {node_count} nodes allow: corrupt header"
                ),
            ));
        }
        Ok(Self::assemble(image, handles, max_searched_nodes as usize))
    }

    /// Write the index image to `path`.
    pub fn write_to(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.image.write_to(path)
    }

    /// The underlying image.
    pub fn image(&self) -> &Image {
        &self.image
    }

    /// The image graph (lengths, adjacency, names).
    pub fn graph_view(&self) -> GraphView<'_, IndexType> {
        self.graph.view(&self.image)
    }

    /// The node with the given GFA name, if any (an ORIGINAL node of a shrunk index).
    pub fn node_by_name(&self, name: &str) -> Option<NodeIndex<IndexType>> {
        match &self.shrink {
            Some(shrink) => shrink
                .view(&self.image)
                .node_by_name(name)
                .map(NodeIndex::from_usize),
            None => self.graph_view().node_by_name(name),
        }
    }

    /// The GFA name of a node (an ORIGINAL node of a shrunk index).
    pub fn node_name(&self, node: NodeIndex<IndexType>) -> std::borrow::Cow<'_, str> {
        match &self.shrink {
            Some(shrink) => shrink.view(&self.image).node_name(node.into_usize()),
            None => self.graph_view().node_name(node),
        }
    }

    /// The number of graph nodes.
    pub fn node_count(&self) -> usize {
        self.graph_view().node_count()
    }

    /// The shrink alias of a shrunk index (`None` otherwise): where an ORIGINAL position lies in
    /// G′ or in a pruned chain (see [`crate::shrink`]).
    pub fn shrink_view(&self) -> Option<ShrinkView<'_>> {
        self.shrink.map(|shrink| shrink.view(&self.image))
    }

    /// Shrunk index only: where ORIGINAL position `p` lies (its super-node and the offset along it).
    pub fn super_location(&self, p: OriginalPos) -> Option<SuperLocation> {
        let graph = self.graph_view();
        self.shrink_view().map(|view| {
            view.locate(p, |g| {
                graph.node_len(NodeIndex::from_usize(g)).into_usize() as u64
            })
        })
    }

    /// The start-to-start distance between two different directed nodes of the engine's own graph
    /// (G′ of a shrunk index), offset 0 to offset 0, or `None` when no walk connects them: the term
    /// [`Self::distance`] answers two positions on different nodes from, `d - source offset +
    /// target offset`. For a caller that resolves its positions onto G′ once and then answers many
    /// pairs from few node pairs.
    pub fn g_start_to_start(
        &self,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        debug_assert_ne!(a, b, "the same directed node: the walk leaves and returns");
        let topology = self.topology_view();
        if topology.node_component_index(a.into_bidirected())
            != topology.node_component_index(b.into_bidirected())
        {
            return None;
        }
        self.start_to_start_any(a, b)
    }

    /// The shortest walk from the start of directed node `a` of the engine's own graph that leaves
    /// through its end and returns to its start, or `None`: two positions `s` and `t` on `a` with `t`
    /// behind `s` are `around - s + t` apart (the walk [`Self::distance`] takes for them).
    pub fn g_around(&self, a: DirectedNodeIndex<IndexType>) -> Option<GfaPathLength<IndexType>> {
        let start = GfaLocation::new(a, crate::location::GfaNodeOffset::from_usize(0));
        self.same_node_around(start, start)
    }

    fn topology_view(&self) -> NodeTopology<'_, IndexType> {
        self.topology.view(&self.image)
    }

    fn lift_view(&self) -> BlockCutLift<'_, IndexType> {
        self.lift.view(&self.image)
    }

    /// The record offset of a block, or `None` for a small block.
    #[inline]
    fn block_offset(&self, block: BlockIndex<IndexType>) -> Option<usize> {
        block_dir_offset(self.image.get(self.block_dir), block.into_usize())
    }

    #[inline]
    fn has_block(&self, block: BlockIndex<IndexType>) -> bool {
        self.block_offset(block).is_some()
    }

    /// Run a closure with the block's query engine over its image record.
    fn with_block_engine<R>(
        &self,
        block: BlockIndex<IndexType>,
        f: impl FnOnce(&BlockQueryEngine<'_, IndexType>) -> R,
    ) -> R {
        let offset = self
            .block_offset(block)
            .expect("no record stored for this block index");
        let view = BlockView::read(&self.image, block, offset);
        let engine = BlockQueryEngine::new(self.topology_view(), &view, self.graph_view());
        f(&engine)
    }

    /// The length in bp of `node` (an ORIGINAL node of a shrunk index), or `None` when the index has
    /// no such node. For checking positions where they enter (see [`Self::location_is_valid`]), not
    /// for the query path.
    pub fn node_len(&self, node: NodeIndex<IndexType>) -> Option<u64> {
        let v = node.into_usize();
        match self.shrink_view() {
            Some(view) => (v < view.node_count()).then(|| view.node_len(v)),
            None => {
                let graph = self.graph_view();
                (v < graph.node_count()).then(|| graph.node_len(node).into_usize() as u64)
            }
        }
    }

    /// Whether `location` is a position [`Self::distance`] accepts: a node of the index (an
    /// ORIGINAL node of a shrunk index) and an offset `0..=len` (`len` is the cut point after the
    /// last base, so a zero-length node has offset 0 only).
    pub fn location_is_valid(&self, location: GfaLocation<IndexType>) -> bool {
        self.node_len(location.node().into_bidirected())
            .is_some_and(|len| location.offset().into_usize() as u64 <= len)
    }

    /// The shortest-path distance from `source` to `target`, or `None` when no walk connects them
    /// (see the module docs for how a query resolves).
    ///
    /// Both positions must be valid ([`Self::location_is_valid`]): a node of the index and an
    /// offset `0..=len`. A query does no work it does not need, so this is checked only in debug
    /// builds; in a release build an invalid position gives a wrong answer or a panic. Check
    /// positions once where they enter, as `biopath query` does when it reads its query file.
    pub fn distance(
        &self,
        source: GfaLocation<IndexType>,
        target: GfaLocation<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        debug_assert!(
            self.location_is_valid(source),
            "invalid source position {source}"
        );
        debug_assert!(
            self.location_is_valid(target),
            "invalid target position {target}"
        );
        match &self.shrink {
            Some(shrink) => self.distance_shrunk(*shrink, source, target),
            None => self.distance_core(source, target),
        }
    }

    /// [`Self::distance`] on the engine's own graph.
    fn distance_core(
        &self,
        source: GfaLocation<IndexType>,
        target: GfaLocation<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let source_offset = source.offset().into_length();
        let target_offset = target.offset().into_length();
        let mut best: Option<GfaPathLength<IndexType>> = None;

        if source.node() == target.node() && target.offset() >= source.offset() {
            // Exact: any other walk leaves the node and comes back, so it is at least as long.
            return Some(target_offset - source_offset);
        }
        let topology = self.topology_view();
        if topology.node_component_index(source.node().into_bidirected())
            != topology.node_component_index(target.node().into_bidirected())
        {
            return best;
        }

        if source.node() != target.node() {
            if let Some(distance) = self.start_to_start_any(source.node(), target.node()) {
                relax(&mut best, distance - source_offset + target_offset);
            }
        } else {
            best = self.same_node_around(source, target);
        }
        best
    }

    /// [`Self::start_to_start`], plus the two orientations of a node that lies in no block
    /// (isolated but for self-loops, as a node of a shrunk graph can be): the lift has nothing to
    /// answer from there, and the only walks are the node's own self-loop arcs.
    fn start_to_start_any(
        &self,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let node = a.into_bidirected();
        if node == b.into_bidirected()
            && self
                .topology_view()
                .node_block_indices(node)
                .next()
                .is_none()
        {
            if a == b {
                return Some(GfaPathLength::from_usize(0));
            }
            // Two states: `b` is reached only by an arc into it, from `a` or from `b` itself.
            return self
                .out_neighbors(a)
                .any(|to| to == b)
                .then(|| self.graph_view().node_len(node));
        }
        self.start_to_start(a, b)
    }

    /// The shortest walk from `source` to `target` on the SAME directed node that leaves the node
    /// first (out of its end, back in at its start), whatever the two offsets.
    fn same_node_around(
        &self,
        source: GfaLocation<IndexType>,
        target: GfaLocation<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        debug_assert_eq!(source.node(), target.node());
        let source_offset = source.offset().into_length();
        let target_offset = target.offset().into_length();
        let graph = self.graph_view();
        let source_length = graph.node_len(source.node().into_bidirected());
        let mut best = None;
        for to in self.out_neighbors(source.node()) {
            // A self-loop arc returns straight to the node's start (a node of a shrunk graph can carry
            // one and lie in no block, where the lift has nothing to answer from).
            let distance = if to == source.node() {
                Some(GfaPathLength::from_usize(0))
            } else {
                self.start_to_start_any(to, source.node())
            };
            if let Some(distance) = distance {
                relax(
                    &mut best,
                    source_length + distance - source_offset + target_offset,
                );
            }
        }
        best
    }

    /// A query on ORIGINAL positions of a shrunk index: resolved through the alias, with the G′
    /// sub-queries answered by [`Self::distance_core`] / [`Self::same_node_around`].
    fn distance_shrunk(
        &self,
        shrink: ShrinkSections,
        source: GfaLocation<IndexType>,
        target: GfaLocation<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        struct Oracle<'e, IndexType: GraphIndexInteger + 'static> {
            engine: &'e DistanceIndex<IndexType>,
            graph: GraphView<'e, IndexType>,
        }
        impl<IndexType: GraphIndexInteger + 'static> ShrunkOracle for Oracle<'_, IndexType> {
            fn node_len(&mut self, node: usize) -> u64 {
                self.graph
                    .node_len(NodeIndex::from_usize(node))
                    .into_usize() as u64
            }
            fn distance(
                &mut self,
                source: ShrunkPos,
                target: ShrunkPos,
                around: bool,
            ) -> Option<u64> {
                let location = |p: ShrunkPos| {
                    GfaLocation::new(
                        NodeIndex::from_usize(p.node).into_directed(p.forward),
                        crate::location::GfaNodeOffset::from_usize(p.offset as usize),
                    )
                };
                let (source, target) = (location(source), location(target));
                let distance = if around {
                    self.engine.same_node_around(source, target)
                } else {
                    self.engine.distance_core(source, target)
                };
                distance.map(|d| d.into_usize() as u64)
            }
        }
        let position = |l: GfaLocation<IndexType>| OriginalPos {
            node: l.node().into_bidirected().into_usize(),
            forward: l.node().is_forward(),
            offset: l.offset().into_usize() as u64,
        };
        let mut oracle = Oracle {
            engine: self,
            graph: self.graph_view(),
        };
        shrink
            .view(&self.image)
            .distance(position(source), position(target), &mut oracle)
            .map(|d| GfaPathLength::from_usize(d as usize))
    }

    /// The start-to-start distance between two directed nodes: the minimum of the in-block
    /// distance in every block holding both and the cross-block route of the lift.
    fn start_to_start(
        &self,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let mut best: Option<GfaPathLength<IndexType>> = None;
        let topology = self.topology_view();
        let lift = self.lift_view();

        // Intra-block direct: any block containing both endpoints.
        {
            let mut a_blocks = self.source_blocks.borrow_mut();
            a_blocks.clear();
            a_blocks.extend(topology.node_block_indices(a.into_bidirected()));
            let mut workspace = self.workspace.borrow_mut();
            for block in topology.node_block_indices(b.into_bidirected()) {
                if a_blocks.contains(&block) {
                    let distance = if self.has_block(block) {
                        self.with_block_engine(block, |engine| {
                            engine.start_to_start_with(a, b, &mut workspace)
                        })
                    } else {
                        self.block_start_to_start(a, b, block)
                    };
                    if let Some(distance) = distance {
                        relax(&mut best, distance);
                    }
                }
            }
        }

        // The exact within-block distance `d_block(from -> to)`: the SPQR block engine's
        // `start_to_start` (which projects an off-skeleton endpoint onto its child-subtree poles), or
        // a small block's on-demand search. Used by the cross-block route below.
        let intra = |from, to, block_index: BlockIndex<IndexType>| {
            self.block_start_to_start(from, to, block_index)
        };
        // Cross-block route via the block-cut lifting engine. All intra-block distances it needs
        // (seed/read cut-node vectors, same-block flip) come from the node rows or `intra` above.
        if let Some(distance) = lift.cross(&topology, &intra, a, b) {
            relax(&mut best, distance);
        }
        best
    }

    /// The distances from `source` to every directed node of `block`, via one block-restricted
    /// Dijkstra; `read` extracts whatever the caller needs from the completed search.
    fn block_source_distances<R>(
        &self,
        source: DirectedNodeIndex<IndexType>,
        block: BlockIndex<IndexType>,
        read: impl FnOnce(&BlockDijkstra<IndexType>) -> R,
    ) -> R {
        let topology = self.topology_view();
        let graph = self.graph_view();
        let mut block_dijkstra = self.block_dijkstra.borrow_mut();
        block_dijkstra.run(&graph, source, |node| {
            topology.node_block_indices(node).any(|b| b == block)
        });
        read(&block_dijkstra)
    }

    /// The exact in-block `start_to_start(from -> to)`: a record block's engine, a simple block's
    /// rows, or a small block's (no record) block-restricted Dijkstra.
    fn block_start_to_start(
        &self,
        from: DirectedNodeIndex<IndexType>,
        to: DirectedNodeIndex<IndexType>,
        block: BlockIndex<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        if self.has_block(block) {
            // One build-sized scratch for every call (allocating one per call is slow). A nested
            // call (the workspace already borrowed) falls back to a fresh one.
            let mut shared = self.intra_workspace.try_borrow_mut();
            let mut fresh = None;
            let workspace = match shared {
                Ok(ref mut workspace) => &mut **workspace,
                Err(_) => fresh.insert(DijkstraWorkspace::new()),
            };
            self.with_block_engine(block, |engine| {
                engine.start_to_start_with(from, to, workspace)
            })
        } else if from == to {
            Some(GfaPathLength::from_usize(0))
        } else if self.lift_view().block_is_simple(block.into_usize()) {
            // A simple block keeps no arcs: its rows answer (see `simple_block_distance`).
            self.lift_view().simple_block_distance(
                &self.topology_view(),
                from,
                to,
                block.into_usize(),
            )
        } else {
            self.block_source_distances(from, block, |bd| bd.get(to))
        }
    }

    /// The directed nodes one arc out of `node` reaches: its stored arcs, or (a simple-block node,
    /// which keeps none) its closed-form mask's. Allocates nothing.
    fn out_neighbors(
        &self,
        node: DirectedNodeIndex<IndexType>,
    ) -> impl Iterator<Item = DirectedNodeIndex<IndexType>> + '_ {
        let graph = self.graph_view();
        let v = node.into_bidirected();
        let (stored, closed_form) = if graph.has_adjacency(v.into_usize()) {
            (Some(graph.image_out_neighbors(node)), None)
        } else {
            let topology = self.topology_view();
            let block = topology
                .node_block_indices(v)
                .next()
                .expect("a node without arcs lies in a simple block");
            let Some(RowRef::Closed { mask }) = topology.row_ref(v) else {
                panic!("a node without arcs has closed-form rows");
            };
            let neighbors = self.lift_view().closed_form_neighbors(
                block.into_usize(),
                mask,
                usize::from(!node.is_forward()),
            );
            (None, Some(neighbors))
        };
        stored
            .into_iter()
            .flatten()
            .chain(closed_form.into_iter().flatten())
    }
}

// A `DistanceIndex` moves between threads (see its docs).
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<DistanceIndex<u32>>();
};
