//! Persistence: [`ShrinkAttach`] writes the alias and the original names into the image (as the
//! [`ShrinkSink`] the engine build calls), and [`ShrinkSections`] finds them again.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::GfaNodeData,
};

use super::{
    ShrinkPlan,
    layout::RunLayout,
    records::{AliasBlock, FarPorts, LenBlock, LongLen, P1Rec, PrunedChain, RunRec},
    view::ShrinkView,
};
use crate::{
    graph_view::{NameIds, NameSections},
    image::{Image, Sec},
    sections,
};

const SHRINK_NAME_IDS: NameIds = NameIds {
    runs: sections::SHRINK_NAME_RUNS,
    run_order: sections::SHRINK_NAME_RUN_ORDER,
    bytes: sections::SHRINK_NAME_BYTES,
    offsets: sections::SHRINK_NAME_OFFSETS,
    hash: sections::SHRINK_NAME_HASH,
};

/// What the engine build needs from a shrink plan to write the shrink sections (see
/// [`ShrinkAttach`]): object safe, so the engine stays generic over the input graph only.
pub trait ShrinkSink {
    fn write(&self, image: &Image) -> ShrinkSections;
    /// `(node count, edge count, fingerprint)` of the ORIGINAL graph, for the image header.
    fn original_header(&self) -> (u64, u64, u64);
}

/// A shrink plan together with the original graph (whose names the index keeps).
pub struct ShrinkAttach<'a, IndexType: GraphIndexInteger, NodeData, EdgeData> {
    pub plan: &'a ShrinkPlan<IndexType>,
    pub original: &'a BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    /// The original nodes' names as integer runs ([`crate::graph_view::integer_name_runs`]) when
    /// the caller already has them (GBZ input: the node ids), so the names are not read back from
    /// `original` (226M heap strings on a whole-genome graph). `None`: scan `original`'s names.
    pub original_names: Option<&'a [crate::graph_view::NameRun]>,
}

/// Panics unless the caller's `runs` name the nodes of `original` as `original` does, checked on up to
/// ~1000 evenly spaced nodes plus the last (a full check would be the very scan the runs avoid).
fn check_name_runs<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    original: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    runs: &[crate::graph_view::NameRun],
) {
    let n = original.node_count();
    assert!(
        runs.first().is_some_and(|r| r.node == 0) || n == 0,
        "the caller's name runs must start at node 0"
    );
    let step = (n / 1000).max(1);
    for v in (0..n).step_by(step).chain(n.checked_sub(1)) {
        let run = runs[runs.partition_point(|r| r.node <= v as u64) - 1];
        let expected = run.name + (v as u64 - run.node);
        let name = original.node_data(NodeIndex::from_usize(v)).name();
        assert!(
            name.parse::<u64>().ok() == Some(expected) && !name.starts_with('0')
                || name == "0" && expected == 0,
            "the caller's name runs say node {v} is named {expected}, the graph says {name:?}"
        );
    }
}

impl<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData> ShrinkSink
    for ShrinkAttach<'_, IndexType, NodeData, EdgeData>
{
    fn write(&self, image: &Image) -> ShrinkSections {
        let alias = AliasSections::write(image, &self.plan.runs);
        let node_count = self.original.node_count();
        let names = self.write_names(image);
        ShrinkSections {
            alias,
            names,
            node_count,
        }
    }

    fn original_header(&self) -> (u64, u64, u64) {
        (
            self.original.node_count() as u64,
            self.original.edge_count() as u64,
            self.plan.original_fingerprint,
        )
    }
}

impl<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>
    ShrinkAttach<'_, IndexType, NodeData, EdgeData>
{
    /// The original nodes' names: from the caller's integer runs when given, else scanned.
    fn write_names(&self, image: &Image) -> NameSections {
        let n = self.original.node_count();
        match self.original_names {
            Some(runs) => {
                check_name_runs(self.original, runs);
                NameSections::write_runs(image, n, runs.to_vec(), SHRINK_NAME_IDS)
            }
            None => NameSections::write(
                image,
                n,
                |v| self.original.node_data(NodeIndex::from_usize(v)).name(),
                SHRINK_NAME_IDS,
            ),
        }
    }
}

/// Handles to the shrink sections of an image.
#[derive(Clone, Copy, Debug)]
pub struct ShrinkSections {
    alias: AliasSections,
    names: NameSections,
    node_count: usize,
}

impl ShrinkSections {
    pub fn from_image(image: &Image) -> std::io::Result<Self> {
        Ok(Self {
            alias: AliasSections::from_image(image)?,
            names: NameSections::from_image(image, SHRINK_NAME_IDS)?,
            node_count: image.header().graph_node_count as usize,
        })
    }

    pub fn view<'a>(&self, image: &'a Image) -> ShrinkView<'a> {
        let alias = &self.alias;
        ShrinkView {
            blocks: image.get(alias.blocks),
            runs: image.get(alias.runs),
            p1: image.get(alias.p1),
            chains: image.get(alias.chains),
            far: image.get(alias.far),
            lens: image.get(alias.lens),
            dict: image.get(alias.dict),
            long_lens: image.get(alias.long_lens),
            names: self.names.view(image),
            node_count: self.node_count,
        }
    }
}

/// Handles to the alias sections (see [`RunLayout`]).
#[derive(Clone, Copy, Debug)]
struct AliasSections {
    blocks: Sec<AliasBlock>,
    runs: Sec<RunRec>,
    p1: Sec<P1Rec>,
    chains: Sec<PrunedChain>,
    far: Sec<FarPorts>,
    lens: Sec<LenBlock>,
    dict: Sec<u32>,
    /// The lengths outside the dictionary, `(node, len)` sorted by node.
    long_lens: Sec<LongLen>,
}

impl AliasSections {
    /// Write the finished `layout`'s arrays.
    fn write(image: &Image, layout: &RunLayout) -> Self {
        let runs: Vec<RunRec> = layout.runs.iter().map(|r| r.1).collect();
        let p1: Vec<P1Rec> = layout.p1.iter().map(|r| r.1).collect();
        Self {
            blocks: image.push_section(sections::SHRINK_ALIAS_BLOCKS, &layout.blocks),
            runs: image.push_section(sections::SHRINK_RUNS, &runs),
            p1: image.push_section(sections::SHRINK_P1, &p1),
            chains: image.push_section(sections::SHRINK_PCHAINS, &layout.chains),
            far: image.push_section(sections::SHRINK_FAR_PORTS, &layout.far),
            lens: image.push_section(sections::SHRINK_LEN_BLOCKS, &layout.lens),
            dict: image.push_section(sections::SHRINK_LEN_DICT, &layout.dict),
            long_lens: image.push_section(sections::SHRINK_LONG_LENS, &layout.escapes),
        }
    }

    fn from_image(image: &Image) -> std::io::Result<Self> {
        Ok(Self {
            blocks: image.require(sections::SHRINK_ALIAS_BLOCKS, "shrink.alias_blocks")?,
            runs: image.require(sections::SHRINK_RUNS, "shrink.runs")?,
            p1: image.require(sections::SHRINK_P1, "shrink.p1")?,
            chains: image.require(sections::SHRINK_PCHAINS, "shrink.pchains")?,
            far: image.require(sections::SHRINK_FAR_PORTS, "shrink.far_ports")?,
            lens: image.require(sections::SHRINK_LEN_BLOCKS, "shrink.len_blocks")?,
            dict: image.require(sections::SHRINK_LEN_DICT, "shrink.len_dict")?,
            long_lens: image.require(sections::SHRINK_LONG_LENS, "shrink.long_lens")?,
        })
    }
}
