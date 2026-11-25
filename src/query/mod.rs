//! `biopath query`: answer a query file and report how long each query took.
//!
//! - [`dp`]: the dp engine, loaded from a `.bpi` index (`--index-in`) or built in memory from the
//!   graph, and the timed query loop.
//! - [`dijkstra`]: `--engine dijkstra` (plain whole-graph Dijkstra, the brute-force baseline) and the
//!   `--compare-whole-graph-dijkstra` cross-check of the dp answers.
//! - [`report`]: the query time distribution and progress lines, the result TSV, and the timings
//!   (`--timing-out` JSON and the human-readable log).

mod dijkstra;
mod dp;
mod report;

use std::{borrow::Cow, collections::HashMap, path::PathBuf, str::FromStr};

use anyhow::Context;
use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaNodeData, PlainGfaEdgeData},
};
use clap::Parser;
use log::{LevelFilter, info, warn};
use spqr_shortest_path_index::gfa_graph_extensions::LengthOnlyGfaNodeData;
use spqr_tree::graph::StaticGraph;

use crate::{
    io_util::{check_compressed_output_file, check_output_file, read_optionally_compressed_file},
    query_file::{Query, read_query_file},
    word_size::{parse_word_size, with_word_size},
};

/// The graph the dp engine and its Dijkstra cross-check read: node names and lengths, no sequence
/// bytes (see `LengthOnlyGfaNodeData`).
type DpGraph<IndexType> =
    BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

/// The shortest-path engine to answer queries with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Engine {
    /// The SPQR dp engine: loaded from `--index-in` (a `.bpi` from `biopath index`), or built in
    /// memory from `--graph-gfa-in` / `--graph-gbz-in` when no index is given.
    Dp,
    /// Plain whole-graph Dijkstra on `--graph-gfa-in` (the brute-force baseline).
    Dijkstra,
}

#[derive(Parser)]
pub struct Cli {
    /// The log level: off, error, warn, info (a summary of each stage), debug or trace. The log
    /// goes to stderr.
    #[clap(long, default_value = "info")]
    pub(crate) log_level: LevelFilter,

    /// Which query engine to use.
    #[clap(long, value_enum, default_value_t = Engine::Dp)]
    engine: Engine,

    /// The GFA file containing the graph (plain or gzipped). Needed by `--engine dijkstra`, by the
    /// in-memory dp build (no `--index-in`) and by `--compare-whole-graph-dijkstra`; an index is
    /// self-contained.
    #[clap(long)]
    graph_gfa_in: Option<PathBuf>,

    /// A GBZ file instead of `--graph-gfa-in`, for `--engine dp` (the in-process build and the
    /// `--compare-whole-graph-dijkstra` cross-check). Nodes are named by their GBZ node id.
    #[clap(long, conflicts_with = "graph_gfa_in")]
    graph_gbz_in: Option<PathBuf>,

    /// A `.bpi` image written by `biopath index`, memory-mapped (and prefaulted) and queried without
    /// rebuilding (and without the graph). Without it, `--engine dp` builds the engine in memory
    /// first. `--engine dp` only.
    #[clap(long)]
    index_in: Option<PathBuf>,

    /// Accepted for existing command lines, as `populate` only: an index is always memory-mapped
    /// and prefaulted whole, which is what `populate` named.
    #[clap(long, hide = true, requires = "index_in", value_parser = parse_index_load)]
    index_load: Option<String>,

    /// Integer word size (32 or 64; default 32) of the in-memory dp engine. An index records its
    /// own; a different value there only warns. `--engine dp` only.
    #[clap(long, value_parser = parse_word_size)]
    word_size: Option<u8>,

    /// Also run whole-graph Dijkstra on the same queries and log the average per-query time of
    /// each engine (SPQR DP vs whole-graph Dijkstra) together with a correctness check. Needs the
    /// graph (`--graph-gfa-in` or `--graph-gbz-in`). `--engine dp` only.
    #[clap(long)]
    compare_whole_graph_dijkstra: bool,

    /// Build a hub-labeling distance oracle for any SPQR skeleton with at least this many directed
    /// nodes (default 200), so its intra-block search is answered by a search-free label lookup
    /// instead of a Dijkstra. Lower = more skeletons labeled (more memory, faster queries); a very
    /// large value disables labeling (pure Dijkstra). For the in-memory build: an index records its
    /// own; a different value there only warns. `--engine dp` only.
    #[clap(long)]
    rnode_label_threshold: Option<usize>,

    /// A tab-separated file containing the queries to run.
    /// The columns are `source_node_id`, `source_orientation`, `source_offset`, `target_node_id`, `target_orientation`, `target_offset`.
    /// The last three columns can be repeated to specify multiple target locations for the same source.
    #[clap(long)]
    query_in: PathBuf,

    /// The output file for the query results: per query, the source and every target as
    /// `node offset orientation` (offset before orientation), each target followed by its
    /// distance in bp (`None` when unreachable). A `.gz` extension compresses it.
    #[clap(long)]
    query_out: PathBuf,

    /// If specified, write timing information to the given file in JSON format.
    #[clap(long)]
    timing_out: Option<PathBuf>,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    cli.check_flags()?;
    // Fail now, not after the queries, on an output that cannot be written.
    check_compressed_output_file(&cli.query_out, "--query-out")?;
    if let Some(timing_out) = &cli.timing_out {
        check_output_file(timing_out, "--timing-out")?;
    }
    match cli.engine {
        Engine::Dp if cli.index_in.is_some() => dp::run_from_index(cli),
        Engine::Dp => with_word_size!(cli.word_size(), dp::run_in_memory(cli)),
        Engine::Dijkstra => dijkstra::run::<u64>(cli),
    }
}

/// The word size of the in-memory dp engine without `--word-size`.
const DEFAULT_WORD_SIZE: u8 = 32;

impl Cli {
    /// `--word-size`, or its default.
    fn word_size(&self) -> u8 {
        self.word_size.unwrap_or(DEFAULT_WORD_SIZE)
    }

    /// `--rnode-label-threshold`, or its default.
    fn label_threshold(&self) -> usize {
        self.rnode_label_threshold
            .unwrap_or(spqr_shortest_path_index::distance_index::LABEL_MIN_DIRECTED_NODES)
    }

    /// Refuse a flag the chosen mode would ignore, and a mode without the inputs it needs, before
    /// anything is read.
    fn check_flags(&self) -> anyhow::Result<()> {
        let has_graph = self.graph_gfa_in.is_some() || self.graph_gbz_in.is_some();
        match self.engine {
            Engine::Dijkstra => {
                let ignored: Vec<&str> = [
                    ("--index-in", self.index_in.is_some()),
                    ("--index-load", self.index_load.is_some()),
                    ("--graph-gbz-in", self.graph_gbz_in.is_some()),
                    (
                        "--compare-whole-graph-dijkstra",
                        self.compare_whole_graph_dijkstra,
                    ),
                    (
                        "--rnode-label-threshold",
                        self.rnode_label_threshold.is_some(),
                    ),
                    ("--word-size", self.word_size.is_some()),
                ]
                .into_iter()
                .filter_map(|(flag, given)| given.then_some(flag))
                .collect();
                anyhow::ensure!(
                    ignored.is_empty(),
                    "{} cannot be used with --engine dijkstra (the whole-graph Dijkstra baseline \
                     reads the graph from --graph-gfa-in and builds no index)",
                    ignored.join(", ")
                );
                anyhow::ensure!(
                    self.graph_gfa_in.is_some(),
                    "--engine dijkstra needs the graph: --graph-gfa-in"
                );
            }
            Engine::Dp => {
                anyhow::ensure!(
                    self.index_in.is_some() || has_graph,
                    "--engine dp needs an index (--index-in) or the graph to build one in memory \
                     (--graph-gfa-in or --graph-gbz-in)"
                );
                anyhow::ensure!(
                    !self.compare_whole_graph_dijkstra || has_graph,
                    "--compare-whole-graph-dijkstra needs the graph: --graph-gfa-in or --graph-gbz-in"
                );
                if self.index_in.is_some() && has_graph && !self.compare_whole_graph_dijkstra {
                    warn!(
                        "the graph is not read: with --index-in it is only needed by \
                         --compare-whole-graph-dijkstra"
                    );
                }
            }
        }
        Ok(())
    }
}

/// The GFA path, or an error naming the flag.
fn require_gfa(cli: &Cli) -> anyhow::Result<&PathBuf> {
    cli.graph_gfa_in.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "--graph-gfa-in is required for this mode{}",
            if cli.graph_gbz_in.is_some() {
                " (--graph-gbz-in works with --engine dp only)"
            } else {
                ""
            }
        )
    })
}

/// The hidden `--index-load`: `populate`, the one way an index is loaded, or an error.
fn parse_index_load(value: &str) -> Result<String, String> {
    if value == "populate" {
        Ok(value.to_string())
    } else {
        Err(
            "only `populate` exists: an index is always memory-mapped and prefaulted whole"
                .to_string(),
        )
    }
}

/// The node of every GFA name, to resolve the query file against a graph.
fn node_name_index<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
) -> HashMap<Cow<'_, str>, NodeIndex<IndexType>> {
    info!("Building node name index");
    graph
        .node_indices()
        .map(|node_index| (graph.node_name(node_index), node_index))
        .collect()
}

/// Read `--query-in`, resolving the node names with `node_by_name` (a node and its length in bp,
/// against which the offsets are checked).
fn read_queries<IndexType: GraphIndexInteger + FromStr>(
    cli: &Cli,
    node_by_name: impl Fn(&str) -> Option<(NodeIndex<IndexType>, u64)>,
) -> anyhow::Result<Vec<Query<IndexType>>>
where
    <IndexType as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    info!("Reading queries from file {:?}", cli.query_in);
    read_optionally_compressed_file(&cli.query_in, |reader| {
        read_query_file(reader, &cli.query_in, node_by_name)
    })
    .with_context(|| format!("Failed to read query file: {:?}", cli.query_in))
}
