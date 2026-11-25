//! `biopath index`: build the dp engine once and write it as a `.bpi` index image.
//!
//! The steps: read the graph (GFA or GBZ) → shrink it (by default; off with `--no-shrink` or
//! `SPQR_SHRINK=0`, see `shrink`) → decompose it (SPQR, in process) → build the engine into the
//! image, with the build options of [`crate::build_options`] → write the image.

use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context;
use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray, index::GraphIndexInteger, io::gfa1::PlainGfaEdgeData,
};
use clap::Parser;
use log::{LevelFilter, info, warn};
use serde::{Deserialize, Serialize};
use spqr_shortest_path_index::{
    decomposition::build_spqr_decomposition,
    distance_index::{BuildOptions, DistanceIndex},
    gfa_graph_extensions::LengthOnlyGfaNodeData,
    graph_limits,
    graph_view::NameRun,
    shrink::{self, ShrinkAttach, ShrinkPlan},
};
use spqr_tree::decomposition::SPQRDecomposition;

use crate::{
    build_options::{build_options_from_env, shrink_from_env},
    io_util::{check_output_directory, check_output_file},
    word_size::{parse_word_size, with_word_size},
};

/// The graph the index is built from: node names and lengths, no sequence bytes.
type IndexGraph<IndexType> =
    BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

/// Which index to build. There is one; the flag stays so existing command lines keep working.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum IndexEngine {
    /// The dp engine as a `.bpi` image (memory-mapped by `biopath query --engine dp --index-in`).
    Dp,
}

#[derive(Parser)]
pub struct Cli {
    /// The log level: off, error, warn, info (a summary of each stage), debug or trace. The log
    /// goes to stderr.
    #[clap(long, default_value = "info")]
    pub(crate) log_level: LevelFilter,

    /// Which index to build; there is one, so the flag is hidden and kept for existing command lines.
    #[clap(long, value_enum, default_value_t = IndexEngine::Dp, hide = true)]
    engine: IndexEngine,

    /// The GFA file containing the graph to index.
    #[clap(long, required_unless_present = "graph_gbz_in")]
    graph_gfa_in: Option<PathBuf>,

    /// A GBZ file (vg / Giraffe) to index instead of a GFA. Nodes are named by their GBZ node id
    /// and the edges are the GBZ's, the graph `vg index -j` builds on from the same file. GBZ
    /// format version 1 (`vg gbwt --gbz-version 1` converts newer ones).
    #[clap(long, conflicts_with = "graph_gfa_in")]
    graph_gbz_in: Option<PathBuf>,

    /// The output file for the index image (`.bpi`; uncompressed: it is memory-mapped at query time).
    #[clap(long)]
    index_out: PathBuf,

    /// Build a hub-label oracle for every SPQR skeleton with at least this many directed nodes (see
    /// `biopath query --rnode-label-threshold`).
    #[clap(long, default_value_t = spqr_shortest_path_index::distance_index::LABEL_MIN_DIRECTED_NODES)]
    rnode_label_threshold: usize,

    /// Index the graph as it is, without the exact pre-reduction (`shrink`) that is on by
    /// default (`SPQR_SHRINK=0` turns it off too). The answers are the same; the index is larger.
    #[clap(long)]
    no_shrink: bool,

    /// If specified, write timing information to the given file in JSON format.
    #[clap(long)]
    timing_out: Option<PathBuf>,

    /// The integer size of every index structure and distance: 32 or 64. A graph whose distances
    /// may not fit is refused before the build, with the word size to use.
    #[clap(long, default_value = "32", value_parser = parse_word_size)]
    word_size: u8,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    let IndexEngine::Dp = cli.engine;
    with_word_size!(cli.word_size, run_dp(cli))
}

/// The `--timing-out` JSON; the keys are read by the benchmark scripts.
#[derive(Serialize, Deserialize)]
struct DpIndexTiming {
    graph_reading_time: f64,
    /// The graph reduction (0 when off).
    #[serde(default)]
    shrink_time: f64,
    spqr_building_time: f64,
    engine_building_time: f64,
    index_writing_time: f64,
    index_bytes: u64,
    total_time: f64,
}

/// `--engine dp`: build the dp engine into an image and write it as a `.bpi` file.
fn run_dp<IndexType: GraphIndexInteger + Send + Sync + 'static>(cli: Cli) -> anyhow::Result<()> {
    let total_start_timestamp = Instant::now();
    check_uncompressed(&cli.index_out)?;
    // Fail now, not after the build, on an output that cannot be written.
    check_output_directory(&cli.index_out, "--index-out")?;
    if let Some(timing_out) = &cli.timing_out {
        check_output_file(timing_out, "--timing-out")?;
    }

    let start_timestamp = Instant::now();
    let (graph, original_names) = read_graph::<IndexType>(&cli)?;
    // Every original position must be a word; the shrunk graph is checked below.
    graph_limits::check_positions(&graph)?;
    let graph_reading_time = start_timestamp.elapsed();

    let start_timestamp = Instant::now();
    let shrink_plan = shrink_if_enabled(&cli, &graph);
    let shrink_time = start_timestamp.elapsed();
    let index_graph = shrink_plan.as_ref().map_or(&graph, |plan| &plan.graph);
    // The engine's distance arithmetic is unchecked: refuse a graph whose distances may not fit.
    graph_limits::check_build(index_graph)?;

    let start_timestamp = Instant::now();
    info!("Building SPQR decomposition in-process from the graph");
    let spqr_decomposition = build_spqr_decomposition(index_graph);
    let spqr_building_time = start_timestamp.elapsed();
    info!("Obtained SPQR decomposition in {spqr_building_time:.2?}");

    let start_timestamp = Instant::now();
    info!("Building DP query engine into the index image");
    let shrink = shrink_plan.as_ref().map(|plan| ShrinkAttach {
        plan,
        original: &graph,
        original_names: original_names.as_deref(),
    });
    let engine = build_engine(&cli, &graph, shrink.as_ref(), spqr_decomposition);
    let engine_building_time = start_timestamp.elapsed();
    info!("Built DP query engine in {engine_building_time:.2?}");

    let start_timestamp = Instant::now();
    let index_bytes = write_index(&engine, &cli.index_out)?;
    let index_writing_time = start_timestamp.elapsed();
    let total_time = total_start_timestamp.elapsed();

    let timing_info = DpIndexTiming {
        graph_reading_time: graph_reading_time.as_secs_f64(),
        shrink_time: shrink_time.as_secs_f64(),
        spqr_building_time: spqr_building_time.as_secs_f64(),
        engine_building_time: engine_building_time.as_secs_f64(),
        index_writing_time: index_writing_time.as_secs_f64(),
        index_bytes,
        total_time: total_time.as_secs_f64(),
    };
    report_timing(&timing_info, cli.timing_out.as_deref())
}

/// A dp index image is memory-mapped at query time, so `--index-out` must not be compressed.
fn check_uncompressed(index_out: &Path) -> anyhow::Result<()> {
    let ext = index_out.extension().and_then(|e| e.to_str()).unwrap_or("");
    if matches!(ext, "gz" | "gzip" | "bgz" | "bgzf" | "bgzip") {
        anyhow::bail!(
            "a dp index image must be written uncompressed (it is memory-mapped at query time); \
             choose an uncompressed --index-out such as graph.bpi"
        );
    }
    Ok(())
}

/// The graph to index and, for GBZ input, its node names: the GBZ ids as integer runs, handed to
/// the shrink sections.
fn read_graph<IndexType: GraphIndexInteger>(
    cli: &Cli,
) -> anyhow::Result<(IndexGraph<IndexType>, Option<Vec<NameRun>>)> {
    let (graph, original_names) = if let Some(gbz_in) = &cli.graph_gbz_in {
        info!("Reading graph from GBZ file {gbz_in:?}");
        let (graph, runs) = crate::gbz_input::read_gbz_with_name_runs::<IndexType>(gbz_in)?;
        (graph, Some(runs))
    } else {
        (read_gfa(cli)?, None)
    };
    info!(
        "Graph has {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count(),
    );
    Ok((graph, original_names))
}

/// The graph of `--graph-gfa-in`.
fn read_gfa<IndexType: GraphIndexInteger>(cli: &Cli) -> anyhow::Result<IndexGraph<IndexType>> {
    let gfa_in = cli
        .graph_gfa_in
        .as_ref()
        .expect("clap requires --graph-gfa-in or --graph-gbz-in");
    info!("Reading graph from GFA file {gfa_in:?}");
    crate::gfa_input::read_gfa(gfa_in)
}

/// Reduce the graph exactly before the decomposition (see `shrink`) unless `--no-shrink` or
/// `SPQR_SHRINK=0`; the index is then built on the reduced graph and keeps the original names plus
/// an alias. `None` when the shrink is off or the reduction refuses the graph.
fn shrink_if_enabled<IndexType: GraphIndexInteger>(
    cli: &Cli,
    graph: &IndexGraph<IndexType>,
) -> Option<ShrinkPlan<IndexType>> {
    let start_timestamp = Instant::now();
    if cli.no_shrink || !shrink_from_env() {
        info!("Indexing the graph as it is (no shrink)");
        return None;
    }
    info!("Shrinking the graph (off with --no-shrink or SPQR_SHRINK=0)");
    match shrink::shrink_graph(graph) {
        Ok(plan) => {
            let st = &plan.stats;
            info!(
                "[counters] shrink: nodes {} -> {} ({:.1}%), edges {} -> {}, pruned allele chains \
                 {}, tip chains {}, in {:.2?}",
                st.original_nodes,
                st.shrunk_nodes,
                100.0 * st.shrunk_nodes as f64 / st.original_nodes.max(1) as f64,
                st.original_edges,
                st.shrunk_edges,
                st.allele_chains,
                st.tip_chains,
                start_timestamp.elapsed(),
            );
            Some(plan)
        }
        Err(reason) => {
            warn!("shrink: {reason}; indexing the graph as it is");
            None
        }
    }
}

/// Build the engine on the shrunk graph when `shrink` is given (keeping `graph`'s names and the
/// alias), else on `graph`; `spqr_decomposition` is of the graph the engine is built on.
fn build_engine<IndexType: GraphIndexInteger + Send + Sync + 'static>(
    cli: &Cli,
    graph: &IndexGraph<IndexType>,
    shrink: Option<&ShrinkAttach<'_, IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>>,
    spqr_decomposition: SPQRDecomposition<'_, IndexGraph<IndexType>>,
) -> DistanceIndex<IndexType> {
    let options = build_options_from_env(cli.rnode_label_threshold);
    match shrink {
        Some(attach) => {
            let options = BuildOptions {
                shrink: Some(attach),
                ..options
            };
            DistanceIndex::build(&attach.plan.graph, spqr_decomposition, &options)
        }
        None => DistanceIndex::build(graph, spqr_decomposition, &options),
    }
}

/// Write the engine's image to `index_out`; the number of bytes written.
fn write_index<IndexType: GraphIndexInteger + 'static>(
    engine: &DistanceIndex<IndexType>,
    index_out: &Path,
) -> anyhow::Result<u64> {
    let index_bytes = engine.image().used() as u64;
    info!("Writing index image ({index_bytes} bytes) to file {index_out:?}");
    engine
        .write_to(index_out)
        .with_context(|| format!("Failed to write index image to file: {index_out:?}"))?;
    Ok(index_bytes)
}

/// Log the timings and write them to `timing_out` if given.
fn report_timing(timing_info: &DpIndexTiming, timing_out: Option<&Path>) -> anyhow::Result<()> {
    info!(
        "Indexing completed successfully in {:.2}s",
        timing_info.total_time
    );
    info!(
        "Index timings in seconds: graph {:.2}, decomposition {:.2}, engine {:.2}, write {:.2}, total {:.2}",
        timing_info.graph_reading_time,
        timing_info.spqr_building_time,
        timing_info.engine_building_time,
        timing_info.index_writing_time,
        timing_info.total_time,
    );
    if let Some(timing_out) = timing_out {
        info!("Writing timing information to file {timing_out:?}");
        let timing_out_file = File::create(timing_out)
            .with_context(|| format!("Failed to create timing output file: {timing_out:?}"))?;
        serde_json::to_writer_pretty(timing_out_file, timing_info).with_context(|| {
            format!("Failed to write timing information to file: {timing_out:?}")
        })?;
    }
    Ok(())
}
