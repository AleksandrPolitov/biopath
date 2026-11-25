//! `--engine dp`: get the dp engine, either loaded from a `.bpi` index (`--index-in`, word size from
//! the image header) or built in memory from the graph, then answer the queries one at a time.

use std::{
    str::FromStr,
    time::{Duration, Instant},
};

use anyhow::Context;
use bidirected_adjacency_array::{index::GraphIndexInteger, io::gfa1::GfaNodeData};
use indicatif::ProgressBar;
use log::{info, warn};
use spqr_shortest_path_index::{
    decomposition::build_spqr_decomposition, distance_index::DistanceIndex, graph_limits,
    graph_view, image::Image, path::OptionalGfaPathLength,
};

use super::{
    Cli, DpGraph, dijkstra,
    report::{self, QueryTimes, StageTimes},
};
use crate::{build_options::build_options_from_env, query_file::Query, word_size::with_word_size};

/// The graph from `--graph-gbz-in` or `--graph-gfa-in`; `purpose` is appended to the log line.
fn read_dp_graph<IndexType: GraphIndexInteger>(
    cli: &Cli,
    purpose: &str,
) -> anyhow::Result<DpGraph<IndexType>> {
    if let Some(gbz_in) = &cli.graph_gbz_in {
        info!("Reading graph from GBZ file {gbz_in:?}{purpose}");
        return crate::gbz_input::read_gbz(gbz_in);
    }
    let gfa_in = super::require_gfa(cli)?;
    info!("Reading graph from GFA file {gfa_in:?}{purpose}");
    crate::gfa_input::read_gfa(gfa_in)
}

/// `--engine dp` without an index: read the graph and the queries, build the engine in memory, and
/// answer the queries.
pub(super) fn run_in_memory<IndexType: GraphIndexInteger + FromStr + Send + Sync + 'static>(
    cli: Cli,
) -> anyhow::Result<()>
where
    <IndexType as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let total_start_timestamp = Instant::now();
    let (graph, queries, mut times) = read_graph_and_queries::<IndexType>(&cli)?;

    let start_timestamp = Instant::now();
    info!("Building SPQR decomposition in-process from the graph");
    let spqr_decomposition = build_spqr_decomposition(&graph);
    info!(
        "Obtained SPQR decomposition in {:.2?}",
        start_timestamp.elapsed()
    );
    times.spqr_reading = start_timestamp.elapsed();

    let start_timestamp = Instant::now();
    info!("Building DP query engine");
    // The engine takes the decomposition by value and releases it once its node topology and block
    // trees are extracted, before the per-block DP where the build's memory peaks.
    let engine = DistanceIndex::build(
        &graph,
        spqr_decomposition,
        &build_options_from_env(cli.label_threshold()),
    );
    times.engine_building = start_timestamp.elapsed();
    info!("Built DP query engine in {:.2?}", times.engine_building);
    finish_dp_run(
        &cli,
        &engine,
        queries,
        Some(&graph),
        times,
        total_start_timestamp,
    )
}

/// Read the graph, then the queries through a node name index that is released before the engine
/// build, so it is not resident at the build's memory peak.
fn read_graph_and_queries<IndexType: GraphIndexInteger + FromStr>(
    cli: &Cli,
) -> anyhow::Result<(DpGraph<IndexType>, Vec<Query<IndexType>>, StageTimes)>
where
    <IndexType as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let mut times = StageTimes::default();
    let start_timestamp = Instant::now();
    let graph = read_dp_graph::<IndexType>(cli, "")?;
    // The engine's distance arithmetic is unchecked: refuse a graph whose distances may not fit.
    graph_limits::check_build(&graph)?;
    info!(
        "Graph has {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count(),
    );
    times.graph_reading = start_timestamp.elapsed();

    let start_timestamp = Instant::now();
    let node_name_index = super::node_name_index(&graph);
    times.name_index = start_timestamp.elapsed();

    let start_timestamp = Instant::now();
    let queries = super::read_queries(cli, |node_name| {
        let &node = node_name_index.get(node_name)?;
        Some((node, graph.node_data(node).sequence().len() as u64))
    })?;
    times.query_reading = start_timestamp.elapsed();
    drop(node_name_index);
    Ok((graph, queries, times))
}

/// `--engine dp --index-in`: open the `.bpi` image and answer the queries from it. The word size
/// comes from the image header.
pub(super) fn run_from_index(cli: Cli) -> anyhow::Result<()> {
    let path = cli.index_in.clone().expect("checked by the caller");
    let total_start_timestamp = Instant::now();
    info!("Opening index image {path:?}");
    let image =
        Image::open(&path).with_context(|| format!("Failed to open index image {path:?}"))?;
    let header = image.header();
    info!(
        "Index image: {} bytes, word size {} bits, {} nodes, {} edges, {} sections",
        image.used(),
        header.word_size * 8,
        header.graph_node_count,
        header.graph_edge_count,
        header.toc_len,
    );
    with_word_size!(
        u64::from(header.word_size) * 8,
        run_loaded(cli, image, total_start_timestamp),
        other => anyhow::bail!("Unsupported word size {other} bits in the index image")
    )
}

fn run_loaded<IndexType: GraphIndexInteger + FromStr + 'static>(
    cli: Cli,
    image: Image,
    total_start_timestamp: Instant,
) -> anyhow::Result<()>
where
    <IndexType as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let engine = DistanceIndex::<IndexType>::from_image(image)
        .with_context(|| "Failed to open the index image as a dp engine")?;
    let index_reading_time = total_start_timestamp.elapsed();
    info!("Loaded DP engine from the index in {index_reading_time:.2?}");
    warn_on_ignored_build_flags(&cli, &engine);

    let start_timestamp = Instant::now();
    let queries = super::read_queries(&cli, |node_name| {
        let node = engine.node_by_name(node_name)?;
        Some((node, engine.node_len(node)?))
    })?;
    let query_reading_time = start_timestamp.elapsed();

    let (graph, graph_reading_time) = read_cross_check_graph::<IndexType>(&cli, &engine)?;

    finish_dp_run(
        &cli,
        &engine,
        queries,
        graph.as_ref(),
        StageTimes {
            graph_reading: graph_reading_time,
            index_reading: index_reading_time,
            query_reading: query_reading_time,
            ..StageTimes::default()
        },
        total_start_timestamp,
    )
}

/// Build-affecting settings live in the image; warn when a CLI flag given explicitly asks for
/// another.
fn warn_on_ignored_build_flags<IndexType: GraphIndexInteger + 'static>(
    cli: &Cli,
    engine: &DistanceIndex<IndexType>,
) {
    use spqr_shortest_path_index::sections::knob;
    let header = engine.image().header();
    let threshold = header.build_knobs[knob::LABEL_THRESHOLD];
    if let Some(given) = cli.rnode_label_threshold
        && given as u64 != threshold
    {
        warn!(
            "--rnode-label-threshold {given} is ignored: the index was built with threshold {threshold}"
        );
    }
    let word_bits = u64::from(header.word_size) * 8;
    if let Some(given) = cli.word_size
        && u64::from(given) != word_bits
    {
        warn!("--word-size {given} is ignored: the index was built with word size {word_bits}");
    }
}

/// The graph for `--compare-whole-graph-dijkstra` (`None` without it) and its reading time, checked
/// to be the graph the index was built from.
fn read_cross_check_graph<IndexType: GraphIndexInteger + 'static>(
    cli: &Cli,
    engine: &DistanceIndex<IndexType>,
) -> anyhow::Result<(Option<DpGraph<IndexType>>, Duration)> {
    if !cli.compare_whole_graph_dijkstra {
        return Ok((None, Duration::ZERO));
    }
    let start_timestamp = Instant::now();
    let graph = read_dp_graph::<IndexType>(cli, " (for the Dijkstra cross-check)")?;
    let graph_reading_time = start_timestamp.elapsed();
    let fingerprint = graph_view::graph_fingerprint(&graph);
    let expected = engine.image().header().graph_fingerprint;
    if fingerprint != expected {
        let source = cli.graph_gbz_in.as_ref().or(cli.graph_gfa_in.as_ref());
        anyhow::bail!(
            "the graph {source:?} is not the graph this index was built from (fingerprint {fingerprint:#x} != {expected:#x}); \
             check an index built from a GBZ against that GBZ (a GFA of the same graph may list its edges in another order)"
        );
    }
    Ok((Some(graph), graph_reading_time))
}

/// Answer `queries` with a built or loaded dp engine, report, cross-check against whole-graph
/// Dijkstra when asked (needs `graph`), and write the results and timings.
fn finish_dp_run<IndexType: GraphIndexInteger + 'static>(
    cli: &Cli,
    engine: &DistanceIndex<IndexType>,
    mut queries: Vec<Query<IndexType>>,
    graph: Option<&DpGraph<IndexType>>,
    stages: StageTimes,
    total_start_timestamp: Instant,
) -> anyhow::Result<()> {
    if cli.compare_whole_graph_dijkstra && graph.is_none() {
        anyhow::bail!(
            "--compare-whole-graph-dijkstra needs the graph: --graph-gfa-in or --graph-gbz-in"
        );
    }

    let times = answer_queries(engine, &mut queries);
    let average_secs = times.average_secs();
    info!(
        "Finished executing {} queries in {:.2?} ({:.1}µs per query)",
        queries.len(),
        times.total,
        average_secs * 1_000_000.0,
    );
    report::log_time_distribution(&times);

    if cli.compare_whole_graph_dijkstra {
        let graph = graph.expect("--compare-whole-graph-dijkstra needs the graph (checked above)");
        dijkstra::cross_check(graph, &queries, average_secs);
    }

    let start_timestamp = Instant::now();
    report::write_query_results(&cli.query_out, &queries, |node| engine.node_name(node))?;
    let query_writing_time = start_timestamp.elapsed();
    report::finish(
        cli.timing_out.as_deref(),
        stages,
        &times,
        query_writing_time,
        total_start_timestamp,
    )
}

/// Answer every query (setting its distances), timing each one: the timer covers the engine calls
/// only, as vg_mindist times `minimum_distance` only.
fn answer_queries<IndexType: GraphIndexInteger + 'static>(
    engine: &DistanceIndex<IndexType>,
    queries: &mut [Query<IndexType>],
) -> QueryTimes {
    let start_timestamp = Instant::now();
    let total_queries = queries.len();
    let mut single_query_execution_times = Vec::with_capacity(queries.len());
    info!("Executing queries");
    let progress_bar =
        ProgressBar::new(queries.len().try_into().unwrap()).with_message("Executing queries");
    let mut dp_running_time = Duration::ZERO;

    for (query_index, query) in queries.iter_mut().enumerate() {
        let mut distances: Vec<OptionalGfaPathLength<IndexType>> =
            Vec::with_capacity(query.targets().len());
        let single_query_start_timestamp = Instant::now();
        // The dp engine answers one source/target pair at a time.
        for &target in query.targets() {
            distances.push(engine.distance(*query.source(), target).into());
        }
        let elapsed = single_query_start_timestamp.elapsed();
        single_query_execution_times.push(elapsed);
        dp_running_time += elapsed;
        report::log_slow_query(query_index, elapsed, query);
        query.set_distances(distances);

        progress_bar.inc(1);
        report::log_dp_progress(query_index + 1, total_queries, dp_running_time);
    }

    progress_bar.finish_and_clear();
    QueryTimes {
        single: single_query_execution_times,
        total: start_timestamp.elapsed(),
    }
}
