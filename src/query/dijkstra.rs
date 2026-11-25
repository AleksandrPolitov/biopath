//! Plain whole-graph Dijkstra: `--engine dijkstra` (the brute-force baseline) and the
//! `--compare-whole-graph-dijkstra` cross-check of the dp answers.

use std::{collections::HashMap, str::FromStr, time::Instant};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::GraphIndexInteger,
    io::gfa1::{GfaEdgeData, GfaNodeData, PlainGfaEdgeData},
};
use indicatif::ProgressBar;
use log::{info, warn};
use spqr_shortest_path_index::{
    dijkstra::{
        GfaDijkstra,
        gfa_path::GfaPath,
        location_index::{multi::MultiGfaLocationIndex, single::SingleGfaLocationIndex},
    },
    gfa_graph_extensions::LengthOnlyGfaNodeData,
    location::GfaLocation,
    path::OptionalGfaPathLength,
};
use spqr_tree::graph::StaticGraph;

use super::{
    Cli, DpGraph,
    report::{self, QueryTimes, StageTimes},
};
use crate::query_file::Query;

/// The shortest paths from the query's source to its targets (a single-target search for one target).
fn shortest_paths<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
    dijkstra: &mut GfaDijkstra<'_, IndexType, NodeData, EdgeData>,
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    query: &Query<IndexType>,
) -> HashMap<GfaLocation<IndexType>, GfaPath<IndexType>> {
    if query.targets().len() == 1 {
        dijkstra.shortest_paths(
            *query.source(),
            &SingleGfaLocationIndex::new_target(query.targets()[0]),
        )
    } else {
        dijkstra.shortest_paths(
            *query.source(),
            &MultiGfaLocationIndex::new_targets(graph, query.targets().iter().copied()),
        )
    }
}

/// The length of the path to every target of the query (`None` where there is none).
fn target_distances<IndexType: GraphIndexInteger>(
    paths: &HashMap<GfaLocation<IndexType>, GfaPath<IndexType>>,
    query: &Query<IndexType>,
) -> Vec<OptionalGfaPathLength<IndexType>> {
    query
        .targets()
        .iter()
        .map(|&target| paths.get(&target).map(|path| path.length()).into())
        .collect()
}

/// `--engine dijkstra`: answer every query with a whole-graph Dijkstra on the GFA.
pub(super) fn run<IndexType: GraphIndexInteger + FromStr>(cli: Cli) -> anyhow::Result<()>
where
    <IndexType as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let total_start_timestamp = Instant::now();

    let start_timestamp = Instant::now();
    let graph = read_gfa::<IndexType>(&cli)?;
    let graph_reading_time = start_timestamp.elapsed();

    let start_timestamp = Instant::now();
    let node_name_index = super::node_name_index(&graph);
    let node_name_index_building_time = start_timestamp.elapsed();

    let start_timestamp = Instant::now();
    let mut queries = super::read_queries(&cli, |node_name| {
        let &node = node_name_index.get(node_name)?;
        Some((node, graph.node_data(node).sequence().len() as u64))
    })?;
    let query_reading_time = start_timestamp.elapsed();

    let start_timestamp = Instant::now();
    info!("Initialising Dijkstra data structures");
    let mut dijkstra = GfaDijkstra::new(&graph);
    let dijkstra_initialisation_time = start_timestamp.elapsed();

    let times = answer_queries(&mut dijkstra, &graph, &mut queries);
    info!(
        "Finished executing {} queries in {:.2?} ({:.0}µs per query)",
        queries.len(),
        times.total,
        if queries.is_empty() {
            0.0
        } else {
            times.total.as_secs_f64() / queries.len() as f64 * 1_000_000.0
        },
    );

    let start_timestamp = Instant::now();
    report::write_query_results(&cli.query_out, &queries, |node| graph.node_name(node))?;
    let query_writing_time = start_timestamp.elapsed();
    report::finish(
        cli.timing_out.as_deref(),
        StageTimes {
            graph_reading: graph_reading_time,
            name_index: node_name_index_building_time,
            query_reading: query_reading_time,
            engine_building: dijkstra_initialisation_time,
            ..StageTimes::default()
        },
        &times,
        query_writing_time,
        total_start_timestamp,
    )
}

/// The GFA's node names, lengths and links (the search reads no sequence).
fn read_gfa<IndexType: GraphIndexInteger>(cli: &Cli) -> anyhow::Result<DpGraph<IndexType>> {
    let gfa_in = super::require_gfa(cli)?;
    info!("Reading graph from GFA file {:?}", gfa_in);
    let graph = crate::gfa_input::read_gfa(gfa_in)?;
    info!(
        "Graph has {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count(),
    );
    Ok(graph)
}

/// Answer every query (setting its distances); the timer covers the search only.
fn answer_queries<IndexType: GraphIndexInteger>(
    dijkstra: &mut GfaDijkstra<'_, IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>,
    graph: &DpGraph<IndexType>,
    queries: &mut [Query<IndexType>],
) -> QueryTimes {
    let start_timestamp = Instant::now();
    let mut single_query_execution_times = Vec::with_capacity(queries.len());
    info!("Executing queries");
    let progress_bar =
        ProgressBar::new(queries.len().try_into().unwrap()).with_message("Executing queries");

    for query in queries.iter_mut() {
        let single_query_start_timestamp = Instant::now();
        let paths = shortest_paths(dijkstra, graph, query);
        single_query_execution_times.push(single_query_start_timestamp.elapsed());
        query.set_distances(target_distances(&paths, query));
        progress_bar.inc(1);
    }

    progress_bar.finish_and_clear();
    QueryTimes {
        single: single_query_execution_times,
        total: start_timestamp.elapsed(),
    }
}

/// `--compare-whole-graph-dijkstra`: answer the dp-answered `queries` again with whole-graph
/// Dijkstra, print every differing target to stderr, and log both engines' mean query times and
/// whether all answers match. `dp_average_secs` is the dp engine's mean query time.
pub(super) fn cross_check<IndexType: GraphIndexInteger>(
    graph: &DpGraph<IndexType>,
    queries: &[Query<IndexType>],
    dp_average_secs: f64,
) {
    info!("Running whole-graph Dijkstra on the same queries for comparison");
    let mut oracle = GfaDijkstra::new(graph);
    let mut dijkstra_query_time = std::time::Duration::ZERO;
    let mut mismatches = 0usize;
    let dp_avg_us = dp_average_secs * 1_000_000.0;

    for (query_index, query) in queries.iter().enumerate() {
        let single_query_start_timestamp = Instant::now();
        let oracle_distances = target_distances(&shortest_paths(&mut oracle, graph, query), query);
        dijkstra_query_time += single_query_start_timestamp.elapsed();
        mismatches += report_mismatches(query_index, query, &oracle_distances);

        // Whole-graph Dijkstra is the slow side (a full early-stop search per query); log
        // progress with both engines' running averages so it can be watched live.
        let done = query_index + 1;
        if done.is_multiple_of(25) || done == queries.len() {
            let dijkstra_avg_us = dijkstra_query_time.as_secs_f64() / done as f64 * 1_000_000.0;
            info!(
                "  whole-graph Dijkstra: {}/{} done, {} left | Dijkstra avg {:.2}ms/query, DP avg {:.1}µs/query ({:.0}x) | mismatches {}",
                done,
                queries.len(),
                queries.len() - done,
                dijkstra_query_time.as_secs_f64() / done as f64 * 1000.0,
                dp_avg_us,
                if dp_avg_us > 0.0 {
                    dijkstra_avg_us / dp_avg_us
                } else {
                    f64::INFINITY
                },
                mismatches,
            );
        }
    }

    let dijkstra_average_query_time =
        dijkstra_query_time.as_secs_f64() / queries.len().max(1) as f64;
    info!(
        "Average query time over {} queries: SPQR = {:.1}µs, whole-graph Dijkstra = {:.1}µs (Dijkstra/SPQR = {:.1}x)",
        queries.len(),
        dp_average_secs * 1_000_000.0,
        dijkstra_average_query_time * 1_000_000.0,
        if dp_average_secs > 0.0 {
            dijkstra_average_query_time / dp_average_secs
        } else {
            f64::INFINITY
        },
    );
    if mismatches == 0 {
        info!("All query results match whole-graph Dijkstra ✓");
    } else {
        warn!("{mismatches} target distances differ from whole-graph Dijkstra!");
    }
}

/// Print every target whose dp distance differs from Dijkstra's to stderr; the number of them.
fn report_mismatches<IndexType: GraphIndexInteger>(
    query_index: usize,
    query: &Query<IndexType>,
    oracle_distances: &[OptionalGfaPathLength<IndexType>],
) -> usize {
    let mut mismatches = 0;
    for (i, (spqr, oracle)) in query.distances().iter().zip(oracle_distances).enumerate() {
        if spqr != oracle {
            eprintln!(
                "[MISMATCH] q{query_index} tgt#{i}: src={:?} tgt={:?} spqr={spqr:?} oracle={oracle:?}",
                query.source(),
                query.targets()[i],
            );
            mismatches += 1;
        }
    }
    mismatches
}
