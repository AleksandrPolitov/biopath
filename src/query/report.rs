//! What `biopath query` reports: the query time distribution and the dp progress lines, the result
//! TSV, and the timings (`--timing-out` JSON and the human-readable log).

use std::{
    borrow::Cow,
    fmt::Write as _,
    fs::File,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::Context;
use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};
use log::info;
use serde::{Deserialize, Serialize};

use crate::{
    io_util::{write_human_readable_time_to_string, write_optionally_compressed_file},
    query_file::Query,
};

/// The times of the stages before the query loop (zero where a stage did not run).
#[derive(Default, Clone, Copy)]
pub(super) struct StageTimes {
    pub(super) graph_reading: Duration,
    pub(super) name_index: Duration,
    pub(super) spqr_reading: Duration,
    pub(super) index_reading: Duration,
    pub(super) query_reading: Duration,
    /// The in-memory dp engine build, or the Dijkstra initialisation; reported as
    /// `dijkstra_initialisation_time` (0 when the dp engine is loaded from an index).
    pub(super) engine_building: Duration,
}

/// The wall time of every query and of the whole query loop.
pub(super) struct QueryTimes {
    pub(super) single: Vec<Duration>,
    pub(super) total: Duration,
}

impl QueryTimes {
    /// The mean of the single query times, in seconds (0 without queries).
    pub(super) fn average_secs(&self) -> f64 {
        mean(self.single.iter().map(|t| t.as_secs_f64()))
    }
}

/// The `--timing-out` JSON; the keys are read by the benchmark scripts.
#[derive(Serialize, Deserialize)]
struct QueryTiming {
    graph_reading_time: f64,
    node_name_index_building_time: f64,
    spqr_reading_time: f64,
    index_reading_time: f64,
    query_reading_time: f64,
    dijkstra_initialisation_time: f64,
    single_query_execution_times: Vec<f64>,
    total_query_execution_time: f64,
    query_writing_time: f64,

    compute_time: f64,
    io_time: f64,

    total_time: f64,
}

impl QueryTiming {
    fn new(
        stages: StageTimes,
        queries: &QueryTimes,
        query_writing: Duration,
        total: Duration,
    ) -> Self {
        let compute = stages.engine_building + queries.total;
        let io = stages.graph_reading
            + stages.name_index
            + stages.spqr_reading
            + stages.index_reading
            + stages.query_reading
            + query_writing;
        Self {
            graph_reading_time: stages.graph_reading.as_secs_f64(),
            node_name_index_building_time: stages.name_index.as_secs_f64(),
            spqr_reading_time: stages.spqr_reading.as_secs_f64(),
            index_reading_time: stages.index_reading.as_secs_f64(),
            query_reading_time: stages.query_reading.as_secs_f64(),
            dijkstra_initialisation_time: stages.engine_building.as_secs_f64(),
            single_query_execution_times: queries.single.iter().map(|t| t.as_secs_f64()).collect(),
            total_query_execution_time: queries.total.as_secs_f64(),
            query_writing_time: query_writing.as_secs_f64(),

            compute_time: compute.as_secs_f64(),
            io_time: io.as_secs_f64(),

            total_time: total.as_secs_f64(),
        }
    }

    /// One `name: time` line per stage, plus the mean query time and the query count.
    fn human_readable(&self) -> String {
        let query_count = self.single_query_execution_times.len();
        let average = mean(self.single_query_execution_times.iter().copied());
        let before_count = [
            ("graph_reading_time", self.graph_reading_time),
            (
                "node_name_index_building_time",
                self.node_name_index_building_time,
            ),
            ("spqr_reading_time", self.spqr_reading_time),
            ("index_reading_time", self.index_reading_time),
            ("query_reading_time", self.query_reading_time),
            (
                "dijkstra_initialisation_time",
                self.dijkstra_initialisation_time,
            ),
            (
                "total_query_execution_time",
                self.total_query_execution_time,
            ),
            ("average_query_execution_time", average),
        ];
        let after_count = [
            ("query_writing_time", self.query_writing_time),
            ("compute_time", self.compute_time),
            ("io_time", self.io_time),
            ("total_time", self.total_time),
        ];
        let mut text = String::new();
        for (name, seconds) in before_count {
            write_time_line(&mut text, name, seconds);
        }
        writeln!(text, "query_count: {query_count}").unwrap();
        for (name, seconds) in after_count {
            write_time_line(&mut text, name, seconds);
        }
        text
    }
}

/// The mean of `values` (0 when there are none).
pub(super) fn mean(values: impl ExactSizeIterator<Item = f64>) -> f64 {
    let count = values.len();
    if count == 0 {
        return 0.0;
    }
    values.sum::<f64>() / count as f64
}

fn write_time_line(text: &mut String, name: &str, seconds: f64) {
    writeln!(
        text,
        "{name}: {}",
        write_human_readable_time_to_string(seconds)
    )
    .unwrap();
}

/// The end of a query run: log the timings and write them to `timing_out` if given.
pub(super) fn finish(
    timing_out: Option<&Path>,
    stages: StageTimes,
    queries: &QueryTimes,
    query_writing: Duration,
    total_start: Instant,
) -> anyhow::Result<()> {
    let total_time = total_start.elapsed();
    info!("Querying completed successfully");

    let timing_info = QueryTiming::new(stages, queries, query_writing, total_time);
    info!("Query timings:\n{}", timing_info.human_readable());

    if let Some(timing_out) = timing_out {
        info!("Writing timing information to file {timing_out:?}");

        let timing_out_file = File::create(timing_out)
            .with_context(|| format!("Failed to create timing output file: {timing_out:?}"))?;

        serde_json::to_writer_pretty(timing_out_file, &timing_info).with_context(|| {
            format!("Failed to write timing information to file: {timing_out:?}")
        })?;

        info!("Timing information written successfully");
    }
    Ok(())
}

/// Write the result TSV: per query, the source and then every target with its distance (`None`
/// when unreachable), each location as `name offset orientation`.
pub(super) fn write_query_results<'name, IndexType: GraphIndexInteger>(
    query_out: &Path,
    queries: &[Query<IndexType>],
    node_name: impl Fn(NodeIndex<IndexType>) -> Cow<'name, str>,
) -> anyhow::Result<()> {
    info!("Writing query results to file {query_out:?}");
    let orientation = |forward: bool| if forward { "+" } else { "-" };
    write_optionally_compressed_file(query_out, |writer| {
        for query in queries {
            let source = query.source();
            write!(
                writer,
                "{}\t{}\t{}",
                node_name(source.node().into_bidirected()),
                source.offset(),
                orientation(source.node().is_forward()),
            )?;

            for (target, distance) in query.targets().iter().zip(query.distances()) {
                write!(
                    writer,
                    "\t{}\t{}\t{}\t{}",
                    node_name(target.node().into_bidirected()),
                    target.offset(),
                    orientation(target.node().is_forward()),
                    distance
                        .into_option()
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "None".to_string()),
                )?;
            }

            writeln!(writer)?;
        }
        Ok(())
    })
    .with_context(|| format!("Failed to write query results to file: {query_out:?}"))
}

/// The query time distribution (not just the mean), so the tail is not averaged away.
pub(super) fn log_time_distribution(times: &QueryTimes) {
    if times.single.is_empty() {
        return;
    }
    let mut micros: Vec<f64> = times
        .single
        .iter()
        .map(|t| t.as_secs_f64() * 1_000_000.0)
        .collect();
    micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| micros[((micros.len() as f64 * p) as usize).min(micros.len() - 1)];
    info!(
        "Query time distribution (µs): p50={:.1} p90={:.1} p99={:.1} max={:.1} (min={:.1}, mean={:.1})",
        pct(0.50),
        pct(0.90),
        pct(0.99),
        micros[micros.len() - 1],
        micros[0],
        times.average_secs() * 1_000_000.0,
    );
}

/// The dp progress line, every 1000 queries and after the last: a line every 25 queries costs
/// ~0.5 µs per query outside the timer and leaves the next query cache-cold.
pub(super) fn log_dp_progress(done: usize, total: usize, running_time: Duration) {
    if done.is_multiple_of(1000) || done == total {
        info!(
            "  DP engine: {}/{} queries done, {} left, running avg {:.1}µs/query",
            done,
            total,
            total - done,
            running_time.as_secs_f64() / done as f64 * 1_000_000.0,
        );
    }
}

/// Flag a query slower than 0.2 s as it finishes.
pub(super) fn log_slow_query<IndexType: GraphIndexInteger>(
    query_index: usize,
    elapsed: Duration,
    query: &Query<IndexType>,
) {
    if elapsed.as_secs_f64() > 0.2 {
        info!(
            "  [slow-query] q{} took {:.2?} ({} targets), src {:?}",
            query_index,
            elapsed,
            query.targets().len(),
            query.source(),
        );
    }
}
