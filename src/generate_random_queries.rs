use std::path::PathBuf;

use anyhow::Context;
use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaNodeData, PlainGfaEdgeData},
};
use clap::Parser;
use indicatif::ProgressBar;
use log::{LevelFilter, info};
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng, seq::IteratorRandom};
use spqr_shortest_path_index::{
    gfa_graph_extensions::LengthOnlyGfaNodeData,
    graph_limits,
    location::{GfaLocation, GfaNodeOffset},
};
use spqr_tree::graph::StaticGraph;

use crate::{
    io_util::{check_compressed_output_file, write_optionally_compressed_file},
    query_file::{Query, write_query_file},
    word_size::{parse_word_size, with_word_size},
};

#[derive(Parser)]
pub struct Cli {
    /// The log level: off, error, warn, info (a summary of each stage), debug or trace. The log
    /// goes to stderr.
    #[clap(long, default_value = "info")]
    pub(crate) log_level: LevelFilter,

    /// The GFA file containing the graph to index.
    #[clap(long)]
    graph_gfa_in: PathBuf,

    /// The output file for the generated queries.
    ///
    /// The file format is tab-separated.
    /// The columns are `source_node_id`, `source_orientation`, `source_offset`, `target_node_id`, `target_orientation`, `target_offset`.
    /// The last three columns can be repeated to specify multiple target locations for the same source.
    #[clap(long)]
    query_out: PathBuf,

    /// The number of random queries to generate.
    #[clap(long, default_value = "1000")]
    amount: usize,

    /// The seed for the random generator.
    #[clap(long, default_value = "0")]
    random_seed: u64,

    /// Generate **reachable** queries by a guided random bidirected walk of this many steps,
    /// instead of two independent random locations.
    ///
    /// For each query: pick a random node, choose a side that has an outgoing edge (retry if the
    /// node is a sink on both sides; pick randomly if both sides work), then walk this many steps
    /// following a uniformly random outgoing edge at each step (stopping early at a dead end). The
    /// source is the walk start and the target is the walk end, so a directed path of length at
    /// most `walk_length` always exists between them. Omit for the default independent-random
    /// (mostly unreachable) queries.
    #[clap(long)]
    walk_length: Option<usize>,

    /// The integer size of the graph's node indices and lengths: 32 or 64. A graph too large for it
    /// is refused with the word size to use.
    #[clap(long, default_value = "32", value_parser = parse_word_size)]
    word_size: u8,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    with_word_size!(cli.word_size, run_with_word_size(cli))
}

fn run_with_word_size<IndexType: GraphIndexInteger>(cli: Cli) -> anyhow::Result<()> {
    check_compressed_output_file(&cli.query_out, "--query-out")?;
    info!("Reading graph from GFA file {:?}", cli.graph_gfa_in);
    let graph: BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData> =
        crate::gfa_input::read_gfa(&cli.graph_gfa_in)?;
    graph_limits::check_positions(&graph)?;
    info!(
        "Graph has {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count(),
    );
    anyhow::ensure!(
        graph.node_count() > 0,
        "the graph {:?} has no nodes: there is no position to query",
        cli.graph_gfa_in
    );

    let mut rng = SmallRng::seed_from_u64(cli.random_seed);
    let mut queries = Vec::new();
    let progress_bar =
        ProgressBar::new(cli.amount.try_into().unwrap()).with_message("Generating queries");

    match cli.walk_length {
        Some(k) => info!(
            "Generating {} reachable queries via random bidirected walk of length {k}",
            cli.amount
        ),
        None => info!("Generating {} independent-random queries", cli.amount),
    }

    for _ in 0..cli.amount {
        let query = match cli.walk_length {
            Some(k) => generate_walk_query(&graph, &mut rng, k)?,
            None => {
                let source_location = generate_random_location(&graph, &mut rng)?;
                let target_location = generate_random_location(&graph, &mut rng)?;
                Query::new(source_location, vec![target_location])
            }
        };
        queries.push(query);
        progress_bar.inc(1);
    }

    progress_bar.finish_and_clear();
    info!("Generated {} queries", queries.len());

    info!("Writing queries to file {:?}", cli.query_out);
    write_optionally_compressed_file(&cli.query_out, |writer| {
        write_query_file(writer, &queries, |node_index| {
            Some(graph.node_name(node_index))
        })
    })
    .with_context(|| format!("Failed to write query file: {:?}", cli.query_out))?;

    info!("Finished writing query file");

    Ok(())
}

/// A uniformly random node, orientation and offset (0 on a zero-length node).
fn generate_random_location<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    rng: &mut impl Rng,
) -> anyhow::Result<GfaLocation<IndexType>> {
    let node = (0..graph.node_count())
        .choose(rng)
        .ok_or_else(|| anyhow::anyhow!("the graph has no nodes"))?;
    let directed_node =
        DirectedNodeIndex::from_bidirected(NodeIndex::from_usize(node), rng.random_bool(0.5));
    Ok(GfaLocation::new(
        directed_node,
        random_offset(graph, directed_node, rng),
    ))
}

/// Generate a **reachable** query by a guided random bidirected walk.
///
/// Pick a random node, choose a side (orientation) that has an outgoing edge — retrying with a
/// new node if both sides are sinks, choosing randomly if both sides work — then walk up to
/// `walk_length` steps, following a uniformly random outgoing edge each step and stopping early
/// at a dead end. The source is the walk start and the target is the walk end, so a directed path
/// (of length at most `walk_length`) always exists from source to target.
fn generate_walk_query<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    rng: &mut impl Rng,
    walk_length: usize,
) -> anyhow::Result<Query<IndexType>> {
    let node_count = graph.node_count();
    anyhow::ensure!(
        node_count > 0,
        "cannot generate walk queries on an empty graph"
    );

    // Pick a random node with at least one walkable side, starting from that side. Bounded retries
    // so a graph with edges cannot loop forever while an edgeless graph still fails cleanly.
    const MAX_TRIES: usize = 10_000;
    let mut start = None;
    for _ in 0..MAX_TRIES {
        let node = NodeIndex::from_usize((0..node_count).choose(rng).unwrap());
        let forward = DirectedNodeIndex::from_bidirected(node, true);
        let reverse = DirectedNodeIndex::from_bidirected(node, false);
        let forward_ok = graph.iter_outgoing_edges(forward).next().is_some();
        let reverse_ok = graph.iter_outgoing_edges(reverse).next().is_some();
        start = match (forward_ok, reverse_ok) {
            (true, true) => Some(if rng.random_bool(0.5) {
                forward
            } else {
                reverse
            }),
            (true, false) => Some(forward),
            (false, true) => Some(reverse),
            (false, false) => None,
        };
        if start.is_some() {
            break;
        }
    }
    let start = start.ok_or_else(|| {
        anyhow::anyhow!(
            "could not find a node with an outgoing edge in {MAX_TRIES} tries; \
             is the graph edgeless? omit --walk-length for independent-random queries"
        )
    })?;

    // Walk up to `walk_length` steps along uniformly random outgoing edges (reservoir-sampled, so
    // no per-step allocation), stopping early if a dead end is reached.
    let mut current = start;
    for _ in 0..walk_length {
        match graph.iter_outgoing_edges(current).choose(rng) {
            Some(edge) => current = edge.to(),
            None => break,
        }
    }
    let end = current;

    let source = GfaLocation::new(start, random_offset(graph, start, rng));
    let target = GfaLocation::new(end, random_offset(graph, end, rng));
    Ok(Query::new(source, vec![target]))
}

/// A uniformly random offset within a directed node's sequence (0 for an empty sequence).
fn random_offset<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    node: DirectedNodeIndex<IndexType>,
    rng: &mut impl Rng,
) -> GfaNodeOffset<IndexType> {
    let len = graph.node_data(node.into_bidirected()).sequence().len();
    GfaNodeOffset::from_usize((0..len).choose(rng).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    type Graph = BidirectedAdjacencyArray<u32, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

    fn graph_from_gfa(gfa: &str) -> Graph {
        spqr_shortest_path_index::gfa_reader::read_gfa1(gfa.as_bytes(), |_| {}).expect("valid GFA")
    }

    /// Independent directed-reachability check (BFS), deliberately NOT reusing the walk logic:
    /// is `to` reachable from `from` by following outgoing edges? Small test graphs, so a linear
    /// `seen` list is fine.
    fn reachable(graph: &Graph, from: DirectedNodeIndex<u32>, to: DirectedNodeIndex<u32>) -> bool {
        if from == to {
            return true;
        }
        let mut seen = vec![from];
        let mut stack = vec![from];
        while let Some(current) = stack.pop() {
            for edge in graph.iter_outgoing_edges(current) {
                let next = edge.to();
                if next == to {
                    return true;
                }
                if !seen.contains(&next) {
                    seen.push(next);
                    stack.push(next);
                }
            }
        }
        false
    }

    #[test]
    fn walk_queries_are_reachable() {
        // Branching graph so walks are not a single forced path: N0->N1->{N2->N3, N4->N3}.
        let gfa = "H\tVN:Z:1.0\n\
                   S\tN0\tAAAA\nS\tN1\tCCCC\nS\tN2\tGGGG\nS\tN3\tTTTT\nS\tN4\tACGT\n\
                   L\tN0\t+\tN1\t+\t0M\nL\tN1\t+\tN2\t+\t0M\nL\tN2\t+\tN3\t+\t0M\n\
                   L\tN1\t+\tN4\t+\t0M\nL\tN4\t+\tN3\t+\t0M";
        let graph = graph_from_gfa(gfa);
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..200 {
            let query = generate_walk_query(&graph, &mut rng, 4).expect("walk query");
            let source = query.source().node();
            let target = query.targets()[0].node();
            assert!(
                reachable(&graph, source, target),
                "walk query source {source:?} -> target {target:?} must be reachable",
            );
        }
    }

    #[test]
    fn walk_respects_dead_ends() {
        // Linear chain N0->N1->N2 (N2's forward side is a sink). A walk far longer than the chain
        // must stop at a real node, and source->target must still be reachable.
        let gfa = "H\tVN:Z:1.0\n\
                   S\tN0\tAA\nS\tN1\tCC\nS\tN2\tGG\n\
                   L\tN0\t+\tN1\t+\t0M\nL\tN1\t+\tN2\t+\t0M";
        let graph = graph_from_gfa(gfa);
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..100 {
            let query = generate_walk_query(&graph, &mut rng, 100).expect("walk query");
            assert!(reachable(
                &graph,
                query.source().node(),
                query.targets()[0].node(),
            ));
        }
    }

    #[test]
    fn independent_locations_on_zero_length_nodes_have_offset_0() {
        let graph = graph_from_gfa("S\tN0\t\nS\tN1\tCC\nL\tN0\t+\tN1\t+\t0M\n");
        let mut rng = SmallRng::seed_from_u64(3);
        for _ in 0..100 {
            let location = generate_random_location(&graph, &mut rng).expect("a location");
            let len = graph
                .node_data(location.node().into_bidirected())
                .sequence()
                .len();
            assert!(location.offset().into_usize() < len.max(1));
        }
        let empty = graph_from_gfa("H\tVN:Z:1.0\n");
        assert!(generate_random_location(&empty, &mut rng).is_err());
    }

    #[test]
    fn edgeless_graph_errors() {
        // Two nodes, no links: no walkable side anywhere -> clean error, not a panic/hang.
        let graph = graph_from_gfa("H\tVN:Z:1.0\nS\tN0\tAAAA\nS\tN1\tCCCC");
        let mut rng = SmallRng::seed_from_u64(1);
        assert!(generate_walk_query(&graph, &mut rng, 3).is_err());
    }
}
