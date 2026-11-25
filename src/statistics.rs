//! `biopath statistics`: size statistics of a graph and its SPQR decomposition. It prints a census
//! of the SPQR skeletons (node types, directed skeleton sizes, the largest R- and S-nodes, the SPQR
//! tree depths) and the graph's counts, and writes the counts and the per-component / block /
//! SPQR-node size lists as JSON or TOML. `--shrink` measures G′, the graph `biopath index` builds
//! the index on by default, instead of the input graph.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::File,
    io::Write,
    path::PathBuf,
    time::Instant,
};

use anyhow::Context;
use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::GraphIndexInteger,
    io::gfa1::{GfaEdgeData, GfaNodeData, PlainGfaEdgeData},
};
use clap::Parser;
use itertools::Itertools;
use log::{LevelFilter, debug, info, warn};
use serde::{Deserialize, Serialize};
use spqr_shortest_path_index::{
    decomposition::build_spqr_decomposition,
    gfa_graph_extensions::LengthOnlyGfaNodeData,
    shrink::{ShrinkPlan, shrink_graph},
};
use spqr_tree::{
    decomposition::{SPQRDecomposition, SPQRNodeType, indices::BlockIndex},
    graph::StaticGraph,
};

use crate::{
    io_util::check_output_file,
    word_size::{parse_word_size, with_word_size},
};

#[derive(Parser)]
pub struct Cli {
    /// The log level: off, error, warn, info (a summary of each stage), debug or trace. The log
    /// goes to stderr.
    #[clap(long, default_value = "info")]
    pub(crate) log_level: LevelFilter,

    /// The GFA file containing the graph (plain or gzipped).
    #[clap(long)]
    graph_gfa_in: PathBuf,

    /// The output file for the statistics in JSON format.
    #[clap(long)]
    statistics_json_out: Option<PathBuf>,

    /// The output file for the statistics in TOML format.
    #[clap(long)]
    statistics_toml_out: Option<PathBuf>,

    /// The integer size of the graph's node indices and lengths: 32 or 64. A graph too large for it
    /// is refused with the word size to use.
    #[clap(long, default_value = "32", value_parser = parse_word_size)]
    word_size: u8,

    /// Reduce the graph exactly first (the shrink `biopath index` applies unless `--no-shrink`), so
    /// every statistic describes G′, the graph the index is built on, instead of the input graph.
    /// Falls back to the input graph (with a warning) when the reduction refuses the graph.
    #[clap(long)]
    shrink: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct Statistics<IndexType> {
    node_count: IndexType,
    edge_count: IndexType,

    component_count: IndexType,
    block_count: IndexType,
    spqr_node_count: IndexType,

    component_node_counts: Vec<IndexType>,
    block_node_counts: Vec<IndexType>,
    spqr_node_node_counts: Vec<IndexType>,
    spqr_node_densities_sorted_by_node_count: Vec<f32>,
    spqr_r_node_node_counts: Vec<IndexType>,
    spqr_r_node_densities_sorted_by_node_count: Vec<f32>,
    spqr_s_node_node_counts: Vec<IndexType>,
    spqr_p_node_node_counts: Vec<IndexType>,

    component_block_counts: Vec<IndexType>,
    block_spqr_node_counts: Vec<IndexType>,
    /// Per block, the number of cut nodes incident to it (undirected: a block with `c` cut nodes
    /// has `2c` directed cut states).
    block_cut_node_counts: Vec<IndexType>,
    /// Per distinct cut node, the number of blocks it separates (its degree in the block-cut forest).
    cut_node_block_counts: Vec<IndexType>,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    for out in [&cli.statistics_json_out, &cli.statistics_toml_out]
        .into_iter()
        .flatten()
    {
        check_output_file(out, "the statistics output")?;
    }
    with_word_size!(cli.word_size, run_with_word_size(cli))
}

type InputGraph<IndexType> =
    BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

fn run_with_word_size<IndexType: GraphIndexInteger>(cli: Cli) -> anyhow::Result<()> {
    info!("Reading graph from GFA file {:?}", cli.graph_gfa_in);
    let graph: InputGraph<IndexType> = crate::gfa_input::read_gfa(&cli.graph_gfa_in)?;
    spqr_shortest_path_index::graph_limits::check_positions(&graph)?;

    debug!(
        "Graph has {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count(),
    );

    if !cli.shrink {
        println!("shrink: off; statistics are on the input graph (pass --shrink for G′)");
        return run_on_graph::<IndexType, _, _>(&cli, &graph);
    }
    match shrink(&graph) {
        Some(plan) => run_on_graph::<IndexType, _, _>(&cli, &plan.graph),
        None => run_on_graph::<IndexType, _, _>(&cli, &graph),
    }
}

/// `--shrink`: G′, or `None` when the reduction refuses the graph (the statistics then describe the
/// input graph).
fn shrink<IndexType: GraphIndexInteger>(
    graph: &InputGraph<IndexType>,
) -> Option<ShrinkPlan<IndexType>> {
    info!("Shrinking the graph (--shrink)");
    let start = Instant::now();
    match shrink_graph(graph) {
        Ok(plan) => {
            let st = &plan.stats;
            info!(
                "[counters] shrink: nodes {} -> {} ({:.1}%), edges {} -> {} ({} duplicate links dropped), \
                 pruned allele chains {} ({} nodes), tip chains {} ({} nodes), cyclic chains {}, \
                 max node length {}, in {:.2?}",
                st.original_nodes,
                st.shrunk_nodes,
                100.0 * st.shrunk_nodes as f64 / st.original_nodes.max(1) as f64,
                st.original_edges,
                st.shrunk_edges,
                st.duplicate_edges,
                st.allele_chains,
                st.allele_nodes,
                st.tip_chains,
                st.tip_nodes,
                st.cyclic_chains,
                st.max_node_len,
                start.elapsed(),
            );
            println!(
                "shrink: input nodes {} edges {} -> G′ nodes {} edges {} ({:.1}% of nodes kept); \
                 every statistic below is on G′",
                st.original_nodes,
                st.original_edges,
                st.shrunk_nodes,
                st.shrunk_edges,
                100.0 * st.shrunk_nodes as f64 / st.original_nodes.max(1) as f64,
            );
            Some(plan)
        }
        Err(reason) => {
            warn!("--shrink: {reason}; collecting the statistics on the input graph instead");
            println!("shrink: REFUSED ({reason}); statistics are on the input graph");
            None
        }
    }
}

/// The statistics proper, on whichever graph (the input or G′) `run_with_word_size` settled on.
fn run_on_graph<IndexType, NodeData, EdgeData>(
    cli: &Cli,
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
) -> anyhow::Result<()>
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    info!("Building SPQR decomposition in-process from the graph");
    let spqr_decomposition = build_spqr_decomposition(graph);

    info!("Collecting statistics");
    let statistics = collect_statistics(graph, &spqr_decomposition);
    let census = Census::measure(&spqr_decomposition);
    census.print(statistics.node_count, statistics.edge_count);

    info!("Printing short statistics");
    println!("node_count = {}", statistics.node_count);
    println!("edge_count = {}", statistics.edge_count);
    println!("component_count = {}", statistics.component_count);
    println!("block_count = {}", statistics.block_count);
    println!("spqr_node_count = {}", statistics.spqr_node_count);

    write_statistics(cli, &statistics)?;
    info!("Finished");
    Ok(())
}

/// `values`, largest first.
fn descending<T: Ord>(values: impl Iterator<Item = T>) -> Vec<T> {
    values.sorted().rev().collect()
}

/// Edge density (skeleton edges per vertex) of each SPQR node of `nodes`, ordered by vertex count
/// descending (ties by edge count).
fn densities_by_node_count(nodes: impl Iterator<Item = (usize, usize)>) -> Vec<f32> {
    nodes
        .sorted()
        .rev()
        .map(|(node_count, edge_count)| edge_count as f32 / node_count as f32)
        .collect()
}

/// The size counts and sorted size lists of the graph and its decomposition.
fn collect_statistics<Graph: StaticGraph>(
    graph: &Graph,
    spqr: &SPQRDecomposition<'_, Graph>,
) -> Statistics<usize> {
    // Per-block cut-node counts, and per-cut-node block degrees. Cut nodes are reached through the
    // blocks that share them, so the degrees are deduplicated by the underlying graph node.
    let mut block_cut_node_counts = Vec::new();
    let mut cut_node_block_counts_by_node = HashMap::new();
    for (_, block) in spqr.iter_blocks() {
        let mut count = 0usize;
        for cut in block.iter_cut_nodes() {
            count += 1;
            let cut_node = spqr.cut_node(cut);
            cut_node_block_counts_by_node
                .entry(cut_node.node())
                .or_insert_with(|| cut_node.iter_adjacent_blocks().count());
        }
        block_cut_node_counts.push(count);
    }

    let spqr_nodes = || spqr.iter_spqr_nodes().map(|(_, node)| node);
    Statistics {
        node_count: graph.node_count(),
        edge_count: graph.edge_count(),

        component_count: spqr.component_count(),
        block_count: spqr.block_count(),
        spqr_node_count: spqr.spqr_node_count(),

        component_node_counts: descending(spqr.iter_components().map(|c| c.1.node_count())),
        block_node_counts: descending(spqr.iter_blocks().map(|b| b.1.node_count())),
        spqr_node_node_counts: descending(spqr_nodes().map(|n| n.node_count())),
        spqr_node_densities_sorted_by_node_count: densities_by_node_count(
            spqr_nodes().map(|n| (n.node_count(), n.skeleton_edge_count())),
        ),
        spqr_r_node_node_counts: descending(
            spqr_nodes()
                .filter(|n| n.is_r_node())
                .map(|n| n.node_count()),
        ),
        spqr_r_node_densities_sorted_by_node_count: densities_by_node_count(
            spqr_nodes()
                .filter(|n| n.is_r_node())
                .map(|n| (n.node_count(), n.skeleton_edge_count())),
        ),
        spqr_s_node_node_counts: descending(
            spqr_nodes()
                .filter(|n| n.is_s_node())
                .map(|n| n.node_count()),
        ),
        spqr_p_node_node_counts: descending(
            spqr_nodes()
                .filter(|n| n.is_p_node())
                .map(|n| n.node_count()),
        ),

        component_block_counts: descending(spqr.iter_components().map(|c| c.1.block_count())),
        block_spqr_node_counts: descending(spqr.iter_blocks().map(|b| b.1.spqr_node_count())),
        block_cut_node_counts: descending(block_cut_node_counts.into_iter()),
        cut_node_block_counts: descending(cut_node_block_counts_by_node.into_values()),
    }
}

/// `--statistics-json-out` and `--statistics-toml-out`.
fn write_statistics(cli: &Cli, statistics: &Statistics<usize>) -> anyhow::Result<()> {
    if let Some(json_out) = &cli.statistics_json_out {
        info!("Writing full statistics to JSON file {:?}", json_out);
        let mut file = File::create(json_out)
            .with_context(|| format!("Failed to create JSON output file {:?}", json_out))?;
        serde_json::to_writer(&mut file, statistics)
            .with_context(|| format!("Failed to write statistics to JSON file {:?}", json_out))?;
    }

    if let Some(toml_out) = &cli.statistics_toml_out {
        info!("Writing full statistics to TOML file {:?}", toml_out);
        let mut file = File::create(toml_out)
            .with_context(|| format!("Failed to create TOML output file {:?}", toml_out))?;
        let toml_string =
            toml::to_string(statistics).with_context(|| "Failed to format statistics as TOML")?;
        file.write_all(toml_string.as_bytes())
            .with_context(|| format!("Failed to write TOML output file {:?}", toml_out))?;
    }
    Ok(())
}

// The skeleton census. Convention: `k_v` is the DIRECTED skeleton size of SPQR node v = 2 x its
// skeleton vertex count, poles included. Poles are shared with the parent, so a pole is counted at
// both endpoints of every SPQR tree edge. Two sums are reported:
//   * inclusive: Sum_v k_v (what an index over the skeletons stores) ~= 2n + 4 * (tree edges)
//   * exclusive: the 2 shared poles subtracted at every non-root node; it must equal 2 * Sum_B |B|,
//     because each vertex of a block is introduced at exactly one SPQR node (the shallowest one
//     containing it).

/// The census of the SPQR skeletons, gathered in one pass over the blocks.
#[derive(Default)]
struct Census {
    s_cnt: usize,
    p_cnt: usize,
    r_cnt: usize,
    /// Sum_v k_v, poles counted at every node containing them.
    sum_kv_incl: u128,
    /// Poles charged once (the root keeps them): must equal 2 * Sum_B |B|.
    sum_kv_excl: u128,
    /// The directed states by node type.
    sum_ks: u128,
    sum_kp: u128,
    sum_kr: u128,
    /// Directed R-node skeleton sizes, largest first.
    k_r: Vec<usize>,
    /// Directed S-node skeleton sizes, largest first.
    k_s: Vec<usize>,
    /// The SPQR tree depth of every block, and of the blocks with >= 2 SPQR nodes.
    depths: Vec<usize>,
    nontrivial_depths: Vec<usize>,
    /// Sum_B (#cut nodes of B), and the number of distinct cut nodes.
    cut_incidences: u128,
    distinct_cuts: usize,
    /// The largest block: vertices, SPQR tree depth, SPQR nodes.
    biggest_block_nodes: usize,
    biggest_block_depth: usize,
    biggest_block_spqr: usize,
    /// Sum_B |B|: blocks overlap at cut nodes and omit isolated vertices.
    sum_block_vertices: u128,
}

impl Census {
    fn measure<Graph: StaticGraph>(spqr: &SPQRDecomposition<'_, Graph>) -> Self {
        let mut census = Self::default();
        let mut distinct_cuts = HashSet::new();
        for (block_index, block) in spqr.iter_blocks() {
            for cut in block.iter_cut_nodes() {
                census.cut_incidences += 1;
                distinct_cuts.insert(spqr.cut_node(cut).node());
            }
            census.sum_block_vertices += block.node_count() as u128;
            let mut block_spqr_nodes = 0usize;
            let mut block_kv_sum: u128 = 0;
            for (_, node) in spqr.iter_spqr_nodes_in_block(block_index) {
                block_spqr_nodes += 1;
                let k_v = 2 * node.node_count(); // directed states
                block_kv_sum += k_v as u128;
                census.add_node(node.spqr_node_type(), k_v);
            }
            census.sum_kv_incl += block_kv_sum;
            // Each of the (block_spqr_nodes - 1) tree edges shares 2 poles = 4 directed states counted
            // twice; charging them once leaves exactly the block's own directed states.
            census.sum_kv_excl +=
                block_kv_sum.saturating_sub(4 * (block_spqr_nodes.saturating_sub(1)) as u128);

            let d = block_spqr_tree_depth(spqr, block_index);
            census.add_block_depth(block.node_count(), block_spqr_nodes, d);
        }
        census.distinct_cuts = distinct_cuts.len();
        census.k_r.sort_unstable_by(|a, b| b.cmp(a));
        census.k_s.sort_unstable_by(|a, b| b.cmp(a));
        census
    }

    /// Count one SPQR node of type `node_type` with `k_v` directed states.
    fn add_node(&mut self, node_type: SPQRNodeType, k_v: usize) {
        match node_type {
            SPQRNodeType::SNode => {
                self.s_cnt += 1;
                self.sum_ks += k_v as u128;
                self.k_s.push(k_v);
            }
            SPQRNodeType::PNode => {
                self.p_cnt += 1;
                self.sum_kp += k_v as u128;
            }
            SPQRNodeType::RNode => {
                self.r_cnt += 1;
                self.sum_kr += k_v as u128;
                self.k_r.push(k_v);
            }
        }
    }

    /// Record the SPQR tree depth `d` of a block with `block_nodes` vertices and `block_spqr_nodes`
    /// SPQR nodes.
    fn add_block_depth(&mut self, block_nodes: usize, block_spqr_nodes: usize, d: usize) {
        self.depths.push(d);
        if block_spqr_nodes >= 2 {
            self.nontrivial_depths.push(d);
        }
        if block_nodes > self.biggest_block_nodes {
            self.biggest_block_nodes = block_nodes;
            self.biggest_block_depth = d;
            self.biggest_block_spqr = block_spqr_nodes;
        }
    }

    /// The largest SPQR tree depth of a block.
    fn d_max(&self) -> usize {
        self.depths.iter().copied().max().unwrap_or(0)
    }

    /// The census of a graph with `n` nodes and `m` edges.
    fn print(&self, n: usize, m: usize) {
        println!();
        println!("================ SPQR skeleton census ================");
        println!(
            "convention: k_v = 2 x skeleton vertex count (DIRECTED states), poles INCLUDED at every node"
        );
        println!("graph size        n = {n}   m = {m}   2n = {}", 2 * n);
        println!(
            "SPQR nodes        S = {}   P = {}   R = {}   (total {})",
            self.s_cnt,
            self.p_cnt,
            self.r_cnt,
            self.s_cnt + self.p_cnt + self.r_cnt
        );
        self.print_states_by_type();
        self.print_skeleton_sums(n);
        self.print_largest();
        self.print_depths();
    }

    /// The directed states of the S-, P- and R-nodes, where the node counts count the nodes.
    fn print_states_by_type(&self) {
        let (sum_ks, sum_kp, sum_kr) = (self.sum_ks, self.sum_kp, self.sum_kr);
        let (s_cnt, p_cnt, r_cnt) = (self.s_cnt, self.p_cnt, self.r_cnt);
        let mass_total = self.sum_kv_incl.max(1);
        println!(
            "states by type    Sum_S k_v = {sum_ks} ({:.2}%)   Sum_P k_v = {sum_kp} ({:.2}%)   Sum_R k_R = {sum_kr} ({:.2}%)",
            100.0 * sum_ks as f64 / mass_total as f64,
            100.0 * sum_kp as f64 / mass_total as f64,
            100.0 * sum_kr as f64 / mass_total as f64,
        );
        println!(
            "                  => S/P by count = {:.2}%, but S/P by states = {:.2}%   (R holds {:.2}% of all directed states)",
            100.0 * (s_cnt + p_cnt) as f64 / (s_cnt + p_cnt + r_cnt).max(1) as f64,
            100.0 * (sum_ks + sum_kp) as f64 / mass_total as f64,
            100.0 * sum_kr as f64 / mass_total as f64,
        );
    }

    /// The skeleton sums and their consistency check.
    fn print_skeleton_sums(&self, n: usize) {
        let (sum_kv_incl, sum_kv_excl) = (self.sum_kv_incl, self.sum_kv_excl);
        let (sum_block_vertices, cut_incidences) = (self.sum_block_vertices, self.cut_incidences);
        println!(
            "skeleton sizes    Sum_v k_v (poles incl) = {sum_kv_incl}   [ratio to 2n = {:.3}]",
            sum_kv_incl as f64 / (2 * n).max(1) as f64
        );
        // The check is anchored on the MEASURED block vertex sum, not on n: blocks OVERLAP at cut
        // nodes (a cut node lives in every block it separates) AND omit vertices in no block at all
        // (isolated / degree-0). So the exact identity is
        //     Sum_v k_v (poles excl) == 2 * Sum_B |B|
        // and the deviation of Sum_B |B| from n is then explained by cut duplication minus the
        // unblocked vertices. If the FIRST identity fails, the skeletons really are wrong.
        let cut_dup = cut_incidences.saturating_sub(self.distinct_cuts as u128);
        let expected_excl = 2 * sum_block_vertices;
        println!(
            "                  Sum_v k_v (poles excl) = {sum_kv_excl}   [must equal 2*Sum_B|B| = {expected_excl}] -> {}",
            if sum_kv_excl == expected_excl {
                "OK"
            } else {
                "*** MISMATCH: skeletons suspect ***"
            },
        );
        let unblocked = (n as i128) + (cut_dup as i128) - (sum_block_vertices as i128);
        println!(
            "                  Sum_B|B| = {sum_block_vertices} = n({n}) + cut_dup({cut_dup}) - unblocked({unblocked})   [{cut_incidences} cut incidences over {} distinct cuts]",
            self.distinct_cuts,
        );
        println!(
            "                  (Sum_v k_v equals 2n only when cut_dup = 0 AND every vertex is in a block)"
        );
    }

    /// The largest R- and S-nodes and the R-node size histogram.
    fn print_largest(&self) {
        let (k_r, k_s) = (&self.k_r, &self.k_s);
        if k_r.is_empty() {
            println!("largest R-nodes   none (the decomposition has only S- and P-nodes)");
        } else {
            println!(
                "largest R-nodes   max k_R = {}   top 20: {:?}",
                k_r[0],
                &k_r[..k_r.len().min(20)]
            );
        }
        println!(
            "largest S-nodes   max k_S = {}   top 20: {:?}",
            k_s.first().copied().unwrap_or(0),
            &k_s[..k_s.len().min(20)]
        );
        if !k_r.is_empty() {
            println!("R-node sizes      k_R histogram (log-binned, directed size):");
            print_log_histogram(k_r.iter().map(|&k| k as u64));
        }
    }

    /// The SPQR tree depths.
    fn print_depths(&self) {
        let d_max = self.d_max();
        let mean = |depths: &[usize]| {
            if depths.is_empty() {
                0.0
            } else {
                depths.iter().sum::<usize>() as f64 / depths.len() as f64
            }
        };
        let d_mean = mean(&self.depths);
        let d_mean_nt = mean(&self.nontrivial_depths);
        println!(
            "SPQR tree depth   D per block: mean = {d_mean:.2} over all {} blocks   max = {d_max}",
            self.depths.len()
        );
        println!(
            "                  mean = {d_mean_nt:.2} over the {} non-trivial blocks (>=2 SPQR nodes); \
             largest block: {} vertices, {} SPQR nodes, depth {}",
            self.nontrivial_depths.len(),
            self.biggest_block_nodes,
            self.biggest_block_spqr,
            self.biggest_block_depth,
        );
    }
}

/// Print a log-binned histogram of `values` (decades `10^b .. 10^(b+1) - 1`, the last bin open),
/// one indented line per non-empty bin.
fn print_log_histogram(values: impl IntoIterator<Item = u64>) {
    let mut bucket = [0usize; 11];
    for value in values {
        let mut b = 0usize;
        let mut lim = 10u64;
        while value >= lim && b < 10 {
            b += 1;
            lim = lim.saturating_mul(10);
        }
        bucket[b] += 1;
    }
    for (b, &count) in bucket.iter().enumerate() {
        if count > 0 {
            let lo = 10u64.pow(b as u32);
            println!(
                "        {:>10} .. {:<12} : {}",
                lo,
                lo.saturating_mul(10) - 1,
                count
            );
        }
    }
}

/// The SPQR-tree depth of one block, by BFS from an arbitrary root over the block's SPQR tree.
/// `O(S_block)`: only the depth is needed, not a rooted tree with an LCA table. Depth of a
/// single-node tree is 0.
fn block_spqr_tree_depth<Graph: StaticGraph>(
    spqr: &SPQRDecomposition<'_, Graph>,
    block_index: BlockIndex<Graph::IndexType>,
) -> usize {
    let entries: Vec<_> = spqr.iter_spqr_nodes_in_block(block_index).collect();
    if entries.len() <= 1 {
        return 0;
    }
    let local: HashMap<_, usize> = entries
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, i))
        .collect();
    let mut adjacency = vec![Vec::new(); entries.len()];
    for (li, (id, node)) in entries.iter().enumerate() {
        for edge_index in node.iter_incident_spqr_edges() {
            let (a, b) = spqr.spqr_edge(edge_index).endpoints();
            let other = if a == *id { b } else { a };
            if let Some(&lj) = local.get(&other) {
                adjacency[li].push(lj);
            }
        }
    }
    let mut depth = vec![usize::MAX; entries.len()];
    let mut queue = VecDeque::new();
    depth[0] = 0;
    queue.push_back(0usize);
    let mut max_depth = 0;
    while let Some(u) = queue.pop_front() {
        for &v in &adjacency[u] {
            if depth[v] == usize::MAX {
                depth[v] = depth[u] + 1;
                max_depth = max_depth.max(depth[v]);
                queue.push_back(v);
            }
        }
    }
    max_depth
}
