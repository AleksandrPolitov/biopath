//! `biopath export-shrunk-gfa`: write G′, the graph `biopath index` builds the index on (unless
//! `--no-shrink`), as a GFA that vg can load, so vg's distance index can be built on it and
//! compared with the index of the input graph.
//!
//! G′ keeps only node lengths, so every segment is a run of `N`. vg's distance index cannot store a
//! node longer than 2047 bp, and merged G′ nodes reach megabases, so `--chop` splits every G′ node
//! into a path of pieces of at most that many bp, with the distances between G′ node ends unchanged
//! ([`write_chopped_gfa`] has the layout). No paths are written.

use std::{path::PathBuf, time::Instant};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray, index::GraphIndexInteger, io::gfa1::PlainGfaEdgeData,
};
use clap::Parser;
use log::{LevelFilter, info};
use spqr_shortest_path_index::{
    chopped_gfa::write_chopped_gfa, gfa_graph_extensions::LengthOnlyGfaNodeData,
    shrink::shrink_graph,
};

use crate::{
    io_util::{check_compressed_output_file, write_optionally_compressed_file},
    word_size::{parse_word_size, with_word_size},
};

#[derive(Parser)]
pub struct Cli {
    /// The log level: off, error, warn, info (a summary of each stage), debug or trace. The log
    /// goes to stderr.
    #[clap(long, default_value = "info")]
    pub(crate) log_level: LevelFilter,

    /// The GFA file containing the graph to shrink.
    #[clap(long)]
    graph_gfa_in: PathBuf,

    /// The output GFA (G′, chopped). A `.gz` extension compresses it.
    #[clap(long)]
    gfa_out: PathBuf,

    /// Split every G′ node into pieces of at most this many bp (vg needs <= 2047). 0 = no chopping.
    #[clap(long, default_value = "1024")]
    chop: u64,

    /// The integer size of the graph's node indices and lengths: 32 or 64. A graph too large for it
    /// is refused with the word size to use.
    #[clap(long, default_value = "32", value_parser = parse_word_size)]
    word_size: u8,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    check_compressed_output_file(&cli.gfa_out, "--gfa-out")?;
    with_word_size!(cli.word_size, run_with_word_size(cli))
}

fn run_with_word_size<IndexType: GraphIndexInteger>(cli: Cli) -> anyhow::Result<()> {
    let start = Instant::now();
    info!("Reading graph from GFA file {:?}", cli.graph_gfa_in);
    let graph: BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData> =
        crate::gfa_input::read_gfa(&cli.graph_gfa_in)?;
    spqr_shortest_path_index::graph_limits::check_positions(&graph)?;
    info!(
        "Graph has {} nodes and {} edges, read in {:.2?}",
        graph.node_count(),
        graph.edge_count(),
        start.elapsed()
    );

    let start = Instant::now();
    info!("Shrinking the graph");
    let plan = shrink_graph(&graph)
        .map_err(|reason| anyhow::anyhow!("the reduction refused the graph: {reason}"))?;
    drop(graph);
    let st = plan.stats.clone();
    info!(
        "[counters] shrink: nodes {} -> {} ({:.1}%), edges {} -> {} ({} duplicate links dropped), \
         pruned allele chains {} ({} nodes; {} dominated by a direct link), tip chains {} \
         ({} nodes), cyclic chains {}, max node length {}, in {:.2?}",
        st.original_nodes,
        st.shrunk_nodes,
        100.0 * st.shrunk_nodes as f64 / st.original_nodes.max(1) as f64,
        st.original_edges,
        st.shrunk_edges,
        st.duplicate_edges,
        st.allele_chains,
        st.allele_nodes,
        st.direct_chains,
        st.tip_chains,
        st.tip_nodes,
        st.cyclic_chains,
        st.max_node_len,
        start.elapsed(),
    );

    let start = Instant::now();
    info!("Writing G′ to {:?} (chop {})", cli.gfa_out, cli.chop);
    let export = write_optionally_compressed_file(&cli.gfa_out, |writer| {
        Ok(write_chopped_gfa(&plan.graph, cli.chop, writer)?)
    })?;
    info!(
        "Wrote {} segments and {} links in {:.2?}",
        export.segments,
        export.links,
        start.elapsed()
    );

    // One parseable summary line (`gs_*` keys: G′).
    println!(
        "export: input_nodes={} input_edges={} gs_nodes={} gs_edges={} pruned_allele_chains={} pruned_allele_nodes={} \
         tip_chains={} tip_nodes={} segments={} links={} chop={} max_gs_node_bp={} total_bp={}",
        st.original_nodes,
        st.original_edges,
        st.shrunk_nodes,
        st.shrunk_edges,
        st.allele_chains,
        st.allele_nodes,
        st.tip_chains,
        st.tip_nodes,
        export.segments,
        export.links,
        cli.chop,
        export.max_node_len,
        export.total_bp,
    );
    Ok(())
}
