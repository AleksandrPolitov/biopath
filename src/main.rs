use clap::{Parser, Subcommand};

mod build_options;
mod export_shrunk;
mod gbz_input;
mod generate_random_queries;
mod gfa_input;
mod index;
mod inspect;
mod io_util;
mod query;
mod query_file;
mod statistics;
mod word_size;

/// Exact shortest-path distances between positions of a bidirected pangenome graph (GFA or GBZ),
/// answered by an SPQR-tree index.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[clap(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    /// Build the index of a graph (GFA or GBZ) and write it as a `.bpi` image.
    Index(index::Cli),
    /// Print a `.bpi` image's header, build knobs and section table.
    IndexInspect(inspect::Cli),
    /// Answer a query file with the dp engine (from a `.bpi` index, or built in memory) or with
    /// plain whole-graph Dijkstra, and report the query times.
    Query(query::Cli),
    /// Size statistics of a graph and its SPQR decomposition: counts, size lists (also as JSON or
    /// TOML) and a census of the SPQR skeletons.
    Statistics(statistics::Cli),
    /// Write G′, the shrunk graph `index` builds on (unless `--no-shrink`), as a chopped GFA that
    /// vg can load.
    ExportShrunkGfa(export_shrunk::Cli),
    /// Write random queries for a graph: reachable ones from random walks, or independent random
    /// pairs.
    GenerateRandomQueries(generate_random_queries::Cli),
}

fn main() -> anyhow::Result<()> {
    reset_sigpipe();
    let cli = Cli::parse();
    initialise_logging(&cli);

    match cli.command {
        CliCommand::Index(cli) => index::run(cli)?,
        CliCommand::IndexInspect(cli) => inspect::run(cli)?,
        CliCommand::Query(cli) => query::run(cli)?,
        CliCommand::Statistics(cli) => statistics::run(cli)?,
        CliCommand::ExportShrunkGfa(cli) => export_shrunk::run(cli)?,
        CliCommand::GenerateRandomQueries(cli) => generate_random_queries::run(cli)?,
    }

    Ok(())
}

/// Restore the default SIGPIPE action, which the Rust runtime sets to "ignore": a closed pipe
/// (`biopath index-inspect … | head`) then ends the process quietly, as for any Unix tool, instead
/// of `println!` panicking on the write error.
#[cfg(target_os = "linux")]
fn reset_sigpipe() {
    const SIGPIPE: i32 = 13;
    const SIG_DFL: usize = 0;
    unsafe extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    // SAFETY: installing the default disposition for SIGPIPE has no preconditions; it runs before
    // any other thread exists.
    unsafe {
        signal(SIGPIPE, SIG_DFL);
    }
}

#[cfg(not(target_os = "linux"))]
fn reset_sigpipe() {}

/// The log goes to stderr, so stdout carries only a subcommand's own output (`index-inspect`,
/// `statistics`).
fn initialise_logging(cli: &Cli) {
    let log_level = match &cli.command {
        CliCommand::Index(cli) => cli.log_level,
        CliCommand::IndexInspect(cli) => cli.log_level,
        CliCommand::Query(cli) => cli.log_level,
        CliCommand::Statistics(cli) => cli.log_level,
        CliCommand::ExportShrunkGfa(cli) => cli.log_level,
        CliCommand::GenerateRandomQueries(cli) => cli.log_level,
    };

    use simplelog::*;
    CombinedLogger::init(vec![TermLogger::new(
        log_level,
        Config::default(),
        TerminalMode::Stderr,
        ColorChoice::Auto,
    )])
    .unwrap();
}
