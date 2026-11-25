//! `biopath index-inspect`: print a `.bpi` index image's header and table of contents.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use log::LevelFilter;
use spqr_shortest_path_index::{
    image::{Image, SectionId},
    sections::{knob, section_name},
};

#[derive(Parser)]
pub struct Cli {
    /// The log level: off, error, warn, info (a summary of each stage), debug or trace. The log
    /// goes to stderr.
    #[clap(long, default_value = "info")]
    pub(crate) log_level: LevelFilter,

    /// The `.bpi` index image to inspect.
    #[clap(long)]
    index_in: PathBuf,
}

fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    let image = Image::open(&cli.index_in)
        .with_context(|| format!("Failed to open index image {:?}", cli.index_in))?;
    let header = image.header();
    println!("file:            {:?}", cli.index_in);
    println!("format version:  {}", header.format_version);
    println!("word size:       {} bits", header.word_size * 8);
    println!(
        "image bytes:     {} ({})",
        header.used_bytes,
        fmt_bytes(header.used_bytes)
    );
    println!("graph nodes:     {}", header.graph_node_count);
    println!("graph edges:     {}", header.graph_edge_count);
    println!("graph fingerprint: {:#018x}", header.graph_fingerprint);
    let knobs = header.build_knobs;
    println!("build knobs:");
    println!("  label threshold:  {}", knobs[knob::LABEL_THRESHOLD]);
    println!("  small block max:  {}", knobs[knob::SMALL_BLOCK_MAX]);
    let seg = knobs[knob::SNODE_SEGTREE_MIN];
    if seg == u64::MAX {
        println!("  S-node segtree:   off");
    } else {
        println!("  S-node segtree:   >= {seg} states");
    }
    match knobs[knob::NODE_ROW_MAX_CUTS] {
        0 => println!("  node cut rows:    off"),
        max => println!("  node cut rows:    blocks with <= {max} cuts"),
    }
    match knobs[knob::PROJ_TABLE_MIN] {
        0 => println!("  projection tables: off"),
        min => println!("  projection tables: inner skeletons >= {min} states"),
    }
    let flags = knobs[knob::FLAGS];
    let names = [
        (knob::FLAG_SMALL_BLOCK_FAST, "small-block-fast"),
        (knob::FLAG_SHRUNK, "shrunk"),
    ];
    let set: Vec<&str> = names
        .iter()
        .filter(|(bit, _)| flags & bit != 0)
        .map(|(_, n)| *n)
        .collect();
    println!("  flags:            {:#x} [{}]", flags, set.join(", "));
    let mut toc = image.toc();
    toc.sort_by_key(|e| e.off);
    println!("sections ({}):", toc.len());
    println!(
        "  {:<4} {:<30} {:>14} {:>14} {:>12} {:>6}",
        "id", "name", "offset", "bytes", "elements", "elem"
    );
    let mut total = 0u64;
    for e in &toc {
        let bytes = e.len * e.elem_size as u64;
        total += bytes;
        println!(
            "  {:<4} {:<30} {:>14} {:>14} {:>12} {:>6}",
            e.id,
            section_name(SectionId(e.id)),
            e.off,
            bytes,
            e.len,
            e.elem_size
        );
    }
    println!(
        "section bytes:   {} ({}); block records + padding: {}",
        total,
        fmt_bytes(total),
        fmt_bytes(header.used_bytes.saturating_sub(total)),
    );
    Ok(())
}
