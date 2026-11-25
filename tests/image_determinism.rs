//! The same graph and options give a byte-identical `.bpi` at word size 64, across processes.
//!
//! Golden manifests pin the word-size-32 images; this covers 64, where generic records mix wide
//! index words with `u32` fields. A record with implicit padding writes uninitialised bytes, which
//! differ between processes (heap and stack contents, ASLR), so every image is built twice, each
//! by its own `biopath index` process, and compared. The configs force every oracle tier (labels,
//! S-node segment trees, projection tables, node rows) and pin each lift backend, so every record
//! type is written.

use std::{path::Path, process::Command};

/// A build configuration: a name, `SPQR_*` variables and extra `biopath index` flags.
struct Config {
    name: &'static str,
    env: &'static [(&'static str, &'static str)],
    flags: &'static [&'static str],
}

/// The forced configs of `tests/golden/configs.tsv`.
const CONFIGS: &[Config] = &[
    Config {
        name: "prod-forced",
        env: &[
            ("SPQR_SHRINK", "1"),
            ("SPQR_SNODE_SEGTREE_MIN", "4"),
            ("SPQR_PROJ_TABLE_MIN", "4"),
            ("SPQR_NODE_ROW_MAX_CUTS", "2"),
            ("SPQR_SMALL_BLOCK_MAX", "2"),
        ],
        flags: &["--rnode-label-threshold", "4"],
    },
    Config {
        name: "noshrink-forced-prefix",
        env: &[
            ("SPQR_SHRINK", "0"),
            ("SPQR_SNODE_SEGTREE_MIN", "4"),
            ("SPQR_PROJ_TABLE_MIN", "4"),
            ("SPQR_NODE_ROW_MAX_CUTS", "2"),
            ("SPQR_SMALL_BLOCK_FAST", "0"),
            ("SPQR_LIFT_BACKEND", "prefix"),
        ],
        flags: &["--rnode-label-threshold", "4"],
    },
    Config {
        name: "noshrink-forced-segtree",
        env: &[
            ("SPQR_SHRINK", "0"),
            ("SPQR_SNODE_SEGTREE_MIN", "4"),
            ("SPQR_PROJ_TABLE_MIN", "4"),
            ("SPQR_NODE_ROW_MAX_CUTS", "2"),
            ("SPQR_SMALL_BLOCK_FAST", "0"),
            ("SPQR_LIFT_BACKEND", "segtree"),
        ],
        flags: &["--rnode-label-threshold", "4"],
    },
];

/// Graphs with R-, S- and P-nodes, many blocks and flip-capable cuts.
const GRAPHS: &[&str] = &[
    "micb-kir3dl1",
    "GCA-41.clean.largest-block",
    "gfaz_example",
    "thread_determinism",
    "twoflip_hub",
    "tiny1",
];

/// The word size whose images the golden manifests do not pin.
const WORD_SIZE: u32 = 64;

/// Build `graph` under `config` at `word_size` in a fresh `biopath index` process; the image bytes.
fn build(graph: &str, config: &Config, word_size: u32) -> Vec<u8> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let gfa = root
        .join("tests/golden/graphs")
        .join(format!("{graph}.gfa"));
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "determinism-{}-{graph}-{}-{word_size}.bpi",
        std::process::id(),
        config.name
    ));
    let output = Command::new(env!("CARGO_BIN_EXE_biopath"))
        .arg("index")
        .arg("--graph-gfa-in")
        .arg(&gfa)
        .arg("--word-size")
        .arg(word_size.to_string())
        .arg("--index-out")
        .arg(&out)
        .args(config.flags)
        .env_clear()
        .env("RAYON_NUM_THREADS", "1")
        .envs(config.env.iter().copied())
        .output()
        .expect("cannot run biopath");
    assert!(
        output.status.success(),
        "biopath index failed on {graph} ({}, word size {word_size}):\n{}",
        config.name,
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&out).expect("cannot read the image");
    std::fs::remove_file(&out).expect("cannot remove the image");
    bytes
}

#[test]
fn images_are_byte_identical_across_processes_at_word_size_64() {
    for config in CONFIGS {
        for &graph in GRAPHS {
            let first = build(graph, config, WORD_SIZE);
            let second = build(graph, config, WORD_SIZE);
            assert_eq!(first.len(), second.len());
            if let Some(at) = first.iter().zip(&second).position(|(a, b)| a != b) {
                panic!(
                    "{graph} ({}, word size {WORD_SIZE}): two builds differ at byte {at} of {}",
                    config.name,
                    first.len()
                );
            }
        }
    }
}
