//! The build options from the environment: the one place the CLI maps its `SPQR_*` build variables
//! onto the engine's [`BuildOptions`], for `biopath index` and the in-memory build of
//! `biopath query`, and decides whether `index` shrinks the graph. The engine reads no build
//! parameter from the environment.
//!
//! | variable | default | option |
//! |---|---|---|
//! | `SPQR_SHRINK` | on | [`shrink_from_env`] (`index` only, also off with `--no-shrink`); `0` = no shrink |
//! | `SPQR_SNODE_SEGTREE_MIN` | `200` | [`Tiers::segtree_min`]; `0` = no S-node segment trees |
//! | `SPQR_PROJ_TABLE_MIN` | `1000` | [`Tiers::table_min`]; `0` = no pole-projection tables |
//! | `SPQR_SMALL_BLOCK_FAST` | on | [`SmallBlocks::enabled`]; `0` = every block gets a record |
//! | `SPQR_SMALL_BLOCK_MAX` | `8` | [`SmallBlocks::max_nodes`] |
//! | `SPQR_NODE_ROW_MAX_CUTS` | `8` | [`LiftOptions::node_row_max_cuts`]; `0` = no node cut rows |
//! | `SPQR_LIFT_BACKEND` | `auto` | [`LiftOptions::backend`]: `prefix`, `segtree`, else `auto` |
//!
//! The hub-label threshold [`Tiers::label_min`] is the `--rnode-label-threshold` flag. A variable
//! that is unset or does not parse keeps the default.

use std::str::FromStr;

use spqr_shortest_path_index::distance_index::{
    BuildOptions, LiftBackend, LiftOptions, SmallBlocks, Tiers,
};

/// The engine's build options: the hub-label threshold `label_threshold` (the
/// `--rnode-label-threshold` flag) and the variables of the table above.
pub fn build_options_from_env(label_threshold: usize) -> BuildOptions<'static> {
    let defaults = BuildOptions::default();
    BuildOptions {
        tiers: Tiers {
            label_min: label_threshold,
            segtree_min: match std::env::var("SPQR_SNODE_SEGTREE_MIN")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
            {
                Some(0) => usize::MAX,
                Some(n) => n,
                None => defaults.tiers.segtree_min,
            },
            table_min: parsed("SPQR_PROJ_TABLE_MIN").unwrap_or(defaults.tiers.table_min),
        },
        small_blocks: SmallBlocks {
            enabled: !is_zero("SPQR_SMALL_BLOCK_FAST"),
            max_nodes: parsed("SPQR_SMALL_BLOCK_MAX").unwrap_or(defaults.small_blocks.max_nodes),
        },
        lift: LiftOptions {
            backend: match std::env::var("SPQR_LIFT_BACKEND").as_deref() {
                Ok("prefix") => LiftBackend::Prefix,
                Ok("segtree") => LiftBackend::Segtree,
                _ => LiftBackend::Auto,
            },
            node_row_max_cuts: parsed("SPQR_NODE_ROW_MAX_CUTS")
                .unwrap_or(defaults.lift.node_row_max_cuts),
        },
        ..defaults
    }
}

/// Whether `biopath index` shrinks the graph (`shrink`): on unless `SPQR_SHRINK=0`. The
/// in-memory build of `biopath query` never shrinks.
pub fn shrink_from_env() -> bool {
    !is_zero("SPQR_SHRINK")
}

/// Whether the variable `name` is `0` (an on/off switch that is on by default).
fn is_zero(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "0")
}

/// The variable `name` parsed as a `T` (`None` when it is unset or does not parse).
fn parsed<T: FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.parse().ok()
}
