//! The one runtime switch the engine reads from the environment, a performance switch of the build
//! arena: `SPQR_ARENA_HUGEPAGE=0` asks for no transparent huge pages ([`arena_hugepages`]). It
//! changes neither an index nor an answer. Build parameters are never read from the environment:
//! they are [`BuildOptions`](crate::distance_index::BuildOptions) fields, which the `biopath` CLI
//! fills from its `SPQR_*` build variables.

use std::sync::OnceLock;

/// Whether the build arena asks for transparent huge pages (on unless `SPQR_ARENA_HUGEPAGE=0`).
/// Read once, on first use.
#[inline]
pub fn arena_hugepages() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("SPQR_ARENA_HUGEPAGE").is_ok_and(|v| v == "0"))
}
