# Vendored code

## `spqr-tree/`

[spqr-tree](https://crates.io/crates/spqr-tree) 5.1.0 by Sebastian Schmidt, licensed BSD-2-Clause
(see [`spqr-tree/LICENSE`](spqr-tree/LICENSE); biopath itself is MIT). The root `Cargo.toml`
substitutes it for the crates.io release with `[patch.crates-io]`.

It differs from the crates.io release in one file, `src/decomposition/builder.rs`: the builder's
consistency checks are `debug_assert!`s, and one check that scans every node's incident edges runs
only in debug builds. In release builds those checks made building a large biconnected block
O(Σ K²). The decomposition the builder produces is unchanged. The full difference is
[`spqr-tree-5.1.0-release-asserts.patch`](spqr-tree-5.1.0-release-asserts.patch); applying it to the
crates.io source (`patch -p1 < spqr-tree-5.1.0-release-asserts.patch` inside the unpacked crate)
gives `spqr-tree/src/`.
