# spqr-shortest-path-index

The library behind [biopath](../README.md): an exact shortest-path distance index for bidirected
sequence graphs (GFA), built on the block-cut tree and the SPQR trees of the graph's biconnected
blocks. `DistanceIndex` builds the index, writes it as a flat `.bpi` image, memory-maps it back and
answers distance queries between positions `(node, orientation, offset)`.

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — the design: build pipeline, stored structures, query
  path, image format, module map and testing.
- [`../README.md`](../README.md) and [`../USAGE.md`](../USAGE.md) — the `biopath` CLI that builds and
  queries the index.
- `src/lib.rs` — the crate docs (`cargo doc --no-deps -p spqr-shortest-path-index`).
