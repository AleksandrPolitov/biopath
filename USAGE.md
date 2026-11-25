# biopath reference

The complete reference for the `biopath` command line: what a position and a distance are, which
graphs it accepts, every subcommand and flag, the file formats, the timing output and the
environment variables. [README.md](README.md) has the overview and a quick start;
[ARCHITECTURE.md](ARCHITECTURE.md) explains how the index works. Every subcommand also prints its
options with `biopath <subcommand> --help`.

## Positions and distances

A position is `(node, orientation, offset)`: a node (GFA segment name, or GBZ node id), `+` or
`-`, and an offset `0 ≤ offset ≤ length(node)`. The offset is a cut point between bases, counted
from the start of the node *in that orientation*: `(A, +, x)` is the point after the first `x`
bases of `A`, and `(A, -, x)` is the same point as `(A, +, length(A) − x)` read in reverse.

The distance from `(A, o, x)` to `(B, p, y)` is the number of bases on the shortest walk that
leaves `A` in orientation `o` and enters `B` in orientation `p`:
`(length(A) − x) + Σ(lengths of the nodes in between) + y`. On the same oriented node with `y ≥ x`
it is `y − x` (`0` for the same point). Otherwise the walk must leave the node and come back, which
is `None` when no such walk exists.

## Input graphs and limits

- **GFA 1** (versions 1.0–1.2). Only `S` and `L` lines are used; `P` and `W` lines are skipped,
  `C` lines are skipped with a warning.
- **Blunt links only.** Every `L` line's overlap must be `0M` (or `*` / missing). Graphs written by
  vg, Minigraph-Cactus and PGGB are blunt. Assembly graphs whose links overlap (`55M`) are refused
  with the offending line; so are `J` lines and GFA 2.
- **Segment lengths** come from the sequence, or from the `LN:i:` tag when the sequence is `*`.
  Segment names must be unique.
- **GBZ** (vg / Giraffe's graph + haplotype index, `--graph-gbz-in`) in format version 1, which the
  HPRC v2.1 GBZs use. vg 1.77 writes version 2 by default; `vg gbwt -Z in.gbz --gbz-format
  --gbz-version 1 -g out.gbz` converts one (biopath prints this command when it refuses a file).
- **Word size.** Every index structure uses one unsigned integer type, `--word-size 32` (the
  default) or `64`. The build refuses a graph whose distances may not fit and says which word size
  to use: per connected component of the indexed graph, `4 × (2 × total node length + longest node)`
  must stay below the word's maximum, about 0.5 Gbp of nodes per component at 32 bits. With the
  default shrink, the bound applies to the shrunk graph, so whole-chromosome human graphs index at
  32 bits; the in-memory build of `query` and `index --no-shrink` may need `--word-size 64`.
- **At most 4000 cut nodes per block.** A larger block stops the build; this is a hard limit of the
  index.
- **The `.bpi` file** is native-endian and records its word size and format version (currently
  19); a binary refuses any other version, so rebuild indexes after upgrading. Load-time checks
  refuse truncated, foreign or wrong-version files, but a `.bpi` is trusted input like a vg `.dist`:
  do not load files from untrusted sources.
- **Linux only** (64-bit). The index image uses Linux `mmap`/`madvise`; other systems refuse to
  compile.
- **Memory.** `query --index-in` maps the whole `.bpi` into memory and prefaults it before the
  first query, so the process needs about the file's size in RAM and every query runs warm.

## Subcommands

| subcommand | purpose |
|---|---|
| `index` | build the index of a graph (GFA or GBZ) and write it as a `.bpi` file |
| `query` | answer a query file from an index (or an index built in memory), or with plain whole-graph Dijkstra |
| `generate-random-queries` | write a random query file for a graph |
| `index-inspect` | print a `.bpi` file's header, build options and section table |
| `statistics` | size statistics of a graph and its SPQR decomposition |
| `export-shrunk-gfa` | write the shrunk graph the index is built on as a GFA that vg can load |

Every subcommand takes `--log-level <off|error|warn|info|debug|trace>` (default `info`; the log
goes to stderr) and `--help`; `biopath --version` prints the version.

### `index`

```bash
biopath index --graph-gfa-in <GFA> --index-out <BPI> [flags]
```

| flag | default | meaning |
|---|---|---|
| `--graph-gfa-in <PATH>` | — | input GFA (plain or gzipped) |
| `--graph-gbz-in <PATH>` | — | input GBZ instead of a GFA (see [Input graphs](#input-graphs-and-limits)) |
| `--index-out <PATH>` | — | the `.bpi` file (uncompressed; written to a temporary file and renamed into place) |
| `--rnode-label-threshold <N>` | `200` | hub labels for every skeleton with at least `N` directed nodes; lower = bigger index, faster queries; a huge value disables labels |
| `--no-shrink` | off | index the graph as it is, without the exact pre-reduction (same answers, bigger index); `SPQR_SHRINK=0` does the same |
| `--word-size <32\|64>` | `32` | integer width of the index (see [limits](#input-graphs-and-limits)) |
| `--timing-out <PATH>` | — | write the timings as JSON |

The output paths are checked before the graph is read. The build assembles the image in memory and
writes it at the end; the same graph and options always give a byte-identical file. Build options
that `index-inspect` lists are recorded in the header.

### `query`

```bash
biopath query --index-in <BPI> --query-in <TSV> --query-out <TSV> [flags]
```

| flag | default | meaning |
|---|---|---|
| `--query-in <PATH>` | — | query file (see [format](#query-and-result-files)) |
| `--query-out <PATH>` | — | result file (a `.gz` extension compresses it) |
| `--engine <dp\|dijkstra>` | `dp` | `dp`: the index; `dijkstra`: plain whole-graph Dijkstra on `--graph-gfa-in` (the baseline, always 64-bit) |
| `--index-in <PATH>` | — | a `.bpi` from `biopath index`, memory-mapped and prefaulted whole at load; without it, `dp` builds the index in memory from the graph first (without shrinking) |
| `--graph-gfa-in <PATH>` | — | the graph: for `dijkstra`, for the in-memory build, and for `--compare-whole-graph-dijkstra` |
| `--graph-gbz-in <PATH>` | — | a GBZ instead of `--graph-gfa-in` (`dp` only) |
| `--compare-whole-graph-dijkstra` | off | also answer every query with whole-graph Dijkstra and report both engines' mean query times and every mismatch (`dp` only; needs the graph) |
| `--word-size <32\|64>` | `32` | integer width of the in-memory build (an index records its own) |
| `--rnode-label-threshold <N>` | `200` | as for `index`, for the in-memory build (an index records its own) |
| `--timing-out <PATH>` | — | write the timings as JSON, with every per-query time |

`--engine dijkstra` refuses the flags that only apply to `dp`. With `--index-in`, a `--word-size` or
`--rnode-label-threshold` that differs from the index's only warns.

With `--index-in`, a graph given for the cross-check is checked against the fingerprint stored in
the index (check an index built from a GBZ against that GBZ: a GFA of the same graph may list its
edges in another order). The log reports `Finished executing N queries in …`, `Query time
distribution (µs): p50=… p90=… p99=… max=…` and, with the cross-check, `All query results match
whole-graph Dijkstra` or the number of differing answers (each also printed). Whole-graph Dijkstra
is slow on big graphs (up to hundreds of ms per query), so cross-check a modest number of queries.

`DistanceIndex` (the library type behind `query`) is not shared between threads: a multi-threaded
caller opens one per thread; memory-mapped instances share the page cache.

### `generate-random-queries`

```bash
biopath generate-random-queries --graph-gfa-in <GFA> --query-out <TSV> [--amount 1000] \
    [--random-seed 0] [--walk-length K]
```

- **Independent random** (default): source and target are each a uniformly random position. On
  real graphs these are almost always unreachable (`None`).
- **Reachable** (`--walk-length K`): pick a random node and a side with an outgoing link, walk up to
  `K` uniformly random links (stopping at a dead end); the source is the walk's start and the target
  its end, so a walk between them exists.

The same seed gives the same queries.

### `index-inspect`

```bash
biopath index-inspect --index-in graph.bpi
```

Prints the format version, word size, graph size and fingerprint, the recorded build options
(label threshold, small-block bound, tier thresholds, node-row cap, shrink and small-block flags)
and every section with its offset and size.

### `statistics`

```bash
biopath statistics --graph-gfa-in G.gfa [--shrink] [--statistics-json-out s.json] [--statistics-toml-out s.toml]
```

Size statistics of the graph and its SPQR decomposition (node, edge, block and SPQR-node counts and
size lists) and a census of the SPQR skeletons. `--shrink` describes the shrunk graph the index is
built on instead of the input.

### `export-shrunk-gfa`

```bash
biopath export-shrunk-gfa --graph-gfa-in G.gfa --gfa-out G.shrunk.gfa [--chop 1024]
```

Writes the shrunk graph that `biopath index` builds on as a GFA that vg can load: every node is a
run of `N`s of its length, split into pieces of at most `--chop` bp (vg needs at most 2047; `0`
keeps whole nodes). It lets vg's own distance index be built on the same reduced graph.

## Query and result files

Query files are tab-separated, one query per line: a source position, then one or more target
positions. Blank lines and lines starting with `#` are skipped.

```text
src_node  src_orient  src_offset   tgt_node  tgt_orient  tgt_offset  [ tgt_node  tgt_orient  tgt_offset ... ]
```

A malformed line, an unknown node or an offset beyond the node's end is an error naming the line.

Result files repeat every position as `node offset orientation` (offset before orientation) and
append the distance after each target, an integer or `None`:

```text
src_node  src_offset  src_orient   tgt_node  tgt_offset  tgt_orient  distance  [ ... ]
```

## Timing output

`--timing-out <json>` writes pretty JSON.

- `query`: `graph_reading_time`, `node_name_index_building_time`, `spqr_reading_time` (the
  in-memory decomposition), `index_reading_time` (the `--index-in` load), `query_reading_time`,
  `dijkstra_initialisation_time` (the in-memory build, or the Dijkstra set-up),
  `single_query_execution_times` (every query), `total_query_execution_time`,
  `query_writing_time`, `compute_time`, `io_time`, `total_time`, in seconds.
- `index`: `graph_reading_time`, `shrink_time`, `spqr_building_time`, `engine_building_time`,
  `index_writing_time`, `index_bytes`, `total_time`; the log ends with `Indexing completed
  successfully in …` and `Index timings in seconds: …`.

## Environment variables

**Build options**, read when an index is built (`index`, or the in-memory build of `query`). The
CLI maps them onto the engine's build options in one place, `src/build_options.rs`; the engine
library reads no build parameter from the environment. Their effect is part of the image;
`index-inspect` lists the ones recorded in the header.

| variable | default | effect |
|---|---|---|
| `SPQR_SHRINK` | **on** | `index` only: shrink the graph exactly before the decomposition; `=0` (or `--no-shrink`) indexes the graph as it is |
| `SPQR_SMALL_BLOCK_FAST` | **on** | `=0` disables: a block of at most `SPQR_SMALL_BLOCK_MAX` nodes gets no precomputed record and is searched directly at query time |
| `SPQR_SMALL_BLOCK_MAX` | `8` | node threshold of the small-block path |
| `SPQR_SNODE_SEGTREE_MIN` | `200` | cycle (S-node) skeletons with at least this many directed states get a segment tree (`0` = none) |
| `SPQR_PROJ_TABLE_MIN` | `1000` | inner skeletons with at least this many directed states get a pole-projection table (`0` = none) |
| `SPQR_NODE_ROW_MAX_CUTS` | `8` | nodes of blocks with at most this many cut nodes store their distances to the cuts (`0` = none) |
| `SPQR_LIFT_BACKEND` | `auto` | cross-block distances: `prefix` (closed-form prefix sums), `segtree` (chain segment trees), `auto` (prefix when at most 2% of the steps are exceptions) |
| `RAYON_NUM_THREADS` | all cores | threads of the parallel build steps (the image does not depend on it) |

The hub-label threshold is the `--rnode-label-threshold` flag.

One runtime switch, read by the engine (`engine/src/env.rs`), changes neither answers nor the
index: `SPQR_ARENA_HUGEPAGE=0` asks for no transparent huge pages for the build's memory.
