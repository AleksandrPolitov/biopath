# biopath

**Exact shortest-path distances between positions of a bidirected pangenome graph.**

biopath builds an index of a sequence graph (GFA or GBZ) once and saves it as a `.bpi` file. From
that file alone, without the graph, it answers *"how many bases lie on the shortest walk from
position A to position B?"* in about a microsecond. Every answer is exact: the same as a
whole-graph Dijkstra search, or `None` when no walk exists.

It is a research implementation that shows how an index built on the SPQR decomposition of a graph
works and how fast it is. It runs on 64-bit Linux.

## Performance

On the 24 chromosome graphs (chr1–22, X, Y) of the HPRC release 2.1 GRCh38 `.full` pangenome,
against the minimum-distance index of vg 1.77, with 5,000 queries per graph and the index loaded in
memory. All 120,000 answers agree.

| | biopath | vg 1.77 |
|---|---|---|
| query latency, p50 / p99 (median over the graphs) | 1.0 / 6.3 µs | 6.8 / 34.1 µs |
| index size, all 24 graphs | 2.7 GiB | 44.7 GiB |
| peak build memory, chr1 (the largest) | 6.2 GiB | 48.4 GiB |
| build time, all 24 graphs | 52 min | 93 min |

biopath's build time includes reading the GFA; vg's is the `vg index` step alone.

## Install

You need 64-bit Linux, [Rust](https://rustup.rs) installed with rustup, and a C linker (`cc`,
which every Rust build uses). Build from source:

```bash
git clone https://github.com/AleksandrPolitov/biopath
cd biopath
cargo build --release          # → target/release/biopath
```

The repository pins its compiler (Rust 1.96.0, `rust-toolchain.toml`); rustup downloads it on the
first build, whatever Rust you have installed. The result is the single binary
`target/release/biopath`: run it by that path, or copy it to a directory on your `PATH` to call it
as `biopath`, as the examples below do.

## Quick start

The repository contains a small real graph (BRCA2 from vg's test data, 1,134 nodes). From a
checkout:

```bash
BIN=./target/release/biopath
GFA=tests/golden/graphs/cactus-BRCA2.gfa

# 1. build the index
$BIN index --graph-gfa-in $GFA --index-out brca2.bpi

# 2. write a query file: a source position, then one or more target positions
printf '1\t+\t0\t100\t+\t5\t1000\t+\t0\n500\t-\t3\t1\t-\t0\n1\t-\t0\t1\t+\t0\n' > q.tsv

# 3. answer the queries from the index alone
$BIN query --index-in brca2.bpi --query-in q.tsv --query-out out.tsv
cat out.tsv
```

```text
1	0	+	100	5	+	7276	1000	0	+	74649
500	3	-	1	0	-	39006
1	0	-	1	0	+	None
```

The first query has two targets. The third is unreachable: no walk leads from `1-` back to `1+`.

## Usage

**1. Build an index** once per graph:

```bash
biopath index --graph-gfa-in graph.gfa --index-out graph.bpi
biopath index --graph-gbz-in graph.gbz --index-out graph.bpi    # a vg / Giraffe GBZ
```

The GFA must be GFA 1 with blunt links (`0M` overlaps), as vg, Minigraph-Cactus and PGGB write it.
A GBZ must be format version 1 (the HPRC v2.1 GBZs are); for a newer one biopath prints the `vg`
command that converts it. If biopath says a graph's distances may not fit, add `--word-size 64`.

**2. Write the queries**, one per line, tab-separated: a source position, then one or more target
positions. A position is three columns: the node (GFA segment name or GBZ node id), the orientation
`+` or `-`, and the offset in bases from the start of the node in that orientation.

```text
1	+	0	100	+	5
```

To get test queries, `biopath generate-random-queries --graph-gfa-in graph.gfa --query-out q.tsv
--walk-length 20` writes 1,000 random queries whose targets are reachable.

**3. Answer them:**

```bash
biopath query --index-in graph.bpi --query-in q.tsv --query-out out.tsv
```

`out.tsv` repeats every position as `node offset orientation` (offset first) and puts the distance
in bases, or `None`, after each target. The log (stderr) reports the query time percentiles. To
check the answers against plain Dijkstra, add `--graph-gfa-in graph.gfa
--compare-whole-graph-dijkstra` (slow on big graphs, so use a few hundred queries).

Three more subcommands: `index-inspect` prints what a `.bpi` file contains, `statistics` reports
graph and SPQR-tree sizes, and `export-shrunk-gfa` writes the reduced graph the index is built on
as a GFA that vg can load. `biopath <subcommand> --help` lists every option, and
[USAGE.md](USAGE.md) is the full reference: input limits, file formats, timing output and
environment variables.

## How it works

`biopath index` first shrinks the graph without changing any distance (unbranched chains merged,
dominated alleles and dead-end tips removed), keeping a map so queries still use the original
positions. It then decomposes the graph into its **block-cut tree** (2-connected blocks joined at
cut nodes) and an **SPQR tree** per block, and stores small orientation-aware distance tables along
both trees. Large rigid parts get hub labels, long cycles get segment trees. A query combines a few
of these tables and never searches the whole graph. [ARCHITECTURE.md](ARCHITECTURE.md) describes
the design in detail.

## Testing

```bash
cargo test --workspace                           # unit, fixture and fuzz tests against Dijkstra
cargo build --profile fast && BIN=target/fast/biopath scripts/golden.sh check
```

The golden check builds every graph in `tests/golden/graphs/` under several configurations and
compares index bytes and answers with a recorded manifest ([tests/golden/README.md](tests/golden/README.md)).

## License

MIT, see [LICENSE](LICENSE). `vendor/spqr-tree` is a lightly patched copy of the
[spqr-tree](https://crates.io/crates/spqr-tree) crate, which keeps its own BSD 2-Clause license
([vendor/spqr-tree/LICENSE](vendor/spqr-tree/LICENSE), see [vendor/README.md](vendor/README.md));
the test graphs taken from vg carry vg's MIT license
([tests/golden/graphs/LICENSE-vg](tests/golden/graphs/LICENSE-vg)).
