# biopath

Exact shortest-path distances between positions of a bidirected pangenome graph.

biopath builds an index of a graph (GFA or GBZ) once, saves it as a `.bpi` file, and answers
distance queries from that file alone. Every answer equals a whole-graph Dijkstra search. Linux
only.

## Install

Needs Linux, [rustup](https://rustup.rs) and a C linker (`cc`). The pinned Rust version is
downloaded on the first build.

```bash
git clone https://github.com/AleksandrPolitov/biopath
cd biopath
cargo build --release          # → target/release/biopath
```

## Usage

```bash
# build the index once per graph (use --graph-gbz-in for a GBZ)
biopath index --graph-gfa-in graph.gfa --index-out graph.bpi

# answer a file of queries from the index
biopath query --index-in graph.bpi --query-in q.tsv --query-out out.tsv
```

`q.tsv` has one query per line, tab-separated: a source position `node orientation offset`, then
one or more target positions. `out.tsv` lists each position as `node offset orientation`, followed
by its distance in bases (`None` if unreachable).

### Example

With the small BRCA2 graph in the repository:

```bash
BIN=./target/release/biopath
GFA=tests/golden/graphs/cactus-BRCA2.gfa

# 1. build the index
$BIN index --graph-gfa-in $GFA --index-out brca2.bpi

# 2. write three queries: a source position, then one or more targets
#    (each position is: node orientation offset; tr turns the spaces into tabs)
tr -s ' ' '\t' > q.tsv <<'EOF'
1 + 0     100 + 5    1000 + 0
500 - 3   1 - 0
1 - 0     1 + 0
EOF

# 3. answer them
$BIN query --index-in brca2.bpi --query-in q.tsv --query-out out.tsv
cat out.tsv
```

```text
1	0	+	100	5	+	7276	1000	0	+	74649
500	3	-	1	0	-	39006
1	0	-	1	0	+	None
```

The first query has two targets, 7,276 and 74,649 bases away. The third is unreachable: no walk
leads from node `1` read backwards to node `1` read forwards.

`biopath <command> --help` lists all options. [USAGE.md](USAGE.md) is the full reference (input
requirements, file formats, all subcommands); [ARCHITECTURE.md](ARCHITECTURE.md) explains how the
index works.

## Performance

24 chromosome graphs of the HPRC v2.1 GRCh38 `.full` pangenome, 5,000 queries per graph, compared
with the distance index of vg 1.77 (all answers agree):

| | biopath | vg 1.77 |
|---|---|---|
| query time, p50 / p99 | 1.0 / 6.3 µs | 6.8 / 34.1 µs |
| index size, all 24 graphs | 2.7 GiB | 44.7 GiB |
| peak build memory, chr1 | 6.2 GiB | 48.4 GiB |

## License

MIT ([LICENSE](LICENSE)). The vendored [spqr-tree](vendor/README.md) crate (BSD-2-Clause) and the
test graphs from vg ([MIT](tests/golden/graphs/LICENSE-vg)) keep their own licenses.
