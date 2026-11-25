# Architecture

biopath answers **exact shortest-path distances between positions of a bidirected sequence graph**
from a precomputed index. This document explains how the index is built, what it stores and why,
how a query reads it, and how correctness is checked. It is written for a developer who knows
graphs and shortest paths but not this code base. The CLI is documented in [USAGE.md](USAGE.md);
every module carries its own `//!` docs, which go deeper than this overview.

The workspace has two crates:

- `engine/` — the library `spqr-shortest-path-index`: the index itself (crate root
  `engine/src/lib.rs`, a flat module tree).
- the root package `biopath` (`src/`) — the CLI: graph input (GFA, GBZ), the build options, the
  query loop, timing reports, and `export-shrunk-gfa` (the shrunk graph as a GFA for vg).

`vendor/spqr-tree` is a copy of the spqr-tree crate (the SPQR-tree data structure) whose builder
runs its consistency checks in debug builds only ([vendor/README.md](vendor/README.md)). biopath
supports 64-bit Linux only (`engine/src/lib.rs` refuses to compile elsewhere).

## 1. The problem

**Graph.** A GFA graph with blunt links (`0M` overlaps). A node (segment) has a sequence length
and two *sides*; a link joins one side of a node to one side of another (or the same) node. A walk
passes through a node in one of two **orientations** (`+` forward, `-` reverse) and must leave it
through the side opposite the one it entered. Orientation is therefore part of the state: a
**directed node** `v^o` is a node with an orientation, and the graph is handled as a directed graph
over the `2n` directed nodes, in which every arc `u^a -> v^b` has its reverse complement
`v^¬b -> u^¬a`.

**Query.** A position is `(node, orientation, offset)`, the offset counting bases from the start of
the directed node. The distance from `(A, o, x)` to `(B, p, y)` is the length of a shortest walk:
the rest of `A` after `x`, the full length of every node in between, then `y` bases of `B`. A target
at or after the source on the same directed node is reached directly. The answer is exact, or
`None` when no walk exists.

**Cost model.** Internally every distance is *start-to-start*: an arc `x -> y` costs `len(x)` (the
node it leaves), so `d(a -> b)` between directed nodes is the walk from the start of `a` to the start
of `b`. This composes associatively, and it gives the **reversal identity**

```text
d(x -> y) = d(¬y -> ¬x) + len(x) - len(y)
```

which several structures use to store one direction and derive the other. Query offsets are added
at the end.

**Algebra.** Distances compose in the min-plus semiring (`⊕ = min`, `⊗ = +`, `∞` = no walk;
`engine/src/min_plus/`). Because a walk enters and leaves every node in one of two orientations,
the distance between two nodes is a small orientation-indexed matrix: a `2x2` between two cut
nodes, a `4x4` over the states `(s,+), (s,-), (t,+), (t,-)` of a two-pole region. Min-plus has no
inverse (no prefix differences) and `⊗` is not idempotent (no sparse tables), which shapes the
path-product structures below. A distance is one word of the index's integer type (`--word-size`
32 or 64, default 32) with `MAX` meaning `∞`; `graph_limits.rs` refuses, before a build, a graph
whose distances may not fit (per connected component, `4 × (2 × total length + longest node)` must
stay below `MAX`, so a sum of up to four distances cannot wrap).

## 2. The pipeline

```text
GFA / GBZ                                     gfa_reader, src/gfa_input.rs, src/gbz_input.rs
   │
   ▼
shrink ─────────────► G′ and the alias                     shrink/
   │
   ▼
SPQR decomposition ─► block-cut tree, rooted SPQR trees    decomposition, block_spqr_tree
   │
   ▼
per-block records ──► inside/outside DP, pole lift,        engine_build/record, block_record,
   │                  oracle tiers                         inside_dp, outside_dp, pole_lift
   ▼
block-cut lift ─────► flip closures, M_B / M_aug,          block_cut_lift, prefix_lift,
   │                  path products, node rows             chain_segtree, chain_nav, node_rows
   ▼
image (.bpi) ───────► header, TOC, sections, records       image, sections, topology, graph_view
```

The caller (`biopath index`, `src/index/mod.rs`) reads the graph (`gfa_reader.rs`: a strict GFA 1
reader that takes lengths from `LN` tags for `*` sequences and refuses overlapping links, duplicate
names and GFA 2), checks it against the word size (`graph_limits.rs`), shrinks it
(`shrink::shrink_graph`) and decomposes it (`decomposition::build_spqr_decomposition`).
`DistanceIndex::build` (`distance_index.rs`) then runs the stages of `engine_build.rs` in a fixed
order: the order of the image writes fixes the section offsets, so it is part of the format. Every
build parameter arrives in `BuildOptions` (tier thresholds, small-block bound, lift options, the
shrink sink); the engine reads no build parameter from the environment. The CLI fills
the options from its flags and `SPQR_*` variables in one place, `src/build_options.rs`.
`build_report.rs` owns all build logging and timing; the stages call its API and never format output
themselves. It logs through the `log` crate, a short summary at `info` (the library is silent unless
the caller installs a logger).

### 2.1 Shrink (`shrink/`)

An exact pre-reduction, on by default in `biopath index` (`--no-shrink` or `SPQR_SHRINK=0` skips
it). Two nodes whose facing sides carry exactly one link, to each other, are always walked straight
through; a maximal run of them is a **chain** and behaves like one node. A chain whose two outer
sides each carry one link is an **allele** between two partner sides, and only the shortest allele
between them (or a direct link) can lie on a shortest walk. A **tip** (a chain with one outer side
unlinked) is never walked through. A worklist prunes dominated alleles and tips one at a time and
re-merges what the prune leaves walked straight through (`reduce.rs`). The surviving chains become
the nodes of the smaller graph **G′**, which has the same distances.

Every original position must still be queryable, so the index keeps an **alias** from original nodes
to G′ (`records.rs`, `layout.rs`, `view.rs`): each original node lies in a G′ node or in a pruned
chain, and a pruned chain records its *ports* (where a walk is forced to leave or enter it) and a
*level* (one more than the largest level of a pruned chain whose port lies on it). A query on
original positions resolves both positions; a position in a pruned chain *hops* out through its
port, lower level first, until both positions sit on G′ (`distance.rs`), where the rest of the
engine answers. Levels are stored in a byte; when both positions lie in chains whose levels
saturated (≥ 255), each side's hop sequence is walked once and the cheapest meeting point (inside a
shared chain, or one engine query on G′) is taken, which is exact and linear in the hops. *Why:*
every later structure is sized by G′, and pangenome graphs, which are mostly bubbles, lose most of
their nodes to the reduction.

### 2.2 SPQR decomposition (`decomposition.rs`, `block_spqr_tree.rs`, `rooted_tree.rs`)

The graph (G′, or the input with `--no-shrink`) is decomposed in process with `spqr-rust`:
connected components, the **block-cut tree** (biconnected blocks joined at cut nodes), and the
**SPQR tree** of every block (series S-nodes = cycles, parallel P-nodes = bonds, rigid R-nodes =
triconnected skeletons). Each SPQR node has a *skeleton* in which every child subtree is collapsed
to one virtual edge between two **poles**. `BlockSpqrTree` roots each block's tree and records, per
non-root node, its **reference poles** (the poles of the virtual edge to its parent); `RootedTree`
gives depths and binary-lifting LCA.

*Why:* two blocks share at most one node, so a walk between blocks passes through the cut nodes on
the block-cut path, in order; and an SPQR subtree meets the rest of its block only at its two poles,
so everything inside it is summarised by a `4x4` over the pole states.

**Small blocks.** A block of at most 8 graph nodes (`SmallBlocks`, `SPQR_SMALL_BLOCK_MAX`) gets no
record at all: a query searches it with a block-restricted Dijkstra (`block_dijkstra.rs`), which is
exact because a shortest walk between two nodes of a block never leaves it. Most blocks of a
pangenome graph are small, so this saves most of the per-block build.

### 2.3 Per-block records (`engine_build/record.rs`, `block_record.rs`, `block_engine/`)

Every block that is not small gets a **record**: one contiguous byte range in the image holding its
rooted SPQR tree and every structure a query reads inside the block. The DP pipeline that fills it:

1. **Inside DP** (`inside_dp.rs`), bottom-up: for every SPQR node, the `4x4` start-to-start
   distances between its reference-pole states *through its own subtree*, computed on the node's
   orientation-doubled skeleton with every child virtual edge weighted by the child's matrix. A
   P-node is a closed form, no search.
2. **Outside DP** (`outside_dp.rs`), top-down: `D_out`, the `4x4` among the same pole states over
   walks that *avoid* the subtree (through the rest of the block, and via flip closures the rest of
   the graph). It includes pole self-flips (`s^+ -> s^-`), which exist only in a bidirected graph.
   S-node parents use an `O(k)` cycle closed form (`s_node_outside.rs`); big parents use hub labels.
3. **Pole lift** (`pole_lift.rs`): per non-root node, the `4x4` transition from its poles to its
   parent's poles. A query climbs an endpoint's pole distances up the tree one level at a time;
   real climbs are short.
4. **Skeletons and oracles** (`block_engine/precompute.rs`, `block_engine/tiers.rs`).

A query inside a block searches two skeletons of an SPQR node: its **inner skeleton** (to *project*
an endpoint onto the node's reference poles) and its **LCA-role skeleton** (the inner one plus
`D_out`, when the node is the LCA of the endpoints). A small skeleton is searched by Dijkstra over
a CSR (`skeleton.rs`, with an epoch-reset workspace so a query allocates nothing). A big one gets a
precomputed **oracle**, by tier (thresholds in directed states, `Tiers`):

| tier | default | what it stores | why |
|---|---|---|---|
| S-node segment tree (`s_node_segtree/`) | ≥ 200 | the cycle as a ring of leaf networks: runs of regular leaves as prefix sums, exceptions as closed `4x4`s, a segment tree over the blocks | an S-node is a long cycle: Dijkstra is `O(m log m)` per query and labels cost about three times the bytes; the ring answers both roles in `O(1)` inside a run, `O(log K)` across runs |
| hub labels (`skeleton_index/`) | ≥ 200 | a 2-hop labeling (pruned landmark labeling) of the LCA-role skeleton | the giant R-node of a tangled block has thousands of states; a label merge replaces the search |
| pole-projection table | ≥ 1000 | the four pole distances of every inner-skeleton state | a projection becomes four reads; a labeled node with a table keeps no inner arcs |

Hub labels store only out-labels: with the reversal identity, the in-label of `t` is the out-label
of `¬t` with mirrored hubs, and dense ids are numbered so that `rev(id) = n - 1 - id`. The hub
order is by degree (`skeleton_index/pll.rs`); on 25 HPRC chromosome graphs a race against a
sampled-betweenness order saved only 0.18% of the entries at about twice the label build time. A
skeleton of at most 2^16 states stores its labels
with `u16` hubs.

The record keeps only what queries read: the tree with depths and an ancestor table, reference
poles, pole-lift steps, inner skeletons (without arcs where a table and labels answer instead),
`D_out` of the nodes whose LCA search runs over a skeleton, and the oracles. Sub-array offsets are
derived from the counts in the record header (`image/layout.rs`), not stored.

### 2.4 Flip closures and flip baking (`block_cut_lift/flips.rs`, `engine_build/blocks.rs`)

A walk can reverse a cut node's orientation (`c^+ ⇝ c^-`) by an excursion into the graph hanging
off it and back. The **flip closure** of a cut is the `2x2` of the cheapest such flips:
`down[c]` through the pendant subtree below `c` (bottom-up), `up[c]` through everything else
(top-down), `full[c]` both. Blocks are built children first over the block-cut forest, and every
record block is built with its child cuts' `down[c]` **baked in** as `c^a -> c^b` edges, so its
engine answers flips through its child cuts itself; only the parent cut's flip stays explicit.

### 2.5 Block-cut lift (`block_cut_lift/`, `prefix_lift.rs`, `chain_segtree.rs`, `chain_nav/`, `node_rows.rs`)

The cross-block part of a query, as a min-plus product along the block-cut forest instead of a
search.

- **Cut matrices.** A block's raw matrix `M_B` holds the in-block distances between its `2c`
  directed cuts, read through its engine (`cut_matrix.rs`; a block may have at most 4000 cuts).
  Its augmented matrix `M_aug` closes `M_B` with every cut's `full[c]` as a self-loop
  (`gadget.rs`, `aug.rs`), so it covers walks that leave the block any number of times. Because
  the closures are whole-graph, the lift is exact.
- **Path products.** On the cut forest each step is the `2x2` of a block's `M_aug` between a cut and
  its parent cut. The product along a path comes from one of two backends (`path_product.rs`):
  - the **prefix lift** (`prefix_lift.rs`): a step is *regular* when the block touches each cut on
    one side only; then the step is fully described by one length `D` and the two flip closures,
    regular steps compose by adding `D`, and one prefix sum over the root path answers any run in
    `O(1)`. Irregular steps ("lane swaps") keep explicit `2x2`s, found through the nearest exception
    above. Chosen when at most 2% of the steps are exceptions (`MAX_EXCEPTION_FRACTION`).
  - **chain segment trees** (`chain_segtree.rs`): the general backend. The cut forest is split
    into heavy-path chains, each with a min-plus segment tree per direction; `O(log chain)` per
    chain crossed. A segment tree needs only a monoid, which `(Mat2, ⊗, I)` is.

  `SPQR_LIFT_BACKEND` forces either. On pangenome graphs the forest is nearly a path, so both are
  close to `O(1)` chains per query.
- **Chain layout** (`chain_nav/`, `topology/numbering.rs`): the block-cut tree is laid out in
  heavy-path chains with blocks and cut ids numbered in chain order, so parent, depth, LCA and the
  nearest exception are arithmetic on a per-chain record rather than stored pointers.
- **Node rows** (`node_rows.rs`, `block_cut_lift/row_*.rs`): each non-cut node of a block with few
  cuts (default ≤ 8) stores its distances to the block's cuts with `M_aug` folded in, so the
  cross-block route reads a few cells and never opens the block's record. *Full* rows hold both
  orientations of every cut; *half* rows only each cut's away orientation (a block touching each
  cut on one side); *flip* rows the raw distances to the few flip-capable cuts of a many-cut block;
  a *closed-form* node (all arcs lead to cuts) keeps only a first-hop mask. A block whose nodes are
  all closed forms (a *simple* block) keeps no arcs at all.

### 2.6 Node topology and graph view (`topology/`, `graph_view/`)

Every graph node has one 8-byte `NodeRecord`: its length (long lengths escape to a rank-indexed
table), its kind (cut, record-block, plain, simple) and a kind-specific id — a cut id, or the
position of its header in the lift's row table. A record-block node's **SPQR location** is two
numbers: the deepest SPQR node containing it and how far its containing subtree reaches above
(`topology/locate.rs`). The topology is extracted right after the decomposition so the
decomposition can be dropped before the per-block stage, where build memory peaks.

The graph view stores what queries read from the graph: lengths (from the node records), the CSR
adjacency of the nodes whose arcs a query can search, the GFA names (integer runs, or bytes with a
hash index) and a fingerprint of lengths and arcs. With it, `query --index-in` runs without the
GFA.

## 3. The query path

`DistanceIndex::distance(source, target)` (`distance_index.rs`):

```text
original positions ──alias──► G′ positions          (shrunk index only; shrink/distance.rs)
same directed node, target ahead ──► offset difference
different components ──► None
otherwise: min over
   ├─ every block holding both endpoints:
   │     record block   ─► BlockQueryEngine::start_to_start   (block_engine/)
   │     simple block   ─► its rows                            (block_cut_lift/row_cells.rs)
   │     small block    ─► block-restricted Dijkstra           (block_dijkstra.rs)
   └─ cross-block route ─► BlockCutLift::cross                 (block_cut_lift/query.rs)
+ source/target offsets
```

**Inside a record block** (`block_engine/`): locate the deepest SPQR nodes `x_a`, `x_b` holding the
endpoints and their LCA `L` (`query.rs`); project the source onto `x_a`'s poles (segment tree,
projection table, or inner-skeleton Dijkstra, memoized per thread in `memo.rs`) and climb the pole
lift to the child of `L` (`projection.rs`); read the distances over `L`'s LCA-role skeleton (segment
tree, label merges, or one multi-source Dijkstra; `lca.rs`); meet the target's reverse projection
(`target.rs`).

**Across blocks** (`block_cut_lift/query.rs`): anchor each endpoint at its cut or its block; seed
the source at its on-path cut (from its row, or through its block's engine), lift up to the forest
LCA, turn through the LCA block's `M_aug` (or the LCA cut's flip closure), lift down, and read into
the target. Two endpoints in one block also try the route that leaves and re-enters through
flip-capable cuts.

There is never a whole-graph search. The search scratch is sized at load from the header
(`MAX_SEARCHED_NODES`) and reused across queries.

## 4. The image (`.bpi`)

Everything the index keeps lives in one flat byte **image** (`image/`), built in RAM and written as
a `.bpi` file, or memory-mapped back:

```text
┌──────────────┬────────────────────────┬──────────────────────────────────────────────┐
│ header 4 KiB │ TOC: 512 × 32-byte     │ sections (page-aligned) and block records    │
│              │ (id, elem size/align,  │ (unregistered ranges found through the       │
│              │  offset, count)        │  sparse block directory)                     │
└──────────────┴────────────────────────┴──────────────────────────────────────────────┘
```

- **Build.** `Image::new_arena` reserves one large anonymous mapping that never moves and hands out
  bump-allocated typed slices, so finished blocks are readable through the same view types queries
  use while the rest is still appended (the lift reads `M_B` through block engines this way).
  `Image::finish` writes header and TOC; `write_to` writes the used prefix to a temporary file in
  the target's directory and renames it over the target, so a process that has the old file mapped
  keeps a valid mapping.
- **Load.** `Image::open` validates the header and the TOC (every section inside the file, aligned
  and with the element layout this binary expects), memory-maps the file and prefaults all of it,
  so the load pays for reading the file and every query runs warm.
- **Plain data.** Every element type is an `ImageRecord` (`image/record.rs`): `#[repr(C)]`, `Copy`,
  pointer-free (`u8` instead of `bool` or enums), so `&[u8] -> &[T]` is a reinterpretation, and
  without padding at either word size (a compile-time check; a generic record fills what would be
  a gap with an explicit zeroed `Padded` slot), so no uninitialised byte reaches the file. The
  format is native-endian; the header records
  the word size and an endianness check, and each TOC entry its element size and alignment, so a
  mismatched binary refuses a file instead of misreading it. In-image hash tables (`image/hash.rs`)
  replace `HashMap`s.
- **Trust.** The load checks catch truncated, foreign and wrong-version files. Offsets inside
  sections (the block directory, record-internal offsets) are not checked on the query path, so a
  `.bpi` is trusted input, like a vg `.dist`: a crafted file can make a query read out of bounds.
- **Version.** The header carries `FORMAT_VERSION` (currently **19**, `image/arena.rs`); any other
  version is refused and the index must be rebuilt.
- **Section ids** (`sections.rs`) are stable: sections are looked up by id, never by position, and
  unknown ids are ignored. **An id is never renumbered or reused**; ids of removed sections are
  listed in `RETIRED`.
- **Build knobs.** The header's `build_knobs` slots (`sections::knob`) record the label threshold,
  the small-block bound, the S-node segment-tree and projection-table thresholds, the node-row cap,
  the small-block and shrunk flags, and the largest skeleton a query can search. The lift backend
  (`SPQR_LIFT_BACKEND`) also shapes the image but is not recorded. `biopath index-inspect` prints
  header and TOC.

## 5. The CLI (`src/`)

| subcommand | module | does |
|---|---|---|
| `index` | `src/index/` | read GFA/GBZ → shrink → decompose → `DistanceIndex::build` → write `.bpi` |
| `query` | `src/query/` | `--engine dp` (default): load `--index-in`, or build in memory (no shrink); `--engine dijkstra`: whole-graph Dijkstra; `--compare-whole-graph-dijkstra` checks every answer |
| `index-inspect` | `src/inspect.rs` | print header, knobs and TOC |
| `generate-random-queries` | `src/generate_random_queries.rs` | reachable (random walk) or independent random queries |
| `statistics` | `src/statistics.rs` | size statistics of the graph and its SPQR decomposition |
| `export-shrunk-gfa` | `src/export_shrunk.rs` | G′ as a chopped GFA that vg can load |

`--word-size` instantiates every structure with one integer type (`src/word_size.rs`); a loaded
index uses the word size in its header. Log lines and `--timing-out` JSON keys are read by
benchmark scripts and are kept stable.

**Development loop.** `cargo build --profile fast` builds the release code without LTO, with
incremental compilation (`target/fast/biopath`); index bytes and answers do not depend on LTO.

## 6. Module map (`engine/src/`)

| concept | modules |
|---|---|
| entry point: build, open, query | `distance_index` (`BuildOptions`, `DistanceIndex::distance`) |
| build stages, build log | `engine_build` (`blocks`, `record`), `build_report` |
| exact pre-reduction and alias | `shrink` |
| SPQR decomposition, rooted block trees | `decomposition`, `block_spqr_tree`, `rooted_tree` |
| per-block DPs | `inside_dp`, `outside_dp`, `s_node_outside`, `pole_lift` |
| per-block record and in-block query | `block_record`, `block_engine` (`tiers`, `projection`, `lca`, `target`, `memo`) |
| skeleton search and oracles | `skeleton` (CSR + Dijkstra workspace), `skeleton_index` (hub labels), `s_node_segtree` |
| small-block search | `block_dijkstra` |
| cross-block lift | `block_cut_lift`, `prefix_lift`, `chain_segtree`, `chain_nav`, `node_rows` |
| image, schema, stored graph | `image`, `sections`, `topology`, `graph_view` |
| graph input and size limits | `gfa_reader`, `gfa_graph_extensions`, `graph_limits` |
| algebra and basic types | `min_plus`, `orientation`, `path`, `location`, `fast_hash` |
| the one runtime switch | `env` |
| GFA writer for vg | `chopped_gfa` |
| test oracle (not part of the index) | `dijkstra` |

`env.rs` holds the one runtime switch, `SPQR_ARENA_HUGEPAGE`; it changes neither an index nor an
answer.

## 7. Invariants

- **Exactness.** Every answer equals the whole-graph Dijkstra distance. There are no heuristics, no
  approximate tiers and no fallback to a whole-graph search: each structure is an exact min-plus
  composition of exact sub-distances, and every oracle tier returns the value a Dijkstra over the
  same arcs returns.
- **Determinism.** The same graph and options give a byte-identical image, with or without LTO.
  The parallel build steps are order-independent (tests compare them with their sequential
  drivers).
- **Behaviour vs layout.** Answers depend only on the graph; the image depends on the graph, the
  build options and `FORMAT_VERSION`. A change that alters the image on purpose changes the golden
  manifest (below) and says so; a refactor leaves it byte-identical.
- **Format.** Section ids are never reused; an incompatible layout change bumps `FORMAT_VERSION`.
- **Sizing limits.** Values must fit the index word (`MAX` is `∞`): `graph_limits.rs` refuses a
  graph that may not fit before the build starts. The build stops (panics, naming the limit) on a
  block with more than 4000 cut nodes.

## 8. How correctness is tested

- **Dijkstra oracle.** `dijkstra::GfaDijkstra` is a plain whole-graph Dijkstra with the same cost
  model. It is no part of the index; the tests, `query --engine dijkstra` and
  `--compare-whole-graph-dijkstra` check the index against it.
- **Unit and fixture tests** (`cargo test --workspace --profile fast`): per module, e.g. labels,
  S-node segment trees and the outside closed form against skeleton Dijkstra, the min-plus algebra,
  rooted-tree LCA against a naive climb, image round trips and version refusal, and the end-to-end
  fixtures of `engine/test_files/` in `distance_index/tests.rs`.
- **Fuzz tests.** `distance_index/lift_fuzz_tests.rs` builds random bidirected graphs (random link
  orientations, so hairpins, inversions and flip-capable cuts everywhere) and checks every query
  against whole-graph Dijkstra; `shrink/fuzz_tests.rs` does the same for the shrunk index on
  bubble-rich graphs; `block_cut_lift/gadget.rs` fuzzes the gadget closure. A mismatch prints the
  GFA so it can become a fixture.
- **Golden manifests** (`scripts/golden.sh`, `tests/golden/`). For every corpus graph under every
  config of `tests/golden/configs.tsv` (production, no shrink, and "forced" configs that lower all
  tier thresholds so every oracle runs on tiny graphs, with each lift backend pinned) it builds a
  `.bpi`, answers the stored queries, and compares the SHA-256 of image
  and answers with `manifest.tsv`. It also runs the index-free paths
  (`--engine dijkstra`, the in-memory build, the cross-check) and requires equal answers. Every
  config builds at word size 32; `tests/image_determinism.rs` builds the forced configs at 64 twice,
  in separate processes, and compares the bytes.

  ```bash
  cargo build --profile fast
  BIN=target/fast/biopath scripts/golden.sh check
  scripts/golden.sh check                                      # target/release/biopath
  ```

  `scripts/golden.sh record` rewrites the manifest (and checks every answer against
  Dijkstra) for a change that alters the image on purpose.
