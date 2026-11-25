//! Randomized end-to-end check of the shrunk index (`shrink`): every answer on ORIGINAL
//! positions must equal the distance in the original graph.
//!
//! The generator builds what the reduction targets — SNP bubbles, indels with a direct link,
//! insertions, nested bubbles, inverted alleles, tips, self-loops, duplicate links — on a backbone with
//! occasional inverted nodes, plus random extra links that create irreducible regions. Locations cover
//! every node in both orientations at offset 0, an interior offset and the node end, so positions that
//! coincide after gluing (end of `v` = start of `w`) are exercised. A mismatch prints the GFA.
//!
//! Pruned chains at saturated levels (no known pruning order) are checked twice: on the random
//! graphs with the saturation threshold lowered, and on bubbles nested past the real threshold, whose
//! hops per query must stay linear in the nesting depth.

use std::{cmp::Reverse, collections::BinaryHeap, io::Cursor};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, NodeIndex},
    io::gfa1::PlainGfaEdgeData,
};

use super::{
    OriginalPos, ShrinkAttach, ShrinkPlan,
    distance::{SATURATED_LEVEL_OVERRIDE, WORK},
    layout::set_p1_delta_limit_override,
    records::LEVEL_SATURATED,
    shrink_graph,
    side_graph::{entry_side, exit_side, side_node},
};
use crate::{
    chopped_gfa::write_chopped_gfa,
    decomposition::build_spqr_decomposition,
    dijkstra::{GfaDijkstra, location_index::single::SingleGfaLocationIndex},
    distance_index::{BuildOptions, DistanceIndex, LABEL_MIN_DIRECTED_NODES},
    gfa_graph_extensions::{GfaNodeDataExt, LengthOnlyGfaNodeData},
    location::{GfaLocation, GfaNodeOffset},
    path::{GfaPathLength, OptionalGfaPathLength},
    sections,
};

type Graph = BidirectedAdjacencyArray<u32, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

/// Deterministic xorshift.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn sign(&mut self) -> char {
        if self.below(2) == 0 { '+' } else { '-' }
    }
}

struct Gfa {
    lens: Vec<usize>,
    links: Vec<(usize, char, usize, char)>,
}

impl Gfa {
    fn node(&mut self, rng: &mut Rng) -> usize {
        self.node_of_len(1 + rng.below(3) as usize)
    }

    fn node_of_len(&mut self, len: usize) -> usize {
        self.lens.push(len);
        self.lens.len() - 1
    }

    fn link(&mut self, u: usize, su: char, v: usize, sv: char) {
        self.links.push((u, su, v, sv));
    }

    /// A chain of `k` new nodes from `(u, su)`, returning its last node (walked forward).
    fn chain(&mut self, rng: &mut Rng, u: usize, su: char, k: usize) -> usize {
        let (mut prev, mut sp) = (u, su);
        for _ in 0..k {
            let x = self.node(rng);
            self.link(prev, sp, x, '+');
            (prev, sp) = (x, '+');
        }
        prev
    }

    fn text(&self) -> String {
        let mut gfa = String::from("H\tVN:Z:1.0\n");
        for (v, &len) in self.lens.iter().enumerate() {
            gfa.push_str(&format!("S\tn{v}\t{}\n", "A".repeat(len)));
        }
        for &(u, su, v, sv) in &self.links {
            gfa.push_str(&format!("L\tn{u}\t{su}\tn{v}\t{sv}\t0M\n"));
        }
        gfa
    }
}

/// A backbone of `events` variation sites plus `extra` random links.
fn bubbly_gfa(rng: &mut Rng, events: usize, extra: usize) -> String {
    let mut g = Gfa {
        lens: Vec::new(),
        links: Vec::new(),
    };
    let mut cur = g.node(rng);
    let mut so = '+';
    for _ in 0..events {
        let next = g.node(rng);
        let no = if rng.below(8) == 0 { '-' } else { '+' };
        match rng.below(13) {
            0 => g.link(cur, so, next, no),
            1 => {
                // SNP / multi-allelic site: 2-3 single-node alleles.
                for _ in 0..2 + rng.below(2) {
                    let a = g.node(rng);
                    g.link(cur, so, a, '+');
                    g.link(a, '+', next, no);
                }
            }
            2 => {
                // Deletion: an allele chain plus a direct link.
                let k = 1 + rng.below(3) as usize;
                let last = g.chain(rng, cur, so, k);
                g.link(last, '+', next, no);
                g.link(cur, so, next, no);
            }
            3 => {
                // Insertion without alternative: a plain chain.
                let last = {
                    let k = 1 + rng.below(2) as usize;
                    g.chain(rng, cur, so, k)
                };
                g.link(last, '+', next, no);
            }
            4 => {
                // Nested: an outer allele x1 -> {inner SNP} -> x2, and an alternative allele y.
                let x1 = g.chain(rng, cur, so, 1);
                let x2 = g.node(rng);
                for _ in 0..2 {
                    let a = g.node(rng);
                    g.link(x1, '+', a, '+');
                    g.link(a, '+', x2, '+');
                }
                g.link(x2, '+', next, no);
                let y = {
                    let k = 1 + rng.below(2) as usize;
                    g.chain(rng, cur, so, k)
                };
                g.link(y, '+', next, no);
            }
            5 => {
                // An inverted allele next to a forward one.
                let a = g.node(rng);
                g.link(cur, so, a, '-');
                g.link(a, '-', next, no);
                let b = g.chain(rng, cur, so, 1);
                g.link(b, '+', next, no);
            }
            6 => {
                // A tip (pendant chain) off the backbone.
                let side = if rng.below(2) == 0 {
                    so
                } else {
                    if so == '+' { '-' } else { '+' }
                };
                {
                    let k = 1 + rng.below(2) as usize;
                    g.chain(rng, cur, side, k)
                };
                g.link(cur, so, next, no);
            }
            7 => {
                // A self-loop.
                g.link(cur, rng.sign(), cur, rng.sign());
                g.link(cur, so, next, no);
            }
            8 => {
                // Alleles of mixed lengths plus a duplicated link.
                let a = {
                    let k = 1 + rng.below(3) as usize;
                    g.chain(rng, cur, so, k)
                };
                g.link(a, '+', next, no);
                let b = g.chain(rng, cur, so, 1);
                g.link(b, '+', next, no);
                g.link(b, '+', next, no);
            }
            10 => {
                // Two adjacent sites (a "lattice"): cur -> {a1, b1} -> {a2, b2} -> next, with 3 or 4 of
                // the 4 links between the sites.
                let (a1, b1, a2, b2) = (g.node(rng), g.node(rng), g.node(rng), g.node(rng));
                for x in [a1, b1] {
                    g.link(cur, so, x, '+');
                }
                for y in [a2, b2] {
                    g.link(y, '+', next, no);
                }
                let skip = rng.below(5); // 4 = keep all four links
                for (i, (x, y)) in [(a1, a2), (a1, b2), (b1, a2), (b1, b2)]
                    .into_iter()
                    .enumerate()
                {
                    if i as u64 != skip {
                        g.link(x, '+', y, '+');
                    }
                }
            }
            11 => {
                // A diamond with a chord (K4-like): cur -> {a, b} -> next plus a -> b, so `a` has one
                // link on its left and two on its right, `b` the mirror image.
                let (a, b) = (g.node(rng), g.node(rng));
                g.link(cur, so, a, '+');
                g.link(cur, so, b, '+');
                g.link(a, '+', b, '+');
                g.link(a, '+', next, no);
                g.link(b, '+', next, no);
            }
            12 => {
                // Overlapping deletions: cur -> a -> b -> next with cur -> b and a -> next, sometimes a
                // third allele c from a, so one end of `a` carries up to three links.
                let (a, b) = (g.node(rng), g.node(rng));
                g.link(cur, so, a, '+');
                g.link(a, '+', b, '+');
                g.link(b, '+', next, no);
                g.link(cur, so, b, '+');
                g.link(a, '+', next, no);
                if rng.below(2) == 0 {
                    let c = g.node(rng);
                    g.link(a, '+', c, '+');
                    g.link(c, '+', next, no);
                }
            }
            _ => {
                // A hairpin allele: leaves `cur` and comes back into `cur` reversed.
                let a = g.chain(rng, cur, so, 1);
                g.link(a, '+', cur, if so == '+' { '-' } else { '+' });
                g.link(cur, so, next, no);
            }
        }
        cur = next;
        so = no;
    }
    let n = g.lens.len();
    for _ in 0..extra {
        let (u, v) = (rng.below(n as u64) as usize, rng.below(n as u64) as usize);
        let (su, sv) = (rng.sign(), rng.sign());
        g.link(u, su, v, sv);
    }
    g.text()
}

/// Bubbles nested `depth` deep from side `from`: `s(i) → s(i+1)` and `t(i+1) → t(i)` with a shorter
/// allele `s(i) → a(i) → t(i)` at each depth, `s(depth) → t(depth)` innermost and `t(0)` linked to
/// `to` (none: the nest is a tip). The chain `s(i) a(i) t(i)` is pruned at level `depth + 1 - i`.
/// Returns `[s(i), a(i), t(i)]` per depth (`a(depth)` = `s(depth)`).
fn nest(
    g: &mut Gfa,
    rng: &mut Rng,
    from: (usize, char),
    to: Option<(usize, char)>,
    depth: usize,
) -> Vec<[usize; 3]> {
    let mut levels = Vec::new();
    for i in 0..=depth {
        let s = g.node_of_len(2 + rng.below(3) as usize);
        let t = g.node_of_len(2 + rng.below(3) as usize);
        let a = if i < depth {
            g.node_of_len(1 + rng.below(2) as usize)
        } else {
            s
        };
        levels.push([s, a, t]);
    }
    let (u, su) = from;
    g.link(u, su, levels[0][0], '+');
    if let Some((v, sv)) = to {
        g.link(levels[0][2], '+', v, sv);
    }
    for w in levels.windows(2) {
        let ([s, a, t], [s2, _, t2]) = (w[0], w[1]);
        g.link(s, '+', s2, '+');
        g.link(t2, '+', t, '+');
        g.link(s, '+', a, '+');
        g.link(a, '+', t, '+');
    }
    let [s, _, t] = levels[depth];
    g.link(s, '+', t, '+');
    levels
}

/// Every directed start-to-start distance from `source`'s successors (leaving `source` from
/// `offset` costs `len - offset`), as the whole-graph Dijkstra defines it.
fn arrivals(graph: &Graph, source: GfaLocation<u32>) -> Vec<Option<u64>> {
    let len = |d: usize| {
        let l: GfaPathLength<u32> = graph.node_data(NodeIndex::from_usize(d / 2)).len();
        l.into_usize() as u64
    };
    let directed = 2 * graph.node_count();
    let mut dist: Vec<Option<u64>> = vec![None; directed];
    let mut heap = BinaryHeap::new();
    let s = source.node().into_usize();
    let first = len(s) - source.offset().into_usize() as u64;
    for edge in graph.iter_outgoing_edges(source.node()) {
        heap.push(Reverse((first, edge.to().into_usize())));
    }
    while let Some(Reverse((cost, d))) = heap.pop() {
        if dist[d].is_some() {
            continue;
        }
        dist[d] = Some(cost);
        let leave = cost + len(d);
        for edge in graph.iter_outgoing_edges(DirectedNodeIndex::from_usize(d)) {
            if dist[edge.to().into_usize()].is_none() {
                heap.push(Reverse((leave, edge.to().into_usize())));
            }
        }
    }
    dist
}

fn reference(
    graph: &Graph,
    arrive: &[Option<u64>],
    source: GfaLocation<u32>,
    target: GfaLocation<u32>,
) -> Option<u64> {
    let (os, ot) = (
        source.offset().into_usize() as u64,
        target.offset().into_usize() as u64,
    );
    let direct = (source.node() == target.node() && ot >= os).then(|| ot - os);
    let outer = arrive[target.node().into_usize()].map(|c| c + ot);
    let _ = graph;
    match (direct, outer) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// Check that the reduction reached its fixpoint on G′: no link is left between two different nodes
/// whose facing sides carry only it (it would have been glued), and no tip-shaped node (one side
/// unlinked, the other carrying one link to another node) is left. Zero-length nodes are exempt, as
/// in the reduction.
fn check_fixpoint(plan: &ShrinkPlan<u32>) -> Result<(), String> {
    let g = &plan.graph;
    let n = g.node_count();
    let len = |v: usize| -> usize {
        let l: GfaPathLength<u32> = g.node_data(NodeIndex::from_usize(v)).len();
        l.into_usize()
    };
    let mut pairs: Vec<(usize, usize)> = g
        .iter_edges()
        .map(|e| {
            let view = g.edge(e);
            let (from, to) = (view.from(), view.to());
            let a = exit_side(from.into_bidirected().into_usize(), from.is_forward());
            let b = entry_side(to.into_bidirected().into_usize(), to.is_forward());
            (a.min(b), a.max(b))
        })
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    let mut deg = vec![0usize; 2 * n];
    let mut partner = vec![usize::MAX; 2 * n];
    for &(a, b) in &pairs {
        deg[a] += 1;
        deg[b] += 1;
        partner[a] = b;
        partner[b] = a;
    }
    for &(a, b) in &pairs {
        let (u, v) = (side_node(a), side_node(b));
        if u != v && deg[a] == 1 && deg[b] == 1 && len(u) > 0 && len(v) > 0 {
            return Err(format!(
                "link between G′ sides {a} and {b} should have been glued"
            ));
        }
    }
    for v in (0..n).filter(|&v| len(v) > 0) {
        let (l, r) = (2 * v, 2 * v + 1);
        let outside = |s: usize| deg[s] == 1 && side_node(partner[s]) != v;
        if (deg[l] == 0 && outside(r)) || (deg[r] == 0 && outside(l)) {
            return Err(format!("tip-shaped G′ node {v} is left"));
        }
    }
    Ok(())
}

/// Build the shrunk index of `gfa` and compare every location pair with the original graph, then once
/// more with almost every single-node record's ports escaped to the far table.
fn assert_shrunk_matches(
    gfa: &str,
    label_threshold: usize,
    check_against_dijkstra: bool,
) -> (usize, usize) {
    let out = assert_shrunk_matches_once(gfa, label_threshold, check_against_dijkstra);
    set_p1_delta_limit_override(Some(1));
    assert_shrunk_matches_once(gfa, label_threshold, false);
    set_p1_delta_limit_override(None);
    out
}

fn assert_shrunk_matches_once(
    gfa: &str,
    label_threshold: usize,
    check_against_dijkstra: bool,
) -> (usize, usize) {
    let graph: Graph = BidirectedAdjacencyArray::read_gfa1(Cursor::new(gfa)).unwrap();
    let plan = shrink_graph(&graph).unwrap_or_else(|e| panic!("shrink failed ({e}) on:\n{gfa}"));
    check_fixpoint(&plan).unwrap_or_else(|e| panic!("shrink fixpoint: {e} on:\n{gfa}"));
    let spqr = build_spqr_decomposition(&plan.graph);
    let attach = ShrinkAttach {
        plan: &plan,
        original: &graph,
        original_names: None,
    };
    let options = BuildOptions {
        shrink: Some(&attach),
        ..BuildOptions::with_label_threshold(label_threshold)
    };
    let engine = DistanceIndex::build(&plan.graph, spqr, &options);
    for entry in engine.image().toc() {
        assert!(
            sections::is_live(entry.id),
            "section id {} is not live",
            entry.id
        );
    }
    let locations: Vec<GfaLocation<u32>> = graph
        .iter_nodes()
        .flat_map(|node| {
            let len: GfaPathLength<u32> = graph.node_data(node).len();
            let len = len.into_usize() as u32;
            let mut offsets = vec![0, len / 2, len];
            offsets.dedup();
            [node.into_directed(true), node.into_directed(false)]
                .into_iter()
                .flat_map(move |d| {
                    offsets
                        .clone()
                        .into_iter()
                        .map(move |o| GfaLocation::new(d, GfaNodeOffset::from_usize(o as usize)))
                })
        })
        .collect();
    let mut oracle = GfaDijkstra::new(&graph);
    for (i, &source) in locations.iter().enumerate() {
        let arrive = arrivals(&graph, source);
        for (j, &target) in locations.iter().enumerate() {
            let expected = reference(&graph, &arrive, source, target);
            if check_against_dijkstra && (i * 31 + j) % 97 == 0 {
                let dijkstra = OptionalGfaPathLength::from(
                    oracle
                        .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                        .values()
                        .map(|path| path.length())
                        .next(),
                );
                let reference = OptionalGfaPathLength::from(
                    expected.map(|d| GfaPathLength::from_usize(d as usize)),
                );
                assert_eq!(
                    reference, dijkstra,
                    "test reference differs from GfaDijkstra {source} -> {target} on:\n{gfa}"
                );
            }
            let shrunk = engine
                .distance(source, target)
                .map(|d| d.into_usize() as u64);
            assert_eq!(
                shrunk, expected,
                "shrunk distance {source} -> {target} differs from the original graph \
                 (shrunk to {} of {} nodes, {:?}) on:\n{gfa}",
                plan.stats.shrunk_nodes, plan.stats.original_nodes, plan.stats,
            );
        }
    }
    (plan.stats.original_nodes, plan.stats.shrunk_nodes)
}

/// The alias must give every original node its length, and a pruned chain whose port lies in another
/// pruned chain must have the smaller level (the port's chain was pruned later).
fn assert_alias_lengths_and_levels(gfa: &str) {
    use super::{OriginalPos, ShrinkSink, ShrunkOracle, ShrunkPos};
    use crate::image::Image;
    struct LenOracle<'a>(&'a Graph);
    impl ShrunkOracle for LenOracle<'_> {
        fn node_len(&mut self, node: usize) -> u64 {
            let len: GfaPathLength<u32> = self.0.node_data(NodeIndex::from_usize(node)).len();
            len.into_usize() as u64
        }
        fn distance(&mut self, _: ShrunkPos, _: ShrunkPos, _: bool) -> Option<u64> {
            unreachable!("probe never asks the engine")
        }
    }
    let graph: Graph = BidirectedAdjacencyArray::read_gfa1(Cursor::new(gfa)).unwrap();
    let plan = shrink_graph(&graph).unwrap_or_else(|e| panic!("shrink failed ({e}) on:\n{gfa}"));
    let attach = ShrinkAttach {
        plan: &plan,
        original: &graph,
        original_names: None,
    };
    let image = Image::new_default_arena().unwrap();
    let secs = attach.write(&image);
    let view = secs.view(&image);
    let mut oracle = LenOracle(&plan.graph);
    let n = graph.node_count();
    assert_eq!(view.node_count(), n);
    for v in 0..n {
        let original: GfaPathLength<u32> = graph.node_data(NodeIndex::from_usize(v)).len();
        let len = view.node_len(v);
        assert_eq!(
            len,
            original.into_usize() as u64,
            "length of node {v} on:\n{gfa}"
        );
        for forward in [true, false] {
            for offset in [0, len / 2, len] {
                let p = OriginalPos {
                    node: v,
                    forward,
                    offset,
                };
                let mine = view.probe(p, &mut oracle);
                let Some((_, (exit, entry))) = mine.chain else {
                    continue;
                };
                for port in [exit, entry] {
                    // A pruned tip's unlinked end has no port.
                    if port == u32::MAX {
                        continue;
                    }
                    let q = OriginalPos {
                        node: (port / 2) as usize,
                        forward: port % 2 == 0,
                        offset: 0,
                    };
                    let theirs = view.probe(q, &mut oracle);
                    if theirs.chain.is_some() {
                        assert!(
                            mine.level < theirs.level,
                            "chain of node {v} (level {}) has a port in node {} (level {}) on:\n{gfa}",
                            mine.level,
                            q.node,
                            theirs.level
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn alias_lengths_and_levels_on_every_position() {
    let mut rng = Rng(0xa11a_5f0e_1dea_0001);
    for round in 0..rounds(400) {
        let gfa = if round % 2 == 0 {
            let extra = if round % 3 == 0 {
                rng.below(4) as usize
            } else {
                0
            };
            bubbly_gfa(&mut rng, 1 + round % 12, extra)
        } else {
            let n = 3 + (round % 10);
            let mut g = Gfa {
                lens: Vec::new(),
                links: Vec::new(),
            };
            for _ in 0..n {
                g.node(&mut rng);
            }
            for v in 1..n {
                let u = rng.below(v as u64) as usize;
                let (su, sv) = (rng.sign(), rng.sign());
                g.link(u, su, v, sv);
            }
            for _ in 0..rng.below(n as u64 + 1) {
                let (u, v) = (rng.below(n as u64) as usize, rng.below(n as u64) as usize);
                let (su, sv) = (rng.sign(), rng.sign());
                g.link(u, su, v, sv);
            }
            g.text()
        };
        assert_alias_lengths_and_levels(&gfa);
    }
}

/// Debug helper: check the GFA at `SHRINK_FUZZ_GFA` (exit status = verdict; used to minimise fuzz
/// failures). `cargo test -p spqr-shortest-path-index --release one_gfa_from_env -- --ignored`.
#[test]
#[ignore]
fn one_gfa_from_env() {
    let path = std::env::var("SHRINK_FUZZ_GFA").expect("SHRINK_FUZZ_GFA");
    let gfa = std::fs::read_to_string(path).unwrap();
    assert_shrunk_matches(&gfa, LABEL_MIN_DIRECTED_NODES, false);
}

/// `SHRINK_FUZZ_ROUNDS` scales the round counts (default 1) for deeper local runs.
fn rounds(base: usize) -> usize {
    base * std::env::var("SHRINK_FUZZ_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

#[test]
fn shrunk_queries_match_the_original_graph_on_bubbly_graphs() {
    let mut rng = Rng(0x5eed_5a1c_0ffe_e001);
    let (mut before, mut after) = (0, 0);
    for round in 0..rounds(600) {
        let events = 1 + (round % 12);
        let extra = if round % 3 == 0 {
            rng.below(4) as usize
        } else {
            0
        };
        let gfa = bubbly_gfa(&mut rng, events, extra);
        let threshold = if round % 4 == 0 {
            1
        } else {
            LABEL_MIN_DIRECTED_NODES
        };
        let (b, a) = assert_shrunk_matches(&gfa, threshold, round % 10 == 0);
        before += b;
        after += a;
    }
    // The generator is mostly reducible: the reduction must actually happen.
    assert!(after * 2 < before, "shrunk {after} of {before} nodes");
}

#[test]
fn shrunk_queries_match_the_original_graph_on_random_graphs() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for round in 0..rounds(600) {
        let n = 3 + (round % 10);
        let mut g = Gfa {
            lens: Vec::new(),
            links: Vec::new(),
        };
        for _ in 0..n {
            g.node(&mut rng);
        }
        for v in 1..n {
            let u = rng.below(v as u64) as usize;
            let (su, sv) = (rng.sign(), rng.sign());
            g.link(u, su, v, sv);
        }
        for _ in 0..rng.below(n as u64 + 1) {
            let (u, v) = (rng.below(n as u64) as usize, rng.below(n as u64) as usize);
            let (su, sv) = (rng.sign(), rng.sign());
            g.link(u, su, v, sv);
        }
        assert_shrunk_matches(&g.text(), LABEL_MIN_DIRECTED_NODES, round % 10 == 0);
    }
}

/// `export-shrunk-gfa`: the chopped G_s GFA has the same start-to-start distances between G_s nodes
/// (via their first entered pieces) as G_s itself, for every chop size.
#[test]
fn chopped_shrunk_gfa_keeps_the_distances_of_the_shrunk_graph() {
    let mut rng = Rng(0x0e8b_07c4_1a5e_d00d);
    for round in 0..rounds(300) {
        let extra = if round % 3 == 0 {
            rng.below(4) as usize
        } else {
            0
        };
        let gfa = bubbly_gfa(&mut rng, 1 + round % 12, extra);
        let graph: Graph = BidirectedAdjacencyArray::read_gfa1(Cursor::new(&gfa)).unwrap();
        let plan =
            shrink_graph(&graph).unwrap_or_else(|e| panic!("shrink failed ({e}) on:\n{gfa}"));
        let gs = &plan.graph;
        for chop in [0, 1, 2, 5] {
            let mut text = Vec::new();
            let export = write_chopped_gfa(gs, chop, &mut text).unwrap();
            let chopped: Graph = BidirectedAdjacencyArray::read_gfa1(Cursor::new(&text)).unwrap();
            assert_eq!(chopped.node_count() as u64, export.segments);
            let piece = |(id, forward): (u64, bool)| {
                NodeIndex::<u32>::from_usize(id as usize - 1).into_directed(forward)
            };
            let at_start =
                |d: GfaLocation<u32>| GfaLocation::new(d.node(), GfaNodeOffset::from_usize(0));
            for s in gs.iter_nodes() {
                for sf in [true, false] {
                    let want = arrivals(
                        gs,
                        at_start(GfaLocation::new(
                            s.into_directed(sf),
                            GfaNodeOffset::from_usize(0),
                        )),
                    );
                    let start = piece(export.ids.start(s.into_usize(), sf));
                    let got = arrivals(
                        &chopped,
                        GfaLocation::new(start, GfaNodeOffset::from_usize(0)),
                    );
                    for t in gs.iter_nodes() {
                        for tf in [true, false] {
                            let target = piece(export.ids.start(t.into_usize(), tf));
                            assert_eq!(
                                want[t.into_directed(tf).into_usize()],
                                got[target.into_usize()],
                                "chop {chop}: G_s {s:?}{sf} -> {t:?}{tf} differs in the export of:\n{gfa}\nexport:\n{}",
                                String::from_utf8_lossy(&text),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn shrunk_queries_match_the_original_graph_with_low_saturated_levels() {
    let mut rng = Rng(0x05a7_0ade_1e7e_1001);
    for round in 0..rounds(300) {
        let gfa = if round % 3 == 2 {
            let n = 3 + (round % 10);
            let mut g = Gfa {
                lens: Vec::new(),
                links: Vec::new(),
            };
            for _ in 0..n {
                g.node(&mut rng);
            }
            for v in 1..n {
                let u = rng.below(v as u64) as usize;
                let (su, sv) = (rng.sign(), rng.sign());
                g.link(u, su, v, sv);
            }
            for _ in 0..rng.below(n as u64 + 1) {
                let (u, v) = (rng.below(n as u64) as usize, rng.below(n as u64) as usize);
                let (su, sv) = (rng.sign(), rng.sign());
                g.link(u, su, v, sv);
            }
            g.text()
        } else {
            let extra = if round % 4 == 0 {
                rng.below(4) as usize
            } else {
                0
            };
            bubbly_gfa(&mut rng, 1 + round % 12, extra)
        };
        // Levels start at 1: from threshold 1 every pair of pruned chains is saturated.
        SATURATED_LEVEL_OVERRIDE.set(Some(1 + (round % 3) as u32));
        assert_shrunk_matches_once(&gfa, LABEL_MIN_DIRECTED_NODES, false);
    }
    SATURATED_LEVEL_OVERRIDE.set(None);
}

/// Two nests of bubbles deeper than [`LEVEL_SATURATED`] around an irreducible core: one on a cycle
/// through it, one a tip hanging off it. A query between two saturated chains used to try both hop
/// orders at every hop (`binomial(I + J, I)` recursions); now every answer must match the original
/// graph with at most one engine query and hops linear in the depths.
#[test]
fn saturated_nested_bubbles_answer_exactly_in_linear_hops() {
    const CYCLE_DEPTH: usize = 290;
    const TIP_DEPTH: usize = 280;
    let mut rng = Rng(0x0dee_95a7_0000_0001);
    let mut g = Gfa {
        lens: Vec::new(),
        links: Vec::new(),
    };
    // The core: x → {p, q} → y with p → q (no chain, no allele).
    let [x, p, q, y] = std::array::from_fn(|_| g.node(&mut rng));
    for (u, v) in [(x, p), (x, q), (p, q), (p, y), (q, y)] {
        g.link(u, '+', v, '+');
    }
    let cycle = nest(&mut g, &mut rng, (y, '+'), Some((x, '+')), CYCLE_DEPTH);
    let tip = nest(&mut g, &mut rng, (p, '+'), None, TIP_DEPTH);
    let gfa = g.text();

    let graph: Graph = BidirectedAdjacencyArray::read_gfa1(Cursor::new(&gfa)).unwrap();
    let plan = shrink_graph(&graph).unwrap();
    assert_eq!(
        plan.stats.max_level as usize,
        CYCLE_DEPTH.max(TIP_DEPTH + 1)
    );
    let spqr = build_spqr_decomposition(&plan.graph);
    let attach = ShrinkAttach {
        plan: &plan,
        original: &graph,
        original_names: None,
    };
    let options = BuildOptions {
        shrink: Some(&attach),
        ..BuildOptions::with_label_threshold(LABEL_MIN_DIRECTED_NODES)
    };
    let engine = DistanceIndex::build(&plan.graph, spqr, &options);

    // Depths around the saturation boundary (`depth + 1 - i = LEVEL_SATURATED`), the top and the
    // bottom of each nest, and the core.
    let around = |depth: usize, more: &[usize]| {
        let edge = depth + 1 - LEVEL_SATURATED as usize;
        let mut depths: Vec<usize> = (0..3).chain(edge - 2..=edge + 2).collect();
        depths.extend(more.iter().chain(&[depth - 1, depth]));
        depths
    };
    let mut nodes = vec![x, p, q, y];
    for (levels, depths) in [
        (&cycle, around(CYCLE_DEPTH, &[17, 120])),
        (&tip, around(TIP_DEPTH, &[12, 100])),
    ] {
        for i in depths {
            nodes.extend(levels[i]);
        }
    }
    nodes.sort_unstable();
    nodes.dedup();
    let locations: Vec<GfaLocation<u32>> = nodes
        .iter()
        .flat_map(|&v| {
            let len = g.lens[v] as u32;
            [true, false].into_iter().flat_map(move |forward| {
                [0, len / 2, len].into_iter().map(move |o| {
                    let node = NodeIndex::<u32>::from_usize(v).into_directed(forward);
                    GfaLocation::new(node, GfaNodeOffset::from_usize(o as usize))
                })
            })
        })
        .collect();
    let saturated = |l: GfaLocation<u32>| {
        let position = OriginalPos {
            node: l.node().into_bidirected().into_usize(),
            forward: l.node().is_forward(),
            offset: l.offset().into_usize() as u64,
        };
        engine
            .super_location(position)
            .and_then(|s| s.level())
            .is_some_and(|level| level >= LEVEL_SATURATED)
    };

    // Each side hops through at most one nest, once per chain.
    let max_hops = 2 * (CYCLE_DEPTH.max(TIP_DEPTH) + 1) as u64;
    let (mut saturated_pairs, mut answered) = (0, 0);
    for &source in &locations {
        let arrive = arrivals(&graph, source);
        for &target in &locations {
            let expected = reference(&graph, &arrive, source, target);
            WORK.set([0; 2]);
            let shrunk = engine
                .distance(source, target)
                .map(|d| d.into_usize() as u64);
            let [hops, g_queries] = WORK.get();
            assert_eq!(shrunk, expected, "{source} -> {target}");
            assert!(
                hops <= max_hops && g_queries <= 1,
                "{source} -> {target}: {hops} hops, {g_queries} G′ queries",
            );
            if saturated(source) && saturated(target) {
                saturated_pairs += 1;
                answered += usize::from(expected.is_some());
            }
        }
    }
    // The sample reaches the saturated case, answered and unreachable.
    assert!(
        saturated_pairs > 10_000 && answered > 1_000 && answered < saturated_pairs,
        "{saturated_pairs} saturated pairs, {answered} reachable"
    );
}
