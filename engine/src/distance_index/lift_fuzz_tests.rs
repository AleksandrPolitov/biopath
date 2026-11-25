//! Randomized end-to-end check of the index's query path against whole-graph
//! Dijkstra, on small random bidirected graphs decomposed in-process — the production pipeline.
//!
//! Random link orientations put hairpins and inversions everywhere, so blocks get flip-capable cut
//! nodes and some routes must leave a block several times: the shape the hand-written fixture suite
//! missed (see `test_files/twoflip.*`). A mismatch prints the GFA and the location pair so it can be
//! turned into a fixture.

use std::io::Cursor;

use bidirected_adjacency_array::{graph::BidirectedAdjacencyArray, io::gfa1::PlainGfaEdgeData};
use spqr_tree::graph::StaticGraph;

use super::{BuildOptions, DistanceIndex, LABEL_MIN_DIRECTED_NODES, LiftOptions};
use crate::{
    decomposition::build_spqr_decomposition,
    dijkstra::{GfaDijkstra, location_index::single::SingleGfaLocationIndex},
    gfa_graph_extensions::LengthOnlyGfaNodeData,
    location::GfaLocation,
    path::OptionalGfaPathLength,
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

    fn sign(&mut self) -> &'static str {
        if self.below(2) == 0 { "+" } else { "-" }
    }
}

/// A connected random bidirected graph on `n` nodes: a random spanning tree (bridges, so cut nodes)
/// plus `extra` random links (cycles, so non-trivial blocks), every link with random orientations.
/// Occasional parallel links and self-loops come for free.
fn random_gfa(rng: &mut Rng, n: usize, extra: usize) -> String {
    let mut gfa = String::from("H\tVN:Z:1.0\n");
    for v in 0..n {
        let length = 1 + rng.below(3) as usize;
        gfa.push_str(&format!("S\tn{v}\t{}\n", "A".repeat(length)));
    }
    let link = |gfa: &mut String, rng: &mut Rng, u: usize, v: usize| {
        let (su, sv) = (rng.sign(), rng.sign());
        gfa.push_str(&format!("L\tn{u}\t{su}\tn{v}\t{sv}\t0M\n"));
    };
    for v in 1..n {
        let u = rng.below(v as u64) as usize;
        link(&mut gfa, rng, u, v);
    }
    for _ in 0..extra {
        let (u, v) = (rng.below(n as u64) as usize, rng.below(n as u64) as usize);
        link(&mut gfa, rng, u, v);
    }
    gfa
}

/// Every location pair (both orientations, offsets 0 and 1) must match whole-graph Dijkstra.
fn assert_matches_dijkstra(gfa: &str, label_threshold: usize) {
    assert_matches_dijkstra_with(gfa, &BuildOptions::with_label_threshold(label_threshold));
}

/// [`assert_matches_dijkstra`] with the given build options.
fn assert_matches_dijkstra_with(gfa: &str, options: &BuildOptions) {
    let graph: Graph = BidirectedAdjacencyArray::read_gfa1(Cursor::new(gfa)).unwrap();
    let spqr = build_spqr_decomposition(&graph);
    let engine = DistanceIndex::build(&graph, spqr, options);
    let mut oracle = GfaDijkstra::new(&graph);
    let locations: Vec<GfaLocation<u32>> = graph
        .node_indices()
        .flat_map(|node| [node.into_directed(true), node.into_directed(false)])
        .flat_map(|directed| {
            [
                GfaLocation::new(directed, 0.into()),
                GfaLocation::new(directed, 1.into()),
            ]
        })
        .collect();
    for &source in &locations {
        for &target in &locations {
            let answer = OptionalGfaPathLength::from(engine.distance(source, target));
            let expected = OptionalGfaPathLength::from(
                oracle
                    .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                    .values()
                    .map(|path| path.length())
                    .next(),
            );
            assert_eq!(
                answer, expected,
                "distance {source} -> {target} ({:?}, {:?}) differs from whole-graph \
                 Dijkstra on:\n{gfa}",
                options.tiers, options.lift,
            );
        }
    }
}

/// Blocks with MANY cut nodes of which only a few are flip-capable — the shape where `source_seed` /
/// `target_read` skip every cut except the flip-capable ones and the on-path cut. A backbone cycle
/// (one block) gets a pendant at every node: most are plain forward chains (cuts that cannot flip),
/// a few hold a hairpin (flip-capable), and some pendants branch again so the lift has deeper cut paths.
fn many_cut_block_gfa(rng: &mut Rng, cycle: usize) -> String {
    let mut gfa = String::from("H\tVN:Z:1.0\n");
    let mut next = 0usize;
    let mut segment = |gfa: &mut String, rng: &mut Rng| {
        let v = next;
        next += 1;
        gfa.push_str(&format!(
            "S\tn{v}\t{}\n",
            "A".repeat(1 + rng.below(3) as usize)
        ));
        v
    };
    let backbone: Vec<usize> = (0..cycle).map(|_| segment(&mut gfa, rng)).collect();
    // A random orientation on one backbone link now and then puts inversions inside the block too.
    for i in 0..cycle {
        let (u, v) = (backbone[i], backbone[(i + 1) % cycle]);
        let sv = if rng.below(6) == 0 { "-" } else { "+" };
        gfa.push_str(&format!("L\tn{u}\t+\tn{v}\t{sv}\t0M\n"));
    }
    for &anchor in &backbone {
        // Leave some backbone nodes without a pendant: they are the block's non-cut nodes, the only
        // endpoints whose queries take the seed / read fold (a cut-node endpoint reads `full[]`).
        if rng.below(3) == 0 {
            continue;
        }
        let mut tail = anchor;
        for _ in 0..1 + rng.below(3) {
            let v = segment(&mut gfa, rng);
            gfa.push_str(&format!("L\tn{tail}\t+\tn{v}\t+\t0M\n"));
            tail = v;
        }
        match rng.below(5) {
            // Hairpin: the pendant can turn around, so `anchor` is a flip-capable cut.
            0 => gfa.push_str(&format!("L\tn{tail}\t+\tn{tail}\t-\t0M\n")),
            // A side branch off the pendant: more cut nodes deeper in the block-cut tree.
            1 => {
                let v = segment(&mut gfa, rng);
                gfa.push_str(&format!("L\tn{tail}\t+\tn{v}\t{}\t0M\n", rng.sign()));
            }
            _ => {}
        }
    }
    gfa
}

#[test]
fn queries_match_dijkstra_on_many_cut_blocks() {
    let mut rng = Rng(0xc0ff_ee5e_ed0b_adca);
    for round in 0..300 {
        let cycle = 3 + (round % 9);
        let gfa = many_cut_block_gfa(&mut rng, cycle);
        let threshold = if round % 3 == 0 {
            1
        } else {
            LABEL_MIN_DIRECTED_NODES
        };
        assert_matches_dijkstra(&gfa, threshold);
    }
}

/// Node cut rows on every eligible block (the cap raised to 64 so the many-cut fixtures qualify) and
/// with rows off (every cross-block read through the block records): both must match whole-graph
/// Dijkstra on both graph shapes that put flip-capable cuts inside blocks.
#[test]
fn queries_match_dijkstra_with_and_without_node_rows() {
    let with_rows = |node_row_max_cuts, label_threshold| BuildOptions {
        lift: LiftOptions {
            node_row_max_cuts,
            ..LiftOptions::default()
        },
        ..BuildOptions::with_label_threshold(label_threshold)
    };
    for max_cuts in [64, 0] {
        let mut rng = Rng(0xf11b_7ab1_e5ee_d001);
        for round in 0..300 {
            let gfa = many_cut_block_gfa(&mut rng, 3 + (round % 9));
            let threshold = if round % 3 == 0 {
                1
            } else {
                LABEL_MIN_DIRECTED_NODES
            };
            assert_matches_dijkstra_with(&gfa, &with_rows(max_cuts, threshold));
        }
        for round in 0..1500 {
            let n = 4 + (round % 10);
            let extra = rng.below(n as u64 + 1) as usize;
            let gfa = random_gfa(&mut rng, n, extra);
            let threshold = if round % 4 == 0 {
                1
            } else {
                LABEL_MIN_DIRECTED_NODES
            };
            assert_matches_dijkstra_with(&gfa, &with_rows(max_cuts, threshold));
        }
    }
}

#[test]
fn queries_match_dijkstra_on_random_bidirected_graphs() {
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    for round in 0..1500 {
        let n = 4 + (round % 10);
        let extra = rng.below(n as u64 + 1) as usize;
        let gfa = random_gfa(&mut rng, n, extra);
        // Every fourth graph also labels every skeleton, exercising the labeled query path.
        let threshold = if round % 4 == 0 {
            1
        } else {
            LABEL_MIN_DIRECTED_NODES
        };
        assert_matches_dijkstra(&gfa, threshold);
    }
}

/// A block with more than `MAX_BLOCK_CUTS` cut nodes (a cycle with a pendant at every node) fails
/// the build with a clear message.
#[test]
#[should_panic(expected = "cut nodes; at most 4000 are supported")]
fn block_with_too_many_cuts_fails_the_build() {
    use crate::block_cut_lift::MAX_BLOCK_CUTS;
    let cycle = MAX_BLOCK_CUTS + 1;
    let mut gfa = String::from("H\tVN:Z:1.0\n");
    for v in 0..2 * cycle {
        gfa.push_str(&format!("S\tn{v}\tA\n"));
    }
    for v in 0..cycle {
        gfa.push_str(&format!("L\tn{v}\t+\tn{}\t+\t0M\n", (v + 1) % cycle));
        gfa.push_str(&format!("L\tn{v}\t+\tn{}\t+\t0M\n", cycle + v));
    }
    let graph: Graph = BidirectedAdjacencyArray::read_gfa1(Cursor::new(gfa)).unwrap();
    let spqr = build_spqr_decomposition(&graph);
    DistanceIndex::build(&graph, spqr, &BuildOptions::default());
}
