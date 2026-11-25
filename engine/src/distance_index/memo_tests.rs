//! Two indexes in one process must not share the block engines' thread-local memos.
//!
//! The pole-projection memo is keyed by `(image, block, seed, SPQR node)`. Two indexes of graphs
//! with the same topology but different node lengths have the same blocks, SPQR trees and node
//! ids, so without the image in the key one index would read the other's projections: at query
//! time on a shared thread, and at build time on a shared rayon thread (the `M_B` fill warms its
//! projections through the memo), baking the other graph's distances into the image.

use std::io::Cursor;

use bidirected_adjacency_array::{graph::BidirectedAdjacencyArray, io::gfa1::PlainGfaEdgeData};
use spqr_tree::graph::StaticGraph;

use super::{BuildOptions, DistanceIndex};
use crate::{
    decomposition::build_spqr_decomposition,
    dijkstra::{GfaDijkstra, location_index::single::SingleGfaLocationIndex},
    gfa_graph_extensions::LengthOnlyGfaNodeData,
    location::GfaLocation,
    path::OptionalGfaPathLength,
};

type Graph = BidirectedAdjacencyArray<u32, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

/// Two 20-node paths and a link between `s` and `t` (a P-node whose two S-node children have 22
/// vertices: inner skeletons above the memo's size cutoff), with a pendant node on four path nodes
/// (cuts, so the lift fills the block's `M_B`). Node `i` has length `len(i)`.
fn two_long_cycles(len: impl Fn(usize) -> usize) -> String {
    const PATH: usize = 20;
    let mut gfa = String::from("H\tVN:Z:1.0\n");
    let mut nodes = 0;
    let mut node = |gfa: &mut String| {
        let id = nodes;
        gfa.push_str(&format!("S\tn{id}\t{}\n", "A".repeat(len(id))));
        nodes += 1;
        id
    };
    let link = |gfa: &mut String, u: usize, v: usize| {
        gfa.push_str(&format!("L\tn{u}\t+\tn{v}\t+\t0M\n"));
    };
    let s = node(&mut gfa);
    let t = node(&mut gfa);
    for pendant_at in [[3, 15], [7, 11]] {
        let mut prev = s;
        for i in 0..PATH {
            let v = node(&mut gfa);
            link(&mut gfa, prev, v);
            if pendant_at.contains(&i) {
                let p = node(&mut gfa);
                link(&mut gfa, v, p);
            }
            prev = v;
        }
        link(&mut gfa, prev, t);
    }
    link(&mut gfa, t, s);
    gfa
}

fn read(gfa: &str) -> Graph {
    BidirectedAdjacencyArray::read_gfa1(Cursor::new(gfa)).unwrap()
}

fn build(graph: &Graph) -> DistanceIndex<u32> {
    DistanceIndex::build(
        graph,
        build_spqr_decomposition(graph),
        &BuildOptions::default(),
    )
}

/// Every pair of directed nodes (offset 0) must match whole-graph Dijkstra.
fn assert_matches_dijkstra(engine: &DistanceIndex<u32>, graph: &Graph, name: &str) {
    let mut oracle = GfaDijkstra::new(graph);
    let locations: Vec<GfaLocation<u32>> = graph
        .node_indices()
        .flat_map(|node| [node.into_directed(true), node.into_directed(false)])
        .map(|directed| GfaLocation::new(directed, 0.into()))
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
            assert_eq!(answer, expected, "{name}: {source} -> {target}");
        }
    }
}

#[test]
fn two_indexes_in_one_process_answer_independently() {
    let a = read(&two_long_cycles(|_| 1));
    let b = read(&two_long_cycles(|i| 1 + (i * 7) % 5));
    // One pool thread builds both indexes, so it holds A's build-time memo entries during B's build.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (index_a, index_b) = pool.install(|| (build(&a), build(&b)));
    // B's image equals B built on a fresh thread (empty memos, as in a process of its own).
    let fresh = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| build(&b));
    assert!(
        index_b.image().as_bytes() == fresh.image().as_bytes(),
        "B's image depends on the index built before it"
    );
    // This thread queries A, then B, then A again.
    assert_matches_dijkstra(&index_a, &a, "A");
    assert_matches_dijkstra(&index_b, &b, "B");
    assert_matches_dijkstra(&index_a, &a, "A again");
}
