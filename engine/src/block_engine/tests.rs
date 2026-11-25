use std::{fs::File, io::BufReader};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    io::gfa1::{PlainGfaEdgeData, PlainGfaNodeData},
};
use spqr_tree::{decomposition::SPQRDecomposition, graph::StaticGraph};

use super::test_support::{TestBlock, TestLabels};
use crate::{
    dijkstra::{GfaDijkstra, location_index::single::SingleGfaLocationIndex},
    gfa_graph_extensions::GfaNodeDataExt,
    location::GfaLocation,
    min_plus::MinPlus,
    orientation::Orientation,
    path::OptionalGfaPathLength,
    skeleton::DijkstraWorkspace,
};

type Graph = BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData>;

fn read_graph(path: &str) -> Graph {
    BidirectedAdjacencyArray::read_gfa1(BufReader::new(File::open(path).unwrap())).unwrap()
}

/// Every location-to-location query in a single-block graph must match whole-graph Dijkstra, over
/// all directed nodes and a range of offsets (including reverse orientations and same-node cases).
fn assert_queries_match_oracle(gfa_path: &str, spqr_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr =
        SPQRDecomposition::read_plain_spqr(&graph, BufReader::new(File::open(spqr_path).unwrap()))
            .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let tb = TestBlock::build(&graph, &spqr, block_index, TestLabels::None);
    let view = tb.view();
    let engine = tb.engine(&view);

    let mut oracle = GfaDijkstra::new(&graph);

    let locations: Vec<GfaLocation<u32>> = graph
        .node_indices()
        .flat_map(|node| [node.into_directed(true), node.into_directed(false)])
        .flat_map(|directed| {
            // offsets 0 and 1 (node sequences have length 2 in these fixtures).
            [
                GfaLocation::new(directed, 0.into()),
                GfaLocation::new(directed, 1.into()),
            ]
        })
        .collect();

    for &source in &locations {
        for &target in &locations {
            let engine_distance = OptionalGfaPathLength::from(engine.distance(source, target));
            let oracle_distance = OptionalGfaPathLength::from(
                oracle
                    .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                    .values()
                    .map(|path| path.length())
                    .next(),
            );
            assert_eq!(
                engine_distance, oracle_distance,
                "mismatch in {gfa_path}: {source} -> {target}",
            );
        }
    }
}

/// Same as [`assert_queries_match_oracle`] but forces the **hub-labeling** LCA path on every
/// skeleton (threshold 0), so the labeled query engine is validated against whole-graph Dijkstra
/// on the R-node and orientation-flip fixtures — including the target-lifts-up `combine_target` case.
fn assert_labeled_queries_match_oracle(gfa_path: &str, spqr_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr =
        SPQRDecomposition::read_plain_spqr(&graph, BufReader::new(File::open(spqr_path).unwrap()))
            .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    // Threshold 0 => every skeleton gets labels => the labeled LCA path is always taken.
    let tb = TestBlock::build(&graph, &spqr, block_index, TestLabels::Precompute(0));
    let view = tb.view();
    let engine = tb.engine(&view);

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
            let engine_distance = OptionalGfaPathLength::from(engine.distance(source, target));
            let oracle_distance = OptionalGfaPathLength::from(
                oracle
                    .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                    .values()
                    .map(|path| path.length())
                    .next(),
            );
            assert_eq!(
                engine_distance, oracle_distance,
                "labeled mismatch in {gfa_path}: {source} -> {target}",
            );
        }
    }
}

/// The projection cache must be a pure speedup: warming every node's `pole_projection` and querying
/// through `with_projection_cache` gives byte-identical distances to the uncached engine — over BOTH the
/// Dijkstra LCA path and the labeled path (the one the batched cut-matrix fill actually uses) — and both
/// must match whole-graph Dijkstra. This is the exactness guarantee for the projection-sharing M_B fill.
fn assert_cached_matches(gfa_path: &str, spqr_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr =
        SPQRDecomposition::read_plain_spqr(&graph, BufReader::new(File::open(spqr_path).unwrap()))
            .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let tb_plain = TestBlock::build(&graph, &spqr, block_index, TestLabels::None);
    let tb_labeled = TestBlock::build(&graph, &spqr, block_index, TestLabels::Precompute(0));

    let cuts: Vec<_> = graph.node_indices().collect();
    let locations: Vec<GfaLocation<u32>> = graph
        .node_indices()
        .flat_map(|node| [node.into_directed(true), node.into_directed(false)])
        .flat_map(|d| [GfaLocation::new(d, 0.into()), GfaLocation::new(d, 1.into())])
        .collect();
    let mut oracle = GfaDijkstra::new(&graph);

    for use_labels in [false, true] {
        let tb = if use_labels { &tb_labeled } else { &tb_plain };
        let view = tb.view();
        let engine = tb.engine(&view);
        let mut ws = DijkstraWorkspace::new();
        let cache = engine.warm_cut_projections(&cuts, &mut ws);
        let cached = tb.engine(&view).with_projection_cache(&cache);

        for &source in &locations {
            for &target in &locations {
                let uncached = OptionalGfaPathLength::from(engine.distance(source, target));
                let with_cache = OptionalGfaPathLength::from(cached.distance(source, target));
                assert_eq!(
                    with_cache, uncached,
                    "cache changed result ({gfa_path}, labels={use_labels}): {source} -> {target}",
                );
                let oracle_distance = OptionalGfaPathLength::from(
                    oracle
                        .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                        .values()
                        .map(|path| path.length())
                        .next(),
                );
                assert_eq!(
                    uncached, oracle_distance,
                    "engine != oracle ({gfa_path}): {source} -> {target}",
                );
            }
        }
    }
}

/// The LCA-grouped [`BlockQueryEngine::fill_cut_matrix`] must be byte-identical to `dim*dim`
/// independent `start_to_start` calls (source==target ⇒ 0), over both the Dijkstra LCA path
/// (`use_labels=false`) and the labeled path (`use_labels=true`). This is the exactness guarantee
/// that lets the block-cut lift use the batched fill in place of the per-pair fill.
fn assert_fill_cut_matrix_matches(gfa_path: &str, spqr_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr =
        SPQRDecomposition::read_plain_spqr(&graph, BufReader::new(File::open(spqr_path).unwrap()))
            .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let tb_plain = TestBlock::build(&graph, &spqr, block_index, TestLabels::None);
    let tb_labeled = TestBlock::build(&graph, &spqr, block_index, TestLabels::Precompute(0));

    let cuts: Vec<_> = graph.node_indices().collect();
    let dim = 2 * cuts.len();

    for use_labels in [false, true] {
        let tb = if use_labels { &tb_labeled } else { &tb_plain };
        let view = tb.view();
        let engine = tb.engine(&view);

        // Per-pair reference: exactly what the fill replaces.
        let mut reference = vec![MinPlus::infinity(); dim * dim];
        for r in 0..dim {
            let source = cuts[r / 2].into_directed(Orientation::from_index(r % 2).is_forward());
            for c in 0..dim {
                let target = cuts[c / 2].into_directed(Orientation::from_index(c % 2).is_forward());
                reference[r * dim + c] = if source == target {
                    MinPlus::zero()
                } else {
                    engine
                        .start_to_start(source, target)
                        .map(MinPlus::finite)
                        .unwrap_or_else(MinPlus::infinity)
                };
            }
        }

        let fast = engine.fill_cut_matrix(&cuts);
        assert_eq!(
            fast.len(),
            reference.len(),
            "fill_cut_matrix dim mismatch ({gfa_path}, labels={use_labels})",
        );
        for idx in 0..reference.len() {
            assert_eq!(
                fast[idx],
                reference[idx],
                "fill_cut_matrix != per-pair ({gfa_path}, labels={use_labels}) at r={}, c={}",
                idx / dim,
                idx % dim,
            );
        }
    }
}

/// The bidirected walk-reversal identity a batched `target_read` would rest on: reversing a walk
/// `a -> b` yields a walk `b^-1 -> a^-1` whose start-to-start length drops `len(a)` and gains
/// `len(b)`, so `D(a -> b) + len(b) == D(b^-1 -> a^-1) + len(a)` (stated additively to stay in
/// unsigned `GfaPathLength`), with both sides infinite together.
fn assert_reversal_identity(gfa_path: &str, spqr_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr =
        SPQRDecomposition::read_plain_spqr(&graph, BufReader::new(File::open(spqr_path).unwrap()))
            .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let tb = TestBlock::build(&graph, &spqr, block_index, TestLabels::None);
    let view = tb.view();
    let engine = tb.engine(&view);

    let directed: Vec<_> = graph
        .node_indices()
        .flat_map(|node| [node.into_directed(true), node.into_directed(false)])
        .collect();

    for &a in &directed {
        for &b in &directed {
            if a == b {
                continue;
            }
            let forward = engine.start_to_start(a, b);
            let reverse = engine.start_to_start(b.invert(), a.invert());
            let length_a = graph.node_data(a.into_bidirected()).len();
            let length_b = graph.node_data(b.into_bidirected()).len();
            match (forward, reverse) {
                (Some(f), Some(r)) => assert_eq!(
                    f + length_b,
                    r + length_a,
                    "reversal identity broken in {gfa_path}: {a} -> {b} = {f}, {} -> {} = {r}",
                    b.invert(),
                    a.invert(),
                ),
                (None, None) => {}
                _ => panic!(
                    "reversal reachability mismatch in {gfa_path}: {a} -> {b} = {forward:?}, \
                     {} -> {} = {reverse:?}",
                    b.invert(),
                    a.invert(),
                ),
            }
        }
    }
}

#[test]
fn reversal_identity_two_triangles() {
    assert_reversal_identity(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn reversal_identity_hairpin() {
    assert_reversal_identity(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn reversal_identity_rnode_self_flip() {
    assert_reversal_identity(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
    );
}

#[test]
fn reversal_identity_chain5() {
    assert_reversal_identity("test_files/chain5.gfa", "test_files/chain5.spqr");
}

#[test]
fn fill_cut_matrix_matches_per_pair_two_triangles() {
    assert_fill_cut_matrix_matches(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn fill_cut_matrix_matches_per_pair_hairpin() {
    assert_fill_cut_matrix_matches(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn fill_cut_matrix_matches_per_pair_rnode_self_flip() {
    assert_fill_cut_matrix_matches(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
    );
}

#[test]
fn projection_cache_matches_uncached_two_triangles() {
    assert_cached_matches(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn projection_cache_matches_uncached_hairpin() {
    assert_cached_matches(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn projection_cache_matches_uncached_rnode_self_flip() {
    assert_cached_matches(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
    );
}

#[test]
fn labeled_queries_match_oracle_two_triangles() {
    assert_labeled_queries_match_oracle(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn labeled_queries_match_oracle_hairpin() {
    assert_labeled_queries_match_oracle(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn labeled_queries_match_oracle_chain5() {
    assert_labeled_queries_match_oracle("test_files/chain5.gfa", "test_files/chain5.spqr");
}

#[test]
fn labeled_queries_match_oracle_rnode_self_flip() {
    assert_labeled_queries_match_oracle(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
    );
}

/// Same as [`assert_queries_match_oracle`] but puts an S-node **segment tree** on every S-node
/// (threshold 1), so the tree answers both the pole projections and the LCA reads of every S-node
/// LCA — validated against whole-graph Dijkstra over all location pairs.
fn assert_segtree_queries_match_oracle(gfa_path: &str, spqr_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr =
        SPQRDecomposition::read_plain_spqr(&graph, BufReader::new(File::open(spqr_path).unwrap()))
            .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let tb = TestBlock::build_with_segtree_min(&graph, &spqr, block_index, TestLabels::None, 1);
    let view = tb.view();
    assert!(
        (0..view.node_count()).any(|l| view.segtree(l).is_some()),
        "{gfa_path}: the fixture must have at least one S-node for this test to mean anything"
    );
    let engine = tb.engine(&view);

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
            let engine_distance = OptionalGfaPathLength::from(engine.distance(source, target));
            let oracle_distance = OptionalGfaPathLength::from(
                oracle
                    .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                    .values()
                    .map(|path| path.length())
                    .next(),
            );
            assert_eq!(
                engine_distance, oracle_distance,
                "segtree mismatch in {gfa_path}: {source} -> {target}",
            );
        }
    }

    // The build-time cut matrix must agree with the per-pair engine when LCAs are segtree nodes.
    let cuts: Vec<_> = graph.node_indices().collect();
    let matrix = engine.fill_cut_matrix(&cuts);
    let dim = 2 * cuts.len();
    for (i, &a) in cuts.iter().enumerate() {
        for oi in 0..2 {
            for (j, &b) in cuts.iter().enumerate() {
                for oj in 0..2 {
                    let from = a.into_directed(Orientation::from_index(oi).is_forward());
                    let to = b.into_directed(Orientation::from_index(oj).is_forward());
                    let want = if from == to {
                        MinPlus::zero()
                    } else {
                        engine
                            .start_to_start(from, to)
                            .map(MinPlus::finite)
                            .unwrap_or_else(MinPlus::infinity)
                    };
                    assert_eq!(
                        matrix[(2 * i + oi) * dim + 2 * j + oj],
                        want,
                        "segtree cut matrix {from} -> {to}"
                    );
                }
            }
        }
    }
}

#[test]
fn segtree_queries_match_oracle_two_triangles() {
    assert_segtree_queries_match_oracle(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn segtree_queries_match_oracle_hairpin() {
    assert_segtree_queries_match_oracle(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn segtree_queries_match_oracle_chain5() {
    assert_segtree_queries_match_oracle("test_files/chain5.gfa", "test_files/chain5.spqr");
}

#[test]
fn segtree_queries_match_oracle_rnode_self_flip() {
    assert_segtree_queries_match_oracle(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
    );
}

#[test]
fn queries_match_oracle_two_triangles() {
    assert_queries_match_oracle(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn queries_match_oracle_hairpin() {
    assert_queries_match_oracle(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
    );
}
