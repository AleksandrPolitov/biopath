use std::{fs::File, io::BufReader};

use bidirected_adjacency_array::io::gfa1::GfaNodeData;
use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    io::gfa1::{PlainGfaEdgeData, PlainGfaNodeData},
};
use spqr_tree::{decomposition::SPQRDecomposition, graph::StaticGraph};

use super::{BuildOptions, DistanceIndex, LABEL_MIN_DIRECTED_NODES, SmallBlocks, Tiers};
use crate::{
    block_engine::test_support::{TestBlock, TestLabels},
    dijkstra::{GfaDijkstra, location_index::single::SingleGfaLocationIndex},
    location::GfaLocation,
    path::OptionalGfaPathLength,
};

type Graph = BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData>;

/// The default options with a segment tree on every S-node (threshold 1) and the hub-label
/// threshold `label_min`.
fn segtree_everywhere(label_min: usize) -> BuildOptions<'static> {
    BuildOptions {
        tiers: Tiers {
            label_min,
            segtree_min: 1,
            ..Tiers::default()
        },
        ..BuildOptions::default()
    }
}

/// The default options with hub labels on every SPQR node (threshold 1).
fn labels_everywhere() -> BuildOptions<'static> {
    BuildOptions::with_label_threshold(1)
}

#[test]
fn all_pairs_match_oracle_segtree_tiny1_multi_block() {
    assert_all_pairs_match_oracle_with(
        "test_files/tiny1.gfa",
        "test_files/tiny1.spqr",
        &segtree_everywhere(LABEL_MIN_DIRECTED_NODES),
    );
}

#[test]
fn all_pairs_match_oracle_segtree_two_triangles() {
    assert_all_pairs_match_oracle_with(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
        &segtree_everywhere(LABEL_MIN_DIRECTED_NODES),
    );
}

#[test]
fn all_pairs_match_oracle_segtree_hairpin() {
    assert_all_pairs_match_oracle_with(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
        &segtree_everywhere(LABEL_MIN_DIRECTED_NODES),
    );
}

#[test]
fn all_pairs_match_oracle_segtree_chain5() {
    assert_all_pairs_match_oracle_with(
        "test_files/chain5.gfa",
        "test_files/chain5.spqr",
        &segtree_everywhere(LABEL_MIN_DIRECTED_NODES),
    );
}

#[test]
fn all_pairs_match_oracle_segtree_rnode_self_flip() {
    assert_all_pairs_match_oracle_with(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
        &segtree_everywhere(LABEL_MIN_DIRECTED_NODES),
    );
}

#[test]
fn all_pairs_match_oracle_segtree_two_excursions() {
    assert_all_pairs_match_oracle_with(
        "test_files/twoflip.gfa",
        "test_files/twoflip.spqr",
        &segtree_everywhere(LABEL_MIN_DIRECTED_NODES),
    );
}

/// Segment trees with a label threshold of 1 as well: every S-node gets a tree, every other node
/// labels — the mixed dispatch.
#[test]
fn all_pairs_match_oracle_segtree_and_labels_two_triangles() {
    assert_all_pairs_match_oracle_with(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
        &segtree_everywhere(1),
    );
}

#[test]
fn file_round_trip_segtree_tiny1_multi_block() {
    assert_file_round_trip(
        "test_files/tiny1.gfa",
        "test_files/tiny1.spqr",
        &segtree_everywhere(LABEL_MIN_DIRECTED_NODES),
    );
}

#[test]
fn file_round_trip_segtree_rnode_self_flip() {
    assert_file_round_trip(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
        &segtree_everywhere(1),
    );
}

fn read_graph(path: &str) -> Graph {
    BidirectedAdjacencyArray::read_gfa1(BufReader::new(File::open(path).unwrap())).unwrap()
}

/// The full engine must match whole-graph Dijkstra for every location pair, including cross-block
/// routes through cut nodes.
fn assert_all_pairs_match_oracle(gfa_path: &str, spqr_path: &str) {
    assert_all_pairs_match_oracle_with(gfa_path, spqr_path, &BuildOptions::default());
}

/// Like [`assert_all_pairs_match_oracle`], with the given build options. A hub-label threshold of
/// `1` labels every SPQR node, so it forces the labeled OutsideDp child-fill, the reuse of those
/// oracles as query oracles, and the post-build memory reclaim (dropped inside matrices, emptied
/// labeled LCA skeletons) — all cross-checked against whole-graph Dijkstra.
fn assert_all_pairs_match_oracle_with(gfa_path: &str, spqr_path: &str, options: &BuildOptions) {
    let graph = read_graph(gfa_path);
    let spqr =
        SPQRDecomposition::read_plain_spqr(&graph, BufReader::new(File::open(spqr_path).unwrap()))
            .unwrap();
    let engine = DistanceIndex::build(&graph, &spqr, options);

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
            let index_distance = OptionalGfaPathLength::from(engine.distance(source, target));
            let oracle_distance = OptionalGfaPathLength::from(
                oracle
                    .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                    .values()
                    .map(|path| path.length())
                    .next(),
            );
            assert_eq!(
                index_distance, oracle_distance,
                "block-cut lifting mismatch in {gfa_path}: {source} -> {target}",
            );
        }
    }
}

#[test]
fn all_pairs_match_oracle_tiny1_multi_block() {
    // tiny1 is a chain of five triangle blocks joined by cut nodes c, e, g, i: exercises the
    // cross-block lift.
    assert_all_pairs_match_oracle("test_files/tiny1.gfa", "test_files/tiny1.spqr");
}

#[test]
fn all_pairs_match_oracle_two_triangles_single_block() {
    // A single block (no cut nodes): the cross-block layer is empty and the engine falls through to
    // the intra-block distance.
    assert_all_pairs_match_oracle(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn all_pairs_match_oracle_hairpin_single_block() {
    assert_all_pairs_match_oracle(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn all_pairs_match_oracle_chain_unequal_lengths() {
    // Nested separation pairs with DIFFERENT poles ({N0,N2} vs {N2,N4}) and UNEQUAL node lengths:
    // regression for the start-to-start reverse-complement length correction.
    assert_all_pairs_match_oracle("test_files/chain5.gfa", "test_files/chain5.spqr");
}

#[test]
fn all_pairs_match_oracle_rnode_self_flip() {
    // A multigraph with R-nodes and orientation flips: regression for the full 4x4 inside/outside
    // pole-state matrices (pole self-flips through a subtree / through the outside).
    assert_all_pairs_match_oracle(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
    );
}

#[test]
fn all_pairs_match_oracle_two_excursions_out_of_a_block() {
    // A same-block, same-strand query whose only route leaves the block TWICE (flip at the downstream
    // cut, run the block on the - strand, flip at the upstream cut): regression for the lift path's
    // former at-most-one-excursion `same_block_cross`, which returned `None` for `Na+ -> Nb+`.
    assert_all_pairs_match_oracle("test_files/twoflip.gfa", "test_files/twoflip.spqr");
}

#[test]
fn all_pairs_match_oracle_labeled_two_excursions_out_of_a_block() {
    assert_all_pairs_match_oracle_with(
        "test_files/twoflip.gfa",
        "test_files/twoflip.spqr",
        &labels_everywhere(),
    );
}

/// Labels stored wide, as a skeleton of more than `NARROW_LABEL_MAX_STATES` states stores them,
/// answer every pair like whole-graph Dijkstra. The fixtures are tiny, so the test stores every
/// label of this thread's builds wide (and checks that the image changes), with every block a
/// labeled record block.
#[test]
fn all_pairs_match_oracle_with_wide_labels() {
    use crate::skeleton_index::WIDE_LABELS;
    let options = BuildOptions {
        small_blocks: SmallBlocks {
            enabled: false,
            ..SmallBlocks::default()
        },
        ..labels_everywhere()
    };
    for (gfa, spqr) in [
        ("test_files/twoflip.gfa", "test_files/twoflip.spqr"),
        (
            "test_files/rnode_selfflip.gfa",
            "test_files/rnode_selfflip.spqr",
        ),
    ] {
        let graph = read_graph(gfa);
        let image = || {
            let spqr = BufReader::new(File::open(spqr).unwrap());
            let spqr = SPQRDecomposition::read_plain_spqr(&graph, spqr).unwrap();
            DistanceIndex::build(&graph, spqr, &options)
                .image()
                .as_bytes()
                .to_vec()
        };
        let narrow = image();
        WIDE_LABELS.set(true);
        assert!(narrow != image(), "{gfa}: the labels are stored wide");
        assert_all_pairs_match_oracle_with(gfa, spqr, &options);
        WIDE_LABELS.set(false);
    }
}

/// A two-excursion fixture whose middle block has two cuts, each flip-capable through its pendant
/// block, under two rootings of the block-cut tree: routes must leave the middle block any number
/// of times, through its `M_aug` and (when it is not the root) its `gadget_up` and lift steps.
fn assert_twoflip_rooted_all_pairs_match_oracle(spqr_path: &str) {
    let graph = BidirectedAdjacencyArray::<u32, PlainGfaNodeData, PlainGfaEdgeData>::read_gfa1(
        BufReader::new(File::open("test_files/twoflip_hub.gfa").unwrap()),
    )
    .unwrap();
    let spqr =
        SPQRDecomposition::read_plain_spqr(&graph, BufReader::new(File::open(spqr_path).unwrap()))
            .unwrap();
    let engine = DistanceIndex::build(&graph, &spqr, &BuildOptions::with_label_threshold(2));

    let mut oracle = GfaDijkstra::new(&graph);
    let locations: Vec<GfaLocation<u32>> = graph
        .node_indices()
        .flat_map(|node| [node.into_directed(true), node.into_directed(false)])
        .flat_map(|d| [GfaLocation::new(d, 0.into()), GfaLocation::new(d, 1.into())])
        .collect();
    for &source in &locations {
        for &target in &locations {
            let want = OptionalGfaPathLength::from(
                oracle
                    .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                    .values()
                    .map(|path| path.length())
                    .next(),
            );
            let got = OptionalGfaPathLength::from(engine.distance(source, target));
            assert_eq!(got, want, "mismatch ({spqr_path}): {source} -> {target}");
        }
    }
}

#[test]
fn all_pairs_match_oracle_two_excursions_through_a_root_block() {
    // The middle block is the component's root block (no parent cut).
    assert_twoflip_rooted_all_pairs_match_oracle("test_files/twoflip_hub.spqr");
}

#[test]
fn all_pairs_match_oracle_two_excursions_through_a_non_root_block() {
    // A pendant block is listed first, so it is the root and the middle block hangs below cut `Nz`.
    assert_twoflip_rooted_all_pairs_match_oracle("test_files/twoflip_hub_nonroot.spqr");
}

// Threshold 1: label every SPQR node so the labeled build/query path (and the post-build memory
// reclaim) is exercised on the multi-block, chain, and R-node/flip fixtures, all vs the oracle.
#[test]
fn all_pairs_match_oracle_labeled_tiny1_multi_block() {
    assert_all_pairs_match_oracle_with(
        "test_files/tiny1.gfa",
        "test_files/tiny1.spqr",
        &labels_everywhere(),
    );
}

#[test]
fn all_pairs_match_oracle_labeled_chain_unequal_lengths() {
    assert_all_pairs_match_oracle_with(
        "test_files/chain5.gfa",
        "test_files/chain5.spqr",
        &labels_everywhere(),
    );
}

#[test]
fn all_pairs_match_oracle_labeled_rnode_self_flip() {
    assert_all_pairs_match_oracle_with(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
        &labels_everywhere(),
    );
}

/// A real 395-node genome-graph block (single biconnected block). All-pairs is infeasible here
/// (~2.5M queries x per-query index rebuilds), so we build the block engine once and sample a large
/// number of deterministic random queries against the oracle.
#[test]
fn sampled_queries_match_oracle_gca_block() {
    let graph = BidirectedAdjacencyArray::<u32, PlainGfaNodeData, PlainGfaEdgeData>::read_gfa1(
        BufReader::new(File::open("test_files/GCA-41.clean.largest-block.gfa").unwrap()),
    )
    .unwrap();
    let spqr = SPQRDecomposition::read_plain_spqr(
        &graph,
        BufReader::new(File::open("test_files/GCA-41.clean.largest-block.spqr").unwrap()),
    )
    .unwrap();
    assert_eq!(
        spqr.block_count(),
        1,
        "the largest block is a single biconnected block"
    );

    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let tb = TestBlock::build(&graph, &spqr, block_index, TestLabels::None);
    let view = tb.view();
    let engine = tb.engine(&view);

    let mut oracle = GfaDijkstra::new(&graph);
    let nodes: Vec<_> = graph.node_indices().collect();

    // Deterministic LCG so the sample is reproducible.
    let mut state: u64 = 0x5EED_1234;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };

    for _ in 0..4000 {
        let s = nodes[next() % nodes.len()];
        let t = nodes[next() % nodes.len()];
        let sdir = s.into_directed(next() % 2 == 0);
        let tdir = t.into_directed(next() % 2 == 0);
        let slen = graph.node_data(s).sequence().len();
        let tlen = graph.node_data(t).sequence().len();
        let source = GfaLocation::new(sdir, (next() % (slen + 1)).into());
        let target = GfaLocation::new(tdir, (next() % (tlen + 1)).into());

        let got = OptionalGfaPathLength::from(engine.distance(source, target));
        let want = OptionalGfaPathLength::from(
            oracle
                .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                .values()
                .map(|path| path.length())
                .next(),
        );
        assert_eq!(got, want, "mismatch for {source} -> {target}");
    }
}

/// The on-disk round trip: build the engine (into its image), write the image to a file, load it
/// back, and check every location pair against the in-process engine AND
/// whole-graph Dijkstra. This is the `biopath index` / `biopath query --index-in` contract.
fn assert_file_round_trip(gfa_path: &str, spqr_path: &str, options: &BuildOptions) {
    let graph = read_graph(gfa_path);
    let spqr =
        SPQRDecomposition::read_plain_spqr(&graph, BufReader::new(File::open(spqr_path).unwrap()))
            .unwrap();
    let built = DistanceIndex::build(&graph, &spqr, options);
    // Unique per call: two tests round-tripping the same fixture run in parallel threads.
    static CALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "bpi-roundtrip-{}-{}-{}",
        std::process::id(),
        CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        gfa_path.replace('/', "_")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("index.bpi");
    built.write_to(&path).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len() as usize,
        built.image().used()
    );
    for entry in built.image().toc() {
        assert!(
            crate::sections::is_live(entry.id),
            "section id {} is not live",
            entry.id
        );
    }

    let mut oracle = GfaDijkstra::new(&graph);
    let locations: Vec<GfaLocation<u32>> = graph
        .node_indices()
        .flat_map(|node| [node.into_directed(true), node.into_directed(false)])
        .flat_map(|d| [GfaLocation::new(d, 0.into()), GfaLocation::new(d, 1.into())])
        .collect();
    let loaded = DistanceIndex::<u32>::open(&path).unwrap();
    assert_eq!(loaded.node_count(), graph.node_count());
    // Names resolve through the image's name table.
    for node in graph.node_indices() {
        let name = graph.node_name(node);
        assert_eq!(loaded.node_by_name(&name), Some(node), "name {name}");
        assert_eq!(loaded.graph_view().node_name(node), name);
    }
    assert_eq!(loaded.node_by_name("no-such-node"), None);
    for &source in &locations {
        for &target in &locations {
            let want = OptionalGfaPathLength::from(
                oracle
                    .shortest_paths(source, &SingleGfaLocationIndex::new_target(target))
                    .values()
                    .map(|path| path.length())
                    .next(),
            );
            let in_process = OptionalGfaPathLength::from(built.distance(source, target));
            let from_file = OptionalGfaPathLength::from(loaded.distance(source, target));
            assert_eq!(
                in_process, want,
                "in-process != oracle ({gfa_path}): {source} -> {target}"
            );
            assert_eq!(
                from_file, want,
                "loaded != oracle ({gfa_path}): {source} -> {target}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn file_round_trip_tiny1_multi_block() {
    assert_file_round_trip(
        "test_files/tiny1.gfa",
        "test_files/tiny1.spqr",
        &BuildOptions::default(),
    );
}

#[test]
fn file_round_trip_labeled_two_excursions() {
    assert_file_round_trip(
        "test_files/twoflip.gfa",
        "test_files/twoflip.spqr",
        &labels_everywhere(),
    );
}

#[test]
fn file_round_trip_rnode_self_flip() {
    assert_file_round_trip(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
        &labels_everywhere(),
    );
}
