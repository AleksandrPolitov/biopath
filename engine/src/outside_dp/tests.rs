use std::{fs::File, io::BufReader};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::NodeIndex,
    io::gfa1::{PlainGfaEdgeData, PlainGfaNodeData},
};
use spqr_tree::decomposition::SPQRDecomposition;

use super::OutsideDp;
use crate::pole_lift::slot;
use crate::{
    block_engine::Tiers,
    block_spqr_tree::BlockSpqrTree,
    dijkstra::{GfaDijkstra, location_index::single::SingleGfaLocationIndex},
    inside_dp::InsideDp,
    location::GfaLocation,
    min_plus::{Mat2, MinPlus, MinPlusMatrix},
    orientation::Orientation,
    path::GfaPathLength,
    pole_lift::Pole,
};

use Orientation::Forward as F;

type Graph = BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData>;

fn read_graph(path: &str) -> Graph {
    BidirectedAdjacencyArray::read_gfa1(BufReader::new(File::open(path).unwrap())).unwrap()
}

fn read_spqr<'g>(graph: &'g Graph, path: &str) -> SPQRDecomposition<'g, Graph> {
    SPQRDecomposition::read_plain_spqr(graph, BufReader::new(File::open(path).unwrap())).unwrap()
}

/// The `s -> t` block entry of a full 4x4 outside matrix.
fn st_entry(m: &MinPlusMatrix<u32, 4>, from: Orientation, to: Orientation) -> Option<usize> {
    m.get(slot(Pole::Source, from), slot(Pole::Target, to))
        .into_option()
        .map(GfaPathLength::into_usize)
}

fn om_entry(m: &Mat2<u32>, from: Orientation, to: Orientation) -> Option<usize> {
    m.get(from.index(), to.index())
        .into_option()
        .map(GfaPathLength::into_usize)
}

/// The whole-graph distance matrix between two segments, via the brute-force oracle.
fn whole_graph_matrix(
    oracle: &mut GfaDijkstra<u32, PlainGfaNodeData, PlainGfaEdgeData>,
    from: NodeIndex<u32>,
    to: NodeIndex<u32>,
) -> Mat2<u32> {
    let mut matrix = Mat2::infinity();
    for from_orientation in Orientation::all() {
        for to_orientation in Orientation::all() {
            let distance = oracle
                .shortest_paths(
                    GfaLocation::new(from.into_directed(from_orientation.is_forward()), 0.into()),
                    &SingleGfaLocationIndex::new_target(GfaLocation::new(
                        to.into_directed(to_orientation.is_forward()),
                        0.into(),
                    )),
                )
                .values()
                .map(|path| path.length())
                .next();
            if let Some(distance) = distance {
                matrix.set(
                    from_orientation.index(),
                    to_orientation.index(),
                    MinPlus::finite(distance),
                );
            }
        }
    }
    matrix
}

#[test]
fn two_triangles_outside_matrices_and_outside_wins() {
    let graph = read_graph("test_files/two_triangles.gfa");
    let spqr = read_spqr(&graph, "test_files/two_triangles.spqr");
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let block_tree = BlockSpqrTree::new(&spqr, block_index);
    let inside = InsideDp::compute(&graph, block_tree);
    let outside = OutsideDp::compute(&inside);

    // Locals: 0 = S1 (root), 1 = P0, 2 = S2. The s -> t outside route for both is 2.
    assert_eq!(st_entry(outside.outside_matrix(1), F, F), Some(2));

    // D_out(S2): the route outside S2 (the direct a-b edge, reached via P0) costs 2, beating the
    // inside route a-d-b (cost 4). This is the "outside route wins" case.
    assert_eq!(st_entry(outside.outside_matrix(2), F, F), Some(2));
    assert_eq!(st_entry(inside.inside_matrix(2).unwrap(), F, F), Some(4));
    assert!(
        st_entry(outside.outside_matrix(2), F, F)
            < st_entry(inside.inside_matrix(2).unwrap(), F, F),
        "the outside route must beat the inside route for S2",
    );
}

/// Core soundness invariant: for every non-root node, `inside ⊕ D_out(s->t)` equals the whole-graph
/// pole distance `s -> t` in every orientation combination.
fn assert_combined_matches_whole_graph(gfa_path: &str, spqr_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr = read_spqr(&graph, spqr_path);
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let block_tree = BlockSpqrTree::new(&spqr, block_index);
    let inside = InsideDp::compute(&graph, block_tree);
    let outside = OutsideDp::compute(&inside);

    let mut oracle = GfaDijkstra::new(&graph);

    for local in 0..inside.block_tree().node_count() {
        let Some((s, t)) = inside.block_tree().node(local).reference_poles() else {
            continue;
        };
        let inside_m = inside.inside_matrix(local).unwrap();
        let outside_m = outside.outside_matrix(local);
        let expected = whole_graph_matrix(&mut oracle, s, t);

        for from in Orientation::all() {
            for to in Orientation::all() {
                let combined = inside_m
                    .get(slot(Pole::Source, from), slot(Pole::Target, to))
                    .oplus(outside_m.get(slot(Pole::Source, from), slot(Pole::Target, to)))
                    .into_option()
                    .map(GfaPathLength::into_usize);
                assert_eq!(
                    combined,
                    om_entry(&expected, from, to),
                    "combined != whole-graph at local {local} (s{} -> t{}) in {gfa_path}",
                    from.sign(),
                    to.sign(),
                );
            }
        }
    }
}

/// The labeled outside DP ([`OutsideDp::compute_with_labels`]) must produce byte-identical outside
/// matrices to the plain Dijkstra path ([`OutsideDp::compute`]). Threshold `1` forces every parent
/// through the label-lookup child-fill, so this exercises the labeled path everywhere.
fn assert_labeled_outside_matches_plain(gfa_path: &str, spqr_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr = read_spqr(&graph, spqr_path);
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let block_tree = BlockSpqrTree::new(&spqr, block_index);
    let inside = InsideDp::compute(&graph, block_tree);

    let plain = OutsideDp::compute(&inside);
    // The default S-node segment-tree threshold keeps the small S-node parents off the cycle
    // closed form, which stashes no labels.
    let tiers = Tiers {
        label_min: 1,
        ..Default::default()
    };
    let (labeled, labels) = OutsideDp::compute_with_labels(&inside, &tiers);

    for local in 0..inside.block_tree().node_count() {
        // Every node's outside is filled at build, so the labeled path must reproduce the plain
        // Dijkstra path everywhere.
        assert_eq!(
            plain.outside_matrix(local),
            labeled.outside_matrix(local),
            "labeled outside != plain outside at local {local} in {gfa_path}",
        );
    }
    // Every parent with children must have been labeled and stashed.
    assert_eq!(labels.len(), inside.block_tree().node_count());
    for (local, stashed) in labels.iter().enumerate() {
        if !inside.block_tree().tree().children(local).is_empty() {
            assert!(
                stashed.is_some(),
                "expected stashed PLL labels at parent local {local} in {gfa_path}",
            );
        }
    }
}

#[test]
fn labeled_outside_matches_plain_two_triangles() {
    assert_labeled_outside_matches_plain(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn labeled_outside_matches_plain_hairpin() {
    assert_labeled_outside_matches_plain(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn labeled_outside_matches_plain_rnode_selfflip() {
    assert_labeled_outside_matches_plain(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
    );
}

#[test]
fn labeled_outside_matches_plain_chain5() {
    assert_labeled_outside_matches_plain("test_files/chain5.gfa", "test_files/chain5.spqr");
}

#[test]
fn combined_matches_whole_graph_two_triangles() {
    assert_combined_matches_whole_graph(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn combined_matches_whole_graph_hairpin() {
    assert_combined_matches_whole_graph(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
    );
}
