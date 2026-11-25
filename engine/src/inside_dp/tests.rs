use std::{fs::File, io::BufReader};

use crate::gfa_graph_extensions::GfaNodeDataExt;
use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, NodeIndex},
    io::gfa1::{GfaNodeData, PlainGfaEdgeData, PlainGfaNodeData},
};
use spqr_tree::{decomposition::SPQRDecomposition, graph::StaticGraph};

use super::InsideDp;
use crate::{
    block_spqr_tree::BlockSpqrTree,
    dijkstra::{GfaDijkstra, location_index::single::SingleGfaLocationIndex},
    location::GfaLocation,
    min_plus::MinPlusMatrix,
    orientation::Orientation,
    path::GfaPathLength,
    pole_lift::{Pole, slot},
};

use Orientation::{Forward as F, Reverse as R};

type Graph = BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData>;

fn read_graph(path: &str) -> Graph {
    BidirectedAdjacencyArray::read_gfa1(BufReader::new(File::open(path).unwrap())).unwrap()
}

fn read_spqr<'g>(graph: &'g Graph, path: &str) -> SPQRDecomposition<'g, Graph> {
    SPQRDecomposition::read_plain_spqr(graph, BufReader::new(File::open(path).unwrap())).unwrap()
}

fn index_of(graph: &Graph, name: &str) -> NodeIndex<u32> {
    graph
        .node_indices()
        .find(|&i| graph.node_data(i).name() == name)
        .unwrap()
}

/// Flip-baking regression gate: `compute_with_flips` with an EMPTY closure map must be byte-identical
/// to plain `compute` — so the whole baking path is a true no-op unless closures are supplied (the
/// `SPQR_BAKE_FLIPS`-off default).
#[test]
fn compute_with_flips_empty_matches_compute() {
    use crate::{fast_hash::FastMap, min_plus::Mat2};

    let graph = read_graph("test_files/two_triangles.gfa");
    let spqr = read_spqr(&graph, "test_files/two_triangles.spqr");
    let (block_index, _) = spqr.iter_blocks().next().unwrap();

    let base = InsideDp::compute(&graph, BlockSpqrTree::new(&spqr, block_index));
    let empty: FastMap<NodeIndex<u32>, Mat2<u32>> = FastMap::default();
    let baked =
        InsideDp::compute_with_flips(&graph, BlockSpqrTree::new(&spqr, block_index), &empty);

    let n = base.block_tree().node_count();
    assert_eq!(n, baked.block_tree().node_count());
    for local in 0..n {
        assert_eq!(
            base.inside_matrix(local),
            baked.inside_matrix(local),
            "empty-closure baking changed inside matrix at local {local}",
        );
    }
}

/// A baked flip closure can only *add* real routes, so every inside distance can only decrease
/// (min-plus monotonicity) — never increase — and a cheap flip on a real vertex actually enters the
/// composition (at least one entry strictly drops). This is the soundness invariant the whole scheme
/// relies on: baking never corrupts a distance, it only exposes flip routes.
#[test]
fn baking_a_flip_only_lowers_inside_distances() {
    use crate::{
        fast_hash::FastMap,
        min_plus::{Mat2, MinPlus},
    };

    let graph = read_graph("test_files/two_triangles.gfa");
    let spqr = read_spqr(&graph, "test_files/two_triangles.spqr");
    let (block_index, _) = spqr.iter_blocks().next().unwrap();

    let base = InsideDp::compute(&graph, BlockSpqrTree::new(&spqr, block_index));

    // A cheap flip closure (z^+ <-> z^- both cost 1) on EVERY segment, so the bake is guaranteed to
    // reach some node's poles regardless of which vertices this fixture's skeletons introduce.
    let closure = Mat2::from_rows([
        [MinPlus::infinity(), MinPlus::finite_usize(1)],
        [MinPlus::finite_usize(1), MinPlus::infinity()],
    ]);
    let mut closures: FastMap<NodeIndex<u32>, Mat2<u32>> = FastMap::default();
    for z in graph.node_indices() {
        closures.insert(z, closure);
    }
    let baked =
        InsideDp::compute_with_flips(&graph, BlockSpqrTree::new(&spqr, block_index), &closures);

    let key = |m: MinPlus<u32>| {
        m.into_option()
            .map(|p| p.into_usize())
            .unwrap_or(usize::MAX)
    };
    let mut any_decrease = false;
    for local in 0..base.block_tree().node_count() {
        if let (Some(b), Some(k)) = (base.inside_matrix(local), baked.inside_matrix(local)) {
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        key(k.get(i, j)) <= key(b.get(i, j)),
                        "baking INCREASED an inside distance at local {local} ({i},{j})",
                    );
                    any_decrease |= key(k.get(i, j)) < key(b.get(i, j));
                }
            }
        }
    }
    assert!(
        any_decrease,
        "the cheap baked flip never entered the composition"
    );
}

/// The `s -> t` block entry of a full 4x4 inside pole-state matrix.
fn entry(matrix: &MinPlusMatrix<u32, 4>, from: Orientation, to: Orientation) -> Option<usize> {
    matrix
        .get(slot(Pole::Source, from), slot(Pole::Target, to))
        .into_option()
        .map(GfaPathLength::into_usize)
}

/// The parallel depth-level driver in [`InsideDp::compute`] must produce byte-identical inside
/// matrices to the serial postorder reference — over S/P chains and R-node fixtures.
fn assert_parallel_inside_matches_serial(gfa: &str, spqr: &str) {
    let graph = read_graph(gfa);
    let decomp = read_spqr(&graph, spqr);
    let (block_index, _) = decomp.iter_blocks().next().unwrap();
    let parallel = InsideDp::compute(&graph, BlockSpqrTree::new(&decomp, block_index));
    let serial = InsideDp::compute_serial(&graph, BlockSpqrTree::new(&decomp, block_index));
    assert_eq!(
        parallel.inside, serial.inside,
        "parallel != serial inside DP for {gfa}"
    );
}

#[test]
fn parallel_inside_matches_serial_two_triangles() {
    assert_parallel_inside_matches_serial(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
    );
}

#[test]
fn parallel_inside_matches_serial_chain5() {
    assert_parallel_inside_matches_serial("test_files/chain5.gfa", "test_files/chain5.spqr");
}

#[test]
fn parallel_inside_matches_serial_rnode_selfflip() {
    assert_parallel_inside_matches_serial(
        "test_files/rnode_selfflip.gfa",
        "test_files/rnode_selfflip.spqr",
    );
}

/// Hand-computed inside matrices for the all-forward two-triangles graph (every segment length 2).
#[test]
fn two_triangles_inside_matrices_match_hand_computation() {
    let graph = read_graph("test_files/two_triangles.gfa");
    let spqr = read_spqr(&graph, "test_files/two_triangles.spqr");
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let block_tree = BlockSpqrTree::new(&spqr, block_index);
    let dp = InsideDp::compute(&graph, block_tree);

    // Locals: 0 = S1 (root), 1 = P0, 2 = S2. Reference poles of P0 and S2 are (a, b).

    // S2 = triangle a-d-b, only forward links a+ -> d+ -> b+. a+ -> b+ costs len(a)+len(d) = 4.
    let s2 = dp.inside_matrix(2).unwrap();
    assert_eq!(entry(s2, F, F), Some(4));
    assert_eq!(entry(s2, F, R), None);
    assert_eq!(entry(s2, R, F), None);
    assert_eq!(entry(s2, R, R), None);

    // P0 = direct edge a+ -> b+ (cost len(a) = 2) in parallel with the S2 child (cost 4); min is 2.
    let p0 = dp.inside_matrix(1).unwrap();
    assert_eq!(entry(p0, F, F), Some(2));
    assert_eq!(entry(p0, F, R), None);
    assert_eq!(entry(p0, R, F), None);
    assert_eq!(entry(p0, R, R), None);

    // The root has no reference poles and therefore no inside matrix.
    assert!(dp.inside_matrix(0).is_none());
}

/// For a single-block, single-component graph, the root skeleton (with children folded in)
/// represents the whole graph, so its start-to-start distances must equal whole-graph Dijkstra.
fn assert_root_skeleton_matches_whole_graph(gfa_path: &str, spqr_path: &str, segments: &[&str]) {
    let graph = read_graph(gfa_path);
    let spqr = read_spqr(&graph, spqr_path);
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let block_tree = BlockSpqrTree::new(&spqr, block_index);
    let dp = InsideDp::compute(&graph, block_tree);
    let root = dp.block_tree().root_local();

    let mut oracle = GfaDijkstra::new(&graph);

    for &source_name in segments {
        for source_orientation in Orientation::all() {
            let source: DirectedNodeIndex<u32> =
                index_of(&graph, source_name).into_directed(source_orientation.is_forward());
            let skeleton = dp.skeleton_shortest_paths(root, source);

            for &target_name in segments {
                for target_orientation in Orientation::all() {
                    let target: DirectedNodeIndex<u32> = index_of(&graph, target_name)
                        .into_directed(target_orientation.is_forward());

                    let via_skeleton = skeleton.get(&target).map(|&d| d.into_usize());
                    let via_whole_graph = oracle
                        .shortest_paths(
                            GfaLocation::new(source, 0.into()),
                            &SingleGfaLocationIndex::new_target(GfaLocation::new(target, 0.into())),
                        )
                        .values()
                        .map(|path| path.length().into_usize())
                        .next();

                    assert_eq!(
                        via_skeleton,
                        via_whole_graph,
                        "distance mismatch in {gfa_path}: {source_name}{} -> {target_name}{}",
                        source_orientation.sign(),
                        target_orientation.sign(),
                    );
                }
            }
        }
    }
}

#[test]
fn two_triangles_root_skeleton_matches_whole_graph() {
    // Root skeleton spans segments a, b, c (d is interior to the S2 child).
    assert_root_skeleton_matches_whole_graph(
        "test_files/two_triangles.gfa",
        "test_files/two_triangles.spqr",
        &["a", "b", "c"],
    );
}

#[test]
fn two_triangles_hairpin_root_skeleton_matches_whole_graph() {
    // Same undirected structure (so the same .spqr applies) but a c+ -> b- flip makes reverse
    // orientations and dead ends reachable, exercising the orientation bookkeeping end to end.
    assert_root_skeleton_matches_whole_graph(
        "test_files/two_triangles_hairpin.gfa",
        "test_files/two_triangles.spqr",
        &["a", "b", "c"],
    );
}

/// The full 4x4 inside matrix must satisfy the start-to-start reverse-complement identity:
/// `D(A^a -> B^b) + len(B) == D(B^{¬b} -> A^{¬a}) + len(A)` for every pair of pole-states.
#[test]
fn inside_matrices_are_reverse_complement_consistent() {
    for gfa_path in [
        "test_files/two_triangles.gfa",
        "test_files/two_triangles_hairpin.gfa",
    ] {
        let graph = read_graph(gfa_path);
        let spqr = read_spqr(&graph, "test_files/two_triangles.spqr");
        let (block_index, _) = spqr.iter_blocks().next().unwrap();
        let block_tree = BlockSpqrTree::new(&spqr, block_index);
        let dp = InsideDp::compute(&graph, block_tree);

        for local in 0..dp.block_tree().node_count() {
            let Some((s, t)) = dp.block_tree().node(local).reference_poles() else {
                continue;
            };
            let matrix = dp.inside_matrix(local).unwrap();
            let seg = |pole: Pole| if pole == Pole::Source { s } else { t };
            let len = |pole: Pole| {
                <PlainGfaNodeData as GfaNodeDataExt<u32>>::len(graph.node_data(seg(pole)))
                    .into_usize()
            };

            for a_pole in Pole::all() {
                for a in Orientation::all() {
                    for b_pole in Pole::all() {
                        for b in Orientation::all() {
                            let forward = matrix
                                .get(slot(a_pole, a), slot(b_pole, b))
                                .into_option()
                                .map(GfaPathLength::into_usize);
                            let reverse = matrix
                                .get(slot(b_pole, b.invert()), slot(a_pole, a.invert()))
                                .into_option()
                                .map(GfaPathLength::into_usize);
                            assert_eq!(
                                forward.map(|d| d + len(b_pole)),
                                reverse.map(|d| d + len(a_pole)),
                                "rc inconsistency at local {local} in {gfa_path}",
                            );
                        }
                    }
                }
            }
        }
    }
}
