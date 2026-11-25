use std::{collections::HashMap, fs::File, io::BufReader};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, NodeIndex},
    io::gfa1::{GfaNodeData, PlainGfaEdgeData, PlainGfaNodeData},
};
use spqr_tree::{decomposition::SPQRDecomposition, graph::StaticGraph};

use super::{Pole, PoleLift, PoleVector, parent_transitions_parallel, parent_transitions_serial};
use crate::{
    block_spqr_tree::BlockSpqrTree, inside_dp::InsideDp, min_plus::MinPlus,
    orientation::Orientation, path::GfaPathLength, rooted_tree::RootedTree,
    skeleton::DijkstraWorkspace,
};

type Graph = BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData>;

/// Stage-1 flip-baking gate: a flip closure baked with [`super::bake_flip_edge`] is a real edge that
/// the inside-DP kernel ([`pole_state_matrix`]) finds and folds into the node's `4x4` relation. A bare
/// node cannot flip (the identity forbids a free `z^+ -> z^-`), so the baked entry is the *only* source
/// of the flip distance here — proving the primitive works end-to-end through the actual search, not
/// just as a map insert. (Composition of an *in-graph* flip up the tree is already covered by the
/// `two_triangles_hairpin` fixture in `query/tests.rs`.)
#[test]
fn baked_flip_edge_is_found_and_folded_by_the_kernel() {
    use bidirected_adjacency_array::index::NodeIndex;

    use super::{Pole, bake_flip_edge, slot};
    use crate::{
        fast_hash::FastMap,
        inside_dp::{SkeletonAdjacency, pole_state_matrix},
        min_plus::Mat2,
        skeleton::Skeleton,
    };
    use Orientation::{Forward as F, Reverse as R};

    let z: NodeIndex<u32> = NodeIndex::from_usize(0);
    let t: NodeIndex<u32> = NodeIndex::from_usize(1);

    // A skeleton with one real link z^+ -> t^+ (and its reverse complement), so both poles exist but
    // neither pole can flip on its own.
    let mut adjacency: SkeletonAdjacency<u32> = FastMap::default();
    adjacency
        .entry(z.into_directed(true))
        .or_default()
        .push((t.into_directed(true), GfaPathLength::from_usize(3)));
    adjacency
        .entry(t.into_directed(false))
        .or_default()
        .push((z.into_directed(false), GfaPathLength::from_usize(3)));

    let mut ws = DijkstraWorkspace::new();

    // Baseline: without a baked flip, z^+ -> z^- is unreachable.
    let base = pole_state_matrix(
        &Skeleton::from_adjacency(&adjacency),
        &mut ws,
        (z, t),
        (z, t),
    );
    assert!(
        base.get(slot(Pole::Source, F), slot(Pole::Source, R))
            .is_infinite()
    );
    assert!(
        base.get(slot(Pole::Source, R), slot(Pole::Source, F))
            .is_infinite()
    );

    // Bake a pendant flip closure: z^+ -> z^- costs 7, z^- -> z^+ costs 9 (off-diagonal only).
    let closure = Mat2::from_rows([
        [MinPlus::infinity(), MinPlus::finite_usize(7)],
        [MinPlus::finite_usize(9), MinPlus::infinity()],
    ]);
    bake_flip_edge(&mut adjacency, z, &closure);

    let baked = pole_state_matrix(
        &Skeleton::from_adjacency(&adjacency),
        &mut ws,
        (z, t),
        (z, t),
    );
    // With the flip baked, the kernel finds it as the shortest z^+ -> z^- route...
    assert_eq!(
        baked.get(slot(Pole::Source, F), slot(Pole::Source, R)),
        MinPlus::finite_usize(7),
    );
    assert_eq!(
        baked.get(slot(Pole::Source, R), slot(Pole::Source, F)),
        MinPlus::finite_usize(9),
    );
    // ...and composes it with a real link: z^- -> (flip-back 9) -> z^+ -> (link, len 3) -> t^+.
    assert_eq!(
        baked.get(slot(Pole::Source, R), slot(Pole::Target, F)),
        MinPlus::finite_usize(9 + 3),
    );
}

#[test]
fn lifting_to_self_is_identity() {
    let tree = RootedTree::new(3, &[(0, 1), (1, 2)], 0);
    let pole_lift = PoleLift::<u32> {
        step: vec![None; 3],
    };
    let vector = PoleVector::from_entries([
        MinPlus::finite_usize(1),
        MinPlus::infinity(),
        MinPlus::finite_usize(4),
        MinPlus::finite_usize(2),
    ]);
    assert_eq!(pole_lift.lift(&tree, 2, 2, vector), vector);
}

// --- Semantic transition check against assembled skeletons ----------------------------------------

fn read_graph(path: &str) -> Graph {
    BidirectedAdjacencyArray::read_gfa1(BufReader::new(File::open(path).unwrap())).unwrap()
}

fn index_of(graph: &Graph, name: &str) -> NodeIndex<u32> {
    graph
        .node_indices()
        .find(|&i| graph.node_data(i).name() == name)
        .unwrap()
}

fn pole_vector_from_distances<S: std::hash::BuildHasher>(
    distances: &HashMap<DirectedNodeIndex<u32>, GfaPathLength<u32>, S>,
    poles: (NodeIndex<u32>, NodeIndex<u32>),
) -> PoleVector<u32> {
    let mut vector = PoleVector::infinity();
    for (pole, segment) in [(Pole::Source, poles.0), (Pole::Target, poles.1)] {
        for orientation in Orientation::all() {
            if let Some(&distance) = distances.get(&segment.into_directed(orientation.is_forward()))
            {
                vector.set(pole, orientation, MinPlus::finite(distance));
            }
        }
    }
    vector
}

/// Lifting a pole vector from the leaf S2 up to P0 must reproduce the distances computed directly in
/// P0's assembled skeleton — for query points (a, b) that are present in both skeletons.
fn assert_lift_matches_assembled_skeleton(gfa_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr = SPQRDecomposition::read_plain_spqr(
        &graph,
        BufReader::new(File::open("test_files/two_triangles.spqr").unwrap()),
    )
    .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let block_tree = BlockSpqrTree::new(&spqr, block_index);
    let inside = InsideDp::compute(&graph, block_tree);
    let pole_lift = PoleLift::compute(&inside);
    let tree = inside.block_tree().tree().clone();

    // Locals: 0 = S1 (root), 1 = P0, 2 = S2. Both have reference poles (a, b).
    let s2 = 2usize;
    let p0 = 1usize;
    let s2_poles = inside.block_tree().node(s2).reference_poles().unwrap();
    let p0_poles = inside.block_tree().node(p0).reference_poles().unwrap();

    for source_name in ["a", "b"] {
        for orientation in Orientation::all() {
            let source = index_of(&graph, source_name).into_directed(orientation.is_forward());

            // Vector at S2, then lifted to P0.
            let at_s2 =
                pole_vector_from_distances(&inside.skeleton_shortest_paths(s2, source), s2_poles);
            let lifted = pole_lift.lift(&tree, s2, p0, at_s2);

            // Ground truth: distances to P0's poles computed directly in P0's skeleton.
            let expected =
                pole_vector_from_distances(&inside.skeleton_shortest_paths(p0, source), p0_poles);

            assert_eq!(
                lifted,
                expected,
                "lift S2->P0 mismatch for source {source_name}{} in {gfa_path}",
                orientation.sign(),
            );
        }
    }
}

/// Pass B (the oversized parent's parallel schedule) must produce byte-identical transitions to
/// Pass A (the serial per-parent path) — the exactness guarantee for the giant R-node, which never
/// appears in a fixture. Compare the two helpers directly on every real assembled-skeleton parent.
fn assert_parallel_transitions_match_serial(gfa_path: &str) {
    let graph = read_graph(gfa_path);
    let spqr = SPQRDecomposition::read_plain_spqr(
        &graph,
        BufReader::new(File::open("test_files/two_triangles.spqr").unwrap()),
    )
    .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let block_tree = BlockSpqrTree::new(&spqr, block_index);
    let inside = InsideDp::compute(&graph, block_tree);
    let node_count = inside.block_tree().node_count();

    let mut workspace = DijkstraWorkspace::new();
    let mut tested_any = false;
    for parent in 0..node_count {
        let block_tree = inside.block_tree();
        if block_tree.tree().children(parent).is_empty()
            || block_tree.node(parent).reference_poles().is_none()
        {
            continue;
        }
        let mut serial = parent_transitions_serial(&inside, parent, &mut workspace);
        let mut parallel = parent_transitions_parallel(&inside, parent);
        serial.sort_by_key(|&(child, _)| child);
        parallel.sort_by_key(|&(child, _)| child);
        assert_eq!(
            serial, parallel,
            "parallel != serial pole transitions for parent {parent} in {gfa_path}",
        );
        tested_any = true;
    }
    assert!(
        tested_any,
        "no parent with children+poles exercised in {gfa_path}"
    );
}

#[test]
fn parallel_transitions_match_serial_two_triangles() {
    assert_parallel_transitions_match_serial("test_files/two_triangles.gfa");
}

#[test]
fn parallel_transitions_match_serial_hairpin() {
    assert_parallel_transitions_match_serial("test_files/two_triangles_hairpin.gfa");
}

#[test]
fn lift_matches_assembled_skeleton_two_triangles() {
    assert_lift_matches_assembled_skeleton("test_files/two_triangles.gfa");
}

#[test]
fn lift_matches_assembled_skeleton_hairpin() {
    assert_lift_matches_assembled_skeleton("test_files/two_triangles_hairpin.gfa");
}
