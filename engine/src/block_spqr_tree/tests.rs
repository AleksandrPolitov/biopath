use crate::rooted_tree::TreeAccess;
use std::{collections::HashSet, fs::File, io::BufReader};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::NodeIndex,
    io::gfa1::{GfaNodeData, PlainGfaEdgeData, PlainGfaNodeData},
};
use spqr_tree::decomposition::{SPQRDecomposition, SPQRNodeType};

use super::BlockSpqrTree;

type Graph = BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData>;

fn read_graph(path: &str) -> Graph {
    BidirectedAdjacencyArray::read_gfa1(BufReader::new(File::open(path).unwrap())).unwrap()
}

fn name(graph: &Graph, node: NodeIndex<u32>) -> String {
    graph.node_data(node).name().into_owned()
}

/// The unordered pair of segment names of two bidirected pole nodes.
fn pole_names(graph: &Graph, poles: (NodeIndex<u32>, NodeIndex<u32>)) -> (String, String) {
    let mut names = [name(graph, poles.0), name(graph, poles.1)];
    names.sort();
    (names[0].clone(), names[1].clone())
}

#[test]
fn tiny1_every_block_is_a_single_s_node() {
    let graph = read_graph("test_files/tiny1.gfa");
    let spqr = SPQRDecomposition::read_plain_spqr(
        &graph,
        BufReader::new(File::open("test_files/tiny1.spqr").unwrap()),
    )
    .unwrap();

    let mut block_count = 0;
    for (block_index, block) in spqr.iter_blocks() {
        block_count += 1;
        let block_node_names: HashSet<String> =
            block.iter_nodes().map(|node| name(&graph, node)).collect();

        let tree = BlockSpqrTree::new(&spqr, block_index);

        // tiny1's blocks are triangles: exactly one S-node, so the tree is a single root node.
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.root_local(), 0);
        assert_eq!(tree.node(0).node_type(), SPQRNodeType::SNode);
        assert_eq!(tree.tree().parent(0), None);
        assert_eq!(tree.node(0).reference_poles(), None);
        assert!(tree.node(0).child_virtual_edges().is_empty());

        // Every real skeleton edge is an actual link between two nodes of this block.
        let real = tree.real_directed_edges(0, &graph);
        assert_eq!(real.len(), 3, "a triangle S-node has three real edges");
        for edge in real {
            let from = name(&graph, edge.from.into_bidirected());
            let to = name(&graph, edge.to.into_bidirected());
            assert!(block_node_names.contains(&from), "{from} not in block");
            assert!(block_node_names.contains(&to), "{to} not in block");
            assert_ne!(from, to);
        }
    }
    assert_eq!(block_count, 5, "tiny1 has five blocks");
}

#[test]
fn two_triangles_roots_a_three_node_spqr_tree() {
    let graph = read_graph("test_files/two_triangles.gfa");
    let spqr = SPQRDecomposition::read_plain_spqr(
        &graph,
        BufReader::new(File::open("test_files/two_triangles.spqr").unwrap()),
    )
    .unwrap();

    // A single biconnected block containing the whole graph.
    let mut blocks = spqr.iter_blocks();
    let (block_index, _) = blocks.next().unwrap();
    assert!(blocks.next().is_none(), "two_triangles has a single block");

    let tree = BlockSpqrTree::new(&spqr, block_index);
    assert_eq!(tree.node_count(), 3);

    // File order S1, P0, S2 -> local ids 0, 1, 2, rooted at 0. The tree is the path S1 - P0 - S2.
    assert_eq!(tree.node(0).node_type(), SPQRNodeType::SNode);
    assert_eq!(tree.node(1).node_type(), SPQRNodeType::PNode);
    assert_eq!(tree.node(2).node_type(), SPQRNodeType::SNode);

    assert_eq!(tree.root_local(), 0);
    assert_eq!(tree.tree().parent(0), None);
    assert_eq!(tree.tree().parent(1), Some(0));
    assert_eq!(tree.tree().parent(2), Some(1));
    assert_eq!(tree.tree().depth(0), 0);
    assert_eq!(tree.tree().depth(1), 1);
    assert_eq!(tree.tree().depth(2), 2);

    // Both separation pairs are {a, b}.
    assert_eq!(tree.node(0).reference_poles(), None);
    assert_eq!(
        pole_names(&graph, tree.node(1).reference_poles().unwrap()),
        ("a".to_string(), "b".to_string()),
    );
    assert_eq!(
        pole_names(&graph, tree.node(2).reference_poles().unwrap()),
        ("a".to_string(), "b".to_string()),
    );

    // Child virtual edges: root -> P0, P0 -> S2, leaf S2 has none.
    let root_children = tree.node(0).child_virtual_edges();
    assert_eq!(root_children.len(), 1);
    assert_eq!(root_children[0].child_local, 1);
    let mid_children = tree.node(1).child_virtual_edges();
    assert_eq!(mid_children.len(), 1);
    assert_eq!(mid_children[0].child_local, 2);
    assert!(tree.node(2).child_virtual_edges().is_empty());
}

#[test]
fn two_triangles_reference_and_child_poles_agree() {
    // The pole pair a non-root node records as its reference (parent) virtual edge must be exactly
    // the pole pair its parent records as the child virtual edge for it. A mistake in edge/local
    // bookkeeping would break this invariant.
    let graph = read_graph("test_files/two_triangles.gfa");
    let spqr = SPQRDecomposition::read_plain_spqr(
        &graph,
        BufReader::new(File::open("test_files/two_triangles.spqr").unwrap()),
    )
    .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let tree = BlockSpqrTree::new(&spqr, block_index);

    for local in 0..tree.node_count() {
        let Some(reference_poles) = tree.node(local).reference_poles() else {
            continue;
        };
        let parent = tree.tree().parent(local).unwrap();
        let child_entry = tree
            .node(parent)
            .child_virtual_edges()
            .iter()
            .find(|edge| edge.child_local == local)
            .expect("parent must record a child virtual edge for this node");

        assert_eq!(
            pole_names(&graph, reference_poles),
            pole_names(&graph, child_entry.poles),
            "reference/child pole mismatch at local {local}",
        );
    }
}

#[test]
fn two_triangles_real_edges_match_the_gfa_links() {
    let graph = read_graph("test_files/two_triangles.gfa");
    let spqr = SPQRDecomposition::read_plain_spqr(
        &graph,
        BufReader::new(File::open("test_files/two_triangles.spqr").unwrap()),
    )
    .unwrap();
    let (block_index, _) = spqr.iter_blocks().next().unwrap();
    let tree = BlockSpqrTree::new(&spqr, block_index);

    let real_edge_name_set = |local: usize| -> HashSet<(String, String)> {
        tree.real_directed_edges(local, &graph)
            .map(|edge| {
                let mut names = [
                    name(&graph, edge.from.into_bidirected()),
                    name(&graph, edge.to.into_bidirected()),
                ];
                names.sort();
                (names[0].clone(), names[1].clone())
            })
            .collect()
    };

    // S1 = triangle a-c-b: real edges a-c and c-b (the a-b edge lives in the P-node).
    assert_eq!(
        real_edge_name_set(0),
        HashSet::from([
            ("a".to_string(), "c".to_string()),
            ("b".to_string(), "c".to_string()),
        ]),
    );
    // P0 holds the direct a-b edge.
    assert_eq!(
        real_edge_name_set(1),
        HashSet::from([("a".to_string(), "b".to_string())]),
    );
    // S2 = triangle a-d-b: real edges a-d and d-b.
    assert_eq!(
        real_edge_name_set(2),
        HashSet::from([
            ("a".to_string(), "d".to_string()),
            ("b".to_string(), "d".to_string()),
        ]),
    );
}
