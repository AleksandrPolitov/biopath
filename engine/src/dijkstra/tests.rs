use bidirected_adjacency_array::{
    graph::{BidirectedAdjacencyArray, BidirectedEdge},
    index::DirectedNodeIndex,
    io::gfa1::{PlainGfaEdgeData, PlainGfaNodeData},
};

use crate::{
    dijkstra::{
        GfaDijkstra,
        gfa_path::{GfaPath, PathElement},
        location_index::{multi::MultiGfaLocationIndex, single::SingleGfaLocationIndex},
    },
    location::GfaLocation,
};

#[test]
fn test_simple_tight_source() {
    let nodes = vec![
        PlainGfaNodeData::new("A", "AAA"),
        PlainGfaNodeData::new("B", "BBB"),
        PlainGfaNodeData::new("C", "CCC"),
        PlainGfaNodeData::new("D", "DDDDDDDDD"),
        PlainGfaNodeData::new("E", "EEE"),
    ];
    let edges =
        [(0, 2, 0), (2, 4, 0), (4, 8, 0), (0, 6, 0), (6, 8, 0)].map(|(from, to, overlap)| {
            BidirectedEdge::new_gfa(
                DirectedNodeIndex::new(from),
                DirectedNodeIndex::new(to),
                overlap,
            )
        });
    let graph =
        BidirectedAdjacencyArray::<u8, _, _>::new(nodes.into(), FromIterator::from_iter(edges));

    let mut dijkstra = GfaDijkstra::new(&graph);
    let paths = dijkstra.shortest_paths(
        GfaLocation::new(0.into(), 3.into()),
        &SingleGfaLocationIndex::new_target(GfaLocation::new(8.into(), 2.into())),
    );
    assert_eq!(paths.len(), 1);
    let path = paths.values().next().unwrap();

    let expected_path = vec![
        PathElement::new(0.into(), 3.into(), 3.into()),
        PathElement::new(2.into(), 0.into(), 3.into()),
        PathElement::new(4.into(), 0.into(), 3.into()),
        PathElement::new(8.into(), 0.into(), 2.into()),
    ];
    let expected_path = GfaPath::new(expected_path, 8.into());

    assert_eq!(path.length(), expected_path.length());
    assert_eq!(
        path.iter().collect::<Vec<_>>(),
        expected_path.iter().collect::<Vec<_>>(),
        "Paths differ:\nExpected: {expected_path:?}\nActual:   {path:?}",
    );
}

#[test]
fn test_simple_tight_target() {
    let nodes = vec![
        PlainGfaNodeData::new("A", "AAA"),
        PlainGfaNodeData::new("B", "BBB"),
        PlainGfaNodeData::new("C", "CCC"),
        PlainGfaNodeData::new("D", "DDDDDDDDD"),
        PlainGfaNodeData::new("E", "EEE"),
    ];
    let edges =
        [(0, 2, 0), (2, 4, 0), (4, 8, 0), (0, 6, 0), (6, 8, 0)].map(|(from, to, overlap)| {
            BidirectedEdge::new_gfa(
                DirectedNodeIndex::new(from),
                DirectedNodeIndex::new(to),
                overlap,
            )
        });
    let graph =
        BidirectedAdjacencyArray::<u8, _, _>::new(nodes.into(), FromIterator::from_iter(edges));

    let mut dijkstra = GfaDijkstra::new(&graph);
    let paths = dijkstra.shortest_paths(
        GfaLocation::new(0.into(), 2.into()),
        &SingleGfaLocationIndex::new_target(GfaLocation::new(8.into(), 0.into())),
    );
    assert_eq!(paths.len(), 1);
    let path = paths.values().next().unwrap();

    let expected_path = vec![
        PathElement::new(0.into(), 2.into(), 3.into()),
        PathElement::new(2.into(), 0.into(), 3.into()),
        PathElement::new(4.into(), 0.into(), 3.into()),
        PathElement::new(8.into(), 0.into(), 0.into()),
    ];
    let expected_path = GfaPath::new(expected_path, 7.into());

    assert_eq!(path.length(), expected_path.length());
    assert_eq!(
        path.iter().collect::<Vec<_>>(),
        expected_path.iter().collect::<Vec<_>>(),
        "Paths differ:\nExpected: {expected_path:?}\nActual:   {path:?}",
    );
}

#[test]
fn test_simple_tight_source_and_target() {
    let nodes = vec![
        PlainGfaNodeData::new("A", "AAA"),
        PlainGfaNodeData::new("B", "BBB"),
        PlainGfaNodeData::new("C", "CCC"),
        PlainGfaNodeData::new("D", "DDDDDDDDD"),
        PlainGfaNodeData::new("E", "EEE"),
    ];
    let edges =
        [(0, 2, 0), (2, 4, 0), (4, 8, 0), (0, 6, 0), (6, 8, 0)].map(|(from, to, overlap)| {
            BidirectedEdge::new_gfa(
                DirectedNodeIndex::new(from),
                DirectedNodeIndex::new(to),
                overlap,
            )
        });
    let graph =
        BidirectedAdjacencyArray::<u8, _, _>::new(nodes.into(), FromIterator::from_iter(edges));

    let mut dijkstra = GfaDijkstra::new(&graph);
    let paths = dijkstra.shortest_paths(
        GfaLocation::new(0.into(), 3.into()),
        &SingleGfaLocationIndex::new_target(GfaLocation::new(8.into(), 0.into())),
    );
    assert_eq!(paths.len(), 1);
    let path = paths.values().next().unwrap();

    let expected_path = vec![
        PathElement::new(0.into(), 3.into(), 3.into()),
        PathElement::new(2.into(), 0.into(), 3.into()),
        PathElement::new(4.into(), 0.into(), 3.into()),
        PathElement::new(8.into(), 0.into(), 0.into()),
    ];
    let expected_path = GfaPath::new(expected_path, 6.into());

    assert_eq!(path.length(), expected_path.length());
    assert_eq!(
        path.iter().collect::<Vec<_>>(),
        expected_path.iter().collect::<Vec<_>>(),
        "Paths differ:\nExpected: {expected_path:?}\nActual:   {path:?}",
    );
}

#[test]
fn test_cycle() {
    let nodes = vec![
        PlainGfaNodeData::new("A", "AAA"),
        PlainGfaNodeData::new("B", "BBB"),
        PlainGfaNodeData::new("C", "CCC"),
    ];
    let edges = [(0, 2, 0), (2, 4, 0), (4, 0, 0)].map(|(from, to, overlap)| {
        BidirectedEdge::new_gfa(
            DirectedNodeIndex::new(from),
            DirectedNodeIndex::new(to),
            overlap,
        )
    });
    let graph =
        BidirectedAdjacencyArray::<u8, _, _>::new(nodes.into(), FromIterator::from_iter(edges));

    let mut dijkstra = GfaDijkstra::new(&graph);
    let paths = dijkstra.shortest_paths(
        GfaLocation::new(0.into(), 3.into()),
        &SingleGfaLocationIndex::new_target(GfaLocation::new(0.into(), 0.into())),
    );
    assert_eq!(paths.len(), 1);
    let path = paths.values().next().unwrap();

    let expected_path = vec![
        PathElement::new(0.into(), 3.into(), 3.into()),
        PathElement::new(2.into(), 0.into(), 3.into()),
        PathElement::new(4.into(), 0.into(), 3.into()),
        PathElement::new(0.into(), 0.into(), 0.into()),
    ];
    let expected_path = GfaPath::new(expected_path, 6.into());

    assert_eq!(path.length(), expected_path.length());
    assert_eq!(
        path.iter().collect::<Vec<_>>(),
        expected_path.iter().collect::<Vec<_>>(),
        "Paths differ:\nExpected: {expected_path:?}\nActual:   {path:?}",
    );
}

#[test]
fn test_internal_path() {
    let nodes = vec![PlainGfaNodeData::new("A", "AAAAA")];
    let edges = [(0, 0, 0)].map(|(from, to, overlap)| {
        BidirectedEdge::new_gfa(
            DirectedNodeIndex::new(from),
            DirectedNodeIndex::new(to),
            overlap,
        )
    });
    let graph =
        BidirectedAdjacencyArray::<u8, _, _>::new(nodes.into(), FromIterator::from_iter(edges));

    let mut dijkstra = GfaDijkstra::new(&graph);
    let paths = dijkstra.shortest_paths(
        GfaLocation::new(0.into(), 1.into()),
        &SingleGfaLocationIndex::new_target(GfaLocation::new(0.into(), 3.into())),
    );
    assert_eq!(paths.len(), 1);
    let path = paths.values().next().unwrap();

    let expected_path = vec![PathElement::new(0.into(), 1.into(), 3.into())];
    let expected_path = GfaPath::new(expected_path, 2.into());

    assert_eq!(path.length(), expected_path.length());
    assert_eq!(
        path.iter().collect::<Vec<_>>(),
        expected_path.iter().collect::<Vec<_>>(),
        "Paths differ:\nExpected: {expected_path:?}\nActual:   {path:?}",
    );
}

#[test]
fn test_internal_empty_path() {
    let nodes = vec![PlainGfaNodeData::new("A", "AAAAA")];
    let edges = [(0, 0, 0)].map(|(from, to, overlap)| {
        BidirectedEdge::new_gfa(
            DirectedNodeIndex::new(from),
            DirectedNodeIndex::new(to),
            overlap,
        )
    });
    let graph =
        BidirectedAdjacencyArray::<u8, _, _>::new(nodes.into(), FromIterator::from_iter(edges));

    let mut dijkstra = GfaDijkstra::new(&graph);
    let paths = dijkstra.shortest_paths(
        GfaLocation::new(0.into(), 1.into()),
        &SingleGfaLocationIndex::new_target(GfaLocation::new(0.into(), 1.into())),
    );
    assert_eq!(paths.len(), 1);
    let path = paths.values().next().unwrap();

    let expected_path = vec![PathElement::new(0.into(), 1.into(), 1.into())];
    let expected_path = GfaPath::new(expected_path, 0.into());

    assert_eq!(path.length(), expected_path.length());
    assert_eq!(
        path.iter().collect::<Vec<_>>(),
        expected_path.iter().collect::<Vec<_>>(),
        "Paths differ:\nExpected: {expected_path:?}\nActual:   {path:?}",
    );
}

#[test]
fn test_self_loop_external() {
    let nodes = vec![PlainGfaNodeData::new("A", "AAAAA")];
    let edges = [(0, 0, 0)].map(|(from, to, overlap)| {
        BidirectedEdge::new_gfa(
            DirectedNodeIndex::new(from),
            DirectedNodeIndex::new(to),
            overlap,
        )
    });
    let graph =
        BidirectedAdjacencyArray::<u8, _, _>::new(nodes.into(), FromIterator::from_iter(edges));

    let mut dijkstra = GfaDijkstra::new(&graph);
    let paths = dijkstra.shortest_paths(
        GfaLocation::new(0.into(), 4.into()),
        &SingleGfaLocationIndex::new_target(GfaLocation::new(0.into(), 2.into())),
    );
    assert_eq!(paths.len(), 1);
    let path = paths.values().next().unwrap();

    let expected_path = vec![
        PathElement::new(0.into(), 4.into(), 5.into()),
        PathElement::new(0.into(), 0.into(), 2.into()),
    ];
    let expected_path = GfaPath::new(expected_path, 3.into());

    assert_eq!(path.length(), expected_path.length());
    assert_eq!(
        path.iter().collect::<Vec<_>>(),
        expected_path.iter().collect::<Vec<_>>(),
        "Paths differ:\nExpected: {expected_path:?}\nActual:   {path:?}",
    );
}

/// A chain A -> B -> C of 3, 3 and 5 bp.
fn chain_graph() -> BidirectedAdjacencyArray<u8, PlainGfaNodeData, PlainGfaEdgeData> {
    let nodes = vec![
        PlainGfaNodeData::new("A", "AAA"),
        PlainGfaNodeData::new("B", "BBB"),
        PlainGfaNodeData::new("C", "CCCCC"),
    ];
    let edges = [(0, 2), (2, 4)].map(|(from, to)| {
        BidirectedEdge::new_gfa(DirectedNodeIndex::new(from), DirectedNodeIndex::new(to), 0)
    });
    BidirectedAdjacencyArray::new(nodes.into(), FromIterator::from_iter(edges))
}

#[test]
fn several_targets_on_one_directed_node() {
    let graph = chain_graph();
    let mut dijkstra = GfaDijkstra::new(&graph);
    let source = GfaLocation::new(0.into(), 1.into());
    let targets = [
        GfaLocation::new(4.into(), 0.into()),
        GfaLocation::new(4.into(), 1.into()),
        GfaLocation::new(4.into(), 0.into()),
        GfaLocation::new(4.into(), 5.into()),
        GfaLocation::new(2.into(), 2.into()),
    ];
    let paths =
        dijkstra.shortest_paths(source, &MultiGfaLocationIndex::new_targets(&graph, targets));
    let length = |target: GfaLocation<u8>| paths[&target].length().into_usize();
    assert_eq!(paths.len(), 4);
    assert_eq!(length(targets[0]), 5);
    assert_eq!(length(targets[1]), 6);
    assert_eq!(length(targets[3]), 10);
    assert_eq!(length(targets[4]), 4);
}

#[test]
#[should_panic(expected = "exceeds the 8-bit word size")]
fn a_distance_beyond_the_word_size_panics_instead_of_wrapping() {
    let nodes = vec![
        PlainGfaNodeData::new("A", "A".repeat(200)),
        PlainGfaNodeData::new("B", "B".repeat(200)),
    ];
    let edges = [BidirectedEdge::new_gfa(
        DirectedNodeIndex::new(0),
        DirectedNodeIndex::new(2),
        0,
    )];
    let graph =
        BidirectedAdjacencyArray::<u8, _, _>::new(nodes.into(), FromIterator::from_iter(edges));
    GfaDijkstra::new(&graph).shortest_paths(
        GfaLocation::new(0.into(), 0.into()),
        &SingleGfaLocationIndex::new_target(GfaLocation::new(2.into(), 100.into())),
    );
}
