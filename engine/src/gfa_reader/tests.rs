use std::io::BufReader;

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    io::gfa1::{GfaNodeData, PlainGfaEdgeData, PlainGfaNodeData},
};

use super::*;

type Graph<I = u32> = BidirectedAdjacencyArray<I, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

/// Read `gfa`, collecting the warnings.
fn read(gfa: &str) -> (Result<Graph, GfaReadError>, Vec<String>) {
    let mut warnings = Vec::new();
    let graph = read_gfa1(gfa.as_bytes(), |warning| warnings.push(warning.to_string()));
    (graph, warnings)
}

fn ok(gfa: &str) -> Graph {
    let (graph, warnings) = read(gfa);
    assert!(warnings.is_empty(), "unexpected warnings {warnings:?}");
    graph.unwrap_or_else(|error| panic!("{error}"))
}

/// The error message of reading `gfa`, which must fail on `line`.
fn error(gfa: &str, line: usize) -> String {
    let error = read(gfa).0.expect_err("the read must fail");
    assert_eq!(error.line, line, "{error}");
    error.to_string()
}

fn lens(graph: &Graph) -> Vec<(String, usize)> {
    graph
        .iter_nodes()
        .map(|node| {
            let data = graph.node_data(node);
            (data.name().into_owned(), data.sequence().len())
        })
        .collect()
}

fn links(graph: &Graph) -> Vec<(String, bool, String, bool)> {
    graph
        .iter_edges()
        .map(|edge| {
            let edge = graph.edge(edge);
            let name = |node: bidirected_adjacency_array::index::DirectedNodeIndex<u32>| {
                graph.node_data(node.into_bidirected()).name().into_owned()
            };
            (
                name(edge.from()),
                edge.from().is_forward(),
                name(edge.to()),
                edge.to().is_forward(),
            )
        })
        .collect()
}

#[test]
fn builds_the_graph_of_the_crate_reader_on_every_fixture() {
    for fixture in [
        "chain5",
        "GCA-41.clean.largest-block",
        "paper_bridge",
        "rnode_selfflip",
        "tiny1",
        "twoflip",
        "twoflip_hub",
        "two_triangles",
        "two_triangles_hairpin",
    ] {
        let path = format!("{}/test_files/{fixture}.gfa", env!("CARGO_MANIFEST_DIR"));
        let gfa = std::fs::read_to_string(&path).unwrap();
        let ours: BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData> =
            read_gfa1(gfa.as_bytes(), |_| {}).unwrap();
        let theirs: BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData> =
            BidirectedAdjacencyArray::read_gfa1(BufReader::new(gfa.as_bytes())).unwrap();
        assert!(ours == theirs, "{fixture}: the graphs differ");
    }
}

#[test]
fn reads_segments_links_and_skips_comments_paths_and_blank_lines() {
    let graph = ok("H\tVN:Z:1.1\r\n\
                    # a comment\n\
                    \n\
                    S\t1\tACGT\tSN:Z:chr1\n\
                    S\t2\t*\tLN:i:8\n\
                    S\t3\t\n\
                    L\t1\t+\t2\t-\t0M\n\
                    L\t2\t-\t3\t+\t*\n\
                    L\t3\t+\t1\t+\n\
                    L\t1\t-\t3\t-\t0M0I\tID:Z:x\n\
                    P\tp\t1+,2-,3+\t*\n\
                    W\tHG002\t1\tchr1\t0\t12\t>1<2>3\n  \n");
    assert_eq!(
        lens(&graph),
        [("1".into(), 4), ("2".into(), 8), ("3".into(), 0)]
    );
    assert_eq!(
        links(&graph),
        [
            ("1".into(), true, "2".into(), false),
            ("2".into(), false, "3".into(), true),
            ("3".into(), true, "1".into(), true),
            ("1".into(), false, "3".into(), false),
        ]
    );
}

#[test]
fn a_length_only_segment_allocates_no_sequence() {
    // 1 Gbp given by LN only: a length-only node keeps the length, not a gigabyte of bases.
    let graph = ok("S\tbig\t*\tLN:i:1000000000\n");
    assert_eq!(
        graph.node_data(NodeIndex::from_usize(0)).sequence().len(),
        1_000_000_000
    );
    let plain: BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData> =
        read_gfa1("S\tx\t*\tLN:i:3\n".as_bytes(), |_| {}).unwrap();
    assert_eq!(plain.node_data(NodeIndex::from_usize(0)).sequence(), "NNN");
}

#[test]
fn links_may_precede_their_segments() {
    let graph = ok("L\t1\t+\t2\t+\t0M\nS\t1\tA\nS\t2\tC\n");
    assert_eq!(links(&graph), [("1".into(), true, "2".into(), true)]);
}

#[test]
fn header_versions() {
    for version in ["1", "1.0", "1.1", "1.2"] {
        ok(&format!("H\tVN:Z:{version}\nS\t1\tA\n"));
    }
    // Header lines without a version, and anywhere in the file.
    ok("H\tNM:Z:graph\nS\t1\tA\nH\tRS:Z:x\n");
    let message = error("H\tVN:Z:2.0\nS\t1\t4\tACGT\n", 1);
    assert!(message.contains("GFA 2 is not supported"), "{message}");
    let (graph, warnings) = read("H\tVN:Z:0.9\nH\tVN:Z:0.8\nS\t1\tA\n");
    assert!(graph.is_ok());
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("\"0.9\""), "{warnings:?}");
}

#[test]
fn segment_lengths() {
    let message = error("S\t1\tA\nS\tx\t*\n", 2);
    assert!(
        message.contains("\"x\"") && message.contains("LN"),
        "{message}"
    );
    let message = error("S\tx\tACGT\tLN:i:8\n", 1);
    assert!(
        message.contains("4 bases") && message.contains("8"),
        "{message}"
    );
    ok("S\tx\tACGT\tLN:i:4\n");
    error("S\tx\t*\tLN:i:-1\n", 1);
    // An unreadable LN beside a sequence (two tags joined by a space, as in a vg test graph).
    let (graph, warnings) = read("S\tx\tACGT\tLN:i:4 FC:i:3\n");
    assert_eq!(lens(&graph.unwrap()), [("x".into(), 4)]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    error("S\tx\t*\tLN:i:4\tLN:i:5\n", 1);
    // `*` forgotten: the length tag stands in the sequence field.
    let message = error("S\tx\tLN:i:8\n", 1);
    assert!(message.contains("is a tag"), "{message}");
    error("S\tx\tAC\u{e9}GT\n", 1);
    error("S\n", 1);
}

#[test]
fn duplicate_segment_names_are_refused() {
    let message = error("S\t1\tACGT\nS\t2\tAC\nS\t2\tACGTACGT\n", 3);
    assert!(
        message.contains("\"2\"") && message.contains("twice"),
        "{message}"
    );
}

#[test]
fn links_must_be_blunt() {
    for overlap in ["0M", "*", "0M0I", "00M", "0="] {
        ok(&format!("S\t1\tA\nS\t2\tC\nL\t1\t+\t2\t+\t{overlap}\n"));
    }
    for overlap in ["4M", "4M2I", "70000M", "0M1D"] {
        let message = error(&format!("S\t1\tA\nS\t2\tC\nL\t1\t+\t2\t+\t{overlap}\n"), 3);
        assert!(
            message.contains("blunt") && message.contains(overlap),
            "{message}"
        );
    }
    for overlap in ["M", "0", "4Q", "abc"] {
        let message = error(&format!("S\t1\tA\nS\t2\tC\nL\t1\t+\t2\t+\t{overlap}\n"), 3);
        assert!(message.contains("invalid overlap"), "{message}");
    }
}

#[test]
fn links_need_known_segments_and_orientations() {
    let message = error("S\t1\tA\nL\t1\t+\t9\t+\t0M\nS\t2\tC\n", 2);
    assert!(message.contains("\"9\""), "{message}");
    let message = error("S\t1\tA\nS\t2\tC\nL\t1\tx\t2\t+\t0M\n", 3);
    assert!(message.contains("neither + nor -"), "{message}");
    error("S\t1\tA\nL\t1\t+\t1\n", 2);
}

#[test]
fn line_types() {
    for record in [
        "E\te\t1+\t2+\t0\t1\t0\t1\t*",
        "F\tx",
        "G\tg\t1+\t2+\t0\t*",
        "O\to\t1+",
        "U\tu\t1",
    ] {
        let message = error(&format!("S\t1\tA\n{record}\n"), 2);
        assert!(message.contains("GFA 2 is not supported"), "{message}");
    }
    let message = error("S\t1\tA\nS\t2\tC\nJ\t1\t+\t2\t+\t*\n", 3);
    assert!(message.contains("jump"), "{message}");
    // Containments and unknown types are skipped with one warning per type.
    let (graph, warnings) =
        read("S\t1\tA\nC\t1\t+\t1\t+\t0\t0M\nC\t1\t+\t1\t+\t0\t0M\nX\ty\nX\tz\n");
    assert_eq!(graph.unwrap().node_count(), 1);
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    let message = error("S\t1\tA\nS 2 C\n", 2);
    assert!(message.contains("space-separated"), "{message}");
}

#[test]
fn counts_must_fit_the_word_size() {
    let gfa: String = (0..200).map(|i| format!("S\t{i}\tA\n")).collect();
    let error = read_gfa1::<u8, LengthOnlyGfaNodeData, PlainGfaEdgeData>(gfa.as_bytes(), |_| {})
        .expect_err("200 nodes do not fit u8");
    assert_eq!(error.line, 128);
    assert!(
        matches!(error.kind, GfaReadErrorKind::WordSize(_)),
        "{error}"
    );
    assert!(error.to_string().contains("--word-size 8"), "{error}");
    assert!(
        read_gfa1::<u16, LengthOnlyGfaNodeData, PlainGfaEdgeData>(gfa.as_bytes(), |_| {}).is_ok()
    );
    let links: String = std::iter::once("S\t1\tA\n".to_string())
        .chain((0..200).map(|_| "L\t1\t+\t1\t+\t0M\n".to_string()))
        .collect();
    let error = read_gfa1::<u8, LengthOnlyGfaNodeData, PlainGfaEdgeData>(links.as_bytes(), |_| {})
        .expect_err("200 edges do not fit u8");
    assert_eq!(error.line, 129);
}
