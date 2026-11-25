use super::*;
use bidirected_adjacency_array::io::gfa1::{PlainGfaEdgeData, PlainGfaNodeData};

type Graph = BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData>;

fn graph_named(names: &[&str]) -> Graph {
    let mut gfa = String::from("H\tVN:Z:1.0\n");
    for name in names {
        gfa.push_str(&format!("S\t{name}\tACGT\n"));
    }
    for pair in names.windows(2) {
        gfa.push_str(&format!("L\t{}\t+\t{}\t+\t0M\n", pair[0], pair[1]));
    }
    Graph::read_gfa1(&mut gfa.as_bytes()).expect("valid GFA")
}

fn assert_round_trip(names: &[&str], view: &GraphView<'_, u32>) {
    for (v, name) in names.iter().enumerate() {
        let node = NodeIndex::from_usize(v);
        assert_eq!(view.node_name(node), *name);
        assert_eq!(view.node_by_name(name), Some(node), "name {name}");
    }
}

#[test]
fn integer_names_round_trip_through_runs() {
    // Runs 5..=7 and 10..=11, a descent to 3, a lone 100.
    let names = ["5", "6", "7", "10", "11", "3", "100"];
    let graph = graph_named(&names);
    let image = Image::new_arena(1 << 20).unwrap();
    let records = NodeLenSecs::write_plain(&image, &graph);
    let (mut sections, _) = GraphSections::write(&image, &graph, records);
    sections.write_adjacency(&image, &graph, |_| true);
    assert!(matches!(sections.names, NameSections::Runs { .. }));
    let view = sections.view(&image);
    assert_round_trip(&names, &view);
    for absent in [
        "0",
        "2",
        "4",
        "8",
        "9",
        "12",
        "99",
        "101",
        "007",
        "05",
        "-1",
        "",
        "1a",
        " 5",
        "+5",
        "18446744073709551616",
    ] {
        assert_eq!(view.node_by_name(absent), None, "name {absent:?}");
    }
}

#[test]
fn caller_runs_write_the_same_table_as_the_scan() {
    let names = ["5", "6", "7", "10", "11", "3", "100"];
    let graph = graph_named(&names);
    let runs_of = |image: &Image, sections: NameSections| -> (Vec<NameRun>, Vec<u32>) {
        let NameSections::Runs { runs, order } = sections else {
            panic!("integer names write runs")
        };
        (image.get(runs).to_vec(), image.get(order).to_vec())
    };
    let scan_image = Image::new_arena(1 << 20).unwrap();
    let scanned = NameSections::write(
        &scan_image,
        names.len(),
        |v| graph.node_data(NodeIndex::from_usize(v)).name(),
        GRAPH_NAME_IDS,
    );
    let runs = integer_name_runs(names.iter().map(|n| n.parse().ok())).unwrap();
    assert_eq!(runs.len(), 4, "runs 5..=7, 10..=11, 3, 100");
    let runs_image = Image::new_arena(1 << 20).unwrap();
    let given = NameSections::write_runs(&runs_image, names.len(), runs, GRAPH_NAME_IDS);
    assert_eq!(runs_of(&scan_image, scanned), runs_of(&runs_image, given));
    assert_eq!(integer_name_runs([Some(1), None, Some(3)]), None);
    assert_eq!(
        integer_name_runs([Some(u64::MAX), Some(0)]).map(|r| r.len()),
        Some(2),
        "no wrap-around run"
    );
    assert_eq!(integer_name_runs(std::iter::empty()), Some(Vec::new()));
}

#[test]
fn non_integer_names_keep_the_byte_table() {
    let names = ["5", "007", "x1"];
    let graph = graph_named(&names);
    let image = Image::new_arena(1 << 20).unwrap();
    let records = NodeLenSecs::write_plain(&image, &graph);
    let (mut sections, _) = GraphSections::write(&image, &graph, records);
    sections.write_adjacency(&image, &graph, |_| true);
    assert!(matches!(sections.names, NameSections::Bytes { .. }));
    let view = sections.view(&image);
    assert_round_trip(&names, &view);
    assert_eq!(view.node_by_name("7"), None);
}
