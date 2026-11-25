//! Loading refuses a corrupt header instead of panicking on it.

use std::io::Cursor;

use bidirected_adjacency_array::{graph::BidirectedAdjacencyArray, io::gfa1::PlainGfaEdgeData};

use super::{BuildOptions, DistanceIndex};
use crate::{
    decomposition::build_spqr_decomposition, gfa_graph_extensions::LengthOnlyGfaNodeData,
    image::Header, sections::knob,
};

type Graph = BidirectedAdjacencyArray<u32, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

#[test]
fn a_huge_searched_skeleton_size_is_refused() {
    let gfa =
        "S\ta\tAA\nS\tb\tA\nS\tc\tA\nL\ta\t+\tb\t+\t0M\nL\tb\t+\tc\t+\t0M\nL\tc\t+\ta\t+\t0M\n";
    let graph: Graph = BidirectedAdjacencyArray::read_gfa1(Cursor::new(gfa)).unwrap();
    let index = DistanceIndex::build(
        &graph,
        build_spqr_decomposition(&graph),
        &BuildOptions::default(),
    );
    let dir = std::env::temp_dir().join(format!("bpi-test-knob-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("index.bpi");
    index.write_to(&path).unwrap();
    assert!(DistanceIndex::<u32>::open(&path).is_ok());
    let mut bytes = std::fs::read(&path).unwrap();
    let at = std::mem::offset_of!(Header, build_knobs) + 8 * knob::MAX_SEARCHED_NODES;
    bytes[at..at + 8].copy_from_slice(&(u64::MAX / 4).to_ne_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = DistanceIndex::<u32>::open(&path).err().unwrap();
    assert!(err.to_string().contains("corrupt header"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}
