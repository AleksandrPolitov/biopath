//! GFA input of every subcommand: the engine's strict reader of blunt GFA 1 graphs
//! ([`spqr_shortest_path_index::gfa_reader`]) over a plain or gzipped file, with its warnings
//! logged.

use std::path::Path;

use anyhow::Context;
use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray, index::GraphIndexInteger, io::gfa1::PlainGfaEdgeData,
};
use log::warn;
use spqr_shortest_path_index::gfa_reader::{FromGfaSegment, read_gfa1};

use crate::io_util::read_optionally_compressed_file;

/// Read the graph of the GFA file at `path`.
pub fn read_gfa<IndexType, NodeData, EdgeData>(
    path: &Path,
) -> anyhow::Result<BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>
where
    IndexType: GraphIndexInteger,
    NodeData: FromGfaSegment,
    EdgeData: From<PlainGfaEdgeData>,
{
    read_optionally_compressed_file(path, |reader| {
        read_gfa1(reader, |warning| warn!("GFA file {path:?}: {warning}"))
            .with_context(|| format!("Failed to parse GFA file {path:?}"))
    })
    .with_context(|| format!("Failed to read GFA file: {path:?}"))
}
