//! GBZ input (`--graph-gbz-in`): the graph of a GBZ file (vg's graph + haplotype index, the input of
//! Giraffe and `vg index -j`) as the same `BidirectedAdjacencyArray` a GFA read gives.
//!
//! Nodes are the GBZ nodes in id order, named by their id; edges are the GBZ's edges, each bidirected
//! edge once. A GBZ graph has exactly the adjacencies some path uses, so this is the graph
//! `vg index -j` indexes when given the same GBZ, and the graph `vg convert -f --no-translation`
//! exports. Reading it here avoids that export: vg's GFA writer decompresses every GBWT record over
//! 1 KiB into memory before writing a line, far more memory than the GBZ itself on a big pangenome.

use std::{path::Path, time::Instant};

use anyhow::Context;
use bidirected_adjacency_array::{
    graph::{BidirectedAdjacencyArray, BidirectedEdge},
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::PlainGfaEdgeData,
};
use gbwt::{GBZ, Orientation};
use log::{info, warn};
use spqr_shortest_path_index::{
    gfa_graph_extensions::LengthOnlyGfaNodeData,
    graph_limits,
    graph_view::{NameRun, integer_name_runs},
};

/// Progress line interval while building the graph (nodes).
const PROGRESS_NODES: usize = 10_000_000;

/// The GBZ file format version the `gbwt` crate reads. vg 1.7x writes version 2 (zstd-compressed
/// node sequences) by default; `vg gbwt --gbz-version 1` converts a file back.
const READABLE_GBZ_VERSION: u32 = 1;

/// The format version in a GBZ file's header (`GBZ ` tag, then a little-endian u32), if it is one.
fn gbz_file_version(path: &Path) -> anyhow::Result<Option<u32>> {
    use std::io::Read;
    let mut header = [0u8; 8];
    let mut file = std::fs::File::open(path).with_context(|| format!("Failed to open {path:?}"))?;
    if file.read_exact(&mut header).is_err() || &header[..4] != b"GBZ " {
        return Ok(None);
    }
    Ok(Some(u32::from_le_bytes(header[4..8].try_into().unwrap())))
}

/// Load a GBZ file (held in memory, compressed, about the size of the file).
fn load_gbz(path: &Path) -> anyhow::Result<GBZ> {
    let start = Instant::now();
    info!("Loading GBZ file {path:?}");
    match gbz_file_version(path)? {
        None => anyhow::bail!("{path:?} is not a GBZ file (no `GBZ ` header)"),
        Some(version) if version != READABLE_GBZ_VERSION => anyhow::bail!(
            "{path:?} is GBZ format version {version}; biopath reads version {READABLE_GBZ_VERSION}. \
             Convert it with `vg gbwt -Z {} --gbz-format --gbz-version 1 -g <out.gbz>`",
            path.display()
        ),
        Some(_) => {}
    }
    let gbz: GBZ = simple_sds::serialize::load_from(path)
        .with_context(|| format!("Failed to load GBZ file {path:?}"))?;
    info!(
        "GBZ has {} nodes (ids {}..={}) and {} paths; loaded in {:.2?}",
        gbz.nodes(),
        gbz.min_node(),
        gbz.max_node(),
        gbz.paths(),
        start.elapsed()
    );
    if gbz.has_translation() {
        warn!(
            "the GBZ has a segment-to-node translation: node names are GBZ node ids (what vg uses), \
             not the original GFA segment names"
        );
    }
    Ok(gbz)
}

/// The GBWT encoding of an oriented node: `2 * id + (reverse as usize)`.
fn handle(id: usize, orientation: Orientation) -> usize {
    2 * id + (orientation == Orientation::Reverse) as usize
}

/// The graph of `gbz`: every node (name = id, length = sequence length) and every edge once.
///
/// The GBWT stores each bidirected edge `(u, o) -> (v, p)` in both directions: as a successor of
/// `(u, o)` and as the successor `(u, !o)` of `(v, !p)`. Only the view with `handle(u, o) <=
/// flip(handle(v, p))` is kept; a reversing self loop is one view only, and it passes.
pub fn graph_from_gbz<IndexType: GraphIndexInteger>(
    gbz: &GBZ,
) -> anyhow::Result<BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>> {
    let start = Instant::now();
    let node_count = gbz.nodes();
    anyhow::ensure!(node_count > 0, "the GBZ has no nodes");
    anyhow::ensure!(
        node_count < u32::MAX as usize,
        "the GBZ has {node_count} nodes; at most {} are supported",
        u32::MAX - 1
    );
    // Node indices of the word size are made below: refuse a graph they cannot number.
    graph_limits::check_counts::<IndexType>(node_count, 0)?;
    let min_node = gbz.min_node();
    // GBZ node ids form a range with possible gaps; node index of id = index_of[id - min_node].
    let mut index_of = vec![u32::MAX; gbz.max_node() - min_node + 1];
    let mut nodes = Vec::with_capacity(node_count);
    for id in gbz.node_iter() {
        let len = gbz
            .sequence_len(id)
            .ok_or_else(|| anyhow::anyhow!("GBZ node {id} has no sequence"))?;
        index_of[id - min_node] = nodes.len() as u32;
        nodes.push(LengthOnlyGfaNodeData::with_len(&id.to_string(), len));
        if nodes.len() % PROGRESS_NODES == 0 {
            info!(
                "[gbz] nodes {}/{node_count} ({:.2?})",
                nodes.len(),
                start.elapsed()
            );
        }
    }
    let node_index =
        |id: usize| NodeIndex::<IndexType>::from_usize(index_of[id - min_node] as usize);

    let mut edges = Vec::new();
    for (done, id) in gbz.node_iter().enumerate() {
        for orientation in [Orientation::Forward, Orientation::Reverse] {
            let from = handle(id, orientation);
            let successors = gbz
                .successors(id, orientation)
                .ok_or_else(|| anyhow::anyhow!("GBZ node {id} has no record"))?;
            for (to_id, to_orientation) in successors {
                if from > handle(to_id, to_orientation) ^ 1 {
                    continue;
                }
                edges.push(BidirectedEdge {
                    from: node_index(id),
                    from_forward: orientation == Orientation::Forward,
                    to: node_index(to_id),
                    to_forward: to_orientation == Orientation::Forward,
                    data: PlainGfaEdgeData::new(0),
                });
            }
        }
        if (done + 1) % PROGRESS_NODES == 0 {
            info!(
                "[gbz] edges of {}/{node_count} nodes: {} edges ({:.2?})",
                done + 1,
                edges.len(),
                start.elapsed()
            );
        }
    }
    drop(index_of);
    graph_limits::check_counts::<IndexType>(nodes.len(), edges.len())?;
    info!(
        "[gbz] graph: {} nodes, {} edges in {:.2?}",
        nodes.len(),
        edges.len(),
        start.elapsed()
    );
    Ok(BidirectedAdjacencyArray::new(nodes.into(), edges.into()))
}

/// Load the GBZ at `path` and build its graph; the GBZ is freed before returning.
pub fn read_gbz<IndexType: GraphIndexInteger>(
    path: &Path,
) -> anyhow::Result<BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>> {
    let gbz = load_gbz(path)?;
    graph_from_gbz(&gbz)
}

/// [`read_gbz`], plus the node names as integer runs: node `v` is named by its GBZ id, so the runs
/// come from the ids (gaps in the id space start new runs) and the index never reads the names back.
pub fn read_gbz_with_name_runs<IndexType: GraphIndexInteger>(
    path: &Path,
) -> anyhow::Result<(
    BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>,
    Vec<NameRun>,
)> {
    let gbz = load_gbz(path)?;
    let graph = graph_from_gbz(&gbz)?;
    Ok((graph, name_runs(&gbz)))
}

/// The GBZ node ids, in node order (the order [`graph_from_gbz`] numbers the nodes), as name runs.
fn name_runs(gbz: &GBZ) -> Vec<NameRun> {
    integer_name_runs(gbz.node_iter().map(|id| Some(id as u64))).expect("node ids are integers")
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, path::PathBuf};

    use bidirected_adjacency_array::io::gfa1::GfaNodeData;

    use super::*;

    type Graph = BidirectedAdjacencyArray<u32, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

    /// `test_files/gbz/small.gbz` was built by vg 1.77 (`vg gbwt -G small.gfa --gbz-format`) and
    /// `small.vg.gfa` is vg's own export of it (`vg convert -f --no-translation small.gbz`). The
    /// graph has two components, bubbles, inversions, a hairpin and a reversing self loop.
    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_files/gbz")
            .join(name)
    }

    fn nodes(graph: &Graph) -> Vec<(String, usize)> {
        graph
            .iter_nodes()
            .map(|n| {
                (
                    graph.node_data(n).name().into_owned(),
                    graph.node_data(n).sequence().len(),
                )
            })
            .collect()
    }

    /// Every edge as `(name, forward, name, forward)` in the orientation whose first side is smaller.
    fn edges(graph: &Graph) -> BTreeSet<(String, bool, String, bool)> {
        graph
            .iter_edges()
            .map(|e| {
                let edge = graph.edge(e);
                let (from, to) = (edge.from(), edge.to());
                let name = |n: NodeIndex<u32>| graph.node_data(n).name().into_owned();
                let a = (name(from.into_bidirected()), from.is_forward());
                let b = (name(to.into_bidirected()), to.is_forward());
                // The same edge read from its other end: flip(b) -> flip(a).
                let (a2, b2) = ((b.0.clone(), !b.1), (a.0.clone(), !a.1));
                let key = |x: &(String, bool)| (x.0.parse::<u64>().unwrap(), x.1);
                if (key(&a), key(&b)) <= (key(&a2), key(&b2)) {
                    (a.0, a.1, b.0, b.1)
                } else {
                    (a2.0, a2.1, b2.0, b2.1)
                }
            })
            .collect()
    }

    #[test]
    fn gbz_graph_equals_vg_export() {
        let from_gbz: Graph = read_gbz(&fixture("small.gbz")).unwrap();
        let from_gfa: Graph = spqr_shortest_path_index::gfa_reader::read_gfa1(
            std::io::BufReader::new(std::fs::File::open(fixture("small.vg.gfa")).unwrap()),
            |warning| panic!("vg's export reads without a warning: {warning}"),
        )
        .unwrap();
        assert_eq!(nodes(&from_gbz), nodes(&from_gfa));
        assert_eq!(
            from_gbz.edge_count(),
            from_gfa.edge_count(),
            "every edge exactly once"
        );
        assert_eq!(edges(&from_gbz), edges(&from_gfa));
    }

    #[test]
    fn name_runs_are_the_graph_names() {
        let (graph, runs): (Graph, _) = read_gbz_with_name_runs(&fixture("small.gbz")).unwrap();
        let scanned = integer_name_runs(
            graph
                .iter_nodes()
                .map(|n| graph.node_data(n).name().parse::<u64>().ok()),
        )
        .unwrap();
        assert_eq!(runs, scanned);
        assert!(
            runs.len() > 1,
            "the fixture's id space has gaps (unvisited alleles), so several runs"
        );
    }

    #[test]
    fn newer_gbz_version_is_refused_with_the_conversion_command() {
        let err = load_gbz(&fixture("small.v2.gbz")).unwrap_err().to_string();
        assert!(
            err.contains("version 2") && err.contains("--gbz-version 1"),
            "{err}"
        );
        let err = load_gbz(&fixture("small.vg.gfa")).unwrap_err().to_string();
        assert!(err.contains("not a GBZ file"), "{err}");
    }
}
