//! G′: the chains left after the reduction become its nodes, the alive links between their ends its
//! links. The runs of every G′ node go to the alias as it is numbered.

use std::collections::HashMap;

use bidirected_adjacency_array::{
    graph::{BidirectedAdjacencyArray, BidirectedEdge},
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::PlainGfaEdgeData,
};
use tagged_vec::TaggedVec;

use super::{
    OFF_MASK, ShrinkStats,
    chains::Chains,
    layout::RunLayout,
    records::RUN_ID_MASK,
    side_graph::{ChainMember, SideGraph, is_right, side, side_node},
};
use crate::gfa_graph_extensions::LengthOnlyGfaNodeData;

/// A chain left after the reduction, walked from its lower end: one node of G′.
struct KeptChain {
    members: Vec<ChainMember>,
    /// The outer side the walk enters through (the G′ node's left side).
    entry: usize,
    /// The outer side the walk leaves through (the G′ node's right side).
    exit: usize,
    len: u64,
    cyclic: bool,
}

/// G′ from the chains left in `sg`; every G′ node's runs go to `layout`. `Err` when G′ or one of its
/// nodes exceeds the alias encoding or the index width.
pub(super) fn assemble_g_prime<IndexType: GraphIndexInteger>(
    sg: &SideGraph,
    chains: &mut Chains,
    layout: &mut RunLayout,
    stats: &mut ShrinkStats,
) -> Result<BidirectedAdjacencyArray<IndexType, LengthOnlyGfaNodeData, PlainGfaEdgeData>, String> {
    let mut internal = vec![false; sg.edges.len()];
    let kept = kept_chains(sg, chains, &mut internal);
    // A run record addresses at most 2^29 G′ nodes (a multi-node pruned chain index stays below
    // that: there are fewer than `MAX_ORIGINAL_NODES / 2` of them).
    if kept.len() > RUN_ID_MASK as usize + 1 {
        return Err(format!(
            "G′ has {} nodes; the alias encodes at most {}",
            kept.len(),
            RUN_ID_MASK as usize + 1
        ));
    }
    // The single-node pruned set is complete: runs are cut by rank among the other nodes.
    layout.seal_singles(sg.lens.len());
    let mut g_side = HashMap::new();
    let nodes = g_prime_nodes::<IndexType>(&kept, sg, layout, &mut g_side, stats)?;
    let links = g_prime_links(sg, &internal, &g_side);
    let edges: Vec<BidirectedEdge<IndexType, PlainGfaEdgeData>> = links
        .iter()
        .map(|&(x, y)| {
            let (x, y) = (x as usize, y as usize);
            // Leaving through a right side is forward; entering through a left side is forward.
            let from = NodeIndex::from_usize(side_node(x)).into_directed(is_right(x));
            let to = NodeIndex::from_usize(side_node(y)).into_directed(!is_right(y));
            BidirectedEdge::new(from, to, PlainGfaEdgeData::new(0))
        })
        .collect();
    stats.shrunk_nodes = nodes.len();
    stats.shrunk_edges = edges.len();
    Ok(BidirectedAdjacencyArray::new(
        TaggedVec::from(nodes),
        TaggedVec::from(edges),
    ))
}

/// The chains left in `sg`, ordered by their first member, each walked from its lower end (the
/// lower side id; a single node from its left side). Marks the links walked inside them in
/// `internal`.
fn kept_chains(sg: &SideGraph, chains: &mut Chains, internal: &mut [bool]) -> Vec<KeptChain> {
    let n = sg.lens.len();
    let roots: Vec<usize> = (0..n)
        .filter(|&v| sg.alive_node[v] && chains.find(v) == v)
        .collect();
    let mut kept = Vec::with_capacity(roots.len());
    let mut members = Vec::new();
    for r in roots {
        let (mut entry, mut exit) = chains.ends(r);
        let cyclic = chains.cyclic[r];
        if !cyclic && exit < entry {
            std::mem::swap(&mut entry, &mut exit);
        }
        sg.walk(entry, exit, &mut members, |e| internal[e] = true);
        kept.push(KeptChain {
            members: members.clone(),
            entry,
            exit,
            len: chains.len[r],
            cyclic,
        });
    }
    kept.sort_unstable_by_key(|c| c.members[0].node);
    kept
}

/// The G′ nodes, numbered in `kept` order: their runs go to `layout`, and `g_side` maps each outer
/// side to its G′ side.
fn g_prime_nodes<IndexType: GraphIndexInteger>(
    kept: &[KeptChain],
    sg: &SideGraph,
    layout: &mut RunLayout,
    g_side: &mut HashMap<usize, u32>,
    stats: &mut ShrinkStats,
) -> Result<Vec<LengthOnlyGfaNodeData>, String> {
    let index_max: usize = IndexType::max_value().try_into().unwrap_or(usize::MAX);
    let mut nodes = Vec::with_capacity(kept.len());
    for (gid, chain) in kept.iter().enumerate() {
        let len = chain.len;
        if len > OFF_MASK || len > index_max as u64 {
            return Err(format!(
                "merged node of length {len} exceeds the index width"
            ));
        }
        stats.max_node_len = stats.max_node_len.max(len);
        stats.cyclic_chains += usize::from(chain.cyclic);
        layout.add_runs(gid, false, &chain.members, &sg.lens);
        g_side.insert(chain.entry, side(gid, false) as u32);
        g_side.insert(chain.exit, side(gid, true) as u32);
        nodes.push(LengthOnlyGfaNodeData::with_len(
            &(gid + 1).to_string(),
            len as usize,
        ));
    }
    Ok(nodes)
}

/// The G′ links as G′ side pairs `(x, y)`, `x <= y`: the alive links between chain ends, sorted and
/// deduplicated.
fn g_prime_links(
    sg: &SideGraph,
    internal: &[bool],
    g_side: &HashMap<usize, u32>,
) -> Vec<(u32, u32)> {
    let mut links: Vec<(u32, u32)> = Vec::new();
    for (e, &(a, b)) in sg.edges.iter().enumerate() {
        if !sg.alive_edge[e] || internal[e] {
            continue;
        }
        let (x, y) = (g_side[&(a as usize)], g_side[&(b as usize)]);
        links.push((x.min(y), x.max(y)));
    }
    links.sort_unstable();
    links.dedup();
    links
}
