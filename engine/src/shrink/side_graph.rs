//! The side graph the reduction works on, and the two id spaces of the shrink.
//!
//! - A *side* id names one end of a node: `2v` is the left side of node `v`, `2v + 1` its right side.
//!   A link joins two sides. A directed node leaves through its exit side (forward: the right side)
//!   and is entered through its entry side (forward: the left side).
//! - A *directed* id names an oriented original node as the alias stores ports: `2v` is `v`
//!   forward, `2v + 1` reverse.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};

use crate::gfa_graph_extensions::GfaNodeDataExt;

/// The left or right side of `node`.
#[inline]
pub(super) fn side(node: usize, right: bool) -> usize {
    2 * node + usize::from(right)
}

/// The node side `s` belongs to.
#[inline]
pub(super) fn side_node(s: usize) -> usize {
    s / 2
}

/// Whether `s` is a right side.
#[inline]
pub(super) fn is_right(s: usize) -> bool {
    s % 2 == 1
}

/// The other side of the node `s` belongs to.
#[inline]
pub(super) fn other_side(s: usize) -> usize {
    s ^ 1
}

/// The side `node` walked `forward` (or not) leaves through.
#[inline]
pub(super) fn exit_side(node: usize, forward: bool) -> usize {
    side(node, forward)
}

/// The side `node` walked `forward` (or not) is entered through.
#[inline]
pub(super) fn entry_side(node: usize, forward: bool) -> usize {
    side(node, !forward)
}

/// The directed id of `node` walked `forward` (or not).
#[inline]
pub(super) fn dir_id(node: usize, forward: bool) -> u32 {
    (2 * node + usize::from(!forward)) as u32
}

/// The `(node, forward)` of a directed id.
#[inline]
pub(super) fn dir_decode(d: u32) -> (usize, bool) {
    ((d / 2) as usize, d.is_multiple_of(2))
}

/// The same node walked the other way.
#[inline]
pub(super) fn reverse_dir(d: u32) -> u32 {
    d ^ 1
}

/// A node of a chain as a walk through the chain passes it.
#[derive(Clone, Copy, Debug)]
pub(super) struct ChainMember {
    pub(super) node: usize,
    /// Entered through its left side.
    pub(super) forward: bool,
    /// Its start offset in the chain's walk.
    pub(super) offset: u64,
}

/// The mutable reduction state over the side graph (see the module docs for side ids).
pub(super) struct SideGraph {
    pub(super) lens: Vec<u64>,
    /// Zero-length nodes: never chained or pruned.
    pub(super) blocked: Vec<bool>,
    pub(super) alive_node: Vec<bool>,
    /// Deduplicated links as side pairs `(a, b)`, `a <= b`.
    pub(super) edges: Vec<(u32, u32)>,
    pub(super) alive_edge: Vec<bool>,
    /// CSR over sides: the links at each side (a link joining a side to itself appears twice).
    side_off: Vec<u32>,
    side_edges: Vec<u32>,
    /// Link endpoints per side among alive links.
    pub(super) deg: Vec<u32>,
}

impl SideGraph {
    /// The side graph of `graph`: its node lengths and its links deduplicated. `Err` when a link
    /// overlaps.
    pub(super) fn new<
        IndexType: GraphIndexInteger,
        NodeData: GfaNodeData,
        EdgeData: GfaEdgeData,
    >(
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    ) -> Result<Self, String> {
        let lens: Vec<u64> = (0..graph.node_count())
            .map(|v| {
                let len: crate::path::GfaPathLength<IndexType> =
                    graph.node_data(NodeIndex::from_usize(v)).len();
                len.into_usize() as u64
            })
            .collect();
        let edges = side_links(graph)?;
        let (side_off, side_edges) = links_by_side(2 * lens.len(), &edges);
        let deg: Vec<u32> = side_off.windows(2).map(|w| w[1] - w[0]).collect();
        Ok(Self {
            blocked: lens.iter().map(|&l| l == 0).collect(),
            alive_node: vec![true; lens.len()],
            lens,
            alive_edge: vec![true; edges.len()],
            edges,
            side_off,
            side_edges,
            deg,
        })
    }

    /// The alive links at side `s`.
    pub(super) fn links_at(&self, s: usize) -> impl Iterator<Item = usize> + '_ {
        self.side_edges[self.side_off[s] as usize..self.side_off[s + 1] as usize]
            .iter()
            .map(|&e| e as usize)
            .filter(|&e| self.alive_edge[e])
    }

    /// The side across link `e` from its endpoint `s`.
    fn across(&self, e: usize, s: usize) -> usize {
        let (a, b) = self.edges[e];
        let other = if a as usize == s { b } else { a };
        other as usize
    }

    /// The single alive link at side `s` and its other endpoint.
    pub(super) fn only_link(&self, s: usize) -> (usize, usize) {
        let e = self
            .links_at(s)
            .next()
            .unwrap_or_else(|| unreachable!("side {s} has degree 1 but no alive link"));
        (e, self.across(e, s))
    }

    /// Kill link `e`.
    fn kill_link(&mut self, e: usize) {
        if self.alive_edge[e] {
            self.alive_edge[e] = false;
            let (a, b) = self.edges[e];
            self.deg[a as usize] -= 1;
            self.deg[b as usize] -= 1;
        }
    }

    /// Whether the link between sides `s` and `t` is walked straight through (each side has only it).
    pub(super) fn chainable(&self, s: usize, t: usize) -> bool {
        side_node(s) != side_node(t)
            && self.deg[s] == 1
            && self.deg[t] == 1
            && !self.blocked[side_node(s)]
            && !self.blocked[side_node(t)]
    }

    /// Whether an alive link joins sides `p` and `q` directly (it dominates every allele between
    /// them).
    pub(super) fn has_direct_link(&self, p: u32, q: u32) -> bool {
        let key = (p.min(q), p.max(q));
        self.links_at(key.0 as usize).any(|e| self.edges[e] == key)
    }

    /// The members of the chain entered at side `a` and left at side `b`, in walk order; `internal`
    /// sees every link walked inside the chain.
    pub(super) fn walk(
        &self,
        a: usize,
        b: usize,
        members: &mut Vec<ChainMember>,
        mut internal: impl FnMut(usize),
    ) {
        members.clear();
        let mut entry = a;
        let mut offset = 0u64;
        loop {
            let node = side_node(entry);
            members.push(ChainMember {
                node,
                forward: !is_right(entry),
                offset,
            });
            offset += self.lens[node];
            let exit = other_side(entry);
            if exit == b {
                return;
            }
            let (e, t) = self.only_link(exit);
            internal(e);
            entry = t;
        }
    }

    /// Kill a node and its links.
    pub(super) fn remove_node(&mut self, v: usize) {
        self.alive_node[v] = false;
        for s in [side(v, false), side(v, true)] {
            let links: Vec<usize> = self.links_at(s).collect();
            for e in links {
                self.kill_link(e);
            }
        }
    }
}

/// The links of `graph` as side pairs `(a, b)`, `a <= b`, sorted and deduplicated. `Err` when a link
/// overlaps.
fn side_links<IndexType: GraphIndexInteger, NodeData, EdgeData: GfaEdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
) -> Result<Vec<(u32, u32)>, String> {
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(graph.edge_count());
    for e in graph.iter_edges() {
        let view = graph.edge(e);
        if view.data().overlap() != 0 {
            return Err("the graph has overlapping links".to_string());
        }
        let (from, to) = (view.from(), view.to());
        let a = exit_side(from.into_bidirected().into_usize(), from.is_forward()) as u32;
        let b = entry_side(to.into_bidirected().into_usize(), to.is_forward()) as u32;
        edges.push((a.min(b), a.max(b)));
    }
    edges.sort_unstable();
    edges.dedup();
    Ok(edges)
}

/// The links at each of `sides` sides as a CSR `(offsets, link ids)`.
fn links_by_side(sides: usize, edges: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let mut side_off = vec![0u32; sides + 1];
    for &(a, b) in edges {
        side_off[a as usize + 1] += 1;
        side_off[b as usize + 1] += 1;
    }
    for s in 0..sides {
        side_off[s + 1] += side_off[s];
    }
    let mut fill = side_off.clone();
    let mut side_edges = vec![0u32; side_off[sides] as usize];
    for (e, &(a, b)) in edges.iter().enumerate() {
        for s in [a, b] {
            side_edges[fill[s as usize] as usize] = e as u32;
            fill[s as usize] += 1;
        }
    }
    (side_off, side_edges)
}
