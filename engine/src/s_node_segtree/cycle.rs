//! The cycle of an S-node skeleton: its vertices in cycle order and the leaf networks between
//! consecutive positions, the input of both the ring form and the tree.

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};

use super::{SegVertexRec, orientation_bit};
use crate::{
    image::Padded,
    min_plus::{
        MinPlusMatrix,
        network::Mat4,
        word::{inf, relax},
    },
    orientation::Orientation,
    pole_lift::{Pole, slot},
    skeleton::SkeletonAccess,
};

/// The cycle edges as unordered node pairs (self-pairs dropped): the S-node's structure when given,
/// else the pairs the skeleton's arcs connect.
pub(super) fn cycle_pairs<IndexType: GraphIndexInteger, S: SkeletonAccess<IndexType> + ?Sized>(
    inner: &S,
    cycle_edges: Option<&[(NodeIndex<IndexType>, NodeIndex<IndexType>)]>,
) -> Vec<(usize, usize)> {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    match cycle_edges {
        Some(edges) => pairs.extend(edges.iter().map(|&(a, b)| (a.into_usize(), b.into_usize()))),
        None => {
            let (node_of_id, offsets, targets) =
                (inner.node_of_id(), inner.offsets(), inner.edge_target());
            for from_id in 0..node_of_id.len() {
                let u = node_of_id[from_id].into_bidirected().into_usize();
                for i in offsets[from_id] as usize..offsets[from_id + 1] as usize {
                    pairs.push((
                        u,
                        node_of_id[targets[i] as usize]
                            .into_bidirected()
                            .into_usize(),
                    ));
                }
            }
        }
    }
    pairs.retain(|&(a, b)| a != b);
    pairs
}

/// The vertices of an S-node skeleton in cycle order. A non-root node's order starts at its
/// reference pole `s` and ends at `t`, so the closing edge `(v_{m−1}, v_0)` is the reference edge.
pub(super) struct Cycle {
    /// The distinct vertex nodes, sorted: a vertex's local id is its index.
    nodes: Vec<usize>,
    /// The locals in cycle order.
    order: Vec<usize>,
    /// The cycle position of every local.
    pos_of_local: Vec<u32>,
}

impl Cycle {
    /// The cycle of the vertex `pairs`, or `None` (a decline) when they are not a single cycle or
    /// a single path of at least three vertices, or (for a non-root node) do not run from `s` to
    /// `t`.
    pub(super) fn of<IndexType: GraphIndexInteger>(
        pairs: &[(usize, usize)],
        reference_poles: Option<(NodeIndex<IndexType>, NodeIndex<IndexType>)>,
    ) -> Option<Self> {
        let mut nodes: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        nodes.sort_unstable();
        nodes.dedup();
        let m = nodes.len();
        if m < 3 {
            return None; // decline: fewer than 3 vertices
        }
        let mut cycle = Self {
            nodes,
            order: Vec::with_capacity(m),
            pos_of_local: vec![u32::MAX; m],
        };
        let (neighbours, endpoints) = cycle.vertex_adjacency(pairs)?;
        let (start, first_step) = cycle.walk_start(&neighbours, &endpoints, reference_poles)?;
        cycle.walk(&neighbours, start, first_step)?;
        if let Some((_, t)) = reference_poles
            && cycle.order[m - 1] != cycle.local_of(t.into_usize()).expect("checked above")
        {
            return None; // decline: walk does not end at the reference pole t
        }
        Some(cycle)
    }

    /// The number of cycle vertices.
    pub(super) fn m(&self) -> usize {
        self.nodes.len()
    }

    fn local_of(&self, node: usize) -> Option<usize> {
        self.nodes.binary_search(&node).ok()
    }

    /// The cycle neighbours of every vertex (`usize::MAX` = none) and the path endpoints (the
    /// vertices with one neighbour); `None` unless every degree is 1 or 2 and there are 0 or 2
    /// endpoints.
    #[allow(clippy::type_complexity)]
    fn vertex_adjacency(&self, pairs: &[(usize, usize)]) -> Option<(Vec<[usize; 2]>, Vec<usize>)> {
        let m = self.m();
        let mut neighbours: Vec<[usize; 2]> = vec![[usize::MAX; 2]; m];
        let mut degree = vec![0u8; m];
        for &(a, b) in pairs {
            let (u, v) = (
                self.local_of(a).expect("an endpoint"),
                self.local_of(b).expect("an endpoint"),
            );
            for (x, y) in [(u, v), (v, u)] {
                if neighbours[x].contains(&y) {
                    continue;
                }
                if degree[x] >= 2 {
                    return None; // decline: a vertex with 3+ cycle neighbours (branch)
                }
                neighbours[x][degree[x] as usize] = y;
                degree[x] += 1;
            }
        }
        let endpoints: Vec<usize> = (0..m).filter(|&v| degree[v] == 1).collect();
        if degree.contains(&0) || !matches!(endpoints.len(), 0 | 2) {
            // decline: not one cycle or one path (isolated vertex / odd endpoint count)
            return None;
        }
        Some((neighbours, endpoints))
    }

    /// The walk's first vertex and its first step. Non-root: start at `s` and step away from `t`
    /// (on a path, `s` and `t` must be its ends). Root: any start on the cycle (or an endpoint,
    /// should the ring not be closed).
    fn walk_start<IndexType: GraphIndexInteger>(
        &self,
        neighbours: &[[usize; 2]],
        endpoints: &[usize],
        reference_poles: Option<(NodeIndex<IndexType>, NodeIndex<IndexType>)>,
    ) -> Option<(usize, usize)> {
        let Some((s, t)) = reference_poles else {
            let start = if endpoints.len() == 2 {
                endpoints[0]
            } else {
                0
            };
            return Some((start, neighbours[start][0]));
        };
        let (Some(ls), Some(lt)) = (self.local_of(s.into_usize()), self.local_of(t.into_usize()))
        else {
            return None; // decline: a reference pole is not a vertex
        };
        if ls == lt {
            return None; // decline: reference poles coincide
        }
        if endpoints.len() == 2 {
            if !(endpoints.contains(&ls) && endpoints.contains(&lt)) {
                return None; // decline: open path whose ends are not the reference poles
            }
            return Some((ls, neighbours[ls][0]));
        }
        // A closed ring already containing an `s`–`t` edge: walk away from `t` first so the walk
        // ends at `t` and that edge becomes the closing edge.
        let [n0, n1] = neighbours[ls];
        if n0 != lt && n1 != lt {
            return None; // decline: closed ring without an s-t edge for a non-root
        }
        Some((ls, if n0 == lt { n1 } else { n0 }))
    }

    /// Walk the cycle from `start` (first step `first_step`), numbering the positions; `None` unless
    /// the walk visits every vertex exactly once.
    fn walk(&mut self, neighbours: &[[usize; 2]], start: usize, first_step: usize) -> Option<()> {
        let m = self.m();
        let (mut prev, mut cur) = (start, first_step);
        self.order.push(start);
        self.pos_of_local[start] = 0;
        while self.order.len() < m {
            if self.pos_of_local[cur] != u32::MAX {
                return None; // decline: walk closed early (several cycles)
            }
            self.pos_of_local[cur] = self.order.len() as u32;
            self.order.push(cur);
            let [n0, n1] = neighbours[cur];
            let next = if n0 != prev { n0 } else { n1 };
            if next == usize::MAX {
                break; // reached the far end of an open path
            }
            prev = cur;
            cur = next;
        }
        if self.order.len() != m {
            return None; // decline: walk did not visit every vertex
        }
        Some(())
    }

    /// The node length of each cycle position (a virtual edge's pass arc weighs the child's inside
    /// distance, not the node length, so the lengths come from the graph, not the arcs).
    pub(super) fn lens<IndexType: GraphIndexInteger>(
        &self,
        len_of: &dyn Fn(NodeIndex<IndexType>) -> u64,
    ) -> Vec<u64> {
        self.order
            .iter()
            .map(|&local| len_of(NodeIndex::from_usize(self.nodes[local])))
            .collect()
    }

    /// The raw leaf networks of the skeleton's arcs: `leaves[e]` over `(v_e, v_{e+1})` carries the
    /// pass-through arcs of that edge and the rungs of `v_e`; `leaves[m−1]` is the closing network
    /// over `(v_{m−1}, v_0)`. `None` (a decline) on an arc that leaves the cycle or is a chord.
    pub(super) fn leaf_networks<IndexType, S>(&self, inner: &S) -> Option<Vec<Mat4<IndexType>>>
    where
        IndexType: GraphIndexInteger,
        S: SkeletonAccess<IndexType> + ?Sized,
    {
        let m = self.m();
        let node_of_id = inner.node_of_id();
        let (offsets, targets, weights) =
            (inner.offsets(), inner.edge_target(), inner.edge_weight());
        let mut leaves = vec![[inf::<IndexType>(); 16]; m];
        for from_id in 0..node_of_id.len() {
            let from = node_of_id[from_id];
            let Some(lf) = self.local_of(from.into_bidirected().into_usize()) else {
                return None; // decline: an arc endpoint outside the cycle
            };
            let (ps, os) = (self.pos_of_local[lf] as usize, orientation_bit(from));
            for i in offsets[from_id] as usize..offsets[from_id + 1] as usize {
                let to = node_of_id[targets[i] as usize];
                let Some(lt) = self.local_of(to.into_bidirected().into_usize()) else {
                    return None; // decline: an arc endpoint outside the cycle
                };
                let (pt, ot) = (self.pos_of_local[lt] as usize, orientation_bit(to));
                let w: IndexType = weights[i].into_raw();
                let cell = if ps == pt {
                    if os == ot {
                        continue; // a zero self-loop: never on a shortest path
                    }
                    &mut leaves[ps][4 * os + ot]
                } else if pt == ps + 1 {
                    &mut leaves[ps][4 * os + 2 + ot]
                } else if ps == pt + 1 {
                    &mut leaves[pt][4 * (2 + os) + ot]
                } else if ps == m - 1 && pt == 0 {
                    &mut leaves[m - 1][4 * os + 2 + ot]
                } else if ps == 0 && pt == m - 1 {
                    &mut leaves[m - 1][4 * (2 + os) + ot]
                } else {
                    return None; // decline: chord (arc between non-adjacent cycle positions)
                };
                relax(cell, w);
            }
        }
        Some(leaves)
    }

    /// The vertex records: every vertex's node and cycle position, sorted by node.
    pub(super) fn vertex_recs<IndexType: GraphIndexInteger>(&self) -> Vec<SegVertexRec<IndexType>> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(local, &node)| SegVertexRec {
                node: NodeIndex::from_usize(node),
                pos: Padded::new(self.pos_of_local[local]),
            })
            .collect()
    }
}

/// The LCA-role closing network: the inner one with the node's `D_out` folded in (a non-root node
/// with an outside matrix; otherwise the inner one). `D_out` is indexed by [`slot`] over `(s, t)`;
/// the closing network's poles are `(v_{m−1}, v_0) = (t, s)`, so `s` maps to states `2..4` and `t`
/// to `0..2`.
pub(super) fn closing_with_outside<IndexType: GraphIndexInteger>(
    closing_inner: &Mat4<IndexType>,
    reference_poles: Option<(NodeIndex<IndexType>, NodeIndex<IndexType>)>,
    outside: Option<&MinPlusMatrix<IndexType, 4>>,
) -> Mat4<IndexType> {
    let mut closing_lca = *closing_inner;
    let (Some(_), Some(outside)) = (reference_poles, outside) else {
        return closing_lca;
    };
    let idx = |pole: Pole, o: Orientation| match pole {
        Pole::Source => 2 + o.index(),
        Pole::Target => o.index(),
    };
    for from_pole in Pole::all() {
        for from_o in Orientation::all() {
            for to_pole in Pole::all() {
                for to_o in Orientation::all() {
                    if from_pole == to_pole && from_o == to_o {
                        continue;
                    }
                    if let Some(w) = outside
                        .get(slot(from_pole, from_o), slot(to_pole, to_o))
                        .into_option()
                    {
                        relax(
                            &mut closing_lca[4 * idx(from_pole, from_o) + idx(to_pole, to_o)],
                            w.into_raw(),
                        );
                    }
                }
            }
        }
    }
    closing_lca
}
