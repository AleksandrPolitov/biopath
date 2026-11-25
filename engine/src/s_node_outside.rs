//! Linear-time outside matrices for the children of an **S-node** parent.
//!
//! The outside DP ([`crate::outside_dp`]) needs, for every child `C` of a parent `P`, the
//! all-pairs distances among `C`'s four pole states in `P`'s *augmented* skeleton (the inner skeleton
//! with every child's inside matrix, plus `P`'s own outside at its reference poles). That skeleton is
//! the same for all children, and the general path runs one Dijkstra over it per distinct child pole
//! state. On an S-node with `k` cycle vertices that is ~`2k` searches over ~`2k` states, i.e.
//! `O(k² log k)` per S-node.
//!
//! An S-node skeleton is a cycle `v_0 … v_{k-1}`: every arc of the augmented skeleton either joins two
//! consecutive cycle vertices or stays on one vertex (a same-pole flip through a child, a hosted
//! self-loop, or a baked flip). Cutting the cycle at edge `i` leaves two 2-terminal networks between
//! `v_i` and `v_{i+1}`: the edge itself (`N_i`) and the path around the rest of the cycle (`R_i`).
//! A walk between two terminal states alternates between the two freely, so the answer is their
//! parallel composition `close4(N_i ⊕ R_i)`. Every `R_i` is a suffix product times a prefix product
//! of the series `merge` monoid from [`crate::min_plus::network`], so all `k` of them cost
//! `O(k)` merges in total.
//!
//! Exact: the networks are built from the very arcs the general path searches, and the min-plus
//! closure computes the same integer shortest distances as Dijkstra. Anything that is not one clean
//! cycle of at least three vertices returns `None`, and the caller falls back to the general path.

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex};

use crate::{
    fast_hash::FastMap,
    inside_dp::SkeletonAdjacency,
    min_plus::{
        MinPlus, MinPlusMatrix,
        network::{Mat4, close_network, merge, parallel, swap_poles},
        word::{inf, relax},
    },
    orientation::Orientation,
    pole_lift::{Pole, slot},
};

#[cfg(test)]
mod tests;

#[inline]
fn orientation_of<IndexType: GraphIndexInteger>(node: DirectedNodeIndex<IndexType>) -> usize {
    if node.is_forward() { 0 } else { 1 }
}

/// The outside matrix of every child of an S-node parent, or `None` if `adjacency` is not a clean
/// cycle (the caller then runs the general per-source Dijkstra, which is always correct).
///
/// `adjacency` is the parent's augmented skeleton exactly as the general path would search it;
/// `children` lists each child's reference poles `(s_C, t_C)`. The result is aligned with `children`,
/// each matrix indexed by [`slot`] over `(s_C, t_C)` like [`crate::outside_dp::OutsideDp`].
pub(crate) fn cycle_outside_matrices<IndexType: GraphIndexInteger>(
    adjacency: &SkeletonAdjacency<IndexType>,
    children: &[(NodeIndex<IndexType>, NodeIndex<IndexType>)],
) -> Option<Vec<MinPlusMatrix<IndexType, 4>>> {
    let arcs = CycleArcs::split(adjacency);
    let order = arcs.cycle_order()?;
    let cycle = CutCycle::new(&arcs, order)?;
    children
        .iter()
        .map(|&poles| cycle.child_outside(&arcs, poles))
        .collect()
}

/// The arcs of an augmented S-node skeleton, split into per-vertex rungs and per-vertex-pair edge
/// matrices. Vertices are numbered in order of first appearance.
struct CycleArcs<IndexType: GraphIndexInteger> {
    id_of: FastMap<NodeIndex<IndexType>, u32>,
    /// `rungs[v]` = the 2x2 same-vertex arcs at `v`, `[2*o_from + o_to]`.
    rungs: Vec<[IndexType; 4]>,
    /// Inter-vertex arcs per unordered pair `(lo, hi)`, over states `[lo+, lo-, hi+, hi-]`.
    pairs: FastMap<(u32, u32), Mat4<IndexType>>,
}

impl<IndexType: GraphIndexInteger> CycleArcs<IndexType> {
    fn split(adjacency: &SkeletonAdjacency<IndexType>) -> Self {
        let mut arcs = Self {
            id_of: FastMap::default(),
            rungs: Vec::new(),
            pairs: FastMap::default(),
        };
        for (&from, out) in adjacency {
            let u = arcs.intern(from.into_bidirected());
            let ou = orientation_of(from);
            for &(to, weight) in out {
                let v = arcs.intern(to.into_bidirected());
                arcs.add(u, ou, v, orientation_of(to), weight.into_raw());
            }
        }
        arcs
    }

    fn intern(&mut self, segment: NodeIndex<IndexType>) -> u32 {
        let rungs = &mut self.rungs;
        *self.id_of.entry(segment).or_insert_with(|| {
            rungs.push([inf(); 4]);
            (rungs.len() - 1) as u32
        })
    }

    /// Add the arc `(u, ou) -> (v, ov)` of weight `w`: a rung when `u == v`.
    fn add(&mut self, u: u32, ou: usize, v: u32, ov: usize, w: IndexType) {
        if u == v {
            relax(&mut self.rungs[u as usize][2 * ou + ov], w);
            return;
        }
        let (lo, hi) = if u < v { (u, v) } else { (v, u) };
        let matrix = self.pairs.entry((lo, hi)).or_insert([inf(); 16]);
        let su = if u == lo { ou } else { 2 + ou };
        let sv = if v == lo { ov } else { 2 + ov };
        relax(&mut matrix[4 * su + sv], w);
    }

    /// The vertices in cycle order from vertex `0`, or `None` unless the pairs form one cycle of at
    /// least three vertices (every vertex with exactly two neighbours).
    fn cycle_order(&self) -> Option<Vec<u32>> {
        let k = self.rungs.len();
        if k < 3 {
            return None;
        }
        let mut neighbours: Vec<[u32; 2]> = vec![[u32::MAX; 2]; k];
        let mut degree = vec![0u8; k];
        for &(a, b) in self.pairs.keys() {
            for (x, y) in [(a, b), (b, a)] {
                let d = &mut degree[x as usize];
                if *d >= 2 {
                    return None;
                }
                neighbours[x as usize][*d as usize] = y;
                *d += 1;
            }
        }
        if degree.iter().any(|&d| d != 2) {
            return None;
        }
        let mut order: Vec<u32> = Vec::with_capacity(k);
        let (mut previous, mut current) = (u32::MAX, 0u32);
        loop {
            order.push(current);
            let [n0, n1] = neighbours[current as usize];
            let next = if n0 != previous { n0 } else { n1 };
            previous = current;
            current = next;
            if current == 0 {
                break;
            }
            if order.len() >= k {
                return None; // walked past every vertex without closing: not a single cycle
            }
        }
        // Closed early: the arcs form several disjoint cycles.
        (order.len() == k).then_some(order)
    }
}

/// The cycle `v_0 … v_{k-1}` cut into its closed edge networks `N_i` over `(v_i, v_{i+1})`, with
/// the prefix and suffix series products that give every rest of the cycle `R_i`.
struct CutCycle<IndexType: GraphIndexInteger> {
    order: Vec<u32>,
    network: Vec<Mat4<IndexType>>,
    /// The cycle edge of each unordered vertex pair.
    edge_of: FastMap<(u32, u32), usize>,
    /// `prefix[j] = N_0 ⋈ … ⋈ N_{j-1}` over `(v_0, v_j)`, for `1 <= j <= k-1`.
    prefix: Vec<Mat4<IndexType>>,
    /// `suffix[j] = N_j ⋈ … ⋈ N_{k-1}` over `(v_j, v_0)`, for `1 <= j <= k-1`.
    suffix: Vec<Mat4<IndexType>>,
}

impl<IndexType: GraphIndexInteger> CutCycle<IndexType> {
    /// The closed networks and their prefix / suffix products, or `None` if two consecutive
    /// vertices of `order` share no arc.
    fn new(arcs: &CycleArcs<IndexType>, order: Vec<u32>) -> Option<Self> {
        let k = order.len();
        let mut edge_of: FastMap<(u32, u32), usize> = FastMap::default();
        let mut network: Vec<Mat4<IndexType>> = Vec::with_capacity(k);
        for i in 0..k {
            let (a, b) = (order[i], order[(i + 1) % k]);
            let key = if a < b { (a, b) } else { (b, a) };
            let stored = arcs.pairs.get(&key)?;
            network.push(closed_edge(stored, a == key.0, &arcs.rungs[a as usize]));
            edge_of.insert(key, i);
        }
        let mut prefix: Vec<Mat4<IndexType>> = vec![[inf(); 16]; k];
        let mut suffix: Vec<Mat4<IndexType>> = vec![[inf(); 16]; k];
        prefix[1] = network[0];
        for j in 1..k - 1 {
            prefix[j + 1] = merge(&prefix[j], &network[j]);
        }
        suffix[k - 1] = network[k - 1];
        for j in (1..k - 1).rev() {
            suffix[j] = merge(&network[j], &suffix[j + 1]);
        }
        Some(Self {
            order,
            network,
            edge_of,
            prefix,
            suffix,
        })
    }

    /// The outside matrix of the child with reference poles `(s, t)`: the cycle closure at its
    /// edge, mapped onto its pole slots; `None` if its poles are not the ends of a cycle edge.
    fn child_outside(
        &self,
        arcs: &CycleArcs<IndexType>,
        (s, t): (NodeIndex<IndexType>, NodeIndex<IndexType>),
    ) -> Option<MinPlusMatrix<IndexType, 4>> {
        let k = self.order.len();
        let (sa, ta) = (*arcs.id_of.get(&s)?, *arcs.id_of.get(&t)?);
        if sa == ta {
            return None;
        }
        let i = *self
            .edge_of
            .get(&if sa < ta { (sa, ta) } else { (ta, sa) })?;
        // `R_i` runs from `v_{i+1}` around the cycle to `v_i`.
        let rest = if i == 0 {
            self.suffix[1]
        } else if i == k - 1 {
            self.prefix[k - 1]
        } else {
            merge(&self.suffix[i + 1], &self.prefix[i])
        };
        let closed = parallel(&swap_poles(&rest), &self.network[i]);
        Some(pole_slots(&closed, self.order[i], (sa, ta)))
    }
}

/// The closed network of one cycle edge from its stored arcs (`forward`: stored over the edge's own
/// `(v_i, v_{i+1})` order) and the rung of its left terminal `v_i`. Each vertex's rung is attached
/// to exactly one network, the one where it is the left terminal, so every rung appears once in both
/// the edge and the complement of every cut.
fn closed_edge<IndexType: GraphIndexInteger>(
    stored: &Mat4<IndexType>,
    forward: bool,
    rung: &[IndexType; 4],
) -> Mat4<IndexType> {
    let mut n = if forward { *stored } else { swap_poles(stored) };
    for oi in 0..2 {
        for oj in 0..2 {
            relax(&mut n[4 * oi + oj], rung[2 * oi + oj]);
        }
    }
    close_network(&mut n);
    n
}

/// A closed `4x4` over `[(left, ±), (other, ±)]` as a pole-state matrix over the poles `(sa, ta)`.
fn pole_slots<IndexType: GraphIndexInteger>(
    closed: &Mat4<IndexType>,
    left: u32,
    (sa, ta): (u32, u32),
) -> MinPlusMatrix<IndexType, 4> {
    let state = |vertex: u32, orientation: Orientation| {
        if vertex == left {
            orientation.index()
        } else {
            2 + orientation.index()
        }
    };
    let mut matrix = MinPlusMatrix::<IndexType, 4>::infinity();
    for (from_pole, from_vertex) in [(Pole::Source, sa), (Pole::Target, ta)] {
        for from_orientation in Orientation::all() {
            for (to_pole, to_vertex) in [(Pole::Source, sa), (Pole::Target, ta)] {
                for to_orientation in Orientation::all() {
                    matrix.set(
                        slot(from_pole, from_orientation),
                        slot(to_pole, to_orientation),
                        MinPlus::from_word(
                            closed[4 * state(from_vertex, from_orientation)
                                + state(to_vertex, to_orientation)],
                        ),
                    );
                }
            }
        }
    }
    matrix
}
