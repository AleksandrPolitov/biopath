use bidirected_adjacency_array::index::{DirectedNodeIndex, NodeIndex};

use super::cycle_outside_matrices;
use crate::{
    inside_dp::{SkeletonAdjacency, pole_state_matrix},
    path::GfaPathLength,
    skeleton::{DijkstraWorkspace, Skeleton},
};

/// Deterministic xorshift, like the other fuzz tests in this crate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn state(segment: NodeIndex<u32>, orientation: u64) -> DirectedNodeIndex<u32> {
    segment.into_directed(orientation == 0)
}

fn push(
    adjacency: &mut SkeletonAdjacency<u32>,
    from: DirectedNodeIndex<u32>,
    to: DirectedNodeIndex<u32>,
    w: u64,
) {
    adjacency
        .entry(from)
        .or_default()
        .push((to, GfaPathLength::from_usize(w as usize)));
}

/// The cycle edges of a skeleton as `(s, t)` pole pairs.
type PolePairs = Vec<(NodeIndex<u32>, NodeIndex<u32>)>;

/// A random bidirected cycle skeleton over `k` vertices with scattered segment ids. Every cycle edge
/// gets at least one arc between its two endpoints; some vertices get same-vertex rungs. Returns the
/// adjacency and the cycle edges as `(s, t)` pole pairs in a random pole order.
fn random_cycle(rng: &mut Rng, k: usize) -> (SkeletonAdjacency<u32>, PolePairs) {
    let mut segments: Vec<NodeIndex<u32>> = Vec::new();
    while segments.len() < k {
        let s = NodeIndex::from_usize(rng.below(10_000) as usize);
        if !segments.contains(&s) {
            segments.push(s);
        }
    }
    let mut adjacency = SkeletonAdjacency::<u32>::default();
    let mut poles = Vec::new();
    for i in 0..k {
        let (u, v) = (segments[i], segments[(i + 1) % k]);
        // Sparse random arcs between the edge's two endpoints, both directions, including
        // zero weights (inside matrices carry 0 pole-state distances).
        let mut any = false;
        for _ in 0..(1 + rng.below(6)) {
            let (a, b) = if rng.below(2) == 0 { (u, v) } else { (v, u) };
            push(
                &mut adjacency,
                state(a, rng.below(2)),
                state(b, rng.below(2)),
                rng.below(20),
            );
            any = true;
        }
        assert!(any);
        if rng.below(3) == 0 {
            // A same-vertex rung (a flip through a child, or a hosted inverted self-loop).
            let (a, b) = (rng.below(2), rng.below(2));
            push(&mut adjacency, state(u, a), state(u, b), rng.below(20));
        }
        poles.push(if rng.below(2) == 0 { (u, v) } else { (v, u) });
    }
    (adjacency, poles)
}

/// The general path: one Dijkstra per pole state over the whole augmented skeleton.
fn reference(
    adjacency: &SkeletonAdjacency<u32>,
    poles: &[(NodeIndex<u32>, NodeIndex<u32>)],
) -> Vec<crate::min_plus::MinPlusMatrix<u32, 4>> {
    let skeleton = Skeleton::from_adjacency(adjacency);
    let mut workspace = DijkstraWorkspace::new();
    poles
        .iter()
        .map(|&p| pole_state_matrix(&skeleton, &mut workspace, p, p))
        .collect()
}

#[test]
fn random_cycles_match_dijkstra() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut checked = 0;
    for round in 0..3000 {
        let k = 3 + (round % 14);
        let (adjacency, poles) = random_cycle(&mut rng, k);
        let fast = cycle_outside_matrices(&adjacency, &poles)
            .unwrap_or_else(|| panic!("round {round}: a clean {k}-cycle must be accepted"));
        let slow = reference(&adjacency, &poles);
        for (child, (f, s)) in fast.iter().zip(&slow).enumerate() {
            assert_eq!(f, s, "round {round}, k {k}, child {child}");
            checked += 1;
        }
    }
    assert!(checked > 20_000);
}

#[test]
fn long_cycles_match_dijkstra() {
    let mut rng = Rng(12345);
    for k in [50, 257, 1000] {
        let (adjacency, poles) = random_cycle(&mut rng, k);
        let fast = cycle_outside_matrices(&adjacency, &poles).unwrap();
        assert_eq!(fast, reference(&adjacency, &poles), "k {k}");
    }
}

#[test]
fn declines_anything_that_is_not_one_cycle() {
    let seg = |i: usize| NodeIndex::<u32>::from_usize(i);
    let arc = |adjacency: &mut SkeletonAdjacency<u32>, a: usize, b: usize| {
        push(adjacency, state(seg(a), 0), state(seg(b), 0), 1)
    };

    // A path (the cycle is open between 0 and 3).
    let mut path = SkeletonAdjacency::<u32>::default();
    for (a, b) in [(0, 1), (1, 2), (2, 3)] {
        arc(&mut path, a, b);
    }
    assert!(cycle_outside_matrices(&path, &[(seg(0), seg(1))]).is_none());

    // A vertex of degree 3.
    let mut chord = SkeletonAdjacency::<u32>::default();
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)] {
        arc(&mut chord, a, b);
    }
    assert!(cycle_outside_matrices(&chord, &[(seg(0), seg(1))]).is_none());

    // Two disjoint triangles.
    let mut two = SkeletonAdjacency::<u32>::default();
    for (a, b) in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
        arc(&mut two, a, b);
    }
    assert!(cycle_outside_matrices(&two, &[(seg(0), seg(1))]).is_none());

    // A 2-cycle (a bond, never an S-node).
    let mut bond = SkeletonAdjacency::<u32>::default();
    arc(&mut bond, 0, 1);
    arc(&mut bond, 1, 0);
    assert!(cycle_outside_matrices(&bond, &[(seg(0), seg(1))]).is_none());

    // A child whose poles are not a cycle edge.
    let mut square = SkeletonAdjacency::<u32>::default();
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        arc(&mut square, a, b);
    }
    assert!(cycle_outside_matrices(&square, &[(seg(0), seg(1))]).is_some());
    assert!(cycle_outside_matrices(&square, &[(seg(0), seg(2))]).is_none());
    assert!(cycle_outside_matrices(&square, &[(seg(0), seg(9))]).is_none());
}

#[test]
fn isolated_pole_states_match_dijkstra() {
    // Only forward states carry arcs, so every reverse pole state is absent from the skeleton: the
    // general path gives 0 on its diagonal and ∞ elsewhere, and so must the closed form.
    let seg = |i: usize| NodeIndex::<u32>::from_usize(i);
    let mut adjacency = SkeletonAdjacency::<u32>::default();
    for (a, b, w) in [(0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 0, 6)] {
        push(&mut adjacency, state(seg(a), 0), state(seg(b), 0), w);
    }
    let poles: Vec<_> = (0..4).map(|i| (seg(i), seg((i + 1) % 4))).collect();
    let fast = cycle_outside_matrices(&adjacency, &poles).unwrap();
    assert_eq!(fast, reference(&adjacency, &poles));
}
