//! Differential tests: the segment tree must reproduce the skeleton Dijkstra on random bidirected
//! rings — open paths (a non-root node's inner skeleton), closed rings (a root), both roles, and
//! vertex ids that are not in cycle order.

use bidirected_adjacency_array::index::{DirectedNodeIndex, NodeIndex};

use super::*;
use crate::{
    inside_dp::SkeletonAdjacency,
    min_plus::{MinPlus, MinPlusMatrix},
    orientation::Orientation,
    path::GfaPathLength,
    pole_lift::{Pole, add_pole_state_edges, slot},
    skeleton::{DijkstraWorkspace, Skeleton, SkeletonAccess},
};

type I = u32;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

fn directed(node: usize, o: usize) -> DirectedNodeIndex<I> {
    NodeIndex::<I>::from_usize(node).into_directed(o == 0)
}

fn push(
    adj: &mut SkeletonAdjacency<I>,
    from: DirectedNodeIndex<I>,
    to: DirectedNodeIndex<I>,
    w: usize,
) {
    adj.entry(from)
        .or_default()
        .push((to, GfaPathLength::from_usize(w)));
}

/// A random ring over `m` vertices with graph ids `ids[pos]`. Every cycle edge `(pos, pos+1)` — and
/// the closing edge when `closed` — gets a random subset of the 8 arcs between its endpoints' states;
/// every vertex gets random rungs and sometimes a zero self-loop (which the folds emit and the tree
/// must ignore). Returns the adjacency and the ids in cycle order.
fn random_ring(
    rng: &mut Lcg,
    m: usize,
    closed: bool,
    shuffle: bool,
) -> (SkeletonAdjacency<I>, Vec<usize>) {
    let mut ids: Vec<usize> = (0..m).map(|i| 7 * i + 3).collect();
    if shuffle {
        for i in (1..m).rev() {
            let j = rng.below(i + 1);
            ids.swap(i, j);
        }
    }
    let mut adj: SkeletonAdjacency<I> = SkeletonAdjacency::default();
    let edges = if closed { m } else { m - 1 };
    for e in 0..edges {
        let (a, b) = (ids[e], ids[(e + 1) % m]);
        let mut any = false;
        for oa in 0..2 {
            for ob in 0..2 {
                if rng.chance(60) {
                    push(
                        &mut adj,
                        directed(a, oa),
                        directed(b, ob),
                        1 + rng.below(20),
                    );
                    any = true;
                }
                if rng.chance(60) {
                    push(
                        &mut adj,
                        directed(b, ob),
                        directed(a, oa),
                        1 + rng.below(20),
                    );
                    any = true;
                }
            }
        }
        if !any {
            // Keep the ring connected so the vertex set is the full cycle.
            push(&mut adj, directed(a, 0), directed(b, 0), 1 + rng.below(20));
        }
    }
    for &v in &ids {
        for o in 0..2 {
            if rng.chance(40) {
                push(
                    &mut adj,
                    directed(v, o),
                    directed(v, 1 - o),
                    1 + rng.below(30),
                );
            }
            if rng.chance(30) {
                push(&mut adj, directed(v, o), directed(v, o), 0);
            }
        }
    }
    (adj, ids)
}

fn random_outside(rng: &mut Lcg) -> MinPlusMatrix<I, 4> {
    let mut m = MinPlusMatrix::<I, 4>::infinity();
    for i in 0..4 {
        for j in 0..4 {
            if i == j {
                if rng.chance(50) {
                    m.set(i, j, MinPlus::finite_usize(0));
                }
            } else if rng.chance(55) {
                m.set(i, j, MinPlus::finite_usize(1 + rng.below(40)));
            }
        }
    }
    m
}

/// All-pairs `d` over a skeleton by Dijkstra, indexed by directed node.
fn dijkstra_all_pairs(
    skeleton: &Skeleton<I>,
) -> impl Fn(DirectedNodeIndex<I>, DirectedNodeIndex<I>) -> Option<GfaPathLength<I>> + '_ {
    move |from, to| {
        let mut ws = DijkstraWorkspace::new();
        let id = skeleton.node_id(from)?;
        ws.run(skeleton, &[(id, GfaPathLength::from_usize(0))]);
        skeleton.node_id(to).and_then(|t| ws.get(t))
    }
}

fn states(ids: &[usize]) -> Vec<DirectedNodeIndex<I>> {
    ids.iter()
        .flat_map(|&v| [directed(v, 0), directed(v, 1)])
        .collect()
}

fn check_role(
    tree: &SNodeSegTree<I>,
    skeleton: &Skeleton<I>,
    role: Role,
    ids: &[usize],
    what: &str,
) {
    let oracle = dijkstra_all_pairs(skeleton);
    for &from in &states(ids) {
        for &to in &states(ids) {
            let got = tree.distance(from, to, role);
            let want = if from == to {
                Some(GfaPathLength::from_usize(0))
            } else {
                oracle(from, to)
            };
            assert_eq!(got, want, "{what}: {role:?} {from:?} -> {to:?}");
        }
    }
}

#[test]
fn open_path_matches_dijkstra_in_both_roles() {
    let mut rng = Lcg(0x5EC7_1234);
    for case in 0..300 {
        let m = 3 + rng.below(10);
        let (adj, ids) = random_ring(&mut rng, m, false, case % 2 == 1);
        let inner = Skeleton::from_adjacency(&adj);
        let poles = (
            NodeIndex::from_usize(ids[0]),
            NodeIndex::from_usize(ids[m - 1]),
        );
        let outside = random_outside(&mut rng);
        let tree = SNodeSegTree::build(&inner, Some(poles), Some(&outside), &|_| 1, None)
            .unwrap_or_else(|| panic!("case {case}: a clean path must build"));
        assert_eq!(tree.m(), m);
        check_role(&tree, &inner, Role::Inner, &ids, &format!("case {case}"));
        let mut with = adj.clone();
        add_pole_state_edges(&mut with, poles.0, poles.1, &outside);
        let lca = Skeleton::from_adjacency(&with);
        check_role(&tree, &lca, Role::Lca, &ids, &format!("case {case}"));

        // Pole projection = inner distances from the seed to `s^±` and `t^±`, in slot order.
        let oracle = dijkstra_all_pairs(&inner);
        for &seed in &states(&ids) {
            let got = tree.pole_projection(seed).expect("seed is a vertex");
            for (pole, vertex) in [(Pole::Source, ids[0]), (Pole::Target, ids[m - 1])] {
                for o in 0..2 {
                    let target = directed(vertex, o);
                    let want = if seed == target {
                        Some(GfaPathLength::from_usize(0))
                    } else {
                        oracle(seed, target)
                    };
                    assert_eq!(
                        got[slot(pole, Orientation::from_index(o))].into_option(),
                        want,
                        "case {case}: projection {seed:?} -> {target:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn closed_ring_root_matches_dijkstra() {
    let mut rng = Lcg(0xC10_5ED);
    for case in 0..300 {
        let m = 3 + rng.below(10);
        let (adj, ids) = random_ring(&mut rng, m, true, case % 3 == 0);
        let skeleton = Skeleton::from_adjacency(&adj);
        let tree = SNodeSegTree::build(&skeleton, None, None, &|_| 1, None)
            .unwrap_or_else(|| panic!("case {case}: a clean ring must build"));
        check_role(&tree, &skeleton, Role::Inner, &ids, &format!("case {case}"));
        check_role(&tree, &skeleton, Role::Lca, &ids, &format!("case {case}"));
    }
}

#[test]
fn closed_ring_with_reference_edge_inside_matches_dijkstra() {
    // A non-root whose inner skeleton already contains arcs between its reference poles: the tree
    // must still orient the cycle so that edge is the closing edge, and fold `D_out` onto it.
    let mut rng = Lcg(0xA11_C1E);
    for case in 0..200 {
        let m = 3 + rng.below(8);
        let (adj, ids) = random_ring(&mut rng, m, true, case % 2 == 0);
        let inner = Skeleton::from_adjacency(&adj);
        let poles = (
            NodeIndex::from_usize(ids[0]),
            NodeIndex::from_usize(ids[m - 1]),
        );
        let outside = random_outside(&mut rng);
        let tree = SNodeSegTree::build(&inner, Some(poles), Some(&outside), &|_| 1, None)
            .unwrap_or_else(|| panic!("case {case}: must build"));
        check_role(&tree, &inner, Role::Inner, &ids, &format!("case {case}"));
        let mut with = adj.clone();
        add_pole_state_edges(&mut with, poles.0, poles.1, &outside);
        check_role(
            &tree,
            &Skeleton::from_adjacency(&with),
            Role::Lca,
            &ids,
            &format!("case {case}"),
        );
    }
}

#[test]
fn declines_non_rings() {
    let mut rng = Lcg(7);
    // A chord.
    let (mut adj, ids) = random_ring(&mut rng, 6, true, false);
    push(&mut adj, directed(ids[0], 0), directed(ids[3], 0), 5);
    assert!(
        SNodeSegTree::build(&Skeleton::from_adjacency(&adj), None, None, &|_| 1, None).is_none()
    );
    // A branch (degree 3).
    let (mut adj, ids) = random_ring(&mut rng, 5, true, false);
    push(&mut adj, directed(ids[1], 0), directed(99, 0), 5);
    push(&mut adj, directed(99, 0), directed(ids[1], 0), 5);
    assert!(
        SNodeSegTree::build(&Skeleton::from_adjacency(&adj), None, None, &|_| 1, None).is_none()
    );
    // Too short.
    let (adj, _) = random_ring(&mut rng, 2, true, false);
    assert!(
        SNodeSegTree::build(&Skeleton::from_adjacency(&adj), None, None, &|_| 1, None).is_none()
    );
    // An open path whose ends are not the reference poles.
    let (adj, ids) = random_ring(&mut rng, 5, false, false);
    let bad_poles = (NodeIndex::from_usize(ids[0]), NodeIndex::from_usize(ids[2]));
    let outside = random_outside(&mut rng);
    assert!(
        SNodeSegTree::build(
            &Skeleton::from_adjacency(&adj),
            Some(bad_poles),
            Some(&outside),
            &|_| 1,
            None
        )
        .is_none()
    );
}

#[test]
fn unknown_nodes_are_unreachable() {
    let mut rng = Lcg(99);
    let (adj, ids) = random_ring(&mut rng, 4, true, true);
    let tree =
        SNodeSegTree::build(&Skeleton::from_adjacency(&adj), None, None, &|_| 1, None).unwrap();
    assert_eq!(
        tree.distance(directed(12345, 0), directed(ids[0], 0), Role::Lca),
        None
    );
    assert_eq!(tree.pos_of(NodeIndex::from_usize(12345)), None);
    assert!(tree.pole_projection(directed(12345, 1)).is_none());
}

/// A generated regular ring skeleton (see [`regular_ring`]).
struct RegularRing {
    adj: SkeletonAdjacency<I>,
    /// The vertex ids in cycle order.
    ids: Vec<usize>,
    /// The vertex lengths by position.
    lens: Vec<usize>,
    /// The lanes the closing edge joins at `v_{m-1}` and `v_0`.
    closing_lanes: (usize, usize),
}

/// A reversal-consistent **regular** ring over `m` vertices: every cycle edge passes in exactly one
/// orientation pair (the cw lane, `c[pos]` at each vertex) with its reverse-complement mirror, arc
/// weights are the source node's length (the engine's start-to-start shape), random rungs weigh the
/// vertex length, plus the zero self-loops the folds emit. The ring form covers it as one run, so the
/// build must take it.
fn regular_ring(
    rng: &mut Lcg,
    m: usize,
    closed: bool,
    shuffle: bool,
    cross_closing: bool,
) -> RegularRing {
    let mut ids: Vec<usize> = (0..m).map(|i| 7 * i + 3).collect();
    if shuffle {
        for i in (1..m).rev() {
            let j = rng.below(i + 1);
            ids.swap(i, j);
        }
    }
    let lens: Vec<usize> = (0..m).map(|_| 1 + rng.below(20)).collect();
    let c: Vec<usize> = (0..m).map(|_| rng.below(2)).collect();
    let mut adj: SkeletonAdjacency<I> = SkeletonAdjacency::default();
    let edges = if closed { m } else { m - 1 };
    // The closing edge (e == m-1) may connect the opposite lanes at either end (the bubble-allele
    // geometry: both cycle edges on the same side of a pole), the path edges never do. The lanes are
    // returned so an outside matrix over the same poles can agree with them.
    let closing_lanes = if cross_closing {
        (c[m - 1] ^ rng.below(2), c[0] ^ rng.below(2))
    } else {
        (c[m - 1], c[0])
    };
    for e in 0..edges {
        let (pa, pb) = (e, (e + 1) % m);
        let (a, b) = (ids[pa], ids[pb]);
        let (oa, ob) = if e == m - 1 {
            closing_lanes
        } else {
            (c[pa], c[pb])
        };
        push(&mut adj, directed(a, oa), directed(b, ob), lens[pa]);
        push(&mut adj, directed(b, 1 - ob), directed(a, 1 - oa), lens[pb]);
    }
    for pos in 0..m {
        let v = ids[pos];
        for o in 0..2 {
            if rng.chance(40) {
                push(&mut adj, directed(v, o), directed(v, 1 - o), lens[pos]);
            }
            if rng.chance(30) {
                push(&mut adj, directed(v, o), directed(v, o), 0);
            }
        }
    }
    RegularRing {
        adj,
        ids,
        lens,
        closing_lanes,
    }
}

/// A reversal-consistent `D_out` over `(s = v_0, t = v_{m-1})`: a pass `t^{ct} -> s^{cs}` in the given
/// lanes with its mirror `s^{¬cs} -> t^{¬ct}` (`W - len(t) + len(s)`), and random flips at either pole.
fn regular_outside(
    rng: &mut Lcg,
    lens: &[usize],
    lanes: (usize, usize),
    m: usize,
) -> MinPlusMatrix<I, 4> {
    let (ls, lt) = (lens[0], lens[m - 1]);
    let (ct, cs) = lanes;
    let mut mat = MinPlusMatrix::<I, 4>::infinity();
    if rng.chance(70) {
        let w = lt + rng.below(40);
        mat.set(
            slot(Pole::Target, Orientation::from_index(ct)),
            slot(Pole::Source, Orientation::from_index(cs)),
            MinPlus::finite_usize(w),
        );
        mat.set(
            slot(Pole::Source, Orientation::from_index(1 - cs)),
            slot(Pole::Target, Orientation::from_index(1 - ct)),
            MinPlus::finite_usize(w - lt + ls),
        );
    }
    for pole in Pole::all() {
        for o in 0..2 {
            if rng.chance(40) {
                mat.set(
                    slot(pole, Orientation::from_index(o)),
                    slot(pole, Orientation::from_index(1 - o)),
                    MinPlus::finite_usize(1 + rng.below(40)),
                );
            }
        }
    }
    mat
}

/// The ring form must be taken on every regular ring (one run) and reproduce the skeleton Dijkstra
/// in both roles (open paths and closed rings, root and non-root), including the pole projection.
#[test]
fn ring_form_on_regular_rings_matches_dijkstra() {
    let mut rng = Lcg(0x9E1F_0A11);
    for case in 0..400 {
        let m = 3 + rng.below(12);
        let closed = case % 2 == 0;
        let root = closed && case % 4 == 0;
        // A root ring's cycle start is arbitrary, so a crossed closing would land mid-path: those
        // keep the tree (future work: rotate the start to the break). Non-roots cross freely.
        let RegularRing {
            adj,
            ids,
            lens,
            closing_lanes: lanes,
            ..
        } = regular_ring(&mut rng, m, closed, case % 3 == 0, !root);
        let inner = Skeleton::from_adjacency(&adj);
        let what = format!("case {case} (m={m}, closed={closed})");
        let len_by_id: std::collections::HashMap<usize, u64> = ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| (id, lens[pos] as u64))
            .collect();
        let len_of = |n: NodeIndex<I>| len_by_id[&n.into_usize()];
        if root {
            let tree = SNodeSegTree::build(&inner, None, None, &len_of, None)
                .unwrap_or_else(|| panic!("{what}: a regular root ring must build"));
            assert!(
                tree.is_ring() && tree.exceptions() == 0,
                "{what}: a regular root ring is one run"
            );
            check_role(&tree, &inner, Role::Inner, &ids, &what);
            check_role(&tree, &inner, Role::Lca, &ids, &what);
            continue;
        }
        let poles = (
            NodeIndex::from_usize(ids[0]),
            NodeIndex::from_usize(ids[m - 1]),
        );
        let outside = regular_outside(&mut rng, &lens, lanes, m);
        let tree = SNodeSegTree::build(&inner, Some(poles), Some(&outside), &len_of, None)
            .unwrap_or_else(|| panic!("{what}: a regular ring must build"));
        assert!(
            tree.is_ring() && tree.exceptions() == 0,
            "{what}: a regular ring is one run"
        );
        check_role(&tree, &inner, Role::Inner, &ids, &what);
        let mut with = adj.clone();
        add_pole_state_edges(&mut with, poles.0, poles.1, &outside);
        check_role(
            &tree,
            &Skeleton::from_adjacency(&with),
            Role::Lca,
            &ids,
            &what,
        );
        let oracle = dijkstra_all_pairs(&inner);
        for &seed in &states(&ids) {
            let got = tree.pole_projection(seed).expect("seed is a vertex");
            for (pole, vertex) in [(Pole::Source, ids[0]), (Pole::Target, ids[m - 1])] {
                for o in 0..2 {
                    let target = directed(vertex, o);
                    let want = if seed == target {
                        Some(GfaPathLength::from_usize(0))
                    } else {
                        oracle(seed, target)
                    };
                    assert_eq!(
                        got[slot(pole, Orientation::from_index(o))].into_option(),
                        want,
                        "{what}: projection {seed:?} -> {target:?}"
                    );
                }
            }
        }
    }
}

/// Pole projections of `tree` against the inner Dijkstra, for every state.
fn check_projection(tree: &SNodeSegTree<I>, inner: &Skeleton<I>, ids: &[usize], what: &str) {
    let m = ids.len();
    let oracle = dijkstra_all_pairs(inner);
    for &seed in &states(ids) {
        let got = tree.pole_projection(seed).expect("seed is a vertex");
        for (pole, vertex) in [(Pole::Source, ids[0]), (Pole::Target, ids[m - 1])] {
            for o in 0..2 {
                let target = directed(vertex, o);
                let want = if seed == target {
                    Some(GfaPathLength::from_usize(0))
                } else {
                    oracle(seed, target)
                };
                assert_eq!(
                    got[slot(pole, Orientation::from_index(o))].into_option(),
                    want,
                    "{what}: projection {seed:?} -> {target:?}"
                );
            }
        }
    }
}

/// The tree form (the fallback when the ring form declines) must reproduce the skeleton Dijkstra
/// in both roles on every regular ring.
#[test]
fn tree_form_on_regular_rings_matches_dijkstra() {
    let mut rng = Lcg(0x9E1F_0A12);
    for case in 0..200 {
        let m = 3 + rng.below(12);
        let closed = case % 2 == 0;
        let root = closed && case % 4 == 0;
        let RegularRing {
            adj,
            ids,
            lens,
            closing_lanes: lanes,
            ..
        } = regular_ring(&mut rng, m, closed, case % 3 == 0, !root);
        let inner = Skeleton::from_adjacency(&adj);
        let what = format!("tree case {case} (m={m}, closed={closed})");
        let len_by_id: std::collections::HashMap<usize, u64> = ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| (id, lens[pos] as u64))
            .collect();
        let len_of = |n: NodeIndex<I>| len_by_id[&n.into_usize()];
        if root {
            let tree = SNodeSegTree::build_with_form(&inner, None, None, &len_of, None, Form::Tree)
                .unwrap_or_else(|| panic!("{what}: a regular root ring must build"));
            assert!(!tree.is_ring(), "{what}: the tree form");
            check_role(&tree, &inner, Role::Inner, &ids, &what);
            check_role(&tree, &inner, Role::Lca, &ids, &what);
            continue;
        }
        let poles = (
            NodeIndex::from_usize(ids[0]),
            NodeIndex::from_usize(ids[m - 1]),
        );
        let outside = regular_outside(&mut rng, &lens, lanes, m);
        let tree = SNodeSegTree::build_with_form(
            &inner,
            Some(poles),
            Some(&outside),
            &len_of,
            None,
            Form::Tree,
        )
        .unwrap_or_else(|| panic!("{what}: a regular ring must build"));
        assert!(!tree.is_ring(), "{what}: the tree form");
        check_role(&tree, &inner, Role::Inner, &ids, &what);
        let mut with = adj.clone();
        add_pole_state_edges(&mut with, poles.0, poles.1, &outside);
        check_role(
            &tree,
            &Skeleton::from_adjacency(&with),
            Role::Lca,
            &ids,
            &what,
        );
        check_projection(&tree, &inner, &ids, &what);
    }
}

/// What a path leaf of an [`irregular_ring`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeafKind {
    /// One lane, mirrored, consistent with the previous leaf's arrival.
    Regular,
    /// One lane, mirrored, departing from the orientation the previous leaf did NOT arrive in.
    LaneBreak,
    /// Both orientation pairs pass (each mirrored): a palindromic child.
    TwoLane,
    /// A lane plus a cross pass into the other orientation of the far vertex (an inversion child).
    Cross,
    /// No pass arc in any orientation (a child attached on one side of both poles).
    Dead,
    /// Arbitrary arcs, like [`random_ring`]'s edges: anything goes.
    Wild,
}

/// A generated irregular ring skeleton (see [`irregular_ring`]).
struct IrregularRing {
    adj: SkeletonAdjacency<I>,
    /// The vertex ids in cycle order.
    ids: Vec<usize>,
    /// The vertex lengths by position.
    lens: Vec<usize>,
    /// The cycle edges as node pairs.
    edges: Vec<(NodeIndex<I>, NodeIndex<I>)>,
    /// The kind of every leaf.
    kinds: Vec<LeafKind>,
}

/// A reversal-consistent ring over `m` vertices whose path leaves are a random mix of
/// [`LeafKind`]s; random rungs (weighing at least the vertex length, the engine's shape) and zero
/// self-loops at every vertex. The closing edge, when `closed`, is `Wild`.
fn irregular_ring(
    rng: &mut Lcg,
    m: usize,
    closed: bool,
    shuffle: bool,
    exception_percent: usize,
) -> IrregularRing {
    let mut ids: Vec<usize> = (0..m).map(|i| 7 * i + 3).collect();
    if shuffle {
        for i in (1..m).rev() {
            let j = rng.below(i + 1);
            ids.swap(i, j);
        }
    }
    let lens: Vec<usize> = (0..m).map(|_| 1 + rng.below(20)).collect();
    let mut adj: SkeletonAdjacency<I> = SkeletonAdjacency::default();
    let mut edges = Vec::new();
    let mut kinds = Vec::new();
    // `(a, oa) -> (b, ob)` at `w` plus its mirror `(b, ¬ob) -> (a, ¬oa)` at `w − len(a) + len(b)`.
    let mirrored = |adj: &mut SkeletonAdjacency<I>,
                    a: usize,
                    oa: usize,
                    b: usize,
                    ob: usize,
                    w: usize,
                    la: usize,
                    lb: usize| {
        push(adj, directed(a, oa), directed(b, ob), w);
        push(adj, directed(b, 1 - ob), directed(a, 1 - oa), w - la + lb);
    };
    let mut arrive: Option<usize> = None; // the orientation the previous leaf arrives in at this vertex
    let edges_n = if closed { m } else { m - 1 };
    for e in 0..edges_n {
        let (pa, pb) = (e, (e + 1) % m);
        let (a, b) = (ids[pa], ids[pb]);
        edges.push((NodeIndex::from_usize(a), NodeIndex::from_usize(b)));
        let kind = if e == m - 1 {
            LeafKind::Wild
        } else if !rng.chance(exception_percent) {
            LeafKind::Regular
        } else {
            [
                LeafKind::LaneBreak,
                LeafKind::TwoLane,
                LeafKind::Cross,
                LeafKind::Dead,
                LeafKind::Wild,
            ][rng.below(5)]
        };
        kinds.push(kind);
        let oa = match kind {
            LeafKind::LaneBreak => 1 - arrive.unwrap_or_else(|| rng.below(2)),
            _ => arrive.unwrap_or_else(|| rng.below(2)),
        };
        let ob = rng.below(2);
        let w = lens[pa] + rng.below(30);
        match kind {
            LeafKind::Regular | LeafKind::LaneBreak => {
                mirrored(&mut adj, a, oa, b, ob, w, lens[pa], lens[pb]);
                arrive = Some(ob);
            }
            LeafKind::TwoLane => {
                mirrored(&mut adj, a, oa, b, ob, w, lens[pa], lens[pb]);
                mirrored(
                    &mut adj,
                    a,
                    1 - oa,
                    b,
                    1 - ob,
                    lens[pa] + rng.below(30),
                    lens[pa],
                    lens[pb],
                );
                arrive = None;
            }
            LeafKind::Cross => {
                mirrored(&mut adj, a, oa, b, ob, w, lens[pa], lens[pb]);
                mirrored(
                    &mut adj,
                    a,
                    oa,
                    b,
                    1 - ob,
                    lens[pa] + rng.below(30),
                    lens[pa],
                    lens[pb],
                );
                arrive = None;
            }
            LeafKind::Dead => {
                arrive = None;
            }
            LeafKind::Wild => {
                for x in 0..2 {
                    for y in 0..2 {
                        if rng.chance(50) {
                            push(&mut adj, directed(a, x), directed(b, y), 1 + rng.below(40));
                        }
                        if rng.chance(50) {
                            push(&mut adj, directed(b, y), directed(a, x), 1 + rng.below(40));
                        }
                    }
                }
                arrive = None;
            }
        }
    }
    for pos in 0..m {
        let v = ids[pos];
        for o in 0..2 {
            if rng.chance(35) {
                push(
                    &mut adj,
                    directed(v, o),
                    directed(v, 1 - o),
                    lens[pos] + rng.below(30),
                );
            }
            if rng.chance(30) {
                push(&mut adj, directed(v, o), directed(v, o), 0);
            }
        }
    }
    IrregularRing {
        adj,
        ids,
        lens,
        edges,
        kinds,
    }
}

/// The ring form must be taken on every cycle — lane breaks, two-lane, cross, dead and wild
/// leaves included, with the cycle edges given so dead leaves keep their place — and reproduce
/// the skeleton Dijkstra in both roles, plus the pole projection.
#[test]
fn ring_form_on_irregular_rings_matches_dijkstra() {
    let mut rng = Lcg(0x5A1E_C0DE);
    let mut exceptions_seen = 0usize;
    for case in 0..600 {
        let m = 3 + rng.below(if case % 5 == 0 { 40 } else { 10 });
        let closed = case % 2 == 0;
        let root = closed && case % 4 == 0;
        let pct = [0, 15, 40, 100][case % 4];
        let IrregularRing {
            adj,
            ids,
            lens,
            edges,
            kinds,
        } = irregular_ring(&mut rng, m, closed, case % 3 == 0, pct);
        let inner = Skeleton::from_adjacency(&adj);
        let what = format!("case {case} (m={m}, closed={closed}, root={root}, kinds={kinds:?})");
        let len_by_id: std::collections::HashMap<usize, u64> = ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| (id, lens[pos] as u64))
            .collect();
        let len_of = |n: NodeIndex<I>| len_by_id[&n.into_usize()];
        let (poles, outside) = if root {
            (None, None)
        } else {
            (
                Some((
                    NodeIndex::from_usize(ids[0]),
                    NodeIndex::from_usize(ids[m - 1]),
                )),
                Some(random_outside(&mut rng)),
            )
        };
        let tree = SNodeSegTree::build(&inner, poles, outside.as_ref(), &len_of, Some(&edges))
            .unwrap_or_else(|| panic!("{what}: a cycle must build"));
        assert!(tree.is_ring(), "{what}: the ring form");
        assert_eq!(
            tree.m(),
            m,
            "{what}: every cycle vertex, dead leaves included"
        );
        exceptions_seen += tree.exceptions();
        check_role(&tree, &inner, Role::Inner, &ids, &what);
        match (poles, outside) {
            (Some(poles), Some(outside)) => {
                let mut with = adj.clone();
                add_pole_state_edges(&mut with, poles.0, poles.1, &outside);
                check_role(
                    &tree,
                    &Skeleton::from_adjacency(&with),
                    Role::Lca,
                    &ids,
                    &what,
                );
                check_projection(&tree, &inner, &ids, &what);
            }
            _ => check_role(&tree, &inner, Role::Lca, &ids, &what),
        }
    }
    assert!(
        exceptions_seen > 500,
        "the generator must produce exceptions ({exceptions_seen})"
    );
}

/// A dead leaf without the structural cycle: the arcs alone no longer form one cycle/path, so the
/// build may decline — but never answer wrongly — and with the cycle given it must build.
#[test]
fn dead_leaves_need_the_structural_cycle() {
    let mut rng = Lcg(0xDEAD_1EAF);
    let mut declined = 0usize;
    for case in 0..100 {
        let m = 4 + rng.below(8);
        let IrregularRing {
            adj,
            ids,
            lens,
            edges,
            kinds,
        } = irregular_ring(&mut rng, m, true, false, 100);
        if !kinds.contains(&LeafKind::Dead) {
            continue;
        }
        let inner = Skeleton::from_adjacency(&adj);
        let len_by_id: std::collections::HashMap<usize, u64> = ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| (id, lens[pos] as u64))
            .collect();
        let len_of = |n: NodeIndex<I>| len_by_id[&n.into_usize()];
        let what = format!("dead case {case}");
        match SNodeSegTree::build(&inner, None, None, &len_of, None) {
            None => declined += 1,
            Some(tree) => {
                // Built from the arcs alone: exact among the vertices it knows (an isolated
                // vertex — dead leaves on both sides, no rungs — is not among them).
                let oracle = dijkstra_all_pairs(&inner);
                for &from in &states(&ids) {
                    for &to in &states(&ids) {
                        if tree.pos_of(from.into_bidirected()).is_none()
                            || tree.pos_of(to.into_bidirected()).is_none()
                        {
                            continue;
                        }
                        let want = if from == to {
                            Some(GfaPathLength::from_usize(0))
                        } else {
                            oracle(from, to)
                        };
                        assert_eq!(
                            tree.distance(from, to, Role::Lca),
                            want,
                            "{what} (arcs only): {from:?} -> {to:?}"
                        );
                    }
                }
            }
        }
        let tree = SNodeSegTree::build(&inner, None, None, &len_of, Some(&edges))
            .unwrap_or_else(|| panic!("{what}: with the cycle given it must build"));
        assert_eq!(tree.m(), m);
        check_role(&tree, &inner, Role::Inner, &ids, &what);
        check_role(&tree, &inner, Role::Lca, &ids, &what);
    }
    assert!(
        declined > 0,
        "some dead ring must have declined without its cycle"
    );
}
