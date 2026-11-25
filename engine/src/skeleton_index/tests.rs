//! All-pairs differential tests: [`SkeletonLabels`] must return exactly the same distances as a
//! Dijkstra over the same skeleton adjacency, on random reversal-closed doubled skeletons with the
//! start-to-start (per-edge `len(from)`) weight shape, including unreachable pairs.

use crate::skeleton_index::LabelsAccess;
use bidirected_adjacency_array::index::DirectedNodeIndex;

use super::SkeletonLabels;
use crate::{
    inside_dp::{SkeletonAdjacency, multi_source_dijkstra},
    path::GfaPathLength,
};

type Ix = u16;

/// A small deterministic LCG for reproducible random graphs.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn upto(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

fn node(i: usize) -> DirectedNodeIndex<Ix> {
    DirectedNodeIndex::from_usize(i)
}

/// Every ordered pair of `0..n` agrees with a Dijkstra over `adjacency`.
fn assert_all_pairs_match(adjacency: &SkeletonAdjacency<Ix>, lengths: &[u64], n: usize) {
    let labels = SkeletonLabels::build_bidirected(adjacency, |d: DirectedNodeIndex<Ix>| {
        lengths[d.into_bidirected().into_usize()]
    });
    for s in 0..n {
        let dijkstra = multi_source_dijkstra(adjacency, &[(node(s), GfaPathLength::from_usize(0))]);
        for t in 0..n {
            let want = dijkstra.get(&node(t)).copied();
            let got = labels.distance(node(s), node(t));
            assert_eq!(
                got, want,
                "mismatch {s} -> {t}: labels={got:?} dijkstra={want:?}",
            );
        }
    }
}

/// A **reversal-closed** doubled skeleton: `m` underlying vertices, each with a random length and two
/// orientations (directed id `2v` = `v^+`, `2v+1` = `v^-`, so `invert = id ^ 1`, `into_bidirected =
/// id >> 1`). Every random bidirected edge contributes both the transition `u^a -> v^b` (weight
/// `len(u)`) and its reverse-complement `v^{!b} -> u^{!a}` (weight `len(v)`), so the RI holds — exactly
/// what [`SkeletonLabels::build_bidirected`] requires. Returns `(adjacency, lengths)`.
fn random_doubled_skeleton(
    rng: &mut Lcg,
    m: usize,
    extra_edges: usize,
) -> (SkeletonAdjacency<Ix>, Vec<u64>) {
    let lengths: Vec<u64> = (0..m).map(|_| 1 + rng.upto(50) as u64).collect();
    let mut adjacency: SkeletonAdjacency<Ix> = Default::default();
    for _ in 0..(m + extra_edges) {
        let u = rng.upto(m);
        let v = rng.upto(m);
        let a = rng.upto(2);
        let b = rng.upto(2);
        if u == v {
            continue;
        }
        let from = node(2 * u + a);
        let to = node(2 * v + b);
        adjacency
            .entry(from)
            .or_default()
            .push((to, GfaPathLength::from_usize(lengths[u] as usize)));
        // Reverse-complement: v^{!b} -> u^{!a}, weight len(v).
        let rfrom = node(2 * v + (b ^ 1));
        let rto = node(2 * u + (a ^ 1));
        adjacency
            .entry(rfrom)
            .or_default()
            .push((rto, GfaPathLength::from_usize(lengths[v] as usize)));
    }
    for i in 0..2 * m {
        adjacency.entry(node(i)).or_default();
    }
    (adjacency, lengths)
}

#[test]
fn matches_dijkstra_on_random_skeletons() {
    let mut rng = Lcg(0xB1D1_2E57);
    for _ in 0..80 {
        let m = 1 + rng.upto(20);
        let n = 2 * m;
        let extra = rng.upto(3 * n);
        let (adjacency, lengths) = random_doubled_skeleton(&mut rng, m, extra);
        assert_all_pairs_match(&adjacency, &lengths, n);
    }
}

#[test]
fn matches_dijkstra_on_dense_skeleton() {
    // A denser skeleton stresses label pruning and multi-hop shortest paths.
    let mut rng = Lcg(0x1234_5678);
    let m = 30;
    let (adjacency, lengths) = random_doubled_skeleton(&mut rng, m, 16 * m);
    assert_all_pairs_match(&adjacency, &lengths, 2 * m);
}

/// The RI-halved build is exact on larger random skeletons too (up to 40 vertices, 80 states).
#[test]
fn ri_build_exact_on_larger_random_skeletons() {
    let mut rng = Lcg(0x0BDE_2026);
    for _ in 0..60 {
        let m = 1 + rng.upto(40);
        let n = 2 * m;
        let extra = rng.upto(3 * n);
        let (adjacency, lengths) = random_doubled_skeleton(&mut rng, m, extra);
        assert_all_pairs_match(&adjacency, &lengths, n);
    }
}

#[test]
fn handles_self_reachability_and_cycles() {
    // A doubled cycle 0+ -> 1+ -> 2+ -> 3+ -> 0+ (and its reverse complement) with distinct
    // lengths: every state reaches every state of its strand, and a state reaches itself around
    // the cycle.
    let lengths = [3u64, 7, 2, 5];
    let mut adjacency: SkeletonAdjacency<Ix> = Default::default();
    for i in 0..4 {
        let j = (i + 1) % 4;
        adjacency
            .entry(node(2 * i))
            .or_default()
            .push((node(2 * j), GfaPathLength::from_usize(lengths[i] as usize)));
        adjacency.entry(node(2 * j + 1)).or_default().push((
            node(2 * i + 1),
            GfaPathLength::from_usize(lengths[j] as usize),
        ));
    }
    assert_all_pairs_match(&adjacency, &lengths, 8);
}
