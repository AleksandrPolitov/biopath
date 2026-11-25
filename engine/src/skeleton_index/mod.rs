//! An exact directed distance oracle over a fixed orientation-doubled skeleton, built as a
//! **2-hop hub labeling** (directed Pruned Landmark Labeling).
//!
//! This is the acceleration structure that replaces the per-query skeleton Dijkstra on large SPQR
//! nodes (above all, the single giant R-node). It is built once from a
//! [`SkeletonAdjacency`] and then answers
//! `dist(from -> to)` for any two directed skeleton nodes by a label merge — no search.
//!
//! # Cost model
//!
//! The adjacency already encodes the start-to-start cost model (a directed edge `x -> y` weighs
//! `len(x)`), so the distances this returns are exactly the start-to-start distances the DP composes
//! — identical to what a Dijkstra over the same adjacency returns (asserted in the tests).
//!
//! # Correctness
//!
//! Hub labeling is exact for non-negative edge weights: for every reachable ordered pair `(s, t)`
//! there is a hub `h` on some shortest `s -> t` path that lies in both `L_out(s)` and `L_in(t)`, so
//! the merge recovers the true distance; unreachable pairs share no hub and return `None`.
//!
//! # Storage: the mirror form
//!
//! The build ([`SkeletonLabels::build_bidirected`]) exploits the reversal identity of a doubled
//! skeleton, `d(h -> t) = d(t̄ -> h̄) + len(h) - len(t)`: the in-label of `t` is the out-label
//! of `t̄` with every hub mirrored, so only the out-labels are stored. Dense ids are numbered so that
//! `rev(id) = n - 1 - id`, and an entry stores `2·d(x -> h) + len(h)` instead of `d(x -> h)`. Merging
//! `L_out(s)` against `L_out(t̄)` walked backwards (which lists the mirrored hubs in ascending order)
//! yields `2·(d(s -> t) + len(t))` at a shared hub, so `d(s -> t) = sum / 2 - len(t)`. This halves
//! the stored labels and the pruned searches.
//!
//! # Hub order
//!
//! The pruned searches run from the reversal pairs in order of their larger total degree
//! (`pll.rs`). The labels are exact for any order; the order decides how many entries they keep.
//! A sampled-betweenness order raced against it kept 0.18% fewer entries over 25 HPRC chromosome
//! graphs at about twice the label build time, so degree is the only order.

mod pll;
mod record;
#[cfg(test)]
mod tests;

pub use record::{LabelRec, LabelSizes, LabelTable, LabelsView};

use std::collections::HashMap;

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use crate::{
    image::{Padded, hash, image_record},
    inside_dp::SkeletonAdjacency,
    path::GfaPathLength,
};
use pll::{Csr, build_labels_ri};

const INF: u64 = u64::MAX;

/// Flattened (CSR) 2-hop labels of one side: node `v`'s list is `flat[offsets[v]..offsets[v + 1]]`,
/// sorted by hub. One 8-byte entry per label at the default word size (the merge is pure memory
/// traffic, so the entry width is what it costs).
struct LabelCsr<IndexType: GraphIndexInteger> {
    /// Row starts, length `node_count + 1`.
    offsets: Vec<u32>,
    /// `(hub, value)` pairs, concatenated per node, each row sorted by hub.
    flat: Vec<LabelEntry<IndexType>>,
}

/// One 2-hop label: a hub's dense id and the stored value `2·d + len(hub)` (the mirror form). Plain
/// data (8 B at word size 32). Both fields are at least 4 bytes and as wide as the index word, so no
/// word size leaves padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelEntry<IndexType: GraphIndexInteger> {
    pub hub: Padded<u32, IndexType>,
    pub distance: Padded<GfaPathLength<IndexType>, u32>,
}
image_record!(impl<I> LabelEntry<I> = [Padded<u32, I>, Padded<GfaPathLength<I>, u32>]);

/// Largest state count of a skeleton whose labels are stored narrow: `u16` hubs in one array and
/// the values in another (6 B per entry at word size 32, both arrays naturally aligned — a packed
/// 6-byte record measured 10-15% slower on merge-heavy queries).
pub const NARROW_LABEL_MAX_STATES: usize = 1 << 16;

#[cfg(test)]
thread_local! {
    /// Tests: store every label of this thread's builds wide, as only a skeleton of more than
    /// [`NARROW_LABEL_MAX_STATES`] states is otherwise, so small graphs exercise the wide form.
    pub(crate) static WIDE_LABELS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the labels of a skeleton of `states` states are stored narrow.
fn stores_narrow(states: usize) -> bool {
    #[cfg(test)]
    if WIDE_LABELS.get() {
        return false;
    }
    states <= NARROW_LABEL_MAX_STATES
}

/// A narrow label row: the hubs and the values of one node's label, as two parallel slices.
#[derive(Clone, Copy)]
pub struct NarrowRow<'a, IndexType: GraphIndexInteger> {
    pub hubs: &'a [u16],
    pub values: &'a [GfaPathLength<IndexType>],
}

/// Read access to one node's label row (wide or narrow), for the merge.
pub trait LabelRow: Copy {
    fn len(&self) -> usize;
    fn hub(&self, i: usize) -> u32;
    fn value(&self, i: usize) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<IndexType: GraphIndexInteger> LabelRow for &[LabelEntry<IndexType>] {
    #[inline]
    fn len(&self) -> usize {
        <[LabelEntry<IndexType>]>::len(self)
    }
    #[inline]
    fn hub(&self, i: usize) -> u32 {
        self[i].hub.get()
    }
    #[inline]
    fn value(&self, i: usize) -> usize {
        self[i].distance.get().into_usize()
    }
}

impl<IndexType: GraphIndexInteger> LabelRow for NarrowRow<'_, IndexType> {
    #[inline]
    fn len(&self) -> usize {
        self.hubs.len()
    }
    #[inline]
    fn hub(&self, i: usize) -> u32 {
        self.hubs[i] as u32
    }
    #[inline]
    fn value(&self, i: usize) -> usize {
        self.values[i].into_usize()
    }
}

/// The largest value a stored label may hold: `IndexType::MAX` is the `MinPlus` sentinel and the
/// mirror form needs one bit of headroom (`2d`), so a graph whose skeleton distances approach the
/// word size must be built with a wider `--word-size`.
fn label_value_max<IndexType: GraphIndexInteger>() -> u64 {
    crate::min_plus::wide::finite_max::<IndexType>()
}

impl<IndexType: GraphIndexInteger> LabelCsr<IndexType> {
    /// Compress finished per-node label vectors into CSR, storing `(hub, d)` as `2·d + len[hub]`
    /// (the mirror form, see the module docs). `INF` never reaches here — it is a search-time
    /// sentinel, never a stored label.
    fn from_vecs_mirror(vecs: &[Vec<(u32, u64)>], len: &[u64]) -> Self {
        let max = label_value_max::<IndexType>();
        let mut offsets = Vec::with_capacity(vecs.len() + 1);
        offsets.push(0u32);
        let mut total = 0u32;
        for v in vecs {
            total += v.len() as u32;
            offsets.push(total);
        }
        let mut flat = Vec::with_capacity(total as usize);
        for v in vecs {
            for &(hub, distance) in v {
                debug_assert!(
                    distance != INF,
                    "INF is a search sentinel, never a stored label"
                );
                let stored = 2 * distance + len[hub as usize];
                assert!(
                    stored <= max,
                    "a hub label value ({stored}) exceeds the index word size: rebuild with a larger --word-size"
                );
                flat.push(LabelEntry {
                    hub: Padded::new(hub),
                    distance: Padded::new(GfaPathLength::from_usize(stored as usize)),
                });
            }
        }
        Self { offsets, flat }
    }

    #[inline]
    fn row(&self, v: u32) -> &[LabelEntry<IndexType>] {
        let start = self.offsets[v as usize] as usize;
        let end = self.offsets[v as usize + 1] as usize;
        &self.flat[start..end]
    }
}

/// Process-unique id source for [`SkeletonLabels`], so a query-side memo can key on the oracle
/// identity. A raw pointer would be ABA-unsafe (a dropped oracle's address can be reused by a new
/// one, silently poisoning memo entries); a monotonic id cannot collide.
static LABELS_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A directed 2-hop hub-labeling distance oracle over a fixed skeleton.
pub struct SkeletonLabels<IndexType: GraphIndexInteger> {
    /// Process-unique identity (see [`LABELS_ID`]); keys the query-side `distance_id` memo.
    id: u64,
    /// Dense id -> directed node (parallel to the label vectors).
    node_of_id: Vec<DirectedNodeIndex<IndexType>>,
    /// Directed node -> dense id (touched only at query seed/read time).
    id_of_node: HashMap<DirectedNodeIndex<IndexType>, u32>,
    /// `l_out.row(v)` = sorted-by-hub out-label of `v`, `2·d(v -> hub) + len(hub)` per entry; the
    /// in-label of `t` is the out-label of `rev(t) = n - 1 - t`.
    l_out: LabelCsr<IndexType>,
    /// The node length of each dense id (the `len(t)` term of the merge).
    len_of_id: Vec<GfaPathLength<IndexType>>,
    /// Whether the record stores the labels narrow (at most [`NARROW_LABEL_MAX_STATES`] states).
    narrow: bool,
}

impl<IndexType: GraphIndexInteger> SkeletonLabels<IndexType> {
    /// Intern every directed node that appears as a source or target into a dense id, and the
    /// arcs over those ids.
    #[allow(clippy::type_complexity)]
    fn intern(
        adjacency: &SkeletonAdjacency<IndexType>,
    ) -> (
        HashMap<DirectedNodeIndex<IndexType>, u32>,
        Vec<DirectedNodeIndex<IndexType>>,
        Vec<(u32, u32, u64)>,
    ) {
        let mut id_of_node: HashMap<DirectedNodeIndex<IndexType>, u32> = HashMap::new();
        let mut node_of_id: Vec<DirectedNodeIndex<IndexType>> = Vec::new();
        let intern = |n: DirectedNodeIndex<IndexType>,
                      id_of_node: &mut HashMap<DirectedNodeIndex<IndexType>, u32>,
                      node_of_id: &mut Vec<DirectedNodeIndex<IndexType>>|
         -> u32 {
            *id_of_node.entry(n).or_insert_with(|| {
                let id = node_of_id.len() as u32;
                node_of_id.push(n);
                id
            })
        };
        let mut edges: Vec<(u32, u32, u64)> = Vec::new();
        for (&from, out) in adjacency {
            let u = intern(from, &mut id_of_node, &mut node_of_id);
            for &(to, w) in out {
                let v = intern(to, &mut id_of_node, &mut node_of_id);
                edges.push((u, v, w.into_usize() as u64));
            }
        }
        (id_of_node, node_of_id, edges)
    }

    /// Build the hub-labeling oracle of an orientation-doubled skeleton adjacency, exploiting the
    /// **reversal identity** to run half the pruned searches and to store half the labels (the mirror
    /// form, see the module docs). Instead of `2n` searches (a forward one writing `l_in` and a
    /// backward one writing `l_out` per root), run only the `n` backward searches that populate
    /// `l_out`, deriving each root's pivot `l_in[r]` on the fly from `l_out[rev(r)]`; the in-labels
    /// are never materialized.
    ///
    /// REQUIRES the adjacency to be **reversal-closed**: for every arc `x -> y` the reverse-complement
    /// arc `y_bar -> x_bar` is present (with the RI-consistent weight). Every orientation-doubled DP
    /// skeleton satisfies this. `len_of(x)` is the orientation-independent length of `x`'s
    /// underlying vertex. Exact (asserted against a skeleton Dijkstra in the tests).
    pub fn build_bidirected(
        adjacency: &SkeletonAdjacency<IndexType>,
        len_of: impl Fn(DirectedNodeIndex<IndexType>) -> u64,
    ) -> Self {
        let (mut id_of_node, mut node_of_id, edges) = Self::intern(adjacency);
        // Reversal-close the interned id set so every id has a partner (a node may appear only via
        // arcs whose complement's endpoints are otherwise absent).
        let mut scan = 0;
        while scan < node_of_id.len() {
            let inv = node_of_id[scan].invert();
            id_of_node.entry(inv).or_insert_with(|| {
                let id = node_of_id.len() as u32;
                node_of_id.push(inv);
                id
            });
            scan += 1;
        }
        let n = node_of_id.len();
        // Mirror numbering: the k-th reversal pair (by first appearance) gets ids `k` and `n-1-k`.
        let mut new_of_old = vec![u32::MAX; n];
        let mut pairs = 0u32;
        for old in 0..n {
            if new_of_old[old] != u32::MAX {
                continue;
            }
            let partner = id_of_node[&node_of_id[old].invert()] as usize;
            debug_assert_ne!(partner, old, "a directed node is never its own reversal");
            new_of_old[old] = pairs;
            new_of_old[partner] = (n as u32 - 1) - pairs;
            pairs += 1;
        }
        debug_assert_eq!(2 * pairs as usize, n);
        let mut mirrored = vec![DirectedNodeIndex::from_usize(0); n];
        for old in 0..n {
            mirrored[new_of_old[old] as usize] = node_of_id[old];
        }
        let node_of_id = mirrored;
        let id_of_node: HashMap<DirectedNodeIndex<IndexType>, u32> = node_of_id
            .iter()
            .enumerate()
            .map(|(id, &node)| (node, id as u32))
            .collect();
        let edges: Vec<(u32, u32, u64)> = edges
            .into_iter()
            .map(|(u, v, w)| (new_of_old[u as usize], new_of_old[v as usize], w))
            .collect();
        let rev_id: Vec<u32> = (0..n as u32).map(|id| (n as u32 - 1) - id).collect();
        let len: Vec<u64> = (0..n).map(|id| len_of(node_of_id[id])).collect();

        let forward = Csr::build(n, edges.iter().map(|&(u, v, w)| (u, v, w)));
        let reverse = Csr::build(n, edges.iter().map(|&(u, v, w)| (v, u, w)));

        let l_out = build_labels_ri(&forward, &reverse, &rev_id, &len);
        let l_out = LabelCsr::from_vecs_mirror(&l_out, &len);
        Self {
            id: LABELS_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            node_of_id,
            id_of_node,
            l_out,
            len_of_id: len
                .iter()
                .map(|&l| GfaPathLength::from_usize(l as usize))
                .collect(),
            narrow: stores_narrow(n),
        }
    }

    /// Element totals for persisting this oracle into a [`LabelTable`].
    fn sizes(&self) -> LabelSizes {
        LabelSizes {
            nodes: self.node_of_id.len() as u32,
            hash_slots: hash::node_capacity_for(self.node_of_id.len()) as u32,
            entries: self.l_out.flat.len() as u32,
            narrow: u32::from(self.narrow),
        }
    }
}

impl<IndexType: GraphIndexInteger> LabelsAccess<IndexType> for SkeletonLabels<IndexType> {
    #[inline]
    fn id(&self) -> u64 {
        self.id
    }
    #[inline]
    fn node_count(&self) -> usize {
        self.node_of_id.len()
    }
    #[inline]
    fn node_id(&self, node: DirectedNodeIndex<IndexType>) -> Option<u32> {
        self.id_of_node.get(&node).copied()
    }
    #[inline]
    fn distance_id(&self, from: u32, to: u32) -> Option<GfaPathLength<IndexType>> {
        let n_minus_1 = self.node_of_id.len() as u32 - 1;
        mirror_distance(
            self.l_out.row(from),
            self.l_out.row(n_minus_1 - to),
            n_minus_1,
            self.len_of_id[to as usize],
        )
    }
}

/// Read access to a 2-hop label oracle, implemented by the owned [`SkeletonLabels`] and the
/// image-backed [`LabelsView`].
pub trait LabelsAccess<IndexType: GraphIndexInteger> {
    /// Process-unique identity (keys the query-side `distance_id` memo). Stable for the oracle's
    /// lifetime; never reused after a drop.
    fn id(&self) -> u64;
    /// The number of directed nodes in the indexed skeleton.
    fn node_count(&self) -> usize;
    /// The dense id of a directed node, if present in this skeleton.
    fn node_id(&self, node: DirectedNodeIndex<IndexType>) -> Option<u32>;
    /// The exact start-to-start distance over dense ids (from [`Self::node_id`]), or `None` if
    /// unreachable. Hot path for the query engine, which resolves ids once and then reads many
    /// targets.
    fn distance_id(&self, from: u32, to: u32) -> Option<GfaPathLength<IndexType>>;

    /// The exact start-to-start distance `from -> to`, or `None` if `to` is unreachable from `from`
    /// (or either endpoint is absent from this skeleton).
    fn distance(
        &self,
        from: DirectedNodeIndex<IndexType>,
        to: DirectedNodeIndex<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let s = self.node_id(from)?;
        let t = self.node_id(to)?;
        self.distance_id(s, t)
    }
}

/// The mirror-form merge (see the module docs): `a = L_out(s)` ascending by hub against
/// `b = L_out(t̄)` walked backwards, so `n_minus_1 - hub` ascends too; a shared hub contributes
/// `2·d(s -> h) + len(h) + 2·d(t̄ -> h̄) + len(h) = 2·(d(s -> t) + len(t))`.
///
/// Branchless: the loop is branch-bound, not memory-bound (hub comparisons are unpredictable), so
/// the pointers advance by comparison flags and the equal case folds in through a mask, which lets
/// the body compile to cmovs.
#[inline]
pub fn mirror_distance<IndexType: GraphIndexInteger, R: LabelRow>(
    a: R,
    b: R,
    n_minus_1: u32,
    len_t: GfaPathLength<IndexType>,
) -> Option<GfaPathLength<IndexType>> {
    let (mut i, mut j) = (0usize, b.len());
    let mut best = usize::MAX;
    while i < a.len() && j > 0 {
        let (ha, hb) = (a.hub(i), n_minus_1 - b.hub(j - 1));
        let equal = (ha == hb) as usize;
        let mask = equal.wrapping_sub(1);
        let sum = a.value(i) + b.value(j - 1);
        best = best.min(sum | mask);
        i += (ha <= hb) as usize;
        j -= (hb <= ha) as usize;
    }
    (best != usize::MAX).then(|| GfaPathLength::from_usize((best >> 1) - len_t.into_usize()))
}
