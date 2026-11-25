//! A cache-friendly skeleton graph and a reusable Dijkstra workspace.
//!
//! The query hot path runs many small Dijkstras over the same fixed skeletons. Representing each
//! skeleton once as CSR arrays over dense integer ids (instead of a `HashMap` adjacency) and reusing
//! a single distance/heap workspace across all queries (epoch-reset, so clearing is O(1)) removes
//! the per-query allocation and hashing that otherwise dominate on small graphs.

use std::{cmp::Reverse, collections::BinaryHeap};

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use crate::{
    fast_hash::FastMap,
    image::{
        Image,
        hash::{self, NodeSlot},
        layout::Layout,
    },
    inside_dp::SkeletonAdjacency,
    min_plus::MinPlusMatrix,
    path::GfaPathLength,
};

/// Skeletons with at most this many nodes drop the `id_of_node` intern map and linear-scan
/// `node_of_id` for `node_id` lookups instead. The map (a hashbrown bucket array + control bytes,
/// power-of-2 rounded) is the heaviest of a skeleton's allocations; on graphs made of millions of
/// tiny blocks it dominates RSS, while the scan over `<= THRESHOLD` nodes is cheaper than hashing.
/// Large skeletons (e.g. a giant R-node block's) keep the map so their many `node_id` calls stay O(1).
const SKELETON_MAP_THRESHOLD: usize = 32;

/// Read access to a CSR skeleton, implemented by the owned [`Skeleton`] (build time) and the
/// image-backed [`SkeletonView`] (query time), so the search kernels are written once.
pub trait SkeletonAccess<IndexType: GraphIndexInteger> {
    /// Dense id -> directed node.
    fn node_of_id(&self) -> &[DirectedNodeIndex<IndexType>];
    /// CSR row starts, length `node_count + 1`.
    fn offsets(&self) -> &[u32];
    /// Neighbour dense ids, concatenated per source.
    fn edge_target(&self) -> &[u32];
    /// Neighbour edge weights, aligned with `edge_target`.
    fn edge_weight(&self) -> &[GfaPathLength<IndexType>];
    /// The dense id of a directed node, if present.
    fn node_id(&self, node: DirectedNodeIndex<IndexType>) -> Option<u32>;

    /// The number of dense nodes.
    #[inline]
    fn node_count(&self) -> usize {
        self.node_of_id().len()
    }

    /// The number of directed edges stored (a view's edge arrays may extend past its own edges:
    /// its offsets are absolute in its block's arrays).
    #[inline]
    fn edge_count(&self) -> usize {
        let offsets = self.offsets();
        (offsets[offsets.len() - 1] - offsets[0]) as usize
    }

    /// Extra arcs among up to four pole states, relaxed by the search when one of them is popped (the
    /// LCA role's `D_out`, see [`LcaView`]); `None` for a plain CSR skeleton.
    #[inline]
    fn pole_overlay(&self) -> Option<&PoleOverlay<IndexType>> {
        None
    }
}

/// The `D_out` overlay of an [`LcaView`]: the outside matrix between an SPQR node's four reference-pole
/// states (indexed by `crate::pole_lift::slot`) and each state's dense id in the search.
#[derive(Clone, Copy, Debug)]
pub struct PoleOverlay<IndexType: GraphIndexInteger> {
    states: [DirectedNodeIndex<IndexType>; 4],
    /// Dense id of each state, `u32::MAX` if it has no finite entry (it then carries no overlay arc).
    ids: [u32; 4],
    /// States with a finite entry but no inner arc, numbered after the inner skeleton's nodes.
    virtual_count: u32,
    weights: MinPlusMatrix<IndexType, 4>,
}

/// An SPQR node's LCA-role skeleton as stored: its inner skeleton plus its outside matrix `D_out` as a
/// [`PoleOverlay`]. The search relaxes the overlay's arcs when a pole state is popped, so it explores
/// the inner arcs plus one arc per finite off-diagonal `D_out` entry without a second CSR. A pole
/// state with a finite `D_out` entry (as row or column, diagonal included) but no inner arc gets a
/// virtual id past the inner ones, in slot order.
#[derive(Clone, Copy)]
pub struct LcaView<'a, IndexType: GraphIndexInteger> {
    inner: SkeletonView<'a, IndexType>,
    overlay: Option<PoleOverlay<IndexType>>,
}

impl<'a, IndexType: GraphIndexInteger> LcaView<'a, IndexType> {
    /// `inner` plus `D_out` over the given pole states (`None` at a block root, or for a node whose LCA
    /// search never uses the stored skeleton). `mask` has bit `4 * row + col` set for every finite
    /// `D_out` entry.
    pub fn new(
        inner: SkeletonView<'a, IndexType>,
        outside: Option<(
            [DirectedNodeIndex<IndexType>; 4],
            MinPlusMatrix<IndexType, 4>,
            u16,
        )>,
    ) -> Self {
        let overlay = outside.and_then(|(states, weights, mask)| {
            let inner_nodes = inner.node_of_id();
            let inner_count = inner_nodes.len() as u32;
            let mut ids = [u32::MAX; 4];
            let mut virtual_count = 0u32;
            // The inner skeleton numbers its pole states first, in slot order (see
            // `Skeleton::from_adjacency_first`), so a present state is at the head of `node_of_id`.
            let mut head = 0usize;
            for k in 0..4 {
                let present = inner_nodes.get(head) == Some(&states[k]);
                if present {
                    head += 1;
                }
                let finite = mask & (0x000f << (4 * k)) != 0 || mask & (0x1111 << k) != 0;
                if finite {
                    ids[k] = if present {
                        head as u32 - 1
                    } else {
                        // Not at the head: absent from the inner skeleton (or one not built pole-first).
                        inner.node_id(states[k]).unwrap_or_else(|| {
                            virtual_count += 1;
                            inner_count + virtual_count - 1
                        })
                    };
                }
            }
            ids.iter().any(|&id| id != u32::MAX).then_some(PoleOverlay {
                states,
                ids,
                virtual_count,
                weights,
            })
        });
        Self { inner, overlay }
    }
}

impl<'a, IndexType: GraphIndexInteger> SkeletonAccess<IndexType> for LcaView<'a, IndexType> {
    #[inline]
    fn node_of_id(&self) -> &[DirectedNodeIndex<IndexType>] {
        self.inner.node_of_id()
    }
    #[inline]
    fn offsets(&self) -> &[u32] {
        self.inner.offsets()
    }
    #[inline]
    fn edge_target(&self) -> &[u32] {
        self.inner.edge_target()
    }
    #[inline]
    fn edge_weight(&self) -> &[GfaPathLength<IndexType>] {
        self.inner.edge_weight()
    }
    #[inline]
    fn node_id(&self, node: DirectedNodeIndex<IndexType>) -> Option<u32> {
        self.inner.node_id(node).or_else(|| {
            let overlay = self.overlay.as_ref()?;
            let k = overlay.states.iter().position(|&state| state == node)?;
            (overlay.ids[k] != u32::MAX).then_some(overlay.ids[k])
        })
    }
    #[inline]
    fn node_count(&self) -> usize {
        self.inner.node_count()
            + self
                .overlay
                .as_ref()
                .map_or(0, |o| o.virtual_count as usize)
    }
    #[inline]
    fn pole_overlay(&self) -> Option<&PoleOverlay<IndexType>> {
        self.overlay.as_ref()
    }
}

/// Linear-scan `node_id` for small skeletons: `node_of_id[id] == node` iff `id` is `node`'s dense
/// id, so this returns exactly what a map would.
#[inline]
fn scan_node_id<IndexType: GraphIndexInteger>(
    node_of_id: &[DirectedNodeIndex<IndexType>],
    node: DirectedNodeIndex<IndexType>,
) -> Option<u32> {
    node_of_id.iter().position(|&n| n == node).map(|i| i as u32)
}

/// The hash key of a directed node in the image `NodeSlot` tables: its raw index.
#[inline]
pub(crate) fn node_key<IndexType: GraphIndexInteger>(
    node: DirectedNodeIndex<IndexType>,
) -> IndexType {
    node.into_raw()
}

/// An image-backed skeleton: borrowed CSR arrays plus an optional flat hash for `node_id` (present
/// for skeletons above `SKELETON_MAP_THRESHOLD`, exactly where the owned type keeps its map).
#[derive(Clone, Copy)]
pub struct SkeletonView<'a, IndexType: GraphIndexInteger> {
    node_of_id: &'a [DirectedNodeIndex<IndexType>],
    offsets: &'a [u32],
    edge_target: &'a [u32],
    edge_weight: &'a [GfaPathLength<IndexType>],
    hash: &'a [NodeSlot<IndexType>],
}

impl<'a, IndexType: GraphIndexInteger> SkeletonAccess<IndexType> for SkeletonView<'a, IndexType> {
    #[inline]
    fn node_of_id(&self) -> &[DirectedNodeIndex<IndexType>] {
        self.node_of_id
    }
    #[inline]
    fn offsets(&self) -> &[u32] {
        self.offsets
    }
    #[inline]
    fn edge_target(&self) -> &[u32] {
        self.edge_target
    }
    #[inline]
    fn edge_weight(&self) -> &[GfaPathLength<IndexType>] {
        self.edge_weight
    }
    #[inline]
    fn node_id(&self, node: DirectedNodeIndex<IndexType>) -> Option<u32> {
        if self.hash.is_empty() {
            scan_node_id(self.node_of_id, node)
        } else {
            hash::lookup_node(self.hash, node_key(node))
        }
    }
}

/// Per-skeleton entry of a [`SkelTable`]: where the skeleton's nodes start in the table's
/// concatenated arrays (its edges start at `offsets[node_start]`: the table is ONE block-wide CSR).
/// One sentinel entry follows the last skeleton, so no lengths are stored.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SkelRec {
    pub node_start: u32,
}
crate::image::image_record!(SkelRec = [u32]);

/// A skeleton that carries a flat hash table (above `SKELETON_MAP_THRESHOLD` nodes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SkelHashRec {
    pub local: u32,
    pub slot_start: u32,
    pub slot_count: u32,
}
crate::image::image_record!(SkelHashRec = [u32, u32, u32]);

/// Element totals of a [`SkelTable`], stored in the owning record's header.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SkelTableSizes {
    pub nodes: u32,
    pub edges: u32,
    pub hash_slots: u32,
    pub hashed: u32,
}
crate::image::image_record!(SkelTableSizes = [u32, u32, u32, u32]);

impl SkelTableSizes {
    /// Totals over a block's skeletons.
    pub fn of<IndexType: GraphIndexInteger>(skeletons: &[Skeleton<IndexType>]) -> Self {
        let mut sizes = Self::default();
        for skeleton in skeletons {
            sizes.nodes += skeleton.node_count() as u32;
            sizes.edges += skeleton.edge_count() as u32;
            if skeleton.id_of_node.is_some() {
                sizes.hashed += 1;
                sizes.hash_slots += hash::node_capacity_for(skeleton.node_count()) as u32;
            }
        }
        sizes
    }
}

/// The relative offsets of a [`SkelTable`]'s arrays inside a record (see [`Layout`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct SkelTableLayout {
    pub recs: usize,
    pub node_of_id: usize,
    pub offsets: usize,
    pub edge_target: usize,
    pub edge_weight: usize,
    pub hash_recs: usize,
    pub hash_slots: usize,
}

impl SkelTableLayout {
    /// Lay the table out for `count` skeletons with the given totals.
    pub fn place<IndexType: GraphIndexInteger>(
        layout: &mut Layout,
        count: usize,
        sizes: &SkelTableSizes,
    ) -> Self {
        Self {
            recs: layout.field::<SkelRec>(count + 1),
            node_of_id: layout.field::<DirectedNodeIndex<IndexType>>(sizes.nodes as usize),
            offsets: layout.field::<u32>(sizes.nodes as usize + 1),
            edge_target: layout.field::<u32>(sizes.edges as usize),
            edge_weight: layout.field::<GfaPathLength<IndexType>>(sizes.edges as usize),
            hash_recs: layout.field::<SkelHashRec>(sizes.hashed as usize),
            hash_slots: layout.field::<NodeSlot<IndexType>>(sizes.hash_slots as usize),
        }
    }
}

/// A block's skeletons (one per SPQR node) packed into concatenated arrays inside the image.
#[derive(Clone, Copy)]
pub struct SkelTable<'a, IndexType: GraphIndexInteger> {
    recs: &'a [SkelRec],
    node_of_id: &'a [DirectedNodeIndex<IndexType>],
    offsets: &'a [u32],
    edge_target: &'a [u32],
    edge_weight: &'a [GfaPathLength<IndexType>],
    hash_recs: &'a [SkelHashRec],
    hash_slots: &'a [NodeSlot<IndexType>],
}

impl<'a, IndexType: GraphIndexInteger> SkelTable<'a, IndexType> {
    /// Write `skeletons` into a record at absolute byte offset `base` with the given layout.
    ///
    /// # Safety
    /// `base + layout` must lie within a record this thread reserved from `image` (see
    /// `Image::get_at_mut`).
    pub unsafe fn write(
        image: &Image,
        base: usize,
        layout: &SkelTableLayout,
        sizes: &SkelTableSizes,
        skeletons: Vec<Skeleton<IndexType>>,
    ) {
        let count = skeletons.len();
        let recs = unsafe { image.get_at_mut::<SkelRec>(base + layout.recs, count + 1) };
        let node_of_id = unsafe {
            image.get_at_mut::<DirectedNodeIndex<IndexType>>(
                base + layout.node_of_id,
                sizes.nodes as usize,
            )
        };
        let offsets =
            unsafe { image.get_at_mut::<u32>(base + layout.offsets, sizes.nodes as usize + 1) };
        let edge_target =
            unsafe { image.get_at_mut::<u32>(base + layout.edge_target, sizes.edges as usize) };
        let edge_weight = unsafe {
            image.get_at_mut::<GfaPathLength<IndexType>>(
                base + layout.edge_weight,
                sizes.edges as usize,
            )
        };
        let hash_recs = unsafe {
            image.get_at_mut::<SkelHashRec>(base + layout.hash_recs, sizes.hashed as usize)
        };
        let hash_slots = unsafe {
            image.get_at_mut::<NodeSlot<IndexType>>(
                base + layout.hash_slots,
                sizes.hash_slots as usize,
            )
        };
        let (mut n, mut e, mut h, mut k) = (0usize, 0usize, 0usize, 0usize);
        // Consume skeleton by skeleton so the owned arrays are freed as they are copied: the
        // transient over the owned total is one skeleton, not the whole table.
        for (local, skeleton) in skeletons.into_iter().enumerate() {
            recs[local] = SkelRec {
                node_start: n as u32,
            };
            let nc = skeleton.node_count();
            let ec = skeleton.edge_count();
            node_of_id[n..n + nc].copy_from_slice(&skeleton.node_of_id);
            // Absolute in the block's edge arrays; the row end doubles as the next skeleton's start.
            for (slot, &row) in offsets[n..n + nc + 1].iter_mut().zip(&skeleton.offsets) {
                *slot = e as u32 + row;
            }
            edge_target[e..e + ec].copy_from_slice(&skeleton.edge_target);
            edge_weight[e..e + ec].copy_from_slice(&skeleton.edge_weight);
            if skeleton.id_of_node.is_some() {
                let cap = hash::node_capacity_for(nc);
                let slots = &mut hash_slots[h..h + cap];
                slots.fill(NodeSlot::empty());
                hash::build_nodes_into(
                    slots,
                    skeleton
                        .node_of_id
                        .iter()
                        .enumerate()
                        .map(|(id, &node)| (node_key(node), id as u32)),
                );
                hash_recs[k] = SkelHashRec {
                    local: local as u32,
                    slot_start: h as u32,
                    slot_count: cap as u32,
                };
                h += cap;
                k += 1;
            }
            n += nc;
            e += ec;
        }
        recs[count] = SkelRec {
            node_start: n as u32,
        };
        offsets[n] = e as u32;
        debug_assert_eq!(
            (n, e, h, k),
            (
                sizes.nodes as usize,
                sizes.edges as usize,
                sizes.hash_slots as usize,
                sizes.hashed as usize
            )
        );
    }

    /// Read the table of `count` skeletons at `base` with the given layout.
    pub fn read(
        image: &'a Image,
        base: usize,
        layout: &SkelTableLayout,
        sizes: &SkelTableSizes,
        count: usize,
    ) -> Self {
        Self {
            recs: image.get_at(base + layout.recs, count + 1),
            node_of_id: image.get_at(base + layout.node_of_id, sizes.nodes as usize),
            offsets: image.get_at(base + layout.offsets, sizes.nodes as usize + 1),
            edge_target: image.get_at(base + layout.edge_target, sizes.edges as usize),
            edge_weight: image.get_at(base + layout.edge_weight, sizes.edges as usize),
            hash_recs: image.get_at(base + layout.hash_recs, sizes.hashed as usize),
            hash_slots: image.get_at(base + layout.hash_slots, sizes.hash_slots as usize),
        }
    }

    /// The skeleton of block-local SPQR node `local`.
    #[inline]
    pub fn get(&self, local: usize) -> SkeletonView<'a, IndexType> {
        let (ns, ne) = (
            self.recs[local].node_start as usize,
            self.recs[local + 1].node_start as usize,
        );
        let hash = match self
            .hash_recs
            .binary_search_by_key(&(local as u32), |h| h.local)
        {
            Ok(i) => {
                let h = self.hash_recs[i];
                &self.hash_slots[h.slot_start as usize..(h.slot_start + h.slot_count) as usize]
            }
            Err(_) => &[],
        };
        // The offsets are absolute, so the view's edge arrays are the block's (from 0).
        SkeletonView {
            node_of_id: &self.node_of_id[ns..ne],
            offsets: &self.offsets[ns..ne + 1],
            edge_target: self.edge_target,
            edge_weight: self.edge_weight,
            hash,
        }
    }
}

/// A directed skeleton in CSR form over dense node ids `0..node_count` (the owned, build-time form;
/// see [`SkeletonView`] for the image-backed one).
pub struct Skeleton<IndexType: GraphIndexInteger> {
    /// Dense id -> directed node.
    node_of_id: Vec<DirectedNodeIndex<IndexType>>,
    /// Directed node -> dense id (only touched at seed/read time, not in the hot loop). `None` for
    /// skeletons with `<= SKELETON_MAP_THRESHOLD` nodes, which linear-scan `node_of_id` instead —
    /// byte-identical, since dense ids are assigned in `node_of_id` order.
    id_of_node: Option<FastMap<DirectedNodeIndex<IndexType>, u32>>,
    /// CSR row starts, length `node_count + 1`.
    offsets: Vec<u32>,
    /// Neighbour dense ids, concatenated per source.
    edge_target: Vec<u32>,
    /// Neighbour edge weights, aligned with `edge_target`.
    edge_weight: Vec<GfaPathLength<IndexType>>,
}

impl<IndexType: GraphIndexInteger> Skeleton<IndexType> {
    /// An empty skeleton (no nodes, no edges). Used to release a skeleton that is retained by index
    /// but never read at query time (e.g. the LCA-role skeleton of a labeled SPQR node).
    pub fn empty() -> Self {
        Self::from_adjacency(&SkeletonAdjacency::default())
    }

    /// Drop the arcs but keep the dense id numbering, so `node_id` / `node_of_id` / `node_count`
    /// still answer: for an SPQR node whose LCA search and pole projection are both served by
    /// precomputed structures (labels + projection table) no search ever runs over the skeleton,
    /// and its arcs would be dead weight in the record.
    pub fn drop_arcs(&mut self) {
        self.offsets = vec![0; self.node_of_id.len() + 1];
        self.edge_target = Vec::new();
        self.edge_weight = Vec::new();
    }

    /// Build the CSR skeleton from a `HashMap` adjacency.
    pub fn from_adjacency(adjacency: &SkeletonAdjacency<IndexType>) -> Self {
        Self::from_adjacency_first(adjacency, &[])
    }

    /// Like [`Self::from_adjacency`], but the nodes of `first` that occur in the skeleton get the
    /// lowest dense ids, in `first` order. An SPQR node's inner skeleton puts its four reference-pole
    /// states first, so [`LcaView`] finds their ids by reading the head of `node_of_id` instead of
    /// hashing (the node set and every distance are unchanged; only the numbering moves).
    pub fn from_adjacency_first(
        adjacency: &SkeletonAdjacency<IndexType>,
        first: &[DirectedNodeIndex<IndexType>],
    ) -> Self {
        let (id_of_node, node_of_id) = intern_nodes(adjacency, first);
        let node_count = node_of_id.len();
        let (offsets, edge_target, edge_weight) =
            DedupArcs::new(node_count).csr(adjacency, &id_of_node);
        // Retain the intern map only for large skeletons; tiny ones (the overwhelming majority on a
        // block-cut graph) drop it and linear-scan `node_of_id` in `node_id`, saving one heavy
        // allocation each.
        let id_of_node = (node_count > SKELETON_MAP_THRESHOLD).then_some(id_of_node);
        Self {
            node_of_id,
            id_of_node,
            offsets,
            edge_target,
            edge_weight,
        }
    }
}

/// Dense ids for every node of `adjacency` that appears as a source or a target, in order of first
/// appearance, except that the nodes of `first` present come first, in `first` order.
#[allow(clippy::type_complexity)]
fn intern_nodes<IndexType: GraphIndexInteger>(
    adjacency: &SkeletonAdjacency<IndexType>,
    first: &[DirectedNodeIndex<IndexType>],
) -> (
    FastMap<DirectedNodeIndex<IndexType>, u32>,
    Vec<DirectedNodeIndex<IndexType>>,
) {
    let mut id_of_node: FastMap<DirectedNodeIndex<IndexType>, u32> = FastMap::default();
    let mut node_of_id: Vec<DirectedNodeIndex<IndexType>> = Vec::new();
    let intern = |node: DirectedNodeIndex<IndexType>,
                  id_of_node: &mut FastMap<DirectedNodeIndex<IndexType>, u32>,
                  node_of_id: &mut Vec<DirectedNodeIndex<IndexType>>|
     -> u32 {
        *id_of_node.entry(node).or_insert_with(|| {
            let id = node_of_id.len() as u32;
            node_of_id.push(node);
            id
        })
    };
    for (&from, edges) in adjacency {
        intern(from, &mut id_of_node, &mut node_of_id);
        for &(to, _) in edges {
            intern(to, &mut id_of_node, &mut node_of_id);
        }
    }
    if first.iter().any(|node| id_of_node.contains_key(node)) {
        let head: Vec<DirectedNodeIndex<IndexType>> = first
            .iter()
            .copied()
            .filter(|node| id_of_node.contains_key(node))
            .collect();
        node_of_id = head
            .iter()
            .copied()
            .chain(node_of_id.into_iter().filter(|node| !first.contains(node)))
            .collect();
        for (id, &node) in node_of_id.iter().enumerate() {
            id_of_node.insert(node, id as u32);
        }
    }
    (id_of_node, node_of_id)
}

/// The CSR arcs of a skeleton adjacency, keeping only the arcs a search can use: no self-loops
/// (weights are non-negative, so a loop never improves a distance; the `4x4` pole-state folds emit
/// one per diagonal entry) and one arc per `(source, target)` pair carrying the minimum weight
/// (parallel arcs come from several folds relating the same two states). Every node keeps its
/// dense id, so lookups are unchanged and every shortest distance over the skeleton is the same.
struct DedupArcs {
    /// The arc slot of each target in the current source's row (`u32::MAX` = none yet).
    slot_of: Vec<u32>,
    /// The targets `slot_of` holds for the current row, reset after it.
    touched: Vec<u32>,
}

impl DedupArcs {
    fn new(node_count: usize) -> Self {
        Self {
            slot_of: vec![u32::MAX; node_count],
            touched: Vec::new(),
        }
    }

    /// `(offsets, edge_target, edge_weight)`: the distinct out-neighbours per source are counted
    /// first (prefix-summed into the offsets), then each row is filled.
    #[allow(clippy::type_complexity)]
    fn csr<IndexType: GraphIndexInteger>(
        &mut self,
        adjacency: &SkeletonAdjacency<IndexType>,
        id_of_node: &FastMap<DirectedNodeIndex<IndexType>, u32>,
    ) -> (Vec<u32>, Vec<u32>, Vec<GfaPathLength<IndexType>>) {
        let node_count = self.slot_of.len();
        let mut offsets = vec![0u32; node_count + 1];
        for (&from, edges) in adjacency {
            let from_id = id_of_node[&from];
            offsets[from_id as usize + 1] = self.row_degree(from_id, edges, id_of_node);
        }
        for i in 0..node_count {
            offsets[i + 1] += offsets[i];
        }
        let edge_len = *offsets.last().unwrap() as usize;
        let mut edge_target = vec![0u32; edge_len];
        let mut edge_weight = vec![GfaPathLength::from_usize(0); edge_len];
        for (&from, edges) in adjacency {
            let from_id = id_of_node[&from];
            let mut cursor = offsets[from_id as usize];
            for &(to, weight) in edges {
                let to_id = id_of_node[&to];
                if to_id == from_id {
                    continue;
                }
                match self.slot_of[to_id as usize] {
                    u32::MAX => {
                        self.slot_of[to_id as usize] = cursor;
                        self.touched.push(to_id);
                        edge_target[cursor as usize] = to_id;
                        edge_weight[cursor as usize] = weight;
                        cursor += 1;
                    }
                    slot => {
                        let kept = &mut edge_weight[slot as usize];
                        if weight < *kept {
                            *kept = weight;
                        }
                    }
                }
            }
            self.reset_row();
        }
        (offsets, edge_target, edge_weight)
    }

    /// The number of distinct out-neighbours of `from_id` other than itself.
    fn row_degree<IndexType: GraphIndexInteger>(
        &mut self,
        from_id: u32,
        edges: &[(DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>)],
        id_of_node: &FastMap<DirectedNodeIndex<IndexType>, u32>,
    ) -> u32 {
        let mut degree = 0u32;
        for &(to, _) in edges {
            let to_id = id_of_node[&to];
            if to_id != from_id && self.slot_of[to_id as usize] == u32::MAX {
                self.slot_of[to_id as usize] = 0;
                self.touched.push(to_id);
                degree += 1;
            }
        }
        self.reset_row();
        degree
    }

    fn reset_row(&mut self) {
        for &id in &self.touched {
            self.slot_of[id as usize] = u32::MAX;
        }
        self.touched.clear();
    }
}

impl<IndexType: GraphIndexInteger> SkeletonAccess<IndexType> for Skeleton<IndexType> {
    #[inline]
    fn node_of_id(&self) -> &[DirectedNodeIndex<IndexType>] {
        &self.node_of_id
    }
    #[inline]
    fn offsets(&self) -> &[u32] {
        &self.offsets
    }
    #[inline]
    fn edge_target(&self) -> &[u32] {
        &self.edge_target
    }
    #[inline]
    fn edge_weight(&self) -> &[GfaPathLength<IndexType>] {
        &self.edge_weight
    }
    #[inline]
    fn node_id(&self, node: DirectedNodeIndex<IndexType>) -> Option<u32> {
        match &self.id_of_node {
            // Large skeleton: O(1) map lookup.
            Some(map) => map.get(&node).copied(),
            // Small skeleton: linear scan (node count <= SKELETON_MAP_THRESHOLD).
            None => scan_node_id(&self.node_of_id, node),
        }
    }
}

/// A reusable multi-source Dijkstra workspace over [`Skeleton`]s.
///
/// Distances are kept in a flat array reset in O(1) via an epoch stamp, and the heap is reused, so
/// running a query allocates nothing once the arrays are large enough.
pub struct DijkstraWorkspace<IndexType: GraphIndexInteger> {
    distance: Vec<GfaPathLength<IndexType>>,
    stamp: Vec<u32>,
    epoch: u32,
    heap: BinaryHeap<Reverse<(GfaPathLength<IndexType>, u32)>>,
}

impl<IndexType: GraphIndexInteger> Default for DijkstraWorkspace<IndexType> {
    fn default() -> Self {
        Self::new()
    }
}

impl<IndexType: GraphIndexInteger> DijkstraWorkspace<IndexType> {
    pub fn new() -> Self {
        Self {
            distance: Vec::new(),
            stamp: Vec::new(),
            epoch: 0,
            heap: BinaryHeap::new(),
        }
    }

    /// Size the distance/stamp arrays for `node_count` dense ids and the heap for as many entries, so a
    /// following [`Self::run`] on a skeleton of that size allocates (almost) nothing.
    pub fn reserve(&mut self, node_count: usize) {
        if self.distance.len() < node_count {
            self.distance
                .resize(node_count, GfaPathLength::from_usize(0));
            self.stamp.resize(node_count, 0);
        }
        self.heap
            .reserve(node_count.saturating_sub(self.heap.len()));
    }

    /// Run multi-source Dijkstra over `skeleton` from the given `(dense id, distance)` seeds. After
    /// this returns, [`Self::get`] reports the shortest distance to each reachable dense id.
    pub fn run<S: SkeletonAccess<IndexType> + ?Sized>(
        &mut self,
        skeleton: &S,
        seeds: &[(u32, GfaPathLength<IndexType>)],
    ) {
        let node_count = skeleton.node_count();
        let (offsets, edge_target, edge_weight) = (
            skeleton.offsets(),
            skeleton.edge_target(),
            skeleton.edge_weight(),
        );
        if self.distance.len() < node_count {
            self.distance
                .resize(node_count, GfaPathLength::from_usize(0));
            self.stamp.resize(node_count, 0);
        }
        // O(1) reset: bump the epoch; any `stamp != epoch` reads as "unvisited".
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Wrapped around; clear stamps so stale entries cannot alias the new epoch.
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
        self.heap.clear();

        for &(id, distance) in seeds {
            let id_usize = id as usize;
            if self.stamp[id_usize] != self.epoch || distance < self.distance[id_usize] {
                self.stamp[id_usize] = self.epoch;
                self.distance[id_usize] = distance;
                self.heap.push(Reverse((distance, id)));
            }
        }

        let overlay = skeleton.pole_overlay();
        // Overlay-only states (virtual ids) have no CSR row.
        let csr_nodes = offsets.len() - 1;
        while let Some(Reverse((distance, id))) = self.heap.pop() {
            let id_usize = id as usize;
            if self.stamp[id_usize] == self.epoch && distance > self.distance[id_usize] {
                continue; // stale heap entry
            }
            if id_usize < csr_nodes {
                let start = offsets[id_usize] as usize;
                let end = offsets[id_usize + 1] as usize;
                for edge in start..end {
                    self.relax(edge_target[edge], distance + edge_weight[edge]);
                }
            }
            if let Some(overlay) = overlay {
                for k in 0..4 {
                    if overlay.ids[k] != id {
                        continue;
                    }
                    for c in 0..4 {
                        // The diagonal is a self-loop: it never improves a distance.
                        if c == k {
                            continue;
                        }
                        if let Some(weight) = overlay.weights.get(k, c).into_option() {
                            self.relax(overlay.ids[c], distance + weight);
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn relax(&mut self, neighbour: u32, candidate: GfaPathLength<IndexType>) {
        let neighbour_usize = neighbour as usize;
        let improved =
            self.stamp[neighbour_usize] != self.epoch || candidate < self.distance[neighbour_usize];
        if improved {
            self.stamp[neighbour_usize] = self.epoch;
            self.distance[neighbour_usize] = candidate;
            self.heap.push(Reverse((candidate, neighbour)));
        }
    }

    /// The distance to dense id `id` from the last [`Self::run`], or `None` if unreached.
    pub fn get(&self, id: u32) -> Option<GfaPathLength<IndexType>> {
        let id_usize = id as usize;
        if self.stamp[id_usize] == self.epoch {
            Some(self.distance[id_usize])
        } else {
            None
        }
    }
}
