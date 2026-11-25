//! The cross-block query ([`BlockCutLift::cross`]): the same-block route through a block's
//! flip-capable cuts, or the route along the block-cut path (seed, lift up, turn, lift down,
//! read).

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex};
use spqr_tree::decomposition::indices::BlockIndex;

use super::{BlockCutLift, IntraDistance, orientation_index, reversed_length};
use crate::{
    min_plus::{Mat2, MinPlus},
    node_rows::RowKind,
    orientation::Orientation,
    path::GfaPathLength,
    topology::{NodeTopology, RowRef},
};

/// Per flip-capable cut of a same-block route: its position in the block's `M_aug`,
/// `D_B(a -> cut^o)` and `D_B(cut^o -> b)`.
type FlipEnd<IndexType> = (usize, [MinPlus<IndexType>; 2], [MinPlus<IndexType>; 2]);

impl<'a, IndexType: GraphIndexInteger + 'static> BlockCutLift<'a, IndexType> {
    /// The cross-block start-to-start distance from directed node `a` to `b` (both at offset 0),
    /// or `None` if there is no cross-block route. The caller still takes the min with the direct
    /// intra-block distance.
    pub fn cross(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        intra: &impl IntraDistance<IndexType>,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
    ) -> Option<GfaPathLength<IndexType>> {
        let a_anchor = self.anchor(topology, a)?;
        let b_anchor = self.anchor(topology, b)?;
        if a_anchor == b_anchor && self.is_block(a_anchor) {
            // Same block: the direct intra-block distance (computed by the caller) covers routes
            // that stay inside the block. A route may still leave and re-enter — possibly several
            // times, e.g. flip at one cut, run the block on the other strand, flip at another —
            // which `same_block_cross` covers through the block's augmented cut matrix.
            return self.same_block_cross(topology, intra, a, b, a_anchor);
        }
        // Different anchors, or one cut node (its whole-graph flip closure): the block-cut path.
        self.path_cross(topology, intra, (a, a_anchor), (b, b_anchor))
    }

    /// The cross-block route along the block-cut path between the anchors: the seed at the
    /// source's on-path cut `c1`, lifted up to the LCA, turned through it, lifted down to the
    /// target's on-path cut `cm`, and the read into `b`.
    fn path_cross(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        intra: &impl IntraDistance<IndexType>,
        (a, a_anchor): (DirectedNodeIndex<IndexType>, usize),
        (b, b_anchor): (DirectedNodeIndex<IndexType>, usize),
    ) -> Option<GfaPathLength<IndexType>> {
        let lca = self.nav.lca(a_anchor, b_anchor);

        // Source: the on-path cut c1 next to `a`, its seed, lifted to the source leg's top cut.
        let (c1, seed) = self.source_seed(topology, intra, a, a_anchor, b_anchor, lca);
        let source_top = self.side_top(a_anchor, b_anchor, lca);
        let vs = self.lift_up_vector(c1, source_top, seed);

        // Target: symmetric.
        let (cm, read) = self.target_read(topology, intra, b, b_anchor, a_anchor, lca);
        let target_top = self.side_top(b_anchor, a_anchor, lca);

        // The connector across the LCA: a flip closure when both legs meet at one cut node, else
        // the LCA block's augmented turn between the two side tops.
        let connector = if source_top == target_top {
            self.full(source_top)
        } else {
            self.aug_matrix(lca, source_top, target_top)
        };
        let v_meet = connector.left_multiply_vector(&vs);
        let v_cm = self.lift_down_vector(target_top, cm, v_meet);

        let mut best = MinPlus::infinity();
        for o in 0..2 {
            best.min_assign(v_cm[o].otimes(read[o]));
        }
        best.into_option()
    }

    /// The same-block cross route: the shortest `a -> b` walk (both in block `B`) that leaves `B` at
    /// least once, or `None` if there is none.
    ///
    /// Distinct cut nodes of `B` are separated by `B`, so every excursion out of `B` leaves and
    /// re-enters through the same cut. An excursion that comes back on the same strand is a closed
    /// detour and never helps; a useful one flips strand, so it happens at a **flip-capable** cut. Such
    /// a walk therefore splits into `a -> z^p` inside `B` (up to its first exit), `z^p -> w^q` (from
    /// the first exit to the last re-entry, any number of excursions and in-block legs between), and
    /// `w^q -> b` inside `B`. The middle part is exactly what the augmented cut matrix `M_aug` closes
    /// over (in-block cut-to-cut distances plus the flip closure at every cut), so
    ///
    /// `min over flip-capable z, w and p, q of  D_B(a -> z^p) + M_aug(z^p -> w^q) + D_B(w^q -> b)`
    ///
    /// is exact. (`z == w` with one excursion is the single-trip case; `z != w` covers walks that must
    /// leave twice, such as flipping at one cut, running `B` on the other strand, and flipping at
    /// another — which a single-trip minimum misses.)
    ///
    /// With node rows the whole thing is `min over w^q of row_a[w^q] + row_b̄[w^{¬q}] + len(w) -
    /// len(b)`: `row_a[w^q]` folds the first excursion's leg through `M_aug` into `w^q`, and the
    /// reversed last leg `D_B(w^q -> b) = D_B(b̄ -> w^{¬q}) + len(w) - len(b)` is read from `b̄`'s row
    /// (whose fold through `M_aug` only adds further excursions, which `M_aug`'s closure already
    /// contains, so the minimum is unchanged). Every cut of the block may be `w`: a non-flip cut only
    /// contributes in-block walks, which the direct answer already covers. Without rows, `D_B` comes
    /// from the flip rows or from the intra-block engine via `intra` (`4 x #flip cuts` calls).
    fn same_block_cross(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        intra: &impl IntraDistance<IndexType>,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
        block: usize,
    ) -> Option<GfaPathLength<IndexType>> {
        if self.flip_list(block).is_empty() {
            return None; // no cut node can flip through a pendant block: the direct answer is optimal.
        }
        let kind = self.row_kind(block);
        let rows = (
            topology.row_ref(a.into_bidirected()),
            topology.row_ref(b.into_bidirected()),
        );
        if let (RowKind::Full | RowKind::Half, (Some(ra), Some(rb))) = (kind, rows) {
            let list = self.block_cut_list(block);
            return self.rows_route(topology, a, b, block, kind, ra, rb, &list);
        }
        let ends = match (kind, rows) {
            (RowKind::Flip, (Some(RowRef::Stored(ra)), Some(RowRef::Stored(rb)))) => {
                self.flip_ends_by_rows(topology, a, b, block, ra, rb)
            }
            _ => self.flip_ends_by_intra(intra, a, b, block),
        };
        let derived;
        let aug: &[MinPlus<IndexType>] = if self.aug_is_derived(block) {
            derived = self.derived_aug_dense(block);
            &derived
        } else {
            self.aug_dense(block)
        };
        let dim = 2 * self.block_cut_list(block).len();
        let mut best = MinPlus::infinity();
        for &(z, to_z, _) in &ends {
            for (p, to) in to_z.into_iter().enumerate() {
                if to.is_infinite() {
                    continue;
                }
                let row = &aug[(2 * z + p) * dim..(2 * z + p + 1) * dim];
                for &(w, _, from_w) in &ends {
                    for (q, from) in from_w.into_iter().enumerate() {
                        best.min_assign(to.otimes(row[2 * w + q]).otimes(from));
                    }
                }
            }
        }
        best.into_option()
    }

    /// The [`FlipEnd`]s of a block's flip-capable cuts from its raw flip rows: `D_B(a -> cut^o)` is
    /// a cell of `a`'s row, `D_B(cut^o -> b)` the reversal of a cell of `b̄`'s row.
    fn flip_ends_by_rows(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
        block: usize,
        row_a: usize,
        row_b: usize,
    ) -> Vec<FlipEnd<IndexType>> {
        let flips = self.flip_list(block);
        let ra = row_a + orientation_index(a) * 2 * flips.len();
        let rb = row_b + (1 - orientation_index(b)) * 2 * flips.len();
        let b_len = topology.node_len(b.into_bidirected()).into_usize() as u64;
        flips
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, &cid)| {
                let cid = cid as usize;
                let c_len = self.cut_len(cid).into_usize() as u64;
                let mut to_c = [MinPlus::infinity(); 2];
                let mut from_c = [MinPlus::infinity(); 2];
                for o in 0..2 {
                    to_c[o] = self.node_rows[ra + 2 * i + o];
                    if let Some(d) = self.node_rows[rb + 2 * i + (1 - o)].into_option() {
                        from_c[o] = MinPlus::finite(reversed_length(d, c_len, b_len));
                    }
                }
                (self.aug_local(block, cid), to_c, from_c)
            })
            .collect()
    }

    /// The [`FlipEnd`]s of a block's flip-capable cuts through the block's engine: four `intra`
    /// calls per cut.
    fn flip_ends_by_intra(
        &self,
        intra: &impl IntraDistance<IndexType>,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
        block: usize,
    ) -> Vec<FlipEnd<IndexType>> {
        let block_index = BlockIndex::from_usize(block);
        self.flip_list(block)
            .as_slice()
            .iter()
            .map(|&cid| {
                let cid = cid as usize;
                let c = self.cut_node(cid);
                let mut to_c = [MinPlus::infinity(); 2];
                let mut from_c = [MinPlus::infinity(); 2];
                for o in 0..2 {
                    let c_dir = c.into_directed(Orientation::from_index(o).is_forward());
                    if let Some(d) = intra(a, c_dir, block_index) {
                        to_c[o] = MinPlus::finite(d);
                    }
                    if let Some(d) = intra(c_dir, b, block_index) {
                        from_c[o] = MinPlus::finite(d);
                    }
                }
                (self.aug_local(block, cid), to_c, from_c)
            })
            .collect()
    }

    /// The bipartite tree id anchoring a directed node: its cut tree id if it is a cut node, else
    /// its (unique) block's tree id. `None` if the node is in no block (isolated).
    fn anchor(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        node: DirectedNodeIndex<IndexType>,
    ) -> Option<usize> {
        let segment = node.into_bidirected();
        if let Some(c) = topology.cut_id(segment) {
            return Some(self.tree_id_of_cut(c));
        }
        let block_index = topology.node_block_indices(segment).next()?;
        Some(block_index.into_usize())
    }

    /// The cut node `c1` adjacent to `a`'s anchor on the path towards `b_anchor`, plus the seed
    /// vector `D(a -> c1^o)`.
    ///
    /// A cut endpoint is `c1` itself and seeds its whole-graph flip closure. Inside a block with
    /// full or half rows the seed is two row cells. Otherwise `seed[o] = min over z^p of
    /// D_B(a -> z^p) + M_aug(z^p -> c1^o)`, where only `z = c1` and the flip-capable cuts can attain
    /// the minimum: drop a shortest walk's same-strand excursions (closed detours, never negative);
    /// if none is left the walk stays in the block (the `z = c1` term), else its first excursion
    /// leaves at some cut `z` and returns flipped through `z`'s outside, so `z` is flip-capable (or
    /// `c1`), the walk before it is in-block and the rest a cut-to-cut walk that `M_aug` covers. This
    /// is the argument of `same_block_cross`; it takes `2 x (flip cuts + 1)` `intra` calls instead
    /// of `2 x (block cuts)`, and flip rows answer all but the `c1` term.
    fn source_seed(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        intra: &impl IntraDistance<IndexType>,
        a: DirectedNodeIndex<IndexType>,
        a_anchor: usize,
        b_anchor: usize,
        lca: usize,
    ) -> (usize, [MinPlus<IndexType>; 2]) {
        if !self.is_block(a_anchor) {
            let cut = self.cut_of(a_anchor);
            let ao = orientation_index(a);
            return (cut, [0, 1].map(|o| self.full_entry(cut, ao, o)));
        }
        let c1_tree = self.adjacent_cut_tree(a_anchor, b_anchor, lca);
        let c1 = self.cut_of(c1_tree);
        let block = a_anchor;
        let kind = self.row_kind(block);
        let row = topology.row_ref(a.into_bidirected());
        let o = orientation_index(a);
        if let (RowKind::Full | RowKind::Half, Some(row)) = (kind, row) {
            let list = self.block_cut_list(block);
            let pos = self.adjacent_cut_pos(block, c1_tree);
            let a_len = topology.node_len(a.into_bidirected());
            return (c1, self.row_cells(block, kind, row, a_len, o, &list, pos));
        }
        let to_cut = |z: usize| -> [MinPlus<IndexType>; 2] {
            let z_node = self.cut_node(z);
            [0, 1].map(|p| intra_entry(intra, a, directed(z_node, p), block))
        };
        let mut seed = [MinPlus::infinity(); 2];
        fold_through(&mut seed, to_cut(c1), || self.aug_matrix(block, c1, c1));
        let flips = self.flip_list(block);
        let flips = flips.as_slice();
        if let (RowKind::Flip, Some(RowRef::Stored(start))) = (kind, row) {
            let base = start + o * 2 * flips.len();
            for (i, &z) in flips.iter().enumerate() {
                let z = z as usize;
                if z != c1 {
                    let to_z = [
                        self.node_rows[base + 2 * i],
                        self.node_rows[base + 2 * i + 1],
                    ];
                    fold_through(&mut seed, to_z, || self.aug_matrix(block, z, c1));
                }
            }
        } else {
            for &z in flips {
                let z = z as usize;
                if z != c1 {
                    fold_through(&mut seed, to_cut(z), || self.aug_matrix(block, z, c1));
                }
            }
        }
        (c1, seed)
    }

    /// The cut node `cm` adjacent to `b`'s anchor on the path towards `a_anchor`, plus the read
    /// vector `D(cm^o -> b)`: the mirror of [`Self::source_seed`] (the walk's last flipping
    /// excursion, after which it stays in the block). Rows give `D(cm^o -> b)` as the reversed walk
    /// `b̄ -> cm^{¬o}` ([`reversed_length`]); `b` is never a cut node here (a cut endpoint anchors at
    /// its cut).
    fn target_read(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        intra: &impl IntraDistance<IndexType>,
        b: DirectedNodeIndex<IndexType>,
        b_anchor: usize,
        a_anchor: usize,
        lca: usize,
    ) -> (usize, [MinPlus<IndexType>; 2]) {
        if !self.is_block(b_anchor) {
            let cut = self.cut_of(b_anchor);
            let bo = orientation_index(b);
            return (cut, [0, 1].map(|o| self.full_entry(cut, o, bo)));
        }
        let cm_tree = self.adjacent_cut_tree(b_anchor, a_anchor, lca);
        let cm = self.cut_of(cm_tree);
        let block = b_anchor;
        let b_seg = b.into_bidirected();
        let kind = self.row_kind(block);
        let row = topology.row_ref(b_seg);
        let ob = 1 - orientation_index(b);
        let b_len_ix = topology.node_len(b_seg);
        let b_len = b_len_ix.into_usize() as u64;
        if let (RowKind::Full | RowKind::Half, Some(row)) = (kind, row) {
            let list = self.block_cut_list(block);
            let pos = self.adjacent_cut_pos(block, cm_tree);
            let cm_len = self.cut_len(cm).into_usize() as u64;
            let cells = self.row_cells(block, kind, row, b_len_ix, ob, &list, pos);
            let read = [0, 1].map(|o| match cells[1 - o].into_option() {
                Some(d) => MinPlus::finite(reversed_length(d, cm_len, b_len)),
                None => MinPlus::infinity(),
            });
            return (cm, read);
        }
        let from_cut = |z: usize| -> [MinPlus<IndexType>; 2] {
            let z_node = self.cut_node(z);
            [0, 1].map(|p| intra_entry(intra, directed(z_node, p), b, block))
        };
        let mut read = [MinPlus::infinity(); 2];
        fold_from(&mut read, || self.aug_matrix(block, cm, cm), from_cut(cm));
        let flips = self.flip_list(block);
        let flips = flips.as_slice();
        if let (RowKind::Flip, Some(RowRef::Stored(start))) = (kind, row) {
            let base = start + ob * 2 * flips.len();
            for (i, &z) in flips.iter().enumerate() {
                let z = z as usize;
                if z == cm {
                    continue;
                }
                let z_len = self.cut_len(z).into_usize() as u64;
                let z_to_b =
                    [0, 1].map(
                        |p| match self.node_rows[base + 2 * i + 1 - p].into_option() {
                            Some(d) => MinPlus::finite(reversed_length(d, z_len, b_len)),
                            None => MinPlus::infinity(),
                        },
                    );
                fold_from(&mut read, || self.aug_matrix(block, cm, z), z_to_b);
            }
        } else {
            for &z in flips {
                let z = z as usize;
                if z != cm {
                    fold_from(&mut read, || self.aug_matrix(block, cm, z), from_cut(z));
                }
            }
        }
        (cm, read)
    }

    /// Position in `block`'s cut list of the cut adjacent to it on the path towards the other endpoint
    /// (`cut_tree` from [`Self::adjacent_cut_tree`]).
    #[inline]
    fn adjacent_cut_pos(&self, block: usize, cut_tree: usize) -> usize {
        self.block_cut_list(block)
            .position(self.cut_of(cut_tree) as u32)
            .expect("the adjacent cut is in the block's cut list")
    }

    /// The cut node (tree id) adjacent to block `anchor` on the path towards `other`.
    fn adjacent_cut_tree(&self, anchor: usize, other: usize, lca: usize) -> usize {
        if anchor == lca {
            // `anchor` is the LCA block: the path leaves it downward towards `other`.
            self.nav.child_on_path_to(anchor, other).unwrap()
        } else {
            // `anchor` is strictly below the LCA: the path leaves it upward.
            self.nav.parent(anchor).unwrap()
        }
    }

    /// The cut node (dense id) at the top of `anchor`'s leg (just below the LCA, or the LCA itself
    /// when the LCA is a cut node).
    fn side_top(&self, anchor: usize, other: usize, lca: usize) -> usize {
        if self.is_block(lca) {
            // Cut child of the LCA block on this side. When `anchor == lca`, "this side" is towards
            // the other endpoint (the source is inside the LCA block, so its top is the target-side
            // child); both sides then coincide.
            let toward = if anchor == lca { other } else { anchor };
            self.cut_of(self.nav.child_on_path_to(lca, toward).unwrap())
        } else {
            // The LCA is a cut node: both legs meet there.
            self.cut_of(lca)
        }
    }
}

/// Cut `node` in orientation index `o`.
fn directed<IndexType: GraphIndexInteger>(
    node: NodeIndex<IndexType>,
    o: usize,
) -> DirectedNodeIndex<IndexType> {
    node.into_directed(Orientation::from_index(o).is_forward())
}

/// `intra(from, to, block)` as a min-plus scalar (`+∞` if unreached).
fn intra_entry<IndexType: GraphIndexInteger>(
    intra: &impl IntraDistance<IndexType>,
    from: DirectedNodeIndex<IndexType>,
    to: DirectedNodeIndex<IndexType>,
    block: usize,
) -> MinPlus<IndexType> {
    intra(from, to, BlockIndex::from_usize(block))
        .map(MinPlus::finite)
        .unwrap_or_else(MinPlus::infinity)
}

/// `acc[o] min= to[p] + m(p -> o)` over the finite `to[p]` (`m` is read only if one is finite).
fn fold_through<IndexType: GraphIndexInteger>(
    acc: &mut [MinPlus<IndexType>; 2],
    to: [MinPlus<IndexType>; 2],
    m: impl FnOnce() -> Mat2<IndexType>,
) {
    if to.iter().all(MinPlus::is_infinite) {
        return;
    }
    m().relax_left_product(acc, to);
}

/// `acc[o] min= m(o -> p) + from[p]` over the finite `from[p]` (`m` is read only if one is finite).
fn fold_from<IndexType: GraphIndexInteger>(
    acc: &mut [MinPlus<IndexType>; 2],
    m: impl FnOnce() -> Mat2<IndexType>,
    from: [MinPlus<IndexType>; 2],
) {
    if from.iter().all(MinPlus::is_infinite) {
        return;
    }
    m().relax_right_product(acc, from);
}
