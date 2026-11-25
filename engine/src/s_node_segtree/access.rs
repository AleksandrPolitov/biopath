//! Read access to one S-node oracle and its query kernels, shared by the owned build-time oracle
//! and the image-backed view.

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex};

use super::{
    PfxVertex, Role, SEG_KIND_RING, SEG_KIND_TREE, SegRunRec, SegVertexRec, orientation_bit,
    ring::Ring, tree::query_tree,
};
use crate::{
    min_plus::{
        MinPlus,
        network::{Mat4, merge, parallel, swap_poles},
        wide::{INF, narrow},
    },
    orientation::Orientation,
    path::GfaPathLength,
    pole_lift::{Pole, slot},
};

/// Read access to one S-node oracle, implemented by the owned [`SNodeSegTree`](super::SNodeSegTree)
/// (build time) and the image-backed [`SegTreeView`](super::SegTreeView) (query time), so the query
/// kernels are written once.
pub trait SegTreeAccess<IndexType: GraphIndexInteger> {
    /// Cycle vertices (`≥ 3`).
    fn m(&self) -> usize;
    /// The vertices, sorted by graph node.
    fn vertex_recs(&self) -> &[SegVertexRec<IndexType>];
    /// Tree form: the tree nodes (`2(m−1)−1`), then `closing_inner`, then `closing_lca`. Ring form:
    /// the block tree, the two closings, the run summaries.
    fn mats(&self) -> &[Mat4<IndexType>];
    /// Ring form: the `m` [`PfxVertex`] entries; empty for the tree form.
    fn pfx(&self) -> &[PfxVertex<IndexType>];
    /// Ring form: the runs; empty otherwise.
    fn runs(&self) -> &[SegRunRec];
    /// Ring form: the number of blocks under the block tree; `0` otherwise.
    fn blocks(&self) -> usize;

    /// [`SEG_KIND_TREE`] or [`SEG_KIND_RING`].
    #[inline]
    fn kind(&self) -> u32 {
        if !self.runs().is_empty() {
            SEG_KIND_RING
        } else {
            SEG_KIND_TREE
        }
    }

    /// Whether this oracle is in the ring form.
    #[inline]
    fn is_ring(&self) -> bool {
        self.kind() == SEG_KIND_RING
    }

    /// Ring form: the number of exception leaves on the path (the closing is not counted); `0`
    /// otherwise.
    #[inline]
    fn exceptions(&self) -> usize {
        self.runs().len().saturating_sub(1)
    }

    /// The cycle position of a graph node, if it is a vertex of this skeleton.
    #[inline]
    fn pos_of(&self, node: NodeIndex<IndexType>) -> Option<usize> {
        let key = node.into_usize();
        let recs = self.vertex_recs();
        recs.binary_search_by(|r| r.node.into_usize().cmp(&key))
            .ok()
            .map(|i| recs[i].pos.get() as usize)
    }

    /// The tree nodes (tree form).
    #[inline]
    fn tree(&self) -> &[Mat4<IndexType>] {
        let mats = self.mats();
        &mats[..mats.len() - 2]
    }

    /// The closing network over `(v_{m−1}, v_0)` for a role.
    #[inline]
    fn closing(&self, role: Role) -> &Mat4<IndexType> {
        let mats = self.mats();
        if self.is_ring() {
            &mats[2 * self.blocks() - 1 + role as usize]
        } else {
            &mats[mats.len() - 2 + role as usize]
        }
    }

    /// The ring form's pieces (ring form only).
    #[inline]
    fn ring(&self) -> Ring<'_, IndexType> {
        debug_assert!(self.is_ring());
        let b = self.blocks();
        let mats = self.mats();
        Ring {
            m: self.m(),
            pfx: self.pfx(),
            runs: self.runs(),
            blocks: b,
            tree: &mats[..2 * b - 1],
            closings: &mats[2 * b - 1..2 * b + 1],
            summaries: &mats[2 * b + 1..],
        }
    }

    /// Product of the path leaves `[l, r)` (`0 ≤ l < r ≤ m−1`), over poles `(v_l, v_r)` (tree form).
    #[inline]
    fn tree_range(&self, l: usize, r: usize) -> Option<Mat4<IndexType>> {
        query_tree(self.tree(), 0, 0, self.m() - 1, l, r)
    }

    /// The `4 × 4` all-pairs `d` over the boundary states of two distinct positions `p` and `q`, laid
    /// out `[(p,+), (p,−), (q,+), (q,−)]`: both arcs of the cycle between them, in parallel and closed.
    fn cut_pair(&self, p: usize, q: usize, role: Role) -> Mat4<IndexType> {
        debug_assert_ne!(p, q);
        if self.is_ring() {
            return self.ring().cut_pair(p, q, role);
        }
        let m = self.m();
        let (lo, hi) = if p < q { (p, q) } else { (q, p) };
        // The direct arc: leaves `lo..hi−1`, poles `(v_lo, v_hi)`; never contains the closing edge.
        let forward = self.tree_range(lo, hi).expect("non-empty range");
        // The wrap-around arc: leaves `hi..m−2`, the closing edge, leaves `0..lo−1`; poles `(v_hi, v_lo)`.
        let mut wrap = *self.closing(role);
        if hi < m - 1 {
            wrap = merge(
                &self.tree_range(hi, m - 1).expect("non-empty suffix"),
                &wrap,
            );
        }
        if lo > 0 {
            wrap = merge(&wrap, &self.tree_range(0, lo).expect("non-empty prefix"));
        }
        let combined = parallel(&forward, &swap_poles(&wrap));
        if p < q {
            combined
        } else {
            swap_poles(&combined)
        }
    }

    /// `d` between the two states of one position (raw orientations `o != o2`): the cheapest flip
    /// there over the role's skeleton.
    fn flip(&self, pos: usize, o: usize, o2: usize, role: Role) -> IndexType {
        debug_assert_ne!(o, o2);
        if self.is_ring() {
            let r = self.ring();
            return narrow(r.flip(role, pos, r.lane(pos, o)));
        }
        // Cut at that position and any other; both of its states are boundary states of the cut.
        let q = if pos + 1 < self.m() { pos + 1 } else { 0 };
        self.cut_pair(pos, q, role)[4 * o + o2]
    }

    /// `d` (empty walk allowed) from directed node `from` to directed node `to` over the role's
    /// skeleton; `None` when unreachable or when either node is not a vertex of this skeleton.
    #[cfg(test)]
    fn distance(
        &self,
        from: DirectedNodeIndex<IndexType>,
        to: DirectedNodeIndex<IndexType>,
        role: Role,
    ) -> Option<GfaPathLength<IndexType>> {
        let pf = self.pos_of(from.into_bidirected())?;
        let pt = self.pos_of(to.into_bidirected())?;
        let (of, ot) = (orientation_bit(from), orientation_bit(to));
        let raw = if pf == pt {
            if of == ot {
                return Some(GfaPathLength::from_usize(0));
            }
            self.flip(pf, of, ot, role)
        } else if self.is_ring() {
            let r = self.ring();
            narrow(r.distance(role, pf, r.lane(pf, of), pt, r.lane(pt, ot)))
        } else {
            self.cut_pair(pf, pt, role)[4 * of + 2 + ot]
        };
        crate::min_plus::word::length(raw)
    }

    /// The inner-skeleton distances from `seed` to the node's two reference poles — `v_0 = s` and
    /// `v_{m−1} = t` — in slot order `[s^+, s^−, t^+, t^−]` (the pole-state slots); `None` when `seed` is not
    /// a vertex of this skeleton. Shares the prefix/suffix products between the two pole reads.
    /// Only meaningful for a non-root node (whose closing edge is its reference edge).
    fn pole_projection(
        &self,
        seed: DirectedNodeIndex<IndexType>,
    ) -> Option<[MinPlus<IndexType>; 4]> {
        let m = self.m();
        let p = self.pos_of(seed.into_bidirected())?;
        let o = orientation_bit(seed);
        let mut out = [MinPlus::infinity(); 4];
        if self.is_ring() {
            let r = self.ring();
            let x = r.lane(p, o);
            for (pole, pos) in [(Pole::Source, 0usize), (Pole::Target, m - 1)] {
                let segs = r.pair_segs(Role::Inner, p, pos);
                for o2 in 0..2 {
                    let d = r.dist_with(Role::Inner, p, x, pos, r.lane(pos, o2), &segs);
                    if d != INF {
                        out[slot(pole, Orientation::from_index(o2))] =
                            MinPlus::finite(GfaPathLength::from_usize(d as usize));
                    }
                }
            }
            return Some(out);
        }
        let closing = *self.closing(Role::Inner);
        let mut put = |pole: Pole, mat: &Mat4<IndexType>, same_vertex: bool| {
            for o2 in 0..2 {
                let raw = if same_vertex {
                    mat[4 * o + o2]
                } else {
                    mat[4 * o + 2 + o2]
                };
                out[slot(pole, Orientation::from_index(o2))] = MinPlus::from_word(raw);
            }
        };
        if p == 0 {
            // The seed sits at `s`: one cut `(0, m−1)` yields both poles.
            let forward = self.tree_range(0, m - 1).expect("m >= 3");
            let c = parallel(&forward, &swap_poles(&closing));
            put(Pole::Source, &c, true);
            put(Pole::Target, &c, false);
        } else if p == m - 1 {
            // The seed sits at `t`: cut `(m−1, 0)` over `[(v_{m−1},±), (v_0,±)]`.
            let forward = swap_poles(&self.tree_range(0, m - 1).expect("m >= 3"));
            let c = parallel(&forward, &closing);
            put(Pole::Target, &c, true);
            put(Pole::Source, &c, false);
        } else {
            let prefix = self.tree_range(0, p).expect("non-empty prefix"); // (v_0, v_p)
            let suffix = self.tree_range(p, m - 1).expect("non-empty suffix"); // (v_p, v_{m−1})
            // Cut `(p, 0)` over `[(v_p,±), (v_0,±)]`: direct = prefix reversed, wrap = suffix ⋈ closing.
            let to_s = parallel(&swap_poles(&prefix), &merge(&suffix, &closing));
            put(Pole::Source, &to_s, false);
            // Cut `(p, m−1)` over `[(v_p,±), (v_{m−1},±)]`: direct = suffix, wrap = closing ⋈ prefix.
            let to_t = parallel(&suffix, &swap_poles(&merge(&closing, &prefix)));
            put(Pole::Target, &to_t, false);
        }
        Some(out)
    }
}
