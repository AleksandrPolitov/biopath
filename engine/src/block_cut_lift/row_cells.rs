//! Reading the node cut rows at query time: a node's two cells for one cut (full, half or closed
//! form), the route between two nodes of one block through both their rows, and the in-block
//! distance of a simple block, whose nodes keep no arcs.

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use super::{BlockCutLift, cut_lists::CutList, orientation_index, reversed_length};
use crate::{
    chain_nav::block_bits,
    min_plus::{MinPlus, wide::INF},
    node_rows::RowKind,
    path::GfaPathLength,
    topology::{NodeTopology, ROW_MASK_BITS, RowRef},
};

impl<'a, IndexType: GraphIndexInteger + 'static> BlockCutLift<'a, IndexType> {
    /// The node-row kind of a block.
    #[inline]
    pub(super) fn row_kind(&self, block: usize) -> RowKind {
        RowKind::from_u8((self.block_words[block] & block_bits::ROW_KIND_MASK) as u8)
    }

    /// The away orientation of the cut at position `j` of a half-row block (the traversal leaving
    /// the block's side; stored at build from the block's arcs at that cut).
    #[inline]
    fn away_orientation(&self, block: usize, j: usize) -> usize {
        let bits = self.block_words[block];
        if bits & block_bits::SPECIAL != 0 {
            ((self.special(block).0.away >> j) & 1) as usize
        } else if j == 0 {
            usize::from(bits & block_bits::AWAY0 != 0)
        } else {
            usize::from(bits & block_bits::AWAY1 != 0)
        }
    }

    /// Both cells `q = 0, 1` of a node's rows for cut `j`: `[o][2 * j + q]` of a full-row block, or
    /// the half-row cell plus the cut's flip closure for the orientation that is not the cut's away
    /// orientation; for a closed-form node, `len + min over its arcs y of v^o of M_aug(y -> cut_j^q)`
    /// (`len` = the node's length; one `M_aug` block per arc serves both cells).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn row_cells(
        &self,
        block: usize,
        kind: RowKind,
        row: RowRef,
        len: GfaPathLength<IndexType>,
        o: usize,
        list: &CutList<'_>,
        j: usize,
    ) -> [MinPlus<IndexType>; 2] {
        let c = list.len();
        let start = match row {
            RowRef::Stored(start) => start,
            RowRef::Closed { mask } => {
                let to = list.get(j) as usize;
                let mut bits = (mask >> (o as u32 * ROW_MASK_BITS)) & ((1 << ROW_MASK_BITS) - 1);
                let mut best = [MinPlus::infinity(); 2];
                while bits != 0 {
                    let y = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let m = self.aug_matrix(block, list.get(y / 2) as usize, to);
                    for (q, slot) in best.iter_mut().enumerate() {
                        *slot = slot.oplus(m.get(y % 2, q));
                    }
                }
                return best.map(|d| d.otimes(MinPlus::finite(len)));
            }
        };
        match kind {
            RowKind::Full => {
                let at = start + o * 2 * c + 2 * j;
                [self.node_rows[at], self.node_rows[at + 1]]
            }
            RowKind::Half => {
                let cut = list.get(j) as usize;
                let away = self.away_orientation(block, j);
                let base = self.node_rows[start + o * c + j];
                let other = base.otimes(self.full_entry(cut, away, 1 - away));
                if away == 0 {
                    [base, other]
                } else {
                    [other, base]
                }
            }
            RowKind::Flip | RowKind::None => {
                unreachable!("row cells are read on full/half rows only")
            }
        }
    }

    /// The route from non-cut `a` to non-cut `b` of one block through its cuts, from both rows: `min
    /// over cuts w and q of row(a)[w^q] + row(b̄)[w^{¬q}] + len(w) - len(b)` (the reversal identity
    /// turns `b̄`'s row into the distances to `b`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rows_route(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
        block: usize,
        kind: RowKind,
        sa: RowRef,
        sb: RowRef,
        list: &CutList<'_>,
    ) -> Option<GfaPathLength<IndexType>> {
        let (oa, ob) = (orientation_index(a), 1 - orientation_index(b));
        let (a_len, b_len_ix) = (
            topology.node_len(a.into_bidirected()),
            topology.node_len(b.into_bidirected()),
        );
        let b_len = b_len_ix.into_usize() as u64;
        let mut best = INF;
        for j in 0..list.len() {
            let w_len = self.cut_len(list.get(j) as usize).into_usize() as u64;
            let xs = self.row_cells(block, kind, sa, a_len, oa, list, j);
            let ys = self.row_cells(block, kind, sb, b_len_ix, ob, list, j);
            for q in 0..2 {
                let (Some(x), Some(y)) = (xs[q].into_option(), ys[1 - q].into_option()) else {
                    continue;
                };
                best = best.min(x.into_usize() as u64 + y.into_usize() as u64 + w_len - b_len);
            }
        }
        (best != INF).then(|| GfaPathLength::from_usize(best as usize))
    }

    /// The out-neighbours of a closed-form node of `block` in orientation `o` (`0` forward): its mask's
    /// arcs, all to the block's cuts (see `crate::topology::RowRef::Closed`).
    pub fn closed_form_neighbors(
        &self,
        block: usize,
        mask: u32,
        o: usize,
    ) -> impl Iterator<Item = DirectedNodeIndex<IndexType>> + use<'a, IndexType> {
        let lift = *self;
        let list = self.block_cut_list(block);
        let mut bits = (mask >> (o as u32 * ROW_MASK_BITS)) & ((1 << ROW_MASK_BITS) - 1);
        std::iter::from_fn(move || {
            (bits != 0).then(|| {
                let y = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                lift.cut_node(list.get(y / 2) as usize)
                    .into_directed(y.is_multiple_of(2))
            })
        })
    }

    /// Whether every non-cut node of `block` has closed-form rows and it has no record (see
    /// [`block_bits::SIMPLE_ROWS`]): its nodes keep no arcs in the index.
    #[inline]
    pub fn block_is_simple(&self, block: usize) -> bool {
        self.block_words[block] & block_bits::SIMPLE_ROWS != 0
    }

    /// The start-to-start distance from `a` to `b`, two distinct directed nodes of a
    /// [simple](Self::block_is_simple) block, through its rows: a simple node's arcs all lead to cuts,
    /// so every walk between them leaves and enters through the block's cuts, which the rows (and
    /// `M_aug` between cuts) cover. That is the whole-graph distance, so it also bounds the in-block
    /// one (the caller takes the minimum with the cross-block route).
    pub fn simple_block_distance(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        a: DirectedNodeIndex<IndexType>,
        b: DirectedNodeIndex<IndexType>,
        block: usize,
    ) -> Option<GfaPathLength<IndexType>> {
        let kind = self.row_kind(block);
        let list = self.block_cut_list(block);
        let pos = |node: DirectedNodeIndex<IndexType>| {
            topology.cut_id(node.into_bidirected()).map(|c| {
                list.position(c as u32)
                    .expect("a cut of the block is in its list")
            })
        };
        let (a_seg, b_seg) = (a.into_bidirected(), b.into_bidirected());
        let (pa, pb) = (orientation_index(a), orientation_index(b));
        match (pos(a), pos(b)) {
            (None, None) => {
                let (sa, sb) = (topology.row_ref(a_seg)?, topology.row_ref(b_seg)?);
                self.rows_route(topology, a, b, block, kind, sa, sb, &list)
            }
            // `D(a -> w^q)` is a's row cell.
            (None, Some(j)) => {
                let cells = self.row_cells(
                    block,
                    kind,
                    topology.row_ref(a_seg)?,
                    topology.node_len(a_seg),
                    pa,
                    &list,
                    j,
                );
                cells[pb].into_option()
            }
            // `D(w^p -> b) = D(b̄ -> w^{¬p}) + len(w) - len(b)` (the reversal identity).
            (Some(i), None) => {
                let b_len = topology.node_len(b_seg);
                let cells = self.row_cells(
                    block,
                    kind,
                    topology.row_ref(b_seg)?,
                    b_len,
                    1 - pb,
                    &list,
                    i,
                );
                let w_len = self.cut_len(list.get(i) as usize).into_usize() as u64;
                Some(reversed_length(
                    cells[1 - pa].into_option()?,
                    w_len,
                    b_len.into_usize() as u64,
                ))
            }
            (Some(i), Some(j)) => self
                .aug_matrix(block, list.get(i) as usize, list.get(j) as usize)
                .get(pa, pb)
                .into_option(),
        }
    }
}
