//! The node-row table: every block's rows filled (in parallel over blocks) and laid out in node
//! order, each node's start in its topology record.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use rayon::prelude::*;

use super::{aug::BlockAug, row_plan::RowPlan};
use crate::{
    gfa_graph_extensions::GfaNodeDataExt,
    image::Image,
    min_plus::{Mat2, MinPlus},
    node_rows::{BakedArc, RowKind, block_flip_rows, block_rows},
    path::GfaPathLength,
    sections,
    topology::{NodeRows, NodeTopology, ROW_CLOSED, ROW_MASK_BITS, push_header_word},
};

/// Cut count up to which a simple node's rows may be stored as a closed form (its first-hop mask has
/// [`ROW_MASK_BITS`] bits per orientation, two per directed cut).
const CLOSED_ROW_MAX_CUTS: usize = ROW_MASK_BITS as usize / 2;

/// The inputs of the node-row table, per block.
pub(super) struct RowBlocks<'l, IndexType: GraphIndexInteger> {
    pub(super) cuts: &'l [Vec<NodeIndex<IndexType>>],
    pub(super) aug: &'l [BlockAug<IndexType>],
    pub(super) flip_cuts: &'l [Vec<(NodeIndex<IndexType>, Mat2<IndexType>)>],
    pub(super) bake_arcs: &'l [Vec<BakedArc<IndexType>>],
    pub(super) plan: &'l RowPlan,
}

/// One block's stored rows (its row nodes in node order, closed-form nodes skipped) and each row
/// node's closed-form mask.
struct BlockRows<IndexType: GraphIndexInteger> {
    rows: Vec<MinPlus<IndexType>>,
    masks: Vec<Option<u32>>,
}

impl<IndexType: GraphIndexInteger> BlockRows<IndexType> {
    fn empty() -> Self {
        Self {
            rows: Vec::new(),
            masks: Vec::new(),
        }
    }
}

impl<IndexType: GraphIndexInteger + Send + Sync> RowBlocks<'_, IndexType> {
    /// Build the row table and push it: per block of kind `Full`/`Half`, one block-restricted
    /// backward search per directed cut and the `O(c^2)` fold with its `M_aug`, a simple node whose
    /// rows equal their closed form storing only its first-hop mask; per block of kind `Flip`, the
    /// raw rows to its flip cuts. In parallel over blocks, then laid out in node order (see
    /// [`NodeRows`]), and the node records re-pointed at it. Returns, per block, whether it is
    /// simple (no record, every row node a closed form): its nodes keep no arcs.
    pub(super) fn write<NodeData, EdgeData>(
        &self,
        image: &Image,
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        topology: &NodeTopology<'_, IndexType>,
    ) -> Vec<bool>
    where
        NodeData: GfaNodeData + Sync,
        EdgeData: GfaEdgeData + Sync,
    {
        let members = self.members(topology);
        let per_block: Vec<BlockRows<IndexType>> = (0..self.cuts.len())
            .into_par_iter()
            .map(|b| self.block_rows(graph, topology, b, &members[b]))
            .collect();
        let simple: Vec<bool> = per_block
            .iter()
            .zip(&members)
            .map(|(block, members)| {
                !block.masks.is_empty()
                    && block.masks.iter().all(Option::is_some)
                    && !members.iter().any(|&v| topology.record_loc(v).is_some())
            })
            .collect();
        let (rows, layout) = self.lay_out(topology, &per_block);
        let total = rows.len() as u64;
        let section = image.push_section(sections::LIFT_NODE_ROWS, &rows);
        topology.rewrite_row_starts(image, &layout, section, total);
        simple
    }

    fn kind(&self, block: usize) -> RowKind {
        self.plan.kind(block)
    }

    /// The stored entries per row node of `block`.
    fn width(&self, block: usize) -> usize {
        match self.kind(block) {
            RowKind::Full => 4 * self.cuts[block].len(),
            RowKind::Half => 2 * self.cuts[block].len(),
            RowKind::Flip => 4 * self.flip_cuts[block].len(),
            RowKind::None => 0,
        }
    }

    /// The members of every block with rows, in node order (cut nodes included: in-block routes
    /// pass through them).
    fn members(&self, topology: &NodeTopology<'_, IndexType>) -> Vec<Vec<NodeIndex<IndexType>>> {
        let mut members: Vec<Vec<NodeIndex<IndexType>>> = vec![Vec::new(); self.cuts.len()];
        for v in 0..topology.node_count() {
            let node = NodeIndex::from_usize(v);
            for block in topology.node_block_indices(node) {
                let b = block.into_usize();
                if self.kind(b) != RowKind::None {
                    members[b].push(node);
                }
            }
        }
        members
    }

    /// The rows of one block.
    fn block_rows<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &self,
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        topology: &NodeTopology<'_, IndexType>,
        b: usize,
        members: &[NodeIndex<IndexType>],
    ) -> BlockRows<IndexType> {
        if self.kind(b) == RowKind::None || members.is_empty() {
            return BlockRows::empty();
        }
        let row_nodes: Vec<NodeIndex<IndexType>> = members
            .iter()
            .copied()
            .filter(|&v| !topology.is_cut_node(v))
            .collect();
        if row_nodes.is_empty() {
            return BlockRows::empty();
        }
        match self.kind(b) {
            RowKind::Full | RowKind::Half => self.cut_rows(graph, b, members, &row_nodes),
            RowKind::Flip => {
                let flips: Vec<NodeIndex<IndexType>> =
                    self.flip_cuts[b].iter().map(|&(cut, _)| cut).collect();
                BlockRows {
                    rows: block_flip_rows(graph, members, &flips, &row_nodes, &self.bake_arcs[b]),
                    masks: vec![None; row_nodes.len()],
                }
            }
            RowKind::None => BlockRows::empty(),
        }
    }

    /// The full or half rows of one block: its full rows, each row node's closed form, and the
    /// cells of the nodes that are no closed form (all of them, or each cut's away orientation).
    fn cut_rows<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        &self,
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        b: usize,
        members: &[NodeIndex<IndexType>],
        row_nodes: &[NodeIndex<IndexType>],
    ) -> BlockRows<IndexType> {
        let (cuts, aug) = (&self.cuts[b], &self.aug[b].aug);
        let c = cuts.len();
        let full = block_rows(graph, members, cuts, aug, row_nodes);
        let masks: Vec<Option<u32>> = row_nodes
            .iter()
            .zip(full.chunks_exact(4 * c))
            .map(|(&v, node_row)| closed_form_mask(graph, cuts, aug, v, node_row))
            .collect();
        let mut rows =
            Vec::with_capacity(masks.iter().filter(|m| m.is_none()).count() * self.width(b));
        for (node_row, mask) in full.chunks_exact(4 * c).zip(&masks) {
            if mask.is_some() {
                continue;
            }
            if self.kind(b) == RowKind::Full {
                rows.extend_from_slice(node_row);
            } else {
                // Half rows: each cut's cell in its away orientation (see `RowKind::Half`).
                for o in 0..2 {
                    for (j, &a) in self.plan.away[b].iter().enumerate() {
                        rows.push(node_row[o * 2 * c + 2 * j + a]);
                    }
                }
            }
        }
        BlockRows { rows, masks }
    }

    /// Lay the rows out in node order: a header for every node that needs one (`[block]`, plus the
    /// SPQR location of a record-block node) followed by its stored cells; a closed-form or
    /// row-less node outside a record block needs none. Returns the table and every node's layout.
    fn lay_out(
        &self,
        topology: &NodeTopology<'_, IndexType>,
        per_block: &[BlockRows<IndexType>],
    ) -> (Vec<MinPlus<IndexType>>, Vec<NodeRows>) {
        let mut layout = vec![NodeRows::Simple { mask: 0 }; topology.node_count()];
        let mut next_row_node = vec![0usize; per_block.len()];
        let mut next_entry = vec![0usize; per_block.len()];
        let mut rows: Vec<MinPlus<IndexType>> =
            Vec::with_capacity(per_block.iter().map(|block| block.rows.len()).sum());
        for (v, slot) in layout.iter_mut().enumerate() {
            let node = NodeIndex::from_usize(v);
            if topology.is_cut_node(node) {
                continue;
            }
            let Some(block) = topology.node_block_indices(node).next() else {
                continue;
            };
            let b = block.into_usize();
            let BlockRows {
                rows: block_rows,
                masks,
                ..
            } = &per_block[b];
            let (mask, cells) = if masks.is_empty() {
                (0, None)
            } else {
                let k = next_row_node[b];
                next_row_node[b] += 1;
                match masks[k] {
                    Some(mask) => (mask, None),
                    None => {
                        let (w, start) = (self.width(b), next_entry[b]);
                        next_entry[b] += w;
                        (0, Some(&block_rows[start..start + w]))
                    }
                }
            };
            let record_loc = topology.record_loc(node);
            if record_loc.is_none() && cells.is_none() {
                *slot = NodeRows::Simple { mask };
                continue;
            }
            let header = u32::try_from(rows.len())
                .ok()
                .filter(|&h| h < ROW_CLOSED)
                .expect("cut rows exceed the record field");
            push_header_word(&mut rows, offset_u32(b));
            if let Some(loc) = record_loc {
                push_header_word(&mut rows, loc.loc);
            }
            if let Some(cells) = cells {
                rows.extend_from_slice(cells);
            }
            *slot = NodeRows::Header {
                header,
                cells: cells.is_some(),
                mask,
            };
        }
        (rows, layout)
    }
}

/// The first-hop mask stored instead of the rows of row node `v` (of a block with `cuts` and
/// row-major `M_aug` `aug`), or `None`: when `v` is simple (every arc leads to a cut of the block),
/// its rows may equal its closed form `len(v) + min over v^o's arcs y of M_aug(y -> w^q)`, checked
/// against its `4c` full-row cells `full`; a closed form is stored as a mask when the block has at
/// most [`CLOSED_ROW_MAX_CUTS`] cuts.
fn closed_form_mask<IndexType, NodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    cuts: &[NodeIndex<IndexType>],
    aug: &[MinPlus<IndexType>],
    v: NodeIndex<IndexType>,
    full: &[MinPlus<IndexType>],
) -> Option<u32>
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    let c = cuts.len();
    let dim = 2 * c;
    let len_v: GfaPathLength<IndexType> = graph.node_data(v).len();
    let len = len_v.into_usize() as u64;
    // The directed cuts each orientation's arcs reach; `None` = an arc to a non-cut.
    let first_hops = |o: usize| -> Option<Vec<usize>> {
        graph
            .iter_outgoing_edges(v.into_directed(o == 0))
            .map(|edge| {
                let to = edge.to();
                cuts.iter()
                    .position(|&cut| cut == to.into_bidirected())
                    .map(|i| 2 * i + usize::from(!to.is_forward()))
            })
            .collect()
    };
    let hops = [first_hops(0), first_hops(1)];
    let simple = hops.iter().all(Option::is_some);
    let closed = simple && {
        let cell = |o: usize, w: usize| -> MinPlus<IndexType> {
            hops[o]
                .as_ref()
                .unwrap()
                .iter()
                .filter_map(|&y| aug[y * dim + w].into_option())
                .map(|d| d.into_usize() as u64 + len)
                .min()
                .map_or(MinPlus::infinity(), |d| {
                    MinPlus::finite(GfaPathLength::from_usize(d as usize))
                })
        };
        (0..2).all(|o| (0..dim).all(|w| full[o * dim + w] == cell(o, w)))
    };
    (closed && c <= CLOSED_ROW_MAX_CUTS).then(|| {
        let mut mask = 0u32;
        for (o, hops) in hops.iter().enumerate() {
            for &y in hops.as_ref().unwrap() {
                mask |= 1 << (o as u32 * ROW_MASK_BITS + y as u32);
            }
        }
        mask
    })
}

fn offset_u32(value: usize) -> u32 {
    u32::try_from(value).expect("an id exceeds u32")
}
