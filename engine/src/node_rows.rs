//! Per-node cut rows: every non-cut graph node of an eligible block stores, for each cut `w` of its
//! block and each orientation, the augmented distance `min over z^p of D_B(v^o -> z^p) + M_aug(z^p ->
//! w^q)` — exactly the seed [`crate::block_cut_lift::BlockCutLift`] folds on the cross-block path,
//! precomputed for every cut of the block. With them the cross-block route reads two cells for the
//! source leg, two for the target leg (through the start-to-start reversal identity) and `4c` for a
//! same-block excursion, and never opens the block's record.
//!
//! The rows are laid out in node order in the lift's row table, each node's header position in its
//! [`crate::topology::NodeRecord`]. A block gets one row kind ([`RowKind`]):
//! - **full** rows for a block with `1..=max_cuts` cuts: `[orientation][2 * cut position + cut
//!   orientation]`, `4c` entries per node, `M_aug` folded in ([`block_rows`]);
//! - **half** rows for such a block that touches each cut on one side: each cut's away orientation
//!   only, `2c` entries per node ([`RowKind::Half`]);
//! - **flip** rows for a block with more cuts but at most [`FLIP_RESERVE_CUTS`] flip-capable ones:
//!   the raw `D_B(v^o -> f_i^p)` to those flip cuts only (`[o][2i + p]`, `4F` entries;
//!   [`block_flip_rows`]). The on-path cut of a cross-block query is then still read through the
//!   block's record.
//!
//! The lift, which knows the flip cuts, decides the kind, fills the rows and lays them out (a simple
//! node whose rows equal their closed form stores none; see `crate::topology::RowRef`).
//!
//! Build: one block-restricted backward SSSP per directed cut over the block's edges (the same
//! cost model as [`crate::block_dijkstra::BlockDijkstra`]: leaving a node costs its length, the
//! source is not self-settled; a flip-baked block's baked flip edges included), then the `O(c^2)`
//! fold with `M_aug` per directed node. Columns are resolved by binary search over the block's
//! sorted member list, so a block costs `O(block)` whatever its ids look like.

use std::{cmp::Reverse, collections::BinaryHeap};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};

use crate::{
    gfa_graph_extensions::GfaNodeDataExt,
    min_plus::{
        MinPlus,
        wide::{INF, add},
    },
    path::GfaPathLength,
};

/// The default cap on a row block's cuts
/// ([`LiftOptions::node_row_max_cuts`](crate::block_cut_lift::LiftOptions::node_row_max_cuts)):
/// pangenome blocks rarely have more than 2 cuts; the cap keeps a many-cut block (full rows
/// cost `16 c` bytes per node) on flip rows instead.
pub const DEFAULT_MAX_CUTS: usize = 8;

/// Flip-capable cuts a many-cut block may have for flip rows (`4F` entries per node); more flip
/// cuts than this leaves the block on the record path.
pub const FLIP_RESERVE_CUTS: usize = 2;

/// What a block's node rows hold (persisted as one byte per block by the lift).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RowKind {
    /// No rows: a block with no cuts or too many flip cuts, or rows turned off.
    None = 0,
    /// Full rows: `M_aug`-folded distances to every cut of the block.
    Full = 1,
    /// Flip rows: raw block distances to the block's flip-capable cuts.
    Flip = 2,
    /// Half rows, for a block that touches each of its cuts on one side: full rows with only each
    /// cut's "away" orientation kept (`2c` entries per node, `[o][j]`), the one leaving the block.
    /// A route from inside the block to a cut's other orientation must first pass the cut in the
    /// away orientation and flip, so that cell is `row + full[cut](away -> other)`, read on demand.
    Half = 3,
}

impl RowKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Full,
            2 => Self::Flip,
            3 => Self::Half,
            _ => Self::None,
        }
    }
}

/// A baked flip edge set at one cut of a flip-baked block: `(cut, w)` with `w[p][q]` the closure's
/// start-to-start distance `cut^p -> cut^q` (`u64::MAX` = none; orientation `0` = forward).
pub type BakedArc<IndexType> = (NodeIndex<IndexType>, [[u64; 2]; 2]);

/// Whether a block with `cut_count` cuts carries FULL node rows: `1..=row_max_cuts` cuts.
pub fn block_has_rows(cut_count: usize, row_max_cuts: usize) -> bool {
    row_max_cuts > 0 && cut_count >= 1 && cut_count <= row_max_cuts
}

/// Row entries per node of a block with `cut_count` cuts (full rows).
pub fn entries_per_node(cut_count: usize) -> usize {
    4 * cut_count
}

/// The flip rows of `row_nodes` (a subset of `members`, in `members` order): `4F` entries per node,
/// `[o][2 * i + p] = D_B(v^o -> flip_cuts[i]^p)`, the raw block-restricted distance (one block SSSP
/// per directed flip cut; `members` sorted, cut nodes included).
pub fn block_flip_rows<IndexType, NodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    members: &[NodeIndex<IndexType>],
    flip_cuts: &[NodeIndex<IndexType>],
    row_nodes: &[NodeIndex<IndexType>],
    baked: &[BakedArc<IndexType>],
) -> Vec<MinPlus<IndexType>>
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    let raw = block_raw_distances(graph, members, flip_cuts, baked);
    let columns = 2 * members.len();
    let dim = 2 * flip_cuts.len();
    let mut out = Vec::with_capacity(row_nodes.len() * 2 * dim);
    for &v in row_nodes {
        let k = members
            .binary_search(&v)
            .expect("a row node is a member of its block");
        for o in 0..2 {
            let x = 2 * k + o;
            for d in 0..dim {
                out.push(MinPlus::from_wide_clamped(raw[d * columns + x]));
            }
        }
    }
    out
}

/// Epoch-stamped Dijkstra over a reverse CSR over compact columns.
struct Sssp {
    dist: Vec<u64>,
    stamp: Vec<u32>,
    epoch: u32,
    heap: BinaryHeap<Reverse<(u64, u32)>>,
}

impl Sssp {
    fn new(n: usize) -> Self {
        Self {
            dist: vec![INF; n],
            stamp: vec![0; n],
            epoch: 0,
            heap: BinaryHeap::new(),
        }
    }

    /// Backward SSSP toward `source`: afterwards `get(x) = d_block(x -> source)`. The source's CSR
    /// neighbours (its original in-neighbours `u`) are seeded at `len(u)`; the source itself is not
    /// self-settled, so `get(source)` is a cycle length or `INF`.
    fn run(&mut self, off: &[u32], to: &[u32], w: &[u64], source: u32) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
        let e = self.epoch;
        self.heap.clear();
        let relax = |dist: &mut [u64],
                     stamp: &mut [u32],
                     heap: &mut BinaryHeap<Reverse<(u64, u32)>>,
                     v: u32,
                     nd: u64| {
            if stamp[v as usize] != e || nd < dist[v as usize] {
                stamp[v as usize] = e;
                dist[v as usize] = nd;
                heap.push(Reverse((nd, v)));
            }
        };
        for i in off[source as usize] as usize..off[source as usize + 1] as usize {
            relax(&mut self.dist, &mut self.stamp, &mut self.heap, to[i], w[i]);
        }
        while let Some(Reverse((d, u))) = self.heap.pop() {
            if self.stamp[u as usize] == e && d > self.dist[u as usize] {
                continue;
            }
            for i in off[u as usize] as usize..off[u as usize + 1] as usize {
                let nd = d.saturating_add(w[i]);
                relax(&mut self.dist, &mut self.stamp, &mut self.heap, to[i], nd);
            }
        }
    }

    #[inline]
    fn get(&self, v: u32) -> u64 {
        if self.stamp[v as usize] == self.epoch {
            self.dist[v as usize]
        } else {
            INF
        }
    }
}

/// `d_block(x -> cut^p)` for every directed member `x` (column `2 * rank + orientation`, members in
/// `members` order) and every directed cut (`2 * i + p`, `p = 0` forward), as `rows[d * columns +
/// x]`; `u64::MAX` = unreachable. `members` must be sorted by node id and contain the block's cut
/// nodes.
pub fn block_raw_distances<IndexType, NodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    members: &[NodeIndex<IndexType>],
    cuts: &[NodeIndex<IndexType>],
    baked: &[BakedArc<IndexType>],
) -> Vec<u64>
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    debug_assert!(
        members.windows(2).all(|w| w[0] < w[1]),
        "members sorted by node id"
    );
    let csr = ReverseCsr::of_block(graph, members, baked);
    let columns = 2 * members.len();
    let mut rows = vec![INF; 2 * cuts.len() * columns];
    let mut sssp = Sssp::new(columns);
    for (i, &cut) in cuts.iter().enumerate() {
        for p in 0..2 {
            let source =
                column(members, cut.into_directed(p == 0)).expect("a block's cut node is a member");
            sssp.run(&csr.off, &csr.to, &csr.w, source as u32);
            let row = &mut rows[(2 * i + p) * columns..(2 * i + p + 1) * columns];
            for (x, slot) in row.iter_mut().enumerate() {
                *slot = sssp.get(x as u32);
            }
        }
    }
    rows
}

/// The column of directed node `v` among the directed members (`2 * rank + orientation`), if it
/// is one.
fn column<IndexType: GraphIndexInteger>(
    members: &[NodeIndex<IndexType>],
    v: DirectedNodeIndex<IndexType>,
) -> Option<usize> {
    members
        .binary_search(&v.into_bidirected())
        .ok()
        .map(|k| 2 * k + usize::from(!v.is_forward()))
}

/// The reverse CSR of a block over its member columns: an arc `u -> v` of the block sits under `v`
/// carrying `(u, len(u))`; then the baked flip edges.
struct ReverseCsr {
    off: Vec<u32>,
    to: Vec<u32>,
    w: Vec<u64>,
}

impl ReverseCsr {
    fn of_block<IndexType, NodeData, EdgeData>(
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        members: &[NodeIndex<IndexType>],
        baked: &[BakedArc<IndexType>],
    ) -> Self
    where
        IndexType: GraphIndexInteger,
        NodeData: GfaNodeData,
        EdgeData: GfaEdgeData,
    {
        let columns = 2 * members.len();
        let baked_arcs = baked_arcs(members, baked);
        let mut off = vec![0u32; columns + 1];
        for_each_block_arc(graph, members, |_, to, _| off[to + 1] += 1);
        for &(_, to, _) in &baked_arcs {
            off[to + 1] += 1;
        }
        for i in 0..columns {
            off[i + 1] += off[i];
        }
        let total = off[columns] as usize;
        let mut csr = Self {
            to: vec![0u32; total],
            w: vec![0u64; total],
            off,
        };
        let mut cursor = csr.off.clone();
        let mut push = |from: usize, to: usize, weight: u64| {
            let slot = cursor[to] as usize;
            csr.to[slot] = from as u32;
            csr.w[slot] = weight;
            cursor[to] += 1;
        };
        for_each_block_arc(graph, members, &mut push);
        for &(from, to, weight) in &baked_arcs {
            push(from, to, weight);
        }
        csr
    }
}

/// Call `arc(from, to, len(from))` for every arc between two directed members (columns), members
/// in order.
fn for_each_block_arc<IndexType, NodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    members: &[NodeIndex<IndexType>],
    mut arc: impl FnMut(usize, usize, u64),
) where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    for (k, &node) in members.iter().enumerate() {
        let len: GfaPathLength<IndexType> = graph.node_data(node).len();
        for o in 0..2 {
            let u = node.into_directed(o == 0);
            debug_assert_eq!(column(members, u), Some(2 * k + o));
            for edge in graph.iter_outgoing_edges(u) {
                if let Some(to) = column(members, edge.to()) {
                    arc(2 * k + o, to, len.into_usize() as u64);
                }
            }
        }
    }
}

/// The baked flip edges of a flip-baked block (its child cuts' closures `down[z]`) as columns: `z^p
/// -> z^q` costs the closure's start-to-start distance, exactly as in the block's engine.
fn baked_arcs<IndexType: GraphIndexInteger>(
    members: &[NodeIndex<IndexType>],
    baked: &[BakedArc<IndexType>],
) -> Vec<(usize, usize, u64)> {
    baked
        .iter()
        .flat_map(|&(z, m)| (0..2).flat_map(move |p| (0..2).map(move |q| (z, p, q, m[p][q]))))
        .filter(|&(_, p, q, w)| p != q && w != INF)
        .map(|(z, p, q, w)| {
            let from = column(members, z.into_directed(p == 0))
                .expect("a baked cut is a member of its block");
            let to = column(members, z.into_directed(q == 0))
                .expect("a baked cut is a member of its block");
            (from, to, w)
        })
        .collect()
}

/// The cut rows of `row_nodes` (a subset of `members`, in `members` order): `4c` entries per node,
/// `[o][2 * j + q] = min over d of raw[d][v^o] + aug[d * 2c + 2 * j + q]`, where `aug` is the block's
/// row-major `M_aug` over its directed cuts (`cuts` order) and `raw` is [`block_raw_distances`]. A sum
/// at or above the index type's infinity is clamped one below it.
pub fn block_rows<IndexType, NodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    members: &[NodeIndex<IndexType>],
    cuts: &[NodeIndex<IndexType>],
    aug: &[MinPlus<IndexType>],
    row_nodes: &[NodeIndex<IndexType>],
) -> Vec<MinPlus<IndexType>>
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    let c = cuts.len();
    let dim = 2 * c;
    debug_assert_eq!(aug.len(), dim * dim, "M_aug is (2c)^2");
    let raw = block_raw_distances(graph, members, cuts, &[]);
    let columns = 2 * members.len();
    let aug_u64: Vec<u64> = aug.iter().map(|m| m.to_wide()).collect();
    let mut out = Vec::with_capacity(row_nodes.len() * entries_per_node(c));
    for &v in row_nodes {
        let k = members
            .binary_search(&v)
            .expect("a row node is a member of its block");
        for o in 0..2 {
            let x = 2 * k + o;
            for j in 0..c {
                for q in 0..2 {
                    let mut best = INF;
                    for d in 0..dim {
                        best = best.min(add(raw[d * columns + x], aug_u64[d * dim + 2 * j + q]));
                    }
                    out.push(MinPlus::from_wide_clamped(best));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use bidirected_adjacency_array::{
        graph::BidirectedAdjacencyArray,
        index::{DirectedNodeIndex, NodeIndex},
        io::gfa1::{PlainGfaEdgeData, PlainGfaNodeData},
    };

    use super::{block_raw_distances, block_rows};
    use crate::{block_dijkstra::BlockDijkstra, min_plus::MinPlus};

    type Ix = u16;
    type Graph = BidirectedAdjacencyArray<Ix, PlainGfaNodeData, PlainGfaEdgeData>;

    const GFA: &str = "H\tVN:Z:1.0\n\
        S\ta\tAAAA\n\
        S\tb\tCC\n\
        S\tc\tGGGGGG\n\
        S\td\tT\n\
        S\te\tAAG\n\
        L\ta\t+\tb\t+\t0M\n\
        L\tb\t+\tc\t+\t0M\n\
        L\ta\t+\tc\t-\t0M\n\
        L\tc\t+\td\t+\t0M\n\
        L\tb\t-\td\t+\t0M\n\
        L\td\t+\te\t+\t0M\n\
        L\te\t+\ta\t-\t0M\n";

    /// The raw rows must equal the block-restricted Dijkstra from every member to every directed cut,
    /// on a block in the middle of the id range (members `{b, c, d}`, cuts `{d, b}`).
    #[test]
    fn raw_distances_match_block_dijkstra() {
        let graph: Graph =
            BidirectedAdjacencyArray::read_gfa1(BufReader::new(GFA.as_bytes())).unwrap();
        let n = graph.node_count();
        let members: Vec<NodeIndex<Ix>> = (1..=3).map(NodeIndex::from_usize).collect();
        let in_block = |v: NodeIndex<Ix>| (1..=3).contains(&v.into_usize());
        let cuts: Vec<NodeIndex<Ix>> = [3usize, 1]
            .iter()
            .map(|&i| NodeIndex::from_usize(i))
            .collect();
        let raw = block_raw_distances(&graph, &members, &cuts, &[]);
        let columns = 2 * members.len();
        let mut bd = BlockDijkstra::new(2 * n);
        for (k, &node) in members.iter().enumerate() {
            for o in 0..2 {
                let x = node.into_directed(o == 0);
                bd.run(&graph, x, in_block);
                for (i, &cut) in cuts.iter().enumerate() {
                    for p in 0..2 {
                        let dir: DirectedNodeIndex<Ix> = cut.into_directed(p == 0);
                        let want = bd
                            .get(dir)
                            .map(|d| d.into_usize() as u64)
                            .unwrap_or(u64::MAX);
                        assert_eq!(
                            raw[(2 * i + p) * columns + 2 * k + o],
                            want,
                            "x={x:?} dir={dir:?}"
                        );
                    }
                }
            }
        }
    }

    /// The fold with an identity-like `M_aug` (zero diagonal, `∞` elsewhere) reproduces the raw rows.
    #[test]
    fn fold_with_identity_aug_is_the_raw_row() {
        let graph: Graph =
            BidirectedAdjacencyArray::read_gfa1(BufReader::new(GFA.as_bytes())).unwrap();
        let members: Vec<NodeIndex<Ix>> = (0..5).map(NodeIndex::from_usize).collect();
        let cuts: Vec<NodeIndex<Ix>> = vec![NodeIndex::from_usize(0), NodeIndex::from_usize(4)];
        let dim = 2 * cuts.len();
        let mut aug = vec![MinPlus::<Ix>::infinity(); dim * dim];
        for d in 0..dim {
            aug[d * dim + d] = MinPlus::zero();
        }
        let row_nodes: Vec<NodeIndex<Ix>> = (1..=3).map(NodeIndex::from_usize).collect();
        let rows = block_rows(&graph, &members, &cuts, &aug, &row_nodes);
        let raw = block_raw_distances(&graph, &members, &cuts, &[]);
        let columns = 2 * members.len();
        for (r, &v) in row_nodes.iter().enumerate() {
            let k = v.into_usize();
            for o in 0..2 {
                for d in 0..dim {
                    let got = rows[r * 2 * dim + o * dim + d]
                        .into_option()
                        .map(|x| x.into_usize() as u64);
                    let want = raw[d * columns + 2 * k + o];
                    assert_eq!(
                        got,
                        (want != u64::MAX).then_some(want),
                        "v={v:?} o={o} d={d}"
                    );
                }
            }
        }
    }
}
