//! The bottom-up inside DP of one block.
//!
//! For each non-root SPQR node (children first: deepest tree level first, a level across the pool)
//! it assembles the node's orientation-doubled directed skeleton and computes the `4x4` start-to-start
//! distances between its four reference-pole states *through its own subtree* (real skeleton edges
//! plus the children's matrices, excluding the parent reference edge). A P-node's matrix is the
//! closure of its four bond states, without a search. The flip closures of the block's child cuts
//! are baked into the skeletons that own them as same-vertex edges (`compute_with_flips`). The
//! skeletons the later stages search are assembled here too: [`InsideDp::build_skeleton_adjacency`]
//! and, with a node's `D_out`, [`InsideDp::lca_adjacency`].
//!
//! # Cost model
//!
//! Distances are **start-to-start**: traversing a directed edge `x -> y` costs `len(x)` (the node
//! you leave), and arriving at `y` adds nothing. This matches [`GfaDijkstra`](crate::dijkstra) at
//! offsets `0 -> 0` and composes associatively, which is what makes the matrix composition of the
//! inside and outside DPs sound.
//!
//! # Directed skeleton
//!
//! - A real link `u^a -> v^b` contributes the directed edge `u^a -> v^b` with weight `len(u)` and
//!   its reverse complement `v^{¬b} -> u^{¬a}` with weight `len(v)`.
//! - A child virtual edge with poles `(p, q)` and child matrix `C` (start-to-start `p -> q` through
//!   the child) contributes `p^{o} -> q^{o'}` with weight `C(o, o')`, and the reverse traversal
//!   `q -> p` via `rc(C)`.

use crate::rooted_tree::TreeAccess;
#[cfg(test)]
use std::{cmp::Reverse, collections::BinaryHeap};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use rayon::prelude::*;
use spqr_tree::decomposition::SPQRNodeType;

use crate::{
    block_spqr_tree::BlockSpqrTree,
    fast_hash::FastMap,
    gfa_graph_extensions::GfaNodeDataExt,
    min_plus::{Mat2, MinPlus, MinPlusMatrix},
    orientation::Orientation,
    path::GfaPathLength,
    pole_lift::{Pole, add_pole_state_edges, slot},
    skeleton::{DijkstraWorkspace, Skeleton, SkeletonAccess},
    skeleton_index::SkeletonLabels,
};

#[cfg(test)]
mod tests;

/// The adjacency of an orientation-doubled directed skeleton: for each directed node, its finite
/// weighted out-edges.
pub type SkeletonAdjacency<IndexType> = FastMap<
    DirectedNodeIndex<IndexType>,
    Vec<(DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>)>,
>;

/// The bottom-up inside DP for a single block's rooted SPQR tree.
pub struct InsideDp<'a, IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData: GfaEdgeData>
{
    graph: &'a BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    block_tree: BlockSpqrTree<IndexType>,
    /// Full 4x4 inside pole-state matrix per local SPQR node id; `None` for the root (which has no
    /// reference poles). Indexed by [`slot`]; includes pole self-flips through the subtree.
    inside: Vec<Option<MinPlusMatrix<IndexType, 4>>>,
    /// Flip-baking (`compute_with_flips`): the flip closures to inject into each local node's skeleton
    /// adjacency during assembly, one `(cut z, down[z])` per cut assigned to the SPQR node that owns it.
    /// Empty for the plain [`Self::compute`] path, so it is a no-op unless flip closures are supplied.
    /// Applied via [`crate::pole_lift::bake_flip_edge`] in [`Self::build_skeleton_adjacency`].
    flip_closures_per_node: Vec<Vec<(NodeIndex<IndexType>, Mat2<IndexType>)>>,
}

impl<'a, IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData: GfaEdgeData>
    InsideDp<'a, IndexType, NodeData, EdgeData>
{
    /// Run the bottom-up DP, computing the inside matrix of every non-root SPQR node.
    ///
    /// A node folds only its DIRECT children's inside matrices, and in a rooted tree a child is
    /// always exactly one level deeper — so all SPQR nodes at the *same* tree depth are mutually
    /// independent. We process depth levels **deepest-first, in parallel within each level** (one
    /// reusable [`DijkstraWorkspace`] per rayon worker), scattering each level's results into
    /// `inside` after its barrier. This is a valid topological order identical to the serial
    /// postorder (every child is a strictly deeper level, and the per-node min-plus fold is
    /// order-independent), so the result is byte-identical — it just runs the millions of tiny
    /// per-node skeleton searches concurrently instead of one at a time.
    pub fn compute(
        graph: &'a BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        block_tree: BlockSpqrTree<IndexType>,
    ) -> Self
    where
        IndexType: Send + Sync,
        NodeData: Sync,
        EdgeData: Sync,
    {
        let node_count = block_tree.node_count();
        let mut this = Self {
            graph,
            block_tree,
            inside: vec![None; node_count],
            flip_closures_per_node: Vec::new(),
        };
        this.run_dp();
        this
    }

    /// Like [`Self::compute`], but bakes each supplied flip closure into the skeleton of the SPQR node
    /// that owns its cut, so the resulting inside matrices (and everything composed from them — the
    /// labels' adjacency, `D_out`, queries) include the flip as an ordinary route. `closures` maps a
    /// cut segment `z` to `down[z]` (its pendant-local orientation-flip closure). A flip through
    /// `z` is then not assembled at query time (`intra(a->z) + down[z] + intra(z->b)`): it is an
    /// edge.
    ///
    /// Owner assignment is one pass over real skeleton edges: `z` is baked into the first SPQR node whose
    /// real edges touch `z` (any containing node is sound — the closure composes upward regardless). A
    /// cut with no containing real edge (pure virtual pole) falls back to its shallowest incident node.
    pub fn compute_with_flips(
        graph: &'a BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        block_tree: BlockSpqrTree<IndexType>,
        closures: &FastMap<NodeIndex<IndexType>, Mat2<IndexType>>,
    ) -> Self
    where
        IndexType: Send + Sync,
        NodeData: Sync,
        EdgeData: Sync,
    {
        let node_count = block_tree.node_count();
        let flip_closures_per_node = Self::assign_flip_closures(graph, &block_tree, closures);
        // Every supplied closure must be baked: the query does not take a baked cut's flip
        // explicitly, so a closure left out would surface as a too-long distance. A cut of the
        // block always has a real edge in some skeleton of it, so this is a hard invariant, not a
        // heuristic.
        let assigned: usize = flip_closures_per_node.iter().map(Vec::len).sum();
        assert_eq!(
            assigned,
            closures.len(),
            "flip baking: every child-cut closure must be owned by a skeleton of its block"
        );
        let mut this = Self {
            graph,
            block_tree,
            inside: vec![None; node_count],
            flip_closures_per_node,
        };
        this.run_dp();
        this
    }

    /// Assign each cut's flip closure to an owning SPQR node (one that contains the cut as a real
    /// skeleton vertex). One pass over every node's real directed edges; `O(block edges)`. Shallowest
    /// node first, so a cut is baked into the highest node that contains it (fewest composition levels
    /// to reach the root/oracle adjacency); any containing node would be sound.
    fn assign_flip_closures(
        graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        block_tree: &BlockSpqrTree<IndexType>,
        closures: &FastMap<NodeIndex<IndexType>, Mat2<IndexType>>,
    ) -> Vec<Vec<(NodeIndex<IndexType>, Mat2<IndexType>)>> {
        let node_count = block_tree.node_count();
        let mut per_node = vec![Vec::new(); node_count];
        if closures.is_empty() {
            return per_node;
        }
        let mut assigned: FastMap<NodeIndex<IndexType>, ()> = FastMap::default();
        let mut order: Vec<usize> = (0..node_count).collect();
        order.sort_by_key(|&local| (block_tree.tree().depth(local), local));
        for local in order {
            for edge in block_tree.real_directed_edges(local, graph) {
                for endpoint in [edge.from.into_bidirected(), edge.to.into_bidirected()] {
                    if closures.contains_key(&endpoint) && !assigned.contains_key(&endpoint) {
                        assigned.insert(endpoint, ());
                        per_node[local].push((endpoint, closures[&endpoint]));
                    }
                }
            }
        }
        per_node
    }

    /// The shared bottom-up depth-level DP driver used by [`Self::compute`] and
    /// [`Self::compute_with_flips`]. Deepest level first, parallel within a level above a threshold.
    fn run_dp(&mut self)
    where
        IndexType: Send + Sync,
        NodeData: Sync,
        EdgeData: Sync,
    {
        let node_count = self.block_tree.node_count();
        // Bucket every node by its tree depth (owned, so the level loop can mutate `self.inside`).
        let mut nodes_by_depth: Vec<Vec<usize>> = Vec::new();
        {
            let tree = self.block_tree.tree();
            for local in 0..node_count {
                let depth = tree.depth(local);
                if depth >= nodes_by_depth.len() {
                    nodes_by_depth.resize(depth + 1, Vec::new());
                }
                nodes_by_depth[depth].push(local);
            }
        }

        // Deepest level first: when a node is processed, all of its children (one level deeper) are
        // already in `self.inside`. Writes within a level are disjoint (distinct locals), and the
        // par_iter borrows `&self` immutably then the scatter borrows `self.inside` mutably — the two
        // are sequential per level, so there is no aliasing.
        // A rayon fan-out is only worth its dispatch cost (plus the `collect` allocation) on a big
        // block. On a block-cut graph of ~10^6 tiny blocks a level is 1-2 nodes wide, so the par_iter
        // pays a dispatch per level per block — measured as idle workers (~11% of build samples).
        // Serial levels reuse one workspace and write straight through: nodes within a level never
        // read each other (they depend only on the deeper level), so immediate writes are safe.
        const PARALLEL_LEVEL_THRESHOLD: usize = 64;
        let mut serial_workspace = DijkstraWorkspace::new();
        for level in nodes_by_depth.iter().rev() {
            if level.len() >= PARALLEL_LEVEL_THRESHOLD {
                let computed: Vec<Option<(usize, MinPlusMatrix<IndexType, 4>)>> = level
                    .par_iter()
                    .map_init(DijkstraWorkspace::new, |workspace, &local| {
                        self.node_inside_matrix(local, workspace)
                            .map(|matrix| (local, matrix))
                    })
                    .collect();
                for entry in computed.into_iter().flatten() {
                    self.inside[entry.0] = Some(entry.1);
                }
            } else {
                for &local in level {
                    let computed = self.node_inside_matrix(local, &mut serial_workspace);
                    if let Some(matrix) = computed {
                        self.inside[local] = Some(matrix);
                    }
                }
            }
        }
    }

    /// The inside `4x4` matrix of one SPQR node, folding its direct children's already-computed
    /// inside matrices, or `None` for the root (no reference poles). Reads only `self.inside[child]`
    /// for DIRECT children, so it is safe to call concurrently across same-depth nodes.
    fn node_inside_matrix(
        &self,
        local: usize,
        workspace: &mut DijkstraWorkspace<IndexType>,
    ) -> Option<MinPlusMatrix<IndexType, 4>> {
        let (s, t) = self.block_tree.node(local).reference_poles()?;

        // A P-node (bond) skeleton is just the two poles, so its inside matrix is a direct
        // `4x4` closure — no adjacency map / CSR / Dijkstra. S- and R-nodes keep the general search.
        let closed_form = if self.block_tree.node(local).node_type() == SPQRNodeType::PNode {
            self.p_node_matrix(local, s, t, None)
        } else {
            None
        };
        Some(match closed_form {
            Some(closed) => {
                // Debug builds cross-check the closed form against the general search on every
                // P-node of every test graph — the closed form must be an exact substitute.
                #[cfg(debug_assertions)]
                {
                    let adjacency = self.build_skeleton_adjacency(local);
                    let skeleton = Skeleton::from_adjacency(&adjacency);
                    let search = pole_state_matrix(&skeleton, workspace, (s, t), (s, t));
                    debug_assert_eq!(
                        closed, search,
                        "P-node closed form disagrees with Dijkstra at local {local}"
                    );
                }
                closed
            }
            None => {
                let adjacency = self.build_skeleton_adjacency(local);
                let skeleton = Skeleton::from_adjacency(&adjacency);
                // Inside distances: source and target pole segments are both this node's poles.
                pole_state_matrix(&skeleton, workspace, (s, t), (s, t))
            }
        })
    }

    /// Serial postorder reference build (test-only), to assert the parallel depth-level driver in
    /// [`Self::compute`] is byte-identical. Reuses [`Self::node_inside_matrix`], so it isolates the
    /// only thing that changed — the scheduling (level-parallel vs one-node-at-a-time postorder).
    #[cfg(test)]
    fn compute_serial(
        graph: &'a BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
        block_tree: BlockSpqrTree<IndexType>,
    ) -> Self {
        let node_count = block_tree.node_count();
        let mut this = Self {
            graph,
            block_tree,
            inside: vec![None; node_count],
            flip_closures_per_node: Vec::new(),
        };
        let postorder: Vec<usize> = this.block_tree.tree().postorder().to_vec();
        let mut workspace = DijkstraWorkspace::new();
        for local in postorder {
            if let Some(matrix) = this.node_inside_matrix(local, &mut workspace) {
                this.inside[local] = Some(matrix);
            }
        }
        this
    }

    /// The full 4x4 inside pole-state matrix of a non-root SPQR node, or `None` for the root.
    #[cfg(test)]
    pub fn inside_matrix(&self, local: usize) -> Option<&MinPlusMatrix<IndexType, 4>> {
        self.inside[local].as_ref()
    }

    /// The rooted block SPQR tree this DP was computed for.
    pub fn block_tree(&self) -> &BlockSpqrTree<IndexType> {
        &self.block_tree
    }

    /// Release the per-node inside `4x4` matrices. They are consumed during the build (assembling
    /// skeletons, labels, the outside/pole-lift DPs) and are never read at query time — which needs
    /// only `block_tree` and `graph` — so freeing them after the build reclaims their memory
    /// (`~128 B` per SPQR node, hundreds of MiB on the giant block). Every node's inside matrix is
    /// gone afterwards, so this must run after all build-time consumers.
    pub fn clear_matrices(&mut self) {
        self.inside = Vec::new();
    }

    /// The graph this DP was computed for.
    pub fn graph(&self) -> &'a BidirectedAdjacencyArray<IndexType, NodeData, EdgeData> {
        self.graph
    }

    /// Closed-form pole-state matrix for a **P-node** (bond). A bond has exactly two vertices
    /// `v0`, `v1`; every real edge and every child virtual edge connects them, so all skeleton arcs
    /// lie among the four `(v0/v1, orientation)` states and the all-pairs matrix is simply their `4x4`
    /// min-plus closure — computed directly (accumulate arcs into a `[4x4]`, then Floyd–Warshall) with
    /// **no adjacency map, CSR, or Dijkstra**.
    ///
    /// - Inside DP: `(v0, v1)` are the node's own reference poles and `out_fold` is `None`.
    /// - Outside DP: `(v0, v1)` are the P-node *parent*'s bond vertices and `out_fold` is the parent's
    ///   own outside matrix (folded in as arcs, matching the augmented skeleton) — the result is the
    ///   augmented all-pairs matrix, identical for every child of that parent.
    ///
    /// Returns `None` (caller falls back to the general search) if any endpoint is unexpectedly not a
    /// bond vertex, which a true bond never has.
    pub(crate) fn p_node_matrix(
        &self,
        local: usize,
        v0: NodeIndex<IndexType>,
        v1: NodeIndex<IndexType>,
        out_fold: Option<&MinPlusMatrix<IndexType, 4>>,
    ) -> Option<MinPlusMatrix<IndexType, 4>> {
        let bond = BondSlots { v0, v1 };
        let mut base = MinPlusMatrix::<IndexType, 4>::identity();
        // Real links: forward traversal plus its reverse complement (same arcs as
        // `build_skeleton_adjacency`).
        for edge in self.block_tree.real_directed_edges(local, self.graph) {
            let len_from = self.graph.node_data(edge.from.into_bidirected()).len();
            let len_to = self.graph.node_data(edge.to.into_bidirected()).len();
            base.relax(
                bond.slot(edge.from)?,
                bond.slot(edge.to)?,
                MinPlus::finite(len_from),
            );
            base.relax(
                bond.slot(edge.to.invert())?,
                bond.slot(edge.from.invert())?,
                MinPlus::finite(len_to),
            );
        }
        // Child virtual edges: fold each child's `4x4` inside relation into the bond-vertex slots.
        for child in self.block_tree.node(local).child_virtual_edges() {
            let matrix = self.inside[child.child_local]
                .as_ref()
                .expect("child inside matrix must be computed before its parent (postorder)");
            let slots = bond.pole_slots(child.poles)?;
            for i in 0..4 {
                for j in 0..4 {
                    base.relax(slots[i], slots[j], matrix.get(i, j));
                }
            }
        }
        // Baked flip edges owned by this bond (flip-baking build): `z^a -> z^b` at a bond vertex, the
        // same edges `build_skeleton_adjacency` adds, so the closed form stays an exact substitute.
        if let Some(closures) = self.flip_closures_per_node.get(local) {
            for (cut, closure) in closures {
                for from in Orientation::all() {
                    for to in Orientation::all() {
                        if let Some(weight) = closure.get(from.index(), to.index()).into_option() {
                            base.relax(
                                bond.slot(cut.into_directed(from.is_forward()))?,
                                bond.slot(cut.into_directed(to.is_forward()))?,
                                MinPlus::finite(weight),
                            );
                        }
                    }
                }
            }
        }
        // Outside DP: fold the parent's own outside matrix (indexed by the same `(v0, v1)` slots).
        if let Some(out) = out_fold {
            for i in 0..4 {
                for j in 0..4 {
                    base.relax(i, j, out.get(i, j));
                }
            }
        }
        Some(base.closure())
    }

    /// Assemble the orientation-doubled directed skeleton of `local`.
    ///
    /// Uses the node's real skeleton edges and its child virtual edges (weighted by the children's
    /// already-computed inside matrices). The parent reference edge is excluded by construction (it
    /// is neither a real edge nor a child virtual edge of this node).
    pub fn build_skeleton_adjacency(&self, local: usize) -> SkeletonAdjacency<IndexType> {
        let mut adjacency: SkeletonAdjacency<IndexType> = FastMap::default();

        // Real links: forward traversal plus its reverse complement, each costing the node left.
        for edge in self.block_tree.real_directed_edges(local, self.graph) {
            let len_from = self.graph.node_data(edge.from.into_bidirected()).len();
            let len_to = self.graph.node_data(edge.to.into_bidirected()).len();
            adjacency
                .entry(edge.from)
                .or_default()
                .push((edge.to, len_from));
            adjacency
                .entry(edge.to.invert())
                .or_default()
                .push((edge.from.invert(), len_to));
        }

        // Child virtual edges: fold in the child's full 4x4 inside relation (both directions and
        // pole self-flips through the child).
        for child in self.block_tree.node(local).child_virtual_edges() {
            let (p, q) = child.poles;
            let matrix = self.inside[child.child_local]
                .as_ref()
                .expect("child inside matrix must be computed before its parent (postorder)");
            add_pole_state_edges(&mut adjacency, p, q, matrix);
        }

        // Baked flip edges (`compute_with_flips`): the pendant flip closures of the cuts this SPQR node
        // owns, as same-vertex `z^a -> z^b` edges. Empty for the plain `compute` path.
        if let Some(closures) = self.flip_closures_per_node.get(local) {
            for (cut, closure) in closures {
                crate::pole_lift::bake_flip_edge(&mut adjacency, *cut, closure);
            }
        }

        adjacency
    }

    /// The LCA-role skeleton adjacency of `local`: its skeleton ([`Self::build_skeleton_adjacency`])
    /// plus its outside matrix `outside` (`D_out`) between its reference-pole states, the routes that
    /// leave its subtree. The outside DP searches it as a parent, a query as the LCA; the hub labels
    /// of a node are built over it.
    pub fn lca_adjacency(
        &self,
        local: usize,
        outside: &MinPlusMatrix<IndexType, 4>,
    ) -> SkeletonAdjacency<IndexType> {
        let mut adjacency = self.build_skeleton_adjacency(local);
        self.add_outside_edges(&mut adjacency, local, outside);
        adjacency
    }

    /// Add `local`'s outside matrix `outside` to a skeleton adjacency as edges between its
    /// reference-pole states (nothing at the root).
    pub fn add_outside_edges(
        &self,
        adjacency: &mut SkeletonAdjacency<IndexType>,
        local: usize,
        outside: &MinPlusMatrix<IndexType, 4>,
    ) {
        if let Some((s, t)) = self.block_tree.node(local).reference_poles() {
            add_pole_state_edges(adjacency, s, t, outside);
        }
    }

    /// The hub labels of a skeleton adjacency of this block.
    pub fn skeleton_labels(
        &self,
        adjacency: &SkeletonAdjacency<IndexType>,
    ) -> SkeletonLabels<IndexType> {
        SkeletonLabels::build_bidirected(adjacency, |d: DirectedNodeIndex<IndexType>| {
            self.sequence_len(d.into_bidirected())
        })
    }

    /// The length of a node's sequence (what `GfaNodeDataExt::len` returns, read directly to avoid
    /// the `IndexType`-ambiguous extension trait).
    pub fn sequence_len(&self, node: NodeIndex<IndexType>) -> u64 {
        self.graph.node_data(node).sequence().len() as u64
    }

    /// Shortest start-to-start distances from `source` to every reachable directed node within the
    /// assembled skeleton of `local`, for differential testing.
    #[cfg(test)]
    pub fn skeleton_shortest_paths(
        &self,
        local: usize,
        source: DirectedNodeIndex<IndexType>,
    ) -> FastMap<DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>> {
        dijkstra(&self.build_skeleton_adjacency(local), source)
    }
}

/// The four states of a bond (P-node) skeleton: vertex `v0` at slots `0..2`, `v1` at `2..4`, forward
/// before reverse, matching [`slot`] over `(v0, v1)`.
#[derive(Clone, Copy)]
pub(crate) struct BondSlots<IndexType: GraphIndexInteger> {
    pub(crate) v0: NodeIndex<IndexType>,
    pub(crate) v1: NodeIndex<IndexType>,
}

impl<IndexType: GraphIndexInteger> BondSlots<IndexType> {
    /// The slot of a directed node, `None` unless it is a bond vertex.
    pub(crate) fn slot(&self, v: DirectedNodeIndex<IndexType>) -> Option<usize> {
        let bidirected = v.into_bidirected();
        let pole_base = if bidirected == self.v0 {
            0
        } else if bidirected == self.v1 {
            2
        } else {
            return None;
        };
        Some(pole_base + if v.is_forward() { 0 } else { 1 })
    }

    /// The bond slot of each pole state of the pole pair `(p, q)`, in [`slot`] order; `None` unless
    /// both poles are bond vertices.
    pub(crate) fn pole_slots(
        &self,
        (p, q): (NodeIndex<IndexType>, NodeIndex<IndexType>),
    ) -> Option<[usize; 4]> {
        let mut slots = [0; 4];
        for pole in Pole::all() {
            let segment = if pole == Pole::Source { p } else { q };
            for orientation in Orientation::all() {
                slots[slot(pole, orientation)] =
                    self.slot(segment.into_directed(orientation.is_forward()))?;
            }
        }
        Some(slots)
    }
}

/// Fill a `4x4` pole-state min-plus matrix by running one dense-workspace Dijkstra per source
/// pole-state over `skeleton` and reading each target pole-state.
///
/// Rows are indexed by `slot(from_pole, orientation)` over `from_poles = (s, t)`, columns by
/// `slot(to_pole, orientation)` over `to_poles`. This is the shared inner kernel of the inside,
/// outside, and pole-lift phases: they differ only in which skeleton they pass and whether the
/// source/target pole segments coincide. Building the skeleton once and reusing an epoch-reset dense
/// workspace here (instead of a fresh `HashMap` Dijkstra per call) is what keeps the whole-block
/// build near-linear.
pub(crate) fn pole_state_matrix<IndexType: GraphIndexInteger>(
    skeleton: &Skeleton<IndexType>,
    workspace: &mut DijkstraWorkspace<IndexType>,
    from_poles: (NodeIndex<IndexType>, NodeIndex<IndexType>),
    to_poles: (NodeIndex<IndexType>, NodeIndex<IndexType>),
) -> MinPlusMatrix<IndexType, 4> {
    let from_segment = |pole: Pole| {
        if pole == Pole::Source {
            from_poles.0
        } else {
            from_poles.1
        }
    };
    let to_segment = |pole: Pole| {
        if pole == Pole::Source {
            to_poles.0
        } else {
            to_poles.1
        }
    };
    let mut matrix = MinPlusMatrix::<IndexType, 4>::infinity();
    for from_pole in Pole::all() {
        for from_orientation in Orientation::all() {
            let from_slot = slot(from_pole, from_orientation);
            let source = from_segment(from_pole).into_directed(from_orientation.is_forward());
            let Some(source_id) = skeleton.node_id(source) else {
                // Isolated source: a `HashMap` Dijkstra seeded here reaches only the source itself
                // at distance 0, so record 0 for any target that is the same directed node.
                for to_pole in Pole::all() {
                    for to_orientation in Orientation::all() {
                        let target = to_segment(to_pole).into_directed(to_orientation.is_forward());
                        if target == source {
                            matrix.set(
                                from_slot,
                                slot(to_pole, to_orientation),
                                MinPlus::finite(GfaPathLength::from_usize(0)),
                            );
                        }
                    }
                }
                continue;
            };
            workspace.run(skeleton, &[(source_id, GfaPathLength::from_usize(0))]);
            for to_pole in Pole::all() {
                for to_orientation in Orientation::all() {
                    let target = to_segment(to_pole).into_directed(to_orientation.is_forward());
                    if let Some(distance) =
                        skeleton.node_id(target).and_then(|id| workspace.get(id))
                    {
                        matrix.set(
                            from_slot,
                            slot(to_pole, to_orientation),
                            MinPlus::finite(distance),
                        );
                    }
                }
            }
        }
    }
    matrix
}

/// The edge-reversed (transposed) adjacency: `D(a -> b)` in `adjacency` equals `D(b -> a)` in the
/// result. Used to answer "distance from many sources to a few fixed targets" with one Dijkstra per
/// target (seeded in the transposed graph) instead of one per source.
pub(crate) fn transpose_adjacency<IndexType: GraphIndexInteger>(
    adjacency: &SkeletonAdjacency<IndexType>,
) -> SkeletonAdjacency<IndexType> {
    let mut transposed: SkeletonAdjacency<IndexType> = FastMap::default();
    for (&from, edges) in adjacency {
        for &(to, weight) in edges {
            transposed.entry(to).or_default().push((from, weight));
        }
    }
    transposed
}

/// A plain Dijkstra over a finite-weight directed adjacency, returning distances to reachable nodes.
#[cfg(test)]
fn dijkstra<IndexType: GraphIndexInteger>(
    adjacency: &SkeletonAdjacency<IndexType>,
    source: DirectedNodeIndex<IndexType>,
) -> FastMap<DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>> {
    multi_source_dijkstra(adjacency, &[(source, GfaPathLength::from_usize(0))])
}

/// Multi-source Dijkstra over a finite-weight directed adjacency.
#[cfg(test)]
pub(crate) fn multi_source_dijkstra<IndexType: GraphIndexInteger>(
    adjacency: &SkeletonAdjacency<IndexType>,
    seeds: &[(DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>)],
) -> FastMap<DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>> {
    let mut distances: FastMap<DirectedNodeIndex<IndexType>, GfaPathLength<IndexType>> =
        FastMap::default();
    let mut heap = BinaryHeap::new();
    for &(node, distance) in seeds {
        if distances
            .get(&node)
            .is_none_or(|&current| distance < current)
        {
            distances.insert(node, distance);
            heap.push(Reverse((distance, node)));
        }
    }

    while let Some(Reverse((distance, node))) = heap.pop() {
        if distances.get(&node).is_some_and(|&best| distance > best) {
            continue; // stale heap entry
        }
        let Some(edges) = adjacency.get(&node) else {
            continue;
        };
        for &(neighbour, weight) in edges {
            let candidate = distance + weight;
            let improved = distances
                .get(&neighbour)
                .is_none_or(|&current| candidate < current);
            if improved {
                distances.insert(neighbour, candidate);
                heap.push(Reverse((candidate, neighbour)));
            }
        }
    }

    distances
}
