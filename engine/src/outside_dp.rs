//! The top-down outside DP (`D_out`) of one block.
//!
//! For each non-root SPQR node `C` with reference poles `(s_C, t_C)` we compute the **full outside
//! relation**: the shortest distance between any two of `C`'s four pole-states
//! `{s_C^+, s_C^-, t_C^+, t_C^-}` that avoids `C`'s subtree interior (i.e. travels through the rest
//! of the block, and recursively the rest of the graph). Unlike the undirected engine, this must
//! include **pole self-flips** such as `s_C^+ -> s_C^-`: in a bidirected graph a pole can flip
//! orientation by leaving the subtree and returning.
//!
//! It is stored as a `4x4` min-plus matrix indexed by [`Pole`] x [`Orientation`].
//!
//! # Computation
//!
//! `D_out(C)` is the all-pairs distance among `C`'s pole states in the parent's augmented skeleton:
//! its skeleton (real edges + child virtual edges, `C`'s own included) plus the parent's own
//! `D_out` at its reference poles ([`InsideDp::lca_adjacency`]). Routes that stay in the parent's
//! subtree are found directly; routes that escape further out use `D_out(parent)`. The root's
//! outside is `∞`. Parents are processed top-down, each filling all its children from one
//! augmented skeleton, by the first method that applies (`TopDown::fill_children`):
//!
//! - a P-node below the label threshold: the `4x4` closure of its bond states;
//! - an S-node: the cycle closed form ([`crate::s_node_outside`]), `O(k)` for all children;
//! - a skeleton at the label threshold (see [`crate::block_engine::Tiers`]): hub labels over
//!   it, read for every child and kept as the node's LCA-role oracle;
//! - otherwise one Dijkstra per distinct child pole state, across the pool for a parent with many.

use crate::skeleton_index::LabelsAccess;
use bidirected_adjacency_array::{
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use rayon::prelude::*;
use spqr_tree::decomposition::SPQRNodeType;

use crate::{
    block_engine::Tiers,
    block_spqr_tree::BlockSpqrTree,
    fast_hash::FastMap,
    inside_dp::{BondSlots, InsideDp, SkeletonAdjacency},
    min_plus::{MinPlus, MinPlusMatrix},
    orientation::Orientation,
    path::GfaPathLength,
    pole_lift::{Pole, slot},
    s_node_outside::cycle_outside_matrices,
    skeleton::{DijkstraWorkspace, Skeleton, SkeletonAccess},
    skeleton_index::SkeletonLabels,
};

#[cfg(test)]
mod tests;

/// The top-down outside DP for a single block's rooted SPQR tree.
pub struct OutsideDp<IndexType: GraphIndexInteger> {
    /// Full outside matrix per local SPQR node id; the root's is `∞` (unused).
    outside: Vec<MinPlusMatrix<IndexType, 4>>,
}

/// Per SPQR node: the hub labels the outside DP built over its augmented skeleton, or `None`. Boxed
/// so an all-`None` vector costs one pointer per node (millions of label-less blocks).
pub type NodeLabels<IndexType> = Vec<Option<Box<SkeletonLabels<IndexType>>>>;

/// A parent with at least this many child source states runs their searches across the pool.
const PARALLEL_SOURCE_THRESHOLD: usize = 1024;

/// One outside entry from a source pole state: `(child, from slot, to slot, distance)`.
type OutsideEntry<IndexType> = (usize, usize, usize, GfaPathLength<IndexType>);

impl<IndexType: GraphIndexInteger> OutsideDp<IndexType> {
    /// Run the top-down DP on top of a completed [`InsideDp`] (no label acceleration).
    #[cfg(test)]
    pub fn compute<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        inside: &InsideDp<'_, IndexType, NodeData, EdgeData>,
    ) -> Self
    where
        IndexType: Send + Sync,
    {
        let no_labels = Tiers {
            label_min: usize::MAX,
            ..Tiers::default()
        };
        Self::compute_with_labels(inside, &no_labels).0
    }

    /// Run the top-down DP on top of a completed [`InsideDp`]. A parent whose augmented skeleton is
    /// at the label tier of `tiers` (the giant R-node, big S-cycles) builds a hub-labeling oracle
    /// once and answers its children's outside entries by label lookups instead of one
    /// block-restricted Dijkstra per source pole-state (the dominant build cost). The built labels
    /// are returned so the query engine can reuse them: a parent's augmented skeleton is exactly the
    /// skeleton it uses when it is the LCA, so no second label build is needed. A `label_min` of
    /// `usize::MAX` disables labeling entirely.
    pub fn compute_with_labels<NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
        inside: &InsideDp<'_, IndexType, NodeData, EdgeData>,
        tiers: &Tiers,
    ) -> (Self, NodeLabels<IndexType>)
    where
        IndexType: Send + Sync,
    {
        let tree = inside.block_tree().tree();
        let mut pass = TopDown::new(inside, *tiers);
        // Preorder (parents before children) is the reverse of the postorder. Each parent fills all
        // its children from its augmented skeleton, built once; `outside[parent]` is final by then,
        // since its own parent came earlier.
        for &parent in tree.postorder().iter().rev() {
            if !tree.children(parent).is_empty() {
                pass.fill_children(parent);
            }
        }
        pass.finish()
    }

    /// The full outside matrix of an SPQR node (the root's is `∞`).
    pub fn outside_matrix(&self, local: usize) -> &MinPlusMatrix<IndexType, 4> {
        &self.outside[local]
    }
}

/// The top-down pass over one block: each parent fills its children's outside matrices from its
/// augmented skeleton (inner skeleton plus its own `D_out` at its reference poles).
struct TopDown<'i, 'g, IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData: GfaEdgeData> {
    inside: &'i InsideDp<'g, IndexType, NodeData, EdgeData>,
    tiers: Tiers,
    outside: Vec<MinPlusMatrix<IndexType, 4>>,
    labels: NodeLabels<IndexType>,
    workspace: DijkstraWorkspace<IndexType>,
}

impl<'i, 'g, IndexType, NodeData, EdgeData> TopDown<'i, 'g, IndexType, NodeData, EdgeData>
where
    IndexType: GraphIndexInteger + Send + Sync,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    fn new(inside: &'i InsideDp<'g, IndexType, NodeData, EdgeData>, tiers: Tiers) -> Self {
        let node_count = inside.block_tree().node_count();
        Self {
            inside,
            tiers,
            outside: vec![MinPlusMatrix::infinity(); node_count],
            labels: (0..node_count).map(|_| None).collect(),
            workspace: DijkstraWorkspace::new(),
        }
    }

    fn finish(self) -> (OutsideDp<IndexType>, NodeLabels<IndexType>) {
        (
            OutsideDp {
                outside: self.outside,
            },
            self.labels,
        )
    }

    fn block_tree(&self) -> &'i BlockSpqrTree<IndexType> {
        self.inside.block_tree()
    }

    /// Fill the outside matrices of `parent`'s children, by the first method that applies: the bond
    /// closed form (a P-node below the label threshold), the cycle closed form (an S-node that will
    /// carry a segment tree, whatever its size), hub labels (a skeleton at the label threshold), the
    /// cycle closed form (any other S-node), else one search per child source state. A closed form
    /// that finds no clean bond or cycle falls through to the next method.
    fn fill_children(&mut self, parent: usize) {
        let node_type = self.block_tree().node(parent).node_type();
        if node_type == SPQRNodeType::PNode && self.tiers.bond_unlabeled() && self.bond_fill(parent)
        {
            return;
        }
        let adjacency = self.inside.lca_adjacency(parent, &self.outside[parent]);
        let skeleton = Skeleton::from_adjacency(&adjacency);
        let states = skeleton.node_count();
        if self.tiers.segtree(node_type, states) && self.cycle_fill(parent, &adjacency) {
            return;
        }
        if self.tiers.labels(states) {
            self.label_fill(parent, &adjacency);
            return;
        }
        if node_type == SPQRNodeType::SNode && self.cycle_fill(parent, &adjacency) {
            return;
        }
        self.search_fill(parent, &skeleton);
    }

    /// P-node parent: the augmented skeleton is the two bond vertices (4 states), so every child's
    /// outside is the same `4x4` closure (real edges, child folds, the parent's own outside) mapped
    /// onto the child's poles, without a search. `false` (children so far written, the next method
    /// overwrites them) if a pole is not a bond vertex, which a true bond never has.
    fn bond_fill(&mut self, parent: usize) -> bool {
        let block_tree = self.block_tree();
        let node = block_tree.node(parent);
        let reference = node.reference_poles();
        // Bond vertices: the parent's reference poles, or (for the root) any child's poles.
        let Some(bond) = reference.or_else(|| node.child_virtual_edges().first().map(|c| c.poles))
        else {
            return false;
        };
        // Fold the parent's own outside only when it has reference poles (the root has none).
        let out_parent = reference.map(|_| self.outside[parent]);
        let Some(augmented) =
            self.inside
                .p_node_matrix(parent, bond.0, bond.1, out_parent.as_ref())
        else {
            return false;
        };
        #[cfg(debug_assertions)]
        let debug_skeleton =
            Skeleton::from_adjacency(&self.inside.lca_adjacency(parent, &self.outside[parent]));
        for &child in block_tree.tree().children(parent) {
            let Some(poles) = block_tree.node(child).reference_poles() else {
                continue;
            };
            let Some(matrix) = bond_child_matrix(
                &augmented,
                BondSlots {
                    v0: bond.0,
                    v1: bond.1,
                },
                poles,
            ) else {
                return false;
            };
            #[cfg(debug_assertions)]
            debug_assert_eq!(
                matrix,
                crate::inside_dp::pole_state_matrix(
                    &debug_skeleton,
                    &mut self.workspace,
                    poles,
                    poles
                ),
                "P-node parent outside closed form mismatch: parent {parent}, child {child}"
            );
            self.outside[child] = matrix;
        }
        true
    }

    /// S-node parent: every child's outside in `O(k)` total by cutting the cycle at the child's
    /// edge (prefix/suffix series products) instead of ~`2k` searches over the `2k`-state cycle.
    /// `false` (nothing written) if the skeleton is not one clean cycle.
    fn cycle_fill(&mut self, parent: usize, adjacency: &SkeletonAdjacency<IndexType>) -> bool {
        let block_tree = self.block_tree();
        let (posed, poles): (Vec<usize>, Vec<_>) = block_tree
            .tree()
            .children(parent)
            .iter()
            .filter_map(|&child| block_tree.node(child).reference_poles().map(|p| (child, p)))
            .unzip();
        let Some(matrices) = cycle_outside_matrices(adjacency, &poles) else {
            return false;
        };
        #[cfg(debug_assertions)]
        {
            let skeleton = Skeleton::from_adjacency(adjacency);
            for ((&child, &p), matrix) in posed.iter().zip(&poles).zip(&matrices) {
                debug_assert_eq!(
                    *matrix,
                    crate::inside_dp::pole_state_matrix(&skeleton, &mut self.workspace, p, p),
                    "S-node parent outside closed form mismatch: parent {parent}, child {child}"
                );
            }
        }
        for (&child, matrix) in posed.iter().zip(matrices) {
            self.outside[child] = matrix;
        }
        true
    }

    /// Large parent (the giant R-node, big S-cycles): hub labels over the augmented skeleton, built
    /// once, answer every child's outside entries by lookup instead of one search per source state.
    /// The labels are kept: they answer this node's LCA-role searches.
    fn label_fill(&mut self, parent: usize, adjacency: &SkeletonAdjacency<IndexType>) {
        let block_tree = self.block_tree();
        let children = block_tree.tree().children(parent);
        let labels = self.inside.skeleton_labels(adjacency);
        for &child in children {
            // The root has no reference poles and no outside to compute.
            if let Some(poles) = block_tree.node(child).reference_poles() {
                set_label_entries(&mut self.outside[child], &labels, poles);
            }
        }
        self.labels[parent] = Some(Box::new(labels));
    }

    /// Small parent: one search over the augmented skeleton per distinct child source state (a state
    /// shared by several children, e.g. adjacent edges of a cycle, is searched once); a parent with
    /// many source states searches them across the pool.
    fn search_fill(&mut self, parent: usize, skeleton: &Skeleton<IndexType>) {
        let block_tree = self.block_tree();
        let source_groups = child_source_groups(block_tree, parent);
        if source_groups.len() >= PARALLEL_SOURCE_THRESHOLD {
            let groups: Vec<(DirectedNodeIndex<IndexType>, SourceGroup)> =
                source_groups.into_iter().collect();
            let per_source: Vec<Vec<OutsideEntry<IndexType>>> = groups
                .par_iter()
                .map_init(DijkstraWorkspace::new, |workspace, (source, group)| {
                    let mut out = Vec::new();
                    solve_source(block_tree, skeleton, workspace, *source, group, &mut out);
                    out
                })
                .collect();
            for entries in per_source {
                set_entries(&mut self.outside, entries);
            }
        } else {
            let mut entries = Vec::new();
            for (source, group) in &source_groups {
                solve_source(
                    block_tree,
                    skeleton,
                    &mut self.workspace,
                    *source,
                    group,
                    &mut entries,
                );
            }
            set_entries(&mut self.outside, entries);
        }
    }
}

/// A child's outside matrix from a bond parent's augmented `4x4` (over the bond's slots), or `None`
/// if one of the child's pole states is not a bond vertex.
fn bond_child_matrix<IndexType: GraphIndexInteger>(
    augmented: &MinPlusMatrix<IndexType, 4>,
    bond: BondSlots<IndexType>,
    poles: (NodeIndex<IndexType>, NodeIndex<IndexType>),
) -> Option<MinPlusMatrix<IndexType, 4>> {
    let slots = bond.pole_slots(poles)?;
    let mut matrix = MinPlusMatrix::<IndexType, 4>::infinity();
    for i in 0..4 {
        for j in 0..4 {
            matrix.set(i, j, augmented.get(slots[i], slots[j]));
        }
    }
    Some(matrix)
}

/// Set a child's finite outside entries from the parent's labels. Matches the search exactly: the
/// distance to self is `0` even for an isolated pole state; absent or unreachable targets stay `∞`.
fn set_label_entries<IndexType: GraphIndexInteger>(
    outside: &mut MinPlusMatrix<IndexType, 4>,
    labels: &SkeletonLabels<IndexType>,
    (s, t): (NodeIndex<IndexType>, NodeIndex<IndexType>),
) {
    let segment = |pole: Pole| if pole == Pole::Source { s } else { t };
    for from_pole in Pole::all() {
        for from_orientation in Orientation::all() {
            let source = segment(from_pole).into_directed(from_orientation.is_forward());
            for to_pole in Pole::all() {
                for to_orientation in Orientation::all() {
                    let target = segment(to_pole).into_directed(to_orientation.is_forward());
                    let distance = if source == target {
                        Some(GfaPathLength::from_usize(0))
                    } else {
                        labels.distance(source, target)
                    };
                    if let Some(distance) = distance {
                        outside.set(
                            slot(from_pole, from_orientation),
                            slot(to_pole, to_orientation),
                            MinPlus::finite(distance),
                        );
                    }
                }
            }
        }
    }
}

/// The children whose outside entries one source pole state fills: `(child, from slot)` pairs.
type SourceGroup = Vec<(usize, usize)>;

/// Every child's source pole states of `parent`, grouped by directed node: `source -> [(child,
/// from slot)]`.
fn child_source_groups<IndexType: GraphIndexInteger>(
    block_tree: &BlockSpqrTree<IndexType>,
    parent: usize,
) -> FastMap<DirectedNodeIndex<IndexType>, SourceGroup> {
    let mut source_groups: FastMap<DirectedNodeIndex<IndexType>, SourceGroup> = FastMap::default();
    for &child in block_tree.tree().children(parent) {
        let Some((s_c, t_c)) = block_tree.node(child).reference_poles() else {
            continue; // root has no reference poles and no outside to compute.
        };
        for from_pole in Pole::all() {
            let from_segment = if from_pole == Pole::Source { s_c } else { t_c };
            for from_orientation in Orientation::all() {
                let source = from_segment.into_directed(from_orientation.is_forward());
                source_groups
                    .entry(source)
                    .or_default()
                    .push((child, slot(from_pole, from_orientation)));
            }
        }
    }
    source_groups
}

/// The outside entries one source pole state yields, from one search over `skeleton` on
/// `workspace`: `(child, from slot, to slot, distance)` for every child of `group`. Target-independent
/// per source, so a source shared by many children is searched once.
fn solve_source<IndexType: GraphIndexInteger>(
    block_tree: &BlockSpqrTree<IndexType>,
    skeleton: &Skeleton<IndexType>,
    workspace: &mut DijkstraWorkspace<IndexType>,
    source: DirectedNodeIndex<IndexType>,
    group: &[(usize, usize)],
    out: &mut Vec<OutsideEntry<IndexType>>,
) {
    let source_id = skeleton.node_id(source);
    if let Some(source_id) = source_id {
        workspace.run(skeleton, &[(source_id, GfaPathLength::from_usize(0))]);
    }
    for &(child, from_slot) in group {
        let (s_c, t_c) = block_tree.node(child).reference_poles().unwrap();
        for to_pole in Pole::all() {
            let to_segment = if to_pole == Pole::Source { s_c } else { t_c };
            for to_orientation in Orientation::all() {
                let target = to_segment.into_directed(to_orientation.is_forward());
                let distance = if source_id.is_some() {
                    skeleton.node_id(target).and_then(|id| workspace.get(id))
                } else if target == source {
                    // Isolated source: reaches only itself at distance 0.
                    Some(GfaPathLength::from_usize(0))
                } else {
                    None
                };
                if let Some(distance) = distance {
                    out.push((child, from_slot, slot(to_pole, to_orientation), distance));
                }
            }
        }
    }
}

/// Set the finite outside entries a parent's searches produced.
fn set_entries<IndexType: GraphIndexInteger>(
    outside: &mut [MinPlusMatrix<IndexType, 4>],
    entries: Vec<OutsideEntry<IndexType>>,
) {
    for (child, from_slot, to_slot, distance) in entries {
        outside[child].set(from_slot, to_slot, MinPlus::finite(distance));
    }
}
