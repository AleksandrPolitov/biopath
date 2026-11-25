//! The pole lift: the single-step transitions that climb a pole projection up the SPQR tree.
//!
//! At query time a source/target lies in some deep SPQR node and must be projected up to an
//! ancestor's poles. A **pole-distance vector** holds the distances from a fixed query point to the
//! four `(pole, orientation)` combinations `(s^+, s^-, t^+, t^-)` of an SPQR node's reference poles.
//!
//! Because a node's subtree connects to the rest of the block only through its two poles, the
//! distances to a parent's poles are obtained from the distances to the child's poles by a single
//! `4x4` min-plus **transition matrix**
//! `T[(child pole, o)][(parent pole, o')] = D_parent(child pole^o -> parent pole^o')`, evaluated in
//! the parent's assembled skeleton. Lifting across several levels composes the transitions one
//! level at a time (real climbs are short).

use crate::rooted_tree::TreeAccess;
use bidirected_adjacency_array::{
    index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use rayon::prelude::*;

use crate::{
    inside_dp::{InsideDp, transpose_adjacency},
    min_plus::{
        Mat2, MinPlus, MinPlusMatrix,
        sparse::{self, SparseMat4},
    },
    orientation::Orientation,
    path::GfaPathLength,
    skeleton::{DijkstraWorkspace, Skeleton, SkeletonAccess},
};

#[cfg(test)]
mod tests;

/// Which reference pole of an SPQR node: the source pole `s` or the target pole `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pole {
    /// The first reference pole `s`.
    Source,
    /// The second reference pole `t`.
    Target,
}

impl Pole {
    /// Both poles, in index order.
    pub fn all() -> [Pole; 2] {
        [Pole::Source, Pole::Target]
    }

    fn index(self) -> usize {
        match self {
            Pole::Source => 0,
            Pole::Target => 1,
        }
    }
}

/// Index into a 4-entry pole vector / transition / pole-state matrix for a `(pole, orientation)`
/// pair.
pub(crate) fn slot(pole: Pole, orientation: Orientation) -> usize {
    pole.index() * 2 + orientation.index()
}

/// Add the edges of a `4x4` pole-state matrix (indexed by [`slot`]) between two pole segments to a
/// skeleton adjacency. Used to fold a child subtree (inside matrix) or the outside relation into a
/// skeleton, including pole self-flips such as `s^+ -> s^-`.
pub(crate) fn add_pole_state_edges<IndexType: GraphIndexInteger>(
    adjacency: &mut crate::inside_dp::SkeletonAdjacency<IndexType>,
    source_pole: bidirected_adjacency_array::index::NodeIndex<IndexType>,
    target_pole: bidirected_adjacency_array::index::NodeIndex<IndexType>,
    matrix: &MinPlusMatrix<IndexType, 4>,
) {
    let segment = |pole: Pole| match pole {
        Pole::Source => source_pole,
        Pole::Target => target_pole,
    };
    for from_pole in Pole::all() {
        for from_orientation in Orientation::all() {
            let from = segment(from_pole).into_directed(from_orientation.is_forward());
            for to_pole in Pole::all() {
                for to_orientation in Orientation::all() {
                    if let Some(weight) = matrix
                        .get(
                            slot(from_pole, from_orientation),
                            slot(to_pole, to_orientation),
                        )
                        .into_option()
                    {
                        adjacency.entry(from).or_default().push((
                            segment(to_pole).into_directed(to_orientation.is_forward()),
                            weight,
                        ));
                    }
                }
            }
        }
    }
}

/// Bake a cut node's orientation-flip closure into a skeleton adjacency as **same-vertex edges**.
///
/// A flip at cut `z` — leaving `z` in one orientation and returning in the other via a self-contained
/// detour that touches the rest of the skeleton only at `z` (e.g. the pendant subtree hanging off `z`,
/// i.e. `down[z]`) — is, structurally, an edge between `z`'s two directed states. Adding `z^a -> z^b`
/// weighted `closure(a, b)` for every finite entry lets any search over this skeleton use the flip like
/// an ordinary edge. The inside DP then folds it into `z`'s subtree pole-state matrix and composes it
/// upward through the virtual edges, so a query resolves a flip route as an ordinary distance — no
/// per-query flip closure, no cut-to-everywhere table.
///
/// Because every baked entry is a real walk length and flips are non-negative, this only *adds* valid
/// routes: every min stays a min over real walks, and a route that would chain two flips is strictly
/// costlier (so the answer is never wrong, only potentially cheaper where a flip genuinely helps). The
/// closure must be the detour's *local* cost (oracle-independent) so no fixpoint is introduced.
pub(crate) fn bake_flip_edge<IndexType: GraphIndexInteger>(
    adjacency: &mut crate::inside_dp::SkeletonAdjacency<IndexType>,
    cut: bidirected_adjacency_array::index::NodeIndex<IndexType>,
    closure: &Mat2<IndexType>,
) {
    for from in Orientation::all() {
        for to in Orientation::all() {
            if let Some(weight) = closure.get(from.index(), to.index()).into_option() {
                adjacency
                    .entry(cut.into_directed(from.is_forward()))
                    .or_default()
                    .push((cut.into_directed(to.is_forward()), weight));
            }
        }
    }
}

/// Distances from a fixed query point to the four `(pole, orientation)` combinations of an SPQR
/// node's reference poles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoleVector<IndexType: GraphIndexInteger> {
    entries: [MinPlus<IndexType>; 4],
}

impl<IndexType: GraphIndexInteger> PoleVector<IndexType> {
    /// The all-`+∞` vector (nothing reachable).
    pub fn infinity() -> Self {
        Self {
            entries: [MinPlus::infinity(); 4],
        }
    }

    /// Build from raw entries in slot order `[s^+, s^-, t^+, t^-]`.
    pub fn from_entries(entries: [MinPlus<IndexType>; 4]) -> Self {
        Self { entries }
    }

    /// The distance to `pole` in `orientation`.
    pub fn get(&self, pole: Pole, orientation: Orientation) -> MinPlus<IndexType> {
        self.entries[slot(pole, orientation)]
    }

    /// Set the distance to `pole` in `orientation`.
    pub fn set(&mut self, pole: Pole, orientation: Orientation, value: MinPlus<IndexType>) {
        self.entries[slot(pole, orientation)] = value;
    }

    fn as_array(&self) -> &[MinPlus<IndexType>; 4] {
        &self.entries
    }
}

/// An index of pole transitions for one block's rooted SPQR tree: the single-step table, lifted by
/// a naive `O(depth)` climb (a climb on a real block is a few levels).
pub struct PoleLift<IndexType: GraphIndexInteger> {
    /// Single-step transition `T_{v -> parent(v)}`, present only when `parent(v)` has poles (i.e. is
    /// not the root).
    step: Vec<Option<MinPlusMatrix<IndexType, 4>>>,
}

/// Fill one parent pole-state column of a child's `4x4` transition matrix from a completed
/// transposed-skeleton search: `matrix[child_slot][parent_slot] = D(child_pole^o -> parent_pole)`,
/// read as `D_transposed(parent_pole -> child_pole^o) = workspace.get(child id)`. Shared by the
/// serial (small-parent) and parallel (oversized-parent) paths so both compute identical values.
fn fill_child_column<IndexType: GraphIndexInteger>(
    matrix: &mut MinPlusMatrix<IndexType, 4>,
    parent_slot: usize,
    parent_directed: DirectedNodeIndex<IndexType>,
    source_reachable: bool,
    skeleton: &Skeleton<IndexType>,
    workspace: &DijkstraWorkspace<IndexType>,
    child_poles: (NodeIndex<IndexType>, NodeIndex<IndexType>),
) {
    let (s_c, t_c) = child_poles;
    for child_pole in Pole::all() {
        let child_segment = if child_pole == Pole::Source { s_c } else { t_c };
        for child_orientation in Orientation::all() {
            let child_directed = child_segment.into_directed(child_orientation.is_forward());
            let distance = if source_reachable {
                skeleton
                    .node_id(child_directed)
                    .and_then(|id| workspace.get(id))
            } else if child_directed == parent_directed {
                // Isolated parent pole: reachable only from itself at distance 0.
                Some(GfaPathLength::from_usize(0))
            } else {
                None
            };
            if let Some(distance) = distance {
                matrix.set(
                    slot(child_pole, child_orientation),
                    parent_slot,
                    MinPlus::finite(distance),
                );
            }
        }
    }
}

/// Transitions for one small `parent`, reusing a caller-owned `workspace` and running the 4
/// pole-state Dijkstras serially (parallelism is across parents). Returns `(child_local, matrix)`
/// only for children that have reference poles. `parent` must have children and reference poles.
fn parent_transitions_serial<IndexType, NodeData, EdgeData>(
    inside: &InsideDp<'_, IndexType, NodeData, EdgeData>,
    parent: usize,
    workspace: &mut DijkstraWorkspace<IndexType>,
) -> Vec<(usize, MinPlusMatrix<IndexType, 4>)>
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    let block_tree = inside.block_tree();
    let (s_p, t_p) = block_tree.node(parent).reference_poles().unwrap();
    let adjacency = inside.build_skeleton_adjacency(parent);
    let transposed = transpose_adjacency(&adjacency);
    let skeleton = Skeleton::from_adjacency(&transposed);

    let mut mats: Vec<(usize, MinPlusMatrix<IndexType, 4>)> = block_tree
        .tree()
        .children(parent)
        .iter()
        .filter(|&&child| block_tree.node(child).reference_poles().is_some())
        .map(|&child| (child, MinPlusMatrix::infinity()))
        .collect();

    for parent_pole in Pole::all() {
        let parent_segment = if parent_pole == Pole::Source {
            s_p
        } else {
            t_p
        };
        for parent_orientation in Orientation::all() {
            let parent_slot = slot(parent_pole, parent_orientation);
            let parent_directed = parent_segment.into_directed(parent_orientation.is_forward());
            let source_id = skeleton.node_id(parent_directed);
            if let Some(source_id) = source_id {
                workspace.run(&skeleton, &[(source_id, GfaPathLength::from_usize(0))]);
            }
            for (child, matrix) in mats.iter_mut() {
                let (s_c, t_c) = block_tree.node(*child).reference_poles().unwrap();
                fill_child_column(
                    matrix,
                    parent_slot,
                    parent_directed,
                    source_id.is_some(),
                    &skeleton,
                    workspace,
                    (s_c, t_c),
                );
            }
        }
    }
    mats
}

/// Transitions for one oversized `parent` (the giant R-node): the 4 pole-state Dijkstras run in
/// parallel (each into its own workspace, so four giant workspaces are held together), then the
/// per-child reads run in parallel over the children. Identical values to
/// [`parent_transitions_serial`], different schedule.
fn parent_transitions_parallel<IndexType, NodeData, EdgeData>(
    inside: &InsideDp<'_, IndexType, NodeData, EdgeData>,
    parent: usize,
) -> Vec<(usize, MinPlusMatrix<IndexType, 4>)>
where
    IndexType: GraphIndexInteger + Send + Sync,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    let block_tree = inside.block_tree();
    let (s_p, t_p) = block_tree.node(parent).reference_poles().unwrap();
    let adjacency = inside.build_skeleton_adjacency(parent);
    let transposed = transpose_adjacency(&adjacency);
    let skeleton = Skeleton::from_adjacency(&transposed);

    // The 4 parent pole-states (slot + directed node).
    let pole_states: Vec<(usize, DirectedNodeIndex<IndexType>)> = Pole::all()
        .into_iter()
        .flat_map(|parent_pole| {
            let segment = if parent_pole == Pole::Source {
                s_p
            } else {
                t_p
            };
            Orientation::all()
                .into_iter()
                .map(move |o| (slot(parent_pole, o), segment.into_directed(o.is_forward())))
        })
        .collect();

    // Run the 4 transposed-skeleton one-to-all Dijkstras concurrently, keeping each workspace to read.
    let searches: Vec<(
        usize,
        DirectedNodeIndex<IndexType>,
        bool,
        DijkstraWorkspace<IndexType>,
    )> = pole_states
        .into_par_iter()
        .map(|(parent_slot, parent_directed)| {
            let mut workspace = DijkstraWorkspace::new();
            let source_id = skeleton.node_id(parent_directed);
            if let Some(source_id) = source_id {
                workspace.run(&skeleton, &[(source_id, GfaPathLength::from_usize(0))]);
            }
            (parent_slot, parent_directed, source_id.is_some(), workspace)
        })
        .collect();

    // Assemble each child's matrix from the 4 searches, in parallel over children.
    block_tree
        .tree()
        .children(parent)
        .par_iter()
        .filter_map(|&child| {
            let (s_c, t_c) = block_tree.node(child).reference_poles()?;
            let mut matrix = MinPlusMatrix::infinity();
            for (parent_slot, parent_directed, source_reachable, workspace) in &searches {
                fill_child_column(
                    &mut matrix,
                    *parent_slot,
                    *parent_directed,
                    *source_reachable,
                    &skeleton,
                    workspace,
                    (s_c, t_c),
                );
            }
            Some((child, matrix))
        })
        .collect()
}

impl<IndexType: GraphIndexInteger> PoleLift<IndexType> {
    /// Build the pole-lift index from a completed inside DP.
    pub fn compute<NodeData: GfaNodeData + Sync, EdgeData: GfaEdgeData + Sync>(
        inside: &InsideDp<'_, IndexType, NodeData, EdgeData>,
    ) -> Self
    where
        IndexType: Send + Sync,
    {
        let block_tree = inside.block_tree();
        let tree = block_tree.tree();
        let node_count = block_tree.node_count();

        // Each parent's transitions are computed independently, and every child has exactly one
        // parent, so the `step[child]` writes across parents are disjoint — the whole pass is
        // embarrassingly parallel. We split it in two so the one oversized parent (the giant R-node,
        // ~10M-node skeleton, 4 one-to-all Dijkstras) does not serialize a shared worker:
        //  - Pass A: the many small parents, parallel across parents (one workspace per rayon worker).
        //  - Pass B: the oversized parent(s), where the 4 pole Dijkstras and the child reads are
        //    themselves parallelized. A parent is "oversized" purely by child count; this only
        //    affects *which* parallel schedule runs, never the values computed (both paths are the
        //    identical transposed-skeleton one-to-all reads).
        const OVERSIZED_CHILDREN: usize = 100_000;

        let mut step: Vec<Option<MinPlusMatrix<IndexType, 4>>> = vec![None; node_count];

        // Pass A — small parents, parallel across parents.
        let small: Vec<Vec<(usize, MinPlusMatrix<IndexType, 4>)>> = (0..node_count)
            .into_par_iter()
            .filter(|&parent| {
                let children = tree.children(parent);
                !children.is_empty()
                    && children.len() < OVERSIZED_CHILDREN
                    && block_tree.node(parent).reference_poles().is_some()
            })
            .map_init(DijkstraWorkspace::new, |workspace, parent| {
                parent_transitions_serial(inside, parent, workspace)
            })
            .collect();
        for part in small {
            for (child, matrix) in part {
                step[child] = Some(matrix);
            }
        }

        // Pass B — oversized parent(s) (in practice exactly one: the giant R-node), each with its
        // 4 pole-state Dijkstras and per-child reads parallelized internally. This runs after Pass A's
        // barrier, so its internal `par_iter`s are top-level (not nested inside Pass A).
        for parent in 0..node_count {
            let children = tree.children(parent);
            if children.len() < OVERSIZED_CHILDREN
                || block_tree.node(parent).reference_poles().is_none()
            {
                continue;
            }
            for (child, matrix) in parent_transitions_parallel(inside, parent) {
                step[child] = Some(matrix);
            }
        }

        Self { step }
    }

    /// Lift a pole vector from `from` up to its ancestor `to` (both must have poles), climbing one
    /// level at a time.
    #[cfg(test)]
    pub fn lift<T: TreeAccess + ?Sized>(
        &self,
        tree: &T,
        from: usize,
        to: usize,
        vector: PoleVector<IndexType>,
    ) -> PoleVector<IndexType> {
        climb(self, tree, from, to, vector)
    }

    /// The single-step transitions in the image's sparse form (see [`SparseMat4`]). The root (and any
    /// node without a transition) gets mask `0`: it reads as the all-`∞` matrix, which the climb never
    /// consults (a climb stops at its ancestor).
    pub fn sparse_steps(&self) -> SparseMat4<IndexType> {
        SparseMat4::encode(self.step.iter().map(Option::as_ref))
    }
}

/// The four directed pole states of an SPQR node with reference poles `(s, t)`, indexed by [`slot`] —
/// the row/column order of its pole-state matrices (the same mapping [`add_pole_state_edges`] uses).
pub(crate) fn pole_states<IndexType: GraphIndexInteger>(
    s: bidirected_adjacency_array::index::NodeIndex<IndexType>,
    t: bidirected_adjacency_array::index::NodeIndex<IndexType>,
) -> [DirectedNodeIndex<IndexType>; 4] {
    let mut states = [s.into_directed(true); 4];
    for pole in Pole::all() {
        let segment = match pole {
            Pole::Source => s,
            Pole::Target => t,
        };
        for orientation in Orientation::all() {
            states[slot(pole, orientation)] = segment.into_directed(orientation.is_forward());
        }
    }
    states
}

/// Read access to the single-step pole transitions, implemented by the owned [`PoleLift`] and the
/// image-backed [`PoleStepsView`].
pub trait PoleSteps<IndexType: GraphIndexInteger> {
    /// `T_{v -> parent(v)}`, or `None` where `v`'s parent has no poles (the root).
    fn step(&self, v: usize) -> Option<MinPlusMatrix<IndexType, 4>>;

    /// One climb step: `vector · T_{v -> parent(v)}`, or `None` at the root.
    #[inline]
    fn lift_step(
        &self,
        v: usize,
        vector: &[MinPlus<IndexType>; 4],
    ) -> Option<[MinPlus<IndexType>; 4]> {
        self.step(v)
            .map(|transition| transition.left_multiply_vector(vector))
    }
}

impl<IndexType: GraphIndexInteger> PoleSteps<IndexType> for PoleLift<IndexType> {
    #[inline]
    fn step(&self, v: usize) -> Option<MinPlusMatrix<IndexType, 4>> {
        self.step[v]
    }
}

/// Image-backed single-step transitions in the [`SparseMat4`] form (the root's reads as `∞`, never
/// consulted).
#[derive(Clone, Copy)]
pub struct PoleStepsView<'a, IndexType: GraphIndexInteger> {
    pub masks: &'a [u16],
    pub offsets: &'a [u32],
    pub values: &'a [MinPlus<IndexType>],
}

impl<IndexType: GraphIndexInteger> PoleSteps<IndexType> for PoleStepsView<'_, IndexType> {
    #[inline]
    fn step(&self, v: usize) -> Option<MinPlusMatrix<IndexType, 4>> {
        Some(sparse::decode(self.masks, self.offsets, self.values, v))
    }

    /// The product straight from the sparse entries ([`sparse::left_multiply`]).
    #[inline]
    fn lift_step(
        &self,
        v: usize,
        vector: &[MinPlus<IndexType>; 4],
    ) -> Option<[MinPlus<IndexType>; 4]> {
        Some(sparse::left_multiply(
            self.masks,
            self.offsets,
            self.values,
            v,
            vector,
        ))
    }
}

impl<IndexType: GraphIndexInteger> PoleStepsView<'_, IndexType> {
    /// Lift a pole vector from `from` up to its ancestor `to` by the naive climb.
    pub fn lift<T: TreeAccess + ?Sized>(
        &self,
        tree: &T,
        from: usize,
        to: usize,
        vector: PoleVector<IndexType>,
    ) -> PoleVector<IndexType> {
        debug_assert!(tree.is_ancestor(to, from), "to must be an ancestor of from");
        climb(self, tree, from, to, vector)
    }
}

/// The naive `O(depth)` climb shared by the owned and image-backed transitions.
fn climb<IndexType: GraphIndexInteger, S: PoleSteps<IndexType> + ?Sized, T: TreeAccess + ?Sized>(
    steps: &S,
    tree: &T,
    from: usize,
    to: usize,
    mut vector: PoleVector<IndexType>,
) -> PoleVector<IndexType> {
    debug_assert!(tree.is_ancestor(to, from), "to must be an ancestor of from");
    let mut current = from;
    while current != to {
        vector = PoleVector::from_entries(
            steps
                .lift_step(current, vector.as_array())
                .expect("every node on the lift path must have a pole transition"),
        );
        current = tree
            .parent(current)
            .expect("to must be an ancestor of from");
    }
    vector
}
