//! The build of an S-node oracle: the cycle and its leaf networks (`cycle.rs`), then the ring form
//! (`ring_build.rs`) or the tree; and the per-block pass that builds every qualifying S-node's
//! oracle.

use bidirected_adjacency_array::{
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use rayon::prelude::*;

use super::{
    Form, PfxVertex, SegRunRec, SegTreeAccess, SegVertexRec,
    cycle::{Cycle, closing_with_outside, cycle_pairs},
    ring_build::ring_form,
    tree::{build_tree, mat_count},
};
use crate::{
    block_engine::Tiers,
    inside_dp::InsideDp,
    min_plus::{
        MinPlusMatrix,
        network::{Mat4, close_network},
        word::inf,
    },
    outside_dp::OutsideDp,
    skeleton::{Skeleton, SkeletonAccess},
};

/// The owned (build-time) oracle of one S-node: a segment tree, or the ring form (`runs`
/// non-empty).
pub struct SNodeSegTree<IndexType: GraphIndexInteger> {
    pub(super) m: usize,
    pub(super) vertices: Vec<SegVertexRec<IndexType>>,
    pub(super) mats: Vec<Mat4<IndexType>>,
    pub(super) pfx: Vec<PfxVertex<IndexType>>,
    pub(super) runs: Vec<SegRunRec>,
    pub(super) blocks: usize,
}

impl<IndexType: GraphIndexInteger> SegTreeAccess<IndexType> for SNodeSegTree<IndexType> {
    #[inline]
    fn m(&self) -> usize {
        self.m
    }
    #[inline]
    fn vertex_recs(&self) -> &[SegVertexRec<IndexType>] {
        &self.vertices
    }
    #[inline]
    fn mats(&self) -> &[Mat4<IndexType>] {
        &self.mats
    }
    #[inline]
    fn pfx(&self) -> &[PfxVertex<IndexType>] {
        &self.pfx
    }
    #[inline]
    fn runs(&self) -> &[SegRunRec] {
        &self.runs
    }
    #[inline]
    fn blocks(&self) -> usize {
        self.blocks
    }
}

impl<IndexType: GraphIndexInteger> SNodeSegTree<IndexType> {
    /// Build the oracle of one S-node from its **inner** skeleton (real edges + folded children, no
    /// `D_out`), its reference poles (`None` at the block root), its `D_out` (`None` at the root),
    /// the node lengths, and — when known — its cycle edges as node pairs (the S-node's child
    /// virtual edges and real edges), so that a dead child (one with no pass arc) still counts as
    /// a cycle edge; without them the cycle is read off the skeleton's arcs.
    ///
    /// Returns `None` when the skeleton is not a clean ring: fewer than three vertices, a vertex with
    /// more than two cycle neighbours, a chord, or (for a non-root node) a path whose ends are not the
    /// reference poles. The caller then keeps the skeleton/label representation, so a `None` is never
    /// a wrong answer. On a cycle the ring form never declines.
    pub fn build<S: SkeletonAccess<IndexType> + ?Sized>(
        inner: &S,
        reference_poles: Option<(NodeIndex<IndexType>, NodeIndex<IndexType>)>,
        outside: Option<&MinPlusMatrix<IndexType, 4>>,
        len_of: &dyn Fn(NodeIndex<IndexType>) -> u64,
        cycle_edges: Option<&[(NodeIndex<IndexType>, NodeIndex<IndexType>)]>,
    ) -> Option<Self> {
        Self::build_with_form(
            inner,
            reference_poles,
            outside,
            len_of,
            cycle_edges,
            Form::Ring,
        )
    }

    /// [`Self::build`] with an explicit representation (the ring form falls back to the tree when it
    /// declines).
    pub fn build_with_form<S: SkeletonAccess<IndexType> + ?Sized>(
        inner: &S,
        reference_poles: Option<(NodeIndex<IndexType>, NodeIndex<IndexType>)>,
        outside: Option<&MinPlusMatrix<IndexType, 4>>,
        len_of: &dyn Fn(NodeIndex<IndexType>) -> u64,
        cycle_edges: Option<&[(NodeIndex<IndexType>, NodeIndex<IndexType>)]>,
        form: Form,
    ) -> Option<Self> {
        let pairs = cycle_pairs(inner, cycle_edges);
        let cycle = Cycle::of(&pairs, reference_poles)?;
        let m = cycle.m();
        let lens = cycle.lens(len_of);
        let leaves = cycle.leaf_networks(inner)?;
        let closing_inner = leaves[m - 1];
        let closing_lca = closing_with_outside(&closing_inner, reference_poles, outside);
        let vertices = cycle.vertex_recs();
        // The ring form reads the RAW networks (closure would hide the lane structure).
        if form == Form::Ring
            && let Some(ring) = ring_form(m, &leaves[..m - 1], &closing_inner, &closing_lca, &lens)
        {
            return Some(Self {
                m,
                vertices,
                mats: ring.mats,
                pfx: ring.pfx,
                runs: ring.runs,
                blocks: ring.blocks,
            });
        }
        Some(Self::tree_form(
            vertices,
            leaves,
            closing_inner,
            closing_lca,
        ))
    }

    /// The tree form: the segment tree over the `m−1` closed path leaves, then the two closed
    /// closing networks.
    fn tree_form(
        vertices: Vec<SegVertexRec<IndexType>>,
        mut leaves: Vec<Mat4<IndexType>>,
        mut closing_inner: Mat4<IndexType>,
        mut closing_lca: Mat4<IndexType>,
    ) -> Self {
        let m = vertices.len();
        close_network(&mut closing_inner);
        close_network(&mut closing_lca);
        for leaf in leaves[..m - 1].iter_mut() {
            close_network(leaf);
        }
        let mut mats = vec![[inf::<IndexType>(); 16]; mat_count(m)];
        build_tree(&mut mats[..2 * (m - 1) - 1], &leaves[..m - 1], 0, 0, m - 1);
        mats[mat_count(m) - 2] = closing_inner;
        mats[mat_count(m) - 1] = closing_lca;
        Self {
            m,
            vertices,
            mats,
            pfx: Vec::new(),
            runs: Vec::new(),
            blocks: 0,
        }
    }
}

/// Build the segment tree of every qualifying S-node of a block: an S-node whose inner skeleton is at
/// the segment-tree tier ([`Tiers::segtree`]) and is a clean cycle. Aligned with the block's local
/// ids; `None` where no tree is built.
pub fn precompute_s_node_segtrees<'a, IndexType, NodeData, EdgeData>(
    inside: &InsideDp<'a, IndexType, NodeData, EdgeData>,
    outside: &OutsideDp<IndexType>,
    inner: &[Skeleton<IndexType>],
    tiers: &Tiers,
) -> Vec<Option<Box<SNodeSegTree<IndexType>>>>
where
    IndexType: GraphIndexInteger + Send + Sync,
    NodeData: GfaNodeData + Send + Sync,
    EdgeData: GfaEdgeData + Send + Sync,
{
    let block_tree = inside.block_tree();
    let node_count = block_tree.node_count();
    let build = |local: usize| -> Option<Box<SNodeSegTree<IndexType>>> {
        if !tiers.segtree(
            block_tree.node(local).node_type(),
            inner[local].node_count(),
        ) {
            return None;
        }
        let node = block_tree.node(local);
        let poles = node.reference_poles();
        let out = poles.map(|_| outside.outside_matrix(local));
        let len_of = |node: NodeIndex<IndexType>| inside.sequence_len(node);
        // The S-node's cycle edges (child virtual edges and real edges, the reference edge excluded),
        // so a dead child — one with no pass arc in any orientation — still counts as a cycle edge.
        let mut edges: Vec<(NodeIndex<IndexType>, NodeIndex<IndexType>)> =
            node.child_virtual_edges().iter().map(|c| c.poles).collect();
        edges.extend(
            block_tree
                .real_directed_edges(local, inside.graph())
                .map(|e| (e.from.into_bidirected(), e.to.into_bidirected())),
        );
        SNodeSegTree::build(&inner[local], poles, out, &len_of, Some(&edges)).map(Box::new)
    };
    // Same dispatch gate as `precompute_skeletons`: a rayon fan-out only pays off on a big block.
    const PARALLEL_NODE_THRESHOLD: usize = 64;
    if node_count >= PARALLEL_NODE_THRESHOLD {
        (0..node_count).into_par_iter().map(build).collect()
    } else {
        (0..node_count).map(build).collect()
    }
}
