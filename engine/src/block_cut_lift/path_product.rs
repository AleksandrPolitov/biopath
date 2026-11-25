//! The path product over the cut forest: a `2`-vector of cut distances carried along the cut path
//! between a cut and its ancestor, one step matrix per cut (`step_up[c]`: `c -> parent cut`, read
//! off the `M_aug` of the block between them; `step_down[c]`: the reverse). Two backends:
//!
//! - the closed-form prefix lift ([`prefix_lift`]) when at most
//!   [`prefix_lift::MAX_EXCEPTION_FRACTION`] of the steps are exceptions to its closed form and the
//!   prefix fits the index word: a run of regular steps costs `O(1)`;
//! - the chain segment trees ([`chain_segtree`]) otherwise: `O(log chain)` per chain crossed.
//!
//! [`LiftBackend::Prefix`] / [`LiftBackend::Segtree`] (the build option
//! [`LiftOptions::backend`](super::LiftOptions::backend)) forces one.

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::{BlockCutLift, aug::BlockAug, forest::BlockCutForest};
use crate::{
    chain_segtree::{self, LiftSegTreesView, SegSections},
    image::Image,
    min_plus::{Mat2, MinPlus},
    prefix_lift::{self, LiftBackend},
};

/// The backend's section handles (which backend: `LiftMeta::flags`).
#[derive(Clone, Copy, Debug)]
pub(super) enum BackendSections<IndexType: GraphIndexInteger> {
    Prefix(prefix_lift::PrefixSections<IndexType>),
    Segtree(SegSections<IndexType>),
}

/// The path-product backend as built (owned, build-time only).
pub(super) enum OwnedBackend<IndexType: GraphIndexInteger> {
    Prefix(prefix_lift::PrefixLift<IndexType>),
    Segtree(chain_segtree::LiftSegTrees<IndexType>),
}

impl<IndexType: GraphIndexInteger> OwnedBackend<IndexType> {
    pub(super) fn prefix(&self) -> Option<&prefix_lift::PrefixLift<IndexType>> {
        match self {
            OwnedBackend::Prefix(p) => Some(p),
            OwnedBackend::Segtree(_) => None,
        }
    }

    /// Push the backend's sections.
    pub(super) fn write(&self, image: &Image) {
        match self {
            OwnedBackend::Prefix(p) => {
                p.write(image);
            }
            OwnedBackend::Segtree(t) => {
                t.write(image);
            }
        }
    }
}

/// The path-product backend over the cut forest (with each cut's parent in it): the closed-form
/// prefix lift when `backend` and the gate allow it ([`prefix_lift::choose_prefix`]), else the
/// chain segment trees. A step is the `M_aug` of the block between a cut and its parent cut.
pub(super) fn build_backend<IndexType: GraphIndexInteger>(
    backend: LiftBackend,
    forest: &BlockCutForest<IndexType>,
    block_aug: &[BlockAug<IndexType>],
    full: &[Mat2<IndexType>],
    cut_order: &[usize],
) -> (Vec<Option<usize>>, OwnedBackend<IndexType>) {
    let nc = forest.number_of_cuts;
    let cut_parent: Vec<Option<usize>> = (0..nc).map(|cut| forest.cut_parent(cut)).collect();
    let (step_up, step_down) = step_matrices(forest, block_aug, &cut_parent);
    let (candidate, stats) =
        prefix_lift::PrefixLift::build(nc, &cut_parent, cut_order, &step_up, full);
    let overflowed = candidate.is_none();
    let built = if prefix_lift::choose_prefix(backend, &stats, overflowed) {
        OwnedBackend::Prefix(candidate.expect("a chosen prefix lift was built"))
    } else {
        OwnedBackend::Segtree(chain_segtree::LiftSegTrees::build(
            nc,
            &cut_parent,
            cut_order,
            &step_up,
            &step_down,
        ))
    };
    (cut_parent, built)
}

/// The step matrices of the cut forest, up (`cut -> parent cut`) and down (`parent cut -> cut`),
/// read off the `M_aug` of the block between them (identity at a forest root).
fn step_matrices<IndexType: GraphIndexInteger>(
    forest: &BlockCutForest<IndexType>,
    block_aug: &[BlockAug<IndexType>],
    cut_parent: &[Option<usize>],
) -> (Vec<Mat2<IndexType>>, Vec<Mat2<IndexType>>) {
    let mut step_up = vec![Mat2::identity(); forest.number_of_cuts];
    let mut step_down = vec![Mat2::identity(); forest.number_of_cuts];
    for (cut, parent) in cut_parent.iter().enumerate() {
        let Some(parent) = *parent else {
            continue;
        };
        let bid = forest
            .parent_block(cut)
            .expect("a cut with a parent cut has a parent block");
        let (child_segment, parent_segment) = (forest.cut_segment[cut], forest.cut_segment[parent]);
        step_up[cut] = block_aug[bid].matrix(child_segment, parent_segment);
        step_down[cut] = block_aug[bid].matrix(parent_segment, child_segment);
    }
    (step_up, step_down)
}

/// The path-product backend as the query reads it.
#[derive(Clone, Copy)]
pub(super) enum BackendView<'a, IndexType: GraphIndexInteger> {
    /// The closed-form prefix lift.
    Prefix(prefix_lift::PrefixLiftView<'a, IndexType>),
    /// Heavy-path chain segment trees (the general backend).
    Segtree(LiftSegTreesView<'a, IndexType>),
}

impl<'a, IndexType: GraphIndexInteger + 'static> BlockCutLift<'a, IndexType> {
    /// The prefix backend, which every derived-`M_aug` block relies on.
    #[inline]
    pub(super) fn prefix(&self) -> &prefix_lift::PrefixLiftView<'a, IndexType> {
        match &self.backend {
            BackendView::Prefix(p) => p,
            _ => unreachable!("derived M_aug exists only under the prefix lift"),
        }
    }

    /// Lift `vector` (at cut node `from`) up to its cut ancestor `to`.
    pub(super) fn lift_up_vector(
        &self,
        from: usize,
        to: usize,
        vector: [MinPlus<IndexType>; 2],
    ) -> [MinPlus<IndexType>; 2] {
        match &self.backend {
            BackendView::Prefix(p) => p.lift_up_vector(from, to, vector),
            BackendView::Segtree(t) => t.lift_up_vector(from, to, vector),
        }
    }

    /// Push `vector` (at cut ancestor `from`) down to its cut descendant `to`.
    pub(super) fn lift_down_vector(
        &self,
        from: usize,
        to: usize,
        vector: [MinPlus<IndexType>; 2],
    ) -> [MinPlus<IndexType>; 2] {
        match &self.backend {
            BackendView::Prefix(p) => p.lift_down_vector(from, to, vector),
            BackendView::Segtree(t) => t.lift_down_vector(from, to, vector),
        }
    }
}
