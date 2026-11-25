//! Flip closures: per cut `c`, the cheapest way between its two orientations (`c^+ ⇝ c^-`) through
//! the rest of the graph, as a `2x2` min-plus matrix with a zero diagonal.
//!
//! - `down[c]` goes through the pendant subtree below `c`, bottom-up: a block's `gadget_up` is the
//!   diagonal at its parent cut of its gadget with its child cuts' `down` as self-loops, and a
//!   cut's `down` is the closure of its child blocks' `gadget_up`.
//! - `up[c]` goes through everything else, top-down ([`full_closures`]).
//! - `full[c] = close(down[c] min up[c])` is the whole-graph flip closure, folded into every
//!   `M_aug` and stored in the cut records.
//!
//! The bottom-up half runs during the per-block build ([`FlipBaker`]), children first, so that
//! every record block's engine is built with its child cuts' `down[c]` baked in as `c^a -> c^b`
//! edges: the engine answers flips through its child cuts itself, and only its parent cut's flip
//! stays explicit. The lift starts from the result ([`BakedForest`]).

use std::collections::HashMap;

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};
use rayon::prelude::*;
use spqr_tree::decomposition::indices::BlockIndex;

use super::{
    CutMatrixFill, IntraDistance, cut_matrix::raw_cut_matrix, forest::BlockCutForest,
    gadget::GadgetDijkstra,
};
use crate::{
    fast_hash::FastMap,
    min_plus::{Mat2, MinPlus},
    node_rows::BakedArc,
    topology::NodeTopology,
};

/// The block-cut forest as the flip-baking build leaves it, handed to [`super::BlockCutLift::new`]:
/// every block's raw cut-to-cut matrix `M_B` and the bottom-up half of the flip closures.
/// `baked[b]` marks a block whose engine was built with every child cut's pendant flip closure
/// `down[c]` baked in: such a block's explicit flip list keeps only its parent cut, and its flip rows
/// are baked distances.
pub struct BakedForest<IndexType: GraphIndexInteger> {
    pub(super) forest: BlockCutForest<IndexType>,
    /// Per block: its raw `M_B` over `forest.block_cuts[b]` (`(2c)^2` row-major, zero diagonal).
    pub(super) raw: Vec<Vec<MinPlus<IndexType>>>,
    pub(super) baked: Vec<bool>,
    /// Per cut: the flip closure `down[c]` through its pendant subtree (the blocks below it).
    pub(super) down: Vec<Mat2<IndexType>>,
    /// Per block: its subtree's `2x2` at its parent cut (`+∞` at a component root).
    pub(super) gadget_up: Vec<Mat2<IndexType>>,
}

/// The bottom-up flip pass of the per-block build (see the module docs).
///
/// Blocks are finished in `BlockCutForest::blocks_postorder` order. Before a block is built,
/// [`Self::child_closures`] closes `down[c]` for its child cuts (over blocks that are all
/// finished). After it is built, [`Self::finish_block`] reads its raw cut-to-cut matrix `M_B`
/// through the block's own (baked) engine — a record-less small block through the block-restricted
/// Dijkstra, [`Self::finish_block_raw`] — and folds its `gadget_up` toward the parent cut.
pub struct FlipBaker<IndexType: GraphIndexInteger> {
    pub forest: BlockCutForest<IndexType>,
    raw: Vec<Vec<MinPlus<IndexType>>>,
    gadget_up: Vec<Mat2<IndexType>>,
    down: Vec<Mat2<IndexType>>,
    down_done: Vec<bool>,
    finished: Vec<bool>,
    baked: Vec<bool>,
    gadget: GadgetDijkstra<IndexType>,
}

impl<IndexType: GraphIndexInteger + Send + Sync> FlipBaker<IndexType> {
    pub fn new(topology: &NodeTopology<'_, IndexType>) -> Self {
        let forest = BlockCutForest::new(topology);
        let (nb, nc) = (forest.number_of_blocks, forest.number_of_cuts);
        Self {
            forest,
            raw: vec![Vec::new(); nb],
            gadget_up: vec![Mat2::infinity(); nb],
            down: vec![Mat2::infinity(); nc],
            down_done: vec![false; nc],
            finished: vec![false; nb],
            baked: vec![false; nb],
            gadget: GadgetDijkstra::new(),
        }
    }

    fn close_down(&mut self, cut: usize) {
        if self.down_done[cut] {
            return;
        }
        let mut combined = Mat2::infinity();
        for &child_block in self.forest.cut_child_blocks(cut) {
            assert!(
                self.finished[child_block],
                "flip baking: block {child_block} below cut {cut} must be finished before the block above it"
            );
            combined = combined.elementwise_min(&self.gadget_up[child_block]);
        }
        self.down[cut] = combined.closure();
        self.down_done[cut] = true;
    }

    /// Close `down[c]` for every child cut of `bid` and return the flip-capable ones (a finite
    /// orientation-changing entry), keyed by cut segment: the closures to bake into `bid`'s engine.
    pub fn child_closures(&mut self, bid: usize) -> FastMap<NodeIndex<IndexType>, Mat2<IndexType>> {
        let mut out = FastMap::default();
        let child_cuts: Vec<usize> = self.forest.child_cuts(bid).collect();
        for cut in child_cuts {
            self.close_down(cut);
            let d = self.down[cut];
            if can_flip(&d) {
                out.insert(self.forest.cut_segment[cut], d);
            }
        }
        out
    }

    /// Finish `bid` once its engine exists: its raw `M_B` and its `gadget_up` toward the parent
    /// cut. `baked` records that the engine carries the child closures of [`Self::child_closures`]
    /// (a record block); `intra` answers within-block start-to-start distances through that engine
    /// (a small block: the block-restricted Dijkstra).
    pub fn finish_block(
        &mut self,
        bid: usize,
        baked: bool,
        intra: &(impl IntraDistance<IndexType> + Sync),
        cut_matrix: &(impl CutMatrixFill<IndexType> + Sync),
    ) {
        let block_index = BlockIndex::from_usize(bid);
        let cuts = &self.forest.block_cuts[bid];
        let raw = raw_cut_matrix(bid, cuts, block_index, intra, cut_matrix);
        self.finish_block_raw(bid, baked, raw);
    }

    /// [`Self::finish_block`] for a block whose raw `M_B` the caller already has (over
    /// `forest.block_cuts[bid]`, `(2c)^2` row-major, diagonal zero): a small block's, from one
    /// block-restricted Dijkstra per directed cut.
    pub fn finish_block_raw(&mut self, bid: usize, baked: bool, raw: Vec<MinPlus<IndexType>>) {
        let cuts = &self.forest.block_cuts[bid];
        debug_assert_eq!(raw.len(), 4 * cuts.len() * cuts.len());
        let child_cuts: Vec<usize> = self.forest.child_cuts(bid).collect();
        for &cut in &child_cuts {
            self.close_down(cut);
        }
        if let Some(pc) = self.forest.parent_cut(bid) {
            let cuts = &self.forest.block_cuts[bid];
            let local_of = |segment: NodeIndex<IndexType>| {
                cuts.iter()
                    .position(|&c| c == segment)
                    .expect("a cut of the block")
            };
            let loops: Vec<(usize, Mat2<IndexType>)> = child_cuts
                .iter()
                .map(|&c| (local_of(self.forest.cut_segment[c]), self.down[c]))
                .collect();
            let parent_local = local_of(self.forest.cut_segment[pc]);
            self.gadget_up[bid] = self
                .gadget
                .diagonal(&raw, 2 * cuts.len(), &loops, parent_local);
        }
        self.raw[bid] = raw;
        self.baked[bid] = baked;
        self.finished[bid] = true;
    }

    /// The forest, raw matrices and bottom-up closures for [`super::BlockCutLift::new`] (every
    /// block must be finished, which closes `down[c]` of every cut).
    pub fn finish(self) -> BakedForest<IndexType> {
        assert!(
            self.finished.iter().all(|&f| f),
            "flip baking: every block is finished before the lift"
        );
        debug_assert!(
            self.down_done.iter().all(|&d| d),
            "every cut is the child cut of a finished block"
        );
        BakedForest {
            forest: self.forest,
            raw: self.raw,
            baked: self.baked,
            down: self.down,
            gadget_up: self.gadget_up,
        }
    }
}

/// Whether a flip closure lets its cut flip: a finite orientation-changing entry.
pub(super) fn can_flip<IndexType: GraphIndexInteger>(closure: &Mat2<IndexType>) -> bool {
    closure.get(0, 1).into_option().is_some() || closure.get(1, 0).into_option().is_some()
}

/// The whole-graph flip closure `full[c] = close(down[c] min up[c])` of every cut, from the
/// bottom-up `down` and `gadget_up` of the flip baking and the top-down pass `up` (see
/// [`TopDown`]), run level by level over the cuts by increasing depth (`cut_order`).
pub(super) fn full_closures<IndexType: GraphIndexInteger + Send + Sync>(
    forest: &BlockCutForest<IndexType>,
    raw: &[Vec<MinPlus<IndexType>>],
    positions: &[HashMap<NodeIndex<IndexType>, usize>],
    down: &[Mat2<IndexType>],
    gadget_up: &[Mat2<IndexType>],
    cut_order: &[usize],
) -> Vec<Mat2<IndexType>> {
    let pass = TopDown {
        forest,
        raw,
        positions,
        down,
        gadget_up,
    };
    let mut levels: Vec<Vec<usize>> = Vec::new();
    for &cut in cut_order {
        let depth = forest.cut_depth(cut);
        if depth >= levels.len() {
            levels.resize(depth + 1, Vec::new());
        }
        levels[depth].push(cut);
    }
    let mut up = vec![Mat2::infinity(); forest.number_of_cuts];
    for level in &levels {
        // One reusable gadget search per worker, not per cut.
        let computed: Vec<(usize, Mat2<IndexType>)> = level
            .par_iter()
            .map_init(GadgetDijkstra::new, |gadget, &cut| {
                (cut, pass.up_closure(gadget, &up, cut))
            })
            .collect();
        for (cut, value) in computed {
            up[cut] = value;
        }
    }
    (0..forest.number_of_cuts)
        .map(|c| down[c].elementwise_min(&up[c]).closure())
        .collect()
}

/// The top-down flip pass: `up[c]`, the closure of cut `c` through everything outside its
/// pendant subtree. For a cut below block `B` it is the diagonal at `c` of `B`'s gadget with the
/// other child cuts' `down` and, at `B`'s parent cut `p`, the environment `close(up[p] min
/// gadget_up of B's siblings under p)`. It reads `up` only of the cut two tree levels up, so the
/// cuts of one depth level are independent.
struct TopDown<'f, IndexType: GraphIndexInteger> {
    forest: &'f BlockCutForest<IndexType>,
    raw: &'f [Vec<MinPlus<IndexType>>],
    positions: &'f [HashMap<NodeIndex<IndexType>, usize>],
    down: &'f [Mat2<IndexType>],
    gadget_up: &'f [Mat2<IndexType>],
}

impl<IndexType: GraphIndexInteger> TopDown<'_, IndexType> {
    /// `up[cut]`, once `up` holds every shallower level. Only the cut's own `2x2` diagonal of the
    /// gadget is read, so [`GadgetDijkstra::diagonal`] runs two `O(dim^2)` searches from its poles
    /// instead of the `O(dim^3)` full closure.
    fn up_closure(
        &self,
        gadget: &mut GadgetDijkstra<IndexType>,
        up: &[Mat2<IndexType>],
        cut: usize,
    ) -> Mat2<IndexType> {
        let forest = self.forest;
        let Some(bid) = forest.parent_block(cut) else {
            return Mat2::infinity();
        };
        let parent_cut = forest.parent_cut(bid);
        // The environment at the block's parent cut: everything above it and the block's siblings
        // under it (leave-one-out over the parent cut's child blocks).
        let env = match parent_cut {
            None => Mat2::infinity(),
            Some(parent_cut) => {
                let mut combined = up[parent_cut];
                for &sibling_block in forest.cut_child_blocks(parent_cut) {
                    if sibling_block != bid {
                        combined = combined.elementwise_min(&self.gadget_up[sibling_block]);
                    }
                }
                combined.closure()
            }
        };
        let local = |c: usize| self.positions[bid][&forest.cut_segment[c]];
        let mut loops: Vec<(usize, Mat2<IndexType>)> = forest
            .child_cuts(bid)
            .filter(|&sibling| sibling != cut)
            .map(|sibling| (local(sibling), self.down[sibling]))
            .collect();
        if let Some(parent_cut) = parent_cut {
            loops.push((local(parent_cut), env));
        }
        let dim = 2 * forest.block_cuts[bid].len();
        gadget.diagonal(&self.raw[bid], dim, &loops, local(cut))
    }
}

/// Per block: its flip-capable cuts, in its cut-list order, with their flip closure through the
/// outside of the block: `down[c]` at a child cut (its pendant subtree), `full[c]` at the parent
/// cut (which may include a flip back through the block; that never beats the direct answer). A
/// cut that cannot flip outside the block cannot improve on an in-block distance, so the query
/// skips it. A flip-baked block keeps only its parent cut: its engine takes its child cuts' flips.
pub(super) fn block_flip_cuts<IndexType: GraphIndexInteger>(
    forest: &BlockCutForest<IndexType>,
    baked: &[bool],
    down: &[Mat2<IndexType>],
    full: &[Mat2<IndexType>],
) -> Vec<Vec<(NodeIndex<IndexType>, Mat2<IndexType>)>> {
    (0..forest.number_of_blocks)
        .map(|bid| {
            let parent_cut = forest.parent_cut(bid);
            forest.block_cuts[bid]
                .iter()
                .filter_map(|&segment| {
                    let cid = forest.cut_id[&segment];
                    if baked[bid] && Some(cid) != parent_cut {
                        return None;
                    }
                    let flip = if Some(cid) == parent_cut {
                        full[cid]
                    } else {
                        down[cid]
                    };
                    can_flip(&flip).then_some((segment, flip))
                })
                .collect()
        })
        .collect()
}

/// Per flip-baked block: the closures `down[c]` of its child cuts that can flip, as the arcs its
/// engine carries. Its flip rows are distances in that baked graph, matching its engine: the
/// parent-cut flip row then also covers routes that flip at a child cut first.
pub(super) fn baked_arcs<IndexType: GraphIndexInteger>(
    forest: &BlockCutForest<IndexType>,
    baked: &[bool],
    down: &[Mat2<IndexType>],
    down_flip: &[bool],
) -> Vec<Vec<BakedArc<IndexType>>> {
    (0..forest.number_of_blocks)
        .map(|bid| {
            if !baked[bid] {
                return Vec::new();
            }
            forest
                .child_cuts(bid)
                .filter(|&c| down_flip[c])
                .map(|c| {
                    let d = down[c];
                    (
                        forest.cut_segment[c],
                        [
                            [d.get(0, 0).to_wide(), d.get(0, 1).to_wide()],
                            [d.get(1, 0).to_wide(), d.get(1, 1).to_wide()],
                        ],
                    )
                })
                .collect()
        })
        .collect()
}
