//! Heavy-path chains with min-plus segment trees: the general path-product backend of
//! [`crate::block_cut_lift`] (the closed-form [`crate::prefix_lift`] takes over when its gate
//! passes).
//!
//! The cut forest is split into heavy chains; each chain keeps its step matrices (the `2x2` of
//! each cut's step to its parent cut) in two segment trees, one per direction, and a lift leg
//! multiplies `O(log chain)` tree nodes per chain it crosses. Min-plus has no inverse (so no prefix
//! products of general steps) and `⊗` is not idempotent (so no sparse tables), but a segment tree
//! needs only a monoid, and `(Mat2, ⊗, I)` is one (as in [`crate::s_node_segtree`], over S-node
//! cycles). The block-cut forest of a pangenome is very nearly a path, about one heavy chain per
//! chromosome, so a cut path rarely leaves its chain; a table costs `2 · 2n` matrices.
//!
//! # Composition order
//!
//! `Mat2::mul` is the standard matrix product and `left_multiply_vector(v)` is `v · M`, so the
//! order of `⊗` matters: on the up leg the deeper factor is on the left (descending), on the down
//! leg the shallower one (ascending).
//!
//! With a chain laid out top-first (`A[0]` shallowest) and `step[t]` the matrix of the edge from
//! `A[t]` to its parent, a path `A[j] -> A[i]` (`i < j`) therefore needs
//! `step[j] ⊗ step[j-1] ⊗ … ⊗ step[i+1]` for the up leg and the same range in the opposite order
//! for the down leg. One segment tree serves both, combining in reverse for the up leg.
//!
//! `step[0]` is the chain's **light edge** — the step out of `A[0]` into its parent chain — so it is
//! stored as an ordinary leaf and read as the range `[0, 0]`.
//!
//! [`LiftSegTrees`] decomposes the cut forest and builds the trees; the query reads them from the
//! image through [`LiftSegTreesView`].

use bidirected_adjacency_array::index::GraphIndexInteger;

use crate::{
    image::{Image, Sec},
    min_plus::{Mat2, MinPlus},
    sections,
};

/// Sentinel for "this cut is a forest root" in [`LiftSegTrees::cut_parent`].
const NO_PARENT: u32 = u32::MAX;

/// An iterative segment tree over a non-commutative monoid of `2x2` min-plus matrices: leaves at
/// `[n, 2n)` of a flat `2n` array, inner node `i` the product of its children `2i` and `2i + 1`
/// (see [`range_product`] for the query).
struct MatSegTree<IndexType: GraphIndexInteger> {
    tree: Vec<Mat2<IndexType>>,
}

impl<IndexType: GraphIndexInteger> MatSegTree<IndexType> {
    /// The tree over `leaves`; `reversed` for the up tree (see [`combine`]).
    fn build(leaves: Vec<Mat2<IndexType>>, reversed: bool) -> Self {
        let n = leaves.len();
        let mut tree = vec![Mat2::identity(); 2 * n.max(1)];
        if n == 0 {
            return Self { tree };
        }
        for (i, leaf) in leaves.into_iter().enumerate() {
            tree[n + i] = leaf;
        }
        for i in (1..n).rev() {
            tree[i] = combine(reversed, &tree[2 * i], &tree[2 * i + 1]);
        }
        Self { tree }
    }
}

/// `a ⊗ b`, or `b ⊗ a` for the up tree (`reversed`), so that a range reads back in descending
/// index order there.
#[inline]
fn combine<IndexType: GraphIndexInteger>(
    reversed: bool,
    a: &Mat2<IndexType>,
    b: &Mat2<IndexType>,
) -> Mat2<IndexType> {
    if reversed { b.mul(a) } else { a.mul(b) }
}

/// The product over leaves `[lo, hi]` (identity when `lo > hi`) of a flat `2n` segment-tree array,
/// bottom-up. The left and right frontiers accumulate separately and combine once at the end,
/// which keeps the non-commutative `⊗` in order.
#[inline]
fn range_product<IndexType: GraphIndexInteger>(
    tree: &[Mat2<IndexType>],
    n: usize,
    reversed: bool,
    lo: usize,
    hi: usize,
) -> Mat2<IndexType> {
    if n == 0 || lo > hi {
        return Mat2::identity();
    }
    let mut left = Mat2::identity();
    let mut right = Mat2::identity();
    let mut l = lo + n;
    let mut r = hi + 1 + n;
    while l < r {
        if l & 1 == 1 {
            left = combine(reversed, &left, &tree[l]);
            l += 1;
        }
        if r & 1 == 1 {
            r -= 1;
            right = combine(reversed, &tree[r], &right);
        }
        l >>= 1;
        r >>= 1;
    }
    combine(reversed, &left, &right)
}

/// Typed handles to a persisted [`LiftSegTrees`].
#[derive(Clone, Copy, Debug)]
pub struct SegSections<IndexType: GraphIndexInteger> {
    chain_of: Sec<u32>,
    index_in_chain: Sec<u32>,
    chain_top: Sec<u32>,
    cut_parent: Sec<u32>,
    /// `chains + 1` entries: the start of each chain's `2n` matrices in `up`/`down`.
    tree_offsets: Sec<u64>,
    up: Sec<Mat2<IndexType>>,
    down: Sec<Mat2<IndexType>>,
}

impl<IndexType: GraphIndexInteger> SegSections<IndexType> {
    pub fn from_image(image: &Image) -> std::io::Result<Self> {
        Ok(Self {
            chain_of: image.require(sections::LIFT_SEG_CHAIN_OF, "lift.seg_chain_of")?,
            index_in_chain: image
                .require(sections::LIFT_SEG_INDEX_IN_CHAIN, "lift.seg_index_in_chain")?,
            chain_top: image.require(sections::LIFT_SEG_CHAIN_TOP, "lift.seg_chain_top")?,
            cut_parent: image.require(sections::LIFT_SEG_CUT_PARENT, "lift.seg_cut_parent")?,
            tree_offsets: image
                .require(sections::LIFT_SEG_TREE_OFFSETS, "lift.seg_tree_offsets")?,
            up: image.require(sections::LIFT_SEG_UP, "lift.seg_up")?,
            down: image.require(sections::LIFT_SEG_DOWN, "lift.seg_down")?,
        })
    }

    pub fn view<'a>(&self, image: &'a Image) -> LiftSegTreesView<'a, IndexType> {
        LiftSegTreesView {
            chain_of: image.get(self.chain_of),
            index_in_chain: image.get(self.index_in_chain),
            chain_top: image.get(self.chain_top),
            cut_parent: image.get(self.cut_parent),
            tree_offsets: image.get(self.tree_offsets),
            up: image.get(self.up),
            down: image.get(self.down),
        }
    }
}

/// The image-backed [`LiftSegTrees`]: per-chain `2n` matrix arrays concatenated.
#[derive(Clone, Copy)]
pub struct LiftSegTreesView<'a, IndexType: GraphIndexInteger> {
    chain_of: &'a [u32],
    index_in_chain: &'a [u32],
    chain_top: &'a [u32],
    cut_parent: &'a [u32],
    tree_offsets: &'a [u64],
    up: &'a [Mat2<IndexType>],
    down: &'a [Mat2<IndexType>],
}

impl<'a, IndexType: GraphIndexInteger> LiftSegTreesView<'a, IndexType> {
    #[inline]
    fn chain_n(&self, chain: usize) -> usize {
        ((self.tree_offsets[chain + 1] - self.tree_offsets[chain]) / 2) as usize
    }

    #[inline]
    fn up_range(&self, chain: usize, lo: usize, hi: usize) -> Mat2<IndexType> {
        let (s, e) = (
            self.tree_offsets[chain] as usize,
            self.tree_offsets[chain + 1] as usize,
        );
        range_product(&self.up[s..e], self.chain_n(chain), true, lo, hi)
    }

    #[inline]
    fn down_range(&self, chain: usize, lo: usize, hi: usize) -> Mat2<IndexType> {
        let (s, e) = (
            self.tree_offsets[chain] as usize,
            self.tree_offsets[chain + 1] as usize,
        );
        range_product(&self.down[s..e], self.chain_n(chain), false, lo, hi)
    }

    /// Lift `vector` at cut `from` up to its cut ancestor `to`, as `v · Π step_up`.
    pub fn lift_up_vector(
        &self,
        from: usize,
        to: usize,
        vector: [MinPlus<IndexType>; 2],
    ) -> [MinPlus<IndexType>; 2] {
        self.up_product(from, to).left_multiply_vector(&vector)
    }

    /// Push `vector` at cut ancestor `from` down to its descendant `to`.
    pub fn lift_down_vector(
        &self,
        from: usize,
        to: usize,
        vector: [MinPlus<IndexType>; 2],
    ) -> [MinPlus<IndexType>; 2] {
        self.down_product(from, to).left_multiply_vector(&vector)
    }

    fn up_product(&self, from: usize, to: usize) -> Mat2<IndexType> {
        let mut matrix = Mat2::identity();
        let mut cur = from;
        while self.chain_of[cur] != self.chain_of[to] {
            let chain = self.chain_of[cur] as usize;
            let t = self.index_in_chain[cur] as usize;
            matrix = matrix.mul(&self.up_range(chain, 1, t));
            matrix = matrix.mul(&self.up_range(chain, 0, 0));
            let top = self.chain_top[chain] as usize;
            let parent = self.cut_parent[top];
            debug_assert_ne!(
                parent, NO_PARENT,
                "up_product left the forest before reaching `to`"
            );
            cur = parent as usize;
        }
        let chain = self.chain_of[cur] as usize;
        let (hi, lo) = (
            self.index_in_chain[cur] as usize,
            self.index_in_chain[to] as usize,
        );
        if hi > lo {
            matrix = matrix.mul(&self.up_range(chain, lo + 1, hi));
        }
        matrix
    }

    fn down_product(&self, from: usize, to: usize) -> Mat2<IndexType> {
        let mut matrix = Mat2::identity();
        let mut cur = to;
        while self.chain_of[cur] != self.chain_of[from] {
            let chain = self.chain_of[cur] as usize;
            let t = self.index_in_chain[cur] as usize;
            matrix = self.down_range(chain, 1, t).mul(&matrix);
            matrix = self.down_range(chain, 0, 0).mul(&matrix);
            let top = self.chain_top[chain] as usize;
            let parent = self.cut_parent[top];
            debug_assert_ne!(
                parent, NO_PARENT,
                "down_product left the forest before reaching `from`"
            );
            cur = parent as usize;
        }
        let chain = self.chain_of[cur] as usize;
        let (hi, lo) = (
            self.index_in_chain[cur] as usize,
            self.index_in_chain[from] as usize,
        );
        if hi > lo {
            matrix = self.down_range(chain, lo + 1, hi).mul(&matrix);
        }
        matrix
    }
}

/// Heavy-path chains over the cut-node forest, each carrying an up and a down segment tree (the
/// build; the query reads them through [`LiftSegTreesView`]).
pub struct LiftSegTrees<IndexType: GraphIndexInteger> {
    /// Cut -> its chain.
    chain_of: Vec<u32>,
    /// Cut -> its position within that chain (`0` = shallowest).
    index_in_chain: Vec<u32>,
    /// Chain -> the cut at position `0`.
    chain_top: Vec<u32>,
    /// Cut -> parent cut, or [`NO_PARENT`].
    cut_parent: Vec<u32>,
    up: Vec<MatSegTree<IndexType>>,
    down: Vec<MatSegTree<IndexType>>,
}

impl<IndexType: GraphIndexInteger> LiftSegTrees<IndexType> {
    /// Decompose the cut-node forest into heavy paths and build the per-chain trees.
    ///
    /// `cut_parent_opt[c]` is `c`'s parent cut (`None` at a forest root). `cut_order` must list
    /// parents before children. `step_up[c]` / `step_down[c]` are the matrices of the step between
    /// `c` and its parent cut.
    pub fn build(
        number_of_cuts: usize,
        cut_parent_opt: &[Option<usize>],
        cut_order: &[usize],
        step_up: &[Mat2<IndexType>],
        step_down: &[Mat2<IndexType>],
    ) -> Self {
        let mut cut_parent = vec![NO_PARENT; number_of_cuts];
        for (c, parent) in cut_parent_opt.iter().enumerate().take(number_of_cuts) {
            if let Some(p) = parent {
                cut_parent[c] = *p as u32;
            }
        }
        let heavy = heavy_children(&cut_parent, cut_order);
        let chains = Chains::of(&cut_parent, &heavy, cut_order);
        let leaves = |steps: &[Mat2<IndexType>], members: &[usize]| -> Vec<Mat2<IndexType>> {
            members.iter().map(|&c| steps[c]).collect()
        };
        let up = (chains.members.iter())
            .map(|members| MatSegTree::build(leaves(step_up, members), true))
            .collect();
        let down = (chains.members.iter())
            .map(|members| MatSegTree::build(leaves(step_down, members), false))
            .collect();
        Self {
            chain_of: chains.chain_of,
            index_in_chain: chains.index_in_chain,
            chain_top: chains.chain_top,
            cut_parent,
            up,
            down,
        }
    }

    /// Write the trees into `image` as sections.
    pub fn write(&self, image: &Image) -> SegSections<IndexType> {
        let mut tree_offsets = Vec::with_capacity(self.up.len() + 1);
        let mut total = 0u64;
        tree_offsets.push(0u64);
        for t in &self.up {
            total += t.tree.len() as u64;
            tree_offsets.push(total);
        }
        let (up_sec, up_out) =
            image.alloc_section::<Mat2<IndexType>>(sections::LIFT_SEG_UP, total as usize);
        let (down_sec, down_out) =
            image.alloc_section::<Mat2<IndexType>>(sections::LIFT_SEG_DOWN, total as usize);
        for (c, (u, d)) in self.up.iter().zip(&self.down).enumerate() {
            let (s, e) = (tree_offsets[c] as usize, tree_offsets[c + 1] as usize);
            debug_assert_eq!(u.tree.len(), d.tree.len());
            up_out[s..e].copy_from_slice(&u.tree);
            down_out[s..e].copy_from_slice(&d.tree);
        }
        SegSections {
            chain_of: image.push_section(sections::LIFT_SEG_CHAIN_OF, &self.chain_of),
            index_in_chain: image
                .push_section(sections::LIFT_SEG_INDEX_IN_CHAIN, &self.index_in_chain),
            chain_top: image.push_section(sections::LIFT_SEG_CHAIN_TOP, &self.chain_top),
            cut_parent: image.push_section(sections::LIFT_SEG_CUT_PARENT, &self.cut_parent),
            tree_offsets: image.push_section(sections::LIFT_SEG_TREE_OFFSETS, &tree_offsets),
            up: up_sec,
            down: down_sec,
        }
    }
}

/// The heavy child of every cut (the child with the largest subtree, ties to the lowest id;
/// [`NO_PARENT`] at a leaf), from subtree sizes summed children-before-parents.
fn heavy_children(cut_parent: &[u32], cut_order: &[usize]) -> Vec<u32> {
    let n = cut_parent.len();
    let mut subtree = vec![1usize; n];
    for &c in cut_order.iter().rev() {
        if cut_parent[c] != NO_PARENT {
            subtree[cut_parent[c] as usize] += subtree[c];
        }
    }
    let mut heavy = vec![NO_PARENT; n];
    for (c, &p) in cut_parent.iter().enumerate() {
        if p == NO_PARENT {
            continue;
        }
        let cur = heavy[p as usize];
        if cur == NO_PARENT || subtree[c] > subtree[cur as usize] {
            heavy[p as usize] = c as u32;
        }
    }
    heavy
}

/// The heavy-path chains of the cut forest: a cut continues its parent's chain iff it is the
/// parent's heavy child.
struct Chains {
    chain_of: Vec<u32>,
    index_in_chain: Vec<u32>,
    chain_top: Vec<u32>,
    /// Per chain: its cuts, top first.
    members: Vec<Vec<usize>>,
}

impl Chains {
    fn of(cut_parent: &[u32], heavy: &[u32], cut_order: &[usize]) -> Self {
        let n = cut_parent.len();
        let mut chains = Self {
            chain_of: vec![NO_PARENT; n],
            index_in_chain: vec![0u32; n],
            chain_top: Vec::new(),
            members: Vec::new(),
        };
        for &c in cut_order {
            let p = cut_parent[c];
            if p != NO_PARENT && heavy[p as usize] == c as u32 {
                let chain = chains.chain_of[p as usize];
                chains.chain_of[c] = chain;
                chains.index_in_chain[c] = chains.members[chain as usize].len() as u32;
                chains.members[chain as usize].push(c);
            } else {
                chains.chain_of[c] = chains.members.len() as u32;
                chains.index_in_chain[c] = 0;
                chains.chain_top.push(c as u32);
                chains.members.push(vec![c]);
            }
        }
        chains
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mp(v: u32) -> MinPlus<u32> {
        MinPlus::finite_usize(v as usize)
    }

    /// A deterministic pseudo-random `2x2` with finite entries.
    fn mat(seed: u32) -> Mat2<u32> {
        let mut m = Mat2::identity();
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for i in 0..2 {
            for j in 0..2 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                m.set(i, j, mp((s >> 20) % 50 + 1));
            }
        }
        m
    }

    /// Reference: compose the base steps one edge at a time, exactly as the doubling recurrence
    /// defines the composed tables.
    fn naive_up(parent: &[Option<usize>], step: &[Mat2<u32>], from: usize, to: usize) -> Mat2<u32> {
        let mut m = Mat2::identity();
        let mut c = from;
        while c != to {
            m = m.mul(&step[c]);
            c = parent[c].expect("`to` must be an ancestor of `from`");
        }
        m
    }

    fn naive_down(
        parent: &[Option<usize>],
        step: &[Mat2<u32>],
        from: usize,
        to: usize,
    ) -> Mat2<u32> {
        let mut m = Mat2::identity();
        let mut c = to;
        while c != from {
            m = step[c].mul(&m);
            c = parent[c].expect("`from` must be an ancestor of `to`");
        }
        m
    }

    /// Builds the trees of a forest, then checks every ancestor pair of the image view against the
    /// naive composition.
    fn check(parent: Vec<Option<usize>>) {
        let n = parent.len();
        let order: Vec<usize> = {
            // Parents before children: repeatedly emit any cut whose parent is already emitted.
            let mut out = Vec::new();
            let mut done = vec![false; n];
            while out.len() < n {
                for c in 0..n {
                    if !done[c] && parent[c].is_none_or(|p| done[p]) {
                        done[c] = true;
                        out.push(c);
                    }
                }
            }
            out
        };
        let step_up: Vec<Mat2<u32>> = (0..n).map(|c| mat(c as u32 + 1)).collect();
        let step_down: Vec<Mat2<u32>> = (0..n).map(|c| mat(c as u32 + 9001)).collect();
        let trees = LiftSegTrees::build(n, &parent, &order, &step_up, &step_down);
        let image = Image::new_arena(1 << 20).unwrap();
        let t = trees.write(&image).view(&image);
        let ancestors = |mut c: usize| {
            let mut out = vec![c];
            while let Some(p) = parent[c] {
                out.push(p);
                c = p;
            }
            out
        };
        let mut pairs = 0;
        for from in 0..n {
            for to in ancestors(from) {
                assert_eq!(
                    t.up_product(from, to),
                    naive_up(&parent, &step_up, from, to),
                    "up {from} -> {to}",
                );
                assert_eq!(
                    t.down_product(to, from),
                    naive_down(&parent, &step_down, to, from),
                    "down {to} -> {from}",
                );
                pairs += 1;
            }
        }
        assert!(
            pairs >= n,
            "expected at least one pair per cut, got {pairs}"
        );
    }

    #[test]
    fn single_chain_matches_naive() {
        // 0 <- 1 <- 2 <- ... <- 39: one chain, the pangenome case.
        let parent = (0..40)
            .map(|c| if c == 0 { None } else { Some(c - 1) })
            .collect();
        check(parent);
    }

    #[test]
    fn binary_tree_matches_naive() {
        // Maximally bushy: forces many light edges, so the chain-crossing path is exercised.
        let parent = (0..31usize)
            .map(|c| if c == 0 { None } else { Some((c - 1) / 2) })
            .collect();
        check(parent);
    }

    #[test]
    fn caterpillar_and_forest_match_naive() {
        // A spine with a leaf hanging off each vertex, plus a second component.
        let mut parent: Vec<Option<usize>> = vec![None];
        for i in 1..12 {
            parent.push(Some(i - 1)); // spine
            parent.push(Some(i)); // leaf off the spine
        }
        parent.push(None); // second component root
        let root2 = parent.len() - 1;
        parent.push(Some(root2));
        check(parent);
    }

    #[test]
    fn zero_length_path_is_identity() {
        let parent: Vec<Option<usize>> = vec![None, Some(0), Some(1)];
        let order = vec![0, 1, 2];
        let step_up: Vec<Mat2<u32>> = (0..3).map(|c| mat(c as u32 + 1)).collect();
        let step_down: Vec<Mat2<u32>> = (0..3).map(|c| mat(c as u32 + 77)).collect();
        let trees = LiftSegTrees::build(3, &parent, &order, &step_up, &step_down);
        let image = Image::new_arena(1 << 20).unwrap();
        let t = trees.write(&image).view(&image);
        for c in 0..3 {
            assert_eq!(t.up_product(c, c), Mat2::identity(), "up {c} -> {c}");
            assert_eq!(t.down_product(c, c), Mat2::identity(), "down {c} -> {c}");
        }
    }
}
