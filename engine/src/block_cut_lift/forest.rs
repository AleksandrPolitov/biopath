//! The block-cut forest: blocks and cut nodes alternating, rooted under one super-root that joins
//! the connected components.

use std::collections::HashMap;

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};
use spqr_tree::decomposition::indices::BlockIndex;

use crate::{
    rooted_tree::{RootedTree, TreeAccess},
    topology::NodeTopology,
};

/// The most cut nodes a block may have. A block's cut matrix `M_aug` is dense (`(2c)^2` entries,
/// filled by `(2c)^2` intra-block queries), so blocks above this are not supported: the build
/// fails on them.
pub const MAX_BLOCK_CUTS: usize = 4000;

/// The bipartite block-cut forest: tree ids `0..nb` are the blocks, `nb..nb + nc` the cut nodes
/// (dense cut ids are the topology's), `nb + nc` the super-root, whose children are the components'
/// lowest-id blocks. The flip baker builds it and the lift takes it over, so both agree on which
/// side of a cut is "down".
pub struct BlockCutForest<IndexType: GraphIndexInteger> {
    pub number_of_blocks: usize,
    pub number_of_cuts: usize,
    pub cut_segment: Vec<NodeIndex<IndexType>>,
    pub cut_id: HashMap<NodeIndex<IndexType>, usize>,
    pub block_cuts: Vec<Vec<NodeIndex<IndexType>>>,
    pub super_root: usize,
    pub tree: RootedTree,
}

impl<IndexType: GraphIndexInteger> BlockCutForest<IndexType> {
    /// The forest of `topology`'s blocks. Panics if a block has more than [`MAX_BLOCK_CUTS`] cut
    /// nodes.
    pub fn new(topology: &NodeTopology<'_, IndexType>) -> Self {
        let (cut_segment, cut_id, block_cuts) = incidence(topology);
        let (number_of_blocks, number_of_cuts) = (block_cuts.len(), cut_segment.len());
        let super_root = number_of_blocks + number_of_cuts;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (bid, cuts) in block_cuts.iter().enumerate() {
            for segment in cuts {
                edges.push((bid, number_of_blocks + cut_id[segment]));
            }
        }
        let roots = component_roots(super_root, number_of_blocks, &edges);
        edges.extend(roots.into_iter().map(|block| (super_root, block)));
        let tree = RootedTree::new(super_root + 1, &edges, super_root);
        Self {
            number_of_blocks,
            number_of_cuts,
            cut_segment,
            cut_id,
            block_cuts,
            super_root,
            tree,
        }
    }

    /// The parent cut (dense id) of block `bid`, `None` at a component root.
    pub fn parent_cut(&self, bid: usize) -> Option<usize> {
        let parent = self.tree.parent(bid)?;
        (parent != self.super_root).then(|| parent - self.number_of_blocks)
    }

    /// The parent block of cut `cut` (`None` only under the super-root, which has no cut children).
    pub(super) fn parent_block(&self, cut: usize) -> Option<usize> {
        let parent = self
            .tree
            .parent(self.number_of_blocks + cut)
            .expect("a cut node has a parent");
        (parent != self.super_root).then_some(parent)
    }

    /// The parent of cut `cut` in the cut forest: the parent cut of its parent block.
    pub(super) fn cut_parent(&self, cut: usize) -> Option<usize> {
        self.parent_block(cut).and_then(|bid| self.parent_cut(bid))
    }

    /// The child cuts (dense ids) of block `bid`.
    pub fn child_cuts(&self, bid: usize) -> impl Iterator<Item = usize> + '_ {
        self.tree
            .children(bid)
            .iter()
            .map(move |&t| t - self.number_of_blocks)
    }

    /// The child blocks of cut `cut`.
    pub fn cut_child_blocks(&self, cut: usize) -> &[usize] {
        self.tree.children(self.number_of_blocks + cut)
    }

    /// Every block id in postorder: each block after all blocks below its child cuts.
    pub fn blocks_postorder(&self) -> Vec<usize> {
        self.tree
            .postorder()
            .iter()
            .copied()
            .filter(|&t| t < self.number_of_blocks)
            .collect()
    }

    /// Every cut by increasing depth (ties in id order), so each comes after its cut ancestors.
    pub(super) fn cuts_by_depth(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.number_of_cuts).collect();
        order.sort_by_key(|&c| self.tree.depth(self.number_of_blocks + c));
        order
    }

    /// The tree depth of cut `cut`.
    pub(super) fn cut_depth(&self, cut: usize) -> usize {
        self.tree.depth(self.number_of_blocks + cut)
    }

    /// Per block: the position of each of its cuts in its cut list (the `M_B` / `M_aug` order).
    pub(super) fn cut_positions(&self) -> Vec<HashMap<NodeIndex<IndexType>, usize>> {
        self.block_cuts
            .iter()
            .map(|cuts| {
                cuts.iter()
                    .enumerate()
                    .map(|(i, &segment)| (segment, i))
                    .collect()
            })
            .collect()
    }
}

/// The topology's cut nodes (cut ids are the topology's, cut nodes in node order, so a query
/// endpoint's record names its cut tree node directly), their ids, and every block's cut list.
type Incidence<IndexType> = (
    Vec<NodeIndex<IndexType>>,
    HashMap<NodeIndex<IndexType>, usize>,
    Vec<Vec<NodeIndex<IndexType>>>,
);

/// The block-cut incidence of `topology`. Panics if a cut node is in no block's list or a block has
/// more than [`MAX_BLOCK_CUTS`] cuts.
fn incidence<IndexType: GraphIndexInteger>(
    topology: &NodeTopology<'_, IndexType>,
) -> Incidence<IndexType> {
    let cut_segment: Vec<NodeIndex<IndexType>> = topology.cut_nodes();
    let mut cut_id: HashMap<NodeIndex<IndexType>, usize> =
        HashMap::with_capacity(cut_segment.len());
    for (cid, &segment) in cut_segment.iter().enumerate() {
        cut_id.insert(segment, cid);
    }
    let mut block_cuts: Vec<Vec<NodeIndex<IndexType>>> = vec![Vec::new(); topology.block_count()];
    let mut incident = vec![false; cut_segment.len()];
    for (bid, cuts) in block_cuts.iter_mut().enumerate() {
        for segment in topology.block_cut_nodes(BlockIndex::from_usize(bid)) {
            incident[cut_id[&segment]] = true;
            cuts.push(segment);
        }
    }
    assert!(
        incident.iter().all(|&i| i),
        "every cut node of the topology must be listed as a cut of some block"
    );
    for (bid, cuts) in block_cuts.iter().enumerate() {
        assert!(
            cuts.len() <= MAX_BLOCK_CUTS,
            "block {bid} has {} cut nodes; at most {MAX_BLOCK_CUTS} are supported. This is a hard \
             limit of the index (the block-cut lift keeps a dense distance matrix over a block's \
             directed cut nodes): this graph cannot be indexed",
            cuts.len(),
        );
    }
    (cut_segment, cut_id, block_cuts)
}

/// The lowest-id block of every connected component of the block-cut graph `edges` over `nodes`
/// tree nodes (blocks `0..number_of_blocks` first), in component order: the blocks the super-root
/// links to.
fn component_roots(nodes: usize, number_of_blocks: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut adjacency = vec![Vec::new(); nodes];
    for &(a, b) in edges {
        adjacency[a].push(b);
        adjacency[b].push(a);
    }
    let mut visited = vec![false; nodes];
    let mut roots = Vec::new();
    for start in 0..nodes {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut root = (start < number_of_blocks).then_some(start);
        while let Some(node) = stack.pop() {
            if node < number_of_blocks {
                root.get_or_insert(node);
            }
            for &next in &adjacency[node] {
                if !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }
        roots.extend(root);
    }
    roots
}
