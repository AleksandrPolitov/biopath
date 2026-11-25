//! The engine's block numbering (chain order) against the decomposition's.

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::GraphIndexInteger,
    io::gfa1::{GfaEdgeData, GfaNodeData},
};
use spqr_tree::decomposition::{SPQRDecomposition, indices::BlockIndex};

use super::{incidence::BlockCutIncidence, layout::block_cut_layout, offset};
use crate::chain_nav::ChainLayout;

/// The block numbering under which the chain layout reads blocks arithmetically: every component's
/// root block first (in decomposition order), then the heavy child block of each cut node in dense
/// cut-id order (cut `k`'s heavy child becomes block `roots + k`), then every other block in
/// decomposition order.
///
/// The layout's id-based rules keep the same tree under this numbering: a component's root is its
/// lowest block id (roots come first), a heavy child wins ties as the lowest id among its siblings
/// (light children come after every heavy child) and block-headed chains keep their relative order.
/// [`super::TopologySections::write`] checks the result.
pub fn chain_block_numbering<IndexType, NodeData, EdgeData>(
    spqr: &SPQRDecomposition<'_, BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>>,
) -> BlockNumbering
where
    IndexType: GraphIndexInteger,
    NodeData: GfaNodeData,
    EdgeData: GfaEdgeData,
{
    let incidence = BlockCutIncidence::read(spqr, BlockIndex::from_usize);
    let layout = block_cut_layout(&incidence);
    BlockNumbering::from_new_of_old(chain_order(
        &layout,
        incidence.block_count(),
        incidence.cut_count(),
    ))
}

/// `new_of_old` of the chain block numbering of the `nb` blocks under `layout` (see
/// [`chain_block_numbering`]).
fn chain_order(layout: &ChainLayout, nb: usize, nc: usize) -> Vec<usize> {
    let super_root = (nb + nc) as u32;
    let mut new_of_old = vec![usize::MAX; nb];
    let mut next = 0usize;
    for (b, slot) in new_of_old.iter_mut().enumerate() {
        if layout.parent[b] == super_root {
            *slot = next;
            next += 1;
        }
    }
    let roots = next;
    for (k, &tmp) in layout.cut_old_of_new.iter().enumerate() {
        let heavy = layout.heavy[nb + tmp as usize] as usize;
        assert!(heavy < nb, "every cut node has a child block");
        new_of_old[heavy] = roots + k;
    }
    next = roots + nc;
    for slot in new_of_old.iter_mut().filter(|s| **s == usize::MAX) {
        *slot = next;
        next += 1;
    }
    assert_eq!(next, nb, "chain_block_numbering: not a permutation");
    new_of_old
}

/// The root blocks `R` when the blocks of `layout` are in chain numbering: the component roots
/// are blocks `0..R` and cut `k`'s heavy child is block `R + k`.
pub(super) fn chain_numbering_roots(layout: &ChainLayout, nb: usize) -> Option<usize> {
    let super_root = (nb + layout.cut_old_of_new.len()) as u32;
    let roots = (0..nb)
        .take_while(|&b| layout.parent[b] == super_root)
        .count();
    let chain_numbered = (roots..nb).all(|b| layout.parent[b] != super_root)
        && layout
            .cut_old_of_new
            .iter()
            .enumerate()
            .all(|(k, &tmp)| layout.heavy[nb + tmp as usize] as usize == roots + k);
    chain_numbered.then_some(roots)
}

/// The engine's block ids against the decomposition's: the engine numbers blocks its own way (see
/// [`chain_block_numbering`]) and translates only where it reads the decomposition (the topology
/// build and the block-tree extraction); everything downstream sees engine ids only.
#[derive(Clone, Debug)]
pub struct BlockNumbering {
    new_of_old: Vec<u32>,
    old_of_new: Vec<u32>,
}

impl BlockNumbering {
    /// The permutation `old -> new_of_old[old]`.
    pub fn from_new_of_old(new_of_old: Vec<usize>) -> Self {
        let mut old_of_new = vec![u32::MAX; new_of_old.len()];
        for (old, &new) in new_of_old.iter().enumerate() {
            assert_eq!(
                old_of_new[new],
                u32::MAX,
                "BlockNumbering: not a permutation"
            );
            old_of_new[new] = offset(old);
        }
        Self {
            new_of_old: new_of_old.into_iter().map(offset).collect(),
            old_of_new,
        }
    }

    /// The engine id of a decomposition block.
    #[inline]
    pub fn new_id<IndexType: GraphIndexInteger>(
        &self,
        old: BlockIndex<IndexType>,
    ) -> BlockIndex<IndexType> {
        BlockIndex::from_usize(self.new_of_old[old.into_usize()] as usize)
    }

    /// The decomposition block of an engine id.
    #[inline]
    pub fn old_id<IndexType: GraphIndexInteger>(&self, new: usize) -> BlockIndex<IndexType> {
        BlockIndex::from_usize(self.old_of_new[new] as usize)
    }

    pub fn len(&self) -> usize {
        self.new_of_old.len()
    }

    pub fn is_empty(&self) -> bool {
        self.new_of_old.is_empty()
    }
}
