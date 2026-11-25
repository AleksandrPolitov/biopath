//! The block-cut tree of a decomposition and its chain layout.

use std::collections::VecDeque;

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::incidence::BlockCutIncidence;
use crate::chain_nav::{self, ChainLayout};

/// The block-cut tree of `incidence` and its chain layout. Tree ids: `0..nb` the blocks,
/// `nb + tmp` the cut nodes (provisional ids), `nb + nc` a super-root over every component's
/// lowest block id (the same rule the lift roots its forest by).
pub(super) fn block_cut_layout<IndexType: GraphIndexInteger>(
    incidence: &BlockCutIncidence<IndexType>,
) -> ChainLayout {
    let (nb, nc) = (incidence.block_count(), incidence.cut_count());
    let reps = component_roots(incidence);
    let super_root = nb + nc;
    let tree = UndirectedCsr::build(super_root + 1, |edge| {
        for_each_incidence(incidence, &mut *edge);
        for &r in &reps {
            edge(r, super_root);
        }
    });
    let (parent, depth) = bfs_parents(&tree, super_root);
    drop(tree);
    let component_u32: Vec<u32> = incidence
        .block_component
        .iter()
        .map(|c| c.into_usize() as u32)
        .collect();
    ChainLayout::build(
        nb,
        nc,
        parent,
        depth,
        &component_u32,
        incidence.component_count as u32,
    )
}

/// Call `edge(block, nb + cut)` for every block-cut incidence, blocks in order and each block's
/// cuts in the decomposition's order.
fn for_each_incidence<IndexType: GraphIndexInteger>(
    incidence: &BlockCutIncidence<IndexType>,
    edge: &mut dyn FnMut(usize, usize),
) {
    let nb = incidence.block_count();
    for b in 0..nb {
        for &cut in incidence.block_cuts(b) {
            edge(b, nb + incidence.cut_tmp(cut));
        }
    }
}

/// The lowest block id of every component of the block-cut forest, ascending (a DFS from each
/// lowest unvisited block).
fn component_roots<IndexType: GraphIndexInteger>(
    incidence: &BlockCutIncidence<IndexType>,
) -> Vec<usize> {
    let nb = incidence.block_count();
    let total = nb + incidence.cut_count();
    let forest = UndirectedCsr::build(total, |edge| for_each_incidence(incidence, edge));
    let mut roots = Vec::new();
    let mut seen = vec![false; total];
    let mut stack = Vec::new();
    for start in 0..nb {
        if seen[start] {
            continue;
        }
        roots.push(start);
        seen[start] = true;
        stack.push(start);
        while let Some(v) = stack.pop() {
            for &w in forest.neighbors(v) {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    stack.push(w as usize);
                }
            }
        }
    }
    debug_assert!(seen.iter().all(|&x| x), "every cut node lies in a block");
    roots
}

/// BFS parent (the root is its own) and depth of every node of the tree `tree` from `root`.
fn bfs_parents(tree: &UndirectedCsr, root: usize) -> (Vec<u32>, Vec<u32>) {
    let total = tree.node_count();
    let mut parent = vec![chain_nav::NONE; total];
    let mut depth = vec![0u32; total];
    let mut queue = VecDeque::with_capacity(total);
    parent[root] = root as u32;
    queue.push_back(root);
    while let Some(v) = queue.pop_front() {
        for &w in tree.neighbors(v) {
            let w = w as usize;
            if parent[w] == chain_nav::NONE {
                parent[w] = v as u32;
                depth[w] = depth[v] + 1;
                queue.push_back(w);
            }
        }
    }
    debug_assert!(
        parent.iter().all(|&p| p != chain_nav::NONE),
        "the block-cut graph is a forest"
    );
    (parent, depth)
}

/// An undirected graph over `0..n` in CSR form: each edge in both lists, in edge order.
struct UndirectedCsr {
    offsets: Vec<u32>,
    targets: Vec<u32>,
}

impl UndirectedCsr {
    /// The graph of the edges `edges` reports (called twice: once to count, once to fill).
    fn build(n: usize, edges: impl Fn(&mut dyn FnMut(usize, usize))) -> Self {
        let mut offsets = vec![0u32; n + 1];
        edges(&mut |a, b| {
            offsets[a + 1] += 1;
            offsets[b + 1] += 1;
        });
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut targets = vec![0u32; offsets[n] as usize];
        let mut cursor = offsets.clone();
        edges(&mut |a, b| {
            targets[cursor[a] as usize] = b as u32;
            cursor[a] += 1;
            targets[cursor[b] as usize] = a as u32;
            cursor[b] += 1;
        });
        Self { offsets, targets }
    }

    fn node_count(&self) -> usize {
        self.offsets.len() - 1
    }

    fn neighbors(&self, v: usize) -> &[u32] {
        &self.targets[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}
