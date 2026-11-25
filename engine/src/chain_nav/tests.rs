use bidirected_adjacency_array::index::NodeIndex;
use spqr_tree::decomposition::indices::BlockIndex;

use super::{layout::ParentTree, *};
use crate::{
    image::Padded,
    min_plus::MinPlus,
    path::GfaPathLength,
    rooted_tree::{ChainTreeArrays, ChainTreeView},
};

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A random block-cut forest: blocks and cut nodes alternate, every cut node has >= 2 blocks,
/// several components, each component's root block is its lowest block id.
fn random_forest(seed: u64) -> (usize, usize, Vec<Vec<usize>>) {
    let mut s = seed | 1;
    let nb = 2 + (next(&mut s) % 40) as usize;
    let mut block_cuts: Vec<Vec<usize>> = vec![Vec::new(); nb];
    let mut nc = 0usize;
    // Blocks in id order; block 0 starts component 0; each later block either starts a new
    // component or attaches to an existing cut (as a light/heavy child) or to an existing block
    // through a new cut node.
    let mut cut_blocks: Vec<Vec<usize>> = Vec::new();
    for b in 1..nb {
        match next(&mut s) % 10 {
            0 => {} // new component
            1 | 2 if nc > 0 => {
                let c = (next(&mut s) % nc as u64) as usize;
                block_cuts[b].push(c);
                cut_blocks[c].push(b);
            }
            _ => {
                let p = (next(&mut s) % b as u64) as usize;
                let c = nc;
                nc += 1;
                cut_blocks.push(vec![p, b]);
                block_cuts[p].push(c);
                block_cuts[b].push(c);
            }
        }
    }
    (nb, nc, block_cuts)
}

/// Components (lowest block id first), the super-root edges and BFS parent/depth of a block-cut
/// forest, and its chain layout.
fn forest_layout(
    nb: usize,
    nc: usize,
    block_cuts: &[Vec<usize>],
) -> (ChainLayout, Vec<u32>, Vec<u32>) {
    let total = nb + nc + 1;
    let root = nb + nc;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); total];
    for (b, cuts) in block_cuts.iter().enumerate() {
        for &c in cuts {
            adj[b].push(nb + c);
            adj[nb + c].push(b);
        }
    }
    // Components: DFS from the lowest unvisited id (blocks first) → representative block.
    let mut comp = vec![usize::MAX; nb + nc];
    let mut block_component = vec![0u32; nb];
    let mut ncomp = 0usize;
    for start in 0..nb + nc {
        if comp[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        comp[start] = ncomp;
        let mut rep = None;
        while let Some(v) = stack.pop() {
            if v < nb {
                rep = Some(rep.map_or(v, |r: usize| r.min(v)));
                block_component[v] = ncomp as u32;
            }
            for &w in &adj[v] {
                if comp[w] == usize::MAX {
                    comp[w] = ncomp;
                    stack.push(w);
                }
            }
        }
        if let Some(r) = rep {
            adj[root].push(r);
            adj[r].push(root);
        }
        ncomp += 1;
    }
    // BFS parent/depth.
    let mut parent = vec![u32::MAX; total];
    let mut depth = vec![0u32; total];
    let mut seen = vec![false; total];
    let mut queue = std::collections::VecDeque::new();
    seen[root] = true;
    parent[root] = root as u32;
    queue.push_back(root);
    while let Some(v) = queue.pop_front() {
        for &w in &adj[v] {
            if !seen[w] {
                seen[w] = true;
                parent[w] = v as u32;
                depth[w] = depth[v] + 1;
                queue.push_back(w);
            }
        }
    }
    assert!(seen.iter().all(|&x| x), "a forest");
    (
        ChainLayout::build(
            nb,
            nc,
            parent.clone(),
            depth.clone(),
            &block_component,
            ncomp as u32,
        ),
        parent,
        depth,
    )
}

#[test]
fn navigation_matches_chain_tree_view() {
    for seed in 1..200u64 {
        let (nb, nc, roots, block_cuts) = chain_numbered_forest(seed);
        let total = nb + nc + 1;
        let (layout, parent, depth) = forest_layout(nb, nc, &block_cuts);
        assert_chain_numbered(&layout, nb, roots, seed);
        let arrays = ChainTreeArrays::build(&ParentTree {
            parent: &parent,
            depth: &depth,
            root: nb + nc,
            max_depth: depth.iter().copied().max().unwrap(),
        });
        let view = ChainTreeView {
            depth: &depth,
            parent: &arrays.parent,
            head: &arrays.head,
            heavy: &arrays.heavy,
        };
        let records = NavRecords::of(&layout, nb, nc, roots);
        let nav = records.nav(&layout, nb, nc, roots);
        // The records are in the NEW cut order; old tree ids map to new ones.
        let new_id = |old: usize| -> usize {
            if old >= nb && old < nb + nc {
                nb + layout.cut_new_id[old - nb] as usize
            } else {
                old
            }
        };
        for (v, &depth_v) in depth.iter().enumerate() {
            assert_eq!(
                nav.tpos(new_id(v)),
                layout.tpos[v],
                "seed {seed}: tpos of {v}"
            );
            assert_eq!(nav.depth(new_id(v)), depth_v, "seed {seed}: depth of {v}");
            assert_eq!(
                nav.parent(new_id(v)),
                view.parent(v).map(new_id),
                "seed {seed}: parent of {v}"
            );
            let heavy = (arrays.heavy[v] != u32::MAX).then(|| new_id(arrays.heavy[v] as usize));
            assert_eq!(nav.heavy(new_id(v)), heavy, "seed {seed}: heavy of {v}");
            if v < nb {
                assert_eq!(
                    nav.chain_of_block(v),
                    layout.chain_of[v] as usize,
                    "seed {seed}: chain of block {v}"
                );
            }
            if v >= nb && v < nb + nc {
                let mut want: Vec<usize> = records.children[v - nb].clone();
                want.sort_unstable();
                let mut got: Vec<usize> = nav
                    .cut_child_blocks(new_id(v) - nb)
                    .map(|b| b.into_usize())
                    .collect();
                got.sort_unstable();
                assert_eq!(got, want, "seed {seed}: children of cut {v}");
                assert_eq!(nav.cut_depth(new_id(v) - nb), (depth_v - 2) / 2);
            }
            for w in 0..total {
                assert_eq!(
                    nav.lca(new_id(v), new_id(w)),
                    new_id(view.lca(v, w)),
                    "seed {seed}: lca {v} {w}"
                );
                assert_eq!(
                    nav.child_on_path_to(new_id(v), new_id(w)),
                    view.child_on_path_to(v, w).map(new_id),
                    "seed {seed}: child_on_path_to {v} {w}"
                );
            }
        }
    }
}

/// A random forest renumbered into the chain block numbering (as `chain_block_numbering` derives
/// it): `(nb, nc, roots, block_cuts)`.
fn chain_numbered_forest(seed: u64) -> (usize, usize, usize, Vec<Vec<usize>>) {
    let (nb, nc, old_block_cuts) = random_forest(seed * 7919);
    let root = nb + nc;
    let (old_layout, _, _) = forest_layout(nb, nc, &old_block_cuts);
    let mut new_of_old = vec![usize::MAX; nb];
    let mut roots = 0usize;
    for (b, slot) in new_of_old.iter_mut().enumerate() {
        if old_layout.parent[b] as usize == root {
            *slot = roots;
            roots += 1;
        }
    }
    for (k, &tmp) in old_layout.cut_old_of_new.iter().enumerate() {
        new_of_old[old_layout.heavy[nb + tmp as usize] as usize] = roots + k;
    }
    let mut next_id = roots + nc;
    for slot in new_of_old.iter_mut().filter(|s| **s == usize::MAX) {
        *slot = next_id;
        next_id += 1;
    }
    assert_eq!(next_id, nb, "seed {seed}: a permutation");
    let mut block_cuts = vec![Vec::new(); nb];
    for (old, cuts) in old_block_cuts.into_iter().enumerate() {
        block_cuts[new_of_old[old]] = cuts;
    }
    (nb, nc, roots, block_cuts)
}

/// The layout of the renumbered forest keeps the chain numbering: the roots first, cut `k`'s heavy
/// child at block `roots + k`.
fn assert_chain_numbered(layout: &ChainLayout, nb: usize, roots: usize, seed: u64) {
    let root = layout.parent.len() - 1;
    for b in 0..nb {
        assert_eq!(
            layout.parent[b] as usize == root,
            b < roots,
            "seed {seed}: roots first"
        );
    }
    for (k, &tmp) in layout.cut_old_of_new.iter().enumerate() {
        assert_eq!(
            layout.heavy[nb + tmp as usize] as usize,
            roots + k,
            "seed {seed}: heavy child of cut {k}"
        );
    }
}

/// The navigation records of a layout, as the topology writes them, plus every cut's child blocks
/// (by provisional id) for the check.
struct NavRecords {
    children: Vec<Vec<usize>>,
    cuts: Vec<CutRec<u32>>,
    heads: Vec<HeadBlockRec>,
    extras: Vec<BlockIndex<u32>>,
    extra_dir: Vec<CutExtraDir>,
    block_words: Vec<u32>,
    no_blocks: Vec<RankBlock>,
    extra_blocks: Vec<RankBlock>,
}

impl NavRecords {
    fn of(layout: &ChainLayout, nb: usize, nc: usize, roots: usize) -> Self {
        // Child lists: heavy first; cuts with two or more children get an extra list.
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); nc];
        for b in 0..nb {
            let p = layout.parent[b] as usize;
            if p >= nb && p < nb + nc {
                children[p - nb].push(b);
            }
        }
        let mut extras: Vec<BlockIndex<u32>> = Vec::new();
        let mut extra_dir: Vec<CutExtraDir> = Vec::new();
        let cuts: Vec<CutRec<u32>> = (0..nc)
            .map(|new| {
                let old = layout.cut_old_of_new[new] as usize;
                let v = nb + old;
                let heavy = layout.heavy[v] as usize;
                let mut flags = layout.chain_of[v] << cut_flags::CHAIN_SHIFT;
                if new as u32 == layout.chains[layout.chain_of[v] as usize].cut_start {
                    flags |= cut_flags::FIRST_IN_CHAIN;
                }
                if children[old].len() >= 2 {
                    flags |= cut_flags::HAS_EXTRA;
                    extra_dir.push(CutExtraDir {
                        cut: new as u32,
                        start: extras.len() as u32,
                    });
                    extras.push(BlockIndex::from_usize(children[old].len()));
                    extras.push(BlockIndex::from_usize(heavy));
                    extras.extend(
                        children[old]
                            .iter()
                            .filter(|&&c| c != heavy)
                            .map(|&c| BlockIndex::from_usize(c)),
                    );
                }
                CutRec {
                    node: NodeIndex::from_usize(old),
                    p: GfaPathLength::from_usize(0),
                    len: GfaPathLength::from_usize(0),
                    f01: MinPlus::infinity(),
                    f10: MinPlus::infinity(),
                    flags: Padded::new(flags),
                }
            })
            .collect();
        let heads: Vec<HeadBlockRec> = (0..nb)
            .filter(|&b| b < roots || b >= roots + nc)
            .map(|b| HeadBlockRec {
                tpos: layout.tpos[b],
                chain: layout.chain_of[b],
                special: NONE,
            })
            .collect();
        let extra_blocks = rank_bits(nc, extra_dir.iter().map(|d| d.cut as usize), 0);
        Self {
            children,
            cuts,
            heads,
            extras,
            extra_dir,
            block_words: vec![0u32; nb],
            no_blocks: rank_bits(nc, [], 0),
            extra_blocks,
        }
    }

    fn nav<'a>(
        &'a self,
        layout: &'a ChainLayout,
        nb: usize,
        nc: usize,
        roots: usize,
    ) -> ChainNav<'a, u32> {
        ChainNav {
            nb,
            nc,
            roots,
            cuts: &self.cuts,
            block_words: &self.block_words,
            heads: &self.heads,
            heavy_special: RankBits {
                blocks: &self.no_blocks,
            },
            chains: &layout.chains,
            extras: &self.extras,
            extra_rank: RankBits {
                blocks: &self.extra_blocks,
            },
            extra_dir: &self.extra_dir,
            exc: &[],
        }
    }
}
