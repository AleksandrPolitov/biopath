//! The build-time chain layout of a block-cut tree.

use super::{ChainRec, NONE};
use crate::rooted_tree::{ChainTreeArrays, TreeAccess};

/// A parent-pointer tree for [`ChainTreeArrays::build`] (`up_flat` is just the parent row).
pub(super) struct ParentTree<'a> {
    pub(super) parent: &'a [u32],
    pub(super) depth: &'a [u32],
    pub(super) root: usize,
    pub(super) max_depth: u32,
}

impl TreeAccess for ParentTree<'_> {
    fn depth_slice(&self) -> &[u32] {
        self.depth
    }
    fn up_flat(&self) -> &[u32] {
        self.parent
    }
    fn root(&self) -> usize {
        self.root
    }
    fn log(&self) -> u32 {
        0
    }
    fn max_depth(&self) -> u32 {
        self.max_depth
    }
}

/// The build-time chain layout of a block-cut tree over tree ids `0..nb` (blocks), `nb..nb+nc`
/// (cut nodes, in a provisional order) and `nb+nc` (the super-root).
pub struct ChainLayout {
    /// Tree position and chain of every tree id.
    pub tpos: Vec<u32>,
    pub chain_of: Vec<u32>,
    /// The chains (structural fields; the exception fields are the lift's).
    pub chains: Vec<ChainRec>,
    /// Provisional cut id -> dense cut id (chain order), and back.
    pub cut_new_id: Vec<u32>,
    pub cut_old_of_new: Vec<u32>,
    /// Parent / depth / heavy child per PROVISIONAL tree id (the build's own numbering).
    pub parent: Vec<u32>,
    pub depth: Vec<u32>,
    pub heavy: Vec<u32>,
}

impl ChainLayout {
    /// Lay the tree out. `parent[root] == root`; `block_component[b]` is each block's component and
    /// `component_sentinel` what a block-less chain reports.
    pub fn build(
        nb: usize,
        nc: usize,
        parent: Vec<u32>,
        depth: Vec<u32>,
        block_component: &[u32],
        component_sentinel: u32,
    ) -> Self {
        let total = nb + nc + 1;
        debug_assert_eq!(parent.len(), total);
        let max_depth = depth.iter().copied().max().unwrap_or(0);
        let ChainTreeArrays { head, heavy, .. } = ChainTreeArrays::build(&ParentTree {
            parent: &parent,
            depth: &depth,
            root: nb + nc,
            max_depth,
        });
        let heads = chain_heads(&head, &depth);
        let mut layout = Self {
            tpos: vec![NONE; total],
            chain_of: vec![NONE; total],
            chains: Vec::with_capacity(heads.len()),
            cut_new_id: vec![NONE; nc],
            cut_old_of_new: Vec::with_capacity(nc),
            parent,
            depth,
            heavy,
        };
        let mut next = 0u32;
        for (k, &head) in heads.iter().enumerate() {
            let walk = layout.walk_chain(k, head as usize, next);
            let component = match walk.first_block_id {
                NONE => component_sentinel,
                b => block_component[b as usize],
            };
            let rec = layout.chain_record(head as usize, next, &walk, component);
            layout.chains.push(rec);
            next += walk.len;
        }
        debug_assert_eq!(next as usize, total);
        debug_assert_eq!(layout.cut_old_of_new.len(), nc);
        layout
    }

    /// The number of blocks of the tree.
    pub fn block_count(&self) -> usize {
        self.tpos.len() - self.cut_count() - 1
    }

    /// The number of cut nodes of the tree.
    pub fn cut_count(&self) -> usize {
        self.cut_new_id.len()
    }

    fn is_cut(&self, v: usize) -> bool {
        v >= self.block_count() && v < self.block_count() + self.cut_count()
    }

    /// Lay chain `k` out down the heavy path from `head`, starting at tree position `start`: the
    /// positions, the chain ids and the dense ids of its cut nodes.
    fn walk_chain(&mut self, k: usize, head: usize, start: u32) -> ChainWalk {
        let nb = self.block_count();
        let mut walk = ChainWalk {
            len: 0,
            cut_off: NONE,
            first_block_id: NONE,
        };
        let mut v = head;
        loop {
            self.tpos[v] = start + walk.len;
            self.chain_of[v] = k as u32;
            if self.is_cut(v) {
                if walk.cut_off == NONE {
                    walk.cut_off = walk.len;
                }
                self.cut_new_id[v - nb] = self.cut_old_of_new.len() as u32;
                self.cut_old_of_new.push((v - nb) as u32);
            } else if v < nb && walk.first_block_id == NONE {
                walk.first_block_id = v as u32;
            }
            walk.len += 1;
            match self.heavy[v] {
                u32::MAX => return walk,
                child => v = child as usize,
            }
        }
    }

    /// The record of the chain just walked from `head` at `start`. Ids in the record are the new
    /// tree ids (cuts in chain order); the attach node lies on an earlier chain, so its new id is
    /// already known.
    fn chain_record(&self, head: usize, start: u32, walk: &ChainWalk, component: u32) -> ChainRec {
        let cuts_so_far = self.cut_old_of_new.len() as u32;
        let (cut_start, cut_off) = match walk.cut_off {
            NONE => (cuts_so_far, walk.len),
            off => (cuts_so_far - (walk.len - off).div_ceil(2), off),
        };
        let p = self.parent[head] as usize;
        ChainRec {
            tpos_start: start,
            len: walk.len,
            cut_start,
            cut_off,
            attach_tpos: self.tpos[p],
            attach_id: self.new_id(p),
            attach_chain: self.chain_of[p],
            head_id: self.new_id(head),
            head_depth: self.depth[head],
            first_block_id: walk.first_block_id,
            component,
            exc_start: 0,
            exc_count: 0,
            exc_inherit: NONE,
        }
    }

    /// Tree id `v` with a cut node under its dense id.
    fn new_id(&self, v: usize) -> u32 {
        if self.is_cut(v) {
            (self.block_count() as u32) + self.cut_new_id[v - self.block_count()]
        } else {
            v as u32
        }
    }
}

/// One chain's walk down its heavy path: its length, the offset of its first cut node and its
/// first block (`NONE` if none).
struct ChainWalk {
    len: u32,
    cut_off: u32,
    first_block_id: u32,
}

/// The chain heads (`head[v] == v`) by `(depth, id)`: a chain's attach node lies on a chain whose
/// head is shallower, so this order is parent-before-child.
fn chain_heads(head: &[u32], depth: &[u32]) -> Vec<u32> {
    let mut heads: Vec<u32> = (0..head.len() as u32)
        .filter(|&v| head[v as usize] == v)
        .collect();
    heads.sort_unstable_by_key(|&v| (depth[v as usize], v));
    heads
}
