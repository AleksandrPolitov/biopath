//! The image record of a block's S-node oracles: the per-oracle records and the packed table.

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};

use super::{SNodeSegTree, SegTreeAccess};
use crate::{
    image::{Image, Padded, image_record, layout::Layout},
    min_plus::network::Mat4,
};

/// One cycle vertex of a stored tree: its graph node and cycle position. Sorted by node. `pos` is
/// as wide as the index word, so neither word size leaves padding.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SegVertexRec<IndexType: GraphIndexInteger> {
    pub node: NodeIndex<IndexType>,
    pub pos: Padded<u32, IndexType>,
}
image_record!(impl<I> SegVertexRec<I> = [NodeIndex<I>, Padded<u32, I>]);

/// Directory entry of one S-node oracle in a block's [`SegTable`]; sorted by `local`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SegRec {
    pub local: u32,
    /// Cycle vertices.
    pub m: u32,
    /// Start of this oracle's vertices in the table's vertex array.
    pub vertex_start: u32,
    /// Start and count of this oracle's matrices in the table's matrix array. Tree form: the tree
    /// nodes then the two closing networks. Ring form: the block tree, the two closing networks,
    /// then two "ring minus run" summaries per non-empty run.
    pub mat_start: u32,
    pub mats: u32,
    /// [`SEG_KIND_TREE`] or [`SEG_KIND_RING`].
    pub kind: u32,
    /// Ring form: start of the `m` [`PfxVertex`] entries in the table's prefix array.
    pub pfx_start: u32,
    /// Ring form: start and count of its [`SegRunRec`] entries in the table's run array.
    pub run_start: u32,
    pub runs: u32,
    /// Ring form: number of blocks under its block tree.
    pub blocks: u32,
}
crate::image::image_record!(SegRec = [u32, u32, u32, u32, u32, u32, u32, u32, u32, u32]);

/// [`SegRec::kind`]: the segment tree over the path leaves.
pub const SEG_KIND_TREE: u32 = 0;
/// [`SegRec::kind`]: the run/exception ring form (see [`SegRunRec`] and the module docs).
pub const SEG_KIND_RING: u32 = 1;

/// One run of a **ring-form** oracle: a maximal stretch of regular leaves, ended by an exception
/// leaf (the closing for the last run). Runs partition the cycle positions in order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SegRunRec {
    /// First cycle position of the run (its last is the next run's `start − 1`, or `m − 1`).
    pub start: u32,
    /// Block index of the exception leaf ending the run; `blocks` (one past the end) for the last
    /// run, whose exception is the closing network, kept outside the block sequence.
    pub x_block: u32,
    /// Slot of the run's two "ring minus this run" summaries (`[Inner, Lca]`) in the oracle's
    /// summary matrices, or `u32::MAX` for an empty run (a single position between two exceptions).
    pub summary: u32,
}
crate::image::image_record!(SegRunRec = [u32, u32, u32]);

/// One cycle position of a **ring-form** S-node oracle. Inside a run every leaf passes between its
/// two vertices in exactly one orientation pair — the cw lane — so a walk changes direction only at
/// a vertex flip (a rung), and a distance within the run is a prefix difference plus at most two
/// flips. Stored per position: ~28 B instead of the tree's two `4x4` matrices (128 B) per vertex.
///
/// No padding at either word size: `flags` is as wide as the index word, so `f` starts on a word
/// boundary (a plain `u32` at word size 32).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PfxVertex<IndexType: GraphIndexInteger> {
    /// Sum of the cw-lane pass costs from the start of the run to `v_e`.
    pub p: IndexType,
    /// Node length of `v_e`.
    pub len: IndexType,
    /// Bit 0: the cw orientation `c_e` of `v_e` (its state `(v_e, c_e)` moves cw); bits `1..`: the
    /// index of the run containing `v_e`.
    pub flags: Padded<u32, IndexType>,
    /// The cheapest flip at `v_e` per role and direction: `[inner cw->ccw, inner ccw->cw,
    /// lca cw->ccw, lca ccw->cw]` (`MAX` = none), global over the role's whole skeleton.
    pub f: [IndexType; 4],
}
image_record!(impl<I> PfxVertex<I> = [I, I, Padded<u32, I>, [I; 4]]);

/// Element totals of a [`SegTable`], stored in the owning record's header.
#[derive(Clone, Copy, Debug, Default)]
pub struct SegTableSizes {
    pub count: u32,
    pub vertices: u32,
    pub mats: u32,
    pub pfx: u32,
    pub runs: u32,
}

impl SegTableSizes {
    /// Totals over a block's trees.
    pub fn of<IndexType: GraphIndexInteger>(
        segtrees: &[Option<Box<SNodeSegTree<IndexType>>>],
    ) -> Self {
        let mut sizes = Self::default();
        for tree in segtrees.iter().flatten() {
            sizes.count += 1;
            sizes.vertices += tree.m as u32;
            sizes.mats += tree.mats.len() as u32;
            sizes.pfx += tree.pfx.len() as u32;
            sizes.runs += tree.runs.len() as u32;
        }
        sizes
    }
}

/// The relative offsets of a [`SegTable`]'s arrays inside a record (see [`Layout`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct SegTableLayout {
    pub recs: usize,
    pub vertices: usize,
    pub mats: usize,
    pub pfx: usize,
    pub runs: usize,
}

impl SegTableLayout {
    /// Lay the table out for the given totals.
    pub fn place<IndexType: GraphIndexInteger>(layout: &mut Layout, sizes: &SegTableSizes) -> Self {
        Self {
            recs: layout.field::<SegRec>(sizes.count as usize),
            vertices: layout.field::<SegVertexRec<IndexType>>(sizes.vertices as usize),
            mats: layout.field::<Mat4<IndexType>>(sizes.mats as usize),
            pfx: layout.field::<PfxVertex<IndexType>>(sizes.pfx as usize),
            runs: layout.field::<SegRunRec>(sizes.runs as usize),
        }
    }
}

/// A block's S-node oracles packed into concatenated arrays inside the image.
#[derive(Clone, Copy)]
pub struct SegTable<'a, IndexType: GraphIndexInteger> {
    recs: &'a [SegRec],
    vertices: &'a [SegVertexRec<IndexType>],
    mats: &'a [Mat4<IndexType>],
    pfx: &'a [PfxVertex<IndexType>],
    runs: &'a [SegRunRec],
}

impl<'a, IndexType: GraphIndexInteger> SegTable<'a, IndexType> {
    /// Write `segtrees` (indexed by local id) into a record at absolute byte offset `base`.
    ///
    /// # Safety
    /// `base + layout` must lie within a record this thread reserved from `image`.
    pub unsafe fn write(
        image: &Image,
        base: usize,
        layout: &SegTableLayout,
        sizes: &SegTableSizes,
        segtrees: Vec<Option<Box<SNodeSegTree<IndexType>>>>,
    ) {
        let recs = unsafe { image.get_at_mut::<SegRec>(base + layout.recs, sizes.count as usize) };
        let vertices = unsafe {
            image.get_at_mut::<SegVertexRec<IndexType>>(
                base + layout.vertices,
                sizes.vertices as usize,
            )
        };
        let mats =
            unsafe { image.get_at_mut::<Mat4<IndexType>>(base + layout.mats, sizes.mats as usize) };
        let pfx = unsafe {
            image.get_at_mut::<PfxVertex<IndexType>>(base + layout.pfx, sizes.pfx as usize)
        };
        let runs =
            unsafe { image.get_at_mut::<SegRunRec>(base + layout.runs, sizes.runs as usize) };
        let (mut k, mut v, mut t, mut x, mut u) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for (local, tree) in segtrees.into_iter().enumerate() {
            let Some(tree) = tree else { continue };
            recs[k] = SegRec {
                local: local as u32,
                m: tree.m as u32,
                vertex_start: v as u32,
                mat_start: t as u32,
                mats: tree.mats.len() as u32,
                kind: tree.kind(),
                pfx_start: x as u32,
                run_start: u as u32,
                runs: tree.runs.len() as u32,
                blocks: tree.blocks as u32,
            };
            vertices[v..v + tree.m].copy_from_slice(&tree.vertices);
            mats[t..t + tree.mats.len()].copy_from_slice(&tree.mats);
            pfx[x..x + tree.pfx.len()].copy_from_slice(&tree.pfx);
            runs[u..u + tree.runs.len()].copy_from_slice(&tree.runs);
            v += tree.m;
            t += tree.mats.len();
            x += tree.pfx.len();
            u += tree.runs.len();
            k += 1;
            drop(tree);
        }
        debug_assert_eq!(
            (k, v, t, x, u),
            (
                sizes.count as usize,
                sizes.vertices as usize,
                sizes.mats as usize,
                sizes.pfx as usize,
                sizes.runs as usize
            )
        );
    }

    /// Read the table at `base` with the given layout and totals.
    pub fn read(
        image: &'a Image,
        base: usize,
        layout: &SegTableLayout,
        sizes: &SegTableSizes,
    ) -> Self {
        Self {
            recs: image.get_at(base + layout.recs, sizes.count as usize),
            vertices: image.get_at(base + layout.vertices, sizes.vertices as usize),
            mats: image.get_at(base + layout.mats, sizes.mats as usize),
            pfx: image.get_at(base + layout.pfx, sizes.pfx as usize),
            runs: image.get_at(base + layout.runs, sizes.runs as usize),
        }
    }

    /// The tree of block-local SPQR node `local`, if it has one.
    #[inline]
    pub fn get(&self, local: usize) -> Option<SegTreeView<'a, IndexType>> {
        if self.recs.is_empty() {
            return None;
        }
        let i = self
            .recs
            .binary_search_by_key(&(local as u32), |r| r.local)
            .ok()?;
        Some(self.view(&self.recs[i]))
    }

    fn view(&self, rec: &SegRec) -> SegTreeView<'a, IndexType> {
        let m = rec.m as usize;
        let pfx_len = if rec.kind == SEG_KIND_TREE { 0 } else { m };
        SegTreeView {
            m,
            vertices: &self.vertices[rec.vertex_start as usize..rec.vertex_start as usize + m],
            mats: &self.mats[rec.mat_start as usize..rec.mat_start as usize + rec.mats as usize],
            pfx: &self.pfx[rec.pfx_start as usize..rec.pfx_start as usize + pfx_len],
            runs: &self.runs[rec.run_start as usize..rec.run_start as usize + rec.runs as usize],
            blocks: rec.blocks as usize,
        }
    }
}

/// An image-backed S-node oracle (see [`SegTreeAccess`]).
#[derive(Clone, Copy)]
pub struct SegTreeView<'a, IndexType: GraphIndexInteger> {
    m: usize,
    vertices: &'a [SegVertexRec<IndexType>],
    mats: &'a [Mat4<IndexType>],
    pfx: &'a [PfxVertex<IndexType>],
    runs: &'a [SegRunRec],
    blocks: usize,
}

impl<IndexType: GraphIndexInteger> SegTreeAccess<IndexType> for SegTreeView<'_, IndexType> {
    #[inline]
    fn m(&self) -> usize {
        self.m
    }
    #[inline]
    fn vertex_recs(&self) -> &[SegVertexRec<IndexType>] {
        self.vertices
    }
    #[inline]
    fn mats(&self) -> &[Mat4<IndexType>] {
        self.mats
    }
    #[inline]
    fn pfx(&self) -> &[PfxVertex<IndexType>] {
        self.pfx
    }
    #[inline]
    fn runs(&self) -> &[SegRunRec] {
        self.runs
    }
    #[inline]
    fn blocks(&self) -> usize {
        self.blocks
    }
}
