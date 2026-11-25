//! One block's record in the image: a fixed header, then the rooted SPQR tree (depths and lifting
//! table), every node's reference poles, the pole-lift steps and the `D_out` overlays (sparse
//! `4x4`s), the pole-projection tables, the inner skeletons (one block-wide CSR), the S-node
//! segment trees and the hub labels. Everything a query touches for an endpoint block sits in one
//! contiguous range (a cold query faults in a couple of pages per block, not one per structure), and
//! sub-array offsets are derived from the counts in the header rather than stored (see
//! [`crate::image::layout::Layout`]). See [`crate::block_engine`] for how a query reads it.

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex, OptionalNodeIndex};
use spqr_tree::decomposition::indices::BlockIndex;

use crate::{
    block_spqr_tree::BlockSpqrTree,
    image::{Image, ImageRecord, layout::Layout},
    min_plus::{
        MinPlus, MinPlusMatrix,
        sparse::{self, DIAG_MASK, SparseMat4},
    },
    pole_lift::{PoleLift, PoleStepsView, pole_states},
    rooted_tree::RootedTreeView,
    s_node_segtree::{SNodeSegTree, SegTable, SegTableLayout, SegTableSizes, SegTreeView},
    skeleton::{LcaView, SkelTable, SkelTableLayout, SkelTableSizes, Skeleton, SkeletonView},
    skeleton_index::{LabelRec, LabelTable, LabelsView, SkeletonLabels},
};

/// Directory entry of a block that has a record: the record's absolute byte offset. The directory
/// (`BLOCK_DIR_SPARSE`) is sorted by block and lists only blocks with a record — under the
/// small-block fast path that is a few thousand of millions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockDirRec {
    pub block: u32,
    pub _pad: u32,
    pub off: u64,
}
crate::image::image_record!(BlockDirRec = [u32, u32, u64]);

/// The sparse directory of a dense per-block offset table (`u64::MAX` = no record).
pub fn sparse_block_dir(dense: &[u64]) -> Vec<BlockDirRec> {
    dense
        .iter()
        .enumerate()
        .filter(|(_, off)| **off != u64::MAX)
        .map(|(block, &off)| BlockDirRec {
            block: block as u32,
            _pad: 0,
            off,
        })
        .collect()
}

/// The record offset of `block` in a sorted sparse directory, or `None` (small-block fast path).
#[inline]
pub fn block_dir_offset(dir: &[BlockDirRec], block: usize) -> Option<usize> {
    dir.binary_search_by_key(&(block as u32), |r| r.block)
        .ok()
        .map(|i| dir[i].off as usize)
}

/// The fixed header of a block record (72 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockHeader {
    pub node_count: u32,
    pub tree_root: u32,
    pub tree_log: u32,
    pub tree_max_depth: u32,
    pub inner: SkelTableSizes,
    pub label_count: u32,
    /// S-node segment trees: count, total cycle vertices, total `4x4` matrices.
    pub seg_count: u32,
    pub seg_vertices: u32,
    pub seg_mats: u32,
    /// S-node ring-form position entries (see `s_node_segtree::PfxVertex`).
    pub seg_pfx: u32,
    /// S-node ring-form runs (see `s_node_segtree::SegRunRec`).
    pub seg_runs: u32,
    /// Finite pole-lift step entries stored (see [`SparseMat4`]).
    pub step_values: u32,
    /// Finite `D_out` entries stored (the LCA-role overlay, same encoding).
    pub dout_values: u32,
    /// Skeletons with a pole-projection table, and the total entries of those tables
    /// (`4 x states` each; see [`BlockView::projection`]).
    pub proj_count: u32,
    pub proj_values: u32,
}
crate::image::image_record!(
    BlockHeader = [
        u32,
        u32,
        u32,
        u32,
        SkelTableSizes,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32
    ]
);

/// Directory entry of one pole-projection table: `proj_data[start..next.start]` belongs to `local`
/// (sorted by `local`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ProjRec {
    pub local: u32,
    pub start: u32,
}
crate::image::image_record!(ProjRec = [u32, u32]);

impl BlockHeader {
    fn seg_sizes(&self) -> SegTableSizes {
        SegTableSizes {
            count: self.seg_count,
            vertices: self.seg_vertices,
            mats: self.seg_mats,
            pfx: self.seg_pfx,
            runs: self.seg_runs,
        }
    }
}

/// Per-SPQR-node record: only what a query reads. (The global SPQR node id and the node type are
/// build-time facts; no query path needs them.)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NodeRec<IndexType: GraphIndexInteger> {
    /// Both sentinel-none at the root; otherwise the two reference poles.
    pub reference_poles: [OptionalNodeIndex<IndexType>; 2],
}
crate::image::image_record!(impl<I> NodeRec<I> = [[OptionalNodeIndex<I>; 2]]);

/// The derived sub-array offsets of a record body (relative to the record start, after the header).
#[derive(Clone, Copy, Debug, Default)]
struct BodyLayout {
    depth: usize,
    up_flat: usize,
    nodes: usize,
    step_masks: usize,
    step_offsets: usize,
    step_values: usize,
    dout_masks: usize,
    dout_offsets: usize,
    dout_values: usize,
    proj_recs: usize,
    proj_data: usize,
    inner: SkelTableLayout,
    seg: SegTableLayout,
    label_recs: usize,
    label_blob: usize,
    total: usize,
}

impl BodyLayout {
    fn of<IndexType: GraphIndexInteger>(hdr: &BlockHeader, label_blob_bytes: usize) -> Self {
        let n = hdr.node_count as usize;
        let mut layout = Layout::new();
        layout.field::<BlockHeader>(1);
        let depth = layout.field::<u32>(n);
        let up_flat = layout.field::<u32>((hdr.tree_log as usize + 1) * n);
        let nodes = layout.field::<NodeRec<IndexType>>(n);
        let step_masks = layout.field::<u16>(n);
        let step_offsets = layout.field::<u32>(n);
        let step_values = layout.field::<MinPlus<IndexType>>(hdr.step_values as usize);
        let dout_masks = layout.field::<u16>(n);
        let dout_offsets = layout.field::<u32>(n);
        let dout_values = layout.field::<MinPlus<IndexType>>(hdr.dout_values as usize);
        let proj_recs = layout.field::<ProjRec>(hdr.proj_count as usize);
        let proj_data = layout.field::<MinPlus<IndexType>>(hdr.proj_values as usize);
        let inner = SkelTableLayout::place::<IndexType>(&mut layout, n, &hdr.inner);
        let seg = SegTableLayout::place::<IndexType>(&mut layout, &hdr.seg_sizes());
        let label_recs = layout.field::<LabelRec>(hdr.label_count as usize);
        let label_blob = layout.blob(label_blob_bytes);
        Self {
            depth,
            up_flat,
            nodes,
            step_masks,
            step_offsets,
            step_values,
            dout_masks,
            dout_offsets,
            dout_values,
            proj_recs,
            proj_data,
            inner,
            seg,
            label_recs,
            label_blob,
            total: layout.total(),
        }
    }
}

/// Write one block's record into `image`; returns its absolute byte offset.
///
/// `first_label_ordinal` numbers this block's label oracles image-wide (the caller advances it by
/// the number of `Some` entries in `labels`). `lca_outside[local]` is the node's `D_out` where its
/// LCA search runs over the stored skeleton (see [`lca_overlays`]); the LCA-role skeleton itself is not
/// stored, only `inner` plus this overlay (see [`LcaView`]). `projections[local]` is the node's
/// pole-projection table (`4 x states` entries), if it has one.
#[allow(clippy::too_many_arguments)]
pub fn write_block<IndexType: GraphIndexInteger>(
    image: &Image,
    tree: &BlockSpqrTree<IndexType>,
    pole_lift: PoleLift<IndexType>,
    inner: Vec<Skeleton<IndexType>>,
    lca_outside: Vec<Option<MinPlusMatrix<IndexType, 4>>>,
    labels: Vec<Option<Box<SkeletonLabels<IndexType>>>>,
    segtrees: Vec<Option<Box<SNodeSegTree<IndexType>>>>,
    projections: Vec<Option<Vec<MinPlus<IndexType>>>>,
    first_label_ordinal: u32,
) -> u64 {
    let n = tree.node_count();
    debug_assert_eq!(inner.len(), n);
    debug_assert_eq!(lca_outside.len(), n);
    debug_assert_eq!(segtrees.len(), n);
    debug_assert_eq!(projections.len(), n);
    let (proj_recs, proj_total) = projection_directory(&projections);
    let rooted = tree.tree().as_view();
    let (label_recs, label_blob_bytes) = LabelTable::plan(&labels, first_label_ordinal);
    let seg_sizes = SegTableSizes::of(&segtrees);
    let steps = pole_lift.sparse_steps();
    drop(pole_lift);
    // The search never relaxes a `D_out` diagonal entry (a self-loop), so only its mask bit (which
    // decides the overlay's node set) is kept.
    let dout = SparseMat4::encode_values(lca_outside.iter().map(Option::as_ref), !DIAG_MASK);
    drop(lca_outside);
    let hdr = BlockHeader {
        node_count: n as u32,
        tree_root: rooted.root,
        tree_log: rooted.log,
        tree_max_depth: rooted.max_depth,
        inner: SkelTableSizes::of(&inner),
        label_count: label_recs.len() as u32,
        seg_count: seg_sizes.count,
        seg_vertices: seg_sizes.vertices,
        seg_mats: seg_sizes.mats,
        seg_pfx: seg_sizes.pfx,
        seg_runs: seg_sizes.runs,
        step_values: steps.values.len() as u32,
        dout_values: dout.values.len() as u32,
        proj_count: proj_recs.len() as u32,
        proj_values: u32::try_from(proj_total).expect("projection tables exceed u32"),
    };
    let layout = BodyLayout::of::<IndexType>(&hdr, label_blob_bytes);
    // A block whose record is tens of MB (the giant block of a tangled pangenome) frees that much
    // owned heap while copying; return it to the OS between the pieces so the peak stays one piece
    // over the owned total (see `trim_heap`).
    let trim = layout.total >= TRIM_RECORD_BYTES;
    let (sec, _) = image.alloc::<u8>(layout.total);
    let record = Record {
        image,
        base: sec.off as usize,
    };
    // SAFETY: every range below lies inside the record just reserved by this thread.
    unsafe {
        record.put(0, &[hdr]);
        record.put(layout.depth, rooted.depth);
        record.put(layout.up_flat, rooted.up_flat);
        record.put(layout.nodes, &node_recs(tree));
        record.put_sparse(
            layout.step_masks,
            layout.step_offsets,
            layout.step_values,
            steps,
        );
        record.put_sparse(
            layout.dout_masks,
            layout.dout_offsets,
            layout.dout_values,
            dout,
        );
        record.put(layout.proj_recs, &proj_recs);
        record.put_projections(layout.proj_data, proj_total, projections);
        // Each table consumes its owned data as it copies, so the build's transient over the owned
        // block is one skeleton / one label oracle, not the whole block twice.
        SkelTable::write(image, record.base, &layout.inner, &hdr.inner, inner);
        if trim {
            crate::image::trim_heap();
        }
        SegTable::write(image, record.base, &layout.seg, &seg_sizes, segtrees);
        LabelTable::write(
            image,
            record.base + layout.label_recs,
            record.base + layout.label_blob,
            &label_recs,
            labels,
        );
        if trim {
            crate::image::trim_heap();
        }
    }
    record.base as u64
}

/// The directory of a block's pole-projection tables and their total entries.
fn projection_directory<IndexType: GraphIndexInteger>(
    projections: &[Option<Vec<MinPlus<IndexType>>>],
) -> (Vec<ProjRec>, usize) {
    let mut proj_recs: Vec<ProjRec> = Vec::new();
    let mut proj_total = 0usize;
    for (local, table) in projections.iter().enumerate() {
        if let Some(table) = table {
            proj_recs.push(ProjRec {
                local: local as u32,
                start: u32::try_from(proj_total).expect("projection tables exceed u32"),
            });
            proj_total += table.len();
        }
    }
    (proj_recs, proj_total)
}

/// The per-node records: each node's reference poles (both none at the root).
fn node_recs<IndexType: GraphIndexInteger>(
    tree: &BlockSpqrTree<IndexType>,
) -> Vec<NodeRec<IndexType>> {
    (0..tree.node_count())
        .map(|local| NodeRec {
            reference_poles: match tree.node(local).reference_poles() {
                Some((s, t)) => [
                    OptionalNodeIndex::from_option(Some(s)),
                    OptionalNodeIndex::from_option(Some(t)),
                ],
                None => [OptionalNodeIndex::new_none(); 2],
            },
        })
        .collect()
}

/// A record being written: its image and absolute base offset.
struct Record<'i> {
    image: &'i Image,
    base: usize,
}

impl Record<'_> {
    /// Copy `data` to the record-relative byte offset `off`.
    ///
    /// # Safety
    /// The range must lie inside the record this thread reserved.
    unsafe fn put<T: ImageRecord>(&self, off: usize, data: &[T]) {
        unsafe { self.image.get_at_mut::<T>(self.base + off, data.len()) }.copy_from_slice(data);
    }

    /// Copy a [`SparseMat4`] table to its three offsets, consuming it.
    ///
    /// # Safety
    /// As [`Self::put`].
    unsafe fn put_sparse<IndexType: GraphIndexInteger>(
        &self,
        masks: usize,
        offsets: usize,
        values: usize,
        table: SparseMat4<IndexType>,
    ) {
        unsafe {
            self.put(masks, &table.masks);
            self.put(offsets, &table.offsets);
            self.put(values, &table.values);
        }
    }

    /// Copy the pole-projection tables, concatenated in local order, to `off` (`total` entries).
    ///
    /// # Safety
    /// As [`Self::put`].
    unsafe fn put_projections<IndexType: GraphIndexInteger>(
        &self,
        off: usize,
        total: usize,
        projections: Vec<Option<Vec<MinPlus<IndexType>>>>,
    ) {
        let out = unsafe {
            self.image
                .get_at_mut::<MinPlus<IndexType>>(self.base + off, total)
        };
        let mut cursor = 0usize;
        for table in projections.into_iter().flatten() {
            out[cursor..cursor + table.len()].copy_from_slice(&table);
            cursor += table.len();
        }
        debug_assert_eq!(cursor, total);
    }
}

/// Records at least this large trim the heap between their pieces (see [`write_block`]).
const TRIM_RECORD_BYTES: usize = 64 << 20;

/// A block's record, resolved into slices.
#[derive(Clone, Copy)]
pub struct BlockView<'a, IndexType: GraphIndexInteger> {
    /// [`Image::instance_id`] of the image holding the record (the query memos key on it).
    image_id: u64,
    block: BlockIndex<IndexType>,
    tree: RootedTreeView<'a>,
    nodes: &'a [NodeRec<IndexType>],
    steps: PoleStepsView<'a, IndexType>,
    inner: SkelTable<'a, IndexType>,
    dout_masks: &'a [u16],
    dout_offsets: &'a [u32],
    dout_values: &'a [MinPlus<IndexType>],
    proj_recs: &'a [ProjRec],
    proj_data: &'a [MinPlus<IndexType>],
    seg: SegTable<'a, IndexType>,
    labels: LabelTable<'a, IndexType>,
}

impl<'a, IndexType: GraphIndexInteger> BlockView<'a, IndexType> {
    /// Resolve the record at absolute byte offset `base`.
    pub fn read(image: &'a Image, block: BlockIndex<IndexType>, base: usize) -> Self {
        let hdr: BlockHeader = *image.get_one::<BlockHeader>(base);
        let n = hdr.node_count as usize;
        // The label blob length is only needed for the layout total, which nothing reads here.
        let layout = BodyLayout::of::<IndexType>(&hdr, 0);
        Self {
            image_id: image.instance_id(),
            block,
            tree: RootedTreeView {
                depth: image.get_at(base + layout.depth, n),
                up_flat: image.get_at(base + layout.up_flat, (hdr.tree_log as usize + 1) * n),
                root: hdr.tree_root,
                log: hdr.tree_log,
                max_depth: hdr.tree_max_depth,
            },
            nodes: image.get_at(base + layout.nodes, n),
            steps: PoleStepsView {
                masks: image.get_at(base + layout.step_masks, n),
                offsets: image.get_at(base + layout.step_offsets, n),
                values: image.get_at(base + layout.step_values, hdr.step_values as usize),
            },
            inner: SkelTable::read(image, base, &layout.inner, &hdr.inner, n),
            dout_masks: image.get_at(base + layout.dout_masks, n),
            dout_offsets: image.get_at(base + layout.dout_offsets, n),
            dout_values: image.get_at(base + layout.dout_values, hdr.dout_values as usize),
            proj_recs: image.get_at(base + layout.proj_recs, hdr.proj_count as usize),
            proj_data: image.get_at(base + layout.proj_data, hdr.proj_values as usize),
            seg: SegTable::read(image, base, &layout.seg, &hdr.seg_sizes()),
            labels: LabelTable::read(
                image,
                base + layout.label_recs,
                base + layout.label_blob,
                hdr.label_count as usize,
            ),
        }
    }

    /// The block this record belongs to.
    pub fn block(&self) -> BlockIndex<IndexType> {
        self.block
    }

    /// The process-unique id of the image holding the record ([`Image::instance_id`]).
    pub fn image_id(&self) -> u64 {
        self.image_id
    }

    /// The number of SPQR nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The rooted SPQR tree (parent/LCA queries).
    pub fn tree(&self) -> &RootedTreeView<'a> {
        &self.tree
    }

    /// The poles `(s, t)` of the virtual edge to the parent, or `None` for the root.
    #[inline]
    pub fn reference_poles(
        &self,
        local: usize,
    ) -> Option<(NodeIndex<IndexType>, NodeIndex<IndexType>)> {
        let poles = self.nodes[local].reference_poles;
        let x = poles[0].into_option()?;
        Some((x, poles[1].expect("both reference poles are present")))
    }

    /// The pole-lift single-step transitions.
    pub fn steps(&self) -> &PoleStepsView<'a, IndexType> {
        &self.steps
    }

    /// The inner skeleton (no `D_out`) of `local`.
    #[inline]
    pub fn inner(&self, local: usize) -> SkeletonView<'a, IndexType> {
        self.inner.get(local)
    }

    /// The LCA-role skeleton (with `D_out`) of `local`: its inner skeleton plus the stored `D_out`
    /// overlay (none at the root or where no outside matrix was stored).
    #[inline]
    pub fn lca(&self, local: usize) -> LcaView<'a, IndexType> {
        let outside = match (self.dout_masks[local], self.reference_poles(local)) {
            (0, _) | (_, None) => None,
            (mask, Some((s, t))) => Some((
                pole_states(s, t),
                sparse::decode_values(
                    self.dout_masks,
                    self.dout_offsets,
                    self.dout_values,
                    local,
                    !DIAG_MASK,
                ),
                mask,
            )),
        };
        LcaView::new(self.inner.get(local), outside)
    }

    /// The pole-projection table of `local`, if any: `4 x states` entries, `[4 * id + slot]` =
    /// `D_inner(state id -> pole slot)` in slot order `[s^+, s^-, t^+, t^-]`.
    #[inline]
    pub fn projection(&self, local: usize) -> Option<&'a [MinPlus<IndexType>]> {
        let i = self
            .proj_recs
            .binary_search_by_key(&(local as u32), |r| r.local)
            .ok()?;
        let start = self.proj_recs[i].start as usize;
        let end = self
            .proj_recs
            .get(i + 1)
            .map_or(self.proj_data.len(), |r| r.start as usize);
        Some(&self.proj_data[start..end])
    }

    /// The label oracle of `local`, if any.
    #[inline]
    pub fn labels(&self, local: usize) -> Option<LabelsView<'a, IndexType>> {
        self.labels.get(local)
    }

    /// The S-node segment-tree oracle of `local`, if any (it answers both roles of the node).
    #[inline]
    pub fn segtree(&self, local: usize) -> Option<SegTreeView<'a, IndexType>> {
        self.seg.get(local)
    }
}

/// The `D_out` overlays of a block's record: node `local` keeps `outside_matrix(local)` iff it has
/// reference poles and a query searches its LCA-role skeleton (`lca_kept`; a labeled or
/// segment-tree node answers its LCA search without it). [`BlockView::lca`] assembles that skeleton
/// from the inner one and this overlay.
pub fn lca_overlays<IndexType: GraphIndexInteger>(
    tree: &BlockSpqrTree<IndexType>,
    outside_matrix: impl Fn(usize) -> MinPlusMatrix<IndexType, 4>,
    lca_kept: impl Fn(usize) -> bool,
) -> Vec<Option<MinPlusMatrix<IndexType, 4>>> {
    (0..tree.node_count())
        .map(|local| {
            (tree.node(local).reference_poles().is_some() && lca_kept(local))
                .then(|| outside_matrix(local))
        })
        .collect()
}
