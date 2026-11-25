//! A block's label oracles in the image: the per-oracle directory, the packed blob and the
//! image-backed [`LabelsView`].

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use super::{LabelEntry, LabelsAccess, NarrowRow, SkeletonLabels, mirror_distance};
use crate::{
    image::{
        Image,
        hash::{self, NodeSlot},
        layout::Layout,
    },
    path::GfaPathLength,
    skeleton::node_key,
};

/// The stored entries of an image-backed oracle: wide (`u32` hubs) or narrow (`u16` hubs in one
/// array, values in another).
#[derive(Clone, Copy)]
enum LabelFlat<'a, IndexType: GraphIndexInteger> {
    Wide(&'a [LabelEntry<IndexType>]),
    Narrow {
        hubs: &'a [u16],
        values: &'a [GfaPathLength<IndexType>],
    },
}

/// An image-backed label oracle (mirror form; see [`LabelsAccess`]).
#[derive(Clone, Copy)]
pub struct LabelsView<'a, IndexType: GraphIndexInteger> {
    id: u64,
    node_of_id: &'a [DirectedNodeIndex<IndexType>],
    hash: &'a [NodeSlot<IndexType>],
    len_of_id: &'a [GfaPathLength<IndexType>],
    offsets: &'a [u32],
    flat: LabelFlat<'a, IndexType>,
}

impl<IndexType: GraphIndexInteger> LabelsAccess<IndexType> for LabelsView<'_, IndexType> {
    #[inline]
    fn id(&self) -> u64 {
        self.id
    }
    #[inline]
    fn node_count(&self) -> usize {
        self.node_of_id.len()
    }
    #[inline]
    fn node_id(&self, node: DirectedNodeIndex<IndexType>) -> Option<u32> {
        hash::lookup_node(self.hash, node_key(node))
    }
    #[inline]
    fn distance_id(&self, from: u32, to: u32) -> Option<GfaPathLength<IndexType>> {
        let n_minus_1 = self.node_of_id.len() as u32 - 1;
        let tbar = n_minus_1 - to;
        let (fs, fe) = (
            self.offsets[from as usize] as usize,
            self.offsets[from as usize + 1] as usize,
        );
        let (ts, te) = (
            self.offsets[tbar as usize] as usize,
            self.offsets[tbar as usize + 1] as usize,
        );
        let len_t = self.len_of_id[to as usize];
        match self.flat {
            LabelFlat::Wide(f) => mirror_distance(&f[fs..fe], &f[ts..te], n_minus_1, len_t),
            LabelFlat::Narrow { hubs, values } => mirror_distance(
                NarrowRow {
                    hubs: &hubs[fs..fe],
                    values: &values[fs..fe],
                },
                NarrowRow {
                    hubs: &hubs[ts..te],
                    values: &values[ts..te],
                },
                n_minus_1,
                len_t,
            ),
        }
    }
}

/// Element totals of one persisted oracle (mirror form: out-labels only).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LabelSizes {
    pub nodes: u32,
    pub hash_slots: u32,
    pub entries: u32,
    /// `1` when the entries are narrow (`u16` hubs; at most [`super::NARROW_LABEL_MAX_STATES`] states).
    pub narrow: u32,
}
crate::image::image_record!(LabelSizes = [u32, u32, u32, u32]);

/// Relative offsets of one persisted oracle's arrays inside its blob.
#[derive(Clone, Copy, Debug, Default)]
struct LabelLayout {
    node_of_id: usize,
    hash: usize,
    len_of_id: usize,
    offsets: usize,
    /// Wide entries, or the narrow hubs.
    flat: usize,
    /// Narrow values (unused when wide).
    values: usize,
    total: usize,
}

impl LabelLayout {
    fn of<IndexType: GraphIndexInteger>(sizes: &LabelSizes) -> Self {
        let mut layout = Layout::new();
        let n = sizes.nodes as usize;
        let this = Self {
            node_of_id: layout.field::<DirectedNodeIndex<IndexType>>(n),
            hash: layout.field::<NodeSlot<IndexType>>(sizes.hash_slots as usize),
            len_of_id: layout.field::<GfaPathLength<IndexType>>(n),
            offsets: layout.field::<u32>(n + 1),
            flat: if sizes.narrow != 0 {
                layout.field::<u16>(sizes.entries as usize)
            } else {
                layout.field::<LabelEntry<IndexType>>(sizes.entries as usize)
            },
            values: if sizes.narrow != 0 {
                layout.field::<GfaPathLength<IndexType>>(sizes.entries as usize)
            } else {
                0
            },
            total: 0,
        };
        Self {
            total: layout.total(),
            ..this
        }
    }
}

/// Directory entry of one labeled SPQR node in a block's [`LabelTable`]; sorted by `local`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LabelRec {
    pub local: u32,
    /// Image-wide ordinal of this oracle (its `id` is derived from it).
    pub ordinal: u32,
    pub sizes: LabelSizes,
    /// Byte offset of the oracle's blob, relative to the table's blob start.
    pub blob_off: u64,
}
crate::image::image_record!(LabelRec = [u32, u32, LabelSizes, u64]);

/// A block's (usually zero or few) label oracles, as a sorted directory plus one packed blob.
#[derive(Clone, Copy)]
pub struct LabelTable<'a, IndexType: GraphIndexInteger> {
    image: &'a Image,
    recs: &'a [LabelRec],
    blob_base: usize,
    _t: std::marker::PhantomData<IndexType>,
}

impl<'a, IndexType: GraphIndexInteger> LabelTable<'a, IndexType> {
    /// The directory entries and total blob bytes for `labels` (indexed by local id, `None` = no
    /// oracle). `first_ordinal` numbers the oracles image-wide.
    pub fn plan(
        labels: &[Option<Box<SkeletonLabels<IndexType>>>],
        first_ordinal: u32,
    ) -> (Vec<LabelRec>, usize) {
        let mut recs = Vec::new();
        let mut blob = 0usize;
        for (local, label) in labels.iter().enumerate() {
            let Some(label) = label else { continue };
            let sizes = label.sizes();
            recs.push(LabelRec {
                local: local as u32,
                ordinal: first_ordinal + recs.len() as u32,
                sizes,
                blob_off: blob as u64,
            });
            blob += LabelLayout::of::<IndexType>(&sizes).total;
        }
        (recs, blob)
    }

    /// Write the directory (at absolute `recs_base`) and the blob (at absolute `blob_base`).
    ///
    /// # Safety
    /// Both ranges must lie within a record this thread reserved from `image`.
    pub unsafe fn write(
        image: &Image,
        recs_base: usize,
        blob_base: usize,
        recs: &[LabelRec],
        mut labels: Vec<Option<Box<SkeletonLabels<IndexType>>>>,
    ) {
        let out = unsafe { image.get_at_mut::<LabelRec>(recs_base, recs.len()) };
        out.copy_from_slice(recs);
        for rec in recs {
            // Take the oracle out so its arrays are freed right after they are copied.
            let label = labels[rec.local as usize]
                .take()
                .expect("label directory entry without an oracle");
            let layout = LabelLayout::of::<IndexType>(&rec.sizes);
            let base = blob_base + rec.blob_off as usize;
            let n = rec.sizes.nodes as usize;
            unsafe {
                image
                    .get_at_mut::<DirectedNodeIndex<IndexType>>(base + layout.node_of_id, n)
                    .copy_from_slice(&label.node_of_id);
                let slots = image.get_at_mut::<NodeSlot<IndexType>>(
                    base + layout.hash,
                    rec.sizes.hash_slots as usize,
                );
                slots.fill(NodeSlot::empty());
                hash::build_nodes_into(
                    slots,
                    label
                        .node_of_id
                        .iter()
                        .enumerate()
                        .map(|(id, &node)| (node_key(node), id as u32)),
                );
                image
                    .get_at_mut::<GfaPathLength<IndexType>>(base + layout.len_of_id, n)
                    .copy_from_slice(&label.len_of_id);
                image
                    .get_at_mut::<u32>(base + layout.offsets, n + 1)
                    .copy_from_slice(&label.l_out.offsets);
                if rec.sizes.narrow != 0 {
                    let n_entries = label.l_out.flat.len();
                    let hubs = image.get_at_mut::<u16>(base + layout.flat, n_entries);
                    let values = image
                        .get_at_mut::<GfaPathLength<IndexType>>(base + layout.values, n_entries);
                    for (k, e) in label.l_out.flat.iter().enumerate() {
                        hubs[k] = e.hub.get() as u16;
                        values[k] = e.distance.get();
                    }
                } else {
                    image
                        .get_at_mut::<LabelEntry<IndexType>>(
                            base + layout.flat,
                            label.l_out.flat.len(),
                        )
                        .copy_from_slice(&label.l_out.flat);
                }
            }
            drop(label);
        }
    }

    /// Read a table of `count` oracles.
    pub fn read(image: &'a Image, recs_base: usize, blob_base: usize, count: usize) -> Self {
        Self {
            image,
            recs: image.get_at(recs_base, count),
            blob_base,
            _t: std::marker::PhantomData,
        }
    }

    /// The oracle of block-local SPQR node `local`, if it has one.
    pub fn get(&self, local: usize) -> Option<LabelsView<'a, IndexType>> {
        let i = self
            .recs
            .binary_search_by_key(&(local as u32), |r| r.local)
            .ok()?;
        Some(self.view(&self.recs[i]))
    }

    fn view(&self, rec: &LabelRec) -> LabelsView<'a, IndexType> {
        let layout = LabelLayout::of::<IndexType>(&rec.sizes);
        let base = self.blob_base + rec.blob_off as usize;
        let n = rec.sizes.nodes as usize;
        LabelsView {
            id: (self.image.instance_id() << 32) | rec.ordinal as u64,
            node_of_id: self.image.get_at(base + layout.node_of_id, n),
            hash: self
                .image
                .get_at(base + layout.hash, rec.sizes.hash_slots as usize),
            len_of_id: self.image.get_at(base + layout.len_of_id, n),
            offsets: self.image.get_at(base + layout.offsets, n + 1),
            flat: if rec.sizes.narrow != 0 {
                LabelFlat::Narrow {
                    hubs: self
                        .image
                        .get_at(base + layout.flat, rec.sizes.entries as usize),
                    values: self
                        .image
                        .get_at(base + layout.values, rec.sizes.entries as usize),
                }
            } else {
                LabelFlat::Wide(
                    self.image
                        .get_at(base + layout.flat, rec.sizes.entries as usize),
                )
            },
        }
    }
}
