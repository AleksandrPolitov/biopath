//! Where a non-cut node's cut rows are: the row forms ([`RowRef`]), the header words of the lift's
//! row table and the build-time relayout of the records onto that table.

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};

use super::{
    NodeTopology, TopologyMeta,
    record::{HAS_CELLS, K_CUT, K_PLAIN, K_RECORD, K_SIMPLE, LEN_ESCAPE, NodeRecord},
};
use crate::{
    image::{Image, Sec},
    min_plus::MinPlus,
    path::GfaPathLength,
};

/// Row-field bit of a non-cut node whose rows are a closed form (see [`RowRef::Closed`]); the low
/// bits are its first-hop mask. A stored row start is below it.
pub const ROW_CLOSED: u32 = 1 << 30;
/// Bits of a closed-form first-hop mask per orientation: bit `2i + p` = an arc to the block's cut `i`
/// (cut-list order) in orientation `p`; orientation `o`'s bits start at `o * ROW_MASK_BITS`.
pub const ROW_MASK_BITS: u32 = 6;

/// Where a non-cut node's cut rows are (see [`crate::node_rows`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowRef {
    /// Stored: the node's first entry in the lift's row table.
    Stored(usize),
    /// A simple node (every arc leaves to a cut of its block) whose cells are
    /// `len(v) + min over the arcs y of v^o of M_aug(y -> w^q)`; the mask lists the arcs' directed
    /// cuts (see [`ROW_MASK_BITS`]). The build stores this form only where it equals the rows.
    Closed { mask: u32 },
}

/// Row-table slots per header word: one when the index type holds every `u32` below its infinity
/// (a header word never equals `u32::MAX`), else enough `(bits - 1)`-bit chunks.
#[inline]
pub fn header_slots<IndexType: GraphIndexInteger>() -> usize {
    let bits = 8 * std::mem::size_of::<IndexType>();
    if bits >= 32 {
        1
    } else {
        32usize.div_ceil(bits - 1)
    }
}

/// Append header word `value` to the row table (see [`header_slots`]).
pub fn push_header_word<IndexType: GraphIndexInteger>(
    rows: &mut Vec<MinPlus<IndexType>>,
    value: u32,
) {
    let slots = header_slots::<IndexType>();
    if slots == 1 {
        assert_ne!(value, u32::MAX, "a header word is below the infinity");
        rows.push(MinPlus::finite(GfaPathLength::from_usize(value as usize)));
        return;
    }
    let chunk = 8 * std::mem::size_of::<IndexType>() - 1;
    for s in 0..slots {
        let part = (u64::from(value) >> (s * chunk)) & ((1u64 << chunk) - 1);
        rows.push(MinPlus::finite(GfaPathLength::from_usize(part as usize)));
    }
}

/// Read the header word at slot `at` (see [`push_header_word`]).
#[inline]
pub(super) fn read_header_word<IndexType: GraphIndexInteger>(
    rows: &[MinPlus<IndexType>],
    at: usize,
) -> u32 {
    let slots = header_slots::<IndexType>();
    let part = |i: usize| {
        rows[i]
            .into_option()
            .expect("a header word is finite")
            .into_usize() as u64
    };
    if slots == 1 {
        return part(at) as u32;
    }
    let chunk = 8 * std::mem::size_of::<IndexType>() - 1;
    (0..slots).fold(0u64, |acc, s| acc | part(at + s) << (s * chunk)) as u32
}

/// How the lift's row layout settles one non-cut node (see [`NodeTopology::rewrite_row_starts`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeRows {
    /// No header: a node outside a record block whose rows are a closed form (`mask`) or absent (`0`).
    Simple { mask: u32 },
    /// A header at `header` in the row table (`[block]`, or `[block, location]` for a record-block
    /// node); `cells` = stored cells follow it; `mask` = a record-block node's closed form.
    Header { header: u32, cells: bool, mask: u32 },
}

impl<'a, IndexType: GraphIndexInteger> NodeTopology<'a, IndexType> {
    /// Header word `i` of the node whose header starts at slot `header` (see [`header_slots`]).
    #[inline]
    pub(super) fn header_word(&self, header: usize, i: usize) -> usize {
        read_header_word(self.rows, header + i * header_slots::<IndexType>()) as usize
    }

    /// Where a non-cut node's cut rows are, if it has any.
    #[inline]
    pub fn row_ref(&self, node: NodeIndex<IndexType>) -> Option<RowRef> {
        let record = &self.records[node.into_usize()];
        let header_len = header_slots::<IndexType>()
            * match record.kind() {
                K_PLAIN => 1,
                K_RECORD if !self.rows.is_empty() => 2,
                K_SIMPLE | K_RECORD => {
                    return (record.mask() != 0).then_some(RowRef::Closed {
                        mask: record.mask(),
                    });
                }
                _ => return None,
            };
        if record.lo & HAS_CELLS != 0 {
            Some(RowRef::Stored(record.id as usize + header_len))
        } else {
            (record.mask() != 0).then_some(RowRef::Closed {
                mask: record.mask(),
            })
        }
    }

    /// Total cut-row entries laid out (the lift's row table has exactly this many).
    pub fn row_total(&self) -> usize {
        self.row_total
    }

    /// Build-time only: settle every non-cut node from the lift's row layout (`rows[node]`, see
    /// [`NodeRows`]): a simple node keeps its block and takes its closed-form mask; a node with a
    /// header becomes plain (outside a record block) or stays a record-block node, pointing at its
    /// header in the row table `rows_sec` (`row_total` entries), which the meta records. The caller
    /// must not read this view afterwards: it re-resolves the sections with
    /// [`super::TopologySections::view`].
    pub fn rewrite_row_starts(
        &self,
        image: &Image,
        rows: &[NodeRows],
        rows_sec: Sec<MinPlus<IndexType>>,
        row_total: u64,
    ) {
        let n = self.node_count();
        assert_eq!(rows.len(), n, "one row layout per node");
        // SAFETY: the ranges are sections this build pushed into `image`; the only live view of them
        // is this one, whose records the caller no longer reads.
        unsafe {
            let recs =
                image.get_at_mut::<NodeRecord<IndexType>>(self.records_sec.off as usize, n + 1);
            for (rec, layout) in recs.iter_mut().zip(rows) {
                let kind = rec.kind();
                if kind == K_CUT {
                    continue;
                }
                let len = (rec.lo & LEN_ESCAPE) as usize;
                *rec = match (*layout, kind) {
                    (NodeRows::Simple { mask }, K_SIMPLE) => {
                        NodeRecord::new(len, K_SIMPLE, mask, rec.id)
                    }
                    (
                        NodeRows::Header {
                            header,
                            cells,
                            mask,
                        },
                        K_SIMPLE | K_RECORD,
                    ) => {
                        let mut record = NodeRecord::new(
                            len,
                            if kind == K_RECORD { K_RECORD } else { K_PLAIN },
                            mask,
                            header,
                        );
                        if cells {
                            record.lo |= HAS_CELLS;
                        }
                        record
                    }
                    (layout, kind) => {
                        panic!("NodeTopology: row layout {layout:?} for a node of kind {kind}")
                    }
                };
            }
            let meta = &mut image.get_at_mut::<TopologyMeta>(self.meta_sec.off as usize, 1)[0];
            meta.row_total = row_total;
            meta.rows_off = rows_sec.off;
            meta.rows_len = rows_sec.len;
        }
    }
}
