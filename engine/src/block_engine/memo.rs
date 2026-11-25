//! The query path's thread-local memos: label merges and pole projections.
//!
//! Both memoize pure functions of the immutable record, so they are exact. They are thread-local
//! because a `RefCell` field would make the engine `!Sync`, and the `M_B` fill shares one engine
//! across the pool; each thread holds its own copy, typed `Any` since the engine is generic over the
//! index word.

use std::{any::Any, cell::RefCell, collections::HashMap};

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger};

use crate::{path::GfaPathLength, pole_lift::PoleVector, skeleton_index::LabelsAccess};

/// The label-merge memo's bound: thousands of queries' distinct triples (about 12 per query); on
/// overflow it clears, which is cheap next to the merges it saves.
const MERGE_MEMO_CAP: usize = 65_536;

type MergeMemo<IndexType> = HashMap<(u64, u32, u32), Option<GfaPathLength<IndexType>>>;

thread_local! {
    /// The label-merge memo: `distance_id(seed, target)` keyed by the oracle's process-unique id (a
    /// pointer could be reused by a later oracle). A labeled read repeats most of its merges, and a
    /// hash probe costs a fraction of a merge.
    static MERGE_MEMO: RefCell<Option<Box<dyn Any>>> = const { RefCell::new(None) };
}

/// Memoized `labels.distance_id(sid, tid)`.
pub(super) fn memoized_distance_id<
    IndexType: GraphIndexInteger + 'static,
    L: LabelsAccess<IndexType> + ?Sized,
>(
    labels: &L,
    sid: u32,
    tid: u32,
) -> Option<GfaPathLength<IndexType>> {
    let key = (labels.id(), sid, tid);
    MERGE_MEMO.with(|cell| {
        let mut slot = cell.borrow_mut();
        if !slot
            .as_ref()
            .is_some_and(|m| m.is::<MergeMemo<IndexType>>())
        {
            *slot = Some(Box::new(MergeMemo::<IndexType>::new()));
        }
        let Some(memo) = slot
            .as_mut()
            .and_then(|m| m.downcast_mut::<MergeMemo<IndexType>>())
        else {
            return labels.distance_id(sid, tid);
        };
        if let Some(&cached) = memo.get(&key) {
            return cached;
        }
        if memo.len() > MERGE_MEMO_CAP {
            memo.clear();
        }
        let value = labels.distance_id(sid, tid);
        memo.insert(key, value);
        value
    })
}

/// The projection memo's bound (about 3 MB per thread); a graph with more live keys degrades to
/// periodic clears, which costs nothing where nothing repeats.
const MEMO_CAP: usize = 65_536;

/// Skip the projection memo when `x`'s inner skeleton has fewer than this many nodes: the search is
/// then cheaper than the probe (ungated, the memo cost 8% query time on a graph of a million tiny
/// blocks). The same cutoff as a skeleton's intern map.
pub(super) const MEMO_MIN_SKELETON: usize = 32;

/// `(image instance id, block, seed, x)`.
type ProjMemoKey<IndexType> = (u64, usize, DirectedNodeIndex<IndexType>, usize);
type ProjMemo<IndexType> = HashMap<ProjMemoKey<IndexType>, PoleVector<IndexType>>;

thread_local! {
    /// The projection memo: `pole_projection(seed, x)` keyed by the block too (`x` is block-local),
    /// so a block's cut projections stay warm for every later query through it, and by the image's
    /// process-unique id: two indexes in one process share the thread's memo (a query thread, or a
    /// rayon thread of two builds) and number their blocks alike. The `M_B` fill warms its cache
    /// through this memo, then reads only that cache.
    static PROJ_MEMO: RefCell<Option<Box<dyn Any>>> = const { RefCell::new(None) };
}

/// Run `f` with this thread's projection memo, clearing it if it has grown past [`MEMO_CAP`].
/// Returns `None` if the memo is unavailable (a type mismatch).
pub(super) fn with_proj_memo<IndexType: GraphIndexInteger + 'static, R>(
    f: impl FnOnce(&mut ProjMemo<IndexType>) -> R,
) -> Option<R> {
    PROJ_MEMO.with(|cell| {
        let mut slot = cell.borrow_mut();
        if !slot
            .as_ref()
            .is_some_and(|map| map.is::<ProjMemo<IndexType>>())
        {
            *slot = Some(Box::new(ProjMemo::<IndexType>::new()));
        }
        let cache = slot.as_mut()?.downcast_mut::<ProjMemo<IndexType>>()?;
        if cache.len() > MEMO_CAP {
            cache.clear();
        }
        Some(f(cache))
    })
}
