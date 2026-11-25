//! The raw cut-to-cut matrix `M_B` of a block: the in-block distances between its directed cuts,
//! read through the block's engine.

use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex};
use rayon::prelude::*;
use spqr_tree::decomposition::indices::BlockIndex;

use super::{CutMatrixFill, IntraDistance};
use crate::{min_plus::MinPlus, orientation::Orientation};

/// From this many entries on (about 160 cuts), a block's `M_B` comes from its engine's one-pass
/// fill instead of one in-block query per entry.
const BIG_BLOCK_ENTRIES: usize = 100_000;

/// The entries of a one-pass fill cross-checked against single in-block queries.
const CROSS_CHECK_SAMPLES: usize = 512;

/// The raw cut-to-cut matrix `M_B` of block `bid` over its directed cuts (`(2c)^2`, row-major,
/// `cuts` order, `2 * i + o`, zero diagonal).
///
/// A small block asks `intra` for every entry. A big one takes the engine's one-pass fill
/// `cut_matrix` (each cut's pole projection warmed once, every pair composed through the cache),
/// cross-checked on [`CROSS_CHECK_SAMPLES`] entries against `intra`; on any mismatch the exact
/// dense fill (every entry through `intra`, rows across the pool) replaces it, with a warning, so no
/// wrong `M_B` can ship.
pub(super) fn raw_cut_matrix<IndexType: GraphIndexInteger + Send + Sync>(
    bid: usize,
    cuts: &[NodeIndex<IndexType>],
    block: BlockIndex<IndexType>,
    intra: &(impl IntraDistance<IndexType> + Sync),
    cut_matrix: &(impl CutMatrixFill<IndexType> + Sync),
) -> Vec<MinPlus<IndexType>> {
    let cells = CutCells { cuts, block, intra };
    if cells.dim() * cells.dim() >= BIG_BLOCK_ENTRIES {
        checked_fill(bid, &cells, cut_matrix)
    } else {
        cells.pairwise_fill()
    }
}

/// A big block's `M_B`: the one-pass fill if every sample matches `intra`, else the dense fill.
fn checked_fill<
    IndexType: GraphIndexInteger + Send + Sync,
    Intra: IntraDistance<IndexType> + Sync,
>(
    bid: usize,
    cells: &CutCells<'_, IndexType, Intra>,
    cut_matrix: &impl CutMatrixFill<IndexType>,
) -> Vec<MinPlus<IndexType>> {
    let entries = cells.dim() * cells.dim();
    let matrix = cut_matrix(cells.block, cells.cuts);
    let samples: Vec<usize> = (0..entries)
        .step_by((entries / CROSS_CHECK_SAMPLES).max(1))
        .collect();
    let mismatches = cells.mismatches(&matrix, &samples);
    if mismatches == 0 {
        matrix
    } else {
        log::warn!(
            "block {bid}: the one-pass cut matrix differs from in-block queries on {mismatches} of \
             {} samples; filling it entry by entry",
            samples.len(),
        );
        cells.dense_fill()
    }
}

/// The entries of one block's `M_B`, each through an in-block query.
struct CutCells<'c, IndexType: GraphIndexInteger, Intra> {
    cuts: &'c [NodeIndex<IndexType>],
    block: BlockIndex<IndexType>,
    intra: &'c Intra,
}

impl<IndexType, Intra> CutCells<'_, IndexType, Intra>
where
    IndexType: GraphIndexInteger + Send + Sync,
    Intra: IntraDistance<IndexType> + Sync,
{
    fn dim(&self) -> usize {
        2 * self.cuts.len()
    }

    /// The directed cut of row or column `i` (`2 * cut + orientation`).
    fn directed(&self, i: usize) -> DirectedNodeIndex<IndexType> {
        self.cuts[i / 2].into_directed(Orientation::from_index(i % 2).is_forward())
    }

    /// The entry `(row, col)`: `0` on the diagonal, else the in-block distance.
    fn entry(&self, row: usize, col: usize) -> MinPlus<IndexType> {
        let (source, target) = (self.directed(row), self.directed(col));
        if source == target {
            MinPlus::zero()
        } else {
            (self.intra)(source, target, self.block)
                .map(MinPlus::finite)
                .unwrap_or_else(MinPlus::infinity)
        }
    }

    /// Every entry, row by row on the calling thread.
    fn pairwise_fill(&self) -> Vec<MinPlus<IndexType>> {
        let dim = self.dim();
        (0..dim * dim)
            .map(|i| self.entry(i / dim, i % dim))
            .collect()
    }

    /// Every entry, the rows across the pool.
    fn dense_fill(&self) -> Vec<MinPlus<IndexType>> {
        let dim = self.dim();
        let rows: Vec<Vec<MinPlus<IndexType>>> = (0..dim)
            .into_par_iter()
            .map(|row| (0..dim).map(|col| self.entry(row, col)).collect())
            .collect();
        rows.into_iter().flatten().collect()
    }

    /// The `samples` (row-major indices) at which `matrix` differs from the in-block query, the
    /// samples across the pool (each query may be a skeleton search on a giant block; the fill
    /// left its caches warm).
    fn mismatches(&self, matrix: &[MinPlus<IndexType>], samples: &[usize]) -> usize {
        let dim = self.dim();
        samples
            .par_iter()
            .filter(|&&i| matrix[i] != self.entry(i / dim, i % dim))
            .count()
    }
}
