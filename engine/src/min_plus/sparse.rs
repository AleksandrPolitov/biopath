//! Per-node `4x4` min-plus matrices in the image's sparse form ([`SparseMat4`]): only the finite
//! entries are stored, and equal matrices share one run of values.

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::{MinPlus, MinPlusMatrix, word::inf};
use crate::fast_hash::FastMap;

/// Per-node `4x4` min-plus matrices as persisted (the pole-lift steps and the LCA role's `D_out`):
/// node `v`'s matrix has its finite entries where `masks[v]` has a bit (`4 * row + col`), stored
/// row-major from `values[offsets[v]]`; mask `0` is the all-`∞` matrix (or "none"). A dense matrix is
/// 64 B and most entries are `∞` (about 4 of 16 for the steps on pangenome graphs), and equal matrices
/// are common — every child of a P-node lifts through the same pole pair — so equal ones share one run.
pub struct SparseMat4<IndexType: GraphIndexInteger> {
    pub masks: Vec<u16>,
    pub offsets: Vec<u32>,
    pub values: Vec<MinPlus<IndexType>>,
}

/// The diagonal bits (`4k + k`) of a [`SparseMat4`] mask.
pub const DIAG_MASK: u16 = 0x8421;

impl<IndexType: GraphIndexInteger> SparseMat4<IndexType> {
    /// Encode one matrix per node (`None` = mask `0`).
    pub fn encode<'m>(
        matrices: impl Iterator<Item = Option<&'m MinPlusMatrix<IndexType, 4>>>,
    ) -> Self
    where
        IndexType: 'm,
    {
        Self::encode_values(matrices, u16::MAX)
    }

    /// [`Self::encode`] storing values only for the finite entries whose bit is in `stored`: the
    /// mask still records every finite entry, and decoding (with the same `stored`) reads the others
    /// as zero — for entries no reader uses (the `D_out` diagonal).
    pub fn encode_values<'m>(
        matrices: impl Iterator<Item = Option<&'m MinPlusMatrix<IndexType, 4>>>,
        stored: u16,
    ) -> Self
    where
        IndexType: 'm,
    {
        let (lower, _) = matrices.size_hint();
        let mut table = Self {
            masks: Vec::with_capacity(lower),
            offsets: Vec::with_capacity(lower),
            values: Vec::new(),
        };
        // Distinct matrices by their 16 raw entries (`MAX` = `∞`, unstored finite = `0`), so equal
        // ones share one run.
        let mut runs: FastMap<[IndexType; 16], u32> = Default::default();
        for matrix in matrices {
            let mut key = [inf(); 16];
            let mut mask = 0u16;
            if let Some(matrix) = matrix {
                for (k, cell) in key.iter_mut().enumerate() {
                    if let Some(d) = matrix.get(k / 4, k % 4).into_option() {
                        *cell = if stored >> k & 1 == 1 {
                            d.into_raw()
                        } else {
                            IndexType::zero()
                        };
                        mask |= 1 << k;
                    }
                }
            }
            let offset = match matrix {
                Some(matrix) if mask != 0 => {
                    let values = &mut table.values;
                    *runs.entry(key).or_insert_with(|| {
                        let offset = values.len() as u32;
                        values.extend(
                            (0..16)
                                .filter(|k| (mask & stored) >> k & 1 == 1)
                                .map(|k| matrix.get(k / 4, k % 4)),
                        );
                        offset
                    })
                }
                _ => 0,
            };
            table.masks.push(mask);
            table.offsets.push(offset);
        }
        table
    }
}

/// Decode node `v` of a [`SparseMat4`] table held as three slices.
#[inline]
pub fn decode<IndexType: GraphIndexInteger>(
    masks: &[u16],
    offsets: &[u32],
    values: &[MinPlus<IndexType>],
    v: usize,
) -> MinPlusMatrix<IndexType, 4> {
    decode_values(masks, offsets, values, v, u16::MAX)
}

/// Decode node `v` of a table written by [`SparseMat4::encode_values`] with the same `stored`.
#[inline]
pub fn decode_values<IndexType: GraphIndexInteger>(
    masks: &[u16],
    offsets: &[u32],
    values: &[MinPlus<IndexType>],
    v: usize,
    stored: u16,
) -> MinPlusMatrix<IndexType, 4> {
    let mut matrix = MinPlusMatrix::infinity();
    let mut bits = masks[v] & stored;
    let mut at = offsets[v] as usize;
    while bits != 0 {
        let k = bits.trailing_zeros() as usize;
        matrix.set(k / 4, k % 4, values[at]);
        at += 1;
        bits &= bits - 1;
    }
    let mut zeros = masks[v] & !stored;
    while zeros != 0 {
        let k = zeros.trailing_zeros() as usize;
        matrix.set(k / 4, k % 4, MinPlus::zero());
        zeros &= zeros - 1;
    }
    matrix
}

/// `vector · M` for node `v`'s matrix, straight from the sparse entries: an `∞` entry contributes
/// nothing to `min_i vector[i] + M[i][j]`, so only the stored ones are visited (no matrix is
/// materialised).
#[inline]
pub fn left_multiply<IndexType: GraphIndexInteger>(
    masks: &[u16],
    offsets: &[u32],
    values: &[MinPlus<IndexType>],
    v: usize,
    vector: &[MinPlus<IndexType>; 4],
) -> [MinPlus<IndexType>; 4] {
    let mut out = [MinPlus::infinity(); 4];
    let mut mask = masks[v];
    let mut at = offsets[v] as usize;
    while mask != 0 {
        let k = mask.trailing_zeros() as usize;
        out[k % 4].min_assign(vector[k / 4].otimes(values[at]));
        at += 1;
        mask &= mask - 1;
    }
    out
}
