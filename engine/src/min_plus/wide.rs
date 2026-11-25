//! The `u64` working domain: distances the block searches and the S-node ring form compute in
//! `u64`, with `u64::MAX` as `∞`, before they are stored back as words or [`MinPlus`] values.
//!
//! [`widen`] and [`narrow`] convert exactly between a word and this domain;
//! [`MinPlus::from_wide_clamped`] and [`MinPlus::to_wide`] convert a typed value.

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::{MinPlus, word::inf};
use crate::path::GfaPathLength;

/// `∞` (no walk) in the `u64` domain.
pub const INF: u64 = u64::MAX;

/// `a ⊗ b`, `∞` absorbing (finite sums stay far below `u64::MAX`).
#[inline]
pub fn add(a: u64, b: u64) -> u64 {
    if a == INF || b == INF { INF } else { a + b }
}

/// `a ⊗ b ⊗ c`.
#[inline]
pub fn add3(a: u64, b: u64, c: u64) -> u64 {
    add(add(a, b), c)
}

/// The largest finite distance a word holds, in the `u64` domain: one below the word's `∞`
/// (`IndexType::MAX`).
#[inline]
pub fn finite_max<IndexType: GraphIndexInteger>() -> u64 {
    IndexType::max_value()
        .to_u64()
        .expect("an index word fits in u64")
        - 1
}

/// A word in the `u64` domain (`∞` to [`INF`]).
#[inline]
pub fn widen<IndexType: GraphIndexInteger>(word: IndexType) -> u64 {
    if word == inf() {
        INF
    } else {
        GfaPathLength::<IndexType>::from_raw(word).into_usize() as u64
    }
}

/// A `u64` distance as a word ([`INF`] to `∞`); a finite value must fit below the word's `∞`.
#[inline]
pub fn narrow<IndexType: GraphIndexInteger>(value: u64) -> IndexType {
    if value == INF {
        inf()
    } else {
        GfaPathLength::<IndexType>::from_usize(value as usize).into_raw()
    }
}

impl<IndexType: GraphIndexInteger> MinPlus<IndexType> {
    /// A `u64` distance as a typed value: [`INF`] is `+∞`, a finite value is clamped one below the
    /// word's `∞`.
    pub fn from_wide_clamped(value: u64) -> Self {
        let clamp = finite_max::<IndexType>();
        if value == INF {
            Self::infinity()
        } else {
            Self::finite(GfaPathLength::from_usize(value.min(clamp) as usize))
        }
    }

    /// This value in the `u64` domain (`+∞` to [`INF`]).
    pub fn to_wide(self) -> u64 {
        self.into_option()
            .map_or(INF, |length| length.into_usize() as u64)
    }
}
