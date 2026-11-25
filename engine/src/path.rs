//! Path lengths: [`GfaPathLength`] (and its optional form) with the arithmetic between lengths and
//! node offsets ([`GfaNodeOffset`]).

use std::ops::{Add, AddAssign, Sub, SubAssign};

use bidirected_adjacency_array::index::GraphIndexInteger;
use optional_numeric_index::implement_generic_index;

use crate::location::GfaNodeOffset;

implement_generic_index!(pub GfaPathLength, pub OptionalGfaPathLength);

impl<IndexType: GraphIndexInteger> GfaPathLength<IndexType> {
    pub fn into_offset(self) -> GfaNodeOffset<IndexType> {
        GfaNodeOffset::from_raw(self.into_raw())
    }

    /// `self + rhs`, or `None` when the sum does not fit: it overflows `IndexType` or reaches
    /// `IndexType::MAX`, which is reserved for "no path". The `+` operator does not check (it is on
    /// the query hot path, and in release builds it wraps); the graph checks of
    /// [`crate::graph_limits`] keep the index's sums in range.
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        self.into_raw()
            .checked_add(&rhs.into_raw())
            .filter(|&sum| sum != IndexType::max_value())
            .map(Self::from_raw)
    }
}

impl<IndexType: GraphIndexInteger> Sub for GfaNodeOffset<IndexType> {
    type Output = GfaPathLength<IndexType>;

    fn sub(self, rhs: Self) -> Self::Output {
        GfaPathLength::from_raw(self.into_raw() - rhs.into_raw())
    }
}

impl<IndexType: GraphIndexInteger> AddAssign for GfaNodeOffset<IndexType> {
    fn add_assign(&mut self, rhs: Self) {
        *self = GfaNodeOffset::from_raw(self.into_raw() + rhs.into_raw());
    }
}

impl<IndexType: GraphIndexInteger> SubAssign for GfaNodeOffset<IndexType> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = GfaNodeOffset::from_raw(self.into_raw() - rhs.into_raw());
    }
}

impl<IndexType: GraphIndexInteger> Sub<GfaNodeOffset<IndexType>> for GfaPathLength<IndexType> {
    type Output = GfaNodeOffset<IndexType>;

    fn sub(self, rhs: GfaNodeOffset<IndexType>) -> Self::Output {
        GfaNodeOffset::from_raw(self.into_raw() - rhs.into_raw())
    }
}

impl<IndexType: GraphIndexInteger> Add for GfaPathLength<IndexType> {
    type Output = GfaPathLength<IndexType>;

    fn add(self, rhs: Self) -> Self::Output {
        GfaPathLength::from_raw(self.into_raw() + rhs.into_raw())
    }
}

impl<IndexType: GraphIndexInteger> Sub for GfaPathLength<IndexType> {
    type Output = GfaPathLength<IndexType>;

    fn sub(self, rhs: Self) -> Self::Output {
        GfaPathLength::from_raw(self.into_raw() - rhs.into_raw())
    }
}
