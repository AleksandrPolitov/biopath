//! One distance of the typed min-plus semiring, [`MinPlus`] (see [`super`] for the conventions).
//!
//! Addition `⊕` is `min`, multiplication `⊗` is `+`. The additive identity is
//! [`MinPlus::infinity`] (no path), the multiplicative identity is [`MinPlus::zero`] (empty path).
//! Infinity absorbs under `⊗`: `∞ ⊗ x = x ⊗ ∞ = ∞`. "No path" is a value, so it composes with
//! path concatenation (`⊗`) and the cheaper alternative (`⊕`) without a sentinel overflowing.
//! [`relax`] is `⊕=` on an optional length, the form a query's answer takes.

use std::cmp::Ordering;

use bidirected_adjacency_array::index::GraphIndexInteger;

use super::word;
use crate::path::{GfaPathLength, OptionalGfaPathLength};

/// A value in the min-plus semiring over [`GfaPathLength`], where `None` represents `+∞`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct MinPlus<IndexType: GraphIndexInteger> {
    /// `IndexType::MAX` represents `+∞`; every other value is a finite path length. Path-length
    /// indices already reserve `MAX`, so this has exactly the same value domain as the former
    /// `Option<GfaPathLength>` representation at half the size for `u32`.
    value: OptionalGfaPathLength<IndexType>,
}

impl<IndexType: GraphIndexInteger> MinPlus<IndexType> {
    /// The additive identity `+∞` (no path).
    pub fn infinity() -> Self {
        Self {
            value: OptionalGfaPathLength::new_none(),
        }
    }

    /// The multiplicative identity `0` (empty path).
    pub fn zero() -> Self {
        Self {
            value: OptionalGfaPathLength::from(GfaPathLength::from_usize(0)),
        }
    }

    /// A finite value.
    pub fn finite(length: GfaPathLength<IndexType>) -> Self {
        Self {
            value: OptionalGfaPathLength::from(length),
        }
    }

    /// A finite value given as a `usize`.
    #[cfg(test)]
    pub fn finite_usize(length: usize) -> Self {
        Self::finite(GfaPathLength::from_usize(length))
    }

    /// The value of a raw word (`IndexType::MAX` = `+∞`, see [`super::word`]): the same bits.
    #[inline]
    pub fn from_word(word: IndexType) -> Self {
        Self {
            value: OptionalGfaPathLength::from(word::length(word)),
        }
    }

    /// Whether this value is `+∞` (no path).
    pub fn is_infinite(&self) -> bool {
        self.value.is_none()
    }

    /// The finite value, or `None` if `+∞`.
    pub fn into_option(self) -> Option<GfaPathLength<IndexType>> {
        self.value.into_option()
    }

    /// Semiring addition `⊕ = min`. `+∞` is the identity.
    pub fn oplus(self, other: Self) -> Self {
        match (self.value.into_option(), other.value.into_option()) {
            (None, value) | (value, None) => Self {
                value: OptionalGfaPathLength::from(value),
            },
            (Some(a), Some(b)) => Self {
                value: OptionalGfaPathLength::from(if a <= b { a } else { b }),
            },
        }
    }

    /// Semiring multiplication `⊗ = +`. `+∞` absorbs.
    pub fn otimes(self, other: Self) -> Self {
        match (self.value.into_option(), other.value.into_option()) {
            (Some(a), Some(b)) => Self {
                value: OptionalGfaPathLength::from(a + b),
            },
            _ => Self::infinity(),
        }
    }

    /// In-place `⊕` (keep the cheaper of the two).
    pub fn min_assign(&mut self, other: Self) {
        *self = self.oplus(other);
    }
}

/// `best ⊕= candidate` on an optional length (`None` = `+∞`): lower `best` to `candidate` if it is
/// smaller (or `best` is unset).
pub fn relax<IndexType: GraphIndexInteger>(
    best: &mut Option<GfaPathLength<IndexType>>,
    candidate: GfaPathLength<IndexType>,
) {
    if best.is_none_or(|current| candidate < current) {
        *best = Some(candidate);
    }
}

// Total order in which `+∞` is the greatest element, so that `oplus` really is `min` under this
// order and `min`/`max` on collections behave as expected.
impl<IndexType: GraphIndexInteger> PartialOrd for MinPlus<IndexType> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<IndexType: GraphIndexInteger> Ord for MinPlus<IndexType> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.value.into_option(), other.value.into_option()) {
            (None, None) => Ordering::Equal,
            // `None` (`+∞`) is the greatest element.
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(a), Some(b)) => a.cmp(&b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type M = MinPlus<u32>;

    fn f(x: usize) -> M {
        MinPlus::finite_usize(x)
    }

    fn inf() -> M {
        MinPlus::infinity()
    }

    fn val(m: M) -> Option<usize> {
        m.into_option().map(|l| l.into_usize())
    }

    #[test]
    fn constructors_and_predicates() {
        assert!(inf().is_infinite());
        assert!(!f(0).is_infinite());
        assert!(!M::zero().is_infinite());
        assert_eq!(val(M::zero()), Some(0));
        assert_eq!(val(f(7)), Some(7));
        assert_eq!(val(inf()), None);
    }

    #[test]
    fn oplus_is_min_with_infinity_identity() {
        // `+∞` is the additive identity.
        assert_eq!(inf().oplus(f(5)), f(5));
        assert_eq!(f(5).oplus(inf()), f(5));
        assert_eq!(inf().oplus(inf()), inf());

        // `⊕` takes the smaller finite value.
        assert_eq!(f(3).oplus(f(5)), f(3));
        assert_eq!(f(5).oplus(f(3)), f(3));
        assert_eq!(f(4).oplus(f(4)), f(4));
    }

    #[test]
    fn otimes_is_addition_with_infinity_absorbing() {
        // `0` is the multiplicative identity.
        assert_eq!(M::zero().otimes(f(9)), f(9));
        assert_eq!(f(9).otimes(M::zero()), f(9));

        // `⊗` adds finite values.
        assert_eq!(f(3).otimes(f(5)), f(8));

        // `+∞` absorbs.
        assert_eq!(inf().otimes(f(5)), inf());
        assert_eq!(f(5).otimes(inf()), inf());
        assert_eq!(inf().otimes(inf()), inf());
    }

    #[test]
    fn oplus_is_commutative_and_associative() {
        let values = [inf(), f(0), f(2), f(5), f(9)];
        for &a in &values {
            for &b in &values {
                assert_eq!(a.oplus(b), b.oplus(a), "commutativity {a:?} {b:?}");
                for &c in &values {
                    assert_eq!(
                        a.oplus(b).oplus(c),
                        a.oplus(b.oplus(c)),
                        "associativity {a:?} {b:?} {c:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn otimes_is_associative() {
        let values = [inf(), f(0), f(1), f(4), f(6)];
        for &a in &values {
            for &b in &values {
                for &c in &values {
                    assert_eq!(
                        a.otimes(b).otimes(c),
                        a.otimes(b.otimes(c)),
                        "associativity {a:?} {b:?} {c:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn distributes_multiplication_over_addition() {
        // Semiring law: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c), including with `+∞`.
        let values = [inf(), f(0), f(2), f(3), f(7)];
        for &a in &values {
            for &b in &values {
                for &c in &values {
                    assert_eq!(
                        a.otimes(b.oplus(c)),
                        a.otimes(b).oplus(a.otimes(c)),
                        "left distributivity {a:?} {b:?} {c:?}"
                    );
                    assert_eq!(
                        b.oplus(c).otimes(a),
                        b.otimes(a).oplus(c.otimes(a)),
                        "right distributivity {a:?} {b:?} {c:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn ordering_treats_infinity_as_greatest() {
        assert!(f(0) < f(1));
        assert!(f(5) < inf());
        assert!(inf() > f(1000));
        assert_eq!(inf().cmp(&inf()), Ordering::Equal);

        // `oplus` agrees with the `min` under this order.
        let values = [inf(), f(0), f(2), f(8)];
        for &a in &values {
            for &b in &values {
                assert_eq!(a.oplus(b), a.min(b));
            }
        }
    }

    #[test]
    fn min_assign_keeps_cheaper() {
        let mut m = inf();
        m.min_assign(f(5));
        assert_eq!(m, f(5));
        m.min_assign(f(9));
        assert_eq!(m, f(5));
        m.min_assign(f(2));
        assert_eq!(m, f(2));
    }

    #[test]
    fn u32_min_plus_is_one_word() {
        assert_eq!(
            std::mem::size_of::<MinPlus<u32>>(),
            std::mem::size_of::<u32>(),
            "infinity must use the path-length sentinel, not a full-width Option tag",
        );
    }
}
