//! Segment orientation (`+` forward / `-` reverse) as a two-valued index type.
//!
//! In a bidirected graph every segment can be traversed in two orientations. A walk arriving at a
//! segment does so in a definite orientation, and continuing the walk requires leaving in a
//! consistent orientation. Orientation is therefore an index into the small distance matrices used
//! throughout the engine.

/// The orientation in which a segment is traversed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Orientation {
    /// The `+` (forward) orientation.
    Forward,
    /// The `-` (reverse / reverse-complement) orientation.
    Reverse,
}

impl Orientation {
    /// The opposite orientation (`+ <-> -`).
    pub fn invert(self) -> Self {
        match self {
            Orientation::Forward => Orientation::Reverse,
            Orientation::Reverse => Orientation::Forward,
        }
    }

    /// Whether this is the forward (`+`) orientation.
    pub fn is_forward(self) -> bool {
        matches!(self, Orientation::Forward)
    }

    /// The matrix index for this orientation (`+` -> 0, `-` -> 1).
    pub fn index(self) -> usize {
        match self {
            Orientation::Forward => 0,
            Orientation::Reverse => 1,
        }
    }

    /// The orientation for a matrix index (`0 -> +`, `1 -> -`).
    ///
    /// # Panics
    /// Panics if `index` is not `0` or `1`.
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Orientation::Forward,
            1 => Orientation::Reverse,
            other => panic!("invalid orientation index {other}"),
        }
    }

    /// Both orientations, in index order.
    pub fn all() -> [Orientation; 2] {
        [Orientation::Forward, Orientation::Reverse]
    }

    /// The GFA sign character for this orientation.
    #[cfg(test)]
    pub fn sign(self) -> char {
        if self.is_forward() { '+' } else { '-' }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_is_an_involution() {
        for o in Orientation::all() {
            assert_eq!(o.invert().invert(), o);
            assert_ne!(o.invert(), o);
        }
    }

    #[test]
    fn index_roundtrips() {
        for o in Orientation::all() {
            assert_eq!(Orientation::from_index(o.index()), o);
        }
        assert_eq!(Orientation::Forward.index(), 0);
        assert_eq!(Orientation::Reverse.index(), 1);
    }

    #[test]
    fn invert_flips_index() {
        for o in Orientation::all() {
            assert_eq!(o.invert().index(), 1 - o.index());
        }
    }

    #[test]
    fn forward_and_sign() {
        assert!(Orientation::Forward.is_forward());
        assert!(!Orientation::Reverse.is_forward());
        assert_eq!(Orientation::Forward.sign(), '+');
        assert_eq!(Orientation::Reverse.sign(), '-');
    }

    #[test]
    #[should_panic]
    fn from_index_rejects_out_of_range() {
        Orientation::from_index(2);
    }
}
