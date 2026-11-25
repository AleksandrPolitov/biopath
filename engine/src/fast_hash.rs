//! A fast, non-cryptographic hasher for the DP hot paths.
//!
//! The inside DP runs millions of tiny skeleton Dijkstras, each keyed on integer
//! [`DirectedNodeIndex`] values. The standard library's default `HashMap` uses SipHash, which is
//! cryptographically strong but far too slow when the same small maps are built and probed millions
//! of times. This module provides the classic `FxHash` algorithm (the one used by rustc) as a
//! `BuildHasher`, plus the [`FastMap`] alias, so those hot maps hash an integer key in a
//! couple of instructions instead of a full SipHash round.
//!
//! [`DirectedNodeIndex`]: bidirected_adjacency_array::index::DirectedNodeIndex

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// The FxHash hasher: a rotate-xor-multiply mix, identical to rustc's `FxHasher`.
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback for byte slices: fold 8 bytes at a time, then the tail.
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        for &byte in chunks.remainder() {
            self.add(u64::from(byte));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

/// A `BuildHasher` that produces [`FxHasher`]s.
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// A `HashMap` using the fast [`FxHasher`]. Drop-in for `std::collections::HashMap` in hot paths.
pub type FastMap<K, V> = HashMap<K, V, FxBuildHasher>;
