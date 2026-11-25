//! A flat open-addressing hash table stored inside the image: `u64 key -> u32 value`, linear probing,
//! power-of-two capacity at most half full. Replaces the per-structure `HashMap`s (whose heap layout
//! cannot live in a file) for the sparse `directed node -> dense id` lookups at query seed/read time.
//! One or two cache lines per lookup; the key `u64::MAX` marks an empty slot.

use super::{Padded, image_record};

/// One slot. Plain data; `key == EMPTY_KEY` means unused.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashSlot {
    pub key: u64,
    pub value: u32,
    pub _pad: u32,
}
crate::image::image_record!(HashSlot = [u64, u32, u32]);

pub const EMPTY_KEY: u64 = u64::MAX;

impl HashSlot {
    pub const EMPTY: Self = Self {
        key: EMPTY_KEY,
        value: 0,
        _pad: 0,
    };
}

/// The slot count for `count` keys: a power of two with load factor at most `1/2`.
pub fn capacity_for(count: usize) -> usize {
    (count * 2).max(4).next_power_of_two()
}

/// `splitmix64` finaliser — a fast, well-mixing 64-bit integer hash.
#[inline]
pub fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// FNV-1a over bytes, for string keys (the GFA node names).
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Reserve the empty marker.
    if h == EMPTY_KEY { 0 } else { h }
}

/// Look up a key that is a *hash* (possibly colliding): every slot whose key matches is offered to
/// `accept`, which verifies the candidate value against the real key; the first accepted value wins.
/// Probes at most every slot once, so a table without an empty slot (a corrupt file) ends the
/// search.
pub fn lookup_verified(
    slots: &[HashSlot],
    key: u64,
    mut accept: impl FnMut(u32) -> bool,
) -> Option<u32> {
    if slots.is_empty() {
        return None;
    }
    let mask = slots.len() - 1;
    let mut i = (mix(key) as usize) & mask;
    for _ in 0..slots.len() {
        let slot = &slots[i];
        if slot.key == EMPTY_KEY {
            return None;
        }
        if slot.key == key && accept(slot.value) {
            return Some(slot.value);
        }
        i = (i + 1) & mask;
    }
    None
}

/// Insert allowing duplicate (hash) keys — for [`lookup_verified`] tables.
pub fn build_into_multi(slots: &mut [HashSlot], pairs: impl Iterator<Item = (u64, u32)>) {
    debug_assert!(slots.len().is_power_of_two());
    let mask = slots.len() - 1;
    for (key, value) in pairs {
        let mut i = (mix(key) as usize) & mask;
        while slots[i].key != EMPTY_KEY {
            i = (i + 1) & mask;
        }
        slots[i] = HashSlot {
            key,
            value,
            _pad: 0,
        };
    }
}

/// One slot of a `directed node -> dense id` table (skeletons and label oracles). The key is the
/// directed node's raw index at the graph's own width — 8 B per slot with 32-bit indexes, against 16 B
/// for a [`HashSlot`] — and `I::max_value()` marks an empty slot (no node has that index: it is the
/// `Optional*` none sentinel). `value` is as wide as the index word, so neither word size leaves
/// padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeSlot<I: Copy> {
    pub key: I,
    pub value: Padded<u32, I>,
}
image_record!(impl<I> NodeSlot<I> = [I, Padded<u32, I>]);

impl<I: bidirected_adjacency_array::index::GraphIndexInteger> NodeSlot<I> {
    /// An unused slot.
    pub fn empty() -> Self {
        Self::new(I::max_value(), 0)
    }

    fn new(key: I, value: u32) -> Self {
        Self {
            key,
            value: Padded::new(value),
        }
    }
}

/// The slot count of a [`NodeSlot`] table for `count` keys: a power of two with load factor at most
/// `3/4`. Linear probing over 8-byte slots keeps a lookup within one or two cache lines at that load,
/// as the half-full 16-byte [`HashSlot`] tables do, in about a third of the bytes.
pub fn node_capacity_for(count: usize) -> usize {
    (count * 4).div_ceil(3).max(4).next_power_of_two()
}

/// Fill `slots` (all [`NodeSlot::empty`], power-of-two length) with distinct `(key, value)` pairs.
pub fn build_nodes_into<I: bidirected_adjacency_array::index::GraphIndexInteger>(
    slots: &mut [NodeSlot<I>],
    pairs: impl Iterator<Item = (I, u32)>,
) {
    debug_assert!(slots.len().is_power_of_two());
    let mask = slots.len() - 1;
    for (key, value) in pairs {
        debug_assert_ne!(key, I::max_value(), "the empty marker cannot be a key");
        let mut i = (mix(key.to_u64().unwrap()) as usize) & mask;
        loop {
            if slots[i].key == I::max_value() {
                slots[i] = NodeSlot::new(key, value);
                break;
            }
            debug_assert_ne!(slots[i].key, key, "duplicate key in node hash table");
            i = (i + 1) & mask;
        }
    }
}

/// Look `key` up in a [`NodeSlot`] table; `None` if absent. `slots` may be empty (always `None`).
/// Probes at most every slot once, so a table without an empty slot (a corrupt file) ends the
/// search.
#[inline]
pub fn lookup_node<I: bidirected_adjacency_array::index::GraphIndexInteger>(
    slots: &[NodeSlot<I>],
    key: I,
) -> Option<u32> {
    if slots.is_empty() {
        return None;
    }
    let mask = slots.len() - 1;
    let mut i = (mix(key.to_u64().unwrap()) as usize) & mask;
    for _ in 0..slots.len() {
        let slot = &slots[i];
        if slot.key == key {
            return Some(slot.value.get());
        }
        if slot.key == I::max_value() {
            return None;
        }
        i = (i + 1) & mask;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_tables_round_trip_and_miss() {
        let keys: Vec<u32> = (0..1000u32).map(|i| i * 7919 + 13).collect();
        let mut slots = vec![NodeSlot::empty(); node_capacity_for(keys.len())];
        build_nodes_into(
            &mut slots,
            keys.iter().enumerate().map(|(i, &k)| (k, i as u32)),
        );
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(lookup_node(&slots, k), Some(i as u32));
        }
        assert_eq!(lookup_node(&slots, 5), None);
        assert_eq!(lookup_node::<u32>(&[], 5), None);
    }

    #[test]
    fn verified_lookup_resolves_collisions() {
        let names = ["a", "b", "c"];
        // Force every name onto the same hash key.
        let mut slots = vec![HashSlot::EMPTY; 8];
        build_into_multi(&mut slots, (0..3).map(|i| (42, i as u32)));
        for (i, name) in names.iter().enumerate() {
            let found = lookup_verified(&slots, 42, |v| names[v as usize] == *name);
            assert_eq!(found, Some(i as u32));
        }
        assert_eq!(lookup_verified(&slots, 42, |_| false), None);
        assert_eq!(lookup_verified(&slots, 43, |_| true), None);
    }

    #[test]
    fn full_tables_end_a_miss() {
        let mut nodes = vec![NodeSlot::empty(); 4];
        build_nodes_into(&mut nodes, (0..4u32).map(|k| (k, k)));
        assert_eq!(lookup_node(&nodes, 9), None);
        let mut hashes = vec![HashSlot::EMPTY; 4];
        build_into_multi(&mut hashes, (0..4).map(|i| (7, i)));
        assert_eq!(lookup_verified(&hashes, 7, |_| false), None);
        assert_eq!(lookup_verified(&hashes, 8, |_| true), None);
    }
}
