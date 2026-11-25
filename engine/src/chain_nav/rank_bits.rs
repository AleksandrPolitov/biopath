//! A set over `0..n` with O(1) rank.

/// 64 members of a [`RankBits`] set and the rank of the first (one cache-line-friendly 16 bytes,
/// so a rank costs one load).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RankBlock {
    pub word: u64,
    pub rank: u32,
    pub _pad: u32,
}
crate::image::image_record!(RankBlock = [u64, u32, u32]);

/// A set over `0..n` with O(1) rank: per 64 elements a [`RankBlock`] holding their bits and the rank
/// of the first plus a base (so the rank is directly an index into a list that starts with `base`
/// other entries).
#[derive(Clone, Copy, Debug, Default)]
pub struct RankBits<'a> {
    pub blocks: &'a [RankBlock],
}

impl RankBits<'_> {
    /// `base +` the number of set members below `i` (the index of `i` when `i` is a member).
    #[inline]
    pub fn rank(&self, i: usize) -> usize {
        let block = &self.blocks[i / 64];
        block.rank as usize + (block.word & ((1u64 << (i % 64)) - 1)).count_ones() as usize
    }

    /// Whether `i` is a member.
    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        self.blocks[i / 64].word >> (i % 64) & 1 == 1
    }
}

/// The blocks of a [`RankBits`] over `0..n` with the given members (ascending) and rank base.
pub fn rank_bits(n: usize, members: impl IntoIterator<Item = usize>, base: u32) -> Vec<RankBlock> {
    let mut blocks = vec![RankBlock::default(); n.div_ceil(64)];
    for i in members {
        blocks[i / 64].word |= 1 << (i % 64);
    }
    let mut count = base;
    for block in &mut blocks {
        block.rank = count;
        count += block.word.count_ones();
    }
    blocks
}
