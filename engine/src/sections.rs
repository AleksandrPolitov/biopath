//! The index image's schema: stable section ids and header knob slots.
//!
//! **Never renumber or reuse an id.** A new structure gets a new id; a retired one goes into
//! [`RETIRED`]. Loaders look sections up by id and ignore unknown ones, which is what lets a derived
//! structure (e.g. the lift's range-query backend) be swapped without rewriting the rest of a file.

use crate::image::SectionId;

// ---- graph (adjacency, names; lengths live in the node records) ------------------------------
pub const GRAPH_ADJ_OFFSETS: SectionId = SectionId(2);
pub const GRAPH_ADJ_TARGETS: SectionId = SectionId(3);
/// The nodes that keep their arcs (O(1)-rank set over node ids; the adjacency CSR covers only them,
/// in rank order): a simple small-block node in a block whose rows are all closed forms keeps none.
pub const GRAPH_ADJ_RANK: SectionId = SectionId(9);
pub const GRAPH_NAME_BYTES: SectionId = SectionId(4);
pub const GRAPH_NAME_OFFSETS: SectionId = SectionId(5);
pub const GRAPH_NAME_HASH: SectionId = SectionId(6);
pub const GRAPH_NAME_RUNS: SectionId = SectionId(7);
pub const GRAPH_NAME_RUN_ORDER: SectionId = SectionId(8);

// ---- node topology ------------------------------------------------------------------------------
/// One 8-byte record per graph node (length, kind, closed-form mask, cut id / block / row header) +
/// sentinel.
pub const TOPO_NODE_RECS: SectionId = SectionId(154);
pub const TOPO_META: SectionId = SectionId(105);
/// Chain layout (`crate::chain_nav`): one `CutRec` per cut node (dense id = chain order), one
/// word per block, a `HeadBlockRec` per block that is no cut's heavy child, one `ChainRec` per
/// heavy-path chain. Under the chain block numbering a cut's blocks and a heavy block's position are
/// derived.
pub const TOPO_CUT_RECS: SectionId = SectionId(140);
pub const TOPO_BLOCK_WORDS: SectionId = SectionId(138);
pub const TOPO_HEAD_BLOCKS: SectionId = SectionId(139);
pub const TOPO_CHAIN_RECS: SectionId = SectionId(128);
/// The child lists of the cut nodes in three or more blocks: count, heavy child, light children;
/// and where each cut's list starts (`CutExtraDir`, sorted by cut).
pub const TOPO_CUT_EXTRA_BLOCKS: SectionId = SectionId(130);
pub const TOPO_CUT_EXTRA_DIR: SectionId = SectionId(141);
/// O(1)-rank sets over cut ids (`crate::chain_nav::RankBits`, one `RankBlock` per 64 ids): whose
/// heavy child is a special block (the rank indexes the special records), and which have extra
/// blocks (indexes the directory).
pub const TOPO_HEAVY_SPECIAL_WORDS: SectionId = SectionId(142);
pub const TOPO_CUT_EXTRA_WORDS: SectionId = SectionId(144);
/// Node lengths at or above `topology::LEN_ESCAPE` (O(1)-rank set over node ids, and the values).
pub const TOPO_LONG_LEN_WORDS: SectionId = SectionId(146);
pub const TOPO_LONG_LENS: SectionId = SectionId(148);
/// The cut nodes with SPQR locations (O(1)-rank set over cut ids) and where each one's list starts.
pub const TOPO_CUT_LOC_WORDS: SectionId = SectionId(151);
pub const TOPO_CUT_LOC_STARTS: SectionId = SectionId(153);
/// Blocks whose cut count is not 2, or root blocks (`SpecialBlockRec`, sorted by block), and their
/// cut lists (dense cut ids in the decomposition's order).
pub const TOPO_SPECIAL_BLOCK_RECS: SectionId = SectionId(131);
pub const TOPO_SPECIAL_BLOCK_CUTS: SectionId = SectionId(132);
/// SPQR locations that do not fit inline in their node record: every cut node's, one per record
/// block containing it (see `crate::topology::SpqrLoc`).
pub const TOPO_SPQR_LOCS: SectionId = SectionId(137);

// ---- per-block records ------------------------------------------------------------------------
/// One `BlockDirRec` per block THAT HAS A RECORD, sorted by block (the small-block fast path
/// leaves most blocks without one): the record's offset. The records themselves are unregistered
/// ranges of the image.
pub const BLOCK_DIR_SPARSE: SectionId = SectionId(22);

// ---- block-cut lift -----------------------------------------------------------------------------
pub const LIFT_META: SectionId = SectionId(30);
/// Chain segment-tree backend (when the prefix lift is not chosen).
pub const LIFT_SEG_CHAIN_OF: SectionId = SectionId(55);
pub const LIFT_SEG_INDEX_IN_CHAIN: SectionId = SectionId(56);
pub const LIFT_SEG_CHAIN_TOP: SectionId = SectionId(57);
pub const LIFT_SEG_CUT_PARENT: SectionId = SectionId(58);
pub const LIFT_SEG_TREE_OFFSETS: SectionId = SectionId(59);
pub const LIFT_SEG_UP: SectionId = SectionId(60);
pub const LIFT_SEG_DOWN: SectionId = SectionId(61);
/// Per-node cut rows (see `crate::node_rows`), in node order; starts in `TOPO_NODE_RECS`.
pub const LIFT_NODE_ROWS: SectionId = SectionId(110);
/// Directory of the blocks with more than `AUG_SORTED_MIN_CUTS` cuts (`AugSortedDir`, sorted by
/// block): where their sorted `(cut, pos)` lists sit in `LIFT_AUG_SORTED_SPARSE`.
pub const LIFT_AUG_SORTED_DIR: SectionId = SectionId(117);
pub const LIFT_AUG_SORTED_SPARSE: SectionId = SectionId(118);
/// Closed-form ("prefix") lift backend (`crate::prefix_lift`; `LiftMeta` flag bit 0): the
/// irregular steps' matrices (the per-cut data sit in `TOPO_CUT_RECS`).
pub const LIFT_PFX_IRR_MATS: SectionId = SectionId(124);
/// Chain layout, lift side: the exception cuts grouped per chain (`ChainRec::exc_start`), the
/// dense `M_aug` of the blocks that keep one (`AugDenseDir`, sorted by block, + data), and the
/// flip-capable cuts of the special blocks (`SpecialBlockRec::flip_start`).
pub const LIFT_CHAIN_EXC: SectionId = SectionId(133);
pub const LIFT_AUG_DENSE_DIR: SectionId = SectionId(134);
pub const LIFT_AUG_DENSE_DATA: SectionId = SectionId(135);
pub const LIFT_SPECIAL_FLIPS: SectionId = SectionId(136);

// ---- shrunk index (`shrink`): the original graph's names and the original -> shrunk alias -----
/// `(node, length)` of the original nodes whose length is not in `SHRINK_LEN_DICT`, sorted by node.
pub const SHRINK_LONG_LENS: SectionId = SectionId(162);
pub const SHRINK_NAME_RUNS: SectionId = SectionId(164);
pub const SHRINK_NAME_RUN_ORDER: SectionId = SectionId(165);
pub const SHRINK_NAME_BYTES: SectionId = SectionId(166);
pub const SHRINK_NAME_OFFSETS: SectionId = SectionId(167);
pub const SHRINK_NAME_HASH: SectionId = SectionId(168);
/// The alias (see `shrink`): one `PrunedChain` per multi-node pruned chain, and the far ports of
/// single-node chains whose deltas did not fit.
pub const SHRINK_PCHAINS: SectionId = SectionId(173);
pub const SHRINK_FAR_PORTS: SectionId = SectionId(174);
/// 8-byte `RunRec` per run and 4-byte `P1Rec` per single-node pruned chain.
pub const SHRINK_RUNS: SectionId = SectionId(175);
pub const SHRINK_P1: SectionId = SectionId(176);
/// The dictionary of the 254 most frequent original lengths (code 255 = `SHRINK_LONG_LENS`).
pub const SHRINK_LEN_DICT: SectionId = SectionId(178);
/// One 32-byte `AliasBlock` per 64 original ids (run-head and single-node bits with their ranks and
/// the previous head) and one 24-byte `LenBlock` per 16 ids (prefix sum + length codes).
pub const SHRINK_ALIAS_BLOCKS: SectionId = SectionId(180);
pub const SHRINK_LEN_BLOCKS: SectionId = SectionId(181);

/// Ids of retired sections (and of a few dropped before any release): never written, never reused.
pub const RETIRED: &[u32] = &[
    1, 10, 11, 12, 13, 14, 15, 20, 21, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 62, 63, 64, 70, 71, 72, 73, 74, 80, 81, 82, 83, 90, 91,
    100, 101, 102, 103, 104, 111, 112, 113, 114, 115, 116, 119, 120, 121, 122, 123, 125, 126, 127,
    129, 149, 150, 155, 160, 161, 163, 169, 170, 171, 172, 177, 179, 182, 183, 184, 185, 186, 187,
    190, 191, 192, 193,
];

/// Header `build_knobs` slots. Slots 1, 2, 5 and 8 (knobs of retired tiers) and 11-15 are unused
/// (`0`).
pub mod knob {
    pub const LABEL_THRESHOLD: usize = 0;
    pub const SMALL_BLOCK_MAX: usize = 3;
    pub const FLAGS: usize = 4;
    /// Largest skeleton (directed nodes) a query can Dijkstra; sizes the intra scratch at load
    /// without scanning the records.
    pub const MAX_SEARCHED_NODES: usize = 6;
    /// Minimum directed states for an S-node to carry a segment-tree oracle (`u64::MAX` = off).
    pub const SNODE_SEGTREE_MIN: usize = 7;
    /// Largest cut count of a block whose nodes carry cut rows (`0` = no rows).
    pub const NODE_ROW_MAX_CUTS: usize = 9;
    /// Minimum directed states of an inner skeleton to carry a pole-projection table (`0` = none).
    pub const PROJ_TABLE_MIN: usize = 10;

    // `FLAGS` bits; the others are unused (`0`).
    /// The small-block fast path was on
    /// ([`SmallBlocks::enabled`](crate::distance_index::SmallBlocks::enabled)).
    pub const FLAG_SMALL_BLOCK_FAST: u64 = 1 << 1;
    /// Built on a shrunk graph
    /// ([`BuildOptions::shrink`](crate::distance_index::BuildOptions::shrink)): queries name
    /// ORIGINAL nodes and go through the alias.
    pub const FLAG_SHRUNK: u64 = 1 << 7;
}

/// The name of a live section id (`None` for a retired or unknown one).
fn live_name(id: u32) -> Option<&'static str> {
    Some(match id {
        2 => "graph.adj_offsets",
        3 => "graph.adj_targets",
        4 => "graph.name_bytes",
        5 => "graph.name_offsets",
        6 => "graph.name_hash",
        7 => "graph.name_runs",
        8 => "graph.name_run_order",
        9 => "graph.adj_rank",
        22 => "blocks.dir_sparse",
        30 => "lift.meta",
        55 => "lift.seg_chain_of",
        56 => "lift.seg_index_in_chain",
        57 => "lift.seg_chain_top",
        58 => "lift.seg_cut_parent",
        59 => "lift.seg_tree_offsets",
        60 => "lift.seg_up",
        61 => "lift.seg_down",
        105 => "topology.meta",
        110 => "lift.node_rows",
        117 => "lift.aug_sorted_dir",
        118 => "lift.aug_sorted",
        124 => "lift.pfx_irr_mats",
        128 => "topology.chain_recs",
        130 => "topology.cut_extra_blocks",
        131 => "topology.special_block_recs",
        132 => "topology.special_block_cuts",
        133 => "lift.chain_exc",
        134 => "lift.aug_dense_dir",
        135 => "lift.aug_dense_data",
        136 => "lift.special_flips",
        137 => "topology.spqr_locs",
        138 => "topology.block_words",
        139 => "topology.head_blocks",
        140 => "topology.cut_recs",
        141 => "topology.cut_extra_dir",
        142 => "topology.heavy_special_rank",
        144 => "topology.cut_extra_rank",
        146 => "topology.long_len_rank",
        148 => "topology.long_lens",
        151 => "topology.cut_loc_rank",
        153 => "topology.cut_loc_starts",
        154 => "topology.node_recs",
        162 => "shrink.long_lens",
        164 => "shrink.name_runs",
        165 => "shrink.name_run_order",
        166 => "shrink.name_bytes",
        167 => "shrink.name_offsets",
        168 => "shrink.name_hash",
        173 => "shrink.pchains",
        174 => "shrink.far_ports",
        175 => "shrink.runs",
        176 => "shrink.p1",
        178 => "shrink.len_dict",
        180 => "shrink.alias_blocks",
        181 => "shrink.len_blocks",
        _ => return None,
    })
}

/// Whether `id` is a live section id (one the current build may write).
#[cfg(test)]
pub(crate) fn is_live(id: u32) -> bool {
    live_name(id).is_some()
}

/// Human-readable name of a section id (for `index-inspect`).
pub fn section_name(id: SectionId) -> &'static str {
    live_name(id.0).unwrap_or(if RETIRED.contains(&id.0) {
        "retired"
    } else {
        "unknown"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retired_ids_are_sorted_and_not_live() {
        assert!(RETIRED.windows(2).all(|w| w[0] < w[1]));
        for &id in RETIRED {
            assert!(!is_live(id), "retired section id {id} is live");
        }
    }
}
