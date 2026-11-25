//! Which oracle answers each SPQR node's searches: the tiers.
//!
//! A query searches two skeletons of a node: its inner skeleton, to project an endpoint onto the
//! node's reference poles, and its LCA-role skeleton (the inner one plus the node's `D_out`) when the
//! node is the LCA of the endpoints. A small skeleton is searched by Dijkstra; a big one gets a
//! precomputed oracle. The thresholds count directed states ([`Tiers`]):
//!
//! - **S-node segment tree** (`segtree_min`, default 200): an S-node whose inner skeleton is a
//!   clean cycle of at least this many states gets a segment tree, which answers both roles; it
//!   keeps neither skeletons nor labels.
//! - **Hub labels** (`label_min`, default 200): a node whose LCA-role skeleton has at least this
//!   many states gets hub labels for its LCA search. A parent's labels come from the outside DP,
//!   which fills its children's `D_out` by label lookups.
//! - **Pole-projection table** (`table_min`, default 1000): an inner skeleton of at least this many
//!   states, or of a labeled node, stores the four pole distances of every state. A labeled node
//!   with a table (or the root, which has no poles) then keeps no inner arcs.
//!
//! The thresholds are part of the build options
//! ([`BuildOptions::tiers`](crate::distance_index::BuildOptions::tiers)). Queries read a node's
//! oracles in that order ([`LcaOracle`], [`ProjectionOracle`]).

use bidirected_adjacency_array::index::{GraphIndexInteger, NodeIndex};
use spqr_tree::decomposition::SPQRNodeType;

use crate::{
    block_record::BlockView,
    min_plus::MinPlus,
    s_node_segtree::SegTreeView,
    skeleton::{LcaView, SkeletonView},
    skeleton_index::LabelsView,
};

/// The default label threshold: labels from 200 states on remove the query tail of Dijkstras over
/// 400-2000-state skeletons; lower thresholds gain nothing.
pub const LABEL_MIN_DIRECTED_NODES: usize = 200;

/// The default S-node segment-tree threshold (below it the skeleton Dijkstra is cheaper). Equals the
/// label threshold, so by default the tree replaces labels on every S-node it applies to.
pub const SNODE_SEGTREE_MIN_STATES: usize = 200;

/// The default pole-projection table threshold: a projection Dijkstra over fewer states costs about
/// what the table read saves, and the table costs `16` bytes per state.
pub const PROJ_TABLE_MIN_STATES: usize = 1000;

/// The states of a P-node (bond) skeleton: its two vertices in both orientations.
const BOND_STATES: usize = 4;

/// The tier thresholds of a build, in directed skeleton states (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tiers {
    /// An LCA-role skeleton from this size on gets hub labels (`usize::MAX`: none).
    pub label_min: usize,
    /// An S-node's inner skeleton from this size on gets a segment tree (`usize::MAX`: none).
    pub segtree_min: usize,
    /// An inner skeleton from this size on gets a pole-projection table (`0`: none).
    pub table_min: usize,
}

impl Default for Tiers {
    /// The production thresholds: [`LABEL_MIN_DIRECTED_NODES`], [`SNODE_SEGTREE_MIN_STATES`] and
    /// [`PROJ_TABLE_MIN_STATES`].
    fn default() -> Self {
        Self {
            label_min: LABEL_MIN_DIRECTED_NODES,
            segtree_min: SNODE_SEGTREE_MIN_STATES,
            table_min: PROJ_TABLE_MIN_STATES,
        }
    }
}

impl Tiers {
    /// Whether a node of `node_type` whose inner skeleton has `inner_states` states gets a segment
    /// tree (if its skeleton is a clean cycle).
    pub fn segtree(&self, node_type: SPQRNodeType, inner_states: usize) -> bool {
        node_type == SPQRNodeType::SNode && inner_states >= self.segtree_min
    }

    /// Whether an LCA-role skeleton of `lca_states` states gets hub labels.
    pub fn labels(&self, lca_states: usize) -> bool {
        lca_states >= self.label_min
    }

    /// Whether a bond's skeleton stays below the label threshold, so the outside DP fills its
    /// children by the bond closed form.
    pub fn bond_unlabeled(&self) -> bool {
        !self.labels(BOND_STATES)
    }

    /// Whether a node with reference poles and no segment tree, whose inner skeleton has
    /// `inner_states` states, gets a pole-projection table.
    pub fn table(&self, inner_states: usize, labeled: bool) -> bool {
        self.table_min != 0 && (inner_states >= self.table_min || labeled)
    }

    /// Whether a node's inner skeleton keeps its arcs: some search still runs over it unless its
    /// LCA search is labeled and its projection is a table (or it has no poles to project onto).
    pub fn keeps_inner_arcs(labeled: bool, table: bool, has_poles: bool) -> bool {
        !labeled || (!table && has_poles)
    }
}

/// What answers a node's LCA search, in the order a query tries them.
pub enum LcaOracle<'a, IndexType: GraphIndexInteger> {
    /// The S-node segment tree: a few range products.
    SegTree(SegTreeView<'a, IndexType>),
    /// Hub labels: a label merge per seed and target.
    Labels(LabelsView<'a, IndexType>),
    /// A multi-source Dijkstra over the LCA-role skeleton (inner skeleton plus `D_out` overlay).
    Skeleton(LcaView<'a, IndexType>),
}

impl<'a, IndexType: GraphIndexInteger> LcaOracle<'a, IndexType> {
    /// The oracle of node `local` of `block`.
    #[inline]
    pub fn of(block: &BlockView<'a, IndexType>, local: usize) -> Self {
        if let Some(seg) = block.segtree(local) {
            return Self::SegTree(seg);
        }
        if let Some(labels) = block.labels(local) {
            return Self::Labels(labels);
        }
        Self::Skeleton(block.lca(local))
    }
}

/// What answers a pole projection onto a node's reference poles, in the order a query tries them.
pub enum ProjectionOracle<'a, IndexType: GraphIndexInteger> {
    /// The block root: no poles, every projection is `∞`.
    Root,
    /// The S-node segment tree: two range products.
    SegTree(SegTreeView<'a, IndexType>),
    /// The pole-projection table: row `id` of the seed's dense id in the inner skeleton.
    Table {
        table: &'a [MinPlus<IndexType>],
        inner: SkeletonView<'a, IndexType>,
    },
    /// A Dijkstra over the inner skeleton from the seed, read at the poles.
    Skeleton {
        inner: SkeletonView<'a, IndexType>,
        poles: (NodeIndex<IndexType>, NodeIndex<IndexType>),
    },
}

impl<'a, IndexType: GraphIndexInteger> ProjectionOracle<'a, IndexType> {
    /// The oracle of node `local` of `block`.
    #[inline]
    pub fn of(block: &BlockView<'a, IndexType>, local: usize) -> Self {
        let Some(poles) = block.reference_poles(local) else {
            return Self::Root;
        };
        if let Some(seg) = block.segtree(local) {
            return Self::SegTree(seg);
        }
        let inner = block.inner(local);
        match block.projection(local) {
            Some(table) => Self::Table { table, inner },
            None => Self::Skeleton { inner, poles },
        }
    }
}
