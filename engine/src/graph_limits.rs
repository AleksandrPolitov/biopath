//! Whether a graph fits the index's integer type (the CLI's `--word-size`).
//!
//! Every node index, edge index, offset and distance of the index is one word of `IndexType`
//! (`u32` or `u64`), and `IndexType::MAX` is reserved: it is the "none" of every
//! optional index and the `∞` of every distance. The typed arithmetic does not check for overflow
//! in release builds (a query adds a handful of distances on its hot path), so a graph whose
//! numbers may not fit has to be refused before it is built on. The checks here run in `O(n + m)`
//! and allocate one bit per node:
//!
//! - **Counts** ([`check_counts`]): a graph of `n` nodes and `m` edges has directed-node slots
//!   `0..=2n` and `2m` directed edges, so `2n < MAX` and `2m < MAX`. The other typed indices of
//!   the engine stay below these: SPQR nodes and blocks are at most `m`, cut nodes and components
//!   at most `n`.
//! - **Positions** ([`check_positions`]): every node length is below `MAX`, so every offset
//!   `0..=len` of a query position is a word.
//! - **Distances** ([`check_build`]): within one connected component, a shortest walk passes every
//!   directed node at most once, so a distance between two positions is at most
//!   `D = 2·Σ len + max len` (the sum and the maximum over the component's nodes). The engine adds up
//!   to four such terms before it takes a minimum, so `4·D < MAX` is required of every component.
//!   Components are independent (two positions in different components are answered `None` before
//!   any arithmetic), so the bound is per component, never over the whole graph.
//!
//! [`check_build`] is the precondition of [`DistanceIndex::build`] for the graph it is given (G′ for
//! a shrunk index). The original graph of a shrunk index needs only [`check_positions`]: the alias
//! that maps its positions onto G′ computes in `u64`.
//!
//! [`DistanceIndex::build`]: crate::distance_index::DistanceIndex::build

use std::fmt;

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::GfaNodeData,
};
use bitvec::vec::BitVec;

/// A graph too large for the integer type: what does not fit, and which word sizes it would fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordSizeError {
    /// The width of the integer type in bits.
    pub bits: u32,
    /// What does not fit.
    pub overflow: Overflow,
}

/// The number that does not fit a [`WordSizeError`]'s integer type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Overflow {
    /// Too many nodes (`2n` directed-node slots must stay below `MAX`).
    Nodes { nodes: usize },
    /// Too many edges (`2m` directed edges must stay below `MAX`).
    Edges { edges: usize },
    /// A node too long for its offsets to be words.
    NodeLength { node: usize, len: u64 },
    /// A connected component whose distances may not fit (see the module docs).
    Distances {
        /// The total length of the component's nodes.
        component_bp: u128,
        /// The length of its longest node.
        max_node_bp: u64,
    },
}

impl WordSizeError {
    /// `overflow` for words of type `IndexType`.
    pub fn new<IndexType: GraphIndexInteger>(overflow: Overflow) -> Self {
        WordSizeError {
            bits: bits::<IndexType>(),
            overflow,
        }
    }
}

impl Overflow {
    /// The value that must stay below `MAX`.
    fn required(&self) -> u128 {
        match *self {
            Overflow::Nodes { nodes } => 2 * nodes as u128,
            Overflow::Edges { edges } => 2 * edges as u128,
            Overflow::NodeLength { len, .. } => u128::from(len),
            Overflow::Distances {
                component_bp,
                max_node_bp,
            } => 4 * distance_bound(component_bp, max_node_bp),
        }
    }
}

/// `D = 2·Σ + max`: no distance between two positions of a component exceeds it.
fn distance_bound(component_bp: u128, max_node_bp: u64) -> u128 {
    2 * component_bp + u128::from(max_node_bp)
}

/// The reserved maximum of a `bits`-bit word.
fn word_max(bits: u32) -> u128 {
    (1u128 << bits) - 1
}

impl fmt::Display for WordSizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = word_max(self.bits);
        write!(f, "graph too large for --word-size {}: ", self.bits)?;
        match self.overflow {
            Overflow::Nodes { nodes } => write!(
                f,
                "{nodes} nodes do not fit (at most {} do: twice the node count must stay below {max})",
                (max - 1) / 2
            )?,
            Overflow::Edges { edges } => write!(
                f,
                "{edges} edges do not fit (at most {} do: twice the edge count must stay below {max})",
                (max - 1) / 2
            )?,
            Overflow::NodeLength { node, len } => write!(
                f,
                "node {node} is {len} bp long, offsets must stay below {max}"
            )?,
            Overflow::Distances {
                component_bp,
                max_node_bp,
            } => write!(
                f,
                "a connected component has {component_bp} bp of nodes (longest {max_node_bp} bp), \
                 so distances may reach {} and sums of four of them {}, not below {max}",
                distance_bound(component_bp, max_node_bp),
                self.overflow.required()
            )?,
        }
        let required = self.overflow.required();
        let fitting: Vec<String> = [32, 64]
            .into_iter()
            .filter(|&bits| bits > self.bits && required < word_max(bits))
            .map(|bits| bits.to_string())
            .collect();
        match fitting.split_last() {
            None => write!(f, "; no supported word size fits"),
            Some((last, [])) => write!(f, "; use --word-size {last}"),
            Some((last, rest)) => write!(f, "; use --word-size {} or {last}", rest.join(", ")),
        }
    }
}

impl std::error::Error for WordSizeError {}

/// The width of `IndexType` in bits.
fn bits<IndexType: GraphIndexInteger>() -> u32 {
    IndexType::max_value().count_ones()
}

/// The reserved maximum of `IndexType`.
fn max<IndexType: GraphIndexInteger>() -> u128 {
    word_max(bits::<IndexType>())
}

fn fail<IndexType: GraphIndexInteger>(overflow: Overflow) -> Result<(), WordSizeError> {
    Err(WordSizeError::new::<IndexType>(overflow))
}

/// The most nodes a graph with `IndexType` indices may have (`2n < MAX`).
pub fn max_nodes<IndexType: GraphIndexInteger>() -> usize {
    usize::try_from((max::<IndexType>() - 1) / 2).unwrap_or(usize::MAX)
}

/// The most edges a graph with `IndexType` indices may have (`2m < MAX`).
pub fn max_edges<IndexType: GraphIndexInteger>() -> usize {
    max_nodes::<IndexType>()
}

/// Whether `nodes` nodes and `edges` edges fit `IndexType` (see the module docs). Check it before
/// a graph is assembled: assembling one that does not fit panics.
pub fn check_counts<IndexType: GraphIndexInteger>(
    nodes: usize,
    edges: usize,
) -> Result<(), WordSizeError> {
    if nodes > max_nodes::<IndexType>() {
        return fail::<IndexType>(Overflow::Nodes { nodes });
    }
    if edges > max_edges::<IndexType>() {
        return fail::<IndexType>(Overflow::Edges { edges });
    }
    Ok(())
}

/// Whether every position of `graph` is a word: its counts fit and every node is shorter than
/// `IndexType::MAX`, so every offset `0..=len` fits. What the original graph of a shrunk index
/// needs (its alias computes in `u64`).
pub fn check_positions<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
) -> Result<(), WordSizeError> {
    check_counts::<IndexType>(graph.node_count(), graph.edge_count())?;
    let limit = max::<IndexType>();
    for node in graph.iter_nodes() {
        let len = node_len(graph, node);
        if u128::from(len) >= limit {
            return fail::<IndexType>(Overflow::NodeLength {
                node: node.into_usize(),
                len,
            });
        }
    }
    Ok(())
}

/// Whether [`DistanceIndex::build`] can index `graph` with `IndexType` words: its counts fit and,
/// in every connected component, four times the distance bound `D = 2·Σ len + max len` stays below
/// `IndexType::MAX` (see the module docs). Also enough for whole-graph Dijkstra on `graph`.
///
/// [`DistanceIndex::build`]: crate::distance_index::DistanceIndex::build
pub fn check_build<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
) -> Result<(), WordSizeError> {
    check_counts::<IndexType>(graph.node_count(), graph.edge_count())?;
    let limit = max::<IndexType>();
    let mut seen: BitVec = BitVec::repeat(false, graph.node_count());
    let mut stack = Vec::new();
    for root in graph.iter_nodes() {
        if seen[root.into_usize()] {
            continue;
        }
        seen.set(root.into_usize(), true);
        stack.push(root);
        let (mut component_bp, mut max_node_bp) = (0u128, 0u64);
        while let Some(node) = stack.pop() {
            let len = node_len(graph, node);
            component_bp += u128::from(len);
            max_node_bp = max_node_bp.max(len);
            for forward in [true, false] {
                for edge in graph.iter_outgoing_edges(node.into_directed(forward)) {
                    let next = edge.to().into_bidirected();
                    if !seen[next.into_usize()] {
                        seen.set(next.into_usize(), true);
                        stack.push(next);
                    }
                }
            }
        }
        let overflow = Overflow::Distances {
            component_bp,
            max_node_bp,
        };
        if overflow.required() >= limit {
            return fail::<IndexType>(overflow);
        }
    }
    Ok(())
}

/// A node's length as a plain number (the typed length panics where it does not fit).
fn node_len<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    node: NodeIndex<IndexType>,
) -> u64 {
    graph.node_data(node).sequence().len() as u64
}

#[cfg(test)]
mod tests {
    use bidirected_adjacency_array::{graph::BidirectedEdge, io::gfa1::PlainGfaEdgeData};

    use super::*;
    use crate::gfa_graph_extensions::LengthOnlyGfaNodeData;

    type Graph<I> = BidirectedAdjacencyArray<I, LengthOnlyGfaNodeData, PlainGfaEdgeData>;

    /// A graph of nodes with the given lengths and the given `(from, to)` forward links.
    fn graph<I: GraphIndexInteger>(lens: &[usize], links: &[(usize, usize)]) -> Graph<I> {
        let nodes: Vec<_> = lens
            .iter()
            .enumerate()
            .map(|(i, &len)| LengthOnlyGfaNodeData::with_len(&i.to_string(), len))
            .collect();
        let edges: Vec<_> = links
            .iter()
            .map(|&(from, to)| BidirectedEdge {
                from: NodeIndex::from_usize(from),
                from_forward: true,
                to: NodeIndex::from_usize(to),
                to_forward: true,
                data: PlainGfaEdgeData::new(0),
            })
            .collect();
        BidirectedAdjacencyArray::new(nodes.into(), edges.into())
    }

    #[test]
    fn a_chain_of_three_400mbp_nodes_needs_64_bits() {
        let chain = [400_000_000; 3];
        let error = check_build(&graph::<u32>(&chain, &[(0, 1), (1, 2)])).unwrap_err();
        assert_eq!(
            error.overflow,
            Overflow::Distances {
                component_bp: 1_200_000_000,
                max_node_bp: 400_000_000
            }
        );
        let message = error.to_string();
        assert!(
            message.starts_with(
                "graph too large for --word-size 32: a connected component has 1200000000 bp"
            ),
            "{message}"
        );
        assert!(message.ends_with("use --word-size 64"), "{message}");
        assert!(check_build(&graph::<u64>(&chain, &[(0, 1), (1, 2)])).is_ok());
        // Each node alone is a component of its own: 4 * (2 * 4e8 + 4e8) is still too much for u32.
        assert!(check_build(&graph::<u32>(&chain, &[])).is_err());
        assert!(check_positions(&graph::<u32>(&chain, &[(0, 1), (1, 2)])).is_ok());
    }

    #[test]
    fn the_bound_is_per_component() {
        // Two components of 250 Mbp each (D = 2 * 250M + 150M = 650M, 4D = 2.6G < u32::MAX);
        // joined, D = 1150M and 4D = 4.6G would not fit.
        let lens = [100_000_000, 150_000_000, 100_000_000, 150_000_000];
        assert!(check_build(&graph::<u32>(&lens, &[(0, 1), (2, 3)])).is_ok());
        assert!(check_build(&graph::<u32>(&lens, &[(0, 1), (1, 2), (2, 3)])).is_err());
    }

    #[test]
    fn counts_and_lengths_must_fit() {
        assert_eq!(max_nodes::<u32>(), (u32::MAX as usize - 1) / 2);
        assert!(check_counts::<u32>(max_nodes::<u32>(), max_edges::<u32>()).is_ok());
        let error = check_counts::<u32>(3_000_000_000, 0).unwrap_err();
        assert_eq!(
            error.overflow,
            Overflow::Nodes {
                nodes: 3_000_000_000
            }
        );
        assert!(error.to_string().ends_with("use --word-size 64"));
        assert!(check_counts::<u32>(0, max_edges::<u32>() + 1).is_err());
        assert!(check_counts::<u64>(usize::MAX / 4, usize::MAX / 4).is_ok());
        let error = WordSizeError::new::<u64>(Overflow::Nodes { nodes: usize::MAX });
        assert!(error.to_string().ends_with("no supported word size fits"));

        // The length check is generic: a 16-bit word shows it without a 4 GiB node.
        let error = check_positions(&graph::<u16>(&[70_000, 2], &[(0, 1)])).unwrap_err();
        assert_eq!(
            error.overflow,
            Overflow::NodeLength {
                node: 0,
                len: 70_000
            }
        );
        assert!(check_positions(&graph::<u16>(&[65_534, 2], &[(0, 1)])).is_ok());
        assert!(check_positions(&graph::<u16>(&[65_535], &[])).is_err());
    }
}
