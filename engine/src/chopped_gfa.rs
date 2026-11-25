//! Write a graph as GFA 1 that vg can load: vg's distance index cannot store a node longer than
//! 2047 bp (an 11-bit field), so every node is split into a path of pieces of at most `chop` bp,
//! each piece a run of `N` (only node lengths matter). Pieces are numbered from 1 in node order; a
//! link leaves or enters a node at its first piece (left side) or its last piece (right side), so
//! distances between node ends are unchanged. No paths are written. biopath's `export-shrunk-gfa`
//! writes G′ (see [`crate::shrink`]) this way.

use std::io::{self, Write};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::GfaNodeData,
};

/// The ids of the pieces a graph's nodes are chopped into: node `v` is `first[v] ..= first[v + 1] - 1`.
pub(crate) struct ChoppedIds {
    first: Vec<u64>,
}

impl ChoppedIds {
    /// Where a walk entering directed node (`v`, `forward`) at its start enters the chopped graph.
    pub(crate) fn start(&self, v: usize, forward: bool) -> (u64, bool) {
        if forward {
            (self.first[v], true)
        } else {
            (self.first[v + 1] - 1, false)
        }
    }

    /// Where a walk leaving directed node (`v`, `forward`) at its end leaves the chopped graph.
    pub(crate) fn end(&self, v: usize, forward: bool) -> (u64, bool) {
        if forward {
            (self.first[v + 1] - 1, true)
        } else {
            (self.first[v], false)
        }
    }
}

/// What [`write_chopped_gfa`] wrote.
pub struct Export {
    /// The piece ids of every node (the tests map positions onto the export).
    #[cfg(test)]
    pub(crate) ids: ChoppedIds,
    /// The segments (pieces).
    pub segments: u64,
    /// The links: between consecutive pieces of a node, and the graph's own.
    pub links: u64,
    /// The longest node, in bp.
    pub max_node_len: u64,
    /// The summed node lengths, in bp.
    pub total_bp: u64,
}

/// Write `graph` as GFA 1 with every node split into pieces of at most `chop` bp (0 = whole nodes),
/// each piece a run of `N`. Fails on a zero-length node (vg cannot load one).
pub fn write_chopped_gfa<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
    chop: u64,
    out: &mut (impl Write + ?Sized),
) -> io::Result<Export> {
    let n = graph.node_count();
    let lens: Vec<u64> = (0..n)
        .map(|v| graph.node_data(NodeIndex::from_usize(v)).sequence().len() as u64)
        .collect();
    if let Some(v) = lens.iter().position(|&len| len == 0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "node {} has length 0; vg cannot load zero-length segments",
                graph.node_data(NodeIndex::from_usize(v)).name()
            ),
        ));
    }
    let piece = |len: u64| if chop == 0 { len } else { chop.min(len) };
    let mut first = Vec::with_capacity(n + 1);
    let mut next = 1u64;
    for &len in &lens {
        first.push(next);
        next += len.div_ceil(piece(len));
    }
    first.push(next);
    let ids = ChoppedIds { first };

    let max_node_len = lens.iter().copied().max().unwrap_or(0);
    let bases = vec![b'N'; lens.iter().map(|&len| piece(len)).max().unwrap_or(0) as usize];
    writeln!(out, "H\tVN:Z:1.0")?;
    let mut links = 0u64;
    for (v, &len) in lens.iter().enumerate() {
        let (step, mut id, mut left) = (piece(len), ids.first[v], len);
        while left > 0 {
            let here = step.min(left);
            write!(out, "S\t{id}\t")?;
            out.write_all(&bases[..here as usize])?;
            out.write_all(b"\n")?;
            id += 1;
            left -= here;
        }
    }
    for v in 0..n {
        for id in ids.first[v]..ids.first[v + 1] - 1 {
            writeln!(out, "L\t{id}\t+\t{}\t+\t0M", id + 1)?;
            links += 1;
        }
    }
    let sign = |forward: bool| if forward { '+' } else { '-' };
    for e in graph.iter_edges() {
        let view = graph.edge(e);
        let (from, to) = (view.from(), view.to());
        let (a, fa) = ids.end(from.into_bidirected().into_usize(), from.is_forward());
        let (b, fb) = ids.start(to.into_bidirected().into_usize(), to.is_forward());
        writeln!(out, "L\t{a}\t{}\t{b}\t{}\t0M", sign(fa), sign(fb))?;
        links += 1;
    }
    out.flush()?;
    Ok(Export {
        segments: next - 1,
        links,
        max_node_len,
        total_bp: lens.iter().sum::<u64>(),
        #[cfg(test)]
        ids,
    })
}
