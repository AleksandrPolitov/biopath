//! A strict reader of blunt GFA 1 graphs ([`read_gfa1`]).
//!
//! It builds the same [`BidirectedAdjacencyArray`] as the reader of the `bidirected-adjacency-array`
//! crate (nodes in `S`-line order, edges in `L`-line order, so graphs, fingerprints and index images
//! are identical), but refuses what that reader turns silently into a wrong graph, and says on which
//! line:
//!
//! - **Lines.** Each line is trimmed (CRLF, surrounding spaces). Empty lines and `#` comments are
//!   skipped. Fields are tab-separated; a line that seems space-separated is an error.
//! - **`H`.** `VN:Z:1`, `1.0`, `1.1` and `1.2` are GFA 1; `VN:Z:2…` is refused ("GFA 2 is not
//!   supported"); another version is read as GFA 1 with one warning. Header lines may appear
//!   anywhere.
//! - **`S name sequence [tags]`.** The length is the sequence's; a `*` sequence takes its length from
//!   the `LN:i:` tag (`*` without one is an error), an empty one from `LN` if present, else 0. A
//!   sequence must be printable ASCII (only its length is used), must not be a tag (`S x LN:i:8`
//!   lacks its `*`) and must agree with its `LN` (an unreadable `LN` beside a sequence is skipped
//!   with a warning). A segment name may be defined once.
//! - **`L from +|- to +|- [overlap] [tags]`.** Both names must be segments (an `L` line may precede
//!   their `S` lines). biopath needs a **blunt** graph: the overlap must be missing, `*`, or a CIGAR
//!   whose every operation has length 0 (`0M`); any other overlap is an error.
//! - **Other lines.** `P` and `W` (paths) are skipped silently. `C` (containment) lines are skipped
//!   with one warning: they add no link, so the blunt graph of the `L` lines is unchanged (vg ignores
//!   them too). `J` (GFA 1.2 jump) lines are refused: a jump is a gapped link, which a blunt graph
//!   cannot represent, and dropping it would answer `None` where a walk exists. The GFA 2 record
//!   types `E`, `F`, `G`, `O` and `U` are refused ("GFA 2 is not supported"). Lines of any other type
//!   are skipped with one warning per type.
//! - **Word size.** More nodes or edges than the index type can number is an error (see
//!   [`crate::graph_limits`]); whether the distances fit is checked separately, before a build.
//!
//! Lines are read as bytes into one reused buffer, so a megabyte-long `W` line costs a copy and no
//! allocation. Warnings go to the caller's callback (the CLI logs them).

use std::{collections::hash_map::Entry, fmt, io::BufRead};

use bidirected_adjacency_array::{
    graph::{BidirectedAdjacencyArray, BidirectedEdge},
    index::{GraphIndexInteger, NodeIndex},
    io::gfa1::{PlainGfaEdgeData, PlainGfaNodeData},
};

use crate::{
    fast_hash::FastMap,
    gfa_graph_extensions::LengthOnlyGfaNodeData,
    graph_limits::{self, Overflow, WordSizeError},
};

#[cfg(test)]
mod tests;

/// Node data that [`read_gfa1`] can build from an `S` line.
pub trait FromGfaSegment {
    /// The node of segment `name` of length `len`; `sequence` is its bases when the line spells
    /// them (`None` for `*` or an empty sequence).
    fn from_gfa_segment(name: &str, sequence: Option<&[u8]>, len: usize) -> Self;
}

/// Keeps the name and the length; no sequence byte is allocated.
impl FromGfaSegment for LengthOnlyGfaNodeData {
    fn from_gfa_segment(name: &str, _sequence: Option<&[u8]>, len: usize) -> Self {
        LengthOnlyGfaNodeData::with_len(name, len)
    }
}

/// Keeps the sequence; a segment given by its length only gets `len` bytes of `N`.
impl FromGfaSegment for PlainGfaNodeData {
    fn from_gfa_segment(name: &str, sequence: Option<&[u8]>, len: usize) -> Self {
        match sequence {
            // The reader accepts only printable ASCII sequences.
            Some(bases) => PlainGfaNodeData::new(name, String::from_utf8_lossy(bases)),
            None => PlainGfaNodeData::new(name, "N".repeat(len)),
        }
    }
}

/// Why [`read_gfa1`] refused a file: the 1-based line number and what is wrong there.
#[derive(Debug)]
pub struct GfaReadError {
    /// The line the error is about (for an I/O error, the line being read).
    pub line: usize,
    pub kind: GfaReadErrorKind,
}

/// What [`read_gfa1`] found wrong.
#[derive(Debug)]
pub enum GfaReadErrorKind {
    /// Reading the file failed.
    Io(std::io::Error),
    /// The line is not a valid line of a blunt GFA 1 graph.
    Invalid(String),
    /// The graph has more nodes or edges than the index type can number.
    WordSize(WordSizeError),
}

impl fmt::Display for GfaReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            GfaReadErrorKind::Io(error) => write!(f, "line {}: I/O error: {error}", self.line),
            GfaReadErrorKind::Invalid(message) => write!(f, "line {}: {message}", self.line),
            GfaReadErrorKind::WordSize(error) => write!(f, "line {}: {error}", self.line),
        }
    }
}

/// The message carries the cause, so there is no separate source (an error chain would print the
/// cause twice).
impl std::error::Error for GfaReadError {}

/// An `L` line end that named a segment not yet defined: resolved once the file is read.
struct PendingEnd {
    edge: usize,
    from: bool,
    name: Box<str>,
    line: usize,
}

/// Read a blunt GFA 1 graph (see the module docs for what is accepted). `warn` receives each
/// warning (an unknown version, skipped line types), at most once per kind.
pub fn read_gfa1<IndexType, NodeData, EdgeData>(
    mut reader: impl BufRead,
    mut warn: impl FnMut(&str),
) -> Result<BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>, GfaReadError>
where
    IndexType: GraphIndexInteger,
    NodeData: FromGfaSegment,
    EdgeData: From<PlainGfaEdgeData>,
{
    let max_nodes = graph_limits::max_nodes::<IndexType>();
    let max_edges = graph_limits::max_edges::<IndexType>();
    let mut node_of_name: FastMap<Box<str>, usize> = FastMap::default();
    let mut nodes: Vec<NodeData> = Vec::new();
    let mut edges: Vec<BidirectedEdge<IndexType, EdgeData>> = Vec::new();
    let mut pending: Vec<PendingEnd> = Vec::new();
    let mut warned: Vec<Vec<u8>> = Vec::new();
    let mut buffer = Vec::new();
    let mut line_number = 0;

    loop {
        buffer.clear();
        let read = reader
            .read_until(b'\n', &mut buffer)
            .map_err(|error| GfaReadError {
                line: line_number + 1,
                kind: GfaReadErrorKind::Io(error),
            })?;
        if read == 0 {
            break;
        }
        line_number += 1;
        let invalid = |message: String| GfaReadError {
            line: line_number,
            kind: GfaReadErrorKind::Invalid(message),
        };
        let line = buffer.trim_ascii();
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        let mut fields = line.split(|&byte| byte == b'\t');
        let record = fields.next().unwrap_or_default();
        match record {
            b"H" => {
                for field in fields {
                    if let Some(version) = field.strip_prefix(b"VN:Z:") {
                        match version {
                            b"1" | b"1.0" | b"1.1" | b"1.2" => {}
                            _ if version.starts_with(b"2") => {
                                return Err(invalid(gfa2_message(&format!(
                                    "the header declares version {}",
                                    text(version)
                                ))));
                            }
                            _ => {
                                if first_time(&mut warned, b"VN") {
                                    warn(&format!(
                                        "line {line_number}: unrecognised GFA version {:?}; reading the file as GFA 1",
                                        text(version)
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            b"S" => {
                if nodes.len() == max_nodes {
                    return Err(GfaReadError {
                        line: line_number,
                        kind: GfaReadErrorKind::WordSize(WordSizeError::new::<IndexType>(
                            Overflow::Nodes {
                                nodes: max_nodes + 1,
                            },
                        )),
                    });
                }
                let Segment {
                    name,
                    sequence,
                    len,
                    ignored_ln,
                } = parse_segment(fields).map_err(invalid)?;
                if let Some(tag) = ignored_ln
                    && first_time(&mut warned, b"LN")
                {
                    warn(&format!(
                        "line {line_number}: segment {name:?}: ignoring the invalid length tag {:?} (the sequence gives the length)",
                        text(tag)
                    ));
                }
                match node_of_name.entry(name.into()) {
                    Entry::Occupied(_) => {
                        return Err(invalid(format!("segment {name:?} is defined twice")));
                    }
                    Entry::Vacant(entry) => entry.insert(nodes.len()),
                };
                nodes.push(NodeData::from_gfa_segment(name, sequence, len));
            }
            b"L" => {
                if edges.len() == max_edges {
                    return Err(GfaReadError {
                        line: line_number,
                        kind: GfaReadErrorKind::WordSize(WordSizeError::new::<IndexType>(
                            Overflow::Edges {
                                edges: max_edges + 1,
                            },
                        )),
                    });
                }
                let link = parse_link(fields).map_err(invalid)?;
                let mut end = |name: &str, from: bool| match node_of_name.get(name) {
                    Some(&node) => NodeIndex::from_usize(node),
                    None => {
                        pending.push(PendingEnd {
                            edge: edges.len(),
                            from,
                            name: name.into(),
                            line: line_number,
                        });
                        NodeIndex::from_usize(0)
                    }
                };
                let from = end(link.from, true);
                let to = end(link.to, false);
                edges.push(BidirectedEdge {
                    from,
                    from_forward: link.from_forward,
                    to,
                    to_forward: link.to_forward,
                    data: EdgeData::from(PlainGfaEdgeData::new(0)),
                });
            }
            b"P" | b"W" => {}
            b"C" => {
                if first_time(&mut warned, record) {
                    warn(&format!(
                        "line {line_number}: skipping the containment (C) lines: they add no link to a blunt graph"
                    ));
                }
            }
            b"J" => {
                return Err(invalid(
                    "jump (J) lines are not supported: a jump is a gapped link, and biopath needs a \
                     blunt graph whose links are all L lines with overlap 0M"
                        .to_string(),
                ));
            }
            b"E" | b"F" | b"G" | b"O" | b"U" => {
                return Err(invalid(gfa2_message(&format!(
                    "{} is a GFA 2 line type",
                    text(record)
                ))));
            }
            _ if record.contains(&b' ') => {
                return Err(invalid(
                    "the line seems space-separated; GFA fields are separated by tabs".to_string(),
                ));
            }
            _ => {
                if first_time(&mut warned, record) {
                    warn(&format!(
                        "line {line_number}: skipping the GFA lines of unknown type {:?}",
                        text(record)
                    ));
                }
            }
        }
    }

    for end in pending {
        let node = *node_of_name.get(&end.name).ok_or_else(|| GfaReadError {
            line: end.line,
            kind: GfaReadErrorKind::Invalid(format!(
                "the link names segment {:?}, which no S line defines",
                end.name
            )),
        })?;
        let edge = &mut edges[end.edge];
        if end.from {
            edge.from = NodeIndex::from_usize(node);
        } else {
            edge.to = NodeIndex::from_usize(node);
        }
    }
    // The name index is no longer needed: free it before the graph is assembled.
    drop(node_of_name);
    Ok(BidirectedAdjacencyArray::new(nodes.into(), edges.into()))
}

/// The error text of a GFA 2 input.
fn gfa2_message(why: &str) -> String {
    format!("GFA 2 is not supported ({why}); biopath reads blunt GFA 1 graphs")
}

/// Whether `key` is new to `warned` (and remember it).
fn first_time(warned: &mut Vec<Vec<u8>>, key: &[u8]) -> bool {
    if warned.iter().any(|seen| seen == key) {
        return false;
    }
    warned.push(key.to_vec());
    true
}

/// A field as text for a message.
fn text(bytes: &[u8]) -> std::borrow::Cow<'_, str> {
    String::from_utf8_lossy(bytes)
}

/// An `S` line.
struct Segment<'a> {
    name: &'a str,
    /// The bases, if the line spells them.
    sequence: Option<&'a [u8]>,
    len: usize,
    /// An `LN` tag that could not be read but is not needed: the sequence spells the length.
    ignored_ln: Option<&'a [u8]>,
}

/// The `S` line of the given fields (after its record type).
fn parse_segment<'a>(mut fields: impl Iterator<Item = &'a [u8]>) -> Result<Segment<'a>, String> {
    let name = match fields.next() {
        Some(name) if !name.is_empty() => {
            std::str::from_utf8(name).map_err(|_| "the segment name is not UTF-8".to_string())?
        }
        _ => return Err("an S line needs a segment name".to_string()),
    };
    let sequence = fields.next().unwrap_or_default();
    let spelled = !sequence.is_empty() && sequence != b"*";
    if spelled && let Some(&bad) = sequence.iter().find(|byte| !byte.is_ascii_graphic()) {
        return Err(format!(
            "segment {name:?}: the sequence contains the byte {bad:#04x}; expected printable ASCII or *"
        ));
    }
    if spelled && is_tag(sequence) {
        return Err(format!(
            "segment {name:?}: the sequence field {:?} is a tag; write * for a segment without a sequence",
            text(sequence)
        ));
    }
    let (mut ln, mut bad_ln) = (None, None);
    for tag in fields {
        if let Some(value) = tag.strip_prefix(b"LN:i:") {
            match std::str::from_utf8(value)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
            {
                Some(value) if ln.is_some_and(|ln| ln != value) => {
                    return Err(format!("segment {name:?}: two different LN tags"));
                }
                Some(value) => ln = Some(value),
                None => bad_ln = Some(tag),
            }
        }
    }
    if let (false, Some(tag)) = (spelled, bad_ln) {
        return Err(format!(
            "segment {name:?}: invalid length tag {:?}",
            text(tag)
        ));
    }
    let len = match (spelled, ln) {
        (true, Some(ln)) if ln != sequence.len() => {
            return Err(format!(
                "segment {name:?}: the sequence has {} bases but its LN tag says {ln}",
                sequence.len()
            ));
        }
        (true, _) => sequence.len(),
        (false, Some(ln)) => ln,
        (false, None) if sequence == b"*" => {
            return Err(format!(
                "segment {name:?} has no sequence (*) and no LN:i: length tag"
            ));
        }
        (false, None) => 0,
    };
    Ok(Segment {
        name,
        sequence: spelled.then_some(sequence),
        len,
        ignored_ln: bad_ln,
    })
}

/// Whether a field has the form of an optional tag (`LN:i:8`, `SN:Z:chr1`).
fn is_tag(field: &[u8]) -> bool {
    matches!(
        field,
        [a, b, b':', t, b':', ..]
            if a.is_ascii_alphabetic() && b.is_ascii_alphanumeric() && b"AifZJHB".contains(t)
    )
}

/// The ends of an `L` line.
struct Link<'a> {
    from: &'a str,
    from_forward: bool,
    to: &'a str,
    to_forward: bool,
}

/// The ends of an `L` line (after its record type), which must be blunt.
fn parse_link<'a>(mut fields: impl Iterator<Item = &'a [u8]>) -> Result<Link<'a>, String> {
    let mut next = |what: &str| {
        fields.next().ok_or_else(|| {
            format!("an L line needs 5 fields (L, from, +/-, to, +/-); {what} is missing")
        })
    };
    let name = |field: &'a [u8]| {
        std::str::from_utf8(field).map_err(|_| "a segment name is not UTF-8".to_string())
    };
    let orientation = |field: &[u8]| match field {
        b"+" => Ok(true),
        b"-" => Ok(false),
        other => Err(format!(
            "the orientation {:?} is neither + nor -",
            text(other)
        )),
    };
    let from = name(next("the from segment")?)?;
    let from_forward = orientation(next("the from orientation")?)?;
    let to = name(next("the to segment")?)?;
    let to_forward = orientation(next("the to orientation")?)?;
    let overlap = fields.next().unwrap_or_default();
    if !is_blunt(overlap)? {
        return Err(format!(
            "the link {from}{} -> {to}{} overlaps by {}; biopath needs a blunt graph (every L line \
             with overlap 0M or *); bluntify it first, e.g. with GetBlunted \
             (https://github.com/vgteam/GetBlunted)",
            if from_forward { '+' } else { '-' },
            if to_forward { '+' } else { '-' },
            text(overlap)
        ));
    }
    Ok(Link {
        from,
        from_forward,
        to,
        to_forward,
    })
}

/// Whether an `L` line's overlap field is blunt: missing, `*`, or a CIGAR all of whose operations
/// have length 0. An invalid CIGAR is an error.
fn is_blunt(overlap: &[u8]) -> Result<bool, String> {
    if overlap.is_empty() || overlap == b"*" {
        return Ok(true);
    }
    let invalid = || {
        format!(
            "invalid overlap {:?} (expected a CIGAR such as 0M, or *)",
            text(overlap)
        )
    };
    let mut blunt = true;
    let mut digits = 0;
    for &byte in overlap {
        match byte {
            b'0' => digits += 1,
            b'1'..=b'9' => {
                blunt = false;
                digits += 1;
            }
            b'M' | b'I' | b'D' | b'N' | b'S' | b'H' | b'P' | b'X' | b'=' if digits > 0 => {
                digits = 0;
            }
            _ => return Err(invalid()),
        }
    }
    if digits > 0 {
        return Err(invalid());
    }
    Ok(blunt)
}
