//! Query files (`--query-in`) and result files (`--query-out`): positions as `node orientation
//! offset`, tab-separated.

use std::{
    borrow::Cow,
    io::{BufRead, Write},
    path::Path,
};

use anyhow::{Context, anyhow, bail};
use bidirected_adjacency_array::index::{DirectedNodeIndex, GraphIndexInteger, NodeIndex};
use itertools::Itertools;
use spqr_shortest_path_index::{
    location::{GfaLocation, GfaNodeOffset},
    path::OptionalGfaPathLength,
};

pub struct Query<IndexType> {
    source: GfaLocation<IndexType>,
    targets: Vec<GfaLocation<IndexType>>,
    distances: Vec<OptionalGfaPathLength<IndexType>>,
}

impl<IndexType> Query<IndexType> {
    pub fn new(source: GfaLocation<IndexType>, targets: Vec<GfaLocation<IndexType>>) -> Self {
        Self {
            source,
            targets,
            distances: Vec::new(),
        }
    }

    pub fn source(&self) -> &GfaLocation<IndexType> {
        &self.source
    }

    pub fn targets(&self) -> &[GfaLocation<IndexType>] {
        &self.targets
    }

    pub fn distances(&self) -> &[OptionalGfaPathLength<IndexType>] {
        &self.distances
    }

    pub fn set_distances(&mut self, distances: Vec<OptionalGfaPathLength<IndexType>>) {
        self.distances = distances;
    }
}

/// Read a query file: per line a source position and one or more target positions, each three
/// tab-separated columns `node orientation offset` (see USAGE.md). Blank lines and lines starting
/// with `#` are skipped. `node_by_name` resolves a node name to its node and its length in bp (for
/// a shrunk index, the ORIGINAL node and length); an offset must lie in `0..=len`. Every error names
/// its line.
pub fn read_query_file<IndexType: GraphIndexInteger>(
    mut reader: impl BufRead,
    file_name: impl AsRef<Path>,
    node_by_name: impl Fn(&str) -> Option<(NodeIndex<IndexType>, u64)>,
) -> anyhow::Result<Vec<Query<IndexType>>> {
    let file_name = file_name.as_ref();
    let mut queries = Vec::new();
    let mut line = String::new();
    let mut line_number = 0usize;
    loop {
        line.clear();
        let read = reader.read_line(&mut line).with_context(|| {
            format!(
                "Failed to read line {} of query file {file_name:?}",
                line_number + 1
            )
        })?;
        if read == 0 {
            break;
        }
        line_number += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let query = parse_query_line(trimmed, &node_by_name)
            .with_context(|| format!("Invalid query in file {file_name:?}, line {line_number}"))?;
        queries.push(query);
    }
    Ok(queries)
}

/// One query line (trimmed, not blank).
fn parse_query_line<IndexType: GraphIndexInteger>(
    line: &str,
    node_by_name: impl Fn(&str) -> Option<(NodeIndex<IndexType>, u64)>,
) -> anyhow::Result<Query<IndexType>> {
    let columns = line.split('\t').collect_vec();
    if columns.len() < 6 || !columns.len().is_multiple_of(3) {
        let hint = if columns.iter().any(|column| column.contains(' ')) {
            " (the line seems space-separated; columns are separated by tabs)"
        } else {
            ""
        };
        bail!(
            "expected a source and at least one target, three tab-separated columns each \
             (node, orientation, offset), got {} columns{hint}",
            columns.len()
        );
    }
    let mut locations = columns
        .chunks(3)
        .map(|column| parse_location(column[0], column[1], column[2], &node_by_name));
    let source = locations.next().expect("at least six columns")?;
    let targets = locations.collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Query::new(source, targets))
}

/// One `node orientation offset` position.
fn parse_location<IndexType: GraphIndexInteger>(
    node_name: &str,
    orientation: &str,
    offset: &str,
    node_by_name: impl Fn(&str) -> Option<(NodeIndex<IndexType>, u64)>,
) -> anyhow::Result<GfaLocation<IndexType>> {
    let (node, len) =
        node_by_name(node_name).ok_or_else(|| anyhow!("node {node_name:?} is not in the graph"))?;
    let forward = match orientation {
        "+" => true,
        "-" => false,
        _ => bail!("node {node_name:?}: the orientation {orientation:?} is neither + nor -"),
    };
    let offset_value = offset.parse::<u64>().map_err(|_| {
        anyhow!("node {node_name:?}: the offset {offset:?} is not a non-negative integer")
    })?;
    if offset_value > len {
        bail!(
            "node {node_name:?}: offset {offset_value} is beyond the node's end (its length is \
             {len}, so offsets run 0..={len})"
        );
    }
    // A node's length fits the word size (checked when the graph or index was built), so every
    // offset up to it does; the check keeps a foreign index from causing a panic.
    let offset = usize::try_from(offset_value)
        .ok()
        .and_then(|raw| IndexType::try_from(raw).ok())
        .filter(|&raw| raw != IndexType::max_value())
        .ok_or_else(|| {
            anyhow!("node {node_name:?}: offset {offset_value} does not fit the word size")
        })?;
    Ok(GfaLocation::new(
        DirectedNodeIndex::from_bidirected(node, forward),
        GfaNodeOffset::from_raw(offset),
    ))
}

pub fn write_query_file<'index, IndexType: GraphIndexInteger>(
    mut writer: impl Write,
    queries: &[Query<IndexType>],
    node_to_name_index: impl Fn(NodeIndex<IndexType>) -> Option<Cow<'index, str>>,
) -> anyhow::Result<()> {
    for query in queries {
        write_location(&mut writer, &query.source, &node_to_name_index)
            .with_context(|| "Error writing query source location".to_string())?;

        for target in &query.targets {
            write!(writer, "\t").with_context(|| "Error writing query".to_string())?;
            write_location(&mut writer, target, &node_to_name_index)
                .with_context(|| "Error writing query target location".to_string())?;
        }

        writeln!(writer).with_context(|| "Error writing query".to_string())?;
    }

    Ok(())
}

fn write_location<'index, IndexType: GraphIndexInteger>(
    mut writer: impl Write,
    location: &GfaLocation<IndexType>,
    node_to_name_index: impl Fn(NodeIndex<IndexType>) -> Option<Cow<'index, str>>,
) -> anyhow::Result<()> {
    let node = location.node().into_bidirected();
    let node_name = node_to_name_index(node)
        .ok_or_else(|| anyhow!("Node index {node:?} not found in index"))?;
    let orientation = if location.node().is_forward() {
        "+"
    } else {
        "-"
    };
    let offset = location.offset();
    write!(writer, "{}\t{}\t{}", node_name, orientation, offset).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nodes "a" (2 bp), "b" (0 bp) and "c" (5 bp).
    fn node_by_name(name: &str) -> Option<(NodeIndex<u32>, u64)> {
        let (node, len) = match name {
            "a" => (0, 2),
            "b" => (1, 0),
            "c" => (2, 5),
            _ => return None,
        };
        Some((NodeIndex::from_usize(node), len))
    }

    fn read(text: &str) -> anyhow::Result<Vec<Query<u32>>> {
        read_query_file(text.as_bytes(), "q.tsv", node_by_name)
    }

    /// The error of reading `text`, with its causes.
    fn error(text: &str) -> String {
        format!("{:#}", read(text).err().expect("the read must fail"))
    }

    #[test]
    fn reads_queries_and_skips_blank_and_comment_lines() {
        let queries =
            read("# source, targets\n\na\t+\t0\tc\t-\t5\tb\t+\t0\r\n  \nc\t-\t2\ta\t+\t2\n")
                .unwrap();
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0].targets().len(), 2);
        let target = queries[0].targets()[0];
        assert_eq!(target.node(), NodeIndex::from_usize(2).into_directed(false));
        assert_eq!(target.offset().into_usize(), 5);
    }

    #[test]
    fn offsets_run_from_0_to_the_node_length() {
        assert!(read("a\t+\t2\tb\t+\t0\n").is_ok());
        let message = error("a\t+\t0\tc\t+\t1\n\na\t+\t3\tc\t+\t0\n");
        assert!(message.contains("line 3"), "{message}");
        assert!(message.contains("offset 3 is beyond"), "{message}");
        let message = error("a\t+\t0\tb\t+\t1\n");
        assert!(message.contains("0..=0"), "{message}");
        for offset in ["4294967295", "99999999999999999999", "-1", "x"] {
            let message = error(&format!("a\t+\t0\tc\t+\t{offset}\n"));
            assert!(message.contains("line 1"), "{message}");
        }
    }

    #[test]
    fn every_error_names_its_line() {
        let message = error("a\t+\t0\tc\t+\t0\nz\t+\t0\tc\t+\t0\n");
        assert!(
            message.contains("line 2") && message.contains("\"z\""),
            "{message}"
        );
        let message = error("a\t+\t0\tc\t*\t0\n");
        assert!(message.contains("neither + nor -"), "{message}");
        let message = error("a\t+\t0\tc\t+\n");
        assert!(message.contains("5 columns"), "{message}");
        let message = error("a\t+\t0\n");
        assert!(message.contains("3 columns"), "{message}");
        let message = error("a + 0 c + 0\n");
        assert!(message.contains("space-separated"), "{message}");
    }
}
