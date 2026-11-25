//! The GFA node names: integer runs, or a byte table with a hash index.

use std::borrow::Cow;

use crate::{
    image::{
        Image, Sec, SectionId,
        hash::{self, HashSlot},
    },
    sections,
};

/// One run of consecutive integer node names: nodes `node .. next.node` are named `name, name + 1, …`
/// (the last entry is a sentinel holding the node count).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NameRun {
    pub node: u64,
    pub name: u64,
}
crate::image::image_record!(NameRun = [u64, u64]);

/// The integer names of nodes `0, 1, 2, ...` (in node order) as maximal runs of consecutive names:
/// a new run starts wherever a name is not its predecessor's plus one. `None` as soon as one name is
/// not an integer (`None` item). The GBZ reader builds these from its node ids, so the index never
/// reads the names back.
pub fn integer_name_runs(names: impl IntoIterator<Item = Option<u64>>) -> Option<Vec<NameRun>> {
    let mut runs: Vec<NameRun> = Vec::new();
    let mut previous: Option<u64> = None;
    for (v, name) in names.into_iter().enumerate() {
        let name = name?;
        if previous.and_then(|p| p.checked_add(1)) != Some(name) {
            runs.push(NameRun {
                node: v as u64,
                name,
            });
        }
        previous = Some(name);
    }
    Some(runs)
}

/// `name` as an integer if it is in canonical decimal form (digits only, no leading zero, fits `u64`),
/// so that integer and string compare alike: `"007"` is not the node named `7`.
fn canonical_integer(name: &str) -> Option<u64> {
    let bytes = name.as_bytes();
    let canonical = !bytes.is_empty()
        && bytes.iter().all(u8::is_ascii_digit)
        && (bytes[0] != b'0' || bytes.len() == 1);
    if canonical { name.parse().ok() } else { None }
}

/// The section ids of one name table (the graph's own, or the original graph's of a shrunk index).
#[derive(Clone, Copy, Debug)]
pub(crate) struct NameIds {
    pub runs: SectionId,
    pub run_order: SectionId,
    pub bytes: SectionId,
    pub offsets: SectionId,
    pub hash: SectionId,
}

/// The engine graph's own name table.
pub(crate) const GRAPH_NAME_IDS: NameIds = NameIds {
    runs: sections::GRAPH_NAME_RUNS,
    run_order: sections::GRAPH_NAME_RUN_ORDER,
    bytes: sections::GRAPH_NAME_BYTES,
    offsets: sections::GRAPH_NAME_OFFSETS,
    hash: sections::GRAPH_NAME_HASH,
};

/// How node names are stored. Pangenome graphs name their nodes with integers that mostly ascend
/// with the node order, so they store runs of consecutive names: a binary search either way
/// (node -> name, and name -> node through `order`, the runs sorted by name), a few bytes per run
/// where the byte table plus its hash index costs ~50 B per node. Any other graph keeps the byte table.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NameSections {
    Bytes {
        bytes: Sec<u8>,
        offsets: Sec<u32>,
        hash: Sec<HashSlot>,
    },
    Runs {
        runs: Sec<NameRun>,
        order: Sec<u32>,
    },
}

impl NameSections {
    /// Write the names of nodes `0..n` (`name_of(v)`) under `ids`: integer runs when every name is a
    /// canonical integer, else bytes + offsets + a hash index (name hash -> node, verified).
    pub(crate) fn write<'n>(
        image: &Image,
        n: usize,
        name_of: impl Fn(usize) -> Cow<'n, str>,
        ids: NameIds,
    ) -> Self {
        // One streaming pass: the runs grow while every name is an integer (no `n`-sized vector).
        let runs = integer_name_runs((0..n).map(|v| canonical_integer(&name_of(v))));
        if let Some(runs) = runs {
            return Self::write_runs(image, n, runs, ids);
        }
        let mut name_offsets = vec![0u32; n + 1];
        let mut bytes_total = 0usize;
        for (v, end) in name_offsets[1..].iter_mut().enumerate() {
            bytes_total += name_of(v).len();
            *end = u32::try_from(bytes_total).unwrap_or_else(|_| {
                panic!(
                    "the graph's node names take more than {} bytes, a hard limit of the index: \
                     shorten them, or name the nodes with integers (stored without a name table)",
                    u32::MAX
                )
            });
        }
        let (name_bytes, bytes_out) = image.alloc_section::<u8>(ids.bytes, bytes_total);
        let cap = hash::capacity_for(n);
        let (name_hash, slots) = image.alloc_section::<HashSlot>(ids.hash, cap);
        slots.fill(HashSlot::EMPTY);
        let mut keys = Vec::with_capacity(n);
        for (v, &start) in name_offsets[..n].iter().enumerate() {
            let name = name_of(v);
            let name: &str = &name;
            let start = start as usize;
            bytes_out[start..start + name.len()].copy_from_slice(name.as_bytes());
            keys.push((hash::hash_bytes(name.as_bytes()), v as u32));
        }
        hash::build_into_multi(slots, keys.into_iter());
        NameSections::Bytes {
            bytes: name_bytes,
            offsets: image.push_section(ids.offsets, &name_offsets),
            hash: name_hash,
        }
    }

    /// Write a name table whose names are integer runs already known to the caller (`runs` as
    /// [`integer_name_runs`] returns them for the names of nodes `0..n`), without reading any name.
    pub(crate) fn write_runs(
        image: &Image,
        n: usize,
        mut runs: Vec<NameRun>,
        ids: NameIds,
    ) -> Self {
        debug_assert!(
            runs.windows(2).all(|w| w[0].node < w[1].node)
                && runs.last().map_or(n == 0, |r| r.node < n as u64)
        );
        let mut order: Vec<u32> = (0..runs.len() as u32).collect();
        order.sort_unstable_by_key(|&r| runs[r as usize].name);
        runs.push(NameRun {
            node: n as u64,
            name: 0,
        });
        NameSections::Runs {
            runs: image.push_section(ids.runs, &runs),
            order: image.push_section(ids.run_order, &order),
        }
    }

    /// Look a name table up in a loaded image.
    pub(crate) fn from_image(image: &Image, ids: NameIds) -> std::io::Result<Self> {
        Ok(match image.section(ids.runs)? {
            Some(runs) => NameSections::Runs {
                runs,
                order: image.require(ids.run_order, "name_run_order")?,
            },
            None => NameSections::Bytes {
                bytes: image.require(ids.bytes, "name_bytes")?,
                offsets: image.require(ids.offsets, "name_offsets")?,
                hash: image.require(ids.hash, "name_hash")?,
            },
        })
    }

    /// Resolve against the image.
    pub(crate) fn view<'a>(&self, image: &'a Image) -> NameView<'a> {
        match *self {
            NameSections::Bytes {
                bytes,
                offsets,
                hash,
            } => NameView::Bytes {
                bytes: image.get(bytes),
                offsets: image.get(offsets),
                hash: image.get(hash),
            },
            NameSections::Runs { runs, order } => NameView::Runs {
                runs: image.get(runs),
                order: image.get(order),
            },
        }
    }
}

/// A resolved name table (see [`NameSections`]).
#[derive(Clone, Copy)]
pub(crate) enum NameView<'a> {
    Bytes {
        bytes: &'a [u8],
        offsets: &'a [u32],
        hash: &'a [HashSlot],
    },
    Runs {
        runs: &'a [NameRun],
        order: &'a [u32],
    },
}

impl<'a> NameView<'a> {
    /// The name of node `v`.
    pub(crate) fn node_name(&self, v: usize) -> Cow<'a, str> {
        match *self {
            NameView::Bytes { bytes, offsets, .. } => {
                String::from_utf8_lossy(&bytes[offsets[v] as usize..offsets[v + 1] as usize])
            }
            NameView::Runs { runs, .. } => {
                // The run holding `v`: the last whose first node is <= v (the sentinel is past every node).
                let run = runs[runs.partition_point(|run| run.node <= v as u64) - 1];
                Cow::Owned((run.name + (v as u64 - run.node)).to_string())
            }
        }
    }

    /// The node named `name`, if any.
    pub(crate) fn node_by_name(&self, name: &str) -> Option<usize> {
        match *self {
            NameView::Bytes {
                bytes,
                offsets,
                hash,
            } => {
                let key = hash::hash_bytes(name.as_bytes());
                hash::lookup_verified(hash, key, |v| {
                    let v = v as usize;
                    &bytes[offsets[v] as usize..offsets[v + 1] as usize] == name.as_bytes()
                })
                .map(|v| v as usize)
            }
            NameView::Runs { runs, order } => {
                let value = canonical_integer(name)?;
                // The run with the greatest first name <= value, then check `value` is inside it.
                let i = order.partition_point(|&r| runs[r as usize].name <= value);
                let r = *order.get(i.checked_sub(1)?)? as usize;
                let (run, next) = (runs[r], runs[r + 1]);
                let offset = value - run.name;
                (offset < next.node - run.node).then_some((run.node + offset) as usize)
            }
        }
    }
}
