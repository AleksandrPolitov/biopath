//! The reduction: the side graph, its chains, the worklist that prunes dominated alleles, then G′.

use std::collections::{HashMap, VecDeque};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::GraphIndexInteger,
    io::gfa1::{GfaEdgeData, GfaNodeData},
};

use super::{
    MAX_ORIGINAL_NODES, OFF_MASK, ShrinkPlan, ShrinkStats,
    assemble::assemble_g_prime,
    chains::Chains,
    layout::RunLayout,
    records::{NO_PORT, PrunedChain},
    side_graph::{ChainMember, SideGraph, dir_id, is_right, side_node},
};
use crate::graph_view::graph_fingerprint;

/// Reduce `graph` (see the module docs). `Err` when the graph cannot be shrunk (overlapping links,
/// sizes beyond the alias encoding); the caller then indexes the graph as it is.
pub fn shrink_graph<IndexType: GraphIndexInteger, NodeData: GfaNodeData, EdgeData: GfaEdgeData>(
    graph: &BidirectedAdjacencyArray<IndexType, NodeData, EdgeData>,
) -> Result<ShrinkPlan<IndexType>, String> {
    let n = graph.node_count();
    if n as u64 > MAX_ORIGINAL_NODES {
        return Err(format!("{n} nodes exceed the alias encoding"));
    }
    let sg = SideGraph::new(graph)?;
    let stats = ShrinkStats {
        original_nodes: n,
        original_edges: graph.edge_count(),
        duplicate_edges: graph.edge_count() - sg.edges.len(),
        ..Default::default()
    };
    let mut reducer = Reducer::new(sg, stats);
    reducer.run()?;
    let Reducer {
        sg,
        mut chains,
        mut layout,
        mut stats,
        ..
    } = reducer;
    let shrunk = assemble_g_prime(&sg, &mut chains, &mut layout, &mut stats)?;
    layout.finish(n, &sg.lens);
    Ok(ShrinkPlan {
        graph: shrunk,
        runs: layout,
        stats,
        original_fingerprint: graph_fingerprint(graph),
    })
}

/// The worklist reduction. Prunes are applied one at a time, each valid in the graph of the moment,
/// so chain ids follow the pruning order.
struct Reducer {
    sg: SideGraph,
    chains: Chains,
    /// The alias of the chains pruned so far.
    layout: RunLayout,
    stats: ShrinkStats,
    /// Dependency levels of pruned chains (see the alias): the largest level among the pruned
    /// chains whose port lies on each node.
    incoming_level: Vec<u32>,
    /// The allele kept so far between each unordered pair of attachment sides (its root).
    kept_alleles: HashMap<(u32, u32), u32>,
    /// Chain roots to examine.
    queue: VecDeque<u32>,
    /// The members of the chain being pruned.
    members: Vec<ChainMember>,
}

impl Reducer {
    fn new(sg: SideGraph, stats: ShrinkStats) -> Self {
        let n = sg.lens.len();
        Self {
            chains: Chains::new(&sg.lens),
            sg,
            layout: RunLayout::default(),
            stats,
            incoming_level: vec![0; n],
            kept_alleles: HashMap::new(),
            queue: VecDeque::new(),
            members: Vec::new(),
        }
    }

    /// Glue the chains, then run the worklist until nothing changes.
    fn run(&mut self) -> Result<(), String> {
        self.glue_initial_chains();
        while let Some(r) = self.queue.pop_front() {
            if let Some(victim) = self.victim(r as usize) {
                self.prune(victim)?;
            }
        }
        Ok(())
    }

    /// Join every link walked straight through and queue every chain.
    fn glue_initial_chains(&mut self) {
        for e in 0..self.sg.edges.len() {
            let (x, y) = (self.sg.edges[e].0 as usize, self.sg.edges[e].1 as usize);
            if self.sg.chainable(x, y) {
                self.chains.join(x, y);
            }
        }
        let n = self.sg.lens.len();
        self.queue = (0..n)
            .filter(|&v| self.chains.find(v) == v)
            .map(|v| v as u32)
            .collect();
    }

    /// An allele's unordered pair of attachment sides, if `root` is one.
    fn allele_key(&mut self, root: usize) -> Option<(u32, u32)> {
        let (p, q) = self.chains.allele_partners(&self.sg, root)?;
        Some((p.min(q) as u32, p.max(q) as u32))
    }

    /// The attachment sides `(entry, exit)` of chain `root` if it may be pruned: an allele's two, or
    /// a tip's one (`None` at its unlinked end).
    fn attachments(&mut self, root: usize) -> Option<(Option<usize>, Option<usize>)> {
        if let Some((p, q)) = self.chains.allele_partners(&self.sg, root) {
            return Some((Some(p), Some(q)));
        }
        self.chains.tip_partner(&self.sg, root)
    }

    /// The chain the step at `r` prunes, if any: `r` when it is a tip or an
    /// allele dominated by a direct link or by the allele kept between the same sides, that kept
    /// allele when `r` is shorter (ties: the lower root is kept). The first allele between two sides
    /// is kept.
    fn victim(&mut self, r: usize) -> Option<usize> {
        let Some(key) = self.allele_key(r) else {
            let tip = self.chains.tip_partner(&self.sg, r).is_some();
            return tip.then_some(r);
        };
        if self.sg.has_direct_link(key.0, key.1) {
            self.stats.direct_chains += 1;
            return Some(r);
        }
        match self.kept_alleles.get(&key).copied() {
            Some(d) if d as usize != r && self.allele_key(d as usize) == Some(key) => {
                let d = d as usize;
                if (self.chains.len[d], d) <= (self.chains.len[r], r) {
                    Some(r)
                } else {
                    self.kept_alleles.insert(key, r as u32);
                    Some(d)
                }
            }
            _ => {
                self.kept_alleles.insert(key, r as u32);
                None
            }
        }
    }

    /// Prune `victim`, walked from end `a` to end `b`, entered from `pa`'s node and leaving into
    /// `pb`'s (a tip lacks one of them): record its alias, remove its nodes, then glue or requeue its
    /// attachment sides.
    fn prune(&mut self, victim: usize) -> Result<(), String> {
        let (a, b) = self.chains.ends(victim);
        let (pa, pb) = self
            .attachments(victim)
            .unwrap_or_else(|| unreachable!("a pruned chain is an allele or a tip"));
        let tip = pa.is_none() || pb.is_none();
        let len = self.chains.len[victim];
        if len > OFF_MASK {
            return Err(format!(
                "pruned chain of length {len} exceeds the alias encoding"
            ));
        }
        // Entered from the end of the node that leaves through `pa`; exits into the start of the
        // node that enters through `pb`.
        let entry = pa.map_or(NO_PORT, |p| dir_id(side_node(p), is_right(p)));
        let exit = pb.map_or(NO_PORT, |p| dir_id(side_node(p), !is_right(p)));
        self.sg.walk(a, b, &mut self.members, |_| {});
        let level = self.raise_level(pa, pb);
        if let [single] = self.members[..] {
            self.layout
                .add_single(single.node, single.forward, entry, exit, level);
        } else {
            let chain = PrunedChain {
                len: len as u32,
                exit,
                entry,
                level,
            };
            self.layout.add_chain(chain, self.members.clone());
        }
        if tip {
            self.stats.tip_chains += 1;
            self.stats.tip_nodes += self.members.len();
        } else {
            self.stats.allele_chains += 1;
            self.stats.allele_nodes += self.members.len();
        }
        for m in &self.members {
            self.sg.remove_node(m.node);
        }
        for p in [pa, pb].into_iter().flatten() {
            self.reattach(p);
        }
        Ok(())
    }

    /// The level of the chain being pruned (its members in `self.members`): one more than the
    /// largest level whose port lies on one of its nodes. Its ports' nodes `pa`, `pb` get at least it.
    fn raise_level(&mut self, pa: Option<usize>, pb: Option<usize>) -> u32 {
        let level = 1 + self
            .members
            .iter()
            .map(|m| self.incoming_level[m.node])
            .max()
            .unwrap_or(0);
        for p in [pa, pb].into_iter().flatten() {
            let port = &mut self.incoming_level[side_node(p)];
            *port = (*port).max(level);
        }
        self.stats.max_level = self.stats.max_level.max(level);
        level
    }

    /// Attachment side `p` lost a link: glue the link it is now walked straight through, else
    /// recheck its chain.
    fn reattach(&mut self, p: usize) {
        let node = side_node(p);
        if !self.sg.alive_node[node] {
            return;
        }
        if self.sg.deg[p] == 1 && !self.sg.blocked[node] {
            let (_, t) = self.sg.only_link(p);
            if self.sg.chainable(p, t) {
                let root = self.chains.join(p, t);
                self.queue.push_back(root as u32);
                return;
            }
        }
        self.queue.push_back(self.chains.find(node) as u32);
    }
}
