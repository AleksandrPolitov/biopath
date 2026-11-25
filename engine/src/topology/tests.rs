use std::{collections::HashSet, fs::File, io::BufReader};

use bidirected_adjacency_array::{
    graph::BidirectedAdjacencyArray,
    index::NodeIndex,
    io::gfa1::{GfaNodeData, PlainGfaEdgeData, PlainGfaNodeData},
};
use spqr_tree::decomposition::{SPQRDecomposition, indices::BlockIndex};

use super::{BlockNumbering, FORCE_LOC_LIST, NodeTopology, TopologySections};
use crate::{block_spqr_tree::BlockSpqrTree, rooted_tree::TreeAccess};

type Graph = BidirectedAdjacencyArray<u32, PlainGfaNodeData, PlainGfaEdgeData>;
type Spqr<'g> = SPQRDecomposition<'g, Graph>;

fn read_graph(gfa: &str) -> Graph {
    BidirectedAdjacencyArray::read_gfa1(BufReader::new(File::open(gfa).unwrap())).unwrap()
}

fn read_spqr<'g>(graph: &'g Graph, spqr_file: &str) -> Spqr<'g> {
    SPQRDecomposition::read_plain_spqr(graph, BufReader::new(File::open(spqr_file).unwrap()))
        .unwrap()
}

/// Every accessor of the projection must agree with the decomposition it was built from, and each
/// SPQR location must answer exactly what the block tree's own `SPQRNodeIndex -> local` lookup does:
/// `x` is the deepest containing SPQR node, and every ancestor-or-self of `x` contains the node iff
/// the lookup lists it.
fn assert_matches_decomposition(gfa: &str, spqr_file: &str) {
    let graph = read_graph(gfa);
    let spqr = read_spqr(&graph, spqr_file);
    // The topology's block ids are the chain numbering; the decomposition's are translated through it.
    let numbering = super::chain_block_numbering(&spqr);
    let blocks: Vec<_> = spqr.iter_blocks().map(|(_, block)| block).collect();
    // Blocks read from a `.spqr` file with fewer than three nodes carry no SPQR node (and so no tree).
    // Indexed by topology id.
    let trees: Vec<Option<BlockSpqrTree<u32>>> = (0..numbering.len())
        .map(|b| {
            let old = numbering.old_id(b);
            (blocks[old.into_usize()].spqr_node_count() > 0).then(|| BlockSpqrTree::new(&spqr, old))
        })
        .collect();
    let image = crate::image::Image::new_arena(1 << 30).unwrap();
    let (secs, build_locs) = TopologySections::write(&image, &spqr, &numbering, |b| {
        trees[b].as_ref().map(BlockSpqrTree::tree)
    });
    let check = Check {
        gfa,
        spqr: &spqr,
        numbering: &numbering,
        topology: secs.view_build(&image, &build_locs),
    };
    check.block_cut_lists();
    check.nodes();
    for v in 0..graph.node_count() {
        check.locations(NodeIndex::from_usize(v), &trees);
    }
    check.cut_blocks();
    check.block_parents_and_specials();
}

/// One fixture's topology against its decomposition.
struct Check<'c> {
    gfa: &'c str,
    spqr: &'c Spqr<'c>,
    numbering: &'c BlockNumbering,
    topology: NodeTopology<'c, u32>,
}

impl Check<'_> {
    /// Every block's cut list, in the decomposition's order.
    fn block_cut_lists(&self) {
        let (spqr, topology) = (self.spqr, &self.topology);
        assert_eq!(topology.block_count(), spqr.block_count());
        for (block_index, block) in spqr.iter_blocks() {
            let cuts: Vec<NodeIndex<u32>> = block
                .iter_cut_nodes()
                .map(|c| spqr.cut_node(c).node())
                .collect();
            assert_eq!(
                topology.block_cut_nodes(self.numbering.new_id(block_index)),
                cuts
            );
        }
    }

    /// Every node's component, cut id, length and blocks; the cut ids are a permutation.
    fn nodes(&self) {
        let (gfa, spqr, topology) = (self.gfa, self.spqr, &self.topology);
        let graph = spqr.graph();
        let mut cut_ids_seen = vec![false; topology.cut_count()];
        for v in 0..graph.node_count() {
            let node = NodeIndex::from_usize(v);
            if spqr.node_block_indices(node).next().is_some() {
                assert_eq!(
                    topology.node_component_index(node),
                    spqr.node_component_index(node)
                );
            }
            assert_eq!(topology.is_cut_node(node), spqr.is_cut_node(node));
            assert_eq!(
                topology.node_len(node).into_usize(),
                graph.node_data(node).sequence().len()
            );
            if spqr.is_cut_node(node) {
                let id = topology.cut_id(node).expect("a cut node has a dense id");
                assert_eq!(topology.cut_nodes()[id], node);
                assert!(!cut_ids_seen[id], "cut ids are a permutation");
                cut_ids_seen[id] = true;
                assert_eq!(topology.cut_recs()[id].node, node);
            } else {
                assert_eq!(topology.cut_id(node), None);
            }
            let blocks: HashSet<_> = spqr
                .node_block_indices(node)
                .map(|b| self.numbering.new_id(b))
                .collect();
            let got: Vec<_> = topology.node_block_indices(node).collect();
            assert_eq!(got.len(), blocks.len(), "{gfa}: node {v} block count");
            assert_eq!(
                got.into_iter().collect::<HashSet<_>>(),
                blocks,
                "{gfa}: node {v}"
            );
        }
        assert!(
            cut_ids_seen.iter().all(|&s| s),
            "{gfa}: every cut id is used"
        );
    }

    /// `node`'s SPQR location in every block against the block tree's lookup.
    fn locations(&self, node: NodeIndex<u32>, trees: &[Option<BlockSpqrTree<u32>>]) {
        let (gfa, v) = (self.gfa, node.into_usize());
        for (old_block, _) in self.spqr.iter_blocks() {
            let block_index = self.numbering.new_id(old_block);
            let got = self.topology.spqr_location(node, block_index);
            let Some(block_tree) = &trees[block_index.into_usize()] else {
                assert_eq!(
                    got, None,
                    "{gfa}: node {v}, block {block_index:?} has no tree"
                );
                continue;
            };
            let tree = block_tree.tree();
            let locals: Vec<usize> = (self.spqr.node_spqr_node_indices(node))
                .filter_map(|spqr_node| block_tree.local_of(spqr_node))
                .collect();
            let Some(x) = locals.iter().copied().max_by_key(|&l| tree.depth(l)) else {
                assert_eq!(got, None, "{gfa}: node {v}, block {block_index:?}");
                continue;
            };
            let got =
                got.unwrap_or_else(|| panic!("{gfa}: node {v}, block {block_index:?} is located"));
            assert_eq!(
                got.x, x,
                "{gfa}: node {v}, block {block_index:?}: deepest SPQR node"
            );
            let mut ancestor = Some(x);
            while let Some(a) = ancestor {
                assert_eq!(
                    got.contains_ancestor(tree, a),
                    locals.contains(&a),
                    "{gfa}: node {v}, block {block_index:?}: ancestor {a} of x = {x}"
                );
                ancestor = tree.parent(a);
            }
        }
    }

    /// The chain layout's blocks of every cut (chain numbering: the parent from its chain, the
    /// heavy child `roots + c`) contain it in the decomposition, and are all its blocks.
    fn cut_blocks(&self) {
        let (gfa, topology) = (self.gfa, &self.topology);
        let nav = topology.chain_nav();
        let nb = topology.block_count();
        for c in 0..topology.cut_count() {
            let rec = &topology.cut_recs()[c];
            let parent = BlockIndex::from_usize(nav.parent_block_of_cut(c));
            let heavy = BlockIndex::from_usize(
                nav.heavy_child_of_cut(c)
                    .expect("every cut has a child block"),
            );
            assert_eq!(
                heavy.into_usize(),
                topology.root_blocks() + c,
                "{gfa}: cut {c} heavy child"
            );
            assert!(
                topology.block_cut_nodes(parent).contains(&rec.node),
                "{gfa}: cut {c} parent"
            );
            assert!(
                topology.block_cut_nodes(heavy).contains(&rec.node),
                "{gfa}: cut {c} child"
            );
            assert_eq!(nav.parent(nb + c), Some(parent.into_usize()));
            let blocks: HashSet<_> = topology.node_block_indices(rec.node).collect();
            let mut from_layout: HashSet<_> = [parent, heavy].into_iter().collect();
            from_layout.extend(topology.cut_extra_blocks(c));
            assert_eq!(from_layout, blocks, "{gfa}: cut {c} blocks");
        }
    }

    /// Every block's parent cut is one of its cuts (or it hangs under the super-root), and the
    /// special-block table lists exactly the blocks whose cut count is not 2 (and the roots), with
    /// their cut lists in the decomposition's order.
    fn block_parents_and_specials(&self) {
        let (gfa, topology) = (self.gfa, &self.topology);
        let nav = topology.chain_nav();
        let (nb, nc) = (topology.block_count(), topology.cut_count());
        for b in 0..nb {
            let cuts = topology.block_cut_nodes(BlockIndex::from_usize(b));
            match nav.parent(b) {
                Some(p) if p < nb + nc => {
                    assert!(
                        cuts.contains(&topology.cut_recs()[p - nb].node),
                        "{gfa}: block {b} parent cut"
                    );
                }
                Some(p) => assert_eq!(p, nb + nc, "{gfa}: block {b} hangs under the super-root"),
                None => panic!("{gfa}: a block has a parent"),
            }
            let special = topology.special_block(b);
            if let Some((rec, _)) = special {
                // The O(1) special index (head record or heavy-child rank) names the same record.
                let by_index = topology
                    .special_block_recs_slice()
                    .get(nav.special_index(b))
                    .unwrap();
                assert!(
                    std::ptr::eq(by_index, rec),
                    "{gfa}: block {b} special index"
                );
            }
            let root = nav.parent(b) == Some(nb + nc);
            assert_eq!(
                special.is_some(),
                cuts.len() != 2 || root,
                "{gfa}: block {b} special iff not a plain 2-cut block"
            );
            if let Some((_, ids)) = special {
                let listed: Vec<_> = ids
                    .iter()
                    .map(|&id| topology.cut_recs()[id as usize].node)
                    .collect();
                assert_eq!(listed, cuts, "{gfa}: block {b} special cut list order");
            }
        }
    }
}

#[test]
fn topology_matches_decomposition_on_every_fixture() {
    // Twice: locations inline where they fit, then every location out of line.
    for force_list in [false, true] {
        FORCE_LOC_LIST.set(force_list);
        topology_matches_decomposition_on_fixtures();
    }
    FORCE_LOC_LIST.set(false);
}

fn topology_matches_decomposition_on_fixtures() {
    for (gfa, spqr) in [
        ("test_files/tiny1.gfa", "test_files/tiny1.spqr"),
        (
            "test_files/two_triangles.gfa",
            "test_files/two_triangles.spqr",
        ),
        ("test_files/chain5.gfa", "test_files/chain5.spqr"),
        (
            "test_files/rnode_selfflip.gfa",
            "test_files/rnode_selfflip.spqr",
        ),
        (
            "test_files/paper_bridge.gfa",
            "test_files/paper_bridge.spqr",
        ),
        (
            "test_files/GCA-41.clean.largest-block.gfa",
            "test_files/GCA-41.clean.largest-block.spqr",
        ),
    ] {
        assert_matches_decomposition(gfa, spqr);
    }
}

/// Before the lift lays the rows out, no node has rows (the topology reserves none: the lift's row
/// layout settles each node's form, see `NodeTopology::rewrite_row_starts`).
#[test]
fn no_rows_before_the_lift() {
    for (gfa, spqr_file) in [
        (
            "test_files/two_triangles.gfa",
            "test_files/two_triangles.spqr",
        ),
        ("test_files/chain5.gfa", "test_files/chain5.spqr"),
        (
            "test_files/GCA-41.clean.largest-block.gfa",
            "test_files/GCA-41.clean.largest-block.spqr",
        ),
    ] {
        let graph = read_graph(gfa);
        let spqr = read_spqr(&graph, spqr_file);
        let image = crate::image::Image::new_arena(1 << 30).unwrap();
        let (secs, build_locs, _) = TopologySections::write_with_all_trees(&image, &spqr);
        let topology = secs.view_build(&image, &build_locs);
        for v in 0..graph.node_count() {
            assert_eq!(
                topology.row_ref(NodeIndex::from_usize(v)),
                None,
                "{gfa}: node {v}"
            );
        }
        assert_eq!(topology.row_total(), 0, "{gfa}");
    }
}
