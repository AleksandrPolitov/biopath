//! A rooted tree with binary-lifting LCA queries.
//!
//! `spqr-tree` exposes the SPQR tree of a block as SPQR nodes connected by SPQR (virtual) edges,
//! but without a root or parent/child structure. The engine needs a rooted tree so that each
//! non-root SPQR node has a single *reference* (parent) virtual edge whose poles are its reference
//! poles, and so that queries can find the lowest common ancestor of two nodes.
//!
//! This type is deliberately generic over abstract node ids `0..node_count` so it can be unit
//! tested in isolation from the SPQR decomposition; [`crate::block_spqr_tree::BlockSpqrTree`] maps
//! `SPQRNodeIndex`es onto these ids.

/// A rooted tree over the node ids `0..node_count`, supporting ancestor and LCA queries in
/// `O(log node_count)`.
#[derive(Debug, Clone)]
pub struct RootedTree {
    root: u32,
    depth: Box<[u32]>,
    /// Children in CSR form: the children of `v` are `children_flat[children_offsets[v]..
    /// children_offsets[v + 1]]` (two flat vectors instead of one `Vec` per node, which on a million
    /// tiny block trees dominated the allocation count). Build-time only: the image keeps the depths
    /// and the lifting table.
    children_flat: Box<[usize]>,
    children_offsets: Box<[usize]>,
    /// Build-time only, like the children.
    postorder: Box<[usize]>,
    /// Binary-lifting table flattened row-major: `up_flat[k * node_count + v]` is the `2^k`-th
    /// ancestor of `v`, saturating at the root (the root is its own ancestor beyond depth 0). One
    /// allocation instead of `(log+1)`. Stored as `u32` (node ids are `< node_count`, well under
    /// `u32::MAX`) — the dominant retained term on a large tree ((log+1) x node_count). Callers see
    /// `usize` via a cast at the boundary.
    up_flat: Box<[u32]>,
    /// Rows in `up_flat` minus one. Sized from the tree's **depth**, not its node count: ancestry
    /// saturates at the root, so no query ever needs a longer jump. A broad shallow tree (the usual
    /// SPQR shape) then needs `⌈log₂(depth+1)⌉` rows instead of `⌈log₂(node_count)⌉`: 8 instead of
    /// 21 for a 100-deep tree of 680 k nodes.
    log: u32,
    /// The greatest depth in the tree, so [`Self::kth_ancestor`] can answer an over-long jump
    /// without indexing past the last lifting row.
    max_depth: u32,
}

/// Read access to a rooted tree's parent/LCA arrays, implemented by the owned [`RootedTree`] and the
/// image-backed [`RootedTreeView`], so ancestor queries are written once.
pub trait TreeAccess {
    /// Depth of every node (root `0`).
    fn depth_slice(&self) -> &[u32];
    /// Binary-lifting table flattened row-major: `up_flat[k * n + v]` = `2^k`-th ancestor of `v`.
    fn up_flat(&self) -> &[u32];
    /// The root node id.
    fn root(&self) -> usize;
    /// Rows in `up_flat` minus one.
    fn log(&self) -> u32;
    /// The greatest depth in the tree.
    fn max_depth(&self) -> u32;

    /// The number of nodes.
    #[inline]
    fn node_count(&self) -> usize {
        self.depth_slice().len()
    }

    /// The parent of `v`, or `None` if `v` is the root.
    #[inline]
    fn parent(&self, v: usize) -> Option<usize> {
        (v != self.root()).then_some(self.up_flat()[v] as usize)
    }

    /// The depth of `v` (the root has depth `0`).
    #[inline]
    fn depth(&self, v: usize) -> usize {
        self.depth_slice()[v] as usize
    }

    /// The `k`-th ancestor of `v`, saturating at the root (further than the root returns the root).
    fn kth_ancestor(&self, mut v: usize, k: usize) -> usize {
        // Beyond the deepest node every jump has already saturated at the root, and the lifting
        // table is only sized for jumps up to `max_depth`.
        if k > self.max_depth() as usize {
            return self.root();
        }
        let up = self.up_flat();
        let n = self.node_count();
        let mut bit = 0;
        while (1usize << bit) <= k {
            if (k >> bit) & 1 == 1 {
                v = up[bit * n + v] as usize;
            }
            bit += 1;
        }
        v
    }

    /// Whether `ancestor` is an ancestor of (or equal to) `descendant`.
    fn is_ancestor(&self, ancestor: usize, descendant: usize) -> bool {
        let ancestor_depth = self.depth(ancestor);
        let descendant_depth = self.depth(descendant);
        ancestor_depth <= descendant_depth
            && self.kth_ancestor(descendant, descendant_depth - ancestor_depth) == ancestor
    }

    /// The lowest common ancestor of `u` and `v`.
    fn lca(&self, mut u: usize, mut v: usize) -> usize {
        if self.depth(u) < self.depth(v) {
            std::mem::swap(&mut u, &mut v);
        }
        // Raise the deeper node to the same depth.
        u = self.kth_ancestor(u, self.depth(u) - self.depth(v));
        if u == v {
            return u;
        }
        // Raise both together until their parents coincide.
        let up = self.up_flat();
        let n = self.node_count();
        for k in (0..=self.log() as usize).rev() {
            if up[k * n + u] != up[k * n + v] {
                u = up[k * n + u] as usize;
                v = up[k * n + v] as usize;
            }
        }
        up[u] as usize
    }

    /// The child of `ancestor` on the path towards `descendant`, or `None` if `descendant` is not a
    /// strict descendant of `ancestor`.
    fn child_on_path_to(&self, ancestor: usize, descendant: usize) -> Option<usize> {
        if ancestor == descendant || !self.is_ancestor(ancestor, descendant) {
            return None;
        }
        Some(self.kth_ancestor(
            descendant,
            self.depth(descendant) - self.depth(ancestor) - 1,
        ))
    }
}

/// An image-backed rooted tree (parent/LCA arrays only; see [`TreeAccess`]).
#[derive(Clone, Copy, Debug)]
pub struct RootedTreeView<'a> {
    pub depth: &'a [u32],
    pub up_flat: &'a [u32],
    pub root: u32,
    pub log: u32,
    pub max_depth: u32,
}

impl TreeAccess for RootedTreeView<'_> {
    #[inline]
    fn depth_slice(&self) -> &[u32] {
        self.depth
    }
    #[inline]
    fn up_flat(&self) -> &[u32] {
        self.up_flat
    }
    #[inline]
    fn root(&self) -> usize {
        self.root as usize
    }
    #[inline]
    fn log(&self) -> u32 {
        self.log
    }
    #[inline]
    fn max_depth(&self) -> u32 {
        self.max_depth
    }
}

impl TreeAccess for RootedTree {
    #[inline]
    fn depth_slice(&self) -> &[u32] {
        &self.depth
    }
    #[inline]
    fn up_flat(&self) -> &[u32] {
        &self.up_flat
    }
    #[inline]
    fn root(&self) -> usize {
        self.root as usize
    }
    #[inline]
    fn log(&self) -> u32 {
        self.log
    }
    #[inline]
    fn max_depth(&self) -> u32 {
        self.max_depth
    }
}

/// Every node's parent (`None` at the root) and depth, by a BFS from `root` over the undirected
/// `edges` (iterative: SPQR trees can be deep).
///
/// # Panics
/// Panics if an endpoint is out of range or the edges do not connect every node.
fn bfs_parents(
    node_count: usize,
    edges: &[(usize, usize)],
    root: usize,
) -> (Vec<Option<usize>>, Vec<u32>) {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(a, b) in edges {
        assert!(
            a < node_count && b < node_count,
            "edge endpoint out of range"
        );
        adjacency[a].push(b);
        adjacency[b].push(a);
    }
    let mut parent = vec![None; node_count];
    let mut depth = vec![0u32; node_count];
    let mut visited = vec![false; node_count];
    let mut queue = std::collections::VecDeque::new();
    visited[root] = true;
    queue.push_back(root);
    while let Some(v) = queue.pop_front() {
        for &w in &adjacency[v] {
            if !visited[w] {
                visited[w] = true;
                parent[w] = Some(v);
                depth[w] = depth[v] + 1;
                queue.push_back(w);
            }
        }
    }
    assert!(
        visited.iter().all(|&v| v),
        "the given edges do not form a single connected tree over all nodes",
    );
    (parent, depth)
}

/// The children in CSR form, `(offsets, flat)` (counting sort by parent): the children of `p` are
/// `flat[offsets[p]..offsets[p + 1]]`, in ascending order.
fn children_csr(parent: &[Option<usize>]) -> (Vec<usize>, Vec<usize>) {
    let node_count = parent.len();
    let mut children_offsets = vec![0usize; node_count + 1];
    for p in parent.iter().flatten() {
        children_offsets[p + 1] += 1;
    }
    for i in 0..node_count {
        children_offsets[i + 1] += children_offsets[i];
    }
    let mut children_flat = vec![0usize; children_offsets[node_count]];
    let mut cursor = children_offsets.clone();
    for (v, p) in parent.iter().enumerate() {
        if let Some(p) = *p {
            children_flat[cursor[p]] = v;
            cursor[p] += 1;
        }
    }
    (children_offsets, children_flat)
}

/// The nodes in DFS postorder from `root` (children strictly before their parent), iteratively.
fn postorder(root: usize, children_offsets: &[usize], children_flat: &[usize]) -> Vec<usize> {
    let mut postorder = Vec::with_capacity(children_offsets.len() - 1);
    let mut stack = vec![(root, false)];
    while let Some((v, processed)) = stack.pop() {
        if processed {
            postorder.push(v);
        } else {
            stack.push((v, true));
            for &c in &children_flat[children_offsets[v]..children_offsets[v + 1]] {
                stack.push((c, false));
            }
        }
    }
    postorder
}

/// The binary-lifting table flattened row-major (`up_flat[k * node_count + v]` is the `2^k`-th
/// ancestor of `v`, saturating at the root) and its `log`, the smallest value with
/// `2^(log + 1) > max_depth`: the table is sized by the depth, since ancestry saturates at the root.
fn lifting_table(parent: &[Option<usize>], root: usize, max_depth: u32) -> (Vec<u32>, usize) {
    let node_count = parent.len();
    let mut log = 0;
    while (1usize << (log + 1)) <= max_depth as usize {
        log += 1;
    }
    let mut up_flat = vec![root as u32; (log + 1) * node_count];
    for v in 0..node_count {
        up_flat[v] = parent[v].unwrap_or(v) as u32; // k = 0 row
    }
    for k in 1..=log {
        let prev = (k - 1) * node_count;
        let cur = k * node_count;
        for v in 0..node_count {
            up_flat[cur + v] = up_flat[prev + up_flat[prev + v] as usize];
        }
    }
    (up_flat, log)
}

impl RootedTree {
    /// Root the tree given by `edges` (undirected) over `node_count` nodes at `root`.
    ///
    /// # Panics
    /// Panics if `node_count == 0`, if `root`/an endpoint is out of range, or if the edges do not
    /// connect every node (i.e. they are not a single tree over all nodes).
    pub fn new(node_count: usize, edges: &[(usize, usize)], root: usize) -> Self {
        assert!(node_count > 0, "a rooted tree needs at least one node");
        assert!(root < node_count, "root {root} out of range");
        assert!(
            node_count < u32::MAX as usize,
            "rooted tree exceeds u32 local-id space"
        );
        let (parent, depth) = bfs_parents(node_count, edges, root);
        let (children_offsets, children_flat) = children_csr(&parent);
        let postorder = postorder(root, &children_offsets, &children_flat);
        let max_depth = depth.iter().copied().max().unwrap_or(0);
        let (up_flat, log) = lifting_table(&parent, root, max_depth);
        Self {
            root: root as u32,
            depth: depth.into_boxed_slice(),
            children_flat: children_flat.into_boxed_slice(),
            children_offsets: children_offsets.into_boxed_slice(),
            postorder: postorder.into_boxed_slice(),
            up_flat: up_flat.into_boxed_slice(),
            log: log as u32,
            max_depth,
        }
    }

    /// Borrow as an image-style view.
    pub fn as_view(&self) -> RootedTreeView<'_> {
        RootedTreeView {
            depth: &self.depth,
            up_flat: &self.up_flat,
            root: self.root,
            log: self.log,
            max_depth: self.max_depth,
        }
    }

    /// Bytes of the binary-lifting ancestor table `up_flat` (`(log+1) x node_count` entries).
    #[cfg(test)]
    pub fn lift_table_bytes(&self) -> usize {
        self.up_flat.len() * std::mem::size_of::<u32>()
    }

    /// The children of `v`, ascending.
    pub fn children(&self, v: usize) -> &[usize] {
        &self.children_flat[self.children_offsets[v]..self.children_offsets[v + 1]]
    }

    /// A postorder over the nodes: every node appears after all of its descendants. Suitable for a
    /// bottom-up DP.
    pub fn postorder(&self) -> &[usize] {
        &self.postorder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent (naive) LCA via parent chains, used to cross-check the binary-lifting LCA.
    fn naive_lca(tree: &RootedTree, u: usize, v: usize) -> usize {
        let mut ancestors_u = vec![u];
        let mut cur = u;
        while let Some(p) = tree.parent(cur) {
            ancestors_u.push(p);
            cur = p;
        }
        let mut cur = v;
        loop {
            if ancestors_u.contains(&cur) {
                return cur;
            }
            cur = tree
                .parent(cur)
                .expect("reached root without meeting u's ancestors");
        }
    }

    #[test]
    fn single_node() {
        let tree = RootedTree::new(1, &[], 0);
        assert_eq!(tree.root(), 0);
        assert_eq!(tree.parent(0), None);
        assert_eq!(tree.depth(0), 0);
        assert_eq!(tree.lca(0, 0), 0);
        assert_eq!(tree.postorder(), &[0]);
        assert_eq!(tree.child_on_path_to(0, 0), None);
        assert!(tree.is_ancestor(0, 0));
        assert_eq!(
            tree.lift_table_bytes(),
            std::mem::size_of::<u32>(),
            "a depth-zero tree needs exactly one parent row"
        );
    }

    #[test]
    fn path_rooted_at_end() {
        // 0 - 1 - 2 - 3 - 4, rooted at 0.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
        let tree = RootedTree::new(5, &edges, 0);
        for v in 0..5 {
            assert_eq!(tree.depth(v), v);
        }
        assert_eq!(tree.lca(4, 2), 2);
        assert_eq!(tree.lca(3, 1), 1);
        assert_eq!(tree.lca(4, 0), 0);
        assert_eq!(tree.lca(2, 2), 2);
        assert!(tree.is_ancestor(1, 4));
        assert!(!tree.is_ancestor(3, 1));
        assert_eq!(tree.child_on_path_to(0, 4), Some(1));
        assert_eq!(tree.child_on_path_to(2, 4), Some(3));
        assert_eq!(tree.child_on_path_to(4, 2), None); // not a descendant
    }

    #[test]
    fn path_rooted_in_middle_changes_structure() {
        // Same chain 0-1-2-3-4 but rooted at 2: depths grow outward, LCA of the two ends is 2.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
        let tree = RootedTree::new(5, &edges, 2);
        assert_eq!(tree.depth(2), 0);
        assert_eq!(tree.depth(0), 2);
        assert_eq!(tree.depth(4), 2);
        assert_eq!(tree.lca(0, 4), 2);
        assert_eq!(tree.lca(0, 1), 1);
        assert_eq!(tree.child_on_path_to(2, 0), Some(1));
        assert_eq!(tree.child_on_path_to(2, 4), Some(3));
    }

    #[test]
    fn star() {
        // center 0 with leaves 1,2,3,4.
        let edges = [(0, 1), (0, 2), (0, 3), (0, 4)];
        let tree = RootedTree::new(5, &edges, 0);
        for leaf in 1..5 {
            assert_eq!(tree.depth(leaf), 1);
            assert_eq!(tree.child_on_path_to(0, leaf), Some(leaf));
        }
        assert_eq!(tree.lca(1, 4), 0);
        assert_eq!(tree.lca(2, 2), 2);
    }

    #[test]
    fn balanced_binary_tree() {
        // Complete binary tree with parent(i) = (i-1)/2 for i in 1..15.
        let edges: Vec<(usize, usize)> = (1..15).map(|i| ((i - 1) / 2, i)).collect();
        let tree = RootedTree::new(15, &edges, 0);
        // 3 and 4 are children of 1; their LCA is 1.
        assert_eq!(tree.lca(3, 4), 1);
        // 7 and 8 are grandchildren under 1 (7 under 3, 8 under 3? no: parent(7)=3, parent(8)=3).
        assert_eq!(tree.lca(7, 8), 3);
        // Leaves in different halves: 7 (under 1) and 12 (under 2) -> root 0.
        assert_eq!(tree.lca(7, 12), 0);
        // child_on_path_to root towards a deep leaf goes through the correct half.
        assert_eq!(tree.child_on_path_to(0, 7), Some(1));
        assert_eq!(tree.child_on_path_to(0, 12), Some(2));
    }

    #[test]
    fn postorder_places_children_before_parents() {
        let edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)];
        let tree = RootedTree::new(6, &edges, 0);
        let post = tree.postorder();
        assert_eq!(post.len(), 6);
        let position: std::collections::HashMap<usize, usize> =
            post.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        for v in 0..6 {
            for &c in tree.children(v) {
                assert!(
                    position[&c] < position[&v],
                    "child {c} must precede parent {v}"
                );
            }
        }
    }

    #[test]
    fn lca_matches_naive_on_many_trees_and_roots() {
        // A handful of deterministic, irregular trees (caterpillar, skewed, branching), each rooted
        // at every possible node, cross-checked against the naive parent-chain LCA for all pairs.
        let trees: Vec<(usize, Vec<(usize, usize)>)> = vec![
            (1, vec![]),
            (2, vec![(0, 1)]),
            // caterpillar
            (7, vec![(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (1, 6)]),
            // skewed with a couple of branches
            (
                8,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (0, 7)],
            ),
            // bushy
            (
                9,
                vec![
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (1, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                    (7, 8),
                ],
            ),
        ];

        for (n, edges) in &trees {
            for root in 0..*n {
                let tree = RootedTree::new(*n, edges, root);
                for u in 0..*n {
                    for v in 0..*n {
                        let expected = naive_lca(&tree, u, v);
                        assert_eq!(tree.lca(u, v), expected, "n={n} root={root} u={u} v={v}",);
                    }
                }
            }
        }
    }

    #[test]
    fn is_ancestor_and_child_on_path_are_consistent_with_lca() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (0, 6)];
        let tree = RootedTree::new(7, &edges, 0);
        for a in 0..7 {
            for d in 0..7 {
                let is_anc = tree.is_ancestor(a, d);
                // is_ancestor(a, d) iff lca(a, d) == a.
                assert_eq!(is_anc, tree.lca(a, d) == a, "a={a} d={d}");

                match tree.child_on_path_to(a, d) {
                    None => assert!(a == d || !is_anc),
                    Some(child) => {
                        assert_eq!(tree.parent(child), Some(a));
                        assert!(tree.is_ancestor(child, d));
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "not form a single connected tree")]
    fn disconnected_input_panics() {
        // Node 2 is isolated.
        RootedTree::new(3, &[(0, 1)], 0);
    }
}

/// A rooted tree navigated by heavy-path chains instead of binary lifting: each node's `parent` (the
/// root is its own), the `head` of its chain and its `heavy` child (`u32::MAX` at a leaf). That is
/// 12 B per node next to the depths, where binary lifting stores `4 (log2 depth + 1)` B. The
/// block-cut tree of a pangenome is very nearly a path (`log2 depth` ≈ 20, almost every query on one
/// chain), so this is several times smaller and an LCA takes a couple of chain hops instead of ~20
/// random reads.
pub struct ChainTreeArrays {
    pub parent: Vec<u32>,
    pub head: Vec<u32>,
    pub heavy: Vec<u32>,
}

impl ChainTreeArrays {
    /// The chains of `tree` (heavy child = the child with the largest subtree, ties to the lowest id).
    pub fn build<T: TreeAccess + ?Sized>(tree: &T) -> Self {
        let depth = tree.depth_slice();
        let n = depth.len();
        let root = tree.root();
        let parent: Vec<u32> = (0..n)
            .map(|v| tree.parent(v).map_or(v as u32, |p| p as u32))
            .collect();
        // Nodes by ascending depth (a stable counting sort), so sizes accumulate bottom-up and heads
        // propagate top-down in one pass each.
        let max_depth = depth.iter().copied().max().unwrap_or(0) as usize;
        let mut start = vec![0usize; max_depth + 2];
        for &d in depth {
            start[d as usize + 1] += 1;
        }
        for d in 0..=max_depth {
            start[d + 1] += start[d];
        }
        let mut order = vec![0u32; n];
        for (v, &d) in depth.iter().enumerate() {
            order[start[d as usize]] = v as u32;
            start[d as usize] += 1;
        }
        let mut size = vec![1u32; n];
        for &v in order.iter().rev() {
            if v as usize != root {
                size[parent[v as usize] as usize] += size[v as usize];
            }
        }
        let mut heavy = vec![u32::MAX; n];
        let mut heavy_size = vec![0u32; n];
        for &v in &order {
            if v as usize != root {
                let p = parent[v as usize] as usize;
                if size[v as usize] > heavy_size[p] {
                    heavy_size[p] = size[v as usize];
                    heavy[p] = v;
                }
            }
        }
        let mut head = vec![0u32; n];
        for &v in &order {
            let p = parent[v as usize];
            head[v as usize] = if v as usize == root || heavy[p as usize] != v {
                v
            } else {
                head[p as usize]
            };
        }
        Self {
            parent,
            head,
            heavy,
        }
    }
}

/// An image-backed [`ChainTreeArrays`] with the depths: parent, LCA and child-on-path queries.
#[derive(Clone, Copy, Debug)]
pub struct ChainTreeView<'a> {
    pub depth: &'a [u32],
    pub parent: &'a [u32],
    pub head: &'a [u32],
    pub heavy: &'a [u32],
}

impl ChainTreeView<'_> {
    /// The parent of `v`, or `None` if `v` is the root.
    #[inline]
    pub fn parent(&self, v: usize) -> Option<usize> {
        let p = self.parent[v] as usize;
        (p != v).then_some(p)
    }

    /// The lowest common ancestor of `a` and `b`.
    #[inline]
    pub fn lca(&self, mut a: usize, mut b: usize) -> usize {
        while self.head[a] != self.head[b] {
            let (head_a, head_b) = (self.head[a] as usize, self.head[b] as usize);
            if self.depth[head_a] > self.depth[head_b] {
                a = self.parent[head_a] as usize;
            } else {
                b = self.parent[head_b] as usize;
            }
        }
        if self.depth[a] <= self.depth[b] { a } else { b }
    }

    /// The child of `ancestor` on the path towards `descendant`, or `None` if `descendant` is not a
    /// strict descendant of `ancestor`.
    pub fn child_on_path_to(&self, ancestor: usize, descendant: usize) -> Option<usize> {
        if ancestor == descendant {
            return None;
        }
        let mut d = descendant;
        loop {
            let head = self.head[d] as usize;
            if head == self.head[ancestor] as usize {
                // Same chain: `ancestor`'s heavy child leads down to `d` iff `d` is below it.
                return (self.depth[d] > self.depth[ancestor])
                    .then(|| self.heavy[ancestor] as usize);
            }
            let p = self.parent[head] as usize;
            if p == ancestor {
                return Some(head);
            }
            if p == head {
                return None; // reached the root without meeting `ancestor`
            }
            d = p;
        }
    }
}

#[cfg(test)]
mod chain_tree_tests {
    use super::*;

    /// Deterministic xorshift.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Every `parent`, `lca` and `child_on_path_to` answer of the chain view equals the binary-lifting
    /// tree's, on random trees from bushy to path-like (the block-cut tree shape) — every pair.
    #[test]
    fn chain_tree_matches_binary_lifting() {
        let mut state = 0x5eed_c4a1_17ee_0001u64;
        for round in 0..300 {
            let n: usize = 1 + (round % 60);
            // Parent of node v (v >= 1) within the last `window` nodes: window 1 = a path.
            let window = 1 + (next(&mut state) % 8) as usize;
            let edges: Vec<(usize, usize)> = (1..n)
                .map(|v| {
                    let lo = v.saturating_sub(window);
                    (lo + (next(&mut state) as usize) % (v - lo), v)
                })
                .collect();
            let root = 0;
            let tree = RootedTree::new(n, &edges, root);
            let view = tree.as_view();
            let chains = ChainTreeArrays::build(&view);
            let chain = ChainTreeView {
                depth: view.depth,
                parent: &chains.parent,
                head: &chains.head,
                heavy: &chains.heavy,
            };
            for a in 0..n {
                assert_eq!(chain.parent(a), view.parent(a), "parent({a}) n={n}");
                for b in 0..n {
                    assert_eq!(chain.lca(a, b), view.lca(a, b), "lca({a},{b}) n={n}");
                    assert_eq!(
                        chain.child_on_path_to(a, b),
                        view.child_on_path_to(a, b),
                        "child_on_path_to({a},{b}) n={n}"
                    );
                }
            }
        }
    }
}
