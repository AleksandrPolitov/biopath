//! Min-plus (tropical) arithmetic over path lengths: the engine's distance algebra, in one place.
//!
//! `⊕` is `min` (the cheaper of two alternative walks) and `⊗` is `+` (one walk after another).
//! `∞` (no walk) is the identity of `⊕` and absorbs under `⊗`; `0` (the empty walk) is the
//! identity of `⊗`. A matrix product composes two regions through the ports they share,
//! `(A ⊗ B)[i][j] = min_k A[i][k] + B[k][j]`; an elementwise min puts two regions over the same
//! ports side by side; the closure (Floyd–Warshall) gives the all-pairs distances of a region whose
//! walks may pass through its ports any number of times.
//!
//! # What the engine uses it for
//!
//! A bidirected walk enters and leaves every node in one of two orientations, so a distance between
//! two nodes is a bundle of orientation-indexed distances:
//!
//! - a `2 × 2` per pair of nodes ([`Mat2`]; index `0` = `+`, `1` = `−`; the row is the orientation
//!   the walk leaves the source in, the column the one it enters the target in): the block-cut
//!   lift's step and augmented matrices between cut nodes, and every cut node's flip closure;
//! - a `4 × 4` per two-pole network, over the boundary states `[(s,+), (s,−), (t,+), (t,−)]` of its
//!   two poles: the inside / outside pole-state matrices and the pole-lift transitions of an SPQR
//!   node ([`MinPlusMatrix`]`<_, 4>`, persisted as [`sparse::SparseMat4`]) and the networks of the
//!   S-node oracles ([`network::Mat4`]);
//! - a dense `2c × 2c` over a block's `c` cut nodes (the lift's `M_B` and `M_aug`), closed by
//!   [`close_dense`] and read pairwise by [`MinPlusMatrix::block_of`].
//!
//! # `∞` and the two representations
//!
//! A distance is one `IndexType` word, `IndexType::MAX` meaning `∞`: the value domain of
//! `GfaPathLength`, which reserves `MAX`. Both representations have these bits, in memory and in the
//! image, so a value crosses between them unchanged ([`MinPlus::from_word`]); they differ in `⊗`:
//!
//! - **typed** ([`MinPlus`], [`MinPlusMatrix`], [`Mat2`]): `∞` is a value matched explicitly, and a
//!   finite `⊗` is plain addition of path lengths, which must stay below the `∞` word. The DP build,
//!   the pole lift and the block-cut lift (whose `2 × 2` products are also on the query path).
//! - **raw words** ([`word`], [`network`]): `⊗` is `saturating_add`, so `∞` absorbs without a branch,
//!   and a sum beyond the representable distances becomes `∞`, the right answer there. The S-node
//!   networks, whose series composition [`network::merge`] runs for every range product of an
//!   S-node oracle's build and query.
//!
//! Neither is forced on the other: the typed form keeps the build's sums exact, the raw form keeps
//! the S-node kernels branch-free. Each operation is defined once per representation (typed / raw):
//!
//! - `∞`, `⊗`: [`MinPlus::infinity`], [`MinPlus::otimes`] / [`word::inf`], [`word::sat_add`];
//! - `⊕=`: [`MinPlus::min_assign`], [`MinPlusMatrix::relax`], and [`relax`] on an
//!   `Option<GfaPathLength>` (`None` = `∞`, the form a query's answer takes) / [`word::relax`];
//! - elementwise min: [`MinPlusMatrix::elementwise_min`] / [`word::min_entries`];
//! - identity: [`MinPlusMatrix::identity`] / the zero diagonal of [`network::close_network`];
//! - product: [`MinPlusMatrix::mul`] and the vector products / [`network::merge`] (series);
//! - closure: [`close_dense`], also behind [`MinPlusMatrix::closure`] / [`word::close_square`],
//!   also behind [`network::close4`].
//!
//! The block searches and the S-node ring form compute in `u64` with `u64::MAX` as `∞` ([`wide`]):
//! [`wide::widen`] and [`wide::narrow`] convert words exactly, [`MinPlus::from_wide_clamped`] and
//! [`MinPlus::to_wide`] typed values.
//!
//! # Files
//!
//! - `scalar.rs`: one distance, [`MinPlus`], and [`relax`].
//! - `matrix.rs`: the typed `N × N` matrices, [`MinPlusMatrix`] and [`Mat2`], their vector products
//!   and the dense closure [`close_dense`].
//! - [`sparse`]: the image form of the typed `4 × 4`s and its sparse vector product.
//! - [`word`]: raw words and their closure.
//! - [`network`]: the raw `4 × 4` two-pole networks, their series and parallel composition.
//! - [`wide`]: the `u64` working domain.

mod matrix;
pub mod network;
mod scalar;
pub mod sparse;
pub mod wide;
pub mod word;

pub use matrix::{Mat2, MinPlusMatrix, close_dense};
pub use scalar::{MinPlus, relax};
