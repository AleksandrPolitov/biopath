//! The closed-form ("prefix") path-product backend of the block-cut lift (default when it applies;
//! the chain segment trees of [`crate::chain_segtree`] remain the general backend).
//!
//! # The closed form
//!
//! Let `u_c` be the orientation of cut node `c` that leaves it towards its parent block. For a step
//! `c -> p` (through `c`'s parent block) with matrix `M[o][q] = d(c^o -> p^q)` the step is
//! **regular** when
//!
//! ```text
//! M[u_c][u_p]   = D                              (the direct cost)
//! M[u_c][¬u_p]  = D + full[p](u_p -> ¬u_p)        (pass p upward, then its cheapest flip)
//! M[¬u_c][u_p]  = full[c](¬u_c -> u_c) + D        (flip at c first)
//! M[¬u_c][¬u_p] = full[c](¬u_c -> u_c) + D + full[p](u_p -> ¬u_p)
//! ```
//!
//! which holds whenever the block touches each of its cut nodes on one side only: a cut node
//! separates, so a route to `p^{¬u_p}` must first pass `p^{u_p}`, and `full[p]` is by definition the
//! cheapest `p^{u_p} -> p^{¬u_p}` route. Regular steps compose: the product of a run of regular steps
//! is again of this shape with `D` the sum of the steps' `D`s, so one prefix `P(c) = Σ D` over the
//! root path answers any run in O(1) from `P`, the two `full` closures and the two orientation bits.
//!
//! A step whose matrix is not of this shape (a block with edges on both sides of a cut node — a
//! "lane swap") keeps its explicit `2x2` and is composed where the path crosses it, found through the
//! nearest exception above (per chain, see [`crate::chain_nav::ChainNav::irr_above`]); an
//! all-infinite step is "blocked". The build counts both and keeps the segment trees instead when
//! they are not rare (see [`choose_prefix`]).
//!
//! The down product is the reversal identity applied to the up product:
//! `d(y^q -> x^o) = d(x^{¬o} -> y^{¬q}) + len(y) - len(x)`.
//!
//! The per-cut data (`P`, the orientation bit, the `full` off-diagonals, the node length) live in
//! the topology's cut records ([`crate::chain_nav::CutRec`], filled by the lift); this module
//! persists only the irregular steps' matrices.

use bidirected_adjacency_array::index::GraphIndexInteger;

use crate::{
    chain_nav::{ChainNav, CutRec, cut_flags},
    image::{Image, Padded, Sec, image_record},
    min_plus::{Mat2, MinPlus},
    path::GfaPathLength,
    sections,
};

/// "No such cut" in the per-cut pointer arrays.
pub const NONE: u32 = u32::MAX;

/// `orient` bits: the cut's up-orientation, and whether its own step is irregular / blocked.
const ORIENT_BIT: u8 = 1;
const IRREGULAR_BIT: u8 = 2;
const BLOCKED_BIT: u8 = 4;

/// The explicit matrix of an irregular step (`cut -> parent cut`); the list is sorted by cut.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IrregularStep<IndexType: GraphIndexInteger> {
    /// As wide as the index word, so a 64-bit matrix leaves no padding after it.
    pub cut: Padded<u32, IndexType>,
    pub mat: Mat2<IndexType>,
}
image_record!(impl<I> IrregularStep<I> = [Padded<u32, I>, Mat2<I>]);

/// Step classification counts (the gate).
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefixStats {
    pub regular: usize,
    pub irregular: usize,
    pub blocked: usize,
}

impl PrefixStats {
    pub fn steps(&self) -> usize {
        self.regular + self.irregular + self.blocked
    }
}

/// Which path-product backend the block-cut lift uses (a build option,
/// [`LiftOptions::backend`](crate::block_cut_lift::LiftOptions::backend)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LiftBackend {
    /// The prefix lift when at most [`MAX_EXCEPTION_FRACTION`] of the steps are exceptions and
    /// its prefix fits the index word, else the chain segment trees.
    #[default]
    Auto,
    /// The prefix lift whenever its prefix fits the index word.
    Prefix,
    /// Always the chain segment trees.
    Segtree,
}

/// Exceptions (irregular + blocked steps) above this share of the steps keep the segment trees:
/// every exception on a query path costs two extra matrix products.
pub const MAX_EXCEPTION_FRACTION: f64 = 0.02;

/// Whether `backend` puts the prefix lift on a lift with these step counts (`overflowed`: its
/// prefix does not fit the index word).
pub fn choose_prefix(backend: LiftBackend, stats: &PrefixStats, overflowed: bool) -> bool {
    match backend {
        LiftBackend::Prefix => !overflowed,
        LiftBackend::Segtree => false,
        LiftBackend::Auto => {
            !overflowed
                && (stats.irregular + stats.blocked) as f64
                    <= MAX_EXCEPTION_FRACTION * stats.steps().max(1) as f64
        }
    }
}

/// The owned (build-time) prefix backend.
pub struct PrefixLift<IndexType: GraphIndexInteger> {
    /// Per cut: the sum of the regular steps' direct costs from the cut up to its forest root
    /// (exceptions contribute `0`; a run never spans one).
    p: Vec<GfaPathLength<IndexType>>,
    /// Per cut: [`ORIENT_BIT`] | [`IRREGULAR_BIT`] | [`BLOCKED_BIT`].
    orient: Vec<u8>,
    /// Per cut: the nearest ancestor-or-self cut whose step is an exception (`NONE` if none);
    /// empty when the lift has no exceptions at all.
    irr_above: Vec<u32>,
    /// The irregular steps' matrices, sorted by cut.
    irr_mats: Vec<IrregularStep<IndexType>>,
}

impl<IndexType: GraphIndexInteger> PrefixLift<IndexType> {
    /// Classify every step and build the prefix. `cut_order` lists parents before children,
    /// `step_up[c]` is the matrix of `c -> parent(c)` and `full[c]` the whole-graph flip closure.
    /// Returns `None` (with the stats) when a prefix sum overflows the index word.
    pub fn build(
        number_of_cuts: usize,
        cut_parent: &[Option<usize>],
        cut_order: &[usize],
        step_up: &[Mat2<IndexType>],
        full: &[Mat2<IndexType>],
    ) -> (Option<Self>, PrefixStats) {
        let steps = StepClasses::of(number_of_cuts, cut_parent, cut_order, step_up, full);
        let max = crate::min_plus::wide::finite_max::<IndexType>();
        let Some((p, irr_above)) = steps.prefix_sums(cut_parent, cut_order, max) else {
            return (None, steps.stats);
        };
        let orient: Vec<u8> = (steps.bits.iter().zip(&steps.up))
            .map(|(&bits, &up)| bits | if up == 1 { ORIENT_BIT } else { 0 })
            .collect();
        let irr_mats: Vec<IrregularStep<IndexType>> = (0..number_of_cuts)
            .filter(|&c| steps.bits[c] & IRREGULAR_BIT != 0)
            .map(|c| IrregularStep {
                cut: Padded::new(c as u32),
                mat: step_up[c],
            })
            .collect();
        let lift = Self {
            p: p.iter()
                .map(|&v| GfaPathLength::from_usize(v as usize))
                .collect(),
            orient,
            irr_above,
            irr_mats,
        };
        (Some(lift), steps.stats)
    }

    /// Whether `cut`'s own step (to its parent cut) is regular, i.e. of the closed-form shape.
    #[inline]
    pub fn is_regular_step(&self, cut: usize) -> bool {
        self.orient[cut] & (IRREGULAR_BIT | BLOCKED_BIT) == 0
    }

    /// The prefix sum of `cut`.
    #[inline]
    pub fn p(&self, cut: usize) -> GfaPathLength<IndexType> {
        self.p[cut]
    }

    /// `cut`'s bits of [`cut_flags`] (orientation, irregular, blocked).
    #[inline]
    pub fn record_flags(&self, cut: usize) -> u32 {
        let o = self.orient[cut];
        (if o & ORIENT_BIT != 0 {
            cut_flags::ORIENT
        } else {
            0
        }) | (if o & IRREGULAR_BIT != 0 {
            cut_flags::IRREGULAR
        } else {
            0
        }) | (if o & BLOCKED_BIT != 0 {
            cut_flags::BLOCKED
        } else {
            0
        })
    }

    /// Whether `cut`'s own step is an exception (irregular or blocked).
    #[inline]
    pub fn is_exception(&self, cut: usize) -> bool {
        !self.is_regular_step(cut)
    }

    /// The nearest ancestor-or-self cut of `cut` whose step is an exception (`NONE` if none).
    #[inline]
    pub fn irr_above(&self, cut: usize) -> u32 {
        if self.irr_above.is_empty() {
            NONE
        } else {
            self.irr_above[cut]
        }
    }

    /// Write the backend into `image`: only the irregular steps' matrices (the per-cut data go into
    /// the cut records).
    pub fn write(&self, image: &Image) -> PrefixSections<IndexType> {
        PrefixSections {
            irr_mats: image.push_section(sections::LIFT_PFX_IRR_MATS, &self.irr_mats),
        }
    }
}

/// Every step classified (indexed by the cut whose step it is): the up-orientations found so far
/// (`u8::MAX` = unset), the exception bits, a regular step's direct cost, and the counts.
struct StepClasses {
    up: Vec<u8>,
    bits: Vec<u8>,
    direct: Vec<u64>,
    stats: PrefixStats,
}

/// One step's class (see the module docs).
enum StepClass<IndexType: GraphIndexInteger> {
    /// All four entries infinite.
    Blocked,
    /// Of the closed-form shape with up-orientations `(u_c, u_p)` and direct cost `direct`.
    Regular {
        u_c: usize,
        u_p: usize,
        direct: MinPlus<IndexType>,
    },
    Irregular,
}

impl StepClasses {
    /// Classify the step of every non-root cut, parents first: a parent's up-orientation, once a
    /// regular step fixed it, constrains its children's steps.
    fn of<IndexType: GraphIndexInteger>(
        n: usize,
        cut_parent: &[Option<usize>],
        cut_order: &[usize],
        step_up: &[Mat2<IndexType>],
        full: &[Mat2<IndexType>],
    ) -> Self {
        let mut steps = Self {
            up: vec![u8::MAX; n],
            bits: vec![0u8; n],
            direct: vec![0u64; n],
            stats: PrefixStats::default(),
        };
        for &c in cut_order {
            let Some(p) = cut_parent[c] else { continue };
            match classify_step(&step_up[c], &full[c], &full[p], steps.up[p]) {
                StepClass::Blocked => {
                    steps.bits[c] |= BLOCKED_BIT;
                    steps.stats.blocked += 1;
                }
                StepClass::Regular { u_c, u_p, direct } => {
                    steps.up[c] = u_c as u8;
                    if steps.up[p] == u8::MAX {
                        steps.up[p] = u_p as u8;
                    }
                    steps.direct[c] = direct
                        .into_option()
                        .expect("a regular step has a finite direct cost")
                        .into_usize() as u64;
                    steps.stats.regular += 1;
                }
                StepClass::Irregular => {
                    steps.bits[c] |= IRREGULAR_BIT;
                    steps.stats.irregular += 1;
                }
            }
        }
        steps
    }

    /// The prefix sums `P` (parents first) and the nearest exception at or above every cut (empty
    /// when no step is an exception); `None` when a sum exceeds `max`.
    fn prefix_sums(
        &self,
        cut_parent: &[Option<usize>],
        cut_order: &[usize],
        max: u64,
    ) -> Option<(Vec<u64>, Vec<u32>)> {
        let n = self.bits.len();
        let mut p = vec![0u64; n];
        let mut irr_above = vec![NONE; n];
        let mut any_exception = false;
        for &c in cut_order {
            let exception = self.bits[c] & (IRREGULAR_BIT | BLOCKED_BIT) != 0;
            any_exception |= exception;
            let above = match cut_parent[c] {
                Some(parent) => {
                    p[c] = p[parent] + self.direct[c];
                    if p[c] > max {
                        return None;
                    }
                    irr_above[parent]
                }
                None => NONE,
            };
            irr_above[c] = if exception { c as u32 } else { above };
        }
        if !any_exception {
            irr_above = Vec::new();
        }
        Some((p, irr_above))
    }
}

/// The class of the step `c -> p` with matrix `m`, given the flip closures of both cuts and `p`'s
/// up-orientation if already fixed (`u8::MAX` = free).
fn classify_step<IndexType: GraphIndexInteger>(
    m: &Mat2<IndexType>,
    full_c: &Mat2<IndexType>,
    full_p: &Mat2<IndexType>,
    up_p: u8,
) -> StepClass<IndexType> {
    if (0..2).all(|o| (0..2).all(|q| m.get(o, q).is_infinite())) {
        return StepClass::Blocked;
    }
    let fits = |u_c: usize, u_p: usize| {
        let direct = m.get(u_c, u_p);
        let fc = full_c.get(1 - u_c, u_c);
        let fp = full_p.get(u_p, 1 - u_p);
        !direct.is_infinite()
            && m.get(u_c, 1 - u_p) == direct.otimes(fp)
            && m.get(1 - u_c, u_p) == fc.otimes(direct)
            && m.get(1 - u_c, 1 - u_p) == fc.otimes(direct).otimes(fp)
    };
    let found = if up_p != u8::MAX {
        let u_p = up_p as usize;
        (0..2).find(|&u_c| fits(u_c, u_p)).map(|u_c| (u_c, u_p))
    } else {
        (0..2)
            .flat_map(|u_c| (0..2).map(move |u_p| (u_c, u_p)))
            .find(|&(u_c, u_p)| fits(u_c, u_p))
    };
    match found {
        Some((u_c, u_p)) => StepClass::Regular {
            u_c,
            u_p,
            direct: m.get(u_c, u_p),
        },
        None => StepClass::Irregular,
    }
}

/// Typed handles to a persisted [`PrefixLift`].
#[derive(Clone, Copy, Debug)]
pub struct PrefixSections<IndexType: GraphIndexInteger> {
    irr_mats: Sec<IrregularStep<IndexType>>,
}

impl<IndexType: GraphIndexInteger> PrefixSections<IndexType> {
    pub fn from_image(image: &Image) -> std::io::Result<Self> {
        Ok(Self {
            irr_mats: image.require(sections::LIFT_PFX_IRR_MATS, "lift.pfx_irr_mats")?,
        })
    }

    /// Resolve against the image, over the lift's cut records and navigation.
    pub fn view<'a>(
        &self,
        image: &'a Image,
        cuts: &'a [CutRec<IndexType>],
        nav: ChainNav<'a, IndexType>,
    ) -> PrefixLiftView<'a, IndexType> {
        PrefixLiftView {
            irr_mats: image.get(self.irr_mats),
            cuts,
            nav,
        }
    }
}

/// The image-backed prefix backend: the irregular steps plus the cut records and navigation it
/// reads `P`, the orientation bits, the flip closures and the node lengths from.
#[derive(Clone, Copy)]
pub struct PrefixLiftView<'a, IndexType: GraphIndexInteger> {
    irr_mats: &'a [IrregularStep<IndexType>],
    cuts: &'a [CutRec<IndexType>],
    nav: ChainNav<'a, IndexType>,
}

/// An entry of a cut's flip closure `full[c]` from its record (`0` on the diagonal).
#[inline]
pub fn full_entry<IndexType: GraphIndexInteger>(
    rec: &CutRec<IndexType>,
    i: usize,
    j: usize,
) -> MinPlus<IndexType> {
    if i == j {
        MinPlus::zero()
    } else if i == 0 {
        rec.f01
    } else {
        rec.f10
    }
}

impl<'a, IndexType: GraphIndexInteger> PrefixLiftView<'a, IndexType> {
    /// Whether `cut`'s own step is regular (see [`PrefixLift::is_regular_step`]).
    #[inline]
    pub fn is_regular_step(&self, cut: usize) -> bool {
        self.cuts[cut].flags.get() & (cut_flags::IRREGULAR | cut_flags::BLOCKED) == 0
    }

    /// The step matrix `d(cut^o -> parent^q)` of a regular step (one closed-form run).
    pub fn step_up_matrix(&self, cut: usize) -> Mat2<IndexType> {
        debug_assert!(self.is_regular_step(cut));
        let parent = self.nav.parent_cut(cut).expect("a step has a parent cut");
        self.closed(cut, parent)
    }

    /// The reversed step matrix `d(parent^q -> cut^o)` of a regular step.
    pub fn step_down_matrix(&self, cut: usize) -> Mat2<IndexType> {
        debug_assert!(self.is_regular_step(cut));
        let parent = self.nav.parent_cut(cut).expect("a step has a parent cut");
        self.down_product(parent, cut)
    }

    /// The closed-form product over a run of regular steps from `a` up to its ancestor `b`.
    #[inline]
    fn closed(&self, a: usize, b: usize) -> Mat2<IndexType> {
        if a == b {
            return Mat2::identity();
        }
        let (ra, rb) = (&self.cuts[a], &self.cuts[b]);
        let ua = (ra.flags.get() & cut_flags::ORIENT) as usize;
        let ub = (rb.flags.get() & cut_flags::ORIENT) as usize;
        let direct = MinPlus::finite(GfaPathLength::from_usize(
            ra.p.into_usize() - rb.p.into_usize(),
        ));
        let fa = full_entry(ra, 1 - ua, ua);
        let fb = full_entry(rb, ub, 1 - ub);
        let mut m = Mat2::infinity();
        m.set(ua, ub, direct);
        m.set(ua, 1 - ub, direct.otimes(fb));
        m.set(1 - ua, ub, fa.otimes(direct));
        m.set(1 - ua, 1 - ub, fa.otimes(direct).otimes(fb));
        m
    }

    /// `d(from^o -> to^q)` for the cut ancestor `to` of `from` (the up product, deeper factor left).
    pub fn up_product(&self, from: usize, to: usize) -> Mat2<IndexType> {
        if self.nav.exc.is_empty() {
            return self.closed(from, to);
        }
        let mut m = Mat2::identity();
        let mut cur = from;
        let to_pos = self.nav.tpos_of_cut(to);
        loop {
            let z = self.nav.irr_above(cur);
            if z == NONE || self.nav.tpos_of_cut(z as usize) <= to_pos {
                return m.mul(&self.closed(cur, to));
            }
            let z = z as usize;
            if self.cuts[z].flags.get() & cut_flags::BLOCKED != 0 {
                return Mat2::infinity();
            }
            let i = self
                .irr_mats
                .binary_search_by_key(&(z as u32), |r| r.cut.get())
                .expect("an irregular step keeps its matrix");
            m = m.mul(&self.closed(cur, z)).mul(&self.irr_mats[i].mat);
            cur = self
                .nav
                .parent_cut(z)
                .expect("an exception step has a parent cut");
        }
    }

    /// `d(from^q -> to^o)` for the cut descendant `to` of `from` (the down product), by the reversal
    /// identity applied to the up product.
    pub fn down_product(&self, from: usize, to: usize) -> Mat2<IndexType> {
        let up = self.up_product(to, from);
        let shift = self.cuts[from].len.into_usize() as i64 - self.cuts[to].len.into_usize() as i64;
        let mut down = Mat2::infinity();
        for q in 0..2 {
            for o in 0..2 {
                if let Some(v) = up.get(1 - o, 1 - q).into_option() {
                    let value = v.into_usize() as i64 + shift;
                    debug_assert!(value >= 0, "a reversed distance is non-negative");
                    down.set(
                        q,
                        o,
                        MinPlus::finite(GfaPathLength::from_usize(value as usize)),
                    );
                }
            }
        }
        down
    }

    pub fn lift_up_vector(
        &self,
        from: usize,
        to: usize,
        vector: [MinPlus<IndexType>; 2],
    ) -> [MinPlus<IndexType>; 2] {
        self.up_product(from, to).left_multiply_vector(&vector)
    }

    pub fn lift_down_vector(
        &self,
        from: usize,
        to: usize,
        vector: [MinPlus<IndexType>; 2],
    ) -> [MinPlus<IndexType>; 2] {
        self.down_product(from, to).left_multiply_vector(&vector)
    }
}
