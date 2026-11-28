// SPDX-License-Identifier: MIT OR Apache-2.0

//! The recorded calls that are ill-posed: CuTe DSL answers them, but the
//! answer is not the layout the operation is defined to give, so cute-rs
//! refuses them. Each class is recognised from the call's inputs alone (never
//! from either answer), by walking the static algebra as C++ CuTe 4.8.1 does.
//! The classes of the atom questions are recognised in [`crate::atoms::ill_posed`],
//! which reads the sizes of atoms and tiles it needs from cute-rs.
//! A call counts as ill-posed only when cute-rs refuses it; where cute-rs
//! answers, it is compared as any other call.

use std::panic::{self, AssertUnwindSafe};

use cute::{Layout, raked_product};

/// Flat static modes `(extent, stride)`.
type Modes = Vec<(usize, usize)>;

/// Why a call is ill-posed, as the report names the class.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Class {
    /// The layout whose complement the operation takes is not injective.
    ///
    /// The complement walks the filtered modes in increasing stride order;
    /// when a stride is smaller than where the modes of smaller strides end,
    /// two coordinates meet and the gap before that stride has extent 0. C++
    /// CuTe refuses this with `static_assert(not is_constant<0, ...>,
    /// "Non-injective Layout detected in complement.")` in
    /// `detail::complement_prefix` (include/cute/layout.hpp). CuTe DSL does
    /// not check it and returns extents of 0 (`complement((2,2,3,3):(6,4,1,2),
    /// 24)` is `(0,0,0,2):(3,6,8,12)`), and the products and divides built on
    /// such a complement go on from there.
    NonInjective,
    /// A composition for which C++ CuTe's `composition_impl` builds a
    /// layout that is not the composition: some index of the right-hand side
    /// gets an offset other than `lhs(rhs(i))`.
    ///
    /// C++ CuTe's "Stride Divisibility Condition" in `composition_impl`
    /// (`rest_stride % curr_shape == 0 or rest_stride < curr_shape`) admits
    /// a stride smaller than a mode that does not divide it, and the walk
    /// then continues into the next mode as if the stride had divided the
    /// extent, or the modes of the right-hand side together step past the
    /// mode: `(3,2,2,5):(1,6,2,6) ∘ 8:2` gives `(2,2,2):(2,6,2)`,
    /// whose element 2 is 6, where the left-hand side sends 4 to 7. No static
    /// assertion in C++ CuTe refuses it (its "Shape Divisibility Condition"
    /// catches only the cases where the extents do not divide); CuTe DSL
    /// returns the same wrong layout. cute-rs refuses it (see
    /// `compose_static` in src/layout/layout.rs).
    InexactComposition,
    /// An upcast by `n` of a static stride that neither divides `n` nor is a
    /// multiple of it: C++ CuTe's `static_assert(Stride::value % N == 0 or N
    /// % Stride::value == 0, "Divisibility condition")` in `upcast`. CuTe DSL
    /// rounds the stride up (`recast_layout(16, 4, (4):(3))` is `(2):(1)`),
    /// which no longer addresses the same elements.
    RecastIndivisible,
    /// A downcast of a layout with no stride-1 leaf: C++ CuTe's
    /// `CUTE_STATIC_ASSERT(has_int1<Stride>::value, "Downcast requires
    /// adjacent elements")` in `downcast`. CuTe DSL multiplies every stride,
    /// which leaves the narrower elements between them unaddressed.
    RecastNoUnitStride,
    /// A tiled copy whose threads each hold a number of values that is not a
    /// multiple of what one copy of its atom moves: C++ CuTe's
    /// `CUTE_STATIC_ASSERT_V(TiledNumVal{} % AtomNumVal{} == Int<0>{}, "TiledCopy uses too few vals
    /// for selected CopyAtom")` in `TiledCopy` (include/cute/atom/copy_atom.hpp). CuTe DSL builds
    /// the tiled copy, and its partitions repeat values with stride 0.
    CopyTooFewValues,
    /// A cotiled copy whose atom layout reaches addresses the data layout does not hold: C++
    /// CuTe's `CUTE_STATIC_ASSERT_V(... , "The memory pointed to by AtomTVLayout does not exist
    /// in the DataLayout.")` in `make_cotiled_copy`.
    CotiledOutsideData,
    /// `local_partition` by a thread layout that is not a bijection onto `[0, size)`: thread
    /// `i` has no coordinate, or several. C++ CuTe has no assertion; its `get_flat_coord`
    /// gives a coordinate that means something only for compact layouts.
    ThreadLayoutNotCompact,
    /// A warpgroup MMA with A in registers and MN-major: C++ CuTe's
    /// `static_assert(tnspA == GMMA::Major::K, "Register source operand A must have K major
    /// layout.")` (include/cute/arch/mma_sm90_gmma.hpp); `wgmma.mma_async` with A in registers
    /// has no transpose of A.
    RegisterAMnMajor,
    /// A tcgen05 Int8 MMA whose N is neither 8 nor a multiple of 16: C++ CuTe's
    /// `static_assert(N == 8 || ((N % 16 == 0) && (16 <= N) && (N <= 256)), ...)` in
    /// `SM100_MMA_S8_SS` (include/cute/arch/mma_sm100_umma.hpp).
    UmmaInt8N,
    /// The tensor-memory accumulator of an `M = 64` tcgen05 MMA tiled over a first rest mode
    /// of odd extent: C++ CuTe's `tmem_frg_1sm` (include/cute/atom/mma_traits_sm100_frag.hpp)
    /// tiles the atom `((16,4),N):((1,32),128)` into its 16-lane gaps first, and the composition
    /// in that `tiled_product` fails its shape divisibility assertion when the first tile count
    /// is not even. CuTe DSL places such tiles in columns, or prints `x` for strides it lacks.
    TmemLaneGap,
}

impl Class {
    pub const ALL: [Class; 10] = [
        Class::NonInjective,
        Class::InexactComposition,
        Class::RecastIndivisible,
        Class::RecastNoUnitStride,
        Class::CopyTooFewValues,
        Class::CotiledOutsideData,
        Class::ThreadLayoutNotCompact,
        Class::RegisterAMnMajor,
        Class::UmmaInt8N,
        Class::TmemLaneGap,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Class::NonInjective => "complement of a non-injective layout",
            Class::InexactComposition => "inexact composition (indivisible stride)",
            Class::RecastIndivisible => "upcast of an indivisible stride",
            Class::RecastNoUnitStride => "downcast with no stride-1 leaf",
            Class::CopyTooFewValues => "tiled copy of too few values for its atom",
            Class::CotiledOutsideData => "cotiled copy outside its data layout",
            Class::ThreadLayoutNotCompact => "local_partition by a non-compact thread layout",
            Class::RegisterAMnMajor => "warpgroup MMA with an MN-major A in registers",
            Class::UmmaInt8N => "tcgen05 Int8 MMA of an N C++ CuTe refuses",
            Class::TmemLaneGap => "M = 64 TMEM accumulator tiles in its lane gaps",
        }
    }
}

fn modes(l: &Layout) -> Modes {
    l.flat_modes()
}

/// C++ CuTe's `coalesce` of flat static modes: size-1 modes dropped,
/// contiguous modes merged.
fn coalesce(m: &[(usize, usize)]) -> Modes {
    let mut out: Modes = Vec::new();
    for &(s, d) in m {
        if s == 1 {
            continue;
        }
        if let Some(last) = out.last_mut()
            && last.0 * last.1 == d
        {
            last.0 *= s;
            continue;
        }
        out.push((s, d));
    }
    out
}

/// C++ CuTe's `filter`, flat: stride-0 modes dropped, then coalesced.
fn filter(l: &Layout) -> Modes {
    coalesce(&modes(l).into_iter().filter(|&(_, d)| d != 0).collect::<Modes>())
}

/// Whether `complement` of `l` meets a gap of extent 0 (see
/// [`Class::NonInjective`]).
fn non_injective(l: &Layout) -> bool {
    let mut m = filter(l);
    m.sort_by_key(|&(_, d)| d);
    let mut end = 1;
    for (s, d) in m {
        if d < end {
            return true;
        }
        end = s * d;
    }
    false
}

/// The gaps of an injective `l` and where it ends (C++ CuTe's
/// `complement_prefix`).
fn prefix(l: &Layout) -> (Modes, usize) {
    let mut m = filter(l);
    m.sort_by_key(|&(_, d)| d);
    let (mut gaps, mut end) = (Vec::new(), 1);
    for (s, d) in m {
        gaps.push((d / end, end));
        end = s * d;
    }
    (gaps, end)
}

/// C++ CuTe's `complement(l, cotarget)` for a static cotarget of the extents
/// `cotarget`, flat and coalesced.
fn complement(l: &Layout, cotarget: &[usize]) -> Modes {
    let (mut out, end) = prefix(l);
    let (mut span, mut stride) = (end, end);
    for &c in cotarget {
        let rest = c.div_ceil(span);
        out.push((rest, stride));
        stride *= rest;
        span = span.div_ceil(c);
    }
    coalesce(&out)
}

/// C++ CuTe's `detail::coalesce_x` of flat static modes: as [`coalesce`],
/// but a last mode of extent 1 is kept with extent 2.
fn coalesce_x(m: &[(usize, usize)]) -> Modes {
    let Some((&(ls, ld), rest)) = m.split_last() else { return vec![(1, 0)] };
    let mut out = vec![(if ls == 1 { 2 } else { ls }, ld)];
    for &(s, d) in rest.iter().rev() {
        if s == 1 {
            continue;
        }
        if s * d == out[0].1 {
            out[0] = (s * out[0].0, d);
            continue;
        }
        out.insert(0, (s, d));
    }
    out
}

/// C++ CuTe's `composition_impl` of the flat, `coalesce_x`-ed `lhs` with the
/// right-hand leaf `s:d`, as flat modes, or `None` where C++ CuTe's static
/// assertions refuse it.
fn compose_leaf(lhs: &[(usize, usize)], s: usize, d: usize) -> Option<Modes> {
    let ((_, last), init) = lhs.split_last()?;
    if d == 0 {
        return Some(vec![(s, 0)]);
    }
    let (mut rest_shape, mut rest_stride, mut out) = (s, d, Vec::new());
    for &(cs, cd) in init {
        if !(rest_stride.is_multiple_of(cs) || rest_stride < cs) {
            return None;
        }
        let next_shape = cs.div_ceil(rest_stride);
        let next_stride = rest_stride.div_ceil(cs);
        if next_shape == 1 || rest_shape == 1 {
            rest_stride = next_stride;
            continue;
        }
        let new_shape = next_shape.min(rest_shape);
        if !rest_shape.is_multiple_of(new_shape) {
            return None;
        }
        out.push((new_shape, rest_stride * cd));
        rest_shape /= new_shape;
        rest_stride = next_stride;
    }
    if out.is_empty() || rest_shape != 1 {
        out.push((rest_shape, rest_stride * last));
    }
    Some(out)
}

/// The offset of index `i` in flat modes, the last mode taking whatever is
/// left (so a layout goes on past its domain, as composition takes it).
fn offset(m: &[(usize, usize)], mut i: usize) -> usize {
    let mut off = 0;
    for (k, &(s, d)) in m.iter().enumerate() {
        let c = if k + 1 == m.len() { i } else { i % s };
        off += c * d;
        if k + 1 < m.len() {
            i /= s;
        }
    }
    off
}

/// Whether C++ CuTe's composition of `lhs` with the right-hand leaves `rhs`
/// is defined (passes its static assertions) but is not the composition:
/// some index `i` of the right-hand side, sent to `lhs(rhs(i))` by the
/// definition, gets another offset (see [`Class::InexactComposition`]).
fn inexact_composition(lhs: &Layout, rhs: &[(usize, usize)]) -> bool {
    let lhs = coalesce_x(&modes(lhs));
    let Some(parts) = rhs.iter().map(|&(s, d)| compose_leaf(&lhs, s, d)).collect::<Option<Vec<Modes>>>() else {
        return false;
    };
    let size: usize = rhs.iter().map(|&(s, _)| s).product();
    (0..size).any(|i| {
        let (mut rest, mut got, mut at) = (i, 0, 0);
        for (&(s, d), part) in rhs.iter().zip(&parts) {
            let c = rest % s;
            rest /= s;
            got += offset(part, c);
            at += c * d;
        }
        got != offset(&lhs, at)
    })
}

/// `logical_divide(l, t)` is ill-posed: `t` is not injective, or the
/// composition of `l` with `(t, complement(t, shape(coalesce(l))))` is
/// inexact.
fn divide(l: &Layout, t: &Layout) -> Option<Class> {
    if non_injective(t) {
        return Some(Class::NonInjective);
    }
    let shape: Vec<usize> = modes(&l.coalesce()).into_iter().map(|(s, _)| s).collect();
    let mut rhs = modes(t);
    rhs.extend(complement(t, &shape));
    inexact_composition(l, &rhs).then_some(Class::InexactComposition)
}

/// A product of `block` by `tiler` is ill-posed: `block` is not injective,
/// or the composition of `complement(block, size(block) * cosize(tiler))`
/// with `tiler` is inexact. (The `1:0` modes blocked and raked products pad
/// with change neither.)
fn product(block: &Layout, tiler: &Layout) -> Option<Class> {
    if non_injective(block) {
        return Some(Class::NonInjective);
    }
    let comp = complement(block, &[block.size() * tiler.cosize()]);
    let comp = Layout::from_flat(if comp.is_empty() { &[(1, 0)] } else { &comp });
    inexact_composition(&comp, &modes(tiler)).then_some(Class::InexactComposition)
}

fn layout(s: &str) -> Layout {
    s.parse().unwrap_or_else(|e| panic!("unreadable layout {s}: {e}"))
}

/// The class of an ill-posed call, or `None`.
pub fn class(op: &str, a: &[&str]) -> Option<Class> {
    match op {
        "complement" => non_injective(&layout(a[0])).then_some(Class::NonInjective),
        "composition" => inexact_composition(&layout(a[0]), &modes(&layout(a[1]))).then_some(Class::InexactComposition),
        "logical_divide" => divide(&layout(a[0]), &layout(a[1])),
        "zipped_divide" | "tiled_divide" | "flat_divide" => {
            let l = layout(a[0]);
            a[1].split(';').enumerate().find_map(|(i, t)| divide(&l.mode(i), &layout(t)))
        }
        "logical_product" | "blocked_product" | "raked_product" | "zipped_product" | "tiled_product"
        | "flat_product" => product(&layout(a[0]), &layout(a[1])),
        "make_layout_tv" => {
            let (thr, val) = (layout(a[0]), layout(a[1]));
            product(&thr, &val).or_else(|| {
                // `right_inverse(raked_product(thr, val)) ∘ (size(thr), size(val))`.
                let mn = panic::catch_unwind(AssertUnwindSafe(|| raked_product(&thr, &val))).ok()?;
                let tv = [(thr.size(), 1), (val.size(), thr.size())];
                inexact_composition(&mn.right_inverse(), &tv).then_some(Class::InexactComposition)
            })
        }
        "max_common_vector" | "max_common_layout" => {
            let (x, y) = (layout(a[0]), layout(a[1]));
            inexact_composition(&x, &modes(&y.right_inverse())).then_some(Class::InexactComposition)
        }
        "recast_layout" => {
            let (new, old): (usize, usize) = (a[0].parse().ok()?, a[1].parse().ok()?);
            let strides: Vec<usize> = modes(&layout(a[2])).into_iter().map(|(_, d)| d).collect();
            if new > old {
                let n = new / old;
                strides
                    .iter()
                    .any(|&d| d != 0 && !d.is_multiple_of(n) && !n.is_multiple_of(d))
                    .then_some(Class::RecastIndivisible)
            } else if new < old {
                (!strides.contains(&1)).then_some(Class::RecastNoUnitStride)
            } else {
                None
            }
        }
        _ => None,
    }
}
