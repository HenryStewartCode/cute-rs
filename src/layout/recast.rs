// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/layout.hpp (upcast, downcast and recast_layout of a layout),
//   include/cute/swizzle_layout.hpp (upcast, downcast and recast_layout of
//   a swizzle) and include/cute/layout_composed.hpp (upcast, downcast and
//   recast_layout of a composed layout),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Recasting layouts between element widths: CuTe's `recast_layout`.

use super::basis::StrideLeaf;
use super::index::IndexExpr;
use super::layout::Layout;
use super::swizzle::{ComposedLayout, Inner, Swizzle};
use super::tree::Tree;

/// Layouts that [`recast_layout`] takes: [`Layout`] and [`ComposedLayout`].
pub trait Recast: Sized {
    /// The same memory in elements `n` times wider.
    #[doc(hidden)]
    fn recast_up(&self, n: usize) -> Self;
    /// The same memory in elements `n` times narrower.
    #[doc(hidden)]
    fn recast_down(&self, n: usize) -> Self;
}

fn leaves(layout: &Layout) -> Vec<(IndexExpr, IndexExpr)> {
    let shape = layout.shape().flat_leaves();
    let stride = layout.stride().flat_leaves();
    shape
        .into_iter()
        .zip(stride)
        .map(|(s, d)| match d {
            StrideLeaf::Int(d) => (s, d),
            StrideLeaf::Basis(b) => panic!(
                "recast_layout: the layout {layout} has the basis stride {b}; recasting counts offsets in elements, \
                 which a basis layout does not have"
            ),
        })
        .collect()
}

fn rebuild(layout: &Layout, new: Vec<(IndexExpr, IndexExpr)>) -> Layout {
    let (shape, stride): (Vec<IndexExpr>, Vec<IndexExpr>) = new.into_iter().unzip();
    let profile = layout.shape();
    let stride: Vec<StrideLeaf> = stride.into_iter().map(StrideLeaf::Int).collect();
    Layout::new(Tree::unflatten(&shape, profile), Tree::unflatten(&stride, profile))
}

impl Layout {
    /// CuTe's `upcast<N>` as CUTLASS 4.8.1 computes it, for
    /// [`recast_layout`]: each static non-zero stride `d` becomes
    /// `sign(d) * ceil_div(|d|, n)` and its extent `ceil_div(s, ceil_div(n,
    /// |d|))` (so the stride-1 leaf's extent shrinks by `n`); a static
    /// stride must divide `n` or be a multiple of it. A runtime stride is
    /// assumed to be a multiple of `n` and divided by it. ([`Layout::upcast`]
    /// is the CUTLASS 3.x form that tiled copies use.)
    #[track_caller]
    pub fn upcast_by(&self, n: usize) -> Layout {
        assert!(n > 0, "upcast_by: the factor must be positive");
        let new = leaves(self)
            .into_iter()
            .map(|(s, d)| match d.as_static_signed() {
                Some(0) => (s, d),
                Some(v) => {
                    let a = v.unsigned_abs() as usize;
                    assert!(
                        a.is_multiple_of(n) || n.is_multiple_of(a),
                        "upcast of {self} by {n}: the stride {v} neither divides {n} nor is a multiple of it (CuTe's \
                         divisibility condition)"
                    );
                    let stride = a.div_ceil(n) as i64 * v.signum();
                    (IndexExpr::ceil_div(s, IndexExpr::Static(n.div_ceil(a))), IndexExpr::from_signed(stride))
                }
                None => (s, IndexExpr::quotient(d, IndexExpr::Static(n))),
            })
            .collect();
        rebuild(self, new)
    }

    /// CuTe's `downcast<N>`: the layout of the same memory in elements `n`
    /// times narrower. The extent of each stride-1 (or -1) leaf grows by
    /// `n`, and every other stride is multiplied by `n`. The layout must
    /// have a leaf of stride 1 or -1 (adjacent elements).
    #[track_caller]
    pub fn downcast(&self, n: usize) -> Layout {
        assert!(n > 0, "downcast: the factor must be positive");
        let leaves = leaves(self);
        assert!(
            leaves.iter().any(|(_, d)| matches!(d.as_static_signed(), Some(1 | -1))),
            "downcast of {self}: it needs a mode of stride 1 (adjacent elements)"
        );
        let n_expr = IndexExpr::Static(n);
        let new = leaves
            .into_iter()
            .map(|(s, d)| match d.as_static_signed() {
                Some(1 | -1) => (IndexExpr::product(s, n_expr.clone()), d),
                Some(v) => (s, IndexExpr::from_signed(v * n as i64)),
                None => (s, IndexExpr::product(d, n_expr.clone())),
            })
            .collect();
        rebuild(self, new)
    }
}

impl Recast for Layout {
    fn recast_up(&self, n: usize) -> Layout {
        self.upcast_by(n)
    }

    fn recast_down(&self, n: usize) -> Layout {
        self.downcast(n)
    }
}

fn log2_exact(n: usize, what: &str) -> u32 {
    assert!(n.is_power_of_two(), "recast_layout of a swizzled layout: the ratio {n} of {what} must be a power of two");
    n.trailing_zeros()
}

impl Swizzle {
    /// CuTe's `upcast<N>` of a swizzle: its base moves down by `log2(n)`;
    /// past zero, the bits shrink instead.
    #[track_caller]
    pub fn upcast(&self, n: usize) -> Swizzle {
        let shift = log2_exact(n, "the upcast") as i64;
        let base = self.base as i64 - shift;
        if base >= 0 {
            Swizzle::new(self.bits, base as u32, self.shift)
        } else {
            Swizzle::new((self.bits as i64 + base).max(0) as u32, 0, self.shift)
        }
    }

    /// CuTe's `downcast<N>` of a swizzle: its base moves up by `log2(n)`.
    #[track_caller]
    pub fn downcast(&self, n: usize) -> Swizzle {
        Swizzle::new(self.bits, self.base + log2_exact(n, "the downcast"), self.shift)
    }
}

impl Recast for ComposedLayout {
    /// The inner (swizzle or layout), the offset and the outer layout all
    /// upcast, as cutegen's `upcast` of a composed layout: `8:1 o 6 o
    /// (2,4):(1,2)` by 2 is `4:1 o 3 o (1,4):(1,1)`.
    #[track_caller]
    fn recast_up(&self, n: usize) -> ComposedLayout {
        let inner = match &self.inner {
            Inner::Swizzle(sw) => Inner::Swizzle(sw.upcast(n)),
            Inner::Layout(l) => Inner::Layout(l.upcast_by(n)),
        };
        let offset = self.offset.map(&mut |o: &IndexExpr| {
            assert!(o.is_multiple_of(n), "recast_layout: the offset {o:?} of {self} is not a multiple of {n}");
            o.clone() / n
        });
        ComposedLayout { inner, offset, outer: self.outer.upcast_by(n) }
    }

    #[track_caller]
    fn recast_down(&self, n: usize) -> ComposedLayout {
        let inner = match &self.inner {
            Inner::Swizzle(sw) => Inner::Swizzle(sw.downcast(n)),
            Inner::Layout(l) => Inner::Layout(l.downcast(n)),
        };
        let offset = self.offset.map(&mut |o: &IndexExpr| o.clone() * n);
        ComposedLayout { inner, offset, outer: self.outer.downcast(n) }
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// CuTe DSL's `cute.recast_layout`, new width first: the layout of the same
/// memory viewed as elements of `new_type_bits` bits instead of
/// `old_type_bits` (C++ CuTe's `recast_layout<OldType, NewType>`). Equal
/// widths give `src` back; a wider element upcasts by the ratio
/// ([`Layout::upcast_by`]), a narrower one downcasts ([`Layout::downcast`]),
/// and otherwise, with `g` the widths' gcd, it is the downcast by `old / g`
/// of the upcast by `new / g`. A swizzled layout ([`ComposedLayout`])
/// recasts its layout and moves its swizzle's base by `log2` of the ratio,
/// which must be a power of two. Basis strides are refused.
///
/// ```
/// use cute::prelude::*;
/// let bytes = Layout::new((32, 4), (1, 32));
/// assert_eq!(recast_layout(32, 8, &bytes), Layout::new((8, 4), (1, 8)));
/// // An Int4 view of bytes: twice the elements.
/// assert_eq!(recast_layout(4, 8, &Layout::col_major(16)), Layout::col_major(32));
/// ```
#[track_caller]
pub fn recast_layout<L: Recast + Clone>(new_type_bits: usize, old_type_bits: usize, src_layout: &L) -> L {
    assert!(new_type_bits > 0 && old_type_bits > 0, "recast_layout: the type widths must be positive");
    let g = gcd(new_type_bits, old_type_bits);
    let (up, down) = (new_type_bits / g, old_type_bits / g);
    let upcast = if up > 1 { src_layout.recast_up(up) } else { src_layout.clone() };
    if down > 1 { upcast.recast_down(down) } else { upcast }
}
