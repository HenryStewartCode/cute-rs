//! Composed layouts whose inner function is a Rust closure (CuTe DSL's
//! `make_composed_layout` with a callable as its inner).

use std::fmt;
use std::rc::Rc;

use super::basis::tuple_sum;
use super::index::IndexExpr;
use super::layout::{Coord, Layout, Shape, SliceCoord, Tileable};
use super::tree::HasShape;

/// The composed layout `inner o offset o outer` whose inner function is a
/// closure: `c -> inner(offset + outer(c))`, the offset added to the outer
/// layout's value leaf by leaf (`tuple_add`). It is CuTe DSL's
/// `make_composed_layout(callable, offset, outer)`.
///
/// The closure runs when the kernel is built, every time the layout is
/// called, so it may record operations: a gather layout's closure reads an
/// index tensor ([`Tensor::load`](crate::Tensor::load)) and returns the
/// offset it holds. Its result is whatever the closure returns.
///
/// As for [`ComposedLayout`](super::ComposedLayout), the outer layout gives
/// the shape and is what slicing and the layout algebra act on
/// ([`Tileable`]: the divides, [`local_partition`](crate::local_partition)'s
/// tiling, [`coalesce`](super::coalesce)); a slice moves the offset of the
/// modes it fixes into the composed offset. Such a layout has no memory
/// representation, so [`make_tensor`](crate::make_tensor) refuses it, as
/// CuTe DSL does.
pub struct FnComposedLayout<F> {
    inner: Rc<F>,
    name: Rc<str>,
    offset: Coord,
    outer: Layout,
}

impl<F> Clone for FnComposedLayout<F> {
    fn clone(&self) -> Self {
        FnComposedLayout {
            inner: self.inner.clone(),
            name: self.name.clone(),
            offset: self.offset.clone(),
            outer: self.outer.clone(),
        }
    }
}

impl<F> FnComposedLayout<F> {
    /// The inner function.
    pub fn inner(&self) -> &F {
        &self.inner
    }

    /// The offset between the two (CuTe DSL's `ComposedLayout.offset`).
    pub fn offset(&self) -> &Coord {
        &self.offset
    }

    /// The outer layout (CuTe DSL's `ComposedLayout.outer`).
    pub fn outer(&self) -> &Layout {
        &self.outer
    }

    /// The name the inner function prints as (`fn` unless
    /// [`named`](FnComposedLayout::named) set one).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The same layout, printing its inner function as `name`.
    pub fn named(&self, name: &str) -> Self {
        FnComposedLayout { name: name.into(), ..self.clone() }
    }

    pub fn shape(&self) -> &Shape {
        self.outer.shape()
    }

    /// Size of the domain; panics if it depends on a runtime extent.
    #[track_caller]
    pub fn size(&self) -> usize {
        self.outer.size()
    }

    /// Number of top-level modes of the outer layout.
    pub fn rank(&self) -> usize {
        self.outer.rank()
    }

    /// Nesting depth of the outer layout's shape.
    pub fn depth(&self) -> usize {
        self.outer.depth()
    }

    fn with_outer(&self, outer: Layout) -> Self {
        FnComposedLayout { outer, ..self.clone() }
    }

    /// Mode `i` of the outer layout, with the same inner and offset.
    pub fn mode(&self, i: usize) -> Self {
        self.with_outer(self.outer.mode(i))
    }

    /// The mode at a path (CuTe DSL's `cute.get(layout, mode=...)`).
    pub fn get(&self, path: &[usize]) -> Self {
        self.with_outer(self.outer.get(path))
    }

    /// Select (and reorder) top-level modes of the outer layout.
    pub fn select(&self, modes: &[usize]) -> Self {
        self.with_outer(self.outer.select(modes))
    }

    /// Group modes `[begin, end)` of the outer layout into one.
    #[track_caller]
    pub fn group_modes(&self, begin: usize, end: usize) -> Self {
        self.with_outer(self.outer.group(begin, end))
    }

    /// Merge contiguous modes of the outer layout.
    pub fn coalesce(&self) -> Self {
        self.with_outer(self.outer.coalesce())
    }

    /// Slice with `coord` (CuTe DSL's `cute.slice_`): the kept modes of the
    /// outer layout, the offset of the fixed ones added to the composed
    /// offset.
    #[track_caller]
    pub fn slice(&self, coord: impl Into<SliceCoord>) -> Self {
        let (outer, off) = self.outer.slice_and_offset_in(&coord.into(), "FnComposedLayout::slice");
        FnComposedLayout { offset: tuple_sum(&self.offset, &off), outer, ..self.clone() }
    }

    /// `offset + outer(coord)`, what the inner function is called with.
    #[track_caller]
    pub fn inner_argument(&self, coord: impl Into<Coord>) -> Coord {
        let at = self.outer.eval_checked(&coord.into(), "FnComposedLayout::call");
        tuple_sum(&self.offset, &at)
    }

    /// Evaluate at `coord`: `inner(offset + outer(coord))` (CuTe DSL's
    /// `layout(coord)`).
    #[track_caller]
    pub fn call<R>(&self, coord: impl Into<Coord>) -> R
    where
        F: Fn(Coord) -> R,
    {
        (self.inner)(self.inner_argument(coord))
    }
}

/// The algebra acts on the outer layout, keeping the inner function and the
/// offset.
impl<F> Tileable for FnComposedLayout<F> {
    fn layout_ref(&self) -> &Layout {
        &self.outer
    }

    fn with_layout(&self, layout: Layout) -> Self {
        self.with_outer(layout)
    }
}

impl<F> HasShape for FnComposedLayout<F> {
    type Leaf = IndexExpr;

    fn shape(&self) -> &Shape {
        self.outer.shape()
    }
}

/// `(inner o offset o outer)`, as CuTe DSL prints a composed layout with a
/// function inner: `(fn o (1,0) o (8,4):(1@0,1@1))`.
impl<F> fmt::Display for FnComposedLayout<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} o {} o {})", self.name, self.offset, self.outer)
    }
}

impl<F> fmt::Debug for FnComposedLayout<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// CuTe DSL's `make_composed_layout(inner, offset, outer)` with a function
/// as the inner: the layout `c -> inner(offset + outer(c))`. For a swizzle
/// or a layout as the inner, [`make_composed_layout`](super::make_composed_layout).
///
/// ```
/// use cute::prelude::*;
///
/// // The notebook's custom transformation: (x, y) -> (x, y + 1) over an
/// // identity layout, offset by (1, 0).
/// let inner = |c: Coord| {
///     let (x, y) = (c.mode(0).as_leaf().unwrap().clone(), c.mode(1).as_leaf().unwrap().clone());
///     (x.as_static().unwrap(), y.as_static().unwrap() + 1)
/// };
/// let layout = make_composed_layout_fn(inner, (1, 0), make_identity_layout((8, 4)));
/// assert_eq!(layout.to_string(), "(fn o (1,0) o (8,4):(1@0,1@1))");
/// assert_eq!(layout.call(0), (1, 1));
/// assert_eq!(layout.call((2, 3)), (3, 4));
/// ```
#[track_caller]
pub fn make_composed_layout_fn<F>(inner: F, offset: impl Into<Coord>, outer: Layout) -> FnComposedLayout<F> {
    FnComposedLayout { inner: Rc::new(inner), name: "fn".into(), offset: offset.into(), outer }
}
