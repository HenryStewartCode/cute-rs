//! The bytes a layout's codomain spans (CuTe DSL's `size_in_bytes`).

use super::index::IndexExpr;
use super::layout::{HasCosize, Layout};
use super::swizzle::{ComposedLayout, Inner};
use crate::ir::DType;
use crate::value::Numeric;

/// The layouts [`size_in_bytes`] measures: a [`Layout`], a
/// [`ComposedLayout`], a reference to either, or an `Option` of one
/// (`None` measures 0 bytes).
pub trait SizeInBytes {
    /// The number of elements the layout's codomain spans, or `None` for
    /// no layout.
    fn span(&self) -> Option<IndexExpr>;
}

impl SizeInBytes for Layout {
    fn span(&self) -> Option<IndexExpr> {
        Some(self.cosize_at(&[]))
    }
}

/// A swizzled composed layout spans its outer layout's cosize (the swizzle
/// permutes within it); one whose inner is a layout spans the inner
/// layout's.
impl SizeInBytes for ComposedLayout {
    fn span(&self) -> Option<IndexExpr> {
        Some(match &self.inner {
            Inner::Swizzle(_) => self.outer.cosize_at(&[]),
            Inner::Layout(l) => l.cosize_at(&[]),
        })
    }
}

impl<L: SizeInBytes + ?Sized> SizeInBytes for &L {
    fn span(&self) -> Option<IndexExpr> {
        (**self).span()
    }
}

impl<L: SizeInBytes> SizeInBytes for Option<L> {
    fn span(&self) -> Option<IndexExpr> {
        self.as_ref().and_then(SizeInBytes::span)
    }
}

/// The bytes that elements of `T` laid out by `layout` span (CuTe DSL's
/// `cute.size_in_bytes(dtype, layout)`): `ceil_div(cosize(layout) * bits,
/// 8)`, so sub-byte types round up to a whole byte. A swizzled composed
/// layout counts its outer layout's cosize and one with a layout inner its
/// inner layout's; `None` gives 0. A runtime cosize gives a runtime
/// expression ([`IndexExpr::as_static`] gives the number when it is
/// static). It is what sizes a shared-memory stage and a TMA transaction:
///
/// ```
/// use cute::prelude::*;
///
/// let atom = make_composed_layout(make_swizzle(3, 4, 3), 0, Layout::row_major((8, 64)));
/// let staged = tile_to_shape(&atom, (128, 64, 4));
/// let stage = staged.slice((.., .., 0));
/// assert_eq!(size_in_bytes::<F16>(&stage).as_static(), Some(128 * 64 * 2));
/// assert_eq!(size_in_bytes::<F16>(None::<&Layout>).as_static(), Some(0));
/// ```
pub fn size_in_bytes<T: Numeric>(layout: impl SizeInBytes) -> IndexExpr {
    size_in_bytes_dtype(T::DTYPE, layout)
}

/// [`size_in_bytes`] with the element type as a value.
pub fn size_in_bytes_dtype(dtype: DType, layout: impl SizeInBytes) -> IndexExpr {
    match layout.span() {
        None => IndexExpr::Static(0),
        Some(n) => IndexExpr::ceil_div(n * IndexExpr::Static(dtype.bits()), IndexExpr::Static(8)),
    }
}
