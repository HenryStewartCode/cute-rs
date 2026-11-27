//! CuTe layouts: hierarchical tuples, the layout algebra, swizzles and
//! symbolic index expressions.

mod basis;
mod bytes;
mod fn_composed;
pub(crate) mod image_mask;
mod index;
// `layout::layout` holds `Layout`, as CuTe's `layout.py` does.
#[allow(clippy::module_inception)]
mod layout;
mod parse;
mod recast;
mod shape_arith;
mod swizzle;
mod tree;

pub(crate) use basis::tuple_sum;
pub use basis::{
    BasisGet, BasisMode, BasisScale, E, Ratio, ScaledBasis, StrideLeaf, basis_get, basis_value, make_basis_like,
    make_identity_layout,
};
pub use bytes::{SizeInBytes, size_in_bytes, size_in_bytes_dtype};
pub use fn_composed::{FnComposedLayout, make_composed_layout_fn};
pub use image_mask::make_layout_image_mask;
pub use index::IndexExpr;
pub use layout::{
    Coord, Crd2Idx, HasCosize, Layout, LayoutLike, LayoutSource, Shape, SliceCoord, Stride, Tileable, Tiler,
    append_ones, blocked_product, coalesce, complement, composition, cosize, crd2idx, depth, dice, flat_divide,
    flat_product, idx2crd, is_congruent, is_major, is_weakly_congruent, leading_dim, left_inverse, logical_divide,
    logical_product, make_fragment_like_layout, make_layout, make_layout_like, make_layout_tv, make_ordered_layout,
    max_common_layout, max_common_vector, nullspace, prepend_ones, raked_product, rank, right_inverse, shape, size,
    tile_to_shape, tile_to_shape_ordered, tiled_divide, tiled_product, zipped_divide, zipped_product,
};
pub use parse::ParseLayoutError;
pub use recast::{Recast, recast_layout};
pub use shape_arith::{EqualOperand, ShapeTiler, ceil_div, equal, round_up, shape_div};
pub use swizzle::{
    ComposedLayout, Inner, Swizzle, SwizzlePortion, get_nonswizzle_portion, get_swizzle_portion, make_composed_layout,
    make_swizzle, tile_to_shape_swizzled, tile_to_shape_swizzled_ordered,
};
pub(crate) use tree::tree_from_tuple;
pub use tree::{HasShape, IntTuple, StaticLeaf, Tree};

#[cfg(test)]
mod tests;
