//! CuTe layouts: hierarchical tuples, the layout algebra, swizzles and
//! symbolic index expressions.

mod index;
#[allow(clippy::module_inception)]
mod layout;
mod swizzle;
mod tree;

pub use index::IndexExpr;
pub use layout::{
    Coord, Layout, SliceCoord, Tileable, Tiler, append_ones, blocked_product, coalesce, complement, composition, dice,
    flat_divide, flat_product, idx2crd, is_congruent, left_inverse, logical_divide, logical_product, make_layout,
    make_layout_like, make_layout_tv, make_ordered_layout, max_common_layout, max_common_vector, prepend_ones,
    raked_product, right_inverse, round_up, tile_to_shape, tiled_divide, tiled_product, zipped_divide, zipped_product,
};
pub use swizzle::{ComposedLayout, Swizzle, make_composed_layout, make_swizzle, tile_to_shape_swizzled};
pub use tree::{IntTuple, Tree, ceil_div, shape_div};

#[cfg(test)]
mod tests;
