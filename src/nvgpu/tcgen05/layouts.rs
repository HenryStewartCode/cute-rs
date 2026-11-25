// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/mma_traits_sm100.hpp (Layout_MN_SW128_32B_Atom, the
//   *_SpAtom sparse atoms and UMMA::tile_to_mma_shape),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The SM100 shared-memory layout atoms and `tile_to_mma_shape`.

use crate::ir::DType;
use crate::layout::{
    ComposedLayout, IntTuple, Layout, Shape, Swizzle, Tileable, Tiler, Tree, blocked_product, make_composed_layout,
    tile_to_shape_ordered, tiled_divide,
};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::warpgroup;
use crate::value::Numeric;

/// The kinds of SM100 (UMMA) shared-memory layout atom (CuTe DSL's
/// `tcgen05.SmemLayoutAtomKind`): the eight SM90 kinds
/// ([`warpgroup::SmemLayoutAtomKind`]) and `MN_SW128_32B`, the MN-major
/// 128-byte swizzle with 32-byte atomicity (`Swizzle<2,5,2>` on bytes),
/// which only UMMA reads.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SmemLayoutAtomKind {
    MN_INTER,
    MN_SW32,
    MN_SW64,
    MN_SW128,
    MN_SW128_32B,
    K_INTER,
    K_SW32,
    K_SW64,
    K_SW128,
}

impl SmemLayoutAtomKind {
    pub const ALL: [SmemLayoutAtomKind; 9] = [
        SmemLayoutAtomKind::MN_INTER,
        SmemLayoutAtomKind::MN_SW32,
        SmemLayoutAtomKind::MN_SW64,
        SmemLayoutAtomKind::MN_SW128,
        SmemLayoutAtomKind::MN_SW128_32B,
        SmemLayoutAtomKind::K_INTER,
        SmemLayoutAtomKind::K_SW32,
        SmemLayoutAtomKind::K_SW64,
        SmemLayoutAtomKind::K_SW128,
    ];

    /// The SM90 kind this one shares its atom with (all but `MN_SW128_32B`).
    pub fn sm90(self) -> Option<warpgroup::SmemLayoutAtomKind> {
        use warpgroup::SmemLayoutAtomKind as W;
        Some(match self {
            SmemLayoutAtomKind::MN_INTER => W::MN_INTER,
            SmemLayoutAtomKind::MN_SW32 => W::MN_SW32,
            SmemLayoutAtomKind::MN_SW64 => W::MN_SW64,
            SmemLayoutAtomKind::MN_SW128 => W::MN_SW128,
            SmemLayoutAtomKind::MN_SW128_32B => return None,
            SmemLayoutAtomKind::K_INTER => W::K_INTER,
            SmemLayoutAtomKind::K_SW32 => W::K_SW32,
            SmemLayoutAtomKind::K_SW64 => W::K_SW64,
            SmemLayoutAtomKind::K_SW128 => W::K_SW128,
        })
    }

    /// The major mode the atom is for.
    pub fn major_mode(self) -> OperandMajorMode {
        match self.sm90() {
            Some(k) => k.major_mode(),
            None => OperandMajorMode::MN,
        }
    }

    /// Contiguous bits along the major mode: 128, 256, 512 or 1024.
    pub fn contiguous_bits(self) -> usize {
        match self.sm90() {
            Some(k) => k.contiguous_bits(),
            None => 1024,
        }
    }

    /// The atom's swizzle on byte addresses: `Swizzle<B,4,3>` for the SM90
    /// kinds, `Swizzle<2,5,2>` for `MN_SW128_32B`.
    pub fn byte_swizzle(self) -> Swizzle {
        match self.sm90() {
            Some(k) => k.byte_swizzle(),
            None => Swizzle { bits: 2, base: 5, shift: 2 },
        }
    }

    /// Rows of the atom: 8, except 4 for `MN_SW128_32B` (whose
    /// `Swizzle<2,5,2>` repeats every 4 rows of 128 bytes).
    pub fn rows(self) -> usize {
        if self == SmemLayoutAtomKind::MN_SW128_32B { 4 } else { 8 }
    }
}

/// Check that elements of `element_type` fit the atoms, and return their
/// width in bits.
#[track_caller]
fn atom_element_bits(func: &str, element_type: DType) -> usize {
    let bits = element_type.bits();
    assert!(
        bits.is_power_of_two() && (4..=128).contains(&bits),
        "{func}: {} is {bits} bits wide; an SM100 shared-memory layout atom takes elements of 4, 8, 16, 32, 64 or \
         128 bits",
        element_type.cute_name()
    );
    bits
}

/// The SM100 shared-memory layout atom of `kind` for elements of `T` (CuTe
/// DSL's `tcgen05.make_smem_layout_atom(kind, element_type)`), a composed
/// layout in units of elements. See [`make_smem_layout_atom_dtype`].
#[track_caller]
pub fn make_smem_layout_atom<T: Numeric>(kind: SmemLayoutAtomKind) -> ComposedLayout {
    make_smem_layout_atom_dtype(kind, T::DTYPE)
}

/// [`make_smem_layout_atom`] for an element type given as a [`DType`].
///
/// The eight kinds shared with SM90 are [`warpgroup::make_smem_layout_atom_dtype`]'s
/// atoms. `MN_SW128_32B` is CuTe C++'s `Layout_MN_SW128_32B_Atom`, `sw o 0
/// o (c,4):(1,c)` with `c = 1024 / width` and `sw` the byte swizzle
/// `Swizzle<2,5,2>` restated on elements (as for every atom here: cute-rs's
/// composed layouts swizzle element offsets): 4 rows, the swizzle's period.
///
/// CuTe DSL 4.8.1 builds this atom with 8 rows. Tiled K-first (as the
/// staged layouts of MN-major operands are) the two give the same layout,
/// but tiled MN-first only the 4-row atom keeps the 4-row blocks along K
/// evenly spaced, as the UMMA descriptor's canonical form needs, so cute-rs
/// follows C++.
///
/// Panics for elements narrower than 4 bits, wider than 128, or whose width
/// is not a power of two (the 6-bit floats).
#[track_caller]
pub fn make_smem_layout_atom_dtype(kind: SmemLayoutAtomKind, element_type: DType) -> ComposedLayout {
    let bits = atom_element_bits("make_smem_layout_atom", element_type);
    if let Some(k) = kind.sm90() {
        return warpgroup::make_smem_layout_atom_dtype(k, element_type);
    }
    let c = kind.contiguous_bits() / bits;
    let sw = element_swizzle(kind, bits, "make_smem_layout_atom");
    make_composed_layout(sw, 0, Layout::new((c, kind.rows()), (1, c)))
}

#[track_caller]
fn element_swizzle(kind: SmemLayoutAtomKind, unit_bits: usize, func: &str) -> Swizzle {
    kind.byte_swizzle().to_elements(unit_bits).unwrap_or_else(|why| panic!("{func}: {why}"))
}

/// The sparse SM100 shared-memory layout atom of `kind` for elements of `T`
/// with sparsity `sparsity` (CuTe DSL's
/// `tcgen05.make_sparse_smem_layout_atom(kind, element_type, sparsity=2)`).
/// See [`make_sparse_smem_layout_atom_dtype`].
#[track_caller]
pub fn make_sparse_smem_layout_atom<T: Numeric>(kind: SmemLayoutAtomKind, sparsity: usize) -> ComposedLayout {
    make_sparse_smem_layout_atom_dtype(kind, T::DTYPE, sparsity)
}

/// [`make_sparse_smem_layout_atom`] for an element type given as a
/// [`DType`]: the dense atom's outer layout `L` blocked into
/// `blocked_product((1,S), L)`, as CuTe C++'s `Layout_*_SpAtom<T, S>`
/// aliases build it. The result counts units of `width / S` bits (the
/// sparse pointer's unit: `S` logical K elements share one stored
/// element's bits), so the byte swizzle is restated on units of that
/// width. `sparsity` must be a power of two from 1 to the element width.
#[track_caller]
pub fn make_sparse_smem_layout_atom_dtype(
    kind: SmemLayoutAtomKind,
    element_type: DType,
    sparsity: usize,
) -> ComposedLayout {
    let bits = atom_element_bits("make_sparse_smem_layout_atom", element_type);
    assert!(
        sparsity.is_power_of_two() && sparsity <= bits,
        "make_sparse_smem_layout_atom: sparsity {sparsity} is not a power of two from 1 to the {bits}-bit width of {}",
        element_type.cute_name()
    );
    let c = kind.contiguous_bits() / bits;
    let dense = match kind.major_mode() {
        OperandMajorMode::MN => Layout::new((c, kind.rows()), (1, c)),
        OperandMajorMode::K => Layout::new((8, c), (c, 1)),
    };
    let sparse = blocked_product(&Layout::col_major((1, sparsity)), &dense);
    let sw = element_swizzle(kind, bits / sparsity, "make_sparse_smem_layout_atom");
    make_composed_layout(sw, 0, sparse)
}

/// Tile a layout atom to an MMA tile shape (CuTe DSL's
/// `tcgen05.tile_to_mma_shape(atom, mma_tile_shape, order=None)`, CuTe
/// C++'s `UMMA::tile_to_mma_shape`). `mma_tile_shape` is `((MMA_MN, MMA_K),
/// REST_MN, REST_K, ...)`: the shape of one MMA's operand tile, how many
/// tiles along MN and K, and further modes (pipeline stages). The atom is
/// tiled to `((MMA_MN, REST_MN), (MMA_K, REST_K), ...)` in the mode order
/// `order` (0-based, a permutation of the modes of that shape, as
/// [`tile_to_shape_ordered`] takes; `None` is column-major) and divided
/// back into MMA tiles: the result is `((MMA_MN, MMA_K), REST_MN, REST_K,
/// ...)`. A composed atom keeps its swizzle.
///
/// The shape must be static and of rank 3 or more with a rank-2 first mode.
#[track_caller]
pub fn tile_to_mma_shape<X: Tileable>(atom: &X, mma_tile_shape: impl Into<Shape>, order: Option<&[usize]>) -> X {
    let shape: Shape = mma_tile_shape.into();
    let s: IntTuple = shape.as_static().unwrap_or_else(|| {
        panic!("tile_to_mma_shape: the MMA tile shape {shape} has runtime extents; it must be static")
    });
    assert!(
        s.rank() >= 3 && !s.mode(0).is_leaf() && s.mode(0).rank() == 2,
        "tile_to_mma_shape: the MMA tile shape {shape} is not ((MMA_MN, MMA_K), REST_MN, REST_K, ...)"
    );
    let first = s.mode(0);
    let mut modes: Vec<IntTuple> = vec![
        Tree::tuple([first.mode(0).clone(), s.mode(1).clone()]),
        Tree::tuple([first.mode(1).clone(), s.mode(2).clone()]),
    ];
    modes.extend(s.modes()[3..].iter().cloned());
    let mn_shape: IntTuple = Tree::tuple(modes);
    let default: Vec<usize> = (0..mn_shape.rank()).collect();
    let order = order.unwrap_or(&default);
    let tiled = tile_to_shape_ordered(atom, Shape::from(&mn_shape), order);
    let mma = first.product_each();
    tiled_divide(&tiled, Tiler::from(&mma))
}
