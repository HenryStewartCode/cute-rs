// SPDX-License-Identifier: MIT OR Apache-2.0

//! CuTe DSL's SM90 shared-memory layout helpers (`cutlass.utils.sm90`,
//! `hopper_helpers`): the layout atom for an operand's majorness and major
//! extent ([`get_smem_layout_atom`]), and the staged shared-memory layouts
//! of A, B and the epilogue ([`make_smem_layout_a`], [`make_smem_layout_b`],
//! [`make_smem_layout_epi`]), and the tiled warpgroup MMA for a pair of
//! operand types ([`make_trivial_tiled_mma`]). They return [`ComposedLayout`]s in units of
//! elements, which [`make_smem_tensor`](crate::make_smem_tensor) and
//! `MemRange::get_tensor_composed` take.

use super::LayoutEnum;
use crate::atom::copy::{CopyAtom, CopyOp, make_copy_atom};
use crate::atom::mma::{MmaAtom, TiledMma, make_tiled_mma};
use crate::ir::DType;
use crate::layout::Layout;
use crate::layout::{ComposedLayout, Shape, tile_to_shape_ordered};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::warpgroup::{
    MmaF8Op, MmaF16BF16Op, MmaI8Op, OperandSource, SmemLayoutAtomKind, make_smem_layout_atom_dtype,
};
use crate::value::Numeric;

/// The widest swizzled atom whose row fits a whole number of times into
/// `major_mode_size` elements of `element_type` along the major mode (CuTe
/// DSL's `get_smem_layout_atom(layout, element_type, major_mode_size)`):
/// SW128 if the extent in bits is a multiple of 1024, SW64 of 512, SW32 of
/// 256, else INTER; `MN_*` for a column-major (MN-major) operand, `K_*` for
/// a row-major one. Panics unless `major_mode_size` is a multiple of 8.
#[track_caller]
pub fn get_smem_layout_atom(layout: LayoutEnum, element_type: DType, major_mode_size: usize) -> SmemLayoutAtomKind {
    use SmemLayoutAtomKind::*;
    assert!(
        major_mode_size.is_multiple_of(8),
        "get_smem_layout_atom: the major mode size {major_mode_size} is not a multiple of 8"
    );
    let bits = major_mode_size * element_type.bits();
    let [sw128, sw64, sw32, inter] = match layout.sm90_mma_major_mode() {
        OperandMajorMode::MN => [MN_SW128, MN_SW64, MN_SW32, MN_INTER],
        OperandMajorMode::K => [K_SW128, K_SW64, K_SW32, K_INTER],
    };
    if bits.is_multiple_of(1024) {
        sw128
    } else if bits.is_multiple_of(512) {
        sw64
    } else if bits.is_multiple_of(256) {
        sw32
    } else {
        inter
    }
}

/// The staged shared-memory layout of A (CuTe DSL's
/// `make_smem_layout_a(a_layout, mma_tiler_mnk, a_dtype, num_stages)`):
/// the atom for A's majorness and its K extent (K-major) or M extent
/// (M-major), tiled to `(M, K, num_stages)` in the mode order `(0, 1, 2)`
/// for K-major A and `(1, 0, 2)` for M-major A.
#[track_caller]
pub fn make_smem_layout_a(
    a_layout: LayoutEnum,
    mma_tiler_mnk: (usize, usize, usize),
    a_dtype: DType,
    num_stages: usize,
) -> ComposedLayout {
    let (m, _, k) = mma_tiler_mnk;
    let k_major = a_layout.is_k_major_a();
    staged(a_layout, a_dtype, if k_major { k } else { m }, (m, k, num_stages), k_major)
}

/// The staged shared-memory layout of B (CuTe DSL's
/// `make_smem_layout_b(b_layout, mma_tiler_mnk, b_dtype, num_stages)`):
/// [`make_smem_layout_a`] on `(N, K, num_stages)`, the atom from the K
/// extent (K-major) or the N extent (N-major).
#[track_caller]
pub fn make_smem_layout_b(
    b_layout: LayoutEnum,
    mma_tiler_mnk: (usize, usize, usize),
    b_dtype: DType,
    num_stages: usize,
) -> ComposedLayout {
    let (_, n, k) = mma_tiler_mnk;
    let k_major = b_layout.is_k_major_b();
    staged(b_layout, b_dtype, if k_major { k } else { n }, (n, k, num_stages), k_major)
}

fn staged(
    layout: LayoutEnum,
    dtype: DType,
    major: usize,
    shape: (usize, usize, usize),
    in_order: bool,
) -> ComposedLayout {
    let atom = make_smem_layout_atom_dtype(get_smem_layout_atom(layout, dtype, major), dtype);
    let order: &[usize] = if in_order { &[0, 1, 2] } else { &[1, 0, 2] };
    tile_to_shape_ordered(&atom, shape, order)
}

/// The staged shared-memory layout of the epilogue tile (CuTe DSL's
/// `make_smem_layout_epi(epi_dtype, epi_layout, epi_tile, epi_stage,
/// smem_trg_shape=None, smem_order=None)`): the atom for C's majorness and
/// its N extent (N-major, row-major C) or M extent (M-major), tiled to
/// `smem_trg_shape` (by default `(M, N, epi_stage)`) in the order
/// `smem_order` (by default `(1, 0, 2)` for M-major C and `(0, 1, 2)` for
/// N-major C).
#[track_caller]
pub fn make_smem_layout_epi(
    epi_dtype: DType,
    epi_layout: LayoutEnum,
    epi_tile: (usize, usize),
    epi_stage: usize,
    smem_trg_shape: Option<Shape>,
    smem_order: Option<&[usize]>,
) -> ComposedLayout {
    let (m, n) = epi_tile;
    let major = if epi_layout.is_n_major_c() { n } else { m };
    let atom = make_smem_layout_atom_dtype(get_smem_layout_atom(epi_layout, epi_dtype, major), epi_dtype);
    let shape = smem_trg_shape.unwrap_or_else(|| (m, n, epi_stage).into());
    let default: &[usize] = if epi_layout.is_m_major_c() { &[1, 0, 2] } else { &[0, 1, 2] };
    tile_to_shape_ordered(&atom, shape, smem_order.unwrap_or(default))
}

/// A tiled warpgroup MMA for the operand types (CuTe DSL's
/// `make_trivial_tiled_mma(a_dtype, b_dtype, a_leading_mode,
/// b_leading_mode, acc_dtype, atom_layout_mnk, tiler_mn, a_source)`, also
/// `sm90_make_trivial_tiled_mma`): the
/// atom of instruction shape `(tiler_mn.0, tiler_mn.1, K)` for the types,
/// tiled over `atom_layout_mnk` (column-major). `tiler_mn` is the
/// instruction's `(64, N)`, not the CTA tile.
///
/// | A and B                                  | operation                    | K  |
/// |------------------------------------------|------------------------------|----|
/// | Float16 or BFloat16, the same            | [`MmaF16BF16Op`]             | 16 |
/// | Float8E4M3FN or Float8E5M2 (each)        | [`MmaF8Op`]                  | 32 |
/// | Int8 or Uint8 (each)                     | [`MmaI8Op`]                  | 32 |
///
/// Panics for any other pair (CuTe DSL raises a `TypeError`), and for
/// parameters the operation refuses.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn make_trivial_tiled_mma(
    a_dtype: DType,
    b_dtype: DType,
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    acc_dtype: DType,
    atom_layout_mnk: (usize, usize, usize),
    tiler_mn: (usize, usize),
    a_source: OperandSource,
) -> TiledMma {
    use DType::*;
    let (m, n) = tiler_mn;
    let atom: MmaAtom = match (a_dtype, b_dtype) {
        (F16 | BF16, _) => {
            assert!(
                a_dtype == b_dtype,
                "make_trivial_tiled_mma: TypeError: a_dtype {a_dtype} and b_dtype {b_dtype} differ; a Float16 or \
                 BFloat16 warpgroup MMA takes A and B of the same type"
            );
            MmaF16BF16Op::new(a_dtype, acc_dtype, (m, n, 16), a_source, a_leading_mode, b_leading_mode).into()
        }
        (F8E4M3FN | F8E5M2, F8E4M3FN | F8E5M2) => {
            MmaF8Op::new(a_dtype, b_dtype, acc_dtype, (m, n, 32), a_source, a_leading_mode, b_leading_mode).into()
        }
        (I8 | U8, I8 | U8) => {
            MmaI8Op::new(a_dtype, b_dtype, acc_dtype, (m, n, 32), a_source, a_leading_mode, b_leading_mode).into()
        }
        _ => panic!(
            "make_trivial_tiled_mma: TypeError: no SM90 warpgroup MMA takes a_dtype {a_dtype} with b_dtype {b_dtype} \
             (Float16/BFloat16 pairs, Float8E4M3FN/Float8E5M2 or Int8/Uint8)"
        ),
    };
    make_tiled_mma(atom, Layout::col_major(atom_layout_mnk))
}

/// [`make_trivial_tiled_mma`] by its other CuTe DSL name.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn sm90_make_trivial_tiled_mma(
    a_dtype: DType,
    b_dtype: DType,
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    acc_dtype: DType,
    atom_layout_mnk: (usize, usize, usize),
    tiler_mn: (usize, usize),
    a_source: OperandSource,
) -> TiledMma {
    make_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        a_leading_mode,
        b_leading_mode,
        acc_dtype,
        atom_layout_mnk,
        tiler_mn,
        a_source,
    )
}

// The SM90 epilogue helpers.

/// The widest register-to-shared store atom for an epilogue's D under the
/// constraint of its global layout (CuTe DSL's `get_smem_store_op(layout_d,
/// elem_ty_d, elem_ty_acc)`): `stmatrix.m8n8.x4` (transposed for an M-major
/// D) for a 16-bit `D`, a plain [`CopyOp::Universal`] copy of one element
/// otherwise. The accumulator type does not change the choice; it is taken
/// for parity with CuTe DSL.
#[track_caller]
pub fn get_smem_store_op<D: Numeric>(layout_d: LayoutEnum, acc: DType) -> CopyAtom<D> {
    let _ = acc;
    if D::DTYPE.bits() == 16 {
        make_copy_atom::<D>(CopyOp::StMatrix { num: 4, trans: layout_d.is_m_major_c() })
    } else {
        make_copy_atom::<D>(CopyOp::Universal { bits: D::DTYPE.bits().max(8) })
    }
}

/// The name CuTe DSL keeps for [`get_smem_store_op`].
#[track_caller]
pub fn sm90_get_smem_store_op<D: Numeric>(layout_d: LayoutEnum, acc: DType) -> CopyAtom<D> {
    get_smem_store_op::<D>(layout_d, acc)
}

/// The epilogue tile for a CTA tile `(M, N, K)` (CuTe DSL's
/// `compute_tile_shape_or_override`): `epi_tile_override` if given;
/// otherwise `(min(128, M), min(32, N))` for a cooperative kernel and
/// `(min(64, M), min(n, N))` for a ping-pong one, `n` being 64 for 8-bit
/// elements and 32 for the rest.
pub fn compute_tile_shape_or_override(
    tile_shape_mnk: (usize, usize, usize),
    element_type: DType,
    is_cooperative: bool,
    epi_tile_override: Option<(usize, usize)>,
) -> (usize, usize) {
    if let Some(tile) = epi_tile_override {
        return tile;
    }
    let (m, n, _) = tile_shape_mnk;
    if is_cooperative {
        (m.min(128), n.min(32))
    } else {
        let n_perf = if element_type.bits() == 8 { 64 } else { 32 };
        (m.min(64), n.min(n_perf))
    }
}
