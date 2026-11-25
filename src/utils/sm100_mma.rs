// SPDX-License-Identifier: MIT OR Apache-2.0

//! CuTe DSL's SM100 MMA and shared-memory layout helpers (roadmap part
//! m10-umma-kinds; `cutlass.utils` from `blackwell_helpers`): the tiled
//! tcgen05 MMA for a pair of operand types ([`make_trivial_tiled_mma`]), the
//! layout atom for an operand's majorness and tile ([`get_smem_layout_atom_ab`]),
//! and the staged shared-memory layouts of an operand
//! ([`make_smem_layout`]), of A and B ([`make_smem_layout_a`],
//! [`make_smem_layout_b`]) and of C ([`make_smem_layout_c`]). Re-exported
//! from [`utils::sm100`](super::sm100).
//!
//! The atom choice is the widest swizzle whose atom width (1024, 512, 256
//! or 128 bits along the major mode) divides the operand's major extent,
//! which is also how CUTLASS C++'s `sm100_smem_selector` chooses.

use crate::atom::mma::{MmaAtom, TiledMma, make_tiled_mma};
use crate::ir::DType;
use crate::layout::{ComposedLayout, IntTuple, Layout, Shape, Tileable, Tree, tile_to_shape_ordered};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::cpasync::CtaGroup;
use crate::nvgpu::tcgen05::{
    MmaF8F6F4Op, MmaF16BF16Op, MmaI8Op, MmaMXF4NVF4Op, MmaMXF4Op, MmaMXF8F6F4Op, MmaTF32Op, OperandSource,
    SM103MmaMXF4NVF4Op, SM103MmaMXF4Op, SmemLayoutAtomKind, make_smem_layout_atom_dtype, tile_to_mma_shape,
};

/// The tiled tcgen05 MMA for the operand types (CuTe DSL's
/// `make_trivial_tiled_mma(a_dtype, b_dtype, a_leading_mode,
/// b_leading_mode, acc_dtype, cta_group, mma_tiler_mn, a_source)`): the
/// atom of instruction shape `(mma_tiler_mn.0, mma_tiler_mn.1, K)` for the
/// types, in a [`TiledMma`] of one atom.
///
/// | A and B                                   | operation          | K  |
/// |-------------------------------------------|--------------------|----|
/// | Float16 or BFloat16, the same             | [`MmaF16BF16Op`]   | 16 |
/// | TFloat32 or Float32, the same             | [`MmaTF32Op`]      | 8  |
/// | Int8 or Uint8, the same                   | [`MmaI8Op`]        | 32 |
/// | the F8F6F4 formats (each)                 | [`MmaF8F6F4Op`]    | 32 |
///
/// Panics for any other pair (CuTe DSL raises a `TypeError`), and for
/// parameters the operation refuses. Only the form with separate `a_dtype`
/// and `b_dtype` exists (CuTe DSL's single-`ab_dtype` form is deprecated).
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn make_trivial_tiled_mma(
    a_dtype: DType,
    b_dtype: DType,
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    acc_dtype: DType,
    cta_group: CtaGroup,
    mma_tiler_mn: (usize, usize),
    a_source: OperandSource,
) -> TiledMma {
    use DType::*;
    let (m, n) = mma_tiler_mn;
    let f8f6f4 = |t: DType| matches!(t, F8E4M3FN | F8E5M2 | F6E2M3FN | F6E3M2FN | F4E2M1FN);
    let atom: MmaAtom = match (a_dtype, b_dtype) {
        (F16 | BF16, _) if a_dtype == b_dtype => {
            MmaF16BF16Op::new(a_dtype, acc_dtype, (m, n, 16), cta_group, a_source, a_leading_mode, b_leading_mode)
                .into()
        }
        (TF32 | F32, _) if a_dtype == b_dtype => {
            MmaTF32Op::new((m, n, 8), cta_group, a_source, a_leading_mode, b_leading_mode).into()
        }
        (I8 | U8, _) if a_dtype == b_dtype => {
            MmaI8Op::new(a_dtype, (m, n, 32), cta_group, a_source, a_leading_mode, b_leading_mode).into()
        }
        (a, b) if f8f6f4(a) && f8f6f4(b) => MmaF8F6F4Op::new(
            a_dtype,
            b_dtype,
            acc_dtype,
            (m, n, 32),
            cta_group,
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
        .into(),
        _ => panic!(
            "make_trivial_tiled_mma: TypeError: no tcgen05 MMA takes a_dtype {a_dtype} with b_dtype {b_dtype} (the \
             same Float16, BFloat16, TFloat32/Float32 or Int8/Uint8 type for both, or F8F6F4 formats)"
        ),
    };
    make_tiled_mma(atom, Layout::col_major((1, 1, 1)))
}

// m10-umma-blockscaled: the block-scaled dispatchers.

/// UE8M0 scale factors for the 32-element vectors (MXF4, MXF8F6F4).
#[track_caller]
fn check_e8m0(what: &str, sf_dtype: DType, sf_vec_size: usize) {
    assert!(
        sf_dtype == DType::F8E8M0FNU,
        "{what}: ValueError: scale factors of {sf_vec_size} elements are Float8E8M0FNU (UE8M0), not {sf_dtype}; \
         Float8E4M3FN scales go with sf_vec_size 16 (NVF4)"
    );
}

/// The tiled block-scaled tcgen05 MMA (CuTe DSL's
/// `make_blockscaled_trivial_tiled_mma(a_dtype, b_dtype, a_leading_mode,
/// b_leading_mode, sf_dtype, sf_vec_size, cta_group, mma_tiler_mn,
/// a_source)`, roadmap part m10-umma-blockscaled), one atom of instruction
/// shape `(mma_tiler_mn.0, mma_tiler_mn.1, K)`:
///
/// | A and B                     | sf_vec_size | operation           | K  |
/// |-----------------------------|-------------|---------------------|----|
/// | Float4E2M1FN, both          | 32          | [`MmaMXF4Op`]       | 64 |
/// | Float4E2M1FN, both          | 16          | [`MmaMXF4NVF4Op`]   | 64 |
/// | the F8F6F4 formats (each)   | 32          | [`MmaMXF8F6F4Op`]   | 32 |
///
/// Panics (CuTe DSL's `ValueError`) for an FP4 pair with another
/// `sf_vec_size`, and (`TypeError`) for other types. CuTe DSL uses UE8M0
/// for the 32-element forms whatever `sf_dtype` says; cute-rs refuses a
/// `sf_dtype` other than Float8E8M0FNU there, and an F8F6F4 `sf_vec_size`
/// other than 32. Only the form with separate `a_dtype` and `b_dtype`
/// exists (the single-`ab_dtype` form is deprecated in CuTe DSL).
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn make_blockscaled_trivial_tiled_mma(
    a_dtype: DType,
    b_dtype: DType,
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    sf_dtype: DType,
    sf_vec_size: usize,
    cta_group: CtaGroup,
    mma_tiler_mn: (usize, usize),
    a_source: OperandSource,
) -> TiledMma {
    use DType::*;
    let what = "make_blockscaled_trivial_tiled_mma";
    let (m, n) = mma_tiler_mn;
    let f8f6f4 = |t: DType| matches!(t, F8E4M3FN | F8E5M2 | F6E2M3FN | F6E3M2FN | F4E2M1FN);
    let atom: MmaAtom = match (a_dtype, b_dtype) {
        (F4E2M1FN, F4E2M1FN) => match sf_vec_size {
            32 => {
                check_e8m0(what, sf_dtype, 32);
                MmaMXF4Op::new((m, n, 64), cta_group, a_source).into()
            }
            16 => MmaMXF4NVF4Op::new(sf_dtype, (m, n, 64), cta_group, a_source).into(),
            v => panic!("{what}: ValueError: an FP4 block-scaled MMA has sf_vec_size 16 or 32, not {v}"),
        },
        (a, b) if f8f6f4(a) && f8f6f4(b) => {
            assert!(
                sf_vec_size == 32,
                "{what}: ValueError: .kind::mxf8f6f4 scales vectors of 32 elements, not sf_vec_size {sf_vec_size}"
            );
            check_e8m0(what, sf_dtype, 32);
            MmaMXF8F6F4Op::new(a, b, (m, n, 32), cta_group, a_source, a_leading_mode, b_leading_mode).into()
        }
        _ => panic!(
            "{what}: TypeError: no block-scaled tcgen05 MMA takes a_dtype {a_dtype} with b_dtype {b_dtype} (Float4E2M1FN \
             for both, or F8F6F4 formats)"
        ),
    };
    make_tiled_mma(atom, Layout::col_major((1, 1, 1)))
}

/// The tiled sm_103a K = 96 block-scaled MMA (CuTe DSL's
/// `sm103_make_blockscaled_trivial_tiled_mma(sf_dtype, sf_vec_size,
/// cta_group, mma_tiler_mn, a_source)`): [`SM103MmaMXF4Op`] for
/// `sf_vec_size` 32 (UE8M0 scales; cute-rs refuses another `sf_dtype`) and
/// [`SM103MmaMXF4NVF4Op`]`(sf_dtype)` for 16, each of instruction shape
/// `(mma_tiler_mn.0, mma_tiler_mn.1, 96)`; panics (`ValueError`) for other
/// sizes.
#[track_caller]
pub fn sm103_make_blockscaled_trivial_tiled_mma(
    sf_dtype: DType,
    sf_vec_size: usize,
    cta_group: CtaGroup,
    mma_tiler_mn: (usize, usize),
    a_source: OperandSource,
) -> TiledMma {
    let what = "sm103_make_blockscaled_trivial_tiled_mma";
    let (m, n) = mma_tiler_mn;
    let atom: MmaAtom = match sf_vec_size {
        32 => {
            check_e8m0(what, sf_dtype, 32);
            SM103MmaMXF4Op::new((m, n, 96), cta_group, a_source).into()
        }
        16 => SM103MmaMXF4NVF4Op::new(sf_dtype, (m, n, 96), cta_group, a_source).into(),
        v => panic!("{what}: ValueError: sf_vec_size is {v}; the K = 96 FP4 MMAs take 16 or 32"),
    };
    make_tiled_mma(atom, Layout::col_major((1, 1, 1)))
}

/// The SM100 shared-memory layout atom for an operand (CuTe DSL's
/// `get_smem_layout_atom_ab(major_mode, element_type, smem_shape_mn_k,
/// sparsity=1)`): from the extent of the major mode of the `(MN, K)` tile
/// (K for a K-major operand, divided by `sparsity`; MN otherwise) in bits,
/// SW128 if it is a multiple of 1024, SW64 of 512, SW32 of 256, else INTER;
/// an MN-major 32-bit operand whose extent is a multiple of 1024 bits takes
/// `MN_SW128_32B`. Panics unless the extent is a multiple of 8 elements and
/// of 128 bits.
#[track_caller]
pub fn get_smem_layout_atom_ab(
    major_mode: OperandMajorMode,
    element_type: DType,
    smem_shape_mn_k: (usize, usize),
    sparsity: usize,
) -> SmemLayoutAtomKind {
    use SmemLayoutAtomKind::*;
    let k_major = major_mode == OperandMajorMode::K;
    let size = if k_major { smem_shape_mn_k.1 } else { smem_shape_mn_k.0 };
    assert!(
        size.is_multiple_of(8),
        "get_smem_layout_atom_ab: the major mode's extent {size} ({major_mode:?}-major tile {smem_shape_mn_k:?}) is \
         not a multiple of 8"
    );
    let mut bits = size * element_type.bits();
    if k_major {
        assert!(sparsity >= 1, "get_smem_layout_atom_ab: the sparsity is {sparsity}");
        bits /= sparsity;
    }
    assert!(
        bits.is_multiple_of(128),
        "get_smem_layout_atom_ab: the major mode spans {bits} bits of {element_type}, not a multiple of 128"
    );
    let [sw128, sw64, sw32, inter] =
        if k_major { [K_SW128, K_SW64, K_SW32, K_INTER] } else { [MN_SW128, MN_SW64, MN_SW32, MN_INTER] };
    if !k_major && element_type.bits() == 32 && bits.is_multiple_of(1024) {
        MN_SW128_32B
    } else if bits.is_multiple_of(1024) {
        sw128
    } else if bits.is_multiple_of(512) {
        sw64
    } else if bits.is_multiple_of(256) {
        sw32
    } else {
        inter
    }
}

/// A staged shared-memory layout for an operand tile (CuTe DSL's
/// `make_smem_layout(leading_mode, smem_tile_shape, a_dtype, num_stages)`):
/// the [`get_smem_layout_atom_ab`] atom tiled to `(MN, K, num_stages)` in
/// the order `(0, 1, 2)` for a K-major operand and `(1, 0, 2)` otherwise,
/// each mode then coalesced.
#[track_caller]
pub fn make_smem_layout(
    leading_mode: OperandMajorMode,
    smem_tile_shape: (usize, usize),
    a_dtype: DType,
    num_stages: usize,
) -> ComposedLayout {
    let atom = make_smem_layout_atom_dtype(get_smem_layout_atom_ab(leading_mode, a_dtype, smem_tile_shape, 1), a_dtype);
    let order: &[usize] = if leading_mode == OperandMajorMode::K { &[0, 1, 2] } else { &[1, 0, 2] };
    let (mn, k) = smem_tile_shape;
    let tiled = tile_to_shape_ordered(&atom, (mn, k, num_stages), order);
    let profile = IntTuple::from(vec![1usize, 1, 1]);
    tiled.with_layout(tiled.layout_ref().coalesce_profile(&profile))
}

/// The MMA partition shape `((MMA_MN, MMA_K), REST_MN, REST_K)`, its `(MN, K)`
/// extents, and the shape with `num_stages` appended.
fn staged_mma_shape(shape: &Shape, num_stages: usize, what: &str) -> ((usize, usize), Shape) {
    let s = shape.as_static().unwrap_or_else(|| panic!("{what}: the partition shape {shape} is not static"));
    assert!(s.rank() == 3 && s.mode(0).rank() == 2, "{what}: the partition shape {shape} is not ((MN, K), MN', K')");
    let ext = (s.mode(0).mode(0).size() * s.mode(1).size(), s.mode(0).mode(1).size() * s.mode(2).size());
    let staged: IntTuple = Tree::tuple(s.modes().iter().cloned().chain([IntTuple::from(num_stages)]));
    (ext, Shape::from(&staged))
}

/// The staged layout of one operand from its partition shape.
fn make_smem_layout_operand(
    major: OperandMajorMode,
    shape: &Shape,
    dtype: DType,
    num_stages: usize,
    what: &str,
) -> ComposedLayout {
    let (ext, staged) = staged_mma_shape(shape, num_stages, what);
    let atom = make_smem_layout_atom_dtype(get_smem_layout_atom_ab(major, dtype, ext, 1), dtype);
    let order: &[usize] = if major == OperandMajorMode::K { &[0, 1, 2] } else { &[1, 0, 2] };
    tile_to_mma_shape(&atom, staged, Some(order))
}

fn op_majors(tiled_mma: &TiledMma, what: &str) -> (OperandMajorMode, OperandMajorMode) {
    let crate::atom::mma::MmaOp::Sm100(kind) = tiled_mma.atom().op() else {
        panic!("{what}: the tiled MMA {:?} is not a tcgen05 MMA", tiled_mma.atom().op())
    };
    let major = |t: bool| if t { OperandMajorMode::MN } else { OperandMajorMode::K };
    (major(kind.trans_a), major(kind.trans_b))
}

/// The staged shared-memory layout of A (CuTe DSL's
/// `make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, num_stages, *,
/// is_k_major=None)`): A's partition shape of the `(M, K)` tile
/// ([`TiledMma::partition_shape_a`]), the [`get_smem_layout_atom_ab`] atom
/// for its extents and majorness (the MMA's, unless `is_k_major` says), and
/// [`tile_to_mma_shape`] of the atom to that shape with `num_stages`
/// appended, in the order `(0, 1, 2)` for K-major A and `(1, 0, 2)`
/// otherwise.
#[track_caller]
pub fn make_smem_layout_a(
    tiled_mma: &TiledMma,
    mma_tiler_mnk: (usize, usize, usize),
    a_dtype: DType,
    num_stages: usize,
    is_k_major: Option<bool>,
) -> ComposedLayout {
    let what = "make_smem_layout_a";
    let major = match is_k_major {
        Some(k) => {
            if k {
                OperandMajorMode::K
            } else {
                OperandMajorMode::MN
            }
        }
        None => op_majors(tiled_mma, what).0,
    };
    let (m, _, k) = mma_tiler_mnk;
    make_smem_layout_operand(major, &tiled_mma.partition_shape_a((m, k)), a_dtype, num_stages, what)
}

/// The staged shared-memory layout of B (CuTe DSL's
/// `make_smem_layout_b(...)`): [`make_smem_layout_a`] for the `(N, K)` tile.
#[track_caller]
pub fn make_smem_layout_b(
    tiled_mma: &TiledMma,
    mma_tiler_mnk: (usize, usize, usize),
    b_dtype: DType,
    num_stages: usize,
    is_k_major: Option<bool>,
) -> ComposedLayout {
    let what = "make_smem_layout_b";
    let major = match is_k_major {
        Some(k) => {
            if k {
                OperandMajorMode::K
            } else {
                OperandMajorMode::MN
            }
        }
        None => op_majors(tiled_mma, what).1,
    };
    let (_, n, k) = mma_tiler_mnk;
    make_smem_layout_operand(major, &tiled_mma.partition_shape_b((n, k)), b_dtype, num_stages, what)
}

/// The staged shared-memory layout of C (CuTe DSL's
/// `sm100.make_smem_layout_c(tiled_mma, mma_tiler_mnk, c_dtype, num_stages,
/// is_m_major)`): C's partition shape of the `(M, N)` tile, the atom for its
/// extents (M-major when `is_m_major`, else N-major, the "K" side of
/// [`get_smem_layout_atom_ab`]), tiled with [`tile_to_mma_shape`] in the
/// order `(1, 0, 2)` when M-major and `(0, 1, 2)` otherwise.
#[track_caller]
pub fn make_smem_layout_c(
    tiled_mma: &TiledMma,
    mma_tiler_mnk: (usize, usize, usize),
    c_dtype: DType,
    num_stages: usize,
    is_m_major: bool,
) -> ComposedLayout {
    let (m, n, _) = mma_tiler_mnk;
    let major = if is_m_major { OperandMajorMode::MN } else { OperandMajorMode::K };
    make_smem_layout_operand(major, &tiled_mma.partition_shape_c((m, n)), c_dtype, num_stages, "make_smem_layout_c")
}
