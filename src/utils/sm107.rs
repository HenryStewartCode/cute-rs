// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/mma_traits_sm107.hpp and include/cute/arch/mma_sm107_umma.hpp
//   (the SM107 MMA operations the trivial tiled MMAs choose among, and `with(b_reuse)`,
//   the B collector variants behind with_bkeep and with_breuse),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports. The epilogue tile table of
// SM107's NVFP4 GEMMs is a table of numbers CuTe DSL 4.8.1 documents by its
// behaviour (`cutlass.utils.rubin_helpers`); its code is not copied.

//! CuTe DSL's Rubin (SM107) `cutlass.utils` helpers (`cutlass.utils.rubin_helpers`): the resource
//! architecture ([`canonicalize_rubin_resource_arch`]: SM107 has 576 TMEM
//! columns and 327 KiB of shared memory per block, see
//! [`Arch::smem_capacity_bytes`](crate::Arch::smem_capacity_bytes) and
//! [`get_max_tmem_alloc_cols`](crate::arch::get_max_tmem_alloc_cols)), the
//! epilogue tile of the NVFP4 tiles Rubin tunes ([`compute_epilogue_tile_shape`]),
//! the tiled MMAs with collector buffers ([`make_trivial_tiled_mma`],
//! [`make_blockscaled_trivial_tiled_mma`]) and the B-collector variants of a
//! Rubin tiled MMA ([`TiledMma::with_bkeep`], [`TiledMma::with_breuse`]),
//! and the sparse metadata layouts with Rubin's FP4 atom
//! ([`make_tensor_e_atom_layout`], [`make_sparse_gmem_layout_e`],
//! [`make_sparse_smem_layout_e`], [`make_sparse_tmem_layout_e`]).
//!
//! No installed assembler knows sm_107 (PTX 9.4), so the Rubin MMAs run on
//! the simulator and are checked as MLIR text only.

use super::LayoutEnum;
use crate::Arch;
use crate::atom::mma::{MmaAtom, MmaOp, TiledMma, make_tiled_mma};
use crate::ir::{CollectorOp, DType, UmmaDataKind};
use crate::layout::{ComposedLayout, IntTuple, Layout, Shape, Swizzle, Tiler, Tree, make_composed_layout};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::cpasync::CtaGroup;
use crate::nvgpu::tcgen05::{
    Field, MmaF16BF16Op, MmaTF32Op, OperandSource, SM107BlockScaledMmaMXF8F6F4Op, SM107MmaFP8Op, SM107MmaMXF4NVF4Op,
    tile_to_mma_shape,
};

/// The base architecture the Rubin resource maps (shared-memory and TMEM
/// capacity) are keyed by, for `arch` (CuTe DSL's
/// `canonicalize_rubin_resource_arch`): [`Arch::Sm107`] for `sm_107`,
/// `sm_107a` and `sm_107f`. Any other name, or one no architecture has,
/// is an error (CuTe DSL raises a `ValueError`).
pub fn canonicalize_rubin_resource_arch(arch: &str) -> Result<Arch, String> {
    match arch.parse::<Arch>() {
        Ok(a) if a.version() == (10, 7) => Ok(Arch::Sm107),
        _ => Err(format!("canonicalize_rubin_resource_arch: `{arch}` is not a Rubin resource architecture (sm_107)")),
    }
}

/// The epilogue tiler of a Rubin GEMM (CuTe DSL's Rubin
/// `compute_epilogue_tile_shape(mma_op, cta_tile_shape, use_2cta_instrs,
/// layout_d, elem_ty_d, layout_c=, elem_ty_c=)`): the tile SM107 tunes for
/// its MXF4NVF4 MMA on a 128 or 256 by 256 CTA tile of a CTA pair, as
/// `(m:1, n:1)`: 32-element scale factors give `(128, 64)` for both CTA
/// heights, 16-element ones `(128, 32)` for 128 rows and `(128, 64)` for
/// 256. Any other MMA or tile is
/// [`compute_epilogue_tile_shape`](super::compute_epilogue_tile_shape)'s
/// (SM100's).
#[track_caller]
pub fn compute_epilogue_tile_shape(
    mma_op: &MmaOp,
    cta_tile_shape: &[usize],
    use_2cta_instrs: bool,
    layout_d: LayoutEnum,
    elem_ty_d: DType,
    layout_c: Option<LayoutEnum>,
    elem_ty_c: Option<DType>,
) -> Tiler {
    if let MmaOp::Sm100(kind) = mma_op
        && kind.sm107.is_some()
        && kind.kind == UmmaDataKind::Mxf4Nvf4
        && let Some(scale) = kind.scale
        && use_2cta_instrs
        && cta_tile_shape.len() >= 2
        && kind.a.bits() == 4
        && kind.b.bits() == 4
        && cta_tile_shape[1] == 256
    {
        let tile_n = match (scale.vec, cta_tile_shape[0]) {
            (32, 128 | 256) | (16, 256) => Some(64),
            (16, 128) => Some(32),
            _ => None,
        };
        if let Some(tile_n) = tile_n {
            return Tree::Branch(vec![Tree::Leaf(Some(Layout::new(128, 1))), Tree::Leaf(Some(Layout::new(tile_n, 1)))]);
        }
    }
    super::compute_epilogue_tile_shape(cta_tile_shape, use_2cta_instrs, layout_d, elem_ty_d, layout_c, elem_ty_c, None)
}

/// The tiled Rubin tcgen05 MMA for the operand types (CuTe DSL's Rubin
/// `make_trivial_tiled_mma(a_dtype, b_dtype, a_leading_mode,
/// b_leading_mode, acc_dtype, cta_group, mma_tiler, atom_layout_mnk,
/// permutation_mnk, a_source, a_collector_op, b_collector_op)`): the atom of
/// instruction shape `mma_tiler` (M, N, K) tiled over `atom_layout_mnk`.
///
/// | A and B                                  | operation              | K        |
/// |------------------------------------------|------------------------|----------|
/// | Float16 or BFloat16, the same            | [`MmaF16BF16Op`]       | 16       |
/// | TFloat32 or Float32, the same            | [`MmaTF32Op`]          | 8        |
/// | Float8E4M3FN or Float8E5M2 (each)        | [`SM107MmaFP8Op`]      | 32 or 64 |
///
/// The 16-bit and TF32 MMAs have no collector buffers: a collector op
/// other than `DISCARD` for them panics (CuTe DSL's `TypeError`), as does
/// any other pair of types.
#[track_caller]
pub fn make_trivial_tiled_mma(
    a_dtype: DType,
    b_dtype: DType,
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    acc_dtype: DType,
    cta_group: CtaGroup,
    mma_tiler: (usize, usize, usize),
    atom_layout_mnk: (usize, usize, usize),
    permutation_mnk: Option<[Option<Layout>; 3]>,
    a_source: OperandSource,
    a_collector_op: CollectorOp,
    b_collector_op: CollectorOp,
) -> TiledMma {
    use DType::*;
    let what = "make_trivial_tiled_mma";
    let no_collectors = |name: &str| {
        assert!(
            a_collector_op == CollectorOp::DISCARD && b_collector_op == CollectorOp::DISCARD,
            "{what}: TypeError: the {name} Rubin MMA does not support collector ops"
        );
    };
    let atom: MmaAtom = match (a_dtype, b_dtype) {
        (F16 | BF16, _) if a_dtype == b_dtype => {
            no_collectors("F16/BF16");
            MmaF16BF16Op::new(a_dtype, acc_dtype, mma_tiler, cta_group, a_source, a_leading_mode, b_leading_mode).into()
        }
        (TF32 | F32, _) if a_dtype == b_dtype => {
            no_collectors("TF32");
            MmaTF32Op::new(mma_tiler, cta_group, a_source, a_leading_mode, b_leading_mode).into()
        }
        (F8E4M3FN | F8E5M2, F8E4M3FN | F8E5M2) => SM107MmaFP8Op::new(
            a_dtype,
            b_dtype,
            acc_dtype,
            mma_tiler,
            cta_group,
            a_source,
            a_leading_mode,
            b_leading_mode,
            a_collector_op,
            b_collector_op,
        )
        .into(),
        _ => panic!(
            "{what}: TypeError: unsupported a_dtype and b_dtype, got {a_dtype} and {b_dtype} (the same Float16, \
             BFloat16 or TFloat32/Float32 type for both, or Float8E4M3FN and Float8E5M2)"
        ),
    };
    tile(atom, atom_layout_mnk, permutation_mnk)
}

fn tile(atom: MmaAtom, atom_layout_mnk: (usize, usize, usize), permutation: Option<[Option<Layout>; 3]>) -> TiledMma {
    let mma = make_tiled_mma(atom, Layout::col_major(atom_layout_mnk));
    match permutation {
        Some(p) => mma.with_permutation_mnk(p),
        None => mma,
    }
}

/// The tiled Rubin block-scaled tcgen05 MMA (CuTe DSL's Rubin
/// `make_blockscaled_trivial_tiled_mma(a_dtype, b_dtype, a_leading_mode,
/// b_leading_mode, sf_dtype, sf_vec_size, cta_group, mma_inst_shape,
/// a_source, a_collector_op, b_collector_op, atom_layout_mnk,
/// permutation_mnk)`):
///
/// | A and B                                    | operation                           | K   |
/// |--------------------------------------------|-------------------------------------|-----|
/// | Float4E2M1FN, both                         | [`SM107MmaMXF4NVF4Op`]              | 128 |
/// | Float4/6/8 formats (each)                  | [`SM107BlockScaledMmaMXF8F6F4Op`]   | 64  |
///
/// Panics (`TypeError`) for other types.
#[track_caller]
pub fn make_blockscaled_trivial_tiled_mma(
    a_dtype: DType,
    b_dtype: DType,
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    sf_dtype: DType,
    sf_vec_size: usize,
    cta_group: CtaGroup,
    mma_inst_shape: (usize, usize, usize),
    a_source: OperandSource,
    a_collector_op: CollectorOp,
    b_collector_op: CollectorOp,
    atom_layout_mnk: (usize, usize, usize),
    permutation_mnk: Option<[Option<Layout>; 3]>,
) -> TiledMma {
    use DType::*;
    let wide = |t: DType| matches!(t, F4E2M1FN | F6E3M2FN | F6E2M3FN | F8E4M3FN | F8E5M2);
    let atom: MmaAtom = match (a_dtype, b_dtype) {
        (F4E2M1FN, F4E2M1FN) => SM107MmaMXF4NVF4Op::new(
            sf_dtype,
            sf_vec_size,
            mma_inst_shape,
            cta_group,
            a_source,
            a_collector_op,
            b_collector_op,
        )
        .into(),
        (a, b) if wide(a) && wide(b) => SM107BlockScaledMmaMXF8F6F4Op::new(
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
            mma_inst_shape,
            cta_group,
            a_source,
            a_leading_mode,
            b_leading_mode,
            a_collector_op,
            b_collector_op,
        )
        .into(),
        _ => panic!(
            "make_blockscaled_trivial_tiled_mma: TypeError: unsupported a_dtype/b_dtype, got a_dtype: {a_dtype} and \
             b_dtype: {b_dtype}"
        ),
    };
    tile(atom, atom_layout_mnk, permutation_mnk)
}

impl TiledMma {
    /// This Rubin tiled MMA with B's collector op `collector_op` (CuTe DSL's
    /// `TiledMmaWrapper._make_variant`), the other fields copied.
    #[track_caller]
    fn with_b_collector(&self, collector_op: CollectorOp, accumulate: Option<bool>, what: &str) -> TiledMma {
        let MmaOp::Sm100(mut kind) = self.atom().op() else {
            panic!("cute: {what}: the tiled MMA {:?} is not a tcgen05 MMA", self.atom().op())
        };
        let Some(sm107) = kind.sm107 else {
            panic!(
                "cute: {what}: the tiled MMA {:?} is not a Rubin (sm_107) MMA, which has collector buffers",
                kind.ptx()
            )
        };
        kind.sm107 = Some(crate::ir::Sm107Umma { collector_b: collector_op, ..sm107 });
        let mut mma = self.with_op(MmaOp::Sm100(kind));
        if let Some(acc) = accumulate {
            mma.set(Field::ACCUMULATE, acc);
        }
        mma
    }

    /// A copy whose MMAs keep B in the collector buffer for later ones
    /// (CuTe DSL's `with_bkeep(accumulate=None)`): `b_collector_op =
    /// FILL`; with `accumulate`, also sets
    /// [`Field::ACCUMULATE`].
    /// Panics unless this is a Rubin tcgen05 MMA.
    #[track_caller]
    pub fn with_bkeep(&self, accumulate: Option<bool>) -> TiledMma {
        self.with_b_collector(CollectorOp::FILL, accumulate, "with_bkeep")
    }

    /// A copy whose MMAs consume the B the collector buffer kept (CuTe
    /// DSL's `with_breuse(accumulate=None)`): `b_collector_op = LASTUSE`,
    /// which also releases the buffer. Use the atom's
    /// [`with_collectors`](crate::nvgpu::tcgen05::SM107MmaFP8Op::with_collectors)
    /// for the middle `USE` form.
    #[track_caller]
    pub fn with_breuse(&self, accumulate: Option<bool>) -> TiledMma {
        self.with_b_collector(CollectorOp::LASTUSE, accumulate, "with_breuse")
    }
}

/// The metadata atom for raw A of `a_raw_dtype` on Rubin (CuTe DSL's Rubin
/// `make_tensor_e_atom_layout`): `(128, 128)` row-major for the 8-bit types
/// **and Float4E2M1FN** (SM100's FP4 atom is `(128, 256)`), the SM100
/// interleavings for TFloat32 and the 16-bit types. Panics for another type.
#[track_caller]
pub fn make_tensor_e_atom_layout(a_raw_dtype: DType) -> Layout {
    use DType::*;
    match a_raw_dtype {
        I8 | U8 | F8E4M3FN | F8E5M2 | F4E2M1FN => Layout::new((128, 128), (128, 1)),
        TF32 | F32 | F16 | BF16 => super::sm100::make_tensor_e_atom_layout(a_raw_dtype),
        t => panic!("cute: make_tensor_e_atom_layout: no sparse metadata atom for A of {}", t.cute_name()),
    }
}

/// The global-memory layout of the metadata of an `(M, N, K, L)` problem
/// (CuTe DSL's Rubin `make_sparse_gmem_layout_e`): M and K rounded up to
/// whole [`make_tensor_e_atom_layout`] atoms, tiled M-block first, then
/// K-blocks, then batches.
#[track_caller]
pub fn make_sparse_gmem_layout_e(problem_shape_mnkl: (usize, usize, usize, usize), a_raw_dtype: DType) -> Layout {
    let (m, _, k, l) = problem_shape_mnkl;
    let atom = make_tensor_e_atom_layout(a_raw_dtype);
    let (am, ak) = (atom.mode(0).size(), atom.mode(1).size());
    let cosize = atom.cosize();
    let (m_al, k_al) = (m.div_ceil(am) * am, k.div_ceil(ak) * ak);
    let mode = |a: &Layout, n: usize, stride: usize| Layout::from_modes([a.clone(), Layout::new(n, stride)]);
    Layout::from_modes([
        mode(&atom.mode(0), m_al / am, cosize),
        mode(&atom.mode(1), k_al / ak, m_al * ak),
        Layout::new(l, m_al * k_al),
    ])
}

/// The staged shared-memory layout of the metadata on Rubin (CuTe DSL's
/// Rubin `make_sparse_smem_layout_e(tiled_mma, mma_tiler_e_mnk,
/// a_raw_dtype, num_stages)`): the [`make_tensor_e_atom_layout`] atom under
/// the identity swizzle `Swizzle(0, 4, 3)`, tiled K-block first over `(M,
/// 384, num_stages)` (three 128-wide K tiles of metadata per stage; M is
/// `mma_tiler_e_mnk.0`). `tiled_mma` is not read (as in CuTe DSL).
#[track_caller]
pub fn make_sparse_smem_layout_e(
    _tiled_mma: &TiledMma,
    mma_tiler_e_mnk: (usize, usize, usize),
    a_raw_dtype: DType,
    num_stages: usize,
) -> ComposedLayout {
    let e_atom = make_tensor_e_atom_layout(a_raw_dtype);
    let (am, ak) = (e_atom.mode(0).size(), e_atom.mode(1).size());
    let m = mma_tiler_e_mnk.0;
    assert!(
        m.is_multiple_of(am) && 384usize.is_multiple_of(ak),
        "cute: make_sparse_smem_layout_e: M = {m} and K = 384 are not whole {am} x {ak} metadata atoms"
    );
    let atom = make_composed_layout(Swizzle::new(0, 4, 3), 0, e_atom);
    let shape = Tree::tuple([
        Tree::tuple([IntTuple::from(am), IntTuple::from(ak)]),
        IntTuple::from(m / am),
        IntTuple::from(384 / ak),
        IntTuple::from(num_stages),
    ]);
    tile_to_mma_shape(&atom, Shape::from(&shape), Some(&[0, 1, 2]))
}

/// The TMEM layout of the metadata on Rubin: SM100's
/// [`make_sparse_tmem_layout_e`](super::sm100::make_sparse_tmem_layout_e)
/// (the same `tmem_e_frg` atoms; for FP4, 128 rows by 128 K).
#[track_caller]
pub fn make_sparse_tmem_layout_e(tmem_shape: impl Into<Shape>, a_raw_dtype: DType) -> Layout {
    super::sm100::make_sparse_tmem_layout_e(tmem_shape, a_raw_dtype)
}
