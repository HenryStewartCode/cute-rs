// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The tile shapes and the choices of the TMEM load, the stmatrix store and the epilogue's
// shared-memory atom worked here follow NVIDIA CUTLASS 4.8.1's
//   include/cutlass/epilogue/collective/builders/sm100_builder.inl,
// and the mainloop the epilogues drain is the example port of
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py and fp16_gemm_1.py,
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file is cute-rs's
// own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! The SM100 epilogue helpers.
//!
//! * host tests of the TMEM load, the shared-memory store, the epilogue's atom and its staged
//!   layout over the matrix of D types, majors and CTA counts, against values worked from the
//!   C++ builder, and the diagnostics;
//! * the accumulators of the `blackwell_fp16_gemm` example (1-CTA and CTA-pair mainloops)
//!   drained through `epilogue_tma_store` and through the predicated `epilogue`, for Float16,
//!   BFloat16 and Float32 D in both majors, with an identity and a scaling `epilogue_op`, bit
//!   for bit against the CPU (which checks that the stmatrix fragments match the TMEM load's
//!   thread-value layout);
//! * the sm_100a PTX, assembled by `ptxas` when it is on PATH.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/blackwell_fp16_gemm.rs"]
mod blackwell_fp16_gemm;

use blackwell_fp16_gemm::{EpilogueCtx, FP16_GEMM_0, FP16_GEMM_1, Fp16GemmConfig, small_ints};
use cute::cpasync;
use cute::memory::SmemPartition;
use cute::pipeline::{Agent, CooperativeGroup, PipelineTmaStore};
use cute::prelude::*;
use cute::tcgen05::{Pack, Repetition, SmemLayoutAtomKind};
use cute::utils::compute_epilogue_tile_shape;
use cute::utils::gemm::sm100::{EpilogueConfig, epilogue, epilogue_tma_store};
use cute::utils::sm100::{
    get_smem_layout_atom_epi, get_tmem_load_op, make_smem_layout_epi, select_smem_store_op, select_tmem_load_op,
};

// ---------------------------------------------------------------------------------------
// Host: the choices.
// ---------------------------------------------------------------------------------------

/// `(m, n)` as the epilogue tiler.
fn tile(m: usize, n: usize) -> Tiler {
    Tiler::from(&IntTuple::from((m, n)))
}

fn shape_of(op: CopyOp) -> (cute::ir::TmemShape, u32, bool) {
    match op {
        CopyOp::TmemLoad { shape, num, pack } => (shape, num, pack),
        other => panic!("not a TMEM load: {other:?}"),
    }
}

/// The TMEM load for a CTA tile of `(128, 256)` (`(64, 256)` per CTA of a pair) and an
/// epilogue tile.
fn load_for(
    layout: LayoutEnum,
    d: DType,
    acc: DType,
    epi: (usize, usize),
    two_cta: bool,
    cta_m: usize,
) -> (cute::ir::TmemShape, u32, bool) {
    let c = select_tmem_load_op(&[cta_m, 256, 64], layout, d, acc, &tile(epi.0, epi.1), two_cta, None);
    shape_of(c.op())
}

/// CuTe C++'s `sm100_get_tmem_load_op`, worked by hand for a warp's `(32, 32)` part of the
/// tile (`epi (128, 32)` on 4x1 warps: 32 datapaths, 1024 bits of Float32 columns), and for
/// `(16, 32)` parts (`epi (64, 32)`: 16 datapaths).
#[test]
fn tmem_load_choice_matches_the_c_plus_plus_builder() {
    use LayoutEnum::{COL_MAJOR as M, ROW_MAJOR as N};
    use cute::ir::TmemShape::*;
    let (f16, bf16, f32, i8) = (DType::F16, DType::BF16, DType::F32, DType::I8);
    // 32 datapaths (epi tile 128 x 32 on four warps along M, 1024 bits of columns).
    let e = (128, 32);
    assert_eq!(load_for(N, f32, f32, e, false, 128), (S32x32b, 32, false));
    assert_eq!(load_for(M, f32, f32, e, false, 128), (S32x32b, 32, false));
    assert_eq!(load_for(N, f16, f32, e, false, 128), (S32x32b, 32, false));
    assert_eq!(load_for(N, bf16, f32, e, false, 128), (S32x32b, 32, false));
    assert_eq!(load_for(M, f16, f32, e, false, 128), (S16x256b, 4, false));
    assert_eq!(load_for(M, bf16, f32, e, false, 128), (S16x256b, 4, false));
    assert_eq!(load_for(N, i8, f32, e, false, 128), (S32x32b, 32, false));
    assert_eq!(load_for(M, i8, f32, e, false, 128), (S16x256b, 4, false));
    // 16-bit accumulators are packed: 512 bits of columns.
    assert_eq!(load_for(N, f16, f16, e, false, 128), (S32x32b, 16, true));
    assert_eq!(load_for(M, f16, f16, e, false, 128), (S16x128b, 4, true));
    // 16 datapaths (epi tile 64 x 64 on four warps along M: 2048 bits of columns).
    let e = (64, 64);
    assert_eq!(load_for(N, f32, f32, e, false, 128), (S16x128b, 16, false));
    assert_eq!(load_for(M, f32, f32, e, false, 128), (S16x256b, 8, false));
    assert_eq!(load_for(N, f16, f32, e, false, 128), (S16x256b, 8, false));
    assert_eq!(load_for(M, f16, f32, e, false, 128), (S16x256b, 8, false));
    assert_eq!(load_for(N, i8, f32, e, false, 128), (S16x32bx2, 32, false));
    assert_eq!(load_for(M, i8, f32, e, false, 128), (S16x256b, 8, false));
    assert_eq!(load_for(N, f16, f16, e, false, 128), (S16x128b, 8, true));
    assert_eq!(load_for(M, f16, f16, e, false, 128), (S16x128b, 8, true));
    // The widest repetition the columns allow: 8192 bits of Float32 columns.
    assert_eq!(load_for(N, f16, f32, (128, 256), false, 128), (S32x32b, 128, false));
    assert_eq!(load_for(M, f16, f32, (128, 256), false, 128), (S16x256b, 32, false));
    assert_eq!(load_for(N, f32, f32, (64, 256), false, 128), (S16x128b, 64, false));
    // A CTA pair with 64 rows per CTA takes (2, 2) warps: a (64, 64) tile gives each warp
    // (32, 32), and a (32, 64) one 16 rows.
    assert_eq!(load_for(N, f16, f32, (64, 64), true, 64), (S32x32b, 32, false));
    assert_eq!(load_for(M, f16, f32, (64, 64), true, 64), (S16x256b, 4, false));
    assert_eq!(load_for(N, f32, f32, (32, 64), true, 64), (S16x128b, 8, false));
    // 6- and 4-bit D need 32 datapaths and use 32x32b.
    assert_eq!(load_for(N, DType::F6E2M3FN, f32, (128, 32), false, 128), (S32x32b, 32, false));
    assert_eq!(load_for(M, DType::F4E2M1FN, f32, (128, 32), false, 128), (S32x32b, 32, false));
    // The atom carries the type and the repetition.
    let atom = get_tmem_load_op::<f32>(&[128, 256, 64], N, DType::F16, &tile(128, 32), false, None);
    assert_eq!(atom.op(), CopyOp::TmemLoad { shape: S32x32b, num: 32, pack: false });
    let atom = get_tmem_load_op::<F16>(&[128, 256, 64], M, DType::F16, &tile(128, 32), false, None);
    assert_eq!(atom.op(), CopyOp::TmemLoad { shape: S16x128b, num: 4, pack: true });
    let _ = (Pack::None, Repetition::X1);
}

/// `epilogue_tile` as `compute_epilogue_tile_shape` gives it, with the warps along N: the
/// layout's mode N is `(tile_n / 2, 2):(1, cta_n / 2)`, and each warp takes `tile_n / 2`.
#[test]
fn tmem_load_takes_the_tile_split_by_the_warp_layout() {
    let epi = compute_epilogue_tile_shape(&[64, 256, 64], true, LayoutEnum::ROW_MAJOR, DType::F16, None, None, None);
    // (2, 2) warps over a (64, 64) tile: (32, 32) each, 1024 bits of columns.
    let c = select_tmem_load_op(&[64, 256, 64], LayoutEnum::ROW_MAJOR, DType::F16, DType::F32, &epi, true, None);
    assert_eq!(shape_of(c.op()), (cute::ir::TmemShape::S32x32b, 32, false));
    // The same tile with the warps along M only has 16 datapaths.
    let epi =
        compute_epilogue_tile_shape(&[64, 256, 64], true, LayoutEnum::ROW_MAJOR, DType::F16, None, None, Some((4, 1)));
    let c =
        select_tmem_load_op(&[64, 256, 64], LayoutEnum::ROW_MAJOR, DType::F16, DType::F32, &epi, true, Some((4, 1)));
    assert_eq!(shape_of(c.op()).0, cute::ir::TmemShape::S16x256b);
}

fn store_for(layout: LayoutEnum, d_bits: usize, acc_bits: usize, load: CopyOp) -> CopyOp {
    select_smem_store_op(layout, d_bits, acc_bits, &load)
}

fn ld(shape: cute::ir::TmemShape, num: u32, pack: bool) -> CopyOp {
    CopyOp::TmemLoad { shape, num, pack }
}

/// The stmatrix choice of CuTe C++'s `sm100_get_smem_store_op`.
#[test]
fn smem_store_choice_matches_the_c_plus_plus_builder() {
    use LayoutEnum::{COL_MAJOR as M, ROW_MAJOR as N};
    use cute::ir::TmemShape::*;
    let stm = |num, trans| CopyOp::StMatrix { num, trans };
    let stm8 = |num| CopyOp::StMatrix16x8x8b { num, trans: true };
    // 32-bit accumulators, 16-bit D: any repeated 16x256b load; x1 has two matrices' values.
    for n in [2, 4, 8, 16, 32] {
        assert_eq!(store_for(N, 16, 32, ld(S16x256b, n, false)), stm(4, false));
        assert_eq!(store_for(M, 16, 32, ld(S16x256b, n, false)), stm(4, true));
    }
    assert_eq!(store_for(N, 16, 32, ld(S16x256b, 1, false)), stm(2, false));
    assert_eq!(store_for(M, 16, 32, ld(S16x256b, 1, false)), stm(2, true));
    // 32-bit D: N-major with 16x128b; M-major (and 32x32b loads) have no stmatrix.
    assert_eq!(store_for(N, 32, 32, ld(S16x128b, 16, false)), stm(4, false));
    assert_eq!(store_for(N, 32, 32, ld(S16x128b, 1, false)), stm(2, false));
    assert_eq!(store_for(M, 32, 32, ld(S16x256b, 8, false)), CopyOp::Universal { bits: 32 });
    assert_eq!(store_for(N, 32, 32, ld(S32x32b, 32, false)), CopyOp::Universal { bits: 32 });
    // 16-bit D from 32x32b: plain.
    assert_eq!(store_for(N, 16, 32, ld(S32x32b, 32, false)), CopyOp::Universal { bits: 16 });
    // 16-bit accumulators: packed 16x128b only.
    assert_eq!(store_for(N, 16, 16, ld(S16x128b, 8, true)), stm(4, false));
    assert_eq!(store_for(M, 16, 16, ld(S16x128b, 1, true)), stm(2, true));
    assert_eq!(store_for(N, 16, 16, ld(S16x128b, 8, false)), CopyOp::Universal { bits: 16 });
    assert_eq!(store_for(N, 16, 16, ld(S32x32b, 8, true)), CopyOp::Universal { bits: 16 });
    // 8-bit D, M-major: 16x256b loads, 4, 2 or 1 matrices of 16x8 bytes.
    assert_eq!(store_for(M, 8, 32, ld(S16x256b, 8, false)), stm8(4));
    assert_eq!(store_for(M, 8, 32, ld(S16x256b, 4, false)), stm8(4));
    assert_eq!(store_for(M, 8, 32, ld(S16x256b, 2, false)), stm8(2));
    assert_eq!(store_for(M, 8, 32, ld(S16x256b, 1, false)), stm8(1));
    assert_eq!(store_for(N, 8, 32, ld(S16x32bx2, 32, false)), CopyOp::Universal { bits: 8 });
    // Not a TMEM load: plain.
    assert_eq!(store_for(N, 16, 32, CopyOp::Universal { bits: 32 }), CopyOp::Universal { bits: 16 });
}

/// The atom of `get_smem_layout_atom_epi` and the layout of `make_smem_layout_epi`.
#[test]
fn epilogue_smem_layouts() {
    use LayoutEnum::{COL_MAJOR as M, ROW_MAJOR as N};
    use SmemLayoutAtomKind::*;
    let e = tile(128, 32);
    // N-major: the N extent (32) in bits: 512 for 16-bit, 1024 for 32-bit.
    assert_eq!(get_smem_layout_atom_epi(N, DType::F16, &e), K_SW64);
    assert_eq!(get_smem_layout_atom_epi(N, DType::BF16, &e), K_SW64);
    assert_eq!(get_smem_layout_atom_epi(N, DType::F32, &e), K_SW128);
    assert_eq!(get_smem_layout_atom_epi(N, DType::I8, &e), K_SW32);
    // M-major: the M extent (128): 2048 bits for 16-bit, 4096 for 32-bit (the 32-bit atom).
    assert_eq!(get_smem_layout_atom_epi(M, DType::F16, &e), MN_SW128);
    assert_eq!(get_smem_layout_atom_epi(M, DType::F32, &e), MN_SW128_32B);
    assert_eq!(get_smem_layout_atom_epi(M, DType::I8, &e), MN_SW128);
    assert_eq!(get_smem_layout_atom_epi(N, DType::F16, &tile(64, 64)), K_SW128);
    assert_eq!(get_smem_layout_atom_epi(N, DType::F16, &tile(128, 8)), K_INTER);
    // The warps' slices along N are apart in the layout tiler, so only a warp's share counts.
    let epi = compute_epilogue_tile_shape(&[64, 256, 64], true, LayoutEnum::ROW_MAJOR, DType::F16, None, None, None);
    assert_eq!(get_smem_layout_atom_epi(N, DType::F16, &epi), K_SW64);

    // The staged layout: (M, N, stage), the atom's swizzle, N contiguous for N-major C.
    for (layout, dtype, stages) in [(N, DType::F16, 2), (M, DType::F16, 3), (N, DType::F32, 2), (M, DType::F32, 1)] {
        let l = make_smem_layout_epi(dtype, layout, &e, stages);
        assert_eq!(l.size(), 128 * 32 * stages, "{layout:?} {dtype}");
        assert_eq!(l.cosize(), 128 * 32 * stages, "{layout:?} {dtype}");
        assert_eq!(l.shape().mode(2).size(), stages);
        let at = |i: usize, j: usize| l.call(i + 128 * j);
        // The contiguous direction: N for N-major C, M for M-major.
        if layout == N {
            assert_eq!(at(0, 1) % 8, 1);
        } else {
            assert_eq!(at(1, 0) % 8, 1);
        }
    }
    assert_eq!(make_smem_layout_epi(DType::Bool, N, &e, 1).size(), 128 * 32);
}

/// Diagnostics.
#[test]
fn epilogue_helper_diagnostics() {
    let msg = |f: &dyn Fn()| {
        let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a panic");
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    };
    // 8 datapaths per warp: a 32-row tile over four warps.
    let m = msg(&|| {
        select_tmem_load_op(&[128, 256, 64], LayoutEnum::ROW_MAJOR, DType::F16, DType::F32, &tile(32, 64), false, None);
    });
    assert!(m.contains("has 8 datapaths") && m.contains("16 or 32"), "{m}");
    // A tile the warp layout does not divide.
    let m = msg(&|| {
        select_tmem_load_op(&[64, 256, 64], LayoutEnum::ROW_MAJOR, DType::F16, DType::F32, &tile(64, 33), true, None);
    });
    assert!(m.contains("does not divide the epilogue tile 64x33"), "{m}");
    // Narrow D with 16 datapaths.
    let m = msg(&|| {
        select_tmem_load_op(
            &[128, 256, 64],
            LayoutEnum::ROW_MAJOR,
            DType::F4E2M1FN,
            DType::F32,
            &tile(64, 64),
            false,
            None,
        );
    });
    assert!(m.contains("needs 32 datapaths"), "{m}");
    // A type pair no instruction serves.
    let m = msg(&|| {
        select_tmem_load_op(
            &[128, 256, 64],
            LayoutEnum::ROW_MAJOR,
            DType::F32,
            DType::F16,
            &tile(128, 32),
            false,
            None,
        );
    });
    assert!(m.contains("no TMEM load serves"), "{m}");
    // The epilogue tile has two modes.
    let m = msg(&|| {
        get_smem_layout_atom_epi(LayoutEnum::ROW_MAJOR, DType::F16, &Tiler::from(128usize));
    });
    assert!(m.contains("two modes"), "{m}");
}

// ---------------------------------------------------------------------------------------
// Simulator: the helpers draining the example's accumulators.
// ---------------------------------------------------------------------------------------

/// Host D types.
trait HostD: Numeric + Copy + PartialEq + std::fmt::Debug {
    fn from_f(x: f32) -> Self;
    fn to_f(self) -> f32;
}

impl HostD for F16 {
    fn from_f(x: f32) -> F16 {
        F16::from_f32(x)
    }
    fn to_f(self) -> f32 {
        self.to_f32()
    }
}

impl HostD for BF16 {
    fn from_f(x: f32) -> BF16 {
        BF16::from_f32(x)
    }
    fn to_f(self) -> f32 {
        self.to_f32()
    }
}

impl HostD for f32 {
    fn from_f(x: f32) -> f32 {
        x
    }
    fn to_f(self) -> f32 {
        self
    }
}

/// How an epilogue is run.
#[derive(Clone, Copy, Debug)]
struct Case {
    layout: LayoutEnum,
    /// A TMA store (`epilogue_tma_store`) or direct stores (`epilogue`).
    tma: bool,
    /// Multiply by 2 in the epilogue operation.
    scale: bool,
    /// Direct stores only: C is `(ragged_m, ragged_n)` smaller than the MMA tiles.
    ragged: bool,
    /// The epilogue tile `(M, N)`, instead of `compute_epilogue_tile_shape`'s.
    epi: Option<(usize, usize)>,
    /// Direct stores only: `overlapping_accum` (the stage from the phase, subtiles reversed on
    /// phase 0, an early release).
    overlap: bool,
}

/// The pieces of C a TMA-store epilogue needs from the host function.
struct TmaC<D> {
    c: Tensor<D>,
    atom: CopyAtom<D>,
    tma_tensor: Tensor<Ident>,
    smem_layout: Option<ComposedLayout>,
    epi_tile: Tiler,
    cfg: EpilogueConfig,
}

/// C `(mc, nc)` with the contiguous mode's extent a multiple of 16 elements.
fn declare_c<D: HostD>(layout: LayoutEnum, m: Int32, n: Int32, ragged: bool) -> Tensor<D> {
    if ragged {
        let (mc, nc) = if layout.is_m_major_c() {
            (param_extent("mc", 16), param_extent("nc", 1))
        } else {
            (param_extent("mc", 1), param_extent("nc", 16))
        };
        let l = if layout.is_m_major_c() { Layout::new((mc, nc), (1, mc)) } else { Layout::new((mc, nc), (nc, 1)) };
        param_tensor_aligned::<D>("c", l, 32)
    } else {
        let l = if layout.is_m_major_c() { Layout::new((m, n), (1, m)) } else { Layout::new((m, n), (n, 1)) };
        param_tensor_aligned::<D>("c", l, 32)
    }
}

/// The epilogue's configuration for the example's tiles.
fn epilogue_config<D: HostD>(cfg: &Fp16GemmConfig, layout: LayoutEnum, num_c_stage: usize) -> EpilogueConfig {
    let (tm, tn, tk) = cfg.mma_tiler_mnk;
    let two = cfg.atom_ctas() == 2;
    let mut e = EpilogueConfig::new((tm / cfg.atom_ctas(), tn, tk), layout, D::DTYPE, DType::F32, two);
    e.num_c_stage = num_c_stage;
    e
}

fn build_drain<D: HostD>(cfg: Fp16GemmConfig, case: Case, num_c_stage: usize) -> Kernel {
    let Case { layout, tma, scale, ragged, epi, overlap } = case;
    let make_c = move |cfg: &Fp16GemmConfig, _: &TiledMma, m: Int32, n: Int32| -> TmaC<D> {
        let c = declare_c::<D>(layout, m, n, ragged);
        let ecfg = epilogue_config::<D>(cfg, layout, num_c_stage);
        let (cm, cn, _) = ecfg.cta_tile_shape_mnk;
        let epi_tile = match epi {
            Some((em, en)) => tile(em, en),
            None => compute_epilogue_tile_shape(&[cm, cn], ecfg.use_2cta_instrs, layout, D::DTYPE, None, None, None),
        };
        if tma {
            let smem_layout = make_smem_layout_epi(D::DTYPE, layout, &epi_tile, num_c_stage);
            let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileS2GOp,
                &c,
                smem_layout.slice((.., .., 0)),
                epi_tile.clone(),
                1,
            )
            .into();
            TmaC { c, atom, tma_tensor, smem_layout: Some(smem_layout), epi_tile, cfg: ecfg }
        } else {
            // The direct epilogue needs neither: C's own coordinates are the tensor.
            let tma_tensor = make_identity_tensor(c.shape().clone());
            let atom = make_copy_atom::<D>(CopyOp::Universal { bits: D::DTYPE.bits().max(8) });
            TmaC { c, atom, tma_tensor, smem_layout: None, epi_tile, cfg: ecfg }
        }
    };
    let drain = move |ctx: &EpilogueCtx, c: &TmaC<D>| {
        let (tm, tn, _) = ctx.cfg.mma_tiler_mnk;
        // The accumulator with a stage mode: ((M, N), MMA_M, MMA_N, STAGE)
        let Tree::Branch(mut modes) = ctx.acc_shape.clone() else { panic!("a tuple shape") };
        modes.push(Shape::from(ctx.cfg.acc_stage));
        let staged = ctx.tiled_mma.make_fragment_c::<f32>(&Shape::from(Tree::Branch(modes)));
        let t_ct_acc = make_tensor(ctx.tmem_ptr.clone(), staged.layout().clone());
        let op = move |v: TensorSsa<D>| if scale { (v.to::<f32>() * 2.0f32).to::<D>() } else { v };
        let coord = [ctx.tile_m, ctx.tile_n];
        if tma {
            // (bM, bN, RestM, RestN)
            let g_c = local_tile(&c.tma_tensor, (tm, tn), (.., ..));
            // (MMA, MMA_M, MMA_N, RestM, RestN)
            let t_cg_c = ctx.thr_mma.partition_c(&g_c);
            let s_c = ctx.smem.allocate_tensor::<D>(
                c.smem_layout.clone().expect("a TMA store's layout"),
                1024,
                None,
                SmemPartition::User,
            );
            let c_pipeline = PipelineTmaStore::create(
                c.cfg.num_c_stage,
                CooperativeGroup::new(Agent::Thread, 32 * c.cfg.epilogue_warp_id.len() as i32),
            );
            epilogue_tma_store::<f32, D, _, _>(
                &c.cfg,
                ctx.tidx,
                ctx.warp_idx,
                &c.atom,
                &t_ct_acc,
                &s_c,
                &t_cg_c,
                &c.epi_tile,
                Int32::from(0),
                &op,
                &coord,
                &ctx.acc_consumer_state,
                ctx.acc_pipeline,
                &c_pipeline,
            );
            if_(ctx.warp_idx.eq(c.cfg.epilogue_warp_id[0] as i32), || c_pipeline.producer_tail());
        } else {
            let g_c = local_tile(&c.c, (tm, tn), (.., ..));
            let t_cg_c = ctx.thr_mma.partition_c(&g_c);
            let cc = local_tile(&c.tma_tensor, (tm, tn), (.., ..));
            let t_cc_c = ctx.thr_mma.partition_c(&cc);
            epilogue::<f32, D, _>(
                &c.cfg,
                ctx.tidx,
                &t_ct_acc,
                &t_cg_c,
                &c.epi_tile,
                &op,
                &coord,
                &ctx.acc_consumer_state,
                ctx.acc_pipeline,
                ragged.then_some(&t_cc_c),
                ragged.then_some(&c.c),
                overlap,
            );
        }
    };
    cfg.build_with(4, make_c, drain)
}

/// Simulate `cfg`'s mainloop with `case`'s epilogue on an `m x n x k` problem (`(mc, nc)` for
/// a ragged C) and check C bit for bit.
fn run_drain<D: HostD>(cfg: Fp16GemmConfig, case: Case, m: usize, n: usize, k: usize) -> SimReport {
    let (mc, nc) =
        if case.ragged { if case.layout.is_m_major_c() { (m - 16, n - 6) } else { (m - 6, n - 16) } } else { (m, n) };
    let kernel = build_drain::<D>(cfg, case, 2);
    let a = small_ints(m * k, 17);
    let b = small_ints(n * k, 23);
    let mut args = vec![KernelArg::scalar(m as i32), KernelArg::scalar(n as i32), KernelArg::scalar(k as i32)];
    args.extend([KernelArg::buffer(&a), KernelArg::buffer(&b)]);
    if case.ragged {
        args.extend([KernelArg::scalar(mc as i32), KernelArg::scalar(nc as i32)]);
    }
    let init = vec![D::from_f(-7.0); mc * nc];
    args.push(KernelArg::buffer(&init));
    let report = kernel.simulate(cfg.launch(m, n), &mut args).unwrap_or_else(|e| panic!("{case:?}: {e}"));
    let got = args.last().unwrap().to_vec::<D>();
    let at = |r: usize, c: usize| if case.layout.is_m_major_c() { c * mc + r } else { r * nc + c };
    let mut bad = 0;
    for r in 0..mc {
        for c in 0..nc {
            let acc: f32 = (0..k).map(|i| a[r * k + i].to_f32() * b[c * k + i].to_f32()).sum();
            let d = D::from_f(acc);
            let want = if case.scale { D::from_f(d.to_f() * 2.0) } else { d };
            if got[at(r, c)] != want {
                if bad < 3 {
                    eprintln!("{case:?} ({r},{c}): got {:?}, want {want:?}", got[at(r, c)]);
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{case:?} on {cfg:?}: {bad} elements differ");
    report
}

const N_MAJOR: LayoutEnum = LayoutEnum::ROW_MAJOR;
const M_MAJOR: LayoutEnum = LayoutEnum::COL_MAJOR;

/// The 1-CTA mainloop (`fp16_gemm_0`: two `(128, 256)` tiles, two k-tiles) drained with
/// `epilogue_tma_store`: Float16, BFloat16 and Float32 D in both majors, with an identity and
/// a scaling `epilogue_op`. Float16 and BFloat16 M-major D take `stmatrix.trans` from
/// `16x256b` loads, N-major ones plain stores from `32x32b` loads, Float32 N-major D
/// `stmatrix` of `16x128b` loads only on a 16-datapath tile (here plain stores).
fn tma_matrix(cfg: Fp16GemmConfig, m: usize, n: usize, k: usize) {
    for layout in [N_MAJOR, M_MAJOR] {
        for scale in [false, true] {
            let case = Case { layout, tma: true, scale, ragged: false, epi: None, overlap: false };
            run_drain::<F16>(cfg, case, m, n, k);
            run_drain::<BF16>(cfg, case, m, n, k);
            // The M-major Float32 tile takes the 128-byte swizzle with 32-byte atoms
            // (`MN_SW128_32B`, TMA's s128b_atom_32b).
            run_drain::<f32>(cfg, case, m, n, k);
        }
    }
}

#[test]
fn tma_store_epilogue_on_one_cta() {
    tma_matrix(FP16_GEMM_0, 128, 512, 128);
}

/// The M-major Float32 epilogue's TMA descriptor takes the
/// 128-byte swizzle of 32-byte atoms (`CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B`, code 4; tests/prims.rs
/// checks that targets before sm_100 refuse it).
#[test]
fn m_major_f32_tma_store_takes_the_32_byte_atom_swizzle() {
    let case = Case { layout: M_MAJOR, tma: true, scale: false, ragged: false, epi: None, overlap: false };
    let kernel = build_drain::<f32>(FP16_GEMM_0, case, 2);
    let swizzles: Vec<_> =
        kernel.ir().tensor_maps.iter().filter_map(|m| m.source.as_ref().map(|s| s.spec.swizzle)).collect();
    assert!(swizzles.contains(&cute::TensorMapSwizzle::s128b_atom_32b), "{swizzles:?}");
    assert_eq!(cute::TensorMapSwizzle::s128b_atom_32b as u32, 4);
    assert!(cute::validate::check_target(kernel.ir(), Arch::Sm100a).is_ok());
}

#[test]
fn tma_store_epilogue_on_a_cta_pair() {
    // Five A/B stages leave room for the C stages in shared memory.
    tma_matrix(Fp16GemmConfig { ab_stages: 5, ..FP16_GEMM_1 }, 256, 256, 128);
}

/// The predicated `epilogue` on ragged C (a few rows or columns short of the MMA tiles): the
/// stores past C's shape are skipped and nothing else changes, for the 1-CTA mainloop and the
/// CTA pair.
fn direct_matrix(cfg: Fp16GemmConfig, m: usize, n: usize, k: usize) {
    for layout in [N_MAJOR, M_MAJOR] {
        for scale in [false, true] {
            let case = Case { layout, tma: false, scale, ragged: true, epi: None, overlap: false };
            run_drain::<F16>(cfg, case, m, n, k);
            run_drain::<BF16>(cfg, case, m, n, k);
            run_drain::<f32>(cfg, case, m, n, k);
        }
    }
}

#[test]
fn direct_epilogue_on_one_cta() {
    direct_matrix(FP16_GEMM_0, 128, 512, 128);
}

#[test]
fn direct_epilogue_on_a_cta_pair() {
    direct_matrix(FP16_GEMM_1, 256, 256, 128);
}

/// Without predication (a full C), `epilogue` stores every element.
#[test]
fn direct_epilogue_without_predication() {
    for layout in [N_MAJOR, M_MAJOR] {
        let case = Case { layout, tma: false, scale: true, ragged: false, epi: None, overlap: false };
        run_drain::<F16>(FP16_GEMM_0, case, 128, 256, 64);
    }
}

/// A CTA pair whose MMA has M = 128 (64 rows per CTA) takes the epilogue's four warps as 2x2.
#[test]
fn epilogue_on_a_cta_pair_with_64_rows_per_cta() {
    let cfg = Fp16GemmConfig {
        mma_inst_shape_mnk: (128, 256, 16),
        mma_tiler_mnk: (128, 256, 64),
        name: "blackwell_fp16_gemm_pair_m128",
        ..FP16_GEMM_1
    };
    for layout in [N_MAJOR, M_MAJOR] {
        let case = Case { layout, tma: false, scale: true, ragged: true, epi: None, overlap: false };
        run_drain::<F16>(cfg, case, 128, 256, 64);
        let case = Case { layout, tma: true, scale: false, ragged: false, epi: None, overlap: false };
        run_drain::<F16>(cfg, case, 128, 256, 64);
    }
}

/// Sixteen datapaths per warp: a 1-CTA MMA of M = 64 keeps D in 16 lanes of each lane
/// quadrant, so a `(64, 64)` epilogue tile on four warps gives each 16 rows. The `16x256b`
/// and `16x128b` loads and their `stmatrix` stores (four matrices; two for a single
/// repetition, from an 8-column tile), Float16, BFloat16 and Float32 D, through shared memory
/// and directly.
#[test]
fn epilogue_with_16_datapaths_per_warp() {
    let cfg = Fp16GemmConfig {
        mma_inst_shape_mnk: (64, 256, 16),
        mma_tiler_mnk: (64, 256, 64),
        name: "blackwell_fp16_gemm_m64",
        ..FP16_GEMM_0
    };
    for layout in [N_MAJOR, M_MAJOR] {
        for epi in [None, Some((64, 8))] {
            let direct = Case { layout, tma: false, scale: true, ragged: true, epi, overlap: false };
            run_drain::<F16>(cfg, direct, 64, 256, 64);
            run_drain::<BF16>(cfg, direct, 64, 256, 64);
            run_drain::<f32>(cfg, direct, 64, 256, 64);
            let tma = Case { layout, tma: true, scale: false, ragged: false, epi, overlap: false };
            run_drain::<F16>(cfg, tma, 64, 256, 64);
            run_drain::<BF16>(cfg, tma, 64, 256, 64);
            if layout == N_MAJOR {
                run_drain::<f32>(cfg, tma, 64, 256, 64);
            }
        }
    }
    // Four Float32 columns per warp: a single `16x128b` repetition (the values of two
    // matrices only), loaded and stored directly (a shared-memory atom needs 8 columns).
    let narrow = Case { layout: N_MAJOR, tma: false, scale: true, ragged: true, epi: Some((64, 4)), overlap: false };
    run_drain::<f32>(cfg, narrow, 64, 256, 64);
}

// ---------------------------------------------------------------------------------------
// The sm_100a PTX.
// ---------------------------------------------------------------------------------------

fn compile_ptx(tc: &Toolchain, kernel: &Kernel) -> String {
    let name = kernel.ir().name.clone();
    kernel
        .compile_with(tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .ptx()
        .to_string()
}

/// Assemble `ptx` with `ptxas -arch=sm_100a` when `ptxas` is on PATH.
fn ptxas_check(name: &str, ptx: &str) {
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    let Some(ptxas) = ptxas else { return };
    let base = std::env::temp_dir().join(format!("cute-epilogue-helpers-{}-{name}", std::process::id()));
    let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
    std::fs::write(&src, ptx).unwrap();
    let out = std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&cubin);
    assert!(
        out.status.success(),
        "{name}: ptxas: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The epilogues' instructions: `tcgen05.ld` of each shape, `stmatrix` of each width and
/// transposition, the TMA store and the predicated vector stores.
#[test]
fn epilogue_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let m64 = Fp16GemmConfig {
        mma_inst_shape_mnk: (64, 256, 16),
        mma_tiler_mnk: (64, 256, 64),
        name: "blackwell_fp16_gemm_m64",
        ..FP16_GEMM_0
    };
    let tma = |layout, epi, scale| Case { layout, tma: true, scale, ragged: false, epi, overlap: false };
    let direct = |layout, ragged| Case { layout, tma: false, scale: true, ragged, epi: None, overlap: false };
    let kernels: Vec<(&str, Kernel, &[&str])> = vec![
        // 16x256b loads and four transposed stmatrix matrices, a TMA store of the tile.
        (
            "f16_m_major_tma",
            build_drain::<F16>(FP16_GEMM_0, tma(M_MAJOR, None, false), 2),
            &[
                "tcgen05.ld.sync.aligned.16x256b.x4.b32",
                "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                "fence.proxy.async.shared::cta",
                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group",
                "cp.async.bulk.commit_group",
            ],
        ),
        // 32x32b loads and plain shared stores for N-major Float16.
        (
            "f16_n_major_tma",
            build_drain::<F16>(FP16_GEMM_0, tma(N_MAJOR, None, true), 2),
            &["tcgen05.ld.sync.aligned.32x32b.x32.b32", "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"],
        ),
        // 16 datapaths: 16x128b with the 32-bit-element stmatrix (two matrices for x1), and 16x256b.
        (
            "f32_n_major_m64_tma",
            build_drain::<f32>(m64, tma(N_MAJOR, Some((64, 8)), false), 2),
            &["tcgen05.ld.sync.aligned.16x128b.x2.b32", "stmatrix.sync.aligned.m8n8.x4.shared.b16"],
        ),
        // One repetition of 16x256b has two matrices' values.
        (
            "f16_m_major_m64_tma_x1",
            build_drain::<F16>(m64, tma(M_MAJOR, Some((64, 8)), false), 2),
            &["tcgen05.ld.sync.aligned.16x256b.x1.b32", "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16"],
        ),
        (
            "bf16_n_major_m64_tma",
            build_drain::<BF16>(m64, tma(N_MAJOR, None, false), 2),
            &["tcgen05.ld.sync.aligned.16x256b.x8.b32", "stmatrix.sync.aligned.m8n8.x4.shared.b16"],
        ),
        // M-major Float32 through the s128b_atom_32b TMA store.
        (
            "f32_m_major_tma",
            build_drain::<f32>(FP16_GEMM_0, tma(M_MAJOR, None, false), 2),
            &["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"],
        ),
        // Straight to global memory, predicated.
        (
            "f32_n_major_direct",
            build_drain::<f32>(FP16_GEMM_0, direct(N_MAJOR, true), 2),
            &["tcgen05.ld.sync.aligned.32x32b.x32.b32", "st.global"],
        ),
        (
            "f16_m_major_direct",
            build_drain::<F16>(FP16_GEMM_1, direct(M_MAJOR, true), 2),
            &["tcgen05.ld.sync.aligned.16x256b.x4.b32", "st.global"],
        ),
    ];
    for (name, kernel, want) in kernels {
        let ptx = compile_ptx(&tc, &kernel);
        if let Some(dir) = std::env::var_os("CUTE_DUMP_PTX") {
            std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ptx")), &ptx).unwrap();
        }
        for w in want {
            // LLVM 21 prints stmatrix's shape after the count and transposition
            // (`.x4.trans.m8n8`), LLVM 22 before them (`.m8n8.x4.trans`).
            let alt = w.replace(".m8n8.x1", ".x1").replace(".m8n8.x2", ".x2").replace(".m8n8.x4", ".x4");
            let alt = alt.replace(".x1.", ".x1.m8n8.").replace(".x2.", ".x2.m8n8.").replace(".x4.", ".x4.m8n8.");
            let alt = alt.replace(".trans.m8n8.", ".trans.").replace(".m8n8.trans.", ".trans.m8n8.");
            assert!(
                ptx.contains(w) || (w.starts_with("stmatrix") && ptx.contains(&alt)),
                "{name}: no `{w}` in the PTX"
            );
        }
        ptxas_check(name, &ptx);
    }
}

/// `overlapping_accum`: the stage index is the state's phase, the first tile's subtiles are
/// drained backwards, and the stage is released after subtile 0.
#[test]
fn direct_epilogue_with_an_overlapping_accumulator() {
    let mut case = Case { layout: N_MAJOR, tma: false, scale: true, ragged: true, epi: None, overlap: true };
    run_drain::<F16>(FP16_GEMM_0, case, 128, 256, 64);
    case.layout = M_MAJOR;
    run_drain::<f32>(FP16_GEMM_0, case, 128, 256, 64);
}
