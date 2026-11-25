// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The round-trip kernel's structure (operands staged in shared memory, the 2:4 metadata in
// TMEM, a k-block loop of sparse MMAs, a commit to an mbarrier and a tcgen05.ld epilogue)
// follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py and
//   examples/python/CuTeDSL/utils/sparse_utils.py (the metadata codes),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! Rubin cutlass.utils helpers and the sm_107a TMEM load with sparse compression (roadmap
//! part m10-rubin-extras). Host tests of the helpers' choices; simulator kernels for
//! `tcgen05.ld{.red}.spcompress` over every row operation and the NaN option against plain
//! Rust; a compress-then-sparse-MMA round trip; a Rubin tiled MMA made by
//! `utils::sm107::make_trivial_tiled_mma` with B reuse (`with_bkeep`, `with_breuse`) running a
//! GEMM tile; the target rule; and the MLIR text. No installed toolchain assembles sm_107a, so
//! nothing here is assembled.

use cute::prelude::*;
use cute::tcgen05::{
    self, CollectorOp, CtaGroup, Field, Ld32x32bOp, LdRedSPCompress32x32bOp, LdSPCompress32x32bOp, MmaFP8SparseOp,
    OperandSource, Repetition, SM107MmaMXF4NVF4Op, SmemLayoutAtomKind, SparseMetadataFormat, St32x32bOp, TmemLoadRedOp,
};
use cute::utils::sm107;
use std::cmp::Ordering;

const LANE: usize = 1 << 16;
const OPS: [TmemLoadRedOp; 4] = [TmemLoadRedOp::Max, TmemLoadRedOp::Min, TmemLoadRedOp::MaxAbs, TmemLoadRedOp::MinAbs];

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// Test values: the special ones first, then random bit patterns and moderate values.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
        1.5,
        -1.5,
        3.0e-39,
    ];
    let mut r = lcg(seed);
    (0..n)
        .map(|i| {
            if i < specials.len() {
                specials[i]
            } else if r().is_multiple_of(5) {
                f32::from_bits((r() << 7 ^ r()) as u32)
            } else {
                (r() % 2001) as f32 / 100.0 - 10.0
            }
        })
        .collect()
}

fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// The reference of PTX's `min`/`max` reduction.
fn reduce(vals: &[f32], op: TmemLoadRedOp, nan: bool) -> f32 {
    let (abs, min) = (op.is_abs(), op.is_min());
    let mut acc: Option<f32> = None;
    for &v in vals {
        let v = if abs { v.abs() } else { v };
        acc = Some(match acc {
            None => v,
            Some(a) if a.is_nan() || v.is_nan() => {
                if nan || (a.is_nan() && v.is_nan()) {
                    f32::NAN
                } else if a.is_nan() {
                    v
                } else {
                    a
                }
            }
            Some(a) if a == v => {
                if min == a.is_sign_negative() {
                    a
                } else {
                    v
                }
            }
            Some(a) => {
                if (a < v) == min {
                    a
                } else {
                    v
                }
            }
        });
    }
    acc.unwrap()
}

/// Whether `a` is preferred to `b`: Greater if so.
fn prefer(a: f32, b: f32, op: TmemLoadRedOp, nan: bool) -> Ordering {
    let class = |v: f32| if v.is_nan() { if nan { 2 } else { 0 } } else { 1 };
    if class(a) != class(b) {
        return class(a).cmp(&class(b));
    }
    if a.is_nan() {
        return Ordering::Equal;
    }
    let (a, b) = if op.is_abs() { (a.abs(), b.abs()) } else { (a, b) };
    let o = a.total_cmp(&b);
    if op.is_min() { o.reverse() } else { o }
}

/// The compression of one row: the kept values and the packed 4-bit codes.
fn compress_row(row: &[f32], op: TmemLoadRedOp, nan: bool) -> (Vec<f32>, Vec<u32>) {
    let mut kept = Vec::new();
    let mut meta = vec![0u32; (row.len() / 32).max(1)];
    for (g, group) in row.chunks(4).enumerate() {
        let mut idx = [0usize, 1, 2, 3];
        idx.sort_by(|&i, &j| prefer(group[j], group[i], op, nan));
        let (a, b) = (idx[0].min(idx[1]), idx[0].max(idx[1]));
        kept.extend([group[a], group[b]]);
        meta[g / 8] |= ((a | b << 2) as u32) << (4 * (g % 8));
    }
    (kept, meta)
}

fn x32(num: usize) -> Layout {
    Layout::from_modes([Layout::new((num, 32), (1, LANE))])
}

/// One warp stores `input` (lane `i`'s `cols` values at `input[i * cols..]`) to TMEM and loads
/// it back with the sparse-compress atom `op`, writing each thread's compressed data, metadata
/// and reduction. With `wait` false it reads the loaded registers before the wait.
fn sp_kernel(
    name: &'static str,
    cols: usize,
    op: impl Into<CopyOp> + Copy + 'static,
    with_red: bool,
    wait: bool,
) -> Kernel {
    let op = op.into();
    let nm = (cols / 32).max(1);
    Kernel::builder(name).block_dim([32, 1, 1]).target(Arch::Sm107a).build(move || {
        let input = param_tensor::<f32>("input", Layout::col_major(32 * cols));
        let cdata = param_tensor::<f32>("cdata", Layout::col_major(32 * cols / 2));
        let meta = param_tensor::<u32>("meta", Layout::col_major(32 * nm));
        let red = param_tensor::<f32>("red", Layout::col_major(32));
        let slot = alloc_smem::<u32>(1, 4);
        let t = thread_idx().x;
        if_(warp_idx().eq(0), || alloc_tmem(cols.max(32) as i32, &slot, None, Arch::Sm107a));
        sync_threads();
        let p = retrieve_tmem_ptr::<f32>(16, &slot);
        let st = make_copy_atom::<f32>(St32x32bOp::new(Repetition::X4));
        let frag = make_fragment::<f32>(4);
        for k in 0..cols / 4 {
            for e in 0..4 {
                frag.store(e, input.load(t * cols as i32 + (4 * k + e) as i32));
            }
            copy(&st, &frag, &make_tensor(p.clone() + (4 * k) as i32, x32(4)));
        }
        fence_view_async_tmem_store();
        sync_threads();
        let atom = make_copy_atom::<f32>(op);
        let cd = make_fragment::<f32>(cols / 2);
        let md = make_fragment::<u32>(nm);
        let r = make_fragment::<f32>(1);
        copy_atom_call_sp(&atom, &make_tensor(p.clone(), x32(cols)), &cd, &md, with_red.then_some(&r));
        if wait {
            fence_view_async_tmem_load();
        }
        for k in 0..cols / 2 {
            cdata.store(t * (cols / 2) as i32 + k as i32, cd.load(k));
        }
        for k in 0..nm {
            meta.store(t * nm as i32 + k as i32, md.load(k));
        }
        if with_red {
            red.store(t, r.load(0));
        }
        fence_view_async_tmem_load();
        sync_threads();
        if_(warp_idx().eq(0), || dealloc_tmem(&p, cols.max(32) as i32, None, Arch::Sm107a));
    })
}

/// A kernel's compressed data, metadata and reductions.
type SpOut = (Vec<f32>, Vec<u32>, Vec<f32>);

fn run_sp(kernel: &Kernel, input: &[f32], cols: usize) -> Result<SpOut, String> {
    let nm = (cols / 32).max(1);
    let mut args = vec![
        KernelArg::buffer(input),
        KernelArg::buffer(&vec![0f32; 32 * cols / 2]),
        KernelArg::buffer(&vec![0u32; 32 * nm]),
        KernelArg::buffer(&[0f32; 32]),
    ];
    kernel
        .simulate_all_orders(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args, &SimOptions::default())
        .map_err(|e| e.to_string())?;
    Ok((args[1].to_vec::<f32>(), args[2].to_vec::<u32>(), args[3].to_vec::<f32>()))
}

fn check_rows(tag: &str, cols: usize, input: &[f32], op: TmemLoadRedOp, nan: bool, got: &SpOut, red: bool) {
    let nm = (cols / 32).max(1);
    for lane in 0..32 {
        let row = &input[lane * cols..(lane + 1) * cols];
        let (kept, meta) = compress_row(row, op, nan);
        let mine = &got.0[lane * cols / 2..(lane + 1) * cols / 2];
        assert!(kept.iter().zip(mine).all(|(a, b)| same_f32(*a, *b)), "{tag}: lane {lane}: {mine:?} vs {kept:?}");
        assert_eq!(&got.1[lane * nm..(lane + 1) * nm], &meta[..], "{tag}: lane {lane}'s metadata");
        if red {
            let want = reduce(row, op, nan);
            assert!(same_f32(got.2[lane], want), "{tag}: lane {lane}: reduction {} vs {want}", got.2[lane]);
        }
    }
}

#[test]
fn spcompress_matches_the_reference_for_each_row_op() {
    for cols in [4usize, 8, 32, 128] {
        let input = values(32 * cols, cols as u64);
        for op in OPS {
            let rep = match cols {
                4 => Repetition::X4,
                8 => Repetition::X8,
                32 => Repetition::X32,
                _ => Repetition::X128,
            };
            let kernel = sp_kernel("sp_plain", cols, LdSPCompress32x32bOp::new(rep, op), false, true);
            let got = run_sp(&kernel, &input, cols).unwrap_or_else(|e| panic!("{e}"));
            check_rows(&format!("{op:?} x{cols}"), cols, &input, op, false, &got, false);
        }
    }
}

#[test]
fn red_spcompress_matches_the_reference_with_and_without_nan() {
    for cols in [8usize, 64] {
        let rep = if cols == 8 { Repetition::X8 } else { Repetition::X64 };
        let input = values(32 * cols, 5 + cols as u64);
        for op in OPS {
            for nan in [false, true] {
                let kernel = sp_kernel("sp_red", cols, LdRedSPCompress32x32bOp::new(rep, op).with_nan(nan), true, true);
                let got = run_sp(&kernel, &input, cols).unwrap_or_else(|e| panic!("{e}"));
                check_rows(&format!("{op:?} nan={nan} x{cols}"), cols, &input, op, nan, &got, true);
            }
        }
    }
}

/// Rows with no NaN, one NaN among finite values, all NaN, and ties.
#[test]
fn spcompress_nan_and_tie_rows() {
    let cols = 16;
    let mut input = values(32 * cols, 17);
    for (lane, row) in input.chunks_mut(cols).enumerate() {
        match lane % 4 {
            0 => row.iter_mut().filter(|v| v.is_nan()).for_each(|v| *v = 0.25),
            1 => row[5] = f32::NAN,
            2 => row.fill(f32::NAN),
            _ => row.iter_mut().for_each(|v| *v = 1.0),
        }
    }
    for op in [TmemLoadRedOp::Max, TmemLoadRedOp::MinAbs] {
        for nan in [false, true] {
            let kernel =
                sp_kernel("sp_nan", cols, LdRedSPCompress32x32bOp::new(Repetition::X16, op).with_nan(nan), true, true);
            let got = run_sp(&kernel, &input, cols).unwrap_or_else(|e| panic!("{e}"));
            check_rows(&format!("{op:?} nan={nan}"), cols, &input, op, nan, &got, true);
        }
    }
}

#[test]
fn spcompress_results_are_pending_until_the_wait() {
    let kernel = sp_kernel("sp_early", 8, LdSPCompress32x32bOp::new(Repetition::X8, TmemLoadRedOp::Max), false, false);
    let e = run_sp(&kernel, &values(32 * 8, 1), 8).unwrap_err();
    assert!(e.contains("before a tcgen05.wait::ld"), "{e}");
}

fn panic_text(f: impl FnOnce()) -> String {
    let f = std::panic::AssertUnwindSafe(f);
    let e = std::panic::catch_unwind(f).map(|_| ()).unwrap_err();
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
fn spcompress_targets_mlir_and_build_checks() {
    let check = |k: &Kernel, a: Arch| cute::validate::check_target(k.ir(), a);
    let kernel = sp_kernel(
        "sp_target",
        8,
        LdRedSPCompress32x32bOp::new(Repetition::X8, TmemLoadRedOp::MaxAbs).with_nan(true),
        true,
        true,
    );
    check(&kernel, Arch::Sm107a).unwrap_or_else(|e| panic!("{}", e.message));
    // The architecture-specific target only: no f family, no other Blackwell.
    for a in [Arch::Sm107f, Arch::Sm107, Arch::Sm100a, Arch::Sm103a, Arch::Sm110a] {
        let e = check(&kernel, a).unwrap_err().message;
        eprintln!("{a}: {e}");
    }
    // The op's own rule, apart from the TMEM allocation's: sm_107a, PTX 9.4.
    let only = cute::validate::TCGEN05_LD_SPCOMPRESS;
    assert_eq!(only.chips, &[(107, 94)]);
    assert!(only.families.is_empty());
    let mlir = kernel.to_mlir();
    let want = "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x8.max.sp2of4.abs.NaN.f32.idx4 {$0}, {$1, $2, $3, $4}, $5, [$6];";
    assert!(mlir.contains(want), "no `{want}` in\n{mlir}");
    let plain =
        sp_kernel("sp_target_plain", 64, LdSPCompress32x32bOp::new(Repetition::X64, TmemLoadRedOp::Min), false, true)
            .to_mlir();
    assert!(
        plain.contains("tcgen05.ld.spcompress.sync.aligned.32x32b.x64.min.sp2of4.f32.idx4 {$0, $1}, {$2,"),
        "{plain}"
    );
    if let Some(tc) = Toolchain::discover() {
        for t in [&mlir, &plain] {
            tc.verify(t).unwrap_or_else(|e| panic!("{e}\n{t}"));
        }
    } else {
        eprintln!("skipping the mlir-opt check: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    // Build-time checks.
    let e = panic_text(|| {
        make_copy_atom::<f32>(LdSPCompress32x32bOp::new(Repetition::X2, TmemLoadRedOp::Max));
    });
    assert!(e.contains("x4 to .x128") && e.contains("x2"), "{e}");
    let e = panic_text(|| {
        make_copy_atom::<F16>(LdSPCompress32x32bOp::new(Repetition::X8, TmemLoadRedOp::Max));
    });
    assert!(e.contains("Float32"), "{e}");
    let e = panic_text(|| {
        Kernel::builder("sp_misuse").block_dim([32, 1, 1]).target(Arch::Sm107a).build(|| {
            let slot = alloc_smem::<u32>(1, 4);
            let p = retrieve_tmem_ptr::<f32>(16, &slot);
            let atom = make_copy_atom::<f32>(LdSPCompress32x32bOp::new(Repetition::X8, TmemLoadRedOp::Max));
            let view = make_tensor(p, x32(8));
            // copy refuses it, and the red destination is given exactly for the .red form.
            copy(&atom, &view, &make_fragment::<f32>(4));
        });
    });
    assert!(e.contains("copy_atom_call_sp"), "{e}");
    let e = panic_text(|| {
        Kernel::builder("sp_misuse2").block_dim([32, 1, 1]).target(Arch::Sm107a).build(|| {
            let slot = alloc_smem::<u32>(1, 4);
            let p = retrieve_tmem_ptr::<f32>(16, &slot);
            let atom = make_copy_atom::<f32>(LdSPCompress32x32bOp::new(Repetition::X8, TmemLoadRedOp::Max));
            let (cd, md, r) = (make_fragment::<f32>(4), make_fragment::<u32>(1), make_fragment::<f32>(1));
            copy_atom_call_sp(&atom, &make_tensor(p, x32(8)), &cd, &md, Some(&r));
        });
    });
    assert!(e.contains("exactly for the .red form"), "{e}");
}

// ---- Rubin helpers: host tests of their choices -------------------------------------------

#[test]
fn rubin_resource_limits_and_arch_names() {
    for a in ["sm_107", "sm_107a", "sm_107f"] {
        assert_eq!(sm107::canonicalize_rubin_resource_arch(a), Ok(Arch::Sm107), "{a}");
    }
    for a in ["sm_100a", "sm_90a", "sm_999", ""] {
        let e = sm107::canonicalize_rubin_resource_arch(a).unwrap_err();
        assert!(e.contains("Rubin"), "{a}: {e}");
    }
    // 327 KiB of shared memory per block and 576 TMEM columns, on every sm_107 target.
    for a in [Arch::Sm107, Arch::Sm107a, Arch::Sm107f] {
        assert_eq!(a.smem_capacity_bytes(), 327 * 1024);
        assert_eq!(cute::arch::get_max_tmem_alloc_cols(a), 576);
    }
    assert_eq!(Arch::Sm100a.smem_capacity_bytes(), 227 * 1024);
    assert_eq!(cute::validate::MAX_SHARED_BYTES, 327 * 1024);
}

#[test]
fn rubin_epilogue_tile_table() {
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let nv = |vec| -> MmaOp {
        let sf = if vec == 16 { DType::F8E4M3FN } else { DType::F8E8M0FNU };
        MmaAtom::from(SM107MmaMXF4NVF4Op::new(sf, vec, (128, 64), one, s, CollectorOp::DISCARD, CollectorOp::DISCARD))
            .op()
    };
    let tile = |op: &MmaOp, m: usize, n: usize, pair: bool| {
        sm107::compute_epilogue_tile_shape(op, &[m, n], pair, LayoutEnum::ROW_MAJOR, DType::F16, None, None)
    };
    let by_mode = |t: &Tiler| match t {
        Tree::Branch(v) => v.iter().map(|x| if let Tree::Leaf(Some(l)) = x { l.size() } else { 0 }).collect::<Vec<_>>(),
        _ => panic!("not a by-mode tiler"),
    };
    assert_eq!(by_mode(&tile(&nv(32), 128, 256, true)), [128, 64]);
    assert_eq!(by_mode(&tile(&nv(32), 256, 256, true)), [128, 64]);
    assert_eq!(by_mode(&tile(&nv(16), 128, 256, true)), [128, 32]);
    assert_eq!(by_mode(&tile(&nv(16), 256, 256, true)), [128, 64]);
    // Anything else is SM100's choice.
    let other = tile(&nv(16), 128, 256, false);
    let sm100 = cute::utils::compute_epilogue_tile_shape(
        &[128, 256],
        false,
        LayoutEnum::ROW_MAJOR,
        DType::F16,
        None,
        None,
        None,
    );
    assert_eq!(format!("{other:?}"), format!("{sm100:?}"));
}

#[test]
fn rubin_trivial_tiled_mma_dispatch_and_collector_variants() {
    use CollectorOp::{DISCARD, FILL, LASTUSE};
    use OperandMajorMode::K;
    let e4 = DType::F8E4M3FN;
    let make = |a_c, b_c| {
        sm107::make_trivial_tiled_mma(
            e4,
            DType::F8E5M2,
            K,
            K,
            DType::F32,
            CtaGroup::ONE,
            (128, 64, 64),
            (1, 1, 1),
            None,
            OperandSource::SMEM,
            a_c,
            b_c,
        )
    };
    let base = make(DISCARD, DISCARD);
    let b_of = |m: &TiledMma| match m.atom().op() {
        MmaOp::Sm100(k) => k.sm107.expect("a Rubin MMA").collector_b,
        o => panic!("{o:?}"),
    };
    let MmaOp::Sm100(kind) = base.atom().op() else { panic!() };
    assert_eq!(kind.shape_mnk(), (128, 64, 64));
    assert_eq!(b_of(&base), DISCARD);
    let keep = base.with_bkeep(None);
    let reuse = base.with_breuse(None);
    assert_eq!((b_of(&keep), b_of(&reuse), b_of(&base)), (FILL, LASTUSE, DISCARD));
    // The variants differ from their source only in the collector (and the accumulate field).
    assert_eq!(keep.thr_layout_vmnk(), base.thr_layout_vmnk());
    assert_eq!(make(DISCARD, FILL).atom().op(), keep.atom().op());
    assert_eq!(b_of(&make(DISCARD, LASTUSE)), LASTUSE);
    // An F16 pair is K = 16, and has no collectors.
    let f16 = sm107::make_trivial_tiled_mma(
        DType::F16,
        DType::F16,
        K,
        K,
        DType::F32,
        CtaGroup::ONE,
        (128, 64, 16),
        (1, 1, 1),
        None,
        OperandSource::SMEM,
        DISCARD,
        DISCARD,
    );
    let MmaOp::Sm100(k) = f16.atom().op() else { panic!() };
    assert_eq!(k.shape_mnk(), (128, 64, 16));
    let e = panic_text(|| {
        sm107::make_trivial_tiled_mma(
            DType::F16,
            DType::F16,
            K,
            K,
            DType::F32,
            CtaGroup::ONE,
            (128, 64, 16),
            (1, 1, 1),
            None,
            OperandSource::SMEM,
            DISCARD,
            FILL,
        );
    });
    assert!(e.contains("does not support collector ops"), "{e}");
    let e = panic_text(|| {
        f16.with_bkeep(None);
    });
    assert!(e.contains("not a Rubin"), "{e}");
    let e = panic_text(|| {
        sm107::make_trivial_tiled_mma(
            DType::I8,
            DType::I8,
            K,
            K,
            DType::I32,
            CtaGroup::ONE,
            (128, 64, 32),
            (1, 1, 1),
            None,
            OperandSource::SMEM,
            DISCARD,
            DISCARD,
        );
    });
    assert!(e.contains("TypeError"), "{e}");
    // Block-scaled: FP4 pairs are the K = 128 NVF4 MMA, the 8-bit mix K = 64.
    let nv = sm107::make_blockscaled_trivial_tiled_mma(
        DType::F4E2M1FN,
        DType::F4E2M1FN,
        K,
        K,
        DType::F8E4M3FN,
        16,
        CtaGroup::ONE,
        (128, 64, 128),
        OperandSource::SMEM,
        DISCARD,
        FILL,
        (1, 1, 1),
        None,
    );
    let MmaOp::Sm100(k) = nv.atom().op() else { panic!() };
    assert_eq!((k.shape_mnk(), k.scale.unwrap().vec), ((128, 64, 128), 16));
    assert_eq!(b_of(&nv), FILL);
    assert_eq!(b_of(&nv.with_breuse(None)), LASTUSE);
    let mx = sm107::make_blockscaled_trivial_tiled_mma(
        e4,
        DType::F4E2M1FN,
        K,
        K,
        DType::F8E8M0FNU,
        32,
        CtaGroup::ONE,
        (128, 64, 64),
        OperandSource::SMEM,
        DISCARD,
        DISCARD,
        (1, 1, 1),
        None,
    );
    let MmaOp::Sm100(k) = mx.atom().op() else { panic!() };
    assert_eq!(k.shape_mnk(), (128, 64, 64));
}

#[test]
fn rubin_sparse_metadata_layouts() {
    // FP4's atom is 128 x 128 on Rubin (SM100's is 128 x 256); the others agree.
    assert_eq!(sm107::make_tensor_e_atom_layout(DType::F4E2M1FN), Layout::new((128, 128), (128, 1)));
    assert_eq!(cute::utils::sm100::make_tensor_e_atom_layout(DType::F4E2M1FN), Layout::new((128, 256), (256, 1)));
    for t in [DType::F8E4M3FN, DType::F16, DType::BF16, DType::TF32] {
        assert_eq!(sm107::make_tensor_e_atom_layout(t), cute::utils::sm100::make_tensor_e_atom_layout(t));
    }
    let g = sm107::make_sparse_gmem_layout_e((256, 64, 256, 2), DType::F4E2M1FN);
    assert_eq!(g.cosize(), 2 * 256 * 256);
    assert_eq!(sm107::make_sparse_gmem_layout_e((100, 64, 100, 1), DType::F8E4M3FN).cosize(), 128 * 128);
    // The shared-memory layout covers (M, 384) per stage.
    let mma = sm107::make_trivial_tiled_mma(
        DType::F16,
        DType::F16,
        OperandMajorMode::K,
        OperandMajorMode::K,
        DType::F32,
        CtaGroup::ONE,
        (128, 64, 16),
        (1, 1, 1),
        None,
        OperandSource::SMEM,
        CollectorOp::DISCARD,
        CollectorOp::DISCARD,
    );
    let s = sm107::make_sparse_smem_layout_e(&mma, (128, 64, 128), DType::F8E4M3FN, 2);
    assert_eq!(s.layout_ref().cosize(), 128 * 384 * 2);
}

// ---- Compress, then multiply ---------------------------------------------------------------

/// The tile `C = prune(A) B^T` (E4M3, M = 128, N = 64, K = 128): each of 128 threads loads its
/// row of dense Float32 A into TMEM, `tcgen05.ld.spcompress` keeps the two largest magnitudes
/// of every four (as bytes of compressed A in shared memory) and the 2:4 metadata (stored to
/// TMEM where the sparse MMA reads it), and two K = 64 sparse MMAs multiply.
fn round_trip_kernel() -> Kernel {
    const M: usize = 128;
    const N: usize = 64;
    const K: usize = 128;
    let e4 = DType::F8E4M3FN;
    let atom: MmaAtom = MmaFP8SparseOp::new(
        e4,
        DType::F32,
        (M, N),
        CtaGroup::ONE,
        OperandSource::SMEM,
        OperandMajorMode::K,
        OperandMajorMode::K,
        SparseMetadataFormat::Tid,
    )
    .into();
    let kb = atom.shape_mnk()[2];
    let e_ty = cute::utils::sm100::make_sparse_e_dtype(e4);
    Kernel::builder("rubin_round_trip").block_dim([128, 1, 1]).target(Arch::Sm107a).build(move || {
        let ga = param_tensor::<f32>("a", Layout::new((M, K), (K, 1)));
        let gb = param_tensor::<u8>("b", Layout::new((N, K), (K, 1)));
        let gc = param_tensor::<f32>("c", Layout::new((M, N), (N, 1)));
        let mut mma = make_tiled_mma(atom.clone(), Layout::col_major((1, 1, 1)));
        let sa_layout = cute::utils::sm100::make_sparse_smem_layout_a(&mma, (M, N, K), e4, 1, 2);
        let sb_layout = cute::utils::sm100::make_smem_layout_b(&mma, (M, N, K), e4, 1, None);
        let sa = make_smem_tensor_aligned::<F8E4M3FN>(sa_layout, 1024).slice((.., .., .., 0));
        let sb = make_smem_tensor_aligned::<F8E4M3FN>(sb_layout, 1024).slice((.., .., .., 0));
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let warp = make_warp_uniform(warp_idx());
        let quad = ((warp % 4) * (32 * LANE) as i32) as Int32;
        // Columns: the accumulator 0..64, the metadata 64..68, the dense A 128..256.
        let (e_col, a_col) = (N, 128usize);
        if_(warp.eq(0), || alloc_tmem(256, &slot, None, Arch::Sm107a));
        // B into shared memory (byte `k` of row `n` at row + 64 * k).
        let sb8 = sb.recast::<u8>();
        for_range(0, (N * K / 128) as i32, 1, |j| {
            let idx = j * 128 + t;
            sb8.store(idx, gb.load((idx % N as i32, idx / N as i32)));
        });
        sync_threads();
        let tmem = retrieve_tmem_ptr::<f32>(16, &slot);
        let tmem8 = retrieve_tmem_ptr::<u8>(16, &slot);
        let tmem32 = retrieve_tmem_ptr::<u32>(16, &slot);
        // Dense A: thread `t`'s row to TMEM lane `t % 32` of its warp's quadrant.
        let st = make_copy_atom::<f32>(St32x32bOp::new(Repetition::X32));
        let frag = make_fragment::<f32>(32);
        for chunk in 0..K / 32 {
            for e in 0..32 {
                frag.store(e, ga.load((t, (32 * chunk + e) as i32)));
            }
            copy(&st, &frag, &make_tensor(tmem.clone() + quad + (a_col + 32 * chunk) as i32, x32(32)));
        }
        fence_view_async_tmem_store();
        sync_threads();
        // Compress: 64 kept values and 4 metadata registers per row.
        let ld = make_copy_atom::<f32>(LdSPCompress32x32bOp::new(Repetition::X128, TmemLoadRedOp::MaxAbs));
        let cd = make_fragment::<f32>(K / 2);
        let md = make_fragment::<u32>(K / 32);
        copy_atom_call_sp(&ld, &make_tensor(tmem.clone() + quad + a_col as i32, x32(K)), &cd, &md, None);
        fence_view_async_tmem_load();
        // The kept values as E4M3 bytes: byte `k` of row `t` at row + 128 * k.
        let sa8 = sa.recast::<u8>();
        for k in 0..K / 2 {
            sa8.store(t + (M * k) as i32, cd.load(k).to::<F8E4M3FN>().bitcast::<u8>());
        }
        // The metadata registers to columns e_col.. of the row's lane.
        let sm = make_copy_atom::<u32>(St32x32bOp::new(Repetition::X4));
        copy(&sm, &md, &make_tensor(tmem32.clone() + quad + e_col as i32, x32(4)));
        fence_view_async_tmem_store();
        fence_view_async_shared();
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let te_logical = cute::utils::sm100::make_sparse_tmem_layout_e(((128, kb), 1, K / kb), e4);
        let te_bytes = recast_layout_from_sparse(&te_logical, e_ty, 8);
        let te = make_tensor(tmem8.clone() + (4 * e_col) as i32, te_bytes);
        let acc =
            make_tensor(tmem.clone(), mma.make_fragment_c::<f32>(&mma.partition_shape_c((M, N))).layout().clone());
        let fa = mma.make_fragment_a(&sa);
        let fb = mma.make_fragment_b(&sb);
        if_(warp.eq(0), || {
            mma.set(Field::ACCUMULATE, false);
            for j in 0..K / kb {
                let (aj, bj, ej) = (fa.slice((.., .., j)), fb.slice((.., .., j)), te.slice((.., .., j)));
                gemm(&mma, &acc, (&aj, &ej), &bj, &acc);
                mma.set(Field::ACCUMULATE, true);
            }
            if_(elect_one(), || tcgen05::commit(&mbar, None, CtaGroup::ONE));
        });
        mbarrier_wait(&mbar, 0);
        let row = (warp % 4) * 32 + lane_idx();
        let ldc = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X32));
        for chunk in 0..N / 32 {
            let view = make_tensor(tmem.clone() + quad + (32 * chunk) as i32, x32(32));
            let frag = make_fragment::<f32>(32);
            copy(&ldc, &view, &frag);
            fence_view_async_tmem_load();
            for e in 0..32 {
                gc.store((row, (32 * chunk + e) as i32), frag.load(e));
            }
        }
        fence_view_async_tmem_load();
        sync_threads();
        if_(warp.eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&tmem, 256, None, Arch::Sm107a);
        });
    })
}

fn halves(n: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..n).map(|_| (r() % 9) as f32 * 0.5 - 2.0).collect()
}

#[test]
fn compress_then_sparse_mma_round_trips_in_the_simulator() {
    const M: usize = 128;
    const N: usize = 64;
    const K: usize = 128;
    let (a, b) = (halves(M * K, 41), halves(N * K, 42));
    let b_bytes: Vec<u8> = b.iter().map(|&x| F8E4M3FN::from_f32(x).0).collect();
    let kernel = round_trip_kernel();
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&b_bytes), KernelArg::buffer(&vec![0f32; M * N])];
    let report =
        kernel.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.tcgen05_mma_instructions, 2);
    let got = args[2].to_vec::<f32>();
    for r in 0..M {
        let (kept, _) = compress_row(&a[r * K..(r + 1) * K], TmemLoadRedOp::MaxAbs, false);
        // The pruned row, the zeros put back: recompute from the keep decision of each group.
        let mut pruned = vec![0f32; K];
        for (g, group) in a[r * K..(r + 1) * K].chunks(4).enumerate() {
            let mut idx = [0usize, 1, 2, 3];
            idx.sort_by(|&i, &j| prefer(group[j], group[i], TmemLoadRedOp::MaxAbs, false));
            for &i in &idx[..2] {
                pruned[4 * g + i] = group[i];
            }
        }
        assert_eq!(kept.len(), K / 2);
        for col in 0..N {
            let want: f32 = (0..K).map(|k| pruned[k] * b[col * K + k]).sum();
            assert_eq!(got[r * N + col], want, "C({r}, {col})");
        }
    }
}

// ---- A Rubin tiled MMA with B reuse running a GEMM tile ------------------------------------

fn operand_layout(kind: SmemLayoutAtomKind, mn: usize, k: usize) -> ComposedLayout {
    let atom = tcgen05::make_smem_layout_atom::<F8E4M3FN>(kind);
    tile_to_shape_swizzled(&atom, (mn, k))
}

/// Two MMAs `D_i = A_i B^T` (E4M3, 128 x 64 x 64 each) of the tiled MMAs of
/// `make_trivial_tiled_mma`, the first `with_bkeep` and the second `with_breuse` (or neither).
fn breuse_kernel(name: &'static str, reuse: bool) -> Kernel {
    Kernel::builder(name).block_dim([128, 1, 1]).target(Arch::Sm107a).build(move || {
        let base = sm107::make_trivial_tiled_mma(
            DType::F8E4M3FN,
            DType::F8E4M3FN,
            OperandMajorMode::K,
            OperandMajorMode::K,
            DType::F32,
            CtaGroup::ONE,
            (128, 64, 64),
            (1, 1, 1),
            None,
            OperandSource::SMEM,
            CollectorOp::DISCARD,
            CollectorOp::DISCARD,
        );
        let mmas = if reuse {
            [base.with_bkeep(Some(false)), base.with_breuse(Some(false))]
        } else {
            [base.clone(), base.clone()]
        };
        let ga = param_tensor::<u8>("a", Layout::new((256, 64), (64, 1)));
        let gb = param_tensor::<u8>("b", Layout::new((64, 64), (64, 1)));
        let gc = param_tensor::<f32>("c", Layout::new((128, 128), (128, 1)));
        let sas: Vec<_> = (0..2)
            .map(|_| make_smem_tensor_aligned::<F8E4M3FN>(operand_layout(SmemLayoutAtomKind::K_SW64, 128, 64), 1024))
            .collect();
        let sb = make_smem_tensor_aligned::<F8E4M3FN>(operand_layout(SmemLayoutAtomKind::K_SW64, 64, 64), 1024);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let warp = make_warp_uniform(warp_idx());
        if_(warp.eq(0), || alloc_tmem(128, &slot, None, Arch::Sm107a));
        for (i, sa) in sas.iter().enumerate() {
            let sa8 = sa.recast::<u8>();
            for_range(0, 64, 1, |j| sa8.store(t + 128 * j, ga.load((128 * i as i32 + t, j))));
        }
        let sb8 = sb.recast::<u8>();
        for_range(0, 32, 1, |j| {
            let idx = j * 128 + t;
            sb8.store(idx, gb.load((idx % 64, idx / 64)));
        });
        fence_view_async_shared();
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let tmem = retrieve_tmem_ptr::<f32>(16, &slot);
        let fb = mmas[0].make_fragment_b(&tiled_divide(&sb, (64, 64)));
        if_(warp.eq(0), || {
            for (i, mma) in mmas.iter().enumerate() {
                let fa = mma.make_fragment_a(&tiled_divide(&sas[i], (128, 64)));
                let layout = mma.make_fragment_c::<f32>(&mma.partition_shape_c((128, 64))).layout().clone();
                let acc = make_tensor(tmem.clone() + (64 * i) as i32, layout);
                gemm(mma, &acc, &fa, &fb, &acc);
            }
            if_(elect_one(), || tcgen05::commit(&mbar, None, CtaGroup::ONE));
        });
        mbarrier_wait(&mbar, 0);
        let row = (warp % 4) * 32 + lane_idx();
        let ld = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X32));
        for chunk in 0..4 {
            let view = make_tensor(tmem.clone() + (warp % 4) * (32 * LANE) as i32 + (32 * chunk), x32(32));
            let frag = make_fragment::<f32>(32);
            copy(&ld, &view, &frag);
            fence_view_async_tmem_load();
            for e in 0..32 {
                gc.store((row, (32 * chunk + e)), frag.load(e));
            }
        }
        fence_view_async_tmem_load();
        sync_threads();
        if_(warp.eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&tmem, 128, None, Arch::Sm107a);
        });
    })
}

#[test]
fn tiled_mma_with_b_reuse_runs_a_gemm_tile() {
    let (av, bv) = (halves(256 * 64, 7), halves(64 * 64, 8));
    let a: Vec<u8> = av.iter().map(|&x| F8E4M3FN::from_f32(x).0).collect();
    let b: Vec<u8> = bv.iter().map(|&x| F8E4M3FN::from_f32(x).0).collect();
    let mut outs = Vec::new();
    for (name, reuse) in [("rubin_breuse", true), ("rubin_no_reuse", false)] {
        let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0f32; 128 * 128])];
        breuse_kernel(name, reuse)
            .simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        outs.push(args[2].to_vec::<f32>());
    }
    assert_eq!(outs[0], outs[1]);
    for i in 0..2 {
        for r in 0..128 {
            for col in 0..64 {
                let want: f32 = (0..64).map(|k| av[(128 * i + r) * 64 + k] * bv[col * 64 + k]).sum();
                assert_eq!(outs[0][r * 128 + 64 * i + col], want, "D{i}({r}, {col})");
            }
        }
    }
    // The reuse kernel's MLIR has the FILL and the LASTUSE forms.
    let mlir = breuse_kernel("mlir_breuse", true).to_mlir();
    assert!(mlir.contains("collector::b::fill") && mlir.contains("collector::b::lastuse"), "{mlir}");
    if let Some(tc) = Toolchain::discover() {
        tc.verify(&mlir).unwrap_or_else(|e| panic!("{e}"));
    }
}

#[test]
fn the_round_trip_kernel_builds_for_sm107a_and_lowers_to_mlir() {
    let kernel = round_trip_kernel();
    cute::validate::check_target(kernel.ir(), Arch::Sm107a).unwrap_or_else(|e| panic!("{}", e.message));
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("tcgen05.ld.spcompress.sync.aligned.32x32b.x128.max.sp2of4.abs.f32.idx4"), "{mlir}");
    assert!(mlir.contains("tcgen05.mma.sp.cta_group::1.kind::f8f6f4"), "{mlir}");
    if let Some(tc) = Toolchain::discover() {
        tc.verify(&mlir).unwrap_or_else(|e| panic!("{e}"));
    }
}
