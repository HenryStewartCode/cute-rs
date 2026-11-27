// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The kernels' structure (TMEM allocation, partition_shape_C / make_fragment_C over the TMEM
// pointer, a k-block loop that sets ACCUMULATE after the first MMA, a commit to an mbarrier,
// and a make_tmem_copy epilogue) follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! tcgen05 MMAs with A in tensor memory and the runtime modifiers: TS-UMMA tiles (A written with `tcgen05.st`, then `wait::st`) for F16 and
//! Int8 on the simulator against plain Rust; NEGATE_A/NEGATE_B toggled inside a
//! `for_range_carry` loop; a disable-output-lane mask leaving TMEM lanes as they were; the
//! diagnostics; the sm_100a PTX (assembled by `ptxas` when it is on PATH); and the host
//! helpers `utils::sm100::make_trivial_tiled_mma` and the shared-memory layout heuristics.

use cute::prelude::*;
use cute::tcgen05::{
    self, CtaGroup, Field, Ld16x32bx2Op, Ld32x32bOp, MmaF16BF16Op, MmaI8Op, MmaTF32Op, OperandSource, Pack, Repetition,
    SmemLayoutAtomKind, St32x32bOp, make_tmem_copy,
};
use cute::utils::sm100;

const K: OperandMajorMode = OperandMajorMode::K;
const MN: OperandMajorMode = OperandMajorMode::MN;

fn reps(r: usize) -> Repetition {
    match r {
        8 => Repetition::X8,
        16 => Repetition::X16,
        32 => Repetition::X32,
        64 => Repetition::X64,
        _ => Repetition::X128,
    }
}

/// Read the `(m, n)` accumulator `acc` (TMEM) into the row-major global `c`, as
/// tests/umma_kinds.rs does: `tcgen05.ld.32x32b` for M = 128, `.16x32bx2` per warp for M = 64.
fn epilogue<C: Numeric>(acc: &Tensor<C>, c: &Tensor<C>, m: usize, n: usize) {
    let t = thread_idx().x;
    let half = C::DTYPE.bits() == 16;
    let block = acc.layout().mode(0);
    let tile = make_tensor(acc.ptr(), Layout::from_modes([block.mode(0), block.mode(1)]));
    if m == 128 {
        let ld =
            if half { Ld32x32bOp::new(reps(n / 2)).with_pack(Pack::Pack16bIn32b) } else { Ld32x32bOp::new(reps(n)) };
        let tiled = make_tmem_copy(&make_copy_atom::<C>(ld), &tile);
        let thr = tiled.get_slice(t);
        let src = thr.partition_s(&tile);
        let dst = thr.partition_d(c);
        let frag = make_fragment_like::<C, _>(&dst);
        copy(&tiled, &src, &frag);
        fence_view_async_tmem_load();
        basic_copy(&frag, &dst);
    } else {
        let per_word = 32 / C::DTYPE.bits();
        let lane_stride = (1usize << 16) * per_word;
        let w = warp_idx() % 4;
        let layout = Layout::from_modes([Layout::new((n, 16), (per_word, lane_stride))]);
        let view = make_tensor(acc.ptr() + w * (32 * lane_stride) as i32, layout);
        let num = if half { n / 4 } else { n / 2 };
        let ld = Ld16x32bx2Op::new(reps(num));
        let ld = if half { ld.with_pack(Pack::Pack16bIn32b) } else { ld };
        let frag = make_fragment::<C>(n / 2);
        copy(&make_copy_atom::<C>(ld), &view, &frag);
        fence_view_async_tmem_load();
        let l = lane_idx();
        let row = w * 16 + l % 16;
        let col0 = (l / 16) * (n / 2) as i32;
        for e in 0..n / 2 {
            c.store((row, col0 + e as i32), frag.load(e));
        }
    }
}

/// How the k-blocks are issued.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// ACCUMULATE after the first k-block.
    Plain,
    /// The same, without the `tcgen05.wait::st` after writing A.
    NoWaitSt,
    /// A `for_range_carry` loop over the k-blocks setting NEGATE_A on odd ones and NEGATE_B
    /// from the third on.
    Negate,
    /// Two k-blocks, the second with [`MASK`] as its disable-output-lane mask.
    Mask,
}

/// The disable-output-lane mask of [`Mode::Mask`]: lanes 4 to 7, 32, 63, 64 to 79 and 127.
const MASK: [u32; 4] = [0x0000_00f0, 0x8000_0001, 0x0000_ffff, 0x8000_0000];

/// The A column of the TMEM allocation (D is at column 0).
const A_COL: i32 = 128;

/// One `(m, n)` tile of `C = A B^T` by a CTA of four warps with A `(m, k)` in TMEM, written
/// from the row-major 32-bit words `p` (`k * bits / 32` per row) by `tcgen05.st.32x32b`, and B
/// `(n, k)` K-major in shared memory.
fn ts_kernel<TA: Numeric, TB: Numeric, C: Numeric>(
    name: &str,
    op: impl Into<MmaAtom>,
    (m, n, k): (usize, usize, usize),
    mode: Mode,
) -> Kernel {
    let atom: MmaAtom = op.into();
    let kb = atom.shape_mnk()[2];
    let words = k * TA::DTYPE.bits() / 32;
    Kernel::builder(name).block_dim([128, 1, 1]).target(Arch::Sm100a).build(move || {
        let p = param_tensor::<u32>("p", Layout::new((m, words), (words, 1)));
        let b = param_tensor_aligned::<TB>("b", Layout::new((n, k), (k, 1)), 16);
        let c = param_tensor::<C>("c", Layout::new((m, n), (n, 1)));
        let mma = make_tiled_mma(atom.clone(), Layout::col_major((1, 1, 1)));
        let b_atom = tcgen05::make_smem_layout_atom::<TB>(SmemLayoutAtomKind::K_SW128);
        let sb = make_smem_tensor_aligned::<TB>(tile_to_shape_swizzled(&b_atom, (n, k)), 1024);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        for_range(0, (n * k / 128) as i32, 1, |j| {
            let idx = j * 128 + t;
            let (r, col) = (idx % n as i32, idx / n as i32);
            sb.store((r, col), b.load((r, col)));
        });
        fence_view_async_shared();
        if_(warp_idx().eq(0), || alloc_tmem(256, &slot, None, Arch::Sm100a));
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        // Each warp writes A's rows to its 32 lanes: row L for M = 128; for M = 64 the rows
        // of Layout F in the first 16 lanes (the other 16 get copies, which the MMA ignores).
        let w = warp_idx() % 4;
        let l = lane_idx();
        let row = if m == 128 { w * 32 + l } else { w * 16 + l % 16 };
        let frag = make_fragment::<u32>(words);
        for e in 0..words {
            frag.store(e, p.load((row, e as i32)));
        }
        let words_ptr = retrieve_tmem_ptr::<u32>(16, &slot) + (w * (32 << 16) + A_COL);
        let view = make_tensor(words_ptr, Layout::from_modes([Layout::new((words, 32), (1, 1 << 16))]));
        copy(&make_copy_atom::<u32>(St32x32bOp::new(reps(words))), &frag, &view);
        if mode != Mode::NoWaitSt {
            fence_view_async_tmem_store();
        }
        sync_threads();
        let per = (32 / TA::DTYPE.bits()) as i32;
        let a_ptr = retrieve_tmem_ptr::<TA>(16, &slot) + A_COL * per;
        let ta = make_tensor(a_ptr, mma.make_tmem_fragment_a::<TA>(&mma.partition_shape_a((m, k))).layout().clone());
        let thr = mma.get_slice(0);
        let fb = mma.make_fragment_b(&thr.partition_b(&sb));
        let tmem_ptr = retrieve_tmem_ptr::<C>(16, &slot);
        let acc =
            make_tensor(tmem_ptr.clone(), mma.make_fragment_c::<C>(&mma.partition_shape_c((m, n))).layout().clone());
        if_(warp_idx().eq(0), || {
            let mut mma = mma.clone();
            mma.set(Field::ACCUMULATE, false);
            match mode {
                Mode::Plain | Mode::NoWaitSt => {
                    for j in 0..k / kb {
                        gemm(&mma, &acc, &ta.slice((.., .., j)), &fb.slice((.., .., j)), &acc);
                        mma.set(Field::ACCUMULATE, true);
                    }
                }
                Mode::Negate => {
                    mma.set(Field::NEGATE_A, false);
                    mma.set(Field::NEGATE_B, false);
                    for_range_carry(0, (k / kb) as i32, 1, mma, |j, mut mma| {
                        mma.set(Field::ACCUMULATE, j.ne(0));
                        mma.set(Field::NEGATE_A, (j % 2).eq(1));
                        mma.set(Field::NEGATE_B, j.ge(2));
                        gemm(&mma, &acc, &ta.slice((.., .., j)), &fb.slice((.., .., j)), &acc);
                        mma
                    });
                }
                Mode::Mask => {
                    gemm(&mma, &acc, &ta.slice((.., .., 0)), &fb.slice((.., .., 0)), &acc);
                    mma.set(Field::ACCUMULATE, true);
                    let mask = MASK.map(|w| w as i32);
                    gemm_disable_output_lane(&mma, &acc, &ta.slice((.., .., 1)), &fb.slice((.., .., 1)), &acc, &mask);
                }
            }
            if_(elect_one(), || tcgen05::commit(&mbar, None, CtaGroup::ONE));
        });
        mbarrier_wait(&mbar, 0);
        epilogue(&acc, &c, m, n);
        sync_threads();
        if_(warp_idx().eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&tmem_ptr, 256, None, Arch::Sm100a)
        });
    })
}

/// A deterministic stream of numbers.
fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// `n` values from -2 to 2 in steps of 0.5.
fn small_halves(n: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..n).map(|_| (r() % 9) as f32 * 0.5 - 2.0).collect()
}

/// The row-major `(rows, k)` codes `codes` (of `bits` bits each) packed into 32-bit words,
/// the first in the low bits.
fn pack(codes: &[u32], bits: usize) -> Vec<u32> {
    codes
        .chunks(32 / bits)
        .map(|g| g.iter().enumerate().fold(0u32, |w, (i, &x)| w | (x & ((1u64 << bits) - 1) as u32) << (bits * i)))
        .collect()
}

fn simulate(kernel: &Kernel, args: &mut [KernelArg]) -> Result<u64, String> {
    let cfg = LaunchConfig::new([1, 1, 1], [128, 1, 1]);
    kernel.simulate(cfg, args).map(|r| r.tcgen05_mma_instructions).map_err(|e| e.to_string())
}

/// The float reference: each `kb`-wide k-block `blk` added in Float32 with the signs
/// `sign(blk)` (A's, B's), then rounded with `round`, row `r` left as `keep(r, blk)` says.
fn reference(
    a: &[f32],
    b: &[f32],
    (m, n, k, kb): (usize, usize, usize, usize),
    sign: impl Fn(usize) -> (f32, f32),
    keep: impl Fn(usize, usize) -> bool,
    round: impl Fn(f32) -> f32,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for blk in 0..k / kb {
                if keep(r, blk) {
                    continue;
                }
                let (sa, sb) = sign(blk);
                for kk in blk * kb..blk * kb + kb {
                    acc += sa * a[r * k + kk] * sb * b[col * k + kk];
                }
                acc = round(acc);
            }
            out[r * n + col] = acc;
        }
    }
    out
}

/// Run an F16 TS tile in `mode` with accumulator `C`, and compare with the reference.
fn run_f16<C: Numeric + Copy + Default>(
    (m, n, k): (usize, usize, usize),
    mode: Mode,
    to_f: fn(C) -> f32,
    round: fn(f32) -> f32,
) -> Result<(), String> {
    let name = format!("umma_ts_f16_{}_{m}x{n}x{k}", C::DTYPE);
    let op = MmaF16BF16Op::new(DType::F16, C::DTYPE, (m, n), CtaGroup::ONE, OperandSource::TMEM, K, K);
    let kernel = ts_kernel::<F16, F16, C>(&name, op, (m, n, k), mode);
    let (av, bv) = (small_halves(m * k, 5 + m as u64), small_halves(n * k, 17 + n as u64));
    let a_codes: Vec<u32> = av.iter().map(|&x| F16::from_f32(x).0 as u32).collect();
    let b_half: Vec<F16> = bv.iter().map(|&x| F16::from_f32(x)).collect();
    let mut args = vec![
        KernelArg::buffer(&pack(&a_codes, 16)),
        KernelArg::buffer(&b_half),
        KernelArg::buffer(&vec![C::default(); m * n]),
    ];
    let mmas = simulate(&kernel, &mut args)?;
    assert_eq!(mmas, (k / 16) as u64, "{name}");
    let sign = |blk: usize| match mode {
        Mode::Negate => (if blk % 2 == 1 { -1.0 } else { 1.0 }, if blk >= 2 { -1.0 } else { 1.0 }),
        _ => (1.0, 1.0),
    };
    let masked = |lane: usize| MASK[lane / 32] >> (lane % 32) & 1 == 1;
    let keep = |r: usize, blk: usize| mode == Mode::Mask && blk == 1 && masked(r);
    let want = reference(&av, &bv, (m, n, k, 16), sign, keep, round);
    let got = args[2].to_vec::<C>();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(to_f(*g), *w, "{name}: C({}, {})", i / n, i % n);
    }
    Ok(())
}

fn f16_round(x: f32) -> f32 {
    F16::from_f32(x).to_f32()
}

#[test]
fn ts_f16_tiles_match_the_rust_reference() {
    run_f16::<f32>((128, 64, 32), Mode::Plain, |x| x, |x| x).unwrap();
    run_f16::<F16>((64, 64, 32), Mode::Plain, |x| x.to_f32(), f16_round).unwrap();
}

#[test]
fn ts_i8_tile_matches_the_rust_reference() {
    let (m, n, k) = (128, 64, 64);
    let op = MmaI8Op::new(DType::I8, (m, n), CtaGroup::ONE, OperandSource::TMEM, K, K);
    let kernel = ts_kernel::<i8, i8, i32>("umma_ts_i8", op, (m, n, k), Mode::Plain);
    let mut r = lcg(23);
    let av: Vec<i8> = (0..m * k).map(|_| r() as i8).collect();
    let bv: Vec<i8> = (0..n * k).map(|_| r() as i8).collect();
    let a_codes: Vec<u32> = av.iter().map(|&x| x as u8 as u32).collect();
    let mut args =
        vec![KernelArg::buffer(&pack(&a_codes, 8)), KernelArg::buffer(&bv), KernelArg::buffer(&vec![0i32; m * n])];
    assert_eq!(simulate(&kernel, &mut args).unwrap(), 2);
    let got = args[2].to_vec::<i32>();
    for row in 0..m {
        for col in 0..n {
            let want: i32 = (0..k).map(|kk| av[row * k + kk] as i32 * bv[col * k + kk] as i32).sum();
            assert_eq!(got[row * n + col], want, "C({row}, {col})");
        }
    }
}

#[test]
fn runtime_negation_in_a_carried_loop_matches_the_rust_reference() {
    run_f16::<f32>((128, 64, 64), Mode::Negate, |x| x, |x| x).unwrap();
}

#[test]
fn disable_output_lane_keeps_the_masked_lanes() {
    run_f16::<f32>((128, 64, 32), Mode::Mask, |x| x, |x| x).unwrap();
}

fn panic_text(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map(|_| ()).unwrap_err();
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
fn ts_umma_diagnostics() {
    // A written by tcgen05.st without the wait::st before the MMA.
    let e = run_f16::<f32>((128, 64, 32), Mode::NoWaitSt, |x| x, |x| x).unwrap_err();
    assert!(e.contains("reads A from TMEM") && e.contains("tcgen05.wait::st"), "{e}");
    // A transposed A from TMEM.
    let e = panic_text(|| {
        MmaF16BF16Op::new(DType::F16, DType::F32, (128, 64), CtaGroup::ONE, OperandSource::TMEM, MN, K);
    });
    assert!(e.contains("OperandSource.TMEM") && e.contains("K-major"), "{e}");
    let e = panic_text(|| {
        MmaTF32Op::new((128, 64), CtaGroup::ONE, OperandSource::TMEM, MN, K);
    });
    assert!(e.contains("MmaTF32Op") && e.contains("K-major"), "{e}");
    // get() of DISABLE_OUTPUT_LANE, set() of it as a Boolean, negation of .kind::i8.
    let atom: MmaAtom =
        MmaF16BF16Op::new(DType::F16, DType::F32, (128, 64), CtaGroup::ONE, OperandSource::SMEM, K, K).into();
    let a = atom.clone();
    let e = panic_text(move || {
        Kernel::builder("get_mask").build(move || {
            a.get(Field::DISABLE_OUTPUT_LANE);
        });
    });
    assert!(e.contains("get(Field.DISABLE_OUTPUT_LANE): a write-only field"), "{e}");
    let mut a = atom.clone();
    let e = panic_text(move || a.set(Field::DISABLE_OUTPUT_LANE, true));
    assert!(e.contains("set_disable_output_lane"), "{e}");
    let mut a = atom.clone();
    let e = panic_text(move || a.set_disable_output_lane(&[0i32; 3]));
    assert!(e.contains("3 words"), "{e}");
    let mut i8: MmaAtom = MmaI8Op::new(DType::I8, (128, 64), CtaGroup::ONE, OperandSource::SMEM, K, K).into();
    let e = panic_text(move || i8.set(Field::NEGATE_A, true));
    assert!(e.contains("kind::i8") && e.contains("no negation"), "{e}");
    // The block-scaled SM120 atoms do not take the field.
    let mut bs: MmaAtom = cute::warp::MmaMXF8Op::new(DType::F8E4M3FN, DType::F32, DType::F8E8M0FNU).into();
    let e = panic_text(move || bs.set_disable_output_lane(&[0i32; 4]));
    assert!(e.contains("not one of its fields"), "{e}");
    // An operand pair make_trivial_tiled_mma has no MMA for.
    let e = panic_text(|| {
        sm100::make_trivial_tiled_mma(
            DType::F16,
            DType::BF16,
            K,
            K,
            DType::F32,
            CtaGroup::ONE,
            (128, 128),
            OperandSource::SMEM,
        );
    });
    assert!(e.contains("TypeError") && e.contains("Float16") && e.contains("BFloat16"), "{e}");
    // An extent that is not a multiple of 8.
    let e = panic_text(|| {
        sm100::get_smem_layout_atom_ab(K, DType::F16, (128, 12), 1);
    });
    assert!(e.contains("not a multiple of 8"), "{e}");
}

#[test]
fn make_trivial_tiled_mma_dispatch() {
    use cute::UmmaDataKind as D;
    let one = CtaGroup::ONE;
    let s = OperandSource::SMEM;
    let kind = |a, b, acc, src| {
        let mma = sm100::make_trivial_tiled_mma(a, b, K, MN, acc, one, (128, 64), src);
        let MmaOp::Sm100(kind) = mma.atom().op() else { panic!("not a tcgen05 MMA") };
        kind
    };
    for (a, acc, data, kb) in [
        (DType::F16, DType::F32, D::F16, 16),
        (DType::BF16, DType::F32, D::F16, 16),
        (DType::TF32, DType::F32, D::Tf32, 8),
        (DType::F32, DType::F32, D::Tf32, 8),
        (DType::I8, DType::I32, D::I8, 32),
        (DType::U8, DType::I32, D::I8, 32),
    ] {
        let k = kind(a, a, acc, s);
        assert_eq!(
            (k.kind, k.shape_mnk(), k.acc, k.trans_a, k.trans_b, k.a_tmem),
            (data, (128, 64, kb), acc, false, true, false),
            "{a}"
        );
    }
    let k = kind(DType::F4E2M1FN, DType::F8E5M2, DType::F16, s);
    assert_eq!((k.kind, k.a, k.b, k.acc, k.k()), (D::F8F6F4, DType::F4E2M1FN, DType::F8E5M2, DType::F16, 32));
    let k = kind(DType::F16, DType::F16, DType::F16, OperandSource::TMEM);
    assert!(k.a_tmem && k.acc == DType::F16);
    // One atom, one CTA "thread".
    let mma = sm100::make_trivial_tiled_mma(DType::F16, DType::F16, K, K, DType::F32, one, (128, 256), s);
    assert_eq!(mma.num_threads(), 1);
    assert_eq!(mma.atom().fields(), &[Field::ACCUMULATE, Field::NEGATE_A, Field::NEGATE_B, Field::DISABLE_OUTPUT_LANE]);
}

#[test]
fn smem_layout_atoms_ab() {
    use SmemLayoutAtomKind::*;
    let f = sm100::get_smem_layout_atom_ab;
    // (major, dtype, (MN, K)) -> atom: K-major by K's bits, MN-major by MN's.
    let cases = [
        (K, DType::F16, (128, 64), K_SW128),
        (K, DType::BF16, (128, 32), K_SW64),
        (K, DType::F16, (128, 16), K_SW32),
        (K, DType::F16, (128, 8), K_INTER),
        (MN, DType::F16, (128, 64), MN_SW128),
        (MN, DType::BF16, (32, 64), MN_SW64),
        (MN, DType::F16, (16, 64), MN_SW32),
        (MN, DType::F16, (8, 64), MN_INTER),
        (K, DType::TF32, (128, 32), K_SW128),
        (K, DType::F32, (128, 8), K_SW32),
        (MN, DType::TF32, (128, 32), MN_SW128_32B),
        (MN, DType::F32, (16, 32), MN_SW64),
        (K, DType::F8E4M3FN, (128, 128), K_SW128),
        (K, DType::F8E5M2, (128, 64), K_SW64),
        (MN, DType::F8E4M3FN, (128, 128), MN_SW128),
        (MN, DType::F8E4M3FN, (64, 128), MN_SW64),
        (K, DType::I8, (128, 32), K_SW32),
        (K, DType::U8, (128, 16), K_INTER),
        (MN, DType::I8, (256, 128), MN_SW128),
        (MN, DType::I8, (32, 128), MN_SW32),
    ];
    for (major, t, shape, want) in cases {
        assert_eq!(f(major, t, shape, 1), want, "{major:?} {t} {shape:?}");
    }
    // Sparsity halves K's bits.
    assert_eq!(f(K, DType::F16, (128, 64), 2), K_SW64);
    assert_eq!(f(MN, DType::F16, (128, 64), 2), MN_SW128);
}

/// The element offset of K-major `(m, k)` in a K_SW128 tile of 16-bit elements, worked by
/// hand: rows of 64 elements, 8-row atoms one after another along M, then along K, with the
/// 128-byte swizzle (element bits 3-5 XOR bits 6-8).
fn k_sw128_f16(m: usize, k: usize, rows: usize) -> usize {
    let x = (m % 8) * 64 + k % 64 + (m / 8) * 512 + (k / 64) * rows * 64;
    x ^ ((x >> 3) & 0b111_000)
}

/// The same for an MN-major MN_SW128 tile: columns of 64 elements along M.
fn mn_sw128_f16(m: usize, k: usize, cols: usize) -> usize {
    let x = (k % 8) * 64 + m % 64 + (k / 8) * 512 + (m / 64) * cols * 64;
    x ^ ((x >> 3) & 0b111_000)
}

#[test]
fn smem_layouts_for_the_tutorial_tiles() {
    let one = CtaGroup::ONE;
    let s = OperandSource::SMEM;
    // fp16_gemm_0.py: mma_tiler (128, 256, 64), Float16, both K-major; dense_gemm.py's
    // (128, 128, 64) with an MN-major B. The shapes are ((MMA_MN, 16), 1, 4, stages), whose
    // column-major index is mn + MN * k + MN * 64 * stage.
    let mma = sm100::make_trivial_tiled_mma(DType::F16, DType::F16, K, K, DType::F32, one, (128, 256), s);
    let tiler = (128, 256, 64);
    let a = sm100::make_smem_layout_a(&mma, tiler, DType::F16, 4, None);
    let b = sm100::make_smem_layout_b(&mma, tiler, DType::F16, 4, None);
    assert_eq!(a.shape().as_static().unwrap().product_each().to_string(), "(2048,1,4,4)");
    assert_eq!(b.shape().as_static().unwrap().product_each().to_string(), "(4096,1,4,4)");
    for (layout, rows) in [(&a, 128), (&b, 256)] {
        for stage in [0, 3] {
            for (m, k) in [(0, 0), (1, 9), (7, 63), (8, 0), (37, 21), (rows - 1, 63)] {
                let idx = m + rows * k + rows * 64 * stage;
                assert_eq!(layout.call(idx), k_sw128_f16(m, k, rows) + stage * rows * 64, "({m}, {k}, {stage})");
            }
        }
    }
    let mma = sm100::make_trivial_tiled_mma(DType::F16, DType::F16, K, MN, DType::F32, one, (128, 128), s);
    let b = sm100::make_smem_layout_b(&mma, (128, 128, 64), DType::F16, 2, None);
    assert_eq!(b.shape().as_static().unwrap().product_each().to_string(), "(2048,1,4,2)");
    for (n, k) in [(0, 0), (9, 1), (63, 7), (64, 0), (100, 33), (127, 63)] {
        assert_eq!(b.call(n + 128 * k + 128 * 64), mn_sw128_f16(n, k, 64) + 128 * 64, "({n}, {k})");
    }
    // is_k_major overrides the MMA's majorness; make_smem_layout with the same tile agrees
    // with make_smem_layout_a as a function.
    let a_mn = sm100::make_smem_layout_a(&mma, (128, 128, 64), DType::F16, 1, Some(false));
    for (m, k) in [(0, 0), (70, 9), (127, 63)] {
        assert_eq!(a_mn.call(m + 128 * k), mn_sw128_f16(m, k, 64), "({m}, {k})");
    }
    let plain = sm100::make_smem_layout(K, (128, 64), DType::F16, 4);
    for i in [0, 1, 777, 128 * 64 * 4 - 1] {
        assert_eq!(plain.call(i), a.call(i), "{i}");
    }
    // C: (128, 256) Float16, N-major and M-major.
    let mma = sm100::make_trivial_tiled_mma(DType::F16, DType::F16, K, K, DType::F32, one, (128, 256), s);
    let c = sm100::make_smem_layout_c(&mma, tiler, DType::F16, 1, false);
    assert_eq!(c.shape().as_static().unwrap().product_each().to_string(), "(32768,1,1,1)");
    for (m, n) in [(0, 0), (9, 70), (127, 255)] {
        assert_eq!(c.call(m + 128 * n), k_sw128_f16(m, n, 128), "({m}, {n})");
    }
    let c = sm100::make_smem_layout_c(&mma, tiler, DType::F16, 1, true);
    for (m, n) in [(0, 0), (70, 9), (127, 255)] {
        assert_eq!(c.call(m + 128 * n), mn_sw128_f16(m, n, 256), "({m}, {n})");
    }
    // Int8 and FP8, K-major, K = 128: K_SW128; TF32 MN-major: MN_SW128_32B.
    let mma = sm100::make_trivial_tiled_mma(DType::I8, DType::I8, K, K, DType::I32, one, (128, 128), s);
    let a = sm100::make_smem_layout_a(&mma, (128, 128, 128), DType::I8, 3, None);
    assert_eq!(a.shape().as_static().unwrap().product_each().to_string(), "(4096,1,4,3)");
    let f8 = sm100::make_trivial_tiled_mma(DType::F8E4M3FN, DType::F8E4M3FN, K, K, DType::F32, one, (128, 128), s);
    let b8 = sm100::make_smem_layout_b(&f8, (128, 128, 128), DType::F8E4M3FN, 3, None);
    assert_eq!(a.call(5 + 128 * 77), b8.call(5 + 128 * 77));
    let tf = sm100::make_trivial_tiled_mma(DType::TF32, DType::TF32, MN, K, DType::F32, one, (128, 128), s);
    let a32 = sm100::make_smem_layout_a(&tf, (128, 128, 32), DType::TF32, 2, None);
    assert_eq!(a32.shape().as_static().unwrap().product_each().to_string(), "(1024,1,4,2)");
}

#[test]
fn ts_umma_ptx_for_sm100a() {
    let op = MmaF16BF16Op::new(DType::F16, DType::F32, (128, 64), CtaGroup::ONE, OperandSource::TMEM, K, K);
    let i8 = MmaI8Op::new(DType::I8, (128, 64), CtaGroup::ONE, OperandSource::TMEM, K, K);
    let cases = [
        ts_kernel::<F16, F16, f32>("ptx_ts_f16", op, (128, 64, 32), Mode::Plain),
        ts_kernel::<i8, i8, i32>("ptx_ts_i8", i8, (128, 64, 64), Mode::Plain),
        ts_kernel::<F16, F16, f32>("ptx_ts_negate", op, (128, 64, 64), Mode::Negate),
        ts_kernel::<F16, F16, f32>("ptx_ts_mask", op, (128, 64, 32), Mode::Mask),
    ];
    for kernel in &cases {
        cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap();
        let e = cute::validate::check_target(kernel.ir(), Arch::Sm90a).unwrap_err();
        assert!(e.message.contains("tcgen05"), "{}", e.message);
    }
    let mlir = cases[3].to_mlir();
    assert!(mlir.contains("kind::f16 [$0], [$1], $2, $3, {$6, $7, $8, $9}, q;"), "{mlir}");
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    for kernel in cases {
        let name = kernel.ir().name.clone();
        let compiled = kernel
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptx = compiled.ptx();
        let kind = if name.contains("i8") { "kind::i8" } else { "kind::f16" };
        let w = format!("tcgen05.mma.cta_group::1.{kind} [%r");
        assert!(ptx.contains(&w), "{name}: no `{w}` in the PTX");
        assert!(ptx.lines().any(|l| l.contains(kind) && l.contains("], [%r")), "{name}: A is not a TMEM address");
        if name.ends_with("mask") {
            assert!(ptx.lines().any(|l| l.contains(kind) && l.contains(", {%r")), "{name}: no disable-output-lane");
        }
        let Some(ptxas) = &ptxas else { continue };
        let dir = std::env::temp_dir().join(format!("cute-umma-ts-{}-{name}", std::process::id()));
        let (src, cubin) = (dir.with_extension("ptx"), dir.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{name}: ptxas: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
