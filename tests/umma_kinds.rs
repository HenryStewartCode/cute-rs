// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The kernels' structure (TMEM allocation, partition_shape_C / make_fragment_C over the TMEM
// pointer, a k-block loop that sets ACCUMULATE after the first MMA, a commit to an mbarrier,
// and a make_tmem_copy epilogue) follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! The dense tcgen05 UMMA kinds TF32, I8 and F8F6F4 (roadmap part m10-umma-dense-kinds):
//! single-CTA, single-tile GEMMs (128x128 and 64x64, several k-blocks) on the simulator,
//! bit-exact against plain Rust: tf32 -> f32 (including operands with low mantissa bits the
//! tensor core drops); s8 and u8 -> s32; every A x B pair of the five f8f6f4 formats with F16
//! and F32 accumulators (FP6/FP4 from the padded shared-memory image); MN-major E4M3 and
//! E5M2. Plus the build-time checks, the targets, and the sm_100a PTX (assembled by `ptxas`
//! when it is on PATH).

use cute::prelude::*;
use cute::tcgen05::{
    self, CtaGroup, Field, Ld16x32bx2Op, Ld32x32bOp, MmaF8F6F4Op, MmaI8Op, MmaTF32Op, OperandSource, Pack, Repetition,
    SmemLayoutAtomKind, make_tmem_copy,
};

fn operand_layout<T: Numeric>(kind: SmemLayoutAtomKind, mn: usize, k: usize) -> ComposedLayout {
    let atom = tcgen05::make_smem_layout_atom::<T>(kind);
    if kind.major_mode() == OperandMajorMode::K {
        tile_to_shape_swizzled(&atom, (mn, k))
    } else {
        tile_to_shape_swizzled_ordered(&atom, (mn, k), &[1, 0])
    }
}

/// An `(mn, k)` operand in global memory, row-major (K-major) or column-major (MN-major).
fn global_layout(kind: SmemLayoutAtomKind, mn: usize, k: usize) -> Layout {
    if kind.major_mode() == OperandMajorMode::K { Layout::new((mn, k), (k, 1)) } else { Layout::new((mn, k), (1, mn)) }
}

/// Copy the `(mn, k)` global operand `g` into shared tensor `s` with plain stores by all
/// 128 threads.
fn fill<T: Numeric>(g: &Tensor<T>, s: &Tensor<T>, mn: usize, k: usize) {
    let t = thread_idx().x;
    for_range(0, (mn * k / 128) as i32, 1, |j| {
        let idx = j * 128 + t;
        let (r, c) = (idx % mn as i32, idx / mn as i32);
        s.store((r, c), g.load((r, c)));
    });
}

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
/// tests/umma.rs does: `tcgen05.ld.32x32b` for M = 128, `.16x32bx2` per warp for M = 64.
fn epilogue<C: Numeric>(acc: &Tensor<C>, c: &Tensor<C>, m: usize, n: usize) {
    let t = thread_idx().x;
    let half = C::DTYPE.bits() == 16;
    let op = |cols: usize| {
        let ld = if half {
            Ld32x32bOp::new(reps(cols / 2)).with_pack(Pack::Pack16bIn32b)
        } else {
            Ld32x32bOp::new(reps(cols))
        };
        make_copy_atom::<C>(ld)
    };
    let block = acc.layout().mode(0);
    let tile = make_tensor(acc.ptr(), Layout::from_modes([block.mode(0), block.mode(1)]));
    if m == 128 {
        let tiled = make_tmem_copy(&op(n), &tile);
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

/// The shape of one test tile.
#[derive(Clone, Copy)]
struct Tile {
    m: usize,
    n: usize,
    k: usize,
    a: SmemLayoutAtomKind,
    b: SmemLayoutAtomKind,
}

/// One `(m, n)` tile of `C = A B^T` (A `(m, k)`, B `(n, k)`, C row-major) by a CTA of four
/// warps; `TA` and `TB` are the shared-memory element types (Uint8 for the padded FP6/FP4
/// image).
fn tile_kernel<TA: Numeric, TB: Numeric, C: Numeric>(
    name: &str,
    op: impl Into<MmaAtom>,
    tile: Tile,
    target: Arch,
) -> Kernel {
    let atom: MmaAtom = op.into();
    let kb = atom.shape_mnk()[2];
    let Tile { m, n, k, a: a_kind, b: b_kind } = tile;
    Kernel::builder(name).block_dim([128, 1, 1]).target(target).build(move || {
        let a = param_tensor_aligned::<TA>("a", global_layout(a_kind, m, k), 16);
        let b = param_tensor_aligned::<TB>("b", global_layout(b_kind, n, k), 16);
        let c = param_tensor::<C>("c", Layout::new((m, n), (n, 1)));
        let mma = make_tiled_mma(atom.clone(), Layout::col_major((1, 1, 1)));
        let sa = make_smem_tensor_aligned::<TA>(operand_layout::<TA>(a_kind, m, k), 1024);
        let sb = make_smem_tensor_aligned::<TB>(operand_layout::<TB>(b_kind, n, k), 1024);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let cols = n.next_power_of_two().max(32) as i32;
        fill(&a, &sa, m, k);
        fill(&b, &sb, n, k);
        fence_view_async_shared();
        if_(warp_idx().eq(0), || alloc_tmem(cols, &slot, None, Arch::Sm100a));
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let tmem_ptr = retrieve_tmem_ptr::<C>(16, &slot);
        let thr = mma.get_slice(0);
        let (fa, fb) = (mma.make_fragment_a(&thr.partition_a(&sa)), mma.make_fragment_b(&thr.partition_b(&sb)));
        let acc_shape = mma.partition_shape_c((m, n));
        let acc = make_tensor(tmem_ptr.clone(), mma.make_fragment_c::<C>(&acc_shape).layout().clone());
        if_(warp_idx().eq(0), || {
            let mut mma = mma.clone();
            mma.set(Field::ACCUMULATE, false);
            for j in 0..k / kb {
                gemm(&mma, &acc, &fa.slice((.., .., j)), &fb.slice((.., .., j)), &acc);
                mma.set(Field::ACCUMULATE, true);
            }
            if_(elect_one(), || tcgen05::commit(&mbar, None, CtaGroup::ONE));
        });
        mbarrier_wait(&mbar, 0);
        epilogue(&acc, &c, m, n);
        sync_threads();
        if_(warp_idx().eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&tmem_ptr, cols, None, Arch::Sm100a)
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

/// `n` values from -2 to 2 in steps of 0.5, exact in all five f8f6f4 formats (and TF32).
fn small_halves(n: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..n).map(|_| (r() % 9) as f32 * 0.5 - 2.0).collect()
}

/// The operand formats of the tests.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fmt {
    E4M3,
    E5M2,
    E2M3,
    E3M2,
    E2M1,
}

const FMTS: [Fmt; 5] = [Fmt::E4M3, Fmt::E5M2, Fmt::E2M3, Fmt::E3M2, Fmt::E2M1];

impl Fmt {
    fn dtype(self) -> DType {
        match self {
            Fmt::E4M3 => DType::F8E4M3FN,
            Fmt::E5M2 => DType::F8E5M2,
            Fmt::E2M3 => DType::F6E2M3FN,
            Fmt::E3M2 => DType::F6E3M2FN,
            Fmt::E2M1 => DType::F4E2M1FN,
        }
    }

    /// The format's code of `x`.
    fn code(self, x: f32) -> u8 {
        match self {
            Fmt::E4M3 => F8E4M3FN::from_f32(x).0,
            Fmt::E5M2 => F8E5M2::from_f32(x).0,
            Fmt::E2M3 => F6E2M3FN::from_f32(x).0,
            Fmt::E3M2 => F6E3M2FN::from_f32(x).0,
            Fmt::E2M1 => F4E2M1FN::from_f32(x).0,
        }
    }

    /// The global-memory bytes of the row-major `(rows, k)` values `v`: one byte each for
    /// FP8, and for FP6/FP4 the padded image (each 16 values packed, low bits first, into
    /// the start of a 16-byte slot).
    fn image(self, v: &[f32]) -> Vec<u8> {
        let codes: Vec<u8> = v.iter().map(|&x| self.code(x)).collect();
        let w = self.dtype().bits();
        if w == 8 {
            return codes;
        }
        codes
            .chunks(16)
            .flat_map(|g| {
                let mut slot = [0u8; 16];
                for (i, &x) in g.iter().enumerate() {
                    for bit in 0..w {
                        let at = w * i + bit;
                        slot[at / 8] |= ((x >> bit) & 1) << (at % 8);
                    }
                }
                slot
            })
            .collect()
    }

    /// The kernel argument of [`image`](Self::image): an FP8 buffer, or the Uint8 image.
    fn arg(self, img: &[u8]) -> KernelArg {
        match self {
            Fmt::E4M3 => KernelArg::buffer(&img.iter().map(|&b| F8E4M3FN(b)).collect::<Vec<_>>()),
            Fmt::E5M2 => KernelArg::buffer(&img.iter().map(|&b| F8E5M2(b)).collect::<Vec<_>>()),
            _ => KernelArg::buffer(img),
        }
    }
}

/// The reference for floats: each `kb`-wide k-block added to C in Float32 in order of K and
/// rounded to the accumulator (`round`), the simulator's documented order.
fn reference(
    a: impl Fn(usize, usize) -> f32,
    b: impl Fn(usize, usize) -> f32,
    (m, n, k, kb): (usize, usize, usize, usize),
    round: impl Fn(f32) -> f32,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for blk in 0..k / kb {
                for kk in blk * kb..blk * kb + kb {
                    acc += a(r, kk) * b(col, kk);
                }
                acc = round(acc);
            }
            out[r * n + col] = acc;
        }
    }
    out
}

fn simulate(kernel: &Kernel, args: &mut [KernelArg], all_orders: bool) -> u64 {
    let cfg = LaunchConfig::new([1, 1, 1], [128, 1, 1]);
    let name = kernel.ir().name.clone();
    let report = if all_orders {
        kernel.simulate_all_orders(cfg, args, &SimOptions::default())
    } else {
        kernel.simulate(cfg, args)
    };
    report.unwrap_or_else(|e| panic!("{name}: {e}")).tcgen05_mma_instructions
}

/// Run one f8f6f4 tile with formats `fa` x `fb` and accumulator `C` (F16 or f32).
fn run_f8f6f4<C: Numeric + Copy + Default>(fa: Fmt, fb: Fmt, tile: Tile, to_f: fn(C) -> f32, round: fn(f32) -> f32) {
    let Tile { m, n, k, a: a_kind, b: b_kind } = tile;
    let name = format!("umma_f8f6f4_{fa:?}_{fb:?}_{}_{m}", C::DTYPE);
    let op = MmaF8F6F4Op::new(
        fa.dtype(),
        fb.dtype(),
        C::DTYPE,
        (m, n),
        CtaGroup::ONE,
        OperandSource::SMEM,
        a_kind.major_mode(),
        b_kind.major_mode(),
    );
    // The shared-memory element type: the FP8 type itself, or Uint8 for the padded image.
    macro_rules! build {
        ($TA:ty, $TB:ty) => {
            tile_kernel::<$TA, $TB, C>(&name, op, tile, Arch::Sm100a)
        };
    }
    macro_rules! with_b {
        ($TA:ty) => {
            match fb {
                Fmt::E4M3 => build!($TA, F8E4M3FN),
                Fmt::E5M2 => build!($TA, F8E5M2),
                _ => build!($TA, u8),
            }
        };
    }
    let kernel = match fa {
        Fmt::E4M3 => with_b!(F8E4M3FN),
        Fmt::E5M2 => with_b!(F8E5M2),
        _ => with_b!(u8),
    };
    let (av, bv) = (small_halves(m * k, 3 + m as u64), small_halves(n * k, 11 + n as u64));
    // Global memory holds the operands in the order global_layout gives (K-major: row-major).
    let mn_order = |v: &[f32], rows: usize, kind: SmemLayoutAtomKind| -> Vec<f32> {
        if kind.major_mode() == OperandMajorMode::K {
            v.to_vec()
        } else {
            (0..rows * k).map(|i| v[(i % rows) * k + i / rows]).collect()
        }
    };
    let a_img = fa.image(&mn_order(&av, m, a_kind));
    let b_img = fb.image(&mn_order(&bv, n, b_kind));
    let mut args = vec![fa.arg(&a_img), fb.arg(&b_img), KernelArg::buffer(&vec![C::default(); m * n])];
    assert_eq!(simulate(&kernel, &mut args, false), (k / 32) as u64, "{name}");
    let want = reference(|r, kk| av[r * k + kk], |c, kk| bv[c * k + kk], (m, n, k, 32), round);
    let got = args[2].to_vec::<C>();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(to_f(*g), *w, "{name}: C({}, {})", i / n, i % n);
    }
}

fn f16_round(x: f32) -> f32 {
    F16::from_f32(x).to_f32()
}

/// Every A x B pair of the five formats, with F16 and F32 accumulators, on 64x64 tiles
/// (K = 64, two k-blocks), plus 128x128 tiles for a few pairs.
#[test]
fn f8f6f4_every_format_pair_matches_the_rust_reference() {
    use SmemLayoutAtomKind::*;
    let small = Tile { m: 64, n: 64, k: 64, a: K_SW64, b: K_SW32 };
    for fa in FMTS {
        for fb in FMTS {
            run_f8f6f4::<f32>(fa, fb, small, |x| x, |x| x);
            run_f8f6f4::<F16>(fa, fb, small, |x| x.to_f32(), f16_round);
        }
    }
    let big = Tile { m: 128, n: 128, k: 128, a: K_SW128, b: K_SW128 };
    run_f8f6f4::<f32>(Fmt::E2M1, Fmt::E3M2, big, |x| x, |x| x);
    run_f8f6f4::<F16>(Fmt::E4M3, Fmt::E2M3, big, |x| x.to_f32(), f16_round);
}

/// MN-major E4M3 and E5M2 operands (A, B or both; an 8-bit MN-major B takes N in steps of
/// 16).
#[test]
fn f8f6f4_mn_major_fp8_matches_the_rust_reference() {
    use SmemLayoutAtomKind::*;
    run_f8f6f4::<f32>(Fmt::E4M3, Fmt::E5M2, Tile { m: 128, n: 128, k: 128, a: MN_SW128, b: MN_SW64 }, |x| x, |x| x);
    run_f8f6f4::<F16>(
        Fmt::E5M2,
        Fmt::E4M3,
        Tile { m: 64, n: 64, k: 64, a: MN_SW32, b: K_SW64 },
        |x| x.to_f32(),
        f16_round,
    );
    run_f8f6f4::<f32>(Fmt::E2M1, Fmt::E5M2, Tile { m: 64, n: 32, k: 64, a: K_SW64, b: MN_INTER }, |x| x, |x| x);
}

/// TF32: exact small values, and the same values with random low mantissa bits set, which
/// the tensor core drops (so C is the same).
#[test]
fn tf32_tiles_match_the_rust_reference() {
    use SmemLayoutAtomKind::*;
    for (tile, noisy, all_orders) in [
        (Tile { m: 128, n: 128, k: 32, a: K_SW128, b: K_SW128 }, false, true),
        (Tile { m: 128, n: 128, k: 32, a: K_SW128, b: K_SW64 }, true, false),
        (Tile { m: 64, n: 64, k: 32, a: K_SW32, b: K_INTER }, true, false),
    ] {
        let Tile { m, n, k, .. } = tile;
        let name = format!("umma_tf32_{m}_{noisy}");
        let op = MmaTF32Op::new((m, n), CtaGroup::ONE, OperandSource::SMEM, OperandMajorMode::K, OperandMajorMode::K);
        let kernel = tile_kernel::<f32, f32, f32>(&name, op, tile, Arch::Sm100a);
        let (av, bv) = (small_halves(m * k, 5), small_halves(n * k, 9));
        let mut noise = lcg(77);
        let mut dirty = |v: &[f32]| -> Vec<f32> {
            v.iter().map(|&x| if noisy { f32::from_bits(x.to_bits() | (noise() & 0x1fff) as u32) } else { x }).collect()
        };
        let (ad, bd) = (dirty(&av), dirty(&bv));
        if noisy {
            assert!(ad.iter().zip(&av).any(|(d, x)| d != x), "{name}: the noise changed nothing");
        }
        let mut args = vec![KernelArg::buffer(&ad), KernelArg::buffer(&bd), KernelArg::buffer(&vec![0f32; m * n])];
        assert_eq!(simulate(&kernel, &mut args, all_orders), (k / 8) as u64, "{name}");
        let want = reference(|r, kk| av[r * k + kk], |c, kk| bv[c * k + kk], (m, n, k, 8), |x| x);
        assert_eq!(args[2].to_vec::<f32>(), want, "{name}");
    }
}

/// s8 x s8 and u8 x u8 -> s32 over the whole 8-bit ranges, exact.
#[test]
fn i8_tiles_match_the_rust_reference() {
    use SmemLayoutAtomKind::*;
    fn run<T: Numeric + Copy>(signed: bool, tile: Tile, enc: fn(i32) -> T, all_orders: bool) {
        let Tile { m, n, k, a: a_kind, b: b_kind } = tile;
        let name = format!("umma_i8_{signed}_{m}");
        let dtype = if signed { DType::I8 } else { DType::U8 };
        let op =
            MmaI8Op::new(dtype, (m, n), CtaGroup::ONE, OperandSource::SMEM, a_kind.major_mode(), b_kind.major_mode());
        let kernel = tile_kernel::<T, T, i32>(&name, op, tile, Arch::Sm100a);
        let mut r = lcg(m as u64 + signed as u64);
        let mut vals =
            |len: usize| -> Vec<i32> { (0..len).map(|_| (r() % 256) as i32 - if signed { 128 } else { 0 }).collect() };
        let (av, bv) = (vals(m * k), vals(n * k));
        let order = |v: &[i32], rows: usize, kind: SmemLayoutAtomKind| -> Vec<T> {
            if kind.major_mode() == OperandMajorMode::K {
                v.iter().map(|&x| enc(x)).collect()
            } else {
                (0..rows * k).map(|i| enc(v[(i % rows) * k + i / rows])).collect()
            }
        };
        let (a, b) = (order(&av, m, a_kind), order(&bv, n, b_kind));
        let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0i32; m * n])];
        assert_eq!(simulate(&kernel, &mut args, all_orders), (k / 32) as u64, "{name}");
        let got = args[2].to_vec::<i32>();
        for row in 0..m {
            for col in 0..n {
                let want: i32 = (0..k).map(|kk| av[row * k + kk] * bv[col * k + kk]).sum();
                assert_eq!(got[row * n + col], want, "{name}: C({row}, {col})");
            }
        }
    }
    run::<i8>(true, Tile { m: 128, n: 128, k: 128, a: K_SW128, b: K_SW128 }, |x| x as i8, true);
    run::<u8>(false, Tile { m: 128, n: 128, k: 128, a: K_SW128, b: MN_SW128 }, |x| x as u8, false);
    run::<i8>(true, Tile { m: 64, n: 64, k: 64, a: MN_SW64, b: K_SW32 }, |x| x as i8, false);
    run::<u8>(false, Tile { m: 64, n: 64, k: 64, a: K_INTER, b: K_SW64 }, |x| x as u8, false);
}

fn panic_text(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let e = std::panic::catch_unwind(f).map(|_| ()).unwrap_err();
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
#[allow(deprecated)]
fn illegal_operations_panic_when_built() {
    use OperandMajorMode::{K, MN};
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    // An N an Int8 MMA does not take (40: above 32 the step is 16), and an MN-major 8-bit B
    // with N = 24.
    let e = panic_text(|| {
        MmaI8Op::new(DType::I8, (128, 40), one, s, K, K);
    });
    assert!(e.contains("N is 40") && e.contains("steps of 16"), "{e}");
    let e = panic_text(|| {
        MmaI8Op::new(DType::U8, (128, 24), one, s, K, MN);
    });
    assert!(e.contains("N is 24") && e.contains("MN-major"), "{e}");
    MmaI8Op::new(DType::I8, (64, 24), one, s, MN, K);
    let e = panic_text(|| {
        MmaI8Op::new(DType::I32, (128, 64), one, s, K, K);
    });
    assert!(e.contains("ab_dtype"), "{e}");
    // MN-major FP4 and FP6.
    for (a, b, am, bm) in [(DType::F4E2M1FN, DType::F8E4M3FN, MN, K), (DType::F8E4M3FN, DType::F6E3M2FN, K, MN)] {
        let e = panic_text(move || {
            MmaF8F6F4Op::new(a, b, DType::F32, (128, 64), one, s, am, bm);
        });
        assert!(e.contains("must be K-major"), "{e}");
    }
    // Formats .kind::f8f6f4 does not take, and the FP8-only alias's mixed or non-FP8 types.
    let e = panic_text(|| {
        MmaF8F6F4Op::new(DType::F16, DType::F8E4M3FN, DType::F32, (128, 64), one, s, K, K);
    });
    assert!(e.contains("'a_dtype' is") && e.contains("does not take"), "{e}");
    let e = panic_text(|| {
        MmaF8F6F4Op::new(DType::F8E4M3FN, DType::F8E4M3, DType::F32, (128, 64), one, s, K, K);
    });
    assert!(e.contains("'b_dtype' is"), "{e}");
    let e = panic_text(|| {
        MmaF8F6F4Op::new(DType::F8E4M3FN, DType::F8E5M2, DType::I32, (128, 64), one, s, K, K);
    });
    assert!(e.contains("acc_dtype"), "{e}");
    let e = panic_text(|| {
        tcgen05::MmaFP8Op::new(DType::F6E2M3FN, DType::F32, (128, 64), one, s, K, K);
    });
    assert!(e.contains("MmaFP8Op") && e.contains("ab_dtype"), "{e}");
    let fp8 = tcgen05::MmaFP8Op::new(DType::F8E5M2, DType::F16, (64, 32), one, s, K, K).kind();
    assert_eq!(fp8.idesc(), 1 << 7 | 1 << 10 | 4 << 17 | 4 << 24);
    // K, M and the CTA-pair form.
    let e = panic_text(|| {
        MmaTF32Op::new((128, 64, 16), one, s, K, K);
    });
    assert!(e.contains("K is 16") && e.contains("K = 8"), "{e}");
    let e = panic_text(|| {
        MmaTF32Op::new((96, 64), one, s, K, K);
    });
    assert!(e.contains("M is 96"), "{e}");
    let e = panic_text(|| {
        MmaTF32Op::new((64, 64), CtaGroup::TWO, s, K, K);
    });
    // m10-cta-pair: a CTA pair's M is 128 or 256.
    assert!(e.contains("M is 64") && e.contains("128 or 256"), "{e}");
    // A packed FP4 shared-memory tensor (not the padded image).
    let e = panic_text(|| {
        Kernel::builder("packed_fp4").block_dim([128, 1, 1]).build(|| {
            let op = MmaF8F6F4Op::new(DType::F4E2M1FN, DType::F8E4M3FN, DType::F32, (128, 64), one, s, K, K);
            let mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
            let sa = make_smem_tensor_aligned::<F4E2M1FN>(Layout::new((128, 32), (32, 1)), 1024);
            mma.make_fragment_a(&mma.get_slice(0).partition_a(&sa));
        });
    });
    assert!(e.contains("padded"), "{e}");
}

/// A kernel of just one k-block of MMAs on `(128, 128)` and `(64, 128)` shared tiles of `T`
/// (no TMEM allocation, which sm_90a would refuse before the MMA).
fn gemm_only<T: Numeric, C: Numeric>(name: &str, op: impl Into<MmaAtom>) -> Kernel {
    use SmemLayoutAtomKind::K_SW128;
    let op: MmaAtom = op.into();
    let k = 128 * 8 / T::DTYPE.bits();
    Kernel::builder(name).block_dim([128, 1, 1]).build(move || {
        let mma = make_tiled_mma(op.clone(), Layout::col_major((1, 1, 1)));
        let sa = make_smem_tensor_aligned::<T>(operand_layout::<T>(K_SW128, 128, k), 1024);
        let sb = make_smem_tensor_aligned::<T>(operand_layout::<T>(K_SW128, 64, k), 1024);
        let thr = mma.get_slice(0);
        let (fa, fb) = (mma.make_fragment_a(&thr.partition_a(&sa)), mma.make_fragment_b(&thr.partition_b(&sb)));
        let ptr = make_ptr::<C>(Int32::from(0), AddressSpace::Tmem, Some(16));
        let acc = make_tensor(ptr, mma.make_fragment_c::<C>(&mma.partition_shape_c((128, 64))).layout().clone());
        gemm(&mma, &acc, &fa.slice((.., .., 0)), &fb.slice((.., .., 0)), &acc);
    })
}

/// kind::tf32 and kind::f8f6f4 on sm_100a and the sm_100f/sm_110f families; kind::i8 on
/// `a` targets only; every kind refused on sm_90a, naming the gemm's source line.
#[test]
fn targets_of_the_kinds() {
    use OperandMajorMode::K;
    use SmemLayoutAtomKind::K_SW128;
    let tile = Tile { m: 128, n: 64, k: 32, a: K_SW128, b: K_SW128 };
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let tf32 = tile_kernel::<f32, f32, f32>("t_tf32", MmaTF32Op::new((128, 64), one, s, K, K), tile, Arch::Sm100a);
    let i8 = tile_kernel::<i8, i8, i32>("t_i8", MmaI8Op::new(DType::I8, (128, 64), one, s, K, K), tile, Arch::Sm100a);
    let f8 = tile_kernel::<u8, F8E4M3FN, f32>(
        "t_f8f6f4",
        MmaF8F6F4Op::new(DType::F4E2M1FN, DType::F8E4M3FN, DType::F32, (128, 64), one, s, K, K),
        tile,
        Arch::Sm100a,
    );
    let check = |k: &Kernel, a: Arch| cute::validate::check_target(k.ir(), a);
    for k in [&tf32, &f8] {
        for a in [Arch::Sm100a, Arch::Sm100f, Arch::Sm110a, Arch::Sm110f] {
            check(k, a).unwrap_or_else(|e| panic!("{}: {a:?}: {e}", k.ir().name));
        }
    }
    check(&i8, Arch::Sm100a).unwrap();
    check(&i8, Arch::Sm110a).unwrap();
    for a in [Arch::Sm100f, Arch::Sm110f] {
        let e = check(&i8, a).unwrap_err();
        assert!(e.message.contains("kind::i8") && e.message.contains("sm_100a"), "{a:?}: {}", e.message);
    }
    let i8_only = gemm_only::<u8, i32>("t_i8_only", MmaI8Op::new(DType::U8, (128, 64), one, s, K, K));
    let tf32_only = gemm_only::<f32, f32>("t_tf32_only", MmaTF32Op::new((128, 64), one, s, K, K));
    let f8_only = gemm_only::<u8, F16>(
        "t_f8f6f4_only",
        MmaF8F6F4Op::new(DType::F6E2M3FN, DType::F4E2M1FN, DType::F16, (128, 64), one, s, K, K),
    );
    let e = check(&i8_only, Arch::Sm100f).unwrap_err();
    assert!(e.message.contains("kind::i8"), "{}", e.message);
    check(&i8_only, Arch::Sm100a).unwrap();
    for (k, what) in [(&tf32_only, "kind::tf32"), (&i8_only, "kind::i8"), (&f8_only, "kind::f8f6f4")] {
        let e = check(k, Arch::Sm90a).unwrap_err();
        assert!(e.message.contains(what), "{}", e.message);
        let loc = e.loc.expect("a source location");
        assert!(loc.file().ends_with("umma_kinds.rs"), "{}", loc.file());
    }
}

#[test]
fn umma_kinds_ptx_for_sm100a() {
    use OperandMajorMode::{K, MN};
    use SmemLayoutAtomKind::*;
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let tf32 = MmaTF32Op::new((128, 128), one, s, K, K);
    let i8 = MmaI8Op::new(DType::U8, (128, 128), one, s, K, MN);
    let f4 = MmaF8F6F4Op::new(DType::F4E2M1FN, DType::F6E3M2FN, DType::F16, (64, 64), one, s, K, K);
    let f8 = MmaF8F6F4Op::new(DType::F8E5M2, DType::F8E4M3FN, DType::F32, (128, 128), one, s, MN, K);
    let cases: Vec<(Kernel, u32, &str)> = vec![
        (
            tile_kernel::<f32, f32, f32>(
                "ptx_tf32",
                tf32,
                Tile { m: 128, n: 128, k: 32, a: K_SW128, b: K_SW64 },
                Arch::Sm100a,
            ),
            tf32.kind().idesc(),
            "kind::tf32",
        ),
        (
            tile_kernel::<u8, u8, i32>(
                "ptx_i8",
                i8,
                Tile { m: 128, n: 128, k: 64, a: K_SW128, b: MN_SW64 },
                Arch::Sm100a,
            ),
            i8.kind().idesc(),
            "kind::i8",
        ),
        (
            tile_kernel::<u8, u8, F16>(
                "ptx_f4f6",
                f4,
                Tile { m: 64, n: 64, k: 64, a: K_SW64, b: K_SW32 },
                Arch::Sm100a,
            ),
            f4.kind().idesc(),
            "kind::f8f6f4",
        ),
        (
            tile_kernel::<F8E5M2, F8E4M3FN, f32>(
                "ptx_f8",
                f8,
                Tile { m: 128, n: 128, k: 64, a: MN_SW128, b: K_SW128 },
                Arch::Sm100a,
            ),
            f8.kind().idesc(),
            "kind::f8f6f4",
        ),
    ];
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    for (kernel, idesc, kind) in cases {
        let name = kernel.ir().name.clone();
        let compiled = kernel
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptx = compiled.ptx();
        let w = format!("tcgen05.mma.cta_group::1.{kind}");
        assert!(ptx.contains(&w), "{name}: no `{w}` in the PTX");
        // The instruction descriptor, moved into a register as an immediate.
        assert!(ptx.contains(&format!(", {idesc};")), "{name}: no immediate {idesc} ({idesc:#x}) in the PTX");
        let Some(ptxas) = &ptxas else { continue };
        let dir = std::env::temp_dir().join(format!("cute-umma-kinds-{}-{name}", std::process::id()));
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
