// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The kernels' structure (TMEM allocation, partition_shape_C / make_fragment_C over the TMEM
// pointer, a k-block loop that sets ACCUMULATE after the first MMA, a commit to an mbarrier,
// and a make_tmem_copy epilogue) follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! tcgen05 UMMAs (F16/BF16) into TMEM and `tcgen05.commit` (roadmap part m10-umma-f16):
//! single-CTA, single-tile GEMMs (128x128x64 and 64x64x64; f16->f32, bf16->f32, f16->f16;
//! every legal shared-memory layout atom for A and for B) on the simulator, bit-exact against
//! plain Rust on small integers; a two-stage double buffer released by commits; the
//! simulator's misuse reports; build-time checks; and the sm_100a PTX (assembled by `ptxas`
//! when it is on PATH).

use cute::prelude::*;
use cute::tcgen05::{
    self, CtaGroup, Field, Ld16x32bx2Op, Ld32x32bOp, MmaF16BF16Op, OperandSource, Pack, Repetition, SmemLayoutAtomKind,
    make_tmem_copy,
};

/// K of the single-tile kernels: four k-blocks of 16.
const K: usize = 64;

/// What a kernel gets wrong on purpose (for the simulator's reports).
#[derive(Clone, Copy, Default)]
struct Faults {
    no_proxy_fence: bool,
    no_wait: bool,
    overwrite_in_flight: bool,
    divergent_gemm: bool,
}

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

fn umma_op<T: Numeric, C: Numeric>(m: usize, n: usize, a: SmemLayoutAtomKind, b: SmemLayoutAtomKind) -> TiledMma {
    let op = MmaF16BF16Op::new(
        T::DTYPE,
        C::DTYPE,
        (m, n),
        CtaGroup::ONE,
        OperandSource::SMEM,
        a.major_mode(),
        b.major_mode(),
    );
    make_tiled_mma(op, Layout::col_major((1, 1, 1)))
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

/// Read the `(m, n)` accumulator `acc` (TMEM, the layout make_fragment_c gives) into the
/// row-major global `c`: a make_tmem_copy of `tcgen05.ld.32x32b` (packed for a 16-bit
/// accumulator) for M = 128; for M = 64 (whose D is in the first 16 lanes of each warp's
/// quadrant, PTX Layout F) one `.32x32b` load per warp, of which lanes 0 to 15 hold rows.
fn epilogue<C: Numeric>(acc: &Tensor<C>, c: &Tensor<C>, m: usize, n: usize) {
    let t = thread_idx().x;
    let half = C::DTYPE.bits() == 16;
    let reps = |r: usize| match r {
        8 => Repetition::X8,
        16 => Repetition::X16,
        32 => Repetition::X32,
        64 => Repetition::X64,
        _ => Repetition::X128,
    };
    let op = |cols: usize| {
        let ld = if half {
            Ld32x32bOp::new(reps(cols / 2)).with_pack(Pack::Pack16bIn32b)
        } else {
            Ld32x32bOp::new(reps(cols))
        };
        make_copy_atom::<C>(ld)
    };
    // The (M, N) block of the first MMA tile, as a rank-2 TMEM tensor.
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
        // .16x32bx2 over the 16 lanes of D in the warp's quadrant: thread l holds TMEM lane
        // l % 16, columns (l / 16) * n / 2 onwards.
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

/// One `(m, n)` tile of `C = A B^T` (A `(m, K)`, B `(n, K)`, C row-major) by a CTA of four
/// warps: warp 0 issues the MMAs (ACCUMULATE false for the first k-block) and its elected
/// lane commits to an mbarrier, which every warp waits on before the epilogue.
fn tile_kernel<T: Numeric, C: Numeric>(
    name: &'static str,
    m: usize,
    n: usize,
    a_kind: SmemLayoutAtomKind,
    b_kind: SmemLayoutAtomKind,
    faults: Faults,
) -> Kernel {
    Kernel::builder(name).block_dim([128, 1, 1]).target(Arch::Sm100a).build(move || {
        let a = param_tensor_aligned::<T>("a", global_layout(a_kind, m, K), 16);
        let b = param_tensor_aligned::<T>("b", global_layout(b_kind, n, K), 16);
        let c = param_tensor::<C>("c", Layout::new((m, n), (n, 1)));
        let mma = umma_op::<T, C>(m, n, a_kind, b_kind);
        let sa = make_smem_tensor_aligned::<T>(operand_layout::<T>(a_kind, m, K), 1024);
        let sb = make_smem_tensor_aligned::<T>(operand_layout::<T>(b_kind, n, K), 1024);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let cols = n.next_power_of_two().max(32) as i32;
        fill(&a, &sa, m, K);
        fill(&b, &sb, n, K);
        if !faults.no_proxy_fence {
            fence_view_async_shared();
        }
        if_(warp_idx().eq(0), || alloc_tmem(cols, &slot, None, Arch::Sm100a));
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let tmem_ptr = retrieve_tmem_ptr::<C>(16, &slot);
        let thr = mma.get_slice(0);
        let (fa, fb) = (mma.make_fragment_a(&thr.partition_a(&sa)), mma.make_fragment_b(&thr.partition_b(&sb)));
        let acc_shape = mma.partition_shape_c((m, n));
        let acc = make_tensor(tmem_ptr.clone(), mma.make_fragment_c::<C>(&acc_shape).layout().clone());
        let issue = {
            let (mma, acc, fa, fb, mbar) = (mma.clone(), acc.clone(), fa.clone(), fb.clone(), mbar.clone());
            move || {
                let mut mma = mma;
                mma.set(Field::ACCUMULATE, false);
                for kb in 0..K / 16 {
                    gemm(&mma, &acc, &fa.slice((.., .., kb)), &fb.slice((.., .., kb)), &acc);
                    mma.set(Field::ACCUMULATE, true);
                }
                if_(elect_one(), || tcgen05::commit(&mbar, None, CtaGroup::ONE));
            }
        };
        if faults.divergent_gemm {
            if_(warp_idx().eq(0) & lane_idx().lt(16), issue);
        } else {
            if_(warp_idx().eq(0), issue);
        }
        if faults.overwrite_in_flight {
            sa.store((t % m as i32, 0), a.load((0, 0)));
        }
        if !faults.no_wait {
            mbarrier_wait(&mbar, 0);
        }
        epilogue(&acc, &c, m, n);
        sync_threads();
        if_(warp_idx().eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&tmem_ptr, cols, None, Arch::Sm100a)
        });
    })
}

/// Host conversions of the operand and accumulator types.
trait HostFloat: Numeric + Copy {
    fn host(x: f32) -> Self;
    fn f(self) -> f32;
}

impl HostFloat for F16 {
    fn host(x: f32) -> Self {
        F16::from_f32(x)
    }
    fn f(self) -> f32 {
        self.to_f32()
    }
}

impl HostFloat for BF16 {
    fn host(x: f32) -> Self {
        BF16::from_f32(x)
    }
    fn f(self) -> f32 {
        self.to_f32()
    }
}

impl HostFloat for f32 {
    fn host(x: f32) -> Self {
        x
    }
    fn f(self) -> f32 {
        self
    }
}

/// Small integers from -4 to 4, the same on every run.
fn small_ints(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) % 9) as f32 - 4.0
        })
        .collect()
}

/// The reference: each 16-wide k-block added to C in Float32 in order of K and rounded to
/// `C` (the simulator's documented order; exact here, as the data are small integers).
fn reference<C: HostFloat>(
    a: impl Fn(usize, usize) -> f32,
    b: impl Fn(usize, usize) -> f32,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for kb in 0..k / 16 {
                for kk in kb * 16..kb * 16 + 16 {
                    acc += a(r, kk) * b(col, kk);
                }
                acc = C::host(acc).f();
            }
            out[r * n + col] = acc;
        }
    }
    out
}

/// Run the tile kernel on every schedule and check C.
fn run_tile<T: HostFloat, C: HostFloat>(
    name: &'static str,
    m: usize,
    n: usize,
    a_kind: SmemLayoutAtomKind,
    b_kind: SmemLayoutAtomKind,
) {
    let kernel = tile_kernel::<T, C>(name, m, n, a_kind, b_kind, Faults::default());
    let (av, bv) = (small_ints(m * K, 1 + m as u64), small_ints(n * K, 7 + n as u64));
    let a: Vec<T> = av.iter().map(|&x| T::host(x)).collect();
    let b: Vec<T> = bv.iter().map(|&x| T::host(x)).collect();
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![C::host(0.0); m * n])];
    let report = kernel
        .simulate_all_orders(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(report.tcgen05_mma_instructions, (K / 16) as u64, "{name}");
    let a_mn = a_kind.major_mode() == OperandMajorMode::MN;
    let b_mn = b_kind.major_mode() == OperandMajorMode::MN;
    let at_a = |r: usize, k: usize| if a_mn { av[k * m + r] } else { av[r * K + k] };
    let at_b = |r: usize, k: usize| if b_mn { bv[k * n + r] } else { bv[r * K + k] };
    let want = reference::<C>(at_a, at_b, m, n, K);
    let got = args[2].to_vec::<C>();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.f(), *w, "{name}: C({}, {})", i / n, i % n);
    }
}

#[test]
fn umma_tiles_match_the_rust_reference() {
    use SmemLayoutAtomKind::*;
    run_tile::<F16, f32>("umma_128_f16_f32_ksw128_ksw128", 128, 128, K_SW128, K_SW128);
    run_tile::<BF16, f32>("umma_128_bf16_f32_mnsw128_mnsw64", 128, 128, MN_SW128, MN_SW64);
    run_tile::<F16, F16>("umma_128_f16_f16_kinter_mninter", 128, 128, K_INTER, MN_INTER);
    run_tile::<BF16, f32>("umma_128_bf16_f32_mninter_kinter", 128, 128, MN_INTER, K_INTER);
    run_tile::<F16, f32>("umma_64_f16_f32_ksw32_mnsw32", 64, 64, K_SW32, MN_SW32);
    run_tile::<BF16, f32>("umma_64_bf16_f32_mnsw12832b_ksw64", 64, 64, MN_SW128_32B, K_SW64);
    run_tile::<F16, F16>("umma_64_f16_f16_ksw64_mnsw12832b", 64, 64, K_SW64, MN_SW128_32B);
    run_tile::<F16, f32>("umma_64_f16_f32_mnsw64_ksw128", 64, 64, MN_SW64, K_SW128);
    run_tile::<BF16, f32>("umma_64_bf16_f32_mnsw32_mnsw128", 64, 64, MN_SW32, MN_SW128);
}

/// A 128x128x128 GEMM through two shared-memory stages of 32 columns of K: all threads
/// fill a stage (after waiting on its `empty` mbarrier from the second round on), warp 0
/// issues two k-blocks of MMAs from it and commits to its `empty` mbarrier, and a last
/// commit to `full` releases the accumulator to the epilogue.
fn double_buffer_kernel() -> Kernel {
    const KT: usize = 128;
    const KS: usize = 32;
    Kernel::builder("umma_double_buffer").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
        use SmemLayoutAtomKind::K_SW64;
        let a = param_tensor_aligned::<F16>("a", Layout::new((128, KT), (KT, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((128, KT), (KT, 1)), 16);
        let c = param_tensor::<f32>("c", Layout::new((128, 128), (128, 1)));
        let mma = umma_op::<F16, f32>(128, 128, K_SW64, K_SW64);
        let stages: Vec<(Tensor<F16>, Tensor<F16>)> = (0..2)
            .map(|_| {
                (
                    make_smem_tensor_aligned::<F16>(operand_layout::<F16>(K_SW64, 128, KS), 1024),
                    make_smem_tensor_aligned::<F16>(operand_layout::<F16>(K_SW64, 128, KS), 1024),
                )
            })
            .collect();
        let empty = alloc_smem::<u64>(2, 8);
        let full = alloc_smem::<u64>(1, 8);
        let slot = alloc_smem::<u32>(1, 4);
        let t = thread_idx().x;
        if_(warp_idx().eq(0), || alloc_tmem(128, &slot, None, Arch::Sm100a));
        if_(t.eq(0), || {
            mbarrier_init(&empty, 1);
            mbarrier_init(&(empty.clone() + 1), 1);
            mbarrier_init(&full, 1);
        });
        mbarrier_init_fence();
        sync_threads();
        let tmem_ptr = retrieve_tmem_ptr::<f32>(16, &slot);
        let thr = mma.get_slice(0);
        let acc = make_tensor(
            tmem_ptr.clone(),
            mma.make_fragment_c::<f32>(&mma.partition_shape_c((128, 128))).layout().clone(),
        );
        let mut mma = mma;
        mma.set(Field::ACCUMULATE, false);
        for chunk in 0..KT / KS {
            let s = chunk % 2;
            let (sa, sb) = &stages[s];
            let bar = empty.clone() + s as i32;
            if chunk >= 2 {
                mbarrier_wait(&bar, ((chunk / 2 - 1) % 2) as i32);
            }
            let off = (chunk * KS) as i32;
            let ga = make_tensor(a.ptr() + off, Layout::new((128, KS), (KT, 1)));
            let gb = make_tensor(b.ptr() + off, Layout::new((128, KS), (KT, 1)));
            fill(&ga, sa, 128, KS);
            fill(&gb, sb, 128, KS);
            fence_view_async_shared();
            sync_threads();
            let (fa, fb) = (mma.make_fragment_a(&thr.partition_a(sa)), mma.make_fragment_b(&thr.partition_b(sb)));
            let issue = {
                let (m2, acc, full) = (mma.clone(), acc.clone(), full.clone());
                move || {
                    let mut m2 = m2;
                    for kb in 0..KS / 16 {
                        gemm(&m2, &acc, &fa.slice((.., .., kb)), &fb.slice((.., .., kb)), &acc);
                        m2.set(Field::ACCUMULATE, true);
                    }
                    if_(elect_one(), || tcgen05::commit(&bar, None, CtaGroup::ONE));
                    if chunk == KT / KS - 1 {
                        if_(elect_one(), || tcgen05::commit(&full, None, CtaGroup::ONE));
                    }
                }
            };
            if_(warp_idx().eq(0), issue);
            mma.set(Field::ACCUMULATE, true);
        }
        mbarrier_wait(&full, 0);
        epilogue(&acc, &c, 128, 128);
        sync_threads();
        if_(warp_idx().eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&tmem_ptr, 128, None, Arch::Sm100a)
        });
    })
}

#[test]
fn umma_double_buffer_matches_the_rust_reference() {
    let kernel = double_buffer_kernel();
    let (av, bv) = (small_ints(128 * 128, 3), small_ints(128 * 128, 5));
    let a: Vec<F16> = av.iter().map(|&x| F16::from_f32(x)).collect();
    let b: Vec<F16> = bv.iter().map(|&x| F16::from_f32(x)).collect();
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0f32; 128 * 128])];
    let report = kernel
        .simulate_all_orders(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.tcgen05_mma_instructions, 8);
    let want = reference::<f32>(|r, k| av[r * 128 + k], |r, k| bv[r * 128 + k], 128, 128, 128);
    assert_eq!(args[2].to_vec::<f32>(), want);
}

/// The simulator's report for a 128x64 f16->f32 tile kernel with `faults`.
fn report_for(faults: Faults) -> String {
    use SmemLayoutAtomKind::K_SW128;
    let kernel = tile_kernel::<F16, f32>("umma_misuse", 128, 64, K_SW128, K_SW128, faults);
    let (a, b) = (vec![F16::from_f32(1.0); 128 * K], vec![F16::from_f32(1.0); 64 * K]);
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0f32; 128 * 64])];
    match kernel.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args) {
        Ok(_) => panic!("expected a simulator report"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn reading_d_before_the_commit_is_observed_is_reported() {
    let e = report_for(Faults { no_wait: true, ..Default::default() });
    // The epilogue warps run ahead of the MMA (nothing wrote D yet), meet it in flight, or
    // read its landed accumulator without having waited, depending on the schedule.
    assert!(
        e.contains("still in flight")
            || e.contains("has not observed the MMA's completion")
            || e.contains("nothing wrote"),
        "{e}"
    );
}

#[test]
fn overwriting_an_operand_in_flight_is_reported() {
    let e = report_for(Faults { overwrite_in_flight: true, ..Default::default() });
    assert!(e.contains("still reads"), "{e}");
}

#[test]
fn a_missing_proxy_fence_is_reported() {
    let e = report_for(Faults { no_proxy_fence: true, ..Default::default() });
    assert!(e.contains("fence.proxy.async"), "{e}");
}

#[test]
fn gemm_from_a_divergent_warp_is_reported() {
    let e = report_for(Faults { divergent_gemm: true, ..Default::default() });
    assert!(e.contains("executed by 16 of the 32 lanes"), "{e}");
}

fn panic_text(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let e = std::panic::catch_unwind(f).map(|_| ()).unwrap_err();
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
fn illegal_operations_panic_when_built() {
    use OperandMajorMode::{K as KM, MN};
    let new = |ab, acc, shape: (usize, usize), cg, src, a| {
        move || {
            MmaF16BF16Op::new(ab, acc, shape, cg, src, a, KM);
        }
    };
    let src = OperandSource::SMEM;
    let e = panic_text(new(DType::F16, DType::F32, (96, 64), CtaGroup::ONE, src, KM));
    assert!(e.contains("M is 96"), "{e}");
    let e = panic_text(new(DType::F16, DType::F32, (128, 12), CtaGroup::ONE, src, KM));
    assert!(e.contains("N is 12"), "{e}");
    let e = panic_text(new(DType::BF16, DType::F16, (128, 64), CtaGroup::ONE, src, KM));
    assert!(e.contains("Float32 only"), "{e}");
    let e = panic_text(new(DType::F32, DType::F32, (128, 64), CtaGroup::ONE, src, KM));
    assert!(e.contains("ab_dtype"), "{e}");
    // m10-cta-pair: a CTA pair's N goes in steps of 16, and its A is in shared memory.
    let e = panic_text(new(DType::F16, DType::F32, (256, 24), CtaGroup::TWO, src, KM));
    assert!(e.contains("N is 24") && e.contains("multiple of 16"), "{e}");
    let e = panic_text(new(DType::F16, DType::F32, (128, 64), CtaGroup::TWO, OperandSource::TMEM, KM));
    assert!(e.contains("OperandSource.SMEM"), "{e}");
    let e = panic_text(new(DType::F16, DType::F32, (128, 64), CtaGroup::ONE, OperandSource::TMEM, MN));
    // A from TMEM arrived with m10-umma-kinds; an MN-major A there is refused.
    assert!(e.contains("always K-major"), "{e}");
}

#[test]
fn tcgen05_mma_and_commit_need_sm100a() {
    use SmemLayoutAtomKind::K_SW128;
    let kernel = tile_kernel::<F16, f32>("umma_target", 128, 64, K_SW128, K_SW128, Faults::default());
    cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap();
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm90a).unwrap_err().message;
    assert!(e.contains("tcgen05"), "{e}");
}

#[test]
fn umma_ptx_for_sm100a() {
    use SmemLayoutAtomKind::*;
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let cases: Vec<(Kernel, u32)> = vec![
        (
            tile_kernel::<F16, f32>("umma_ptx_f16", 128, 128, K_SW128, MN_SW64, Faults::default()),
            MmaF16BF16Op::new(
                DType::F16,
                DType::F32,
                (128, 128),
                CtaGroup::ONE,
                OperandSource::SMEM,
                OperandMajorMode::K,
                OperandMajorMode::MN,
            )
            .kind()
            .idesc(),
        ),
        (
            tile_kernel::<BF16, f32>("umma_ptx_bf16", 64, 64, MN_SW32, K_INTER, Faults::default()),
            MmaF16BF16Op::new(
                DType::BF16,
                DType::F32,
                (64, 64),
                CtaGroup::ONE,
                OperandSource::SMEM,
                OperandMajorMode::MN,
                OperandMajorMode::K,
            )
            .kind()
            .idesc(),
        ),
        (
            double_buffer_kernel(),
            MmaF16BF16Op::new(
                DType::F16,
                DType::F32,
                (128, 128),
                CtaGroup::ONE,
                OperandSource::SMEM,
                OperandMajorMode::K,
                OperandMajorMode::K,
            )
            .kind()
            .idesc(),
        ),
    ];
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    for (kernel, idesc) in cases {
        let name = kernel.ir().name.clone();
        let compiled = kernel
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptx = compiled.ptx();
        for w in ["tcgen05.mma.cta_group::1.kind::f16", "tcgen05.commit", "tcgen05.ld.sync.aligned"] {
            assert!(ptx.contains(w), "{name}: no `{w}` in the PTX");
        }
        // The instruction descriptor, moved into a register as an immediate.
        assert!(ptx.contains(&format!(", {idesc};")), "{name}: no immediate {idesc} ({idesc:#x}) in the PTX");
        let Some(ptxas) = &ptxas else { continue };
        let dir = std::env::temp_dir().join(format!("cute-umma-{}-{name}", std::process::id()));
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

#[test]
fn umma_mlir_text() {
    use SmemLayoutAtomKind::K_SW128;
    let text = tile_kernel::<F16, f32>("umma_mlir", 128, 64, K_SW128, K_SW128, Faults::default()).to_mlir();
    for w in
        ["tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, q;", "nvvm.tcgen05.commit", "nvvm.tcgen05.fence <after>"]
    {
        assert!(text.contains(w), "no `{w}` in\n{text}");
    }
}

#[test]
fn mbarrier_array_tcgen05_arrivals_commit() {
    let kernel = Kernel::builder("umma_mbarrier_array").block_dim([32, 1, 1]).target(Arch::Sm100a).build(|| {
        let group = pipeline::CooperativeGroup::new(pipeline::Agent::Thread, 1);
        let array =
            pipeline::MbarrierArray::new(&alloc_smem::<u64>(2, 8), 2, (pipeline::PipelineOp::TCGen05Mma, group), 0);
        array.arrive(1, None, Some(pipeline::CtaGroup::One));
    });
    let text = kernel.to_mlir();
    assert!(text.contains("nvvm.tcgen05.commit"), "{text}");
}
