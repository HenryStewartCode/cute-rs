// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The warp-specialised GEMM's structure (a TMA or transform warp feeding the MMA warp, the
// MMA warp handing accumulator stages in TMEM to four epilogue warps, each through one of
// cute_ext's convenience pipelines) follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute_ext/blackwell/dense_gemm/input_transform_gemm.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! cute_ext's pipelines (roadmap part m11-cute-ext-pipelines) on the simulator, against plain
//! Rust: a thread-kind GenericPipeline with try-wait tokens (also on the GPU with the `cuda`
//! feature), a create_state_at retry sweep, TMA -> LD_SHARED with cute_ext's TMA loads,
//! SM100_LAUNCH_CONTROL -> LD_SHARED, the circular-buffer accounting on thread kinds with
//! (count_per_stage, count_per_iteration) = (4, 1), (2, 2) and (1, 2) and a premature refill
//! the race detector catches, and a warp-specialised GEMM through TMAToUMMAPipeline (plain and
//! circular), AsyncToUMMAPipeline and UMMAtoAsyncPipeline; arrival counts against the m9/m10
//! classes' own; the kernels compiled for sm_90a / sm_100a and assembled when `ptxas` is
//! installed; diagnostics last.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::experimental::{
    self, ArrivalCountOptions, ArrivalMask, AsyncToUMMAPipeline, GenericPipeline, OperationType, PipelineClass,
    PipelineLocale, Side, StageToken, TMAToAsyncPipeline, TMAToUMMACircularPipeline, TMAToUMMAPipeline,
    UMMAtoAsyncPipeline,
};
use cute::memory::{TmemAllocator, TmemAllocatorOptions, TmemBufferPool, compute_tmem_cols_from_layout};
use cute::pipeline::{
    Agent, CooperativeGroup, NamedBarrier, PipelineAsync, PipelineAsyncOptions, PipelineAsyncUmma,
    PipelineClcFetchAsync, PipelineClcFetchAsyncOptions, PipelineTmaAsync, PipelineTmaAsyncOptions, PipelineTmaUmma,
    PipelineTmaUmmaOptions, PipelineUmmaAsync, PipelineUmmaOptions,
};
use cute::prelude::*;
use cute::tcgen05::{self, CtaGroup, Field, Ld32x32bOp, MmaF16BF16Op, OperandSource, Repetition, SmemLayoutAtomKind};

use OperationType::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// Seeded schedules with random async completion, plus the built-in ones.
fn schedules(seeds: u64) -> Vec<SimOptions> {
    let mut runs: Vec<SimOptions> =
        Schedule::ALL.iter().map(|&schedule| SimOptions { schedule, ..SimOptions::default() }).collect();
    for seed in 0..seeds {
        runs.push(SimOptions {
            schedule: Schedule::Random { seed },
            async_completion: AsyncCompletion::Random(seed),
            ..SimOptions::default()
        });
    }
    runs
}

// ---------------------------------------------------------------------------------------------
// (1) Thread kinds: GenericPipeline with try-wait tokens
// ---------------------------------------------------------------------------------------------

const ITEMS: usize = 10;
const STAGES: usize = 2;

/// Warp 0 produces (`ST_SHARED`), warp 1 consumes (`LD_SHARED`) through a GenericPipeline of
/// two stages, each side probing first and passing its token to the acquire or wait.
/// `counts` gets the producer's (true, false) tokens and the consumer's.
fn token_kernel(arch: Arch, producer_count: i32) -> Kernel {
    Kernel::builder("cute_ext_pipe_tokens").block_dim([64, 1, 1]).target(arch).build(move || {
        let src = param_tensor::<u32>("src", Layout::col_major(ITEMS * 32));
        let sums = param_tensor::<u32>("sums", Layout::col_major(32));
        let counts = param_tensor::<u32>("counts", Layout::col_major(4));
        let pipe = GenericPipeline::create(StShared, LdShared, producer_count, 32, STAGES, None);
        let data = make_smem_tensor::<u32>(Layout::col_major(STAGES * 32));
        let lane = lane_idx();
        let zero = || Uint32::from(0u32);
        let write_counts = |base: i32, yes: Uint32, no: Uint32| {
            if_(lane.eq(0), || {
                counts.store(base, yes);
                counts.store(base + 1, no);
            });
        };
        if_else(
            warp_idx().eq(0),
            || {
                let start = (zero(), zero(), pipe.clone());
                let (yes, no, mut p) = for_range_carry(0, ITEMS as i32, 1, start, |k, (yes, no, mut p)| {
                    let token = p.producer_try_acquire(None);
                    let (_, i) = p.producer_acquire_and_get_stage(Some(token));
                    data.store(i * 32 + lane, src.load(k * 32 + lane) * 3u32 + k.to::<u32>());
                    p.producer_commit_and_advance(false, false);
                    let t = token.to::<u32>();
                    (yes + t, no + (1u32 - t), p)
                });
                p.producer_tail();
                write_counts(0, yes, no);
            },
            || {
                let start = (zero(), zero(), zero(), pipe.clone());
                let (sum, yes, no, _) = for_range_carry(0, ITEMS as i32, 1, start, |_, (sum, yes, no, mut c)| {
                    let token = c.consumer_try_wait(None);
                    let (_, i) = c.consumer_wait_and_get_stage(Some(token));
                    let x = data.load(i * 32 + lane);
                    c.consumer_release_and_advance();
                    let t = token.to::<u32>();
                    (sum + x, yes + t, no + (1u32 - t), c)
                });
                sums.store(lane, sum);
                write_counts(2, yes, no);
            },
        );
    })
}

fn token_args() -> Vec<KernelArg> {
    let src: Vec<u32> = (0..ITEMS * 32).map(|i| (i as u32 * 7919) % 1000).collect();
    vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 32]), KernelArg::buffer(&[0u32; 4])]
}

/// The sums each lane should get, and the run's token counts.
fn check_token_run(args: &[KernelArg]) -> [u32; 4] {
    let src = args[0].to_vec::<u32>();
    let want: Vec<u32> = (0..32).map(|l| (0..ITEMS).map(|k| src[k * 32 + l] * 3 + k as u32).sum()).collect();
    assert_eq!(args[1].to_vec::<u32>(), want);
    let c = args[2].to_vec::<u32>();
    assert_eq!(c[0] + c[1], ITEMS as u32);
    assert_eq!(c[2] + c[3], ITEMS as u32);
    [c[0], c[1], c[2], c[3]]
}

#[test]
fn generic_pipeline_tokens_take_both_paths() {
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    for arch in [Arch::Sm89, Arch::Sm90a] {
        let kernel = token_kernel(arch, 32);
        let mut totals = [0u32; 4];
        for opts in schedules(16) {
            let mut args = token_args();
            kernel.simulate_with(cfg, &mut args, &opts).unwrap_or_else(|e| panic!("{arch:?} {:?}: {e}", opts.schedule));
            for (t, c) in totals.iter_mut().zip(check_token_run(&args)) {
                *t += c;
            }
        }
        let [p_true, p_false, c_true, c_false] = totals;
        assert!(p_true > 0 && p_false > 0, "{arch:?}: producer tokens {p_true} true, {p_false} false");
        assert!(c_true > 0 && c_false > 0, "{arch:?}: consumer tokens {c_true} true, {c_false} false");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn generic_pipeline_on_the_gpu() {
    use cute::cuda::CudaContext;
    let Ok(ctx) = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")) else { return };
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let arch = ctx.best_target(&tc).expect("a target");
    // sm_89 and below spin on mbarrier.test_wait; sm_90 and above use try_wait. Above sm_89 the
    // kernel is built for the device's own target: architecture-specific code (sm_90a) does not
    // run on a newer GPU such as sm_120a.
    let kernel = token_kernel(if arch.sm() < 90 { Arch::Sm89 } else { arch }, 32);
    let compiled = kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{e}"));
    let f = ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"));
    for _ in 0..5 {
        let mut args = token_args();
        ctx.launch(&kernel, &f, LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
        check_token_run(&args);
    }
}

// ---------------------------------------------------------------------------------------------
// (2) create_pipeline_state_at: a consumer sweeping the stages out of order
// ---------------------------------------------------------------------------------------------

const SWEEP_STAGES: usize = 4;
const SWEEP_ROUNDS: usize = 3;

/// Low-level ops: warp 0 fills the four stages in order each round; warp 1 visits them in
/// reverse with states made by create_pipeline_state_at, probing each first, and writes what it
/// read at (round, stage).
fn sweep_kernel() -> Kernel {
    Kernel::builder("cute_ext_pipe_sweep").block_dim([64, 1, 1]).target(Arch::Sm90a).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(SWEEP_ROUNDS * SWEEP_STAGES * 32));
        let (pipe, pstate, _) =
            experimental::create_pipeline(SWEEP_STAGES, StShared, LdShared, 32, 32, ArrivalMask::None, None);
        let data = make_smem_tensor::<i32>(Layout::col_major(SWEEP_STAGES * 32));
        let lane = lane_idx();
        if_else(
            warp_idx().eq(0),
            || {
                let total = (SWEEP_ROUNDS * SWEEP_STAGES) as i32;
                let s = for_range_carry(0, total, 1, pstate, |k, s| {
                    let s = experimental::producer_acquire(&pipe, &s);
                    let (_, i) = experimental::get_pipeline_produce_stage(&pipe, &s);
                    data.store(i * 32 + lane, k * 100 + lane);
                    experimental::producer_commit(&pipe, &s, false, false);
                    experimental::pipeline_advance_iterator(&pipe, &s)
                });
                experimental::pipeline_tail(&pipe, &s);
            },
            || {
                for r in 0..SWEEP_ROUNDS {
                    for st in (0..SWEEP_STAGES).rev() {
                        let s = experimental::create_pipeline_state_at(&pipe, SWEEP_STAGES, st as i32, (r % 2) as i32);
                        let token = experimental::consumer_try_wait(&pipe, &s, None);
                        let s = experimental::consumer_wait_with(&pipe, &s, Some(token));
                        let x = data.load(s.index() * 32 + lane);
                        experimental::consumer_release(&pipe, &s, false, false);
                        out.store(((r * SWEEP_STAGES + st) * 32) as i32 + lane, x);
                    }
                }
            },
        );
    })
}

#[test]
fn create_state_at_sweeps_stages_out_of_order() {
    let kernel = sweep_kernel();
    for opts in schedules(8) {
        let mut args = vec![KernelArg::buffer(&vec![-1i32; SWEEP_ROUNDS * SWEEP_STAGES * 32])];
        kernel
            .simulate_with(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args, &opts)
            .unwrap_or_else(|e| panic!("{:?}: {e}", opts.schedule));
        let got = args[0].to_vec::<i32>();
        for r in 0..SWEEP_ROUNDS {
            for st in 0..SWEEP_STAGES {
                for l in 0..32 {
                    let k = (r * SWEEP_STAGES + st) as i32;
                    assert_eq!(got[(r * SWEEP_STAGES + st) * 32 + l], k * 100 + l as i32, "({r}, {st}, {l})");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// (3) TMA -> LD_SHARED with cute_ext's TMA loads
// ---------------------------------------------------------------------------------------------

const TA_TILES: usize = 6;

/// Warp 4 TMA-loads six 32x32 f32 tiles through a two-stage TMAToAsyncPipeline; warps 0-3
/// (LD_SHARED, 128 arrivals) each sum eight elements of every tile.
fn tma_async_kernel() -> Kernel {
    Kernel::builder("cute_ext_pipe_tma_async").block_dim([160, 1, 1]).target(Arch::Sm90a).build(|| {
        let a = param_tensor_aligned::<f32>("a", Layout::new((TA_TILES * 32, 32), (32, 1)), 16);
        let out = param_tensor::<f32>("out", Layout::col_major(128));
        let s = experimental::allocate::<f32>(AddressSpace::Smem, Layout::new((32, 32, 2), (32, 1, 1024)), 128);
        let pipe = TMAToAsyncPipeline::create(2, LdShared, 128);
        assert_eq!(pipe.raw_pipeline().class(), PipelineClass::TmaAsync);
        if_(warp_idx().eq(4), || {
            let mut p = for_range_carry(0, TA_TILES as i32, 1, pipe.clone(), |k, mut p| {
                let (tok, i) = p.producer_acquire_and_get_stage(None);
                experimental::tma_load(
                    &local_tile(&a, (32, 32), (k, 0)),
                    &s.slice((.., .., i)),
                    &experimental::get_mbarrier(&tok),
                );
                p.producer_commit_and_advance(false, false);
                p
            });
            p.producer_tail();
        });
        if_(warp_idx().lt(4), || {
            let t = thread_idx().x;
            let (sum, _) =
                for_range_carry(0, TA_TILES as i32, 1, (Float32::from(0.0), pipe.clone()), |_, (sum, mut c)| {
                    let (_, i) = c.consumer_wait_and_get_stage(None);
                    let mut acc = sum;
                    for j in 0..8 {
                        let e = t * 8 + j;
                        acc = acc + s.load((e / 32, e % 32, i));
                    }
                    c.consumer_release_and_advance();
                    (acc, c)
                });
            out.store(t, sum);
        });
    })
}

#[test]
fn tma_to_async_pipeline_matches_rust() {
    let kernel = tma_async_kernel();
    let a: Vec<f32> = (0..TA_TILES * 1024).map(|i| ((i * 13) % 17) as f32 - 8.0).collect();
    let want: Vec<f32> =
        (0..128).map(|t| (0..TA_TILES).map(|k| (0..8).map(|j| a[k * 1024 + t * 8 + j]).sum::<f32>()).sum()).collect();
    for opts in schedules(4) {
        let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&[0f32; 128])];
        kernel
            .simulate_with(LaunchConfig::new([1, 1, 1], [160, 1, 1]), &mut args, &opts)
            .unwrap_or_else(|e| panic!("{:?}: {e}", opts.schedule));
        assert_eq!(args[1].to_vec::<f32>(), want);
    }
}

// ---------------------------------------------------------------------------------------------
// (4) SM100_LAUNCH_CONTROL -> LD_SHARED
// ---------------------------------------------------------------------------------------------

/// CTA 0's warp 0 issues four cluster launch control queries through a two-stage
/// GenericPipeline; warp 1 decodes each response. With one CTA resident, the queries cancel
/// CTAs 1, 2 and 3 in launch order, and the fourth finds the queue empty.
fn clc_kernel() -> Kernel {
    Kernel::builder("cute_ext_pipe_clc").block_dim([64, 1, 1]).target(Arch::Sm100a).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(16));
        let response = make_smem_tensor_aligned::<i32>(Layout::col_major(8), 16);
        let count = experimental::get_arrival_count(
            2,
            Sm100LaunchControl,
            LdShared,
            Side::Producer,
            &ArrivalCountOptions::new().num_threads_per_cta(32).elect_one_sync(true),
        );
        assert_eq!(count.as_constant(), Some(1));
        let pipe = GenericPipeline::create(Sm100LaunchControl, LdShared, count, 32, 2, None);
        assert_eq!(pipe.raw_pipeline().class(), PipelineClass::ClcFetchAsync);
        let lane = lane_idx();
        if_(block_idx().x.eq(0), || {
            if_else(
                warp_idx().eq(0),
                || {
                    let mut p = for_range_carry(0, 4, 1, pipe.clone(), |_, mut p| {
                        let (tok, i) = p.producer_acquire_and_get_stage(None);
                        let mbar = experimental::get_mbarrier(&tok);
                        if_(lane.eq(0), || {
                            experimental::mbarrier_expect_tx(&mbar, 16, None, false);
                            issue_clc_query(&mbar, &response.ptr().offset(i * 4), false);
                        });
                        p.producer_commit_and_advance(false, false);
                        p
                    });
                    p.producer_tail();
                },
                || {
                    for_range_carry(0, 4, 1, pipe.clone(), |k, mut c| {
                        let (_, i) = c.consumer_wait_and_get_stage(None);
                        let (x, _, _, valid) = clc_response(&response.ptr().offset(i * 4));
                        if_(lane.eq(0), || {
                            out.store(k * 2, x);
                            out.store(k * 2 + 1, valid);
                        });
                        c.consumer_release_and_advance();
                        c
                    });
                },
            );
        });
    })
}

#[test]
fn launch_control_pipeline_hands_out_responses() {
    let kernel = clc_kernel();
    let mut args = vec![KernelArg::buffer(&[-1i32; 16])];
    kernel.simulate(LaunchConfig::new([4, 1, 1], [64, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let got = args[0].to_vec::<i32>();
    assert_eq!(&got[..6], &[1, 1, 2, 1, 3, 1], "{got:?}");
    assert_eq!(got[7], 0, "{got:?}");
}

// ---------------------------------------------------------------------------------------------
// (5) The circular buffer on thread kinds
// ---------------------------------------------------------------------------------------------

const CB_STAGES: usize = 3;
const CB_UNITS: usize = 24;

/// Warp 0 fills stages of `c` units (32 lanes each) through the low-level ops; warp 1
/// consumes `i` units per iteration through the circular operations and sums them per lane.
/// With `premature`, warp 1 instead releases each stage as soon as it starts it (a raw
/// consumer_release at unit offset 0), so the producer refills it while units remain.
fn circular_kernel(c: usize, i: usize, premature: bool) -> Kernel {
    Kernel::builder("cute_ext_pipe_circular").block_dim([64, 1, 1]).target(Arch::Sm90a).build(move || {
        let src = param_tensor::<u32>("src", Layout::col_major(CB_UNITS * 32));
        let sums = param_tensor::<u32>("sums", Layout::col_major(32));
        let (pipe, pstate, cstate) = experimental::create_pipeline(CB_STAGES, StShared, LdShared, 32, 32, false, None);
        let data = make_smem_tensor::<u32>(Layout::col_major(CB_STAGES * c * 32));
        let lane = lane_idx();
        if_else(
            warp_idx().eq(0),
            || {
                let s = for_range_carry(0, (CB_UNITS / c) as i32, 1, pstate, |k, s| {
                    let s = experimental::producer_acquire(&pipe, &s);
                    for u in 0..c as i32 {
                        let unit = k * c as i32 + u;
                        data.store(
                            (s.index() * c as i32 + u) * 32 + lane,
                            src.load(unit * 32 + lane) + unit.to::<u32>(),
                        );
                    }
                    experimental::producer_commit(&pipe, &s, false, false);
                    experimental::pipeline_advance_iterator(&pipe, &s)
                });
                experimental::pipeline_tail(&pipe, &s);
            },
            || {
                let cb = experimental::create_circular_buffer_pipeline(&pipe, &cstate, CB_STAGES, c, i);
                let (sum, _) =
                    for_range_carry(0, (CB_UNITS / i) as i32, 1, (Uint32::from(0u32), cb), |_, (sum, cb)| {
                        experimental::circular_buffer_pipeline_consume(&pipe, &cb, false);
                        if premature {
                            if_(cb.unit_offset().eq(0), || {
                                experimental::consumer_release(&pipe, &cb.stage_state(), false, false)
                            });
                        }
                        let mut acc = sum;
                        for j in 0..i as i32 {
                            let (stage, unit) = if i <= c {
                                (cb.stage_index(), cb.unit_offset() + j)
                            } else {
                                ((cb.stage_index() + j / c as i32) % CB_STAGES as i32, Int32::from(j % c as i32))
                            };
                            acc = acc + data.load((stage * c as i32 + unit) * 32 + lane);
                        }
                        if !premature {
                            experimental::circular_buffer_pipeline_consumer_release(&pipe, &cb, false, false);
                        }
                        (acc, experimental::circular_buffer_pipeline_advance_iterator(&pipe, &cb))
                    });
                sums.store(lane, sum);
            },
        );
    })
}

fn circular_args() -> (Vec<KernelArg>, Vec<u32>) {
    let src: Vec<u32> = (0..CB_UNITS * 32).map(|i| (i as u32 * 31) % 97).collect();
    let want = (0..32).map(|l| (0..CB_UNITS).map(|u| src[u * 32 + l] + u as u32).sum()).collect();
    (vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 32])], want)
}

#[test]
fn circular_buffer_consumes_units_in_order() {
    for (c, i) in [(4, 1), (2, 2), (1, 2), (4, 2), (2, 4)] {
        let kernel = circular_kernel(c, i, false);
        for opts in schedules(6) {
            let (mut args, want) = circular_args();
            kernel
                .simulate_with(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args, &opts)
                .unwrap_or_else(|e| panic!("({c}, {i}) {:?}: {e}", opts.schedule));
            assert_eq!(args[1].to_vec::<u32>(), want, "({c}, {i})");
        }
    }
}

/// A stage released before its last unit is read lets the producer refill it under the
/// consumer: the race detector reports the producer's store against the consumer's load.
#[test]
fn circular_buffer_premature_refill_is_a_race() {
    let kernel = circular_kernel(4, 1, true);
    let mut reported = false;
    for opts in schedules(8) {
        let (mut args, _) = circular_args();
        if let Err(e) = kernel.simulate_with(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args, &opts) {
            let e = e.to_string();
            assert!(e.contains("data race"), "{e}");
            reported = true;
        }
    }
    assert!(reported, "no schedule reported the premature refill");
}

// ---------------------------------------------------------------------------------------------
// (6) The warp-specialised GEMM: TMA/threads -> UMMA -> T2R
// ---------------------------------------------------------------------------------------------

const M: usize = 128;
const NT: usize = 64;
const N_TILES: usize = 4;
const K: usize = 192;
const N: usize = NT * N_TILES;
/// k-blocks (units) of 16 per output tile.
const UNITS: usize = K / 16;
const AB_STAGES: usize = 4;
const ACC_STAGES: usize = 2;
const EPI: usize = 128;
const MMA_WARP: i32 = 4;

/// How the A/B stages are filled and consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Feed {
    /// A TMA warp and TMAToUMMAPipeline, stages of 64 (4 units).
    Tma,
    /// TMAToUMMACircularPipeline with (count_per_stage, count_per_iteration).
    Circular(usize, usize),
    /// Two transform warps storing with ST_SHARED through AsyncToUMMAPipeline.
    Threads,
}

impl Feed {
    fn units_per_stage(self) -> usize {
        match self {
            Feed::Circular(c, _) => c,
            _ => 4,
        }
    }
    fn threads(self) -> usize {
        if self == Feed::Threads { EPI + 96 } else { EPI + 64 }
    }
}

fn umma_op() -> TiledMma {
    let op = MmaF16BF16Op::new(
        DType::F16,
        DType::F32,
        (M, NT),
        CtaGroup::ONE,
        OperandSource::SMEM,
        OperandMajorMode::K,
        OperandMajorMode::K,
    );
    make_tiled_mma(op, Layout::col_major((1, 1, 1)))
}

fn staged(mn: usize, ks: usize) -> ComposedLayout {
    let kind = match ks {
        16 => SmemLayoutAtomKind::K_SW32,
        32 => SmemLayoutAtomKind::K_SW64,
        _ => SmemLayoutAtomKind::K_SW128,
    };
    tile_to_shape_swizzled(&tcgen05::make_smem_layout_atom::<F16>(kind), (mn, ks, AB_STAGES))
}

/// The producer side of the three A/B pipelines.
trait AbProducer: Carry + Clone {
    fn acquire(&mut self) -> (StageToken, Int32);
    fn commit_advance(&mut self);
    fn tail(&mut self);
}

macro_rules! ab_producer {
    ($t:ty) => {
        impl AbProducer for $t {
            fn acquire(&mut self) -> (StageToken, Int32) {
                self.producer_acquire_and_get_stage(None)
            }
            fn commit_advance(&mut self) {
                self.producer_commit_and_advance(false, false);
            }
            fn tail(&mut self) {
                self.producer_tail();
            }
        }
    };
}
ab_producer!(TMAToUMMAPipeline);
ab_producer!(TMAToUMMACircularPipeline);
ab_producer!(AsyncToUMMAPipeline);

/// The TMA warp: per output tile, every stage of K.
fn tma_producer<P: AbProducer>(
    ab: &P,
    a: &Tensor<F16>,
    b: &Tensor<F16>,
    sa: &Tensor<F16>,
    sb: &Tensor<F16>,
    ks: usize,
) {
    let mut p = for_range_carry(0, N_TILES as i32, 1, ab.clone(), |n, mut p| {
        for kt in 0..(K / ks) as i32 {
            let (tok, i) = p.acquire();
            let bar = experimental::get_mbarrier(&tok);
            experimental::tma_load(&local_tile(a, (M, ks), (0, kt)), &sa.slice((.., .., i)), &bar);
            experimental::tma_load(&local_tile(b, (NT, ks), (n, kt)), &sb.slice((.., .., i)), &bar);
            p.commit_advance();
        }
        p
    });
    p.tail();
}

/// The transform warps (64 threads): per stage, plain loads from global memory and stores to
/// the swizzled stage (ST_SHARED; the commit fences for the async proxy).
fn thread_producer(ab: &AsyncToUMMAPipeline, a: &Tensor<F16>, b: &Tensor<F16>, sa: &Tensor<F16>, sb: &Tensor<F16>) {
    let ks = 64usize;
    let pt = thread_idx().x - (EPI + 32) as i32;
    let mut p = for_range_carry(0, N_TILES as i32, 1, ab.clone(), |n, mut p| {
        for kt in 0..(K / ks) as i32 {
            let (_, i) = p.acquire();
            let (sta, stb) = (sa.slice((.., .., i)), sb.slice((.., .., i)));
            for_range(0, (M * ks / 64) as i32, 1, |j| {
                let e = j * 64 + pt;
                let (r, k) = (e / ks as i32, e % ks as i32);
                sta.store((r, k), a.load((r, kt * ks as i32 + k)));
            });
            for_range(0, (NT * ks / 64) as i32, 1, |j| {
                let e = j * 64 + pt;
                let (r, k) = (e / ks as i32, e % ks as i32);
                stb.store((r, k), b.load((n * NT as i32 + r, kt * ks as i32 + k)));
            });
            p.commit_advance();
        }
        p
    });
    p.tail();
}

/// One k-block's MMA from stage `stage`, unit `kb`.
#[allow(clippy::too_many_arguments)]
fn mma_unit(
    mma: &TiledMma,
    m: &mut TiledMma,
    sa: &Tensor<F16>,
    sb: &Tensor<F16>,
    acc: &Tensor<f32>,
    stage: Int32,
    kb: usize,
) {
    let thr = mma.get_slice(0);
    let fa = mma.make_fragment_a(&thr.partition_a(&sa.slice((.., .., stage))));
    let fb = mma.make_fragment_b(&thr.partition_b(&sb.slice((.., .., stage))));
    gemm(m, acc, &fa.slice((.., .., kb)), &fb.slice((.., .., kb)), acc);
    m.set(Field::ACCUMULATE, true);
}

/// The warp-specialised GEMM `C = A B^T` (A `(M, K)`, B `(N, K)`, K-major f16; C `(M, N)`
/// row-major f32).
fn gemm_kernel(name: &'static str, feed: Feed) -> Kernel {
    let threads = feed.threads();
    Kernel::builder(name).block_dim([threads as u32, 1, 1]).target(Arch::Sm100a).build(move || {
        let a = param_tensor_aligned::<F16>("a", Layout::new((M, K), (K, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((N, K), (K, 1)), 16);
        let c = param_tensor::<f32>("c", Layout::new((M, N), (N, 1)));
        let mma = umma_op();
        let ups = feed.units_per_stage();
        let ks = ups * 16;
        let sa = experimental::allocate::<F16>(AddressSpace::Smem, staged(M, ks), 1024);
        let sb = experimental::allocate::<F16>(AddressSpace::Smem, staged(NT, ks), 1024);

        let acc_layout = mma.make_fragment_c::<f32>(&mma.partition_shape_c((M, NT))).layout().clone();
        let stage_cols = compute_tmem_cols_from_layout(&acc_layout, DType::F32);
        let cols = ACC_STAGES * stage_cols;
        let mut tmem = TmemAllocator::new(
            NamedBarrier::new(1, EPI + 32),
            TmemAllocatorOptions { arch: Arch::Sm100a, ..Default::default() },
        );
        tmem.allocate(cols);
        let acc_view = |tmem: &TmemAllocator| {
            tmem.wait_for_alloc();
            let mut pool = TmemBufferPool::new(&tmem.retrieve_ptr::<f32>(), cols);
            let acc0 = pool.allocate_tensor::<f32>(&acc_layout);
            let _acc1 = pool.allocate::<f32>(&acc_layout);
            acc0
        };

        let mma_kind = Sm100Mma1SmSs;
        let acc_pipe = UMMAtoAsyncPipeline::create(ACC_STAGES, Sm100CopyT2r, EPI as i32, mma_kind, None);
        assert_eq!(acc_pipe.raw_pipeline().class(), PipelineClass::UmmaAsync);

        // The MMA warp: per output tile, acquire an accumulator stage, wait for each A/B stage,
        // issue its four k-blocks and release it (one elected commit), then commit the
        // accumulator.
        macro_rules! mma_warp {
            ($ab:expr) => {
                if_(warp_idx().eq(MMA_WARP), || {
                    let acc0 = acc_view(&tmem);
                    let start = (acc_pipe.clone(), $ab.clone());
                    let (mut accp, _) = for_range_carry(0, N_TILES as i32, 1, start, |_, (mut accp, mut ab)| {
                        let (_, ai) = accp.producer_acquire_and_get_stage(None);
                        let acc = make_tensor(acc0.ptr().offset(ai * stage_cols as i32), acc0.layout().clone());
                        let mut m = mma.clone();
                        m.set(Field::ACCUMULATE, false);
                        for _ in 0..UNITS / 4 {
                            let (_, i) = ab.consumer_wait_and_get_stage(None);
                            for kb in 0..4 {
                                mma_unit(&mma, &mut m, &sa, &sb, &acc, i, kb);
                            }
                            ab.consumer_release_and_advance();
                        }
                        accp.producer_commit_and_advance(false, false);
                        (accp, ab)
                    });
                    accp.producer_tail();
                });
            };
        }

        match feed {
            Feed::Tma => {
                let ab = TMAToUMMAPipeline::create(AB_STAGES, mma_kind, None, None);
                assert_eq!(ab.raw_pipeline().class(), PipelineClass::TmaUmma);
                if_(warp_idx().eq(5), || tma_producer(&ab, &a, &b, &sa, &sb, ks));
                mma_warp!(ab);
            }
            Feed::Threads => {
                let ab = AsyncToUMMAPipeline::create(AB_STAGES, StShared, 64, mma_kind);
                if_(warp_idx().ge(5), || thread_producer(&ab, &a, &b, &sa, &sb));
                mma_warp!(ab);
            }
            Feed::Circular(cu, it) => {
                let ab = TMAToUMMACircularPipeline::create(AB_STAGES, mma_kind, cu, it, None, None);
                assert_eq!((ab.count_per_stage(), ab.count_per_iteration()), (cu, it));
                if_(warp_idx().eq(5), || tma_producer(&ab, &a, &b, &sa, &sb, ks));
                if_(warp_idx().eq(MMA_WARP), || {
                    let acc0 = acc_view(&tmem);
                    let (mut accp, _) = for_range_carry(
                        0,
                        N_TILES as i32,
                        1,
                        (acc_pipe.clone(), ab.clone()),
                        |_, (mut accp, mut ab)| {
                            let (_, ai) = accp.producer_acquire_and_get_stage(None);
                            let acc = make_tensor(acc0.ptr().offset(ai * stage_cols as i32), acc0.layout().clone());
                            let mut m = mma.clone();
                            m.set(Field::ACCUMULATE, false);
                            for iter in 0..UNITS / it {
                                ab.consumer_wait();
                                let cb = ab.circular_state();
                                for j in 0..it {
                                    let unit = iter * it + j;
                                    let stage = if it <= cu {
                                        cb.stage_index()
                                    } else {
                                        (cb.stage_index() + (j / cu) as i32) % AB_STAGES as i32
                                    };
                                    mma_unit(&mma, &mut m, &sa, &sb, &acc, stage, unit % cu);
                                }
                                ab.consumer_release_and_advance();
                            }
                            accp.producer_commit_and_advance(false, false);
                            (accp, ab)
                        },
                    );
                    accp.producer_tail();
                });
            }
        }

        // Epilogue warps 0-3: wait for an accumulator stage, load it from TMEM, release it
        // (SM100_COPY_T2R: the release waits for the loads) and store the tile.
        if_(warp_idx().lt(4), || {
            let acc0 = acc_view(&tmem);
            let t = thread_idx().x;
            for_range_carry(0, N_TILES as i32, 1, acc_pipe.clone(), |n, mut cc| {
                let (_, ei) = cc.consumer_wait_and_get_stage(None);
                let acc = make_tensor(acc0.ptr().offset(ei * stage_cols as i32), acc0.layout().clone());
                let block = acc.layout().mode(0);
                let tile = make_tensor(acc.ptr(), Layout::from_modes([block.mode(0), block.mode(1)]));
                let ctile = local_tile(&c, (M, NT), (0, n));
                let tiled = tcgen05::make_tmem_copy(&make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X64)), &tile);
                let thr = tiled.get_slice(t);
                let (src, dst) = (thr.partition_s(&tile), thr.partition_d(&ctile));
                let frag = make_fragment_like::<f32, _>(&dst);
                copy(&tiled, &src, &frag);
                cc.consumer_release_and_advance();
                basic_copy(&frag, &dst);
                cc
            });
            tmem.relinquish_alloc_permit();
            NamedBarrier::new(2, EPI).arrive_and_wait();
            tmem.free(&acc0.ptr(), 0);
        });
    })
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

fn gemm_inputs() -> (Vec<KernelArg>, Vec<f32>) {
    let (av, bv) = (small_ints(M * K, 11), small_ints(N * K, 13));
    let a: Vec<F16> = av.iter().map(|&x| F16::from_f32(x)).collect();
    let b: Vec<F16> = bv.iter().map(|&x| F16::from_f32(x)).collect();
    let mut want = vec![0f32; M * N];
    for r in 0..M {
        for col in 0..N {
            want[r * N + col] = (0..K).map(|k| av[r * K + k] * bv[col * K + k]).sum();
        }
    }
    (vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0f32; M * N])], want)
}

fn run_gemm(name: &'static str, feed: Feed) {
    let kernel = gemm_kernel(name, feed);
    let (mut args, want) = gemm_inputs();
    let report = kernel
        .simulate(LaunchConfig::new([1, 1, 1], [feed.threads() as u32, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(report.tcgen05_mma_instructions, (N_TILES * UNITS) as u64, "{name}");
    assert_eq!(args[2].to_vec::<f32>(), want, "{name}");
}

#[test]
fn tma_to_umma_gemm_matches_rust() {
    run_gemm("cute_ext_pipe_gemm_tma", Feed::Tma);
}

#[test]
fn thread_to_umma_gemm_matches_rust() {
    run_gemm("cute_ext_pipe_gemm_threads", Feed::Threads);
}

#[test]
fn circular_tma_to_umma_gemms_match_rust() {
    run_gemm("cute_ext_pipe_gemm_c41", Feed::Circular(4, 1));
    run_gemm("cute_ext_pipe_gemm_c22", Feed::Circular(2, 2));
    run_gemm("cute_ext_pipe_gemm_c12", Feed::Circular(1, 2));
}

// ---------------------------------------------------------------------------------------------
// (7) Arrival counts against the classes'
// ---------------------------------------------------------------------------------------------

fn count(
    stage: usize,
    p: impl Into<experimental::OperationTypes>,
    c: impl Into<experimental::OperationTypes>,
    side: Side,
    o: ArrivalCountOptions,
) -> i64 {
    experimental::get_arrival_count(stage, p, c, side, &o).as_constant().map(i64::from).expect("a static count")
}

fn opts(threads: i32) -> ArrivalCountOptions {
    ArrivalCountOptions::new().num_threads_per_cta(threads)
}

#[test]
fn arrival_counts_agree_with_the_classes() {
    let pair = Layout::col_major((2usize, 1usize, 1usize, 1usize));
    let mcast = Layout::col_major((1usize, 2usize, 2usize, 1usize));
    Kernel::builder("cute_ext_pipe_counts").block_dim([256, 1, 1]).target(Arch::Sm100a).build(move || {
        let static_count = |a: &cute::pipeline::MbarrierArray| a.arrive_count().as_static().expect("static");
        let warps = |n: i32| CooperativeGroup::new(Agent::Warp, n);
        let threads = |n: i32| CooperativeGroup::new(Agent::Thread, n);
        // Thread kinds -> thread kinds: PipelineAsync counts its groups' threads.
        let cls = PipelineAsync::create(
            2,
            warps(2),
            warps(4),
            PipelineAsyncOptions { defer_sync: true, ..Default::default() },
        );
        assert_eq!(count(2, StShared, LdShared, Side::Producer, opts(64)), static_count(cls.sync_object_full()));
        assert_eq!(count(2, StShared, LdShared, Side::Consumer, opts(128)), static_count(cls.sync_object_empty()));
        assert_eq!(count(2, [StShared, Sm100CopyR2t], LdShared, Side::Producer, opts(128).elect_one_sync(true)), 4);
        assert_eq!(
            count(2, [StShared, Sm100CopyR2t], LdShared, Side::Producer, opts(64).op_subset(&[Sm100CopyR2t])),
            64
        );
        assert_eq!(count(2, SregCtaid, LdShared, Side::Producer, opts(1)), 1);
        // TMA -> thread kinds: one TMA arrival; the class's consumers signal once per warp.
        let cls = PipelineTmaAsync::create(
            2,
            threads(1),
            warps(4),
            0,
            PipelineTmaAsyncOptions { defer_sync: true, ..Default::default() },
        );
        assert_eq!(
            count(2, Sm90TmaLoad, LdShared, Side::Producer, opts(32).elect_one_sync(true)),
            static_count(cls.sync_object_full())
        );
        assert_eq!(count(2, Sm90TmaLoad, LdShared, Side::Producer, opts(32)), 1);
        let cls = PipelineTmaAsync::create(
            2,
            threads(1),
            threads(4),
            0,
            PipelineTmaAsyncOptions { defer_sync: true, ..Default::default() },
        );
        assert_eq!(
            count(2, Sm90TmaLoad, LdShared, Side::Consumer, opts(128).elect_one_sync(true)),
            static_count(cls.sync_object_empty())
        );
        // TMA -> UMMA, one CTA: one arrival each side.
        let one = threads(1);
        let cls =
            PipelineTmaUmma::create(2, one, one, 0, PipelineTmaUmmaOptions { defer_sync: true, ..Default::default() });
        assert_eq!(
            count(2, Sm90TmaLoad, Sm100Mma1SmSs, Side::Producer, opts(32).elect_one_sync(true)),
            static_count(cls.sync_object_full())
        );
        assert_eq!(
            count(2, Sm90TmaLoad, Sm100Mma1SmSs, Side::Consumer, opts(32).elect_one_sync(true)),
            static_count(cls.sync_object_empty())
        );
        // Multicast TMA -> UMMA on a (1, 2, 2, 1) cluster: one commit per multicast partner.
        let cls = PipelineTmaUmma::create(
            2,
            one,
            warps(1),
            0,
            PipelineTmaUmmaOptions {
                cta_layout_vmnk: Some(mcast.clone()),
                enable_multicast_signaling: true,
                defer_sync: true,
                ..Default::default()
            },
        );
        assert_eq!(
            count(2, Sm90TmaLoadMulticast, Sm100Mma1SmSs, Side::Consumer, opts(32).elect_one_sync(true).fan_in(3)),
            static_count(cls.sync_object_empty())
        );
        // 2SM TMA -> 2SM UMMA: the leader alone arms the full barrier.
        assert_eq!(
            count(
                2,
                Sm100TmaLoad2Sm,
                Sm100Mma2SmSs,
                Side::Producer,
                opts(32).elect_one_sync(true).elect_leader_cta(true)
            ),
            1
        );
        assert_eq!(count(2, Sm100TmaLoad2Sm, Sm100Mma2SmSs, Side::Producer, opts(32).elect_one_sync(true)), 2);
        // Thread kinds -> UMMA.
        let cls = PipelineAsyncUmma::create(
            2,
            threads(64),
            one,
            PipelineUmmaOptions { defer_sync: true, ..Default::default() },
        );
        assert_eq!(count(2, StShared, Sm100Mma1SmSs, Side::Producer, opts(64)), static_count(cls.sync_object_full()));
        assert_eq!(
            count(2, StShared, Sm100Mma1SmSs, Side::Consumer, opts(32).elect_one_sync(true)),
            static_count(cls.sync_object_empty())
        );
        // UMMA -> thread kinds, one CTA and a pair (the pair's consumers release to the leader:
        // the class's convention counts both CTAs' threads).
        let cls = PipelineUmmaAsync::create(
            2,
            one,
            threads(128),
            PipelineUmmaOptions { defer_sync: true, ..Default::default() },
        );
        assert_eq!(
            count(2, Sm100Mma1SmSs, Sm100CopyT2r, Side::Producer, opts(32).elect_one_sync(true)),
            static_count(cls.sync_object_full())
        );
        assert_eq!(
            count(2, Sm100Mma1SmSs, Sm100CopyT2r, Side::Consumer, opts(128)),
            static_count(cls.sync_object_empty())
        );
        let cls = PipelineUmmaAsync::create(
            2,
            one,
            threads(256),
            PipelineUmmaOptions { cta_layout_vmnk: Some(pair.clone()), defer_sync: true, ..Default::default() },
        );
        assert_eq!(
            count(2, Sm100Mma2SmSs, Sm100CopyT2r, Side::Consumer, opts(128)),
            static_count(cls.sync_object_empty())
        );
        assert_eq!(count(2, Sm100Mma2SmSs, Sm100CopyT2r, Side::Producer, opts(32).elect_one_sync(true).fan_in(1)), 1);
        // Cluster launch control -> LD_SHARED.
        let cls = PipelineClcFetchAsync::create(
            2,
            one,
            threads(64),
            16,
            PipelineClcFetchAsyncOptions { defer_sync: true, ..Default::default() },
        );
        assert_eq!(
            count(2, Sm100LaunchControl, LdShared, Side::Producer, opts(32).elect_one_sync(true)),
            static_count(cls.sync_object_full())
        );
        assert_eq!(
            count(2, Sm100LaunchControl, LdShared, Side::Consumer, opts(64)),
            static_count(cls.sync_object_empty())
        );
        // A cluster locale: a pair's leader owns the full barriers, so both CTAs' producers count.
        let locale =
            PipelineLocale::cluster(experimental::MbarrierOwner::CtaPairLeader, experimental::MbarrierOwner::All);
        assert_eq!(count(2, StShared, LdShared, Side::Producer, opts(32).locale(locale)), 64);
        assert_eq!(count(2, StShared, LdShared, Side::Consumer, opts(32).locale(locale)), 32);

        // The convenience classes' barriers carry those counts.
        let p = TMAToUMMAPipeline::create(2, Sm100Mma1SmSs, None, None);
        assert_eq!(static_count(p.raw_pipeline().full_barriers()), 1);
        assert_eq!(static_count(p.raw_pipeline().empty_barriers()), 1);
        let p = TMAToUMMAPipeline::create(2, Sm100Mma2SmSs, Some(Sm100TmaLoad2Sm), Some(&pair));
        assert_eq!(static_count(p.raw_pipeline().full_barriers()), 1);
        assert_eq!(static_count(p.raw_pipeline().empty_barriers()), 1);
        assert!(p.raw_pipeline().arrival_mask().is_some());
        let p = TMAToUMMAPipeline::create(2, Sm100Mma1SmSs, Some(Sm90TmaLoadMulticast), Some(&mcast));
        assert_eq!(static_count(p.raw_pipeline().empty_barriers()), 3);
        let p = UMMAtoAsyncPipeline::create(2, Sm100CopyT2r, 256, Sm100Mma2SmTs, Some(&pair));
        assert_eq!(static_count(p.raw_pipeline().full_barriers()), 1);
        assert_eq!(p.raw_pipeline().owners().1, experimental::MbarrierOwner::CtaPairLeader);
        let p = TMAToAsyncPipeline::create(2, Sm100CopyT2r, 128);
        assert_eq!(static_count(p.raw_pipeline().full_barriers()), 1);
        let p = AsyncToUMMAPipeline::create(2, [StShared, Sm100CopyR2t], 128, Sm100Mma1SmTs);
        assert_eq!(p.raw_pipeline().class(), PipelineClass::AsyncUmma);
        assert_eq!(static_count(p.raw_pipeline().empty_barriers()), 1);
        assert_eq!(p.raw_pipeline().type_string(), "!lir.pipeline<2, [SM100_COPY_R2T, ST_SHARED] -> SM100_MMA_1SM_TS>");
    });
}

// ---------------------------------------------------------------------------------------------
// (8) Compiling: sm_90a and sm_100a
// ---------------------------------------------------------------------------------------------

/// Compile `kernel` for `arch` and assemble it when `ptxas` is installed; returns the PTX.
fn assemble(kernel: &Kernel, tc: &Toolchain, arch: Arch, flag: &str) -> String {
    let ptx = kernel.compile_ptx(tc, arch).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    let ptxas = std::env::var("CUTE_PTXAS").unwrap_or_else(|_| "ptxas".into());
    let dir = std::env::temp_dir().join(format!("cute_ext_pipes_{}_{}", kernel.name(), std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("k.ptx");
    std::fs::write(&file, &ptx).unwrap();
    match std::process::Command::new(&ptxas)
        .arg(format!("-arch={flag}"))
        .arg(&file)
        .arg("-o")
        .arg(dir.join("k.cubin"))
        .output()
    {
        Ok(out) => assert!(out.status.success(), "{}: ptxas: {}", kernel.name(), String::from_utf8_lossy(&out.stderr)),
        Err(_) => eprintln!("skipping the assembly: no ptxas found"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    ptx
}

/// A 2SM TMA -> UMMA pipeline and a 2SM UMMA -> T2R pipeline on a (2, 1, 1) cluster, built and
/// exercised once each (not simulated: no 2-CTA MMAs run here).
fn two_sm_kernel() -> Kernel {
    Kernel::builder("cute_ext_pipe_2sm").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
        let pair = Layout::col_major((2usize, 1usize, 1usize, 1usize));
        let mut ab = TMAToUMMAPipeline::create(2, Sm100Mma2SmSs, Some(Sm100TmaLoad2Sm), Some(&pair));
        let mut acc = UMMAtoAsyncPipeline::create(2, Sm100CopyT2r, 256, Sm100Mma2SmSs, Some(&pair));
        if_(warp_idx().eq(0), || {
            ab.producer_acquire_and_get_stage(None);
            ab.producer_commit_and_advance(false, false);
            ab.consumer_wait();
            ab.consumer_release_and_advance();
            acc.producer_acquire();
            acc.producer_commit_and_advance(false, false);
        });
        acc.consumer_wait();
        acc.consumer_release_and_advance();
    })
}

#[test]
fn kernels_compile_for_sm90a_and_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = assemble(&token_kernel(Arch::Sm90a, 32), &tc, Arch::Sm90a, "sm_90a");
    assert!(ptx.contains("mbarrier.try_wait"), "{ptx}");
    let ptx = assemble(&token_kernel(Arch::Sm89, 32), &tc, Arch::Sm89, "sm_89");
    assert!(ptx.contains("mbarrier.test_wait") && !ptx.contains("mbarrier.try_wait"), "{ptx}");
    let ptx = assemble(&tma_async_kernel(), &tc, Arch::Sm90a, "sm_90a");
    assert!(ptx.contains("cp.async.bulk.tensor.2d.shared::cluster.global"), "{ptx}");
    assemble(&circular_kernel(4, 1, false), &tc, Arch::Sm90a, "sm_90a");
    let ptx = assemble(&gemm_kernel("cute_ext_pipe_gemm_c12", Feed::Circular(1, 2)), &tc, Arch::Sm100a, "sm_100a");
    assert!(ptx.contains("tcgen05.commit"), "{ptx}");
    let ptx = assemble(&gemm_kernel("cute_ext_pipe_gemm_threads", Feed::Threads), &tc, Arch::Sm100a, "sm_100a");
    assert!(ptx.contains("fence.proxy.async.shared::cta"), "{ptx}");
    let ptx = assemble(&clc_kernel(), &tc, Arch::Sm100a, "sm_100a");
    assert!(ptx.contains("clusterlaunchcontrol.try_cancel"), "{ptx}");
    let ptx = assemble(&two_sm_kernel(), &tc, Arch::Sm100a, "sm_100a");
    assert!(ptx.contains("cta_group::2") && ptx.contains("multicast::cluster"), "{ptx}");
}

// ---------------------------------------------------------------------------------------------
// (9) Diagnostics
// ---------------------------------------------------------------------------------------------

fn in_kernel(f: impl FnOnce() + 'static) -> String {
    build_panic(move || {
        Kernel::builder("diag").block_dim([128, 1, 1]).target(Arch::Sm100a).build(f);
    })
}

#[test]
fn diagnostics() {
    // Unsupported pairs.
    let e = in_kernel(|| {
        GenericPipeline::create(Sm100Mma1SmSs, Sm100Mma1SmSs, 1, 1, 2, None);
    });
    assert!(e.contains("no pipeline runs from SM100_MMA_1SM_SS to SM100_MMA_1SM_SS"), "{e}");
    let e = in_kernel(|| {
        GenericPipeline::create(LdShared, Sm90TmaLoad, 1, 1, 2, None);
    });
    assert!(e.contains("no pipeline runs from"), "{e}");
    let e = in_kernel(|| {
        GenericPipeline::create([Sm90TmaLoad, StShared], Sm100Mma1SmSs, 1, 1, 2, None);
    });
    assert!(e.contains("arrive in different ways"), "{e}");
    let e = in_kernel(|| {
        GenericPipeline::create([StShared, StShared], LdShared, 1, 1, 2, None);
    });
    assert!(e.contains("appears twice"), "{e}");
    let e = in_kernel(|| {
        GenericPipeline::create(ReduceDownShuffle, LdShared, 1, 1, 2, None);
    });
    assert!(e.contains("warp reduction"), "{e}");
    // The legacy multicast form, a mask without a UMMA side, a missing mask.
    let e = in_kernel(|| {
        experimental::create_pipeline(2, Sm90TmaLoad, Sm100Mma1SmSs, 1, 1, true, None);
    });
    assert!(e.contains("multicast = true"), "{e}");
    let e = in_kernel(|| {
        experimental::create_pipeline_with_mask(2, StShared, LdShared, 32, 32, 3i16, None);
    });
    assert!(e.contains("no UMMA side"), "{e}");
    let e = in_kernel(|| {
        experimental::create_pipeline(2, Sm100TmaLoad2Sm, Sm100Mma2SmSs, 1, 1, ArrivalMask::None, None);
    });
    assert!(e.contains("need an arrival mask"), "{e}");
    let e = in_kernel(|| {
        experimental::create_pipeline(2, StShared, LdShared, 0, 32, ArrivalMask::None, None);
    });
    assert!(e.contains("at least 1"), "{e}");
    // 2SM without cluster_layout_vmnk.
    let e = in_kernel(|| {
        TMAToUMMAPipeline::create(2, Sm100Mma2SmSs, Some(Sm100TmaLoad2Sm), None);
    });
    assert!(e.contains("needs cluster_layout_vmnk"), "{e}");
    let e = in_kernel(|| {
        TMAToUMMAPipeline::create(2, Sm100Mma2SmSs, None, None);
    });
    assert!(e.contains("SM100_TMA_LOAD_2SM"), "{e}");
    let e = in_kernel(|| {
        UMMAtoAsyncPipeline::create(2, Sm100CopyT2r, 128, Sm100Mma2SmSs, None);
    });
    assert!(e.contains("needs cluster_layout_vmnk"), "{e}");
    let e = in_kernel(|| {
        TMAToUMMAPipeline::create(2, LdShared, None, None);
    });
    assert!(e.contains("is not a UMMA operation"), "{e}");
    // Arrival-count queries.
    let e = in_kernel(|| {
        count(2, StShared, LdShared, Side::Producer, opts(48).elect_one_sync(true));
    });
    assert!(e.contains("multiple of 32"), "{e}");
    let e = in_kernel(|| {
        count(2, Sm90TmaLoadMulticast, Sm100Mma1SmSs, Side::Consumer, opts(32));
    });
    assert!(e.contains("pass fan_in"), "{e}");
    let e = in_kernel(|| {
        count(2, StShared, LdShared, Side::Producer, opts(32).op_subset(&[LdShared]));
    });
    assert!(e.contains("op_subset names LD_SHARED"), "{e}");
    // The circular pipeline.
    let e = in_kernel(|| {
        let p = TMAToUMMACircularPipeline::create(2, Sm100Mma1SmSs, 4, 1, None, None);
        p.get_consumer_stage();
    });
    assert!(e.contains("get_consumer_stage is unsupported"), "{e}");
    let e = in_kernel(|| {
        let mut p = TMAToUMMACircularPipeline::create(2, Sm100Mma1SmSs, 4, 1, None, None);
        p.consumer_tail();
    });
    assert!(e.contains("consumer_tail is unsupported"), "{e}");
    let e = in_kernel(|| {
        TMAToUMMACircularPipeline::create(2, Sm100Mma1SmSs, 3, 2, None, None);
    });
    assert!(e.contains("must divide the other"), "{e}");
    let e = in_kernel(|| {
        TMAToUMMACircularPipeline::create(2, Sm100Mma1SmSs, 1, 4, None, None);
    });
    assert!(e.contains("spans more than the ring"), "{e}");
    let e = build_panic(|| {
        experimental::MbarrierOwner::rank(-1);
    });
    assert!(e.contains("never negative"), "{e}");

    // Wrong arrival counts, on the simulator: one warp commits 32 arrivals per stage.
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    let e = token_kernel(Arch::Sm90a, 64).simulate(cfg, &mut token_args()).unwrap_err().to_string();
    assert!(e.contains("deadlock"), "too high a count: {e}");
    let e = token_kernel(Arch::Sm90a, 16).simulate(cfg, &mut token_args()).unwrap_err().to_string();
    assert!(e.contains("more arrivals than") || e.contains("phase aliasing"), "too low a count: {e}");
}
