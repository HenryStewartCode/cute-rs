//! Software-pipelined loops, `for_range_pipelined` (CuTe DSL's `cutlass.range(.., prefetch_stages=S)`):
//! the rotated order of produce and consume calls for every small distance,
//! trip count and step (static and runtime bounds); `tensorop_gemm`'s
//! mainloop rotated by the helper against the hand-rotated one; a single warp
//! producing into and consuming from a `PipelineCpAsync`; and a TMA + WGMMA
//! loop over a `PipelineTmaAsync`, simulated and assembled for sm_90a. With
//! the `cuda` feature the SM89 kernels also run on the GPU.

#[allow(dead_code)]
#[path = "../examples/tensorop_gemm.rs"]
mod tensorop_gemm;

use cute::ir::{Op, Region};
use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineBarriers, PipelineCpAsync, PipelineCpAsyncOptions, PipelineTmaAsync,
    PipelineTmaAsyncOptions, PipelineUserType, make_pipeline_state,
};
use cute::prelude::*;
use cute::warpgroup::{self, OperandSource, SmemLayoutAtomKind, make_smem_layout_atom};
use tensorop_gemm::GemmConfig;

// ---------------------------------------------------------------------------
// The rotated order.
// ---------------------------------------------------------------------------

/// Slots of the ring the order kernel's producer writes into (more than the
/// largest distance tested).
const RING: usize = 5;
/// Log records the order kernel has room for: two per iteration.
const RECORDS: usize = 2 * 12;

/// One thread runs `for_range_pipelined(lo, hi, step, S)` (static bounds, or
/// the runtime parameters `lo` and `hi`). `produce(k)` writes `10 k + 3` into
/// the ring slot of its call number and logs `(k, 0, value)`; `consume(k)`
/// reads the slot of its call number and logs `(k, 1, value read)`. The
/// producer carries its call count, the consumer its count and the sum it
/// read, which end up after the log.
fn order_kernel(s: i64, step: i32, bounds: Option<(i32, i32)>) -> Kernel {
    let name = match bounds {
        Some((lo, hi)) => format!("order_s{s}_st{step}_{lo}_{hi}").replace('-', "m"),
        None => format!("order_s{s}_st{step}_runtime"),
    };
    Kernel::builder(name).block_dim([1, 1, 1]).build(move || {
        let (lo, hi) = match bounds {
            Some((lo, hi)) => (Int32::from(lo), Int32::from(hi)),
            None => (param_scalar::<i32>("lo"), param_scalar::<i32>("hi")),
        };
        let out = param_tensor::<i32>("out", Layout::col_major(RECORDS * 3 + 3));
        let ring = make_smem_tensor::<i32>(Layout::col_major(RING));
        let pos = Var::new(Int32::from(0));
        let log = |k: Int32, role: i32, value: Int32| {
            let p = pos.get();
            out.store(p * 3, k);
            out.store(p * 3 + 1, role);
            out.store(p * 3 + 2, value);
            pos.set(p + 1);
        };
        let zero = Int32::from(0);
        let (produced, (consumed, sum)) = for_range_pipelined(
            lo,
            hi,
            step,
            Prefetch::stages(s),
            (zero, (zero, zero)),
            |k, n| {
                let value = k * 10 + 3;
                ring.store(n % RING as i32, value);
                log(k, 0, value);
                n + 1
            },
            |k, (n, sum)| {
                let value = ring.load(n % RING as i32);
                log(k, 1, value);
                (n + 1, sum + value)
            },
        );
        out.store((RECORDS * 3) as i32, produced);
        out.store((RECORDS * 3 + 1) as i32, consumed);
        out.store((RECORDS * 3 + 2) as i32, sum);
    })
}

/// The plain-Rust output of [`order_kernel`]: the prologue's produces, the
/// steady state's produce `k + S` then consume `k`, the drain's consumes.
fn order_reference(s: usize, lo: i32, step: i32, trips: usize) -> Vec<i32> {
    let k = |j: usize| lo + j as i32 * step;
    let mut order = Vec::new();
    let ahead = s.min(trips);
    order.extend((0..ahead).map(|j| (j, 0)));
    for j in 0..trips - ahead {
        order.push((j + s, 0));
        order.push((j, 1));
    }
    order.extend((trips - ahead..trips).map(|j| (j, 1)));
    let mut out = vec![-1; RECORDS * 3 + 3];
    for (i, &(j, role)) in order.iter().enumerate() {
        out[i * 3..i * 3 + 3].copy_from_slice(&[k(j), role, k(j) * 10 + 3]);
    }
    let sum: i32 = (0..trips).map(|j| k(j) * 10 + 3).sum();
    out[RECORDS * 3..].copy_from_slice(&[trips as i32, trips as i32, sum]);
    out
}

/// Every distance 0 to 4, trip count 0 to 12 and step 1 and 2, with static
/// bounds (the prologue and drain unrolled) and runtime ones: the log is the
/// rotated order, and every consume reads what the produce of its `k` wrote.
#[test]
fn rotated_order_matches_the_reference() {
    let cfg = LaunchConfig::new([1, 1, 1], [1, 1, 1]);
    for s in 0..=4usize {
        for step in [1, 2] {
            let runtime = order_kernel(s as i64, step, None);
            for trips in 0..=12usize {
                // An upper bound that is not on the grid for step 2.
                let hi_of = |lo: i32| lo + trips as i32 * step - (step - 1);
                let (lo, hi) = (2, hi_of(2));
                let mut args = vec![KernelArg::buffer(&[-1i32; RECORDS * 3 + 3])];
                order_kernel(s as i64, step, Some((lo, hi)))
                    .simulate(cfg, &mut args)
                    .unwrap_or_else(|e| panic!("static s{s} step{step} trips{trips}: {e}"));
                assert_eq!(
                    args[0].to_vec::<i32>(),
                    order_reference(s, lo, step, trips),
                    "static s{s} st{step} t{trips}"
                );

                let (lo, hi) = (-3, hi_of(-3));
                let mut args =
                    vec![KernelArg::scalar(lo), KernelArg::scalar(hi), KernelArg::buffer(&[-1i32; RECORDS * 3 + 3])];
                runtime
                    .simulate(cfg, &mut args)
                    .unwrap_or_else(|e| panic!("runtime s{s} step{step} trips{trips}: {e}"));
                assert_eq!(
                    args[2].to_vec::<i32>(),
                    order_reference(s, lo, step, trips),
                    "runtime s{s} st{step} t{trips}"
                );
            }
        }
    }
}

/// Static bounds emit one loop at most (none when the trip count is at most
/// the distance), runtime ones a prologue, a steady state and a drain loop;
/// distance 0 is the plain loop.
#[test]
fn loops_emitted() {
    let loops = |k: &Kernel| k.ir().count_ops(|op| matches!(op, Op::For { .. }));
    assert_eq!(loops(&order_kernel(2, 1, Some((0, 2)))), 0);
    assert_eq!(loops(&order_kernel(2, 1, Some((0, 9)))), 1);
    assert_eq!(loops(&order_kernel(0, 1, Some((0, 9)))), 1);
    assert_eq!(loops(&order_kernel(2, 2, None)), 3);
    assert_eq!(loops(&order_kernel(0, 2, None)), 1);
}

// ---------------------------------------------------------------------------
// tensorop_gemm's mainloop.
// ---------------------------------------------------------------------------

/// The first `For` in `r` (depth first) with a tensor-core MMA in its body.
fn mainloop(r: &Region) -> Option<&Region> {
    r.insts.iter().find_map(|i| match &i.op {
        Op::For { body, .. } if count(body, &|op| matches!(op, Op::Mma { .. })) > 0 => Some(body),
        op => op.regions().into_iter().find_map(mainloop),
    })
}

fn count(r: &Region, pred: &dyn Fn(&Op) -> bool) -> usize {
    r.insts
        .iter()
        .map(|i| pred(&i.op) as usize + i.op.regions().into_iter().map(|r| count(r, pred)).sum::<usize>())
        .sum()
}

/// The copies, MMAs, waits and barriers of one K tile of `kernel`'s mainloop.
fn per_tile_counts(kernel: &Kernel) -> [usize; 6] {
    let body = mainloop(&kernel.ir().body).expect("a mainloop");
    [
        count(body, &|op| matches!(op, Op::CpAsync { .. })),
        count(body, &|op| matches!(op, Op::CpAsyncCommitGroup)),
        count(body, &|op| matches!(op, Op::CpAsyncWaitGroup { .. })),
        count(body, &|op| matches!(op, Op::Barrier)),
        count(body, &|op| matches!(op, Op::LdMatrix { .. })),
        count(body, &|op| matches!(op, Op::Mma { .. })),
    ]
}

/// The configurations both mainloops run: K tiles fewer than, equal to and
/// more than the distance, 2 and 3 stages, f16, bf16 and FP8.
fn gemm_configs() -> Vec<GemmConfig> {
    vec![
        GemmConfig::new(128, 128, 32),
        GemmConfig::new(128, 128, 64),
        GemmConfig::new(256, 128, 96),
        GemmConfig::new(128, 128, 256),
        GemmConfig { stages: 2, swizzle: false, ..GemmConfig::new(128, 128, 128) },
        GemmConfig { ab: DType::BF16, ..GemmConfig::new(128, 128, 160) },
        GemmConfig::fp8(128, 128, 256, DType::F8E4M3FN),
    ]
}

/// `tensorop_gemm` rotated by `for_range_pipelined`: the exact CPU result (so
/// bit-identical to the hand-rotated loop, which `run` holds to the same
/// sums) under every schedule, and the same copies, waits, barriers,
/// `ldmatrix`es and MMAs per K tile as the hand-rotated mainloop.
#[test]
fn tensorop_gemm_with_the_helper() {
    for cfg in gemm_configs() {
        let pipelined = GemmConfig { pipelined: true, ..cfg };
        tensorop_gemm::run(pipelined);
        if cfg.k / cfg.bk > cfg.stages - 1 {
            let (hand, helper) = (tensorop_gemm::build(cfg), tensorop_gemm::build(pipelined));
            assert_eq!(per_tile_counts(&hand), per_tile_counts(&helper), "{cfg:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// One warp producing into and consuming from a PipelineCpAsync.
// ---------------------------------------------------------------------------

/// Tiles of [`single_warp_cp_async`]: more than three passes of a 4-stage ring.
pub const SW_ITEMS: usize = 13;
/// Words per tile: 16 bytes per thread.
const SW_TILE: usize = 128;

/// One warp loads [`SW_ITEMS`] tiles with `cp.async` into a `stages`-stage
/// `PipelineCpAsync` (acquire, copy, commit) and consumes them (wait, write
/// `3 x + 1` of every word, release), `ahead` tiles behind, through
/// `for_range_pipelined` (bounded by the stage count when `bounded`).
pub fn single_warp_cp_async(stages: usize, ahead: i64, bounded: bool) -> Kernel {
    Kernel::builder(format!("single_warp_cp_async_s{stages}_a{ahead}")).block_dim([32, 1, 1]).target(Arch::Sm89).build(
        move || {
            let src = param_tensor::<u32>("src", Layout::col_major(SW_ITEMS * SW_TILE));
            let out = param_tensor::<u32>("out", Layout::col_major(SW_ITEMS * SW_TILE));
            let warp = CooperativeGroup::new(Agent::Warp, 1);
            let pipeline = PipelineCpAsync::create(stages, warp, warp, PipelineCpAsyncOptions::default());
            let data = make_smem_tensor::<u32>(Layout::col_major(stages * SW_TILE));
            let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
            let chunk = Layout::col_major(4);
            let t = thread_idx().x;
            let tile = SW_TILE as i32;
            let prefetch = Prefetch::stages(ahead);
            let prefetch = if bounded { prefetch.bounded_by(pipeline.num_stages()) } else { prefetch };
            let (mut producer, _) = for_range_pipelined(
                0,
                SW_ITEMS as i32,
                1,
                prefetch,
                pipeline.make_participants(),
                |k, mut p| {
                    let handle = p.acquire_and_advance(None);
                    // Offsets as multiples of 4 (a constant 0 tile times the
                    // tile size would lose its divisibility).
                    let from = src.ptr().offset((k * (tile / 4) + t) * 4);
                    let to = data.ptr().offset((handle.index() * (tile / 4) + t) * 4);
                    copy(&atom, &make_tensor(&from, chunk.clone()), &make_tensor(&to, chunk.clone()));
                    handle.commit();
                    p
                },
                |k, mut c| {
                    let handle = c.wait_and_advance(None);
                    for j in 0..SW_TILE / 32 {
                        let w = t + (j * 32) as i32;
                        out.store(k * tile + w, data.load(handle.index() * tile + w) * 3u32 + 1u32);
                    }
                    handle.release();
                    c
                },
            );
            producer.tail();
        },
    )
}

fn single_warp_inputs() -> Vec<KernelArg> {
    let src: Vec<u32> = (0..(SW_ITEMS * SW_TILE) as u32).map(|i| i.wrapping_mul(2654435761) >> 7).collect();
    vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![0u32; SW_ITEMS * SW_TILE])]
}

fn single_warp_check(args: &[KernelArg]) {
    let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
    for (i, (&x, &y)) in src.iter().zip(&out).enumerate() {
        assert_eq!(y, x.wrapping_mul(3).wrapping_add(1), "word {i}");
    }
}

/// The single-warp ring at distance `stages - 1` (and smaller), for 2, 3 and
/// 4 stages, under the built-in and 8 seeded schedules with random `cp.async`
/// completion.
#[test]
fn single_warp_cp_async_pipeline() {
    let cfg = LaunchConfig::new([1, 1, 1], [32, 1, 1]);
    let seeds: Vec<u64> = (0..8).collect();
    for stages in 2..=4usize {
        for ahead in [stages as i64 - 1, 1] {
            let kernel = single_warp_cp_async(stages, ahead, true);
            let mut args = single_warp_inputs();
            kernel.simulate_schedules(cfg, &mut args, &SimOptions::default(), &seeds).unwrap_or_else(|e| panic!("{e}"));
            single_warp_check(&args);
        }
    }
}

// ---------------------------------------------------------------------------
// A TMA + WGMMA loop over a PipelineTmaAsync.
// ---------------------------------------------------------------------------

/// K tiles of [`tma_wgmma_loop`]: two passes of its 4 stages.
const TW_TILES: usize = 8;
/// Stages of [`tma_wgmma_loop`].
const TW_STAGES: usize = 4;
/// K of a tile, and N.
const TW_K: usize = 64;
const TW_N: usize = 128;

/// One warpgroup computes a `64 x 128` tile of `C = A B^T` over
/// [`TW_TILES`] K tiles of 64: warp 0 acquires a stage of a 4-stage
/// `PipelineTmaAsync` and loads A and B into it with TMA; every warp waits for
/// the stage, issues the four k-blocks of warpgroup MMAs, waits for them and
/// releases it. The loop is `for_range_pipelined` at distance `stages - 2`, as
/// CuTe DSL's Blackwell software-pipelined GEMM chooses it.
fn tma_wgmma_loop() -> Kernel {
    const K: usize = TW_TILES * TW_K;
    Kernel::builder("tma_wgmma_pipelined_loop").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, K), (K, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((TW_N, K), (K, 1)), 16);
        let c = param_tensor::<f32>("c", Layout::new((64, TW_N), (TW_N, 1)));
        let op = warpgroup::MmaF16BF16Op::new(
            DType::F16,
            DType::F32,
            (64, TW_N),
            OperandSource::SMEM,
            OperandMajorMode::K,
            OperandMajorMode::K,
        );
        let mut mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
        let atom = make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128);
        let la = tile_to_shape_swizzled(&atom, (64, TW_K, TW_STAGES));
        let lb = tile_to_shape_swizzled(&atom, (TW_N, TW_K, TW_STAGES));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom_a, ta) = cpasync::make_tiled_tma_atom(g2s, &a, la.clone(), (64, TW_K), 1).into();
        let (atom_b, tb) = cpasync::make_tiled_tma_atom(g2s, &b, lb.clone(), (TW_N, TW_K), 1).into();
        let sa = make_smem_tensor_aligned::<F16>(la, 1024);
        let sb = make_smem_tensor_aligned::<F16>(lb, 1024);
        let pipeline = PipelineTmaAsync::create(
            TW_STAGES,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, 4),
            ((64 + TW_N) * TW_K * 2) as i32,
            PipelineTmaAsyncOptions::default(),
        );
        let one = make_layout(1, 0);
        let ga = local_tile(&ta, (64, TW_K), (0, ..));
        let gb = local_tile(&tb, (TW_N, TW_K), (0, ..));
        let (tsa, tga) = cpasync::tma_partition(&atom_a, 0, &one, &sa.group_modes(0, 2), &ga.group_modes(0, 2));
        let (tsb, tgb) = cpasync::tma_partition(&atom_b, 0, &one, &sb.group_modes(0, 2), &gb.group_modes(0, 2));
        let t = thread_idx().x;
        let thr = mma.get_slice(t);
        let (tcsa, tcsb) = (thr.partition_a(&sa), thr.partition_b(&sb));
        let (tcra, tcrb) = (mma.make_fragment_a(&tcsa), mma.make_fragment_b(&tcsb));
        let tcgc = thr.partition_c(&c);
        let acc = thr.partition_fragment_c::<f32, _>(&c);
        mma.set(MmaField::ACCUMULATE, false);
        let producer_state = make_pipeline_state(PipelineUserType::Producer, TW_STAGES);
        let consumer_state = make_pipeline_state(PipelineUserType::Consumer, TW_STAGES);
        let (_, (acc, _, _)) = for_range_pipelined(
            0,
            TW_TILES as i32,
            1,
            Prefetch::stages(TW_STAGES as i64 - 2).bounded_by(pipeline.num_stages()),
            (producer_state, (acc, mma, consumer_state)),
            |k, mut state| {
                if_(warp_idx().eq(0), || {
                    pipeline.producer_acquire(&state, None);
                    let bar = pipeline.producer_get_barrier(&state);
                    let st = state.index();
                    copy_with(&atom_a, &tga.slice((.., k)), &tsa.slice((.., st)), CopyArgs::new().tma_bar_ptr(&bar));
                    copy_with(&atom_b, &tgb.slice((.., k)), &tsb.slice((.., st)), CopyArgs::new().tma_bar_ptr(&bar));
                    pipeline.producer_commit(&state);
                });
                state.advance();
                state
            },
            |_k, (acc, mut mma, mut state)| {
                pipeline.consumer_wait(&state, None);
                warpgroup::fence();
                let st = state.index();
                for kb in 0..TW_K / 16 {
                    gemm(&mma, &acc, &tcra.slice((.., .., kb, st)), &tcrb.slice((.., .., kb, st)), &acc);
                    mma.set(MmaField::ACCUMULATE, true);
                }
                warpgroup::commit_group();
                warpgroup::wait_group(0);
                pipeline.consumer_release(&state);
                state.advance();
                (acc, mma, state)
            },
        );
        for i in 0..acc.size() {
            tcgc.store(i, acc.load(i));
        }
    })
}

/// [`tma_wgmma_loop`] on small-integer inputs under the built-in and 4 seeded
/// schedules with random async completion: the exact plain-Rust product.
#[test]
fn tma_wgmma_pipelined_loop_matches_rust() {
    const K: usize = TW_TILES * TW_K;
    let kernel = tma_wgmma_loop();
    let small = |n: usize, seed: usize| -> Vec<F16> {
        (0..n).map(|i| F16::from_f32(((i * 7 + seed * 13) % 5) as f32 - 2.0)).collect()
    };
    let (a, b) = (small(64 * K, 1), small(TW_N * K, 2));
    let fresh = || vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0f32; 64 * TW_N])];
    let cfg = LaunchConfig::new([1, 1, 1], [128, 1, 1]);
    let check = |args: &[KernelArg]| {
        let c = args[2].to_vec::<f32>();
        for r in 0..64 {
            for col in 0..TW_N {
                let want: f32 = (0..K).map(|k| a[r * K + k].to_f32() * b[col * K + k].to_f32()).sum();
                assert_eq!(c[r * TW_N + col], want, "C({r}, {col})");
            }
        }
    };
    let mut args = fresh();
    let report = kernel.simulate_all_orders(cfg, &mut args, &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    assert!(report.tma_instructions > 0, "{report}");
    check(&args);
    let mut args = fresh();
    kernel.simulate_schedules(cfg, &mut args, &SimOptions::default(), &[0, 1, 2, 3]).unwrap_or_else(|e| panic!("{e}"));
    check(&args);
}

/// [`tma_wgmma_loop`]'s PTX for sm_90a, and with `ptxas` on `PATH` its
/// assembly (without serialised MMAs).
#[test]
fn tma_wgmma_pipelined_loop_assembles_for_sm_90a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = tma_wgmma_loop().compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
    assert!(ptx.contains("wgmma.mma_async.sync.aligned") && ptx.contains("cp.async.bulk.tensor"));
    let Some(ptxas) =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|p| p.is_file()))
    else {
        eprintln!("skipping ptxas: not on PATH");
        return;
    };
    let path = std::env::temp_dir().join(format!("cute-range-prefetch-{}.ptx", std::process::id()));
    std::fs::write(&path, &ptx).unwrap();
    let out = std::process::Command::new(&ptxas)
        .args(["-arch=sm_90a", "-v", "-o", "/dev/null"])
        .arg(&path)
        .output()
        .expect("running ptxas");
    let _ = std::fs::remove_file(&path);
    let log = String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "ptxas failed:\n{log}");
    assert!(!log.contains("serialized"), "ptxas serialised the warpgroup MMAs:\n{log}");
}

// ---------------------------------------------------------------------------
// On the SM89 GPU.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use cute::cuda::CudaContext;

    fn device() -> Option<(CudaContext, Toolchain)> {
        let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
        let tc = Toolchain::discover();
        if tc.is_none() {
            eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        }
        Some((ctx, tc?))
    }

    /// Run on the simulator and the GPU from the same arguments.
    fn sim_and_gpu(
        (ctx, tc): &(CudaContext, Toolchain),
        kernel: &Kernel,
        cfg: LaunchConfig,
        args: &[KernelArg],
    ) -> (Vec<KernelArg>, Vec<KernelArg>) {
        let mut sim = args.to_vec();
        kernel.simulate(cfg, &mut sim).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
        let mut gpu = args.to_vec();
        kernel.run_cuda(ctx, tc, cfg, &mut gpu).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
        (sim, gpu)
    }

    /// The single-warp `PipelineCpAsync` ring at distance `stages - 1`, 10
    /// launches each: the plain-Rust result, bit-identical to the simulator.
    #[test]
    fn single_warp_cp_async_pipeline_on_the_gpu() {
        let Some(device) = device() else { return };
        let cfg = LaunchConfig::new([1, 1, 1], [32, 1, 1]);
        for stages in 2..=4usize {
            let kernel = single_warp_cp_async(stages, stages as i64 - 1, true);
            for _ in 0..10 {
                let (sim, gpu) = sim_and_gpu(&device, &kernel, cfg, &single_warp_inputs());
                single_warp_check(&gpu);
                assert!(sim == gpu, "{}: the GPU and the simulator differ", kernel.name());
            }
        }
    }

    /// `tensorop_gemm` with the helper and by hand on the GPU (f16 and bf16):
    /// bit-identical C from both and from the simulator.
    #[test]
    fn tensorop_gemm_with_the_helper_on_the_gpu() {
        let Some(device) = device() else { return };
        // Not FP8: its mma.sync needs an LLVM 23 mlir-opt.
        for cfg in gemm_configs().into_iter().filter(|c| c.ab != DType::F8E4M3FN) {
            let launch = LaunchConfig::new([(cfg.m / cfg.bm) as u32, (cfg.n / cfg.bn) as u32, 1], [128, 1, 1]);
            let n = |len: usize, seed: u32| -> Vec<f32> {
                (0..len as u32)
                    .map(|i| ((i.wrapping_mul(2654435761).wrapping_add(seed) >> 13) % 7) as f32 - 3.0)
                    .collect()
            };
            let (a, b) = (n(cfg.m * cfg.k, 1), n(cfg.n * cfg.k, 2));
            let args = vec![
                tensorop_gemm::input(cfg.ab, &a),
                tensorop_gemm::input(cfg.ab, &b),
                tensorop_gemm::input(cfg.c, &vec![0.0; cfg.m * cfg.n]),
            ];
            let hand = tensorop_gemm::build(cfg);
            let helper = tensorop_gemm::build(GemmConfig { pipelined: true, ..cfg });
            let (_, hand_gpu) = sim_and_gpu(&device, &hand, launch, &args);
            let (sim, gpu) = sim_and_gpu(&device, &helper, launch, &args);
            assert!(sim == gpu, "{cfg:?}: the GPU and the simulator differ");
            assert!(gpu == hand_gpu, "{cfg:?}: the helper and the hand-rotated loop differ");
        }
    }
}
