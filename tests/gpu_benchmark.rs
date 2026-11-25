// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The walkthrough follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/notebooks/benchmark_autotune.ipynb
// (its steps: autotune the element-wise add over copy_bits with update_on_change=["M", "N"],
// tune it with `tune`, and benchmark it on 1024 x 1024 with a workspace generator of 10
// workspaces, 10 warmup and 100 timed iterations, reporting the memory throughput), a notebook
// with no header of its own: the README.md of its folder carries
//   Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// and CUTLASS's LICENSE.txt
//   Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those steps are under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! The benchmark and autotune helpers on a real GPU, as CuTe DSL's `benchmark_autotune`
//! notebook walks through them on the element-wise add of `examples/elementwise_add_autotune.rs`:
//! stable positive times; workspace rotation with the count `get_workspace_count` gives;
//! CUDA graphs taking launch overhead out of a small kernel; `tune` choosing a valid tile and
//! skipping the unbuildable and unimplementable ones; and an `Autotuner` that keeps the
//! winner and tunes again when `M` or `N` changes. Needs the `cuda` feature; skipped (with a
//! note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_benchmark
//! ```

#[allow(dead_code)]
#[path = "../examples/elementwise_add_autotune.rs"]
mod elementwise_add_autotune;

use std::cell::RefCell;

use cute::cuda::{CudaContext, DeviceArg};
use cute::prelude::*;
use cute::testing::{
    BenchmarkOptions, TuneKey, benchmark, benchmark_with_workspaces, get_workspace_count, workspace_count_for_l2,
};
use elementwise_add_autotune::gpu::{Problem, autotuner, build, device_args, key, packed, tune_problem};
use elementwise_add_autotune::{CANDIDATES, TileConfig, kernel};

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

const BEST: TileConfig = TileConfig { threads: (4, 32), copy_bits: 128 };

/// `benchmark` of the 1024 x 1024 add, eager and from a graph: positive, finite and stable
/// across repeats (within a factor of 2), and the result is still right.
#[test]
fn benchmark_times_are_stable_and_positive() {
    let Some((ctx, tc)) = device() else { return };
    let stream = ctx.stream().unwrap();
    let p = Problem::new(&ctx, 1024, 1024).unwrap();
    for graphs in [false, true] {
        let options = BenchmarkOptions::new().use_cuda_graphs(graphs);
        let mut times = Vec::new();
        for _ in 0..3 {
            let launch = build(&ctx, &tc, &p, &BEST).unwrap();
            times.push(benchmark(&stream, &options, launch).unwrap());
        }
        let (lo, hi) = (times.iter().cloned().fold(f64::MAX, f64::min), times.iter().cloned().fold(0.0, f64::max));
        let gbs = (3 * 1024 * 1024 * 4) as f64 / (lo * 1e-6) / 1e9;
        println!("graphs={graphs}: {times:?} us, {gbs:.1} GB/s");
        assert!(lo > 0.0 && hi.is_finite() && hi < 2.0 * lo, "graphs={graphs}: {times:?}");
    }
    stream.synchronize().unwrap();
    p.check().unwrap();
    // Graphs need a stream that can be captured.
    let launch = build(&ctx, &tc, &p, &BEST).unwrap();
    let e = benchmark(&ctx.default_stream(), &BenchmarkOptions::new().use_cuda_graphs(true), launch).unwrap_err();
    assert!(e.to_string().contains("legacy default stream"), "{e}");
}

/// The notebook's benchmark with a workspace generator: the count from the device's L2 size
/// for 1024 x 1024 problems, one generated workspace each, used in turn through the warmup
/// and the timed iterations, eagerly and in graph mode (where each launch is captured once);
/// every workspace's C is right afterwards.
#[test]
fn workspaces_rotate_through_the_expected_count() {
    let Some((ctx, tc)) = device() else { return };
    let stream = ctx.stream().unwrap();
    let (m, n) = (1024, 1024);
    let one = 3 * m * n * 4;
    let l2 = ctx.device_info().unwrap().l2_cache_bytes;
    let count = get_workspace_count(&ctx, one, 10, 100).unwrap();
    assert_eq!(count, workspace_count_for_l2(l2, one as u64, 10, 100));
    assert_eq!(count, ((l2 * 3) as usize / one + 1).min(110));
    println!("L2 {l2} bytes: {count} workspaces of {one} bytes");

    let k = kernel(BEST);
    let arch = ctx.best_target(&tc).unwrap();
    let function = ctx.load(&k.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap()).unwrap();
    let cfg = BEST.launch(m, n);
    let scalars = packed(&k, m, n).unwrap();
    for graphs in [false, true] {
        let generated = RefCell::new(0);
        let used = RefCell::new(Vec::new());
        let problems: Vec<Problem<'_>> = Vec::new();
        let problems = RefCell::new(problems);
        let options = BenchmarkOptions::new().workspace_count(count).use_cuda_graphs(graphs);
        let t = benchmark_with_workspaces(
            &stream,
            &options,
            |i| {
                *generated.borrow_mut() += 1;
                problems.borrow_mut().push(Problem::new(&ctx, m, n)?);
                Ok(i)
            },
            |s, &i| {
                used.borrow_mut().push(i);
                let problems = problems.borrow();
                let args: Vec<DeviceArg<'_>> = device_args(&k, &scalars, &problems[i]);
                Ok(ctx.launch_builder(&k, &function, cfg).stream(s).args(&args).submit()?)
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(*generated.borrow(), count);
        let want: Vec<usize> = (0..110).map(|i| i % count).collect();
        assert_eq!(*used.borrow(), want, "graphs={graphs}");
        for p in problems.borrow().iter() {
            p.check().unwrap();
        }
        let gbs = one as f64 / (t * 1e-6) / 1e9;
        println!("graphs={graphs}: {t:.2} us per iteration over {count} workspaces, {gbs:.1} GB/s");
        assert!(t > 0.0);
    }
}

/// A small kernel (one block of 32 threads): replaying a graph of 1000 launches costs less per
/// launch than queuing them one by one.
#[test]
fn graphs_have_less_overhead_for_small_kernels() {
    let Some((ctx, tc)) = device() else { return };
    let stream = ctx.stream().unwrap();
    let small = TileConfig { threads: (1, 32), copy_bits: 32 };
    let p = Problem::new(&ctx, 4, 32).unwrap();
    let mut results = Vec::new();
    for attempt in 0..3 {
        let options = |g| BenchmarkOptions::new().warmup_iterations(100).iterations(1000).use_cuda_graphs(g);
        let eager = benchmark(&stream, &options(false), build(&ctx, &tc, &p, &small).unwrap()).unwrap();
        let graph = benchmark(&stream, &options(true), build(&ctx, &tc, &p, &small).unwrap()).unwrap();
        println!("attempt {attempt}: eager {eager:.3} us, graph {graph:.3} us per launch");
        results.push((eager, graph));
        if graph < eager {
            break;
        }
    }
    stream.synchronize().unwrap();
    p.check().unwrap();
    assert!(results.iter().any(|(e, g)| g < e), "{results:?}");
}

/// The notebook's `tune`: over the candidates, on rows of 1002 elements (a multiple of 2 but
/// not of 4), it skips `copy_bits = 1` (the build panics) and the 128-bit tiles
/// (`CantImplement`), and picks a valid tile, whose launch is right.
#[test]
fn tune_picks_a_valid_configuration_and_skips_the_rest() {
    let Some((ctx, tc)) = device() else { return };
    let stream = ctx.stream().unwrap();
    let p = Problem::new(&ctx, 512, 1002).unwrap();
    let options = BenchmarkOptions::new().warmup_iterations(5).iterations(50);
    let tuned = tune_problem(&ctx, &tc, &stream, &p, &options).unwrap();
    for t in &tuned.trials {
        println!("{:?}: {:?}", t.config, t.outcome);
    }
    assert!(matches!(tuned.config.copy_bits, 32 | 64), "{:?}", tuned.config);
    assert!(tuned.time_us > 0.0);
    let skipped: Vec<(TileConfig, String)> = tuned.skipped().map(|(c, e)| (*c, e.to_string())).collect();
    assert_eq!(skipped.len(), 4, "{skipped:?}");
    assert!(skipped[0].0.copy_bits == 1 && skipped[0].1.contains("panicked"), "{skipped:?}");
    assert!(skipped[1..].iter().all(|(c, e)| c.copy_bits == 128 && e.contains("not a multiple")), "{skipped:?}");
    assert_eq!(tuned.trials.len(), CANDIDATES.len());
    let mut launch = build(&ctx, &tc, &p, &tuned.config).unwrap();
    launch(&stream, 0).unwrap();
    stream.synchronize().unwrap();
    p.check().unwrap();
}

/// `autotune_jit` with `update_on_change=["M", "N"]`: the first call for a size tunes, a
/// repeat uses the winner, a new `N` (or `M`) tunes again, and a change to a part it does not
/// watch does not.
#[test]
fn the_autotuner_caches_the_winner_and_retunes_when_the_key_changes() {
    let Some((ctx, tc)) = device() else { return };
    let stream = ctx.stream().unwrap();
    let mut tuner = autotuner(BenchmarkOptions::new().warmup_iterations(5).iterations(30));
    let builds = RefCell::new(0);
    let mut run = |m: usize, n: usize, extra: usize| {
        let p = Problem::new(&ctx, m, n).unwrap();
        let k: TuneKey = key(&p).part("call", extra);
        let best = tuner
            .get_or_tune(&stream, &k, |cfg| {
                *builds.borrow_mut() += 1;
                build(&ctx, &tc, &p, cfg)
            })
            .unwrap()
            .config;
        let mut launch = build(&ctx, &tc, &p, &best).unwrap();
        launch(&stream, 0).unwrap();
        stream.synchronize().unwrap();
        p.check().unwrap();
        (best, tuner.tunings())
    };
    let (first, t) = run(1024, 512, 0);
    assert_eq!(t, 1);
    assert_eq!(*builds.borrow(), CANDIDATES.len());
    assert_eq!(run(1024, 512, 1), (first, 1), "cached; the unwatched part is ignored");
    assert_eq!(*builds.borrow(), CANDIDATES.len());
    assert_eq!(run(1024, 1024, 2).1, 2, "N changed");
    assert_eq!(run(512, 1024, 3).1, 3, "M changed");
    assert_eq!(run(1024, 512, 4), (first, 3), "the first size again");
    assert_eq!(*builds.borrow(), 3 * CANDIDATES.len());
}
