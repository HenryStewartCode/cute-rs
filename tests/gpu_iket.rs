//! IKET traces on a real GPU: the warp-specialised hand-off and the instrumented
//! `tensorop_gemm` (written as a Chrome trace that loads in Perfetto). Needs the `cuda`
//! feature; skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_iket
//! ```

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/tensorop_gemm.rs"]
mod tensorop_gemm;

mod iket_kernels;

use cute::cuda::CudaContext;
use cute::prelude::*;
use iket_kernels::*;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn sim_trace(kernel: &Kernel, launch: LaunchConfig, args: &mut [KernelArg]) -> IketTrace {
    let mut module = kernel.sim_module();
    module.simulate(launch, args).unwrap_or_else(|e| panic!("{e}"));
    let words = module.read_global(IKET_GLOBAL).unwrap().to_vec::<u32>();
    kernel.decode_iket(&words).unwrap()
}

fn gpu_trace(
    ctx: &CudaContext,
    tc: &Toolchain,
    kernel: &Kernel,
    launch: LaunchConfig,
    args: &mut [KernelArg],
) -> IketTrace {
    // A kernel built for a declared target compiles only for it; its PTX still runs on newer
    // GPUs through the driver's JIT.
    let arch = kernel.target().unwrap_or_else(|| ctx.best_target(tc).expect("a target"));
    let compiled = kernel.compile_with(tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{e}"));
    let f = ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"));
    ctx.launch(kernel, &f, launch, args).unwrap_or_else(|e| panic!("{e}"));
    let words = f.global(IKET_GLOBAL).unwrap().read::<u32>().unwrap();
    kernel.decode_iket(&words).unwrap()
}

/// Each warp's timestamps never decrease (the GPU's clock is monotonic within a warp).
fn assert_monotonic(trace: &IketTrace) {
    let mut warps: Vec<(u32, u32)> = trace.records.iter().map(|r| (r.cta, r.warp)).collect();
    warps.sort_unstable();
    warps.dedup();
    for (cta, warp) in warps {
        let ts: Vec<u64> = trace.warp_records(cta, warp).iter().map(|r| r.ts).collect();
        assert!(ts.windows(2).all(|w| w[0] <= w[1]), "CTA {cta} warp {warp}: {ts:?}");
    }
}

/// The GPU's trace of the hand-off has the simulator's events, per warp and in order, and pairs
/// the same ranges.
#[test]
fn handoff_trace_matches_the_simulator() {
    let Some((ctx, tc)) = device() else { return };
    let kernel = traced_handoff(Some(IketOptions { capacity: 256, clock: IketClock::Globaltimer }));
    let sim = sim_trace(&kernel, handoff_cfg(), &mut handoff_args());
    let mut args = handoff_args();
    let gpu = gpu_trace(&ctx, &tc, &kernel, handoff_cfg(), &mut args);
    assert_eq!(args[1].to_vec::<u32>(), handoff_sums());
    assert!(gpu.unmatched.is_empty() && !gpu.overflowed, "{:?}", gpu.unmatched);
    for warp in 0..2 {
        assert_eq!(warp_events(&gpu, warp), warp_events(&sim, warp), "warp {warp}");
    }
    assert_eq!(gpu.spans.len(), sim.spans.len());
    assert_monotonic(&gpu);
    // The hardware clock moved: the spans have real durations.
    assert!(gpu.spans.iter().any(|s| s.end > s.start));
}

/// `tensorop_gemm` instrumented as CuTe DSL's `fp16_gemm_4_iket.py` instruments its GEMM:
/// every warp pushes and pops the prologue, each K tile's wait and issue, and the epilogue.
/// The trace is written for Perfetto.
#[test]
fn instrumented_gemm_trace_for_perfetto() {
    let Some((ctx, tc)) = device() else { return };
    let (m, n, k) = (256, 256, 128);
    let cfg = tensorop_gemm::GemmConfig {
        iket: Some(IketOptions { capacity: 8192, clock: IketClock::Globaltimer }),
        ..tensorop_gemm::GemmConfig::new(m, n, k)
    };
    let kernel = tensorop_gemm::build(cfg);
    let launch = LaunchConfig::new([(m / cfg.bm) as u32, (n / cfg.bn) as u32, 1], [128, 1, 1]);
    let (a, b) = (tensorop_gemm::small_integers(m * k, 1), tensorop_gemm::small_integers(n * k, 2));
    let mut args = vec![
        tensorop_gemm::input(cfg.ab, &a),
        tensorop_gemm::input(cfg.ab, &b),
        tensorop_gemm::input(cfg.c, &vec![0.0; m * n]),
    ];
    let trace = gpu_trace(&ctx, &tc, &kernel, launch, &mut args);
    // C is right.
    let c = args[2].to_f32();
    for (i, j) in [(0, 0), (17, 200), (255, 255), (128, 64)] {
        let want: f32 = (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum();
        assert_eq!(c[i * n + j], want, "C[{i},{j}]");
    }
    let (ctas, warps, k_tiles) = (4usize, 4usize, k / cfg.bk);
    assert!(trace.unmatched.is_empty() && !trace.overflowed, "{:?}", trace.unmatched);
    let count = |name: &str| trace.spans.iter().filter(|s| s.name == name).count();
    assert_eq!(count("prologue"), ctas * warps);
    assert_eq!(count("wait"), ctas * warps * k_tiles);
    assert_eq!(count("issue"), ctas * warps * k_tiles);
    assert_eq!(count("epilogue"), ctas * warps);
    assert_monotonic(&trace);
    // Each wait carries its K tile as an i32 payload.
    let tiles: std::collections::BTreeSet<i64> = trace
        .spans
        .iter()
        .filter(|s| s.name == "wait")
        .map(|s| s.start_payload.expect("a payload").as_f64() as i64)
        .collect();
    assert_eq!(tiles, (0..k_tiles as i64).collect());
    // The same kernel on the simulator records the same events per warp.
    let mut sim_args = vec![
        tensorop_gemm::input(cfg.ab, &a),
        tensorop_gemm::input(cfg.ab, &b),
        tensorop_gemm::input(cfg.c, &vec![0.0; m * n]),
    ];
    let sim = sim_trace(&kernel, launch, &mut sim_args);
    for cta in 0..ctas as u32 {
        for warp in 0..warps as u32 {
            let names = |t: &IketTrace| -> Vec<String> {
                t.warp_records(cta, warp).iter().map(|r| format!("{:?}:{}", r.kind, r.event)).collect()
            };
            assert_eq!(names(&trace), names(&sim), "CTA {cta} warp {warp}");
        }
    }

    let path = std::env::var_os("CUTE_IKET_TRACE_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("tensorop_gemm_iket_trace.json"));
    trace.write_chrome_trace(&path).unwrap();
    let json = std::fs::read_to_string(&path).unwrap();
    assert!(json.starts_with("{\"traceEvents\": [") && json.contains("\"name\": \"prologue\""));
    eprintln!("wrote {}", path.display());
}
