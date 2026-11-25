//! `PipelineCpAsync`, `PipelineOrder`, the grouped GEMM scheduler and the
//! contiguous group search on a real GPU (roadmap parts
//! m9-pipeline-cp-async-order and m9-grouped-gemm-schedulers), each held to
//! its plain-Rust check and bit for bit to the simulator. The feature
//! kernels also run in `tests/gpu.rs` with the other sm_89 ones, and the
//! `async_pipeline` example there. Needs the `cuda` feature; skipped (with
//! a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_cpasync_grouped
//! ```

mod feature_kernels;
#[allow(dead_code)]
#[path = "../examples/grouped_gemm.rs"]
mod grouped_gemm;
#[allow(dead_code)]
mod grouped_kernels;

use cute::cuda::CudaContext;
use cute::prelude::*;
use grouped_kernels::*;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

/// Run `kernel` on the simulator and on the GPU from the same arguments.
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

/// The `cp.async` rings and the turn logs (sm_89) on small and random
/// data, 10 launches each: the plain-Rust result, bit-identical to the
/// simulator.
#[test]
fn rings_and_logs_on_the_gpu() {
    let Some(device) = device() else { return };
    for case in feature_kernels::cpasync_order_cases_sm89() {
        for small in [true, false] {
            let input = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            for _ in 0..10 {
                let (sim, gpu) = sim_and_gpu(&device, &case.kernel, case.cfg, &input);
                (case.check)(&gpu);
                assert!(sim == gpu, "{}: the GPU and the simulator differ", case.name);
            }
        }
    }
}

/// Every grouped problem in every search mode: every CTA tile of every
/// group visited once with the right group, coordinates and shape, the
/// same records as the simulator.
#[test]
fn grouped_scheduler_on_the_gpu() {
    let Some(device) = device() else { return };
    for p in grouped_problems() {
        for mode in SearchMode::ALL {
            let kernel = grouped_enumeration_kernel(&p, mode);
            let (cfg, args) = grouped_enumeration_launch(&p);
            let (sim, gpu) = sim_and_gpu(&device, &kernel, cfg, &args);
            check_grouped_enumeration(&p, &gpu);
            assert!(sim == gpu, "{mode:?} {p:?}: the GPU and the simulator differ");
        }
    }
}

/// The contiguous search over random boundaries: the transcription's
/// states, as on the simulator.
#[test]
fn contiguous_search_on_the_gpu() {
    let Some(device) = device() else { return };
    for p in contiguous_problems() {
        let kernel = contiguous_search_kernel(&p);
        let (cfg, args) = contiguous_search_launch(&p);
        let (sim, gpu) = sim_and_gpu(&device, &kernel, cfg, &args);
        check_contiguous_search(&p, &gpu);
        assert!(sim == gpu, "{p:?}: the GPU and the simulator differ");
    }
}

/// `examples/grouped_gemm.rs`: the persistent `mma.sync` grouped GEMM on
/// every problem (direct and cached shapes, 3 and 64 persistent CTAs) and
/// the MoE layer (1 and 3 CTAs), exact against the CPU and bit-identical
/// to the simulator.
#[test]
fn grouped_gemm_example_on_the_gpu() {
    let Some(device) = device() else { return };
    for groups in grouped_gemm::problems() {
        for cached in [false, true] {
            let kernel = grouped_gemm::build(groups.group_count(), cached);
            for ctas in [3, 64] {
                let (cfg, args, a, b) = grouped_gemm::problem(&groups, ctas);
                let (sim, gpu) = sim_and_gpu(&device, &kernel, cfg, &args);
                grouped_gemm::check(&groups, &a, &b, gpu[2].as_f32());
                assert!(sim == gpu, "{groups:?} cached {cached}: the GPU and the simulator differ");
            }
        }
    }
    for p in grouped_gemm::moe_problems() {
        let kernel = grouped_gemm::moe(p.tokens.len());
        for ctas in [1, 3] {
            for _ in 0..5 {
                let (cfg, args, x, w) = grouped_gemm::moe_problem(&p, ctas);
                let (sim, gpu) = sim_and_gpu(&device, &kernel, cfg, &args);
                grouped_gemm::moe_check(&p, &x, &w, gpu[2].as_f32());
                assert!(sim == gpu, "{p:?}: the GPU and the simulator differ");
            }
        }
    }
}
