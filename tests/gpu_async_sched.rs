//! `PipelineAsync`'s token kernel, `PipelineEmpty`, `HardwareInfo` and the
//! static persistent tile schedulers on a real GPU. The hand-off feature
//! kernels run in `tests/gpu.rs` with the other sm_89 feature kernels.
//! Needs the `cuda` feature; skipped (with a note) without an NVIDIA GPU or
//! an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_async_sched
//! ```

#[allow(dead_code)]
#[path = "../examples/persistent_gemm.rs"]
mod persistent_gemm;
#[allow(dead_code)]
mod pipeline_kernels;

use cute::cuda::{CudaContext, FuncAttribute, HardwareInfo};
use cute::prelude::*;
use pipeline_kernels::schedulers::*;
use pipeline_kernels::*;

fn device() -> Option<(HardwareInfo, Toolchain)> {
    let info = HardwareInfo::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((info, toolchain?))
}

/// Blocks per SM of an empty kernel taking all the opt-in shared memory,
/// computed here independently of `HardwareInfo`.
fn empty_kernel_blocks_per_sm(ctx: &CudaContext) -> u32 {
    let ptx = ".version 7.0\n.target sm_50\n.address_size 64\n\n.visible .entry probe()\n{\n\tret;\n}\n";
    let f = ctx.load_ptx(ptx, "probe").unwrap();
    let optin = ctx.device_attribute(cute::cuda::DeviceAttribute::MaxSharedMemoryPerBlockOptin).unwrap();
    f.set_attribute(FuncAttribute::MaxDynamicSharedSizeBytes, optin).unwrap();
    let smem = f.available_dynamic_smem_per_block(1, 1).unwrap();
    f.max_active_blocks_per_multiprocessor(1, smem).unwrap()
}

/// SM count, L2 size, the clusters of one CTA that fit (blocks per SM times
/// SMs below sm_90) and the errors: a cluster of 2 below sm_90, cluster
/// sizes 0 and 33, a device id past the device count.
#[test]
fn hardware_info_queries() {
    let Some((info, _)) = device() else { return };
    let ctx = info.context();
    let sms = info.get_device_multiprocessor_count().unwrap();
    let l2 = info.get_l2_cache_size_in_bytes().unwrap();
    assert!(sms > 0 && l2 > 0, "{sms} SMs, {l2} bytes of L2");
    let clusters = info.get_max_active_clusters(1, None).unwrap();
    assert!(clusters > 0);
    let stream = ctx.stream().unwrap();
    assert_eq!(info.get_max_active_clusters(1, Some(&stream)).unwrap(), clusters);
    let (major, _) = ctx.compute_capability();
    if major < 9 {
        assert_eq!(clusters, empty_kernel_blocks_per_sm(ctx) * sms);
        let e = info.get_max_active_clusters(2, None).unwrap_err();
        assert!(e.message.contains("clusters of 2 CTAs need thread block clusters (sm_90 or newer)"), "{e}");
    } else {
        assert!(info.get_max_active_clusters(2, None).unwrap() > 0);
    }
    for bad in [0, 33] {
        let e = info.get_max_active_clusters(bad, None).unwrap_err();
        assert!(e.message.contains(&format!("a cluster has 1 to 32 CTAs, not {bad}")), "{e}");
    }
    let count = CudaContext::device_count().unwrap();
    let e = HardwareInfo::new(count).err().expect("a device id past the count is refused");
    assert!(e.message.contains(&format!("device {count} is out of range")), "{e}");
    println!("{sms} SMs, {l2} bytes of L2, {clusters} resident clusters of one CTA");
}

/// The token kernel on the GPU: the data is right and each side counted a
/// token per item, whichever way the probes went; then the generic body
/// over `PipelineAsync` and `PipelineEmpty`, bit-identical to the
/// simulator.
#[test]
fn tokens_and_pipeline_empty_on_the_gpu() {
    let Some((info, tc)) = device() else { return };
    let ctx = info.context();
    let kernel = token_kernel();
    let mut totals = [0u32; 4];
    for _ in 0..20 {
        let mut args = token_args();
        kernel.run_cuda(ctx, &tc, token_cfg(), &mut args).unwrap_or_else(|e| panic!("{e}"));
        let counts = check_token_run(&args);
        totals.iter_mut().zip(counts).for_each(|(t, c)| *t += c);
    }
    println!("tokens over 20 launches (producer true/false, consumer true/false): {totals:?}");
    for empty in [false, true] {
        let kernel = generic_kernel(empty);
        let mut sim = generic_args();
        kernel.simulate(generic_cfg(), &mut sim).unwrap_or_else(|e| panic!("{e}"));
        let mut gpu = generic_args();
        kernel.run_cuda(ctx, &tc, generic_cfg(), &mut gpu).unwrap_or_else(|e| panic!("{e}"));
        check_generic(&gpu);
        assert!(sim == gpu, "{}: the GPU and the simulator differ", kernel.name());
    }
}

/// The enumeration kernels of unclustered schedules (unswizzled, swizzled
/// along M and N; static, host-passed and runtime schedules) on the GPU,
/// held to the model and to the simulator.
#[test]
fn schedulers_on_the_gpu() {
    let Some((info, tc)) = device() else { return };
    let ctx = info.context();
    for tiles in [[5, 3, 1], [7, 4, 2]] {
        for (swizzle, raster_along_m) in [(1, true), (1, false), (2, true), (4, false)] {
            for max_active_clusters in [3, 1000] {
                let s = TileSchedule { tiles, cluster: [1, 1, 1], swizzle, raster_along_m, max_active_clusters };
                for from in [ParamsFrom::Static, ParamsFrom::Host, ParamsFrom::Runtime] {
                    let kernel = enumeration_kernel(&s, from);
                    let (cfg, mut gpu) = enumeration_launch(&s, from);
                    let mut sim = gpu.clone();
                    kernel.simulate(cfg, &mut sim).unwrap_or_else(|e| panic!("{e}"));
                    kernel.run_cuda(ctx, &tc, cfg, &mut gpu).unwrap_or_else(|e| panic!("{s:?} {from:?}: {e}"));
                    check_enumeration(&s, &gpu);
                    assert!(sim == gpu, "{s:?} {from:?}: the GPU and the simulator differ");
                }
            }
        }
    }
    // The runtime scheduler with one CTA per SM, stopping at a runtime bound.
    let ctas = info.get_device_multiprocessor_count().unwrap() as usize;
    for inner_mode in [0, 1] {
        let kernel = runtime_scheduler_kernel(ctas, inner_mode);
        let (tiles, limit) = ([5, 7], 6);
        let (cfg, mut args) = runtime_scheduler_launch(ctas, tiles, limit);
        kernel.run_cuda(ctx, &tc, cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
        check_runtime_scheduler(ctas, tiles, limit, inner_mode, &args);
    }
}

/// `examples/persistent_gemm.rs` with the grid from
/// `HardwareInfo::get_max_active_clusters(1)`, smaller than the tile
/// count, in each schedule: exact against the CPU and bit-identical to the
/// simulator. The kernel's own cluster occupancy
/// (`CudaFunction::max_active_clusters`) sizes a second launch.
#[test]
fn persistent_gemm_on_the_gpu() {
    let Some((info, tc)) = device() else { return };
    let ctx = info.context();
    let clusters = info.get_max_active_clusters(1, None).unwrap();
    for options in persistent_gemm::variants() {
        let kernel = persistent_gemm::build(options);
        let arch = ctx.arch().map_or(Arch::Sm80, |a| if a.sm() >= 90 { Arch::Sm90 } else { a });
        let compiled = kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap();
        let function = ctx.load(&compiled).unwrap();
        let own = function.max_active_clusters([128, 1, 1], 0, [1, 1, 1], None).unwrap();
        assert!(own > 0);
        if ctx.compute_capability().0 < 9 {
            let per_sm = function.max_active_blocks_per_multiprocessor(128, 0).unwrap();
            assert_eq!(own, per_sm * info.get_device_multiprocessor_count().unwrap());
        }
        let tiles_wanted = (clusters.max(own) as usize + 5).max(8);
        let (m, n, k) = (128 * tiles_wanted.div_ceil(4), 128 * 4, 64);
        for ctas in [clusters, own] {
            let (launch, mut gpu, a, b) = persistent_gemm::problem(m, n, k, ctas, options);
            assert!(launch.grid[2] < ((m / 128) * (n / 128)) as u32, "the grid covers every tile");
            let mut sim = gpu.clone();
            kernel.simulate(launch, &mut sim).unwrap_or_else(|e| panic!("{e}"));
            kernel.run_cuda(ctx, &tc, launch, &mut gpu).unwrap_or_else(|e| panic!("{e}"));
            persistent_gemm::check(m, n, k, &a, &b, gpu[2].as_f32());
            assert!(sim == gpu, "{options:?}: the GPU and the simulator differ");
            println!("persistent_gemm {m}x{n}x{k} on {} CTAs ({options:?}): correct", launch.grid[2]);
        }
    }
}
