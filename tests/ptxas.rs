//! PTX for targets no available GPU can run, checked by NVIDIA's assembler:
//! registered kernels are compiled with `Kernel::compile_ptx` for each of
//! their architectures and assembled with `ptxas -arch=<chip>`, and any
//! error fails the test. Skipped (with a note) unless `ptxas` is on `PATH`
//! and an `mlir-opt` is found:
//!
//! ```text
//! PATH=/usr/local/cuda/bin:$PATH cargo test --test ptxas
//! ```

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/block_reduce.rs"]
mod block_reduce;
#[path = "../examples/cp_async_mbarrier_arrive.rs"]
mod cp_async_mbarrier_arrive;
mod feature_kernels;
#[path = "../examples/fp4_x2_tensor.rs"]
mod fp4_x2_tensor;
#[path = "../examples/pipeline_software_fill.rs"]
mod pipeline_software_fill;
#[path = "../examples/redux_sync.rs"]
mod redux_sync;
#[path = "../examples/runtime_gemm.rs"]
mod runtime_gemm;
#[path = "../examples/single_warp_pipeline.rs"]
mod single_warp_pipeline;
#[path = "../examples/tensorop_gemm.rs"]
mod tensorop_gemm;
#[path = "../examples/warp_named_barrier.rs"]
mod warp_named_barrier;
// The TMA tutorials.
#[path = "../examples/tma_copy.rs"]
mod tma_copy;
#[path = "../examples/tma_transpose.rs"]
mod tma_transpose;

use std::path::{Path, PathBuf};
use std::process::Command;

use cute::prelude::*;
use feature_kernels::Case;

/// The Hopper-or-newer targets, with and without their
/// architecture-specific features.
const HOPPER_AND_NEWER: [Arch; 4] = [Arch::Sm90, Arch::Sm90a, Arch::Sm100a, Arch::Sm120a];

/// Every target.
const EVERY_ARCH: [Arch; 9] =
    [Arch::Sm80, Arch::Sm86, Arch::Sm89, Arch::Sm90, Arch::Sm90a, Arch::Sm100, Arch::Sm100a, Arch::Sm120, Arch::Sm120a];

/// The targets from sm_86 on, which have `max.xorsign.abs`.
const SM86_AND_NEWER: [Arch; 8] =
    [Arch::Sm86, Arch::Sm89, Arch::Sm90, Arch::Sm90a, Arch::Sm100, Arch::Sm100a, Arch::Sm120, Arch::Sm120a];

/// The targets from sm_89 on, which have the FP8 conversions.
const SM89_AND_NEWER: [Arch; 7] =
    [Arch::Sm89, Arch::Sm90, Arch::Sm90a, Arch::Sm100, Arch::Sm100a, Arch::Sm120, Arch::Sm120a];

/// The targets of Float32 `redux.sync` and the UE8M0 conversions that cute-rs has (sm_100a; the
/// sm_100f family is not a target yet).
const SM100A: [Arch; 1] = [Arch::Sm100a];

/// The kernels to assemble and the targets to assemble each for: the
/// feature kernels that need SM90 (which the SM89 GPU cannot run, among them
/// the cluster registers and `aggr_smem_size`) for every Hopper-or-newer
/// target, the integer and bit instructions, the carry chains, the
/// hand-written PTX and MLIR, `dynamic_smem_size`, `get_dyn_smem_size`,
/// `total_smem_size`, `cp.async.mbarrier.arrive`, and the ordered and
/// cache-hinted loads and stores, prefetches and G2R and R2G copies, which
/// the GPU test only checks on SM89, and the warp-collective kernels
/// (`feature_kernels::warp_collectives`) for every target, `fmax` and `fmin`
/// (`max.xorsign.abs`) for every target from sm_86 on, and Float32
/// `redux.sync` for sm_100a, and the kernels over tensors of runtime
/// extents (`feature_kernels::dynamic_tiles` and `runtime_gemm` with
/// runtime extents), whose divisions by runtime extents the GPU test only
/// runs on SM89, for every target. Parts that add SM90+ kernels register
/// them here.
fn registered() -> Vec<(Case, &'static [Arch])> {
    let mut kernels: Vec<(Case, &'static [Arch])> =
        feature_kernels::sm90().into_iter().map(|c| (c, &HOPPER_AND_NEWER[..])).collect();
    for case in [
        feature_kernels::bit_fields_i32(),
        feature_kernels::bit_fields_u32(),
        feature_kernels::bit_fields_i64(),
        feature_kernels::bit_fields_u64(),
        feature_kernels::logic_and_shifts(),
        feature_kernels::int_multiplies(),
        feature_kernels::bit_counts(),
        feature_kernels::carry_words_u32(),
        feature_kernels::carry_words_u64(),
        feature_kernels::carry_multiply(),
        feature_kernels::carry_in_loop(),
        feature_kernels::inline_ptx_named(),
        feature_kernels::inline_asm_raw(),
        feature_kernels::mlir_ops(),
        feature_kernels::inline_ptx_pointers(),
        feature_kernels::mlir_nsmid(),
        feature_kernels::mixed_carry_flag(),
        feature_kernels::carry_around_division(),
        feature_kernels::mlir_pointers(),
        feature_kernels::carry_around_collectives(),
        // dynamic_smem_size, get_dyn_smem_size and total_smem_size (inline PTX) in the dynamic
        // window, and cp.async.mbarrier.arrive with and without .noinc.
        feature_kernels::dynamic_region(),
        feature_kernels::cp_async_mbarrier(),
        // Carry chains around both.
        feature_kernels::carry_around_smem_registers(),
        // Ordered and cache-hinted loads and stores, prefetches, and G2R and R2G copies.
        feature_kernels::ordered_handoffs(),
        feature_kernels::cache_hints(),
        feature_kernels::prefetches(),
        feature_kernels::global_copy_atoms(),
        feature_kernels::cotiled_copies(),
        feature_kernels::strong_vectors(),
    ] {
        kernels.push((case, &EVERY_ARCH[..]));
    }
    // The warp collectives on every target, fmax and fmin from sm_86 on, and
    // Float32 redux.sync, which no available GPU runs, for sm_100a.
    kernels.extend(feature_kernels::warp_collectives().into_iter().map(|c| (c, &EVERY_ARCH[..])));
    kernels.extend(feature_kernels::sm86().into_iter().map(|c| (c, &SM86_AND_NEWER[..])));
    // The FP8 conversions, FP8 mma.sync and LdMatrix8x16x8b from sm_89 on, and
    // the sparse mma.sp on every target.
    kernels.extend(feature_kernels::sm89().into_iter().map(|c| (c, &SM89_AND_NEWER[..])));
    kernels.extend(feature_kernels::mma_sparse().into_iter().map(|c| (c, &EVERY_ARCH[..])));
    kernels.extend(feature_kernels::sm100a().into_iter().map(|c| (c, &SM100A[..])));
    // Tensor memory.
    kernels.extend(feature_kernels::sm100().into_iter().map(|c| (c, &SM100A[..])));
    // setmaxnreg on sm_90a.
    kernels.extend(feature_kernels::sm90a().into_iter().map(|c| (c, &[Arch::Sm90a][..])));
    // The FP4 and FP6 cvt pairs on sm_120a too.
    kernels.extend(feature_kernels::sm120a().into_iter().map(|c| (c, &[Arch::Sm120a][..])));
    // FP4 and FP6 storage, FP6 straddling words, on every target.
    kernels.push((feature_kernels::mx_storage(), &EVERY_ARCH[..]));
    // The packed half-precision operations, packed exp2 and the conversions,
    // inline PTX the GPU test only runs on SM89, for every target (their SM90
    // forms, bf16x2 add and sub, ex2.approx.ftz.bf16x2, .oob and cvt.rn.bf16.s8,
    // are in feature_kernels::sm90 above).
    kernels.extend(feature_kernels::packed_half_and_conversions().into_iter().map(|c| (c, &EVERY_ARCH[..])));
    kernels.push((feature_kernels::exp2_f16x2(), &EVERY_ARCH[..]));
    // Tensors of runtime extents on every target.
    kernels.push((feature_kernels::dynamic_tiles(), &EVERY_ARCH[..]));
    let gemm = Case {
        name: "runtime_gemm_dynamic",
        kernel: runtime_gemm::build_dynamic(),
        cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]),
        small_only: true,
        check: |_| {},
    };
    kernels.push((gemm, &EVERY_ARCH[..]));
    // The atomics and reductions (inline PTX, with explicit semantics and
    // scope) on every target; their BFloat16 and cluster-scope forms are in
    // feature_kernels::sm90 above.
    for case in [
        feature_kernels::atomic_orders(),
        feature_kernels::atomic_lock(),
        feature_kernels::atomic_float_adds(),
        feature_kernels::atomic_float_extrema(),
        feature_kernels::reductions(),
    ] {
        kernels.push((case, &EVERY_ARCH[..]));
    }
    // The ported reduction examples' kernels no available GPU runs: the
    // BFloat16 block sum (red.add.noftz.bf16, sm_90) and redux_sync.py's
    // Float32 redux.sync kernels (sm_100a).
    let example = |kernel: Kernel| Case {
        name: "reduction example",
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]),
        small_only: true,
        check: |_| {},
    };
    kernels.push((example(block_reduce::build::<BF16>(4, 4, block_reduce::ReduceOp::Add)), &HOPPER_AND_NEWER[..]));
    kernels.push((example(redux_sync::float_reductions()), &SM100A[..]));
    kernels.push((example(redux_sync::redux_vs_shuffle(true)), &SM100A[..]));
    // The conversion examples: the FP8 GEMM (FP8 mma.sync) and kernel 4 of redux_sync.py with the shuffle
    // tree (the FP8 cvt) from sm_89 on, kernel 4 with redux.sync on sm_100a, and the FP4 recast
    // decoded with integer operations everywhere and with cvt.rn.f16x2.e2m1x2 on sm_100a and
    // sm_120a.
    let fp8 = tensorop_gemm::GemmConfig::fp8(256, 256, 256, DType::F8E4M3FN);
    kernels.push((example(tensorop_gemm::build(fp8)), &SM89_AND_NEWER[..]));
    let fp8_c = tensorop_gemm::GemmConfig { ab: DType::F8E5M2, c: DType::F8E5M2, ..fp8 };
    kernels.push((example(tensorop_gemm::build(fp8_c)), &SM89_AND_NEWER[..]));
    kernels.push((example(redux_sync::absmax_quantize(redux_sync::DEFAULT_NUM_ROWS, false)), &SM89_AND_NEWER[..]));
    kernels.push((example(redux_sync::absmax_quantize(redux_sync::DEFAULT_NUM_ROWS, true)), &SM100A[..]));
    kernels.push((example(fp4_x2_tensor::build(fp4_x2_tensor::Decode::Integer)), &EVERY_ARCH[..]));
    kernels.push((example(fp4_x2_tensor::build(fp4_x2_tensor::Decode::Cvt)), &[Arch::Sm100a, Arch::Sm120a][..]));
    // The warp-specialised examples' SM90 builds (try_wait and the init
    // fence) and the cluster fan-in, which the SM89 GPU cannot run.
    kernels.push((example(warp_named_barrier::build(2, Arch::Sm90)), &HOPPER_AND_NEWER[..]));
    kernels.push((example(pipeline_software_fill::build(2, Arch::Sm90)), &HOPPER_AND_NEWER[..]));
    kernels.push((example(cp_async_mbarrier_arrive::cta(Arch::Sm90)), &HOPPER_AND_NEWER[..]));
    kernels.push((example(cp_async_mbarrier_arrive::cluster()), &HOPPER_AND_NEWER[..]));
    // The TMA tutorials (TMA loads and stores, sm_90; their kernels declare the target).
    kernels.push((example(tma_copy::Sm100SimpleCopyKernel::new().build()), &HOPPER_AND_NEWER[..]));
    kernels.push((example(tma_transpose::Sm100MatrixTransposeKernelV1::new().build()), &HOPPER_AND_NEWER[..]));
    kernels
}

/// `ptxas` on `PATH`.
fn find_ptxas() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("ptxas")).find(|p| p.is_file())
}

/// Assemble `ptx` for `arch`; the error is `ptxas`'s output.
fn assemble(ptxas: &Path, ptx: &str, arch: Arch, name: &str) -> Result<(), String> {
    assemble_for(ptxas, ptx, arch.chip(), name)
}

/// Assemble `ptx` for the target named `chip`.
fn assemble_for(ptxas: &Path, ptx: &str, chip: &str, name: &str) -> Result<(), String> {
    let stem = std::env::temp_dir().join(format!("cute-ptxas-{}-{name}-{chip}", std::process::id()));
    let (source, cubin) = (stem.with_extension("ptx"), stem.with_extension("cubin"));
    std::fs::write(&source, ptx).map_err(|e| format!("writing {}: {e}", source.display()))?;
    let output = Command::new(ptxas)
        .arg(format!("-arch={chip}"))
        .arg(&source)
        .arg("-o")
        .arg(&cubin)
        .output()
        .map_err(|e| format!("running {}: {e}", ptxas.display()));
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&cubin);
    let output = output?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr)))
    }
}

#[test]
fn registered_kernels_assemble_for_their_targets() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: no ptxas on PATH");
        return;
    };
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let jobs: Vec<(Kernel, Arch)> = registered()
        .into_iter()
        .flat_map(|(case, archs)| archs.iter().map(move |&arch| (case.kernel.clone(), arch)))
        // A kernel that declares its target (KernelOptions::target) compiles for that
        // target only.
        .filter(|(kernel, arch)| kernel.target().is_none_or(|t| t == *arch))
        .collect();
    // A few at a time: each job runs mlir-opt and then ptxas.
    let workers = 4;
    let failures: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let (jobs, toolchain, ptxas) = (&jobs, &toolchain, &ptxas);
                scope.spawn(move || {
                    jobs.iter()
                        .skip(w)
                        .step_by(workers)
                        .filter_map(|(kernel, arch)| {
                            let name = kernel.name();
                            let options = feature_kernels::options_for(kernel, toolchain).gpu_arch(*arch);
                            let result = kernel
                                .compile_with(toolchain, &options)
                                .map(|c| c.ptx().to_string())
                                .map_err(|e| e.to_string())
                                .and_then(|ptx| assemble(ptxas, &ptx, *arch, name));
                            result.err().map(|e| format!("{name} for {}: {e}", arch.chip()))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("worker thread")).collect()
    });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("assembled {} kernel/target pairs with {}", jobs.len(), ptxas.display());
}

/// `ptxas` (12.8 on the test pod, PTX ISA 8.7 at most) assembles
/// `single_warp_pipeline` for every target it has: sm_80, sm_86, sm_87,
/// sm_89, sm_90, sm_90a, sm_100a, sm_101a (Thor under its old name, which
/// cute-rs compiles for `Arch::Sm110a` when the LLVM has no sm_110) and
/// sm_120a, each at its own PTX ISA version.
#[test]
fn single_warp_pipeline_assembles_for_every_ptx87_target() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: no ptxas on PATH");
        return;
    };
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let mlir = single_warp_pipeline::build().to_mlir();
    let targets = [
        (Arch::Sm80.chip(), Arch::Sm80.min_ptx()),
        (Arch::Sm86.chip(), Arch::Sm86.min_ptx()),
        (Arch::Sm87.chip(), Arch::Sm87.min_ptx()),
        (Arch::Sm89.chip(), Arch::Sm89.min_ptx()),
        (Arch::Sm90.chip(), Arch::Sm90.min_ptx()),
        (Arch::Sm90a.chip(), Arch::Sm90a.min_ptx()),
        (Arch::Sm100a.chip(), Arch::Sm100a.min_ptx()),
        (Arch::Sm110a.legacy_chip().unwrap(), Arch::Sm110a.legacy_min_ptx().unwrap()),
        (Arch::Sm120a.chip(), Arch::Sm120a.min_ptx()),
    ];
    for (chip, ptx) in targets {
        assert!(ptx <= 87, "{chip}");
        let code = toolchain.compile_ptx_for_chip(&mlir, chip, ptx).unwrap_or_else(|e| panic!("{chip}: {e}"));
        assert!(code.contains(&format!(".target {chip}\n")), "{chip}:\n{code}");
        assemble_for(&ptxas, &code, chip, "single_warp_pipeline").unwrap_or_else(|e| panic!("{chip}: {e}"));
    }
}

/// The TMA store, reduction, prefetch, cache-hint,
/// multicast, peer-copy and `st.async` kernels assemble for sm_90 and sm_90a (they are
/// also among the registered kernels; this runs them alone, for the lane's checks).
#[test]
fn tma_store_and_cluster_copy_kernels_assemble_for_sm90() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: no ptxas on PATH");
        return;
    };
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let cases = feature_kernels::tma_store_cases_sm90().into_iter().chain(feature_kernels::cluster_copy_cases_sm90());
    let mut failures = Vec::new();
    let mut count = 0;
    for case in cases {
        for arch in [Arch::Sm90, Arch::Sm90a] {
            let options = feature_kernels::options_for(&case.kernel, &toolchain).gpu_arch(arch);
            let result = case
                .kernel
                .compile_with(&toolchain, &options)
                .map(|c| c.ptx().to_string())
                .map_err(|e| e.to_string())
                .and_then(|ptx| assemble(&ptxas, &ptx, arch, case.name));
            if let Err(e) = result {
                failures.push(format!("{} for {}: {e}", case.name, arch.chip()));
            }
            count += 1;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("assembled {count} kernel/target pairs with {}", ptxas.display());
}

/// The im2col kernels for sm_90 and sm_90a, and the gather4, scatter4,
/// byte-masked and FP4/FP6 kernels for sm_100a (ptxas 12.8 takes PTX
/// up to 8.7; the sm_107f override kernels have no assembler yet).
#[test]
fn tma_mode_kernels_assemble_for_their_targets() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: no ptxas on PATH");
        return;
    };
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let cases = feature_kernels::tma_im2col_cases_sm90()
        .into_iter()
        .map(|c| (c, &[Arch::Sm90, Arch::Sm90a, Arch::Sm100a][..]))
        .chain(feature_kernels::tma_sm100_cases().into_iter().map(|c| (c, &[Arch::Sm100a][..])));
    let mut failures = Vec::new();
    let mut count = 0;
    for (case, arches) in cases {
        for &arch in arches {
            let options = feature_kernels::options_for(&case.kernel, &toolchain).gpu_arch(arch);
            let result = case
                .kernel
                .compile_with(&toolchain, &options)
                .map(|c| c.ptx().to_string())
                .map_err(|e| e.to_string())
                .and_then(|ptx| assemble(&ptxas, &ptx, arch, case.name));
            if let Err(e) = result {
                failures.push(format!("{} for {}: {e}", case.name, arch.chip()));
            }
            count += 1;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("assembled {count} kernel/target pairs with {}", ptxas.display());
}

/// The TMEM allocation kernels assemble for sm_100a, and their PTX has the
/// tcgen05 instructions (also among the registered kernels; this runs them alone).
#[test]
fn tmem_kernels_assemble_for_sm100a() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: no ptxas on PATH");
        return;
    };
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let mut failures = Vec::new();
    for case in feature_kernels::sm100() {
        let options = feature_kernels::options_for(&case.kernel, &toolchain).gpu_arch(Arch::Sm100a);
        let ptx = match case.kernel.compile_with(&toolchain, &options) {
            Ok(c) => c.ptx().to_string(),
            Err(e) => {
                failures.push(format!("{}: {e}", case.name));
                continue;
            }
        };
        let g = if case.name.ends_with("pair") { 2 } else { 1 };
        for want in [
            format!("tcgen05.alloc.cta_group::{g}.sync.aligned.shared::cta.b32"),
            format!("tcgen05.dealloc.cta_group::{g}.sync.aligned.b32"),
            format!("tcgen05.relinquish_alloc_permit.cta_group::{g}.sync.aligned"),
        ] {
            if !ptx.contains(&want) {
                failures.push(format!("{}: no `{want}` in the PTX", case.name));
            }
        }
        if let Err(e) = assemble(&ptxas, &ptx, Arch::Sm100a, case.name) {
            failures.push(format!("{}: {e}", case.name));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
