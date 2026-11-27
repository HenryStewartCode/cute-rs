//! Every example kernel on a real GPU and on the simulator, comparing every
//! output element. Needs the `cuda` feature, and is skipped (with a note)
//! without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --features cuda --test gpu
//! ```
//!
//! `spill_kernels` and `launch_errors_are_clear` port CUTLASS 4.8.1's
//! `examples/python/CuTeDSL/experimental/compiler_diagnostic/ptxas_spill_cases.py`
//! and `cuda_launch_error.py`, Copyright (c) 2026 NVIDIA CORPORATION &
//! AFFILIATES. All rights reserved. BSD-3-Clause; see THIRD-PARTY-NOTICES.md.
//!
//! Kernels are compiled at the default options plus the environment, so
//! `CUTE_RS_COMPILER_OPT=--opt-level=0` runs the whole suite at O0. Where outputs are
//! compared with the simulator bit for bit, contraction is off (`fp_contract(false)`): a
//! fused multiply-add rounds once, and the simulator rounds every operation.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/ampere_sgemm.rs"]
mod ampere_sgemm;
#[path = "../examples/async_pipeline.rs"]
mod async_pipeline;
#[path = "../examples/block_reduce.rs"]
mod block_reduce;
#[path = "../examples/cooperative_launch.rs"]
mod cooperative_launch;
#[path = "../examples/cp_async_mbarrier_arrive.rs"]
mod cp_async_mbarrier_arrive;
#[path = "../examples/cta_norm.rs"]
mod cta_norm;
#[path = "../examples/cute_ext_memcpy.rs"]
mod cute_ext_memcpy;
#[path = "../examples/cute_ext_sgemm.rs"]
mod cute_ext_sgemm;
#[path = "../examples/elementwise_add.rs"]
mod elementwise_add;
#[path = "../examples/elementwise_apply.rs"]
mod elementwise_apply;
#[path = "../examples/fp4_x2_tensor.rs"]
mod fp4_x2_tensor;
#[path = "../examples/pipeline_software_fill.rs"]
mod pipeline_software_fill;
#[path = "../examples/red.rs"]
mod red;
#[path = "../examples/redux_sync.rs"]
mod redux_sync;
#[path = "../examples/rmsnorm_cluster.rs"]
mod rmsnorm_cluster;
#[path = "../examples/runtime_gemm.rs"]
mod runtime_gemm;
#[path = "../examples/sgemm.rs"]
mod sgemm;
#[path = "../examples/single_warp_pipeline.rs"]
mod single_warp_pipeline;
#[path = "../examples/tensorop_gemm.rs"]
mod tensorop_gemm;
#[path = "../examples/warp_named_barrier.rs"]
mod warp_named_barrier;

mod feature_kernels;
mod math_cases;
mod math_ref;
mod support;

use cute::cuda::{CudaContext, DeviceArg};
use cute::ir::Param;
use cute::prelude::*;
use math_cases::{Ty, approx, fast_within, flushed};
use tensorop_gemm::GemmConfig;

/// The first GPU and an `mlir-opt`, or `None` (with a note) if either is
/// missing.
fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

#[derive(Clone, Copy, Debug)]
enum Data {
    /// Small integers, whose products and sums are exact in f32.
    SmallIntegers,
    /// Values in `[-1, 1)` that use every f32 bit, so sums have to round.
    Random,
}

/// One input per kernel parameter, the same on every run.
fn inputs(kernel: &Kernel, data: Data) -> Vec<KernelArg> {
    let ir = kernel.ir();
    let mut state = 1u64;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        state >> 40
    };
    let mut scalars = [1.5f32, -0.5].into_iter().cycle();
    ir.args
        .iter()
        .map(|param| match param {
            Param::Buffer(b) => {
                let decl = ir.buffer(*b);
                let values: Vec<f32> = (0..decl.len.expect("examples have sized buffers"))
                    .map(|_| match data {
                        Data::SmallIntegers => (next() % 17) as f32 - 8.0,
                        Data::Random => next() as f32 / (1 << 23) as f32 - 1.0,
                    })
                    .collect();
                match decl.elem {
                    DType::F16 => KernelArg::f16_from_f32(&values),
                    DType::BF16 => KernelArg::buffer(&values.iter().map(|&x| BF16::from_f32(x)).collect::<Vec<_>>()),
                    DType::F8E4M3FN => KernelArg::fp8_from_f32::<F8E4M3FN>(&values),
                    DType::F8E5M2 => KernelArg::fp8_from_f32::<F8E5M2>(&values),
                    _ => KernelArg::f32(values),
                }
            }
            Param::Scalar { .. } => KernelArg::ScalarF32(scalars.next().unwrap()),
            Param::ByValue { .. } => panic!("no by-value parameters in the examples"),
        })
        .collect()
}

/// The parameter holding the result: the last buffer, in every example.
fn output_index(kernel: &Kernel) -> usize {
    kernel.ir().args.iter().rposition(|p| matches!(p, Param::Buffer(_))).expect("kernel has a buffer parameter")
}

/// Run `kernel` from the same inputs on the simulator and on the GPU, and
/// return each one's output.
fn run_both(
    device: &(CudaContext, Toolchain),
    kernel: &Kernel,
    cfg: LaunchConfig,
    input: &[KernelArg],
) -> (Vec<f32>, Vec<f32>) {
    let (ctx, toolchain) = device;
    let name = kernel.name();
    let mut on_sim = input.to_vec();
    kernel.simulate(cfg, &mut on_sim).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut on_gpu = input.to_vec();
    // FP8 MMAs as inline PTX where the mlir-opt predates their NVVM form.
    let options = feature_kernels::options_for(kernel, toolchain).with_env().unwrap_or_else(|e| panic!("{e}"));
    kernel.run_cuda_with(ctx, toolchain, &options, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{name}: {e}"));
    let output = output_index(kernel);
    for i in (0..input.len()).filter(|&i| i != output) {
        assert!(on_gpu[i] == input[i], "{name}: the GPU changed input parameter {i}");
    }
    (on_sim[output].to_f32(), on_gpu[output].to_f32())
}

fn assert_identical(kernel: &Kernel, data: Data, sim: &[f32], gpu: &[f32]) {
    let differ: Vec<usize> = (0..sim.len()).filter(|&i| sim[i].to_bits() != gpu[i].to_bits()).collect();
    if let Some(&i) = differ.first() {
        panic!(
            "{}: {} of {} outputs differ on {data:?} data, first at element {i}: simulator {:e}, GPU {:e}",
            kernel.name(),
            differ.len(),
            sim.len(),
            sim[i],
            gpu[i]
        );
    }
}

/// A kernel built from plain arithmetic. IEEE 754 fixes the result of every
/// operation, so the GPU must match the simulator bit for bit on any data.
fn check_bit_exact(kernel: &Kernel, cfg: LaunchConfig) {
    let Some(device) = device() else { return };
    for data in [Data::SmallIntegers, Data::Random] {
        let (sim, gpu) = run_both(&device, kernel, cfg, &inputs(kernel, data));
        assert_identical(kernel, data, &sim, &gpu);
    }
}

fn absolute(arg: &KernelArg) -> KernelArg {
    match arg {
        KernelArg::F32(v) => KernelArg::f32(v.iter().map(|x| x.abs()).collect()),
        KernelArg::F16(v) => KernelArg::F16(v.iter().map(|x| F16::from_f32(x.to_f32().abs())).collect()),
        KernelArg::Bytes { elem: DType::BF16, .. } => KernelArg::buffer(
            &arg.to_vec::<BF16>().iter().map(|x| BF16::from_f32(x.to_f32().abs())).collect::<Vec<_>>(),
        ),
        // Clearing the sign bit is the absolute value in both FP8 formats.
        KernelArg::Bytes { elem: elem @ (DType::F8E4M3FN | DType::F8E5M2), data } => {
            KernelArg::Bytes { elem: *elem, data: data.iter().map(|b| b & 0x7f).collect() }
        }
        other => other.clone(),
    }
}

/// A kernel adding `terms` products into each output with `mma.sync`.
///
/// Sums of small integers are exact, so those outputs must match bit for bit.
/// Other data has to round, and the GPU does not round like the simulator's
/// in-order sum. A product of two f16 or bf16 values is exact in f32, and
/// each addition is off by at most one ulp (2^-23, relative) in any rounding
/// mode, so each result is within `terms * 2^-23 * sum|a * b|` of the exact
/// sum and the two results are within twice that of each other.
fn check_tensor_core(kernel: &Kernel, cfg: LaunchConfig, terms: usize) {
    let Some(device) = device() else { return };
    let (sim, gpu) = run_both(&device, kernel, cfg, &inputs(kernel, Data::SmallIntegers));
    assert_identical(kernel, Data::SmallIntegers, &sim, &gpu);

    let input = inputs(kernel, Data::Random);
    let (sim, gpu) = run_both(&device, kernel, cfg, &input);

    // The kernel run on |A| and |B| gives sum|a * b| for every output.
    let mut magnitude: Vec<KernelArg> = input.iter().map(absolute).collect();
    kernel.simulate(cfg, &mut magnitude).unwrap_or_else(|e| panic!("{e}"));
    let magnitude = magnitude[output_index(kernel)].as_f32();

    // SM89's FP8 mma.sync keeps fewer accumulator bits than Float32 (the PTX ISA leaves the
    // accumulation precision unspecified; FP8 products are exact in Float32, and no Float32
    // summation order strays past the bound below, so this is the hardware, not the simulator).
    // On random data its results stray from the simulator's Float32 sums by 2^-13.7 to 2^-12.0
    // of the magnitude for K from 64 to 2048, not growing with K. FP8 kernels are allowed 2^-13
    // of it for each m16n8k32 instruction (K / 32, plus one), on top of the Float32 bound: the
    // worst case if each instruction loses up to 2^-13 of the magnitude it accumulates.
    let fp8 = kernel.ir().buffers.iter().any(|b| matches!(b.elem, DType::F8E4M3FN | DType::F8E5M2));
    let per_mma = if fp8 { (terms / 32 + 1) as f64 * 2f64.powi(-13) } else { 0.0 };
    for (i, ((&s, &g), &m)) in sim.iter().zip(&gpu).zip(magnitude).enumerate() {
        let allowed = ((2 * terms) as f64 * 2f64.powi(-23) + per_mma) * m as f64;
        let difference = (s as f64 - g as f64).abs();
        assert!(
            difference <= allowed,
            "{}: element {i} differs by {difference:e}, more than rounding allows ({allowed:e}): simulator {s:e}, GPU {g:e}",
            kernel.name()
        );
    }
}

fn check_tensorop_gemm(cfg: GemmConfig) {
    let launch = LaunchConfig::new([(cfg.m / cfg.bm) as u32, (cfg.n / cfg.bn) as u32, 1], [128, 1, 1]);
    check_tensor_core(&tensorop_gemm::build(cfg), launch, cfg.k);
}

/// Every buffer as raw bytes: the simulator and the GPU must agree exactly.
fn assert_all_buffers_equal(name: &str, data: &str, sim: &[KernelArg], gpu: &[KernelArg]) {
    for (i, (s, g)) in sim.iter().zip(gpu).enumerate() {
        assert!(s == g, "{name}: parameter {i} differs between the simulator and the GPU on {data} data");
    }
}

/// The feature kernels this GPU runs: every one in `feature_kernels::all`,
/// those of `feature_kernels::sm86` from sm_86 on and those of
/// `feature_kernels::sm89` from sm_89 on.
fn runnable_features(ctx: &CudaContext) -> Vec<feature_kernels::Case> {
    let mut cases = feature_kernels::all();
    if ctx.arch().is_some_and(|arch| arch.sm() >= 86) {
        cases.extend(feature_kernels::sm86());
    }
    if ctx.arch().is_some_and(|arch| arch.sm() >= 89) {
        cases.extend(feature_kernels::sm89());
    }
    cases
}

/// Every type, operation and intrinsic in `tests/feature_kernels`.
#[test]
fn every_feature_matches_the_simulator() {
    let Some((ctx, toolchain)) = device() else { return };
    for case in runnable_features(&ctx) {
        let datasets: &[(bool, &str)] =
            if case.small_only { &[(true, "small-integer")] } else { &[(true, "small-integer"), (false, "random")] };
        for &(small, label) in datasets {
            let input = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            let mut on_sim = input.clone();
            case.kernel.simulate(case.cfg, &mut on_sim).unwrap_or_else(|e| panic!("{}: {e}", case.name));
            let mut on_gpu = input;
            let options =
                feature_kernels::options_for(&case.kernel, &toolchain).with_env().unwrap_or_else(|e| panic!("{e}"));
            case.kernel
                .run_cuda_with(&ctx, &toolchain, &options, case.cfg, &mut on_gpu)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            assert_all_buffers_equal(case.name, label, &on_sim, &on_gpu);
        }
    }
}

/// The kernels the simulator cannot run (inline PTX through pointers, a
/// memory clobber, hand-written MLIR reading `%nsmid`), held to plain Rust
/// on the GPU instead, at the defaults and at O0 with line information.
#[test]
fn gpu_only_kernels_match_rust() {
    let Some((ctx, toolchain)) = device() else { return };
    let o0 = CompileOptions::new().opt_level(0).generate_line_info(true);
    for case in feature_kernels::gpu_only() {
        for (small, label) in [(true, "small-integer"), (false, "random")] {
            let input = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            let mut args = input.clone();
            case.kernel
                .run_cuda(&ctx, &toolchain, case.cfg, &mut args)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
            let mut at_o0 = input;
            case.kernel
                .run_cuda_with(&ctx, &toolchain, &o0, case.cfg, &mut at_o0)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            assert_all_buffers_equal(case.name, label, &args, &at_o0);
        }
    }
}

/// The runtime-shape ports on the GPU, one compiled kernel per layout for every size:
/// `elementwise_add.py`'s kernel and `sgemm.py`'s for each major mode and
/// both copy widths, the same bits as the simulator (which the example
/// tests hold to CPU references) at sizes that are not tile multiples, K
/// below bK included, and a larger problem against the CPU reference.
#[test]
fn runtime_shape_example_ports_match_the_simulator() {
    let Some((ctx, toolchain)) = device() else { return };
    let add = elementwise_add::dynamic();
    for (m, n) in [(300, 512), (17, 130), (3, 12)] {
        let (_, sim) = elementwise_add::run_dynamic(&add, m, n);
        let (a, b) = elementwise_add::dynamic_inputs(m, n);
        let mut host = elementwise_add::dynamic_host(&a, &b, m, n);
        let cfg = LaunchConfig::new(elementwise_add::dynamic_grid(m, n), [128, 1, 1]);
        add.run_cuda(&ctx, &toolchain, cfg, &mut host).unwrap_or_else(|e| panic!("elementwise_add {m}x{n}: {e}"));
        let gpu = host.2.to_vec::<f32>();
        assert!(sim.iter().zip(&gpu).all(|(x, y)| x.to_bits() == y.to_bits()), "elementwise_add {m}x{n}");
    }
    for majors in [
        ampere_sgemm::Majors::new('m', 'n', 'n', 16),
        ampere_sgemm::Majors::new('m', 'n', 'm', 4),
        ampere_sgemm::Majors::new('k', 'k', 'n', 1),
        ampere_sgemm::Majors::new('k', 'n', 'm', 4),
    ] {
        let kernel = ampere_sgemm::build_for(majors);
        let d = majors.divisibility;
        let on_gpu = |m: usize, n: usize, k: usize, seed: u64| {
            let (mut host, a, b) = ampere_sgemm::problem(majors, m, n, k, seed);
            kernel
                .run_cuda(&ctx, &toolchain, ampere_sgemm::launch(m, n), &mut host)
                .unwrap_or_else(|e| panic!("{majors:?} {m}x{n}x{k}: {e}"));
            (ampere_sgemm::c_row_major(&host.2, m, n), a, b)
        };
        for (m, n, k) in [(128usize, 128usize, 8), (144, 112, 5), (48, 272, 37)] {
            let (m, n) = (m.next_multiple_of(d), n.next_multiple_of(d));
            let (_, sim) = ampere_sgemm::run(&kernel, majors, m, n, k);
            let (gpu, _, _) = on_gpu(m, n, k, 7);
            assert!(sim.iter().zip(&gpu).all(|(x, y)| x.to_bits() == y.to_bits()), "{majors:?} {m}x{n}x{k}");
        }
        let (m, n, k) = (520usize.next_multiple_of(d), 392usize.next_multiple_of(d), 301);
        let (gpu, a, b) = on_gpu(m, n, k, 11);
        let want = ampere_sgemm::reference(m, n, k, &a, &b);
        assert!(gpu.iter().zip(&want).all(|(x, y)| x.to_bits() == y.to_bits()), "{majors:?} {m}x{n}x{k}");
    }
}

/// One compiled kernel run at several problem sizes, on buffers uploaded
/// once and reused.
#[test]
fn runtime_shapes_from_one_kernel() {
    let Some((ctx, toolchain)) = device() else { return };
    let kernel = runtime_gemm::build();
    let arch = ctx.arch().expect("SM80 or newer");
    let options = CompileOptions::new().gpu_arch(arch).with_env().unwrap_or_else(|e| panic!("{e}"));
    let compiled = kernel.compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"));
    let function = ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"));
    for (m, n, k) in [(128, 128, 32), (256, 128, 96), (128, 256, 160), (512, 512, 512)] {
        let (launch, mut args, a, b) = runtime_gemm::problem(m, n, k);
        let stats = ctx.launch(&kernel, &function, launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
        assert!(stats.kernel_seconds > 0.0);
        runtime_gemm::check(m, n, k, &a, &b, args[2].as_f32());
    }

    let (launch, mut args, a, b) = runtime_gemm::problem(256, 256, 64);
    let (ua, ub, uc) = (ctx.upload(&args[0]).unwrap(), ctx.upload(&args[1]).unwrap(), ctx.upload(&args[2]).unwrap());
    let device_args = [
        DeviceArg::Buffer(&ua),
        DeviceArg::Buffer(&ub),
        DeviceArg::Buffer(&uc),
        DeviceArg::Scalar(args[3].clone()),
        DeviceArg::Scalar(args[4].clone()),
    ];
    for _ in 0..3 {
        ctx.launch_async(&kernel, &function, launch, &device_args).unwrap_or_else(|e| panic!("{e}"));
    }
    ctx.synchronize().unwrap_or_else(|e| panic!("{e}"));
    ctx.download(&uc, &mut args[2]).unwrap_or_else(|e| panic!("{e}"));
    runtime_gemm::check(256, 256, 64, &a, &b, args[2].as_f32());
}

#[test]
fn elementwise_add() {
    check_bit_exact(&elementwise_add::naive(), LaunchConfig::new(elementwise_add::NAIVE_GRID, [256, 1, 1]));
    check_bit_exact(&elementwise_add::tiled(), LaunchConfig::new(elementwise_add::TILED_GRID, [128, 1, 1]));
}

#[test]
fn elementwise_apply() {
    use elementwise_apply::{ApplyOp, bf16_kernel, f16_kernel, f32_kernel};
    for op in ApplyOp::F32 {
        let a = f32_kernel(op);
        check_bit_exact(&a.kernel, a.cfg);
    }
    for op in ApplyOp::HALF {
        for a in [f16_kernel(op), bf16_kernel(op)] {
            check_bit_exact(&a.kernel, a.cfg);
        }
    }
}

/// Every operation is IEEE or precise (`rsqrt`), and the reductions' order
/// is fixed, so the GPU matches the simulator bit for bit.
#[test]
fn cta_norm() {
    for cfg in cta_norm::CONFIGS {
        check_bit_exact(&cta_norm::cta_norm::<F16>(cfg), cfg.launch());
        check_bit_exact(&cta_norm::cta_norm::<BF16>(cfg), cfg.launch());
        check_bit_exact(&cta_norm::cta_norm::<f32>(cfg), cfg.launch());
    }
}

#[test]
fn sgemm() {
    // Two 128 x 128 tiles of C.
    check_bit_exact(&sgemm::build(256, 128, 32), LaunchConfig::new([2, 1, 1], [256, 1, 1]));
}

#[test]
fn single_warp_pipeline() {
    check_tensor_core(&single_warp_pipeline::build(), LaunchConfig::new([1, 1, 1], [32, 1, 1]), 128);
}

#[test]
fn tensorop_gemm_grid_and_short_pipeline() {
    check_tensorop_gemm(GemmConfig::new(256, 128, 96));
}

#[test]
fn tensorop_gemm_long_k_loop() {
    check_tensorop_gemm(GemmConfig::new(256, 256, 256));
}

#[test]
fn tensorop_gemm_unswizzled() {
    check_tensorop_gemm(GemmConfig { swizzle: false, ..GemmConfig::new(256, 256, 256) });
}

/// The swizzle on the shared-memory pointers (`Swizzle<2,4,3>` on bytes).
#[test]
fn tensorop_gemm_pointer_swizzle() {
    check_tensorop_gemm(GemmConfig { ptr_swizzle: true, ..GemmConfig::new(256, 256, 256) });
}

/// The A and B stages in one shared-memory struct (`cute_struct!`): `cp.async` and `ldmatrix`
/// through the struct's byte buffer, f16 and FP8.
#[test]
fn tensorop_gemm_struct_storage() {
    check_tensorop_gemm(GemmConfig { smem_struct: true, ..GemmConfig::new(256, 256, 256) });
    check_tensorop_gemm(GemmConfig { smem_struct: true, ..GemmConfig::fp8(256, 128, 192, DType::F8E4M3FN) });
}

#[test]
fn tensorop_gemm_two_stages_unswizzled() {
    check_tensorop_gemm(GemmConfig { stages: 2, swizzle: false, ..GemmConfig::new(128, 128, 64) });
}

/// `tensorop_gemm.py`'s FP8 path on SM89 (`MmaFP8Op(Float8E4M3FN, Float32, (16, 8, 32))`,
/// `LdMatrix8x16x8bOp(False, 4)`, the 128x128x64 tile): bit-exact against the simulator on small
/// integers; on random data within the f32 summation bound plus the FP8 MMA's coarser
/// accumulation (2^-13 of the magnitude per instruction; see `check_tensor_core`).
#[test]
fn tensorop_gemm_fp8_e4m3() {
    if !device().is_some_and(|(ctx, _)| ctx.arch().is_some_and(|a| a.sm() >= 89)) {
        return eprintln!("skipping: FP8 mma.sync needs sm_89");
    }
    check_tensorop_gemm(GemmConfig::fp8(256, 256, 256, DType::F8E4M3FN));
}

/// Float8E5M2 inputs, and FP8 C (the accumulators rounded with `.satfinite`), which must match
/// bit for bit on small integers, whose sums are exact before the rounding.
#[test]
fn tensorop_gemm_fp8_e5m2_and_fp8_c() {
    let Some(device) = device() else { return };
    if !device.0.arch().is_some_and(|a| a.sm() >= 89) {
        return eprintln!("skipping: FP8 mma.sync needs sm_89");
    }
    check_tensorop_gemm(GemmConfig::fp8(256, 128, 192, DType::F8E5M2));
    for (ab, c) in [(DType::F8E4M3FN, DType::F8E4M3FN), (DType::F8E5M2, DType::F8E5M2)] {
        let cfg = GemmConfig { c, ..GemmConfig::fp8(256, 128, 128, ab) };
        let kernel = tensorop_gemm::build(cfg);
        let launch = LaunchConfig::new([2, 1, 1], [128, 1, 1]);
        let (sim, gpu) = run_both(&device, &kernel, launch, &inputs(&kernel, Data::SmallIntegers));
        assert_identical(&kernel, Data::SmallIntegers, &sim, &gpu);
    }
}

/// The same GEMM on bf16 inputs (`MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))`).
#[test]
fn tensorop_gemm_bf16() {
    check_tensorop_gemm(GemmConfig { ab: DType::BF16, ..GemmConfig::new(256, 128, 96) });
}

/// Inputs per `cute::math` function and type on the GPU: every bit pattern
/// of Float16 and BFloat16, and this many sweep inputs (specials, hard cases,
/// uniform samples and random bit patterns) of Float32 and Float64.
fn math_sweep() -> usize {
    std::env::var("CUTE_MATH_SWEEP").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 16)
}

/// Run a math kernel on the simulator and on the GPU; the raw bits of its
/// last buffer from each.
fn math_both(
    device: &(CudaContext, Toolchain),
    kernel: &Kernel,
    n: usize,
    args: Vec<KernelArg>,
    ty: Ty,
) -> (Vec<u64>, Vec<u64>) {
    let (ctx, toolchain) = device;
    let cfg = math_cases::launch(n);
    let mut on_sim = args.clone();
    kernel.simulate(cfg, &mut on_sim).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    let mut on_gpu = args;
    kernel.run_cuda(ctx, toolchain, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    (ty.bits(on_sim.last().unwrap()), ty.bits(on_gpu.last().unwrap()))
}

/// The results that differ, described, or `None`. `nan_any` accepts any NaN
/// for a NaN (for operations whose NaN results PTX leaves unspecified); so
/// does a Float64 result with two NaN operands, whose payload depends on the
/// order the compiler gives the operands.
fn differences(what: &str, ty: Ty, args: &[Vec<f64>], sim: &[u64], gpu: &[u64], nan_any: bool) -> Option<String> {
    let per = sim.len() / args.len().max(1);
    let unspecified = |i: usize| {
        let nans = args[i / per.max(1)].iter().filter(|a| a.is_nan()).count();
        nan_any || (ty == Ty::F64 && nans >= 2)
    };
    let differ: Vec<usize> = (0..sim.len())
        .filter(|&i| sim[i] != gpu[i] && !(unspecified(i) && ty.value(sim[i]).is_nan() && ty.value(gpu[i]).is_nan()))
        .collect();
    let &first = differ.first()?;
    let at = &args[first / per.max(1)];
    Some(format!(
        "{what}: {} of {} results differ, first for {at:?} (result {}): simulator {:e} ({:#x}), GPU {:e} ({:#x})",
        differ.len(),
        sim.len(),
        first % per.max(1),
        ty.value(sim[first]),
        sim[first],
        ty.value(gpu[first]),
        gpu[first]
    ))
}

/// Every `cute::math` function in precise mode (the default) computes the
/// simulator's bits on the GPU, for every type: the expansions use only
/// exactly rounded operations. Fast mode stays within the bounds each
/// function documents, and functions that fastmath leaves alone give the
/// precise bits.
#[test]
fn math_functions_match_the_simulator() {
    let Some(device) = device() else { return };
    let n = math_sweep();
    let mut failures: Vec<String> = Vec::new();
    for case in math_cases::unary_cases() {
        for ty in math_cases::TYPES {
            let xs = math_cases::unary_inputs(&case, n, ty);
            let len = xs.len();
            let kernel = math_cases::unary_kernel(case.name, ty, len, false);
            let (sim, gpu) = math_both(&device, &kernel, len, math_cases::unary_args(ty, &xs), ty);
            let args: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
            failures.extend(differences(&format!("{} {}", case.name, ty.name()), ty, &args, &sim, &gpu, false));
            let fast = math_cases::unary_kernel(case.name, ty, len, true);
            let (_, gpu_fast) = math_both(&device, &fast, len, math_cases::unary_args(ty, &xs), ty);
            match case.fast.filter(|_| ty != Ty::F64) {
                None => failures.extend(differences(
                    &format!("{} {} with fastmath (which has no fast form)", case.name, ty.name()),
                    ty,
                    &args,
                    &gpu,
                    &gpu_fast,
                    false,
                )),
                Some(bound) => {
                    let mut checked = 0;
                    for (&x, &b) in xs.iter().zip(&gpu_fast) {
                        let got = ty.value(b);
                        let want = (case.reference)(x);
                        let ok = fast_within(ty, x, 0.0, got, &want, bound);
                        let ok_flushed =
                            flushed(x).and_then(|z| fast_within(ty, z, 0.0, got, &(case.reference)(z), bound));
                        if ok.is_none() && ok_flushed.is_none() {
                            continue;
                        }
                        checked += 1;
                        if ok != Some(true) && ok_flushed != Some(true) {
                            failures.push(format!(
                                "{} fast {}({x:e}) = {got:e} on the GPU, true value {:e}: outside the documented bound",
                                case.name,
                                ty.name(),
                                approx(&want)
                            ));
                            break;
                        }
                    }
                    if checked < len / 8 {
                        failures.push(format!(
                            "{} fast {}: only {checked} of {len} results have a bound",
                            case.name,
                            ty.name()
                        ));
                    }
                }
            }
        }
    }
    for case in math_cases::binary_cases() {
        for ty in math_cases::TYPES {
            let pairs = (case.inputs)(n, ty);
            let len = pairs.len();
            let kernel = math_cases::binary_kernel(case.name, ty, len, false);
            let (sim, gpu) = math_both(&device, &kernel, len, math_cases::binary_args(ty, &pairs), ty);
            let args: Vec<Vec<f64>> = pairs.iter().map(|&(x, y)| vec![x, y]).collect();
            failures.extend(differences(&format!("{} {}", case.name, ty.name()), ty, &args, &sim, &gpu, false));
            let fast = math_cases::binary_kernel(case.name, ty, len, true);
            let (_, gpu_fast) = math_both(&device, &fast, len, math_cases::binary_args(ty, &pairs), ty);
            match case.fast.filter(|_| ty != Ty::F64) {
                None => failures.extend(differences(
                    &format!("{} {} with fastmath (which has no fast form)", case.name, ty.name()),
                    ty,
                    &args,
                    &gpu,
                    &gpu_fast,
                    false,
                )),
                Some(bound) => {
                    let mut checked = 0;
                    for (&(x, y), &b) in pairs.iter().zip(&gpu_fast) {
                        let got = ty.value(b);
                        let want = (case.reference)(x, y);
                        let Some(ok) = fast_within(ty, x, y, got, &want, bound) else { continue };
                        checked += 1;
                        if !ok {
                            failures.push(format!(
                                "{} fast {}({x:e}, {y:e}) = {got:e} on the GPU, true value {:e}: outside the documented bound",
                                case.name,
                                ty.name(),
                                approx(&want)
                            ));
                            break;
                        }
                    }
                    if checked < len / 8 {
                        failures.push(format!(
                            "{} fast {}: only {checked} of {len} results have a bound",
                            case.name,
                            ty.name()
                        ));
                    }
                }
            }
        }
    }
    for ty in math_cases::TYPES {
        // sincos, fpowi, the exact functions, min/max/clamp.
        let case = math_cases::unary_case("sin");
        let xs = math_cases::unary_inputs(&case, n / 4, ty);
        let args: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
        for name in ["sincos.sin", "sincos.cos"] {
            let kernel = math_cases::unary_kernel(name, ty, xs.len(), false);
            let (sim, gpu) = math_both(&device, &kernel, xs.len(), math_cases::unary_args(ty, &xs), ty);
            failures.extend(differences(&format!("{name} {}", ty.name()), ty, &args, &sim, &gpu, false));
        }

        let cases = math_cases::fpowi_inputs(ty, n / 4);
        let (base, exponent): (Vec<f64>, Vec<i32>) = cases.iter().copied().unzip();
        let fargs = vec![ty.buffer(&base), KernelArg::buffer(&exponent), ty.buffer(&vec![0.0; cases.len()])];
        let (sim, gpu) = math_both(&device, &math_cases::fpowi_kernel(ty, cases.len()), cases.len(), fargs, ty);
        let args: Vec<Vec<f64>> = cases.iter().map(|&(b, k)| vec![b, k as f64]).collect();
        failures.extend(differences(&format!("fpowi {}", ty.name()), ty, &args, &sim, &gpu, false));

        let xs = math_cases::exact_inputs(ty);
        let eargs = vec![ty.buffer(&xs), ty.buffer(&vec![0.0; 7 * xs.len()])];
        let (sim, gpu) = math_both(&device, &math_cases::exact_kernel(ty, xs.len()), xs.len(), eargs, ty);
        let args: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
        // Single instructions whose NaNs the PTX ISA leaves unspecified: the
        // simulator returns the GPU's (see `sim::float::ptx_nan`).
        failures.extend(differences(
            &format!("floor/ceil/trunc/roundeven/abs/absf/neg {}", ty.name()),
            ty,
            &args,
            &sim,
            &gpu,
            false,
        ));

        let pairs = math_cases::minmax_inputs(ty);
        let (a, b): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
        let margs = vec![ty.buffer(&a), ty.buffer(&b), ty.buffer(&vec![0.0; 6 * pairs.len()])];
        let (sim, gpu) = math_both(&device, &math_cases::minmax_kernel(ty, pairs.len()), pairs.len(), margs, ty);
        let args: Vec<Vec<f64>> = pairs.iter().map(|&(x, y)| vec![x, y]).collect();
        failures.extend(differences(&format!("min/max/clamp {}", ty.name()), ty, &args, &sim, &gpu, false));
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// `abs`, `-x`/`neg` and `copysign` give the simulator's bits on the GPU for
/// every type: they are bit operations, so NaNs keep their payloads (and
/// signaling NaNs stay signaling), alone in a kernel, applied to a product,
/// with a constant sign and with their bits used as integers.
#[test]
fn math_sign_operations_match_the_simulator() {
    let Some(device) = device() else { return };
    let (ctx, toolchain) = &device;
    let mut failures: Vec<String> = Vec::new();
    for ty in math_cases::TYPES {
        let xs = math_cases::sign_bit_inputs(ty);
        let ss = math_cases::sign_bit_partners(ty);
        let n = xs.len();
        let args: Vec<Vec<f64>> = xs.iter().zip(&ss).map(|(&x, &s)| vec![ty.value(x), ty.value(s)]).collect();
        for op in math_cases::PLAIN_OPS {
            let kernel = math_cases::plain_kernel(op, ty, n);
            let pargs = vec![ty.buffer_bits(&xs), ty.buffer_bits(&ss), ty.buffer_bits(&vec![0; n])];
            let (sim, gpu) = math_both(&device, &kernel, n, pargs, ty);
            failures.extend(differences(&format!("{op} {} alone", ty.name()), ty, &args, &sim, &gpu, false));
        }
        let kernel = math_cases::sign_context_kernel(ty, n);
        let cfg = math_cases::launch(n);
        let cargs = vec![
            ty.buffer_bits(&xs),
            ty.buffer_bits(&ss),
            ty.buffer_bits(&vec![0; n * math_cases::SIGN_CONTEXT_FLOATS]),
            ty.int_buffer(n * math_cases::SIGN_CONTEXT_INTS),
        ];
        let (mut on_sim, mut on_gpu) = (cargs.clone(), cargs);
        kernel.simulate(cfg, &mut on_sim).unwrap_or_else(|e| panic!("{e}"));
        kernel.run_cuda(ctx, toolchain, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{e}"));
        let what = format!("abs/neg/copysign of arguments and products {}", ty.name());
        failures.extend(differences(&what, ty, &args, &ty.bits(&on_sim[2]), &ty.bits(&on_gpu[2]), false));
        // The integer columns: a Float64 product of two NaNs may be either
        // NaN on the GPU (see `differences`), and so its bits.
        let (sim, gpu) = (ty.int_bits(&on_sim[3]), ty.int_bits(&on_gpu[3]));
        let two_nans = |i: usize| ty == Ty::F64 && args[i].iter().all(|a| a.is_nan());
        let per = math_cases::SIGN_CONTEXT_INTS;
        if let Some(i) = (0..sim.len()).find(|&i| sim[i] != gpu[i] && !two_nans(i / per)) {
            failures.push(format!(
                "integer use of abs/neg/copysign {}: for {:?} (column {}) simulator {:#x}, GPU {:#x}",
                ty.name(),
                args[i / per],
                i % per,
                sim[i],
                gpu[i]
            ));
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// The classification predicates, `ipowi`, `absi`, the explicit rounding
/// modes and `ftz=` (exact IEEE operations, so bit for bit, as are `abs`,
/// `min`, `max` and `clamp` with `ftz=`), and the explicit hardware
/// approximations (`approx=`, `full=`) within their PTX ISA bounds.
#[test]
fn math_options_and_predicates_match_the_simulator() {
    let Some(device) = device() else { return };
    let (ctx, toolchain) = &device;
    let mut failures: Vec<String> = Vec::new();
    for ty in math_cases::TYPES {
        let xs: Vec<f64> = if matches!(ty, Ty::F16 | Ty::BF16) {
            (0..65536).map(|b| ty.value(b)).collect()
        } else {
            math_cases::exact_inputs(ty)
        };
        let kernel = math_cases::classify_kernel(ty, xs.len());
        let cfg = math_cases::launch(xs.len());
        let args = vec![ty.buffer(&xs), KernelArg::buffer(&vec![false; 4 * xs.len()])];
        let (mut on_sim, mut on_gpu) = (args.clone(), args);
        kernel.simulate(cfg, &mut on_sim).unwrap();
        kernel.run_cuda(ctx, toolchain, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{e}"));
        if on_sim[1] != on_gpu[1] {
            failures.push(format!("isnan/isinf/isfinite/isnormal {} differ", ty.name()));
        }
    }
    let (ints, exps) = math_cases::ipowi_inputs();
    let n = ints.len();
    let kernel = math_cases::ipowi_kernel::<i32>(n);
    let args = vec![KernelArg::buffer(&ints), KernelArg::buffer(&exps), KernelArg::buffer(&vec![0i32; n])];
    let (mut on_sim, mut on_gpu) = (args.clone(), args);
    kernel.simulate(math_cases::launch(n), &mut on_sim).unwrap();
    kernel.run_cuda(ctx, toolchain, math_cases::launch(n), &mut on_gpu).unwrap_or_else(|e| panic!("{e}"));
    if on_sim[2] != on_gpu[2] {
        failures.push("ipowi i32 differs".to_string());
    }
    // absi and abs on integers.
    fn absi_differs<T: Integer + Bitwise>(device: &(CudaContext, Toolchain)) -> Option<String> {
        let xs: Vec<T> = math_cases::absi_inputs(T::DTYPE.bits() as u32).into_iter().map(T::from_bits).collect();
        let n = xs.len();
        let kernel = math_cases::absi_kernel::<T>(n);
        let args = vec![KernelArg::buffer(&xs), KernelArg::buffer(&vec![T::from_bits(0); 2 * n])];
        let (mut on_sim, mut on_gpu) = (args.clone(), args);
        kernel.simulate(math_cases::launch(n), &mut on_sim).unwrap();
        kernel.run_cuda(&device.0, &device.1, math_cases::launch(n), &mut on_gpu).unwrap_or_else(|e| panic!("{e}"));
        (on_sim[1] != on_gpu[1]).then(|| format!("absi/abs {} differ", T::DTYPE))
    }
    failures.extend(absi_differs::<i8>(&device));
    failures.extend(absi_differs::<i16>(&device));
    failures.extend(absi_differs::<i32>(&device));
    failures.extend(absi_differs::<i64>(&device));
    failures.extend(absi_differs::<u32>(&device));
    // abs, min, max and clamp with ftz (and propagate_nan), on subnormals
    // and NaNs with payloads.
    let pairs = math_cases::ftz_inputs();
    let n = pairs.len();
    let (a, b): (Vec<u64>, Vec<u64>) = pairs.iter().copied().unzip();
    let fargs =
        vec![Ty::F32.buffer_bits(&a), Ty::F32.buffer_bits(&b), Ty::F32.buffer_bits(&vec![0; n * math_cases::FTZ_OPS])];
    let (sim, gpu) = math_both(&device, &math_cases::ftz_kernel(n), n, fargs, Ty::F32);
    let args: Vec<Vec<f64>> = pairs.iter().map(|&(x, y)| vec![Ty::F32.value(x), Ty::F32.value(y)]).collect();
    failures.extend(differences("abs/min/max/clamp with ftz f32", Ty::F32, &args, &sim, &gpu, false));
    for ty in [Ty::F32, Ty::F64] {
        let pairs = math_cases::option_inputs(ty);
        let len = pairs.len();
        let (a, b): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
        let args: Vec<Vec<f64>> = pairs.iter().map(|&(x, y)| vec![x, y]).collect();
        let mut options: Vec<(String, MathOptions)> =
            [RoundingMode::NearestEven, RoundingMode::Zero, RoundingMode::Up, RoundingMode::Down]
                .into_iter()
                .map(|r| (format!("rounding {r:?}"), MathOptions::rounding(r)))
                .collect();
        if ty == Ty::F32 {
            options.push(("ftz".to_string(), MathOptions::rounding(RoundingMode::NearestEven).with_ftz()));
            options.push(("rz with ftz".to_string(), MathOptions::rounding(RoundingMode::Zero).with_ftz()));
        }
        for (label, opts) in options {
            let kernel = math_cases::options_kernel(ty, len, opts);
            let oargs = vec![ty.buffer(&a), ty.buffer(&b), ty.buffer(&vec![0.0; 6 * len])];
            let (sim, gpu) = math_both(&device, &kernel, len, oargs, ty);
            failures.extend(differences(
                &format!("add/mul/fma/div/sqrt/rcp {} with {label}", ty.name()),
                ty,
                &args,
                &sim,
                &gpu,
                false,
            ));
        }
        for ftz in [false, true] {
            let ops = math_cases::approx_ops(ty, ftz);
            let kernel = math_cases::approx_kernel(ty, len, ftz);
            let oargs = vec![ty.buffer(&a), ty.buffer(&b), ty.buffer(&vec![0.0; ops.len() * len])];
            let (_, gpu) = math_both(&device, &kernel, len, oargs, ty);
            for (k, op) in ops.iter().enumerate() {
                let mut checked = 0;
                for (i, &(x, y)) in pairs.iter().enumerate() {
                    let got = ty.value(gpu[i * ops.len() + k]);
                    let ok = (op.check)(x, y, got, ftz);
                    let Some(ok) = ok else { continue };
                    checked += 1;
                    if !ok {
                        failures.push(format!(
                            "{} {}(ftz {ftz}) of ({x:e}, {y:e}) = {got:e} on the GPU: outside its PTX ISA bound",
                            op.name,
                            ty.name()
                        ));
                        break;
                    }
                }
                if checked < len / 8 {
                    failures.push(format!("{} {}: only {checked} of {len} results checked", op.name, ty.name()));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// A kernel compiled at every optimization level, with and without line information, loads
/// with the matching JIT options (`CU_JIT_OPTIMIZATION_LEVEL`, `CU_JIT_GENERATE_LINE_INFO`)
/// and computes what the simulator does. Plain PTX loads too, and a target the device cannot
/// run is refused before the driver sees it.
#[test]
fn modules_load_with_each_jit_option() {
    let Some((ctx, toolchain)) = device() else { return };
    let arch = ctx.arch().expect("SM80 or newer");
    let kernel = elementwise_add::tiled();
    let cfg = LaunchConfig::new(elementwise_add::TILED_GRID, [128, 1, 1]);
    let input = inputs(&kernel, Data::Random);
    let mut on_sim = input.clone();
    kernel.simulate(cfg, &mut on_sim).unwrap();
    for level in 0..=3 {
        for line_info in [false, true] {
            let options =
                CompileOptions::new().gpu_arch(arch).opt_level(level).generate_line_info(line_info).use_ptxas(false);
            let compiled = kernel.compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"));
            let want = if level == 3 { 4 } else { level };
            assert_eq!(compiled.jit_options()[0], JitOption::OptimizationLevel(want));
            assert_eq!(compiled.jit_options().contains(&JitOption::GenerateLineInfo(true)), line_info);
            let function = ctx.load(&compiled).unwrap_or_else(|e| panic!("O{level}, line info {line_info}: {e}"));
            let mut on_gpu = input.clone();
            ctx.launch(&kernel, &function, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{e}"));
            assert!(on_gpu[2] == on_sim[2], "O{level}, line info {line_info}: the GPU differs from the simulator");
            let mut again = input.clone();
            kernel.run_cuda_with(&ctx, &toolchain, &options, cfg, &mut again).unwrap_or_else(|e| panic!("{e}"));
            assert!(again[2] == on_sim[2]);
        }
    }
    let plain = kernel.compile_ptx(&toolchain, arch).unwrap();
    assert!(ctx.load_ptx(&plain, kernel.name()).is_ok());
    let newer = match arch {
        Arch::Sm80 | Arch::Sm86 | Arch::Sm89 => Some(Arch::Sm90),
        _ => None,
    };
    if let Some(newer) = newer {
        let compiled = kernel.compile_with(&toolchain, &CompileOptions::new().gpu_arch(newer)).unwrap();
        let e = ctx.load(&compiled).err().expect("a newer target is refused").to_string();
        assert!(e.contains("was compiled for sm_90") && e.contains("cannot run"), "{e}");
    }
}

/// Every example and feature kernel without LLVM's optimizations and with line information:
/// the GPU still matches the simulator bit for bit.
#[test]
fn every_kernel_at_o0_with_line_info() {
    let Some((ctx, toolchain)) = device() else { return };
    let options = CompileOptions::new().opt_level(0).generate_line_info(true);
    let gemm = GemmConfig::new(256, 128, 96);
    let examples = [
        (elementwise_add::tiled(), LaunchConfig::new(elementwise_add::TILED_GRID, [128, 1, 1])),
        (sgemm::build(256, 128, 32), LaunchConfig::new([2, 1, 1], [256, 1, 1])),
        (
            tensorop_gemm::build(gemm),
            LaunchConfig::new([(gemm.m / gemm.bm) as u32, (gemm.n / gemm.bn) as u32, 1], [128, 1, 1]),
        ),
        (single_warp_pipeline::build(), LaunchConfig::new([1, 1, 1], [32, 1, 1])),
    ];
    for (kernel, cfg) in examples {
        let input = inputs(&kernel, Data::SmallIntegers);
        let mut on_sim = input.clone();
        kernel.simulate(cfg, &mut on_sim).unwrap_or_else(|e| panic!("{e}"));
        let mut on_gpu = input;
        kernel.run_cuda_with(&ctx, &toolchain, &options, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{e}"));
        assert_all_buffers_equal(kernel.name(), "small-integer", &on_sim, &on_gpu);
    }
    for case in runnable_features(&ctx) {
        let input = feature_kernels::inputs_with_len(&case.kernel, true, 256);
        let mut on_sim = input.clone();
        case.kernel.simulate(case.cfg, &mut on_sim).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let mut on_gpu = input;
        let options = feature_kernels::options_for(&case.kernel, &toolchain).or(options.clone());
        case.kernel
            .run_cuda_with(&ctx, &toolchain, &options, case.cfg, &mut on_gpu)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_all_buffers_equal(case.name, "small-integer", &on_sim, &on_gpu);
    }
}

/// A CUDA toolkit tool, on `PATH` or in `/usr/local/cuda/bin`.
fn cuda_tool(name: &str) -> Option<std::path::PathBuf> {
    let dirs = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default();
    dirs.into_iter()
        .chain([std::path::PathBuf::from("/usr/local/cuda/bin")])
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

/// Run a tool: whether it succeeded, and its output.
fn run_tool(cmd: &std::path::Path, args: &[&str]) -> (bool, String) {
    let out = std::process::Command::new(cmd).args(args).output().expect("run the tool");
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

/// A temporary directory of this test process, removed when dropped.
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!("cute-rs-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Assemble `ptx` with `ptxas` for `sm` at `level`, with `-lineinfo` when asked; the cubin, or
/// the assembler's log.
fn assemble(
    ptxas: &std::path::Path,
    dir: &TempDir,
    name: &str,
    ptx: &str,
    sm: &str,
    level: u32,
    line_info: bool,
) -> Result<std::path::PathBuf, String> {
    let file = dir.0.join(format!("{name}_{sm}_o{level}_{line_info}.ptx"));
    std::fs::write(&file, ptx).unwrap();
    let cubin = file.with_extension("cubin");
    let (arch, opt) = (format!("-arch={sm}"), format!("-O{level}"));
    let mut args = vec![arch.as_str(), opt.as_str()];
    if line_info {
        args.push("-lineinfo");
    }
    args.extend([file.to_str().unwrap(), "-o", cubin.to_str().unwrap()]);
    match run_tool(ptxas, &args) {
        (true, _) => Ok(cubin),
        (false, log) => Err(log),
    }
}

/// The kernels that need SM90 or newer cannot run on older GPUs, but their PTX must assemble:
/// at O0 and O3, with and without line information, for sm_90a and sm_100a (the sm_100a-only
/// kernels for sm_100a, and the sm_120a ones, the FP4 and FP6 conversions, for sm_120a).
#[test]
fn sm90_kernels_assemble_with_ptxas_at_every_level() {
    let (Some(toolchain), Some(ptxas)) = (Toolchain::discover(), cuda_tool("ptxas")) else {
        eprintln!("skipping: no mlir-opt or ptxas");
        return;
    };
    let dir = TempDir::new("sm90-ptxas");
    let mut failures = Vec::new();
    let hopper: &[(Arch, &str)] = &[(Arch::Sm90a, "sm_90a"), (Arch::Sm100a, "sm_100a")];
    let sm100a: &[(Arch, &str)] = &[(Arch::Sm100a, "sm_100a")];
    let sm120a: &[(Arch, &str)] = &[(Arch::Sm120a, "sm_120a")];
    let cases = feature_kernels::sm90().into_iter().map(|c| (c, hopper));
    let cases = cases
        .chain(feature_kernels::sm100a().into_iter().map(|c| (c, sm100a)))
        .chain(feature_kernels::sm90a().into_iter().map(|c| (c, &[(Arch::Sm90a, "sm_90a")][..])))
        .chain(feature_kernels::sm120a().into_iter().map(|c| (c, sm120a)));
    for (case, targets) in cases {
        for &(arch, sm) in targets {
            for (level, line_info) in [(0, false), (0, true), (3, false), (3, true)] {
                let options = CompileOptions::new()
                    .gpu_arch(arch)
                    .opt_level(level)
                    .generate_line_info(line_info)
                    .use_ptxas(false);
                let compiled = case.kernel.compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"));
                if let Err(log) = assemble(&ptxas, &dir, case.name, compiled.ptx(), sm, level, line_info) {
                    failures.push(format!("{} for {sm} at O{level}, line info {line_info}: {log}", case.name));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// With the CUDA toolkit's `ptxas` and `nvdisasm`, line-information PTX assembles at every
/// level, and the SASS maps back to the Rust lines.
#[test]
fn ptxas_and_nvdisasm_show_the_rust_lines() {
    let Some((ctx, toolchain)) = device() else { return };
    let (Some(ptxas), Some(nvdisasm)) = (cuda_tool("ptxas"), cuda_tool("nvdisasm")) else {
        eprintln!("skipping: no ptxas and nvdisasm");
        return;
    };
    let arch = ctx.arch().expect("SM80 or newer");
    // The PTX's own target, which may be architecture-specific (`sm_120a`); ptxas refuses to
    // assemble such PTX for the plain compute capability (`sm_120`).
    let sm = arch.chip().to_string();
    let dir = TempDir::new("line-info");
    let source = include_str!("../examples/elementwise_add.rs");
    let line = source.lines().position(|l| l.contains("a.load((m, n)) + b.load((m, n))")).unwrap() + 1;
    let want = format!("elementwise_add.rs\", line {line}");
    let mut failures = Vec::new();
    for level in 0..=3 {
        let options = CompileOptions::new().gpu_arch(arch).opt_level(level).generate_line_info(true);
        let compiled = elementwise_add::naive().compile_with(&toolchain, &options).unwrap();
        match assemble(&ptxas, &dir, "naive", compiled.ptx(), &sm, level, true) {
            Err(log) => failures.push(format!("ptxas at O{level}: {log}")),
            Ok(cubin) => {
                let (ok, sass) = run_tool(&nvdisasm, &["-g", "-c", cubin.to_str().unwrap()]);
                if !ok || !sass.contains(&want) {
                    failures.push(format!("nvdisasm at O{level} lacks `{want}`:\n{sass}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// One compiled `dynamic_tiles`, whose global tensors have launch-time
/// extents, launched at several extents (grids and buffer sizes, partial
/// tiles at the bottom and right edges): the GPU computes the plain-Rust
/// result, leaves the padding past the tensors alone, and matches the
/// simulator on every buffer.
#[test]
fn dynamic_tiles_at_several_extents_from_one_kernel() {
    let Some((ctx, toolchain)) = device() else { return };
    let case = feature_kernels::dynamic_tiles();
    let arch = ctx.arch().expect("SM80 or newer");
    let options = CompileOptions::new().gpu_arch(arch).with_env().unwrap_or_else(|e| panic!("{e}"));
    let compiled = case.kernel.compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"));
    let function = ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"));
    for (m, n) in [(1, 4), (13, 20), (17, 44), (64, 4), (100, 100), (300, 36), (9, 1000)] {
        for pad in [0, 37] {
            let (cfg, input) = feature_kernels::dynamic_tiles_problem(m, n, pad, (m * 7919 + n + pad) as u64);
            let mut on_gpu = input.clone();
            ctx.launch(&case.kernel, &function, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{m} x {n}: {e}"));
            let want = feature_kernels::dynamic_tiles_expected(m, n, &input);
            assert_eq!(on_gpu[3].to_vec::<i32>(), want, "{m} x {n}, padded by {pad}");
            let mut on_sim = input;
            case.kernel.simulate(cfg, &mut on_sim).unwrap_or_else(|e| panic!("{m} x {n}: {e}"));
            assert_all_buffers_equal(case.name, &format!("{m} x {n} (padded by {pad})"), &on_sim, &on_gpu);
        }
    }
}

/// `runtime_gemm` with `M`, `N` and `K` as the runtime extents of its
/// tensors, compiled once and launched at several problem sizes: correct on
/// random small integers, and bit for bit the result of the static-bound
/// kernel on the GPU, there and on random values whose sums round.
#[test]
fn runtime_extents_gemm_from_one_kernel() {
    let Some((ctx, toolchain)) = device() else { return };
    let arch = ctx.arch().expect("SM80 or newer");
    let options = CompileOptions::new().gpu_arch(arch).with_env().unwrap_or_else(|e| panic!("{e}"));
    let (bounded, dynamic) = (runtime_gemm::build(), runtime_gemm::build_dynamic());
    let compile = |k: &Kernel| k.compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"));
    let (bounded_ptx, dynamic_ptx) = (compile(&bounded), compile(&dynamic));
    let bounded_fn = ctx.load(&bounded_ptx).unwrap_or_else(|e| panic!("{e}"));
    let dynamic_fn = ctx.load(&dynamic_ptx).unwrap_or_else(|e| panic!("{e}"));
    let same_bits = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits());
    for (m, n, k) in [(128, 128, 32), (256, 128, 96), (128, 256, 160), (512, 384, 512)] {
        let (launch, mut args, a, b) = runtime_gemm::problem_dynamic(m, n, k);
        ctx.launch(&dynamic, &dynamic_fn, launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
        runtime_gemm::check(m, n, k, &a, &b, args[2].as_f32());
        let (launch, mut static_args, ..) = runtime_gemm::problem(m, n, k);
        ctx.launch(&bounded, &bounded_fn, launch, &mut static_args).unwrap_or_else(|e| panic!("{e}"));
        let same = same_bits(args[2].as_f32(), static_args[2].as_f32());
        assert!(same, "{m}x{n}x{k}: the runtime-extent kernel's result differs from the static-bound kernel's");

        let (a, b) = (runtime_gemm::random_values(m * k, 3), runtime_gemm::random_values(n * k, 4));
        let (launch, mut static_args) = runtime_gemm::problem_on(m, n, k, &a, &b);
        let mut args = runtime_gemm::with_m(m, static_args.clone());
        ctx.launch(&dynamic, &dynamic_fn, launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
        ctx.launch(&bounded, &bounded_fn, launch, &mut static_args).unwrap_or_else(|e| panic!("{e}"));
        assert!(args[2].as_f32().iter().any(|&x| x != 0.0), "{m}x{n}x{k}: no result on random values");
        let same = same_bits(args[2].as_f32(), static_args[2].as_f32());
        assert!(same, "{m}x{n}x{k}: the kernels' results differ on random values");
    }
}

/// The dynamic shared-memory region behind static buffers at several launch sizes, from 512
/// bytes to the device's opt-in limit (above 48 KB the launcher opts the kernel in): the GPU
/// matches the simulator, which matches the plain-Rust reference, including the size registers.
/// A region one word past the limit is refused before the driver sees it.
#[test]
fn dynamic_shared_memory_at_several_launch_sizes() {
    let Some((ctx, toolchain)) = device() else { return };
    let kernel = feature_kernels::dynamic_region_kernel();
    let start = feature_kernels::DYNAMIC_REGION_START;
    assert_eq!(kernel.ir().shared_memory_layout().dynamic_start, start as usize);
    let limit = ctx.max_shared_memory_per_block() as u32;
    let threads = [64, 1, 1];
    for bytes in
        [512, feature_kernels::DYNAMIC_REGION_BYTES, 48 * 1024 - start, 48 * 1024 - start + 4, 60_000, limit - start]
    {
        let cfg = LaunchConfig::new([1, 1, 1], threads).with_dynamic_smem(bytes);
        let input = feature_kernels::inputs(&kernel, false);
        let mut on_sim = input.clone();
        kernel.simulate(cfg, &mut on_sim).unwrap_or_else(|e| panic!("{bytes} bytes: {e}"));
        feature_kernels::check_dynamic_region(&on_sim, bytes);
        let mut on_gpu = input;
        kernel.run_cuda(&ctx, &toolchain, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{bytes} bytes: {e}"));
        assert_all_buffers_equal(kernel.name(), &format!("a {bytes}-byte region's"), &on_sim, &on_gpu);
    }
    // The opt-in limit per block (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN): 99 KB
    // on SM89, less than the 100 KB per multiprocessor.
    if ctx.compute_capability() == (8, 9) {
        assert_eq!(limit, 101_376);
    }
    let cfg = LaunchConfig::new([1, 1, 1], threads).with_dynamic_smem(limit - start + 4);
    let mut args = feature_kernels::inputs(&kernel, false);
    let e = kernel.run_cuda(&ctx, &toolchain, cfg, &mut args).unwrap_err().to_string();
    let want = format!("needs {} bytes of shared memory per block ({start} sized when it was built", limit + 4);
    assert!(e.contains(&want) && e.contains(&format!("allows a block {limit}")), "{e}");
}

/// A kernel with `static_bytes` of shared memory in one buffer, or one byte and then `big` bytes
/// 16-byte aligned, and a dynamic region aligned to `region` bytes if it is `Some`. Every buffer
/// is stored to and read at runtime indices, so that LLVM keeps all of it; each thread writes the
/// raw `%dynamic_smem_size` (read with inline PTX, which the simulator answers with `raw`, what
/// the launch should pass to the driver), `dynamic_smem_size`, `get_dyn_smem_size`,
/// `total_smem_size` and what it read.
fn smem_sizes_kernel(static_bytes: usize, big: usize, region: Option<usize>, raw: i32) -> Kernel {
    Kernel::builder("smem_sizes").block_dim([32, 1, 1]).build(move || {
        let x = param_tensor::<u8>("x", Layout::col_major(32));
        let out = param_tensor::<i32>("out", Layout::row_major((32, 5)));
        let t = thread_idx().x;
        let mut buffers = Vec::new();
        if static_bytes > 0 {
            buffers.push((alloc_smem::<u8>(static_bytes, 1), static_bytes));
        }
        if big > 0 {
            buffers.push((alloc_smem::<u8>(1, 1), 1));
            buffers.push((alloc_smem::<u8>(big, 16), big));
        }
        if let Some(alignment) = region {
            buffers.push((get_dyn_smem::<u8>(Some(alignment)), 1));
        }
        let v = x.load(t);
        let k = v.to::<i32>() * 977;
        for (b, n) in &buffers {
            b.offset((k + t) % *n as i32).store(v);
        }
        sync_threads();
        let mut sum = Int32::from(0);
        for (b, n) in &buffers {
            sum = sum + b.offset((k + (31 - t)) % *n as i32).load().to::<i32>();
        }
        let echo = |x: &[u64]| vec![x[0]];
        let window = inline_ptx_with_sim::<i32>("mov.u32 $0, %dynamic_smem_size;", &[Int32::from(raw).into()], echo);
        out.store((t, 0), window);
        out.store((t, 1), dynamic_smem_size());
        out.store((t, 2), get_dyn_smem_size());
        out.store((t, 3), total_smem_size());
        out.store((t, 4), sum);
    })
}

/// The raw `%dynamic_smem_size` is the dynamic shared memory the launch passes the driver, which
/// holds the build-time buffers too when they live in the dynamic window (above 48 KB, or with a
/// dynamic region); `dynamic_smem_size` and `get_dyn_smem_size` read the region alone, the bytes
/// the launch asked for; and `%total_smem_size` on an SM89 GPU is the static and dynamic bytes
/// rounded up to the 128-byte allocation unit (bytes, not a count of units), with static buffers
/// laid out in declaration order: the simulator returns the same for each layout. Windows of
/// exactly 128 bytes and 48 KB show that a kernel in the window has no static shared memory
/// (`gpu.dynamic_shared_memory` would leave a byte of it, read as 256 and refused at 48 KB without
/// opting in).
#[test]
fn shared_memory_size_registers_match_the_gpu() {
    let Some((ctx, toolchain)) = device() else { return };
    // (static bytes, big, region alignment, region bytes) and the GPU's (raw %dynamic_smem_size,
    // %total_smem_size).
    type SizeCase = ((usize, usize, Option<usize>, u32), (i32, i32));
    let cases: [SizeCase; 19] = [
        ((0, 0, None, 0), (0, 0)),
        ((0, 0, None, 128), (128, 128)),
        ((0, 0, None, 129), (129, 256)),
        ((129, 0, None, 0), (0, 256)),
        ((100, 0, None, 28), (28, 128)),
        ((100, 0, None, 29), (29, 256)),
        ((1000, 0, None, 1000), (1000, 2048)),
        // A byte, then 112 or 113 bytes at byte 16.
        ((0, 112, None, 0), (0, 128)),
        ((0, 113, None, 0), (0, 256)),
        ((0, 0, Some(1), 129), (129, 256)),
        ((0, 0, Some(1), 128), (128, 128)),
        ((0, 0, Some(1), 48 * 1024), (48 * 1024, 48 * 1024)),
        // The region at byte 112, behind 100 static bytes in the window.
        ((100, 0, Some(1), 4096), (4208, 4224)),
        ((100, 0, Some(1), 16), (128, 128)),
        // The region at byte 1024, as it asks.
        ((100, 0, Some(1024), 100), (1124, 1152)),
        // Above 48 KB the build-time buffers are dynamic shared memory too.
        ((0, 50_000, None, 100), (50_116, 50_176)),
        ((0, 50_000, Some(1), 100), (50_116, 50_176)),
        ((0, 50_000, None, 32), (50_048, 50_048)),
        ((100, 0, None, 60_000), (60_000, 60_160)),
    ];
    for ((static_bytes, big, region, bytes), (raw, total)) in cases {
        let kernel = smem_sizes_kernel(static_bytes, big, region, raw);
        let layout = kernel.ir().shared_memory_layout();
        assert_eq!(layout.launch_bytes(bytes as usize), raw as usize, "the launch's dynamic shared memory");
        let cfg = LaunchConfig::new([1, 1, 1], [32, 1, 1]).with_dynamic_smem(bytes);
        let input = vec![KernelArg::buffer(&[9u8; 32]), KernelArg::buffer(&[0i32; 160])];
        let what = format!("{static_bytes} static bytes, {big} big, region {region:?}, {bytes} dynamic");
        let mut on_sim = input.clone();
        kernel.simulate(cfg, &mut on_sim).unwrap_or_else(|e| panic!("{what}: {e}"));
        let row = &on_sim[1].to_vec::<i32>()[..5];
        let buffers = (static_bytes > 0) as i32 + 2 * (big > 0) as i32 + region.is_some() as i32;
        assert_eq!(row, [raw, bytes as i32, bytes as i32, total, 9 * buffers], "{what}");
        let mut on_gpu = input;
        kernel.run_cuda(&ctx, &toolchain, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_all_buffers_equal(kernel.name(), &what, &on_sim, &on_gpu);
    }
}

/// Words for a half-precision sweep: every 16-bit value in each half of `a`
/// (the low half `v`, the high half `v ^ 0x8000`) against each partner in
/// `b` and `c`.
fn half_sweep_inputs(partners: &[u16]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let n = partners.len();
    let word = |lo: u16, hi: u16| (lo as u32 | (hi as u32) << 16) as i32;
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for p in 0..n {
        for v in 0..=0xffffu16 {
            a.push(word(v, v ^ 0x8000));
            b.push(word(partners[p], partners[(p + 5) % n]));
            c.push(word(partners[(p + 3) % n], partners[(p + 11) % n]));
        }
    }
    (a, b, c)
}

/// The partners of the sweeps: the feature kernels' special values and
/// pseudo-random patterns.
fn half_partners(specials: &[u16]) -> Vec<u16> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut v = specials.to_vec();
    for _ in 0..10 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        v.push(state as u16);
    }
    v
}

/// Every Float16 value against partners (the special values, and others),
/// for every packed and scalar Float16 operation and option this GPU has
/// (add and sub with `rn`, `ftz` and `sat`; fma with `ftz`, `sat` and
/// `relu`; `fma_f16`; `fma` on Float16 values), and every BFloat16 value for
/// BFloat16 fma (`fma_packed_bf16x2`, `fma_bf16`, `fma` on BFloat16 values,
/// with and without `relu`): the GPU gives the simulator's bits.
#[test]
fn half_precision_sweeps_match_the_simulator() {
    use cute::ir::HalfFn::{Add, Fma, Sub};
    use feature_kernels::{SweepForm, half_opts as o, half_sweep};
    let Some((ctx, toolchain)) = device() else { return };
    let none = HalfOptions::default();
    let relu = o(false, false, false, true);
    let add_options =
        vec![none, o(true, false, false, false), o(false, true, false, false), o(false, false, true, false)];
    let add_options = [add_options, vec![o(true, true, true, false)]].concat();
    let fma_options = vec![none, o(false, true, false, false), o(false, false, true, false), relu];
    let fma_options = [fma_options, vec![o(false, true, true, false), o(false, true, false, true)]].concat();
    let cases = [
        (Add, DType::F16, SweepForm::Packed, add_options.clone()),
        (Sub, DType::F16, SweepForm::Packed, add_options),
        (Fma, DType::F16, SweepForm::Packed, fma_options.clone()),
        (Fma, DType::F16, SweepForm::Scalar, fma_options),
        (Fma, DType::F16, SweepForm::Plain, vec![none]),
        (Fma, DType::BF16, SweepForm::Packed, vec![none, relu]),
        (Fma, DType::BF16, SweepForm::Scalar, vec![none, relu]),
        (Fma, DType::BF16, SweepForm::Plain, vec![none]),
    ];
    let mut failures = Vec::new();
    for (func, elem, form, options) in cases {
        let specials =
            if elem == DType::F16 { &feature_kernels::F16_SPECIALS } else { &feature_kernels::BF16_SPECIALS };
        let (a, b, c) = half_sweep_inputs(&half_partners(specials));
        let n = a.len();
        let kernel = half_sweep(func, elem, form, &options, n);
        let args = vec![
            KernelArg::buffer(&a),
            KernelArg::buffer(&b),
            KernelArg::buffer(&c),
            KernelArg::buffer(&vec![0i32; n * options.len()]),
        ];
        let cfg = LaunchConfig::new([(n / 256) as u32, 1, 1], [256, 1, 1]);
        let (mut on_sim, mut on_gpu) = (args.clone(), args);
        kernel.simulate(cfg, &mut on_sim).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
        kernel.run_cuda(&ctx, &toolchain, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
        let (sim, gpu) = (on_sim[3].to_vec::<i32>(), on_gpu[3].to_vec::<i32>());
        let m = options.len();
        let differ: Vec<usize> = (0..sim.len()).filter(|&i| sim[i] != gpu[i]).collect();
        if let Some(&i) = differ.first() {
            let w = i / m;
            failures.push(format!(
                "{}: {} of {} results differ; first with {:?}, a {:#010x}, b {:#010x}, c {:#010x}: simulator \
                 {:#010x}, GPU {:#010x}",
                kernel.name(),
                differ.len(),
                sim.len(),
                options[i % m],
                a[w],
                b[w],
                c[w],
                sim[i],
                gpu[i]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `exp2_packed_f16x2` (`ex2.approx.f16x2`) on every Float16 value: the GPU
/// gives one of the two Float16 values on either side of the exact `2^x`
/// (the correctly rounded value or the other neighbour; `2^x` itself when it
/// is a Float16), so it is within 1 ulp (the PTX ISA's bound, 2^-9.9
/// relative, is about 1 ulp); it gives the correctly rounded value where
/// that is zero or infinite, the special values the PTX ISA lists, the
/// canonical NaN for NaN, and the operand where the predicate is false. The
/// largest error measured on an SM89 GPU is 0.9999 ulp.
#[test]
fn exp2_packed_is_within_one_ulp() {
    let Some((ctx, toolchain)) = device() else { return };
    let case = feature_kernels::exp2_f16x2();
    let args = vec![
        KernelArg::buffer(&vec![0u32; feature_kernels::EXP2_WORDS]),
        KernelArg::buffer(&vec![0u32; feature_kernels::EXP2_WORDS]),
    ];
    let (mut on_sim, mut on_gpu) = (args.clone(), args);
    case.kernel.simulate(case.cfg, &mut on_sim).unwrap();
    case.kernel.run_cuda(&ctx, &toolchain, case.cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{e}"));
    (case.check)(&on_sim);
    let f = &feature_kernels::F16_FORMAT;
    let (out, guarded) = (on_gpu[0].to_vec::<u32>(), on_gpu[1].to_vec::<u32>());
    let mut worst = (0.0f64, 0u16);
    for g in 0..feature_kernels::EXP2_WORDS {
        let word = (2 * g as u32) | ((2 * g as u32 + 1) << 16);
        assert_eq!(guarded[g], if g % 3 != 0 { out[g] } else { word }, "predicated, word {g}");
        for k in 0..2 {
            let (x, r) = ((word >> (16 * k)) as u16, (out[g] >> (16 * k)) as u16);
            let (v, got) = (f.value(x), f.value(r));
            let exact = v.exp2();
            let exact_bits = f.exp2(x);
            if v.is_nan() {
                assert_eq!(r, 0x7fff, "ex2.approx.f16 of NaN {x:#06x}");
                continue;
            }
            // Results that are infinite or zero, exactly or once rounded, must
            // be the correctly rounded value.
            let nearest = f.value(exact_bits);
            if !exact.is_finite() || exact == 0.0 || nearest.is_infinite() || nearest == 0.0 {
                assert_eq!(r, exact_bits, "ex2.approx.f16 of {v:e}: {got:e}");
                continue;
            }
            // Others must be one of the two Float16 values on either side of
            // 2^x (positive Float16 values are in the order of their bits),
            // which is 2^x itself for an integer x. For any other x, f64's
            // 2^x is far from every Float16 value, so it tells the side.
            let neighbours = if v.fract() == 0.0 {
                assert_eq!(nearest, exact, "2^{v} is a Float16");
                [exact_bits; 2]
            } else {
                assert!((nearest - exact).abs() > exact * 2f64.powi(-40), "2^{v} is too close to a Float16");
                if nearest < exact { [exact_bits, exact_bits + 1] } else { [exact_bits - 1, exact_bits] }
            };
            assert!(
                neighbours.contains(&r),
                "ex2.approx.f16 of {v:e} is {got:e}, not one of the Float16 values on either side of {exact:e}"
            );
            let error = (got - exact).abs() / f.ulp(exact);
            if error > worst.0 {
                worst = (error, x);
            }
        }
    }
    eprintln!("ex2.approx.f16x2: largest error {:.4} ulp, at {:#06x}", worst.0, worst.1);
}

/// The Float32 conversions on half a million patterns (every exponent,
/// mantissas at the rounding points and at random): `cvt.rn.satfinite.bf16x2.f32`,
/// `cvt.rn.bf16.f32` and `cvt.rna.tf32.f32` give the simulator's bits on the
/// GPU, NaNs, saturation and ties included.
#[test]
fn float_conversions_match_the_simulator() {
    let Some((ctx, toolchain)) = device() else { return };
    let xs = feature_kernels::cvt_sweep_inputs();
    let n = xs.len() / 256 * 256;
    let kernel = feature_kernels::cvt_sweep(n);
    let args = vec![KernelArg::buffer(&xs[..n]), KernelArg::buffer(&vec![0u32; 3 * n])];
    let cfg = LaunchConfig::new([(n / 256) as u32, 1, 1], [256, 1, 1]);
    let (mut on_sim, mut on_gpu) = (args.clone(), args);
    kernel.simulate(cfg, &mut on_sim).unwrap();
    kernel.run_cuda(&ctx, &toolchain, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{e}"));
    let (sim, gpu) = (on_sim[1].to_vec::<u32>(), on_gpu[1].to_vec::<u32>());
    let names = ["cvt.rn.satfinite.bf16x2.f32", "cvt.rn.bf16.f32", "cvt.rna.tf32.f32"];
    let differ: Vec<usize> = (0..sim.len()).filter(|&i| sim[i] != gpu[i]).collect();
    if let Some(&i) = differ.first() {
        panic!(
            "{} of {} conversions differ; first {} of {:#010x}: simulator {:#010x}, GPU {:#010x}",
            differ.len(),
            sim.len(),
            names[i % 3],
            xs[i / 3],
            sim[i],
            gpu[i]
        );
    }
    // The simulator's results are the plain-Rust references'.
    for (i, &x) in xs[..n].iter().enumerate() {
        let packed = feature_kernels::bf16_satfinite_ref(x) as u32
            | (feature_kernels::bf16_satfinite_ref(x ^ 0x8000_0000) as u32) << 16;
        assert_eq!(sim[3 * i], packed, "satfinite of {x:#010x}");
        assert_eq!(sim[3 * i + 1], feature_kernels::bf16_rn_ref(x) as u32, "bf16 of {x:#010x}");
        assert_eq!(sim[3 * i + 2], feature_kernels::tf32_ref(x), "tf32 of {x:#010x}");
    }
}

/// Run `kernel` on `args` on the simulator and on the GPU, returning both
/// sets of buffers.
fn sim_and_gpu(
    device: &(CudaContext, Toolchain),
    kernel: &Kernel,
    cfg: LaunchConfig,
    args: &[KernelArg],
) -> (Vec<KernelArg>, Vec<KernelArg>) {
    let (ctx, toolchain) = device;
    let mut on_sim = args.to_vec();
    kernel.simulate(cfg, &mut on_sim).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    let mut on_gpu = args.to_vec();
    // Without contraction: callers compare the two bit for bit.
    let options = CompileOptions::new().fp_contract(false).with_env().unwrap_or_else(|e| panic!("{e}"));
    kernel
        .run_cuda_with(ctx, toolchain, &options, cfg, &mut on_gpu)
        .unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    (on_sim, on_gpu)
}

/// `block_smem_reduce.py`'s kernel for every dtype, operation and lane width
/// the SM89 GPU runs (the BFloat16 sum needs sm_90), on its data and on
/// random Float32 data. Each result is one the kernel can give; the
/// warps' `red`s land in an order that varies, so sums of data that round
/// may differ from the simulator's, and everything else must match it.
#[test]
fn block_reduce_matches_the_simulator() {
    use block_reduce::{ReduceDtype, ReduceOp};
    let Some(device) = device() else { return };
    fn one<T: ReduceDtype + std::fmt::Debug>(
        device: &(CudaContext, Toolchain),
        items: usize,
        op: ReduceOp,
        args: Vec<KernelArg>,
        exact: bool,
    ) {
        let kernel = block_reduce::build::<T>(2, items, op);
        let (sim, gpu) = sim_and_gpu(device, &kernel, block_reduce::launch(2), &args);
        block_reduce::check::<T>(&gpu, op);
        if exact {
            assert_all_buffers_equal(kernel.name(), "the example's", &sim, &gpu);
        }
    }
    for op in ReduceOp::ALL {
        for items in [2, 4, 8] {
            one::<i32>(&device, items, op, block_reduce::example_args::<i32>(2, items), true);
            one::<f32>(&device, items, op, block_reduce::example_args::<f32>(2, items), true);
            one::<F16>(&device, items, op, block_reduce::example_args::<F16>(2, items), op != ReduceOp::Add);
            if op != ReduceOp::Add {
                one::<BF16>(&device, items, op, block_reduce::example_args::<BF16>(2, items), true);
            }
        }
        let rows = 2 * block_reduce::ROWS_PER_CTA;
        let src: Vec<f32> = (0..rows * block_reduce::THREADS * 4)
            .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % 1_000_003) as f32 / 65536.0 - 7.5)
            .collect();
        let args = vec![KernelArg::f32(src), KernelArg::f32(vec![0.0; rows])];
        one::<f32>(&device, 4, op, args, op != ReduceOp::Add);
    }
}

/// `redux_sync.py`'s kernel 1 (the eight integer `redux.sync` operations and
/// the shuffle tree) on its data and random data, and kernel 3's shuffle
/// abs-max (its `redux.sync.max.abs.f32` needs sm_100a), on the GPU, held to
/// the simulator and to Rust.
#[test]
fn redux_sync_matches_the_simulator() {
    let Some(device) = device() else { return };
    let mut state = 11u64;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 32) as u32
    };
    let mut ints = vec![redux_sync::int_inputs()];
    ints.extend((0..3).map(|_| (0..32).map(|_| next() as i32).collect::<Vec<i32>>()));
    for vals in ints {
        let args = [KernelArg::buffer(&vals), KernelArg::buffer(&[0i32; 8]), KernelArg::buffer(&[0i32; 8])];
        let (sim, gpu) = sim_and_gpu(&device, &redux_sync::integer_reductions(), redux_sync::launch(), &args);
        assert_all_buffers_equal("integer_reductions", "the example's", &sim, &gpu);
        let want = redux_sync::integer_reference(&vals);
        assert_eq!((gpu[1].to_vec::<i32>(), gpu[2].to_vec::<i32>()), (want.to_vec(), want.to_vec()), "{vals:?}");
    }
    let mut floats = vec![redux_sync::float_inputs()];
    floats.extend((0..3).map(|_| (0..32).map(|_| (next() >> 8) as f32 / (1 << 20) as f32 - 8.0).collect::<Vec<f32>>()));
    for vals in floats {
        let args = [KernelArg::f32(vals.clone()), KernelArg::f32(vec![0.0]), KernelArg::f32(vec![0.0])];
        let (sim, gpu) = sim_and_gpu(&device, &redux_sync::redux_vs_shuffle(false), redux_sync::launch(), &args);
        assert_all_buffers_equal("shuffle_absmax", "the example's", &sim, &gpu);
        assert_eq!(gpu[2].as_f32(), [redux_sync::float_reference(&vals)[2]], "{vals:?}");
    }
}

/// `red.py` on the GPU: the Int32 sum and maximum and the Float32 sum, at
/// its default size and with a partial last block.
#[test]
fn red_matches_the_simulator() {
    let Some(device) = device() else { return };
    for (n, threads) in [(red::DEFAULT_N, red::DEFAULT_THREADS), (1000, 256)] {
        for (kernel, args) in
            [(red::build::<i32>(n, threads), red::s32_args(n)), (red::build::<f32>(n, threads), red::f32_args(n))]
        {
            let (sim, gpu) = sim_and_gpu(&device, &kernel, red::launch(n, threads), &args);
            red::check(&gpu);
            assert_all_buffers_equal(kernel.name(), "the example's", &sim, &gpu);
        }
    }
}

/// The warp-specialised examples' SM89 builds on the GPU (`mbarrier_test_wait`
/// spins): `warp_named_barrier.py` and `pipeline_software_fill.py` at both
/// depths and two tile counts, `cp_async_mbarrier_arrive_cta.py`, and
/// `cooperative_launch.py`'s grid barrier as a cooperative launch
/// (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`; the simulator runs the grid resident), each held to the
/// example's check and to the simulator.
#[test]
fn warp_specialized_examples_match_the_simulator() {
    let Some(device) = device() else { return };
    for stages in [2, 4] {
        for tiles in [8, 16] {
            let kernel = warp_named_barrier::build(stages, Arch::Sm89);
            let (sim, gpu) =
                sim_and_gpu(&device, &kernel, warp_named_barrier::launch(), &warp_named_barrier::args(tiles));
            warp_named_barrier::check(&gpu);
            assert_all_buffers_equal(kernel.name(), "the example's", &sim, &gpu);
            let kernel = pipeline_software_fill::build(stages, Arch::Sm89);
            let args = pipeline_software_fill::args(2 * tiles);
            let (sim, gpu) = sim_and_gpu(&device, &kernel, pipeline_software_fill::launch(), &args);
            pipeline_software_fill::check(&gpu);
            assert_all_buffers_equal(kernel.name(), "the example's", &sim, &gpu);
        }
    }
    let kernel = cp_async_mbarrier_arrive::cta(Arch::Sm89);
    let (sim, gpu) =
        sim_and_gpu(&device, &kernel, cp_async_mbarrier_arrive::cta_launch(), &cp_async_mbarrier_arrive::args(1));
    cp_async_mbarrier_arrive::check(&gpu);
    assert_all_buffers_equal(kernel.name(), "the example's", &sim, &gpu);
    let kernel = cooperative_launch::build();
    let mut sim = cooperative_launch::args();
    kernel.simulate(cooperative_launch::launch(), &mut sim).unwrap_or_else(|e| panic!("{e}"));
    let (ctx, toolchain) = &device;
    let mut gpu = cooperative_launch::args();
    for _ in 0..20 {
        gpu = cooperative_launch::args();
        kernel.run_cuda(ctx, toolchain, cooperative_launch::launch(), &mut gpu).unwrap_or_else(|e| panic!("{e}"));
        cooperative_launch::check(&gpu);
    }
    assert_all_buffers_equal(kernel.name(), "the example's", &sim, &gpu);
}

/// `redux_sync.py`'s kernel 4 with the shuffle-tree abs-max on SM89 (the FP8 `cvt` needs sm_89):
/// within the example's tolerances, and against the simulator. The abs-max and the scale
/// (`amax * (1 / 448)`, no reciprocal instruction) are exact and must match bit for bit. Only
/// `448 / amax` uses `rcp.approx.ftz.f32`, whose result the PTX ISA bounds but does not fix;
/// the GPU's may differ from the simulator's correctly rounded one by an ulp, which moves a
/// quantised value by one FP8 step where `x * 448 / amax` lies on or next to a rounding
/// boundary (2 of 2048 values on SM89).
#[test]
fn redux_sync_absmax_quantize_matches_the_simulator() {
    let Some(device) = device() else { return };
    if !device.0.arch().is_some_and(|a| a.sm() >= 89) {
        return eprintln!("skipping: the FP8 cvt needs sm_89");
    }
    for (rows, seed) in [(redux_sync::DEFAULT_NUM_ROWS, 0), (64, 3)] {
        let inp = redux_sync::quantize_inputs(rows, seed);
        let args = [
            KernelArg::buffer(&inp),
            KernelArg::buffer(&vec![F8E4M3FN(0); rows * WARP_SIZE]),
            KernelArg::f32(vec![0.0; rows]),
        ];
        let kernel = redux_sync::absmax_quantize(rows, false);
        let (sim, gpu) = sim_and_gpu(&device, &kernel, redux_sync::quantize_launch(rows), &args);
        let (gpu_fp8, gpu_scales) = (gpu[1].to_vec::<F8E4M3FN>(), gpu[2].as_f32().to_vec());
        redux_sync::verify_quantized(&inp, &gpu_fp8, &gpu_scales);
        let (sim_fp8, sim_scales) = (sim[1].to_vec::<F8E4M3FN>(), sim[2].as_f32().to_vec());
        for (r, (&g, &s)) in gpu_scales.iter().zip(&sim_scales).enumerate() {
            assert_eq!(g.to_bits(), s.to_bits(), "row {r}: scale {g:e} vs {s:e}");
        }
        let mut differ = 0;
        for (i, (g, s)) in gpu_fp8.iter().zip(&sim_fp8).enumerate() {
            // Same sign, at most one code apart.
            assert!(
                g.0 >> 7 == s.0 >> 7 && (g.0 as i32 - s.0 as i32).abs() <= 1,
                "element {i}: {:#x} vs {:#x}",
                g.0,
                s.0
            );
            differ += usize::from(g.0 != s.0);
        }
        assert!(differ * 50 <= gpu_fp8.len(), "{differ} of {} FP8 values differ", gpu_fp8.len());
    }
}

/// `fp4_x2_tensor.py`'s recast, decoded with integer operations on the GPU: bit for bit the
/// simulator's, on the example's random bytes and on every byte value.
#[test]
fn fp4_x2_tensor_matches_the_simulator() {
    let Some(device) = device() else { return };
    let kernel = fp4_x2_tensor::build(fp4_x2_tensor::Decode::Integer);
    let every: Vec<u8> = (0..=255).collect();
    for bytes in [fp4_x2_tensor::random_bytes(1), every] {
        let (sim, gpu) = sim_and_gpu(&device, &kernel, fp4_x2_tensor::launch(), &fp4_x2_tensor::args(&bytes));
        assert_all_buffers_equal("fp4_x2_tensor", "byte", &sim, &gpu);
        let want: Vec<u16> = fp4_x2_tensor::reference(&bytes).iter().map(|&x| F16::from_f32(x).0).collect();
        assert_eq!(gpu[2].to_vec::<F16>().iter().map(|h| h.0).collect::<Vec<_>>(), want);
    }
}

/// The device mapping on this GPU: an SM89 (Ada) device's target is sm_89,
/// both as its own and as the best one the toolchain and driver handle; the
/// driver's version gives the newest PTX it loads.
#[test]
fn device_targets() {
    let Some((ctx, toolchain)) = device() else { return };
    let (major, minor) = ctx.compute_capability();
    let driver = ctx.driver_version().unwrap_or_else(|e| panic!("{e}"));
    assert!(driver >= 11000, "driver {driver}");
    assert_eq!(ctx.driver_ptx().unwrap(), cute::driver_ptx_ceiling(driver));
    eprintln!(
        "{} (CC {major}.{minor}), driver {}.{} (PTX {}), arch {:?}, best target {:?}",
        ctx.name(),
        driver / 1000,
        driver % 1000 / 10,
        ctx.driver_ptx().unwrap(),
        ctx.arch(),
        ctx.best_target(&toolchain)
    );
    if (major, minor) == (8, 9) {
        assert_eq!((ctx.arch(), ctx.best_target(&toolchain)), (Some(Arch::Sm89), Some(Arch::Sm89)));
    }
    // run_cuda picks the best target.
    let kernel = Kernel::build("device_targets", || {
        let out = param_tensor::<f32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        out.store(t, t.to::<f32>() * 2.0f32);
    });
    let mut args = vec![KernelArg::f32(vec![0.0; 32])];
    kernel.run_cuda(&ctx, &toolchain, LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<f32>()[31], 62.0);
}

/// `launch_async` takes device arguments without the simulator's checks:
/// it still refuses an extent that breaks its `param_extent` divisor,
/// which the kernel was built on.
#[test]
fn launch_async_checks_extent_divisors() {
    let Some((ctx, toolchain)) = device() else { return };
    let kernel = Kernel::builder("extent_divisor").block_dim([32, 1, 1]).build(|| {
        let x = param_ptr::<f32>("x");
        let n = param_extent("n", 4);
        let t = thread_idx().x;
        if_(t.lt(n), || x.store_at(t, 1.0f32));
    });
    let arch = ctx.arch().expect("SM80 or newer");
    let options = CompileOptions::new().gpu_arch(arch);
    let compiled = kernel.compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"));
    let function = ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"));
    let buf = ctx.upload(&KernelArg::f32(vec![0.0; 32])).unwrap();
    let cfg = LaunchConfig::new([1, 1, 1], [32, 1, 1]);
    let bad = [DeviceArg::Buffer(&buf), DeviceArg::Scalar(KernelArg::ScalarI32(6))];
    let e = ctx.launch_async(&kernel, &function, cfg, &bad).expect_err("6 is no multiple of 4");
    assert!(e.to_string().contains("must be a multiple of 4"), "{e}");
    let good = [DeviceArg::Buffer(&buf), DeviceArg::Scalar(KernelArg::ScalarI32(8))];
    ctx.launch_async(&kernel, &function, cfg, &good).unwrap_or_else(|e| panic!("{e}"));
    ctx.synchronize().unwrap_or_else(|e| panic!("{e}"));
}

/// The kernels declared through the kernel argument protocol
/// (`feature_kernels::param_cases`: `TensorSpec` tensors, layout, `Option`,
/// tuple, `FastDivmod`, marked host tensors, a shared `SymInt`, arrays,
/// struct and native-struct arguments), each compiled once and launched at
/// several problem sizes from host values: the GPU computes the plain-Rust
/// results.
#[test]
fn parameter_protocol_kernels_match_rust() {
    let Some((ctx, toolchain)) = device() else { return };
    let arch = ctx.arch().expect("SM80 or newer");
    let options = CompileOptions::new().gpu_arch(arch).with_env().unwrap_or_else(|e| panic!("{e}"));
    for case in feature_kernels::params::param_cases() {
        let compiled: Vec<_> = case
            .kernels
            .iter()
            .map(|k| (k.name().to_string(), k.compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"))))
            .collect();
        let functions: Vec<_> =
            compiled.iter().map(|(n, c)| (n.clone(), ctx.load(c).unwrap_or_else(|e| panic!("{n}: {e}")))).collect();
        let run = |k: &Kernel, cfg: LaunchConfig, args: &mut Vec<KernelArg>| {
            let f = &functions.iter().find(|(n, _)| n == k.name()).expect("a kernel of the case").1;
            ctx.launch(k, f, cfg, args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
        };
        (case.check)(&run);
    }
}

/// Device `printf` output, which the driver writes to the process's stdout,
/// matches the simulator's lines (sorted: thread order varies). The kernels
/// run in a child process, which the test reads.
#[test]
fn printf_matches_the_simulator() {
    use support::child;
    use support::printf_cases::{GLOBAL_POINTER_LINE, everything, hello};
    if child::is_child() {
        let Some((ctx, toolchain)) = device() else { return };
        child::begin();
        for case in [everything(), hello()] {
            let mut args = (case.args)();
            case.kernel.run_cuda(&ctx, &toolchain, case.cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
            if case.n > 0 {
                assert_eq!(args[2].to_f32(), (0..case.n).map(|i| 3.0 * i as f32).collect::<Vec<_>>());
            }
        }
        child::end();
        return;
    }
    if device().is_none() {
        return;
    }
    let out = child::run_child("printf_matches_the_simulator");
    assert!(out.success, "the child failed:\n{}\n{}", out.stdout, out.stderr);
    let mut want: Vec<String> = Vec::new();
    for case in [everything(), hello()] {
        let mut args = (case.args)();
        let report = case.kernel.simulate(case.cfg, &mut args).unwrap();
        want.extend(report.printed.concat().lines().map(str::to_string));
    }
    const SHARED_POINTER_LINE: &str = "shared pointer ";
    let global = |l: &String| l.starts_with(GLOBAL_POINTER_LINE);
    let gpu_global: Vec<&String> = out.lines.iter().filter(|l| global(l)).collect();
    assert_eq!(gpu_global.len(), 1, "{:?}", out.lines);
    let addr = &gpu_global[0][GLOBAL_POINTER_LINE.len()..];
    assert!(
        addr.strip_prefix("0x")
            .is_some_and(|h| !h.is_empty() && h.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())),
        "{addr}"
    );
    // From sm_90 the hardware reserves the first 1 KiB of each block's shared memory, so the
    // GPU's shared addresses are the simulator's plus 0x400 (printf shows address 0 as "(nil)").
    let reserved = device().map_or(0, |(ctx, _)| if ctx.compute_capability().0 >= 9 { 0x400 } else { 0 });
    let unreserve = |l: &String| -> String {
        let Some(rest) = l.strip_prefix(SHARED_POINTER_LINE) else { return l.clone() };
        let words: Vec<String> = rest
            .split(' ')
            .map(|w| match w.strip_prefix("0x").and_then(|h| u64::from_str_radix(h, 16).ok()) {
                Some(a) if a >= reserved => match a - reserved {
                    0 => "(nil)".to_string(),
                    a => format!("{a:#x}"),
                },
                _ => w.to_string(),
            })
            .collect();
        format!("{SHARED_POINTER_LINE}{}", words.join(" "))
    };
    let gpu_lines: Vec<String> = out.lines.iter().map(unreserve).collect();
    let rest = |v: &[String]| child::sorted(&v.iter().filter(|l| !global(l)).cloned().collect::<Vec<_>>());
    assert_eq!(rest(&gpu_lines), rest(&want), "GPU:\n{}", out.stdout);
}

/// The two kernels of CUTLASS's
/// `examples/python/CuTeDSL/experimental/compiler_diagnostic/ptxas_spill_cases.py`
/// (BSD-3-Clause; see THIRD-PARTY-NOTICES.md): 64 values per thread held
/// across a barrier and stored reversed, which spills under
/// `--maxrregcount=16`; and a 256-entry per-thread array indexed by a value
/// loaded from memory, which lives in local memory.
fn spill_kernels() -> (Kernel, Kernel, u32) {
    const VALUES: usize = 64;
    const LOCAL: usize = 256;
    let line = line!() + 3;
    // No declared block size: a launch bound (.reqntid) would let ptxas
    // ignore --maxrregcount.
    let spill = Kernel::build("global_reverse_register_spill", || {
        let src = param_ptr::<i32>("src");
        let dst = param_ptr::<i32>("dst");
        let base = thread_idx().x * VALUES as i32;
        let values: Vec<Int32> = (0..VALUES).map(|i| src.load_at(base + i as i32)).collect();
        sync_threads();
        for i in 0..VALUES {
            dst.store_at(base + i as i32, values[VALUES - 1 - i]);
        }
    });
    let local = Kernel::builder("dynamic_index_local_memory").block_dim([32, 1, 1]).build(|| {
        let src = param_ptr::<i32>("src");
        let dst = param_ptr::<i32>("dst");
        let t = thread_idx().x;
        let base = t * LOCAL as i32;
        let regs = make_rmem_tensor::<i32>(Layout::col_major(LOCAL));
        for i in 0..LOCAL {
            regs.store(i, src.load_at(base + i as i32));
        }
        sync_threads();
        let idx = src.load_at(t) & (LOCAL as i32 - 1);
        dst.store_at(t, regs.load(idx));
    });
    (spill, local, line)
}

/// Spills are a warning at the kernel's `Kernel::build` line, local memory
/// a remark; warnings-as-errors fails the compile; the JIT's verbose log
/// gives the same numbers as ptxas -v, and the driver the same registers.
#[test]
fn ptxas_diagnostics() {
    let Some((ctx, tc)) = device() else { return };
    if tc.cuda_tools().ptxas.is_none() {
        eprintln!("skipping: no ptxas");
        return;
    }
    let (spill, local, line) = spill_kernels();
    let arch = ctx.best_target(&tc).unwrap();
    let opts = CompileOptions::new().gpu_arch(arch).ptxas_options("--maxrregcount=16");
    let c = spill.compile_with(&tc, &opts.clone().warnings(&["ptx"])).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("{}", c.ptxas_log().unwrap_or_default());
    let usage = c.resource_usage().unwrap().clone();
    assert!(usage.spills(), "{usage:?}");
    let w = &c.diagnostics()[0];
    assert_eq!((w.severity, w.category.as_str()), (Severity::Warning, "ptx"), "{w}");
    let at = w.loc.as_ref().unwrap();
    assert_eq!((at.file.as_str(), at.line), (file!(), line), "{w}");
    // Without warnings asked for, none are reported.
    assert!(spill.compile_with(&tc, &opts).unwrap().diagnostics().is_empty());
    let e = spill.compile_with(&tc, &opts.clone().warnings(&[]).warnings_as_errors(true)).unwrap_err();
    assert!(matches!(&e, CompileError::Warnings(ws) if ws.len() == 1), "{e}");
    assert_eq!(e.loc().map(|l| l.line), Some(line));
    // Ignored, the warning goes and the compile succeeds.
    let quiet = opts.clone().warnings(&[]).warnings_as_errors(true).warnings_ignore(true);
    assert!(spill.compile_with(&tc, &quiet).unwrap().diagnostics().is_empty());
    // The JIT: the same numbers from its verbose log, read in a child
    // process with the driver's compute cache off (a cached module has no
    // log).
    let out = support::child::run_child_with("jit_log_resource_usage", &[("CUDA_CACHE_DISABLE", "1")]);
    assert!(out.success, "{}\n{}", out.stdout, out.stderr);
    let log = out.lines.join("\n");
    eprintln!("JIT log:\n{log}");
    let from_jit = ResourceUsage::from_jit_log(&log, spill.name()).expect("a resource report in the JIT log");
    assert_eq!(
        (from_jit.registers, from_jit.spill_store_bytes, from_jit.spill_load_bytes),
        (usage.registers, usage.spill_store_bytes, usage.spill_load_bytes)
    );
    let jc = spill
        .compile_with(&tc, &CompileOptions::new().gpu_arch(arch).ptxas_options("--maxrregcount=16").use_ptxas(false))
        .unwrap();
    assert!(jc.resource_usage().is_none());
    let driver = ctx.load(&jc).unwrap().resource_usage().unwrap();
    assert_eq!(driver.registers, usage.registers);
    assert_eq!(driver.stack_bytes, usage.stack_bytes);
    // Local memory: a remark.
    let c = local.compile_with(&tc, &CompileOptions::new().gpu_arch(arch).use_ptxas(true).remarks(&["ptx"])).unwrap();
    eprintln!("{}", c.ptxas_log().unwrap_or_default());
    assert!(c.resource_usage().unwrap().stack_bytes.unwrap() >= 1024, "{:?}", c.resource_usage());
    assert!(
        c.diagnostics().iter().any(|d| d.severity == Severity::Remark && d.message.contains("bytes of local memory")),
        "{:?}",
        c.diagnostics()
    );
    // Both still compute the right thing.
    let n = 32 * 256;
    let src: Vec<i32> = (0..n as i32).collect();
    let mut args = [KernelArg::i32(src.clone()), KernelArg::i32(vec![0; n])];
    spill.run_cuda_with(&ctx, &tc, &opts, LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    for t in 0..32 {
        for i in 0..64 {
            assert_eq!(args[1].as_i32()[t * 64 + i], src[t * 64 + 63 - i]);
        }
    }
}

/// In a child process (see `ptxas_diagnostics`): the verbose JIT log of the
/// spilling kernel, between the child markers.
#[test]
fn jit_log_resource_usage() {
    if !support::child::is_child() {
        return;
    }
    let Some((ctx, tc)) = device() else { return };
    let (spill, _, _) = spill_kernels();
    let arch = ctx.best_target(&tc).unwrap();
    let jit = CompileOptions::new().gpu_arch(arch).ptxas_options("--maxrregcount=16 -v").use_ptxas(false);
    let f = ctx.load(&spill.compile_with(&tc, &jit).unwrap()).unwrap();
    support::child::begin();
    println!("{}", f.jit_log());
    support::child::end();
}

/// CUTLASS's `experimental/compiler_diagnostic/cuda_launch_error.py`: a
/// block of 4096 threads and 1 GiB of dynamic shared memory. The checked
/// launch refuses both before the driver; launched on the device, the
/// driver's error comes with its name and description.
#[test]
fn launch_errors_are_clear() {
    let Some((ctx, tc)) = device() else { return };
    let k = Kernel::build("copy_one", || {
        let src = param_ptr::<f32>("src");
        let dst = param_ptr::<f32>("dst");
        dst.store_at(0, src.load_at(0));
    });
    let args = || [KernelArg::f32(vec![1.0]), KernelArg::f32(vec![0.0])];
    let threads = LaunchConfig::new([1, 1, 1], [4096, 1, 1]);
    let e = k.run_cuda(&ctx, &tc, threads, &mut args()).unwrap_err().to_string();
    eprintln!("{e}");
    assert!(e.contains("4096"), "{e}");
    let smem = LaunchConfig::new([1, 1, 1], [1, 1, 1]).with_dynamic_smem(1 << 30);
    let e = k.run_cuda(&ctx, &tc, smem, &mut args()).unwrap_err().to_string();
    eprintln!("{e}");
    assert!(e.contains("shared memory"), "{e}");
    let compiled = k.compile_with(&tc, &CompileOptions::new().gpu_arch(ctx.best_target(&tc).unwrap())).unwrap();
    let f = ctx.load(&compiled).unwrap();
    let (a, b) = (ctx.upload(&args()[0]).unwrap(), ctx.upload(&args()[1]).unwrap());
    let e = ctx.launch_on_device(&k, &f, threads, &[DeviceArg::Buffer(&a), DeviceArg::Buffer(&b)]).unwrap_err();
    eprintln!("{e}");
    assert_eq!(e.code, Some(1));
    assert!(e.message.contains("CUDA_ERROR_INVALID_VALUE: invalid argument"), "{e}");
}

/// RuntimeAssertion on the GPU: the flags come back and verify reports the
/// failing store.
#[test]
fn runtime_assertions_on_the_gpu() {
    use cute::testing::{DeviceAssertions, RuntimeAssertion};
    let Some((ctx, tc)) = device() else { return };
    let ra = RuntimeAssertion::new(2).init_value(true);
    let spec = ra.spec();
    let k = Kernel::builder("gpu_runtime_asserts").block_dim([1, 1, 1]).build(|| {
        let x = param_scalar::<i32>("x");
        let checks = DeviceAssertions::param("checks", spec);
        checks.store(0, x.gt(0), "x is positive");
        checks.store(1, x.ne(7), "x is not 7");
    });
    for (x, want) in [(3, None), (7, Some(1))] {
        let mut ra = ra.clone();
        let mut args = vec![KernelArg::scalar(x), ra.kernel_arg().unwrap()];
        k.run_cuda(&ctx, &tc, LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
        ra.read_back(&args[1]);
        assert_eq!(ra.verify(&k).err().map(|f| f.index), want, "x = {x}: {:?}", ra.flags());
    }
}

/// `async_pipeline.ipynb`'s kernels built for SM89 on the GPU (20 launches
/// each): the notebook's outputs, bit-identical to the simulator.
#[test]
fn async_pipeline_example_matches_the_simulator() {
    let Some(device) = device() else { return };
    for variant in async_pipeline::Variant::ALL {
        let kernel = async_pipeline::build(variant, Arch::Sm89);
        for _ in 0..20 {
            let (sim, gpu) = sim_and_gpu(&device, &kernel, async_pipeline::launch(), &async_pipeline::args(variant));
            async_pipeline::check(variant, &gpu);
            assert_all_buffers_equal(kernel.name(), "the notebook's", &sim, &gpu);
        }
    }
}

/// `CopyG2ROp` and `CopyR2GOp` atoms with an L2 cache policy
/// (`ld.global.L2::cache_hint`, `st.global.L2::cache_hint`; sm_80, PTX 7.4) give the
/// simulator's bits on the GPU, which are the inputs.
#[test]
fn copy_atom_cache_hints_match_the_simulator() {
    let Some((ctx, toolchain)) = device() else { return };
    let case = feature_kernels::copy_atom_cache_hints();
    for (small, label) in [(true, "small-integer"), (false, "random")] {
        let input = feature_kernels::inputs_with_len(&case.kernel, small, 256);
        let mut on_sim = input.clone();
        case.kernel.simulate(case.cfg, &mut on_sim).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let mut on_gpu = input;
        case.kernel.run_cuda(&ctx, &toolchain, case.cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&on_gpu);
        assert_all_buffers_equal(case.name, label, &on_sim, &on_gpu);
    }
    let ptx = case.kernel.compile_ptx(&toolchain, Arch::Sm89).unwrap();
    assert!(ptx.contains("ld.global.L2::cache_hint") && ptx.contains("st.global.L2::cache_hint"), "{ptx}");
}

// The single-CTA form of the cluster RMSNorm port.

/// One unit in the last place of `v` in a type with `mant` mantissa bits
/// and smallest normal exponent `min_exp` (the subnormal spacing below it).
fn rmsnorm_ulp(v: f32, mant: i32, min_exp: i32) -> f64 {
    let e = if v == 0.0 { min_exp as f64 } else { (v.abs() as f64).log2().floor().max(min_exp as f64) };
    2f64.powf(e - mant as f64)
}

/// `rmsnorm.py`'s kernel as SM89 runs it (cluster_n = 1, one CTA
/// per row tile, including N = 20000, which SM90 splits over two CTAs), on
/// Float16, BFloat16 and Float32 at several M from one built kernel. With
/// the precise `rsqrt` every operation is IEEE and the reductions' order is
/// fixed, so the GPU matches the simulator bit for bit. With the example's
/// `fastmath` `rsqrt` (`rsqrt.approx.ftz.f32`, within 2^-22.9 relative,
/// which the simulator rounds correctly) each output is within one unit in
/// the last place of the simulator's for Float16 and BFloat16 (the relative
/// difference only moves a rounding), and four for Float32 (`rstd`, then
/// two rounded products).
#[test]
fn rmsnorm_single_cta_matches_the_simulator() {
    use rmsnorm_cluster::{RMSNormConfig, check, inputs, output_index, rmsnorm};
    let Some(device) = device() else { return };
    fn one<T: Numeric>(
        device: &(CudaContext, Toolchain),
        n: usize,
        to: impl Fn(f32) -> T + Copy,
        from: impl Fn(T) -> f32 + Copy,
        (tol, mant, min_exp, ulps): (f64, i32, i32, f64),
    ) {
        for fastmath in [false, true] {
            let cfg = RMSNormConfig { fastmath, ..RMSNormConfig::new(T::DTYPE, n, true, 89) };
            assert_eq!(cfg.cluster_n, 1);
            let kernel = rmsnorm::<T>(cfg);
            for m in [1, cfg.rows_per_block + 1, 2 * cfg.rows_per_block + 3] {
                let (sim, gpu) = sim_and_gpu(device, &kernel, cfg.launch(m), &inputs(&cfg, m, to));
                let name = format!("{} (M = {m})", kernel.name());
                check(&name, &cfg, &gpu, from, tol);
                if !fastmath {
                    assert_all_buffers_equal(&name, "random", &sim, &gpu);
                    continue;
                }
                let o = output_index(&cfg);
                let (s, g) = (sim[o].to_f32(), gpu[o].to_f32());
                for (i, (&s, &g)) in s.iter().zip(&g).enumerate() {
                    let bound = ulps * rmsnorm_ulp(s.abs().max(g.abs()), mant, min_exp);
                    assert!(((s - g) as f64).abs() <= bound, "{name}: element {i}: simulator {s:e}, GPU {g:e}");
                }
            }
        }
    }
    for n in [1000, 4096, 20000] {
        one::<F16>(&device, n, F16::from_f32, F16::to_f32, (4e-3, 10, -14, 1.0));
        one::<BF16>(&device, n, BF16::from_f32, BF16::to_f32, (2e-2, 7, -126, 1.0));
        one::<f32>(&device, n, |x| x, |x| x, (1e-4, 23, -126, 4.0));
    }
}

/// The cute_ext foundation ports on the GPU: the memcpy bit-exact at
/// the ragged default shape and others, from one compiled kernel; the GEMM bit-exact on small
/// integers and within the Float32 summation bound on random data.
#[test]
fn cute_ext_ports_on_the_gpu() {
    let Some((ctx, toolchain)) = device() else { return };
    let memcpy = cute_ext_memcpy::kernel();
    for mnl in [cute_ext_memcpy::DEFAULT_MNL, (256, 128, 2), (40, 65, 3)] {
        let mut host = cute_ext_memcpy::problem(mnl);
        let a = host.0.to_vec::<F16>();
        memcpy
            .run_cuda(&ctx, &toolchain, cute_ext_memcpy::launch(mnl), &mut host)
            .unwrap_or_else(|e| panic!("cute_ext_memcpy {mnl:?}: {e}"));
        let d = host.1.to_vec::<F16>();
        let want = cute_ext_memcpy::reference(&a);
        assert!(d.iter().zip(&want).all(|(x, y)| x.0 == y.0), "cute_ext_memcpy {mnl:?}");
    }
    for mnkl in [cute_ext_sgemm::DEFAULT_MNKL, (128, 64, 96, 3)] {
        let kernel = cute_ext_sgemm::build(mnkl);
        let (m, n, _, l) = mnkl;
        for small in [true, false] {
            let (a, b) = cute_ext_sgemm::inputs(mnkl, small, 7);
            let mut args = cute_ext_sgemm::args(&a, &b, m * n * l);
            kernel
                .run_cuda(&ctx, &toolchain, cute_ext_sgemm::launch(mnkl), &mut args)
                .unwrap_or_else(|e| panic!("cute_ext_sgemm {mnkl:?}: {e}"));
            let d = args[2].as_f32();
            if small {
                let want = cute_ext_sgemm::reference(mnkl, &a, &b);
                assert!(d.iter().zip(&want).all(|(&x, &y)| x as f64 == y), "cute_ext_sgemm {mnkl:?} small integers");
            } else {
                cute_ext_sgemm::within_bound(mnkl, &a, &b, d)
                    .unwrap_or_else(|e| panic!("cute_ext_sgemm {mnkl:?}: {e}"));
            }
        }
    }
}
