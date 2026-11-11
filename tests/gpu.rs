//! Every example kernel on a real GPU and on the simulator, comparing every
//! output element. Needs the `cuda` feature, and is skipped (with a note)
//! without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --features cuda --test gpu
//! ```

#[allow(dead_code)]
#[path = "../examples/elementwise_add.rs"]
mod elementwise_add;
#[allow(dead_code)]
#[path = "../examples/runtime_gemm.rs"]
mod runtime_gemm;
#[allow(dead_code)]
#[path = "../examples/sgemm.rs"]
mod sgemm;
#[allow(dead_code)]
#[path = "../examples/single_warp_pipeline.rs"]
mod single_warp_pipeline;
#[allow(dead_code)]
#[path = "../examples/tensorop_gemm.rs"]
mod tensorop_gemm;

mod feature_kernels;
mod math_cases;
mod math_ref;

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
                    _ => KernelArg::f32(values),
                }
            }
            Param::Scalar { .. } => KernelArg::ScalarF32(scalars.next().unwrap()),
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
    kernel.run_cuda(ctx, toolchain, cfg, &mut on_gpu).unwrap_or_else(|e| panic!("{name}: {e}"));
    let output = output_index(kernel);
    for i in (0..input.len()).filter(|&i| i != output) {
        assert!(on_gpu[i] == input[i], "{name}: the GPU changed input parameter {i}");
    }
    (on_sim[output].as_f32().to_vec(), on_gpu[output].as_f32().to_vec())
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

    for (i, ((&s, &g), &m)) in sim.iter().zip(&gpu).zip(magnitude).enumerate() {
        let allowed = (2 * terms) as f64 * 2f64.powi(-23) * m as f64;
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

/// Every type, operation and intrinsic in `tests/feature_kernels`.
#[test]
fn every_feature_matches_the_simulator() {
    let Some((ctx, toolchain)) = device() else { return };
    for case in feature_kernels::all() {
        let datasets: &[(bool, &str)] =
            if case.small_only { &[(true, "small-integer")] } else { &[(true, "small-integer"), (false, "random")] };
        for &(small, label) in datasets {
            let input = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            let mut on_sim = input.clone();
            case.kernel.simulate(case.cfg, &mut on_sim).unwrap_or_else(|e| panic!("{}: {e}", case.name));
            let mut on_gpu = input;
            case.kernel
                .run_cuda(&ctx, &toolchain, case.cfg, &mut on_gpu)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            assert_all_buffers_equal(case.name, label, &on_sim, &on_gpu);
        }
    }
}

/// One compiled kernel run at several problem sizes, on buffers uploaded
/// once and reused.
#[test]
fn runtime_shapes_from_one_kernel() {
    let Some((ctx, toolchain)) = device() else { return };
    let kernel = runtime_gemm::build();
    let arch = ctx.arch().expect("SM80 or newer");
    let ptx = kernel.compile_ptx(&toolchain, arch).unwrap_or_else(|e| panic!("{e}"));
    let function = ctx.load_ptx(&ptx, kernel.name()).unwrap_or_else(|e| panic!("{e}"));
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

#[test]
fn tensorop_gemm_two_stages_unswizzled() {
    check_tensorop_gemm(GemmConfig { stages: 2, swizzle: false, ..GemmConfig::new(128, 128, 64) });
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
