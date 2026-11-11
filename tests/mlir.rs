//! Every example kernel lowers to MLIR that a stock `mlir-opt` accepts and
//! compiles to PTX containing the instructions the kernel is built around.
//! Skipped (with a note) when no `mlir-opt` is installed.

#[allow(dead_code)]
#[path = "../examples/elementwise_add.rs"]
mod elementwise_add;
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

use cute::prelude::*;
use math_cases::Ty;

fn toolchain() -> Option<Toolchain> {
    let tc = Toolchain::discover();
    if tc.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    tc
}

fn ptx(kernel: &Kernel, tc: &Toolchain) -> String {
    tc.verify(&kernel.to_mlir()).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    kernel.compile_ptx(tc, Arch::Sm80).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()))
}

fn count(ptx: &str, pattern: &str) -> usize {
    ptx.lines().filter(|l| l.contains(pattern)).count()
}

#[test]
fn elementwise_kernels() {
    let Some(tc) = toolchain() else { return };
    let naive = ptx(&elementwise_add::naive(), &tc);
    assert!(naive.contains(".entry elementwise_add_naive"));
    let tiled = ptx(&elementwise_add::tiled(), &tc);
    assert_eq!(count(&tiled, "ld.global.v4"), 8, "{tiled}");
    assert_eq!(count(&tiled, "st.global.v4"), 4, "{tiled}");
}

#[test]
fn sgemm_kernel() {
    let Some(tc) = toolchain() else { return };
    let p = ptx(&sgemm::build(256, 256, 64), &tc);
    assert_eq!(count(&p, "fma.rn.f32"), 8 * 64, "one FMA per (k, m, n) of the 8 x 8 x 8 thread block");
    assert!(count(&p, "bar.sync") >= 2);
}

#[test]
fn tensorop_gemm_kernel() {
    let Some(tc) = toolchain() else { return };
    let p = ptx(&tensorop_gemm::build(tensorop_gemm::GemmConfig::new(256, 256, 256)), &tc);
    assert_eq!(count(&p, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"), 64);
    assert_eq!(count(&p, "ldmatrix.sync.aligned.m8n8.x4.shared.b16"), 8);
    assert_eq!(count(&p, "ldmatrix.sync.aligned.m8n8.x2.shared.b16"), 16);
    assert!(count(&p, "cp.async.cg.shared.global") > 0);
    assert!(p.lines().any(|l| l.contains("cp.async.wait_group") && l.trim_end().ends_with("1;")));
    assert!(p.contains(".shared .align 128"));
}

#[test]
fn tensorop_gemm_bf16_kernel() {
    let Some(tc) = toolchain() else { return };
    let cfg = tensorop_gemm::GemmConfig { ab: DType::BF16, ..tensorop_gemm::GemmConfig::new(256, 256, 256) };
    let p = ptx(&tensorop_gemm::build(cfg), &tc);
    assert_eq!(count(&p, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"), 64);
    assert_eq!(count(&p, "ldmatrix.sync.aligned.m8n8.x4.shared.b16"), 8);
    assert_eq!(count(&p, "ldmatrix.sync.aligned.m8n8.x2.shared.b16"), 16);
}

/// Every `mma.sync` kind compiles to its PTX instruction.
#[test]
fn every_mma_kind_compiles() {
    let Some(tc) = toolchain() else { return };
    let kinds = [
        MmaKind::M16N8K16F32F16,
        MmaKind::M16N8K8F32F16,
        MmaKind::M16N8K16F16F16,
        MmaKind::M16N8K8F16F16,
        MmaKind::M16N8K16F32BF16,
        MmaKind::M16N8K8F32BF16,
        MmaKind::M16N8K4Tf32,
        MmaKind::M16N8K8Tf32,
        MmaKind::M8N8K4F64,
    ];
    for kind in kinds {
        fn build<TA: Numeric, TC: Numeric>(kind: MmaKind) -> Kernel {
            Kernel::builder("one_mma").block_dim([32, 1, 1]).build(|| {
                let (m, n, k) = kind.shape_mnk();
                let a = param_tensor::<TA>("a", Layout::col_major((m, k)));
                let b = param_tensor::<TA>("b", Layout::col_major((n, k)));
                let c = param_tensor::<TC>("c", Layout::col_major((m, n)));
                let mma = make_tiled_mma(kind, Layout::col_major((1, 1, 1)));
                let thr = mma.get_slice(lane_idx());
                let (fa, fb, acc) = (
                    thr.partition_fragment_a::<TA, _>(&a),
                    thr.partition_fragment_b::<TA, _>(&b),
                    thr.partition_fragment_c::<TC, _>(&c),
                );
                autovec_copy(&thr.partition_a(&a), &fa);
                autovec_copy(&thr.partition_b(&b), &fb);
                autovec_copy(&thr.partition_c(&c), &acc);
                gemm(&mma, &acc, &fa, &fb, &acc);
                autovec_copy(&acc, &thr.partition_c(&c));
            })
        }
        let kernel = match kind.dtypes() {
            (DType::F16, _, DType::F32) => build::<F16, f32>(kind),
            (DType::F16, _, _) => build::<F16, F16>(kind),
            (DType::BF16, ..) => build::<BF16, f32>(kind),
            (DType::F32, ..) => build::<f32, f32>(kind),
            _ => build::<f64, f64>(kind),
        };
        let p = ptx(&kernel, &tc);
        assert_eq!(count(&p, &kind.ptx()), 1, "{kind:?}:\n{p}");
    }
}

#[test]
fn single_warp_pipeline_kernel() {
    let Some(tc) = toolchain() else { return };
    let p = ptx(&single_warp_pipeline::build(), &tc);
    assert_eq!(count(&p, "mma.sync.aligned.m16n8k16"), 8);
    assert_eq!(count(&p, "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"), 8);
    assert!(count(&p, "cp.async.ca.shared.global") > 0);
}

#[test]
fn debug_locations_round_trip() {
    let Some(tc) = toolchain() else { return };
    let kernel = elementwise_add::naive();
    let mlir = kernel.to_mlir_with(&MlirOptions { locations: true });
    assert!(mlir.contains("elementwise_add.rs\":"), "{mlir}");
    tc.verify(&mlir).unwrap();
}

#[test]
fn every_architecture_compiles() {
    let Some(tc) = toolchain() else { return };
    let kernel = single_warp_pipeline::build();
    for arch in [
        Arch::Sm80,
        Arch::Sm86,
        Arch::Sm89,
        Arch::Sm90,
        Arch::Sm90a,
        Arch::Sm100,
        Arch::Sm100a,
        Arch::Sm120,
        Arch::Sm120a,
    ] {
        let p = kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        assert!(p.contains(&format!(".target {}", arch.chip())), "{arch:?}");
    }
}

#[test]
fn named_barrier_instructions() {
    let Some(tc) = toolchain() else { return };
    let p = ptx(&feature_kernels::named_barriers().kernel, &tc);
    let compact = |l: &str| l.split_whitespace().collect::<Vec<_>>().join(" ");
    let lines: Vec<String> = p.lines().map(compact).collect();
    for want in [
        "bar.sync 1, 128;",
        "bar.sync 2, 128;",
        "bar.arrive 3, 256;",
        "bar.sync 3, 256;",
        "barrier.arrive 6, 256;",
        "bar.sync 6, 256;",
        "bar.sync 5;",
    ] {
        assert_eq!(lines.iter().filter(|l| l.as_str() == want).count(), 1, "{want}\n{p}");
    }
    // The NamedBarrier whose id is computed at run time takes it in a register.
    let runtime_id = |l: &&String| l.starts_with("bar.sync %r") && l.ends_with(", 128;");
    assert_eq!(lines.iter().filter(runtime_id).count(), 1, "{p}");
}

/// mbarriers compile for sm_80, raising the PTX version to 7.1 for
/// `test_wait.parity`.
#[test]
fn sm80_mbarriers() {
    let Some(tc) = toolchain() else { return };
    let p = ptx(&feature_kernels::mbarriers_sm80().kernel, &tc);
    assert!(p.contains(".version 7.1"), "{p}");
    assert_eq!(count(&p, "mbarrier.init.shared.b64"), 1, "{p}");
    assert_eq!(count(&p, "mbarrier.arrive.shared.b64"), 2, "{p}");
    assert_eq!(count(&p, "mbarrier.test_wait.parity.shared.b64"), 7, "{p}");
}

/// The SM90 mbarrier operations are refused for older targets, and compile
/// for sm_90 (PTX 8.6 for the relaxed arrival). Each peer-CTA form goes
/// through `mapa` and a `.shared::cluster` operand.
#[test]
fn sm90_mbarriers() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::mbarriers_sm90().kernel;
    for arch in [Arch::Sm80, Arch::Sm89] {
        let e = kernel.compile_ptx(&tc, arch).unwrap_err().to_string();
        assert!(e.contains("needs sm_90 or newer") && e.contains(arch.chip()), "{e}");
        assert!(e.contains("feature_kernels/mod.rs"), "the error should point at the kernel source: {e}");
    }
    tc.verify(&kernel.to_mlir()).unwrap_or_else(|e| panic!("{e}"));
    let p = kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.contains(".version 8.6"), "{p}");
    for (pattern, n) in [
        ("fence.mbarrier_init.release.cluster;", 1),
        ("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [", 1),
        ("mbarrier.arrive.release.cluster.shared::cta.b64 _, [", 1),
        ("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [", 1),
        ("mapa.shared::cluster.u64 remote,", 3),
        ("mbarrier.arrive.shared::cluster.b64 _, [remote];", 1),
        ("mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64 [remote],", 1),
        ("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [remote],", 1),
        ("mbarrier.try_wait.parity.shared::cta.b64", 5),
        ("mbarrier.test_wait.parity.shared.b64", 2),
        ("barrier.cluster.arrive.aligned;", 1),
        ("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes", 3),
    ] {
        assert_eq!(count(&p, pattern), n, "{pattern}\n{p}");
    }
}

/// Assemble `ptx` for `arch` with `ptxas` (`$CUTE_PTXAS`, or `ptxas` on the
/// `PATH`) when one is installed, as on a CUDA machine; skipped otherwise.
fn assemble_with_ptxas(ptx: &str, arch: Arch, name: &str) {
    let ptxas = std::env::var_os("CUTE_PTXAS").unwrap_or_else(|| "ptxas".into());
    let dir = std::env::temp_dir().join(format!("cute-rs-ptxas-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (input, output) = (dir.join("kernel.ptx"), dir.join("kernel.cubin"));
    std::fs::write(&input, ptx).unwrap();
    let result = std::process::Command::new(&ptxas)
        .arg(format!("-arch={}", arch.chip()))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output();
    std::fs::remove_dir_all(&dir).unwrap();
    match result {
        Err(e) => eprintln!("skipping the ptxas check: cannot run {}: {e}", ptxas.to_string_lossy()),
        Ok(out) => assert!(
            out.status.success(),
            "ptxas rejected the {} PTX:\n{}\n{ptx}",
            arch.chip(),
            String::from_utf8_lossy(&out.stderr)
        ),
    }
}

/// Each thread sums a column of `x` in a loop with hints `opts`, over a
/// runtime trip count (the `n` parameter) or a static one.
fn hinted_loop(opts: LoopOptions, static_trips: Option<i32>) -> Kernel {
    Kernel::builder("hinted_loop").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32 * 256));
        let out = param_tensor::<f32>("out", Layout::col_major(32));
        let n = param_scalar::<i32>("n");
        let t = thread_idx().x;
        let upper = static_trips.map_or(n, Int32::from);
        let sum = for_range_carry_with(0, upper, 1, opts, Float32::from(0.0), |i, acc| acc + x.load(i * 32 + t));
        out.store(t, sum);
    })
}

/// `cutlass.range(n, unroll=4)` unrolls the loop body 4 times (plus one copy
/// in the remainder loop a runtime trip count needs; LLVM then marks both
/// loops `nounroll`), and `unroll=1` keeps it rolled with `.pragma
/// "nounroll"`.
#[test]
fn loop_unroll_hints() {
    let Some(tc) = toolchain() else { return };
    let four = ptx(&hinted_loop(LoopOptions::unroll(4), None), &tc);
    assert_eq!(count(&four, "ld.global"), 4 + 1, "{four}");
    let rolled = ptx(&hinted_loop(LoopOptions::unroll(1), None), &tc);
    assert_eq!(count(&rolled, "ld.global"), 1, "{rolled}");
    assert_eq!(count(&rolled, ".pragma \"nounroll\";"), 1, "{rolled}");
}

/// `unroll_full=True` unrolls a loop with a static trip count completely,
/// which LLVM would not do by itself for 64 iterations.
#[test]
fn loop_unroll_full_hint() {
    let Some(tc) = toolchain() else { return };
    let full = ptx(&hinted_loop(LoopOptions::unroll_full(), Some(64)), &tc);
    assert_eq!(count(&full, "ld.global"), 64, "{full}");
    assert_eq!(count(&full, "bra"), 0, "a fully unrolled loop has no branch:\n{full}");
    let plain = ptx(&hinted_loop(LoopOptions::default(), Some(64)), &tc);
    assert!(count(&plain, "ld.global") < 64, "{plain}");
}

/// `cutlass.range(..., at_least_once=True)` lets the compiler drop the check
/// that skips a loop running zero times (the MLIR assumes `lower < upper`):
/// a rolled loop over a runtime trip count keeps only its back edge.
#[test]
fn loop_at_least_once_hint() {
    let Some(tc) = toolchain() else { return };
    let rolled = ptx(&hinted_loop(LoopOptions::unroll(1), None), &tc);
    assert_eq!(count(&rolled, "bra"), 2, "the zero-trip bypass and the back edge:\n{rolled}");
    let once = LoopOptions { unroll: Some(1), at_least_once: true, ..LoopOptions::default() };
    let once = ptx(&hinted_loop(once, None), &tc);
    assert_eq!(count(&once, "bra"), 1, "only the back edge:\n{once}");
    assert_eq!(count(&once, "ld.global"), 1, "{once}");
}

/// `cutlass.range(..., vectorize=True)` needs sm_100, like CuTe DSL's
/// UNSUP_ARCH check, and compiles for sm_100a and sm_120 together with an
/// unroll hint. (LLVM 21 to 23 accept the hint but still emit scalar
/// `add.rn.f32`: NVPTX does not vectorise this loop to packed f32x2.)
#[test]
fn loop_vectorize_hint() {
    let Some(tc) = toolchain() else { return };
    let kernel = hinted_loop(LoopOptions { vectorize: true, unroll: Some(4), ..LoopOptions::default() }, None);
    for arch in [Arch::Sm80, Arch::Sm90a] {
        let e = kernel.compile_ptx(&tc, arch).unwrap_err().to_string();
        let want = format!("needs sm_100 or newer; the kernel is being compiled for {}", arch.chip());
        assert!(e.contains("a vectorized loop") && e.contains(&want), "{e}");
        assert!(e.contains("tests/mlir.rs"), "the error should point at the loop: {e}");
    }
    tc.verify(&kernel.to_mlir()).unwrap_or_else(|e| panic!("{e}"));
    for arch in [Arch::Sm100a, Arch::Sm120] {
        let p = kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        assert!(p.contains(&format!(".target {}", arch.chip())), "{p}");
        assert_eq!(count(&p, "ld.global"), 4 + 1, "the unroll hint still applies:\n{p}");
        assemble_with_ptxas(&p, arch, "vectorize");
    }
}

/// `warpsize` reads PTX's predefined constant `WARP_SZ` through NVVM
/// (`llvm.nvvm.read.ptx.sreg.warpsize`; LLVM knows it is 32), and
/// `make_warp_uniform` is a hint that emits no instruction.
#[test]
fn warp_helpers() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::warp_helpers().kernel;
    assert_eq!(kernel.dump().matches("make_warp_uniform").count(), 3, "{}", kernel.dump());
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("nvvm.read.ptx.sreg.warpsize"), "{mlir}");
    let p = ptx(&kernel, &tc);
    assert_eq!(count(&p, "shfl"), 0, "{p}");
}

/// PTX instructions that compute a hardware approximation.
const APPROXIMATIONS: [&str; 10] =
    ["approx", "ex2.", "lg2.", "sin.", "cos.", "tanh.", "rsqrt.", "div.full", "sqrt.approx", "rcp.approx"];

/// The approximations a function's fast (Float32) form is built from.
fn fast_instructions(name: &str) -> &'static [&'static str] {
    match name {
        "exp" | "exp2" => &["ex2.approx.ftz.f32"],
        "log" | "log2" | "log10" => &["lg2.approx.ftz.f32"],
        "sin" => &["sin.approx.ftz.f32"],
        "cos" => &["cos.approx.ftz.f32"],
        "tan" => &["sin.approx.ftz.f32", "cos.approx.ftz.f32", "div.approx.ftz.f32"],
        "tanh" => &["tanh.approx.f32"],
        "sqrt" => &["sqrt.approx.ftz.f32"],
        "rsqrt" => &["rsqrt.approx.ftz.f32"],
        "rcp" => &["rcp.approx.ftz.f32"],
        "div" => &["div.approx.ftz.f32"],
        "pow" => &["lg2.approx.ftz.f32", "ex2.approx.ftz.f32"],
        _ => &[],
    }
}

/// Every `cute::math` function on every type verifies with `mlir-opt` and
/// compiles for sm_80. In precise mode the PTX has no hardware
/// approximation (and so no libdevice call either: there is none to link);
/// in fast mode the Float32 forms use the approximations their
/// documentation names, with `.ftz`.
#[test]
fn math_functions_compile() {
    let Some(tc) = toolchain() else { return };
    let mut jobs: Vec<(&'static str, Ty, bool, bool)> = Vec::new();
    for ty in math_cases::TYPES {
        for fastmath in [false, true] {
            jobs.extend(math_cases::unary_cases().iter().map(|c| (c.name, ty, fastmath, false)));
            jobs.extend(math_cases::binary_cases().iter().map(|c| (c.name, ty, fastmath, true)));
        }
    }
    let failures: Vec<String> = std::thread::scope(|scope| {
        let chunks: Vec<_> = jobs
            .chunks(jobs.len().div_ceil(8))
            .map(|chunk| {
                let tc = &tc;
                scope.spawn(move || {
                    let mut failures = Vec::new();
                    for &(name, ty, fastmath, binary) in chunk {
                        let kernel = if binary {
                            math_cases::binary_kernel(name, ty, 256, fastmath)
                        } else {
                            math_cases::unary_kernel(name, ty, 256, fastmath)
                        };
                        let what = format!("{name} {} (fastmath {fastmath})", ty.name());
                        if let Err(e) = tc.verify(&kernel.to_mlir()) {
                            failures.push(format!("{what}: {e}"));
                            continue;
                        }
                        let p = match kernel.compile_ptx(tc, Arch::Sm80) {
                            Ok(p) => p,
                            Err(e) => {
                                failures.push(format!("{what}: {e}"));
                                continue;
                            }
                        };
                        let body: Vec<&str> =
                            p.lines().map(str::trim).filter(|l| !l.starts_with("//") && !l.starts_with('.')).collect();
                        let found: Vec<&str> =
                            APPROXIMATIONS.iter().copied().filter(|a| body.iter().any(|l| l.contains(a))).collect();
                        let expected = if fastmath && ty != Ty::F64 { fast_instructions(name) } else { &[] };
                        if expected.is_empty() && !found.is_empty() {
                            failures.push(format!("{what}: approximations {found:?} in\n{p}"));
                        }
                        for e in expected {
                            if !body.iter().any(|l| l.contains(e)) {
                                failures.push(format!("{what}: no {e} in\n{p}"));
                            }
                        }
                        if p.contains("__nv_") {
                            failures.push(format!("{what}: a libdevice call in\n{p}"));
                        }
                    }
                    failures
                })
            })
            .collect();
        chunks.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `approx=`, `ftz=`, `rounding=` and `full=` choose one PTX instruction
/// each, and a precise function compiles for every architecture.
#[test]
fn math_instruction_options() {
    let Some(tc) = toolchain() else { return };
    for (ty, opts, expect) in [
        (
            Ty::F32,
            MathOptions::rounding(RoundingMode::Zero),
            &["add.rz.f32", "mul.rz.f32", "fma.rz.f32", "div.rz.f32", "sqrt.rz.f32", "rcp.rz.f32"][..],
        ),
        (
            Ty::F32,
            MathOptions::rounding(RoundingMode::Up).with_ftz(),
            &[
                "add.rp.ftz.f32",
                "mul.rp.ftz.f32",
                "fma.rp.ftz.f32",
                "div.rp.ftz.f32",
                "sqrt.rp.ftz.f32",
                "rcp.rp.ftz.f32",
            ],
        ),
        (
            Ty::F64,
            MathOptions::rounding(RoundingMode::Down),
            &["add.rm.f64", "mul.rm.f64", "fma.rm.f64", "div.rm.f64", "sqrt.rm.f64", "rcp.rm.f64"],
        ),
    ] {
        let p = ptx(&math_cases::options_kernel(ty, 256, opts), &tc);
        for e in expect {
            assert!(count(&p, e) >= 1, "{opts:?}: no {e} in\n{p}");
        }
    }
    for (ty, ftz, expect) in [
        (
            Ty::F32,
            false,
            &[
                "ex2.approx.f32",
                "lg2.approx.f32",
                "sin.approx.f32",
                "cos.approx.f32",
                "tanh.approx.f32",
                "rsqrt.approx.f32",
                "sqrt.approx.f32",
                "rcp.approx.f32",
                "div.approx.f32",
                "div.full.f32",
            ][..],
        ),
        (
            Ty::F32,
            true,
            &[
                "ex2.approx.ftz.f32",
                "lg2.approx.ftz.f32",
                "sin.approx.ftz.f32",
                "cos.approx.ftz.f32",
                "rsqrt.approx.ftz.f32",
                "sqrt.approx.ftz.f32",
                "rcp.approx.ftz.f32",
                "div.approx.ftz.f32",
                "div.full.ftz.f32",
            ],
        ),
        (Ty::F64, false, &["rsqrt.approx.f64"]),
        (Ty::F64, true, &["rsqrt.approx.ftz.f64", "rcp.approx.ftz.f64"]),
    ] {
        let p = ptx(&math_cases::approx_kernel(ty, 256, ftz), &tc);
        for e in expect {
            assert!(count(&p, e) >= 1, "{} ftz {ftz}: no {e} in\n{p}", ty.name());
        }
    }
    let p = ptx(&math_cases::minmax_kernel(Ty::F32, 256), &tc);
    assert!(count(&p, "min.NaN.f32") >= 1 && count(&p, "max.NaN.f32") >= 1, "{p}");
    let p = ptx(&math_cases::ftz_kernel(256), &tc);
    for e in ["abs.ftz.f32", "min.ftz.f32", "max.ftz.f32", "min.ftz.NaN.f32", "max.ftz.NaN.f32"] {
        assert!(count(&p, e) >= 1, "no {e} in\n{p}");
    }
    for name in ["sin", "pow"] {
        for ty in [Ty::F32, Ty::F64] {
            let kernel = if name == "pow" {
                math_cases::binary_kernel(name, ty, 256, false)
            } else {
                math_cases::unary_kernel(name, ty, 256, false)
            };
            for arch in [Arch::Sm80, Arch::Sm86, Arch::Sm89, Arch::Sm90, Arch::Sm100, Arch::Sm120] {
                kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{name} {} on {arch:?}: {e}", ty.name()));
            }
        }
    }
}

/// `abs`, `-x` and `copysign` on floats are the integer instructions
/// `and.bN` and `xor.bN` on the sign bit, wherever they are: never
/// `abs.f32` or `neg.f32` (whose NaNs differ), whether alone in a kernel,
/// applied to a product, given a constant sign or followed by integer
/// arithmetic. `absi` compiles for every integer type.
#[test]
fn math_sign_operations_are_bit_operations() {
    let Some(tc) = toolchain() else { return };
    for ty in math_cases::TYPES {
        let w = ty.width();
        let mut kernels: Vec<(String, Kernel)> = math_cases::PLAIN_OPS
            .iter()
            .map(|&op| (format!("{op} {}", ty.name()), math_cases::plain_kernel(op, ty, 256)))
            .collect();
        kernels.push((format!("context {}", ty.name()), math_cases::sign_context_kernel(ty, 256)));
        for (what, kernel) in kernels {
            let p = ptx(&kernel, &tc);
            let float_sign_ops: Vec<&str> =
                p.lines().map(str::trim).filter(|l| l.starts_with("abs.") || l.starts_with("neg.")).collect();
            assert!(float_sign_ops.is_empty(), "{what}: {float_sign_ops:?} in\n{p}");
            let and = count(&p, &format!("and.b{w}"));
            let xor = count(&p, &format!("xor.b{w}"));
            let (want_and, want_xor) = match what.split(' ').next().unwrap() {
                "abs" | "absf" => (true, false),
                "neg" | "minus" => (false, true),
                _ => (true, true),
            };
            assert!(and >= want_and as usize && xor >= want_xor as usize, "{what}: and {and}, xor {xor} in\n{p}");
        }
    }
    for kernel in [
        math_cases::absi_kernel::<i8>(256),
        math_cases::absi_kernel::<i16>(256),
        math_cases::absi_kernel::<i32>(256),
        math_cases::absi_kernel::<i64>(256),
        math_cases::absi_kernel::<u32>(256),
    ] {
        ptx(&kernel, &tc);
    }
}
