//! Every example kernel lowers to MLIR that a stock `mlir-opt` accepts and
//! compiles to PTX containing the instructions the kernel is built around.
//! Skipped (with a note) when no `mlir-opt` is installed.

#[allow(dead_code)]
#[path = "../examples/ampere_sgemm.rs"]
mod ampere_sgemm;
#[allow(dead_code)]
#[path = "../examples/async_pipeline.rs"]
mod async_pipeline;
#[allow(dead_code)]
#[path = "../examples/block_reduce.rs"]
mod block_reduce;
#[allow(dead_code)]
#[path = "../examples/cooperative_launch.rs"]
mod cooperative_launch;
#[allow(dead_code)]
#[path = "../examples/cp_async_mbarrier_arrive.rs"]
mod cp_async_mbarrier_arrive;
#[allow(dead_code)]
#[path = "../examples/cta_norm.rs"]
mod cta_norm;
#[allow(dead_code)]
#[path = "../examples/elementwise_add.rs"]
mod elementwise_add;
#[allow(dead_code)]
#[path = "../examples/elementwise_apply.rs"]
mod elementwise_apply;
#[allow(dead_code)]
#[path = "../examples/fp4_x2_tensor.rs"]
mod fp4_x2_tensor;
#[allow(dead_code)]
#[path = "../examples/grouped_gemm.rs"]
mod grouped_gemm;
#[allow(dead_code)]
#[path = "../examples/pipeline_software_fill.rs"]
mod pipeline_software_fill;
#[allow(dead_code)]
#[path = "../examples/red.rs"]
mod red;
#[allow(dead_code)]
#[path = "../examples/redux_sync.rs"]
mod redux_sync;
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
#[allow(dead_code)]
#[path = "../examples/warp_named_barrier.rs"]
mod warp_named_barrier;
// The TMA tutorials (m7-port-tma-tutorials).
#[allow(dead_code)]
#[path = "../examples/tma_copy.rs"]
mod tma_copy;
#[allow(dead_code)]
#[path = "../examples/tma_transpose.rs"]
mod tma_transpose;
// The Hopper dense GEMM port (m8-port-hopper).
#[allow(dead_code)]
#[path = "../examples/hopper_gemm.rs"]
mod hopper_gemm;
// The Hopper persistent GEMM port (m9-hopper-persistent-gemm).
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../examples/hopper_persistent_gemm.rs"]
mod hopper_persistent_gemm;

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

/// Whether `tc` can emit `arch` at its own PTX ISA version (LLVM 23 has
/// no sm_107; LLVM 21 knows sm_110 only as sm_101).
fn emittable(tc: &Toolchain, arch: Arch) -> bool {
    tc.targets().is_none_or(|s| s.emit(arch).is_some_and(|(_, min, max)| min <= max))
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

/// The elementwise_apply port moves its row-major tiles in 128-bit
/// vectors: per thread, 16 loads per input and 16 stores, on every type.
#[test]
fn elementwise_apply_kernels() {
    use elementwise_apply::ApplyOp;
    let Some(tc) = toolchain() else { return };
    for op in ApplyOp::F32 {
        let p = ptx(&elementwise_apply::f32_kernel(op).kernel, &tc);
        assert_eq!(count(&p, "ld.global.v4"), 16 * op.inputs(), "{op:?}\n{p}");
        assert_eq!(count(&p, "st.global.v4"), 16, "{op:?}\n{p}");
    }
    for kernel in
        [elementwise_apply::f16_kernel(ApplyOp::Mul).kernel, elementwise_apply::bf16_kernel(ApplyOp::Add).kernel]
    {
        let p = ptx(&kernel, &tc);
        assert_eq!((count(&p, "ld.global.v4"), count(&p, "st.global.v4")), (32, 16), "{p}");
    }
}

/// cta_norm's chunks move with 128-bit global loads and stores (one per
/// chunk and tensor for Float16 and BFloat16, two for Float32), under the
/// chunk's predicate.
#[test]
fn cta_norm_kernels() {
    let Some(tc) = toolchain() else { return };
    for cfg in cta_norm::CONFIGS {
        let chunks = cfg.n.div_ceil(cfg.threads * 8);
        let tensors = if cfg.norm == cta_norm::Norm::Layer { 3 } else { 2 };
        for (p, per) in [
            (ptx(&cta_norm::cta_norm::<F16>(cfg), &tc), 1),
            (ptx(&cta_norm::cta_norm::<BF16>(cfg), &tc), 1),
            (ptx(&cta_norm::cta_norm::<f32>(cfg), &tc), 2),
        ] {
            assert_eq!(count(&p, "ld.global.v4"), chunks * tensors * per, "{cfg:?}\n{p}");
            assert_eq!(count(&p, "st.global.v4"), chunks * per, "{cfg:?}\n{p}");
            assert!(p.contains("shfl.sync.bfly.b32"), "{p}");
        }
    }
}

#[test]
fn sgemm_kernel() {
    let Some(tc) = toolchain() else { return };
    let p = ptx(&sgemm::build(256, 256, 64), &tc);
    // One FMA per (k, m, n) of the 8 x 8 x 8 thread block, plus one per accumulator for
    // the epilogue's `alpha * acc + beta * c`, which contraction (the default) fuses.
    assert_eq!(count(&p, "fma.rn.f32"), 8 * 64 + 64, "one FMA per (k, m, n) and per output");
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
    let mlir = kernel.to_mlir_with(&MlirOptions { locations: true, ..Default::default() });
    assert!(mlir.contains("elementwise_add.rs\":"), "{mlir}");
    tc.verify(&mlir).unwrap();
}

#[test]
fn every_architecture_compiles() {
    let Some(tc) = toolchain() else { return };
    let kernel = single_warp_pipeline::build();
    let support = tc.targets().cloned();
    for arch in Arch::ALL {
        let emit = support.as_ref().map(|s| s.emit(arch));
        match (kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)), emit) {
            (Ok(c), Some(Some((name, min, _)))) => {
                let p = c.ptx();
                assert!(p.contains(&format!(".target {name}\n")), "{arch:?}:\n{p}");
                assert!(p.contains(&format!(".version {}.{}\n", min / 10, min % 10)), "{arch:?}:\n{p}");
                assert_eq!((c.target_name(), c.ptx_version(), c.note().is_some()), (name, min, name != arch.chip()));
            }
            (Ok(c), None) => assert!(c.ptx().contains(&format!(".target {}\n", arch.chip())), "{arch:?}"),
            (Err(e), Some(None)) => {
                let e = e.to_string();
                assert!(e.contains(&format!("cannot emit {}", arch.chip())) && e.contains("no such target"), "{e}");
            }
            (r, emit) => panic!("{arch:?}: {:?} with {emit:?}", r.map(|c| c.target_name().to_string())),
        }
    }
    // Every target compiles to MLIR that this mlir-opt verifies, sm_107
    // (which no LLVM emits yet) included.
    tc.verify(&kernel.to_mlir()).unwrap();
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

/// The cluster barrier lowers to the convergent NVVM operations (never
/// inline PTX), which need sm_90; `.relaxed` needs PTX 8.0.
#[test]
fn cluster_barrier_ptx() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::cluster_barriers_sm90([4, 1, 1], [2, 1, 1]).kernel;
    let e = kernel.compile_ptx(&tc, Arch::Sm89).unwrap_err().to_string();
    assert!(e.contains("needs sm_90 or newer"), "{e}");
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("nvvm.cluster.arrive.relaxed {aligned}") && !mlir.contains("barrier.cluster"), "{mlir}");
    let p = kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.contains(".version 8.0"), "{p}");
    // LLVM may merge the arrivals of the two branches into one.
    assert!((1..=2).contains(&count(&p, "barrier.cluster.arrive;")), "{p}");
    for (pattern, n) in [
        ("barrier.cluster.wait;", 1),
        ("barrier.cluster.arrive.relaxed.aligned;", 1),
        ("barrier.cluster.wait.aligned;", 2),
        ("barrier.cluster.arrive.aligned;", 1),
    ] {
        assert_eq!(count(&p, pattern), n, "{pattern}\n{p}");
    }
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

/// CuTe DSL's integer and bit instructions are each one PTX instruction:
/// pure inline PTX (no `has_side_effects`, so LLVM may CSE and move it),
/// with `lop3`'s table as an immediate, or LLVM's own `mul.hi`. They need
/// nothing beyond sm_80 and PTX 7.0, and compile for every architecture.
#[test]
fn int_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernels = [
        feature_kernels::bit_fields_i32(),
        feature_kernels::bit_fields_u32(),
        feature_kernels::bit_fields_i64(),
        feature_kernels::bit_fields_u64(),
        feature_kernels::logic_and_shifts(),
        feature_kernels::int_multiplies(),
    ];
    let mut all = String::new();
    for case in &kernels {
        let mlir = case.kernel.to_mlir();
        for line in mlir.lines().filter(|l| l.contains("llvm.inline_asm")) {
            assert!(!line.contains("has_side_effects"), "{}: {line}", case.name);
        }
        let p = ptx(&case.kernel, &tc);
        assert!(p.contains(".version 7.0"), "{}: integer instructions need no newer PTX\n{p}", case.name);
        all.push_str(&p);
    }
    let compact = |l: &str| l.split_whitespace().collect::<Vec<_>>().join(" ");
    let lines: Vec<String> = all.lines().map(compact).collect();
    let has = |prefix: &str| lines.iter().any(|l| l.starts_with(prefix));
    for want in [
        "bfe.s32 ",
        "bfe.u32 ",
        "bfe.s64 ",
        "bfe.u64 ",
        "bfi.b32 ",
        "bfi.b64 ",
        "lop3.b32 ",
        "shf.l.wrap.b32 ",
        "shf.r.wrap.b32 ",
        "shf.l.clamp.b32 ",
        "shf.r.clamp.b32 ",
        "prmt.b32 ",
        "prmt.b32.f4e ",
        "prmt.b32.b4e ",
        "prmt.b32.rc8 ",
        "prmt.b32.ecl ",
        "prmt.b32.ecr ",
        "prmt.b32.rc16 ",
        "mul.hi.s32 ",
        "mul.hi.u32 ",
        "mul.hi.s64 ",
        "mul.hi.u64 ",
        "mul.wide.s16 ",
        "mul.wide.u16 ",
        "mul.wide.s32 ",
        "mul.wide.u32 ",
        "mul24.lo.s32 ",
        "mul24.lo.u32 ",
        "mul24.hi.s32 ",
        "mul24.hi.u32 ",
        "mad24.lo.s32 ",
        "mad24.lo.u32 ",
        "mad24.hi.s32 ",
        "mad24.hi.u32 ",
        "add.sat.s32 ",
        "sub.sat.s32 ",
    ] {
        assert!(has(want), "no {want} in\n{all}");
    }
    // Every table of the lop3 kernel, as the instruction's immediate.
    let lop3: Vec<&String> = lines.iter().filter(|l| l.starts_with("lop3.b32 ")).collect();
    for lut in [0x00, 0xff, 0x80, 0xfe, 0x40, 0x1a, 0x96, 0xe8, 0xca, 0xd8] {
        let immediate = format!(", {lut:#04x};");
        assert_eq!(lop3.iter().filter(|l| l.ends_with(&immediate)).count(), 1, "{immediate}\n{lop3:?}");
    }
    let kernel = feature_kernels::logic_and_shifts().kernel;
    for arch in [Arch::Sm80, Arch::Sm86, Arch::Sm89, Arch::Sm90, Arch::Sm90a, Arch::Sm100a, Arch::Sm120a] {
        assert_eq!(cute::validate::check_target(kernel.ir(), arch).unwrap(), arch.min_ptx(), "{arch:?}");
        kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
    }
}

/// `bfe` and `bfi` take their start and length through the arithmetic that
/// keeps them where the GPU follows the PTX ISA (the GPU test checks the
/// results): the low byte for the 64-bit forms, a `bfe.s32` start clamped to
/// 31 and a `bfi.b32` length cut to the room below the msb, and `bfe.u32`
/// as it is.
#[test]
fn bit_field_operands_stay_where_the_gpu_follows_the_isa() {
    let kernel = Kernel::builder("bit_field_operands").block_dim([32, 1, 1]).build(|| {
        let p = param_tensor::<i32>("p", Layout::col_major(32));
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let y = param_tensor::<u64>("y", Layout::col_major(32));
        let t = thread_idx().x;
        let (start, length, a, b) = (p.load(t), p.load((t + 1) % 32), x.load(t), y.load(t));
        p.store(t, bfe(a.bitcast::<i32>(), start, length));
        x.store(t, bfe(a, start, length) ^ bfi(a, a, start, length));
        y.store(
            t,
            bfe(b, start, length) ^ bfi(b, b, start, length) ^ bfe(b.bitcast::<i64>(), start, length).bitcast::<u64>(),
        );
    });
    let mlir = kernel.to_mlir();
    // Each SSA name's defining operation.
    let defs: std::collections::HashMap<&str, &str> = mlir.lines().filter_map(|l| l.trim().split_once(" = ")).collect();
    let def = |name: &str| *defs.get(name).unwrap_or_else(|| panic!("{name} is not defined in\n{mlir}"));
    // `arith.<op> <a>, <b> : i32` with constant `b` (by value).
    let op = |name: &str, want_op: &str, constant: Option<u32>| -> String {
        let d = def(name);
        let rest = d.strip_prefix(&format!("arith.{want_op} ")).unwrap_or_else(|| panic!("{name} = {d}"));
        let (lhs, rhs) = rest.trim_end_matches(" : i32").split_once(", ").unwrap();
        if let Some(c) = constant {
            assert_eq!(def(rhs), format!("arith.constant {c} : i32"), "{name} = {d}");
        }
        lhs.to_string()
    };
    let mut seen = Vec::new();
    for line in mlir.lines().filter(|l| l.contains("llvm.inline_asm")) {
        let inst = line.split('"').nth(1).unwrap().split(' ').next().unwrap().to_string();
        let operands: Vec<&str> =
            line.split('"').nth(4).unwrap().split(':').next().unwrap().split(',').map(str::trim).collect();
        let (start, length) = (operands[operands.len() - 2], operands[operands.len() - 1]);
        match inst.as_str() {
            "bfe.u64" | "bfe.s64" | "bfi.b64" => {
                op(start, "andi", Some(255));
                op(length, "andi", Some(255));
            }
            "bfe.s32" => {
                let low = op(start, "minui", Some(31));
                op(&low, "andi", Some(255));
                assert!(def(length).starts_with("memref.load"), "{line}");
            }
            "bfi.b32" => {
                op(start, "andi", Some(255));
                let low = op(length, "minui", None);
                op(&low, "andi", Some(255));
                let room = def(length).split(", ").nth(1).unwrap().trim_end_matches(" : i32");
                assert!(def(room).starts_with("arith.subi"), "{line}");
            }
            "bfe.u32" => {
                assert!(def(start).starts_with("memref.load") && def(length).starts_with("memref.load"), "{line}");
            }
            other => panic!("unexpected {other}"),
        }
        seen.push(inst);
    }
    seen.sort();
    assert_eq!(seen, ["bfe.s32", "bfe.s64", "bfe.u32", "bfe.u64", "bfi.b32", "bfi.b64"], "{mlir}");
    if let Some(tc) = toolchain() {
        ptx(&kernel, &tc);
    }
}

/// CuTe DSL's names for the bit counts compile to the PTX bit counts, and
/// `globaltimer_lo` reads `%globaltimer_lo` afresh each time (inline PTX with
/// side effects, which LLVM neither merges nor drops).
#[test]
fn bit_counts_and_timer() {
    let case = feature_kernels::bit_counts();
    let mlir = case.kernel.to_mlir();
    let timer: Vec<&str> = mlir.lines().filter(|l| l.contains("%globaltimer_lo")).collect();
    assert_eq!(timer.len(), 2, "{mlir}");
    assert!(timer.iter().all(|l| l.contains("llvm.inline_asm has_side_effects")), "{timer:?}");
    let Some(tc) = toolchain() else { return };
    let p = ptx(&case.kernel, &tc);
    for want in ["clz.b32", "clz.b64", "brev.b32", "brev.b64", "popc.b32", "popc.b64", "%globaltimer;"] {
        assert!(p.contains(want), "no {want} in\n{p}");
    }
    assert_eq!(count(&p, "%globaltimer_lo"), 2, "{p}");
}

/// `tensorop_gemm` with its swizzle on the shared-memory pointers (on
/// bytes) compiles to the PTX of the composed-layout form (on elements),
/// but for the name and the buffers' alignment: the pointer swizzle's
/// period, 512 bytes, against `make_smem_tensor`'s 128.
#[test]
fn pointer_swizzle_gemm_matches_the_composed_form() {
    let Some(tc) = toolchain() else { return };
    let cfg = tensorop_gemm::GemmConfig::new(256, 256, 256);
    let composed = ptx(&tensorop_gemm::build(cfg), &tc);
    let pointer = ptx(&tensorop_gemm::build(tensorop_gemm::GemmConfig { ptr_swizzle: true, ..cfg }), &tc);
    assert_eq!(count(&pointer, ".align 512"), 2, "{pointer}");
    let pointer = pointer.replace("tensorop_gemm_ptr_swizzle", "tensorop_gemm").replace(".align 512", ".align 128");
    assert_eq!(pointer, composed);
}

// ---------------------------------------------------------------------------
// Compile options, the split pipeline, optimization levels and line information.
// ---------------------------------------------------------------------------

/// Every example kernel and the target it is compiled for.
fn examples() -> Vec<(Kernel, Arch)> {
    let gemm = tensorop_gemm::GemmConfig::new(256, 256, 256);
    vec![
        (elementwise_add::naive(), Arch::Sm80),
        (elementwise_add::tiled(), Arch::Sm80),
        (sgemm::build(256, 256, 64), Arch::Sm80),
        (tensorop_gemm::build(gemm), Arch::Sm80),
        (tensorop_gemm::build(tensorop_gemm::GemmConfig { ab: DType::BF16, ..gemm }), Arch::Sm80),
        (tensorop_gemm::build(tensorop_gemm::GemmConfig { swizzle: false, ..gemm }), Arch::Sm80),
        (single_warp_pipeline::build(), Arch::Sm80),
        (warp_named_barrier::build(2, Arch::Sm89), Arch::Sm80),
        (warp_named_barrier::build(4, Arch::Sm90), Arch::Sm90),
        (pipeline_software_fill::build(2, Arch::Sm89), Arch::Sm80),
        (pipeline_software_fill::build(4, Arch::Sm90), Arch::Sm90),
        (cp_async_mbarrier_arrive::cta(Arch::Sm89), Arch::Sm80),
        (cp_async_mbarrier_arrive::cta(Arch::Sm90), Arch::Sm90),
        (cp_async_mbarrier_arrive::cluster(), Arch::Sm90a),
        (cooperative_launch::build(), Arch::Sm80),
        (runtime_gemm::build(), Arch::Sm80),
        (runtime_gemm::build_dynamic(), Arch::Sm80),
        (elementwise_add::dynamic().into_kernel(), Arch::Sm80),
        (ampere_sgemm::build_for(ampere_sgemm::Majors::new('m', 'n', 'n', 16)).into_kernel(), Arch::Sm80),
        (block_reduce::build::<i32>(4, 4, block_reduce::ReduceOp::Add), Arch::Sm80),
        (block_reduce::build::<f32>(4, 4, block_reduce::ReduceOp::Max), Arch::Sm80),
        (block_reduce::build::<F16>(4, 4, block_reduce::ReduceOp::Min), Arch::Sm80),
        (block_reduce::build::<BF16>(4, 4, block_reduce::ReduceOp::Add), Arch::Sm90),
        (redux_sync::integer_reductions(), Arch::Sm80),
        (redux_sync::float_reductions(), Arch::Sm100a),
        (redux_sync::redux_vs_shuffle(true), Arch::Sm100a),
        (redux_sync::redux_vs_shuffle(false), Arch::Sm80),
        (red::build::<i32>(red::DEFAULT_N, red::DEFAULT_THREADS), Arch::Sm80),
        (red::build::<f32>(red::DEFAULT_N, red::DEFAULT_THREADS), Arch::Sm80),
    ]
}

/// The m3 example kernels, with their targets: they need newer targets or NVVM forms (the FP8
/// `mma.sync` and its `nvvm.mma.sync` of LLVM 23) than [`examples`] holds, so they are compiled
/// through [`feature_kernels::options_for`] with the feature kernels.
fn m3_examples() -> Vec<(String, Kernel, Arch)> {
    use tensorop_gemm::GemmConfig;
    let fp8 = GemmConfig::fp8(256, 256, 256, DType::F8E4M3FN);
    [
        (tensorop_gemm::build(fp8), Arch::Sm89),
        (tensorop_gemm::build(GemmConfig { ab: DType::F8E5M2, c: DType::F8E5M2, ..fp8 }), Arch::Sm89),
        (redux_sync::absmax_quantize(redux_sync::DEFAULT_NUM_ROWS, true), Arch::Sm100a),
        (redux_sync::absmax_quantize(redux_sync::DEFAULT_NUM_ROWS, false), Arch::Sm89),
        (fp4_x2_tensor::build(fp4_x2_tensor::Decode::Integer), Arch::Sm80),
        (fp4_x2_tensor::build(fp4_x2_tensor::Decode::Cvt), Arch::Sm100a),
    ]
    .into_iter()
    .map(|(k, a)| (k.name().to_string(), k, a))
    .collect()
}

/// Every example and feature kernel, with its target, and the `cute::math` and loop-hint
/// kernels of [`math_and_loop_kernels`].
fn every_kernel() -> Vec<(String, Kernel, Arch)> {
    let examples = examples().into_iter().map(|(k, a)| (k.name().to_string(), k, a));
    let features = feature_kernels::all().into_iter().map(|c| (c.name.to_string(), c.kernel, Arch::Sm80));
    let sm90 = feature_kernels::sm90().into_iter().map(|c| (c.name.to_string(), c.kernel, Arch::Sm90));
    let sm86 = feature_kernels::sm86().into_iter().map(|c| (c.name.to_string(), c.kernel, Arch::Sm86));
    let sm100a = feature_kernels::sm100a().into_iter().map(|c| (c.name.to_string(), c.kernel, Arch::Sm100a));
    let gpu_only = feature_kernels::gpu_only().into_iter().map(|c| (c.name.to_string(), c.kernel, Arch::Sm80));
    let sm89 = feature_kernels::sm89().into_iter().map(|c| (c.name.to_string(), c.kernel, Arch::Sm89));
    let sm120a = feature_kernels::sm120a().into_iter().map(|c| (format!("{}_sm120a", c.name), c.kernel, Arch::Sm120a));
    let sm90a = feature_kernels::sm90a().into_iter().map(|c| (c.name.to_string(), c.kernel, Arch::Sm90a));
    let sm100 = feature_kernels::sm100().into_iter().map(|c| (c.name.to_string(), c.kernel, Arch::Sm100a));
    examples
        .chain(m3_examples())
        .chain(features)
        .chain(sm90)
        .chain(math_and_loop_kernels())
        .chain(sm86)
        .chain(sm100a)
        .chain(gpu_only)
        .chain(sm89)
        .chain(sm120a)
        .chain(sm90a)
        .chain(sm100)
        .collect()
}

/// `kernel`'s MLIR for `tc`: with the MMAs whose NVVM form `tc`'s LLVM predates as inline PTX
/// ([`feature_kernels::options_for`]).
fn mlir_for(kernel: &Kernel, tc: &Toolchain) -> String {
    let options = feature_kernels::options_for(kernel, tc);
    kernel.to_mlir_with(&MlirOptions {
        mma_inline_ptx: options.get_mma_inline_ptx().unwrap_or(false),
        fp_contract: options.get_fp_contract().unwrap_or(true),
        llvm_major: tc.llvm_version(),
        ..Default::default()
    })
}

/// What `cute::math` and the loop hints emit beyond the feature kernels (whose `cute::math`
/// kernels are precise): each explicit PTX instruction form (the rounding modes, the
/// approximations with and without `.ftz`, `rcp.approx.ftz.f32` as `nvvm.rcp.approx.ftz.f`,
/// `min` and `max` with NaN propagation or `.ftz`), fast mode, the sign-bit operations among
/// other code, and a vectorized loop (sm_100a) that also runs at least once.
fn math_and_loop_kernels() -> Vec<(String, Kernel, Arch)> {
    let up_ftz = MathOptions::rounding(RoundingMode::Up).with_ftz();
    let kernels = [
        ("options_f32_rp_ftz", math_cases::options_kernel(Ty::F32, 256, up_ftz)),
        ("options_f64_rm", math_cases::options_kernel(Ty::F64, 256, MathOptions::rounding(RoundingMode::Down))),
        ("approx_f32", math_cases::approx_kernel(Ty::F32, 256, false)),
        ("approx_f32_ftz", math_cases::approx_kernel(Ty::F32, 256, true)),
        ("approx_f64_ftz", math_cases::approx_kernel(Ty::F64, 256, true)),
        ("minmax_f32", math_cases::minmax_kernel(Ty::F32, 256)),
        ("ftz_f32", math_cases::ftz_kernel(256)),
        ("sign_context_f32", math_cases::sign_context_kernel(Ty::F32, 256)),
        ("sign_context_bf16", math_cases::sign_context_kernel(Ty::BF16, 256)),
        ("tan_f32_fast", math_cases::unary_kernel("tan", Ty::F32, 256, true)),
        ("pow_f16_fast", math_cases::binary_kernel("pow", Ty::F16, 256, true)),
    ];
    let vectorized = LoopOptions { vectorize: true, unroll: Some(4), at_least_once: true, ..LoopOptions::default() };
    let vectorized = ("hinted_loop_vectorized".to_string(), hinted_loop(vectorized, None), Arch::Sm100a);
    kernels.into_iter().map(|(name, k)| (name.to_string(), k, Arch::Sm80)).chain([vectorized]).collect()
}

fn compile(kernel: &Kernel, tc: &Toolchain, options: &CompileOptions) -> CompiledKernel {
    kernel.compile_with(tc, options).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()))
}

/// The operands of a `.loc` directive line (not `.local`).
fn loc_directive(line: &str) -> Option<&str> {
    line.trim_start().strip_prefix(".loc").filter(|rest| rest.starts_with(char::is_whitespace))
}

/// The PTX without its line information: `.loc` and `.file` directives, debug sections, the
/// labels debug information adds, and blank lines (the asm printer adds one after inline PTX
/// that has a location).
fn without_line_info(ptx: &str) -> String {
    let debug = |l: &&str| {
        let t = l.trim_start();
        t.is_empty()
            || loc_directive(t).is_some()
            || t.starts_with(".file")
            || (t.starts_with(".section") && t.contains(".debug_"))
            || t.starts_with("$L__func_begin")
            || t.starts_with("$L__func_end")
            || t.starts_with("$L__tmp")
    };
    ptx.lines().filter(|l| !debug(l)).collect::<Vec<_>>().join("\n")
}

/// The number of 128-bit accesses (`op` is `ld` or `st`) in any state space and vector shape.
fn wide_accesses(ptx: &str, op: &str) -> usize {
    let shapes = [".v4.b32", ".v4.f32", ".v4.u32", ".v4.s32", ".v2.b64", ".v2.f64", ".v2.u64", ".v2.s64"];
    ptx.lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with(&format!("{op}.")) && shapes.iter().any(|s| l.contains(s)))
        .count()
}

/// The installed `mlir-opt`s: the discovered one, and every Homebrew, distribution or
/// `/workspace` LLVM, each once.
fn installed_toolchains() -> Vec<Toolchain> {
    let mut paths: Vec<std::path::PathBuf> = Toolchain::discover().into_iter().map(|t| t.mlir_opt).collect();
    for root in ["/opt/homebrew/opt", "/usr/local/opt", "/usr/lib", "/workspace"] {
        let Ok(entries) = std::fs::read_dir(root) else { continue };
        let mut dirs: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("llvm")))
            .collect();
        dirs.sort();
        paths.extend(dirs.into_iter().map(|d| d.join("bin").join("mlir-opt")));
    }
    let mut seen = Vec::new();
    let mut toolchains = Vec::new();
    for p in paths.into_iter().filter(|p| p.is_file()) {
        let canonical = std::fs::canonicalize(&p).unwrap_or(p.clone());
        if !seen.contains(&canonical) {
            seen.push(canonical);
            // cute-rs needs LLVM 21 or newer (older pipelines lack options it uses, such as
            // use-vector-alignment): an older distribution LLVM is skipped.
            match mlir_opt_major_version(&p) {
                Some(major) if major >= 21 => toolchains.push(Toolchain::new(p)),
                Some(major) => eprintln!("skipping {}: LLVM {major} is older than 21", p.display()),
                None => eprintln!("skipping {}: cannot read its LLVM version from `mlir-opt --version`", p.display()),
            }
        }
    }
    toolchains
}

/// The LLVM major version of an `mlir-opt`, from its `--version` output ("LLVM version 22.1.8",
/// "Homebrew LLVM version 21.1.8", "Ubuntu LLVM version 19.1.1").
fn mlir_opt_major_version(mlir_opt: &std::path::Path) -> Option<u32> {
    let out = std::process::Command::new(mlir_opt).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    llvm_major_version(&text)
}

/// The major version in `mlir-opt --version`'s output.
fn llvm_major_version(text: &str) -> Option<u32> {
    let rest = &text[text.find("LLVM version ")? + "LLVM version ".len()..];
    rest.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

#[test]
fn llvm_versions_are_read_from_mlir_opt() {
    assert_eq!(llvm_major_version("LLVM (http://llvm.org/):\n  LLVM version 22.1.8\n  Optimized build.\n"), Some(22));
    assert_eq!(llvm_major_version("Homebrew LLVM version 21.1.8\n  Optimized build.\n"), Some(21));
    assert_eq!(llvm_major_version("Ubuntu LLVM version 19.1.1\n"), Some(19));
    assert_eq!(llvm_major_version("mlir-opt: unknown option"), None);
    for tc in installed_toolchains() {
        let major = mlir_opt_major_version(&tc.mlir_opt);
        assert!(major.is_some_and(|m| m >= 21), "{}: {major:?}", tc.mlir_opt.display());
    }
}

/// At default options the split pipeline gives exactly the PTX of the single
/// `gpu-lower-to-nvvm-pipeline` run it replaced, for every example and feature kernel
/// without by-value parameters.
#[test]
fn default_ptx_is_unchanged_by_the_split() {
    let Some(tc) = toolchain() else { return };
    let mut failures = Vec::new();
    for (name, kernel, arch) in every_kernel() {
        let options = feature_kernels::options_for(&kernel, &tc);
        let compiled = compile(&kernel, &tc, &options.clone().gpu_arch(arch));
        assert_eq!(
            (compiled.name(), compiled.arch(), compiled.opt_level(), compiled.line_info()),
            (kernel.name(), arch, 3, false)
        );
        // By-value parameters (a TMA atom's hidden descriptor) need the stage hook that marks
        // them `llvm.byval`, which the single run predates.
        let by_value = kernel.ir().args.iter().any(|p| matches!(p, cute::ir::Param::ByValue { .. }));
        let reference = tc.compile_ptx_reference(&mlir_for(&kernel, &tc), arch, compiled.ptx_version()).unwrap();
        if compiled.ptx() != reference && !by_value {
            failures.push(name);
        }
        // `compile_ptx` uses the default options, so compare it with a default compile
        // (`options_for` turns contraction off, which changes the PTX).
        if options.get_mma_inline_ptx() != Some(true) {
            let default = compile(&kernel, &tc, &CompileOptions::new().gpu_arch(arch));
            assert_eq!(kernel.compile_ptx(&tc, arch).unwrap(), default.ptx(), "{}", kernel.name());
        }
    }
    assert!(failures.is_empty(), "the split pipeline changed the PTX of {failures:?}");
}

/// Each installed `mlir-opt` (LLVM 21 and 23 on the Mac, 22 on the pod) expands the pipeline
/// itself, without the fallback list, and its split pipeline matches its single run.
#[test]
fn pipeline_is_derived_from_every_installed_mlir_opt() {
    let toolchains = installed_toolchains();
    if toolchains.is_empty() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    }
    for tc in toolchains {
        let who = tc.mlir_opt.display().to_string();
        for (arch, ptx, level) in [(Arch::Sm80, 70, 3), (Arch::Sm90a, 80, 0), (Arch::Sm120, 87, 2)] {
            let pipeline = tc.lowering_pipeline(arch, ptx, level);
            assert_eq!(pipeline.fallback_reason(), None, "{who}");
            let stage1 = pipeline.stage1();
            for want in [format!("chip={}", arch.chip()), format!("features=+ptx{ptx}"), format!("O={level}")] {
                assert!(stage1.contains(&want), "{who}: {want} missing from {stage1}");
            }
            assert!(pipeline.stage2().starts_with("builtin.module(gpu-to-llvm{"), "{who}");
            // Remembered, not derived again.
            assert!(std::sync::Arc::ptr_eq(&pipeline, &tc.lowering_pipeline(arch, ptx, level)));
        }
        for (kernel, arch) in examples() {
            let compiled = compile(&kernel, &tc, &CompileOptions::new().gpu_arch(arch));
            let reference = tc.compile_ptx_reference(&kernel.to_mlir(), arch, compiled.ptx_version()).unwrap();
            assert!(compiled.ptx() == reference, "{who}: the split changed the PTX of {}", kernel.name());
        }
    }
}

/// MLIR 21's pass list, which a `Toolchain` falls back to when its `mlir-opt`'s own expansion
/// cannot be parsed, still compiles on every installed `mlir-opt`: every example and feature
/// kernel gets the PTX of the single `gpu-lower-to-nvvm-pipeline` run, and at O0 to O2 the
/// examples get the PTX of the derived pipeline.
#[test]
fn fallback_pipeline_compiles_like_the_derived_one() {
    let toolchains = installed_toolchains();
    if toolchains.is_empty() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    }
    // (toolchain, kernel, MLIR, target, PTX version, level). The PTX version is the one
    // compile_with uses: what the instructions need, raised to the target's minimum by both
    // pipelines.
    let mut jobs = Vec::new();
    for (t, tc) in toolchains.iter().enumerate() {
        for (name, kernel, arch) in every_kernel() {
            let ptx = cute::validate::check_target(kernel.ir(), arch).unwrap();
            jobs.push((t, name, mlir_for(&kernel, tc), arch, ptx, 3));
        }
        for (kernel, arch) in examples() {
            let ptx = cute::validate::check_target(kernel.ir(), arch).unwrap();
            for level in 0..=2 {
                jobs.push((t, format!("{} at O{level}", kernel.name()), kernel.to_mlir(), arch, ptx, level));
            }
        }
    }
    // Each job runs mlir-opt three to five times, so the jobs share a few threads.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failures = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get().min(8)) {
            s.spawn(|| {
                while let Some((t, name, mlir, arch, ptx, level)) =
                    jobs.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
                {
                    let tc = &toolchains[*t];
                    let fallback = LoweringPipeline::fallback_for(*arch, *ptx, *level);
                    assert!(fallback.fallback_reason().is_some());
                    let expected = match level {
                        3 => tc.compile_ptx_reference(mlir, *arch, *ptx),
                        _ => tc.lower_with(&tc.lowering_pipeline(*arch, *ptx, *level), mlir),
                    };
                    let failure = match (tc.lower_with(&fallback, mlir), expected) {
                        (Ok(got), Ok(want)) if got == want => continue,
                        (Ok(_), Ok(_)) => "the fallback list changed the PTX".to_string(),
                        (Err(e), _) => format!("the fallback list failed: {e}"),
                        (_, Err(e)) => format!("the reference failed: {e}"),
                    };
                    failures.lock().unwrap().push(format!("{}: {name}: {failure}", tc.mlir_opt.display()));
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Without LLVM's cleanups (O0) and at O1 and O2, the proven 128-bit accesses, `mma.sync`,
/// `ldmatrix`, `cp.async` and inline PTX survive; and the level reaches LLVM.
#[test]
fn every_opt_level_keeps_the_vector_accesses_and_tensor_cores() {
    let Some(tc) = toolchain() else { return };
    for level in [0, 1, 2, 3] {
        let options = CompileOptions::new().gpu_arch(Arch::Sm80).opt_level(level);
        let ptx = |k: &Kernel| compile(k, &tc, &options).into_ptx();
        let tiled = ptx(&elementwise_add::tiled());
        assert_eq!((wide_accesses(&tiled, "ld"), wide_accesses(&tiled, "st")), (8, 4), "O{level}:\n{tiled}");
        let gemm = ptx(&tensorop_gemm::build(tensorop_gemm::GemmConfig::new(256, 256, 256)));
        assert_eq!(count(&gemm, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"), 64, "O{level}");
        assert_eq!(count(&gemm, "ldmatrix.sync.aligned.m8n8.x4.shared.b16"), 8, "O{level}");
        assert_eq!(count(&gemm, "ldmatrix.sync.aligned.m8n8.x2.shared.b16"), 16, "O{level}");
        assert!(count(&gemm, "cp.async.cg.shared.global") > 0, "O{level}");
        let pipeline = ptx(&single_warp_pipeline::build());
        assert_eq!(count(&pipeline, "mma.sync.aligned.m16n8k16"), 8, "O{level}");
        assert_eq!(count(&pipeline, "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"), 8, "O{level}");
        let sgemm = ptx(&sgemm::build(256, 256, 64));
        assert_eq!(count(&sgemm, "fma.rn.f32"), 8 * 64 + 64, "O{level}");
        let asm = ptx(&feature_kernels::inline_ptx_fragment().kernel);
        assert!(asm.contains("popc.b32"), "O{level}: the inline PTX is missing:\n{asm}");
        assert!(tc.lowering_pipeline(Arch::Sm80, 70, level).stage1().contains(&format!("O={level}")));
    }
    let at =
        |level| compile(&elementwise_add::tiled(), &tc, &CompileOptions::new().gpu_arch(Arch::Sm80).opt_level(level));
    let (o0, o3) = (at(0), at(3));
    assert!(o0.ptx().lines().count() > o3.ptx().lines().count(), "O0 should not be optimized");
    assert_eq!((o0.opt_level(), o3.opt_level()), (0, 3));
}

/// Every example and feature kernel compiles at O0, including the SM90 ones.
#[test]
fn every_kernel_compiles_at_o0() {
    let Some(tc) = toolchain() else { return };
    let failures: Vec<String> = every_kernel()
        .into_iter()
        .filter_map(|(name, k, arch)| {
            k.compile_with(&tc, &feature_kernels::options_for(&k, &tc).gpu_arch(arch).opt_level(0))
                .err()
                .map(|e| format!("{name}: {e}"))
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The `.file` table of PTX: file number to path.
fn files(ptx: &str) -> Vec<(u32, String)> {
    ptx.lines()
        .filter_map(|l| {
            let rest = l.trim_start().strip_prefix(".file")?;
            let (n, path) = rest.trim_start().split_once(char::is_whitespace)?;
            Some((n.parse().ok()?, path.trim().trim_matches('"').to_string()))
        })
        .collect()
}

/// The `(file, line, column)` of the last `.loc` before the first instruction that starts
/// with one of `opcodes`.
fn loc_before(ptx: &str, opcodes: &[&str]) -> (u32, u32, u32) {
    let lines: Vec<&str> = ptx.lines().collect();
    let at = lines
        .iter()
        .position(|l| opcodes.iter().any(|o| l.trim_start().starts_with(o)))
        .unwrap_or_else(|| panic!("no {opcodes:?} in\n{ptx}"));
    let loc = lines[..at].iter().rev().find_map(|l| loc_directive(l)).expect(".loc");
    let n: Vec<u32> = loc.split_whitespace().take(3).map(|w| w.parse().unwrap()).collect();
    (n[0], n[1], n[2])
}

/// The 1-based line and column of `needle` in `source`.
fn position(source: &str, needle: &str) -> (u32, u32) {
    let (i, line) = source.lines().enumerate().find(|(_, l)| l.contains(needle)).expect("needle");
    (i as u32 + 1, line.find(needle).unwrap() as u32 + 1)
}

/// With line information, the PTX maps instructions back to the Rust lines and columns that
/// built them, in the kernel's file and in a helper's, and names only Rust files.
#[test]
fn line_info_maps_ptx_to_rust_lines() {
    let Some(tc) = toolchain() else { return };
    // Without contraction, so that the helper's `x * 3.0 + 1.0` keeps its own multiply.
    let options = CompileOptions::new().gpu_arch(Arch::Sm80).generate_line_info(true).fp_contract(false);

    // A known statement of the elementwise_add example.
    let compiled = compile(&elementwise_add::naive(), &tc, &options);
    assert!(compiled.line_info());
    let ptx = compiled.ptx();
    let (file, line, column) = loc_before(ptx, &["add.f32", "add.rn.f32"]);
    let (want_line, want_column) =
        position(include_str!("../examples/elementwise_add.rs"), "a.load((m, n)) + b.load((m, n))");
    assert_eq!((line, column), (want_line, want_column), "{ptx}");
    let table = files(ptx);
    let path = &table.iter().find(|(n, _)| *n == file).expect("the .loc's file").1;
    assert!(path.ends_with("examples/elementwise_add.rs"), "{path}");
    assert!(std::path::Path::new(path).is_file(), "{path} should be the source file");
    // Line information only: no DWARF sections, so ptxas accepts it at any level.
    assert!(ptx.lines().any(|l| l.trim() == ".target sm_80"), "{ptx}");
    assert!(!ptx.contains(".debug_info") && !ptx.contains(".debug_abbrev"), "{ptx}");

    // An operation built by a helper in another file than the kernel.
    let kernel = Kernel::builder("line_info_helper").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, feature_kernels::triple_plus_one(x.load(t)));
    });
    let ptx = compile(&kernel, &tc, &options).into_ptx();
    let (file, line, column) = loc_before(&ptx, &["mul.f32", "mul.rn.f32"]);
    assert_eq!((line, column), position(include_str!("feature_kernels/mod.rs"), "x * 3.0"), "{ptx}");
    let table = files(&ptx);
    let path = &table.iter().find(|(n, _)| *n == file).expect("the helper's file").1;
    assert!(path.ends_with("tests/feature_kernels/mod.rs"), "{path}\n{ptx}");
    let (file, line, _) = loc_before(&ptx, &["st.global"]);
    let this = &table.iter().find(|(n, _)| *n == file).expect("this file").1;
    assert!(this.ends_with("tests/mlir.rs"), "{this}");
    assert_eq!(line, position(include_str!("mlir.rs"), "x.store(t, feature_kernels::triple_plus_one").0);
}

/// Line information does not change a single instruction, and every location names a Rust
/// file (an operation without one would get the MLIR text's position), for every example and
/// feature kernel.
#[test]
fn line_info_changes_no_instruction() {
    let Some(tc) = toolchain() else { return };
    let mut failures = Vec::new();
    for (name, kernel, arch) in every_kernel() {
        let options = feature_kernels::options_for(&kernel, &tc).gpu_arch(arch);
        let plain = compile(&kernel, &tc, &options);
        let lined = compile(&kernel, &tc, &options.clone().generate_line_info(true));
        if without_line_info(lined.ptx()) != without_line_info(plain.ptx()) {
            failures.push(format!("{name}: line information changed the instructions"));
        }
        let table = files(lined.ptx());
        if table.is_empty() || table.iter().any(|(_, p)| !p.ends_with(".rs")) {
            failures.push(format!("{name}: .file table {table:?}"));
        }
        for l in lined.ptx().lines().filter_map(loc_directive) {
            let n: u32 = l.split_whitespace().next().unwrap().parse().unwrap();
            if !table.iter().any(|(f, _)| *f == n) {
                failures.push(format!("{name}: .loc{l} names no file"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// With locations on, every operation has one: the instruction that built it, or the
/// kernel's `Kernel::build` call.
#[test]
fn every_operation_is_located() {
    let Some(tc) = toolchain() else { return };
    for (name, kernel, _) in every_kernel() {
        let mma_inline_ptx = feature_kernels::options_for(&kernel, &tc).get_mma_inline_ptx().unwrap_or(false);
        for line_info in [false, true] {
            let mlir = kernel.to_mlir_with(&MlirOptions {
                locations: true,
                line_info,
                mma_inline_ptx,
                llvm_major: tc.llvm_version(),
                ..Default::default()
            });
            for l in mlir.lines().map(str::trim) {
                let structure = l.ends_with('{') || l.ends_with(':') || l.starts_with('#');
                assert!(structure || l.contains(" loc("), "{name}: unlocated `{l}`");
            }
            tc.verify(&mlir).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
    let mlir =
        elementwise_add::naive().to_mlir_with(&MlirOptions { locations: false, line_info: true, ..Default::default() });
    assert!(mlir.contains("#llvm.di_subprogram<") && mlir.contains("emissionKind = DebugDirectivesOnly"));
    assert!(mlir.contains("} loc(fused<#di_sp>["), "{mlir}");
}

/// A stage hook that changes nothing leaves the PTX alone; the helpers read and set
/// attributes of the lowered module; a hook that raises the alignment of dynamic shared
/// memory shows in the PTX.
#[test]
fn stage_hooks() {
    let Some(tc) = toolchain() else { return };
    let options = CompileOptions::new().gpu_arch(Arch::Sm80);
    // The examples for sm_80 (the others need newer targets).
    for (kernel, _) in examples().into_iter().filter(|&(_, arch)| arch == Arch::Sm80) {
        let identity = options.clone().stage_hook(|_| Ok(())).stage_hook(|m| {
            let text = m.text().to_string();
            m.set_text(text);
            Ok(())
        });
        assert_eq!(compile(&kernel, &tc, &identity).ptx(), compile(&kernel, &tc, &options).ptx(), "{}", kernel.name());
    }

    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let record = seen.clone();
    let reads = options.clone().stage_hook(move |m| {
        let name = m.kernel_name().to_string();
        record.lock().unwrap().push(m.argument_attribute(&name, 0, "llvm.align"));
        record.lock().unwrap().push(m.global_attribute("__cute_dynamic_smem", "alignment"));
        Ok(())
    });
    let dynamic = feature_kernels::dynamic_shared_memory().kernel;
    compile(&dynamic, &tc, &reads);
    // The window is as aligned as its buffer (`make_smem_tensor` aligns to 128 bytes).
    assert_eq!(*seen.lock().unwrap(), [Some("16 : i64".to_string()), Some("128 : i64".to_string())]);

    // The dynamic shared memory: `.extern .shared .align 128 .b8 __cute_dynamic_smem[];`.
    let shared_align = |ptx: &str| -> Vec<String> {
        ptx.lines()
            .filter(|l| l.contains(".shared") && l.contains("__cute_dynamic_smem") && !l.contains("mov"))
            .filter_map(|l| l.split(".align").nth(1)?.split_whitespace().next().map(str::to_string))
            .collect()
    };
    let plain = compile(&dynamic, &tc, &options).into_ptx();
    assert_eq!(shared_align(&plain), ["128"], "{plain}");
    let aligned =
        options.clone().stage_hook(|m| m.set_global_attribute("__cute_dynamic_smem", "alignment", "1024 : i64"));
    let ptx = compile(&dynamic, &tc, &aligned).into_ptx();
    assert_eq!(shared_align(&ptx), ["1024"], "{ptx}");

    // Argument attributes reach the PTX too.
    let naive = elementwise_add::naive();
    let wider = options.clone().stage_hook(|m| {
        let name = m.kernel_name().to_string();
        m.set_argument_attribute(&name, 2, "llvm.align", "64 : i64")
    });
    let ptx = compile(&naive, &tc, &wider).into_ptx();
    assert!(ptx.contains(".param .u64 .ptr .align 64 elementwise_add_naive_param_2"), "{ptx}");

    let failing = options.clone().stage_hook(|_| Err("no thanks".to_string()));
    let e = naive.compile_with(&tc, &failing).unwrap_err();
    assert!(matches!(e, CompileError::Hook(_)) && e.to_string().contains("hook 1 of 1: no thanks"), "{e}");
    let missing = options.clone().stage_hook(|m| m.set_global_attribute("nope", "alignment", "8 : i64"));
    let e = naive.compile_with(&tc, &missing).unwrap_err().to_string();
    assert!(e.contains("no llvm.mlir.global @nope"), "{e}");
    let breaking = options.clone().stage_hook(|m| {
        m.set_text(m.text().replace("llvm.return", "llvm.retrun"));
        Ok(())
    });
    let e = naive.compile_with(&tc, &breaking).unwrap_err().to_string();
    assert!(e.contains("after 1 stage hook(s) edited the module"), "{e}");
}

/// Option strings reach the compiler, and compiling needs a target.
#[test]
fn compile_options_from_strings() {
    let Some(tc) = toolchain() else { return };
    let kernel = elementwise_add::tiled();
    let parsed = CompileOptions::parse("--gpu-arch sm_86 --opt-level 1 --generate-line-info").unwrap();
    let compiled = compile(&kernel, &tc, &parsed);
    assert_eq!(
        (compiled.arch(), compiled.ptx_version(), compiled.opt_level(), compiled.line_info()),
        (Arch::Sm86, 71, 1, true)
    );
    assert!(compiled.ptx().contains(".target sm_86") && compiled.ptx().contains(".loc"));
    let typed = CompileOptions::new().gpu_arch(Arch::Sm86).opt_level(1).generate_line_info(true);
    assert_eq!(compile(&kernel, &tc, &typed).ptx(), compiled.ptx());
    assert_eq!(compiled.options().get_opt_level(), Some(1));

    let e = kernel.compile_with(&tc, &CompileOptions::new()).unwrap_err();
    assert!(matches!(e, CompileError::Options(_)), "{e}");
    let e = e.to_string();
    assert!(e.contains("no target architecture") && e.contains("elementwise_add_tiled"), "{e}");
    // compile_with does not read the environment, and the message says how to add it.
    assert!(e.contains("does not read the environment; CompileOptions::with_env adds CUTE_RS_ARCH"), "{e}");
    let env = |name: &str| (name == "CUTE_RS_ARCH").then(|| "sm_86".to_string());
    let from_env = CompileOptions::new().with_env_from(env).unwrap();
    assert_eq!(compile(&kernel, &tc, &from_env).arch(), Arch::Sm86);
}

// ---------------------------------------------------------------------------
// Warp collectives: shuffles, redux.sync, match.sync, bar.warp.sync, fmax and fmin.
// ---------------------------------------------------------------------------

/// The PTX lines with their whitespace collapsed.
fn compact_lines(ptx: &str) -> Vec<String> {
    ptx.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect()
}

/// The operands of `instruction` lines (`shfl.sync.idx.b32 %r1, %r2, 3, 31, -1;` gives `["%r1", "%r2", "3",
/// "31", "-1"]`).
fn operands_of<'a>(lines: &'a [String], instruction: &str) -> Vec<Vec<&'a str>> {
    let prefix = format!("{instruction} ");
    lines
        .iter()
        .filter_map(|l| l.strip_prefix(&prefix))
        .map(|rest| rest.trim_end_matches(';').split(", ").collect())
        .collect()
}

/// Shuffles are one `shfl.sync` per 32-bit word (two for a 64-bit value), with the offset, CuTe DSL's
/// `mask_and_clamp` and the member mask as given: constants as immediates (31 for the whole warp, 0 for `up`,
/// 6151 = 0x1807 for segments of 8, 4111 = 0x100f for 16), runtime values in registers.
#[test]
fn shuffle_instructions() {
    let Some(tc) = toolchain() else { return };
    let kind = |k: ShuffleKind| match k {
        ShuffleKind::Idx => "idx",
        ShuffleKind::Up => "up",
        ShuffleKind::Down => "down",
        ShuffleKind::Bfly => "bfly",
    };
    for (case, words) in [(feature_kernels::shuffles_f64(), 2), (feature_kernels::shuffles_i32(), 1)] {
        let p = ptx(&case.kernel, &tc);
        let lines = compact_lines(&p);
        for (k, offset, clamp) in feature_kernels::SHUFFLES {
            let found = operands_of(&lines, &format!("shfl.sync.{}.b32", kind(k)))
                .into_iter()
                .filter(|ops| {
                    let offset_ok = offset.map_or(ops[2].starts_with("%r"), |o| ops[2] == o.to_string());
                    offset_ok && ops[3] == clamp.to_string() && ops[4] == "-1"
                })
                .count();
            let same = feature_kernels::SHUFFLES.iter().filter(|&&s| s == (k, offset, clamp)).count();
            assert_eq!(found, words * same, "{}: shfl.sync.{} by {offset:?} with {clamp:#x}\n{p}", case.name, kind(k));
        }
        // The half-warps' shuffles in divergent code, each with its own member mask.
        let masked = |k: &str, offset: &str, mask: &str| {
            operands_of(&lines, &format!("shfl.sync.{k}.b32"))
                .iter()
                .filter(|ops| ops[2] == offset && ops[3] == "31" && ops[4] == mask)
                .count()
        };
        assert_eq!(masked("bfly", "1", "65535"), words, "{}\n{p}", case.name);
        assert_eq!(masked("down", "2", "-65536"), words, "{}\n{p}", case.name);
    }
}

/// `redux.sync` compiles to the convergent NVVM intrinsics, one instruction per kind (`max` and `min` on a Uint32
/// are the unsigned forms), with the member mask in a register; it needs nothing beyond sm_80 and PTX 7.0.
#[test]
fn redux_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::redux_int().kernel;
    let mlir = kernel.to_mlir();
    assert_eq!(mlir.matches("nvvm.redux.sync").count(), 2 * 8 + 5, "{mlir}");
    assert!(!mlir.contains("llvm.inline_asm"), "{mlir}");
    let p = ptx(&kernel, &tc);
    assert!(p.contains(".version 7.0"), "{p}");
    let lines = compact_lines(&p);
    for (instruction, n) in [
        ("redux.sync.add.s32", 3),
        ("redux.sync.and.b32", 2),
        ("redux.sync.or.b32", 2),
        ("redux.sync.xor.b32", 3),
        ("redux.sync.max.s32", 2),
        ("redux.sync.min.s32", 2),
        ("redux.sync.max.u32", 4),
        ("redux.sync.min.u32", 3),
    ] {
        assert_eq!(operands_of(&lines, instruction).len(), n, "{instruction}\n{p}");
    }
    // The member mask is a register operand, even when it is a constant.
    let masks: Vec<&str> = operands_of(&lines, "redux.sync.add.s32").into_iter().map(|ops| ops[2]).collect();
    assert!(masks.iter().all(|m| m.starts_with("%r")), "{masks:?}");
    for arch in Arch::ALL.into_iter().filter(|&a| emittable(&tc, a)) {
        kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
    }
}

/// Float32 `redux.sync` is architecture-specific: it compiles for sm_100a (PTX 8.6) with `.abs` and `.NaN` as
/// given, and every other target is refused, naming the targets that have it and the kernel's line.
#[test]
fn float_redux_needs_sm100a() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::redux_f32().kernel;
    let p = kernel.compile_ptx(&tc, Arch::Sm100a).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.contains(".version 8.6") && p.contains(".target sm_100a"), "{p}");
    let lines = compact_lines(&p);
    for (instruction, n) in [
        ("redux.sync.max.f32", 3),
        ("redux.sync.max.abs.f32", 2),
        ("redux.sync.max.NaN.f32", 2),
        ("redux.sync.max.abs.NaN.f32", 2),
        ("redux.sync.min.f32", 3),
        ("redux.sync.min.abs.f32", 2),
        ("redux.sync.min.NaN.f32", 2),
        ("redux.sync.min.abs.NaN.f32", 2),
    ] {
        assert_eq!(operands_of(&lines, instruction).len(), n, "{instruction}\n{p}");
    }
    // The sm_100f family has it too: sm_100f, and the a and f targets of
    // sm_103 and sm_107.
    let family = [Arch::Sm100a, Arch::Sm100f, Arch::Sm103a, Arch::Sm103f, Arch::Sm107a, Arch::Sm107f];
    for &arch in &family[1..] {
        assert_eq!(cute::validate::check_target(kernel.ir(), arch).unwrap(), arch.min_ptx().max(88), "{arch:?}");
    }
    for arch in Arch::ALL.into_iter().filter(|a| !family.contains(a)) {
        let e = kernel.compile_ptx(&tc, arch).unwrap_err().to_string();
        let want = format!(
            "redux.sync.max.f32 needs sm_100a (or the sm_100f family with PTX 8.8); the kernel is being compiled for {}",
            arch.chip()
        );
        assert!(e.contains(&want) && e.contains("feature_kernels/mod.rs"), "{arch:?}: {e}");
    }
}

/// `match_sync` is `match.{any,all}.sync` on 32- and 64-bit values; `sync_warp` is `bar.warp.sync` with the mask
/// given, an immediate or a register.
#[test]
fn match_and_warp_barrier_instructions() {
    let Some(tc) = toolchain() else { return };
    let p = ptx(&feature_kernels::warp_match().kernel, &tc);
    let lines = compact_lines(&p);
    for (instruction, n) in
        [("match.any.sync.b32", 3), ("match.any.sync.b64", 3), ("match.all.sync.b32", 3), ("match.all.sync.b64", 3)]
    {
        assert_eq!(operands_of(&lines, instruction).len(), n, "{instruction}\n{p}");
    }
    let mask = |instruction: &str| -> Vec<String> {
        operands_of(&lines, instruction).into_iter().map(|ops| ops.last().unwrap().to_string()).collect()
    };
    assert!(mask("match.any.sync.b32").contains(&"1431655765".to_string()), "0x55555555\n{p}");
    assert!(mask("match.all.sync.b64").contains(&"-1431655766".to_string()), "0xaaaaaaaa\n{p}");
    let p = ptx(&feature_kernels::warp_barriers().kernel, &tc);
    let masks: Vec<String> =
        operands_of(&compact_lines(&p), "bar.warp.sync").into_iter().map(|ops| ops[0].to_string()).collect();
    assert_eq!(masks.len(), 4, "{p}");
    for immediate in ["-1", "-65536", "65535"] {
        assert_eq!(masks.iter().filter(|m| *m == immediate).count(), 1, "{immediate}\n{p}");
    }
    assert_eq!(masks.iter().filter(|m| m.starts_with("%r")).count(), 1, "{p}");
}

/// `fmax` and `fmin` with options are one pure `max`/`min` instruction with the modifiers in the PTX ISA's order
/// (`.ftz`, `.NaN`, `.xorsign.abs`); without options they are LLVM's `maxnum` and `minnum` (`max.f32`). The
/// `.xorsign.abs` forms need sm_86 and PTX 7.2: they raise the PTX version on sm_86 and are refused on sm_80.
#[test]
fn fmax_fmin_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::fmax_fmin().kernel;
    for line in kernel.to_mlir().lines().filter(|l| l.contains("llvm.inline_asm")) {
        assert!(!line.contains("has_side_effects"), "{line}");
    }
    let e = kernel.compile_ptx(&tc, Arch::Sm80).unwrap_err().to_string();
    assert!(
        e.contains("min and max with .xorsign.abs (fmax and fmin with abs) needs sm_86 or newer")
            && e.contains("compiled for sm_80")
            && e.contains("feature_kernels/mod.rs"),
        "{e}"
    );
    let p = kernel.compile_ptx(&tc, Arch::Sm86).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.contains(".version 7.2"), "{p}");
    let lines = compact_lines(&p);
    for func in ["max", "min"] {
        for ftz in ["", ".ftz"] {
            for nan in ["", ".NaN"] {
                for abs in ["", ".xorsign.abs"] {
                    let instruction = format!("{func}{ftz}{nan}{abs}.f32");
                    assert_eq!(operands_of(&lines, &instruction).len(), 7, "{instruction}\n{p}");
                }
            }
        }
    }
    let p = kernel.compile_ptx(&tc, Arch::Sm89).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.contains(".version 7.8"), "{p}");
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm120).unwrap(), 87);
}

// ---------------------------------------------------------------------------
// Extended-precision arithmetic.
// ---------------------------------------------------------------------------

/// The opcode of a PTX instruction line (without a predicate), if the line is
/// one.
fn opcode(line: &str) -> Option<&str> {
    let mut words = line.split_whitespace();
    let first = words.next()?;
    let first = if first.starts_with('@') { words.next()? } else { first };
    (!first.starts_with("//") && !first.starts_with('.')).then_some(first)
}

/// Whether an opcode is one of the extended-precision instructions, the only
/// ones that touch the carry flag, and whether it reads and writes it.
fn carry_use(opcode: &str) -> Option<(bool, bool)> {
    let mut parts = opcode.split('.');
    let base = parts.next()?;
    let cc = opcode.split('.').any(|p| p == "cc");
    match base {
        "addc" | "subc" | "madc" => Some((true, cc)),
        "add" | "sub" | "mad" if cc => Some((false, true)),
        _ => None,
    }
}

/// The carry-flag instructions of PTX in order, with `None` for whatever ends
/// a stretch of straight-line code: a label, a branch, a call or a return.
fn carry_sequence(ptx: &str) -> Vec<Option<String>> {
    ptx.lines()
        .filter_map(|line| {
            let t = line.trim();
            if t.ends_with(':') && !t.starts_with("//") {
                return Some(None);
            }
            let op = opcode(t)?;
            if op.starts_with("bra") || op.starts_with("call") || op == "ret;" || op == "exit;" || op == "ret" {
                return Some(None);
            }
            carry_use(op).map(|_| Some(op.to_string()))
        })
        .collect()
}

/// The carry instructions of a kernel's IR, in program order, as PTX opcodes.
fn carry_instructions(kernel: &Kernel) -> Vec<String> {
    fn walk(r: &cute::ir::Region, ir: &cute::ir::KernelIr, out: &mut Vec<String>) {
        for inst in &r.insts {
            if let cute::ir::Op::CarryInst { func, args, .. } = &inst.op {
                out.push(cute::ir::carry_inst_name(*func, ir.dtype(args[0])));
            }
            for region in inst.op.regions() {
                walk(region, ir, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(&kernel.ir().body, kernel.ir(), &mut out);
    out
}

/// Each extended-precision instruction is one PTX instruction of inline
/// assembly with side effects, as in CuTe DSL, and in the PTX of the carry
/// feature kernels, at O0 to O3 and with line information, every carry-in
/// follows a carry-out in the same stretch of straight-line code, with
/// nothing between them that touches the flag: the only carry instructions
/// in the PTX are the kernel's own, in its order (LLVM neither drops,
/// duplicates, reorders nor adds one). That includes chains around 64-bit
/// divisions by constants and a 32-bit division by a runtime value, which
/// LLVM lowers without a branch, around warp collectives, and around the
/// special-register reads and `cp.async.mbarrier.arrive` (with the cluster
/// registers in a kernel for sm_90 alone). The carry instructions need no
/// newer PTX than the target's minimum (sm_80's 7.0); the kernels around the
/// special registers need 7.1 for their mbarrier waits, and 8.1 for
/// `aggr_smem_size`.
#[test]
fn carry_chains_stay_in_order() {
    // Each kernel, its targets and the PTX version its other instructions need.
    let every = &[Arch::Sm80, Arch::Sm89, Arch::Sm90][..];
    let cases = [
        (feature_kernels::carry_words_u32(), every, 0),
        (feature_kernels::carry_words_u64(), every, 0),
        (feature_kernels::carry_multiply(), every, 0),
        (feature_kernels::carry_in_loop(), every, 0),
        (feature_kernels::carry_around_division(), every, 0),
        (feature_kernels::carry_around_collectives(), every, 0),
        (feature_kernels::carry_around_smem_registers(), every, 71),
        (feature_kernels::carry_around_smem_registers_sm90(), &[Arch::Sm90][..], 81),
    ];
    for (case, ..) in &cases {
        let mlir = case.kernel.to_mlir();
        let asm: Vec<&str> = mlir
            .lines()
            .filter(|l| {
                l.contains("llvm.inline_asm") && l.split('"').nth(1).and_then(opcode).and_then(carry_use).is_some()
            })
            .collect();
        assert_eq!(asm.len(), carry_instructions(&case.kernel).len(), "{}:\n{mlir}", case.name);
        assert!(asm.iter().all(|l| l.contains("llvm.inline_asm has_side_effects \"")), "{}: {asm:#?}", case.name);
    }
    let Some(tc) = toolchain() else { return };
    for (case, archs, other_ptx) in &cases {
        let expected = carry_instructions(&case.kernel);
        let straight = !case.name.contains("loop");
        for (level, line_info) in [(3, false), (0, false), (3, true), (2, false), (1, false)] {
            for &arch in *archs {
                let options = CompileOptions::new().gpu_arch(arch).opt_level(level).generate_line_info(line_info);
                let compiled = compile(&case.kernel, &tc, &options);
                let ptx = compiled.ptx();
                assert_eq!(compiled.ptx_version(), arch.min_ptx().max(*other_ptx), "{}", case.name);
                let what = format!("{} for {} at O{level}, line info {line_info}", case.name, arch.chip());
                let mut flag = false;
                let sequence = carry_sequence(ptx);
                for item in &sequence {
                    match item {
                        None => flag = false,
                        Some(op) => {
                            let (reads, writes) = carry_use(op).unwrap();
                            assert!(!reads || flag, "{what}: {op} reads a carry flag nothing set:\n{ptx}");
                            flag |= writes;
                        }
                    }
                }
                let ops: Vec<String> = sequence.into_iter().flatten().collect();
                if straight {
                    assert_eq!(ops, expected, "{what}:\n{ptx}");
                } else {
                    // A loop's chains may be unrolled, but each stays whole
                    // (as checked above) and none is lost.
                    for op in &expected {
                        let n = |list: &[String]| list.iter().filter(|o| *o == op).count();
                        assert!(n(&ops) >= n(&expected), "{what}: {ops:?}");
                    }
                }
            }
        }
    }
}

/// The warp collectives keep the carry flag (`cute::ir::carry_flag_breaker`): in the PTX of
/// `feature_kernels::carry_around_collectives`, at O0 to O3 and with line information, each of
/// its 15 `shfl.sync`, `redux.sync`, `match.sync`, `vote.sync` and `bar.warp.sync` instructions
/// sits inside a carry chain, after a carry-out and before the carry-in that ends the chain, with
/// no label there; the shuffles with a runtime segment mask and clamp take it in a register.
#[test]
fn carry_chains_run_through_warp_collectives() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::carry_around_collectives().kernel;
    let collectives = ["shfl.sync.", "redux.sync.", "match.", "vote.sync.", "bar.warp.sync"];
    for (level, line_info) in [(3, false), (0, false), (3, true), (2, false), (1, false)] {
        for arch in [Arch::Sm80, Arch::Sm89, Arch::Sm90] {
            let options = CompileOptions::new().gpu_arch(arch).opt_level(level).generate_line_info(line_info);
            let compiled = compile(&kernel, &tc, &options);
            let ptx = compiled.ptx();
            let what = format!("{} at O{level}, line info {line_info}", arch.chip());
            let (mut chain, mut inside, mut runtime_clamps) = (false, 0, 0);
            for line in ptx.lines().map(str::trim) {
                if line.ends_with(':') && !line.starts_with("//") {
                    assert!(!chain, "{what}: a label inside a carry chain:\n{ptx}");
                    continue;
                }
                let Some(op) = opcode(line) else { continue };
                if let Some((_, writes)) = carry_use(op) {
                    chain = writes;
                } else if collectives.iter().any(|c| op.starts_with(c)) {
                    assert!(chain, "{what}: {line} is outside the carry chains:\n{ptx}");
                    inside += 1;
                    let rest = &line[line.find(op).unwrap() + op.len()..];
                    let operands: Vec<&str> = rest.trim().trim_end_matches(';').split(", ").collect();
                    if op.starts_with("shfl.sync.") && operands[3].starts_with("%r") {
                        runtime_clamps += 1;
                    }
                }
            }
            assert_eq!((inside, runtime_clamps), (15, 3), "{what}:\n{ptx}");
        }
    }
}

// ---------------------------------------------------------------------------
// Hand-written PTX and MLIR.
// ---------------------------------------------------------------------------

/// The `llvm.inline_asm` lines of a kernel's MLIR, as (side effects,
/// template, constraints).
fn asm_statements(kernel: &Kernel) -> Vec<(bool, String, String)> {
    kernel
        .to_mlir()
        .lines()
        .filter(|l| l.contains("llvm.inline_asm"))
        .map(|l| {
            let quoted: Vec<&str> = l.split('"').collect();
            (l.contains("llvm.inline_asm has_side_effects"), quoted[1].to_string(), quoted[3].to_string())
        })
        .collect()
}

/// CuTe DSL's named operands become LLVM's positional ones: the write-only
/// outputs, then the read-write ones, then the read-only inputs, the
/// read-write operands' tied inputs and the predicate (`@$n`, constraint
/// `b`), each in its type's register class (`f` for Float32 constants, never
/// the immediate `n`; `b` for a Boolean constant), except that an integer
/// constant is the immediate `n`, as CuTe DSL's NVVM lowering makes it, with
/// a literal `$` escaped. Statements have side effects unless marked pure,
/// and a guarded write-only output is selected against 0. The PTX has each
/// instruction as written, with the immediates as LLVM prints them.
#[test]
fn inline_ptx_named_operands_become_positional() {
    let kernel = feature_kernels::inline_ptx_named().kernel;
    let asm = asm_statements(&kernel);
    let want: [(bool, &str, &str); 12] = [
        (false, "mov.b64 {$0, $1}, $2;", "=r,=r,l"),
        (true, "add.u32 $1, $1, $2;\\0A\\09shf.l.wrap.b32 $0, $3, $3, $2;", "=r,=r,r,r,1"),
        (true, "@$3 max.u32 $0, $0, $1;", "=r,r,0,b"),
        (true, "@$3 xor.b32 $0, $1, $2;", "=r,r,r,b"),
        (false, "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f"),
        (true, "{ .reg .b16 t$$h; add.u16 t$$h, $1, $2; mov.b16 $0, t$$h; }", "=h,h,n"),
        (true, "selp.b32 $0, $1, $2, $3;", "=r,r,r,b"),
        (false, "lop3.b32 $0, $1, $2, $3, $4;", "=r,r,r,r,n"),
        (false, "add.u32 $0, $1, $2;", "=r,r,n"),
        (false, "xor.b64 $0, $1, $2;", "=l,l,n"),
        (false, "selp.b32 $0, $1, $2, $3;", "=r,r,r,b"),
        (true, "prefetch.global.L2 [$0];", "l"),
    ];
    let got: Vec<(bool, &str, &str)> = asm.iter().map(|(e, t, c)| (*e, t.as_str(), c.as_str())).collect();
    assert_eq!(got, want);
    let mlir = kernel.to_mlir();
    assert_eq!(mlir.lines().filter(|l| l.contains("= arith.select") && l.contains("%unguarded")).count(), 1, "{mlir}");
    // The prefetch's address is computed as the mbarrier operations' are.
    assert!(mlir.contains("memref.extract_aligned_pointer_as_index %arg_xs"), "{mlir}");
    let Some(tc) = toolchain() else { return };
    let p = ptx(&kernel, &tc);
    assert!(p.contains(".version 7.0"), "{p}");
    let compact: Vec<String> = p.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
    for want in [
        "mov.b64 {%r",
        "shf.l.wrap.b32 %r",
        "fma.rn.f32 ",
        "{ .reg .b16 t$h; add.u16 t$h, %rs",
        "selp.b32 %r",
        "prefetch.global.L2 [%rd",
    ] {
        assert!(compact.iter().any(|l| l.starts_with(want)), "no {want} in\n{p}");
    }
    // Integer constants are immediates, printed sign-extended; the Float32
    // and Boolean constants are registers.
    for (op, immediate) in
        [("lop3.b32 %r", ", 150;"), ("add.u32 %r", ", -16;"), ("xor.b64 %rd", ", -9223372036854775807;")]
    {
        assert!(compact.iter().any(|l| l.starts_with(op) && l.ends_with(immediate)), "no {op}..{immediate} in\n{p}");
    }
    assert!(compact.iter().any(|l| l.contains("add.u16 t$h, %rs") && l.contains(", 7;")), "{p}");
    assert!(compact.iter().all(|l| !l.starts_with("fma.rn.f32") || l.matches('%').count() == 4), "{p}");
    assert!(compact.iter().filter(|l| l.starts_with("selp.b32 %r")).all(|l| l.contains(", %p")), "{p}");
    for guarded in ["max.u32 %r", "xor.b32 %r"] {
        assert!(compact.iter().any(|l| l.starts_with("@%p") && l.contains(guarded)), "no guarded {guarded} in\n{p}");
    }
    for arch in [Arch::Sm86, Arch::Sm89, Arch::Sm90a, Arch::Sm100a, Arch::Sm120] {
        kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
    }
}

/// Raw inline assembly keeps its template and constraint string, with `+r`
/// as an output plus an input tied to it (appended after the other inputs),
/// an immediate as an immediate, and a memory clobber, which also gives the
/// statement side effects (MLIR would otherwise delete a store whose
/// statement has no outputs), whether `.memory_clobber()` adds it or the
/// constraint string has it: in the PTX, the store through the pointer
/// sits between the ordinary store and the reload of the same word. A
/// shared-memory address is a 32-bit register in inline PTX (and in raw
/// assembly that asks for `r`), a global one 64-bit, and a constant offset
/// in an address an immediate.
#[test]
fn inline_asm_keeps_its_constraints() {
    let raw = asm_statements(&feature_kernels::inline_asm_raw().kernel);
    let want: [(bool, &str, &str); 4] = [
        (false, "add.u32 $0, $0, $2;\\0A\\09mul.lo.u32 $1, $2, $2;", "=r,=r,r,0"),
        (false, "sub.u32 $0, $0, $1;", "=r,r,0"),
        (false, "shl.b64 $0, $1, $2;", "=l,l,n"),
        (true, "fma.rn.f64 $0, $1, $2, $3;", "=d,d,d,d,~{memory}"),
    ];
    let got: Vec<(bool, &str, &str)> = raw.iter().map(|(e, t, c)| (*e, t.as_str(), c.as_str())).collect();
    assert_eq!(got, want);
    let pointers = feature_kernels::inline_ptx_pointers().kernel;
    let asm = asm_statements(&pointers);
    let want: [(bool, &str, &str); 8] = [
        (true, "ld.global.nc.u32 $0, [$1];", "=r,l"),
        (true, "ld.shared.u64 $0, [$1];", "=l,r"),
        (true, "@$2 st.global.u64 [$0], $1;", "l,l,b"),
        (true, "st.shared.u32 [$0], $1;", "l,r,~{memory}"),
        (true, "ld.u32 $0, [$1];", "=r,l"),
        (true, "sub.u32 $0, $2, $1;", "=r,r,r"),
        (true, "ld.global.nc.u32 $0, [$1+$2];", "=r,l,n"),
        (true, "st.shared.u32 [$0], $1;", "r,r,~{memory}"),
    ];
    let got: Vec<(bool, &str, &str)> = asm.iter().map(|(e, t, c)| (*e, t.as_str(), c.as_str())).collect();
    assert_eq!(got, want);
    let Some(tc) = toolchain() else { return };
    let p = ptx(&feature_kernels::inline_asm_raw().kernel, &tc);
    assert!(p.lines().any(|l| l.trim_start().starts_with("shl.b64") && l.trim_end().ends_with(", 7;")), "{p}");
    for level in [0, 3] {
        let p = compile(&pointers, &tc, &CompileOptions::new().gpu_arch(Arch::Sm80).opt_level(level)).into_ptx();
        let lines: Vec<&str> = p.lines().map(str::trim).collect();
        let asm_store = lines.iter().position(|l| l.starts_with("st.shared.u32")).unwrap_or_else(|| panic!("{p}"));
        let address = lines[asm_store].split(['[', ']']).nth(1).unwrap();
        let before =
            lines[..asm_store].iter().rposition(|l| l.starts_with("st.shared.b32") || l.starts_with("st.shared.u32"));
        let after =
            lines[asm_store..].iter().position(|l| l.starts_with("ld.shared.b32") || l.starts_with("ld.shared.u32"));
        assert!(before.is_some() && after.is_some(), "O{level}: the ordinary store and load around the asm:\n{p}");
        if level == 3 {
            assert!(lines[before.unwrap()].contains(&format!("[{address}]")), "O{level}:\n{p}");
        }
        for want in ["ld.global.nc.u32 %r", "ld.shared.u64 %rd"] {
            assert!(lines.iter().any(|l| l.starts_with(want)), "O{level}: no {want} in\n{p}");
        }
        assert!(lines.iter().any(|l| l.starts_with("@%p") && l.contains("st.global.u64 [")), "O{level}:\n{p}");
        // The shared addresses are 32-bit registers, the global one with a
        // constant offset an immediate.
        assert!(lines.iter().any(|l| l.starts_with("ld.shared.u64 %rd") && l.contains("[%r")), "O{level}:\n{p}");
        assert!(lines.iter().any(|l| l.starts_with("sub.u32 %r")), "O{level}:\n{p}");
        assert!(lines.iter().any(|l| l.starts_with("ld.global.nc.u32 %r") && l.ends_with("+4];")), "O{level}:\n{p}");
        // The store whose clobber is in its constraint string is kept, and
        // the word it wrote is loaded again after it.
        let marked = lines.iter().rposition(|l| l.starts_with("st.shared.u32 [%r")).unwrap_or_else(|| panic!("{p}"));
        assert!(
            lines[marked..].iter().any(|l| l.starts_with("ld.shared.b32") || l.starts_with("ld.shared.u32")),
            "O{level}: the reload after the clobber:\n{p}"
        );
    }
}

/// A hand-written MLIR operation is emitted as written, with its operands
/// and results named, and compiles: `nvvm.read.ptx.sreg.nsmid` reads
/// `%nsmid`, LLVM's byte swap and funnel shift become `prmt` and `shf`, and
/// `arith.mului_extended` gives both halves of the product (one
/// `mul.wide.u32`). A pointer operand is an LLVM pointer in its buffer's
/// address space, made from its address: `nvvm.cp.async` copies from
/// global into shared memory, and loads and stores go through global and
/// generic pointers.
#[test]
fn mlir_ops_are_emitted_as_written() {
    let ops = feature_kernels::mlir_ops().kernel;
    let mlir = ops.to_mlir();
    let lines: Vec<&str> = mlir.lines().map(str::trim).collect();
    let find = |op: &str| *lines.iter().find(|l| l.contains(op)).unwrap_or_else(|| panic!("no {op} in\n{mlir}"));
    let bswap = find("llvm.intr.bswap(");
    assert!(bswap.starts_with("%v") && bswap.contains(" = llvm.intr.bswap(%v") && bswap.ends_with(": (i32) -> i32"));
    let extended = find("arith.mului_extended");
    assert_eq!(extended.split(" = ").next().unwrap().split(", ").count(), 2, "{extended}");
    assert!(find("llvm.intr.fshl(").contains("(i32, i32, i32) -> i32"));
    let nsmid = feature_kernels::mlir_nsmid().kernel;
    assert!(nsmid.to_mlir().contains(" = nvvm.read.ptx.sreg.nsmid : i32"));
    let Some(tc) = toolchain() else { return };
    let p = ptx(&ops, &tc);
    for want in ["prmt.b32", "shf.l.wrap.b32", "mul.wide.u32"] {
        assert!(p.contains(want), "no {want} in\n{p}");
    }
    let p = ptx(&nsmid, &tc);
    assert_eq!(count(&p, "%nsmid"), 1, "the two reads of %nsmid should be one (both are pure reads):\n{p}");
    let pointers = feature_kernels::mlir_pointers().kernel;
    let mlir = pointers.to_mlir();
    let made: Vec<&str> = mlir
        .lines()
        .filter(|l| l.contains("= llvm.inttoptr %sum"))
        .map(|l| l.split(" to ").nth(1).unwrap().split_whitespace().next().unwrap())
        .collect();
    for ty in ["!llvm.ptr<1>", "!llvm.ptr<3>", "!llvm.ptr"] {
        assert!(made.contains(&ty), "no pointer of type {ty} in {made:?}:\n{mlir}");
    }
    assert!(mlir.contains("nvvm.cp.async.shared.global %ptr"), "{mlir}");
    let p = ptx(&pointers, &tc);
    let compact = p.split_whitespace().collect::<Vec<_>>().join(" ");
    for want in ["cp.async.ca.shared.global [", "cp.async.commit_group;", "cp.async.wait_group 0;"] {
        assert!(compact.contains(want), "no {want} in\n{p}");
    }
    assert!(p.contains("ld.global.u32") || p.contains("ld.global.b32"), "{p}");
    assert!(p.contains("st.global.u32") || p.contains("st.global.b32"), "{p}");
    for arch in [Arch::Sm89, Arch::Sm90a, Arch::Sm100a, Arch::Sm120] {
        pointers.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
    }
}

/// An error of LLVM's that `mlir-opt` prints fails the compile even when it
/// exits with success, as LLVM 23's does for an inline-assembly constraint
/// NVPTX does not know (it drops the statement from the PTX): on every
/// installed `mlir-opt` that compiles cute-rs's kernels (LLVM 21 and newer;
/// an older one, such as the pod's LLVM 19, is skipped), MLIR whose
/// `llvm.inline_asm` asks for the unknown constraint `x` (which the
/// validator would reject in a kernel) does not compile.
#[test]
fn mlir_opt_errors_fail_the_compile() {
    let kernel = Kernel::builder("constraint_x").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let y: Uint32 = InlineAsm::new("add.u32 $0, $1, $2;", "=r,r,r").output::<u32>().arg(x.load(t)).arg(t).emit();
        x.store(t, y);
    });
    let mlir = kernel.to_mlir();
    assert_eq!(count(&mlir, "\"=r,r,r\""), 1, "{mlir}");
    let broken = mlir.replace("\"=r,r,r\"", "\"=r,r,x\"");
    let mut checked = 0;
    for tc in installed_toolchains() {
        let who = tc.mlir_opt.display().to_string();
        if let Err(e) = tc.compile_ptx(&mlir, Arch::Sm80) {
            eprintln!("skipping {who}, which does not compile cute-rs's kernels: {e}");
            continue;
        }
        tc.verify(&broken).unwrap_or_else(|e| panic!("{who}: the MLIR itself is valid: {e}"));
        let e = tc.compile_ptx(&broken, Arch::Sm80).expect_err(&who);
        assert!(e.to_string().contains("unknown asm constraint 'x'"), "{who}: {e}");
        checked += 1;
    }
    assert!(checked > 0 || Toolchain::discover().is_none(), "no installed mlir-opt compiled the kernel");
}

/// Each carry instruction is the inline assembly CuTe DSL emits for it, with
/// the same template, constraint string and side effects (`add.cc.u32 $0,
/// $1, $2;` with `=r,r,r`, `mad.lo.cc.s64 $0, $1, $2, $3;` with `=l,l,l,l`,
/// ...), the carry-in-and-out forms spelled the same way.
#[test]
fn carry_instructions_match_cute_dsl() {
    let carry = |kernel: &Kernel| -> Vec<(bool, String, String)> {
        asm_statements(kernel).into_iter().filter(|(_, t, _)| opcode(t).and_then(carry_use).is_some()).collect()
    };
    let spelled = |name: &str, register: &str| {
        let arity = if name.starts_with("mad") { 3 } else { 2 };
        let operands: Vec<String> = (0..=arity).map(|k| format!("${k}")).collect();
        let constraints: Vec<String> =
            std::iter::once(format!("={register}")).chain((0..arity).map(|_| register.to_string())).collect();
        (true, format!("{name} {};", operands.join(", ")), constraints.join(","))
    };
    for (case, w, register) in
        [(feature_kernels::carry_words_u32(), 32, "r"), (feature_kernels::carry_words_u64(), 64, "l")]
    {
        let names = [
            "add.cc", "addc.cc", "addc.cc", "addc.cc", "addc", "sub.cc", "subc.cc", "subc.cc", "subc.cc", "subc",
            "add.cc", "addc", "addc", "sub.cc", "subc", "subc",
        ];
        let want: Vec<_> = names.iter().map(|n| spelled(&format!("{n}.u{w}"), register)).collect();
        assert_eq!(carry(&case.kernel)[..want.len()], want[..], "{}", case.name);
    }
    let names = [
        ("mad.lo.cc.u32", "r"),
        ("madc.lo.cc.u32", "r"),
        ("addc.u32", "r"),
        ("mad.lo.cc.u32", "r"),
        ("addc.cc.u32", "r"),
        ("addc.u32", "r"),
        ("mad.lo.cc.s32", "r"),
        ("madc.lo.cc.s32", "r"),
        ("madc.lo.s32", "r"),
        ("mad.lo.cc.u64", "l"),
        ("madc.lo.u64", "l"),
        ("mad.lo.cc.s64", "l"),
        ("madc.lo.cc.s64", "l"),
        ("madc.lo.s64", "l"),
    ];
    let want: Vec<_> = names.iter().map(|(n, r)| spelled(n, r)).collect();
    assert_eq!(carry(&feature_kernels::carry_multiply().kernel), want);
}

/// `increment_coord` on runtime coordinates verifies, and its compares and
/// selects survive into the sm_80 PTX.
#[test]
fn increment_coord_kernel() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::increment_coords().kernel;
    let selects = kernel.ir().count_ops(|op| matches!(op, cute::ir::Op::Select { .. }));
    assert_eq!(selects, 22, "two selects for each runtime leaf that can wrap\n{}", kernel.dump());
    let p = ptx(&kernel, &tc);
    assert!(count(&p, "setp.") >= 4 && count(&p, "selp.b32") >= 8, "{p}");
}

/// Layouts with runtime extents divide by them at run time: the index
/// expressions of `dynamic_layouts` become unsigned divisions and remainders
/// by runtime values, which verify and survive into the sm_80 PTX (LLVM may
/// narrow them, here to 16 bits, as it knows the extents are small). The
/// composition whose right-hand extent is compared with a static one at run
/// time takes a minimum.
#[test]
fn dynamic_layout_kernel() {
    use cute::ir::{BinaryOp, Op};
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::dynamic_layouts().kernel;
    let insts = &kernel.ir().body.insts;
    let constants: Vec<_> =
        insts.iter().filter_map(|i| if let Op::Const { result, .. } = i.op { Some(result) } else { None }).collect();
    let by_runtime_value = |which: BinaryOp| {
        insts
            .iter()
            .filter(|i| matches!(&i.op, Op::Binary { op, rhs, .. } if *op == which && !constants.contains(rhs)))
            .count()
    };
    assert!(by_runtime_value(BinaryOp::DivU) >= 5 && by_runtime_value(BinaryOp::RemU) >= 4, "{}", kernel.dump());
    let mins = insts.iter().filter(|i| matches!(&i.op, Op::Binary { op: BinaryOp::Min, .. })).count();
    assert!(mins >= 1, "{}", kernel.dump());
    let p = ptx(&kernel, &tc);
    assert!(count(&p, "div.u") >= 4 && count(&p, "rem.u") >= 4, "{p}");
}

/// Tensors of runtime extents: `dynamic_tiles` keeps its 128-bit loads and
/// stores (as many of each as LLVM unrolls the loop over the runtime number
/// of tiles), guarded by compares with the runtime extents; and
/// `runtime_gemm` with `M`, `N` and `K` as runtime extents (`build_dynamic`)
/// issues the static-bound kernel's tensor-core, `ldmatrix` and `cp.async`
/// instructions, in no more PTX.
#[test]
fn runtime_extent_kernels() {
    let Some(tc) = toolchain() else { return };
    let tiles = ptx(&feature_kernels::dynamic_tiles().kernel, &tc);
    assert!(wide_accesses(&tiles, "ld") >= 1 && wide_accesses(&tiles, "st") >= 1, "{tiles}");
    assert_eq!(wide_accesses(&tiles, "ld"), wide_accesses(&tiles, "st"), "{tiles}");
    assert!(count(&tiles, "ld.global") == wide_accesses(&tiles, "ld"), "only 128-bit loads:\n{tiles}");
    // The bounds predicate orders Int32 coordinates against `m` and `n`: at
    // least one signed compare with each. The loop over the tiles counts
    // its unsigned trip count with equality and unsigned compares, so the
    // same kernel with `copy` for `copy_if` has no signed ordering compare.
    let signed = |cmp: &str| count(&tiles, &format!("setp.{cmp}.s"));
    let compares: usize = ["lt", "le", "gt", "ge"].into_iter().map(signed).sum();
    assert!(compares >= 2, "no bounds compares:\n{tiles}");
    let bounded = ptx(&runtime_gemm::build(), &tc);
    let dynamic = ptx(&runtime_gemm::build_dynamic(), &tc);
    for op in
        ["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", "ldmatrix.sync.aligned", "cp.async.cg.shared.global"]
    {
        assert_eq!(count(&dynamic, op), count(&bounded, op), "{op}");
        assert!(count(&dynamic, op) > 0, "{op}");
    }
    let lines = |p: &str| p.lines().count();
    assert!(
        lines(&dynamic) <= lines(&bounded),
        "{} lines with runtime extents, {} with static bounds",
        lines(&dynamic),
        lines(&bounded)
    );
}

/// The m5 ports, built once for runtime shapes, verify and compile for
/// sm_80. `elementwise_add.py`'s kernel moves one element per instruction
/// (nothing is known of the runtime row stride), bounds-checked against the
/// runtime extents. `sgemm.py`'s fills shared memory with `cp.async`: 16-byte
/// copies when `max_alignment` of the M-major A's and N-major B's mode 0 is
/// a multiple of 16, 4-byte ones otherwise and for K-major operands. Its
/// k-block loop carries `unroll = <full = true>` and LLVM unrolls it: the
/// main loop holds the 8 x 64 FMAs of a K tile and the per-thread fragments
/// live in registers, not local memory.
#[test]
fn m5_example_ports() {
    let Some(tc) = toolchain() else { return };
    let add = ptx(elementwise_add::dynamic().kernel(), &tc);
    assert_eq!(count(&add, "ld.global.v"), 0, "{add}");
    assert_eq!(count(&add, "ld.global.b32"), 32, "{add}");
    assert_eq!(count(&add, "st.global.b32"), 16, "{add}");
    for majors in [
        ampere_sgemm::Majors::new('m', 'n', 'n', 16),
        ampere_sgemm::Majors::new('m', 'n', 'm', 4),
        ampere_sgemm::Majors::new('k', 'k', 'n', 1),
    ] {
        let kernel = ampere_sgemm::build_for(majors);
        let mlir = kernel.to_mlir();
        assert_eq!(count(&mlir, "loop_annotation = #llvm.loop_annotation<unroll = <full = true>>"), 1, "{mlir}");
        let p = ptx(&kernel, &tc);
        let cp = |w: &str| p.lines().filter(|l| l.contains("cp.async.ca.shared.global") && l.contains(w)).count();
        // Three copies of A and of B: the residue tile, the prologue's
        // second tile and the main loop's.
        let want = if majors.divisibility == 16 { (6, 0) } else { (0, 24) };
        assert_eq!((cp(", 16,"), cp(", 4,")), want, "{majors:?}\n{p}");
        assert_eq!(count(&p, "fma.rn.f32"), 8 * 64, "{majors:?}\n{p}");
        assert_eq!(count(&p, ".local"), 0, "{majors:?}\n{p}");
        assert!(p.contains("bar.sync \t1, 256;") || p.contains("barrier.sync \t1, 256;"), "{majors:?}\n{p}");
    }
}

/// A kernel with a dynamic shared-memory region keeps all of its shared memory in the dynamic
/// window, an external array with no static shared memory (`gpu.dynamic_shared_memory` would
/// leave a byte of it), the build-time buffers first (at bytes 0 and 100) and the region at byte
/// 240; the size registers are pure `mov`s from `%dynamic_smem_size` and `%total_smem_size`, and
/// `dynamic_smem_size` and `get_dyn_smem_size` subtract the region's start. None needs more than
/// sm_80 and PTX 7.0.
#[test]
fn dynamic_shared_memory_region() {
    let kernel = feature_kernels::dynamic_region_kernel();
    let mlir = kernel.to_mlir();
    let globals: Vec<&str> = mlir.lines().filter(|l| l.contains("memref.global")).map(str::trim).collect();
    assert_eq!(globals.len(), 1, "{mlir}");
    assert!(
        globals[0].starts_with(
            "memref.global @__cute_dynamic_smem : memref<0xi8, #gpu.address_space<workgroup>> {alignment = 16 : i64}"
        ),
        "{mlir}"
    );
    assert!(!mlir.contains("gpu.dynamic_shared_memory"), "{mlir}");
    // Each buffer is a view of the window at its start: `%v = memref.view %dyn_smem[%c][]`.
    let constant = |name: &str| -> usize {
        let def = format!("{name} = arith.constant ");
        let line = mlir.lines().map(str::trim).find(|l| l.starts_with(&def)).expect("a constant");
        line[def.len()..].split_whitespace().next().unwrap().parse().unwrap()
    };
    let views: Vec<(&str, usize)> = mlir
        .lines()
        .filter_map(|l| {
            let (view, rest) = l.trim().split_once(" = memref.view %dyn_smem[")?;
            Some((view, constant(rest.split(']').next()?)))
        })
        .collect();
    assert_eq!(views, [("%smem0_i32", 0), ("%smem1_i16", 100), ("%dynamic_smem_i32", 240), ("%dynamic_smem_i8", 240)]);
    // `dynamic_smem_size` and `get_dyn_smem_size` are one read, as they are one register.
    let asm: Vec<&str> = mlir.lines().filter(|l| l.contains("llvm.inline_asm")).collect();
    assert_eq!(asm.len(), 2, "{mlir}");
    assert!(asm.iter().all(|l| !l.contains("has_side_effects")), "{asm:?}");
    let Some(tc) = toolchain() else { return };
    let p = ptx(&kernel, &tc);
    assert!(p.contains(".version 7.0"), "{p}");
    let lines: Vec<String> = p.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
    let shared: Vec<&String> = lines.iter().filter(|l| l.contains(".shared .align")).collect();
    assert_eq!(shared, [".extern .shared .align 16 .b8 __cute_dynamic_smem[];"], "{p}");
    for register in ["%dynamic_smem_size;", "%total_smem_size;"] {
        assert_eq!(lines.iter().filter(|l| l.starts_with("mov.u32 %r") && l.ends_with(register)).count(), 1, "{p}");
    }
    assert!(lines.iter().any(|l| l.starts_with("add.s32 %r") && l.ends_with(", -240;")), "{p}");
}

/// `block_idx_in_cluster` and `cluster_size` read `%cluster_ctarank` and `%cluster_nctarank`
/// through NVVM, and `aggr_smem_size` `%aggr_smem_size` through a pure `mov`: sm_90, and PTX 8.1
/// for `%aggr_smem_size`. Older targets are refused, naming the line.
#[test]
fn cluster_rank_and_size_registers() {
    let kernel = feature_kernels::cluster_ranks_kernel();
    let mlir = kernel.to_mlir();
    for op in ["nvvm.read.ptx.sreg.cluster.ctarank : i32", "nvvm.read.ptx.sreg.cluster.nctarank : i32"] {
        assert_eq!(mlir.matches(op).count(), 1, "{op}\n{mlir}");
    }
    let Some(tc) = toolchain() else { return };
    for arch in [Arch::Sm80, Arch::Sm89] {
        let e = kernel.compile_ptx(&tc, arch).unwrap_err().to_string();
        assert!(e.contains("the cluster special registers needs sm_90 or newer"), "{e}");
        assert!(e.contains("feature_kernels/mod.rs"), "{e}");
    }
    let p = kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.contains(".version 8.1"), "{p}");
    let lines: Vec<String> = p.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
    for register in ["%cluster_ctarank;", "%cluster_nctarank;", "%aggr_smem_size;", "%total_smem_size;"] {
        assert_eq!(lines.iter().filter(|l| l.starts_with("mov.u32 %r") && l.ends_with(register)).count(), 1, "{p}");
    }
}

/// `cp_async_mbarrier_arrive_noinc` and `cp_async_mbarrier_arrive` call the LLVM intrinsics,
/// which LLVM 21 to 23 print as the PTX 7.0 form (`.shared`, not `.shared::cta`), so they compile
/// for every target from sm_80.
#[test]
fn cp_async_mbarrier_arrive_instructions() {
    let kernel = feature_kernels::cp_async_mbarrier().kernel;
    let mlir = kernel.to_mlir();
    for (name, n) in
        [("llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared", 2), ("llvm.nvvm.cp.async.mbarrier.arrive.shared", 1)]
    {
        assert_eq!(mlir.matches(&format!("llvm.call_intrinsic \"{name}\"")).count(), n, "{name}\n{mlir}");
    }
    let Some(tc) = toolchain() else { return };
    let p = ptx(&kernel, &tc);
    assert!(p.contains(".version 7.1"), "test_wait.parity needs PTX 7.1:\n{p}");
    let lines: Vec<String> = p.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
    for (prefix, n) in
        [("cp.async.mbarrier.arrive.noinc.shared.b64 [", 2), ("cp.async.mbarrier.arrive.shared.b64 [", 1)]
    {
        assert_eq!(lines.iter().filter(|l| l.starts_with(prefix)).count(), n, "{prefix}\n{p}");
    }
    for arch in Arch::ALL.into_iter().filter(|&a| emittable(&tc, a)) {
        kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
    }
}

/// The special-register reads and `cp.async.mbarrier.arrive` keep the carry flag too
/// (`cute::ir::carry_flag_breaker`): in the PTX of `feature_kernels::carry_around_smem_kernel`,
/// whose region starts behind its buffers sized at build time, at O0 to O3, with and without line
/// information, no label, branch or call sits inside a carry chain, and the reads of
/// `%dynamic_smem_size` (which `get_dyn_smem_size` shares), `%total_smem_size` and
/// `%aggr_smem_size` (pure inline PTX) and both `cp.async.mbarrier.arrive` instructions each sit
/// inside one, after a carry-out and before the carry-in that ends it. At O0 so do
/// `dynamic_smem_size`'s subtraction of the region's start and the reads of `%cluster_ctarank`
/// and `%cluster_nctarank` (NVVM); from O1 on LLVM may move those to their uses past the chain,
/// which leaves it whole.
#[test]
fn carry_chains_run_through_smem_registers() {
    let Some(tc) = toolchain() else { return };
    let start = feature_kernels::SMEM_CARRY_REGION_START;
    let subtraction = format!(", -{start};");
    // What the chains run through, and whether LLVM may move it out of them from O1 on.
    let sizes = [
        ("%dynamic_smem_size;", false),
        (subtraction.as_str(), true),
        ("cp.async.mbarrier.arrive.noinc.shared.b64", false),
        ("%total_smem_size;", false),
        ("cp.async.mbarrier.arrive.shared.b64", false),
    ];
    let cluster = [("%cluster_ctarank;", true), ("%cluster_nctarank;", true), ("%aggr_smem_size;", false)];
    let cases = [
        (feature_kernels::carry_around_smem_registers(), &[Arch::Sm80, Arch::Sm89, Arch::Sm90][..], &[][..]),
        (feature_kernels::carry_around_smem_registers_sm90(), &[Arch::Sm90][..], &cluster[..]),
    ];
    for (case, archs, extra) in &cases {
        assert_eq!(case.kernel.ir().shared_memory_layout().dynamic_start, start as usize, "{}", case.name);
        for level in 0..=3 {
            let mut want: Vec<&str> =
                sizes.iter().chain(*extra).filter(|(_, movable)| level == 0 || !movable).map(|(w, _)| *w).collect();
            want.sort_unstable();
            for line_info in [false, true] {
                for &arch in *archs {
                    let options = CompileOptions::new().gpu_arch(arch).opt_level(level).generate_line_info(line_info);
                    let compiled = compile(&case.kernel, &tc, &options);
                    let ptx = compiled.ptx();
                    let what = format!("{} for {} at O{level}, line info {line_info}", case.name, arch.chip());
                    let (mut chain, mut inside) = (false, Vec::new());
                    for line in ptx.lines().map(str::trim) {
                        if line.ends_with(':') && !line.starts_with("//") {
                            assert!(!chain, "{what}: a label inside a carry chain:\n{ptx}");
                            continue;
                        }
                        let Some(op) = opcode(line) else { continue };
                        if let Some((_, writes)) = carry_use(op) {
                            chain = writes;
                        } else if ["bra", "brx", "call", "ret", "exit"].iter().any(|b| op.starts_with(b)) {
                            assert!(!chain, "{what}: {line} inside a carry chain:\n{ptx}");
                        } else if let Some(&name) = want.iter().find(|w| line.contains(*w)) {
                            assert!(chain, "{what}: {line} is outside the carry chains:\n{ptx}");
                            inside.push(name);
                        }
                    }
                    inside.sort_unstable();
                    assert_eq!(inside, want, "{what}:\n{ptx}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Memory-ordered and cache-hinted loads and stores, prefetch.
// ---------------------------------------------------------------------------

/// Whether a line of `ptx` contains `pattern`.
fn has(ptx: &str, pattern: &str) -> bool {
    count(ptx, pattern) > 0
}

/// The lines of `ptx` that contain `pattern` and do not directly follow a
/// `fence.sc` (which a seq_cst access puts before its acquire load or
/// release store), not counting blank and comment lines.
fn count_unfenced(ptx: &str, pattern: &str) -> usize {
    let lines: Vec<&str> = ptx.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with("//")).collect();
    (0..lines.len())
        .filter(|&k| lines[k].contains(pattern) && (k == 0 || !lines[k - 1].starts_with("fence.sc")))
        .count()
}

/// The hand-offs' ordered accesses: a scalar ordered access without hints is
/// an LLVM atomic or volatile access with a sync scope, anything else one PTX
/// instruction of inline assembly with side effects, and each becomes the
/// PTX instruction it names. Loads in spin loops may be duplicated by loop
/// rotation, so only stores are counted exactly.
#[test]
fn ordered_accesses() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::ordered_handoffs().kernel;
    let mlir = kernel.to_mlir();
    for want in [
        "atomic syncscope(\"block\") release {alignment = 4 : i64} : i32, !llvm.ptr<3>",
        "atomic syncscope(\"block\") acquire {alignment = 4 : i64} : !llvm.ptr<3> -> i32",
        "atomic syncscope(\"device\") release {alignment = 4 : i64} : i32, !llvm.ptr<1>",
        "atomic syncscope(\"device\") acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32",
        "atomic syncscope(\"device\") seq_cst {alignment = 4 : i64} : !llvm.ptr<1> -> i32",
        "llvm.store volatile",
        "llvm.inline_asm has_side_effects \"ld.acquire.gpu.global.L1::evict_last.b32 $0, [$1];\"",
        "llvm.inline_asm has_side_effects \"st.release.gpu.global.v2.b32 [$0], {$1, $2};\"",
    ] {
        assert!(mlir.contains(want), "no {want} in\n{mlir}");
    }
    let p = ptx(&kernel, &tc);
    assert!(p.contains(".version 7.4"), "the eviction priorities need PTX 7.4\n{p}");
    for (pattern, n) in [
        ("st.release.cta.shared.b32", 1),
        // The release store and the seq_cst one.
        ("st.release.gpu.global.b32", 2),
        ("st.relaxed.cta.shared.b32", 1),
        ("fence.acq_rel.cta;", 4),
        ("st.volatile.shared.b32", 1),
        ("st.release.gpu.global.v2.b32", 1),
        ("st.release.gpu.global.L1::no_allocate.b32", 1),
    ] {
        assert_eq!(count(&p, pattern), n, "{pattern}\n{p}");
    }
    for pattern in [
        "ld.acquire.cta.shared.b32",
        "ld.acquire.gpu.global.b32",
        "ld.relaxed.cta.shared.b32",
        "fence.sc.gpu;",
        "ld.volatile.shared.b32",
        "ld.acquire.gpu.global.v2.b32",
        "ld.acquire.gpu.global.L1::evict_last.b32",
    ] {
        assert!(has(&p, pattern), "no {pattern} in\n{p}");
    }
    // seq_cst is a fence before the access: one for the store, and one before each load.
    assert!(count(&p, "fence.sc.gpu;") >= 2, "{p}");
    // The seq_cst spin loads are acquire loads too: the plain acquire spin
    // on the global flag is one that no fence.sc precedes.
    assert!(count_unfenced(&p, "ld.acquire.gpu.global.b32") >= 1, "{p}");
    assert_eq!(count_unfenced(&p, "st.release.gpu.global.b32"), 1, "{p}");
    // Cluster scope needs sm_90 and PTX 7.8.
    let cluster = feature_kernels::cluster_scope_handoff().kernel;
    for arch in [Arch::Sm80, Arch::Sm89] {
        let e = cluster.compile_ptx(&tc, arch).unwrap_err().to_string();
        assert!(e.contains("an ordered load or store with cluster scope needs sm_90 or newer"), "{e}");
    }
    let p = cluster.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    for pattern in [
        "st.release.cluster.shared.b32",
        "ld.acquire.cluster.shared.b32",
        "st.release.cluster.global.L1::no_allocate.v2.b32",
        "ld.acquire.cluster.global.L1::evict_last.v2.b32",
        "ld.relaxed.cluster.shared.b32",
    ] {
        assert!(has(&p, pattern), "no {pattern} in\n{p}");
    }
}

/// Every cache hint reaches the PTX as the qualifier it names, in the order
/// `ptxas` takes, on scalars, vectors and packed narrow elements; MMIO
/// raises the PTX version to 8.2 on every target.
#[test]
fn cache_hinted_accesses() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::cache_hints().kernel;
    let p = ptx(&kernel, &tc);
    assert!(p.contains(".version 8.2"), "MMIO needs PTX 8.2\n{p}");
    for pattern in [
        "createpolicy.fractional.L2::evict_last.b64",
        "ld.global.ca.b32",
        "ld.global.cg.b32",
        "ld.global.cs.b32",
        "ld.global.lu.b32",
        "ld.global.cv.b32",
        "st.global.wb.b32",
        "st.global.cg.b32",
        "st.global.cs.b32",
        "st.global.wt.b32",
        "ld.global.L1::evict_normal.b32",
        "ld.global.L1::evict_first.b32",
        "ld.global.L1::evict_last.b32",
        "ld.global.L1::evict_unchanged.b32",
        "ld.global.L1::no_allocate.b32",
        "st.global.L1::evict_normal.b32",
        "st.global.L1::evict_first.b32",
        "st.global.L1::evict_last.b32",
        "st.global.L1::evict_unchanged.b32",
        "st.global.L1::no_allocate.b32",
        "ld.global.L2::64B.b32",
        "ld.global.L2::128B.b32",
        "ld.global.L2::256B.b32",
        "ld.global.cs.L2::128B.b32",
        "ld.global.L2::cache_hint.b32",
        "st.global.L2::cache_hint.b32",
        "st.release.gpu.global.L2::cache_hint.b32",
        "st.relaxed.sys.global.L1::no_allocate.b32",
        "ld.mmio.relaxed.sys.global.b32",
        "st.mmio.relaxed.sys.global.b32",
        "st.release.gpu.global.L1::evict_first.b32",
        "ld.global.L1::evict_last.L2::cache_hint.L2::64B.b32",
        "ld.relaxed.gpu.global.L1::evict_first.L2::256B.b32",
        "ld.acquire.sys.global.L2::cache_hint.b32",
        "ld.acquire.gpu.global.L1::evict_last.b32",
        "ld.volatile.global.L2::128B.b32",
        "ld.volatile.global.L2::64B.v4.b32",
        "st.volatile.global.v4.b32",
        "ld.relaxed.cta.global.L1::no_allocate.v2.b32",
        "st.global.cs.v2.b32",
        "ld.acquire.gpu.global.L2::cache_hint.v2.b32",
        "st.release.gpu.global.L1::evict_last.L2::cache_hint.v2.b32",
        "ld.global.cg.v4.b32",
        "fence.sc.sys;",
        "st.release.sys.global.v4.b32",
        "st.shared.wt.b32",
        "ld.shared.cs.b32",
        "st.volatile.shared.v2.b32",
        "ld.volatile.shared.v2.b32",
        // Weak accesses: eight halves and sixteen bytes in four words, two
        // bytes in one.
        "ld.global.L1::evict_last.v4.b32",
        "st.global.cs.v4.b32",
        "st.global.wt.v4.b32",
        "ld.global.cg.b16",
        "st.global.L1::evict_first.b16",
        "ld.global.L1::evict_last.b8",
        "st.global.cs.b8",
        "ld.global.L1::no_allocate.v2.b64",
        "st.global.wt.v2.b64",
        "ld.global.L2::cache_hint.b64",
        "ld.global.L2::256B.v4.b32",
        "st.global.L1::evict_unchanged.v4.b32",
        "ld.acquire.gpu.global.L1::evict_first.b16",
        "st.relaxed.gpu.global.L1::evict_last.b16",
        "ld.acquire.sys.global.L1::evict_first.b8",
        "st.release.gpu.global.L1::no_allocate.b8",
        // The LLVM atomic accesses of the other types.
        "ld.relaxed.gpu.global.b16",
        "st.release.gpu.global.b16",
        "ld.acquire.gpu.global.b64",
        "st.release.gpu.global.b64",
        "ld.volatile.global.b64",
        "st.relaxed.sys.global.b64",
        "ld.relaxed.gpu.global.b8",
        "st.relaxed.cta.global.b8",
        // Scalar constant (non-coherent) loads.
        "ld.global.nc.b16",
        "ld.global.cs.nc.L2::64B.b8",
    ] {
        assert!(has(&p, pattern), "no {pattern} in\n{p}");
    }
    // Strong vectors of narrow elements keep their element type, one
    // instruction per four elements (the sixteen volatile bytes take four).
    for (pattern, n) in [
        ("ld.volatile.global.v4.b8", 4),
        ("ld.acquire.gpu.global.v4.b8", 1),
        ("st.relaxed.cta.global.v4.b8", 1),
        ("ld.relaxed.gpu.global.L1::evict_last.v2.b16", 1),
        ("st.release.gpu.global.v2.b16", 1),
        ("ld.volatile.global.v2.b8", 1),
        ("st.release.gpu.global.v2.b8", 1),
    ] {
        assert_eq!(count(&p, pattern), n, "{pattern}\n{p}");
    }
    assert_eq!(count_unfenced(&p, "st.release.gpu.global.v2.b8"), 0, "the seq_cst store follows a fence\n{p}");
    for kernel in [
        feature_kernels::cache_hints().kernel,
        feature_kernels::ordered_handoffs().kernel,
        feature_kernels::strong_vectors().kernel,
    ] {
        let need = cute::validate::check_target(kernel.ir(), Arch::Sm80).unwrap();
        for arch in [Arch::Sm80, Arch::Sm86, Arch::Sm89, Arch::Sm90, Arch::Sm90a, Arch::Sm100a, Arch::Sm120a] {
            let compiled = compile(&kernel, &tc, &CompileOptions::new().gpu_arch(arch));
            assert_eq!(compiled.ptx_version(), need.max(arch.min_ptx()), "{}: {arch:?}", kernel.name());
        }
    }
}

/// Each prefetch form is one PTX instruction: to L1 or L2, with an L2
/// eviction priority (PTX 7.4), predicated, uniform, and generic for
/// per-thread memory.
#[test]
fn prefetch_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::prefetches().kernel;
    let p = ptx(&kernel, &tc);
    assert!(p.contains(".version 7.4"), "{p}");
    let compact = |l: &str| l.split_whitespace().collect::<Vec<_>>().join(" ");
    let lines: Vec<String> = p.lines().map(compact).collect();
    let n = |prefix: &str| lines.iter().filter(|l| l.starts_with(prefix)).count();
    assert_eq!(n("prefetch.global.L1 [%rd"), 2, "{p}");
    assert_eq!(n("prefetch.global.L2 [%rd"), 1, "{p}");
    assert_eq!(n("prefetch.global.L2::evict_last [%rd"), 1, "{p}");
    assert_eq!(n("prefetch.global.L2::evict_normal [%rd"), 1, "{p}");
    assert_eq!(n("prefetchu.L1 [%rd"), 1, "{p}");
    assert_eq!(n("@%p"), 2, "{p}");
    assert!(lines.iter().any(|l| l.starts_with("@%p") && l.contains("prefetch.global.L2 [")), "{p}");
    assert!(lines.iter().any(|l| l.starts_with("@%p") && l.contains("prefetch.L2 [")), "{p}");
    assert_eq!(n("prefetch.L1 [%rd"), 1, "{p}");
    let mlir = kernel.to_mlir();
    let asm = mlir.lines().filter(|l| l.contains("llvm.inline_asm has_side_effects") && l.contains("prefetch"));
    assert_eq!(asm.count(), 9, "{mlir}");
}

/// The G2R and R2G atoms' copies carry their attributes into the PTX, and
/// tiled copies from `make_cotiled_copy` keep each thread's four values one
/// 128-bit access.
#[test]
fn global_copy_atom_instructions() {
    let Some(tc) = toolchain() else { return };
    let p = ptx(&feature_kernels::cotiled_copies().kernel, &tc);
    // Two tiles: one invariant load, one store of data and one of thread
    // indices per thread each.
    assert_eq!(count(&p, "ld.global.nc.v4.b32"), 2, "{p}");
    assert_eq!(count(&p, "st.global.v4."), 4, "{p}");
    assert_eq!(count(&p, "ld.global."), 2, "{p}");
    let p = ptx(&feature_kernels::global_copy_atoms().kernel, &tc);
    // Two 128-bit loads per thread for each acquire atom (o1, and orm into a
    // make_rmem_tensor) and for the seq_cst one (o5), whose loads follow a
    // fence.sc; eight 32-bit seq_cst stores per thread at CTA scope (o5).
    for (pattern, n) in [
        ("ld.acquire.gpu.global.v4.b32", 6),
        ("fence.sc.gpu;", 2),
        ("fence.sc.cta;", 8),
        ("st.release.cta.global.b32", 8),
        ("st.release.cta.global.v4.b32", 2),
        ("st.volatile.global.v4.b16", 2),
    ] {
        assert_eq!(count(&p, pattern), n, "{pattern}\n{p}");
    }
    assert_eq!(count_unfenced(&p, "ld.acquire.gpu.global.v4.b32"), 4, "{p}");
    assert_eq!(count_unfenced(&p, "st.release.cta.global.b32"), 0, "{p}");
    for pattern in [
        "ld.volatile.global.v4.b32",
        "st.release.gpu.global.L1::evict_last.v4.b32",
        "st.relaxed.cta.global.L1::no_allocate.v2.b32",
        "ld.global.cs.nc.L2::256B.v4.b32",
        "st.global.cs.v4.b32",
        "ld.global.nc.v4.b32",
        "st.global.wt.v4.b32",
        "ld.global.lu.L2::cache_hint.v2.b32",
        "st.global.cg.L2::cache_hint.v2.b32",
        "ld.mmio.relaxed.sys.global.b32",
        "st.mmio.relaxed.sys.global.b32",
        "ld.relaxed.gpu.global.b32",
        "st.release.gpu.global.b32",
        "ld.global.nc.L1::evict_first.v4.b32",
    ] {
        assert!(has(&p, pattern), "no {pattern} in\n{p}");
    }
    assert!(p.contains(".version 8.2"), "MMIO needs PTX 8.2\n{p}");
}

/// Strong vectors of 8- and 16-bit elements keep their element type in the
/// PTX, so each element is one access as the memory model counts them, and
/// take one instruction per four elements beyond four, with a seq_cst
/// access's fence once, before the first.
#[test]
fn strong_vector_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::strong_vectors().kernel;
    let mlir = kernel.to_mlir();
    let asm =
        "llvm.inline_asm has_side_effects \"ld.relaxed.gpu.global.v4.b16 {$0, $1, $2, $3}, [$4];\", \"=h,=h,=h,=h,l\"";
    assert_eq!(count(&mlir, asm), 2, "{mlir}");
    let asm = "llvm.inline_asm has_side_effects \"st.release.cta.shared.v4.b8 [$0], {$1, $2, $3, $4};\", \"l,c,c,c,c\"";
    assert_eq!(count(&mlir, asm), 3, "{mlir}");
    let p = ptx(&kernel, &tc);
    for (pattern, n) in [
        ("ld.relaxed.gpu.global.v4.b16", 2),
        ("st.release.gpu.global.v4.b16", 2),
        ("ld.acquire.cta.global.v4.b8", 4),
        ("fence.sc.gpu;", 1),
        ("st.release.gpu.global.v4.b8", 4),
        ("fence.sc.sys;", 1),
        ("ld.acquire.sys.global.v4.b8", 2),
        ("st.volatile.global.v4.b8", 2),
        ("ld.volatile.global.v4.b16", 1),
        ("st.relaxed.cta.global.v4.b16", 1),
        ("ld.acquire.gpu.global.L1::evict_first.L2::cache_hint.v2.b16", 1),
        ("st.relaxed.sys.global.L1::no_allocate.L2::cache_hint.v2.b16", 1),
        ("st.release.cta.shared.v4.b8", 3),
        ("ld.acquire.cta.shared.v4.b8", 2),
    ] {
        assert_eq!(count(&p, pattern), n, "{pattern}\n{p}");
    }
    // One fence for each seq_cst access, before its first instruction.
    assert_eq!(count_unfenced(&p, "st.release.gpu.global.v4.b8"), 3, "{p}");
    assert_eq!(count_unfenced(&p, "ld.acquire.sys.global.v4.b8"), 1, "{p}");
    // The flag's acquire is one byte, as are the elements of the release.
    assert!(has(&p, "ld.acquire.cta.shared.b8"), "{p}");
    // No strong access moves a packed word: every one is 8 or 16 bits.
    for line in p.lines().map(str::trim) {
        let op = line.split_whitespace().next().unwrap_or("");
        let strong = [".relaxed.", ".acquire.", ".release.", ".volatile."].iter().any(|q| op.contains(q));
        if (op.starts_with("ld.") || op.starts_with("st.")) && strong {
            assert!(op.ends_with(".b8") || op.ends_with(".b16"), "a strong access of a packed word: {line}\n{p}");
        }
    }
}

// ---------------------------------------------------------------------------
// Packed half-precision arithmetic, packed exp2, the narrow conversions and Pack/Unpack.
// ---------------------------------------------------------------------------

/// How many PTX lines are `instruction` (operands follow it).
fn instructions(lines: &[String], instruction: &str) -> usize {
    operands_of(lines, instruction).len()
}

/// The packed and scalar Float16 operations are one pure PTX instruction each, with the modifiers in the PTX ISA's
/// order; the predicated ones are `@p`-guarded, their result tied to the first operand (a `mov` of it in front).
/// Nothing needs more than sm_80 and PTX 7.0.
#[test]
fn half_f16_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::half_f16().kernel;
    let mlir = kernel.to_mlir();
    for line in mlir.lines().filter(|l| l.contains("llvm.inline_asm")) {
        assert!(!line.contains("has_side_effects"), "{line}");
    }
    assert!(mlir.contains("\"=r,r,r,b,0\""), "the predicated forms tie the result to the first operand\n{mlir}");
    let p = ptx(&kernel, &tc);
    assert!(p.contains(".version 7.0"), "{p}");
    let lines = compact_lines(&p);
    // Five sets of operands (four table rounds and the data); the default add and sub also appear in the tuple
    // form and under a predicate.
    for (instruction, n) in [
        ("add.f16x2", 5),
        ("add.rn.f16x2", 10),
        ("add.ftz.f16x2", 5),
        ("add.sat.f16x2", 5),
        ("add.rn.ftz.sat.f16x2", 5),
        ("sub.f16x2", 5),
        ("sub.rn.f16x2", 5),
        ("sub.ftz.f16x2", 5),
        ("sub.sat.f16x2", 5),
        ("sub.ftz.sat.f16x2", 5),
        ("fma.rn.f16x2", 5),
        ("fma.rn.ftz.f16x2", 5),
        ("fma.rn.sat.f16x2", 5),
        ("fma.rn.relu.f16x2", 10),
        ("fma.rn.ftz.sat.f16x2", 5),
        ("fma.rn.ftz.relu.f16x2", 5),
        ("fma.rn.f16", 10),
        ("fma.rn.ftz.relu.f16", 5),
        ("fma.rn.sat.f16", 5),
    ] {
        assert_eq!(instructions(&lines, instruction), n, "{instruction}\n{p}");
    }
    let guarded = |op: &str| lines.iter().filter(|l| l.starts_with("@%p") && l.contains(op)).count();
    assert_eq!(guarded(" add.f16x2 "), 5, "{p}");
    assert_eq!(guarded(" sub.ftz.f16x2 "), 5, "{p}");
    for arch in Arch::ALL {
        assert_eq!(cute::validate::check_target(kernel.ir(), arch).unwrap(), arch.min_ptx(), "{arch:?}");
    }
}

/// BFloat16 fma is `fma.rn{.relu}.bf16x2` and `fma.rn{.relu}.bf16` (sm_80, PTX 7.0); `fma` on BFloat16 values is
/// `fma.rn.bf16` too, rounding once, not an f32 fma.
#[test]
fn half_bf16_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::half_bf16().kernel;
    let p = ptx(&kernel, &tc);
    assert!(p.contains(".version 7.0"), "{p}");
    let lines = compact_lines(&p);
    for (instruction, n) in
        [("fma.rn.bf16x2", 12), ("fma.rn.relu.bf16x2", 6), ("fma.rn.bf16", 12), ("fma.rn.relu.bf16", 6)]
    {
        assert_eq!(instructions(&lines, instruction), n, "{instruction}\n{p}");
    }
    assert_eq!(instructions(&lines, "fma.rn.f32"), 0, "no fma through f32\n{p}");
}

/// The forms that need sm_90: `add.bf16x2`, `sub.bf16x2` (PTX 7.8), `ex2.approx.ftz.bf16x2` (PTX 7.8) and fma with
/// `.oob` (PTX 8.1, which raises sm_90's 8.0). Older targets are refused at the user's line.
#[test]
fn half_sm90_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::half_sm90().kernel;
    for arch in [Arch::Sm80, Arch::Sm86, Arch::Sm89] {
        let e = kernel.compile_ptx(&tc, arch).unwrap_err().to_string();
        let want = format!(
            "add.bf16x2 (add_packed_bf16x2) needs sm_90 or newer; the kernel is being compiled for {}",
            arch.chip()
        );
        assert!(e.contains(&want) && e.contains("feature_kernels/mod.rs"), "{e}");
    }
    let p = kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.contains(".version 8.1"), "{p}");
    let lines = compact_lines(&p);
    for (instruction, n) in [
        ("add.bf16x2", 5),
        ("add.rn.bf16x2", 5),
        ("sub.bf16x2", 5),
        ("sub.rn.bf16x2", 5),
        ("ex2.approx.ftz.bf16x2", 10),
        ("fma.rn.oob.f16x2", 5),
        ("fma.rn.oob.relu.f16x2", 5),
        ("fma.rn.oob.bf16x2", 5),
        ("fma.rn.oob.relu.bf16x2", 5),
        ("fma.rn.oob.f16", 5),
        ("fma.rn.oob.relu.bf16", 5),
    ] {
        let unguarded = instructions(&lines, instruction);
        let guarded = lines.iter().filter(|l| l.starts_with("@%p") && l.contains(&format!(" {instruction} "))).count();
        assert_eq!(unguarded + guarded, n, "{instruction}\n{p}");
    }
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm100).unwrap(), 86);
}

/// `exp2_packed_f16x2` is `ex2.approx.f16x2` (sm_75, so PTX 7.0 on sm_80), predicated where asked.
#[test]
fn exp2_packed_instruction() {
    let Some(tc) = toolchain() else { return };
    let p = ptx(&feature_kernels::exp2_f16x2().kernel, &tc);
    assert!(p.contains(".version 7.0"), "{p}");
    let lines = compact_lines(&p);
    assert_eq!(instructions(&lines, "ex2.approx.f16x2"), 1, "{p}");
    assert_eq!(lines.iter().filter(|l| l.starts_with("@%p") && l.contains(" ex2.approx.f16x2 ")).count(), 1, "{p}");
}

/// `cvt_f32x2_bf16x2` is `cvt.rn.satfinite.bf16x2.f32` with element 1 first, which needs PTX 8.1 and raises the
/// version from sm_80's 7.0 and sm_89's 7.8; `cvt_f32_bf16` is `cvt.rn.bf16.f32` and `cvt_f32_tf32`
/// `cvt.rna.tf32.f32`.
#[test]
fn float_conversion_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::cvt_floats().kernel;
    for arch in [Arch::Sm80, Arch::Sm86, Arch::Sm89, Arch::Sm90] {
        let p = kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        assert!(p.contains(".version 8.1"), "{arch:?}\n{p}");
    }
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm100).unwrap(), 86);
    let p = ptx(&kernel, &tc);
    let lines = compact_lines(&p);
    assert_eq!(instructions(&lines, "cvt.rn.satfinite.bf16x2.f32"), 32, "{p}");
    assert_eq!(instructions(&lines, "cvt.rna.tf32.f32"), 32, "{p}");
    assert!(instructions(&lines, "cvt.rn.bf16.f32") >= 1, "{p}");
    // The packed conversion of the data (x, y) takes y, then x: x lands in the low half.
    let kernel = Kernel::builder("bf16x2_order").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        out.store(t, cvt_f32x2_bf16x2::<Int32>([x.load(t), x.load(t + 32)]));
    });
    let lines = compact_lines(&ptx(&kernel, &tc));
    let loads: Vec<String> = lines.iter().filter(|l| l.starts_with("ld.global")).cloned().collect();
    let cvt = &operands_of(&lines, "cvt.rn.satfinite.bf16x2.f32")[0];
    let loaded = |offset: &str| {
        loads
            .iter()
            .find(|l| l.ends_with(&format!("{offset}];")))
            .map(|l| l.split(' ').nth(1).unwrap().trim_end_matches(',').to_string())
    };
    assert_eq!(loaded("+128").as_deref(), Some(cvt[1]), "element 1 first: {loads:?} {cvt:?}");
}

/// The int8 conversions sign-extend each byte with `dp4a.s32.s32` against one-hot bytes and a zero accumulator,
/// then convert with `cvt.rn.f32.s32`; the int8 to BFloat16 conversion on the Ampere path adds
/// `cvt.rn.satfinite.bf16x2.f32` per pair.
#[test]
fn int8_conversion_instructions() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::cvt_int8().kernel;
    let p = ptx(&kernel, &tc);
    let lines = compact_lines(&p);
    let dp4a = operands_of(&lines, "dp4a.s32.s32");
    // 4 + 4 + 4 + 2 + 2 + 2 for the Float32 conversions; the Ampere path for lengths 1 to 9 of the same nine values
    // shares its prefixes (pure instructions merge), leaving two quads (8) and two pairs (2 + 2).
    assert_eq!(dp4a.len(), 18 + 12, "{p}");
    // Each dp4a result is converted (the single values left over at lengths 1, 5 and 9 convert through Float32 as
    // well, which LLVM may fold to a narrower cvt).
    assert!(instructions(&lines, "cvt.rn.f32.s32") >= 18 + 12, "{p}");
    assert_eq!(instructions(&lines, "cvt.rn.satfinite.bf16x2.f32"), 2 + 2 + 1 + 1, "{p}");
    assert!(p.contains(".version 8.1"), "{p}");
    // The masks are 0x1, 0x100, 0x10000 and 0x1000000, in registers set from immediates.
    for mask in ["1", "256", "65536", "16777216"] {
        assert!(lines.iter().any(|l| l.starts_with("mov.b32") && l.ends_with(&format!(", {mask};"))), "{mask}\n{p}");
    }
    let p = kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.contains("dp4a.s32.s32"), "{p}");
}

/// `cvt_i8_bf16` is `cvt.rn.bf16.s8` (sm_90, PTX 7.8): refused below sm_90.
#[test]
fn int8_to_bf16_instruction_needs_sm90() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::cvt_i8_bf16_sm90().kernel;
    let e = kernel.compile_ptx(&tc, Arch::Sm89).unwrap_err().to_string();
    assert!(
        e.contains("cvt.rn.bf16.s8 (cvt_i8_bf16) needs sm_90 or newer; the kernel is being compiled for sm_89"),
        "{e}"
    );
    let p = kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(instructions(&compact_lines(&p), "cvt.rn.bf16.s8"), 4, "{p}");
}

/// The s2f6x2 int8 conversions (`cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2` with the scale `0x8585`, and
/// `cvt.rn.bf16.s8` for a single value) are refused for every target but the `a` targets that have them, verify as
/// MLIR, and compile to PTX 9.1 for sm_100a and sm_120a when the `mlir-opt` can emit PTX 9.1 (LLVM 23). An older
/// one (LLVM 21 and 22) ignores the request for PTX 9.1 and writes its own `.version`, which compiling refuses.
#[test]
fn s2f6x2_conversions_need_sm100a_class_targets() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::cvt_i8_bf16_s2f6x2().kernel;
    let mlir = kernel.to_mlir();
    tc.verify(&mlir).unwrap_or_else(|e| panic!("{e}"));
    assert!(mlir.contains("cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 $0, $1, sf;"), "{mlir}");
    assert!(mlir.contains("mov.b16 sf, 0x8585;"), "{mlir}");
    let chips = [Arch::Sm100a, Arch::Sm103a, Arch::Sm107a, Arch::Sm110a, Arch::Sm120a, Arch::Sm121a];
    for &arch in &chips {
        assert_eq!(cute::validate::check_target(kernel.ir(), arch).unwrap(), arch.min_ptx().max(91), "{arch:?}");
    }
    for arch in Arch::ALL.into_iter().filter(|a| !chips.contains(a)) {
        let e = kernel.compile_ptx(&tc, arch).unwrap_err().to_string();
        let want = format!(
            "needs sm_100a, sm_103a, sm_107a, sm_110a, sm_120a or sm_121a; the kernel is being compiled for {}",
            arch.chip()
        );
        assert!(e.contains(&want), "{arch:?}: {e}");
    }
    for arch in [Arch::Sm100a, Arch::Sm120a] {
        assert_eq!(cute::validate::check_target(kernel.ir(), arch).unwrap(), 91);
        match kernel.compile_ptx(&tc, arch) {
            Ok(p) => {
                assert!(p.contains(".version 9.1"), "{p}");
                // 2 + 1 for the direct calls; the lengths 1 to 9 share their prefixes: two quads (2 + 2), two
                // pairs, and a single value at each odd length's end.
                let lines = compact_lines(&p);
                assert_eq!(instructions(&lines, "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2"), 3 + 6, "{p}");
                assert_eq!(instructions(&lines, "cvt.rn.bf16.s8"), 5, "{p}");
            }
            Err(e) => {
                let e = e.to_string();
                let want = format!("needs PTX ISA 9.1 on {}, but ", arch.chip());
                assert!(
                    e.contains("kernel `cvt_i8_bf16_s2f6x2`")
                        && e.contains(&want)
                        && e.contains("its LLVM predates PTX ISA 9.1 (set CUTE_MLIR_OPT"),
                    "only an mlir-opt without PTX 9.1 may fail, and only for that: {e}"
                );
                eprintln!("skipping the PTX check for {}: {e}", arch.chip());
            }
        }
    }
}

/// `pack` and `unpack` are a vector of the parts bitcast to the whole (and back), in every width; `mma.sync` on
/// BFloat16 packs its pairs the same way.
#[test]
fn pack_and_unpack_lower_to_vector_bitcasts() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::pack_unpack().kernel;
    let mlir = kernel.to_mlir();
    for (vector, whole) in [
        ("vector<2xf16>", "i32"),
        ("vector<2xbf16>", "i32"),
        ("vector<4xi8>", "i32"),
        ("vector<2xi8>", "i16"),
        ("vector<2xi16>", "f32"),
        ("vector<2xi32>", "i64"),
        ("vector<2xf32>", "i64"),
        ("vector<4xi16>", "i64"),
        ("vector<8xi8>", "f64"),
    ] {
        assert!(mlir.contains(&format!(": {vector} to {whole}")), "{vector} to {whole}\n{mlir}");
    }
    assert!(mlir.contains(": i64 to vector<8xi8>") && mlir.contains(": i32 to vector<2xf16>"), "{mlir}");
    ptx(&kernel, &tc);
    let mma = feature_kernels::mma_bf16().kernel.to_mlir();
    assert!(mma.contains("vector.from_elements") && mma.contains(": vector<2xbf16> to i32"), "{mma}");
}

// ---------------------------------------------------------------------------
// Atomics and reductions.
// ---------------------------------------------------------------------------

/// The `atom` and `red` opcodes of `ptx`, sorted.
fn atomic_opcodes(ptx: &str) -> Vec<String> {
    let mut ops: Vec<String> = ptx
        .lines()
        .filter_map(opcode)
        .filter(|op| op.starts_with("atom.") || op.starts_with("red."))
        .map(str::to_string)
        .collect();
    ops.sort();
    ops
}

/// Atomics and reductions are PTX `atom` and `red` with explicit semantics
/// and scope, written by cute-rs rather than LLVM, so every LLVM (21 drops
/// `atomicrmw` orderings and scopes, 23 keeps them) gives the same
/// instructions: the default is relaxed at GPU scope, every `sem=` and
/// `scope=` reaches the instruction, the half-precision adds carry
/// `.noftz`, and BFloat16 and cluster scope need sm_90.
#[test]
fn atomic_instructions() {
    let toolchains = installed_toolchains();
    if toolchains.is_empty() {
        eprintln!("skipping: no mlir-opt found");
        return;
    }
    let kernels = [
        feature_kernels::atomic_orders().kernel,
        feature_kernels::atomic_lock().kernel,
        feature_kernels::atomic_float_adds().kernel,
        feature_kernels::atomic_float_extrema().kernel,
        feature_kernels::reductions().kernel,
    ];
    let sm90 = feature_kernels::atomics_sm90().kernel;
    for kernel in kernels.iter().chain([&sm90]) {
        let mlir = kernel.to_mlir();
        assert!(!mlir.contains("llvm.atomicrmw") && !mlir.contains("llvm.cmpxchg"), "{}", kernel.name());
    }
    let mut first: Option<Vec<Vec<String>>> = None;
    for tc in &toolchains {
        let mut all = Vec::new();
        for (kernel, arch) in kernels.iter().map(|k| (k, Arch::Sm89)).chain([(&sm90, Arch::Sm90)]) {
            let p = kernel.compile_ptx(tc, arch).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
            all.push(atomic_opcodes(&p));
        }
        let ops: Vec<&String> = all.iter().flatten().collect();
        for want in [
            "atom.relaxed.gpu.global.add.u32",
            "atom.relaxed.gpu.shared.add.u32",
            "atom.acq_rel.sys.shared.cas.b32",
            "atom.acq_rel.sys.global.cas.b64",
            "atom.acquire.cta.shared.cas.b32",
            "atom.release.cta.shared.exch.b32",
            "atom.release.gpu.global.min.s32",
            "atom.relaxed.sys.shared.max.u64",
            "atom.acq_rel.cta.global.xor.b64",
            "atom.relaxed.gpu.global.add.f32",
            "atom.relaxed.gpu.shared.add.f64",
            "atom.relaxed.gpu.global.add.noftz.f16",
            "atom.relaxed.gpu.global.add.noftz.f16x2",
            "atom.relaxed.gpu.global.max.s32",
            "atom.relaxed.gpu.shared.min.u32",
            "red.relaxed.gpu.global.add.u32",
            "red.release.gpu.global.add.f32",
            "red.relaxed.gpu.shared.add.f64",
            "red.relaxed.gpu.shared.add.noftz.f16",
            "red.release.gpu.shared.add.noftz.f16x2",
            "red.relaxed.sys.global.min.u32",
            "red.relaxed.cta.global.min.u32",
            "red.relaxed.cta.shared.add.u64",
            "red.relaxed.cta.global.xor.b64",
            "atom.relaxed.cluster.global.add.noftz.bf16",
            "atom.relaxed.gpu.shared.add.noftz.bf16x2",
            "red.release.cluster.shared.add.noftz.bf16",
            "red.release.cluster.global.add.noftz.bf16x2",
            "atom.acq_rel.cluster.shared.add.u32",
            "red.release.cluster.global.add.u32",
        ] {
            assert!(ops.iter().any(|op| *op == want), "no {want} with {}", tc.mlir_opt.display());
        }
        match &first {
            None => first = Some(all),
            Some(f) => assert_eq!(f, &all, "{} emits other atomics", tc.mlir_opt.display()),
        }
    }
    // BFloat16 and cluster scope need sm_90 and PTX 7.8.
    let e = sm90.compile_ptx(&toolchains[0], Arch::Sm89).unwrap_err().to_string();
    assert!(e.contains("needs sm_90 or newer"), "{e}");
}

/// `Boolean(x)` is a comparison with zero (unordered for floats), TFloat32
/// is `cvt.rna.tf32.f32`, and an MMA on TFloat32 fragments is the same PTX
/// as one on Float32 fragments.
#[test]
fn boolean_and_tf32_conversions() {
    let Some(tc) = toolchain() else { return };
    let bools = ptx(&feature_kernels::boolean_conversions().kernel, &tc);
    assert!(bools.contains("setp.neu.f32") && bools.contains("setp.neu.f16"), "{bools}");
    assert!(bools.contains("setp.ne.b32"), "{bools}");
    let tf32 = ptx(&feature_kernels::tf32_conversions().kernel, &tc);
    assert!(count(&tf32, "cvt.rna.tf32.f32") >= 2, "{tf32}");
    let mma = ptx(&feature_kernels::mma_tf32_fragments().kernel, &tc);
    // The two MMAs read the same registers, so LLVM keeps one.
    let tf32_mma = |l: &&str| l.contains("mma.sync.aligned.m16n8k8") && l.contains(".tf32");
    assert_eq!(mma.lines().filter(tf32_mma).count(), 1, "{mma}");
    let mlir = feature_kernels::mma_tf32_fragments().kernel.to_mlir();
    assert!(!mlir.contains("arith.bitcast") || !mlir.contains("f32 to f32"), "{mlir}");
}

/// The FP8 conversions are inline `cvt`s, under PTX 8.1 on sm_89 and the
/// target's own version from sm_90; the UE8M0 pairs are sm_100a's.
#[test]
fn fp8_conversions_are_cvt_with_the_right_ptx_version() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::fp8_conversions().kernel;
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm89).unwrap(), 81);
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm90).unwrap(), 78);
    let sm89 = kernel.compile_ptx(&tc, Arch::Sm89).unwrap();
    assert!(sm89.contains(".version 8.1") && sm89.contains(".target sm_89"), "{sm89}");
    for inst in [
        "cvt.rn.satfinite.e4m3x2.f32",
        "cvt.rn.satfinite.e5m2x2.f32",
        "cvt.rn.satfinite.relu.e4m3x2.f32",
        "cvt.rn.satfinite.e4m3x2.f16x2",
        "cvt.rn.satfinite.relu.e5m2x2.f16x2",
        "cvt.rn.f16x2.e4m3x2",
        "cvt.rn.f16x2.e5m2x2",
        "cvt.rn.relu.f16x2.e4m3x2",
    ] {
        assert!(sm89.contains(inst), "{inst}:\n{sm89}");
    }
    let sm90 = kernel.compile_ptx(&tc, Arch::Sm90).unwrap();
    assert!(sm90.contains(".version 7.8"), "{sm90}");
    // FP8 values live in bytes: no f8 type reaches MLIR.
    assert!(!kernel.to_mlir().contains("f8E"), "{}", kernel.to_mlir());
    let ue8m0 = feature_kernels::ue8m0_pairs().kernel.compile_ptx(&tc, Arch::Sm100a).unwrap();
    for inst in ["cvt.rz.satfinite.ue8m0x2.f32", "cvt.rp.satfinite.ue8m0x2.bf16x2", "cvt.rn.bf16x2.ue8m0x2"] {
        assert!(ue8m0.contains(inst), "{inst}:\n{ue8m0}");
    }
}

/// Int128 is `i128` in MLIR, whose division LLVM expands inline and whose
/// scalar parameter is 16 bytes; Int4 lives in a sign-extended `i8`.
#[test]
fn int128_and_int4_lowering() {
    let Some(tc) = toolchain() else { return };
    let wide = feature_kernels::int128_ops().kernel;
    let mlir = wide.to_mlir();
    assert!(mlir.contains("arith.divsi") && mlir.contains(": i128"), "{mlir}");
    let wide_ptx = ptx(&wide, &tc);
    assert!(wide_ptx.contains(".param .align 16 .b8 int128_ops_param_3[16]"), "{wide_ptx}");
    assert!(!wide_ptx.contains("call.uni") && !wide_ptx.contains("__divti3"), "no libcalls:\n{wide_ptx}");
    let narrow = feature_kernels::int4_conversions().kernel;
    let mlir = narrow.to_mlir();
    assert!(!mlir.contains(": i4") && mlir.contains("arith.shrsi"), "{mlir}");
    ptx(&narrow, &tc);
}

/// The FP8 `mma.sync` and sparse `mma.sp` forms: each kind's PTX instruction at the PTX ISA
/// version it needs (8.4 for the FP8 m16n8k32 f32 form, 8.7 for the others, 8.5 for
/// `mma.sp::ordered_metadata`), through `nvvm.mma.sync` / `nvvm.mma.sp.sync` on every installed
/// `mlir-opt` new enough and as inline PTX on every one; an older `mlir-opt` without the
/// option is refused with a clear error; and the `ldmatrix` of `LdMatrix8x16x8bOp` is the
/// `.b16` form on a 16-bit view.
#[test]
fn fp8_and_sparse_mma_ptx() {
    let toolchains = installed_toolchains();
    if toolchains.is_empty() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    }
    let cases = feature_kernels::mma_fp8()
        .into_iter()
        .map(|c| (c, Arch::Sm89))
        .chain(feature_kernels::mma_sparse().into_iter().map(|c| (c, Arch::Sm80)));
    let cases: Vec<_> = cases.collect();
    for tc in &toolchains {
        let have = tc.llvm_version().expect("mlir-opt --version names its LLVM");
        for (case, arch) in &cases {
            let kind = match case.kernel.ir().body.insts.iter().find_map(|i| match &i.op {
                cute::ir::Op::Mma { kind, .. } => Some(*kind),
                _ => None,
            }) {
                Some(kind) => kind,
                None => panic!("{}: no mma", case.name),
            };
            let version = if kind.is_sparse() {
                "8.5"
            } else if kind.ptx().contains("m16n8k32.row.col.f32") {
                "8.4"
            } else {
                "8.7"
            };
            let need = case.kernel.min_llvm_version().unwrap();
            for inline in [false, true] {
                let options = CompileOptions::new().gpu_arch(*arch).mma_inline_ptx(inline);
                match case.kernel.compile_with(tc, &options) {
                    Ok(compiled) => {
                        let ptx = compiled.ptx();
                        assert!(inline || have >= need, "{}: compiled on LLVM {have}", case.name);
                        assert!(ptx.contains(&format!(".version {version}")), "{}: {ptx}", case.name);
                        assert!(ptx.contains(&kind.ptx()), "{}: no {} in {ptx}", case.name, kind.ptx());
                    }
                    Err(CompileError::Unsupported(e)) if !inline && have < need => {
                        assert!(e.contains(&format!("LLVM {need}")) && e.contains("mma_inline_ptx"), "{e}");
                    }
                    Err(e) => panic!("{} on LLVM {have} (inline {inline}): {e}", case.name),
                }
            }
        }
    }
    let ldsm = feature_kernels::ldmatrix_8x16x8b_fp8().kernel;
    let mlir = ldsm.to_mlir();
    assert!(mlir.contains("nvgpu.ldmatrix") && mlir.contains("-> vector<4x2xi16>"), "{mlir}");
}

/// `autovec_copy`'s vectors in the PTX: 16-byte vectors for row-major
/// tiles and their fragments (with a runtime row stride, too) and swizzled
/// shared memory, `.L1::evict_first` loads and `.L1::no_allocate` stores
/// from `autovec_copy_with`, and the 32-byte vectors of 32-byte-aligned
/// buffers, two `.v4` accesses for sm_80 and one 32-byte access for sm_100
/// (PTX ISA 8.8), with every installed LLVM.
/// `autovec_copy` of a column-major tile whose column stride is a
/// `param_extent` promised a multiple of 4: the tile's `max_alignment` (4,
/// from the extent's divisor) lets each column move as one 16-byte vector,
/// `ld.global.v4` and `st.global.v4` for sm_80; with no promise, one element
/// at a time.
#[test]
fn autovec_max_alignment_in_ptx() {
    use cute::ir::Op;
    let case = feature_kernels::autovec_max_alignment();
    let vectors = |k: &Kernel| k.ir().count_ops(|op| matches!(op, Op::Load { results, .. } if results.len() == 4));
    assert_eq!(vectors(&case.kernel), 2);
    let unknown = Kernel::builder("autovec_no_divisor").block_dim([4, 1, 1]).build(|| {
        let m = param_extent("m", 1);
        let x = param_tensor::<f32>("x", Layout::col_major((m, 2)));
        let tile = local_tile(&x, (4, 2), (thread_idx().x, 0));
        assert_eq!(tile.layout().max_alignment(), 1);
        let frag = make_fragment_like::<f32, f32>(&tile);
        autovec_copy(&tile, &frag);
        autovec_copy(&frag, &tile);
    });
    assert_eq!(vectors(&unknown), 0);
    for tc in installed_toolchains() {
        let p = ptx(&case.kernel, &tc);
        assert_eq!((count(&p, "ld.global.v4."), count(&p, "st.global.v4.")), (2, 2), "{p}");
    }
}

#[test]
fn autovec_copy_vectors_in_ptx() {
    for tc in installed_toolchains() {
        let sm80 = |k: &Kernel| ptx(k, &tc);
        let p = sm80(&feature_kernels::autovec_row_major().kernel);
        assert_eq!((count(&p, "ld.global.v4."), count(&p, "st.global.v4.")), (8, 8), "{p}");
        let p = sm80(&feature_kernels::autovec_runtime_stride().kernel);
        assert_eq!((count(&p, "ld.global.v4."), count(&p, "st.global.v4.")), (8, 8), "{p}");
        let p = sm80(&feature_kernels::autovec_swizzled().kernel);
        assert_eq!((count(&p, "ld.shared.v4."), count(&p, "st.global.v4.")), (8, 8), "{p}");
        let p = sm80(&feature_kernels::autovec_evict().kernel);
        assert_eq!(count(&p, "ld.global.L1::evict_first.v4."), 8, "{p}");
        assert_eq!(count(&p, "st.global.L1::no_allocate.v4."), 8, "{p}");
        let wide = feature_kernels::autovec_256().kernel;
        let p = sm80(&wide);
        assert_eq!((count(&p, "ld.global.v4."), count(&p, "st.global.v4.")), (2, 2), "{p}");
        // One 32-byte access each way: `.v8.b32`, or `.v4.b64` (LLVM packs
        // the f32 pairs).
        let p = wide.compile_ptx(&tc, Arch::Sm100a).unwrap_or_else(|e| panic!("{e}"));
        let wide32 = |op: &str| count(&p, &format!("{op}.global.v8.")) + count(&p, &format!("{op}.global.v4.b64"));
        assert_eq!((wide32("ld"), wide32("st")), (1, 1), "{p}");
        assert!(p.contains(".version 8.8"), "{p}");
        assert_with_ptxas(&p, Arch::Sm100a);
    }
}

/// `ptxas` from CUDA 12.8 stops at PTX 8.7, so the sm_100 PTX of 32-byte
/// vectors (PTX 8.8) is only assembled by a newer one.
fn assert_with_ptxas(ptx: &str, arch: Arch) {
    let version = ptx.lines().find_map(|l| l.trim().strip_prefix(".version ")).unwrap_or("");
    let ptxas = std::env::var_os("CUTE_PTXAS").unwrap_or_else(|| "ptxas".into());
    let supported = std::process::Command::new(&ptxas)
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .is_some_and(|v| !v.contains("release 12.") || v.contains("release 12.9"));
    if version.trim() >= "8.8" && !supported {
        eprintln!("skipping the ptxas check of PTX {version}: this ptxas is older than CUDA 12.9");
        return;
    }
    assemble_with_ptxas(ptx, arch, "autovec_256");
}

/// The bulk stores, bulk async-groups and proxy fences in the sm_90a PTX,
/// with every installed LLVM (and `ptxas` when there is one); compiling them
/// for sm_89 fails naming the first sm_90 instruction, and
/// `fence.proxy.alias` raises sm_80's PTX ISA version to 7.5.
#[test]
fn bulk_stores_and_proxy_fences_in_ptx() {
    let reuse = feature_kernels::bulk_store_reuse().kernel;
    let round_trip = feature_kernels::cp_async_bulk_round_trip().kernel;
    let alias = feature_kernels::fence_proxy_alias().kernel;
    for tc in installed_toolchains() {
        let p = reuse.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        for (pattern, n) in [
            ("fence.proxy.async.shared::cta;", 2),
            ("cp.async.bulk.global.shared::cta.bulk_group", 2),
            ("cp.async.bulk.commit_group", 2),
            ("cp.async.bulk.wait_group.read", 2),
            ("cp.async.bulk.wait_group", 3),
        ] {
            assert_eq!(count(&p, pattern), n, "{pattern}\n{p}");
        }
        assemble_with_ptxas(&p, Arch::Sm90a, "bulk_store_reuse");
        let p = round_trip.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        // LLVM may unroll the loops of four tiles.
        let (stores, loads) = (
            count(&p, "cp.async.bulk.global.shared::cta.bulk_group"),
            count(&p, "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"),
        );
        assert!(matches!(stores, 1 | 4) && matches!(loads, 1 | 4), "{p}");
        assemble_with_ptxas(&p, Arch::Sm90a, "cp_async_bulk_round_trip");
        let p = alias.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
        assert!(p.contains(".version 7.5") && count(&p, "fence.proxy.alias;") == 1, "{p}");
    }
    if let Some(tc) = toolchain() {
        let e = reuse.compile_ptx(&tc, Arch::Sm89).unwrap_err().to_string();
        assert!(e.contains("fence.proxy.async.shared::cta needs sm_90 or newer"), "{e}");
        let e = round_trip.compile_ptx(&tc, Arch::Sm89).unwrap_err().to_string();
        assert!(e.contains("fence.mbarrier_init needs sm_90 or newer"), "{e}");
        // A kernel whose first sm_90 instruction is the bulk store.
        let store = Kernel::build("bulk_store_only", || {
            let out = param_tensor::<f32>("out", Layout::col_major(64));
            let smem = alloc_smem::<f32>(64, 128);
            cp_async_bulk_global_shared_cta(&out.ptr(), &smem, 256);
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, false);
        });
        let e = store.compile_ptx(&tc, Arch::Sm89).unwrap_err().to_string();
        assert!(e.contains("cp.async.bulk.global.shared::cta.bulk_group needs sm_90 or newer"), "{e}");
    }
}

/// The default int4 conversions are `lop3`, `prmt` and `fma.rn.bf16x2` (BFloat16) and `prmt`
/// alone (E4M3), with no `cvt` but the single values' `cvt.rn.bf16.f32`, and compile for sm_80
/// at PTX 7.0 (CuTe DSL allows them only on sm_100a-class targets).
#[test]
fn int4_conversions_are_lop3_prmt_and_fma() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::int4_fast_conversions().kernel;
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm80).unwrap(), 70);
    let p = ptx(&kernel, &tc);
    let lines = compact_lines(&p);
    assert!(instructions(&lines, "lop3.b32") > 0, "{p}");
    assert!(instructions(&lines, "prmt.b32") > 0, "{p}");
    assert!(instructions(&lines, "fma.rn.bf16x2") > 0, "{p}");
    assert!(!p.contains("s2f6x2") && !p.contains("cvt.rn.bf16.s8"), "{p}");
}

/// The `with_shuffle` int4 conversions issue the scaled `.s2f6x2` cvt with the scale `0x8181`:
/// refused for every target but the `a` targets that have it, verified as MLIR, and compiled to
/// PTX 9.1 for sm_100a and sm_120a when the `mlir-opt` can emit it (LLVM 23).
#[test]
fn int4_shuffle_conversions_need_ptx_9_1() {
    let Some(tc) = toolchain() else { return };
    let kernel = feature_kernels::int4_bf16_shuffle().kernel;
    let mlir = kernel.to_mlir();
    tc.verify(&mlir).unwrap_or_else(|e| panic!("{e}"));
    assert!(mlir.contains("mov.b16 sf, 0x8181;") && !mlir.contains("0x8585"), "{mlir}");
    for arch in [Arch::Sm80, Arch::Sm89, Arch::Sm90a, Arch::Sm100] {
        let e = kernel.compile_ptx(&tc, arch).unwrap_err().to_string();
        assert!(e.contains("needs sm_100a, sm_103a, sm_107a, sm_110a, sm_120a or sm_121a"), "{arch:?}: {e}");
    }
    for arch in [Arch::Sm100a, Arch::Sm120a] {
        assert_eq!(cute::validate::check_target(kernel.ir(), arch).unwrap(), 91);
        match kernel.compile_ptx(&tc, arch) {
            Ok(p) => {
                assert!(p.contains(".version 9.1"), "{p}");
                assert!(instructions(&compact_lines(&p), "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2") > 0);
            }
            Err(e) => {
                let e = e.to_string();
                assert!(e.contains("its LLVM predates PTX ISA 9.1"), "only a pre-PTX 9.1 mlir-opt may fail: {e}");
                eprintln!("skipping the PTX check for {}: {e}", arch.chip());
            }
        }
    }
}

/// The 8-bit `ldmatrix` forms: each PTX instruction (`.m8n16` with both unpackings, `.m16n16`
/// with `.b8` and both unpackings, `.x1` to `.x4`) through `nvvm.ldmatrix` on every installed
/// `mlir-opt` new enough (LLVM 22) and as inline PTX on every one, for sm_100a and sm_120a at
/// PTX 8.6 and 8.7; an older `mlir-opt` without `mma_inline_ptx` is refused naming LLVM 22, and
/// every other target is refused naming the targets that have them.
#[test]
fn ldmatrix_b8_forms_ptx() {
    let toolchains = installed_toolchains();
    if toolchains.is_empty() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    }
    let kernel = feature_kernels::ldmatrix_b8_forms().kernel;
    assert_eq!(kernel.min_llvm_version(), Some(22));
    let forms = [
        "ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64",
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32",
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64",
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32",
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 ",
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 ",
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32",
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64",
    ];
    for tc in &toolchains {
        let have = tc.llvm_version().expect("mlir-opt --version names its LLVM");
        for (arch, version) in [(Arch::Sm100a, "8.6"), (Arch::Sm120a, "8.7")] {
            for inline in [false, true] {
                let options = CompileOptions::new().gpu_arch(arch).mma_inline_ptx(inline);
                match kernel.compile_with(tc, &options) {
                    Ok(compiled) => {
                        let ptx = compiled.ptx();
                        assert!(inline || have >= 22, "compiled on LLVM {have}");
                        assert!(ptx.contains(&format!(".version {version}")), "{ptx}");
                        for form in forms {
                            assert!(ptx.contains(form), "LLVM {have}, inline {inline}: no {form} in {ptx}");
                        }
                    }
                    Err(CompileError::Unsupported(e)) if !inline && have < 22 => {
                        assert!(e.contains("LLVM 22") && e.contains("mma_inline_ptx"), "{e}");
                    }
                    Err(e) => panic!("LLVM {have} (inline {inline}) for {}: {e}", arch.chip()),
                }
            }
        }
        for arch in [Arch::Sm80, Arch::Sm89, Arch::Sm90a, Arch::Sm100, Arch::Sm120] {
            let e = kernel.compile_ptx(tc, arch).unwrap_err().to_string();
            assert!(
                e.contains("needs sm_100a, sm_101a, sm_110a or sm_120a (or the sm_100f family with PTX 8.8")
                    && e.contains(&format!("being compiled for {}", arch.chip())),
                "{arch:?}: {e}"
            );
        }
    }
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("eltType = #nvvm.ld_st_matrix_elt_type<b8x16.b4x16_p64>"), "{mlir}");
    assert!(mlir.contains("layout = #nvvm.mma_layout<col>, shape = #nvvm.ld_st_matrix_shape<m = 16, n = 16>"));
}

/// Shared-storage structs (`cute_struct!`, `cute_union!`): one byte buffer accessed through a
/// typed view per element type (u64 mbarriers, f32, f16, u16, u8 and the rest) verifies and
/// compiles for sm_80 and sm_90, and `tensorop_gemm`'s struct storage keeps its `cp.async`,
/// `ldmatrix` and `mma.sync` instructions.
#[test]
fn shared_storage_structs() {
    let Some(tc) = toolchain() else { return };
    let gemm = tensorop_gemm::GemmConfig::new(256, 256, 256);
    let kernels: Vec<Kernel> = feature_kernels::struct_cases()
        .into_iter()
        .map(|c| c.kernel)
        .chain([tensorop_gemm::build(tensorop_gemm::GemmConfig { smem_struct: true, ..gemm })])
        .collect();
    for kernel in &kernels {
        let mlir = kernel.to_mlir();
        tc.verify(&mlir).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
        assert!(mlir.contains("memref.view"), "{}: the struct's byte buffer should have typed views", kernel.name());
        for arch in [Arch::Sm80, Arch::Sm90] {
            let p = kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{} for {arch:?}: {e}", kernel.name()));
            assert!(p.contains(&format!(".target {}", arch.chip())), "{p}");
        }
    }
    let pipeline = kernels[0].compile_ptx(&tc, Arch::Sm80).unwrap();
    assert!(count(&pipeline, "mbarrier.arrive") >= 2 && count(&pipeline, "mbarrier.init") >= 1, "{pipeline}");
    let buffers = ptx(&tensorop_gemm::build(gemm), &tc);
    let storage = ptx(kernels.last().unwrap(), &tc);
    for pattern in ["cp.async.cg.shared.global", "ldmatrix.sync", "mma.sync.aligned.m16n8k16"] {
        assert_eq!(count(&storage, pattern), count(&buffers, pattern), "{pattern}:\n{storage}");
    }
}

cute_struct! {
    struct AddressedOnly { head: u32, tail: MemRange<f32>[4] }
}

/// A struct whose members only give addresses (`ptrtoint`) in a kernel with
/// a dynamic shared-memory window: its buffer, a byte range of the window,
/// still gets a typed view to take the addresses from.
#[test]
fn struct_addressed_only_in_dynamic_window() {
    let kernel = Kernel::builder("struct_addressed_only").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let dynamic = get_dyn_smem::<f32>(None);
        let s = AddressedOnly::allocate();
        let at = (s.tail.data_ptr().toint_i64() - s.base().toint_i64()).to::<u32>();
        out.store(thread_idx().x, at + dynamic.load().to::<u32>());
    });
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("memref.view %dyn_smem"), "{mlir}");
    let Some(tc) = toolchain() else { return };
    tc.verify(&mlir).unwrap();
    kernel.compile_ptx(&tc, Arch::Sm80).unwrap();
}

// ---- m9: pipeline primitives ----

/// The sm_89 pipeline kernels wait with `mbarrier.test_wait` and have
/// neither `mbarrier.try_wait` nor `fence.mbarrier_init`; each stage of each
/// array is initialised once.
#[test]
fn portable_pipeline_ptx_for_sm89() {
    let Some(tc) = toolchain() else { return };
    for (case, inits) in [
        (feature_kernels::mbarrier_array_handoff(), 6),
        (feature_kernels::async_load_handoff(1024), 4),
        (feature_kernels::async_load_handoff(56 * 1024), 4),
    ] {
        let p = case.kernel.compile_ptx(&tc, Arch::Sm89).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(count(&p, "mbarrier.init.shared"), inits, "{}\n{p}", case.name);
        assert!(count(&p, "mbarrier.test_wait.parity.shared") >= 2, "{}\n{p}", case.name);
        assert!(!p.contains("mbarrier.try_wait") && !p.contains("fence.mbarrier_init"), "{}\n{p}", case.name);
    }
    let p = feature_kernels::async_load_handoff(1024).kernel.compile_ptx(&tc, Arch::Sm89).unwrap();
    assert!(p.contains("cp.async.mbarrier.arrive.noinc.shared.b64"), "{p}");
}

/// The sm_90 pipeline kernels emit what CuTe DSL's pipelines emit: the
/// init fence, `mbarrier.try_wait.parity`, the TMA arm's
/// `mbarrier.arrive.expect_tx`, peer arrivals through `mapa` on
/// `.shared::cluster` addresses, and the cluster barrier of the init sync.
#[test]
fn sm90_pipeline_ptx() {
    let Some(tc) = toolchain() else { return };
    let ptx = |case: feature_kernels::Case| {
        case.kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{}: {e}", case.name))
    };
    let p = ptx(feature_kernels::tma_load_pipeline());
    for pattern in [
        "fence.mbarrier_init.release.cluster;",
        "mbarrier.try_wait.parity.shared",
        "mbarrier.arrive.expect_tx.",
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes",
    ] {
        assert!(p.contains(pattern), "{pattern}\n{p}");
    }
    let p = ptx(feature_kernels::cluster_pipeline_init(true));
    assert!(p.contains("mapa.shared::cluster") && p.contains(".shared::cluster.b64 _, ["), "{p}");
    assert_eq!(count(&p, "barrier.cluster.arrive.relaxed;"), 1, "{p}");
    assert_eq!(count(&p, "barrier.cluster.arrive;"), 1, "{p}");
    assert_eq!(count(&p, "barrier.cluster.wait;"), 2, "{p}");
    let p = ptx(feature_kernels::cluster_pipeline_init(false));
    assert!(!p.contains("barrier.cluster.arrive.relaxed"), "{p}");
    assert_eq!(count(&p, "barrier.cluster.arrive;"), 2, "{p}");
    let p = ptx(feature_kernels::clc_load_arrivals());
    assert!(p.contains("mapa.shared::cluster") && p.contains("mbarrier.arrive.expect_tx."), "{p}");
}

// ---- m9: PipelineAsync ----

/// `PipelineAsync` built without a target emits what CuTe DSL's does: the
/// init fence and CTA barrier of `create`, and `mbarrier.try_wait.parity`
/// waits. Built for sm_89, it waits with `mbarrier.test_wait` spins and has
/// neither `mbarrier.try_wait` nor `fence.mbarrier_init`. The cluster-mask
/// kernel's peer arrivals go through `mapa`.
#[test]
fn pipeline_async_ptx() {
    use feature_kernels::{HandoffStyle, pipeline_async_handoff, pipeline_async_handoff_for};
    let Some(tc) = toolchain() else { return };
    for style in [HandoffStyle::Participants, HandoffStyle::States, HandoffStyle::Handles] {
        let case = pipeline_async_handoff_for(style, None);
        let p = case.kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        for pattern in ["mbarrier.try_wait.parity.shared", "fence.mbarrier_init.release.cluster;", "bar.sync"] {
            assert!(p.contains(pattern), "{}: {pattern}\n{p}", case.name);
        }
        assert!(!p.contains("mbarrier.test_wait"), "{}\n{p}", case.name);
        assert_eq!(count(&p, "mbarrier.init.shared"), 8, "{}\n{p}", case.name);
        let case = pipeline_async_handoff(style);
        let p = case.kernel.compile_ptx(&tc, Arch::Sm89).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert!(count(&p, "mbarrier.test_wait.parity.shared") >= 2 && p.contains("bar.sync"), "{}\n{p}", case.name);
        assert!(!p.contains("mbarrier.try_wait") && !p.contains("fence.mbarrier_init"), "{}\n{p}", case.name);
    }
    let case = feature_kernels::pipeline_async_cluster_masks();
    let p = case.kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    assert!(p.contains("mapa.shared::cluster") && p.contains("mbarrier.try_wait.parity.shared"), "{p}");
    assert!(p.contains("fence.mbarrier_init.release.cluster;") && p.contains("barrier.cluster.wait"), "{p}");
}

// ---- m9: PipelineCpAsync, PipelineOrder and async_pipeline.ipynb ----

/// `async_pipeline.ipynb`'s pipeline kernels built for sm_90 emit what CuTe
/// DSL's do (`fence.mbarrier_init`, `mbarrier.try_wait.parity` waits, and
/// `2 * stages` barrier inits); built for sm_89 they spin on
/// `mbarrier.test_wait` and have neither `try_wait` nor the init fence.
/// The synced kernel has no mbarrier at all.
#[test]
fn async_pipeline_example_ptx() {
    use async_pipeline::{Variant, build};
    let Some(tc) = toolchain() else { return };
    for variant in [Variant::OneStage, Variant::Staged, Variant::TryTokens] {
        let p = build(variant, Arch::Sm90).compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{variant:?}: {e}"));
        for pattern in ["mbarrier.try_wait.parity.shared", "fence.mbarrier_init.release.cluster;", "bar.sync"] {
            assert!(p.contains(pattern), "{variant:?}: {pattern}\n{p}");
        }
        assert!(!p.contains("mbarrier.test_wait"), "{variant:?}\n{p}");
        assert_eq!(count(&p, "mbarrier.init.shared"), 2 * variant.stages(), "{variant:?}\n{p}");
        let p = build(variant, Arch::Sm89).compile_ptx(&tc, Arch::Sm89).unwrap_or_else(|e| panic!("{variant:?}: {e}"));
        assert!(count(&p, "mbarrier.test_wait.parity.shared") >= 2, "{variant:?}\n{p}");
        assert!(!p.contains("mbarrier.try_wait") && !p.contains("fence.mbarrier_init"), "{variant:?}\n{p}");
    }
    let p = build(Variant::Synced, Arch::Sm89).compile_ptx(&tc, Arch::Sm89).unwrap();
    assert!(!p.contains("mbarrier") && count(&p, "bar.sync") >= 3, "{p}");
}

/// The `PipelineCpAsync` rings commit with
/// `cp.async.mbarrier.arrive.noinc` after `cp.async.cg` copies and no
/// `cp.async` group, and the `PipelineOrder` logs arrive with
/// `mbarrier.arrive`; both, built for sm_89, spin on `mbarrier.test_wait`.
#[test]
fn cpasync_order_ptx() {
    let Some(tc) = toolchain() else { return };
    for case in feature_kernels::cpasync_order_cases_sm89() {
        let p = case.kernel.compile_ptx(&tc, Arch::Sm89).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert!(count(&p, "mbarrier.test_wait.parity.shared") >= 1, "{}\n{p}", case.name);
        assert!(!p.contains("mbarrier.try_wait") && !p.contains("fence.mbarrier_init"), "{}\n{p}", case.name);
        if case.name.starts_with("pipeline_cp_async") {
            assert!(p.contains("cp.async.mbarrier.arrive.noinc.shared.b64"), "{}\n{p}", case.name);
            assert!(p.contains("cp.async.cg.shared.global"), "{}\n{p}", case.name);
            assert!(!p.contains("cp.async.commit_group"), "{}\n{p}", case.name);
        } else {
            assert!(p.contains("mbarrier.arrive.shared"), "{}\n{p}", case.name);
        }
    }
}

// ---- m9: grouped GEMM schedulers ----

/// The grouped GEMM's search is a warp-wide scan (`shfl.sync.up`), ballot
/// and population count, with `shfl.sync.idx` broadcasts, around
/// `mma.sync`; the MoE layer's tile records go through mbarriers.
#[test]
fn grouped_gemm_example_ptx() {
    let Some(tc) = toolchain() else { return };
    for cached in [false, true] {
        let p = grouped_gemm::build(6, cached).compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
        for pattern in
            ["shfl.sync.up.b32", "vote.sync.ballot.b32", "popc.b32", "shfl.sync.idx.b32", "mma.sync.aligned.m16n8k16"]
        {
            assert!(p.contains(pattern), "cached {cached}: {pattern}\n{p}");
        }
        // Two searches (the first tile's and the loop's), of five scan steps each.
        assert_eq!(count(&p, "shfl.sync.up.b32"), 10, "cached {cached}\n{p}");
    }
    let p = grouped_gemm::moe(4).compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.contains("mbarrier.test_wait.parity.shared") && !p.contains("shfl.sync"), "{p}");
}

/// The TMA tutorials' sm_90 PTX (roadmap part m7-port-tma-tutorials), with every installed
/// LLVM (and `ptxas` when there is one): `tma_v0`'s copy has a TMA load to shared memory
/// completing on an mbarrier (initialised, fenced, expecting the box's bytes), a TMA
/// store and its bulk async-group; `tma_v1`'s transpose adds the shared-memory proxy fence
/// of the transpose warps, their named barrier, and two boxes per load (the 128-byte
/// swizzle).
#[test]
fn tma_tutorials_in_ptx() {
    let copy = tma_copy::Sm100SimpleCopyKernel::new().build();
    let transpose = tma_transpose::Sm100MatrixTransposeKernelV1::new().build();
    for tc in installed_toolchains() {
        let p = copy.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
        for pattern in [
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group",
            "mbarrier.init.shared",
            "mbarrier.expect_tx",
            "fence.mbarrier_init.release.cluster",
            "mbarrier.try_wait.parity",
            "cp.async.bulk.commit_group",
            "cp.async.bulk.wait_group.read \t0",
        ] {
            assert!(p.contains(pattern), "{pattern}\n{p}");
        }
        assemble_with_ptxas(&p, Arch::Sm90, "tma_copy");
        let p = transpose.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
        for pattern in [
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group",
            "fence.proxy.async.shared::cta;",
            "bar.sync",
            "mbarrier.try_wait.parity",
            "cp.async.bulk.wait_group.read \t0",
        ] {
            assert!(p.contains(pattern), "{pattern}\n{p}");
        }
        assert_eq!(count(&p, "cp.async.bulk.tensor.2d.shared::cluster.global"), 2, "{p}");
        assemble_with_ptxas(&p, Arch::Sm90, "tma_transpose");
    }
}

/// The in-kernel tensor-map updates in the sm_90a PTX (roadmap part m7-tensormap-update),
/// with every installed LLVM (LLVM 21 has no `nvvm.tensormap.replace`, which the inline
/// PTX avoids) and `ptxas` when there is one; plain sm_90 refuses `tensormap.replace`.
#[test]
fn tensormap_updates_in_ptx() {
    let replace = feature_kernels::tensormap_replace_load().kernel;
    let smem = feature_kernels::tensormap_grouped_smem().kernel;
    let gmem = feature_kernels::tensormap_grouped_gmem().kernel;
    for tc in installed_toolchains() {
        let p = replace.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        for pattern in [
            "tensormap.replace.tile.global_address.global.b1024.b64",
            "tensormap.replace.tile.global_dim.global.b1024.b32",
            "fence.proxy.tensormap::generic.release.gpu;",
            "fence.proxy.tensormap::generic.acquire.gpu",
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
        ] {
            assert!(p.contains(pattern), "{pattern}\n{p}");
        }
        assert!(p.contains(".version 8.3"), "{p}");
        assemble_with_ptxas(&p, Arch::Sm90a, "tensormap_replace_load");
        let p = smem.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        for pattern in [
            "tensormap.replace.tile.global_address.shared::cta.b1024.b64",
            "tensormap.replace.tile.global_stride.shared::cta.b1024.b64",
            "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned",
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group",
        ] {
            assert!(p.contains(pattern), "{pattern}\n{p}");
        }
        assemble_with_ptxas(&p, Arch::Sm90a, "tensormap_grouped_smem");
        let p = gmem.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        assert!(p.contains("tensormap.replace.tile.global_stride.global.b1024.b64"), "{p}");
        assemble_with_ptxas(&p, Arch::Sm90a, "tensormap_grouped_gmem");
        let e = replace.compile_ptx(&tc, Arch::Sm90).unwrap_err().to_string();
        assert!(e.contains("tensormap.replace") && e.contains("sm_90a"), "{e}");
    }
}

/// The Hopper dense GEMM port (roadmap part m8-port-hopper) in the sm_90a PTX of every
/// installed LLVM, with `ptxas` when there is one: WGMMA, the multicast TMA loads of a 2x2
/// cluster, the consumer release on a peer CTA's empty barrier (`mapa`, a cluster-scope
/// arrival), `stmatrix`, the async-proxy fence before the TMA stores, and their bulk
/// async-groups. LLVM 21 compiles it, but its PTX is wrong: the kernel's shared memory is
/// above 48 KB, and LLVM 21 drops the offsets of the buffers in the dynamic window (A and B
/// would land on the mbarriers), so it needs LLVM 22 or newer to run.
#[test]
fn hopper_gemm_in_ptx() {
    use hopper_gemm::{HopperWgmmaGemmKernel, Majors};
    let gemm = HopperWgmmaGemmKernel::new(DType::F32, (128, 128), (2, 2)).setup(
        DType::F16,
        DType::F16,
        DType::F16,
        Majors::KKN,
    );
    let kernel = gemm.build();
    for tc in installed_toolchains() {
        let p = kernel.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        for pattern in [
            "wgmma.fence.sync.aligned",
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16",
            "wgmma.commit_group.sync.aligned",
            "wgmma.wait_group.sync.aligned",
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster",
            "mbarrier.arrive.expect_tx.shared::cta.b64",
            "mapa.shared::cluster",
            "mbarrier.arrive.release.cluster.shared::cluster.b64",
            "fence.proxy.async.shared::cta",
            "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group",
            "cp.async.bulk.commit_group",
            "cp.async.bulk.wait_group.read",
            "prefetch.tensormap",
            "barrier.cluster.arrive.relaxed",
            "barrier.cluster.wait",
        ] {
            assert!(p.contains(pattern), "{pattern}\n{p}");
        }
        // LLVM 21 writes the count before the shape.
        assert!(
            p.contains("stmatrix.sync.aligned.m8n8.x4.shared.b16")
                || p.contains("stmatrix.sync.aligned.x4.m8n8.shared.b16"),
            "{p}"
        );
        // Two MMA_M tiles of 64 rows, four k-blocks: eight MMAs per k-tile, in the
        // prologue and the main loop.
        assert_eq!(count(&p, "wgmma.mma_async.sync.aligned"), 16, "{p}");
        assemble_with_ptxas(&p, Arch::Sm90a, "hopper_gemm");
    }
}

/// The TMA pipelines in sm_90a PTX (roadmap part m9-pipeline-tma), with every
/// installed LLVM, and assembled by `ptxas` when there is one: `PipelineTmaAsync`
/// arms each stage with `mbarrier.arrive.expect_tx` (in the lane `elect_one`
/// picks), its multicast loads are `cp.async.bulk.tensor ... .multicast::cluster`,
/// its consumers release peers' stages with `mapa` and a cluster-scope arrive on a
/// `shared::cluster` mbarrier, and its deferred init is a relaxed cluster arrive and
/// a wait; `PipelineTmaStore` is `cp.async.bulk.commit_group` and
/// `cp.async.bulk.wait_group.read`.
#[test]
fn tma_pipelines_in_ptx() {
    let multicast = feature_kernels::tma_pipeline_multicast(2, 2, true).kernel;
    let ring = feature_kernels::tma_pipeline_ring(true).kernel;
    let store = feature_kernels::tma_store_pipeline_epilogue().kernel;
    for tc in installed_toolchains() {
        let p = multicast.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        for pattern in [
            "mbarrier.arrive.expect_tx.shared::cta.b64",
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster",
            "mapa.shared::cluster.u64",
            "mbarrier.arrive.release.cluster.shared::cluster.b64",
            "barrier.cluster.arrive.relaxed",
            "barrier.cluster.wait",
            "mbarrier.try_wait.parity.shared::cta.b64",
        ] {
            assert!(count(&p, pattern) > 0, "{pattern}\n{p}");
        }
        assemble_with_ptxas(&p, Arch::Sm90a, "tma_pipeline_multicast");
        let p = ring.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        // One CTA: the consumers arrive on their own CTA's barrier, and there is no
        // cluster barrier.
        assert!(count(&p, "mbarrier.arrive.expect_tx.shared::cta.b64") > 0, "{p}");
        assert!(count(&p, "mbarrier.arrive.shared.b64") > 0, "{p}");
        assert_eq!(count(&p, "mapa") + count(&p, "barrier.cluster"), 0, "{p}");
        assemble_with_ptxas(&p, Arch::Sm90a, "tma_pipeline_ring");
        let p = store.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        for pattern in [
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group",
            "cp.async.bulk.commit_group",
            "cp.async.bulk.wait_group.read \t1",
            "cp.async.bulk.wait_group.read \t0",
            "fence.proxy.async.shared::cta",
        ] {
            assert!(count(&p, pattern) > 0, "{pattern}\n{p}");
        }
        assemble_with_ptxas(&p, Arch::Sm90a, "tma_store_pipeline_epilogue");
    }
}

/// The Hopper persistent, warp-specialised GEMM in sm_90a PTX (roadmap part
/// m9-hopper-persistent-gemm), with every installed LLVM, and assembled by
/// `ptxas` when there is one: the DMA warpgroup gives registers back
/// (`setmaxnreg.dec`) and the MMA warpgroups take them (`setmaxnreg.inc`),
/// the WGMMAs, the multicast TMA loads of a cluster, the TMA stores in bulk
/// groups, and the deferred cluster init. Two MMA warpgroups use the
/// example's 40 and 232 registers; one uses 248, the launch bound's budget.
#[test]
fn hopper_persistent_gemm_in_ptx() {
    use hopper_persistent_gemm::{HopperWgmmaGemmPersistentKernel, Majors};
    let one = HopperWgmmaGemmPersistentKernel::new(DType::F32, (128, 128), (2, 2), 2, true).setup(
        DType::F16,
        DType::F16,
        DType::F16,
        Majors::KKN,
    );
    let two = HopperWgmmaGemmPersistentKernel::new(DType::F32, (128, 256), (2, 1), 1, false).setup(
        DType::F16,
        DType::F16,
        DType::F16,
        Majors::KKN,
    );
    for (gemm, inc, name) in [(&one, 248, "hopper_persistent_gemm"), (&two, 232, "hopper_persistent_gemm_2wg")] {
        let kernel = gemm.build();
        for tc in installed_toolchains() {
            let p = kernel.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
            for pattern in [
                "setmaxnreg.dec.sync.aligned.u32".to_string(),
                "setmaxnreg.inc.sync.aligned.u32".to_string(),
                "wgmma.fence.sync.aligned".to_string(),
                "wgmma.mma_async.sync.aligned.m64n".to_string(),
                "wgmma.commit_group.sync.aligned".to_string(),
                "wgmma.wait_group.sync.aligned".to_string(),
                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                    .to_string(),
                "mbarrier.arrive.expect_tx.shared::cta.b64".to_string(),
                "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group".to_string(),
                "cp.async.bulk.commit_group".to_string(),
                "cp.async.bulk.wait_group.read".to_string(),
                "barrier.cluster.arrive.relaxed".to_string(),
                "barrier.cluster.wait".to_string(),
            ] {
                assert!(p.contains(&pattern), "{pattern}\n{p}");
            }
            let flat = p.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(flat.contains("setmaxnreg.dec.sync.aligned.u32 40;"), "{p}");
            assert!(flat.contains(&format!("setmaxnreg.inc.sync.aligned.u32 {inc};")), "{p}");
            assemble_with_ptxas(&p, Arch::Sm90a, name);
        }
    }
}

/// `CompileOptions::fp_contract` (on by default) marks Float `+`, `-` and `*`
/// contractible, so a multiply feeding an add becomes one `fma`; turned off they stay
/// separate, correctly rounded operations (`mul.rn`, `add.rn`), as the simulator computes
/// them. What the precise `math` functions build is never contracted.
#[test]
fn fp_contract_fuses_multiply_add() {
    let kernel = Kernel::builder("fp_contract_probe").block_dim([32, 1, 1]).build(|| {
        let x = param_ptr::<f32>("x");
        let i = thread_idx().x;
        let (a, b, c) = (x.load_at(i * 3), x.load_at(i * 3 + 1), x.load_at(i * 3 + 2));
        x.store_at(i, a * b + c - a / b);
    });
    let off = kernel.to_mlir_with(&MlirOptions { fp_contract: false, ..Default::default() });
    assert!(!off.contains("fastmath<contract>"), "{off}");
    let on = kernel.to_mlir_with(&MlirOptions::default());
    assert!(on.contains("arith.mulf") && on.contains("fastmath<contract>"), "{on}");
    assert!(on.lines().filter(|l| l.contains("arith.divf")).all(|l| !l.contains("fastmath")), "{on}");
    // The precise exponential's arithmetic is exact; `math::mul` contracts only with fastmath.
    let precise = Kernel::builder("fp_contract_exact").block_dim([32, 1, 1]).build(|| {
        let x = param_ptr::<f32>("x");
        let i = thread_idx().x;
        x.store_at(i, cute::math::exp(x.load_at(i), false));
        x.store_at(
            i + 32,
            cute::math::add(cute::math::mul(x.load_at(i), x.load_at(i + 1), false), x.load_at(i + 2), false),
        );
    });
    let text = precise.to_mlir_with(&MlirOptions::default());
    assert!(text.contains("arith.mulf") && !text.contains("fastmath<contract>"), "{text}");
    let fast = Kernel::builder("fp_contract_fast").block_dim([32, 1, 1]).build(|| {
        let x = param_ptr::<f32>("x");
        let i = thread_idx().x;
        x.store_at(i, cute::math::add(cute::math::mul(x.load_at(i), x.load_at(i + 1), true), x.load_at(i + 2), true));
    });
    assert!(fast.to_mlir_with(&MlirOptions::default()).contains("fastmath<contract>"));
    let Some(tc) = toolchain() else { return };
    let opts = |on: Option<bool>| {
        let o = CompileOptions::new().gpu_arch(Arch::Sm80);
        match on {
            Some(on) => o.fp_contract(on),
            None => o,
        }
    };
    let off = kernel.compile_with(&tc, &opts(Some(false))).unwrap();
    let default = kernel.compile_with(&tc, &opts(None)).unwrap();
    assert_eq!(count(off.ptx(), "fma.rn.f32"), 0, "{}", off.ptx());
    assert!(count(off.ptx(), "mul.rn.f32") > 0, "{}", off.ptx());
    assert!(count(default.ptx(), "fma.rn.f32") > 0, "{}", default.ptx());
    assert_eq!(count(default.ptx(), "mul.rn.f32"), 0, "{}", default.ptx());
}
