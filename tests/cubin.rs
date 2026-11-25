//! Cubins on a real GPU: the `ptxas` path and the PTX path agree, `--maxrregcount` takes
//! effect, SASS carries Rust lines, the driver's linker makes a cubin when `ptxas` is off, and a
//! JIT error reports the assembler's log. Needs the `cuda` feature; skipped (with a note)
//! without a GPU, an `mlir-opt` or (for the `ptxas` tests) a `ptxas`:
//!
//! ```text
//! cargo test --release --features cuda --test cubin
//! ```

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

use cute::cuda::{CudaContext, ModuleImage};
use cute::ir::Param;
use cute::prelude::*;
use cute::{Artifact, CudaTools};

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn has_ptxas(t: &Toolchain) -> bool {
    let has = t.cuda_tools().ptxas.is_some();
    if !has {
        eprintln!("skipping: no ptxas");
    }
    has
}

/// Small integers, so every sum is exact and the results are comparable bit for bit.
fn inputs(kernel: &Kernel) -> Vec<KernelArg> {
    let ir = kernel.ir();
    let mut state = 1u64;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        state >> 40
    };
    ir.args
        .iter()
        .map(|param| match param {
            Param::Buffer(b) => {
                let decl = ir.buffer(*b);
                let values: Vec<f32> = (0..decl.len.unwrap()).map(|_| (next() % 17) as f32 - 8.0).collect();
                match decl.elem {
                    DType::F16 => KernelArg::f16_from_f32(&values),
                    DType::BF16 => KernelArg::buffer(&values.iter().map(|&x| BF16::from_f32(x)).collect::<Vec<_>>()),
                    _ => KernelArg::f32(values),
                }
            }
            Param::Scalar { .. } => KernelArg::ScalarF32(1.5),
            Param::ByValue { .. } => panic!("no by-value parameters here"),
        })
        .collect()
}

fn same(a: &[KernelArg], b: &[KernelArg]) -> bool {
    a.iter().zip(b).all(|(x, y)| x == y)
}

fn kernels() -> Vec<(Kernel, LaunchConfig)> {
    let gemm = tensorop_gemm::GemmConfig::new(256, 128, 96);
    vec![
        (elementwise_add::naive(), LaunchConfig::new(elementwise_add::NAIVE_GRID, [256, 1, 1])),
        (elementwise_add::tiled(), LaunchConfig::new(elementwise_add::TILED_GRID, [128, 1, 1])),
        (sgemm::build(256, 128, 32), LaunchConfig::new([2, 1, 1], [256, 1, 1])),
        (
            tensorop_gemm::build(gemm),
            LaunchConfig::new([(gemm.m / gemm.bm) as u32, (gemm.n / gemm.bn) as u32, 1], [128, 1, 1]),
        ),
        (single_warp_pipeline::build(), LaunchConfig::new([1, 1, 1], [32, 1, 1])),
    ]
}

/// A kernel with no launch bounds (a known block size becomes `.maxntid`, and ptxas then
/// ignores `--maxrregcount`) that keeps 80 loaded values live at once.
fn register_hog() -> Kernel {
    Kernel::build("register_hog", || {
        let xs = param_tensor::<f32>("xs", Layout::col_major(4096));
        let out = param_tensor::<f32>("out", Layout::col_major(64));
        let t = thread_idx().x;
        let loaded: Vec<Float32> = (0..80).map(|i| xs.load(t + i * 40)).collect();
        let mut sum = loaded[79];
        for (i, v) in loaded.iter().enumerate().take(79) {
            sum = sum + *v * loaded[(i * 7 + 3) % 80];
        }
        out.store(t, sum);
    })
}

fn options_for(ctx: &CudaContext, t: &Toolchain) -> CompileOptions {
    CompileOptions::new().gpu_arch(ctx.best_target(t).expect("a target for this device"))
}

#[test]
fn the_cubin_path_matches_the_ptx_path() {
    let Some((ctx, t)) = device() else { return };
    if !has_ptxas(&t) {
        return;
    }
    for (kernel, cfg) in kernels() {
        let input = inputs(&kernel);
        let base = options_for(&ctx, &t);
        let via_ptxas = kernel.compile_with(&t, &base.clone().use_ptxas(true)).unwrap();
        assert!(via_ptxas.cubin().is_some(), "{}: ptxas made no cubin", kernel.name());
        let via_jit = kernel.compile_with(&t, &base.clone().use_ptxas(false)).unwrap();
        assert!(via_jit.cubin().is_none());
        let mut a = input.clone();
        ctx.launch(&kernel, &ctx.load(&via_ptxas).unwrap(), cfg, &mut a).unwrap_or_else(|e| panic!("{e}"));
        let mut b = input.clone();
        ctx.launch(&kernel, &ctx.load(&via_jit).unwrap(), cfg, &mut b).unwrap_or_else(|e| panic!("{e}"));
        let mut sim = input;
        kernel.simulate(cfg, &mut sim).unwrap();
        assert!(same(&a, &b), "{}: the cubin and the JIT disagree", kernel.name());
        assert!(same(&a, &sim), "{}: the cubin disagrees with the simulator", kernel.name());
        // The same cubin loads as bytes, and its module has no info log without -v.
        let f = ctx.load_module(ModuleImage::Cubin(via_ptxas.cubin().unwrap()), &[], kernel.name()).unwrap();
        assert!(f.jit_log().is_empty());
    }
}

#[test]
fn ptxas_options_take_effect_on_the_gpu() {
    let Some((ctx, t)) = device() else { return };
    if !has_ptxas(&t) {
        return;
    }
    let kernel = register_hog();
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    let base = options_for(&ctx, &t);
    let regs = |c: &CompiledKernel| {
        let log = c.ptxas_log().unwrap();
        let rest = &log[log.find("Used ").unwrap() + 5..];
        rest.split_whitespace().next().unwrap().parse::<u32>().unwrap()
    };
    let free = kernel.compile_with(&t, &base.clone().use_ptxas(true)).unwrap();
    let capped = kernel.compile_with(&t, &base.clone().ptxas_options("--maxrregcount=32")).unwrap();
    eprintln!("registers: {} free, {} with --maxrregcount=32", regs(&free), regs(&capped));
    assert!(regs(&capped) <= 32 && regs(&capped) < regs(&free));
    // The capped kernel still computes the same result.
    let input = inputs(&kernel);
    let (mut a, mut b) = (input.clone(), input);
    ctx.launch(&kernel, &ctx.load(&free).unwrap(), cfg, &mut a).unwrap();
    ctx.launch(&kernel, &ctx.load(&capped).unwrap(), cfg, &mut b).unwrap();
    assert!(same(&a, &b));
}

#[test]
fn sass_has_rust_lines() {
    let Some((ctx, t)) = device() else { return };
    if t.cuda_tools().nvdisasm.is_none() {
        eprintln!("skipping: no nvdisasm");
        return;
    }
    let dir = std::env::temp_dir().join(format!("cute-rs-cubin-test-{}", std::process::id()));
    let options = options_for(&ctx, &t).keep_cubin(true).keep_sass(true).dump_dir(&dir).generate_line_info(true);
    let compiled = elementwise_add::naive().compile_with(&t, &options).unwrap();
    let sass = compiled.sass().expect("SASS");
    assert!(sass.contains("elementwise_add.rs"), "{sass}");
    assert!(compiled.artifacts().iter().all(|n| n.path.is_some()), "{:?}", compiled.artifacts());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_drivers_linker_makes_a_cubin_without_ptxas() {
    let Some((ctx, t)) = device() else { return };
    // ptxas pointing nowhere is how CUTE_RS_PTXAS turns it off; here the toolchain has none.
    let bare = t.clone().with_cuda_tools(CudaTools::default());
    let dir = std::env::temp_dir().join(format!("cute-rs-link-test-{}", std::process::id()));
    for (kernel, cfg) in kernels().into_iter().take(3) {
        let options = options_for(&ctx, &t).keep_cubin(true).keep_sass(true).dump_dir(&dir);
        let compiled = kernel.compile_with(&bare, &options).unwrap();
        assert!(compiled.cubin().is_none(), "no ptxas, so no cubin at compile time");
        let input = inputs(&kernel);
        let mut linked = input.clone();
        kernel.run_cuda_with(&ctx, &bare, &options, cfg, &mut linked).unwrap_or_else(|e| panic!("{e}"));
        let mut plain = input;
        kernel.run_cuda_with(&ctx, &bare, &options_for(&ctx, &t), cfg, &mut plain).unwrap();
        assert!(same(&linked, &plain), "{}", kernel.name());
    }
    // attach_cubin makes the cubin artifact on a compiled kernel, as ELF, and writes it.
    let kernel = elementwise_add::naive();
    let mut compiled = kernel.compile_with(&bare, &options_for(&ctx, &t).keep_cubin(true).dump_dir(&dir)).unwrap();
    assert!(compiled.artifacts().iter().any(|n| n.artifact == Artifact::Cubin && n.unavailable.is_some()));
    ctx.attach_cubin(&mut compiled, &bare).unwrap();
    assert_eq!(&compiled.cubin().unwrap()[..4], b"\x7fELF");
    let note = compiled.artifacts().iter().find(|n| n.artifact == Artifact::Cubin).unwrap();
    assert!(note.path.as_ref().is_some_and(|p| p.exists()), "{note:?}");
    // A register limit passes through the linker as CU_JIT_MAX_REGISTERS.
    let capped =
        ctx.link_cubin(compiled.ptx(), "naive", &[cute::JitOption::MaxRegisters(16)]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(&capped[..4], b"\x7fELF");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn jit_flags_and_logs() {
    let Some((ctx, t)) = device() else { return };
    let bare = t.clone().with_cuda_tools(CudaTools::default());
    let kernel = elementwise_add::naive();
    let cfg = LaunchConfig::new(elementwise_add::NAIVE_GRID, [256, 1, 1]);
    let base = options_for(&ctx, &t);
    // The driver caches JIT results, and a cache hit has no log: a limit no earlier run used.
    let limit = 17 + std::process::id() % 200;
    // -v maps to CU_JIT_LOG_VERBOSE: the function carries the assembler's report.
    let compiled =
        kernel.compile_with(&bare, &base.clone().ptxas_options(&format!("--maxrregcount={limit} -v"))).unwrap();
    let function = ctx.load(&compiled).unwrap();
    eprintln!("jit log: {}", function.jit_log());
    assert!(function.jit_log().contains("registers"), "{:?}", function.jit_log());
    let mut args = inputs(&kernel);
    ctx.launch(&kernel, &function, cfg, &mut args).unwrap();
    // A flag only ptxas has is refused, naming ptxas, at compile time and not at load.
    let e = kernel.compile_with(&bare, &base.ptxas_options("--warn-on-spills")).unwrap_err().to_string();
    assert!(e.contains("ptxas") && e.contains("--warn-on-spills"), "{e}");
}

#[test]
fn invalid_inline_ptx_reports_the_assemblers_message() {
    let Some((ctx, t)) = device() else { return };
    let kernel = Kernel::builder("bad_ptx").block_dim([32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(32));
        let t = thread_idx().x;
        let bits: Uint32 = inline_ptx_with_sim("bogus.instruction.b32 $0, $1;", &[xs.load(t).into()], |a| vec![a[0]]);
        xs.store(t, bits);
    });
    let base = options_for(&ctx, &t);
    // Through the driver's JIT, the error log comes with the CUDA error.
    let compiled = kernel.compile_with(&t, &base.clone().use_ptxas(false)).unwrap();
    let e = ctx.load(&compiled).err().expect("invalid PTX").to_string();
    eprintln!("{e}");
    assert!(e.contains("cuModuleLoadDataEx failed") && e.contains("the assembler's log:"), "{e}");
    assert!(e.contains("bogus"), "{e}");
    // Through ptxas, its own output.
    if has_ptxas(&t) {
        let e = kernel.compile_with(&t, &base.use_ptxas(true)).unwrap_err();
        assert!(matches!(e, CompileError::Ptxas(_)), "{e}");
        assert!(e.to_string().contains("bogus"), "{e}");
    }
}

/// The two instructions CUDA 12.8's `ptxas` (V12.8.93) miscompiles in the feature kernels
/// (`int_multiplies`, and `casts` at -O0), which the driver's newer JIT compiles correctly:
///
/// * `sub.sat.s32 d, x, INT_MIN` from -O1 up comes out as `add.sat.s32 d, x, INT_MIN`
///   (`x - INT_MIN` rewritten as `x + (-INT_MIN)`, and `-INT_MIN` wraps to `INT_MIN`): the SASS
///   is `IADD3 R, R, -0x80000000` and a sign test;
/// * `st.b32 [a], %rd` of a 64-bit register that a `cvt.f64` wrote stores the double converted
///   to an unsigned integer (`F2I.U32.F64.TRUNC`) instead of its low 32 bits, which the PTX ISA
///   ("Operand Size Exceeding Instruction-Type Size") says are chopped. LLVM 22 at -O0 splits
///   the 8-byte stores of `casts` into such pairs.
fn ptxas_12_8_probe() -> Kernel {
    Kernel::builder("ptxas_12_8_probe").block_dim([64, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(64));
        let sat = param_tensor::<i32>("sat", Layout::col_major(64));
        let wide = param_tensor::<i64>("wide", Layout::col_major(64));
        let t = thread_idx().x;
        let x = xs.load(t);
        sat.store(t, sub_sat_int(x, i32::MIN));
        wide.store(t, x.to::<f64>().bitcast::<i64>());
    })
}

/// cute-rs's PTX for the two instructions is right (the driver's JIT matches the simulator at
/// -O0 and -O3), and where `ptxas` differs it differs exactly as its known miscompiles do.
#[test]
fn ptxas_differences_are_its_known_miscompiles() {
    let Some((ctx, t)) = device() else { return };
    let kernel = ptxas_12_8_probe();
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    let xs: Vec<i32> = (0..64).map(|i: i32| i.wrapping_mul(0x2f6b_1d3f).wrapping_add(1000)).collect();
    let input = vec![KernelArg::buffer(&xs), KernelArg::buffer(&[0i32; 64]), KernelArg::buffer(&[0i64; 64])];
    let mut sim = input.clone();
    kernel.simulate(cfg, &mut sim).unwrap();
    let (want_sat, want_wide) = (sim[1].to_vec::<i32>(), sim[2].to_vec::<i64>());
    assert_eq!(want_sat, xs.iter().map(|x| x.saturating_sub(i32::MIN)).collect::<Vec<_>>());
    let release = t.cuda_tools().ptxas.as_ref().and_then(|p| p.release);
    for level in [0, 3] {
        let base = options_for(&ctx, &t).opt_level(level);
        let run = |c: &CompiledKernel| {
            let mut args = input.clone();
            ctx.launch(&kernel, &ctx.load(c).unwrap(), cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
            (args[1].to_vec::<i32>(), args[2].to_vec::<i64>())
        };
        let jit = run(&kernel.compile_with(&t, &base.clone().use_ptxas(false)).unwrap());
        assert_eq!(jit, (want_sat.clone(), want_wide.clone()), "the driver's JIT at O{level}");
        if t.cuda_tools().ptxas.is_none() {
            continue;
        }
        let (sat, wide) = run(&kernel.compile_with(&t, &base.use_ptxas(true)).unwrap());
        let (mut sat_bug, mut st_bug) = (0, 0);
        for i in 0..64 {
            if sat[i] != want_sat[i] {
                assert_eq!(sat[i], xs[i].saturating_add(i32::MIN), "O{level}, x = {:#x}: not the known bug", xs[i]);
                sat_bug += 1;
            }
            if wide[i] != want_wide[i] {
                let double = f64::from_bits(want_wide[i] as u64);
                let known = (want_wide[i] as u64 & !0xffff_ffff) | u64::from(double as u32);
                assert_eq!(wide[i] as u64, known, "O{level}, x = {}: not the known bug", xs[i]);
                st_bug += 1;
            }
        }
        if sat_bug + st_bug > 0 {
            eprintln!(
                "ptxas {release:?} at -O{level}: {sat_bug} sub.sat.s32 x, INT_MIN and {st_bug} st.b32 of an f64 \
                 register miscompiled (the known ptxas bugs; the driver's JIT is right)"
            );
        }
    }
}
