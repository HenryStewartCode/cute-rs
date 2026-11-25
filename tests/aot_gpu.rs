//! Exported kernels on a GPU: loaded back with `CudaContext::load_exported` in a fresh process
//! (no kernel IR, no `mlir-opt`) and launched with checked arguments, and called from a C
//! program built against the export with the system C compiler and `-lcuda`. Both are compared
//! with the simulator. Needs the `cuda` feature, an NVIDIA GPU and an `mlir-opt`; the C program
//! also needs `cc` and `cuda.h`.
//!
//! ```text
//! cargo test --features cuda --test aot_gpu
//! ```

#[allow(dead_code)]
#[path = "../examples/runtime_gemm.rs"]
mod runtime_gemm;

mod support;

use std::path::{Path, PathBuf};
use std::process::Command;

use cute::cuda::{CudaContext, DeviceArg};
use cute::prelude::*;
use support::child;

const N: usize = 256;
const ALPHA: f32 = 1.5;
const COUNT: i32 = 200;
/// Where the parent tells the child to find the exports.
const DIR_ENV: &str = "CUTE_RS_AOT_DIR";

/// `out[i] = alpha * x[i] + y[i]` for `i < n` (a multiple of 4), 128 threads a block.
fn axpy() -> Kernel {
    Kernel::builder("aot_axpy").block_dim([128, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((N,)));
        let y = param_tensor::<f32>("y", Layout::col_major((N,)));
        let out = param_tensor::<f32>("out", Layout::col_major((N,)));
        let alpha = param_scalar::<f32>("alpha");
        let n = param_extent("n", 4);
        let i = block_idx().x * 128 + thread_idx().x;
        if_(i.lt(n), || out.store(i, alpha * x.load(i) + y.load(i)));
    })
}

fn axpy_inputs() -> (Vec<f32>, Vec<f32>) {
    ((0..N).map(|i| (i % 17) as f32 - 8.0).collect(), (0..N).map(|i| (i % 5) as f32 * 0.25).collect())
}

fn axpy_config() -> LaunchConfig {
    LaunchConfig::new([2, 1, 1], [128, 1, 1])
}

/// The simulator's output bits.
fn simulated() -> Vec<u32> {
    let (x, y) = axpy_inputs();
    let mut args = vec![
        KernelArg::f32(x),
        KernelArg::f32(y),
        KernelArg::f32(vec![0.0; N]),
        KernelArg::ScalarF32(ALPHA),
        KernelArg::ScalarI32(COUNT),
    ];
    axpy().simulate(axpy_config(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    args[2].as_f32().iter().map(|v| v.to_bits()).collect()
}

/// The first GPU and an `mlir-opt`, or `None` (with a note).
fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("aot-gpu-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The child: load both exports, launch them with checked arguments and print the results.
fn child_side() {
    let dir = PathBuf::from(std::env::var_os(DIR_ENV).expect("the export directory"));
    let ctx = CudaContext::new(0).unwrap();
    let f = ctx.load_exported(dir.join("aot_axpy.sig")).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(f.signature().map(|s| s.params.len()), Some(5));
    let (x, y) = axpy_inputs();
    let (x, y, mut out) = (KernelArg::f32(x), KernelArg::f32(y), KernelArg::f32(vec![0.0; N]));
    let (dx, dy, dout) = (ctx.upload(&x).unwrap(), ctx.upload(&y).unwrap(), ctx.upload(&out).unwrap());
    let launch = |alpha: KernelArg, n: KernelArg, x: DeviceArg<'_>| {
        let args =
            [x, DeviceArg::Buffer(&dy), DeviceArg::Buffer(&dout), DeviceArg::Scalar(alpha), DeviceArg::Scalar(n)];
        f.launch_builder(axpy_config()).unwrap().args(&args).submit()
    };
    // The exported signature checks every argument.
    for (alpha, n, x) in [
        (KernelArg::ScalarF32(ALPHA), KernelArg::ScalarI32(6), DeviceArg::Buffer(&dx)),
        (KernelArg::ScalarI32(1), KernelArg::ScalarI32(COUNT), DeviceArg::Buffer(&dx)),
        (KernelArg::ScalarF32(ALPHA), KernelArg::ScalarI32(COUNT), DeviceArg::View { buffer: &dx, byte_offset: 4 }),
    ] {
        println!("rejected: {}", launch(alpha, n, x).unwrap_err());
    }
    launch(KernelArg::ScalarF32(ALPHA), KernelArg::ScalarI32(COUNT), DeviceArg::Buffer(&dx)).unwrap();
    ctx.synchronize().unwrap();
    ctx.download(&dout, &mut out).unwrap();
    child::begin();
    for v in out.as_f32() {
        println!("{:08x}", v.to_bits());
    }
    child::end();

    // runtime_gemm: one export for every problem size, from the PTX.
    let gemm = ctx.load_exported(dir.join("runtime_gemm.sig")).unwrap_or_else(|e| panic!("{e}"));
    for (m, n, k) in [(128, 128, 32), (256, 128, 96)] {
        let (cfg, mut args, a, b) = runtime_gemm::problem(m, n, k);
        let bufs: Vec<_> = args[..3].iter().map(|a| ctx.upload(a).unwrap()).collect();
        let dargs = [
            DeviceArg::Buffer(&bufs[0]),
            DeviceArg::Buffer(&bufs[1]),
            DeviceArg::Buffer(&bufs[2]),
            DeviceArg::Scalar(args[3].clone()),
            DeviceArg::Scalar(args[4].clone()),
        ];
        gemm.launch_builder(cfg).unwrap().args(&dargs).submit().unwrap_or_else(|e| panic!("{e}"));
        ctx.synchronize().unwrap();
        ctx.download(&bufs[2], &mut args[2]).unwrap();
        runtime_gemm::check(m, n, k, &a, &b, args[2].as_f32());
        println!("runtime_gemm {m}x{n}x{k}: correct");
    }
}

#[test]
fn exported_kernels_round_trip_in_a_fresh_process() {
    if child::is_child() {
        child_side();
        return;
    }
    let Some((ctx, toolchain)) = device() else { return };
    let arch = ctx.arch().expect("SM80 or newer");
    let dir = scratch("round-trip");
    // The axpy with a cubin when ptxas is there (the cubin path), runtime_gemm as PTX.
    let ptxas = toolchain.cuda_tools().ptxas.is_some();
    let options = CompileOptions::new().gpu_arch(arch).use_ptxas(ptxas);
    let compiled = axpy().compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"));
    let no_object = cute::ExportOptions { compile_object: false, ..Default::default() };
    compiled.export_c_with(&dir, "aot_axpy", "", &no_object).unwrap_or_else(|e| panic!("{e}"));
    let gemm = runtime_gemm::build().compile_with(&toolchain, &CompileOptions::new().gpu_arch(arch)).unwrap();
    gemm.export_c_with(&dir, "runtime_gemm", "", &no_object).unwrap();
    drop(ctx);

    let out =
        child::run_child_with("exported_kernels_round_trip_in_a_fresh_process", &[(DIR_ENV, dir.to_str().unwrap())]);
    assert!(out.success, "child failed:\n{}\n{}", out.stdout, out.stderr);
    let got: Vec<u32> = out.lines.iter().map(|l| u32::from_str_radix(l, 16).unwrap()).collect();
    assert_eq!(got, simulated());
    for want in [
        "argument 4 (`n`) must be a multiple of 4, got 6",
        "argument 3 (`alpha`) must be a f32 scalar",
        "argument 0: `x` is at 0x",
        "runtime_gemm 128x128x32: correct",
        "runtime_gemm 256x128x96: correct",
    ] {
        assert!(out.stdout.contains(want), "no `{want}` in:\n{}", out.stdout);
    }

    // An export the device cannot run, by its stamp.
    let manifest = std::fs::read_to_string(dir.join("aot_axpy.sig")).unwrap();
    let other = if arch.major() == 9 { "sm_100a" } else { "sm_90a" };
    let edited = manifest
        .lines()
        .filter(|l| !l.starts_with("cubin_target") && !l.starts_with("image cubin"))
        .map(|l| if l.starts_with("target ") { format!("target {other}") } else { l.to_string() })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(dir.join("other.sig"), edited).unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let e = ctx.load_exported(dir.join("other.sig")).err().expect("another target fails");
    assert!(e.to_string().contains(&format!("was compiled for {other}, which a device")), "{e}");
    std::fs::write(dir.join("v2.sig"), manifest.replacen("cute-rs-export 1", "cute-rs-export 2", 1)).unwrap();
    let e = ctx.load_exported(dir.join("v2.sig")).err().expect("another format fails");
    assert!(e.to_string().contains("format version 2"), "{e}");
    std::fs::remove_dir_all(&dir).unwrap();
}

const MAIN_C: &str = r#"#include "aot_axpy.h"

#include <stdio.h>
#include <string.h>

#define N 256

static void check(aot_axpy_Kernel_Module_t *m, const aot_axpy_Launch_Config_t *c, const aot_axpy_Args_t *a) {
    CUresult r = aot_axpy_Kernel_Launch(m, c, a);
    printf("rejected %d: %s\n", (int)r, m->error);
}

int main(void) {
    aot_axpy_Kernel_Module_t m;
    aot_axpy_Args_t a, bad;
    aot_axpy_Launch_Config_t c = {{2, 1, 1}, {128, 1, 1}, {1, 1, 1}, 0, NULL};
    aot_axpy_Launch_Config_t wrong = {{2, 1, 1}, {64, 1, 1}, {0, 0, 0}, 0, NULL};
    float x[N], y[N], out[N];
    CUdeviceptr dx, dy, dout, small;
    CUresult r;
    int i;
    for (i = 0; i < N; i++) {
        x[i] = (float)(i % 17) - 8.0f;
        y[i] = (float)(i % 5) * 0.25f;
        out[i] = 0.0f;
    }
    r = aot_axpy_Kernel_Module_Load(&m);
    if (r != CUDA_SUCCESS) {
        printf("load failed %d: %s\n", (int)r, m.error);
        return 1;
    }
    if (cuMemAlloc(&dx, sizeof x) || cuMemAlloc(&dy, sizeof y) || cuMemAlloc(&dout, sizeof out) ||
        cuMemAlloc(&small, 64) || cuMemcpyHtoD(dx, x, sizeof x) || cuMemcpyHtoD(dy, y, sizeof y) ||
        cuMemcpyHtoD(dout, out, sizeof out)) {
        printf("memory failed\n");
        return 1;
    }
    a.x = dx;
    a.y = dy;
    a.out = dout;
    a.alpha = 1.5f;
    a.n = 200;
    bad = a; bad.n = 6; check(&m, &c, &bad);
    bad = a; bad.x = dx + 4; check(&m, &c, &bad);
    bad = a; bad.out = small; check(&m, &c, &bad);
    check(&m, &wrong, &a);
    r = aot_axpy_Kernel_Launch(&m, &c, &a);
    if (r != CUDA_SUCCESS || cuCtxSynchronize() || cuMemcpyDtoH(out, dout, sizeof out)) {
        printf("launch failed %d: %s\n", (int)r, m.error);
        return 1;
    }
    printf("from_cubin %d\n", m.from_cubin);
    printf("@@begin@@\n");
    for (i = 0; i < N; i++) {
        unsigned int bits;
        memcpy(&bits, &out[i], sizeof bits);
        printf("%08x\n", bits);
    }
    printf("@@end@@\n");
    cuMemFree(dx);
    cuMemFree(dy);
    cuMemFree(dout);
    cuMemFree(small);
    return aot_axpy_Kernel_Module_Unload(&m) == CUDA_SUCCESS ? 0 : 1;
}
"#;

#[test]
fn c_program_matches_the_simulator() {
    let Some((ctx, toolchain)) = device() else { return };
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skipping: no C compiler `{cc}`");
        return;
    }
    let arch = ctx.arch().expect("SM80 or newer");
    drop(ctx);
    let dir = scratch("c-program");
    for use_ptxas in [false, true] {
        if use_ptxas && toolchain.cuda_tools().ptxas.is_none() {
            continue;
        }
        let options = CompileOptions::new().gpu_arch(arch).use_ptxas(use_ptxas);
        let compiled = axpy().compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"));
        let report = compiled.export_c(&dir, "aot_axpy", "").unwrap_or_else(|e| panic!("{e}"));
        let Some(object) = report.object else {
            eprintln!("skipping: {:?}", report.notes);
            std::fs::remove_dir_all(&dir).unwrap();
            return;
        };
        std::fs::write(dir.join("main.c"), MAIN_C).unwrap();
        let prog = dir.join("main");
        let build = Command::new(&cc)
            .args(["-std=c99", "-Wall", "-Wextra", "-Werror", "-I"])
            .arg(&dir)
            .arg("-I")
            .arg(cute::aot::cuda_include_dir().expect("the object was built, so cuda.h was found"))
            .arg(dir.join("main.c"))
            .arg(&object)
            .args(["-lcuda", "-o"])
            .arg(&prog)
            .output()
            .unwrap();
        assert!(build.status.success(), "{}", String::from_utf8_lossy(&build.stderr));
        let run = Command::new(&prog).output().unwrap();
        let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
        assert!(run.status.success(), "{stdout}\n{}", String::from_utf8_lossy(&run.stderr));
        let body = stdout.split_once("@@begin@@\n").and_then(|(_, r)| r.split_once("@@end@@")).unwrap().0;
        let got: Vec<u32> = body.lines().map(|l| u32::from_str_radix(l, 16).unwrap()).collect();
        assert_eq!(got, simulated(), "use_ptxas {use_ptxas}");
        assert!(stdout.contains(&format!("from_cubin {}", u8::from(use_ptxas))), "{stdout}");
        for want in [
            "rejected 1: argument 4 (`n`) must be a multiple of 4, got 6",
            "rejected 1: argument 0 (`x`) must be 16-byte aligned, got 0x",
            "rejected 1: argument 2 (`out`) needs 256 Float32 elements (1024 bytes), the pointer reaches 64",
            "rejected 1: kernel `aot_axpy` was built for blocks of (128, 1, 1), launched with (64, 1, 1)",
        ] {
            assert!(stdout.contains(want), "no `{want}` in:\n{stdout}");
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
