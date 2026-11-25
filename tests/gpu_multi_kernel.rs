//! Several kernels in one module on a GPU: a producer and a consumer sharing a `__device__`
//! global, loaded once and looked up by name, checked against the simulator; the split-KV
//! decode example (split kernel, then `reduction_kernel`); the module's export loaded back
//! by name and called from a C program; and, with `CUTE_RS_TVM_FFI_MODULE_LIB=<path to
//! libtvm_ffi_module.so>` (built by `cargo build --release --features cuda --example
//! tvm_ffi_module`), two TVM-FFI symbols of one library called from C, sharing one module load.
//! Skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --features cuda --test gpu_multi_kernel
//! ```

mod multi_kernel_kernels;

#[allow(dead_code)]
#[path = "../examples/split_kv_decode.rs"]
mod split_kv_decode;

use std::path::{Path, PathBuf};
use std::process::Command;

use cute::cuda::{CudaContext, CudaModule, DeviceArg};
use cute::prelude::*;
use multi_kernel_kernels::*;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("gpu-multi-kernel-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Producer, then consumer, `rounds` times through `m`, then the counter read and rewritten
/// through the module: the consumer sees what the producer and the host left in the module's
/// globals.
fn producer_then_consumer(ctx: &CudaContext, m: &CudaModule<'_>) {
    let (producer, consumer) = (multi_kernel_kernels::producer(), multi_kernel_kernels::consumer());
    let (fp, fc) = (m.function("mk_producer").unwrap(), m.function("mk_consumer").unwrap());
    for round in 1..=2 {
        let mut px = vec![KernelArg::i32(input())];
        ctx.launch(&producer, fp, LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]), &mut px)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut cx = vec![KernelArg::i32(vec![0; N]), KernelArg::ScalarI32(BIAS)];
        ctx.launch(&consumer, fc, LaunchConfig::new([1, 1, 1], [160, 1, 1]), &mut cx).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(cx[0].as_i32(), expected(round), "round {round}");
    }
    let calls = m.global("mk_calls").unwrap();
    assert_eq!(calls.read::<i32>().unwrap(), [2]);
    calls.write(&[5i32]).unwrap();
    let mut cx = vec![KernelArg::i32(vec![0; N]), KernelArg::ScalarI32(BIAS)];
    ctx.launch(&consumer, fc, LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut cx).unwrap();
    assert_eq!(cx[0].as_i32(), expected(5));
    // Each kernel's own launch checks still hold: the producer requires 128 threads.
    let mut px = vec![KernelArg::i32(input())];
    let e = ctx.launch(&producer, fp, LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut px).unwrap_err().to_string();
    assert!(e.contains("128"), "{e}");
}

#[test]
fn a_two_kernel_module_shares_its_globals() {
    let Some((ctx, tc)) = device() else { return };
    let arch = ctx.best_target(&tc).expect("a target for this device");
    for use_ptxas in [false, true] {
        if use_ptxas && tc.cuda_tools().ptxas.is_none() {
            eprintln!("note: no ptxas, the cubin path is not run");
            continue;
        }
        let options = CompileOptions::new().gpu_arch(arch).use_ptxas(use_ptxas).no_cache(true);
        let compiled = module().compile_with(&tc, &options).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(compiled.ptx().matches(".entry ").count(), 2);
        assert_eq!(compiled.cubin().is_some(), use_ptxas);
        let loads = ctx.modules_loaded();
        let m = ctx.load_kernel_module(&compiled).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(ctx.modules_loaded(), loads + 1, "one cuModuleLoadDataEx for both kernels");
        assert!(m.is_one_module());
        assert_eq!(m.kernel_names(), ["mk_producer", "mk_consumer"]);
        assert!(m.function("nothing").err().unwrap().to_string().contains("has no kernel `nothing`"));
        producer_then_consumer(&ctx, &m);
        if use_ptxas {
            // ptxas reports both kernels' resources in the module's log.
            for k in compiled.kernels() {
                assert!(k.resource_usage().is_some(), "{}: {:?}", k.name(), compiled.ptxas_log());
            }
        }
    }
}

/// The split-KV decode example: both launches from one loaded module, matching the
/// simulator.
#[test]
fn split_kv_decode_matches_the_simulator() {
    use split_kv_decode::*;
    let Some((ctx, tc)) = device() else { return };
    let arch = ctx.best_target(&tc).expect("a target");
    let compiled = module().compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{e}"));
    let m = ctx.load_kernel_module(&compiled).unwrap_or_else(|e| panic!("{e}"));
    let (q, k, v) = inputs();
    let bufs: Vec<_> = [
        KernelArg::f32(q),
        KernelArg::f32(k),
        KernelArg::f32(v),
        KernelArg::f32(vec![0.0; HEADS * MAX_SPLITS * D]),
        KernelArg::f32(vec![0.0; HEADS * MAX_SPLITS]),
        KernelArg::f32(vec![0.0; HEADS * D]),
        KernelArg::f32(vec![0.0; HEADS]),
    ]
    .iter()
    .map(|a| ctx.upload(a).unwrap())
    .collect();
    let b = |i: usize| DeviceArg::Buffer(&bufs[i]);
    let stream = ctx.stream().unwrap();
    m.function("split_kv_decode")
        .unwrap()
        .launch_builder(split_config())
        .unwrap()
        .stream(&stream)
        .args(&[b(0), b(1), b(2), b(3), b(4)])
        .submit()
        .unwrap_or_else(|e| panic!("{e}"));
    m.function("reduction_kernel")
        .unwrap()
        .launch_builder(reduction_config())
        .unwrap()
        .stream(&stream)
        .args(&[b(5), b(6), b(3), b(4), DeviceArg::Scalar(KernelArg::ScalarI32(SPLITS as i32))])
        .submit()
        .unwrap_or_else(|e| panic!("{e}"));
    stream.synchronize().unwrap();
    let (o, lse) = (bufs[5].to_vec::<f32>().unwrap(), bufs[6].to_vec::<f32>().unwrap());
    let (sim_o, sim_lse) = simulate().unwrap_or_else(|e| panic!("{e}"));
    assert!(max_error(&o, &sim_o) < 1e-5 && max_error(&lse, &sim_lse) < 1e-5, "{o:?}\n{sim_o:?}");
    let (q, k, v) = inputs();
    let (want_o, want_lse) = reference(&q, &k, &v);
    assert!(max_error(&o, &want_o) < 1e-4 && max_error(&lse, &want_lse) < 1e-4);
}

/// The module's export loaded back (no IR, no `mlir-opt`), its kernels found by name; an old
/// single-kernel export loads as a module of one.
#[test]
fn an_exported_module_loads_by_name() {
    let Some((ctx, tc)) = device() else { return };
    let arch = ctx.best_target(&tc).expect("a target");
    let dir = scratch("export");
    let use_ptxas = tc.cuda_tools().ptxas.is_some();
    let options = CompileOptions::new().gpu_arch(arch).use_ptxas(use_ptxas);
    let compiled = module().compile_with(&tc, &options).unwrap_or_else(|e| panic!("{e}"));
    let plain = cute::aot::ExportOptions { compile_object: false, ..Default::default() };
    let report = compiled.export_c_with(&dir, "mk_pair", "", &plain).unwrap_or_else(|e| panic!("{e}"));
    let m = ctx.load_exported_module(&report.manifest).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((m.name(), m.kernel_names()), ("mk_pair", vec!["mk_producer", "mk_consumer"]));
    producer_then_consumer(&ctx, &m);
    let e = ctx.load_exported(&report.manifest).err().unwrap().to_string();
    assert!(e.contains("ExportedModule::read"), "{e}");

    let single = producer().compile_with(&tc, &options).unwrap();
    let report = single.export_c_with(dir.join("single"), "mk_producer", "", &plain).unwrap();
    let one = ctx.load_exported_module(&report.manifest).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(one.kernel_names(), ["mk_producer"]);
    std::fs::remove_dir_all(&dir).unwrap();
}

const MAIN_C: &str = r#"#include "mk_pair.h"

#include <stdio.h>

#define N 128

int main(void) {
    mk_pair_Module_t m;
    mk_pair_mk_producer_Args_t pa;
    mk_pair_mk_consumer_Args_t ca;
    mk_pair_Launch_Config_t pc = {{1, 1, 1}, {N, 1, 1}, {1, 1, 1}, 0, NULL};
    mk_pair_Launch_Config_t cc = {{1, 1, 1}, {160, 1, 1}, {1, 1, 1}, 0, NULL};
    mk_pair_Launch_Config_t wrong = {{1, 1, 1}, {64, 1, 1}, {1, 1, 1}, 0, NULL};
    int x[N], out[N];
    CUdeviceptr dx, dout;
    CUresult r;
    int i;
    for (i = 0; i < N; i++) {
        x[i] = i - 50;
        out[i] = 0;
    }
    r = mk_pair_Module_Load(&m);
    if (r != CUDA_SUCCESS) {
        printf("load failed %d: %s\n", (int)r, m.error);
        return 1;
    }
    if (cuMemAlloc(&dx, sizeof x) || cuMemAlloc(&dout, sizeof out) || cuMemcpyHtoD(dx, x, sizeof x) ||
        cuMemcpyHtoD(dout, out, sizeof out)) {
        printf("memory failed\n");
        return 1;
    }
    pa.x = dx;
    ca.out = dout;
    ca.bias = 7;
    r = mk_pair_mk_producer_Launch(&m, &wrong, &pa);
    printf("rejected %d: %s\n", (int)r, m.error);
    r = mk_pair_mk_producer_Launch(&m, &pc, &pa);
    if (r == CUDA_SUCCESS) r = mk_pair_mk_consumer_Launch(&m, &cc, &ca);
    if (r != CUDA_SUCCESS || cuCtxSynchronize() || cuMemcpyDtoH(out, dout, sizeof out)) {
        printf("launch failed %d: %s\n", (int)r, m.error);
        return 1;
    }
    printf("from_cubin %d\n", m.from_cubin);
    printf("@@begin@@\n");
    for (i = 0; i < N; i++) printf("%d\n", out[i]);
    printf("@@end@@\n");
    cuMemFree(dx);
    cuMemFree(dout);
    return mk_pair_Module_Unload(&m) == CUDA_SUCCESS ? 0 : 1;
}
"#;

/// A C program built against the module's export calls the producer, then the consumer.
#[test]
fn a_c_program_calls_both_kernels() {
    let Some((ctx, toolchain)) = device() else { return };
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skipping: no C compiler `{cc}`");
        return;
    }
    let arch = ctx.best_target(&toolchain).expect("a target");
    drop(ctx);
    let dir = scratch("c-program");
    for use_ptxas in [false, true] {
        if use_ptxas && toolchain.cuda_tools().ptxas.is_none() {
            continue;
        }
        let options = CompileOptions::new().gpu_arch(arch).use_ptxas(use_ptxas);
        let compiled = module().compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"));
        let report = compiled.export_c(&dir, "mk_pair", "").unwrap_or_else(|e| panic!("{e}"));
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
        let got: Vec<i32> = body.lines().map(|l| l.parse().unwrap()).collect();
        assert_eq!(got, expected(1), "use_ptxas {use_ptxas}");
        assert!(stdout.contains(&format!("from_cubin {}", u8::from(use_ptxas))), "{stdout}");
        assert!(stdout.contains("kernel `mk_producer` was built for blocks of (128, 1, 1)"), "{stdout}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The C program for the TVM-FFI library: its own error and stream stubs, both symbols called
/// in turn, and the library's module-load count.
const HARNESS: &str = r#"
#include <cuda.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { int32_t device_type; int32_t device_id; } DLDevice;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
typedef struct {
  void* data; DLDevice device; int32_t ndim; DLDataType dtype;
  int64_t* shape; int64_t* strides; uint64_t byte_offset;
} DLTensor;
typedef struct { int32_t type_index; uint32_t zero_padding; union { int64_t v_int64; void* v_ptr; } v; } TVMFFIAny;
typedef int (*SafeCall)(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);
typedef uint64_t (*Loads)(void);

static char raised_kind[64];
static char raised_message[1024];

void TVMFFIErrorSetRaisedFromCStrParts(const char* kind, const char** parts, int32_t n) {
  snprintf(raised_kind, sizeof raised_kind, "%s", kind);
  raised_message[0] = 0;
  for (int i = 0; i < n; ++i)
    if (parts[i]) strncat(raised_message, parts[i], sizeof raised_message - strlen(raised_message) - 1);
}

void* TVMFFIEnvGetStream(int32_t device_type, int32_t device_id) {
  (void)device_type; (void)device_id;
  return 0;
}

#define CHECK(x) do { CUresult r = (x); if (r) { printf("FAIL %s = %d\n", #x, (int)r); return 1; } } while (0)

int main(int argc, char** argv) {
  CUdevice dev; CUcontext ctx;
  (void)argc;
  CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev, 0));
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev)); CHECK(cuCtxSetCurrent(ctx));
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) { printf("FAIL dlopen: %s\n", dlerror()); return 1; }
  SafeCall stash = (SafeCall)dlsym(lib, "__tvm_ffi_mk_stash");
  SafeCall unstash = (SafeCall)dlsym(lib, "__tvm_ffi_mk_unstash");
  Loads loads = (Loads)dlsym(lib, "cute_rs_tvm_ffi_module_loads");
  if (!stash || !unstash || !loads) { printf("FAIL dlsym\n"); return 1; }

  enum { N = 1000 };
  static float a[N], b[N];
  for (int i = 0; i < N; ++i) { a[i] = (float)i * 0.25f; b[i] = -1; }
  CUdeviceptr da, db;
  CHECK(cuMemAlloc(&da, sizeof a)); CHECK(cuMemAlloc(&db, sizeof b));
  CHECK(cuMemcpyHtoD(da, a, sizeof a)); CHECK(cuMemcpyHtoD(db, b, sizeof b));
  int64_t shape[1] = {N};
  DLTensor ta = {(void*)da, {2, 0}, 1, {2, 32, 1}, shape, 0, 0};
  DLTensor tb = {(void*)db, {2, 0}, 1, {2, 32, 1}, shape, 0, 0};
  TVMFFIAny arg; arg.type_index = 7; arg.zero_padding = 0;
  TVMFFIAny result; result.type_index = 1;
  arg.v.v_ptr = &ta;
  if (stash(0, &arg, 1, &result) != 0) { printf("FAIL mk_stash: %s: %s\n", raised_kind, raised_message); return 1; }
  arg.v.v_ptr = &tb;
  if (unstash(0, &arg, 1, &result) != 0) { printf("FAIL mk_unstash: %s: %s\n", raised_kind, raised_message); return 1; }
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(b, db, sizeof b));
  for (int i = 0; i < N; ++i)
    if (b[i] != a[i] * 2.0f + 1.0f) { printf("FAIL b[%d] = %f\n", i, b[i]); return 1; }
  printf("module loads: %llu\n", (unsigned long long)loads());
  if (loads() != 1) { printf("FAIL the two functions loaded %llu modules\n", (unsigned long long)loads()); return 1; }
  printf("PASS\n");
  return 0;
}
"#;

/// Two TVM-FFI symbols of one library, one module: what the first stashes in the module's
/// globals, the second reads, after a single module load.
#[test]
fn two_tvm_ffi_symbols_share_one_module() {
    let Some((_ctx, _tc)) = device() else { return };
    let Some(lib) = std::env::var_os("CUTE_RS_TVM_FFI_MODULE_LIB").map(PathBuf::from) else {
        eprintln!("skipping: set CUTE_RS_TVM_FFI_MODULE_LIB to the built libtvm_ffi_module.so");
        return;
    };
    assert!(lib.exists(), "CUTE_RS_TVM_FFI_MODULE_LIB={} does not exist", lib.display());
    let dir = scratch("tvm-ffi");
    let (src, exe) = (dir.join("harness.c"), dir.join("harness"));
    std::fs::write(&src, HARNESS).unwrap();
    let include = cute::aot::cuda_include_dir().expect("cuda.h");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let out = Command::new(&cc)
        .arg(&src)
        .arg("-I")
        .arg(&include)
        .args(["-rdynamic", "-o"])
        .arg(&exe)
        .args(["-lcuda", "-ldl"])
        .arg(format!("-L{}", include.parent().unwrap().join("lib64/stubs").display()))
        .output()
        .unwrap();
    assert!(out.status.success(), "cc: {}", String::from_utf8_lossy(&out.stderr));
    let out = Command::new(&exe).arg(&lib).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("{stdout}{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && stdout.contains("PASS"), "{stdout}");
    std::fs::remove_dir_all(&dir).ok();
}
