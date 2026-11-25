//! TVM-FFI callables on a real GPU: exported safe calls (`tvm_ffi_export!`) called through
//! their C ABI on `TVMFFIAny` arguments holding DLTensors over device memory, with a stub
//! runtime whose `TVMFFIEnvGetStream` returns a stream of the test's, checked against the
//! simulator and refusing bad arguments before any launch. With
//! `CUTE_RS_TVM_FFI_LIB=<path to libtvm_ffi_add.so>` (built by
//! `cargo build --release --features cuda --example tvm_ffi_add`) a C program compiled with
//! `cc` loads the library and calls it with its own stubs; adding
//! `CUTE_RS_TVM_FFI_PYTHON=<python with tvm_ffi and torch>` runs `examples/tvm_ffi/*.py`, the
//! ports of CuTe DSL's `jit_and_use_in_torch.py` and `error_reporting.py`. Skipped (with a
//! note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_tvm_ffi
//! ```

#[path = "support/tvm_ffi_kernels.rs"]
mod kernels;

use std::path::PathBuf;
use std::process::Command;

use cute::cuda::{CaptureMode, CudaContext};
use cute::prelude::*;
use cute::tvm_ffi::{TVMFFIAny, TVMFFISafeCallType, TvmFfiFunction};
use kernels::*;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn compile_add() -> Result<TvmFfiFunction<AddParams>, Box<dyn std::error::Error>> {
    let tc = Toolchain::discover().ok_or("no mlir-opt")?;
    Ok(TvmFfiFunction::compile(add_kernel(), &tc, &CompileOptions::new(), add_config)?.with_arg_names(&["a", "b", "c"]))
}

fn compile_scale() -> Result<TvmFfiFunction<ScaleParams>, Box<dyn std::error::Error>> {
    let tc = Toolchain::discover().ok_or("no mlir-opt")?;
    let f = TvmFfiFunction::compile(scale_kernel(), &tc, &CompileOptions::new(), scale_config)?;
    Ok(f.with_arg_names(&["x", "y", "alpha"]))
}

/// The add from an export (no `mlir-opt` at call time): `ExportedKernel` of a compile.
fn exported_add() -> Result<TvmFfiFunction<AddParams>, Box<dyn std::error::Error>> {
    let tc = Toolchain::discover().ok_or("no mlir-opt")?;
    let ctx = CudaContext::new(0)?;
    let arch = ctx.best_target(&tc).ok_or("no target")?;
    let k = add_kernel();
    let compiled = k.compile_with(&tc, &CompileOptions::new().gpu_arch(arch))?;
    let export = cute::ExportedKernel::from_compiled(&compiled);
    Ok(TvmFfiFunction::from_exported(k, export, add_config)?)
}

cute::tvm_ffi_export!(gpu_add, compile_add());
cute::tvm_ffi_export!(gpu_scale, compile_scale());
cute::tvm_ffi_export!(gpu_add_exported, exported_add());

fn call(f: TVMFFISafeCallType, args: &[TVMFFIAny]) -> Result<(), (String, String)> {
    install_stubs();
    let mut result = TVMFFIAny::int(7);
    // SAFETY: valid arguments whose tensors outlive the call.
    match unsafe { f(std::ptr::null_mut(), args.as_ptr(), args.len() as i32, &mut result) } {
        0 => Ok(()),
        _ => Err(take_raised().expect("an error raised")),
    }
}

const M: usize = 37;
const N: usize = 64;

fn inputs() -> (Vec<f32>, Vec<f32>) {
    ((0..M * N).map(|i| (i % 31) as f32 - 15.0).collect(), (0..M * N).map(|i| (i % 7) as f32 * 0.25).collect())
}

/// The add through `__tvm_ffi_gpu_add` on device memory, on the environment's stream, and
/// from an export, matching the simulator; then the call captured into a graph on that
/// stream (so nothing runs until the graph does), and on the legacy stream.
#[test]
fn add_through_the_c_abi_on_the_environment_stream() {
    let Some((ctx, _tc)) = device() else { return };
    let (a, b) = inputs();
    // The simulator's answer.
    let mut host = (
        HostTensor::from_slice(&a, Layout::row_major((M, N))),
        HostTensor::from_slice(&b, Layout::row_major((M, N))),
        HostTensor::from_slice(&vec![0.0f32; M * N], Layout::row_major((M, N))),
    );
    let sim = Kernel::builder("sim_add").block_dim([128, 1, 1]).jit_for(
        &host,
        |a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>| {
            let n = N as i32;
            for_range(block_idx().x * 128 + thread_idx().x, (M * N) as i32, grid_dim().x * 128, |i| {
                c.store((i / n, i % n), a.load((i / n, i % n)) + b.load((i / n, i % n)));
            });
        },
    );
    sim.simulate(LaunchConfig::new([19, 1, 1], [128, 1, 1]), &mut host).unwrap();
    let want = host.2.to_vec::<f32>();

    let (da, db) = (ctx.upload_slice(&a).unwrap(), ctx.upload_slice(&b).unwrap());
    let dc = ctx.alloc::<f32>(M * N).unwrap();
    dc.memset(0).unwrap();
    let shape = [M as i64, N as i64];
    let (ta, tb, tc) = (Dl::f32(da.as_raw(), &shape), Dl::f32(db.as_raw(), &shape), Dl::f32(dc.as_raw(), &shape));
    let stream = ctx.stream().unwrap();
    set_stream(stream.as_raw());

    // Eager, and through the export.
    for f in [gpu_add as TVMFFISafeCallType, gpu_add_exported] {
        dc.memset(0).unwrap();
        call(f, &[ta.any(), tb.any(), tc.any()]).unwrap_or_else(|e| panic!("{e:?}"));
        stream.synchronize().unwrap();
        assert_eq!(dc.to_vec::<f32>().unwrap(), want);
    }

    // Captured: the launch went to the environment's stream (the module is loaded already).
    dc.memset(0).unwrap();
    let capture = stream.begin_capture(CaptureMode::Relaxed).unwrap();
    let called = call(gpu_add, &[ta.any(), tb.any(), tc.any()]);
    let graph = capture.end().unwrap();
    called.unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(graph.node_count().unwrap(), 1);
    assert!(dc.to_vec::<f32>().unwrap().iter().all(|&x| x == 0.0), "nothing ran before the graph");
    graph.instantiate().unwrap().launch(&stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(dc.to_vec::<f32>().unwrap(), want);

    // The legacy stream when the environment has none.
    set_stream(std::ptr::null_mut());
    dc.memset(0).unwrap();
    call(gpu_add, &[ta.any(), tb.any(), tc.any()]).unwrap_or_else(|e| panic!("{e:?}"));
    ctx.synchronize().unwrap();
    assert_eq!(dc.to_vec::<f32>().unwrap(), want);
}

/// A scalar argument, and the checks failing on real device memory before any launch: the
/// output is untouched.
#[test]
fn scale_and_its_checks_on_device_memory() {
    let Some((ctx, _tc)) = device() else { return };
    set_stream(std::ptr::null_mut());
    let x: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let dx = ctx.upload_slice(&x).unwrap();
    let dy = ctx.alloc_bytes(256 * 4 + 64).unwrap();
    dy.memset(0).unwrap();
    let (tx, ty) = (Dl::f32(dx.as_raw(), &[256]), Dl::f32(dy.as_raw(), &[256]));
    call(gpu_scale, &[tx.any(), ty.any(), TVMFFIAny::float(0.5)]).unwrap_or_else(|e| panic!("{e:?}"));
    ctx.synchronize().unwrap();
    let mut y = vec![0f32; 256];
    dy.read(0, &mut y).unwrap();
    assert_eq!(y, x.iter().map(|v| v * 0.5).collect::<Vec<_>>());

    dy.memset(0).unwrap();
    // 8 bytes into the buffer: not the 16 bytes the kernel assumes.
    let off = Dl::f32(dy.as_raw() + 8, &[256]);
    let (kind, m) = call(gpu_scale, &[tx.any(), off.any(), TVMFFIAny::float(0.5)]).unwrap_err();
    assert_eq!(kind, "ValueError");
    assert!(m.contains("argument 1 (`y`)") && m.contains("not 16-byte aligned"), "{m}");
    // 254 elements: not a multiple of 4.
    let (short_x, short_y) = (Dl::f32(dx.as_raw(), &[254]), Dl::f32(dy.as_raw(), &[254]));
    let (kind, m) = call(gpu_scale, &[short_x.any(), short_y.any(), TVMFFIAny::float(0.5)]).unwrap_err();
    assert_eq!(kind, "ValueError");
    assert!(m.contains("multiple of 4, got 254"), "{m}");
    // A tensor where a float goes.
    let (kind, m) = call(gpu_scale, &[tx.any(), ty.any(), tx.any()]).unwrap_err();
    assert_eq!(kind, "TypeError", "{m}");
    ctx.synchronize().unwrap();
    assert!(dy.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0), "no launch happened");
}

/// The path of the `tvm_ffi_add` shared library, when the environment names one.
fn library() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("CUTE_RS_TVM_FFI_LIB")?);
    assert!(path.exists(), "CUTE_RS_TVM_FFI_LIB={} does not exist", path.display());
    Some(path)
}

/// The C program: its own `TVMFFIErrorSetRaisedFromCStrParts` and `TVMFFIEnvGetStream`
/// (exported with `-rdynamic`, where the library's `dlsym` finds them), the driver API for
/// memory, and the library loaded with `dlopen`.
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

static char raised_kind[64];
static char raised_message[1024];
static void* env_stream = 0;
static int stream_queries = 0;

void TVMFFIErrorSetRaisedFromCStrParts(const char* kind, const char** parts, int32_t n) {
  snprintf(raised_kind, sizeof raised_kind, "%s", kind);
  raised_message[0] = 0;
  for (int i = 0; i < n; ++i)
    if (parts[i]) strncat(raised_message, parts[i], sizeof raised_message - strlen(raised_message) - 1);
}

void* TVMFFIEnvGetStream(int32_t device_type, int32_t device_id) {
  ++stream_queries;
  return (device_type == 2 && device_id == 0) ? env_stream : (void*)1;
}

#define CHECK(x) do { CUresult r = (x); if (r) { printf("FAIL %s = %d\n", #x, (int)r); return 1; } } while (0)

int main(int argc, char** argv) {
  CUdevice dev; CUcontext ctx; CUstream stream;
  CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev, 0));
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev)); CHECK(cuCtxSetCurrent(ctx));
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  env_stream = stream;

  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) { printf("FAIL dlopen: %s\n", dlerror()); return 1; }
  SafeCall add_one = (SafeCall)dlsym(lib, "__tvm_ffi_add_one");
  SafeCall add_one_static = (SafeCall)dlsym(lib, "__tvm_ffi_add_one_static");
  if (!add_one || !add_one_static) { printf("FAIL dlsym\n"); return 1; }

  enum { N = 1000 };
  float a[N], b[N];
  for (int i = 0; i < N; ++i) { a[i] = (float)i * 0.5f; b[i] = 0; }
  CUdeviceptr da, db;
  CHECK(cuMemAlloc(&da, sizeof a)); CHECK(cuMemAlloc(&db, sizeof b));
  CHECK(cuMemcpyHtoD(da, a, sizeof a)); CHECK(cuMemsetD8(db, 0, sizeof b));

  int64_t shape[1] = {N};
  DLTensor ta = {(void*)da, {2, 0}, 1, {2, 32, 1}, shape, 0, 0};
  DLTensor tb = {(void*)db, {2, 0}, 1, {2, 32, 1}, shape, 0, 0};
  TVMFFIAny args[2];
  args[0].type_index = 7; args[0].zero_padding = 0; args[0].v.v_ptr = &ta;
  args[1].type_index = 7; args[1].zero_padding = 0; args[1].v.v_ptr = &tb;
  TVMFFIAny result; result.type_index = 1;
  int rc = add_one(0, args, 2, &result);
  if (rc != 0) { printf("FAIL add_one: %s: %s\n", raised_kind, raised_message); return 1; }
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuMemcpyDtoH(b, db, sizeof b));
  for (int i = 0; i < N; ++i)
    if (b[i] != a[i] + 1.0f) { printf("FAIL b[%d] = %f\n", i, b[i]); return 1; }
  printf("add_one: ok (result type %d, %d stream queries)\n", result.type_index, stream_queries);

  /* error_reporting.py: 5 elements for a kernel of (10,). */
  int64_t five[1] = {5};
  ta.shape = five; tb.shape = five;
  rc = add_one_static(0, args, 2, &result);
  printf("add_one_static(5 elements): rc %d, %s: %s\n", rc, raised_kind, raised_message);
  if (rc != -1 || strcmp(raised_kind, "ValueError") != 0) { printf("FAIL expected a ValueError\n"); return 1; }
  /* An int in place of a tensor. */
  args[1].type_index = 1; args[1].v.v_int64 = 3;
  rc = add_one(0, args, 2, &result);
  printf("add_one(t, 3): rc %d, %s: %s\n", rc, raised_kind, raised_message);
  if (rc != -1 || strcmp(raised_kind, "TypeError") != 0) { printf("FAIL expected a TypeError\n"); return 1; }
  printf("PASS\n");
  return 0;
}
"#;

/// The shared library called from C, with the C program's stubs.
#[test]
fn a_c_program_calls_the_shared_library() {
    let Some((_ctx, _tc)) = device() else { return };
    let Some(lib) = library() else {
        eprintln!("skipping: set CUTE_RS_TVM_FFI_LIB to the built libtvm_ffi_add.so");
        return;
    };
    let dir = std::env::temp_dir().join(format!("cute-rs-tvm-ffi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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
    assert!(stdout.contains("argument 0 (`a`)"), "{stdout}");
    std::fs::remove_dir_all(&dir).ok();
}

/// `examples/tvm_ffi/*.py`: the shared library loaded with `tvm_ffi.load_module` and called on
/// torch tensors.
#[test]
fn python_calls_it_on_torch_tensors() {
    let Some((_ctx, _tc)) = device() else { return };
    let (Some(lib), Some(python)) = (library(), std::env::var_os("CUTE_RS_TVM_FFI_PYTHON")) else {
        eprintln!("skipping: set CUTE_RS_TVM_FFI_LIB and CUTE_RS_TVM_FFI_PYTHON (a Python with tvm_ffi and torch)");
        return;
    };
    let examples = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/tvm_ffi");
    for script in ["jit_and_use_in_torch.py", "error_reporting.py"] {
        let out = Command::new(&python).arg(examples.join(script)).arg(&lib).output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        println!("{script}:\n{stdout}{}", String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success() && stdout.trim_end().ends_with("ok"), "{script}: {stdout}");
    }
}
