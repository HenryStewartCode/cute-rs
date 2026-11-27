//! Several kernels in one module (CuTe DSL's `cute.compile` of a host function that launches
//! several kernels): the module's MLIR and PTX, its checks, the simulator over shared module
//! globals, the JIT cache key, and the AOT export of a module. The PTX tests need an
//! `mlir-opt` (skipped with a note otherwise); the GPU side is tests/gpu_multi_kernel.rs.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

mod multi_kernel_kernels;

#[path = "../examples/split_kv_decode.rs"]
mod split_kv_decode;

#[path = "../examples/tvm_ffi_module.rs"]
mod tvm_ffi_module;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use cute::aot::{ExportOptions, ExportedKernel, ExportedModule};

use cute::prelude::*;
use multi_kernel_kernels::*;

/// Tests that count the compile caches' work run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn toolchain() -> Option<Toolchain> {
    let t = Toolchain::discover();
    if t.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    t
}

#[test]
fn one_module_has_every_kernel() {
    let m = module();
    let mlir = m.to_mlir();
    assert_eq!(mlir.matches("gpu.module @kernels").count(), 1, "{mlir}");
    assert_eq!(mlir.matches("gpu.func @").count(), 2, "{mlir}");
    assert!(mlir.contains("gpu.func @mk_producer(") && mlir.contains("gpu.func @mk_consumer("), "{mlir}");
    // One declaration of each shared global; each kernel keeps its own launch bounds.
    assert_eq!(mlir.matches("memref.global @mk_handoff").count(), 1, "{mlir}");
    assert_eq!(mlir.matches("memref.global @mk_calls").count(), 1, "{mlir}");
    assert!(mlir.contains("nvvm.reqntid = array<i32: 128, 1, 1>"), "{mlir}");
    assert!(mlir.contains("nvvm.maxntid = array<i32: 256, 1, 1>"), "{mlir}");
    // Both kernels' static shared buffers, under different symbols.
    let shared: Vec<&str> = mlir.lines().filter(|l| l.contains("memref.global \"private\"")).collect();
    assert_eq!(shared.len(), 2, "{mlir}");
    assert!(shared[1].contains("@mk_consumer_"), "{shared:?}");
}

#[test]
fn a_module_of_one_kernel_is_the_kernel() {
    let k = producer();
    let m = KernelModule::from(k.clone());
    assert_eq!(m.name(), "mk_producer");
    assert_eq!(m.to_mlir(), k.to_mlir());
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(tc) = toolchain() else { return };
    let compiled = m.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(compiled, k.compile_ptx(&tc, Arch::Sm80).unwrap());
}

#[test]
fn module_checks() {
    let err = |r: Result<KernelModule, ModuleError>| r.unwrap_err().to_string();
    let e = err(KernelModule::new("m", [producer(), producer()]));
    assert!(e.contains("two kernels named `mk_producer`"), "{e}");
    let e = err(KernelModule::new("m", Vec::<Kernel>::new()));
    assert!(e.contains("has no kernels"), "{e}");
    let e = err(KernelModule::new("not a name", [producer()]));
    assert!(e.contains("must be a non-empty identifier"), "{e}");
    // A global declared differently (another length) by two kernels.
    let other = Kernel::build("mk_other", || {
        let h = module_global::<i32>(Some("mk_handoff"), 4, AddressSpace::Gmem, None, false);
        h.store(0, 1);
    });
    let e = err(KernelModule::new("m", [producer(), other]));
    assert!(e.contains("declare module global `mk_handoff` differently"), "{e}");
    // A global named like a kernel.
    let clash = Kernel::build("mk_clash", || {
        let h = module_global::<i32>(Some("mk_producer"), 1, AddressSpace::Gmem, None, false);
        h.store(0, 1);
    });
    let e = err(KernelModule::new("m", [producer(), clash]));
    assert!(e.contains("which is also a kernel's name"), "{e}");
    // Kernels built for different targets.
    let a = Kernel::builder("mk_a").target(Arch::Sm80).build(|| {});
    let b = Kernel::builder("mk_b").target(Arch::Sm90a).build(|| {});
    let e = err(KernelModule::new("m", [a.clone(), b]));
    assert!(e.contains("but a module has one target"), "{e}");
    // A module target other than the options'.
    let Some(tc) = toolchain() else { return };
    let m = KernelModule::new("m", [a, producer()]).unwrap();
    let e = m.compile_ptx(&tc, Arch::Sm89).unwrap_err().to_string();
    assert!(e.contains("module `m` was built for sm_80"), "{e}");
}

#[test]
fn the_simulator_shares_module_globals() {
    let m = module();
    let mut sim = m.sim_module();
    for launch in 1..=2 {
        let mut px = vec![KernelArg::i32(input())];
        sim.simulate("mk_producer", LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]), &mut px)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut cx = vec![KernelArg::i32(vec![0; N]), KernelArg::ScalarI32(BIAS)];
        sim.simulate("mk_consumer", LaunchConfig::new([1, 1, 1], [160, 1, 1]), &mut cx)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(cx[0].as_i32(), expected(launch), "after {launch} producer launches");
    }
    assert_eq!(sim.read_global("mk_calls").unwrap().to_vec::<i32>(), [2]);
    sim.write_global("mk_calls", &KernelArg::i32(vec![5])).unwrap();
    let mut cx = vec![KernelArg::i32(vec![0; N]), KernelArg::ScalarI32(BIAS)];
    sim.simulate("mk_consumer", LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut cx).unwrap();
    assert_eq!(cx[0].as_i32(), expected(5));
    let e = sim.simulate("nothing", LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut []).unwrap_err().to_string();
    assert!(e.contains("has no kernel `nothing` (it has mk_producer, mk_consumer)"), "{e}");
    assert!(sim.read_global("nothing").unwrap_err().contains("no module global `nothing`"));
}

#[test]
fn the_ptx_has_one_entry_per_kernel() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(tc) = toolchain() else { return };
    for line_info in [false, true] {
        let options = CompileOptions::new().gpu_arch(Arch::Sm89).use_ptxas(false).generate_line_info(line_info);
        let c = module().compile_with(&tc, &options).unwrap_or_else(|e| panic!("line_info {line_info}: {e}"));
        let ptx = c.ptx();
        assert_eq!(ptx.matches(".version").count(), 1, "{ptx}");
        assert_eq!(ptx.matches(".entry mk_producer").count(), 1, "{ptx}");
        assert_eq!(ptx.matches(".entry mk_consumer").count(), 1, "{ptx}");
        assert_eq!(ptx.matches(".global .align").filter(|_| true).count(), 2, "{ptx}");
        assert!(ptx.contains(".reqntid 128, 1, 1") && ptx.contains(".maxntid 256, 1, 1"), "{ptx}");
        assert_eq!(c.kernel_names(), ["mk_producer", "mk_consumer"]);
        for k in c.kernels() {
            assert_eq!(k.ptx(), ptx);
            assert_eq!(k.signature().target, Some(Arch::Sm89));
        }
        assert_eq!(c.kernel("mk_consumer").unwrap().signature().params.len(), 2);
        if line_info {
            assert!(ptx.contains(".file"), "{ptx}");
        }
    }
}

#[test]
fn the_cache_key_covers_every_kernel() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(tc) = toolchain() else { return };
    let options = CompileOptions::new().gpu_arch(Arch::Sm80).use_ptxas(false);
    let runs = || cache_stats().mlir_opt_runs;
    let base = KernelModule::new("mk_cache", [producer(), consumer_with(2)]).unwrap();
    base.compile_with(&tc, &options).unwrap();
    let before = runs();
    base.compile_with(&tc, &options).unwrap();
    assert_eq!(runs(), before, "a second compile of the same module hits the cache");
    // Another consumer, or another producer, is another module.
    let changed = KernelModule::new("mk_cache", [producer(), consumer_with(3)]).unwrap();
    changed.compile_with(&tc, &options).unwrap();
    assert_eq!(runs(), before + 1);
    let other_producer = Kernel::builder("mk_producer").block_dim([N as u32, 1, 1]).build(|| {
        let _x = param_tensor::<i32>("x", Layout::col_major((N,)));
        let h = module_global::<i32>(Some("mk_handoff"), N, AddressSpace::Gmem, None, false);
        h.store(thread_idx().x, 0);
    });
    KernelModule::new("mk_cache", [other_producer, consumer_with(2)]).unwrap().compile_with(&tc, &options).unwrap();
    assert_eq!(runs(), before + 2);
}

/// A stub `cuda.h` with what the generated C uses, with the CUDA 12.8 header's values.
const STUB_CUDA_H: &str = r#"/* A stub cuda.h for syntax checks: the types, values and functions the export uses. */
#ifndef STUB_CUDA_H
#define STUB_CUDA_H
#include <stddef.h>
#define CUDA_VERSION 12080
typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef enum cudaError_enum {
    CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_VALUE = 1, CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209, CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222, CUDA_ERROR_NOT_SUPPORTED = 801
} CUresult;
typedef enum { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76 } CUdevice_attribute;
typedef enum { CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8 } CUfunction_attribute;
typedef enum { CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4 } CUlaunchAttributeID;
typedef union { char pad[64]; struct { unsigned int x; unsigned int y; unsigned int z; } clusterDim; } CUlaunchAttributeValue;
typedef struct { CUlaunchAttributeID id; char pad[8 - sizeof(CUlaunchAttributeID)]; CUlaunchAttributeValue value; } CUlaunchAttribute;
typedef struct {
    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
    CUstream hStream; CUlaunchAttribute *attrs; unsigned int numAttrs;
} CUlaunchConfig;
CUresult cuInit(unsigned int Flags);
CUresult cuDriverGetVersion(int *driverVersion);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext *pctx);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams, void **extra);
CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra);
CUresult cuGetErrorName(CUresult error, const char **pStr);
#endif
"#;

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("multi-kernel-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The module's export (format 2) as files and back, its C checked by the system C compiler
/// against a stub `cuda.h`; a module of one kernel exports as the kernel does, and old
/// single-kernel exports read as modules of one.
#[test]
fn a_module_exports_together() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(tc) = toolchain() else { return };
    let dir = scratch("export");
    let stub = dir.join("stub");
    std::fs::create_dir_all(&stub).unwrap();
    std::fs::write(stub.join("cuda.h"), STUB_CUDA_H).unwrap();
    let options = CompileOptions::new().gpu_arch(Arch::Sm89).use_ptxas(false);
    let compiled = module().compile_with(&tc, &options).unwrap_or_else(|e| panic!("{e}"));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let have_cc = Command::new(&cc).arg("--version").output().is_ok_and(|o| o.status.success());
    let export =
        ExportOptions { compile_object: have_cc, cc: Some(cc.clone().into()), cuda_include: Some(stub.clone()) };
    let report = compiled.export_c_with(&dir, "mk_pair", "", &export).unwrap_or_else(|e| panic!("{e}"));
    let manifest = std::fs::read_to_string(&report.manifest).unwrap();
    assert!(manifest.starts_with("cute-rs-export 2\n"), "{manifest}");
    assert!(manifest.contains("module mk_pair\nkernels 2\nsignature\nkernel mk_producer\n"), "{manifest}");
    assert!(manifest.contains("signature\nkernel mk_consumer\n"), "{manifest}");
    let back = ExportedModule::read(&report.manifest).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, compiled.to_exported());
    assert_eq!(back.kernel_names(), ["mk_producer", "mk_consumer"]);
    assert_eq!(back.kernel("mk_consumer").unwrap().signature.params.len(), 2);
    let e = ExportedKernel::read(&report.manifest).unwrap_err().to_string();
    assert!(e.contains("read it with ExportedModule::read"), "{e}");
    let header = std::fs::read_to_string(&report.header).unwrap();
    for want in [
        "#define MK_PAIR_CUTE_RS_EXPORT_FORMAT 2",
        "#define MK_PAIR_NUM_KERNELS 2",
        "#define MK_PAIR_MK_CONSUMER_INDEX 1",
        "CUfunction functions[2];",
        "} mk_pair_mk_producer_Args_t;",
        "CUresult mk_pair_Module_Load(mk_pair_Module_t *module);",
        "CUresult mk_pair_mk_consumer_Launch(mk_pair_Module_t *module, const mk_pair_Launch_Config_t *config, const \
         mk_pair_mk_consumer_Args_t *args);",
    ] {
        assert!(header.contains(want), "no `{want}` in:\n{header}");
    }
    let source = std::fs::read_to_string(&report.source).unwrap();
    assert_eq!(source.matches("cuModuleLoadData(&m->module").count(), 1, "one module load");
    assert_eq!(source.matches("cuModuleGetFunction(").count(), 2);
    if have_cc {
        for (file, lang) in [(&report.header, "c"), (&report.source, "c"), (&report.header, "c++")] {
            let std = if lang == "c" { "-std=c99" } else { "-std=c++11" };
            let out = Command::new(&cc)
                .args(["-fsyntax-only", "-Wall", "-Wextra", "-Werror", "-pedantic", std, "-x", lang, "-I"])
                .arg(&stub)
                .arg("-I")
                .arg(&dir)
                .arg(file)
                .output()
                .unwrap();
            assert!(out.status.success(), "{}:\n{}", file.display(), String::from_utf8_lossy(&out.stderr));
        }
        assert!(report.object.as_ref().is_some_and(|o| o.is_file()), "{:?}", report.notes);
    }

    // A module of one kernel: the kernel's own export, byte for byte, read back as a module.
    let one = KernelModule::from(producer()).compile_with(&tc, &options).unwrap();
    let single = producer().compile_with(&tc, &options).unwrap();
    let (a, b) = (dir.join("one"), dir.join("single"));
    let plain = ExportOptions { compile_object: false, ..ExportOptions::default() };
    one.export_c_with(&a, "k", "", &plain).unwrap();
    single.export_c_with(&b, "k", "", &plain).unwrap();
    for f in ["k.h", "k.c", "k.sig", "k.ptx"] {
        assert_eq!(std::fs::read(a.join(f)).unwrap(), std::fs::read(b.join(f)).unwrap(), "{f}");
    }
    let old = ExportedModule::read(b.join("k.sig")).unwrap();
    assert_eq!((old.name.as_str(), old.kernel_names(), old.stamp.format), ("mk_producer", vec!["mk_producer"], 1));
    if std::env::var_os("MK_KEEP").is_none() {
        std::fs::remove_dir_all(&dir).unwrap();
    } else {
        eprintln!("KEPT {}", dir.display());
    }
}

#[test]
fn malformed_module_manifests_are_rejected() {
    let sig = |name: &str| {
        let mut s = producer().signature();
        s.name = name.to_string();
        (s.target, s.ptx_version) = (Some(Arch::Sm89), Some(78));
        s.to_text()
    };
    let head = "cute-rs-export 2\ncrate 0.1.0\nimage ptx a.ptx\nmodule m\n";
    let text = format!("{head}kernels 2\nsignature\n{}signature\n{}", sig("a"), sig("b"));
    let m = ExportedModule::from_parts(&text, Some(".version 7.8\n"), None).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(m.kernel_names(), ["a", "b"]);
    let e = ExportedModule::from_parts(
        &format!("{head}kernels 3\nsignature\n{}signature\n{}", sig("a"), sig("b")),
        Some(""),
        None,
    )
    .unwrap_err();
    assert!(e.message().contains("`kernels 3`, but it has 2 signatures"), "{e}");
    let e = ExportedModule::from_parts(
        &format!("{head}kernels 2\nsignature\n{}signature\n{}", sig("a"), sig("a")),
        Some(""),
        None,
    )
    .unwrap_err();
    assert!(e.message().contains("two kernels named `a`"), "{e}");
    let e =
        ExportedModule::from_parts(&text.replace("cute-rs-export 2", "cute-rs-export 9"), Some(""), None).unwrap_err();
    assert!(e.message().contains("reads versions 1 and 2"), "{e}");
    let e = ExportedModule::from_parts(&text.replace("module m\n", ""), Some(""), None).unwrap_err();
    assert!(e.message().contains("no `module` line"), "{e}");
    // A kernel taken out of a module is a format-1 export, which reads back.
    let a = m.kernel("a").unwrap();
    assert_eq!(a.stamp.format, 1);
    let manifest = a.manifest(Some("a.ptx"), None);
    assert!(manifest.starts_with("cute-rs-export 1\n"), "{manifest}");
    ExportedKernel::from_parts(&manifest, Some(".version 7.8\n"), None).unwrap_or_else(|e| panic!("{e}"));
    // So does a format-2 module of one kernel, written again.
    let one = format!("{head}kernels 1\nsignature\n{}", sig("a"));
    let one = ExportedModule::from_parts(&one, Some(".version 7.8\n"), None).unwrap_or_else(|e| panic!("{e}"));
    let again = one.manifest(Some("a.ptx"), None);
    ExportedModule::from_parts(&again, Some(".version 7.8\n"), None).unwrap_or_else(|e| panic!("{again}: {e}"));
}

/// The split-KV decode example (CuTe DSL's MLA decode: split kernel, then `reduction_kernel`)
/// on the simulator, against attention on the host.
#[test]
fn split_kv_decode_example() {
    use split_kv_decode::*;
    let (q, k, v) = inputs();
    let (want_o, want_lse) = reference(&q, &k, &v);
    let (o, lse) = simulate().unwrap_or_else(|e| panic!("{e}"));
    assert!(max_error(&o, &want_o) < 1e-4, "{}", max_error(&o, &want_o));
    assert!(max_error(&lse, &want_lse) < 1e-4, "{}", max_error(&lse, &want_lse));
    assert_eq!(module().to_mlir().matches("gpu.func @").count(), 2);
}

/// The TVM-FFI library's two kernels on the simulator: what the first stashes in the module's
/// globals, the second reads; and their callables are made from the compiled module.
#[test]
fn tvm_ffi_module_kernels() {
    use cute::external::DlpackTensor;
    use tvm_ffi_module::*;
    let (stash, unstash) = (stash_kernel::<DlpackTensor>(), unstash_kernel::<DlpackTensor>());
    let m = KernelModule::new("mk_tvm", [stash.kernel().clone(), unstash.kernel().clone()]).unwrap();
    let mut sim = m.sim_module();
    let n = 10usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
    // Each kernel takes the tensor's address and its dynamic extent.
    let args = |data: Vec<f32>| vec![KernelArg::f32(data), KernelArg::ScalarI32(n as i32)];
    assert_eq!(stash.kernel().signature().params.len(), 2, "{}", stash.kernel().signature());
    let cfg = LaunchConfig::new([1, 1, 1], [1, 1, 1]);
    sim.simulate("mk_stash_kernel", cfg, &mut args(a.clone())).unwrap_or_else(|e| panic!("{e}"));
    let mut b = args(vec![-1.0; n]);
    sim.simulate("mk_unstash_kernel", cfg, &mut b).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(b[0].as_f32(), a.iter().map(|x| x * 2.0 + 1.0).collect::<Vec<_>>());
    assert_eq!(sim.read_global("mk_count").unwrap().to_vec::<i32>(), [n as i32]);

    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(tc) = toolchain() else { return };
    let compiled = m.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap_or_else(|e| panic!("{e}"));
    let f = TvmFfiFunction::from_module(unstash_kernel::<DlpackTensor>(), &compiled, one_thread).unwrap();
    assert_eq!(f.compiled().map(|k| k.name()), Some("mk_unstash_kernel"));
    assert_eq!(f.compiled().unwrap().ptx(), compiled.ptx());
    let other =
        Kernel::builder("not_in_it").jit(make_fake_tensor::<f32>((4,), (1,)), |_: DlTensor<f32, DlpackTensor>| {});
    let e = TvmFfiFunction::from_module(other, &compiled, one_thread).unwrap_err();
    assert!(e.contains("module `mk_tvm` has no kernel `not_in_it` (it has mk_stash_kernel, mk_unstash_kernel)"), "{e}");
    let exported = compiled.to_exported();
    let f = TvmFfiFunction::from_exported_module(stash_kernel::<DlpackTensor>(), &exported, one_thread).unwrap();
    assert_eq!(f.signature().name, "mk_stash_kernel");
}

/// Kernels with dynamic shared memory share the module's one extern declaration, at the
/// strictest alignment either needs; each keeps its own window.
#[test]
fn dynamic_shared_memory_is_declared_once() {
    let dynamic = |name: &str, align: usize| {
        Kernel::build(name, move || {
            let out = param_tensor::<i32>("out", Layout::col_major((32,)));
            let p = get_dyn_smem::<i32>(Some(align));
            let t = thread_idx().x;
            if_(t.eq(0), || p.store(7i32));
            sync_threads();
            out.store(t, p.load());
        })
    };
    let m = KernelModule::new("mk_dyn", [dynamic("mk_dyn_a", 16), dynamic("mk_dyn_b", 128)]).unwrap();
    let mlir = m.to_mlir();
    let externs: Vec<&str> = mlir.lines().filter(|l| l.contains("memref.global @__cute_dynamic_smem")).collect();
    assert_eq!(externs.len(), 1, "{mlir}");
    assert!(externs[0].contains("alignment = 128"), "{}", externs[0]);
    let mut sim = m.sim_module();
    for k in ["mk_dyn_a", "mk_dyn_b"] {
        let mut args = vec![KernelArg::i32(vec![0; 32])];
        let cfg = LaunchConfig { dynamic_smem_bytes: 4, ..LaunchConfig::new([1, 1, 1], [32, 1, 1]) };
        sim.simulate(k, cfg, &mut args).unwrap_or_else(|e| panic!("{k}: {e}"));
        assert_eq!(args[0].as_i32(), [7; 32]);
    }
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(tc) = toolchain() else { return };
    let ptx = m.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(ptx.matches(".extern .shared").count(), 1, "{ptx}");
    assert!(ptx.contains(".extern .shared .align 128"), "{ptx}");
}

/// The two-kernel module's MLIR, as a golden file (`CUTE_RS_BLESS=1` rewrites it).
#[test]
fn golden_module_mlir() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/multi_kernel_golden/mk_pair.mlir");
    let text = module().to_mlir();
    if std::env::var_os("CUTE_RS_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &text).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(text == want, "mk_pair.mlir differs from the golden file; CUTE_RS_BLESS=1 rewrites it:\n{text}");
}

/// Module-level symbols the lowering makes (printf format strings, `__assertfail`), by-value
/// parameters, line information and a printf format that looks like a debug-info alias:
/// two kernels with each in one module lower, with or without line information.
#[test]
fn kernels_with_printf_assertions_and_by_value_params_share_a_module() {
    let kernel = |name: &str| {
        Kernel::builder(name).block_dim([32, 1, 1]).build(|| {
            let out = param_tensor::<i32>("out", Layout::col_major((32,)));
            let s = param_bytes("s", 8, 4, false);
            let t = thread_idx().x;
            cute::testing::assert_(s.load::<i32>(0).lt(t), "#di_sp distinct[1]<> in a message");
            if_(t.eq(0), || printf("#di_file distinct[1]<> {}\n", &[t.into()]));
            out.store(t, s.load::<i32>(4) + t);
        })
    };
    let m = KernelModule::new("mk_lowered", [kernel("mk_low_a"), kernel("mk_low_b")]).unwrap();
    let mlir = m.to_mlir();
    assert_eq!(mlir.matches("#di_file distinct[1]<> ").count(), 2, "{mlir}");
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(tc) = toolchain() else { return };
    for line_info in [false, true] {
        let options = CompileOptions::new()
            .gpu_arch(Arch::Sm89)
            .use_ptxas(false)
            .enable_assertions(true)
            .generate_line_info(line_info)
            .no_cache(true);
        let c = m.compile_with(&tc, &options).unwrap_or_else(|e| panic!("line_info {line_info}: {e}"));
        let ptx = c.ptx();
        assert_eq!(ptx.matches(".entry mk_low_").count(), 2, "{ptx}");
        assert!(ptx.contains("__assertfail"), "{ptx}");
        if line_info {
            let mlir = c.mlir();
            assert!(mlir.contains("#di_file distinct[1]<> "), "{mlir}");
        }
    }
}

/// Unnamed module globals are each kernel's own, so two kernels that both declare one cannot
/// share a module (their automatic names would make them one symbol).
#[test]
fn unnamed_module_globals_do_not_merge() {
    let kernel = |name: &str| {
        Kernel::build(name, || {
            let g = module_global::<i32>(None, 1, AddressSpace::Gmem, None, false);
            g.store(0, 1i32);
        })
    };
    let e = KernelModule::new("mk_anon", [kernel("mk_anon_a"), kernel("mk_anon_b")]).unwrap_err();
    assert!(e.message().contains("unnamed module global"), "{e}");
    // One kernel's unnamed global is fine.
    let named = Kernel::build("mk_anon_c", || {
        let g = module_global::<i32>(Some("mk_anon_named"), 1, AddressSpace::Gmem, None, false);
        g.store(0, 1i32);
    });
    KernelModule::new("mk_anon", [kernel("mk_anon_a"), named]).unwrap();
}
