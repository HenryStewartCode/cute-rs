//! Ahead-of-time export: the signature's text format, the export manifest and its version and
//! target checks, golden C headers for a static kernel and for `runtime_gemm` (whose `n` and
//! `k` are launch-time values), and the generated C checked by the system C compiler against a
//! minimal stub `cuda.h`. Files go to Cargo's per-test temporary directory and are removed.
//!
//! `CUTE_RS_BLESS=1` rewrites the golden headers.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/runtime_gemm.rs"]
mod runtime_gemm;

mod abi_kernels;

use std::path::{Path, PathBuf};
use std::process::Command;

use cute::ParamKind;
use cute::aot::{ExportOptions, ExportedKernel, ImageChoice};
use cute::ir::DType;
use cute::prelude::*;

/// A kernel with a pointer of known length, a divisible scalar, a 64-bit scalar, a float
/// scalar and a by-value parameter.
fn leaves() -> Kernel {
    Kernel::builder("aot_leaves").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((64,)));
        let n = param_extent("n", 4);
        let wide = param_extent_i64("wide", 1);
        let alpha = param_scalar::<f32>("alpha");
        let s = param_bytes("s", 16, 8, false);
        let i = thread_idx().x;
        if_(i.lt(n), || x.store(i, s.load::<f32>(0) * alpha + wide.to::<f32>()));
    })
}

/// `kernel`'s export for `target` at PTX `ptx`, with a stand-in PTX and, if asked, a cubin.
fn exported(kernel: &Kernel, target: Arch, ptx: u32, cubin: Option<Arch>) -> ExportedKernel {
    let mut sig = kernel.signature();
    sig.target = Some(target);
    sig.ptx_version = Some(ptx);
    let text = format!(".version {}.{}\n.target {}\n", ptx / 10, ptx % 10, target.chip());
    ExportedKernel::new(sig, Some(text), cubin.map(|t| (vec![0x7f, b'E', b'L', b'F', 0, 1, 2, 3], t)))
        .unwrap_or_else(|e| panic!("{e}"))
}

/// A fresh directory under Cargo's temporary directory for tests.
fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("aot-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn signature_text_round_trips() {
    let mut kernels = vec![leaves(), runtime_gemm::build(), runtime_gemm::build_dynamic()];
    kernels.extend([
        abi_kernels::globals_kernel(),
        abi_kernels::readonly_kernel(),
        abi_kernels::grid_constant_kernel(),
    ]);
    kernels.push(abi_kernels::static_smem_kernel());
    for k in &kernels {
        let mut sig = k.signature();
        sig.target = Some(Arch::Sm89);
        sig.ptx_version = Some(78);
        let text = sig.to_text();
        let back = KernelSignature::from_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        let mut want = sig.clone();
        want.smem.starts.clear();
        assert_eq!(back, want, "{text}");
        assert_eq!(back.to_text(), text);
    }
    let text = exported(&leaves(), Arch::Sm89, 78, None).signature.to_text();
    assert_eq!(
        text,
        "kernel aot_leaves\ntarget sm_89\nptx_version 78\nblock_dim 32 1 1\nsmem static_bytes=0 in_window=false \
         dynamic_start=0 window_align=16 dynamic_region=false\nparam x pointer Float32 len=64 align=16\nparam n scalar \
         Int32 multiple_of=4\nparam wide scalar Int64 multiple_of=1\nparam alpha scalar Float32 multiple_of=1\nparam s \
         by_value size=16 align=8 grid_constant=false\n"
    );
}

#[test]
fn malformed_signatures_are_rejected() {
    let good = "kernel k\nsmem static_bytes=0 in_window=false dynamic_start=0 window_align=16 dynamic_region=false\n";
    assert!(KernelSignature::from_text(good).is_ok());
    let cases = [
        (
            "smem static_bytes=0 in_window=false dynamic_start=0 window_align=16 dynamic_region=false\n",
            1,
            "no `kernel` line",
        ),
        ("kernel k\n", 1, "no `smem` line"),
        ("kernel k\nkernel j\n", 2, "`kernel` appears twice"),
        ("kernel k\nfoo 1\n", 2, "unknown item `foo`"),
        ("kernel k\ntarget sm_75\n", 2, "unknown GPU architecture `sm_75`"),
        ("kernel k\nblock_dim 32 1\n", 2, "block_dim takes three numbers"),
        ("kernel k\nptx_version x\n", 2, "ptx_version must be a number"),
        ("kernel k\nparam x pointer Float99 len=4 align=16\n", 2, "unknown element type `Float99`"),
        ("kernel k\nparam x pointer Float32 align=16 len=4\n", 2, "expected `len=...`, found `align=16`"),
        ("kernel k\nparam x pointer Float32 len=4\n", 2, "expected 2 fields"),
        ("kernel k\nparam x scalar Int32 multiple_of=0\n", 2, "multiple_of must be at least 1"),
        (
            "kernel k\nparam x scalar Int32 multiple_of=1\nparam x scalar Int32 multiple_of=1\n",
            3,
            "parameter `x` appears twice",
        ),
        ("kernel k\nparam x tensor\n", 2, "expected `pointer <type> ...`"),
        ("kernel k\nparam x by_value size=16 align=8 grid_constant=yes\n", 2, "grid_constant must be true or false"),
        ("kernel k\nglobal g Float32 len=4 constant=true readonly=true init=0g\n", 2, "init must be hex bytes"),
        ("kernel k\nsmem static_bytes=0\n", 2, "expected 5 fields"),
        (
            "kernel k\nsmem static_bytes=4294967296 in_window=false dynamic_start=0 window_align=16 dynamic_region=false\n",
            2,
            "static_bytes must be at most 4294967295",
        ),
        ("kernel k\nparam x pointer Float64 len=18446744073709551615 align=8\n", 2, "more bytes than"),
        (
            "kernel k\nglobal g Float32 len=4 constant=true readonly=true init=-\nglobal g Int32 len=1 constant=false \
             readonly=false init=-\n",
            3,
            "global `g` appears twice",
        ),
        ("kernel k\nglobal g Int32 len=1 constant=false readonly=false init=0000000000\n", 2, "5 bytes of init"),
    ];
    for (body, line, want) in cases {
        let text = if body.contains("smem") || want.contains("smem") {
            body.to_string()
        } else {
            format!("{body}{}", &good[9..])
        };
        let e = KernelSignature::from_text(&text).expect_err(&text);
        assert!(e.message.contains(want), "{text}: {e}");
        assert_eq!(e.line, line, "{text}: {e}");
    }
}

#[test]
fn manifest_stamp_and_compatibility() {
    let k = exported(&leaves(), Arch::Sm89, 78, Some(Arch::Sm89));
    let manifest = k.manifest(Some("k.ptx"), Some("k.cubin"));
    assert!(manifest.starts_with(&format!("cute-rs-export 1\ncrate {}\ncubin_target sm_89\n", cute::VERSION)));
    let back = ExportedKernel::from_parts(&manifest, k.ptx.as_deref(), k.cubin.as_deref()).unwrap();
    assert_eq!(back, k);

    // The cubin where the device runs it as a binary; the PTX elsewhere.
    assert_eq!(k.choose_image((8, 9), Some(87)), Ok(ImageChoice::Cubin));
    assert_eq!(k.choose_image((9, 0), Some(87)), Ok(ImageChoice::Ptx));
    let e = k.choose_image((8, 6), Some(87)).unwrap_err();
    assert!(
        e.message().contains("compiled for sm_89, which a device of compute capability 8.6 (sm_86) cannot run"),
        "{e}"
    );

    // Edited stamps: another format version, an `a` target, newer PTX than the driver's.
    let e = ExportedKernel::from_parts(&manifest.replacen("cute-rs-export 1", "cute-rs-export 2", 1), None, None)
        .unwrap_err();
    assert!(e.message().contains("the export is format version 2, and this cute-rs"), "{e}");
    let e =
        ExportedKernel::from_parts(&manifest.replacen("cute-rs-export 1", "something 1", 1), None, None).unwrap_err();
    assert!(e.message().contains("not a cute-rs export manifest"), "{e}");
    let edited = manifest.replace("target sm_89", "target sm_90a").replace("cubin_target sm_89", "cubin_target sm_90a");
    let k90 = ExportedKernel::from_parts(&edited, k.ptx.as_deref(), k.cubin.as_deref()).unwrap();
    let e = k90.choose_image((8, 9), Some(87)).unwrap_err();
    assert!(e.message().contains("compiled for sm_90a, which a device of compute capability 8.9"), "{e}");
    let k88 = ExportedKernel::from_parts(&manifest.replace("ptx_version 78", "ptx_version 88"), k.ptx.as_deref(), None);
    let e = k88.unwrap_err();
    assert!(e.message().contains("only a cubin has one"), "{e}");
    let edited = manifest.replace("ptx_version 78", "ptx_version 88").replace("cubin_target sm_89\n", "");
    let k88 = ExportedKernel::from_parts(&edited, k.ptx.as_deref(), None).unwrap();
    let e = k88.choose_image((8, 9), Some(87)).unwrap_err();
    assert!(
        e.message()
            .contains("is PTX ISA 8.8, but the CUDA driver loads PTX ISA 8.7 at most (PTX ISA 8.8 needs CUDA 12.9)"),
        "{e}"
    );
    assert_eq!(k88.choose_image((8, 9), None), Ok(ImageChoice::Ptx));
    let e = ExportedKernel::from_parts(&manifest.replace("image ptx k.ptx\n", "image ptx k.ptx\nbogus\n"), None, None)
        .unwrap_err();
    assert!(e.message().contains("line 5: unexpected line `bogus`"), "{e}");
    let e = ExportedKernel::from_parts(&manifest.replace("len=64 align=16", "len=64 align=x"), None, None).unwrap_err();
    assert!(e.message().contains("line 12: align must be a number"), "{e}");
    let e = ExportedKernel::from_parts(&manifest, None, None).unwrap_err();
    assert!(e.message().contains("has neither PTX nor a cubin"), "{e}");
    // Images are files beside the manifest, not paths elsewhere.
    for f in ["../k.ptx", "/etc/k.ptx", ".k.ptx"] {
        let e = ExportedKernel::from_parts(&manifest.replace("image ptx k.ptx", &format!("image ptx {f}")), None, None)
            .unwrap_err();
        assert!(e.message().contains("is not a file name beside the manifest"), "{f}: {e}");
    }
}

/// Compare `text` with the golden file `name`, or rewrite it under `CUTE_RS_BLESS`.
fn golden(name: &str, text: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/aot_golden").join(name);
    if std::env::var_os("CUTE_RS_BLESS").is_some() {
        std::fs::write(&path, text).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(text == want, "{name} differs from the golden file; CUTE_RS_BLESS=1 rewrites it:\n{text}");
}

#[test]
fn golden_headers() {
    let k = exported(&leaves(), Arch::Sm89, 78, Some(Arch::Sm89));
    golden("aot_leaves.h", &k.c_header("aot_leaves", "leaves").replace(cute::VERSION, "@VERSION@"));
    let k = exported(&runtime_gemm::build(), Arch::Sm80, 70, None);
    golden("runtime_gemm.h", &k.c_header("runtime_gemm", "").replace(cute::VERSION, "@VERSION@"));
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

/// The system C compiler, if there is one.
fn cc() -> Option<String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    match Command::new(&cc).arg("--version").output() {
        Ok(o) if o.status.success() => Some(cc),
        _ => {
            eprintln!("skipping: no C compiler `{cc}`");
            None
        }
    }
}

#[test]
fn generated_c_compiles_against_a_stub_cuda_h() {
    let Some(cc) = cc() else { return };
    let dir = scratch("stub");
    let stub = dir.join("stub");
    std::fs::create_dir_all(&stub).unwrap();
    std::fs::write(stub.join("cuda.h"), STUB_CUDA_H).unwrap();
    let no_params = Kernel::builder("aot_empty").build(|| {
        let _ = thread_idx().x;
    });
    let mut globals = abi_kernels::globals_kernel().signature();
    (globals.target, globals.ptx_version) = (Some(Arch::Sm80), Some(70));
    let cases = [
        ("aot_leaves", exported(&leaves(), Arch::Sm89, 78, Some(Arch::Sm89))),
        ("runtime_gemm", exported(&runtime_gemm::build(), Arch::Sm80, 70, None)),
        ("aot_empty", exported(&no_params, Arch::Sm100f, 88, None)),
        ("aot_smem", exported(&abi_kernels::static_smem_kernel(), Arch::Sm90a, 80, None)),
        ("aot_globals", ExportedKernel::new(globals, Some(".version 7.0\n".into()), None).unwrap()),
    ];
    let options = ExportOptions { compile_object: true, cc: Some(cc.clone().into()), cuda_include: Some(stub.clone()) };
    for (name, k) in cases {
        let report = k.write_c(&dir, name, "", &options).unwrap_or_else(|e| panic!("{e}"));
        for f in [&report.header, &report.source, &report.manifest, &report.ptx] {
            assert!(f.is_file(), "{}", f.display());
        }
        assert_eq!(report.cubin.is_some(), k.cubin.is_some());
        // Strict warnings on the header and the source, as C and as C++ for the header.
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
        let object = report.object.as_ref().unwrap_or_else(|| panic!("{name}.o: {:?}", report.notes));
        assert!(object.is_file());
        // The manifest reads back to the same kernel.
        assert_eq!(ExportedKernel::read(&report.manifest).unwrap(), k);
    }
    // No cuda.h: the object is skipped with a note.
    let k = exported(&leaves(), Arch::Sm89, 78, None);
    let options = ExportOptions { cuda_include: Some(dir.join("nowhere")), ..ExportOptions::default() };
    let report = k.write_c(&dir, "no_object", "x", &options).unwrap();
    assert!(report.object.is_none());
    assert!(report.notes.iter().any(|n| n.contains("no cubin")), "{:?}", report.notes);
    assert!(report.notes.iter().any(|n| n.contains("no_object.o was not built: no cuda.h")), "{:?}", report.notes);
    let e = k.write_c(&dir, "bad name", "x", &options).unwrap_err();
    assert!(e.message().contains("`bad name` is not a file name"), "{e}");
    // A name the manifest cannot carry is an error, not a panic.
    let mut odd = k.clone();
    odd.signature.params[0].name = "x y".into();
    let e = odd.write_c(&dir, "odd", "x", &options).unwrap_err();
    assert!(e.message().contains("parameter name `x y` cannot be written"), "{e}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Names that are not C: comment terminators, trigraphs, keywords, narrow divisors.
#[test]
fn hostile_names_stay_inside_c() {
    let Some(cc) = cc() else { return };
    let dir = scratch("names");
    let stub = dir.join("stub");
    std::fs::create_dir_all(&stub).unwrap();
    std::fs::write(stub.join("cuda.h"), STUB_CUDA_H).unwrap();
    let mut k = exported(&leaves(), Arch::Sm89, 78, None);
    k.signature.name = "k*/??/x".into();
    k.signature.params[0].name = "class".into();
    k.signature.params[1].name = "n*/".into();
    // An Int8 leaf with a divisor past its range (only 0 divides by it).
    k.signature.params[2].kind = ParamKind::Scalar { dtype: DType::I8, multiple_of: 256 };
    let options = ExportOptions { compile_object: false, cc: None, cuda_include: None };
    let report = k.write_c(&dir, "names", "", &options).unwrap_or_else(|e| panic!("{e}"));
    let source = std::fs::read_to_string(&report.source).unwrap();
    assert!(source.contains("\"k*/\\?\\?/x\""), "{source}");
    assert!(source.contains("(long long)a->wide % 256LL != 0"), "{source}");
    let header = std::fs::read_to_string(&report.header).unwrap();
    assert!(header.contains("CUdeviceptr class_;"), "{header}");
    for (file, lang) in [(&report.header, "c"), (&report.source, "c"), (&report.header, "c++")] {
        let std = if lang == "c" { "-std=c99" } else { "-std=c++11" };
        let out = Command::new(&cc)
            .args(["-fsyntax-only", "-Wall", "-Wextra", "-Werror", "-pedantic", "-Wtrigraphs", std, "-x", lang, "-I"])
            .arg(&stub)
            .arg("-I")
            .arg(&dir)
            .arg(file)
            .output()
            .unwrap();
        assert!(out.status.success(), "{}:\n{}", file.display(), String::from_utf8_lossy(&out.stderr));
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// With an `mlir-opt`: a compiled kernel exports its own PTX, target and PTX version.
#[test]
fn compiled_kernel_exports() {
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt");
        return;
    };
    let compiled = leaves()
        .compile_with(&toolchain, &CompileOptions::new().gpu_arch(Arch::Sm89))
        .unwrap_or_else(|e| panic!("{e}"));
    let dir = scratch("compiled");
    let options = ExportOptions { compile_object: false, ..ExportOptions::default() };
    let report = compiled.export_c_with(&dir, "aot_leaves", "leaves", &options).unwrap_or_else(|e| panic!("{e}"));
    let k = ExportedKernel::read(&report.manifest).unwrap();
    assert_eq!(k.ptx.as_deref(), Some(compiled.ptx()));
    assert_eq!((k.target(), k.ptx_version()), (Arch::Sm89, compiled.ptx_version()));
    assert_eq!(k.signature.params, compiled.signature().params);
    assert_eq!(k.cubin.as_deref(), compiled.cubin());
    let header = std::fs::read_to_string(&report.header).unwrap();
    assert!(header.contains("#define LEAVES_TARGET \"sm_89\""), "{header}");
    assert_eq!(compiled.to_exported(), k);
    std::fs::remove_dir_all(&dir).unwrap();
}
