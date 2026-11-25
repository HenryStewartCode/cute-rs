//! Kernels and device functions shared by `tests/extern_calls.rs` and `tests/gpu_extern.rs`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cute::ffi::{AddressSpace, BitCode};
use cute::prelude::*;

pub const N: usize = 64;

/// `x[i] = i`, `y[i] = 100 + i`.
pub fn inputs() -> (Vec<f32>, Vec<f32>) {
    ((0..N).map(|i| i as f32).collect(), (0..N).map(|i| 100.0 + i as f32).collect())
}

/// What [`bitcode_kernel`] leaves in `x`: `3 x + (y + 1)`.
pub fn expected(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(a, b)| 3.0 * a + (b + 1.0)).collect()
}

/// `ext_scale(x) = 3 x`, from the bitcode at `bc`, with a raw-bits simulation.
pub fn ext_scale(bc: &Path, inline: bool) -> ExternFn {
    ExternFn::new("ext_scale")
        .param::<f32>()
        .returns::<f32>()
        .inline(inline)
        .source(BitCode::new(bc))
        .sim(|a| vec![(f32::from_bits(a[0] as u32) * 3.0).to_bits() as u64])
}

/// `ext_load(p, i) = p[i] + 1` on a global pointer, from the bitcode at `bc`, simulated by DSL
/// code that reads through the pointer.
pub fn ext_load(bc: &Path, inline: bool) -> ExternFn {
    ExternFn::new("ext_load")
        .param_ptr(AddressSpace::Gmem)
        .param::<i32>()
        .returns::<f32>()
        .inline(inline)
        .source(BitCode::new(bc))
        .sim_body(|args| {
            let p = args[0].ptr::<f32>();
            let i = args[1].value::<i32>();
            Some((p.offset(i).load() + 1.0f32).into())
        })
}

/// `x[t] = ext_scale(x[t]) + ext_load(&y[0], t)`; `ext_scale` is inlined (when `inline`),
/// `ext_load` never.
pub fn bitcode_kernel(bc: &Path, inline: bool) -> Kernel {
    let (scale, load) = (ext_scale(bc, inline), ext_load(bc, false));
    Kernel::builder(if inline { "ext_inline" } else { "ext_called" }).block_dim([N as u32, 1, 1]).build(move || {
        let x = param_tensor::<f32>("x", Layout::col_major(N));
        let y = param_tensor::<f32>("y", Layout::col_major(N));
        let t = thread_idx().x;
        let a = scale.call_value::<f32>(&[x.load(t).into()]);
        let b = load.call_value::<f32>(&[(&y.ptr()).into(), t.into()]);
        x.store(t, a + b);
    })
}

/// `extern "C" __device__ float axpy(float a, float x, float y) { return a * x + y; }`
pub fn axpy() -> DeviceFunction {
    DeviceFunction::builder("axpy").param::<f32>().param::<f32>().param::<f32>().returns::<f32>().build(|args| {
        let (a, x, y) = (args[0].value::<f32>(), args[1].value::<f32>(), args[2].value::<f32>());
        Some((a * x + y).into())
    })
}

/// `extern "C" __device__ void scale_in_place(float *p, float s) { *p *= s; }`
pub fn scale_in_place() -> DeviceFunction {
    DeviceFunction::builder("scale_in_place").param_ptr::<f32>().param::<f32>().build(|args| {
        let (p, s) = (args[0].ptr::<f32>(), args[1].value::<f32>());
        p.store(p.load() * s);
        None
    })
}

/// `x[t] = axpy(a, x[t], y[t]); scale_in_place(&x[t], 0.5)`, calling the device functions
/// (not inlined: their code is linked at load time).
pub fn axpy_caller() -> Kernel {
    let (f, g) = (axpy(), scale_in_place());
    Kernel::builder("axpy_caller").block_dim([N as u32, 1, 1]).build(move || {
        let x = param_tensor::<f32>("x", Layout::col_major(N));
        let y = param_tensor::<f32>("y", Layout::col_major(N));
        let a = param_scalar::<f32>("a");
        let t = thread_idx().x;
        let r = f.call(&[a.into(), x.load(t).into(), y.load(t).into()]).expect("a result");
        x.store(t, cute::CallArg::Value(r).value::<f32>());
        g.call(&[(&x.ptr().offset(t)).into(), 0.5f32.into_value().into()]);
    })
}

/// What [`axpy_caller`] leaves in `x` for `a`.
pub fn axpy_expected(a: f32, x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(xi, yi)| (a * xi + yi) * 0.5).collect()
}

/// The toolchain and `tests/data/extern_funcs.ll` assembled into a scratch directory, or
/// `None` (with a note) without `mlir-opt` or `llvm-as`.
pub fn toolchain_and_bitcode(tag: &str) -> Option<(Toolchain, PathBuf)> {
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt (set CUTE_MLIR_OPT)");
        return None;
    };
    let beside = toolchain.mlir_opt.with_file_name("llvm-as");
    let llvm_as = if beside.is_file() { beside } else { PathBuf::from("llvm-as") };
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bc = dir.join("extern_funcs.bc");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/extern_funcs.ll");
    match Command::new(&llvm_as).arg(&src).arg("-o").arg(&bc).output() {
        Ok(out) if out.status.success() => Some((toolchain, bc)),
        Ok(out) => panic!("llvm-as failed: {}", String::from_utf8_lossy(&out.stderr)),
        Err(e) => {
            eprintln!("skipping: no llvm-as ({}: {e})", llvm_as.display());
            None
        }
    }
}
