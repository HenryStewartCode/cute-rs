//! External device functions, standalone device functions and flag locks on a GPU, each against the simulator: calls of functions from linked
//! bitcode (inlined and not), libdevice's `__nv_erff` and `__nv_powf` (when the CUDA toolkit's
//! libdevice is installed), a kernel linked at load time with the device functions it calls,
//! relocatable device code (`ptxas -c`, linked into a CUDA C program when `nvcc` is
//! installed), a flag handshake of co-resident blocks, and multicast memory refused on a GPU
//! without it. Skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --features cuda --test gpu_extern
//! ```

#[path = "extern_kernels/mod.rs"]
mod extern_kernels;

use std::path::{Path, PathBuf};
use std::process::Command;

use cute::cuda::{CudaContext, CudaFunction, ModuleImage, MulticastGroup};
use cute::ffi::BitCode;
use cute::prelude::*;
use cute::utils::distributed::{red_add1, spin_lock_atom_cas_acquire_wait, spin_lock_ld_lt_relaxed_wait};
use extern_kernels::*;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn load<'c>(ctx: &'c CudaContext, tc: &Toolchain, k: &Kernel) -> CudaFunction<'c> {
    let arch = ctx.best_target(tc).expect("a target");
    let compiled = k.compile_with(tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{e}"));
    ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"))
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("gpu-extern-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn bitcode_calls_match_the_simulator() {
    let Some((ctx, tc)) = device() else { return };
    let Some((_, bc)) = toolchain_and_bitcode("gpu-extern") else { return };
    for inline in [true, false] {
        let k = bitcode_kernel(&bc, inline);
        let (x, y) = inputs();
        let cfg = LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]);
        let mut sim = vec![KernelArg::f32(x.clone()), KernelArg::f32(y.clone())];
        k.simulate(cfg, &mut sim).unwrap();
        let mut gpu = vec![KernelArg::f32(x.clone()), KernelArg::f32(y.clone())];
        ctx.launch(&k, &load(&ctx, &tc, &k), cfg, &mut gpu).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(gpu[0].as_f32(), expected(&x, &y), "inline {inline}");
        assert_eq!(gpu[0].as_f32(), sim[0].as_f32(), "inline {inline}");
    }
}

/// libdevice's `float __nv_erff(float)` and `float __nv_powf(float, float)`, simulated by
/// cute-rs's own `erf` and `pow`.
#[test]
fn libdevice_through_extern() {
    let Some((ctx, tc)) = device() else { return };
    let libdevice = PathBuf::from("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc");
    if !libdevice.is_file() {
        eprintln!("skipping: no {} (libdevice ships with the CUDA toolkit)", libdevice.display());
        return;
    }
    let erff = ExternFn::new("__nv_erff")
        .param::<f32>()
        .returns::<f32>()
        .convergent(false)
        .source(BitCode::new(&libdevice))
        .sim_body(|a| Some(cute::math::erf(a[0].value::<f32>(), false).into()));
    let powf = ExternFn::new("__nv_powf")
        .param::<f32>()
        .param::<f32>()
        .returns::<f32>()
        .convergent(false)
        .source(BitCode::new(&libdevice))
        .sim_body(|a| Some(cute::math::pow(a[0].value::<f32>(), a[1].value::<f32>(), false).into()));
    let k = Kernel::builder("libdevice_calls").block_dim([128, 1, 1]).build(move || {
        let x = param_tensor::<f32>("x", Layout::col_major(128));
        let e = param_tensor::<f32>("e", Layout::col_major(128));
        let p = param_tensor::<f32>("p", Layout::col_major(128));
        let t = thread_idx().x;
        let v = x.load(t);
        e.store(t, erff.call_value::<f32>(&[v.into()]));
        p.store(t, powf.call_value::<f32>(&[v.into(), 1.5f32.into_value().into()]));
    });
    let xs: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0).abs() / 16.0).collect();
    let cfg = LaunchConfig::new([1, 1, 1], [128, 1, 1]);
    let mut sim = vec![KernelArg::f32(xs.clone()), KernelArg::f32(vec![0.0; 128]), KernelArg::f32(vec![0.0; 128])];
    k.simulate(cfg, &mut sim).unwrap();
    let mut gpu = vec![KernelArg::f32(xs.clone()), KernelArg::f32(vec![0.0; 128]), KernelArg::f32(vec![0.0; 128])];
    ctx.launch(&k, &load(&ctx, &tc, &k), cfg, &mut gpu).unwrap_or_else(|e| panic!("{e}"));
    for out in [1, 2] {
        for (i, (g, s)) in gpu[out].as_f32().iter().zip(sim[out].as_f32()).enumerate() {
            let ulps = (g.to_bits() as i64 - s.to_bits() as i64).abs();
            assert!(ulps <= 2, "output {out}, x = {}: GPU {g}, simulator {s}", xs[i]);
        }
    }
}

/// A kernel calling `axpy` and `scale_in_place` (`.extern .func` in its PTX), linked with
/// their PTX by the driver.
#[test]
fn device_functions_linked_at_load_time() {
    let Some((ctx, tc)) = device() else { return };
    let arch = ctx.best_target(&tc).expect("a target");
    let caller = axpy_caller();
    let ptx = caller.compile_ptx(&tc, arch).unwrap();
    assert!(ptx.contains(".extern .func"), "{ptx}");
    let f_ptx = axpy().compile_ptx(&tc, arch).unwrap();
    let g_ptx = scale_in_place().compile_ptx(&tc, arch).unwrap();
    // Loading the kernel alone fails: the functions are unresolved.
    assert!(ctx.load_linked(&[&ptx], "axpy_caller").is_err());
    let f = ctx.load_linked(&[&ptx, &f_ptx, &g_ptx], "axpy_caller").unwrap_or_else(|e| panic!("{e}"));
    let (x, y) = inputs();
    let cfg = LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]);
    let mut gpu = vec![KernelArg::f32(x.clone()), KernelArg::f32(y.clone()), KernelArg::ScalarF32(3.0)];
    ctx.launch(&caller, &f, cfg, &mut gpu).unwrap_or_else(|e| panic!("{e}"));
    let mut sim = vec![KernelArg::f32(x.clone()), KernelArg::f32(y.clone()), KernelArg::ScalarF32(3.0)];
    caller.simulate(cfg, &mut sim).unwrap();
    assert_eq!(gpu[0].as_f32(), axpy_expected(3.0, &x, &y));
    assert_eq!(gpu[0].as_f32(), sim[0].as_f32());
}

/// Relocatable device code: `ptxas -c`, then (with `nvcc`) device-linked with CUDA C code that
/// declares `extern "C" __device__ float axpy(float, float, float)`; the linked cubin is loaded
/// and launched through the driver.
#[test]
fn relocatable_device_code() {
    let Some((ctx, tc)) = device() else { return };
    let Some(ptxas) = tc.cuda_tools().ptxas.clone() else {
        eprintln!("skipping: no ptxas");
        return;
    };
    let arch = ctx.best_target(&tc).expect("a target");
    let obj = axpy().compile_relocatable(&tc, arch).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(&obj[..4], b"\x7fELF");
    let nvcc = ptxas.path.with_file_name("nvcc");
    if !nvcc.is_file() {
        eprintln!("skipping the CUDA C link: no nvcc beside {}", ptxas.path.display());
        return;
    }
    let dir = scratch("rdc");
    std::fs::write(dir.join("axpy.cubin"), &obj).unwrap();
    let source = r#"
extern "C" __device__ float axpy(float a, float x, float y);
extern "C" __global__ void k(float *out) { out[threadIdx.x] = axpy(2.0f, (float)threadIdx.x, 1.0f); }
"#;
    std::fs::write(dir.join("k.cu"), source).unwrap();
    let sm = format!("-arch=sm_{}", arch.chip().trim_start_matches("sm_").trim_end_matches(['a', 'f']));
    for args in [
        vec!["-rdc=true", sm.as_str(), "-c", "k.cu", "-o", "k.o"],
        vec![sm.as_str(), "-dlink", "-cubin", "k.o", "axpy.cubin", "-o", "linked.cubin"],
    ] {
        let out = Command::new(&nvcc).current_dir(&dir).args(&args).output().unwrap();
        assert!(out.status.success(), "nvcc {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
    let linked = std::fs::read(dir.join("linked.cubin")).unwrap();
    let f = ctx.load_module(ModuleImage::Cubin(&linked), &[], "k").unwrap_or_else(|e| panic!("{e}"));
    // A kernel of the same signature, for the launcher to check and pack the arguments.
    let stub = Kernel::build("k", || {
        let _ = param_tensor::<f32>("out", Layout::col_major(32));
    });
    let mut args = vec![KernelArg::f32(vec![0.0; 32])];
    ctx.launch(&stub, &f, LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].as_f32(), (0..32).map(|i| 2.0 * i as f32 + 1.0).collect::<Vec<f32>>());
}

/// Thread 0 of each block adds one to every block's flag, then waits for its own to reach
/// the block count (acquire CAS, reset to 0, or a relaxed load).
fn handshake(cas: bool) -> Kernel {
    Kernel::build(if cas { "gpu_handshake_cas" } else { "gpu_handshake_ld" }, move || {
        let flags = param_tensor::<i32>("flag", Layout::col_major(8));
        let out = param_tensor::<i32>("out", Layout::col_major(8));
        let n = param_scalar::<i32>("n");
        let b = block_idx().x;
        if_(thread_idx().x.eq(0), || {
            for j in 0..8 {
                if_(n.gt(j), || red_add1(&flags.ptr().offset(j), MemoryOrder::Release, MemoryScope::Gpu));
            }
            if cas {
                spin_lock_atom_cas_acquire_wait(&flags.ptr().offset(b), n, 0, MemoryScope::Gpu);
            } else {
                spin_lock_ld_lt_relaxed_wait(&flags.ptr().offset(b), n, MemoryScope::Gpu);
            }
            out.store(b, b + 1);
        });
    })
}

#[test]
fn flag_handshake_on_the_gpu() {
    let Some((ctx, tc)) = device() else { return };
    for cas in [true, false] {
        let k = handshake(cas);
        let f = load(&ctx, &tc, &k);
        for blocks in 2..=8u32 {
            let cfg = LaunchConfig::new([blocks, 1, 1], [32, 1, 1]).with_cooperative(true);
            let args =
                || vec![KernelArg::i32(vec![0; 8]), KernelArg::i32(vec![0; 8]), KernelArg::ScalarI32(blocks as i32)];
            let mut gpu = args();
            ctx.launch(&k, &f, cfg, &mut gpu).unwrap_or_else(|e| panic!("{e}"));
            let mut sim = args();
            k.simulate(cfg, &mut sim).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(gpu[0].as_i32(), sim[0].as_i32(), "cas {cas}, {blocks} blocks");
            assert_eq!(gpu[1].as_i32(), sim[1].as_i32(), "cas {cas}, {blocks} blocks");
            let want: Vec<i32> = (0..8).map(|b| if b < blocks as i32 { b + 1 } else { 0 }).collect();
            assert_eq!(gpu[1].as_i32(), want);
        }
    }
}

#[test]
fn multicast_needs_hardware_support() {
    let Some((ctx, _)) = device() else { return };
    if ctx.multicast_supported().unwrap() {
        let g = MulticastGroup::new(&[&ctx], 1 << 20).unwrap_or_else(|e| panic!("{e}"));
        assert!(g.size() >= 1 << 20 && g.multicast_ptr(0) != 0 && g.unicast_ptr(0) != 0);
        return;
    }
    let err = MulticastGroup::new(&[&ctx], 1 << 20).err().expect("no multicast support");
    assert!(err.to_string().contains("does not support multicast memory"), "{err}");
}
