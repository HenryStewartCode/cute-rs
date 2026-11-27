//! External memory on a real GPU: DLPack tensors over device memory allocated with the raw
//! driver API (converted explicitly and implicitly), raw device pointers and buffer views
//! (with their alignment errors), and a null pointer for an unused parameter. Needs the
//! `cuda` feature; skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_external
//! ```

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/call_bypass_dlpack.rs"]
mod call_bypass_dlpack;
#[path = "support/dlpack_alloc.rs"]
mod dlpack_alloc;

use cute::cuda::{CudaContext, CudaFunction, DeviceArg, DevicePtr};
use cute::external::dlpack::{DLDevice, device_type};
use cute::prelude::*;
use dlpack_alloc::{Owned, f32_type, managed};

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

/// `cuMemAlloc_v2`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2` and `cuMemFree_v2`, called directly
/// (signatures from cuda.h), as another library allocating device memory would.
#[cfg(unix)]
mod raw_driver {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    type Alloc = unsafe extern "C" fn(*mut u64, usize) -> c_int;
    type HtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> c_int;
    type DtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> c_int;
    type Free = unsafe extern "C" fn(u64) -> c_int;

    /// Device memory from the raw driver API, freed on drop.
    pub struct RawAlloc {
        pub ptr: u64,
        pub bytes: usize,
        free: Free,
        dtoh: DtoH,
    }

    fn sym(name: &std::ffi::CStr) -> *mut c_void {
        // SAFETY: the driver library and a NUL-terminated name.
        let p = unsafe { dlsym(dlopen(c"libcuda.so.1".as_ptr(), 2), name.as_ptr()) };
        assert!(!p.is_null(), "{name:?}");
        p
    }

    impl RawAlloc {
        /// `values` uploaded to fresh device memory (the context must be current).
        pub fn upload(values: &[f32]) -> RawAlloc {
            let bytes = std::mem::size_of_val(values);
            // SAFETY: entry points with these signatures; valid pointers and sizes.
            unsafe {
                let alloc: Alloc = std::mem::transmute(sym(c"cuMemAlloc_v2"));
                let htod: HtoD = std::mem::transmute(sym(c"cuMemcpyHtoD_v2"));
                let mut ptr = 0u64;
                assert_eq!(alloc(&mut ptr, bytes), 0);
                assert_eq!(htod(ptr, values.as_ptr().cast(), bytes), 0);
                RawAlloc {
                    ptr,
                    bytes,
                    free: std::mem::transmute::<*mut c_void, Free>(sym(c"cuMemFree_v2")),
                    dtoh: std::mem::transmute::<*mut c_void, DtoH>(sym(c"cuMemcpyDtoH_v2")),
                }
            }
        }

        pub fn to_vec(&self) -> Vec<f32> {
            let mut out = vec![0f32; self.bytes / 4];
            // SAFETY: `out` holds `bytes` bytes.
            assert_eq!(unsafe { (self.dtoh)(out.as_mut_ptr().cast(), self.ptr, self.bytes) }, 0);
            out
        }
    }

    impl Drop for RawAlloc {
        fn drop(&mut self) {
            // SAFETY: allocated by cuMemAlloc and freed once.
            unsafe { (self.free)(self.ptr) };
        }
    }
}

use raw_driver::RawAlloc;

type AddParams<H> = (DlTensor<f32, H>, DlTensor<f32, H>, DlTensor<f32, H>);

/// `c = a + b` over `m x n` tensors, a grid-stride loop of one element per thread.
fn add_kernel<H: DlpackSource>(host: &(H, H, H)) -> TypedKernel<AddParams<H>> {
    Kernel::builder("dl_add_gpu").block_dim([128, 1, 1]).jit_for(
        host,
        |a: DlTensor<f32, H>, b: DlTensor<f32, H>, c: DlTensor<f32, H>| {
            let m: Int32 = size(a.tensor(), &[0]).into_value();
            let n: Int32 = size(a.tensor(), &[1]).into_value();
            let start = block_idx().x * 128 + thread_idx().x;
            for_range(start, m * n, grid_dim().x * 128, |i| {
                let (r, k) = (i / n, i % n);
                c.store((r, k), a.load((r, k)) + b.load((r, k)));
            });
        },
    )
}

const M: usize = 37;
const N: usize = 64;

fn inputs() -> (Vec<f32>, Vec<f32>) {
    ((0..M * N).map(|i| (i % 31) as f32 - 15.0).collect(), (0..M * N).map(|i| (i % 7) as f32 * 0.25).collect())
}

fn cfg() -> LaunchConfig {
    LaunchConfig::new([4, 1, 1], [128, 1, 1])
}

/// The simulator's answer, on host DLPack tensors.
fn simulated(a: &[f32], b: &[f32]) -> Vec<f32> {
    let cpu = DLDevice { device_type: device_type::KDL_CPU, device_id: 0 };
    let mk = |v: &[f32]| managed(Owned::f32(v), f32_type(), &[M as i64, N as i64], None, 0, 0, cpu, true);
    let ((ta, _), (tb, _), (tc, cc)) = (mk(a), mk(b), mk(&vec![0.0; M * N]));
    let mut host = (ta, tb, tc);
    add_kernel(&host).simulate(cfg(), &mut host).unwrap_or_else(|e| panic!("{e}"));
    drop(host);
    cc.data_f32()
}

/// Element-wise addition on DLPack tensors over raw-driver device memory, converted
/// explicitly (`from_dlpack(t).mark_layout_dynamic()`) and implicitly (the `DlpackTensor`
/// itself), matching the simulator; and on host DLPack tensors copied in and back.
#[test]
fn dlpack_device_tensors_match_the_simulator() {
    let Some((ctx, tc)) = device() else { return };
    let (a, b) = inputs();
    let want = simulated(&a, &b);
    let gpu = DLDevice { device_type: device_type::KDL_CUDA, device_id: 0 };
    for implicit in [false, true] {
        let (da, db, dc) = (RawAlloc::upload(&a), RawAlloc::upload(&b), RawAlloc::upload(&vec![0.0; M * N]));
        let mk = |p: &RawAlloc| managed(Owned::Raw(p.ptr), f32_type(), &[M as i64, N as i64], None, 0, 0, gpu, true);
        let ((ta, ca), (tb, cb), (tcc, ccc)) = (mk(&da), mk(&db), mk(&dc));
        if implicit {
            let mut host = (ta, tb, tcc);
            let k = add_kernel(&host);
            k.run_cuda(&ctx, &tc, cfg(), &mut host).unwrap_or_else(|e| panic!("{e}"));
        } else {
            let ext = |t| from_dlpack(t, FromDlpack::default()).unwrap().mark_layout_dynamic(None).unwrap();
            let mut host = (ext(ta), ext(tb), ext(tcc));
            assert_eq!(host.0.layout_string(), "(?,?):(?{i64},1)");
            let k = add_kernel(&host);
            k.run_cuda(&ctx, &tc, cfg(), &mut host).unwrap_or_else(|e| panic!("{e}"));
        }
        assert_eq!((ca.get(), cb.get(), ccc.get()), (1, 1, 1), "each deleter once");
        assert_eq!(dc.to_vec(), want, "implicit: {implicit}");
    }
    // Host memory, copied in and written back.
    let cpu = DLDevice { device_type: device_type::KDL_CPU, device_id: 0 };
    let mk = |v: &[f32]| managed(Owned::f32(v), f32_type(), &[M as i64, N as i64], None, 0, 0, cpu, true);
    let ((ta, _), (tb, _), (tcc, ccc)) = (mk(&a), mk(&b), mk(&vec![0.0; M * N]));
    let mut host = (ta, tb, tcc);
    add_kernel(&host).run_cuda(&ctx, &tc, cfg(), &mut host).unwrap_or_else(|e| panic!("{e}"));
    drop(host);
    assert_eq!(ccc.data_f32(), want);
    // Memory of another device ordinal is refused.
    let (da, db, dc) = (RawAlloc::upload(&a), RawAlloc::upload(&b), RawAlloc::upload(&a));
    let other = DLDevice { device_type: device_type::KDL_CUDA, device_id: 3 };
    let mk = |p: &RawAlloc, d| managed(Owned::Raw(p.ptr), f32_type(), &[M as i64, N as i64], None, 0, 0, d, true);
    let mut host = (mk(&da, gpu).0, mk(&db, gpu).0, mk(&dc, other).0);
    let e = add_kernel(&host).run_cuda(&ctx, &tc, cfg(), &mut host).unwrap_err();
    assert!(e.to_string().contains("argument `arg2` is memory of CUDA device 3"), "{e}");
}

/// `call_bypass_dlpack.py` on raw device pointers (`make_ptr` with `assumed_align=32`),
/// matching the simulator, and the alignment errors, which name the address and the
/// alignment.
#[test]
fn raw_pointers_and_views() {
    let Some((ctx, tc)) = device() else { return };
    let (m, n, k, l) = (32, 16, 8, 2);
    let want = call_bypass_dlpack::run(m, n, k, l);
    let (a, b) = call_bypass_dlpack::inputs(m, n, k, l);
    let kernel = call_bypass_dlpack::build();
    let f = load(&ctx, &tc, &kernel);
    let (da, db) = (ctx.upload_slice(&a).unwrap(), ctx.upload_slice(&b).unwrap());
    let dc = ctx.alloc_bytes(2 * m * n * l + 64).unwrap();
    let ptr = |buf: &cute::cuda::DeviceBuffer<'_>, offset: u64, align: usize| {
        // SAFETY: an address inside a live buffer of this context, with its remaining bytes.
        unsafe {
            DevicePtr::from_raw(DType::F16, buf.as_raw() + offset, Some(align), Some(buf.bytes() - offset as usize))
        }
    };
    fn with_c<'a>(a: DevicePtr, b: DevicePtr, c: DeviceArg<'a>, s: [usize; 4]) -> Vec<DeviceArg<'a>> {
        let mut v = vec![DeviceArg::Ptr(a), DeviceArg::Ptr(b), c];
        v.extend(s.map(|x| DeviceArg::Scalar(KernelArg::scalar(x as i32))));
        v
    }
    let (pa, pb) = (ptr(&da, 0, 32).unwrap(), ptr(&db, 0, 32).unwrap());
    let args = |c| with_c(pa, pb, c, [m, n, k, l]);
    let cfg = call_bypass_dlpack::launch(m, n, l);
    // C at a 32-byte offset into its buffer, as a raw pointer.
    let c_args = args(DeviceArg::Ptr(ptr(&dc, 32, 32).unwrap()));
    ctx.launch_on_device(&kernel, &f, cfg, &c_args).unwrap_or_else(|e| panic!("{e}"));
    let mut got = vec![F16::from_f32(0.0); m * n * l];
    dc.read(32, &mut got).unwrap();
    assert_eq!(got, want);
    // The same through a borrowed view of the buffer.
    dc.memset(0).unwrap();
    let c_args = args(dc.view(32, 2 * m * n * l).unwrap());
    ctx.launch_on_device(&kernel, &f, cfg, &c_args).unwrap_or_else(|e| panic!("{e}"));
    dc.read(32, &mut got).unwrap();
    assert_eq!(got, want);
    // make_ptr's own check: an address off its assumed alignment is refused.
    let e = ptr(&dc, 2, 32).unwrap_err();
    assert!(e.message.contains("not a multiple of the pointer's assumed alignment of 32 bytes"), "{e}");
    // A pointer that promises less than the parameter declares is refused at launch.
    let c_args = args(DeviceArg::Ptr(ptr(&dc, 2, 2).unwrap()));
    let e = ctx.launch_on_device(&kernel, &f, cfg, &c_args).unwrap_err();
    assert!(e.message.contains("argument 2") && e.message.contains("not 32-byte aligned"), "{e}");
    let c_args = args(dc.view(16, 2 * m * n * l).unwrap());
    let e = ctx.launch_on_device(&kernel, &f, cfg, &c_args).unwrap_err();
    assert!(e.message.contains("not 32-byte aligned"), "{e}");
    // A view past the end, and a raw pointer of the wrong element type.
    assert!(dc.view(64, 2 * m * n * l + 8).is_err());
    // SAFETY: a live buffer.
    let wrong = unsafe { DevicePtr::from_raw(DType::F32, dc.as_raw(), Some(32), None) }.unwrap();
    let e = ctx.launch_on_device(&kernel, &f, cfg, &args(DeviceArg::Ptr(wrong))).unwrap_err();
    assert!(e.message.contains("is a pointer to f16, but the raw pointer given is to f32"), "{e}");
}

/// A null pointer (CuTe DSL's `nullptr`) for a parameter the launch does not use.
#[test]
fn a_null_pointer_for_an_unused_parameter() {
    let Some((ctx, tc)) = device() else { return };
    let kernel = Kernel::builder("maybe_bias").block_dim([64, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let bias = param_ptr::<f32>("bias");
        let use_bias = param_scalar::<i32>("use_bias");
        let i = thread_idx().x;
        if_(use_bias.ne(0), || x.store(i, x.load(i) + bias.load_at(i)));
        if_(use_bias.eq(0), || x.store(i, x.load(i) * 2.0f32));
    });
    let f = load(&ctx, &tc, &kernel);
    let x = ctx.upload_slice(&(0..64).map(|i| i as f32).collect::<Vec<_>>()).unwrap();
    let args = [
        DeviceArg::Buffer(&x),
        DeviceArg::Ptr(DevicePtr::null(DType::F32)),
        DeviceArg::Scalar(KernelArg::scalar(0i32)),
    ];
    assert!(DevicePtr::null(DType::F32).is_null());
    ctx.launch_on_device(&kernel, &f, LaunchConfig::new([1, 1, 1], [64, 1, 1]), &args)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(x.to_vec::<f32>().unwrap(), (0..64).map(|i| i as f32 * 2.0).collect::<Vec<_>>());
}
