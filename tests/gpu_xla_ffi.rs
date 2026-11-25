//! The XLA FFI handler on a real GPU: call frames and an API table
//! built by hand around device buffers, the handler called as XLA would call it, the results
//! checked against the host's sum, plus the compile cache, aliasing, the stages and the
//! diagnostics. Skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_xla_ffi
//! ```

#[path = "support/tvm_ffi_kernels.rs"]
mod kernels;

use std::cell::Cell;
use std::ffi::{CStr, c_char, c_void};

use cute::cuda::CudaContext;
use cute::prelude::*;
use cute::tvm_ffi::TvmFfiFunction;
use cute::xla_ffi::*;
use kernels::*;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

thread_local! {
    static STREAM: Cell<usize> = const { Cell::new(0) };
}

struct TestError {
    code: i32,
    message: String,
}

unsafe extern "C" fn test_error_create(args: *mut XlaErrorCreateArgs) -> *mut XlaError {
    // SAFETY: the handler passes valid arguments with a NUL-terminated message.
    let a = unsafe { &*args };
    let message = unsafe { CStr::from_ptr(a.message) }.to_string_lossy().into_owned();
    Box::into_raw(Box::new(TestError { code: a.errc, message })) as *mut XlaError
}

unsafe extern "C" fn test_stream_get(args: *mut XlaStreamGetArgs) -> *mut XlaError {
    // SAFETY: valid arguments.
    unsafe { (*args).stream = STREAM.with(|s| s.get()) as *mut c_void };
    std::ptr::null_mut()
}

enum Attr {
    Str(String, &'static str),
    I64(String, i64),
    F32(String, f32),
    Order(String, Vec<i64>),
}

/// A call frame over buffers (dtype code, address, dims), its API table and everything the
/// pointers point at.
struct Frame {
    args: Vec<(i32, u64, Vec<i64>)>,
    rets: Vec<(i32, u64, Vec<i64>)>,
    attrs: Vec<Attr>,
    stage: i32,
}

impl Frame {
    fn new(target: &str) -> Frame {
        Frame {
            args: vec![],
            rets: vec![],
            attrs: vec![Attr::Str("cute_target".into(), Box::leak(target.to_string().into_boxed_str()))],
            stage: stage::EXECUTE,
        }
    }

    fn arg(mut self, address: u64, dims: &[i64]) -> Frame {
        self.args.push((data_type::F32, address, dims.to_vec()));
        self
    }

    fn ret(mut self, address: u64, dims: &[i64]) -> Frame {
        self.rets.push((data_type::F32, address, dims.to_vec()));
        self
    }

    fn call(&self) -> Result<(), (i32, String)> {
        let api = XlaApi {
            struct_size: std::mem::size_of::<XlaApi>(),
            extension_start: std::ptr::null_mut(),
            api_version: XlaApiVersion {
                struct_size: std::mem::size_of::<XlaApiVersion>(),
                extension_start: std::ptr::null_mut(),
                major_version: 0,
                minor_version: 1,
            },
            internal_api: std::ptr::null_mut(),
            error_create: Some(test_error_create),
            error_get_details: std::ptr::null_mut(),
            error_destroy: std::ptr::null_mut(),
            handler_register: std::ptr::null_mut(),
            stream_get: Some(test_stream_get),
        };
        let mk = |dtype: i32, address: u64, dims: &Vec<i64>| {
            let mut dims = dims.clone();
            let b = Box::new(XlaBuffer {
                struct_size: std::mem::size_of::<XlaBuffer>(),
                extension_start: std::ptr::null_mut(),
                dtype,
                data: address as usize as *mut c_void,
                rank: dims.len() as i64,
                dims: dims.as_mut_ptr(),
            });
            (b, dims)
        };
        let abufs: Vec<_> = self.args.iter().map(|(d, a, s)| mk(*d, *a, s)).collect();
        let rbufs: Vec<_> = self.rets.iter().map(|(d, a, s)| mk(*d, *a, s)).collect();
        let mut atypes = vec![BUFFER_TYPE; abufs.len()];
        let mut rtypes = vec![BUFFER_TYPE; rbufs.len()];
        let mut aptrs: Vec<*mut c_void> = abufs.iter().map(|b| &*b.0 as *const XlaBuffer as *mut c_void).collect();
        let mut rptrs: Vec<*mut c_void> = rbufs.iter().map(|b| &*b.0 as *const XlaBuffer as *mut c_void).collect();

        // Attributes: names, spans, scalars and arrays kept alive here.
        let mut names: Vec<Box<XlaByteSpan>> = vec![];
        let mut types: Vec<i32> = vec![];
        let mut values: Vec<*mut c_void> = vec![];
        let mut keep: Vec<Box<dyn std::any::Any>> = vec![];
        for a in &self.attrs {
            let (name, ty, value): (&String, i32, *mut c_void) = match a {
                Attr::Str(n, s) => {
                    let span = Box::new(XlaByteSpan { ptr: s.as_ptr() as *const c_char, len: s.len() });
                    let p = &*span as *const XlaByteSpan as *mut c_void;
                    keep.push(span);
                    (n, attr_type::STRING, p)
                }
                Attr::I64(n, v) => {
                    let v = Box::new(*v);
                    let sc = Box::new(XlaScalar { dtype: data_type::S64, value: &*v as *const i64 as *mut c_void });
                    let p = &*sc as *const XlaScalar as *mut c_void;
                    keep.push(v);
                    keep.push(sc);
                    (n, attr_type::SCALAR, p)
                }
                Attr::F32(n, v) => {
                    let v = Box::new(*v);
                    let sc = Box::new(XlaScalar { dtype: data_type::F32, value: &*v as *const f32 as *mut c_void });
                    let p = &*sc as *const XlaScalar as *mut c_void;
                    keep.push(v);
                    keep.push(sc);
                    (n, attr_type::SCALAR, p)
                }
                Attr::Order(n, o) => {
                    let o = o.clone();
                    let arr =
                        Box::new(XlaArray { dtype: data_type::S64, size: o.len(), data: o.as_ptr() as *mut c_void });
                    let p = &*arr as *const XlaArray as *mut c_void;
                    keep.push(Box::new(o));
                    keep.push(arr);
                    (n, attr_type::ARRAY, p)
                }
            };
            names.push(Box::new(XlaByteSpan { ptr: name.as_ptr() as *const c_char, len: name.len() }));
            types.push(ty);
            values.push(value);
        }
        let mut name_ptrs: Vec<*mut XlaByteSpan> = names.iter_mut().map(|n| &mut **n as *mut XlaByteSpan).collect();

        let mut frame = XlaCallFrame {
            struct_size: std::mem::size_of::<XlaCallFrame>(),
            extension_start: std::ptr::null_mut(),
            api: &api,
            ctx: std::ptr::null_mut(),
            stage: self.stage,
            args: XlaArgs {
                struct_size: std::mem::size_of::<XlaArgs>(),
                extension_start: std::ptr::null_mut(),
                size: abufs.len() as i64,
                types: atypes.as_mut_ptr(),
                args: aptrs.as_mut_ptr(),
            },
            rets: XlaRets {
                struct_size: std::mem::size_of::<XlaRets>(),
                extension_start: std::ptr::null_mut(),
                size: rbufs.len() as i64,
                types: rtypes.as_mut_ptr(),
                rets: rptrs.as_mut_ptr(),
            },
            attrs: XlaAttrs {
                struct_size: std::mem::size_of::<XlaAttrs>(),
                extension_start: std::ptr::null_mut(),
                size: types.len() as i64,
                types: types.as_mut_ptr(),
                names: name_ptrs.as_mut_ptr(),
                attrs: values.as_mut_ptr(),
            },
            future: std::ptr::null_mut(),
        };
        // SAFETY: a valid frame over live buffers.
        let err = unsafe { handle_call_frame(&mut frame) };
        drop(keep);
        if err.is_null() {
            Ok(())
        } else {
            // SAFETY: made by `test_error_create`.
            let e = unsafe { Box::from_raw(err as *mut TestError) };
            Err((e.code, e.message))
        }
    }
}

fn build_add(spec: &XlaCallSpec, exported: bool) -> Result<XlaTarget, String> {
    // The kernel's row stride is a symbol shared by all three tensors: the buffers must be
    // compact row-major (the default layout).
    for t in spec.args.iter().chain(&spec.rets) {
        if t.dtype != DType::F32 {
            return Err(format!("float32 only, got {}", t.dtype.cute_name()));
        }
    }
    let tc = Toolchain::discover().ok_or("no mlir-opt")?;
    let f = if exported {
        let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
        let arch = ctx.best_target(&tc).ok_or("no target")?;
        let k = add_kernel();
        let compiled = k.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).map_err(|e| e.to_string())?;
        TvmFfiFunction::from_exported(k, cute::ExportedKernel::from_compiled(&compiled), add_config)
            .map_err(|e| e.to_string())?
    } else {
        TvmFfiFunction::compile(add_kernel(), &tc, &CompileOptions::new(), add_config).map_err(|e| e.to_string())?
    };
    Ok(XlaTarget::new(Box::new(f.with_arg_names(&["a", "b", "c"]))))
}

const M: usize = 37;
const N: usize = 64;

fn data() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..M * N).map(|i| (i % 31) as f32 - 15.0).collect();
    let b: Vec<f32> = (0..M * N).map(|i| (i % 7) as f32 * 0.25).collect();
    let want = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    (a, b, want)
}

#[test]
fn handler_runs_cached_aliased_and_diagnosed_calls() {
    let Some((ctx, _tc)) = device() else { return };
    let (a, b, want) = data();
    let (da, db) = (ctx.upload_slice(&a).unwrap(), ctx.upload_slice(&b).unwrap());
    let dc = ctx.alloc::<f32>(M * N).unwrap();
    let stream = ctx.stream().unwrap();
    STREAM.with(|s| s.set(stream.as_raw() as usize));
    let shape = [M as i64, N as i64];

    register_cutlass_call("xla_add", |s| build_add(s, false));
    register_cutlass_call("xla_add_exported", |s| build_add(s, true));
    register_cutlass_call("xla_add_inplace", |s| Ok(build_add(s, false)?.alias(0, 0)));
    assert!(is_cutlass_call_registered("xla_add"));
    let call = |t: &str| Frame::new(t).arg(da.as_raw(), &shape).arg(db.as_raw(), &shape).ret(dc.as_raw(), &shape);

    // Execute: the result matches; the first call misses the cache, the second hits.
    dc.memset(0).unwrap();
    call("xla_add").call().unwrap_or_else(|e| panic!("{e:?}"));
    stream.synchronize().unwrap();
    assert_eq!(dc.to_vec::<f32>().unwrap(), want);
    assert_eq!(call_cache_stats("xla_add"), CallCacheStats { hits: 0, misses: 1 });
    dc.memset(0).unwrap();
    call("xla_add").call().unwrap();
    stream.synchronize().unwrap();
    assert_eq!(dc.to_vec::<f32>().unwrap(), want);
    assert_eq!(call_cache_stats("xla_add"), CallCacheStats { hits: 1, misses: 1 });

    // Another shape is another entry (the kernel's symbols take the new extents).
    let (m2, n2) = (5i64, 16i64);
    let mut f =
        Frame::new("xla_add").arg(da.as_raw(), &[m2, n2]).arg(db.as_raw(), &[m2, n2]).ret(dc.as_raw(), &[m2, n2]);
    dc.memset(0).unwrap();
    f.call().unwrap_or_else(|e| panic!("{e:?}"));
    stream.synchronize().unwrap();
    assert_eq!(dc.to_vec::<f32>().unwrap()[..80], want[..80]);
    assert_eq!(call_cache_stats("xla_add").misses, 2);

    // An attribute is part of the key (`device` is read by the handler, not passed on).
    f.attrs.push(Attr::I64("device".into(), 0));
    f.call().unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(call_cache_stats("xla_add").misses, 3);
    f.call().unwrap();
    assert_eq!(call_cache_stats("xla_add").hits, 2);

    // The same call from an export (no mlir-opt at call time).
    dc.memset(0).unwrap();
    call("xla_add_exported").call().unwrap_or_else(|e| panic!("{e:?}"));
    stream.synchronize().unwrap();
    assert_eq!(dc.to_vec::<f32>().unwrap(), want);

    // Aliasing: the result is argument 0's memory, which the kernel updates in place.
    let inplace = ctx.upload_slice(&a).unwrap();
    Frame::new("xla_add_inplace")
        .arg(inplace.as_raw(), &shape)
        .arg(db.as_raw(), &shape)
        .ret(inplace.as_raw(), &shape)
        .call()
        .unwrap_or_else(|e| panic!("{e:?}"));
    stream.synchronize().unwrap();
    assert_eq!(inplace.to_vec::<f32>().unwrap(), want);
    // XLA giving the result its own buffer where an alias was declared is refused.
    let (code, msg) = call("xla_add_inplace").call().unwrap_err();
    assert_eq!(code, error_code::FAILED_PRECONDITION, "{msg}");
    assert!(msg.contains("must alias argument 0"), "{msg}");

    // The instantiate, prepare and initialize stages do nothing (not even look the target up).
    for st in [stage::INSTANTIATE, stage::PREPARE, stage::INITIALIZE] {
        let mut f = Frame::new("no_such_target");
        f.stage = st;
        f.call().unwrap();
    }

    // Diagnostics, each before any launch.
    dc.memset(0).unwrap();
    let (code, msg) = call("no_such_target").call().unwrap_err();
    assert_eq!((code, msg.as_str()), (error_code::NOT_FOUND, "no cutlass_call target `no_such_target` is registered"));
    let mut nameless = call("xla_add");
    nameless.attrs.clear();
    let (code, msg) = nameless.call().unwrap_err();
    assert_eq!(code, error_code::INVALID_ARGUMENT);
    assert!(msg.contains("cute_target"), "{msg}");
    // A wrong buffer count.
    let (code, msg) = Frame::new("xla_add").arg(da.as_raw(), &shape).ret(dc.as_raw(), &shape).call().unwrap_err();
    assert_eq!(code, error_code::INVALID_ARGUMENT);
    assert!(msg.contains("takes 3 arguments, got 2"), "{msg}");
    // A wrong dtype in the frame (float64): refused by the builder, naming the type.
    let mut f64_frame = call("xla_add");
    f64_frame.args[1].0 = data_type::F64;
    let (code, msg) = f64_frame.call().unwrap_err();
    assert_eq!(code, error_code::INTERNAL);
    assert!(msg.contains("float32 only, got Float64"), "{msg}");
    // An unmapped XLA type (complex64 is 15).
    let mut c64 = call("xla_add");
    c64.args[0].0 = 15;
    let (code, msg) = c64.call().unwrap_err();
    assert_eq!(code, error_code::INVALID_ARGUMENT);
    assert!(msg.contains("argument 0") && msg.contains("no cute-rs type"), "{msg}");
    // Shapes that disagree between the buffers (the kernel's shared symbols).
    let (code, msg) = Frame::new("xla_add")
        .arg(da.as_raw(), &shape)
        .arg(db.as_raw(), &[M as i64, N as i64 - 1])
        .ret(dc.as_raw(), &shape)
        .call()
        .unwrap_err();
    assert_eq!(code, error_code::INVALID_ARGUMENT);
    assert!(msg.contains("xla_add: "), "{msg}");
    // A column-major layout does not fit a kernel whose inner stride is 1.
    let mut col = call("xla_add");
    col.attrs.push(Attr::Order("arg_m2m_0".into(), vec![0, 1]));
    let (code, msg) = col.call().unwrap_err();
    assert_eq!(code, error_code::INVALID_ARGUMENT);
    assert!(msg.contains("argument 0"), "{msg}");
    // A layout that is not a permutation.
    let mut bad = call("xla_add");
    bad.attrs.push(Attr::Order("ret_m2m_0".into(), vec![0, 0]));
    let (code, msg) = bad.call().unwrap_err();
    assert_eq!(code, error_code::INVALID_ARGUMENT);
    assert!(msg.contains("not a permutation"), "{msg}");
    // Nothing above wrote the output, and releasing the cache forces a rebuild.
    assert!(dc.to_vec::<f32>().unwrap().iter().all(|&x| x == 0.0));
    let misses = call_cache_stats("xla_add").misses;
    release_compile_cache();
    call("xla_add").call().unwrap();
    assert_eq!(call_cache_stats("xla_add").misses, misses + 1);
    assert!(unregister_cutlass_call("xla_add"));
    assert!(call("xla_add").call().is_err());
}

/// A scalar attribute becomes a scalar kernel argument after the tensors.
#[test]
fn scalar_attributes_are_kernel_arguments() {
    let Some((ctx, _tc)) = device() else { return };
    let x: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let dx = ctx.upload_slice(&x).unwrap();
    let dy = ctx.alloc::<f32>(256).unwrap();
    STREAM.with(|s| s.set(0));
    register_cutlass_call("xla_scale", |_| {
        let tc = Toolchain::discover().ok_or("no mlir-opt")?;
        let f = TvmFfiFunction::compile(scale_kernel(), &tc, &CompileOptions::new(), scale_config)
            .map_err(|e| e.to_string())?;
        Ok(XlaTarget::new(Box::new(f.with_arg_names(&["x", "y", "alpha"]))))
    });
    let mut f = Frame::new("xla_scale").arg(dx.as_raw(), &[256]).ret(dy.as_raw(), &[256]);
    f.attrs.push(Attr::F32("alpha".into(), 2.5));
    dy.memset(0).unwrap();
    f.call().unwrap_or_else(|e| panic!("{e:?}"));
    ctx.synchronize().unwrap();
    assert_eq!(dy.to_vec::<f32>().unwrap(), x.iter().map(|v| v * 2.5).collect::<Vec<_>>());
    // A NaN, which the launch configuration refuses.
    let mut g = Frame::new("xla_scale").arg(dx.as_raw(), &[256]).ret(dy.as_raw(), &[256]);
    g.attrs.push(Attr::F32("alpha".into(), f32::NAN));
    let (code, msg) = g.call().unwrap_err();
    assert_eq!(code, error_code::INVALID_ARGUMENT);
    assert!(msg.contains("alpha is NaN"), "{msg}");
}
