//! Running kernels on an NVIDIA GPU through the CUDA driver API
//! (cargo feature `cuda`).
//!
//! The driver library (`libcuda.so.1`) is loaded at runtime with `dlopen`, so
//! building needs no CUDA installation; creating a [`CudaContext`] fails with
//! a descriptive error on machines without an NVIDIA driver. Kernels are
//! compiled to PTX with [`Kernel::compile_ptx`] and JIT-compiled by the
//! driver when loaded.
//!
//! ```no_run
//! # use cute::prelude::*;
//! # use cute::cuda::CudaContext;
//! # fn demo(kernel: &Kernel, mut args: Vec<KernelArg>) -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = CudaContext::new(0)?;
//! let toolchain = Toolchain::discover().ok_or("mlir-opt not found")?;
//! kernel.run_cuda(&ctx, &toolchain, LaunchConfig::new([4, 1, 1], [128, 1, 1]), &mut args)?;
//! # Ok(()) }
//! ```

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::time::Instant;

use crate::backend::{Arch, Toolchain};
use crate::builder::Kernel;
use crate::ir::Param;
use crate::sim::{KernelArg, LaunchConfig};
use crate::validate::MAX_STATIC_SHARED_BYTES;

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 81;
const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
const CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION: c_int = 4;

/// `CUlaunchAttribute`: an id and a 64-byte value union.
#[repr(C)]
#[derive(Clone, Copy)]
struct LaunchAttribute {
    id: c_int,
    pad: c_int,
    value: [u32; 16],
}

/// `CUlaunchConfig`.
#[repr(C)]
struct LaunchExConfig {
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared_mem_bytes: c_uint,
    stream: *mut c_void,
    attrs: *mut LaunchAttribute,
    num_attrs: c_uint,
}

/// What the driver reports about one launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaunchStats {
    /// Seconds from submitting the kernel until it finished, with no host or
    /// device copies included.
    pub kernel_seconds: f64,
}

#[derive(Debug)]
pub struct CudaError {
    pub message: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA: {}", self.message)
    }
}

impl std::error::Error for CudaError {}

fn err(message: impl Into<String>) -> CudaError {
    CudaError { message: message.into() }
}

#[cfg(unix)]
mod dl {
    use std::ffi::{c_char, c_int, c_void};

    pub const RTLD_NOW: c_int = 2;

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    unsafe extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        pub fn dlerror() -> *const c_char;
    }
}

/// Driver entry points, resolved once per context.
struct Api {
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    module_get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    #[allow(clippy::type_complexity)]
    launch_kernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    get_error_name: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    func_set_attribute: unsafe extern "C" fn(CuFunction, c_int, c_int) -> CuResult,
    /// `cuLaunchKernelEx`, on drivers new enough to have it.
    launch_kernel_ex:
        Option<unsafe extern "C" fn(*const LaunchExConfig, CuFunction, *mut *mut c_void, *mut *mut c_void) -> CuResult>,
}

#[cfg(unix)]
fn load_api() -> Result<Api, CudaError> {
    // SAFETY: dlopen with a valid NUL-terminated name.
    let mut handle = unsafe { dl::dlopen(c"libcuda.so.1".as_ptr(), dl::RTLD_NOW) };
    if handle.is_null() {
        // SAFETY: dlerror returns null or a NUL-terminated string.
        let why = unsafe { dl::dlerror() };
        let why =
            if why.is_null() { String::new() } else { unsafe { CStr::from_ptr(why) }.to_string_lossy().into_owned() };
        // SAFETY: as above.
        handle = unsafe { dl::dlopen(c"libcuda.so".as_ptr(), dl::RTLD_NOW) };
        if handle.is_null() {
            if cfg!(target_os = "macos") {
                return Err(err(
                    "NVIDIA's CUDA driver is not available on macOS; use Kernel::simulate, or run on Linux with an NVIDIA GPU",
                ));
            }
            return Err(err(format!("could not load the CUDA driver library: {why}")));
        }
    }
    let load = |names: &[&CStr]| -> Result<*mut c_void, CudaError> {
        names
            .iter()
            // SAFETY: handle is a live dlopen handle; the name is NUL-terminated.
            .map(|n| unsafe { dl::dlsym(handle, n.as_ptr()) })
            .find(|p| !p.is_null())
            .ok_or_else(|| err(format!("the CUDA driver does not export {:?}", names[0])))
    };
    /// Reinterpret a symbol address as the function pointer type `F`.
    fn func<F: Copy>(p: *mut c_void) -> F {
        assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<*mut c_void>());
        // SAFETY: `p` is the address of the driver function whose signature
        // the corresponding `Api` field declares.
        unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) }
    }
    Ok(Api {
        init: func(load(&[c"cuInit"])?),
        device_get: func(load(&[c"cuDeviceGet"])?),
        device_get_attribute: func(load(&[c"cuDeviceGetAttribute"])?),
        device_get_name: func(load(&[c"cuDeviceGetName"])?),
        primary_ctx_retain: func(load(&[c"cuDevicePrimaryCtxRetain"])?),
        primary_ctx_release: func(load(&[c"cuDevicePrimaryCtxRelease_v2", c"cuDevicePrimaryCtxRelease"])?),
        ctx_set_current: func(load(&[c"cuCtxSetCurrent"])?),
        ctx_synchronize: func(load(&[c"cuCtxSynchronize"])?),
        module_load_data: func(load(&[c"cuModuleLoadData"])?),
        module_unload: func(load(&[c"cuModuleUnload"])?),
        module_get_function: func(load(&[c"cuModuleGetFunction"])?),
        mem_alloc: func(load(&[c"cuMemAlloc_v2"])?),
        mem_free: func(load(&[c"cuMemFree_v2"])?),
        memcpy_htod: func(load(&[c"cuMemcpyHtoD_v2"])?),
        memcpy_dtoh: func(load(&[c"cuMemcpyDtoH_v2"])?),
        launch_kernel: func(load(&[c"cuLaunchKernel"])?),
        get_error_name: func(load(&[c"cuGetErrorName"])?),
        func_set_attribute: func(load(&[c"cuFuncSetAttribute"])?),
        launch_kernel_ex: load(&[c"cuLaunchKernelEx", c"cuLaunchKernelEx_ptsz"]).ok().map(func),
    })
}

#[cfg(not(unix))]
fn load_api() -> Result<Api, CudaError> {
    Err(err("the CUDA launcher currently supports Unix-like systems (Linux) only"))
}

/// A CUDA device's primary context.
pub struct CudaContext {
    api: Api,
    device: CuDevice,
    ctx: CuContext,
}

impl CudaContext {
    /// Initialise the driver and retain device `ordinal`'s primary context.
    pub fn new(ordinal: usize) -> Result<CudaContext, CudaError> {
        let api = load_api()?;
        let mut device = 0;
        let mut ctx = std::ptr::null_mut();
        // SAFETY: valid out-pointers into locals; calls follow the driver API.
        unsafe {
            check(&api, (api.init)(0), "cuInit")?;
            check(&api, (api.device_get)(&mut device, ordinal as c_int), "cuDeviceGet")?;
            check(&api, (api.primary_ctx_retain)(&mut ctx, device), "cuDevicePrimaryCtxRetain")?;
            check(&api, (api.ctx_set_current)(ctx), "cuCtxSetCurrent")?;
        }
        Ok(CudaContext { api, device, ctx })
    }

    pub fn name(&self) -> String {
        let mut buf = [0 as c_char; 256];
        // SAFETY: buf is writable for its length.
        unsafe { (self.api.device_get_name)(buf.as_mut_ptr(), buf.len() as c_int, self.device) };
        // SAFETY: the driver NUL-terminates the name within the buffer.
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    }

    /// `(major, minor)` compute capability.
    pub fn compute_capability(&self) -> (i32, i32) {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: valid out-pointers.
        unsafe {
            (self.api.device_get_attribute)(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.device);
            (self.api.device_get_attribute)(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.device);
        }
        (major, minor)
    }

    /// The best supported compilation target for this device, if it can run
    /// SM80 code at all.
    pub fn arch(&self) -> Option<Arch> {
        match self.compute_capability() {
            (8, 6..=7) => Some(Arch::Sm86),
            (8, 9) => Some(Arch::Sm89),
            (9.., _) => Some(Arch::Sm90),
            (8, _) => Some(Arch::Sm80),
            _ => None,
        }
    }

    /// Load PTX and look up entry point `name`.
    pub fn load_ptx(&self, ptx: &str, name: &str) -> Result<CudaFunction<'_>, CudaError> {
        let image = CString::new(ptx).map_err(|_| err("PTX contains a NUL byte"))?;
        let entry = CString::new(name).map_err(|_| err("kernel name contains a NUL byte"))?;
        let mut module = std::ptr::null_mut();
        let mut function = std::ptr::null_mut();
        // SAFETY: image is NUL-terminated PTX; out-pointers are valid.
        unsafe {
            self.make_current()?;
            check(&self.api, (self.api.module_load_data)(&mut module, image.as_ptr().cast()), "cuModuleLoadData")?;
            let r = (self.api.module_get_function)(&mut function, module, entry.as_ptr());
            if let Err(e) = check(&self.api, r, "cuModuleGetFunction") {
                (self.api.module_unload)(module);
                return Err(e);
            }
        }
        Ok(CudaFunction { ctx: self, module, function })
    }

    fn make_current(&self) -> Result<(), CudaError> {
        // SAFETY: ctx is the retained primary context.
        check(&self.api, unsafe { (self.api.ctx_set_current)(self.ctx) }, "cuCtxSetCurrent")
    }

    /// Copy `arg` (a buffer) to the device.
    pub fn upload(&self, arg: &KernelArg) -> Result<DeviceBuffer<'_>, CudaError> {
        self.make_current()?;
        let (ptr, bytes) = host_bytes(arg);
        let mut device: CuDevicePtr = 0;
        // SAFETY: allocates `bytes` bytes, then copies from live host memory.
        unsafe {
            check(&self.api, (self.api.mem_alloc)(&mut device, bytes.max(1)), "cuMemAlloc")?;
            let buffer = DeviceBuffer { ctx: self, ptr: device, bytes };
            check(&self.api, (self.api.memcpy_htod)(device, ptr, bytes), "cuMemcpyHtoD")?;
            Ok(buffer)
        }
    }

    /// Copy `buffer` from the device into `arg`, which must be the same size.
    pub fn download(&self, buffer: &DeviceBuffer<'_>, arg: &mut KernelArg) -> Result<(), CudaError> {
        self.make_current()?;
        let (ptr, bytes) = host_bytes_mut(arg);
        if bytes != buffer.bytes {
            return Err(err(format!("downloading {} bytes into {bytes}", buffer.bytes)));
        }
        // SAFETY: both ranges are `bytes` long.
        check(&self.api, unsafe { (self.api.memcpy_dtoh)(ptr, buffer.ptr, bytes) }, "cuMemcpyDtoH")
    }

    /// Wait for everything launched so far.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.make_current()?;
        // SAFETY: the context is current.
        check(&self.api, unsafe { (self.api.ctx_synchronize)() }, "kernel execution")
    }

    fn submit(
        &self,
        kernel: &Kernel,
        function: &CudaFunction<'_>,
        cfg: LaunchConfig,
        args: &[DeviceArg<'_>],
    ) -> Result<(), CudaError> {
        let ir = kernel.ir();
        if args.len() != ir.args.len() {
            return Err(err(format!(
                "kernel `{}` takes {} arguments, {} were given",
                ir.name,
                ir.args.len(),
                args.len()
            )));
        }
        self.make_current()?;
        let api = &self.api;
        // Storage for every parameter value; the launch takes pointers to it.
        let mut storage: Vec<u64> = Vec::with_capacity(args.len());
        for (i, (param, arg)) in ir.args.iter().zip(args).enumerate() {
            storage.push(match (param, arg) {
                (Param::Buffer(_), DeviceArg::Buffer(b)) => b.ptr,
                (Param::Scalar { .. }, DeviceArg::Scalar(s)) => match s {
                    KernelArg::ScalarF32(x) => x.to_bits() as u64,
                    KernelArg::ScalarI32(x) => *x as u32 as u64,
                    KernelArg::ScalarF16(x) => x.0 as u64,
                    KernelArg::ScalarBool(x) => *x as u64,
                    KernelArg::Scalar { bits, .. } => *bits,
                    other => return Err(err(format!("argument {i} is {}, not a scalar", other.describe()))),
                },
                _ => return Err(err(format!("argument {i} does not match the kernel's parameter"))),
            });
        }
        // Little-endian: a pointer to the u64 slot is also a valid pointer to
        // a narrower scalar stored in its low bytes.
        let mut params: Vec<*mut c_void> = storage.iter_mut().map(|s| (s as *mut u64).cast()).collect();

        let smem = ir.shared_memory_bytes();
        let dynamic = if smem > MAX_STATIC_SHARED_BYTES { smem } else { 0 };
        if dynamic > 0 {
            let available = self.max_shared_memory_per_block();
            if dynamic > available {
                return Err(err(format!(
                    "the kernel needs {dynamic} bytes of shared memory; the device allows {available}"
                )));
            }
            // SAFETY: the function handle is live and the size is in range.
            let r = unsafe {
                (api.func_set_attribute)(
                    function.function,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dynamic as c_int,
                )
            };
            check(api, r, "cuFuncSetAttribute")?;
        }

        let [gx, gy, gz] = cfg.grid;
        let [bx, by, bz] = cfg.block;
        let clustered = cfg.cluster != [1, 1, 1];
        match api.launch_kernel_ex {
            Some(launch_ex) => {
                let mut attr = LaunchAttribute { id: CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, pad: 0, value: [0; 16] };
                attr.value[..3].copy_from_slice(&cfg.cluster);
                let config = LaunchExConfig {
                    grid: cfg.grid,
                    block: cfg.block,
                    shared_mem_bytes: dynamic as c_uint,
                    stream: std::ptr::null_mut(),
                    attrs: if clustered { &mut attr } else { std::ptr::null_mut() },
                    num_attrs: clustered as c_uint,
                };
                // SAFETY: `config` and `params` outlive the call, which
                // copies the argument values.
                let r = unsafe { launch_ex(&config, function.function, params.as_mut_ptr(), std::ptr::null_mut()) };
                check(api, r, "cuLaunchKernelEx")
            }
            None if clustered => Err(err("this driver has no cuLaunchKernelEx, which block clusters need")),
            None => {
                // SAFETY: params point at live storage matching the signature.
                let r = unsafe {
                    (api.launch_kernel)(
                        function.function,
                        gx,
                        gy,
                        gz,
                        bx,
                        by,
                        bz,
                        dynamic as c_uint,
                        std::ptr::null_mut(),
                        params.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                };
                check(api, r, "cuLaunchKernel")
            }
        }
    }

    /// Launch `function` on buffers already on the device and wait for it.
    pub fn launch_on_device(
        &self,
        kernel: &Kernel,
        function: &CudaFunction<'_>,
        cfg: LaunchConfig,
        args: &[DeviceArg<'_>],
    ) -> Result<LaunchStats, CudaError> {
        let start = Instant::now();
        self.submit(kernel, function, cfg, args)?;
        self.synchronize()?;
        Ok(LaunchStats { kernel_seconds: start.elapsed().as_secs_f64() })
    }

    /// Launch `function` without waiting. Kernels run in submission order;
    /// [`synchronize`](Self::synchronize) waits for them. The buffers must
    /// stay alive until then.
    pub fn launch_async(
        &self,
        kernel: &Kernel,
        function: &CudaFunction<'_>,
        cfg: LaunchConfig,
        args: &[DeviceArg<'_>],
    ) -> Result<(), CudaError> {
        self.submit(kernel, function, cfg, args)
    }

    /// Copy `args` to the device, launch `function` (compiled from `kernel`),
    /// wait for it, and copy every buffer back into `args`.
    pub fn launch(
        &self,
        kernel: &Kernel,
        function: &CudaFunction<'_>,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
    ) -> Result<LaunchStats, CudaError> {
        crate::sim::check_launch(kernel.ir(), cfg, args).map_err(err)?;
        let mut uploaded = Vec::with_capacity(args.len());
        for (param, arg) in kernel.ir().args.iter().zip(args.iter()) {
            uploaded.push(match param {
                Param::Buffer(_) => Some(self.upload(arg)?),
                Param::Scalar { .. } => None,
            });
        }
        let device_args: Vec<DeviceArg<'_>> = uploaded
            .iter()
            .zip(args.iter())
            .map(|(b, a)| match b {
                Some(b) => DeviceArg::Buffer(b),
                None => DeviceArg::Scalar(a.clone()),
            })
            .collect();
        let stats = self.launch_on_device(kernel, function, cfg, &device_args)?;
        for (b, a) in uploaded.iter().zip(args.iter_mut()) {
            if let Some(b) = b {
                self.download(b, a)?;
            }
        }
        Ok(stats)
    }

    /// The most shared memory this device gives one block (opting in).
    pub fn max_shared_memory_per_block(&self) -> usize {
        let mut bytes = 0;
        // SAFETY: a valid out-pointer.
        unsafe {
            (self.api.device_get_attribute)(
                &mut bytes,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                self.device,
            )
        };
        bytes as usize
    }
}

/// A buffer in device memory, freed when dropped.
pub struct DeviceBuffer<'a> {
    ctx: &'a CudaContext,
    ptr: CuDevicePtr,
    bytes: usize,
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: ptr was returned by cuMemAlloc.
        unsafe { (self.ctx.api.mem_free)(self.ptr) };
    }
}

/// An argument to a launch on device memory.
pub enum DeviceArg<'a> {
    Buffer(&'a DeviceBuffer<'a>),
    /// One of the scalar [`KernelArg`] variants.
    Scalar(KernelArg),
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // SAFETY: releases the primary context retained in `new`.
        unsafe { (self.api.primary_ctx_release)(self.device) };
    }
}

/// A loaded kernel entry point.
pub struct CudaFunction<'a> {
    ctx: &'a CudaContext,
    module: CuModule,
    function: CuFunction,
}

impl Drop for CudaFunction<'_> {
    fn drop(&mut self) {
        // SAFETY: module was loaded by cuModuleLoadData.
        unsafe { (self.ctx.api.module_unload)(self.module) };
    }
}

fn check(api: &Api, r: CuResult, what: &str) -> Result<(), CudaError> {
    if r == 0 {
        return Ok(());
    }
    let mut name: *const c_char = std::ptr::null();
    // SAFETY: cuGetErrorName writes a static string pointer.
    unsafe { (api.get_error_name)(r, &mut name) };
    let name = if name.is_null() {
        format!("error {r}")
    } else {
        // SAFETY: static NUL-terminated string owned by the driver.
        unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
    };
    Err(err(format!("{what} failed: {name}")))
}

fn host_bytes(arg: &KernelArg) -> (*const c_void, usize) {
    match arg {
        KernelArg::F32(v) => (v.as_ptr().cast(), v.len() * 4),
        KernelArg::F16(v) => (v.as_ptr().cast(), v.len() * 2),
        KernelArg::I32(v) => (v.as_ptr().cast(), v.len() * 4),
        KernelArg::Bool(v) => (v.as_ptr().cast(), v.len()),
        KernelArg::Bytes { data, .. } => (data.as_ptr().cast(), data.len()),
        _ => unreachable!("not a buffer"),
    }
}

fn host_bytes_mut(arg: &mut KernelArg) -> (*mut c_void, usize) {
    match arg {
        KernelArg::F32(v) => (v.as_mut_ptr().cast(), v.len() * 4),
        KernelArg::F16(v) => (v.as_mut_ptr().cast(), v.len() * 2),
        KernelArg::I32(v) => (v.as_mut_ptr().cast(), v.len() * 4),
        KernelArg::Bool(v) => (v.as_mut_ptr().cast(), v.len()),
        KernelArg::Bytes { data, .. } => (data.as_mut_ptr().cast(), data.len()),
        _ => unreachable!("not a buffer"),
    }
}

impl Kernel {
    /// Compile for `ctx`'s device with `toolchain`, run, and copy results
    /// back into `args` (the same argument convention as
    /// [`Kernel::simulate`]).
    pub fn run_cuda(
        &self,
        ctx: &CudaContext,
        toolchain: &Toolchain,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
    ) -> Result<LaunchStats, Box<dyn std::error::Error>> {
        let arch = ctx.arch().ok_or_else(|| {
            err(format!(
                "{} (sm_{}{}) is older than SM80",
                ctx.name(),
                ctx.compute_capability().0,
                ctx.compute_capability().1
            ))
        })?;
        let ptx = self.compile_ptx(toolchain, arch)?;
        let function = ctx.load_ptx(&ptx, self.name())?;
        Ok(ctx.launch(self, &function, cfg, args)?)
    }
}
