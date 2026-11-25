//! Running kernels on an NVIDIA GPU through the CUDA driver API
//! (cargo feature `cuda`).
//!
//! The driver library (`libcuda.so.1`) is loaded at runtime with `dlopen`, so
//! building needs no CUDA installation; creating a [`CudaContext`] fails with
//! a descriptive error on machines without an NVIDIA driver. Kernels are
//! compiled to PTX with [`Kernel::compile_with`] and JIT-compiled by the
//! driver when loaded ([`CudaContext::load`]), with the JIT options the
//! compile options ask for.
//!
//! ```no_run
//! # use cute::prelude::*;
//! # use cute::cuda::CudaContext;
//! # fn demo(kernel: &Kernel, mut args: Vec<KernelArg>) -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = CudaContext::new(0)?;
//! let toolchain = Toolchain::discover().ok_or("mlir-opt not found")?;
//! let cfg = LaunchConfig::new([4, 1, 1], [128, 1, 1]);
//! kernel.run_cuda(&ctx, &toolchain, cfg, &mut args)?;
//!
//! // Or with explicit options, for Nsight Compute: line information, no optimization.
//! let options = CompileOptions::new().generate_line_info(true).opt_level(0);
//! kernel.run_cuda_with(&ctx, &toolchain, &options, cfg, &mut args)?;
//! # Ok(()) }
//! ```

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::time::Instant;

use crate::backend::{Arch, Artifact, CompileOptions, CompiledKernel, JitOption, Toolchain};
use crate::builder::Kernel;
use crate::external::PointerSpec;
use crate::ir::Param;
use crate::log::Level;
use crate::signature::KernelSignature;
use crate::sim::{KernelArg, LaunchConfig};

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;
type CuStream = *mut c_void;
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;
type CuEvent = *mut c_void;

mod device;
mod event;
mod graph;
mod hardware_info;
mod launch;
mod memory;
mod module;
mod stream;
// Several kernels in one module (m6b-multi-kernel).
mod kernel_module;
pub use kernel_module::CudaModule;
mod tensor_map_encode;
// m11-extern-distributed: load-time linking and multicast memory.
mod distributed;
pub use distributed::MulticastGroup;

/// Host-built tensor maps (CuTe DSL's `cutlass.experimental.cuda` tensor-map helpers),
/// which [`CudaContext::encode_tensor_map`] encodes with the driver.
pub use crate::tensor_map;
pub use device::{DeviceAttribute, DeviceInfo, FuncAttribute};
pub use event::{CudaEvent, EventFlags};
pub use graph::{CaptureGuard, CaptureMode, CudaGraph, CudaGraphExec};
pub use hardware_info::HardwareInfo;
pub use launch::LaunchBuilder;
pub use memory::{DeviceRepr, Pending, PinnedBuffer};
use module::LoadedModule;
pub use module::{UnloadCallbackToken, register_library_unload_callback, unregister_library_unload_callback};
pub use stream::CudaStream;
pub use tensor_map_encode::{CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED, EncodedTensorMap};

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
const CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION: c_int = 4;
const CU_JIT_INFO_LOG_BUFFER: c_int = 3;
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
const CU_JIT_INPUT_PTX: c_int = 1;
/// Size of the buffers the JIT and the linker write their logs to.
const JIT_LOG_BYTES: usize = 16 * 1024;

/// The image a module is loaded from.
#[derive(Clone, Copy, Debug)]
pub enum ModuleImage<'a> {
    /// PTX text, which the driver JIT-compiles (a NUL is appended for the driver).
    Ptx(&'a str),
    /// A cubin (an ELF image), passed as bytes.
    Cubin(&'a [u8]),
    /// A fatbin, passed as bytes.
    Fatbin(&'a [u8]),
}

type LinkState = *mut c_void;

/// A context's loaded modules with their JIT info logs, by a hash of image and options.
type ModuleCache = crate::backend::cache::Lru<[u8; 32], (std::rc::Rc<LoadedModule>, String)>;

/// `CUlaunchAttribute`: an id, padding to 8 bytes and a 64-byte value union, 8-byte
/// aligned as the union's pointer members make it.
#[repr(C, align(8))]
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
    /// device copies included. It is a host timer around the launch and the
    /// synchronisation after it, so it includes the launch latency and the wait: for the
    /// device time of a kernel, use [`testing::benchmark`](crate::testing::benchmark),
    /// which times many launches with CUDA events.
    pub kernel_seconds: f64,
}

#[derive(Debug)]
pub struct CudaError {
    pub message: String,
    /// The driver's `CUresult`, for an error the driver returned.
    pub code: Option<i32>,
}

/// `CUDA_ERROR_ASSERT`: a device assertion failed. The context is unusable
/// afterwards; [`CudaContext::reset`] makes a fresh one.
pub const CUDA_ERROR_ASSERT: i32 = 710;

impl CudaError {
    /// Whether a device assertion failed (`CUDA_ERROR_ASSERT`).
    pub fn is_device_assert(&self) -> bool {
        self.code == Some(CUDA_ERROR_ASSERT)
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA: {}", self.message)
    }
}

impl std::error::Error for CudaError {}

fn err(message: impl Into<String>) -> CudaError {
    CudaError { message: message.into(), code: None }
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
    ctx_get_device: unsafe extern "C" fn(*mut CuDevice) -> CuResult,
    ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    module_load_data_ex:
        unsafe extern "C" fn(*mut CuModule, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> CuResult,
    link_create: unsafe extern "C" fn(c_uint, *mut c_int, *mut *mut c_void, *mut LinkState) -> CuResult,
    #[allow(clippy::type_complexity)]
    link_add_data: unsafe extern "C" fn(
        LinkState,
        c_int,
        *mut c_void,
        usize,
        *const c_char,
        c_uint,
        *mut c_int,
        *mut *mut c_void,
    ) -> CuResult,
    link_complete: unsafe extern "C" fn(LinkState, *mut *mut c_void, *mut usize) -> CuResult,
    link_destroy: unsafe extern "C" fn(LinkState) -> CuResult,
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
    get_error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    primary_ctx_reset: unsafe extern "C" fn(CuDevice) -> CuResult,
    func_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuFunction) -> CuResult,
    driver_get_version: unsafe extern "C" fn(*mut c_int) -> CuResult,
    func_set_attribute: unsafe extern "C" fn(CuFunction, c_int, c_int) -> CuResult,
    /// `cuLaunchKernelEx`, on drivers new enough to have it.
    launch_kernel_ex:
        Option<unsafe extern "C" fn(*const LaunchExConfig, CuFunction, *mut *mut c_void, *mut *mut c_void) -> CuResult>,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    device_total_mem: unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult,
    stream_create: unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult,
    stream_synchronize: unsafe extern "C" fn(CuStream) -> CuResult,
    stream_destroy: unsafe extern "C" fn(CuStream) -> CuResult,
    stream_query: unsafe extern "C" fn(CuStream) -> CuResult,
    stream_is_capturing: unsafe extern "C" fn(CuStream, *mut c_int) -> CuResult,
    memcpy_htod_async: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize, CuStream) -> CuResult,
    memcpy_dtoh_async: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult,
    memcpy_dtod: unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize) -> CuResult,
    memcpy_dtod_async: unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize, CuStream) -> CuResult,
    mem_alloc_host: unsafe extern "C" fn(*mut *mut c_void, usize) -> CuResult,
    mem_free_host: unsafe extern "C" fn(*mut c_void) -> CuResult,
    memset_d8: unsafe extern "C" fn(CuDevicePtr, u8, usize) -> CuResult,
    memset_d8_async: unsafe extern "C" fn(CuDevicePtr, u8, usize, CuStream) -> CuResult,
    module_get_global: unsafe extern "C" fn(*mut CuDevicePtr, *mut usize, CuModule, *const c_char) -> CuResult,
    /// The stream-ordered allocator (CUDA 11.2).
    mem_alloc_async: Option<unsafe extern "C" fn(*mut CuDevicePtr, usize, CuStream) -> CuResult>,
    mem_free_async: Option<unsafe extern "C" fn(CuDevicePtr, CuStream) -> CuResult>,
    /// Stream capture and graphs (CUDA 10 and 11.4 for the flags form).
    stream_begin_capture: Option<unsafe extern "C" fn(CuStream, c_int) -> CuResult>,
    stream_end_capture: Option<unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult>,
    graph_instantiate: Option<unsafe extern "C" fn(*mut CuGraphExec, CuGraph, u64) -> CuResult>,
    graph_launch: Option<unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult>,
    graph_destroy: Option<unsafe extern "C" fn(CuGraph) -> CuResult>,
    graph_exec_destroy: Option<unsafe extern "C" fn(CuGraphExec) -> CuResult>,
    graph_get_nodes: Option<unsafe extern "C" fn(CuGraph, *mut *mut c_void, *mut usize) -> CuResult>,
    event_create: unsafe extern "C" fn(*mut CuEvent, c_uint) -> CuResult,
    event_record: unsafe extern "C" fn(CuEvent, CuStream) -> CuResult,
    event_query: unsafe extern "C" fn(CuEvent) -> CuResult,
    event_synchronize: unsafe extern "C" fn(CuEvent) -> CuResult,
    event_elapsed_time: unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
    event_destroy: unsafe extern "C" fn(CuEvent) -> CuResult,
    stream_wait_event: unsafe extern "C" fn(CuStream, CuEvent, c_uint) -> CuResult,
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)`.
    occupancy_max_active_blocks: unsafe extern "C" fn(*mut c_int, CuFunction, c_int, usize) -> CuResult,
    /// `cuTensorMapEncodeTiled` and `cuTensorMapEncodeIm2col` (CUDA 12.0), on drivers that
    /// have them.
    tensor_map_encode_tiled: Option<tensor_map_encode::EncodeTiled>,
    tensor_map_encode_im2col: Option<tensor_map_encode::EncodeIm2col>,
    /// `cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)`
    /// (CUDA 11.0) and `cuOccupancyMaxActiveClusters(numClusters, func, config)` (CUDA 11.8),
    /// for [`HardwareInfo`], on drivers that have them.
    occupancy_available_dynamic_smem: Option<unsafe extern "C" fn(*mut usize, CuFunction, c_int, c_int) -> CuResult>,
    occupancy_max_active_clusters:
        Option<unsafe extern "C" fn(*mut c_int, CuFunction, *const LaunchExConfig) -> CuResult>,
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
        ctx_get_device: func(load(&[c"cuCtxGetDevice"])?),
        ctx_synchronize: func(load(&[c"cuCtxSynchronize"])?),
        module_load_data_ex: func(load(&[c"cuModuleLoadDataEx"])?),
        link_create: func(load(&[c"cuLinkCreate_v2", c"cuLinkCreate"])?),
        link_add_data: func(load(&[c"cuLinkAddData_v2", c"cuLinkAddData"])?),
        link_complete: func(load(&[c"cuLinkComplete"])?),
        link_destroy: func(load(&[c"cuLinkDestroy"])?),
        module_unload: func(load(&[c"cuModuleUnload"])?),
        module_get_function: func(load(&[c"cuModuleGetFunction"])?),
        mem_alloc: func(load(&[c"cuMemAlloc_v2"])?),
        mem_free: func(load(&[c"cuMemFree_v2"])?),
        memcpy_htod: func(load(&[c"cuMemcpyHtoD_v2"])?),
        memcpy_dtoh: func(load(&[c"cuMemcpyDtoH_v2"])?),
        launch_kernel: func(load(&[c"cuLaunchKernel"])?),
        get_error_name: func(load(&[c"cuGetErrorName"])?),
        get_error_string: func(load(&[c"cuGetErrorString"])?),
        primary_ctx_reset: func(load(&[c"cuDevicePrimaryCtxReset_v2", c"cuDevicePrimaryCtxReset"])?),
        func_get_attribute: func(load(&[c"cuFuncGetAttribute"])?),
        driver_get_version: func(load(&[c"cuDriverGetVersion"])?),
        func_set_attribute: func(load(&[c"cuFuncSetAttribute"])?),
        launch_kernel_ex: load(&[c"cuLaunchKernelEx", c"cuLaunchKernelEx_ptsz"]).ok().map(func),
        device_get_count: func(load(&[c"cuDeviceGetCount"])?),
        device_total_mem: func(load(&[c"cuDeviceTotalMem_v2"])?),
        stream_create: func(load(&[c"cuStreamCreate"])?),
        stream_synchronize: func(load(&[c"cuStreamSynchronize"])?),
        stream_destroy: func(load(&[c"cuStreamDestroy_v2"])?),
        stream_query: func(load(&[c"cuStreamQuery"])?),
        stream_is_capturing: func(load(&[c"cuStreamIsCapturing"])?),
        memcpy_htod_async: func(load(&[c"cuMemcpyHtoDAsync_v2"])?),
        memcpy_dtoh_async: func(load(&[c"cuMemcpyDtoHAsync_v2"])?),
        memcpy_dtod: func(load(&[c"cuMemcpyDtoD_v2"])?),
        memcpy_dtod_async: func(load(&[c"cuMemcpyDtoDAsync_v2"])?),
        mem_alloc_host: func(load(&[c"cuMemAllocHost_v2"])?),
        mem_free_host: func(load(&[c"cuMemFreeHost"])?),
        memset_d8: func(load(&[c"cuMemsetD8_v2"])?),
        memset_d8_async: func(load(&[c"cuMemsetD8Async"])?),
        module_get_global: func(load(&[c"cuModuleGetGlobal_v2"])?),
        mem_alloc_async: load(&[c"cuMemAllocAsync"]).ok().map(func),
        mem_free_async: load(&[c"cuMemFreeAsync"]).ok().map(func),
        stream_begin_capture: load(&[c"cuStreamBeginCapture_v2"]).ok().map(func),
        stream_end_capture: load(&[c"cuStreamEndCapture"]).ok().map(func),
        graph_instantiate: load(&[c"cuGraphInstantiateWithFlags"]).ok().map(func),
        graph_launch: load(&[c"cuGraphLaunch"]).ok().map(func),
        graph_destroy: load(&[c"cuGraphDestroy"]).ok().map(func),
        graph_exec_destroy: load(&[c"cuGraphExecDestroy"]).ok().map(func),
        graph_get_nodes: load(&[c"cuGraphGetNodes"]).ok().map(func),
        event_create: func(load(&[c"cuEventCreate"])?),
        event_record: func(load(&[c"cuEventRecord"])?),
        event_query: func(load(&[c"cuEventQuery"])?),
        event_synchronize: func(load(&[c"cuEventSynchronize"])?),
        event_elapsed_time: func(load(&[c"cuEventElapsedTime_v2", c"cuEventElapsedTime"])?),
        event_destroy: func(load(&[c"cuEventDestroy_v2", c"cuEventDestroy"])?),
        stream_wait_event: func(load(&[c"cuStreamWaitEvent"])?),
        occupancy_max_active_blocks: func(load(&[c"cuOccupancyMaxActiveBlocksPerMultiprocessor"])?),
        tensor_map_encode_tiled: load(&[c"cuTensorMapEncodeTiled"]).ok().map(func),
        tensor_map_encode_im2col: load(&[c"cuTensorMapEncodeIm2col"]).ok().map(func),
        occupancy_available_dynamic_smem: load(&[c"cuOccupancyAvailableDynamicSMemPerBlock"]).ok().map(func),
        occupancy_max_active_clusters: load(&[c"cuOccupancyMaxActiveClusters"]).ok().map(func),
    })
}

#[cfg(not(unix))]
fn load_api() -> Result<Api, CudaError> {
    Err(err("the CUDA launcher currently supports Unix-like systems (Linux) only"))
}

/// A CUDA device's primary context.
pub struct CudaContext {
    /// The loaded modules, by a hash of their image and JIT options (dropped before the
    /// context is released).
    modules: std::cell::RefCell<ModuleCache>,
    /// `cuModuleLoadDataEx` calls so far.
    loads: std::cell::Cell<u64>,
    api: Api,
    ordinal: usize,
    device: CuDevice,
    ctx: CuContext,
    /// The primary context was released by a failed [`reset`](Self::reset).
    released: bool,
}

/// The ordinal of the device the calling thread's current CUDA context belongs to (what a
/// framework that called into the library has made current), or an error if the thread has
/// no current context.
pub fn current_device_ordinal() -> Result<i32, CudaError> {
    let api = load_api()?;
    let mut device = 0;
    // SAFETY: a valid out-pointer into a local.
    check(&api, unsafe { (api.ctx_get_device)(&mut device) }, "cuCtxGetDevice")?;
    Ok(device)
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
        Ok(CudaContext {
            modules: std::cell::RefCell::new(crate::backend::cache::Lru::new()),
            loads: std::cell::Cell::new(0),
            api,
            ordinal,
            device,
            ctx,
            released: false,
        })
    }

    /// The device's name (`cuDeviceGetName`), or `device N` if the driver cannot say.
    pub fn name(&self) -> String {
        let mut buf = [0 as c_char; 256];
        // SAFETY: buf is writable for its length.
        let r = unsafe { (self.api.device_get_name)(buf.as_mut_ptr(), buf.len() as c_int, self.device) };
        if r != 0 {
            return format!("device {}", self.ordinal);
        }
        // SAFETY: the driver NUL-terminates the name within the buffer.
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    }

    /// `(major, minor)` compute capability; `(0, 0)` if the driver cannot say (which no
    /// target runs on).
    pub fn compute_capability(&self) -> (i32, i32) {
        let major = self.raw_device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        let minor = self.raw_device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        match (major, minor) {
            (Ok(major), Ok(minor)) => (major, minor),
            _ => (0, 0),
        }
    }

    /// The device's own target ([`Arch::from_compute_capability`]): the
    /// architecture-specific one from Hopper on (`sm_90a` for CC 9.0,
    /// `sm_103a` for 10.3), the baseline before (`sm_89` for 8.9), or `None`
    /// for a device older than SM80 or one cute-rs does not know.
    pub fn arch(&self) -> Option<Arch> {
        let (major, minor) = self.compute_capability();
        Arch::from_compute_capability(u32::try_from(major).ok()?, u32::try_from(minor).ok()?)
    }

    /// The CUDA driver's version (`cuDriverGetVersion`): `1000 * major + 10
    /// * minor`, 12080 for CUDA 12.8.
    pub fn driver_version(&self) -> Result<u32, CudaError> {
        let mut v: c_int = 0;
        // SAFETY: a valid out-pointer.
        check(&self.api, unsafe { (self.api.driver_get_version)(&mut v) }, "cuDriverGetVersion")?;
        Ok(v as u32)
    }

    /// The newest PTX ISA version the driver's JIT accepts
    /// ([`driver_ptx_ceiling`](crate::driver_ptx_ceiling)).
    pub fn driver_ptx(&self) -> Result<u32, CudaError> {
        Ok(crate::driver_ptx_ceiling(self.driver_version()?))
    }

    /// The most specific target this device runs that `toolchain` emits and
    /// the driver loads ([`best_target`](crate::best_target)): the device's
    /// `a` target, falling back to a family target and then to a baseline
    /// when the toolchain or the driver lacks the newer PTX.
    pub fn best_target(&self, toolchain: &Toolchain) -> Option<Arch> {
        let (major, minor) = self.compute_capability();
        let cc = (u32::try_from(major).ok()?, u32::try_from(minor).ok()?);
        let driver = self.driver_ptx().ok();
        match toolchain.targets() {
            Some(support) => crate::best_target(cc, driver, support),
            None => crate::best_target(cc, driver, &crate::TargetSupport::for_llvm(u32::MAX)),
        }
    }

    /// Load PTX and look up entry point `name`, with the driver JIT's default options.
    pub fn load_ptx(&self, ptx: &str, name: &str) -> Result<CudaFunction<'_>, CudaError> {
        self.load_module(ModuleImage::Ptx(ptx), &[], name)
    }

    /// Load a module from `image` with `cuModuleLoadDataEx` and look up entry point `entry`.
    /// `options` are the JIT's (PTX only; the driver has nothing to compile in a cubin). The
    /// JIT's error log is always attached, so a failure reports the assembler's message and not
    /// a bare `CUDA_ERROR_INVALID_PTX`; the info log (`-v`'s resource usage with
    /// [`JitOption::LogVerbose`]) is on the returned function ([`CudaFunction::jit_log`]).
    ///
    /// Modules are cached per context by a hash of the image and the JIT options (bounded by
    /// `CUTE_RS_JIT_CACHE_MAX_ELEMS`; `CUTE_RS_NO_CACHE` turns it off): loading the same
    /// image again shares the loaded module, which is unloaded when its last function and its
    /// cache entry are gone.
    pub fn load_module(
        &self,
        image: ModuleImage<'_>,
        options: &[JitOption],
        entry: &str,
    ) -> Result<CudaFunction<'_>, CudaError> {
        self.load_module_with(image, options, entry, !crate::backend::cache::no_cache_env())
    }

    /// `cuModuleLoadDataEx` calls this context has made (cache hits make none).
    pub fn modules_loaded(&self) -> u64 {
        self.loads.get()
    }

    /// The modules in this context's module cache.
    pub fn module_cache_len(&self) -> usize {
        self.modules.borrow().len()
    }

    /// Drop this context's module cache (a module is unloaded once no function holds it).
    pub fn clear_module_cache(&self) {
        let drained = self.modules.borrow_mut().drain();
        drop(drained);
    }

    fn load_module_with(
        &self,
        image: ModuleImage<'_>,
        options: &[JitOption],
        entry: &str,
        use_cache: bool,
    ) -> Result<CudaFunction<'_>, CudaError> {
        let (module, info) = self.load_image(image, options, entry, use_cache)?;
        self.function(module, entry, info)
    }

    /// Load (or find in the module cache) the module of `image`, for `entry`: the module and
    /// the JIT's info log.
    fn load_image(
        &self,
        image: ModuleImage<'_>,
        options: &[JitOption],
        entry: &str,
        use_cache: bool,
    ) -> Result<(std::rc::Rc<LoadedModule>, String), CudaError> {
        let (bytes, kind): (Vec<u8>, &str) = match image {
            ModuleImage::Ptx(ptx) => {
                if ptx.as_bytes().contains(&0) {
                    return Err(err("PTX contains a NUL byte"));
                }
                (ptx.bytes().chain(std::iter::once(0)).collect(), "PTX")
            }
            ModuleImage::Cubin(b) => (b.to_vec(), "cubin"),
            ModuleImage::Fatbin(b) => (b.to_vec(), "fatbin"),
        };
        let options = if matches!(image, ModuleImage::Ptx(_)) { options } else { &[] };
        let key = use_cache.then(|| {
            let mut keyed = format!("{kind}\0{options:?}\0").into_bytes();
            keyed.extend_from_slice(&bytes);
            crate::backend::cache::sha256(&keyed)
        });
        if let Some(k) = &key {
            let hit = self.modules.borrow_mut().get(k);
            if let Some(hit) = hit {
                crate::cute_log!(Level::Debug, "module cache hit for `{entry}` ({kind}, {} bytes)", bytes.len());
                return Ok(hit);
            }
        }
        let mut logs = JitLogs::new();
        let (mut codes, mut values) = logs.attach(options);
        crate::cute_log!(
            Level::Debug,
            "loading a {kind} module of {} bytes for `{entry}` with JIT options {options:?}",
            bytes.len()
        );
        let mut module = std::ptr::null_mut();
        self.make_current()?;
        let started = Instant::now();
        // SAFETY: `bytes` is the image (NUL-terminated for PTX); the option arrays hold
        // `codes.len()` entries and outlive the call, as do the log buffers they point at.
        let r = unsafe {
            (self.api.module_load_data_ex)(
                &mut module,
                bytes.as_ptr().cast(),
                codes.len() as c_uint,
                codes.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        let (info, error) = logs.texts();
        if !info.is_empty() {
            crate::cute_log!(Level::Debug, "JIT log for `{entry}`:\n{info}");
        }
        check_with_log(&self.api, r, "cuModuleLoadDataEx", &error)?;
        self.loads.set(self.loads.get() + 1);
        crate::cute_log!(Level::Debug, "loaded `{entry}` in {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        let module =
            std::rc::Rc::new(LoadedModule::new(module, self.ctx, self.api.ctx_set_current, self.api.module_unload));
        if let Some(k) = key {
            let evicted =
                self.modules.borrow_mut().put(k, (module.clone(), info.clone()), crate::backend::cache::max_elems());
            drop(evicted);
        }
        Ok((module, info))
    }

    /// Load a compiled kernel. A cubin (made by `ptxas`, or [`attach_cubin`](Self::attach_cubin))
    /// is preferred when this device can run one built for the kernel's target
    /// ([`Arch::can_run_binary_built_for`]); otherwise the PTX is JIT-compiled with the options
    /// the kernel asks for ([`CompiledKernel::try_jit_options`]: its optimization level, line
    /// information and the `ptxas` flags the JIT has an option for). Fails if this device cannot
    /// run the kernel's target.
    fn load_inner(&self, kernel: &CompiledKernel) -> Result<CudaFunction<'_>, CudaError> {
        let (module, info) = self.load_compiled_image(kernel)?;
        self.function(module, kernel.name(), info)
    }

    /// The module of a compiled kernel ([`load_inner`](Self::load_inner)) and the JIT's info log.
    fn load_compiled_image(&self, kernel: &CompiledKernel) -> Result<(std::rc::Rc<LoadedModule>, String), CudaError> {
        let use_cache = kernel.options().get_no_cache() != Some(true) && !crate::backend::cache::no_cache_env();
        let (major, minor) = self.compute_capability();
        if !runs_on(kernel.arch(), major, minor) {
            return Err(err(format!(
                "kernel `{}` was compiled for {}, which {} (sm_{major}{minor}) cannot run",
                kernel.name(),
                kernel.arch().chip(),
                self.name()
            )));
        }
        if let (Some(cubin), Some(target)) = (kernel.cubin(), kernel.cubin_target()) {
            match self.arch() {
                Some(device) if device.can_run_binary_built_for(target) => {
                    return self.load_image(ModuleImage::Cubin(cubin), &[], kernel.name(), use_cache);
                }
                _ => crate::cute_log!(
                    Level::Debug,
                    "the cubin of `{}` is built for {}, which this device cannot run as a binary: loading the PTX",
                    kernel.name(),
                    target.chip()
                ),
            }
        }
        if let Ok(driver) = self.driver_version() {
            let ceiling = crate::driver_ptx_ceiling(driver);
            if kernel.ptx_version() > ceiling {
                let why = kernel
                    .ptx_reason()
                    .map_or_else(|| format!("its target, {}", kernel.arch().chip()), |w| w.to_string());
                return Err(err(format!(
                    "kernel `{}` is PTX ISA {}.{} (for {why}), but the CUDA driver ({}.{}) loads PTX ISA {}.{} at most",
                    kernel.name(),
                    kernel.ptx_version() / 10,
                    kernel.ptx_version() % 10,
                    driver / 1000,
                    driver % 1000 / 10,
                    ceiling / 10,
                    ceiling % 10
                )));
            }
        }
        let jit = kernel.try_jit_options().map_err(|e| err(e.to_string()))?;
        self.load_image(ModuleImage::Ptx(kernel.ptx()), &jit, kernel.name(), use_cache)
    }

    /// Load a compiled kernel ([`load_inner`](Self::load_inner)) and keep its signature on the
    /// function, so that [`CudaFunction::launch_builder`] and [`CudaFunction::global`] can
    /// check against it.
    pub fn load(&self, kernel: &CompiledKernel) -> Result<CudaFunction<'_>, CudaError> {
        let mut f = self.load_inner(kernel)?;
        f.signature = Some(kernel.signature().clone());
        Ok(f)
    }

    /// Load an exported kernel ([`CompiledKernel::export_c`]) from its manifest (`<name>.sig`)
    /// and the images beside it, without the kernel's IR (CuTe DSL's `cute.runtime.load_module`).
    /// The function carries the exported signature, so [`CudaFunction::launch_builder`] checks
    /// and packs arguments as for a kernel compiled in this process. Fails on another export
    /// format version, a device that cannot run the kernel's target, or PTX newer than the
    /// driver loads ([`ExportedKernel::choose_image`]).
    ///
    /// [`ExportedKernel::choose_image`]: crate::aot::ExportedKernel::choose_image
    pub fn load_exported(&self, path: impl AsRef<std::path::Path>) -> Result<CudaFunction<'_>, CudaError> {
        let kernel = crate::aot::ExportedKernel::read(path).map_err(|e| err(e.to_string()))?;
        self.load_exported_kernel(&kernel)
    }

    /// [`load_exported`](Self::load_exported) from an [`ExportedKernel`] in memory
    /// ([`ExportedKernel::from_parts`], for instance from `include_bytes!`).
    ///
    /// [`ExportedKernel`]: crate::aot::ExportedKernel
    /// [`ExportedKernel::from_parts`]: crate::aot::ExportedKernel::from_parts
    pub fn load_exported_kernel(&self, kernel: &crate::aot::ExportedKernel) -> Result<CudaFunction<'_>, CudaError> {
        let (major, minor) = self.compute_capability();
        let cc = (major.max(0) as u32, minor.max(0) as u32);
        let choice = kernel.choose_image(cc, self.driver_ptx().ok()).map_err(|e| err(e.to_string()))?;
        let name = &kernel.signature.name;
        let mut f = match (choice, &kernel.cubin, &kernel.ptx) {
            (crate::aot::ImageChoice::Cubin, Some(cubin), _) => {
                self.load_module(ModuleImage::Cubin(cubin), &[], name)?
            }
            (_, _, Some(ptx)) => self.load_module(ModuleImage::Ptx(ptx), &[], name)?,
            _ => return Err(err(format!("exported kernel `{name}` has no image this device loads"))),
        };
        f.signature = Some(kernel.signature.clone());
        Ok(f)
    }

    /// Turn `ptx` into a cubin with the driver's linker (`cuLinkCreate`, `cuLinkAddData`,
    /// `cuLinkComplete`), for machines with the driver but no `ptxas`. The cubin is for this
    /// device's chip; `options` are the JIT options the linker takes (a register limit, the
    /// optimization level, line information, `-v`). A failure reports the linker's error log.
    pub fn link_cubin(&self, ptx: &str, name: &str, options: &[JitOption]) -> Result<Vec<u8>, CudaError> {
        if ptx.as_bytes().contains(&0) {
            return Err(err("PTX contains a NUL byte"));
        }
        let mut logs = JitLogs::new();
        let (mut codes, mut values) = logs.attach(options);
        crate::cute_log!(Level::Debug, "linking a cubin for `{name}` with the driver's linker, options {options:?}");
        self.make_current()?;
        let mut state: LinkState = std::ptr::null_mut();
        // SAFETY: option arrays and log buffers outlive the calls; `state` is a valid out-pointer.
        let r = unsafe {
            (self.api.link_create)(codes.len() as c_uint, codes.as_mut_ptr(), values.as_mut_ptr(), &mut state)
        };
        check(&self.api, r, "cuLinkCreate")?;
        struct Destroy<'a>(&'a Api, LinkState);
        impl Drop for Destroy<'_> {
            fn drop(&mut self) {
                // SAFETY: the state came from cuLinkCreate and is destroyed once.
                unsafe { (self.0.link_destroy)(self.1) };
            }
        }
        let _destroy = Destroy(&self.api, state);
        let image: Vec<u8> = ptx.bytes().chain(std::iter::once(0)).collect();
        let label = CString::new(name).map_err(|_| err("kernel name contains a NUL byte"))?;
        // SAFETY: the image and name outlive the call, which copies the data.
        let r = unsafe {
            (self.api.link_add_data)(
                state,
                CU_JIT_INPUT_PTX,
                image.as_ptr() as *mut c_void,
                image.len(),
                label.as_ptr(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        check_with_log(&self.api, r, "cuLinkAddData", &logs.texts().1)?;
        let mut cubin: *mut c_void = std::ptr::null_mut();
        let mut size = 0usize;
        // SAFETY: valid out-pointers; the cubin is owned by the link state, which lives until
        // the end of this function, and is copied before then.
        let r = unsafe { (self.api.link_complete)(state, &mut cubin, &mut size) };
        let (info, error) = logs.texts();
        if !info.is_empty() {
            crate::cute_log!(Level::Debug, "linker log for `{name}`:\n{info}");
        }
        check_with_log(&self.api, r, "cuLinkComplete", &error)?;
        // SAFETY: `cubin` points at `size` bytes owned by the link state.
        Ok(unsafe { std::slice::from_raw_parts(cubin.cast::<u8>(), size) }.to_vec())
    }

    /// Give `kernel` a cubin made with the driver's linker if it has none (`ptxas` was not found
    /// or was turned off): the cubin artifact and the SASS then exist, and
    /// [`load`](Self::load) takes the cubin on devices that run binaries built for this one
    /// (the linker's SASS is for this device, whatever the kernel's target).
    pub fn attach_cubin(&self, kernel: &mut CompiledKernel, toolchain: &Toolchain) -> Result<(), CudaError> {
        if kernel.cubin().is_some() {
            return Ok(());
        }
        let device = self.arch().ok_or_else(|| {
            err(format!(
                "{} (compute capability {:?}) is not an architecture cute-rs knows, so a cubin linked for it \
                 could not be matched to devices later",
                self.name(),
                self.compute_capability()
            ))
        })?;
        let jit = kernel.try_jit_options().map_err(|e| err(e.to_string()))?;
        let cubin = self.link_cubin(kernel.ptx(), kernel.name(), &jit)?;
        kernel.set_cubin(cubin, device, toolchain);
        Ok(())
    }

    /// Look up entry point `name` in a loaded module (unloaded on failure once nothing else
    /// holds it).
    fn function(
        &self,
        module: std::rc::Rc<LoadedModule>,
        name: &str,
        jit_log: String,
    ) -> Result<CudaFunction<'_>, CudaError> {
        let entry = CString::new(name).map_err(|_| err("kernel name contains a NUL byte"))?;
        let mut function = std::ptr::null_mut();
        self.make_current()?;
        // SAFETY: the module is live; the out-pointer is valid.
        let r = unsafe { (self.api.module_get_function)(&mut function, module.module, entry.as_ptr()) };
        check(&self.api, r, "cuModuleGetFunction")?;
        Ok(CudaFunction { ctx: self, module, function, jit_log, signature: None })
    }

    fn make_current(&self) -> Result<(), CudaError> {
        // SAFETY: ctx is the retained primary context.
        check(&self.api, unsafe { (self.api.ctx_set_current)(self.ctx) }, "cuCtxSetCurrent")
    }

    /// Copy `arg` (a buffer) to the device.
    pub fn upload(&self, arg: &KernelArg) -> Result<DeviceBuffer<'_>, CudaError> {
        if arg.buffer_type().is_none() {
            return Err(err(format!("cannot upload {}: not a buffer", arg.describe())));
        }
        self.make_current()?;
        let (ptr, bytes) = host_bytes(arg);
        let mut device: CuDevicePtr = 0;
        // SAFETY: a valid out-pointer.
        unsafe {
            check(&self.api, (self.api.mem_alloc)(&mut device, bytes.max(1)), "cuMemAlloc")?;
        }
        let buffer = DeviceBuffer { ctx: self, ptr: device, bytes };
        // SAFETY: copies `bytes` bytes from live host memory into the new allocation.
        unsafe {
            check(&self.api, (self.api.memcpy_htod)(device, ptr, bytes), "cuMemcpyHtoD")?;
        }
        self.settle_legacy_stream()?;
        Ok(buffer)
    }

    /// Wait for the legacy NULL stream (`cuStreamSynchronize(0)`). The synchronous copies and
    /// memsets run on it, and some return before the device has finished (`cuMemcpyHtoD`
    /// from pageable memory returns once the bytes are staged; `cuMemcpyDtoD` and
    /// `cuMemsetD8` do not wait at all). Work on a non-blocking stream does not wait for the
    /// legacy stream, so without this a launch on one could read the memory before the copy
    /// lands.
    pub(crate) fn settle_legacy_stream(&self) -> Result<(), CudaError> {
        // SAFETY: the legacy stream of the current context.
        check(&self.api, unsafe { (self.api.stream_synchronize)(std::ptr::null_mut()) }, "cuStreamSynchronize")
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

    /// Wait for everything launched so far on every stream of the context
    /// (`cuCtxSynchronize`).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.make_current()?;
        // SAFETY: the context is current.
        check(&self.api, unsafe { (self.api.ctx_synchronize)() }, "kernel execution")
    }

    /// Launch `function` on buffers already on the device and wait for it (on the legacy
    /// NULL stream).
    pub fn launch_on_device(
        &self,
        kernel: &Kernel,
        function: &CudaFunction<'_>,
        cfg: LaunchConfig,
        args: &[DeviceArg<'_>],
    ) -> Result<LaunchStats, CudaError> {
        let start = Instant::now();
        let stream = self.default_stream();
        self.launch_builder(kernel, function, cfg).args(args).submit()?;
        stream.synchronize()?;
        Ok(LaunchStats { kernel_seconds: start.elapsed().as_secs_f64() })
    }

    /// Launch `function` on the legacy NULL stream without waiting. Kernels run in
    /// submission order; [`synchronize`](Self::synchronize) waits for them. The buffers must
    /// stay alive until then. [`launch_builder`](Self::launch_builder) launches on another
    /// stream.
    pub fn launch_async(
        &self,
        kernel: &Kernel,
        function: &CudaFunction<'_>,
        cfg: LaunchConfig,
        args: &[DeviceArg<'_>],
    ) -> Result<(), CudaError> {
        self.launch_builder(kernel, function, cfg).args(args).submit()
    }

    /// Queue `function` on `stream` without waiting ([`launch_builder`](Self::launch_builder)
    /// with a stream): nothing is allocated or synchronised, so it can be captured into a
    /// graph.
    pub fn launch_on(
        &self,
        stream: &CudaStream<'_>,
        kernel: &Kernel,
        function: &CudaFunction<'_>,
        cfg: LaunchConfig,
        args: &[DeviceArg<'_>],
    ) -> Result<(), CudaError> {
        self.launch_builder(kernel, function, cfg).stream(stream).args(args).submit()
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
        self.launch_external(kernel, function, cfg, args, &[])
    }

    /// [`launch`](Self::launch) where some buffer arguments are external memory
    /// ([`TypedKernel::pack_full`](crate::TypedKernel::pack_full)): device memory is passed by
    /// address (its argument is a placeholder, and it must be on this context's device);
    /// everything else is copied in and back.
    pub fn launch_external(
        &self,
        kernel: &Kernel,
        function: &CudaFunction<'_>,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        externals: &[Option<crate::params::ExternalArg>],
    ) -> Result<LaunchStats, CudaError> {
        let device_of = |i: usize| externals.get(i).copied().flatten().and_then(|e| e.device);
        if (0..args.len()).all(|i| device_of(i).is_none()) {
            crate::sim::check_launch(&kernel.signature(), cfg, args).map_err(err)?;
        }
        let names = crate::params::param_names(kernel.ir());
        let mut uploaded = Vec::with_capacity(args.len());
        for (i, (param, arg)) in kernel.ir().args.iter().zip(args.iter()).enumerate() {
            uploaded.push(match (param, device_of(i)) {
                (Param::Buffer(_), None) => Some(self.upload(arg)?),
                (Param::Buffer(_), Some((_, _, id))) if id != self.ordinal as i32 => {
                    return Err(err(format!(
                        "argument `{}` is memory of CUDA device {id}, but the launch is on device {} ({})",
                        names[i],
                        self.ordinal,
                        self.name()
                    )));
                }
                _ => None,
            });
        }
        let device_args: Vec<DeviceArg<'_>> = uploaded
            .iter()
            .zip(args.iter())
            .zip(&kernel.ir().args)
            .enumerate()
            .map(|(i, ((b, a), p))| match (b, p, device_of(i)) {
                (Some(b), _, _) => DeviceArg::Buffer(b),
                (None, Param::Buffer(id), Some((address, bytes, _))) => DeviceArg::Ptr(DevicePtr {
                    spec: PointerSpec::new(kernel.ir().buffer(*id).elem, crate::tensor::AddressSpace::Gmem, Some(1)),
                    address,
                    bytes: Some(bytes),
                }),
                (None, Param::ByValue { .. }, _) => DeviceArg::ByValue(a.byte_data().to_vec()),
                (None, _, _) => DeviceArg::Scalar(a.clone()),
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

    /// The most shared memory this device gives one block (opting in), or 0 if the driver
    /// cannot say.
    pub fn max_shared_memory_per_block(&self) -> usize {
        self.raw_device_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).map_or(0, |b| b as usize)
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
        let _ = self.ctx.make_current();
        // SAFETY: ptr was returned by cuMemAlloc.
        unsafe { (self.ctx.api.mem_free)(self.ptr) };
    }
}

/// A raw device address as a launch argument (CuTe DSL's `cute.runtime.make_ptr` and
/// `nullptr`): its [`PointerSpec`] (element type, memory space, assumed alignment), the
/// address and, when known, the bytes behind it (for the launch's bounds check).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevicePtr {
    spec: PointerSpec,
    address: u64,
    bytes: Option<usize>,
}

impl DevicePtr {
    /// A pointer to `dtype` elements at `address` in global memory, aligned to `assumed_align`
    /// bytes (the element's size by default), which the address must be a multiple of, with
    /// `bytes` bytes behind it when known.
    ///
    /// # Safety
    ///
    /// `address` is device memory of the context the launch runs on (or memory it can reach),
    /// at least `bytes` long, alive until the kernels using it finish.
    pub unsafe fn from_raw(
        dtype: crate::ir::DType,
        address: u64,
        assumed_align: Option<usize>,
        bytes: Option<usize>,
    ) -> Result<DevicePtr, CudaError> {
        let spec = PointerSpec::new(dtype, crate::tensor::AddressSpace::Gmem, assumed_align);
        spec.check_address(address).map_err(err)?;
        Ok(DevicePtr { spec, address, bytes })
    }

    /// A null pointer to `dtype` elements (CuTe DSL's `nullptr`), for a parameter the kernel
    /// does not use in this launch.
    pub fn null(dtype: crate::ir::DType) -> DevicePtr {
        DevicePtr { spec: PointerSpec::new(dtype, crate::tensor::AddressSpace::Gmem, None), address: 0, bytes: None }
    }

    pub fn spec(&self) -> PointerSpec {
        self.spec
    }

    pub fn address(&self) -> u64 {
        self.address
    }

    pub fn bytes(&self) -> Option<usize> {
        self.bytes
    }

    pub fn is_null(&self) -> bool {
        self.address == 0
    }
}

impl<'c> DeviceBuffer<'c> {
    /// The `bytes` bytes of this buffer from `byte_offset`, as a launch argument (a borrowed
    /// view; the launch checks the parameter's alignment and length against it).
    pub fn view(&self, byte_offset: usize, bytes: usize) -> Result<DeviceArg<'_>, CudaError> {
        if byte_offset.checked_add(bytes).is_none_or(|end| end > self.bytes) {
            return Err(err(format!(
                "a view of {bytes} bytes at byte {byte_offset} of a buffer of {} bytes",
                self.bytes
            )));
        }
        Ok(DeviceArg::Slice { buffer: self, byte_offset, bytes })
    }
}

/// An argument to a launch on device memory.
pub enum DeviceArg<'a> {
    Buffer(&'a DeviceBuffer<'a>),
    /// A pointer into a buffer: its start plus `byte_offset` bytes (checked against the
    /// parameter's alignment).
    View {
        buffer: &'a DeviceBuffer<'a>,
        byte_offset: usize,
    },
    /// `bytes` bytes of a buffer from `byte_offset` ([`DeviceBuffer::view`]).
    Slice {
        buffer: &'a DeviceBuffer<'a>,
        byte_offset: usize,
        bytes: usize,
    },
    /// A raw device address ([`DevicePtr::from_raw`]) or a null pointer ([`DevicePtr::null`]).
    Ptr(DevicePtr),
    /// One of the scalar [`KernelArg`] variants.
    Scalar(KernelArg),
    /// The bytes of a by-value parameter ([`param_struct`](crate::param_struct)).
    ByValue(Vec<u8>),
}

impl CudaContext {
    /// Reset the device's primary context (`cuDevicePrimaryCtxReset`) and
    /// retain a fresh one, which recovers from errors that leave the
    /// context unusable. Taking `&mut self` means no buffer or function of
    /// the old context survives. A failed device assertion
    /// (`CUDA_ERROR_ASSERT`) cannot be recovered this way: the driver keeps refusing
    /// the process (checked with driver 595 on an SM89, where retaining the
    /// reset context fails with `CUDA_ERROR_ASSERT`), so `reset` returns
    /// that error; run kernels that may assert in a child process.
    pub fn reset(&mut self) -> Result<(), CudaError> {
        // No function survives (`&mut self`), so this unloads every cached module.
        self.clear_module_cache();
        // SAFETY: the device is valid; the old context is released before
        // the reset and the new one retained before use.
        unsafe {
            (self.api.ctx_set_current)(std::ptr::null_mut());
            (self.api.primary_ctx_release)(self.device);
            // Until a new context is retained, `Drop` has nothing to release.
            self.released = true;
            self.ctx = std::ptr::null_mut();
            check(&self.api, (self.api.primary_ctx_reset)(self.device), "cuDevicePrimaryCtxReset")?;
            let mut ctx = std::ptr::null_mut();
            check(&self.api, (self.api.primary_ctx_retain)(&mut ctx, self.device), "cuDevicePrimaryCtxRetain")?;
            self.ctx = ctx;
            self.released = false;
        }
        self.make_current()
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // Unload the cached modules while the context is still retained.
        self.clear_module_cache();
        if !self.released {
            // SAFETY: releases the primary context retained in `new`.
            unsafe { (self.api.primary_ctx_release)(self.device) };
        }
    }
}

/// A loaded kernel entry point.
pub struct CudaFunction<'a> {
    ctx: &'a CudaContext,
    /// The module, shared with the context's module cache and other functions loaded from it.
    module: std::rc::Rc<LoadedModule>,
    function: CuFunction,
    jit_log: String,
    /// The signature of the compiled kernel it was loaded from.
    signature: Option<KernelSignature>,
}

impl CudaFunction<'_> {
    /// The JIT's info log from loading this function's module: with
    /// [`JitOption::LogVerbose`] (`-v`) the assembler's resource usage, else usually empty.
    pub fn jit_log(&self) -> &str {
        &self.jit_log
    }

    /// The signature of the compiled kernel this was loaded from, if any.
    pub fn signature(&self) -> Option<&KernelSignature> {
        self.signature.as_ref()
    }

    fn set_attribute_raw(&self, attribute: c_int, value: c_int) -> Result<(), CudaError> {
        self.ctx.make_current()?;
        // SAFETY: a loaded function.
        let r = unsafe { (self.ctx.api.func_set_attribute)(self.function, attribute, value) };
        check(&self.ctx.api, r, "cuFuncSetAttribute")
    }

    /// Module global `name` ([`module_global`](crate::module_global)), to read and write
    /// from the host between launches (CuTe DSL's `write_to_symbol`).
    pub fn global(&self, name: &str) -> Result<DeviceGlobal<'_>, CudaError> {
        let (ptr, bytes) = self.global_address(name)?;
        let readonly = self.signature.as_ref().and_then(|s| s.global(name)).is_some_and(|g| g.readonly);
        Ok(DeviceGlobal { function: self, name: name.to_string(), ptr, bytes, readonly })
    }

    /// A `cuFuncGetAttribute` value (`CU_FUNC_ATTRIBUTE_*`).
    pub fn attribute(&self, attribute: i32) -> Result<i32, CudaError> {
        let mut v = 0;
        // SAFETY: a valid out-pointer and a loaded function.
        check(
            &self.ctx.api,
            unsafe { (self.ctx.api.func_get_attribute)(&mut v, attribute, self.function) },
            "cuFuncGetAttribute",
        )?;
        Ok(v)
    }

    /// The function's resources as the driver reports them
    /// (`cuFuncGetAttribute`: registers, local, shared and constant bytes,
    /// the block size limit, the PTX and binary versions). Spill counts are
    /// not among them; [`ResourceUsage::from_jit_log`] reads them from a
    /// verbose JIT log.
    pub fn resource_usage(&self) -> Result<crate::diagnostics::ResourceUsage, CudaError> {
        Ok(crate::diagnostics::ResourceUsage {
            registers: Some(self.attribute(4)? as u32),
            stack_bytes: Some(self.attribute(3)? as u32),
            shared_bytes: Some(self.attribute(1)? as u32),
            constant_bytes: Some(self.attribute(2)? as u32),
            max_threads_per_block: Some(self.attribute(0)? as u32),
            ptx_version: Some(self.attribute(5)? as u32),
            binary_version: Some(self.attribute(6)? as u32),
            ..Default::default()
        })
    }
}

/// A module global of a loaded function's module (`cuModuleGetGlobal`).
pub struct DeviceGlobal<'f> {
    function: &'f CudaFunction<'f>,
    name: String,
    ptr: CuDevicePtr,
    bytes: usize,
    readonly: bool,
}

impl DeviceGlobal<'_> {
    /// Its size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Its device address.
    pub fn as_raw(&self) -> u64 {
        self.ptr
    }

    /// Its whole contents (`cuMemcpyDtoH`, on the legacy NULL stream: synchronise a
    /// non-blocking stream whose kernels write it first).
    pub fn read<T: DeviceRepr>(&self) -> Result<Vec<T>, CudaError> {
        let size = std::mem::size_of::<T>();
        if !self.bytes.is_multiple_of(size) {
            return Err(err(format!(
                "global `{}` of {} bytes is no whole number of {size}-byte elements",
                self.name, self.bytes
            )));
        }
        let mut out = vec![T::default(); self.bytes / size];
        let api = &self.function.ctx.api;
        self.function.ctx.make_current()?;
        // SAFETY: `out` holds `bytes` writable bytes.
        check(api, unsafe { (api.memcpy_dtoh)(out.as_mut_ptr().cast(), self.ptr, self.bytes) }, "cuMemcpyDtoH")?;
        Ok(out)
    }

    /// Overwrite it with `values`, which must fill it exactly (`cuMemcpyHtoD`); the next
    /// launch sees them. A global declared read-only is refused: the kernel may have folded
    /// its value.
    pub fn write<T: DeviceRepr>(&self, values: &[T]) -> Result<(), CudaError> {
        if self.readonly {
            return Err(err(format!(
                "module global `{}` is read-only (an LLVM constant): the kernel may have folded its value, so a \
                 write would go unseen",
                self.name
            )));
        }
        let bytes = std::mem::size_of_val(values);
        if bytes != self.bytes {
            return Err(err(format!("module global `{}` is {} bytes, got {bytes}", self.name, self.bytes)));
        }
        let api = &self.function.ctx.api;
        self.function.ctx.make_current()?;
        // SAFETY: `values` is `bytes` readable bytes and the global is that large.
        check(api, unsafe { (api.memcpy_htod)(self.ptr, values.as_ptr().cast(), bytes) }, "cuMemcpyHtoD")?;
        self.function.ctx.settle_legacy_stream()
    }
}

/// The log buffers of one JIT or linker call.
struct JitLogs {
    info: Vec<u8>,
    error: Vec<u8>,
}

impl JitLogs {
    fn new() -> JitLogs {
        JitLogs { info: vec![0; JIT_LOG_BYTES], error: vec![0; JIT_LOG_BYTES] }
    }

    /// The option arrays for `options` plus the two logs. Integer values go in place of their
    /// pointers, as the driver expects.
    fn attach(&mut self, options: &[JitOption]) -> (Vec<c_int>, Vec<*mut c_void>) {
        let mut codes: Vec<c_int> = options.iter().map(|o| o.code() as c_int).collect();
        let mut values: Vec<*mut c_void> = options.iter().map(|o| o.value() as usize as *mut c_void).collect();
        for (code, value) in [
            (CU_JIT_ERROR_LOG_BUFFER, self.error.as_mut_ptr() as usize),
            (CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, JIT_LOG_BYTES),
            (CU_JIT_INFO_LOG_BUFFER, self.info.as_mut_ptr() as usize),
            (CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, JIT_LOG_BYTES),
        ] {
            codes.push(code);
            values.push(value as *mut c_void);
        }
        (codes, values)
    }

    /// `(info, error)` as text.
    fn texts(&self) -> (String, String) {
        let text = |b: &[u8]| {
            let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
            String::from_utf8_lossy(&b[..end]).trim().to_string()
        };
        (text(&self.info), text(&self.error))
    }
}

/// [`check`], with the JIT or linker's error log appended to the message.
fn check_with_log(api: &Api, r: CuResult, what: &str, log: &str) -> Result<(), CudaError> {
    check(api, r, what).map_err(|mut e| {
        if !log.is_empty() {
            e.message.push_str(&format!("\nthe assembler's log:\n{log}"));
        }
        e
    })
}

fn check(api: &Api, r: CuResult, what: &str) -> Result<(), CudaError> {
    if r == 0 {
        return Ok(());
    }
    let text = |f: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult| {
        let mut s: *const c_char = std::ptr::null();
        // SAFETY: the driver writes a static string pointer (or leaves null).
        unsafe { f(r, &mut s) };
        // SAFETY: a static NUL-terminated string owned by the driver.
        (!s.is_null()).then(|| unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned())
    };
    let name = text(api.get_error_name).unwrap_or_else(|| format!("error {r}"));
    let mut message = match text(api.get_error_string) {
        Some(d) => format!("{what} failed: {name}: {d}"),
        None => format!("{what} failed: {name}"),
    };
    match r {
        900 => message.push_str(
            " (the call cannot run while a stream is capturing a graph: graphs record stream-ordered work only, \
             so launch with LaunchBuilder::stream on the capturing stream, and allocate, upload and synchronise \
             before the capture or after it)",
        ),
        901 => message.push_str(
            " (an earlier call that cannot be captured invalidated the stream's capture: end it, drop the graph \
             and capture again with only stream-ordered work)",
        ),
        _ => {}
    }
    if r == CUDA_ERROR_ASSERT {
        message.push_str(
            " (a device assertion failed: the device printed its message, file and line on stderr; the error is \
             sticky, so this process can no longer use the GPU, not even after CudaContext::reset: run kernels \
             that may assert in a child process)",
        );
    }
    Err(CudaError { message, code: Some(r) })
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

/// Whether a device of compute capability `major.minor` runs PTX compiled for `arch`: PTX for
/// a baseline target runs on it and every later architecture (the driver JIT-compiles it),
/// family-specific PTX (`sm_100f`) on the later chips of its family (the same major version),
/// and architecture-specific PTX (`sm_90a` and the like) on that exact chip only.
fn runs_on(arch: Arch, major: i32, minor: i32) -> bool {
    let device = (major.max(0) as u32, minor.max(0) as u32);
    match arch.suffix() {
        "a" => device == arch.version(),
        "f" => device.0 == arch.major() && device.1 >= arch.minor(),
        _ => device >= arch.version(),
    }
}

impl Kernel {
    /// Compile for `ctx`'s device with `toolchain` at the default options plus the
    /// environment (`CUTE_RS_COMPILER_OPT`, `CUTE_RS_ARCH`, `CUTE_RS_LINEINFO`), run, and copy
    /// results back into `args` (the same argument convention as [`Kernel::simulate`]).
    pub fn run_cuda(
        &self,
        ctx: &CudaContext,
        toolchain: &Toolchain,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
    ) -> Result<LaunchStats, Box<dyn std::error::Error>> {
        self.run_cuda_with(ctx, toolchain, &CompileOptions::new().with_env()?, cfg, args)
    }

    /// Compile with `options` (for `ctx`'s device unless they name a target), load with the
    /// JIT options they ask for, run, and copy results back into `args`.
    pub fn run_cuda_with(
        &self,
        ctx: &CudaContext,
        toolchain: &Toolchain,
        options: &CompileOptions,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
    ) -> Result<LaunchStats, Box<dyn std::error::Error>> {
        let options = match options.get_gpu_arch().or(self.target()) {
            Some(_) => options.clone(),
            None => {
                let arch = ctx.best_target(toolchain).ok_or_else(|| {
                    let (major, minor) = ctx.compute_capability();
                    err(format!(
                        "{} (sm_{major}{minor}) has no target that cute-rs knows and {} and the CUDA driver handle",
                        ctx.name(),
                        toolchain.mlir_opt.display()
                    ))
                })?;
                options.clone().gpu_arch(arch)
            }
        };
        let mut compiled = self.compile_with(toolchain, &options)?;
        // A cubin or SASS kept on a machine without ptxas comes from the driver's linker.
        if compiled.cubin().is_none()
            && (options.get_keep(Artifact::Cubin) == Some(true) || options.get_keep(Artifact::Sass) == Some(true))
        {
            ctx.attach_cubin(&mut compiled, toolchain)?;
        }
        let function = ctx.load(&compiled)?;
        Ok(ctx.launch(self, &function, cfg, args)?)
    }

    /// [`run_cuda_with`](Self::run_cuda_with) where some buffer arguments are external memory
    /// ([`CudaContext::launch_external`]).
    pub fn run_cuda_external(
        &self,
        ctx: &CudaContext,
        toolchain: &Toolchain,
        options: &CompileOptions,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        externals: &[Option<crate::params::ExternalArg>],
    ) -> Result<LaunchStats, Box<dyn std::error::Error>> {
        if externals.iter().all(|e| e.and_then(|e| e.device).is_none()) {
            return self.run_cuda_with(ctx, toolchain, options, cfg, args);
        }
        let options = match options.get_gpu_arch().or(self.target()) {
            Some(_) => options.clone(),
            None => options.clone().gpu_arch(ctx.best_target(toolchain).ok_or("no target for this device")?),
        };
        let compiled = self.compile_with(toolchain, &options)?;
        let function = ctx.load(&compiled)?;
        Ok(ctx.launch_external(self, &function, cfg, args, externals)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_a_device_runs() {
        assert!(runs_on(Arch::Sm80, 8, 9) && runs_on(Arch::Sm89, 8, 9) && runs_on(Arch::Sm80, 12, 0));
        assert!(!runs_on(Arch::Sm90, 8, 9) && !runs_on(Arch::Sm86, 8, 0));
        assert!(runs_on(Arch::Sm90a, 9, 0) && !runs_on(Arch::Sm90a, 10, 0) && !runs_on(Arch::Sm100a, 10, 3));
        assert!(runs_on(Arch::Sm100, 12, 0) && !runs_on(Arch::Sm120, 10, 0));
        // Family PTX runs on the later chips of its family only.
        assert!(runs_on(Arch::Sm100f, 10, 3) && runs_on(Arch::Sm120f, 12, 1) && !runs_on(Arch::Sm100f, 12, 0));
        assert!(runs_on(Arch::Sm110a, 11, 0) && runs_on(Arch::Sm103, 11, 0) && !runs_on(Arch::Sm121, 12, 0));
    }
}
