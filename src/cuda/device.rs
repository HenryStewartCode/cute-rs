//! Device and function queries (CuTe DSL's `cutlass.cuda.get_device_info`,
//! `get_device_attribute`, `get_driver_version`, `set_kernel_attribute`), module loading
//! from files, and raw handles for interop.

use std::ffi::{CString, c_int, c_void};
use std::fmt;
use std::path::Path;

use super::{CudaContext, CudaError, CudaFunction, ModuleImage, check, err, load_api};

/// A `CUdevice_attribute`, as `cuDeviceGetAttribute` takes it (values from `cuda.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceAttribute {
    MaxThreadsPerBlock = 1,
    /// Shared memory a block may use without opting in (48 KB).
    MaxSharedMemoryPerBlock = 8,
    /// Bytes of `__constant__` memory.
    TotalConstantMemory = 9,
    WarpSize = 10,
    MaxRegistersPerBlock = 12,
    /// Kilohertz.
    ClockRate = 13,
    MultiprocessorCount = 16,
    GlobalMemoryBusWidth = 37,
    L2CacheSize = 38,
    MaxThreadsPerMultiprocessor = 39,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    MaxSharedMemoryPerMultiprocessor = 81,
    MaxRegistersPerMultiprocessor = 82,
    CooperativeLaunch = 95,
    /// The most shared memory a block may opt in to (101376 bytes on SM89).
    MaxSharedMemoryPerBlockOptin = 97,
    MaxBlocksPerMultiprocessor = 106,
    /// Shared memory the driver reserves in each block.
    ReservedSharedMemoryPerBlock = 111,
    /// `cuMemAllocAsync` and memory pools.
    MemoryPoolsSupported = 115,
    /// Thread block clusters (SM90 on).
    ClusterLaunch = 120,
}

/// A `CUfunction_attribute`, as `cuFuncGetAttribute` and `cuFuncSetAttribute` take it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FuncAttribute {
    MaxThreadsPerBlock = 0,
    SharedSizeBytes = 1,
    ConstSizeBytes = 2,
    LocalSizeBytes = 3,
    NumRegs = 4,
    PtxVersion = 5,
    BinaryVersion = 6,
    CacheModeCa = 7,
    /// Settable: the most dynamic shared memory a launch may ask for.
    MaxDynamicSharedSizeBytes = 8,
    /// Settable: the share of the unified L1/shared storage to make shared memory, in
    /// percent (-1 for the driver's default, 0 for the most L1, 100 for the most shared).
    PreferredSharedMemoryCarveout = 9,
    ClusterSizeMustBeSet = 10,
    RequiredClusterWidth = 11,
    RequiredClusterHeight = 12,
    RequiredClusterDepth = 13,
    NonPortableClusterSizeAllowed = 14,
    ClusterSchedulingPolicyPreference = 15,
}

/// A summary of a device (CuTe DSL's `DeviceInfo`); `Display` prints it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub sm_count: u32,
    pub l2_cache_bytes: u64,
    /// Shared memory a block may opt in to.
    pub shared_memory_per_block_optin: u64,
    pub shared_memory_per_sm: u64,
    pub total_memory: u64,
    pub driver_version: u32,
    pub cluster_launch: bool,
    pub memory_pools: bool,
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        writeln!(f, "device {}: {}", self.ordinal, self.name)?;
        writeln!(f, "  compute capability: {}.{}", self.compute_capability.0, self.compute_capability.1)?;
        writeln!(f, "  SMs: {}", self.sm_count)?;
        writeln!(f, "  L2 cache: {:.1} MiB", mib(self.l2_cache_bytes))?;
        writeln!(
            f,
            "  shared memory: {} bytes per block (opt-in), {} per SM",
            self.shared_memory_per_block_optin, self.shared_memory_per_sm
        )?;
        writeln!(f, "  global memory: {:.0} MiB", mib(self.total_memory))?;
        writeln!(f, "  clusters: {}", if self.cluster_launch { "yes" } else { "no" })?;
        write!(f, "  driver: CUDA {}.{}", self.driver_version / 1000, self.driver_version % 1000 / 10)
    }
}

impl CudaContext {
    /// The number of CUDA devices (`cuDeviceGetCount`); fails where there is no driver.
    pub fn device_count() -> Result<usize, CudaError> {
        let api = load_api()?;
        let mut n: c_int = 0;
        // SAFETY: cuInit first, then a valid out-pointer.
        unsafe {
            check(&api, (api.init)(0), "cuInit")?;
            check(&api, (api.device_get_count)(&mut n), "cuDeviceGetCount")?;
        }
        Ok(n.max(0) as usize)
    }

    /// The device ordinal this context was created for.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// A device attribute (`cuDeviceGetAttribute`).
    pub fn device_attribute(&self, attribute: DeviceAttribute) -> Result<i32, CudaError> {
        self.raw_device_attribute(attribute as c_int)
    }

    pub(super) fn raw_device_attribute(&self, attribute: c_int) -> Result<i32, CudaError> {
        let mut v: c_int = 0;
        // SAFETY: a valid out-pointer.
        let r = unsafe { (self.api.device_get_attribute)(&mut v, attribute, self.device) };
        check(&self.api, r, "cuDeviceGetAttribute")?;
        Ok(v)
    }

    /// Bytes of global memory (`cuDeviceTotalMem`).
    pub fn total_memory(&self) -> Result<u64, CudaError> {
        let mut bytes = 0usize;
        // SAFETY: a valid out-pointer.
        check(&self.api, unsafe { (self.api.device_total_mem)(&mut bytes, self.device) }, "cuDeviceTotalMem")?;
        Ok(bytes as u64)
    }

    /// Name, compute capability, SMs, caches, memory and driver of the device.
    pub fn device_info(&self) -> Result<DeviceInfo, CudaError> {
        let a = |x: DeviceAttribute| self.device_attribute(x);
        Ok(DeviceInfo {
            ordinal: self.ordinal,
            name: self.name(),
            compute_capability: (
                a(DeviceAttribute::ComputeCapabilityMajor)?,
                a(DeviceAttribute::ComputeCapabilityMinor)?,
            ),
            sm_count: a(DeviceAttribute::MultiprocessorCount)? as u32,
            l2_cache_bytes: a(DeviceAttribute::L2CacheSize)? as u64,
            shared_memory_per_block_optin: a(DeviceAttribute::MaxSharedMemoryPerBlockOptin)? as u64,
            shared_memory_per_sm: a(DeviceAttribute::MaxSharedMemoryPerMultiprocessor)? as u64,
            total_memory: self.total_memory()?,
            driver_version: self.driver_version()?,
            cluster_launch: a(DeviceAttribute::ClusterLaunch).unwrap_or(0) != 0,
            memory_pools: a(DeviceAttribute::MemoryPoolsSupported).unwrap_or(0) != 0,
        })
    }

    /// The `CUcontext` (the device's primary context) and `CUdevice` handles, for interop.
    pub fn as_raw(&self) -> (*mut c_void, i32) {
        (self.ctx, self.device)
    }

    /// Load a cubin and look up entry point `entry`.
    pub fn load_cubin(&self, cubin: &[u8], entry: &str) -> Result<CudaFunction<'_>, CudaError> {
        self.load_module(ModuleImage::Cubin(cubin), &[], entry)
    }

    /// Load a fatbin and look up entry point `entry`.
    pub fn load_fatbin(&self, fatbin: &[u8], entry: &str) -> Result<CudaFunction<'_>, CudaError> {
        self.load_module(ModuleImage::Fatbin(fatbin), &[], entry)
    }

    /// Load the module in the file at `path` (CuTe DSL's `load_cubin_module`): a cubin (an ELF image), a
    /// fatbin (by its magic number), or PTX text otherwise.
    pub fn load_module_file(&self, path: impl AsRef<Path>, entry: &str) -> Result<CudaFunction<'_>, CudaError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
        if bytes.starts_with(b"\x7fELF") {
            self.load_cubin(&bytes, entry)
        } else if bytes.starts_with(&0xBA55_ED50u32.to_le_bytes()) {
            self.load_fatbin(&bytes, entry)
        } else {
            let text = String::from_utf8(bytes)
                .map_err(|_| err(format!("{} is neither a cubin, a fatbin nor PTX text", path.display())))?;
            self.load_ptx(&text, entry)
        }
    }
}

impl CudaFunction<'_> {
    /// A function attribute (`cuFuncGetAttribute`).
    pub fn get_attribute(&self, attribute: FuncAttribute) -> Result<i32, CudaError> {
        self.attribute(attribute as i32)
    }

    /// Set a function attribute (`cuFuncSetAttribute`): the dynamic shared memory limit or
    /// the shared memory carveout, say. Allowed while a stream is capturing a graph.
    pub fn set_attribute(&self, attribute: FuncAttribute, value: i32) -> Result<(), CudaError> {
        self.ctx.make_current()?;
        // SAFETY: a loaded function.
        let r = unsafe { (self.ctx.api.func_set_attribute)(self.function, attribute as c_int, value) };
        check(&self.ctx.api, r, &format!("cuFuncSetAttribute({attribute:?}, {value})"))
    }

    /// The `CUfunction` and `CUmodule` handles; this keeps owning them.
    pub fn as_raw(&self) -> (*mut c_void, *mut c_void) {
        (self.function, self.module.module)
    }

    /// The address and size of module global `name` (`cuModuleGetGlobal`).
    pub(super) fn global_address(&self, name: &str) -> Result<(u64, usize), CudaError> {
        let c = CString::new(name).map_err(|_| err("a global's name contains a NUL byte"))?;
        let (mut ptr, mut bytes) = (0u64, 0usize);
        self.ctx.make_current()?;
        // SAFETY: valid out-pointers and a loaded module.
        let r = unsafe { (self.ctx.api.module_get_global)(&mut ptr, &mut bytes, self.module.module, c.as_ptr()) };
        check(&self.ctx.api, r, &format!("cuModuleGetGlobal(`{name}`)"))?;
        Ok((ptr, bytes))
    }
}
