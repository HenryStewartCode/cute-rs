//! Linking PTX modules at load time, and multicast memory (roadmap lane
//! m11-extern-distributed).
//!
//! [`CudaContext::load_linked`] links several PTX modules with the driver's JIT linker
//! (`cuLinkCreate`, `cuLinkAddData` with `CU_JIT_INPUT_PTX`, `cuLinkComplete`) and loads the
//! result: a kernel whose PTX declares an `.extern .func` (a call of an
//! [`ExternFn`](crate::ffi::ExternFn) that is not inlined) runs with the
//! [`DeviceFunction`](crate::ffi::DeviceFunction) that defines it, without a CUDA toolkit.
//!
//! [`MulticastGroup`] makes multicast memory over the driver's virtual memory management
//! (`cuMulticastGetGranularity`, `cuMulticastCreate`, `cuMulticastAddDevice`, `cuMemCreate`,
//! `cuMulticastBindMem`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess`): a buffer on each
//! device, and on each device a unicast address of its own buffer and a multimem address of
//! them all. It needs devices with `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` (NVSwitch systems
//! of sm_90 and newer GPUs) and a driver of CUDA 12.1 or newer, and fails with a clear message
//! otherwise. The symbols are resolved when a group is made, not when a context is, so older
//! drivers keep working for everything else.

use std::ffi::{CStr, CString, c_int, c_uint, c_void};

use super::{
    CU_JIT_INPUT_PTX, CuDevicePtr, CudaContext, CudaError, CudaFunction, JitLogs, LinkState, ModuleImage, check,
    check_with_log, err,
};
use crate::backend::JitOption;
use crate::log::Level;

/// `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`.
const CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED: c_int = 132;
/// `CU_MEM_ALLOCATION_TYPE_PINNED`.
const CU_MEM_ALLOCATION_TYPE_PINNED: c_int = 1;
/// `CU_MEM_LOCATION_TYPE_DEVICE`.
const CU_MEM_LOCATION_TYPE_DEVICE: c_int = 1;
/// `CU_MEM_ACCESS_FLAGS_PROT_READWRITE`.
const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: c_int = 3;
/// `CU_MULTICAST_GRANULARITY_RECOMMENDED`.
const CU_MULTICAST_GRANULARITY_RECOMMENDED: c_int = 1;
/// `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED`.
const CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: c_int = 1;

impl CudaContext {
    /// Link PTX `modules` into one cubin for this device with the driver's linker. `options`
    /// are the JIT options the linker takes. A failure reports the linker's error log (an
    /// unresolved `.extern .func` names the symbol).
    pub fn link_ptx(&self, modules: &[&str], options: &[JitOption]) -> Result<Vec<u8>, CudaError> {
        if modules.is_empty() {
            return Err(err("link_ptx: no modules to link"));
        }
        let mut logs = JitLogs::new();
        let (mut codes, mut values) = logs.attach(options);
        crate::cute_log!(Level::Debug, "linking {} PTX modules with the driver's linker", modules.len());
        self.make_current()?;
        let mut state: LinkState = std::ptr::null_mut();
        // SAFETY: option arrays and log buffers outlive the calls; `state` is a valid out-pointer.
        let r = unsafe {
            (self.api.link_create)(codes.len() as c_uint, codes.as_mut_ptr(), values.as_mut_ptr(), &mut state)
        };
        check(&self.api, r, "cuLinkCreate")?;
        struct Destroy<'a>(&'a super::Api, LinkState);
        impl Drop for Destroy<'_> {
            fn drop(&mut self) {
                // SAFETY: the state came from cuLinkCreate and is destroyed once.
                unsafe { (self.0.link_destroy)(self.1) };
            }
        }
        let _destroy = Destroy(&self.api, state);
        for (i, ptx) in modules.iter().enumerate() {
            if ptx.as_bytes().contains(&0) {
                return Err(err(format!("PTX module {i} contains a NUL byte")));
            }
            let image: Vec<u8> = ptx.bytes().chain(std::iter::once(0)).collect();
            let label = CString::new(format!("module{i}.ptx")).expect("no NUL");
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
        }
        let mut cubin: *mut c_void = std::ptr::null_mut();
        let mut size = 0usize;
        // SAFETY: valid out-pointers; the cubin is owned by the link state, copied before it goes.
        let r = unsafe { (self.api.link_complete)(state, &mut cubin, &mut size) };
        check_with_log(&self.api, r, "cuLinkComplete", &logs.texts().1)?;
        // SAFETY: `cubin` points at `size` bytes owned by the link state.
        Ok(unsafe { std::slice::from_raw_parts(cubin.cast::<u8>(), size) }.to_vec())
    }

    /// Link PTX `modules` ([`link_ptx`](Self::link_ptx)) and look up entry point `entry` in
    /// the result: a kernel calling an `.extern .func` with the module defining it, such as a
    /// [`DeviceFunction`](crate::ffi::DeviceFunction)'s PTX.
    pub fn load_linked(&self, modules: &[&str], entry: &str) -> Result<CudaFunction<'_>, CudaError> {
        let cubin = self.link_ptx(modules, &[])?;
        self.load_module(ModuleImage::Cubin(&cubin), &[], entry)
    }

    /// Whether this device supports multicast objects (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`).
    pub fn multicast_supported(&self) -> Result<bool, CudaError> {
        let mut v = 0;
        // SAFETY: a valid out-pointer and device.
        let r =
            unsafe { (self.api.device_get_attribute)(&mut v, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, self.device) };
        check(&self.api, r, "cuDeviceGetAttribute(MULTICAST_SUPPORTED)")?;
        Ok(v != 0)
    }
}

/// `CUmemLocation`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MemLocation {
    kind: c_int,
    id: c_int,
}

/// `CUmemAllocationProp` (32 bytes on 64-bit Linux).
#[repr(C)]
struct MemAllocationProp {
    kind: c_int,
    requested_handle_types: c_int,
    location: MemLocation,
    win32_handle_meta_data: *mut c_void,
    compression_type: u8,
    gpu_direct_rdma_capable: u8,
    usage: u16,
    reserved: [u8; 4],
}

/// `CUmulticastObjectProp`.
#[repr(C)]
struct MulticastObjectProp {
    num_devices: c_uint,
    size: usize,
    handle_types: u64,
    flags: u64,
}

/// `CUmemAccessDesc`.
#[repr(C)]
struct MemAccessDesc {
    location: MemLocation,
    flags: c_int,
}

type Handle = u64;
type CuResult = c_int;

/// The virtual-memory and multicast entry points (CUDA 10.2 and 12.1).
struct VmmApi {
    mc_get_granularity: unsafe extern "C" fn(*mut usize, *const MulticastObjectProp, c_int) -> CuResult,
    mc_create: unsafe extern "C" fn(*mut Handle, *const MulticastObjectProp) -> CuResult,
    mc_add_device: unsafe extern "C" fn(Handle, c_int) -> CuResult,
    mc_bind_mem: unsafe extern "C" fn(Handle, usize, Handle, usize, usize, u64) -> CuResult,
    mc_unbind: unsafe extern "C" fn(Handle, c_int, usize, usize) -> CuResult,
    mem_get_granularity: unsafe extern "C" fn(*mut usize, *const MemAllocationProp, c_int) -> CuResult,
    mem_create: unsafe extern "C" fn(*mut Handle, usize, *const MemAllocationProp, u64) -> CuResult,
    mem_release: unsafe extern "C" fn(Handle) -> CuResult,
    address_reserve: unsafe extern "C" fn(*mut CuDevicePtr, usize, usize, CuDevicePtr, u64) -> CuResult,
    address_free: unsafe extern "C" fn(CuDevicePtr, usize) -> CuResult,
    mem_map: unsafe extern "C" fn(CuDevicePtr, usize, usize, Handle, u64) -> CuResult,
    mem_unmap: unsafe extern "C" fn(CuDevicePtr, usize) -> CuResult,
    mem_set_access: unsafe extern "C" fn(CuDevicePtr, usize, *const MemAccessDesc, usize) -> CuResult,
}

#[cfg(unix)]
fn load_vmm() -> Result<VmmApi, CudaError> {
    // SAFETY: dlopen with a valid NUL-terminated name; the driver is already loaded by the
    // contexts, so this only takes another reference.
    let handle = unsafe { super::dl::dlopen(c"libcuda.so.1".as_ptr(), super::dl::RTLD_NOW) };
    if handle.is_null() {
        return Err(err("could not load the CUDA driver library for multicast memory"));
    }
    let load = |name: &CStr| -> Result<*mut c_void, CudaError> {
        // SAFETY: a live handle and a NUL-terminated name.
        let p = unsafe { super::dl::dlsym(handle, name.as_ptr()) };
        if p.is_null() {
            Err(err(format!(
                "the CUDA driver does not export {name:?}: multicast memory needs a driver of CUDA 12.1 or newer"
            )))
        } else {
            Ok(p)
        }
    };
    fn func<F: Copy>(p: *mut c_void) -> F {
        assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<*mut c_void>());
        // SAFETY: `p` is the driver function whose signature the field declares (checked
        // against cuda.h of CUDA 12.8).
        unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) }
    }
    Ok(VmmApi {
        mc_get_granularity: func(load(c"cuMulticastGetGranularity")?),
        mc_create: func(load(c"cuMulticastCreate")?),
        mc_add_device: func(load(c"cuMulticastAddDevice")?),
        mc_bind_mem: func(load(c"cuMulticastBindMem")?),
        mc_unbind: func(load(c"cuMulticastUnbind")?),
        mem_get_granularity: func(load(c"cuMemGetAllocationGranularity")?),
        mem_create: func(load(c"cuMemCreate")?),
        mem_release: func(load(c"cuMemRelease")?),
        address_reserve: func(load(c"cuMemAddressReserve")?),
        address_free: func(load(c"cuMemAddressFree")?),
        mem_map: func(load(c"cuMemMap")?),
        mem_unmap: func(load(c"cuMemUnmap")?),
        mem_set_access: func(load(c"cuMemSetAccess")?),
    })
}

#[cfg(not(unix))]
fn load_vmm() -> Result<VmmApi, CudaError> {
    Err(err("multicast memory needs the CUDA driver on a Unix-like system"))
}

/// One device's part of a [`MulticastGroup`].
struct Member<'c> {
    ctx: &'c CudaContext,
    memory: Handle,
    unicast: CuDevicePtr,
    multicast: CuDevicePtr,
}

/// Multicast memory over several devices: `bytes` bytes on each, with on each device a
/// unicast address of its own buffer ([`unicast_ptr`](Self::unicast_ptr)) and a multimem
/// address of them all ([`multicast_ptr`](Self::multicast_ptr)), for a kernel's
/// [`param_tensor_multicast`](crate::utils::distributed::param_tensor_multicast) parameter.
/// Everything is unmapped and released when it is dropped.
pub struct MulticastGroup<'c> {
    api: VmmApi,
    handle: Handle,
    size: usize,
    /// The alignment of every mapping: the larger of the multicast and allocation
    /// granularities.
    align: usize,
    members: Vec<Member<'c>>,
}

impl<'c> MulticastGroup<'c> {
    /// Make multicast memory of at least `bytes` bytes over the devices of `contexts` (one
    /// context per device, in rank order). Fails, saying why, when a device lacks multicast
    /// support (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`, which single GPUs and GPUs before
    /// sm_90 lack) or the driver lacks the multicast API.
    pub fn new(contexts: &[&'c CudaContext], bytes: usize) -> Result<MulticastGroup<'c>, CudaError> {
        if contexts.is_empty() || bytes == 0 {
            return Err(err("MulticastGroup::new: needs at least one device and one byte"));
        }
        for ctx in contexts {
            if !ctx.multicast_supported()? {
                return Err(err(format!(
                    "{} (device {}) does not support multicast memory (CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED is 0): \
                     multimem needs NVLink-connected sm_90 or newer GPUs behind an NVSwitch",
                    ctx.name(),
                    ctx.ordinal()
                )));
            }
        }
        let api = load_vmm()?;
        let ctx0 = contexts[0];
        ctx0.make_current()?;
        let mut prop =
            MulticastObjectProp { num_devices: contexts.len() as c_uint, size: bytes, handle_types: 0, flags: 0 };
        let mut granularity = 0usize;
        // SAFETY: valid pointers to a local property struct and out-value.
        let r = unsafe { (api.mc_get_granularity)(&mut granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED) };
        check(&ctx0.api, r, "cuMulticastGetGranularity")?;
        let mut size = bytes.next_multiple_of(granularity.max(1));
        let mut align = granularity.max(1);
        for ctx in contexts {
            let p = alloc_prop(ctx.device);
            let mut g = 0usize;
            // SAFETY: as above.
            let r = unsafe { (api.mem_get_granularity)(&mut g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) };
            check(&ctx.api, r, "cuMemGetAllocationGranularity")?;
            size = size.next_multiple_of(g.max(1));
            align = align.max(g);
        }
        prop.size = size;
        let mut handle: Handle = 0;
        // SAFETY: valid pointers.
        check(&ctx0.api, unsafe { (api.mc_create)(&mut handle, &prop) }, "cuMulticastCreate")?;
        // From here on, dropping `group` releases whatever has been made.
        let mut group = MulticastGroup { api, handle, size, align, members: Vec::new() };
        for ctx in contexts {
            // SAFETY: a live multicast handle and a device.
            check(&ctx.api, unsafe { (group.api.mc_add_device)(handle, ctx.device) }, "cuMulticastAddDevice")?;
        }
        for ctx in contexts {
            ctx.make_current()?;
            let p = alloc_prop(ctx.device);
            let mut memory: Handle = 0;
            // SAFETY: valid pointers.
            check(&ctx.api, unsafe { (group.api.mem_create)(&mut memory, size, &p, 0) }, "cuMemCreate")?;
            // SAFETY: live handles; the whole allocation is bound at offset 0.
            let r = unsafe { (group.api.mc_bind_mem)(handle, 0, memory, 0, size, 0) };
            if let Err(e) = check(&ctx.api, r, "cuMulticastBindMem") {
                // SAFETY: the allocation is ours and unmapped.
                unsafe { (group.api.mem_release)(memory) };
                return Err(e);
            }
            // A member as soon as it is bound, so that a failure below unbinds and releases
            // it (Drop skips the mappings still 0).
            group.members.push(Member { ctx, memory, unicast: 0, multicast: 0 });
            let unicast = group.map(ctx, memory)?;
            group.members.last_mut().expect("pushed").unicast = unicast;
            let multicast = group.map(ctx, handle)?;
            group.members.last_mut().expect("pushed").multicast = multicast;
        }
        Ok(group)
    }

    /// Reserve an address range on `ctx`'s device, map `handle` there and make it read-write.
    fn map(&self, ctx: &CudaContext, handle: Handle) -> Result<CuDevicePtr, CudaError> {
        let mut ptr: CuDevicePtr = 0;
        // SAFETY: a valid out-pointer.
        let r = unsafe { (self.api.address_reserve)(&mut ptr, self.size, self.align, 0, 0) };
        check(&ctx.api, r, "cuMemAddressReserve")?;
        // SAFETY: a reserved range of `size` bytes and a live handle.
        let r = unsafe { (self.api.mem_map)(ptr, self.size, 0, handle, 0) };
        if let Err(e) = check(&ctx.api, r, "cuMemMap") {
            // SAFETY: the range was reserved above.
            unsafe { (self.api.address_free)(ptr, self.size) };
            return Err(e);
        }
        let access = MemAccessDesc {
            location: MemLocation { kind: CU_MEM_LOCATION_TYPE_DEVICE, id: ctx.device },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: a mapped range and one access descriptor.
        let r = unsafe { (self.api.mem_set_access)(ptr, self.size, &access, 1) };
        if let Err(e) = check(&ctx.api, r, "cuMemSetAccess") {
            // SAFETY: the range was mapped and reserved above.
            unsafe {
                (self.api.mem_unmap)(ptr, self.size);
                (self.api.address_free)(ptr, self.size);
            }
            return Err(e);
        }
        Ok(ptr)
    }

    /// The bytes on each device (rounded up to the granularity).
    pub fn size(&self) -> usize {
        self.size
    }

    /// The number of devices.
    pub fn devices(&self) -> usize {
        self.members.len()
    }

    /// Rank `rank`'s unicast address of its own buffer (for ordinary loads and stores, and
    /// for flags polled with the spin locks).
    pub fn unicast_ptr(&self, rank: usize) -> u64 {
        self.members[rank].unicast
    }

    /// Rank `rank`'s multimem address of every device's buffer (for the multimem
    /// instructions).
    pub fn multicast_ptr(&self, rank: usize) -> u64 {
        self.members[rank].multicast
    }
}

impl Drop for MulticastGroup<'_> {
    fn drop(&mut self) {
        for m in &self.members {
            let _ = m.ctx.make_current();
            // SAFETY: the ranges and handles were made by `new` and are released once.
            unsafe {
                for ptr in [m.unicast, m.multicast] {
                    if ptr != 0 {
                        (self.api.mem_unmap)(ptr, self.size);
                        (self.api.address_free)(ptr, self.size);
                    }
                }
                (self.api.mc_unbind)(self.handle, m.ctx.device, 0, self.size);
                (self.api.mem_release)(m.memory);
            }
        }
        // SAFETY: the multicast object is released once, after every member.
        unsafe { (self.api.mem_release)(self.handle) };
    }
}

/// Pinned device memory on `device`, with no export handle.
fn alloc_prop(device: c_int) -> MemAllocationProp {
    MemAllocationProp {
        kind: CU_MEM_ALLOCATION_TYPE_PINNED,
        requested_handle_types: 0,
        location: MemLocation { kind: CU_MEM_LOCATION_TYPE_DEVICE, id: device },
        win32_handle_meta_data: std::ptr::null_mut(),
        compression_type: 0,
        gpu_direct_rdma_capable: 0,
        usage: 0,
        reserved: [0; 4],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_struct_layouts() {
        // Checked against cuda.h of CUDA 12.8.
        assert_eq!(std::mem::size_of::<MemAllocationProp>(), 32);
        assert_eq!(std::mem::size_of::<MulticastObjectProp>(), 32);
        assert_eq!(std::mem::size_of::<MemAccessDesc>(), 12);
        assert_eq!(std::mem::offset_of!(MemAllocationProp, win32_handle_meta_data), 16);
    }
}
