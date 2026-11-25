//! Device and page-locked host memory: allocation without upload, copies at offsets,
//! `memset`, stream-ordered allocation and copies (CuTe DSL's `cutlass.cuda.allocate`,
//! `deallocate`, `memcpy_h2d` and `memcpy_d2h`).
//!
//! Asynchronous copies read or write host memory after the call returns, so they are safe
//! only on page-locked memory that outlives them: [`PinnedBuffer`]s, which a [`Pending`]
//! keeps borrowed until the stream has finished with them. Copies from any other slice are
//! `unsafe` ([`CudaStream::upload_raw_async`]).

use std::cell::Cell;
use std::ffi::c_void;

use super::{CuDevicePtr, CudaContext, CudaError, CudaStream, DeviceBuffer, DevicePtr, check, err};
use crate::value::F16;

/// A plain value type device memory can hold and copy bit for bit: the integer and
/// floating-point primitives and [`F16`].
///
/// # Safety
///
/// Every bit pattern of the type's size must be a valid value, and the type must have no
/// padding.
pub unsafe trait DeviceRepr: Copy + Default + 'static {}

macro_rules! device_repr {
    ($($t:ty),*) => { $(unsafe impl DeviceRepr for $t {})* };
}
device_repr!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64, F16);

fn bytes_of<T: DeviceRepr>(len: usize) -> Result<usize, CudaError> {
    len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| err(format!("{len} elements overflow a size")))
}

/// Check a host-side access of `bytes` bytes at `byte_offset` of a raw pointer: not null,
/// aligned for `T` and inside the bytes behind it when those are known. Returns the address.
fn raw_range<T: DeviceRepr>(ptr: &DevicePtr, byte_offset: usize, bytes: usize, what: &str) -> Result<u64, CudaError> {
    if ptr.is_null() {
        return Err(err(format!("{what} through a null device pointer")));
    }
    let address = ptr.address().checked_add(byte_offset as u64).ok_or_else(|| {
        err(format!("{what} at byte {byte_offset} of device address {:#x} overflows the address space", ptr.address()))
    })?;
    let align = std::mem::align_of::<T>() as u64;
    if !address.is_multiple_of(align) {
        return Err(err(format!("{what} at device address {address:#x}, which is not {align}-byte aligned")));
    }
    if let Some(limit) = ptr.bytes()
        && byte_offset.checked_add(bytes).is_none_or(|end| end > limit)
    {
        return Err(err(format!(
            "{what} of {bytes} bytes at byte {byte_offset} of device address {:#x} is outside the {limit} bytes behind it",
            ptr.address()
        )));
    }
    Ok(address)
}

/// Name the address in a driver error from a raw access.
fn at_address(mut e: CudaError, what: &str, address: u64, bytes: usize) -> CudaError {
    e.message = format!("{what} of {bytes} bytes at device address {address:#x}: {}", e.message);
    e
}

impl CudaContext {
    /// Read `count` elements of `T` from the raw device address `ptr` plus `byte_offset`
    /// (`cuMemcpyDtoH`, on the legacy NULL stream), waiting for it. The pointer must be non-null,
    /// aligned for `T` and, when its byte count is known, the range must lie inside it. An
    /// address the driver cannot read returns an error naming it.
    ///
    /// # Safety
    ///
    /// The address must be device memory of this context (or memory it can reach) holding
    /// `count` elements, alive for the call. Nothing else may be writing it.
    pub unsafe fn read_raw<T: DeviceRepr>(
        &self,
        ptr: &DevicePtr,
        byte_offset: usize,
        count: usize,
    ) -> Result<Vec<T>, CudaError> {
        let bytes = bytes_of::<T>(count)?;
        let address = raw_range::<T>(ptr, byte_offset, bytes, "a raw read")?;
        let mut out = vec![T::default(); count];
        self.make_current()?;
        // SAFETY: `out` is `bytes` writable bytes; the caller vouches for the device range.
        let r = unsafe { (self.api.memcpy_dtoh)(out.as_mut_ptr().cast(), address, bytes) };
        check(&self.api, r, "cuMemcpyDtoH").map_err(|e| at_address(e, "a raw read", address, bytes))?;
        Ok(out)
    }

    /// Write `values` to the raw device address `ptr` plus `byte_offset` (`cuMemcpyHtoD`),
    /// waiting for it; the checks and the error are those of [`read_raw`](Self::read_raw).
    ///
    /// # Safety
    ///
    /// The address must be device memory of this context (or memory it can reach) with room
    /// for `values`, alive for the call, and not in use by running kernels.
    pub unsafe fn write_raw<T: DeviceRepr>(
        &self,
        ptr: &DevicePtr,
        byte_offset: usize,
        values: &[T],
    ) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(values);
        let address = raw_range::<T>(ptr, byte_offset, bytes, "a raw write")?;
        self.make_current()?;
        // SAFETY: `values` is `bytes` readable bytes; the caller vouches for the device range.
        let r = unsafe { (self.api.memcpy_htod)(address, values.as_ptr().cast(), bytes) };
        check(&self.api, r, "cuMemcpyHtoD").map_err(|e| at_address(e, "a raw write", address, bytes))?;
        self.settle_legacy_stream()
    }
}

impl CudaStream<'_> {
    /// Queue a copy from the raw device address `ptr` plus `byte_offset` into `dst`
    /// (`cuMemcpyDtoHAsync`); the checks are those of [`CudaContext::read_raw`].
    ///
    /// # Safety
    ///
    /// As [`CudaContext::read_raw`], and `dst` must stay alive and untouched until the stream
    /// has been synchronised.
    pub unsafe fn read_raw_async<T: DeviceRepr>(
        &self,
        ptr: &DevicePtr,
        byte_offset: usize,
        dst: &mut [T],
    ) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(dst);
        let address = raw_range::<T>(ptr, byte_offset, bytes, "a raw read")?;
        self.ctx.make_current()?;
        // SAFETY: the caller keeps `dst` alive and untouched and vouches for the device range.
        let r = unsafe { (self.ctx.api.memcpy_dtoh_async)(dst.as_mut_ptr().cast(), address, bytes, self.raw) };
        check(&self.ctx.api, r, "cuMemcpyDtoHAsync").map_err(|e| at_address(e, "a raw read", address, bytes))
    }

    /// Queue a copy of `src` to the raw device address `ptr` plus `byte_offset`
    /// (`cuMemcpyHtoDAsync`); the checks are those of [`CudaContext::read_raw`].
    ///
    /// # Safety
    ///
    /// As [`CudaContext::write_raw`], and `src` must stay alive and unchanged until the stream
    /// has been synchronised.
    pub unsafe fn write_raw_async<T: DeviceRepr>(
        &self,
        ptr: &DevicePtr,
        byte_offset: usize,
        src: &[T],
    ) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(src);
        let address = raw_range::<T>(ptr, byte_offset, bytes, "a raw write")?;
        self.ctx.make_current()?;
        // SAFETY: the caller keeps `src` alive and unchanged and vouches for the device range.
        let r = unsafe { (self.ctx.api.memcpy_htod_async)(address, src.as_ptr().cast(), bytes, self.raw) };
        check(&self.ctx.api, r, "cuMemcpyHtoDAsync").map_err(|e| at_address(e, "a raw write", address, bytes))
    }
}

impl CudaContext {
    /// Allocate `len` elements of `T` (`cuMemAlloc`) without copying anything to them: the
    /// contents are undefined until written ([`DeviceBuffer::memset`], a copy or a kernel).
    pub fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<DeviceBuffer<'_>, CudaError> {
        self.alloc_bytes(bytes_of::<T>(len)?)
    }

    /// Allocate `bytes` bytes of device memory (256-byte aligned).
    pub fn alloc_bytes(&self, bytes: usize) -> Result<DeviceBuffer<'_>, CudaError> {
        self.make_current()?;
        let mut ptr: CuDevicePtr = 0;
        // SAFETY: a valid out-pointer.
        check(&self.api, unsafe { (self.api.mem_alloc)(&mut ptr, bytes.max(1)) }, "cuMemAlloc")?;
        Ok(DeviceBuffer { ctx: self, ptr, bytes })
    }

    /// Allocate `values.len()` elements and copy `values` there.
    pub fn upload_slice<T: DeviceRepr>(&self, values: &[T]) -> Result<DeviceBuffer<'_>, CudaError> {
        let buffer = self.alloc::<T>(values.len())?;
        buffer.write(0, values)?;
        Ok(buffer)
    }

    /// Allocate `bytes` bytes in stream order (`cuMemAllocAsync`, from the device's default
    /// memory pool): the memory may be used by work queued on `stream` after this call. Needs
    /// a device with memory pools ([`DeviceAttribute::MemoryPoolsSupported`]); free it in
    /// stream order with [`DeviceBuffer::free_async`] (dropping it frees it with `cuMemFree`,
    /// which waits for the device).
    ///
    /// [`DeviceAttribute::MemoryPoolsSupported`]: super::DeviceAttribute::MemoryPoolsSupported
    pub fn alloc_async(&self, bytes: usize, stream: &CudaStream<'_>) -> Result<DeviceBuffer<'_>, CudaError> {
        let alloc = self.api.mem_alloc_async.ok_or_else(|| err("this CUDA driver has no cuMemAllocAsync"))?;
        if self.device_attribute(super::DeviceAttribute::MemoryPoolsSupported)? == 0 {
            return Err(err(format!(
                "{} does not support memory pools (CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED is 0), which \
                 cuMemAllocAsync needs: use alloc_bytes",
                self.name()
            )));
        }
        self.make_current()?;
        let mut ptr: CuDevicePtr = 0;
        // SAFETY: a valid out-pointer and a live stream of this context.
        check(&self.api, unsafe { alloc(&mut ptr, bytes.max(1), stream.raw) }, "cuMemAllocAsync")?;
        Ok(DeviceBuffer { ctx: self, ptr, bytes })
    }

    /// Allocate `len` elements of page-locked host memory (`cuMemAllocHost`), zeroed.
    pub fn alloc_pinned<T: DeviceRepr>(&self, len: usize) -> Result<PinnedBuffer<'_, T>, CudaError> {
        let bytes = bytes_of::<T>(len)?;
        self.make_current()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: a valid out-pointer.
        check(&self.api, unsafe { (self.api.mem_alloc_host)(&mut ptr, bytes.max(1)) }, "cuMemAllocHost")?;
        // SAFETY: `ptr` holds `bytes` writable bytes.
        unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, bytes) };
        Ok(PinnedBuffer { ctx: self, ptr: ptr.cast(), len, in_flight: Cell::new(false) })
    }

    /// Page-locked host memory holding a copy of `values`.
    pub fn pinned_from<T: DeviceRepr>(&self, values: &[T]) -> Result<PinnedBuffer<'_, T>, CudaError> {
        let mut pinned = self.alloc_pinned::<T>(values.len())?;
        pinned.copy_from_slice(values);
        Ok(pinned)
    }
}

impl<'ctx> DeviceBuffer<'ctx> {
    /// The buffer's size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The device address (`CUdeviceptr`), for interop with other CUDA libraries. The
    /// buffer still owns the memory.
    pub fn as_raw(&self) -> u64 {
        self.ptr
    }

    /// The context the buffer belongs to.
    pub fn context(&self) -> &'ctx CudaContext {
        self.ctx
    }

    fn range(&self, byte_offset: usize, bytes: usize, what: &str) -> Result<CuDevicePtr, CudaError> {
        match byte_offset.checked_add(bytes) {
            Some(end) if end <= self.bytes => Ok(self.ptr + byte_offset as u64),
            _ => Err(err(format!(
                "{what} of {bytes} bytes at byte {byte_offset} is outside the buffer of {} bytes",
                self.bytes
            ))),
        }
    }

    /// Copy `values` to byte `byte_offset` of the buffer (`cuMemcpyHtoD`), waiting for it.
    pub fn write<T: DeviceRepr>(&self, byte_offset: usize, values: &[T]) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(values);
        let dst = self.range(byte_offset, bytes, "a write")?;
        self.ctx.make_current()?;
        // SAFETY: `values` is `bytes` readable bytes; the range is inside the allocation.
        check(
            &self.ctx.api,
            unsafe { (self.ctx.api.memcpy_htod)(dst, values.as_ptr().cast(), bytes) },
            "cuMemcpyHtoD",
        )?;
        self.ctx.settle_legacy_stream()
    }

    /// Copy bytes from byte `byte_offset` of the buffer into `out` (`cuMemcpyDtoH`). The copy
    /// runs on the legacy NULL stream, which does not wait for non-blocking streams:
    /// synchronise the stream whose work writes the buffer first.
    pub fn read<T: DeviceRepr>(&self, byte_offset: usize, out: &mut [T]) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(out);
        let src = self.range(byte_offset, bytes, "a read")?;
        self.ctx.make_current()?;
        // SAFETY: `out` is `bytes` writable bytes of a type every bit pattern is valid for.
        check(&self.ctx.api, unsafe { (self.ctx.api.memcpy_dtoh)(out.as_mut_ptr().cast(), src, bytes) }, "cuMemcpyDtoH")
    }

    /// The whole buffer as elements of `T` (its bytes must be a whole number of them).
    pub fn to_vec<T: DeviceRepr>(&self) -> Result<Vec<T>, CudaError> {
        let size = std::mem::size_of::<T>();
        if !self.bytes.is_multiple_of(size) {
            return Err(err(format!("a buffer of {} bytes is no whole number of {size}-byte elements", self.bytes)));
        }
        let mut out = vec![T::default(); self.bytes / size];
        self.read(0, &mut out)?;
        Ok(out)
    }

    /// Copy `bytes` bytes from byte `src_offset` of `src` to byte `dst_offset` of this
    /// buffer (`cuMemcpyDtoD`), waiting for it.
    pub fn copy_from(
        &self,
        dst_offset: usize,
        src: &DeviceBuffer<'_>,
        src_offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        let dst = self.range(dst_offset, bytes, "a copy")?;
        let from = src.range(src_offset, bytes, "a copy")?;
        self.ctx.make_current()?;
        // SAFETY: both ranges are inside their allocations.
        check(&self.ctx.api, unsafe { (self.ctx.api.memcpy_dtod)(dst, from, bytes) }, "cuMemcpyDtoD")?;
        self.ctx.settle_legacy_stream()
    }

    /// Set every byte of the buffer to `value` (`cuMemsetD8`), waiting for it.
    pub fn memset(&self, value: u8) -> Result<(), CudaError> {
        self.ctx.make_current()?;
        // SAFETY: the range is the allocation.
        check(&self.ctx.api, unsafe { (self.ctx.api.memset_d8)(self.ptr, value, self.bytes) }, "cuMemsetD8")?;
        self.ctx.settle_legacy_stream()
    }

    /// Free the buffer in stream order (`cuMemFreeAsync`): work queued on `stream` before
    /// this call may still use it.
    pub fn free_async(self, stream: &CudaStream<'_>) -> Result<(), CudaError> {
        let free = self.ctx.api.mem_free_async.ok_or_else(|| err("this CUDA driver has no cuMemFreeAsync"))?;
        self.ctx.make_current()?;
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: the pointer was allocated by this context and is freed once (Drop does not run).
        check(&this.ctx.api, unsafe { free(this.ptr, stream.raw) }, "cuMemFreeAsync")
    }
}

/// Page-locked host memory (`cuMemAllocHost`) of `len` elements of `T`, which asynchronous
/// copies may read and write while the host goes on. It dereferences to a slice; freed with
/// `cuMemFreeHost` when dropped.
pub struct PinnedBuffer<'ctx, T: DeviceRepr> {
    ctx: &'ctx CudaContext,
    ptr: *mut T,
    len: usize,
    /// An asynchronous copy may still use the memory: set by the safe copies, cleared when
    /// their [`Pending`] has waited. Should a `Pending` be leaked (`mem::forget`), the next
    /// access waits for the whole context instead.
    in_flight: Cell<bool>,
}

impl<T: DeviceRepr> PinnedBuffer<'_, T> {
    fn settle(&self) {
        if self.in_flight.get() {
            let _ = self.ctx.synchronize();
            self.in_flight.set(false);
        }
    }
}

impl<T: DeviceRepr> std::ops::Deref for PinnedBuffer<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.settle();
        // SAFETY: `ptr` holds `len` initialised elements (zeroed at allocation).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: DeviceRepr> std::ops::DerefMut for PinnedBuffer<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.settle();
        // SAFETY: as above, and `&mut self` is the only access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: DeviceRepr> Drop for PinnedBuffer<'_, T> {
    fn drop(&mut self) {
        self.settle();
        let _ = self.ctx.make_current();
        // SAFETY: allocated by cuMemAllocHost and freed once.
        unsafe { (self.ctx.api.mem_free_host)(self.ptr.cast()) };
    }
}

/// Asynchronous copies queued on a stream, which keep the host memory they use borrowed
/// until [`wait`](Pending::wait) (or dropping this, which waits too) has synchronised the
/// stream.
#[must_use = "dropping a Pending waits for the stream; call wait() to see its errors"]
pub struct Pending<'s> {
    stream: &'s CudaStream<'s>,
    done: bool,
    /// The page-locked buffer's in-flight flag.
    flag: &'s Cell<bool>,
}

impl<'s> Pending<'s> {
    /// Wait for the stream (`cuStreamSynchronize`), releasing the borrowed host memory.
    pub fn wait(mut self) -> Result<(), CudaError> {
        self.done = true;
        let r = self.stream.synchronize();
        if r.is_ok() {
            self.flag.set(false);
        }
        r
    }
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if !self.done && self.stream.synchronize().is_ok() {
            self.flag.set(false);
        }
    }
}

impl<'ctx> CudaStream<'ctx> {
    /// Refuse page-locked memory of another context: a leaked [`Pending`] is covered by
    /// synchronising the buffer's own context, which must therefore be the stream's.
    fn same_context<T: DeviceRepr>(&self, pinned: &PinnedBuffer<'_, T>) -> Result<(), CudaError> {
        if self.ctx.ctx != pinned.ctx.ctx {
            return Err(err("an asynchronous copy between a stream and page-locked memory of another context: \
                 allocate the PinnedBuffer from the stream's context"));
        }
        Ok(())
    }

    /// Refuse a copy into or out of host memory on a capturing stream: the graph would
    /// touch the host memory again at every replay, after a [`Pending`] has released it.
    fn no_host_copy_in_capture(&self, what: &str) -> Result<(), CudaError> {
        if self.is_capturing()? {
            return Err(err(format!(
                "{what} on a stream that is capturing a graph: the graph would read or write the host memory \
                 at every replay, after the borrow has ended; copy before the capture, or use the unsafe raw \
                 copies and keep the memory alive while the graph exists"
            )));
        }
        Ok(())
    }

    /// Queue a copy of `src` to byte `byte_offset` of `dst` (`cuMemcpyHtoDAsync`). The
    /// returned [`Pending`] keeps `src` borrowed until the stream has finished.
    pub fn upload_async<'a, T: DeviceRepr>(
        &'a self,
        src: &'a PinnedBuffer<'_, T>,
        dst: &'a DeviceBuffer<'_>,
        byte_offset: usize,
    ) -> Result<Pending<'a>, CudaError> {
        self.same_context(src)?;
        self.no_host_copy_in_capture("an asynchronous upload")?;
        src.settle();
        // SAFETY: page-locked memory, borrowed until the stream is synchronised (and flagged
        // in flight, should the Pending be leaked).
        unsafe { self.upload_raw_async(src, dst, byte_offset)? };
        src.in_flight.set(true);
        Ok(Pending { stream: self, done: false, flag: &src.in_flight })
    }

    /// Queue a copy of byte `byte_offset` onwards of `src` into `dst`
    /// (`cuMemcpyDtoHAsync`). The returned [`Pending`] keeps `dst` borrowed (mutably) until
    /// the stream has finished.
    pub fn download_async<'a, T: DeviceRepr>(
        &'a self,
        src: &'a DeviceBuffer<'_>,
        byte_offset: usize,
        dst: &'a mut PinnedBuffer<'_, T>,
    ) -> Result<Pending<'a>, CudaError> {
        self.same_context(dst)?;
        self.no_host_copy_in_capture("an asynchronous download")?;
        // SAFETY: page-locked memory, borrowed until the stream is synchronised (and flagged
        // in flight, should the Pending be leaked).
        unsafe { self.download_raw_async(src, byte_offset, dst)? };
        dst.in_flight.set(true);
        let dst: &'a PinnedBuffer<'_, T> = dst;
        Ok(Pending { stream: self, done: false, flag: &dst.in_flight })
    }

    /// Queue a copy of `src` to byte `byte_offset` of `dst` (`cuMemcpyHtoDAsync`) from any
    /// host memory.
    ///
    /// # Safety
    ///
    /// `src` must stay alive and unchanged until the copy has run (the stream is
    /// synchronised), and for every replay of a graph that captured it. Pageable memory is
    /// staged by the driver; page-locked memory is read by the copy engine later.
    pub unsafe fn upload_raw_async<T: DeviceRepr>(
        &self,
        src: &[T],
        dst: &DeviceBuffer<'_>,
        byte_offset: usize,
    ) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(src);
        let to = dst.range(byte_offset, bytes, "an upload")?;
        self.ctx.make_current()?;
        // SAFETY: the caller keeps `src` alive; the range is inside the allocation.
        let r = unsafe { (self.ctx.api.memcpy_htod_async)(to, src.as_ptr().cast(), bytes, self.raw) };
        check(&self.ctx.api, r, "cuMemcpyHtoDAsync")
    }

    /// Queue a copy from byte `byte_offset` of `src` into `dst` (`cuMemcpyDtoHAsync`).
    ///
    /// # Safety
    ///
    /// `dst` must stay alive and must not be read or written until the copy has run.
    pub unsafe fn download_raw_async<T: DeviceRepr>(
        &self,
        src: &DeviceBuffer<'_>,
        byte_offset: usize,
        dst: &mut [T],
    ) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(dst);
        let from = src.range(byte_offset, bytes, "a download")?;
        self.ctx.make_current()?;
        // SAFETY: the caller keeps `dst` alive and untouched; the range is in the allocation.
        let r = unsafe { (self.ctx.api.memcpy_dtoh_async)(dst.as_mut_ptr().cast(), from, bytes, self.raw) };
        check(&self.ctx.api, r, "cuMemcpyDtoHAsync")
    }

    /// Queue a device-to-device copy of `bytes` bytes (`cuMemcpyDtoDAsync`); capturable.
    pub fn copy_async(
        &self,
        dst: &DeviceBuffer<'_>,
        dst_offset: usize,
        src: &DeviceBuffer<'_>,
        src_offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        let to = dst.range(dst_offset, bytes, "a copy")?;
        let from = src.range(src_offset, bytes, "a copy")?;
        self.ctx.make_current()?;
        // SAFETY: both ranges are inside their allocations.
        let r = unsafe { (self.ctx.api.memcpy_dtod_async)(to, from, bytes, self.raw) };
        check(&self.ctx.api, r, "cuMemcpyDtoDAsync")
    }

    /// Queue setting every byte of `buffer` to `value` (`cuMemsetD8Async`); capturable.
    pub fn memset_async(&self, buffer: &DeviceBuffer<'_>, value: u8) -> Result<(), CudaError> {
        self.ctx.make_current()?;
        // SAFETY: the range is the allocation.
        let r = unsafe { (self.ctx.api.memset_d8_async)(buffer.ptr, value, buffer.bytes, self.raw) };
        check(&self.ctx.api, r, "cuMemsetD8Async")
    }
}
