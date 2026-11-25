//! DLPack managed tensors built in tests, over host memory or a raw device address, with
//! deleters that count their calls (and keep a copy of host data as it was when deleted).

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cute::DlpackTensor;
use cute::external::dlpack::{DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLTensor, type_code};

/// Counts a tensor's deleter calls and keeps its host data as it was when deleted.
#[derive(Clone)]
pub struct Counter(Arc<(AtomicUsize, Mutex<Vec<u8>>)>);

impl Counter {
    pub fn get(&self) -> usize {
        self.0.0.load(Ordering::SeqCst)
    }

    /// The host data at deletion, as f32s.
    pub fn data_f32(&self) -> Vec<f32> {
        self.0.1.lock().unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }
}

/// The memory of a test tensor.
pub enum Owned {
    /// Host bytes, copied into 8-byte-aligned storage.
    Bytes(Vec<u8>),
    /// A device (or other) address the tensor points at and does not own.
    Raw(u64),
}

impl Owned {
    pub fn f32(v: &[f32]) -> Owned {
        Owned::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }

    pub fn bytes(v: &[u8]) -> Owned {
        Owned::Bytes(v.to_vec())
    }
}

pub fn f32_type() -> DLDataType {
    DLDataType { code: type_code::KDL_FLOAT, bits: 32, lanes: 1 }
}

struct Ctx {
    storage: Vec<u64>,
    len: usize,
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    counter: Counter,
}

impl Ctx {
    fn finish(self) {
        // SAFETY: storage holds at least `len` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(self.storage.as_ptr().cast::<u8>(), self.len) };
        *self.counter.0.1.lock().unwrap() = bytes.to_vec();
        self.counter.0.0.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn delete_versioned(p: *mut DLManagedTensorVersioned) {
    // SAFETY: made by `managed` from boxes, deleted once.
    unsafe {
        let m = Box::from_raw(p);
        Box::from_raw(m.manager_ctx as *mut Ctx).finish();
    }
}

unsafe extern "C" fn delete_managed(p: *mut DLManagedTensor) {
    // SAFETY: as above.
    unsafe {
        let m = Box::from_raw(p);
        Box::from_raw(m.manager_ctx as *mut Ctx).finish();
    }
}

/// A managed tensor (versioned or not) over `data`.
#[allow(clippy::too_many_arguments)]
pub fn managed(
    data: Owned,
    dtype: DLDataType,
    shape: &[i64],
    strides: Option<&[i64]>,
    byte_offset: u64,
    flags: u64,
    device: DLDevice,
    versioned: bool,
) -> (DlpackTensor, Counter) {
    make(data, dtype, shape, strides, byte_offset, flags, device, versioned.then_some(1))
}

/// A versioned managed tensor claiming DLPack major version `major`.
pub fn managed_version(
    data: Owned,
    dtype: DLDataType,
    shape: &[i64],
    device: DLDevice,
    major: u32,
) -> (DlpackTensor, Counter) {
    make(data, dtype, shape, None, 0, 0, device, Some(major))
}

#[allow(clippy::too_many_arguments)]
fn make(
    data: Owned,
    dtype: DLDataType,
    shape: &[i64],
    strides: Option<&[i64]>,
    byte_offset: u64,
    flags: u64,
    device: DLDevice,
    version: Option<u32>,
) -> (DlpackTensor, Counter) {
    let counter = Counter(Arc::new((AtomicUsize::new(0), Mutex::new(Vec::new()))));
    let (storage, len, raw) = match data {
        Owned::Bytes(b) => {
            let mut storage = vec![0u64; b.len().div_ceil(8)];
            // SAFETY: storage holds at least b.len() bytes.
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), storage.as_mut_ptr().cast::<u8>(), b.len()) };
            (storage, b.len(), None)
        }
        Owned::Raw(a) => (Vec::new(), 0, Some(a)),
    };
    let mut ctx = Box::new(Ctx {
        storage,
        len,
        shape: shape.to_vec(),
        strides: strides.map(<[i64]>::to_vec),
        counter: counter.clone(),
    });
    let data_ptr = raw.map_or(ctx.storage.as_mut_ptr().cast(), |a| a as usize as *mut std::ffi::c_void);
    let dl_tensor = DLTensor {
        data: data_ptr,
        device,
        ndim: shape.len() as i32,
        dtype,
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut().map_or(std::ptr::null_mut(), |s| s.as_mut_ptr()),
        byte_offset,
    };
    let manager_ctx = Box::into_raw(ctx).cast();
    let t = if let Some(major) = version {
        let m = Box::new(DLManagedTensorVersioned {
            version: cute::external::dlpack::DLPackVersion { major, minor: 3 },
            manager_ctx,
            deleter: Some(delete_versioned),
            flags,
            dl_tensor,
        });
        // SAFETY: a live managed tensor whose ownership passes to the DlpackTensor.
        unsafe { DlpackTensor::from_versioned(Box::into_raw(m)) }
    } else {
        let m = Box::new(DLManagedTensor { dl_tensor, manager_ctx, deleter: Some(delete_managed) });
        // SAFETY: as above.
        unsafe { DlpackTensor::from_managed(Box::into_raw(m)) }
    };
    (t, counter)
}
