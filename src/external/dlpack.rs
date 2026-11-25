//! The DLPack ABI (`dlpack.h`, version 1.3): the C structs a framework exports a tensor
//! as, written out in Rust with the field order, types and constants of the header, and
//! [`DlpackTensor`], which owns (or borrows) one.
//!
//! The struct layouts and numbers follow DLPack's `dlpack.h` (Apache-2.0, "Copyright (c)
//! 2017 - by Contributors"); THIRD-PARTY-NOTICES.md says so.

use std::ffi::c_void;
use std::fmt;
use std::rc::Rc;

use crate::ir::DType;

/// The DLPack version these definitions follow (`DLPACK_MAJOR_VERSION`).
pub const DLPACK_MAJOR_VERSION: u32 = 1;
/// `DLPACK_MINOR_VERSION`.
pub const DLPACK_MINOR_VERSION: u32 = 3;

/// `DLPACK_FLAG_BITMASK_READ_ONLY`: the consumer must not write the tensor.
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;
/// `DLPACK_FLAG_BITMASK_IS_COPIED`: the producer copied the data for this export.
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 1 << 1;
/// `DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED`: sub-byte elements each take a whole byte.
pub const DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED: u64 = 1 << 2;

/// `DLDeviceType` values (an `int32_t` enum in C).
pub mod device_type {
    pub const KDL_CPU: i32 = 1;
    pub const KDL_CUDA: i32 = 2;
    pub const KDL_CUDA_HOST: i32 = 3;
    pub const KDL_OPENCL: i32 = 4;
    pub const KDL_VULKAN: i32 = 7;
    pub const KDL_METAL: i32 = 8;
    pub const KDL_VPI: i32 = 9;
    pub const KDL_ROCM: i32 = 10;
    pub const KDL_ROCM_HOST: i32 = 11;
    pub const KDL_EXT_DEV: i32 = 12;
    pub const KDL_CUDA_MANAGED: i32 = 13;
    pub const KDL_ONE_API: i32 = 14;
    pub const KDL_WEB_GPU: i32 = 15;
    pub const KDL_HEXAGON: i32 = 16;
    pub const KDL_MAIA: i32 = 17;
    pub const KDL_TRN: i32 = 18;
}

/// `DLDataTypeCode` values.
pub mod type_code {
    pub const KDL_INT: u8 = 0;
    pub const KDL_UINT: u8 = 1;
    pub const KDL_FLOAT: u8 = 2;
    pub const KDL_OPAQUE_HANDLE: u8 = 3;
    pub const KDL_BFLOAT: u8 = 4;
    pub const KDL_COMPLEX: u8 = 5;
    pub const KDL_BOOL: u8 = 6;
    pub const KDL_FLOAT8_E3M4: u8 = 7;
    pub const KDL_FLOAT8_E4M3: u8 = 8;
    pub const KDL_FLOAT8_E4M3B11FNUZ: u8 = 9;
    pub const KDL_FLOAT8_E4M3FN: u8 = 10;
    pub const KDL_FLOAT8_E4M3FNUZ: u8 = 11;
    pub const KDL_FLOAT8_E5M2: u8 = 12;
    pub const KDL_FLOAT8_E5M2FNUZ: u8 = 13;
    pub const KDL_FLOAT8_E8M0FNU: u8 = 14;
    pub const KDL_FLOAT6_E2M3FN: u8 = 15;
    pub const KDL_FLOAT6_E3M2FN: u8 = 16;
    pub const KDL_FLOAT4_E2M1FN: u8 = 17;
}

/// `DLPackVersion`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}

/// `DLDevice`: the device type ([`device_type`]) and its ordinal.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

/// `DLDataType`: the type code ([`type_code`]), the bits of one lane and the lanes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

/// `DLTensor`: the data pointer, device, rank, element type, shape and strides (in
/// elements; null for compact row-major) and the byte offset of the first element.
#[repr(C)]
#[derive(Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

/// `DLManagedTensor` (unversioned): the tensor, the producer's context and its deleter.
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// `DLManagedTensorVersioned`: the version, the producer's context, the deleter, the flags
/// (`DLPACK_FLAG_BITMASK_*`) and the tensor.
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}

/// What a DLPack tensor holds, or why it cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DlpackError(pub String);

impl fmt::Display for DlpackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DLPack: {}", self.0)
    }
}

impl std::error::Error for DlpackError {}

pub(crate) fn dl_err<T>(m: impl Into<String>) -> Result<T, DlpackError> {
    Err(DlpackError(m.into()))
}

/// The cute-rs element type of a DLPack data type and how many logical elements one DLPack
/// element packs: 1, or 2 for `float4_e2m1fn_x2` and 4 for the packed FP6 forms. Lanes other
/// than 1 are refused except for those packed sub-byte forms.
pub fn dtype_of(t: DLDataType, flags: u64) -> Result<(DType, usize), DlpackError> {
    use type_code::*;
    let scalar = match (t.code, t.bits) {
        (KDL_INT, 4) => Some(DType::I4),
        (KDL_INT, 8) => Some(DType::I8),
        (KDL_INT, 16) => Some(DType::I16),
        (KDL_INT, 32) => Some(DType::I32),
        (KDL_INT, 64) => Some(DType::I64),
        (KDL_INT, 128) => Some(DType::I128),
        (KDL_UINT, 8) => Some(DType::U8),
        (KDL_UINT, 16) => Some(DType::U16),
        (KDL_UINT, 32) => Some(DType::U32),
        (KDL_UINT, 64) => Some(DType::U64),
        (KDL_UINT, 128) => Some(DType::U128),
        (KDL_FLOAT, 16) => Some(DType::F16),
        (KDL_FLOAT, 32) => Some(DType::F32),
        (KDL_FLOAT, 64) => Some(DType::F64),
        (KDL_BFLOAT, 16) => Some(DType::BF16),
        (KDL_BOOL, 8) => Some(DType::Bool),
        (KDL_FLOAT8_E4M3, 8) => Some(DType::F8E4M3),
        (KDL_FLOAT8_E4M3B11FNUZ, 8) => Some(DType::F8E4M3B11FNUZ),
        (KDL_FLOAT8_E4M3FN, 8) => Some(DType::F8E4M3FN),
        (KDL_FLOAT8_E5M2, 8) => Some(DType::F8E5M2),
        (KDL_FLOAT8_E8M0FNU, 8) => Some(DType::F8E8M0FNU),
        (KDL_FLOAT6_E2M3FN, 6) => Some(DType::F6E2M3FN),
        (KDL_FLOAT6_E3M2FN, 6) => Some(DType::F6E3M2FN),
        (KDL_FLOAT4_E2M1FN, 4) => Some(DType::F4E2M1FN),
        _ => None,
    };
    let Some(dtype) = scalar else {
        return dl_err(format!(
            "data type code {} with {} bits has no cute-rs element type (cute-rs takes the integer, unsigned, \
             float, bfloat, bool, float8 e4m3/e4m3fn/e4m3b11fnuz/e5m2/e8m0fnu, float6 and float4 codes)",
            t.code, t.bits
        ));
    };
    if dtype.is_sub_byte() && flags & DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED != 0 {
        return dl_err(format!(
            "{} elements padded to a byte each (DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED) are not supported: \
             cute-rs packs sub-byte elements",
            dtype.cute_name()
        ));
    }
    match (dtype, t.lanes) {
        (_, 1) => Ok((dtype, 1)),
        (DType::F4E2M1FN, 2) => Ok((dtype, 2)),
        (DType::F6E2M3FN | DType::F6E3M2FN, 4) => Ok((dtype, 4)),
        (_, lanes) => dl_err(format!(
            "{} lanes of {}: vector element types are not supported (only the packed float4_e2m1fn_x2 and \
             float6 x4 forms have lanes)",
            lanes,
            dtype.cute_name()
        )),
    }
}

/// Who owns the DLPack object.
enum Owner {
    Managed(*mut DLManagedTensor),
    Versioned(*mut DLManagedTensorVersioned),
    Borrowed(*const DLTensor),
}

struct Inner(Owner);

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: the constructors' contracts: a live managed tensor whose ownership was
        // passed to us, deleted exactly once (here, when the last clone goes).
        unsafe {
            match self.0 {
                Owner::Managed(p) => {
                    if let Some(d) = (*p).deleter {
                        d(p)
                    }
                }
                Owner::Versioned(p) => {
                    if let Some(d) = (*p).deleter {
                        d(p)
                    }
                }
                Owner::Borrowed(_) => {}
            }
        }
    }
}

/// A DLPack tensor: an owned `DLManagedTensor` or `DLManagedTensorVersioned`, whose deleter
/// runs exactly once, when the last clone is dropped (including on every error path of a
/// conversion), or a borrowed `DLTensor`, which is never deleted. Clones share the object.
///
/// This is the unsafe newtype other crates wrap their DLPack export in: see
/// [`DlpackSource`](crate::DlpackSource).
#[derive(Clone)]
pub struct DlpackTensor {
    inner: Rc<Inner>,
}

impl fmt::Debug for DlpackTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.dl_tensor();
        f.debug_struct("DlpackTensor")
            .field("device", &t.device)
            .field("dtype", &t.dtype)
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("byte_offset", &t.byte_offset)
            .field("flags", &self.flags())
            .finish()
    }
}

impl DlpackTensor {
    /// Take ownership of an unversioned managed tensor (a framework's `__dlpack__()` capsule
    /// of version 0): its deleter runs when this (and every clone) is dropped.
    ///
    /// # Safety
    ///
    /// `p` points at a live `DLManagedTensor` whose ownership passes to the result; its data,
    /// shape and strides stay valid until the deleter runs, and nothing else deletes it.
    pub unsafe fn from_managed(p: *mut DLManagedTensor) -> DlpackTensor {
        assert!(!p.is_null(), "cute: DlpackTensor::from_managed: a null pointer");
        DlpackTensor { inner: Rc::new(Inner(Owner::Managed(p))) }
    }

    /// Take ownership of a versioned managed tensor (DLPack 1.x, `__dlpack__(max_version=(1,
    /// x))`). A major version other than 1 is refused when the tensor is converted (and the
    /// deleter still runs).
    ///
    /// # Safety
    ///
    /// As [`from_managed`](Self::from_managed).
    pub unsafe fn from_versioned(p: *mut DLManagedTensorVersioned) -> DlpackTensor {
        assert!(!p.is_null(), "cute: DlpackTensor::from_versioned: a null pointer");
        DlpackTensor { inner: Rc::new(Inner(Owner::Versioned(p))) }
    }

    /// Borrow a bare `DLTensor`, which is never deleted.
    ///
    /// # Safety
    ///
    /// `t`, its data, shape and strides stay valid while the result or anything made from it
    /// (an [`ExternalTensor`](crate::ExternalTensor)) lives.
    pub unsafe fn borrowed(t: *const DLTensor) -> DlpackTensor {
        assert!(!t.is_null(), "cute: DlpackTensor::borrowed: a null pointer");
        DlpackTensor { inner: Rc::new(Inner(Owner::Borrowed(t))) }
    }

    /// The `DLTensor`.
    pub fn dl_tensor(&self) -> &DLTensor {
        // SAFETY: the constructors' contracts keep the object alive while `inner` is.
        unsafe {
            match self.inner.0 {
                Owner::Managed(p) => &(*p).dl_tensor,
                Owner::Versioned(p) => &(*p).dl_tensor,
                Owner::Borrowed(t) => &*t,
            }
        }
    }

    /// The version of a versioned tensor.
    pub fn version(&self) -> Option<DLPackVersion> {
        // SAFETY: as in `dl_tensor`.
        match self.inner.0 {
            Owner::Versioned(p) => Some(unsafe { (*p).version }),
            _ => None,
        }
    }

    /// The flags of a versioned tensor (0 for the others).
    pub fn flags(&self) -> u64 {
        // SAFETY: as in `dl_tensor`.
        match self.inner.0 {
            Owner::Versioned(p) => unsafe { (*p).flags },
            _ => 0,
        }
    }

    /// Whether the producer marked the tensor read-only.
    pub fn is_read_only(&self) -> bool {
        self.flags() & DLPACK_FLAG_BITMASK_READ_ONLY != 0
    }

    /// Whether it owns the object (and so will call its deleter).
    pub fn is_owned(&self) -> bool {
        !matches!(self.inner.0, Owner::Borrowed(_))
    }

    /// The shape.
    pub fn shape(&self) -> Vec<i64> {
        let t = self.dl_tensor();
        if t.ndim <= 0 || t.shape.is_null() {
            return Vec::new();
        }
        // SAFETY: a DLTensor's shape holds `ndim` entries.
        unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) }.to_vec()
    }

    /// The strides in elements, `None` when the tensor gives none (compact row-major).
    pub fn strides(&self) -> Option<Vec<i64>> {
        let t = self.dl_tensor();
        if t.strides.is_null() || t.ndim <= 0 {
            return None;
        }
        // SAFETY: non-null strides hold `ndim` entries.
        Some(unsafe { std::slice::from_raw_parts(t.strides, t.ndim as usize) }.to_vec())
    }
}
