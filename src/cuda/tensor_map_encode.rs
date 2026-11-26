//! Encoding tensor maps with the driver (`cuTensorMapEncodeTiled`, CUDA 12.0), and the
//! descriptors of a launch: hidden ones computed from the tensor parameter's device address
//! and the scalar arguments, and host-built ones re-encoded from the simulator's encoding.

use std::ffi::{c_int, c_uint, c_void};

use super::{CuResult, CudaContext, CudaError, check, err};
use crate::signature::{KernelSignature, LaunchValue};
use crate::tma::{MAX_RANK, TENSOR_MAP_BYTES, TmaDesc};

/// `cuTensorMapEncodeTiled(tensorMap, dataType, rank, globalAddress, globalDim,
/// globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill)`.
pub(super) type EncodeTiled = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    u32,
    *mut c_void,
    *const u64,
    *const u64,
    *const u32,
    *const u32,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
) -> CuResult;

/// `cuTensorMapEncodeIm2col(tensorMap, dataType, rank, globalAddress, globalDim,
/// globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel,
/// pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill)`, resolved for
/// roadmap part m7-tma-im2col.
pub(super) type EncodeIm2col = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    u32,
    *mut c_void,
    *const u64,
    *const u64,
    *const c_int,
    *const c_int,
    u32,
    u32,
    *const u32,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
) -> CuResult;

/// `CUDA_ERROR_NOT_SUPPORTED`, what the encoder returns below compute capability 9.0.
pub const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;
/// `CUDA_ERROR_INVALID_VALUE`, what the encoder returns for a field it rejects.
pub const CUDA_ERROR_INVALID_VALUE: i32 = 1;

/// A driver-encoded tensor map (`CUtensorMap`): 128 opaque bytes, 64-byte aligned.
#[repr(C, align(64))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct EncodedTensorMap(pub [u8; TENSOR_MAP_BYTES]);

// `CUtensorMap` is `CU_TENSOR_MAP_NUM_QWORDS` (16) `cuuint64_t`, `alignas(64)` (cuda.h).
const _: () =
    assert!(std::mem::size_of::<EncodedTensorMap>() == 16 * 8 && std::mem::align_of::<EncodedTensorMap>() == 64);

impl std::fmt::Debug for EncodedTensorMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EncodedTensorMap({} bytes)", TENSOR_MAP_BYTES)
    }
}

impl CudaContext {
    /// Whether the driver has `cuTensorMapEncodeTiled` (CUDA 12.0 and newer).
    pub fn has_tensor_map_encode(&self) -> bool {
        self.api.tensor_map_encode_tiled.is_some()
    }

    /// Whether the driver has `cuTensorMapEncodeIm2col`.
    pub fn has_tensor_map_encode_im2col(&self) -> bool {
        self.api.tensor_map_encode_im2col.is_some()
    }

    /// Encode `desc` with the driver's `cuTensorMapEncodeTiled`, without checking it first:
    /// the driver's own verdict, for comparing with [`TmaDesc::validate`]. The address must
    /// be device memory of this context for the descriptor to be used.
    pub fn encode_tensor_map_unchecked(&self, desc: &TmaDesc) -> Result<EncodedTensorMap, CudaError> {
        if desc.im2col.is_some() {
            return self.encode_im2col_unchecked(desc);
        }
        let Some(encode) = self.api.tensor_map_encode_tiled else {
            return Err(err(format!(
                "the CUDA driver ({}) has no cuTensorMapEncodeTiled: tensor maps need CUDA 12.0 or newer",
                self.driver_version().map_or("version unknown".to_string(), |v| format!("version {v}"))
            )));
        };
        self.make_current()?;
        let mut out = EncodedTensorMap([0; TENSOR_MAP_BYTES]);
        let r = desc.rank;
        // The driver reads `rank` entries of each array (`rank - 1` strides): pad copies with
        // zeros so that an out-of-range rank (here to ask for the driver's verdict on it)
        // cannot make it read past them.
        let n = (r as usize).max(MAX_RANK);
        let pad64 = |a: &[u64]| -> Vec<u64> { a.iter().copied().chain(std::iter::repeat(0)).take(n).collect() };
        let pad32 = |a: &[u32]| -> Vec<u32> { a.iter().copied().chain(std::iter::repeat(0)).take(n).collect() };
        let (dims, strides) = (pad64(&desc.global_dims), pad64(&desc.global_strides));
        let (boxes, elem_strides) = (pad32(&desc.box_dims), pad32(&desc.element_strides));
        // SAFETY: `out` is 128 bytes, 64-byte aligned (a `CUtensorMap`); the arrays hold at
        // least `rank` entries and outlive the call.
        let result = unsafe {
            encode(
                out.0.as_mut_ptr().cast(),
                desc.data_type.code(),
                r,
                desc.global_address as *mut c_void,
                dims.as_ptr(),
                strides.as_ptr(),
                boxes.as_ptr(),
                elem_strides.as_ptr(),
                desc.interleave.code(),
                desc.swizzle.code(),
                desc.l2_promotion.code(),
                desc.oob_fill.code(),
            )
        };
        if result == CUDA_ERROR_NOT_SUPPORTED && self.compute_capability().0 < 9 {
            let (major, minor) = self.compute_capability();
            let base = check(&self.api, result, "cuTensorMapEncodeTiled").unwrap_err();
            return Err(CudaError {
                message: format!(
                    "{}: tensor maps need compute capability 9.0 or newer, and {} is sm_{major}{minor} (the driver \
                     checks some fields first: a zero extent, box or element stride, or a rank outside 1 to 5, is \
                     CUDA_ERROR_INVALID_VALUE even there)",
                    base.message,
                    self.name()
                ),
                code: base.code,
            });
        }
        // The 4- and 6-bit encodings came with sm_100: an H100's driver refuses the valid ones too.
        if result == CUDA_ERROR_INVALID_VALUE
            && desc.data_type.group_slot_bytes().is_some()
            && self.compute_capability().0 < 10
        {
            let (major, minor) = self.compute_capability();
            let base = check(&self.api, result, "cuTensorMapEncodeTiled").unwrap_err();
            return Err(CudaError {
                message: format!(
                    "{}: the 4- and 6-bit tensor-map types ({}) need compute capability 10.0 or newer, and {} is \
                     sm_{major}{minor}",
                    base.message,
                    desc.data_type,
                    self.name()
                ),
                code: base.code,
            });
        }
        check(&self.api, result, "cuTensorMapEncodeTiled")?;
        Ok(out)
    }

    /// [`encode_tensor_map_unchecked`](CudaContext::encode_tensor_map_unchecked) of an im2col
    /// descriptor, with `cuTensorMapEncodeIm2col`.
    fn encode_im2col_unchecked(&self, desc: &TmaDesc) -> Result<EncodedTensorMap, CudaError> {
        let Some(encode) = self.api.tensor_map_encode_im2col else {
            return Err(err(format!(
                "the CUDA driver ({}) has no cuTensorMapEncodeIm2col: tensor maps need CUDA 12.0 or newer",
                self.driver_version().map_or("version unknown".to_string(), |v| format!("version {v}"))
            )));
        };
        let p = desc.im2col.expect("an im2col descriptor");
        self.make_current()?;
        let mut out = EncodedTensorMap([0; TENSOR_MAP_BYTES]);
        let r = desc.rank;
        let n = (r as usize).max(MAX_RANK);
        let pad64 = |a: &[u64]| -> Vec<u64> { a.iter().copied().chain(std::iter::repeat(0)).take(n).collect() };
        let pad32 = |a: &[u32]| -> Vec<u32> { a.iter().copied().chain(std::iter::repeat(0)).take(n).collect() };
        let (dims, strides) = (pad64(&desc.global_dims), pad64(&desc.global_strides));
        let elem_strides = pad32(&desc.element_strides);
        let lower: Vec<c_int> = p.lower_corner.to_vec();
        let upper: Vec<c_int> = p.upper_corner.to_vec();
        // SAFETY: `out` is a 64-byte-aligned `CUtensorMap`; the arrays hold at least
        // `rank` entries (the corners `rank - 2`, at most three) and outlive the call.
        let result = unsafe {
            encode(
                out.0.as_mut_ptr().cast(),
                desc.data_type.code(),
                r,
                desc.global_address as *mut c_void,
                dims.as_ptr(),
                strides.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                p.channels_per_pixel,
                p.pixels_per_column,
                elem_strides.as_ptr(),
                desc.interleave.code(),
                desc.swizzle.code(),
                desc.l2_promotion.code(),
                desc.oob_fill.code(),
            )
        };
        check(&self.api, result, "cuTensorMapEncodeIm2col")?;
        Ok(out)
    }

    /// Check `desc` ([`TmaDesc::validate`]) and encode it with `cuTensorMapEncodeTiled`.
    pub fn encode_tensor_map(&self, desc: &TmaDesc) -> Result<EncodedTensorMap, CudaError> {
        desc.validate().map_err(err)?;
        self.encode_tensor_map_unchecked(desc)
            .map_err(|e| CudaError { message: format!("{} (for {desc})", e.message), code: e.code })
    }
}

/// The driver-encoded descriptors of a launch, one per tensor-map parameter: hidden ones
/// computed from the launch values of the user's parameters (`values`, one per user
/// parameter), host-bound ones decoded from their bytes. Returns `(parameter, encoding)`.
pub(super) fn launch_tensor_maps(
    ctx: &CudaContext,
    sig: &KernelSignature,
    values: &[LaunchValue<'_>],
) -> Result<Vec<(usize, Box<EncodedTensorMap>)>, CudaError> {
    let mut out = Vec::new();
    for t in &sig.tensor_maps {
        let name = &sig.params[t.param].name;
        let desc = match &t.source {
            None => match values.get(t.param) {
                Some(LaunchValue::Bytes(b)) if TmaDesc::is_sim_encoding(b) => {
                    TmaDesc::from_sim_bytes(b).map_err(|e| err(format!("argument {} (`{name}`): {e}", t.param)))?
                }
                _ => {
                    return Err(err(format!(
                        "argument {} (`{name}`) is a tensor map: pass KernelArg::tensor_map of a host-built TensorMap",
                        t.param
                    )));
                }
            },
            Some(src) => {
                let address = match values.get(src.tensor_param) {
                    Some(LaunchValue::Pointer { address, .. }) => *address,
                    _ => return Err(err(format!("argument {} is not a device pointer", src.tensor_param))),
                };
                let scalar = |i: usize| match values.get(i) {
                    Some(LaunchValue::Scalar(a)) => a.scalar_bits().map(|(ty, bits)| {
                        let w = ty.bits() as u32;
                        if ty.is_signed() && w < 128 {
                            ((bits << (128 - w)) as i128 >> (128 - w)) as i64
                        } else {
                            bits as i64
                        }
                    }),
                    _ => None,
                };
                sig.hidden_tensor_map(t, address, &scalar).map_err(err)?
            }
        };
        let enc = ctx.encode_tensor_map(&desc).map_err(|e| CudaError {
            message: format!("argument {} (`{name}`): {}", t.param, e.message),
            code: e.code,
        })?;
        out.push((t.param, Box::new(enc)));
    }
    Ok(out)
}
