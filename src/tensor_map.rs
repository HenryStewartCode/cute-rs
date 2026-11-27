//! Host-built tensor maps (CuTe DSL's `cutlass.experimental.cuda` tensor-map helpers):
//! [`create_tensor_map_tiled`] from explicit dimensions and strides,
//! [`create_tensor_map_tiled_from_view`] from a strided view, and the [`TensorMap`] they
//! return, which a launch binds to a [`param_tensormap`](crate::param_tensormap) with
//! [`KernelArg::tensor_map`].
//!
//! A [`TensorMap`] holds the decoded descriptor ([`TmaDesc`]). The CUDA launcher encodes it
//! with the driver's `cuTensorMapEncodeTiled` when it launches (the driver needs a context,
//! and the descriptor's address must be device memory of it); the simulator passes its own
//! encoding, whose address must then be one of the simulator's (global buffers start at
//! [`SIM_GLOBAL_BASE`](crate::sim::SIM_GLOBAL_BASE), one per buffer parameter in order,
//! 256-byte aligned with a 64 KiB gap after each).
//!
//! As in CuTe DSL, dimensions are in TMA's column-major order (dimension 0 is the innermost,
//! contiguous one) and [`create_tensor_map_tiled`] takes the strides of dimensions 1 and up
//! in 16-byte units; [`TmaDesc`] and the driver take bytes.

use std::fmt;

use crate::ir::DType;
use crate::layout::{IndexExpr, Layout, StrideLeaf};
use crate::sim::KernelArg;
pub use crate::tma::{
    TensorMapDataType, TensorMapFloatOOBFill, TensorMapInterleave, TensorMapL2Promotion, TensorMapOptions,
    TensorMapSwizzle, TmaDesc,
};

/// The transfer format of a tensor map (CuTe DSL's `TensorMapDataFormat`): the element
/// type's own encoding, raw bytes, the flush-to-zero `f32` and TF32 forms that tensor
/// reductions use, and the FP4/FP6 forms of SM100:
/// `B4X16` packed, `B4X16_P64` and `B6X16_P32` padded to 16-byte slots.
// The variants are named after the CUDA enumerators.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TensorMapDataFormat {
    #[default]
    DEFAULT,
    BYTE,
    F32_FTZ,
    TF32_FTZ,
    B4X16,
    B4X16_P64,
    B6X16_P32,
}

impl fmt::Display for TensorMapDataFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// The encoding a tensor of `dtype` uses with `format` (the element type's own when
/// `None` or `DEFAULT`), or why there is none.
pub(crate) fn resolve_format(dtype: DType, format: Option<TensorMapDataFormat>) -> Result<TensorMapDataType, String> {
    use TensorMapDataFormat as F;
    let natural = || {
        TensorMapDataType::for_dtype(dtype).ok_or_else(|| {
            format!(
                "{} has no tensor-map encoding (TMA moves 8- to 64-bit integers and floats{})",
                dtype.cute_name(),
                if dtype.is_sub_byte() { "; of the sub-byte types, FP4 and FP6" } else { "" }
            )
        })
    };
    Ok(match format.unwrap_or_default() {
        F::DEFAULT => natural()?,
        F::BYTE if dtype.bits() == 8 && !dtype.is_sub_byte() => TensorMapDataType::uint8,
        F::F32_FTZ if dtype == DType::F32 => TensorMapDataType::float32_ftz,
        F::TF32_FTZ if matches!(dtype, DType::F32 | DType::TF32) => TensorMapDataType::tfloat32_ftz,
        F::B4X16 if matches!(dtype, DType::F4E2M1FN | DType::F4E2M1FNx2) => TensorMapDataType::f416u4_align8b,
        F::B4X16_P64 if matches!(dtype, DType::F4E2M1FN | DType::F4E2M1FNx2) => TensorMapDataType::f416u4_align16b,
        F::B6X16_P32 if matches!(dtype, DType::F6E2M3FN | DType::F6E3M2FN) => TensorMapDataType::f416u6_align16b,
        f => return Err(format!("the {f} tensor-map format does not apply to {}", dtype.cute_name())),
    })
}

/// A tensor map built on the host (CuTe DSL's `TensorMap`): the descriptor and what it was
/// built from, the element type and the transfer format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMap {
    desc: TmaDesc,
    dtype: DType,
    format: TensorMapDataFormat,
}

impl TensorMap {
    /// The descriptor.
    pub fn desc(&self) -> &TmaDesc {
        &self.desc
    }

    /// The element type (CuTe DSL's `element_type`, `dtype`).
    pub fn element_type(&self) -> DType {
        self.dtype
    }

    /// The transfer format.
    pub fn tma_format(&self) -> TensorMapDataFormat {
        self.format
    }

    /// The swizzle.
    pub fn swizzle(&self) -> TensorMapSwizzle {
        self.desc.swizzle
    }

    /// The box extents, TMA order.
    pub fn box_dims(&self) -> &[u32] {
        &self.desc.box_dims[..self.desc.rank as usize]
    }

    /// Logical elements in one box: the product of the box extents (CuTe DSL's
    /// `box_volume`).
    pub fn box_volume(&self) -> u64 {
        self.box_dims().iter().map(|&b| u64::from(b)).product()
    }

    /// The type that sizes one box in shared memory (CuTe DSL's `shared_storage_dtype`).
    pub fn shared_storage_dtype(&self) -> DType {
        self.dtype
    }

    /// Shared-memory bytes of one box (CuTe DSL's `shared_storage_bytes`): 16 per 16
    /// values for the padded FP4/FP6 formats.
    pub fn shared_storage_bytes(&self) -> u64 {
        self.desc.smem_box_bytes()
    }

    /// Global bytes one copy of a box completes, what `mbarrier_arrive_expect_tx` expects
    /// (CuTe DSL's `global_tx_bytes`): packed for the FP4/FP6 formats.
    pub fn global_tx_bytes(&self) -> u64 {
        self.desc.box_bytes()
    }
}

impl KernelArg {
    /// A tensor map for a [`param_tensormap`](crate::param_tensormap) parameter: 128 bytes
    /// holding the simulator's encoding of `map`, which the CUDA launcher re-encodes with
    /// `cuTensorMapEncodeTiled`.
    pub fn tensor_map(map: &TensorMap) -> KernelArg {
        KernelArg::by_value(map.desc.to_sim_bytes().to_vec())
    }

    /// [`KernelArg::tensor_map`] of a bare descriptor.
    pub fn tensor_map_desc(desc: &TmaDesc) -> KernelArg {
        KernelArg::by_value(desc.to_sim_bytes().to_vec())
    }
}

/// Build a tiled tensor map from explicit parameters (CuTe DSL's
/// `create_tensor_map_tiled`): `global_dims` in TMA order (dimension 0 innermost),
/// `global_strides` for dimensions 1 and up in 16-byte units, `box_dims` in TMA order.
/// Checks the rank, the lengths and everything `cuTensorMapEncodeTiled` checks
/// ([`check_tensor_map`](crate::tma::check_tensor_map)).
pub fn create_tensor_map_tiled(
    global_address: u64,
    dtype: DType,
    global_dims: &[u64],
    global_strides: &[u64],
    box_dims: &[u32],
    options: &TensorMapOptions,
) -> Result<TensorMap, String> {
    let rank = global_dims.len();
    if !(1..=crate::tma::MAX_RANK).contains(&rank) {
        return Err(format!("create_tensor_map_tiled: the tensor has rank {rank}, outside 1 to 5"));
    }
    if box_dims.len() != rank {
        return Err(format!("create_tensor_map_tiled: {} box extents for rank {rank}", box_dims.len()));
    }
    if global_strides.len() != rank - 1 {
        return Err(format!(
            "create_tensor_map_tiled: {} strides for rank {rank}: give one per dimension after the first ({})",
            global_strides.len(),
            rank - 1
        ));
    }
    if let Some(t) = &options.traversal_strides
        && t.len() != rank
    {
        return Err(format!("create_tensor_map_tiled: {} traversal strides for rank {rank}", t.len()));
    }
    let data_type = resolve_format(dtype, options.tma_format).map_err(|e| format!("create_tensor_map_tiled: {e}"))?;
    let mut d = TmaDesc::new(data_type, rank as u32);
    d.global_address = global_address;
    d.global_dims[..rank].copy_from_slice(global_dims);
    for (k, s) in global_strides.iter().enumerate() {
        d.global_strides[k] =
            s.checked_mul(16).ok_or_else(|| format!("create_tensor_map_tiled: stride {s} overflows"))?;
    }
    d.box_dims[..rank].copy_from_slice(box_dims);
    if let Some(t) = &options.traversal_strides {
        d.element_strides[..rank].copy_from_slice(t);
    }
    d.interleave = options.interleave;
    d.swizzle = options.swizzle;
    d.l2_promotion = options.l2_promotion;
    d.oob_fill = options.oob_fill;
    d.validate().map_err(|e| format!("create_tensor_map_tiled: {e}"))?;
    Ok(TensorMap { desc: d, dtype, format: options.tma_format.unwrap_or_default() })
}

/// A strided view of global memory for [`create_tensor_map_tiled_from_view`]: an address,
/// an element type, and a flat shape and stride in elements, in the tensor's own mode order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMapView {
    pub address: u64,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
}

impl TensorMapView {
    /// The view of a static `layout` (flattened) of `dtype` elements at `address`.
    pub fn of_layout(address: u64, dtype: DType, layout: &Layout) -> Result<TensorMapView, String> {
        let flat = layout.flatten();
        let mut shape = Vec::new();
        let mut stride = Vec::new();
        for (e, s) in flat.shape().flat_leaves().into_iter().zip(flat.stride().flat_leaves()) {
            let (Some(e), StrideLeaf::Int(s)) = (e.as_static(), s) else {
                return Err(format!("TensorMapView: layout {layout} is not static with integer strides"));
            };
            let s = s.as_static().ok_or_else(|| format!("TensorMapView: layout {layout} has a runtime stride"))?;
            shape.push(e as u64);
            stride.push(s as u64);
        }
        Ok(TensorMapView { address, dtype, shape, stride })
    }
}

/// Build a tiled tensor map from a strided view (CuTe DSL's
/// `create_tensor_map_tiled_from_view`): the TMA dimensions are the view's modes in
/// increasing stride order (or `stride_order`, innermost first; the innermost must have
/// stride 1), and `box_dims` is in the view's own mode order. A row-major `(M, K)` view
/// gives TMA dimensions `(K, M)`.
pub fn create_tensor_map_tiled_from_view(
    view: &TensorMapView,
    box_dims: &[u32],
    stride_order: Option<&[usize]>,
    options: &TensorMapOptions,
) -> Result<TensorMap, String> {
    let fail = |m: String| format!("create_tensor_map_tiled_from_view: {m}");
    if view.shape.len() != view.stride.len() || box_dims.len() != view.shape.len() {
        return Err(fail(format!(
            "{} extents, {} strides and {} box extents: give one of each per mode",
            view.shape.len(),
            view.stride.len(),
            box_dims.len()
        )));
    }
    let shape: Vec<IndexExpr> = view.shape.iter().map(|&e| IndexExpr::Static(e as usize)).collect();
    let stride: Vec<StrideLeaf> = view.stride.iter().map(|&s| StrideLeaf::Int(IndexExpr::Static(s as usize))).collect();
    let modes = crate::tma::tma_modes(&shape, &stride, stride_order).map_err(fail)?;
    let elem = resolve_format(view.dtype, options.tma_format).map_err(fail)?.bytes();
    let dims: Vec<u64> = modes.iter().map(|(_, e, _)| e.as_static().unwrap_or(0) as u64).collect();
    let mut strides = Vec::new();
    for (i, _, s) in modes.iter().skip(1) {
        let bytes = s.as_static().unwrap_or(0) as u64 * elem;
        if !bytes.is_multiple_of(16) {
            return Err(fail(format!("mode {i}'s stride is {bytes} bytes, not a multiple of 16")));
        }
        strides.push(bytes / 16);
    }
    let boxes: Vec<u32> = modes.iter().map(|(i, _, _)| box_dims[*i]).collect();
    create_tensor_map_tiled(view.address, view.dtype, &dims, &strides, &boxes, options).map_err(fail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_from_explicit_parameters() {
        // A row-major (M, K) = (256, 128) f16 array: TMA dims (K, M), row stride 256 bytes
        // = 16 units.
        let opts = TensorMapOptions::new().swizzle(TensorMapSwizzle::s128b);
        let m = create_tensor_map_tiled(0x10000, DType::F16, &[128, 256], &[16], &[64, 128], &opts).unwrap();
        assert_eq!(m.desc().global_strides[0], 256);
        assert_eq!(m.box_volume(), 64 * 128);
        assert_eq!(m.shared_storage_bytes(), 64 * 128 * 2);
        assert_eq!(m.global_tx_bytes(), 64 * 128 * 2);
        assert_eq!(m.element_type(), DType::F16);
        assert_eq!(m.shared_storage_dtype(), DType::F16);
        let arg = KernelArg::tensor_map(&m);
        assert_eq!(TmaDesc::from_sim_bytes(arg.byte_data()).unwrap(), *m.desc());
        let e = create_tensor_map_tiled(0x10000, DType::F16, &[128, 256], &[16, 1], &[64, 128], &opts).unwrap_err();
        assert!(e.contains("2 strides for rank 2"), "{e}");
        let e = create_tensor_map_tiled(0x10000, DType::F16, &[1; 6], &[1; 5], &[1; 6], &opts).unwrap_err();
        assert!(e.contains("rank 6"), "{e}");
        // The swizzle span: 128 halves are 256 bytes, wider than 128.
        let e = create_tensor_map_tiled(0x10000, DType::F16, &[128, 256], &[16], &[128, 8], &opts).unwrap_err();
        assert!(e.contains("at most 128 bytes, got 256"), "{e}");
        // FP4: packed by default, padded with B4X16_P64.
        let m =
            create_tensor_map_tiled(0x10000, DType::F4E2M1FN, &[128], &[], &[32], &TensorMapOptions::new()).unwrap();
        assert_eq!(m.desc().data_type, TensorMapDataType::f416u4_align8b);
        assert_eq!((m.shared_storage_bytes(), m.global_tx_bytes()), (16, 16));
        let f = TensorMapOptions::new().tma_format(TensorMapDataFormat::B4X16_P64);
        let m = create_tensor_map_tiled(0x10000, DType::F4E2M1FN, &[256], &[], &[128], &f).unwrap();
        assert_eq!((m.shared_storage_bytes(), m.global_tx_bytes()), (128, 64));
        let e = create_tensor_map_tiled(0x10000, DType::U8, &[128], &[], &[128], &f).unwrap_err();
        assert!(e.contains("does not apply to Uint8"), "{e}");
        let e =
            create_tensor_map_tiled(0x10000, DType::Bool, &[128], &[], &[32], &TensorMapOptions::new()).unwrap_err();
        assert!(e.contains("no tensor-map encoding"), "{e}");
        let f = TensorMapOptions::new().tma_format(TensorMapDataFormat::F32_FTZ);
        let m = create_tensor_map_tiled(0x10000, DType::F32, &[64], &[], &[32], &f).unwrap();
        assert_eq!(m.desc().data_type, TensorMapDataType::float32_ftz);
    }

    #[test]
    fn tiled_from_a_view_sorts_modes_by_stride() {
        let view = TensorMapView::of_layout(0x2000, DType::F32, &Layout::row_major((64, 32))).unwrap();
        let m = create_tensor_map_tiled_from_view(&view, &[16, 8], None, &TensorMapOptions::new()).unwrap();
        assert_eq!(&m.desc().global_dims[..2], &[32, 64]);
        assert_eq!(m.desc().global_strides[0], 128);
        assert_eq!(m.box_dims(), &[8, 16]);
        let e =
            create_tensor_map_tiled_from_view(&view, &[16, 8], Some(&[0, 1]), &TensorMapOptions::new()).unwrap_err();
        assert!(e.contains("not 1"), "{e}");
        let odd = TensorMapView::of_layout(0x2000, DType::F32, &Layout::row_major((64, 30))).unwrap();
        let e = create_tensor_map_tiled_from_view(&odd, &[16, 6], None, &TensorMapOptions::new()).unwrap_err();
        assert!(e.contains("120 bytes, not a multiple of 16"), "{e}");
    }
}
