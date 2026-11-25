// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device-side host tensors: a buffer with its element type and layout.

use crate::cuda::{CudaContext, CudaError, DeviceBuffer};
use crate::ir::DType;
use crate::layout::Layout;
use crate::params::HostTensor as ParamTensor;
use crate::sim::KernelArg;

use super::HostTensor;
use crate::value::Numeric;

/// A tensor in device memory: the buffer, its element type and the [`Layout`] over it, what
/// [`crate::testing::convert`] takes. The buffer is freed when this is dropped.
pub struct DeviceTensor<'c> {
    buffer: DeviceBuffer<'c>,
    dtype: DType,
    layout: Layout,
}

impl<'c> DeviceTensor<'c> {
    /// A tensor over a buffer that holds at least the layout's elements of `dtype` (a smaller
    /// buffer panics).
    #[track_caller]
    pub fn from_buffer(buffer: DeviceBuffer<'c>, dtype: DType, layout: Layout) -> DeviceTensor<'c> {
        let need = dtype.storage_bytes(layout.cosize());
        assert!(
            buffer.bytes() >= need,
            "cute: a {dtype} tensor of layout {layout} needs {need} bytes, but the buffer has {}",
            buffer.bytes()
        );
        DeviceTensor { buffer, dtype, layout }
    }

    /// Copy a launch tensor to the device.
    pub fn upload(ctx: &'c CudaContext, t: &ParamTensor) -> Result<DeviceTensor<'c>, CudaError> {
        let buffer = ctx.upload(t.data())?;
        Ok(DeviceTensor::from_buffer(buffer, t.element_type(), t.layout()))
    }

    /// Copy a typed host tensor to the device.
    pub fn upload_typed<T: Numeric>(ctx: &'c CudaContext, t: &HostTensor<T>) -> Result<DeviceTensor<'c>, CudaError> {
        DeviceTensor::upload(ctx, &t.to_param())
    }

    /// A zeroed tensor.
    pub fn zeros(ctx: &'c CudaContext, dtype: DType, layout: Layout) -> Result<DeviceTensor<'c>, CudaError> {
        let bytes = dtype.storage_bytes(layout.cosize()).max(1);
        let buffer = ctx.alloc_bytes(bytes)?;
        buffer.memset(0)?;
        Ok(DeviceTensor::from_buffer(buffer, dtype, layout))
    }

    /// Copy back to the host.
    pub fn download(&self) -> Result<ParamTensor, CudaError> {
        let mut data = vec![0u8; self.buffer.bytes()];
        self.buffer.read(0, &mut data)?;
        data.truncate(self.dtype.storage_bytes(self.layout.cosize()));
        Ok(ParamTensor::new(KernelArg::Bytes { elem: self.dtype, data }, self.layout.clone()))
    }

    /// Copy back to the host as a typed tensor (`T` must be the element type).
    pub fn download_typed<T: Numeric>(&self) -> Result<HostTensor<T>, CudaError> {
        assert_eq!(
            T::DTYPE,
            self.dtype,
            "cute: DeviceTensor::download_typed::<{}> of a {} tensor",
            T::DTYPE,
            self.dtype
        );
        Ok(HostTensor::from_param(&self.download()?))
    }

    pub fn buffer(&self) -> &DeviceBuffer<'c> {
        &self.buffer
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}
