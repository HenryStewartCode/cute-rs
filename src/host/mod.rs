// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host tensors for tests and interop (CuTe DSL's `cutlass.torch` helpers, without torch):
//! seeded initial values ([`TensorInit`]), permuted and majorness-aware layouts
//! ([`create_and_permute_tensor`], [`matrix`]), a typed [`HostTensor`] that keeps a
//! [`Layout`] beside its data, the leading-dimension query ([`get_leading_dim`]), GEMM
//! operands ([`prepare_tensors_for_gemm`]), narrow-type fills through the device conversion
//! kernel ([`convert_cute_tensor`], [`cute_tensor_like`]) and raw-pointer views
//! ([`as_tensor`]). [`interop`] maps element types to DLPack and PyTorch names.
//!
//! # Differences from `cutlass.torch`
//!
//! * No torch (or other tensor crate) dependency and no `rand`: values come from a seeded
//!   SplitMix64 generator ([`Rng`](crate::utils::gemm::Rng)) with Box-Muller normals, so a
//!   seed gives the same tensor on every run and platform, **but not the values torch's
//!   generator gives** for the same seed. Every creator takes an explicit `seed`.
//! * `TensorInit::Skip` leaves the buffer zeroed (a Rust `Vec` is never uninitialised).
//! * Tensors are copies. Interop with Rust tensor crates goes through DLPack
//!   ([`from_dlpack`](crate::from_dlpack)); [`as_tensor`] copies from raw host memory.
//! * Logical element order is colexicographic (the first mode fastest), as everywhere in
//!   cute-rs.
//!
//! # Streams
//!
//! A launch takes the stream it queues on from the caller: pass a `CUstream` another library
//! owns (PyTorch's current stream, a cuBLAS handle's stream) as
//! [`CudaStream::from_raw`](crate::cuda::CudaStream::from_raw)`(&ctx, raw)` (`unsafe`: it must
//! be a live stream of the context and outlive the value; it is not destroyed on drop) and
//! give it to [`LaunchBuilder::stream`](crate::cuda::LaunchBuilder::stream). A null `raw` is
//! the legacy default stream ([`CudaContext::default_stream`](crate::cuda::CudaContext::default_stream)).
//! An optional `tch` feature that reads torch's current stream is future work.

pub mod interop;

#[cfg(feature = "cuda")]
mod device;

#[cfg(feature = "cuda")]
pub use device::DeviceTensor;
pub use interop::{describe_dtype, dl_data_type, dtype_from_dl, dtype_from_torch_name, torch_dtype_name};

use crate::ir::DType;
use crate::layout::{IntTuple, Layout};
use crate::numeric::{NarrowRounding, decode, encode, tf32_rna_bits};
use crate::nvgpu::OperandMajorMode;
use crate::params::HostTensor as ParamTensor;
use crate::sim::KernelArg;
use crate::testing::ConvertError;
use crate::utils::gemm::Rng;
use crate::value::Numeric;

// ---- codecs --------------------------------------------------------------------------

/// `x` as the raw bits of `dtype` (not for the packed views or 128-bit integers): floats
/// round to nearest even (E8M0 up) and the narrow formats (8 bits and fewer) saturate to
/// their largest finite value as PTX `.satfinite` does, Float16 and BFloat16 overflow to
/// infinity; integers truncate toward zero and saturate to their range (NaN is 0); a Boolean
/// is `x != 0`.
pub fn encode_f64(dtype: DType, x: f64) -> u64 {
    match dtype {
        DType::Bool => (x != 0.0) as u64,
        DType::F64 => x.to_bits(),
        DType::F32 => (x as f32).to_bits() as u64,
        DType::TF32 => tf32_rna_bits((x as f32).to_bits()) as u64,
        t if t.is_packed_view() || t.bits() > 64 => {
            panic!("cute: cannot encode a host value as {}", t.cute_name())
        }
        t => match t.float_format() {
            Some(f) => {
                let narrow = t.bits() <= 8;
                let rounding = if t == DType::F8E8M0FNU { NarrowRounding::Up } else { NarrowRounding::NearestEven };
                encode(f, x, rounding, narrow, false) as u64
            }
            None => {
                let w = t.bits() as u32;
                let (lo, hi): (i128, i128) =
                    if t.is_signed() { (-(1i128 << (w - 1)), (1i128 << (w - 1)) - 1) } else { (0, (1i128 << w) - 1) };
                let v = if x.is_nan() { 0 } else { (x.trunc().clamp(lo as f64, hi as f64)) as i128 };
                (v as u128 & ((1u128 << w) - 1)) as u64
            }
        },
    }
}

/// The value of the raw bits `bits` of `dtype` (every NaN encoding decodes to NaN).
pub fn decode_f64(dtype: DType, bits: u64) -> f64 {
    match dtype {
        DType::Bool => (bits & 1) as f64,
        DType::F64 => f64::from_bits(bits),
        DType::F32 | DType::TF32 => f32::from_bits(bits as u32) as f64,
        t if t.is_packed_view() || t.bits() > 64 => {
            panic!("cute: cannot decode a host value of {}", t.cute_name())
        }
        t => match t.float_format() {
            Some(f) => decode(f, bits as u32),
            None => {
                let w = t.bits() as u32;
                let v = bits & (u64::MAX >> (64 - w));
                if t.is_signed() { ((v << (64 - w)) as i64 >> (64 - w)) as f64 } else { v as f64 }
            }
        },
    }
}

/// `x` as a `T`.
pub fn from_f64<T: Numeric>(x: f64) -> T {
    T::from_bits(encode_f64(T::DTYPE, x))
}

/// `v` as an f64.
pub fn to_f64<T: Numeric>(v: T) -> f64 {
    decode_f64(T::DTYPE, v.raw_bits())
}

/// `bits` (raw element bits) packed as the bytes of a `dtype` buffer: sub-byte values packed
/// element 0 first in the low bits, others little-endian.
pub fn pack_elements(dtype: DType, bits: &[u64]) -> Vec<u8> {
    let w = dtype.storage_bits();
    let mut data = vec![0u8; dtype.storage_bytes(bits.len())];
    for (i, &x) in bits.iter().enumerate() {
        if dtype.is_sub_byte() {
            for b in 0..w {
                let at = i * w + b;
                data[at / 8] |= ((x >> b & 1) as u8) << (at % 8);
            }
        } else {
            let n = w / 8;
            data[i * n..(i + 1) * n].copy_from_slice(&x.to_le_bytes()[..n]);
        }
    }
    data
}

/// The raw bits of the first `n` elements of a `dtype` buffer's `bytes` (the inverse of
/// [`pack_elements`]).
pub fn unpack_elements(dtype: DType, bytes: &[u8], n: usize) -> Vec<u64> {
    let w = dtype.storage_bits();
    (0..n)
        .map(|i| {
            if dtype.is_sub_byte() {
                (0..w).fold(0u64, |acc, b| acc | ((bytes[(i * w + b) / 8] >> ((i * w + b) % 8) & 1) as u64) << b)
            } else {
                let k = w / 8;
                let mut le = [0u8; 8];
                le[..k.min(8)].copy_from_slice(&bytes[i * k..i * k + k.min(8)]);
                u64::from_le_bytes(le)
            }
        })
        .collect()
}

/// The bytes of a buffer argument as the device sees them.
pub fn arg_bytes(arg: &KernelArg) -> Vec<u8> {
    arg.to_bytes()
}

// ---- seeded values ---------------------------------------------------------------------

impl Rng {
    /// A uniform f64 in `[0, 1)` (53 random bits).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal (Box-Muller; one of the pair per call).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// How a tensor's values start (CuTe DSL's `TensorInitType` with its config classes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorInit {
    /// Leave the values zeroed.
    Skip,
    /// Every element is the value (`ScalarInitConfig`, default 0.0).
    Scalar(f64),
    /// Integers drawn uniformly from `min..max` (the upper bound excluded, as torch's
    /// `random_` excludes it), held as f32 (`RandomInitConfig`, default -2 and 2).
    Random { min: i64, max: i64 },
    /// `normal(mean, std) * scale` (`GaussianInitConfig`, default 0, 1 and 1).
    Gaussian { mean: f64, std: f64, scale: f64 },
}

impl TensorInit {
    /// `RandomInitConfig`'s defaults, `min_val=-2` and `max_val=2`.
    pub const RANDOM: TensorInit = TensorInit::Random { min: -2, max: 2 };
    /// `GaussianInitConfig`'s defaults.
    pub const GAUSSIAN: TensorInit = TensorInit::Gaussian { mean: 0.0, std: 1.0, scale: 1.0 };
    /// `ScalarInitConfig`'s default, 0.0.
    pub const ZERO: TensorInit = TensorInit::Scalar(0.0);

    /// `n` f32 values drawn from `rng`.
    pub fn draw(&self, n: usize, rng: &mut Rng) -> Vec<f32> {
        match *self {
            TensorInit::Skip => vec![0.0; n],
            TensorInit::Scalar(v) => vec![v as f32; n],
            TensorInit::Random { min, max } => {
                assert!(min < max, "cute: TensorInit::Random needs min < max, not {min}..{max}");
                let span = (max - min) as u64;
                (0..n).map(|_| (min + (rng.next_u64() % span) as i64) as f32).collect()
            }
            TensorInit::Gaussian { mean, std, scale } => {
                (0..n).map(|_| ((mean + std * rng.normal()) * scale) as f32).collect()
            }
        }
    }
}

// ---- HostTensor<T> ---------------------------------------------------------------------

/// A tensor on the host: its elements in memory order and the [`Layout`] mapping
/// coordinates to them, so a permuted (strided) view keeps its data in the order it was
/// created. Sub-byte elements take a whole `T` each here; [`to_kernel_arg`](Self::to_kernel_arg)
/// packs them.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor<T: Numeric> {
    data: Vec<T>,
    layout: Layout,
}

impl<T: Numeric> HostTensor<T> {
    /// `data` seen through the static `layout`, which must reach no element past the end.
    #[track_caller]
    pub fn new(data: Vec<T>, layout: Layout) -> HostTensor<T> {
        assert!(layout.is_static(), "cute: HostTensor: the layout {layout} of a host tensor must be static");
        let need = layout.cosize();
        assert!(
            data.len() >= need,
            "cute: HostTensor: the layout {layout} reaches {need} elements but the data has {}",
            data.len()
        );
        HostTensor { data, layout }
    }

    /// A tensor of `layout` with every element `value`.
    pub fn full(layout: Layout, value: T) -> HostTensor<T> {
        let n = layout.cosize();
        HostTensor::new(vec![value; n], layout)
    }

    /// A tensor of `layout` with every element zero.
    pub fn zeros(layout: Layout) -> HostTensor<T> {
        HostTensor::full(layout, T::from_bits(0))
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The elements in memory order.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    pub fn shape(&self) -> Vec<usize> {
        self.layout.static_shape().flat_leaves()
    }

    pub fn strides(&self) -> Vec<usize> {
        self.layout.static_stride().flat_leaves()
    }

    /// The number of elements.
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    fn check_coord(&self, coord: &[usize]) -> usize {
        let (shape, stride) = (self.shape(), self.strides());
        assert!(self.layout.depth() <= 1, "cute: HostTensor: {} is hierarchical", self.layout);
        assert_eq!(coord.len(), shape.len(), "cute: the coordinate {coord:?} does not fit the shape {shape:?}");
        for (c, s) in coord.iter().zip(&shape) {
            assert!(c < s, "cute: the coordinate {coord:?} is outside the shape {shape:?}");
        }
        coord.iter().zip(&stride).map(|(c, d)| c * d).sum()
    }

    /// The element at `coord` (one index per mode of a flat layout).
    #[track_caller]
    pub fn get(&self, coord: &[usize]) -> T {
        self.data[self.check_coord(coord)]
    }

    #[track_caller]
    pub fn set(&mut self, coord: &[usize], value: T) {
        let i = self.check_coord(coord);
        self.data[i] = value;
    }

    /// The memory index of every element in logical (colexicographic) order.
    pub fn offsets(&self) -> Vec<usize> {
        let (shape, stride) = (self.shape(), self.strides());
        (0..shape.iter().product::<usize>())
            .map(|mut i| {
                let mut off = 0;
                for (&s, &d) in shape.iter().zip(&stride) {
                    off += (i % s) * d;
                    i /= s;
                }
                off
            })
            .collect()
    }

    /// The values in logical (colexicographic) order as f64.
    pub fn to_f64(&self) -> Vec<f64> {
        self.offsets().into_iter().map(|o| to_f64(self.data[o])).collect()
    }

    /// The index of the mode with stride 1 ([`get_leading_dim`]).
    pub fn leading_dim(&self) -> Option<usize> {
        get_leading_dim(&self.shape(), &self.strides())
    }

    /// The memory as a kernel argument (sub-byte elements packed).
    pub fn to_kernel_arg(&self) -> KernelArg {
        KernelArg::buffer(&self.data)
    }

    /// The tensor as a launch passes it: the buffer with the layout, static until marked
    /// dynamic.
    pub fn to_param(&self) -> ParamTensor {
        ParamTensor::new(self.to_kernel_arg(), self.layout.clone())
    }

    /// A tensor of `layout` over the contents of a buffer argument (what a run left in it).
    #[track_caller]
    pub fn from_kernel_arg(arg: &KernelArg, layout: Layout) -> HostTensor<T> {
        HostTensor::new(arg.to_vec::<T>(), layout)
    }

    /// A tensor over the contents of a launch tensor, with its layout.
    pub fn from_param(t: &ParamTensor) -> HostTensor<T> {
        HostTensor::from_kernel_arg(t.data(), t.layout())
    }

    /// Every element converted to `U` on the host ([`encode_f64`]'s rounding), same layout.
    pub fn convert<U: Numeric>(&self) -> HostTensor<U> {
        HostTensor { data: self.data.iter().map(|&v| from_f64::<U>(to_f64(v))).collect(), layout: self.layout.clone() }
    }
}

/// The index of the mode with stride 1, none if there is none (CuTe DSL's
/// `get_leading_dim`). Where several modes have stride 1, one of extent above 1 wins; of
/// several, the first.
pub fn get_leading_dim(shape: &[usize], strides: &[usize]) -> Option<usize> {
    let ones = (0..strides.len()).filter(|&i| strides[i] == 1);
    ones.clone().find(|&i| shape.get(i).is_some_and(|&s| s > 1)).or_else(|| ones.clone().next())
}

fn layout_of(shape: &[usize], strides: &[usize]) -> Layout {
    Layout::new(IntTuple::from(shape.to_vec()), IntTuple::from(strides.to_vec()))
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// A tensor of `T` whose values start as `init` draws them (as f32, then converted to `T`),
/// created compact and row-major (the last mode fastest, as torch does) with `shape` and
/// then permuted: the result's mode `i` is mode `permute_order[i]` of the created tensor, a
/// strided view of the same data (CuTe DSL's `create_and_permute_torch_tensor`). Values are
/// drawn in memory order from `seed`.
#[track_caller]
pub fn create_and_permute_tensor<T: Numeric>(
    shape: &[usize],
    permute_order: Option<&[usize]>,
    init: TensorInit,
    seed: u64,
) -> HostTensor<T> {
    let base = row_major_strides(shape);
    let (shape, strides): (Vec<usize>, Vec<usize>) = match permute_order {
        None => (shape.to_vec(), base),
        Some(order) => {
            let mut seen = vec![false; shape.len()];
            assert!(
                order.len() == shape.len()
                    && order.iter().all(|&o| o < shape.len() && !std::mem::replace(&mut seen[o], true)),
                "cute: permute_order {order:?} is not a permutation of the {} modes",
                shape.len()
            );
            (order.iter().map(|&o| shape[o]).collect(), order.iter().map(|&o| base[o]).collect())
        }
    };
    let n: usize = shape.iter().product();
    let values = init.draw(n, &mut Rng::new(seed));
    let data = values.into_iter().map(|v| from_f64::<T>(v as f64)).collect();
    HostTensor::new(data, layout_of(&shape, &strides))
}

/// An `(mode0, mode1, l)` tensor whose stride-1 mode is mode 0 (`is_mode0_major`) or mode 1,
/// with the batch mode slowest (CuTe DSL's `matrix`): strides `(1, mode0, mode0 * mode1)` or
/// `(mode1, 1, mode0 * mode1)`.
#[track_caller]
pub fn matrix<T: Numeric>(
    l: usize,
    mode0: usize,
    mode1: usize,
    is_mode0_major: bool,
    init: TensorInit,
    seed: u64,
) -> HostTensor<T> {
    if is_mode0_major {
        create_and_permute_tensor(&[l, mode1, mode0], Some(&[2, 1, 0]), init, seed)
    } else {
        create_and_permute_tensor(&[l, mode0, mode1], Some(&[1, 2, 0]), init, seed)
    }
}

/// Whether `dtype` is filled through the device conversion kernel by [`convert_cute_tensor`]:
/// Int4, the FP8 and scale-factor formats, FP6 and FP4.
fn is_narrow(dtype: DType) -> bool {
    dtype == DType::I4 || (dtype.is_float() && dtype.bits() <= 8 && !dtype.is_packed_view())
}

/// Fill `dst` (a launch tensor of any element type) with the values of `f32_src`, which has
/// the same shape (CuTe DSL's `convert_cute_tensor`). Int4, FP8, E8M0, FP6 and FP4 go
/// through [`convert_sim`](crate::testing::convert_sim) (the leading-mode rules of
/// [`crate::testing::convert`] apply); every other type, and the NV E5M3 scale type (which the device
/// conversion lacks), is rounded on the host with [`encode_f64`]. `dst` may have other
/// strides than `f32_src`.
pub fn convert_cute_tensor(f32_src: &HostTensor<f32>, dst: &mut ParamTensor) -> Result<(), ConvertError> {
    let dtype = dst.element_type();
    let dst_layout = dst.layout();
    let dst_shape = dst_layout.static_shape().flat_leaves();
    if dst_shape != f32_src.shape() {
        return Err(ConvertError(format!(
            "convert_cute_tensor: the shapes differ: {:?} and {dst_shape:?}",
            f32_src.shape()
        )));
    }
    if is_narrow(dtype) && dtype != DType::F8E5M3FNU && crate::testing::CONVERT_DTYPES.contains(&dtype) {
        return crate::testing::convert_sim(&f32_src.to_param(), dst);
    }
    // Host path: encode each element at the destination's offset.
    let dst_view = HostTensor::<u8>::new(vec![0; dst_layout.cosize()], dst_layout);
    let mut bits = vec![0u64; dst_view.data().len()];
    for (o, v) in dst_view.offsets().into_iter().zip(f32_src.to_f64()) {
        bits[o] = encode_f64(dtype, v);
    }
    *dst.data_mut() = KernelArg::Bytes { elem: dtype, data: pack_elements(dtype, &bits) };
    Ok(())
}

/// A launch tensor of element type `dtype` holding the values of `data_ref` (CuTe DSL's
/// `cute_tensor_like`): a zeroed tensor of the same shape and strides filled by
/// [`convert_cute_tensor`], its leading dimension made dynamic when `is_dynamic_layout`,
/// promising `assumed_align` bytes. It owns its data copy ([`ParamTensor::data`]).
pub fn cute_tensor_like(
    data_ref: &HostTensor<f32>,
    dtype: DType,
    is_dynamic_layout: bool,
    assumed_align: Option<usize>,
) -> Result<ParamTensor, ConvertError> {
    let layout = data_ref.layout().clone();
    let zeros = KernelArg::Bytes { elem: dtype, data: vec![0; dtype.storage_bytes(layout.cosize())] };
    let mut t = ParamTensor::new(zeros, layout);
    if let Some(a) = assumed_align {
        t = t.with_assumed_align(a);
    }
    convert_cute_tensor(data_ref, &mut t)?;
    if is_dynamic_layout {
        let lead = data_ref.leading_dim().map(|d| d as i32);
        t = t.mark_layout_dynamic(lead).map_err(|e| ConvertError(e.to_string()))?;
    }
    Ok(t)
}

/// The operands of a GEMM `C = A B` of `(m, n, k, l)` on the host.
#[derive(Clone, Debug)]
pub struct GemmHostTensors {
    /// `(m, k, l)`, K-major (stride 1 along k), as launch tensors with the leading dimension dynamic.
    pub a: ParamTensor,
    /// `(n, k, l)`, K-major.
    pub b: ParamTensor,
    /// `(m, n, l)`, N-major, zero.
    pub c: ParamTensor,
    /// The operands' values in f32, rounded to their element types and widened back, in the
    /// operands' layouts: what a reference GEMM should read.
    pub a_ref: HostTensor<f32>,
    pub b_ref: HostTensor<f32>,
    pub c_ref: HostTensor<f32>,
}

/// Random A and B and a zero C for a GEMM of `mnkl = (m, n, k, l)` (CuTe DSL's
/// `prepare_tensors_for_gemm`): [`matrix`] operands drawn with [`TensorInit::RANDOM`] from
/// `seed`, `seed + 1`, filled into tensors of the given element types.
pub fn prepare_tensors_for_gemm(
    mnkl: (usize, usize, usize, usize),
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    seed: u64,
) -> Result<GemmHostTensors, ConvertError> {
    let (m, n, k, l) = mnkl;
    let make = |mode0, mode1, dtype: DType, init, seed| -> Result<(ParamTensor, HostTensor<f32>), ConvertError> {
        let f = matrix::<f32>(l, mode0, mode1, false, init, seed);
        let t = cute_tensor_like(&f, dtype, true, Some(16))?;
        let rounded = HostTensor::new(
            f.data().iter().map(|&v| decode_f64(dtype, encode_f64(dtype, v as f64)) as f32).collect(),
            f.layout().clone(),
        );
        Ok((t, rounded))
    };
    let (a, a_ref) = make(m, k, a_dtype, TensorInit::RANDOM, seed)?;
    let (b, b_ref) = make(n, k, b_dtype, TensorInit::RANDOM, seed.wrapping_add(1))?;
    let (c, c_ref) = make(m, n, c_dtype, TensorInit::ZERO, seed.wrapping_add(2))?;
    Ok(GemmHostTensors { a, b, c, a_ref, b_ref, c_ref })
}

/// The majorness of a GEMM operand `(mn, k, l)` a layout has: [`OperandMajorMode::K`] when
/// mode 1 has stride 1, `MN` when mode 0 does.
pub fn operand_major_mode(layout: &Layout) -> Option<OperandMajorMode> {
    match get_leading_dim(&layout.static_shape().flat_leaves(), &layout.static_stride().flat_leaves())? {
        0 => Some(OperandMajorMode::MN),
        1 => Some(OperandMajorMode::K),
        _ => None,
    }
}

/// A tensor of the `shape.iter().product()` elements at `ptr`, copied, row-major compact
/// (CuTe DSL's `as_tensor` over a raw pointer, which wraps rather than copies).
///
/// # Safety
///
/// `ptr` points at that many initialised, readable `T`s.
pub unsafe fn as_tensor<T: Numeric>(ptr: *const T, shape: &[usize]) -> HostTensor<T> {
    let n: usize = shape.iter().product();
    // SAFETY: the caller's contract.
    let data = if n == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec() };
    HostTensor::new(data, layout_of(shape, &row_major_strides(shape)))
}

/// [`as_tensor`] for a type known only at run time: `ptr` holds `shape.iter().product()`
/// elements of `dtype` (sub-byte ones packed), copied into a launch tensor, row-major compact.
///
/// # Safety
///
/// `ptr` points at `dtype.storage_bytes(n)` readable bytes.
pub unsafe fn as_tensor_dyn(ptr: *const u8, shape: &[usize], dtype: DType) -> ParamTensor {
    let n: usize = shape.iter().product();
    let bytes = dtype.storage_bytes(n);
    // SAFETY: the caller's contract.
    let data = if bytes == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(ptr, bytes) }.to_vec() };
    ParamTensor::new(KernelArg::Bytes { elem: dtype, data }, layout_of(shape, &row_major_strides(shape)))
}
