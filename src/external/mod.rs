//! External memory: pointer descriptors for raw device addresses (CuTe DSL's
//! `cute.runtime.make_ptr` and `nullptr`) and tensors from other frameworks through DLPack
//! (`from_dlpack`, and the implicit conversion of a framework tensor passed straight to a
//! kernel).
//!
//! [`from_dlpack`] wraps a [`DlpackTensor`] without copying: the layout is static, from the
//! tensor's shape and strides (compact row-major when it gives none), the element type comes
//! from its DLPack data type, the start is `data + byte_offset`, and the alignment assumed
//! defaults to the element's size. A packed `float4_e2m1fn_x2` tensor is seen with its
//! logical FP4 shape (`(128, 128)` packed is `(128, 256)`). The result, an
//! [`ExternalTensor`], supports CuTe DSL's `mark_layout_dynamic` and
//! `mark_compact_shape_dynamic`, element-type relabelling for narrow types, `data_ptr`,
//! `dynamic_shapes_mask`, `dynamic_strides_mask`, `leading_dim` and a cache key.
//!
//! A kernel takes one through a [`DlTensor`] parameter, which is a [`Tensor`] inside the
//! kernel and whose host value is any [`DlpackSource`]: an [`ExternalTensor`] keeps the
//! layout it was given, and a [`DlpackTensor`] passed as it is gets CuTe DSL's implicit
//! conversion, `from_dlpack(t).mark_layout_dynamic()`: every extent and stride dynamic except
//! a unit leading stride. CUDA memory (`kDLCUDA`) is passed to a CUDA launch by address;
//! host memory (`kDLCPU`, `kDLCUDAHost`) is copied in and written back, on the simulator or
//! the GPU. Another crate's tensor type becomes a host value by implementing
//! [`DlpackSource`] through its DLPack export:
//!
//! ```ignore
//! impl cute::DlpackSource for MyTensor {
//!     fn to_external(&self) -> Result<cute::ExternalTensor, cute::DlpackError> {
//!         // SAFETY: `export_dlpack` hands over a live DLManagedTensorVersioned.
//!         let t = unsafe { cute::DlpackTensor::from_versioned(self.export_dlpack()) };
//!         t.to_external()
//!     }
//! }
//! ```

pub mod dlpack;

use std::fmt;
use std::marker::PhantomData;

pub use dlpack::{
    DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLPackVersion, DLTensor, DlpackError, DlpackTensor,
};
use dlpack::{device_type, dl_err, dtype_of};

use crate::ir::DType;
use crate::layout::{IntTuple, Layout, Tree};
use crate::params::{
    ArgError, ArgPack, ExternalArg, HostTensor, KernelParam, SpecError, SpecLeaf, SpecTuple, TensorSpec,
    declare_tensor, pack_tensor, pack_tensor_with,
};
use crate::sim::KernelArg;
use crate::tensor::{AddressSpace, Tensor};
use crate::value::Numeric;

/// The description of a pointer argument (CuTe DSL's `_Pointer`): its element type, memory
/// space and the alignment assumed for its address. Its [`cache_key`](Self::cache_key) is
/// CuTe DSL's `__cache_key__`, `(dtype, addr_space, assumed_align)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PointerSpec {
    pub dtype: DType,
    pub mem_space: AddressSpace,
    pub assumed_align: usize,
}

impl PointerSpec {
    /// A spec of `dtype` in `mem_space`, aligned to `assumed_align` bytes (by default the
    /// element's size). Panics unless the alignment is a power of two.
    #[track_caller]
    pub fn new(dtype: DType, mem_space: AddressSpace, assumed_align: Option<usize>) -> PointerSpec {
        let align = assumed_align.unwrap_or(dtype.storage_bytes(1).max(1));
        assert!(align.is_power_of_two(), "cute: PointerSpec: an alignment of {align} bytes is not a power of two");
        PointerSpec { dtype, mem_space, assumed_align: align }
    }

    /// The key to cache compiled kernels on: the spec itself.
    pub fn cache_key(&self) -> PointerSpec {
        *self
    }

    /// Check `address` against the assumed alignment (CuTe DSL's `make_ptr` asserts it).
    pub fn check_address(&self, address: u64) -> Result<(), String> {
        if !address.is_multiple_of(self.assumed_align as u64) {
            return Err(format!(
                "the address {address:#x} is not a multiple of the pointer's assumed alignment of {} bytes",
                self.assumed_align
            ));
        }
        Ok(())
    }
}

/// The key of a [`TensorSpec`] ([`TensorSpec::cache_key`]): the element type, memory space
/// and alignment, and the shape and stride with each dynamic leaf as its width, divisibility
/// and the position of its [`SymInt`](crate::SymInt)'s first occurrence in the spec (so two
/// leaves that are one `SymInt` key differently from two separate ones, while specs built
/// the same way from fresh `SymInt`s key the same).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorSpecKey {
    dtype: DType,
    memspace: AddressSpace,
    assumed_align: usize,
    shape: Tree<KeyLeaf>,
    stride: Tree<KeyLeaf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum KeyLeaf {
    Static(usize),
    Dynamic { width: u32, divisibility: usize, first: usize },
}

impl TensorSpec {
    /// The key to cache compiled kernels on ([`TensorSpecKey`]; CuTe DSL's `__cache_key__`).
    /// It numbers the spec's own [`SymInt`](crate::SymInt)s only: for a kernel of several
    /// tensors that may share one, key them together with [`cache_keys`](Self::cache_keys).
    pub fn cache_key(&self) -> TensorSpecKey {
        self.key_with(&mut Vec::new())
    }

    /// The keys of several specs of one kernel, numbering their [`SymInt`](crate::SymInt)s
    /// together, so specs that share one (a GEMM's `M` in A and C) key differently from
    /// specs with separate ones.
    pub fn cache_keys(specs: &[&TensorSpec]) -> Vec<TensorSpecKey> {
        let mut seen = Vec::new();
        specs.iter().map(|s| s.key_with(&mut seen)).collect()
    }

    fn key_with(&self, seen: &mut Vec<u64>) -> TensorSpecKey {
        let mut key = |t: &SpecTuple| {
            t.map(&mut |l: &SpecLeaf| match l {
                SpecLeaf::Static(v) => KeyLeaf::Static(*v),
                SpecLeaf::Dynamic(s) => {
                    let first = seen.iter().position(|&id| id == s.id()).unwrap_or_else(|| {
                        seen.push(s.id());
                        seen.len() - 1
                    });
                    KeyLeaf::Dynamic { width: s.width(), divisibility: s.divisibility(), first }
                }
            })
        };
        let (shape, stride) = (key(self.shape()), key(self.stride()));
        TensorSpecKey {
            dtype: self.element_type(),
            memspace: self.memspace(),
            assumed_align: self.assumed_align(),
            shape,
            stride,
        }
    }
}

/// How [`from_dlpack`] reads a tensor (CuTe DSL's `from_dlpack(t, assumed_align=None,
/// use_32bit_stride=False, force_tf32=False)`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FromDlpack {
    /// The alignment of the start, in bytes; the element's size by default.
    pub assumed_align: Option<usize>,
    /// Dynamic strides made later are 32-bit (the layout must fit).
    pub use_32bit_stride: bool,
    /// A Float32 tensor is relabelled TFloat32.
    pub force_tf32: bool,
}

/// A tensor of another framework seen through DLPack ([`from_dlpack`]): its memory, its
/// static layout until marked dynamic, and the [`TensorSpec`] a kernel taking it is built
/// for. It keeps the DLPack object alive (its deleter runs when the last clone goes).
#[derive(Clone)]
pub struct ExternalTensor {
    source: DlpackTensor,
    /// The layout and spec, over an empty placeholder buffer.
    layout: HostTensor,
    /// `data + byte_offset`.
    address: u64,
    device: DLDevice,
    read_only: bool,
}

impl fmt::Debug for ExternalTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExternalTensor({} at {:#x} on {:?})", self.layout.spec(), self.address, self.device)
    }
}

/// The spec's text, CuTe DSL's: `Tensor<f32, (?,?), (?{i64},1)>`.
impl fmt::Display for ExternalTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.layout.spec().fmt(f)
    }
}

fn placeholder(dtype: DType) -> KernelArg {
    KernelArg::Bytes { elem: dtype, data: Vec::new() }
}

fn tuple(v: &[usize]) -> IntTuple {
    Tree::Branch(v.iter().map(|&x| Tree::Leaf(x)).collect())
}

/// Wrap a DLPack tensor without copying (CuTe DSL's `cute.runtime.from_dlpack`). Errors
/// (an unsupported element type or lanes, negative strides, a misaligned start, a DLPack
/// major version other than 1) drop `t`, which runs its deleter.
pub fn from_dlpack(t: DlpackTensor, opts: FromDlpack) -> Result<ExternalTensor, DlpackError> {
    if let Some(v) = t.version()
        && v.major != dlpack::DLPACK_MAJOR_VERSION
    {
        return dl_err(format!(
            "DLPack version {}.{} is not supported (cute-rs reads major version {})",
            v.major,
            v.minor,
            dlpack::DLPACK_MAJOR_VERSION
        ));
    }
    let dl = t.dl_tensor();
    let (mut dtype, pack) = dtype_of(dl.dtype, t.flags())?;
    let shape = t.shape();
    if shape.is_empty() {
        return dl_err("a 0-dimensional tensor has no layout to pass; reshape it to (1,)");
    }
    if let Some(e) = shape.iter().find(|&&e| e < 0) {
        return dl_err(format!("the shape {shape:?} has a negative extent {e}"));
    }
    let strides = match t.strides() {
        Some(s) => s,
        None => {
            let mut s = vec![0i64; shape.len()];
            let mut running = 1i64;
            for i in (0..shape.len()).rev() {
                s[i] = running;
                running = match running.checked_mul(shape[i].max(1)) {
                    Some(r) => r,
                    None => return dl_err(format!("the shape {shape:?} has more elements than fit in 64 bits")),
                };
            }
            s
        }
    };
    if let Some(d) = strides.iter().find(|&&d| d < 0) {
        return dl_err(format!(
            "the strides {strides:?} have a negative stride {d}, which cute-rs layouts cannot express"
        ));
    }
    let mut shape: Vec<usize> = shape.iter().map(|&e| e as usize).collect();
    let mut stride: Vec<usize> = strides.iter().map(|&d| d as usize).collect();
    if pack > 1 {
        // The packed mode is the one of unit stride (of size above 1 when there are several).
        let ones: Vec<usize> = (0..shape.len()).filter(|&i| stride[i] == 1).collect();
        let packed = match ones.iter().copied().filter(|&i| shape[i] > 1).collect::<Vec<_>>()[..] {
            [i] => i,
            [] if !ones.is_empty() => *ones.last().unwrap(),
            _ => {
                return dl_err(format!(
                    "a packed {} tensor needs exactly one mode of stride 1 to unpack, but its strides are {strides:?}",
                    dtype.cute_name()
                ));
            }
        };
        for i in 0..shape.len() {
            let unpacked = if i == packed { shape[i].checked_mul(pack) } else { stride[i].checked_mul(pack) };
            let Some(v) = unpacked else {
                return dl_err(format!("the shape {shape:?} and strides {strides:?} overflow when unpacked"));
            };
            if i == packed {
                shape[i] = v;
            } else {
                stride[i] = v;
            }
        }
    }
    // The span the layout reaches, which the copy-in, write-back and launch bounds rely on,
    // must be addressable.
    let span = if shape.contains(&0) {
        Some(0)
    } else {
        shape.iter().zip(&stride).try_fold(1usize, |acc, (&e, &d)| (e - 1).checked_mul(d)?.checked_add(acc))
    };
    if span.is_none_or(|s| s > isize::MAX as usize / 16) {
        return dl_err(format!(
            "the shape {shape:?} with strides {stride:?} reaches more memory than can be addressed"
        ));
    }
    if dl.data.is_null() && span != Some(0) {
        return dl_err("the tensor's data pointer is null");
    }
    if opts.force_tf32 && dtype == DType::F32 {
        dtype = DType::TF32;
    }
    let size = dtype.storage_bytes(1).max(1);
    let align = opts.assumed_align.unwrap_or(size);
    if !align.is_power_of_two() {
        return dl_err(format!("an assumed alignment of {align} bytes is not a power of two"));
    }
    let address = (dl.data as u64).wrapping_add(dl.byte_offset);
    if !address.is_multiple_of(align as u64) {
        return dl_err(format!(
            "the tensor starts at {address:#x} (data + byte_offset {}), which is not a multiple of the assumed \
             alignment of {align} bytes",
            dl.byte_offset
        ));
    }
    let mut layout = HostTensor::new(placeholder(dtype), Layout::new(tuple(&shape), tuple(&stride)))
        .with_32bit_stride(opts.use_32bit_stride);
    if opts.assumed_align.is_some() {
        layout = layout.with_assumed_align(align);
    }
    let (device, read_only) = (dl.device, t.is_read_only());
    Ok(ExternalTensor { source: t, layout, address, device, read_only })
}

impl ExternalTensor {
    /// The DLPack object it reads.
    pub fn source(&self) -> &DlpackTensor {
        &self.source
    }

    pub fn element_type(&self) -> DType {
        self.layout.element_type()
    }

    /// The logical shape (FP4 elements for a packed `float4_e2m1fn_x2` tensor).
    pub fn shape(&self) -> &IntTuple {
        self.layout.shape()
    }

    pub fn stride(&self) -> &IntTuple {
        self.layout.stride()
    }

    /// The spec a kernel taking it is built for.
    pub fn spec(&self) -> &TensorSpec {
        self.layout.spec()
    }

    /// The spec's layout: `(?,?):(?{i64},1)`.
    pub fn layout_string(&self) -> String {
        self.layout.layout_string()
    }

    /// The address of the first element (`data + byte_offset`; CuTe DSL's `data_ptr`).
    pub fn data_ptr(&self) -> u64 {
        self.address
    }

    pub fn device(&self) -> DLDevice {
        self.device
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Which modes have a dynamic extent (CuTe DSL's `dynamic_shapes_mask`).
    pub fn dynamic_shapes_mask(&self) -> Vec<bool> {
        self.spec().shape().leaves().iter().map(|l| l.as_sym().is_some()).collect()
    }

    /// Which modes have a dynamic stride (CuTe DSL's `dynamic_strides_mask`).
    pub fn dynamic_strides_mask(&self) -> Vec<bool> {
        self.spec().stride().leaves().iter().map(|l| l.as_sym().is_some()).collect()
    }

    /// The mode of unit stride (CuTe DSL's `leading_dim`): the one mode of stride 1, or the
    /// one of stride 1 and extent above 1 among several; `None` if there is none or it is
    /// ambiguous.
    pub fn leading_dim(&self) -> Option<usize> {
        let (shape, stride) = (self.shape().leaves(), self.stride().leaves());
        let ones: Vec<usize> = (0..stride.len()).filter(|&i| *stride[i] == 1).collect();
        match ones[..] {
            [i] => Some(i),
            _ => {
                let big: Vec<usize> = ones.into_iter().filter(|&i| *shape[i] > 1).collect();
                (big.len() == 1).then(|| big[0])
            }
        }
    }

    /// The bytes from the start to one past the last element the layout reaches.
    pub fn span_bytes(&self) -> usize {
        let (s, d) = (self.shape().leaves(), self.stride().leaves());
        if s.iter().any(|&&x| x == 0) {
            return 0;
        }
        let span = 1 + s.iter().zip(d).map(|(&&s, &d)| (s - 1) * d).sum::<usize>();
        self.element_type().storage_bytes(span)
    }

    /// The key to cache compiled kernels on (CuTe DSL's `__cache_key__`): the spec's.
    pub fn cache_key(&self) -> TensorSpecKey {
        self.spec().cache_key()
    }

    /// Make every mode dynamic except a unit leading stride (CuTe DSL's
    /// `mark_layout_dynamic(leading_dim=None)`; see [`HostTensor::mark_layout_dynamic`]).
    pub fn mark_layout_dynamic(mut self, leading_dim: Option<i32>) -> Result<ExternalTensor, SpecError> {
        self.layout = self.layout.mark_layout_dynamic(leading_dim)?;
        Ok(self)
    }

    /// Make one mode's extent dynamic in a compact layout (CuTe DSL's
    /// `mark_compact_shape_dynamic(mode, stride_order=None, divisibility=1)`; see
    /// [`HostTensor::mark_compact_shape_dynamic`]).
    pub fn mark_compact_shape_dynamic(
        mut self,
        mode: usize,
        stride_order: Option<&[usize]>,
        divisibility: usize,
    ) -> Result<ExternalTensor, SpecError> {
        self.layout = self.layout.mark_compact_shape_dynamic(mode, stride_order, divisibility)?;
        Ok(self)
    }

    /// The same memory read as `dtype` with the same shape and strides (CuTe DSL's
    /// `element_type` assignment, for narrow types a framework lacks: a `uint8` tensor read
    /// as Float4E2M1FN). The new type may not be wider than the old one.
    pub fn with_element_type(mut self, dtype: DType) -> Result<ExternalTensor, DlpackError> {
        let old = self.element_type();
        if dtype.storage_bits() > old.storage_bits() {
            return dl_err(format!(
                "relabelling {} elements as {}: the new type is wider ({} bits over {}), so the tensor would \
                 reach past its memory",
                old.cute_name(),
                dtype.cute_name(),
                dtype.storage_bits(),
                old.storage_bits()
            ));
        }
        self.layout = self.layout.relabel(dtype, placeholder(dtype));
        Ok(self)
    }

    /// Whether the memory is host memory the simulator (and a copy-in launch) can read.
    pub fn is_host(&self) -> bool {
        matches!(self.device.device_type, device_type::KDL_CPU | device_type::KDL_CUDA_HOST)
    }

    /// Whether it is CUDA device memory, passed to a launch by address.
    pub fn is_cuda(&self) -> bool {
        matches!(self.device.device_type, device_type::KDL_CUDA | device_type::KDL_CUDA_MANAGED)
    }

    fn device_name(&self) -> String {
        let name = match self.device.device_type {
            device_type::KDL_CPU => "kDLCPU",
            device_type::KDL_CUDA => "kDLCUDA",
            device_type::KDL_CUDA_HOST => "kDLCUDAHost",
            device_type::KDL_CUDA_MANAGED => "kDLCUDAManaged",
            device_type::KDL_ROCM => "kDLROCM",
            device_type::KDL_METAL => "kDLMetal",
            _ => "a device cute-rs does not run on",
        };
        format!("{name} (device type {}, id {})", self.device.device_type, self.device.device_id)
    }

    /// A host tensor holding a copy of the memory (host memory only): the span from the
    /// start, seen through the tensor's layout and spec.
    pub fn host_tensor(&self) -> Result<HostTensor, DlpackError> {
        if !self.is_host() {
            return dl_err(format!(
                "the tensor is in {}, not host memory (kDLCPU): the simulator and copy-in launches read host \
                 memory only",
                self.device_name()
            ));
        }
        let bytes = self.span_bytes();
        let data = if bytes == 0 {
            Vec::new()
        } else {
            // SAFETY: the DLPack contract: the tensor's elements, from data + byte_offset to the
            // last one the layout reaches, are readable host memory while `source` lives.
            unsafe { std::slice::from_raw_parts(self.address as usize as *const u8, bytes) }.to_vec()
        };
        let dtype = self.element_type();
        Ok(self.layout.clone().relabel(dtype, KernelArg::Bytes { elem: dtype, data }))
    }

    /// Write `data` (a buffer of the span's bytes, as [`host_tensor`](Self::host_tensor)
    /// made it and a run updated it) back into the host memory. A read-only tensor is left
    /// alone.
    pub fn write_back(&self, data: &KernelArg) {
        if self.read_only || !self.is_host() {
            return;
        }
        if let KernelArg::Bytes { data, .. } = data {
            let n = data.len().min(self.span_bytes());
            // SAFETY: as in `host_tensor`, and the producer did not mark the memory read-only.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.address as usize as *mut u8, n) };
        }
    }

    /// Pack this tensor for the parameter `name` of spec `spec`: host memory as a copy,
    /// device memory by address.
    pub(crate) fn pack(&self, name: &str, spec: &TensorSpec, out: &mut ArgPack) -> Result<(), ArgError> {
        let at = out.args().len();
        let what = |e: DlpackError| ArgError::new(&crate::params::leaf_name(name), e.0);
        if self.is_host() {
            pack_tensor(name, spec, &self.host_tensor().map_err(what)?, out)?;
            out.set_external(at, ExternalArg { device: None, read_only: self.read_only });
        } else if self.is_cuda() {
            let bytes = self.span_bytes();
            let dtype = self.element_type();
            let have = if dtype.storage_bits() == 0 { 0 } else { bytes * 8 / dtype.storage_bits() };
            pack_tensor_with(name, spec, &self.layout, placeholder(dtype), have, out)?;
            out.set_external(
                at,
                ExternalArg { device: Some((self.address, bytes, self.device.device_id)), read_only: self.read_only },
            );
        } else {
            return Err(what(DlpackError(format!(
                "the tensor is in {}: cute-rs launches take kDLCPU and kDLCUDA memory",
                self.device_name()
            ))));
        }
        Ok(())
    }
}

/// A host value that is a DLPack tensor ([`DlTensor`]'s host value): an [`ExternalTensor`]
/// as converted, or a [`DlpackTensor`] converted implicitly. Other crates implement it for
/// their tensor types through their DLPack export (see the [module documentation](self)).
pub trait DlpackSource {
    /// The tensor to pass.
    fn to_external(&self) -> Result<ExternalTensor, DlpackError>;
}

impl DlpackSource for ExternalTensor {
    fn to_external(&self) -> Result<ExternalTensor, DlpackError> {
        Ok(self.clone())
    }
}

/// CuTe DSL's implicit conversion of a framework tensor passed to a `@cute.jit` function
/// (`TensorAdapter`): `from_dlpack(t).mark_layout_dynamic()`.
impl DlpackSource for DlpackTensor {
    fn to_external(&self) -> Result<ExternalTensor, DlpackError> {
        from_dlpack(self.clone(), FromDlpack::default())?.mark_layout_dynamic(None).map_err(|e| DlpackError(e.0))
    }
}

/// A tensor parameter whose host value is a DLPack tensor (any [`DlpackSource`] `H`; a
/// [`DlpackTensor`], converted implicitly, by default). Inside the kernel it is a
/// [`Tensor<T>`] (through `Deref`).
pub struct DlTensor<T: Numeric, H = DlpackTensor> {
    tensor: Tensor<T>,
    _host: PhantomData<fn() -> H>,
}

impl<T: Numeric, H> Clone for DlTensor<T, H> {
    fn clone(&self) -> Self {
        DlTensor { tensor: self.tensor.clone(), _host: PhantomData }
    }
}

impl<T: Numeric, H> std::ops::Deref for DlTensor<T, H> {
    type Target = Tensor<T>;
    fn deref(&self) -> &Tensor<T> {
        &self.tensor
    }
}

impl<T: Numeric, H> DlTensor<T, H> {
    /// The tensor.
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }
}

impl<T: Numeric, H: DlpackSource> KernelParam for DlTensor<T, H> {
    type Spec = TensorSpec;
    type Host = H;

    #[track_caller]
    fn declare(name: &str, spec: &TensorSpec) -> Self {
        DlTensor { tensor: declare_tensor(name, spec), _host: PhantomData }
    }

    #[track_caller]
    fn spec_of(host: &H) -> TensorSpec {
        host.to_external().unwrap_or_else(|e| panic!("cute: {e}")).spec().clone()
    }

    fn pack(name: &str, spec: &TensorSpec, host: &H, out: &mut ArgPack) -> Result<(), ArgError> {
        let t = host.to_external().map_err(|e| ArgError::new(&crate::params::leaf_name(name), e.0))?;
        t.pack(name, spec, out)
    }

    fn write_back(_: &TensorSpec, host: &mut H, buffers: &mut dyn Iterator<Item = KernelArg>) {
        let data = buffers.next().expect("cute: write_back: too few buffers");
        if let Ok(t) = host.to_external() {
            t.write_back(&data);
        }
    }

    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        (self.tensor.layout() != other.tensor.layout())
            .then(|| format!("a tensor of layout {:?} became one of {:?}", self.tensor.layout(), other.tensor.layout()))
    }
}
