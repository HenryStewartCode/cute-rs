// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device dtype conversion (CuTe DSL's `cute.testing.convert(src, dst)`): a kernel that
//! converts every element of one tensor into another tensor of the same shape and another
//! element type, so that tests can fill narrow-typed tensors (FP8, FP6, FP4, Int4) from
//! Float32 values without a host codec in the way.
//!
//! The rules:
//!
//! * the two tensors have the same shape and rank, and each has exactly one mode of extent
//!   above 1 and stride 1, its *leading mode*, the same mode in both;
//! * each thread converts one vector of `elem_per_copy` elements along the leading mode
//!   ([`elem_per_copy`]): 16 when either type is 6-bit (16 x 6 bits are 96 bits), 8 when
//!   either is 4-bit, 4 when either is 8-bit, else 2. (CuTe DSL's order puts the 4-bit test
//!   before the 6-bit one, which gives 8 for an FP4/FP6 pair, whose FP6 side then is not a
//!   whole number of 32-bit words; 16 is the smallest count that is.)
//!   The leading extent must be a multiple of it;
//! * the conversion is [`TensorSsa::to`](crate::TensorSsa::to), so FP4 and FP6 destinations
//!   use the paired `cvt` instructions (sm_100a and sm_120a; the simulator runs them
//!   everywhere).
//!
//! [`convert_sim`] runs the kernel on the simulator over host tensors; [`convert`] (feature
//! `cuda`) launches it on [`DeviceTensor`](crate::host::DeviceTensor)s. The kernel is built
//! once per (shape, strides, source type, destination type) and cached.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use crate::arch::{block_idx, thread_idx};
use crate::builder::Kernel;
use crate::control::if_;
use crate::ir::DType;
use crate::layout::{IndexExpr, IntTuple, Layout, SliceCoord, Tree};
use crate::numeric::{F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E4M3FN, F8E5M2, F8E8M0FNU, I4};
use crate::params::HostTensor;
use crate::sim::LaunchConfig;
use crate::tensor::{local_tile, param_tensor};
use crate::value::{BF16, F16, Numeric};

/// Why a conversion was refused or failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvertError(pub String);

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "convert: {}", self.0)
    }
}

impl std::error::Error for ConvertError {}

fn fail<T>(m: impl Into<String>) -> Result<T, ConvertError> {
    Err(ConvertError(m.into()))
}

/// The element types [`convert`] handles.
pub const CONVERT_DTYPES: [DType; 13] = [
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::F8E4M3FN,
    DType::F8E5M2,
    DType::F8E8M0FNU,
    DType::F6E2M3FN,
    DType::F6E3M2FN,
    DType::F4E2M1FN,
    DType::I4,
    DType::I8,
    DType::U8,
    DType::I32,
];

/// How many elements one thread converts for a pair of types (see the module docs).
pub fn elem_per_copy(src: DType, dst: DType) -> usize {
    let either = |bits: usize| src.bits() == bits || dst.bits() == bits;
    if either(6) {
        16
    } else if either(4) {
        8
    } else if either(8) {
        4
    } else {
        2
    }
}

/// The index of the one mode of `shape` and `stride` with extent above 1 and stride 1.
fn leading_mode(shape: &[usize], stride: &[usize]) -> Result<usize, ConvertError> {
    let leading: Vec<usize> = (0..shape.len()).filter(|&i| shape[i] > 1 && stride[i] == 1).collect();
    match leading[..] {
        [i] => Ok(i),
        _ => fail(format!(
            "the leading mode must be unique: exactly one mode may have extent above 1 and stride 1, but \
             shape {shape:?} with strides {stride:?} has {} (modes {leading:?})",
            leading.len()
        )),
    }
}

/// What a conversion of one pair of tensors does: the key the kernel is cached under.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    shape: Vec<usize>,
    src_stride: Vec<usize>,
    dst_stride: Vec<usize>,
    src: DType,
    dst: DType,
}

impl Key {
    /// Check the pair and find its leading mode and vector width.
    fn plan(&self) -> Result<(usize, usize), ConvertError> {
        for t in [self.src, self.dst] {
            if !CONVERT_DTYPES.contains(&t) {
                return fail(format!(
                    "{} tensors cannot be converted on the device (supported: {})",
                    t.cute_name(),
                    CONVERT_DTYPES.map(|t| t.cute_name()).join(", ")
                ));
            }
        }
        let lead = leading_mode(&self.shape, &self.src_stride)?;
        let dst_lead = leading_mode(&self.shape, &self.dst_stride)?;
        if lead != dst_lead {
            return fail(format!(
                "the source's leading mode is {lead} but the destination's is {dst_lead}: vectors along one mode \
                 cannot be stored along another"
            ));
        }
        let epc = elem_per_copy(self.src, self.dst);
        if !self.shape[lead].is_multiple_of(epc) {
            return fail(format!(
                "the leading extent {} (mode {lead}) is not a multiple of {epc}, the elements one thread converts \
                 between {} and {}",
                self.shape[lead],
                self.src.cute_name(),
                self.dst.cute_name()
            ));
        }
        for (t, stride) in [(self.src, &self.src_stride), (self.dst, &self.dst_stride)] {
            if t.is_word_packed() {
                for (i, (&s, &d)) in self.shape.iter().zip(stride).enumerate() {
                    if i != lead && s > 1 && !d.is_multiple_of(epc) {
                        return fail(format!(
                            "mode {i} of the {} tensor has stride {d}, not a multiple of {epc}: sub-byte vectors \
                             must start on a 32-bit word",
                            t.cute_name()
                        ));
                    }
                }
            }
        }
        Ok((lead, epc))
    }

    fn tiles(&self, lead: usize, epc: usize) -> usize {
        self.shape.iter().enumerate().map(|(i, &s)| if i == lead { s / epc } else { s }).product()
    }

    /// Threads per block and blocks.
    fn launch(&self, lead: usize, epc: usize) -> LaunchConfig {
        let tiles = self.tiles(lead, epc);
        let block = tiles.clamp(1, 128);
        LaunchConfig::new([tiles.div_ceil(block) as u32, 1, 1], [block as u32, 1, 1])
    }
}

/// The kernel: thread `t` converts the tile `t` (in colexicographic order over the tile
/// grid) of the source into the same tile of the destination.
fn build_kernel<S: Numeric, D: Numeric>(key: &Key, lead: usize, epc: usize) -> Kernel {
    let name = format!("convert_{}_{}", S::DTYPE.cute_name(), D::DTYPE.cute_name());
    let launch = key.launch(lead, epc);
    let block = launch.block[0] as usize;
    let tiles = key.tiles(lead, epc);
    let (src_layout, dst_layout) = (
        Layout::new(IntTuple::from(key.shape.clone()), IntTuple::from(key.src_stride.clone())),
        Layout::new(IntTuple::from(key.shape.clone()), IntTuple::from(key.dst_stride.clone())),
    );
    let tile: Vec<usize> = (0..key.shape.len()).map(|i| if i == lead { epc } else { 1 }).collect();
    let grid: Vec<usize> = (0..key.shape.len()).map(|i| key.shape[i] / tile[i]).collect();
    Kernel::builder(&name).block_dim([block as u32, 1, 1]).build(move || {
        let src = param_tensor::<S>("src", src_layout.clone());
        let dst = param_tensor::<D>("dst", dst_layout.clone());
        let t = block_idx().x * block as i32 + thread_idx().x;
        let body = || {
            let mut coords = Vec::with_capacity(grid.len());
            let mut stride = 1usize;
            for (k, &g) in grid.iter().enumerate() {
                let c = if stride == 1 { t } else { t / stride as i32 };
                let c = if k + 1 == grid.len() { c } else { c % g as i32 };
                coords.push(Tree::Leaf(Some(IndexExpr::from(c))));
                stride *= g;
            }
            let coord: SliceCoord = Tree::Branch(coords);
            let tiler = IntTuple::from(tile.clone());
            let from = local_tile(&src, &tiler, coord.clone());
            let to = local_tile(&dst, &tiler, coord);
            to.store_ssa(&from.load_ssa().to::<D>());
        };
        if tiles.is_multiple_of(block) {
            body();
        } else {
            if_(t.lt(tiles as i32), body);
        }
    })
}

macro_rules! with_dtype {
    ($dtype:expr, $T:ident => $body:expr, else $other:expr) => {
        match $dtype {
            DType::F32 => {
                type $T = f32;
                $body
            }
            DType::F16 => {
                type $T = F16;
                $body
            }
            DType::BF16 => {
                type $T = BF16;
                $body
            }
            DType::F8E4M3FN => {
                type $T = F8E4M3FN;
                $body
            }
            DType::F8E5M2 => {
                type $T = F8E5M2;
                $body
            }
            DType::F8E8M0FNU => {
                type $T = F8E8M0FNU;
                $body
            }
            DType::F6E2M3FN => {
                type $T = F6E2M3FN;
                $body
            }
            DType::F6E3M2FN => {
                type $T = F6E3M2FN;
                $body
            }
            DType::F4E2M1FN => {
                type $T = F4E2M1FN;
                $body
            }
            DType::I4 => {
                type $T = I4;
                $body
            }
            DType::I8 => {
                type $T = i8;
                $body
            }
            DType::U8 => {
                type $T = u8;
                $body
            }
            DType::I32 => {
                type $T = i32;
                $body
            }
            _ => $other,
        }
    };
}

/// A built kernel and how to launch it.
#[derive(Clone)]
pub(crate) struct Built {
    pub kernel: Arc<Kernel>,
    pub launch: LaunchConfig,
}

static CACHE: OnceLock<Mutex<HashMap<Key, Built>>> = OnceLock::new();

/// Number of kernels [`convert`] and [`convert_sim`] have built (each distinct key once).
pub fn convert_kernels_built() -> usize {
    CACHE.get().map_or(0, |c| c.lock().unwrap_or_else(|e| e.into_inner()).len())
}

fn built(key: Key) -> Result<Built, ConvertError> {
    let (lead, epc) = key.plan()?;
    let cache = CACHE.get_or_init(Default::default);
    if let Some(b) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(b.clone());
    }
    let kernel = with_dtype!(key.src, S => with_dtype!(key.dst, D => build_kernel::<S, D>(&key, lead, epc),
        else unreachable!("checked by plan")), else unreachable!("checked by plan"));
    let b = Built { kernel: Arc::new(kernel), launch: key.launch(lead, epc) };
    cache.lock().unwrap_or_else(|e| e.into_inner()).insert(key, b.clone());
    Ok(b)
}

fn flat_modes(layout: &Layout, what: &str) -> Result<(Vec<usize>, Vec<usize>), ConvertError> {
    if layout.depth() > 1 {
        return fail(format!("the {what} layout {layout} is hierarchical; only flat layouts are converted"));
    }
    Ok((layout.static_shape().flat_leaves(), layout.static_stride().flat_leaves()))
}

pub(crate) fn key_of(src: (DType, &Layout), dst: (DType, &Layout)) -> Result<Built, ConvertError> {
    let (shape, src_stride) = flat_modes(src.1, "source")?;
    let (dst_shape, dst_stride) = flat_modes(dst.1, "destination")?;
    if shape.len() != dst_shape.len() {
        return fail(format!(
            "the ranks differ: the source has {} modes, the destination {}",
            shape.len(),
            dst_shape.len()
        ));
    }
    if shape != dst_shape {
        return fail(format!("the shapes differ: {shape:?} and {dst_shape:?}"));
    }
    built(Key { shape, src_stride, dst_stride, src: src.0, dst: dst.0 })
}

/// Convert `src` into `dst` on the simulator (the same kernel [`convert`] launches): `dst`
/// keeps its shape and strides and gets the converted values in its own element type.
/// A destination element no source element maps to (strides that skip memory) is left alone.
pub fn convert_sim(src: &HostTensor, dst: &mut HostTensor) -> Result<(), ConvertError> {
    let b = key_of((src.element_type(), &src.layout()), (dst.element_type(), &dst.layout()))?;
    let mut args = [src.data().clone(), dst.data().clone()];
    b.kernel
        .simulate(b.launch, &mut args)
        .map_err(|e| ConvertError(format!("the simulator refused the kernel: {e}")))?;
    let [_, out] = args;
    *dst.data_mut() = out;
    Ok(())
}

#[cfg(feature = "cuda")]
pub use device::convert;

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use crate::backend::{CompileOptions, Toolchain};
    use crate::cuda::{CudaContext, DeviceArg};
    use crate::host::DeviceTensor;

    /// Convert `src` into `dst` on the GPU (CuTe DSL's `cute.testing.convert(src, dst)`); the
    /// two are device tensors of one shape (see the module docs for the rules). The kernel is
    /// compiled for the device's best target, so an FP4 or FP6 destination needs a device that
    /// runs the paired `cvt` instructions (sm_100a, sm_120a) and fails to compile elsewhere.
    pub fn convert(ctx: &CudaContext, src: &DeviceTensor<'_>, dst: &DeviceTensor<'_>) -> Result<(), ConvertError> {
        let b = key_of((src.dtype(), src.layout()), (dst.dtype(), dst.layout()))?;
        let tc = Toolchain::discover().ok_or_else(|| {
            ConvertError("no mlir-opt found, so the kernel cannot be compiled (set CUTE_MLIR_OPT)".into())
        })?;
        let arch = ctx.best_target(&tc).ok_or_else(|| ConvertError("no usable GPU target".into()))?;
        let compiled = b
            .kernel
            .compile_with(&tc, &CompileOptions::new().gpu_arch(arch))
            .map_err(|e| ConvertError(format!("compiling for {arch:?}: {e}")))?;
        let function = ctx.load(&compiled).map_err(|e| ConvertError(e.to_string()))?;
        ctx.launch_on_device(
            &b.kernel,
            &function,
            b.launch,
            &[DeviceArg::Buffer(src.buffer()), DeviceArg::Buffer(dst.buffer())],
        )
        .map_err(|e| ConvertError(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_width_follows_the_narrowest_type() {
        assert_eq!(elem_per_copy(DType::F32, DType::F16), 2);
        assert_eq!(elem_per_copy(DType::F32, DType::F8E5M2), 4);
        assert_eq!(elem_per_copy(DType::I4, DType::F32), 8);
        assert_eq!(elem_per_copy(DType::F6E2M3FN, DType::F16), 16);
        assert_eq!(elem_per_copy(DType::F4E2M1FN, DType::F6E3M2FN), 16);
    }
}
