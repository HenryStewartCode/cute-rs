// SPDX-License-Identifier: MIT OR Apache-2.0

//! Torch-free GEMM host tensors (CuTe DSL's `cutlass.utils.gemm`, from
//! `tensor_utils`): random A, B and D operands of a given majorness
//! ([`create_gemm_tensor`], [`get_gemm_tensor`], [`get_gemm_tensors`]),
//! block-scaled scale factors in the packed layout the hardware reads
//! ([`create_scale_factor_tensor`], [`unpack_scale_factors`]), FP4
//! decoding ([`decode_float4e2m1fn`]), a seeded random number generator
//! ([`Rng`], no dependencies) and a CPU reference GEMM
//! ([`reference_gemm`], f64 accumulation).
//!
//! Operands are `(MN, K, L)` tensors (`L` the batch): MN-major has strides
//! `(1, MN, MN * K)`, K-major `(K, 1, MN * K)`. A [`GemmTensor`] holds the
//! buffer as a [`KernelArg`] with the matching [`Layout`], and the logical
//! values as f32 for the reference.
//!
//! Note for the block-scaled port (m10): the scale-factor index map here is
//! written from the explicit formula, not from the `BlockScaledBasicChunk`
//! layout; cross-check them against `blockscaled-sf-gmem-layout` when that
//! lands.

/// CuTe DSL's `cutlass.utils.gemm.sm100`: the packaged Blackwell epilogues.
pub mod sm100 {
    pub use crate::utils::sm100_epilogue::{
        EpilogueConfig, SmemCopyPartition, TmemCopyPartition, epilogue, epilogue_smem_copy_and_partition,
        epilogue_tma_store, epilogue_tmem_copy_and_partition, transform_partitioned_tensor_layout,
    };
}

use crate::ir::DType;
use crate::layout::Layout;
use crate::numeric::{NarrowRounding, encode};
use crate::nvgpu::OperandMajorMode;
use crate::params::HostTensor;
use crate::sim::KernelArg;
use crate::testing::CantImplement;

/// A small seeded generator (SplitMix64), so that tests draw the same
/// operands on every run and platform.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform index below `n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::below(0)");
        (self.next_u64() % n as u64) as usize
    }

    /// One of `items`, uniformly.
    pub fn choose<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// The bytes CuTe DSL draws FP4 operands from: each holds two FP4 values
/// from `{0, 1, -1}`, the low nibble first.
pub const FP4_BYTES: [u8; 9] = [0x00, 0x02, 0x0A, 0x20, 0x22, 0x2A, 0xA0, 0xA2, 0xAA];

/// The values of an FP4 (E2M1) code, indexed by the 4-bit pattern.
pub const FP4_E2M1_LUT: [f32; 16] =
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// A GEMM operand on the host: a `(MN, K, L)` tensor in its physical order.
#[derive(Clone, Debug)]
pub struct GemmTensor {
    /// The buffer, in memory order (`layout` maps coordinates to its
    /// elements). An FP4 operand is a byte buffer of one byte per logical
    /// element (CuTe DSL over-allocates it 2x on purpose; only the first
    /// half of the bytes is read, as FP4 elements).
    pub data: KernelArg,
    /// `(MN, K, L)` with the strides of the majorness.
    pub layout: Layout,
    pub dtype: DType,
    pub major: OperandMajorMode,
    shape: (usize, usize, usize),
    values: Vec<f32>,
}

impl GemmTensor {
    /// `(MN, K, L)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    /// The strides `(MN, K, L)` of the majorness.
    pub fn strides(&self) -> (usize, usize, usize) {
        gemm_strides(self.shape, self.major)
    }

    /// The buffer index of element `(mn, k, l)`.
    pub fn offset(&self, mn: usize, k: usize, l: usize) -> usize {
        let (smn, sk, sl) = self.strides();
        mn * smn + k * sk + l * sl
    }

    /// The logical values, `mn + MN * (k + K * l)`, as f32.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The value of element `(mn, k, l)`.
    pub fn value(&self, mn: usize, k: usize, l: usize) -> f32 {
        let (m, kk, _) = self.shape;
        self.values[mn + m * (k + kk * l)]
    }

    /// The operand as a host tensor with its leading dimension marked
    /// dynamic (CuTe DSL's `get_gemm_tensor`): mode 1 for K-major, mode 0
    /// for MN-major. An FP4 operand is a `u8` tensor.
    pub fn host_tensor(&self) -> HostTensor {
        let leading = if self.major == OperandMajorMode::K { 1 } else { 0 };
        HostTensor::new(self.data.clone(), self.layout.clone())
            .mark_layout_dynamic(Some(leading))
            .expect("the leading stride of a GEMM operand is 1")
    }
}

fn gemm_strides((mn, k, _): (usize, usize, usize), major: OperandMajorMode) -> (usize, usize, usize) {
    match major {
        OperandMajorMode::MN => (1, mn, mn * k),
        OperandMajorMode::K => (k, 1, mn * k),
    }
}

/// The bytes of `x` in element type `dtype`, little-endian.
fn encode_value(dtype: DType, x: i32) -> Result<Vec<u8>, CantImplement> {
    let bytes = dtype.bits() / 8;
    if let Some(f) = dtype.float_format()
        && matches!(dtype.bits(), 8 | 16 | 32)
    {
        let bits = encode(f, x as f64, NarrowRounding::NearestEven, true, false);
        return Ok(bits.to_le_bytes()[..bytes].to_vec());
    }
    let is_int = matches!(
        dtype,
        DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U8 | DType::U16 | DType::U32 | DType::U64
    );
    if is_int {
        // Unsigned types wrap -1 as torch's `random_(-1, 2).to(dtype)` does.
        return Ok((x as i64).to_le_bytes()[..bytes].to_vec());
    }
    Err(CantImplement::new(format!("create_gemm_tensor: unsupported element type {dtype}")))
}

/// A random GEMM operand (CuTe DSL's `create_gemm_tensor_torch`): shape
/// `(mn_or_n, k_or_n, l)`, strides of `major`, each element drawn from
/// `{-1, 0, 1}`. An FP4 (`Float4E2M1FN`) operand is bytes drawn from
/// [`FP4_BYTES`], one byte per logical element (see [`GemmTensor::data`]).
pub fn create_gemm_tensor(
    mn_or_n: usize,
    k_or_n: usize,
    l: usize,
    major: OperandMajorMode,
    dtype: DType,
    rng: &mut Rng,
) -> Result<GemmTensor, CantImplement> {
    let shape = (mn_or_n, k_or_n, l);
    let n = mn_or_n * k_or_n * l;
    let (smn, sk, sl) = gemm_strides(shape, major);
    let layout = Layout::new((mn_or_n, k_or_n, l), (smn, sk, sl));
    // Draw in logical order (so the values do not depend on the majorness),
    // then scatter into memory order.
    let at = |mn: usize, k: usize, l: usize| mn * smn + k * sk + l * sl;
    let coords = |i: usize| (i % mn_or_n, i / mn_or_n % k_or_n, i / (mn_or_n * k_or_n));
    if dtype == DType::F4E2M1FN {
        let mut logical = vec![0u8; n];
        for b in logical.iter_mut() {
            *b = rng.choose(&FP4_BYTES);
        }
        let mut data = vec![0u8; n];
        for (i, &b) in logical.iter().enumerate() {
            let (m, k, l) = coords(i);
            data[at(m, k, l)] = b;
        }
        let values = decode_float4e2m1fn(&logical, shape);
        return Ok(GemmTensor {
            data: KernelArg::Bytes { elem: DType::U8, data },
            layout,
            dtype,
            major,
            shape,
            values,
        });
    }
    let size = dtype.bits() / 8;
    let mut data = vec![0u8; n * size];
    let mut values = vec![0f32; n];
    let draws: Vec<i32> = (0..n).map(|_| rng.below(3) as i32 - 1).collect();
    for (i, &x) in draws.iter().enumerate() {
        let (m, k, l) = coords(i);
        let bytes = encode_value(dtype, x)?;
        let o = at(m, k, l) * size;
        data[o..o + size].copy_from_slice(&bytes);
        // An unsigned type holds -1 as its maximum, which is what the
        // reference must see.
        values[i] = match dtype {
            DType::U8 if x < 0 => 255.0,
            DType::U16 if x < 0 => 65535.0,
            DType::U32 if x < 0 => 4294967295.0,
            DType::U64 if x < 0 => u64::MAX as f32,
            _ => x as f32,
        };
    }
    Ok(GemmTensor { data: KernelArg::Bytes { elem: dtype, data }, layout, dtype, major, shape, values })
}

/// The host tensor of an operand with its leading dimension dynamic (CuTe
/// DSL's `get_gemm_tensor`); see [`GemmTensor::host_tensor`].
pub fn get_gemm_tensor(tensor: &GemmTensor) -> HostTensor {
    tensor.host_tensor()
}

/// Operands A `(M, K, L)`, B `(N, K, L)` and D `(M, N, L)` of a GEMM, each
/// random like [`create_gemm_tensor`] (CuTe DSL's `get_gemm_tensors`, which
/// also initialises D). `majors` and `dtypes` are `(A, B, D)`.
pub fn get_gemm_tensors(
    m: usize,
    n: usize,
    k: usize,
    l: usize,
    majors: (OperandMajorMode, OperandMajorMode, OperandMajorMode),
    dtypes: (DType, DType, DType),
    rng: &mut Rng,
) -> Result<(GemmTensor, GemmTensor, GemmTensor), CantImplement> {
    let a = create_gemm_tensor(m, k, l, majors.0, dtypes.0, rng)?;
    let b = create_gemm_tensor(n, k, l, majors.1, dtypes.1, rng)?;
    let d = create_gemm_tensor(m, n, l, majors.2, dtypes.2, rng)?;
    Ok((a, b, d))
}

/// Decode FP4 (E2M1) data into f32 (CuTe DSL's `decode_float4e2m1fn`).
/// `bytes` holds the `(MN, K, L)` tensor's logical elements (`mn + MN * (k +
/// K * l)`) as bytes of two values each, the low nibble the even element
/// and the high nibble the odd one, over the elements counted in
/// `(L, MN, K)` order; as [`create_gemm_tensor`] over-allocates, only the
/// first half of the bytes is read. The result is in logical order.
pub fn decode_float4e2m1fn(bytes: &[u8], (mn, k, l): (usize, usize, usize)) -> Vec<f32> {
    let n = mn * k * l;
    assert!(bytes.len() >= n, "decode_float4e2m1fn: {} bytes for {n} elements", bytes.len());
    let mut out = vec![0f32; n];
    for li in 0..l {
        for m in 0..mn {
            for kk in 0..k {
                let p = (li * mn + m) * k + kk;
                // The byte at index p / 2 of the (L, MN, K) order.
                let (b, q) = (p / 2, p % 2);
                let (bm, bk, bl) = (b / k % mn, b % k, b / (mn * k));
                let byte = bytes[bm + mn * (bk + k * bl)];
                out[m + mn * (kk + k * li)] = FP4_E2M1_LUT[(byte >> (4 * q) & 0xF) as usize];
            }
        }
    }
    out
}

const SF_ATOM: usize = 512;
const SF_ATOM_MN: usize = 128;
const SF_ATOM_K: usize = 4;

/// The index into the packed scale factors of the factor that scales element
/// `(mn, k, l)` of an `(MN, K, L)` operand with one factor per `sf_vec_size`
/// elements along K: the 512-element atoms are 128 rows by 4 factors, the
/// row `r` of an atom at `16 (r % 32) + 4 (r / 32)`.
pub fn scale_factor_offset(
    (mn, k, l): (usize, usize, usize),
    sf_vec_size: usize,
    (mn_total, k_total): (usize, usize),
) -> usize {
    let k_tile = SF_ATOM_K * sf_vec_size;
    let k_tiles = k_total.div_ceil(k_tile);
    let mn_tiles = mn_total.div_ceil(SF_ATOM_MN);
    (mn / SF_ATOM_MN) * (SF_ATOM * k_tiles)
        + (k / k_tile) * SF_ATOM
        + 16 * (mn % 32)
        + SF_ATOM_K * ((mn % SF_ATOM_MN) / 32)
        + (k / sf_vec_size) % SF_ATOM_K
        + l * SF_ATOM * mn_tiles * k_tiles
}

/// The number of scale factors of an `(MN, K, L)` operand after padding MN
/// to a multiple of 128 and `ceil(K / sf_vec_size)` to a multiple of 4.
pub fn scale_factor_len(mn: usize, k: usize, l: usize, sf_vec_size: usize) -> usize {
    mn.div_ceil(SF_ATOM_MN) * SF_ATOM_MN * k.div_ceil(sf_vec_size).div_ceil(SF_ATOM_K) * SF_ATOM_K * l
}

/// Expand packed scale factors (`sf`, as many as [`scale_factor_len`]) to a
/// dense `(MN, K, L)` tensor in logical order (CuTe DSL's
/// `unpack_scale_factors`): element `(m, k, l)` holds the factor that scales
/// it.
pub fn unpack_scale_factors(sf: &[f32], sf_vec_size: usize, mn: usize, k: usize, l: usize) -> Vec<f32> {
    assert!(sf.len() >= scale_factor_len(mn, k, l, sf_vec_size), "unpack_scale_factors: too few scale factors");
    let mut out = vec![0f32; mn * k * l];
    for li in 0..l {
        for kk in 0..k {
            for m in 0..mn {
                out[m + mn * (kk + k * li)] = sf[scale_factor_offset((m, kk, li), sf_vec_size, (mn, k))];
            }
        }
    }
    out
}

/// The inverse of [`unpack_scale_factors`]: pack a dense `(MN, K, L)` tensor
/// that is constant over each block of `sf_vec_size` elements along K. The
/// padding is zero.
pub fn pack_scale_factors(dense: &[f32], sf_vec_size: usize, mn: usize, k: usize, l: usize) -> Vec<f32> {
    assert_eq!(dense.len(), mn * k * l, "pack_scale_factors: wrong number of values");
    let mut sf = vec![0f32; scale_factor_len(mn, k, l, sf_vec_size)];
    for li in 0..l {
        for kk in 0..k {
            for m in 0..mn {
                sf[scale_factor_offset((m, kk, li), sf_vec_size, (mn, k))] = dense[m + mn * (kk + k * li)];
            }
        }
    }
    sf
}

/// Random scale factors for a block-scaled operand: the packed data and the
/// dense reference.
#[derive(Clone, Debug)]
pub struct ScaleFactors {
    /// The dense f32 `(MN, K, L)` tensor, in logical order, to multiply
    /// into the operand for a reference (CuTe DSL's `sf_ref`).
    pub reference: Vec<f32>,
    /// The packed scale factors as `sf_dtype` bytes, in the order the
    /// hardware reads (`sf_cute`).
    pub data: KernelArg,
    /// The same values as f32, one per element of `data` (`sf_f32`).
    pub packed: Vec<f32>,
}

/// Random scale factors in the packed layout (CuTe DSL's
/// `create_scale_factor_tensor`): MN is padded to a multiple of 128 and
/// `ceil(K / sf_vec_size)` to a multiple of 4, values come from `{1, 2, 4}`
/// and are stored as `sf_dtype` (any float type of 8, 16 or 32 bits).
pub fn create_scale_factor_tensor(
    mn: usize,
    k: usize,
    l: usize,
    sf_vec_size: usize,
    sf_dtype: DType,
    rng: &mut Rng,
) -> Result<ScaleFactors, CantImplement> {
    let n = scale_factor_len(mn, k, l, sf_vec_size);
    let packed: Vec<f32> = (0..n).map(|_| rng.choose(&[1.0f32, 2.0, 4.0])).collect();
    let mut data = Vec::with_capacity(n * sf_dtype.bits() / 8);
    for &x in &packed {
        data.extend(encode_value(sf_dtype, x as i32)?);
    }
    let reference = unpack_scale_factors(&packed, sf_vec_size, mn, k, l);
    Ok(ScaleFactors { reference, data: KernelArg::Bytes { elem: sf_dtype, data }, packed })
}

/// `D = A B^T` over `(M, K, L)` and `(N, K, L)` operands, accumulated in
/// f64, as `(M, N, L)` in logical order (`m + M * (n + N * l)`).
pub fn reference_gemm_values(a: &[f32], b: &[f32], (m, n, k, l): (usize, usize, usize, usize)) -> Vec<f64> {
    assert_eq!(a.len(), m * k * l, "reference_gemm: A has the wrong size");
    assert_eq!(b.len(), n * k * l, "reference_gemm: B has the wrong size");
    let mut d = vec![0f64; m * n * l];
    for li in 0..l {
        for ni in 0..n {
            for mi in 0..m {
                let mut acc = 0f64;
                for ki in 0..k {
                    acc += a[mi + m * (ki + k * li)] as f64 * b[ni + n * (ki + k * li)] as f64;
                }
                d[mi + m * (ni + n * li)] = acc;
            }
        }
    }
    d
}

/// [`reference_gemm_values`] of two operands (`A` `(M, K, L)`, `B`
/// `(N, K, L)`). Panics when K or L differ.
pub fn reference_gemm(a: &GemmTensor, b: &GemmTensor) -> Vec<f64> {
    let ((m, k, l), (n, kb, lb)) = (a.shape(), b.shape());
    assert_eq!((k, l), (kb, lb), "reference_gemm: A is {:?} and B {:?}", a.shape(), b.shape());
    reference_gemm_values(a.values(), b.values(), (m, n, k, l))
}

/// The elements of `values` (logical order) laid out in memory order for
/// `shape` and `major`, e.g. to compare a kernel's D buffer with
/// [`reference_gemm`].
pub fn to_memory_order<T: Copy + Default>(
    values: &[T],
    shape: (usize, usize, usize),
    major: OperandMajorMode,
) -> Vec<T> {
    let (mn, k, l) = shape;
    let (smn, sk, sl) = gemm_strides(shape, major);
    let mut out = vec![T::default(); mn * k * l];
    for li in 0..l {
        for kk in 0..k {
            for m in 0..mn {
                out[m * smn + kk * sk + li * sl] = values[m + mn * (kk + k * li)];
            }
        }
    }
    out
}
