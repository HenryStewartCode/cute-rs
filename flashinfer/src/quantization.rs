// SPDX-License-Identifier: Apache-2.0

//! Bit packing of boolean masks, after FlashInfer's `packbits` and `segment_packbits`
//! (`include/flashinfer/quantization.cuh`, Python `flashinfer.quantization.packbits`).
//! This is cute-rs's own code: it follows the published design (one thread per output
//! byte, 256 threads per block, one block per segment for the segmented form), but no
//! FlashInfer source was translated.
//!
//! Like `numpy.packbits`, [`pack_kernel`] packs `n` booleans (one byte each, 0 or 1) into
//! `ceil(n / 8)` bytes, the last padded with zeros: with [`Order::Big`] the first element is
//! the most significant bit of a byte, with [`Order::Little`] the least significant.
//! [`segment_kernel`] does so for each segment `input[indptr[b]..indptr[b + 1]]`, into
//! `output[out_indptr[b]..]`, `out_indptr` being the prefix sum of `ceil(len / 8)`.
//!
//! Where FlashInfer's `BlockLoad` loads a thread's eight booleans one byte at a time or as
//! a vector, depending on the alignment, this loads one aligned 8-byte word when the group
//! is whole and aligned, and gathers it byte by byte otherwise (the tail, or a segment that
//! starts off an 8-byte boundary); then the eight 0/1 bytes become one byte with a single
//! multiply (`word * 0x0102040810204080 >> 56` for little-endian bit order, the magic
//! `0x8040201008040201` for big): the multiply moves byte `i` to bit `i` (or `7 - i`) of
//! the top byte, and no two terms collide. The input base is assumed 8-byte aligned.

use cute::prelude::*;

pub use crate::norm::Dtype;
use crate::rope::Half;

/// Threads per block, each producing one output byte.
pub const THREADS: usize = 256;

/// The bit order inside each output byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// The first element is the most significant bit (`numpy`'s default, FlashInfer's).
    Big,
    /// The first element is the least significant bit.
    Little,
}

impl Order {
    fn tag(self) -> &'static str {
        match self {
            Order::Big => "big",
            Order::Little => "little",
        }
    }

    /// The multiplier that gathers the low bit of each byte into the top byte.
    fn magic(self) -> u64 {
        match self {
            Order::Big => 0x8040201008040201,
            Order::Little => 0x0102040810204080,
        }
    }
}

/// Eight booleans of the segment at `base + start` from element `rel` (a segment of `n` elements): one word if `fast`
/// and all eight exist, else byte by byte, elements past `n` reading as 0. The word's byte
/// `i` is element `rel + i`.
fn word8(base: &Ptr<u8>, start: Int32, rel: Int64, n: Int64, fast: Boolean) -> Uint64 {
    let full = fast & (rel + 8i64).le(n);
    let words = base.cast::<u64>();
    let p = base.offset(start);
    if_else_yield(
        full,
        // `fast` means `start` is a multiple of 8: the word index is `start / 8 + rel / 8`.
        || words.offset(start / 8i32 + (rel / 8i64).to::<i32>()).load_at(0),
        || {
            let mut w = Uint64::from(0u64);
            for i in 0..8i64 {
                let b =
                    if_else_yield((rel + i).lt(n), || p.offset(rel + i).load_at(0).to::<u64>(), || Uint64::from(0u64));
                w = w | (b << (8 * i) as u64);
            }
            w
        },
    )
}

/// The byte of eight 0/1 bytes.
fn gather(word: Uint64, order: Order) -> Uint8 {
    ((word * order.magic()) >> 56u64).to::<u8>()
}

/// `packbits`: parameters `input` (bytes 0/1), `output`, and the i64 element count. Launch
/// with `ceil(n / 2048)` blocks of [`THREADS`] threads.
pub fn pack_kernel(order: Order) -> Kernel {
    Kernel::builder(format!("flashinfer_packbits_{}", order.tag())).block_dim([THREADS as u32, 1, 1]).build(|| {
        let input = param_ptr_aligned::<u8>("input", 8);
        let output = param_ptr::<u8>("output");
        let n = param_scalar::<i64>("num_elements");
        let byte = block_idx().x.to::<i64>() * THREADS as i64 + thread_idx().x.to::<i64>();
        let rel = byte * 8i64;
        if_(rel.lt(n), || {
            let w = word8(&input, Int32::from(0i32), rel, n, Boolean::from(true));
            output.store_at(byte, gather(w, order));
        });
    })
}

/// `segment_packbits`: parameters `input`, `output`, `input_indptr`, `output_indptr` (i32).
/// One block of [`THREADS`] threads per segment.
pub fn segment_kernel(order: Order) -> Kernel {
    Kernel::builder(format!("flashinfer_segment_packbits_{}", order.tag())).block_dim([THREADS as u32, 1, 1]).build(
        || {
            let input = param_ptr_aligned::<u8>("input", 8);
            let output = param_ptr::<u8>("output");
            let in_indptr = param_ptr::<i32>("input_indptr");
            let out_indptr = param_ptr::<i32>("output_indptr");
            let b = block_idx().x;
            let start = in_indptr.load_at(b);
            let n = (in_indptr.load_at(b + 1) - start).to::<i64>();
            let dst = output.offset(out_indptr.load_at(b));
            let aligned = (start & 7i32).eq(0);
            // Each pass of the block covers 8 * THREADS elements.
            for_range(0, n.to::<i32>(), (8 * THREADS) as i32, |pass| {
                let byte = pass / 8 + thread_idx().x;
                let rel = (pass + thread_idx().x * 8).to::<i64>();
                if_(rel.lt(n), || {
                    let w = word8(&input, start, rel, n, aligned);
                    dst.store_at(byte, gather(w, order));
                });
            });
        },
    )
}

/// A packing test problem: random booleans in segments.
pub struct Problem {
    pub bits: Vec<bool>,
    pub indptr: Vec<i32>,
}

impl Problem {
    /// Segments of the given lengths, random bits, from `seed`.
    pub fn new(lens: &[usize], seed: u64) -> Self {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut indptr = vec![0i32];
        let mut bits = vec![];
        for &l in lens {
            for _ in 0..l {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                bits.push(s >> 40 & 1 == 1);
            }
            indptr.push(bits.len() as i32);
        }
        Problem { bits, indptr }
    }

    /// The packed bytes of each segment, concatenated: `numpy.packbits` per segment.
    pub fn reference(&self, order: Order) -> Vec<u8> {
        let mut out = vec![];
        for w in self.indptr.windows(2) {
            for chunk in self.bits[w[0] as usize..w[1] as usize].chunks(8) {
                let mut byte = 0u8;
                for (i, &b) in chunk.iter().enumerate() {
                    byte |= (b as u8) << if order == Order::Big { 7 - i } else { i };
                }
                out.push(byte);
            }
        }
        out
    }

    /// `output_indptr`: the prefix sums of `ceil(len / 8)`.
    pub fn out_indptr(&self) -> Vec<i32> {
        let mut o = vec![0i32];
        for w in self.indptr.windows(2) {
            o.push(o.last().unwrap() + (w[1] - w[0] + 7) / 8);
        }
        o
    }

    fn input(&self) -> KernelArg {
        KernelArg::Bytes { elem: DType::U8, data: self.bits.iter().map(|&b| b as u8).collect() }
    }

    fn output(&self) -> KernelArg {
        KernelArg::Bytes { elem: DType::U8, data: vec![0xAA; *self.out_indptr().last().unwrap() as usize] }
    }

    /// The arguments of [`pack_kernel`] (the whole array as one segment).
    pub fn pack_args(&self) -> Vec<KernelArg> {
        vec![
            self.input(),
            KernelArg::Bytes { elem: DType::U8, data: vec![0xAA; self.bits.len().div_ceil(8)] },
            KernelArg::Scalar { ty: DType::I64, bits: self.bits.len() as u64 },
        ]
    }

    /// The arguments of [`segment_kernel`].
    pub fn segment_args(&self) -> Vec<KernelArg> {
        vec![self.input(), self.output(), KernelArg::I32(self.indptr.clone()), KernelArg::I32(self.out_indptr())]
    }
}

/// The bytes of a kernel's output argument.
fn bytes(a: &KernelArg) -> Vec<u8> {
    a.to_vec::<u8>()
}

/// Simulate [`pack_kernel`] on all of `p`'s bits as one array; the number of wrong bytes.
pub fn simulate_pack(p: &Problem, order: Order) -> usize {
    let mut args = p.pack_args();
    let k = pack_kernel(order);
    let blocks = p.bits.len().div_ceil(8 * THREADS).max(1) as u32;
    k.simulate(LaunchConfig::new([blocks, 1, 1], [THREADS as u32, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let whole = Problem { bits: p.bits.clone(), indptr: vec![0, p.bits.len() as i32] };
    let want = whole.reference(order);
    let got = bytes(&args[1]);
    got.iter().zip(&want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len())
}

/// Simulate [`segment_kernel`] on `p`; the number of wrong bytes.
pub fn simulate_segment(p: &Problem, order: Order) -> usize {
    let mut args = p.segment_args();
    let k = segment_kernel(order);
    let batch = (p.indptr.len() - 1) as u32;
    k.simulate(LaunchConfig::new([batch, 1, 1], [THREADS as u32, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let want = p.reference(order);
    let got = bytes(&args[1]);
    got.iter().zip(&want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len())
}

// ---------------------------------------------------------------------------------------
// per_token_group_quant_8bit
// ---------------------------------------------------------------------------------------

/// The 8-bit type `per_token_group_quant_8bit` writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Out8 {
    E4M3,
    E5M2,
    Int8,
}

impl Out8 {
    pub fn name(self) -> &'static str {
        match self {
            Out8::E4M3 => "e4m3",
            Out8::E5M2 => "e5m2",
            Out8::Int8 => "int8",
        }
    }

    /// The clamp range of FlashInfer's launcher (`min_8bit`, `max_8bit`).
    pub fn range(self) -> (f32, f32) {
        match self {
            Out8::E4M3 => (-448.0, 448.0),
            Out8::E5M2 => (-57344.0, 57344.0),
            Out8::Int8 => (-128.0, 127.0),
        }
    }

    /// The byte of `v` (already clamped to [`Out8::range`]): the type's round-to-nearest-even
    /// for FP8, `floor(v + 0.5)` for int8.
    pub fn encode(self, v: f32) -> u8 {
        match self {
            Out8::E4M3 => F8E4M3FN::from_f32(v).0,
            Out8::E5M2 => F8E5M2::from_f32(v).0,
            Out8::Int8 => (v + 0.5).floor() as i32 as i8 as u8,
        }
    }

    /// The value a byte stands for.
    pub fn decode(self, b: u8) -> f32 {
        match self {
            Out8::E4M3 => F8E4M3FN(b).to_f32(),
            Out8::E5M2 => F8E5M2(b).to_f32(),
            Out8::Int8 => b as i8 as f32,
        }
    }
}

/// An element type [`group_quant_kernel`] can write.
trait Quant8: Numeric {
    /// `y`, clamped to the type's range, as the type.
    fn quant(y: Float32) -> Value<Self>;
}

impl Quant8 for F8E4M3FN {
    fn quant(y: Float32) -> Value<Self> {
        y.to::<F8E4M3FN>()
    }
}

impl Quant8 for F8E5M2 {
    fn quant(y: Float32) -> Value<Self> {
        y.to::<F8E5M2>()
    }
}

impl Quant8 for i8 {
    fn quant(y: Float32) -> Value<Self> {
        (y + 0.5f32).floor().to::<i8>()
    }
}

/// Values per lane: one 16-byte load of 16-bit input and one 8-byte store.
const QVEC: usize = 8;

/// Threads per block of [`group_quant_kernel`].
pub const QUANT_THREADS: usize = 256;

/// The scale layout and rounding of one `per_token_group_quant_8bit` launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QuantMode {
    /// Scales stored `[groups_per_row, scale_stride]` (token fastest), as
    /// `column_major_scales=True`.
    pub column_major: bool,
    /// Round each scale up to a power of two (`scale_ue8m0`); needs `column_major`.
    pub ue8m0: bool,
}

/// `per_token_group_quant_8bit` (`PerTokenGroupQuant8BitRegisterKernel`; own code, after
/// FlashInfer's register-resident design): the last dimension of the input is cut into groups
/// of `group_size` (64, 128 or 256); each group is `group_size / 8` lanes of a warp, every
/// lane loading 8 values with one 16-byte vector, so a 128-value group lives in registers of
/// half a warp, reduced to its absolute maximum by a butterfly (no shared memory, no block
/// barrier). `scale = max(eps, absmax) / max_8bit` (IEEE division; with `ue8m0` rounded up
/// to a power of two); each value becomes `clamp(x / scale)` converted to the output type
/// (`x * (1 / scale)` for `ue8m0`, which is the same for a power of two) and the lane stores
/// 8 bytes. Lanes of a group past the end recompute the last group and store nothing, so the
/// shuffles never see a partly exited warp. Parameters: `input` (16-byte aligned), `output_q`
/// (8-byte aligned), `output_s` (f32), `eps` (f32), `num_groups`, `groups_per_row`,
/// `scale_stride` (i32). Launch with `ceil(num_groups / (256 / (group_size / 8)))` blocks of
/// [`QUANT_THREADS`].
pub fn group_quant_kernel(dtype: Dtype, out: Out8, group_size: usize, mode: QuantMode) -> Kernel {
    match (dtype, out) {
        (Dtype::F16, Out8::E4M3) => group_quant_of::<F16, F8E4M3FN>(dtype, out, group_size, mode),
        (Dtype::F16, Out8::E5M2) => group_quant_of::<F16, F8E5M2>(dtype, out, group_size, mode),
        (Dtype::F16, Out8::Int8) => group_quant_of::<F16, i8>(dtype, out, group_size, mode),
        (Dtype::BF16, Out8::E4M3) => group_quant_of::<BF16, F8E4M3FN>(dtype, out, group_size, mode),
        (Dtype::BF16, Out8::E5M2) => group_quant_of::<BF16, F8E5M2>(dtype, out, group_size, mode),
        (Dtype::BF16, Out8::Int8) => group_quant_of::<BF16, i8>(dtype, out, group_size, mode),
    }
}

/// The launch of [`group_quant_kernel`] for `num_groups` groups.
pub fn group_quant_launch(num_groups: usize, group_size: usize) -> LaunchConfig {
    let per_block = QUANT_THREADS / (group_size / QVEC);
    LaunchConfig::new([num_groups.div_ceil(per_block).max(1) as u32, 1, 1], [QUANT_THREADS as u32, 1, 1])
}

fn group_quant_of<T: Half, O: Quant8>(dtype: Dtype, out: Out8, group_size: usize, mode: QuantMode) -> Kernel {
    assert!(matches!(group_size, 64 | 128 | 256), "group_size {group_size}: 64, 128 or 256");
    assert!(!mode.ue8m0 || mode.column_major, "scale_ue8m0 requires column_major scales");
    let lanes = group_size / QVEC;
    let per_block = (QUANT_THREADS / lanes) as i32;
    let (lo, hi) = out.range();
    let name = format!(
        "flashinfer_per_token_group_quant_8bit_{}_{}_g{group_size}{}{}",
        dtype.name(),
        out.name(),
        if mode.column_major { "_cm" } else { "" },
        if mode.ue8m0 { "_ue8m0" } else { "" }
    );
    Kernel::builder(&name).block_dim([QUANT_THREADS as u32, 1, 1]).build(|| {
        let input = param_ptr_aligned::<T>("input", 16);
        let output_q = param_ptr_aligned::<O>("output_q", 8);
        let output_s = param_ptr::<f32>("output_s");
        let eps = param_scalar::<f32>("eps");
        let num_groups = param_scalar::<i32>("num_groups");
        let groups_per_row = param_scalar::<i32>("groups_per_row");
        let scale_stride = param_scalar::<i32>("scale_stride");

        let tid = thread_idx().x;
        let lane = tid % lanes as i32;
        let group = block_idx().x * per_block + tid / lanes as i32;
        let valid = group.lt(num_groups);
        let g = select(valid, group, num_groups - 1i32);
        let at = g * group_size as i32 + lane * QVEC as i32;

        let x = load_vector::<T, QVEC>(&input.offset(at), LoadOptions::default());
        let xs: Vec<Float32> = x.iter().map(|v| v.to::<f32>()).collect();
        let absmax = xs.iter().fold(eps, |m, v| m.max(v.abs()));
        let absmax = warp_reduction_max(absmax, lanes);

        let mut scale = absmax / hi;
        if mode.ue8m0 {
            let s = scale.abs().max(1e-10f32);
            let bits = s.bitcast::<u32>();
            let exponent = (bits >> 23u32) & 0xffu32;
            let rounded = exponent + select((bits & 0x7fffffu32).ne(0u32), 1u32, 0u32);
            scale = (rounded << 23u32).bitcast::<f32>();
        }
        if_(valid & lane.eq(0), || {
            let slot = if mode.column_major { (g % groups_per_row) * scale_stride + g / groups_per_row } else { g };
            output_s.store_at(slot, scale);
        });

        let inv = Float32::constant(1.0) / scale;
        let q: [Value<O>; QVEC] = std::array::from_fn(|k| {
            let x = xs[k];
            let y = if mode.ue8m0 {
                x * inv
            } else {
                // `x / scale` as the division's fast path: a quotient from the reciprocal and
                // one residual correction (contracted into fmas), without `div.rn`'s slow-path
                // branch; correctly rounded for the normal operands here.
                let q = x * inv;
                q + (x - q * scale) * inv
            };
            O::quant(y.max(lo).min(hi))
        });
        if_(valid, || {
            store_vector::<O, QVEC>(&output_q.offset(at), q, StoreOptions::default());
        });
    })
}

/// A `per_token_group_quant_8bit` problem with its host data.
pub struct GroupQuantProblem {
    pub dtype: Dtype,
    pub out: Out8,
    pub tokens: usize,
    pub hidden: usize,
    pub group_size: usize,
    pub mode: QuantMode,
    pub eps: f32,
    /// `[tokens, hidden]`, exact in `dtype`.
    pub x: Vec<f32>,
}

impl GroupQuantProblem {
    /// Normally distributed-ish inputs with a few outlier channels and a few all-zero
    /// groups, from `seed`.
    pub fn new(
        dtype: Dtype,
        out: Out8,
        tokens: usize,
        hidden: usize,
        group_size: usize,
        mode: QuantMode,
        seed: u64,
    ) -> Self {
        assert!(hidden.is_multiple_of(group_size));
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let mut x: Vec<f32> = (0..tokens * hidden)
            .map(|i| {
                let v = next() * 3.0 + next();
                dtype.round(if i % 97 == 5 { v * 40.0 } else { v })
            })
            .collect();
        // A group of zeros: the scale is `eps / max`.
        if x.len() >= group_size * 3 {
            x[group_size * 2..group_size * 3].fill(0.0);
        }
        GroupQuantProblem { dtype, out, tokens, hidden, group_size, mode, eps: 1e-10, x }
    }

    pub fn groups_per_row(&self) -> usize {
        self.hidden / self.group_size
    }

    pub fn num_groups(&self) -> usize {
        self.tokens * self.groups_per_row()
    }

    /// The scale tensor's row stride (tokens) in column-major layout.
    pub fn scale_stride(&self) -> usize {
        self.tokens
    }

    pub fn kernel(&self) -> Kernel {
        group_quant_kernel(self.dtype, self.out, self.group_size, self.mode)
    }

    pub fn launch(&self) -> LaunchConfig {
        group_quant_launch(self.num_groups(), self.group_size)
    }

    /// A zeroed `output_q`.
    pub fn out_buffer(&self) -> KernelArg {
        let n = self.x.len();
        match self.out {
            Out8::E4M3 => KernelArg::buffer(&vec![F8E4M3FN(0); n]),
            Out8::E5M2 => KernelArg::buffer(&vec![F8E5M2(0); n]),
            Out8::Int8 => KernelArg::buffer(&vec![0i8; n]),
        }
    }

    /// The bytes of a kernel's `output_q` argument.
    pub fn out_bytes(&self, a: &KernelArg) -> Vec<u8> {
        match self.out {
            Out8::E4M3 => a.to_vec::<F8E4M3FN>().into_iter().map(|v| v.0).collect(),
            Out8::E5M2 => a.to_vec::<F8E5M2>().into_iter().map(|v| v.0).collect(),
            Out8::Int8 => a.to_vec::<i8>().into_iter().map(|v| v as u8).collect(),
        }
    }

    /// `input, output_q, output_s, eps, num_groups, groups_per_row, scale_stride`.
    pub fn args(&self) -> Vec<KernelArg> {
        vec![
            self.dtype.buffer(&self.x),
            self.out_buffer(),
            KernelArg::f32(vec![f32::NAN; self.num_groups()]),
            KernelArg::ScalarF32(self.eps),
            KernelArg::ScalarI32(self.num_groups() as i32),
            KernelArg::ScalarI32(self.groups_per_row() as i32),
            KernelArg::ScalarI32(self.scale_stride() as i32),
        ]
    }

    /// The expected `(output_q bytes, scales)` in the layout the kernel writes: the scale of
    /// group `g` (token `g / gpr`, group `g % gpr`) at `g` row-major or
    /// `(g % gpr) * tokens + g / gpr` column-major. Every step is the f32 operation the
    /// kernel does (nothing contracts), so the result is exact.
    pub fn reference(&self) -> (Vec<u8>, Vec<f32>) {
        let (lo, hi) = self.out.range();
        let gs = self.group_size;
        let gpr = self.groups_per_row();
        let mut q = vec![0u8; self.x.len()];
        let mut s = vec![0f32; self.num_groups()];
        for g in 0..self.num_groups() {
            let grp = &self.x[g * gs..(g + 1) * gs];
            let absmax = grp.iter().fold(self.eps, |m, v| m.max(v.abs()));
            let mut scale = absmax / hi;
            if self.mode.ue8m0 {
                let b = scale.abs().max(1e-10).to_bits();
                let r = ((b >> 23) & 0xff) + ((b & 0x7fffff) != 0) as u32;
                scale = f32::from_bits(r << 23);
            }
            let inv = 1.0 / scale;
            for (i, &v) in grp.iter().enumerate() {
                let y = if self.mode.ue8m0 { v * inv } else { v / scale };
                q[g * gs + i] = self.out.encode(y.max(lo).min(hi));
            }
            let slot = if self.mode.column_major { (g % gpr) * self.tokens + g / gpr } else { g };
            s[slot] = scale;
        }
        (q, s)
    }

    /// The number of wrong bytes and scales in kernel output (`q` bytes, `s` scales).
    pub fn mismatches(&self, q: &[u8], s: &[f32]) -> usize {
        let (wq, ws) = self.reference();
        let bq = q.iter().zip(&wq).filter(|(a, b)| a != b).count() + q.len().abs_diff(wq.len());
        let bs = s.iter().zip(&ws).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + s.len().abs_diff(ws.len());
        bq + bs
    }

    /// The largest `|dequantized - x|` over the largest scale: at most about half a step of
    /// the output type, a check on the reference itself.
    pub fn dequant_error(&self, q: &[u8], s: &[f32]) -> f32 {
        let gpr = self.groups_per_row();
        let mut worst = 0f32;
        for (i, &b) in q.iter().enumerate() {
            let g = i / self.group_size;
            let slot = if self.mode.column_major { (g % gpr) * self.tokens + g / gpr } else { g };
            let d = self.out.decode(b) * s[slot];
            worst = worst.max((d - self.x[i]).abs() / s[slot]);
        }
        worst
    }
}

/// Simulate [`GroupQuantProblem::kernel`]; the number of wrong bytes and scales.
pub fn simulate_group_quant(p: &GroupQuantProblem) -> usize {
    let k = p.kernel();
    let mut args = p.args();
    k.simulate(p.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.mismatches(&p.out_bytes(&args[1]), &args[2].to_vec::<f32>())
}

/// The GPU driver: compiled kernels, correctness checks and CUDA-graph timings.
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{GroupQuantProblem, Order, Problem, THREADS, pack_kernel, segment_kernel};

    fn compile<'c>(
        ctx: &'c CudaContext,
        tc: &Toolchain,
        k: Kernel,
    ) -> Result<(Kernel, CudaFunction<'c>), Box<dyn Error>> {
        let arch = ctx.best_target(tc).ok_or("no target for this GPU")?;
        let f = ctx.load(&k.compile_with(tc, &CompileOptions::new().gpu_arch(arch))?)?;
        Ok((k, f))
    }

    /// Both kernels for one bit order, compiled and loaded once.
    pub struct Kernels<'c> {
        pub pack: (Kernel, CudaFunction<'c>),
        pub segment: (Kernel, CudaFunction<'c>),
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &Toolchain, order: Order) -> Result<Self, Box<dyn Error>> {
            Ok(Kernels {
                pack: compile(ctx, tc, pack_kernel(order))?,
                segment: compile(ctx, tc, segment_kernel(order))?,
            })
        }
    }

    /// Run `packbits` (`segmented == false`, all bits as one array) or `segment_packbits`
    /// once, count the wrong bytes, and optionally time it (graph replays, microseconds).
    pub fn run(
        ctx: &CudaContext,
        ks: &Kernels<'_>,
        order: Order,
        p: &Problem,
        segmented: bool,
        time: bool,
    ) -> Result<(usize, Option<f64>), Box<dyn Error>> {
        let host = if segmented { p.segment_args() } else { p.pack_args() };
        let bufs: Vec<_> = host
            .iter()
            .map(|a| match a {
                KernelArg::Scalar { .. } => Ok(None),
                _ => ctx.upload(a).map(Some),
            })
            .collect::<Result<_, _>>()?;
        let args: Vec<DeviceArg<'_>> = host
            .iter()
            .zip(&bufs)
            .map(|(a, b)| match b {
                Some(b) => DeviceArg::Buffer(b),
                None => DeviceArg::Scalar(a.clone()),
            })
            .collect();
        let (kk, f) = if segmented { (&ks.segment.0, &ks.segment.1) } else { (&ks.pack.0, &ks.pack.1) };
        let blocks = if segmented { p.indptr.len() - 1 } else { p.bits.len().div_ceil(8 * THREADS) };
        let cfg = LaunchConfig::new([blocks.max(1) as u32, 1, 1], [THREADS as u32, 1, 1]);
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(kk, f, cfg).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let got = bufs[1].as_ref().unwrap().to_vec::<u8>()?;
        let want = if segmented {
            p.reference(order)
        } else {
            Problem { bits: p.bits.clone(), indptr: vec![0, p.bits.len() as i32] }.reference(order)
        };
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(20).iterations(200).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((bad, t))
    }

    /// Run [`GroupQuantProblem`] once on the GPU: the number of wrong bytes and scales
    /// against [`GroupQuantProblem::reference`], and (if `time`) the CUDA-graph GPU time in
    /// microseconds.
    pub fn run_group_quant(
        ctx: &CudaContext,
        tc: &Toolchain,
        p: &GroupQuantProblem,
        time: bool,
    ) -> Result<(usize, Option<f64>), Box<dyn Error>> {
        let (k, f) = compile(ctx, tc, p.kernel())?;
        let host = p.args();
        let bufs: Vec<_> = host
            .iter()
            .map(|a| match a {
                KernelArg::ScalarF32(_) | KernelArg::ScalarI32(_) => Ok(None),
                _ => ctx.upload(a).map(Some),
            })
            .collect::<Result<_, _>>()?;
        let args: Vec<DeviceArg<'_>> = host
            .iter()
            .zip(&bufs)
            .map(|(a, b)| match b {
                Some(b) => DeviceArg::Buffer(b),
                None => DeviceArg::Scalar(a.clone()),
            })
            .collect();
        let cfg = p.launch();
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(&k, &f, cfg).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let q = bufs[1].as_ref().unwrap().to_vec::<u8>()?;
        let s = bufs[2].as_ref().unwrap().to_vec::<f32>()?;
        let bad = p.mismatches(&q, &s);
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(20).iterations(200).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((bad, t))
    }
}
