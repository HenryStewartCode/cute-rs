//! CuTe DSL's narrow conversion helpers (`cute.arch.cvt_f32x2_bf16x2`,
//! `cvt_f32_tf32`, `cvt_i8x4_to_f32x4`, `cvt_i8_bf16_intrinsic`,
//! `sext_unpacked_i4x4_to_i8x4`, ...). CuTe DSL takes and returns MLIR
//! vectors; these take arrays of values or their bits
//! packed into one integer, and return arrays.

use std::panic::Location;

use super::half::Half2;
use crate::builder::{self, LocationScope};
use crate::ir::{CvtFn, IntFn, ValueId};
use crate::value::{BF16, BFloat16, Float32, Int8, Int16, Int32, IntoValue, Uint16, Uint32, Value, pack, unpack};

/// Four Int8 values in either of the forms the int8 conversions take (CuTe
/// DSL's `vector<4xi8>`): an array, or their bits packed into a 32-bit word
/// (element 0 in the low byte), as an Int32 or a Uint32.
pub trait Int8x4: Copy {
    /// The four values as one word.
    fn to_word(self) -> Uint32;
    /// The four values of this form held in `word`.
    fn from_word(word: Uint32) -> Self;
}

impl Int8x4 for [Int8; 4] {
    #[track_caller]
    fn to_word(self) -> Uint32 {
        pack(self)
    }

    #[track_caller]
    fn from_word(word: Uint32) -> Self {
        unpack(word)
    }
}

impl Int8x4 for Int32 {
    #[track_caller]
    fn to_word(self) -> Uint32 {
        self.bitcast()
    }

    #[track_caller]
    fn from_word(word: Uint32) -> Self {
        word.bitcast()
    }
}

impl Int8x4 for Uint32 {
    fn to_word(self) -> Uint32 {
        self
    }

    fn from_word(word: Uint32) -> Self {
        word
    }
}

/// Two Int8 values in either of the forms the int8 conversions take (CuTe
/// DSL's `vector<2xi8>`): an array, or their bits packed into 16 bits
/// (element 0 in the low byte), as an Int16 or a Uint16.
pub trait Int8x2: Copy {
    /// The two values as 16 bits.
    fn to_half_word(self) -> Int16;
}

impl Int8x2 for [Int8; 2] {
    #[track_caller]
    fn to_half_word(self) -> Int16 {
        pack(self)
    }
}

impl Int8x2 for Int16 {
    fn to_half_word(self) -> Int16 {
        self
    }
}

impl Int8x2 for Uint16 {
    #[track_caller]
    fn to_half_word(self) -> Int16 {
        self.bitcast()
    }
}

/// A conversion with modifiers.
#[track_caller]
fn cvt<R: crate::value::Numeric>(func: CvtFn, args: Vec<ValueId>) -> Value<R> {
    let loc = Location::caller();
    Value::from_id(builder::with(|b| b.cvt(func, args, loc)))
}

/// Float32 to BFloat16, rounded to nearest even (CuTe DSL's
/// `cute.arch.cvt_f32_bf16`, `cvt.rn.bf16.f32`): the same as
/// `x.to::<BF16>()`. An infinity or a value that rounds past the largest
/// BFloat16 gives an infinity, and a NaN the canonical NaN (`0x7fff`).
#[track_caller]
pub fn cvt_f32_bf16(src_f32: impl IntoValue<f32>) -> BFloat16 {
    src_f32.into_value().to::<BF16>()
}

/// Two Float32 values to BFloat16, rounded to nearest even and saturated to
/// the finite range (CuTe DSL's `cute.arch.cvt_f32x2_bf16x2`,
/// `cvt.rn.satfinite.bf16x2.f32`): a magnitude above the largest finite
/// BFloat16 (`0x7f7f`, about 3.39e38), infinities included, gives that
/// value with its sign, where [`cvt_f32_bf16`] gives an infinity; a NaN
/// gives the canonical NaN (`0x7fff`). The result is a pair in the form the
/// caller asks for ([`Half2`]): an Int32 holding element 0 in its low half,
/// or a 2-tuple. Needs PTX 8.1, which it raises the kernel's PTX version to
/// (from 7.0 on sm_80, 7.8 on sm_89 and 8.0 on sm_90).
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("to_bf16", || {
///     let x = param_tensor::<f32>("x", Layout::col_major(64));
///     let out = param_tensor::<i32>("out", Layout::col_major(32));
///     let t = thread_idx().x;
///     let pair: Int32 = cvt_f32x2_bf16x2([x.load(2 * t), x.load(2 * t + 1)]);
///     out.store(t, pair);
/// });
/// ```
#[track_caller]
pub fn cvt_f32x2_bf16x2<P: Half2<BF16>>(src_vec2: [Float32; 2]) -> P {
    let _at = LocationScope::new();
    let args = src_vec2.iter().map(|v| v.checked()).collect();
    P::from_packed(cvt(CvtFn::F32x2ToBf16x2, args))
}

/// A Float32 rounded to TF32, for use as an operand of a TF32 MMA (CuTe
/// DSL's `cute.arch.cvt_f32_tf32`, `cvt.rna.tf32.f32`): an Int32 holding the
/// Float32 bits of the nearest value with 10 stored mantissa bits (ties away
/// from zero), the low 13 bits cleared. Values past the largest TF32 value
/// round to infinity. Infinities and NaNs have their low 13 bits cleared, as
/// an SM89 GPU does (the PTX ISA does not say), so a NaN whose payload is
/// only in those bits becomes an infinity. [`Value::to_tf32`] gives the same
/// bits as a Float32, which MMA fragments hold.
#[track_caller]
pub fn cvt_f32_tf32(src_f32: impl IntoValue<f32>) -> Int32 {
    cvt(CvtFn::F32ToTf32, vec![src_f32.into_value().checked()])
}

impl Float32 {
    /// This value rounded to TF32 ([`cvt_f32_tf32`], `cvt.rna.tf32.f32`), as
    /// the Float32 with those bits: what [`MmaTF32Op`](crate::MmaTF32Op)'s
    /// operands hold (the tensor core ignores their low 13 mantissa bits, so
    /// rounding first gives the nearest TF32 product rather than a truncated
    /// one).
    #[track_caller]
    pub fn to_tf32(self) -> Float32 {
        cvt_f32_tf32(self).bitcast()
    }
}

/// `dp4a.s32.s32 word, mask, 0`: the byte of `word` that `mask` selects
/// (one byte of it 1), sign-extended.
#[track_caller]
fn sign_extend_byte(word: Int32, mask: i32) -> Int32 {
    let loc = Location::caller();
    let (mask, zero) = (Int32::constant(mask).checked(), Int32::constant(0).checked());
    let args = vec![word.checked(), mask, zero];
    Value::from_id(builder::with(|b| b.int_inst(IntFn::Dp4a, args, crate::ir::DType::I32, loc)))
}

/// Four Int8 values to Float32 (CuTe DSL's `cute.arch.cvt_i8x4_to_f32x4`):
/// as one 32-bit word, each byte sign-extended with `dp4a.s32.s32` against
/// `0x1`, `0x100`, `0x10000` and `0x1000000` with a zero accumulator, then
/// `cvt.rn.f32.s32`. The values are exact. `src_vec4` is an `[Int8; 4]` or
/// the four packed into an Int32 or Uint32 ([`Int8x4`]).
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("widen", || {
///     let x = param_tensor::<i32>("x", Layout::col_major(32));
///     let out = param_tensor::<f32>("out", Layout::col_major(128));
///     let t = thread_idx().x;
///     for (k, v) in cvt_i8x4_to_f32x4(x.load(t)).into_iter().enumerate() {
///         out.store(t * 4 + k as i32, v);
///     }
/// });
/// ```
#[track_caller]
pub fn cvt_i8x4_to_f32x4(src_vec4: impl Int8x4) -> [Float32; 4] {
    let _at = LocationScope::new();
    let word: Int32 = src_vec4.to_word().bitcast();
    std::array::from_fn(|k| sign_extend_byte(word, 1 << (8 * k)).to::<f32>())
}

/// Two Int8 values to Float32 (CuTe DSL's `cute.arch.cvt_i8x2_to_f32x2`):
/// their 16 bits zero-extended to a word, then as [`cvt_i8x4_to_f32x4`] for
/// the two low bytes. `src_vec2` is an `[Int8; 2]` or the two packed into
/// an Int16 or Uint16 ([`Int8x2`]).
#[track_caller]
pub fn cvt_i8x2_to_f32x2(src_vec2: impl Int8x2) -> [Float32; 2] {
    let _at = LocationScope::new();
    let word: Int32 = src_vec2.to_half_word().bitcast::<u16>().to::<u32>().bitcast();
    std::array::from_fn(|k| sign_extend_byte(word, 1 << (8 * k)).to::<f32>())
}

/// An Int8 to BFloat16, exactly (CuTe DSL's `cute.arch.cvt_i8_bf16`,
/// `cvt.rn.bf16.s8`, sm_90 and PTX 7.8).
#[track_caller]
pub fn cvt_i8_bf16(src_i8: impl IntoValue<i8>) -> BFloat16 {
    cvt(CvtFn::S8ToBf16, vec![src_i8.into_value().checked()])
}

/// Two Int8 values to BFloat16, exactly (CuTe DSL's
/// `cute.arch.cvt_i8x2_to_bf16x2`): their 16 bits as an `.s2f6x2` pair
/// (each byte its int8 value times 2^-6) converted by
/// `cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2` with the scale factor
/// `0x8585` (2^6 for both), which gives the int8 values. The instruction
/// is architecture-specific: sm_100a, sm_103a, sm_107a, sm_110a, sm_120a
/// and sm_121a, from PTX 9.1 (CUDA 13.1), which only LLVM 23 or newer and
/// `ptxas` 13.1 or newer handle; [`cvt_i8_bf16_intrinsic`] converts on every
/// target. The result is a pair in the form the caller asks for ([`Half2`]).
#[track_caller]
pub fn cvt_i8x2_to_bf16x2<P: Half2<BF16>>(src_vec2: impl Int8x2) -> P {
    let _at = LocationScope::new();
    P::from_packed(cvt(CvtFn::S2f6x2ToBf16x2 { scale: 0x85 }, vec![src_vec2.to_half_word().checked()]))
}

/// Four Int8 values to BFloat16, exactly (CuTe DSL's
/// `cute.arch.cvt_i8x4_to_bf16x4`): [`cvt_i8x2_to_bf16x2`] on each half of
/// the word, with its targets.
#[track_caller]
pub fn cvt_i8x4_to_bf16x4(src_vec4: impl Int8x4) -> [BFloat16; 4] {
    let _at = LocationScope::new();
    let [low, high]: [Int16; 2] = unpack(src_vec4.to_word());
    let (a, b): (BFloat16, BFloat16) = cvt_i8x2_to_bf16x2(low);
    let (c, d): (BFloat16, BFloat16) = cvt_i8x2_to_bf16x2(high);
    [a, b, c, d]
}

/// How [`cvt_i8_bf16_intrinsic_with`] converts. CuTe DSL picks by the
/// architecture it compiles for; cute-rs kernels are built before a target
/// is chosen, so the Ampere path is the default on every target and the
/// other an explicit choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum I8ToBf16Path {
    /// CuTe DSL's path from Ampere on, for every target: four values at a
    /// time sign-extended with `dp4a` and converted with `cvt.rn.f32.s32`
    /// ([`cvt_i8x4_to_f32x4`]), then `cvt.rn.satfinite.bf16x2.f32` per pair
    /// ([`cvt_f32x2_bf16x2`]); two left over the same way
    /// ([`cvt_i8x2_to_f32x2`]); one left over through Float32 (`sitofp`,
    /// then `cvt.rn.bf16.f32`). Needs PTX 8.1 (`.satfinite`).
    #[default]
    Ampere,
    /// CuTe DSL's path on sm_100a, sm_103a, sm_107a, sm_110a, sm_120a and
    /// sm_121a with CUDA 13.1: [`cvt_i8x4_to_bf16x4`] four at a time,
    /// [`cvt_i8x2_to_bf16x2`] for two left over and [`cvt_i8_bf16`]
    /// (`cvt.rn.bf16.s8`) for one. The kernel then needs one of those
    /// targets and PTX 9.1 (a `length` of 1 converts with `cvt.rn.bf16.s8`
    /// alone, which needs sm_90 and PTX 7.8).
    S2f6x2,
}

/// The first `length` values of `vec_i8` converted to BFloat16, exactly
/// (CuTe DSL's `cute.arch.cvt_i8_bf16_intrinsic(vec_i8, length)`), on the
/// Ampere path, which every target has ([`I8ToBf16Path::Ampere`]). `length`
/// is at least 1 and at most `vec_i8.len()`.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("dequantize", || {
///     let q = param_tensor::<i8>("q", Layout::col_major(256));
///     let out = param_tensor::<BF16>("out", Layout::col_major(256));
///     let t = thread_idx().x;
///     let values: Vec<Int8> = (0..8).map(|k| q.load(t * 8 + k)).collect();
///     for (k, v) in cvt_i8_bf16_intrinsic(&values, 8).into_iter().enumerate() {
///         out.store(t * 8 + k as i32, v);
///     }
/// });
/// ```
#[track_caller]
pub fn cvt_i8_bf16_intrinsic(vec_i8: &[Int8], length: usize) -> Vec<BFloat16> {
    cvt_i8_bf16_intrinsic_with(vec_i8, length, I8ToBf16Path::Ampere)
}

/// [`cvt_i8_bf16_intrinsic`] on the given path.
#[track_caller]
pub fn cvt_i8_bf16_intrinsic_with(vec_i8: &[Int8], length: usize, path: I8ToBf16Path) -> Vec<BFloat16> {
    assert!(
        (1..=vec_i8.len()).contains(&length),
        "cute: cvt_i8_bf16_intrinsic: length is {length}, but it must be from 1 to vec_i8.len() ({})",
        vec_i8.len()
    );
    let _at = LocationScope::new();
    let mut out = Vec::with_capacity(length);
    let mut rest = &vec_i8[..length];
    while rest.len() >= 4 {
        let quad: [Int8; 4] = [rest[0], rest[1], rest[2], rest[3]];
        match path {
            I8ToBf16Path::Ampere => {
                let [a, b, c, d] = cvt_i8x4_to_f32x4(quad);
                let (e0, e1): (BFloat16, BFloat16) = cvt_f32x2_bf16x2([a, b]);
                let (e2, e3): (BFloat16, BFloat16) = cvt_f32x2_bf16x2([c, d]);
                out.extend([e0, e1, e2, e3]);
            }
            I8ToBf16Path::S2f6x2 => out.extend(cvt_i8x4_to_bf16x4(quad)),
        }
        rest = &rest[4..];
    }
    if rest.len() >= 2 {
        let pair = [rest[0], rest[1]];
        let (e0, e1): (BFloat16, BFloat16) = match path {
            I8ToBf16Path::Ampere => cvt_f32x2_bf16x2(cvt_i8x2_to_f32x2(pair)),
            I8ToBf16Path::S2f6x2 => cvt_i8x2_to_bf16x2(pair),
        };
        out.extend([e0, e1]);
        rest = &rest[2..];
    }
    if let [last] = rest {
        out.push(match path {
            I8ToBf16Path::Ampere => cvt_f32_bf16(last.to::<i32>().to::<f32>()),
            I8ToBf16Path::S2f6x2 => cvt_i8_bf16(*last),
        });
    }
    out
}

/// Four int4 values held one per byte, in the low nibble with the high
/// nibble zero, sign-extended to Int8 (CuTe DSL's
/// `cute.arch.sext_unpacked_i4x4_to_i8x4`): `(x + 0x78787878) ^ 0x78787878`
/// on the 32-bit word, which never carries from one byte into the next.
/// `src_vec4` is an `[Int8; 4]` or the word as an Int32 or Uint32
/// ([`Int8x4`]), and the result has its form.
#[track_caller]
pub fn sext_unpacked_i4x4_to_i8x4<V: Int8x4>(src_vec4: V) -> V {
    let _at = LocationScope::new();
    let word = src_vec4.to_word();
    V::from_word((word + 0x7878_7878u32) ^ 0x7878_7878u32)
}

/// The first `length` unpacked int4 values of `vec_unpacked_i4` (one per
/// byte, in the low nibble) sign-extended to Int8, four at a time with
/// [`sext_unpacked_i4x4_to_i8x4`] (CuTe DSL's
/// `cute.arch.sext_unpacked_i4_i8_intrinsic(vec, length)`). `length` is a
/// positive multiple of 4, at most `vec_unpacked_i4.len()`.
#[track_caller]
pub fn sext_unpacked_i4_i8_intrinsic(vec_unpacked_i4: &[Int8], length: usize) -> Vec<Int8> {
    assert!(
        length > 0 && length.is_multiple_of(4) && length <= vec_unpacked_i4.len(),
        "cute: sext_unpacked_i4_i8_intrinsic: length is {length}, but it must be a multiple of 4 from 4 to \
         vec_unpacked_i4.len() ({})",
        vec_unpacked_i4.len()
    );
    let _at = LocationScope::new();
    vec_unpacked_i4[..length].chunks(4).flat_map(|c| sext_unpacked_i4x4_to_i8x4([c[0], c[1], c[2], c[3]])).collect()
}

/// An Int4 to BFloat16, exactly (CuTe DSL's `cute.arch.cvt_i4_bf16`):
/// sign-extended to Int32, converted to Float32 (`sitofp`) and rounded to
/// BFloat16 (`cvt.rn.bf16.f32`).
#[track_caller]
pub fn cvt_i4_bf16(src_i4: crate::numeric::Int4) -> BFloat16 {
    let _at = LocationScope::new();
    cvt_f32_bf16(src_i4.to::<i32>().to::<f32>())
}

/// The bits of `values` (element 0 in the low nibble) zero-extended to a
/// word: CuTe DSL's bitcast of a `vector<Nxi4>` and `extui` to `i32`.
#[track_caller]
fn int4_word<const N: usize>(values: [crate::numeric::Int4; N]) -> Uint32 {
    match N {
        2 => {
            let byte: crate::value::Uint8 = pack([values[0], values[1]]);
            byte.to::<u32>()
        }
        4 => {
            let half: Uint16 = pack([values[0], values[1], values[2], values[3]]);
            half.to::<u32>()
        }
        8 => pack(std::array::from_fn::<_, 8, _>(|k| values[k])),
        _ => unreachable!("2, 4 or 8 int4 values"),
    }
}

/// The int4-to-BFloat16 conversion of CuTe DSL's `cvt_i4_to_bf16_impl`, on
/// `num_elts` (2, 4 or 8) int4 values in `src` (the upper nibbles zero):
/// `lop3` (LUT `0x6A`, `(a & b) ^ c`) biases each even nibble to `x ^ 8` in
/// its byte, and each odd nibble of `src >> 4` likewise, setting the top
/// bit of the last byte too; `prmt` with selectors `0xF4F0` to `0xF7F3`
/// puts element `2k` and `2k + 1` in the low byte of the halves of word `k`,
/// with `0xFF` above them (the sign of that top bit) and the top bit cleared
/// in the last word; and `fma.rn.bf16x2` with `0x83808380` (-2^-120) and
/// `0xC308C308` (-136) maps the BFloat16 `-(2^127 + 2^120 b)` of each byte
/// `b` to `b - 8`, the int4 value. The result is `num_elts / 2` words of
/// BFloat16 pairs (element `2k` in the low half of word `k`).
#[track_caller]
fn cvt_i4_to_bf16_impl(src: Uint32, num_elts: usize) -> Vec<Int32> {
    let shifted = src >> 4u32;
    let biased = super::lop3::<u32>(src, 0x0F0F_0F0Fu32, 0x0808_0808u32, 0x6A);
    let shifted = super::lop3::<u32>(shifted, 0x0F0F_0F0Fu32, 0x8808_0808u32, 0x6A);
    let mut words: Vec<Uint32> = [0xF4F0u32, 0xF5F1, 0xF6F2, 0xF7F3][..num_elts / 2]
        .iter()
        .map(|&sel| super::prmt::<u32>(biased, shifted, sel))
        .collect();
    let last = words.len() - 1;
    words[last] = words[last] & 0xFF7F_FFFFu32;
    let (mul, bias) = (Int32::from(0x8380_8380u32 as i32), Int32::from(0xC308_C308u32 as i32));
    words
        .into_iter()
        .map(|w| super::half::fma_packed_bf16x2::<Int32>(w.bitcast(), mul, bias, super::half::HalfOptions::default()))
        .collect()
}

/// The `with_shuffle` int4-to-BFloat16 conversion of CuTe DSL's
/// `cvt_i4_to_bf16_with_shuffle_impl`: the high nibble of each byte (`src &
/// 0xF0F0F0F0`) and the low one moved up (`(src & 0x0F0F0F0F) << 4`) are
/// int8 values 16 times the int4 ones, which the scaled `.s2f6x2` cvt with
/// the scale `0x81` (2^2, times s2f6's 2^-6) converts exactly, a pair per
/// half word. For 8 values in the order `(0, 2, 1, 3, 4, 6, 5, 7)` (4 values
/// to `(0, 2, 1, 3)`) the words are the BFloat16 pairs in natural order; 2
/// values are not shuffled, and a `prmt` (`0x5410`) joins their halves.
#[track_caller]
fn cvt_i4_to_bf16_with_shuffle_impl(src: Uint32, num_elts: usize) -> Vec<Int32> {
    let odd = src & 0xF0F0_F0F0u32;
    let even = (src & 0x0F0F_0F0Fu32) << 4u32;
    let pairs = |w: Uint32| -> [Int32; 2] {
        let [lo, hi]: [Int16; 2] = unpack(w);
        [lo, hi].map(|h| cvt(CvtFn::S2f6x2ToBf16x2 { scale: 0x81 }, vec![h.checked()]))
    };
    let [r02, r46] = pairs(even);
    let [r13, r57] = pairs(odd);
    match num_elts {
        2 => vec![super::prmt::<i32>(r02, r13, 0x5410)],
        4 => vec![r02, r13],
        _ => vec![r02, r13, r46, r57],
    }
}

/// `num_elts` int4 values to BFloat16 pairs, on the chosen path.
#[track_caller]
fn cvt_i4_bf16_words<const N: usize>(src: [crate::numeric::Int4; N], with_shuffle: bool) -> Vec<BFloat16> {
    let _at = LocationScope::new();
    let word = int4_word(src);
    let words = if with_shuffle { cvt_i4_to_bf16_with_shuffle_impl(word, N) } else { cvt_i4_to_bf16_impl(word, N) };
    words
        .into_iter()
        .flat_map(|w| {
            let (lo, hi): (BFloat16, BFloat16) = Half2::<BF16>::from_packed(w);
            [lo, hi]
        })
        .collect()
}

/// Two Int4 values to BFloat16, exactly (CuTe DSL's
/// `cute.arch.cvt_i4x2_to_bf16x2(src_vec2, with_shuffle=)`), in the pair
/// form the caller asks for ([`Half2`]).
///
/// The default path is `lop3`, `prmt` and `fma.rn.bf16x2` (sm_80, PTX 7.0;
/// see [`cvt_i4_bf16_intrinsic`]). With `with_shuffle` it is the scaled
/// `.s2f6x2` cvt, which needs sm_100a, sm_103a, sm_107a, sm_110a, sm_120a or
/// sm_121a and PTX 9.1 (CUDA 13.1, LLVM 23), as [`cvt_i8x2_to_bf16x2`]; two
/// values are not shuffled.
#[track_caller]
pub fn cvt_i4x2_to_bf16x2<P: Half2<BF16>>(src_vec2: [crate::numeric::Int4; 2], with_shuffle: bool) -> P {
    let v = cvt_i4_bf16_words(src_vec2, with_shuffle);
    let _at = LocationScope::new();
    P::from_packed(pack([v[0], v[1]]))
}

/// Four Int4 values to BFloat16, exactly (CuTe DSL's
/// `cute.arch.cvt_i4x4_to_bf16x4(src_vec4, with_shuffle=)`). With
/// `with_shuffle` the input is in the order `(0, 2, 1, 3)` and the result in
/// natural order; the paths are those of [`cvt_i4x2_to_bf16x2`].
#[track_caller]
pub fn cvt_i4x4_to_bf16x4(src_vec4: [crate::numeric::Int4; 4], with_shuffle: bool) -> [BFloat16; 4] {
    let v = cvt_i4_bf16_words(src_vec4, with_shuffle);
    [v[0], v[1], v[2], v[3]]
}

/// Eight Int4 values (a 32-bit word of them) to BFloat16, exactly (CuTe
/// DSL's `cute.arch.cvt_i4x8_to_bf16x8(src_vec8, with_shuffle=)`). With
/// `with_shuffle` the input is in the order `(0, 2, 1, 3, 4, 6, 5, 7)` and
/// the result in natural order; the paths are those of
/// [`cvt_i4x2_to_bf16x2`].
#[track_caller]
pub fn cvt_i4x8_to_bf16x8(src_vec8: [crate::numeric::Int4; 8], with_shuffle: bool) -> [BFloat16; 8] {
    let v = cvt_i4_bf16_words(src_vec8, with_shuffle);
    std::array::from_fn(|k| v[k])
}

/// The first `length` Int4 values of `vec_i4` converted to BFloat16,
/// exactly (CuTe DSL's `cute.arch.cvt_i4_bf16_intrinsic(vec_i4, length,
/// with_shuffle=)`): eight at a time with [`cvt_i4x8_to_bf16x8`], then four,
/// two and one ([`cvt_i4_bf16`]) left over. With `with_shuffle` every group
/// of eight (and a trailing group of four) is in the shuffled order those
/// functions describe, and the result is in natural order.
///
/// CuTe DSL allows this function only on sm_100a-class targets, though its
/// default path needs nothing newer than sm_80 (`fma.rn.bf16x2`); cute-rs
/// allows it on every target, and the `with_shuffle` path on the targets of
/// the scaled `.s2f6x2` cvt with PTX 9.1. `length` is at least 1 and at
/// most `vec_i4.len()`.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("dequantize_int4", || {
///     let q = param_tensor::<u32>("q", Layout::col_major(32));
///     let out = param_tensor::<BF16>("out", Layout::col_major(256));
///     let t = thread_idx().x;
///     let values: [Int4; 8] = unpack(q.load(t));
///     for (k, v) in cvt_i4_bf16_intrinsic(&values, 8, false).into_iter().enumerate() {
///         out.store(t * 8 + k as i32, v);
///     }
/// });
/// ```
#[track_caller]
pub fn cvt_i4_bf16_intrinsic(vec_i4: &[crate::numeric::Int4], length: usize, with_shuffle: bool) -> Vec<BFloat16> {
    assert!(
        (1..=vec_i4.len()).contains(&length),
        "cute: cvt_i4_bf16_intrinsic: length is {length}, but it must be from 1 to vec_i4.len() ({})",
        vec_i4.len()
    );
    let _at = LocationScope::new();
    let mut out = Vec::with_capacity(length);
    let mut rest = &vec_i4[..length];
    while rest.len() >= 8 {
        out.extend(cvt_i4x8_to_bf16x8(std::array::from_fn(|k| rest[k]), with_shuffle));
        rest = &rest[8..];
    }
    if rest.len() >= 4 {
        out.extend(cvt_i4x4_to_bf16x4(std::array::from_fn(|k| rest[k]), with_shuffle));
        rest = &rest[4..];
    }
    if rest.len() >= 2 {
        let (a, b): (BFloat16, BFloat16) = cvt_i4x2_to_bf16x2([rest[0], rest[1]], with_shuffle);
        out.extend([a, b]);
        rest = &rest[2..];
    }
    if let [last] = rest {
        out.push(cvt_i4_bf16(*last));
    }
    out
}

/// The E4M3 bytes of the int4 nibbles of `src`, four per word (CuTe DSL's
/// `cvt_i4x8_to_mxf8x8_impl`): for each half of `src`, a `prmt` looks up
/// each nibble's low 3 bits in the table of E4M3 encodings of 0 to 7
/// (`0x44403800`, `0x4E4C4A48`), another in that of -8 to -1 (`0xCACCCED0`,
/// `0xB8C0C4C8`), and a third picks between them by the nibble's sign bit
/// (`(src & 0x88888888) >> 1`, or-ed into the selector `0x32103210`).
#[track_caller]
fn cvt_i4x8_to_mxf8x8_impl(src: Uint32) -> [Uint32; 2] {
    let mut sign = (src & 0x8888_8888u32) >> 1u32;
    let mut index = src & 0x7777_7777u32;
    std::array::from_fn(|_| {
        let selector = sign | 0x3210_3210u32;
        let pos = super::prmt::<u32>(0x4440_3800u32, 0x4E4C_4A48u32, index);
        let neg = super::prmt::<u32>(0xCACC_CED0u32, 0xB8C0_C4C8u32, index);
        let word = super::prmt::<u32>(pos, neg, selector);
        sign = sign >> 16u32;
        index = index >> 16u32;
        word
    })
}

/// Eight Int4 values to FP8 bytes, exactly (CuTe DSL's
/// `cute.arch.cvt_i4x8_to_mxf8x8(src_vec8, dst_type=)`): every int4 value
/// is an E4M3 value, whose encoding three `prmt`s per four values look up
/// (sm_20; it runs on every target). `F` is the 8-bit type the bytes are
/// read as, typically [`F8E4M3FN`](crate::F8E4M3FN) (CuTe DSL's callers
/// often take [`Int8`] and bitcast later); the bytes are E4M3 encodings
/// whatever `F` is. The operands of MXF8 block-scaled MMAs.
#[track_caller]
pub fn cvt_i4x8_to_mxf8x8<F: crate::value::Numeric>(src_vec8: [crate::numeric::Int4; 8]) -> [Value<F>; 8] {
    assert_eq!(F::DTYPE.bits(), 8, "cute: cvt_i4x8_to_mxf8x8: dst_type must be an 8-bit type, not {}", F::DTYPE);
    let _at = LocationScope::new();
    let words = cvt_i4x8_to_mxf8x8_impl(int4_word(src_vec8));
    let bytes: Vec<Int8> = words.into_iter().flat_map(|w| unpack::<i8, u32, 4>(w)).collect();
    std::array::from_fn(|k| bytes[k].bitcast())
}

/// The first `length` Int4 values of `vec_i4` as FP8 bytes, eight at a time
/// with [`cvt_i4x8_to_mxf8x8`] (CuTe DSL's
/// `cute.arch.cvt_i4_mxf8_intrinsic(vec_i4, length, dst_type)`). `length` is
/// a positive multiple of 8, at most `vec_i4.len()`.
#[track_caller]
pub fn cvt_i4_mxf8_intrinsic<F: crate::value::Numeric>(
    vec_i4: &[crate::numeric::Int4],
    length: usize,
) -> Vec<Value<F>> {
    assert!(
        length > 0 && length.is_multiple_of(8) && length <= vec_i4.len(),
        "cute: cvt_i4_mxf8_intrinsic: length is {length}; it converts groups of 8 values, so it takes 8, 16, ... up to \
         vec_i4.len() ({})",
        vec_i4.len()
    );
    let _at = LocationScope::new();
    vec_i4[..length].chunks(8).flat_map(|c| cvt_i4x8_to_mxf8x8::<F>(std::array::from_fn(|k| c[k]))).collect()
}

/// The FP8 formats PTX converts on sm_89 and newer: [`F8E4M3FN`](crate::F8E4M3FN)
/// (`.e4m3`) and [`F8E5M2`](crate::F8E5M2) (`.e5m2`).
pub trait Fp8Format: crate::value::Numeric {
    #[doc(hidden)]
    const KIND: crate::ir::Fp8Kind;
}

impl Fp8Format for crate::numeric::F8E4M3FN {
    const KIND: crate::ir::Fp8Kind = crate::ir::Fp8Kind::E4M3;
}

impl Fp8Format for crate::numeric::F8E5M2 {
    const KIND: crate::ir::Fp8Kind = crate::ir::Fp8Kind::E5M2;
}

/// The two bytes of a 16-bit cvt result as 8-bit floats (element 0 low).
#[track_caller]
fn bytes_of<F: crate::value::Numeric>(packed: Int16) -> [Value<F>; 2] {
    let [lo, hi]: [Int8; 2] = unpack(packed);
    [lo.bitcast(), hi.bitcast()]
}

/// Two 8-bit floats as the 16-bit operand of a cvt (element 0 low).
#[track_caller]
fn bytes_to<F: crate::value::Numeric>(pair: [Value<F>; 2]) -> Int16 {
    pack([pair[0].bitcast::<i8>(), pair[1].bitcast::<i8>()])
}

/// Two Float32 values to FP8 (`cvt.rn.satfinite{.relu}.{e4m3x2,e5m2x2}.f32`):
/// rounded to nearest even, a magnitude past the format's largest finite
/// value (infinities included) saturating to it with its sign, a NaN giving
/// the format's NaN, and with `relu` a negative result giving `+0`. Needs
/// sm_89 (PTX 8.1 there, 7.8 on sm_90 and newer). `x.to::<F8E4M3FN>()` is
/// the scalar form.
///
/// ```
/// use cute::prelude::*;
/// let k = Kernel::build("to_fp8", || {
///     let x = param_tensor::<f32>("x", Layout::col_major(64));
///     let out = param_tensor::<F8E4M3FN>("out", Layout::col_major(64));
///     let t = thread_idx().x;
///     let [a, b] = cvt_f32x2_to_f8x2::<F8E4M3FN>([x.load(2 * t), x.load(2 * t + 1)], false);
///     out.store(2 * t, a);
///     out.store(2 * t + 1, b);
/// });
/// assert_eq!(cute::validate::check_target(k.ir(), Arch::Sm89).unwrap(), 81);
/// ```
#[track_caller]
pub fn cvt_f32x2_to_f8x2<F: Fp8Format>(src_vec2: [Float32; 2], relu: bool) -> [Value<F>; 2] {
    let _at = LocationScope::new();
    let args = src_vec2.iter().map(|v| v.checked()).collect();
    bytes_of(cvt(CvtFn::F32x2ToF8x2 { to: F::KIND, relu }, args))
}

/// A pair of Float16 values (packed in an Int32, element 0 low, or a
/// 2-tuple) to FP8 (`cvt.rn.satfinite{.relu}.{e4m3x2,e5m2x2}.f16x2`), as
/// [`cvt_f32x2_to_f8x2`] rounds.
#[track_caller]
pub fn cvt_f16x2_to_f8x2<F: Fp8Format>(src: impl Half2<crate::value::F16>, relu: bool) -> [Value<F>; 2] {
    let _at = LocationScope::new();
    let packed = src.to_packed().checked();
    bytes_of(cvt(CvtFn::F16x2ToF8x2 { to: F::KIND, relu }, vec![packed]))
}

/// A pair of FP8 values to Float16, exactly (`cvt.rn{.relu}.f16x2.{e4m3x2,
/// e5m2x2}`; with `relu` a negative value gives `+0` and a NaN the
/// canonical NaN), in the pair form the caller asks for.
#[track_caller]
pub fn cvt_f8x2_to_f16x2<F: Fp8Format, P: Half2<crate::value::F16>>(src: [Value<F>; 2], relu: bool) -> P {
    let _at = LocationScope::new();
    let packed = bytes_to(src).checked();
    P::from_packed(cvt(CvtFn::F8x2ToF16x2 { from: F::KIND, relu }, vec![packed]))
}

/// Two Float32 values to the UE8M0 scale factor
/// (`cvt.{rz,rp}.satfinite.ue8m0x2.f32`; `round_up` picks `.rp`): the sign
/// is ignored, a magnitude past `2^127` saturates to it and a NaN gives
/// `0xFF`. sm_100a, sm_101a and sm_120a (PTX 8.6), or their families from
/// PTX 8.8. `x.to::<F8E8M0FNU>()` rounds up, with integer operations
/// instead (so it runs on every target).
#[track_caller]
pub fn cvt_f32x2_to_ue8m0x2(src_vec2: [Float32; 2], round_up: bool) -> [crate::numeric::Float8E8M0FNU; 2] {
    let _at = LocationScope::new();
    let args = src_vec2.iter().map(|v| v.checked()).collect();
    bytes_of(cvt(CvtFn::F32x2ToUe8m0x2 { round_up }, args))
}

/// Four Float32 values to UE5M3 (CuTe DSL's `cute.arch.cvt_f32x4_to_fnv8e5m3x4`): two `cvt.rp.satfinite.ue5m3x2.f32`,
/// each rounding a pair toward +infinity (as scale factors are rounded up)
/// and giving the 8-bit values in order, element 0 first. UE5M3 is unsigned
/// (5 exponent, 3 mantissa bits; no infinity, `0xff` the only NaN), so the
/// sign is ignored, a magnitude past the largest finite value (infinities
/// included) saturates to it, and a NaN gives `0xff`. PTX ISA 9.4 (the
/// sm_107f family): no installed LLVM or ptxas has it, so it can be checked
/// in the simulator and as MLIR text only.
#[track_caller]
pub fn cvt_f32x4_to_fnv8e5m3x4(src_vec_f32: [Float32; 4]) -> [crate::numeric::FloatNV8E5M3FNU; 4] {
    let _at = LocationScope::new();
    let pair = |a: &Float32, b: &Float32| bytes_of(cvt(CvtFn::F32x2ToUe5m3x2, vec![a.checked(), b.checked()]));
    let [a, b] = pair(&src_vec_f32[0], &src_vec_f32[1]);
    let [c, d] = pair(&src_vec_f32[2], &src_vec_f32[3]);
    [a, b, c, d]
}

/// A pair of BFloat16 values to UE8M0
/// (`cvt.{rz,rp}{.satfinite}.ue8m0x2.bf16x2`); without `satfinite` a value
/// past `2^127` (an infinity included) gives NaN. The targets of
/// [`cvt_f32x2_to_ue8m0x2`]. (The PTX ISA's `.relu` form is left out:
/// ptxas 12.8 refuses it.)
#[track_caller]
pub fn cvt_bf16x2_to_ue8m0x2(
    src: impl Half2<BF16>,
    round_up: bool,
    satfinite: bool,
) -> [crate::numeric::Float8E8M0FNU; 2] {
    let _at = LocationScope::new();
    let packed = src.to_packed().checked();
    bytes_of(cvt(CvtFn::Bf16x2ToUe8m0x2 { round_up, satfinite }, vec![packed]))
}

/// A pair of UE8M0 scale factors to BFloat16, exactly
/// (`cvt.rn.bf16x2.ue8m0x2`; `2^-127` is a BFloat16 subnormal). The targets
/// of [`cvt_f32x2_to_ue8m0x2`].
#[track_caller]
pub fn cvt_ue8m0x2_to_bf16x2<P: Half2<BF16>>(src: [crate::numeric::Float8E8M0FNU; 2]) -> P {
    let _at = LocationScope::new();
    let packed = bytes_to(src).checked();
    P::from_packed(cvt(CvtFn::Ue8m0x2ToBf16x2, vec![packed]))
}

/// The FP6 formats of the FP6 pair conversions ([`cvt_f32x2_to_f6x2`],
/// [`cvt_f6x2_to_f16x2`]): Float6E2M3FN and Float6E3M2FN.
pub trait Fp6Format: crate::value::Numeric {
    #[doc(hidden)]
    const KIND: crate::ir::MxKind;
}

impl Fp6Format for crate::numeric::F6E2M3FN {
    const KIND: crate::ir::MxKind = crate::ir::MxKind::E2M3;
}

impl Fp6Format for crate::numeric::F6E3M2FN {
    const KIND: crate::ir::MxKind = crate::ir::MxKind::E3M2;
}

/// Two Float32 values to FP4 E2M1 (`cvt.rn.satfinite{.relu}.e2m1x2.f32`):
/// rounded to nearest even, a magnitude past 6 (infinities included)
/// saturating to ±6, a NaN giving +6 (E2M1 has no NaN), and with `relu` a
/// negative result giving `+0`. Float4E2M1FN is not a `to()` destination
/// (CuTe DSL refuses it), so values convert in pairs. sm_100a, sm_101a and
/// sm_120a (PTX 8.6), or their families from PTX 8.8.
///
/// ```
/// use cute::prelude::*;
/// let k = Kernel::build("to_fp4", || {
///     let x = param_tensor::<f32>("x", Layout::col_major(16));
///     let out = param_tensor::<F4E2M1FN>("out", Layout::col_major(16));
///     let frag = make_fragment::<F4E2M1FN>(8);
///     for i in 0..4 {
///         let [a, b] = cvt_f32x2_to_f4e2m1x2([x.load(2 * i), x.load(2 * i + 1)], false);
///         frag.store(2 * i, a);
///         frag.store(2 * i + 1, b);
///     }
///     autovec_copy(&frag, &local_tile(&out, 8, 0));
/// });
/// assert_eq!(cute::validate::check_target(k.ir(), Arch::Sm100a).unwrap(), 86);
/// assert!(cute::validate::check_target(k.ir(), Arch::Sm90).is_err());
/// ```
#[track_caller]
pub fn cvt_f32x2_to_f4e2m1x2(src_vec2: [Float32; 2], relu: bool) -> [crate::numeric::Float4E2M1FN; 2] {
    let _at = LocationScope::new();
    let args = src_vec2.iter().map(|v| v.checked()).collect();
    let packed: Int16 = cvt(CvtFn::F32x2ToMx { to: crate::ir::MxKind::E2M1, relu }, args);
    unpack(packed)
}

/// Two Float32 values to FP6 (`cvt.rn.satfinite{.relu}.{e2m3x2,e3m2x2}.f32`),
/// as [`cvt_f32x2_to_f4e2m1x2`] rounds and saturates (to ±7.5 for E2M3, ±28
/// for E3M2; a NaN gives the positive one). The targets of
/// [`cvt_f32x2_to_f4e2m1x2`].
#[track_caller]
pub fn cvt_f32x2_to_f6x2<F: Fp6Format>(src_vec2: [Float32; 2], relu: bool) -> [Value<F>; 2] {
    let _at = LocationScope::new();
    let args = src_vec2.iter().map(|v| v.checked()).collect();
    let packed: Int16 = cvt(CvtFn::F32x2ToMx { to: F::KIND, relu }, args);
    let bytes: [crate::value::Uint8; 2] = unpack(packed);
    bytes.map(|b| unpack::<F, u8, 1>(b)[0])
}

/// An FP4 pair as the 16-bit operand of the E2M1 cvt: its low byte, lane 0
/// in the low nibble.
#[track_caller]
fn fp4_pair(src: [crate::numeric::Float4E2M1FN; 2]) -> ValueId {
    let packed: Int16 = pack(src);
    packed.checked()
}

/// Two FP4 E2M1 values to Float16, exactly (CuTe DSL's
/// `cute.arch.cvt_f4e2m1x2_to_f16x2`, `cvt.rn.f16x2.e2m1x2`), lane 0 in the
/// low half, in the pair form the caller asks for. The targets of
/// [`cvt_f32x2_to_f4e2m1x2`] (PTX 8.6).
#[track_caller]
pub fn cvt_f4e2m1x2_to_f16x2<P: Half2<crate::value::F16>>(src_vec2: [crate::numeric::Float4E2M1FN; 2]) -> P {
    cvt_f4e2m1x2_to_f16x2_with(src_vec2, false)
}

/// [`cvt_f4e2m1x2_to_f16x2`] with `.relu` when `relu` (a negative value
/// gives `+0`).
#[track_caller]
pub fn cvt_f4e2m1x2_to_f16x2_with<P: Half2<crate::value::F16>>(
    src_vec2: [crate::numeric::Float4E2M1FN; 2],
    relu: bool,
) -> P {
    let _at = LocationScope::new();
    let packed = fp4_pair(src_vec2);
    P::from_packed(cvt(CvtFn::MxToF16x2 { from: crate::ir::MxKind::E2M1, relu }, vec![packed]))
}

/// One FP4 E2M1 value to Float16 (CuTe DSL's `cute.arch.cvt_f4e2m1_f16`):
/// [`cvt_f4e2m1x2_to_f16x2`] of the value and a zero, keeping lane 0.
#[track_caller]
pub fn cvt_f4e2m1_f16(src: crate::numeric::Float4E2M1FN) -> crate::value::Float16 {
    let zero = Value::constant(crate::numeric::F4E2M1FN(0));
    let (lo, _): (crate::value::Float16, crate::value::Float16) = cvt_f4e2m1x2_to_f16x2([src, zero]);
    lo
}

/// Four FP4 E2M1 values to Float16 (CuTe DSL's
/// `cute.arch.cvt_f4e2m1x4_to_f16x4`): two `cvt.rn.f16x2.e2m1x2`.
#[track_caller]
pub fn cvt_f4e2m1x4_to_f16x4(src_vec4: [crate::numeric::Float4E2M1FN; 4]) -> [crate::value::Float16; 4] {
    let mut out = Vec::with_capacity(4);
    for pair in src_vec4.chunks(2) {
        let (lo, hi): (crate::value::Float16, crate::value::Float16) = cvt_f4e2m1x2_to_f16x2([pair[0], pair[1]]);
        out.extend([lo, hi]);
    }
    out.try_into().unwrap_or_else(|_| unreachable!())
}

/// Eight FP4 E2M1 values (a 32-bit word of them) to Float16 (CuTe DSL's
/// `cute.arch.cvt_f4e2m1x8_to_f16x8`): four `cvt.rn.f16x2.e2m1x2`.
#[track_caller]
pub fn cvt_f4e2m1x8_to_f16x8(src_vec8: [crate::numeric::Float4E2M1FN; 8]) -> [crate::value::Float16; 8] {
    let mut out = Vec::with_capacity(8);
    for pair in src_vec8.chunks(2) {
        let (lo, hi): (crate::value::Float16, crate::value::Float16) = cvt_f4e2m1x2_to_f16x2([pair[0], pair[1]]);
        out.extend([lo, hi]);
    }
    out.try_into().unwrap_or_else(|_| unreachable!())
}

/// Two FP6 values to Float16, exactly (`cvt.rn{.relu}.f16x2.{e2m3x2,
/// e3m2x2}`; with `relu` a negative value gives `+0`), lane 0 in the low
/// half. The targets of [`cvt_f32x2_to_f4e2m1x2`].
#[track_caller]
pub fn cvt_f6x2_to_f16x2<F: Fp6Format, P: Half2<crate::value::F16>>(src_vec2: [Value<F>; 2], relu: bool) -> P {
    let _at = LocationScope::new();
    let bytes: [crate::value::Uint8; 2] = src_vec2.map(|v| pack::<F, u8, 1>([v]));
    let packed: Int16 = pack(bytes);
    P::from_packed(cvt(CvtFn::MxToF16x2 { from: F::KIND, relu }, vec![packed.checked()]))
}
