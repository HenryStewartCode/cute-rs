// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! Half-precision operations, packing and the conversion kernels and sweeps.

use super::*;
// ---------------------------------------------------------------------------
// Pack and Unpack, the packed and scalar half-precision operations, packed
// exp2 and the narrow conversions.
// ---------------------------------------------------------------------------

/// A reference for the half-precision instructions in plain Rust, from the
/// PTX ISA: a 16-bit float format with `p` significant bits, normal
/// exponents `emin..=emax`.
pub struct HalfFormat {
    pub p: i32,
    pub emin: i32,
    pub emax: i32,
    pub bf16: bool,
}

/// Float16.
pub const F16_FORMAT: HalfFormat = HalfFormat { p: 11, emin: -14, emax: 15, bf16: false };

/// BFloat16.
pub const BF16_FORMAT: HalfFormat = HalfFormat { p: 8, emin: -126, emax: 127, bf16: true };

impl HalfFormat {
    pub fn value(&self, bits: u16) -> f64 {
        if self.bf16 { f32::from_bits((bits as u32) << 16) as f64 } else { F16(bits).to_f32() as f64 }
    }

    /// The bits of a value the format holds exactly.
    pub fn bits(&self, v: f64) -> u16 {
        if self.bf16 { ((v as f32).to_bits() >> 16) as u16 } else { F16::from_f32(v as f32).0 }
    }

    pub fn is_nan(&self, bits: u16) -> bool {
        self.value(bits).is_nan()
    }

    pub fn is_subnormal(&self, bits: u16) -> bool {
        let v = self.value(bits).abs();
        v != 0.0 && v < 2f64.powi(self.emin)
    }

    /// The size of one unit in the last place at `|v|` (finite, nonzero).
    pub fn ulp(&self, v: f64) -> f64 {
        2f64.powi((v.abs().log2().floor() as i32).max(self.emin) - (self.p - 1))
    }

    /// The exact value `s + e` (an f64 and its error, `|e|` at most half an
    /// f64 ulp of `s`) rounded to nearest even; `zero_negative` is the sign
    /// of an exact zero.
    pub fn round(&self, s: f64, e: f64, zero_negative: bool) -> u16 {
        if s.is_nan() {
            return 0x7fff;
        }
        if s.is_infinite() {
            return self.bits(s);
        }
        if s == 0.0 && e == 0.0 {
            return if zero_negative { 0x8000 } else { 0 };
        }
        let ulp = self.ulp(s);
        let q = s / ulp;
        // A tie in s is decided by the error term.
        let r = if q - q.floor() == 0.5 && e != 0.0 {
            if e > 0.0 { q.floor() + 1.0 } else { q.floor() }
        } else {
            q.round_ties_even()
        };
        let v = r * ulp;
        if v.abs() >= 2f64.powi(self.emax + 1) {
            return self.bits(if v > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY });
        }
        if v == 0.0 {
            return if s < 0.0 { 0x8000 } else { 0 };
        }
        self.bits(v)
    }

    /// Whether the exact nonzero value `s + e`, rounded to nearest even with
    /// `p` significant bits as though the exponent range were unbounded, is
    /// below the smallest normal number: IEEE 754's tininess after rounding,
    /// which decides `.ftz` on an SM89 GPU.
    pub fn tiny(&self, s: f64, e: f64) -> bool {
        let smallest = 2f64.powi(self.emin);
        if s == 0.0 || s.abs() >= smallest {
            return false;
        }
        let ulp = 2f64.powi(s.abs().log2().floor() as i32 - (self.p - 1));
        let q = s / ulp;
        let r = if q - q.floor() == 0.5 && e != 0.0 {
            if e > 0.0 { q.floor() + 1.0 } else { q.floor() }
        } else {
            q.round_ties_even()
        };
        (r * ulp).abs() < smallest
    }

    /// `a + b` exactly, as an f64 and its error (Knuth's TwoSum).
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }

    /// One element of `add`, `sub` or `fma` with `mods`: `.ftz` flushes
    /// subnormal inputs and results to zeros of their signs (a result when
    /// it is tiny after rounding, as an SM89 GPU detects it), the exact
    /// result rounds once to nearest even, NaN results are `0x7fff`, `.sat`
    /// clamps to `[0, 1]` (NaN and `-0` to `+0`), `.relu` makes a negative
    /// result `+0`, and `.oob` gives `+0` for a multiplicand that is the out-of-bounds NaN
    /// `0x7fff` (any other NaN gives the canonical NaN, as a B200 does).
    pub fn op(&self, func: cute::ir::HalfFn, mods: cute::ir::HalfModifiers, a: u16, b: u16, c: u16) -> u16 {
        use cute::ir::HalfFn;
        if mods.oob && (a == 0x7fff || b == 0x7fff) {
            return 0;
        }
        let flush = |x: u16| if mods.ftz && self.is_subnormal(x) { x & 0x8000 } else { x };
        let (a, b, c) = (flush(a), flush(b), flush(c));
        let (x, y, z) = (self.value(a), self.value(b), self.value(c));
        let negative = |bits: u16| bits & 0x8000 != 0;
        let (s, e, zero_negative) = match func {
            HalfFn::Add | HalfFn::Sub => {
                let sub = func == HalfFn::Sub;
                let (s, e) = Self::two_sum(x, if sub { -y } else { y });
                let both_negative = negative(a) && (negative(b) != sub);
                (s, e, both_negative && x == 0.0 && y == 0.0)
            }
            HalfFn::Fma => {
                let product = x * y;
                let (s, e) = Self::two_sum(product, z);
                (s, e, product == 0.0 && z == 0.0 && (negative(a) != negative(b)) && negative(c))
            }
            HalfFn::Ex2 => return self.exp2(a),
        };
        let r = if mods.ftz && s.is_finite() && self.tiny(s, e) {
            if s < 0.0 { 0x8000 } else { 0 }
        } else {
            self.round(s, e, zero_negative)
        };
        let r = flush(r);
        let v = self.value(r);
        if mods.sat {
            return if v.is_nan() || v <= 0.0 {
                0
            } else if v >= 1.0 {
                self.bits(1.0)
            } else {
                r
            };
        }
        if v.is_nan() {
            0x7fff
        } else if mods.relu && negative(r) {
            0
        } else {
            r
        }
    }

    /// `2^a` rounded to nearest even (what `ex2.approx` approximates); in
    /// BFloat16 subnormal inputs and results flush to zero.
    pub fn exp2(&self, a: u16) -> u16 {
        let a = if self.bf16 && self.is_subnormal(a) { a & 0x8000 } else { a };
        let x = self.value(a);
        let r = if x.is_nan() { 0x7fff } else { self.round(x.exp2(), 0.0, false) };
        if self.bf16 && self.is_subnormal(r) { r & 0x8000 } else { r }
    }

    /// [`HalfFormat::op`] on both elements of packed words.
    pub fn op2(&self, func: cute::ir::HalfFn, mods: cute::ir::HalfModifiers, a: u32, b: u32, c: u32) -> u32 {
        let half = |w: u32, k: u32| (w >> (16 * k)) as u16;
        (0..2).fold(0, |d, k| d | (self.op(func, mods, half(a, k), half(b, k), half(c, k)) as u32) << (16 * k))
    }

    /// `a * b + c` rounded once, as `fma.rn` computes it (and `fma` on
    /// Float16 and BFloat16 values).
    pub fn fma(&self, a: u16, b: u16, c: u16) -> u16 {
        let rn = cute::ir::HalfModifiers { rn: true, ..cute::ir::HalfModifiers::default() };
        self.op(cute::ir::HalfFn::Fma, rn, a, b, c)
    }
}

/// The PTX modifiers of `HalfOptions` for the reference (fma always `.rn`).
pub fn half_mods(func: cute::ir::HalfFn, opts: HalfOptions) -> cute::ir::HalfModifiers {
    let fma = func == cute::ir::HalfFn::Fma;
    cute::ir::HalfModifiers { rn: opts.rn || fma, oob: opts.oob, ftz: opts.ftz, sat: opts.sat, relu: opts.relu }
}

/// Options for the kernels: `rn`, `ftz`, `sat` and `relu`.
pub fn half_opts(rn: bool, ftz: bool, sat: bool, relu: bool) -> HalfOptions {
    HalfOptions { rn, ftz, sat, relu, ..HalfOptions::default() }
}

/// Float16 bit patterns the packed kernels combine: signed zeros, the
/// subnormal extremes, the smallest normals, 1 and its neighbours, a third,
/// sums that tie (`1 + 2^-11`), the largest finite values, infinities, NaNs
/// (quiet, signaling, negative) and values whose products overflow.
pub const F16_SPECIALS: [u16; 22] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x3c01, 0x3555, 0x1000, 0x1400,
    0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x7d01, 0xfe55, 0x5bff,
];

/// BFloat16 bit patterns, as [`F16_SPECIALS`], with the factors of a product
/// that is a midpoint (`1.125 * 0.90625 = 261/256`).
pub const BF16_SPECIALS: [u16; 22] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x3f81, 0x3eab, 0x3b80, 0x3c00,
    0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0x7f81, 0xffc5, 0x3f90,
];

/// Rounds of table operands per thread in the half-precision kernels.
pub(super) const HALF_ROUNDS: usize = 4;

/// The packed operands of round `r` of thread `t` from `table`: words `a`,
/// `b` and `c`, each two table entries, and the predicate (every other
/// thread and round).
pub fn half_table_operands(table: &[u16], t: usize, r: usize) -> (u32, u32, u32, bool) {
    let n = table.len();
    let idx = (t + T * r) % (n * n);
    let (i, j) = (idx / n, idx % n);
    let w = |lo: usize, hi: usize| table[lo % n] as u32 | (table[hi % n] as u32) << 16;
    (w(i, j), w(j, i + j), w(i + 3, j + 5), (t + r).is_multiple_of(2))
}

/// The same in a kernel: `table` holds the entries (read at run time, so
/// nothing folds).
pub(super) fn half_table_operands_in_kernel(
    table: &Tensor<u32>,
    n: usize,
    t: Int32,
    r: usize,
) -> (Int32, Int32, Int32, Boolean) {
    let n = n as i32;
    let idx = (t + (T * r) as i32) % (n * n);
    let (i, j) = (idx / n, idx % n);
    let w = |lo: Int32, hi: Int32| (table.load(lo % n) | (table.load(hi % n) << 16u32)).bitcast::<i32>();
    (w(i, j), w(j, i + j), w(i + 3, j + 5), ((t + r as i32) % 2).eq(0))
}

/// A table of 16-bit patterns in local memory.
pub(super) fn half_table(values: &[u16]) -> Tensor<u32> {
    let table = make_rmem_tensor::<u32>(Layout::col_major(values.len()));
    for (i, &v) in values.iter().enumerate() {
        table.store(i, v as u32);
    }
    table
}

/// A Float16 or BFloat16 as its bits in a Uint32.
pub(super) fn half_bits<T: Numeric>(v: Value<T>) -> Uint32 {
    v.bitcast::<u16>().to::<u32>()
}

/// Columns of [`half_f16`].
const HALF_F16_COLS: usize = 24;

/// Packed and scalar Float16 arithmetic: `add_packed_f16x2` and
/// `sub_packed_f16x2` with every combination of `rn`, `ftz` and `sat` and
/// with a predicate, `fma_packed_f16x2` with `ftz`, `sat` and `relu`, the
/// tuple forms, `fma_f16` with options and `fma` on Float16 values; on pairs
/// of [`F16_SPECIALS`] and on the input data (as bits, so every pattern
/// occurs).
pub fn half_f16() -> Case {
    const COLS: usize = (HALF_ROUNDS + 1) * HALF_F16_COLS;
    let kernel = Kernel::builder("half_f16").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let ys = param_tensor::<i32>("ys", Layout::col_major(T));
        let zs = param_tensor::<i32>("zs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let table = half_table(&F16_SPECIALS);
        let mut sets: Vec<(Int32, Int32, Int32, Boolean)> =
            (0..HALF_ROUNDS).map(|r| half_table_operands_in_kernel(&table, F16_SPECIALS.len(), t, r)).collect();
        let x = xs.load(t);
        sets.push((x, ys.load(t), zs.load(t), (x & 4).eq(0)));
        for (r, (a, b, c, p)) in sets.into_iter().enumerate() {
            let o = half_opts;
            let [alo, ahi]: [Float16; 2] = unpack(a);
            let [blo, bhi]: [Float16; 2] = unpack(b);
            let [clo, chi]: [Float16; 2] = unpack(c);
            // The tuple forms on other halves than the packed ones, so that
            // they stay separate instructions.
            let tuple_add = add_packed_f16x2((alo, bhi), (blo, ahi), o(true, false, false, false));
            let tuple_fma = fma_packed_f16x2((ahi, alo), (bhi, blo), (chi, clo), o(false, false, false, true));
            let results: [Uint32; HALF_F16_COLS] = [
                add_packed_f16x2(a, b, HalfOptions::default()).bitcast(),
                add_packed_f16x2(a, b, o(true, false, false, false)).bitcast(),
                add_packed_f16x2(a, b, o(false, true, false, false)).bitcast(),
                add_packed_f16x2(a, b, o(false, false, true, false)).bitcast(),
                add_packed_f16x2(a, b, o(true, true, true, false)).bitcast(),
                sub_packed_f16x2(a, b, HalfOptions::default()).bitcast(),
                sub_packed_f16x2(a, b, o(true, false, false, false)).bitcast(),
                sub_packed_f16x2(a, b, o(false, true, false, false)).bitcast(),
                sub_packed_f16x2(a, b, o(false, false, true, false)).bitcast(),
                sub_packed_f16x2(a, b, o(false, true, true, false)).bitcast(),
                fma_packed_f16x2(a, b, c, HalfOptions::default()).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, true, false, false)).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, false, true, false)).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, false, false, true)).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, true, true, false)).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, true, false, true)).bitcast(),
                add_packed_f16x2(a, b, HalfOptions { predicate: Some(p), ..HalfOptions::default() }).bitcast(),
                sub_packed_f16x2(a, b, HalfOptions { predicate: Some(p), ftz: true, ..HalfOptions::default() })
                    .bitcast(),
                pack([tuple_add.0, tuple_add.1]),
                half_bits(fma_f16(alo, blo, clo, HalfOptions::default())),
                half_bits(fma_f16(ahi, bhi, chi, o(false, true, false, true))),
                half_bits(fma_f16(alo, bhi, chi, o(false, false, true, false))),
                half_bits(fma(ahi, blo, clo)),
                pack([tuple_fma.0, tuple_fma.1]),
            ];
            for (j, v) in results.into_iter().enumerate() {
                out.store((t, r * HALF_F16_COLS + j), v);
            }
        }
    });
    fn check(args: &[KernelArg]) {
        use cute::ir::HalfFn::{Add, Fma, Sub};
        const COLS: usize = (HALF_ROUNDS + 1) * HALF_F16_COLS;
        let (xs, ys, zs) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let out = args[3].to_vec::<u32>();
        let f = &F16_FORMAT;
        let o = half_opts;
        for t in 0..T {
            let mut sets: Vec<(u32, u32, u32, bool)> =
                (0..HALF_ROUNDS).map(|r| half_table_operands(&F16_SPECIALS, t, r)).collect();
            sets.push((xs[t] as u32, ys[t] as u32, zs[t] as u32, xs[t] & 4 == 0));
            for (r, &(a, b, c, p)) in sets.iter().enumerate() {
                let op2 = |func, opts: HalfOptions, a, b, c| f.op2(func, half_mods(func, opts), a, b, c);
                let op = |func, opts: HalfOptions, a: u32, b: u32, c: u32| {
                    f.op(func, half_mods(func, opts), a as u16, b as u16, c as u16) as u32
                };
                let none = HalfOptions::default();
                let want: [u32; HALF_F16_COLS] = [
                    op2(Add, none, a, b, 0),
                    op2(Add, o(true, false, false, false), a, b, 0),
                    op2(Add, o(false, true, false, false), a, b, 0),
                    op2(Add, o(false, false, true, false), a, b, 0),
                    op2(Add, o(true, true, true, false), a, b, 0),
                    op2(Sub, none, a, b, 0),
                    op2(Sub, o(true, false, false, false), a, b, 0),
                    op2(Sub, o(false, true, false, false), a, b, 0),
                    op2(Sub, o(false, false, true, false), a, b, 0),
                    op2(Sub, o(false, true, true, false), a, b, 0),
                    op2(Fma, none, a, b, c),
                    op2(Fma, o(false, true, false, false), a, b, c),
                    op2(Fma, o(false, false, true, false), a, b, c),
                    op2(Fma, o(false, false, false, true), a, b, c),
                    op2(Fma, o(false, true, true, false), a, b, c),
                    op2(Fma, o(false, true, false, true), a, b, c),
                    if p { op2(Add, none, a, b, 0) } else { a },
                    if p { op2(Sub, o(false, true, false, false), a, b, 0) } else { a },
                    op2(
                        Add,
                        o(true, false, false, false),
                        a & 0xffff | b & 0xffff_0000,
                        b & 0xffff | a & 0xffff_0000,
                        0,
                    ),
                    op(Fma, none, a, b, c),
                    op(Fma, o(false, true, false, true), a >> 16, b >> 16, c >> 16),
                    op(Fma, o(false, false, true, false), a, b >> 16, c >> 16),
                    op(Fma, none, a >> 16, b, c),
                    op2(Fma, o(false, false, false, true), a.rotate_left(16), b.rotate_left(16), c.rotate_left(16)),
                ];
                for (j, w) in want.into_iter().enumerate() {
                    let got = out[t * COLS + r * HALF_F16_COLS + j];
                    assert_eq!(got, w, "half_f16 column {j}, thread {t}, round {r}: {a:#010x} {b:#010x} {c:#010x}");
                }
            }
        }
    }
    Case { name: "half_f16", kernel, cfg: launch(1), small_only: false, check }
}

/// Columns of [`half_bf16`].
const HALF_BF16_COLS: usize = 6;

/// BFloat16 fma, which needs sm_80: `fma_packed_bf16x2` with and without
/// `relu` (and its tuple form), `fma_bf16`, and `fma` on BFloat16 values,
/// now `fma.rn.bf16` rounding once; on pairs of [`BF16_SPECIALS`], on a
/// product that is a midpoint plus the smallest normal (rounding twice would
/// tie and go to even), and on the input data.
pub fn half_bf16() -> Case {
    const SETS: usize = HALF_ROUNDS + 2;
    const COLS: usize = SETS * HALF_BF16_COLS;
    let kernel = Kernel::builder("half_bf16").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let ys = param_tensor::<i32>("ys", Layout::col_major(T));
        let zs = param_tensor::<i32>("zs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let table = half_table(&BF16_SPECIALS);
        let mut sets: Vec<(Int32, Int32, Int32)> = (0..HALF_ROUNDS)
            .map(|r| {
                let (a, b, c, _) = half_table_operands_in_kernel(&table, BF16_SPECIALS.len(), t, r);
                (a, b, c)
            })
            .collect();
        let (a, b, c) = half_bf16_midpoint();
        sets.push((xs.load(t) & 0xffff_0000u32 as i32 | a, ys.load(t) & 0xffff_0000u32 as i32 | b, Int32::from(c)));
        sets.push((xs.load(t), ys.load(t), zs.load(t)));
        for (r, (a, b, c)) in sets.into_iter().enumerate() {
            let [alo, ahi]: [BFloat16; 2] = unpack(a);
            let [blo, bhi]: [BFloat16; 2] = unpack(b);
            let [clo, chi]: [BFloat16; 2] = unpack(c);
            let relu = HalfOptions { relu: true, ..HalfOptions::default() };
            let tuple = fma_packed_bf16x2((ahi, alo), (bhi, blo), (chi, clo), HalfOptions::default());
            let results: [Uint32; HALF_BF16_COLS] = [
                fma_packed_bf16x2(a, b, c, HalfOptions::default()).bitcast(),
                fma_packed_bf16x2(a, b, c, relu).bitcast(),
                half_bits(fma_bf16(alo, blo, clo, HalfOptions::default())),
                half_bits(fma_bf16(ahi, bhi, chi, relu)),
                half_bits(fma(ahi, blo, chi)),
                pack([tuple.0, tuple.1]),
            ];
            for (j, v) in results.into_iter().enumerate() {
                out.store((t, r * HALF_BF16_COLS + j), v);
            }
        }
    });
    fn check(args: &[KernelArg]) {
        use cute::ir::HalfFn::Fma;
        const SETS: usize = HALF_ROUNDS + 2;
        const COLS: usize = SETS * HALF_BF16_COLS;
        let (xs, ys, zs) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let out = args[3].to_vec::<u32>();
        let f = &BF16_FORMAT;
        let none = half_mods(Fma, HalfOptions::default());
        let relu = half_mods(Fma, HalfOptions { relu: true, ..HalfOptions::default() });
        for t in 0..T {
            let mut sets: Vec<(u32, u32, u32)> = (0..HALF_ROUNDS)
                .map(|r| {
                    let (a, b, c, _) = half_table_operands(&BF16_SPECIALS, t, r);
                    (a, b, c)
                })
                .collect();
            let (a, b, c) = half_bf16_midpoint();
            let high = |v: i32| v as u32 & 0xffff_0000;
            sets.push((high(xs[t]) | a as u32, high(ys[t]) | b as u32, c as u32));
            sets.push((xs[t] as u32, ys[t] as u32, zs[t] as u32));
            for (r, &(a, b, c)) in sets.iter().enumerate() {
                let want = [
                    f.op2(Fma, none, a, b, c),
                    f.op2(Fma, relu, a, b, c),
                    f.op(Fma, none, a as u16, b as u16, c as u16) as u32,
                    f.op(Fma, relu, (a >> 16) as u16, (b >> 16) as u16, (c >> 16) as u16) as u32,
                    f.op(Fma, none, (a >> 16) as u16, b as u16, (c >> 16) as u16) as u32,
                    f.op2(Fma, none, a.rotate_left(16), b.rotate_left(16), c.rotate_left(16)),
                ];
                for (j, w) in want.into_iter().enumerate() {
                    let got = out[t * COLS + r * HALF_BF16_COLS + j];
                    assert_eq!(got, w, "half_bf16 column {j}, thread {t}, round {r}: {a:#010x} {b:#010x} {c:#010x}");
                }
            }
        }
        // The midpoint round: 261/256 + 2^-126 rounds up, to 1 + 3 * 2^-7.
        assert_eq!(out[COLS - 2 * HALF_BF16_COLS + 2] & 0xffff, 0x3f83);
    }
    Case { name: "half_bf16", kernel, cfg: launch(1), small_only: false, check }
}

/// BFloat16 `a`, `b` and `c` (low halves): `a * b` is 261/256, a midpoint,
/// and `c` the smallest normal, so the exact sum rounds up; through f32 it
/// would round to the midpoint and then to even (down).
pub fn half_bf16_midpoint() -> (i32, i32, i32) {
    (0x3f90, 0x3f68, 0x0080)
}

/// Columns of [`half_sm90`].
pub(super) const HALF_SM90_COLS: usize = 12;

/// Words of [`exp2_f16x2`]: each holds two Float16 values, `2g` and
/// `2g + 1` for word `g`, so every value occurs once.
pub const EXP2_WORDS: usize = 32768;

/// `exp2_packed_f16x2` on every Float16 value, alone and with a predicate
/// (true in two words of three); a word per thread, no inputs. The GPU's
/// `ex2.approx.f16x2` is an approximation: `tests/gpu.rs` holds it to its
/// bound, the simulator to the correctly rounded value.
pub fn exp2_f16x2() -> Case {
    let kernel = Kernel::builder("exp2_f16x2").block_dim([256, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(EXP2_WORDS));
        let guarded = param_tensor::<u32>("guarded", Layout::col_major(EXP2_WORDS));
        let g = block_idx().x * 256 + thread_idx().x;
        let word = (g * 2) | ((g * 2 + 1) << 16);
        out.store(g, exp2_packed_f16x2(word, None).bitcast::<u32>());
        let p = (g % 3).ne(0);
        guarded.store(g, exp2_packed_f16x2(word, Some(p)).bitcast::<u32>());
    });
    fn check(args: &[KernelArg]) {
        let (out, guarded) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        let f = &F16_FORMAT;
        for g in 0..EXP2_WORDS {
            let word = (2 * g as u32) | ((2 * g as u32 + 1) << 16);
            let want = f.exp2(word as u16) as u32 | (f.exp2((word >> 16) as u16) as u32) << 16;
            assert_eq!(out[g], want, "exp2_packed_f16x2 of {word:#010x}");
            assert_eq!(guarded[g], if g % 3 != 0 { want } else { word }, "predicated, word {g}");
        }
    }
    Case { name: "exp2_f16x2", kernel, cfg: launch_n((EXP2_WORDS / 256) as u32, 256), small_only: false, check }
}

/// Float32 bit patterns for the conversions: signed zeros, subnormals (a
/// BFloat16 tie among them), 1, BFloat16 and TF32 ties (`0x...8000`,
/// `0x...1000`), the largest finite values and the patterns just around the
/// BFloat16 and TF32 overflow thresholds, infinities, and NaNs whose payload
/// lies in the high bits, only in the low 13 bits, or both.
pub const CVT_F32_SPECIALS: [u32; 30] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x0001_8000,
    0x007f_ffff,
    0x007f_f000,
    0x0080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_1000,
    0xbf80_1000,
    0x3f80_3000,
    0x3f80_8000,
    0x3f81_8000,
    0x3f80_0fff,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f7f_8000,
    0x7f7f_7fff,
    0x7f7f_f000,
    0x7f7f_efff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0001,
    0x7f80_0001,
    0x7f80_1000,
    0x7fff_ffff,
    0x7fbf_ffff,
];

/// `cvt.rn.satfinite.bf16.f32` on Float32 bits, from the PTX ISA: NaN gives
/// the canonical NaN, a magnitude above BFloat16's largest finite value its
/// largest finite value with the sign, others round to nearest even.
pub fn bf16_satfinite_ref(x: u32) -> u16 {
    let v = f32::from_bits(x);
    let f = &BF16_FORMAT;
    if v.is_nan() {
        return 0x7fff;
    }
    let largest = f.value(0x7f7f);
    if (v as f64).abs() > largest {
        return if v < 0.0 { 0xff7f } else { 0x7f7f };
    }
    f.round(v as f64, 0.0, v.is_sign_negative())
}

/// `cvt.rn.bf16.f32` on Float32 bits: to nearest even, overflowing to
/// infinity; NaN gives the canonical NaN (as on an SM89 GPU).
pub fn bf16_rn_ref(x: u32) -> u16 {
    let v = f32::from_bits(x);
    if v.is_nan() { 0x7fff } else { BF16_FORMAT.round(v as f64, 0.0, v.is_sign_negative()) }
}

/// `cvt.rna.tf32.f32` on Float32 bits: the nearest value with 11 significant
/// bits, ties away from zero, overflowing to infinity; infinities and NaNs
/// keep their bits but the low 13 (what an SM89 GPU does).
pub fn tf32_ref(x: u32) -> u32 {
    let v = f32::from_bits(x) as f64;
    if !v.is_finite() {
        return x & !0x1fff;
    }
    if v == 0.0 {
        return x;
    }
    let ulp = 2f64.powi((v.abs().log2().floor() as i32).max(-126) - 10);
    let magnitude = (v.abs() / ulp + 0.5).floor() * ulp;
    let r = if v < 0.0 { -magnitude } else { magnitude };
    if r.abs() >= 2f64.powi(128) { (x & 0x8000_0000) | 0x7f80_0000 } else { (r as f32).to_bits() }
}

/// Rounds of table pairs per thread in [`cvt_floats`].
const CVT_ROUNDS: usize = 15;

/// Columns of [`cvt_floats`].
const CVT_COLS: usize = 5;

/// The Float32 conversions: `cvt_f32x2_bf16x2` (packed, and as a tuple of
/// the swapped pair), `cvt_f32_bf16`, `cvt_f32_tf32` and `Value::to_tf32`,
/// on every pair of [`CVT_F32_SPECIALS`] and on the input data. Needs PTX 8.1
/// (`.satfinite`).
pub fn cvt_floats() -> Case {
    const COLS: usize = (CVT_ROUNDS + 1) * CVT_COLS;
    let kernel = Kernel::builder("cvt_floats").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(T));
        let ys = param_tensor::<f32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let n = CVT_F32_SPECIALS.len() as i32;
        let table = make_rmem_tensor::<u32>(Layout::col_major(CVT_F32_SPECIALS.len()));
        for (i, v) in CVT_F32_SPECIALS.into_iter().enumerate() {
            table.store(i, v);
        }
        let mut pairs: Vec<(Float32, Float32)> = (0..CVT_ROUNDS as i32)
            .map(|r| {
                let p = (t + T as i32 * r) % (n * n);
                (table.load(p / n).bitcast::<f32>(), table.load(p % n).bitcast::<f32>())
            })
            .collect();
        pairs.push((xs.load(t), ys.load(t)));
        for (r, (x, y)) in pairs.into_iter().enumerate() {
            let packed: Int32 = cvt_f32x2_bf16x2([x, y]);
            let (lo, hi): (BFloat16, BFloat16) = cvt_f32x2_bf16x2([y, x]);
            let results: [Uint32; CVT_COLS] = [
                packed.bitcast(),
                pack([lo, hi]),
                half_bits(cvt_f32_bf16(x)),
                cvt_f32_tf32(x).bitcast(),
                y.to_tf32().bitcast(),
            ];
            for (j, v) in results.into_iter().enumerate() {
                out.store((t, r * CVT_COLS + j), v);
            }
        }
    });
    fn check(args: &[KernelArg]) {
        const COLS: usize = (CVT_ROUNDS + 1) * CVT_COLS;
        let (xs, ys, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>(), args[2].to_vec::<u32>());
        let n = CVT_F32_SPECIALS.len();
        for t in 0..T {
            let mut pairs: Vec<(u32, u32)> = (0..CVT_ROUNDS)
                .map(|r| (t + T * r) % (n * n))
                .map(|p| (CVT_F32_SPECIALS[p / n], CVT_F32_SPECIALS[p % n]))
                .collect();
            pairs.push((xs[t].to_bits(), ys[t].to_bits()));
            for (r, &(x, y)) in pairs.iter().enumerate() {
                let sat = |lo: u32, hi: u32| bf16_satfinite_ref(lo) as u32 | (bf16_satfinite_ref(hi) as u32) << 16;
                let want = [sat(x, y), sat(y, x), bf16_rn_ref(x) as u32, tf32_ref(x), tf32_ref(y)];
                for (j, w) in want.into_iter().enumerate() {
                    let got = out[t * COLS + r * CVT_COLS + j];
                    assert_eq!(got, w, "cvt_floats column {j}, thread {t}: {x:#010x} {y:#010x}: got {got:#010x}");
                }
            }
        }
    }
    Case { name: "cvt_floats", kernel, cfg: launch(1), small_only: false, check }
}

/// Int8 values of thread `t`'s `k`-th operand in [`cvt_int8`]: consecutive
/// over the threads, so every value occurs.
pub fn cvt_int8_value(t: usize, k: usize, per_thread: usize) -> i8 {
    (t * per_thread + k) as u8 as i8
}

/// Columns of [`cvt_int8`]: 18 Float32 results, then 45 BFloat16 ones.
const CVT_INT8_COLS: usize = 18 + 45;

/// The int8 conversions: `cvt_i8x4_to_f32x4` on every int8 value (four per
/// thread, as an array) and on the input data as an Int32 and a Uint32
/// word, `cvt_i8x2_to_f32x2` on arrays and on Int16 and Uint16 halves of the
/// data, and `cvt_i8_bf16_intrinsic` (the Ampere path, which every target
/// has) on every length from 1 to 9 (nine consecutive values per thread).
pub fn cvt_int8() -> Case {
    let kernel = Kernel::builder("cvt_int8").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, CVT_INT8_COLS)));
        let t = thread_idx().x;
        let x = xs.load(t);
        let value = |k: i32, per: i32| ((t * per + k) & 0xff).to::<i8>();
        let quad: [Int8; 4] = std::array::from_fn(|k| value(k as i32, 4));
        let mut results: Vec<Uint32> = Vec::new();
        let bits = |v: Float32| v.bitcast::<u32>();
        results.extend(cvt_i8x4_to_f32x4(quad).map(bits));
        results.extend(cvt_i8x4_to_f32x4(x).map(bits));
        results.extend(cvt_i8x4_to_f32x4(x.bitcast::<u32>().rotate_bytes()).map(bits));
        results.extend(cvt_i8x2_to_f32x2([quad[1], quad[3]]).map(bits));
        results.extend(cvt_i8x2_to_f32x2(x.to::<i16>()).map(bits));
        results.extend(cvt_i8x2_to_f32x2((x.bitcast::<u32>() >> 16u32).to::<u16>()).map(bits));
        let nine: Vec<Int8> = (0..9).map(|k| value(k, 9)).collect();
        for len in 1..=9 {
            results.extend(cvt_i8_bf16_intrinsic(&nine, len).into_iter().map(half_bits));
        }
        assert_eq!(results.len(), CVT_INT8_COLS);
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            let x = xs[t] as u32;
            let bytes = |w: u32| w.to_le_bytes().map(|b| b as i8);
            let mut want: Vec<u32> = Vec::new();
            let quad: [i8; 4] = std::array::from_fn(|k| cvt_int8_value(t, k, 4));
            let f = |v: i8| (v as f32).to_bits();
            want.extend(quad.map(f));
            want.extend(bytes(x).map(f));
            want.extend(bytes(x.rotate_right(8)).map(f));
            want.extend([quad[1], quad[3]].map(f));
            want.extend([bytes(x)[0], bytes(x)[1]].map(f));
            want.extend([bytes(x)[2], bytes(x)[3]].map(f));
            let nine: Vec<i8> = (0..9).map(|k| cvt_int8_value(t, k, 9)).collect();
            for len in 1..=9 {
                want.extend(nine[..len].iter().map(|&v| BF16::from_f32(v as f32).0 as u32));
            }
            assert_eq!(out[t * CVT_INT8_COLS..(t + 1) * CVT_INT8_COLS], want, "cvt_int8, thread {t}, data {x:#010x}");
        }
    }
    Case { name: "cvt_int8", kernel, cfg: launch(1), small_only: false, check }
}

/// A Uint32's bytes rotated down by one (byte 1 to byte 0), with a funnel
/// shift.
trait RotateBytes {
    fn rotate_bytes(self) -> Uint32;
}

impl RotateBytes for Uint32 {
    fn rotate_bytes(self) -> Uint32 {
        shf(self, self, 8, ShfKind::Right)
    }
}

/// Unpacked int4 words of [`sext_i4`]: word `i` holds the four hex digits of
/// `i`, one per byte (digit 0 in byte 0).
pub fn sext_word(i: u32) -> u32 {
    (0..4).fold(0, |w, k| w | ((i >> (4 * k)) & 0xf) << (8 * k))
}

/// Threads of [`sext_i4`], and words per thread.
const SEXT_THREADS: usize = 4096;
const SEXT_PER_THREAD: usize = 16;

/// `sext_unpacked_i4x4_to_i8x4` on every combination of four int4 values
/// (65536 words) as Uint32 words, on the first of each thread's words as an
/// array, and `sext_unpacked_i4_i8_intrinsic` on its first two words (eight
/// values).
pub fn sext_i4() -> Case {
    let words = SEXT_THREADS * SEXT_PER_THREAD;
    let kernel = Kernel::builder("sext_i4").block_dim([256, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(words));
        let out4 = param_tensor::<i8>("out4", Layout::col_major(SEXT_THREADS * 4));
        let out8 = param_tensor::<i8>("out8", Layout::col_major(SEXT_THREADS * 8));
        let g = block_idx().x * 256 + thread_idx().x;
        let word = |i: Int32| {
            let i = i.bitcast::<u32>();
            (0..4).fold(Uint32::from(0u32), |w, k| w | ((i >> (4 * k) as u32) & 0xfu32) << (8 * k) as u32)
        };
        let mut first: Vec<Uint32> = Vec::new();
        for k in 0..SEXT_PER_THREAD as i32 {
            let i = g * SEXT_PER_THREAD as i32 + k;
            let w = word(i);
            out.store(i, sext_unpacked_i4x4_to_i8x4(w));
            if k < 2 {
                first.push(w);
            }
        }
        let array: [Int8; 4] = unpack(first[0]);
        for (k, v) in sext_unpacked_i4x4_to_i8x4(array).into_iter().enumerate() {
            out4.store(g * 4 + k as i32, v);
        }
        let values: Vec<Int8> = first.iter().flat_map(|&w| unpack::<i8, u32, 4>(w)).collect();
        for (k, v) in sext_unpacked_i4_i8_intrinsic(&values, 8).into_iter().enumerate() {
            out8.store(g * 8 + k as i32, v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (out, out4, out8) = (args[0].to_vec::<u32>(), args[1].to_vec::<i8>(), args[2].to_vec::<i8>());
        let sext = |w: u32| w.to_le_bytes().map(|b| ((b as i8) << 4) >> 4);
        for (i, &got) in out.iter().enumerate() {
            let want = u32::from_le_bytes(sext(sext_word(i as u32)).map(|b| b as u8));
            assert_eq!(got, want, "sext of {:#010x}", sext_word(i as u32));
        }
        for g in 0..SEXT_THREADS {
            let first = (g * SEXT_PER_THREAD) as u32;
            assert_eq!(out4[g * 4..g * 4 + 4], sext(sext_word(first)), "array form, thread {g}");
            let both: Vec<i8> = sext(sext_word(first)).into_iter().chain(sext(sext_word(first + 1))).collect();
            assert_eq!(out8[g * 8..g * 8 + 8], both[..], "intrinsic, thread {g}");
        }
    }
    let cfg = launch_n((SEXT_THREADS / 256) as u32, 256);
    Case { name: "sext_i4", kernel, cfg, small_only: false, check }
}

/// `cvt_i8_bf16` (`cvt.rn.bf16.s8`, sm_90) on every int8 value, four per
/// thread.
pub fn cvt_i8_bf16_sm90() -> Case {
    let kernel = Kernel::builder("cvt_i8_bf16_sm90").block_dim([T as u32, 1, 1]).build(|| {
        let out = param_tensor::<BF16>("out", Layout::col_major(4 * T));
        let t = thread_idx().x;
        for k in 0..4 {
            let v = ((t * 4 + k) & 0xff).to::<i8>();
            out.store(t * 4 + k, cvt_i8_bf16(v));
        }
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<BF16>();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v.0, BF16::from_f32(i as u8 as i8 as f32).0, "cvt_i8_bf16 of {}", i as u8 as i8);
        }
    }
    Case { name: "cvt_i8_bf16_sm90", kernel, cfg: launch(1), small_only: false, check }
}

/// The int8 to BFloat16 conversions of sm_100a-class targets (PTX 9.1):
/// `cvt_i8x2_to_bf16x2`, `cvt_i8x4_to_bf16x4` and `cvt_i8_bf16_intrinsic`
/// on the s2f6x2 path for every length from 1 to 9. Neither `ptxas` 12.8 nor
/// LLVM 21 or 22 handles PTX 9.1: it is in no kernel list, but simulated and
/// checked as MLIR (and as PTX when the `mlir-opt` is LLVM 23).
pub fn cvt_i8_bf16_s2f6x2() -> Case {
    const COLS: usize = 6 + 45;
    let kernel = Kernel::builder("cvt_i8_bf16_s2f6x2").block_dim([T as u32, 1, 1]).build(|| {
        let out = param_tensor::<BF16>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let value = |k: i32, per: i32| ((t * per + k) & 0xff).to::<i8>();
        let quad: [Int8; 4] = std::array::from_fn(|k| value(k as i32, 4));
        let mut results: Vec<BFloat16> = Vec::new();
        results.extend(cvt_i8x4_to_bf16x4(quad));
        let (lo, hi): (BFloat16, BFloat16) = cvt_i8x2_to_bf16x2([quad[2], quad[0]]);
        results.extend([lo, hi]);
        let nine: Vec<Int8> = (0..9).map(|k| value(k, 9)).collect();
        for len in 1..=9 {
            results.extend(cvt_i8_bf16_intrinsic_with(&nine, len, I8ToBf16Path::S2f6x2));
        }
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        const COLS: usize = 6 + 45;
        let out = args[0].to_vec::<BF16>();
        let bf = |v: i8| BF16::from_f32(v as f32).0;
        for t in 0..T {
            let quad: [i8; 4] = std::array::from_fn(|k| cvt_int8_value(t, k, 4));
            let mut want: Vec<u16> = quad.map(bf).to_vec();
            want.extend([bf(quad[2]), bf(quad[0])]);
            let nine: Vec<i8> = (0..9).map(|k| cvt_int8_value(t, k, 9)).collect();
            for len in 1..=9 {
                want.extend(nine[..len].iter().map(|&v| bf(v)));
            }
            let got: Vec<u16> = out[t * COLS..(t + 1) * COLS].iter().map(|v| v.0).collect();
            assert_eq!(got, want, "thread {t}");
        }
    }
    Case { name: "cvt_i8_bf16_s2f6x2", kernel, cfg: launch(1), small_only: false, check }
}

/// Columns of [`pack_unpack`].
const PACK_COLS: usize = 20;

/// `pack` and `unpack` between every width: two Float16, two BFloat16, four
/// Int8 and two Uint8 values into 32 and 16 bits, two Int16 into a Float32
/// (NaN patterns kept), two Int32 or Float32 and four Uint16 into 64 bits,
/// eight Int8 into a Float64, and back, on the input data's bits.
pub fn pack_unpack() -> Case {
    let kernel = Kernel::builder("pack_unpack").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let out = param_tensor::<u64>("out", Layout::row_major((T, PACK_COLS)));
        let t = thread_idx().x;
        let (x, y) = (xs.load(t), ys.load(t) * 0x9e37_79b9u32);
        let [x0, x1]: [Float16; 2] = unpack(x);
        let [y0, y1]: [Float16; 2] = unpack(y);
        let [bx0, bx1]: [BFloat16; 2] = unpack(x);
        let [by0, by1]: [BFloat16; 2] = unpack(y);
        let xb: [Int8; 4] = unpack(x);
        let yb: [Int8; 4] = unpack(y);
        let xu: [Uint8; 4] = unpack(x);
        let yu: [Uint8; 4] = unpack(y);
        let xh: [Int16; 2] = unpack(x);
        let yh: [Uint16; 2] = unpack(y);
        let wide = |v: Uint32| v.to::<u64>();
        let i32_bits = |v: Int32| wide(v.bitcast());
        let xy: Int64 = pack([x.bitcast::<i32>(), y.bitcast::<i32>()]);
        let floats: Uint64 = pack([x.bitcast::<f32>(), y.bitcast::<f32>()]);
        let quads: Int64 = pack([xh[1].bitcast::<u16>(), yh[0], xh[0].bitcast::<u16>(), yh[1]]);
        let eight: Float64 = pack([xb[0], yb[1], xb[2], yb[3], yb[0], xb[1], yb[2], xb[3]]);
        let [w0, w1]: [Float32; 2] = unpack(floats);
        let bytes: [Uint8; 8] = unpack(xy);
        let results: [Uint64; PACK_COLS] = [
            i32_bits(pack([x0, y1])),
            wide(pack([by0, bx1])),
            wide(pack([xb[0], yb[1], xb[2], yb[3]])),
            pack::<u8, i16, 2>([xu[3], yu[0]]).bitcast::<u16>().to::<u64>(),
            wide(pack::<i16, f32, 2>([xh[0], yh[1].bitcast()]).bitcast()),
            xy.bitcast(),
            floats,
            quads.bitcast(),
            eight.bitcast(),
            wide(w0.bitcast()),
            wide(w1.bitcast()),
            bytes[0].to(),
            bytes[3].to(),
            bytes[4].to(),
            bytes[7].to(),
            half_bits(x1).to(),
            half_bits(y0).to(),
            half_bits(bx0).to(),
            half_bits(by1).to(),
            wide(pack([x0, x1])),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u64>());
        for t in 0..T {
            let (x, y) = (xs[t], ys[t].wrapping_mul(0x9e37_79b9));
            let (xb, yb) = (x.to_le_bytes(), y.to_le_bytes());
            let (lo, hi) = (|w: u32| (w & 0xffff) as u64, |w: u32| (w >> 16) as u64);
            let join8 = |b: [u8; 8]| u64::from_le_bytes(b);
            let want: [u64; PACK_COLS] = [
                lo(x) | hi(y) << 16,
                lo(y) | hi(x) << 16,
                u32::from_le_bytes([xb[0], yb[1], xb[2], yb[3]]) as u64,
                (xb[3] as u64) | (yb[0] as u64) << 8,
                lo(x) | hi(y) << 16,
                x as u64 | (y as u64) << 32,
                x as u64 | (y as u64) << 32,
                hi(x) | lo(y) << 16 | lo(x) << 32 | hi(y) << 48,
                join8([xb[0], yb[1], xb[2], yb[3], yb[0], xb[1], yb[2], xb[3]]),
                x as u64,
                y as u64,
                xb[0] as u64,
                xb[3] as u64,
                yb[0] as u64,
                yb[3] as u64,
                hi(x),
                lo(y),
                lo(x),
                hi(y),
                x as u64,
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(out[t * PACK_COLS + j], w, "pack_unpack column {j}, thread {t}: {x:#010x} {y:#010x}");
            }
        }
    }
    Case { name: "pack_unpack", kernel, cfg: launch(1), small_only: false, check }
}

/// The kernels that run on every target (in [`all`]): Pack
/// and Unpack, Float16 and BFloat16 (fma) arithmetic, the Float32 and int8
/// conversions and the int4 sign extension.
pub fn packed_half_and_conversions() -> Vec<Case> {
    vec![pack_unpack(), half_f16(), half_bf16(), cvt_floats(), cvt_int8(), sext_i4()]
}

/// The half-precision sweep for `tests/gpu.rs`: `func` on `n` words (pairs
/// of `elem` values) for each of `options`, `out[i * options.len() + m]`
/// from `a[i]`, `b[i]` and `c[i]`; the packed forms, or with `scalar` the
/// scalar fma (`fma_f16`, `fma_bf16`) on the low halves, or with `plain` the
/// `fma` of `Value`s.
pub fn half_sweep(func: cute::ir::HalfFn, elem: DType, form: SweepForm, options: &[HalfOptions], n: usize) -> Kernel {
    use cute::ir::HalfFn;
    let options = options.to_vec();
    let name = format!("sweep_{}_{}_{:?}", func.ptx(), elem.name(), form).to_lowercase();
    Kernel::builder(name).block_dim([256, 1, 1]).build(move || {
        let a = param_tensor::<i32>("a", Layout::col_major(n));
        let b = param_tensor::<i32>("b", Layout::col_major(n));
        let c = param_tensor::<i32>("c", Layout::col_major(n));
        let out = param_tensor::<i32>("out", Layout::col_major(n * options.len()));
        let i = block_idx().x * 256 + thread_idx().x;
        let (x, y, z) = (a.load(i), b.load(i), c.load(i));
        let m = options.len() as i32;
        for (k, &opts) in options.iter().enumerate() {
            let r: Int32 = match (form, elem) {
                (SweepForm::Packed, DType::F16) => match func {
                    HalfFn::Add => add_packed_f16x2(x, y, opts),
                    HalfFn::Sub => sub_packed_f16x2(x, y, opts),
                    HalfFn::Fma => fma_packed_f16x2(x, y, z, opts),
                    HalfFn::Ex2 => exp2_packed_f16x2(x, opts.predicate),
                },
                (SweepForm::Packed, _) => match func {
                    HalfFn::Add => add_packed_bf16x2(x, y, opts),
                    HalfFn::Sub => sub_packed_bf16x2(x, y, opts),
                    HalfFn::Fma => fma_packed_bf16x2(x, y, z, opts),
                    HalfFn::Ex2 => exp2_packed_bf16x2(x, opts.predicate),
                },
                (_, DType::F16) => {
                    let [xl, _]: [Float16; 2] = unpack(x);
                    let [yl, _]: [Float16; 2] = unpack(y);
                    let [zl, _]: [Float16; 2] = unpack(z);
                    let r = if form == SweepForm::Scalar { fma_f16(xl, yl, zl, opts) } else { fma(xl, yl, zl) };
                    half_bits(r).bitcast()
                }
                _ => {
                    let [xl, _]: [BFloat16; 2] = unpack(x);
                    let [yl, _]: [BFloat16; 2] = unpack(y);
                    let [zl, _]: [BFloat16; 2] = unpack(z);
                    let r = if form == SweepForm::Scalar { fma_bf16(xl, yl, zl, opts) } else { fma(xl, yl, zl) };
                    half_bits(r).bitcast()
                }
            };
            out.store(i * m + k as i32, r);
        }
    })
}

/// Which form of the operation [`half_sweep`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepForm {
    /// The packed operation on Int32 pairs.
    Packed,
    /// `fma_f16` or `fma_bf16` on the low halves.
    Scalar,
    /// `fma` on the low halves' `Value`s.
    Plain,
}

/// The Float32 conversions on `n` bit patterns for `tests/gpu.rs`:
/// `out[3i..3i + 3]` is `cvt_f32x2_bf16x2([x, -x])`, `cvt_f32_bf16(x)` and
/// `cvt_f32_tf32(x)` for `x = xs[i]`.
pub fn cvt_sweep(n: usize) -> Kernel {
    Kernel::builder("cvt_sweep").block_dim([256, 1, 1]).build(move || {
        let xs = param_tensor::<u32>("xs", Layout::col_major(n));
        let out = param_tensor::<u32>("out", Layout::col_major(3 * n));
        let i = block_idx().x * 256 + thread_idx().x;
        let bits = xs.load(i);
        let (x, negated) = (bits.bitcast::<f32>(), (bits ^ 0x8000_0000u32).bitcast::<f32>());
        out.store(i * 3, cvt_f32x2_bf16x2::<Int32>([x, negated]).bitcast::<u32>());
        out.store(i * 3 + 1, half_bits(cvt_f32_bf16(x)));
        out.store(i * 3 + 2, cvt_f32_tf32(x).bitcast::<u32>());
    })
}

/// Float32 patterns for [`cvt_sweep`]: every sign and exponent with
/// mantissas around the BFloat16 and TF32 rounding points (ties, one either
/// side, the extremes) and pseudo-random ones.
pub fn cvt_sweep_inputs() -> Vec<u32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let edges =
        [0u32, 1, 0x7f_ffff, 0x40_0000, 0x0fff, 0x1000, 0x1001, 0x1fff, 0x3000, 0x7fff, 0x8000, 0x8001, 0x1_8000];
    let mut v = Vec::new();
    for top in 0..512u32 {
        for (k, &m) in edges.iter().enumerate() {
            v.push(top << 23 | m);
            v.push(top << 23 | (m ^ (0x7f_0000 >> k)) & 0x7f_ffff);
        }
        for _ in 0..998 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(top << 23 | (state as u32 & 0x7f_ffff));
        }
    }
    v
}
