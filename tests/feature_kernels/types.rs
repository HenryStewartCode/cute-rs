//! The numeric type system: Boolean and TFloat32 conversions, MMAs on
//! TFloat32 fragments, implicit promotion, the FP8 and scale-factor
//! conversions, Int4 and the 128-bit integers.

use cute::prelude::*;

use super::{Case, T, launch};

/// f32 bit patterns every conversion kernel sees in its first lanes: zeros,
/// NaNs (quiet, signalling, negative), infinities, the extremes, and ties.
pub const F32_SPECIALS: [u32; 16] = [
    0x0000_0000,
    0x8000_0000,
    0x7fc0_0000,
    0xffc0_0001,
    0x7f80_0001,
    0x7f80_0000,
    0xff80_0000,
    0x7f7f_ffff,
    0x0000_0001,
    0x3f80_1000,
    0xbf80_3000,
    0x43e0_0000,
    0x43e8_0000,
    0x4760_0000,
    0x4770_0000,
    0x3a80_0000,
];

/// The f32 bits of lane `t`: a special value in the first lanes, else the
/// input.
pub fn with_specials(t: Int32, bits: Uint32) -> Uint32 {
    let mut v = bits;
    for (k, &s) in F32_SPECIALS.iter().enumerate() {
        v = select(t.eq(k as i32), s, v);
    }
    v
}

/// The host side of [`with_specials`] (by global thread index in the
/// kernels of several blocks).
pub fn host_specials(t: usize, bits: u32) -> u32 {
    F32_SPECIALS.get(t).copied().unwrap_or(bits)
}

/// CuTe DSL's `Boolean(x)` from every type: `x != 0`, unordered for floats.
pub fn boolean_conversions() -> Case {
    const N: usize = 9;
    let kernel = Kernel::builder("boolean_conversions").block_dim([T as u32, 1, 1]).build(|| {
        let bits = param_tensor::<u32>("bits", Layout::col_major(T));
        let ints = param_tensor::<i32>("ints", Layout::col_major(T));
        let out = param_tensor::<bool>("out", Layout::row_major((T, N)));
        let t = thread_idx().x;
        let f = with_specials(t, bits.load(t)).bitcast::<f32>();
        let i = ints.load(t);
        let results: [Boolean; N] = [
            f.to(),
            f.to::<F16>().to(),
            f.to::<BF16>().to(),
            f.to::<f64>().to(),
            f.to::<TF32>().to(),
            i.to(),
            i.to::<i8>().to(),
            i.to::<u64>().to(),
            (i - 3).to::<u16>().to(),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (bits, ints, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>(), args[2].to_vec::<bool>());
        for t in 0..T {
            let f = f32::from_bits(host_specials(t, bits[t]));
            let i = ints[t];
            let want: [bool; N] = [
                f != 0.0 || f.is_nan(),
                F16::from_f32(f).to_f32() != 0.0 || f.is_nan(),
                BF16::from_f32(f).to_f32() != 0.0 || f.is_nan(),
                f != 0.0 || f.is_nan(),
                TF32::from_f32(f).to_f32() != 0.0 || TF32::from_f32(f).to_f32().is_nan(),
                i != 0,
                i as i8 != 0,
                i as u64 != 0,
                (i.wrapping_sub(3)) as u16 != 0,
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(out[t * N + j], w, "conversion {j} of lane {t} ({f:?}, {i})");
            }
        }
    }
    Case { name: "boolean_conversions", kernel, cfg: launch(1), small_only: false, check }
}

/// Float32 to TFloat32 (`cvt.rna.tf32.f32`), and back, and TFloat32 from
/// the other types.
pub fn tf32_conversions() -> Case {
    const N: usize = 4;
    let kernel = Kernel::builder("tf32_conversions").block_dim([T as u32, 1, 1]).build(|| {
        let bits = param_tensor::<u32>("bits", Layout::col_major(T));
        let ints = param_tensor::<i32>("ints", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, N)));
        let t = thread_idx().x;
        let f = with_specials(t, bits.load(t)).bitcast::<f32>();
        let x: TFloat32 = f.to();
        let results: [Uint32; N] = [
            x.bitcast(),
            x.to::<f64>().to::<f32>().bitcast(),
            ints.load(t).to::<TF32>().bitcast(),
            f.to::<F16>().to::<TF32>().bitcast(),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (bits, ints, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            let f = f32::from_bits(host_specials(t, bits[t]));
            let x = TF32::from_f32(f).0;
            // f32 -> f64 -> f32 keeps a NaN's payload (quieted).
            let back = if x.is_nan() { x.to_bits() | 0x40_0000 } else { x.to_bits() };
            let want: [u32; N] = [
                x.to_bits(),
                back,
                TF32::from_f32(ints[t] as f32).0.to_bits(),
                if f.is_nan() { 0x7fff_e000 } else { TF32::from_f32(F16::from_f32(f).to_f32()).0.to_bits() },
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(out[t * N + j], w, "conversion {j} of lane {t} ({f:?})");
            }
        }
    }
    Case { name: "tf32_conversions", kernel, cfg: launch(1), small_only: false, check }
}

/// An `m16n8k8` TF32 MMA twice: on Float32 fragments holding TF32-rounded
/// values (`to_tf32`) and on TFloat32 fragments (`to::<TF32>()`). The two
/// accumulators must be bit-identical.
pub fn mma_tf32_fragments() -> Case {
    let (m, n, k) = (16, 8, 8);
    let kernel = Kernel::builder("mma_tf32_fragments").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::col_major((m, k)));
        let b = param_tensor::<f32>("b", Layout::col_major((n, k)));
        let c_f32 = param_tensor::<f32>("c_f32", Layout::col_major((m, n)));
        let c_tf32 = param_tensor::<f32>("c_tf32", Layout::col_major((m, n)));
        let mma = make_tiled_mma(MmaTF32Op::new((16, 8, 8)), Layout::col_major((1, 1, 1)));
        let thr = mma.get_slice(lane_idx());
        let (ta, tb) = (thr.partition_a(&a), thr.partition_b(&b));
        let (fa, fb) = (thr.partition_fragment_a::<f32, _>(&a), thr.partition_fragment_b::<f32, _>(&b));
        let (ga, gb) = (thr.partition_fragment_a::<TF32, _>(&a), thr.partition_fragment_b::<TF32, _>(&b));
        for i in 0..fa.size() {
            fa.store(i, ta.load(i).to_tf32());
            ga.store(i, ta.load(i).to::<TF32>());
        }
        for i in 0..fb.size() {
            fb.store(i, tb.load(i).to_tf32());
            gb.store(i, tb.load(i).to::<TF32>());
        }
        for (out, use_tf32) in [(&c_f32, false), (&c_tf32, true)] {
            let acc = thr.partition_fragment_c::<f32, _>(out);
            acc.fill(0.0f32);
            if use_tf32 {
                gemm(&mma, &acc, &ga, &gb, &acc);
            } else {
                gemm(&mma, &acc, &fa, &fb, &acc);
            }
            let tc = thr.partition_c(out);
            for i in 0..acc.size() {
                tc.store(i, acc.load(i));
            }
        }
    });
    fn check(args: &[KernelArg]) {
        let (m, n, k) = (16, 8, 8);
        let (a, b) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        let (c1, c2) = (args[2].to_vec::<f32>(), args[3].to_vec::<f32>());
        for i in 0..m {
            for j in 0..n {
                assert_eq!(c1[i + m * j].to_bits(), c2[i + m * j].to_bits(), "C[{i},{j}]");
                let want: f64 =
                    (0..k).map(|l| TF32::from_f32(a[i + m * l]).0 as f64 * TF32::from_f32(b[j + n * l]).0 as f64).sum();
                assert_eq!(c1[i + m * j] as f64, want, "C[{i},{j}]");
            }
        }
    }
    Case { name: "mma_tf32_fragments", kernel, cfg: LaunchConfig::new([1, 1, 1], [32, 1, 1]), small_only: true, check }
}

/// Arithmetic and comparisons on values of different types, promoted as
/// CuTe DSL promotes them.
pub fn mixed_promotion() -> Case {
    const N: usize = 10;
    let kernel = Kernel::builder("mixed_promotion").block_dim([T as u32, 1, 1]).build(|| {
        let ints = param_tensor::<i32>("ints", Layout::col_major(T));
        let words = param_tensor::<u32>("words", Layout::col_major(T));
        let halves = param_tensor::<F16>("halves", Layout::col_major(T));
        let floats = param_tensor::<f32>("floats", Layout::col_major(T));
        let out = param_tensor::<f64>("out", Layout::row_major((T, N)));
        let flags = param_tensor::<bool>("flags", Layout::row_major((T, 3)));
        let t = thread_idx().x;
        let (i, u, h, f) = (ints.load(t), words.load(t), halves.load(t), floats.load(t));
        let (s, c, l) = (i.to::<i16>(), u.to::<u8>(), i.to::<i64>() * 1000);
        let results: [Float64; N] = [
            (i + u).to(),
            (u * i).to(),
            (h + i).to(),
            (h * f).to(),
            (f - l).to(),
            (c + s).to(),
            (s ^ c).to(),
            (u >> (c & 7u8)).to(),
            (l / (u | 1u32)).to(),
            (h.to::<BF16>() + s).to(),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
        flags.store((t, 0), i.lt(u));
        flags.store((t, 1), s.ge(c));
        flags.store((t, 2), h.lt(f));
    });
    fn check(args: &[KernelArg]) {
        let (ints, words) = (args[0].to_vec::<i32>(), args[1].to_vec::<u32>());
        let (halves, floats) = (args[2].to_vec::<F16>(), args[3].to_vec::<f32>());
        let (out, flags) = (args[4].to_vec::<f64>(), args[5].to_vec::<bool>());
        for t in 0..T {
            let (i, u, h, f) = (ints[t], words[t], halves[t].to_f32(), floats[t]);
            let (s, c, l) = (i as i16, u as u8, (i as i64).wrapping_mul(1000));
            let as_f16 = |x: f32| F16::from_f32(x).to_f32();
            let bf_plus = as_f16(as_f16(BF16::from_f32(h).to_f32()) + as_f16(s as f32));
            let want: [f64; N] = [
                (i as u32).wrapping_add(u) as f64,
                u.wrapping_mul(i as u32) as f64,
                (h + i as f32) as f64,
                (h * f) as f64,
                f as f64 - l as f64,
                (c as i16).wrapping_add(s) as f64,
                (s ^ c as i16) as f64,
                (u >> (c & 7)) as f64,
                (l / (u | 1) as i64) as f64,
                bf_plus as f64,
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(out[t * N + j], w, "operation {j} of lane {t}");
            }
            assert_eq!(flags[t * 3], (i as u32) < u, "flag 0 of lane {t}");
            assert_eq!(flags[t * 3 + 1], s >= c as i16, "flag 1 of lane {t}");
            assert_eq!(flags[t * 3 + 2], h < f, "flag 2 of lane {t}");
        }
    }
    Case { name: "mixed_promotion", kernel, cfg: launch(1), small_only: false, check }
}

/// Blocks of [`fp8_conversions`].
const FP8_BLOCKS: usize = 16;
/// Columns of [`fp8_conversions`].
const FP8_N: usize = 18;

/// The f32 value of a host decode, as the device's decode gives it: exact,
/// and the canonical f32 NaN for a NaN.
fn f32_bits_of(x: f64) -> u32 {
    if x.is_nan() { 0x7fff_ffff } else { (x as f32).to_bits() }
}

/// Every FP8 and scale-factor conversion on f32 values (specials in the
/// first lanes) and bytes: `to()` into E4M3FN, E5M2 and E8M0 (from Float32
/// and Float16), the pair forms with `.relu`, and every format back to
/// Float32 (`cvt` for E4M3FN and E5M2, integer operations for the others).
pub fn fp8_conversions() -> Case {
    let kernel = Kernel::builder("fp8_conversions").block_dim([T as u32, 1, 1]).build(|| {
        let n = FP8_BLOCKS * T;
        let bits = param_tensor::<u32>("bits", Layout::col_major(n));
        let bytes = param_tensor::<u8>("bytes", Layout::col_major(n));
        let out = param_tensor::<u32>("out", Layout::row_major((n, FP8_N)));
        let g = block_idx().x * T as i32 + thread_idx().x;
        let x = with_specials(g, bits.load(g)).bitcast::<f32>();
        let byte = bytes.load(g);
        let b2 = bytes.load((g + 1) % n as i32);
        let as_u32 = |v: Uint8| v.to::<u32>();
        let pair16 = |p: [Uint8; 2]| as_u32(p[0]) | as_u32(p[1]) << 8u32;
        let h = x.to::<F16>();
        let e4 = |b: Uint8| b.bitcast::<F8E4M3FN>();
        let results: [Uint32; FP8_N] = [
            as_u32(x.to::<F8E4M3FN>().bitcast()),
            as_u32(x.to::<F8E5M2>().bitcast()),
            pair16(cvt_f32x2_to_f8x2::<F8E4M3FN>([x, -x], true).map(|v| v.bitcast())),
            pair16(cvt_f32x2_to_f8x2::<F8E5M2>([x, -x], true).map(|v| v.bitcast())),
            pair16(cvt_f32x2_to_f8x2::<F8E5M2>([x, x * 0.5f32], false).map(|v| v.bitcast())),
            as_u32(h.to::<F8E4M3FN>().bitcast()),
            pair16(cvt_f16x2_to_f8x2::<F8E5M2>((h, -h), true).map(|v| v.bitcast())),
            as_u32(x.to::<F8E8M0FNU>().bitcast()),
            e4(byte).to::<f32>().bitcast(),
            byte.bitcast::<F8E5M2>().to::<f32>().bitcast(),
            byte.bitcast::<F8E4M3>().to::<f32>().bitcast(),
            byte.bitcast::<F8E4M3B11FNUZ>().to::<f32>().bitcast(),
            byte.bitcast::<F8E8M0FNU>().to::<f32>().bitcast(),
            byte.bitcast::<F8E5M3FNU>().to::<f32>().bitcast(),
            cvt_f8x2_to_f16x2::<F8E4M3FN, Int32>([e4(byte), e4(b2)], true).bitcast(),
            as_u32(e4(byte).to::<F8E5M2>().bitcast()),
            byte.bitcast::<F8E5M2>().to::<F16>().bitcast::<u16>().to(),
            e4(byte).to::<bool>().to(),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((g, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let n = FP8_BLOCKS * T;
        let (bits, bytes, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u8>(), args[2].to_vec::<u32>());
        let rn = NarrowRounding::NearestEven;
        for g in 0..n {
            let x = f32::from_bits(host_specials(g, bits[g]));
            let (byte, b2) = (bytes[g], bytes[(g + 1) % n]);
            let h = F16(cute_f16_bits(x)).to_f32();
            let e4 = |v: f64, relu: bool| F8E4M3FN::from_f64_with(v, rn, true, relu).0 as u32;
            let e5 = |v: f64, relu: bool| F8E5M2::from_f64_with(v, rn, true, relu).0 as u32;
            let half_of = |v: f64| -> u32 { if v.is_nan() { 0x7fff } else { F16::from_f32(v as f32).0 as u32 } };
            let relu_half = |v: f64| {
                if v.is_nan() {
                    0x7fff
                } else if v < 0.0 || v.to_bits() >> 63 == 1 {
                    0
                } else {
                    half_of(v)
                }
            };
            let want: [u32; FP8_N] = [
                e4(x as f64, false),
                e5(x as f64, false),
                e4(x as f64, true) | e4(-x as f64, true) << 8,
                e5(x as f64, true) | e5(-x as f64, true) << 8,
                e5(x as f64, false) | e5((x * 0.5) as f64, false) << 8,
                e4(h as f64, false),
                e5(h as f64, true) | e5(-h as f64, true) << 8,
                F8E8M0FNU::from_f32(x).0 as u32,
                f32_bits_of(F8E4M3FN(byte).to_f64()),
                f32_bits_of(F8E5M2(byte).to_f64()),
                f32_bits_of(F8E4M3(byte).to_f64()),
                f32_bits_of(F8E4M3B11FNUZ(byte).to_f64()),
                f32_bits_of(F8E8M0FNU(byte).to_f64()),
                f32_bits_of(F8E5M3FNU(byte).to_f64()),
                relu_half(F8E4M3FN(byte).to_f64()) | relu_half(F8E4M3FN(b2).to_f64()) << 16,
                e5(F8E4M3FN(byte).to_f64(), false),
                half_of(F8E5M2(byte).to_f64()),
                (F8E4M3FN(byte).to_f64() != 0.0) as u32,
            ];
            for (j, w) in want.into_iter().enumerate() {
                let have = out[g * FP8_N + j];
                // Float32 results of a NaN: any NaN (the GPU test holds the
                // bits to the simulator's).
                let both_nan = (8..14).contains(&j) && f32::from_bits(have).is_nan() && f32::from_bits(w).is_nan();
                assert!(
                    have == w || both_nan,
                    "conversion {j} of lane {g} ({x:?}, byte {byte:#04x}): {have:#x}, want {w:#x}"
                );
            }
        }
    }
    Case {
        name: "fp8_conversions",
        kernel,
        cfg: LaunchConfig::new([FP8_BLOCKS as u32, 1, 1], [T as u32, 1, 1]),
        small_only: false,
        check,
    }
}

/// The Float16 bits `cvt.rn.f16.f32` gives: round to nearest even, and the
/// canonical NaN for a NaN.
pub fn cute_f16_bits(x: f32) -> u16 {
    if x.is_nan() { 0x7fff } else { F16::from_f32(x).0 }
}

/// Lanes of [`fp8_f16_sweep`]: one per Float16 value.
pub const SWEEP_LANES: usize = 1 << 16;
const SWEEP_N: usize = 7;

/// Every Float16 value to E4M3FN and E5M2 (`to()`, and `.relu` pairs), and
/// every byte from E4M3FN and E5M2 to Float16, with and without `.relu`,
/// and every Float16 value to Float32 (`cvt.f32.f16`: exact, every NaN the
/// canonical `0x7fffffff`, as the simulator converts).
pub fn fp8_f16_sweep() -> Case {
    let kernel = Kernel::builder("fp8_f16_sweep").block_dim([256, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::row_major((SWEEP_LANES, SWEEP_N)));
        let g = block_idx().x * 256 + thread_idx().x;
        let h = g.to::<u16>().bitcast::<F16>();
        let byte = (g & 0xff).to::<u8>();
        let as_u32 = |v: Uint8| v.to::<u32>();
        let pair16 = |p: [Uint8; 2]| as_u32(p[0]) | as_u32(p[1]) << 8u32;
        let e4 = byte.bitcast::<F8E4M3FN>();
        let e5 = byte.bitcast::<F8E5M2>();
        let results: [Uint32; SWEEP_N] = [
            as_u32(h.to::<F8E4M3FN>().bitcast()),
            as_u32(h.to::<F8E5M2>().bitcast()),
            pair16(cvt_f16x2_to_f8x2::<F8E4M3FN>((h, -h), true).map(|v| v.bitcast())),
            pair16(cvt_f16x2_to_f8x2::<F8E5M2>((h, -h), true).map(|v| v.bitcast())),
            cvt_f8x2_to_f16x2::<F8E4M3FN, Int32>([e4, e4], false).bitcast(),
            cvt_f8x2_to_f16x2::<F8E5M2, Int32>([e5, e5], true).bitcast(),
            h.to::<f32>().bitcast(),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((g, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<u32>();
        let rn = NarrowRounding::NearestEven;
        for g in 0..SWEEP_LANES {
            let h = F16(g as u16).to_f32() as f64;
            let byte = (g & 0xff) as u8;
            let e4 = |v: f64, relu: bool| F8E4M3FN::from_f64_with(v, rn, true, relu).0 as u32;
            let e5 = |v: f64, relu: bool| F8E5M2::from_f64_with(v, rn, true, relu).0 as u32;
            let half = |v: f64, relu: bool| -> u32 {
                if v.is_nan() {
                    0x7fff
                } else if relu && (v < 0.0 || v.to_bits() >> 63 == 1) {
                    0
                } else {
                    F16::from_f32(v as f32).0 as u32
                }
            };
            let (d4, d5) = (F8E4M3FN(byte).to_f64(), F8E5M2(byte).to_f64());
            let want: [u32; SWEEP_N] = [
                e4(h, false),
                e5(h, false),
                e4(h, true) | e4(-h, true) << 8,
                e5(h, true) | e5(-h, true) << 8,
                half(d4, false) * 0x1_0001,
                half(d5, true) * 0x1_0001,
                if h.is_nan() { 0x7fff_ffff } else { (h as f32).to_bits() },
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(out[g * SWEEP_N + j], w, "column {j} of lane {g:#x}");
            }
        }
    }
    Case {
        name: "fp8_f16_sweep",
        kernel,
        cfg: LaunchConfig::new([(SWEEP_LANES / 256) as u32, 1, 1], [256, 1, 1]),
        small_only: true,
        check,
    }
}

/// The UE8M0 pair conversions (sm_100a): Float32 and BFloat16 pairs to
/// UE8M0 with each rounding and modifier, and UE8M0 pairs to BFloat16.
pub fn ue8m0_pairs() -> Case {
    const N: usize = 7;
    let kernel = Kernel::builder("ue8m0_pairs").block_dim([T as u32, 1, 1]).build(|| {
        let bits = param_tensor::<u32>("bits", Layout::col_major(T));
        let bytes = param_tensor::<u8>("bytes", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, N)));
        let t = thread_idx().x;
        let x = with_specials(t, bits.load(t)).bitcast::<f32>();
        let (b, y) = (x.to::<BF16>(), (-x).to::<BF16>());
        let as_u32 = |v: Float8E8M0FNU| v.bitcast::<u8>().to::<u32>();
        let pair16 = |p: [Float8E8M0FNU; 2]| as_u32(p[0]) | as_u32(p[1]) << 8u32;
        let s = bytes.load(t).bitcast::<F8E8M0FNU>();
        let results: [Uint32; N] = [
            pair16(cvt_f32x2_to_ue8m0x2([x, -x], false)),
            pair16(cvt_f32x2_to_ue8m0x2([x, x * 3.0f32], true)),
            pair16(cvt_bf16x2_to_ue8m0x2((b, y), true, true)),
            pair16(cvt_bf16x2_to_ue8m0x2((b, y), false, false)),
            pair16(cvt_bf16x2_to_ue8m0x2((b, y), true, false)),
            cvt_ue8m0x2_to_bf16x2::<Int32>([s, s]).bitcast(),
            as_u32(x.to::<F8E8M0FNU>()),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (bits, bytes, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u8>(), args[2].to_vec::<u32>());
        let (rz, rp) = (NarrowRounding::Zero, NarrowRounding::Up);
        for t in 0..T {
            let x = f32::from_bits(host_specials(t, bits[t]));
            let bf = |v: f32| if v.is_nan() { f64::NAN } else { BF16::from_f32(v).to_f32() as f64 };
            let (b, y) = (bf(x), bf(-x));
            let u = |v: f64, r, sat, relu| F8E8M0FNU::from_f64_with(v, r, sat, relu).0 as u32;
            let s = F8E8M0FNU(bytes[t]).to_f64();
            let s_bf = if s.is_nan() { 0x7fff } else { BF16::from_f32(s as f32).0 as u32 };
            let want: [u32; N] = [
                u(x as f64, rz, true, false) | u(-x as f64, rz, true, false) << 8,
                u(x as f64, rp, true, false) | u((x * 3.0) as f64, rp, true, false) << 8,
                u(b, rp, true, false) | u(y, rp, true, false) << 8,
                u(b, rz, false, false) | u(y, rz, false, false) << 8,
                u(b, rp, false, false) | u(y, rp, false, false) << 8,
                s_bf * 0x1_0001,
                F8E8M0FNU::from_f32(x).0 as u32,
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(out[t * N + j], w, "conversion {j} of lane {t} ({x:?})");
            }
        }
    }
    Case { name: "ue8m0_pairs", kernel, cfg: launch(1), small_only: false, check }
}

/// Edge values every Int128 lane pair meets first: the extremes, words
/// that carry across 64 bits, and small values.
pub const I128_SPECIALS: [i128; 8] =
    [i128::MIN, i128::MAX, u64::MAX as i128, -(u64::MAX as i128), 1 << 64, -1, 0, (1 << 63) - 1];

/// The Int128 of lane `t`: an edge value in the first lanes, else the input.
fn i128_input(t: Int32, x: Int128, other: bool) -> Int128 {
    let mut v = x;
    for (k, &s) in I128_SPECIALS.iter().enumerate() {
        let s = if other { I128_SPECIALS[(k + 3) % I128_SPECIALS.len()] } else { s };
        v = select(t.eq(k as i32), s, v);
    }
    v
}

fn host_i128(t: usize, x: i128, other: bool) -> i128 {
    let n = I128_SPECIALS.len();
    if t < n { I128_SPECIALS[if other { (t + 3) % n } else { t }] } else { x }
}

const I128_N: usize = 20;

/// Int128 and Uint128 arithmetic, shifts, comparisons, min and max and
/// conversions at edge values, with a 128-bit scalar parameter and
/// 128-bit buffers.
pub fn int128_ops() -> Case {
    let kernel = Kernel::builder("int128_ops").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i128>("xs", Layout::col_major(T));
        let ys = param_tensor::<i128>("ys", Layout::col_major(T));
        let us = param_tensor::<u128>("us", Layout::col_major(T));
        let k = param_scalar::<i128>("k");
        let out = param_tensor::<i128>("out", Layout::row_major((T, I128_N)));
        let t = thread_idx().x;
        let (x, y) = (i128_input(t, xs.load(t), false), i128_input(t, ys.load(t), true));
        let u = us.load(t);
        let d = y | 1i128;
        let sh = (y & 127i128).to::<u128>();
        let ux = x.bitcast::<u128>();
        let results: [Int128; I128_N] = [
            x + y,
            x - y,
            x * y,
            x / d,
            x % d,
            x << (y & 127i128),
            x >> (y & 127i128),
            (ux >> sh).bitcast(),
            (ux / (u | 1u128)).bitcast(),
            (ux % (u | 1u128)).bitcast(),
            x.min(y),
            x.max(y),
            (ux.min(u)).bitcast(),
            select(x.lt(y), 1i128, 0i128) | select(ux.lt(u), 2i128, 0i128) | select(x.eq(y), 4i128, 0i128),
            x.to::<i64>().to(),
            x.to::<u32>().to(),
            t.to::<i128>() - 40,
            (t - 40).to::<u64>().to(),
            x + k,
            (-x) ^ !y,
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, us) = (args[0].to_vec::<i128>(), args[1].to_vec::<i128>(), args[2].to_vec::<u128>());
        let KernelArg::Scalar128 { bits: k, .. } = args[3] else { panic!("the fourth argument is a 128-bit scalar") };
        let out = args[4].to_vec::<i128>();
        for t in 0..T {
            let (x, y, u) = (host_i128(t, xs[t], false), host_i128(t, ys[t], true), us[t]);
            let d = y | 1;
            let sh = (y & 127) as u32;
            let ux = x as u128;
            let want: [i128; I128_N] = [
                x.wrapping_add(y),
                x.wrapping_sub(y),
                x.wrapping_mul(y),
                x.wrapping_div(d),
                x.wrapping_rem(d),
                x << sh,
                x >> sh,
                (ux >> sh) as i128,
                (ux / (u | 1)) as i128,
                (ux % (u | 1)) as i128,
                x.min(y),
                x.max(y),
                ux.min(u) as i128,
                (x < y) as i128 | ((ux < u) as i128) << 1 | ((x == y) as i128) << 2,
                x as i64 as i128,
                x as u32 as i128,
                t as i128 - 40,
                (t as i32 - 40) as u64 as i128,
                x.wrapping_add(k as i128),
                x.wrapping_neg() ^ !y,
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(out[t * I128_N + j], w, "operation {j} of lane {t} ({x}, {y}, {u})");
            }
        }
    }
    Case { name: "int128_ops", kernel, cfg: launch(1), small_only: false, check }
}

const I4_N: usize = 22;

/// Int4 from and to Int8, Int32, Float32 and Float16 over all 16 values
/// (and out-of-range sources, which wrap), its arithmetic, which wraps at
/// 4 bits, its comparisons and its promotion.
pub fn int4_conversions() -> Case {
    let kernel = Kernel::builder("int4_conversions").block_dim([T as u32, 1, 1]).build(|| {
        let ints = param_tensor::<i32>("ints", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, I4_N)));
        let t = thread_idx().x;
        // Every Int4 value, four times; and a partner from the input.
        let a: Int4 = (t - 8).to();
        let b: Int4 = ints.load(t).to();
        let nz = select(b.eq(I4::ZERO), I4::new(3), b);
        let x = t.to::<f32>() * 0.5f32 - 16.25f32;
        let results: [Int32; I4_N] = [
            a.to(),
            a.to::<i8>().to(),
            a.to::<u8>().to(),
            a.to::<f32>().bitcast(),
            a.to::<F16>().bitcast::<i16>().to(),
            b.to(),
            (t * 7).to::<i8>().to::<I4>().to(),
            x.to::<I4>().to(),
            x.to::<F16>().to::<I4>().to(),
            (a + b).to(),
            (a - b).to(),
            (a * b).to(),
            (a / nz).to(),
            (a % nz).to(),
            (-a).to(),
            (a >> I4::new(1)).to(),
            (a << I4::new(2)).to(),
            a.min(b).to(),
            a.max(b).abs().to(),
            select(a.lt(b), 1, 0) | select(a.eq(b), 2, 0),
            (a + t.to::<i8>()).to(),
            a.to::<bool>().to(),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (ints, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        let w = |v: i64| I4::wrapping(v).get() as i32;
        for t in 0..T {
            let a = w(t as i64 - 8);
            let b = w(ints[t] as i64);
            let nz = if b == 0 { 3 } else { b };
            let x = t as f32 * 0.5 - 16.25;
            let want: [i32; I4_N] = [
                a,
                a,
                a as u8 as i32,
                (a as f32).to_bits() as i32,
                F16::from_f32(a as f32).0 as i16 as i32,
                b,
                w(((t * 7) as i8) as i64),
                w(x as i32 as i64),
                w(F16::from_f32(x).to_f32() as i32 as i64),
                w((a + b) as i64),
                w((a - b) as i64),
                w((a * b) as i64),
                w((a / nz) as i64),
                w((a % nz) as i64),
                w(-a as i64),
                a >> 1,
                w((a << 2) as i64),
                a.min(b),
                w(a.max(b).abs() as i64),
                (a < b) as i32 | ((a == b) as i32) << 1,
                (a as i8).wrapping_add(t as i8) as i32,
                (a != 0) as i32,
            ];
            for (j, v) in want.into_iter().enumerate() {
                assert_eq!(out[t * I4_N + j], v, "column {j} of lane {t} (a {a}, b {b})");
            }
        }
    }
    Case { name: "int4_conversions", kernel, cfg: launch(1), small_only: false, check }
}
