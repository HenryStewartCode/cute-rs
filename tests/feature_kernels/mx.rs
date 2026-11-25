//! The OCP MX FP4 and FP6 types: packed storage in global memory and
//! registers, the packed views (FP4x2, FP6x4) and their recasts, the
//! integer-operation decode of `to::<f32>()` (every target), and the
//! sm_100a / sm_120a `cvt` pairs (simulated, compiled and assembled only).

use cute::prelude::*;

use super::{Case, T, host_specials, launch, with_specials};

/// FP4 values per row: one 128-bit vector per thread.
const FP4_ROW: usize = 32;
/// FP6 values per row: one 96-bit group (three words) per thread.
const FP6_ROW: usize = 16;

/// FP4 and FP6 tensors moved through registers in whole words (FP6 values
/// straddling words), decoded to Float32 with integer operations, and the
/// packed views: bytes of FP4x2 read as such and packed from FP4 registers,
/// FP6x4 units copied as one 96-bit group and unpacked into their lanes.
pub fn mx_storage() -> Case {
    let kernel = Kernel::builder("mx_storage").block_dim([T as u32, 1, 1]).build(|| {
        let fp4 = param_tensor::<F4E2M1FN>("fp4", Layout::row_major((T, FP4_ROW)));
        let fp6 = param_tensor::<F6E2M3FN>("fp6", Layout::row_major((T, FP6_ROW)));
        let fp6b = param_tensor::<F6E3M2FN>("fp6b", Layout::row_major((T, FP6_ROW)));
        let fp4_out = param_tensor::<F4E2M1FN>("fp4_out", Layout::row_major((T, FP4_ROW)));
        let fp6_out = param_tensor::<F6E2M3FN>("fp6_out", Layout::row_major((T, FP6_ROW)));
        let decoded = param_tensor::<f32>("decoded", Layout::row_major((T, FP4_ROW + 3 * FP6_ROW)));
        let x2_out = param_tensor::<F4E2M1FNx2>("x2_out", Layout::row_major((T, 2)));
        let x4_out = param_tensor::<F6E2M3FNx4>("x4_out", Layout::row_major((T, 4)));
        let t = thread_idx().x;
        fn row<E>(x: &Tensor<E>, t: Int32) -> Tensor<E> {
            x.slice((t, ..))
        }
        let a = make_fragment::<F4E2M1FN>(FP4_ROW);
        autovec_copy(&row(&fp4, t), &a);
        autovec_copy(&a, &row(&fp4_out, t));
        let b = make_fragment::<F6E2M3FN>(FP6_ROW);
        autovec_copy(&row(&fp6, t), &b);
        autovec_copy(&b, &row(&fp6_out, t));
        let c = make_fragment::<F6E3M2FN>(FP6_ROW);
        autovec_copy(&row(&fp6b, t), &c);
        for i in 0..FP4_ROW {
            decoded.store((t, i as i32), a.load(i).to::<f32>());
        }
        for i in 0..FP6_ROW {
            decoded.store((t, (FP4_ROW + i) as i32), b.load(i).to::<f32>());
            decoded.store((t, (FP4_ROW + FP6_ROW + i) as i32), c.load(i).to::<f32>());
        }
        // FP4x2: a byte of the global tensor, and lanes 2 and 3 packed.
        let bytes = recast_tensor::<F4E2M1FNx2, _>(&fp4);
        assert_eq!(bytes.size(), T * FP4_ROW / 2);
        x2_out.store((t, 0), bytes.load((t, 0)));
        let pairs = a.recast::<F4E2M1FNx2>();
        x2_out.store((t, 1), pairs.load(1));
        // FP6x4: four units (96 bits), unpacked into 16 lanes.
        let units = recast_tensor::<F6E2M3FNx4, _>(&fp6);
        let u = make_fragment::<F6E2M3FNx4>(4);
        autovec_copy(&row(&units, t), &u);
        autovec_copy(&u, &row(&x4_out, t));
        let lanes = u.recast::<F6E2M3FN>();
        for i in 0..FP6_ROW {
            decoded.store((t, (FP4_ROW + 2 * FP6_ROW + i) as i32), lanes.load(i).to::<f32>());
        }
    });
    fn check(args: &[KernelArg]) {
        let fp4 = args[0].to_vec::<F4E2M1FN>();
        let (fp6, fp6b) = (args[1].to_vec::<F6E2M3FN>(), args[2].to_vec::<F6E3M2FN>());
        assert_eq!(args[3].to_vec::<u8>(), args[0].to_vec::<u8>(), "fp4_out");
        assert_eq!(args[4].to_vec::<u8>(), args[1].to_vec::<u8>(), "fp6_out");
        let decoded = args[5].to_vec::<f32>();
        let (x2, x4) = (args[6].to_vec::<F4E2M1FNx2>(), args[7].to_vec::<F6E2M3FNx4>());
        let w = FP4_ROW + 3 * FP6_ROW;
        for t in 0..T {
            for i in 0..FP4_ROW {
                assert_eq!(decoded[t * w + i], fp4[t * FP4_ROW + i].to_f32(), "fp4[{t}][{i}]");
            }
            for i in 0..FP6_ROW {
                let (v, vb) = (fp6[t * FP6_ROW + i].to_f32(), fp6b[t * FP6_ROW + i].to_f32());
                assert_eq!(decoded[t * w + FP4_ROW + i].to_bits(), v.to_bits(), "fp6[{t}][{i}]");
                assert_eq!(decoded[t * w + FP4_ROW + FP6_ROW + i].to_bits(), vb.to_bits(), "fp6b[{t}][{i}]");
                assert_eq!(decoded[t * w + FP4_ROW + 2 * FP6_ROW + i].to_bits(), v.to_bits(), "fp6x4 lane {i}");
            }
            let row = &fp4[t * FP4_ROW..];
            assert_eq!(x2[2 * t], F4E2M1FNx2::new([row[0], row[1]]), "x2[{t}][0]");
            assert_eq!(x2[2 * t + 1], F4E2M1FNx2::new([row[2], row[3]]), "x2[{t}][1]");
            for k in 0..4 {
                let lanes: [F6E2M3FN; 4] = std::array::from_fn(|j| fp6[t * FP6_ROW + 4 * k + j]);
                assert_eq!(x4[4 * t + k], F6E2M3FNx4::new(lanes), "x4[{t}][{k}]");
                assert_eq!(x4[4 * t + k].lanes(), lanes);
            }
        }
    }
    Case { name: "mx_storage", kernel, cfg: launch(1), small_only: false, check }
}

/// Float16 results per lane of [`mx_conversions`]: 16 from FP4, 8 from
/// each FP6 format, and one from `cvt_f4e2m1_f16`.
const HALVES: usize = 16 + 8 + 8 + 1;

/// The f32 inputs of the conversion pairs of lane `t`'s value `x`.
fn pair_inputs(x: f32) -> [[f32; 2]; 4] {
    [[x, -x], [x * 0.25, x * 4.0], [x * 3.0, -x * 0.5], [-x * 16.0, x * 0.0625]]
}

/// The FP4 and FP6 `cvt` pairs (sm_100a, sm_101a and sm_120a): Float32 to
/// E2M1, E2M3 and E3M2 with and without `.relu` (NaN, infinities, ties and
/// MAX_NORM among the inputs), the results stored packed, and back to
/// Float16 (`cvt_f4e2m1x2_to_f16x2` and its x4 and x8 forms,
/// `cvt_f4e2m1_f16`, `cvt_f6x2_to_f16x2`).
pub fn mx_conversions() -> Case {
    let kernel = Kernel::builder("mx_conversions").block_dim([T as u32, 1, 1]).build(|| {
        let bits = param_tensor::<u32>("bits", Layout::col_major(T));
        let fp4 = param_tensor::<F4E2M1FN>("fp4", Layout::row_major((T, 16)));
        let e2m3 = param_tensor::<F6E2M3FN>("e2m3", Layout::row_major((T, 16)));
        let e3m2 = param_tensor::<F6E3M2FN>("e3m2", Layout::row_major((T, 16)));
        let halves = param_tensor::<F16>("halves", Layout::row_major((T, HALVES)));
        let t = thread_idx().x;
        let x = with_specials(t, bits.load(t)).bitcast::<f32>();
        let inputs = [[x, -x], [x * 0.25f32, x * 4.0f32], [x * 3.0f32, -x * 0.5f32], [-x * 16.0f32, x * 0.0625f32]];
        let a = make_fragment::<F4E2M1FN>(16);
        let b = make_fragment::<F6E2M3FN>(16);
        let c = make_fragment::<F6E3M2FN>(16);
        for relu in [false, true] {
            for (k, pair) in inputs.iter().enumerate() {
                let at = 8 * relu as usize + 2 * k;
                let [p, q] = cvt_f32x2_to_f4e2m1x2(*pair, relu);
                a.store(at, p);
                a.store(at + 1, q);
                let [p, q] = cvt_f32x2_to_f6x2::<F6E2M3FN>(*pair, relu);
                b.store(at, p);
                b.store(at + 1, q);
                let [p, q] = cvt_f32x2_to_f6x2::<F6E3M2FN>(*pair, relu);
                c.store(at, p);
                c.store(at + 1, q);
            }
        }
        autovec_copy(&a, &fp4.slice((t, ..)));
        autovec_copy(&b, &e2m3.slice((t, ..)));
        autovec_copy(&c, &e3m2.slice((t, ..)));
        // Back to Float16: FP4 by eights, fours, pairs (with and without
        // relu) and one at a time; FP6 by pairs.
        let mut out: Vec<Float16> = Vec::new();
        let lanes: Vec<Float4E2M1FN> = (0..16).map(|i| a.load(i)).collect();
        out.extend(cvt_f4e2m1x8_to_f16x8(std::array::from_fn(|i| lanes[i])));
        out.extend(cvt_f4e2m1x4_to_f16x4(std::array::from_fn(|i| lanes[8 + i])));
        let (lo, hi): (Float16, Float16) = cvt_f4e2m1x2_to_f16x2([lanes[12], lanes[13]]);
        let (rlo, rhi): (Float16, Float16) = cvt_f4e2m1x2_to_f16x2_with([lanes[14], lanes[15]], true);
        out.extend([lo, hi, rlo, rhi]);
        for (j, relu) in [false, true, false, true].into_iter().enumerate() {
            let (p, q): (Float16, Float16) = cvt_f6x2_to_f16x2([b.load(2 * j), b.load(2 * j + 1)], relu);
            out.extend([p, q]);
        }
        for (j, relu) in [false, true, true, false].into_iter().enumerate() {
            let (p, q): (Float16, Float16) = cvt_f6x2_to_f16x2([c.load(2 * j), c.load(2 * j + 1)], relu);
            out.extend([p, q]);
        }
        out.push(cvt_f4e2m1_f16(lanes[1]));
        for (i, v) in out.into_iter().enumerate() {
            halves.store((t, i as i32), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let bits = args[0].to_vec::<u32>();
        let (fp4, e2m3, e3m2) =
            (args[1].to_vec::<F4E2M1FN>(), args[2].to_vec::<F6E2M3FN>(), args[3].to_vec::<F6E3M2FN>());
        let halves = args[4].to_vec::<F16>();
        let rn = NarrowRounding::NearestEven;
        let half = |v: f64, relu: bool| F16::from_f32(if relu && v.is_sign_negative() { 0.0 } else { v as f32 });
        for t in 0..T {
            let x = f32::from_bits(host_specials(t, bits[t]));
            let mut want4 = Vec::new();
            let (mut want6, mut want6b) = (Vec::new(), Vec::new());
            for relu in [false, true] {
                for pair in pair_inputs(x) {
                    for v in pair {
                        want4.push(F4E2M1FN::from_f64_with(v as f64, rn, true, relu));
                        want6.push(F6E2M3FN::from_f64_with(v as f64, rn, true, relu));
                        want6b.push(F6E3M2FN::from_f64_with(v as f64, rn, true, relu));
                    }
                }
            }
            assert_eq!(&fp4[16 * t..16 * t + 16], &want4[..], "fp4 of lane {t} ({x:?})");
            assert_eq!(&e2m3[16 * t..16 * t + 16], &want6[..], "e2m3 of lane {t} ({x:?})");
            assert_eq!(&e3m2[16 * t..16 * t + 16], &want6b[..], "e3m2 of lane {t} ({x:?})");
            let h = &halves[HALVES * t..HALVES * (t + 1)];
            let mut k = 0;
            for (i, v) in want4.iter().enumerate() {
                let relu = i == 14 || i == 15;
                assert_eq!(h[k].0, half(v.to_f64(), relu).0, "fp4 {i} of lane {t} to f16");
                k += 1;
            }
            for (j, relu) in [false, true, false, true].into_iter().enumerate() {
                for i in [2 * j, 2 * j + 1] {
                    assert_eq!(h[k].0, half(want6[i].to_f64(), relu).0, "e2m3 {i} of lane {t} to f16");
                    k += 1;
                }
            }
            for (j, relu) in [false, true, true, false].into_iter().enumerate() {
                for i in [2 * j, 2 * j + 1] {
                    assert_eq!(h[k].0, half(want6b[i].to_f64(), relu).0, "e3m2 {i} of lane {t} to f16");
                    k += 1;
                }
            }
            assert_eq!(h[k].0, half(want4[1].to_f64(), false).0, "cvt_f4e2m1_f16 of lane {t}");
        }
    }
    Case { name: "mx_conversions", kernel, cfg: launch(1), small_only: false, check }
}
