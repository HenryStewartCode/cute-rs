//! [`TensorSsa::to`] and [`TensorSsa::bitcast`].

use std::panic::Location;

use super::TensorSsa;
use super::ops::retype;
use crate::arch::{
    cvt_f16x2_to_f8x2, cvt_f32x2_to_f4e2m1x2, cvt_f32x2_to_f6x2, cvt_f32x2_to_f8x2, cvt_i4_bf16_intrinsic,
    cvt_i8_bf16_intrinsic,
};
use crate::builder::{self, LocationScope};
use crate::ir::DType;
use crate::numeric::{F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E4M3FN, F8E5M2};
use crate::value::{F16, Numeric, Value};

/// Whether [`convert_values`] converts `from` to `to` other than with
/// [`Value::to`] per element.
pub(crate) fn has_wide_conversion(from: DType, to: DType) -> bool {
    from != to
        && (matches!(to, DType::F4E2M1FN | DType::F6E2M3FN | DType::F6E3M2FN | DType::F8E5M3FNU)
            || matches!(to, DType::F8E4M3FN | DType::F8E5M2)
            || (to == DType::BF16 && matches!(from, DType::I8 | DType::I4)))
}

/// `xs` converted to `U` as [`TensorSsa::to`] converts: [`Value::to`] per
/// element, except where a wider instruction gives the same bits (FP8 from
/// Float32 or Float16 two at a time, Int8 and Int4 to BFloat16 with the
/// Ampere fast paths) or the only one there is (FP4 and FP6, in pairs).
#[track_caller]
pub(crate) fn convert_values<T: Numeric, U: Numeric>(xs: &[Value<T>], what: &str) -> Vec<Value<U>> {
    let _at = LocationScope::new();
    let (from, to) = (T::DTYPE, U::DTYPE);
    if from == to {
        return xs.iter().map(|&v| retype(v)).collect();
    }
    let pairs = |f: &mut dyn FnMut(Value<f32>, Value<f32>) -> [Value<U>; 2]| -> Vec<Value<U>> {
        let wide: Vec<Value<f32>> = xs.iter().map(|v| v.to::<f32>()).collect();
        wide.chunks(2).flat_map(|p| if p.len() == 2 { f(p[0], p[1]).to_vec() } else { vec![p[0].to::<U>()] }).collect()
    };
    match to {
        DType::F4E2M1FN | DType::F6E2M3FN | DType::F6E3M2FN => {
            assert!(
                xs.len().is_multiple_of(2),
                "{what}: {} values convert to {} in pairs (cvt.rn.satfinite.{}x2.f32), so the element count must be \
                 even, not {}",
                to.cute_name(),
                to.cute_name(),
                match to {
                    DType::F4E2M1FN => "e2m1",
                    DType::F6E2M3FN => "e2m3",
                    _ => "e3m2",
                },
                xs.len()
            );
            pairs(&mut |a, b| match to {
                DType::F4E2M1FN => cvt_f32x2_to_f4e2m1x2([a, b], false).map(retype::<F4E2M1FN, U>),
                DType::F6E2M3FN => cvt_f32x2_to_f6x2::<F6E2M3FN>([a, b], false).map(retype),
                _ => cvt_f32x2_to_f6x2::<F6E3M2FN>([a, b], false).map(retype),
            })
        }
        DType::F8E5M3FNU => {
            panic!("{what}: a conversion to FloatNV8E5M3FNU is not supported here; call cvt_f32x4_to_fnv8e5m3x4")
        }
        DType::F8E4M3FN | DType::F8E5M2 if from == DType::F16 => xs
            .chunks(2)
            .flat_map(|p| {
                if p.len() == 1 {
                    return vec![p[0].to::<U>()];
                }
                let (a, b): (Value<F16>, Value<F16>) = (retype(p[0]), retype(p[1]));
                if to == DType::F8E4M3FN {
                    cvt_f16x2_to_f8x2::<F8E4M3FN>((a, b), false).map(retype).to_vec()
                } else {
                    cvt_f16x2_to_f8x2::<F8E5M2>((a, b), false).map(retype).to_vec()
                }
            })
            .collect(),
        DType::F8E4M3FN => pairs(&mut |a, b| cvt_f32x2_to_f8x2::<F8E4M3FN>([a, b], false).map(retype)),
        DType::F8E5M2 => pairs(&mut |a, b| cvt_f32x2_to_f8x2::<F8E5M2>([a, b], false).map(retype)),
        DType::BF16 if from == DType::I8 => {
            let ints: Vec<Value<i8>> = xs.iter().map(|&v| retype(v)).collect();
            cvt_i8_bf16_intrinsic(&ints, ints.len()).into_iter().map(retype).collect()
        }
        DType::BF16 if from == DType::I4 => {
            let ints: Vec<Value<crate::numeric::I4>> = xs.iter().map(|&v| retype(v)).collect();
            cvt_i4_bf16_intrinsic(&ints, ints.len(), false).into_iter().map(retype).collect()
        }
        _ => xs.iter().map(|v| v.to::<U>()).collect(),
    }
}

impl<T: Numeric> TensorSsa<T> {
    /// Every element converted to `U` (CuTe DSL's `TensorSSA.to(dtype)`),
    /// with the same shape. Each element converts as [`Value::to`] does,
    /// with these instructions where they apply:
    ///
    /// * Float32 (and the other types, through Float32) to Float8E4M3FN or
    ///   Float8E5M2 two at a time, `cvt.rn.satfinite.e4m3x2.f32` (sm_89),
    ///   and Float16 with the `.f16x2` form: the same bits as one at a time;
    /// * Int8 to BFloat16 with [`cvt_i8_bf16_intrinsic`] (`dp4a` sign
    ///   extension, `cvt.rn.f32.s32`, `cvt.rn.satfinite.bf16x2.f32`) and Int4
    ///   to BFloat16 with [`cvt_i4_bf16_intrinsic`] (`lop3`, `prmt`,
    ///   `fma.rn.bf16x2`), CuTe DSL's Ampere fast paths, which are exact, so
    ///   the same bits as the plain conversion, on every target (sm_80 on).
    ///   Uint8 converts plainly (the Int8 path would read it as signed);
    /// * Float4E2M1FN, Float6E2M3FN and Float6E3M2FN, which are not scalar
    ///   [`Value::to`] destinations, two at a time through Float32
    ///   (`cvt.rn.satfinite.{e2m1x2,e2m3x2,e3m2x2}.f32`, sm_100a and sm_120a,
    ///   PTX 8.6), so the element count must be even.
    ///
    /// FloatNV8E5M3FNU is not supported here (use `cvt_f32x4_to_fnv8e5m3x4`): the
    /// build panics, naming it.
    #[track_caller]
    pub fn to<U: Numeric>(&self) -> TensorSsa<U> {
        TensorSsa::from_parts(self.shape.clone(), convert_values::<T, U>(&self.values, "TensorSsa::to"))
    }

    /// The bits reinterpreted as `U` (CuTe DSL's `TensorSSA.bitcast(dtype)`):
    /// the total bit count is kept and the element count scaled, as a
    /// `vector.bitcast` does, element 0 in the low bits (little-endian), so
    /// four Float32 become eight Float16 (each Float32 split, its low half
    /// first). The result's shape is flat, `(new_count,)`. Equal widths
    /// reinterpret element by element; otherwise one width must divide the
    /// other and the element count be a whole number of groups (packed with
    /// [`pack`](crate::pack) or split with [`unpack`](crate::unpack)).
    /// Boolean has no bit width in registers and cannot be bitcast.
    #[track_caller]
    pub fn bitcast<U: Numeric>(&self) -> TensorSsa<U> {
        let loc = Location::caller();
        let (tb, ub) = (T::DTYPE.bits(), U::DTYPE.bits());
        let n = self.size();
        assert!(
            T::DTYPE != DType::Bool && U::DTYPE != DType::Bool,
            "TensorSsa::bitcast: Boolean values have no bit width in registers; convert with to() instead"
        );
        let total = n * tb;
        assert!(
            total.is_multiple_of(ub) && (tb.is_multiple_of(ub) || ub.is_multiple_of(tb)),
            "TensorSsa::bitcast: {n} {} values are {total} bits, which {} values of {ub} bits cannot hold exactly \
             (one width must divide the other, and the total bit count be a whole number of both)",
            T::DTYPE.cute_name(),
            U::DTYPE.cute_name()
        );
        let ids: Vec<_> = self.values.iter().map(|v| v.checked()).collect();
        let out: Vec<Value<U>> = if tb == ub {
            self.values.iter().map(|v| v.bitcast::<U>()).collect()
        } else if ub > tb {
            let k = ub / tb;
            ids.chunks(k).map(|c| Value::from_id(builder::with(|b| b.pack(c.to_vec(), U::DTYPE, loc)))).collect()
        } else {
            let k = tb / ub;
            ids.iter().flat_map(|&x| builder::with(|b| b.unpack(x, U::DTYPE, k, loc))).map(Value::from_id).collect()
        };
        TensorSsa::from_parts(crate::IntTuple::from(out.len()), out)
    }
}
