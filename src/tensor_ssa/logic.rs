//! Comparisons, [`where_`], [`any_`] and [`all_`], reductions and the
//! `cute.math` functions over [`TensorSsa`].

use super::ops::{Common, SsaOperand, SsaOperandKind};
use super::{FillValue, TensorSsa, broadcast_shape};
use crate::builder::LocationScope;
use crate::elementwise;
use crate::ir::{CmpPred, DType};
use crate::layout::{IndexExpr, Layout, SliceCoord, Tree};
use crate::math::{self, MathOptions};
use crate::promote::CommonType;
use crate::value::{Arith, Boolean, Float, Numeric, Value, compare_values, select};

/// IEEE 754 `maximum` (CuTe DSL's `cutlass.max` on tensors, `arith.maximumf`):
/// NaN if either operand is NaN, and `+0 > -0`; for integers the larger
/// value. [`math::max_with`] with `propagate_nan` (`max.NaN.f32` on
/// Float32, an expansion on the other float types). Usable with
/// [`TensorSsa::apply_op`].
#[track_caller]
pub fn maximum<T: Arith>(a: Value<T>, b: Value<T>) -> Value<T> {
    math::max_with(a, b, MathOptions { propagate_nan: true, ..Default::default() })
}

/// IEEE 754 `minimum` (CuTe DSL's `cutlass.min` on tensors,
/// `arith.minimumf`): NaN if either operand is NaN, and `-0 < +0`; for
/// integers the smaller value. See [`maximum`].
#[track_caller]
pub fn minimum<T: Arith>(a: Value<T>, b: Value<T>) -> Value<T> {
    math::min_with(a, b, MathOptions { propagate_nan: true, ..Default::default() })
}

macro_rules! comparisons {
    ($($(#[$doc:meta])* $name:ident => $pred:expr, $ordered:expr;)*) => {
        impl<T: Numeric> TensorSsa<T> {$(
            $(#[$doc])*
            #[track_caller]
            pub fn $name<X>(&self, other: X) -> TensorSsa<bool>
            where
                X: SsaOperand<T>,
                T: CommonType<X::Elem>,
            {
                if $ordered {
                    assert!(
                        Common::<T, X>::DTYPE != DType::Bool,
                        "TensorSsa::{}: Boolean tensors are compared only with eq and ne, not ordered",
                        stringify!($name)
                    );
                }
                self.apply_op(other, false, |a, b| compare_values($pred, a, b))
            }
        )*}
    };
}

comparisons! {
    /// `self == other` element by element, after promotion and broadcasting
    /// (CuTe DSL's `==` on tensors): a Boolean tensor.
    eq => CmpPred::Eq, false;
    /// `self != other` (true where either is NaN).
    ne => CmpPred::Ne, false;
    /// `self < other` (false where either is NaN). Not for Boolean tensors.
    lt => CmpPred::Lt, true;
    /// `self <= other`.
    le => CmpPred::Le, true;
    /// `self > other`.
    gt => CmpPred::Gt, true;
    /// `self >= other`.
    ge => CmpPred::Ge, true;
}

/// `cond ? x : y` element by element (CuTe DSL's `cute.where`; `where` is a
/// Rust keyword). `x` and `y` are tensors, [`Value`]s or host numbers of one
/// type `T`, not promoted (CuTe DSL requires one dtype); at least one is a
/// tensor, and a scalar becomes a tensor of that tensor's shape. The three
/// broadcast to one shape, and each element is one `select`.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("relu", || {
///     let x = param_tensor::<f32>("x", Layout::col_major(8));
///     let out = param_tensor::<f32>("out", Layout::col_major(8));
///     let t = x.load_ssa();
///     out.store_ssa(&where_(&t.gt(0.0), &t, 0.0));
/// });
/// ```
///
/// The operands must have one type:
///
/// ```compile_fail
/// use cute::prelude::*;
/// Kernel::build("where_types", || {
///     let a = full::<f32>(4, 1.0);
///     let b = full::<F16>(4, 1.0);
///     let _ = where_(&a.gt(0.0), &a, &b);
/// });
/// ```
///
/// and the condition must be Boolean:
///
/// ```compile_fail
/// use cute::prelude::*;
/// Kernel::build("where_cond", || {
///     let a = full::<f32>(4, 1.0);
///     let _ = where_(&a, &a, 0.0);
/// });
/// ```
#[track_caller]
pub fn where_<T: Numeric, X, Y>(cond: &TensorSsa<bool>, x: X, y: Y) -> TensorSsa<T>
where
    X: SsaOperand<T, Elem = T>,
    Y: SsaOperand<T, Elem = T>,
{
    let _at = LocationScope::new();
    let (x, y) = (x.into_operand(), y.into_operand());
    let shape = match (&x, &y) {
        (SsaOperandKind::Tensor(t), _) | (_, SsaOperandKind::Tensor(t)) => t.shape.clone(),
        _ => panic!(
            "where_: neither x nor y is a tensor; one of them must be (CuTe DSL's cute.where), or use select for \
             scalars"
        ),
    };
    let (x, y) = (x.tensor_like(&shape), y.tensor_like(&shape));
    let shape = broadcast_shape(&[&cond.shape, &x.shape, &y.shape], "where_");
    let c = cond.broadcast_to_shape(&shape, "where_");
    let (x, y) = (x.broadcast_to_shape(&shape, "where_"), y.broadcast_to_shape(&shape, "where_"));
    let values = (0..shape.size()).map(|i| select(c.values[i], x.values[i], y.values[i])).collect();
    TensorSsa::from_parts(shape, values)
}

/// Every element compared with zero (`x != 0`: NaN counts as true, `-0.0`
/// as false).
#[track_caller]
fn nonzero<T: Numeric>(x: &TensorSsa<T>) -> Vec<Boolean> {
    let zero = Value::constant(T::from_bits(0));
    x.values.iter().map(|&v| compare_values(CmpPred::Ne, v, zero)).collect()
}

/// Whether any element is nonzero (CuTe DSL's `cute.any_`): `x != 0` (NaN
/// counts, `-0.0` does not), OR-ed in a balanced tree.
#[track_caller]
pub fn any_<T: Numeric>(x: &TensorSsa<T>) -> Boolean {
    let _at = LocationScope::new();
    elementwise::tree_fold(&nonzero(x), &mut |a, b| a | b)
}

/// Whether every element is nonzero (CuTe DSL's `cute.all_`), as [`any_`]
/// with AND.
#[track_caller]
pub fn all_<T: Numeric>(x: &TensorSsa<T>) -> Boolean {
    let _at = LocationScope::new();
    elementwise::tree_fold(&nonzero(x), &mut |a, b| a & b)
}

/// CuTe DSL's `cute.ReductionOp`: how [`TensorSsa::reduce`] combines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    /// `acc + x`.
    Add,
    /// `acc * x`.
    Mul,
    /// [`maximum`]: NaN-propagating for floats (`max.NaN.f32` on Float32),
    /// the larger value for integers.
    Max,
    /// [`minimum`].
    Min,
}

impl ReductionOp {
    #[track_caller]
    pub(crate) fn combine<T: Arith>(self, acc: Value<T>, x: Value<T>) -> Value<T> {
        match self {
            ReductionOp::Add => acc + x,
            ReductionOp::Mul => acc * x,
            ReductionOp::Max => maximum(acc, x),
            ReductionOp::Min => minimum(acc, x),
        }
    }
}

impl<T: Arith> TensorSsa<T> {
    /// Every element folded into one value (CuTe DSL's `reduce` with a
    /// depth-0 profile): `acc = init`, then `acc = acc op x` for every
    /// element in colexicographic order, exactly that sequence (no
    /// reassociation), so the simulator and the GPU agree bit for bit.
    /// `init` is converted to `T`. Integer `Max` and `Min` are a cute-rs
    /// extension (CuTe DSL 4.8.1's `TensorSSA.reduce` takes floats there).
    #[track_caller]
    pub fn reduce_all(&self, op: ReductionOp, init: impl FillValue<T>) -> Value<T> {
        let _at = LocationScope::new();
        let init = init.fill_value();
        elementwise::seq_fold(init, &self.values, &mut |a, x| op.combine(a, x))
    }

    /// The modes `profile` marks reduced, folded (CuTe DSL's
    /// `TensorSSA.reduce(op, init_val, reduction_profile)`): `..` (`None`)
    /// leaves keep a mode and any index leaf reduces one (a leaf against a
    /// nested mode reduces all of it); the profile must be weakly congruent
    /// with the shape. The result's shape is the kept modes, flattened as
    /// [`slice`](TensorSsa::slice) gives them. Each result element is
    /// `init` combined with the reduced elements one at a time, as in
    /// [`reduce_all`](TensorSsa::reduce_all), in the order CuTe DSL's
    /// `vector.multi_reduction` takes them: lexicographic over the reduced
    /// leaves (the last fastest; one reduced leaf is simply its order). A profile of `..` alone
    /// keeps the tensor; one that is a single index is
    /// [`reduce_all`](TensorSsa::reduce_all).
    ///
    /// ```
    /// use cute::prelude::*;
    /// Kernel::build("row_max", || {
    ///     let x = param_tensor::<f32>("x", Layout::row_major((2, 8)));
    ///     let out = param_tensor::<f32>("out", Layout::col_major(2));
    ///     let t = x.load_ssa();
    ///     // The maximum of each row: mode 1 reduced, mode 0 kept.
    ///     let m = t.reduce(ReductionOp::Max, f32::NEG_INFINITY, (.., 0));
    ///     out.store_ssa(&m);
    /// });
    /// ```
    #[track_caller]
    pub fn reduce(&self, op: ReductionOp, init: impl FillValue<T>, profile: impl Into<SliceCoord>) -> TensorSsa<T> {
        let _at = LocationScope::new();
        let profile = profile.into();
        match &profile {
            Tree::Leaf(None) => return self.clone(),
            Tree::Leaf(Some(_)) => {
                panic!(
                    "TensorSsa::reduce: the profile {} reduces every mode; use reduce_all for a scalar",
                    super::slice_text(&profile)
                )
            }
            Tree::Branch(_) => {}
        }
        assert!(
            profile.weakly_congruent(&self.shape),
            "TensorSsa::reduce: the profile {} is not weakly congruent with the shape {:?}",
            super::slice_text(&profile),
            self.shape
        );
        let init = init.fill_value();
        let zero = || Some(IndexExpr::Static(0));
        let kept_coord = profile.map(&mut |l| l.as_ref().map(|_| IndexExpr::Static(0)));
        let reduced_coord: SliceCoord = profile.map(&mut |l| if l.is_some() { None } else { zero() });
        let layout = Layout::col_major(self.shape.clone());
        let (kept, _) = layout.slice_checked(&kept_coord, "TensorSsa::reduce");
        let (reduced, _) = layout.slice_checked(&reduced_coord, "TensorSsa::reduce");
        let shape = super::static_shape(kept.shape());
        // CuTe DSL folds the reduced elements as `vector.multi_reduction`
        // over the row-major vector of the flattened shape does: in
        // lexicographic order of the reduced leaves (the last fastest).
        let order = lexicographic_order(&super::static_shape(reduced.shape()).flat_leaves());
        let values = (0..kept.size())
            .map(|o| {
                let base = kept.call(o);
                order.iter().fold(init, |acc, &r| op.combine(acc, self.values[base + reduced.call(r)]))
            })
            .collect();
        TensorSsa::from_parts(shape, values)
    }

    /// The larger of `self` and `other` element by element, after promotion
    /// and broadcasting, NaN-propagating as CuTe DSL's `cutlass.max` on
    /// tensors is ([`maximum`]; [`Value::max`] ignores a NaN instead).
    #[track_caller]
    pub fn max<X>(&self, other: X) -> TensorSsa<Common<T, X>>
    where
        X: SsaOperand<T>,
        T: CommonType<X::Elem>,
        Common<T, X>: Arith,
    {
        self.apply_op(other, false, maximum)
    }

    /// The smaller of `self` and `other`, NaN-propagating ([`minimum`]).
    #[track_caller]
    pub fn min<X>(&self, other: X) -> TensorSsa<Common<T, X>>
    where
        X: SsaOperand<T>,
        T: CommonType<X::Elem>,
        Common<T, X>: Arith,
    {
        self.apply_op(other, false, minimum)
    }

    /// `self` clamped to `[lo, hi]` with [`math::clamp`], broadcasting the
    /// three (of this tensor's type).
    #[track_caller]
    pub fn clamp<X, Y>(&self, lo: X, hi: Y, fastmath: bool) -> TensorSsa<T>
    where
        X: SsaOperand<T, Elem = T>,
        Y: SsaOperand<T, Elem = T>,
    {
        self.zip3(lo, hi, "TensorSsa::clamp", |x, l, h| math::clamp(x, l, h, fastmath))
    }

    /// `f(self, b, c)` element by element over the three broadcast to one
    /// shape (a scalar `b` or `c` broadcast to this tensor's shape first).
    #[track_caller]
    fn zip3<X, Y>(
        &self,
        b: X,
        c: Y,
        what: &str,
        mut f: impl FnMut(Value<T>, Value<T>, Value<T>) -> Value<T>,
    ) -> TensorSsa<T>
    where
        X: SsaOperand<T, Elem = T>,
        Y: SsaOperand<T, Elem = T>,
    {
        let _at = LocationScope::new();
        let (b, c) = (b.into_operand().tensor_like(&self.shape), c.into_operand().tensor_like(&self.shape));
        let shape = broadcast_shape(&[&self.shape, &b.shape, &c.shape], what);
        let (a, b, c) = (
            self.broadcast_to_shape(&shape, what),
            b.broadcast_to_shape(&shape, what),
            c.broadcast_to_shape(&shape, what),
        );
        let values = (0..shape.size()).map(|i| f(a.values[i], b.values[i], c.values[i])).collect();
        TensorSsa::from_parts(shape, values)
    }
}

/// The colexicographic indices of an array of leaf extents `extents`,
/// listed in lexicographic order (the last leaf fastest).
fn lexicographic_order(extents: &[usize]) -> Vec<usize> {
    let n: usize = extents.iter().product();
    (0..n)
        .map(|mut r| {
            // Peel the lexicographic coordinate from the last leaf, then
            // weigh it colexicographically.
            let mut coord = vec![0; extents.len()];
            for (c, &e) in coord.iter_mut().zip(extents).rev() {
                *c = r % e;
                r /= e;
            }
            let mut k = 0;
            let mut stride = 1;
            for (&c, &e) in coord.iter().zip(extents) {
                k += c * stride;
                stride *= e;
            }
            k
        })
        .collect()
}

macro_rules! unary_math {
    ($($name:ident),*) => {$(
        #[doc = concat!("[`math::", stringify!($name), "`] of every element (CuTe DSL's `cute.math.",
            stringify!($name), "` of a tensor).")]
        #[track_caller]
        pub fn $name(&self, fastmath: bool) -> TensorSsa<T> {
            self.map(|x| math::$name(x, fastmath))
        }
    )*};
}

macro_rules! predicate_math {
    ($($name:ident),*) => {$(
        #[doc = concat!("[`math::", stringify!($name), "`] of every element: a Boolean tensor.")]
        #[track_caller]
        pub fn $name(&self, fastmath: bool) -> TensorSsa<bool> {
            self.map(|x| math::$name(x, fastmath))
        }
    )*};
}

macro_rules! binary_math {
    ($($name:ident),*) => {$(
        #[doc = concat!("[`math::", stringify!($name), "`] element by element, after promotion and ",
            "broadcasting (as [`apply_op`](TensorSsa::apply_op)).")]
        #[track_caller]
        pub fn $name<X>(&self, other: X, fastmath: bool) -> TensorSsa<Common<T, X>>
        where
            X: SsaOperand<T>,
            T: CommonType<X::Elem>,
            Common<T, X>: Float,
        {
            self.apply_op(other, false, |a, b| math::$name(a, b, fastmath))
        }
    )*};
}

/// The `cute.math` functions over a tensor (CuTe DSL's math functions
/// accept a `TensorSSA`): one scalar function per element, with the
/// scalar's precision and `fastmath`.
impl<T: Float> TensorSsa<T> {
    unary_math!(
        exp, exp2, expm1, log, log2, log10, log1p, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh,
        atanh, erf, erfc, cbrt, sqrt, rsqrt, rcp, round, roundeven, floor, ceil, trunc
    );
    predicate_math!(isnan, isinf, isfinite, isnormal);
    binary_math!(atan2, copysign);

    /// `self * b + c` with one rounding ([`math::fma`]), broadcasting the
    /// three (of this tensor's type).
    #[track_caller]
    pub fn fma<X, Y>(&self, b: X, c: Y, fastmath: bool) -> TensorSsa<T>
    where
        X: SsaOperand<T, Elem = T>,
        Y: SsaOperand<T, Elem = T>,
    {
        self.zip3(b, c, "TensorSsa::fma", |x, y, z| math::fma(x, y, z, fastmath))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn lexicographic_order() {
        // Extents 2 and 3, the second leaf fastest: colex 0, 2, 4, then 1, 3, 5.
        assert_eq!(super::lexicographic_order(&[2, 3]), vec![0, 2, 4, 1, 3, 5]);
        assert_eq!(super::lexicographic_order(&[4]), vec![0, 1, 2, 3]);
    }
}
