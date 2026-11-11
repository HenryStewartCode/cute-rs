//! Runtime control flow.
//!
//! Plain Rust `if` and `for` run while the kernel is being built (CuTe DSL's
//! `range_constexpr`). The functions here instead emit control flow that runs
//! on the GPU (CuTe DSL's `cutlass.range` and dynamic `if`).
//!
//! Values are SSA: a value computed inside a branch or loop body cannot be
//! used after it. State that flows out of a region either lives in a [`Var`],
//! which every region threads through automatically (as CuTe DSL's
//! preprocessor does for the variables assigned in a dynamic region), or is
//! returned from the closure: [`if_else_yield`] yields it, and
//! [`for_range_carry`] and [`while_carry`] thread it through the iterations.
//! Both work with any type implementing [`Carry`]: values, tuples, arrays,
//! vectors, register tensors and structs declared with
//! [`carry_struct!`](crate::carry_struct).
//!
//! Register fragments hold SSA values too, so a runtime loop does not carry
//! a fragment created outside it: writing such a fragment's register after
//! the loop read it panics, because the next iteration would read the value
//! from before the loop. Keep the fragment in a [`Var`] instead.

use std::fmt;
use std::panic::Location;

use crate::builder::{self, CarriedVar, RegionKind, VarId};
use crate::ir::{DType, Loc, LoopOptions, Op, Region, ValueId};
use crate::value::{Boolean, Int32, IntoValue, Numeric, Value};

/// Types that can be yielded from regions, carried through loops and held in
/// a [`Var`]. [`carry_struct!`](crate::carry_struct) implements it for
/// structs.
pub trait Carry: Sized {
    /// Append the SSA values making up `self`.
    fn flatten(&self, out: &mut Vec<ValueId>);
    /// Rebuild a value shaped like `self` from `values`.
    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self;
    /// How `other` differs from `self` in what is not flattened but taken
    /// from `self` when rebuilding, such as a register tensor's layout, or
    /// `None` if they are alike. What a region yields or carries must keep
    /// the structure of what came in, which CuTe DSL treats as part of a
    /// value's type. The default compares nothing.
    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        let _ = other;
        None
    }
}

impl Carry for () {
    fn flatten(&self, _: &mut Vec<ValueId>) {}
    fn rebuild(&self, _: &mut dyn Iterator<Item = ValueId>) -> Self {}
}

impl<T: Numeric> Carry for Value<T> {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        out.push(self.checked());
    }
    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        Value::from_id(values.next().expect("carry: too few values"))
    }
}

/// The first element of `a` whose structure differs from `b`'s.
fn elements_mismatch<'a, C: Carry + 'a>(
    a: impl IntoIterator<Item = &'a C>,
    b: impl IntoIterator<Item = &'a C>,
) -> Option<String> {
    a.into_iter()
        .zip(b)
        .enumerate()
        .find_map(|(i, (x, y))| x.structure_mismatch(y).map(|m| format!("element {i}: {m}")))
}

// The `flatten`s track their caller (as do the operations that call them),
// so that a register tensor they read is reported at the user's line.
impl<C: Carry, const N: usize> Carry for [C; N] {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        for c in self {
            c.flatten(out);
        }
    }
    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        std::array::from_fn(|i| self[i].rebuild(values))
    }
    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        elements_mismatch(self, other)
    }
}

impl<C: Carry> Carry for Vec<C> {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        for c in self {
            c.flatten(out);
        }
    }
    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        self.iter().map(|c| c.rebuild(values)).collect()
    }
    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        if self.len() != other.len() {
            return Some(format!("a vector of {} elements became one of {}", self.len(), other.len()));
        }
        elements_mismatch(self, other)
    }
}

macro_rules! carry_tuple {
    ($(($($name:ident $idx:tt),+)),+) => {$(
        impl<$($name: Carry),+> Carry for ($($name,)+) {
            #[track_caller]
            fn flatten(&self, out: &mut Vec<ValueId>) {
                $(self.$idx.flatten(out);)+
            }
            fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
                ($(self.$idx.rebuild(values),)+)
            }
            fn structure_mismatch(&self, other: &Self) -> Option<String> {
                None$(.or_else(|| {
                    self.$idx.structure_mismatch(&other.$idx).map(|m| format!("element {}: {m}", $idx))
                }))+
            }
        }
    )+};
}
carry_tuple!((A 0), (A 0, B 1), (A 0, B 1, C 2), (A 0, B 1, C 2, D 3), (A 0, B 1, C 2, D 3, E 4), (A 0, B 1, C 2, D 3, E 4, F 5));

/// Implement [`Carry`] for a struct, so that it can be carried through
/// loops, yielded from branches and held in a [`Var`] (CuTe DSL carries
/// dataclasses the same way).
///
/// The first form declares the struct, and carries every field:
///
/// ```
/// use cute::prelude::*;
///
/// carry_struct! {
///     /// A running mean.
///     #[derive(Clone, Copy, Debug)]
///     pub struct Mean {
///         pub sum: Float32,
///         pub count: Int32,
///     }
/// }
///
/// let kernel = Kernel::builder("mean").block_dim([32, 1, 1]).build(|| {
///     let x = param_tensor::<f32>("x", Layout::col_major(32));
///     let out = param_tensor::<f32>("out", Layout::col_major(32));
///     let t = thread_idx().x;
///     let init = Mean { sum: Float32::from(0.0), count: Int32::from(0) };
///     let mean = for_range_carry(0, t + 1, 1, init, |i, m| Mean { sum: m.sum + x.load(i), count: m.count + 1 });
///     out.store(t, mean.sum / mean.count.to::<f32>());
/// });
/// let mut args = vec![KernelArg::f32((0..32).map(|i| i as f32).collect()), KernelArg::f32(vec![0.0; 32])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[1].as_f32()[4], 2.0);
/// ```
///
/// The second form implements `Carry` for an existing struct, which must be
/// `Clone`. The fields listed are carried; the others are build-time
/// constants, kept from the value being rebuilt. Generic parameters go in
/// brackets, and tuple-struct fields are numbers:
///
/// ```
/// use cute::prelude::*;
///
/// #[derive(Clone)]
/// struct Accumulator<T: Numeric> {
///     acc: Tensor<T>,
///     steps: Int32,
///     /// A build-time constant.
///     stages: usize,
/// }
/// carry_struct!([T: Numeric] Accumulator<T> { acc, steps });
///
/// #[derive(Clone, Copy)]
/// struct Pair(Int32, Float32);
/// carry_struct!(Pair { 0, 1 });
/// ```
///
/// The constant fields are part of the struct's type, as a dataclass's
/// static fields are in CuTe DSL: a loop or branch rebuilds the struct
/// with the constants it came in with, and a [`Var`] keeps those of its
/// initial value. Code must not change them in a runtime region; unlike
/// the carried fields' types and register tensor layouts, this is not
/// checked.
#[macro_export]
macro_rules! carry_struct {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident {
            $($(#[$field_attr:meta])* $field_vis:vis $field:ident : $ty:ty),* $(,)?
        }
    ) => {
        $(#[$attr])*
        $vis struct $name {
            $($(#[$field_attr])* $field_vis $field: $ty),*
        }

        impl $crate::Carry for $name {
            #[track_caller]
            fn flatten(&self, out: &mut ::std::vec::Vec<$crate::ir::ValueId>) {
                $($crate::Carry::flatten(&self.$field, out);)*
            }
            fn rebuild(&self, values: &mut dyn ::std::iter::Iterator<Item = $crate::ir::ValueId>) -> Self {
                $name { $($field: $crate::Carry::rebuild(&self.$field, values)),* }
            }
            fn structure_mismatch(&self, other: &Self) -> ::std::option::Option<::std::string::String> {
                let mismatch = ::std::option::Option::None;
                $(let mismatch = mismatch.or_else(|| {
                    $crate::Carry::structure_mismatch(&self.$field, &other.$field)
                        .map(|m| ::std::format!("field {}: {m}", ::std::stringify!($field)))
                });)*
                mismatch
            }
        }
    };
    ([$($generics:tt)*] $ty:ty { $($field:tt),* $(,)? }) => {
        impl<$($generics)*> $crate::Carry for $ty {
            #[track_caller]
            fn flatten(&self, out: &mut ::std::vec::Vec<$crate::ir::ValueId>) {
                $($crate::Carry::flatten(&self.$field, out);)*
            }
            fn rebuild(&self, values: &mut dyn ::std::iter::Iterator<Item = $crate::ir::ValueId>) -> Self {
                let mut rebuilt = ::std::clone::Clone::clone(self);
                $(rebuilt.$field = $crate::Carry::rebuild(&self.$field, values);)*
                rebuilt
            }
            fn structure_mismatch(&self, other: &Self) -> ::std::option::Option<::std::string::String> {
                let mismatch = ::std::option::Option::None;
                $(let mismatch = mismatch.or_else(|| {
                    $crate::Carry::structure_mismatch(&self.$field, &other.$field)
                        .map(|m| ::std::format!("field {}: {m}", ::std::stringify!($field)))
                });)*
                mismatch
            }
        }
    };
    ($ty:ty { $($field:tt),* $(,)? }) => {
        $crate::carry_struct!([] $ty { $($field),* });
    };
}

/// A mutable variable of the kernel being built, which runtime control flow
/// threads through automatically: CuTe DSL's loop-carried and branch-result
/// variables, which its preprocessor finds by rewriting the Python source.
///
/// * A Var touched (read or set) in the body of a runtime loop
///   ([`for_range`] and its variants, [`while_carry`]), or read in a
///   `while_carry` condition, is carried through the loop: its value at loop
///   entry is the initial value, its value at the end of the body is passed
///   to the next iteration, and after the loop it holds the final value (the
///   entry value if the loop runs zero times).
/// * A Var set in either branch of an `if` ([`if_`], [`if_else`],
///   [`if_else_yield`]) becomes a result of the `if`; the other branch
///   passes on the value the Var had before.
/// * Nested regions thread it through every level.
/// * A Var created inside a region is local to it.
///
/// A Var holds any [`Carry`] type: values, tuples, register tensors, and
/// structs declared with [`carry_struct!`](crate::carry_struct). Its type is
/// fixed when it is created: setting it to a value with other flattened
/// types (a longer vector, a register tensor of another size) or another
/// structure (a register tensor with another layout) panics with both
/// source locations. The build-time constant fields of a struct are those
/// of the initial value (see [`carry_struct!`](crate::carry_struct)). A
/// `while_carry` condition can read the Vars created outside it, but not
/// set them.
///
/// [`Var::get`] returns a copy: a register tensor read from a Var and
/// changed is stored in the Var only when it is set back ([`Var::update`]
/// does both). A value read from a Var inside a region belongs to that
/// region, like any other value computed there; read the Var again after
/// the region.
///
/// Inside a runtime loop a Var is a loop-carried value, so what is known
/// about its value at build time does not hold there: it is not a constant
/// and has no known divisibility or bound, even when the loop only reads
/// it (CuTe DSL's preprocessor sees in advance which variables a loop
/// assigns; the builder cannot). Read the Var before the loop and use that
/// value inside to keep what is known, for example an offset that a
/// vectorised copy must prove aligned (the copy's error names the Var when
/// this is why it cannot, as does indexing registers with a Var's value in
/// a loop, which is not static there). After an `if`, a Var holds what both
/// branches' values have in common (the larger bound and the common
/// divisor), or the value itself when both branches set it to the same one;
/// after a loop, the parts the loop left as they came in keep their value.
///
/// A register fragment that a runtime loop updates (reads, then writes)
/// must be kept in a Var: the loop does not carry a fragment created
/// outside it, and the builder panics when it sees such an update.
///
/// ```
/// use cute::prelude::*;
///
/// let kernel = Kernel::builder("triangle").block_dim([32, 1, 1]).build(|| {
///     let out = param_tensor::<i32>("out", Layout::col_major(32));
///     let t = thread_idx().x;
///     let sum = Var::new(Int32::from(0));
///     for_range(0, t, 1, |i| sum.set(sum.get() + i));
///     if_(sum.get().gt(100), || sum.set(100));
///     out.store(t, sum.get());
/// });
/// let mut args = vec![KernelArg::i32(vec![0; 32])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[0].as_i32()[10], 45);
/// assert_eq!(args[0].as_i32()[31], 100);
/// ```
pub struct Var<C: Carry> {
    id: VarId,
    kernel: u32,
    /// The initial value, which gives the shape values are rebuilt in.
    template: C,
}

impl<C: Carry> Var<C> {
    /// A variable holding `init`, local to the region it is created in.
    #[track_caller]
    pub fn new(init: C) -> Var<C> {
        let loc = Location::caller();
        let mut values = Vec::new();
        init.flatten(&mut values);
        let (id, kernel) = builder::with(|b| (b.var_new(values, loc), b.kernel));
        Var { id, kernel, template: init }
    }

    #[track_caller]
    fn check_kernel(&self) {
        let current = builder::with(|b| b.kernel);
        assert_eq!(self.kernel, current, "cute: a Var from another kernel was used");
    }

    /// The current value.
    #[track_caller]
    pub fn get(&self) -> C {
        let loc = Location::caller();
        self.check_kernel();
        let values = builder::with(|b| b.var_get(self.id, loc));
        self.template.rebuild(&mut values.into_iter())
    }

    /// Replace the value. Its flattened types and structure (register
    /// tensor layouts) must be the Var's.
    #[track_caller]
    pub fn set(&self, value: impl Into<C>) {
        let loc = Location::caller();
        let value = value.into();
        let mut values = Vec::new();
        value.flatten(&mut values);
        self.check_kernel();
        let mismatch = self.template.structure_mismatch(&value);
        builder::with(|b| b.var_set(self.id, values, mismatch, loc));
    }

    /// Replace the value with `f` of it.
    #[track_caller]
    pub fn update(&self, f: impl FnOnce(C) -> C) {
        let next = f(self.get());
        self.set(next);
    }
}

impl<C: Carry> fmt::Debug for Var<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Var({})", self.id.0)
    }
}

#[track_caller]
fn flatten_with_types<C: Carry>(c: &C) -> (Vec<ValueId>, Vec<DType>) {
    let mut values = Vec::new();
    c.flatten(&mut values);
    let types = builder::with(|b| values.iter().map(|&v| b.dtype(v)).collect());
    (values, types)
}

/// Panic if what a region produced (`produced`, the `what` of `op` at
/// `loc`) differs in structure from `expected`.
fn check_structure<C: Carry>(expected: &C, produced: &C, op: &str, what: &str, loc: Loc) {
    if let Some(mismatch) = expected.structure_mismatch(produced) {
        panic!("cute: {op} at {loc}: {what} ({mismatch}); a register tensor's layout is part of its type");
    }
}

/// Fresh values of the given types.
fn new_values(b: &mut builder::BuilderState, types: &[DType]) -> Vec<ValueId> {
    types.iter().map(|&t| b.new_value(t)).collect()
}

/// The value both `x` and `y` stand for, if they are the same value or the
/// same constant (made available in the current region).
fn same_value(b: &mut builder::BuilderState, x: ValueId, y: ValueId, loc: Loc) -> Option<ValueId> {
    if x == y {
        return Some(x);
    }
    match (b.constant_of(x), b.constant_of(y)) {
        (Some(c), Some(d)) if c == d => Some(b.constant(c, loc)),
        _ => None,
    }
}

/// What a Var carried through a loop holds after it: each part the body
/// left as it came in (or set back to its entry value) is the entry value,
/// and the others are the loop's results for it.
fn after_loop(b: &builder::BuilderState, carried: &CarriedVar, end: &[ValueId], results: &[ValueId]) -> Vec<ValueId> {
    let parts = carried.init.iter().zip(&carried.args).zip(end).zip(results);
    parts
        .map(|(((&init, &arg), &end), &result)| {
            let unchanged =
                end == arg || end == init || matches!(b.constant_of(end), Some(c) if b.constant_of(init) == Some(c));
            if unchanged { init } else { result }
        })
        .collect()
}

/// Run `then` when `cond` is true.
#[track_caller]
pub fn if_(cond: Boolean, then: impl FnOnce()) {
    if_else_yield(cond, then, || {})
}

/// Run `then` or `els` depending on `cond`.
#[track_caller]
pub fn if_else(cond: Boolean, then: impl FnOnce(), els: impl FnOnce()) {
    if_else_yield(cond, then, els)
}

/// Run one of two branches and return what it yields. Both branches must
/// yield the same shape of values. [`Var`]s set in either branch become
/// results too.
#[track_caller]
pub fn if_else_yield<R: Carry>(cond: Boolean, then: impl FnOnce() -> R, els: impl FnOnce() -> R) -> R {
    let loc = Location::caller();
    let cond = cond.checked();

    builder::with(|b| b.push_region(RegionKind::Branch));
    let then_result = then();
    let (then_yields, then_types) = flatten_with_types(&then_result);
    let then_region = builder::with(|b| b.pop_region());

    builder::with(|b| b.push_region(RegionKind::Branch));
    let else_result = els();
    let (else_yields, else_types) = flatten_with_types(&else_result);
    let else_region = builder::with(|b| b.pop_region());

    assert_eq!(then_types, else_types, "cute: if_else_yield branches yield different types at {loc}");
    check_structure(&then_result, &else_result, "if_else_yield", "the branches yield different structures", loc);
    let results: Vec<ValueId> = builder::with(|b| {
        let (mut then_yields, mut else_yields) = (then_yields, else_yields);
        let mut results = new_values(b, &then_types);
        // The Vars either branch set, in the order they were first set.
        let mut vars: Vec<VarId> = then_region.written.iter().map(|(v, _)| *v).collect();
        let else_only: Vec<VarId> = else_region.written.iter().map(|(v, _)| *v).filter(|v| !vars.contains(v)).collect();
        vars.extend(else_only);
        let mut var_results = Vec::new();
        for var in vars {
            let before = b.var_current(var);
            let then_value = then_region.written_value(var).unwrap_or(&before).clone();
            let else_value = else_region.written_value(var).unwrap_or(&before).clone();
            if then_value == before && else_value == before {
                continue;
            }
            // Only the parts the branches leave different become results.
            // A value both have is defined outside them (a branch cannot
            // see the other's values), and keeps what is known about it.
            let mut after = Vec::new();
            for (&tv, &ev) in then_value.iter().zip(&else_value) {
                match same_value(b, tv, ev, loc) {
                    Some(v) => after.push(v),
                    None => {
                        let r = b.new_value(b.dtype(tv));
                        then_yields.push(tv);
                        else_yields.push(ev);
                        results.push(r);
                        after.push(r);
                    }
                }
            }
            var_results.push((var, after));
        }
        for (k, &r) in results.iter().enumerate() {
            b.merge_facts(r, &[then_yields[k], else_yields[k]]);
        }
        b.push(
            Op::If {
                cond,
                then_region: Region { insts: then_region.insts, yields: then_yields },
                else_region: Region { insts: else_region.insts, yields: else_yields },
                results: results.clone(),
            },
            loc,
        );
        for (var, values) in var_results {
            b.var_assign(var, values);
        }
        results.truncate(then_types.len());
        results
    });
    then_result.rebuild(&mut results.into_iter())
}

/// A runtime loop `for i in (lower..upper).step_by(step)`. `step` must be
/// positive.
#[track_caller]
pub fn for_range(
    lower: impl IntoValue<i32>,
    upper: impl IntoValue<i32>,
    step: impl IntoValue<i32>,
    body: impl FnOnce(Int32),
) {
    for_range_carry_with(lower, upper, step, LoopOptions::default(), (), |i, ()| body(i))
}

/// [`for_range`] with unroll, vectorize and `at_least_once` hints: CuTe
/// DSL's `cutlass.range(start, stop, step, unroll=K, unroll_full=,
/// vectorize=, at_least_once=)`, for example
/// `for_range_with(0, n, 1, LoopOptions::unroll(4), |i| ...)`.
#[track_caller]
pub fn for_range_with(
    lower: impl IntoValue<i32>,
    upper: impl IntoValue<i32>,
    step: impl IntoValue<i32>,
    opts: LoopOptions,
    body: impl FnOnce(Int32),
) {
    for_range_carry_with(lower, upper, step, opts, (), |i, ()| body(i))
}

/// A runtime loop that threads `init` through the iterations: `body` gets
/// the induction variable and the current state and returns the next state.
/// Returns the final state.
#[track_caller]
pub fn for_range_carry<C: Carry>(
    lower: impl IntoValue<i32>,
    upper: impl IntoValue<i32>,
    step: impl IntoValue<i32>,
    init: C,
    body: impl FnOnce(Int32, C) -> C,
) -> C {
    for_range_carry_with(lower, upper, step, LoopOptions::default(), init, body)
}

/// [`for_range_carry`] with the hints of [`for_range_with`].
#[track_caller]
pub fn for_range_carry_with<C: Carry>(
    lower: impl IntoValue<i32>,
    upper: impl IntoValue<i32>,
    step: impl IntoValue<i32>,
    opts: LoopOptions,
    init: C,
    body: impl FnOnce(Int32, C) -> C,
) -> C {
    let loc = Location::caller();
    let (lower, upper, step) = (lower.into_value(), upper.into_value(), step.into_value());
    if let Some(s) = step.as_constant() {
        assert!(s > 0, "cute: for_range step must be positive, got {s}");
    }
    let (lo, hi, st) = (lower.checked(), upper.checked(), step.checked());
    let (init_values, types) = flatten_with_types(&init);

    let (iv, args) = builder::with(|b| {
        let iv = b.new_value(DType::I32);
        let args = new_values(b, &types);
        if let (Some(l), Some(u), Some(s)) = (b.constant_of(lo), b.constant_of(hi), b.constant_of(st)) {
            let (l, u, s) = (l.bits() as i32, u.bits() as i32, s.bits() as i32);
            if l >= 0 && u >= 0 {
                let divisor = gcd(l as usize, s as usize).max(1);
                b.set_facts(iv, builder::ValueFacts { bound: Some((u as usize).max(1)), divisor });
            }
        }
        b.push_region(RegionKind::Loop);
        (iv, args)
    });

    let carried = init.rebuild(&mut args.clone().into_iter());
    let next = body(Value::from_id(iv), carried);
    let (yields, yield_types) = flatten_with_types(&next);
    let region = builder::with(|b| b.pop_region());
    assert_eq!(yield_types, types, "cute: for_range_carry body must return state of the same shape as `init` ({loc})");
    check_structure(&init, &next, "for_range_carry", "the body returns state of another structure than `init`", loc);

    let results = builder::with(|b| {
        let (mut all_init, mut args, mut yields) = (init_values, args, yields);
        let mut results = new_values(b, &types);
        // The Vars the body touched follow the explicit state.
        let mut var_results = Vec::new();
        for (carried, end) in region.carried {
            let var_types = b.var_types(carried.var);
            let values = new_values(b, &var_types);
            all_init.extend(&carried.init);
            args.extend(&carried.args);
            yields.extend(&end);
            results.extend(&values);
            // A Var the body only read, or set back to the value it came
            // in with, leaves the loop as it came in.
            let after = after_loop(b, &carried, &end, &values);
            if after != carried.init {
                var_results.push((carried.var, after));
            }
        }
        // A result is the initial value or what the body yields, which
        // cannot rely on facts about the block arguments (they have none).
        for (k, &r) in results.iter().enumerate() {
            b.merge_facts(r, &[all_init[k], yields[k]]);
        }
        b.push(
            Op::For {
                iv,
                lower: lo,
                upper: hi,
                step: st,
                init: all_init,
                args,
                body: Region { insts: region.insts, yields },
                results: results.clone(),
                opts,
            },
            loc,
        );
        for (var, values) in var_results {
            b.var_assign(var, values);
        }
        results.truncate(types.len());
        results
    });
    init.rebuild(&mut results.into_iter())
}

/// A runtime `while` loop: `cond` sees the current state and decides whether
/// to continue, `body` turns it into the next state. Returns the state when
/// `cond` first fails (CuTe DSL's `while_generate`). [`Var`]s read in `cond`
/// or touched in `body` are carried too; `cond` cannot set them.
#[track_caller]
pub fn while_carry<C: Carry>(init: C, cond: impl FnOnce(C) -> Boolean, body: impl FnOnce(C) -> C) -> C {
    let loc = Location::caller();
    let (init_values, types) = flatten_with_types(&init);

    let cond_args: Vec<ValueId> = builder::with(|b| {
        b.push_region(RegionKind::WhileCond);
        new_values(b, &types)
    });
    let condition = cond(init.rebuild(&mut cond_args.clone().into_iter())).checked();
    let cond_region = builder::with(|b| b.pop_region());

    let body_args: Vec<ValueId> = builder::with(|b| {
        b.push_region(RegionKind::Loop);
        // The body carries the Vars the condition read first, in its order.
        for (carried, _) in &cond_region.carried {
            b.var_seed_carried(carried.var, carried.init.clone());
        }
        // A register the condition read must not be written by the body.
        b.seed_fragment_reads(&cond_region.fragment_reads);
        new_values(b, &types)
    });
    let next = body(init.rebuild(&mut body_args.clone().into_iter()));
    let (yields, yield_types) = flatten_with_types(&next);
    let body_region = builder::with(|b| b.pop_region());
    assert_eq!(yield_types, types, "cute: while_carry body must return state of the same shape as `init` ({loc})");
    check_structure(&init, &next, "while_carry", "the body returns state of another structure than `init`", loc);

    let results = builder::with(|b| {
        let (mut all_init, mut cond_args, mut body_args, mut yields) = (init_values, cond_args, body_args, yields);
        let mut results = new_values(b, &types);
        let mut var_results = Vec::new();
        for (k, (carried, end)) in body_region.carried.into_iter().enumerate() {
            let var_types = b.var_types(carried.var);
            // The condition's view of the Var: its own block arguments if it
            // read the Var, otherwise new ones it does not use.
            let cond_view = match cond_region.carried.get(k) {
                Some((c, _)) => {
                    debug_assert_eq!(c.var, carried.var);
                    c.args.clone()
                }
                None => new_values(b, &var_types),
            };
            let values = new_values(b, &var_types);
            all_init.extend(&carried.init);
            cond_args.extend(cond_view);
            body_args.extend(&carried.args);
            yields.extend(&end);
            results.extend(&values);
            let after = after_loop(b, &carried, &end, &values);
            if after != carried.init {
                var_results.push((carried.var, after));
            }
        }
        // The loop ends with the initial values or what the body yields.
        for (k, &r) in results.iter().enumerate() {
            b.merge_facts(r, &[all_init[k], yields[k]]);
        }
        b.push(
            Op::While {
                init: all_init,
                cond_args,
                cond: Region { insts: cond_region.insts, yields: vec![condition] },
                body_args,
                body: Region { insts: body_region.insts, yields },
                results: results.clone(),
            },
            loc,
        );
        for (var, values) in var_results {
            b.var_assign(var, values);
        }
        results.truncate(types.len());
        results
    });
    init.rebuild(&mut results.into_iter())
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}
