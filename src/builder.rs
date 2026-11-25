//! Kernel construction.
//!
//! A kernel is built by running ordinary Rust code inside [`Kernel::build`].
//! DSL operations (arithmetic on [`Value`](crate::Value)s, tensor loads and
//! stores, copies, MMAs, control flow) append instructions to a builder that
//! lives in a thread-local for the duration of the build, which is what lets
//! operators like `a + b` record IR. Plain Rust control flow runs at build
//! time: a Rust `for` loop unrolls, a Rust `if` specialises.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::ir::{
    BinaryOp, BufferDecl, BufferId, CmpPred, CvtFn, DType, FloatFn, FloatMode, HalfFn, HalfModifiers, HalfType, Inst,
    IntFn, KernelIr, Literal, Loc, MemSpace, Op, Param, Region, SpecialReg, UnaryOp, ValueId,
};
use crate::layout::IndexExpr;

/// What is statically known about a runtime integer value: CuTe DSL's
/// constrained integers (the divisibility `cute.assume` attaches and
/// `cute.get_divisibility` reads), plus a bound.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValueFacts {
    /// Exclusive upper bound.
    pub bound: Option<usize>,
    /// Every value is a multiple of this.
    pub divisor: usize,
    /// Every value is a power of two (`cute.assume(x, pow2=True)`).
    pub pow2: bool,
}

impl Default for ValueFacts {
    fn default() -> Self {
        ValueFacts { bound: None, divisor: 1, pow2: false }
    }
}

impl ValueFacts {
    /// Whether the facts say anything (and so are worth keeping).
    fn is_informative(&self) -> bool {
        self.bound.is_some() || self.divisor > 1 || self.pow2
    }
}

/// Key for common-subexpression elimination of pure operations.
#[derive(Clone, PartialEq, Eq, Hash)]
enum PureKey {
    Const(Literal),
    Unary(UnaryOp, ValueId),
    Binary(BinaryOp, ValueId, ValueId),
    Fma(ValueId, ValueId, ValueId),
    Float(FloatFn, FloatMode, bool, bool, bool, Vec<ValueId>),
    Int(IntFn, DType, Vec<ValueId>),
    Cmp(CmpPred, ValueId, ValueId),
    Select(ValueId, ValueId, ValueId),
    Cast(DType, ValueId),
    Bitcast(DType, ValueId),
    Special(SpecialReg),
    Pack(DType, Vec<ValueId>),
    /// The part type and count; the results are consecutive values.
    Unpack(DType, usize, ValueId),
    Half(HalfFn, HalfType, HalfModifiers, Vec<ValueId>, Option<ValueId>),
    Cvt(CvtFn, Vec<ValueId>),
    Assume(ValueId, u64, bool),
}

/// What a region of the kernel is, which decides what happens to the
/// [`Var`](crate::Var)s used in it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum RegionKind {
    /// The kernel body.
    #[default]
    Root,
    /// A branch of an `if`: the Vars it sets become results of the `if`.
    Branch,
    /// The body of a `for` or `while` loop: the Vars it touches are carried
    /// through the loop.
    Loop,
    /// The condition of a `while` loop: the Vars it reads are carried
    /// through the loop, and it cannot set them.
    WhileCond,
}

/// A mutable DSL variable, by creation order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct VarId(pub(crate) u32);

struct VarInfo {
    /// The flattened types, fixed when the Var is created.
    types: Vec<DType>,
    /// The region it was created in: its depth in the region stack and its
    /// serial number. It is local to that region.
    depth: usize,
    serial: u64,
    loc: Loc,
}

/// A Var carried through a loop: its values at loop entry and the loop
/// body's block arguments standing for it.
#[derive(Clone, Debug)]
pub(crate) struct CarriedVar {
    pub var: VarId,
    pub init: Vec<ValueId>,
    pub args: Vec<ValueId>,
}

/// A register of a fragment: the fragment's storage (by address) and the
/// slot in it.
pub(crate) type FragmentSlot = (usize, usize);

/// A region taken off the stack: its instructions and what happened to the
/// Vars created outside it.
pub(crate) struct ClosedRegion {
    pub insts: Vec<Inst>,
    /// The Vars set in the region, with their values at its end.
    pub written: Vec<(VarId, Vec<ValueId>)>,
    /// Loop regions: the Vars carried through, in the order they were first
    /// touched, with their values at the end of the region.
    pub carried: Vec<(CarriedVar, Vec<ValueId>)>,
    /// Loop regions (and while conditions): the registers of fragments from
    /// outside the region that it read before writing them, and where.
    pub fragment_reads: HashMap<FragmentSlot, Loc>,
}

impl ClosedRegion {
    /// The value of `var` at the end of the region if the region set it.
    pub fn written_value(&self, var: VarId) -> Option<&Vec<ValueId>> {
        self.written.iter().find(|(v, _)| *v == var).map(|(_, values)| values)
    }
}

#[derive(Default)]
struct RegionBuilder {
    insts: Vec<Inst>,
    cse: HashMap<PureKey, ValueId>,
    kind: RegionKind,
    serial: u64,
    /// The first value id allocated after the region was opened: values
    /// with ids from it on were defined in the region (or a region inside
    /// it).
    first_value: u32,
    /// The current value of each Var defined, carried or set in this region.
    var_values: HashMap<VarId, Vec<ValueId>>,
    /// The Vars created outside this region and set in it, in order.
    written: Vec<VarId>,
    /// Loop regions: the Vars carried through the loop, in order.
    carried: Vec<CarriedVar>,
    /// Loop regions: the registers of fragments created before the region
    /// was opened that it read before writing them (with the first read),
    /// and the ones it wrote.
    fragment_reads: HashMap<FragmentSlot, Loc>,
    fragment_writes: HashSet<FragmentSlot>,
    /// The depth of the profiling scope stack when the region opened: a scope opened
    /// in the region closes in it.
    scope_depth: usize,
}

impl RegionKind {
    /// Whether the region runs once per iteration of a runtime loop.
    fn is_loop(self) -> bool {
        matches!(self, RegionKind::Loop | RegionKind::WhileCond)
    }
}

pub(crate) struct BuilderState {
    pub(crate) kernel: u32,
    name: String,
    value_types: Vec<DType>,
    facts: HashMap<ValueId, ValueFacts>,
    /// Values that are not constants but are always 0 (`x * 0`): every number divides them.
    known_zero: HashSet<ValueId>,
    /// Open [`ExactScope`]s: while any is, Float `+`, `-` and `*` are recorded in
    /// `exact_float` and never contracted.
    exact_depth: u32,
    exact_float: std::collections::BTreeSet<ValueId>,
    constants: HashMap<ValueId, Literal>,
    params: Vec<Param>,
    param_names: HashSet<String>,
    buffers: Vec<BufferDecl>,
    regions: Vec<RegionBuilder>,
    next_serial: u64,
    vars: Vec<VarInfo>,
    /// The block arguments that loops carry Vars in, and their Vars.
    var_args: HashMap<ValueId, VarId>,
    pub(crate) block_dim: Option<[u32; 3]>,
    /// The target set with [`KernelOptions::target`].
    target: Option<crate::Arch>,
    /// While set, instructions record this location instead of their own:
    /// the user's call of a library function expanded into many operations
    /// (`cute::math`).
    location: Option<Loc>,
    /// The `Kernel::build` call.
    loc: Loc,
    /// The byte address each raw buffer ([`MemSpace::Raw`], a
    /// [`make_ptr`](crate::make_ptr)) starts at: an `i64` value.
    raw_bases: HashMap<BufferId, ValueId>,
    /// The parameter each [`SymInt`](crate::SymInt) was declared as (by
    /// identity), so that every occurrence of one shares it.
    symints: HashMap<u64, ValueId>,
    /// The messages and locations of `RuntimeAssertion` stores.
    pub(crate) assertion_records: Vec<crate::testing::AssertionRecord>,
    max_threads: Option<[u32; 3]>,
    min_blocks_per_sm: Option<u32>,
    globals: Vec<crate::ir::ModuleGlobal>,
    pub(crate) overlays: Vec<Vec<Vec<BufferId>>>,
    /// Hidden parameters (tensor maps of [`tma_descriptor`](crate::tma_descriptor)),
    /// appended after the kernel's own parameters when the build ends.
    hidden_params: Vec<Param>,
    /// Tensor-map parameters, with `param` counting hidden ones from 0 until the build
    /// ends.
    tensor_maps: Vec<(BufferId, Option<crate::tma::TensorMapSource>)>,
    /// The shared-memory partition queries
    /// ([`memory::calculate_partition_size`](crate::memory::calculate_partition_size)):
    /// each result, a constant whose value is set by [`BuilderState::finish`]
    /// once every buffer is known, with its partition and cumulative flag.
    partition_queries: Vec<(ValueId, crate::ir::SmemPartition, bool)>,
    // m11-extern-distributed.
    pub(crate) externs: Vec<crate::ir::ExternDecl>,
    pub(crate) link_libraries: Vec<std::path::PathBuf>,
    pub(crate) multicast: Vec<BufferId>,
    /// Building a standalone device function ([`crate::ffi::DeviceFunction`]), and its
    /// result once the body has returned it.
    pub(crate) device_function: Option<crate::ir::DeviceFunctionSig>,
    /// Profiling scopes, the symbol registry and IKET tracing (m11-profiling-iket).
    pub(crate) profiling: crate::profiling::ProfilingState,
    /// Code to emit once the body has returned (roadmap part
    /// m11-cute-ext-core: the pooled TMEM allocation of
    /// [`crate::experimental`]), run in order by [`run_finish_hooks`].
    pub(crate) finish_hooks: Vec<Box<dyn FnOnce()>>,
    /// The pooled TMEM allocation of [`crate::experimental::allocate`].
    pub(crate) cute_ext_tmem: Option<crate::experimental::memory::TmemPool>,
    /// The TMA atoms of [`crate::experimental`]'s TMA operations, one per
    /// distinct descriptor (roadmap part m11-cute-ext-tma), by key.
    pub(crate) cute_ext_tma: HashMap<String, std::rc::Rc<dyn std::any::Any>>,
    /// The kernel's device workspace ([`crate::experimental::host_runtime`]).
    pub(crate) workspace: Option<crate::experimental::host_runtime::WorkspaceBuild>,
}

/// Run the hooks registered with [`BuilderState::finish_hooks`], including
/// any that they register in turn.
pub(crate) fn run_finish_hooks() {
    loop {
        let hooks = with(|b| std::mem::take(&mut b.finish_hooks));
        if hooks.is_empty() {
            return;
        }
        for hook in hooks {
            hook();
        }
    }
}

/// Whether the builder is inside runtime control flow (an `if_`, a loop):
/// any region but the kernel body's.
pub(crate) fn in_runtime_region() -> bool {
    with(|b| b.regions.len() > 1)
}

/// Emit what `f` builds at the start of the kernel body, before everything
/// built so far. Only at the body's top level, from a finish hook; `f` sees
/// none of the body's values (its common-subexpression table starts
/// empty).
#[track_caller]
pub(crate) fn emit_at_body_start(f: impl FnOnce()) {
    let (insts, cse) = with(|b| {
        assert_eq!(b.regions.len(), 1, "cute: emit_at_body_start inside a region");
        let r = &mut b.regions[0];
        (std::mem::take(&mut r.insts), std::mem::take(&mut r.cse))
    });
    f();
    with(|b| {
        assert_eq!(b.regions.len(), 1, "cute: emit_at_body_start left a region open");
        let r = &mut b.regions[0];
        r.insts.extend(insts);
        r.cse = cse;
    });
}

thread_local! {
    static BUILDER: RefCell<Option<BuilderState>> = const { RefCell::new(None) };
}

static NEXT_KERNEL: AtomicU32 = AtomicU32::new(1);

/// The name of the dynamic shared-memory region's buffer, a byte array.
pub(crate) const DYNAMIC_SMEM_NAME: &str = "dynamic_smem";

/// Run `f` on the active builder. Panics outside [`Kernel::build`].
#[track_caller]
pub(crate) fn with<R>(f: impl FnOnce(&mut BuilderState) -> R) -> R {
    BUILDER.with(|b| {
        let mut b = b.borrow_mut();
        let state = b.as_mut().unwrap_or_else(|| panic!("cute: DSL operation used outside of Kernel::build"));
        f(state)
    })
}

/// The architecture the kernel being built targets (CuTe DSL's
/// `get_arch_enum()` while tracing): the one [`KernelOptions::target`] set,
/// else the one `CUTE_RS_ARCH` names (the default, as `CUTE_DSL_ARCH` is
/// CuTe DSL's), else `None`. Kernels read it to specialise for the target;
/// only the first is checked when the build ends. Panics outside
/// [`Kernel::build`], and if `CUTE_RS_ARCH` names no target.
#[track_caller]
pub fn target_arch() -> Option<crate::Arch> {
    if let Some(arch) = with(|b| b.target) {
        return Some(arch);
    }
    let text = std::env::var(crate::backend::ENV_ARCH).ok()?;
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    Some(text.parse().unwrap_or_else(|e| panic!("cute: {}={text}: {e}", crate::backend::ENV_ARCH)))
}

/// [`target_arch`] without panicking: the declared target while a kernel is
/// being built, else the one `CUTE_RS_ARCH` names, else `None` (an
/// unparsable `CUTE_RS_ARCH` included).
pub(crate) fn declared_target() -> Option<crate::Arch> {
    let building = BUILDER.with(|b| b.try_borrow().ok().and_then(|b| b.as_ref().map(|s| s.target)));
    if let Some(Some(arch)) = building {
        return Some(arch);
    }
    std::env::var(crate::backend::ENV_ARCH).ok()?.trim().parse().ok()
}

/// The largest number known to divide `v`, or `None` if `v` is always 0, which every number
/// divides. 1 if no kernel is being built (or the builder is already borrowed).
pub(crate) fn value_divisor(v: ValueId) -> Option<usize> {
    BUILDER.with(|b| {
        let Ok(b) = b.try_borrow() else { return Some(1) };
        b.as_ref().map_or(Some(1), |s| s.known_divisor(v))
    })
}

/// Facts about `v` if a kernel is being built (and the builder is not
/// already borrowed); defaults otherwise.
pub(crate) fn value_facts(v: ValueId) -> ValueFacts {
    BUILDER.with(|b| {
        b.try_borrow().ok().and_then(|b| b.as_ref().and_then(|s| s.facts.get(&v).copied())).unwrap_or_default()
    })
}

/// The type of `v` if a kernel is being built (and the builder is not
/// already borrowed) and `v` belongs to it; `None` otherwise (a parsed
/// layout's placeholder, say).
pub(crate) fn value_dtype(v: ValueId) -> Option<DType> {
    BUILDER
        .with(|b| b.try_borrow().ok().and_then(|b| b.as_ref().and_then(|s| s.value_types.get(v.0 as usize).copied())))
}

/// Attributes every instruction built while it lives to its creator's
/// caller (see [`BuilderState::set_location`]).
/// Panic with `msg`, starting with the user's source location `loc`
/// (`file:line:col: msg`), for misuse found inside the DSL's internals,
/// where the panic's own location would name cute-rs.
pub(crate) fn user_panic(loc: Loc, msg: impl std::fmt::Display) -> ! {
    panic!("{}:{}:{}: {msg}", loc.file(), loc.line(), loc.column())
}

pub(crate) struct LocationScope(bool);

impl LocationScope {
    #[track_caller]
    pub(crate) fn new() -> LocationScope {
        let loc = std::panic::Location::caller();
        LocationScope(with(|b| b.set_location(loc)))
    }
}

/// While alive, the Float `+`, `-` and `*` the builder emits are exact: each rounds on its
/// own and is never fused into an `fma`, whatever
/// [`CompileOptions::fp_contract`](crate::CompileOptions::fp_contract) says. The precise
/// [`math`](crate::math) functions open one, since their error bounds, their error-free
/// transformations (`two_sum`, ...) and their bit-for-bit agreement with the simulator
/// depend on each operation's own rounding. Scopes nest; outside a kernel build it does
/// nothing.
pub(crate) struct ExactScope(bool);

impl ExactScope {
    /// An exact scope if `on`, otherwise nothing.
    pub(crate) fn new(on: bool) -> ExactScope {
        let opened = on
            && BUILDER
                .with(|b| b.try_borrow_mut().ok().and_then(|mut b| b.as_mut().map(|s| s.exact_depth += 1)).is_some());
        ExactScope(opened)
    }
}

impl Drop for ExactScope {
    fn drop(&mut self) {
        if self.0 {
            BUILDER.with(|b| {
                if let Ok(mut b) = b.try_borrow_mut()
                    && let Some(state) = b.as_mut()
                {
                    state.exact_depth -= 1;
                }
            });
        }
    }
}

impl Drop for LocationScope {
    fn drop(&mut self) {
        if self.0 {
            BUILDER.with(|b| {
                if let Ok(mut b) = b.try_borrow_mut()
                    && let Some(state) = b.as_mut()
                {
                    state.clear_location();
                }
            });
        }
    }
}

/// The constant `v` was created from, if a kernel is being built.
pub(crate) fn try_constant(v: ValueId) -> Option<Literal> {
    BUILDER.with(|b| b.try_borrow().ok().and_then(|b| b.as_ref().and_then(|s| s.constants.get(&v).copied())))
}

/// A note for an error about `e` that knowing more about a value (a
/// constant, a divisor or a bound) might have avoided: it names the Var when
/// `e` depends on a Var's value inside a runtime loop, about which nothing
/// is known. Empty otherwise.
pub(crate) fn var_value_note(e: &IndexExpr) -> String {
    let mut values = Vec::new();
    e.dependencies(&mut values);
    let found = BUILDER.with(|b| {
        let b = b.try_borrow().ok()?;
        let state = b.as_ref()?;
        values.iter().find_map(|&v| state.var_args.get(&v).map(|var| (v, state.vars[var.0 as usize].loc)))
    });
    let Some((v, created)) = found else { return String::new() };
    format!(
        "; {v} is the value of the Var created at {created}, which a runtime loop carries (a loop carries every \
         Var it touches, even one it only reads), so what is known about the Var's value outside the loop does \
         not hold inside it: read the Var before the loop and use that value"
    )
}

/// The serial number of the region opened last: every region opened later
/// has a larger one. `u64::MAX` when no kernel is being built.
pub(crate) fn latest_serial() -> u64 {
    BUILDER.with(|b| b.try_borrow().ok().and_then(|b| b.as_ref().map(|s| s.next_serial)).unwrap_or(u64::MAX))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A number dividing every product of a multiple of `x` and a multiple of
/// `y`: `x * y`, or, when that does not fit, the largest power of two that
/// divides it (capped at `2^62`), which still divides every such product.
pub(crate) fn mul_divisors(x: usize, y: usize) -> usize {
    x.checked_mul(y).unwrap_or_else(|| 1 << (x.trailing_zeros() + y.trailing_zeros()).min(62))
}

impl BuilderState {
    pub(crate) fn new_value(&mut self, ty: DType) -> ValueId {
        let id = ValueId(self.value_types.len() as u32);
        self.value_types.push(ty);
        id
    }

    pub(crate) fn dtype(&self, v: ValueId) -> DType {
        self.value_types[v.0 as usize]
    }

    pub(crate) fn constant_of(&self, v: ValueId) -> Option<Literal> {
        self.constants.get(&v).copied()
    }

    pub(crate) fn set_facts(&mut self, v: ValueId, facts: ValueFacts) {
        self.facts.insert(v, facts);
    }

    pub(crate) fn facts_of(&self, v: ValueId) -> ValueFacts {
        self.facts.get(&v).copied().unwrap_or_default()
    }

    /// Give `result`, which takes the value of one of `sources` (the yields
    /// of an `if`'s branches, or a loop's initial value and what its body
    /// yields), the facts that hold for all of them: the largest bound and
    /// the greatest common divisor (a constant 0 divides by anything), and
    /// being a power of two if all of them are.
    pub(crate) fn merge_facts(&mut self, result: ValueId, sources: &[ValueId]) {
        if sources.is_empty() {
            return;
        }
        let mut bound = Some(0);
        let mut divisor = 0;
        let mut pow2 = true;
        for &s in sources {
            let facts = self.facts_of(s);
            bound = bound.zip(facts.bound).map(|(a, b)| a.max(b));
            if !self.is_zero_constant(s) {
                divisor = gcd(divisor, facts.divisor);
            }
            pow2 &= facts.pow2;
        }
        let facts = ValueFacts { bound, divisor: divisor.max(1), pow2 };
        if facts.is_informative() {
            self.facts.insert(result, facts);
        }
    }

    /// Whether `v` is an integer constant 0, or an integer the facts prove is always 0.
    fn is_zero_constant(&self, v: ValueId) -> bool {
        self.known_zero.contains(&v) || self.constant_of(v).is_some_and(|c| c.dtype().is_int() && c.bits() == 0)
    }

    /// The largest number known to divide `v`, or `None` if `v` is the
    /// constant 0, which every number divides.
    fn known_divisor(&self, v: ValueId) -> Option<usize> {
        (!self.is_zero_constant(v)).then(|| self.facts_of(v).divisor)
    }

    /// The value of the integer constant `v` (sign-extended for a signed
    /// type), if it is one.
    fn int_constant(&self, v: ValueId) -> Option<i128> {
        let c = self.constant_of(v)?;
        let ty = c.dtype();
        if !ty.is_int() {
            return None;
        }
        let shift = 128 - ty.bits() as u32;
        let bits = c.bits128() as i128;
        Some(if ty.is_signed() { (bits << shift) >> shift } else { bits })
    }

    /// What is known about the result of integer arithmetic `op` from what
    /// is known about its operands, CuTe DSL's divisibility rules for its
    /// integers (assuming the arithmetic does not overflow): a sum or a
    /// difference is a multiple of the operands' greatest common divisor, a
    /// product of the product of their divisors ([`mul_divisors`]), `x << k` of
    /// `x`'s times `2^k`, `x & y` of the larger power of two that divides
    /// one of them (a constant's lowest set bit, for a mask), a remainder by a constant of its gcd with `x`'s, and a
    /// floor division by a constant `c` of `x`'s divisor over `c` when `c`
    /// divides it (and of 1 otherwise). A minimum or maximum, which is one
    /// of the operands, keeps their gcd. A product of powers of two, or one
    /// shifted left, is a power of two.
    /// The facts of `lhs op rhs`, and whether the result is always 0.
    fn binary_facts(&self, op: BinaryOp, lhs: ValueId, rhs: ValueId) -> (ValueFacts, bool) {
        if !self.dtype(lhs).is_int() {
            return (ValueFacts::default(), false);
        }
        let bits = self.dtype(lhs).bits() as i128;
        let (a, b) = (self.known_divisor(lhs), self.known_divisor(rhs));
        let (fa, fb) = (self.facts_of(lhs), self.facts_of(rhs));
        // A non-zero constant divisor or shift amount.
        let c = self.int_constant(rhs).filter(|&c| c != 0).map(i128::unsigned_abs);
        let gcd_of = |a: Option<usize>, b: Option<usize>| match (a, b) {
            (Some(x), Some(y)) => Some(gcd(x, y)),
            (x, None) | (None, x) => x,
        };
        // Trailing zero bits of a divisor (all of them for 0).
        let tz = |d: Option<usize>| d.map_or(usize::BITS - 1, usize::trailing_zeros);
        let pow2_part = |k: u32| 1usize << k.min(usize::BITS - 2);
        let divisor = match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Min | BinaryOp::Max => gcd_of(a, b),
            BinaryOp::Mul => a.zip(b).map(|(x, y)| mul_divisors(x, y)),
            BinaryOp::Shl => match self.int_constant(rhs) {
                Some(k) if (0..bits).contains(&k) => a.map(|x| mul_divisors(x, 1usize << k.min(62) as u32)),
                _ => Some(1),
            },
            // A constant mask counts by its own trailing zero bits (`x & !3`
            // is a multiple of 4).
            BinaryOp::And => {
                let tz_of = |v: ValueId, d: Option<usize>| match self.int_constant(v) {
                    Some(c) if c != 0 => (c as u128).trailing_zeros().min(usize::BITS - 1),
                    _ => tz(d),
                };
                Some(pow2_part(tz_of(lhs, a).max(tz_of(rhs, b))))
            }
            BinaryOp::Rem | BinaryOp::RemU => match c {
                Some(c) => a.map(|x| gcd(x, usize::try_from(c).unwrap_or(1))),
                None => Some(1),
            },
            BinaryOp::Div | BinaryOp::DivU => match (a, c.and_then(|c| usize::try_from(c).ok())) {
                (None, Some(_)) => None,
                (Some(x), Some(c)) if x.is_multiple_of(c) => Some(x / c),
                _ => Some(1),
            },
            _ => Some(1),
        };
        let pow2 = match op {
            BinaryOp::Mul => fa.pow2 && fb.pow2,
            BinaryOp::Shl => fa.pow2 && matches!(self.int_constant(rhs), Some(k) if (0..bits).contains(&k)),
            _ => false,
        };
        // A result that is always 0 (no divisor) has no facts of its own: it is recorded as zero.
        (ValueFacts { bound: None, divisor: divisor.unwrap_or(1).max(1), pow2 }, divisor.is_none())
    }

    pub(crate) fn push(&mut self, mut op: Op, loc: Loc) {
        let loc = self.location.unwrap_or(loc);
        if !self.raw_bases.is_empty() {
            self.rebase_raw(&mut op, loc);
        }
        let scope = self.profiling.current_scope();
        self.regions.last_mut().expect("region stack is never empty while building").insts.push(Inst {
            op,
            loc,
            scope,
        });
    }

    /// Declare a raw buffer starting at the `i64` byte address `base`.
    pub(crate) fn declare_raw_buffer(&mut self, decl: BufferDecl, base: ValueId) -> BufferId {
        debug_assert!(decl.space.raw().is_some() && self.dtype(base) == DType::I64);
        let id = BufferId(self.buffers.len() as u32);
        self.buffers.push(decl);
        self.raw_bases.insert(id, base);
        id
    }

    /// The byte address a raw buffer starts at.
    pub(crate) fn raw_base(&self, b: BufferId) -> Option<ValueId> {
        self.raw_bases.get(&b).copied()
    }

    /// A load, store, atomic or reduction of a raw buffer addresses it by
    /// byte address ([`MemSpace::Raw`]): its element index becomes `base +
    /// index * size`. The other operations that take a buffer take no raw
    /// pointer.
    fn rebase_raw(&mut self, op: &mut Op, loc: Loc) {
        let refuse = |s: &Self, b: BufferId, what: &str| {
            if s.raw_bases.contains_key(&b) {
                panic!(
                    "cute: {what} takes a pointer into a buffer, not a raw pointer (make_ptr, inttoptr, tospace, \
                     Ptr::align or a masked pointer) [{}:{}]: raw pointers are for loads, stores, atomics and the copies \
                     built on them",
                    loc.file(),
                    loc.line()
                )
            }
        };
        let (buffer, index, elem) = match op {
            Op::Load { results, buffer, index, .. } => (*buffer, index, self.dtype(results[0])),
            Op::Store { values, buffer, index, .. } => (*buffer, index, self.dtype(values[0])),
            Op::Atomic { ty, buffer, index, value, .. } | Op::Red { ty, buffer, index, value, .. } => {
                let t = ty.access_elem(self.dtype(*value));
                (*buffer, index, t)
            }
            Op::CpAsync { src, dst, .. } => {
                refuse(self, *src, "cp.async");
                refuse(self, *dst, "cp.async");
                return;
            }
            Op::CpAsyncBulk { src, dst, .. } => {
                refuse(self, *src, "a bulk copy (cp.async.bulk)");
                refuse(self, *dst, "a bulk copy (cp.async.bulk)");
                return;
            }
            Op::TmaLoad { dst, mbar, .. } => {
                refuse(self, *dst, "a TMA load (cp.async.bulk.tensor)");
                refuse(self, *mbar, "a TMA load's mbarrier");
                return;
            }
            Op::TmaStore { src, .. } => return refuse(self, *src, "a TMA store (cp.async.bulk.tensor)"),
            Op::BulkPrefetch { buffer, .. } => return refuse(self, *buffer, "a bulk prefetch"),
            Op::ClcTryCancel { buffer, mbar, .. } => {
                refuse(self, *buffer, "clusterlaunchcontrol.try_cancel");
                refuse(self, *mbar, "clusterlaunchcontrol.try_cancel's mbarrier");
                return;
            }
            Op::ClcQueryCancel { buffer, .. } => return refuse(self, *buffer, "clc_response"),
            Op::StAsync { buffer, mbar, .. } => {
                refuse(self, *buffer, "st.async");
                refuse(self, *mbar, "st.async's mbarrier");
                return;
            }
            Op::LdMatrix { buffer, .. } => return refuse(self, *buffer, "ldmatrix"),
            Op::StMatrix { buffer, .. } => return refuse(self, *buffer, "stmatrix"),
            Op::MbarrierInit { buffer, .. }
            | Op::MbarrierArrive { buffer, .. }
            | Op::MbarrierExpectTx { buffer, .. }
            | Op::MbarrierWait { buffer, .. }
            | Op::CpAsyncMbarrierArrive { buffer, .. }
            | Op::Tcgen05Commit { buffer, .. }
            | Op::MbarrierExt { buffer, .. } => return refuse(self, *buffer, "an mbarrier operation"),
            Op::StBulk { buffer, .. } => return refuse(self, *buffer, "st.bulk"),
            Op::Prefetch { buffer, .. } => return refuse(self, *buffer, "prefetch"),
            Op::Tcgen05 { buffer: Some(buffer), .. } => return refuse(self, *buffer, "tcgen05.alloc's address slot"),
            Op::InlinePtx { inputs: ops, .. } | Op::MlirOp { operands: ops, .. } => {
                for i in ops.iter() {
                    if let crate::ir::AsmInput::Address { buffer, .. } = i {
                        refuse(self, *buffer, "an address operand of inline assembly (pass the pointer as a value)");
                    }
                }
                return;
            }
            _ => return,
        };
        let Some(&base) = self.raw_bases.get(&buffer) else { return };
        if self.buffers[buffer.0 as usize].space == MemSpace::Raw(crate::tensor::AddressSpace::Tmem) {
            panic!(
                "cute: tensor memory is not read or written by loads, stores or atomics: TMEM tensors are accessed \
                 only by tcgen05 copies and MMAs (sm_100) [{}:{}]",
                loc.file(),
                loc.line()
            );
        }
        let wide = if self.dtype(*index) == DType::I64 { *index } else { self.cast(*index, DType::I64, loc) };
        let size = self.constant(Literal::I64(elem.bytes() as i64), loc);
        let off = self.binary(BinaryOp::Mul, wide, size, loc);
        *index = self.binary(BinaryOp::Add, base, off, loc);
    }

    /// Attribute the instructions built from now on to `loc`, unless an
    /// outer call already does. Returns whether this call set it.
    pub(crate) fn set_location(&mut self, loc: Loc) -> bool {
        if self.location.is_some() {
            return false;
        }
        self.location = Some(loc);
        true
    }

    pub(crate) fn clear_location(&mut self) {
        self.location = None;
    }

    fn pure(&mut self, key: PureKey, ty: DType, loc: Loc, make: impl FnOnce(ValueId) -> Op) -> ValueId {
        if let Some(v) = self.regions.iter().rev().find_map(|r| r.cse.get(&key)) {
            return *v;
        }
        let result = self.new_value(ty);
        self.push(make(result), loc);
        self.regions.last_mut().unwrap().cse.insert(key, result);
        result
    }

    pub(crate) fn constant(&mut self, value: Literal, loc: Loc) -> ValueId {
        let v = self.pure(PureKey::Const(value), value.dtype(), loc, |result| Op::Const { result, value });
        self.constants.insert(v, value);
        // A non-negative integer constant is its own bound and divisor.
        if let Some(c) = self.int_constant(v).and_then(|c| usize::try_from(c).ok()) {
            let facts = ValueFacts { bound: c.checked_add(1), divisor: c.max(1), pow2: c.is_power_of_two() };
            self.facts.insert(v, facts);
        }
        v
    }

    pub(crate) fn unary(&mut self, op: UnaryOp, x: ValueId, loc: Loc) -> ValueId {
        let ty = self.dtype(x);
        self.pure(PureKey::Unary(op, x), ty, loc, |result| Op::Unary { result, op, x })
    }

    pub(crate) fn binary(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId, loc: Loc) -> ValueId {
        let (lt, rt) = (self.dtype(lhs), self.dtype(rhs));
        if lt != rt {
            user_panic(loc, format!("cute: {op:?} on mismatched types {lt} and {rt}"));
        }
        let commutative = matches!(
            op,
            BinaryOp::Add
                | BinaryOp::Mul
                | BinaryOp::Min
                | BinaryOp::Max
                | BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::Xor
        );
        let key = if commutative && rhs < lhs { PureKey::Binary(op, rhs, lhs) } else { PureKey::Binary(op, lhs, rhs) };
        let (facts, zero) = self.binary_facts(op, lhs, rhs);
        let result = self.pure(key, lt, loc, |result| Op::Binary { result, op, lhs, rhs });
        if zero {
            self.known_zero.insert(result);
        }
        // A value shared (by deduplication) with an exact scope stays exact.
        if self.exact_depth > 0 && lt.is_float() && matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul) {
            self.exact_float.insert(result);
        }
        if facts.is_informative() && !self.facts.contains_key(&result) {
            self.facts.insert(result, facts);
        }
        result
    }

    pub(crate) fn fma(&mut self, a: ValueId, b: ValueId, c: ValueId, loc: Loc) -> ValueId {
        let ty = self.dtype(a);
        self.pure(PureKey::Fma(a, b, c), ty, loc, |result| Op::Fma { result, a, b, c })
    }

    /// A floating-point instruction with explicit modifiers (validated with
    /// the kernel).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn float_inst(
        &mut self,
        func: FloatFn,
        args: Vec<ValueId>,
        mode: FloatMode,
        ftz: bool,
        nan: bool,
        xorsign_abs: bool,
        loc: Loc,
    ) -> ValueId {
        let ty = self.dtype(args[0]);
        let key = PureKey::Float(func, mode, ftz, nan, xorsign_abs, args.clone());
        self.pure(key, ty, loc, |result| Op::FloatInst { result, func, args, mode, ftz, nan, xorsign_abs })
    }

    /// A PTX integer or bit instruction with a result of type `ty`
    /// (validated with the kernel).
    pub(crate) fn int_inst(&mut self, func: IntFn, args: Vec<ValueId>, ty: DType, loc: Loc) -> ValueId {
        let mut key_args = args.clone();
        if func.is_commutative() && key_args[1] < key_args[0] {
            key_args.swap(0, 1);
        }
        self.pure(PureKey::Int(func, ty, key_args), ty, loc, |result| Op::IntInst { result, func, args })
    }

    pub(crate) fn cmp(&mut self, pred: CmpPred, lhs: ValueId, rhs: ValueId, loc: Loc) -> ValueId {
        let (lt, rt) = (self.dtype(lhs), self.dtype(rhs));
        if lt != rt {
            user_panic(loc, format!("cute: comparison of mismatched types {lt} and {rt}"));
        }
        self.pure(PureKey::Cmp(pred, lhs, rhs), DType::Bool, loc, |result| Op::Cmp { result, pred, lhs, rhs })
    }

    pub(crate) fn select(&mut self, cond: ValueId, on_true: ValueId, on_false: ValueId, loc: Loc) -> ValueId {
        let ty = self.dtype(on_true);
        self.pure(PureKey::Select(cond, on_true, on_false), ty, loc, |result| Op::Select {
            result,
            cond,
            on_true,
            on_false,
        })
    }

    pub(crate) fn cast(&mut self, x: ValueId, to: DType, loc: Loc) -> ValueId {
        let from = self.dtype(x);
        if from == to {
            return x;
        }
        let result = self.pure(PureKey::Cast(to, x), to, loc, |result| Op::Cast { result, x });
        // A widening that keeps every value (to a wider type of the same
        // signedness, or from unsigned to a wider signed type) keeps what
        // is known about it.
        let keeps_value = from.is_int()
            && to.is_int()
            && to.bits() > from.bits()
            && (from.is_signed() == to.is_signed() || !from.is_signed());
        let facts = self.facts_of(x);
        if keeps_value && facts.is_informative() && !self.facts.contains_key(&result) {
            self.facts.insert(result, facts);
        }
        result
    }

    pub(crate) fn bitcast(&mut self, x: ValueId, to: DType, loc: Loc) -> ValueId {
        assert_eq!(self.dtype(x).bits(), to.bits(), "cute: bitcast between {} and {to}", self.dtype(x));
        if self.dtype(x) == to {
            return x;
        }
        self.pure(PureKey::Bitcast(to, x), to, loc, |result| Op::Bitcast { result, x })
    }

    /// The bits of `parts` concatenated into one value of type `ty`, part 0
    /// in the low bits (validated with the kernel).
    pub(crate) fn pack(&mut self, parts: Vec<ValueId>, ty: DType, loc: Loc) -> ValueId {
        let key = PureKey::Pack(ty, parts.clone());
        self.pure(key, ty, loc, |result| Op::Pack { result, parts })
    }

    /// The bits of `x` split into `n` values of type `part`, the first from
    /// the low bits (validated with the kernel).
    pub(crate) fn unpack(&mut self, x: ValueId, part: DType, n: usize, loc: Loc) -> Vec<ValueId> {
        let key = PureKey::Unpack(part, n, x);
        if let Some(&first) = self.regions.iter().rev().find_map(|r| r.cse.get(&key)) {
            return (0..n as u32).map(|k| ValueId(first.0 + k)).collect();
        }
        let results: Vec<ValueId> = (0..n).map(|_| self.new_value(part)).collect();
        self.push(Op::Unpack { results: results.clone(), x }, loc);
        self.regions.last_mut().unwrap().cse.insert(key, results[0]);
        results
    }

    /// A half-precision instruction (validated with the kernel).
    pub(crate) fn half_inst(
        &mut self,
        func: HalfFn,
        ty: HalfType,
        args: Vec<ValueId>,
        mods: HalfModifiers,
        predicate: Option<ValueId>,
        loc: Loc,
    ) -> ValueId {
        let key = PureKey::Half(func, ty, mods, args.clone(), predicate);
        self.pure(key, ty.value_type(), loc, |result| Op::HalfInst { result, func, ty, args, mods, predicate })
    }

    /// A conversion with modifiers (validated with the kernel).
    pub(crate) fn cvt(&mut self, func: CvtFn, args: Vec<ValueId>, loc: Loc) -> ValueId {
        let key = PureKey::Cvt(func, args.clone());
        self.pure(key, func.result_type(), loc, |result| Op::Cvt { result, func, args })
    }

    pub(crate) fn special(&mut self, reg: SpecialReg, loc: Loc) -> ValueId {
        if !reg.is_constant() {
            let result = self.new_value(reg.dtype());
            self.push(Op::Special { result, reg }, loc);
            return result;
        }
        let v = self.pure(PureKey::Special(reg), reg.dtype(), loc, |result| Op::Special { result, reg });
        let bound = match (reg, self.block_dim) {
            (SpecialReg::LaneId, _) => Some(32),
            (SpecialReg::ThreadIdx(d), Some(dims)) => Some(dims[d.index()] as usize),
            _ => None,
        };
        self.facts.insert(v, ValueFacts { bound, divisor: 1, pow2: false });
        v
    }

    /// The width an index expression is computed in, from the types of its
    /// runtime values ([`IndexExpr::width_with`]).
    pub(crate) fn index_width(&self, e: &IndexExpr) -> u32 {
        e.width_with(&|v| self.value_types.get(v.0 as usize).copied())
    }

    /// Turn the index expression `e` into a value of `width` bits, at least
    /// its own width ([`IndexExpr::width`]: 64 when an `i64` value or a
    /// static value beyond `i32` enters it, 32 otherwise). Each
    /// part of it is computed in its own width and widened where it meets a
    /// 64-bit part, so that `i32` index arithmetic (a thread's position in a
    /// tile, say) stays 32-bit; index values are non-negative, so widening
    /// one keeps its value.
    pub(crate) fn materialize_at(&mut self, e: &IndexExpr, width: u32, loc: Loc) -> ValueId {
        let own = self.index_width(e);
        assert!(own <= width, "cute: a {own}-bit index expression cannot be computed in {width} bits: {e:?}");
        if own < width {
            if let Some(v) = e.as_static_signed() {
                return self.constant(Literal::I64(v), loc);
            }
            let narrow = self.materialize_at(e, own, loc);
            return self.cast(narrow, DType::I64, loc);
        }
        let bin = |s: &mut Self, op, a: &IndexExpr, b: &IndexExpr| {
            let (a, b) = (s.materialize_at(a, width, loc), s.materialize_at(b, width, loc));
            s.binary(op, a, b, loc)
        };
        match e {
            IndexExpr::Static(v) if width == 64 => {
                let v = i64::try_from(*v).unwrap_or_else(|_| panic!("cute: index {v} does not fit in i64"));
                self.constant(Literal::I64(v), loc)
            }
            IndexExpr::Static(v) => {
                let v = i32::try_from(*v).unwrap_or_else(|_| panic!("cute: index {v} does not fit in i32"));
                self.constant(Literal::I32(v), loc)
            }
            // A negative constant, and otherwise `0 - a` (wrapping, so the
            // signed value).
            IndexExpr::Neg(a) => match e.as_static_signed() {
                Some(v) if width == 64 => self.constant(Literal::I64(v), loc),
                Some(v) => {
                    let v = i32::try_from(v).unwrap_or_else(|_| panic!("cute: index {v} does not fit in i32"));
                    self.constant(Literal::I32(v), loc)
                }
                None => bin(self, BinaryOp::Sub, &IndexExpr::Static(0), a),
            },
            IndexExpr::Dynamic(v) | IndexExpr::Signed(v) => {
                assert!(
                    (v.0 as usize) < self.value_types.len(),
                    "cute: an index expression uses the runtime value {v}, which does not belong to kernel `{}` (a \
                     `?` in a parsed layout string is a placeholder and cannot be evaluated in a kernel)",
                    self.name
                );
                assert!(
                    matches!(self.dtype(*v), DType::I32 | DType::I64),
                    "cute: index expressions must be built from i32 or i64 values, and {v} is {}",
                    self.dtype(*v)
                );
                *v
            }
            IndexExpr::Add(a, b) => bin(self, BinaryOp::Add, a, b),
            IndexExpr::Sub(a, b) => bin(self, BinaryOp::Sub, a, b),
            IndexExpr::Mul(a, b) => bin(self, BinaryOp::Mul, a, b),
            IndexExpr::Div(a, b) => bin(self, BinaryOp::DivU, a, b),
            IndexExpr::Rem(a, b) => bin(self, BinaryOp::RemU, a, b),
            // Index values are non-negative, so the signed minimum is theirs.
            IndexExpr::Min(a, b) => bin(self, BinaryOp::Min, a, b),
            IndexExpr::And(a, b) => bin(self, BinaryOp::And, a, b),
            IndexExpr::Xor(a, b) => bin(self, BinaryOp::Xor, a, b),
            IndexExpr::Shr(a, b) => bin(self, BinaryOp::ShrU, a, b),
        }
    }

    /// `x`, which the program promises is a multiple of `divby` and, with
    /// `pow2`, a power of two (CuTe DSL's `cute.assume`): an
    /// [`Op::Assume`] whose result carries those facts, on top of what is
    /// known about `x` already.
    pub(crate) fn assume(&mut self, x: ValueId, divby: u64, pow2: bool, loc: Loc) -> ValueId {
        let ty = self.dtype(x);
        let known = self.facts_of(x);
        let result =
            self.pure(PureKey::Assume(x, divby, pow2), ty, loc, |result| Op::Assume { result, x, divby, pow2 });
        let divby = usize::try_from(divby).unwrap_or(usize::MAX);
        let divisor = known.divisor / gcd(known.divisor, divby);
        let facts = ValueFacts { bound: known.bound, divisor: mul_divisors(divisor, divby), pow2: known.pow2 || pow2 };
        if facts.is_informative() {
            self.facts.insert(result, facts);
        }
        result
    }

    pub(crate) fn push_region(&mut self, kind: RegionKind) {
        assert_ne!(kind, RegionKind::Root, "cute: only the kernel body is a root region");
        self.next_serial += 1;
        let first_value = self.value_types.len() as u32;
        let scope_depth = self.profiling.stack.len();
        self.regions.push(RegionBuilder {
            kind,
            serial: self.next_serial,
            first_value,
            scope_depth,
            ..RegionBuilder::default()
        });
    }

    pub(crate) fn pop_region(&mut self) -> ClosedRegion {
        assert!(self.regions.len() > 1, "cute: unbalanced region stack");
        let r = self.regions.pop().unwrap();
        assert_eq!(
            self.profiling.stack.len(),
            r.scope_depth,
            "cute: a profiling Scope must be closed in the if_/for_range/while body that opened it (scope depth {} at \
             the start of the body, {} at its end)",
            r.scope_depth,
            self.profiling.stack.len()
        );
        let value = |v: &VarId| r.var_values.get(v).cloned().expect("a carried or set Var has a value");
        let written = r.written.iter().map(|v| (*v, value(v))).collect();
        let carried = r.carried.iter().map(|c| (c.clone(), value(&c.var))).collect();
        ClosedRegion { insts: r.insts, written, carried, fragment_reads: r.fragment_reads }
    }

    // Register fragments are build-time arrays of SSA values, so a runtime
    // loop does not carry them: a register written in a loop body holds the
    // new value for the rest of that body only. That is wrong exactly when
    // the loop read the register before writing it (the next iteration
    // reads the value from before the loop, where CuTe DSL's register
    // memory would hold the write), so that pattern is reported. Registers
    // of fragments created inside the loop are the iteration's own.

    /// Record that `slot` (a register of a fragment created when the latest
    /// serial was `created`) is read at `loc`.
    pub(crate) fn fragment_read(&mut self, slot: FragmentSlot, created: u64, loc: Loc) {
        for r in self.regions.iter_mut().filter(|r| r.kind.is_loop() && r.serial > created) {
            if !r.fragment_writes.contains(&slot) {
                r.fragment_reads.entry(slot).or_insert(loc);
            }
        }
    }

    /// Record that `slot` is written at `loc`, panicking if a loop around
    /// read it first.
    pub(crate) fn fragment_write(&mut self, slot: FragmentSlot, created: u64, loc: Loc) {
        for r in self.regions.iter_mut().filter(|r| r.kind.is_loop() && r.serial > created) {
            if let Some(read) = r.fragment_reads.get(&slot) {
                panic!(
                    "cute: register {} of a fragment created outside a runtime loop is read at {read} and then \
                     written at {loc} in the loop, but the loop's next iteration would not see the write: \
                     register fragments are build-time arrays of SSA values, which a runtime loop does not carry \
                     by itself. Keep the fragment in a Var and change it with Var::update, or carry it with \
                     for_range_carry",
                    slot.1
                );
            }
            r.fragment_writes.insert(slot);
        }
    }

    /// Whether an `if` branch or loop body opened after serial `created`
    /// (a fragment's creation) is still open: the fragment is then state
    /// from outside the current region (roadmap part m8-wgmma-core).
    pub(crate) fn region_opened_since(&self, created: u64) -> bool {
        self.regions.iter().any(|r| r.kind != RegionKind::Root && r.serial > created)
    }

    /// The serials of the open regions, outermost first: a value used where
    /// the path was `p` is visible wherever `p` is a prefix of the path
    /// (roadmap part m8-wgmma-variants).
    pub(crate) fn region_path(&self) -> Vec<u64> {
        self.regions.iter().map(|r| r.serial).collect()
    }

    /// Whether `v` was defined inside the outermost `if` branch or loop body
    /// opened after serial `created` that is still open (true when there is
    /// none): a register of a fragment from outside the region that holds
    /// such a value was written in the region, which may then rebind it
    /// (roadmap part m8-wgmma-core).
    pub(crate) fn defined_since_region_opened(&self, created: u64, v: ValueId) -> bool {
        match self.regions.iter().find(|r| r.kind != RegionKind::Root && r.serial > created) {
            Some(r) => v.0 >= r.first_value,
            None => true,
        }
    }

    /// Start a `while` body knowing which registers its condition read.
    pub(crate) fn seed_fragment_reads(&mut self, reads: &HashMap<FragmentSlot, Loc>) {
        let region = self.regions.last_mut().unwrap();
        debug_assert_eq!(region.kind, RegionKind::Loop);
        region.fragment_reads.extend(reads);
    }

    // Vars: mutable DSL variables, turned into SSA form as the kernel is
    // built, in the style of Braun et al.'s "Simple and Efficient
    // Construction of Static Single Assignment Form". Each region records
    // the current value of the Vars it defines, carries or sets. Reading a
    // Var looks it up through the enclosing regions; a loop body (or while
    // condition) it passes through gets block arguments for it on first
    // touch, initialised from the value outside, since a later iteration
    // sees what the previous one left. Leaving a region turns what it did to
    // its Vars into results of the region's operation.

    /// A new Var holding `values`, local to the current region.
    pub(crate) fn var_new(&mut self, values: Vec<ValueId>, loc: Loc) -> VarId {
        let var = VarId(self.vars.len() as u32);
        let types = values.iter().map(|&v| self.dtype(v)).collect();
        let depth = self.regions.len() - 1;
        let region = &mut self.regions[depth];
        self.vars.push(VarInfo { types, depth, serial: region.serial, loc });
        region.var_values.insert(var, values);
        var
    }

    /// The flattened types of `var`.
    pub(crate) fn var_types(&self, var: VarId) -> Vec<DType> {
        self.vars[var.0 as usize].types.clone()
    }

    /// Panic unless `var` is used inside the region that created it.
    fn var_check_scope(&self, var: VarId, loc: Loc) {
        let info = &self.vars[var.0 as usize];
        let alive = self.regions.get(info.depth).is_some_and(|r| r.serial == info.serial);
        assert!(
            alive,
            "cute: the Var created at {} is used at {loc} outside the region it was created in (a Var created \
             inside an if_/for_range/while_carry body is local to it)",
            info.loc
        );
    }

    /// The current value of `var`.
    pub(crate) fn var_get(&mut self, var: VarId, loc: Loc) -> Vec<ValueId> {
        self.var_check_scope(var, loc);
        self.var_lookup(var, self.regions.len() - 1)
    }

    /// The value of `var` in the region at `depth`, carrying it through every
    /// loop between there and the region that created it.
    fn var_lookup(&mut self, var: VarId, depth: usize) -> Vec<ValueId> {
        if let Some(values) = self.regions[depth].var_values.get(&var) {
            return values.clone();
        }
        assert!(depth > self.vars[var.0 as usize].depth, "cute: a Var lost its value");
        match self.regions[depth].kind {
            RegionKind::Loop | RegionKind::WhileCond => {
                let init = self.var_lookup(var, depth - 1);
                let args = self.var_block_args(var);
                let region = &mut self.regions[depth];
                region.carried.push(CarriedVar { var, init, args: args.clone() });
                region.var_values.insert(var, args.clone());
                args
            }
            RegionKind::Branch | RegionKind::Root => self.var_lookup(var, depth - 1),
        }
    }

    /// Give `var` the new value `values` (the user's `set` at `loc`), whose
    /// structure differs from the Var's as `mismatch` says.
    pub(crate) fn var_set(&mut self, var: VarId, values: Vec<ValueId>, mismatch: Option<String>, loc: Loc) {
        self.var_check_scope(var, loc);
        let info = &self.vars[var.0 as usize];
        let types: Vec<DType> = values.iter().map(|&v| self.dtype(v)).collect();
        if types != info.types {
            user_panic(
                loc,
                format!(
                    "cute: the Var created at {} holds {}, but is set to {}; a Var's type cannot change",
                    info.loc,
                    type_list(&info.types),
                    type_list(&types)
                ),
            );
        }
        if let Some(mismatch) = mismatch {
            user_panic(
                loc,
                format!(
                    "cute: the Var created at {} is set to a value of another structure ({mismatch}); a Var's type \
                     cannot change, and a register tensor's layout is part of its type",
                    info.loc
                ),
            );
        }
        let depth = self.regions.len() - 1;
        let in_condition = (info.depth + 1..=depth).any(|d| self.regions[d].kind == RegionKind::WhileCond);
        assert!(
            !in_condition,
            "cute: the Var created at {} is set at {loc} inside a while_carry condition, which can only read the \
             Vars created outside it; set it in the loop body",
            info.loc
        );
        // Touch it first, so that the loops around carry it.
        let _ = self.var_lookup(var, depth);
        self.var_assign(var, values);
    }

    /// Record `values` as the current value of `var` in the current region,
    /// as an operation's results or a `set`.
    pub(crate) fn var_assign(&mut self, var: VarId, values: Vec<ValueId>) {
        let depth = self.regions.len() - 1;
        let created_outside = self.vars[var.0 as usize].depth < depth;
        let region = &mut self.regions[depth];
        region.var_values.insert(var, values);
        if created_outside && !region.written.contains(&var) {
            region.written.push(var);
        }
    }

    /// The current value of `var`, which the caller knows to be in scope.
    pub(crate) fn var_current(&mut self, var: VarId) -> Vec<ValueId> {
        self.var_lookup(var, self.regions.len() - 1)
    }

    /// Carry `var` through the loop whose body is the current region, as a
    /// `while` body does for the Vars its condition read.
    pub(crate) fn var_seed_carried(&mut self, var: VarId, init: Vec<ValueId>) -> Vec<ValueId> {
        let args = self.var_block_args(var);
        let region = self.regions.last_mut().unwrap();
        debug_assert_eq!(region.kind, RegionKind::Loop);
        region.carried.push(CarriedVar { var, init, args: args.clone() });
        region.var_values.insert(var, args.clone());
        args
    }

    /// New block arguments for a loop to carry `var` in. Nothing is known
    /// about them; errors that this may cause name the Var
    /// ([`var_value_note`]).
    fn var_block_args(&mut self, var: VarId) -> Vec<ValueId> {
        let args: Vec<ValueId> = self.var_types(var).into_iter().map(|t| self.new_value(t)).collect();
        self.var_args.extend(args.iter().map(|&a| (a, var)));
        args
    }

    pub(crate) fn declare_buffer(&mut self, decl: BufferDecl) -> BufferId {
        let id = BufferId(self.buffers.len() as u32);
        if decl.space == MemSpace::Global {
            self.claim_param_name(&decl.name);
            self.params.push(Param::Buffer(id));
        }
        self.buffers.push(decl);
        id
    }

    /// Declare a by-value parameter of `size` bytes aligned to `align`, held in a read-only
    /// byte buffer.
    pub(crate) fn declare_by_value(&mut self, name: &str, size: usize, align: usize, grid_constant: bool) -> BufferId {
        self.claim_param_name(name);
        let id = BufferId(self.buffers.len() as u32);
        self.buffers.push(BufferDecl {
            name: name.to_string(),
            elem: DType::U8,
            len: Some(size),
            space: MemSpace::Global,
            align,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        });
        self.params.push(Param::ByValue { buffer: id, grid_constant });
        id
    }

    /// Declare a module global (its name is claimed like a parameter's: one C namespace).
    pub(crate) fn declare_module_global(
        &mut self,
        decl: BufferDecl,
        init: Option<Vec<u8>>,
        readonly: bool,
    ) -> BufferId {
        self.claim_param_name(&decl.name);
        let id = BufferId(self.buffers.len() as u32);
        self.buffers.push(decl);
        self.globals.push(crate::ir::ModuleGlobal { buffer: id, init, readonly });
        id
    }

    pub(crate) fn declare_scalar_param(&mut self, name: &str, ty: DType) -> ValueId {
        self.claim_param_name(name);
        let value = self.new_value(ty);
        self.params.push(Param::Scalar { name: name.to_string(), value, multiple_of: 1 });
        value
    }

    /// Declare a non-negative `i32` (or `i64`, as `ty` says) parameter known
    /// to be a multiple of `multiple_of`.
    pub(crate) fn declare_extent_param(&mut self, name: &str, multiple_of: usize, ty: DType) -> ValueId {
        self.claim_param_name(name);
        let value = self.new_value(ty);
        self.params.push(Param::Scalar { name: name.to_string(), value, multiple_of });
        self.facts.insert(value, ValueFacts { bound: None, divisor: multiple_of, pow2: false });
        value
    }

    /// The parameter of the [`SymInt`](crate::SymInt) with identity `id`,
    /// declared as `name` (an `i32` or `i64` extent known to be a multiple
    /// of `multiple_of`) on its first use.
    pub(crate) fn declare_symint(&mut self, id: u64, name: &str, multiple_of: usize, ty: DType) -> ValueId {
        if let Some(&v) = self.symints.get(&id) {
            return v;
        }
        let v = self.declare_extent_param(name, multiple_of, ty);
        self.symints.insert(id, v);
        v
    }

    /// Claim a parameter name. A misuse names the user's line where a
    /// [`LocationScope`] gives it (the `param_*` functions open one).
    fn claim_param_name(&mut self, name: &str) {
        let fail = |loc: Option<Loc>, msg: String| -> ! {
            match loc {
                Some(loc) => user_panic(loc, msg),
                None => panic!("{msg}"),
            }
        };
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            fail(self.location, format!("cute: parameter name {name:?} must be a non-empty identifier"));
        }
        if !self.param_names.insert(name.to_string()) {
            fail(self.location, format!("cute: duplicate parameter name {name:?}"));
        }
    }

    /// Every buffer declared so far, in order.
    pub(crate) fn buffer_ids(&self) -> impl Iterator<Item = BufferId> + use<> {
        (0..self.buffers.len() as u32).map(BufferId)
    }

    /// The kernel's own parameters declared so far.
    pub(crate) fn params(&self) -> &[Param] {
        &self.params
    }

    /// Declare a hidden by-value grid-constant parameter of `size` bytes (a tensor map),
    /// placed after the kernel's own parameters.
    pub(crate) fn declare_hidden_tensor_map(&mut self, name: &str, size: usize, align: usize) -> BufferId {
        self.claim_param_name(name);
        let id = BufferId(self.buffers.len() as u32);
        self.buffers.push(BufferDecl {
            name: name.to_string(),
            elem: DType::U8,
            len: Some(size),
            space: MemSpace::Global,
            align,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        });
        self.hidden_params.push(Param::ByValue { buffer: id, grid_constant: true });
        id
    }

    /// Move buffer parameter `buffer` after the kernel's other parameters
    /// (the device workspace, a trailing argument).
    pub(crate) fn move_param_last(&mut self, buffer: BufferId) {
        if let Some(i) = self.params.iter().position(|p| matches!(p, Param::Buffer(b) if *b == buffer)) {
            let p = self.params.remove(i);
            self.params.push(p);
        }
    }

    /// Record that by-value buffer `buffer` holds a tensor map.
    pub(crate) fn register_tensor_map(&mut self, buffer: BufferId, source: Option<crate::tma::TensorMapSource>) {
        self.tensor_maps.push((buffer, source));
    }

    /// The number of tensor-map parameters registered so far.
    pub(crate) fn tensor_map_count(&self) -> usize {
        self.tensor_maps.len()
    }

    /// The number of module globals declared so far.
    pub(crate) fn global_count(&self) -> usize {
        self.globals.len()
    }

    pub(crate) fn shared_buffer_count(&self) -> usize {
        self.buffers.iter().filter(|b| b.space == MemSpace::Shared).count()
    }

    pub(crate) fn raw_buffer_count(&self) -> usize {
        self.raw_bases.len()
    }

    pub(crate) fn local_buffer_count(&self) -> usize {
        self.buffers.iter().filter(|b| b.space == MemSpace::Local).count()
    }

    /// The kernel's dynamic shared-memory region (bytes sized at launch),
    /// declared on first use, starting at a multiple of `align` bytes: the
    /// largest alignment any use asks for, and at least 16, so that vector
    /// accesses of up to 16 bytes at aligned offsets are aligned.
    pub(crate) fn dynamic_smem_region(&mut self, align: usize) -> BufferId {
        let align = align.max(16);
        if let Some(i) = self.buffers.iter().position(|b| b.space == MemSpace::Shared && b.len.is_none()) {
            let region = &mut self.buffers[i];
            region.align = region.align.max(align);
            return BufferId(i as u32);
        }
        let name = DYNAMIC_SMEM_NAME.to_string();
        self.declare_buffer(BufferDecl {
            name,
            elem: DType::U8,
            len: None,
            space: MemSpace::Shared,
            align,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn buffer(&self, b: BufferId) -> &BufferDecl {
        &self.buffers[b.0 as usize]
    }

    pub(crate) fn buffer_mut(&mut self, b: BufferId) -> &mut BufferDecl {
        &mut self.buffers[b.0 as usize]
    }

    /// An `i32` that [`finish`](Self::finish) sets to the bytes of shared
    /// memory `partition` takes in the finished kernel
    /// ([`SharedMemoryLayout::partition_bytes`](crate::ir::SharedMemoryLayout::partition_bytes)),
    /// so that it counts buffers allocated after it too. It is a constant
    /// that nothing folds while the kernel is built (CuTe DSL's
    /// `cute.smem_partition_size`, resolved after tracing).
    pub(crate) fn partition_size(
        &mut self,
        partition: crate::ir::SmemPartition,
        cumulative: bool,
        loc: Loc,
    ) -> ValueId {
        let result = self.new_value(DType::I32);
        self.push(Op::Const { result, value: Literal::I32(0) }, loc);
        self.partition_queries.push((result, partition, cumulative));
        result
    }

    fn finish(mut self) -> KernelIr {
        assert_eq!(self.regions.len(), 1, "cute: unbalanced region stack");
        let insts = self.regions.pop().unwrap().insts;
        // A device function returns its result: the body's yield.
        let yields = self.device_function.as_ref().and_then(|f| f.result).into_iter().collect();
        let mut body = Region { insts, yields };
        eliminate_dead_code(&mut body);
        if let Some(buffer) = self.workspace.as_ref().map(|w| w.buffer) {
            self.move_param_last(buffer);
        }
        let mut args = self.params;
        args.extend(self.hidden_params);
        let tensor_maps = self
            .tensor_maps
            .into_iter()
            .map(|(buffer, source)| crate::tma::TensorMapParam {
                param: args
                    .iter()
                    .position(|p| matches!(p, Param::ByValue { buffer: b, .. } if *b == buffer))
                    .expect("a tensor map is a by-value parameter"),
                buffer,
                source,
            })
            .collect();
        let queries = std::mem::take(&mut self.partition_queries);
        let workspace = self.workspace.take().map(|w| w.finish(&args));
        let mut ir = KernelIr {
            workspace,
            name: self.name,
            args,
            tensor_maps,
            buffers: self.buffers,
            value_types: self.value_types,
            body,
            block_dim: self.block_dim,
            max_threads: self.max_threads,
            min_blocks_per_sm: self.min_blocks_per_sm,
            globals: self.globals,
            overlays: self.overlays,
            loc: self.loc,
            externs: self.externs,
            link_libraries: self.link_libraries,
            function: self.device_function,
            multicast: self.multicast,
            scopes: std::mem::take(&mut self.profiling.scopes),
            exact_float: std::mem::take(&mut self.exact_float),
        };
        if !queries.is_empty() {
            resolve_partition_queries(&mut ir, &queries);
        }
        ir
    }
}

/// Give each partition-size query ([`BuilderState::partition_size`]) its
/// value from the finished kernel's shared-memory layout.
fn resolve_partition_queries(ir: &mut KernelIr, queries: &[(ValueId, crate::ir::SmemPartition, bool)]) {
    fn walk(r: &mut Region, values: &HashMap<ValueId, i32>) {
        for inst in &mut r.insts {
            if let Op::Const { result, value } = &mut inst.op
                && let Some(&v) = values.get(result)
            {
                *value = Literal::I32(v);
            }
            for region in inst.op.regions_mut() {
                walk(region, values);
            }
        }
    }
    let layout = ir.shared_memory_layout();
    let values: HashMap<ValueId, i32> = queries
        .iter()
        .map(|&(v, p, cumulative)| (v, i32::try_from(layout.partition_bytes(p, cumulative)).unwrap_or(i32::MAX)))
        .collect();
    walk(&mut ir.body, &values);
}

/// Flattened types for messages: `i32`, `(f32, i32)` or `()`.
fn type_list(types: &[DType]) -> String {
    match types {
        [one] => one.to_string(),
        _ => format!("({})", types.iter().map(|t| t.name()).collect::<Vec<_>>().join(", ")),
    }
}

/// Remove pure instructions whose results are never used.
fn eliminate_dead_code(body: &mut Region) {
    fn count_uses(r: &Region, uses: &mut HashMap<ValueId, usize>) {
        for v in &r.yields {
            *uses.entry(*v).or_default() += 1;
        }
        for inst in &r.insts {
            for v in inst.op.operands() {
                *uses.entry(v).or_default() += 1;
            }
            inst.op.regions().into_iter().for_each(|r| count_uses(r, uses));
        }
    }
    fn sweep(r: &mut Region, uses: &HashMap<ValueId, usize>) -> bool {
        let before = r.insts.len();
        r.insts.retain(|i| !i.op.is_pure() || i.op.results().iter().any(|v| uses.contains_key(v)));
        let mut changed = r.insts.len() != before;
        for inst in &mut r.insts {
            for region in inst.op.regions_mut() {
                changed |= sweep(region, uses);
            }
        }
        changed
    }
    loop {
        let mut uses = HashMap::new();
        count_uses(body, &mut uses);
        if !sweep(body, &uses) {
            break;
        }
    }
}

/// Options for building a kernel. Start with [`Kernel::builder`].
#[derive(Clone, Debug)]
pub struct KernelOptions {
    name: String,
    block_dim: Option<[u32; 3]>,
    target: Option<crate::Arch>,
    max_threads: Option<[u32; 3]>,
    min_blocks_per_sm: Option<u32>,
    iket: Option<crate::profiling::IketOptions>,
}

impl KernelOptions {
    /// Turn on IKET in-kernel event tracing ([`mark`](crate::mark),
    /// [`range_start`](crate::range_start), ...): events append records to a hidden module
    /// global, which [`Kernel::decode_iket`] decodes. Off (the calls emit nothing) unless set.
    pub fn iket(mut self, options: crate::profiling::IketOptions) -> Self {
        self.iket = Some(options);
        self
    }

    /// Declare the block size the kernel is written for. Thread indices then
    /// have known bounds (which simplifies index arithmetic), `block_dim()`
    /// becomes a constant, and launches with any other block size are
    /// rejected.
    pub fn block_dim(mut self, dims: [u32; 3]) -> Self {
        assert!(dims.iter().all(|&d| d > 0), "block dimensions must be positive");
        assert!(dims.iter().product::<u32>() <= 1024, "at most 1024 threads per block");
        self.block_dim = Some(dims);
        self
    }

    /// The largest block the kernel may be launched with (CuTe DSL's
    /// `.launch(max_number_threads=[x, y, z])`): PTX's `.maxntid`, in place of the
    /// `.reqntid` a declared [`block_dim`](Self::block_dim) gives (PTX forbids both). Launches
    /// with a larger block in any mode, or more threads, are rejected.
    pub fn max_threads(mut self, dims: [u32; 3]) -> Self {
        assert!(dims.iter().all(|&d| d > 0), "max_threads: dimensions must be positive");
        assert!(dims.iter().product::<u32>() <= 1024, "max_threads: at most 1024 threads per block");
        self.max_threads = Some(dims);
        self
    }

    /// Ask the compiler to fit `n` blocks on each SM (CuTe DSL's `min_blocks_per_mp`): PTX's
    /// `.minnctapersm`, which caps registers per thread. Without a preferred shared-memory
    /// carveout, the CUDA launcher sets one that fits `n` blocks' shared memory on an SM.
    pub fn min_blocks_per_sm(mut self, n: u32) -> Self {
        assert!(n > 0, "min_blocks_per_sm: n must be positive");
        self.min_blocks_per_sm = Some(n);
        self
    }

    /// CuTe DSL's `hint_smem_base_uniform` (on by default) has no LLVM equivalent: shared
    /// memory bases are uniform in cute-rs kernels anyway, so this is accepted and ignored,
    /// like [`make_warp_uniform`](crate::make_warp_uniform).
    pub fn hint_smem_base_uniform(self, _uniform: bool) -> Self {
        self
    }

    /// The architecture the kernel is built for (CuTe DSL traces a kernel
    /// for one target: `cute.compile(..., options="--gpu-arch sm_90a")`, or
    /// `CUTE_DSL_ARCH`). During the build, [`target_arch`] returns it, so
    /// the body can specialise; at its end the kernel is checked against it
    /// ([`check_target`](crate::validate::check_target)), so an instruction
    /// the target lacks fails the build, naming the line that issued it.
    /// [`Kernel::compile_with`] then compiles for it unless the options name
    /// a target, and refuses another one.
    pub fn target(mut self, arch: crate::Arch) -> Self {
        self.target = Some(arch);
        self
    }

    /// Build the kernel by running `body`. Panics if the body misuses the
    /// DSL or produces invalid IR. The call site is recorded as the kernel's
    /// source location (the function of its line information).
    #[track_caller]
    pub fn build(self, body: impl FnOnce()) -> Kernel {
        let loc = std::panic::Location::caller();
        let build_started = std::time::Instant::now();
        crate::cute_log!(crate::log::Level::Info, "building kernel `{}` (at {loc})", self.name);
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                BUILDER.with(|b| *b.borrow_mut() = None);
            }
        }

        BUILDER.with(|b| {
            let mut b = b.borrow_mut();
            assert!(b.is_none(), "cute: Kernel::build cannot be nested");
            *b = Some(BuilderState {
                kernel: NEXT_KERNEL.fetch_add(1, Ordering::Relaxed),
                name: self.name.clone(),
                value_types: Vec::new(),
                facts: HashMap::new(),
                known_zero: HashSet::new(),
                exact_depth: 0,
                exact_float: Default::default(),
                constants: HashMap::new(),
                params: Vec::new(),
                param_names: HashSet::new(),
                buffers: Vec::new(),
                regions: vec![RegionBuilder::default()],
                next_serial: 0,
                vars: Vec::new(),
                var_args: HashMap::new(),
                block_dim: self.block_dim,
                target: self.target,
                location: None,
                loc,
                raw_bases: HashMap::new(),
                symints: HashMap::new(),
                assertion_records: Vec::new(),
                max_threads: self.max_threads,
                min_blocks_per_sm: self.min_blocks_per_sm,
                globals: Vec::new(),
                overlays: Vec::new(),
                hidden_params: Vec::new(),
                tensor_maps: Vec::new(),
                partition_queries: Vec::new(),
                externs: Vec::new(),
                link_libraries: Vec::new(),
                multicast: Vec::new(),
                device_function: None,
                profiling: crate::profiling::ProfilingState::new(self.iket),
                finish_hooks: Vec::new(),
                cute_ext_tmem: None,
                cute_ext_tma: HashMap::new(),
                workspace: None,
            });
        });
        let reset = Reset;
        crate::profiling::init_iket();
        body();
        run_finish_hooks();
        let mut state = BUILDER.with(|b| b.borrow_mut().take()).expect("builder state present");
        drop(reset);

        let assertions = std::mem::take(&mut state.assertion_records);
        if let (Some(b), Some(m)) = (self.block_dim, self.max_threads) {
            assert!(
                b.iter().zip(m).all(|(&b, m)| b <= m),
                "cute: kernel `{}` declares blocks of {b:?}, larger than its max_threads {m:?}",
                self.name
            );
        }
        let (symbols, iket) = state.profiling.finish_build(&self.name);
        let ir = state.finish();
        if let Err(e) = crate::validate::validate(&ir) {
            panic!("cute: kernel `{}` failed validation: {e}", ir.name);
        }
        if let Some(arch) = self.target
            && let Err(e) = crate::validate::check_target(&ir, arch)
        {
            panic!(
                "cute: kernel `{}` is built for {arch} (KernelOptions::target), which it cannot run on: {e}",
                ir.name
            );
        }
        crate::cute_log!(
            crate::log::Level::Info,
            "built kernel `{}` in {:.1} ms",
            ir.name,
            build_started.elapsed().as_secs_f64() * 1e3
        );
        Kernel { ir, target: self.target, assertions, symbols, iket }
    }
}

/// A compiled-to-IR kernel.
#[derive(Clone, Debug)]
pub struct Kernel {
    ir: KernelIr,
    target: Option<crate::Arch>,
    assertions: Vec<crate::testing::AssertionRecord>,
    symbols: crate::profiling::KernelSymbols,
    iket: Option<crate::profiling::IketInfo>,
}

impl Kernel {
    /// Start building a kernel named `name` (an identifier), to set options
    /// before [`KernelOptions::build`].
    pub fn builder(name: impl Into<String>) -> KernelOptions {
        let name = name.into();
        assert!(
            !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "kernel name {name:?} must be a non-empty identifier"
        );
        assert!(
            name != crate::ir::DYNAMIC_SMEM_SYMBOL,
            "kernel name {name:?} is reserved: it names the dynamic shared memory in the PTX"
        );
        KernelOptions { name, block_dim: None, target: None, max_threads: None, min_blocks_per_sm: None, iket: None }
    }

    /// Build a kernel named `name` by running `body`.
    #[track_caller]
    pub fn build(name: impl Into<String>, body: impl FnOnce()) -> Kernel {
        Kernel::builder(name).build(body)
    }

    /// The kernel (and PTX entry point) name.
    pub fn name(&self) -> &str {
        &self.ir.name
    }

    /// The validated IR.
    pub fn ir(&self) -> &KernelIr {
        &self.ir
    }

    /// Construct from IR (validated).
    pub fn from_ir(ir: KernelIr) -> Result<Kernel, crate::validate::ValidationError> {
        crate::validate::validate(&ir)?;
        Ok(Kernel { ir, target: None, assertions: Vec::new(), symbols: Default::default(), iket: None })
    }

    /// The architecture the kernel was built for ([`KernelOptions::target`]).
    pub fn target(&self) -> Option<crate::Arch> {
        self.target
    }

    /// The `RuntimeAssertion` stores the kernel makes: their parameter, index, message
    /// and location.
    pub fn assertion_records(&self) -> &[crate::testing::AssertionRecord] {
        &self.assertions
    }

    /// The profiling symbols the build registered (warp ranges, named barriers).
    pub fn symbols(&self) -> &crate::profiling::KernelSymbols {
        &self.symbols
    }

    /// `kernel_symbols.json`'s content.
    pub fn kernel_symbols_json(&self) -> String {
        self.symbols.to_json(&[])
    }

    /// Write `kernel_symbols.json` into `dir` (CuTe DSL's `dump_profiling_metadata`), with
    /// `extra` merged in as top-level keys. Writes nothing, and returns `None`, when the
    /// kernel registered neither barriers nor warps.
    pub fn dump_profiling_metadata(
        &self,
        dir: impl AsRef<std::path::Path>,
        extra: &[(&str, crate::profiling::MetaValue)],
    ) -> std::io::Result<Option<std::path::PathBuf>> {
        self.symbols.dump(dir.as_ref(), extra)
    }

    /// What decodes an IKET trace, when the kernel was built with [`KernelOptions::iket`].
    pub fn iket(&self) -> Option<&crate::profiling::IketInfo> {
        self.iket.as_ref()
    }

    /// Decode the words of the trace buffer (the module global
    /// [`IKET_GLOBAL`](crate::IKET_GLOBAL), read back after a launch or a simulation).
    pub fn decode_iket(&self, words: &[u32]) -> Result<crate::profiling::IketTrace, String> {
        let info = self.iket.as_ref().ok_or("cute: the kernel was built without KernelOptions::iket")?;
        crate::profiling::IketTrace::decode(info, &self.symbols, words)
    }

    /// Human-readable IR listing.
    pub fn dump(&self) -> String {
        crate::backend::dump(&self.ir)
    }
}
