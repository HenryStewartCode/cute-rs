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
    BinaryOp, BufferDecl, BufferId, CmpPred, DType, FloatFn, FloatMode, Inst, KernelIr, Literal, Loc, MemSpace, Op,
    Param, Region, SpecialReg, UnaryOp, ValueId,
};
use crate::layout::IndexExpr;

/// What is statically known about a runtime integer value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValueFacts {
    /// Exclusive upper bound.
    pub bound: Option<usize>,
    /// Every value is a multiple of this.
    pub divisor: usize,
}

impl Default for ValueFacts {
    fn default() -> Self {
        ValueFacts { bound: None, divisor: 1 }
    }
}

/// Key for common-subexpression elimination of pure operations.
#[derive(Clone, PartialEq, Eq, Hash)]
enum PureKey {
    Const(Literal),
    Unary(UnaryOp, ValueId),
    Binary(BinaryOp, ValueId, ValueId),
    Fma(ValueId, ValueId, ValueId),
    Float(FloatFn, FloatMode, bool, bool, Vec<ValueId>),
    Cmp(CmpPred, ValueId, ValueId),
    Select(ValueId, ValueId, ValueId),
    Cast(DType, ValueId),
    Bitcast(DType, ValueId),
    Special(SpecialReg),
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
    /// While set, instructions record this location instead of their own:
    /// the user's call of a library function expanded into many operations
    /// (`cute::math`).
    location: Option<Loc>,
}

thread_local! {
    static BUILDER: RefCell<Option<BuilderState>> = const { RefCell::new(None) };
}

static NEXT_KERNEL: AtomicU32 = AtomicU32::new(1);

/// Run `f` on the active builder. Panics outside [`Kernel::build`].
#[track_caller]
pub(crate) fn with<R>(f: impl FnOnce(&mut BuilderState) -> R) -> R {
    BUILDER.with(|b| {
        let mut b = b.borrow_mut();
        let state = b.as_mut().unwrap_or_else(|| panic!("cute: DSL operation used outside of Kernel::build"));
        f(state)
    })
}

/// Facts about `v` if a kernel is being built (and the builder is not
/// already borrowed); defaults otherwise.
pub(crate) fn value_facts(v: ValueId) -> ValueFacts {
    BUILDER.with(|b| {
        b.try_borrow().ok().and_then(|b| b.as_ref().and_then(|s| s.facts.get(&v).copied())).unwrap_or_default()
    })
}

/// Attributes every instruction built while it lives to its creator's
/// caller (see [`BuilderState::set_location`]).
pub(crate) struct LocationScope(bool);

impl LocationScope {
    #[track_caller]
    pub(crate) fn new() -> LocationScope {
        let loc = std::panic::Location::caller();
        LocationScope(with(|b| b.set_location(loc)))
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
    /// the greatest common divisor (a constant 0 divides by anything).
    pub(crate) fn merge_facts(&mut self, result: ValueId, sources: &[ValueId]) {
        if sources.is_empty() {
            return;
        }
        let mut bound = Some(0);
        let mut divisor = 0;
        for &s in sources {
            let facts = self.facts_of(s);
            bound = bound.zip(facts.bound).map(|(a, b)| a.max(b));
            if self.constant_of(s) != Some(Literal::I32(0)) {
                divisor = gcd(divisor, facts.divisor);
            }
        }
        let facts = ValueFacts { bound, divisor: divisor.max(1) };
        if facts.bound.is_some() || facts.divisor > 1 {
            self.facts.insert(result, facts);
        }
    }

    pub(crate) fn push(&mut self, op: Op, loc: Loc) {
        let loc = self.location.unwrap_or(loc);
        self.regions.last_mut().expect("region stack is never empty while building").insts.push(Inst { op, loc });
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
        if let Literal::I32(c) = value
            && c >= 0
        {
            let c = c as usize;
            self.facts.insert(v, ValueFacts { bound: Some(c + 1), divisor: c.max(1) });
        }
        v
    }

    pub(crate) fn unary(&mut self, op: UnaryOp, x: ValueId, loc: Loc) -> ValueId {
        let ty = self.dtype(x);
        self.pure(PureKey::Unary(op, x), ty, loc, |result| Op::Unary { result, op, x })
    }

    pub(crate) fn binary(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId, loc: Loc) -> ValueId {
        let (lt, rt) = (self.dtype(lhs), self.dtype(rhs));
        assert_eq!(lt, rt, "cute: {op:?} on mismatched types {lt} and {rt}");
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
        self.pure(key, lt, loc, |result| Op::Binary { result, op, lhs, rhs })
    }

    pub(crate) fn fma(&mut self, a: ValueId, b: ValueId, c: ValueId, loc: Loc) -> ValueId {
        let ty = self.dtype(a);
        self.pure(PureKey::Fma(a, b, c), ty, loc, |result| Op::Fma { result, a, b, c })
    }

    /// A floating-point instruction with explicit modifiers (validated with
    /// the kernel).
    pub(crate) fn float_inst(
        &mut self,
        func: FloatFn,
        args: Vec<ValueId>,
        mode: FloatMode,
        ftz: bool,
        nan: bool,
        loc: Loc,
    ) -> ValueId {
        let ty = self.dtype(args[0]);
        let key = PureKey::Float(func, mode, ftz, nan, args.clone());
        self.pure(key, ty, loc, |result| Op::FloatInst { result, func, args, mode, ftz, nan })
    }

    pub(crate) fn cmp(&mut self, pred: CmpPred, lhs: ValueId, rhs: ValueId, loc: Loc) -> ValueId {
        let (lt, rt) = (self.dtype(lhs), self.dtype(rhs));
        assert_eq!(lt, rt, "cute: comparison of mismatched types {lt} and {rt}");
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
        if self.dtype(x) == to {
            return x;
        }
        self.pure(PureKey::Cast(to, x), to, loc, |result| Op::Cast { result, x })
    }

    pub(crate) fn bitcast(&mut self, x: ValueId, to: DType, loc: Loc) -> ValueId {
        assert_eq!(self.dtype(x).bits(), to.bits(), "cute: bitcast between {} and {to}", self.dtype(x));
        if self.dtype(x) == to {
            return x;
        }
        self.pure(PureKey::Bitcast(to, x), to, loc, |result| Op::Bitcast { result, x })
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
        self.facts.insert(v, ValueFacts { bound, divisor: 1 });
        v
    }

    /// Turn an index expression into an `i32` value.
    pub(crate) fn materialize(&mut self, e: &IndexExpr, loc: Loc) -> ValueId {
        let bin = |s: &mut Self, op, a: &IndexExpr, b: &IndexExpr| {
            let (a, b) = (s.materialize(a, loc), s.materialize(b, loc));
            s.binary(op, a, b, loc)
        };
        match e {
            IndexExpr::Static(v) => {
                let v = i32::try_from(*v).unwrap_or_else(|_| panic!("cute: index {v} does not fit in i32"));
                self.constant(Literal::I32(v), loc)
            }
            IndexExpr::Dynamic(v) => {
                assert_eq!(self.dtype(*v), DType::I32, "cute: index expressions must be built from i32 values");
                *v
            }
            IndexExpr::Add(a, b) => bin(self, BinaryOp::Add, a, b),
            IndexExpr::Mul(a, b) => bin(self, BinaryOp::Mul, a, b),
            IndexExpr::Div(a, b) => bin(self, BinaryOp::DivU, a, b),
            IndexExpr::Rem(a, b) => bin(self, BinaryOp::RemU, a, b),
            IndexExpr::And(a, b) => bin(self, BinaryOp::And, a, b),
            IndexExpr::Xor(a, b) => bin(self, BinaryOp::Xor, a, b),
            IndexExpr::Shr(a, b) => bin(self, BinaryOp::ShrU, a, b),
        }
    }

    pub(crate) fn push_region(&mut self, kind: RegionKind) {
        assert_ne!(kind, RegionKind::Root, "cute: only the kernel body is a root region");
        self.next_serial += 1;
        self.regions.push(RegionBuilder { kind, serial: self.next_serial, ..RegionBuilder::default() });
    }

    pub(crate) fn pop_region(&mut self) -> ClosedRegion {
        assert!(self.regions.len() > 1, "cute: unbalanced region stack");
        let r = self.regions.pop().unwrap();
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
        assert!(
            types == info.types,
            "cute: the Var created at {} holds {}, but is set to {} at {loc}; a Var's type cannot change",
            info.loc,
            type_list(&info.types),
            type_list(&types)
        );
        if let Some(mismatch) = mismatch {
            panic!(
                "cute: the Var created at {} is set to a value of another structure at {loc} ({mismatch}); a Var's \
                 type cannot change, and a register tensor's layout is part of its type",
                info.loc
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

    pub(crate) fn declare_scalar_param(&mut self, name: &str, ty: DType) -> ValueId {
        self.claim_param_name(name);
        let value = self.new_value(ty);
        self.params.push(Param::Scalar { name: name.to_string(), value, multiple_of: 1 });
        value
    }

    /// Declare a non-negative `i32` parameter known to be a multiple of
    /// `multiple_of`.
    pub(crate) fn declare_extent_param(&mut self, name: &str, multiple_of: usize) -> ValueId {
        self.claim_param_name(name);
        let value = self.new_value(DType::I32);
        self.params.push(Param::Scalar { name: name.to_string(), value, multiple_of });
        self.facts.insert(value, ValueFacts { bound: None, divisor: multiple_of });
        value
    }

    fn claim_param_name(&mut self, name: &str) {
        assert!(
            !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "cute: parameter name {name:?} must be a non-empty identifier"
        );
        assert!(self.param_names.insert(name.to_string()), "cute: duplicate parameter name {name:?}");
    }

    pub(crate) fn shared_buffer_count(&self) -> usize {
        self.buffers.iter().filter(|b| b.space == MemSpace::Shared).count()
    }

    pub(crate) fn local_buffer_count(&self) -> usize {
        self.buffers.iter().filter(|b| b.space == MemSpace::Local).count()
    }

    #[allow(dead_code)]
    pub(crate) fn buffer(&self, b: BufferId) -> &BufferDecl {
        &self.buffers[b.0 as usize]
    }

    fn finish(mut self) -> KernelIr {
        assert_eq!(self.regions.len(), 1, "cute: unbalanced region stack");
        let insts = self.regions.pop().unwrap().insts;
        let mut body = Region { insts, yields: Vec::new() };
        eliminate_dead_code(&mut body);
        KernelIr {
            name: self.name,
            args: self.params,
            buffers: self.buffers,
            value_types: self.value_types,
            body,
            block_dim: self.block_dim,
        }
    }
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

/// A compiled-to-IR kernel.
#[derive(Clone, Debug)]
pub struct Kernel {
    ir: KernelIr,
}

/// Options for building a kernel. Start with [`Kernel::builder`].
#[derive(Clone, Debug)]
pub struct KernelOptions {
    name: String,
    block_dim: Option<[u32; 3]>,
}

impl KernelOptions {
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

    /// Build the kernel by running `body`. Panics if the body misuses the
    /// DSL or produces invalid IR.
    pub fn build(self, body: impl FnOnce()) -> Kernel {
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
                constants: HashMap::new(),
                params: Vec::new(),
                param_names: HashSet::new(),
                buffers: Vec::new(),
                regions: vec![RegionBuilder::default()],
                next_serial: 0,
                vars: Vec::new(),
                var_args: HashMap::new(),
                block_dim: self.block_dim,
                location: None,
            });
        });
        let reset = Reset;
        body();
        let state = BUILDER.with(|b| b.borrow_mut().take()).expect("builder state present");
        drop(reset);

        let ir = state.finish();
        if let Err(e) = crate::validate::validate(&ir) {
            panic!("cute: kernel `{}` failed validation: {e}", ir.name);
        }
        Kernel { ir }
    }
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
        KernelOptions { name, block_dim: None }
    }

    /// Build a kernel named `name` by running `body`.
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
        Ok(Kernel { ir })
    }

    /// Human-readable IR listing.
    pub fn dump(&self) -> String {
        crate::backend::dump(&self.ir)
    }
}
