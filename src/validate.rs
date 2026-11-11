//! IR verification: scoping (SSA dominance), types, and instruction
//! constraints. Every kernel is validated when it is built, so backends and
//! the simulator can rely on well-formed IR.

use std::collections::HashMap;
use std::fmt;

use crate::backend::Arch;
use crate::ir::{
    AtomicKind, BarrierKind, BinaryOp, BufferId, CmpPred, DType, FenceScope, FloatFn, FloatMode, KernelIr, Literal,
    Loc, LoopOptions, MbarrierWaitKind, MemScopeKind, MemSpace, Op, Param, Region, SpecialReg, UnaryOp, ValueId,
    VoteKind, float_inst_name,
};

/// Static shared memory available to one block. Kernels using more are
/// compiled to allocate it dynamically, at launch.
pub const MAX_STATIC_SHARED_BYTES: usize = 48 * 1024;

/// The most shared memory any supported GPU gives one block. What a given
/// device allows can be less; the launcher checks.
pub const MAX_SHARED_BYTES: usize = 227 * 1024;

#[derive(Clone, Debug)]
pub struct ValidationError {
    pub message: String,
    pub loc: Option<Loc>,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.loc {
            Some(loc) => write!(f, "{}:{}:{}: {}", loc.file(), loc.line(), loc.column(), self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for ValidationError {}

struct Checker<'a> {
    ir: &'a KernelIr,
    visible: Vec<bool>,
    defined: Vec<bool>,
    loc: Option<Loc>,
    /// Values defined by constants, for checking immediate operands.
    constants: HashMap<ValueId, Literal>,
}

type Result<T = ()> = std::result::Result<T, ValidationError>;

impl Checker<'_> {
    fn err(&self, message: impl Into<String>) -> ValidationError {
        ValidationError { message: message.into(), loc: self.loc }
    }

    fn ty(&self, v: ValueId) -> DType {
        self.ir.dtype(v)
    }

    fn use_(&self, v: ValueId) -> Result {
        let i = v.0 as usize;
        if i >= self.visible.len() {
            return Err(self.err(format!("{v} does not exist")));
        }
        if !self.visible[i] {
            return Err(self.err(if self.defined[i] {
                format!(
                    "{v} is used outside the region that defines it (values computed inside an if/for/while body, \
                     including values read from a Var there, cannot escape it; yield or carry them, or keep them \
                     in a Var and read it after the region)"
                )
            } else {
                format!("{v} is used before it is defined")
            }));
        }
        Ok(())
    }

    fn define(&mut self, v: ValueId, scope: &mut Vec<ValueId>) -> Result {
        let i = v.0 as usize;
        if i >= self.visible.len() {
            return Err(self.err(format!("{v} does not exist")));
        }
        if self.defined[i] {
            return Err(self.err(format!("{v} is defined twice")));
        }
        self.defined[i] = true;
        self.visible[i] = true;
        scope.push(v);
        Ok(())
    }

    fn expect(&self, v: ValueId, ty: DType, what: &str) -> Result {
        if self.ty(v) != ty {
            return Err(self.err(format!("{what} must be {ty}, but {v} is {}", self.ty(v))));
        }
        Ok(())
    }

    fn same(&self, values: &[ValueId], what: &str) -> Result<DType> {
        let t = self.ty(values[0]);
        for &v in &values[1..] {
            if self.ty(v) != t {
                return Err(self.err(format!("{what}: operand types differ ({t} and {})", self.ty(v))));
            }
        }
        Ok(t)
    }

    fn region(&mut self, r: &Region, args: &[ValueId]) -> Result {
        let mut scope = Vec::new();
        for &a in args {
            self.define(a, &mut scope)?;
        }
        for inst in &r.insts {
            self.loc = Some(inst.loc);
            self.inst(&inst.op)?;
            for v in inst.op.results() {
                self.define(v, &mut scope)?;
            }
        }
        for &y in &r.yields {
            self.use_(y)?;
        }
        for v in scope {
            self.visible[v.0 as usize] = false;
        }
        Ok(())
    }

    fn vector_width(&self, n: usize, elem: DType) -> Result {
        let bytes = n * elem.bytes();
        if !n.is_power_of_two() || bytes > 16 {
            return Err(
                self.err(format!("vector access of {n} x {elem} is not a power-of-two width of at most 16 bytes"))
            );
        }
        Ok(())
    }

    fn inst(&mut self, op: &Op) -> Result {
        for v in op.operands() {
            self.use_(v)?;
        }
        match op {
            Op::Const { result, value } => {
                self.expect(*result, value.dtype(), "constant")?;
                self.constants.insert(*result, *value);
            }
            Op::Unary { result, op, x } => {
                let t = self.ty(*x);
                let ok = match op {
                    UnaryOp::Neg => t != DType::Bool,
                    UnaryOp::Abs => t.is_float() || t.is_signed(),
                    UnaryOp::Not => t == DType::Bool || t.is_int(),
                    UnaryOp::Sqrt
                    | UnaryOp::Rsqrt
                    | UnaryOp::Rcp
                    | UnaryOp::Floor
                    | UnaryOp::Ceil
                    | UnaryOp::Trunc
                    | UnaryOp::Rint => t.is_float(),
                    UnaryOp::Exp2 | UnaryOp::Log2 | UnaryOp::Sin | UnaryOp::Cos => t.is_float() && t != DType::F64,
                    UnaryOp::Popc | UnaryOp::Clz | UnaryOp::Ctz | UnaryOp::Brev => t.is_int(),
                };
                if !ok && t == DType::F64 {
                    let (ptx, name) = match op {
                        UnaryOp::Exp2 => ("ex2", "exp2"),
                        UnaryOp::Log2 => ("lg2", "log2"),
                        UnaryOp::Sin => ("sin", "sin"),
                        _ => ("cos", "cos"),
                    };
                    return Err(self.err(format!(
                        "{op:?} is not defined on f64: it is the hardware approximation {ptx}.approx, which exists \
                         for Float32 only; cute::math::{name} computes Float64"
                    )));
                }
                if !ok {
                    return Err(self.err(format!("{op:?} is not defined on {t}")));
                }
                self.expect(*result, t, "unary result")?;
            }
            Op::Binary { result, op, lhs, rhs } => {
                let t = self.same(&[*lhs, *rhs, *result], &format!("{op:?}"))?;
                let ok = match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Min | BinaryOp::Max => {
                        t != DType::Bool
                    }
                    BinaryOp::DivU
                    | BinaryOp::Rem
                    | BinaryOp::RemU
                    | BinaryOp::Shl
                    | BinaryOp::ShrU
                    | BinaryOp::ShrS => t.is_int(),
                    BinaryOp::MulHi => t.is_int() && t.bits() == 32,
                    BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => t.is_int() || t == DType::Bool,
                };
                if !ok {
                    return Err(self.err(format!("{op:?} is not defined on {t}")));
                }
            }
            Op::Fma { result, a, b, c } => {
                let t = self.same(&[*a, *b, *c, *result], "fma")?;
                if !t.is_float() {
                    return Err(self.err(format!("fma is not defined on {t}")));
                }
            }
            Op::FloatInst { result, func, args, mode, ftz, nan } => {
                self.float_inst(*result, *func, args, *mode, *ftz, *nan)?
            }
            Op::Cmp { result, pred, lhs, rhs } => {
                let t = self.same(&[*lhs, *rhs], "comparison")?;
                if t == DType::Bool && !matches!(pred, CmpPred::Eq | CmpPred::Ne) {
                    return Err(self.err("booleans can only be compared for equality"));
                }
                self.expect(*result, DType::Bool, "comparison result")?;
            }
            Op::Select { result, cond, on_true, on_false } => {
                self.expect(*cond, DType::Bool, "select condition")?;
                self.same(&[*on_true, *on_false, *result], "select")?;
            }
            Op::Cast { result, x } => {
                if self.ty(*result) == DType::Bool {
                    return Err(self.err(format!("cannot cast {} to i1; compare instead", self.ty(*x))));
                }
            }
            Op::Bitcast { result, x } => {
                if self.ty(*result).bits() != self.ty(*x).bits() {
                    return Err(self.err(format!("cannot bitcast {} to {}", self.ty(*x), self.ty(*result))));
                }
            }
            Op::Special { result, reg } => self.expect(*result, reg.dtype(), "special register")?,
            Op::Load { results, buffer, index } => {
                self.expect(*index, DType::I32, "load index")?;
                let _ = self.ir.buffer(*buffer);
                let t = self.same(results, "load")?;
                self.vector_width(results.len(), t)?;
            }
            Op::Store { values, buffer, index } => {
                self.expect(*index, DType::I32, "store index")?;
                let _ = self.ir.buffer(*buffer);
                let t = self.same(values, "store")?;
                self.vector_width(values.len(), t)?;
            }
            Op::CpAsync { src, src_index, dst, dst_index, elem, elems, bypass_l1, pred } => {
                let (s, d) = (self.ir.buffer(*src), self.ir.buffer(*dst));
                if s.space != MemSpace::Global || d.space != MemSpace::Shared {
                    return Err(self.err("cp.async copies from global to shared memory"));
                }
                let bytes = *elems as usize * elem.bytes();
                if !matches!(bytes, 4 | 8 | 16) || (*bypass_l1 && bytes != 16) {
                    return Err(self.err(format!("cp.async of {bytes} bytes is not supported")));
                }
                self.expect(*src_index, DType::I32, "cp.async source index")?;
                self.expect(*dst_index, DType::I32, "cp.async destination index")?;
                if let Some(p) = pred {
                    self.expect(*p, DType::Bool, "cp.async predicate")?;
                }
            }
            Op::CpAsyncCommitGroup
            | Op::CpAsyncWaitGroup { .. }
            | Op::Barrier
            | Op::Fence { .. }
            | Op::Printf { .. } => {}
            Op::InlinePtx { .. } => {}
            Op::NamedBarrier { kind, id, threads } => self.named_barrier(*kind, *id, *threads)?,
            Op::WarpBarrier { mask } => self.expect(*mask, DType::U32, "lane mask")?,
            Op::Shuffle { result, value, offset, mask, .. } => {
                if self.ty(*value).bits() != 32 {
                    return Err(self.err(format!("shuffles move 32-bit values, not {}", self.ty(*value))));
                }
                self.expect(*result, self.ty(*value), "shuffle result")?;
                self.expect(*offset, DType::I32, "shuffle offset")?;
                self.expect(*mask, DType::U32, "lane mask")?;
            }
            Op::Vote { result, kind, pred, mask } => {
                self.expect(*pred, DType::Bool, "vote predicate")?;
                self.expect(*mask, DType::U32, "lane mask")?;
                let want = if *kind == VoteKind::Ballot { DType::U32 } else { DType::Bool };
                self.expect(*result, want, "vote result")?;
            }
            Op::MatchAny { result, value, mask } => {
                if self.ty(*value).bits() != 32 {
                    return Err(self.err(format!("match_sync compares 32-bit values, not {}", self.ty(*value))));
                }
                self.expect(*mask, DType::U32, "lane mask")?;
                self.expect(*result, DType::U32, "match result")?;
            }
            Op::ElectOne { result, mask } => {
                self.expect(*mask, DType::U32, "lane mask")?;
                self.expect(*result, DType::Bool, "elect result")?;
            }
            Op::ActiveMask { result } => self.expect(*result, DType::U32, "active mask")?,
            Op::WarpUniform { result, value } => {
                self.expect(*value, DType::I32, "make_warp_uniform operand")?;
                self.expect(*result, DType::I32, "make_warp_uniform result")?;
            }
            Op::MbarrierInit { buffer, index, count } => {
                self.mbarrier(*buffer, *index)?;
                self.expect(*count, DType::I32, "mbarrier count")?;
                if let Some(n) = self.constant_i32(*count)
                    && !(1..1 << 20).contains(&n)
                {
                    return Err(self.err(format!("an mbarrier count of {n} is outside 1 to 2^20 - 1")));
                }
            }
            Op::MbarrierArrive { buffer, index, count, expect_tx, peer, .. } => {
                self.mbarrier(*buffer, *index)?;
                if let Some(c) = count {
                    self.expect(*c, DType::I32, "mbarrier arrival count")?;
                    if let Some(n) = self.constant_i32(*c)
                        && !(1..1 << 20).contains(&n)
                    {
                        return Err(self.err(format!("an mbarrier arrival count of {n} is outside 1 to 2^20 - 1")));
                    }
                    if expect_tx.is_some() {
                        return Err(self.err("mbarrier.arrive.expect_tx arrives once; it takes no arrival count"));
                    }
                }
                if let Some(tx) = expect_tx {
                    self.expect(*tx, DType::I32, "expected bytes")?;
                }
                if let Some(p) = peer {
                    self.expect(*p, DType::I32, "peer CTA rank")?;
                }
            }
            Op::MbarrierExpectTx { buffer, index, bytes, peer, .. } => {
                self.mbarrier(*buffer, *index)?;
                self.expect(*bytes, DType::I32, "expected bytes")?;
                if let Some(p) = peer {
                    self.expect(*p, DType::I32, "peer CTA rank")?;
                }
            }
            Op::MbarrierWait { result, buffer, index, parity, .. } => {
                self.mbarrier(*buffer, *index)?;
                self.expect(*parity, DType::I32, "phase parity")?;
                self.expect(*result, DType::Bool, "mbarrier wait result")?;
            }
            Op::FenceMbarrierInit => {}
            Op::CpAsyncBulk { src, src_index, dst, dst_index, bytes, mbarrier, mbarrier_index, .. } => {
                if self.ir.buffer(*src).space != MemSpace::Global || self.ir.buffer(*dst).space != MemSpace::Shared {
                    return Err(self.err("cp.async.bulk copies from global to shared memory"));
                }
                self.expect(*src_index, DType::I32, "cp.async.bulk source index")?;
                self.expect(*dst_index, DType::I32, "cp.async.bulk destination index")?;
                self.expect(*bytes, DType::I32, "cp.async.bulk size")?;
                self.mbarrier(*mbarrier, *mbarrier_index)?;
            }
            Op::Atomic { result, kind, buffer, index, value, compare } => {
                let t = self.ty(*value);
                if !t.is_int() || t.bits() < 32 {
                    return Err(self.err(format!("atomics operate on 32- and 64-bit integers, not {t}")));
                }
                if self.ir.buffer(*buffer).space == MemSpace::Local {
                    return Err(self.err("atomics need global or shared memory"));
                }
                self.expect(*index, DType::I32, "atomic index")?;
                self.expect(*result, t, "atomic result")?;
                if compare.is_some() != (*kind == AtomicKind::CompareExchange) {
                    return Err(self.err("only compare-and-exchange takes a comparand"));
                }
                if let Some(c) = compare {
                    self.expect(*c, t, "atomic comparand")?;
                }
            }
            Op::LdMatrix { results, buffer, index, num, .. } => {
                let b = self.ir.buffer(*buffer);
                if b.space != MemSpace::Shared {
                    return Err(self.err("ldmatrix reads shared memory"));
                }
                if !matches!(num, 1 | 2 | 4) || results.len() != 2 * *num as usize {
                    return Err(self.err(format!("ldmatrix.x{num} yields {} values", 2 * num)));
                }
                self.expect(*index, DType::I32, "ldmatrix index")?;
                let t = self.same(results, "ldmatrix")?;
                if t.bits() != 16 {
                    return Err(self.err(format!("ldmatrix moves 16-bit elements, not {t}")));
                }
            }
            Op::Mma { kind, d, a, b, c } => {
                let (na, nb, nc) = kind.fragment_sizes();
                let (ta, tb, tc) = kind.dtypes();
                if a.len() != na || b.len() != nb || c.len() != nc || d.len() != nc {
                    return Err(self.err(format!("{kind:?} takes {na}, {nb} and {nc} values per thread")));
                }
                for &v in a {
                    self.expect(v, ta, "mma A")?;
                }
                for &v in b {
                    self.expect(v, tb, "mma B")?;
                }
                for &v in c.iter().chain(d) {
                    self.expect(v, tc, "mma C/D")?;
                }
            }
            Op::If { cond, then_region, else_region, results } => {
                self.expect(*cond, DType::Bool, "if condition")?;
                for r in [then_region, else_region] {
                    self.region(r, &[])?;
                    self.yields_match(&r.yields, results, "if branch")?;
                }
            }
            Op::For { iv, lower, upper, step, init, args, body, results, opts } => {
                for v in [*lower, *upper, *step] {
                    self.expect(v, DType::I32, "loop bound")?;
                }
                self.loop_options(opts, *lower, *upper)?;
                self.expect(*iv, DType::I32, "induction variable")?;
                self.yields_match(init, args, "loop initial values")?;
                self.yields_match(init, results, "loop results")?;
                let mut block_args = vec![*iv];
                block_args.extend(args);
                self.region(body, &block_args)?;
                self.yields_match(&body.yields, results, "loop body")?;
            }
            Op::While { init, cond_args, cond, body_args, body, results } => {
                self.yields_match(init, cond_args, "loop initial values")?;
                self.yields_match(init, body_args, "loop initial values")?;
                self.yields_match(init, results, "loop results")?;
                self.region(cond, cond_args)?;
                if cond.yields.len() != 1 || self.ty(cond.yields[0]) != DType::Bool {
                    return Err(self.err("a while condition yields one boolean"));
                }
                self.region(body, body_args)?;
                self.yields_match(&body.yields, results, "loop body")?;
            }
        }
        Ok(())
    }

    /// A floating-point instruction must exist in PTX with exactly these
    /// modifiers.
    fn float_inst(
        &self,
        result: ValueId,
        func: FloatFn,
        args: &[ValueId],
        mode: FloatMode,
        ftz: bool,
        nan: bool,
    ) -> Result {
        if args.len() != func.arity() {
            return Err(self.err(format!("{} takes {} operands, not {}", func.ptx(), func.arity(), args.len())));
        }
        let mut all = args.to_vec();
        all.push(result);
        let t = self.same(&all, func.ptx())?;
        let name = float_inst_name(func, mode, ftz, nan, t);
        if !matches!(t, DType::F32 | DType::F64) {
            return Err(self.err(format!(
                "{name}: explicit approx, ftz and rounding modifiers need Float32 or Float64 operands, not {t}"
            )));
        }
        let f32 = t == DType::F32;
        use FloatFn::*;
        let (modes_ok, why) = match func {
            Add | Mul | Fma => (matches!(mode, FloatMode::Round(_)), "takes a rounding mode"),
            Div => (
                matches!(mode, FloatMode::Round(_)) || (f32 && matches!(mode, FloatMode::Approx | FloatMode::Full)),
                "takes a rounding mode, or approx or full on Float32",
            ),
            Rcp => (
                matches!(mode, FloatMode::Round(_)) || (mode == FloatMode::Approx && (f32 || ftz)),
                "takes a rounding mode, or approx (on Float64 only as rcp.approx.ftz.f64)",
            ),
            Sqrt => (
                matches!(mode, FloatMode::Round(_)) || (f32 && mode == FloatMode::Approx),
                "takes a rounding mode, or approx on Float32",
            ),
            Rsqrt => (mode == FloatMode::Approx, "is approximate only"),
            Ex2 | Lg2 | Sin | Cos => (f32 && mode == FloatMode::Approx, "is approximate only, and Float32 only"),
            Tanh => (f32 && mode == FloatMode::Approx && !ftz, "exists only as tanh.approx.f32"),
            Abs | Min | Max => (mode == FloatMode::Exact, "takes no rounding mode"),
        };
        if !modes_ok {
            return Err(self.err(format!("{name} is not a PTX instruction: {} {why}", func.ptx())));
        }
        // `.ftz` is a single-precision modifier, except on the two f64
        // approximations that have it (`rcp.rn.ftz.f64` does not exist).
        if ftz && !f32 && !(matches!(func, Rcp | Rsqrt) && mode == FloatMode::Approx) {
            return Err(self.err(format!(
                "{name} is not a PTX instruction: ftz applies to Float32 only, and to Float64 only in \
                 rcp.approx.ftz.f64 and rsqrt.approx.ftz.f64"
            )));
        }
        if nan && (!matches!(func, Min | Max) || !f32) {
            return Err(
                self.err(format!("{name} is not a PTX instruction: NaN propagation exists for Float32 min and max"))
            );
        }
        Ok(())
    }

    /// The hints of a `for` loop (`cutlass.range`'s keyword arguments).
    fn loop_options(&self, opts: &LoopOptions, lower: ValueId, upper: ValueId) -> Result {
        if let (Some(count), true) = (opts.unroll, opts.unroll_full) {
            return Err(
                self.err(format!("for_range_with: unroll ({count}) and unroll_full contradict each other; choose one"))
            );
        }
        if let Some(count) = opts.unroll
            && i32::try_from(count).is_err()
        {
            return Err(self.err(format!("for_range_with: an unroll count of {count} does not fit in i32")));
        }
        if opts.at_least_once
            && let (Some(lo), Some(hi)) = (self.constant_i32(lower), self.constant_i32(upper))
            && lo >= hi
        {
            return Err(
                self.err(format!("for_range_with: the loop is marked at_least_once, but {lo}..{hi} runs zero times"))
            );
        }
        Ok(())
    }

    /// The value of `v` if it is a constant `i32`.
    fn constant_i32(&self, v: ValueId) -> Option<i32> {
        match self.constants.get(&v) {
            Some(Literal::I32(c)) => Some(*c),
            _ => None,
        }
    }

    fn named_barrier(&self, kind: BarrierKind, id: ValueId, threads: Option<ValueId>) -> Result {
        let what = match kind {
            BarrierKind::Sync => "barrier",
            BarrierKind::Arrive | BarrierKind::ArriveUnaligned => "barrier_arrive",
        };
        self.expect(id, DType::I32, "barrier id")?;
        if let Some(c) = self.constant_i32(id)
            && !(0..16).contains(&c)
        {
            return Err(
                self.err(format!("{what}: barrier id {c} is out of range (named barriers are numbered 0 to 15)"))
            );
        }
        let Some(threads) = threads else {
            if kind != BarrierKind::Sync {
                return Err(self.err(format!("{what} needs a thread count")));
            }
            return Ok(());
        };
        self.expect(threads, DType::I32, "barrier thread count")?;
        if let Some(c) = self.constant_i32(threads) {
            if c <= 0 || c % 32 != 0 {
                return Err(self.err(format!("{what}: a thread count of {c} is not a positive multiple of 32")));
            }
            let (limit, of) = match self.ir.block_dim {
                Some(d) => (d.iter().product::<u32>().next_multiple_of(32) as i32, "the block"),
                None => (1024, "any block"),
            };
            if c > limit {
                return Err(self.err(format!("{what}: {c} threads is more than {of} has ({limit})")));
            }
        }
        Ok(())
    }

    fn mbarrier(&self, buffer: BufferId, index: ValueId) -> Result {
        if self.ir.buffer(buffer).space != MemSpace::Shared {
            return Err(self.err("mbarriers live in shared memory"));
        }
        self.expect(index, DType::I32, "mbarrier index")
    }

    fn yields_match(&self, yields: &[ValueId], results: &[ValueId], what: &str) -> Result {
        if yields.len() != results.len() {
            return Err(self.err(format!("{what} yields {} values but {} are expected", yields.len(), results.len())));
        }
        for (&y, &r) in yields.iter().zip(results) {
            if self.ty(y) != self.ty(r) {
                return Err(self.err(format!("{what} yields {} where {} is expected", self.ty(y), self.ty(r))));
            }
        }
        Ok(())
    }
}

/// What an instruction needs from the compilation target: the oldest
/// architecture (`sm`, e.g. 90 for sm_90) and PTX ISA version (`ptx`, e.g.
/// 78 for 7.8) that have it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TargetRequirement {
    pub sm: u32,
    pub ptx: u32,
    /// The feature, for messages.
    pub what: &'static str,
}

/// The target `op` needs, from the PTX ISA's notes on each instruction;
/// `None` for instructions every supported target (sm_80 and newer, PTX
/// 7.0) has.
pub fn target_requirement(op: &Op) -> Option<TargetRequirement> {
    let need = |sm, ptx, what| Some(TargetRequirement { sm, ptx, what });
    let mut needs: Vec<Option<TargetRequirement>> = match op {
        Op::Mma { .. } => vec![need(80, 70, "mma.sync")],
        Op::CpAsync { .. } | Op::CpAsyncCommitGroup | Op::CpAsyncWaitGroup { .. } => vec![need(80, 70, "cp.async")],
        Op::MbarrierInit { .. } => vec![need(80, 70, "mbarrier.init")],
        Op::MbarrierArrive { count, expect_tx, peer, relaxed, scope, .. } => vec![
            need(80, 70, "mbarrier.arrive"),
            count.and(need(90, 78, "an mbarrier.arrive with an arrival count")),
            expect_tx.and(need(90, 80, "mbarrier.arrive.expect_tx")),
            peer.and(need(90, 80, "an mbarrier.arrive on a peer CTA")),
            (*scope == MemScopeKind::Cluster).then_some(()).and(need(90, 80, "a cluster-scope mbarrier.arrive")),
            relaxed.then_some(()).and(need(90, 86, "a relaxed mbarrier.arrive")),
        ],
        Op::MbarrierExpectTx { .. } => vec![need(90, 80, "mbarrier.expect_tx")],
        Op::MbarrierWait { kind: MbarrierWaitKind::Test, .. } => vec![need(80, 71, "mbarrier.test_wait.parity")],
        Op::MbarrierWait { kind: MbarrierWaitKind::Try, .. } => vec![need(90, 78, "mbarrier.try_wait")],
        Op::FenceMbarrierInit => vec![need(90, 80, "fence.mbarrier_init")],
        Op::CpAsyncBulk { .. } => vec![need(90, 80, "cp.async.bulk")],
        Op::Fence { scope: FenceScope::Cluster } => vec![need(90, 78, "fence.acq_rel.cluster")],
        Op::FloatInst { func: FloatFn::Tanh, .. } => vec![need(75, 70, "tanh.approx.f32")],
        Op::FloatInst { nan: true, .. } => vec![need(80, 70, "min and max with .NaN")],
        Op::Special {
            reg:
                SpecialReg::ClusterIdx(_)
                | SpecialReg::ClusterDim(_)
                | SpecialReg::BlockInClusterIdx(_)
                | SpecialReg::BlockInClusterDim(_),
            ..
        } => vec![need(90, 78, "the cluster special registers")],
        // CuTe DSL refuses `cutlass.range(..., vectorize=True)` below sm_100
        // (UNSUP_ARCH); sm_100 needs PTX 8.6.
        Op::For { opts: LoopOptions { vectorize: true, .. }, .. } => {
            vec![need(100, 86, "a vectorized loop (LoopOptions::vectorize, cutlass.range(..., vectorize=True))")]
        }
        _ => vec![],
    };
    let ptx = needs.iter().flatten().map(|r| r.ptx).max()?;
    needs.sort_by_key(|r| r.map(|r| std::cmp::Reverse(r.sm)));
    let first = needs.into_iter().flatten().next()?;
    Some(TargetRequirement { ptx, ..first })
}

/// Check that `ir` can be compiled for `arch`, and return the PTX ISA
/// version to compile it with: the architecture's own, or the newer one an
/// instruction needs.
pub fn check_target(ir: &KernelIr, arch: Arch) -> std::result::Result<u32, ValidationError> {
    fn walk(r: &Region, arch: Arch, ptx: &mut u32) -> std::result::Result<(), ValidationError> {
        for inst in &r.insts {
            if let Some(need) = target_requirement(&inst.op) {
                if need.sm > arch.sm() {
                    return Err(ValidationError {
                        message: format!(
                            "{} needs sm_{} or newer; the kernel is being compiled for {}",
                            need.what,
                            need.sm,
                            arch.chip()
                        ),
                        loc: Some(inst.loc),
                    });
                }
                *ptx = (*ptx).max(need.ptx);
            }
            for region in inst.op.regions() {
                walk(region, arch, ptx)?;
            }
        }
        Ok(())
    }
    let mut ptx = arch.min_ptx();
    walk(&ir.body, arch, &mut ptx)?;
    Ok(ptx)
}

/// Check that `ir` is well formed.
pub fn validate(ir: &KernelIr) -> std::result::Result<(), ValidationError> {
    let n = ir.value_types.len();
    let mut c = Checker { ir, visible: vec![false; n], defined: vec![false; n], loc: None, constants: HashMap::new() };
    let mut scope = Vec::new();
    for p in &ir.args {
        match p {
            Param::Buffer(b) => {
                if ir.buffer(*b).space != MemSpace::Global {
                    return Err(c.err("buffer parameters must be global memory"));
                }
            }
            Param::Scalar { value, .. } => c.define(*value, &mut scope)?,
        }
    }
    let smem = ir.shared_memory_bytes();
    if smem > MAX_SHARED_BYTES {
        return Err(c.err(format!(
            "kernel uses {smem} bytes of shared memory; no GPU gives a block more than {MAX_SHARED_BYTES}"
        )));
    }
    c.region(&ir.body, &[])
}
