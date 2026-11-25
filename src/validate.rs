//! IR verification: scoping (SSA dominance), types, and instruction
//! constraints. Every kernel is validated when it is built, so backends and
//! the simulator can rely on well-formed IR.

use std::collections::HashMap;
use std::fmt;

use crate::backend::Arch;

mod wgmma;
// External calls, device functions and multimem.
mod extern_calls;
use crate::ir::{
    AsmConstraint, AsmInput, AtomicKind, AtomicType, BarrierKind, BinaryOp, BufferId, BulkCompletion, BulkDirection,
    CacheEvictionPriority, CacheOperator, CarryFn, CarryOp, CmpPred, CvtFn, DType, FenceScope, FloatFn, FloatMode,
    HalfFn, HalfModifiers, HalfType, IntFn, KernelIr, L2PrefetchSize, LdMatrixForm, Literal, Loc, LoopOptions,
    MbarrierExtKind, MbarrierWaitKind, MemAttrs, MemScopeKind, MemSpace, MemoryOrder, MmaKind, Op, PackedOp, Param,
    PrefetchCacheLevel, ProxyKind, ReduxKind, Region, SharedMemoryLayout, SpecialReg, StMatrixForm, Tcgen05Inst,
    UnaryOp, ValueId, VoteKind, asm_constraints, asm_empty_constraint, asm_operand_refs, asm_register,
    carry_flag_breaker, float_inst_name, half_inst_name, int_inst_name, redux_inst_name,
};
pub use extern_calls::MULTIMEM_FP8;
// m7-tma-im2col and m7-tma-sm100-modes.
use crate::ir::TmaMode;
use crate::ir::{UmmaDataKind, UmmaKind};

/// Static shared memory available to one block. Kernels using more are
/// compiled to allocate it dynamically, at launch, as are kernels with a
/// dynamic region ([`get_dyn_smem`](crate::get_dyn_smem)).
pub const MAX_STATIC_SHARED_BYTES: usize = 48 * 1024;

/// The most shared memory any supported GPU gives one block, static and
/// dynamic together: sm_107's 327 KiB, the largest
/// [`Arch::smem_capacity_bytes`]. What a given target allows is usually
/// less ([`check_target`] checks the build-time buffers against it); the
/// launcher checks the device.
pub const MAX_SHARED_BYTES: usize = 327 * 1024;

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
    /// Where the kernel's shared buffers live, for their alignment.
    smem: SharedMemoryLayout,
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

    /// A buffer index: an `i32`, or an `i64` (a 64-bit offset).
    fn expect_index(&self, v: ValueId, what: &str) -> Result {
        if !matches!(self.ty(v), DType::I32 | DType::I64) {
            return Err(self.err(format!("{what} must be i32 or i64, but {v} is {}", self.ty(v))));
        }
        Ok(())
    }

    fn same(&self, values: &[ValueId], what: &str) -> Result<DType> {
        let Some(&first) = values.first() else {
            return Err(self.err(format!("{what} of no elements: it moves at least one")));
        };
        let t = self.ty(first);
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
        let mut carry = CarryFlag::Unset;
        for inst in &r.insts {
            self.loc = Some(inst.loc);
            self.inst(&inst.op)?;
            self.carry_flag(&inst.op, inst.loc, &mut carry)?;
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

    /// A vector access of `n` elements of `elem` to `buffer`: a power of two
    /// of at most 16 bytes, or 32 bytes for a plain access of 32- or 64-bit
    /// elements to global memory (PTX's `ld.global.v8.b32` and
    /// `.v4.b64`, sm_100, which LLVM splits in two for older targets).
    fn vector_width(&self, n: usize, elem: DType, buffer: BufferId, plain: bool) -> Result {
        let bytes = n * elem.bytes();
        let wide = plain && self.ir.buffer(buffer).space == MemSpace::Global && matches!(elem.bits(), 32 | 64);
        let max = if wide { 32 } else { 16 };
        if !n.is_power_of_two() || bytes > max {
            let note = if bytes == 32 && !wide {
                " (32-byte vectors are for plain accesses of 32- or 64-bit elements to global memory)"
            } else {
                ""
            };
            return Err(self.err(format!(
                "vector access of {n} x {elem} is not a power-of-two width of at most {max} bytes{note}"
            )));
        }
        Ok(())
    }

    /// Check an atomic (`red`: a reduction) of type `ty` on `value`: an
    /// operation PTX has on the type ([`AtomicType::supports`]), an operand
    /// of the type's width and kind (an Int32 for a packed pair), memory
    /// semantics the instruction takes, and global or shared memory aligned
    /// to the access.
    #[allow(clippy::too_many_arguments)]
    fn atomic(
        &self,
        red: bool,
        kind: AtomicKind,
        ty: AtomicType,
        buffer: BufferId,
        index: ValueId,
        value: ValueId,
        sem: MemoryOrder,
    ) -> Result {
        let inst = if red { "red" } else { "atom" };
        let t = self.ty(value);
        if !ty.supports(kind, red) {
            return Err(self.err(if ty.is_float() && matches!(kind, AtomicKind::Min | AtomicKind::Max) {
                format!(
                    "PTX has no {inst}.{} on a scalar float type: the float {inst} operations are add (use \
                     atomic_fmax or atomic_fmin for Float32)",
                    ty.op_suffix(kind)
                )
            } else {
                format!("PTX has no {inst}.{}", ty.op_suffix(kind))
            }));
        }
        let fits = if ty.is_packed() {
            t == DType::I32
        } else if let Some(half) = ty.half() {
            t == half
        } else if ty.is_float() {
            t.is_float() && t.bits() == ty.bits()
        } else {
            t.is_int() && t.bits() == ty.bits()
        };
        if !fits {
            return Err(self.err(format!("an {inst} of type {} cannot take a {t} operand", ty.ptx())));
        }
        let sems: &[MemoryOrder] = if red {
            &[MemoryOrder::Relaxed, MemoryOrder::Release]
        } else {
            &[MemoryOrder::Relaxed, MemoryOrder::Acquire, MemoryOrder::Release, MemoryOrder::AcqRel]
        };
        if !sems.contains(&sem) {
            let names: Vec<&str> = sems.iter().map(|s| s.name()).collect();
            return Err(self.err(format!(
                "{} cannot be {}: its sem is {}",
                if red { "a reduction (red)" } else { "an atomic" },
                sem.name(),
                names.join(", ")
            )));
        }
        if matches!(self.ir.buffer(buffer).space, MemSpace::Local | MemSpace::Raw(crate::tensor::AddressSpace::Rmem)) {
            return Err(self.err("atomics need global or shared memory"));
        }
        self.expect_index(index, "atomic index")?;
        self.base_alignment(buffer, ty.bits() / 8, if red { "a reduction" } else { "an atomic" })
    }

    /// A `vprintf` call: a supported conversion for each operand, each
    /// fitting its operand's type (as C's variadic promotion leaves it), and
    /// at most [`MAX_PRINTF_ARGS`](crate::MAX_PRINTF_ARGS) operands.
    fn printf(&self, format: &str, args: &[ValueId]) -> Result {
        let pieces = crate::print::parse_format(format, false).map_err(|e| self.err(format!("printf: {e}")))?;
        let convs = crate::print::operand_conversions(&pieces);
        if convs.len() != args.len() {
            return Err(self.err(format!(
                "printf: the format {format:?} has {} conversion(s) but {} operand(s)",
                convs.len(),
                args.len()
            )));
        }
        if args.len() > crate::MAX_PRINTF_ARGS {
            return Err(self.err(format!(
                "printf: {} operands, but vprintf takes at most {}",
                args.len(),
                crate::MAX_PRINTF_ARGS
            )));
        }
        for (c, &a) in convs.iter().zip(args) {
            let t = self.ty(a);
            if t.bits() == 128 || t.is_word_packed() {
                continue; // refused with the other 128-bit and packed uses
            }
            if matches!(t, DType::I8 | DType::I16 | DType::U8 | DType::U16 | DType::I4 | DType::Bool)
                || (t.is_float() && !matches!(t, DType::F32 | DType::F64))
            {
                return Err(self.err(format!(
                    "printf: the operand {a} is a {}, which vprintf does not take: C's variadic promotion widens it \
                     (printf does)",
                    t.cute_name()
                )));
            }
            if !c.accepts(t) {
                return Err(self.err(format!("printf: {}", c.mismatch(t))));
            }
        }
        Ok(())
    }

    /// The operations 128-bit integers and Int4 values take part in.
    /// Int128 and Uint128 have arithmetic, comparisons, selects, integer
    /// conversions, loads and stores and control flow, but no printf
    /// conversion, no inline-assembly register class, no shuffle (CuTe DSL
    /// shuffles 64 bits at most), no atomic and no packing. Int4 values live
    /// in registers only until sub-byte tensors, with no bit counts.
    fn wide_and_int4(&self, op: &Op) -> Result {
        let values: Vec<ValueId> = op.operands().into_iter().chain(op.results()).collect();
        let wide = values.iter().any(|&v| self.ty(v).bits() == 128);
        let packed = values.iter().map(|&v| self.ty(v)).find(|t| t.is_word_packed());
        let int4 = packed.is_some();
        let structural = matches!(
            op,
            Op::Const { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Cmp { .. }
                | Op::Select { .. }
                | Op::Cast { .. }
                | Op::Bitcast { .. }
                | Op::For { .. }
                | Op::If { .. }
                | Op::While { .. }
        );
        if wide {
            let allowed = structural || matches!(op, Op::Load { .. } | Op::Store { .. });
            if !allowed {
                return Err(self.err(format!(
                    "{} cannot take a 128-bit value (Int128 or Uint128): they have arithmetic, comparisons, \
                     conversions, loads and stores, but no printf conversion, inline-assembly register class, \
                     shuffle, atomic or packing",
                    op.name()
                )));
            }
            if let Op::Binary { op: BinaryOp::MulHi, .. } = op {
                return Err(self.err("mul_hi takes Int32, Uint32, Int64 or Uint64 operands, not 128-bit ones"));
            }
            if let Op::Cast { result, x } = op
                && (self.ty(*result).is_float() || self.ty(*x).is_float())
            {
                return Err(self.err(format!(
                    "cannot convert {} to {}: LLVM has no NVPTX lowering of conversions between 128-bit integers and \
                     floats (convert through Int64)",
                    self.ty(*x).cute_name(),
                    self.ty(*result).cute_name()
                )));
            }
        }
        if int4 {
            let name = packed.map_or("", |t| t.cute_name());
            if matches!(op, Op::Load { .. } | Op::Store { .. }) {
                return Err(self.err(format!(
                    "{name} values are not loaded or stored one at a time: sub-byte tensors move whole 32-bit words \
                     (packed, element 0 in the low bits)"
                )));
            }
            let bad_op = match op {
                Op::Unary { op: u, .. } => matches!(u, UnaryOp::Popc | UnaryOp::Clz | UnaryOp::Ctz | UnaryOp::Brev),
                Op::Binary { op: b, .. } => {
                    matches!(b, BinaryOp::MulHi | BinaryOp::DivU | BinaryOp::RemU | BinaryOp::ShrU)
                }
                // Sub-byte memory packs and unpacks its words.
                Op::Pack { .. } | Op::Unpack { .. } => false,
                _ => !structural,
            };
            if bad_op {
                let what = if name == "Int4" {
                    "Int4 has arithmetic, comparisons, conversions, control flow and packing"
                } else {
                    "the sub-byte and packed float types are packed, unpacked, selected and converted with cvt helpers"
                };
                let article = if name.starts_with('I') { "an" } else { "a" };
                return Err(self.err(format!("{} cannot take {article} {name} value: {what}", op.name())));
            }
        }
        Ok(())
    }

    /// An access to another CTA's shared memory:
    /// the buffer is shared memory and the peer an Int32 rank.
    fn dsmem_peer(&self, buffer: BufferId, peer: Option<ValueId>, what: &str) -> Result {
        let Some(peer) = peer else { return Ok(()) };
        let d = self.ir.buffer(buffer);
        if d.space != MemSpace::Shared {
            return Err(self.err(format!(
                "{what} to another CTA's shared memory (distributed shared memory) addresses `{}`, which is not in \
                 shared memory",
                d.name
            )));
        }
        self.expect(peer, DType::I32, "the peer CTA rank of a distributed-shared-memory access")
    }

    /// Refuse `what` (a store, an atomic) to a buffer the device cannot write: a
    /// constant-memory global, a read-only module global or a by-value parameter.
    fn writable(&self, buffer: BufferId, what: &str) -> Result {
        let d = self.ir.buffer(buffer);
        let why = if d.space == MemSpace::Constant {
            Some("a global in constant memory, which the device cannot write (only the host can, between launches)")
        } else if self.ir.globals.iter().any(|g| g.buffer == buffer && g.readonly) {
            Some("a read-only module global")
        } else if self.ir.args.iter().any(|p| matches!(p, Param::ByValue { buffer: b, .. } if *b == buffer)) {
            Some("a by-value parameter, which is read-only")
        } else {
            None
        };
        match why {
            Some(why) => Err(self.err(format!("{what} to `{}`, {why}", d.name))),
            None => Ok(()),
        }
    }

    /// An access of `bytes` bytes to `buffer`. The builder proves its offset
    /// a multiple of `bytes` from the start of the buffer (and the simulator
    /// checks it), so it is aligned, as PTX requires every access to be, if
    /// the buffer starts at a multiple of that much
    /// ([`Self::start_alignment`]).
    fn base_alignment(&self, buffer: BufferId, bytes: usize, what: &str) -> Result {
        let b = self.ir.buffer(buffer);
        let align = self.start_alignment(buffer);
        if bytes > align {
            let hint = if b.space == MemSpace::Shared {
                format!(" (allocate it with alloc_smem(count, {bytes}) or more)")
            } else {
                String::new()
            };
            let aligned = match self.window_start(buffer) {
                Some(start) => {
                    format!("which the dynamic shared-memory window places at byte {start}, only {align}-byte aligned")
                }
                None => format!("which is only declared {align}-byte aligned"),
            };
            return Err(self.err(format!(
                "{what} of {bytes} bytes in `{}`, {aligned}: PTX requires each access to be aligned to its size{hint}",
                b.name
            )));
        }
        Ok(())
    }

    /// Where `buffer` starts in the dynamic shared-memory window, if it is a
    /// shared buffer that lives there (in a kernel with a dynamic region, or
    /// with more static shared memory than fits in 48 KB; see
    /// [`SharedMemoryLayout`]).
    fn window_start(&self, buffer: BufferId) -> Option<usize> {
        self.smem.starts[buffer.0 as usize].filter(|_| self.smem.in_window)
    }

    /// What `buffer`'s start is aligned to: its declared alignment, except
    /// in the dynamic shared-memory window, which places each shared buffer
    /// at a fixed byte (a multiple of its alignment, the dynamic region's
    /// being at least 16 and the largest any `get_dyn_smem` asks for) of a
    /// window aligned to `window_align`: there it is the largest power of two
    /// that divides that byte, up to `window_align`.
    fn start_alignment(&self, buffer: BufferId) -> usize {
        match self.window_start(buffer) {
            Some(0) => self.smem.window_align,
            Some(start) => (1usize << start.trailing_zeros()).min(self.smem.window_align),
            None => self.ir.buffer(buffer).align,
        }
    }

    fn inst(&mut self, op: &Op) -> Result {
        for v in op.operands() {
            self.use_(v)?;
        }
        self.wide_and_int4(op)?;
        self.multicast_access(op)?;
        match op {
            Op::Call { callee, args, results, body } => self.call(*callee, args, results, body.as_ref())?,
            Op::Multimem { op, buffer, index, values, results, .. } => {
                self.expect_index(*index, "a multimem address's index")?;
                self.multimem(op, *buffer, values, results)?;
            }
            Op::Const { result, value } => {
                self.expect(*result, value.dtype(), "constant")?;
                self.constants.insert(*result, *value);
            }
            Op::Unary { result, x, .. } | Op::Binary { result, lhs: x, .. } | Op::Fma { result, a: x, .. }
                if storage_only(self.ty(*x)) || storage_only(self.ty(*result)) =>
            {
                return Err(self.err(storage_only_message(self.ty(*x))));
            }
            Op::Cmp { lhs, .. } if storage_only(self.ty(*lhs)) => {
                return Err(self.err(storage_only_message(self.ty(*lhs))));
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
                    // mul.hi exists for 16-bit types too, but CuTe DSL's
                    // mul_hi takes 32- and 64-bit operands only.
                    BinaryOp::MulHi => t.is_int() && t.bits() >= 32,
                    BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => t.is_int() || t == DType::Bool,
                };
                if !ok && *op == BinaryOp::MulHi {
                    return Err(self.err(format!("mul_hi takes Int32, Uint32, Int64 or Uint64 operands, not {t}")));
                }
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
            Op::FloatInst { result, func, args, mode, ftz, nan, xorsign_abs } => {
                self.float_inst(*result, *func, args, *mode, *ftz, *nan, *xorsign_abs)?
            }
            Op::IntInst { result, func, args } => self.int_inst(*result, *func, args)?,
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
                let (from, to) = (self.ty(*x), self.ty(*result));
                if to == DType::Bool {
                    return Err(self.err(format!("cannot cast {from} to i1; compare instead")));
                }
                if to.is_packed_view() || from.is_packed_view() {
                    return Err(self.err(format!(
                        "cannot convert {} to {}: a packed view holds several values; recast_tensor it to its lanes \
                         ({})",
                        from.cute_name(),
                        to.cute_name(),
                        to.packed_lanes().or(from.packed_lanes()).map_or("", |(lane, _)| lane.cute_name())
                    )));
                }
                if matches!(to, DType::F4E2M1FN | DType::F6E2M3FN | DType::F6E3M2FN) {
                    return Err(self.err(format!(
                        "cannot convert {} to {}: CuTe DSL refuses it as a to() destination; convert pairs of \
                         Float32 values with {} (sm_100a, sm_101a, sm_120a)",
                        from.cute_name(),
                        to.cute_name(),
                        if to == DType::F4E2M1FN { "cvt_f32x2_to_f4e2m1x2" } else { "cvt_f32x2_to_f6x2" }
                    )));
                }
                if matches!(to, DType::F8E4M3 | DType::F8E4M3B11FNUZ) {
                    return Err(self.err(format!(
                        "cannot convert {} to {}: CuTe DSL refuses it as a conversion destination (PTX has no cvt \
                         into it); it is a storage type, which kernels read and convert from",
                        from.cute_name(),
                        to.cute_name()
                    )));
                }
                if to == DType::F8E5M3FNU {
                    return Err(self.err(format!(
                        "cannot convert {} to FloatNV8E5M3FNU: its cvt (.ue5m3x2) needs PTX ISA 9.4, beyond LLVM 21 \
                         and ptxas 12.8; kernels can read it and convert from it",
                        from.cute_name()
                    )));
                }
                if storage_only(from) || storage_only(to) {
                    return Err(self.err(format!(
                        "a cast between {from} and {to} has no single instruction: convert with Value::to"
                    )));
                }
            }
            Op::Bitcast { result, x } => {
                if self.ty(*result).bits() != self.ty(*x).bits() {
                    return Err(self.err(format!("cannot bitcast {} to {}", self.ty(*x), self.ty(*result))));
                }
            }
            Op::Special { result, reg } => self.expect(*result, reg.dtype(), "special register")?,
            Op::Load { results, buffer, index, attrs, pred, pass_thru } => {
                self.expect_index(*index, "load index")?;
                let _ = self.ir.buffer(*buffer);
                let t = self.same(results, "load")?;
                if let Some(p) = pred {
                    self.expect(*p, DType::Bool, "a masked load's predicate")?;
                }
                if !pass_thru.is_empty() {
                    if pred.is_none() {
                        return Err(self.err("a load has pass-through values but no predicate"));
                    }
                    if pass_thru.len() != results.len() {
                        return Err(self.err(format!(
                            "a masked load of {} elements has {} pass-through values",
                            results.len(),
                            pass_thru.len()
                        )));
                    }
                    for &v in pass_thru {
                        self.expect(v, t, "a masked load's pass-through value")?;
                    }
                }
                self.vector_width(results.len(), t, *buffer, attrs.is_plain())?;
                self.base_alignment(*buffer, results.len() * t.bytes(), "a load")?;
                self.mem_attrs(attrs, Access::Load, *buffer, results.len(), t)?;
                self.dsmem_peer(*buffer, attrs.peer, "a load")?;
            }
            Op::Store { values, buffer, index, attrs, pred } => {
                self.writable(*buffer, "a store")?;
                self.expect_index(*index, "store index")?;
                let _ = self.ir.buffer(*buffer);
                let t = self.same(values, "store")?;
                if let Some(p) = pred {
                    self.expect(*p, DType::Bool, "a masked store's predicate")?;
                }
                self.vector_width(values.len(), t, *buffer, attrs.is_plain())?;
                self.base_alignment(*buffer, values.len() * t.bytes(), "a store")?;
                self.mem_attrs(attrs, Access::Store, *buffer, values.len(), t)?;
                self.dsmem_peer(*buffer, attrs.peer, "a store")?;
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
                self.base_alignment(*src, bytes, "cp.async")?;
                self.base_alignment(*dst, bytes, "cp.async")?;
                self.expect_index(*src_index, "cp.async source index")?;
                self.expect_index(*dst_index, "cp.async destination index")?;
                if let Some(p) = pred {
                    self.expect(*p, DType::Bool, "cp.async predicate")?;
                }
            }
            Op::CpAsyncCommitGroup
            | Op::CpAsyncWaitGroup { .. }
            | Op::CpAsyncBulkCommitGroup
            | Op::CpAsyncBulkWaitGroup { .. }
            | Op::FenceProxy { .. }
            | Op::Barrier
            | Op::Fence { .. } => {}
            Op::Printf { format, args } => self.printf(format, args)?,
            Op::Assert { cond, .. } => self.expect(*cond, DType::Bool, "an assertion's condition")?,
            Op::InlinePtx { template, inputs, outputs, constraints, predicated, .. } => {
                self.inline_asm(template, inputs, outputs, constraints, *predicated)?
            }
            Op::NamedBarrier { kind, id, threads } => self.named_barrier(*kind, *id, *threads)?,
            Op::WarpBarrier { mask } => self.expect(*mask, DType::U32, "lane mask")?,
            Op::Shuffle { result, value, offset, mask, clamp, .. } => {
                if self.ty(*value).bits() != 32 {
                    return Err(self.err(format!("shuffles move 32-bit values, not {}", self.ty(*value))));
                }
                self.expect(*result, self.ty(*value), "shuffle result")?;
                self.expect(*offset, DType::I32, "shuffle offset")?;
                self.expect(*mask, DType::U32, "lane mask")?;
                self.expect(*clamp, DType::I32, "shuffle mask_and_clamp")?;
            }
            Op::Vote { result, kind, pred, mask } => {
                self.expect(*pred, DType::Bool, "vote predicate")?;
                self.expect(*mask, DType::U32, "lane mask")?;
                let want = if *kind == VoteKind::Ballot { DType::U32 } else { DType::Bool };
                self.expect(*result, want, "vote result")?;
            }
            Op::Match { result, value, mask, .. } => {
                let t = self.ty(*value);
                if !matches!(t, DType::I32 | DType::U32 | DType::I64 | DType::U64) {
                    return Err(self.err(format!("match_sync compares Int32, Uint32, Int64 or Uint64 values, not {t}")));
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
            Op::MbarrierWait { result, buffer, index, parity, state, time_limit, .. } => {
                self.mbarrier(*buffer, *index)?;
                if *state {
                    self.expect(*parity, DType::I64, "mbarrier state token")?;
                } else {
                    self.expect(*parity, DType::I32, "phase parity")?;
                }
                if let Some(t) = time_limit {
                    self.expect(*t, DType::I32, "mbarrier wait time limit")?;
                }
                self.expect(*result, DType::Bool, "mbarrier wait result")?;
            }
            // m11-prims-nvvm-layer.
            Op::BarrierRed { result, kind, pred, id, threads } => {
                self.expect(*pred, DType::Bool, "barrier_cta_red predicate")?;
                let want = if *kind == crate::ir::BarrierRedKind::Popc { DType::U32 } else { DType::Bool };
                self.expect(*result, want, "barrier_cta_red result")?;
                self.named_barrier(BarrierKind::Sync, *id, *threads)?;
            }
            Op::MbarrierExt { kind, result, buffer, index, value } => {
                self.mbarrier(*buffer, *index)?;
                if kind.has_value() != value.is_some() || kind.has_state() != result.is_some() {
                    return Err(self.err(format!("{}: malformed operands", kind.ptx())));
                }
                if let Some(v) = value {
                    self.expect(*v, DType::I32, "mbarrier count or transaction bytes")?;
                }
                if let Some(r) = result {
                    self.expect(*r, DType::I64, "mbarrier state token")?;
                }
            }
            Op::PrimMarker { marker } => match marker {
                crate::ir::PrimMarker::Nanosleep(t) => self.expect(*t, DType::I32, "nanosleep duration")?,
                crate::ir::PrimMarker::Comment(text) if text.contains(['\n', '\r']) => {
                    return Err(self.err("ptx_add_comment: a comment is one line"));
                }
                _ => {}
            },
            Op::Exit => {}
            Op::StBulk { buffer, index, size, .. } => {
                if self.ir.buffer(*buffer).space != MemSpace::Shared {
                    return Err(self.err("st.bulk writes shared memory (st.bulk.shared::cta)"));
                }
                self.writable(*buffer, "st.bulk")?;
                self.base_alignment(*buffer, 16, "st.bulk")?;
                self.expect_index(*index, "st.bulk index")?;
                self.expect(*size, DType::I64, "st.bulk size")?;
                if let Some(Literal::I64(n)) = self.constants.get(size)
                    && (*n <= 0 || n % 16 != 0)
                {
                    return Err(self.err(format!("st.bulk: a size of {n} bytes is not a positive multiple of 16")));
                }
            }
            Op::Tcgen05Shift { taddr, .. } => self.expect(*taddr, DType::I32, "tcgen05.shift TMEM address")?,
            Op::FenceMbarrierInit | Op::GridDepControl { .. } | Op::ClusterArrive { .. } | Op::ClusterWait { .. } => {}
            Op::CpAsyncBulk {
                src,
                src_index,
                dst,
                dst_index,
                bytes,
                direction,
                completion,
                cache_policy,
                mcast_mask,
                peer,
                byte_mask,
                ..
            } => {
                self.writable(*dst, "a bulk copy")?;
                if let Some(m) = byte_mask {
                    if *direction != BulkDirection::S2G {
                        return Err(self.err("only a bulk copy from shared to global memory takes a byte mask"));
                    }
                    self.expect(*m, DType::I16, "byte mask")?;
                }
                let (from, to) = match direction {
                    BulkDirection::G2S => (MemSpace::Global, MemSpace::Shared),
                    BulkDirection::S2G => (MemSpace::Shared, MemSpace::Global),
                    BulkDirection::S2S => (MemSpace::Shared, MemSpace::Shared),
                };
                if let Some(p) = cache_policy {
                    if *direction == BulkDirection::S2S {
                        return Err(self.err("cp.async.bulk.shared::cluster.shared::cta takes no L2 cache hint"));
                    }
                    self.expect(*p, DType::I64, "L2 cache policy")?;
                }
                if let Some(m) = mcast_mask {
                    if *direction != BulkDirection::G2S {
                        return Err(self.err("only a bulk copy from global to shared memory multicasts"));
                    }
                    self.expect(*m, DType::I16, "multicast CTA mask")?;
                }
                if peer.is_some() != (*direction == BulkDirection::S2S) {
                    return Err(self.err("a shared-to-shared bulk copy, and only one, addresses a peer CTA"));
                }
                if let Some(p) = peer {
                    self.expect(*p, DType::I32, "peer CTA rank")?;
                }
                if self.ir.buffer(*src).space != from || self.ir.buffer(*dst).space != to {
                    return Err(self.err(format!("cp.async.bulk {direction:?} copies from {from:?} to {to:?} memory")));
                }
                self.expect_index(*src_index, "cp.async.bulk source index")?;
                self.expect_index(*dst_index, "cp.async.bulk destination index")?;
                self.expect(*bytes, DType::I32, "cp.async.bulk size")?;
                if let Some(n) = self.constant_i32(*bytes)
                    && (n <= 0 || n % 16 != 0)
                {
                    return Err(self
                        .err(format!("cp.async.bulk of {n} bytes: bulk copies move one or more whole 16-byte units")));
                }
                self.base_alignment(*src, 16, "cp.async.bulk")?;
                self.base_alignment(*dst, 16, "cp.async.bulk")?;
                match (direction, completion) {
                    (BulkDirection::G2S | BulkDirection::S2S, BulkCompletion::Mbarrier { buffer, index }) => {
                        self.mbarrier(*buffer, *index)?
                    }
                    (BulkDirection::S2G, BulkCompletion::BulkGroup) => {}
                    (d, _) => {
                        return Err(self.err(format!(
                            "cp.async.bulk {d:?} reports its completion through {}",
                            if *d == BulkDirection::S2G { "a bulk async-group" } else { "an mbarrier" }
                        )));
                    }
                }
            }
            Op::TmaLoad {
                dst,
                dst_index,
                desc,
                coords,
                mbar,
                mbar_index,
                cache_policy,
                mcast_mask,
                mode,
                overrides,
                ..
            } => {
                self.writable(*dst, "a TMA load")?;
                if self.ir.buffer(*dst).space != MemSpace::Shared {
                    return Err(self.err("a TMA load writes shared memory"));
                }
                self.expect_index(*dst_index, "TMA load destination index")?;
                self.expect(*desc, DType::I64, "tensor-map address")?;
                if coords.is_empty() || coords.len() > crate::tma::MAX_RANK {
                    return Err(self.err(format!("a TMA load takes 1 to 5 coordinates, not {}", coords.len())));
                }
                for &c in coords {
                    self.expect(c, DType::I32, "TMA coordinate")?;
                }
                if matches!(mode, TmaMode::Im2colNoOffs | TmaMode::Scatter4) {
                    return Err(self.err(format!("a TMA load does not take the {} mode (a store's)", mode.ptx())));
                }
                self.tma_mode(mode, coords.len(), "a TMA load")?;
                if let Some(o) = overrides {
                    self.tma_override(o, mode, coords.len(), "a TMA load")?;
                }
                self.mbarrier(*mbar, *mbar_index)?;
                if let Some(p) = cache_policy {
                    self.expect(*p, DType::I64, "L2 cache policy")?;
                }
                self.base_alignment(*dst, 128, "a TMA load's destination")?;
                if let Some(m) = mcast_mask {
                    self.expect(*m, DType::I16, "multicast CTA mask")?;
                }
            }
            // ---- m7-tma-store and m7-cluster-copies ----
            Op::TmaStore { src, src_index, elem, desc, coords, reduce, cache_policy, mode, overrides, .. } => {
                if self.ir.buffer(*src).space != MemSpace::Shared {
                    return Err(self.err("a TMA store reads shared memory"));
                }
                self.expect_index(*src_index, "TMA store source index")?;
                self.expect(*desc, DType::I64, "tensor-map address")?;
                self.tma_coords(coords, "a TMA store")?;
                if matches!(mode, TmaMode::Im2col { .. } | TmaMode::Gather4) {
                    return Err(self.err(format!("a TMA store does not take the {} mode (a load's)", mode.ptx())));
                }
                if reduce.is_some() && *mode == TmaMode::Scatter4 {
                    return Err(self.err("cp.reduce.async.bulk.tensor takes the .tile and .im2col_no_offs modes only"));
                }
                self.tma_mode(mode, coords.len(), "a TMA store")?;
                if let Some(o) = overrides {
                    self.tma_override(o, mode, coords.len(), "a TMA store")?;
                }
                if let Some(p) = cache_policy {
                    self.expect(*p, DType::I64, "L2 cache policy")?;
                }
                if let Some(kind) = reduce {
                    crate::nvgpu::tma_copy::tma_reduce_type_rule(*kind, *elem).map_err(|m| self.err(m))?;
                }
                self.base_alignment(*src, 128, "a TMA store's source")?;
            }
            Op::TmaPrefetch { desc, coords, mode } => {
                self.expect(*desc, DType::I64, "tensor-map address")?;
                self.tma_coords(coords, "a TMA prefetch")?;
                if matches!(mode, TmaMode::Im2colNoOffs | TmaMode::Scatter4) {
                    return Err(self.err(format!("a TMA prefetch does not take the {} mode", mode.ptx())));
                }
                self.tma_mode(mode, coords.len(), "a TMA prefetch")?;
            }
            Op::PrefetchTensormap { desc } => self.expect(*desc, DType::I64, "tensor-map address")?,
            Op::TensormapReplace { addr, space, field, value } => {
                let at = if *space == crate::tensor::AddressSpace::Smem { DType::I32 } else { DType::I64 };
                self.expect(*addr, at, "tensor-map address")?;
                self.expect(*value, field.value_type(), "tensormap.replace's new value")?;
            }
            Op::FenceTensormapRelease => {}
            Op::FenceTensormapAcquire { addr } => self.expect(*addr, DType::I64, "tensor-map address")?,
            Op::TensormapCpFenceproxy { dst, src } => {
                self.expect(*dst, DType::I64, "tensormap.cp_fenceproxy's global address")?;
                self.expect(*src, DType::I32, "tensormap.cp_fenceproxy's shared address")?;
            }
            Op::BulkPrefetch { buffer, index, bytes, .. } => {
                if self.ir.buffer(*buffer).space != MemSpace::Global {
                    return Err(self.err("cp.async.bulk.prefetch.L2 reads global memory"));
                }
                self.expect_index(*index, "bulk prefetch index")?;
                self.expect(*bytes, DType::I32, "bulk prefetch size")?;
                if let Some(n) = self.constant_i32(*bytes)
                    && (n <= 0 || n % 16 != 0)
                {
                    return Err(self.err(format!("cp.async.bulk.prefetch of {n} bytes: a multiple of 16 bytes")));
                }
            }
            Op::StAsync { buffer, index, elem, values, mbar, mbar_index, peer } => {
                if self.ir.buffer(*buffer).space != MemSpace::Shared {
                    return Err(self.err("st.async writes shared memory"));
                }
                self.expect_index(*index, "st.async index")?;
                if ![1, 2, 4].contains(&values.len()) || elem.bits() != 32 {
                    return Err(self.err("st.async stores one, two or four 32-bit values"));
                }
                for &v in values {
                    self.expect(v, DType::I32, "st.async value")?;
                }
                self.mbarrier(*mbar, *mbar_index)?;
                self.expect(*peer, DType::I32, "peer CTA rank")?;
                self.base_alignment(*buffer, 4 * values.len(), "st.async")?;
            }
            Op::Atomic { result, kind, ty, buffer, index, value, compare, sem, scope: _, peer } => {
                self.writable(*buffer, "an atomic")?;
                self.dsmem_peer(*buffer, *peer, "an atomic")?;
                self.atomic(false, *kind, *ty, *buffer, *index, *value, *sem)?;
                self.expect(*result, self.ty(*value), "atomic result")?;
                if compare.is_some() != (*kind == AtomicKind::CompareExchange) {
                    return Err(self.err("only compare-and-exchange takes a comparand"));
                }
                if let Some(c) = compare {
                    self.expect(*c, self.ty(*value), "atomic comparand")?;
                }
            }
            Op::Red { kind, ty, buffer, index, value, sem, scope: _, peer } => {
                self.writable(*buffer, "a reduction")?;
                self.dsmem_peer(*buffer, *peer, "a reduction")?;
                self.atomic(true, *kind, *ty, *buffer, *index, *value, *sem)?;
            }
            Op::StMatrix { values, buffer, index, num, trans, form } => {
                let b = self.ir.buffer(*buffer);
                if b.space != MemSpace::Shared {
                    return Err(self.err("stmatrix writes shared memory"));
                }
                if ![1, 2, 4].contains(num) || values.len() != *num as usize {
                    return Err(self.err(format!("stmatrix.{}.x{num} takes {num} registers", form.shape())));
                }
                let elem_bits = if *form == StMatrixForm::M8n8B16 { 16 } else { 8 };
                // A struct's shared memory is one byte buffer, which its members
                // (here 16-bit ones) view.
                // (32-bit elements: the f32 tile of a Blackwell epilogue, one per register.)
                let wide_ok = *form == StMatrixForm::M8n8B16 && b.elem.bits() == 32;
                if b.elem.bits() != elem_bits && !wide_ok && b.fields.is_empty() {
                    return Err(self.err(format!(
                        "stmatrix.{} stores {elem_bits}-bit elements, not the {} of `{}`",
                        form.shape(),
                        b.elem,
                        b.name
                    )));
                }
                if *form == StMatrixForm::M16n8B8 && !trans {
                    return Err(self.err("stmatrix.m16n8 must be transposed (.trans)"));
                }
                self.expect_index(*index, "stmatrix index")?;
                let t = self.same(values, "stmatrix")?;
                if t.bits() != 32 || !t.is_int() {
                    return Err(self.err(format!("stmatrix takes 32-bit registers, not {t}")));
                }
                // Each thread gives the address of a 16-byte row.
                self.base_alignment(*buffer, 16, "stmatrix")?;
            }
            Op::SetMaxNReg { count, .. } => {
                if !(24..=256).contains(count) || count % 8 != 0 {
                    return Err(self.err(format!(
                        "setmaxnreg: a register count of {count} is not a multiple of 8 from 24 to 256"
                    )));
                }
                if self.ir.block_dim.is_none() && self.ir.max_threads.is_none() {
                    return Err(self.err(
                        "setmaxnreg takes effect only in a kernel with a launch bound (.reqntid or .maxntid): \
                         declare the block with Kernel::builder(..).block_dim or KernelOptions::max_threads",
                    ));
                }
            }
            Op::LdMatrix { results, buffer, index, num, trans, form } => {
                let b = self.ir.buffer(*buffer);
                if b.space != MemSpace::Shared {
                    return Err(self.err("ldmatrix reads shared memory"));
                }
                let nums: &[u32] = if matches!(form, LdMatrixForm::M16n16B8 { .. }) { &[1, 2] } else { &[1, 2, 4] };
                let want = form.results_per_matrix() * num;
                if !nums.contains(num) || results.len() != want as usize {
                    return Err(self.err(format!("ldmatrix.{}.x{num} yields {want} values", form.shape())));
                }
                if !matches!(form.unpack(), None | Some(4 | 6)) {
                    return Err(self.err("ldmatrix unpacks 4- or 6-bit elements"));
                }
                match form {
                    LdMatrixForm::M16n16B8 { .. } if !trans => {
                        return Err(self.err("ldmatrix.m16n16 must be transposed (.trans)"));
                    }
                    LdMatrixForm::M8n16B8 { .. } if *trans => {
                        return Err(self.err("ldmatrix.m8n16 cannot be transposed"));
                    }
                    _ => {}
                }
                self.expect_index(*index, "ldmatrix index")?;
                let t = self.same(results, "ldmatrix")?;
                let bits = if *form == LdMatrixForm::M8n8B16 { 16 } else { 32 };
                if t.bits() != bits {
                    return Err(self.err(format!("ldmatrix.{} yields {bits}-bit values, not {t}", form.shape())));
                }
                // Each thread gives the address of a 16-byte row.
                self.base_alignment(*buffer, 16, "ldmatrix")?;
            }
            Op::Mma { kind, d, a, b, c, e } => {
                match (kind.is_sparse(), e) {
                    (true, Some(e)) => {
                        let t = self.ir.dtype(*e);
                        if t.bits() != 32 || !t.is_int() {
                            return Err(self.err(format!("{kind:?} takes 32-bit integer metadata, not {t}")));
                        }
                    }
                    (true, None) => return Err(self.err(format!("{kind:?} is sparse and needs a metadata operand"))),
                    (false, Some(_)) => return Err(self.err(format!("{kind:?} is dense and takes no metadata"))),
                    (false, None) => {}
                }
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
            // m11-prims-mma-forms.
            Op::MmaSpBlockScaled { kind, d, a, b, c, e, sfa, sfb } => {
                if a.len() != 4 || b.len() != 4 || c.len() != 4 || d.len() != 4 {
                    return Err(self.err(format!("{} takes 4 A and 4 B registers and 4 accumulators", kind.ptx())));
                }
                if let Some(why) = crate::prims::mma::sp_block_scale_error(*kind) {
                    return Err(self.err(format!("{}: {why}", kind.ptx())));
                }
                for &v in a.iter().chain(b).chain([e, sfa, sfb]) {
                    let t = self.ir.dtype(v);
                    if t.bits() != 32 || !t.is_int() {
                        return Err(self.err(format!("{} takes 32-bit integer operand registers, not {t}", kind.ptx())));
                    }
                }
                for &v in c.iter().chain(d) {
                    self.expect(v, DType::F32, "sparse block-scaled mma C/D")?;
                }
            }
            // m10-sm120-blockscaled.
            Op::MmaBlockScaled { kind, d, a, b, c, sfa, sfb } => {
                let (ra, rb) = kind.operand_registers();
                if a.len() != ra || b.len() != rb || c.len() != 4 || d.len() != 4 {
                    return Err(
                        self.err(format!("{} takes {ra} A and {rb} B registers and 4 accumulators", kind.ptx()))
                    );
                }
                for &v in a.iter().chain(b).chain([sfa, sfb]) {
                    let t = self.ir.dtype(v);
                    if t.bits() != 32 || !t.is_int() {
                        return Err(self.err(format!("{} takes 32-bit integer operand registers, not {t}", kind.ptx())));
                    }
                }
                for &v in c.iter().chain(d) {
                    self.expect(v, DType::F32, "block-scaled mma C/D")?;
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
            Op::Redux { result, kind, value, mask, abs, nan } => {
                self.redux(*result, *kind, *value, *mask, *abs, *nan)?
            }
            Op::CarryInst { result, func, args } => self.carry_inst(*result, *func, args)?,
            Op::MlirOp { text, operands, .. } => {
                crate::arch::inline::check_mlir_text(text, operands.len())
                    .map_err(|e| self.err(format!("hand-written MLIR operation: {e}")))?;
                for operand in operands {
                    if let AsmInput::Address { buffer, index, .. } = operand {
                        self.expect_index(*index, "pointer operand index")?;
                        let _ = self.ir.buffer(*buffer);
                    }
                }
            }
            Op::CpAsyncMbarrierArrive { buffer, index, .. } => self.mbarrier(*buffer, *index)?,
            Op::Prefetch { buffer, index, level, evict, uniform, pred, .. } => {
                self.expect_index(*index, "prefetch index")?;
                if let Some(p) = pred {
                    self.expect(*p, DType::Bool, "prefetch predicate")?;
                }
                check_prefetch(self.ir.buffer(*buffer).space, *level, *evict, *uniform).map_err(|m| self.err(m))?;
            }
            Op::Pack { result, parts } => self.packing("pack", *result, parts)?,
            Op::Unpack { results, x } => self.packing("unpack", *x, results)?,
            Op::HalfInst { result, func, ty, args, mods, predicate } => {
                self.half_inst(*result, *func, *ty, args, *mods, *predicate)?
            }
            Op::Cvt { result, func, args } => {
                let what = func.cute_name();
                let want = func.operand_types();
                if args.len() != want.len() {
                    return Err(self.err(format!("{what} takes {} operands, not {}", want.len(), args.len())));
                }
                for (&a, &t) in args.iter().zip(want) {
                    self.expect(a, t, &format!("{what}'s operand"))?;
                }
                self.expect(*result, func.result_type(), &format!("the result of {}", func.ptx()))?;
                if let CvtFn::Packed { op, rnd, ftz } = func {
                    let mixed = !matches!(op, PackedOp::Add | PackedOp::Sub | PackedOp::Mul | PackedOp::Fma);
                    let problem = match op {
                        PackedOp::Fma | PackedOp::FmaHalf { .. } if rnd.is_none() => Some("fma takes a rounding mode"),
                        PackedOp::AddToHalf { .. } | PackedOp::SubToHalf { .. }
                            if *rnd != Some(crate::ir::RoundingMode::NearestEven) =>
                        {
                            Some("rounds to nearest even (.rn)")
                        }
                        _ if *ftz && mixed => Some("takes no .ftz"),
                        _ => None,
                    };
                    if let Some(problem) = problem {
                        return Err(self.err(format!("{}: {what} {problem}", func.ptx())));
                    }
                }
            }
            Op::PtrToInt { result, buffer, index, .. } => {
                self.expect_index(*index, "ptrtoint index")?;
                let want = match self.ir.buffer(*buffer).space {
                    MemSpace::Global | MemSpace::Constant => DType::I64,
                    MemSpace::Shared | MemSpace::Local => DType::I32,
                    MemSpace::Raw(_) => return Err(self.err("ptrtoint of a raw pointer: its integer is its address")),
                };
                self.expect(*result, want, "ptrtoint result")?;
            }
            Op::CvtAddr { result, x, from, to } => {
                use crate::tensor::AddressSpace as A;
                let width = |a: A| match a {
                    A::Generic | A::Gmem => Some(DType::I64),
                    A::Smem | A::Rmem => Some(DType::I32),
                    _ => None,
                };
                let (Some(tf), Some(tt)) = (width(*from), width(*to)) else {
                    return Err(self.err(format!("cvta between {from} and {to}: only generic, gmem, smem and rmem")));
                };
                if (*from == A::Generic) == (*to == A::Generic) {
                    return Err(self.err(format!("cvta from {from} to {to}: exactly one side must be generic")));
                }
                self.expect(*x, tf, "cvta operand")?;
                self.expect(*result, tt, "cvta result")?;
            }
            Op::Assume { result, x, divby, pow2 } => {
                let ty = self.ty(*x);
                if !ty.is_int() {
                    return Err(self.err(format!("assume needs an integer operand, but {x} is {ty}")));
                }
                self.expect(*result, ty, "assume result")?;
                if *divby == 0 {
                    return Err(self.err("assume: the divisor must be positive"));
                }
                if *pow2 && !divby.is_power_of_two() {
                    return Err(self.err(format!("assume: a power of two cannot be a multiple of {divby}")));
                }
            }
            Op::Wgmma { .. } | Op::WgmmaFence { .. } | Op::WgmmaCommitGroup | Op::WgmmaWait { .. } => {
                wgmma::check_op(self.ir, op).map_err(|m| self.err(m))?
            }
            Op::Tcgen05Ld { results, taddr, shape, num, sp: Some(sp), red, pack } => {
                let what = "tcgen05.ld.spcompress";
                self.expect(*taddr, DType::I32, "a tcgen05.ld TMEM address")?;
                if *shape != crate::ir::TmemShape::S32x32b
                    || !(4..=128).contains(num)
                    || !num.is_power_of_two()
                    || *pack
                {
                    return Err(self.err(format!(
                        "{what}.{}.x{num}: tcgen05.ld.spcompress loads Float32 values, unpacked, in .32x32b, repeated \
                         .x4 to .x128",
                        shape.ptx()
                    )));
                }
                if results.len() != *num as usize / 2 || sp.mdata.len() != (*num as usize / 32).max(1) {
                    return Err(self.err(format!(
                        "{what}.x{num} returns {} compressed values and {} metadata registers, not {} and {}",
                        results.len(),
                        sp.mdata.len(),
                        *num / 2,
                        (*num as usize / 32).max(1)
                    )));
                }
                for &r in results {
                    self.expect(r, DType::F32, "a tcgen05.ld.spcompress compressed value")?;
                }
                for &m in &sp.mdata {
                    self.expect(m, DType::I32, "a tcgen05.ld.spcompress metadata register")?;
                }
                if let Some(red) = red {
                    self.expect(red.result, DType::F32, "the reduction of a tcgen05.ld.red.spcompress")?;
                }
            }
            Op::Tcgen05Ld { results, taddr, shape, num, .. }
            | Op::Tcgen05St { values: results, taddr, shape, num, .. } => {
                let what = op.name();
                self.expect(*taddr, DType::I32, "a tcgen05.ld/st TMEM address")?;
                if !num.is_power_of_two() || *num > shape.max_repetition() {
                    return Err(self.err(format!(
                        "{what}.{}.x{num}: the repetition is a power of two from 1 to {}",
                        shape.ptx(),
                        shape.max_repetition()
                    )));
                }
                let ty = results.first().map(|&r| self.ty(r)).unwrap_or(DType::I32);
                let bits = ty.bits();
                if !matches!(bits, 8 | 16 | 32) || ty.is_word_packed() {
                    return Err(self.err(format!("{what}: registers hold elements of 8, 16 or 32 bits, not {ty}")));
                }
                if results.len() != shape.registers(*num) as usize * (32 / bits) {
                    return Err(self.err(format!(
                        "{what}.{}.x{num} moves {} registers of {} {ty} elements, not {} elements",
                        shape.ptx(),
                        shape.registers(*num),
                        32 / bits,
                        results.len()
                    )));
                }
                for &r in results {
                    self.expect(r, ty, "a tcgen05.ld/st register element")?;
                }
                if let Op::Tcgen05Ld { pack, red: Some(red), .. } = op {
                    self.expect(red.result, DType::F32, "the reduction of a tcgen05.ld.red")?;
                    if ty != DType::F32
                        || *pack
                        || *num < 2
                        || !matches!(shape, crate::ir::TmemShape::S32x32b | crate::ir::TmemShape::S16x32bx2)
                    {
                        return Err(self.err(format!(
                            "{what}.red.{}.x{num}: tcgen05.ld.red loads Float32 values, unpacked, in .32x32b or \
                             .16x32bx2, repeated .x2 to .x128",
                            shape.ptx()
                        )));
                    }
                }
            }
            Op::Tcgen05Mma {
                kind,
                taddr,
                desc_a,
                desc_b,
                enable_input_d,
                elect,
                idesc,
                disable_lane,
                sf,
                meta,
                zero_col_mask,
            } => {
                self.expect(*taddr, DType::I32, "a tcgen05.mma TMEM address")?;
                // The prims forms' rules (block-scaled sparse kinds included).
                if let Some(why) = kind.form_error() {
                    return Err(self.err(format!("{}: {why}", kind.ptx())));
                }
                match (zero_col_mask, kind.prims.ws) {
                    (Some(z), true) => self.expect(*z, DType::U64, "a tcgen05.mma.ws zero-column mask")?,
                    (None, _) => {}
                    (Some(_), false) => {
                        return Err(self.err(format!("{}: only a .ws MMA takes a zero-column mask", kind.ptx())));
                    }
                }
                match (meta, kind.sparse) {
                    (Some(e), true) => self.expect(*e, DType::I32, "a tcgen05.mma.sp metadata TMEM address")?,
                    (None, false) => {}
                    _ => {
                        return Err(
                            self.err(format!("{}: a metadata address is given exactly for a sparse kind", kind.ptx()))
                        );
                    }
                }
                // The scale factors' TMEM addresses, exactly for a block-scaled kind.
                match (sf, kind.scale) {
                    (Some((a, b)), Some(_)) => {
                        self.expect(*a, DType::I32, "a tcgen05.mma TMEM address of SFA")?;
                        self.expect(*b, DType::I32, "a tcgen05.mma TMEM address of SFB")?;
                        if !disable_lane.is_empty() {
                            return Err(
                                self.err(format!("{}: a block-scaled MMA has no disable-output-lane", kind.ptx()))
                            );
                        }
                    }
                    (None, None) => {}
                    _ => {
                        return Err(self.err(format!(
                            "{}: scale-factor addresses are given exactly for a block-scaled kind",
                            kind.ptx()
                        )));
                    }
                }
                if kind.a_tmem {
                    self.expect(*desc_a, DType::I32, "a tcgen05.mma TMEM address of A")?;
                } else {
                    self.expect(*desc_a, DType::U64, "a tcgen05.mma shared-memory descriptor")?;
                }
                if let Some(v) = idesc {
                    self.expect(*v, DType::U32, "a tcgen05.mma runtime instruction descriptor")?;
                }
                if !disable_lane.is_empty() && disable_lane.len() != kind.disable_lane_words() {
                    return Err(self.err(format!(
                        "{}: {{disable-output-lane}} has {} words, not {}",
                        kind.ptx(),
                        disable_lane.len(),
                        kind.disable_lane_words()
                    )));
                }
                for &w in disable_lane {
                    self.expect(w, DType::I32, "a tcgen05.mma disable-output-lane word")?;
                }
                self.expect(*desc_b, DType::U64, "a tcgen05.mma shared-memory descriptor")?;
                self.expect(*enable_input_d, DType::Bool, "tcgen05.mma's enable-input-d")?;
                self.expect(*elect, DType::Bool, "tcgen05.mma's elected lane")?;
                if kind.kind.operand_format(kind.a).is_none() || kind.kind.operand_format(kind.b).is_none() {
                    return Err(self.err(format!("{}: no operand format for {} x {}", kind.ptx(), kind.a, kind.b)));
                }
            }
            // m10-clc-scheduling.
            Op::ClcTryCancel { buffer, index, mbar, mbar_index, .. } => {
                if self.ir.buffer(*buffer).space != MemSpace::Shared {
                    return Err(self.err("clusterlaunchcontrol.try_cancel writes its response to shared memory"));
                }
                self.expect_index(*index, "try_cancel response index")?;
                self.mbarrier(*mbar, *mbar_index)?;
                self.base_alignment(*buffer, 16, "clusterlaunchcontrol.try_cancel")?;
            }
            Op::ClcQueryCancel { buffer, index, results, .. } => {
                if self.ir.buffer(*buffer).space != MemSpace::Shared {
                    return Err(self.err("clc_response reads the try_cancel response from shared memory"));
                }
                self.expect_index(*index, "try_cancel response index")?;
                for &r in results {
                    self.expect(r, DType::I32, "a clc_response result")?;
                }
                self.base_alignment(*buffer, 16, "clc_response")?;
            }
            // m10-s2t-copies.
            Op::Tcgen05Cp { taddr, desc, elect, .. } => {
                self.expect(*taddr, DType::I32, "a tcgen05.cp TMEM address")?;
                self.expect(*desc, DType::U64, "a tcgen05.cp shared-memory descriptor")?;
                self.expect(*elect, DType::Bool, "tcgen05.cp's elected lane")?;
            }
            Op::Tcgen05Commit { buffer, index, mask, .. } => {
                self.mbarrier(*buffer, *index)?;
                if let Some(m) = mask {
                    self.expect(*m, DType::I16, "tcgen05.commit's multicast mask")?;
                }
            }
            Op::Tcgen05 { inst, buffer, args, .. } => {
                for &a in args {
                    self.expect(a, DType::I32, inst.name())?;
                }
                if *inst == Tcgen05Inst::Alloc {
                    let b = buffer.ok_or_else(|| self.err("tcgen05.alloc needs a shared-memory slot"))?;
                    if self.ir.buffer(b).space != MemSpace::Shared {
                        return Err(self.err("tcgen05.alloc writes the TMEM address to shared memory"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Inline assembly's constraint string matches its operands: an entry
    /// per output, then per input, then clobbers; each code is one NVPTX
    /// takes, each register class holds its operand's type, an immediate is
    /// an integer constant, an address is a 64-bit register (or a 32-bit one
    /// into shared memory), a tied input has its output's type (and no
    /// output is tied twice); a guard is a Boolean last input that the
    /// template's first instruction is predicated on; and every `$n` of the
    /// template names an operand. LLVM would reject (or abort on) the rest,
    /// or, as LLVM 23 does with a constraint it does not know, report an
    /// error and leave the statement out of the PTX.
    fn inline_asm(
        &self,
        template: &str,
        inputs: &[AsmInput],
        outputs: &[ValueId],
        constraints: &str,
        predicated: bool,
    ) -> Result {
        if let Some(k) = asm_empty_constraint(constraints) {
            return Err(
                self.err(format!("inline assembly: entry {k} of the constraint string {constraints:?} is empty"))
            );
        }
        let entries = asm_constraints(constraints);
        let n_out = entries.iter().take_while(|e| matches!(e, AsmConstraint::Output(_))).count();
        let n_in = entries[n_out..]
            .iter()
            .take_while(|e| matches!(e, AsmConstraint::Input(_) | AsmConstraint::Tied(_)))
            .count();
        if entries[n_out + n_in..].iter().any(|e| !matches!(e, AsmConstraint::Clobber(_))) {
            return Err(self.err(format!(
                "inline assembly: the constraint string {constraints:?} must list the outputs (=), then the inputs, \
                 then the clobbers (~)"
            )));
        }
        if (n_out, n_in) != (outputs.len(), inputs.len()) {
            return Err(self.err(format!(
                "inline assembly: the constraint string {constraints:?} has {n_out} output(s) and {n_in} input(s), \
                 but the statement has {} and {}",
                outputs.len(),
                inputs.len()
            )));
        }
        for (k, (&o, entry)) in outputs.iter().zip(&entries).enumerate() {
            if let AsmConstraint::Output(code) = entry {
                // `&` marks an output written before the inputs are read.
                let code = code.strip_prefix('&').unwrap_or(code);
                self.register_fits(code, self.ty(o), &format!("output ${k}"), false)?;
            }
        }
        let mut tied = vec![false; outputs.len()];
        for (k, (input, entry)) in inputs.iter().zip(&entries[n_out..]).enumerate() {
            let what = format!("input ${}", n_out + k);
            match (input, entry) {
                (AsmInput::Value(v), AsmConstraint::Tied(o)) => {
                    let Some(&output) = outputs.get(*o) else {
                        return Err(
                            self.err(format!("inline assembly: {what} is tied to output ${o}, which does not exist"))
                        );
                    };
                    if std::mem::replace(&mut tied[*o], true) {
                        return Err(self.err(format!("inline assembly: output ${o} is tied to two inputs")));
                    }
                    if self.ty(*v) != self.ty(output) {
                        return Err(self.err(format!(
                            "inline assembly: {what} ({}) is tied to output ${o} ({}), so they must have one type",
                            self.ty(*v),
                            self.ty(output)
                        )));
                    }
                }
                // An immediate is printed into the PTX as a number: LLVM takes
                // an integer constant (and prints it sign-extended), and
                // prints a Boolean as -1, not the 1 the PTX would mean.
                (AsmInput::Value(v), AsmConstraint::Input(code)) if matches!(code.as_str(), "n" | "i") => {
                    match self.constants.get(v) {
                        Some(c) if c.dtype().is_int() => {}
                        Some(c) => {
                            return Err(self.err(format!(
                                "inline assembly: {what} has the immediate constraint {code:?}, which takes an \
                                 integer constant, and this one is {}; pass it in a register ({:?})",
                                c.dtype(),
                                asm_register(c.dtype())
                            )));
                        }
                        None => {
                            return Err(self.err(format!(
                                "inline assembly: {what} has the immediate constraint {code:?}, so it must be a \
                                 constant"
                            )));
                        }
                    }
                }
                (AsmInput::Value(v), AsmConstraint::Input(code)) => {
                    self.register_fits(code, self.ty(*v), &what, true)?
                }
                (AsmInput::Address { buffer, index, .. }, AsmConstraint::Input(code)) => {
                    self.expect_index(*index, "address index")?;
                    let space = self.ir.buffer(*buffer).space;
                    match code.as_str() {
                        "l" => {}
                        "r" if space == MemSpace::Shared => {}
                        "r" => {
                            return Err(self.err(format!(
                                "inline assembly: {what} is the address of {} memory, which needs a 64-bit register \
                                 (constraint \"l\"); only a shared-memory address fits in 32 bits (\"r\")",
                                if space == MemSpace::Global { "global" } else { "register-array (local)" }
                            )));
                        }
                        _ => {
                            return Err(self.err(format!(
                                "inline assembly: {what} is an address, which is a 64-bit register (constraint \
                                 \"l\"), or a 32-bit one (\"r\") for shared memory, not {code:?}"
                            )));
                        }
                    }
                }
                (AsmInput::Address { .. }, _) => {
                    return Err(self.err(format!("inline assembly: {what} is an address, which cannot be tied")));
                }
                _ => unreachable!("inputs are inputs or tied"),
            }
        }
        if predicated {
            let guard = match (inputs.last(), entries.get((n_out + n_in).wrapping_sub(1))) {
                (Some(AsmInput::Value(g)), Some(AsmConstraint::Input(code))) if code == "b" => *g,
                _ => return Err(self.err("inline assembly: a predicated statement's last input is its guard (\"b\")")),
            };
            self.expect(guard, DType::Bool, "an inline PTX predicate")?;
            let prefix = format!("@${} ", n_out + n_in - 1);
            if !template.starts_with(&prefix) {
                return Err(self.err(format!("inline assembly: a predicated template must start with {prefix:?}")));
            }
        }
        let refs = asm_operand_refs(template).map_err(|e| self.err(format!("inline assembly: {e}")))?;
        if let Some(n) = refs.into_iter().find(|&n| n >= n_out + n_in) {
            return Err(self.err(format!(
                "inline assembly: the template refers to ${n}, but there are {} operand(s) ($0 to ${}): {template:?}",
                n_out + n_in,
                (n_out + n_in) as isize - 1
            )));
        }
        Ok(())
    }

    /// `code` is an NVPTX register class (constraint code) that holds values
    /// of `ty`. Other codes are errors: LLVM 21 rejects the ones NVPTX does
    /// not know, but LLVM 23's `mlir-opt` only prints its error and drops
    /// the statement from the PTX, and the generic ones it accepts (`X`, or
    /// `m`, which passes the address of a stack copy) are not what cute-rs's
    /// operands are. `input` adds the immediates (`n`, `i`) to the message.
    fn register_fits(&self, code: &str, ty: DType, what: &str, input: bool) -> Result {
        let (class, holds): (&str, &[DType]) = match code {
            "b" => ("predicate", &[DType::Bool]),
            "c" => ("8-bit", &[DType::I8, DType::U8]),
            "h" => ("16-bit", &[DType::I16, DType::U16, DType::F16, DType::BF16]),
            "r" => ("32-bit integer", &[DType::I32, DType::U32]),
            "l" | "N" => ("64-bit integer", &[DType::I64, DType::U64]),
            "f" => ("Float32", &[DType::F32]),
            "d" => ("Float64", &[DType::F64]),
            "q" => {
                return Err(self.err(format!(
                    "inline assembly: {what} has the constraint \"q\", a 128-bit register, and cute-rs has no \
                     128-bit type; use {:?} for {ty}",
                    asm_register(ty)
                )));
            }
            "n" | "i" => {
                return Err(self.err(format!(
                    "inline assembly: {what} has the immediate constraint {code:?}, but an output is a register; \
                     use {:?} for {ty}",
                    asm_register(ty)
                )));
            }
            _ => {
                return Err(self.err(format!(
                    "inline assembly: {what} has the constraint {code:?}, which is not an NVPTX register class: \
                     b (Boolean), c (8-bit), h (16-bit), r (32-bit), l (64-bit), f (Float32) or d (Float64){}; use \
                     {:?} for {ty}",
                    if input { ", or the immediate n (an integer constant)" } else { "" },
                    asm_register(ty)
                )));
            }
        };
        if !holds.contains(&ty) {
            return Err(self.err(format!(
                "inline assembly: {what} is {ty}, which a {class} register (constraint {code:?}) does not hold; use \
                 {:?}",
                asm_register(ty)
            )));
        }
        Ok(())
    }

    /// A load's or store's memory attributes ([`check_mem_attrs`]), and the
    /// type of its cache-policy operand.
    fn mem_attrs(&self, attrs: &MemAttrs, access: Access, buffer: BufferId, elems: usize, elem: DType) -> Result {
        check_mem_attrs(attrs, access, self.ir.buffer(buffer).space, elems, elem).map_err(|m| self.err(m))?;
        if let Some(p) = attrs.cache_policy
            && !matches!(self.ty(p), DType::I64 | DType::U64)
        {
            return Err(self.err(format!(
                "an L2 cache policy (.L2::cache_hint) is a 64-bit value from createpolicy, not {}",
                self.ty(p)
            )));
        }
        Ok(())
    }

    /// A pack's parts (or an unpack's results) are two or more values of one
    /// type (not Boolean) whose widths add up to that of the packed value.
    fn packing(&self, what: &str, packed: ValueId, parts: &[ValueId]) -> Result {
        let part = self.same(parts, what)?;
        let whole = self.ty(packed);
        if part == DType::Bool || whole == DType::Bool {
            return Err(self.err(format!("{what} cannot take Boolean values (they have no bit width in memory)")));
        }
        // Word-packed values (Int4, FP4, FP6, FP6x4) may fill only the low
        // bits of an integer (the rest zero), as their memory words and
        // containers hold them.
        let partial = part.is_word_packed() && whole.is_int() && parts.len() * part.bits() < whole.bits();
        if parts.is_empty() || (parts.len() < 2 && !partial) {
            return Err(self.err(format!("{what} needs at least two parts, not {}", parts.len())));
        }
        if !partial && parts.len() * part.bits() != whole.bits() {
            return Err(self.err(format!(
                "{what}: {} x {part} is {} bits, but {whole} is {}",
                parts.len(),
                parts.len() * part.bits(),
                whole.bits()
            )));
        }
        Ok(())
    }

    /// An extended-precision instruction reads or writes the carry flag, and
    /// takes Int32, Uint32, Int64 or Uint64 operands of one type, as CuTe DSL
    /// requires.
    fn carry_inst(&self, result: ValueId, func: CarryFn, args: &[ValueId]) -> Result {
        let what = func.cute_name();
        if !func.carry_in && !func.carry_out {
            return Err(self.err(format!(
                "{what}: an extended-precision instruction reads or writes the carry flag, and this one does neither"
            )));
        }
        if args.len() != func.arity() {
            return Err(self.err(format!("{what} takes {} operands, not {}", func.arity(), args.len())));
        }
        let mut all = args.to_vec();
        all.push(result);
        let t = self.same(&all, what)?;
        if !matches!(t, DType::I32 | DType::U32 | DType::I64 | DType::U64) {
            return Err(self.err(format!("{what} takes Int32, Uint32, Int64 or Uint64 operands, not {t}")));
        }
        Ok(())
    }

    /// Follow the carry flag through a region: a carry-in instruction needs
    /// a carry-out instruction of its own kind before it in the same region,
    /// with nothing between them that loses the flag ([`carry_flag_breaker`]:
    /// whatever ends the straight-line code PTX keeps the flag through, such
    /// as control flow, calls, or a 64-bit division that LLVM lowers with a
    /// branch, and inline PTX that uses the flag itself). The kinds are
    /// the additions (adds and multiply-adds) and the subtractions: the PTX
    /// ISA describes one flag for carries and borrows, but on the GPU a
    /// borrow is the inverted carry (`subc` after `add.cc` subtracts one
    /// minus the carry, and `addc` after `sub.cc` adds one minus the
    /// borrow), so a flag set by one kind and read by the other follows
    /// neither reading reliably.
    fn carry_flag(&self, op: &Op, loc: Loc, flag: &mut CarryFlag) -> Result {
        let lose = |flag: &mut CarryFlag, what: &'static str| {
            if let CarryFlag::Set(set) = *flag {
                *flag = CarryFlag::Lost { set, by: (what, loc) };
            }
        };
        match op {
            Op::CarryInst { func, .. } => {
                if func.carry_in {
                    let what = func.cute_name();
                    match *flag {
                        CarryFlag::Set((name, set)) if borrows(name) != (func.op == CarryOp::Sub) => {
                            let (reader, setter) = if borrows(name) {
                                ("an addition reads it as a carry", "a subtraction")
                            } else {
                                ("a subtraction reads it as a borrow", "an addition or multiply-add")
                            };
                            return Err(self.err(format!(
                                "{what} reads the carry flag set by the {name} at {}, but {reader}, which {setter} \
                                 does not leave: the PTX ISA describes one flag for carries and borrows, but on the \
                                 GPU a borrow is the inverted carry, so a chain cannot switch between adding and \
                                 subtracting; start a new chain ({}) or pass the carry as a value (add_with_carry, \
                                 sub_with_borrow)",
                                at(set),
                                if func.op == CarryOp::Sub { "sub_cc" } else { "add_cc or mad_cc" }
                            )));
                        }
                        CarryFlag::Set(_) => {}
                        CarryFlag::Unset => {
                            return Err(self.err(format!(
                                "{what} reads the carry flag, but nothing before it in this block of straight-line \
                                 code sets it: start the chain with add_cc, sub_cc or mad_cc (or a carry-in-and-out \
                                 form) in the same block (the flag does not reach into an if_, for_range or \
                                 while_carry body from outside it)"
                            )));
                        }
                        CarryFlag::Lost { set: (name, set), by: (breaker, by) } => {
                            return Err(self.err(format!(
                                "{what} reads the carry flag set by the {name} at {}, but the {breaker} at {} comes \
                                 between them, and PTX does not keep the carry flag through it; keep each chain in \
                                 straight-line code",
                                at(set),
                                at(by)
                            )));
                        }
                    }
                }
                if func.carry_out {
                    *flag = CarryFlag::Set((func.cute_name(), loc));
                }
            }
            _ => {
                if let Some(why) = carry_flag_breaker(op, |v| self.ty(v), |v| self.constants.contains_key(&v)) {
                    lose(flag, why)
                }
            }
        }
        Ok(())
    }

    /// A half-precision instruction exists in PTX with these modifiers and
    /// types, and takes CuTe DSL's combinations of them.
    fn half_inst(
        &self,
        result: ValueId,
        func: HalfFn,
        ty: HalfType,
        args: &[ValueId],
        mods: HalfModifiers,
        predicate: Option<ValueId>,
    ) -> Result {
        let name = half_inst_name(func, ty, mods);
        if args.len() != func.arity() {
            return Err(self.err(format!("{name} takes {} operands, not {}", func.arity(), args.len())));
        }
        let want = ty.value_type();
        for &a in args {
            self.expect(a, want, &format!("an operand of {name}"))?;
        }
        self.expect(result, want, &format!("the result of {name}"))?;
        if func != HalfFn::Fma && !ty.is_packed() {
            return Err(self.err(format!("{name}: CuTe DSL's {} helpers take packed pairs only", func.ptx())));
        }
        let f16 = ty.elem() == DType::F16;
        let problem = match func {
            HalfFn::Ex2 if mods != HalfModifiers::default() => Some("ex2.approx takes no rounding or other modifiers"),
            HalfFn::Fma if !mods.rn => Some("fma takes the rounding mode .rn"),
            _ if mods.ftz && !f16 => Some("ftz applies to Float16 only"),
            _ if mods.sat && !f16 => Some("sat applies to Float16 only"),
            HalfFn::Add | HalfFn::Sub if mods.relu || mods.oob => Some("relu and oob apply to fma only"),
            _ if mods.sat && mods.relu => Some("sat and relu exclude each other"),
            _ if mods.oob && (mods.ftz || mods.sat) => Some("oob excludes ftz and sat"),
            _ => None,
        };
        if let Some(problem) = problem {
            return Err(self.err(format!("{name} is not a PTX instruction: {problem}")));
        }
        if let Some(p) = predicate {
            if func == HalfFn::Fma {
                return Err(self.err(format!("{name}: CuTe DSL's fma helpers take no predicate")));
            }
            self.expect(p, DType::Bool, &format!("the predicate of {name}"))?;
        }
        Ok(())
    }

    /// `redux.sync` takes its integer kinds on Int32 and Uint32, `fmax` and
    /// `fmin` on Float32, and `.abs` and `.NaN` with those only.
    fn redux(&self, result: ValueId, kind: ReduxKind, value: ValueId, mask: ValueId, abs: bool, nan: bool) -> Result {
        let t = self.ty(value);
        let what = kind.name();
        if kind.is_float() && t != DType::F32 {
            return Err(self.err(format!("warp_redux_sync: {what} reduces Float32 values, not {t}")));
        }
        if !kind.is_float() && !matches!(t, DType::I32 | DType::U32) {
            let hint = if t == DType::F32 { " (Float32 takes fmax, fmin, max and min)" } else { "" };
            return Err(self.err(format!("warp_redux_sync: {what} reduces Int32 or Uint32 values, not {t}{hint}")));
        }
        if (abs || nan) && !kind.is_float() {
            return Err(self.err(format!(
                "{} is not a PTX instruction: abs and nan apply to the Float32 kinds (fmax, fmin) only",
                redux_inst_name(kind, abs, nan, t)
            )));
        }
        self.expect(mask, DType::U32, "lane mask")?;
        self.expect(result, t, "redux result")
    }

    /// A floating-point instruction must exist in PTX with exactly these
    /// modifiers.
    #[allow(clippy::too_many_arguments)]
    fn float_inst(
        &self,
        result: ValueId,
        func: FloatFn,
        args: &[ValueId],
        mode: FloatMode,
        ftz: bool,
        nan: bool,
        xorsign_abs: bool,
    ) -> Result {
        if args.len() != func.arity() {
            return Err(self.err(format!("{} takes {} operands, not {}", func.ptx(), func.arity(), args.len())));
        }
        let mut all = args.to_vec();
        all.push(result);
        let t = self.same(&all, func.ptx())?;
        let name = float_inst_name(func, mode, ftz, nan, xorsign_abs, t);
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
        if xorsign_abs && (!matches!(func, Min | Max) || !f32) {
            return Err(
                self.err(format!("{name} is not a PTX instruction: .xorsign.abs exists for Float32 min and max"))
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

    /// An integer instruction's data operands share one type that the PTX
    /// instruction takes, its bit positions and shift amounts are Int32, and
    /// its result has the type the instruction gives.
    fn int_inst(&self, result: ValueId, func: IntFn, args: &[ValueId]) -> Result {
        let what = func.cute_name();
        if args.len() != func.arity() {
            return Err(self.err(format!("{what} takes {} operands, not {}", func.arity(), args.len())));
        }
        let (data, positions) = args.split_at(func.data_operands());
        let t = self.same(data, what)?;
        let allowed = func.operand_types();
        if !allowed.contains(&t) {
            let names: Vec<&str> = allowed.iter().map(|&a| cute_type_name(a)).collect();
            let list = match names.split_last() {
                Some((last, [])) => last.to_string(),
                Some((last, rest)) => format!("{} or {last}", rest.join(", ")),
                None => unreachable!("every integer instruction takes some type"),
            };
            return Err(self.err(format!("{what} takes {list} operands, not {t}")));
        }
        let names = &func.operand_names()[func.data_operands()..];
        for (&p, name) in positions.iter().zip(names) {
            self.expect(p, DType::I32, &format!("{what}'s {name}"))?;
        }
        self.expect(result, func.result_type(t), &format!("the result of {}", int_inst_name(func, t)))
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

    /// The coordinates of a TMA instruction: one to five i32s.
    fn tma_coords(&self, coords: &[ValueId], what: &str) -> Result {
        if coords.is_empty() || coords.len() > crate::tma::MAX_RANK {
            return Err(self.err(format!("{what} takes 1 to 5 coordinates, not {}", coords.len())));
        }
        for &c in coords {
            self.expect(c, DType::I32, "TMA coordinate")?;
        }
        Ok(())
    }

    /// The coordinates and offsets of a TMA access mode: the im2col modes take 3 to 5 coordinates (and `.im2col` one
    /// i16 offset per spatial dimension), the four-row modes five.
    fn tma_mode(&self, mode: &TmaMode, n: usize, what: &str) -> Result {
        match mode {
            TmaMode::Tile => {}
            TmaMode::Im2col { .. } | TmaMode::Im2colNoOffs => {
                if !(3..=crate::tma::MAX_RANK).contains(&n) {
                    return Err(self.err(format!("{what} in {} mode takes 3 to 5 coordinates, not {n}", mode.ptx())));
                }
                let offs = mode.offsets();
                if matches!(mode, TmaMode::Im2col { .. }) && offs.len() != n - 2 {
                    return Err(self.err(format!(
                        "{what} in .im2col mode takes {} offsets (one per spatial dimension), not {}",
                        n - 2,
                        offs.len()
                    )));
                }
                for &o in offs {
                    self.expect(o, DType::I16, "im2col offset")?;
                }
            }
            TmaMode::Gather4 | TmaMode::Scatter4 => {
                if n != 5 {
                    return Err(self.err(format!(
                        "{what} in {} mode takes five coordinates (a column and four rows), not {n}",
                        mode.ptx()
                    )));
                }
            }
        }
        Ok(())
    }

    /// The `.override::*` operands of a TMA instruction.
    fn tma_override(&self, o: &crate::ir::TmaOverride, mode: &TmaMode, n: usize, what: &str) -> Result {
        self.expect(o.global_address, DType::I64, "overriding global address")?;
        if o.dims.is_empty() {
            return Ok(());
        }
        if *mode != TmaMode::Tile {
            return Err(self.err(format!("{what} overrides the tensor's attributes in .tile mode only")));
        }
        if o.dims.len() != n {
            return Err(self.err(format!("{what} overrides {} extents of a rank-{n} tensor map", o.dims.len())));
        }
        for &d in &o.dims {
            self.expect(d, DType::I16, "overriding extent")?;
        }
        if n == 1 {
            if !o.lower_strides.is_empty() || o.upper_strides.is_some() {
                return Err(self.err(format!("{what} of a rank-1 tensor map overrides its extent only")));
            }
            return Ok(());
        }
        if o.lower_strides.len() != n - 1 || o.upper_strides.is_none() {
            return Err(self.err(format!(
                "{what} overriding a rank-{n} tensor map's extents overrides its {} strides too",
                n - 1
            )));
        }
        for &s in &o.lower_strides {
            self.expect(s, DType::I32, "overriding stride")?;
        }
        self.expect(o.upper_strides.expect("checked"), DType::I16, "overriding upper strides")
    }

    fn mbarrier(&self, buffer: BufferId, index: ValueId) -> Result {
        if self.ir.buffer(buffer).space != MemSpace::Shared {
            return Err(self.err("mbarriers live in shared memory"));
        }
        self.expect_index(index, "mbarrier index")
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

/// Types with no arithmetic of their own: TFloat32 and the 8-bit floats,
/// which are converted (with [`Value::to`](crate::Value::to)) before any.
fn storage_only(t: DType) -> bool {
    t == DType::TF32 || t.is_narrow_float() || t.is_packed_view()
}

fn storage_only_message(t: DType) -> String {
    format!(
        "{} has no arithmetic or comparisons of its own: convert it first (Value::to; the comparison methods \
         convert it to Float32 themselves)",
        t.cute_name()
    )
}

/// CuTe DSL's name of a type (`Int32`, `Float16`, ...).
fn cute_type_name(t: DType) -> &'static str {
    t.cute_name()
}

/// A load or a store, for [`check_mem_attrs`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    Load,
    Store,
}

/// Check that a load or store of `elems` elements of `elem` in `space` may
/// carry the memory attributes `attrs`, by the PTX ISA's rules for `ld` and
/// `st`, which `ptxas` enforces; the error says what PTX does not allow.
/// The validator applies it to every load and store, and copy atoms to
/// their attributes when they are made.
///
/// Beyond PTX, cute-rs gives per-thread (local) memory plain accesses only,
/// and Booleans (stored as bytes) no attributes.
pub fn check_mem_attrs(
    attrs: &MemAttrs,
    access: Access,
    space: MemSpace,
    elems: usize,
    elem: DType,
) -> std::result::Result<(), String> {
    if attrs.is_plain() {
        return Ok(());
    }
    let load = access == Access::Load;
    let op = if load { "load" } else { "store" };
    let order = attrs.order;
    let o = order.name();
    if elem == DType::Bool {
        return Err(format!(
            "ordered and cache-hinted {op}s move integers and floats, not Booleans (use a Uint8 or Int32 flag)"
        ));
    }
    if matches!(space, MemSpace::Local | MemSpace::Raw(crate::tensor::AddressSpace::Rmem)) {
        return Err(format!(
            "per-thread (local) memory takes only plain {op}s: PTX allows ordered and volatile accesses on global \
             and shared memory, and cache hints on global memory"
        ));
    }
    match (access, order) {
        (Access::Load, MemoryOrder::Release) => {
            return Err("a load cannot be a release (MemoryOrder::Release): loads are weak, relaxed, acquire, sc, \
                        volatile, mmio or constant"
                .into());
        }
        (Access::Store, MemoryOrder::Acquire) => {
            return Err("a store cannot be an acquire (MemoryOrder::Acquire): stores are weak, relaxed, release, sc, \
                        volatile or mmio"
                .into());
        }
        (Access::Store, MemoryOrder::Constant) => {
            return Err("MemoryOrder::Constant marks a load of memory that no thread writes (ld.global.nc); a store \
                        cannot be constant"
                .into());
        }
        (_, MemoryOrder::AcqRel) => {
            return Err(format!(
                "acq_rel (MemoryOrder::AcqRel) is an order for atomics; a {op} is relaxed or {}",
                if load { "acquire" } else { "release" }
            ));
        }
        _ => {}
    }
    if !load {
        if attrs.non_coherent {
            return Err("non-coherent (invariant, .nc) accesses are loads; a store cannot be one".into());
        }
        if let Some(p) = attrs.l2_prefetch {
            return Err(format!("a prefetch size (.{}) is for loads: PTX's st takes none", p.ptx()));
        }
    }
    if let Some(c) = attrs.cop {
        if load && !c.for_loads() {
            return Err(format!(".{} is a store cache operator; loads take .ca, .cg, .cs, .lu or .cv", c.ptx()));
        }
        if !load && !c.for_stores() {
            return Err(format!(".{} is a load cache operator; stores take .wb, .cg, .cs or .wt", c.ptx()));
        }
    }
    if !matches!(space, MemSpace::Global | MemSpace::Raw(crate::tensor::AddressSpace::Gmem)) {
        let global_only = if attrs.is_non_coherent() {
            Some("a non-coherent load (ld.global.nc: invariant, or MemoryOrder::Constant)".to_string())
        } else if order == MemoryOrder::Mmio {
            Some(format!("an MMIO {op} (.mmio)"))
        } else if let Some(e) = attrs.l1_eviction {
            Some(format!("an L1 eviction priority (.{})", e.l1_ptx()))
        } else if let Some(p) = attrs.l2_prefetch {
            Some(format!("a prefetch size (.{})", p.ptx()))
        } else if attrs.cache_policy.is_some() {
            Some("an L2 cache policy (.L2::cache_hint)".to_string())
        } else {
            None
        };
        if let Some(what) = global_only {
            let place = match space {
                MemSpace::Shared | MemSpace::Raw(crate::tensor::AddressSpace::Smem) => "shared memory",
                _ => "a generic address",
            };
            return Err(format!("{what} needs global memory, but this {op} is to {place}"));
        }
    }
    if attrs.non_coherent && order.is_strong() {
        return Err(format!("a non-coherent load (.nc, invariant) is weak; it cannot also be {o}"));
    }
    match order {
        MemoryOrder::Mmio => {
            if elems != 1 {
                return Err(format!("an MMIO {op} (.mmio) moves one element; this one moves {elems}"));
            }
            if attrs.has_hints() {
                return Err(format!(
                    "an MMIO {op} (.mmio) takes no cache operator, eviction priority, prefetch size or cache policy"
                ));
            }
            if attrs.scope != FenceScope::Sys {
                return Err(format!(
                    "an MMIO {op} has system scope (PTX allows only .mmio.relaxed.sys): set the scope to \
                     MemoryScope::Sys, not {:?}",
                    attrs.scope
                ));
            }
        }
        MemoryOrder::Volatile => {
            let hint = if let Some(c) = attrs.cop {
                Some(format!("a cache operator (.{})", c.ptx()))
            } else if let Some(e) = attrs.l1_eviction {
                Some(format!("an L1 eviction priority (.{})", e.l1_ptx()))
            } else if attrs.cache_policy.is_some() {
                Some("an L2 cache policy (.L2::cache_hint)".to_string())
            } else {
                None
            };
            if let Some(hint) = hint {
                let allowed =
                    if load { "ld.volatile takes only a prefetch size" } else { "st.volatile takes no hints" };
                return Err(format!("a volatile {op} cannot take {hint}: PTX's {allowed}"));
            }
        }
        MemoryOrder::Relaxed | MemoryOrder::Acquire | MemoryOrder::Release | MemoryOrder::Sc => {
            if let Some(c) = attrs.cop {
                return Err(format!(
                    "a {o} {op} takes no cache operator (.{}): PTX allows cache operators only on weak {op}s",
                    c.ptx()
                ));
            }
        }
        MemoryOrder::Weak | MemoryOrder::Constant => {
            if let (Some(c), Some(e)) = (attrs.cop, attrs.l1_eviction) {
                return Err(format!(
                    "a cache operator (.{}) and an L1 eviction priority (.{}) are separate forms of PTX's {}; give \
                     one of them",
                    c.ptx(),
                    e.l1_ptx(),
                    if load { "ld" } else { "st" }
                ));
            }
            if attrs.is_non_coherent()
                && let Some(c) = attrs.cop
                && !matches!(c, CacheOperator::Ca | CacheOperator::Cg | CacheOperator::Cs)
            {
                return Err(format!("a non-coherent load (ld.global.nc) takes .ca, .cg or .cs, not .{}", c.ptx()));
            }
        }
        MemoryOrder::AcqRel => unreachable!("rejected above"),
    }
    Ok(())
}

/// Check a prefetch of `space` with these options, by the PTX ISA's rules
/// for `prefetch` and `prefetchu`.
pub fn check_prefetch(
    space: MemSpace,
    level: PrefetchCacheLevel,
    evict: Option<CacheEvictionPriority>,
    uniform: bool,
) -> std::result::Result<(), String> {
    if space == MemSpace::Shared {
        return Err("prefetch takes a global or local address (PTX); shared memory is not cached".into());
    }
    if let Some(e) = evict {
        if !matches!(e, CacheEvictionPriority::EvictLast | CacheEvictionPriority::EvictNormal) {
            return Err(format!(
                "a prefetch's eviction priority is evict_last or evict_normal (prefetch.global.L2::evict_*), not \
                 {e:?}"
            ));
        }
        if level != PrefetchCacheLevel::L2 || uniform {
            return Err("an eviction priority applies to an L2 prefetch (prefetch.global.L2::evict_*)".into());
        }
        if space != MemSpace::Global {
            return Err("a prefetch with an eviction priority (prefetch.global.L2::evict_*) needs global memory".into());
        }
    }
    if uniform && level != PrefetchCacheLevel::L1 {
        return Err("a uniform prefetch (prefetchu) fills the L1 cache only".into());
    }
    Ok(())
}

/// What a load's or store's memory attributes need beyond sm_80 and PTX
/// 7.0: the PTX ISA's notes on `ld` and `st` (ordered accesses need sm_70
/// and PTX 6.0, and cache operators sm_20, which every target has).
fn mem_attrs_requirement(attrs: &MemAttrs) -> Vec<Option<TargetRequirement>> {
    let need = |sm, ptx, what| Some(TargetRequirement { sm, ptx, what, arch_specific: None, older_ptx: None });
    let mut needs = Vec::new();
    if attrs.order.is_scoped() && attrs.scope == FenceScope::Cluster {
        needs.push(need(90, 78, "an ordered load or store with cluster scope"));
    }
    if attrs.order == MemoryOrder::Mmio {
        needs.push(need(70, 82, "an MMIO load or store (.mmio)"));
    }
    if attrs.l1_eviction.is_some() {
        needs.push(need(70, 74, "an L1 eviction priority (.L1::evict_*)"));
    }
    match attrs.l2_prefetch {
        Some(L2PrefetchSize::Size256B) => needs.push(need(80, 74, "a 256-byte L2 prefetch size (.L2::256B)")),
        Some(_) => needs.push(need(75, 74, "an L2 prefetch size (.L2::64B, .L2::128B)")),
        None => {}
    }
    if attrs.cache_policy.is_some() {
        needs.push(need(80, 74, "an L2 cache policy (.L2::cache_hint)"));
    }
    needs
}

/// An access to another CTA's shared memory (`.shared::cluster`, after
/// `mapa`): sm_90 and PTX 7.8.
fn dsmem_requirement(peer: Option<ValueId>) -> Option<TargetRequirement> {
    peer.map(|_| TargetRequirement {
        sm: 90,
        ptx: 78,
        what: "an access to distributed shared memory (.shared::cluster)",
        arch_specific: None,
        older_ptx: None,
    })
}

/// What an atomic or reduction needs beyond sm_80 and PTX 7.0, from the
/// PTX ISA's notes on `atom` and `red`: the BFloat16 adds sm_90 (PTX 7.8),
/// as does cluster scope. The memory semantics need sm_70 (PTX 6.0), the
/// Float16 add sm_70 (PTX 6.3), the packed Float16 add sm_60 (PTX 6.2) and
/// the Float64 add sm_60 (PTX 5.0).
fn atomic_requirement(ty: AtomicType, scope: FenceScope) -> Vec<Option<TargetRequirement>> {
    let need = |sm, ptx, what| Some(TargetRequirement { sm, ptx, what, arch_specific: None, older_ptx: None });
    vec![
        match ty {
            AtomicType::Bf16 => need(90, 78, "a BFloat16 atomic add (atom.add.noftz.bf16, red.add.noftz.bf16)"),
            AtomicType::Bf16x2 => {
                need(90, 78, "a packed BFloat16 atomic add (atom.add.noftz.bf16x2, red.add.noftz.bf16x2)")
            }
            _ => None,
        },
        (scope == FenceScope::Cluster).then_some(()).and(need(90, 78, "an atomic or reduction with cluster scope")),
    ]
}

/// What an [`IntFn`] needs beyond sm_80 and PTX 7.0. Each one so far is
/// older (`lop3` needs sm_50 and PTX 4.3, `shf` sm_32 and PTX 3.1, the
/// others sm_20 and PTX 2.0 or less), but the match is exhaustive so that a
/// new function has to say.
fn int_inst_requirement(func: IntFn) -> Option<TargetRequirement> {
    match func {
        IntFn::Bfe
        | IntFn::Bfi
        | IntFn::Lop3 { .. }
        | IntFn::Shf { .. }
        | IntFn::Prmt { .. }
        | IntFn::MulWide
        | IntFn::Mul24 { .. }
        | IntFn::Mad24 { .. }
        | IntFn::AddSat
        | IntFn::SubSat => None,
        // dp4a needs sm_61 and PTX 5.0.
        IntFn::Dp4a => None,
    }
}

/// What a half-precision instruction needs beyond sm_80 and PTX 7.0, from
/// the PTX ISA's notes: `add`, `sub` and `ex2` on BFloat16 need sm_90 (PTX
/// 7.8) and `.oob` sm_90 (PTX 8.1). The others are older: the Float16
/// arithmetic sm_53, `ex2.approx.f16x2` sm_75, and the BFloat16 `fma` and
/// `.relu` sm_80 (PTX 7.0).
fn half_inst_requirement(func: HalfFn, ty: HalfType, mods: HalfModifiers) -> Vec<Option<TargetRequirement>> {
    let need = |sm, ptx, what| Some(TargetRequirement { sm, ptx, what, arch_specific: None, older_ptx: None });
    let bf16 = ty.elem() == DType::BF16;
    vec![
        match (func, bf16) {
            (HalfFn::Add, true) => need(90, 78, "add.bf16x2 (add_packed_bf16x2)"),
            (HalfFn::Sub, true) => need(90, 78, "sub.bf16x2 (sub_packed_bf16x2)"),
            (HalfFn::Ex2, true) => need(90, 78, "ex2.approx.ftz.bf16x2 (exp2_packed_bf16x2)"),
            _ => None,
        },
        mods.oob.then_some(()).and(need(90, 81, "fma with .oob (oob=True)")),
    ]
}

/// The targets of `cvt` with the `.s2f6x2` type, from PTX ISA 9.1: the PTX
/// ISA's target notes list sm_100a, sm_103a, sm_107a, sm_110a, sm_120a and
/// sm_121a, and no family. [`Arch`] has sm_100a and sm_120a so far
/// (m6-arch-targets adds the others).
pub const S2F6X2: ArchSpecific = ArchSpecific {
    chips: &[(100, 91), (103, 91), (107, 91), (110, 91), (120, 91), (121, 91)],
    families: &[],
    baselines: &[],
};

/// The targets of `cvt` with the `.ue8m0x2` type, from PTX ISA 8.6:
/// sm_100a, sm_101a (sm_110a from PTX 9.0) and sm_120a, and from PTX 8.8
/// the sm_100f, sm_101f (sm_110f) and sm_120f families.
pub const UE8M0X2: ArchSpecific = ArchSpecific {
    chips: &[(100, 86), (101, 86), (110, 90), (120, 87)],
    families: &[(100, 88), (101, 88), (110, 90), (120, 88)],
    baselines: &[],
};

/// The mixed-precision packed add, sub and fma (`.f16x2` or `.bf16x2` with
/// `.f32x2`): the sm_107f family from
/// PTX ISA 9.4.
pub const PACKED_MIXED: ArchSpecific = ArchSpecific { chips: &[], families: &[(107, 94)], baselines: &[] };

/// `cvt` with the `.ue5m3x2` type: the
/// sm_107f family from PTX ISA 9.4 (the first PTX ISA that has the type;
/// cute-rs lists the Rubin family, the only one with PTX 9.4 targets
/// documented, and no installed ptxas assembles it).
pub const UE5M3X2: ArchSpecific = ArchSpecific { chips: &[], families: &[(107, 94)], baselines: &[] };

/// What a conversion needs beyond sm_80 and PTX 7.0: `.satfinite` came with
/// PTX 8.1, the integer-to-BFloat16 conversions need sm_90 (PTX 7.8), and
/// `.s2f6x2` is architecture-specific ([`S2F6X2`]). `cvt.rna.tf32.f32` is
/// sm_80 and PTX 7.0.
fn cvt_requirement(func: CvtFn) -> Option<TargetRequirement> {
    let need = |sm, ptx, what| Some(TargetRequirement { sm, ptx, what, arch_specific: None, older_ptx: None });
    match func {
        CvtFn::F32x2ToBf16x2 => need(80, 81, "cvt.rn.satfinite.bf16x2.f32 (cvt_f32x2_bf16x2)"),
        CvtFn::F32ToTf32 => None,
        CvtFn::S8ToBf16 => need(90, 78, "cvt.rn.bf16.s8 (cvt_i8_bf16)"),
        CvtFn::S2f6x2ToBf16x2 { .. } => Some(TargetRequirement {
            sm: S2F6X2.oldest(),
            ptx: S2F6X2.min_ptx(),
            what: "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 (the s2f6x2 int8 conversion)",
            arch_specific: Some(S2F6X2),
            older_ptx: None,
        }),
        // "cvt with .e4m3x2/.e5m2x2 for sm_90 or higher introduced in PTX
        // ISA version 7.8; for sm_89 in PTX ISA version 8.1."
        CvtFn::F32x2ToF8x2 { .. } | CvtFn::F16x2ToF8x2 { .. } | CvtFn::F8x2ToF16x2 { .. } => Some(TargetRequirement {
            sm: 89,
            ptx: 78,
            what: "cvt with .e4m3x2 or .e5m2x2 (an FP8 conversion)",
            arch_specific: None,
            older_ptx: Some((90, 81)),
        }),
        // "cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32" and
        // "cvt.rn{.relu}.f16x2.{e2m1x2/e2m3x2/e3m2x2}": the targets of the
        // UE8M0 conversions.
        CvtFn::F32x2ToMx { .. } | CvtFn::MxToF16x2 { .. } => Some(TargetRequirement {
            sm: UE8M0X2.oldest(),
            ptx: UE8M0X2.min_ptx(),
            what: "cvt with .e2m1x2, .e2m3x2 or .e3m2x2 (an FP4 or FP6 conversion)",
            arch_specific: Some(UE8M0X2),
            older_ptx: None,
        }),
        // add, sub, mul and fma on .f32x2: sm_100 and newer, PTX ISA 8.6.
        CvtFn::Packed { op: PackedOp::Add | PackedOp::Sub | PackedOp::Mul | PackedOp::Fma, .. } => {
            need(100, 86, "the packed .f32x2 arithmetic (add_packed_f32x2, mul_packed_f32x2, ...)")
        }
        // The mixed-precision packed forms (f16x2/bf16x2 with f32x2): sm_107f.
        CvtFn::Packed { .. } => Some(TargetRequirement {
            sm: PACKED_MIXED.oldest(),
            ptx: PACKED_MIXED.min_ptx(),
            what: "the mixed-precision packed arithmetic (f16x2 or bf16x2 with f32x2)",
            arch_specific: Some(PACKED_MIXED),
            older_ptx: None,
        }),
        CvtFn::F32x2ToUe5m3x2 => Some(TargetRequirement {
            sm: UE5M3X2.oldest(),
            ptx: UE5M3X2.min_ptx(),
            what: "cvt.rp.satfinite.ue5m3x2.f32 (cvt_f32x4_to_fnv8e5m3x4)",
            arch_specific: Some(UE5M3X2),
            older_ptx: None,
        }),
        CvtFn::F32x2ToUe8m0x2 { .. } | CvtFn::Bf16x2ToUe8m0x2 { .. } | CvtFn::Ue8m0x2ToBf16x2 => {
            Some(TargetRequirement {
                sm: UE8M0X2.oldest(),
                ptx: UE8M0X2.min_ptx(),
                what: "cvt with .ue8m0x2 (a UE8M0 scale-factor conversion)",
                arch_specific: Some(UE8M0X2),
                older_ptx: None,
            })
        }
    }
}

/// What an instruction needs from the compilation target: the oldest
/// architecture (`sm`, e.g. 90 for sm_90) and PTX ISA version (`ptx`, e.g.
/// 78 for 7.8) that have it. An architecture-specific feature
/// (`arch_specific`) is not inherited by newer architectures: only the
/// targets it lists have it, `sm` is the oldest of them and `ptx` the
/// version every target needs (each listed target can need a newer one).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TargetRequirement {
    pub sm: u32,
    pub ptx: u32,
    /// The feature, for messages.
    pub what: &'static str,
    /// For an architecture- or family-specific feature, the targets that
    /// have it.
    pub arch_specific: Option<ArchSpecific>,
    /// `(sm, ptx)`: targets older than `sm` need PTX version `ptx` instead
    /// (the FP8 conversions are PTX 7.8 on sm_90 but PTX 8.1 on sm_89).
    pub older_ptx: Option<(u32, u32)>,
}

/// The targets that have an architecture-specific feature: the PTX ISA's
/// "requires sm_100a and is supported on sm_100f from PTX ISA version 8.8".
/// An architecture-specific (`a`) target has its own features and those of
/// its family; a family-specific (`f`) target, such as sm_100f, has the
/// features of its family, which the later `f` and `a` targets of the
/// family (sm_103f, sm_103a) have too (the PTX ISA's Table 70 lists the
/// families: sm_10x, sm_11x and sm_12x).
///
/// Targets are given by architecture number (100 for sm_100a and sm_100f),
/// so that targets cute-rs does not have yet are listed too: [`Arch`] has
/// sm_90a, sm_100a and sm_120a, and m6-arch-targets adds the rest, and the
/// `f` targets, which this check accepts by their names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArchSpecific {
    /// The `a` targets that have the feature (100 for sm_100a), each with
    /// the PTX ISA version it needs there.
    pub chips: &'static [(u32, u32)],
    /// The families that have it, by the `f` target that has it first (100
    /// for sm_100f), with the PTX ISA version from which they do.
    pub families: &'static [(u32, u32)],
    /// Baseline alternatives: every target from `sm` on (`sm_100` or newer,
    /// whatever its suffix), with the PTX ISA version it needs there.
    pub baselines: &'static [(u32, u32)],
}

/// An architecture number under the chip's current name: sm_101 was renamed
/// sm_110 in PTX ISA 9.0, so what sm_110a (or sm_110f) has, sm_101a (or
/// sm_101f) has under its old name.
fn current_number(sm: u32) -> u32 {
    if sm == 101 { 110 } else { sm }
}

/// The family an architecture number belongs to: 10 for sm_100 to sm_107,
/// 11 for sm_110 (and sm_101, its name before PTX ISA 9.0), 12 for sm_120
/// and sm_121.
fn arch_family(sm: u32) -> u32 {
    current_number(sm) / 10
}

/// `"7.0"` for PTX ISA version 70.
fn ptx_version_name(ptx: u32) -> String {
    format!("{}.{}", ptx / 10, ptx % 10)
}

/// `a`, `a or b` or `a, b or c`.
fn alternatives(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    }
}

impl ArchSpecific {
    /// The PTX ISA version the feature needs on `arch`, or `None` if `arch`
    /// lacks it. `a` and `f` targets are told apart by the suffix of their
    /// names (`sm_100a`, `sm_100f`).
    pub fn ptx_on(&self, arch: Arch) -> Option<u32> {
        let (sm, chip) = (arch.sm(), arch.chip());
        let (specific, family) = (arch.is_arch_specific(), chip.ends_with('f'));
        let sm_now = current_number(sm);
        let own = self.chips.iter().filter(|&&(c, _)| specific && current_number(c) == sm_now).map(|&(_, ptx)| ptx);
        let inherited = self
            .families
            .iter()
            .filter(|&&(f, _)| (specific || family) && arch_family(f) == arch_family(sm) && sm_now >= current_number(f))
            .map(|&(_, ptx)| ptx);
        let baseline = self.baselines.iter().filter(|&&(n, _)| arch.sm() >= n).map(|&(_, ptx)| ptx);
        own.chain(inherited).chain(baseline).min()
    }

    /// The oldest target that has the feature.
    pub fn oldest(&self) -> u32 {
        self.chips.iter().chain(self.families).chain(self.baselines).map(|&(sm, _)| sm).min().unwrap_or(0)
    }

    /// The lowest PTX ISA version any target with the feature needs.
    pub fn min_ptx(&self) -> u32 {
        self.chips.iter().chain(self.families).chain(self.baselines).map(|&(_, ptx)| ptx).min().unwrap_or(0)
    }

    /// The targets, for messages: `sm_100a (or the sm_100f family with PTX
    /// 8.8)`.
    pub fn describe(&self) -> String {
        let chips: Vec<String> = self.chips.iter().map(|(sm, _)| format!("sm_{sm}a")).collect();
        let mut text = alternatives(&chips);
        let mut ptx: Vec<u32> = self.families.iter().map(|&(_, p)| p).collect();
        ptx.dedup();
        let families: Vec<String> = match ptx.as_slice() {
            [one] => {
                let names: Vec<String> = self.families.iter().map(|(f, _)| format!("sm_{f}f")).collect();
                vec![format!("the {} family with PTX {}", alternatives(&names), ptx_version_name(*one))]
            }
            _ => self
                .families
                .iter()
                .map(|&(f, p)| format!("the sm_{f}f family with PTX {}", ptx_version_name(p)))
                .collect(),
        };
        if !families.is_empty() {
            let rest = alternatives(&families);
            text = if text.is_empty() { rest } else { format!("{text} (or {rest})") };
        }
        let baselines: Vec<String> =
            self.baselines.iter().map(|&(n, p)| format!("sm_{n} or newer with PTX {}", ptx_version_name(p))).collect();
        if !baselines.is_empty() {
            let rest = alternatives(&baselines);
            text = if text.is_empty() { rest } else { format!("{text}, or {rest}") };
        }
        text
    }
}

/// The targets of `ldmatrix` with the `.m16n16` and `.m8n16` shapes, the
/// `.b8` type and the `.src_fmt` and `.dst_fmt` qualifiers, from PTX ISA
/// 8.6: sm_100a, sm_101a (sm_110a from PTX 9.0) and sm_120a, and from PTX
/// 8.8 the sm_100f, sm_101f (sm_110f) and sm_120f families (those of
/// [`UE8M0X2`]).
pub const LDMATRIX_B8: ArchSpecific = ArchSpecific {
    chips: &[(100, 86), (101, 86), (110, 90), (120, 86)],
    families: &[(100, 88), (101, 88), (110, 90), (120, 88)],
    baselines: &[],
};

/// The architecture-specific Hopper features (`wgmma.mma_async`,
/// `wgmma.fence` and the like): sm_90a only, from PTX ISA 8.0; plain sm_90
/// and later chips lack them. m8's WGMMA uses it.
pub const SM90A_ONLY: ArchSpecific = ArchSpecific { chips: &[(90, 80)], families: &[], baselines: &[] };

/// `wgmma.mma_async` on mixed signed and unsigned 8-bit integers
/// (`.s8.u8`, `.u8.s8`): sm_90a from PTX ISA 8.4.
pub const SM90A_PTX84: ArchSpecific = ArchSpecific { chips: &[(90, 84)], families: &[], baselines: &[] };

/// `setmaxnreg`: sm_90a from PTX ISA 8.0, sm_100a and sm_120a from 8.6,
/// the sm_100f and sm_120f families from 8.8 and sm_110f from 9.0 (the PTX
/// ISA's table of architecture-specific features). m8 uses it.
pub const SETMAXNREG: ArchSpecific = ArchSpecific {
    chips: &[(90, 80), (100, 86), (120, 86)],
    families: &[(100, 88), (120, 88), (110, 90)],
    baselines: &[],
};

/// `tensormap.replace` (the PTX ISA's
/// target notes for it): sm_90a from PTX ISA 8.3, sm_100a, sm_101a and
/// sm_120a (8.6; sm_101a is sm_110a from 9.0), and the sm_100f, sm_101f,
/// sm_110f and sm_120f families from 8.8.
pub const TENSORMAP_REPLACE: ArchSpecific = ArchSpecific {
    chips: &[(90, 83), (100, 86), (101, 86), (110, 90), (120, 86)],
    families: &[(100, 88), (101, 88), (110, 90), (120, 88)],
    baselines: &[],
};

/// `.tile::gather4` to `.shared::cluster` (the PTX ISA's
/// target notes for `cp.async.bulk.tensor` and `cp.async.bulk.prefetch.tensor`): sm_100a
/// from PTX ISA 8.6, and the sm_100f family from 8.8.
pub const TMA_GATHER4: ArchSpecific = ArchSpecific { chips: &[(100, 86)], families: &[(100, 88)], baselines: &[] };

/// `.tile::scatter4`: sm_100a and sm_101a (sm_110a from PTX 9.0) from PTX ISA 8.6, and the
/// sm_100f, sm_101f and sm_110f families from 8.8 (9.0 for sm_110f).
pub const TMA_SCATTER4: ArchSpecific = ArchSpecific {
    chips: &[(100, 86), (101, 86), (110, 90)],
    families: &[(100, 88), (101, 88), (110, 90)],
    baselines: &[],
};

/// `.override::global_address` and `.override_attribute` on `cp.async.bulk.tensor`: the
/// sm_107f family from PTX ISA 9.4.
pub const TMA_OVERRIDE: ArchSpecific = ArchSpecific { chips: &[], families: &[(107, 94)], baselines: &[] };

/// `tcgen05.alloc`, `tcgen05.dealloc` and `tcgen05.relinquish_alloc_permit`
/// (the PTX ISA's target notes): sm_100a and
/// sm_101a from PTX ISA 8.6 (sm_101a is sm_110a from 9.0), and the sm_100f,
/// sm_101f and sm_110f families from 8.8 (9.0 for sm_110f).
pub const TCGEN05_ALLOC: ArchSpecific = ArchSpecific {
    chips: &[(100, 86), (101, 86), (110, 90)],
    families: &[(100, 88), (101, 88), (110, 90)],
    baselines: &[],
};

/// `tcgen05.ld.red` (the PTX ISA's
/// target notes): sm_110a (sm_101a before PTX 9.0) and the sm_103f and
/// sm_110f families (so sm_103a too), from PTX ISA 8.8, not sm_100a.
pub const TCGEN05_LD_RED: ArchSpecific =
    ArchSpecific { chips: &[(101, 88), (110, 90)], families: &[(103, 88), (110, 90)], baselines: &[] };

/// `tcgen05.ld{.red}.spcompress`: the
/// architecture-specific sm_107a only, PTX ISA 9.4 (no `f` family).
pub const TCGEN05_LD_SPCOMPRESS: ArchSpecific = ArchSpecific { chips: &[(107, 94)], families: &[], baselines: &[] };

/// `tcgen05.mma.kind::i8`: the PTX
/// ISA's target notes allow it only on the architecture-specific targets
/// sm_100a and sm_101a (PTX ISA 8.6) and sm_110a (9.0), not on any `f`
/// family. CuTe DSL 4.8.1's `MmaI8Op` lists sm_100f and sm_103f among its
/// architectures too; cute-rs follows the PTX ISA, which ptxas enforces.
pub const TCGEN05_MMA_I8: ArchSpecific =
    ArchSpecific { chips: &[(100, 86), (101, 86), (110, 90)], families: &[], baselines: &[] };

// m10-umma-blockscaled.
/// `tcgen05.mma.kind::mxf8f6f4.block_scale` (without `.scale_vec`): the
/// targets of `tcgen05.alloc` (sm_100a, sm_101a/sm_110a, and the `f`
/// families from PTX ISA 8.8, so sm_103a and sm_107a too).
pub const TCGEN05_MMA_MXF8F6F4: ArchSpecific = TCGEN05_ALLOC;

/// `tcgen05.mma.kind::mxf4.block_scale.scale_vec::2X`: the PTX ISA's
/// `.scale_vec::NX` qualifiers are architecture-specific (sm_100a, sm_101a
/// from PTX 8.6; sm_103a from 8.8; sm_107a and sm_110a from 9.0), not on
/// any `f` family (where only `.block16`/`.block32` exist).
pub const TCGEN05_MMA_MXF4: ArchSpecific =
    ArchSpecific { chips: &[(100, 86), (101, 86), (103, 88), (107, 90), (110, 90)], families: &[], baselines: &[] };

/// `tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X`: as
/// [`TCGEN05_MMA_MXF4`], from PTX ISA 8.7 on sm_100a and sm_101a.
pub const TCGEN05_MMA_MXF4NVF4: ArchSpecific =
    ArchSpecific { chips: &[(100, 87), (101, 87), (103, 88), (107, 90), (110, 90)], families: &[], baselines: &[] };

/// The K = 96 forms (`.block32` with `.kind::mxf4`, `.block16` with
/// `.kind::mxf4nvf4`, CuTe DSL's `SM103MmaMXF4Op` and
/// `SM103MmaMXF4NVF4Op`): sm_103a from PTX ISA 8.8 and sm_107a from 9.0.
pub const TCGEN05_MMA_K96: ArchSpecific =
    ArchSpecific { chips: &[(103, 88), (107, 90)], families: &[], baselines: &[] };

/// The Rubin `tcgen05.mma` forms (K = 64 `.kind::f8f6f4`
/// and `.kind::mxf8f6f4`, K = 128 `.kind::mxf4nvf4`, and the collector
/// qualifiers of CuTe DSL's SM107 MMAs): the sm_107f family (so sm_107a
/// too) from PTX ISA 9.4.
pub const TCGEN05_MMA_SM107: ArchSpecific = ArchSpecific { chips: &[], families: &[(107, 94)], baselines: &[] };

/// m11-prims-mma-forms: `tcgen05.mma` with `scale-input-d`: sm_100a from
/// PTX ISA 8.6 and the sm_100f family from 8.8.
pub const TCGEN05_MMA_SCALE_INPUT_D: ArchSpecific =
    ArchSpecific { chips: &[(100, 86)], families: &[(100, 88)], baselines: &[] };

/// The sparse block-scaled `mma.sp` with
/// `.scale_vec::4X`, UE8M0 scales and `.kind::mxf4nvf4`: PTX ISA 9.1 on the
/// sm_120a/sm_121a targets and the sm_120f family.
pub const SM120_SP_BLOCK_SCALED_MMA_4X_UE8M0: ArchSpecific =
    ArchSpecific { chips: &[(120, 91), (121, 91)], families: &[(120, 91)], baselines: &[] };

/// `clusterlaunchcontrol.try_cancel` with `.multicast::cluster::all`
/// (the PTX ISA's target notes): sm_100a,
/// sm_101a (sm_110a from PTX 9.0) and sm_120a from PTX ISA 8.6, and the
/// sm_100f, sm_101f, sm_110f and sm_120f families from 8.8.
pub const CLC_MULTICAST: ArchSpecific = ArchSpecific {
    chips: &[(100, 86), (101, 86), (110, 90), (120, 86)],
    families: &[(100, 88), (101, 88), (110, 90), (120, 88)],
    baselines: &[],
};

/// The `.exclusive` forms of `tcgen05.alloc` and `tcgen05.dealloc`: the
/// sm_107f family from PTX ISA 9.4.
pub const TCGEN05_EXCLUSIVE: ArchSpecific = ArchSpecific { chips: &[], families: &[(107, 94)], baselines: &[] };

/// `redux.sync` on Float32, with or without `.abs` and `.NaN`: sm_100a from
/// PTX ISA 8.6, and the sm_100f family from 8.8.
const REDUX_F32: ArchSpecific = ArchSpecific { chips: &[(100, 86)], families: &[(100, 88)], baselines: &[] };

/// The PTX name of a Float32 `redux.sync` (a constant for messages).
fn redux_f32_name(kind: ReduxKind, abs: bool, nan: bool) -> &'static str {
    match (kind == ReduxKind::Fmax, abs, nan) {
        (true, false, false) => "redux.sync.max.f32",
        (true, true, false) => "redux.sync.max.abs.f32",
        (true, false, true) => "redux.sync.max.NaN.f32",
        (true, true, true) => "redux.sync.max.abs.NaN.f32",
        (false, false, false) => "redux.sync.min.f32",
        (false, true, false) => "redux.sync.min.abs.f32",
        (false, false, true) => "redux.sync.min.NaN.f32",
        (false, true, true) => "redux.sync.min.abs.NaN.f32",
    }
}

// m10-sm120-blockscaled.
/// The warp-level block-scaled `mma.sync` forms (PTX ISA 8.7 on sm_120a and
/// sm_121a; 8.8 on the family targets sm_120f and sm_121f), CuTe DSL's
/// admissible architectures for `MmaSM120BlockScaledOp`.
pub const SM120_BLOCK_SCALED_MMA: ArchSpecific =
    ArchSpecific { chips: &[(120, 87), (121, 87)], families: &[(120, 88)], baselines: &[] };

fn block_scaled_mma_what(kind: crate::ir::BlockScaledMmaKind) -> &'static str {
    match kind.kind_name() {
        "mxf4" => "a block-scaled mma.sync (kind::mxf4)",
        "mxf4nvf4" => "a block-scaled mma.sync (kind::mxf4nvf4)",
        _ => "a block-scaled mma.sync (kind::mxf8f6f4)",
    }
}

/// What an `mma.sync` or `mma.sp` needs (PTX ISA, "mma" and "mma.sp"): FP8
/// operands sm_89 and PTX 8.4, and PTX 8.7 for the m16n8k16 shape or an f16
/// accumulator; `mma.sp::ordered_metadata` PTX 8.5.
fn mma_requirement(kind: MmaKind) -> Option<TargetRequirement> {
    let need = |sm, ptx, what| Some(TargetRequirement { sm, ptx, what, arch_specific: None, older_ptx: None });
    if kind.is_fp8() {
        let (_, _, k) = kind.shape_mnk();
        if k == 16 || kind.dtypes().2 == DType::F16 {
            need(89, 87, "an FP8 mma.sync of shape m16n8k16 or with an f16 accumulator")
        } else {
            need(89, 84, "an FP8 mma.sync")
        }
    } else if kind.is_sparse() {
        need(80, 85, "mma.sp::ordered_metadata")
    } else {
        need(80, 70, "mma.sync")
    }
}

/// The target `op` needs, from the PTX ISA's notes on each instruction;
/// `None` for instructions every supported target (sm_80 and newer, PTX
/// 7.0) has.
pub fn target_requirement(op: &Op) -> Option<TargetRequirement> {
    let need = |sm, ptx, what| Some(TargetRequirement { sm, ptx, what, arch_specific: None, older_ptx: None });
    let specific = |what, targets: ArchSpecific| {
        Some(TargetRequirement {
            sm: targets.oldest(),
            ptx: targets.min_ptx(),
            what,
            arch_specific: Some(targets),
            older_ptx: None,
        })
    };
    let mut needs: Vec<Option<TargetRequirement>> = match op {
        Op::Mma { kind, .. } => vec![mma_requirement(*kind)],
        // m10-sm120-blockscaled.
        Op::MmaBlockScaled { kind, .. } => vec![specific(block_scaled_mma_what(*kind), SM120_BLOCK_SCALED_MMA)],
        // m11-prims-mma-forms: .scale_vec::4X with UE8M0 scales and kind::mxf4nvf4 needs PTX 9.1.
        Op::MmaSpBlockScaled { kind, .. } if kind.scale_vec == 4 && kind.sf == DType::F8E8M0FNU => vec![specific(
            "mma.sp.block_scale.scale_vec::4X with UE8M0 scales (prims::mma_sp_block_scale)",
            SM120_SP_BLOCK_SCALED_MMA_4X_UE8M0,
        )],
        Op::MmaSpBlockScaled { .. } => {
            vec![specific("a sparse block-scaled mma.sp (prims::mma_sp_block_scale)", SM120_BLOCK_SCALED_MMA)]
        }
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
        Op::MbarrierWait { kind: MbarrierWaitKind::Test, state: true, .. } => vec![need(80, 70, "mbarrier.test_wait")],
        Op::MbarrierWait { kind: MbarrierWaitKind::Test, .. } => vec![need(80, 71, "mbarrier.test_wait.parity")],
        Op::MbarrierWait { kind: MbarrierWaitKind::Try, time_limit: Some(_), .. } => {
            vec![need(90, 78, "mbarrier.try_wait with a suspend-time hint")]
        }
        Op::MbarrierWait { kind: MbarrierWaitKind::Try, .. } => vec![need(90, 78, "mbarrier.try_wait")],
        // m11-prims-nvvm-layer.
        Op::MbarrierExt { kind, .. } => vec![match kind {
            MbarrierExtKind::ArriveDropExpectTx => need(90, 80, "mbarrier.arrive_drop.expect_tx"),
            MbarrierExtKind::CompleteTx => need(90, 80, "mbarrier.complete_tx"),
            _ => need(80, 70, kind.ptx()),
        }],
        Op::Fence { scope: FenceScope::Cluster, sc: true } => vec![need(90, 78, "fence.sc.cluster")],
        Op::StBulk { .. } => vec![need(100, 86, "st.bulk")],
        Op::Tcgen05Shift { .. } => vec![specific("tcgen05.shift (prims::tcgen05_shift)", TCGEN05_ALLOC)],
        Op::FenceMbarrierInit => vec![need(90, 80, "fence.mbarrier_init")],
        Op::GridDepControl { launch_dependents: false } => vec![need(90, 78, "griddepcontrol.wait")],
        Op::GridDepControl { launch_dependents: true } => vec![need(90, 78, "griddepcontrol.launch_dependents")],
        Op::ClusterArrive { relaxed: true, .. } => vec![need(90, 80, "barrier.cluster.arrive.relaxed")],
        Op::ClusterArrive { relaxed: false, .. } => vec![need(90, 78, "barrier.cluster.arrive")],
        Op::ClusterWait { .. } => vec![need(90, 78, "barrier.cluster.wait")],
        Op::CpAsyncBulk { direction: BulkDirection::G2S, mcast_mask: Some(_), .. } => {
            vec![need(90, 80, "cp.async.bulk with .multicast::cluster")]
        }
        Op::CpAsyncBulk { direction: BulkDirection::G2S, .. } => vec![need(90, 80, "cp.async.bulk")],
        Op::CpAsyncBulk { direction: BulkDirection::S2S, .. } => {
            vec![need(90, 80, "cp.async.bulk.shared::cluster.shared::cta")]
        }
        // The overrides (sm_107f), gather4 and scatter4 (sm_100a and the f
        // families), and the byte-masked bulk store (sm_100).
        Op::TmaLoad { overrides: Some(_), .. } | Op::TmaStore { overrides: Some(_), .. } => {
            vec![specific("cp.async.bulk.tensor with .override::global_address (a TMA override copy)", TMA_OVERRIDE)]
        }
        // m10-cta-pair: `.cta_group::2` (PTX 8.6 on sm_100a, sm_101a, sm_110a; the f families
        // from 8.8), the targets of tcgen05's CTA pairs.
        Op::TmaLoad { two_cta: true, .. } => {
            vec![specific("cp.async.bulk.tensor with .cta_group::2 (a CTA-pair TMA load)", TCGEN05_ALLOC)]
        }
        Op::TmaLoad { mode: TmaMode::Gather4, .. } => {
            vec![specific("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4", TMA_GATHER4)]
        }
        Op::TmaPrefetch { mode: TmaMode::Gather4, .. } => {
            vec![specific("cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4", TMA_GATHER4)]
        }
        Op::TmaStore { mode: TmaMode::Scatter4, .. } => {
            vec![specific("cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4", TMA_SCATTER4)]
        }
        Op::CpAsyncBulk { byte_mask: Some(_), .. } => {
            vec![need(100, 86, "cp.async.bulk.global.shared::cta.bulk_group.cp_mask (a byte-masked bulk store)")]
        }
        Op::TmaLoad { mcast_mask: Some(_), .. } => vec![need(90, 80, "cp.async.bulk.tensor with .multicast::cluster")],
        Op::TmaStore { reduce: None, .. } => vec![need(90, 80, "cp.async.bulk.tensor (a TMA store)")],
        Op::TmaStore { reduce: Some(_), .. } => vec![need(90, 80, "cp.reduce.async.bulk.tensor")],
        Op::TmaPrefetch { .. } => vec![need(90, 80, "cp.async.bulk.prefetch.tensor")],
        Op::PrefetchTensormap { .. } => vec![need(90, 80, "prefetch.tensormap")],
        Op::TensormapReplace { .. } => {
            vec![specific("tensormap.replace (update_tma_descriptor, tensormap_replace)", TENSORMAP_REPLACE)]
        }
        Op::FenceTensormapRelease => vec![need(90, 83, "fence.proxy.tensormap::generic.release")],
        Op::FenceTensormapAcquire { .. } => vec![need(90, 83, "fence.proxy.tensormap::generic.acquire")],
        Op::TensormapCpFenceproxy { .. } => vec![need(90, 83, "tensormap.cp_fenceproxy")],
        Op::BulkPrefetch { .. } => vec![need(90, 80, "cp.async.bulk.prefetch")],
        Op::StAsync { .. } => vec![need(90, 81, "st.async")],
        Op::CpAsyncBulk { direction: BulkDirection::S2G, .. } => {
            vec![need(90, 80, "cp.async.bulk.global.shared::cta.bulk_group")]
        }
        Op::TmaLoad { .. } => vec![need(90, 80, "cp.async.bulk.tensor")],
        Op::CpAsyncBulkCommitGroup => vec![need(90, 80, "cp.async.bulk.commit_group")],
        Op::CpAsyncBulkWaitGroup { read: false, .. } => vec![need(90, 80, "cp.async.bulk.wait_group")],
        Op::CpAsyncBulkWaitGroup { read: true, .. } => vec![need(90, 80, "cp.async.bulk.wait_group.read")],
        Op::FenceProxy { kind: ProxyKind::Alias } => vec![need(70, 75, "fence.proxy.alias")],
        Op::FenceProxy { kind } => vec![need(90, 80, kind.ptx())],
        Op::Fence { scope: FenceScope::Cluster, sc: false } => vec![need(90, 78, "fence.acq_rel.cluster")],
        Op::FloatInst { func: FloatFn::Tanh, .. } => vec![need(75, 70, "tanh.approx.f32")],
        Op::FloatInst { nan, xorsign_abs, .. } if *nan || *xorsign_abs => vec![
            nan.then_some(()).and(need(80, 70, "min and max with .NaN")),
            xorsign_abs.then_some(()).and(need(86, 72, "min and max with .xorsign.abs (fmax and fmin with abs)")),
        ],
        Op::IntInst { func, .. } => vec![int_inst_requirement(*func)],
        Op::Special {
            reg:
                SpecialReg::ClusterIdx(_)
                | SpecialReg::ClusterDim(_)
                | SpecialReg::BlockInClusterIdx(_)
                | SpecialReg::BlockInClusterDim(_)
                | SpecialReg::ClusterCtaRank
                | SpecialReg::ClusterNCtaRank,
            ..
        } => vec![need(90, 78, "the cluster special registers")],
        Op::Special { reg: SpecialReg::AggrSmemSize, .. } => vec![need(90, 81, "%aggr_smem_size (aggr_smem_size)")],
        // PTX has it from sm_80 (CuTe DSL checks for sm_90); `.shared::cta`
        // would need PTX 7.8, so the emitted form is `.shared`.
        Op::CpAsyncMbarrierArrive { .. } => vec![need(80, 70, "cp.async.mbarrier.arrive")],
        // CuTe DSL refuses `cutlass.range(..., vectorize=True)` below sm_100
        // (UNSUP_ARCH); sm_100 needs PTX 8.6.
        Op::For { opts: LoopOptions { vectorize: true, .. }, .. } => {
            vec![need(100, 86, "a vectorized loop (LoopOptions::vectorize, cutlass.range(..., vectorize=True))")]
        }
        Op::Redux { kind, abs, nan, .. } => vec![
            need(80, 70, "redux.sync"),
            kind.is_float().then_some(()).and(specific(redux_f32_name(*kind, *abs, *nan), REDUX_F32)),
        ],
        Op::CarryInst { func, .. } => vec![carry_inst_requirement(*func)],
        Op::Load { attrs, .. } | Op::Store { attrs, .. } => {
            let mut needs = mem_attrs_requirement(attrs);
            needs.push(dsmem_requirement(attrs.peer));
            needs
        }
        Op::Atomic { ty, scope, peer, .. } | Op::Red { ty, scope, peer, .. } => {
            let mut needs = atomic_requirement(*ty, *scope);
            needs.push(dsmem_requirement(*peer));
            needs
        }
        Op::Prefetch { evict: Some(_), .. } => {
            vec![need(80, 74, "a prefetch with an eviction priority (prefetch.global.L2::evict_*)")]
        }
        Op::HalfInst { func, ty, mods, .. } => half_inst_requirement(*func, *ty, *mods),
        Op::Cvt { func, .. } => vec![cvt_requirement(*func)],
        Op::StMatrix { form: StMatrixForm::M8n8B16, .. } => vec![need(90, 78, "stmatrix (StMatrix8x8x16bOp)")],
        Op::StMatrix { form: StMatrixForm::M16n8B8, .. } => {
            vec![specific("stmatrix.m16n8 with .b8 (StMatrix16x8x8bOp)", LDMATRIX_B8)]
        }
        Op::SetMaxNReg { .. } => {
            vec![specific("setmaxnreg (setmaxregister_increase, setmaxregister_decrease)", SETMAXNREG)]
        }
        // The .u8.s8 and .s8.u8 mixes need PTX ISA 8.4.
        Op::Wgmma { kind, .. } if kind.is_integer() && kind.a != kind.b => {
            vec![specific("wgmma.mma_async with mixed s8/u8 operands", SM90A_PTX84)]
        }
        Op::Wgmma { .. } => vec![specific("wgmma.mma_async (warpgroup MMA)", SM90A_ONLY)],
        Op::WgmmaFence { .. } => vec![specific("wgmma.fence (warpgroup.fence)", SM90A_ONLY)],
        Op::WgmmaCommitGroup => vec![specific("wgmma.commit_group (warpgroup.commit_group)", SM90A_ONLY)],
        Op::Tcgen05 { inst: Tcgen05Inst::CheckAddress, .. } => vec![],
        Op::Tcgen05 { inst, exclusive: true, .. } => vec![specific(
            if *inst == Tcgen05Inst::Alloc { "tcgen05.alloc.exclusive" } else { "tcgen05.dealloc.exclusive" },
            TCGEN05_EXCLUSIVE,
        )],
        // The same targets as tcgen05.alloc.
        // Tcgen05.ld{.red}.spcompress (PTX 9.4, sm_107a).
        Op::Tcgen05Ld { sp: Some(_), red, .. } => vec![specific(
            if red.is_some() {
                "tcgen05.ld.red.spcompress (tcgen05.LdRedSPCompress32x32bOp)"
            } else {
                "tcgen05.ld.spcompress (tcgen05.LdSPCompress32x32bOp)"
            },
            TCGEN05_LD_SPCOMPRESS,
        )],
        Op::Tcgen05Ld { red: Some(_), .. } => {
            vec![specific("tcgen05.ld.red (a TMEM load-reduce copy atom)", TCGEN05_LD_RED)]
        }
        Op::Tcgen05Ld { .. } => vec![specific("tcgen05.ld (a TMEM load copy atom)", TCGEN05_ALLOC)],
        Op::Tcgen05St { .. } => vec![specific("tcgen05.st (a TMEM store copy atom)", TCGEN05_ALLOC)],
        // The same targets (sm_100a, sm_110a; the f families from PTX 8.8).
        // Kind::tf32 and kind::f8f6f4 likewise; kind::i8 on the
        // `a` targets only.
        // The Rubin forms, whatever their kind.
        Op::Tcgen05Mma { kind: UmmaKind { sm107: Some(r), kind, .. }, .. } => vec![specific(
            match (kind, r.wide_k) {
                (UmmaDataKind::F8F6F4, true) => "tcgen05.mma.kind::f8f6f4 with K = 64 (tcgen05.SM107MmaFP8Op)",
                (UmmaDataKind::F8F6F4, false) => {
                    "tcgen05.mma.kind::f8f6f4 with collector usage (tcgen05.SM107MmaFP8Op)"
                }
                (UmmaDataKind::Mxf8f6f4, _) => {
                    "tcgen05.mma.kind::mxf8f6f4.block_scale with K = 64 (tcgen05.SM107BlockScaledMmaMXF8F6F4Op)"
                }
                _ => "tcgen05.mma.kind::mxf4nvf4.block_scale with K = 128 (tcgen05.SM107MmaMXF4NVF4Op)",
            },
            TCGEN05_MMA_SM107,
        )],
        // Scale-input-d (sm_100a from PTX 8.6, the sm_100f family from 8.8)
        // and the weight-stationary i8 form (architecture-specific targets only, as kind::i8).
        Op::Tcgen05Mma { kind, .. } if kind.prims.scale_input_d.is_some() => {
            vec![specific("tcgen05.mma with scale-input-d (prims::tcgen05_mma)", TCGEN05_MMA_SCALE_INPUT_D)]
        }
        Op::Tcgen05Mma { kind, .. } if kind.prims.ws && kind.kind == UmmaDataKind::I8 => {
            vec![specific("tcgen05.mma.ws.kind::i8 (prims::tcgen05_mma_ws)", TCGEN05_MMA_I8)]
        }
        Op::Tcgen05Mma { kind, .. } if kind.prims.ws => {
            vec![specific("tcgen05.mma.ws (prims::tcgen05_mma_ws)", TCGEN05_ALLOC)]
        }
        Op::Tcgen05Mma { kind, .. } => match kind.kind {
            UmmaDataKind::I8 => vec![specific("tcgen05.mma.kind::i8 (tcgen05.MmaI8Op)", TCGEN05_MMA_I8)],
            UmmaDataKind::Tf32 => vec![specific("tcgen05.mma.kind::tf32 (tcgen05.MmaTF32Op)", TCGEN05_ALLOC)],
            UmmaDataKind::F8F6F4 => {
                vec![specific("tcgen05.mma.kind::f8f6f4 (tcgen05.MmaF8F6F4Op)", TCGEN05_ALLOC)]
            }
            UmmaDataKind::F16 => vec![specific(
                if kind.two_cta { "tcgen05.mma.cta_group::2" } else { "tcgen05.mma (cute.gemm)" },
                TCGEN05_ALLOC,
            )],
            // m10-umma-blockscaled.
            UmmaDataKind::Mxf8f6f4 => {
                vec![specific("tcgen05.mma.kind::mxf8f6f4.block_scale (tcgen05.MmaMXF8F6F4Op)", TCGEN05_MMA_MXF8F6F4)]
            }
            _ if kind.scale.is_some_and(|s| s.k96) => vec![specific(
                if kind.kind == UmmaDataKind::Mxf4 {
                    "tcgen05.mma.kind::mxf4.block_scale.block32 with K = 96 (tcgen05.SM103MmaMXF4Op)"
                } else {
                    "tcgen05.mma.kind::mxf4nvf4.block_scale.block16 with K = 96 (tcgen05.SM103MmaMXF4NVF4Op)"
                },
                TCGEN05_MMA_K96,
            )],
            UmmaDataKind::Mxf4 => {
                vec![specific("tcgen05.mma.kind::mxf4.block_scale.scale_vec::2X (tcgen05.MmaMXF4Op)", TCGEN05_MMA_MXF4)]
            }
            UmmaDataKind::Mxf4Nvf4 => vec![specific(
                "tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X (tcgen05.MmaMXF4NVF4Op)",
                TCGEN05_MMA_MXF4NVF4,
            )],
        },
        Op::Tcgen05Commit { .. } => vec![specific("tcgen05.commit (tcgen05::commit)", TCGEN05_ALLOC)],
        // The same targets.
        Op::Tcgen05Cp { .. } => vec![specific("tcgen05.cp (an SMEM-to-TMEM copy atom, tcgen05.Cp..Op)", TCGEN05_ALLOC)],
        // m10-clc-scheduling.
        Op::ClcTryCancel { multicast: false, .. } => vec![need(100, 86, "clusterlaunchcontrol.try_cancel")],
        Op::ClcTryCancel { multicast: true, .. } => {
            vec![specific("clusterlaunchcontrol.try_cancel.multicast::cluster::all (issue_clc_query)", CLC_MULTICAST)]
        }
        Op::ClcQueryCancel { .. } => vec![need(100, 86, "clusterlaunchcontrol.query_cancel (clc_response)")],
        Op::Tcgen05 { inst, .. } => vec![specific(
            match inst {
                Tcgen05Inst::Alloc => "tcgen05.alloc (alloc_tmem)",
                Tcgen05Inst::Dealloc => "tcgen05.dealloc (dealloc_tmem)",
                Tcgen05Inst::WaitLd => "tcgen05.wait::ld (fence_view_async_tmem_load)",
                Tcgen05Inst::WaitSt => "tcgen05.wait::st (fence_view_async_tmem_store)",
                Tcgen05Inst::FenceBeforeThreadSync => "tcgen05.fence::before_thread_sync",
                Tcgen05Inst::FenceAfterThreadSync => "tcgen05.fence::after_thread_sync",
                _ => "tcgen05.relinquish_alloc_permit (relinquish_tmem_alloc_permit)",
            },
            TCGEN05_ALLOC,
        )],
        Op::Multimem { op, .. } => vec![extern_calls::multimem_requirement(op)],
        Op::WgmmaWait { .. } => vec![specific("wgmma.wait_group (warpgroup.wait_group)", SM90A_ONLY)],
        Op::LdMatrix { form, .. } if *form != LdMatrixForm::M8n8B16 => vec![specific(
            match form.unpack() {
                None => "ldmatrix.m16n16 with .b8 (LdMatrix16x16x8bOp or LdMatrix16x8x8bOp)",
                Some(_) => "ldmatrix unpacking 4- or 6-bit elements (unpack_bits)",
            },
            LDMATRIX_B8,
        )],
        _ => vec![],
    };
    let ptx = needs.iter().flatten().map(|r| r.ptx).max()?;
    // An architecture-specific need decides which targets qualify.
    if let Some(&specific) = needs.iter().flatten().find(|r| r.arch_specific.is_some()) {
        return Some(TargetRequirement { ptx, ..specific });
    }
    needs.sort_by_key(|r| r.map(|r| std::cmp::Reverse(r.sm)));
    let first = needs.into_iter().flatten().next()?;
    Some(TargetRequirement { ptx, ..first })
}

/// Check that `ir` can be compiled for `arch`, and return the PTX ISA
/// version to compile it with: the architecture's own, or the newer one an
/// instruction needs.
///
/// The shared memory sized at build time must fit the target too
/// ([`Arch::smem_capacity_bytes`]); the dynamic region's bytes are checked
/// at launch.
pub fn check_target(ir: &KernelIr, arch: Arch) -> std::result::Result<u32, ValidationError> {
    let ptx = check_target_needs(ir, arch)?.0.max(arch.min_ptx());
    let (bytes, capacity) = (ir.shared_memory_bytes(), arch.smem_capacity_bytes());
    if bytes > capacity {
        return Err(ValidationError {
            message: format!(
                "the kernel uses {bytes} bytes of shared memory, more than the {capacity} bytes ({} KiB) a block \
                 can have on {}",
                capacity / 1024,
                arch.chip()
            ),
            loc: Some(ir.loc),
        });
    }
    // A hidden descriptor in a sub-byte (FP4/FP6) format needs SM100's
    // tensor copies (and sm_120a's copies into shared::cluster, which cute-rs issues, lack them).
    if let Some(t) =
        ir.tensor_maps.iter().filter_map(|m| m.source.as_ref().map(|s| s.spec.data_type)).find(|t| t.is_packed())
        && !(100..120).contains(&arch.sm())
    {
        return Err(ValidationError {
            message: format!(
                "a TMA atom moves FP4/FP6 values in the {t} tensor-map format, which needs sm_100 to sm_11x (the \
                 PTX ISA's .b4x16, .b4x16_p64 and .b6x16_p32 types), not {}",
                arch.chip()
            ),
            loc: Some(ir.loc),
        });
    }
    // The 128-byte swizzle of 32-byte atoms is an SM100
    // tensor-map mode.
    if let Some(sw) =
        ir.tensor_maps.iter().filter_map(|m| m.source.as_ref().map(|s| s.spec.swizzle)).find(|s| s.is_atom_mode())
        && arch.sm() < 100
    {
        return Err(ValidationError {
            message: format!(
                "a TMA atom's descriptor uses the {sw} swizzle, an SM100 tensor-map mode, which needs sm_100 or \
                 newer, not {}",
                arch.chip()
            ),
            loc: Some(ir.loc),
        });
    }
    Ok(ptx)
}

/// [`check_target`], returning the PTX ISA version the instructions need
/// (0 if none needs one) and the instruction that needs the newest, with
/// its source line, for messages about a version a toolchain or driver
/// lacks.
pub fn check_target_needs(ir: &KernelIr, arch: Arch) -> std::result::Result<(u32, Option<String>), ValidationError> {
    fn walk(
        ir: &KernelIr,
        r: &Region,
        arch: Arch,
        ptx: &mut u32,
        why: &mut Option<String>,
    ) -> std::result::Result<(), ValidationError> {
        for inst in &r.insts {
            let before = *ptx;
            // A 32-byte vector (to global memory; the validator checks) is
            // PTX's `ld.global.v8.b32` or `.v4.b64` on sm_100 and newer,
            // from PTX ISA 8.8, which LLVM emits only from that version; an
            // older target gets two 16-byte accesses.
            if let Op::Load { results: vals, .. } | Op::Store { values: vals, .. } = &inst.op
                && vals.len() * ir.dtype(vals[0]).bytes() == 32
                && arch.sm() >= 100
            {
                *ptx = (*ptx).max(88);
                if *ptx > before {
                    *why = Some(format!("a 32-byte vector access at {}", at(inst.loc)));
                }
            }
            let before = *ptx;
            if let Some(need) = target_requirement(&inst.op) {
                if let Some(targets) = need.arch_specific {
                    let Some(on_arch) = targets.ptx_on(arch) else {
                        return Err(ValidationError {
                            message: format!(
                                "{} needs {}; the kernel is being compiled for {}",
                                need.what,
                                targets.describe(),
                                arch.chip()
                            ),
                            loc: Some(inst.loc),
                        });
                    };
                    *ptx = (*ptx).max(on_arch);
                } else if need.sm > arch.sm() {
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
                if let Some((sm, older)) = need.older_ptx
                    && arch.sm() < sm
                {
                    *ptx = (*ptx).max(older);
                }
                if *ptx > before {
                    *why = Some(format!("{} at {}", need.what, at(inst.loc)));
                }
            }
            for region in inst.op.regions() {
                walk(ir, region, arch, ptx, why)?;
            }
        }
        Ok(())
    }
    let (mut ptx, mut why) = (0, None);
    walk(ir, &ir.body, arch, &mut ptx, &mut why)?;
    Ok((ptx, why))
}

/// Check that `ir` is well formed.
pub fn validate(ir: &KernelIr) -> std::result::Result<(), ValidationError> {
    let n = ir.value_types.len();
    let layout = ir.shared_memory_layout();
    let block_bytes = layout.block_bytes(0);
    let mut c = Checker {
        ir,
        visible: vec![false; n],
        defined: vec![false; n],
        loc: None,
        constants: HashMap::new(),
        smem: layout,
    };
    let mut scope = Vec::new();
    for p in &ir.args {
        match p {
            Param::Buffer(b) => {
                if ir.buffer(*b).space != MemSpace::Global {
                    return Err(c.err("buffer parameters must be global memory"));
                }
            }
            Param::Scalar { value, .. } => c.define(*value, &mut scope)?,
            Param::ByValue { .. } => {}
        }
    }
    let constant: usize = ir
        .globals
        .iter()
        .map(|g| ir.buffer(g.buffer))
        .filter(|d| d.space == MemSpace::Constant)
        .map(|d| d.len.unwrap_or(0) * d.elem.bytes())
        .sum();
    if constant > crate::MAX_CONSTANT_BYTES {
        return Err(c.err(format!(
            "the kernel's constant-memory globals hold {constant} bytes; a module has at most {} (64 KB) of \
             constant memory",
            crate::MAX_CONSTANT_BYTES
        )));
    }
    // Module globals share the PTX module's namespace with the kernel, the shared buffers
    // (`smem0`, ...) and the symbols the lowering adds.
    for g in &ir.globals {
        let name = &ir.buffer(g.buffer).name;
        let clash = *name == ir.name
            || name == crate::ir::DYNAMIC_SMEM_SYMBOL
            || ["vprintf", "__assertfail", "malloc", "free"].contains(&name.as_str())
            || name.starts_with("printfFormat_")
            || ir.buffers.iter().any(|b| b.space == MemSpace::Shared && b.name == *name);
        if clash {
            return Err(c.err(format!(
                "module global `{name}` has the name of another symbol of the kernel's module (the kernel, a \
                 shared buffer or a symbol the compiler adds): choose another name"
            )));
        }
    }
    if ir.buffers.iter().filter(|b| b.space == MemSpace::Shared && b.len.is_none()).count() > 1 {
        return Err(c.err("a kernel has at most one dynamic shared-memory region (get_dyn_smem)"));
    }
    // The dynamic region's bytes are only known at launch (check_launch adds them).
    if block_bytes > MAX_SHARED_BYTES {
        return Err(c.err(format!(
            "kernel uses {block_bytes} bytes of shared memory; no GPU gives a block more than {MAX_SHARED_BYTES}"
        )));
    }
    c.externs_and_function()?;
    c.region(&ir.body, &[])?;
    wgmma::check_accumulators(ir)
}

/// What the validator knows about the carry flag at a point of a region.
#[derive(Clone, Copy)]
enum CarryFlag {
    /// Nothing in the region has set it yet.
    Unset,
    /// Set by the carry-out instruction (its name) at this location.
    Set((&'static str, Loc)),
    /// Set there, but lost since to the operation described at the second
    /// location.
    Lost { set: (&'static str, Loc), by: (&'static str, Loc) },
}

/// Whether the carry instruction named `name` (a [`CarryFn::cute_name`])
/// is a subtraction, whose flag holds a borrow.
fn borrows(name: &str) -> bool {
    name.starts_with("sub")
}

/// `file:line:column`, for messages.
fn at(loc: Loc) -> String {
    format!("{}:{}:{}", loc.file(), loc.line(), loc.column())
}

/// What a [`CarryFn`] needs beyond sm_80 and PTX 7.0: nothing. The 32-bit
/// adds and subtracts exist on every target (PTX 1.2), the multiply-adds and
/// the 64-bit forms from sm_20 (PTX 3.0 and 4.3). The match is exhaustive so
/// that a new function has to say.
fn carry_inst_requirement(func: CarryFn) -> Option<TargetRequirement> {
    match func.op {
        CarryOp::Add | CarryOp::Sub | CarryOp::MadLo => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A requirement is a set of alternatives: exact chips, families and
    /// baselines, each with its own PTX ISA version; a target gets the
    /// lowest of those it satisfies, and the message lists them all.
    #[test]
    fn requirement_alternatives() {
        // The exact-chip kind: sm_90a only, never plain sm_90 or later chips.
        assert_eq!(SM90A_ONLY.ptx_on(Arch::Sm90a), Some(80));
        for arch in Arch::ALL.into_iter().filter(|&a| a != Arch::Sm90a) {
            assert_eq!(SM90A_ONLY.ptx_on(arch), None, "{arch}");
        }
        assert_eq!(SM90A_ONLY.describe(), "sm_90a");
        // setmaxnreg: chips, then families from a later PTX.
        let on = |a| SETMAXNREG.ptx_on(a);
        assert_eq!([on(Arch::Sm90a), on(Arch::Sm100a), on(Arch::Sm120a)], [Some(80), Some(86), Some(86)]);
        assert_eq!(
            [on(Arch::Sm103a), on(Arch::Sm103f), on(Arch::Sm121f), on(Arch::Sm110a)],
            [Some(88), Some(88), Some(88), Some(90)]
        );
        assert_eq!([on(Arch::Sm90), on(Arch::Sm100), on(Arch::Sm107)], [None, None, None]);
        // The FP4/FP6/UE8M0 conversions of m3: 8.6 on sm_100a and sm_101a
        // (sm_110a's old name), 8.7 on sm_120a, 8.8 on the families.
        let ue = |a| UE8M0X2.ptx_on(a);
        assert_eq!([ue(Arch::Sm100a), ue(Arch::Sm110a), ue(Arch::Sm120a)], [Some(86), Some(86), Some(87)]);
        assert_eq!([ue(Arch::Sm100f), ue(Arch::Sm103a), ue(Arch::Sm107f), ue(Arch::Sm121a)], [Some(88); 4]);
        // sm_110f has it as sm_101f did, from 8.8 (sm_110f itself needs 9.0).
        assert_eq!([ue(Arch::Sm110f), ue(Arch::Sm100), ue(Arch::Sm90a)], [Some(88), None, None]);
        // A baseline alternative beside exact chips.
        let mixed = ArchSpecific { chips: &[(90, 80)], families: &[], baselines: &[(100, 86)] };
        assert_eq!(
            [mixed.ptx_on(Arch::Sm90a), mixed.ptx_on(Arch::Sm90), mixed.ptx_on(Arch::Sm103)],
            [Some(80), None, Some(86)]
        );
        assert_eq!(mixed.ptx_on(Arch::Sm121f), Some(86));
        assert_eq!(mixed.describe(), "sm_90a, or sm_100 or newer with PTX 8.6");
    }

    #[test]
    fn families_follow_the_ptx_isa_table() {
        assert_eq!([100, 103, 107].map(arch_family), [10, 10, 10]);
        assert_eq!([110, 101].map(arch_family), [11, 11], "sm_101 is sm_110's name before PTX ISA 9.0");
        assert_eq!([120, 121].map(arch_family), [12, 12]);
        assert_eq!([101, 110, 100, 103].map(current_number), [110, 110, 100, 103], "sm_101a is sm_110a");
    }

    /// Float32 `redux.sync` is on sm_100a only of today's targets; neither
    /// sm_100 nor another `a` target inherits it.
    #[test]
    fn arch_specific_features_are_not_inherited() {
        assert_eq!(REDUX_F32.ptx_on(Arch::Sm100a), Some(86));
        for arch in [Arch::Sm80, Arch::Sm90, Arch::Sm90a, Arch::Sm100, Arch::Sm120, Arch::Sm120a] {
            assert_eq!(REDUX_F32.ptx_on(arch), None, "{arch:?}");
        }
        assert_eq!((REDUX_F32.oldest(), REDUX_F32.min_ptx()), (100, 86));
        assert_eq!(REDUX_F32.describe(), "sm_100a (or the sm_100f family with PTX 8.8)");
        assert!(Arch::Sm90a.is_arch_specific() && Arch::Sm100a.is_arch_specific() && Arch::Sm120a.is_arch_specific());
        assert!(!Arch::Sm90.is_arch_specific() && !Arch::Sm100.is_arch_specific() && !Arch::Sm89.is_arch_specific());
    }

    /// Several targets and families, as the s2f6x2 conversion lists them,
    /// read as one sentence; a family whose features arrive with different
    /// PTX versions names each.
    #[test]
    fn arch_specific_targets_are_described() {
        let s2f6 = ArchSpecific { chips: &[(100, 91), (103, 91), (110, 91), (120, 91)], families: &[], baselines: &[] };
        assert_eq!(s2f6.describe(), "sm_100a, sm_103a, sm_110a or sm_120a");
        assert_eq!(s2f6.ptx_on(Arch::Sm120a), Some(91));
        assert_eq!(s2f6.ptx_on(Arch::Sm90a), None);
        let families = ArchSpecific { chips: &[], families: &[(100, 88), (110, 88), (120, 88)], baselines: &[] };
        assert_eq!(families.describe(), "the sm_100f, sm_110f or sm_120f family with PTX 8.8");
        // A later `a` target of a family has the family's features.
        assert_eq!(families.ptx_on(Arch::Sm100a), Some(88));
        assert_eq!(families.ptx_on(Arch::Sm120a), Some(88));
        assert_eq!(families.ptx_on(Arch::Sm100), None, "a baseline target has no family features");
        assert_eq!(families.ptx_on(Arch::Sm90a), None);
        let mixed = ArchSpecific { chips: &[(100, 86)], families: &[(100, 88), (120, 90)], baselines: &[] };
        assert_eq!(mixed.describe(), "sm_100a (or the sm_100f family with PTX 8.8 or the sm_120f family with PTX 9.0)");
        assert_eq!(mixed.ptx_on(Arch::Sm100a), Some(86), "the lowest version that has it");
        assert_eq!(mixed.ptx_on(Arch::Sm120a), Some(90));
    }

    /// The `.s2f6x2` conversions are on the six `a` targets the PTX ISA's
    /// target notes list, from PTX 9.1, and no family: of today's targets,
    /// sm_100a and sm_120a.
    #[test]
    fn s2f6x2_targets_follow_the_ptx_isa() {
        assert_eq!(S2F6X2.describe(), "sm_100a, sm_103a, sm_107a, sm_110a, sm_120a or sm_121a");
        assert_eq!((S2F6X2.oldest(), S2F6X2.min_ptx()), (100, 91));
        assert_eq!(S2F6X2.ptx_on(Arch::Sm100a), Some(91));
        assert_eq!(S2F6X2.ptx_on(Arch::Sm120a), Some(91));
        for arch in [Arch::Sm80, Arch::Sm89, Arch::Sm90, Arch::Sm90a, Arch::Sm100, Arch::Sm120] {
            assert_eq!(S2F6X2.ptx_on(arch), None, "{arch:?}");
        }
    }
}
