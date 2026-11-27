//! Carry-chain arithmetic and inline-PTX support (`AsmInput`, `AsmSim`, constraint parsing).

use super::*;

/// Simulates an inline PTX fragment: raw input bits to raw output bits.
pub type InlineSim = fn(&[u64]) -> Vec<u64>;

/// The arithmetic of an [`Op::CarryInst`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CarryOp {
    /// `a + b` (`add.cc`, `addc`).
    Add,
    /// `a - b` (`sub.cc`, `subc`).
    Sub,
    /// The low half of `a * b`, plus `c` (`mad.lo.cc`, `madc.lo`).
    MadLo,
}

/// One of PTX's extended-precision instructions, which [`Op::CarryInst`]
/// issues: an add, a subtract or a multiply-add that reads the carry flag
/// (`CC.CF`), writes it, or both. A carry-in adds the flag (a borrow-in
/// subtracts it); a carry-out sets it to the carry (borrow) out of the
/// operation. One of the two is always there: without either, the
/// instruction would be plain arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CarryFn {
    pub op: CarryOp,
    /// Reads the flag: `addc`, `subc`, `madc`.
    pub carry_in: bool,
    /// Writes the flag: `.cc`.
    pub carry_out: bool,
}

impl CarryFn {
    /// The name of the operation: CuTe DSL's (`cute.arch.add_cc`, `addc`,
    /// `sub_cc`, `subc`, `mad_cc`, `madc`), or cute-rs's for the forms CuTe
    /// DSL lacks (`addc_cc`, `subc_cc`, `madc_cc`).
    pub fn cute_name(self) -> &'static str {
        match (self.op, self.carry_in, self.carry_out) {
            (CarryOp::Add, false, _) => "add_cc",
            (CarryOp::Add, true, false) => "addc",
            (CarryOp::Add, true, true) => "addc_cc",
            (CarryOp::Sub, false, _) => "sub_cc",
            (CarryOp::Sub, true, false) => "subc",
            (CarryOp::Sub, true, true) => "subc_cc",
            (CarryOp::MadLo, false, _) => "mad_cc",
            (CarryOp::MadLo, true, false) => "madc",
            (CarryOp::MadLo, true, true) => "madc_cc",
        }
    }

    /// Number of operands.
    pub fn arity(self) -> usize {
        match self.op {
            CarryOp::Add | CarryOp::Sub => 2,
            CarryOp::MadLo => 3,
        }
    }
}

/// The PTX instruction an [`Op::CarryInst`] issues on operands of type `ty`,
/// as CuTe DSL spells it: `add.cc.u32`, `addc.cc.u64`, `subc.u32`,
/// `mad.lo.cc.s32`, `madc.lo.u64`. The adds and subtracts are `.u` for
/// either signedness (the PTX ISA: "behavior is the same for unsigned and
/// signed integers"); the multiply-adds follow the type, which only names
/// the instruction, since the low half of a product and the carry out of
/// the addition do not depend on it.
pub fn carry_inst_name(func: CarryFn, ty: DType) -> String {
    let w = ty.bits();
    let c = if func.carry_in { "c" } else { "" };
    let cc = if func.carry_out { ".cc" } else { "" };
    match func.op {
        CarryOp::Add => format!("add{c}{cc}.u{w}"),
        CarryOp::Sub => format!("sub{c}{cc}.u{w}"),
        CarryOp::MadLo => format!("mad{c}.lo{cc}.{}{w}", if ty.is_signed() { 's' } else { 'u' }),
    }
}

/// Why `op` loses the carry flag of a chain around it, if it does. PTX
/// keeps the flag only through straight-line code ("not preserved across
/// calls ... mainly intended for use in straight-line code sequences"), so
/// what ends that code in the PTX loses it: a runtime `if` or loop, a call
/// (`printf`), a hand-written MLIR operation (which may branch or call),
/// inline PTX that branches, calls or exits, and a 64-bit integer division
/// or remainder by a value that is not a constant, which LLVM (from O1 on)
/// lowers with a branch to a 32-bit division for when both operands fit
/// (its `BypassSlowDivision`; a constant divisor becomes a multiplication).
/// Inline PTX that uses the flag itself changes it behind the chain's back.
/// The validator rejects a carry-in after any of these, and the simulator
/// forgets the flag there. The warp collectives keep it: each is one
/// convergent instruction (`shfl.sync`, also with a runtime segment mask
/// and clamp, `vote.sync`, `match.{any,all}.sync`, `redux.sync`,
/// `bar.warp.sync` with any member mask) that neither branches nor touches
/// the flag. So do the special-register reads, at most one `mov` each: the
/// thread, block and lane indices and the block and grid sizes through the
/// GPU dialect, `%globaltimer_lo` as inline PTX with side effects, the
/// shared-memory size registers as pure inline PTX (after which
/// `dynamic_smem_size` subtracts the region's start, when it is not 0, with
/// ordinary arithmetic), and the rest through NVVM. So does
/// `cp.async.mbarrier.arrive` (one instruction, from an LLVM intrinsic).
/// `dtype` gives a value's type and `is_constant` whether a constant defines
/// it.
pub fn carry_flag_breaker(
    op: &Op,
    dtype: impl Fn(ValueId) -> DType,
    is_constant: impl Fn(ValueId) -> bool,
) -> Option<&'static str> {
    match op {
        Op::If { .. } => Some("runtime if"),
        Op::For { .. } | Op::While { .. } => Some("runtime loop"),
        Op::Printf { .. } => Some("printf (a call)"),
        Op::Call { .. } => Some("a call of an external device function"),
        Op::Assert { .. } => Some("device assertion (a call)"),
        Op::MlirOp { .. } => Some("hand-written MLIR operation (which may branch or call)"),
        Op::InlinePtx { template, .. } => inline_ptx_carry_breaker(template),
        // NVPTX lowers 128-bit arithmetic with the extended-precision
        // instructions (and its division with branches).
        Op::Binary { lhs: x, .. } | Op::Unary { x, .. } | Op::Cmp { lhs: x, .. } if dtype(*x).bits() == 128 => {
            Some("128-bit integer arithmetic (which LLVM lowers with add.cc and addc)")
        }
        Op::Binary { op: BinaryOp::Div | BinaryOp::DivU | BinaryOp::Rem | BinaryOp::RemU, rhs, .. }
            if dtype(*rhs).is_int() && dtype(*rhs).bits() == 64 && !is_constant(*rhs) =>
        {
            Some("64-bit integer division or remainder by a runtime value (which LLVM lowers with a branch)")
        }
        Op::Shuffle { .. } | Op::Vote { .. } | Op::Match { .. } | Op::Redux { .. } | Op::WarpBarrier { .. } => None,
        Op::Special { .. } | Op::CpAsyncMbarrierArrive { .. } => None,
        // Memory accesses, ordered or hinted, prefetches, packing and the
        // half-precision and conversion instructions: not covered by the
        // documented guarantee above, so they are listed here, not claimed.
        Op::Load { .. } | Op::Store { .. } | Op::Prefetch { .. } => None,
        Op::Pack { .. } | Op::Unpack { .. } | Op::HalfInst { .. } | Op::Cvt { .. } => None,
        Op::Assume { .. } => None,
        Op::Red { .. } => None,
        _ => None,
    }
}

/// Why hand-written PTX loses the carry flag of a chain around it, if it
/// does: it uses one of the extended-precision instructions (`add.cc`,
/// `addc`, `sub.cc`, `subc`, `mad{.lo,.hi}.cc`, `madc`), the only
/// instructions that touch the flag, or it branches, calls, returns or exits
/// (`bra`, `brx.idx`, `call`, `ret`, `exit`), which ends the straight-line
/// code that PTX keeps the flag through.
fn inline_ptx_carry_breaker(template: &str) -> Option<&'static str> {
    let tokens = || template.split(|c: char| c.is_whitespace() || matches!(c, ';' | '{' | '}' | '"'));
    let opcode = |token: &str| token.split('.').next().unwrap_or_default().to_string();
    let uses_flag = tokens().any(|token| {
        matches!(opcode(token).as_str(), "addc" | "subc" | "madc")
            || (matches!(opcode(token).as_str(), "add" | "sub" | "mad") && token.split('.').any(|part| part == "cc"))
    });
    if uses_flag {
        return Some("inline PTX that uses the carry flag");
    }
    let jumps = tokens().any(|token| matches!(opcode(token).as_str(), "bra" | "brx" | "call" | "ret" | "exit"));
    jumps.then_some("inline PTX that branches, calls or exits")
}

/// An input of an [`Op::InlinePtx`] statement or an operand of an
/// [`Op::MlirOp`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AsmInput {
    /// A value, in a register.
    Value(ValueId),
    /// The address of element `index` (counting `elem`s) of a buffer, as a
    /// 64-bit integer: an address in the global state space for global
    /// memory (where it is also the generic address), in the shared state
    /// space for shared memory, and the generic address of a local
    /// (per-thread) buffer.
    Address { buffer: BufferId, index: ValueId, elem: DType },
}

impl AsmInput {
    /// The SSA value the input reads: the value itself, or an address's
    /// index.
    pub fn value(&self) -> ValueId {
        match *self {
            AsmInput::Value(v) | AsmInput::Address { index: v, .. } => v,
        }
    }
}

/// Simulates an inline assembly statement or a hand-written MLIR operation
/// in one thread: given the raw bits of its read-only operands (0 for an
/// address, which the simulator cannot follow) and, mutably, of its
/// read-write operands, it updates the read-write operands and returns the
/// write-only results, as raw bits (zero-extended to 64; extra high bits are
/// ignored).
#[derive(Clone)]
pub struct AsmSim(pub std::sync::Arc<AsmSimFn>);

/// The function an [`AsmSim`] holds.
pub type AsmSimFn = dyn Fn(&[u64], &mut [u64]) -> Vec<u64> + Send + Sync;

impl AsmSim {
    pub fn new(f: impl Fn(&[u64], &mut [u64]) -> Vec<u64> + Send + Sync + 'static) -> AsmSim {
        AsmSim(std::sync::Arc::new(f))
    }

    /// The simulation of the positional inline PTX functions: inputs to
    /// outputs.
    pub fn from_inline(f: InlineSim) -> AsmSim {
        AsmSim::new(move |reads, _| f(reads))
    }

    /// Run it.
    pub fn call(&self, reads: &[u64], read_writes: &mut [u64]) -> Vec<u64> {
        (self.0)(reads, read_writes)
    }
}

impl fmt::Debug for AsmSim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AsmSim(..)")
    }
}

/// One entry of an LLVM inline-assembly constraint string.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AsmConstraint {
    /// An output, with the code after its `=` (`r`, `&l`, ...).
    Output(String),
    /// An input in a register or an immediate (`r`, `l`, `n`, ...).
    Input(String),
    /// An input tied to (in the same register as) the output with this
    /// number.
    Tied(usize),
    /// A clobber, with the name in its braces (`memory`).
    Clobber(String),
    /// A read-write operand (`+r`, the code after `+`): GCC's spelling,
    /// which LLVM IR lacks, as `InlineAsm` takes it.
    ReadWrite(String),
}

/// The entries of an LLVM constraint string, split at its commas and
/// trimmed of spaces (none for an empty string).
pub fn asm_constraints(constraints: &str) -> Vec<AsmConstraint> {
    if constraints.trim().is_empty() {
        return Vec::new();
    }
    constraints
        .split(',')
        .map(|e| {
            let e = e.trim();
            if let Some(code) = e.strip_prefix('=') {
                AsmConstraint::Output(code.to_string())
            } else if let Some(code) = e.strip_prefix('+') {
                AsmConstraint::ReadWrite(code.to_string())
            } else if let Some(name) = e.strip_prefix('~') {
                AsmConstraint::Clobber(name.trim_start_matches('{').trim_end_matches('}').to_string())
            } else if !e.is_empty() && e.bytes().all(|b| b.is_ascii_digit()) {
                AsmConstraint::Tied(e.parse().unwrap_or(usize::MAX))
            } else {
                AsmConstraint::Input(e.to_string())
            }
        })
        .collect()
}

/// The NVPTX inline-assembly register class of a type: its constraint code
/// (`b` predicate, `c` 8-bit, `h` 16-bit, `r` 32-bit and `l` 64-bit
/// integers, `f` and `d` floats; Float16 and BFloat16 in `h`).
pub fn asm_register(ty: DType) -> &'static str {
    match ty {
        DType::Bool => "b",
        DType::I8 | DType::U8 | DType::I4 => "c",
        t if t.is_narrow_float() || t == DType::F4E2M1FNx2 => "c",
        DType::I16 | DType::U16 | DType::F16 | DType::BF16 => "h",
        DType::I32 | DType::U32 | DType::F6E2M3FNx4 | DType::F6E3M2FNx4 => "r",
        // 128-bit integers have no register class: the validator refuses
        // them as inline-assembly operands.
        DType::I64 | DType::U64 | DType::I128 | DType::U128 => "l",
        DType::F32 | DType::TF32 => "f",
        _ => "d",
    }
}

/// Whether a constraint string clobbers memory (`~{memory}`).
pub fn asm_clobbers_memory(constraints: &str) -> bool {
    asm_constraints(constraints).contains(&AsmConstraint::Clobber("memory".into()))
}

/// The position of an empty entry of a constraint string (`"=r,,r"`, a
/// trailing comma, or `=`, `+` or `~{}` alone), which LLVM rejects.
pub fn asm_empty_constraint(constraints: &str) -> Option<usize> {
    asm_constraints(constraints).iter().position(|e| match e {
        AsmConstraint::Output(c)
        | AsmConstraint::Input(c)
        | AsmConstraint::Clobber(c)
        | AsmConstraint::ReadWrite(c) => c.is_empty(),
        AsmConstraint::Tied(_) => false,
    })
}

/// The operand numbers an inline-assembly template refers to (`$3`, `${3}`,
/// `${3:modifier}`), in order, as LLVM reads them in the AT&T dialect: `$$`
/// is a literal `$`, and `$(`, `$|`, `$)` and `${:...}` are not operands.
/// Any other `$` is an error, which LLVM would only report by aborting.
pub fn asm_operand_refs(template: &str) -> Result<Vec<usize>, String> {
    let bytes = template.as_bytes();
    let mut refs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'$' {
            i += 1;
            continue;
        }
        i += 1;
        match bytes.get(i) {
            Some(b'$' | b'(' | b'|' | b')') => i += 1,
            Some(b'{') if bytes.get(i + 1) == Some(&b':') => {
                i = template[i..].find('}').map_or(bytes.len(), |end| i + end + 1);
            }
            Some(&c) if c == b'{' || c.is_ascii_digit() => {
                let braced = c == b'{';
                let start = i + braced as usize;
                let end = start + bytes[start..].iter().take_while(|b| b.is_ascii_digit()).count();
                let number = template[start..end].parse().map_err(|_| {
                    format!("`$` must be followed by an operand number (or written `$$`) in {template:?}")
                })?;
                refs.push(number);
                i = end;
                if braced {
                    match template[end..].find('}') {
                        Some(close) => i = end + close + 1,
                        None => return Err(format!("unclosed `${{` in {template:?}")),
                    }
                }
            }
            _ => return Err(format!("`$` must be followed by an operand number (or written `$$`) in {template:?}")),
        }
    }
    Ok(refs)
}
