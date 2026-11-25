//! Hand-written PTX and MLIR, and custom operations built from them.
//!
//! * [`InlinePtx`] is CuTe DSL's `cute.arch.inline_ptx(ptx_code,
//!   write_only_types=, read_only_args=, read_write_args=, predicate=)`:
//!   PTX with named operands, `{$wN}` for the write-only outputs, `{$rN}`
//!   for the read-only inputs (values, or addresses from a [`Ptr`]) and
//!   `{$rwN}` for the read-write operands, each in the register class of its
//!   type, and an optional Boolean predicate.
//! * [`InlineAsm`] is LLVM inline assembly as CuTe DSL's raw
//!   `llvm.inline_asm` builder takes it: positional `$0, $1, ...` operands
//!   with a constraint string written by hand (`"=r,r,~{memory}"`, where
//!   `+r` is a read-write operand).
//! * [`MlirOp`] emits an MLIR operation written as text, for the `nvvm.*`
//!   (or `llvm.*`, `arith.*`, ...) operations cute-rs has no function for.
//!
//! Each takes its results' types as type parameters and returns typed
//! values: nothing for no result, a value for one and a tuple for several.
//! The simulator runs one only when given a Rust closure that computes what
//! it does ([`InlinePtx::sim`]); otherwise the kernel can be compiled but
//! not simulated. LLVM does not treat inline assembly as convergent and may
//! duplicate it into both sides of a branch, so warp- or CTA-collective
//! instructions need dedicated operations.
//!
//! # Custom operations
//!
//! CuTe DSL marks a Python function that builds IR as an operation with
//! `@cutlass.dsl_user_op`, which hands it the caller's source location. In
//! cute-rs a custom operation is an ordinary Rust function marked
//! `#[track_caller]`: every operation built inside it (through cute-rs's own
//! `#[track_caller]` functions and operators) records the location of the
//! function's caller, so the validator's and the simulator's errors, and the
//! line information of the PTX, point at the line that used it.
//!
//! ```
//! use cute::prelude::*;
//!
//! /// `a * b + c` on the low 24 bits of `a` and `b`, as one instruction.
//! #[track_caller]
//! fn mad24_lo(a: Uint32, b: Uint32, c: Uint32) -> Uint32 {
//!     InlinePtx::new("mad24.lo.u32 {$w0}, {$r0}, {$r1}, {$r2};")
//!         .write::<u32>()
//!         .read(a)
//!         .read(b)
//!         .read(c)
//!         .pure()
//!         .sim(|r, _| vec![((r[0] & 0xff_ffff) * (r[1] & 0xff_ffff) + r[2]) & 0xffff_ffff])
//!         .emit()
//! }
//!
//! let kernel = Kernel::build("custom_op", || {
//!     let x = param_tensor::<u32>("x", Layout::col_major(32));
//!     let t = thread_idx().x;
//!     x.store(t, mad24_lo(x.load(t), 3u32.into(), 1u32.into())); // errors and line info name this line
//! });
//! let mut args = vec![KernelArg::buffer(&[5u32; 32])];
//! kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
//! assert_eq!(args[0].to_vec::<u32>()[0], 16);
//! ```
//!
//! Closures cannot be `#[track_caller]`, so the operations a custom
//! operation builds inside the body of an [`if_`](crate::if_) or a loop
//! record their own lines.

use std::marker::PhantomData;
use std::panic::Location;

use crate::builder;
use crate::ir::{
    AsmConstraint, AsmInput, AsmSim, DType, Loc, MemSpace, Op, ValueId, asm_clobbers_memory, asm_constraints,
    asm_empty_constraint, asm_register,
};
use crate::tensor::Ptr;
use crate::value::{IntoValue, Numeric, Value};

/// The result types of an inline statement, as a tuple of element types:
/// `()`, `(T,)`, `(T, U)` and so on, up to twelve.
pub trait AsmOutputs {
    /// What `emit` returns: `()` without results, a [`Value`] for one, a
    /// tuple of them for several.
    type Values;
    #[doc(hidden)]
    fn wrap(ids: &[ValueId]) -> Self::Values;
}

/// The result types `Self` and one more, of type `X`.
pub trait AsmPush<X: Numeric>: AsmOutputs {
    type Output: AsmOutputs;
}

impl AsmOutputs for () {
    type Values = ();
    fn wrap(_: &[ValueId]) {}
}

impl<A: Numeric> AsmOutputs for (A,) {
    type Values = Value<A>;
    fn wrap(ids: &[ValueId]) -> Value<A> {
        Value::from_id(ids[0])
    }
}

macro_rules! asm_outputs {
    ($($t:ident),+) => {
        impl<$($t: Numeric),+> AsmOutputs for ($($t,)+) {
            type Values = ($(Value<$t>,)+);
            fn wrap(ids: &[ValueId]) -> Self::Values {
                let mut ids = ids.iter();
                ($(Value::<$t>::from_id(*ids.next().expect("one id per result")),)+)
            }
        }
    };
}

asm_outputs!(A, B);
asm_outputs!(A, B, C);
asm_outputs!(A, B, C, D);
asm_outputs!(A, B, C, D, E);
asm_outputs!(A, B, C, D, E, F);
asm_outputs!(A, B, C, D, E, F, G);
asm_outputs!(A, B, C, D, E, F, G, H);
asm_outputs!(A, B, C, D, E, F, G, H, I);
asm_outputs!(A, B, C, D, E, F, G, H, I, J);
asm_outputs!(A, B, C, D, E, F, G, H, I, J, K);
asm_outputs!(A, B, C, D, E, F, G, H, I, J, K, L);

impl<X: Numeric> AsmPush<X> for () {
    type Output = (X,);
}

macro_rules! asm_push {
    ($($t:ident),+) => {
        impl<$($t: Numeric,)+ X: Numeric> AsmPush<X> for ($($t,)+) {
            type Output = ($($t,)+ X);
        }
    };
}

asm_push!(A);
asm_push!(A, B);
asm_push!(A, B, C);
asm_push!(A, B, C, D);
asm_push!(A, B, C, D, E);
asm_push!(A, B, C, D, E, F);
asm_push!(A, B, C, D, E, F, G);
asm_push!(A, B, C, D, E, F, G, H);
asm_push!(A, B, C, D, E, F, G, H, I);
asm_push!(A, B, C, D, E, F, G, H, I, J);
asm_push!(A, B, C, D, E, F, G, H, I, J, K);

/// Panic with a message about the statement built at `loc`.
fn misuse(loc: Loc, what: &str, message: String) -> ! {
    panic!("cute: {}:{}:{}: {what}: {message}", loc.file(), loc.line(), loc.column())
}

/// The kind of a named operand of [`InlinePtx`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Named {
    /// `{$wN}`: a write-only output.
    Write,
    /// `{$rN}`: a read-only input.
    Read,
    /// `{$rwN}`: a read-write operand.
    ReadWrite,
}

impl Named {
    fn prefix(self) -> &'static str {
        match self {
            Named::Write => "w",
            Named::Read => "r",
            Named::ReadWrite => "rw",
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Named::Write => "write-only outputs (.write)",
            Named::Read => "read-only operands (.read, .read_ptr)",
            Named::ReadWrite => "read-write operands (.read_write)",
        }
    }
}

/// Translate a template with named operands, replacing each `{$wN}`,
/// `{$rN}` and `{$rwN}` with what `operand` returns for it and escaping any
/// other `$` as LLVM's `$$`. A positional operand (`$0`, `${0}`) or a name
/// of another form is an error.
fn translate_named(code: &str, mut operand: impl FnMut(Named, usize) -> String) -> Result<String, String> {
    let mut out = String::with_capacity(code.len());
    let mut rest = code;
    while let Some(c) = rest.chars().next() {
        if let Some(after) = rest.strip_prefix("{$") {
            let Some(end) = after.find('}') else {
                return Err(format!("`{{${}` is not closed", after.chars().take(8).collect::<String>()));
            };
            let name = &after[..end];
            let (kind, number) = if let Some(n) = name.strip_prefix("rw") {
                (Named::ReadWrite, n)
            } else if let Some(n) = name.strip_prefix('r') {
                (Named::Read, n)
            } else if let Some(n) = name.strip_prefix('w') {
                (Named::Write, n)
            } else {
                (Named::Read, "")
            };
            let index = (!number.is_empty() && number.bytes().all(|b| b.is_ascii_digit()))
                .then(|| number.parse::<usize>().ok())
                .flatten()
                .ok_or_else(|| {
                    format!(
                        "`{{${name}}}` is not an operand name: inline PTX names its operands {{$wN}} (write-only \
                         outputs), {{$rN}} (read-only operands) and {{$rwN}} (read-write operands)"
                    )
                })?;
            out.push_str(&operand(kind, index));
            rest = &after[end + 1..];
        } else if let Some(after) = rest.strip_prefix('$') {
            if after.starts_with(|c: char| c.is_ascii_digit() || c == '{') {
                return Err(format!(
                    "`${}` is a positional operand, but this template names its operands: write {{$wN}}, {{$rN}} \
                     or {{$rwN}} (InlineAsm takes positional operands)",
                    after.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '{').collect::<String>()
                ));
            }
            out.push_str("$$");
            rest = after;
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    Ok(out)
}

/// The number of PTX statements in a template: its semicolons outside
/// comments (one for a statement without its semicolon).
fn ptx_statements(code: &str) -> usize {
    let mut count = 0;
    let mut rest = code;
    while let Some(c) = rest.chars().next() {
        if rest.starts_with("//") {
            rest = rest.find('\n').map_or("", |end| &rest[end..]);
        } else if rest.starts_with("/*") {
            rest = rest.find("*/").map_or("", |end| &rest[end + 2..]);
        } else {
            count += (c == ';') as usize;
            rest = &rest[c.len_utf8()..];
        }
    }
    count.max(1)
}

/// Hand-written PTX with named operands (CuTe DSL's `cute.arch.inline_ptx`).
///
/// The template names the operands `{$w0}, {$w1}, ...` (write-only
/// outputs, in the order of the [`write`](Self::write) calls), `{$r0},
/// {$r1}, ...` (read-only operands: [`read`](Self::read) and
/// [`read_ptr`](Self::read_ptr)) and `{$rw0}, {$rw1}, ...` (read-write
/// operands, [`read_write`](Self::read_write)), each in its type's register
/// class (`r` for Int32, `l` for Int64 and global addresses, `f` for
/// Float32, `h` for 16-bit types, `b` for Boolean, ...), except that a
/// read-only integer constant is an immediate and a shared-memory address
/// is 32 bits, as in CuTe DSL (see below). Any other `$` is literal PTX (a
/// label such as `$L_loop`). Every name must have an operand and every
/// operand a name; the kernel build panics otherwise, naming the template's
/// line. [`emit`](Self::emit) adds the statement to the kernel and returns
/// the outputs, in the order they were declared: a value for one, a tuple
/// for several. The outputs include the read-write operands' final values,
/// which CuTe DSL does not return (its `nvvm.inline_ptx` has results for
/// the write-only operands alone and drops the rest), so a port of CuTe DSL
/// code that has both gets the read-write values as extra tuple elements.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("split", || {
///     let x = param_tensor::<u64>("x", Layout::col_major(32));
///     let y = param_tensor::<u32>("y", Layout::row_major((32, 2)));
///     let t = thread_idx().x;
///     // Two outputs, and an accumulator that the PTX reads and writes.
///     let (lo, hi, acc) = InlinePtx::new("mov.b64 {{$w0}, {$w1}}, {$r0}; add.u32 {$rw0}, {$rw0}, {$w0};")
///         .write::<u32>()
///         .write::<u32>()
///         .read(x.load(t))
///         .read_write(7u32)
///         .emit();
///     y.store((t, 0), lo ^ acc);
///     y.store((t, 1), hi);
/// });
/// ```
///
/// A read-only integer constant is an immediate (constraint `n`), as the
/// NVVM lowering of CuTe DSL's `inline_ptx` makes every constant, so an
/// operand that PTX takes as a number only, such as `lop3`'s table or the
/// offset in `[{$r0}+{$r1}]`, can be passed as one. LLVM prints the
/// constant sign-extended to 64 bits (a Uint32 `0xffffffff` is `-1`, the
/// same 32 bits in a 32-bit instruction). A Float32 operand is always an
/// `f` register, even when it is a constant (NVVM's `n` is invalid for it,
/// which CuTe DSL works around with a `mov.f32`), and a Boolean constant a
/// `b` register (LLVM would print `true` as `-1`). A value computed at run
/// time is always a register.
///
/// The statement has side effects by default, as CuTe DSL's has: LLVM
/// neither removes it nor moves it past memory accesses or other side
/// effects. [`pure`](Self::pure) marks one whose outputs depend on its
/// operands alone, which may then be merged, moved or removed like
/// arithmetic.
#[must_use = "the PTX is only added to the kernel by emit()"]
pub struct InlinePtx<O = ()> {
    code: String,
    loc: Loc,
    writes: Vec<DType>,
    reads: Vec<AsmInput>,
    read_writes: Vec<ValueId>,
    /// The outputs in declaration order: whether each is read-write, and its
    /// index among its kind.
    order: Vec<(bool, usize)>,
    predicate: Option<ValueId>,
    side_effects: bool,
    sim: Option<AsmSim>,
    _outputs: PhantomData<fn() -> O>,
}

impl InlinePtx<()> {
    /// A statement of `ptx_code`, which names its operands `{$wN}`, `{$rN}`
    /// and `{$rwN}`. Its source location (for errors and line information)
    /// is this call's.
    #[track_caller]
    pub fn new(ptx_code: &str) -> InlinePtx<()> {
        InlinePtx {
            code: ptx_code.to_string(),
            loc: Location::caller(),
            writes: Vec::new(),
            reads: Vec::new(),
            read_writes: Vec::new(),
            order: Vec::new(),
            predicate: None,
            side_effects: true,
            sim: None,
            _outputs: PhantomData,
        }
    }
}

impl<O: AsmOutputs> InlinePtx<O> {
    fn retype<P>(self) -> InlinePtx<P> {
        InlinePtx {
            code: self.code,
            loc: self.loc,
            writes: self.writes,
            reads: self.reads,
            read_writes: self.read_writes,
            order: self.order,
            predicate: self.predicate,
            side_effects: self.side_effects,
            sim: self.sim,
            _outputs: PhantomData,
        }
    }

    /// The next write-only output, `{$wN}`, of type `T` (CuTe DSL's
    /// `write_only_types`).
    pub fn write<T: Numeric>(mut self) -> InlinePtx<O::Output>
    where
        O: AsmPush<T>,
    {
        self.order.push((false, self.writes.len()));
        self.writes.push(T::DTYPE);
        self.retype()
    }

    /// The next read-only operand, `{$rN}`: a value (CuTe DSL's
    /// `read_only_args`), in its type's register, or as an immediate if it
    /// is an integer constant.
    #[track_caller]
    pub fn read<T: Numeric>(mut self, value: impl IntoValue<T>) -> Self {
        self.reads.push(AsmInput::Value(value.into_value().checked()));
        self
    }

    /// The next read-only operand, `{$rN}`: the address `ptr` points at
    /// (CuTe DSL passes a pointer argument as an address too). A
    /// global-memory pointer gives a 64-bit address in the global state
    /// space (`ld.global`), which is also its generic address; a
    /// shared-memory pointer gives a 32-bit address in the shared state
    /// space (constraint `r`, as CuTe DSL's NVVM lowering gives a shared
    /// pointer and CUTLASS's `cast_smem_ptr_to_uint` computes it), so the
    /// PTX must use `.shared` (`ld.shared.u32 {$w0}, [{$r0}];`,
    /// `mapa.shared::cluster.u32`), not generic addressing; and a pointer
    /// into a [`make_rmem_tensor`](crate::make_rmem_tensor) array gives its
    /// 64-bit generic address. The simulator cannot follow an address: the
    /// simulation closure sees 0 in its place, so a kernel whose results
    /// depend on memory read through it can be compiled but not simulated
    /// faithfully, and the simulator does not see memory written through it.
    #[track_caller]
    pub fn read_ptr<T: Numeric>(mut self, ptr: &Ptr<T>) -> Self {
        let (buffer, offset) = ptr.parts();
        let index = crate::tensor::index_operand(offset);
        self.reads.push(AsmInput::Address { buffer, index, elem: T::DTYPE });
        self
    }

    /// The next read-write operand, `{$rwN}`, holding `value` when the PTX
    /// starts (CuTe DSL's `read_write_args`). Its value when the PTX ends is
    /// an output of type `T`, in the order outputs are declared (a cute-rs
    /// extension: CuTe DSL drops it). It is an output register tied to an
    /// input (LLVM's `"=r"` plus `"0"`).
    #[track_caller]
    pub fn read_write<T: Numeric>(mut self, value: impl IntoValue<T>) -> InlinePtx<O::Output>
    where
        O: AsmPush<T>,
    {
        self.order.push((true, self.read_writes.len()));
        self.read_writes.push(value.into_value().checked());
        self.retype()
    }

    /// Run the PTX only where `predicate` holds (CuTe DSL's `predicate=`):
    /// it becomes PTX's guard, `@p`, of the template, which must then be one
    /// instruction. Where `predicate` is false, the read-write operands keep
    /// their values and the write-only outputs are 0 (in CuTe DSL they are
    /// undefined there: the register is never written).
    #[track_caller]
    pub fn predicate(mut self, predicate: impl IntoValue<bool>) -> Self {
        self.predicate = Some(predicate.into_value().checked());
        self
    }

    /// The outputs depend on the operands alone (no memory, not
    /// warp-collective): emit the statement without side effects, so that
    /// LLVM may merge, move or remove it like arithmetic (and the builder
    /// removes it if its outputs are unused). A cute-rs extension.
    pub fn pure(mut self) -> Self {
        self.side_effects = false;
        self
    }

    /// What the simulator runs in place of the PTX, in each thread where the
    /// predicate (if any) holds: given the raw bits of the read-only
    /// operands (`{$r0}, {$r1}, ...`; 0 for an address) and, mutably, of the
    /// read-write operands (`{$rw0}, ...`), it updates the read-write
    /// operands and returns the write-only outputs (`{$w0}, ...`). Bits are
    /// zero-extended to 64; extra high bits of the results are ignored.
    /// Without it the kernel can be compiled but not simulated.
    pub fn sim(mut self, f: impl Fn(&[u64], &mut [u64]) -> Vec<u64> + Send + Sync + 'static) -> Self {
        self.sim = Some(AsmSim::new(f));
        self
    }

    /// Add the statement to the kernel and return its outputs. Panics,
    /// naming the template's line, if a name has no operand, an operand has
    /// no name, the template uses positional operands, or a predicated
    /// template is more than one instruction or has a guard of its own.
    pub fn emit(self) -> O::Values {
        let loc = self.loc;
        let what = "inline_ptx";
        let (nw, nr, nrw) = (self.writes.len(), self.reads.len(), self.read_writes.len());
        let count = |kind: Named| match kind {
            Named::Write => nw,
            Named::Read => nr,
            Named::ReadWrite => nrw,
        };
        let mut used = [vec![false; nw], vec![false; nr], vec![false; nrw]];
        let slot = |kind: Named| match kind {
            Named::Write => 0,
            Named::Read => 1,
            Named::ReadWrite => 2,
        };
        let mut unknown = None;
        // Outputs first (write-only, then read-write), then the read-only
        // inputs, the read-write operands' tied inputs and the predicate.
        let template = translate_named(&self.code, |kind, index| {
            if index >= count(kind) {
                unknown.get_or_insert((kind, index));
            } else {
                used[slot(kind)][index] = true;
            }
            let base = match kind {
                Named::Write => 0,
                Named::ReadWrite => nw,
                Named::Read => nw + nrw,
            };
            format!("${}", base + index)
        })
        .unwrap_or_else(|e| misuse(loc, what, e));
        if let Some((kind, index)) = unknown {
            let have = match count(kind) {
                0 => format!("there are no {}", kind.describe()),
                1 => format!("there is one of the {}, {{${}0}}", kind.describe(), kind.prefix()),
                n => format!("there are {n} {}, {{${p}0}} to {{${p}{}}}", kind.describe(), n - 1, p = kind.prefix()),
            };
            misuse(loc, what, format!("`{{${}{index}}}` names no operand: {have}", kind.prefix()));
        }
        for kind in [Named::Write, Named::Read, Named::ReadWrite] {
            if let Some(index) = used[slot(kind)].iter().position(|u| !u) {
                misuse(
                    loc,
                    what,
                    format!(
                        "operand {{${}{index}}} (one of the {}) does not appear in the template {:?}",
                        kind.prefix(),
                        kind.describe(),
                        self.code
                    ),
                );
            }
        }
        if self.order.is_empty() && !self.side_effects {
            misuse(loc, what, "a pure statement without outputs does nothing, and would be removed".to_string());
        }
        let template = match self.predicate {
            Some(_) if ptx_statements(&self.code) > 1 => misuse(
                loc,
                what,
                format!(
                    "a predicate guards one instruction (PTX's @p), but the template {:?} has {}",
                    self.code,
                    ptx_statements(&self.code)
                ),
            ),
            Some(_) if self.code.trim_start().starts_with('@') => misuse(
                loc,
                what,
                format!(
                    "the template {:?} has its own guard, and a predicate adds another (PTX's @p takes one); \
                     drop one of them",
                    self.code
                ),
            ),
            Some(_) => format!("@${} {template}", nw + nrw + nr + nrw),
            None => template,
        };
        let sim = self.sim;
        let ids = builder::with(|b| {
            let types: Vec<DType> =
                self.writes.iter().copied().chain(self.read_writes.iter().map(|&v| b.dtype(v))).collect();
            let outputs: Vec<ValueId> = types.into_iter().map(|ty| b.new_value(ty)).collect();
            let mut constraints: Vec<String> =
                outputs.iter().map(|&o| format!("={}", asm_register(b.dtype(o)))).collect();
            let mut inputs = self.reads.clone();
            // As LLVM 23 lowers CuTe DSL's `nvvm.inline_ptx`: an integer
            // constant is an immediate and a shared-memory address 32 bits.
            // Float32 constants stay in registers (NVVM's `n` for them is
            // invalid), and so do Booleans (LLVM prints `true` as -1).
            for input in &self.reads {
                constraints.push(match *input {
                    AsmInput::Value(v) if b.constant_of(v).is_some_and(|c| c.dtype().is_int()) => "n".to_string(),
                    AsmInput::Value(v) => asm_register(b.dtype(v)).to_string(),
                    AsmInput::Address { buffer, .. } if b.buffer(buffer).space == MemSpace::Shared => "r".to_string(),
                    AsmInput::Address { .. } => "l".to_string(),
                });
            }
            for (j, &v) in self.read_writes.iter().enumerate() {
                inputs.push(AsmInput::Value(v));
                constraints.push((nw + j).to_string());
            }
            if let Some(p) = self.predicate {
                inputs.push(AsmInput::Value(p));
                constraints.push("b".to_string());
            }
            let op = Op::InlinePtx {
                template,
                inputs,
                outputs: outputs.clone(),
                constraints: constraints.join(","),
                predicated: self.predicate.is_some(),
                side_effects: self.side_effects,
                sim,
            };
            b.push(op, loc);
            self.order.iter().map(|&(rw, i)| if rw { outputs[nw + i] } else { outputs[i] }).collect::<Vec<_>>()
        });
        O::wrap(&ids)
    }
}

/// LLVM inline assembly with a constraint string written by hand, as CuTe
/// DSL's raw `llvm.inline_asm` builder takes it (inside a
/// `dsl_user_op`).
///
/// The template names the operands positionally, `$0, $1, ...`: the
/// outputs, then the inputs (`$$` is a literal `$`). The constraint string
/// has an entry per operand, in that order, then clobbers: `=r` for an
/// output, `r` for an input, `+r` for a read-write operand (GCC's spelling:
/// LLVM IR has no `+`, so it becomes an output plus an input tied to it),
/// an output's number for an input tied to that output, and `~{memory}` for
/// a memory clobber ([`memory_clobber`](Self::memory_clobber) adds one).
/// [`output`](Self::output) declares each output's type (a `+` operand's
/// included), in order, and [`arg`](Self::arg) gives the operands that are
/// read (`+` operands' starting values and the inputs) in the order the
/// constraint string lists them. [`emit`](Self::emit) returns the outputs:
/// a value for one, a tuple for several.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("raw_asm", || {
///     let x = param_tensor::<u32>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     // $0 is read and written, $1 is written, $2 is read.
///     let (sum, product) = InlineAsm::new("add.u32 $0, $0, $2;\n\tmul.lo.u32 $1, $2, $2;", "+r,=r,r")
///         .output::<u32>()
///         .output::<u32>()
///         .arg(x.load(t))
///         .arg(t.bitcast::<u32>())
///         .emit();
///     x.store(t, sum ^ product);
/// });
/// ```
///
/// Unlike [`InlinePtx`], it has no side effects unless asked
/// ([`side_effects`](Self::side_effects)), as in LLVM. A memory clobber,
/// from [`memory_clobber`](Self::memory_clobber) or written in the
/// constraint string, implies them (the backend emits the statement with
/// side effects, since MLIR models inline assembly's effects only through
/// them), so a statement with one is kept even if its outputs are unused.
/// Spaces around the constraint string's entries are dropped, and an empty
/// entry is an error. The validator checks the constraint string against
/// the operands: their number and order, that each code is one NVPTX takes
/// (the register classes `b`, `c`, `h`, `r`, `l`, `f` and `d`, with `&` on
/// an output, and the immediates `n` and `i` on an input), the register
/// class of each type (`r` takes Int32 or Uint32, `l` 64-bit integers and
/// addresses, `f` Float32, ...), that an immediate is an integer constant
/// and a tied input has its output's type, and that every `$n` names an
/// operand.
#[must_use = "the assembly is only added to the kernel by emit()"]
pub struct InlineAsm<O = ()> {
    template: String,
    constraints: String,
    loc: Loc,
    outputs: Vec<DType>,
    args: Vec<AsmInput>,
    side_effects: bool,
    memory: bool,
    sim: Option<AsmSim>,
    _outputs: PhantomData<fn() -> O>,
}

impl InlineAsm<()> {
    /// A statement of `template`, with LLVM constraint string `constraints`.
    /// Its source location is this call's.
    #[track_caller]
    pub fn new(template: &str, constraints: &str) -> InlineAsm<()> {
        InlineAsm {
            template: template.to_string(),
            constraints: constraints.to_string(),
            loc: Location::caller(),
            outputs: Vec::new(),
            args: Vec::new(),
            side_effects: false,
            memory: false,
            sim: None,
            _outputs: PhantomData,
        }
    }
}

impl<O: AsmOutputs> InlineAsm<O> {
    /// The next output (an `=` or `+` entry of the constraint string), of
    /// type `T`.
    pub fn output<T: Numeric>(self) -> InlineAsm<O::Output>
    where
        O: AsmPush<T>,
    {
        let mut outputs = self.outputs;
        outputs.push(T::DTYPE);
        InlineAsm {
            template: self.template,
            constraints: self.constraints,
            loc: self.loc,
            outputs,
            args: self.args,
            side_effects: self.side_effects,
            memory: self.memory,
            sim: self.sim,
            _outputs: PhantomData,
        }
    }

    /// The next operand read: a `+` operand's starting value or an input,
    /// in the constraint string's order.
    #[track_caller]
    pub fn arg<T: Numeric>(mut self, value: impl IntoValue<T>) -> Self {
        self.args.push(AsmInput::Value(value.into_value().checked()));
        self
    }

    /// The next operand read, as the address `ptr` points at, with the
    /// address spaces of [`InlinePtx::read_ptr`]: a 64-bit input (constraint
    /// `l`), or for shared memory either that or the 32-bit address (`r`),
    /// as CUTLASS passes one.
    #[track_caller]
    pub fn arg_ptr<T: Numeric>(mut self, ptr: &Ptr<T>) -> Self {
        let (buffer, offset) = ptr.parts();
        let index = crate::tensor::index_operand(offset);
        self.args.push(AsmInput::Address { buffer, index, elem: T::DTYPE });
        self
    }

    /// Give the statement side effects (LLVM's `has_side_effects`): it is
    /// never removed, and not moved past memory accesses or other side
    /// effects.
    pub fn side_effects(mut self) -> Self {
        self.side_effects = true;
        self
    }

    /// Add a memory clobber (`~{memory}`): LLVM assumes the statement reads
    /// and writes any memory, so it reloads what it had in registers after
    /// it, and the statement is kept even if its outputs are unused (it is
    /// emitted with side effects too).
    pub fn memory_clobber(mut self) -> Self {
        self.memory = true;
        self
    }

    /// What the simulator runs in place of the assembly, as for
    /// [`InlinePtx::sim`]: the read-only operands are the inputs not tied
    /// to an output (0 for an address), in order, the read-write operands
    /// are the outputs tied to an input (`+` operands, or outputs whose
    /// number an input names), in output order, and it returns the other
    /// outputs.
    pub fn sim(mut self, f: impl Fn(&[u64], &mut [u64]) -> Vec<u64> + Send + Sync + 'static) -> Self {
        self.sim = Some(AsmSim::new(f));
        self
    }

    /// Add the statement to the kernel and return its outputs. Panics,
    /// naming the template's line, if the constraint string does not match
    /// the outputs and operands given (the validator checks the rest).
    pub fn emit(self) -> O::Values {
        let (loc, what) = (self.loc, "inline_asm");
        if let Some(k) = asm_empty_constraint(&self.constraints) {
            misuse(loc, what, format!("entry {k} of the constraint string {:?} is empty", self.constraints));
        }
        let entries = asm_constraints(&self.constraints);
        let is_output = |e: &AsmConstraint| matches!(e, AsmConstraint::Output(_) | AsmConstraint::ReadWrite(_));
        let n_out = entries.iter().take_while(|e| is_output(e)).count();
        let n_in = entries[n_out..]
            .iter()
            .take_while(|e| matches!(e, AsmConstraint::Input(_) | AsmConstraint::Tied(_)))
            .count();
        if entries[n_out + n_in..].iter().any(|e| !matches!(e, AsmConstraint::Clobber(_))) {
            misuse(
                loc,
                what,
                format!(
                    "the constraint string {:?} must list the outputs, then the inputs, then the clobbers",
                    self.constraints
                ),
            );
        }
        if n_out != self.outputs.len() {
            misuse(
                loc,
                what,
                format!(
                    "the constraint string {:?} has {n_out} output(s) (= and + entries), but {} type(s) were given \
                     with .output::<T>()",
                    self.constraints,
                    self.outputs.len()
                ),
            );
        }
        let read_writes: Vec<usize> =
            (0..n_out).filter(|&k| matches!(entries[k], AsmConstraint::ReadWrite(_))).collect();
        if self.args.len() != read_writes.len() + n_in {
            misuse(
                loc,
                what,
                format!(
                    "the constraint string {:?} reads {} operand(s) ({} + entries and {n_in} input(s)), but {} were \
                     given with .arg",
                    self.constraints,
                    read_writes.len() + n_in,
                    read_writes.len(),
                    self.args.len()
                ),
            );
        }
        // A memory clobber, from .memory_clobber() or written in the
        // constraint string, keeps the statement (and gives it side effects).
        let memory = self.memory || asm_clobbers_memory(&self.constraints);
        if n_out == 0 && !self.side_effects && !memory {
            misuse(
                loc,
                what,
                "inline assembly with no outputs, no side effects and no memory clobber does nothing, and would be \
                 removed; call .side_effects()"
                    .to_string(),
            );
        }
        // The arguments of the + operands come first, in output order.
        let (starts, inputs) = self.args.split_at(read_writes.len());
        let mut constraints: Vec<String> = entries[..n_out]
            .iter()
            .map(|e| match e {
                AsmConstraint::Output(code) | AsmConstraint::ReadWrite(code) => format!("={code}"),
                _ => unreachable!("outputs come first"),
            })
            .collect();
        let mut all_inputs = inputs.to_vec();
        constraints.extend(entries[n_out..n_out + n_in].iter().map(|e| match e {
            AsmConstraint::Input(code) => code.clone(),
            AsmConstraint::Tied(o) => o.to_string(),
            _ => unreachable!("inputs follow the outputs"),
        }));
        for (&output, start) in read_writes.iter().zip(starts) {
            if !matches!(start, AsmInput::Value(_)) {
                misuse(loc, what, format!("the read-write operand ${output} starts from an address, not a value"));
            }
            all_inputs.push(*start);
            constraints.push(output.to_string());
        }
        let mut clobbers: Vec<String> = entries[n_out + n_in..]
            .iter()
            .map(|e| match e {
                AsmConstraint::Clobber(name) => format!("~{{{name}}}"),
                _ => unreachable!("clobbers come last"),
            })
            .collect();
        if self.memory && !clobbers.iter().any(|c| c == "~{memory}") {
            clobbers.push("~{memory}".to_string());
        }
        constraints.extend(clobbers);
        let (template, side_effects, sim, types) = (self.template, self.side_effects, self.sim, self.outputs);
        let ids = builder::with(|b| {
            let outputs: Vec<ValueId> = types.iter().map(|&ty| b.new_value(ty)).collect();
            let op = Op::InlinePtx {
                template,
                inputs: all_inputs,
                outputs: outputs.clone(),
                constraints: constraints.join(","),
                predicated: false,
                side_effects,
                sim,
            };
            b.push(op, loc);
            outputs
        });
        O::wrap(&ids)
    }
}

/// The operand numbers `%0, %1, ...` of a hand-written MLIR operation, in
/// order, outside string literals; any other SSA name is an error.
fn mlir_operand_refs(text: &str) -> Result<Vec<(usize, std::ops::Range<usize>)>, String> {
    let bytes = text.as_bytes();
    let mut refs = Vec::new();
    let (mut i, mut in_string) = (0, false);
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if in_string => i += 1,
            b'"' => in_string = !in_string,
            b'%' if !in_string => {
                let end = i + 1 + bytes[i + 1..].iter().take_while(|b| b.is_ascii_digit()).count();
                let ident = |b: &u8| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'$' | b'.' | b'-');
                if end == i + 1 || bytes.get(end).is_some_and(ident) {
                    let name: String = text[i..].chars().take_while(|&c| c == '%' || ident(&(c as u8))).collect();
                    return Err(format!(
                        "`{name}` is not an operand: the operands are %0, %1, ... in order (write the operation \
                         without its results, which .result::<T>() declares)"
                    ));
                }
                refs.push((text[i + 1..end].parse().map_err(|_| format!("bad operand in {text:?}"))?, i..end));
                i = end;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    Ok(refs)
}

/// `text` with each operand `%n` replaced by `names[n]`.
pub(crate) fn substitute_mlir_operands(text: &str, names: &[String]) -> String {
    let refs = mlir_operand_refs(text).expect("checked when built");
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for (n, range) in refs {
        out.push_str(&text[last..range.start]);
        out.push_str(&names[n]);
        last = range.end;
    }
    out.push_str(&text[last..]);
    out
}

/// Check a hand-written MLIR operation against `operands` operands: every
/// `%n` names one and each is used; one line, without regions or a
/// location (the backend adds it).
pub(crate) fn check_mlir_text(text: &str, operands: usize) -> Result<(), String> {
    if text.contains(['\n', '\r']) {
        return Err(format!("{text:?} is more than one line; write one operation without regions"));
    }
    if text.trim_end().ends_with('{') || text.contains("loc(") {
        return Err(format!(
            "{text:?} has a region or a location: write one operation without regions, whose location cute-rs adds"
        ));
    }
    let refs = mlir_operand_refs(text)?;
    if let Some((n, _)) = refs.iter().find(|(n, _)| *n >= operands) {
        return Err(format!("`%{n}` names no operand: there are {operands} (%0 to %{})", operands as isize - 1));
    }
    if let Some(n) = (0..operands).find(|&n| !refs.iter().any(|(r, _)| *r == n)) {
        return Err(format!("operand %{n} does not appear in {text:?}"));
    }
    Ok(())
}

/// An MLIR operation written as text: the escape hatch for operations
/// cute-rs has no function for, such as an `nvvm.*` operation (CuTe DSL
/// builds them with the raw `cutlass._mlir.dialects` builders inside a
/// `dsl_user_op`).
///
/// The text is one operation without its results, in any syntax `mlir-opt`
/// accepts (the dialect's own or the generic one), with `%0, %1, ...`
/// standing for the operands in order: values ([`operand`](Self::operand))
/// and pointers ([`operand_ptr`](Self::operand_ptr)). Each
/// [`result`](Self::result) declares a result's type; the text must give
/// the matching MLIR types (integers are signless there: `i32` for a
/// Uint32). cute-rs adds the result names and the location, and `mlir-opt`
/// checks the operation when the kernel is compiled. Operands are scalars
/// and pointers and results scalars (no vectors or structs), and the text
/// is one operation without regions.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("sm_count", || {
///     let out = param_tensor::<i32>("out", Layout::col_major(32));
///     let x = param_tensor::<u32>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     let sms: Int32 = MlirOp::new("nvvm.read.ptx.sreg.nsmid : i32").result::<i32>().emit();
///     out.store(t, sms);
///     let swapped: Uint32 = MlirOp::new("llvm.intr.bswap(%0) : (i32) -> i32")
///         .operand(x.load(t))
///         .result::<u32>()
///         .pure()
///         .sim(|r, _| vec![(r[0] as u32).swap_bytes() as u64])
///         .emit();
///     x.store(t, swapped);
///     // An NVVM operation on pointers: a 4-byte cp.async into shared memory.
///     let smem = alloc_smem::<u32>(32, 16);
///     MlirOp::new("nvvm.cp.async.shared.global %0, %1, 4, cache = ca : !llvm.ptr<3>, !llvm.ptr<1>")
///         .operand_ptr(&smem.offset(t))
///         .operand_ptr(&x.ptr().offset(t))
///         .emit();
/// });
/// ```
///
/// It has side effects unless marked [`pure`](Self::pure): the builder keeps
/// it even if its results are unused.
#[must_use = "the operation is only added to the kernel by emit()"]
pub struct MlirOp<O = ()> {
    text: String,
    loc: Loc,
    operands: Vec<AsmInput>,
    results: Vec<DType>,
    side_effects: bool,
    sim: Option<AsmSim>,
    _outputs: PhantomData<fn() -> O>,
}

impl MlirOp<()> {
    /// An operation of `text`, whose operands are `%0, %1, ...`. Its source
    /// location is this call's.
    #[track_caller]
    pub fn new(text: &str) -> MlirOp<()> {
        MlirOp {
            text: text.to_string(),
            loc: Location::caller(),
            operands: Vec::new(),
            results: Vec::new(),
            side_effects: true,
            sim: None,
            _outputs: PhantomData,
        }
    }
}

impl<O: AsmOutputs> MlirOp<O> {
    /// The next operand, `%n`: a value.
    #[track_caller]
    pub fn operand<T: Numeric>(mut self, value: impl IntoValue<T>) -> Self {
        self.operands.push(AsmInput::Value(value.into_value().checked()));
        self
    }

    /// The next operand, `%n`: `ptr` as an LLVM pointer, for the `nvvm.*`
    /// (and `llvm.*`) operations that take one. A global-memory pointer is
    /// an `!llvm.ptr<1>`, a shared-memory one an `!llvm.ptr<3>` and one into
    /// a [`make_rmem_tensor`](crate::make_rmem_tensor) array a generic
    /// `!llvm.ptr`, made from the address [`InlinePtx::read_ptr`] passes
    /// (`llvm.inttoptr`). The simulation closure sees 0 for it, as for
    /// inline PTX: the simulator neither follows it nor sees what the
    /// operation writes through it.
    #[track_caller]
    pub fn operand_ptr<T: Numeric>(mut self, ptr: &Ptr<T>) -> Self {
        let (buffer, offset) = ptr.parts();
        let index = crate::tensor::index_operand(offset);
        self.operands.push(AsmInput::Address { buffer, index, elem: T::DTYPE });
        self
    }

    /// The next result, of type `T`.
    pub fn result<T: Numeric>(self) -> MlirOp<O::Output>
    where
        O: AsmPush<T>,
    {
        let mut results = self.results;
        results.push(T::DTYPE);
        MlirOp {
            text: self.text,
            loc: self.loc,
            operands: self.operands,
            results,
            side_effects: self.side_effects,
            sim: self.sim,
            _outputs: PhantomData,
        }
    }

    /// The results depend on the operands alone: the builder removes the
    /// operation if they are unused.
    pub fn pure(mut self) -> Self {
        self.side_effects = false;
        self
    }

    /// What the simulator runs in place of the operation: the raw bits of
    /// the operands (0 for a pointer) to those of the results (the
    /// read-write slice is empty). Without it the kernel can be compiled but
    /// not simulated.
    pub fn sim(mut self, f: impl Fn(&[u64], &mut [u64]) -> Vec<u64> + Send + Sync + 'static) -> Self {
        self.sim = Some(AsmSim::new(f));
        self
    }

    /// Add the operation to the kernel and return its results. Panics,
    /// naming its line, if an operand is missing from the text or the text
    /// names one that is not there.
    pub fn emit(self) -> O::Values {
        let loc = self.loc;
        if let Err(e) = check_mlir_text(&self.text, self.operands.len()) {
            misuse(loc, "mlir_op", e);
        }
        if self.results.is_empty() && !self.side_effects {
            misuse(loc, "mlir_op", "a pure operation without results does nothing, and would be removed".to_string());
        }
        let (text, operands, side_effects, sim, types) =
            (self.text, self.operands, self.side_effects, self.sim, self.results);
        let ids = builder::with(|b| {
            let results: Vec<ValueId> = types.iter().map(|&ty| b.new_value(ty)).collect();
            b.push(Op::MlirOp { text, operands, results: results.clone(), side_effects, sim }, loc);
            results
        });
        O::wrap(&ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_operands_translate_to_positional_ones() {
        let numbered = |kind: Named, i: usize| format!("<{}{i}>", kind.prefix());
        assert_eq!(
            translate_named("mov.b64 {{$w0}, {$w1}}, {$r0}; add.u32 {$rw10}, $L_x;", numbered).unwrap(),
            "mov.b64 {<w0>, <w1>}, <r0>; add.u32 <rw10>, $$L_x;"
        );
        for bad in ["add {$x0}", "add {$r}", "add {$w0", "add {$rwx}", "mov $0, {$r0}", "mov ${0}, 1"] {
            assert!(translate_named(bad, numbered).is_err(), "{bad}");
        }
    }

    #[test]
    fn statements_are_counted_outside_comments() {
        assert_eq!(ptx_statements("ld.global.u32 {$w0}, [{$r0}];"), 1);
        assert_eq!(ptx_statements("ld.global.u32 {$w0}, [{$r0}]"), 1);
        assert_eq!(ptx_statements("mov.u32 a, b; // x; y;\n"), 1);
        assert_eq!(ptx_statements("/* ; ; */ add.u32 a, b, c; sub.u32 a, b, c;"), 2);
    }

    #[test]
    fn mlir_operands_are_numbered() {
        assert_eq!(check_mlir_text("llvm.intr.fshl(%0, %1, %2) : (i32, i32, i32) -> i32", 3), Ok(()));
        assert_eq!(check_mlir_text("nvvm.read.ptx.sreg.nsmid : i32", 0), Ok(()));
        assert!(
            check_mlir_text("llvm.intr.bswap(%1) : (i32) -> i32", 1).unwrap_err().contains("`%1` names no operand")
        );
        assert!(check_mlir_text("llvm.intr.bswap(%0) : (i32) -> i32", 2).unwrap_err().contains("operand %1 does not"));
        assert!(check_mlir_text("%r = nvvm.foo", 0).unwrap_err().contains("`%r` is not an operand"));
        assert!(check_mlir_text("scf.execute_region {", 0).is_err());
        assert_eq!(
            substitute_mlir_operands("\"x.y\"(%1, %0) {s = \"%0\"} : (i32, i32) -> i32", &["%v4".into(), "%v9".into()]),
            "\"x.y\"(%v9, %v4) {s = \"%0\"} : (i32, i32) -> i32"
        );
    }

    /// Constraint strings split into their entries, trimmed of spaces, and
    /// empty entries are found.
    #[test]
    fn constraint_strings_parse_into_entries() {
        assert_eq!(
            asm_constraints("=r, +l ,&r,r,0, n,~{memory}"),
            vec![
                AsmConstraint::Output("r".into()),
                AsmConstraint::ReadWrite("l".into()),
                AsmConstraint::Input("&r".into()),
                AsmConstraint::Input("r".into()),
                AsmConstraint::Tied(0),
                AsmConstraint::Input("n".into()),
                AsmConstraint::Clobber("memory".into()),
            ]
        );
        assert_eq!(asm_constraints(""), vec![]);
        assert_eq!(asm_constraints(" "), vec![]);
        assert!(asm_clobbers_memory("=r,r, ~{memory}") && !asm_clobbers_memory("=r,r,~{cc}"));
        assert_eq!(asm_empty_constraint("=r,r,~{memory}"), None);
        assert_eq!(asm_empty_constraint(""), None);
        for (s, k) in [("=r,,r", 1), ("=r,r,", 2), ("=,r", 0), ("r,~{}", 1), ("+", 0), ("=r, ,r", 1)] {
            assert_eq!(asm_empty_constraint(s), Some(k), "{s:?}");
        }
    }

    /// The operand numbers of a template, as LLVM reads them, and the `$`s
    /// that are not operands.
    #[test]
    fn templates_refer_to_numbered_operands() {
        use crate::ir::asm_operand_refs;
        assert_eq!(asm_operand_refs("add.u32 $0, ${1}, ${12:x}; $$L_1: $(a$|b$) ${:uid}"), Ok(vec![0, 1, 12]));
        assert_eq!(asm_operand_refs("@$3 max.u32 $0, $0, $1;"), Ok(vec![3, 0, 0, 1]));
        assert_eq!(asm_operand_refs("membar.gl;"), Ok(vec![]));
        for bad in ["mov.u32 $L, 1;", "mov.u32 ${, 1;", "mov.u32 $", "mov.u32 ${1, 2;", "bra $_x;"] {
            assert!(asm_operand_refs(bad).is_err(), "{bad:?}");
        }
    }
}
