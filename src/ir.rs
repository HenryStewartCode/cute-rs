//! The kernel intermediate representation.
//!
//! A kernel is a tree of structured regions (like MLIR's `scf` dialect) over
//! SSA scalar values. Memory is addressed through buffers: global buffers are
//! kernel parameters, shared buffers are per-block allocations and local
//! buffers are per-thread arrays. Register tensors never appear here: the
//! builder keeps them as arrays of SSA values, which is exactly what they
//! become after register allocation.

use std::fmt;
use std::panic::Location;

/// Source location of the user code that created an instruction.
pub type Loc = &'static Location<'static>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BufferId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Scalar element types.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    pub fn bits(self) -> usize {
        match self {
            DType::Bool => 1,
            DType::I8 | DType::U8 => 8,
            DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 16,
            DType::I32 | DType::U32 | DType::F32 => 32,
            DType::I64 | DType::U64 | DType::F64 => 64,
        }
    }

    /// Bytes occupied in memory (booleans are stored as one byte).
    pub fn bytes(self) -> usize {
        self.bits().div_ceil(8)
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Integer types (booleans are not integers here).
    pub fn is_int(self) -> bool {
        !self.is_float() && self != DType::Bool
    }

    pub fn is_signed(self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
    }

    /// The name used in diagnostics and IR listings.
    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "i1",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
        }
    }

    /// The MLIR type: integers are signless there.
    pub fn mlir(self) -> &'static str {
        match self {
            DType::U8 => "i8",
            DType::U16 => "i16",
            DType::U32 => "i32",
            DType::U64 => "i64",
            other => other.name(),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemSpace {
    Global,
    Shared,
    /// Per-thread memory (the stack), for arrays indexed at runtime.
    Local,
}

/// A constant. Floats are stored as raw bits so literals can be hashed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Literal {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(u16),
    BF16(u16),
    F32(u32),
    F64(u64),
}

impl Literal {
    pub fn dtype(self) -> DType {
        match self {
            Literal::Bool(_) => DType::Bool,
            Literal::I8(_) => DType::I8,
            Literal::I16(_) => DType::I16,
            Literal::I32(_) => DType::I32,
            Literal::I64(_) => DType::I64,
            Literal::U8(_) => DType::U8,
            Literal::U16(_) => DType::U16,
            Literal::U32(_) => DType::U32,
            Literal::U64(_) => DType::U64,
            Literal::F16(_) => DType::F16,
            Literal::BF16(_) => DType::BF16,
            Literal::F32(_) => DType::F32,
            Literal::F64(_) => DType::F64,
        }
    }

    /// Raw bit pattern, zero-extended to 64 bits, as the simulator stores it.
    pub fn bits(self) -> u64 {
        match self {
            Literal::Bool(b) => b as u64,
            Literal::I8(v) => v as u8 as u64,
            Literal::I16(v) => v as u16 as u64,
            Literal::I32(v) => v as u32 as u64,
            Literal::I64(v) => v as u64,
            Literal::U8(v) => v as u64,
            Literal::U16(v) => v as u64,
            Literal::U32(v) => v as u64,
            Literal::U64(v) => v,
            Literal::F16(b) | Literal::BF16(b) => b as u64,
            Literal::F32(b) => b as u64,
            Literal::F64(b) => b,
        }
    }

    /// The zero of `ty`.
    pub fn zero(ty: DType) -> Literal {
        match ty {
            DType::Bool => Literal::Bool(false),
            DType::I8 => Literal::I8(0),
            DType::I16 => Literal::I16(0),
            DType::I32 => Literal::I32(0),
            DType::I64 => Literal::I64(0),
            DType::U8 => Literal::U8(0),
            DType::U16 => Literal::U16(0),
            DType::U32 => Literal::U32(0),
            DType::U64 => Literal::U64(0),
            DType::F16 => Literal::F16(0),
            DType::BF16 => Literal::BF16(0),
            DType::F32 => Literal::F32(0),
            DType::F64 => Literal::F64(0),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    Neg,
    /// Logical not for booleans, bitwise not for integers.
    Not,
    Abs,
    Sqrt,
    Rsqrt,
    /// `2^x` (approximate on hardware: `ex2.approx`).
    Exp2,
    /// `log2(x)` (approximate on hardware: `lg2.approx`).
    Log2,
    /// Sine (approximate on hardware: `sin.approx`).
    Sin,
    /// Cosine (approximate on hardware: `cos.approx`).
    Cos,
    /// `1 / x` (approximate on hardware: `rcp.approx`).
    Rcp,
    Floor,
    Ceil,
    Trunc,
    /// Round to the nearest integer, ties to even.
    Rint,
    /// Number of set bits.
    Popc,
    /// Leading zero bits.
    Clz,
    /// Trailing zero bits.
    Ctz,
    /// Reverse the bits.
    Brev,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    /// Float division, or integer division following the type's signedness.
    Div,
    /// Unsigned integer division (index arithmetic on non-negative values).
    DivU,
    /// Integer remainder following the type's signedness.
    Rem,
    /// Unsigned integer remainder.
    RemU,
    Min,
    Max,
    And,
    Or,
    Xor,
    Shl,
    /// Logical (zero-filling) shift right.
    ShrU,
    /// Arithmetic (sign-filling) shift right.
    ShrS,
    /// The high half of the full-width product.
    MulHi,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CmpPred {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dim {
    X,
    Y,
    Z,
}

impl Dim {
    pub fn index(self) -> usize {
        match self {
            Dim::X => 0,
            Dim::Y => 1,
            Dim::Z => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Dim::X => "x",
            Dim::Y => "y",
            Dim::Z => "z",
        }
    }
}

/// Lane masks relative to the executing lane (`%lanemask_*`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LaneMask {
    Eq,
    Le,
    Lt,
    Ge,
    Gt,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SpecialReg {
    ThreadIdx(Dim),
    BlockIdx(Dim),
    BlockDim(Dim),
    GridDim(Dim),
    LaneId,
    /// Threads per warp (32): PTX's predefined constant `WARP_SZ`, read
    /// through `llvm.nvvm.read.ptx.sreg.warpsize`. (Not a `%` register.)
    WarpSize,
    WarpId,
    NWarpId,
    SmId,
    NSmId,
    GridId,
    LaneMask(LaneMask),
    /// Cycle counter (32 bits). Changes between reads.
    Clock,
    /// Cycle counter (64 bits). Changes between reads.
    Clock64,
    /// Nanosecond timer (64 bits). Changes between reads.
    GlobalTimer,
    ClusterIdx(Dim),
    ClusterDim(Dim),
    BlockInClusterIdx(Dim),
    BlockInClusterDim(Dim),
}

impl SpecialReg {
    pub fn dtype(self) -> DType {
        match self {
            SpecialReg::Clock64 | SpecialReg::GlobalTimer => DType::I64,
            SpecialReg::LaneMask(_) | SpecialReg::Clock => DType::U32,
            _ => DType::I32,
        }
    }

    /// Whether two reads in one thread always agree (timers do not).
    pub fn is_constant(self) -> bool {
        !matches!(self, SpecialReg::Clock | SpecialReg::Clock64 | SpecialReg::GlobalTimer)
    }
}

/// Which lane a `shfl.sync` reads from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ShuffleKind {
    /// The lane given by the offset operand.
    Idx,
    /// `lane - offset`.
    Up,
    /// `lane + offset`.
    Down,
    /// `lane ^ offset`.
    Bfly,
}

/// What a `vote.sync` computes across the warp.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VoteKind {
    All,
    Any,
    /// All participating lanes agree.
    Uni,
    /// A mask of the lanes whose predicate holds.
    Ballot,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AtomicKind {
    Add,
    And,
    Or,
    Xor,
    Min,
    Max,
    Exchange,
    CompareExchange,
}

/// The threads a fence orders against.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FenceScope {
    /// The block.
    Cta,
    /// The block cluster.
    Cluster,
    /// The whole GPU.
    Gpu,
    /// The GPU and the host.
    Sys,
}

/// The threads an mbarrier operation synchronises with (CuTe DSL's
/// `nvvm.MemScopeKind` as the mbarrier operations take it).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MemScopeKind {
    /// The CTA (the default).
    #[default]
    Cta,
    /// The CTA's cluster (SM90 and newer).
    Cluster,
}

/// How an mbarrier phase is waited on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MbarrierWaitKind {
    /// `mbarrier.try_wait`: may suspend the thread for a while (SM90).
    Try,
    /// `mbarrier.test_wait`: returns at once (SM80).
    Test,
}

/// The named-barrier instructions (`bar`, `barrier`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BarrierKind {
    /// `bar.sync id{, count}`: arrive, and wait until the barrier completes.
    Sync,
    /// `bar.arrive id, count`: arrive without waiting.
    Arrive,
    /// `barrier.arrive id, count`: an arrive without `.aligned`, which the
    /// threads of the CTA need not all execute.
    ArriveUnaligned,
}

/// Warp-level matrix multiply-accumulate instructions (`mma.sync`). The name
/// gives the shape, then the accumulator type and the operand type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MmaKind {
    /// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
    M16N8K16F32F16,
    /// `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`
    M16N8K8F32F16,
    /// `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
    M16N8K16F16F16,
    /// `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`
    M16N8K8F16F16,
    /// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
    M16N8K16F32BF16,
    /// `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32`
    M16N8K8F32BF16,
    /// `mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32`: f32 operands
    /// truncated to 19 bits.
    M16N8K4Tf32,
    /// `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`: f32 operands
    /// truncated to 19 bits.
    M16N8K8Tf32,
    /// `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64`
    M8N8K4F64,
}

impl MmaKind {
    pub fn shape_mnk(self) -> (usize, usize, usize) {
        match self {
            MmaKind::M16N8K16F32F16 | MmaKind::M16N8K16F16F16 | MmaKind::M16N8K16F32BF16 => (16, 8, 16),
            MmaKind::M16N8K8F32F16 | MmaKind::M16N8K8F16F16 | MmaKind::M16N8K8F32BF16 | MmaKind::M16N8K8Tf32 => {
                (16, 8, 8)
            }
            MmaKind::M16N8K4Tf32 => (16, 8, 4),
            MmaKind::M8N8K4F64 => (8, 8, 4),
        }
    }

    /// Values held per thread for (A, B, C/D).
    pub fn fragment_sizes(self) -> (usize, usize, usize) {
        match self {
            MmaKind::M16N8K16F32F16 | MmaKind::M16N8K16F16F16 | MmaKind::M16N8K16F32BF16 => (8, 4, 4),
            MmaKind::M16N8K8F32F16 | MmaKind::M16N8K8F16F16 | MmaKind::M16N8K8F32BF16 | MmaKind::M16N8K8Tf32 => {
                (4, 2, 4)
            }
            MmaKind::M16N8K4Tf32 => (2, 1, 4),
            MmaKind::M8N8K4F64 => (1, 1, 2),
        }
    }

    /// Element types of (A, B, C/D). TF32 operands are passed as `f32`.
    pub fn dtypes(self) -> (DType, DType, DType) {
        match self {
            MmaKind::M16N8K16F32F16 | MmaKind::M16N8K8F32F16 => (DType::F16, DType::F16, DType::F32),
            MmaKind::M16N8K16F16F16 | MmaKind::M16N8K8F16F16 => (DType::F16, DType::F16, DType::F16),
            MmaKind::M16N8K16F32BF16 | MmaKind::M16N8K8F32BF16 => (DType::BF16, DType::BF16, DType::F32),
            MmaKind::M16N8K4Tf32 | MmaKind::M16N8K8Tf32 => (DType::F32, DType::F32, DType::F32),
            MmaKind::M8N8K4F64 => (DType::F64, DType::F64, DType::F64),
        }
    }

    /// Whether the operands are TF32 (f32 registers whose low 13 mantissa
    /// bits the tensor core ignores).
    pub fn is_tf32(self) -> bool {
        matches!(self, MmaKind::M16N8K4Tf32 | MmaKind::M16N8K8Tf32)
    }

    /// The PTX instruction, e.g. `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
    pub fn ptx(self) -> String {
        let (m, n, k) = self.shape_mnk();
        let (a, _, c) = self.dtypes();
        let ab = if self.is_tf32() { "tf32" } else { a.name() };
        format!("mma.sync.aligned.m{m}n{n}k{k}.row.col.{c}.{ab}.{ab}.{c}", c = c.name())
    }
}

/// IEEE 754 rounding directions (CuTe DSL's `cute.RoundingMode`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RoundingMode {
    /// To nearest, ties to even (`.rn`).
    NearestEven,
    /// Toward zero (`.rz`).
    Zero,
    /// Toward +infinity (`.rp`).
    Up,
    /// Toward -infinity (`.rm`).
    Down,
}

impl RoundingMode {
    /// The PTX rounding modifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            RoundingMode::NearestEven => "rn",
            RoundingMode::Zero => "rz",
            RoundingMode::Up => "rp",
            RoundingMode::Down => "rm",
        }
    }
}

/// The PTX floating-point instructions that [`Op::FloatInst`] issues with
/// explicit modifiers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatFn {
    /// `add`
    Add,
    /// `mul`
    Mul,
    /// `fma`
    Fma,
    /// `div`
    Div,
    /// `rcp`: `1 / x`.
    Rcp,
    /// `sqrt`
    Sqrt,
    /// `rsqrt`: `1 / sqrt(x)`.
    Rsqrt,
    /// `ex2`: `2^x`.
    Ex2,
    /// `lg2`: `log2(x)`.
    Lg2,
    /// `sin`
    Sin,
    /// `cos`
    Cos,
    /// `tanh`
    Tanh,
    /// `abs`
    Abs,
    /// `min`
    Min,
    /// `max`
    Max,
}

impl FloatFn {
    /// The PTX instruction name.
    pub fn ptx(self) -> &'static str {
        match self {
            FloatFn::Add => "add",
            FloatFn::Mul => "mul",
            FloatFn::Fma => "fma",
            FloatFn::Div => "div",
            FloatFn::Rcp => "rcp",
            FloatFn::Sqrt => "sqrt",
            FloatFn::Rsqrt => "rsqrt",
            FloatFn::Ex2 => "ex2",
            FloatFn::Lg2 => "lg2",
            FloatFn::Sin => "sin",
            FloatFn::Cos => "cos",
            FloatFn::Tanh => "tanh",
            FloatFn::Abs => "abs",
            FloatFn::Min => "min",
            FloatFn::Max => "max",
        }
    }

    /// Number of operands.
    pub fn arity(self) -> usize {
        match self {
            FloatFn::Fma => 3,
            FloatFn::Add | FloatFn::Mul | FloatFn::Div | FloatFn::Min | FloatFn::Max => 2,
            _ => 1,
        }
    }
}

/// How an [`Op::FloatInst`] computes its result.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatMode {
    /// IEEE 754, rounded in the given direction (`.rn`, `.rz`, `.rp`, `.rm`).
    Round(RoundingMode),
    /// The hardware approximation (`.approx`).
    Approx,
    /// `div.full`: a full-range approximate division.
    Full,
    /// Exact results that need no rounding (`abs`, `min`, `max`).
    Exact,
}

impl FloatMode {
    /// The PTX modifier, with its dot (empty for [`FloatMode::Exact`]).
    pub fn ptx(self) -> String {
        match self {
            FloatMode::Round(r) => format!(".{}", r.ptx()),
            FloatMode::Approx => ".approx".into(),
            FloatMode::Full => ".full".into(),
            FloatMode::Exact => String::new(),
        }
    }
}

/// The PTX instruction an [`Op::FloatInst`] issues, e.g. `ex2.approx.ftz.f32`.
pub fn float_inst_name(func: FloatFn, mode: FloatMode, ftz: bool, nan: bool, ty: DType) -> String {
    format!(
        "{}{}{}{}.{}",
        func.ptx(),
        mode.ptx(),
        if ftz { ".ftz" } else { "" },
        if nan { ".NaN" } else { "" },
        ty.name()
    )
}

/// Simulates an inline PTX fragment: raw input bits to raw output bits.
pub type InlineSim = fn(&[u64]) -> Vec<u64>;

/// Hints for a runtime loop: the keyword arguments of CuTe DSL's
/// `cutlass.range(start, stop, step, unroll=, unroll_full=, vectorize=,
/// at_least_once=)`. The default attaches nothing, as in CuTe DSL.
///
/// The unroll and vectorize hints become LLVM loop metadata
/// (`#llvm.loop_annotation`), which NVPTX honours; they never change what the
/// loop computes, and the simulator ignores them. Fields not given keep
/// their defaults: `LoopOptions { vectorize: true, ..LoopOptions::default()
/// }`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LoopOptions {
    /// `unroll=K`: unroll the loop `K` times (`count = K`). 1 keeps it rolled
    /// (`disable = true`, PTX `.pragma "nounroll"`); 0 leaves the factor to
    /// LLVM. A runtime trip count gets a remainder loop.
    pub unroll: Option<u32>,
    /// `unroll_full=True`: unroll the loop completely (`full = true`). LLVM
    /// only does so when it can compute the trip count. Cannot be combined
    /// with `unroll` (CuTe DSL lets `unroll_full` win; this is an error).
    pub unroll_full: bool,
    /// `vectorize=True`: ask LLVM to vectorise the loop (`vectorize =
    /// <disable = false>`). Needs sm_100 or newer, as in CuTe DSL.
    pub vectorize: bool,
    /// `at_least_once=True`: the program promises that the loop runs at
    /// least once in every thread, so the compiler can drop the check
    /// whether to skip it (the MLIR assumes `lower < upper` with
    /// `llvm.intr.assume`). The simulator reports a thread for which it
    /// runs zero times, and the validator constant bounds that do.
    pub at_least_once: bool,
}

impl LoopOptions {
    /// `unroll=count`.
    pub fn unroll(count: u32) -> LoopOptions {
        LoopOptions { unroll: Some(count), ..LoopOptions::default() }
    }

    /// `unroll_full=True`.
    pub fn unroll_full() -> LoopOptions {
        LoopOptions { unroll_full: true, ..LoopOptions::default() }
    }

    /// Whether any hint becomes loop metadata.
    pub fn has_annotation(&self) -> bool {
        self.unroll.is_some() || self.unroll_full || self.vectorize
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    Const {
        result: ValueId,
        value: Literal,
    },
    Unary {
        result: ValueId,
        op: UnaryOp,
        x: ValueId,
    },
    Binary {
        result: ValueId,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    /// `a * b + c` with a single rounding.
    Fma {
        result: ValueId,
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },
    /// A floating-point instruction with explicit PTX modifiers: a rounding
    /// direction, the hardware approximation, flushing subnormal inputs and
    /// results to zero (`ftz`) or NaN propagation (`nan`, for `min` and
    /// `max`). These are `cute.math`'s `approx=`, `ftz=`, `rounding=`,
    /// `full=` and `propagate_nan=` options and its `fastmath=True`
    /// approximations.
    FloatInst {
        result: ValueId,
        func: FloatFn,
        args: Vec<ValueId>,
        mode: FloatMode,
        ftz: bool,
        nan: bool,
    },
    Cmp {
        result: ValueId,
        pred: CmpPred,
        lhs: ValueId,
        rhs: ValueId,
    },
    Select {
        result: ValueId,
        cond: ValueId,
        on_true: ValueId,
        on_false: ValueId,
    },
    /// Numeric conversion to the result's type.
    Cast {
        result: ValueId,
        x: ValueId,
    },
    /// Reinterpret the bits as the result's type (same width).
    Bitcast {
        result: ValueId,
        x: ValueId,
    },
    Special {
        result: ValueId,
        reg: SpecialReg,
    },
    /// Load `results.len()` contiguous elements of the results' type. `index`
    /// counts elements of that type from the start of the buffer.
    Load {
        results: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
    },
    /// Store `values.len()` contiguous elements of the values' type. `index`
    /// counts elements of that type from the start of the buffer.
    Store {
        values: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
    },
    /// `cp.async` of `elems` contiguous `elem`s from global to shared memory.
    /// Indices count `elem`s. When `pred` is false the destination is
    /// zero-filled instead.
    CpAsync {
        src: BufferId,
        src_index: ValueId,
        dst: BufferId,
        dst_index: ValueId,
        elem: DType,
        elems: u32,
        bypass_l1: bool,
        pred: Option<ValueId>,
    },
    CpAsyncCommitGroup,
    /// Wait until at most `pending` committed groups are still in flight.
    CpAsyncWaitGroup {
        pending: u32,
    },
    /// `ldmatrix.sync.aligned.m8n8.x{num}[.trans].shared.b16`. Each lane
    /// passes the address of one 8-element row; `results` holds two 16-bit
    /// values per loaded matrix, register-major.
    LdMatrix {
        results: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
        num: u32,
        trans: bool,
    },
    Mma {
        kind: MmaKind,
        d: Vec<ValueId>,
        a: Vec<ValueId>,
        b: Vec<ValueId>,
        c: Vec<ValueId>,
    },
    /// `bar.sync 0` (`__syncthreads`).
    Barrier,
    /// A named barrier: barrier `id` (0 to 15) completes when `threads`
    /// threads (a multiple of 32; all of the block when `None`) have arrived.
    NamedBarrier {
        kind: BarrierKind,
        id: ValueId,
        threads: Option<ValueId>,
    },
    /// `bar.warp.sync` (`__syncwarp`) over the lanes in `mask`.
    WarpBarrier {
        mask: ValueId,
    },
    /// `shfl.sync`: read `value` from another lane of the warp.
    Shuffle {
        result: ValueId,
        kind: ShuffleKind,
        value: ValueId,
        offset: ValueId,
        mask: ValueId,
    },
    /// `vote.sync` over the lanes in `mask`.
    Vote {
        result: ValueId,
        kind: VoteKind,
        pred: ValueId,
        mask: ValueId,
    },
    /// `match.any.sync`: the lanes holding the same value as this one.
    MatchAny {
        result: ValueId,
        value: ValueId,
        mask: ValueId,
    },
    /// `elect.sync`: true in exactly one of the lanes in `mask`.
    ElectOne {
        result: ValueId,
        mask: ValueId,
    },
    /// `activemask.b32`.
    ActiveMask {
        result: ValueId,
    },
    /// `value`, which the program promises is the same in every lane of the
    /// warp (CuTe DSL's `make_warp_uniform`): a compiler hint, not an
    /// instruction.
    WarpUniform {
        result: ValueId,
        value: ValueId,
    },
    /// Atomic read-modify-write; `result` is the old value. `index` counts
    /// elements of the operand's type.
    Atomic {
        result: ValueId,
        kind: AtomicKind,
        buffer: BufferId,
        index: ValueId,
        value: ValueId,
        compare: Option<ValueId>,
    },
    /// `mbarrier.init`: an mbarrier waiting for `count` arrivals per phase.
    MbarrierInit {
        buffer: BufferId,
        index: ValueId,
        count: ValueId,
    },
    /// `mbarrier.arrive`: `count` arrivals (1 when `None`), after raising the
    /// bytes of bulk copies the phase waits for by `expect_tx` when given
    /// (`.expect_tx`). With `peer`, on the mbarrier at the same offset in
    /// that CTA of the cluster (`mapa` + `.shared::cluster`). `relaxed` drops
    /// the release ordering.
    MbarrierArrive {
        buffer: BufferId,
        index: ValueId,
        count: Option<ValueId>,
        expect_tx: Option<ValueId>,
        peer: Option<ValueId>,
        relaxed: bool,
        scope: MemScopeKind,
    },
    /// `mbarrier.expect_tx`: make the current phase wait for `bytes` more
    /// bytes of bulk copies, without arriving.
    MbarrierExpectTx {
        buffer: BufferId,
        index: ValueId,
        bytes: ValueId,
        peer: Option<ValueId>,
        scope: MemScopeKind,
    },
    /// `mbarrier.try_wait.parity` / `mbarrier.test_wait.parity`: whether the
    /// phase with this parity is over.
    MbarrierWait {
        result: ValueId,
        buffer: BufferId,
        index: ValueId,
        parity: ValueId,
        kind: MbarrierWaitKind,
    },
    /// `fence.mbarrier_init.release.cluster`: make earlier `mbarrier.init`s
    /// visible to the cluster.
    FenceMbarrierInit,
    /// `cp.async.bulk` of `bytes` bytes (a multiple of 16) from global to
    /// shared memory, reporting completion to an mbarrier. Indices count
    /// `elem`s.
    CpAsyncBulk {
        src: BufferId,
        src_index: ValueId,
        dst: BufferId,
        dst_index: ValueId,
        elem: DType,
        bytes: ValueId,
        mbarrier: BufferId,
        mbarrier_index: ValueId,
    },
    /// `fence.acq_rel.{scope}`.
    Fence {
        scope: FenceScope,
    },
    /// `vprintf`.
    Printf {
        format: String,
        args: Vec<ValueId>,
    },
    /// A PTX fragment written by hand. `$0..` name the outputs, followed by
    /// the inputs. The simulator can only run it if `sim` is given.
    InlinePtx {
        template: String,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        side_effects: bool,
        sim: Option<InlineSim>,
    },
    If {
        cond: ValueId,
        then_region: Region,
        else_region: Region,
        results: Vec<ValueId>,
    },
    /// `for iv in (lower..upper).step_by(step)` with loop-carried values:
    /// `args` are the body's views of the carried values, initialised from
    /// `init` and updated from the body's yields; `results` are the final
    /// values. `opts` holds the unroll and vectorize hints.
    For {
        iv: ValueId,
        lower: ValueId,
        upper: ValueId,
        step: ValueId,
        init: Vec<ValueId>,
        args: Vec<ValueId>,
        body: Region,
        results: Vec<ValueId>,
        opts: LoopOptions,
    },
    /// `while cond(args) { args = body(args) }`. `cond` sees `cond_args` and
    /// yields one boolean; `body` sees `body_args` and yields the next state.
    /// `results` are the state when the condition first fails.
    While {
        init: Vec<ValueId>,
        cond_args: Vec<ValueId>,
        cond: Region,
        body_args: Vec<ValueId>,
        body: Region,
        results: Vec<ValueId>,
    },
}

impl Op {
    /// Values defined by this operation (block arguments of nested regions
    /// excluded).
    pub fn results(&self) -> Vec<ValueId> {
        match self {
            Op::Const { result, .. }
            | Op::Unary { result, .. }
            | Op::Binary { result, .. }
            | Op::Fma { result, .. }
            | Op::FloatInst { result, .. }
            | Op::Cmp { result, .. }
            | Op::Select { result, .. }
            | Op::Cast { result, .. }
            | Op::Bitcast { result, .. }
            | Op::Special { result, .. }
            | Op::Shuffle { result, .. }
            | Op::Vote { result, .. }
            | Op::MatchAny { result, .. }
            | Op::ElectOne { result, .. }
            | Op::ActiveMask { result }
            | Op::WarpUniform { result, .. }
            | Op::MbarrierWait { result, .. }
            | Op::Atomic { result, .. } => vec![*result],
            Op::Load { results, .. } | Op::LdMatrix { results, .. } => results.clone(),
            Op::Mma { d, .. } => d.clone(),
            Op::InlinePtx { outputs, .. } => outputs.clone(),
            Op::If { results, .. } | Op::For { results, .. } | Op::While { results, .. } => results.clone(),
            Op::Store { .. }
            | Op::CpAsync { .. }
            | Op::CpAsyncCommitGroup
            | Op::CpAsyncWaitGroup { .. }
            | Op::Barrier
            | Op::NamedBarrier { .. }
            | Op::WarpBarrier { .. }
            | Op::MbarrierInit { .. }
            | Op::MbarrierArrive { .. }
            | Op::MbarrierExpectTx { .. }
            | Op::FenceMbarrierInit
            | Op::CpAsyncBulk { .. }
            | Op::Fence { .. }
            | Op::Printf { .. } => Vec::new(),
        }
    }

    /// Values read by this operation itself (not by nested regions).
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Op::Const { .. }
            | Op::Special { .. }
            | Op::ActiveMask { .. }
            | Op::Fence { .. }
            | Op::FenceMbarrierInit => {
                vec![]
            }
            Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } | Op::WarpUniform { value: x, .. } => {
                vec![*x]
            }
            Op::Binary { lhs, rhs, .. } | Op::Cmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::Fma { a, b, c, .. } => vec![*a, *b, *c],
            Op::FloatInst { args, .. } => args.clone(),
            Op::Select { cond, on_true, on_false, .. } => vec![*cond, *on_true, *on_false],
            Op::Load { index, .. } | Op::LdMatrix { index, .. } => vec![*index],
            Op::Store { values, index, .. } => {
                let mut v = values.clone();
                v.push(*index);
                v
            }
            Op::CpAsync { src_index, dst_index, pred, .. } => {
                let mut v = vec![*src_index, *dst_index];
                v.extend(pred);
                v
            }
            Op::CpAsyncCommitGroup | Op::CpAsyncWaitGroup { .. } | Op::Barrier => vec![],
            Op::NamedBarrier { id, threads, .. } => {
                let mut v = vec![*id];
                v.extend(threads);
                v
            }
            Op::WarpBarrier { mask } | Op::ElectOne { mask, .. } => vec![*mask],
            Op::Shuffle { value, offset, mask, .. } => vec![*value, *offset, *mask],
            Op::Vote { pred, mask, .. } => vec![*pred, *mask],
            Op::MatchAny { value, mask, .. } => vec![*value, *mask],
            Op::Atomic { index, value, compare, .. } => {
                let mut v = vec![*index, *value];
                v.extend(compare);
                v
            }
            Op::MbarrierInit { index, count, .. } => vec![*index, *count],
            Op::MbarrierArrive { index, count, expect_tx, peer, .. } => {
                let mut v = vec![*index];
                v.extend(count);
                v.extend(expect_tx);
                v.extend(peer);
                v
            }
            Op::MbarrierExpectTx { index, bytes, peer, .. } => {
                let mut v = vec![*index, *bytes];
                v.extend(peer);
                v
            }
            Op::MbarrierWait { index, parity, .. } => vec![*index, *parity],
            Op::CpAsyncBulk { src_index, dst_index, bytes, mbarrier_index, .. } => {
                vec![*src_index, *dst_index, *bytes, *mbarrier_index]
            }
            Op::Printf { args, .. } => args.clone(),
            Op::InlinePtx { inputs, .. } => inputs.clone(),
            Op::Mma { a, b, c, .. } => a.iter().chain(b).chain(c).copied().collect(),
            Op::If { cond, .. } => vec![*cond],
            Op::For { lower, upper, step, init, .. } => {
                let mut v = vec![*lower, *upper, *step];
                v.extend(init);
                v
            }
            Op::While { init, .. } => init.clone(),
        }
    }

    /// Nested regions.
    pub fn regions(&self) -> Vec<&Region> {
        match self {
            Op::If { then_region, else_region, .. } => vec![then_region, else_region],
            Op::For { body, .. } => vec![body],
            Op::While { cond, body, .. } => vec![cond, body],
            _ => vec![],
        }
    }

    /// Nested regions, mutably.
    pub fn regions_mut(&mut self) -> Vec<&mut Region> {
        match self {
            Op::If { then_region, else_region, .. } => vec![then_region, else_region],
            Op::For { body, .. } => vec![body],
            Op::While { cond, body, .. } => vec![cond, body],
            _ => vec![],
        }
    }

    /// Pure operations can be removed when their results are unused.
    pub fn is_pure(&self) -> bool {
        matches!(
            self,
            Op::Const { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Fma { .. }
                | Op::FloatInst { .. }
                | Op::Cmp { .. }
                | Op::Select { .. }
                | Op::Cast { .. }
                | Op::Bitcast { .. }
                | Op::Special { .. }
                | Op::WarpUniform { .. }
                | Op::InlinePtx { side_effects: false, .. }
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            Op::Const { .. } => "const",
            Op::Unary { .. } => "unary",
            Op::Binary { .. } => "binary",
            Op::Fma { .. } => "fma",
            Op::FloatInst { .. } => "float instruction",
            Op::Cmp { .. } => "cmp",
            Op::Select { .. } => "select",
            Op::Cast { .. } => "cast",
            Op::Bitcast { .. } => "bitcast",
            Op::Special { .. } => "special register",
            Op::Load { .. } => "load",
            Op::Store { .. } => "store",
            Op::CpAsync { .. } => "cp.async",
            Op::CpAsyncCommitGroup => "cp.async.commit_group",
            Op::CpAsyncWaitGroup { .. } => "cp.async.wait_group",
            Op::LdMatrix { .. } => "ldmatrix",
            Op::Mma { .. } => "mma.sync",
            Op::Barrier => "barrier",
            Op::NamedBarrier { kind: BarrierKind::Sync, .. } => "bar.sync",
            Op::NamedBarrier { kind: BarrierKind::Arrive, .. } => "bar.arrive",
            Op::NamedBarrier { kind: BarrierKind::ArriveUnaligned, .. } => "barrier.arrive",
            Op::WarpBarrier { .. } => "bar.warp.sync",
            Op::Shuffle { .. } => "shfl.sync",
            Op::Vote { .. } => "vote.sync",
            Op::MatchAny { .. } => "match.any.sync",
            Op::ElectOne { .. } => "elect.sync",
            Op::ActiveMask { .. } => "activemask",
            Op::WarpUniform { .. } => "make_warp_uniform",
            Op::Atomic { .. } => "atomic",
            Op::MbarrierInit { .. } => "mbarrier.init",
            Op::MbarrierArrive { expect_tx: Some(_), .. } => "mbarrier.arrive.expect_tx",
            Op::MbarrierArrive { .. } => "mbarrier.arrive",
            Op::MbarrierExpectTx { .. } => "mbarrier.expect_tx",
            Op::MbarrierWait { kind: MbarrierWaitKind::Try, .. } => "mbarrier.try_wait",
            Op::MbarrierWait { kind: MbarrierWaitKind::Test, .. } => "mbarrier.test_wait",
            Op::FenceMbarrierInit => "fence.mbarrier_init",
            Op::CpAsyncBulk { .. } => "cp.async.bulk",
            Op::Fence { .. } => "fence",
            Op::Printf { .. } => "printf",
            Op::InlinePtx { .. } => "inline ptx",
            Op::If { .. } => "if",
            Op::For { .. } => "for",
            Op::While { .. } => "while",
        }
    }
}

/// An instruction: an operation plus the user source location that built it.
#[derive(Clone, Debug)]
pub struct Inst {
    pub op: Op,
    pub loc: Loc,
}

/// A straight-line list of instructions, terminated by the values it yields
/// to its parent (`If` results or the next `For` iteration).
#[derive(Clone, Debug, Default)]
pub struct Region {
    pub insts: Vec<Inst>,
    pub yields: Vec<ValueId>,
}

#[derive(Clone, Debug)]
pub struct BufferDecl {
    pub name: String,
    pub elem: DType,
    /// Number of elements, or `None` for a global buffer sized at launch.
    pub len: Option<usize>,
    pub space: MemSpace,
    /// Byte alignment of the allocation (shared memory) or the alignment the
    /// kernel assumes for the pointer it is given (global memory).
    pub align: usize,
}

#[derive(Clone, Debug)]
pub enum Param {
    Buffer(BufferId),
    /// `multiple_of` is what the kernel assumes the launch value divides by.
    Scalar {
        name: String,
        value: ValueId,
        multiple_of: usize,
    },
}

#[derive(Clone, Debug)]
pub struct KernelIr {
    pub name: String,
    pub args: Vec<Param>,
    pub buffers: Vec<BufferDecl>,
    /// Type of every SSA value, indexed by `ValueId`.
    pub value_types: Vec<DType>,
    pub body: Region,
    /// Block size the kernel was written for, if declared.
    pub block_dim: Option<[u32; 3]>,
}

impl KernelIr {
    pub fn dtype(&self, v: ValueId) -> DType {
        self.value_types[v.0 as usize]
    }

    pub fn buffer(&self, b: BufferId) -> &BufferDecl {
        &self.buffers[b.0 as usize]
    }

    /// Bytes of shared memory used by one block, and each buffer's start.
    pub fn shared_memory_layout(&self) -> (usize, Vec<Option<usize>>) {
        let mut total = 0usize;
        let starts = self
            .buffers
            .iter()
            .map(|b| {
                (b.space == MemSpace::Shared).then(|| {
                    let start = total.next_multiple_of(b.align);
                    total = start + b.len.expect("shared buffers have a size") * b.elem.bytes();
                    start
                })
            })
            .collect();
        (total, starts)
    }

    /// Bytes of shared memory used by one block.
    pub fn shared_memory_bytes(&self) -> usize {
        self.shared_memory_layout().0
    }

    /// Count instructions (recursively) matching `pred`.
    pub fn count_ops(&self, pred: impl Fn(&Op) -> bool + Copy) -> usize {
        fn walk(r: &Region, pred: impl Fn(&Op) -> bool + Copy) -> usize {
            r.insts
                .iter()
                .map(|i| {
                    let nested: usize = i.op.regions().into_iter().map(|r| walk(r, pred)).sum();
                    nested + pred(&i.op) as usize
                })
                .sum()
        }
        walk(&self.body, pred)
    }
}
