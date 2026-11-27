//! Scalar operators, special registers, and the warp-collective and atomic kinds.

use super::*;

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
    /// The low 32 bits of the nanosecond timer. Changes between reads.
    GlobalTimerLo,
    ClusterIdx(Dim),
    ClusterDim(Dim),
    BlockInClusterIdx(Dim),
    BlockInClusterDim(Dim),
    /// The bytes of dynamic shared memory the launch requested
    /// ([`LaunchConfig::dynamic_smem_bytes`](crate::LaunchConfig::dynamic_smem_bytes)),
    /// which the dynamic region ([`get_dyn_smem`](crate::get_dyn_smem)) gets:
    /// CuTe DSL's `dynamic_smem_size` and `get_dyn_smem_size`. It is
    /// `%dynamic_smem_size` less the shared memory sized at build time that
    /// the dynamic window holds in front of the region
    /// ([`SharedMemoryLayout::dynamic_start`]).
    DynamicSmemSize,
    /// `%total_smem_size`: the block's static and dynamic shared memory in
    /// bytes, rounded up to the 128-byte allocation unit (as an SM89 GPU
    /// reports it).
    TotalSmemSize,
    /// `%aggr_smem_size`: the block's shared memory plus the part reserved
    /// for the system (sm_90 and newer).
    AggrSmemSize,
    /// `%cluster_ctarank`: the block's linear rank in its cluster, x
    /// fastest (CuTe DSL's `block_idx_in_cluster`).
    ClusterCtaRank,
    /// `%cluster_nctarank`: the number of blocks in the cluster (CuTe DSL's
    /// `cluster_size`).
    ClusterNCtaRank,
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
        !matches!(self, SpecialReg::Clock | SpecialReg::Clock64 | SpecialReg::GlobalTimer | SpecialReg::GlobalTimerLo)
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

/// What a `match.sync` computes (CuTe DSL's `kind="any"` and `kind="all"`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MatchKind {
    /// `match.any.sync`: the member lanes holding the same value as this one.
    #[default]
    Any,
    /// `match.all.sync`: the member lanes if they all hold the same value,
    /// otherwise 0.
    All,
}

/// The reduction a `redux.sync` computes across the member lanes of a warp
/// (CuTe DSL's `warp_redux_sync` kinds).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ReduxKind {
    /// The sum, wrapping at 32 bits (Int32, Uint32).
    Add,
    /// Bitwise and (Int32, Uint32).
    And,
    /// Bitwise or (Int32, Uint32).
    Or,
    /// Bitwise exclusive or (Int32, Uint32).
    Xor,
    /// The largest value: signed on Int32; `warp_redux_sync` takes it as
    /// [`ReduxKind::Umax`] on Uint32 and [`ReduxKind::Fmax`] on Float32, as
    /// CuTe DSL does.
    Max,
    /// The smallest value: signed on Int32; [`ReduxKind::Umin`] on Uint32
    /// and [`ReduxKind::Fmin`] on Float32.
    Min,
    /// The largest value as unsigned 32-bit integers.
    Umax,
    /// The smallest value as unsigned 32-bit integers.
    Umin,
    /// The largest Float32 (`+0 > -0`), with optional `.abs` and `.NaN`.
    Fmax,
    /// The smallest Float32 (`-0 < +0`), with optional `.abs` and `.NaN`.
    Fmin,
}

impl ReduxKind {
    /// CuTe DSL's name for the kind (and the NVVM dialect's).
    pub fn name(self) -> &'static str {
        match self {
            ReduxKind::Add => "add",
            ReduxKind::And => "and",
            ReduxKind::Or => "or",
            ReduxKind::Xor => "xor",
            ReduxKind::Max => "max",
            ReduxKind::Min => "min",
            ReduxKind::Umax => "umax",
            ReduxKind::Umin => "umin",
            ReduxKind::Fmax => "fmax",
            ReduxKind::Fmin => "fmin",
        }
    }

    /// Whether the kind reduces Float32 values.
    pub fn is_float(self) -> bool {
        matches!(self, ReduxKind::Fmax | ReduxKind::Fmin)
    }
}

/// The PTX instruction an [`Op::Redux`] issues on a value of type `ty`, e.g.
/// `redux.sync.add.s32`, `redux.sync.min.u32`, `redux.sync.xor.b32` or
/// `redux.sync.max.abs.NaN.f32`.
pub fn redux_inst_name(kind: ReduxKind, abs: bool, nan: bool, ty: DType) -> String {
    let (op, suffix) = match kind {
        ReduxKind::Add => ("add", if ty.is_signed() { "s32" } else { "u32" }),
        ReduxKind::And => ("and", "b32"),
        ReduxKind::Or => ("or", "b32"),
        ReduxKind::Xor => ("xor", "b32"),
        ReduxKind::Max => ("max", "s32"),
        ReduxKind::Min => ("min", "s32"),
        ReduxKind::Umax => ("max", "u32"),
        ReduxKind::Umin => ("min", "u32"),
        ReduxKind::Fmax => ("max", "f32"),
        ReduxKind::Fmin => ("min", "f32"),
    };
    format!("redux.sync.{op}{}{}.{suffix}", if abs { ".abs" } else { "" }, if nan { ".NaN" } else { "" })
}

/// The operation of an atomic ([`Op::Atomic`], PTX's `atom`) or a reduction
/// ([`Op::Red`], `red`). Whether `Min` and `Max` compare signed or unsigned
/// integers is the [`AtomicType`]'s.
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

impl AtomicKind {
    /// The PTX operation, without the dot (`add`, `exch`, `cas`).
    pub fn ptx(self) -> &'static str {
        match self {
            AtomicKind::Add => "add",
            AtomicKind::And => "and",
            AtomicKind::Or => "or",
            AtomicKind::Xor => "xor",
            AtomicKind::Min => "min",
            AtomicKind::Max => "max",
            AtomicKind::Exchange => "exch",
            AtomicKind::CompareExchange => "cas",
        }
    }
}

/// The PTX type of an atomic or a reduction: the `dtype=` of CuTe DSL's
/// `cute.arch.red` (`"b32"`, `"u32"`, `"s32"`, `"f32"`, `"f16x2"`, ...),
/// which the atomics derive from their operand. The bit types are for the
/// bit operations, exchange and compare-and-swap; the packed types
/// (`F16x2`, `Bf16x2`) add two Float16 or BFloat16 values held in an
/// [`Int32`](crate::Int32), each half atomically on its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AtomicType {
    B32,
    B64,
    U32,
    U64,
    S32,
    S64,
    F32,
    F64,
    F16,
    F16x2,
    Bf16,
    Bf16x2,
}

impl AtomicType {
    /// The PTX type, without the dot (`u32`, `bf16x2`).
    pub fn ptx(self) -> &'static str {
        match self {
            AtomicType::B32 => "b32",
            AtomicType::B64 => "b64",
            AtomicType::U32 => "u32",
            AtomicType::U64 => "u64",
            AtomicType::S32 => "s32",
            AtomicType::S64 => "s64",
            AtomicType::F32 => "f32",
            AtomicType::F64 => "f64",
            AtomicType::F16 => "f16",
            AtomicType::F16x2 => "f16x2",
            AtomicType::Bf16 => "bf16",
            AtomicType::Bf16x2 => "bf16x2",
        }
    }

    /// The bits of memory the operation updates.
    pub fn bits(self) -> usize {
        match self {
            AtomicType::F16 | AtomicType::Bf16 => 16,
            AtomicType::B64 | AtomicType::U64 | AtomicType::S64 | AtomicType::F64 => 64,
            _ => 32,
        }
    }

    /// Whether the values are floating point (packed ones too).
    pub fn is_float(self) -> bool {
        matches!(
            self,
            AtomicType::F32
                | AtomicType::F64
                | AtomicType::F16
                | AtomicType::F16x2
                | AtomicType::Bf16
                | AtomicType::Bf16x2
        )
    }

    /// Whether the type packs two half-precision values into 32 bits.
    pub fn is_packed(self) -> bool {
        matches!(self, AtomicType::F16x2 | AtomicType::Bf16x2)
    }

    /// Whether it compares signed integers (`s32`, `s64`).
    pub fn is_signed(self) -> bool {
        matches!(self, AtomicType::S32 | AtomicType::S64)
    }

    /// The Float16 or BFloat16 element of a half-precision type.
    pub fn half(self) -> Option<DType> {
        match self {
            AtomicType::F16 | AtomicType::F16x2 => Some(DType::F16),
            AtomicType::Bf16 | AtomicType::Bf16x2 => Some(DType::BF16),
            _ => None,
        }
    }

    /// The element type an operation of this type on a `value` addresses
    /// memory in: the half-precision element of a packed type (its index
    /// counts those), the operand's type otherwise.
    pub fn access_elem(self, value: DType) -> DType {
        if self.is_packed() { self.half().expect("packed types are half precision") } else { value }
    }

    /// The type CuTe DSL's atomics give an operation `kind` on `value`: the
    /// bit type for the bit operations, exchange and compare-and-swap, the
    /// signed or unsigned type for min and max, and for add the value's own
    /// type, except that a 64-bit add is `u64` (PTX has no `add.s64`; the
    /// sum's bits are the same).
    pub fn for_value(kind: AtomicKind, value: DType) -> AtomicType {
        let wide = value.bits() == 64;
        match (kind, value) {
            (_, DType::F32) => AtomicType::F32,
            (_, DType::F64) => AtomicType::F64,
            (_, DType::F16) => AtomicType::F16,
            (_, DType::BF16) => AtomicType::Bf16,
            (AtomicKind::Min | AtomicKind::Max, t) if t.is_signed() => {
                if wide {
                    AtomicType::S64
                } else {
                    AtomicType::S32
                }
            }
            (AtomicKind::Min | AtomicKind::Max, _) | (AtomicKind::Add, DType::U32 | DType::I64 | DType::U64) => {
                if wide { AtomicType::U64 } else { AtomicType::U32 }
            }
            (AtomicKind::Add, _) => AtomicType::S32,
            _ => {
                if wide {
                    AtomicType::B64
                } else {
                    AtomicType::B32
                }
            }
        }
    }

    /// Whether PTX has `atom` (or, with `red`, `red`) with operation `kind`
    /// on this type, from the PTX ISA's tables of scalar atomic and
    /// reduction operations: integer add on `u32`, `s32` and `u64`, min and
    /// max on the four integer types, the bit operations on `b32` and `b64`,
    /// float add on every float type (`f16` and the packed types with
    /// `.noftz`), exchange and compare-and-swap on `b32` and `b64` (`atom`
    /// only). There is no scalar float min or max.
    pub fn supports(self, kind: AtomicKind, red: bool) -> bool {
        use AtomicType::*;
        match kind {
            AtomicKind::Add => matches!(self, U32 | S32 | U64) || self.is_float(),
            AtomicKind::Min | AtomicKind::Max => matches!(self, U32 | S32 | U64 | S64),
            AtomicKind::And | AtomicKind::Or | AtomicKind::Xor => matches!(self, B32 | B64),
            AtomicKind::Exchange | AtomicKind::CompareExchange => !red && matches!(self, B32 | B64),
        }
    }

    /// `kind` and this type as PTX: `add.u32`, `add.noftz.f16x2` (the
    /// half-precision adds require `.noftz`), `cas.b64`.
    pub fn op_suffix(self, kind: AtomicKind) -> String {
        let noftz = if self.half().is_some() { ".noftz" } else { "" };
        format!("{}{noftz}.{}", kind.ptx(), self.ptx())
    }
}

/// The `atom` (`red` with `red`) instruction of an atomic with these
/// attributes on memory in `space`, as cute-rs emits it: the semantics and
/// scope always explicit, then the state space, the operation and the type,
/// e.g. `atom.relaxed.gpu.global.add.u32` or
/// `red.release.cta.shared.add.noftz.f16x2`. Shared memory is `.shared`,
/// which is `.shared::cta`, valid from PTX 2.0 (the `::cta` spelling needs
/// PTX 7.8).
pub fn atomic_inst_name(
    red: bool,
    kind: AtomicKind,
    ty: AtomicType,
    sem: MemoryOrder,
    scope: MemoryScope,
    space: MemSpace,
) -> String {
    use crate::tensor::AddressSpace;
    let space = match space {
        MemSpace::Shared | MemSpace::Raw(AddressSpace::Smem) => ".shared",
        // Another CTA's shared memory.
        MemSpace::Raw(AddressSpace::Dsmem) => ".shared::cluster",
        // A generic address: no state space.
        MemSpace::Raw(AddressSpace::Generic) => "",
        _ => ".global",
    };
    format!("{}.{}.{}{space}.{}", if red { "red" } else { "atom" }, sem.name(), scope.ptx(), ty.op_suffix(kind))
}
