//! External device functions, standalone device functions and multimem instructions.

use super::*;

// ---------------------------------------------------------------------------
// External device functions, standalone device
// functions and multimem instructions.
// ---------------------------------------------------------------------------

/// An external device function of a kernel ([`KernelIr::externs`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExternId(pub u32);

/// A parameter of an external device function: a value of a numeric type,
/// or a pointer (`!llvm.ptr<n>`) into an address space (generic, global or
/// shared).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternParam {
    Value(DType),
    Ptr(crate::tensor::AddressSpace),
}

impl ExternParam {
    /// The MLIR type of the parameter: `f32`, `i32`, `!llvm.ptr<1>`, ...
    /// (`!llvm.ptr` for a generic pointer).
    pub fn mlir(&self) -> String {
        match self {
            ExternParam::Value(t) => t.mlir().to_string(),
            ExternParam::Ptr(crate::tensor::AddressSpace::Generic) => "!llvm.ptr".to_string(),
            ExternParam::Ptr(space) => format!("!llvm.ptr<{}>", *space as u32),
        }
    }
}

/// The declaration of an external device function: its symbol, signature,
/// how calls to it are compiled and how the simulator runs them.
#[derive(Clone, Debug)]
pub struct ExternDecl {
    /// The symbol (the mangled name of an overloaded function).
    pub symbol: String,
    pub params: Vec<ExternParam>,
    /// At most one result.
    pub results: Vec<DType>,
    /// Calls may be inlined (CuTe DSL's `inline=True`, the default); without
    /// it every call is `func.call ... {no_inline}`.
    pub inline: bool,
    /// The prototype carries LLVM's `convergent` (the default), so that LLVM
    /// neither duplicates a call nor moves it into divergent code: the callee
    /// may hold barriers or warp collectives.
    pub convergent: bool,
    /// What the simulator runs for a call when it has no traced body: raw
    /// bits of the value arguments (0 for a pointer) in, the result's bits
    /// out.
    pub sim: Option<AsmSim>,
    /// Where it was first declared.
    pub loc: Loc,
}

impl ExternDecl {
    /// Whether `other` declares the same prototype (symbol, parameters and
    /// results); how calls are compiled and simulated may differ.
    pub fn same_prototype(&self, other: &ExternDecl) -> bool {
        self.symbol == other.symbol && self.params == other.params && self.results == other.results
    }

    /// The prototype as MLIR: `func.func private @sym(f32) -> f32`.
    pub fn prototype(&self) -> String {
        let params: Vec<String> = self.params.iter().map(ExternParam::mlir).collect();
        let results = match self.results.as_slice() {
            [] => String::new(),
            [t] => format!(" -> {}", t.mlir()),
            ts => format!(" -> ({})", ts.iter().map(|t| t.mlir()).collect::<Vec<_>>().join(", ")),
        };
        let attrs = if self.convergent { " attributes {passthrough = [\"convergent\"]}" } else { "" };
        format!("func.func private @{}({}){results}{attrs}", self.symbol, params.join(", "))
    }
}

/// The element type of a `multimem.ld_reduce` (CuTe DSL's typed variants):
/// packed pairs of Float16 or BFloat16 (accumulated in Float32), Float32
/// (accumulated in Float32), and packed quadruples of FP8 (accumulated in
/// Float16), each in one 32-bit register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MultimemType {
    F16x2,
    Bf16x2,
    F32,
    E4m3x4,
    E5m2x4,
}

impl MultimemType {
    /// The PTX type: `f16x2`, `bf16x2`, `f32`, `e4m3x4`, `e5m2x4`.
    pub fn ptx(self) -> &'static str {
        match self {
            MultimemType::F16x2 => "f16x2",
            MultimemType::Bf16x2 => "bf16x2",
            MultimemType::F32 => "f32",
            MultimemType::E4m3x4 => "e4m3x4",
            MultimemType::E5m2x4 => "e5m2x4",
        }
    }

    /// The `.acc::` precision qualifier, if any: Float32 for the halves,
    /// Float16 for FP8, none for Float32.
    pub fn acc(self) -> Option<&'static str> {
        match self {
            MultimemType::F16x2 | MultimemType::Bf16x2 => Some(".acc::f32"),
            MultimemType::E4m3x4 | MultimemType::E5m2x4 => Some(".acc::f16"),
            MultimemType::F32 => None,
        }
    }

    /// The element type, and how many elements a 32-bit register holds.
    pub fn element(self) -> (DType, usize) {
        match self {
            MultimemType::F16x2 => (DType::F16, 2),
            MultimemType::Bf16x2 => (DType::BF16, 2),
            MultimemType::F32 => (DType::F32, 1),
            MultimemType::E4m3x4 => (DType::F8E4M3FN, 4),
            MultimemType::E5m2x4 => (DType::F8E5M2, 4),
        }
    }

    pub fn is_fp8(self) -> bool {
        matches!(self, MultimemType::E4m3x4 | MultimemType::E5m2x4)
    }
}

/// A `multimem` instruction ([`Op::Multimem`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MultimemOp {
    /// `multimem.ld_reduce.weak.global.add{.acc}{.v4,.v2}.type`: the sum over
    /// every replica, `regs` (4, 2 or 1) 32-bit registers.
    LdReduce { ty: MultimemType, regs: u8 },
    /// `multimem.st.weak.global{.v4,.v2}.f32`: `regs` 32-bit registers to
    /// every replica.
    St { regs: u8 },
    /// `multimem.red.{order}.{scope}.global.add.s32 [a], 1`: add one to the
    /// 32-bit integer in every replica.
    RedAdd1 { order: MemoryOrder, scope: MemoryScope },
}

impl MultimemOp {
    /// The bytes it accesses at its address in each replica.
    pub fn bytes(&self) -> usize {
        match self {
            MultimemOp::LdReduce { regs, .. } | MultimemOp::St { regs } => 4 * *regs as usize,
            MultimemOp::RedAdd1 { .. } => 4,
        }
    }

    /// The PTX instruction, without operands.
    pub fn ptx(&self) -> String {
        let vec = |regs: u8| match regs {
            4 => ".v4",
            2 => ".v2",
            _ => "",
        };
        match *self {
            MultimemOp::LdReduce { ty, regs } => {
                format!("multimem.ld_reduce.weak.global.add{}{}.{}", ty.acc().unwrap_or(""), vec(regs), ty.ptx())
            }
            MultimemOp::St { regs } => format!("multimem.st.weak.global{}.f32", vec(regs)),
            MultimemOp::RedAdd1 { order, scope } => {
                let order = if order == MemoryOrder::Release { "release" } else { "relaxed" };
                let scope = if scope == MemoryScope::Sys { "sys" } else { "gpu" };
                format!("multimem.red.{order}.{scope}.global.add.s32")
            }
        }
    }
}

/// A standalone device function's result ([`KernelIr::function`]): the
/// kernel IR is then a `__device__` function, not a kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceFunctionSig {
    /// The value returned (the body's yield), if any.
    pub result: Option<ValueId>,
}
