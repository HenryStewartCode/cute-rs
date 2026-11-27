//! Function kinds for floating-point, integer and half-precision arithmetic, packed forms and conversions, with their PTX names.

use super::*;

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

/// The PTX instruction an [`Op::FloatInst`] issues, e.g. `ex2.approx.ftz.f32`
/// or `max.NaN.xorsign.abs.f32`.
pub fn float_inst_name(func: FloatFn, mode: FloatMode, ftz: bool, nan: bool, xorsign_abs: bool, ty: DType) -> String {
    format!(
        "{}{}{}{}{}.{}",
        func.ptx(),
        mode.ptx(),
        if ftz { ".ftz" } else { "" },
        if nan { ".NaN" } else { "" },
        if xorsign_abs { ".xorsign.abs" } else { "" },
        ty.name()
    )
}

/// How `prmt` picks the four bytes of its result from the eight bytes of its
/// two sources (`{b, a}`, byte 0 the low byte of `a`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PrmtMode {
    /// The generic form, CuTe DSL's `prmt`: the low 16 bits of the control
    /// operand are four 4-bit selectors, one per result byte. A selector's
    /// low 3 bits name a source byte, and its top bit replicates that
    /// byte's sign bit instead of copying it.
    #[default]
    Generic,
    /// `.f4e` (forward 4 extract): result byte `i` is source byte `s + i`,
    /// where `s` is the low 2 bits of the control operand.
    F4e,
    /// `.b4e` (backward 4 extract): result byte `i` is source byte
    /// `(s - i) mod 8`.
    B4e,
    /// `.rc8` (replicate 8): every result byte is source byte `s`.
    Rc8,
    /// `.ecl` (edge clamp left): result byte `i` is source byte `max(i, s)`.
    Ecl,
    /// `.ecr` (edge clamp right): result byte `i` is source byte `min(i, s)`.
    Ecr,
    /// `.rc16` (replicate 16): both halves are source half-word `s & 1`.
    Rc16,
}

impl PrmtMode {
    /// The PTX modifier, with its dot (empty for the generic form).
    pub fn ptx(self) -> &'static str {
        match self {
            PrmtMode::Generic => "",
            PrmtMode::F4e => ".f4e",
            PrmtMode::B4e => ".b4e",
            PrmtMode::Rc8 => ".rc8",
            PrmtMode::Ecl => ".ecl",
            PrmtMode::Ecr => ".ecr",
            PrmtMode::Rc16 => ".rc16",
        }
    }
}

/// The PTX integer and bit instructions that [`Op::IntInst`] issues: CuTe
/// DSL's `cute.arch` integer helpers. Immediates (`lop3`'s lookup table)
/// and modifiers belong to the function; the operands are values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntFn {
    /// `bfe.{s,u}{32,64}`: extract a bit field. Operands: value, start,
    /// length.
    Bfe,
    /// `bfi.b{32,64}`: insert a bit field. Operands: replacement (the
    /// field), value, start, length.
    Bfi,
    /// `lop3.b32` with the lookup table `lut` as its immediate: any bitwise
    /// function of three operands.
    Lop3 { lut: u8 },
    /// `shf.{l,r}.{clamp,wrap}.b32`, a funnel shift: operand a holds bits
    /// 31:0 of a 64-bit value and operand b bits 63:32; the third operand is
    /// the shift amount, clamped to 32 or taken mod 32.
    Shf { left: bool, clamp: bool },
    /// `prmt.b32{.mode}`: pick four of the eight bytes of operands a (bytes
    /// 0-3) and b (bytes 4-7), as the third operand selects.
    Prmt { mode: PrmtMode },
    /// `mul.wide.{s,u}{16,32}`: the whole product of two 16- or 32-bit
    /// integers, as an integer twice as wide.
    MulWide,
    /// `mul24.{lo,hi}.{s,u}32`: the 48-bit product of the low 24 bits of two
    /// 32-bit integers, its bits 31:0 (`lo`) or 47:16 (`hi`).
    Mul24 { hi: bool },
    /// `mad24.{lo,hi}.{s,u}32`: [`IntFn::Mul24`] plus a third operand,
    /// wrapping.
    Mad24 { hi: bool },
    /// `add.sat.s32`: a sum clamped to the Int32 range.
    AddSat,
    /// `sub.sat.s32`: a difference clamped to the Int32 range.
    SubSat,
    /// `dp4a.s32.s32`: operand c plus the dot product of the four signed
    /// bytes of a with the four signed bytes of b. CuTe DSL's int8
    /// conversions (`cvt_i8x4_to_f32x4`) sign-extend each byte with it.
    Dp4a,
}

impl IntFn {
    /// CuTe DSL's name for the operation (`cute.arch.<name>`), for messages.
    pub fn cute_name(self) -> &'static str {
        match self {
            IntFn::Bfe => "bfe",
            IntFn::Bfi => "bfi",
            IntFn::Lop3 { .. } => "lop3",
            IntFn::Shf { .. } => "shf",
            IntFn::Prmt { .. } => "prmt",
            IntFn::MulWide => "mul_wide",
            IntFn::Mul24 { .. } => "mul24",
            IntFn::Mad24 { .. } => "mad24",
            IntFn::AddSat => "add_sat_int",
            IntFn::SubSat => "sub_sat_int",
            IntFn::Dp4a => "dp4a",
        }
    }

    /// The operands' names, as CuTe DSL's parameters.
    pub fn operand_names(self) -> &'static [&'static str] {
        match self {
            IntFn::Bfe => &["value", "start", "length"],
            IntFn::Bfi => &["replacement", "value", "start", "length"],
            IntFn::Lop3 { .. } | IntFn::Mad24 { .. } | IntFn::Dp4a => &["a", "b", "c"],
            IntFn::Shf { .. } => &["a", "b", "shift"],
            IntFn::Prmt { .. } => &["src", "src_reg_shifted", "prmt_indices"],
            IntFn::MulWide | IntFn::Mul24 { .. } | IntFn::AddSat | IntFn::SubSat => &["a", "b"],
        }
    }

    /// Number of operands.
    pub fn arity(self) -> usize {
        self.operand_names().len()
    }

    /// How many leading operands are data, all of one type; the rest are
    /// Int32 bit positions, lengths or shift amounts.
    pub fn data_operands(self) -> usize {
        match self {
            IntFn::Bfe => 1,
            IntFn::Bfi | IntFn::Shf { .. } => 2,
            _ => self.arity(),
        }
    }

    /// The types the data operands may have.
    pub fn operand_types(self) -> &'static [DType] {
        const WORDS: &[DType] = &[DType::I32, DType::U32, DType::I64, DType::U64];
        const WORD32: &[DType] = &[DType::I32, DType::U32];
        match self {
            IntFn::Bfe | IntFn::Bfi => WORDS,
            IntFn::Lop3 { .. } | IntFn::Shf { .. } | IntFn::Prmt { .. } | IntFn::Mul24 { .. } | IntFn::Mad24 { .. } => {
                WORD32
            }
            IntFn::MulWide => &[DType::I16, DType::U16, DType::I32, DType::U32],
            IntFn::AddSat | IntFn::SubSat | IntFn::Dp4a => &[DType::I32],
        }
    }

    /// The result type for data operands of type `ty`: twice as wide for
    /// [`IntFn::MulWide`], `ty` itself for the others.
    pub fn result_type(self, ty: DType) -> DType {
        match (self, ty) {
            (IntFn::MulWide, DType::I16) => DType::I32,
            (IntFn::MulWide, DType::U16) => DType::U32,
            (IntFn::MulWide, DType::I32) => DType::I64,
            (IntFn::MulWide, DType::U32) => DType::U64,
            _ => ty,
        }
    }

    /// Whether the first two operands can be swapped.
    pub fn is_commutative(self) -> bool {
        matches!(self, IntFn::MulWide | IntFn::Mul24 { .. } | IntFn::Mad24 { .. } | IntFn::AddSat | IntFn::Dp4a)
    }
}

/// The PTX instruction an [`Op::IntInst`] issues on data operands of type
/// `ty`, e.g. `bfe.s64` or `shf.l.clamp.b32` (immediates not included).
pub fn int_inst_name(func: IntFn, ty: DType) -> String {
    let sign = if ty.is_signed() { 's' } else { 'u' };
    let half = |hi: bool| if hi { "hi" } else { "lo" };
    let w = ty.bits();
    match func {
        IntFn::Bfe => format!("bfe.{sign}{w}"),
        IntFn::Bfi => format!("bfi.b{w}"),
        IntFn::Lop3 { .. } => "lop3.b32".into(),
        IntFn::Shf { left, clamp } => {
            format!("shf.{}.{}.b32", if left { "l" } else { "r" }, if clamp { "clamp" } else { "wrap" })
        }
        IntFn::Prmt { mode } => format!("prmt.b32{}", mode.ptx()),
        IntFn::MulWide => format!("mul.wide.{sign}{w}"),
        IntFn::Mul24 { hi } => format!("mul24.{}.{sign}32", half(hi)),
        IntFn::Mad24 { hi } => format!("mad24.{}.{sign}32", half(hi)),
        IntFn::AddSat => "add.sat.s32".into(),
        IntFn::SubSat => "sub.sat.s32".into(),
        IntFn::Dp4a => "dp4a.s32.s32".into(),
    }
}

/// The PTX half-precision floating-point instructions that [`Op::HalfInst`]
/// issues: CuTe DSL's packed and scalar half helpers (`add_packed_f16x2`,
/// `fma_bf16`, `exp2_packed_bf16x2`, ...).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum HalfFn {
    /// `add`: `a + b`.
    Add,
    /// `sub`: `a - b`.
    Sub,
    /// `fma`: `a * b + c`, rounded once.
    Fma,
    /// `ex2.approx`: an approximation of `2^a`.
    Ex2,
}

impl HalfFn {
    /// The PTX instruction name.
    pub fn ptx(self) -> &'static str {
        match self {
            HalfFn::Add => "add",
            HalfFn::Sub => "sub",
            HalfFn::Fma => "fma",
            HalfFn::Ex2 => "ex2",
        }
    }

    /// Number of operands.
    pub fn arity(self) -> usize {
        match self {
            HalfFn::Add | HalfFn::Sub => 2,
            HalfFn::Fma => 3,
            HalfFn::Ex2 => 1,
        }
    }
}

/// The operand types of PTX's half-precision instructions: one 16-bit
/// float, or two of them packed into a 32-bit register with element 0 in
/// bits 15:0 (`.f16x2`, `.bf16x2`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum HalfType {
    F16,
    Bf16,
    F16x2,
    Bf16x2,
}

impl HalfType {
    /// The PTX type suffix, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            HalfType::F16 => "f16",
            HalfType::Bf16 => "bf16",
            HalfType::F16x2 => "f16x2",
            HalfType::Bf16x2 => "bf16x2",
        }
    }

    /// The element type, Float16 or BFloat16.
    pub fn elem(self) -> DType {
        match self {
            HalfType::F16 | HalfType::F16x2 => DType::F16,
            HalfType::Bf16 | HalfType::Bf16x2 => DType::BF16,
        }
    }

    /// Whether two elements are packed into one Int32.
    pub fn is_packed(self) -> bool {
        matches!(self, HalfType::F16x2 | HalfType::Bf16x2)
    }

    /// The type of the operands and the result in the IR: Int32 for the
    /// packed forms, the element type otherwise.
    pub fn value_type(self) -> DType {
        if self.is_packed() { DType::I32 } else { self.elem() }
    }
}

/// The modifiers of an [`Op::HalfInst`] (CuTe DSL's `rnd='rn'`, `oob=`,
/// `ftz=`, `sat=` and `relu=`), each a PTX modifier written in this order:
/// `fma.rn.oob.relu.bf16x2`, `add.rn.ftz.sat.f16x2`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct HalfModifiers {
    /// `.rn`: round to nearest even, explicitly (`add` and `sub` round so
    /// without it too, but may then be fused with a multiplication).
    pub rn: bool,
    /// `.oob`: a zero result where an operand is the out-of-bounds NaN.
    pub oob: bool,
    /// `.ftz`: flush subnormal inputs and results to zero (Float16 only).
    pub ftz: bool,
    /// `.sat`: clamp the result to `[0, 1]`, a NaN to `+0` (Float16 only).
    pub sat: bool,
    /// `.relu`: a negative result becomes `+0`, a NaN the canonical NaN.
    pub relu: bool,
}

/// The PTX instruction an [`Op::HalfInst`] issues, e.g.
/// `fma.rn.oob.relu.bf16x2`, `add.rn.ftz.sat.f16x2` or
/// `ex2.approx.ftz.bf16x2` (the BFloat16 `ex2` always flushes subnormals).
pub fn half_inst_name(func: HalfFn, ty: HalfType, mods: HalfModifiers) -> String {
    if func == HalfFn::Ex2 {
        let ftz = if ty.elem() == DType::BF16 { ".ftz" } else { "" };
        return format!("ex2.approx{ftz}.{}", ty.ptx());
    }
    let flag = |on: bool, text: &'static str| if on { text } else { "" };
    format!(
        "{}{}{}{}{}{}.{}",
        func.ptx(),
        flag(mods.rn, ".rn"),
        flag(mods.oob, ".oob"),
        flag(mods.ftz, ".ftz"),
        flag(mods.sat, ".sat"),
        flag(mods.relu, ".relu"),
        ty.ptx()
    )
}

/// The two FP8 formats PTX converts to and from on sm_89 and newer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fp8Kind {
    /// OCP E4M3 (`Float8E4M3FN`, PTX `.e4m3`).
    E4M3,
    /// OCP E5M2 (`Float8E5M2`, PTX `.e5m2`).
    E5M2,
}

impl Fp8Kind {
    /// The format's type.
    pub fn dtype(self) -> DType {
        match self {
            Fp8Kind::E4M3 => DType::F8E4M3FN,
            Fp8Kind::E5M2 => DType::F8E5M2,
        }
    }

    /// The PTX pair type, `e4m3x2` or `e5m2x2`.
    pub fn ptx(self) -> &'static str {
        match self {
            Fp8Kind::E4M3 => "e4m3x2",
            Fp8Kind::E5M2 => "e5m2x2",
        }
    }
}

/// The shape and element type of an [`Op::StMatrix`] (the PTX ISA's
/// `stmatrix`, "Collectively store one or more matrices to shared memory").
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum StMatrixForm {
    /// `.m8n8` with `.b16` (sm_90, PTX 7.8): 8x8 matrices of 16-bit
    /// elements, optionally transposed.
    #[default]
    M8n8B16,
    /// `.m16n8` with `.b8`, `.trans` mandatory: 16x8 matrices of 8-bit
    /// elements stored transposed. Architecture-specific (sm_100a, sm_101a or
    /// sm_110a and sm_120a from PTX 8.6; the f families from 8.8).
    M16n8B8,
}

impl StMatrixForm {
    /// The PTX shape qualifier.
    pub fn shape(self) -> &'static str {
        match self {
            StMatrixForm::M8n8B16 => "m8n8",
            StMatrixForm::M16n8B8 => "m16n8",
        }
    }

    /// The PTX element type.
    pub fn format(self) -> &'static str {
        match self {
            StMatrixForm::M8n8B16 => "b16",
            StMatrixForm::M16n8B8 => "b8",
        }
    }

    /// The PTX instruction (operands not included).
    pub fn ptx(self, num: u32, trans: bool) -> String {
        format!(
            "stmatrix.sync.aligned.{}.x{num}{}.shared.{}",
            self.shape(),
            if trans { ".trans" } else { "" },
            self.format()
        )
    }
}

/// The shape and element format of an [`Op::LdMatrix`] (the PTX ISA's
/// `ldmatrix`, "Warp-level matrix load instruction").
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum LdMatrixForm {
    /// `.m8n8` with `.b16`: 8x8 matrices of 16-bit elements (sm_75, PTX 6.5).
    #[default]
    M8n8B16,
    /// `.m16n16` (`.trans` mandatory, `.x1` or `.x2`): 16x16 matrices of
    /// 8-bit elements, `.b8`, or with `unpack` 4 or 6 the padded sub-byte
    /// sources `.b8x16.b4x16_p64` and `.b8x16.b6x16_p32` expanded to 8-bit
    /// containers. Architecture-specific (sm_100a, sm_101a/sm_110a, sm_120a
    /// from PTX 8.6; the f families from 8.8).
    M16n16B8 { unpack: Option<u8> },
    /// `.m8n16` (`.x1`, `.x2` or `.x4`): 8x16 matrices of padded 4-bit
    /// (`unpack` 4, `.b8x16.b4x16_p64`) or 6-bit (`unpack` 6,
    /// `.b8x16.b6x16_p32`) elements expanded to 8-bit containers. The
    /// targets of [`LdMatrixForm::M16n16B8`].
    M8n16B8 { unpack: u8 },
}

impl LdMatrixForm {
    /// The PTX shape qualifier.
    pub fn shape(self) -> &'static str {
        match self {
            LdMatrixForm::M8n8B16 => "m8n8",
            LdMatrixForm::M16n16B8 { .. } => "m16n16",
            LdMatrixForm::M8n16B8 { .. } => "m8n16",
        }
    }

    /// The PTX element type or `.dst_fmt.src_fmt` pair.
    pub fn format(self) -> &'static str {
        match self.unpack() {
            None if self == LdMatrixForm::M8n8B16 => "b16",
            None => "b8",
            Some(4) => "b8x16.b4x16_p64",
            Some(_) => "b8x16.b6x16_p32",
        }
    }

    /// The source element width that is expanded to 8 bits (4 or 6), if any.
    pub fn unpack(self) -> Option<u8> {
        match self {
            LdMatrixForm::M8n8B16 => None,
            LdMatrixForm::M16n16B8 { unpack } => unpack,
            LdMatrixForm::M8n16B8 { unpack } => Some(unpack),
        }
    }

    /// The rows (one per supplying lane) of each matrix.
    pub fn rows(self) -> u32 {
        match self {
            LdMatrixForm::M16n16B8 { .. } => 16,
            _ => 8,
        }
    }

    /// The results per matrix: two 16-bit values for `.m8n8`, two 32-bit
    /// registers for `.m16n16` and one for `.m8n16`.
    pub fn results_per_matrix(self) -> u32 {
        match self {
            LdMatrixForm::M8n8B16 | LdMatrixForm::M16n16B8 { .. } => 2,
            LdMatrixForm::M8n16B8 { .. } => 1,
        }
    }

    /// The PTX instruction (operands not included).
    pub fn ptx(self, num: u32, trans: bool) -> String {
        format!(
            "ldmatrix.sync.aligned.{}.x{num}{}.shared.{}",
            self.shape(),
            if trans { ".trans" } else { "" },
            self.format()
        )
    }
}

/// The OCP MX FP4 and FP6 formats PTX converts to and from in pairs
/// (sm_100a, sm_101a and sm_120a, PTX 8.6).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MxKind {
    /// FP4 E2M1 (`Float4E2M1FN`, PTX `.e2m1x2`: a pair in one byte).
    E2M1,
    /// FP6 E2M3 (`Float6E2M3FN`, PTX `.e2m3x2`: a pair in two bytes, each
    /// value in a byte's low 6 bits).
    E2M3,
    /// FP6 E3M2 (`Float6E3M2FN`, PTX `.e3m2x2`).
    E3M2,
}

impl MxKind {
    /// The format's type.
    pub fn dtype(self) -> DType {
        match self {
            MxKind::E2M1 => DType::F4E2M1FN,
            MxKind::E2M3 => DType::F6E2M3FN,
            MxKind::E3M2 => DType::F6E3M2FN,
        }
    }

    /// The PTX pair type, `e2m1x2`, `e2m3x2` or `e3m2x2`.
    pub fn ptx(self) -> &'static str {
        match self {
            MxKind::E2M1 => "e2m1x2",
            MxKind::E2M3 => "e2m3x2",
            MxKind::E3M2 => "e3m2x2",
        }
    }
}

/// The arithmetic of a [`CvtFn::Packed`] instruction: PTX's packed `.f32x2` operations (sm_100, PTX
/// 8.6) and the Rubin-only mixed-precision ones (sm_107f, PTX 9.4). A pair of
/// Float32 values is a `u64` (element 0 in the low half) and a pair of
/// Float16 or BFloat16 values an `i32`; each element is computed on its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PackedOp {
    /// `add{.rnd}{.ftz}.f32x2`: `a + b` on two Float32 pairs.
    Add,
    /// `sub{.rnd}{.ftz}.f32x2`.
    Sub,
    /// `mul{.rnd}{.ftz}.f32x2`.
    Mul,
    /// `fma.rnd{.ftz}.f32x2`: `a * b + c`, rounded once.
    Fma,
    /// `add.rn.{f16x2,bf16x2}.f32x2`: two Float32 sums, each rounded once
    /// to a Float16 or BFloat16 (an `i32` pair).
    AddToHalf { bf16: bool },
    /// `sub.rn.{f16x2,bf16x2}.f32x2`.
    SubToHalf { bf16: bool },
    /// `add{.rnd}.f32x2.{f16x2,bf16x2}`: a half pair (widened exactly) plus a
    /// Float32 pair, rounded once.
    AddHalf { bf16: bool },
    /// `sub{.rnd}.f32x2.{f16x2,bf16x2}`: the half pair minus the Float32 pair.
    SubHalf { bf16: bool },
    /// `fma.rnd.f32x2.{f16x2,bf16x2}`: half pair times a Float32 pair plus a
    /// Float32 pair, rounded once.
    FmaHalf { bf16: bool },
}

/// The PTX conversions that [`Op::Cvt`] issues beyond the plain numeric
/// conversions of [`Op::Cast`]: CuTe DSL's `cvt_*` helpers, and the FP8 and
/// UE8M0 conversions behind [`Value::to`](crate::Value::to). Pairs are
/// packed with element 0 in the low half (PTX takes element 1 first and
/// puts it in the high half).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CvtFn {
    /// `cvt.rn.satfinite.bf16x2.f32`: two Float32 values (element 0, then
    /// element 1) to BFloat16, rounded to nearest even and packed into an
    /// Int32 with element 0 in the low half. A NaN gives the canonical NaN
    /// and a magnitude above BFloat16's largest finite value (infinities
    /// included) that value with its sign. PTX takes element 1 first.
    F32x2ToBf16x2,
    /// `cvt.rna.tf32.f32`: a Float32 rounded to TF32 (10 stored mantissa
    /// bits, to nearest with ties away from zero), as an Int32 holding the
    /// bits with the low 13 cleared.
    F32ToTf32,
    /// `cvt.rn.bf16.s8`: an Int8 to BFloat16, exactly.
    S8ToBf16,
    /// `cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2` with the scale
    /// factor `scale` for both elements (the 16-bit operand `0x{scale}{scale}`):
    /// the two bytes of an Int16 (byte 0 low), each read as an s2f6
    /// fixed-point value (its int8 value times 2^-6) and multiplied by the
    /// UE8M0 scale `2^(scale - 127)`, as BFloat16 packed into an Int32 (byte
    /// 0's in the low half). `0x85` (2^6) gives the int8 values
    /// (`cvt_i8x2_to_bf16x2`), `0x81` (2^-4 overall) the high nibbles of the
    /// bytes as signed int4 values (`cvt_i4x2_to_bf16x2(with_shuffle=True)`).
    S2f6x2ToBf16x2 { scale: u8 },
    /// `cvt.rn.satfinite{.relu}.{e4m3x2,e5m2x2}.f32`: two Float32 values to
    /// FP8, rounded to nearest even, a magnitude past the largest finite
    /// value saturating to it (sign kept), packed into an Int16.
    F32x2ToF8x2 { to: Fp8Kind, relu: bool },
    /// `cvt.rn.satfinite{.relu}.{e4m3x2,e5m2x2}.f16x2`: a packed pair of
    /// Float16 values (an Int32) to FP8, as above, packed into an Int16.
    F16x2ToF8x2 { to: Fp8Kind, relu: bool },
    /// `cvt.rn{.relu}.f16x2.{e4m3x2,e5m2x2}`: a pair of FP8 values (an Int16)
    /// to Float16, exactly, packed into an Int32.
    F8x2ToF16x2 { from: Fp8Kind, relu: bool },
    /// `cvt.{rz,rp}.satfinite.ue8m0x2.f32`: two Float32 values to UE8M0
    /// (`round_up` picks `.rp`), packed into an Int16.
    F32x2ToUe8m0x2 { round_up: bool },
    /// `cvt.rp.satfinite.ue5m3x2.f32`:
    /// two Float32 values to UE5M3 (`FloatNV8E5M3FNU`), rounded toward +inf
    /// and saturating to the largest finite value (PTX ISA 9.4), packed into
    /// an Int16 with element 0 in the low byte (PTX takes element 1 first).
    F32x2ToUe5m3x2,
    /// `cvt.{rz,rp}{.satfinite}.ue8m0x2.bf16x2`: a packed pair of BFloat16
    /// values (an Int32) to UE8M0, packed into an Int16. (The PTX ISA lists
    /// a `.relu` form, which ptxas 12.8 refuses.)
    Bf16x2ToUe8m0x2 { round_up: bool, satfinite: bool },
    /// `cvt.rn.bf16x2.ue8m0x2`: a pair of UE8M0 values (an Int16) to
    /// BFloat16, exactly, packed into an Int32.
    Ue8m0x2ToBf16x2,
    /// `cvt.rn.satfinite{.relu}.{e2m1x2,e2m3x2,e3m2x2}.f32`: two Float32
    /// values to FP4 or FP6, rounded to nearest even, a magnitude past the
    /// largest finite value (infinities included) saturating to it with its
    /// sign and a NaN giving `+MAX_NORM`, in an Int16: an FP4 pair in the low
    /// byte (element 0 in the low nibble), an FP6 pair in the two bytes'
    /// low 6 bits (element 0 in the low byte).
    F32x2ToMx { to: MxKind, relu: bool },
    /// `cvt.rn{.relu}.f16x2.{e2m1x2,e2m3x2,e3m2x2}`: an FP4 or FP6 pair, in
    /// [`CvtFn::F32x2ToMx`]'s Int16 form, to Float16, exactly, packed into an
    /// Int32 (element 0 in the low half).
    MxToF16x2 { from: MxKind, relu: bool },
    /// A packed arithmetic instruction ([`PackedOp`]) with a rounding mode
    /// (`None`: no `.rnd`, which PTX lets ptxas fuse; computed as `.rn`) and
    /// `ftz`. Pure.
    Packed { op: PackedOp, rnd: Option<RoundingMode>, ftz: bool },
}

impl CvtFn {
    /// CuTe DSL's name for the helper that issues it (cute-rs's for the
    /// FP8 and UE8M0 pairs), for messages.
    pub fn cute_name(self) -> &'static str {
        match self {
            CvtFn::F32x2ToBf16x2 => "cvt_f32x2_bf16x2",
            CvtFn::F32ToTf32 => "cvt_f32_tf32",
            CvtFn::S8ToBf16 => "cvt_i8_bf16",
            CvtFn::S2f6x2ToBf16x2 { scale: 0x85 } => "cvt_i8x2_to_bf16x2",
            CvtFn::S2f6x2ToBf16x2 { .. } => "cvt_i4_bf16_intrinsic(with_shuffle=True)",
            CvtFn::F32x2ToF8x2 { .. } => "cvt_f32x2_to_f8x2",
            CvtFn::F16x2ToF8x2 { .. } => "cvt_f16x2_to_f8x2",
            CvtFn::F8x2ToF16x2 { .. } => "cvt_f8x2_to_f16x2",
            CvtFn::F32x2ToUe8m0x2 { .. } => "cvt_f32x2_to_ue8m0x2",
            CvtFn::F32x2ToUe5m3x2 => "cvt_f32x4_to_fnv8e5m3x4",
            CvtFn::Bf16x2ToUe8m0x2 { .. } => "cvt_bf16x2_to_ue8m0x2",
            CvtFn::Ue8m0x2ToBf16x2 => "cvt_ue8m0x2_to_bf16x2",
            CvtFn::F32x2ToMx { to: MxKind::E2M1, .. } => "cvt_f32x2_to_f4e2m1x2",
            CvtFn::F32x2ToMx { .. } => "cvt_f32x2_to_f6x2",
            CvtFn::MxToF16x2 { from: MxKind::E2M1, .. } => "cvt_f4e2m1x2_to_f16x2",
            CvtFn::MxToF16x2 { .. } => "cvt_f6x2_to_f16x2",
            CvtFn::Packed { op, .. } => match op {
                PackedOp::Add => "add_packed_f32x2",
                PackedOp::Sub => "sub_packed_f32x2",
                PackedOp::Mul => "mul_packed_f32x2",
                PackedOp::Fma => "fma_packed_f32x2",
                PackedOp::AddToHalf { bf16: false } => "add_packed_f16x2_f32x2_f32x2",
                PackedOp::AddToHalf { bf16: true } => "add_packed_bf16x2_f32x2_f32x2",
                PackedOp::SubToHalf { bf16: false } => "sub_packed_f16x2_f32x2_f32x2",
                PackedOp::SubToHalf { bf16: true } => "sub_packed_bf16x2_f32x2_f32x2",
                PackedOp::AddHalf { bf16: false } => "add_packed_f32x2_f16x2_f32x2",
                PackedOp::AddHalf { bf16: true } => "add_packed_f32x2_bf16x2_f32x2",
                PackedOp::SubHalf { bf16: false } => "sub_packed_f32x2_f16x2_f32x2",
                PackedOp::SubHalf { bf16: true } => "sub_packed_f32x2_bf16x2_f32x2",
                PackedOp::FmaHalf { bf16: false } => "fma_packed_f32x2_f16x2_f32x2_f32x2",
                PackedOp::FmaHalf { bf16: true } => "fma_packed_f32x2_bf16x2_f32x2_f32x2",
            },
        }
    }

    /// The PTX instruction (operands not included).
    pub fn ptx(self) -> String {
        let flag = |on: bool, name: &str| if on { name.to_string() } else { String::new() };
        let round = |up: bool| if up { "rp" } else { "rz" };
        match self {
            CvtFn::F32x2ToBf16x2 => "cvt.rn.satfinite.bf16x2.f32".into(),
            CvtFn::F32ToTf32 => "cvt.rna.tf32.f32".into(),
            CvtFn::S8ToBf16 => "cvt.rn.bf16.s8".into(),
            CvtFn::S2f6x2ToBf16x2 { .. } => "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2".into(),
            CvtFn::F32x2ToF8x2 { to, relu } => format!("cvt.rn.satfinite{}.{}.f32", flag(relu, ".relu"), to.ptx()),
            CvtFn::F16x2ToF8x2 { to, relu } => format!("cvt.rn.satfinite{}.{}.f16x2", flag(relu, ".relu"), to.ptx()),
            CvtFn::F8x2ToF16x2 { from, relu } => format!("cvt.rn{}.f16x2.{}", flag(relu, ".relu"), from.ptx()),
            CvtFn::F32x2ToUe8m0x2 { round_up } => format!("cvt.{}.satfinite.ue8m0x2.f32", round(round_up)),
            CvtFn::F32x2ToUe5m3x2 => "cvt.rp.satfinite.ue5m3x2.f32".into(),
            CvtFn::Bf16x2ToUe8m0x2 { round_up, satfinite } => {
                format!("cvt.{}{}.ue8m0x2.bf16x2", round(round_up), flag(satfinite, ".satfinite"))
            }
            CvtFn::Ue8m0x2ToBf16x2 => "cvt.rn.bf16x2.ue8m0x2".into(),
            CvtFn::F32x2ToMx { to, relu } => format!("cvt.rn.satfinite{}.{}.f32", flag(relu, ".relu"), to.ptx()),
            CvtFn::MxToF16x2 { from, relu } => format!("cvt.rn{}.f16x2.{}", flag(relu, ".relu"), from.ptx()),
            CvtFn::Packed { op, rnd, ftz } => {
                let r = rnd.map(|r| format!(".{}", r.ptx())).unwrap_or_default();
                let half = |bf16: bool| if bf16 { "bf16x2" } else { "f16x2" };
                match op {
                    PackedOp::Add => format!("add{r}{}.f32x2", flag(ftz, ".ftz")),
                    PackedOp::Sub => format!("sub{r}{}.f32x2", flag(ftz, ".ftz")),
                    PackedOp::Mul => format!("mul{r}{}.f32x2", flag(ftz, ".ftz")),
                    PackedOp::Fma => format!("fma{r}{}.f32x2", flag(ftz, ".ftz")),
                    PackedOp::AddToHalf { bf16 } => format!("add.rn.{}.f32x2", half(bf16)),
                    PackedOp::SubToHalf { bf16 } => format!("sub.rn.{}.f32x2", half(bf16)),
                    PackedOp::AddHalf { bf16 } => format!("add{r}.f32x2.{}", half(bf16)),
                    PackedOp::SubHalf { bf16 } => format!("sub{r}.f32x2.{}", half(bf16)),
                    PackedOp::FmaHalf { bf16 } => format!("fma{r}.f32x2.{}", half(bf16)),
                }
            }
        }
    }

    /// The operands' types.
    pub fn operand_types(self) -> &'static [DType] {
        match self {
            CvtFn::F32x2ToBf16x2
            | CvtFn::F32x2ToF8x2 { .. }
            | CvtFn::F32x2ToUe8m0x2 { .. }
            | CvtFn::F32x2ToUe5m3x2
            | CvtFn::F32x2ToMx { .. } => &[DType::F32, DType::F32],
            CvtFn::F32ToTf32 => &[DType::F32],
            CvtFn::S8ToBf16 => &[DType::I8],
            CvtFn::S2f6x2ToBf16x2 { .. }
            | CvtFn::F8x2ToF16x2 { .. }
            | CvtFn::Ue8m0x2ToBf16x2
            | CvtFn::MxToF16x2 { .. } => &[DType::I16],
            CvtFn::F16x2ToF8x2 { .. } | CvtFn::Bf16x2ToUe8m0x2 { .. } => &[DType::I32],
            CvtFn::Packed { op, .. } => match op {
                PackedOp::Add | PackedOp::Sub | PackedOp::Mul => &[DType::U64, DType::U64],
                PackedOp::Fma => &[DType::U64, DType::U64, DType::U64],
                PackedOp::AddToHalf { .. } | PackedOp::SubToHalf { .. } => &[DType::U64, DType::U64],
                PackedOp::AddHalf { .. } | PackedOp::SubHalf { .. } => &[DType::I32, DType::U64],
                PackedOp::FmaHalf { .. } => &[DType::I32, DType::U64, DType::U64],
            },
        }
    }

    /// The result's type.
    pub fn result_type(self) -> DType {
        match self {
            CvtFn::S8ToBf16 => DType::BF16,
            CvtFn::F32x2ToF8x2 { .. }
            | CvtFn::F16x2ToF8x2 { .. }
            | CvtFn::F32x2ToUe8m0x2 { .. }
            | CvtFn::F32x2ToUe5m3x2
            | CvtFn::Bf16x2ToUe8m0x2 { .. }
            | CvtFn::F32x2ToMx { .. } => DType::I16,
            CvtFn::F32x2ToBf16x2
            | CvtFn::F32ToTf32
            | CvtFn::S2f6x2ToBf16x2 { .. }
            | CvtFn::F8x2ToF16x2 { .. }
            | CvtFn::Ue8m0x2ToBf16x2
            | CvtFn::MxToF16x2 { .. } => DType::I32,
            CvtFn::Packed { op, .. } => match op {
                PackedOp::AddToHalf { .. } | PackedOp::SubToHalf { .. } => DType::I32,
                _ => DType::U64,
            },
        }
    }
}
