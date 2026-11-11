//! MLIR emission.
//!
//! Kernels lower to upstream MLIR dialects only (`gpu`, `arith`, `scf`,
//! `memref`, `vector`, `nvgpu`, `nvvm`, plus a few LLVM intrinsics and inline
//! PTX for math functions), so the output can be verified and compiled by a
//! stock `mlir-opt`:
//!
//! ```text
//! mlir-opt kernel.mlir --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-format=isa"
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::ir::{
    AtomicKind, BarrierKind, BinaryOp, BufferId, CmpPred, DType, FenceScope, FloatFn, FloatMode, Inst, KernelIr,
    LaneMask, Literal, LoopOptions, MbarrierWaitKind, MemScopeKind, MemSpace, MmaKind, Op, Param, Region, ShuffleKind,
    SpecialReg, UnaryOp, ValueId, VoteKind, float_inst_name,
};
use crate::validate::MAX_STATIC_SHARED_BYTES;

/// The length given to buffers sized at launch. Bare-pointer kernel arguments
/// need a static memref type; nothing depends on the value.
const RUNTIME_LEN: usize = i32::MAX as usize;

/// Options for [`emit_mlir`].
#[derive(Clone, Debug, Default)]
pub struct MlirOptions {
    /// Attach `loc("file":line:col)` source locations to every operation.
    pub locations: bool,
}

struct Emitter<'a> {
    ir: &'a KernelIr,
    opts: &'a MlirOptions,
    out: String,
    tmp: usize,
    /// Values emitted as another value (`make_warp_uniform`).
    aliases: BTreeMap<ValueId, ValueId>,
    /// Buffers read as more than one element type (or placed in dynamic
    /// shared memory) are bytes with a typed view per element type.
    bytes: Vec<bool>,
    views: BTreeMap<(u32, &'static str), String>,
    dynamic_smem: bool,
}

fn float_literal(bits: u64, ty: DType) -> String {
    let (value, hex) = match ty {
        DType::F16 => (crate::value::f16_bits_to_f32(bits as u16) as f64, format!("0x{bits:04X}")),
        DType::BF16 => (crate::value::bf16_bits_to_f32(bits as u16) as f64, format!("0x{bits:04X}")),
        DType::F64 => (f64::from_bits(bits), format!("0x{bits:016X}")),
        _ => (f32::from_bits(bits as u32) as f64, format!("0x{bits:08X}")),
    };
    let text = if ty == DType::F64 { format!("{value:?}") } else { format!("{:?}", value as f32) };
    if value.is_finite() && text.contains('.') && !text.contains('e') { text } else { hex }
}

fn literal(l: Literal) -> String {
    let ty = l.dtype();
    match l {
        Literal::Bool(b) => b.to_string(),
        Literal::I8(v) => format!("{v} : i8"),
        Literal::I16(v) => format!("{v} : i16"),
        Literal::I32(v) => format!("{v} : i32"),
        Literal::I64(v) => format!("{v} : i64"),
        Literal::U8(v) => format!("{v} : i8"),
        Literal::U16(v) => format!("{v} : i16"),
        Literal::U32(v) => format!("{v} : i32"),
        Literal::U64(v) => format!("{v} : i64"),
        _ => format!("{} : {}", float_literal(l.bits(), ty), ty.mlir()),
    }
}

fn address_space(space: MemSpace) -> &'static str {
    match space {
        MemSpace::Shared => ", #gpu.address_space<workgroup>",
        _ => "",
    }
}

fn escape(text: &str) -> String {
    let mut out = String::new();
    for b in text.bytes() {
        match b {
            b'"' | b'\\' | 0..=31 | 127.. => {
                let _ = write!(out, "\\{b:02X}");
            }
            _ => out.push(b as char),
        }
    }
    out
}

/// The inline-assembly register class of a type.
fn constraint(ty: DType) -> &'static str {
    match ty {
        DType::Bool => "b",
        DType::I8 | DType::U8 => "c",
        DType::I16 | DType::U16 | DType::F16 | DType::BF16 => "h",
        DType::I32 | DType::U32 => "r",
        DType::I64 | DType::U64 => "l",
        DType::F32 => "f",
        DType::F64 => "d",
    }
}

/// Rows and columns of an `nvgpu.mma.sync` operand vector: one row per
/// 32-bit register (64-bit for f64).
fn mma_shape(kind: MmaKind, operand: char) -> (usize, usize) {
    use MmaKind::*;
    match (kind, operand) {
        (M16N8K16F32F16 | M16N8K16F16F16 | M16N8K16F32BF16, 'a') => (4, 2),
        (M16N8K16F32F16 | M16N8K16F16F16 | M16N8K16F32BF16, 'b') => (2, 2),
        (M16N8K8F32F16 | M16N8K8F16F16 | M16N8K8F32BF16, 'a') => (2, 2),
        (M16N8K8F32F16 | M16N8K8F16F16 | M16N8K8F32BF16, 'b') => (1, 2),
        (M16N8K4Tf32, 'a') => (2, 1),
        (M16N8K8Tf32, 'a') => (4, 1),
        (M16N8K8Tf32, 'b') => (2, 1),
        (M16N8K4Tf32 | M8N8K4F64, 'b') | (M8N8K4F64, 'a') => (1, 1),
        (M8N8K4F64, _) => (1, 2),
        _ => (2, 2),
    }
}

impl Emitter<'_> {
    fn v(&self, v: ValueId) -> String {
        format!("%v{}", self.aliases.get(&v).unwrap_or(&v).0)
    }

    fn ty(&self, v: ValueId) -> &'static str {
        self.ir.dtype(v).mlir()
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.tmp += 1;
        format!("%{prefix}{}", self.tmp)
    }

    fn line(&mut self, indent: usize, text: &str, inst: Option<&Inst>) {
        let _ = write!(self.out, "{:indent$}{text}", "", indent = indent);
        if let (true, Some(i)) = (self.opts.locations, inst) {
            let _ = write!(self.out, " loc({:?}:{}:{})", i.loc.file(), i.loc.line(), i.loc.column());
        }
        self.out.push('\n');
    }

    /// The type of the buffer as declared: its own element type, or bytes.
    fn memref_type(&self, b: BufferId) -> String {
        let d = self.ir.buffer(b);
        let suffix = address_space(d.space);
        if self.bytes[b.0 as usize] {
            format!("memref<{}xi8{suffix}>", d.len.map_or(RUNTIME_LEN, |n| n * d.elem.bytes()))
        } else {
            format!("memref<{}x{}{suffix}>", d.len.unwrap_or(RUNTIME_LEN), d.elem.mlir())
        }
    }

    fn view_type(&self, b: BufferId, ty: DType) -> String {
        let d = self.ir.buffer(b);
        let count = d.len.map_or(RUNTIME_LEN, |n| n * d.elem.bytes() / ty.bytes());
        format!("memref<{count}x{}{}>", ty.mlir(), address_space(d.space))
    }

    fn buffer_name(&self, b: BufferId) -> String {
        let d = self.ir.buffer(b);
        match d.space {
            MemSpace::Global => format!("%arg_{}", d.name),
            _ => format!("%{}", d.name),
        }
    }

    /// The value and memref type to access `b` as elements of `ty`.
    fn access(&self, b: BufferId, ty: DType) -> (String, String) {
        if self.bytes[b.0 as usize] {
            (self.views[&(b.0, ty.mlir())].clone(), self.view_type(b, ty))
        } else {
            (self.buffer_name(b), self.memref_type(b))
        }
    }

    fn index(&mut self, indent: usize, v: ValueId) -> String {
        let name = self.fresh("i");
        let text = format!("{name} = arith.index_cast {} : i32 to index", self.v(v));
        self.line(indent, &text, None);
        name
    }

    fn index_const(&mut self, indent: usize, c: usize) -> String {
        let name = self.fresh("c");
        self.line(indent, &format!("{name} = arith.constant {c} : index"), None);
        name
    }

    fn region(&mut self, r: &Region, indent: usize) {
        for inst in &r.insts {
            self.inst(inst, indent);
        }
    }

    fn yield_line(&mut self, r: &Region, indent: usize) {
        if r.yields.is_empty() {
            return;
        }
        let names: Vec<String> = r.yields.iter().map(|&y| self.v(y)).collect();
        let types: Vec<&str> = r.yields.iter().map(|&y| self.ty(y)).collect();
        self.line(indent, &format!("scf.yield {} : {}", names.join(", "), types.join(", ")), None);
    }

    fn results(&self, rs: &[ValueId]) -> String {
        rs.iter().map(|&r| self.v(r)).collect::<Vec<_>>().join(", ")
    }

    /// Build a `rows x cols` vector from scalars, row-major.
    fn vector2d(&mut self, indent: usize, values: &[ValueId], (rows, cols): (usize, usize), ty: &str) -> String {
        let vty = format!("vector<{rows}x{cols}x{ty}>");
        let mut acc = self.fresh("p");
        self.line(indent, &format!("{acc} = ub.poison : {vty}"), None);
        for r in 0..rows {
            let row = self.fresh("row");
            let elems = self.results(&values[r * cols..(r + 1) * cols]);
            self.line(indent, &format!("{row} = vector.from_elements {elems} : vector<{cols}x{ty}>"), None);
            let next = self.fresh("p");
            self.line(
                indent,
                &format!("{next} = vector.insert {row}, {acc} [{r}] : vector<{cols}x{ty}> into {vty}"),
                None,
            );
            acc = next;
        }
        acc
    }

    /// A float's sign bit cleared (`op` = `and`, with the magnitude mask) or
    /// flipped (`xor`, with the sign bit), as an integer instruction in
    /// inline PTX. As plain integer operations or as `fabs`/`fneg`, LLVM
    /// lowers them to `abs.f32`/`neg.f32` or to `and.b32`/`xor.b32`
    /// depending on the code around them, and the GPU's `abs.f32` and
    /// `neg.f32` return the canonical NaN; inline, they are always the bit
    /// operation, which keeps a NaN's payload (IEEE 754's abs and negate).
    fn sign_bits(&mut self, indent: usize, inst: &Inst, result: ValueId, x: ValueId, op: &str, mask: u64) {
        let ty = self.ir.dtype(x);
        let width = ty.bits();
        let (t, int) = (ty.mlir(), format!("i{width}"));
        let reg = match width {
            16 => "h",
            32 => "r",
            _ => "l",
        };
        let (bits, out) = (self.fresh("bits"), self.fresh("sign"));
        self.line(indent, &format!("{bits} = arith.bitcast {} : {t} to {int}", self.v(x)), None);
        let text = format!(
            "{out} = llvm.inline_asm \"{op}.b{width} $0, $1, {mask:#x};\", \"={reg},{reg}\" {bits} : ({int}) -> {int}"
        );
        self.line(indent, &text, Some(inst));
        self.line(indent, &format!("{} = arith.bitcast {out} : {int} to {t}", self.v(result)), None);
    }

    /// Run a PTX math instruction on `x`. `{f}` in `ptx` is the type suffix.
    /// Types narrower than f32 are computed in f32.
    fn float_asm(&mut self, indent: usize, inst: &Inst, result: ValueId, x: ValueId, ptx: &str) {
        let ty = self.ir.dtype(x);
        let double = ty == DType::F64;
        let (reg, suffix, wide) = if double { ("d", "f64", "f64") } else { ("f", "f32", "f32") };
        let ptx = ptx.replace("{f}", suffix);
        let narrow = !matches!(ty, DType::F32 | DType::F64);
        let input = if narrow { self.extend(indent, x) } else { self.v(x) };
        let out = if narrow { self.fresh("f") } else { self.v(result) };
        let text = format!("{out} = llvm.inline_asm \"{ptx} $0, $1;\", \"={reg},{reg}\" {input} : ({wide}) -> {wide}");
        self.line(indent, &text, Some(inst));
        if narrow {
            self.line(indent, &format!("{} = arith.truncf {out} : f32 to {}", self.v(result), ty.mlir()), None);
        }
    }

    /// Extend a narrow float to f32.
    fn extend(&mut self, indent: usize, x: ValueId) -> String {
        let ty = self.ir.dtype(x);
        let ext = self.fresh("f");
        self.line(indent, &format!("{ext} = arith.extf {} : {} to f32", self.v(x), ty.mlir()), None);
        ext
    }

    /// `(operand, type)` for arithmetic: BFloat16 has no arithmetic before
    /// SM90, so it is computed in f32.
    fn arith_operand(&mut self, indent: usize, x: ValueId) -> (String, &'static str) {
        if self.ir.dtype(x) == DType::BF16 { (self.extend(indent, x), "f32") } else { (self.v(x), self.ty(x)) }
    }

    /// Where an arithmetic result goes: straight to `result`, or to a
    /// temporary that [`Self::finish_arith`] narrows.
    fn arith_result(&mut self, result: ValueId) -> String {
        if self.ir.dtype(result) == DType::BF16 { self.fresh("f") } else { self.v(result) }
    }

    fn finish_arith(&mut self, indent: usize, out: &str, result: ValueId) {
        if self.ir.dtype(result) == DType::BF16 {
            self.line(indent, &format!("{} = arith.truncf {out} : f32 to bf16", self.v(result)), None);
        }
    }

    fn cast(&mut self, indent: usize, inst: &Inst, result: ValueId, x: ValueId) {
        let (from, to) = (self.ir.dtype(x), self.ir.dtype(result));
        let (mut value, mut source) = (self.v(x), from);
        // BFloat16 converts to and from everything through f32.
        if (from == DType::BF16 || from == DType::F16) && to == DType::BF16
            || from == DType::BF16 && !matches!(to, DType::F32)
        {
            value = self.extend(indent, x);
            source = DType::F32;
        } else if to == DType::BF16 && from != DType::F32 {
            let f = self.fresh("f");
            let op = cast_op(from, DType::F32);
            self.line(indent, &format!("{f} = arith.{op} {value} : {} to f32", from.mlir()), None);
            value = f;
            source = DType::F32;
        }
        let op = cast_op(source, to);
        let text = format!("{} = arith.{op} {value} : {} to {}", self.v(result), source.mlir(), to.mlir());
        self.line(indent, &text, Some(inst));
    }

    fn special(&mut self, indent: usize, inst: &Inst, result: ValueId, reg: SpecialReg) {
        let gpu = match reg {
            SpecialReg::ThreadIdx(d) => Some(format!("gpu.thread_id {}", d.name())),
            SpecialReg::BlockIdx(d) => Some(format!("gpu.block_id {}", d.name())),
            SpecialReg::BlockDim(d) => Some(format!("gpu.block_dim {}", d.name())),
            SpecialReg::GridDim(d) => Some(format!("gpu.grid_dim {}", d.name())),
            SpecialReg::LaneId => Some("gpu.lane_id".to_string()),
            _ => None,
        };
        if let Some(op) = gpu {
            let idx = self.fresh("s");
            self.line(indent, &format!("{idx} = {op}"), Some(inst));
            self.line(indent, &format!("{} = arith.index_cast {idx} : index to i32", self.v(result)), None);
            return;
        }
        let name = match reg {
            SpecialReg::WarpSize => "warpsize".to_string(),
            SpecialReg::WarpId => "warpid".to_string(),
            SpecialReg::NWarpId => "nwarpid".to_string(),
            SpecialReg::SmId => "smid".to_string(),
            SpecialReg::NSmId => "nsmid".to_string(),
            SpecialReg::GridId => "gridid".to_string(),
            SpecialReg::LaneMask(k) => format!(
                "lanemask.{}",
                match k {
                    LaneMask::Eq => "eq",
                    LaneMask::Le => "le",
                    LaneMask::Lt => "lt",
                    LaneMask::Ge => "ge",
                    LaneMask::Gt => "gt",
                }
            ),
            SpecialReg::Clock => "clock".to_string(),
            SpecialReg::Clock64 => "clock64".to_string(),
            SpecialReg::GlobalTimer => "globaltimer".to_string(),
            SpecialReg::ClusterIdx(d) => format!("clusterid.{}", d.name()),
            SpecialReg::ClusterDim(d) => format!("nclusterid.{}", d.name()),
            SpecialReg::BlockInClusterIdx(d) => format!("cluster.ctaid.{}", d.name()),
            SpecialReg::BlockInClusterDim(d) => format!("cluster.nctaid.{}", d.name()),
            _ => unreachable!("handled above"),
        };
        let text = format!("{} = nvvm.read.ptx.sreg.{name} : {}", self.v(result), reg.dtype().mlir());
        self.line(indent, &text, Some(inst));
    }

    /// The address of element `index` (counting `ty`s) of a buffer, as an i64.
    fn address(&mut self, indent: usize, b: BufferId, ty: DType, index: ValueId) -> String {
        let (name, memref) = self.access(b, ty);
        let base = self.fresh("base");
        self.line(
            indent,
            &format!("{base} = memref.extract_aligned_pointer_as_index {name} : {memref} -> index"),
            None,
        );
        let addr = self.fresh("addr");
        self.line(indent, &format!("{addr} = arith.index_cast {base} : index to i64"), None);
        let wide = self.fresh("ix");
        self.line(indent, &format!("{wide} = arith.extsi {} : i32 to i64", self.v(index)), None);
        let size = self.fresh("size");
        self.line(indent, &format!("{size} = arith.constant {} : i64", ty.bytes()), None);
        let off = self.fresh("off");
        self.line(indent, &format!("{off} = arith.muli {wide}, {size} : i64"), None);
        let sum = self.fresh("sum");
        self.line(indent, &format!("{sum} = arith.addi {addr}, {off} : i64"), None);
        sum
    }

    fn atomic(&mut self, indent: usize, inst: &Inst, op: &Op) {
        let Op::Atomic { result, kind, buffer, index, value, compare } = op else { unreachable!() };
        let ty = self.ir.dtype(*value);
        let space = if self.ir.buffer(*buffer).space == MemSpace::Shared { 3 } else { 1 };
        let sum = self.address(indent, *buffer, ty, *index);
        let ptr = self.fresh("ptr");
        let ptr_ty = format!("!llvm.ptr<{space}>");
        self.line(indent, &format!("{ptr} = llvm.inttoptr {sum} : i64 to {ptr_ty}"), None);
        let t = ty.mlir();
        if *kind == AtomicKind::CompareExchange {
            let pair = self.fresh("pair");
            let text = format!(
                "{pair} = llvm.cmpxchg {ptr}, {}, {} acq_rel monotonic : {ptr_ty}, {t}",
                self.v(compare.expect("validated")),
                self.v(*value)
            );
            self.line(indent, &text, Some(inst));
            self.line(
                indent,
                &format!("{} = llvm.extractvalue {pair}[0] : !llvm.struct<({t}, i1)>", self.v(*result)),
                None,
            );
            return;
        }
        let name = match (kind, ty.is_signed()) {
            (AtomicKind::Add, _) => "add",
            (AtomicKind::And, _) => "_and",
            (AtomicKind::Or, _) => "_or",
            (AtomicKind::Xor, _) => "_xor",
            (AtomicKind::Min, true) => "min",
            (AtomicKind::Min, false) => "umin",
            (AtomicKind::Max, true) => "max",
            (AtomicKind::Max, false) => "umax",
            (AtomicKind::Exchange, _) => "xchg",
            (AtomicKind::CompareExchange, _) => unreachable!("handled above"),
        };
        let text =
            format!("{} = llvm.atomicrmw {name} {ptr}, {} acq_rel : {ptr_ty}, {t}", self.v(*result), self.v(*value));
        self.line(indent, &text, Some(inst));
    }

    /// A bf16 `mma.sync`. `nvgpu.mma.sync` cannot lower bf16 operands, so
    /// this emits `nvvm.mma.sync`, which takes each pair of bf16 values packed
    /// into an i32 (low half first, as in a `.bf16x2` register).
    #[allow(clippy::too_many_arguments)]
    fn mma_bf16(
        &mut self,
        indent: usize,
        inst: &Inst,
        kind: MmaKind,
        d: &[ValueId],
        a: &[ValueId],
        b: &[ValueId],
        c: &[ValueId],
    ) {
        let pack = |e: &mut Self, values: &[ValueId]| -> Vec<String> {
            values
                .chunks(2)
                .map(|pair| {
                    let v = e.fresh("pair");
                    let text = format!("{v} = vector.from_elements {} : vector<2xbf16>", e.results(pair));
                    e.line(indent, &text, None);
                    let r = e.fresh("reg");
                    e.line(indent, &format!("{r} = llvm.bitcast {v} : vector<2xbf16> to i32"), None);
                    r
                })
                .collect()
        };
        let ra = pack(self, a);
        let rb = pack(self, b);
        let rc: Vec<String> = c.iter().map(|&v| self.v(v)).collect();
        let (m, n, k) = kind.shape_mnk();
        let r = self.fresh("mma");
        let ret = format!("!llvm.struct<({})>", vec!["f32"; d.len()].join(", "));
        let text = format!(
            "{r} = nvvm.mma.sync A[{}] B[{}] C[{}] {{layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, \
             multiplicandAPtxType = #nvvm.mma_type<bf16>, multiplicandBPtxType = #nvvm.mma_type<bf16>, \
             shape = #nvvm.shape<m = {m}, n = {n}, k = {k}>}} : (i32, i32, f32) -> {ret}",
            ra.join(", "),
            rb.join(", "),
            rc.join(", ")
        );
        self.line(indent, &text, Some(inst));
        for (idx, out) in d.iter().enumerate() {
            self.line(indent, &format!("{} = llvm.extractvalue {r}[{idx}] : {ret}", self.v(*out)), None);
        }
    }

    fn inline_ptx(&mut self, indent: usize, inst: &Inst, op: &Op) {
        let Op::InlinePtx { template, inputs, outputs, side_effects, .. } = op else { unreachable!() };
        let constraints: Vec<String> = outputs
            .iter()
            .map(|&o| format!("={}", constraint(self.ir.dtype(o))))
            .chain(inputs.iter().map(|&i| constraint(self.ir.dtype(i)).to_string()))
            .collect();
        let effects = if *side_effects { "has_side_effects " } else { "" };
        let args = self.results(inputs);
        let arg_types: Vec<&str> = inputs.iter().map(|&i| self.ty(i)).collect();
        let head = format!("llvm.inline_asm {effects}\"{}\", \"{}\"", escape(template), constraints.join(","));
        let ret = match outputs.len() {
            0 => "()".to_string(),
            1 => self.ty(outputs[0]).to_string(),
            _ => format!("!llvm.struct<({})>", outputs.iter().map(|&o| self.ty(o)).collect::<Vec<_>>().join(", ")),
        };
        let lhs = match outputs.len() {
            0 => String::new(),
            1 => format!("{} = ", self.v(outputs[0])),
            _ => format!("{} = ", self.fresh("asm")),
        };
        let text = format!("{lhs}{head} {args} : ({}) -> {ret}", arg_types.join(", "));
        self.line(indent, &text, Some(inst));
        if outputs.len() > 1 {
            let tuple = lhs.trim_end_matches(" = ").to_string();
            for (i, &o) in outputs.iter().enumerate() {
                self.line(indent, &format!("{} = llvm.extractvalue {tuple}[{i}] : {ret}", self.v(o)), None);
            }
        }
    }

    fn inst(&mut self, inst: &Inst, indent: usize) {
        let i = Some(inst);
        match &inst.op {
            Op::Const { result, value } => {
                let text = format!("{} = arith.constant {}", self.v(*result), literal(*value));
                self.line(indent, &text, i);
            }
            Op::Unary { result, op, x } => self.unary(indent, inst, *result, *op, *x),
            Op::Binary { result, op, lhs, rhs } => self.binary(indent, inst, *result, *op, *lhs, *rhs),
            Op::Fma { result, a, b, c } => {
                let (av, t) = self.arith_operand(indent, *a);
                let (bv, _) = self.arith_operand(indent, *b);
                let (cv, _) = self.arith_operand(indent, *c);
                let out = self.arith_result(*result);
                let text = format!("{out} = llvm.intr.fma({av}, {bv}, {cv}) : ({t}, {t}, {t}) -> {t}");
                self.line(indent, &text, i);
                self.finish_arith(indent, &out, *result);
            }
            // `rcp.approx.ftz.f32` is an NVVM op, as in CuTe DSL.
            Op::FloatInst { result, func: FloatFn::Rcp, args, mode: FloatMode::Approx, ftz: true, nan: false }
                if self.ir.dtype(args[0]) == DType::F32 =>
            {
                let text = format!("{} = nvvm.rcp.approx.ftz.f {} : f32", self.v(*result), self.v(args[0]));
                self.line(indent, &text, i);
            }
            // The NVVM dialect has no op for the other modifier combinations,
            // so they are inline PTX; without side effects, LLVM may still
            // CSE them.
            Op::FloatInst { result, func, args, mode, ftz, nan } => {
                let ty = self.ir.dtype(args[0]);
                let name = float_inst_name(*func, *mode, *ftz, *nan, ty);
                let operands: Vec<String> = (1..=args.len()).map(|k| format!("${k}")).collect();
                let reg = constraint(ty);
                let constraints: Vec<String> =
                    std::iter::once(format!("={reg}")).chain(args.iter().map(|_| reg.to_string())).collect();
                let types = vec![ty.mlir(); args.len()].join(", ");
                let text = format!(
                    "{} = llvm.inline_asm \"{name} $0, {};\", \"{}\" {} : ({types}) -> {}",
                    self.v(*result),
                    operands.join(", "),
                    constraints.join(","),
                    self.results(args),
                    ty.mlir()
                );
                self.line(indent, &text, i);
            }
            Op::Cmp { result, pred, lhs, rhs } => {
                let ty = self.ir.dtype(*lhs);
                let (l, t) = self.arith_operand(indent, *lhs);
                let (r, _) = self.arith_operand(indent, *rhs);
                let text = if ty.is_float() {
                    let p = match pred {
                        CmpPred::Eq => "oeq",
                        CmpPred::Ne => "une",
                        CmpPred::Lt => "olt",
                        CmpPred::Le => "ole",
                        CmpPred::Gt => "ogt",
                        CmpPred::Ge => "oge",
                    };
                    format!("{} = arith.cmpf {p}, {l}, {r} : {t}", self.v(*result))
                } else {
                    let signed = ty.is_signed() || ty == DType::Bool;
                    let p = match (pred, signed) {
                        (CmpPred::Eq, _) => "eq",
                        (CmpPred::Ne, _) => "ne",
                        (CmpPred::Lt, true) => "slt",
                        (CmpPred::Le, true) => "sle",
                        (CmpPred::Gt, true) => "sgt",
                        (CmpPred::Ge, true) => "sge",
                        (CmpPred::Lt, false) => "ult",
                        (CmpPred::Le, false) => "ule",
                        (CmpPred::Gt, false) => "ugt",
                        (CmpPred::Ge, false) => "uge",
                    };
                    format!("{} = arith.cmpi {p}, {l}, {r} : {t}", self.v(*result))
                };
                self.line(indent, &text, i);
            }
            Op::Select { result, cond, on_true, on_false } => {
                let text = format!(
                    "{} = arith.select {}, {}, {} : {}",
                    self.v(*result),
                    self.v(*cond),
                    self.v(*on_true),
                    self.v(*on_false),
                    self.ty(*result)
                );
                self.line(indent, &text, i);
            }
            Op::Cast { result, x } => self.cast(indent, inst, *result, *x),
            Op::Bitcast { result, x } => {
                let text = format!(
                    "{} = arith.bitcast {} : {} to {}",
                    self.v(*result),
                    self.v(*x),
                    self.ty(*x),
                    self.ty(*result)
                );
                self.line(indent, &text, i);
            }
            Op::Special { result, reg } => self.special(indent, inst, *result, *reg),
            Op::Load { results, buffer, index } => {
                let elem = self.ir.dtype(results[0]);
                let ix = self.index(indent, *index);
                let (b, mt) = self.access(*buffer, elem);
                if results.len() == 1 {
                    let text = format!("{} = memref.load {b}[{ix}] : {mt}", self.v(results[0]));
                    self.line(indent, &text, i);
                } else {
                    let n = results.len();
                    let vec = self.fresh("vec");
                    let text = format!(
                        "{vec} = vector.load {b}[{ix}] {{alignment = {} : i64}} : {mt}, vector<{n}x{}>",
                        n * elem.bytes(),
                        elem.mlir()
                    );
                    self.line(indent, &text, i);
                    for (k, r) in results.iter().enumerate() {
                        let text = format!(
                            "{} = vector.extract {vec}[{k}] : {} from vector<{n}x{}>",
                            self.v(*r),
                            elem.mlir(),
                            elem.mlir()
                        );
                        self.line(indent, &text, None);
                    }
                }
            }
            Op::Store { values, buffer, index } => {
                let elem = self.ir.dtype(values[0]);
                let ix = self.index(indent, *index);
                let (b, mt) = self.access(*buffer, elem);
                if values.len() == 1 {
                    let text = format!("memref.store {}, {b}[{ix}] : {mt}", self.v(values[0]));
                    self.line(indent, &text, i);
                } else {
                    let n = values.len();
                    let vec = self.fresh("vec");
                    let text =
                        format!("{vec} = vector.from_elements {} : vector<{n}x{}>", self.results(values), elem.mlir());
                    self.line(indent, &text, None);
                    let text = format!(
                        "vector.store {vec}, {b}[{ix}] {{alignment = {} : i64}} : {mt}, vector<{n}x{}>",
                        n * elem.bytes(),
                        elem.mlir()
                    );
                    self.line(indent, &text, i);
                }
            }
            Op::CpAsync { src, src_index, dst, dst_index, elem, elems, bypass_l1, pred } => {
                let si = self.index(indent, *src_index);
                let di = self.index(indent, *dst_index);
                let src_elems = match pred {
                    Some(p) => {
                        let full = self.index_const(indent, *elems as usize);
                        let zero = self.index_const(indent, 0);
                        let n = self.fresh("n");
                        self.line(indent, &format!("{n} = arith.select {}, {full}, {zero} : index", self.v(*p)), None);
                        format!(", {n}")
                    }
                    None => String::new(),
                };
                let tok = self.fresh("tok");
                let attr = if *bypass_l1 { " {bypassL1}" } else { "" };
                let (sb, st) = self.access(*src, *elem);
                let (db, dt) = self.access(*dst, *elem);
                let text = format!(
                    "{tok} = nvgpu.device_async_copy {sb}[{si}], {db}[{di}], {elems}{src_elems}{attr} : {st} to {dt}"
                );
                self.line(indent, &text, i);
            }
            Op::CpAsyncCommitGroup => self.line(indent, "nvvm.cp.async.commit.group", i),
            Op::CpAsyncWaitGroup { pending } => self.line(indent, &format!("nvvm.cp.async.wait.group {pending}"), i),
            Op::LdMatrix { results, buffer, index, num, trans } => {
                let elem = self.ir.dtype(results[0]);
                let ix = self.index(indent, *index);
                let m = self.fresh("ldm");
                let (b, mt) = self.access(*buffer, elem);
                let t = elem.mlir();
                let text = format!(
                    "{m} = nvgpu.ldmatrix {b}[{ix}] {{numTiles = {num} : i32, transpose = {trans}}} : {mt} -> vector<{num}x2x{t}>"
                );
                self.line(indent, &text, i);
                for (k, r) in results.iter().enumerate() {
                    let text = format!(
                        "{} = vector.extract {m}[{}, {}] : {t} from vector<{num}x2x{t}>",
                        self.v(*r),
                        k / 2,
                        k % 2
                    );
                    self.line(indent, &text, None);
                }
            }
            Op::Mma { kind, d, a, b, c } if kind.dtypes().0 == DType::BF16 => {
                self.mma_bf16(indent, inst, *kind, d, a, b, c)
            }
            Op::Mma { kind, d, a, b, c } => {
                let (ta, tb, tc) = kind.dtypes();
                let (sa, sb, sc) = (mma_shape(*kind, 'a'), mma_shape(*kind, 'b'), mma_shape(*kind, 'c'));
                let va = self.vector2d(indent, a, sa, ta.mlir());
                let vb = self.vector2d(indent, b, sb, tb.mlir());
                let vc = self.vector2d(indent, c, sc, tc.mlir());
                let (m, n, k) = kind.shape_mnk();
                let r = self.fresh("mma");
                let vt = |s: (usize, usize), t: DType| format!("vector<{}x{}x{}>", s.0, s.1, t.mlir());
                let tf32 = if kind.is_tf32() { ", tf32Enabled" } else { "" };
                let text = format!(
                    "{r} = nvgpu.mma.sync({va}, {vb}, {vc}) {{mmaShape = [{m}, {n}, {k}]{tf32}}} : ({}, {}, {}) -> {}",
                    vt(sa, ta),
                    vt(sb, tb),
                    vt(sc, tc),
                    vt(sc, tc)
                );
                self.line(indent, &text, i);
                for (idx, out) in d.iter().enumerate() {
                    let text = format!(
                        "{} = vector.extract {r}[{}, {}] : {} from {}",
                        self.v(*out),
                        idx / sc.1,
                        idx % sc.1,
                        tc.mlir(),
                        vt(sc, tc)
                    );
                    self.line(indent, &text, None);
                }
            }
            Op::Barrier => self.line(indent, "gpu.barrier", i),
            Op::NamedBarrier { kind, id, threads } => {
                // The NVVM barrier intrinsics are convergent, so LLVM never
                // duplicates them into divergent paths (inline PTX would not
                // be). The unaligned arrive is `barrier.arrive`, CuTe DSL's
                // `aligned=False` (`barrier.cta.arrive`, whose `.cta` changes
                // nothing but the PTX version it needs).
                let (id, n) = (self.v(*id), threads.map(|n| self.v(n)));
                let text = match (kind, n) {
                    (BarrierKind::Sync, None) => format!("nvvm.barrier id = {id}"),
                    (BarrierKind::Sync, Some(n)) => format!("nvvm.barrier id = {id} number_of_threads = {n}"),
                    (BarrierKind::Arrive, Some(n)) => format!(
                        "llvm.call_intrinsic \"llvm.nvvm.barrier.cta.arrive.aligned.count\"({id}, {n}) : (i32, i32) -> ()"
                    ),
                    (BarrierKind::ArriveUnaligned, Some(n)) => {
                        format!(
                            "llvm.call_intrinsic \"llvm.nvvm.barrier.cta.arrive.count\"({id}, {n}) : (i32, i32) -> ()"
                        )
                    }
                    (_, None) => unreachable!("validated: arrives have a thread count"),
                };
                self.line(indent, &text, i);
            }
            Op::WarpBarrier { mask } => {
                let text = format!("nvvm.bar.warp.sync {} : i32", self.v(*mask));
                self.line(indent, &text, i);
            }
            Op::Shuffle { result, kind, value, offset, mask } => {
                let ty = self.ir.dtype(*value);
                let float = ty.is_float();
                let word = if float {
                    let w = self.fresh("w");
                    self.line(indent, &format!("{w} = arith.bitcast {} : f32 to i32", self.v(*value)), None);
                    w
                } else {
                    self.v(*value)
                };
                let clamp = self.fresh("clamp");
                // The third operand is `width - 1`, except for `up`, where it
                // is `32 - width`. The width is always the whole warp here.
                let clamp_value = if *kind == ShuffleKind::Up { 0 } else { 31 };
                self.line(indent, &format!("{clamp} = arith.constant {clamp_value} : i32"), None);
                let k = match kind {
                    ShuffleKind::Idx => "idx",
                    ShuffleKind::Up => "up",
                    ShuffleKind::Down => "down",
                    ShuffleKind::Bfly => "bfly",
                };
                let out = if float { self.fresh("w") } else { self.v(*result) };
                let text = format!(
                    "{out} = nvvm.shfl.sync {k} {}, {word}, {}, {clamp} : i32 -> i32",
                    self.v(*mask),
                    self.v(*offset)
                );
                self.line(indent, &text, i);
                if float {
                    self.line(indent, &format!("{} = arith.bitcast {out} : i32 to f32", self.v(*result)), None);
                }
            }
            Op::Vote { result, kind, pred, mask } => {
                let (k, ret) = match kind {
                    VoteKind::All => ("all", "i1"),
                    VoteKind::Any => ("any", "i1"),
                    VoteKind::Uni => ("uni", "i1"),
                    VoteKind::Ballot => ("ballot", "i32"),
                };
                let text =
                    format!("{} = nvvm.vote.sync {k} {}, {} -> {ret}", self.v(*result), self.v(*mask), self.v(*pred));
                self.line(indent, &text, i);
            }
            Op::MatchAny { result, value, mask } => {
                let text = format!(
                    "{} = nvvm.match.sync any {}, {} : {} -> i32",
                    self.v(*result),
                    self.v(*mask),
                    self.v(*value),
                    self.ty(*value)
                );
                self.line(indent, &text, i);
            }
            Op::ElectOne { result, .. } => {
                // `elect.sync` needs PTX 8.0, so pick the lowest active lane
                // by hand, which works on every target.
                let active = self.fresh("active");
                self.line(
                    indent,
                    &format!("{active} = llvm.inline_asm has_side_effects \"activemask.b32 $0;\", \"=r\" : () -> i32"),
                    i,
                );
                let zero = self.fresh("z");
                self.line(indent, &format!("{zero} = arith.constant 0 : i32"), None);
                let negated = self.fresh("neg");
                self.line(indent, &format!("{negated} = arith.subi {zero}, {active} : i32"), None);
                let lowest = self.fresh("low");
                self.line(indent, &format!("{lowest} = arith.andi {active}, {negated} : i32"), None);
                let mine = self.fresh("lane");
                self.line(indent, &format!("{mine} = nvvm.read.ptx.sreg.lanemask.eq : i32"), None);
                self.line(indent, &format!("{} = arith.cmpi eq, {lowest}, {mine} : i32", self.v(*result)), None);
            }
            Op::ActiveMask { result } => {
                let text = format!(
                    "{} = llvm.inline_asm has_side_effects \"activemask.b32 $0;\", \"=r\" : () -> i32",
                    self.v(*result)
                );
                self.line(indent, &text, i);
            }
            op @ Op::Atomic { .. } => self.atomic(indent, inst, op),
            // CuTe DSL's hint has no upstream LLVM equivalent: the result is
            // the value itself.
            Op::WarpUniform { result, value } => {
                let target = *self.aliases.get(value).unwrap_or(value);
                self.aliases.insert(*result, target);
            }
            Op::MbarrierInit { buffer, index, count } => {
                let p = self.address(indent, *buffer, DType::I64, *index);
                let text = format!(
                    "llvm.inline_asm has_side_effects \"mbarrier.init.shared.b64 [$0], $1;\", \"l,r\" {p}, {} : (i64, i32) -> ()",
                    self.v(*count)
                );
                self.line(indent, &text, i);
            }
            Op::MbarrierArrive { buffer, index, count, expect_tx, peer, relaxed, scope } => {
                let p = self.address(indent, *buffer, DType::I64, *index);
                let op = if expect_tx.is_some() { "mbarrier.arrive.expect_tx" } else { "mbarrier.arrive" };
                let explicit = *relaxed || *scope == MemScopeKind::Cluster;
                let sem = if explicit {
                    format!(".{}.{}", if *relaxed { "relaxed" } else { "release" }, scope_name(*scope))
                } else {
                    String::new()
                };
                // The operand after the address: the arrival count or the bytes.
                let extra = count.or(*expect_tx);
                let mut args = vec![p];
                let mut types = vec!["i64"];
                if let Some(v) = extra {
                    args.push(self.v(v));
                    types.push("i32");
                }
                let text = match peer {
                    // The peer CTA's mbarrier, through its shared::cluster address.
                    Some(rank) => {
                        args.push(self.v(*rank));
                        types.push("i32");
                        let rank_arg = if extra.is_some() { "$2" } else { "$1" };
                        let tail = if extra.is_some() { ", $1" } else { "" };
                        let ptx = format!(
                            "{{\n\t.reg .b64 remote;\n\tmapa.shared::cluster.u64 remote, $0, {rank_arg};\n\t\
                             {op}{sem}.shared::cluster.b64 _, [remote]{tail};\n\t}}"
                        );
                        let constraints = vec!["l"; 1].into_iter().chain(vec!["r"; args.len() - 1]).collect::<Vec<_>>();
                        format!(
                            "llvm.inline_asm has_side_effects \"{}\", \"{}\" {} : ({}) -> ()",
                            escape(&ptx),
                            constraints.join(","),
                            args.join(", "),
                            types.join(", ")
                        )
                    }
                    // The plain form keeps to PTX 7.0 (sm_80): `.shared`, and a
                    // state register rather than the `_` sink.
                    None if extra.is_none() && !explicit => format!(
                        "{} = llvm.inline_asm has_side_effects \"mbarrier.arrive.shared.b64 $0, [$1];\", \"=l,l\" {} : (i64) -> i64",
                        self.fresh("state"),
                        args[0]
                    ),
                    None => {
                        let tail = if extra.is_some() { ", $1" } else { "" };
                        let constraints = if extra.is_some() { "l,r" } else { "l" };
                        format!(
                            "llvm.inline_asm has_side_effects \"{op}{sem}.shared::cta.b64 _, [$0]{tail};\", \"{constraints}\" {} : ({}) -> ()",
                            args.join(", "),
                            types.join(", ")
                        )
                    }
                };
                self.line(indent, &text, i);
            }
            Op::MbarrierExpectTx { buffer, index, bytes, peer, scope } => {
                let p = self.address(indent, *buffer, DType::I64, *index);
                // PTX only has a relaxed expect-tx; `.sem` and `.scope` go together.
                let sem = format!(".relaxed.{}", scope_name(*scope));
                let text = match peer {
                    Some(rank) => {
                        let ptx = format!(
                            "{{\n\t.reg .b64 remote;\n\tmapa.shared::cluster.u64 remote, $0, $2;\n\t\
                             mbarrier.expect_tx{sem}.shared::cluster.b64 [remote], $1;\n\t}}"
                        );
                        format!(
                            "llvm.inline_asm has_side_effects \"{}\", \"l,r,r\" {p}, {}, {} : (i64, i32, i32) -> ()",
                            escape(&ptx),
                            self.v(*bytes),
                            self.v(*rank)
                        )
                    }
                    None => format!(
                        "llvm.inline_asm has_side_effects \"mbarrier.expect_tx{sem}.shared::cta.b64 [$0], $1;\", \"l,r\" {p}, {} : (i64, i32) -> ()",
                        self.v(*bytes)
                    ),
                };
                self.line(indent, &text, i);
            }
            Op::MbarrierWait { result, buffer, index, parity, kind } => {
                let p = self.address(indent, *buffer, DType::I64, *index);
                // test_wait.parity needs PTX 7.1 (sm_80); try_wait needs sm_90.
                let ptx = match kind {
                    MbarrierWaitKind::Try => "mbarrier.try_wait.parity.shared::cta.b64",
                    MbarrierWaitKind::Test => "mbarrier.test_wait.parity.shared.b64",
                };
                let text = format!(
                    "{} = llvm.inline_asm has_side_effects \"{ptx} $0, [$1], $2;\", \"=b,l,r\" {p}, {} : (i64, i32) -> i1",
                    self.v(*result),
                    self.v(*parity)
                );
                self.line(indent, &text, i);
            }
            Op::FenceMbarrierInit => self.line(
                indent,
                "llvm.inline_asm has_side_effects \"fence.mbarrier_init.release.cluster;\", \"\" : () -> ()",
                i,
            ),
            Op::CpAsyncBulk { src, src_index, dst, dst_index, elem, bytes, mbarrier, mbarrier_index } => {
                let d = self.address(indent, *dst, *elem, *dst_index);
                let s = self.address(indent, *src, *elem, *src_index);
                let m = self.address(indent, *mbarrier, DType::I64, *mbarrier_index);
                let text = format!(
                    "llvm.inline_asm has_side_effects \"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];\", \"l,l,r,l\" {d}, {s}, {}, {m} : (i64, i64, i32, i64) -> ()",
                    self.v(*bytes)
                );
                self.line(indent, &text, i);
            }
            Op::Fence { scope } => {
                let s = match scope {
                    FenceScope::Cta => "cta",
                    FenceScope::Cluster => "cluster",
                    FenceScope::Gpu => "gpu",
                    FenceScope::Sys => "sys",
                };
                self.line(
                    indent,
                    &format!("llvm.inline_asm has_side_effects \"fence.acq_rel.{s};\", \"\" : () -> ()"),
                    i,
                );
            }
            Op::Printf { format, args } => {
                let mut text = format!("gpu.printf \"{}\"", escape(format));
                if !args.is_empty() {
                    let types: Vec<&str> = args.iter().map(|&a| self.ty(a)).collect();
                    let _ = write!(text, ", {} : {}", self.results(args), types.join(", "));
                }
                self.line(indent, &text, i);
            }
            op @ Op::InlinePtx { .. } => self.inline_ptx(indent, inst, op),
            Op::If { cond, then_region, else_region, results } => {
                let head = if results.is_empty() {
                    format!("scf.if {} {{", self.v(*cond))
                } else {
                    let types: Vec<&str> = results.iter().map(|&r| self.ty(r)).collect();
                    format!("{} = scf.if {} -> ({}) {{", self.results(results), self.v(*cond), types.join(", "))
                };
                self.line(indent, &head, None);
                self.region(then_region, indent + 2);
                self.yield_line(then_region, indent + 2);
                if !else_region.insts.is_empty() || !results.is_empty() {
                    self.line(indent, "} else {", None);
                    self.region(else_region, indent + 2);
                    self.yield_line(else_region, indent + 2);
                }
                self.line(indent, "}", i);
            }
            Op::For { iv, lower, upper, step, init, args, body, results, opts } => {
                if opts.at_least_once {
                    // The promise that the loop runs, as an assumption LLVM
                    // uses to drop the zero-trip check in front of the loop
                    // (CuTe DSL's `at_least_once` attribute is only read by
                    // NVIDIA's compiler). `scf.for` compares signed.
                    let runs = self.fresh("runs");
                    let (lo, hi) = (self.v(*lower), self.v(*upper));
                    self.line(indent, &format!("{runs} = arith.cmpi slt, {lo}, {hi} : i32"), i);
                    self.line(indent, &format!("llvm.intr.assume {runs} : i1"), i);
                }
                let bounds =
                    format!("{} = {} to {} step {}", self.v(*iv), self.v(*lower), self.v(*upper), self.v(*step));
                let head = if results.is_empty() {
                    format!("scf.for {bounds} : i32 {{")
                } else {
                    let iters: Vec<String> =
                        args.iter().zip(init).map(|(&a, &x)| format!("{} = {}", self.v(a), self.v(x))).collect();
                    let types: Vec<&str> = results.iter().map(|&r| self.ty(r)).collect();
                    format!(
                        "{} = scf.for {bounds} iter_args({}) -> ({}) : i32 {{",
                        self.results(results),
                        iters.join(", "),
                        types.join(", ")
                    )
                };
                self.line(indent, &head, None);
                self.region(body, indent + 2);
                self.yield_line(body, indent + 2);
                match loop_annotation(opts) {
                    Some(annotation) => self.line(indent, &format!("}} {{loop_annotation = {annotation}}}"), i),
                    None => self.line(indent, "}", i),
                }
            }
            Op::While { init, cond_args, cond, body_args, body, results } => {
                let types: Vec<&str> = results.iter().map(|&r| self.ty(r)).collect();
                let list = types.join(", ");
                let binds: Vec<String> =
                    cond_args.iter().zip(init).map(|(&a, &x)| format!("{} = {}", self.v(a), self.v(x))).collect();
                let lhs = if results.is_empty() { String::new() } else { format!("{} = ", self.results(results)) };
                self.line(indent, &format!("{lhs}scf.while ({}) : ({list}) -> ({list}) {{", binds.join(", ")), None);
                self.region(cond, indent + 2);
                let forwarded = self.results(cond_args);
                let text = if cond_args.is_empty() {
                    format!("scf.condition({})", self.v(cond.yields[0]))
                } else {
                    format!("scf.condition({}) {forwarded} : {list}", self.v(cond.yields[0]))
                };
                self.line(indent + 2, &text, None);
                self.line(indent, "} do {", None);
                let block: Vec<String> =
                    body_args.iter().zip(&types).map(|(&a, t)| format!("{}: {t}", self.v(a))).collect();
                self.line(indent, &format!("^bb0({}):", block.join(", ")), None);
                self.region(body, indent + 2);
                if body.yields.is_empty() {
                    self.line(indent + 2, "scf.yield", None);
                } else {
                    self.yield_line(body, indent + 2);
                }
                self.line(indent, "}", i);
            }
        }
    }

    fn unary(&mut self, indent: usize, inst: &Inst, result: ValueId, op: UnaryOp, x: ValueId) {
        let i = Some(inst);
        let ty = self.ir.dtype(x);
        let (r, xs, t) = (self.v(result), self.v(x), ty.mlir());
        match op {
            UnaryOp::Neg if ty.is_float() => self.sign_bits(indent, inst, result, x, "xor", 1u64 << (ty.bits() - 1)),
            UnaryOp::Neg => {
                let z = self.fresh("z");
                self.line(indent, &format!("{z} = arith.constant 0 : {t}"), None);
                self.line(indent, &format!("{r} = arith.subi {z}, {xs} : {t}"), i);
            }
            UnaryOp::Not => {
                let ones = self.fresh("ones");
                let lit = if ty == DType::Bool { "true".to_string() } else { format!("-1 : {t}") };
                self.line(indent, &format!("{ones} = arith.constant {lit}"), None);
                self.line(indent, &format!("{r} = arith.xori {xs}, {ones} : {t}"), i);
            }
            UnaryOp::Abs => {
                if ty.is_float() {
                    self.sign_bits(indent, inst, result, x, "and", (1u64 << (ty.bits() - 1)) - 1);
                } else {
                    let neg = self.fresh("neg");
                    let z = self.fresh("z");
                    self.line(indent, &format!("{z} = arith.constant 0 : {t}"), None);
                    self.line(indent, &format!("{neg} = arith.subi {z}, {xs} : {t}"), None);
                    self.line(indent, &format!("{r} = arith.maxsi {xs}, {neg} : {t}"), i);
                }
            }
            UnaryOp::Sqrt => self.float_asm(indent, inst, result, x, "sqrt.rn.{f}"),
            UnaryOp::Rsqrt => self.float_asm(indent, inst, result, x, "rsqrt.approx.{f}"),
            UnaryOp::Exp2 => self.float_asm(indent, inst, result, x, "ex2.approx.{f}"),
            UnaryOp::Log2 => self.float_asm(indent, inst, result, x, "lg2.approx.{f}"),
            UnaryOp::Sin => self.float_asm(indent, inst, result, x, "sin.approx.{f}"),
            UnaryOp::Cos => self.float_asm(indent, inst, result, x, "cos.approx.{f}"),
            UnaryOp::Rcp if ty == DType::F64 => self.float_asm(indent, inst, result, x, "rcp.rn.{f}"),
            UnaryOp::Rcp => self.float_asm(indent, inst, result, x, "rcp.approx.{f}"),
            UnaryOp::Floor => self.float_asm(indent, inst, result, x, "cvt.rmi.{f}.{f}"),
            UnaryOp::Ceil => self.float_asm(indent, inst, result, x, "cvt.rpi.{f}.{f}"),
            UnaryOp::Trunc => self.float_asm(indent, inst, result, x, "cvt.rzi.{f}.{f}"),
            UnaryOp::Rint => self.float_asm(indent, inst, result, x, "cvt.rni.{f}.{f}"),
            UnaryOp::Popc => self.line(indent, &format!("{r} = llvm.intr.ctpop({xs}) : ({t}) -> {t}"), i),
            UnaryOp::Brev => self.line(indent, &format!("{r} = llvm.intr.bitreverse({xs}) : ({t}) -> {t}"), i),
            UnaryOp::Clz | UnaryOp::Ctz => {
                let name = if op == UnaryOp::Clz { "ctlz" } else { "cttz" };
                let text = format!("{r} = \"llvm.intr.{name}\"({xs}) <{{is_zero_poison = false}}> : ({t}) -> {t}");
                self.line(indent, &text, i);
            }
        }
    }

    fn binary(&mut self, indent: usize, inst: &Inst, result: ValueId, op: BinaryOp, lhs: ValueId, rhs: ValueId) {
        let i = Some(inst);
        let ty = self.ir.dtype(lhs);
        if ty.is_float() {
            let name = match op {
                BinaryOp::Add => "addf",
                BinaryOp::Sub => "subf",
                BinaryOp::Mul => "mulf",
                BinaryOp::Div => "divf",
                BinaryOp::Min => "minnumf",
                BinaryOp::Max => "maxnumf",
                _ => unreachable!("validated"),
            };
            let (l, t) = self.arith_operand(indent, lhs);
            let (r, _) = self.arith_operand(indent, rhs);
            let out = self.arith_result(result);
            self.line(indent, &format!("{out} = arith.{name} {l}, {r} : {t}"), i);
            self.finish_arith(indent, &out, result);
            return;
        }
        let signed = ty.is_signed();
        if op == BinaryOp::MulHi {
            let low = self.fresh("lo");
            let name = if signed { "mulsi_extended" } else { "mului_extended" };
            let text =
                format!("{low}, {} = arith.{name} {}, {} : {}", self.v(result), self.v(lhs), self.v(rhs), ty.mlir());
            self.line(indent, &text, i);
            return;
        }
        let name = match (op, signed) {
            (BinaryOp::Add, _) => "addi",
            (BinaryOp::Sub, _) => "subi",
            (BinaryOp::Mul, _) => "muli",
            (BinaryOp::Div, true) => "divsi",
            (BinaryOp::Div, false) | (BinaryOp::DivU, _) => "divui",
            (BinaryOp::Rem, true) => "remsi",
            (BinaryOp::Rem, false) | (BinaryOp::RemU, _) => "remui",
            (BinaryOp::Min, true) => "minsi",
            (BinaryOp::Min, false) => "minui",
            (BinaryOp::Max, true) => "maxsi",
            (BinaryOp::Max, false) => "maxui",
            (BinaryOp::And, _) => "andi",
            (BinaryOp::Or, _) => "ori",
            (BinaryOp::Xor, _) => "xori",
            (BinaryOp::Shl, _) => "shli",
            (BinaryOp::ShrU, _) => "shrui",
            (BinaryOp::ShrS, _) => "shrsi",
            (BinaryOp::MulHi, _) => unreachable!("handled above"),
        };
        let text = format!("{} = arith.{name} {}, {} : {}", self.v(result), self.v(lhs), self.v(rhs), ty.mlir());
        self.line(indent, &text, i);
    }
}

/// The LLVM loop metadata for a loop's hints, as CuTe DSL attaches it to its
/// `scf.for` (`loop_annotation`), which the lowering moves to the loop's
/// back edge: `unroll=K` is `count = K` (with `disable = true` for 1),
/// `unroll_full` is `full = true`, and `vectorize` is `vectorize = <disable =
/// false>` (CuTe DSL's own `cutlass.vectorize` attribute is only understood
/// by NVIDIA's compiler). `at_least_once` is not loop metadata: it becomes
/// an `llvm.intr.assume` in front of the loop. Parameters are in the
/// attribute's declaration order.
fn loop_annotation(opts: &LoopOptions) -> Option<String> {
    if !opts.has_annotation() {
        return None;
    }
    let mut params = Vec::new();
    if opts.vectorize {
        params.push("vectorize = <disable = false>".to_string());
    }
    if opts.unroll_full {
        params.push("unroll = <full = true>".to_string());
    } else if let Some(count) = opts.unroll {
        let disable = if count == 1 { "disable = true, " } else { "" };
        params.push(format!("unroll = <{disable}count = {count} : i32>"));
    }
    Some(format!("#llvm.loop_annotation<{}>", params.join(", ")))
}

fn scope_name(scope: MemScopeKind) -> &'static str {
    match scope {
        MemScopeKind::Cta => "cta",
        MemScopeKind::Cluster => "cluster",
    }
}

fn cast_op(from: DType, to: DType) -> &'static str {
    if from.is_float() && to.is_float() {
        return if to.bits() > from.bits() { "extf" } else { "truncf" };
    }
    if to.is_float() {
        return if from.is_signed() { "sitofp" } else { "uitofp" };
    }
    if from.is_float() {
        return if to.is_signed() { "fptosi" } else { "fptoui" };
    }
    match to.bits().cmp(&from.bits()) {
        std::cmp::Ordering::Greater => {
            if from.is_signed() {
                "extsi"
            } else {
                "extui"
            }
        }
        std::cmp::Ordering::Less => "trunci",
        std::cmp::Ordering::Equal => "bitcast",
    }
}

/// The element types each buffer is accessed as.
fn access_types(ir: &KernelIr) -> Vec<BTreeSet<&'static str>> {
    fn walk(r: &Region, ir: &KernelIr, out: &mut [BTreeSet<&'static str>], types: &mut BTreeMap<&'static str, DType>) {
        for inst in &r.insts {
            let mut note = |b: BufferId, t: DType| {
                out[b.0 as usize].insert(t.mlir());
                types.insert(t.mlir(), t);
            };
            match &inst.op {
                Op::Load { results, buffer, .. } | Op::LdMatrix { results, buffer, .. } => {
                    note(*buffer, ir.dtype(results[0]))
                }
                Op::Store { values, buffer, .. } => note(*buffer, ir.dtype(values[0])),
                Op::Atomic { value, buffer, .. } => note(*buffer, ir.dtype(*value)),
                Op::MbarrierInit { buffer, .. }
                | Op::MbarrierArrive { buffer, .. }
                | Op::MbarrierExpectTx { buffer, .. }
                | Op::MbarrierWait { buffer, .. } => note(*buffer, DType::I64),
                Op::CpAsyncBulk { src, dst, elem, mbarrier, .. } => {
                    note(*src, *elem);
                    note(*dst, *elem);
                    note(*mbarrier, DType::I64);
                }
                Op::CpAsync { src, dst, elem, .. } => {
                    note(*src, *elem);
                    note(*dst, *elem);
                }
                _ => {}
            }
            inst.op.regions().into_iter().for_each(|r| walk(r, ir, out, types));
        }
    }
    let mut out = vec![BTreeSet::new(); ir.buffers.len()];
    let mut types = BTreeMap::new();
    walk(&ir.body, ir, &mut out, &mut types);
    out
}

fn dtype_of(name: &str) -> DType {
    [DType::I8, DType::I16, DType::I32, DType::I64, DType::F16, DType::BF16, DType::F32, DType::F64]
        .into_iter()
        .find(|t| t.mlir() == name)
        .unwrap_or(DType::I8)
}

/// Emit `ir` as an MLIR module containing one `gpu.module` with the kernel.
pub fn emit_mlir(ir: &KernelIr, opts: &MlirOptions) -> String {
    let (smem_bytes, smem_starts) = ir.shared_memory_layout();
    let dynamic_smem = smem_bytes > MAX_STATIC_SHARED_BYTES;
    let used = access_types(ir);
    let bytes: Vec<bool> = ir
        .buffers
        .iter()
        .zip(&used)
        .map(|(b, u)| (dynamic_smem && b.space == MemSpace::Shared) || u.iter().any(|&t| t != b.elem.mlir()))
        .collect();
    let mut e = Emitter {
        ir,
        opts,
        out: String::new(),
        tmp: 0,
        aliases: BTreeMap::new(),
        bytes,
        views: BTreeMap::new(),
        dynamic_smem,
    };
    e.line(0, "module attributes {gpu.container_module} {", None);
    e.line(2, "gpu.module @kernels {", None);
    for (k, b) in ir.buffers.iter().enumerate() {
        if b.space == MemSpace::Shared && !e.dynamic_smem {
            let text = format!(
                "memref.global \"private\" @{} : {} {{alignment = {} : i64}}",
                b.name,
                e.memref_type(BufferId(k as u32)),
                b.align
            );
            e.line(4, &text, None);
        }
    }
    let params: Vec<String> = ir
        .args
        .iter()
        .map(|p| match p {
            Param::Buffer(b) => {
                format!("{}: {} {{llvm.align = {} : i64}}", e.buffer_name(*b), e.memref_type(*b), ir.buffer(*b).align)
            }
            Param::Scalar { value, .. } => format!("{}: {}", e.v(*value), e.ty(*value)),
        })
        .collect();
    let attrs = match ir.block_dim {
        Some([x, y, z]) => format!(" attributes {{known_block_size = array<i32: {x}, {y}, {z}>}}"),
        None => String::new(),
    };
    e.line(4, &format!("gpu.func @{}({}) kernel{attrs} {{", ir.name, params.join(", ")), None);
    let dynamic_type = "memref<?xi8, #gpu.address_space<workgroup>>";
    if e.dynamic_smem {
        e.line(6, &format!("%dyn_smem = gpu.dynamic_shared_memory : {dynamic_type}"), None);
    }
    for (k, b) in ir.buffers.iter().enumerate() {
        let id = BufferId(k as u32);
        match b.space {
            MemSpace::Shared if !e.dynamic_smem => {
                let text = format!("{} = memref.get_global @{} : {}", e.buffer_name(id), b.name, e.memref_type(id));
                e.line(6, &text, None);
            }
            MemSpace::Local => {
                let text = format!(
                    "{} = memref.alloca() {{alignment = {} : i64}} : {}",
                    e.buffer_name(id),
                    b.align,
                    e.memref_type(id)
                );
                e.line(6, &text, None);
            }
            _ => {}
        }
        if !e.bytes[k] {
            continue;
        }
        let start = if b.space == MemSpace::Shared && e.dynamic_smem { smem_starts[k].unwrap_or(0) } else { 0 };
        for &ty in &used[k] {
            let dtype = dtype_of(ty);
            let offset = e.index_const(6, start);
            let view = format!("{}_{ty}", e.buffer_name(id));
            let (source, source_type) = if b.space == MemSpace::Shared && e.dynamic_smem {
                ("%dyn_smem".to_string(), dynamic_type.to_string())
            } else {
                (e.buffer_name(id), e.memref_type(id))
            };
            let text =
                format!("{view} = memref.view {source}[{offset}][] : {source_type} to {}", e.view_type(id, dtype));
            e.line(6, &text, None);
            e.views.insert((k as u32, ty), view);
        }
    }
    e.region(&ir.body, 6);
    e.line(6, "gpu.return", None);
    e.line(4, "}", None);
    e.line(2, "}", None);
    e.line(0, "}", None);
    e.out
}
