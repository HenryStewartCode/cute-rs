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
//!
//! With line information on, the kernel function gets a debug-info subprogram
//! (`#llvm.di_subprogram`, in a compile unit whose emission kind is
//! `DebugDirectivesOnly`), so that LLVM's NVPTX backend emits `.file` and
//! `.loc` directives with the Rust lines and columns. `LineTablesOnly` would
//! also emit DWARF sections and mark the target `, debug`, which `ptxas`
//! refuses for optimized code. MLIR takes an op's file from its scope, so ops
//! built in another file than the kernel's are given a
//! `#llvm.di_lexical_block_file` scope for that file.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};

use crate::ir::DYNAMIC_SMEM_SYMBOL;
use crate::ir::{
    AsmConstraint, AsmInput, AtomicKind, AtomicType, BarrierKind, BinaryOp, BufferId, BulkCompletion, BulkDirection,
    CarryFn, CmpPred, CvtFn, DType, FenceScope, FloatFn, FloatMode, HalfFn, HalfModifiers, HalfType, Inst, IntFn,
    KernelIr, LaneMask, LdMatrixForm, Literal, Loc, LoopOptions, MatchKind, MbarrierWaitKind, MemAttrs, MemScopeKind,
    MemSpace, MemoryOrder, MemoryScope, MmaKind, MxKind, Op, Param, Region, ShuffleKind, SpecialReg, StMatrixForm,
    UnaryOp, ValueId, VoteKind, asm_clobbers_memory, asm_constraints, atomic_inst_name, carry_inst_name,
    float_inst_name, half_inst_name, int_inst_name,
};
use crate::tensor::AddressSpace;

mod wgmma;
// External calls, device functions and multimem.
mod extern_calls;
// m10-tmem-core.
mod tcgen05;

mod prims;

/// The length given to buffers sized at launch. Bare-pointer kernel arguments
/// need a static memref type; nothing depends on the value.
const RUNTIME_LEN: usize = i32::MAX as usize;

/// Options for [`emit_mlir`]. The default matches [`CompileOptions`](crate::CompileOptions)'
/// defaults: no locations, no assertions, the newest NVVM syntax, and contraction on.
#[derive(Clone, Debug)]
pub struct MlirOptions {
    /// Attach `loc("file":line:col)` source locations to every operation:
    /// the user code that built its instruction, or the kernel's
    /// `Kernel::build` call for the code around the body.
    pub locations: bool,
    /// Also give the kernel debug-info scopes, so that the PTX compiled from
    /// it carries line information (`.file` and `.loc`). Implies `locations`.
    pub line_info: bool,
    /// Emit the MMAs that need a recent NVVM dialect (FP8 `mma.sync`, LLVM
    /// 23; `mma.sp`, LLVM 22) as inline PTX instead
    /// ([`CompileOptions::mma_inline_ptx`](crate::CompileOptions::mma_inline_ptx)).
    pub mma_inline_ptx: bool,
    /// Emit device assertions ([`testing::assert_`](crate::testing::assert_))
    /// as `cf.assert`; without it they are left out
    /// ([`CompileOptions::enable_assertions`](crate::CompileOptions::enable_assertions)).
    pub assertions: bool,
    /// The major version of the LLVM whose `mlir-opt` lowers the module, for the operations
    /// whose NVVM syntax changed between versions (`nvvm.stmatrix` takes shape and element
    /// type attributes from LLVM 22, and LLVM 21's lowers to inline assembly that is not
    /// convergent); `None` writes the newest syntax. `Kernel::compile_with` passes the
    /// toolchain's.
    pub llvm_major: Option<u32>,
    /// Emit Float `+`, `-` and `*` with `fastmath<contract>`, so that LLVM and `ptxas` may
    /// fuse them into `fma` ([`CompileOptions::fp_contract`](crate::CompileOptions::fp_contract)).
    pub fp_contract: bool,
}

impl Default for MlirOptions {
    fn default() -> MlirOptions {
        MlirOptions {
            locations: false,
            line_info: false,
            mma_inline_ptx: false,
            assertions: false,
            llvm_major: None,
            fp_contract: true,
        }
    }
}

struct Emitter<'a> {
    ir: &'a KernelIr,
    opts: &'a MlirOptions,
    out: String,
    tmp: usize,
    /// The location of the instruction being emitted.
    current: Option<Loc>,
    /// The profiling scope of the instruction being emitted.
    current_scope: Option<crate::profiling::ScopeId>,
    /// With line information, the source files other than the kernel's, each
    /// with a `#di_lbf<n>` scope.
    other_files: Vec<&'static str>,
    /// Values emitted as another value (`make_warp_uniform`).
    aliases: BTreeMap<ValueId, ValueId>,
    /// Values emitted under another SSA name: an Int4 result after its
    /// re-wrap to 4 bits, an Int4 converted to Int8 (the same container).
    renames: BTreeMap<ValueId, String>,
    /// Buffers read as more than one element type (or placed in dynamic
    /// shared memory) are bytes with a typed view per element type.
    bytes: Vec<bool>,
    views: BTreeMap<(u32, &'static str), String>,
    dynamic_smem: bool,
    /// Where the dynamic shared-memory region starts in the window.
    dynamic_start: usize,
    /// Each shared buffer's start in the window, by buffer (`None` for the
    /// other spaces): [`SharedMemoryLayout::starts`](crate::ir::SharedMemoryLayout).
    window_starts: Vec<Option<usize>>,
    /// The kernel loads or stores tensor memory: thread
    /// synchronisations get the tcgen05 thread-sync fences.
    tcgen05_fences: bool,
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
        Literal::I4(v) => format!("{v} : i8"),
        Literal::I128(v) => format!("{v} : i128"),
        Literal::U128(v) => format!("{v} : i128"),
        Literal::F8(_, b) => format!("{} : i8", b as i8),
        Literal::F6x4(_, b) => format!("{} : i32", b as i32),
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
    crate::ir::asm_register(ty)
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
        let v = *self.aliases.get(&v).unwrap_or(&v);
        match self.renames.get(&v) {
            Some(name) => name.clone(),
            None => format!("%v{}", v.0),
        }
    }

    /// Sign-extend the low 4 bits of the `i8` named `x` into the Int4
    /// `result`'s container.
    fn wrap_int4(&mut self, indent: usize, x: &str, result: ValueId) {
        let (four, up, out) = (self.fresh("c"), self.fresh("i4u"), self.fresh("i4w"));
        self.line(indent, &format!("{four} = arith.constant 4 : i8"), None);
        self.line(indent, &format!("{up} = arith.shli {x}, {four} : i8"), None);
        self.line(indent, &format!("{out} = arith.shrsi {up}, {four} : i8"), None);
        self.renames.insert(result, out);
    }

    fn ty(&self, v: ValueId) -> &'static str {
        self.ir.dtype(v).mlir()
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.tmp += 1;
        format!("%{prefix}{}", self.tmp)
    }

    fn located(&self) -> bool {
        self.opts.locations || self.opts.line_info
    }

    /// Emit one line. With locations on, an operation gets the location of
    /// `inst`, or else of the instruction being emitted, or else of the
    /// kernel; a line that opens a region (ends with `{`) or a block (ends
    /// with `:`) gets none, since an op's location follows its regions.
    fn line(&mut self, indent: usize, text: &str, inst: Option<&Inst>) {
        let _ = write!(self.out, "{:indent$}{text}", "", indent = indent);
        let opens = text.trim_end().ends_with(['{', ':']);
        if self.located() && !opens {
            let loc = inst.map(|i| i.loc).or(self.current).unwrap_or(self.ir.loc);
            let scope = inst.map_or(self.current_scope, |i| i.scope);
            let text = self.scoped_loc(scope, self.loc(loc));
            self.out.push(' ');
            self.out.push_str(&text);
        }
        self.out.push('\n');
    }

    /// `base` (a `loc(...)`) wrapped in the NameLocs of `scope`'s path, outermost name
    /// outside: `loc("a"("b"("file":1:1)))`.
    fn scoped_loc(&self, scope: Option<crate::profiling::ScopeId>, base: String) -> String {
        let Some(mut at) = scope else { return base };
        let mut text = base.strip_prefix("loc(").and_then(|b| b.strip_suffix(')')).unwrap_or(&base).to_string();
        loop {
            let s = &self.ir.scopes[at.0 as usize];
            text = format!("\"{}\"({text})", escape(&s.name));
            match s.parent {
                Some(parent) => at = parent,
                None => break,
            }
        }
        format!("loc({text})")
    }

    /// Emit a line with no location (structure outside the kernel function).
    fn plain_line(&mut self, indent: usize, text: &str) {
        let _ = writeln!(self.out, "{:indent$}{text}", "", indent = indent);
    }

    /// `loc(...)` for code built at `loc`: plain `"file":line:col`, scoped
    /// with a lexical block file when line information is on and `loc` is in
    /// another file than the kernel. (An op that must keep a plain location
    /// whatever its file, such as `cf.assert`, whose `__assertfail` call loses
    /// the file name under a fused location, uses [`Self::file_line_col`].)
    fn loc(&self, loc: Loc) -> String {
        let plain = Self::file_line_col(loc);
        match self.other_files.iter().position(|&f| f == loc.file()) {
            Some(n) if self.opts.line_info => format!("loc(fused<#di_lbf{n}>[{plain}])"),
            _ => format!("loc({plain})"),
        }
    }

    fn file_line_col(loc: Loc) -> String {
        format!("\"{}\":{}:{}", escape(loc.file()), loc.line(), loc.column())
    }

    /// The location of the kernel function: its `Kernel::build` call, with
    /// its debug-info subprogram when line information is on.
    fn function_loc(&self) -> String {
        let plain = Self::file_line_col(self.ir.loc);
        if self.opts.line_info { format!("loc(fused<#di_sp>[{plain}])") } else { format!("loc({plain})") }
    }

    /// The debug-info attributes (`#di_*` aliases) line information needs.
    fn debug_info_header(&mut self) {
        let kernel_file = self.ir.loc.file();
        let mut files = BTreeSet::new();
        collect_files(&self.ir.body, &mut files);
        files.remove(kernel_file);
        self.other_files = files.into_iter().collect();
        let di_file = |file: &str| {
            let (name, dir) = di_file_parts(file);
            format!("#llvm.di_file<\"{}\" in \"{}\">", escape(&name), escape(&dir))
        };
        let mut header = String::new();
        let _ = writeln!(header, "#di_file = {}", di_file(kernel_file));
        for (n, f) in self.other_files.iter().enumerate() {
            let _ = writeln!(header, "#di_file{n} = {}", di_file(f));
        }
        let _ = writeln!(
            header,
            "#di_cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Rust, file = #di_file, \
             producer = \"cute-rs\", isOptimized = true, emissionKind = DebugDirectivesOnly>"
        );
        let _ = writeln!(header, "#di_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>");
        let (name, line) = (&self.ir.name, self.ir.loc.line());
        let _ = writeln!(
            header,
            "#di_sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #di_cu, scope = #di_file, \
             name = \"{name}\", linkageName = \"{name}\", file = #di_file, line = {line}, scopeLine = {line}, \
             subprogramFlags = \"Definition|Optimized\", type = #di_type>"
        );
        for n in 0..self.other_files.len() {
            let _ = writeln!(
                header,
                "#di_lbf{n} = #llvm.di_lexical_block_file<scope = #di_sp, file = #di_file{n}, discriminator = 0>"
            );
        }
        self.out.push_str(&header);
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

    /// A view of `b` to take its address from, for an address counted in
    /// elements of `ty`: [`Self::access`], or for a byte buffer with no
    /// view of `ty` (an address taken, [`Op::PtrToInt`], as a type the
    /// buffer is never accessed as) another of its views, which all start
    /// at the buffer's first byte.
    fn address_view(&self, b: BufferId, ty: DType) -> (String, String) {
        if self.bytes[b.0 as usize] && !self.views.contains_key(&(b.0, ty.mlir())) {
            let found = self.views.iter().find(|((k, _), _)| *k == b.0);
            if let Some(((_, t), view)) = found {
                return (view.clone(), self.view_type(b, dtype_of(t)));
            }
        }
        self.access(b, ty)
    }

    fn index(&mut self, indent: usize, v: ValueId) -> String {
        let name = self.fresh("i");
        let text = format!("{name} = arith.index_cast {} : {} to index", self.v(v), self.ir.dtype(v).mlir());
        self.line(indent, &text, None);
        name
    }

    fn index_const(&mut self, indent: usize, c: usize) -> String {
        let name = self.fresh("c");
        self.line(indent, &format!("{name} = arith.constant {c} : index"), None);
        name
    }

    /// `ix + k` as an index (`ix` itself for `k = 0`).
    fn offset_index(&mut self, indent: usize, ix: &str, k: usize) -> String {
        if k == 0 {
            return ix.to_string();
        }
        let c = self.index_const(indent, k);
        let name = self.fresh("i");
        self.line(indent, &format!("{name} = arith.addi {ix}, {c} : index"), None);
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
        if to == DType::I4 {
            // To the i8 container (through Int32 from a float), then wrapped.
            let container = if from.mlir() == "i8" {
                self.v(x)
            } else {
                let (mut value, mut source) = (self.v(x), from);
                if from.is_float() {
                    let wide = self.fresh("i");
                    let ext = if from == DType::BF16 { self.extend(indent, x) } else { value.clone() };
                    let src_ty = if from == DType::BF16 { "f32" } else { from.mlir() };
                    self.line(indent, &format!("{wide} = arith.fptosi {ext} : {src_ty} to i32"), Some(inst));
                    value = wide;
                    source = DType::I32;
                }
                let byte = self.fresh("i");
                let op = cast_op(source, DType::I8);
                self.line(indent, &format!("{byte} = arith.{op} {value} : {} to i8", source.mlir()), Some(inst));
                byte
            };
            self.wrap_int4(indent, &container, result);
            return;
        }
        if from == DType::I4 && to.mlir() == "i8" {
            // Int8 and Uint8 share the sign-extended container.
            let name = self.v(x);
            self.renames.insert(result, name);
            return;
        }
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
        if reg == SpecialReg::GlobalTimerLo {
            // The NVVM dialect has no globaltimer.lo before LLVM 23. Each
            // read is a new value, as with the NVVM timer reads.
            let results = self.asm_operands(&[result]);
            self.ptx_inst(indent, inst, "mov.u32 $0, %globaltimer_lo;", &results, &[], true);
            return;
        }
        let smem_register = match reg {
            SpecialReg::DynamicSmemSize => Some("dynamic_smem_size"),
            SpecialReg::TotalSmemSize => Some("total_smem_size"),
            SpecialReg::AggrSmemSize => Some("aggr_smem_size"),
            _ => None,
        };
        if let Some(name) = smem_register {
            // LLVM 21 has no NVVM op for the shared-memory size registers
            // (LLVM 23 adds them), so one pure `mov` reads each: they hold
            // the same value for the whole launch.
            let template = format!("mov.u32 $0, %{name};");
            if reg == SpecialReg::DynamicSmemSize && self.dynamic_start > 0 {
                // The launch passed the build-time buffers in the window too.
                let window = self.fresh("window");
                self.ptx_inst(indent, inst, &template, &[(window.clone(), DType::I32)], &[], false);
                let start = self.fresh("c");
                self.line(indent, &format!("{start} = arith.constant {} : i32", self.dynamic_start), None);
                self.line(indent, &format!("{} = arith.subi {window}, {start} : i32", self.v(result)), None);
            } else {
                let results = self.asm_operands(&[result]);
                self.ptx_inst(indent, inst, &template, &results, &[], false);
            }
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
            SpecialReg::ClusterCtaRank => "cluster.ctarank".to_string(),
            SpecialReg::ClusterNCtaRank => "cluster.nctarank".to_string(),
            _ => unreachable!("handled above"),
        };
        let text = format!("{} = nvvm.read.ptx.sreg.{name} : {}", self.v(result), reg.dtype().mlir());
        self.line(indent, &text, Some(inst));
    }

    /// Where shared buffer `b` starts in the dynamic shared-memory window, if
    /// the kernel's shared buffers live there.
    fn window_start(&self, b: BufferId) -> Option<usize> {
        if !self.dynamic_smem || self.ir.buffer(b).space != MemSpace::Shared {
            return None;
        }
        Some(self.window_starts[b.0 as usize].unwrap_or(0))
    }

    /// The address of element `index` (counting `ty`s) of a buffer, as an i64.
    fn address(&mut self, indent: usize, b: BufferId, ty: DType, index: ValueId) -> String {
        if self.ir.buffer(b).space.raw().is_some() {
            // A raw pointer's access: the index is the byte address.
            return self.v(index);
        }
        let addr = match self.window_start(b) {
            // A buffer in the dynamic window: the window's own aligned pointer plus the
            // buffer's start, added here. Taking the pointer of the buffer's view
            // (`memref.view` at that start) would be shorter, but LLVM 21's
            // `--expand-strided-metadata` folds `extract_aligned_pointer_as_index` of a view
            // into the same op on the view's source, dropping the view's byte offset (LLVM
            // 22 and 23 keep it).
            Some(start) => {
                let base = self.fresh("base");
                let text = format!(
                    "{base} = memref.extract_aligned_pointer_as_index %dyn_smem : memref<0xi8, #gpu.address_space<workgroup>> -> index"
                );
                self.line(indent, &text, None);
                let window = self.fresh("window");
                self.line(indent, &format!("{window} = arith.index_cast {base} : index to i64"), None);
                let offset = self.fresh("start");
                self.line(indent, &format!("{offset} = arith.constant {start} : i64"), None);
                let addr = self.fresh("addr");
                self.line(indent, &format!("{addr} = arith.addi {window}, {offset} : i64"), None);
                addr
            }
            None => {
                let (name, memref) = self.address_view(b, ty);
                let base = self.fresh("base");
                self.line(
                    indent,
                    &format!("{base} = memref.extract_aligned_pointer_as_index {name} : {memref} -> index"),
                    None,
                );
                let addr = self.fresh("addr");
                self.line(indent, &format!("{addr} = arith.index_cast {base} : index to i64"), None);
                addr
            }
        };
        // A 64-bit index is used as it is; an i32 one is sign-extended.
        let wide = if self.ir.dtype(index) == DType::I64 {
            self.v(index)
        } else {
            let wide = self.fresh("ix");
            self.line(indent, &format!("{wide} = arith.extsi {} : i32 to i64", self.v(index)), None);
            wide
        };
        let size = self.fresh("size");
        self.line(indent, &format!("{size} = arith.constant {} : i64", ty.bytes()), None);
        let off = self.fresh("off");
        self.line(indent, &format!("{off} = arith.muli {wide}, {size} : i64"), None);
        let sum = self.fresh("sum");
        self.line(indent, &format!("{sum} = arith.addi {addr}, {off} : i64"), None);
        sum
    }

    /// The `.shared::cluster` address (an i64, `mapa`) of element `index`
    /// (counting `ty`s) of shared buffer `b` in CTA `rank` of the cluster. `mapa` is inline PTX: NVVM's `mapa` op gives
    /// a pointer of the shared space in LLVM 21 and of the cluster space
    /// (7) in LLVM 23.
    fn dsmem_address(&mut self, indent: usize, b: BufferId, ty: DType, index: ValueId, rank: ValueId) -> String {
        let local = self.address(indent, b, ty, index);
        let remote = self.fresh("dsmem");
        let text = format!(
            "{remote} = llvm.inline_asm \"mapa.shared::cluster.u64 $0, $1, $2;\", \"=l,l,r\" {local}, {} : (i64, i32) -> i64",
            self.v(rank)
        );
        self.line(indent, &text, None);
        remote
    }

    /// A load into `values` or store of them to another CTA's shared memory: at the `mapa` address, a plain access as an
    /// LLVM load or store through a `.shared::cluster` pointer (address
    /// space 7, which LLVM 21 lowers to `ld.shared::cluster` and
    /// `st.shared::cluster`), one with memory attributes as
    /// [`Self::load_with_attrs`] emits it on that space.
    #[allow(clippy::too_many_arguments)]
    fn dsmem_access(
        &mut self,
        indent: usize,
        inst: &Inst,
        values: &[ValueId],
        buffer: BufferId,
        index: ValueId,
        attrs: &MemAttrs,
        load: bool,
    ) {
        let elem = self.ir.dtype(values[0]);
        let rank = attrs.peer.expect("a distributed-shared-memory access");
        let addr = self.dsmem_address(indent, buffer, elem, index, rank);
        let space = MemSpace::Raw(AddressSpace::Dsmem);
        if !attrs.is_plain() {
            if load {
                self.load_with_attrs_at(indent, inst, values, addr, space, attrs);
            } else {
                self.store_with_attrs_at(indent, inst, values, addr, space, attrs);
            }
            return;
        }
        let n = values.len();
        let align = n * elem.bytes();
        self.ptr_access(indent, inst, values, &addr, space, align, load);
    }

    /// An atomic (`result`, the old value) or a reduction (no result), as
    /// one PTX `atom` or `red` with explicit semantics and scope
    /// ([`atomic_inst_name`]), in inline assembly with side effects. LLVM's
    /// own atomics would carry neither reliably: LLVM 21 drops the ordering
    /// and scope of `atomicrmw` (an acq_rel add becomes a relaxed one at GPU
    /// scope) and the scope of `cmpxchg`, which LLVM 23 keeps, and NVVM has
    /// no reduction op. Scalar Float16 and BFloat16 operands go in and come
    /// out as their bits (`h` registers), packed pairs as Int32s.
    #[allow(clippy::too_many_arguments)]
    fn atomic(
        &mut self,
        indent: usize,
        inst: &Inst,
        result: Option<ValueId>,
        kind: AtomicKind,
        ty: AtomicType,
        (buffer, index, peer): (BufferId, ValueId, Option<ValueId>),
        value: ValueId,
        compare: Option<ValueId>,
        sem: MemoryOrder,
        scope: MemoryScope,
    ) {
        let vt = self.ir.dtype(value);
        let (space, addr) = match peer {
            // Another CTA's shared memory: `.shared::cluster`.
            Some(rank) => (
                MemSpace::Raw(AddressSpace::Dsmem),
                self.dsmem_address(indent, buffer, ty.access_elem(vt), index, rank),
            ),
            None => (self.ir.buffer(buffer).space, self.address(indent, buffer, ty.access_elem(vt), index)),
        };
        let name = atomic_inst_name(result.is_none(), kind, ty, sem, scope, space);
        let half = !ty.is_packed() && ty.half().is_some();
        let reg = if half { DType::I16 } else { vt };
        let mut inputs = vec![(addr, DType::I64)];
        // `cas` takes the comparand before the new value.
        for v in compare.into_iter().chain([value]) {
            let name = if half { self.as_bits16(indent, v) } else { self.v(v) };
            inputs.push((name, reg));
        }
        let Some(result) = result else {
            self.ptx_inst(indent, inst, &format!("{name} [$0], $1;"), &[], &inputs, true);
            return;
        };
        let operands: Vec<String> = (2..=inputs.len()).map(|k| format!("${k}")).collect();
        let template = format!("{name} $0, [$1], {};", operands.join(", "));
        let out = if half { self.fresh("h") } else { self.v(result) };
        self.ptx_inst(indent, inst, &template, &[(out.clone(), reg)], &inputs, true);
        if half {
            let text = format!("{} = arith.bitcast {out} : i16 to {}", self.v(result), vt.mlir());
            self.line(indent, &text, None);
        }
    }

    /// `parts` (SSA names of `n` values of type `part`) as one value of type
    /// `to`, part 0 in the low bits ([`Op::Pack`]): a vector of the parts,
    /// bitcast. The SSA name of the result is `out`, or a fresh one.
    fn pack_bits(&mut self, indent: usize, parts: &[String], part: DType, to: DType, out: Option<String>) -> String {
        if part.is_word_packed() {
            return self.pack_sub_byte(indent, parts, part, to, out);
        }
        let vector = format!("vector<{}x{}>", parts.len(), part.mlir());
        let v = self.fresh("pack");
        self.line(indent, &format!("{v} = vector.from_elements {} : {vector}", parts.join(", ")), None);
        let r = out.unwrap_or_else(|| self.fresh("packed"));
        self.line(indent, &format!("{r} = llvm.bitcast {v} : {vector} to {}", to.mlir()), None);
        r
    }

    /// `x` (an SSA name of type `from`) split into the SSA names `outs` of
    /// type `part`, the first from the low bits ([`Op::Unpack`]).
    fn unpack_bits(&mut self, indent: usize, x: &str, from: DType, part: DType, outs: &[String]) {
        if part.is_word_packed() {
            return self.unpack_sub_byte(indent, x, from, part, outs);
        }
        let vector = format!("vector<{}x{}>", outs.len(), part.mlir());
        let v = self.fresh("unpack");
        self.line(indent, &format!("{v} = llvm.bitcast {x} : {} to {vector}", from.mlir()), None);
        for (k, out) in outs.iter().enumerate() {
            self.line(indent, &format!("{out} = vector.extract {v}[{k}] : {} from {vector}", part.mlir()), None);
        }
    }

    /// [`Self::pack_bits`] of word-packed parts, which live in the low bits
    /// of their MLIR type, an `i8` (Int4 sign-extended, the FP4 and FP6
    /// values zero-extended) or an `i32` (FP6x4): each part's low bits
    /// shifted into place and or-ed together, any bits above them zero.
    fn pack_sub_byte(
        &mut self,
        indent: usize,
        parts: &[String],
        part: DType,
        to: DType,
        out: Option<String>,
    ) -> String {
        let (t, bits) = (to.mlir(), part.bits());
        let mask = self.fresh("c");
        self.line(indent, &format!("{mask} = arith.constant {} : {t}", (1u32 << bits) - 1), None);
        let r = out.unwrap_or_else(|| self.fresh("packed"));
        let mut acc: Option<String> = None;
        for (k, p) in parts.iter().enumerate() {
            let pc = part.mlir();
            let wide = if t == pc {
                p.clone()
            } else {
                let z = self.fresh("pz");
                self.line(indent, &format!("{z} = arith.extui {p} : {pc} to {t}"), None);
                z
            };
            let m = if parts.len() == 1 { r.clone() } else { self.fresh("pm") };
            self.line(indent, &format!("{m} = arith.andi {wide}, {mask} : {t}"), None);
            let placed = if k == 0 {
                m
            } else {
                let (c, sh) = (self.fresh("c"), self.fresh("ps"));
                self.line(indent, &format!("{c} = arith.constant {} : {t}", k * bits), None);
                self.line(indent, &format!("{sh} = arith.shli {m}, {c} : {t}"), None);
                sh
            };
            acc = Some(match acc {
                None => placed,
                Some(a) => {
                    let o = if k + 1 == parts.len() { r.clone() } else { self.fresh("po") };
                    self.line(indent, &format!("{o} = arith.ori {a}, {placed} : {t}"), None);
                    o
                }
            });
        }
        r
    }

    /// [`Self::unpack_bits`] into word-packed parts: each shifted down and
    /// truncated to the type it lives in, then sign-extended from its width
    /// (Int4) or masked to it. Fewer parts than fill `x` read its low bits.
    fn unpack_sub_byte(&mut self, indent: usize, x: &str, from: DType, part: DType, outs: &[String]) {
        let (t, bits) = (from.mlir(), part.bits());
        for (k, out) in outs.iter().enumerate() {
            let shifted = if k == 0 {
                x.to_string()
            } else {
                let (c, sh) = (self.fresh("c"), self.fresh("us"));
                self.line(indent, &format!("{c} = arith.constant {} : {t}", k * bits), None);
                self.line(indent, &format!("{sh} = arith.shrui {x}, {c} : {t}"), None);
                sh
            };
            let pc = part.mlir();
            let byte = if t == pc {
                shifted
            } else {
                let b = self.fresh("ub");
                self.line(indent, &format!("{b} = arith.trunci {shifted} : {t} to {pc}"), None);
                b
            };
            let c = self.fresh("c");
            if part.is_signed() && part.is_int() {
                let up = self.fresh("uu");
                self.line(indent, &format!("{c} = arith.constant {} : {pc}", 8 - bits), None);
                self.line(indent, &format!("{up} = arith.shli {byte}, {c} : {pc}"), None);
                self.line(indent, &format!("{out} = arith.shrsi {up}, {c} : {pc}"), None);
            } else {
                self.line(indent, &format!("{c} = arith.constant {} : {pc}", (1u32 << bits) - 1), None);
                self.line(indent, &format!("{out} = arith.andi {byte}, {c} : {pc}"), None);
            }
        }
    }

    /// A bf16 `mma.sync`. `nvgpu.mma.sync` cannot lower bf16 operands, so
    /// this emits `nvvm.mma.sync`, which takes each pair of bf16 values packed
    /// into an i32 (low half first, as in a `.bf16x2` register), packed as
    /// [`Op::Pack`] is.
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
                    let names: Vec<String> = pair.iter().map(|&v| e.v(v)).collect();
                    e.pack_bits(indent, &names, DType::BF16, DType::I32, None)
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

    /// An FP8 `mma.sync` or a sparse `mma.sp::ordered_metadata.sync` (with
    /// sparsity selector 0), as `nvvm.mma.sync` and `nvvm.mma.sp.sync`, or
    /// with [`MlirOptions::mma_inline_ptx`] as inline PTX. The operand
    /// registers are 32 bits: four FP8 values, two f16 or bf16 values
    /// ([`Op::Pack`]'s order, the first in the low bits); `nvvm` takes f16
    /// pairs (and f16 accumulators) as `vector<2xf16>`.
    #[allow(clippy::too_many_arguments)]
    fn mma_nvvm(
        &mut self,
        indent: usize,
        inst: &Inst,
        kind: MmaKind,
        d: &[ValueId],
        a: &[ValueId],
        b: &[ValueId],
        c: &[ValueId],
        e: Option<ValueId>,
    ) {
        let (ta, _, tc) = kind.dtypes();
        let inline = self.opts.mma_inline_ptx;
        let per = 4 / ta.bytes();
        // One 32-bit register of operand values: an i32, or for nvvm a
        // vector<2xf16>.
        let reg = |e: &mut Self, values: &[ValueId], ty: DType| -> String {
            let names: Vec<String> = values.iter().map(|&v| e.v(v)).collect();
            if ty == DType::F16 && !inline {
                let r = e.fresh("pair");
                e.line(indent, &format!("{r} = vector.from_elements {} : vector<2xf16>", names.join(", ")), None);
                r
            } else {
                e.pack_bits(indent, &names, ty, DType::I32, None)
            }
        };
        let ra: Vec<String> = a.chunks(per).map(|p| reg(self, p, ta)).collect();
        let rb: Vec<String> = b.chunks(per).map(|p| reg(self, p, ta)).collect();
        let rc: Vec<String> = if tc == DType::F16 {
            c.chunks(2).map(|p| reg(self, p, DType::F16)).collect()
        } else {
            c.iter().map(|&v| self.v(v)).collect()
        };
        let meta = e.map(|e| self.v(e));
        if inline {
            // D, A, B, C, then the metadata and the selector.
            let out_ty = if tc == DType::F16 { DType::I32 } else { tc };
            let outs: Vec<(String, DType)> = if tc == DType::F16 {
                (0..rc.len()).map(|_| (self.fresh("d"), out_ty)).collect()
            } else {
                d.iter().map(|&v| (self.v(v), out_ty)).collect()
            };
            let mut k = 0;
            let mut group = |n: usize| {
                let names: Vec<String> = (k..k + n).map(|j| format!("${j}")).collect();
                k += n;
                format!("{{{}}}", names.join(", "))
            };
            let (gd, ga, gb, gc) = (group(outs.len()), group(ra.len()), group(rb.len()), group(rc.len()));
            let tail = if meta.is_some() { format!(", ${k}, 0") } else { String::new() };
            let template = format!("{} {gd}, {ga}, {gb}, {gc}{tail};", kind.ptx());
            let mut inputs: Vec<(String, DType)> = ra.iter().chain(&rb).map(|r| (r.clone(), DType::I32)).collect();
            inputs.extend(rc.iter().map(|r| (r.clone(), out_ty)));
            inputs.extend(meta.iter().map(|m| (m.clone(), DType::I32)));
            self.ptx_inst(indent, inst, &template, &outs, &inputs, true);
            if tc == DType::F16 {
                for (pair, (name, _)) in d.chunks(2).zip(&outs) {
                    let names: Vec<String> = pair.iter().map(|&v| self.v(v)).collect();
                    self.unpack_bits(indent, name, DType::I32, DType::F16, &names);
                }
            }
            return;
        }
        let (m, n, k) = kind.shape_mnk();
        let r = self.fresh("mma");
        let ab_reg = if ta == DType::F16 { "vector<2xf16>" } else { "i32" };
        let c_reg = if tc == DType::F16 { "vector<2xf16>" } else { "f32" };
        let ret = format!("!llvm.struct<({})>", vec![c_reg; rc.len()].join(", "));
        let ptx_type = match ta {
            DType::BF16 => "bf16",
            DType::F8E4M3FN => "e4m3",
            DType::F8E5M2 => "e5m2",
            _ => "",
        };
        let types = if ptx_type.is_empty() {
            String::new()
        } else {
            format!(
                "multiplicandAPtxType = #nvvm.mma_type<{ptx_type}>, multiplicandBPtxType = #nvvm.mma_type<{ptx_type}>, "
            )
        };
        let shape = format!("shape = #nvvm.shape<m = {m}, n = {n}, k = {k}>");
        let text = match meta {
            None => format!(
                "{r} = nvvm.mma.sync A[{}] B[{}] C[{}] {{layoutA = #nvvm.mma_layout<row>, layoutB = \
                 #nvvm.mma_layout<col>, {types}{shape}}} : ({ab_reg}, {ab_reg}, {c_reg}) -> {ret}",
                ra.join(", "),
                rb.join(", "),
                rc.join(", ")
            ),
            Some(meta) => {
                let sel = self.fresh("sel");
                self.line(indent, &format!("{sel} = arith.constant 0 : i32"), None);
                format!(
                    "{r} = nvvm.mma.sp.sync A[{}] B[{}] C[{}] sparseMetadata[{meta}] selector[{sel}] \
                     {{orderedMetadata, {types}{shape}}} : ({ab_reg}, {ab_reg}, {c_reg}) -> {ret}",
                    ra.join(", "),
                    rb.join(", "),
                    rc.join(", ")
                )
            }
        };
        self.line(indent, &text, Some(inst));
        if tc == DType::F16 {
            for (idx, pair) in d.chunks(2).enumerate() {
                let x = self.fresh("dpair");
                self.line(indent, &format!("{x} = llvm.extractvalue {r}[{idx}] : {ret}"), None);
                for (j, &v) in pair.iter().enumerate() {
                    let text = format!("{} = vector.extract {x}[{j}] : f16 from vector<2xf16>", self.v(v));
                    self.line(indent, &text, None);
                }
            }
        } else {
            for (idx, out) in d.iter().enumerate() {
                self.line(indent, &format!("{} = llvm.extractvalue {r}[{idx}] : {ret}", self.v(*out)), None);
            }
        }
    }

    // m10-sm120-blockscaled.
    /// A warp-level block-scaled `mma.sync` (sm_120a): `nvvm.mma.block_scale`
    /// (LLVM 22), or with [`MlirOptions::mma_inline_ptx`] inline PTX (for
    /// LLVM 21, whose NVVM dialect lacks the operation; the inline assembly
    /// is not marked convergent, a caveat it shares with the FP8 and sparse
    /// fallbacks). The operand and scale registers are i32; byte-id and
    /// thread-id are 0.
    fn mma_block_scaled(&mut self, indent: usize, inst: &Inst) {
        let Op::MmaBlockScaled { kind, d, a, b, c, sfa, sfb } = &inst.op else { unreachable!() };
        let ra: Vec<String> = a.iter().map(|&v| self.v(v)).collect();
        let rb: Vec<String> = b.iter().map(|&v| self.v(v)).collect();
        let rc: Vec<String> = c.iter().map(|&v| self.v(v)).collect();
        let (sa, sb) = (self.v(*sfa), self.v(*sfb));
        let zero = self.fresh("sfid");
        self.line(indent, &format!("{zero} = arith.constant 0 : i16"), None);
        if self.opts.mma_inline_ptx {
            let outs: Vec<(String, DType)> = d.iter().map(|&v| (self.v(v), DType::F32)).collect();
            let template = format!(
                "{} {{$0, $1, $2, $3}}, {{$4, $5, $6, $7}}, {{$8, $9}}, {{$10, $11, $12, $13}}, {{$14}}, {{$15, \
                 $16}}, {{$17}}, {{$18, $19}};",
                kind.ptx()
            );
            let mut inputs: Vec<(String, DType)> = ra.iter().chain(&rb).map(|r| (r.clone(), DType::I32)).collect();
            inputs.extend(rc.iter().map(|r| (r.clone(), DType::F32)));
            for s in [&sa, &sb] {
                inputs.extend([(s.clone(), DType::I32), (zero.clone(), DType::I16), (zero.clone(), DType::I16)]);
            }
            self.ptx_inst(indent, inst, &template, &outs, &inputs, true);
            return;
        }
        let (m, n, k) = kind.shape_mnk();
        let r = self.fresh("mma");
        let ret = "!llvm.struct<(f32, f32, f32, f32)>";
        let svs = match kind.scales_per_mma() {
            1 => "x1",
            2 => "x2",
            _ => "x4",
        };
        let text = format!(
            "{r} = nvvm.mma.block_scale A[{}] B[{}] C[{}] scaleA[{sa}, {zero}, {zero}] scaleB[{sb}, {zero}, {zero}] \
             {{shape = #nvvm.shape<m = {m}, n = {n}, k = {k}>, multiplicandAPtxType = #nvvm.mma_type<{}>, \
             multiplicandBPtxType = #nvvm.mma_type<{}>, scaleVecSize = #nvvm.scale_vec_size<{svs}>, \
             blockScaleFormat = #nvvm.block_scale_format<{}>, kind = #nvvm.block_scale_kind<{}>}} : (i32, i32, \
             f32) -> {ret}",
            ra.join(", "),
            rb.join(", "),
            rc.join(", "),
            crate::ir::BlockScaledMmaKind::ptx_type(kind.a),
            crate::ir::BlockScaledMmaKind::ptx_type(kind.b),
            kind.ptx_sf_type(),
            kind.kind_name()
        );
        self.line(indent, &text, Some(inst));
        for (idx, out) in d.iter().enumerate() {
            self.line(indent, &format!("{} = llvm.extractvalue {r}[{idx}] : {ret}", self.v(*out)), None);
        }
    }

    // m11-prims-mma-forms.
    /// A warp-level sparse block-scaled `mma.sp` (sm_120a), always as
    /// inline PTX: mlir-opt 21 has no `nvvm.mma.sp.block_scale` (LLVM 23
    /// does). LLVM does not treat inline assembly as convergent, so the
    /// instruction must stay out of divergent control flow (the caveat of
    /// the FP8, sparse and block-scaled fallbacks). The sparsity selector
    /// is 0; the byte and thread ids are immediates.
    fn mma_sp_block_scaled(&mut self, indent: usize, inst: &Inst) {
        let Op::MmaSpBlockScaled { kind, d, a, b, c, e, sfa, sfb } = &inst.op else { unreachable!() };
        let outs: Vec<(String, DType)> = d.iter().map(|&v| (self.v(v), DType::F32)).collect();
        let template = format!(
            "{} {{$0, $1, $2, $3}}, {{$4, $5, $6, $7}}, {{$8, $9, $10, $11}}, {{$12, $13, $14, $15}}, {{$16}}, 0x0, \
             {{$17}}, {{$18, $19}}, {{$20}}, {{$21, $22}};",
            kind.ptx()
        );
        let mut inputs: Vec<(String, DType)> = a.iter().chain(b).map(|&v| (self.v(v), DType::I32)).collect();
        inputs.extend(c.iter().map(|&v| (self.v(v), DType::F32)));
        inputs.push((self.v(*e), DType::I32));
        for (sf, bid, tid) in [(*sfa, kind.bid_a, kind.tid_a), (*sfb, kind.bid_b, kind.tid_b)] {
            inputs.push((self.v(sf), DType::I32));
            for id in [bid, tid] {
                let c = self.fresh("sfid");
                self.line(indent, &format!("{c} = arith.constant {id} : i16"), None);
                inputs.push((c, DType::I16));
            }
        }
        self.ptx_inst(indent, inst, &template, &outs, &inputs, true);
    }

    /// Values as `(SSA name, type)` operands of [`Self::ptx_inst`].
    fn asm_operands(&self, values: &[ValueId]) -> Vec<(String, DType)> {
        values.iter().map(|&v| (self.v(v), self.ir.dtype(v))).collect()
    }

    /// The operands of a TMA instruction's `.override::*` qualifiers, appended to `operands`: the qualifiers, and the text that goes
    /// inside the tensor-map brackets before the coordinates (`$a, ` with the address
    /// only; `$a, {sizes}, {lower strides}, $upper, ` with the attributes, the sizes and
    /// upper strides as 16-bit registers).
    fn tma_override_operands(
        &mut self,
        _indent: usize,
        o: &crate::ir::TmaOverride,
        operands: &mut Vec<String>,
        types: &mut Vec<&'static str>,
        constraints: &mut Vec<&'static str>,
    ) -> (String, String) {
        let mut push = |name: String, ty: &'static str, c: &'static str| {
            let slot = format!("${}", operands.len());
            operands.push(name);
            types.push(ty);
            constraints.push(c);
            slot
        };
        let a = push(self.v(o.global_address), "i64", "l");
        if o.dims.is_empty() {
            return (".override::global_address".to_string(), format!("{a}, "));
        }
        let sizes: Vec<String> = o.dims.iter().map(|v| push(self.v(*v), "i16", "h")).collect();
        if o.lower_strides.is_empty() {
            return (
                ".override::global_address.override::global_dim".to_string(),
                format!("{a}, {{{}}}, ", sizes.join(", ")),
            );
        }
        let lows: Vec<String> = o.lower_strides.iter().map(|v| push(self.v(*v), "i32", "r")).collect();
        let up = push(self.v(o.upper_strides.expect("validated: the upper strides")), "i16", "h");
        (
            ".override::global_address.override::global_dim_stride".to_string(),
            format!("{a}, {{{}}}, {{{}}}, {up}, ", sizes.join(", "), lows.join(", ")),
        )
    }

    /// Inline PTX with side effects and a memory clobber, taking `inputs`.
    fn asm_memory(&mut self, indent: usize, inst: &Inst, template: &str, inputs: &[(String, DType)]) {
        let constraints: Vec<&str> =
            inputs.iter().map(|(_, t)| constraint(*t)).chain(std::iter::once("~{memory}")).collect();
        let args: Vec<(String, &str)> = inputs.iter().map(|(name, t)| (name.clone(), t.mlir())).collect();
        self.asm_line(indent, inst, template, &constraints.join(","), &[], &args, true);
    }

    /// One PTX instruction (or a fixed sequence of them) as
    /// `llvm.inline_asm`. `template` names the results `$0, $1, ...` and
    /// then the inputs, with any build-time immediates already spliced in.
    /// Operands are SSA names with their types; each type's register class
    /// is its constraint ([`constraint`]). Several results come back as an
    /// LLVM struct and are unpacked into their names. Without
    /// `side_effects` the assembly is pure, so LLVM may CSE, hoist or delete
    /// it like any arithmetic: right for instructions whose result depends
    /// on their operands alone (no memory, not warp-collective).
    fn ptx_inst(
        &mut self,
        indent: usize,
        inst: &Inst,
        template: &str,
        results: &[(String, DType)],
        inputs: &[(String, DType)],
        side_effects: bool,
    ) {
        let constraints: Vec<String> = results
            .iter()
            .map(|(_, t)| format!("={}", constraint(*t)))
            .chain(inputs.iter().map(|(_, t)| constraint(*t).to_string()))
            .collect();
        let args: Vec<(String, &str)> = inputs.iter().map(|(name, t)| (name.clone(), t.mlir())).collect();
        self.asm_line(indent, inst, template, &constraints.join(","), results, &args, side_effects);
    }

    /// One statement of inline assembly with constraint string
    /// `constraints`: `results` (SSA names and types) come out of it, as an
    /// LLVM struct unpacked into their names when there are several, and
    /// `args` (SSA names and MLIR types) go into it.
    #[allow(clippy::too_many_arguments)]
    fn asm_line(
        &mut self,
        indent: usize,
        inst: &Inst,
        template: &str,
        constraints: &str,
        results: &[(String, DType)],
        args: &[(String, &str)],
        side_effects: bool,
    ) {
        let effects = if side_effects { "has_side_effects " } else { "" };
        let arg_types = args.iter().map(|(_, t)| *t).collect::<Vec<_>>().join(", ");
        let args = args.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ");
        let head = format!("llvm.inline_asm {effects}\"{}\", \"{}\"", escape(template), escape(constraints));
        let ret = match results {
            [] => "()".to_string(),
            [(_, t)] => t.mlir().to_string(),
            _ => format!("!llvm.struct<({})>", results.iter().map(|(_, t)| t.mlir()).collect::<Vec<_>>().join(", ")),
        };
        let lhs = match results {
            [] => String::new(),
            [(name, _)] => format!("{name} = "),
            _ => format!("{} = ", self.fresh("asm")),
        };
        self.line(indent, &format!("{lhs}{head} {args} : ({arg_types}) -> {ret}"), Some(inst));
        if results.len() > 1 {
            let tuple = lhs.trim_end_matches(" = ").to_string();
            for (i, (name, _)) in results.iter().enumerate() {
                self.line(indent, &format!("{name} = llvm.extractvalue {tuple}[{i}] : {ret}"), None);
            }
        }
    }

    /// Hand-written inline assembly. Address operands are computed as the
    /// mbarrier operations' are ([`Self::address`]), and cut to 32 bits for
    /// the constraint `r` (a shared-memory address). With a guard, the
    /// outputs not tied to an input are 0 where it is false (in PTX the
    /// guarded instruction does not write them): the statement writes them
    /// to temporaries, which a select picks or replaces with 0. A memory
    /// clobber also gives it side effects: MLIR does not read constraint
    /// strings, and models an `llvm.inline_asm`'s effects only through
    /// `has_side_effects`, so it would otherwise delete a statement whose
    /// outputs are unused (a store) before LLVM sees the clobber.
    fn inline_ptx(&mut self, indent: usize, inst: &Inst, op: &Op) {
        let Op::InlinePtx { template, inputs, outputs, constraints, predicated, side_effects, .. } = op else {
            unreachable!()
        };
        let entries = asm_constraints(constraints);
        let mut args = Vec::new();
        for (k, input) in inputs.iter().enumerate() {
            args.push(match *input {
                AsmInput::Value(v) => (self.v(v), self.ty(v)),
                AsmInput::Address { buffer, index, elem } => {
                    let address = self.address(indent, buffer, elem, index);
                    if matches!(entries.get(outputs.len() + k), Some(AsmConstraint::Input(code)) if code == "r") {
                        // A shared-memory address in 32 bits (the validator
                        // checks the space), as CUTLASS's
                        // `cast_smem_ptr_to_uint` gives it.
                        let narrow = self.fresh("saddr");
                        self.line(indent, &format!("{narrow} = arith.trunci {address} : i64 to i32"), None);
                        (narrow, "i32")
                    } else {
                        (address, "i64")
                    }
                }
            });
        }
        let tied: Vec<usize> = entries
            .into_iter()
            .filter_map(|e| match e {
                AsmConstraint::Tied(o) => Some(o),
                _ => None,
            })
            .collect();
        let guard = predicated.then(|| args.last().expect("a predicated statement has a guard").0.clone());
        let mut results: Vec<(String, DType)> = self.asm_operands(outputs);
        for (k, result) in results.iter_mut().enumerate() {
            if guard.is_some() && !tied.contains(&k) {
                result.0 = self.fresh("unguarded");
            }
        }
        let side_effects = *side_effects || asm_clobbers_memory(constraints);
        self.asm_line(indent, inst, template, constraints, &results, &args, side_effects);
        if let Some(guard) = guard {
            for (k, &o) in outputs.iter().enumerate().filter(|(k, _)| !tied.contains(k)) {
                let ty = self.ir.dtype(o);
                let zero = self.fresh("z");
                self.line(indent, &format!("{zero} = arith.constant {}", literal(Literal::zero(ty))), None);
                let text = format!("{} = arith.select {guard}, {}, {zero} : {}", self.v(o), results[k].0, ty.mlir());
                self.line(indent, &text, None);
            }
        }
    }

    /// A hand-written MLIR operation, with its operands' and results' names
    /// filled in. A pointer operand is made from its address
    /// ([`Self::address`]) in the buffer's LLVM address space: 1 (global), 3
    /// (shared), or generic (0) for a local buffer, whose address is generic.
    fn mlir_op(&mut self, indent: usize, inst: &Inst, text: &str, operands: &[AsmInput], results: &[ValueId]) {
        let mut names = Vec::new();
        for operand in operands {
            names.push(match *operand {
                AsmInput::Value(v) => self.v(v),
                AsmInput::Address { buffer, index, elem } => {
                    let address = self.address(indent, buffer, elem, index);
                    let ty = match self.ir.buffer(buffer).space {
                        MemSpace::Global => "!llvm.ptr<1>",
                        MemSpace::Shared => "!llvm.ptr<3>",
                        MemSpace::Local | MemSpace::Constant => "!llvm.ptr",
                        MemSpace::Raw(_) => unreachable!("an operand address of a raw pointer is passed as an integer"),
                    };
                    let ptr = self.fresh("ptr");
                    self.line(indent, &format!("{ptr} = llvm.inttoptr {address} : i64 to {ty}"), None);
                    ptr
                }
            });
        }
        let body = crate::arch::inline::substitute_mlir_operands(text, &names);
        let line = if results.is_empty() { body } else { format!("{} = {body}", self.results(results)) };
        self.line(indent, &line, Some(inst));
    }

    /// An integer instruction: one pure PTX instruction (LLVM has no
    /// intrinsic for most of them, and CuTe DSL emits the same inline PTX).
    /// `lop3`'s lookup table is its immediate, and `bfe` and `bfi` take their
    /// start and length through [`Self::bit_field_operands`].
    fn int_inst(&mut self, indent: usize, inst: &Inst, result: ValueId, func: IntFn, args: &[ValueId]) {
        let ty = self.ir.dtype(args[0]);
        let operands: Vec<String> = (0..=args.len()).map(|k| format!("${k}")).collect();
        let immediate = match func {
            IntFn::Lop3 { lut } => format!(", {lut:#04x}"),
            _ => String::new(),
        };
        let template = format!("{} {}{immediate};", int_inst_name(func, ty), operands.join(", "));
        let mut inputs = self.asm_operands(args);
        if matches!(func, IntFn::Bfe | IntFn::Bfi) {
            self.bit_field_operands(indent, func, ty, &mut inputs);
        }
        let results = self.asm_operands(&[result]);
        self.ptx_inst(indent, inst, &template, &results, &inputs, false);
    }

    /// `bfe`'s and `bfi`'s start and length (their last two operands),
    /// brought into the range where the GPU gives the PTX ISA's result. The
    /// ISA takes only their low 8 bits, but an SM89 GPU's `bfe` and `bfi` on
    /// 64-bit values use all 32 (and so the low byte is masked here), and
    /// `ptxas` folds a constant `bfe.s32` starting past bit 31 to -1 and a
    /// constant `bfi.b32` field longer than 31 bits as if its length were
    /// taken mod 32, where the hardware follows the ISA. The start of a
    /// `bfe.s32` is clamped to 31, which extracts the same field (the sign
    /// bit, replicated), and a `bfi.b32` field is cut to the bits below the
    /// msb, which are all it can change. This is plain arithmetic, which LLVM
    /// folds away when the start and length are constants. The other
    /// operands (both of `bfe.u32`, the length of `bfe.s32`) go to `ptxas` as
    /// they are: the GPU takes their low byte at runtime, and so does `ptxas`
    /// when it folds a constant, even one whose low byte is 0 (0x100: a start
    /// of 0 or an empty field), as `tests/gpu.rs` checks.
    fn bit_field_operands(&mut self, indent: usize, func: IntFn, ty: DType, inputs: &mut [(String, DType)]) {
        // bfe.u32 gives the ISA's result for every start and length.
        if func == IntFn::Bfe && ty == DType::U32 {
            return;
        }
        let n = inputs.len();
        let (start, length) = (inputs[n - 2].0.clone(), inputs[n - 1].0.clone());
        let constant = |e: &mut Self, value: u32| {
            let name = e.fresh("c");
            e.line(indent, &format!("{name} = arith.constant {value} : i32"), None);
            name
        };
        let binary = |e: &mut Self, op: &str, lhs: &str, rhs: &str| {
            let name = e.fresh("bf");
            e.line(indent, &format!("{name} = arith.{op} {lhs}, {rhs} : i32"), None);
            name
        };
        let byte = constant(self, 0xff);
        let low_start = binary(self, "andi", &start, &byte);
        if ty.bits() == 64 {
            inputs[n - 2].0 = low_start;
            inputs[n - 1].0 = binary(self, "andi", &length, &byte);
        } else if func == IntFn::Bfe {
            let msb = constant(self, 31);
            inputs[n - 2].0 = binary(self, "minui", &low_start, &msb);
        } else {
            // 32 - start wraps for starts past 32, which leave the length as
            // it is (and insert nothing).
            let low_length = binary(self, "andi", &length, &byte);
            let width = constant(self, 32);
            let room = binary(self, "subi", &width, &low_start);
            inputs[n - 2].0 = low_start;
            inputs[n - 1].0 = binary(self, "minui", &low_length, &room);
        }
    }

    /// A 16-bit float as an i16 for inline PTX (`h` registers hold bits).
    fn as_bits16(&mut self, indent: usize, v: ValueId) -> String {
        let bits = self.fresh("h");
        let text = format!("{bits} = arith.bitcast {} : {} to i16", self.v(v), self.ty(v));
        self.line(indent, &text, None);
        bits
    }

    /// A half-precision instruction: one pure PTX instruction, as CuTe DSL
    /// emits it (inline PTX, or `nvvm.fma`, which becomes the same
    /// instruction). The packed forms take and give Int32s (`r` registers);
    /// the scalar ones 16-bit floats, passed as their bits (`h`). With a
    /// predicate the instruction is `@p`-guarded, and its result is tied to
    /// the first operand, which is what a thread whose predicate is false
    /// gets.
    #[allow(clippy::too_many_arguments)]
    fn half_inst(
        &mut self,
        indent: usize,
        inst: &Inst,
        result: ValueId,
        func: HalfFn,
        ty: HalfType,
        args: &[ValueId],
        mods: HalfModifiers,
        predicate: Option<ValueId>,
    ) {
        let name = half_inst_name(func, ty, mods);
        let packed = ty.is_packed();
        let mut inputs: Vec<(String, DType)> = Vec::new();
        for &a in args {
            inputs.push(if packed { (self.v(a), DType::I32) } else { (self.as_bits16(indent, a), DType::I16) });
        }
        let reg_type = if packed { DType::I32 } else { DType::I16 };
        let out = if packed { self.v(result) } else { self.fresh("h") };
        let operands: Vec<String> = (1..=args.len()).map(|k| format!("${k}")).collect();
        match predicate {
            None => {
                let template = format!("{name} $0, {};", operands.join(", "));
                self.ptx_inst(indent, inst, &template, &[(out.clone(), reg_type)], &inputs, false);
            }
            Some(p) => {
                let template = format!("@${} {name} $0, {};", args.len() + 1, operands.join(", "));
                let reg = constraint(reg_type);
                let mut constraints = vec![format!("={reg}")];
                constraints.extend(inputs.iter().map(|_| reg.to_string()));
                constraints.extend(["b".to_string(), "0".to_string()]);
                let first = inputs[0].0.clone();
                inputs.push((self.v(p), DType::Bool));
                inputs.push((first, reg_type));
                let names: Vec<&str> = inputs.iter().map(|(n, _)| n.as_str()).collect();
                let types: Vec<&str> = inputs.iter().map(|(_, t)| t.mlir()).collect();
                let text = format!(
                    "{out} = llvm.inline_asm \"{}\", \"{}\" {} : ({}) -> {}",
                    escape(&template),
                    constraints.join(","),
                    names.join(", "),
                    types.join(", "),
                    reg_type.mlir()
                );
                self.line(indent, &text, Some(inst));
            }
        }
        if !packed {
            let text = format!("{} = arith.bitcast {out} : i16 to {}", self.v(result), self.ty(result));
            self.line(indent, &text, None);
        }
    }

    /// A conversion with modifiers: one pure PTX instruction, as CuTe DSL
    /// emits it. `cvt.rn.satfinite.bf16x2.f32` puts its first source in the
    /// upper half, so element 1 goes first; `cvt.rn.bf16.s8` takes a `.b8`
    /// register, which the low byte of a 16-bit one is moved into (`ptxas`
    /// refuses a wider source register here); the s2f6x2 conversion takes its
    /// scale factor, `0x8585`, in a register.
    fn cvt_inst(&mut self, indent: usize, inst: &Inst, result: ValueId, func: CvtFn, args: &[ValueId]) {
        let ptx = func.ptx();
        let results = self.asm_operands(&[result]);
        match func {
            CvtFn::F32x2ToBf16x2 => {
                let inputs = self.asm_operands(&[args[1], args[0]]);
                self.ptx_inst(indent, inst, &format!("{ptx} $0, $1, $2;"), &results, &inputs, false);
            }
            CvtFn::F32ToTf32 => {
                let inputs = self.asm_operands(args);
                self.ptx_inst(indent, inst, &format!("{ptx} $0, $1;"), &results, &inputs, false);
            }
            CvtFn::S8ToBf16 => {
                let wide = self.fresh("s");
                self.line(indent, &format!("{wide} = arith.extsi {} : i8 to i16", self.v(args[0])), None);
                let bits = self.fresh("h");
                let inputs = [(wide, DType::I16)];
                let template = format!("{{\n\t.reg .b8 byte;\n\tmov.b16 {{byte, _}}, $1;\n\t{ptx} $0, byte;\n\t}}");
                self.ptx_inst(indent, inst, &template, &[(bits.clone(), DType::I16)], &inputs, false);
                self.line(indent, &format!("{} = arith.bitcast {bits} : i16 to bf16", self.v(result)), None);
            }
            CvtFn::S2f6x2ToBf16x2 { scale } => {
                let template =
                    format!("{{\n\t.reg .b16 sf;\n\tmov.b16 sf, 0x{scale:02x}{scale:02x};\n\t{ptx} $0, $1, sf;\n\t}}");
                let inputs = self.asm_operands(args);
                self.ptx_inst(indent, inst, &template, &results, &inputs, false);
            }
            // Two Float32 sources: PTX converts its first source into the
            // high byte, element 1.
            CvtFn::F32x2ToF8x2 { .. } | CvtFn::F32x2ToUe8m0x2 { .. } | CvtFn::F32x2ToUe5m3x2 => {
                let inputs = self.asm_operands(&[args[1], args[0]]);
                self.ptx_inst(indent, inst, &format!("{ptx} $0, $1, $2;"), &results, &inputs, false);
            }
            // An FP4 pair is a .b8 register: the low byte of the Int16.
            CvtFn::F32x2ToMx { to: MxKind::E2M1, .. } => {
                let inputs = self.asm_operands(&[args[1], args[0]]);
                let template = format!(
                    "{{\n\t.reg .b8 lo;\n\t{ptx} lo, $1, $2;\n\tmov.b16 $0, {{lo, lo}};\n\tand.b16 $0, $0, 0x00ff;\n\t}}"
                );
                self.ptx_inst(indent, inst, &template, &results, &inputs, false);
            }
            CvtFn::F32x2ToMx { .. } => {
                let inputs = self.asm_operands(&[args[1], args[0]]);
                self.ptx_inst(indent, inst, &format!("{ptx} $0, $1, $2;"), &results, &inputs, false);
            }
            CvtFn::MxToF16x2 { from: MxKind::E2M1, .. } => {
                let inputs = self.asm_operands(args);
                let template = format!("{{\n\t.reg .b8 b;\n\tmov.b16 {{b, _}}, $1;\n\t{ptx} $0, b;\n\t}}");
                self.ptx_inst(indent, inst, &template, &results, &inputs, false);
            }
            // Packed arithmetic: the pairs are 64-bit (32-bit for half pairs)
            // registers, operated on directly; no side effects, so LLVM may CSE.
            CvtFn::Packed { .. } => {
                let inputs = self.asm_operands(args);
                let operands: Vec<String> = (1..=args.len()).map(|k| format!("${k}")).collect();
                self.ptx_inst(indent, inst, &format!("{ptx} $0, {};", operands.join(", ")), &results, &inputs, false);
            }
            CvtFn::F16x2ToF8x2 { .. }
            | CvtFn::F8x2ToF16x2 { .. }
            | CvtFn::Bf16x2ToUe8m0x2 { .. }
            | CvtFn::Ue8m0x2ToBf16x2
            | CvtFn::MxToF16x2 { .. } => {
                let inputs = self.asm_operands(args);
                self.ptx_inst(indent, inst, &format!("{ptx} $0, $1;"), &results, &inputs, false);
            }
        }
    }

    fn inst(&mut self, inst: &Inst, indent: usize) {
        let outer = self.current.replace(inst.loc);
        let outer_scope = std::mem::replace(&mut self.current_scope, inst.scope);
        let (before, after) = if self.tcgen05_fences { tcgen05::thread_sync_fences(&inst.op) } else { (false, false) };
        if before {
            self.line(indent, "nvvm.tcgen05.fence <before>", None);
        }
        self.inst_body(inst, indent);
        if after {
            self.line(indent, "nvvm.tcgen05.fence <after>", None);
        }
        self.current = outer;
        self.current_scope = outer_scope;
    }

    fn inst_body(&mut self, inst: &Inst, indent: usize) {
        let i = Some(inst);
        match &inst.op {
            // Int4 arithmetic runs on the sign-extended i8 container and
            // wraps back to 4 bits.
            Op::Unary { result, .. } | Op::Binary { result, .. } if self.ir.dtype(*result) == DType::I4 => {
                match &inst.op {
                    Op::Unary { result, op, x } => self.unary(indent, inst, *result, *op, *x),
                    Op::Binary { result, op, lhs, rhs } => self.binary(indent, inst, *result, *op, *lhs, *rhs),
                    _ => unreachable!(),
                }
                let raw = self.v(*result);
                self.wrap_int4(indent, &raw, *result);
            }
            Op::Const { result, value } => {
                let text = format!("{} = arith.constant {}", self.v(*result), literal(*value));
                self.line(indent, &text, i);
            }
            Op::Unary { result, op, x } => self.unary(indent, inst, *result, *op, *x),
            Op::Binary { result, op, lhs, rhs } => self.binary(indent, inst, *result, *op, *lhs, *rhs),
            // `fma.rn.bf16` (sm_80) rounds once; through f32 the result would
            // be rounded twice.
            Op::Fma { result, a, b, c } if self.ir.dtype(*a) == DType::BF16 => {
                let rn = HalfModifiers { rn: true, ..HalfModifiers::default() };
                self.half_inst(indent, inst, *result, HalfFn::Fma, HalfType::Bf16, &[*a, *b, *c], rn, None);
            }
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
            Op::FloatInst {
                result,
                func: FloatFn::Rcp,
                args,
                mode: FloatMode::Approx,
                ftz: true,
                nan: false,
                xorsign_abs: false,
            } if self.ir.dtype(args[0]) == DType::F32 => {
                let text = format!("{} = nvvm.rcp.approx.ftz.f {} : f32", self.v(*result), self.v(args[0]));
                self.line(indent, &text, i);
            }
            // The NVVM dialect has no op for the other modifier combinations,
            // so they are inline PTX; without side effects, LLVM may still
            // CSE them.
            Op::FloatInst { result, func, args, mode, ftz, nan, xorsign_abs } => {
                let ty = self.ir.dtype(args[0]);
                let name = float_inst_name(*func, *mode, *ftz, *nan, *xorsign_abs, ty);
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
            Op::IntInst { result, func, args } => self.int_inst(indent, inst, *result, *func, args),
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
            // Types stored as the same MLIR type (TFloat32 and Float32, an
            // FP8 format and Int8) need no instruction.
            Op::Bitcast { result, x }
                if self.ty(*result) == self.ty(*x)
                    && [*result, *x].iter().any(|&v| stored_as_other(self.ir.dtype(v))) =>
            {
                let source = *self.aliases.get(x).unwrap_or(x);
                self.aliases.insert(*result, source);
            }
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
            Op::Load { results, buffer, index, attrs, pred: Some(p), pass_thru } => {
                let op = Op::Load {
                    results: results.clone(),
                    buffer: *buffer,
                    index: *index,
                    attrs: *attrs,
                    pred: None,
                    pass_thru: vec![],
                };
                let plain = Inst { op, loc: inst.loc, scope: inst.scope };
                self.masked(indent, inst, &plain, *p, results, pass_thru);
            }
            Op::Store { values, buffer, index, attrs, pred: Some(p) } => {
                let op =
                    Op::Store { values: values.clone(), buffer: *buffer, index: *index, attrs: *attrs, pred: None };
                let plain = Inst { op, loc: inst.loc, scope: inst.scope };
                self.masked(indent, inst, &plain, *p, &[], &[]);
            }
            Op::Load { results, buffer, index, attrs, .. } if attrs.peer.is_some() => {
                self.dsmem_access(indent, inst, results, *buffer, *index, attrs, true)
            }
            Op::Store { values, buffer, index, attrs, .. } if attrs.peer.is_some() => {
                self.dsmem_access(indent, inst, values, *buffer, *index, attrs, false)
            }
            Op::Load { results, buffer, index, attrs, .. } if !attrs.is_plain() => {
                self.load_with_attrs(indent, inst, results, *buffer, *index, attrs)
            }
            Op::Store { values, buffer, index, attrs, .. } if !attrs.is_plain() => {
                self.store_with_attrs(indent, inst, values, *buffer, *index, attrs)
            }
            Op::Load { results, buffer, index, .. } if self.ir.buffer(*buffer).space.raw().is_some() => {
                self.raw_access(indent, inst, results, *buffer, *index, true)
            }
            Op::Store { values, buffer, index, .. } if self.ir.buffer(*buffer).space.raw().is_some() => {
                self.raw_access(indent, inst, values, *buffer, *index, false)
            }
            Op::Load { results, buffer, index, .. } => {
                let elem = self.ir.dtype(results[0]);
                let ix = self.index(indent, *index);
                let (b, mt) = self.access(*buffer, elem);
                if results.len() == 1 {
                    let text = format!("{} = memref.load {b}[{ix}] : {mt}", self.v(results[0]));
                    self.line(indent, &text, i);
                } else if elem == DType::Bool {
                    // A Boolean is a byte in memory but an `i1` in MLIR,
                    // whose vectors LLVM stores bit-packed: one byte each.
                    for (k, r) in results.iter().enumerate() {
                        let at = self.offset_index(indent, &ix, k);
                        let text = format!("{} = memref.load {b}[{at}] : {mt}", self.v(*r));
                        self.line(indent, &text, i);
                    }
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
            Op::Store { values, buffer, index, .. } => {
                let elem = self.ir.dtype(values[0]);
                let ix = self.index(indent, *index);
                let (b, mt) = self.access(*buffer, elem);
                if values.len() == 1 {
                    let text = format!("memref.store {}, {b}[{ix}] : {mt}", self.v(values[0]));
                    self.line(indent, &text, i);
                } else if elem == DType::Bool {
                    // One byte per Boolean, as for loads.
                    for (k, v) in values.iter().enumerate() {
                        let at = self.offset_index(indent, &ix, k);
                        let text = format!("memref.store {}, {b}[{at}] : {mt}", self.v(*v));
                        self.line(indent, &text, i);
                    }
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
            Op::LdMatrix { results, buffer, index, num, trans, form } if *form != LdMatrixForm::M8n8B16 => {
                // The 8-bit shapes: the convergent nvvm.ldmatrix with shape
                // and element-type attributes (LLVM 22 on), or inline PTX.
                let sum = self.address(indent, *buffer, DType::I8, *index);
                let regs: Vec<(String, DType)> = results.iter().map(|&r| (self.v(r), DType::I32)).collect();
                if self.opts.mma_inline_ptx {
                    let outs: Vec<String> = (0..regs.len()).map(|k| format!("${k}")).collect();
                    let template = format!("{} {{{}}}, [${}];", form.ptx(*num, *trans), outs.join(", "), regs.len());
                    self.ptx_inst(indent, inst, &template, &regs, &[(sum, DType::I64)], true);
                } else {
                    let ptr = self.fresh("ptr");
                    self.line(indent, &format!("{ptr} = llvm.inttoptr {sum} : i64 to !llvm.ptr<3>"), None);
                    let (m, n) = if matches!(form, LdMatrixForm::M8n16B8 { .. }) { (8, 16) } else { (16, 16) };
                    let layout = if *trans { "col" } else { "row" };
                    let ret = if regs.len() == 1 {
                        "i32".to_string()
                    } else {
                        format!("!llvm.struct<({})>", vec!["i32"; regs.len()].join(", "))
                    };
                    let out = if regs.len() == 1 { regs[0].0.clone() } else { self.fresh("ldm") };
                    let text = format!(
                        "{out} = nvvm.ldmatrix {ptr} {{num = {num} : i32, layout = #nvvm.mma_layout<{layout}>, shape = \
                         #nvvm.ld_st_matrix_shape<m = {m}, n = {n}>, eltType = #nvvm.ld_st_matrix_elt_type<{}>}} : \
                         (!llvm.ptr<3>) -> {ret}",
                        form.format()
                    );
                    self.line(indent, &text, i);
                    if regs.len() > 1 {
                        for (k, (name, _)) in regs.iter().enumerate() {
                            self.line(indent, &format!("{name} = llvm.extractvalue {out}[{k}] : {ret}"), None);
                        }
                    }
                }
            }
            Op::LdMatrix { results, buffer, index, num, trans, .. } => {
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
            Op::StMatrix { values, buffer, index, num, trans, form } => {
                let elem = stmatrix_elem(self.ir, *buffer, *form);
                let sum = self.address(indent, *buffer, elem, *index);
                let regs: Vec<String> = values.iter().map(|&v| self.v(v)).collect();
                if *form == StMatrixForm::M16n8B8 && self.opts.mma_inline_ptx {
                    let ins: Vec<String> = (1..=regs.len()).map(|k| format!("${k}")).collect();
                    let template = format!("{} [$0], {{{}}};", form.ptx(*num, *trans), ins.join(", "));
                    let mut inputs = vec![(sum, DType::I64)];
                    inputs.extend(regs.into_iter().map(|r| (r, DType::I32)));
                    self.ptx_inst(indent, inst, &template, &[], &inputs, true);
                } else {
                    let ptr = self.fresh("ptr");
                    self.line(indent, &format!("{ptr} = llvm.inttoptr {sum} : i64 to !llvm.ptr<3>"), None);
                    let layout = if *trans { "col" } else { "row" };
                    // LLVM 21 knows only .m8n8.b16, without shape or element-type attributes,
                    // and lowers it to inline assembly (not convergent); LLVM 22 on take the
                    // attributes and lower to the convergent intrinsic.
                    let attrs = if self.opts.llvm_major.is_some_and(|v| v < 22) {
                        format!("layout = #nvvm.mma_layout<{layout}>")
                    } else {
                        let (m, n) = if *form == StMatrixForm::M8n8B16 { (8, 8) } else { (16, 8) };
                        format!(
                            "layout = #nvvm.mma_layout<{layout}>, shape = #nvvm.ld_st_matrix_shape<m = {m}, n = {n}>, \
                             eltType = #nvvm.ld_st_matrix_elt_type<{}>",
                            form.format()
                        )
                    };
                    let types = vec!["i32"; regs.len()].join(", ");
                    let text = format!("nvvm.stmatrix {ptr}, {} {{{attrs}}} : !llvm.ptr<3>, {types}", regs.join(", "));
                    self.line(indent, &text, i);
                }
            }
            Op::SetMaxNReg { increase, count } => {
                let kind = if *increase { "increase" } else { "decrease" };
                self.line(indent, &format!("nvvm.setmaxregister {kind} {count}"), i);
            }
            Op::Mma { kind, d, a, b, c, e } if kind.is_fp8() || kind.is_sparse() => {
                self.mma_nvvm(indent, inst, *kind, d, a, b, c, *e)
            }
            Op::Mma { kind, d, a, b, c, .. } if kind.dtypes().0 == DType::BF16 => {
                self.mma_bf16(indent, inst, *kind, d, a, b, c)
            }
            Op::Mma { kind, d, a, b, c, .. } => {
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
                // nothing but the PTX version it needs). The syncs call the
                // intrinsics `nvvm.barrier` lowers to (in LLVM 21, 22 and 23)
                // directly: LLVM 22's `nvvm.barrier` syntax ends with an
                // optional attribute that swallows a following `loc(...)`.
                let (id, n) = (self.v(*id), threads.map(|n| self.v(n)));
                let text = match (kind, n) {
                    (BarrierKind::Sync, None) => {
                        format!("llvm.call_intrinsic \"llvm.nvvm.barrier.cta.sync.aligned.all\"({id}) : (i32) -> ()")
                    }
                    (BarrierKind::Sync, Some(n)) => format!(
                        "llvm.call_intrinsic \"llvm.nvvm.barrier.cta.sync.aligned.count\"({id}, {n}) : (i32, i32) -> ()"
                    ),
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
            Op::Shuffle { result, kind, value, offset, mask, clamp } => {
                let ty = self.ir.dtype(*value);
                let float = ty.is_float();
                let word = if float {
                    let w = self.fresh("w");
                    self.line(indent, &format!("{w} = arith.bitcast {} : f32 to i32", self.v(*value)), None);
                    w
                } else {
                    self.v(*value)
                };
                // The segment mask and clamp, packed (CuTe DSL's `mask_and_clamp`).
                let clamp = self.v(*clamp);
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
            Op::Match { result, kind: MatchKind::Any, value, mask } => {
                let text = format!(
                    "{} = nvvm.match.sync any {}, {} : {} -> i32",
                    self.v(*result),
                    self.v(*mask),
                    self.v(*value),
                    self.ty(*value)
                );
                self.line(indent, &text, i);
            }
            // `match.all` also gives a predicate: whether every member holds
            // the same value, which is the mask being nonzero.
            Op::Match { result, kind: MatchKind::All, value, mask } => {
                let pair = self.fresh("match");
                let ret = "!llvm.struct<(i32, i1)>";
                let text = format!(
                    "{pair} = nvvm.match.sync all {}, {} : {} -> {ret}",
                    self.v(*mask),
                    self.v(*value),
                    self.ty(*value)
                );
                self.line(indent, &text, i);
                self.line(indent, &format!("{} = llvm.extractvalue {pair}[0] : {ret}", self.v(*result)), None);
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
            Op::Atomic { result, kind, ty, buffer, index, value, compare, sem, scope, peer } => {
                let at = (*buffer, *index, *peer);
                self.atomic(indent, inst, Some(*result), *kind, *ty, at, *value, *compare, *sem, *scope)
            }
            Op::Red { kind, ty, buffer, index, value, sem, scope, peer } => {
                let at = (*buffer, *index, *peer);
                self.atomic(indent, inst, None, *kind, *ty, at, *value, None, *sem, *scope)
            }
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
            Op::MbarrierWait { state: true, .. } | Op::MbarrierWait { time_limit: Some(_), .. } => {
                self.prims_op(indent, inst)
            }
            Op::MbarrierWait { result, buffer, index, parity, kind, .. } => {
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
            // The convergent NVVM operations, never inline PTX, which LLVM may
            // duplicate into divergent paths.
            Op::ClusterArrive { relaxed, aligned } => {
                let op = if *relaxed { "nvvm.cluster.arrive.relaxed" } else { "nvvm.cluster.arrive" };
                self.line(indent, &format!("{op}{}", if *aligned { " {aligned}" } else { "" }), i)
            }
            Op::ClusterWait { aligned } => {
                self.line(indent, &format!("nvvm.cluster.wait{}", if *aligned { " {aligned}" } else { "" }), i)
            }
            Op::FenceMbarrierInit => self.line(
                indent,
                "llvm.inline_asm has_side_effects \"fence.mbarrier_init.release.cluster;\", \"\" : () -> ()",
                i,
            ),
            // As CuTe DSL emits them: inline PTX with side effects and a memory clobber (which
            // also sidesteps the NVVM spelling, different in LLVM 21 and 23).
            Op::GridDepControl { launch_dependents } => {
                let what = if *launch_dependents { "launch_dependents" } else { "wait" };
                self.line(
                    indent,
                    &format!("llvm.inline_asm has_side_effects \"griddepcontrol.{what};\", \"~{{memory}}\" : () -> ()"),
                    i,
                )
            }
            // Bulk copies with a cache hint, a multicast mask or a peer CTA: inline PTX with a memory clobber.
            Op::CpAsyncBulk {
                src,
                src_index,
                dst,
                dst_index,
                elem,
                bytes,
                direction,
                completion,
                cache_policy,
                mcast_mask,
                peer,
                byte_mask,
            } if cache_policy.is_some() || mcast_mask.is_some() || peer.is_some() || byte_mask.is_some() => {
                let d = self.address(indent, *dst, *elem, *dst_index);
                let s = self.address(indent, *src, *elem, *src_index);
                let mut inputs = vec![(d, DType::I64), (s, DType::I64), (self.v(*bytes), DType::I32)];
                let mbar = match completion {
                    BulkCompletion::Mbarrier { buffer, index } => {
                        Some(self.address(indent, *buffer, DType::I64, *index))
                    }
                    BulkCompletion::BulkGroup => None,
                };
                let template = match direction {
                    BulkDirection::S2S => {
                        inputs.push((mbar.expect("validated: an mbarrier"), DType::I64));
                        inputs.push((self.v(peer.expect("validated: a peer")), DType::I32));
                        "{\n\t.reg .b64 rdst, rbar;\n\tmapa.shared::cluster.u64 rdst, $0, $4;\n\t\
                         mapa.shared::cluster.u64 rbar, $3, $4;\n\t\
                         cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [rdst], [$1], $2, \
                         [rbar];\n\t}"
                            .to_string()
                    }
                    BulkDirection::G2S => {
                        inputs.push((mbar.expect("validated: an mbarrier"), DType::I64));
                        let mut quals = String::new();
                        let mut tail = String::new();
                        if let Some(m) = mcast_mask {
                            quals.push_str(".multicast::cluster");
                            tail.push_str(&format!(", ${}", inputs.len()));
                            inputs.push((self.v(*m), DType::I16));
                        }
                        if let Some(p) = cache_policy {
                            quals.push_str(".L2::cache_hint");
                            tail.push_str(&format!(", ${}", inputs.len()));
                            inputs.push((self.v(*p), DType::I64));
                        }
                        format!(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes{quals} [$0], [$1], $2, \
                             [$3]{tail};"
                        )
                    }
                    BulkDirection::S2G => {
                        // m7-tma-sm100-modes: `.cp_mask` and its i16 mask come after the cache hint.
                        let mut quals = String::new();
                        let mut tail = String::new();
                        if let Some(p) = cache_policy {
                            quals.push_str(".L2::cache_hint");
                            tail.push_str(&format!(", ${}", inputs.len()));
                            inputs.push((self.v(*p), DType::I64));
                        }
                        if let Some(m) = byte_mask {
                            quals.push_str(".cp_mask");
                            tail.push_str(&format!(", ${}", inputs.len()));
                            inputs.push((self.v(*m), DType::I16));
                        }
                        format!("cp.async.bulk.global.shared::cta.bulk_group{quals} [$0], [$1], $2{tail};")
                    }
                };
                self.asm_memory(indent, inst, &template, &inputs);
            }
            Op::CpAsyncBulk { src, src_index, dst, dst_index, elem, bytes, completion, .. } => {
                let d = self.address(indent, *dst, *elem, *dst_index);
                let s = self.address(indent, *src, *elem, *src_index);
                let text = match completion {
                    BulkCompletion::Mbarrier { buffer, index } => {
                        let m = self.address(indent, *buffer, DType::I64, *index);
                        format!(
                            "llvm.inline_asm has_side_effects \"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];\", \"l,l,r,l\" {d}, {s}, {}, {m} : (i64, i64, i32, i64) -> ()",
                            self.v(*bytes)
                        )
                    }
                    BulkCompletion::BulkGroup => format!(
                        "llvm.inline_asm has_side_effects \"cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;\", \"l,l,r\" {d}, {s}, {} : (i64, i64, i32) -> ()",
                        self.v(*bytes)
                    ),
                };
                self.line(indent, &text, i);
            }
            // Inline PTX, as the nvvm op does not translate in LLVM 21 and is spelt differently
            // in LLVM 23: 32-bit shared addresses for the destination and the mbarrier, the
            // descriptor's generic address, one 32-bit register per coordinate, and a memory
            // clobber. One thread issues it, so inline assembly is safe.
            Op::TmaLoad {
                dst,
                dst_index,
                elem,
                desc,
                coords,
                mbar,
                mbar_index,
                cache_policy,
                mcast_mask,
                mode,
                overrides,
                two_cta,
                ..
            } => {
                let d = self.address(indent, *dst, *elem, *dst_index);
                let d32 = self.fresh("tmadst");
                self.line(indent, &format!("{d32} = arith.trunci {d} : i64 to i32"), None);
                let m = self.address(indent, *mbar, DType::I64, *mbar_index);
                let m32 = self.fresh("tmabar");
                self.line(indent, &format!("{m32} = arith.trunci {m} : i64 to i32"), None);
                // m10-cta-pair: `.cta_group::2` signals the pair's even CTA, whose mbarrier is
                // this CTA's address with the peer bit cleared (CUTLASS's Sm100MmaPeerBitMask).
                let (m32, pair) = if *two_cta {
                    let mask = self.fresh("peermask");
                    self.line(indent, &format!("{mask} = arith.constant {} : i32", 0xFEFF_FFFFu32 as i32), None);
                    let masked = self.fresh("tmabar");
                    self.line(indent, &format!("{masked} = arith.andi {m32}, {mask} : i32"), None);
                    (masked, ".cta_group::2")
                } else {
                    (m32, "")
                };
                // Gather4 gives five coordinates of a 2-D tensor map.
                let n = if mode.is_rows4() { 2 } else { coords.len() };
                let slots: Vec<String> = (0..coords.len()).map(|k| format!("${}", 3 + k)).collect();
                let mut operands = vec![d32, self.v(*desc), m32];
                let mut types = vec!["i32", "i64", "i32"];
                let mut constraints = vec!["r", "l", "r"];
                for c in coords {
                    operands.push(self.v(*c));
                    types.push("i32");
                    constraints.push("r");
                }
                // The im2col offsets (i16) follow the mbarrier.
                let mut offsets = String::new();
                if let crate::ir::TmaMode::Im2col { offsets: o } = mode {
                    let mut s = Vec::new();
                    for v in o {
                        s.push(format!("${}", operands.len()));
                        operands.push(self.v(*v));
                        types.push("i16");
                        constraints.push("h");
                    }
                    offsets = format!(", {{{}}}", s.join(", "));
                }
                // The override operands, inside the brackets before the
                // coordinates.
                let (over_quals, over_ops, wide_mask) = match overrides {
                    Some(o) => {
                        let (q, ops) =
                            self.tma_override_operands(indent, o, &mut operands, &mut types, &mut constraints);
                        (q, ops, !o.dims.is_empty())
                    }
                    None => (String::new(), String::new(), false),
                };
                let coord_text = format!("{over_ops}{{{}}}", slots.join(", "));
                // `.multicast::cluster` and its i16 mask come before the cache hint (with an
                // attribute override, `.multicast::cluster::32b` and an i32 mask).
                let (mc, mask) = match mcast_mask {
                    Some(m) if wide_mask => {
                        let wide = self.fresh("mask32");
                        self.line(indent, &format!("{wide} = arith.extui {} : i16 to i32", self.v(*m)), None);
                        let slot = operands.len();
                        operands.push(wide);
                        types.push("i32");
                        constraints.push("r");
                        (".multicast::cluster::32b", format!(", ${slot}"))
                    }
                    Some(m) => {
                        let slot = operands.len();
                        operands.push(self.v(*m));
                        types.push("i16");
                        constraints.push("h");
                        (".multicast::cluster", format!(", ${slot}"))
                    }
                    None => ("", String::new()),
                };
                let (hint, policy) = match cache_policy {
                    Some(p) => {
                        let slot = operands.len();
                        operands.push(self.v(*p));
                        types.push("i64");
                        constraints.push("l");
                        (".L2::cache_hint", format!(", ${slot}"))
                    }
                    None => ("", String::new()),
                };
                constraints.push("~{memory}");
                let text = format!(
                    "llvm.inline_asm has_side_effects \"cp.async.bulk.tensor.{n}d{pair}.shared::cluster.global{}.mbarrier::complete_tx::bytes{mc}{hint}{over_quals} [$0], [$1, {coord_text}], [$2]{offsets}{mask}{policy};\", \"{}\" {} : ({}) -> ()",
                    mode.ptx(),
                    constraints.join(","),
                    operands.join(", "),
                    types.join(", ")
                );
                self.line(indent, &text, i);
            }
            // ---- m7-tma-store and m7-cluster-copies: inline PTX, as for the TMA load ----
            Op::TmaStore { src, src_index, elem, desc, coords, reduce, cache_policy, mode, overrides, .. } => {
                let s = self.address(indent, *src, *elem, *src_index);
                let s32 = self.fresh("tmasrc");
                self.line(indent, &format!("{s32} = arith.trunci {s} : i64 to i32"), None);
                let n = if mode.is_rows4() { 2 } else { coords.len() };
                let mut inputs = vec![(s32, DType::I32), (self.v(*desc), DType::I64)];
                let slots: Vec<String> = (0..coords.len()).map(|k| format!("${}", 2 + k)).collect();
                inputs.extend(coords.iter().map(|c| (self.v(*c), DType::I32)));
                let (hint, policy) = match cache_policy {
                    Some(p) => {
                        let slot = inputs.len();
                        inputs.push((self.v(*p), DType::I64));
                        (".L2::cache_hint", format!(", ${slot}"))
                    }
                    None => ("", String::new()),
                };
                let head = match reduce {
                    None => format!("cp.async.bulk.tensor.{n}d.global.shared::cta{}.bulk_group", mode.ptx()),
                    Some(k) => format!(
                        "cp.reduce.async.bulk.tensor.{n}d.global.shared::cta.{}{}.bulk_group",
                        k.ptx_name(),
                        mode.ptx()
                    ),
                };
                // The override operands inside the brackets, before the
                // braced coordinates (`[tmap, gAddr, {sizes}, {strides}, upper, {coords}]`).
                let (over_quals, coord_text) = match overrides {
                    Some(o) => {
                        let mut operands: Vec<String> = inputs.iter().map(|(n, _)| n.clone()).collect();
                        let mut types: Vec<&'static str> = Vec::new();
                        let mut constraints: Vec<&'static str> = Vec::new();
                        let before = operands.len();
                        let (q, ops) =
                            self.tma_override_operands(indent, o, &mut operands, &mut types, &mut constraints);
                        for (k, name) in operands.into_iter().enumerate().skip(before) {
                            let ty = match types[k - before] {
                                "i64" => DType::I64,
                                "i32" => DType::I32,
                                _ => DType::I16,
                            };
                            inputs.push((name, ty));
                        }
                        (q, format!("{ops}{{{}}}", slots.join(", ")))
                    }
                    None => (String::new(), format!("{{{}}}", slots.join(", "))),
                };
                let template = format!("{head}{hint}{over_quals} [$1, {coord_text}], [$0]{policy};");
                self.asm_memory(indent, inst, &template, &inputs);
            }
            Op::TmaPrefetch { desc, coords, mode } => {
                let n = if mode.is_rows4() { 2 } else { coords.len() };
                let mut inputs = vec![(self.v(*desc), DType::I64)];
                inputs.extend(coords.iter().map(|c| (self.v(*c), DType::I32)));
                let slots: Vec<String> = (0..coords.len()).map(|k| format!("${}", 1 + k)).collect();
                // The im2col offsets follow the brackets.
                let mut offsets = String::new();
                if let crate::ir::TmaMode::Im2col { offsets: o } = mode {
                    let s: Vec<String> = (0..o.len()).map(|k| format!("${}", inputs.len() + k)).collect();
                    inputs.extend(o.iter().map(|v| (self.v(*v), DType::I16)));
                    offsets = format!(", {{{}}}", s.join(", "));
                }
                let template = format!(
                    "cp.async.bulk.prefetch.tensor.{n}d.L2.global{} [$0, {{{}}}]{offsets};",
                    mode.ptx(),
                    slots.join(", ")
                );
                self.asm_memory(indent, inst, &template, &inputs);
            }
            Op::PrefetchTensormap { desc } => {
                let inputs = [(self.v(*desc), DType::I64)];
                self.asm_memory(indent, inst, "prefetch.tensormap [$0];", &inputs);
            }
            // Inline PTX, as CuTe DSL emits it (LLVM 21 has no
            // nvvm.tensormap.replace, and no LLVM has tensormap.cp_fenceproxy).
            Op::TensormapReplace { addr, space, field, value } => {
                let (ss, at) = match space {
                    crate::tensor::AddressSpace::Smem => (".shared::cta", DType::I32),
                    crate::tensor::AddressSpace::Gmem => (".global", DType::I64),
                    _ => ("", DType::I64),
                };
                let ty = field.value_type();
                let bits = if ty == DType::I64 { "b64" } else { "b32" };
                let ord = field.ord().map(|o| format!("{o}, ")).unwrap_or_default();
                let template = format!("tensormap.replace.tile.{}{ss}.b1024.{bits} [$0], {ord}$1;", field.ptx_name());
                let inputs = [(self.v(*addr), at), (self.v(*value), ty)];
                self.asm_memory(indent, inst, &template, &inputs);
            }
            Op::FenceTensormapRelease => {
                self.asm_memory(indent, inst, "fence.proxy.tensormap::generic.release.gpu;", &[]);
            }
            Op::FenceTensormapAcquire { addr } => {
                let inputs = [(self.v(*addr), DType::I64)];
                self.asm_memory(indent, inst, "fence.proxy.tensormap::generic.acquire.gpu [$0], 128;", &inputs);
            }
            Op::TensormapCpFenceproxy { dst, src } => {
                let inputs = [(self.v(*dst), DType::I64), (self.v(*src), DType::I32)];
                self.asm_memory(
                    indent,
                    inst,
                    "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [$0], [$1], \
                     128;",
                    &inputs,
                );
            }
            Op::BulkPrefetch { buffer, index, elem, bytes } => {
                let a = self.address(indent, *buffer, *elem, *index);
                let inputs = [(a, DType::I64), (self.v(*bytes), DType::I32)];
                self.asm_memory(indent, inst, "cp.async.bulk.prefetch.L2.global [$0], $1;", &inputs);
            }
            // LLVM 21's NVVM has no st.async: PTX, with both addresses mapped to the peer.
            Op::StAsync { buffer, index, elem, values, mbar, mbar_index, peer } => {
                let a = self.address(indent, *buffer, *elem, *index);
                let m = self.address(indent, *mbar, DType::I64, *mbar_index);
                let n = values.len();
                let mut inputs = vec![(a, DType::I64), (m, DType::I64), (self.v(*peer), DType::I32)];
                inputs.extend(values.iter().map(|v| (self.v(*v), DType::I32)));
                let vals: Vec<String> = (0..n).map(|k| format!("${}", 3 + k)).collect();
                let (vec, data) = match n {
                    1 => (String::new(), vals[0].clone()),
                    _ => (format!(".v{n}"), format!("{{{}}}", vals.join(", "))),
                };
                let template = format!(
                    "{{\n\t.reg .b64 raddr, rbar;\n\tmapa.shared::cluster.u64 raddr, $0, $2;\n\t\
                     mapa.shared::cluster.u64 rbar, $1, $2;\n\t\
                     st.async.shared::cluster.mbarrier::complete_tx::bytes{vec}.b32 [raddr], {data}, [rbar];\n\t}}"
                );
                self.asm_memory(indent, inst, &template, &inputs);
            }
            // LLVM 21's NVVM has no cluster launch control ops (LLVM 23
            // does), so inline PTX; the query keeps the .b128 response inside one statement.
            Op::ClcTryCancel { buffer, index, elem, mbar, mbar_index, multicast } => {
                let a = self.address(indent, *buffer, *elem, *index);
                let m = self.address(indent, *mbar, DType::I64, *mbar_index);
                let template = format!(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{}.b128 [$0], [$1];",
                    if *multicast { ".multicast::cluster::all" } else { "" }
                );
                self.asm_memory(indent, inst, &template, &[(a, DType::I64), (m, DType::I64)]);
            }
            Op::ClcQueryCancel { buffer, index, elem, results } => {
                let a = self.address(indent, *buffer, *elem, *index);
                let template = "{\n\t.reg .b128 rclc;\n\t.reg .pred pclc;\n\tld.shared.b128 rclc, [$4];\n\t\
                                clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 pclc, rclc;\n\t\
                                selp.b32 $3, 1, 0, pclc;\n\t\
                                clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 $0, rclc;\n\t\
                                clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 $1, rclc;\n\t\
                                clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 $2, rclc;\n\t}";
                let outs = self.asm_operands(results);
                self.asm_line(indent, inst, template, "=r,=r,=r,=r,l,~{memory}", &outs, &[(a, "i64")], true);
            }
            Op::CpAsyncBulkCommitGroup => self.line(indent, "nvvm.cp.async.bulk.commit.group", i),
            Op::CpAsyncBulkWaitGroup { pending, read } => self.line(
                indent,
                &format!("nvvm.cp.async.bulk.wait_group {pending}{}", if *read { " {read}" } else { "" }),
                i,
            ),
            // Inline PTX with a memory clobber: the fence is per thread (not
            // collective), and NVVM's `nvvm.fence.proxy` lowers differently
            // from one LLVM release to the next.
            Op::FenceProxy { kind } => self.line(
                indent,
                &format!("llvm.inline_asm has_side_effects \"{};\", \"~{{memory}}\" : () -> ()", kind.ptx()),
                i,
            ),
            Op::Fence { scope, sc } => {
                let s = match scope {
                    FenceScope::Cta => "cta",
                    FenceScope::Cluster => "cluster",
                    FenceScope::Gpu => "gpu",
                    FenceScope::Sys => "sys",
                };
                self.line(
                    indent,
                    &format!(
                        "llvm.inline_asm has_side_effects \"fence.{}.{s};\", \"\" : () -> ()",
                        if *sc { "sc" } else { "acq_rel" }
                    ),
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
            Op::Assert { cond, message } => {
                if self.opts.assertions {
                    // Always a plain location: under a fused one,
                    // `__assertfail` loses the file name, and with line
                    // information LLVM rejects the declaration's second
                    // debug attachment.
                    let _ = writeln!(
                        self.out,
                        "{:indent$}cf.assert {}, \"{}\" loc({})",
                        "",
                        self.v(*cond),
                        escape(message),
                        Self::file_line_col(inst.loc),
                        indent = indent
                    );
                }
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
            // The convergent `llvm.nvvm.redux.sync.*` intrinsics, with the
            // member mask in a register (CuTe DSL's Float32 form is inline PTX
            // with an immediate mask).
            Op::Redux { result, kind, value, mask, abs, nan } => {
                let mut attrs: Vec<&str> = Vec::new();
                if *abs {
                    attrs.push("abs = true");
                }
                if *nan {
                    attrs.push("nan = true");
                }
                let attrs = if attrs.is_empty() { String::new() } else { format!(" {{{}}}", attrs.join(", ")) };
                let t = self.ty(*value);
                let text = format!(
                    "{} = nvvm.redux.sync {} {}, {}{attrs} : {t} -> {t}",
                    self.v(*result),
                    kind.name(),
                    self.v(*value),
                    self.v(*mask)
                );
                self.line(indent, &text, i);
            }
            Op::CarryInst { result, func, args } => self.carry_inst(indent, inst, *result, *func, args),
            Op::MlirOp { text, operands, results, .. } => self.mlir_op(indent, inst, text, operands, results),
            Op::CpAsyncMbarrierArrive { buffer, index, noinc } => {
                // The LLVM intrinsic, which every LLVM from 21 on has (the
                // NVVM op is spelled differently in LLVM 21 and 23); it
                // prints the PTX 7.0 form, `.shared` without `::cta`.
                let sum = self.address(indent, *buffer, DType::I64, *index);
                let ptr = self.fresh("ptr");
                self.line(indent, &format!("{ptr} = llvm.inttoptr {sum} : i64 to !llvm.ptr<3>"), None);
                let suffix = if *noinc { ".noinc" } else { "" };
                let text = format!(
                    "llvm.call_intrinsic \"llvm.nvvm.cp.async.mbarrier.arrive{suffix}.shared\"({ptr}) : (!llvm.ptr<3>) -> ()"
                );
                self.line(indent, &text, i);
            }
            Op::Prefetch { buffer, index, elem, level, evict, uniform, pred } => {
                // Inline PTX on every target: LLVM (21 to 23) selects its
                // prefetch intrinsics only for sm_90 with PTX 8.0, where PTX
                // has had `prefetch` since sm_20 (the eviction priorities
                // since sm_80). A global buffer's address is its global
                // address, as for the atomics; a local one's is generic.
                let addr = self.address(indent, *buffer, *elem, *index);
                let global = self.ir.buffer(*buffer).space == MemSpace::Global;
                let op = match (uniform, evict) {
                    (true, _) => "prefetchu.L1".to_string(),
                    (false, Some(e)) => format!("prefetch.global.L2::{}", e.name()),
                    (false, None) => format!("prefetch{}.{}", if global { ".global" } else { "" }, level.ptx()),
                };
                let mut inputs = vec![(addr, DType::I64)];
                let template = match pred {
                    Some(p) => {
                        inputs.push((self.v(*p), DType::Bool));
                        format!("@$1 {op} [$0];")
                    }
                    None => format!("{op} [$0];"),
                };
                self.ptx_inst(indent, inst, &template, &[], &inputs, true);
            }
            Op::Pack { result, parts } => {
                let names: Vec<String> = parts.iter().map(|&p| self.v(p)).collect();
                let (part, to, out) = (self.ir.dtype(parts[0]), self.ir.dtype(*result), self.v(*result));
                self.pack_bits(indent, &names, part, to, Some(out));
            }
            Op::Unpack { results, x } => {
                let outs: Vec<String> = results.iter().map(|&r| self.v(r)).collect();
                let (from, part, input) = (self.ir.dtype(*x), self.ir.dtype(results[0]), self.v(*x));
                self.unpack_bits(indent, &input, from, part, &outs);
            }
            Op::HalfInst { result, func, ty, args, mods, predicate } => {
                self.half_inst(indent, inst, *result, *func, *ty, args, *mods, *predicate)
            }
            Op::Cvt { result, func, args } => self.cvt_inst(indent, inst, *result, *func, args),
            // The result is the value itself. A power-of-two divisor is also
            // told to LLVM, as `llvm.intr.assume((x & (d - 1)) == 0)`, which
            // costs no instruction (LLVM drops the test) and lets it simplify
            // the arithmetic around it; other facts are the builder's only.
            Op::PtrToInt { result, buffer, index, elem } => {
                self.ptr_to_int(indent, inst, *result, *buffer, *index, *elem)
            }
            Op::CvtAddr { result, x, from, to } => self.cvt_addr(indent, inst, *result, *x, *from, *to),
            Op::Wgmma { .. } | Op::WgmmaFence { .. } | Op::WgmmaCommitGroup | Op::WgmmaWait { .. } => {
                self.wgmma(indent, inst)
            }
            Op::Tcgen05 { .. } => self.tcgen05(indent, inst),
            Op::Tcgen05Ld { .. } | Op::Tcgen05St { .. } => self.tcgen05_ldst(indent, inst),
            Op::Tcgen05Mma { .. } => self.tcgen05_mma(indent, inst),
            Op::MmaBlockScaled { .. } => self.mma_block_scaled(indent, inst),
            Op::MmaSpBlockScaled { .. } => self.mma_sp_block_scaled(indent, inst),
            Op::Tcgen05Commit { .. } => self.tcgen05_commit(indent, inst),
            Op::Tcgen05Cp { .. } => self.tcgen05_cp(indent, inst),
            Op::BarrierRed { .. }
            | Op::MbarrierExt { .. }
            | Op::PrimMarker { .. }
            | Op::Exit
            | Op::StBulk { .. }
            | Op::Tcgen05Shift { .. } => self.prims_op(indent, inst),
            Op::Call { callee, args, results, .. } => self.call(indent, inst, *callee, args, results),
            Op::Multimem { op, buffer, index, elem, values, results } => {
                self.multimem(indent, inst, op, *buffer, *index, *elem, values, results)
            }
            Op::Assume { result, x, divby, .. } => {
                let target = *self.aliases.get(x).unwrap_or(x);
                self.aliases.insert(*result, target);
                if *divby > 1 && divby.is_power_of_two() {
                    // The mask of a divisor beyond the type's positive range
                    // is not a valid constant, so LLVM is told the weaker
                    // fact for the largest power of two within that range.
                    let bits = self.ir.dtype(*x).bits() as u32;
                    let mask = (*divby).min(1u64 << (bits - 1)) - 1;
                    let ty = self.ir.dtype(*x).mlir();
                    let (mask_name, low, zero, test) =
                        (self.fresh("amask"), self.fresh("alow"), self.fresh("azero"), self.fresh("atest"));
                    let i = Some(inst);
                    self.line(indent, &format!("{mask_name} = arith.constant {mask} : {ty}"), i);
                    self.line(indent, &format!("{low} = arith.andi {}, {mask_name} : {ty}", self.v(*x)), i);
                    self.line(indent, &format!("{zero} = arith.constant 0 : {ty}"), i);
                    self.line(indent, &format!("{test} = arith.cmpi eq, {low}, {zero} : {ty}"), i);
                    self.line(indent, &format!("llvm.intr.assume {test} : i1"), i);
                }
            }
        }
    }

    /// A masked access: `plain` (the access without its predicate) inside
    /// `scf.if %pred`, its results renamed inside and yielded, the else
    /// branch yielding `pass_thru` or, without it, a frozen `ub.poison`
    /// (an arbitrary but fixed value, not poison: a later integer division
    /// by it or branch on it is then not undefined behaviour that LLVM may
    /// exploit to delete the surrounding code). NVPTX turns
    /// it into a branch over the access (or a predicated `ld`/`st` where it
    /// if-converts). The plain access is not speculatable: no parameter
    /// carries `dereferenceable`, so LLVM cannot hoist it out of the branch.
    fn masked(
        &mut self,
        indent: usize,
        inst: &Inst,
        plain: &Inst,
        pred: ValueId,
        results: &[ValueId],
        pass_thru: &[ValueId],
    ) {
        if results.is_empty() {
            self.line(indent, &format!("scf.if {} {{", self.v(pred)), None);
            self.inst(plain, indent + 2);
            self.line(indent, "}", Some(inst));
            return;
        }
        let types: Vec<&str> = results.iter().map(|&r| self.ty(r)).collect();
        let types = types.join(", ");
        let head = format!("{} = scf.if {} -> ({types}) {{", self.results(results), self.v(pred));
        self.line(indent, &head, None);
        let saved: Vec<Option<String>> = results.iter().map(|r| self.renames.get(r).cloned()).collect();
        for &r in results {
            let name = self.fresh("masked");
            self.renames.insert(r, name);
        }
        self.inst(plain, indent + 2);
        let inner = self.results(results);
        for (&r, old) in results.iter().zip(saved) {
            match old {
                Some(n) => self.renames.insert(r, n),
                None => self.renames.remove(&r),
            };
        }
        self.line(indent + 2, &format!("scf.yield {inner} : {types}"), None);
        self.line(indent, "} else {", None);
        let names: Vec<String> = if pass_thru.is_empty() {
            results
                .iter()
                .map(|&r| {
                    let (p, u) = (self.fresh("poison"), self.fresh("undef"));
                    let ty = self.ty(r);
                    self.line(indent + 2, &format!("{p} = ub.poison : {ty}"), None);
                    self.line(indent + 2, &format!("{u} = llvm.freeze {p} : {ty}"), None);
                    u
                })
                .collect()
        } else {
            pass_thru.iter().map(|&v| self.v(v)).collect()
        };
        self.line(indent + 2, &format!("scf.yield {} : {types}", names.join(", ")), None);
        self.line(indent, "}", Some(inst));
    }

    /// A load with memory attributes. A scalar relaxed, acquire, seq_cst or
    /// volatile load without hints is an LLVM atomic or volatile load
    /// (`ld.acquire.gpu.global.b32` and so on), whose ordering LLVM respects
    /// and lowers correctly with a sync scope; anything else is a PTX `ld`
    /// (or `fence.sc` and an `ld.acquire` for seq_cst) as inline assembly
    /// with side effects, which LLVM neither removes nor moves across other
    /// memory accesses: one instruction, except that a strong load of more
    /// than four 8- or 16-bit elements takes one per four ([`access_words`]).
    fn load_with_attrs(
        &mut self,
        indent: usize,
        inst: &Inst,
        results: &[ValueId],
        buffer: BufferId,
        index: ValueId,
        attrs: &MemAttrs,
    ) {
        let elem = self.ir.dtype(results[0]);
        let space = self.ir.buffer(buffer).space;
        let addr = self.address(indent, buffer, elem, index);
        self.load_with_attrs_at(indent, inst, results, addr, space, attrs);
    }

    /// [`Self::load_with_attrs`] at the i64 address `addr` of `space`.
    fn load_with_attrs_at(
        &mut self,
        indent: usize,
        inst: &Inst,
        results: &[ValueId],
        addr: String,
        space: MemSpace,
        attrs: &MemAttrs,
    ) {
        let elem = self.ir.dtype(results[0]);
        if llvm_carries(attrs, results.len()) {
            let ptr = self.typed_pointer(indent, &addr, space);
            let t = elem.mlir();
            let text = if attrs.order == MemoryOrder::Volatile {
                format!(
                    "{} = llvm.load volatile {ptr} {{alignment = {} : i64}} : !llvm.ptr<{}> -> {t}",
                    self.v(results[0]),
                    elem.bytes(),
                    llvm_space(space)
                )
            } else {
                format!(
                    "{} = llvm.load {ptr} atomic {}{} {{alignment = {} : i64}} : !llvm.ptr<{}> -> {t}",
                    self.v(results[0]),
                    syncscope(attrs.scope),
                    llvm_ordering(attrs.order),
                    elem.bytes(),
                    llvm_space(space)
                )
            };
            self.line(indent, &text, Some(inst));
            return;
        }
        let (count, words, word) = access_words(results.len(), elem, attrs.is_strong());
        let outputs: Vec<String> = (0..words).map(|k| format!("${k}")).collect();
        let dst = if words == 1 { outputs[0].clone() } else { format!("{{{}}}", outputs.join(", ")) };
        let policy = if attrs.cache_policy.is_some() { format!(", ${}", words + 1) } else { String::new() };
        let packed = word != elem;
        // The results each instruction loads (all of them when packed).
        let per_inst = results.len() / count;
        for (n, part) in results.chunks(per_inst).enumerate() {
            let name = mem_inst_name(true, attrs, space, words, word, n == 0);
            let template = format!("{name} {dst}, [${words}]{policy};");
            let names: Vec<String> = if packed {
                (0..words).map(|_| self.fresh("word")).collect()
            } else {
                part.iter().map(|&r| self.v(r)).collect()
            };
            let outs: Vec<(String, DType)> = names.iter().map(|n| (n.clone(), word)).collect();
            let at = self.offset_address(indent, &addr, n * per_inst * elem.bytes());
            let mut inputs = vec![(at, DType::I64)];
            if let Some(p) = attrs.cache_policy {
                inputs.push((self.v(p), self.ir.dtype(p)));
            }
            self.ptx_inst(indent, inst, &template, &outs, &inputs, true);
            if packed {
                let per = word.bytes() / elem.bytes();
                let (t, wt) = (elem.mlir(), word.mlir());
                for (j, w) in names.iter().enumerate() {
                    let vec = self.fresh("unpacked");
                    self.line(indent, &format!("{vec} = llvm.bitcast {w} : {wt} to vector<{per}x{t}>"), None);
                    for k in 0..per {
                        let r = self.v(part[j * per + k]);
                        self.line(
                            indent,
                            &format!("{r} = vector.extract {vec}[{k}] : {t} from vector<{per}x{t}>"),
                            None,
                        );
                    }
                }
            }
        }
    }

    /// A store with memory attributes, as [`Self::load_with_attrs`] emits
    /// loads: an LLVM atomic or volatile store for a scalar relaxed,
    /// release, seq_cst or volatile store without hints, inline PTX `st`
    /// otherwise.
    fn store_with_attrs(
        &mut self,
        indent: usize,
        inst: &Inst,
        values: &[ValueId],
        buffer: BufferId,
        index: ValueId,
        attrs: &MemAttrs,
    ) {
        let elem = self.ir.dtype(values[0]);
        let space = self.ir.buffer(buffer).space;
        let addr = self.address(indent, buffer, elem, index);
        self.store_with_attrs_at(indent, inst, values, addr, space, attrs);
    }

    /// [`Self::store_with_attrs`] at the i64 address `addr` of `space`.
    fn store_with_attrs_at(
        &mut self,
        indent: usize,
        inst: &Inst,
        values: &[ValueId],
        addr: String,
        space: MemSpace,
        attrs: &MemAttrs,
    ) {
        let elem = self.ir.dtype(values[0]);
        if llvm_carries(attrs, values.len()) {
            let ptr = self.typed_pointer(indent, &addr, space);
            let t = elem.mlir();
            let (v, s) = (self.v(values[0]), llvm_space(space));
            let text = if attrs.order == MemoryOrder::Volatile {
                format!("llvm.store volatile {v}, {ptr} {{alignment = {} : i64}} : {t}, !llvm.ptr<{s}>", elem.bytes())
            } else {
                format!(
                    "llvm.store {v}, {ptr} atomic {}{} {{alignment = {} : i64}} : {t}, !llvm.ptr<{s}>",
                    syncscope(attrs.scope),
                    llvm_ordering(attrs.order),
                    elem.bytes()
                )
            };
            self.line(indent, &text, Some(inst));
            return;
        }
        let (count, words, word) = access_words(values.len(), elem, attrs.is_strong());
        let sources: Vec<String> = (1..=words).map(|k| format!("${k}")).collect();
        let src = if words == 1 { sources[0].clone() } else { format!("{{{}}}", sources.join(", ")) };
        let policy = if attrs.cache_policy.is_some() { format!(", ${}", words + 1) } else { String::new() };
        // The values each instruction stores (all of them when packed).
        let per_inst = values.len() / count;
        for (n, part) in values.chunks(per_inst).enumerate() {
            let name = mem_inst_name(false, attrs, space, words, word, n == 0);
            let template = format!("{name} [$0], {src}{policy};");
            let at = self.offset_address(indent, &addr, n * per_inst * elem.bytes());
            let mut inputs = vec![(at, DType::I64)];
            if word == elem {
                inputs.extend(part.iter().map(|&v| (self.v(v), elem)));
            } else {
                let per = word.bytes() / elem.bytes();
                let (t, wt) = (elem.mlir(), word.mlir());
                for chunk in part.chunks(per) {
                    let vec = self.fresh("packed");
                    self.line(
                        indent,
                        &format!("{vec} = vector.from_elements {} : vector<{per}x{t}>", self.results(chunk)),
                        None,
                    );
                    let w = self.fresh("word");
                    self.line(indent, &format!("{w} = llvm.bitcast {vec} : vector<{per}x{t}> to {wt}"), None);
                    inputs.push((w, word));
                }
            }
            if let Some(p) = attrs.cache_policy {
                inputs.push((self.v(p), self.ir.dtype(p)));
            }
            self.ptx_inst(indent, inst, &template, &[], &inputs, true);
        }
    }

    /// `addr` (an i64) plus `bytes`, or `addr` itself for 0.
    fn offset_address(&mut self, indent: usize, addr: &str, bytes: usize) -> String {
        if bytes == 0 {
            return addr.to_string();
        }
        let size = self.fresh("step");
        self.line(indent, &format!("{size} = arith.constant {bytes} : i64"), None);
        let sum = self.fresh("at");
        self.line(indent, &format!("{sum} = arith.addi {addr}, {size} : i64"), None);
        sum
    }

    /// A plain load (`load`) into `values`, or store of them, through a raw
    /// pointer ([`MemSpace::Raw`]): the index is the byte address, made a
    /// pointer of the space with `llvm.inttoptr` and accessed with
    /// `llvm.load`/`llvm.store`, one scalar or one vector. A scalar's
    /// alignment is the pointer's assumed one (at most the element's size);
    /// a vector's is its size, which the builder proved.
    fn raw_access(
        &mut self,
        indent: usize,
        inst: &Inst,
        values: &[ValueId],
        buffer: BufferId,
        index: ValueId,
        load: bool,
    ) {
        let d = self.ir.buffer(buffer);
        let (space, assumed) = (d.space, d.align);
        let elem = self.ir.dtype(values[0]);
        let n = values.len();
        let align = if n == 1 { elem.bytes().min(assumed) } else { n * elem.bytes() };
        let addr = self.v(index);
        self.ptr_access(indent, inst, values, &addr, space, align, load);
    }

    /// A plain load into `values`, or store of them, at the i64 address
    /// `addr` of `space`, aligned to `align` bytes: one scalar or one vector
    /// `llvm.load`/`llvm.store`.
    #[allow(clippy::too_many_arguments)]
    fn ptr_access(
        &mut self,
        indent: usize,
        inst: &Inst,
        values: &[ValueId],
        addr: &str,
        space: MemSpace,
        align: usize,
        load: bool,
    ) {
        let elem = self.ir.dtype(values[0]);
        let n = values.len();
        let ptr = self.typed_pointer(indent, addr, space);
        let pty = format!("!llvm.ptr<{}>", llvm_space(space));
        let t = elem.mlir();
        let vty = format!("vector<{n}x{t}>");
        if load {
            if n == 1 {
                let text =
                    format!("{} = llvm.load {ptr} {{alignment = {align} : i64}} : {pty} -> {t}", self.v(values[0]));
                self.line(indent, &text, Some(inst));
                return;
            }
            let vec = self.fresh("vec");
            self.line(
                indent,
                &format!("{vec} = llvm.load {ptr} {{alignment = {align} : i64}} : {pty} -> {vty}"),
                Some(inst),
            );
            for (k, r) in values.iter().enumerate() {
                let text = format!("{} = vector.extract {vec}[{k}] : {t} from {vty}", self.v(*r));
                self.line(indent, &text, None);
            }
        } else {
            let (v, ty) = if n == 1 {
                (self.v(values[0]), t.to_string())
            } else {
                let vec = self.fresh("vec");
                let text = format!("{vec} = vector.from_elements {} : {vty}", self.results(values));
                self.line(indent, &text, None);
                (vec, vty)
            };
            self.line(
                indent,
                &format!("llvm.store {v}, {ptr} {{alignment = {align} : i64}} : {ty}, {pty}"),
                Some(inst),
            );
        }
    }

    /// The address `x` of space `from` as one of space `to` ([`Op::CvtAddr`]):
    /// a pointer of `from`, `llvm.addrspacecast` to `to`, and its integer.
    fn cvt_addr(
        &mut self,
        indent: usize,
        inst: &Inst,
        result: ValueId,
        x: ValueId,
        from: AddressSpace,
        to: AddressSpace,
    ) {
        let (xt, rt) = (self.ir.dtype(x).mlir(), self.ir.dtype(result).mlir());
        let (p, q) = (self.fresh("ptr"), self.fresh("ptr"));
        let (f, g) = (from as u32, to as u32);
        self.line(indent, &format!("{p} = llvm.inttoptr {} : {xt} to !llvm.ptr<{f}>", self.v(x)), None);
        self.line(indent, &format!("{q} = llvm.addrspacecast {p} : !llvm.ptr<{f}> to !llvm.ptr<{g}>"), Some(inst));
        self.line(indent, &format!("{} = llvm.ptrtoint {q} : !llvm.ptr<{g}> to {rt}", self.v(result)), None);
    }

    /// The address of element `index` of a buffer as an integer of its own
    /// space ([`Op::PtrToInt`]): the 64-bit address, as a 32-bit offset in
    /// the shared window (a shared pointer's integer) or, for a per-thread
    /// buffer (an alloca, whose pointer is generic), the 32-bit offset in
    /// the local window (`addrspacecast` to space 5).
    fn ptr_to_int(
        &mut self,
        indent: usize,
        inst: &Inst,
        result: ValueId,
        buffer: BufferId,
        index: ValueId,
        elem: DType,
    ) {
        let addr = self.address(indent, buffer, elem, index);
        let rt = self.ir.dtype(result).mlir();
        let r = self.v(result);
        match self.ir.buffer(buffer).space {
            MemSpace::Global | MemSpace::Constant => {
                let zero = self.fresh("zero");
                self.line(indent, &format!("{zero} = arith.constant 0 : i64"), None);
                self.line(indent, &format!("{r} = arith.addi {addr}, {zero} : i64"), Some(inst));
            }
            MemSpace::Shared => self.line(indent, &format!("{r} = arith.trunci {addr} : i64 to {rt}"), Some(inst)),
            MemSpace::Local => {
                let (p, q) = (self.fresh("ptr"), self.fresh("ptr"));
                self.line(indent, &format!("{p} = llvm.inttoptr {addr} : i64 to !llvm.ptr"), None);
                self.line(indent, &format!("{q} = llvm.addrspacecast {p} : !llvm.ptr to !llvm.ptr<5>"), Some(inst));
                self.line(indent, &format!("{r} = llvm.ptrtoint {q} : !llvm.ptr<5> to {rt}"), None);
            }
            MemSpace::Raw(_) => unreachable!("a raw pointer's integer is its address, computed by the builder"),
        }
    }

    /// `addr` (an i64) as a pointer into the address space of `space`.
    fn typed_pointer(&mut self, indent: usize, addr: &str, space: MemSpace) -> String {
        let ptr = self.fresh("ptr");
        self.line(indent, &format!("{ptr} = llvm.inttoptr {addr} : i64 to !llvm.ptr<{}>", llvm_space(space)), None);
        ptr
    }

    /// An extended-precision instruction, as CuTe DSL emits it: one PTX
    /// instruction of inline assembly with side effects. The carry flag it
    /// reads or writes is invisible to LLVM, so the side effects are what
    /// keep a chain together: LLVM neither deletes nor reorders side-effecting
    /// inline assembly. Nothing LLVM emits between two of them touches the
    /// flag either: only the extended-precision instructions can, and NVPTX
    /// uses them for integers wider than 64 bits alone, whose arithmetic the
    /// validator keeps out of a chain (`tests/mlir.rs` checks the PTX).
    fn carry_inst(&mut self, indent: usize, inst: &Inst, result: ValueId, func: CarryFn, args: &[ValueId]) {
        let ty = self.ir.dtype(args[0]);
        let operands: Vec<String> = (0..=args.len()).map(|k| format!("${k}")).collect();
        let template = format!("{} {};", carry_inst_name(func, ty), operands.join(", "));
        let (results, inputs) = (self.asm_operands(&[result]), self.asm_operands(args));
        self.ptx_inst(indent, inst, &template, &results, &inputs, true);
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
            let contract = self.opts.fp_contract
                && matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul)
                && !self.ir.exact_float.contains(&result);
            let flags = if contract { " fastmath<contract>" } else { "" };
            self.line(indent, &format!("{out} = arith.{name} {l}, {r}{flags} : {t}"), i);
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

/// The LLVM address space of a pointer into `space` (generic for local
/// memory, whose addresses the kernel holds as generic ones).
fn llvm_space(space: MemSpace) -> u32 {
    match space {
        MemSpace::Global => 1,
        MemSpace::Shared => 3,
        // A module's constant globals are held as generic memrefs.
        MemSpace::Local | MemSpace::Constant => 0,
        // NVVM's numbers: generic 0, global 1, shared 3, local 5.
        MemSpace::Raw(a) => a as u32,
    }
}

/// The `syncscope("...") ` of an LLVM atomic access with `scope`, which
/// NVPTX lowers to the PTX scope: none (the system) for `.sys`.
fn syncscope(scope: MemoryScope) -> &'static str {
    match scope {
        MemoryScope::Cta => "syncscope(\"block\") ",
        MemoryScope::Cluster => "syncscope(\"cluster\") ",
        MemoryScope::Gpu => "syncscope(\"device\") ",
        MemoryScope::Sys => "",
    }
}

/// Whether LLVM can express an access with `attrs` moving `n` elements: a
/// scalar relaxed, acquire, release, seq_cst or volatile access without
/// hints (an LLVM atomic or volatile access). Weak accesses with hints,
/// non-coherent (constant) loads, MMIO and vectors need inline PTX.
fn llvm_carries(attrs: &MemAttrs, n: usize) -> bool {
    n == 1
        && !attrs.has_hints()
        && matches!(
            attrs.order,
            MemoryOrder::Relaxed
                | MemoryOrder::Acquire
                | MemoryOrder::Release
                | MemoryOrder::Sc
                | MemoryOrder::Volatile
        )
}

/// The LLVM ordering of an atomic load or store with `order`.
fn llvm_ordering(order: MemoryOrder) -> &'static str {
    match order {
        MemoryOrder::Relaxed => "monotonic",
        MemoryOrder::Acquire => "acquire",
        MemoryOrder::Release => "release",
        MemoryOrder::Sc => "seq_cst",
        other => unreachable!("{other:?} is not an LLVM atomic ordering"),
    }
}

/// How inline PTX moves `n` elements of `elem`: as `(instructions, words,
/// word type)`, each instruction moving `words` words of the word type
/// (`.v2`, `.v4`) as bits. Elements of 32 bits or more are the words
/// themselves. A weak access packs narrower ones into 16- or 32-bit words,
/// so that one instruction moves them all. A strong one keeps the element
/// type: PTX's memory model treats each element of a vector access as an
/// access of its own, and two accesses are morally strong (they neither race
/// nor fail to synchronise) only if they cover the same bytes, so a packed
/// word would not match another thread's access to one element. PTX has no
/// vector of more than four elements narrower than 32 bits, so a strong
/// access of eight or sixteen takes one instruction per four.
fn access_words(n: usize, elem: DType, strong: bool) -> (usize, usize, DType) {
    if strong {
        let words = n.min(4);
        return (n / words, words, elem);
    }
    let bytes = n * elem.bytes();
    let word = elem.bytes().max(bytes.min(4));
    let ty = if word == elem.bytes() {
        elem
    } else if word == 2 {
        DType::I16
    } else {
        DType::I32
    };
    (1, bytes / word, ty)
}

/// The PTX `ld` (`load`) or `st` of an access with memory attributes to
/// `space`, moving `words` words of `word`: the qualifiers in the order of
/// the ISA's syntax (semantics and scope, state space, cache operator,
/// `.nc`, L1 eviction priority, `.L2::cache_hint`, prefetch size), the
/// vector width and the bit type. A seq_cst access is LLVM's mapping, a
/// `fence.sc` (with `fence`, the first instruction of the access) and then
/// an acquire load or a release store. Local memory is addressed
/// generically.
fn mem_inst_name(load: bool, attrs: &MemAttrs, space: MemSpace, words: usize, word: DType, fence: bool) -> String {
    let scope = attrs.scope.ptx();
    let mut s = String::new();
    if attrs.order == MemoryOrder::Sc && fence {
        s.push_str(&format!("fence.sc.{scope};\n\t"));
    }
    s.push_str(if load { "ld" } else { "st" });
    match attrs.order {
        MemoryOrder::Weak | MemoryOrder::Constant => {}
        MemoryOrder::Relaxed => s.push_str(&format!(".relaxed.{scope}")),
        MemoryOrder::Acquire => s.push_str(&format!(".acquire.{scope}")),
        MemoryOrder::Release => s.push_str(&format!(".release.{scope}")),
        MemoryOrder::Sc => s.push_str(&format!(".{}.{scope}", if load { "acquire" } else { "release" })),
        MemoryOrder::Volatile => s.push_str(".volatile"),
        MemoryOrder::Mmio => s.push_str(".mmio.relaxed.sys"),
        MemoryOrder::AcqRel => unreachable!("validated: loads and stores are not acq_rel"),
    }
    s.push_str(match space {
        MemSpace::Global | MemSpace::Raw(crate::tensor::AddressSpace::Gmem) => ".global",
        MemSpace::Shared | MemSpace::Raw(crate::tensor::AddressSpace::Smem) => ".shared",
        MemSpace::Raw(crate::tensor::AddressSpace::Dsmem) => ".shared::cluster",
        MemSpace::Raw(crate::tensor::AddressSpace::Rmem) => ".local",
        MemSpace::Local | MemSpace::Constant | MemSpace::Raw(_) => "",
    });
    if let Some(c) = attrs.cop {
        s.push_str(&format!(".{}", c.ptx()));
    }
    if attrs.is_non_coherent() {
        s.push_str(".nc");
    }
    if let Some(e) = attrs.l1_eviction {
        s.push_str(&format!(".{}", e.l1_ptx()));
    }
    if attrs.cache_policy.is_some() {
        s.push_str(".L2::cache_hint");
    }
    if let Some(p) = attrs.l2_prefetch {
        s.push_str(&format!(".{}", p.ptx()));
    }
    if words > 1 {
        s.push_str(&format!(".v{words}"));
    }
    s.push_str(&format!(".b{}", word.bits()));
    s
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

/// The element type `stmatrix` addresses its buffer in: the buffer's own when
/// it has 16-bit elements, else (a struct's byte buffer) 16-bit integers.
fn stmatrix_elem(ir: &KernelIr, buffer: BufferId, form: StMatrixForm) -> DType {
    match form {
        StMatrixForm::M8n8B16 => match ir.buffer(buffer).elem {
            e if e.bits() == 16 => e,
            _ => DType::I16,
        },
        _ => DType::I8,
    }
}

/// The element types each buffer is accessed as.
fn access_types(ir: &KernelIr) -> Vec<BTreeSet<&'static str>> {
    fn walk(
        r: &Region,
        ir: &KernelIr,
        out: &mut [BTreeSet<&'static str>],
        types: &mut BTreeMap<&'static str, DType>,
        addressed: &mut BTreeSet<(usize, &'static str)>,
    ) {
        for inst in &r.insts {
            let mut note = |b: BufferId, t: DType| {
                out[b.0 as usize].insert(t.mlir());
                types.insert(t.mlir(), t);
            };
            match &inst.op {
                Op::LdMatrix { buffer, form, .. } if *form != LdMatrixForm::M8n8B16 => note(*buffer, DType::I8),
                Op::StMatrix { buffer, form: StMatrixForm::M16n8B8, .. } => note(*buffer, DType::I8),
                Op::StMatrix { buffer, form, .. } => note(*buffer, stmatrix_elem(ir, *buffer, *form)),
                Op::Load { results, buffer, .. } | Op::LdMatrix { results, buffer, .. } => {
                    note(*buffer, ir.dtype(results[0]))
                }
                Op::Store { values, buffer, .. } => note(*buffer, ir.dtype(values[0])),
                Op::Atomic { value, buffer, ty, .. } | Op::Red { value, buffer, ty, .. } => {
                    note(*buffer, ty.access_elem(ir.dtype(*value)))
                }
                Op::MbarrierInit { buffer, .. }
                | Op::MbarrierArrive { buffer, .. }
                | Op::MbarrierExpectTx { buffer, .. }
                | Op::MbarrierWait { buffer, .. }
                | Op::CpAsyncMbarrierArrive { buffer, .. }
                | Op::Tcgen05Commit { buffer, .. }
                | Op::MbarrierExt { buffer, .. } => note(*buffer, DType::I64),
                Op::StBulk { buffer, elem, .. } => note(*buffer, *elem),
                Op::CpAsyncBulk { src, dst, elem, completion, .. } => {
                    note(*src, *elem);
                    note(*dst, *elem);
                    if let BulkCompletion::Mbarrier { buffer, .. } = completion {
                        note(*buffer, DType::I64);
                    }
                }
                Op::CpAsync { src, dst, elem, .. } => {
                    note(*src, *elem);
                    note(*dst, *elem);
                }
                Op::TmaLoad { dst, elem, mbar, .. } => {
                    note(*dst, *elem);
                    note(*mbar, DType::I64);
                }
                Op::TmaStore { src, elem, .. } => note(*src, *elem),
                Op::Tcgen05 { buffer: Some(b), .. } => note(*b, DType::I32),
                Op::BulkPrefetch { buffer, elem, .. } => note(*buffer, *elem),
                Op::StAsync { buffer, elem, mbar, .. } | Op::ClcTryCancel { buffer, elem, mbar, .. } => {
                    note(*buffer, *elem);
                    note(*mbar, DType::I64);
                }
                Op::ClcQueryCancel { buffer, elem, .. } => note(*buffer, *elem),
                Op::InlinePtx { inputs, .. } | Op::MlirOp { operands: inputs, .. } => {
                    for input in inputs {
                        if let AsmInput::Address { buffer, elem, .. } = *input {
                            note(buffer, elem);
                        }
                    }
                }
                Op::Prefetch { buffer, elem, .. } => note(*buffer, *elem),
                Op::PtrToInt { buffer, elem, .. } => {
                    addressed.insert((buffer.0 as usize, elem.mlir()));
                }
                _ => {}
            }
            inst.op.regions().into_iter().for_each(|r| walk(r, ir, out, types, addressed));
        }
    }
    let mut out = vec![BTreeSet::new(); ir.buffers.len()];
    let mut types = BTreeMap::new();
    let mut addressed = BTreeSet::new();
    walk(&ir.body, ir, &mut out, &mut types, &mut addressed);
    // A shared buffer in the dynamic window is reached through a typed view
    // of the window, so one whose address is taken but which is never
    // accessed (a struct whose members only give addresses) needs a view
    // too: the type of its first address. Buffers with accesses keep theirs,
    // and addresses of other types use one of them (`address_view`).
    if ir.shared_memory_layout().in_window {
        for (b, ty) in addressed {
            if out[b].is_empty() && ir.buffers[b].space == MemSpace::Shared {
                out[b].insert(ty);
            }
        }
    }
    out
}

/// Types stored as another type's MLIR type: TFloat32 (as `f32`), Int4
/// and the 8-bit float formats (as `i8`).
fn stored_as_other(ty: DType) -> bool {
    ty == DType::TF32 || ty == DType::I4 || ty.is_narrow_float() || ty.is_packed_view()
}

/// The type an MLIR storage type names: from the type table, the type
/// whose MLIR type is its own name (Int8 for `i8`, not Uint8, Int4 or an
/// FP8 format), and Boolean for `i1` (a byte per value in memory).
fn dtype_of(name: &str) -> DType {
    if name == DType::Bool.mlir() {
        return DType::Bool;
    }
    DType::ALL.into_iter().find(|&t| t != DType::Bool && t.mlir() == name && t.name() == name).unwrap_or(DType::I8)
}

/// Emit `ir` as an MLIR module containing one `gpu.module` with the kernel.
pub fn emit_mlir(ir: &KernelIr, opts: &MlirOptions) -> String {
    emit_mlir_in_module(ir, opts, &BTreeSet::new())
}

/// [`emit_mlir`], declaring the module globals named in `bytes_globals` as bytes whatever
/// types the kernel accesses them with, so that kernels sharing one module declare a shared
/// global alike (see `backend::module`).
pub(crate) fn emit_mlir_in_module(ir: &KernelIr, opts: &MlirOptions, bytes_globals: &BTreeSet<String>) -> String {
    let smem = ir.shared_memory_layout();
    let dynamic_smem = smem.in_window;
    let used = access_types(ir);
    let bytes: Vec<bool> = ir
        .buffers
        .iter()
        .zip(&used)
        .enumerate()
        .map(|(k, (b, u))| {
            b.space.raw().is_none()
                && ((dynamic_smem && b.space == MemSpace::Shared)
                    || u.iter().any(|&t| t != b.elem.mlir())
                    || (bytes_globals.contains(&b.name) && ir.globals.iter().any(|g| g.buffer.0 as usize == k)))
        })
        .collect();
    let mut e = Emitter {
        ir,
        opts,
        out: String::new(),
        tmp: 0,
        current: None,
        current_scope: None,
        other_files: Vec::new(),
        aliases: BTreeMap::new(),
        renames: BTreeMap::new(),
        bytes,
        views: BTreeMap::new(),
        dynamic_smem,
        dynamic_start: smem.dynamic_start,
        window_starts: smem.starts.clone(),
        tcgen05_fences: ir.count_ops(|op| {
            matches!(
                op,
                Op::Tcgen05Ld { .. }
                    | Op::Tcgen05St { .. }
                    | Op::Tcgen05Mma { .. }
                    | Op::Tcgen05Commit { .. }
                    | Op::Tcgen05Cp { .. }
            )
        }) > 0,
    };
    if opts.line_info {
        e.debug_info_header();
    }
    e.line(0, "module attributes {gpu.container_module} {", None);
    e.line(2, "gpu.module @kernels {", None);
    let dynamic_type = "memref<0xi8, #gpu.address_space<workgroup>>";
    if e.dynamic_smem {
        // An external declaration, which PTX allocates in dynamic shared
        // memory (`gpu.dynamic_shared_memory` would leave one byte of static
        // shared memory behind: see DYNAMIC_SMEM_SYMBOL).
        let text = format!(
            "memref.global @{DYNAMIC_SMEM_SYMBOL} : {dynamic_type} {{alignment = {} : i64}}",
            smem.window_align
        );
        e.line(4, &text, None);
    }
    for g in &ir.globals {
        // A public global in its own address space, read and written in the kernel through
        // a generic memref (memref.memory_space_cast), like any global buffer.
        let d = ir.buffer(g.buffer);
        let space = if d.space == MemSpace::Constant { 4 } else { 1 };
        let ty = e.memref_type(g.buffer).replacen('>', &format!(", {space}>"), 1);
        // Hexadecimal raw data; one element's worth is a splat (zeros without `init`).
        let hex = match &g.init {
            Some(bytes) => bytes.iter().map(|b| format!("{b:02X}")).collect::<String>(),
            None if e.bytes[g.buffer.0 as usize] => "00".to_string(),
            None => "00".repeat(d.elem.bytes()),
        };
        let init = format!(" = dense<\"0x{hex}\">");
        let constant = if g.readonly { "constant " } else { "" };
        let text = format!("memref.global {constant}@{} : {ty}{init} {{alignment = {} : i64}}", d.name, d.align);
        e.line(4, &text, None);
    }
    // External device functions: their prototypes.
    for d in &ir.externs {
        e.line(4, &d.prototype(), None);
    }
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
            // Bytes in the parameter space: the stage hook marks the pointer `llvm.byval`.
            Param::ByValue { buffer, .. } => format!(
                "{}: {} {{llvm.align = {} : i64}}",
                e.buffer_name(*buffer),
                e.memref_type(*buffer),
                ir.buffer(*buffer).align
            ),
        })
        .collect();
    // Launch bounds: `.maxntid` from max_threads, else `.reqntid` from a declared block
    // (PTX forbids both), and `.minnctapersm`.
    let mut bounds = Vec::new();
    match (ir.max_threads, ir.block_dim) {
        (Some([x, y, z]), _) => bounds.push(format!("nvvm.maxntid = array<i32: {x}, {y}, {z}>")),
        (None, Some([x, y, z])) => bounds.push(format!("nvvm.reqntid = array<i32: {x}, {y}, {z}>")),
        (None, None) => {}
    }
    if let Some(n) = ir.min_blocks_per_sm {
        bounds.push(format!("nvvm.minctasm = {n} : i32"));
    }
    let attrs = if bounds.is_empty() { String::new() } else { format!(" attributes {{{}}}", bounds.join(", ")) };
    match ir.function.as_ref().and_then(|f| f.result) {
        // A standalone device function: a public non-kernel function (`.visible .func`).
        Some(r) => {
            e.line(4, &format!("gpu.func @{}({}) -> {} {{", ir.name, params.join(", "), e.ty(r)), None);
        }
        None if ir.function.is_some() => e.line(4, &format!("gpu.func @{}({}) {{", ir.name, params.join(", ")), None),
        None => e.line(4, &format!("gpu.func @{}({}) kernel{attrs} {{", ir.name, params.join(", ")), None),
    }
    if e.dynamic_smem {
        e.line(6, &format!("%dyn_smem = memref.get_global @{DYNAMIC_SMEM_SYMBOL} : {dynamic_type}"), None);
    }
    for (k, b) in ir.buffers.iter().enumerate() {
        let id = BufferId(k as u32);
        if ir.globals.iter().any(|g| g.buffer == id) {
            let space = if b.space == MemSpace::Constant { 4 } else { 1 };
            let ty = e.memref_type(id);
            let spaced = ty.replacen('>', &format!(", {space}>"), 1);
            let raw = format!("%{}_global", b.name);
            e.line(6, &format!("{raw} = memref.get_global @{} : {spaced}", b.name), None);
            let text = format!("{} = memref.memory_space_cast {raw} : {spaced} to {ty}", e.buffer_name(id));
            e.line(6, &text, None);
        }
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
        let start = if b.space == MemSpace::Shared && e.dynamic_smem { smem.starts[k].unwrap_or(0) } else { 0 };
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
    match ir.function.as_ref().and_then(|f| f.result) {
        Some(r) => {
            let text = format!("gpu.return {} : {}", e.v(r), e.ty(r));
            e.line(6, &text, None);
        }
        None => e.line(6, "gpu.return", None),
    }
    if e.located() {
        let close = format!("}} {}", e.function_loc());
        e.plain_line(4, &close);
        e.plain_line(2, "} loc(unknown)");
        e.plain_line(0, "} loc(unknown)");
    } else {
        e.plain_line(4, "}");
        e.plain_line(2, "}");
        e.plain_line(0, "}");
    }
    e.out
}

/// The source files of every instruction in `r`.
fn collect_files(r: &Region, files: &mut BTreeSet<&'static str>) {
    for inst in &r.insts {
        files.insert(inst.loc.file());
        inst.op.regions().into_iter().for_each(|r| collect_files(r, files));
    }
}

/// The name and directory of a source file's `#llvm.di_file`: the Rust path
/// (as `Location::file` gives it) in the directory it is relative to. That is
/// `CARGO_MANIFEST_DIR` when set (cargo sets it for tests, examples and `cargo
/// run`), otherwise the working directory, or the nearest ancestor of either
/// that holds the file (workspace members' paths are relative to the
/// workspace root).
fn di_file_parts(file: &str) -> (String, String) {
    if Path::new(file).is_absolute() {
        return (file.to_string(), String::new());
    }
    let base = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_default();
    let dir = base.ancestors().find(|d| d.join(file).is_file()).unwrap_or(&base);
    (file.to_string(), dir.to_string_lossy().into_owned())
}
