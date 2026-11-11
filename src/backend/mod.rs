//! Code generation: a readable IR listing, MLIR, and PTX via `mlir-opt`.

mod mlir;
mod toolchain;

use std::fmt::Write;

pub use mlir::{MlirOptions, emit_mlir};
pub use toolchain::{Arch, CompileError, Toolchain};

use crate::builder::Kernel;
use crate::ir::{Inst, KernelIr, MemSpace, Op, Param, Region};

impl Kernel {
    /// The kernel as MLIR (upstream dialects only).
    pub fn to_mlir(&self) -> String {
        emit_mlir(self.ir(), &MlirOptions::default())
    }

    /// The kernel as MLIR, with options (e.g. source locations).
    pub fn to_mlir_with(&self, opts: &MlirOptions) -> String {
        emit_mlir(self.ir(), opts)
    }

    /// Compile to PTX. The kernel's parameters are passed as bare pointers
    /// (for buffers) and scalars, in declaration order. Fails with
    /// [`CompileError::Unsupported`] if an instruction needs a newer
    /// architecture than `arch`; the PTX ISA version is raised when an
    /// instruction needs it.
    pub fn compile_ptx(&self, toolchain: &Toolchain, arch: Arch) -> Result<String, CompileError> {
        let ptx = crate::validate::check_target(self.ir(), arch).map_err(|e| {
            CompileError::Unsupported(format!("kernel `{}` cannot run on {}: {e}", self.name(), arch.chip()))
        })?;
        toolchain.compile_ptx_isa(&self.to_mlir(), arch, ptx)
    }
}

/// A human-readable listing of the IR.
pub fn dump(ir: &KernelIr) -> String {
    let mut out = String::new();
    let params: Vec<String> = ir
        .args
        .iter()
        .map(|p| match p {
            Param::Buffer(b) => {
                let d = ir.buffer(*b);
                let len = d.len.map_or("?".to_string(), |n| n.to_string());
                format!("{}: gmem<{len} x {}>", d.name, d.elem)
            }
            Param::Scalar { name, value, .. } => format!("{name} = {value}: {}", ir.dtype(*value)),
        })
        .collect();
    let _ = write!(out, "kernel {}({})", ir.name, params.join(", "));
    if let Some([x, y, z]) = ir.block_dim {
        let _ = write!(out, " block=({x},{y},{z})");
    }
    out.push('\n');
    for b in ir.buffers.iter().filter(|b| b.space == MemSpace::Shared) {
        let _ = writeln!(out, "  shared {}: {} x {} (align {})", b.name, b.len.unwrap_or(0), b.elem, b.align);
    }
    dump_region(ir, &ir.body, 1, &mut out);
    out
}

fn list(vs: &[crate::ir::ValueId]) -> String {
    vs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
}

fn dump_region(ir: &KernelIr, r: &Region, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    for Inst { op, .. } in &r.insts {
        let results = op.results();
        let lhs = if results.is_empty() { String::new() } else { format!("{} = ", list(&results)) };
        let buf = |b: &crate::ir::BufferId| ir.buffer(*b).name.clone();
        let body = match op {
            Op::Const { value, .. } => format!("const {value:?}"),
            Op::Unary { op, x, .. } => format!("{op:?} {x}"),
            Op::Binary { op, lhs, rhs, .. } => format!("{op:?} {lhs}, {rhs}"),
            Op::Fma { a, b, c, .. } => format!("fma {a}, {b}, {c}"),
            Op::FloatInst { func, args, mode, ftz, nan, .. } => {
                let name = crate::ir::float_inst_name(*func, *mode, *ftz, *nan, ir.dtype(args[0]));
                format!("{name} {}", list(args))
            }
            Op::Cmp { pred, lhs, rhs, .. } => format!("cmp {pred:?} {lhs}, {rhs}"),
            Op::Select { cond, on_true, on_false, .. } => format!("select {cond}, {on_true}, {on_false}"),
            Op::Cast { x, .. } => format!("cast {x}"),
            Op::Bitcast { x, .. } => format!("bitcast {x}"),
            Op::Special { reg, .. } => format!("{reg:?}"),
            Op::Load { buffer, index, .. } => format!("load {}[{index}]", buf(buffer)),
            Op::Store { values, buffer, index } => format!("store {} -> {}[{index}]", list(values), buf(buffer)),
            Op::CpAsync { src, src_index, dst, dst_index, elems, pred, .. } => {
                let p = pred.map(|p| format!(" if {p}")).unwrap_or_default();
                format!("cp.async {}[{src_index}] -> {}[{dst_index}] x{elems}{p}", buf(src), buf(dst))
            }
            Op::CpAsyncCommitGroup => "cp.async.commit_group".into(),
            Op::CpAsyncWaitGroup { pending } => format!("cp.async.wait_group {pending}"),
            Op::LdMatrix { buffer, index, num, trans, .. } => {
                format!("ldmatrix.x{num}{} {}[{index}]", if *trans { ".trans" } else { "" }, buf(buffer))
            }
            Op::Mma { kind, a, b, c, .. } => format!("mma {kind:?} a=[{}] b=[{}] c=[{}]", list(a), list(b), list(c)),
            Op::Barrier => "barrier".into(),
            Op::NamedBarrier { id, threads, .. } => {
                let n = threads.map(|n| format!(", {n}")).unwrap_or_default();
                format!("{} {id}{n}", op.name())
            }
            Op::WarpBarrier { mask } => format!("warp barrier {mask}"),
            Op::Shuffle { kind, value, offset, mask, .. } => format!("shuffle {kind:?} {value}, {offset}, {mask}"),
            Op::Vote { kind, pred, mask, .. } => format!("vote {kind:?} {pred}, {mask}"),
            Op::MatchAny { value, mask, .. } => format!("match {value}, {mask}"),
            Op::ElectOne { mask, .. } => format!("elect {mask}"),
            Op::ActiveMask { .. } => "activemask".into(),
            Op::WarpUniform { value, .. } => format!("make_warp_uniform {value}"),
            Op::Atomic { kind, buffer, index, value, compare, .. } => {
                let c = compare.map(|c| format!(" if {c}")).unwrap_or_default();
                format!("atomic {kind:?} {}[{index}], {value}{c}", buf(buffer))
            }
            Op::MbarrierInit { buffer, index, count } => format!("mbarrier.init {}[{index}], {count}", buf(buffer)),
            Op::MbarrierArrive { buffer, index, count, expect_tx, peer, relaxed, scope } => {
                let mut text = format!("mbarrier.arrive {}[{index}]", buf(buffer));
                if let Some(c) = count {
                    let _ = write!(text, " x{c}");
                }
                if let Some(t) = expect_tx {
                    let _ = write!(text, " expect {t}");
                }
                if let Some(p) = peer {
                    let _ = write!(text, " on cta {p}");
                }
                let _ = write!(text, " {}{scope:?}", if *relaxed { "relaxed " } else { "" });
                text
            }
            Op::MbarrierExpectTx { buffer, index, bytes, peer, scope } => {
                let p = peer.map(|p| format!(" on cta {p}")).unwrap_or_default();
                format!("mbarrier.expect_tx {}[{index}], {bytes}{p} {scope:?}", buf(buffer))
            }
            Op::MbarrierWait { buffer, index, parity, .. } => {
                format!("{} {}[{index}], {parity}", op.name(), buf(buffer))
            }
            Op::FenceMbarrierInit => "fence.mbarrier_init".into(),
            Op::CpAsyncBulk { src, src_index, dst, dst_index, bytes, mbarrier, mbarrier_index, .. } => format!(
                "cp.async.bulk {}[{src_index}] -> {}[{dst_index}] x{bytes} mbarrier {}[{mbarrier_index}]",
                buf(src),
                buf(dst),
                buf(mbarrier)
            ),
            Op::Fence { scope } => format!("fence {scope:?}"),
            Op::Printf { format, args } => format!("printf {format:?} {}", list(args)),
            Op::InlinePtx { template, inputs, .. } => format!("ptx {template:?} {}", list(inputs)),
            Op::If { cond, then_region, else_region, .. } => {
                let _ = writeln!(out, "{pad}{lhs}if {cond} {{");
                dump_region(ir, then_region, depth + 1, out);
                let _ = writeln!(out, "{pad}}} else {{");
                dump_region(ir, else_region, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
                continue;
            }
            Op::For { iv, lower, upper, step, init, args, body, opts, .. } => {
                let iters: Vec<String> = args.iter().zip(init).map(|(a, i)| format!("{a} = {i}")).collect();
                let mut hints = String::new();
                if let Some(count) = opts.unroll {
                    let _ = write!(hints, " unroll({count})");
                }
                for (on, name) in [
                    (opts.unroll_full, "unroll_full"),
                    (opts.vectorize, "vectorize"),
                    (opts.at_least_once, "at_least_once"),
                ] {
                    if on {
                        let _ = write!(hints, " {name}");
                    }
                }
                let _ = writeln!(
                    out,
                    "{pad}{lhs}for {iv} in {lower}..{upper} step {step} iter({}){hints} {{",
                    iters.join(", ")
                );
                dump_region(ir, body, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
                continue;
            }
            Op::While { init, cond_args, cond, body_args, body, .. } => {
                let iters: Vec<String> = cond_args.iter().zip(init).map(|(a, i)| format!("{a} = {i}")).collect();
                let _ = writeln!(out, "{pad}{lhs}while ({}) {{", iters.join(", "));
                dump_region(ir, cond, depth + 1, out);
                let _ = writeln!(out, "{pad}}} do ({}) {{", list(body_args));
                dump_region(ir, body, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
                continue;
            }
        };
        let _ = writeln!(out, "{pad}{lhs}{body}");
    }
    if !r.yields.is_empty() {
        let _ = writeln!(out, "{pad}yield {}", list(&r.yields));
    }
}
