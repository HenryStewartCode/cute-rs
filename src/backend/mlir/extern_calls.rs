//! MLIR for external device-function calls, standalone device functions and multimem
//! instructions.
//!
//! An external function is a `func.func private` prototype in the `gpu.module`, marked
//! `passthrough = ["convergent"]` unless it opted out, and a call is `func.call`, with
//! `{no_inline}` when the function is not to be inlined. A pointer argument is made from the
//! element's address ([`Emitter::address`]) as `llvm.inttoptr` into the parameter's address
//! space; a shared-memory pointer passed where the parameter is generic goes through
//! `llvm.addrspacecast` (global and per-thread buffers' addresses already are generic). Bitcode the calls resolve to is linked by `nvvm-attach-target`'s `l=` option
//! ([`LoweringPipeline::with_link_libraries`](crate::backend::LoweringPipeline::with_link_libraries)).
//!
//! A standalone device function is a public non-kernel `gpu.func` with a result, which NVPTX
//! emits as `.visible .func`: the ABI of an `extern "C" __device__` function.
//!
//! The multimem instructions are inline PTX: mlir-opt 21 has no `nvvm.multimem` operations,
//! and CuTe DSL writes them as inline PTX too.

use super::Emitter;
use crate::ir::{AsmInput, BufferId, DType, ExternId, ExternParam, Inst, MemSpace, MultimemOp, ValueId};
use crate::tensor::AddressSpace;

impl Emitter<'_> {
    /// `func.call` of external function `callee`.
    pub(super) fn call(
        &mut self,
        indent: usize,
        inst: &Inst,
        callee: ExternId,
        args: &[AsmInput],
        results: &[ValueId],
    ) {
        let decl = self.ir.extern_decl(callee).clone();
        let mut names = Vec::new();
        for (arg, param) in args.iter().zip(&decl.params) {
            names.push(match (*arg, *param) {
                (AsmInput::Value(v), _) => self.v(v),
                (AsmInput::Address { buffer, index, elem }, ExternParam::Ptr(want)) => {
                    self.pointer_arg(indent, buffer, index, elem, want)
                }
                (AsmInput::Address { buffer, index, elem }, ExternParam::Value(_)) => {
                    // Rejected by the validator; the address keeps the IR printable.
                    self.address(indent, buffer, elem, index)
                }
            });
        }
        let types: Vec<String> = decl.params.iter().map(ExternParam::mlir).collect();
        let lhs = match results {
            [] => String::new(),
            rs => format!("{} = ", rs.iter().map(|r| self.v(*r)).collect::<Vec<_>>().join(", ")),
        };
        let ret = match decl.results.as_slice() {
            [] => "()".to_string(),
            [t] => t.mlir().to_string(),
            ts => format!("({})", ts.iter().map(|t| t.mlir()).collect::<Vec<_>>().join(", ")),
        };
        let attr = if decl.inline { "" } else { " {no_inline}" };
        let text =
            format!("{lhs}func.call @{}({}){attr} : ({}) -> {ret}", decl.symbol, names.join(", "), types.join(", "));
        self.line(indent, &text, Some(inst));
    }

    /// A pointer argument: the address of element `index` of `buffer` as a pointer in address
    /// space `want`.
    fn pointer_arg(
        &mut self,
        indent: usize,
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        want: AddressSpace,
    ) -> String {
        let addr = self.address(indent, buffer, elem, index);
        let have = match self.ir.buffer(buffer).space {
            MemSpace::Shared => AddressSpace::Smem,
            MemSpace::Raw(a) => a,
            // Global buffers' and per-thread buffers' addresses are generic ones too.
            _ => AddressSpace::Generic,
        };
        let ty = |a: AddressSpace| match a {
            AddressSpace::Generic => "!llvm.ptr".to_string(),
            a => format!("!llvm.ptr<{}>", a as u32),
        };
        let start = if have == AddressSpace::Smem { AddressSpace::Smem } else { want };
        let p = self.fresh("ptr");
        self.line(indent, &format!("{p} = llvm.inttoptr {addr} : i64 to {}", ty(start)), None);
        if start == want {
            return p;
        }
        let q = self.fresh("ptr");
        self.line(indent, &format!("{q} = llvm.addrspacecast {p} : {} to {}", ty(start), ty(want)), None);
        q
    }

    /// A multimem instruction on the multimem address of element `index` of `buffer`.
    pub(super) fn multimem(
        &mut self,
        indent: usize,
        inst: &Inst,
        op: &MultimemOp,
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        values: &[ValueId],
        results: &[ValueId],
    ) {
        let addr = self.address(indent, buffer, elem, index);
        let ptx = op.ptx();
        let regs = |first: usize, n: usize| -> String {
            let names: Vec<String> = (first..first + n).map(|k| format!("${k}")).collect();
            if n == 1 { names[0].clone() } else { format!("{{{}}}", names.join(", ")) }
        };
        match op {
            MultimemOp::LdReduce { .. } => {
                let template = format!("{ptx} {}, [${}];", regs(0, results.len()), results.len());
                let outs: Vec<(String, DType)> = results.iter().map(|&r| (self.v(r), self.ir.dtype(r))).collect();
                self.ptx_inst(indent, inst, &template, &outs, &[(addr, DType::I64)], true);
            }
            MultimemOp::St { .. } => {
                let template = format!("{ptx} [$0], {};", regs(1, values.len()));
                let mut inputs = vec![(addr, DType::I64)];
                inputs.extend(values.iter().map(|&v| (self.v(v), self.ir.dtype(v))));
                self.asm_memory(indent, inst, &template, &inputs);
            }
            MultimemOp::RedAdd1 { .. } => {
                self.asm_memory(indent, inst, &format!("{ptx} [$0], 1;"), &[(addr, DType::I64)]);
            }
        }
    }
}
