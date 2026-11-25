//! Validation of external device-function calls, standalone device functions and multimem
//! instructions on multicast buffers.

use super::{ArchSpecific, Checker, Result, TargetRequirement};
use crate::ir::{AsmInput, BufferId, DType, ExternParam, KernelIr, MemSpace, MultimemOp, Op, Param};
use crate::tensor::AddressSpace;

/// The targets of `multimem.ld_reduce` on FP8 (`.e4m3x4`, `.e5m2x4`) and of
/// `.acc::f16`: sm_100a, sm_101a (sm_110a from PTX 9.0), sm_120a and sm_121a
/// from PTX ISA 8.6, and the sm_100f and sm_101f (sm_110f) families from PTX
/// ISA 8.8. The PTX ISA lists no sm_120f family for them: on sm_120 and
/// sm_121 only the `a` targets have them.
pub const MULTIMEM_FP8: ArchSpecific = ArchSpecific {
    chips: &[(100, 86), (101, 86), (110, 90), (120, 86), (121, 86)],
    families: &[(100, 88), (101, 88), (110, 90)],
    baselines: &[],
};

/// What a multimem instruction needs: sm_90 and PTX ISA 8.1; `.acc::f32`
/// PTX ISA 8.2; FP8 the targets of [`MULTIMEM_FP8`].
pub(super) fn multimem_requirement(op: &MultimemOp) -> Option<TargetRequirement> {
    let need = |sm, ptx, what| Some(TargetRequirement { sm, ptx, what, arch_specific: None, older_ptx: None });
    match op {
        MultimemOp::LdReduce { ty, .. } if ty.is_fp8() => Some(TargetRequirement {
            sm: MULTIMEM_FP8.oldest(),
            ptx: MULTIMEM_FP8.min_ptx(),
            what: "multimem.ld_reduce on .e4m3x4 or .e5m2x4 (an FP8 multimem_ld_reduce, with .acc::f16)",
            arch_specific: Some(MULTIMEM_FP8),
            older_ptx: None,
        }),
        MultimemOp::LdReduce { ty, .. } if ty.acc().is_some() => {
            need(90, 82, "multimem.ld_reduce with .acc::f32 (a Float16 or BFloat16 multimem_ld_reduce)")
        }
        MultimemOp::LdReduce { .. } => need(90, 81, "multimem.ld_reduce"),
        MultimemOp::St { .. } => need(90, 81, "multimem.st"),
        MultimemOp::RedAdd1 { .. } => need(90, 81, "multimem.red"),
    }
}

/// The buffers an operation reads or writes through, for the multicast check: the memory
/// operations, the copies, inline assembly's address operands and call arguments.
fn buffers_of(op: &Op) -> Vec<BufferId> {
    let addresses = |inputs: &[AsmInput]| -> Vec<BufferId> {
        inputs
            .iter()
            .filter_map(|i| match i {
                AsmInput::Address { buffer, .. } => Some(*buffer),
                AsmInput::Value(_) => None,
            })
            .collect()
    };
    match op {
        Op::Load { buffer, .. }
        | Op::Store { buffer, .. }
        | Op::Atomic { buffer, .. }
        | Op::Red { buffer, .. }
        | Op::Prefetch { buffer, .. }
        | Op::LdMatrix { buffer, .. }
        | Op::StMatrix { buffer, .. }
        | Op::BulkPrefetch { buffer, .. } => vec![*buffer],
        Op::CpAsync { src, dst, .. } | Op::CpAsyncBulk { src, dst, .. } => vec![*src, *dst],
        Op::TmaLoad { dst, .. } => vec![*dst],
        Op::TmaStore { src, .. } => vec![*src],
        Op::StAsync { buffer, .. } => vec![*buffer],
        Op::InlinePtx { inputs, .. } | Op::MlirOp { operands: inputs, .. } | Op::Call { args: inputs, .. } => {
            addresses(inputs)
        }
        _ => vec![],
    }
}

/// The address space a pointer into `space` is in, for a call argument.
fn pointer_space(space: MemSpace) -> AddressSpace {
    match space {
        MemSpace::Global => AddressSpace::Gmem,
        MemSpace::Shared => AddressSpace::Smem,
        MemSpace::Constant => AddressSpace::Cmem,
        MemSpace::Local => AddressSpace::Rmem,
        MemSpace::Raw(a) => a,
    }
}

impl Checker<'_> {
    /// The module-level checks: one declaration per external symbol, and a device function's
    /// restrictions.
    pub(super) fn externs_and_function(&self) -> Result {
        let ir = self.ir;
        for (i, d) in ir.externs.iter().enumerate() {
            if ir.externs[..i].iter().any(|e| e.symbol == d.symbol) {
                return Err(self.err(format!("external function `{}` is declared twice", d.symbol)));
            }
            if d.results.len() > 1 {
                return Err(self.err(format!(
                    "external function `{}` has {} results: a device function returns at most one value",
                    d.symbol,
                    d.results.len()
                )));
            }
            if d.symbol == ir.name {
                return Err(self.err(format!(
                    "external function `{}` has the name of the kernel that calls it (no recursion)",
                    d.symbol
                )));
            }
        }
        if ir.function.is_some() {
            device_function_restrictions(ir).map_err(|m| self.err(m))?;
        }
        for &b in &ir.multicast {
            let d = ir.buffer(b);
            if d.space != MemSpace::Global {
                return Err(self.err(format!("multicast buffer `{}` must be global memory", d.name)));
            }
        }
        Ok(())
    }

    /// An operation that is neither a call nor a multimem instruction must not touch a
    /// multicast buffer: the PTX ISA allows only `multimem.*` on multimem addresses.
    pub(super) fn multicast_access(&self, op: &Op) -> Result {
        if self.ir.multicast.is_empty() || matches!(op, Op::Multimem { .. }) {
            return Ok(());
        }
        for b in buffers_of(op) {
            if self.ir.is_multicast(b) {
                return Err(self.err(format!(
                    "{} on multicast buffer `{}`: a multimem (multicast) address takes only the multimem \
                     instructions (multimem_ld_reduce, multimem_st, multimem_red_add1); access a unicast \
                     pointer to the local replica instead",
                    op.name(),
                    self.ir.buffer(b).name
                )));
            }
        }
        Ok(())
    }

    /// An external call: the argument count, each argument against its parameter, the
    /// results, and the simulated body.
    pub(super) fn call(
        &mut self,
        callee: crate::ir::ExternId,
        args: &[AsmInput],
        results: &[crate::ir::ValueId],
        body: Option<&crate::ir::Region>,
    ) -> Result {
        let ir = self.ir;
        let Some(d) = ir.externs.get(callee.0 as usize) else {
            return Err(self.err(format!("call of undeclared external function #{}", callee.0)));
        };
        if args.len() != d.params.len() {
            return Err(self.err(format!(
                "external function `{}` takes {} arguments, but the call passes {}",
                d.symbol,
                d.params.len(),
                args.len()
            )));
        }
        for (k, (a, p)) in args.iter().zip(&d.params).enumerate() {
            match (a, p) {
                (AsmInput::Value(v), ExternParam::Value(t)) => {
                    if self.ty(*v) != *t {
                        return Err(self.err(format!(
                            "argument {k} of `{}` must be {t}, but {v} is {}",
                            d.symbol,
                            self.ty(*v)
                        )));
                    }
                }
                (AsmInput::Address { buffer, .. }, ExternParam::Ptr(want)) => {
                    let have = pointer_space(ir.buffer(*buffer).space);
                    let ok = match want {
                        AddressSpace::Generic => matches!(
                            have,
                            AddressSpace::Generic | AddressSpace::Gmem | AddressSpace::Smem | AddressSpace::Rmem
                        ),
                        w => *w == have,
                    };
                    if !ok {
                        return Err(self.err(format!(
                            "argument {k} of `{}` must be a pointer to {:?} memory, but it points into `{}` ({:?})",
                            d.symbol,
                            want,
                            ir.buffer(*buffer).name,
                            have
                        )));
                    }
                }
                (AsmInput::Value(v), ExternParam::Ptr(_)) => {
                    return Err(self.err(format!(
                        "argument {k} of `{}` is a pointer parameter, but the call passes value {v} ({})",
                        d.symbol,
                        self.ty(*v)
                    )));
                }
                (AsmInput::Address { .. }, ExternParam::Value(t)) => {
                    return Err(self.err(format!(
                        "argument {k} of `{}` must be a {t} value, but the call passes a pointer",
                        d.symbol
                    )));
                }
            }
        }
        if results.len() != d.results.len() || results.iter().zip(&d.results).any(|(r, t)| self.ty(*r) != *t) {
            return Err(
                self.err(format!("the call of `{}` does not have the declared results ({:?})", d.symbol, d.results))
            );
        }
        if let Some(r) = body {
            self.region(r, &[])?;
            self.yields_match(&r.yields, results, "a simulated call body")?;
        }
        Ok(())
    }

    /// A multimem instruction: a multicast buffer of global memory, register counts and types,
    /// and an aligned start.
    pub(super) fn multimem(
        &self,
        op: &MultimemOp,
        buffer: BufferId,
        values: &[crate::ir::ValueId],
        results: &[crate::ir::ValueId],
    ) -> Result {
        let d = self.ir.buffer(buffer);
        if !self.ir.is_multicast(buffer) {
            return Err(self.err(format!(
                "{} on `{}`, which is not a multicast buffer: multimem instructions take a multimem address \
                 (param_tensor_multicast or param_ptr_multicast)",
                op.ptx(),
                d.name
            )));
        }
        let (want_values, want_results) = match op {
            MultimemOp::LdReduce { regs, .. } => (0, *regs as usize),
            MultimemOp::St { regs } => (*regs as usize, 0),
            MultimemOp::RedAdd1 { .. } => (0, 0),
        };
        if values.len() != want_values || results.len() != want_results {
            return Err(self.err(format!(
                "{} takes {want_values} registers and returns {want_results}, not {} and {}",
                op.ptx(),
                values.len(),
                results.len()
            )));
        }
        for &v in values.iter().chain(results) {
            if self.ty(v).bits() != 32 || self.ty(v) == DType::Bool {
                return Err(self.err(format!("{} works on 32-bit registers, not {}", op.ptx(), self.ty(v))));
            }
        }
        self.base_alignment(buffer, op.bytes(), "a multimem access")
    }
}

/// What a standalone device function may not have: shared memory, a block size, by-value or
/// tensor-map parameters, module globals and multicast buffers.
fn device_function_restrictions(ir: &KernelIr) -> std::result::Result<(), String> {
    let name = &ir.name;
    if ir.buffers.iter().any(|b| b.space == MemSpace::Shared) {
        return Err(format!(
            "device function `{name}` allocates shared memory: a device function cannot (its callers' kernels own \
             their shared memory); pass a shared-memory pointer instead"
        ));
    }
    if ir.block_dim.is_some() || ir.max_threads.is_some() || ir.min_blocks_per_sm.is_some() {
        return Err(format!(
            "device function `{name}` declares a block size or launch bounds: those belong to kernels"
        ));
    }
    if ir.args.iter().any(|p| matches!(p, Param::ByValue { .. })) || !ir.tensor_maps.is_empty() {
        return Err(format!("device function `{name}` takes a by-value parameter: it takes values and pointers only"));
    }
    if !ir.globals.is_empty() {
        return Err(format!("device function `{name}` declares a module global: declare it in the kernel's module"));
    }
    if !ir.multicast.is_empty() {
        return Err(format!("device function `{name}` takes a multicast parameter: pass a pointer from the kernel"));
    }
    Ok(())
}
