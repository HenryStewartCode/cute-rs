//! The MLIR of the operations of roadmap part m11-prims-nvvm-layer
//! (`cute::prims`): `barrier.cta.red`, the extra mbarrier operations, the
//! markers and hints, `exit`, `st.bulk` and `tcgen05.shift`.
//!
//! The per-thread instructions are inline PTX with side effects, which
//! keeps them the same from one LLVM release to the next (the NVVM ops and
//! intrinsics of the mbarrier family are spelled differently in LLVM 21, 22
//! and 23). The collective ones (`bar.red`, `tcgen05.shift`) call the
//! convergent LLVM intrinsics, except `bar.red` with LLVM 21, which has none
//! for a barrier other than 0 or a thread count.

use super::{Emitter, escape};
use crate::ir::{BarrierRedKind, DType, Inst, MbarrierExtKind, MbarrierWaitKind, Op, PrimMarker};

impl Emitter<'_> {
    /// An inline-PTX statement with side effects: `ptx` with operands
    /// `$0, ...` of `constraints`, the result (if any) named `result`.
    fn prims_asm(
        &mut self,
        indent: usize,
        inst: &Inst,
        ptx: &str,
        constraints: &str,
        args: &[(String, &str)],
        result: Option<(String, &str)>,
    ) {
        let types = args.iter().map(|(_, t)| *t).collect::<Vec<_>>().join(", ");
        let names = args.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ");
        let (lhs, ret) = match &result {
            Some((name, ty)) => (format!("{name} = "), ty.to_string()),
            None => (String::new(), "()".to_string()),
        };
        let text = format!(
            "{lhs}llvm.inline_asm has_side_effects \"{}\", \"{}\" {names} : ({types}) -> {ret}",
            escape(ptx),
            escape(constraints)
        );
        self.line(indent, &text, Some(inst));
    }

    pub(super) fn prims_op(&mut self, indent: usize, inst: &Inst) {
        match &inst.op {
            Op::BarrierRed { result, kind, pred, id, threads } => {
                let (r, p, id) = (self.v(*result), self.v(*pred), self.v(*id));
                let n = threads.map(|n| self.v(n));
                let ret = if *kind == BarrierRedKind::Popc { "i32" } else { "i1" };
                if self.opts.llvm_major.is_some_and(|v| v >= 22) {
                    // LLVM 22's convergent `llvm.nvvm.barrier.cta.red.*` intrinsics.
                    let text = match n {
                        None => format!(
                            "{r} = llvm.call_intrinsic \"llvm.nvvm.barrier.cta.red.{}.aligned.all\"({id}, {p}) : (i32, i1) -> {ret}",
                            kind.name()
                        ),
                        Some(n) => format!(
                            "{r} = llvm.call_intrinsic \"llvm.nvvm.barrier.cta.red.{}.aligned.count\"({id}, {n}, {p}) : (i32, i32, i1) -> {ret}",
                            kind.name()
                        ),
                    };
                    self.line(indent, &text, Some(inst));
                } else {
                    // LLVM 21 has `llvm.nvvm.barrier0.{popc,and,or}` for barrier 0
                    // without a count only; `bar.red` in inline PTX covers every form.
                    let out = if *kind == BarrierRedKind::Popc { "=r" } else { "=b" };
                    let (ptx, constraints, args) = match n {
                        None => (
                            format!("bar.red.{} $0, $1, $2;", kind.ptx()),
                            format!("{out},r,b,~{{memory}}"),
                            vec![(id, "i32"), (p, "i1")],
                        ),
                        Some(n) => (
                            format!("bar.red.{} $0, $1, $2, $3;", kind.ptx()),
                            format!("{out},r,r,b,~{{memory}}"),
                            vec![(id, "i32"), (n, "i32"), (p, "i1")],
                        ),
                    };
                    self.prims_asm(indent, inst, &ptx, &constraints, &args, Some((r, ret)));
                }
            }
            Op::MbarrierExt { kind, result, buffer, index, value } => {
                let p = self.address(indent, *buffer, DType::I64, *index);
                let mut args = vec![(p, "i64")];
                if let Some(v) = value {
                    args.push((self.v(*v), "i32"));
                }
                let space = match kind {
                    MbarrierExtKind::ArriveDropExpectTx | MbarrierExtKind::CompleteTx => ".shared::cta",
                    _ => ".shared",
                };
                let op = kind.ptx();
                let tail = if value.is_some() { if result.is_some() { ", $2" } else { ", $1" } } else { "" };
                match result {
                    Some(r) => {
                        let ptx = format!("{op}{space}.b64 $0, [$1]{tail};");
                        let constraints = if value.is_some() { "=l,l,r,~{memory}" } else { "=l,l,~{memory}" };
                        let r = self.v(*r);
                        self.prims_asm(indent, inst, &ptx, constraints, &args, Some((r, "i64")));
                    }
                    None => {
                        let ptx = format!("{op}{space}.b64 [$0]{tail};");
                        let constraints = if value.is_some() { "l,r,~{memory}" } else { "l,~{memory}" };
                        self.prims_asm(indent, inst, &ptx, constraints, &args, None);
                    }
                }
            }
            Op::PrimMarker { marker } => match marker {
                PrimMarker::Nanosleep(t) => {
                    let t = self.v(*t);
                    self.prims_asm(indent, inst, "nanosleep.u32 $0;", "r", &[(t, "i32")], None);
                }
                PrimMarker::Pmevent(mask) => {
                    self.prims_asm(indent, inst, &format!("pmevent.mask {mask:#x};"), "", &[], None)
                }
                PrimMarker::Breakpoint => self.prims_asm(indent, inst, "brkpt;", "", &[], None),
                PrimMarker::Comment(text) => {
                    // `$` is LLVM's operand marker: `$$` prints one.
                    let ptx = format!("// {}", text.replace('$', "$$"));
                    self.prims_asm(indent, inst, &ptx, "", &[], None)
                }
            },
            // `llvm.nvvm.exit` is `noreturn`, which an `scf` region cannot end
            // with; the statement leaves the control flow to the PTX.
            Op::Exit => self.prims_asm(indent, inst, "exit;", "~{memory}", &[], None),
            Op::StBulk { buffer, index, elem, size } => {
                let p = self.address(indent, *buffer, *elem, *index);
                let n = self.v(*size);
                self.prims_asm(
                    indent,
                    inst,
                    "st.bulk.weak.shared::cta [$0], $1, 0;",
                    "l,l,~{memory}",
                    &[(p, "i64"), (n, "i64")],
                    None,
                );
            }
            Op::Tcgen05Shift { taddr, two_cta } => {
                let ptr = self.fresh("tptr");
                let a = self.v(*taddr);
                self.line(indent, &format!("{ptr} = llvm.inttoptr {a} : i32 to !llvm.ptr<6>"), None);
                let text = format!(
                    "llvm.call_intrinsic \"llvm.nvvm.tcgen05.shift.down.cg{}\"({ptr}) : (!llvm.ptr<6>) -> ()",
                    1 + *two_cta as u8
                );
                self.line(indent, &text, Some(inst));
            }
            Op::MbarrierWait { result, buffer, index, parity, kind, state, time_limit } => {
                // The forms the plain lowering does not cover: a state token or
                // a suspend-time hint.
                let p = self.address(indent, *buffer, DType::I64, *index);
                let (r, token) = (self.v(*result), self.v(*parity));
                let (op, space) = match kind {
                    MbarrierWaitKind::Try => ("mbarrier.try_wait", ".shared::cta"),
                    MbarrierWaitKind::Test => ("mbarrier.test_wait", ".shared"),
                };
                let (parity_q, token_ty, token_c) = if *state { ("", "i64", "l") } else { (".parity", "i32", "r") };
                let mut args = vec![(p, "i64"), (token, token_ty)];
                let mut constraints = format!("=b,l,{token_c}");
                let mut ptx = format!("{op}{parity_q}{space}.b64 $0, [$1], $2");
                if let Some(t) = time_limit {
                    args.push((self.v(*t), "i32"));
                    constraints.push_str(",r");
                    ptx.push_str(", $3");
                }
                ptx.push(';');
                self.prims_asm(indent, inst, &ptx, &constraints, &args, Some((r, "i1")));
            }
            _ => unreachable!("not a prims operation"),
        }
    }
}
