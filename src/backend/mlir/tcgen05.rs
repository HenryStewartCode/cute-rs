// SPDX-License-Identifier: MIT OR Apache-2.0

//! MLIR for the `tcgen05` instructions.
//!
//! `tcgen05.alloc`, `tcgen05.dealloc` and `tcgen05.relinquish_alloc_permit`
//! are warp-collective, so they are LLVM 21's NVVM operations, whose
//! intrinsics are convergent, rather than inline assembly. The sm_107
//! `.exclusive` forms (PTX ISA 9.4) are known to no installed LLVM, so they
//! are inline PTX, which LLVM does not treat as convergent.
//!
//! m10-tmem-ld-st adds `tcgen05.ld`, `tcgen05.st`, `tcgen05.wait::{ld,st}`
//! and the thread-sync fences, again as NVVM operations (their intrinsics
//! are convergent). The syntax is the same in LLVM 21, 22 and 23, except
//! that a single register is an `i32` before LLVM 23 and a `vector<1xi32>`
//! from LLVM 23.
//!
//! m10-umma-f16 adds `tcgen05.mma` and `tcgen05.commit`. LLVM 21 has no
//! `nvvm.tcgen05.mma` (LLVM 23 does), so the MMA is inline PTX: the warp
//! computes its elected lane (the IR's `elect` operand) and the
//! instruction descriptor (a constant, so the PTX shows it as an
//! immediate move), and the assembly issues the MMA from that lane alone
//! (`@p`), with `enable-input-d` as a predicate. Issue is single-thread, so
//! the assembly not being convergent does not matter. `tcgen05.commit` is
//! LLVM 21's `nvvm.tcgen05.commit` (with `multicast_mask` for a mask).
//!
//! **Thread-sync fences.** The PTX ISA's tcgen05 memory model (9.7.18.6)
//! wants `tcgen05.fence::after_thread_sync` between a thread
//! synchronisation and a later tcgen05 operation, and
//! `tcgen05.fence::before_thread_sync` between a tcgen05 operation and a
//! later synchronisation. CuTe DSL code never writes them (its compiler
//! places them), so in a kernel that loads or stores tensor memory the
//! backend puts `before_thread_sync` before every barrier, named-barrier
//! arrive or sync, mbarrier arrive and cluster arrive, and
//! `after_thread_sync` after every barrier, named-barrier sync, mbarrier
//! wait and cluster wait. The fences are cheap, and the simulator needs to
//! check only the `tcgen05.wait` rules. Kernels without TMEM loads or
//! stores (only allocation, say) are emitted unchanged.

use super::Emitter;
use crate::ir::{BarrierKind, DType, Inst, Op, Tcgen05Inst};

/// Whether `op` gets a `tcgen05.fence::before_thread_sync` before it and an
/// `after_thread_sync` after it, in a kernel that loads or stores TMEM.
pub(super) fn thread_sync_fences(op: &Op) -> (bool, bool) {
    match op {
        Op::Barrier => (true, true),
        Op::NamedBarrier { kind: BarrierKind::Sync, .. } | Op::BarrierRed { .. } => (true, true),
        Op::NamedBarrier { .. } | Op::MbarrierArrive { .. } | Op::ClusterArrive { .. } => (true, false),
        // The arrivals among the extra mbarrier operations.
        Op::MbarrierExt { kind, .. } if kind.has_state() => (true, false),
        Op::MbarrierWait { .. } | Op::ClusterWait { .. } => (false, true),
        _ => (false, false),
    }
}

impl Emitter<'_> {
    pub(super) fn tcgen05(&mut self, indent: usize, inst: &Inst) {
        let Op::Tcgen05 { inst: which, two_cta, exclusive, buffer, args } = &inst.op else {
            unreachable!("a tcgen05 instruction")
        };
        let cg = if *two_cta { 2 } else { 1 };
        // `cta_group::1` is the operations' default, written without an attribute; the
        // attribute for `cta_group::2` is `#nvvm.tcgen05_group` in LLVM 21 and
        // `#nvvm.cta_group` from LLVM 22.
        let attr = match (*two_cta, self.opts.llvm_major) {
            (false, _) => String::new(),
            (true, Some(v)) if v >= 22 => " {group = #nvvm.cta_group<cta_2>}".to_string(),
            (true, _) => " {group = #nvvm.tcgen05_group<cta_2>}".to_string(),
        };
        match which {
            Tcgen05Inst::Alloc => {
                let buffer = buffer.expect("tcgen05.alloc writes to a shared buffer");
                let addr = self.address(indent, buffer, DType::I32, args[0]);
                let ncols = self.v(args[1]);
                if *exclusive {
                    let template =
                        format!("tcgen05.alloc.exclusive.cta_group::{cg}.sync.aligned.shared::cta.b32 [$0], $1;");
                    self.ptx_inst(indent, inst, &template, &[], &[(addr, DType::I64), (ncols, DType::I32)], true);
                } else {
                    let ptr = self.fresh("ptr");
                    self.line(indent, &format!("{ptr} = llvm.inttoptr {addr} : i64 to !llvm.ptr<3>"), None);
                    let text = format!("nvvm.tcgen05.alloc {ptr}, {ncols}{attr} : !llvm.ptr<3>, i32");
                    self.line(indent, &text, Some(inst));
                }
            }
            Tcgen05Inst::Dealloc => {
                let (taddr, ncols) = (self.v(args[0]), self.v(args[1]));
                if *exclusive {
                    let template = format!("tcgen05.dealloc.exclusive.cta_group::{cg}.sync.aligned.b32 $0, $1;");
                    self.ptx_inst(indent, inst, &template, &[], &[(taddr, DType::I32), (ncols, DType::I32)], true);
                } else {
                    let ptr = self.fresh("tmem");
                    self.line(indent, &format!("{ptr} = llvm.inttoptr {taddr} : i32 to !llvm.ptr<6>"), None);
                    let text = format!("nvvm.tcgen05.dealloc {ptr}, {ncols}{attr} : !llvm.ptr<6>, i32");
                    self.line(indent, &text, Some(inst));
                }
            }
            Tcgen05Inst::RelinquishAllocPermit => {
                self.line(indent, &format!("nvvm.tcgen05.relinquish_alloc_permit{attr}"), Some(inst));
            }
            // A check for the simulator only.
            Tcgen05Inst::CheckAddress => {}
            Tcgen05Inst::WaitLd => self.line(indent, "nvvm.tcgen05.wait <load>", Some(inst)),
            Tcgen05Inst::WaitSt => self.line(indent, "nvvm.tcgen05.wait <store>", Some(inst)),
            Tcgen05Inst::FenceBeforeThreadSync => self.line(indent, "nvvm.tcgen05.fence <before>", Some(inst)),
            Tcgen05Inst::FenceAfterThreadSync => self.line(indent, "nvvm.tcgen05.fence <after>", Some(inst)),
        }
    }

    /// `tcgen05.ld` and `tcgen05.st` (`nvvm.tcgen05.ld` and
    /// `nvvm.tcgen05.st`): the registers travel as one `vector<Nxi32>` (an
    /// `i32` for one register before LLVM 23), bitcast from or to the
    /// vector of their elements.
    pub(super) fn tcgen05_ldst(&mut self, indent: usize, inst: &Inst) {
        if let Op::Tcgen05Ld { results, taddr, red, sp: Some(sp), .. } = &inst.op {
            return self.tcgen05_ld_spcompress(indent, inst, results, *taddr, red.as_ref(), sp);
        }
        if let Op::Tcgen05Ld { results, taddr, shape, num, red: Some(red), .. } = &inst.op {
            return self.tcgen05_ld_red(indent, inst, results, *taddr, *shape, *num, red);
        }
        let (elems, taddr, shape, num, packed, load) = match &inst.op {
            Op::Tcgen05Ld { results, taddr, shape, num, pack, .. } => (results, *taddr, *shape, *num, *pack, true),
            Op::Tcgen05St { values, taddr, shape, num, unpack } => (values, *taddr, *shape, *num, *unpack, false),
            _ => unreachable!("tcgen05.ld or tcgen05.st"),
        };
        let n = shape.registers(num) as usize;
        let m = elems.len();
        let regs_ty = if n == 1 && self.opts.llvm_major.is_none_or(|v| v < 23) {
            "i32".to_string()
        } else {
            format!("vector<{n}xi32>")
        };
        let et = self.ir.dtype(elems[0]).mlir();
        let elems_ty = if m == 1 { et.to_string() } else { format!("vector<{m}x{et}>") };
        let ptr = self.fresh("tmem");
        let addr = self.v(taddr);
        self.line(indent, &format!("{ptr} = llvm.inttoptr {addr} : i32 to !llvm.ptr<6>"), None);
        let offset = match shape.half_split(num, packed) {
            Some(imm) => {
                let c = self.fresh("split");
                self.line(indent, &format!("{c} = llvm.mlir.constant({imm} : i64) : i64"), None);
                format!(", {c}")
            }
            None => String::new(),
        };
        let attrs = format!("{{shape = #nvvm.tcgen05_ldst_shape<shape_{}>}}", shape.ptx());
        if load {
            let flag = if packed { " pack" } else { "" };
            let direct = m == 1 && regs_ty == elems_ty;
            let ld = if direct { self.v(elems[0]) } else { self.fresh("ld") };
            self.line(indent, &format!("{ld} = nvvm.tcgen05.ld {ptr}{offset}{flag} {attrs} : {regs_ty}"), Some(inst));
            if direct {
                return;
            }
            if m == 1 {
                let text = format!("{} = llvm.bitcast {ld} : {regs_ty} to {elems_ty}", self.v(elems[0]));
                self.line(indent, &text, None);
                return;
            }
            let vec = if regs_ty == elems_ty {
                ld
            } else {
                let c = self.fresh("ld");
                self.line(indent, &format!("{c} = llvm.bitcast {ld} : {regs_ty} to {elems_ty}"), None);
                c
            };
            for (k, &r) in elems.iter().enumerate() {
                let text = format!("{} = vector.extract {vec}[{k}] : {et} from {elems_ty}", self.v(r));
                self.line(indent, &text, None);
            }
        } else {
            let flag = if packed { " unpack" } else { "" };
            let mut value = if m == 1 {
                self.v(elems[0])
            } else {
                let mut acc = self.fresh("st");
                self.line(indent, &format!("{acc} = llvm.mlir.poison : {elems_ty}"), None);
                for (k, &r) in elems.iter().enumerate() {
                    let next = self.fresh("st");
                    let text = format!("{next} = vector.insert {}, {acc} [{k}] : {et} into {elems_ty}", self.v(r));
                    self.line(indent, &text, None);
                    acc = next;
                }
                acc
            };
            if regs_ty != elems_ty {
                let c = self.fresh("st");
                self.line(indent, &format!("{c} = llvm.bitcast {value} : {elems_ty} to {regs_ty}"), None);
                value = c;
            }
            self.line(indent, &format!("nvvm.tcgen05.st {ptr}, {value}{offset}{flag} {attrs} : {regs_ty}"), Some(inst));
        }
    }

    /// `tcgen05.ld{.red}.spcompress.32x32b` (PTX ISA 9.4, sm_107a), as inline PTX (no LLVM has an `nvvm` op for it):
    /// `mdata`, `cdata`{, `redval`}, `[taddr]`. The qualifier spellings are
    /// cute-rs's reading of the PTX syntax and have not been assembled.
    fn tcgen05_ld_spcompress(
        &mut self,
        indent: usize,
        inst: &Inst,
        cdata: &[crate::ir::ValueId],
        taddr: crate::ir::ValueId,
        red: Option<&crate::ir::TmemLdRed>,
        sp: &crate::ir::TmemLdSpCompress,
    ) {
        let num = cdata.len() * 2;
        let mut results: Vec<(String, DType)> = Vec::new();
        let mut regs = |ids: &[crate::ir::ValueId], this: &Self| -> Vec<String> {
            ids.iter()
                .map(|&v| {
                    results.push((this.v(v), this.ir.dtype(v)));
                    format!("${}", results.len() - 1)
                })
                .collect()
        };
        let m = regs(&sp.mdata, self);
        let c = regs(cdata, self);
        let r = red.map(|r| regs(&[r.result], self));
        let addr = format!("${}", results.len());
        let template = format!(
            "tcgen05.ld{}.spcompress.sync.aligned.32x32b.x{num}{}.sp2of4{}{}.f32.idx4 {{{}}}, {{{}}}{}, [{addr}];",
            if red.is_some() { ".red" } else { "" },
            if sp.op.is_min() { ".min" } else { ".max" },
            if sp.op.is_abs() { ".abs" } else { "" },
            if sp.nan { ".NaN" } else { "" },
            m.join(", "),
            c.join(", "),
            r.map(|r| format!(", {}", r[0])).unwrap_or_default(),
        );
        let inputs = vec![(self.v(taddr), DType::I32)];
        self.ptx_inst(indent, inst, &template, &results, &inputs, true);
    }

    /// `tcgen05.ld.red` (`nvvm.tcgen05.ld.red`, LLVM 23): the Float32 data as one `vector<Nxf32>`, then
    /// the Float32 reduction; the PTX ISA's `.red` forms need PTX 8.8.
    #[allow(clippy::too_many_arguments)]
    fn tcgen05_ld_red(
        &mut self,
        indent: usize,
        inst: &Inst,
        elems: &[crate::ir::ValueId],
        taddr: crate::ir::ValueId,
        shape: crate::ir::TmemShape,
        num: u32,
        red: &crate::ir::TmemLdRed,
    ) {
        let _ = num;
        let ptr = self.fresh("tmem");
        let addr = self.v(taddr);
        self.line(indent, &format!("{ptr} = llvm.inttoptr {addr} : i32 to !llvm.ptr<6>"), None);
        let offset = match red.half_split {
            Some(imm) => {
                let c = self.fresh("split");
                self.line(indent, &format!("{c} = llvm.mlir.constant({imm} : i64) : i64"), None);
                format!(", {c}")
            }
            None => String::new(),
        };
        let kind = if red.op.is_min() { "min" } else { "max" };
        let mut attrs = format!("shape = #nvvm.tcgen05_ldst_shape<shape_{}>", shape.ptx());
        if red.op.is_abs() {
            attrs.push_str(", abs");
        }
        if red.nan {
            attrs.push_str(", nan");
        }
        let n = elems.len();
        let vec = self.fresh("ld");
        let text = format!(
            "{vec}, {} = nvvm.tcgen05.ld.red {kind} {ptr}{offset} {{{attrs}}} : vector<{n}xf32>, f32",
            self.v(red.result)
        );
        self.line(indent, &text, Some(inst));
        for (k, &r) in elems.iter().enumerate() {
            let text = format!("{} = vector.extract {vec}[{k}] : f32 from vector<{n}xf32>", self.v(r));
            self.line(indent, &text, None);
        }
    }

    /// `tcgen05.mma`, as inline PTX issued by
    /// the elected lane. m10-umma-kinds: A from TMEM (`[a-tmem]`), a runtime
    /// instruction descriptor, and `{disable-output-lane}`; an MMA with none
    /// of them is emitted as before.
    pub(super) fn tcgen05_mma(&mut self, indent: usize, inst: &Inst) {
        let Op::Tcgen05Mma {
            kind,
            taddr,
            desc_a,
            desc_b,
            enable_input_d,
            elect,
            idesc,
            disable_lane,
            sf,
            meta,
            zero_col_mask,
        } = &inst.op
        else {
            unreachable!("a tcgen05.mma")
        };
        let idesc = match idesc {
            Some(v) => self.v(*v),
            None => {
                let c = self.fresh("idesc");
                self.line(indent, &format!("{c} = arith.constant {} : i32", kind.idesc() as i32), None);
                c
            }
        };
        let lead = self.fresh("elect");
        self.line(indent, &format!("{lead} = arith.extui {} : i1 to i32", self.v(*elect)), None);
        let acc = self.fresh("accum");
        self.line(indent, &format!("{acc} = arith.extui {} : i1 to i32", self.v(*enable_input_d)), None);
        let a = if kind.a_tmem { "[$1]" } else { "$1" };
        let mut inputs = vec![
            (self.v(*taddr), DType::I32),
            (self.v(*desc_a), if kind.a_tmem { DType::I32 } else { DType::U64 }),
            (self.v(*desc_b), DType::U64),
            (idesc, DType::I32),
            (lead, DType::I32),
            (acc, DType::I32),
        ];
        // The operands after the six above, in PTX order (verified against
        // LLVM 23's NVPTX printer and CUTLASS's sparse block-scaled forms):
        // `[d], a, b, [sp-meta], idesc, [sfa], [sfb], {disable-output-lane}, enable-input-d,
        // scale-input-d | zero-column-mask`.
        let next = |inputs: &mut Vec<(String, DType)>, v: String, t: DType| {
            inputs.push((v, t));
            format!("${}", inputs.len() - 1)
        };
        // m10-sparse-umma: `[sp-meta-tmem]` before the descriptor.
        let sp = match meta {
            Some(e) => format!(" [{}],", next(&mut inputs, self.v(*e), DType::I32)),
            None => String::new(),
        };
        // m10-umma-blockscaled: `[sfa-tmem], [sfb-tmem]` after the descriptor.
        let scales = match sf {
            Some((x, y)) => {
                let x = next(&mut inputs, self.v(*x), DType::I32);
                let y = next(&mut inputs, self.v(*y), DType::I32);
                format!(" [{x}], [{y}],")
            }
            None => String::new(),
        };
        let mask = if disable_lane.is_empty() {
            String::new()
        } else {
            let regs: Vec<String> = disable_lane.iter().map(|&w| next(&mut inputs, self.v(w), DType::I32)).collect();
            format!(" {{{}}},", regs.join(", "))
        };
        // The immediate scale-input-d, or a .ws MMA's zero-column mask.
        let tail = match (kind.prims.scale_input_d, zero_col_mask) {
            (Some(s), _) => format!(", {s}"),
            (None, Some(z)) => format!(", {}", next(&mut inputs, self.v(*z), DType::U64)),
            (None, None) => String::new(),
        };
        let template = format!(
            "{{ .reg .pred p, q; setp.ne.b32 p, $4, 0; setp.ne.b32 q, $5, 0; @p {} [$0], {a}, $2,{sp} $3,{scales}{mask} q{tail}; }}",
            kind.ptx()
        );
        self.asm_memory(indent, inst, &template, &inputs);
    }

    /// `tcgen05.cp`, as inline PTX issued by
    /// the elected lane (as `tcgen05.mma` is).
    pub(super) fn tcgen05_cp(&mut self, indent: usize, inst: &Inst) {
        let Op::Tcgen05Cp { shape, two_cta, taddr, desc, elect } = &inst.op else { unreachable!("a tcgen05.cp") };
        let lead = self.fresh("elect");
        self.line(indent, &format!("{lead} = arith.extui {} : i1 to i32", self.v(*elect)), None);
        let template = format!(
            "{{ .reg .pred p; setp.ne.b32 p, $2, 0; @p tcgen05.cp.cta_group::{}.{} [$0], $1; }}",
            if *two_cta { 2 } else { 1 },
            shape.ptx()
        );
        let inputs = vec![(self.v(*taddr), DType::I32), (self.v(*desc), DType::U64), (lead, DType::I32)];
        self.asm_memory(indent, inst, &template, &inputs);
    }

    /// `tcgen05.commit` (`nvvm.tcgen05.commit`, with a multicast mask when
    /// given).
    pub(super) fn tcgen05_commit(&mut self, indent: usize, inst: &Inst) {
        let Op::Tcgen05Commit { buffer, index, mask, two_cta } = &inst.op else { unreachable!("a tcgen05.commit") };
        let sum = self.address(indent, *buffer, DType::I64, *index);
        let ptr = self.fresh("ptr");
        self.line(indent, &format!("{ptr} = llvm.inttoptr {sum} : i64 to !llvm.ptr<3>"), None);
        let attr = match (*two_cta, self.opts.llvm_major) {
            (false, _) => String::new(),
            (true, Some(v)) if v >= 22 => " {group = #nvvm.cta_group<cta_2>}".to_string(),
            (true, _) => " {group = #nvvm.tcgen05_group<cta_2>}".to_string(),
        };
        let text = match mask {
            None => format!("nvvm.tcgen05.commit {ptr}{attr} : !llvm.ptr<3>"),
            Some(m) => format!("nvvm.tcgen05.commit {ptr}, multicast_mask = {}{attr} : !llvm.ptr<3>, i16", self.v(*m)),
        };
        self.line(indent, &text, Some(inst));
    }
}
