// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/mma_traits_sm90_gmma.hpp (warpgroup_fence_operand),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! MLIR for the warpgroup MMA instructions (the FP8 and integer forms, whose immediates differ,
//! and the register form, whose A is four 32-bit registers of packed
//! elements, which the operand fences pass through too).
//!
//! `wgmma.fence`, `wgmma.commit_group` and `wgmma.wait_group` are the NVVM
//! operations whose intrinsics LLVM 21 to 23 have. `wgmma.mma_async` has no
//! intrinsic (NVVM's `nvvm.wgmma.mma_async` becomes inline assembly in
//! `convert-nvvm-to-llvm`), so it is written as inline assembly here, the
//! accumulators tied to its outputs. LLVM then sees the accumulators as
//! ordinary values, which it might copy or materialise late (a zero fill
//! after the `wgmma.fence`, say): ptxas then serialises the MMAs. The
//! operand fences, an empty statement that reads and writes every
//! accumulator (CUTLASS's `warpgroup_fence_operand`), pin them before the
//! fence and after the wait.
//!
//! Float16 accumulators are `.f16x2` registers in PTX: pairs are packed
//! into `i32`s for the instruction and the operand fences, so that LLVM can
//! fold an unpack followed by a pack away.

use super::Emitter;
use crate::ir::{DType, Inst, Op, ValueId};

impl Emitter<'_> {
    pub(super) fn wgmma(&mut self, indent: usize, inst: &Inst) {
        match &inst.op {
            Op::Wgmma { kind, d, c, a, desc_b, scale_d } => {
                let f16 = kind.acc == DType::F16;
                let regs = if f16 { d.len() / 2 } else { d.len() };
                let scale = self.fresh("scaled");
                self.line(indent, &format!("{scale} = arith.extui {} : i1 to i32", self.v(*scale_d)), None);
                let tied = if f16 { self.pack_pairs(indent, c) } else { c.iter().map(|&v| self.v(v)).collect() };
                let reg_ty = if f16 { DType::U32 } else { kind.acc };
                let outs: Vec<String> = if f16 {
                    (0..regs).map(|_| self.fresh("wgd")).collect()
                } else {
                    d.iter().map(|&v| self.v(v)).collect()
                };
                let list = (0..regs).map(|i| format!("${i}")).collect::<Vec<_>>().join(", ");
                // The A operand: a descriptor, or four 32-bit registers of
                // packed elements.
                let a_words = if kind.a_regs { self.pack_words(indent, a) } else { vec![self.v(a[0])] };
                let a_text = if kind.a_regs {
                    format!("{{{}}}", (0..4).map(|i| format!("${}", regs + i)).collect::<Vec<_>>().join(", "))
                } else {
                    format!("${regs}")
                };
                let (b_at, scale_at) = (regs + a_words.len(), regs + a_words.len() + 1);
                // The immediates after scale-d: imm-scale-a and imm-scale-b
                // (floats), then imm-trans-a (shared-memory A) and
                // imm-trans-b (16-bit types).
                let mut imms = Vec::new();
                if !kind.is_integer() {
                    imms.extend(["1".to_string(), "1".to_string()]);
                }
                if kind.is_16bit() {
                    if !kind.a_regs {
                        imms.push((kind.trans_a as u8).to_string());
                    }
                    imms.push((kind.trans_b as u8).to_string());
                }
                let imms: String = imms.iter().map(|i| format!(", {i}")).collect();
                let template = format!(
                    "{{\n.reg .pred p;\nsetp.ne.b32 p, ${scale_at}, 0;\n{} {{{list}}}, {a_text}, ${b_at}, p{imms};\n}}",
                    kind.ptx(),
                );
                let reg = if f16 { "r" } else { super::constraint(kind.acc) };
                let a_constraint = if kind.a_regs { "r" } else { "l" };
                let constraints: Vec<String> = (0..regs)
                    .map(|_| format!("={reg}"))
                    .chain(a_words.iter().map(|_| a_constraint.to_string()))
                    .chain(["l".to_string(), "r".to_string()])
                    .chain((0..regs).map(|i| i.to_string()))
                    .collect();
                let a_ty = if kind.a_regs { "i32" } else { "i64" };
                let mut args: Vec<(String, &str)> = a_words.into_iter().map(|w| (w, a_ty)).collect();
                args.extend([(self.v(*desc_b), "i64"), (scale, "i32")]);
                args.extend(tied.into_iter().map(|t| (t, reg_ty.mlir())));
                let results: Vec<(String, DType)> = outs.iter().map(|o| (o.clone(), reg_ty)).collect();
                self.asm_line(indent, inst, &template, &constraints.join(","), &results, &args, true);
                if f16 {
                    self.unpack_pairs(indent, &outs, d);
                }
            }
            Op::WgmmaFence { results, inputs } => {
                self.operand_fence(indent, inst, results, inputs);
                self.line(indent, "nvvm.wgmma.fence.aligned", Some(inst));
            }
            Op::WgmmaCommitGroup => self.line(indent, "nvvm.wgmma.commit.group.sync.aligned", Some(inst)),
            Op::WgmmaWait { pending, results, inputs, .. } => {
                self.line(indent, &format!("nvvm.wgmma.wait.group.sync.aligned {pending}"), Some(inst));
                self.operand_fence(indent, inst, results, inputs);
            }
            _ => unreachable!("not a WGMMA instruction"),
        }
    }

    /// `i32`s holding the Float16 pairs `values[2i], values[2i + 1]`.
    fn pack_pairs(&mut self, indent: usize, values: &[ValueId]) -> Vec<String> {
        values
            .chunks(2)
            .map(|pair| {
                let (vec, word) = (self.fresh("h2"), self.fresh("h2w"));
                let parts = pair.iter().map(|&v| self.v(v)).collect::<Vec<_>>().join(", ");
                self.line(indent, &format!("{vec} = vector.from_elements {parts} : vector<2xf16>"), None);
                self.line(indent, &format!("{word} = llvm.bitcast {vec} : vector<2xf16> to i32"), None);
                word
            })
            .collect()
    }

    /// `i32`s holding `values` packed four 8-bit or two 16-bit elements to
    /// a word, the first in the low bits (an A fragment in registers).
    fn pack_words(&mut self, indent: usize, values: &[ValueId]) -> Vec<String> {
        let ty = self.ir.dtype(values[0]);
        let per = 32 / ty.bits();
        let elem = ty.mlir();
        values
            .chunks(per)
            .map(|group| {
                let (vec, word) = (self.fresh("aw"), self.fresh("aww"));
                let parts = group.iter().map(|&v| self.v(v)).collect::<Vec<_>>().join(", ");
                self.line(indent, &format!("{vec} = vector.from_elements {parts} : vector<{per}x{elem}>"), None);
                self.line(indent, &format!("{word} = llvm.bitcast {vec} : vector<{per}x{elem}> to i32"), None);
                word
            })
            .collect()
    }

    /// Unpack the `i32` pairs `words` into the Float16 values `out`.
    fn unpack_pairs(&mut self, indent: usize, words: &[String], out: &[ValueId]) {
        for (word, pair) in words.iter().zip(out.chunks(2)) {
            let vec = self.fresh("h2");
            self.line(indent, &format!("{vec} = llvm.bitcast {word} : i32 to vector<2xf16>"), None);
            for (k, &v) in pair.iter().enumerate() {
                let text = format!("{} = vector.extract {vec}[{k}] : f16 from vector<2xf16>", self.v(v));
                self.line(indent, &text, None);
            }
        }
    }

    /// Unpack the `i32` words `words` (of [`Self::pack_words`]) into `out`.
    fn unpack_words(&mut self, indent: usize, words: &[String], out: &[ValueId]) {
        let ty = self.ir.dtype(out[0]);
        let per = 32 / ty.bits();
        let elem = ty.mlir();
        for (word, group) in words.iter().zip(out.chunks(per)) {
            let vec = self.fresh("aw");
            self.line(indent, &format!("{vec} = llvm.bitcast {word} : i32 to vector<{per}x{elem}>"), None);
            for (k, &v) in group.iter().enumerate() {
                let text = format!("{} = vector.extract {vec}[{k}] : {elem} from vector<{per}x{elem}>", self.v(v));
                self.line(indent, &text, None);
            }
        }
    }

    /// The operand fence: an empty statement with side effects that takes
    /// each of `inputs` and gives it back as the matching one of `results`.
    /// Float16 values are fenced in pairs, and BFloat16 pairs and runs of
    /// four 8-bit values (A fragments in registers) as `i32` words.
    fn operand_fence(&mut self, indent: usize, inst: &Inst, results: &[ValueId], inputs: &[ValueId]) {
        // How many values starting at `vs[0]` go into one word (1: alone).
        let group = |ir: &crate::ir::KernelIr, vs: &[ValueId]| -> usize {
            let ty = ir.dtype(vs[0]);
            let per = match ty {
                DType::F16 | DType::BF16 => 2,
                t if t.bits() == 8 && t != DType::Bool => 4,
                _ => return 1,
            };
            if vs.len() >= per && vs[..per].iter().all(|&v| ir.dtype(v) == ty) { per } else { 1 }
        };
        type Operands = Vec<(String, DType)>;
        let (mut ins, mut outs): (Operands, Operands) = (Vec::new(), Vec::new());
        let mut unpack: Vec<(String, usize, usize)> = Vec::new();
        let mut i = 0;
        while i < inputs.len() {
            let rest = &inputs[i..];
            let per = group(self.ir, rest);
            if per > 1 {
                let word = if self.ir.dtype(rest[0]) == DType::F16 {
                    self.pack_pairs(indent, &rest[..2]).remove(0)
                } else {
                    self.pack_words(indent, &rest[..per]).remove(0)
                };
                let out = self.fresh("fenced");
                ins.push((word, DType::U32));
                outs.push((out.clone(), DType::U32));
                unpack.push((out, i, per));
                i += per;
            } else {
                ins.push((self.v(inputs[i]), self.ir.dtype(inputs[i])));
                outs.push((self.v(results[i]), self.ir.dtype(results[i])));
                i += 1;
            }
        }
        if outs.is_empty() {
            return;
        }
        let constraints: Vec<String> = outs
            .iter()
            .map(|(_, t)| format!("={}", super::constraint(*t)))
            .chain((0..ins.len()).map(|k| k.to_string()))
            .collect();
        let args: Vec<(String, &str)> = ins.iter().map(|(n, t)| (n.clone(), t.mlir())).collect();
        self.asm_line(indent, inst, "", &constraints.join(","), &outs, &args, true);
        for (word, at, per) in unpack {
            if self.ir.dtype(results[at]) == DType::F16 {
                self.unpack_pairs(indent, &[word], &results[at..at + 2]);
            } else {
                self.unpack_words(indent, &[word], &results[at..at + per]);
            }
        }
    }
}
