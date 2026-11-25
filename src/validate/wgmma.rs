// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validator's WGMMA checks: each
//! instruction's operands, and where the asynchronous accumulators may go.
//!
//! A `wgmma.mma_async` writes its accumulators when a `wgmma.wait_group`
//! completes it; until then the PTX ISA lets no other instruction touch
//! those registers. In SSA form the accumulators are values: the results of
//! an [`Op::Wgmma`] are *pending*, and so are the values they reach through
//! operand fences, waits that do not complete them, and loop carries and
//! `if` results. A pending value may only be the accumulator input of
//! another wgmma (the chain of same-shape MMAs the PTX ISA allows), the
//! input of an operand fence or wait, a loop's initial value or a region's
//! yield; and it must reach one of them, or a wait never covers it and
//! LLVM may reuse its registers while the tensor cores still write them.

use super::{ValidationError, at};
use crate::ir::{DType, KernelIr, Loc, Op, Region, ValueId};

/// The operands of one WGMMA instruction.
pub(super) fn check_op(ir: &KernelIr, op: &Op) -> Result<(), String> {
    match op {
        Op::Wgmma { kind, d, c, a, desc_b, scale_d } => {
            let n = kind.n;
            if !(8..=256).contains(&n) || n % 8 != 0 {
                return Err(format!("wgmma: N is {n}, not a multiple of 8 from 8 to 256"));
            }
            check_types(kind)?;
            if kind.is_integer() && !(n == 8 || n == 24 || n.is_multiple_of(16)) {
                return Err(format!(
                    "{}: an 8-bit integer warpgroup MMA has N of 8, 24 or a multiple of 16",
                    kind.ptx()
                ));
            }
            if (kind.trans_a || kind.trans_b) && !kind.is_16bit() {
                return Err(format!("{}: 8-bit operands are K-major only (no transpose)", kind.ptx()));
            }
            if kind.a_regs {
                if kind.trans_a {
                    return Err(format!("{}: an A operand in registers is K-major (no transpose)", kind.ptx()));
                }
                if a.len() != kind.a_fragment() {
                    return Err(format!(
                        "{} takes {} A elements per thread from registers",
                        kind.ptx(),
                        kind.a_fragment()
                    ));
                }
                for &v in a {
                    if ir.dtype(v) != kind.a {
                        return Err(format!("{}: A element {v} is {}, not {}", kind.ptx(), ir.dtype(v), kind.a));
                    }
                }
            } else if a.len() != 1 || ir.dtype(a[0]) != DType::U64 {
                return Err("wgmma: the A matrix descriptor is one Uint64".into());
            }
            let want = kind.accumulators();
            if c.len() != want || d.len() != want {
                return Err(format!("{} takes and gives {want} accumulators per thread", kind.ptx()));
            }
            for &v in c.iter().chain(d) {
                if ir.dtype(v) != kind.acc {
                    return Err(format!("{}: accumulator {v} is {}, not {}", kind.ptx(), ir.dtype(v), kind.acc));
                }
            }
            if ir.dtype(*desc_b) != DType::U64 {
                return Err(format!("wgmma: the B matrix descriptor is a Uint64, not {}", ir.dtype(*desc_b)));
            }
            if ir.dtype(*scale_d) != DType::Bool {
                return Err("wgmma: scale-d (the ACCUMULATE field) is a Boolean".into());
            }
            if let Some(b) = ir.block_dim {
                let threads = b.iter().map(|&x| x as usize).product::<usize>();
                // The PTX ISA needs the issuing warpgroup complete; a block
                // whose last warpgroup is partial may still issue from the
                // whole ones (the simulator checks which warpgroup issues).
                if threads < 128 {
                    return Err(format!(
                        "wgmma runs on whole warpgroups of 128 threads, but the block has {threads} threads"
                    ));
                }
            }
        }
        Op::WgmmaFence { results, inputs } => same_types(ir, results, inputs, "wgmma.fence")?,
        Op::WgmmaWait { results, inputs, complete, .. } => {
            same_types(ir, results, inputs, "wgmma.wait_group")?;
            if complete.len() != inputs.len() {
                return Err("wgmma.wait_group: one completion flag per input".into());
            }
        }
        _ => {}
    }
    Ok(())
}

/// The operand and accumulator types of a warpgroup MMA (PTX ISA,
/// "wgmma.mma_async"): F16 or BF16 pairs (BF16 accumulating in F32 only),
/// any pair of E4M3 and E5M2 accumulating in F16 or F32, any pair of S8 and
/// U8 accumulating in S32.
fn check_types(kind: &crate::ir::WgmmaKind) -> Result<(), String> {
    use DType::*;
    let ok = matches!(
        (kind.a, kind.b, kind.acc),
        (F16, F16, F16 | F32)
            | (BF16, BF16, F32)
            | (F8E4M3FN | F8E5M2, F8E4M3FN | F8E5M2, F16 | F32)
            | (I8 | U8, I8 | U8, I32)
    );
    if ok {
        Ok(())
    } else {
        Err(format!(
            "wgmma: {} and {} operands accumulated in {} are no warpgroup MMA (f16/bf16 pairs, e4m3/e5m2 into f16 or \
             f32, s8/u8 into s32)",
            kind.a, kind.b, kind.acc
        ))
    }
}

fn same_types(ir: &KernelIr, results: &[ValueId], inputs: &[ValueId], what: &str) -> Result<(), String> {
    if results.len() != inputs.len() {
        return Err(format!("{what}: one result per fenced operand"));
    }
    for (&r, &i) in results.iter().zip(inputs) {
        if ir.dtype(r) != ir.dtype(i) {
            return Err(format!("{what}: operand {i} is {} but its result {r} is {}", ir.dtype(i), ir.dtype(r)));
        }
    }
    Ok(())
}

/// Pending accumulators, and the wgmma each comes from.
struct Pending {
    from: Vec<Option<Loc>>,
    changed: bool,
}

impl Pending {
    /// `to` is pending if `from` is.
    fn flow(&mut self, to: ValueId, from: ValueId) {
        if let (None, Some(loc)) = (self.from[to.0 as usize], self.from[from.0 as usize]) {
            self.from[to.0 as usize] = Some(loc);
            self.changed = true;
        }
    }

    fn of(&self, v: ValueId) -> Option<Loc> {
        self.from[v.0 as usize]
    }

    fn propagate(&mut self, r: &Region) {
        for inst in &r.insts {
            match &inst.op {
                Op::Wgmma { d, .. } => {
                    for &v in d {
                        if self.from[v.0 as usize].is_none() {
                            self.from[v.0 as usize] = Some(inst.loc);
                            self.changed = true;
                        }
                    }
                }
                Op::WgmmaFence { results, inputs } => {
                    for (&r, &i) in results.iter().zip(inputs) {
                        self.flow(r, i);
                    }
                }
                Op::WgmmaWait { results, inputs, complete, .. } => {
                    for ((&r, &i), &done) in results.iter().zip(inputs).zip(complete) {
                        if !done {
                            self.flow(r, i);
                        }
                    }
                }
                Op::For { init, args, body, results, .. } => {
                    for (i, &a) in args.iter().enumerate() {
                        self.flow(a, init[i]);
                        self.flow(a, body.yields[i]);
                        self.flow(results[i], a);
                    }
                }
                Op::While { init, cond_args, body_args, body, results, .. } => {
                    for (i, &a) in cond_args.iter().enumerate() {
                        self.flow(a, init[i]);
                        self.flow(a, body.yields[i]);
                        self.flow(body_args[i], a);
                        self.flow(results[i], a);
                    }
                }
                Op::If { then_region, else_region, results, .. } => {
                    for (i, &res) in results.iter().enumerate() {
                        self.flow(res, then_region.yields[i]);
                        self.flow(res, else_region.yields[i]);
                    }
                }
                _ => {}
            }
            for region in inst.op.regions() {
                self.propagate(region);
            }
        }
    }

    /// Check every use of a pending value, counting each value's uses.
    fn check_uses(&self, r: &Region, uses: &mut [u32]) -> Result<(), ValidationError> {
        for inst in &r.insts {
            let allowed: &[ValueId] = match &inst.op {
                Op::Wgmma { c, .. } => c,
                Op::WgmmaFence { inputs, .. } | Op::WgmmaWait { inputs, .. } => inputs,
                Op::For { init, .. } | Op::While { init, .. } => init,
                _ => &[],
            };
            for v in inst.op.operands() {
                uses[v.0 as usize] += 1;
                if let Some(from) = self.of(v)
                    && !allowed.contains(&v)
                {
                    return Err(ValidationError {
                        message: format!(
                            "{} reads an accumulator of the warpgroup MMA at {} before a wgmma.wait_group has \
                             completed it: the tensor cores may still be writing it. Wait for the MMA first \
                             (warpgroup::wait_group(0), or wait_group(n) with its group among the completed ones)",
                            inst.op.name(),
                            at(from)
                        ),
                        loc: Some(inst.loc),
                    });
                }
            }
            for region in inst.op.regions() {
                for &y in &region.yields {
                    uses[y.0 as usize] += 1;
                }
                self.check_uses(region, uses)?;
            }
        }
        Ok(())
    }
}

/// Where pending accumulators go (see the module documentation).
pub(super) fn check_accumulators(ir: &KernelIr) -> Result<(), ValidationError> {
    if ir.count_ops(|op| matches!(op, Op::Wgmma { .. })) == 0 {
        return Ok(());
    }
    let mut p = Pending { from: vec![None; ir.value_types.len()], changed: true };
    while p.changed {
        p.changed = false;
        p.propagate(&ir.body);
    }
    let mut uses = vec![0u32; ir.value_types.len()];
    p.check_uses(&ir.body, &mut uses)?;
    for &y in &ir.body.yields {
        uses[y.0 as usize] += 1;
    }
    if let Some(from) = (0..uses.len()).filter(|&v| uses[v] == 0).find_map(|v| p.from[v]) {
        return Err(ValidationError {
            message: "the accumulators of this warpgroup MMA are dropped while it may still be in flight: nothing \
                      waits for them (a fragment overwritten, or the kernel ending, before warpgroup::wait_group \
                      covers the MMA), so their registers could be reused while the tensor cores write them"
                .into(),
            loc: Some(from),
        });
    }
    Ok(())
}
