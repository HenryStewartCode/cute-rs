// SPDX-License-Identifier: MIT OR Apache-2.0

//! The instruction interpreter: `inst`, the big `inst_exec` match on `Op`, and `mma`.

use super::*;

impl<'a> Machine<'a> {
    pub(super) fn inst(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        if self.undefined.is_empty() || active.is_empty() {
            return self.inst_exec(inst, active);
        }
        let tainted = self.undefined_reads(inst, active)?;
        for r in inst.op.results() {
            for &t in active {
                self.undefined.remove(&(r, t));
            }
        }
        self.inst_exec(inst, active)?;
        for (t, loc) in tainted {
            for r in inst.op.results() {
                self.undefined.insert((r, t), loc);
            }
        }
        Ok(())
    }

    /// The lanes of `active` whose results `inst` computes from an
    /// undefined value (a pure operation's operand, or the operand a select
    /// picks), with the masked load that left it undefined; an error for
    /// any other read of one.
    fn undefined_reads(&mut self, inst: &'a Inst, active: &[usize]) -> Result<Vec<(usize, Loc)>, SimError> {
        let op = &inst.op;
        let mut tainted = Vec::new();
        let undefined = |m: &Self, v: ValueId, t: usize| m.undefined.get(&(v, t)).copied();
        let report = |m: &mut Self, t: usize, loc: Loc| -> SimError {
            m.loc = Some(inst.loc);
            m.err(
                format!(
                    "{} reads a value that a masked-off lane of the masked load at {loc} left undefined (the load has \
                     no pass_thru); pick it away with a select (where_) or give the load a pass_thru",
                    op.name()
                ),
                Some(t),
            )
        };
        for &t in active {
            match op {
                Op::Select { cond, on_true, on_false, .. } => {
                    if let Some(loc) = undefined(self, *cond, t) {
                        return Err(report(self, t, loc));
                    }
                    let picked = if self.get(*cond, t) != 0 { *on_true } else { *on_false };
                    if let Some(loc) = undefined(self, picked, t) {
                        tainted.push((t, loc));
                    }
                }
                // An integer division by an undefined value is undefined
                // behaviour (LLVM may assume it never runs), not a value.
                Op::Binary {
                    op: BinaryOp::Div | BinaryOp::Rem | BinaryOp::DivU | BinaryOp::RemU, lhs, rhs, ..
                } if !self.ir.dtype(*rhs).is_float() => {
                    if let Some(loc) = undefined(self, *rhs, t) {
                        return Err(report(self, t, loc));
                    }
                    if let Some(loc) = undefined(self, *lhs, t) {
                        tainted.push((t, loc));
                    }
                }
                Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Fma { .. }
                | Op::FloatInst { .. }
                | Op::IntInst { .. }
                | Op::Cmp { .. }
                | Op::Cast { .. }
                | Op::Bitcast { .. }
                | Op::Pack { .. }
                | Op::Unpack { .. }
                | Op::HalfInst { .. }
                | Op::Cvt { .. } => {
                    if let Some(loc) = op.operands().into_iter().find_map(|v| undefined(self, v, t)) {
                        tainted.push((t, loc));
                    }
                }
                // A masked-off lane of a masked access reads nothing but
                // its flag; a masked load passes its pass-through values'
                // marks on (in `inst_exec`), so they are not a use.
                Op::Store { pred: Some(p), .. } | Op::Load { pred: Some(p), .. } if self.get(*p, t) == 0 => {
                    if let Some(loc) = undefined(self, *p, t) {
                        return Err(report(self, t, loc));
                    }
                }
                Op::Load { index, attrs, pred, .. } => {
                    let read = [*index].into_iter().chain(attrs.cache_policy).chain(*pred);
                    if let Some(loc) = read.into_iter().find_map(|v| undefined(self, v, t)) {
                        return Err(report(self, t, loc));
                    }
                }
                _ => {
                    if let Some(loc) = op.operands().into_iter().find_map(|v| undefined(self, v, t)) {
                        return Err(report(self, t, loc));
                    }
                }
            }
        }
        Ok(tainted)
    }

    fn inst_exec(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        if active.is_empty() {
            return Ok(());
        }
        self.loc = Some(inst.loc);
        self.smem_target = None;
        let mut warps = 0u64;
        let mut last = usize::MAX;
        for &t in active {
            let key = self.warp_key(t);
            if key != last {
                warps += 1;
                last = key;
            }
        }
        self.report.warp_instructions += warps;
        self.clock += 1;
        if !self.pending_results.is_empty() && !wgmma::passes_accumulators(&inst.op) {
            self.check_pending_operands(&inst.op.operands(), active)?;
        }
        let ir = self.ir;
        if is_wide_op(ir, &inst.op) {
            return self.wide_inst(&inst.op, active);
        }
        match &inst.op {
            Op::Const { result, value } => {
                let bits = value.bits();
                self.lanes(active, *result, |_, _| Ok(bits))?;
            }
            Op::Call { callee, args, results, .. } => self.extern_call(active, *callee, args, results)?,
            Op::Multimem { op, buffer, index, elem, values, results } => {
                self.multimem(active, op, *buffer, *index, *elem, values, results)?
            }
            Op::Unary { result, op, x } => {
                let ty = ir.dtype(*x);
                self.lanes(active, *result, |m, t| Ok(unary(*op, ty, m.get(*x, t))))?;
            }
            Op::Binary { result, op, lhs, rhs } => {
                let ty = ir.dtype(*lhs);
                self.lanes(active, *result, |m, t| binary(*op, ty, m.get(*lhs, t), m.get(*rhs, t)))?;
            }
            Op::Fma { result, a, b, c } => {
                let ty = ir.dtype(*a);
                self.lanes(active, *result, |m, t| {
                    let (x, y, z) = (m.get(*a, t), m.get(*b, t), m.get(*c, t));
                    let r = match ty {
                        DType::F64 => f64::from_bits(x).mul_add(f64::from_bits(y), f64::from_bits(z)).to_bits(),
                        // Exact, then rounded once, as fma.rn.f16 and
                        // fma.rn.bf16 do.
                        DType::F16 | DType::BF16 => super::float::fma_rn(ty, x, y, z),
                        _ => from_f32(ty, to_f32(ty, x).mul_add(to_f32(ty, y), to_f32(ty, z))),
                    };
                    Ok(ptx_nan(ty, &[x, y, z], r))
                })?;
            }
            Op::FloatInst { result, func, args, mode, ftz, nan, xorsign_abs } => {
                let ty = ir.dtype(args[0]);
                self.lanes(active, *result, |m, t| {
                    let mut operands = [0u64; 3];
                    for (k, a) in args.iter().enumerate() {
                        operands[k] = m.get(*a, t);
                    }
                    Ok(super::float::float_inst(*func, *mode, *ftz, *nan, *xorsign_abs, ty, &operands[..args.len()]))
                })?;
            }
            Op::IntInst { result, func, args } => {
                let ty = ir.dtype(args[0]);
                self.lanes(active, *result, |m, t| {
                    let mut operands = [0u64; 4];
                    for (k, a) in args.iter().enumerate() {
                        operands[k] = m.get(*a, t);
                    }
                    Ok(super::int::int_inst(*func, ty, &operands[..args.len()]))
                })?;
            }
            Op::Cmp { result, pred, lhs, rhs } => {
                let ty = ir.dtype(*lhs);
                self.lanes(active, *result, |m, t| Ok(compare(*pred, ty, m.get(*lhs, t), m.get(*rhs, t)) as u64))?;
            }
            Op::Select { result, cond, on_true, on_false } => {
                self.lanes(active, *result, |m, t| {
                    Ok(if m.get(*cond, t) != 0 { m.get(*on_true, t) } else { m.get(*on_false, t) })
                })?;
            }
            Op::Cast { result, x } => {
                let (from, to) = (ir.dtype(*x), ir.dtype(*result));
                self.lanes(active, *result, |m, t| Ok(cast(from, to, m.get(*x, t))))?;
            }
            Op::Bitcast { result, x } => {
                self.lanes(active, *result, |m, t| Ok(m.get(*x, t)))?;
            }
            Op::Special { result, reg } => {
                self.lanes(active, *result, |m, t| Ok(m.special(*reg, t)))?;
            }
            // Cache hints change nothing here. A strong load (relaxed,
            // acquire, seq_cst, volatile, MMIO) does not race with strong
            // writes; an acquire joins what the bytes it reads published,
            // and the others leave that for the warp's next fence. A seq_cst
            // load is a fence and an acquire load.
            Op::Load { results, buffer, index, attrs, pred, pass_thru } => {
                // Masked-off lanes read nothing: their results are the
                // pass-through values, or undefined.
                let on: Vec<usize>;
                let active = match pred {
                    Some(p) => {
                        on = active.iter().copied().filter(|&t| self.get(*p, t) != 0).collect();
                        let off: Vec<usize> = active.iter().copied().filter(|&t| self.get(*p, t) == 0).collect();
                        for t in off {
                            for (k, r) in results.iter().enumerate() {
                                match pass_thru.get(k) {
                                    Some(&v) => {
                                        let bits = self.get128(v, t);
                                        self.set128(*r, t, bits);
                                        if let Some(&loc) = self.undefined.get(&(v, t)) {
                                            self.undefined.insert((*r, t), loc);
                                        }
                                    }
                                    None => {
                                        self.set(*r, t, 0);
                                        self.undefined.insert((*r, t), inst.loc);
                                    }
                                }
                            }
                        }
                        if on.is_empty() {
                            return Ok(());
                        }
                        &on[..]
                    }
                    None => active,
                };
                let d = ir.buffer(*buffer);
                let n = results.len();
                let size = ir.dtype(results[0]).bytes();
                let bytes = n * size;
                let strong = attrs.is_strong();
                if attrs.order == MemoryOrder::Sc {
                    self.fence_lanes(active);
                }
                let mut found: Vec<(usize, Vec<u32>)> = Vec::new();
                let mut accesses = Vec::with_capacity(active.len());
                let mut space = d.space;
                for &t in active {
                    let (b, idx) = self.target(*buffer, *index, size, t, "load")?;
                    space = ir.buffer(b).space;
                    // Another CTA's shared memory.
                    self.smem_target = match attrs.peer {
                        Some(p) => Some(self.dsmem_target(p, t, "a load (ld.shared::cluster)")?),
                        None => None,
                    };
                    let start = self.checked_range(b, idx, n, size, bytes, t, "load")?;
                    if attrs.is_non_coherent() {
                        self.nc_read(b, start, bytes, t)?;
                    }
                    if strong && let Some(clock) = self.released_on(b, start, bytes, size, t) {
                        found.push((self.clock_index(t), clock));
                    }
                    if strong && space != MemSpace::Local {
                        let (_, block, byte) = self.release_key(b, start, t);
                        let k = self.clock_index(t);
                        self.warps[k].poll = Some(Poll::Load { buffer: b, block, byte, inst });
                        self.warps[k].poll_epoch = self.epoch;
                    }
                    for (k, r) in results.iter().enumerate() {
                        let at = start + k * size;
                        if size == 16 {
                            // A 128-bit element, as two 8-byte words.
                            let lo = self.read_as(b, at, 8, t, strong)?;
                            let hi = self.read_as(b, at + 8, 8, t, strong)?;
                            self.set128(*r, t, lo as u128 | (hi as u128) << 64);
                        } else {
                            let bits = self.read_as(b, at, size, t, strong)?;
                            self.set(*r, t, bits);
                        }
                    }
                    accesses.push((t, start, bytes));
                }
                self.smem_target = None;
                // After every lane has read: the acquire orders what follows.
                let acquire = matches!(attrs.order, MemoryOrder::Acquire | MemoryOrder::Sc);
                for (k, clock) in found {
                    if acquire {
                        join(&mut self.clocks[k], &clock);
                    } else {
                        match &mut self.pending_acquire[k] {
                            Some(p) => join(p, &clock),
                            slot => *slot = Some(clock),
                        }
                    }
                }
                self.account(space, &accesses, false);
            }
            // A release store publishes its warp's clock on the bytes it
            // writes, and the warp moves on to a new time; another strong
            // store publishes the warp's clock at its last fence; a weak one
            // erases what was published. A seq_cst store is a fence and a
            // release store.
            Op::Store { values, buffer, index, attrs, pred } => {
                let on: Vec<usize>;
                let active = match pred {
                    Some(p) => {
                        on = active.iter().copied().filter(|&t| self.get(*p, t) != 0).collect();
                        if on.is_empty() {
                            return Ok(());
                        }
                        &on[..]
                    }
                    None => active,
                };
                let d = ir.buffer(*buffer);
                let n = values.len();
                let size = ir.dtype(values[0]).bytes();
                let bytes = n * size;
                let strong = attrs.is_strong();
                if attrs.order == MemoryOrder::Sc {
                    self.fence_lanes(active);
                }
                let release = matches!(attrs.order, MemoryOrder::Release | MemoryOrder::Sc);
                let mut published: Vec<(usize, Rc<[u32]>)> = Vec::new();
                if release {
                    for &t in active {
                        let k = self.clock_index(t);
                        if !published.iter().any(|p| p.0 == k) {
                            published.push((k, self.clocks[k].as_slice().into()));
                        }
                    }
                }
                let mut accesses = Vec::with_capacity(active.len());
                let mut space = d.space;
                for &t in active {
                    let (b, idx) = self.target(*buffer, *index, size, t, "store")?;
                    space = ir.buffer(b).space;
                    self.smem_target = match attrs.peer {
                        Some(p) => Some(self.dsmem_target(p, t, "a store (st.shared::cluster)")?),
                        None => None,
                    };
                    let start = self.checked_range(b, idx, n, size, bytes, t, "store")?;
                    for (k, v) in values.iter().enumerate() {
                        let at = start + k * size;
                        if size == 16 {
                            let bits = self.get128(*v, t);
                            self.write_as(b, at, 8, t, bits as u64, strong)?;
                            self.write_as(b, at + 8, 8, t, (bits >> 64) as u64, strong)?;
                        } else {
                            let bits = self.get(*v, t);
                            self.write_as(b, at, size, t, bits, strong)?;
                        }
                    }
                    let how = if release {
                        let k = self.clock_index(t);
                        Publish::Release(published.iter().find(|p| p.0 == k).expect("published").1.clone())
                    } else if strong {
                        Publish::Strong
                    } else {
                        Publish::Weak
                    };
                    self.publish(b, start, bytes, size, t, how);
                    accesses.push((t, start, bytes));
                }
                self.smem_target = None;
                for (k, _) in published {
                    let slot = self.slot_of(k);
                    self.clocks[k][slot] += 1;
                }
                self.account(space, &accesses, true);
            }
            Op::CpAsync { src, src_index, dst, dst_index, elem, elems, pred, .. } => {
                let n = *elems as usize;
                let size = elem.bytes();
                let bytes = n * size;
                let mut accesses = Vec::new();
                for &t in active {
                    // A predicated-off copy reads nothing (it zero-fills), so
                    // its source address, which may be before the tensor's
                    // start (a negative domain_offset), is not checked.
                    let fill = pred.is_some_and(|p| self.get(p, t) == 0);
                    let si = if fill {
                        0
                    } else {
                        self.checked_range(*src, self.index(*src_index, t), n, size, bytes, t, "cp.async source read")?
                    };
                    let di = self.checked_range(
                        *dst,
                        self.index(*dst_index, t),
                        n,
                        size,
                        bytes,
                        t,
                        "cp.async destination write",
                    )?;
                    if !fill && self.pdl.is_some() {
                        self.pdl_access(*src, si..si + bytes, t, false)?;
                    }
                    let data = if fill {
                        vec![0; bytes]
                    } else {
                        self.globals[src.0 as usize].as_ref().expect("global buffers are bound")[si..si + bytes]
                            .to_vec()
                    };
                    self.smem_claim_write(*dst, di, bytes, t, true, false)?;
                    if !fill {
                        accesses.push((t, si, bytes));
                    }
                    let block = self.block_rank(t);
                    let id = self.ops.len();
                    self.ops.push(AsyncOp {
                        thread: t,
                        agent: self.agent_slot(block),
                        writes: vec![AsyncWrite { dst: *dst, block, dst_byte: di, data }],
                        reads: Vec::new(),
                        global_writes: Vec::new(),
                        global_reduce: None,
                        results: Vec::new(),
                        trigger: Trigger::CpAsync { tracked: false },
                        done: None,
                    });
                    self.uncommitted[t].push(id);
                    self.in_flight[t].push(id);
                    if self.opts.async_completion == super::AsyncCompletion::Early {
                        // Complete at issue: as the agent's write, which the
                        // thread's warp is ordered after, as if it had waited.
                        self.land_tracked_copy(id);
                        self.land_cp_async_for_wait(id, t);
                    }
                }
                self.account(MemSpace::Global, &accesses, false);
            }
            Op::CpAsyncCommitGroup => {
                for &t in active {
                    let group = std::mem::take(&mut self.uncommitted[t]);
                    self.groups[t].push_back(group);
                }
            }
            Op::CpAsyncWaitGroup { pending } => {
                for &t in active {
                    while self.groups[t].len() > *pending as usize {
                        let group = self.groups[t].pop_front().unwrap();
                        for id in group {
                            self.land_cp_async_for_wait(id, t);
                        }
                    }
                }
            }
            Op::LdMatrix { results, buffer, index, num, form, .. } if *form != LdMatrixForm::M8n8B16 => {
                let num = *num as usize;
                let rows_per = form.rows() as usize;
                let bases = self.full_warps(active, "ldmatrix")?;
                for base in bases {
                    // Each supplying lane's 16-byte row, expanded to 16
                    // 8-bit containers.
                    let mut rows = vec![[0u8; 16]; rows_per * num];
                    let mut accesses = Vec::with_capacity(rows.len());
                    for (l, row) in rows.iter_mut().enumerate() {
                        let t = base + l;
                        let i =
                            self.checked_range(*buffer, self.index(*index, t), 16, 1, 16, t, "ldmatrix row read")?;
                        let lo = self.read(*buffer, i, 8, t)?;
                        let hi = self.read(*buffer, i + 8, 8, t)?;
                        *row = ldmatrix_expand(lo as u128 | (hi as u128) << 64, form.unpack());
                        accesses.push((t, i, 16));
                    }
                    for lane in 0..32 {
                        for (r, &result) in results.iter().enumerate() {
                            let mut word = 0u64;
                            for j in 0..4 {
                                let byte = match form {
                                    // Figure 108 of the PTX ISA, of the
                                    // matrix as loaded (transposed): lane
                                    // `l` register `2m + h` holds row `l / 4
                                    // + 8h`, columns `4 (l % 4)` to `+3`,
                                    // which are memory rows, and the row is
                                    // the byte within them.
                                    LdMatrixForm::M16n16B8 { .. } => {
                                        rows[16 * (r / 2) + 4 * (lane % 4) + j][lane / 4 + 8 * (r % 2)]
                                    }
                                    // Figure 109: register `m` holds row `l /
                                    // 4` of matrix `m`, columns `4 (l % 4)`
                                    // to `+3`.
                                    _ => rows[8 * r + lane / 4][4 * (lane % 4) + j],
                                };
                                word |= (byte as u64) << (8 * j);
                            }
                            self.set(result, base + lane, word);
                        }
                    }
                    self.shared_phases(&accesses, 16, 8);
                }
            }
            Op::LdMatrix { results, buffer, index, num, trans, .. } => {
                let num = *num as usize;
                let bases = self.full_warps(active, "ldmatrix")?;
                for base in bases {
                    let mut rows = vec![[0u64; 8]; 8 * num];
                    let mut accesses = Vec::with_capacity(8 * num);
                    for (l, row) in rows.iter_mut().enumerate() {
                        let t = base + l;
                        let i = self.checked_range(*buffer, self.index(*index, t), 8, 2, 16, t, "ldmatrix row read")?;
                        for (k, slot) in row.iter_mut().enumerate() {
                            *slot = self.read(*buffer, i + 2 * k, 2, t)?;
                        }
                        accesses.push((t, i, 16));
                    }
                    for lane in 0..32 {
                        for r in 0..num {
                            for j in 0..2 {
                                let v = if *trans {
                                    rows[r * 8 + 2 * (lane % 4) + j][lane / 4]
                                } else {
                                    rows[r * 8 + lane / 4][2 * (lane % 4) + j]
                                };
                                self.set(results[2 * r + j], base + lane, v);
                            }
                        }
                    }
                    self.shared_phases(&accesses, 16, 8);
                }
            }
            Op::StMatrix { values, buffer, index, num, trans, form } => {
                let num = *num as usize;
                let bases = self.full_warps(active, "stmatrix")?;
                for base in bases {
                    // The 8 16-byte rows of each matrix in shared memory, as the PTX ISA's
                    // fragments place the registers' elements in them.
                    let mut rows = vec![[0u8; 16]; 8 * num];
                    for lane in 0..32 {
                        for (m, &v) in values.iter().enumerate() {
                            let word = self.get(v, base + lane) as u32;
                            match form {
                                // Register `m` holds row `l / 4`, columns `2 (l % 4)` and `+1`,
                                // of matrix `m`; transposed, row `r` column `c` lands in
                                // memory row `c` at element `r`.
                                StMatrixForm::M8n8B16 => {
                                    for j in 0..2 {
                                        let (r, c) = (lane / 4, 2 * (lane % 4) + j);
                                        let (mr, mc) = if *trans { (c, r) } else { (r, c) };
                                        let half = (word >> (16 * j)) as u16;
                                        rows[8 * m + mr][2 * mc..2 * mc + 2].copy_from_slice(&half.to_le_bytes());
                                    }
                                }
                                // Byte `q = 2h + j` of register `m` is row `l / 4 + 8h`, column
                                // `2 (l % 4) + j` of the 16x8 matrix `m`, stored transposed:
                                // memory row `c` (of 16 bytes) holds column `c`.
                                StMatrixForm::M16n8B8 => {
                                    for q in 0..4 {
                                        let (r, c) = (lane / 4 + 8 * (q / 2), 2 * (lane % 4) + q % 2);
                                        rows[8 * m + c][r] = (word >> (8 * q)) as u8;
                                    }
                                }
                            }
                        }
                    }
                    let (count, size) = if *form == StMatrixForm::M8n8B16 { (8, 2) } else { (16, 1) };
                    let mut accesses = Vec::with_capacity(rows.len());
                    for (l, row) in rows.iter().enumerate() {
                        let t = base + l;
                        let i = self.checked_range(
                            *buffer,
                            self.index(*index, t),
                            count,
                            size,
                            16,
                            t,
                            "stmatrix row write",
                        )?;
                        let lo = u64::from_le_bytes(row[..8].try_into().unwrap());
                        let hi = u64::from_le_bytes(row[8..].try_into().unwrap());
                        self.write_as(*buffer, i, 8, t, lo, false)?;
                        self.write_as(*buffer, i + 8, 8, t, hi, false)?;
                        accesses.push((t, i, 16));
                    }
                    self.shared_phases(&accesses, 16, 8);
                }
            }
            Op::Mma { kind, d, a, b, c, e } => self.mma(*kind, d, a, b, c, *e, active)?,
            Op::MmaBlockScaled { kind, d, a, b, c, sfa, sfb } => {
                self.mma_block_scaled(*kind, d, a, b, c, *sfa, *sfb, active)?
            }
            Op::MmaSpBlockScaled { kind, d, a, b, c, e, sfa, sfb } => {
                self.mma_sp_block_scaled(*kind, d, [a, b, c], *e, *sfa, *sfb, active)?
            }
            Op::WgmmaFence { .. } | Op::WgmmaCommitGroup | Op::WgmmaWait { .. } => self.wgmma_inst(&inst.op, active)?,
            Op::Tcgen05 { inst: Tcgen05Inst::CheckAddress, args, .. } => self.tmem_check_address(args[0], active)?,
            // The thread-sync fences order the hardware's pipelines only.
            Op::Tcgen05 { inst: Tcgen05Inst::FenceBeforeThreadSync | Tcgen05Inst::FenceAfterThreadSync, .. } => {}
            Op::Tcgen05Ld { .. } | Op::Tcgen05St { .. } => self.tmem_ldst(inst, active)?,
            Op::Tcgen05Mma { .. } => self.umma_issue(inst, active)?,
            Op::Tcgen05Commit { .. } => self.umma_commit(inst, active)?,
            Op::Tcgen05Cp { .. } => self.tmem_cp_issue(inst, active)?,
            Op::ClcTryCancel { .. } => self.clc_try_cancel(inst, active)?,
            Op::ClcQueryCancel { .. } => self.clc_query_cancel(inst, active)?,
            // Handled by the scheduler (`Machine::step`).
            Op::Tcgen05 { .. }
            | Op::SetMaxNReg { .. }
            | Op::Wgmma { .. }
            | Op::Barrier
            | Op::NamedBarrier { .. }
            | Op::BarrierRed { .. }
            | Op::Exit
            | Op::ClusterArrive { .. }
            | Op::ClusterWait { .. }
            | Op::If { .. }
            | Op::For { .. }
            | Op::While { .. } => {
                unreachable!("a barrier or structured instruction")
            }
            Op::WarpBarrier { mask } => {
                for (base, lanes) in self.warps_of(active) {
                    self.member_masks("sync_warp", base, &lanes, *mask)?;
                }
            }
            Op::Shuffle { result, kind, value, offset, mask, clamp } => {
                let mut out = Vec::with_capacity(active.len());
                for (base, lanes) in self.warps_of(active) {
                    let warp = self.warp_of(base);
                    for &lane in &lanes {
                        let t = base + lane;
                        let members = self.get(*mask, t) as u32;
                        if members >> lane & 1 == 0 {
                            return Err(self.err(
                                format!(
                                    "shuffle executed by lane {lane} of warp {warp}, which is not in its member \
                                     mask {members:#010x}"
                                ),
                                Some(t),
                            ));
                        }
                        let source =
                            shuffle_source(*kind, lane, self.get(*offset, t) as u32, self.get(*clamp, t) as u32);
                        let bits = match source {
                            None => self.get(*value, t),
                            Some(s) if members >> s & 1 == 0 => {
                                return Err(self.err(
                                    format!("shuffle reads lane {s}, which is not in the member mask {members:#010x}"),
                                    Some(t),
                                ));
                            }
                            Some(s) if lanes.binary_search(&s).is_ok() => self.get(*value, base + s),
                            Some(s) if s >= self.warp_size(base) => {
                                return Err(self.err(
                                    format!(
                                        "shuffle reads lane {s}, which warp {warp} does not have: it is the last warp \
                                         of a block of {} threads, with {} lanes",
                                        self.threads_per_block,
                                        self.warp_size(base)
                                    ),
                                    Some(t),
                                ));
                            }
                            Some(s) => {
                                return Err(self.err(format!("shuffle reads lane {s}, which is not active"), Some(t)));
                            }
                        };
                        out.push((t, bits));
                    }
                    self.member_masks("shuffle", base, &lanes, *mask)?;
                }
                for (t, bits) in out {
                    self.set(*result, t, bits);
                }
            }
            Op::Vote { result, kind, pred, mask } => {
                let what = match kind {
                    VoteKind::All => "vote_all_sync",
                    VoteKind::Any => "vote_any_sync",
                    VoteKind::Uni => "vote_uni_sync",
                    VoteKind::Ballot => "vote_ballot_sync",
                };
                for (base, lanes) in self.warps_of(active) {
                    // `vote.sync` has the same rules as the other warp
                    // collectives: every non-exited lane of the mask votes.
                    let masks = self.member_masks(what, base, &lanes, *mask)?;
                    let results: Vec<u64> = masks
                        .iter()
                        .map(|&members| {
                            let votes: Vec<(usize, bool)> = (0..self.warp_size(base))
                                .filter(|&o| members >> o & 1 == 1)
                                .map(|o| (o, self.get(*pred, base + o) != 0))
                                .collect();
                            match kind {
                                VoteKind::All => votes.iter().all(|v| v.1) as u64,
                                VoteKind::Any => votes.iter().any(|v| v.1) as u64,
                                VoteKind::Uni => votes.windows(2).all(|w| w[0].1 == w[1].1) as u64,
                                VoteKind::Ballot => votes.iter().filter(|v| v.1).fold(0u64, |acc, v| acc | 1 << v.0),
                            }
                        })
                        .collect();
                    for (&l, bits) in lanes.iter().zip(results) {
                        self.set(*result, base + l, bits);
                    }
                }
            }
            Op::Match { result, kind, value, mask } => {
                for (base, lanes) in self.warps_of(active) {
                    let masks = self.member_masks("match_sync", base, &lanes, *mask)?;
                    // Every lane of each mask executes it (checked above).
                    let results: Vec<u64> = lanes
                        .iter()
                        .zip(&masks)
                        .map(|(&l, &members)| {
                            let mine = self.get(*value, base + l);
                            let (mut group, mut same) = (0u64, 0u64);
                            for o in (0..self.warp_size(base)).filter(|&o| members >> o & 1 == 1) {
                                group |= 1 << o;
                                if self.get(*value, base + o) == mine {
                                    same |= 1 << o;
                                }
                            }
                            match kind {
                                MatchKind::Any => same,
                                MatchKind::All if same == group => group,
                                MatchKind::All => 0,
                            }
                        })
                        .collect();
                    for (&l, bits) in lanes.iter().zip(results) {
                        self.set(*result, base + l, bits);
                    }
                }
            }
            Op::ElectOne { result, mask } => {
                for (base, lanes) in self.warps_of(active) {
                    let members = self.get(*mask, base + lanes[0]) as u32;
                    let leader = lanes.iter().copied().find(|&l| members >> l & 1 == 1);
                    for &l in &lanes {
                        self.set(*result, base + l, (Some(l) == leader) as u64);
                    }
                }
            }
            Op::WarpUniform { result, value } => {
                for (base, lanes) in self.warps_of(active) {
                    let first = self.get(*value, base + lanes[0]);
                    if let Some(&l) = lanes.iter().find(|&&l| self.get(*value, base + l) != first) {
                        return Err(self.err(
                            format!(
                                "make_warp_uniform: lane {l} of warp {} holds {}, but lane {} holds {} (the value is not warp-uniform)",
                                self.warp_of(base),
                                self.get(*value, base + l) as u32 as i32,
                                lanes[0],
                                first as u32 as i32
                            ),
                            Some(base + l),
                        ));
                    }
                    for &l in &lanes {
                        self.set(*result, base + l, first);
                    }
                }
            }
            Op::ActiveMask { result } => {
                for (base, lanes) in self.warps_of(active) {
                    let bits = lanes.iter().fold(0u64, |acc, &l| acc | 1 << l);
                    for &l in &lanes {
                        self.set(*result, base + l, bits);
                    }
                }
            }
            Op::Atomic { result, kind, ty, buffer, index, value, compare, sem, peer, .. } => {
                self.atomic(active, Some(*result), *kind, *ty, (*buffer, *index, *peer), *value, *compare, *sem)?;
                if *kind == AtomicKind::CompareExchange {
                    self.note_failed_cas(inst, active, (*buffer, *index, *peer), *value, *compare)?;
                }
            }
            Op::Red { kind, ty, buffer, index, value, sem, peer, .. } => {
                self.atomic(active, None, *kind, *ty, (*buffer, *index, *peer), *value, None, *sem)?;
            }
            Op::MbarrierInit { buffer, index, count } => {
                self.progress();
                for &t in active {
                    let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "mbarrier.init")?;
                    let arrivals = self.get(*count, t) as u32;
                    if !(1..1 << 20).contains(&arrivals) {
                        return Err(self.err(
                            format!("an mbarrier count of {} is outside 1 to 2^20 - 1", arrivals as i32),
                            Some(t),
                        ));
                    }
                    let key = (self.block_rank(t), buffer.0, byte);
                    if let Some(owed) = self.mbarriers.get(&key).map(|m| m.deferred.len()).filter(|&n| n > 0) {
                        return Err(self.err(
                            format!(
                                "mbarrier.init on an mbarrier that still waits for {owed} arrival(s) of \
                                 cp.async.mbarrier.arrive, whose copies are in flight"
                            ),
                            Some(t),
                        ));
                    }
                    let mbarrier = Mbarrier {
                        arrivals_per_phase: arrivals,
                        arrivals_left: arrivals,
                        tx_count: 0,
                        copies: Vec::new(),
                        deferred: Vec::new(),
                        phase: 0,
                        release: vec![0; self.slots()],
                        completed: vec![0; self.slots()],
                        seen: vec![Vec::new(); self.slots()],
                        init: (self.slot_of(self.clock_index(t)), self.clock_index(t), inst.loc),
                        fenced: None,
                    };
                    self.invalidated.remove(&key);
                    self.mbarriers.insert((self.block_rank(t), buffer.0, byte), mbarrier);
                }
            }
            Op::MbarrierArrive { buffer, index, count, expect_tx, peer, relaxed, scope } => {
                self.progress();
                // Warps that arrived with a release, by key.
                let mut released: Vec<usize> = Vec::new();
                for &t in active {
                    let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "mbarrier.arrive")?;
                    let block = self.mbarrier_block(*peer, t, "mbarrier.arrive")?;
                    let remote = block != self.block_rank(t);
                    if remote {
                        self.check_remote_mbarrier((block, buffer.0, byte), t, "mbarrier.arrive")?;
                    }
                    let n = count.map_or(1, |c| self.get(c, t) as u32);
                    let tx = expect_tx.map_or(0, |v| self.get(v, t) as u32 as u64);
                    // A releasing arrival publishes the warp's clock to the
                    // phase's waiters: on its own CTA's mbarrier, or on a
                    // peer's at cluster scope. At CTA scope a peer's waiters
                    // are outside the scope, so it orders nothing for them.
                    let release = !relaxed && (!remote || *scope == MemScopeKind::Cluster);
                    let k = self.clock_index(t);
                    // A releasing arrival (a pipeline's consumer_release)
                    // with a TMEM load still in flight lets the stage be overwritten under it.
                    if !relaxed && let Some(at) = self.tmem_pending_load(k) {
                        return Err(self.err(
                            format!(
                                "mbarrier.arrive while this warp's tcgen05.ld ({}:{}) has not completed: wait for the \
                                 loads (tcgen05.wait::ld, fence_view_async_tmem_load) before releasing the TMEM they \
                                 read",
                                at.file(),
                                at.line()
                            ),
                            Some(t),
                        ));
                    }
                    let clock = &self.clocks[k];
                    let Some(state) = self.mbarriers.get_mut(&(block, buffer.0, byte)) else {
                        if let Some(e) = self.invalidated_error((block, buffer.0, byte), "mbarrier.arrive", t) {
                            return Err(e);
                        }
                        return Err(self.err("arrival at an mbarrier that was never initialised", Some(t)));
                    };
                    if n == 0 || n > state.arrivals_left {
                        let left = state.arrivals_left;
                        return Err(self.err(
                            format!(
                                "{n} arrival(s) at an mbarrier whose phase is waiting for {left} (more arrivals than it was initialised with?)"
                            ),
                            Some(t),
                        ));
                    }
                    // The arrivals cp.async.mbarrier.arrive owes are not
                    // free for others to make.
                    let owed = state.deferred.len() as u32;
                    if n > state.arrivals_left - owed {
                        let left = state.arrivals_left;
                        if state.deferred.iter().any(|d| d.commit) {
                            // A commit's arrival is owed.
                            return Err(self.err(
                                format!(
                                    "{n} arrival(s) at an mbarrier whose phase is waiting for {left}, {owed} of which \
                                     a tcgen05.commit owes (more arrivals than it was initialised with?): the phase \
                                     would complete before the committed MMAs, so count every arriving thread in the \
                                     mbarrier's count"
                                ),
                                Some(t),
                            ));
                        }
                        return Err(self.err(
                            format!(
                                "{n} arrival(s) at an mbarrier whose phase is waiting for {left}, {owed} of which the \
                                 cp.async copies tracked by cp_async_mbarrier_arrive owe: the phase would complete with \
                                 those copies still in flight (with cp_async_mbarrier_arrive_noinc, the mbarrier's count \
                                 must include one arrival per call)"
                            ),
                            Some(t),
                        ));
                    }
                    if release {
                        join(&mut state.release, clock);
                    } else if *relaxed
                        && (!remote || *scope == MemScopeKind::Cluster)
                        && let Some(fence) = &self.fence_release[k]
                    {
                        // A fence then a relaxed arrival: a release pattern
                        // that publishes the clock at the fence.
                        join(&mut state.release, fence);
                    }
                    // `.expect_tx` raises the tx-count before the arrival.
                    state.tx_count += tx as i64;
                    state.arrivals_left -= n;
                    if state.arrivals_left == 0 && state.tx_count < 0 && state.copies.is_empty() {
                        // Copies that completed early landed more bytes than
                        // the phase expects.
                        let extra = -state.tx_count;
                        return Err(self.err(
                            format!(
                                "the bulk copies reporting to an mbarrier phase landed {extra} byte(s) more than \
                                 its transaction count (mbarrier_expect_tx missing or too small?): the phase never \
                                 completes"
                            ),
                            Some(t),
                        ));
                    }
                    let stranded = if state.complete_if_done() { state.copies.len() } else { 0 };
                    if stranded > 0 {
                        return Err(self.err(
                            format!(
                                "an mbarrier phase completed with {stranded} bulk cop{} reporting to it still in flight: \
                                 its transaction count did not include their bytes (missing mbarrier_expect_tx?)",
                                if stranded == 1 { "y" } else { "ies" }
                            ),
                            Some(t),
                        ));
                    }
                    if release && !released.contains(&k) {
                        released.push(k);
                    }
                }
                // Arriving warps move on to a new time of their own.
                for k in released {
                    let slot = self.slot_of(k);
                    self.clocks[k][slot] += 1;
                }
            }
            Op::MbarrierExpectTx { buffer, index, bytes, peer, .. } => {
                self.progress();
                for &t in active {
                    let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "mbarrier.expect_tx")?;
                    let block = self.mbarrier_block(*peer, t, "mbarrier.expect_tx")?;
                    if block != self.block_rank(t) {
                        self.check_remote_mbarrier((block, buffer.0, byte), t, "mbarrier.expect_tx")?;
                    }
                    let tx = self.get(*bytes, t) as u32 as u64;
                    let Some(state) = self.mbarriers.get_mut(&(block, buffer.0, byte)) else {
                        if let Some(e) = self.invalidated_error((block, buffer.0, byte), "mbarrier.expect_tx", t) {
                            return Err(e);
                        }
                        return Err(self.err("mbarrier.expect_tx on an mbarrier that was never initialised", Some(t)));
                    };
                    state.tx_count += tx as i64;
                    state.complete_if_done();
                }
            }
            Op::MbarrierWait { result, buffer, index, parity, kind, state: by_state, .. } => {
                let what = match kind {
                    MbarrierWaitKind::Try => "mbarrier.try_wait",
                    MbarrierWaitKind::Test => "mbarrier.test_wait",
                };
                let mut failed: Vec<(usize, MbarrierKey, u64)> = Vec::new();
                for &t in active {
                    let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, what)?;
                    let block = self.block_rank(t);
                    let key = (block, buffer.0, byte);
                    // A test_wait polls: with random completion it forces
                    // nothing, so a kernel that polls once and reads anyway
                    // is caught.
                    if *kind == MbarrierWaitKind::Try || self.async_rng.is_none() {
                        self.fire_deferred_arrivals(key);
                        self.land_bulk_copies(key, t)?;
                    }
                    let warp = self.warp_of(t) as usize;
                    let (k, slot) = (self.clock_index(t), self.slot_of(self.clock_index(t)));
                    let raw = self.get(*parity, t);
                    let parity = raw & 1;
                    let Some(state) = self.mbarriers.get(&key) else {
                        if let Some(e) = self.invalidated_error(key, what, t) {
                            return Err(e);
                        }
                        return Err(self.err(format!("{what} on an mbarrier that was never initialised"), Some(t)));
                    };
                    // The parity names the current phase, which is running,
                    // or the one before it, which is over. Waiting for phase
                    // `asked` only means that if the warp knows phase
                    // `asked - 1` is over: otherwise the mbarrier may still be
                    // in that phase on hardware, where the parity names the
                    // phase before it and the wait succeeds at once.
                    // A state token (the simulator's is
                    // the phase its arrival was in) names its phase exactly.
                    let over = if *by_state { state.phase > raw } else { parity != state.phase & 1 };
                    let current = state.phase;
                    let asked = if over { state.phase.saturating_sub(1) } else { state.phase };
                    let known = state.known_phases(&self.clocks[k]);
                    if !*by_state && current >= known + 2 {
                        return Err(self.err(
                            format!(
                                "{what} with parity {parity} by warp {warp}: the mbarrier has completed {current} \
                                 phases, but the warp has seen only {known} complete, so the parity can no longer \
                                 tell phase {known} from phase {} (phase aliasing): the mbarrier ran two phases ahead \
                                 of the waiter, typically because its producer does not wait for an empty barrier \
                                 (back-pressure) before arriving again",
                                known + 2
                            ),
                            Some(t),
                        ));
                    }
                    if !*by_state && known < asked {
                        return Err(self.err(
                            format!(
                                "{what} with parity {parity} asks about phase {asked} of the mbarrier, but warp {warp} \
                                 has not seen phase {} complete: on hardware the mbarrier may still be in that phase, \
                                 where parity {parity} refers to the phase before it, which is over, so the wait succeeds \
                                 at once (the simulator's schedule may not show this). Wait for phase {} first",
                                asked - 1,
                                asked - 1
                            ),
                            Some(t),
                        ));
                    }
                    if over {
                        // Acquire what the arrivals of that phase released,
                        // and remember that this warp has seen it complete.
                        let state = self.mbarriers.get_mut(&key).expect("the mbarrier exists");
                        let clock = &mut self.clocks[k];
                        join(clock, &state.completed);
                        let (now, phases) = (clock[slot], state.phase);
                        let seen = &mut state.seen[slot];
                        match seen.last_mut() {
                            Some(last) if last.1 >= phases => {}
                            Some(last) if last.0 == now => last.1 = phases,
                            _ => seen.push((now, phases)),
                        }
                    }
                    if !over {
                        failed.push((k, key, current));
                    }
                    self.set(*result, t, over as u64);
                }
                // A failed wait is a poll, and a yield point.
                for (k, key, phase) in failed {
                    self.warps[k].poll = Some(Poll::Mbarrier { key, phase, inst });
                    self.warps[k].poll_epoch = self.epoch;
                    self.yield_warp(k);
                }
            }
            Op::MbarrierExt { .. } => self.mbarrier_ext(inst, active)?,
            Op::PrimMarker { marker } => {
                let warps: Vec<usize> = self.warps_of(active).iter().map(|(base, _)| self.clock_index(*base)).collect();
                if let crate::ir::PrimMarker::Nanosleep(_) = marker {
                    // A hint: the warp yields to the others.
                    self.report.nanosleeps += warps.len() as u64;
                    for k in warps {
                        self.yield_warp(k);
                    }
                } else {
                    self.report.markers += warps.len() as u64;
                }
            }
            Op::StBulk { buffer, index, elem, size } => {
                let mut accesses = Vec::new();
                for &t in active {
                    let n = self.get(*size, t) as i64;
                    if n <= 0 || n % 16 != 0 {
                        return Err(
                            self.err(format!("st.bulk: a size of {n} bytes is not a positive multiple of 16"), Some(t))
                        );
                    }
                    let (b, idx) = self.target(*buffer, *index, elem.bytes().max(1), t, "st.bulk")?;
                    let start = self.checked_range(b, idx, 1, elem.bytes().max(1), 16, t, "st.bulk")?;
                    let n = n as usize;
                    if start + n > self.buffer_bytes(b) {
                        return Err(self.err(
                            format!(
                                "st.bulk of {n} bytes from byte {start} runs past the end of its {}-byte buffer",
                                self.buffer_bytes(b)
                            ),
                            Some(t),
                        ));
                    }
                    for at in (start..start + n).step_by(8) {
                        self.write_as(b, at, 8, t, 0, false)?;
                    }
                    self.publish(b, start, n, 8, t, Publish::Weak);
                    accesses.push((t, start, n));
                }
                self.account(MemSpace::Shared, &accesses, true);
            }
            Op::Tcgen05Shift { .. } => {
                return Err(self.err(
                    "tcgen05.shift (prims::tcgen05_shift) is not simulated: compile the kernel for an sm_100a GPU",
                    active.first().copied(),
                ));
            }
            // No-ops in a single launch; in a programmatic dependent launch the
            // wait lets the thread's later accesses see the primary's writes.
            Op::GridDepControl { launch_dependents } => {
                if !launch_dependents && let Some(pdl) = self.pdl.as_mut() {
                    let tpb = self.threads_per_block;
                    for &t in active {
                        pdl.waited.insert((self.blocks[t / tpb], t % tpb));
                    }
                }
            }
            // A release fence for the warp's earlier mbarrier.init ops, which
            // reaches other CTAs through a later cluster-scope edge: a
            // cluster barrier (relaxed too, as PTX's pattern has it) or a
            // release/acquire chain.
            Op::FenceMbarrierInit => {
                let mut fenced: Vec<usize> = Vec::new();
                for &t in active {
                    let k = self.clock_index(t);
                    if !fenced.contains(&k) {
                        fenced.push(k);
                    }
                }
                for k in fenced {
                    let now = self.clocks[k][self.slot_of(k)];
                    for m in self.mbarriers.values_mut() {
                        if m.init.1 == k && m.fenced.is_none() {
                            m.fenced = Some(now);
                        }
                    }
                }
            }
            Op::CpAsyncBulk {
                src,
                src_index,
                dst,
                dst_index,
                elem,
                bytes,
                direction: BulkDirection::G2S,
                completion: BulkCompletion::Mbarrier { buffer: mbarrier, index: mbarrier_index },
                mcast_mask,
                cache_policy,
                ..
            } => {
                let size = elem.bytes();
                let mut accesses = Vec::new();
                let what = if mcast_mask.is_some() { "cp.async.bulk (multicast)" } else { "cp.async.bulk" };
                for &t in active {
                    let count = self.get(*bytes, t) as u32 as usize;
                    if !count.is_multiple_of(16) {
                        return Err(
                            self.err(format!("cp.async.bulk of {count} bytes is not a multiple of 16"), Some(t))
                        );
                    }
                    let n = count / size;
                    let si = self.checked_range(
                        *src,
                        self.index(*src_index, t),
                        n,
                        size,
                        16,
                        t,
                        "cp.async.bulk source read",
                    )?;
                    let di = self.checked_range(
                        *dst,
                        self.index(*dst_index, t),
                        n,
                        size,
                        16,
                        t,
                        "cp.async.bulk destination write",
                    )?;
                    let byte = self.checked_range(
                        *mbarrier,
                        self.index(*mbarrier_index, t),
                        1,
                        8,
                        8,
                        t,
                        "cp.async.bulk mbarrier",
                    )?;
                    let targets = self.cluster_targets(*mcast_mask, t, what)?;
                    for &block in &targets {
                        self.check_bulk_mbarrier((block, mbarrier.0, byte), t, what)?;
                    }
                    if self.pdl.is_some() {
                        self.pdl_access(*src, si..si + count, t, false)?;
                    }
                    let data = self.globals[src.0 as usize].as_ref().expect("global buffers are bound")[si..si + count]
                        .to_vec();
                    for &block in &targets {
                        let key = (block, mbarrier.0, byte);
                        self.issue_to_mbarrier(*dst, block, vec![(di, data.clone())], Vec::new(), key, t, what, true)?;
                    }
                    if cache_policy.is_some() {
                        self.report.cache_hints += 1;
                    }
                    accesses.push((t, si, count));
                }
                for access in accesses {
                    self.account(MemSpace::Global, &[access], false);
                }
            }
            // A copy from this CTA's shared memory to a peer's: read from
            // the source in the background until it lands, at the peer's wait.
            Op::CpAsyncBulk {
                src,
                src_index,
                dst,
                dst_index,
                elem,
                bytes,
                direction: BulkDirection::S2S,
                completion: BulkCompletion::Mbarrier { buffer: mbarrier, index: mbarrier_index },
                peer: Some(peer),
                ..
            } => {
                let what = "cp.async.bulk.shared::cluster.shared::cta";
                let size = elem.bytes();
                for &t in active {
                    let count = self.get(*bytes, t) as u32 as usize;
                    if count == 0 || !count.is_multiple_of(16) {
                        return Err(
                            self.err(format!("{what} of {count} bytes is not one or more 16-byte units"), Some(t))
                        );
                    }
                    let n = count / size;
                    let si = self.checked_range(*src, self.index(*src_index, t), n, size, 16, t, "bulk copy source")?;
                    let di =
                        self.checked_range(*dst, self.index(*dst_index, t), n, size, 16, t, "bulk copy destination")?;
                    let byte = self.checked_range(
                        *mbarrier,
                        self.index(*mbarrier_index, t),
                        1,
                        8,
                        8,
                        t,
                        "bulk copy mbarrier",
                    )?;
                    self.need_clusters(t, what)?;
                    let block = self.mbarrier_block(Some(*peer), t, what)?;
                    let own = self.block_rank(t);
                    let key = (block, mbarrier.0, byte);
                    self.check_bulk_mbarrier(key, t, what)?;
                    self.async_source_may_read(*src, own, si..si + count, t, what)?;
                    let data = self.shared[self.home[src.0 as usize]][own].data[si..si + count].to_vec();
                    self.note_bulk_source_read(t, *src, own, si..si + count);
                    let reads = vec![AsyncRead { buffer: *src, block: own, bytes: si..si + count }];
                    self.issue_to_mbarrier(*dst, block, vec![(di, data)], reads, key, t, what, true)?;
                }
            }
            // st.async: a store into a peer's shared memory that completes
            // on the peer's mbarrier, as a bulk copy does.
            Op::StAsync { buffer, index, values, mbar, mbar_index, peer, .. } => {
                let what = "st.async";
                for &t in active {
                    let n = values.len();
                    let di = self.checked_range(*buffer, self.index(*index, t), n, 4, 4 * n, t, "st.async")?;
                    let byte =
                        self.checked_range(*mbar, self.index(*mbar_index, t), 1, 8, 8, t, "st.async mbarrier")?;
                    self.need_clusters(t, what)?;
                    let block = self.mbarrier_block(Some(*peer), t, what)?;
                    let key = (block, mbar.0, byte);
                    self.check_bulk_mbarrier(key, t, what)?;
                    let data: Vec<u8> = values.iter().flat_map(|&v| (self.get(v, t) as u32).to_le_bytes()).collect();
                    // A generic-proxy store: no proxy fence is involved.
                    self.issue_to_mbarrier(*buffer, block, vec![(di, data)], Vec::new(), key, t, what, false)?;
                }
            }
            Op::CpAsyncBulk {
                src,
                src_index,
                dst,
                dst_index,
                elem,
                bytes,
                direction: BulkDirection::S2G,
                cache_policy,
                byte_mask,
                ..
            } => {
                let size = elem.bytes();
                let mut accesses = Vec::new();
                if cache_policy.is_some() {
                    self.report.cache_hints += active.len() as u64;
                }
                for &t in active {
                    let count = self.get(*bytes, t) as u32 as usize;
                    if count == 0 || !count.is_multiple_of(16) {
                        return Err(self.err(
                            format!("cp.async.bulk of {count} bytes is not one or more whole 16-byte units"),
                            Some(t),
                        ));
                    }
                    let n = count / size;
                    let si = self.checked_range(
                        *src,
                        self.index(*src_index, t),
                        n,
                        size,
                        16,
                        t,
                        "cp.async.bulk source read",
                    )?;
                    let di = self.checked_range(
                        *dst,
                        self.index(*dst_index, t),
                        n,
                        size,
                        16,
                        t,
                        "cp.async.bulk destination write",
                    )?;
                    let (block, k) = (self.block_rank(t), self.clock_index(t));
                    self.bulk_store_may_read(*src, block, si..si + count, t)?;
                    if let Err(why) = self.async_proxy_may_read(*src, block, si..si + count, k) {
                        return Err(self.err(
                            format!(
                                "cp.async.bulk.global.shared::cta reads shared memory through the async proxy, but \
                                 {why}: write it, then fence_view_async_shared() (fence.proxy.async.shared::cta) in \
                                 the writing threads, ordered before the copy (a barrier)"
                            ),
                            Some(t),
                        ));
                    }
                    if self.pending_global > 0 {
                        self.check_pending_global(*dst, di..di + count, t, "overwrites")?;
                    }
                    if self.pdl.is_some() {
                        self.pdl_access(*dst, di..di + count, t, true)?;
                    }
                    let data = self.shared[self.home[src.0 as usize]][block].data[si..si + count].to_vec();
                    // `.cp_mask` writes the bytes of each 16-byte chunk
                    // whose bit of the mask is set, as runs.
                    let global_writes = match byte_mask {
                        None => vec![(*dst, di, data)],
                        Some(m) => {
                            let mask = self.get(*m, t) as u16;
                            let mut runs: Vec<(BufferId, usize, Vec<u8>)> = Vec::new();
                            for (i, &b) in data.iter().enumerate() {
                                if mask >> (i % 16) & 1 == 0 {
                                    continue;
                                }
                                match runs.last_mut() {
                                    Some((_, s, r)) if *s + r.len() == di + i => r.push(b),
                                    _ => runs.push((*dst, di + i, vec![b])),
                                }
                            }
                            runs
                        }
                    };
                    // The snapshot is a read by the issuing thread: a write
                    // by another thread must be ordered after it (and, while
                    // the copy still reads, after the covering wait).
                    self.note_bulk_source_read(t, *src, block, si..si + count);
                    let id = self.ops.len();
                    self.ops.push(AsyncOp {
                        thread: t,
                        agent: self.agent_slot(block),
                        writes: Vec::new(),
                        reads: vec![AsyncRead { buffer: *src, block, bytes: si..si + count }],
                        global_writes,
                        global_reduce: None,
                        results: Vec::new(),
                        trigger: Trigger::BulkGroup,
                        done: None,
                    });
                    self.any_async_reads = true;
                    self.pending_global += 1;
                    self.bulk_uncommitted[t].push(id);
                    if self.opts.async_completion == super::AsyncCompletion::Early {
                        self.land_bulk_store(id);
                    }
                    accesses.push((t, di, count));
                }
                self.account(MemSpace::Global, &accesses, true);
            }
            Op::CpAsyncBulk { completion, direction, .. } => {
                unreachable!("validated: a {direction:?} bulk copy does not complete through {completion:?}")
            }
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
                for &t in active {
                    let access = TmaIssue { desc: *desc, coords, mode, overrides: overrides.as_ref() };
                    let barrier = (*mbar, *mbar_index, *two_cta);
                    self.tma_load(*dst, *dst_index, *elem, &access, barrier, *mcast_mask, t)?;
                }
                if cache_policy.is_some() {
                    self.report.cache_hints += active.len() as u64;
                }
            }
            Op::TmaStore { src, src_index, elem, desc, coords, reduce, cache_policy, mode, overrides, .. } => {
                self.progress();
                for &t in active {
                    let access = TmaIssue { desc: *desc, coords, mode, overrides: overrides.as_ref() };
                    self.tma_store(*src, *src_index, *elem, &access, *reduce, t)?;
                }
                if cache_policy.is_some() {
                    self.report.cache_hints += active.len() as u64;
                }
            }
            // Prefetches have no architectural effect: their operands are checked and they
            // are counted.
            Op::TmaPrefetch { desc, coords, mode } => {
                for &t in active {
                    let access = TmaIssue { desc: *desc, coords, mode, overrides: None };
                    let what = "cp.async.bulk.prefetch.tensor (a TMA prefetch)";
                    let (d, start, offsets) = self.tma_issue_desc(&access, t, what)?;
                    // The access is checked as a load's would be (its filter base inside the
                    // bounding box, its offsets in range).
                    d.access_elements(&tma_access(mode, &start, &offsets))
                        .map_err(|e| self.err(format!("{what}: {e}"), Some(t)))?;
                    self.report.prefetches += 1;
                }
            }
            Op::PrefetchTensormap { desc } => {
                for &t in active {
                    self.tma_map(*desc, t, "prefetch.tensormap")?;
                    self.report.tensormap_prefetches += 1;
                }
            }
            // In-kernel tensor-map updates (exec/tensormap.rs).
            Op::TensormapReplace { addr, space, field, value } => {
                self.progress();
                self.tmap_replace(*addr, *space, *field, *value, active)?;
            }
            Op::FenceTensormapRelease => self.tmap_release(active),
            Op::FenceTensormapAcquire { addr } => self.tmap_acquire(*addr, active)?,
            Op::TensormapCpFenceproxy { dst, src } => {
                self.progress();
                self.tmap_cp_fenceproxy(*dst, *src, active)?;
            }
            Op::BulkPrefetch { buffer, index, elem, bytes } => {
                for &t in active {
                    let count = self.get(*bytes, t) as u32 as usize;
                    if count == 0 || !count.is_multiple_of(16) {
                        return Err(self.err(
                            format!("cp.async.bulk.prefetch of {count} bytes is not one or more 16-byte units"),
                            Some(t),
                        ));
                    }
                    let size = elem.bytes();
                    self.checked_range(
                        *buffer,
                        self.index(*index, t),
                        count / size,
                        size,
                        16,
                        t,
                        "cp.async.bulk.prefetch",
                    )?;
                    self.report.prefetches += 1;
                }
            }
            Op::CpAsyncBulkCommitGroup => {
                for &t in active {
                    let group = std::mem::take(&mut self.bulk_uncommitted[t]);
                    self.bulk_groups[t].push_back(group);
                }
            }
            Op::CpAsyncBulkWaitGroup { pending, read } => {
                for &t in active {
                    let older = self.bulk_groups[t].len().saturating_sub(*pending as usize);
                    // The covered copies have read their sources by now: the
                    // waiting thread reads them last, so another thread's
                    // overwrite must be ordered after this wait (a barrier).
                    let released: Vec<_> = (0..older)
                        .flat_map(|g| &self.bulk_groups[t][g])
                        .flat_map(|&id| &self.ops[id].reads)
                        .map(|r| (r.buffer, r.block, r.bytes.clone()))
                        .collect();
                    for (b, block, bytes) in released {
                        self.note_bulk_source_read(t, b, block, bytes);
                    }
                    if *read {
                        // Their sources are read: the shared memory may be
                        // reused, while their writes may still be pending.
                        for g in 0..older {
                            for id in self.bulk_groups[t][g].clone() {
                                self.ops[id].reads.clear();
                            }
                        }
                    } else {
                        for _ in 0..older {
                            let group = self.bulk_groups[t].pop_front().expect("counted above");
                            for id in group {
                                self.land_bulk_store(id);
                            }
                        }
                    }
                }
            }
            // A proxy fence of the warp: the async-shared ones let the async
            // proxy read (or overwrite) what the warp wrote before them.
            Op::FenceProxy { kind } => {
                if kind.orders_async_shared() {
                    let mut fenced: Vec<usize> = Vec::new();
                    for &t in active {
                        let k = self.clock_index(t);
                        if !fenced.contains(&k) {
                            fenced.push(k);
                            self.fence_proxy_async(k);
                        }
                    }
                }
            }
            // fence.acq_rel: the second half of `ld.relaxed; fence` acquire
            // patterns and the first half of `fence; st.relaxed` release
            // patterns. Every scope holds the warps of a block, the only
            // ones whose clocks meet.
            Op::Fence { .. } => self.fence_lanes(active),
            Op::Assert { cond, message } => {
                if let Some(&t) = active.iter().find(|&&t| self.get(*cond, t) & 1 == 0) {
                    if self.opts.assertions {
                        return Err(self.err(format!("device assertion failed: {message}"), Some(t)));
                    }
                    let loc = self.loc.map(crate::SourceLoc::from);
                    let text = format!("device assertion failed in block {:?}: {message}", self.block_of(t));
                    if !self.report.diagnostics.iter().any(|d| d.message == text && d.loc == loc) {
                        self.report.diagnostics.push(crate::Diagnostic::new(
                            crate::Severity::Warning,
                            "sim",
                            text,
                            loc,
                        ));
                    }
                }
            }
            Op::Printf { format, args } => {
                self.progress();
                for &t in active {
                    let values: Vec<(DType, u64)> = args.iter().map(|a| (ir.dtype(*a), self.get(*a, t))).collect();
                    self.report.printed.push(super::printf::format_printf(format, &values, to_f64));
                }
            }
            Op::InlinePtx { template, inputs, outputs, constraints, predicated, sim, .. } => {
                let Some(f) = sim else {
                    return Err(self.err(
                        format!(
                            "inline PTX `{template}` cannot be simulated; describe it with a Rust closure \
                             (InlinePtx::sim, InlineAsm::sim or inline_ptx_with_sim)"
                        ),
                        None,
                    ));
                };
                // The outputs tied to an input are the read-write operands;
                // the other inputs are read, the other outputs written.
                let entries = crate::ir::asm_constraints(constraints);
                let guard = predicated.then(|| inputs.last().expect("a predicated statement has a guard").value());
                let operands = &inputs[..inputs.len() - *predicated as usize];
                let mut tied = vec![None; outputs.len()];
                let mut reads = Vec::new();
                for (k, input) in operands.iter().enumerate() {
                    match entries.get(outputs.len() + k) {
                        Some(crate::ir::AsmConstraint::Tied(o)) => tied[*o] = Some(input.value()),
                        _ => reads.push(*input),
                    }
                }
                for &t in active {
                    let mut read_writes: Vec<u64> = tied.iter().flatten().map(|v| self.get(*v, t)).collect();
                    let written = if guard.is_none_or(|g| self.get(g, t) != 0) {
                        let bits: Vec<u64> = reads
                            .iter()
                            .map(|input| match input {
                                crate::ir::AsmInput::Value(v) => self.get(*v, t),
                                crate::ir::AsmInput::Address { .. } => 0,
                            })
                            .collect();
                        let written = f.call(&bits, &mut read_writes);
                        let want = tied.iter().filter(|t| t.is_none()).count();
                        if written.len() != want {
                            return Err(self.err(
                                format!("inline PTX simulation returned {} values, not {want}", written.len()),
                                Some(t),
                            ));
                        }
                        written
                    } else {
                        // Not run: read-write operands keep their values,
                        // the other outputs are 0.
                        vec![0; tied.iter().filter(|t| t.is_none()).count()]
                    };
                    let (mut written, mut read_writes) = (written.into_iter(), read_writes.into_iter());
                    for (o, tie) in outputs.iter().zip(&tied) {
                        let bits = if tie.is_some() { read_writes.next() } else { written.next() };
                        self.set(*o, t, mask(bits.expect("one value per output"), ir.dtype(*o).bits()));
                    }
                }
            }
            Op::Redux { result, kind, value, mask, abs, nan } => {
                for (base, lanes) in self.warps_of(active) {
                    let masks = self.member_masks("warp_redux_sync", base, &lanes, *mask)?;
                    let results: Vec<u64> = masks
                        .iter()
                        .map(|&members| {
                            let inputs: Vec<u32> = (0..self.warp_size(base))
                                .filter(|&o| members >> o & 1 == 1)
                                .map(|o| self.get(*value, base + o) as u32)
                                .collect();
                            redux(*kind, *abs, *nan, &inputs) as u64
                        })
                        .collect();
                    for (&l, bits) in lanes.iter().zip(results) {
                        self.set(*result, base + l, bits);
                    }
                }
            }
            Op::CarryInst { result, func, args } => {
                let ty = ir.dtype(args[0]);
                for &t in active {
                    let carry_in = match (func.carry_in, self.carry[t]) {
                        (false, _) => false,
                        (true, Some((flag, borrow))) if borrow == (func.op == crate::ir::CarryOp::Sub) => flag,
                        (true, Some(_)) => {
                            return Err(self.err(
                                format!(
                                    "{} reads a carry flag that the other kind of instruction (an addition or a \
                                     subtraction) set",
                                    func.cute_name()
                                ),
                                Some(t),
                            ));
                        }
                        (true, None) => {
                            return Err(self.err(
                                format!(
                                    "{} reads the carry flag, which no instruction has set in this block of \
                                     straight-line code",
                                    func.cute_name()
                                ),
                                Some(t),
                            ));
                        }
                    };
                    let operands: Vec<u64> = args.iter().map(|a| self.get(*a, t)).collect();
                    let (bits, carry_out) = super::carry::carry_inst(*func, ty, &operands, carry_in);
                    if func.carry_out {
                        self.carry[t] = Some((carry_out, func.op == crate::ir::CarryOp::Sub));
                    }
                    self.set(*result, t, bits);
                }
            }
            Op::MlirOp { text, operands, results, sim, .. } => {
                let Some(f) = sim else {
                    return Err(self.err(
                        format!("the MLIR operation `{text}` cannot be simulated; describe it with MlirOp::sim"),
                        None,
                    ));
                };
                for &t in active {
                    let bits: Vec<u64> = operands
                        .iter()
                        .map(|operand| match operand {
                            crate::ir::AsmInput::Value(v) => self.get(*v, t),
                            crate::ir::AsmInput::Address { .. } => 0,
                        })
                        .collect();
                    let produced = f.call(&bits, &mut []);
                    if produced.len() != results.len() {
                        return Err(self.err(
                            format!(
                                "MLIR operation simulation returned {} values, not {}",
                                produced.len(),
                                results.len()
                            ),
                            Some(t),
                        ));
                    }
                    for (r, bits) in results.iter().zip(produced) {
                        self.set(*r, t, mask(bits, ir.dtype(*r).bits()));
                    }
                }
            }
            Op::CpAsyncMbarrierArrive { buffer, index, noinc } => {
                self.progress();
                for &t in active {
                    let byte =
                        self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "cp.async.mbarrier.arrive")?;
                    let key = (self.block_rank(t), buffer.0, byte);
                    // It tracks every copy the thread issued that is still
                    // in flight, committed or not.
                    let copies = self.in_flight[t].clone();
                    let floor = if self.async_rng.is_some() { self.bulk_time[key.0] } else { 0 };
                    let Some(state) = self.mbarriers.get_mut(&key) else {
                        return Err(
                            self.err("cp.async.mbarrier.arrive on an mbarrier that was never initialised", Some(t))
                        );
                    };
                    let (owed, left) = (state.deferred.len() as u32, state.arrivals_left);
                    if *noinc && left == 0 {
                        // Its arrivals are in, but bulk copies' bytes are not.
                        return Err(self.err(
                            "cp_async_mbarrier_arrive_noinc on an mbarrier whose phase already has all its arrivals \
                             and only waits for transaction bytes: .noinc does not raise the pending count, so the \
                             arrival has no place in the phase (count it in mbarrier_init, or use \
                             cp_async_mbarrier_arrive)",
                            Some(t),
                        ));
                    }
                    if *noinc && owed >= left {
                        return Err(self.err(
                            format!(
                                "cp_async_mbarrier_arrive_noinc on an mbarrier whose phase is waiting for {left} \
                                 arrival(s), all owed by earlier cp.async.mbarrier.arrive: .noinc does not raise the \
                                 pending count, so the mbarrier's count must include one arrival per call"
                            ),
                            Some(t),
                        ));
                    }
                    if !*noinc {
                        // The pending count goes up by one first.
                        if left + 1 >= 1 << 20 {
                            return Err(self.err(
                                format!(
                                    "cp_async_mbarrier_arrive raises an mbarrier's pending count past 2^20 - 1 \
                                     ({left} + 1)"
                                ),
                                Some(t),
                            ));
                        }
                        state.arrivals_left += 1;
                    }
                    state.deferred.push(DeferredArrival { copies: copies.clone(), floor, commit: false });
                    for id in copies {
                        self.ops[id].trigger = Trigger::CpAsync { tracked: true };
                    }
                    if self.opts.async_completion == super::AsyncCompletion::Early || self.async_rng.is_some() {
                        // Its copies are complete (early, or by chance): the
                        // arrive-on fires now.
                        self.fire_ready_arrivals(key);
                    }
                }
            }
            // A hint that reads nothing: PTX allows global and local
            // addresses (the validator checks the space), and an address out
            // of bounds is harmless on hardware, so nothing is checked.
            Op::Prefetch { .. } => {}
            // Little-endian bit concatenation: the w bits of part k at bits k*w.
            Op::Pack { result, parts } => {
                let width = ir.dtype(parts[0]).bits();
                self.lanes(active, *result, |m, t| {
                    Ok(parts
                        .iter()
                        .enumerate()
                        .fold(0u64, |acc, (k, &p)| acc | mask(m.get(p, t), width) << (k * width)))
                })?;
            }
            Op::Unpack { results, x } => {
                let width = ir.dtype(results[0]).bits();
                for &t in active {
                    let bits = self.get(*x, t);
                    for (k, &r) in results.iter().enumerate() {
                        self.set(r, t, mask(bits >> (k * width), width));
                    }
                }
            }
            Op::HalfInst { result, func, ty, args, mods, predicate } => {
                self.lanes(active, *result, |m, t| {
                    let mut operands = [0u64; 3];
                    for (k, a) in args.iter().enumerate() {
                        operands[k] = m.get(*a, t);
                    }
                    // Where the predicate is false the instruction does not
                    // run, and the result is the first operand.
                    if predicate.is_some_and(|p| m.get(p, t) == 0) {
                        return Ok(operands[0]);
                    }
                    Ok(super::float::half_inst(*func, *ty, *mods, &operands[..args.len()]))
                })?;
            }
            Op::Cvt { result, func, args } => {
                self.lanes(active, *result, |m, t| {
                    let operands: Vec<u64> = args.iter().map(|a| m.get(*a, t)).collect();
                    Ok(super::float::cvt_inst(*func, &operands))
                })?;
            }
            // Breaking an assumption is undefined behaviour on the GPU, so
            // the simulator checks every thread's value, as launches check a
            // `param_extent`'s divisor.
            // A buffer's address in the simulator's address map.
            Op::PtrToInt { result, buffer, index, elem } => {
                let bits = ir.dtype(*result).bits();
                for &t in active {
                    let byte = self.index(*index, t) * elem.bytes() as i64;
                    let addr = self.address_of(*buffer, 0).wrapping_add(byte as u64);
                    self.set(*result, t, mask(addr, bits));
                }
            }
            Op::CvtAddr { result, x, from, to } => {
                let (xb, rb) = (ir.dtype(*x).bits(), ir.dtype(*result).bits());
                for &t in active {
                    let a = mask(self.get(*x, t), xb);
                    let out = self.cvt_addr(a, *from, *to).map_err(|m| self.err(m, Some(t)))?;
                    self.set(*result, t, mask(out, rb));
                }
            }
            Op::Assume { result, x, divby, pow2 } => {
                let ty = ir.dtype(*x);
                for &t in active {
                    let bits = self.get(*x, t);
                    let value =
                        if ty.is_signed() { sext(bits, ty.bits()) as i128 } else { mask(bits, ty.bits()) as i128 };
                    if value % *divby as i128 != 0 {
                        return Err(self.err(
                            format!("assume: the value {value} is not a multiple of {divby}, as the kernel assumes"),
                            Some(t),
                        ));
                    }
                    if *pow2 && (value <= 0 || value & (value - 1) != 0) {
                        return Err(self.err(
                            format!("assume: the value {value} is not a power of two, as the kernel assumes"),
                            Some(t),
                        ));
                    }
                    self.set(*result, t, bits);
                }
            }
        }
        // PTX keeps the carry flag only through straight-line code: control
        // flow, calls (`printf` calls `vprintf`), a hand-written MLIR
        // operation, inline PTX that branches or uses the flag, and a 64-bit
        // division that LLVM lowers with a branch lose it (the validator
        // rejects a carry-in after them).
        if crate::ir::carry_flag_breaker(&inst.op, |v| ir.dtype(v), |v| self.constants.contains(&v)).is_some() {
            for &t in active {
                self.carry[t] = None;
            }
        }
        Ok(())
    }

    /// The member mask of each of `lanes` (of the warp starting at thread
    /// `base`) at a warp-collective instruction, after checking what the PTX
    /// ISA leaves undefined: every lane must be in its own mask, and every
    /// lane of a mask that the warp has (the last warp of a block can be
    /// short) must execute the instruction with the same mask (lanes in
    /// other convergence groups have joined it at [`Self::rendezvous`]).
    fn member_masks(&self, what: &str, base: usize, lanes: &[usize], mask: ValueId) -> R<Vec<u32>> {
        let warp = self.warp_of(base);
        let masks: Vec<u32> = lanes.iter().map(|&l| self.get(mask, base + l) as u32).collect();
        for (&l, &m) in lanes.iter().zip(&masks) {
            if m >> l & 1 == 0 {
                return Err(self.err(
                    format!("{what} executed by lane {l} of warp {warp}, which is not in its member mask {m:#010x}"),
                    Some(base + l),
                ));
            }
        }
        for (&l, &m) in lanes.iter().zip(&masks) {
            for k in (0..self.warp_size(base)).filter(|&k| m >> k & 1 == 1) {
                match lanes.binary_search(&k) {
                    Err(_) => {
                        return Err(self.err(
                            format!(
                                "{what} with member mask {m:#010x} is executed by lane {l} of warp {warp}, but lane \
                                 {k} of the mask is not active here: every lane of the mask must execute it, and lane {k} \
                                 waits at the end of a branch or loop for the others to reconverge"
                            ),
                            Some(base + l),
                        ));
                    }
                    Ok(i) if masks[i] != m => {
                        return Err(self.err(
                            format!(
                                "lane {l} of warp {warp} executes {what} with member mask {m:#010x}, which \
                                 includes lane {k}, but lane {k} executes it with {:#010x}",
                                masks[i]
                            ),
                            Some(base + l),
                        ));
                    }
                    Ok(_) => {}
                }
            }
        }
        Ok(masks)
    }

    /// Lanes in the warp starting at thread `base` (the last warp of a block
    /// can be short).
    pub(super) fn warp_size(&self, base: usize) -> usize {
        (self.threads_per_block - base % self.threads_per_block).min(32)
    }

    pub(super) fn mma(
        &mut self,
        kind: MmaKind,
        d: &[ValueId],
        a: &[ValueId],
        b: &[ValueId],
        c: &[ValueId],
        e: Option<ValueId>,
        active: &[usize],
    ) -> R {
        let bases = self.full_warps(active, if kind.is_sparse() { "mma.sp" } else { "mma.sync" })?;
        let (m, n, k) = kind.shape_mnk();
        let (ta, tb, tc) = kind.dtypes();
        // TF32 operands: the tensor core ignores the low 13 mantissa bits.
        let tf32 = |x: f64| -> f64 { if kind.is_tf32() { tf32_truncate(x) } else { x } };
        // FP8 operands decode exactly (their products are exact in f32).
        let to_f64 = |ty: DType, bits: u64| match ty.float_format() {
            Some(f) if ty.is_narrow_float() => crate::numeric::decode(f, bits as u32),
            _ => to_f64(ty, bits),
        };
        for base in bases {
            let mut am = vec![vec![0f64; k]; m];
            let mut bm = vec![vec![0f64; n]; k];
            let mut cm = vec![vec![0f64; n]; m];
            for lane in 0..32 {
                let t = base + lane;
                for (i, v) in a.iter().enumerate() {
                    let (r, col) = mma_a_coord(kind, lane, i);
                    am[r][col] = tf32(to_f64(ta, self.get(*v, t)));
                }
                for (i, v) in b.iter().enumerate() {
                    let (kk, col) = mma_b_coord(kind, lane, i);
                    bm[kk][col] = tf32(to_f64(tb, self.get(*v, t)));
                }
                for (i, v) in c.iter().enumerate() {
                    let (r, col) = mma_c_coord(kind, lane, i);
                    cm[r][col] = to_f64(tc, self.get(*v, t));
                }
            }
            if let Some(e) = e {
                // `am` holds the compressed A in its first k / 2 columns:
                // spread each kept pair to the positions its metadata names.
                let mut dense = vec![vec![0f64; k]; m];
                for (r, row) in dense.iter_mut().enumerate() {
                    for chunk in 0..k / 4 {
                        let (lane, nibble) = sparse_metadata_slot(k, 0, r, chunk);
                        let bits = (self.get(e, base + lane) >> (4 * nibble) & 0xf) as u32;
                        let Some((i0, i1)) = sparse_indices(bits) else {
                            return Err(self.err(
                                format!(
                                    "{} metadata nibble {nibble} of lane {lane} (row {r}, columns {} to {} of A) is \
                                     {bits:#06b}, which is not one of the valid index pairs 0b0100, 0b1000, 0b1001, \
                                     0b1100, 0b1101 and 0b1110 (two increasing positions in the chunk of four)",
                                    kind.ptx(),
                                    4 * chunk,
                                    4 * chunk + 3
                                ),
                                Some(base + lane),
                            ));
                        };
                        row[4 * chunk + i0] = am[r][2 * chunk];
                        row[4 * chunk + i1] = am[r][2 * chunk + 1];
                    }
                }
                am = dense;
            }
            for lane in 0..32 {
                for (i, v) in d.iter().enumerate() {
                    let (r, col) = mma_c_coord(kind, lane, i);
                    let bits = if tc == DType::F64 {
                        (0..k).fold(cm[r][col], |acc, kk| acc + am[r][kk] * bm[kk][col]).to_bits()
                    } else {
                        let acc = (0..k).fold(cm[r][col] as f32, |acc, kk| acc + (am[r][kk] * bm[kk][col]) as f32);
                        from_f32(tc, acc)
                    };
                    self.set(*v, base + lane, bits);
                }
            }
            self.report.mma_instructions += 1;
        }
        Ok(())
    }
}
