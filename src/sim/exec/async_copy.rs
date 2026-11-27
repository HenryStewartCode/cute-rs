// SPDX-License-Identifier: MIT OR Apache-2.0

//! Asynchronous copies: cp.async, bulk copies, TMA loads and stores, and the checks on their in-flight reads and writes.

use super::*;

impl<'a> Machine<'a> {
    /// Whether no asynchronous operation reads memory in the background
    /// (the common case, which skips [`Self::check_async_reads`]).
    pub(super) fn pending_reads_free(&self) -> bool {
        !self.any_async_reads
    }

    /// Report a read by `active` of an operand that an asynchronous
    /// operation still has to write (a register result, such as a WGMMA
    /// accumulator).
    pub(super) fn check_pending_operands(&self, operands: &[ValueId], active: &[usize]) -> R {
        for &v in operands {
            for &t in active {
                if let Some(&id) = self.pending_results.get(&(v, t)) {
                    let issuer = self.ops[id].thread;
                    if self.ops[id].trigger == Trigger::TmemLd {
                        return Err(self.err(
                            format!(
                                "reads a register of a tcgen05.ld (issued by warp {}) before a tcgen05.wait::ld of this \
                                 warp has completed it (fence_view_async_tmem_load): the registers are undefined until \
                                 then",
                                self.warp_of(issuer)
                            ),
                            Some(t),
                        ));
                    }
                    if self.ops[id].trigger == Trigger::Wgmma {
                        return Err(self.err(
                            format!(
                                "reads an accumulator of a wgmma.mma_async (issued by warp {}) before a \
                                 wgmma.wait_group of this warp has completed it (warpgroup::wait_group)",
                                self.warp_of(issuer)
                            ),
                            Some(t),
                        ));
                    }
                    return Err(self.err(
                        format!(
                            "reads the result of an asynchronous operation before waiting for it (issued by warp {} \
                             of {})",
                            self.warp_of(issuer),
                            self.block_name(self.block_rank(issuer))
                        ),
                        Some(t),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Report a write by thread `t` of bytes `range` of shared buffer `b`
    /// that an asynchronous operation in flight still reads.
    pub(super) fn check_async_reads(&self, b: BufferId, range: std::ops::Range<usize>, t: usize) -> R {
        self.check_async_reads_in(b, self.smem_block(t), range, t)
    }

    /// [`Self::check_async_reads`] of shared memory in `block`.
    fn check_async_reads_in(&self, b: BufferId, block: usize, range: std::ops::Range<usize>, t: usize) -> R {
        for op in self.ops.iter().filter(|op| op.in_flight()) {
            for r in &op.reads {
                if r.buffer == b && r.block == block && r.bytes.start < range.end && range.start < r.bytes.end {
                    let byte = range.start.max(r.bytes.start);
                    return Err(self.err(
                        format!(
                            "data race: warp {} overwrites {}, which an asynchronous operation issued by {} still \
                             reads (wait for it to complete first)",
                            self.warp_of(t),
                            smem_place(self.ir.buffer(b), byte),
                            self.slot_name(self.slot_of(self.clock_index(op.thread)), block)
                        ),
                        Some(t),
                    ));
                }
            }
        }
        Ok(())
    }

    /// `fence.proxy.async` (unscoped, `.shared::cta` or `.shared::cluster`)
    /// by warp `k`: the async proxy may then read (or overwrite) what the
    /// warp wrote before it, once the fence is ordered before the issue.
    pub(super) fn fence_proxy_async(&mut self, k: usize) {
        let slot = self.slot_of(k);
        let now = self.clocks[k][slot];
        self.proxy_fences[k].push(now);
        self.clocks[k][slot] += 1;
    }

    /// Before a bulk copy by thread `t` reads bytes `range` of shared buffer
    /// `b` of `block`: no asynchronous copy may still be writing them, and
    /// their last write must be ordered before the issue.
    pub(super) fn bulk_store_may_read(&self, b: BufferId, block: usize, range: std::ops::Range<usize>, t: usize) -> R {
        let k = self.clock_index(t);
        let (slot, clock) = (self.slot_of(k), &self.clocks[k]);
        for i in range {
            let m = &self.shared[self.home[b.0 as usize]][block].meta[i];
            let place = smem_place(self.ir.buffer(b), i);
            if m.inflight > 0 {
                return Err(self.err(
                    format!(
                        "cp.async.bulk.global.shared::cta reads {place}, which an asynchronous copy is still writing \
                         (wait for that copy first)"
                    ),
                    Some(t),
                ));
            }
            if m.written && !m.write_ordered_before(slot, clock) {
                return Err(self.err(
                    format!(
                        "data race: cp.async.bulk.global.shared::cta reads {place}, but nothing orders its last \
                         write, by {}, before the copy (a barrier or an mbarrier wait)",
                        self.slot_name(m.w_slot as usize, block)
                    ),
                    Some(t),
                ));
            }
        }
        Ok(())
    }

    /// Record a read of bytes `range` of shared buffer `b` of `block` by
    /// thread `t` now, for the race detector: what a bulk copy reads on the
    /// thread's behalf, at its issue and at the wait that covers it.
    pub(super) fn note_bulk_source_read(&mut self, t: usize, b: BufferId, block: usize, range: std::ops::Range<usize>) {
        if !self.opts.detect_races {
            return;
        }
        let k = self.clock_index(t);
        let slot = self.slot_of(k);
        let clock = &self.clocks[k];
        let read = ReadEpoch { slot: slot as u16, unit: 0, clock: clock[slot] };
        let sh = &mut self.shared[self.home[b.0 as usize]][block];
        for i in range {
            sh.reads.record(i, read, clock, true);
        }
    }

    /// Report an access (`what`: reads, overwrites) by thread `t` of bytes
    /// `range` of global buffer `b` that a bulk copy still has to write.
    pub(super) fn check_pending_global(&self, b: BufferId, range: std::ops::Range<usize>, t: usize, what: &str) -> R {
        self.check_pending_global_for(b, range, t, what, false)
    }

    /// [`Self::check_pending_global`], for a TMA reduce store when `reduce`: reductions in
    /// flight on the same bytes are atomic with it, so only plain writes conflict.
    fn check_pending_global_for(
        &self,
        b: BufferId,
        range: std::ops::Range<usize>,
        t: usize,
        what: &str,
        reduce: bool,
    ) -> R {
        for op in self.ops.iter().filter(|op| op.in_flight() && !(reduce && op.global_reduce.is_some())) {
            for (dst, byte, data) in &op.global_writes {
                if *dst == b && *byte < range.end && range.start < byte + data.len() {
                    let at = range.start.max(*byte);
                    let elem = self.ir.buffer(b).elem;
                    return Err(self.err(
                        format!(
                            "{what} element {} of `{}`, which a cp.async.bulk{} issued by thread {} of {} is still \
                             writing: its bulk async-group must complete first (cp_async_bulk_wait_group(n, false) by \
                             the issuing thread, then a barrier for the others)",
                            at / elem.bytes().max(1),
                            self.ir.buffer(b).name,
                            if op.global_reduce.is_some() { " (a TMA reduce store)" } else { "" },
                            op.thread % self.threads_per_block,
                            self.block_name(self.block_rank(op.thread))
                        ),
                        Some(t),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Complete bulk copy `id` to global memory: its source is read and its
    /// bytes land in global memory (once; it may have completed early).
    pub(super) fn land_bulk_store(&mut self, id: usize) {
        if !self.ops[id].in_flight() {
            return;
        }
        self.ops[id].reads.clear();
        let reduce = self.ops[id].global_reduce;
        for (dst, byte, data) in std::mem::take(&mut self.ops[id].global_writes) {
            let mem =
                &mut self.globals[dst.0 as usize].as_mut().expect("global buffers are bound")[byte..byte + data.len()];
            match reduce {
                None => mem.copy_from_slice(&data),
                Some((kind, elem)) => {
                    let n = elem.bytes();
                    for (old, new) in mem.chunks_exact_mut(n).zip(data.chunks_exact(n)) {
                        let r = tma_reduce(kind, elem, read_le(old), read_le(new));
                        write_le(old, r);
                    }
                }
            }
        }
        self.pending_global -= 1;
        self.ops[id].done = Some(0);
        self.progress();
    }

    /// Once every warp of `block` has exited, the bulk copies to global
    /// memory its threads issued and whose sources are read land (a kernel's
    /// writes are complete when it ends); those still reading shared memory
    /// are reported by [`Self::check_block_exit`].
    pub(super) fn land_bulk_stores_at_exit(&mut self, block: usize) {
        let w = self.warps_per_block;
        if self.pending_global == 0 || !(0..w).all(|warp| matches!(self.warps[block * w + warp].status, Status::Exited))
        {
            return;
        }
        let ids: Vec<usize> = (0..self.ops.len())
            .filter(|&id| {
                let op = &self.ops[id];
                op.in_flight()
                    && op.trigger == Trigger::BulkGroup
                    && op.reads.is_empty()
                    && self.block_rank(op.thread) == block
            })
            .collect();
        for id in ids {
            self.land_bulk_store(id);
        }
    }

    /// May the async proxy read bytes `range` of shared buffer `b` of
    /// `block` for an operation that warp `issuer` issues now? A generic
    /// write must be followed by a `fence.proxy.async` in the writing warp,
    /// ordered before the issue. Returns why not.
    pub(super) fn async_proxy_may_read(
        &self,
        b: BufferId,
        block: usize,
        range: std::ops::Range<usize>,
        issuer: usize,
    ) -> Result<(), String> {
        let clock = &self.clocks[issuer];
        let base = self.cluster_base(block) * self.warps_per_block;
        for i in range {
            let m = &self.shared[self.home[b.0 as usize]][block].meta[i];
            let writer = m.w_slot as usize;
            if !m.written || self.is_agent(writer) {
                continue;
            }
            let k = base + (writer / self.stride) * self.warps_per_block + writer % self.stride;
            let fence = self.proxy_fences[k].iter().find(|&&f| f >= m.w_clock);
            let place = smem_place(self.ir.buffer(b), i);
            match fence {
                None => {
                    return Err(format!(
                        "{place} was written by {} through the generic proxy, without a fence.proxy.async since",
                        self.slot_name(writer, block)
                    ));
                }
                Some(&f) if k != issuer && clock[writer] < f => {
                    return Err(format!(
                        "{place} was written by {} and fenced (fence.proxy.async), but nothing orders the fence \
                         before this operation's issue",
                        self.slot_name(writer, block)
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    /// Issue a test-only asynchronous operation by thread `t`, with the
    /// effects of TMA and WGMMA: shared-memory writes, background reads and
    /// register results.
    #[cfg(test)]
    pub(super) fn issue_test_op(
        &mut self,
        t: usize,
        writes: Vec<AsyncWrite>,
        reads: Vec<AsyncRead>,
        results: Vec<(ValueId, usize, u64)>,
    ) -> R<usize> {
        for w in &writes {
            self.smem_claim_write(w.dst, w.dst_byte, w.data.len(), t, true, false)?;
        }
        let id = self.ops.len();
        self.any_async_reads |= !reads.is_empty();
        for &(v, thread, _) in &results {
            self.pending_results.insert((v, thread), id);
        }
        let agent = self.agent_slot(self.block_rank(t));
        self.ops.push(AsyncOp {
            thread: t,
            agent,
            writes,
            reads,
            global_writes: Vec::new(),
            global_reduce: None,
            results,
            trigger: Trigger::Test,
            done: None,
        });
        Ok(id)
    }

    /// The tensor map a TMA instruction (`what`) of thread `t` addresses, decoded, with its
    /// `.override::*` operands applied, and the
    /// coordinates and im2col offsets it gives, checked against the descriptor's rank and
    /// the modes the simulator models (no interleave, no SM100 swizzle atoms).
    pub(super) fn tma_issue_desc(
        &self,
        a: &TmaIssue<'_>,
        t: usize,
        what: &str,
    ) -> R<(crate::tma::TmaDesc, Vec<i64>, Vec<i64>)> {
        let mut d = self.tma_map(a.desc, t, what)?;
        let rank = d.rank as usize;
        let im2col_access = matches!(a.mode, crate::ir::TmaMode::Im2col { .. } | crate::ir::TmaMode::Im2colNoOffs);
        if im2col_access != d.im2col.is_some() {
            return Err(self.err(
                format!(
                    "{what}: {} ({d})",
                    if im2col_access {
                        "an im2col access with a tiled tensor map (an im2col copy takes its atom's im2col descriptor)"
                    } else {
                        "a tile-mode access with an im2col tensor map (it takes im2col coordinates and offsets: an \
                         im2col atom's copy)"
                    }
                ),
                Some(t),
            ));
        }
        let want = if a.mode.is_rows4() { 5 } else { rank };
        if a.coords.len() != want {
            return Err(self.err(
                format!(
                    "{what}: {} coordinates for a tensor map of rank {rank}: {}",
                    a.coords.len(),
                    if a.mode.is_rows4() {
                        "the gather4 and scatter4 modes take five (a column and four rows)".to_string()
                    } else {
                        "the dimensions must match".to_string()
                    }
                ),
                Some(t),
            ));
        }
        if d.interleave != crate::tma::TensorMapInterleave::none || d.swizzle.is_unsupported_atom_mode() {
            return Err(self.err(
                format!(
                    "{what}: the simulator models TMA copies without interleave, and of the SM100 swizzle atoms only \
                     s128b_atom_32b ({d})"
                ),
                Some(t),
            ));
        }
        let start: Vec<i64> = a.coords.iter().map(|&c| self.get(c, t) as u32 as i32 as i64).collect();
        let offsets: Vec<i64> = a.mode.offsets().iter().map(|&o| i64::from(self.get(o, t) as u16)).collect();
        if let Some(o) = a.overrides {
            let addr = self.get(o.global_address, t);
            if !addr.is_multiple_of(16) {
                return Err(self
                    .err(format!("{what}: the overriding global address {addr:#x} is not 16-byte aligned"), Some(t)));
            }
            d.global_address = addr;
            if !o.dims.is_empty() {
                if start.iter().any(|&c| c != 0) {
                    return Err(self.err(
                        format!(
                            "{what}: an attribute override needs zero coordinates (the PTX ISA leaves others \
                             undefined), got {start:?}"
                        ),
                        Some(t),
                    ));
                }
                for (i, &v) in o.dims.iter().enumerate() {
                    d.global_dims[i] = self.get(v, t) & 0xff;
                }
                let upper = o.upper_strides.map_or(0, |u| self.get(u, t) & 0xffff);
                for (k, &v) in o.lower_strides.iter().enumerate() {
                    let lower = self.get(v, t) & 0xffff_ffff;
                    d.global_strides[k] = (lower + (((upper >> (4 * k)) & 0xf) << 32)) << 4;
                }
            }
        }
        Ok((d, start, offsets))
    }

    /// The tensor map at the generic address in `desc`, decoded, for `what` of thread `t`.
    pub(super) fn tma_map(&self, desc: ValueId, t: usize, what: &str) -> R<crate::tma::TmaDesc> {
        // The descriptor: 128 bytes at a 64-byte-aligned generic address.
        let desc_addr = self.get(desc, t);
        let (db, dbyte) = self
            .resolve(AddressSpace::Generic, desc_addr)
            .map_err(|why| self.err(format!("{what}: the tensor map's address: {why}"), Some(t)))?;
        if !desc_addr.is_multiple_of(crate::tma::TENSOR_MAP_ALIGN as u64) {
            return Err(self.err(
                format!(
                    "{what}: the tensor map at {desc_addr:#x} is not {}-byte aligned",
                    crate::tma::TENSOR_MAP_ALIGN
                ),
                Some(t),
            ));
        }
        let bytes = match self.ir.buffer(db).space {
            MemSpace::Global | MemSpace::Constant => self.globals[db.0 as usize]
                .as_ref()
                .and_then(|g| g.get(dbyte..dbyte + crate::tma::TENSOR_MAP_BYTES))
                .map(<[u8]>::to_vec),
            _ => None,
        };
        let Some(bytes) = bytes else {
            return Err(self.err(
                format!(
                    "{what}: the tensor map's address {desc_addr:#x} does not hold 128 bytes of global or parameter \
                     memory"
                ),
                Some(t),
            ));
        };
        // A descriptor in memory: its generic writes released to the
        // tensormap proxy and acquired by this thread (prefetches are hints, unchecked).
        let written = if self.tmaps.enabled && !what.contains("prefetch") {
            self.tmap_check_read(db, dbyte, t, what)?
        } else {
            false
        };
        if !crate::tma::TmaDesc::is_sim_encoding(&bytes) && self.ir.buffer(db).space == MemSpace::Global {
            return Err(self.err(
                format!(
                    "{what}: the 128 bytes at byte {dbyte} of `{}` do not hold a tensor map (uninitialised memory, or \
                     not the copy of a descriptor): copy one there with copy_tensormap (or its 16 words) before a TMA \
                     copy uses it",
                    self.ir.buffer(db).name
                ),
                Some(t),
            ));
        }
        let d = crate::tma::TmaDesc::from_sim_bytes(&bytes).map_err(|e| self.err(format!("{what}: {e}"), Some(t)))?;
        if written && let Err(e) = d.validate() {
            return Err(self.err(
                format!(
                    "{what}: the tensor map at byte {dbyte} of `{}`, written in the kernel, is not one \
                     cuTensorMapEncodeTiled would encode: {e}",
                    self.ir.buffer(db).name
                ),
                Some(t),
            ));
        }
        Ok(d)
    }

    /// Before a bulk copy or `st.async` by thread `t` (`what`) reports to the mbarrier
    /// `key`: it was initialised and, in another CTA, its initialisation published to
    /// `t`'s CTA, which has not exited.
    pub(super) fn check_bulk_mbarrier(&self, key: MbarrierKey, t: usize, what: &str) -> R {
        if key.0 != self.block_rank(t) {
            self.check_remote_mbarrier(key, t, what)?;
        }
        if !self.mbarriers.contains_key(&key) {
            let whose = if key.0 == self.block_rank(t) {
                String::new()
            } else {
                format!(" of CTA {}", key.0 % self.cluster_size)
            };
            return Err(self.err(format!("{what} reports to an mbarrier{whose} that was never initialised"), Some(t)));
        }
        Ok(())
    }

    /// Before a copy by thread `t` (`what`) reads bytes `range` of shared buffer `b` of
    /// `block` through the async proxy in the background: no asynchronous copy may still
    /// write them, their last write must be ordered before the issue, and generic writes
    /// must have been fenced.
    pub(super) fn async_source_may_read(
        &self,
        b: BufferId,
        block: usize,
        range: std::ops::Range<usize>,
        t: usize,
        what: &str,
    ) -> R {
        self.bulk_store_may_read(b, block, range.clone(), t)?;
        if let Err(why) = self.async_proxy_may_read(b, block, range, self.clock_index(t)) {
            return Err(self.err(
                format!(
                    "{what} reads shared memory through the async proxy, but {why}: write it, then \
                     fence_view_async_shared() (fence.proxy.async.shared::cta) in the writing threads, ordered before \
                     the copy (a barrier)"
                ),
                Some(t),
            ));
        }
        Ok(())
    }

    /// The CTAs a copy by thread `t` writes: its own, or with `mask` (`what`'s
    /// `ctaMask`) those of its cluster whose bits are set.
    pub(super) fn cluster_targets(&self, mask: Option<ValueId>, t: usize, what: &str) -> R<Vec<usize>> {
        let own = self.block_rank(t);
        let Some(mask) = mask else { return Ok(vec![own]) };
        self.need_clusters(t, what)?;
        let bits = self.get(mask, t) as u16;
        let n = self.cluster_size;
        if n < 16 && bits >> n != 0 {
            return Err(self.err(
                format!(
                    "{what}: the CTA mask {bits:#06x} has bits beyond the cluster's {n} CTA(s) (ranks 0 to {}): \
                     PTX leaves a copy to a CTA outside the cluster undefined",
                    n - 1
                ),
                Some(t),
            ));
        }
        if bits == 0 {
            return Err(self.err(format!("{what}: the CTA mask is 0, so the copy writes no CTA"), Some(t)));
        }
        let base = self.cluster_base(own);
        Ok((0..n).filter(|r| bits >> r & 1 == 1).map(|r| base + r).collect())
    }

    /// Report `what` by thread `t` in a launch without clusters.
    pub(super) fn need_clusters(&self, t: usize, what: &str) -> R {
        if self.cfg.cluster == [1, 1, 1] {
            return Err(self.err(
                format!(
                    "{what} addresses other CTAs of the cluster, but the launch has no clusters (launch with \
                     LaunchConfig::with_cluster)"
                ),
                Some(t),
            ));
        }
        Ok(())
    }

    /// Before an asynchronous copy by thread `t` (`what`) reports to the mbarrier `key`:
    /// it was initialised, and the initialisation published to the async proxy (its own
    /// CTA's, with `fence.mbarrier_init` ordered before the issue) or to `t`'s CTA (a
    /// peer's, through a cluster barrier), and its CTA has not exited.
    pub(super) fn check_copy_mbarrier(&self, key: MbarrierKey, t: usize, what: &str) -> R {
        let place = smem_place(self.ir.buffer(BufferId(key.1)), key.2);
        if key.0 != self.block_rank(t) {
            self.check_remote_mbarrier(key, t, what)?;
        }
        let Some(m) = self.mbarriers.get(&key) else {
            return Err(self.err(
                format!(
                    "{what} reports to the mbarrier {place}{}, which was never initialised",
                    if key.0 == self.block_rank(t) {
                        String::new()
                    } else {
                        format!(" of CTA {}", key.0 % self.cluster_size)
                    }
                ),
                Some(t),
            ));
        };
        if key.0 != self.block_rank(t) {
            return Ok(());
        }
        let k = self.clock_index(t);
        let (slot, _, loc) = m.init;
        let known = self.clocks[k][slot].max(self.init_clocks.get(k).map_or(0, |c| c[slot]));
        if !m.fenced.is_some_and(|f| known >= f) {
            return Err(self.err(
                format!(
                    "{what} reports to the mbarrier {place} before its initialisation ({}:{}) was published to the \
                     async proxy: call mbarrier_init_fence() (fence.mbarrier_init) after mbarrier_init, ordered before \
                     the copy (by a barrier)",
                    loc.file(),
                    loc.line()
                ),
                Some(t),
            ));
        }
        Ok(())
    }

    /// Issue an asynchronous copy by thread `t` that writes `runs` (`(byte, data)`) of
    /// shared buffer `dst` in `block` and reports to the mbarrier `key` of that block:
    /// the async proxy's write is checked and claimed, and the copy is in flight until
    /// the mbarrier's phase lands it. `reads` are shared memory of `t`'s CTA it reads
    /// until then.
    pub(super) fn issue_to_mbarrier(
        &mut self,
        dst: BufferId,
        block: usize,
        runs: Vec<(usize, Vec<u8>)>,
        reads: Vec<AsyncRead>,
        key: MbarrierKey,
        t: usize,
        what: &str,
        async_proxy: bool,
    ) -> R {
        for (to, data) in &runs {
            let range = *to..*to + data.len();
            if async_proxy && let Err(why) = self.async_proxy_may_read(dst, block, range.clone(), self.clock_index(t)) {
                return Err(self.err(
                    format!(
                        "{what} overwrites shared memory through the async proxy, but {why}: write it, then \
                         fence_view_async_shared() (fence.proxy.async.shared::cta) in the writing threads, ordered \
                         before the copy"
                    ),
                    Some(t),
                ));
            }
            if !self.pending_reads_free() {
                self.check_async_reads_in(dst, block, range.clone(), t)?;
            }
            self.smem_claim_write_in(dst, block, *to, data.len(), t, true, false)?;
        }
        if block != self.block_rank(t) {
            self.report.cluster_copies += 1;
        }
        let id = self.ops.len();
        self.any_async_reads |= !reads.is_empty();
        self.ops.push(AsyncOp {
            thread: t,
            agent: self.agent_slot(block),
            writes: runs.into_iter().map(|(dst_byte, data)| AsyncWrite { dst, block, dst_byte, data }).collect(),
            reads,
            global_writes: Vec::new(),
            global_reduce: None,
            results: Vec::new(),
            trigger: Trigger::CompleteTx(key),
            done: None,
        });
        self.mbarriers.get_mut(&key).expect("checked by check_copy_mbarrier").copies.push(id);
        if self.opts.async_completion == super::AsyncCompletion::Early {
            self.complete_bulk_copy(id, key);
        }
        Ok(())
    }

    /// A TMA load by thread `t` (`cp.async.bulk.tensor ... .mbarrier::complete_tx`), in any
    /// access mode (`.tile`, `.im2col`, `.tile::gather4`): the elements the access names ([`TmaDesc::access_elements`],
    /// out-of-bounds ones filled with zeros, or NaNs with `nan_request_zero_fma`) land in
    /// shared memory from the destination's address in the descriptor's encoding (packed or
    /// padded for the sub-byte ones, whose padding is left unwritten) with its
    /// address-based swizzle, and their global bytes count down the mbarrier's transaction
    /// count. It lands as the bulk copies do: when a waiter sees every arrival of the
    /// phase (or at random), and accesses to its bytes before then are reported. With
    /// `mcast` (`.multicast::cluster`), one copy lands at the same offset in each CTA of the
    /// mask, counting down the mbarrier at the same offset in each. With `two_cta`
    /// (`.cta_group::2`) each destination's bytes count down
    /// the mbarrier at that offset in the even CTA of the destination's pair instead, so
    /// one CTA of a pair waits for both CTAs' halves of a 2-CTA MMA's operands.
    ///
    /// [`TmaDesc::access_elements`]: crate::tma::TmaDesc::access_elements
    pub(super) fn tma_load(
        &mut self,
        dst: BufferId,
        dst_index: ValueId,
        elem: DType,
        access: &TmaIssue<'_>,
        (mbar, mbar_index, two_cta): (BufferId, ValueId, bool),
        mcast: Option<ValueId>,
        t: usize,
    ) -> R {
        let what = if mcast.is_some() {
            "cp.async.bulk.tensor (a multicast TMA load)"
        } else {
            "cp.async.bulk.tensor (a TMA load)"
        };
        let (d, start, offsets) = self.tma_issue_desc(access, t, what)?;
        let rank = d.rank as usize;
        let bits = d.data_type.bit_width() as usize;
        // The destination's element: the encoding's, or a byte per value for the padded
        // sub-byte encodings (their 16-byte slots hold 16 values).
        let fits = if d.data_type.is_packed() { elem.bits() == 8 } else { bits == elem.bits() };
        if !fits {
            return Err(self.err(
                format!(
                    "{what}: the tensor map holds {} elements, but the destination is {}",
                    d.data_type.name(),
                    elem.cute_name()
                ),
                Some(t),
            ));
        }
        let elements = d
            .access_elements(&tma_access(access.mode, &start, &offsets))
            .map_err(|e| self.err(format!("{what}: {e}"), Some(t)))?;
        // The destination: 128-byte aligned, and at the start of the swizzle's pattern.
        let idx = self.index(dst_index, t);
        let di = if elem.is_sub_byte() {
            // A packed sub-byte destination: its index counts values (the box starts at a byte).
            self.checked_range(dst, idx * elem.bits() as i64 / 8, 0, 1, 1, t, "TMA load destination")?
        } else {
            self.checked_range(dst, idx, 0, elem.bytes(), 1, t, "TMA load destination")?
        };
        let base = self.address_of(dst, 0);
        let abs = base + di as u64;
        let name = self.ir.buffer(dst).name.clone();
        if !abs.is_multiple_of(128) {
            return Err(self.err(
                format!(
                    "{what}: the destination {} is at shared address {abs:#x}, which is not 128-byte aligned, as a \
                     TMA destination must be (allocate it with make_smem_tensor_aligned or alloc_smem with 128 or \
                     more, and start each box at a multiple of 128 bytes)",
                    smem_place(self.ir.buffer(dst), di)
                ),
                Some(t),
            ));
        }
        self.check_swizzle_base(&d, dst, di, base, t, what)?;
        // The mbarriers: initialised, and their initialisation published.
        let mbyte = self.checked_range(mbar, self.index(mbar_index, t), 1, 8, 8, t, "TMA load mbarrier")?;
        let targets = self.cluster_targets(mcast, t, what)?;
        if two_cta && !self.cluster_size.is_multiple_of(2) {
            return Err(self.err(
                format!(
                    "{what} with .cta_group::2 in a cluster of {} CTA(s): it signals the mbarrier of its CTA pair's \
                     even CTA, and a CTA pair is two CTAs of a cluster with an even number of CTAs",
                    self.cluster_size
                ),
                Some(t),
            ));
        }
        // The mbarrier of each destination's pair leader.
        let signalled = |block: usize| if two_cta { block & !1 } else { block };
        for &block in &targets {
            self.check_copy_mbarrier((signalled(block), mbar.0, mbyte), t, what)?;
        }
        // The image, read through the descriptor: which bytes it writes (the padded
        // encodings skip their padding).
        let box_bytes = d.access_smem_bytes(access.mode.is_rows4()) as usize;
        let mut image = vec![0u8; box_bytes];
        let mut written = vec![false; box_bytes];
        let nan_fill = d.oob_fill == crate::tma::TensorMapFloatOOBFill::nan_request_zero_fma && d.data_type.is_float();
        let fill: u64 = if nan_fill { (1u64 << (bits - 1)) - 1 } else { 0 };
        let mut global_reads = Vec::new();
        for e in &elements {
            let v = match e.global_bit {
                Some(gbit) => {
                    let (span, shift) = (((gbit % 8) as usize + bits).div_ceil(8), gbit % 8);
                    let (gb, gbyte) = self.tma_element(gbit / 8, &e.coords[..rank], span, t, what)?;
                    global_reads.push((t, gbyte, span));
                    (self.read(gb, gbyte, span, t)? >> shift) & mask_bits(bits)
                }
                None => fill,
            };
            put_bits(&mut image, &mut written, e.smem_bit as usize, bits, v);
        }
        // Where its 16-byte chunks land: the swizzle permutes chunks within each 128-byte
        // row of shared addresses. Consecutive written bytes are merged.
        let buffer_bytes = self.buffer_bytes(dst);
        let mut runs: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut chunks: Vec<(usize, usize)> =
            (0..box_bytes / 16).map(|c| ((d.smem_address(abs, 16 * c as u64) - base) as usize, 16 * c)).collect();
        chunks.sort_unstable();
        for (to, from) in chunks {
            if to + 16 > buffer_bytes {
                return Err(self.err(
                    format!(
                        "{what}: the box of {box_bytes} bytes from shared address {abs:#x} runs past the end of \
                         `{name}` ({buffer_bytes} bytes)"
                    ),
                    Some(t),
                ));
            }
            for k in 0..16 {
                if !written[from + k] {
                    continue;
                }
                match runs.last_mut() {
                    Some((s, data)) if *s + data.len() == to + k => data.push(image[from + k]),
                    _ => runs.push((to + k, vec![image[from + k]])),
                }
            }
        }
        for &block in &targets {
            let key = (signalled(block), mbar.0, mbyte);
            // The same box issued twice (by two warps' elected lanes, say) before the first lands.
            let twice = runs.iter().any(|(to, _)| {
                self.ops.iter().any(|op| {
                    op.in_flight()
                        && matches!(op.trigger, Trigger::CompleteTx(k) if k == key)
                        && op.writes_byte(dst, block, *to)
                })
            });
            if twice {
                return Err(self.err(
                    format!(
                        "{what}: another TMA load reporting to the same mbarrier {} is still writing these bytes of \
                         `{name}`{}: the copy was issued twice (each warp that executes a TMA copy issues it from one \
                         elected lane: execute it in one warp, without electing a thread yourself)",
                        smem_place(self.ir.buffer(mbar), mbyte),
                        if block == self.block_rank(t) {
                            String::new()
                        } else {
                            format!(" of CTA {}", block % self.cluster_size)
                        }
                    ),
                    Some(t),
                ));
            }
            self.issue_to_mbarrier(dst, block, runs.clone(), Vec::new(), key, t, what, true)?;
            self.report.tma_bytes += d.access_tx_bytes(access.mode.is_rows4());
        }
        // One instruction, however many CTAs a multicast writes.
        self.report.tma_instructions += 1;
        self.account(MemSpace::Global, &global_reads, false);
        Ok(())
    }

    /// The global buffer and byte of the box element at `addr` (`coords` in the tensor) of
    /// a TMA copy, checked against its buffer's end.
    fn tma_element(&self, addr: u64, coords: &[i64], size: usize, t: usize, what: &str) -> R<(BufferId, usize)> {
        let (gb, gbyte) = self
            .resolve(AddressSpace::Gmem, addr)
            .map_err(|why| self.err(format!("{what}: element {coords:?} of the tensor: {why}"), Some(t)))?;
        if gbyte + size > self.buffer_bytes(gb) {
            return Err(self.err(
                format!(
                    "{what}: element {coords:?} of the tensor is at {addr:#x}, past the end of `{}` (the tensor map's \
                     extents or strides overrun its buffer)",
                    self.ir.buffer(gb).name
                ),
                Some(t),
            ));
        }
        Ok((gb, gbyte))
    }

    /// TMA swizzles by shared address (the PTX ISA's swizzle base offset, from the box's
    /// address bits), so a box may start anywhere 128-byte aligned; the SMEM layout's
    /// swizzle is on offsets from the buffer's start, so the two agree when the buffer
    /// starts at a multiple of the pattern.
    fn check_swizzle_base(
        &self,
        d: &crate::tma::TmaDesc,
        b: BufferId,
        at: usize,
        base: u64,
        t: usize,
        what: &str,
    ) -> R {
        let pattern = d.swizzle.pattern_bytes();
        if d.swizzle != crate::tma::TensorMapSwizzle::none && !base.is_multiple_of(pattern) {
            return Err(self.err(
                format!(
                    "{what}: the box {} is in `{}`, which starts at shared address {base:#x}, not a multiple of the {} \
                     swizzle's {pattern}-byte pattern: TMA swizzles by shared address, so the box would not be where \
                     the SMEM layout's swizzle (on offsets from the buffer's start) puts it (align the buffer to \
                     {pattern} bytes: make_smem_tensor_aligned)",
                    smem_place(self.ir.buffer(b), at),
                    self.ir.buffer(b).name,
                    d.swizzle.name()
                ),
                Some(t),
            ));
        }
        Ok(())
    }

    /// A TMA store by thread `t` (`cp.async.bulk.tensor ... .global.shared::cta ... .bulk_group`)
    /// in any access mode (`.tile`, `.im2col_no_offs`, `.tile::scatter4`), or with `reduce` a reduce store
    /// (`cp.reduce.async.bulk.tensor`): the elements the access names
    /// ([`TmaDesc::access_elements`]) are read from the source through the descriptor's
    /// address-based swizzle (the inverse of a load's placement) and written to the tensor,
    /// those outside it clipped. The async proxy reads the shared memory in the background
    /// (it must have been fenced after generic writes, and must not be written until the
    /// covering `cp.async.bulk.wait_group.read`), and the global writes (or, for a
    /// reduction, element-wise read-modify-writes) land at the covering plain
    /// `cp.async.bulk.wait_group` or the CTA's exit, ordered as the bulk copies to global
    /// memory are.
    ///
    /// [`TmaDesc::access_elements`]: crate::tma::TmaDesc::access_elements
    pub(super) fn tma_store(
        &mut self,
        src: BufferId,
        src_index: ValueId,
        elem: DType,
        access: &TmaIssue<'_>,
        reduce: Option<crate::cpasync::ReductionKind>,
        t: usize,
    ) -> R {
        let what = if reduce.is_some() {
            "cp.reduce.async.bulk.tensor (a TMA reduce store)"
        } else {
            "cp.async.bulk.tensor (a TMA store)"
        };
        let (d, start, offsets) = self.tma_issue_desc(access, t, what)?;
        let rank = d.rank as usize;
        let bits = d.data_type.bit_width() as usize;
        if d.data_type.is_padded() {
            return Err(self.err(
                format!(
                    "{what}: the padded {} encoding is for loads (PTX's .b4x16_p64 does not store, and the FP6 store \
                     form .b6p2x16 is not modelled)",
                    d.data_type.name()
                ),
                Some(t),
            ));
        }
        let fits = if d.data_type.is_packed() { elem.bits() == 8 } else { bits == elem.bits() };
        if !fits || (reduce.is_some() && d.data_type.is_packed()) {
            return Err(self.err(
                format!(
                    "{what}: the tensor map holds {} elements, but the source is {}",
                    d.data_type.name(),
                    elem.cute_name()
                ),
                Some(t),
            ));
        }
        // The .global.shared::cta direction: no negative coordinates, and an im2col bounding
        // box inside the tensor (the PTX ISA's restrictions on tensor copies). The PTX ISA
        // lets a scatter4 store start at a negative coordinate on sm_107f only; a B200 stops
        // with an illegal-instruction error on one, so it is reported for every target.
        if start.iter().any(|&c| c < 0) {
            return Err(self
                .err(format!("{what}: the coordinates {start:?} are negative, which a store does not take"), Some(t)));
        }
        if let Some(p) = &d.im2col {
            let n = rank - 2;
            if p.lower_corner[..n].iter().any(|&c| c < 0) || p.upper_corner[..n].iter().any(|&c| c > 0) {
                return Err(self.err(
                    format!(
                        "{what}: an im2col store's bounding box must stay inside the tensor (lower corners \
                         non-negative, upper ones non-positive), got {:?} and {:?}",
                        &p.lower_corner[..n],
                        &p.upper_corner[..n]
                    ),
                    Some(t),
                ));
            }
        }
        let elements = d
            .access_elements(&tma_access(access.mode, &start, &offsets))
            .map_err(|e| self.err(format!("{what}: {e}"), Some(t)))?;
        let idx = self.index(src_index, t);
        let si = if elem.is_sub_byte() {
            self.checked_range(src, idx * elem.bits() as i64 / 8, 0, 1, 1, t, "TMA store source")?
        } else {
            self.checked_range(src, idx, 0, elem.bytes(), 1, t, "TMA store source")?
        };
        let base = self.address_of(src, 0);
        let abs = base + si as u64;
        let name = self.ir.buffer(src).name.clone();
        if !abs.is_multiple_of(128) {
            return Err(self.err(
                format!(
                    "{what}: the source {} is at shared address {abs:#x}, which is not 128-byte aligned, as a TMA \
                     store's shared memory must be",
                    smem_place(self.ir.buffer(src), si)
                ),
                Some(t),
            ));
        }
        self.check_swizzle_base(&d, src, si, base, t, what)?;
        let block = self.block_rank(t);
        let box_bytes = d.access_smem_bytes(access.mode.is_rows4()) as usize;
        let buffer_bytes = self.buffer_bytes(src);
        // The 16-byte chunks of the box, read where the swizzle put them.
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
        let mut image = vec![0u8; box_bytes];
        for c in 0..box_bytes / 16 {
            let at = (d.smem_address(abs, 16 * c as u64) - base) as usize;
            if at + 16 > buffer_bytes {
                return Err(self.err(
                    format!(
                        "{what}: the box of {box_bytes} bytes from shared address {abs:#x} runs past the end of \
                         `{name}` ({buffer_bytes} bytes)"
                    ),
                    Some(t),
                ));
            }
            image[16 * c..16 * c + 16]
                .copy_from_slice(&self.shared[self.home[src.0 as usize]][block].data[at..at + 16]);
            match ranges.last_mut() {
                Some(r) if r.end == at => r.end = at + 16,
                _ => ranges.push(at..at + 16),
            }
        }
        ranges.sort_by_key(|r| r.start);
        let k = self.clock_index(t);
        for r in &ranges {
            self.bulk_store_may_read(src, block, r.clone(), t)?;
            if let Err(why) = self.async_proxy_may_read(src, block, r.clone(), k) {
                return Err(self.err(
                    format!(
                        "{what} reads shared memory through the async proxy, but {why}: write it, then \
                         fence_view_async_shared() (fence.proxy.async.shared::cta) in the writing threads, ordered \
                         before the store (a barrier)"
                    ),
                    Some(t),
                ));
            }
        }
        // The elements inside the tensor, merged into runs of global bytes (sub-byte ones
        // are gathered into whole bytes, which must be wholly inside or outside).
        let mut global: Vec<(BufferId, usize, Vec<u8>)> = Vec::new();
        let mut accesses = Vec::new();
        let mut partial: std::collections::BTreeMap<(BufferId, usize), (u8, u8)> = Default::default();
        for e in &elements {
            let Some(gbit) = e.global_bit else { continue };
            let size = bits.div_ceil(8);
            let (gb, gbyte) = self.tma_element(gbit / 8, &e.coords[..rank], size, t, what)?;
            if let Some(why) = self.read_only(gb) {
                return Err(self.err(format!("{what} to `{}`, {why}", self.ir.buffer(gb).name), Some(t)));
            }
            let v = get_bits(&image, e.smem_bit as usize, bits);
            if bits < 8 {
                let shift = (gbit % 8) as u32;
                let entry = partial.entry((gb, gbyte)).or_insert((0, 0));
                entry.0 |= ((v as u8) & ((1u8 << bits) - 1)) << shift;
                entry.1 |= (((1u16 << bits) - 1) as u8) << shift;
                continue;
            }
            accesses.push((t, gbyte, size));
            let data = &v.to_le_bytes()[..size];
            match global.last_mut() {
                Some((b, s, run)) if *b == gb && *s + run.len() == gbyte => run.extend_from_slice(data),
                _ => global.push((gb, gbyte, data.to_vec())),
            }
        }
        for ((gb, gbyte), (v, mask)) in partial {
            if mask != 0xff {
                return Err(self.err(
                    format!(
                        "{what}: byte {gbyte} of `{}` is only partly inside the tensor (a sub-byte tensor whose inner \
                         extent ends inside a byte)",
                        self.ir.buffer(gb).name
                    ),
                    Some(t),
                ));
            }
            accesses.push((t, gbyte, 1));
            match global.last_mut() {
                Some((b, s, run)) if *b == gb && *s + run.len() == gbyte => run.push(v),
                _ => global.push((gb, gbyte, vec![v])),
            }
        }
        for (gb, gbyte, data) in &global {
            let range = *gbyte..gbyte + data.len();
            if self.pending_global > 0 {
                self.check_pending_global_for(*gb, range.clone(), t, "overwrites", reduce.is_some())?;
            }
            if self.pdl.is_some() {
                self.pdl_access(*gb, range.clone(), t, true)?;
            }
            self.nc_write(*gb, *gbyte, data.len(), t)?;
        }
        for r in &ranges {
            self.note_bulk_source_read(t, src, block, r.clone());
        }
        let id = self.ops.len();
        self.ops.push(AsyncOp {
            thread: t,
            agent: self.agent_slot(block),
            writes: Vec::new(),
            reads: ranges.into_iter().map(|bytes| AsyncRead { buffer: src, block, bytes }).collect(),
            global_writes: global,
            global_reduce: reduce.map(|k| (k, elem)),
            results: Vec::new(),
            trigger: Trigger::BulkGroup,
            done: None,
        });
        self.any_async_reads = true;
        self.pending_global += 1;
        self.bulk_uncommitted[t].push(id);
        if self.tmaps.enabled {
            self.tmap_note_store(id, access.desc, t);
        }
        self.report.tma_stores += 1;
        self.report.tma_store_bytes += box_bytes as u64;
        self.account(MemSpace::Global, &accesses, true);
        if self.opts.async_completion == super::AsyncCompletion::Early {
            self.land_bulk_store(id);
        }
        Ok(())
    }

    /// The bytes bulk copy or TMA load `id` writes (its transaction bytes).
    fn async_bytes(&self, id: usize) -> i64 {
        self.ops[id].writes.iter().map(|w| w.data.len() as i64).sum()
    }

    /// Land the bulk copies reporting to one mbarrier, as late as the
    /// simulator can: when thread `t` waits on it after every arrival of
    /// the phase is in. (Until then the phase cannot complete wherever they
    /// land, and any access to their destination is reported.) Their bytes
    /// count down the phase's tx-count, which must hold them all: otherwise
    /// the phase would complete with some still in flight.
    pub(super) fn land_bulk_copies(&mut self, key: (usize, u32, usize), t: usize) -> R {
        let Some(state) = self.mbarriers.get(&key) else {
            return Ok(());
        };
        if state.arrivals_left > 0 || state.copies.is_empty() {
            return Ok(());
        }
        let landed: i64 = state.copies.iter().map(|&id| self.async_bytes(id)).sum();
        if landed > state.tx_count {
            let (expected, n) = (state.tx_count, state.copies.len());
            return Err(self.err(
                format!(
                    "{n} bulk cop{} of {landed} bytes in all report to an mbarrier phase whose transaction count is \
                     {expected}: the phase completes once {expected} bytes have landed, with the rest still in flight \
                     (mbarrier_expect_tx too small?)",
                    if n == 1 { "y" } else { "ies" }
                ),
                Some(t),
            ));
        }
        for id in state.copies.clone() {
            self.complete_bulk_copy(id, key);
        }
        Ok(())
    }

    /// Complete bulk copy `id`, which reports to the mbarrier keyed by
    /// `key`: the block's copy agent writes its bytes at a new time of its
    /// own, which the phase's completion publishes to its waiters, and its
    /// bytes count down the phase's tx-count (which may go negative until
    /// `expect_tx` raises it, as PTX allows).
    pub(super) fn complete_bulk_copy(&mut self, id: usize, key: MbarrierKey) {
        let time = self.complete_as_agent(id, false);
        let agent = self.ops[id].agent;
        // A copy from shared memory (to a peer's) has read its source by now, as the
        // destination's agent: a later overwrite must be ordered after this completion,
        // which the destination's mbarrier phase publishes to its waiters.
        let reads = std::mem::take(&mut self.ops[id].reads);
        if self.opts.detect_races {
            let none = vec![0; self.slots()];
            let read = ReadEpoch { slot: agent as u16, unit: 0, clock: time };
            for r in reads {
                let sh = &mut self.shared[self.home[r.buffer.0 as usize]][r.block];
                for i in r.bytes {
                    sh.reads.record(i, read, &none, false);
                }
            }
        }
        let bytes = self.async_bytes(id);
        let state = self.mbarriers.get_mut(&key).expect("the mbarrier exists");
        state.copies.retain(|&c| c != id);
        state.tx_count -= bytes;
        state.release[agent] = state.release[agent].max(time);
        state.complete_if_done();
    }

    /// Complete operation `id` as a write of its agent at a new time of the
    /// agent's, and return that time: its bytes land, its in-flight markers
    /// clear and its register results are written. `tracked` marks the
    /// bytes as a `cp.async` that an mbarrier tracked (for reports).
    pub(super) fn complete_as_agent(&mut self, id: usize, tracked: bool) -> u32 {
        let block = self.ops[id].writes.first().map_or(self.block_rank(self.ops[id].thread), |w| w.block);
        self.bulk_time[block] += 1;
        let time = self.bulk_time[block];
        self.complete(id, self.ops[id].agent, time);
        if tracked {
            for w in &self.ops[id].writes {
                for m in
                    &mut self.shared[self.home[w.dst.0 as usize]][w.block].meta[w.dst_byte..w.dst_byte + w.data.len()]
                {
                    m.tracked_cp_async = true;
                }
            }
        }
        self.ops[id].done = Some(time);
        time
    }

    /// Complete operation `id` as the write of clock slot `writer` at its
    /// own time `time`: land its writes, clear its in-flight markers (and
    /// its thread's list of copies in flight), write its register results.
    pub(super) fn complete(&mut self, id: usize, writer: usize, time: u32) {
        let writes = std::mem::take(&mut self.ops[id].writes);
        for w in &writes {
            let range = w.dst_byte..w.dst_byte + w.data.len();
            self.shared[self.home[w.dst.0 as usize]][w.block].data[range.clone()].copy_from_slice(&w.data);
            self.smem_land(w.dst, w.block, range, writer, time);
        }
        self.ops[id].writes = writes;
        for (v, t, bits) in std::mem::take(&mut self.ops[id].results) {
            // A loop carry may have moved on to another operation's result.
            if self.pending_results.get(&(v, t)) == Some(&id) {
                self.set(v, t, bits);
                self.pending_results.remove(&(v, t));
            }
        }
        let thread = self.ops[id].thread;
        self.in_flight[thread].retain(|&c| c != id);
        self.ops[id].done = Some(0);
        self.progress();
    }

    /// Whether byte `byte` of shared buffer `b` of `block` is the destination
    /// of a bulk copy still in flight (rather than of a `cp.async`).
    pub(super) fn bulk_copy_writing(&self, b: BufferId, block: usize, byte: usize) -> bool {
        self.ops
            .iter()
            .any(|op| op.in_flight() && matches!(op.trigger, Trigger::CompleteTx(_)) && op.writes_byte(b, block, byte))
    }

    /// Whether byte `byte` of shared buffer `b` of `block` is the
    /// destination of a `cp.async` in flight that a `cp.async.mbarrier.arrive`
    /// tracks.
    pub(super) fn tracked_copy_writing(&self, b: BufferId, block: usize, byte: usize) -> bool {
        self.ops.iter().any(|op| {
            op.in_flight() && op.trigger == Trigger::CpAsync { tracked: true } && op.writes_byte(b, block, byte)
        })
    }

    /// Land `cp.async` copy `id` of thread `t`, if it is still in flight, for
    /// `t`'s covering `cp_async_wait_group`, after which `t`'s warp sees it,
    /// as do the warps that synchronize with it later. A copy lands once. A
    /// tracked copy stays the agent's write, for the waiters of the mbarrier
    /// phase that tracks it, with `t`'s warp as a second writer: the wait
    /// orders it, and no other write of the agent, before the warp.
    pub(super) fn land_cp_async_for_wait(&mut self, id: usize, t: usize) {
        let waiter = self.slot_of(self.clock_index(t));
        match (self.ops[id].done, self.ops[id].trigger) {
            (None, Trigger::CpAsync { tracked: false }) => {
                let k = self.clock_index(self.ops[id].thread);
                let slot = self.slot_of(k);
                let clock = self.clocks[k][slot];
                self.complete(id, slot, clock);
                return;
            }
            (None, _) => {
                self.land_tracked_copy(id);
            }
            // Completed as its warp's own write: nothing more to order.
            (Some(0), _) => return,
            // Landed by the mbarrier that tracks it: the wait still makes it
            // visible to the waiting thread.
            (Some(_), _) => {}
        }
        let time = self.ops[id].done.expect("completed above");
        let now = self.clocks[self.clock_index(t)][waiter];
        let agent = self.ops[id].agent;
        // The bytes that still hold what the copy wrote.
        for w in &self.ops[id].writes {
            for m in &mut self.shared[self.home[w.dst.0 as usize]][w.block].meta[w.dst_byte..w.dst_byte + w.data.len()]
            {
                if m.w_slot as usize == agent && m.w_clock == time {
                    m.alt_slot = waiter as u16;
                    m.alt_clock = now;
                }
            }
        }
    }

    /// Land tracked `cp.async` copy `id`, in flight, as a write of its
    /// block's asynchronous agent at a new time of the agent's, and return
    /// that time.
    pub(super) fn land_tracked_copy(&mut self, id: usize) -> u32 {
        if self.ops[id].trigger == Trigger::Tcgen05Mma {
            return self.umma_land(id);
        }
        let tracked = self.ops[id].trigger == Trigger::CpAsync { tracked: true };
        self.complete_as_agent(id, tracked)
    }

    /// Make the arrivals that `cp.async.mbarrier.arrive` owes one mbarrier,
    /// as late as the simulator can: when thread `t` waits on it and they are
    /// all its phase still misses. (Until then the phase cannot complete,
    /// and any access to the copies' destinations is reported.) Their copies
    /// that no `cp_async_wait_group` has landed land first, as writes of the
    /// block's asynchronous agent, and the phase publishes the agent's time
    /// to its waiters, so the data is ordered only before warps that waited
    /// for the phase.
    pub(super) fn fire_deferred_arrivals(&mut self, key: (usize, u32, usize)) {
        let Some(state) = self.mbarriers.get_mut(&key) else { return };
        let owed = state.deferred.len() as u32;
        if owed == 0 || state.arrivals_left > owed {
            return;
        }
        let deferred = std::mem::take(&mut state.deferred);
        let mut published = deferred.iter().map(|d| d.floor).max().unwrap_or(0);
        for id in deferred.into_iter().flat_map(|d| d.copies) {
            let time = match self.ops[id].done {
                Some(time) => time,
                None => self.land_tracked_copy(id),
            };
            published = published.max(time);
        }
        let agent = self.agent_slot(key.0);
        let state = self.mbarriers.get_mut(&key).expect("the mbarrier exists");
        state.release[agent] = state.release[agent].max(published);
        state.arrivals_left -= owed;
        // Bulk copies reporting to the phase land next and complete it.
        if state.copies.is_empty() {
            state.complete_if_done();
        }
    }

    /// Make the arrive-ons owed to the mbarrier keyed by `key` whose copies
    /// have all completed (with [`super::AsyncCompletion::Early`], at once):
    /// each publishes the agent's latest time to the phase.
    pub(super) fn fire_ready_arrivals(&mut self, key: MbarrierKey) {
        let agent = self.agent_slot(key.0);
        let now = self.bulk_time[key.0];
        let Some(state) = self.mbarriers.get_mut(&key) else { return };
        let ops = &self.ops;
        let before = state.deferred.len();
        state.deferred.retain(|d| d.copies.iter().any(|&id| ops[id].in_flight()));
        let fired = (before - state.deferred.len()) as u32;
        if fired > 0 {
            state.release[agent] = state.release[agent].max(now);
            state.arrivals_left -= fired;
            state.complete_if_done();
        }
    }

    /// Thread `t` uses (`what`) the mbarrier keyed by `key` in another CTA
    /// of its cluster: that CTA must not have exited, and the mbarrier's
    /// initialisation must have been published to `t`'s warp
    /// (`fence.mbarrier_init` by the initialising thread, then a
    /// cluster-scope edge to this warp).
    pub(super) fn check_remote_mbarrier(&self, key: MbarrierKey, t: usize, what: &str) -> R {
        let w = self.warps_per_block;
        let (rank, own) = (key.0 % self.cluster_size, self.block_rank(t) % self.cluster_size);
        let name = smem_place(self.ir.buffer(BufferId(key.1)), key.2);
        if (0..w).all(|warp| matches!(self.warps[key.0 * w + warp].status, Status::Exited)) {
            return Err(self.err(
                format!(
                    "{what} by CTA {own} on the mbarrier {name} of CTA {rank}, which has exited: its shared memory is \
                     gone (CUDA requires a cluster barrier before a CTA exits while its peers still use its shared \
                     memory)"
                ),
                Some(t),
            ));
        }
        let Some(m) = self.mbarriers.get(&key) else { return Ok(()) };
        let k = self.clock_index(t);
        let (slot, init_key, loc) = m.init;
        let known = self.clocks[k][slot].max(self.init_clocks[k][slot]);
        if m.fenced.is_some_and(|f| known >= f) {
            return Ok(());
        }
        let how = if m.fenced.is_none() {
            "the initialising thread has not executed mbarrier_init_fence() since"
        } else {
            "nothing orders its mbarrier_init_fence() before this use"
        };
        Err(self.err(
            format!(
                "{what} by CTA {own} on the mbarrier {name} of CTA {rank} before its initialisation was published to \
                 CTA {own}: warp {} of CTA {rank} initialised it ({}:{}), but {how} (publish it with \
                 mbarrier_init_fence() followed by a cluster barrier, cluster_arrive or cluster_arrive_relaxed then \
                 cluster_wait, or a release/acquire chain)",
                init_key % w,
                loc.file(),
                loc.line()
            ),
            Some(t),
        ))
    }

    /// The block whose mbarrier an operation by thread `t` addresses: its
    /// own, or CTA `peer` of its cluster.
    pub(super) fn mbarrier_block(&self, peer: Option<ValueId>, t: usize, what: &str) -> R<usize> {
        let Some(peer) = peer else { return Ok(self.block_rank(t)) };
        let rank = self.get(peer, t) as u32 as i32;
        if rank < 0 || rank as usize >= self.cluster_size {
            return Err(self.err(format!("{what} on CTA {rank} of a cluster of {} CTA(s)", self.cluster_size), Some(t)));
        }
        Ok(self.cluster_base(self.block_rank(t)) + rank as usize)
    }
}
