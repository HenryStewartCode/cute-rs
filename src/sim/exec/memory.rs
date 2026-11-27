// SPDX-License-Identifier: MIT OR Apache-2.0

//! The memory model: address resolution, reads, writes and atomics, shared-memory race tracking, and the vector-clock synchronisation.

use super::*;

impl<'a> Machine<'a> {
    /// The address of byte `byte` of buffer `b` in the buffer's own space
    /// ([`AddrMap`]): a global address, or a shared or local offset.
    pub(super) fn address_of(&self, b: BufferId, byte: usize) -> u64 {
        let k = b.0 as usize;
        match self.ir.buffer(b).space {
            MemSpace::Global | MemSpace::Constant => {
                self.addrs.global[k].expect("global buffers are bound").0 + byte as u64
            }
            MemSpace::Shared => self.smem.starts[k].expect("shared buffers are placed") as u64 + byte as u64,
            MemSpace::Local => self.addrs.local[k].expect("local buffers are placed") + byte as u64,
            MemSpace::Raw(_) => unreachable!("a raw pointer's integer is its address"),
        }
    }

    /// Why buffer `b` cannot be written on the device, if it cannot: a by-value parameter
    /// (the kernel's `.param` aggregate), or a module global in constant memory or declared
    /// read-only.
    pub(super) fn read_only(&self, b: BufferId) -> Option<&'static str> {
        if self.ir.args.iter().any(|p| matches!(p, crate::ir::Param::ByValue { buffer, .. } if *buffer == b)) {
            return Some("a by-value parameter, which is read-only (the kernel's parameter space)");
        }
        let g = self.ir.globals.iter().find(|g| g.buffer == b)?;
        if self.ir.buffer(b).space == MemSpace::Constant {
            Some("a global in constant memory, which the device cannot write")
        } else if g.readonly {
            Some("a read-only module global")
        } else {
            None
        }
    }

    /// The buffer and byte the address `addr` of `space` reaches for thread
    /// `t`, or why it reaches none.
    pub(super) fn resolve(&self, space: AddressSpace, addr: u64) -> Result<(BufferId, usize), String> {
        let window = |base: u64| (base..base + WINDOW_BYTES).contains(&addr).then(|| addr - base);
        match space {
            AddressSpace::Generic => {
                if let Some(off) = window(SIM_SHARED_WINDOW) {
                    return self.resolve(AddressSpace::Smem, off);
                }
                if let Some(off) = window(SIM_LOCAL_WINDOW) {
                    return self.resolve(AddressSpace::Rmem, off);
                }
                self.resolve(AddressSpace::Gmem, addr)
            }
            AddressSpace::Gmem => {
                // The buffer the address is in, or in the first half of the
                // gap after (an overrun of it).
                let found = self
                    .addrs
                    .global
                    .iter()
                    .enumerate()
                    .filter_map(|(k, g)| g.map(|(base, bytes)| (k, base, bytes)))
                    .find(|&(_, base, bytes)| base <= addr && addr < base + bytes.max(1) + GLOBAL_GAP / 2);
                match found {
                    Some((k, base, _)) => Ok((BufferId(k as u32), (addr - base) as usize)),
                    None => Err(format!("{addr:#x} is a wild address: no global buffer of the launch is there")),
                }
            }
            AddressSpace::Smem => {
                for (k, start) in self.smem.starts.iter().enumerate() {
                    let Some(start) = *start else { continue };
                    let b = BufferId(k as u32);
                    let (start, bytes) = (start as u64, self.buffer_bytes(b) as u64);
                    if (start..start + bytes.max(1)).contains(&addr) {
                        return Ok((b, (addr - start) as usize));
                    }
                }
                Err(format!("{addr:#x} is a wild shared address: no shared buffer of the block is there"))
            }
            AddressSpace::Rmem => {
                for (k, start) in self.addrs.local.iter().enumerate() {
                    let Some(start) = *start else { continue };
                    let b = BufferId(k as u32);
                    let bytes = self.ir.buffer(b).len.unwrap_or(0) as u64 * self.ir.buffer(b).elem.bytes() as u64;
                    if (start..start + bytes.max(1)).contains(&addr) {
                        return Ok((b, (addr - start) as usize));
                    }
                }
                Err(format!("{addr:#x} is a wild local address: no per-thread buffer is there"))
            }
            other => Err(format!("{other} memory cannot be read or written by address")),
        }
    }

    /// The buffer and element index (of `size`-byte elements) an access to
    /// buffer `b` at `index` reaches for thread `t`: the buffer itself, or
    /// for a raw pointer the buffer its byte address falls in.
    pub(super) fn target(&self, b: BufferId, index: ValueId, size: usize, t: usize, what: &str) -> R<(BufferId, i64)> {
        let MemSpace::Raw(space) = self.ir.buffer(b).space else { return Ok((b, self.index(index, t))) };
        let addr = self.get(index, t);
        let (rb, byte) =
            self.resolve(space, addr).map_err(|m| self.err(format!("{what} through a pointer: {m}"), Some(t)))?;
        if !byte.is_multiple_of(size) {
            return Err(self.err(
                format!(
                    "misaligned {what}: the address {addr:#x} is byte {byte} of `{}`, not a multiple of the \
                     {size}-byte element",
                    self.ir.buffer(rb).name
                ),
                Some(t),
            ));
        }
        Ok((rb, (byte / size) as i64))
    }

    /// The integer address `x` of `from` as one of `to` (one of them
    /// generic), or why it has none.
    pub(super) fn cvt_addr(&self, x: u64, from: AddressSpace, to: AddressSpace) -> Result<u64, String> {
        use AddressSpace as A;
        let window = |base: u64| (base..base + WINDOW_BYTES).contains(&x).then(|| x - base);
        Ok(match (from, to) {
            (A::Gmem, A::Generic) => x,
            (A::Smem, A::Generic) => SIM_SHARED_WINDOW + (x & 0xffff_ffff),
            (A::Rmem, A::Generic) => SIM_LOCAL_WINDOW + (x & 0xffff_ffff),
            (A::Generic, A::Gmem) => {
                if window(SIM_SHARED_WINDOW).is_some() || window(SIM_LOCAL_WINDOW).is_some() {
                    let which = if window(SIM_SHARED_WINDOW).is_some() { "shared" } else { "local" };
                    return Err(format!(
                        "the generic address {x:#x} is in the {which} window, so it has no global address (cvta.to.global of it is undefined)"
                    ));
                }
                x
            }
            (A::Generic, A::Smem) => window(SIM_SHARED_WINDOW).ok_or_else(|| {
                format!("the generic address {x:#x} is not in the shared window (cvta.to.shared of it is undefined)")
            })?,
            (A::Generic, A::Rmem) => window(SIM_LOCAL_WINDOW).ok_or_else(|| {
                format!("the generic address {x:#x} is not in the local window (cvta.to.local of it is undefined)")
            })?,
            _ => return Err(format!("cannot convert a {from} address to {to}")),
        })
    }

    pub(super) fn buffer_bytes(&self, b: BufferId) -> usize {
        let d = self.ir.buffer(b);
        match d.space {
            MemSpace::Global | MemSpace::Constant => {
                self.globals[b.0 as usize].as_ref().expect("global buffers are bound").len()
            }
            // Only the dynamic shared-memory region is sized at launch.
            _ => d.len.map_or(self.cfg.dynamic_smem_bytes as usize, |n| n * d.elem.bytes()),
        }
    }

    /// Bounds- and alignment-check `n` elements of `size` bytes at element
    /// `idx`, and return the byte offset.
    pub(super) fn checked_range(
        &self,
        b: BufferId,
        idx: i64,
        n: usize,
        size: usize,
        align: usize,
        t: usize,
        what: &str,
    ) -> R<usize> {
        let d = self.ir.buffer(b);
        let bytes = self.buffer_bytes(b);
        if idx < 0 || (idx as usize).saturating_add(n).saturating_mul(size) > bytes {
            // The dynamic region's size is the launch's.
            let sized = if d.space == MemSpace::Shared && d.len.is_none() {
                format!(
                    ": the dynamic shared memory region (get_dyn_smem) has the {bytes} bytes the launch gives it \
                     (LaunchConfig::dynamic_smem_bytes)"
                )
            } else {
                String::new()
            };
            return Err(self.err(
                format!(
                    "{what} of {n} element(s) at index {idx} is out of bounds for `{}` ({} elements){sized}",
                    d.name,
                    bytes / size
                ),
                Some(t),
            ));
        }
        let byte = idx as usize * size;
        if align > 1 && !byte.is_multiple_of(align) {
            return Err(self.err(
                format!("misaligned {what}: element {idx} of `{}` is not {align}-byte aligned", d.name),
                Some(t),
            ));
        }
        Ok(byte)
    }

    pub(super) fn read(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R<u64> {
        self.read_as(b, byte, size, t, false)
    }

    /// A read by thread `t`, strong (ordered, volatile or MMIO) or weak.
    pub(super) fn read_as(&mut self, b: BufferId, byte: usize, size: usize, t: usize, strong: bool) -> R<u64> {
        match self.ir.buffer(b).space {
            MemSpace::Global | MemSpace::Constant => {
                if self.pending_global > 0 {
                    self.check_pending_global(b, byte..byte + size, t, "reads")?;
                }
                if self.pdl.is_some() {
                    self.pdl_access(b, byte..byte + size, t, false)?;
                }
                Ok(read_le(&self.globals[b.0 as usize].as_ref().expect("global buffers are bound")[byte..byte + size]))
            }
            MemSpace::Shared => self.smem_read(b, byte, size, t, strong),
            MemSpace::Local => self.local_read(b, byte, size, t),
            MemSpace::Raw(_) => unreachable!("raw accesses are resolved to a buffer first"),
        }
    }

    /// A write by thread `t`, strong (ordered, volatile, MMIO or atomic) or
    /// weak.
    pub(super) fn write_as(&mut self, b: BufferId, byte: usize, size: usize, t: usize, bits: u64, strong: bool) -> R {
        if let Some(what) = self.read_only(b) {
            return Err(self.err(format!("a store to `{}`, {what}", self.ir.buffer(b).name), Some(t)));
        }
        match self.ir.buffer(b).space {
            MemSpace::Global | MemSpace::Constant => {
                self.progress();
                if self.pending_global > 0 {
                    self.check_pending_global(b, byte..byte + size, t, "overwrites")?;
                }
                if self.pdl.is_some() {
                    self.pdl_access(b, byte..byte + size, t, true)?;
                }
                self.nc_write(b, byte, size, t)?;
                if self.tmaps.enabled {
                    self.tmap_note_write(b, byte, size, t)?;
                }
                write_le(
                    &mut self.globals[b.0 as usize].as_mut().expect("global buffers are bound")[byte..byte + size],
                    bits,
                );
                Ok(())
            }
            MemSpace::Shared => {
                self.progress();
                if !self.pending_reads_free() {
                    self.check_async_reads(b, byte..byte + size, t)?;
                }
                self.smem_claim_write(b, byte, size, t, false, strong)?;
                let block = self.smem_block(t);
                write_le(&mut self.shared[self.home[b.0 as usize]][block].data[byte..byte + size], bits);
                Ok(())
            }
            MemSpace::Local => {
                let mem = self.locals[b.0 as usize].as_mut().expect("local buffers are allocated");
                let base = t * mem.per_thread + byte;
                write_le(&mut mem.data[base..base + size], bits);
                mem.written[base..base + size].fill(true);
                Ok(())
            }
            MemSpace::Raw(_) => unreachable!("raw accesses are resolved to a buffer first"),
        }
    }

    fn local_read(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R<u64> {
        let d = self.ir.buffer(b);
        let mem = self.locals[b.0 as usize].as_ref().expect("local buffers are allocated");
        let base = t * mem.per_thread + byte;
        if self.opts.detect_uninitialized && !mem.written[base..base + size].iter().all(|&w| w) {
            return Err(
                self.err(format!("reads {}[{}], which was never written", d.name, byte / d.elem.bytes()), Some(t))
            );
        }
        Ok(read_le(&mem.data[base..base + size]))
    }

    /// A read of shared memory by thread `t`, strong or weak, of one element
    /// of `size` bytes. A read races with the last write if another warp
    /// made it and nothing orders it before the reader, unless both are
    /// strong and cover the same bytes (morally strong), or the write is
    /// strong and the reading warp has read it with such a strong load
    /// (which orders the warp after the write).
    fn smem_read(&mut self, b: BufferId, byte: usize, size: usize, t: usize, strong: bool) -> R<u64> {
        let block = self.smem_block(t);
        let k = self.clock_index(t);
        let slot = self.slot_of(k);
        let opts = self.opts;
        let d = self.ir.buffer(b);
        let clock = &self.clocks[k];
        let sh = &mut self.shared[self.home[b.0 as usize]][block];
        if strong && opts.detect_races && sh.strong_reads.is_empty() {
            sh.strong_reads = Reads::new(sh.meta.len());
        }
        let mut failure = None;
        let mut inflight = None;
        for i in byte..byte + size {
            let m = &sh.meta[i];
            let unit = if strong { unit_code(i - byte, size) } else { 0 };
            let record = if sh.strong_reads.is_empty() { None } else { sh.strong_reads.by(i, slot).map(|r| r.unit) };
            let observed = record.is_some_and(|r| r & OBSERVED != 0);
            let morally_strong = strong && m.strong && m.unit == unit;
            if m.inflight > 0 {
                inflight = Some(i);
                break;
            } else if opts.detect_uninitialized && !m.written {
                failure = Some(Ok(format!("reads {}, which was never written", smem_place(d, i))));
            } else if opts.detect_races && !m.write_ordered_before(slot, clock) && !morally_strong && !observed {
                failure = Some(Err((i, *m, unit)));
            }
            if failure.is_some() {
                break;
            }
            if opts.detect_races {
                let now = clock[slot];
                if strong {
                    let kept = record.map_or(unit, |r| if r & !OBSERVED == unit { unit } else { MIXED_UNITS });
                    let seen = if morally_strong || observed { OBSERVED } else { 0 };
                    let read = ReadEpoch { slot: slot as u16, unit: kept | seen, clock: now };
                    sh.strong_reads.record(i, read, clock, false);
                } else {
                    sh.reads.record(i, ReadEpoch { slot: slot as u16, unit: 0, clock: now }, clock, true);
                }
            }
        }
        if let Some(i) = inflight {
            return Err(self.err(self.inflight_message("reads", b, block, i), Some(t)));
        }
        if let Some(failure) = failure {
            let msg = match failure {
                Ok(msg) => msg,
                Err((i, m, unit)) => {
                    let (warp, place) = (k % self.warps_per_block, smem_place(d, i));
                    let writer = self.slot_name(m.w_slot as usize, block);
                    if self.is_agent(m.w_slot as usize) && m.tracked_cp_async {
                        format!(
                            "data race: warp {warp} reads {place}, which a cp.async tracked by cp_async_mbarrier_arrive \
                             wrote, without having waited for the mbarrier phase that tracks it"
                        )
                    } else if self.is_agent(m.w_slot as usize) {
                        format!(
                            "data race: warp {warp} reads {place}, which a bulk copy wrote, without having waited for \
                             the mbarrier phase the copy completed"
                        )
                    } else if strong && m.strong {
                        format!(
                            "data race: warp {warp} reads {place}, written by {writer}: both accesses are strong, but {}",
                            mixed_sizes("read", unit, "write", m.unit)
                        )
                    } else {
                        format!(
                            "data race: warp {warp} reads {place}, written by {writer} since the last barrier (missing \
                             sync_threads?)"
                        )
                    }
                }
            };
            return Err(self.err(msg, Some(t)));
        }
        Ok(read_le(&self.shared[self.home[b.0 as usize]][block].data[byte..byte + size]))
    }

    /// The report for an access (`verb`) to byte `i` of shared buffer `b` of
    /// `block` while an asynchronous copy is still writing it.
    fn inflight_message(&self, verb: &str, b: BufferId, block: usize, i: usize) -> String {
        let place = smem_place(self.ir.buffer(b), i);
        if self.bulk_copy_writing(b, block, i) {
            format!(
                "{verb} {place} while a bulk copy is still writing it (wait on the copy's mbarrier until its phase is over)"
            )
        } else if self.tracked_copy_writing(b, block, i) {
            format!(
                "{verb} {place} while a cp.async is still writing it: the mbarrier that cp_async_mbarrier_arrive \
                 made track it has not completed the phase (wait on it, or for the copy's cp.async group)"
            )
        } else if verb == "reads" {
            format!("reads {place} while a cp.async is still writing it (missing cp_async_wait_group?)")
        } else {
            format!("{verb} {place} while a cp.async is still writing it")
        }
    }

    /// Check (and record) a write by `t` to `size` bytes of shared memory,
    /// strong (one element) or weak (an asynchronous copy's is weak). It
    /// races with the last write as a read does ([`Self::smem_read`]), and
    /// with reads by other warps that nothing orders before it, unless both
    /// are strong and cover the same bytes.
    pub(super) fn smem_claim_write(
        &mut self,
        b: BufferId,
        byte: usize,
        size: usize,
        t: usize,
        async_issue: bool,
        strong: bool,
    ) -> R {
        self.smem_claim_write_in(b, self.smem_block(t), byte, size, t, async_issue, strong)
    }

    /// [`Self::smem_claim_write`] of shared memory in `block` (another CTA of `t`'s
    /// cluster for a copy into a peer's shared memory).
    pub(super) fn smem_claim_write_in(
        &mut self,
        b: BufferId,
        block: usize,
        byte: usize,
        size: usize,
        t: usize,
        async_issue: bool,
        strong: bool,
    ) -> R {
        let k = self.clock_index(t);
        let slot = self.slot_of(k);
        let opts = self.opts;
        let clock = &self.clocks[k];
        let sh = &mut self.shared[self.home[b.0 as usize]][block];
        // What went wrong: the byte, and the earlier write or the read (and
        // its unit, if both are strong) it races with.
        let mut failure: Option<(usize, WriteRace)> = None;
        let mut inflight = None;
        for i in byte..byte + size {
            let m = &mut sh.meta[i];
            let unit = if strong { unit_code(i - byte, size) } else { 0 };
            let observed =
                !sh.strong_reads.is_empty() && sh.strong_reads.by(i, slot).is_some_and(|r| r.unit & OBSERVED != 0);
            let morally_strong = strong && m.strong && m.unit == unit;
            if m.inflight > 0 {
                inflight = Some(i);
                break;
            } else if opts.detect_races && !m.write_ordered_before(slot, clock) && !morally_strong && !observed {
                failure = Some((i, Ok(*m)));
                break;
            } else if opts.detect_races {
                // Another warp's read that nothing orders before this write,
                // and the unit of that read if it was strong.
                let unordered = |r: &&ReadEpoch| r.slot as usize != slot && r.clock > clock[r.slot as usize];
                let weak = sh.reads.of(i).iter().find(unordered).map(|r| (r.slot as usize, None));
                let reader = weak.or_else(|| {
                    if sh.strong_reads.is_empty() {
                        return None;
                    }
                    sh.strong_reads
                        .of(i)
                        .iter()
                        .filter(unordered)
                        .find(|r| !(strong && r.unit & !OBSERVED == unit))
                        .map(|r| (r.slot as usize, strong.then_some(r.unit & !OBSERVED)))
                });
                if let Some(reader) = reader {
                    failure = Some((i, Err(reader)));
                    break;
                }
            }
            if async_issue {
                m.inflight += 1;
            } else {
                m.w_clock = clock[slot];
                m.w_slot = slot as u16;
                m.written = true;
                m.tracked_cp_async = false;
                m.alt_clock = 0;
                m.strong = strong;
                m.unit = unit;
                if opts.detect_races {
                    sh.reads.clear(i);
                    if !sh.strong_reads.is_empty() {
                        sh.strong_reads.clear(i);
                    }
                }
            }
        }
        if let Some(i) = inflight {
            return Err(self.err(self.inflight_message("writes", b, block, i), Some(t)));
        }
        let Some((i, what)) = failure else { return Ok(()) };
        let (warp, place) = (k % self.warps_per_block, smem_place(self.ir.buffer(b), i));
        let unit = if strong { unit_code(i - byte, size) } else { 0 };
        let msg = match what {
            Ok(m) => {
                // `warps 0 and 1` in one block, `warp 0 of CTA 1 and warp 1`
                // across blocks.
                let writer = match self.slot_name(m.w_slot as usize, block) {
                    name if name.contains(" of CTA ") => name,
                    name => name.replacen("warp", "warps", 1),
                };
                if self.is_agent(m.w_slot as usize) && m.tracked_cp_async {
                    format!(
                        "data race: warp {warp} overwrites {place}, which a cp.async tracked by \
                         cp_async_mbarrier_arrive wrote, without having waited for the mbarrier phase that tracks it"
                    )
                } else if self.is_agent(m.w_slot as usize) {
                    format!(
                        "data race: warp {warp} overwrites {place}, which a bulk copy wrote, without having waited for \
                         the mbarrier phase the copy completed"
                    )
                } else if strong && m.strong {
                    format!(
                        "data race: {writer} and {warp} both write {place} between barriers: both writes are \
                         strong, but {}",
                        mixed_sizes("write", unit, "write", m.unit)
                    )
                } else {
                    format!("data race: {writer} and {warp} both write {place} between barriers")
                }
            }
            Err((other, read_unit)) => {
                let reader = self.slot_name(other, block);
                match read_unit {
                    Some(read_unit) => format!(
                        "data race: warp {warp} overwrites {place}, which {reader} read since the last barrier: both \
                         accesses are strong, but {}",
                        mixed_sizes("write", unit, "read", read_unit)
                    ),
                    None => format!(
                        "data race: warp {warp} overwrites {place}, which {reader} read since the last barrier \
                         (missing sync_threads?)"
                    ),
                }
            }
        };
        Err(self.err(msg, Some(t)))
    }

    /// Record that bytes of shared memory an asynchronous copy claimed have
    /// landed, written by clock slot `writer` (a warp of `block`'s cluster,
    /// or an agent) at its own time `clock`.
    pub(super) fn smem_land(
        &mut self,
        b: BufferId,
        block: usize,
        range: std::ops::Range<usize>,
        writer: usize,
        clock: u32,
    ) {
        self.progress();
        let sh = &mut self.shared[self.home[b.0 as usize]][block];
        for i in range.clone() {
            let m = &mut sh.meta[i];
            m.inflight -= 1;
            m.written = true;
            m.strong = false;
            m.unit = 0;
            m.w_slot = writer as u16;
            m.w_clock = clock;
            m.tracked_cp_async = false;
            m.alt_clock = 0;
            if !sh.reads.is_empty() {
                sh.reads.clear(i);
            }
            if !sh.strong_reads.is_empty() {
                sh.strong_reads.clear(i);
            }
        }
        // A weak write: nothing published on these bytes survives it.
        if !self.releases.is_empty() {
            for i in range {
                self.releases.remove(&(b.0, block, i));
            }
        }
    }

    /// An atomic update: a strong read and write, which do not race with
    /// other strong accesses (other atomics, ordered loads and stores) but
    /// do with plain ones that nothing orders against them, and which may
    /// not touch memory a `cp.async` is writing.
    pub(super) fn atomic_update(
        &mut self,
        b: BufferId,
        byte: usize,
        size: usize,
        t: usize,
        f: impl FnOnce(u64) -> u64,
    ) -> R<u64> {
        if self.ir.buffer(b).space == MemSpace::Shared {
            let block = self.smem_block(t);
            let d = self.ir.buffer(b);
            let detect = self.opts.detect_uninitialized;
            let sh = &self.shared[self.home[b.0 as usize]][block];
            let failure = (byte..byte + size).find_map(|i| {
                let m = &sh.meta[i];
                if m.inflight > 0 {
                    Some(self.inflight_message("atomic on", b, block, i))
                } else if detect && !m.written {
                    Some(format!("atomic on {}, which was never written", smem_place(d, i)))
                } else {
                    None
                }
            });
            if let Some(msg) = failure {
                return Err(self.err(msg, Some(t)));
            }
        }
        let old = self.read_as(b, byte, size, t, true)?;
        self.write_as(b, byte, size, t, f(old), true)?;
        Ok(old)
    }

    /// An atomic (with `result`, the old value) or a reduction by the
    /// `active` threads, one thread after another. Each element (each half
    /// of a packed pair) is a strong read and write of its own
    /// ([`Self::atomic_update`]). A relaxed read-modify-write continues the
    /// release sequences of the location and adds its warp's clock at its
    /// last fence; an atomic's read is strong, for the warp's next fence to
    /// acquire (a reduction's is not: PTX's `red; fence.acquire` is no
    /// acquire pattern). A release one (release or acq_rel) publishes its warp's clock too,
    /// joined with what the sequence it continues published, and the warp
    /// moves on to a new time. An acquire one (acquire or acq_rel) orders
    /// its warp after what it read was published, once every lane has
    /// read, as an acquire load does.
    pub(super) fn atomic(
        &mut self,
        active: &[usize],
        result: Option<ValueId>,
        kind: AtomicKind,
        ty: AtomicType,
        (buffer, index, peer): (BufferId, ValueId, Option<ValueId>),
        value: ValueId,
        compare: Option<ValueId>,
        sem: MemoryOrder,
    ) -> R {
        let elem = ty.access_elem(self.ir.dtype(value));
        let (size, step) = (ty.bits() / 8, elem.bytes());
        let release = matches!(sem, MemoryOrder::Release | MemoryOrder::AcqRel);
        let acquire = matches!(sem, MemoryOrder::Acquire | MemoryOrder::AcqRel);
        // The type each element is combined as: the half of a packed pair,
        // the integer type min and max compare as.
        let part = match ty.half() {
            Some(half) => half,
            None if ty.is_float() => elem,
            None => match (ty.is_signed(), size) {
                (true, 8) => DType::I64,
                (true, _) => DType::I32,
                (false, 8) => DType::U64,
                (false, _) => DType::U32,
            },
        };
        let mut published: Vec<(usize, Rc<[u32]>)> = Vec::new();
        if release {
            for &t in active {
                let k = self.clock_index(t);
                if !published.iter().any(|p| p.0 == k) {
                    published.push((k, self.clocks[k].as_slice().into()));
                }
            }
        }
        let mut found: Vec<(usize, Vec<u32>)> = Vec::new();
        for &t in active {
            let n = size / step;
            let (buffer, idx) = self.target(buffer, index, step, t, "atomic")?;
            // Another CTA's shared memory.
            self.smem_target = match peer {
                Some(p) => {
                    let what = if result.is_some() {
                        "an atomic (atom.shared::cluster)"
                    } else {
                        "a reduction (red.shared::cluster)"
                    };
                    Some(self.dsmem_target(p, t, what)?)
                }
                None => None,
            };
            if self.ir.buffer(buffer).space == MemSpace::Local {
                // Only a generic address gets here: PTX's atom and red on
                // local memory are undefined.
                return Err(self.err(
                    format!(
                        "an atomic through a generic pointer reaches the per-thread buffer `{}`: atomics need global \
                         or shared memory (PTX leaves atom and red on local memory undefined)",
                        self.ir.buffer(buffer).name
                    ),
                    Some(t),
                ));
            }
            let global = self.ir.buffer(buffer).space == MemSpace::Global;
            let byte = self.checked_range(buffer, idx, n, step, size, t, "atomic")?;
            let operand = mask(self.get(value, t), size * 8);
            let expected = compare.map(|c| mask(self.get(c, t), size * 8));
            let mut old_bits = 0u64;
            for k in 0..n {
                let at = byte + k * step;
                let shift = 8 * k * step;
                let x = mask(operand >> shift, step * 8);
                // A compare-and-swap that fails only reads (so a spin on one is a poll,
                // not progress).
                if kind == AtomicKind::CompareExchange {
                    let current = self.read_as(buffer, at, step, t, true)?;
                    if Some(current) != expected {
                        old_bits |= current << shift;
                        if result.is_some()
                            && let Some(clock) = self.released_on(buffer, at, step, step, t)
                        {
                            found.push((self.clock_index(t), clock));
                        }
                        continue;
                    }
                }
                let old = self.atomic_update(buffer, at, step, t, |old| match kind {
                    AtomicKind::Add if ty.is_float() => {
                        super::float::atomic_add(part, old, x, global && part == DType::F32)
                    }
                    AtomicKind::Add => mask(old.wrapping_add(x), step * 8),
                    AtomicKind::And => old & x,
                    AtomicKind::Or => old | x,
                    AtomicKind::Xor => old ^ x,
                    AtomicKind::Min | AtomicKind::Max => {
                        let less = compare_less(part, x, old);
                        if less == (kind == AtomicKind::Min) { x } else { old }
                    }
                    AtomicKind::Exchange => x,
                    AtomicKind::CompareExchange => {
                        if Some(old) == expected {
                            x
                        } else {
                            old
                        }
                    }
                })?;
                old_bits |= old << shift;
                // A reduction's read is strong but forms no acquire pattern
                // (PTX: `red; fence.acquire` is not one), so it finds nothing.
                if result.is_some()
                    && let Some(clock) = self.released_on(buffer, at, step, step, t)
                {
                    found.push((self.clock_index(t), clock));
                }
                let how = if release {
                    let k = self.clock_index(t);
                    Publish::ReleaseRmw(published.iter().find(|p| p.0 == k).expect("published").1.clone())
                } else {
                    Publish::Rmw
                };
                self.publish(buffer, at, step, step, t, how);
            }
            if let Some(r) = result {
                self.set(r, t, old_bits);
            }
        }
        self.smem_target = None;
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
        for (k, _) in published {
            let slot = self.slot_of(k);
            self.clocks[k][slot] += 1;
        }
        Ok(())
    }

    /// Account one instruction's accesses: `(thread, byte offset, bytes)`.
    pub(super) fn account(&mut self, space: MemSpace, accesses: &[(usize, usize, usize)], write: bool) {
        if accesses.is_empty() || space == MemSpace::Local {
            return;
        }
        let width = accesses[0].2;
        let lanes_per_phase = match width {
            0..=4 => 32,
            8 => 16,
            _ => 8,
        };
        for group in accesses.chunk_by(|a, b| self.warp_key(a.0) == self.warp_key(b.0)).collect::<Vec<_>>() {
            if space == MemSpace::Shared {
                self.shared_phases(group, width, lanes_per_phase);
                continue;
            }
            let mut ranges: Vec<(usize, usize)> = group.iter().map(|&(_, o, _)| (o, o + width)).collect();
            ranges.sort_unstable();
            let (mut unique, mut sectors, mut end, mut last_sector) = (0usize, 0u64, 0usize, usize::MAX);
            for (s, e) in ranges {
                let s2 = s.max(end);
                if e > s2 {
                    unique += e - s2;
                    end = e;
                }
                for sector in s / 32..e.div_ceil(32) {
                    if last_sector == usize::MAX || sector > last_sector {
                        sectors += 1;
                        last_sector = sector;
                    }
                }
            }
            self.report.global_sectors += sectors;
            self.report.global_sectors_ideal += unique.div_ceil(32) as u64;
            if write {
                self.report.global_bytes_written += unique as u64;
            } else {
                self.report.global_bytes_read += unique as u64;
            }
        }
    }

    pub(super) fn shared_phases(&mut self, group: &[(usize, usize, usize)], width: usize, lanes_per_phase: usize) {
        let words = width.div_ceil(4).max(1);
        let mut entries: Vec<(usize, usize, usize)> = Vec::with_capacity(group.len() * words);
        for &(t, off, _) in group {
            let phase = self.lane_of(t) / lanes_per_phase;
            entries.extend((0..words).map(|w| (phase, (off / 4 + w) % 32, off / 4 + w)));
        }
        entries.sort_unstable();
        entries.dedup();
        for phase in entries.chunk_by(|a, b| a.0 == b.0) {
            let worst = phase.chunk_by(|a, b| a.1 == b.1).map(<[_]>::len).max().unwrap_or(1);
            self.report.shared_wavefronts += worst as u64;
            self.report.shared_wavefronts_ideal += 1;
        }
    }

    /// The first thread of every warp in `active`, which must be whole warps.
    pub(super) fn full_warps(&self, active: &[usize], what: &str) -> R<Vec<usize>> {
        if !self.threads_per_block.is_multiple_of(32) {
            return Err(self
                .err(format!("{what} needs whole warps, but the block has {} threads", self.threads_per_block), None));
        }
        let mut bases = Vec::new();
        let mut i = 0;
        while i < active.len() {
            let key = self.warp_key(active[i]);
            let n = active[i..].iter().take_while(|&&t| self.warp_key(t) == key).count();
            if n != 32 {
                return Err(self.err(
                    format!(
                        "{what} executed by {n} of the 32 lanes of warp {} (divergent warp)",
                        self.warp_of(active[i])
                    ),
                    Some(active[i]),
                ));
            }
            bases.push(active[i]);
            i += n;
        }
        Ok(bases)
    }

    /// The active lanes of each warp, as `(first thread of the warp, lanes)`.
    pub(super) fn warps_of(&self, active: &[usize]) -> Vec<(usize, Vec<usize>)> {
        let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
        for &t in active {
            let lane = self.lane_of(t);
            let base = t - lane;
            match out.last_mut() {
                Some((b, lanes)) if *b == base => lanes.push(lane),
                _ => out.push((base, vec![lane])),
            }
        }
        out
    }

    pub(super) fn special(&self, reg: SpecialReg, t: usize) -> u64 {
        let block = self.block_of(t);
        let cfg = self.cfg;
        let lane = self.lane_of(t) as u32;
        match reg {
            SpecialReg::ThreadIdx(d) => self.thread_coord(t)[d.index()] as u64,
            SpecialReg::BlockIdx(d) => block[d.index()] as u64,
            SpecialReg::BlockDim(d) => cfg.block[d.index()] as u64,
            SpecialReg::GridDim(d) => cfg.grid[d.index()] as u64,
            SpecialReg::LaneId => lane as u64,
            SpecialReg::WarpSize => 32,
            SpecialReg::WarpId => self.warp_of(t) as u64,
            SpecialReg::NWarpId => self.threads_per_block.div_ceil(32) as u64,
            SpecialReg::SmId => self.block_rank(t) as u64,
            SpecialReg::NSmId => self.blocks.len() as u64,
            SpecialReg::GridId => 1,
            SpecialReg::LaneMask(kind) => {
                let bit = 1u64 << lane;
                (match kind {
                    LaneMask::Eq => bit,
                    LaneMask::Le => (bit << 1) - 1,
                    LaneMask::Lt => bit - 1,
                    LaneMask::Ge => !(bit - 1),
                    LaneMask::Gt => !((bit << 1) - 1),
                }) & 0xffff_ffff
            }
            SpecialReg::Clock => self.clock & 0xffff_ffff,
            SpecialReg::Clock64 | SpecialReg::GlobalTimer => self.clock,
            SpecialReg::GlobalTimerLo => self.clock & 0xffff_ffff,
            SpecialReg::ClusterIdx(d) => (block[d.index()] / cfg.cluster[d.index()]) as u64,
            SpecialReg::ClusterDim(d) => (cfg.grid[d.index()] / cfg.cluster[d.index()]) as u64,
            SpecialReg::BlockInClusterIdx(d) => (block[d.index()] % cfg.cluster[d.index()]) as u64,
            SpecialReg::BlockInClusterDim(d) => cfg.cluster[d.index()] as u64,
            SpecialReg::DynamicSmemSize => cfg.dynamic_smem_bytes as u64,
            SpecialReg::TotalSmemSize => total_smem_size(&self.smem, cfg.dynamic_smem_bytes as usize) as u64,
            SpecialReg::AggrSmemSize => aggr_smem_size(&self.smem, cfg.dynamic_smem_bytes as usize) as u64,
            // The blocks of a cluster run together, ranked x fastest as
            // `%cluster_ctarank` ranks them.
            SpecialReg::ClusterCtaRank => (self.block_rank(t) % self.cluster_size) as u64,
            SpecialReg::ClusterNCtaRank => self.cluster_size as u64,
        }
    }

    /// The index of thread `t`'s warp in [`Self::clocks`].
    pub(super) fn clock_index(&self, t: usize) -> usize {
        self.block_rank(t) * self.warps_per_block + self.warp_of(t) as usize
    }

    /// The clock slot of warp `k` (by key, `block * warps_per_block +
    /// warp`): vector clocks span the warp's cluster, with `stride` slots
    /// per block, its warps' first and then its agents'.
    pub(super) fn slot_of(&self, k: usize) -> usize {
        let w = self.warps_per_block;
        ((k / w) % self.cluster_size) * self.stride + k % w
    }

    /// The clock slot of `block`'s asynchronous-copy agent.
    pub(super) fn agent_slot(&self, block: usize) -> usize {
        (block % self.cluster_size) * self.stride + self.warps_per_block + COPY_AGENT
    }

    /// Whether clock slot `slot` is an agent's.
    pub(super) fn is_agent(&self, slot: usize) -> bool {
        slot % self.stride >= self.warps_per_block
    }

    /// Clock slots per vector clock: every warp and agent of a cluster.
    pub(super) fn slots(&self) -> usize {
        self.cluster_size * self.stride
    }

    /// How a report by a warp of `block` names clock slot `slot`: `warp 3`
    /// in its own block, `warp 3 of CTA 1` in another of its cluster.
    pub(super) fn slot_name(&self, slot: usize, block: usize) -> String {
        let (rank, i) = (slot / self.stride, slot % self.stride);
        let what =
            if i >= self.warps_per_block { "the asynchronous-copy agent".to_string() } else { format!("warp {i}") };
        if rank == block % self.cluster_size { what } else { format!("{what} of CTA {rank}") }
    }

    /// The first block of `block`'s cluster.
    pub(super) fn cluster_base(&self, block: usize) -> usize {
        block - block % self.cluster_size
    }

    /// The key of byte `byte` of buffer `b`, as thread `t` addresses it, in
    /// [`Self::releases`].
    pub(super) fn release_key(&self, b: BufferId, byte: usize, t: usize) -> (u32, usize, usize) {
        let block = if self.ir.buffer(b).space == MemSpace::Shared { self.smem_block(t) } else { GLOBAL };
        (b.0, block, byte)
    }

    /// The join of the clocks published on the `size` bytes at `byte` of
    /// `b`, which a strong read reads as elements of `elem` bytes, by warps
    /// of thread `t`'s block (the only ones whose clocks it shares). An
    /// element joins what a write published only if that write covered
    /// exactly the element's bytes (a morally strong pair).
    pub(super) fn released_on(&self, b: BufferId, byte: usize, size: usize, elem: usize, t: usize) -> Option<Vec<u32>> {
        if self.releases.is_empty() {
            return None;
        }
        let cluster = self.cluster_ids[self.block_rank(t) / self.cluster_size];
        let mut acc: Option<Vec<u32>> = None;
        for start in (byte..byte + size).step_by(elem) {
            let found: Option<Vec<&Released>> = (0..elem)
                .map(|k| {
                    self.releases
                        .get(&self.release_key(b, start + k, t))
                        .filter(|r| r.cluster == cluster && r.unit == unit_code(k, elem))
                })
                .collect();
            for r in found.into_iter().flatten() {
                match &mut acc {
                    Some(a) => join(a, &r.clock),
                    None => acc = Some(r.clock.to_vec()),
                }
            }
        }
        acc
    }

    /// Update what is published on `size` bytes at `byte` of `b` after
    /// thread `t` wrote them as elements of `elem` bytes ([`Publish`]).
    pub(super) fn publish(&mut self, b: BufferId, byte: usize, size: usize, elem: usize, t: usize, how: Publish) {
        if matches!(how, Publish::Weak) && self.releases.is_empty() {
            return;
        }
        let (block, warp) = (self.block_rank(t), self.warp_of(t) as usize);
        let cluster = self.cluster_ids[block / self.cluster_size];
        let fence = self.fence_release[self.clock_index(t)].clone();
        if matches!(how, Publish::Strong | Publish::Rmw) && self.releases.is_empty() && fence.is_none() {
            return;
        }
        for i in byte..byte + size {
            let key = self.release_key(b, i, t);
            let unit = unit_code((i - byte) % elem, elem);
            // Only a write of the same bytes continues what an earlier one
            // published there: accesses of other sizes are not morally strong.
            let old = self.releases.get(&key).filter(|r| r.unit == unit).cloned();
            let new = match &how {
                Publish::Weak => None,
                Publish::Release(clock) => Some(Released { cluster, block, warp, clock: clock.clone(), unit }),
                Publish::ReleaseRmw(clock) => {
                    let mut joined = clock.to_vec();
                    if let Some(r) = old.as_ref().filter(|r| r.cluster == cluster) {
                        join(&mut joined, &r.clock);
                    }
                    Some(Released { cluster, block, warp, clock: joined.into(), unit })
                }
                Publish::Strong | Publish::Rmw => {
                    // A strong store keeps only what its own warp published
                    // there; an atomic continues anyone's release sequence.
                    let rmw = matches!(how, Publish::Rmw);
                    let kept =
                        old.clone().filter(|r| r.cluster == cluster && ((r.block, r.warp) == (block, warp) || rmw));
                    match (kept, &fence) {
                        (Some(mut r), Some(f)) => {
                            let mut clock = r.clock.to_vec();
                            join(&mut clock, f);
                            r.clock = clock.into();
                            r.warp = warp;
                            Some(r)
                        }
                        (Some(r), None) => Some(r),
                        (None, Some(f)) => Some(Released { cluster, block, warp, clock: f.clone(), unit }),
                        (None, None) if rmw => old,
                        (None, None) => None,
                    }
                }
            };
            match new {
                Some(r) => {
                    self.releases.insert(key, r);
                }
                None => {
                    self.releases.remove(&key);
                }
            }
        }
    }

    /// A fence (`fence.acq_rel`, or the `fence.sc` of a seq_cst access)
    /// executed by warp `k`: it acquires what its strong loads since its
    /// last fence found released, takes its clock as what its later strong
    /// writes publish, and moves on to a new time of its own.
    fn fence_warp(&mut self, k: usize) {
        let slot = self.slot_of(k);
        if let Some(found) = self.pending_acquire[k].take() {
            join(&mut self.clocks[k], &found);
        }
        self.fence_release[k] = Some(self.clocks[k].as_slice().into());
        self.clocks[k][slot] += 1;
    }

    /// [`Self::fence_warp`] for each warp with a thread in `active`.
    pub(super) fn fence_lanes(&mut self, active: &[usize]) {
        let mut done = Vec::new();
        for &t in active {
            let k = self.clock_index(t);
            if !done.contains(&k) {
                done.push(k);
                self.fence_warp(k);
            }
        }
    }

    /// A non-coherent load by `t` of `size` bytes at `byte` of global buffer
    /// `b`: none of them may have been written during the launch.
    pub(super) fn nc_read(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R {
        let Some(state) = self.nc[b.0 as usize].as_mut() else { return Ok(()) };
        if let Some(i) = (byte..byte + size).find(|&i| state.written[i]) {
            let d = self.ir.buffer(b);
            return Err(self.err(
                format!(
                    "a non-coherent load (ld.global.nc: invariant, or MemoryOrder::Constant) reads {}[{}], which the \
                     kernel wrote: the non-coherent cache may return stale data, so it is only for memory that no \
                     thread writes while the kernel runs",
                    d.name,
                    i / d.elem.bytes()
                ),
                Some(t),
            ));
        }
        state.read[byte..byte + size].fill(true);
        Ok(())
    }

    /// A write by `t` of `size` bytes at `byte` of global buffer `b`: none
    /// of them may have been read by a non-coherent load during the launch.
    pub(super) fn nc_write(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R {
        let Some(state) = self.nc[b.0 as usize].as_mut() else { return Ok(()) };
        if let Some(i) = (byte..byte + size).find(|&i| state.read[i]) {
            let d = self.ir.buffer(b);
            return Err(self.err(
                format!(
                    "writes {}[{}], which a non-coherent load (ld.global.nc: invariant, or MemoryOrder::Constant) \
                     reads: such loads are only for memory that no thread writes while the kernel runs",
                    d.name,
                    i / d.elem.bytes()
                ),
                Some(t),
            ));
        }
        state.written[byte..byte + size].fill(true);
        Ok(())
    }

    /// The warps of `block` in `releasing` publish their clocks (joined into
    /// `release` as well); those in `acquiring` then see all of it, and every
    /// warp involved moves on to a new time of its own.
    pub(super) fn synchronize(&mut self, block: usize, releasing: &[usize], acquiring: &[usize], release: &[u32]) {
        let w = self.warps_per_block;
        let mut joined = vec![0; self.slots()];
        join(&mut joined, release);
        for &warp in releasing {
            join(&mut joined, &self.clocks[block * w + warp]);
        }
        for &warp in acquiring {
            join(&mut self.clocks[block * w + warp], &joined);
        }
        for &warp in releasing.iter().chain(acquiring) {
            let slot = self.slot_of(block * w + warp);
            let clock = &mut self.clocks[block * w + warp];
            clock[slot] = clock[slot].max(joined[slot]) + 1;
        }
    }
}
