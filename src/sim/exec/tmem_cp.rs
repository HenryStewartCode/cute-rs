// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's shared-memory-to-TMEM copies, `tcgen05.cp`, modelled from the PTX ISA (9.7.18.9.2) independently of
//! cute-rs's layout algebra.
//!
//! * **Issue.** `cute.copy` of an S2T atom is called by a whole warp, which
//!   elects the lane that issues each `tcgen05.cp` (CuTe DSL elects it
//!   itself, and warns against wrapping the copy in `elect_one`). The
//!   simulator requires the warp to be converged and the TMEM address and
//!   descriptor to be the same in every lane, then executes the copy once.
//! * **Source.** The descriptor is decoded with [`UmmaOperandMap`] as a
//!   K-major operand of `rows x row_bits` bits in 32-bit units (the PTX
//!   ISA's core matrices: 8 rows of 16 bytes, the leading byte offset
//!   between core matrices along the row, the stride byte offset between
//!   groups of 8 rows, and the swizzle). Each byte read must have been
//!   written, by a write ordered before the issue, through the generic
//!   proxy only with a `fence.proxy.async` since, and not by an asynchronous
//!   copy still in flight.
//! * **Destination.** Word `w` of source row `r` is written to TMEM column
//!   `w` after the address's, in the lanes [`TmemCpShape::lanes_of_row`]
//!   gives (one, or 2 or 4 for the broadcast forms) after the address's
//!   lane. The columns must be in a live allocation.
//! * **Completion.** As an MMA: the copy joins the CTA's tensor-core pipe,
//!   in order with its MMAs (which may read what it wrote at once), and is
//!   in flight until a thread observes the mbarrier phase of the
//!   `tcgen05.commit` its warp makes. Until then its source bytes are being
//!   read (overwriting them is reported) and its TMEM cells written (a
//!   `tcgen05.ld` of them is reported). With `cta_group::2` the pair's even
//!   CTA issues it, and each CTA of the pair copies from its own shared
//!   memory into its own TMEM.

use std::ops::Range;

use super::tmem::TMEM_COLUMNS;
use super::umma::UmmaFlight;
use super::{AsyncOp, AsyncRead, Machine, R, Trigger, smem_place};
use crate::ir::{BufferId, Inst, Op, TmemCpShape};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::tcgen05::{UmmaDescEncoding, UmmaOperandMap};
use crate::tensor::AddressSpace;

impl<'a> Machine<'a> {
    /// `tcgen05.cp` issued by the warps of `active`.
    pub(super) fn tmem_cp_issue(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        let Op::Tcgen05Cp { shape, two_cta, taddr, desc, .. } = &inst.op else { unreachable!("a tcgen05.cp") };
        let (shape, two_cta) = (*shape, *two_cta);
        let name = format!("tcgen05.cp.cta_group::{}.{}", if two_cta { 2 } else { 1 }, shape.ptx());
        let w = self.warps_per_block;
        self.progress();
        for (base, lanes) in self.warps_of(active) {
            let k = self.clock_index(base);
            let block = k / w;
            if lanes.len() != 32 {
                return Err(self.err(
                    format!(
                        "{name} (cute.copy of an SMEM-to-TMEM atom) executed by {} of the 32 lanes of warp {}: the \
                         copy is called by a whole, converged warp, which elects the lane that issues it (do not call \
                         it in divergent code or inside elect_one)",
                        lanes.len(),
                        k % w
                    ),
                    Some(base + lanes[0]),
                ));
            }
            for (v, what) in [(*taddr, "TMEM address"), (*desc, "shared-memory descriptor")] {
                let x = self.get(v, base);
                if let Some(l) = (1..32).find(|&l| self.get(v, base + l) != x) {
                    return Err(self.err(
                        format!(
                            "{name}: its {what} is {:#x} in lane {l} of warp {} but {x:#x} in lane 0: the operands of \
                             the copy are the same in every lane of the issuing warp",
                            self.get(v, base + l),
                            k % w
                        ),
                        Some(base + l),
                    ));
                }
            }
            self.tcgen05_group(&name, two_cta, inst.loc, base)?;
            if two_cta && (!self.cluster_size.is_multiple_of(2) || block % 2 == 1) {
                return Err(self.err(
                    format!(
                        "{name} issued by warp {} of CTA {} of a cluster of {}: a CTA pair's copy is issued by the \
                         pair's even CTA (%cluster_ctarank even) of a cluster with an even number of CTAs",
                        k % w,
                        block % self.cluster_size,
                        self.cluster_size
                    ),
                    Some(base),
                ));
            }
            let addr = self.get(*taddr, base) as u32;
            let d = self.get(*desc, base);
            self.tmem_cp_execute(inst, &name, shape, two_cta, addr, d, base, k, block)?;
        }
        Ok(())
    }

    fn tmem_cp_execute(
        &mut self,
        inst: &'a Inst,
        name: &str,
        shape: TmemCpShape,
        two_cta: bool,
        addr: u32,
        desc: u64,
        base: usize,
        k: usize,
        block: usize,
    ) -> R {
        let (lane0, col0) = ((addr >> 16) as usize, (addr & 0xffff) as usize);
        let max_lane0 = if shape == TmemCpShape::S4x256b { 31 } else { 0 };
        if lane0 > max_lane0 {
            return Err(self.err(
                format!(
                    "{name}: the TMEM address {addr:#x} has lane {lane0}, but the copy writes {} from lane {}",
                    if shape == TmemCpShape::S4x256b { "lanes 32 apart" } else { "all 128 lanes" },
                    if shape == TmemCpShape::S4x256b { "0 to 31" } else { "0" }
                ),
                Some(base),
            ));
        }
        let words = shape.row_bits() / 32;
        let cols = col0..col0 + words;
        let pair: Vec<usize> = if two_cta { vec![block, block + 1] } else { vec![block] };
        let mut dest_lanes: Vec<usize> =
            (0..shape.rows()).flat_map(|r| shape.lanes_of_row(r)).map(|l| lane0 + l).collect();
        dest_lanes.sort_unstable();
        for &blk in &pair {
            let allocs = self.tmem_allocs(blk);
            if cols.end > TMEM_COLUMNS
                || !allocs.iter().any(|&(a, c)| a as usize <= cols.start && cols.end <= (a + c) as usize)
            {
                return Err(self.err(
                    format!(
                        "{name}: the destination (columns {} to {} at TMEM address {addr:#x}) is not in a live TMEM \
                         allocation of {} (live allocations: {})",
                        cols.start,
                        cols.end - 1,
                        if blk == block { "this CTA" } else { "its peer CTA" },
                        self.tmem_live(blk)
                    ),
                    Some(base),
                ));
            }
            if let Some((warp, verb, wait, loc)) = self.tmem_ldst_in_flight(blk, cols.start as u32..cols.end as u32) {
                return Err(self.err(
                    format!(
                        "{name} writes TMEM columns {} to {} while the tcgen05.{verb} of warp {} ({}:{}) of them has \
                         not completed ({wait}, then a barrier with the tcgen05 fences, before the copy)",
                        cols.start,
                        cols.end - 1,
                        warp % self.warps_per_block,
                        loc.file(),
                        loc.line()
                    ),
                    Some(base),
                ));
            }
        }
        let map = UmmaOperandMap::new(desc, UmmaDescEncoding::Sm100, OperandMajorMode::K, 32, shape.rows(), words)
            .map_err(|why| {
                self.err(format!("{name}: the descriptor {desc:#018x} cannot be read: {why}"), Some(base))
            })?;
        let mut ranges: Vec<(usize, BufferId, Range<usize>)> = Vec::new();
        let mut out: Vec<(usize, usize, usize, u32)> = Vec::new();
        for &blk in &pair {
            for row in 0..shape.rows() {
                for word in 0..words {
                    let a = map.byte_address(row, word);
                    let (b, byte) = self.resolve(AddressSpace::Smem, a).map_err(|why| {
                        self.err(
                            format!("{name}: word {word} of row {row} (descriptor {desc:#018x}): {why}"),
                            Some(base),
                        )
                    })?;
                    if byte + 4 > self.buffer_bytes(b) {
                        return Err(self.err(
                            format!("{name}: word {word} of row {row} runs past the end of its shared buffer"),
                            Some(base),
                        ));
                    }
                    self.tmem_cp_may_read(name, b, blk, byte..byte + 4, k)?;
                    let sh = &self.shared[self.home[b.0 as usize]][blk];
                    let bits = u32::from_le_bytes(sh.data[byte..byte + 4].try_into().unwrap());
                    for l in shape.lanes_of_row(row) {
                        out.push((blk, lane0 + l, col0 + word, bits));
                    }
                    match ranges.last_mut() {
                        Some((lblk, lb, r)) if *lblk == blk && *lb == b && r.end == byte => r.end = byte + 4,
                        _ => ranges.push((blk, b, byte..byte + 4)),
                    }
                }
            }
        }
        for (blk, buffer, bytes) in &ranges {
            self.async_proxy_may_read(*buffer, *blk, bytes.clone(), k).map_err(|why| {
                self.err(
                    format!(
                        "{name} reads its source through the async proxy, but {why}: generic writes of a tcgen05.cp \
                         source need fence_view_async_shared (fence.proxy.async) before the copy"
                    ),
                    Some(base),
                )
            })?;
        }
        // The cells take their bits now (later MMAs of the pipe read them); they stay in flight.
        for (blk, lane, col, bits) in out {
            self.tmem.blocks.entry(blk).or_default().cell_mut(lane, col).bits = bits;
        }
        let id = self.ops.len();
        let reads = ranges.into_iter().map(|(blk, buffer, bytes)| AsyncRead { buffer, block: blk, bytes }).collect();
        self.ops.push(AsyncOp {
            thread: base,
            agent: self.agent_slot(block),
            writes: Vec::new(),
            reads,
            global_writes: Vec::new(),
            global_reduce: None,
            results: Vec::new(),
            trigger: Trigger::Tcgen05Mma,
            done: None,
        });
        self.any_async_reads = true;
        let lanes: Vec<Range<usize>> = dest_lanes.iter().map(|&l| l..l + 1).collect();
        for &blk in &pair {
            self.tmem.mmas.push(UmmaFlight {
                id,
                block: blk,
                warp: k,
                lanes: lanes.clone(),
                cols: cols.clone(),
                loc: inst.loc,
                a_cells: Vec::new(),
            });
        }
        Ok(())
    }

    /// May the copy issued by warp `k` read bytes `range` of shared buffer
    /// `b` of `block`?
    fn tmem_cp_may_read(&self, name: &str, b: BufferId, block: usize, range: Range<usize>, k: usize) -> R {
        let (slot, clock) = (self.slot_of(k), &self.clocks[k]);
        let w = self.warps_per_block;
        let t = (k / w) * self.threads_per_block + (k % w) * 32;
        for i in range {
            let m = &self.shared[self.home[b.0 as usize]][block].meta[i];
            let place = smem_place(self.ir.buffer(b), i);
            if m.inflight > 0 {
                return Err(self.err(
                    format!(
                        "{name} reads {place} while an asynchronous copy is still writing it (wait for the copy's \
                         mbarrier phase first)"
                    ),
                    Some(t),
                ));
            }
            if !m.written {
                if self.opts.detect_uninitialized {
                    return Err(self.err(format!("{name} reads {place}, which was never written"), Some(t)));
                }
                continue;
            }
            if self.opts.detect_races && !m.write_ordered_before(slot, clock) {
                return Err(self.err(
                    format!(
                        "data race: the {name} of warp {} reads {place}, but nothing orders its last write, by {}, \
                         before the copy (a barrier, or the wait on the mbarrier of the copy that wrote it)",
                        k % w,
                        self.slot_name(m.w_slot as usize, block)
                    ),
                    Some(t),
                ));
            }
        }
        Ok(())
    }
}
