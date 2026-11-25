// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's tensor memory (roadmap part m10-tmem-core), modelled from
//! the PTX ISA ("Tensor Memory", 9.7.18.1, and `tcgen05.alloc`,
//! `tcgen05.dealloc` and `tcgen05.relinquish_alloc_permit`, 9.7.18.7.1).
//!
//! * **Storage.** Each CTA has its own TMEM of 128 lanes by 576 32-bit
//!   columns (sm_107's; other targets use the first 512), never written at
//!   launch. Co-resident CTAs never share it, unlike on hardware, where the
//!   CTAs on an SM share one. Each cell keeps its last writer (a warp, or the
//!   CTA's tensor-core agent) and that writer's clock, for the race checks of
//!   the `tcgen05.ld`/`st` and MMA parts.
//! * **Allocation.** `tcgen05.alloc` is warp-collective: all 32 lanes of one
//!   warp issue it with the same operands. It blocks until the columns are
//!   free, then writes the TMEM address (lane 0 in bits 31:16, the column in
//!   bits 15:0) to the shared-memory word, as a write of the allocating warp
//!   (readers need a barrier; the race detector checks it). It takes the
//!   lowest free range aligned to its size, as the hardware does; code
//!   should not rely on the exact column. With `.cta_group::2` the two CTAs
//!   of a pair (even and odd `%cluster_ctarank`) each issue it with one warp
//!   and get the same columns: one pair allocation that completes once both
//!   have. An allocation that can never be served (no other warp frees
//!   columns) hangs on hardware and is reported as a deadlock listing the
//!   live allocations.
//! * **Rules.** Freeing must name the start of a live allocation with its
//!   column count, `cta_group` and exclusiveness; every allocation must be
//!   freed before the CTA exits; all `tcgen05` instructions of a kernel use
//!   one `cta_group`; no allocation after the CTA relinquished its permit.
//!   sm_107's exclusive allocation (a multiple of 32 columns up to 576) waits
//!   until it would be the only live allocation, and blocks the others
//!   until it is freed.
//!
//! Roadmap part m10-tmem-ld-st adds `tcgen05.ld`, `tcgen05.st` and
//! `tcgen05.wait::{ld,st}` (PTX ISA 9.7.18.8):
//!
//! * **Issue.** Both are warp-collective: all 32 lanes of the warp, one
//!   TMEM address. Every cell they touch ([`crate::ir::TmemShape::cells`]) must be in
//!   a lane of the warp's quadrant (warp `w` of a warpgroup reaches lanes
//!   `32 (w % 4)` to `+31`) and in a live allocation of the CTA.
//! * **Loads.** The cells are read at the issue (so the warp's own later
//!   `tcgen05.st`, which the PTX ISA orders after the load, does not change
//!   what it loads); each must have been written, by a write ordered before
//!   the issue, and not by another warp's store still in flight. The
//!   registers stay pending until the warp's `tcgen05.wait::ld`: reading one
//!   earlier is reported. A write by another warp to the cells between the
//!   issue and the wait is reported as a race.
//! * **Stores.** The cells take their bits at the issue but stay in flight
//!   until the warp's `tcgen05.wait::st`, when they become the warp's write
//!   (later readers need a barrier or an mbarrier after it). Another warp
//!   reading them meanwhile, a store over another warp's load still in
//!   flight, and freeing the columns (`tcgen05.dealloc`) before the wait
//!   are reported. `.unpack::16b` writes the register's halves to the low
//!   halves of two adjacent columns; the simulator writes zero to their
//!   high halves, which code should not rely on.
//! * **Fences.** `tcgen05.fence::{before,after}_thread_sync` only order the
//!   hardware's pipelines; the backend places them around thread
//!   synchronisations by itself, and the simulator ignores them.

use std::collections::HashMap;
use std::ops::Range;

use super::{AsyncOp, Machine, R, Status, Trigger, Wait};
use crate::ir::{BufferId, Inst, Loc, Op, Tcgen05Inst, ValueId};

/// TMEM lanes (rows) per CTA.
pub(super) const TMEM_LANES: usize = 128;
/// TMEM columns per CTA as stored: sm_107's 576.
pub(super) const TMEM_COLUMNS: usize = 576;
/// The columns ordinary (non-exclusive) allocations share.
const SHARED_COLUMNS: u32 = 512;

/// A live allocation of a CTA.
#[derive(Clone)]
struct Allocation {
    col: u32,
    ncols: u32,
    /// The allocating warp (by key).
    warp: usize,
    loc: Loc,
    two_cta: bool,
    exclusive: bool,
}

/// Who wrote a TMEM cell last.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum TmemWriter {
    /// A warp (by key), with `tcgen05.st` or a copy into TMEM.
    Warp(usize),
    /// The CTA's tensor-core agent (an MMA's accumulator).
    TensorCore,
}

/// A TMEM cell: its bits, and its last writer with that writer's clock
/// entry at the write (none while never written).
#[derive(Clone, Copy, Default)]
pub(super) struct Cell {
    pub bits: u32,
    pub writer: Option<(TmemWriter, u32)>,
    /// The warp (by key) whose `tcgen05.st` of the cell is still in flight.
    pub storing: Option<usize>,
}

/// One CTA's tensor memory.
#[derive(Default)]
pub(super) struct BlockTmem {
    allocs: Vec<Allocation>,
    /// Where the CTA relinquished its allocation permit.
    relinquished: Option<Loc>,
    /// `TMEM_LANES * TMEM_COLUMNS` cells, made at the first access.
    cells: Vec<Cell>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl BlockTmem {
    fn cells(&mut self) -> &mut [Cell] {
        if self.cells.is_empty() {
            self.cells = vec![Cell::default(); TMEM_LANES * TMEM_COLUMNS];
        }
        &mut self.cells
    }

    /// Cell (`lane`, `col`), or an error naming it if it was never written.
    pub(super) fn read(&mut self, lane: usize, col: usize) -> Result<Cell, String> {
        let c = self.cells()[lane * TMEM_COLUMNS + col];
        if c.writer.is_none() {
            return Err(format!("TMEM lane {lane}, column {col} is read before anything wrote it"));
        }
        Ok(c)
    }

    /// Write cell (`lane`, `col`) as `writer` at clock entry `time`.
    pub(super) fn write(&mut self, lane: usize, col: usize, bits: u32, writer: TmemWriter, time: u32) {
        self.cells()[lane * TMEM_COLUMNS + col] = Cell { bits, writer: Some((writer, time)), storing: None };
    }

    /// Cell (`lane`, `col`) to change.
    pub(super) fn cell_mut(&mut self, lane: usize, col: usize) -> &mut Cell {
        &mut self.cells()[lane * TMEM_COLUMNS + col]
    }
}

/// A TMEM cell (lane, column) a load read, with its last writer then.
type ReadCell = (usize, usize, Option<(TmemWriter, u32)>);

/// A `tcgen05.ld` of a warp that its `tcgen05.wait::ld` has not completed.
struct PendingLd<'a> {
    /// The asynchronous operation holding its register results.
    id: usize,
    inst: &'a Inst,
    block: usize,
    /// The cells it read, with their last writer at the issue.
    cells: Vec<ReadCell>,
}

/// A `tcgen05.st` of a warp that its `tcgen05.wait::st` has not completed.
struct PendingSt<'a> {
    inst: &'a Inst,
    block: usize,
    cells: Vec<(usize, usize)>,
}

/// The lane-quadrant rule of `tcgen05.ld` and `tcgen05.st` (PTX ISA
/// 9.7.18.8.1): warp `w` of a warpgroup (`w = warp_id % 4`) may access only
/// TMEM lanes `32 w` to `32 w + 31`. Returns why `lane` is out of reach.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn lane_quadrant(warp_id: usize, lane: usize) -> Result<(), String> {
    let q = warp_id % 4;
    if lane / 32 == q {
        return Ok(());
    }
    Err(format!(
        "warp {warp_id} accesses TMEM lane {lane}, but warp {warp_id} of its warpgroup (warp_id % 4 = {q}) may access only \
         lanes {} to {}",
        32 * q,
        32 * q + 31
    ))
}

/// A warp that issued a pending `tcgen05.alloc`, and the shared word to
/// write the address to.
struct Issuer {
    block: usize,
    warp: usize,
    thread: usize,
    buffer: BufferId,
    byte: usize,
}

/// A `tcgen05.alloc` that has not completed: a pair allocation waiting for
/// its peer, or one waiting for columns.
struct PendingAlloc<'a> {
    inst: &'a Inst,
    ncols: u32,
    two_cta: bool,
    exclusive: bool,
    issuers: Vec<Issuer>,
}

/// The tensor memory of the resident CTAs.
#[derive(Default)]
pub(super) struct State<'a> {
    pub(super) blocks: HashMap<usize, BlockTmem>,
    /// Pending allocations, by CTA (the even CTA of a pair).
    pending: HashMap<usize, PendingAlloc<'a>>,
    /// The `cta_group` of the kernel's first `tcgen05` instruction (whether
    /// it is 2), and where.
    group: Option<(bool, Loc)>,
    /// Each warp's (by key) loads and stores in flight.
    loads: HashMap<usize, Vec<PendingLd<'a>>>,
    stores: HashMap<usize, Vec<PendingSt<'a>>>,
    /// The tcgen05 MMAs in flight (roadmap part m10-umma-f16), in issue order.
    pub(super) mmas: Vec<super::umma::UmmaFlight>,
    /// m10-rubin-umma: the collector buffers, by CTA and slot (0: A, 5: the
    /// sm_107 B; m11-prims-mma-forms: `1 + i` the `.ws` B buffer `b<i>`).
    pub(super) collectors: HashMap<(usize, u8), super::umma::Collector>,
}

impl State<'_> {
    /// A cluster starts in the slots `blocks`: fresh, never-written TMEM.
    pub(super) fn start_cluster(&mut self, blocks: Range<usize>) {
        self.blocks.retain(|b, _| !blocks.contains(b));
        self.pending.retain(|b, _| !blocks.contains(b));
        self.loads.retain(|_, v| {
            v.retain(|l| !blocks.contains(&l.block));
            !v.is_empty()
        });
        self.stores.retain(|_, v| {
            v.retain(|l| !blocks.contains(&l.block));
            !v.is_empty()
        });
        self.mmas.retain(|f| !blocks.contains(&f.block));
        self.collectors.retain(|(b, _), _| !blocks.contains(b));
    }
}

fn at(loc: Loc) -> String {
    format!("{}:{}", loc.file(), loc.line())
}

impl<'a> Machine<'a> {
    /// The live allocations of `block`: `(first column, columns)`.
    pub(super) fn tmem_allocs(&self, block: usize) -> Vec<(u32, u32)> {
        self.tmem.blocks.get(&block).map(|b| b.allocs.iter().map(|a| (a.col, a.ncols)).collect()).unwrap_or_default()
    }

    /// The live allocations of `block`, for messages.
    pub(super) fn tmem_live(&self, block: usize) -> String {
        let w = self.warps_per_block;
        let allocs = self.tmem.blocks.get(&block).map(|s| s.allocs.as_slice()).unwrap_or_default();
        if allocs.is_empty() {
            return "none".to_string();
        }
        allocs
            .iter()
            .map(|a| {
                format!(
                    "columns {} to {} ({} by warp {} at {}{}{})",
                    a.col,
                    a.col + a.ncols - 1,
                    a.ncols,
                    a.warp % w,
                    at(a.loc),
                    if a.two_cta { ", a CTA-pair allocation" } else { "" },
                    if a.exclusive { ", exclusive" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// `tcgen05.alloc`, `tcgen05.dealloc` or
    /// `tcgen05.relinquish_alloc_permit` by the warps of `group`.
    pub(super) fn tcgen05_issue(&mut self, inst: &'a Inst, group: &[usize]) -> R {
        let Op::Tcgen05 { inst: which, two_cta, exclusive, buffer, args } = &inst.op else {
            unreachable!("a tcgen05 instruction")
        };
        let (which, two_cta, exclusive) = (*which, *two_cta, *exclusive);
        let w = self.warps_per_block;
        let what = which.name();
        self.progress();
        for &k in group {
            let lanes = self.warp_lanes(k).to_vec();
            let t = lanes[0];
            if matches!(which, Tcgen05Inst::WaitLd | Tcgen05Inst::WaitSt) {
                if lanes.len() != 32 {
                    return Err(self.err(self.divergent(what, lanes.len(), k), Some(t)));
                }
                if which == Tcgen05Inst::WaitLd {
                    self.tmem_wait_ld(k)?;
                } else {
                    self.tmem_wait_st(k);
                }
                self.advance(k);
                continue;
            }
            if lanes.len() != 32 {
                return Err(self.err(
                    format!(
                        "{what} executed by {} of the 32 lanes of warp {}: it is warp-collective (.sync.aligned), so \
                         every lane of the warp executes it together, outside divergent code",
                        lanes.len(),
                        k % w
                    ),
                    Some(t),
                ));
            }
            for &a in args {
                let v = self.get(a, t);
                if let Some(&l) = lanes.iter().find(|&&l| self.get(a, l) != v) {
                    return Err(self.err(
                        format!(
                            "{what}: lane {} of warp {} gives an operand {:#x} where lane 0 gives {v:#x}: every lane \
                             of the warp gives the same operands (.sync.aligned)",
                            self.lane_of(l),
                            k % w,
                            self.get(a, l)
                        ),
                        Some(l),
                    ));
                }
            }
            self.tcgen05_group(what, two_cta, inst.loc, t)?;
            let block = k / w;
            match which {
                Tcgen05Inst::Alloc => {
                    let buffer = buffer.expect("tcgen05.alloc has a shared-memory slot");
                    self.tmem_alloc(inst, k, t, buffer, args, two_cta, exclusive)?;
                }
                Tcgen05Inst::Dealloc => {
                    self.tmem_dealloc(k, t, args, two_cta, exclusive)?;
                    self.advance(k);
                }
                Tcgen05Inst::RelinquishAllocPermit => {
                    self.tmem.blocks.entry(block).or_default().relinquished.get_or_insert(inst.loc);
                    self.advance(k);
                }
                Tcgen05Inst::CheckAddress
                | Tcgen05Inst::WaitLd
                | Tcgen05Inst::WaitSt
                | Tcgen05Inst::FenceBeforeThreadSync
                | Tcgen05Inst::FenceAfterThreadSync => unreachable!("per thread, or handled above"),
            }
        }
        Ok(())
    }

    /// A `tcgen05` instruction `what` (at `loc`, by thread `t`) with
    /// `.cta_group::2` when `two_cta`: the first fixes the kernel's
    /// `cta_group`, and every later one must use the same (m10-cta-pair
    /// adds the MMAs and commits).
    pub(super) fn tcgen05_group(&mut self, what: &str, two_cta: bool, loc: Loc, t: usize) -> R {
        match self.tmem.group {
            Some((g, first)) if g != two_cta => {
                let name = |g: bool| if g { "cta_group::2" } else { "cta_group::1" };
                Err(self.err(
                    format!(
                        "{what} with .{} after a tcgen05 instruction with .{} ({}): all tcgen05 instructions of a \
                         kernel use the same cta_group",
                        name(two_cta),
                        name(g),
                        at(first)
                    ),
                    Some(t),
                ))
            }
            Some(_) => Ok(()),
            None => {
                self.tmem.group = Some((two_cta, loc));
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tmem_alloc(
        &mut self,
        inst: &'a Inst,
        k: usize,
        t: usize,
        buffer: BufferId,
        args: &[ValueId],
        two_cta: bool,
        exclusive: bool,
    ) -> R {
        let w = self.warps_per_block;
        let block = k / w;
        let byte = self.checked_range(buffer, self.index(args[0], t), 1, 4, 4, t, "tcgen05.alloc")?;
        let ncols = self.get(args[1], t) as u32;
        if exclusive {
            if ncols == 0 || !ncols.is_multiple_of(32) || ncols as usize > TMEM_COLUMNS {
                return Err(self.err(
                    format!(
                        "tcgen05.alloc.exclusive of {} columns: an exclusive allocation takes a multiple of 32 \
                         columns, at most {TMEM_COLUMNS}",
                        ncols as i32
                    ),
                    Some(t),
                ));
            }
        } else if !ncols.is_power_of_two() || !(32..=SHARED_COLUMNS).contains(&ncols) {
            return Err(self.err(
                format!(
                    "tcgen05.alloc of {} columns: the count must be a power of two from 32 to {SHARED_COLUMNS}",
                    ncols as i32
                ),
                Some(t),
            ));
        }
        if let Some(loc) = self.tmem.blocks.get(&block).and_then(|s| s.relinquished) {
            return Err(self.err(
                format!(
                    "tcgen05.alloc by a CTA that relinquished its allocation permit at {} \
                     (relinquish_tmem_alloc_permit): it may not allocate tensor memory again",
                    at(loc)
                ),
                Some(t),
            ));
        }
        let base = if two_cta {
            if !self.cluster_size.is_multiple_of(2) {
                return Err(self.err(
                    format!(
                        "tcgen05.alloc.cta_group::2 in a cluster of {} CTA(s): a CTA pair is two CTAs of a cluster \
                         (even and odd %cluster_ctarank), so the cluster has an even number of CTAs",
                        self.cluster_size
                    ),
                    Some(t),
                ));
            }
            block & !1
        } else {
            block
        };
        let issuer = Issuer { block, warp: k, thread: t, buffer, byte };
        if let Some(p) = self.tmem.pending.get(&base) {
            let other = p.issuers[0].warp;
            let mismatch = p.ncols != ncols || p.exclusive != exclusive;
            if !(p.two_cta && two_cta) || p.issuers.iter().any(|i| i.block == block) {
                return Err(self.err(
                    format!(
                        "tcgen05.alloc by warp {} while the tcgen05.alloc of warp {} of {} ({}) has not completed: one \
                         warp of a CTA allocates, and its allocation blocks until it is served",
                        k % w,
                        other % w,
                        self.block_name(other / w),
                        at(p.inst.loc)
                    ),
                    Some(t),
                ));
            }
            if mismatch {
                return Err(self.err(
                    format!(
                        "tcgen05.alloc.cta_group::2 of {ncols} columns by warp {}, but its peer CTA's warp {} \
                         allocates {} columns ({}): both CTAs of a pair allocate the same columns",
                        k % w,
                        other % w,
                        p.ncols,
                        at(p.inst.loc)
                    ),
                    Some(t),
                ));
            }
            self.tmem.pending.get_mut(&base).expect("pending").issuers.push(issuer);
        } else {
            self.tmem.pending.insert(base, PendingAlloc { inst, ncols, two_cta, exclusive, issuers: vec![issuer] });
        }
        self.advance(k);
        self.warps[k].status = Status::Blocked(Wait::TmemAlloc { block: base, inst });
        self.tmem_try_complete(base)
    }

    /// Complete the pending allocation of `base` if every warp it needs has
    /// issued it and the columns are free.
    fn tmem_try_complete(&mut self, base: usize) -> R {
        let Some(p) = self.tmem.pending.get(&base) else { return Ok(()) };
        if p.issuers.len() < if p.two_cta { 2 } else { 1 } {
            return Ok(());
        }
        let blocks: Vec<usize> = p.issuers.iter().map(|i| i.block).collect();
        let allocs = |b: &usize| self.tmem.blocks.get(b).map(|s| s.allocs.as_slice()).unwrap_or_default();
        let col = if p.exclusive {
            blocks.iter().all(|b| allocs(b).is_empty()).then_some(0)
        } else if blocks.iter().any(|b| allocs(b).iter().any(|a| a.exclusive)) {
            None
        } else {
            (0..SHARED_COLUMNS).step_by(p.ncols as usize).find(|&c| {
                blocks.iter().all(|b| allocs(b).iter().all(|a| c + p.ncols <= a.col || a.col + a.ncols <= c))
            })
        };
        let Some(col) = col else { return Ok(()) };
        let p = self.tmem.pending.remove(&base).expect("pending");
        let round = self.round;
        for i in &p.issuers {
            self.tmem.blocks.entry(i.block).or_default().allocs.push(Allocation {
                col,
                ncols: p.ncols,
                warp: i.warp,
                loc: p.inst.loc,
                two_cta: p.two_cta,
                exclusive: p.exclusive,
            });
            // The address (lane 0, column `col`), written by the allocating warp.
            self.write_as(i.buffer, i.byte, 4, i.thread, col as u64, false)?;
            let s = &mut self.warps[i.warp];
            s.status = Status::Runnable;
            s.ready_round = round + 1;
        }
        self.progress();
        Ok(())
    }

    fn tmem_dealloc(&mut self, k: usize, t: usize, args: &[ValueId], two_cta: bool, exclusive: bool) -> R {
        let w = self.warps_per_block;
        let block = k / w;
        let taddr = self.get(args[0], t) as u32;
        let ncols = self.get(args[1], t) as u32;
        let (lane, col) = (taddr >> 16, taddr & 0xffff);
        let found =
            self.tmem.blocks.get(&block).and_then(|s| {
                s.allocs.iter().position(|a| lane == 0 && a.col == col).map(|i| (i, s.allocs[i].clone()))
            });
        let Some((index, a)) = found else {
            return Err(self.err(
                format!(
                    "tcgen05.dealloc of {ncols} columns at TMEM address {taddr:#x}, which is not the start of a live \
                     allocation of this CTA (freed twice, or never allocated); its live allocations: {}",
                    self.tmem_live(block)
                ),
                Some(t),
            ));
        };
        let problem = if a.ncols != ncols {
            Some(format!("{ncols} columns, but the allocation has {}", a.ncols))
        } else if a.two_cta != two_cta {
            Some(format!(
                "cta_group::{}, but the allocation was made with cta_group::{}",
                if two_cta { 2 } else { 1 },
                if a.two_cta { 2 } else { 1 }
            ))
        } else if a.exclusive != exclusive {
            Some(if a.exclusive {
                "a plain dealloc, but an exclusive allocation is freed by tcgen05.dealloc.exclusive of the same size"
                    .to_string()
            } else {
                "dealloc.exclusive, but the allocation is not exclusive".to_string()
            })
        } else {
            None
        };
        if let Some((warp, verb, wait, loc)) = self.tmem_in_flight(block, a.col..a.col + a.ncols) {
            return Err(self.err(
                format!(
                    "tcgen05.dealloc of columns {} to {} while the tcgen05.{verb} of warp {} ({}) into them has not \
                     completed: the warp waits for it ({wait}) before the columns are freed (and a barrier orders \
                     that wait before the dealloc)",
                    a.col,
                    a.col + a.ncols - 1,
                    warp % w,
                    at(loc),
                ),
                Some(t),
            ));
        }
        if let Some(problem) = problem {
            return Err(self.err(
                format!(
                    "tcgen05.dealloc at TMEM address {taddr:#x} (allocated by warp {} at {}) with {problem}",
                    a.warp % w,
                    at(a.loc)
                ),
                Some(t),
            ));
        }
        self.tmem.blocks.get_mut(&block).expect("a TMEM state").allocs.remove(index);
        // Freed columns may serve a waiting allocation of this CTA or its pair.
        let pair = if self.cluster_size.is_multiple_of(2) { block & !1 } else { block };
        self.tmem_try_complete(block)?;
        if pair != block {
            self.tmem_try_complete(pair)?;
        }
        Ok(())
    }

    /// m10-umma-pipelines: where warp `k`'s oldest `tcgen05.ld` that its
    /// `tcgen05.wait::ld` has not completed was issued, if any.
    pub(super) fn tmem_pending_load(&self, k: usize) -> Option<Loc> {
        self.tmem.loads.get(&k).and_then(|v| v.first()).map(|p| p.inst.loc)
    }

    /// A `tcgen05.ld` or `tcgen05.st` of a warp of `block` still in flight
    /// over `cols`: the warp (by key), "ld" or "st", its wait, and where.
    pub(super) fn tmem_in_flight(
        &self,
        block: usize,
        cols: Range<u32>,
    ) -> Option<(usize, &'static str, &'static str, Loc)> {
        // m10-umma-f16: an MMA writing D there (m10-umma-kinds: or reading A there).
        let overlaps = |c: &Range<usize>| (c.start as u32) < cols.end && cols.start < c.end as u32;
        if let Some(f) = self
            .tmem
            .mmas
            .iter()
            .find(|f| f.block == block && (overlaps(&f.cols) || f.a_cells.iter().any(|(_, c)| overlaps(c))))
        {
            return Some((
                f.warp,
                "mma",
                "the wait on the mbarrier its tcgen05.commit arrives on, after_thread_sync",
                f.loc,
            ));
        }
        self.tmem_ldst_in_flight(block, cols)
    }

    /// A `tcgen05.ld` or `tcgen05.st` of a warp of `block` still in flight
    /// over `cols` (MMAs aside).
    pub(super) fn tmem_ldst_in_flight(
        &self,
        block: usize,
        cols: Range<u32>,
    ) -> Option<(usize, &'static str, &'static str, Loc)> {
        let hits = |c: &(usize, usize)| cols.contains(&(c.1 as u32));
        for (&k, v) in &self.tmem.stores {
            if let Some(p) = v.iter().find(|p| p.block == block && p.cells.iter().any(hits)) {
                return Some((k, "st", "tcgen05.wait::st, fence_view_async_tmem_store", p.inst.loc));
            }
        }
        for (&k, v) in &self.tmem.loads {
            if let Some(p) = v.iter().find(|p| p.block == block && p.cells.iter().any(|&(l, c, _)| hits(&(l, c)))) {
                return Some((k, "ld", "tcgen05.wait::ld, fence_view_async_tmem_load", p.inst.loc));
            }
        }
        None
    }

    /// The message for a warp-collective `what` executed by `n` lanes of
    /// warp `k`.
    pub(super) fn divergent(&self, what: &str, n: usize, k: usize) -> String {
        format!(
            "{what} executed by {n} of the 32 lanes of warp {}: it is warp-collective (.sync.aligned), so every lane \
             of the warp executes it together, outside divergent code",
            k % self.warps_per_block
        )
    }

    /// `tcgen05.ld` or `tcgen05.st` by the threads of `active`.
    pub(super) fn tmem_ldst(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        let (elems, taddr, shape, num, packed, load, red, sp) = match &inst.op {
            Op::Tcgen05Ld { results, taddr, shape, num, pack, red, sp } => {
                (results, *taddr, *shape, *num, *pack, true, red.as_ref(), sp.as_ref())
            }
            Op::Tcgen05St { values, taddr, shape, num, unpack } => {
                (values, *taddr, *shape, *num, *unpack, false, None, None)
            }
            _ => unreachable!("tcgen05.ld or tcgen05.st"),
        };
        // Each 32-bit register holds `per` elements of `eb` bits, the first in the low bits.
        let nregs = shape.registers(num) as usize;
        // A sparse-compress load returns half the registers as data: one Float32 each.
        let epr = if sp.is_some() { 1 } else { elems.len() / nregs };
        let eb = 32 / epr as u32;
        let mask = if eb == 32 { u32::MAX } else { (1u32 << eb) - 1 };
        let what = if load { "tcgen05.ld" } else { "tcgen05.st" };
        let w = self.warps_per_block;
        self.progress();
        for (base, lanes) in self.warps_of(active) {
            let k = self.clock_index(base);
            let warp = k % w;
            let block = k / w;
            if lanes.len() != 32 {
                return Err(self.err(self.divergent(what, lanes.len(), k), Some(base + lanes[0])));
            }
            let addr = self.get(taddr, base) as u32;
            if let Some(l) = (1..32).find(|&l| self.get(taddr, base + l) as u32 != addr) {
                return Err(self.err(
                    format!(
                        "{what}: lane {l} of warp {warp} gives the TMEM address {:#x} where lane 0 gives {addr:#x}: \
                         every lane of the warp gives the same address (.sync.aligned)",
                        self.get(taddr, base + l) as u32
                    ),
                    Some(base + l),
                ));
            }
            let (lane0, col0) = ((addr >> 16) as usize, (addr & 0xffff) as usize);
            let split = shape.half_split(num, packed);
            // Each thread's registers: the cells they hold (absolute lane, column).
            let mut cells: Vec<Vec<Vec<(usize, usize)>>> = Vec::with_capacity(32);
            for i in 0..32u32 {
                let per: Vec<Vec<(usize, usize)>> = (0..nregs as u32)
                    .map(|r| {
                        shape
                            .cells(i, r, num, packed)
                            .into_iter()
                            .map(|(l, c)| (lane0 + l as usize, col0 + c as usize))
                            .collect()
                    })
                    .collect();
                cells.push(per);
            }
            let name = format!(
                "{what}{}{}.{}.x{num}{}",
                if red.is_some() { ".red" } else { "" },
                if sp.is_some() { ".spcompress" } else { "" },
                shape.ptx(),
                match (packed, load) {
                    (false, _) => "",
                    (true, true) => ".pack::16b",
                    (true, false) => ".unpack::16b",
                }
            );
            let split_note = split.map(|s| format!(" (half split {s})")).unwrap_or_default();
            // Lanes of the warp's quadrant, columns of a live allocation.
            let allocs: Vec<(u32, u32)> = self
                .tmem
                .blocks
                .get(&block)
                .map(|b| b.allocs.iter().map(|a| (a.col, a.ncols)).collect())
                .unwrap_or_default();
            for &(l, c) in cells.iter().flatten().flatten() {
                if let Some(f) = self.umma_covering(block, l, c) {
                    return Err(self.err(
                        format!(
                            "{name} at TMEM address {addr:#x} reaches lane {l}, column {c}, the accumulator of the \
                             tcgen05.mma of warp {} ({}), which is still in flight: wait on the mbarrier its \
                             tcgen05.commit arrives on (then tcgen05.fence::after_thread_sync, which the backend \
                             places) before accessing D",
                            f.warp % w,
                            at(f.loc)
                        ),
                        Some(base),
                    ));
                }
                // m10-umma-kinds: a store over the A an MMA in flight reads from TMEM.
                if !load && let Some(f) = self.tmem.mmas.iter().find(|f| f.reads_a(block, l, c)) {
                    return Err(self.err(
                        format!(
                            "{name} at TMEM address {addr:#x} writes lane {l}, column {c}, which the tcgen05.mma of \
                             warp {} ({}) reads as A (or as scale factors) and which is still in flight: wait on the \
                             mbarrier its tcgen05.commit arrives on before overwriting it",
                            f.warp % w,
                            at(f.loc)
                        ),
                        Some(base),
                    ));
                }
                if let Err(why) = lane_quadrant(warp, l) {
                    return Err(self.err(
                        format!("{name} at TMEM address {addr:#x}{split_note}: {why} (PTX ISA 9.7.18.8.1)"),
                        Some(base),
                    ));
                }
                if c >= TMEM_COLUMNS || !allocs.iter().any(|&(a, n)| (a as usize..(a + n) as usize).contains(&c)) {
                    return Err(self.err(
                        format!(
                            "{name} at TMEM address {addr:#x}{split_note} reaches lane {l}, column {c}, which is not in \
                             a live TMEM allocation of this CTA (live allocations: {})",
                            self.tmem_live(block)
                        ),
                        Some(base),
                    ));
                }
            }
            if load {
                let mut seen = Vec::new();
                let mut results = Vec::with_capacity(32 * elems.len());
                for (i, per) in cells.iter().enumerate() {
                    let t = base + i;
                    let mut loaded: Vec<u64> = Vec::new();
                    for cs in per.iter() {
                        let mut bits = 0u32;
                        for (h, &(l, c)) in cs.iter().enumerate() {
                            let cell = self.tmem.blocks.entry(block).or_default().cell_mut(l, c).to_owned();
                            self.tmem_may_read(&name, k, l, c, &cell, t)?;
                            bits |= if packed { (cell.bits & 0xffff) << (16 * h) } else { cell.bits };
                            seen.push((l, c, cell.writer));
                        }
                        for j in 0..epr {
                            loaded.push(((bits >> (eb * j as u32)) & mask) as u64);
                        }
                    }
                    // tcgen05.ld.spcompress: two of every four values, and their metadata.
                    if let Some(sp) = sp {
                        let (kept, meta) = super::super::float::tmem_load_spcompress(sp.op, sp.nan, &loaded);
                        for (&id, v) in elems.iter().zip(kept) {
                            results.push((id, t, v));
                        }
                        for (&id, v) in sp.mdata.iter().zip(meta) {
                            results.push((id, t, v));
                        }
                    } else {
                        for (&id, &v) in elems.iter().zip(&loaded) {
                            results.push((id, t, v));
                        }
                    }
                    // tcgen05.ld.red: the thread's reduction over what it loaded.
                    if let Some(red) = red {
                        results.push((red.result, t, super::super::float::tmem_load_red(*red, loaded.iter().copied())));
                    }
                }
                let id = self.ops.len();
                for &(v, t, _) in &results {
                    // A placeholder until the wait lands the result.
                    self.set(v, t, 0);
                    self.pending_results.insert((v, t), id);
                }
                self.ops.push(AsyncOp {
                    thread: base,
                    agent: self.agent_slot(block),
                    writes: Vec::new(),
                    reads: Vec::new(),
                    global_writes: Vec::new(),
                    global_reduce: None,
                    results,
                    trigger: Trigger::TmemLd,
                    done: None,
                });
                self.tmem.loads.entry(k).or_default().push(PendingLd { id, inst, block, cells: seen });
            } else {
                let mut written = Vec::new();
                for (i, per) in cells.iter().enumerate() {
                    let t = base + i;
                    for (r, cs) in per.iter().enumerate() {
                        let bits = (0..epr)
                            .map(|j| ((self.get(elems[r * epr + j], t) as u32) & mask) << (eb * j as u32))
                            .fold(0, |a, b| a | b);
                        for (h, &(l, c)) in cs.iter().enumerate() {
                            self.tmem_may_write(&name, k, block, l, c, t)?;
                            let cell = self.tmem.blocks.entry(block).or_default().cell_mut(l, c);
                            cell.bits = if packed { (bits >> (16 * h)) & 0xffff } else { bits };
                            cell.storing = Some(k);
                            written.push((l, c));
                        }
                    }
                }
                self.tmem.stores.entry(k).or_default().push(PendingSt { inst, block, cells: written });
            }
        }
        Ok(())
    }

    /// May warp `k` (thread `t`) read TMEM cell (`l`, `c`) now?
    fn tmem_may_read(&self, name: &str, k: usize, l: usize, c: usize, cell: &Cell, t: usize) -> R {
        let w = self.warps_per_block;
        if let Some(other) = cell.storing.filter(|&o| o != k) {
            return Err(self.err(
                format!(
                    "{name} reads TMEM lane {l}, column {c} while the tcgen05.st of warp {} into it is still in flight: \
                     the storing warp waits for it (tcgen05.wait::st, fence_view_async_tmem_store) and a barrier or \
                     mbarrier orders that wait before this load",
                    other % w
                ),
                Some(t),
            ));
        }
        match cell.writer {
            None if cell.storing.is_none() && self.opts.detect_uninitialized => Err(self.err(
                format!("{name} reads TMEM lane {l}, column {c}, which nothing wrote (tensor memory starts undefined)"),
                Some(t),
            )),
            Some((TmemWriter::TensorCore, time))
                if self.opts.detect_races && self.clocks[k][self.agent_slot(k / w)] < time =>
            {
                Err(self.err(
                    format!(
                        "data race: {name} by warp {} reads TMEM lane {l}, column {c}, an MMA's accumulator, but the \
                         warp has not observed the MMA's completion: wait on the mbarrier its tcgen05.commit arrives \
                         on (directly, or through a barrier after another warp's wait) before reading D",
                        k % w
                    ),
                    Some(t),
                ))
            }
            Some((TmemWriter::Warp(o), time))
                if o != k
                    && cell.storing.is_none()
                    && self.opts.detect_races
                    && self.clocks[k][self.slot_of(o)] < time =>
            {
                Err(self.err(
                    format!(
                        "data race: {name} by warp {} reads TMEM lane {l}, column {c}, but nothing orders the store \
                         of warp {} that wrote it (completed by its tcgen05.wait::st) before this load (a barrier or \
                         an mbarrier wait after that wait)",
                        k % w,
                        o % w
                    ),
                    Some(t),
                ))
            }
            _ => Ok(()),
        }
    }

    /// May warp `k` (thread `t`) write TMEM cell (`l`, `c`) of `block` now?
    fn tmem_may_write(&mut self, name: &str, k: usize, block: usize, l: usize, c: usize, t: usize) -> R {
        let w = self.warps_per_block;
        let cell = self.tmem.blocks.entry(block).or_default().cell_mut(l, c).to_owned();
        if let Some(other) = cell.storing.filter(|&o| o != k) {
            return Err(self.err(
                format!(
                    "data race: {name} by warp {} writes TMEM lane {l}, column {c} while the tcgen05.st of warp {} \
                     into it is still in flight",
                    k % w,
                    other % w
                ),
                Some(t),
            ));
        }
        for (&o, v) in &self.tmem.loads {
            if o != k && v.iter().any(|p| p.block == block && p.cells.iter().any(|&(pl, pc, _)| (pl, pc) == (l, c))) {
                return Err(self.err(
                    format!(
                        "data race: {name} by warp {} writes TMEM lane {l}, column {c} while the tcgen05.ld of warp {} \
                         reading it has not completed (tcgen05.wait::ld)",
                        k % w,
                        o % w
                    ),
                    Some(t),
                ));
            }
        }
        if let Some((TmemWriter::Warp(o), time)) = cell.writer
            && o != k
            && self.opts.detect_races
            && self.clocks[k][self.slot_of(o)] < time
        {
            return Err(self.err(
                format!(
                    "data race: {name} by warp {} writes TMEM lane {l}, column {c}, but nothing orders the earlier \
                     store of warp {} to it before this one",
                    k % w,
                    o % w
                ),
                Some(t),
            ));
        }
        Ok(())
    }

    /// `tcgen05.wait::ld` by warp `k`: its loads land, unless another agent
    /// wrote what they read since they were issued.
    fn tmem_wait_ld(&mut self, k: usize) -> R {
        let w = self.warps_per_block;
        let Some(pending) = self.tmem.loads.remove(&k) else { return Ok(()) };
        for p in pending {
            for &(l, c, writer) in &p.cells {
                let cell = self.tmem.blocks.entry(p.block).or_default().cell_mut(l, c).to_owned();
                let other = match (cell.storing, cell.writer) {
                    (Some(o), _) if o != k => Some(o),
                    (_, Some((TmemWriter::Warp(o), _))) if o != k && cell.writer != writer => Some(o),
                    _ => None,
                };
                if let Some(o) = other {
                    return Err(self.err(
                        format!(
                            "data race: warp {} stored to TMEM lane {l}, column {c} between the tcgen05.ld of warp {} \
                             ({}) that reads it and that warp's tcgen05.wait::ld",
                            o % w,
                            k % w,
                            at(p.inst.loc)
                        ),
                        Some(self.ops[p.id].thread),
                    ));
                }
            }
            let slot = self.slot_of(k);
            let clock = self.clocks[k][slot];
            self.complete(p.id, slot, clock);
        }
        Ok(())
    }

    /// `tcgen05.wait::st` by warp `k`: its stores become its writes now.
    fn tmem_wait_st(&mut self, k: usize) {
        let Some(pending) = self.tmem.stores.remove(&k) else { return };
        let time = self.clocks[k][self.slot_of(k)];
        for p in pending {
            let b = self.tmem.blocks.entry(p.block).or_default();
            for (l, c) in p.cells {
                let cell = b.cell_mut(l, c);
                if cell.storing == Some(k) {
                    cell.storing = None;
                    cell.writer = Some((TmemWriter::Warp(k), time));
                }
            }
        }
    }

    /// `retrieve_tmem_ptr`'s check: the address `v` each thread of `active`
    /// loaded lies in a live allocation of its CTA.
    pub(super) fn tmem_check_address(&self, v: ValueId, active: &[usize]) -> R {
        for &t in active {
            let addr = self.get(v, t) as u32;
            let block = self.block_rank(t);
            let (lane, col) = (addr >> 16, addr & 0xffff);
            let live = self
                .tmem
                .blocks
                .get(&block)
                .is_some_and(|s| s.allocs.iter().any(|a| lane == 0 && (a.col..a.col + a.ncols).contains(&col)));
            if !live {
                return Err(self.err(
                    format!(
                        "retrieve_tmem_ptr: the address {addr:#x} read from shared memory is not in a live TMEM \
                         allocation of this CTA (live allocations: {})",
                        self.tmem_live(block)
                    ),
                    Some(t),
                ));
            }
        }
        Ok(())
    }

    /// After the last warp of `block` exited: everything it allocated must
    /// have been freed.
    pub(super) fn tmem_check_exit(&self, block: usize) -> R {
        if self.tmem.blocks.get(&block).is_none_or(|s| s.allocs.is_empty()) {
            return Ok(());
        }
        Err(self.err(
            format!(
                "CTA {} ({}) exits with tensor memory still allocated: {}. Free every allocation (dealloc_tmem) \
                 before the kernel exits",
                block % self.cluster_size,
                self.block_name(block),
                self.tmem_live(block)
            ),
            None,
        ))
    }

    /// For hang and deadlock reports: what a warp waiting at the
    /// `tcgen05.alloc` pending for `base` waits for.
    pub(super) fn tmem_wait_report(&self, base: usize, ctx: &str) -> String {
        let Some(p) = self.tmem.pending.get(&base) else { return String::new() };
        let site = format!("tcgen05.alloc of {} columns ({}){ctx}", p.ncols, at(p.inst.loc));
        if p.issuers.len() < if p.two_cta { 2 } else { 1 } {
            let peer = if p.issuers[0].block == base { base + 1 } else { base };
            return format!(
                " at {site}, a CTA-pair allocation that {} has not issued (one warp of each CTA of the pair \
                 executes it)",
                self.block_name(peer)
            );
        }
        let lists: Vec<String> =
            p.issuers.iter().map(|i| format!("{}: {}", self.block_name(i.block), self.tmem_live(i.block))).collect();
        format!(
            " at {site}, which no free range of columns can serve while these allocations are live ({}){}; nothing \
             frees them, so it would hang on hardware",
            lists.join("; "),
            if p.exclusive { " (an exclusive allocation waits until it is the only one)" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_quadrants() {
        assert!(lane_quadrant(0, 0).is_ok());
        assert!(lane_quadrant(5, 63).is_ok());
        assert!(lane_quadrant(7, 127).is_ok());
        let e = lane_quadrant(4, 32).unwrap_err();
        assert!(e.contains("lanes 0 to 31"), "{e}");
    }

    #[test]
    fn cells_start_unwritten() {
        let mut m = BlockTmem::default();
        assert!(m.read(3, 500).is_err());
        m.write(3, 500, 7, TmemWriter::TensorCore, 2);
        let c = m.read(3, 500).unwrap();
        assert_eq!((c.bits, c.writer), (7, Some((TmemWriter::TensorCore, 2))));
        assert!(m.read(3, 575).is_err());
    }
}
