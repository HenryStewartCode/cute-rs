// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's warpgroup MMAs (roadmap part m8-wgmma-core):
//! `wgmma.mma_async`, `wgmma.fence`, `wgmma.commit_group` and
//! `wgmma.wait_group`, modelled from the PTX ISA independently of cute-rs's
//! layout algebra.
//!
//! * **Issue.** `wgmma.mma_async` is `.sync.aligned` over a warpgroup: the
//!   simulator holds each warp at it until all four warps of the warpgroup
//!   have reached the same instruction with all their lanes (a warp that
//!   never comes is reported as a hang), then executes it once. Its
//!   descriptors and `scale-d` must be the same in all 128 threads.
//! * **Operands.** Each descriptor is decoded (start address, leading and
//!   stride byte offsets, swizzle mode; [`decode_descriptor`]) and the
//!   operand read element by element through the PTX ISA's canonical
//!   K-major or MN-major layout ([`canonical_offset`]) and the address-based
//!   swizzle ([`swizzle_address`]). The bytes must have been written, by
//!   writes ordered before the issue in every warp of the warpgroup, and
//!   generic-proxy writes must have been followed by `fence.proxy.async`
//!   (TMA writes are ordered by the mbarrier wait). Until the wait that
//!   completes it, the operation reads them: overwriting them is reported;
//!   after it they count as the waiting warp's reads (a later writer must be
//!   ordered after the wait).
//! * **Register A** (roadmap part m8-wgmma-variants). With A from registers
//!   each thread's fragment is placed by the PTX ISA's A-fragment layout
//!   ([`a_coord`]), and its registers, like the accumulators, must not have
//!   been accessed by another instruction after the warp's last
//!   `wgmma.fence`.
//! * **Arithmetic.** For each accumulator (the D fragment of the PTX ISA:
//!   [`d_coord`]) the simulator starts from C (or 0 when `scale-d` is false)
//!   and adds the K products (16 for 16-bit operands, 32 for 8-bit ones) in
//!   order of K in Float32 (each product of two Float16, BFloat16, E4M3 or
//!   E5M2 values is exact in Float32), then rounds once to the accumulator
//!   type. The hardware's summation order is not documented, and Hopper's
//!   FP8 MMAs accumulate with reduced internal precision, so results agree
//!   with an SM90 GPU exactly only when the sums are exact (small integers,
//!   say), and otherwise within a tolerance. 8-bit integer MMAs are exact:
//!   C plus the products, wrapping modulo 2^32 (no `.satfinite`).
//! * **Completion.** One asynchronous operation per warp holds that warp's
//!   results, which land at the warp's `wgmma.wait_group` that covers it;
//!   reading them earlier is reported, and loop carries and the operand
//!   fences pass them on still pending.
//! * **Fences.** A warp's `wgmma.mma_async` needs a `wgmma.fence` of the warp
//!   before it, after the last access to its accumulators by another
//!   instruction; the accumulators of another MMA of the same shape need
//!   none (the PTX ISA's exception for chained MMAs).

use std::collections::{HashMap, VecDeque};
use std::ops::Range;

use super::{AsyncOp, AsyncRead, Machine, R, Status, Trigger, Wait, from_f32, smem_place, to_f64, warp_list};
use crate::ir::{BufferId, DType, Inst, KernelIr, Op, ValueId, WgmmaKind};
use crate::tensor::AddressSpace;

/// Bytes of a shared buffer.
type ByteRange = (BufferId, Range<usize>);

/// Whether `op` takes in-flight accumulators as operands, which the general
/// check of pending register results must let through.
pub(super) fn passes_accumulators(op: &Op) -> bool {
    matches!(op, Op::Wgmma { .. } | Op::WgmmaFence { .. } | Op::WgmmaCommitGroup | Op::WgmmaWait { .. })
}

/// The last access to an accumulator (for the fence rule).
#[derive(Clone, Copy)]
enum Origin {
    /// The result of a warpgroup MMA of shape `kind`, issued at `at`.
    Mma { kind: WgmmaKind, at: u64 },
    /// Written or read by another instruction at that step.
    Access(u64),
}

/// A `wgmma.mma_async` some warps of a warpgroup have reached.
struct Rendezvous<'a> {
    inst: &'a Inst,
    /// Warps (by key) that have.
    arrived: Vec<usize>,
}

pub(super) struct State<'a> {
    /// The kernel has warpgroup MMAs.
    pub(super) enabled: bool,
    /// The values whose last access the fence rule follows: the accumulator
    /// inputs of the MMAs, and what flows into them (by value id).
    web: Vec<bool>,
    origin: HashMap<(ValueId, usize), Origin>,
    /// Each warp's step at its last `wgmma.fence`.
    fence_at: Vec<Option<u64>>,
    /// Each warp's uncommitted MMAs and committed groups (by operation id).
    uncommitted: Vec<Vec<usize>>,
    groups: Vec<VecDeque<Vec<usize>>>,
    /// The results of MMAs in flight, by (value, thread): what the next MMA
    /// of a chain reads as its C.
    bits: HashMap<(ValueId, usize), u64>,
    /// The MMAs being issued, by (block, warpgroup).
    pending: HashMap<(usize, usize), Rendezvous<'a>>,
}

impl<'a> State<'a> {
    pub(super) fn new(ir: &KernelIr) -> State<'a> {
        let enabled = ir.count_ops(|op| matches!(op, Op::Wgmma { .. })) > 0;
        let mut web = vec![false; if enabled { ir.value_types.len() } else { 0 }];
        if enabled {
            let mut changed = true;
            while changed {
                changed = false;
                mark_web(&ir.body, &mut web, &mut changed);
            }
        }
        State {
            enabled,
            web,
            origin: HashMap::new(),
            fence_at: Vec::new(),
            uncommitted: Vec::new(),
            groups: Vec::new(),
            bits: HashMap::new(),
            pending: HashMap::new(),
        }
    }

    /// A launch with `warps` warps running at once.
    pub(super) fn reset(&mut self, warps: usize) {
        self.origin.clear();
        self.bits.clear();
        self.pending.clear();
        self.fence_at = vec![None; warps];
        self.uncommitted = vec![Vec::new(); warps];
        self.groups = vec![VecDeque::new(); warps];
    }

    /// A cluster starts on these warps, threads and blocks.
    pub(super) fn start_cluster(
        &mut self,
        warps: Range<usize>,
        threads: Range<usize>,
        blocks: Range<usize>,
        others_done: bool,
    ) {
        if !self.enabled {
            return;
        }
        for k in warps {
            self.fence_at[k] = None;
            self.uncommitted[k].clear();
            self.groups[k].clear();
        }
        if others_done {
            self.origin.clear();
            self.bits.clear();
        } else {
            self.origin.retain(|(_, t), _| !threads.contains(t));
            self.bits.retain(|(_, t), _| !threads.contains(t));
        }
        self.pending.retain(|(block, _), _| !blocks.contains(block));
    }
}

/// Add to `web` what flows into the values it holds (the accumulator inputs
/// of the MMAs to start with).
fn mark_web(r: &crate::ir::Region, web: &mut [bool], changed: &mut bool) {
    fn set(web: &mut [bool], v: ValueId, changed: &mut bool) {
        if !web[v.0 as usize] {
            web[v.0 as usize] = true;
            *changed = true;
        }
    }
    for inst in &r.insts {
        let mut set = |web: &mut [bool], v: ValueId| set(web, v, changed);
        let has = |web: &[bool], v: ValueId| web[v.0 as usize];
        match &inst.op {
            Op::Wgmma { kind, c, a, .. } => {
                c.iter().for_each(|&v| set(web, v));
                if kind.a_regs {
                    a.iter().for_each(|&v| set(web, v));
                }
            }
            Op::WgmmaFence { results, inputs } | Op::WgmmaWait { results, inputs, .. } => {
                for (&res, &i) in results.iter().zip(inputs) {
                    if has(web, res) {
                        set(web, i);
                    }
                }
            }
            Op::For { init, args, body, results, .. } => {
                for i in 0..args.len() {
                    if has(web, results[i]) {
                        set(web, args[i]);
                    }
                    if has(web, args[i]) {
                        set(web, init[i]);
                        set(web, body.yields[i]);
                    }
                }
            }
            Op::While { init, cond_args, body_args, body, results, .. } => {
                for i in 0..cond_args.len() {
                    if has(web, results[i]) || has(web, body_args[i]) {
                        set(web, cond_args[i]);
                    }
                    if has(web, cond_args[i]) {
                        set(web, init[i]);
                        set(web, body.yields[i]);
                    }
                }
            }
            Op::If { then_region, else_region, results, .. } => {
                for (i, &res) in results.iter().enumerate() {
                    if has(web, res) {
                        set(web, then_region.yields[i]);
                        set(web, else_region.yields[i]);
                    }
                }
            }
            _ => {}
        }
        for region in inst.op.regions() {
            mark_web(region, web, changed);
        }
    }
}

/// The fields of a 64-bit shared-memory matrix descriptor (PTX ISA, "Matrix
/// Descriptor Format" of the asynchronous warpgroup MMAs): the start
/// address (bits 0-13), the leading dimension byte offset (16-29) and the
/// stride dimension byte offset (32-45), each in 16-byte units; the matrix
/// base offset (49-51); the swizzle mode (62-63: 0 none, 1 128-byte, 2
/// 64-byte, 3 32-byte).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Descriptor {
    pub start: u64,
    pub lbo: u64,
    pub sbo: u64,
    pub base_offset: u64,
    /// The swizzle's bits B (of `Swizzle<B,4,3>` on byte addresses): 0, 1
    /// (32 bytes), 2 (64) or 3 (128).
    pub swizzle_bits: u32,
}

/// Decode a matrix descriptor.
pub fn decode_descriptor(desc: u64) -> Descriptor {
    let field = |at: u32| ((desc >> at) & 0x3fff) << 4;
    let swizzle_bits = match desc >> 62 {
        0 => 0,
        1 => 3,
        2 => 2,
        _ => 1,
    };
    Descriptor { start: field(0), lbo: field(16), sbo: field(32), base_offset: (desc >> 49) & 7, swizzle_bits }
}

/// The byte offset from the start address of element `(mn, k)` (row `mn` of
/// the M or N extent, column `k` of the 16 along K) of an operand of
/// `bytes`-byte elements, before the swizzle, in the PTX ISA's canonical
/// layouts (in units of elements, `T` of them per 16 bytes and `W` 16-byte
/// units per swizzle row: 1, 2, 4 or 8):
///
/// * K-major, no swizzle: `((8, m), (T, 2k)) : ((1T, SBO), (1, LBO))`;
/// * K-major, swizzled: `((8, m), (T, 2k)) : ((WT, SBO), (1, T))`;
/// * MN-major, no swizzle: `((T, 1, m), (8, k)) : ((1, T, SBO), (1T, LBO))`;
/// * MN-major, swizzled: `((T, W, m), (8, k)) : ((1, T, LBO), (WT, SBO))`.
pub fn canonical_offset(d: &Descriptor, mn_major: bool, bytes: u64, mn: u64, k: u64) -> u64 {
    let t = 16 / bytes;
    let w = 1u64 << d.swizzle_bits;
    match (mn_major, d.swizzle_bits) {
        (false, 0) => (mn / 8) * d.sbo + (mn % 8) * 16 + (k / t) * d.lbo + (k % t) * bytes,
        (false, _) => (mn / 8) * d.sbo + (mn % 8) * 16 * w + (k / t) * 16 + (k % t) * bytes,
        (true, 0) => (mn / t) * d.sbo + (mn % t) * bytes + (k % 8) * 16 + (k / 8) * d.lbo,
        (true, _) => {
            let u = mn / t;
            (u / w) * d.lbo + (u % w) * 16 + (mn % t) * bytes + (k % 8) * 16 * w + (k / 8) * d.sbo
        }
    }
}

/// The shared-memory address the swizzle `Swizzle<B,4,3>` maps byte address
/// `addr` to: bits 7 to `7 + B` are XORed into bits 4 to `4 + B` (the
/// 16-byte unit within each run of 8 rows).
pub fn swizzle_address(addr: u64, swizzle_bits: u32) -> u64 {
    addr ^ (((addr >> 7) & ((1 << swizzle_bits) - 1)) << 4)
}

/// The row (M) and column (K) of element `i` of lane `lane` of warp `q` in
/// the register A fragment of a warpgroup MMA on `bits`-bit operands (PTX
/// ISA, the register fragments of `wgmma` `m64nNk16` and `m64nNk32`): warp
/// `q` holds rows `16q` to `16q + 15` as `mma.m16n8k16` (16-bit) or
/// `mma.m16n8k32` (8-bit) does. 16-bit: row `16q + lane / 4 + 8 ((i / 2) %
/// 2)`, column `2 (lane % 4) + i % 2 + 8 (i / 4)`; 8-bit: row `16q + lane /
/// 4 + 8 ((i / 4) % 2)`, column `4 (lane % 4) + i % 4 + 16 (i / 8)`.
pub fn a_coord(q: usize, lane: usize, i: usize, bits: usize) -> (usize, usize) {
    let (g, t) = (lane / 4, lane % 4);
    if bits == 16 {
        (16 * q + g + 8 * ((i / 2) % 2), 2 * t + i % 2 + 8 * (i / 4))
    } else {
        (16 * q + g + 8 * ((i / 4) % 2), 4 * t + i % 4 + 16 * (i / 8))
    }
}

/// The row (M) and column (N) of accumulator `i` of lane `lane` of warp `q`
/// of the warpgroup (PTX ISA, the `m64nNk16` D fragment): row `16q + lane /
/// 4 + 8 ((i / 2) % 2)`, column `8 (i / 4) + 2 (lane % 4) + i % 2`.
pub fn d_coord(q: usize, lane: usize, i: usize) -> (usize, usize) {
    (16 * q + lane / 4 + 8 * ((i / 2) % 2), 8 * (i / 4) + 2 * (lane % 4) + i % 2)
}

impl<'a> Machine<'a> {
    /// `wgmma.mma_async` by the warps of `group`: each joins its
    /// warpgroup's rendezvous at `inst` and waits; the fourth executes it.
    pub(super) fn wgmma_issue(&mut self, inst: &'a Inst, group: &[usize]) -> R {
        let (w, tpb) = (self.warps_per_block, self.threads_per_block);
        let mut full = Vec::new();
        for &k in group {
            let lanes = self.warp_lanes(k).to_vec();
            let (block, warp) = (k / w, k % w);
            if 128 * (warp / 4 + 1) > tpb {
                return Err(self.err(
                    format!(
                        "warp {warp} issues a wgmma.mma_async, but its warpgroup {} is not complete: the block has \
                         {tpb} threads, and a warpgroup MMA runs on a whole warpgroup of 128",
                        warp / 4
                    ),
                    Some(lanes[0]),
                ));
            }
            if lanes.len() != 32 {
                return Err(self.err(
                    format!(
                        "wgmma.mma_async executed by {} of the 32 lanes of warp {warp}: all 128 threads of a \
                         warpgroup issue it together (.sync.aligned), outside divergent code",
                        lanes.len()
                    ),
                    Some(lanes[0]),
                ));
            }
            let wg = warp / 4;
            let p = self.wgmma.pending.entry((block, wg)).or_insert(Rendezvous { inst, arrived: Vec::new() });
            if !std::ptr::eq(p.inst, inst) {
                let (at, others) = (Self::at(p.inst), warp_list(&p.arrived.iter().map(|o| o % w).collect::<Vec<_>>()));
                return Err(self.err(
                    format!(
                        "warp {warp} issues a wgmma.mma_async while warps {others} of warpgroup {wg} issue the one at \
                         {at}: all four warps of a warpgroup issue the same one"
                    ),
                    Some(lanes[0]),
                ));
            }
            if p.arrived.contains(&k) {
                return Err(self.err(
                    format!("warp {warp} issues a wgmma.mma_async again before the rest of warpgroup {wg} has"),
                    Some(lanes[0]),
                ));
            }
            p.arrived.push(k);
            if p.arrived.len() == 4 {
                full.push((block, wg));
            }
            self.advance(k);
            self.warps[k].status = Status::Blocked(Wait::Wgmma { block, group: wg, inst });
        }
        self.progress();
        for (block, wg) in full {
            let p = self.wgmma.pending.remove(&(block, wg)).expect("a rendezvous");
            self.wgmma_execute(p.inst, block, wg)?;
            let round = self.round;
            for k in p.arrived {
                let s = &mut self.warps[k];
                s.status = Status::Runnable;
                s.ready_round = round + 1;
            }
        }
        Ok(())
    }

    /// Execute the warpgroup MMA `inst` of warpgroup `wg` of `block`.
    fn wgmma_execute(&mut self, inst: &'a Inst, block: usize, wg: usize) -> R {
        let Op::Wgmma { kind, d, c, a: a_operand, desc_b, scale_d } = &inst.op else { unreachable!("a wgmma") };
        self.loc = Some(inst.loc);
        let (w, tpb) = (self.warps_per_block, self.threads_per_block);
        let warps: Vec<usize> = (0..4).map(|q| block * w + 4 * wg + q).collect();
        let first = block * tpb + 128 * wg;
        let threads = first..first + 128;
        // Warpgroup-uniform operands.
        let mut uniform = vec![(*desc_b, "B descriptor"), (*scale_d, "scale-d")];
        if !kind.a_regs {
            uniform.insert(0, (a_operand[0], "A descriptor"));
        }
        for (v, what) in uniform {
            let x = self.get(v, first);
            if let Some(t) = threads.clone().find(|&t| self.get(v, t) != x) {
                return Err(self.err(
                    format!(
                        "wgmma.mma_async: its {what} is {:#x} in thread {} but {x:#x} in thread {} of warpgroup {wg}: \
                         the descriptors and scale-d of a warpgroup MMA are the same in all its threads",
                        self.get(v, t),
                        t % tpb,
                        first % tpb
                    ),
                    Some(t),
                ));
            }
        }
        // The fence rule.
        for (q, &k) in warps.iter().enumerate() {
            let fence = self.wgmma.fence_at[k];
            let warp = k % w;
            let Some(fence) = fence else {
                return Err(self.err(
                    format!(
                        "warp {warp} issues a wgmma.mma_async without a warpgroup::fence (wgmma.fence) before it: \
                         the PTX ISA requires one before a warpgroup's first wgmma.mma_async"
                    ),
                    Some(first + 32 * q),
                ));
            };
            for t in first + 32 * q..first + 32 * q + 32 {
                for &v in c {
                    let ok = match self.wgmma.origin.get(&(v, t)).copied().unwrap_or(Origin::Access(0)) {
                        Origin::Mma { kind: k2, at } => k2 == *kind || at < fence,
                        Origin::Access(at) => at < fence,
                    };
                    if !ok {
                        return Err(self.err(
                            format!(
                                "warp {warp} issues a wgmma.mma_async whose accumulators an instruction wrote or read \
                                 after the warp's last warpgroup::fence (wgmma.fence): the PTX ISA requires a \
                                 wgmma.fence between such an access and the MMA"
                            ),
                            Some(t),
                        ));
                    }
                }
                // The register A fragment: written before the fence too.
                for &v in a_operand.iter().filter(|_| kind.a_regs) {
                    let at = match self.wgmma.origin.get(&(v, t)).copied().unwrap_or(Origin::Access(0)) {
                        Origin::Mma { at, .. } | Origin::Access(at) => at,
                    };
                    if at >= fence {
                        return Err(self.err(
                            format!(
                                "warp {warp} issues a wgmma.mma_async whose A fragment registers an instruction wrote \
                                 or read after the warp's last warpgroup::fence (wgmma.fence): the PTX ISA requires a \
                                 wgmma.fence between such an access and the MMA (fill the A fragment before the fence)"
                            ),
                            Some(t),
                        ));
                    }
                }
            }
        }
        // The operands.
        let kk_count = kind.k();
        let (a, mut ranges) = if kind.a_regs {
            // Each thread's fragment, placed by the PTX ISA's A layout.
            let mut a = vec![0f64; 64 * kk_count];
            for q in 0..4 {
                for lane in 0..32 {
                    let t = first + 32 * q + lane;
                    for (i, &v) in a_operand.iter().enumerate() {
                        let (row, col) = a_coord(q, lane, i, kind.a.bits());
                        a[row * kk_count + col] = operand_f64(kind.a, self.get(v, t));
                    }
                }
            }
            (a, Vec::new())
        } else {
            let desc = self.get(a_operand[0], first);
            self.wgmma_operand(desc, 64, kind.trans_a, kind.a, block, &warps, "A")?
        };
        let (b, more) =
            self.wgmma_operand(self.get(*desc_b, first), kind.n as usize, kind.trans_b, kind.b, block, &warps, "B")?;
        ranges.extend(more);
        let ranges = merge_ranges(ranges);
        for &(buffer, ref bytes) in &ranges {
            for &k in &warps {
                let t = self.warp_thread(k);
                self.async_proxy_may_read(buffer, block, bytes.clone(), k).map_err(|why| {
                    self.err(
                        format!(
                            "wgmma.mma_async reads an operand through the async proxy, but {why}: generic writes of a \
                             warpgroup MMA's shared-memory operand need fence_view_async_shared (fence.proxy.async) \
                             before the MMA"
                        ),
                        Some(t),
                    )
                })?;
            }
        }
        let scale = self.get(*scale_d, first) != 0;
        let n = kind.n as usize;
        let now = self.clock;
        for (q, &k) in warps.iter().enumerate() {
            let mut results = Vec::with_capacity(32 * d.len());
            for lane in 0..32 {
                let t = first + 32 * q + lane;
                for (i, (&dv, &cv)) in d.iter().zip(c).enumerate() {
                    let (row, col) = d_coord(q, lane, i);
                    let c_bits = if scale {
                        Some(self.wgmma.bits.get(&(cv, t)).copied().unwrap_or_else(|| self.get(cv, t)))
                    } else {
                        None
                    };
                    let (ar, bc) = (&a[row * kk_count..][..kk_count], &b[col * kk_count..][..kk_count]);
                    let bits = if kind.is_integer() {
                        // Exact, wrapping modulo 2^32 (no .satfinite).
                        let mut acc = c_bits.map_or(0i64, |x| x as u32 as i32 as i64);
                        for (x, y) in ar.iter().zip(bc) {
                            acc += (*x as i64) * (*y as i64);
                        }
                        acc as u32 as u64
                    } else {
                        let mut acc = c_bits.map_or(0.0, |x| to_f64(kind.acc, x) as f32);
                        for (x, y) in ar.iter().zip(bc) {
                            acc += (*x as f32) * (*y as f32);
                        }
                        from_f32(kind.acc, acc)
                    };
                    results.push((dv, t, bits));
                }
            }
            debug_assert!(results.iter().all(|&(_, t, _)| (t - first) / 32 == q) && n == 2 * d.len());
            let id = self.ops.len();
            for &(v, t, bits) in &results {
                // A placeholder until the result lands (loop carries copy it).
                self.set(v, t, 0);
                self.pending_results.insert((v, t), id);
                self.wgmma.bits.insert((v, t), bits);
                if self.wgmma.web[v.0 as usize] {
                    self.wgmma.origin.insert((v, t), Origin::Mma { kind: *kind, at: now });
                }
            }
            let reads = ranges.iter().map(|(buffer, bytes)| AsyncRead { buffer: *buffer, block, bytes: bytes.clone() });
            self.ops.push(AsyncOp {
                thread: self.warp_thread(k),
                agent: self.agent_slot(block),
                writes: Vec::new(),
                reads: reads.collect(),
                global_writes: Vec::new(),
                global_reduce: None,
                results,
                trigger: Trigger::Wgmma,
                done: None,
            });
            self.any_async_reads = true;
            self.wgmma.uncommitted[k].push(id);
        }
        self.report.mma_instructions += 1;
        Ok(())
    }

    /// The first thread of warp `k`.
    fn warp_thread(&self, k: usize) -> usize {
        let w = self.warps_per_block;
        (k / w) * self.threads_per_block + (k % w) * 32
    }

    /// Read a `rows x K` operand (A: 64 rows of M, B: N rows; K of 32
    /// bytes) of `ty` elements through descriptor `desc`, MN-major when
    /// `mn_major`: its values, row-major (`[row * K + k]`), and the bytes
    /// read. Each byte
    /// must be written, by a write ordered before the issue in every warp of
    /// `warps`, and no asynchronous copy may still be writing it.
    #[allow(clippy::too_many_arguments)]
    fn wgmma_operand(
        &mut self,
        desc: u64,
        rows: usize,
        mn_major: bool,
        ty: DType,
        block: usize,
        warps: &[usize],
        what: &str,
    ) -> R<(Vec<f64>, Vec<ByteRange>)> {
        let d = decode_descriptor(desc);
        let t0 = self.warp_thread(warps[0]);
        if d.base_offset != 0 {
            return Err(self.err(
                format!(
                    "wgmma.mma_async: the {what} descriptor {desc:#x} has the matrix base offset {}, which the \
                     simulator does not model (cute-rs builds descriptors with 0, for operands whose swizzle \
                     pattern starts on its repeat)",
                    d.base_offset
                ),
                Some(t0),
            ));
        }
        let bytes = ty.bytes() as u64;
        let kk_count = 32 / bytes as usize;
        let mut values = vec![0f64; rows * kk_count];
        let mut ranges = Vec::new();
        for row in 0..rows {
            for k in 0..kk_count {
                let addr = swizzle_address(
                    d.start + canonical_offset(&d, mn_major, bytes, row as u64, k as u64),
                    d.swizzle_bits,
                );
                let (b, byte) = self.resolve(AddressSpace::Smem, addr).map_err(|why| {
                    self.err(
                        format!("wgmma.mma_async: element ({row}, {k}) of {what} (descriptor {desc:#x}): {why}"),
                        Some(t0),
                    )
                })?;
                let n = bytes as usize;
                if byte + n > self.buffer_bytes(b) {
                    return Err(self.err(
                        format!("wgmma.mma_async: element ({row}, {k}) of {what} runs past the end of its buffer"),
                        Some(t0),
                    ));
                }
                for &k in warps {
                    self.wgmma_may_read(b, block, byte..byte + n, k)?;
                }
                let sh = &self.shared[self.home[b.0 as usize]][block];
                let mut raw = 0u64;
                for (i, &x) in sh.data[byte..byte + n].iter().enumerate() {
                    raw |= (x as u64) << (8 * i);
                }
                values[row * kk_count + k] = operand_f64(ty, raw);
                ranges.push((b, byte..byte + n));
            }
        }
        Ok((values, ranges))
    }

    /// May warp `k`'s share of a warpgroup MMA read bytes `range` of shared
    /// buffer `b` of `block` now?
    fn wgmma_may_read(&self, b: BufferId, block: usize, range: Range<usize>, k: usize) -> R {
        let (slot, clock) = (self.slot_of(k), &self.clocks[k]);
        let t = self.warp_thread(k);
        for i in range {
            let m = &self.shared[self.home[b.0 as usize]][block].meta[i];
            let place = smem_place(self.ir.buffer(b), i);
            if m.inflight > 0 {
                return Err(self.err(
                    format!(
                        "wgmma.mma_async reads {place} while an asynchronous copy is still writing it (wait for the \
                         copy's mbarrier phase first)"
                    ),
                    Some(t),
                ));
            }
            if !m.written {
                if self.opts.detect_uninitialized {
                    return Err(self.err(format!("wgmma.mma_async reads {place}, which was never written"), Some(t)));
                }
                continue;
            }
            if self.opts.detect_races && !m.write_ordered_before(slot, clock) {
                return Err(self.err(
                    format!(
                        "data race: the wgmma.mma_async of warp {} reads {place}, but nothing orders its last write, \
                         by {}, before the MMA (a barrier, or the wait on the mbarrier of the copy that wrote it)",
                        k % self.warps_per_block,
                        self.slot_name(m.w_slot as usize, block)
                    ),
                    Some(t),
                ));
            }
        }
        Ok(())
    }

    /// `wgmma.fence`, `wgmma.commit_group` or `wgmma.wait_group` by the
    /// warps of `active`.
    pub(super) fn wgmma_inst(&mut self, op: &Op, active: &[usize]) -> R {
        let bases = self.full_warps(active, op.name())?;
        for base in bases {
            let k = self.clock_index(base);
            match op {
                Op::WgmmaFence { results, inputs } => {
                    self.wgmma.fence_at[k] = Some(self.clock);
                    self.wgmma_rebind(base, results, inputs);
                }
                Op::WgmmaCommitGroup => {
                    let group = std::mem::take(&mut self.wgmma.uncommitted[k]);
                    self.wgmma.groups[k].push_back(group);
                }
                Op::WgmmaWait { pending, results, inputs, .. } => {
                    while self.wgmma.groups[k].len() > *pending as usize {
                        let group = self.wgmma.groups[k].pop_front().expect("a group");
                        for id in group {
                            self.wgmma_complete(id, k);
                        }
                    }
                    self.wgmma_rebind(base, results, inputs);
                }
                _ => unreachable!("a wgmma instruction"),
            }
        }
        Ok(())
    }

    /// The operand fence of the warp starting at thread `base`: each result
    /// is its input (still pending if the input is).
    fn wgmma_rebind(&mut self, base: usize, results: &[ValueId], inputs: &[ValueId]) {
        for t in base..base + 32 {
            for (&r, &i) in results.iter().zip(inputs) {
                let bits = self.get(i, t);
                self.set(r, t, bits);
                self.carry_pending(i, r, t);
            }
        }
    }

    /// Complete warp `k`'s share `id` of a warpgroup MMA: its results land,
    /// and its operand reads count as the warp's reads now.
    fn wgmma_complete(&mut self, id: usize, k: usize) {
        if !self.ops[id].in_flight() {
            return;
        }
        let t = self.warp_thread(k);
        let reads = std::mem::take(&mut self.ops[id].reads);
        for r in &reads {
            self.note_bulk_source_read(t, r.buffer, r.block, r.bytes.clone());
        }
        for &(v, t, _) in &self.ops[id].results {
            if self.pending_results.get(&(v, t)) == Some(&id) {
                self.wgmma.bits.remove(&(v, t));
            }
        }
        let slot = self.slot_of(k);
        let clock = self.clocks[k][slot];
        self.complete(id, slot, clock);
    }

    /// `to` takes `from`'s value in thread `t` (a loop carry, an `if`
    /// result, an operand fence): if `from` is the pending result of a
    /// warpgroup MMA, so is `to`, and it lands with it.
    pub(super) fn carry_pending(&mut self, from: ValueId, to: ValueId, t: usize) {
        if !self.wgmma.enabled {
            // A tcgen05.ld's pending registers (m10-tmem-ld-st) carry the same way.
            if let Some(id) = self.pending_results.get(&(from, t)).copied() {
                let bits = self.ops[id].results.iter().find(|r| r.0 == from && r.1 == t).map_or(0, |r| r.2);
                self.ops[id].results.push((to, t, bits));
                self.pending_results.insert((to, t), id);
            } else if !self.pending_results.is_empty() {
                self.pending_results.remove(&(to, t));
            }
            return;
        }
        match self.pending_results.get(&(from, t)).copied() {
            Some(id) if self.ops[id].trigger != Trigger::Wgmma => {
                let bits = self.ops[id].results.iter().find(|r| r.0 == from && r.1 == t).map_or(0, |r| r.2);
                self.ops[id].results.push((to, t, bits));
                self.pending_results.insert((to, t), id);
            }
            Some(id) => {
                let bits = self.wgmma.bits.get(&(from, t)).copied().unwrap_or(0);
                self.ops[id].results.push((to, t, bits));
                self.pending_results.insert((to, t), id);
                self.wgmma.bits.insert((to, t), bits);
            }
            None => {
                if self.pending_results.remove(&(to, t)).is_some() {
                    self.wgmma.bits.remove(&(to, t));
                }
            }
        }
        if self.wgmma.web[to.0 as usize] {
            match self.wgmma.origin.get(&(from, t)).copied() {
                Some(o) => self.wgmma.origin.insert((to, t), o),
                None => self.wgmma.origin.insert((to, t), Origin::Access(self.clock)),
            };
        }
    }

    /// An instruction other than the WGMMA ones read or wrote these values in
    /// the threads of `lanes`: the fence rule's accesses.
    pub(super) fn wgmma_note_access(&mut self, op: &Op, lanes: &[usize]) {
        if passes_accumulators(op) {
            return;
        }
        let now = self.clock;
        for v in op.operands().into_iter().chain(op.results()) {
            if self.wgmma.web[v.0 as usize] {
                for &t in lanes {
                    self.wgmma.origin.insert((v, t), Origin::Access(now));
                }
            }
        }
    }

    /// For hang reports: what a warp held at a `wgmma.mma_async` waits for.
    pub(super) fn wgmma_wait_report(&self, block: usize, wg: usize, ctx: &str) -> String {
        let w = self.warps_per_block;
        let Some(p) = self.wgmma.pending.get(&(block, wg)) else { return String::new() };
        let missing: Vec<usize> = (0..4).map(|q| block * w + 4 * wg + q).filter(|k| !p.arrived.contains(k)).collect();
        format!(
            " at wgmma.mma_async ({}){ctx}, which warps {} of warpgroup {wg} have not issued (a warpgroup MMA is \
             issued by all four warps of its warpgroup): {}",
            Self::at(p.inst),
            warp_list(&missing.iter().map(|k| k % w).collect::<Vec<_>>()),
            self.whereabouts_of(&missing, false)
        )
    }
}

/// The value of an operand element of type `ty` (FP8 decoded exactly).
fn operand_f64(ty: DType, bits: u64) -> f64 {
    match ty.float_format() {
        Some(f) if ty.is_narrow_float() => crate::numeric::decode(f, bits as u32),
        _ => to_f64(ty, bits),
    }
}

/// `ranges` sorted, with overlapping or adjacent ones of a buffer merged.
fn merge_ranges(mut ranges: Vec<ByteRange>) -> Vec<ByteRange> {
    ranges.sort_by_key(|(b, r)| (b.0, r.start));
    let mut out: Vec<ByteRange> = Vec::new();
    for (b, r) in ranges {
        match out.last_mut() {
            Some((lb, lr)) if *lb == b && r.start <= lr.end => lr.end = lr.end.max(r.end),
            _ => out.push((b, r)),
        }
    }
    out
}
