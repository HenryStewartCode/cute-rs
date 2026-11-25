// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's tcgen05 MMAs and commits,
//! modelled from the PTX ISA (9.7.18.10 `tcgen05.mma`, its data-path
//! layouts in 9.7.18.10.5, the instruction descriptor in 9.7.18.4.2 and
//! `tcgen05.commit` in 9.7.18.12) independently of cute-rs's layout algebra.
//!
//! * **Issue.** `cute.gemm` is called by a whole warp, which elects one lane
//!   to issue each `tcgen05.mma`. The simulator requires the warp to be
//!   converged (all 32 lanes) and the operands (D's TMEM address, both
//!   descriptors, `enable-input-d`) to be the same in every lane, and then
//!   executes the instruction once: one warp-level call is one MMA of the
//!   CTA's tensor-core pipe, whose MMAs run in issue order.
//! * **Operands.** Both shared-memory descriptors are decoded with
//!   [`UmmaOperandMap`] (the PTX ISA's canonical layouts and swizzles). Each
//!   byte read must have been written, by a write ordered before the issue
//!   in the issuing warp; generic-proxy writes need a `fence.proxy.async`
//!   since (m7's proxy model); no asynchronous copy may still be writing it.
//! * **D.** The accumulator is written into TMEM by the PTX ISA's data-path
//!   layouts for `cta_group::1`: Layout D for M = 128 (row `i` in TMEM lane
//!   `i`) and Layout F for M = 64 (rows `16 q` to `16 q + 15` in lanes `32 q`
//!   to `32 q + 15`), column `j` of D in column `j` after the address. Each
//!   element takes one 32-bit column; a 16-bit accumulator takes its low
//!   half (the simulator writes zero to the high half, which code should not
//!   rely on). D's columns must be in a live allocation, and its lane field
//!   0.
//! * **Arithmetic.** As the WGMMA model: each element starts from D (when
//!   `enable-input-d`) or 0 and adds the K products in order of K in
//!   Float32 (products of 16-bit floats are exact), then rounds once to the
//!   accumulator type. Small-integer data is bit-exact.
//! * **Other kinds**. `.kind::tf32` reads 32-bit
//!   operands and drops their low 13 mantissa bits (as warp `mma.sync`
//!   does, `tf32_truncate`); `.kind::i8` multiplies Int8/Uint8 operands and
//!   adds into the Int32 accumulator exactly (wrapping); `.kind::f8f6f4`
//!   decodes all five formats exactly with the narrow-float codecs. FP6 and
//!   FP4 operands are read from the padded image: the descriptor addresses
//!   them as bytes, and value `i` of each 16-byte slot (`i` the address
//!   modulo 16) is bits `w i` to `w i + w - 1` of the slot (`w` 6 or 4,
//!   PTX's `.b6x16_p32` and `.b4x16_p64`), of which only the bytes holding
//!   those bits are read.
//! * **Completion.** An MMA is in flight from its issue until a thread
//!   observes its completion: its `tcgen05.commit`'s mbarrier phase
//!   completes and a thread waits on it (or, with random completion, at a
//!   random point before that). Until then its operand bytes are being read
//!   (overwriting them is reported) and D's cells are being written (a
//!   `tcgen05.ld` or `tcgen05.st` of them, or freeing them, is reported).
//!   When it lands its cells become the CTA tensor-core agent's writes,
//!   which a warp may read only after waiting on the commit's mbarrier.
//!   Later MMAs of the same CTA may read D at once (the pipe is in order).
//! * **Commit.** `tcgen05.commit` by a thread owes one arrival to the
//!   mbarrier (or, with a mask, to the mbarrier at the same offset of each
//!   CTA named), made once the MMAs its warp issued before have completed
//!   (the same deferred arrival as `cp.async.mbarrier.arrive.noinc`; the
//!   elected lane that issued the MMAs is the committing thread's warp's).
//!   It is per thread: an unelected commit arrives once per lane.
//! * **Sparse**. `tcgen05.mma.sp` reads A compressed
//!   (half of K) through its descriptor and expands each row with the
//!   metadata read from TMEM at `[sp-meta-tmem]` plus the descriptor's
//!   sparse selector (bits 0-1) in columns: each group of four K elements
//!   takes a 4-bit field, the first kept position in bits 0-1 and the
//!   second, higher, in bits 2-3; a field naming no two increasing
//!   positions is reported. The placement is cute-rs's reading of the PTX
//!   ISA's metadata layouts for 128 rows per CTA, written as a formula
//!   here ([`sparse_meta_place`]) and not from the library's layouts; the
//!   ISA figures could not be checked when this was written.
//! * **Prims forms**. An MMA from `cute::prims` is
//!   issued by each thread that executes it (inside `elect_one`), with no
//!   convergence or uniformity rule. `.ws` (weight stationary) writes D by
//!   the data-path layouts of CuTe C++'s `UMMA::tmem_frg_ws` (Layout G for
//!   M = 32: row `i` in lane `i + 32 (j / (N/4))`, column `j % (N/4)`;
//!   Layout E for M = 64: lane `i + 64 (j / (N/2))`, column `j % (N/2)`;
//!   Layout D for M = 128), and reads a `.ws.sp` MMA's metadata for its M
//!   rows from every `128 / M`-lane replica (CuTe C++'s `tmem_e_frg_ws`),
//!   which must agree. B's four collector buffers per CTA (`b0` to `b3`)
//!   and the plain MMA's A collector follow the Rubin rules above; with
//!   [`SimOptions::collector_source_check`](crate::sim::SimOptions) off a
//!   collector keeps its tile whatever happens to the shared memory since.
//!   `.ashift` shifts A's TMEM rows up by one within each 32-lane
//!   sub-partition (lane `r` takes lane `r + 1`'s bits unless `r % 32` is
//!   31) once the MMA has read A; `scale-input-d` scales the D read by
//!   `2^-s`. A nonzero zero-column mask and a nonzero max-shift field are
//!   reported as not modelled (their formats could not be checked). The
//!   block-scaled sparse kinds expand A as the sparse kinds do, FP4 by
//!   pairs (a metadata field per eight K elements naming two of four
//!   pairs), then scale as the block-scaled kinds do.

use std::ops::Range;

use super::tmem::{TMEM_COLUMNS, TmemWriter};
use super::{AsyncOp, AsyncRead, DeferredArrival, Machine, R, Trigger, from_f32, smem_place, tf32_truncate, to_f64};
use crate::ir::{BufferId, CollectorOp, DType, Inst, Loc, Op, UmmaDataKind, UmmaKind};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::tcgen05::{UmmaDescEncoding, UmmaOperandMap};
use crate::tensor::AddressSpace;

/// TMEM cells as lane ranges and a column range.
pub(super) type TmemCells = (Vec<Range<usize>>, Range<usize>);

/// What a collector buffer holds after a `FILL`: the
/// descriptor and form of the tile it was filled from, the tile's values,
/// the shared-memory bytes they were read from (to report a later change)
/// and where the filling MMA was issued.
pub(super) struct Collector {
    desc: u64,
    form: (DType, usize, usize, bool),
    values: Vec<f64>,
    bytes: Vec<(BufferId, usize, Vec<u8>)>,
    loc: Loc,
}

/// An MMA in flight: its asynchronous operation, the CTA, the issuing warp
/// (by key), D's cells and where it was issued.
pub(super) struct UmmaFlight {
    pub id: usize,
    pub block: usize,
    pub warp: usize,
    pub lanes: Vec<Range<usize>>,
    pub cols: Range<usize>,
    pub loc: Loc,
    /// The TMEM cells of an A read from tensor memory (its
    /// lanes and columns), being read until the MMA completes;
    /// And those of the scale factors.
    pub a_cells: Vec<TmemCells>,
}

impl UmmaFlight {
    pub(super) fn covers(&self, block: usize, lane: usize, col: usize) -> bool {
        self.block == block && self.cols.contains(&col) && self.lanes.iter().any(|r| r.contains(&lane))
    }

    /// Whether the MMA reads TMEM cell (`lane`, `col`) of `block` as A.
    pub(super) fn reads_a(&self, block: usize, lane: usize, col: usize) -> bool {
        self.block == block
            && self.a_cells.iter().any(|(lanes, cols)| cols.contains(&col) && lanes.iter().any(|r| r.contains(&lane)))
    }
}

/// The TMEM lane of row `row` of D for an M of `m` with `cta_group::1`
/// (PTX ISA 9.7.18.10.5): Layout D (M = 128) puts row `i` in lane `i`;
/// Layout F (M = 64) puts each 16 rows in the first half of a warp's
/// 32-lane sub-partition.
pub(super) fn d_lane(m: u32, row: usize) -> usize {
    if m == 64 { 32 * (row / 16) + row % 16 } else { row }
}

/// The lanes D occupies in each CTA, as ranges (all 128 for a CTA pair
/// and for a `.ws` MMA).
fn d_lanes(kind: UmmaKind) -> Vec<Range<usize>> {
    if kind.prims.ws {
        return std::iter::once(0..128).collect();
    }
    match kind.m {
        64 => (0..4).map(|q| 32 * q..32 * q + 16).collect(),
        _ if kind.two_cta => std::iter::once(0..128).collect(),
        m => std::iter::once(0..m as usize).collect(),
    }
}

/// The TMEM columns D occupies in each CTA: N, or N / 2 for a CTA pair's
/// M = 128 (Layout B).
fn d_columns(kind: UmmaKind) -> usize {
    if kind.prims.ws {
        // Layouts G, E and D.
        return kind.n as usize * kind.m as usize / 128;
    }
    if kind.two_cta && kind.m == 128 { kind.n as usize / 2 } else { kind.n as usize }
}

/// Where element (`row`, `col`) of D lands (PTX ISA 9.7.18.10.5): the CTA of
/// the pair (0 for `cta_group::1`), the TMEM lane and the column after D's.
/// With `cta_group::2` CTA `i` holds rows `i M / 2` onwards: for M = 256
/// (Layout A) row `r` of its half in lane `r` and column `j` in column `j`;
/// for M = 128 (Layout B) the first `N / 2` columns of its 64 rows in lanes
/// 0 to 63 and the others in lanes 64 to 127, in `N / 2` columns.
pub(super) fn d_place(kind: UmmaKind, row: usize, col: usize) -> (usize, usize, usize) {
    if kind.prims.ws {
        // Layout G (M = 32, 1x4), E (M = 64, 2x2) or D (M = 128).
        let w = d_columns(kind);
        return (0, row + kind.m as usize * (col / w), col % w);
    }
    if !kind.two_cta {
        return (0, d_lane(kind.m, row), col);
    }
    let half = kind.m as usize / 2;
    let (cta, r) = (row / half, row % half);
    if kind.m == 256 {
        (cta, r, col)
    } else {
        let nh = kind.n as usize / 2;
        (cta, r + 64 * (col / nh), col % nh)
    }
}

/// Where the 4-bit metadata field of K group `g` (four
/// logical K elements) of row `row` (of one CTA's 128) of a sparse MMA
/// is in tensor memory: `(lane, column after the metadata's, bit)`.
///
/// For `.kind::f16` (K = 32, eight groups per row, one column per MMA)
/// each lane holds two rows eight apart, half of K each: rows `16 q + r`
/// and `16 q + 8 + r` (`r` < 8) in lane `16 q + 8 h + r` for groups `4 h`
/// to `4 h + 3`, the first row in bits 0-15 and the second in bits 16-31,
/// the groups in order of K in 4-bit steps. For the 8-bit kinds (K = 64,
/// sixteen groups, two columns per MMA) row `i` is in lane `i`, its groups
/// in order of K through the two columns. (cute-rs's model of the PTX
/// ISA's 128-row metadata layouts.)
pub(crate) fn sparse_meta_place(eight_bit: bool, row: usize, g: usize) -> (usize, usize, usize) {
    if eight_bit {
        (row, g / 8, 4 * (g % 8))
    } else {
        let (r, half, q) = (row % 8, row / 8 % 2, row / 16);
        (16 * q + 8 * (g / 4) + r, 0, 4 * (g % 4) + 16 * half)
    }
}

fn at(loc: Loc) -> String {
    format!("{}:{}", loc.file(), loc.line())
}

impl<'a> Machine<'a> {
    /// `tcgen05.mma` issued by the warps of `active`.
    pub(super) fn umma_issue(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        let Op::Tcgen05Mma { kind, taddr, desc_a, desc_b, enable_input_d, idesc, disable_lane, sf, meta, .. } =
            &inst.op
        else {
            unreachable!("a tcgen05.mma")
        };
        let kind = *kind;
        let w = self.warps_per_block;
        self.progress();
        // A prims MMA is issued by each thread that executes it.
        if kind.prims.thread_issue {
            for &t in active {
                let k = self.clock_index(t);
                let block = k / w;
                self.tcgen05_group(&format!("tcgen05.mma.kind::{}", kind.kind.ptx()), kind.two_cta, inst.loc, t)?;
                if kind.two_cta && (!self.cluster_size.is_multiple_of(2) || block % 2 == 1) {
                    return Err(self.err(
                        format!(
                            "{} issued by a thread of CTA {} of its cluster of {}: a CTA pair's MMA is issued by the \
                             pair's even CTA",
                            kind.ptx(),
                            block % self.cluster_size,
                            self.cluster_size
                        ),
                        Some(t),
                    ));
                }
                self.umma_execute(inst, kind, t, k, block)?;
            }
            return Ok(());
        }
        for (base, lanes) in self.warps_of(active) {
            let k = self.clock_index(base);
            let block = k / w;
            if lanes.len() != 32 {
                return Err(self.err(
                    format!(
                        "tcgen05.mma (cute.gemm) executed by {} of the 32 lanes of warp {}: gemm is called by a whole, \
                         converged warp, which elects the lane that issues the MMA (do not call it in divergent code \
                         or inside elect_one)",
                        lanes.len(),
                        k % w
                    ),
                    Some(base + lanes[0]),
                ));
            }
            let mut uniform = vec![
                (*taddr, "TMEM address of D"),
                (*desc_a, if kind.a_tmem { "TMEM address of A" } else { "A descriptor" }),
                (*desc_b, "B descriptor"),
                (*enable_input_d, "enable-input-d (Field.ACCUMULATE)"),
            ];
            // The runtime descriptor and the lane mask.
            uniform.extend(idesc.iter().map(|&v| (v, "instruction descriptor (Field.NEGATE_A/NEGATE_B)")));
            uniform.extend(disable_lane.iter().map(|&v| (v, "disable-output-lane mask")));
            // The scale factors' TMEM addresses.
            if let Some((a, b)) = sf {
                uniform.extend([(*a, "TMEM address of SFA"), (*b, "TMEM address of SFB")]);
            }
            // m10-sparse-umma.
            if let Some(e) = meta {
                uniform.push((*e, "TMEM address of the sparsity metadata"));
            }
            for (v, what) in uniform {
                let x = self.get(v, base);
                if let Some(l) = (1..32).find(|&l| self.get(v, base + l) != x) {
                    return Err(self.err(
                        format!(
                            "tcgen05.mma: its {what} is {:#x} in lane {l} of warp {} but {x:#x} in lane 0: the operands \
                             of an MMA are the same in every lane of the issuing warp",
                            self.get(v, base + l),
                            k % w
                        ),
                        Some(base + l),
                    ));
                }
            }
            self.tcgen05_group(&format!("tcgen05.mma.kind::{}", kind.kind.ptx()), kind.two_cta, inst.loc, base)?;
            // The even CTA of a cluster pair issues the pair's MMAs.
            if kind.two_cta && (!self.cluster_size.is_multiple_of(2) || block % 2 == 1) {
                return Err(self.err(
                    if self.cluster_size.is_multiple_of(2) {
                        format!(
                            "{} issued by warp {} of CTA {} of its cluster, an odd CTA: a CTA pair's MMA is issued by \
                             the pair's even CTA (%cluster_ctarank even), whose instruction reads both CTAs' operands \
                             and writes both CTAs' TMEM; guard gemm with the CTA's rank in its pair",
                            kind.ptx(),
                            k % w,
                            block % self.cluster_size
                        )
                    } else {
                        format!(
                            "{} in a cluster of {} CTA(s): a CTA pair is two CTAs of a cluster with an even number \
                             of CTAs",
                            kind.ptx(),
                            self.cluster_size
                        )
                    },
                    Some(base),
                ));
            }
            self.umma_execute(inst, kind, base, k, block)?;
        }
        Ok(())
    }

    fn umma_execute(&mut self, inst: &'a Inst, kind: UmmaKind, base: usize, k: usize, block: usize) -> R {
        let Op::Tcgen05Mma {
            taddr, desc_a, desc_b, enable_input_d, idesc, disable_lane, sf, meta, zero_col_mask, ..
        } = &inst.op
        else {
            unreachable!()
        };
        let (m, n, kk) = (kind.m as usize, kind.n as usize, kind.k());
        let addr = self.get(*taddr, base) as u32;
        let (lane0, col0) = ((addr >> 16) as usize, (addr & 0xffff) as usize);
        let name = kind.ptx();
        // What cute-rs does not model of the .ws forms.
        if let Some(z) = zero_col_mask {
            let z = self.get(*z, base);
            if z != 0 {
                return Err(self.err(
                    format!(
                        "{name}: zero-column mask {z:#018x}: the simulator models only the mask 0 (no column of B \
                         zeroed); the descriptor's format could not be checked against the PTX ISA"
                    ),
                    Some(base),
                ));
            }
        }
        if kind.prims.max_shift != 0 {
            return Err(self.err(
                format!(
                    "{name}: the instruction descriptor's max-shift field is {}: the simulator does not model .ws B \
                     shifting (leave it 0)",
                    kind.prims.max_shift
                ),
                Some(base),
            ));
        }
        if lane0 != 0 {
            return Err(self
                .err(format!("{name}: D's TMEM address {addr:#x} has lane {lane0}; D starts at lane 0"), Some(base)));
        }
        // A pair's MMA writes D into both CTAs' TMEM, at the same address.
        let pair: Vec<usize> = if kind.two_cta { vec![block, block + 1] } else { vec![block] };
        let cols = col0..col0 + d_columns(kind);
        for &blk in &pair {
            let allocs = self.tmem_allocs(blk);
            if cols.end > TMEM_COLUMNS
                || !allocs.iter().any(|&(a, c)| a as usize <= cols.start && cols.end <= (a + c) as usize)
            {
                return Err(self.err(
                    format!(
                        "{name}: D (columns {} to {} at TMEM address {addr:#x}) is not in a live TMEM allocation of \
                         {} (live allocations: {})",
                        cols.start,
                        cols.end - 1,
                        if blk == block { "this CTA".to_string() } else { "its peer CTA".to_string() },
                        self.tmem_live(blk)
                    ),
                    Some(base),
                ));
            }
            if let Some((warp, verb, wait, loc)) = self.tmem_ldst_in_flight(blk, cols.start as u32..cols.end as u32) {
                return Err(self.err(
                    format!(
                        "{name} writes TMEM columns {} to {} while the tcgen05.{verb} of warp {} ({}) of them has not \
                         completed ({wait}, then a barrier with the tcgen05 fences, before the MMA)",
                        cols.start,
                        cols.end - 1,
                        self.slot_name(self.slot_of(warp), block),
                        at(loc)
                    ),
                    Some(base),
                ));
            }
        }
        // The operands: with a pair, each CTA holds half of A's rows and half of B's at the
        // descriptors' addresses in its own shared memory.
        let v = pair.len();
        let major = |t: bool| if t { OperandMajorMode::MN } else { OperandMajorMode::K };
        // A Rubin MMA's descriptors have the sm_107 encoding.
        let enc = if kind.sm107.is_some() { UmmaDescEncoding::Sm107 } else { UmmaDescEncoding::Sm100 };
        let mut ranges: Vec<(usize, BufferId, Range<usize>)> = Vec::new();
        // The plain MMA's A collector.
        let a_op = kind.sm107.map(|r| r.collector_a).unwrap_or(kind.prims.collector_a);
        let (mut a, a_cells) = if kind.a_tmem {
            let a_addr = self.get(*desc_a, base) as u32;
            let (a, cells) = self.umma_tmem_a(kind, a_addr, k, block, base)?;
            // A collected from TMEM: the collector is keyed by A's address (bit 63 set).
            let form = (kind.a, m, kind.a_k(), false);
            let key = 1u64 << 63 | a_addr as u64;
            let a = match self.umma_collected(a_op, 0, key, form, block, base)? {
                Some(values) => values,
                None => {
                    self.umma_fill(inst, a_op, 0, key, form, block, &a, &[]);
                    a
                }
            };
            (a, vec![cells])
        } else {
            let mut a = Vec::with_capacity(m * kk);
            let op = a_op;
            for &blk in &pair {
                let mut r = Vec::new();
                let desc = self.get(*desc_a, base);
                let form = (kind.a, m / v, kind.a_k(), kind.trans_a);
                let part = match self.umma_collected(op, 0, desc, form, blk, base)? {
                    Some(values) => values,
                    None => {
                        let part = self.umma_operand(
                            desc,
                            major(kind.trans_a),
                            kind.kind,
                            kind.a,
                            m / v,
                            kind.a_k(),
                            k,
                            blk,
                            "A",
                            &mut r,
                            enc,
                        )?;
                        self.umma_fill(inst, op, 0, desc, form, blk, &part, &r);
                        part
                    }
                };
                a.extend(part);
                ranges.extend(r.into_iter().map(|(b, bytes)| (blk, b, bytes)));
            }
            (a, Vec::new())
        };
        let mut a_cells = a_cells;
        let mut b = Vec::with_capacity(n * kk);
        // A .ws MMA's B buffer b<i> is collector slot 1 + i; the Rubin
        // (sm_107) B collector is slot 5.
        let (op, slot) = match kind.prims.collector_b {
            Some((i, op)) => (op, 1 + i),
            None => (kind.sm107.map(|r| r.collector_b).unwrap_or_default(), 5),
        };
        for &blk in &pair {
            let mut r = Vec::new();
            let desc = self.get(*desc_b, base);
            let form = (kind.b, n / v, kk, kind.trans_b);
            let part = match self.umma_collected(op, slot, desc, form, blk, base)? {
                Some(values) => values,
                None => {
                    let part = self.umma_operand(
                        desc,
                        major(kind.trans_b),
                        kind.kind,
                        kind.b,
                        n / v,
                        kk,
                        k,
                        blk,
                        "B",
                        &mut r,
                        enc,
                    )?;
                    self.umma_fill(inst, op, slot, desc, form, blk, &part, &r);
                    part
                }
            };
            b.extend(part);
            ranges.extend(r.into_iter().map(|(bf, bytes)| (blk, bf, bytes)));
        }
        // The runtime negate bits of the descriptor.
        let desc_bits = idesc.map(|v| self.get(v, base) as u32).unwrap_or_else(|| kind.idesc());
        // And the scale-factor IDs.
        let sf_ids = if kind.scale.is_some() { 3 << UmmaKind::A_SF_ID_BIT | 3 << UmmaKind::B_SF_ID_BIT } else { 0 };
        // And the sparse selector.
        let sf_ids = sf_ids | if kind.sparse { UmmaKind::SPARSE_ID2_MASK } else { 0 };
        if desc_bits & !(1 << UmmaKind::NEGATE_A_BIT | 1 << UmmaKind::NEGATE_B_BIT | sf_ids) != kind.idesc() {
            return Err(self.err(
                format!(
                    "{name}: the runtime instruction descriptor {desc_bits:#010x} differs from the instruction's \
                     {:#010x} in more than the negate bits{}",
                    kind.idesc(),
                    if sf_ids != 0 { " and scale-factor IDs or sparse selector" } else { "" }
                ),
                Some(base),
            ));
        }
        // The scale factors, per CTA: SFA for the CTA's rows of A, SFB
        // for all N columns of B (each CTA of a pair reads its own TMEM).
        let mut scales: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        if let (Some(sc), Some((sfa_v, sfb_v))) = (kind.scale, sf) {
            let nsf = kind.scale_factors();
            let a_id = (desc_bits >> UmmaKind::A_SF_ID_BIT & 3) as usize;
            let b_id = (desc_bits >> UmmaKind::B_SF_ID_BIT & 3) as usize;
            let (sfa_addr, sfb_addr) = (self.get(*sfa_v, base) as u32, self.get(*sfb_v, base) as u32);
            for &blk in &pair {
                let (sa, ca) = self.umma_tmem_sf(&name, "SFA", sfa_addr, a_id, m / v, nsf, sc.sf, k, blk, base)?;
                let (sb, cb) = self.umma_tmem_sf(&name, "SFB", sfb_addr, b_id, n, nsf, sc.sf, k, blk, base)?;
                if blk == block {
                    a_cells.extend([ca, cb]);
                }
                scales.push((sa, sb));
            }
        }
        // Expand the compressed A with each CTA's metadata.
        if let Some(e) = meta {
            let id2 = (desc_bits & UmmaKind::SPARSE_ID2_MASK) as usize;
            let addr = self.get(*e, base) as u32;
            let rows = m / v;
            let mut dense = vec![0f64; m * kk];
            for (cta, &blk) in pair.iter().enumerate() {
                let (nibbles, cells) = self.umma_tmem_meta(kind, addr, id2, k, blk, base)?;
                let rows = if kind.prims.ws { m } else { rows };
                if blk == block {
                    a_cells.push(cells);
                }
                // FP4 sparsity is by pairs (a field per eight K elements).
                let pairs = matches!(kind.kind, UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4);
                let span = if pairs { 8 } else { 4 };
                let (kh, groups) = (kk / 2, kk / span);
                for r in 0..rows {
                    let row = cta * rows + r;
                    for g in 0..groups {
                        let nib = nibbles[r * groups + g];
                        let Some((i0, i1)) = super::sparse_indices(nib) else {
                            return Err(self.err(
                                format!(
                                    "{name}: the metadata of row {row}, K elements {} to {} of A is {nib:#06b}, which \
                                     does not name two of the four positions in increasing order (the 2:4 fields are                                      0b0100, 0b1000, 0b1100, 0b1001, 0b1101 and 0b1110)",
                                    span * g,
                                    span * g + span - 1
                                ),
                                Some(base),
                            ));
                        };
                        if pairs {
                            for e in 0..2 {
                                dense[row * kk + 8 * g + 2 * i0 + e] = a[row * kh + 4 * g + e];
                                dense[row * kk + 8 * g + 2 * i1 + e] = a[row * kh + 4 * g + 2 + e];
                            }
                        } else {
                            dense[row * kk + 4 * g + i0] = a[row * kh + 2 * g];
                            dense[row * kk + 4 * g + i1] = a[row * kh + 2 * g + 1];
                        }
                    }
                }
            }
            a = dense;
        }
        for (bit, v) in [(UmmaKind::NEGATE_A_BIT, &mut a), (UmmaKind::NEGATE_B_BIT, &mut b)] {
            if desc_bits >> bit & 1 == 1 {
                v.iter_mut().for_each(|x| *x = -*x);
            }
        }
        // The lanes {disable-output-lane} masks (bit i % 32 of word i / 32: lane i; with a
        // pair, words 4 to 7 mask the peer CTA's lanes).
        let words: Vec<u32> = disable_lane.iter().map(|&v| self.get(v, base) as u32).collect();
        let masked = |cta: usize, lane: usize| {
            let lane = 128 * cta + lane;
            words.get(lane / 32).is_some_and(|w| w >> (lane % 32) & 1 == 1)
        };
        for (blk, buffer, bytes) in &ranges {
            self.async_proxy_may_read(*buffer, *blk, bytes.clone(), k).map_err(|why| {
                self.err(
                    format!(
                        "{name} reads an operand through the async proxy, but {why}: generic writes of an MMA's \
                         shared-memory operand need fence_view_async_shared (fence.proxy.async) before the MMA"
                    ),
                    Some(base),
                )
            })?;
        }
        // D, from C when accumulating.
        let accumulate = self.get(*enable_input_d, base) != 0;
        let mut out = Vec::with_capacity(m * n);
        for row in 0..m {
            for j in 0..n {
                let (cta, lane, dc) = d_place(kind, row, j);
                if masked(cta, lane) {
                    continue;
                }
                let blk = pair[cta];
                let col = col0 + dc;
                let mut acc = 0f32;
                let mut iacc = 0i32;
                if accumulate {
                    let cell = self.tmem.blocks.entry(blk).or_default().cell_mut(lane, col).to_owned();
                    // An earlier MMA of the pipe still in flight wrote it: in order.
                    let piped = self.umma_covering(blk, lane, col).is_some();
                    match cell.writer {
                        _ if piped => {}
                        None => {
                            if self.opts.detect_uninitialized {
                                return Err(self.err(
                                    format!(
                                        "{name} accumulates (Field.ACCUMULATE is true) into TMEM lane {lane}, column \
                                         {col}{}, which nothing wrote: set ACCUMULATE to false for the first MMA of \
                                         a tile",
                                        if blk == block { "" } else { " of the peer CTA" }
                                    ),
                                    Some(base),
                                ));
                            }
                        }
                        Some((TmemWriter::Warp(o), time))
                            if o != k && self.opts.detect_races && self.clocks[k][self.slot_of(o)] < time =>
                        {
                            return Err(self.err(
                                format!(
                                    "data race: {name} accumulates into TMEM lane {lane}, column {col}, but nothing \
                                     orders the tcgen05.st of {} that wrote it before the MMA",
                                    self.slot_name(self.slot_of(o), block)
                                ),
                                Some(base),
                            ));
                        }
                        _ => {}
                    }
                    let bits = if kind.acc.bits() == 16 { cell.bits & 0xffff } else { cell.bits };
                    if kind.acc == DType::I32 {
                        iacc = bits as i32;
                    } else {
                        acc = to_f64(kind.acc, bits as u64) as f32;
                        // D's input scaled by 2^-scale-input-d (exact).
                        if let Some(s) = kind.prims.scale_input_d {
                            acc = (acc as f64 * 2f64.powi(-(s as i32))) as f32;
                        }
                    }
                }
                let (ar, bc) = (&a[row * kk..][..kk], &b[j * kk..][..kk]);
                if kind.acc == DType::I32 {
                    // .kind::i8: exact integer products, wrapping Int32 sums.
                    for (x, y) in ar.iter().zip(bc) {
                        iacc = iacc.wrapping_add((*x as i32) * (*y as i32));
                    }
                    out.push((blk, lane, col, iacc as u32));
                    continue;
                }
                if let Some(sc) = kind.scale {
                    // Each K-block's products times its two scale factors.
                    let (sa, sb) = &scales[cta];
                    let (nsf, r) = (kind.scale_factors(), row % (m / v));
                    let vec = sc.vec as usize;
                    acc = super::blockscale::block_scaled_dot(
                        acc,
                        kk,
                        vec,
                        |c| ar[c],
                        |q| sa[r * nsf + q],
                        |c| bc[c],
                        |q| sb[j * nsf + q],
                    );
                    out.push((blk, lane, col, from_f32(kind.acc, acc) as u32));
                    continue;
                }
                for (x, y) in ar.iter().zip(bc) {
                    acc += (*x as f32) * (*y as f32);
                }
                out.push((blk, lane, col, from_f32(kind.acc, acc) as u32));
            }
        }
        // The cells take their bits now (later MMAs of the pipe read them); they stay in flight.
        for (blk, lane, col, bits) in out {
            self.tmem.blocks.entry(blk).or_default().cell_mut(lane, col).bits = bits;
        }
        // m11-prims-mma-forms: .ashift moves A's rows up by one within each 32-lane
        // sub-partition once the MMA has read A (the last row of each keeps its value).
        if kind.prims.ashift
            && let Some((_, a_cols)) = a_cells.first().cloned()
        {
            let tmem = self.tmem.blocks.entry(block).or_default();
            for col in a_cols {
                for lane in (0..128).filter(|l| l % 32 != 31) {
                    let next = tmem.cell_mut(lane + 1, col).bits;
                    tmem.cell_mut(lane, col).bits = next;
                }
            }
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
        for (cta, &blk) in pair.iter().enumerate() {
            let lanes: Vec<Range<usize>> =
                d_lanes(kind).into_iter().flatten().filter(|&l| !masked(cta, l)).map(|l| l..l + 1).collect();
            let a_cells = if cta == 0 { a_cells.clone() } else { Vec::new() };
            self.tmem.mmas.push(UmmaFlight {
                id,
                block: blk,
                warp: k,
                lanes,
                cols: cols.clone(),
                loc: inst.loc,
                a_cells,
            });
        }
        self.report.tcgen05_mma_instructions += 1;
        Ok(())
    }

    /// Read the `m x k` A operand of `kind` from tensor
    /// memory at `addr` for the MMA warp `k` (thread `t`) of `block` issues,
    /// by the PTX ISA's description of matrix A in tensor memory: row `i`
    /// in the lane of row `i` of D (Layout D for M = 128, Layout F for
    /// M = 64), its K elements packed into consecutive 32-bit columns from
    /// A's column, the first element in the low bits. Each cell must be in a
    /// live allocation and written by a store that has completed
    /// (`tcgen05.wait::st`) and is ordered before the MMA, or by an MMA.
    /// Returns the values, row-major, and the cells read.
    fn umma_tmem_a(&mut self, kind: UmmaKind, addr: u32, k: usize, block: usize, t: usize) -> R<(Vec<f64>, TmemCells)> {
        let name = kind.ptx();
        let (m, kk) = (kind.m as usize, kind.k());
        let bits = kind.a.bits();
        let (lane0, col0) = ((addr >> 16) as usize, (addr & 0xffff) as usize);
        if lane0 != 0 {
            return Err(
                self.err(format!("{name}: A's TMEM address {addr:#x} has lane {lane0}; A starts at lane 0"), Some(t))
            );
        }
        let cols = col0..col0 + kk * bits / 32;
        let allocs = self.tmem_allocs(block);
        if cols.end > TMEM_COLUMNS
            || !allocs.iter().any(|&(a, c)| a as usize <= cols.start && cols.end <= (a + c) as usize)
        {
            return Err(self.err(
                format!(
                    "{name}: A (columns {} to {} at TMEM address {addr:#x}) is not in a live TMEM allocation of this CTA \
                     (live allocations: {})",
                    cols.start,
                    cols.end - 1,
                    self.tmem_live(block)
                ),
                Some(t),
            ));
        }
        let w = self.warps_per_block;
        let mut values = vec![0f64; m * kk];
        for row in 0..m {
            let lane = d_lane(kind.m, row);
            for col in cols.clone() {
                let cell = self.tmem.blocks.entry(block).or_default().cell_mut(lane, col).to_owned();
                if let Some(o) = cell.storing {
                    return Err(self.err(
                        format!(
                            "{name} reads A from TMEM lane {lane}, column {col} while the tcgen05.st of warp {} into \
                             it is still in flight: wait for the store (tcgen05.wait::st, fence_view_async_tmem_store) \
                             and order that wait before the MMA (a barrier with the tcgen05 fences)",
                            o % w
                        ),
                        Some(t),
                    ));
                }
                match cell.writer {
                    None if self.opts.detect_uninitialized && self.umma_covering(block, lane, col).is_none() => {
                        return Err(self.err(
                            format!("{name} reads A from TMEM lane {lane}, column {col}, which nothing wrote"),
                            Some(t),
                        ));
                    }
                    Some((TmemWriter::Warp(o), time))
                        if o != k && self.opts.detect_races && self.clocks[k][self.slot_of(o)] < time =>
                    {
                        return Err(self.err(
                            format!(
                                "data race: {name} reads A from TMEM lane {lane}, column {col}, but nothing orders the \
                                 tcgen05.st of warp {} that wrote it (and its tcgen05.wait::st) before the MMA",
                                o % w
                            ),
                            Some(t),
                        ));
                    }
                    _ => {}
                }
                let per = 32 / bits;
                for e in 0..per {
                    let c = (col - col0) * per + e;
                    let raw = (cell.bits >> (bits * e)) as u64 & ((1u64 << bits) - 1);
                    values[row * kk + c] = match kind.a.float_format() {
                        Some(f) if bits <= 8 => crate::numeric::decode(f, raw as u32),
                        _ if kind.kind == UmmaDataKind::Tf32 => tf32_truncate(to_f64(kind.a, raw)),
                        _ => to_f64(kind.a, raw),
                    };
                }
            }
        }
        Ok((values, (d_lanes(kind), cols)))
    }

    /// Read the sparsity metadata of one CTA's 128 rows
    /// for a sparse `kind` from tensor memory at `addr` (lane 0, an even
    /// column) and selector `id2`, in `block`'s TMEM, by
    /// [`sparse_meta_place`]. Each cell must be in a live allocation and
    /// written by a store that has completed and is ordered before the MMA,
    /// or by a `tcgen05.cp` of the MMA's warp (or of another warp whose
    /// completion this warp has observed). Returns the 4-bit fields
    /// (`[row * groups + g]`) and the cells read.
    fn umma_tmem_meta(
        &mut self,
        kind: UmmaKind,
        addr: u32,
        id2: usize,
        k: usize,
        block: usize,
        t: usize,
    ) -> R<(Vec<u32>, TmemCells)> {
        let name = kind.ptx();
        let (lane0, col) = ((addr >> 16) as usize, (addr & 0xffff) as usize);
        if lane0 != 0 || col % 2 != 0 {
            return Err(self.err(
                format!(
                    "{name}: the metadata's TMEM address {addr:#x} is not at lane 0 of an even column (the lowest \
                     column bit goes into the instruction descriptor's sparse selector)"
                ),
                Some(t),
            ));
        }
        let eight_bit = kind.kind != UmmaDataKind::F16;
        if eight_bit && id2 != 0 {
            return Err(self.err(
                format!("{name}: sparse selector {id2}: an 8-bit sparse MMA's metadata takes two columns, selector 0"),
                Some(t),
            ));
        }
        let col0 = col + id2;
        // FP4 sparsity takes a field per eight K elements.
        let span = if matches!(kind.kind, UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4) { 8 } else { 4 };
        let groups = kind.k() / span;
        let ncols = if eight_bit { 2 } else { 1 };
        let cols = col0..col0 + ncols;
        let allocs = self.tmem_allocs(block);
        if cols.end > TMEM_COLUMNS
            || !allocs.iter().any(|&(a, c)| a as usize <= cols.start && cols.end <= (a + c) as usize)
        {
            return Err(self.err(
                format!(
                    "{name}: the metadata (columns {} to {} at TMEM address {addr:#x}, selector {id2}) is not in a live \
                     TMEM allocation (live allocations: {})",
                    cols.start,
                    cols.end - 1,
                    self.tmem_live(block)
                ),
                Some(t),
            ));
        }
        // A .ws MMA of M rows reads them from each 128 / M-lane replica
        // (CuTe C++'s tmem_e_frg_ws), which must agree.
        let rows = if kind.prims.ws { kind.m as usize } else { 128 };
        let mut out = vec![0u32; rows * groups];
        for row in 0..rows {
            for g in 0..groups {
                let (lane, c, bit) = sparse_meta_place(eight_bit, row, g);
                let mut seen = None;
                for rep in 0..128 / rows {
                    let l = lane + rep * rows;
                    let bits = self.umma_tmem_operand_cell(&name, "the metadata", block, l, col0 + c, k, t)?;
                    let nib = bits >> bit & 0xf;
                    match seen {
                        None => seen = Some(nib),
                        Some(x) if x != nib => {
                            return Err(self.err(
                                format!(
                                    "{name}: the metadata of row {row}, group {g} is {x:#06b} in TMEM lane {lane} but \
                                     {nib:#06b} in lane {l}: a .ws MMA of M = {rows} reads its metadata from each \
                                     {rows}-lane replica, which must hold the same fields"
                                ),
                                Some(t),
                            ));
                        }
                        _ => {}
                    }
                }
                out[row * groups + g] = seen.expect("a replica");
            }
        }
        Ok((out, (std::iter::once(0..128).collect(), cols)))
    }

    /// One 32-bit TMEM cell an MMA reads as an operand (metadata), with the
    /// ordering checks of [`umma_tmem_sf`](Self::umma_tmem_sf).
    #[allow(clippy::too_many_arguments)]
    fn umma_tmem_operand_cell(
        &mut self,
        name: &str,
        what: &str,
        block: usize,
        lane: usize,
        col: usize,
        k: usize,
        t: usize,
    ) -> R<u32> {
        let w = self.warps_per_block;
        let agent = self.agent_slot(block);
        let cell = self.tmem.blocks.entry(block).or_default().cell_mut(lane, col).to_owned();
        if let Some(o) = cell.storing {
            return Err(self.err(
                format!(
                    "{name} reads {what} from TMEM lane {lane}, column {col} while the tcgen05.st of warp {} into it \
                     is still in flight: wait for the store (fence_view_async_tmem_store) and order that wait before                      the MMA",
                    o % w
                ),
                Some(t),
            ));
        }
        let covering = self.umma_covering(block, lane, col).map(|f| (f.warp, f.loc));
        if let Some((fw, floc)) = covering
            && fw != k
        {
            return Err(self.err(
                format!(
                    "data race: {name} of warp {} reads {what} from TMEM lane {lane}, column {col}, which the \
                     tcgen05.cp (or MMA) of warp {} ({}) still in flight writes: issue the copy from the MMA's warp,                      or wait on its tcgen05.commit's mbarrier before the MMA",
                    k % w,
                    fw % w,
                    at(floc)
                ),
                Some(t),
            ));
        }
        let piped = covering.is_some();
        match cell.writer {
            None if !piped && self.opts.detect_uninitialized => {
                return Err(self.err(
                    format!("{name} reads {what} from TMEM lane {lane}, column {col}, which nothing wrote"),
                    Some(t),
                ));
            }
            Some((TmemWriter::Warp(o), time))
                if !piped && o != k && self.opts.detect_races && self.clocks[k][self.slot_of(o)] < time =>
            {
                return Err(self.err(
                    format!(
                        "data race: {name} reads {what} from TMEM lane {lane}, column {col}, but nothing orders the \
                         tcgen05.st of warp {} that wrote it before the MMA",
                        o % w
                    ),
                    Some(t),
                ));
            }
            Some((TmemWriter::TensorCore, time))
                if !piped && self.opts.detect_races && self.clocks[k][agent] < time =>
            {
                let own = self.ops.iter().any(|o| {
                    o.done == Some(time) && o.trigger == Trigger::Tcgen05Mma && self.clock_index(o.thread) == k
                });
                if !own {
                    return Err(self.err(
                        format!(
                            "data race: {name} of warp {} reads {what} from TMEM lane {lane}, column {col}, written by \
                             another warp's tcgen05.cp whose completion this warp has not observed: wait on that                              copy's tcgen05.commit mbarrier (or issue it from the MMA's warp)",
                            k % w
                        ),
                        Some(t),
                    ));
                }
            }
            _ => {}
        }
        Ok(cell.bits)
    }

    /// Read the scale factors `what` (SFA or SFB) of
    /// `rows` rows, `nsf` each, from tensor memory at `addr` for the MMA warp
    /// `k` (thread `t`) issues, in `block`'s TMEM: by
    /// [`sf_place`](crate::nvgpu::tcgen05::mma_blockscaled::sf_place) from
    /// byte `sf_id` (the descriptor's scale-factor ID), each factor in the
    /// same lane of all four 32-lane sub-partitions (which must agree). Each
    /// cell must be in a live allocation and written by a store that has
    /// completed and is ordered before the MMA, by a `tcgen05.cp` the same
    /// warp issued (the tensor-core pipe is in order), or by one another
    /// warp issued whose completion this warp has observed. Returns the
    /// values (`[row * nsf + j]`) and the cells read.
    #[allow(clippy::too_many_arguments)]
    fn umma_tmem_sf(
        &mut self,
        name: &str,
        what: &str,
        addr: u32,
        sf_id: usize,
        rows: usize,
        nsf: usize,
        sf: DType,
        k: usize,
        block: usize,
        t: usize,
    ) -> R<(Vec<f64>, TmemCells)> {
        use crate::nvgpu::tcgen05::mma_blockscaled::sf_place;
        let (lane0, col0) = ((addr >> 16) as usize, (addr & 0xffff) as usize);
        if lane0 != 0 {
            return Err(self.err(
                format!("{name}: {what}'s TMEM address {addr:#x} has lane {lane0}; the scale factors start at lane 0"),
                Some(t),
            ));
        }
        let last = sf_place(rows, rows - 1, sf_id, nsf - 1).1;
        let cols = col0..col0 + last + 1;
        let allocs = self.tmem_allocs(block);
        if cols.end > TMEM_COLUMNS
            || !allocs.iter().any(|&(a, c)| a as usize <= cols.start && cols.end <= (a + c) as usize)
        {
            return Err(self.err(
                format!(
                    "{name}: {what} (columns {} to {} at TMEM address {addr:#x}) is not in a live TMEM allocation \
                     (live allocations: {})",
                    cols.start,
                    cols.end - 1,
                    self.tmem_live(block)
                ),
                Some(t),
            ));
        }
        let w = self.warps_per_block;
        let agent = self.agent_slot(block);
        let mut values = vec![0f64; rows * nsf];
        for row in 0..rows {
            for j in 0..nsf {
                let (l, c, byte) = sf_place(rows, row, sf_id, j);
                let col = col0 + c;
                let mut seen: Option<(usize, u32)> = None;
                for q in 0..4 {
                    let lane = l + 32 * q;
                    let cell = self.tmem.blocks.entry(block).or_default().cell_mut(lane, col).to_owned();
                    if let Some(o) = cell.storing {
                        return Err(self.err(
                            format!(
                                "{name} reads {what} from TMEM lane {lane}, column {col} while the tcgen05.st of warp \
                                 {} into it is still in flight: wait for the store (fence_view_async_tmem_store) and \
                                 order that wait before the MMA",
                                o % w
                            ),
                            Some(t),
                        ));
                    }
                    if let Some(f) = self.umma_covering(block, lane, col)
                        && f.warp != k
                    {
                        return Err(self.err(
                            format!(
                                "data race: {name} of warp {} reads {what} from TMEM lane {lane}, column {col}, which \
                                 the tcgen05.cp (or MMA) of warp {} ({}) still in flight writes: only the issuing \
                                 thread's tensor-core operations are ordered with each other; issue the copy from the \
                                 MMA's warp, or wait on its tcgen05.commit's mbarrier before the MMA",
                                k % w,
                                f.warp % w,
                                at(f.loc)
                            ),
                            Some(t),
                        ));
                    }
                    let piped = self.umma_covering(block, lane, col).is_some();
                    match cell.writer {
                        None if !piped && self.opts.detect_uninitialized => {
                            return Err(self.err(
                                format!(
                                    "{name} reads {what} from TMEM lane {lane}, column {col}, which nothing wrote (the \
                                     scale factors are replicated in all four 32-lane sub-partitions: stage them with \
                                     a 4x32x128b broadcast tcgen05.cp)"
                                ),
                                Some(t),
                            ));
                        }
                        Some((TmemWriter::Warp(o), time))
                            if !piped && o != k && self.opts.detect_races && self.clocks[k][self.slot_of(o)] < time =>
                        {
                            return Err(self.err(
                                format!(
                                    "data race: {name} reads {what} from TMEM lane {lane}, column {col}, but nothing \
                                     orders the tcgen05.st of warp {} that wrote it before the MMA",
                                    o % w
                                ),
                                Some(t),
                            ));
                        }
                        Some((TmemWriter::TensorCore, time))
                            if !piped && self.opts.detect_races && self.clocks[k][agent] < time =>
                        {
                            // A copy of this warp's is ordered by the pipe; another warp's needs its
                            // completion observed.
                            let own = self.ops.iter().any(|o| {
                                o.done == Some(time)
                                    && o.trigger == Trigger::Tcgen05Mma
                                    && self.clock_index(o.thread) == k
                            });
                            if !own {
                                return Err(self.err(
                                    format!(
                                        "data race: {name} of warp {} reads {what} from TMEM lane {lane}, column \
                                         {col}, written by another warp's tcgen05.cp whose completion this warp has \
                                         not observed: wait on that copy's tcgen05.commit mbarrier (or issue it from \
                                         the MMA's warp)",
                                        k % w
                                    ),
                                    Some(t),
                                ));
                            }
                        }
                        _ => {}
                    }
                    let b = (cell.bits >> (8 * byte)) & 0xff;
                    match seen {
                        None => seen = Some((lane, b)),
                        Some((l0, b0)) if b0 != b => {
                            return Err(self.err(
                                format!(
                                    "{name}: {what} factor {j} of row {row} is {b0:#04x} in TMEM lane {l0} but {b:#04x} \
                                     in lane {lane} (column {col}, byte {byte}): the scale factors are replicated in \
                                     all four 32-lane sub-partitions, which must hold the same values"
                                ),
                                Some(t),
                            ));
                        }
                        _ => {}
                    }
                }
                values[row * nsf + j] = super::blockscale::decode_scale(sf, seen.expect("four replicas").1);
            }
        }
        Ok((values, (std::iter::once(0..128).collect(), cols)))
    }

    /// Read an `mn x kk` operand of type `ty` of a `.kind::<data>` MMA
    /// through descriptor `desc`: its values, row-major (`[row * kk + k]`),
    /// TF32 operands truncated. The bytes read are added to `ranges`.
    #[allow(clippy::too_many_arguments)]
    fn umma_operand(
        &mut self,
        desc: u64,
        major: OperandMajorMode,
        data: UmmaDataKind,
        ty: DType,
        mn: usize,
        kk: usize,
        k: usize,
        block: usize,
        what: &str,
        ranges: &mut Vec<(BufferId, Range<usize>)>,
        enc: UmmaDescEncoding,
    ) -> R<Vec<f64>> {
        let t0 = self.warp_first_thread(k);
        let container = data.smem_bits(ty);
        let map = UmmaOperandMap::new(desc, enc, major, container, mn, kk).map_err(|why| {
            self.err(format!("tcgen05.mma: the {what} descriptor {desc:#018x} cannot be read: {why}"), Some(t0))
        })?;
        // A padded FP6/FP4 value: the bytes of its slot holding its bits, and its bit offset.
        let w = ty.bits();
        let padded = container != w;
        let mut values = vec![0f64; mn * kk];
        for row in 0..mn {
            for c in 0..kk {
                // Packed FP4 (.kind::mxf4, mxf4nvf4) shares bytes.
                let bit = map.bit_address(row, c);
                let mut addr = bit / 8;
                let (mut bytes, mut shift) = ((container / 8).max(1), (bit % 8) as usize);
                if padded {
                    let first = (addr & 15) as usize * w;
                    addr = (addr & !15) + (first / 8) as u64;
                    bytes = (first + w - 1) / 8 - first / 8 + 1;
                    shift = first % 8;
                }
                let (b, byte) = self.resolve(AddressSpace::Smem, addr).map_err(|why| {
                    self.err(
                        format!("tcgen05.mma: element ({row}, {c}) of {what} (descriptor {desc:#018x}): {why}"),
                        Some(t0),
                    )
                })?;
                if byte + bytes > self.buffer_bytes(b) {
                    return Err(self.err(
                        format!("tcgen05.mma: element ({row}, {c}) of {what} runs past the end of its buffer"),
                        Some(t0),
                    ));
                }
                self.umma_may_read(b, block, byte..byte + bytes, k)?;
                let sh = &self.shared[self.home[b.0 as usize]][block];
                let mut raw = 0u64;
                for (i, &x) in sh.data[byte..byte + bytes].iter().enumerate() {
                    raw |= (x as u64) << (8 * i);
                }
                raw = (raw >> shift) & ((1u64 << w) - 1);
                values[row * kk + c] = match ty.float_format() {
                    Some(f) if w <= 8 => crate::numeric::decode(f, raw as u32),
                    _ if data == UmmaDataKind::Tf32 => tf32_truncate(to_f64(ty, raw)),
                    _ => to_f64(ty, raw),
                };
                match ranges.last_mut() {
                    Some((lb, r)) if *lb == b && r.end == byte => r.end = byte + bytes,
                    _ => ranges.push((b, byte..byte + bytes)),
                }
            }
        }
        Ok(values)
    }

    /// The operand (`b`: B, else A) of a `USE` or `LASTUSE`
    /// MMA, from the collector buffer of `block`, or `None` when the MMA
    /// reads shared memory. Reports a collector nothing filled, one filled
    /// from another tile or form, and a tile whose shared memory has changed
    /// since the fill; `LASTUSE` empties the collector.
    #[allow(clippy::too_many_arguments)]
    fn umma_collected(
        &mut self,
        op: CollectorOp,
        slot: u8,
        desc: u64,
        form: (DType, usize, usize, bool),
        block: usize,
        t: usize,
    ) -> R<Option<Vec<f64>>> {
        if !op.reads_collector() {
            return Ok(None);
        }
        let (what, which) = match slot {
            0 => ("A", "a".to_string()),
            5 => ("B", "b".to_string()),
            i => ("B", format!("b{}", i - 1)),
        };
        let name = format!("tcgen05.mma .collector::{which}::{}", op.ptx());
        let Some(c) = self.tmem.collectors.get(&(block, slot)) else {
            return Err(self.err(
                format!(
                    "{name} reads {what} from {what}'s collector buffer, but no earlier MMA of this CTA filled it (or a DISCARD or LASTUSE MMA emptied it): the first MMA of a reuse sequence uses CollectorOp.FILL"
                ),
                Some(t),
            ));
        };
        if c.desc != desc || c.form != form {
            return Err(self.err(
                format!(
                    "{name} reads {what} from its collector buffer through descriptor {desc:#018x}, but the buffer was filled from another tile (descriptor {:#018x}, {:?} {}x{}, at {}): a collected operand is reused with the same descriptor and form",
                    c.desc,
                    c.form.0,
                    c.form.1,
                    c.form.2,
                    at(c.loc)
                ),
                Some(t),
            ));
        }
        let check = self.opts.collector_source_check;
        for (buffer, start, bytes) in c.bytes.iter().filter(|_| check) {
            let now = &self.shared[self.home[buffer.0 as usize]][block].data[*start..start + bytes.len()];
            if let Some(i) = now.iter().zip(bytes).position(|(x, y)| x != y) {
                return Err(self.err(
                    format!(
                        "{name} reuses {what} from its collector buffer, but byte {} of shared buffer {} it was filled from (at {}) has changed since: the collector holds the old tile; fill it again (CollectorOp.FILL) after changing the operand",
                        start + i,
                        buffer.0,
                        at(c.loc)
                    ),
                    Some(t),
                ));
            }
        }
        let values = c.values.clone();
        if op == CollectorOp::LASTUSE {
            self.tmem.collectors.remove(&(block, slot));
        }
        Ok(Some(values))
    }

    /// After an MMA read an operand from shared memory:
    /// `FILL` keeps it (and the bytes it came from) in the collector of
    /// `block`; any other usage leaves the collector empty.
    #[allow(clippy::too_many_arguments)]
    fn umma_fill(
        &mut self,
        inst: &'a Inst,
        op: CollectorOp,
        slot: u8,
        desc: u64,
        form: (DType, usize, usize, bool),
        block: usize,
        values: &[f64],
        ranges: &[(BufferId, Range<usize>)],
    ) {
        if op != CollectorOp::FILL {
            self.tmem.collectors.remove(&(block, slot));
            return;
        }
        let bytes = ranges
            .iter()
            .map(|(buffer, r)| {
                let data = &self.shared[self.home[buffer.0 as usize]][block].data;
                (*buffer, r.start, data[r.clone()].to_vec())
            })
            .collect();
        let c = Collector { desc, form, values: values.to_vec(), bytes, loc: inst.loc };
        self.tmem.collectors.insert((block, slot), c);
    }

    /// The first thread of warp `k`.
    fn warp_first_thread(&self, k: usize) -> usize {
        let w = self.warps_per_block;
        (k / w) * self.threads_per_block + (k % w) * 32
    }

    /// May the MMA issued by warp `k` read bytes `range` of shared buffer `b` of `block`?
    fn umma_may_read(&self, b: BufferId, block: usize, range: Range<usize>, k: usize) -> R {
        let (slot, clock) = (self.slot_of(k), &self.clocks[k]);
        let t = self.warp_first_thread(k);
        for i in range {
            let m = &self.shared[self.home[b.0 as usize]][block].meta[i];
            let place = smem_place(self.ir.buffer(b), i);
            if m.inflight > 0 {
                return Err(self.err(
                    format!(
                        "tcgen05.mma reads {place} while an asynchronous copy is still writing it (wait for the copy's \
                         mbarrier phase first)"
                    ),
                    Some(t),
                ));
            }
            if !m.written {
                if self.opts.detect_uninitialized {
                    return Err(self.err(format!("tcgen05.mma reads {place}, which was never written"), Some(t)));
                }
                continue;
            }
            if self.opts.detect_races && !m.write_ordered_before(slot, clock) {
                return Err(self.err(
                    format!(
                        "data race: the tcgen05.mma of warp {} reads {place}, but nothing orders its last write, by {}, \
                         before the MMA (a barrier, or the wait on the mbarrier of the copy that wrote it)",
                        k % self.warps_per_block,
                        self.slot_name(m.w_slot as usize, block)
                    ),
                    Some(t),
                ));
            }
        }
        Ok(())
    }

    /// Land MMA `id` as a write of its CTA's tensor-core agent: its D cells
    /// become the agent's, at a new time of the agent's, which it returns.
    ///
    /// A pair's MMA has a flight in each CTA. Its cells in both
    /// become their CTA's tensor-core writes at one time, to which both
    /// CTAs' agent clocks advance, so that a waiter in either CTA whose
    /// commit arrival publishes its CTA's agent time observes them.
    pub(super) fn umma_land(&mut self, id: usize) -> u32 {
        let mut time = self.complete_as_agent(id, false);
        let flights: Vec<UmmaFlight> = {
            let (mine, rest) = std::mem::take(&mut self.tmem.mmas).into_iter().partition(|f| f.id == id);
            self.tmem.mmas = rest;
            mine
        };
        if flights.len() > 1 {
            time = flights.iter().map(|f| self.bulk_time[f.block]).max().unwrap_or(time).max(time);
            for f in &flights {
                self.bulk_time[f.block] = time;
            }
            self.ops[id].done = Some(time);
        }
        for f in &flights {
            let tmem = self.tmem.blocks.entry(f.block).or_default();
            for lanes in &f.lanes {
                for lane in lanes.clone() {
                    for col in f.cols.clone() {
                        let c = tmem.cell_mut(lane, col);
                        c.writer = Some((TmemWriter::TensorCore, time));
                    }
                }
            }
        }
        time
    }

    /// The MMA in flight whose D covers TMEM cell (`lane`, `col`) of `block`.
    pub(super) fn umma_covering(&self, block: usize, lane: usize, col: usize) -> Option<&UmmaFlight> {
        self.tmem.mmas.iter().find(|f| f.covers(block, lane, col))
    }

    /// `tcgen05.commit` by the threads of `active`.
    pub(super) fn umma_commit(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        let Op::Tcgen05Commit { buffer, index, mask, two_cta } = &inst.op else { unreachable!("a tcgen05.commit") };
        self.progress();
        for &t in active {
            self.tcgen05_group("tcgen05.commit", *two_cta, inst.loc, t)?;
            let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "tcgen05.commit")?;
            let own = self.block_rank(t);
            let blocks: Vec<usize> = match mask {
                None => vec![own],
                Some(m) => {
                    let bits = self.get(*m, t) as u16;
                    let base = self.cluster_base(own);
                    if let Some(r) = (0..16).find(|&r| bits >> r & 1 == 1 && r >= self.cluster_size) {
                        return Err(self.err(
                            format!(
                                "tcgen05.commit with the multicast mask {bits:#06x}, which names CTA {r} of a cluster \
                                 of {} CTA(s)",
                                self.cluster_size
                            ),
                            Some(t),
                        ));
                    }
                    (0..16).filter(|&r| bits >> r & 1 == 1).map(|r| base + r).collect()
                }
            };
            let k = self.clock_index(t);
            let mut copies: Vec<usize> = self.tmem.mmas.iter().filter(|f| f.warp == k).map(|f| f.id).collect();
            // One entry per MMA, not per CTA of its pair.
            copies.dedup();
            if self.opts.async_completion == crate::sim::AsyncCompletion::Early {
                for &id in &copies {
                    self.umma_land(id);
                }
            }
            for blk in blocks {
                let key = (blk, buffer.0, byte);
                if blk != own {
                    self.check_remote_mbarrier(key, t, "tcgen05.commit")?;
                }
                let floor = if self.async_rng.is_some() { self.bulk_time[blk] } else { 0 };
                let Some(state) = self.mbarriers.get_mut(&key) else {
                    return Err(self.err("tcgen05.commit on an mbarrier that was never initialised", Some(t)));
                };
                let (owed, left) = (state.deferred.len() as u32, state.arrivals_left);
                if owed >= left {
                    return Err(self.err(
                        format!(
                            "tcgen05.commit on an mbarrier whose phase has no arrival left for it ({left} pending, \
                             {owed} owed by earlier commits or cp.async arrives): the commit arrives once per \
                             executing thread, so wrap it in elect_one and count it in mbarrier_init"
                        ),
                        Some(t),
                    ));
                }
                state.deferred.push(DeferredArrival { copies: copies.clone(), floor, commit: true });
                self.fire_ready_arrivals(key);
            }
        }
        Ok(())
    }
}
