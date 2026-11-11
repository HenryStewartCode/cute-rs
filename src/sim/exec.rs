//! The simulator's interpreter.
//!
//! Values are kept as raw bits, zero-extended to 64. Memory is byte
//! addressed, so one allocation can be read as several element types and
//! hazards are tracked per byte. The threads of every block in a cluster run
//! in lockstep; without clusters that is one block at a time.
//!
//! Shared-memory races are found with vector clocks at warp granularity:
//! each warp of a block keeps the latest time of every warp of the block
//! that it is ordered after. `sync_threads`, named barriers (`bar.sync`
//! acquires what `bar.arrive` released) and mbarriers (a successful wait
//! acquires what the arrivals of the completed phase released) merge
//! clocks; an access races with an earlier one by another warp unless the
//! accessing warp's clock has caught up with it. Bulk copies are written by
//! a per-block agent with a clock entry of its own, which the mbarrier
//! phase they complete publishes to its waiters.
//!
//! The clocks also say what a warp knows about an mbarrier's phases (those
//! it, or a warp it is ordered after, has seen complete), which is what
//! makes a parity wait meaningful whatever order the warps would run in on
//! hardware.

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::float::{f64_to_f16, is_nan, ptx_nan};
use super::{LaunchConfig, SimError, SimOptions, SimReport};
use crate::ir::{
    AtomicKind, BarrierKind, BinaryOp, BufferId, CmpPred, DType, Inst, KernelIr, LaneMask, Loc, MbarrierWaitKind,
    MemSpace, MmaKind, Op, Region, ShuffleKind, SpecialReg, UnaryOp, ValueId, VoteKind,
};
use crate::value::{bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits};

type R<T = ()> = Result<T, SimError>;

/// Per-byte shared memory state for hazard detection.
#[derive(Clone, Copy, Default)]
struct Meta {
    written: bool,
    inflight: u16,
    /// The writer of the last write (a warp, or the block's bulk-copy agent
    /// at index `warps_per_block`) and its own clock at the time. Clock 0 is
    /// ordered before every access.
    w_warp: u16,
    w_clock: u32,
}

struct Shared {
    data: Vec<u8>,
    meta: Vec<Meta>,
    /// For each byte, the clock of each warp of the block at its last read
    /// since the last write (0: no read). Empty without race detection.
    reads: Vec<u32>,
}

/// The state of one named barrier (`bar.sync` / `bar.arrive`) of a block.
#[derive(Clone, Default)]
struct NamedState {
    /// Warps that have arrived at the current, incomplete use.
    arrived: Vec<usize>,
    /// The warp count that completes it.
    expected: usize,
    /// The join of the arrived warps' clocks.
    release: Vec<u32>,
}

/// `a = max(a, b)` element-wise.
fn join(a: &mut [u32], b: &[u32]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = (*x).max(y);
    }
}

/// `0..3, 5` style list of warp numbers.
fn warp_list(warps: &[usize]) -> String {
    let mut sorted = warps.to_vec();
    sorted.sort_unstable();
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j + 1 < sorted.len() && sorted[j + 1] == sorted[j] + 1 {
            j += 1;
        }
        parts.push(if j > i { format!("{}..={}", sorted[i], sorted[j]) } else { sorted[i].to_string() });
        i = j + 1;
    }
    parts.join(", ")
}

struct Pending {
    dst: BufferId,
    block: usize,
    dst_byte: usize,
    data: Vec<u8>,
    warp: u16,
}

/// One mbarrier's state. It is keyed by its address, so it needs no storage.
struct Mbarrier {
    arrivals_per_phase: u32,
    arrivals_left: u32,
    /// The current phase's tx-count: bytes of bulk copies it still waits
    /// for. `expect_tx` raises it and landing copies lower it; the PTX ISA
    /// lets it go negative, but copies only land here once every arrival of
    /// the phase is in, and landing more than it holds is reported.
    tx_count: u64,
    /// Bulk copies in flight that report to this mbarrier.
    copies: Vec<Pending>,
    /// Completed phases.
    phase: u64,
    /// The join of the clocks of the warps (of the mbarrier's block) that
    /// arrived in the current phase, and in the last completed one. The
    /// entry after the warps' is the block's bulk-copy agent, whose writes
    /// the phase's completion publishes.
    release: Vec<u32>,
    completed: Vec<u32>,
    /// For each warp of the mbarrier's block, the phases it has seen
    /// complete through its own successful waits: `(the warp's own clock
    /// then, completed phases)`, in order. Another warp knows of them once
    /// its clock has caught up with that time.
    seen: Vec<Vec<(u32, u64)>>,
}

impl Mbarrier {
    /// Complete the current phase if its arrivals and transaction bytes are
    /// all in; returns whether it did.
    fn complete_if_done(&mut self) -> bool {
        let done = self.arrivals_left == 0 && self.tx_count == 0;
        if done {
            self.arrivals_left = self.arrivals_per_phase;
            self.phase += 1;
            self.completed = std::mem::replace(&mut self.release, vec![0; self.completed.len()]);
        }
        done
    }

    /// How many phases the warp whose vector clock is `clock` knows to be
    /// complete, from its own waits and those of warps it is ordered after.
    fn known_phases(&self, clock: &[u32]) -> u64 {
        self.seen
            .iter()
            .zip(clock)
            .filter_map(|(seen, &now)| {
                // Entries grow in time and phase: take the last one by `now`.
                let known = seen.partition_point(|&(time, _)| time <= now);
                known.checked_sub(1).map(|i| seen[i].1)
            })
            .max()
            .unwrap_or(0)
    }
}

/// A buffer of per-thread memory: `per_thread` bytes for each thread.
struct LocalMem {
    per_thread: usize,
    data: Vec<u8>,
    written: Vec<bool>,
}

pub(super) struct Machine<'a> {
    ir: &'a KernelIr,
    opts: &'a SimOptions,
    cfg: LaunchConfig,
    globals: Vec<Option<Vec<u8>>>,
    scalars: Vec<(ValueId, u64)>,
    report: SimReport,
    /// The blocks running together, indexed by rank in the cluster.
    blocks: Vec<[u32; 3]>,
    threads_per_block: usize,
    nthreads: usize,
    values: Vec<Vec<u64>>,
    /// `shared[buffer][block rank]`.
    shared: Vec<Vec<Shared>>,
    locals: Vec<Option<LocalMem>>,
    uncommitted: Vec<Vec<Pending>>,
    groups: Vec<VecDeque<Vec<Pending>>>,
    /// Warps per block (the last one may be short).
    warps_per_block: usize,
    /// The vector clock of each warp, by `block rank * warps_per_block + warp`.
    /// It has an entry for every warp of the block and, after them, one for
    /// the block's bulk-copy agent (index `warps_per_block`).
    clocks: Vec<Vec<u32>>,
    /// The latest time of each block's bulk-copy agent: every bulk copy that
    /// lands gets a new one.
    bulk_time: Vec<u32>,
    /// The 16 named barriers of each block.
    named: Vec<Vec<NamedState>>,
    /// By `(block rank, buffer, byte)`.
    mbarriers: HashMap<(usize, u32, usize), Mbarrier>,
    clock: u64,
    loc: Option<Loc>,
}

fn f32b(x: f32) -> u64 {
    x.to_bits() as u64
}

fn bf(b: u64) -> f32 {
    f32::from_bits(b as u32)
}

fn h2f(b: u64) -> f32 {
    f16_bits_to_f32(b as u16)
}

fn f2h(x: f32) -> u64 {
    f32_to_f16_bits(x) as u64
}

fn mask(bits: u64, width: usize) -> u64 {
    if width >= 64 { bits } else { bits & ((1u64 << width) - 1) }
}

fn sext(bits: u64, width: usize) -> i64 {
    let shift = 64 - width as u32;
    ((bits << shift) as i64) >> shift
}

/// An f32 (or narrower) float's value.
fn to_f32(ty: DType, b: u64) -> f32 {
    match ty {
        DType::F32 => bf(b),
        DType::F16 => h2f(b),
        DType::BF16 => bf16_bits_to_f32(b as u16),
        DType::F64 => f64::from_bits(b) as f32,
        DType::Bool => b as f32,
        t if t.is_signed() => sext(b, t.bits()) as f32,
        _ => b as f32,
    }
}

fn from_f32(ty: DType, x: f32) -> u64 {
    match ty {
        DType::F32 => f32b(x),
        DType::F16 => f2h(x),
        DType::BF16 => f32_to_bf16_bits(x) as u64,
        DType::F64 => (x as f64).to_bits(),
        _ => unreachable!("not a float type"),
    }
}

fn to_f64(ty: DType, b: u64) -> f64 {
    match ty {
        DType::F64 => f64::from_bits(b),
        t if t.is_float() => to_f32(t, b) as f64,
        DType::Bool => b as f64,
        t if t.is_signed() => sext(b, t.bits()) as f64,
        _ => b as f64,
    }
}

fn int_binary(op: BinaryOp, ty: DType, a: u64, b: u64) -> Result<u64, String> {
    use BinaryOp::*;
    let w = ty.bits();
    let signed = ty.is_signed();
    let (x, y) = (sext(a, w), sext(b, w));
    let r: u64 = match op {
        Add => a.wrapping_add(b),
        Sub => a.wrapping_sub(b),
        Mul => a.wrapping_mul(b),
        Div | Rem | DivU | RemU if b == 0 => return Err("integer division by zero".into()),
        Div if signed => x.wrapping_div(y) as u64,
        Div | DivU => a / b,
        Rem if signed => x.wrapping_rem(y) as u64,
        Rem | RemU => a % b,
        Min => {
            if signed {
                x.min(y) as u64
            } else {
                a.min(b)
            }
        }
        Max => {
            if signed {
                x.max(y) as u64
            } else {
                a.max(b)
            }
        }
        And => a & b,
        Or => a | b,
        Xor => a ^ b,
        Shl | ShrU | ShrS if b >= w as u64 => return Err(format!("shift by {y} bits")),
        Shl => a << b,
        ShrU => a >> b,
        ShrS => (x >> b) as u64,
        MulHi => {
            if signed {
                ((x as i128 * y as i128) >> w) as u64
            } else {
                ((a as u128 * b as u128) >> w) as u64
            }
        }
    };
    Ok(mask(r, w))
}

/// PTX's `min` and `max` on floats: a NaN operand is ignored (two give a
/// NaN), and `-0.0 < +0.0`.
fn float_min_max(is_max: bool, ty: DType, a: u64, b: u64) -> u64 {
    let (x, y) = (to_f64(ty, a), to_f64(ty, b));
    match (x.is_nan(), y.is_nan()) {
        (true, true) => ptx_nan(ty, &[a, b], a),
        (true, false) => b,
        (false, true) => a,
        _ if x == y => {
            // Equal values: only zeros of opposite signs differ.
            let a_negative = a >> (ty.bits() - 1) & 1 == 1;
            if a_negative != is_max { a } else { b }
        }
        _ => {
            if (x > y) == is_max {
                a
            } else {
                b
            }
        }
    }
}

fn binary(op: BinaryOp, ty: DType, a: u64, b: u64) -> Result<u64, String> {
    use BinaryOp::*;
    Ok(match ty {
        DType::Bool => match op {
            And => a & b & 1,
            Or => (a | b) & 1,
            Xor => (a ^ b) & 1,
            _ => unreachable!("validated"),
        },
        t if t.is_float() && matches!(op, Min | Max) => float_min_max(op == Max, t, a, b),
        DType::F64 => {
            let (x, y) = (f64::from_bits(a), f64::from_bits(b));
            let r = match op {
                Add => x + y,
                Sub => x - y,
                Mul => x * y,
                Div => x / y,
                _ => unreachable!("validated"),
            };
            ptx_nan(DType::F64, &[a, b], r.to_bits())
        }
        t if t.is_float() => {
            let (x, y) = (to_f32(t, a), to_f32(t, b));
            let r = match op {
                Add => x + y,
                Sub => x - y,
                Mul => x * y,
                Div => x / y,
                _ => unreachable!("validated"),
            };
            ptx_nan(t, &[a, b], from_f32(t, r))
        }
        t => int_binary(op, t, a, b)?,
    })
}

fn unary(op: UnaryOp, ty: DType, a: u64) -> u64 {
    if ty == DType::Bool {
        return (a ^ 1) & 1;
    }
    if ty.is_int() {
        let w = ty.bits();
        let x = sext(a, w);
        let r = match op {
            UnaryOp::Neg => x.wrapping_neg() as u64,
            UnaryOp::Not => !a,
            UnaryOp::Abs => x.wrapping_abs() as u64,
            UnaryOp::Popc => a.count_ones() as u64,
            UnaryOp::Clz => (a.leading_zeros() as u64).wrapping_sub(64 - w as u64),
            UnaryOp::Ctz => (a.trailing_zeros() as u64).min(w as u64),
            UnaryOp::Brev => a.reverse_bits() >> (64 - w),
            _ => unreachable!("validated"),
        };
        return mask(r, w);
    }
    // Float abs and negation are emitted as `and.bN` and `xor.bN` on the
    // sign bit (inline PTX): bit operations, which keep a NaN's payload.
    let sign = 1u64 << (ty.bits() - 1);
    match op {
        UnaryOp::Abs => return a & !sign,
        UnaryOp::Neg => return a ^ sign,
        _ => {}
    }
    if ty == DType::F64 {
        let x = f64::from_bits(a);
        let r = match op {
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Rsqrt => 1.0 / x.sqrt(),
            UnaryOp::Rcp => 1.0 / x,
            UnaryOp::Floor => x.floor(),
            UnaryOp::Ceil => x.ceil(),
            UnaryOp::Trunc => x.trunc(),
            UnaryOp::Rint => x.round_ties_even(),
            _ => unreachable!("validated"),
        };
        return ptx_nan(ty, &[a], r.to_bits());
    }
    let x = to_f32(ty, a);
    let r = match op {
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Exp2 => x.exp2(),
        UnaryOp::Log2 => x.log2(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Rcp => 1.0 / x,
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Trunc => x.trunc(),
        UnaryOp::Rint => x.round_ties_even(),
        _ => unreachable!("validated"),
    };
    ptx_nan(ty, &[a], from_f32(ty, r))
}

fn float_to_int(x: f64, to: DType) -> u64 {
    match to {
        DType::I8 => x as i8 as u8 as u64,
        DType::I16 => x as i16 as u16 as u64,
        DType::I32 => x as i32 as u32 as u64,
        DType::I64 => x as i64 as u64,
        DType::U8 => x as u8 as u64,
        DType::U16 => x as u16 as u64,
        DType::U32 => x as u32 as u64,
        DType::U64 => x as u64,
        _ => unreachable!("not an integer type"),
    }
}

fn cast(from: DType, to: DType, b: u64) -> u64 {
    if from == to {
        return b;
    }
    if from.is_float() && to.is_float() {
        if is_nan(from, b) {
            // As cvt does (checked on SM89 by tests/gpu.rs): f32 <-> f64
            // keep the sign and the top of the payload and quiet it;
            // f64 -> f16 gives the quiet NaN of its sign; conversions to the
            // 16-bit types from f32 (and so from bf16 and f16, which convert
            // through f32) give the canonical 0x7fff.
            let sign = b >> (from.bits() - 1) & 1;
            return match (from, to) {
                (DType::F64, DType::F32) => sign << 31 | 0x7fc0_0000 | (b >> 29 & 0x7f_ffff),
                (DType::F32, DType::F64) => sign << 63 | 0x7ff8_0000_0000_0000 | (b & 0x7f_ffff) << 29,
                (DType::F64, DType::F16) => sign << 15 | 0x7e00,
                (_, DType::F16 | DType::BF16) => 0x7fff,
                _ => from_f32(to, to_f32(from, b)),
            };
        }
        if from == DType::F64 && to == DType::F16 {
            // One rounding, as cvt.rn.f16.f64 does (not through f32).
            return f64_to_f16(b);
        }
    }
    if to.is_int() {
        if from.is_float() {
            return float_to_int(to_f64(from, b), to);
        }
        let v = if from.is_signed() { sext(b, from.bits()) as u64 } else { b };
        return mask(v, to.bits());
    }
    match to {
        DType::F64 => to_f64(from, b).to_bits(),
        DType::Bool => unreachable!("validated"),
        _ => from_f32(to, to_f32(from, b)),
    }
}

fn compare(pred: CmpPred, ty: DType, a: u64, b: u64) -> bool {
    use std::cmp::Ordering::*;
    let ordering = if ty.is_float() {
        to_f64(ty, a).partial_cmp(&to_f64(ty, b))
    } else if ty.is_signed() {
        Some(sext(a, ty.bits()).cmp(&sext(b, ty.bits())))
    } else {
        Some(a.cmp(&b))
    };
    match pred {
        CmpPred::Eq => ordering == Some(Equal),
        CmpPred::Ne => ordering != Some(Equal),
        CmpPred::Lt => ordering == Some(Less),
        CmpPred::Le => matches!(ordering, Some(Less | Equal)),
        CmpPred::Gt => ordering == Some(Greater),
        CmpPred::Ge => matches!(ordering, Some(Greater | Equal)),
    }
}

/// `(row, col)` of value `i` of lane `lane` in an `mma.sync` A fragment
/// (PTX ISA, "Matrix Fragments for mma.m16n8k16 / m16n8k8 / m16n8k4 / m8n8k4").
/// `g` is `groupID` and `t` is `threadID_in_group`.
fn mma_a_coord(kind: MmaKind, lane: usize, i: usize) -> (usize, usize) {
    let (g, t) = (lane >> 2, lane % 4);
    match kind {
        // .f16/.bf16 m16n8k16: rows g (a0, a1, a4, a5) and g + 8; columns
        // 2t + (i & 1), plus 8 for a4..a7.
        MmaKind::M16N8K16F32F16 | MmaKind::M16N8K16F16F16 | MmaKind::M16N8K16F32BF16 => {
            (g + 8 * ((i >> 1) & 1), 2 * t + (i & 1) + 8 * (i >> 2))
        }
        // .f16/.bf16 m16n8k8: rows g (a0, a1) and g + 8 (a2, a3).
        MmaKind::M16N8K8F32F16 | MmaKind::M16N8K8F16F16 | MmaKind::M16N8K8F32BF16 => {
            (g + 8 * (i >> 1), 2 * t + (i & 1))
        }
        MmaKind::M16N8K4Tf32 => (g + 8 * i, t),
        // .tf32 m16n8k8: rows g (a0, a2) and g + 8 (a1, a3); columns t (a0,
        // a1) and t + 4 (a2, a3).
        MmaKind::M16N8K8Tf32 => (g + 8 * (i & 1), t + 4 * (i >> 1)),
        MmaKind::M8N8K4F64 => (g, t),
    }
}

/// `(k, n)` of value `i` of lane `lane` in an `mma.sync` B fragment.
fn mma_b_coord(kind: MmaKind, lane: usize, i: usize) -> (usize, usize) {
    let (g, t) = (lane >> 2, lane % 4);
    match kind {
        MmaKind::M16N8K16F32F16 | MmaKind::M16N8K16F16F16 | MmaKind::M16N8K16F32BF16 => {
            (2 * t + (i & 1) + 8 * (i >> 1), g)
        }
        MmaKind::M16N8K8F32F16 | MmaKind::M16N8K8F16F16 | MmaKind::M16N8K8F32BF16 => (2 * t + (i & 1), g),
        // .tf32 m16n8k8: rows t (b0) and t + 4 (b1).
        MmaKind::M16N8K8Tf32 => (t + 4 * i, g),
        MmaKind::M16N8K4Tf32 | MmaKind::M8N8K4F64 => (t, g),
    }
}

/// `(m, n)` of value `i` of lane `lane` in an `mma.sync` C/D fragment.
fn mma_c_coord(kind: MmaKind, lane: usize, i: usize) -> (usize, usize) {
    let (g, t) = (lane >> 2, lane % 4);
    match kind {
        MmaKind::M8N8K4F64 => (g, 2 * t + i),
        _ => (g + 8 * (i >> 1), 2 * t + (i & 1)),
    }
}

fn read_le(bytes: &[u8]) -> u64 {
    bytes.iter().rev().fold(0, |acc, &b| (acc << 8) | b as u64)
}

fn write_le(bytes: &mut [u8], bits: u64) {
    for (k, b) in bytes.iter_mut().enumerate() {
        *b = (bits >> (8 * k)) as u8;
    }
}

/// Expand a C format string. Understands `%d %i %u %x %X %f %e %g %%` with an
/// optional precision.
fn format_printf(format: &str, args: &[(DType, u64)]) -> String {
    let mut out = String::new();
    let mut args = args.iter();
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let mut spec = String::new();
        while let Some(&n) = chars.peek() {
            chars.next();
            if n.is_ascii_alphabetic() || n == '%' {
                spec.push(n);
                break;
            }
            spec.push(n);
        }
        let conv = spec.pop().unwrap_or('%');
        if conv == '%' {
            out.push('%');
            continue;
        }
        let precision = spec.split('.').nth(1).and_then(|p| p.parse::<usize>().ok());
        let Some(&(ty, bits)) = args.next() else {
            out.push_str("<missing>");
            continue;
        };
        let text = match conv {
            'd' | 'i' => (if ty.is_signed() { sext(bits, ty.bits()) } else { bits as i64 }).to_string(),
            'u' => bits.to_string(),
            'x' => format!("{bits:x}"),
            'X' => format!("{bits:X}"),
            'f' | 'F' => format!("{:.*}", precision.unwrap_or(6), to_f64(ty, bits)),
            'e' | 'E' => format!("{:.*e}", precision.unwrap_or(6), to_f64(ty, bits)),
            _ => format!("{}", to_f64(ty, bits)),
        };
        out.push_str(&text);
    }
    out
}

impl<'a> Machine<'a> {
    pub(super) fn new(
        ir: &'a KernelIr,
        opts: &'a SimOptions,
        cfg: LaunchConfig,
        globals: Vec<Option<Vec<u8>>>,
        scalars: Vec<(ValueId, u64)>,
    ) -> Machine<'a> {
        Machine {
            ir,
            opts,
            cfg,
            globals,
            scalars,
            report: SimReport::default(),
            blocks: Vec::new(),
            threads_per_block: cfg.threads_per_block(),
            nthreads: 0,
            values: Vec::new(),
            shared: Vec::new(),
            locals: Vec::new(),
            uncommitted: Vec::new(),
            groups: Vec::new(),
            warps_per_block: cfg.threads_per_block().div_ceil(32),
            clocks: Vec::new(),
            bulk_time: Vec::new(),
            named: Vec::new(),
            mbarriers: HashMap::new(),
            clock: 0,
            loc: None,
        }
    }

    pub(super) fn finish(self) -> (Vec<Option<Vec<u8>>>, SimReport) {
        (self.globals, self.report)
    }

    fn block_rank(&self, t: usize) -> usize {
        t / self.threads_per_block
    }

    fn block_of(&self, t: usize) -> [u32; 3] {
        self.blocks[self.block_rank(t)]
    }

    fn lane_of(&self, t: usize) -> usize {
        (t % self.threads_per_block) % 32
    }

    fn warp_of(&self, t: usize) -> u16 {
        ((t % self.threads_per_block) / 32) as u16
    }

    /// A number identifying this thread's warp across the cluster.
    fn warp_key(&self, t: usize) -> usize {
        self.block_rank(t) * 64 + self.warp_of(t) as usize
    }

    fn thread_coord(&self, t: usize) -> [u32; 3] {
        let t = t % self.threads_per_block;
        let [bx, by, _] = self.cfg.block.map(|d| d as usize);
        [(t % bx) as u32, ((t / bx) % by) as u32, (t / (bx * by)) as u32]
    }

    fn err(&self, message: impl Into<String>, thread: Option<usize>) -> SimError {
        SimError {
            message: message.into(),
            loc: self.loc,
            block: thread.map_or_else(|| self.blocks.first().copied().unwrap_or([0; 3]), |t| self.block_of(t)),
            thread: thread.map(|t| self.thread_coord(t)),
        }
    }

    pub(super) fn run(&mut self) -> R {
        let [gx, gy, gz] = self.cfg.grid;
        let [cx, cy, cz] = self.cfg.cluster;
        for z in (0..gz).step_by(cz as usize) {
            for y in (0..gy).step_by(cy as usize) {
                for x in (0..gx).step_by(cx as usize) {
                    let mut blocks = Vec::new();
                    for bz in z..z + cz {
                        for by in y..y + cy {
                            for bx in x..x + cx {
                                blocks.push([bx, by, bz]);
                            }
                        }
                    }
                    self.run_cluster(blocks)?;
                }
            }
        }
        Ok(())
    }

    fn run_cluster(&mut self, blocks: Vec<[u32; 3]>) -> R {
        let count = blocks.len();
        self.blocks = blocks;
        self.nthreads = self.threads_per_block * count;
        self.loc = None;
        let w = self.warps_per_block;
        self.clocks = (0..count * w)
            .map(|k| {
                let mut clock = vec![0; w + 1];
                clock[k % w] = 1;
                clock
            })
            .collect();
        self.bulk_time = vec![0; count];
        self.named = vec![vec![NamedState::default(); 16]; count];
        self.mbarriers.clear();
        self.values = vec![Vec::new(); self.ir.value_types.len()];
        let reads_per_byte = if self.opts.detect_races { w } else { 0 };
        self.shared = self
            .ir
            .buffers
            .iter()
            .map(|b| {
                if b.space != MemSpace::Shared {
                    return Vec::new();
                }
                let bytes = b.len.expect("shared buffers have a size") * b.elem.bytes();
                (0..count)
                    .map(|_| Shared {
                        data: vec![0xff; bytes],
                        meta: vec![Meta::default(); bytes],
                        reads: vec![0; bytes * reads_per_byte],
                    })
                    .collect()
            })
            .collect();
        let nthreads = self.nthreads;
        self.locals = self
            .ir
            .buffers
            .iter()
            .map(|b| {
                (b.space == MemSpace::Local).then(|| {
                    let per_thread = b.len.expect("local buffers have a size") * b.elem.bytes();
                    LocalMem {
                        per_thread,
                        data: vec![0xff; per_thread * nthreads],
                        written: vec![false; per_thread * nthreads],
                    }
                })
            })
            .collect();
        self.uncommitted = (0..self.nthreads).map(|_| Vec::new()).collect();
        self.groups = (0..self.nthreads).map(|_| VecDeque::new()).collect();
        let all: Vec<usize> = (0..self.nthreads).collect();
        for (v, bits) in self.scalars.clone() {
            for &t in &all {
                self.set(v, t, bits);
            }
        }
        let body = &self.ir.body;
        self.region(body, &all)?;
        self.report.blocks += count as u64;
        Ok(())
    }

    fn get(&self, v: ValueId, t: usize) -> u64 {
        self.values[v.0 as usize][t]
    }

    fn set(&mut self, v: ValueId, t: usize, bits: u64) {
        let n = self.nthreads;
        let slot = &mut self.values[v.0 as usize];
        if slot.is_empty() {
            slot.resize(n, 0);
        }
        slot[t] = bits;
    }

    fn region(&mut self, r: &'a Region, active: &[usize]) -> R {
        for inst in &r.insts {
            self.inst(inst, active)?;
        }
        Ok(())
    }

    fn lanes(&mut self, active: &[usize], result: ValueId, f: impl Fn(&Self, usize) -> Result<u64, String>) -> R {
        for &t in active {
            let bits = f(self, t).map_err(|m| self.err(m, Some(t)))?;
            self.set(result, t, bits);
        }
        Ok(())
    }

    fn index(&self, v: ValueId, t: usize) -> i64 {
        self.get(v, t) as u32 as i32 as i64
    }

    fn buffer_bytes(&self, b: BufferId) -> usize {
        let d = self.ir.buffer(b);
        match d.space {
            MemSpace::Global => self.globals[b.0 as usize].as_ref().expect("global buffers are bound").len(),
            _ => d.len.expect("shared and local buffers have a size") * d.elem.bytes(),
        }
    }

    /// Bounds- and alignment-check `n` elements of `size` bytes at element
    /// `idx`, and return the byte offset.
    #[allow(clippy::too_many_arguments)]
    fn checked_range(
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
        if idx < 0 || (idx as usize + n) * size > bytes {
            return Err(self.err(
                format!(
                    "{what} of {n} element(s) at index {idx} is out of bounds for `{}` ({} elements)",
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

    fn read(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R<u64> {
        match self.ir.buffer(b).space {
            MemSpace::Global => {
                Ok(read_le(&self.globals[b.0 as usize].as_ref().expect("global buffers are bound")[byte..byte + size]))
            }
            MemSpace::Shared => self.smem_read(b, byte, size, t),
            MemSpace::Local => self.local_read(b, byte, size, t),
        }
    }

    fn write(&mut self, b: BufferId, byte: usize, size: usize, t: usize, bits: u64) -> R {
        match self.ir.buffer(b).space {
            MemSpace::Global => {
                write_le(
                    &mut self.globals[b.0 as usize].as_mut().expect("global buffers are bound")[byte..byte + size],
                    bits,
                );
                Ok(())
            }
            MemSpace::Shared => {
                self.smem_claim_write(b, byte, size, t, false)?;
                let block = self.block_rank(t);
                write_le(&mut self.shared[b.0 as usize][block].data[byte..byte + size], bits);
                Ok(())
            }
            MemSpace::Local => {
                let mem = self.locals[b.0 as usize].as_mut().expect("local buffers are allocated");
                let base = t * mem.per_thread + byte;
                write_le(&mut mem.data[base..base + size], bits);
                mem.written[base..base + size].fill(true);
                Ok(())
            }
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

    fn smem_read(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R<u64> {
        let warp = self.warp_of(t) as usize;
        let block = self.block_rank(t);
        let w = self.warps_per_block;
        let opts = self.opts;
        let d = self.ir.buffer(b);
        let (name, elem) = (&d.name, d.elem.bytes());
        let clock = &self.clocks[block * w + warp];
        let sh = &mut self.shared[b.0 as usize][block];
        let mut failure = None;
        let mut inflight = None;
        for i in byte..byte + size {
            let m = &sh.meta[i];
            let e = i / elem;
            failure = if m.inflight > 0 {
                inflight = Some(i);
                break;
            } else if opts.detect_uninitialized && !m.written {
                Some(format!("reads {name}[{e}], which was never written"))
            } else if opts.detect_races && m.w_warp as usize != warp && m.w_clock > clock[m.w_warp as usize] {
                Some(if m.w_warp as usize == w {
                    format!(
                        "data race: warp {warp} reads {name}[{e}], which a bulk copy wrote, without having waited \
                         for the mbarrier phase the copy completed"
                    )
                } else {
                    format!(
                        "data race: warp {warp} reads {name}[{e}], written by warp {} since the last barrier (missing sync_threads?)",
                        m.w_warp
                    )
                })
            } else {
                None
            };
            if failure.is_some() {
                break;
            }
            if opts.detect_races {
                sh.reads[i * w + warp] = clock[warp];
            }
        }
        if let Some(i) = inflight {
            return Err(self.err(self.inflight_message("reads", b, block, i), Some(t)));
        }
        if let Some(msg) = failure {
            return Err(self.err(msg, Some(t)));
        }
        Ok(read_le(&self.shared[b.0 as usize][block].data[byte..byte + size]))
    }

    /// The report for an access (`verb`) to byte `i` of shared buffer `b` of
    /// `block` while an asynchronous copy is still writing it.
    fn inflight_message(&self, verb: &str, b: BufferId, block: usize, i: usize) -> String {
        let d = self.ir.buffer(b);
        let e = i / d.elem.bytes();
        if self.bulk_copy_writing(b, block, i) {
            format!(
                "{verb} {}[{e}] while a bulk copy is still writing it (wait on the copy's mbarrier until its phase is over)",
                d.name
            )
        } else if verb == "reads" {
            format!("reads {}[{e}] while a cp.async is still writing it (missing cp_async_wait_group?)", d.name)
        } else {
            format!("{verb} {}[{e}] while a cp.async is still writing it", d.name)
        }
    }

    /// Check (and record) a write by `t` to `size` bytes of shared memory.
    fn smem_claim_write(&mut self, b: BufferId, byte: usize, size: usize, t: usize, async_issue: bool) -> R {
        let warp = self.warp_of(t) as usize;
        let block = self.block_rank(t);
        let w = self.warps_per_block;
        let opts = self.opts;
        let d = self.ir.buffer(b);
        let (name, elem) = (&d.name, d.elem.bytes());
        let clock = &self.clocks[block * w + warp];
        let sh = &mut self.shared[b.0 as usize][block];
        let mut failure = None;
        let mut inflight = None;
        for i in byte..byte + size {
            let m = &mut sh.meta[i];
            let e = i / elem;
            let reader = || (0..w).find(|&r| r != warp && sh.reads[i * w + r] > clock[r]);
            failure = if m.inflight > 0 {
                inflight = Some(i);
                break;
            } else if opts.detect_races && m.w_warp as usize != warp && m.w_clock > clock[m.w_warp as usize] {
                Some(if m.w_warp as usize == w {
                    format!(
                        "data race: warp {warp} overwrites {name}[{e}], which a bulk copy wrote, without having \
                         waited for the mbarrier phase the copy completed"
                    )
                } else {
                    format!("data race: warps {} and {warp} both write {name}[{e}] between barriers", m.w_warp)
                })
            } else {
                opts.detect_races.then(reader).flatten().map(|other| {
                    format!(
                        "data race: warp {warp} overwrites {name}[{e}], which warp {other} read since the last barrier (missing sync_threads?)"
                    )
                })
            };
            if failure.is_some() {
                break;
            }
            if async_issue {
                m.inflight += 1;
            } else {
                m.w_clock = clock[warp];
                m.w_warp = warp as u16;
                m.written = true;
                if opts.detect_races {
                    sh.reads[i * w..(i + 1) * w].fill(0);
                }
            }
        }
        if let Some(i) = inflight {
            return Err(self.err(self.inflight_message("writes", b, block, i), Some(t)));
        }
        match failure {
            Some(msg) => Err(self.err(msg, Some(t))),
            None => Ok(()),
        }
    }

    /// Record that bytes of shared memory an asynchronous copy claimed have
    /// landed, written by `writer` (a warp of `block`, or its bulk-copy
    /// agent, `warps_per_block`) at its own time `clock`.
    fn smem_land(&mut self, b: BufferId, block: usize, range: std::ops::Range<usize>, writer: usize, clock: u32) {
        let w = self.warps_per_block;
        let sh = &mut self.shared[b.0 as usize][block];
        for i in range {
            let m = &mut sh.meta[i];
            m.inflight -= 1;
            m.written = true;
            m.w_warp = writer as u16;
            m.w_clock = clock;
            if !sh.reads.is_empty() {
                sh.reads[i * w..(i + 1) * w].fill(0);
            }
        }
    }

    /// An atomic update. It is exempt from the race checks but may not touch
    /// memory a `cp.async` is writing.
    fn atomic_update(&mut self, b: BufferId, byte: usize, size: usize, t: usize, f: impl FnOnce(u64) -> u64) -> R<u64> {
        let old = match self.ir.buffer(b).space {
            MemSpace::Global => self.read(b, byte, size, t)?,
            _ => {
                let block = self.block_rank(t);
                let d = self.ir.buffer(b);
                let (name, elem) = (&d.name, d.elem.bytes());
                let detect = self.opts.detect_uninitialized;
                let sh = &self.shared[b.0 as usize][block];
                let failure = (byte..byte + size).find_map(|i| {
                    let m = &sh.meta[i];
                    if m.inflight > 0 {
                        Some(self.inflight_message("atomic on", b, block, i))
                    } else if detect && !m.written {
                        Some(format!("atomic on {name}[{}], which was never written", i / elem))
                    } else {
                        None
                    }
                });
                if let Some(msg) = failure {
                    return Err(self.err(msg, Some(t)));
                }
                read_le(&sh.data[byte..byte + size])
            }
        };
        let new = f(old);
        match self.ir.buffer(b).space {
            MemSpace::Global => self.write(b, byte, size, t, new)?,
            _ => {
                let block = self.block_rank(t);
                write_le(&mut self.shared[b.0 as usize][block].data[byte..byte + size], new);
            }
        }
        Ok(old)
    }

    /// Account one instruction's accesses: `(thread, byte offset, bytes)`.
    fn account(&mut self, space: MemSpace, accesses: &[(usize, usize, usize)], write: bool) {
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

    fn shared_phases(&mut self, group: &[(usize, usize, usize)], width: usize, lanes_per_phase: usize) {
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
    fn full_warps(&self, active: &[usize], what: &str) -> R<Vec<usize>> {
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
    fn warps_of(&self, active: &[usize]) -> Vec<(usize, Vec<usize>)> {
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

    fn special(&self, reg: SpecialReg, t: usize) -> u64 {
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
            SpecialReg::ClusterIdx(d) => (block[d.index()] / cfg.cluster[d.index()]) as u64,
            SpecialReg::ClusterDim(d) => (cfg.grid[d.index()] / cfg.cluster[d.index()]) as u64,
            SpecialReg::BlockInClusterIdx(d) => (block[d.index()] % cfg.cluster[d.index()]) as u64,
            SpecialReg::BlockInClusterDim(d) => cfg.cluster[d.index()] as u64,
        }
    }

    fn inst(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        if active.is_empty() {
            return Ok(());
        }
        self.loc = Some(inst.loc);
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
        let ir = self.ir;
        match &inst.op {
            Op::Const { result, value } => {
                let bits = value.bits();
                self.lanes(active, *result, |_, _| Ok(bits))?;
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
                        // In f64, then to f16: the f64 rounding cannot
                        // change the f16 result (a sum near an f16 midpoint
                        // is exact in f64), so this is fma.rn.f16's single
                        // rounding.
                        DType::F16 => f64_to_f16(to_f64(ty, x).mul_add(to_f64(ty, y), to_f64(ty, z)).to_bits()),
                        // BFloat16 is computed in f32 and rounded (as the
                        // emitted code does).
                        _ => from_f32(ty, to_f32(ty, x).mul_add(to_f32(ty, y), to_f32(ty, z))),
                    };
                    Ok(ptx_nan(ty, &[x, y, z], r))
                })?;
            }
            Op::FloatInst { result, func, args, mode, ftz, nan } => {
                let ty = ir.dtype(args[0]);
                self.lanes(active, *result, |m, t| {
                    let mut operands = [0u64; 3];
                    for (k, a) in args.iter().enumerate() {
                        operands[k] = m.get(*a, t);
                    }
                    Ok(super::float::float_inst(*func, *mode, *ftz, *nan, ty, &operands[..args.len()]))
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
            Op::Load { results, buffer, index } => {
                let d = ir.buffer(*buffer);
                let n = results.len();
                let size = ir.dtype(results[0]).bytes();
                let bytes = n * size;
                let mut accesses = Vec::with_capacity(active.len());
                for &t in active {
                    let start = self.checked_range(*buffer, self.index(*index, t), n, size, bytes, t, "load")?;
                    for (k, r) in results.iter().enumerate() {
                        let bits = self.read(*buffer, start + k * size, size, t)?;
                        self.set(*r, t, bits);
                    }
                    accesses.push((t, start, bytes));
                }
                self.account(d.space, &accesses, false);
            }
            Op::Store { values, buffer, index } => {
                let d = ir.buffer(*buffer);
                let n = values.len();
                let size = ir.dtype(values[0]).bytes();
                let bytes = n * size;
                let mut accesses = Vec::with_capacity(active.len());
                for &t in active {
                    let start = self.checked_range(*buffer, self.index(*index, t), n, size, bytes, t, "store")?;
                    for (k, v) in values.iter().enumerate() {
                        let bits = self.get(*v, t);
                        self.write(*buffer, start + k * size, size, t, bits)?;
                    }
                    accesses.push((t, start, bytes));
                }
                self.account(d.space, &accesses, true);
            }
            Op::CpAsync { src, src_index, dst, dst_index, elem, elems, pred, .. } => {
                let n = *elems as usize;
                let size = elem.bytes();
                let bytes = n * size;
                let mut accesses = Vec::new();
                for &t in active {
                    let si =
                        self.checked_range(*src, self.index(*src_index, t), n, size, bytes, t, "cp.async source read")?;
                    let di = self.checked_range(
                        *dst,
                        self.index(*dst_index, t),
                        n,
                        size,
                        bytes,
                        t,
                        "cp.async destination write",
                    )?;
                    let fill = pred.is_some_and(|p| self.get(p, t) == 0);
                    let data = if fill {
                        vec![0; bytes]
                    } else {
                        self.globals[src.0 as usize].as_ref().expect("global buffers are bound")[si..si + bytes]
                            .to_vec()
                    };
                    self.smem_claim_write(*dst, di, bytes, t, true)?;
                    if !fill {
                        accesses.push((t, si, bytes));
                    }
                    let (warp, block) = (self.warp_of(t), self.block_rank(t));
                    self.uncommitted[t].push(Pending { dst: *dst, block, dst_byte: di, data, warp });
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
                        for copy in group {
                            let range = copy.dst_byte..copy.dst_byte + copy.data.len();
                            let sh = &mut self.shared[copy.dst.0 as usize][copy.block];
                            sh.data[range.clone()].copy_from_slice(&copy.data);
                            let warp = copy.warp as usize;
                            let clock = self.clocks[copy.block * self.warps_per_block + warp][warp];
                            self.smem_land(copy.dst, copy.block, range, warp, clock);
                        }
                    }
                }
            }
            Op::LdMatrix { results, buffer, index, num, trans } => {
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
            Op::Mma { kind, d, a, b, c } => self.mma(*kind, d, a, b, c, active)?,
            Op::Barrier => {
                for start in (0..self.nthreads).step_by(self.threads_per_block) {
                    let n = active.iter().filter(|&&t| t >= start && t < start + self.threads_per_block).count();
                    if n != 0 && n != self.threads_per_block {
                        return Err(self.err(
                            format!(
                                "sync_threads() reached by {n} of {} threads (barriers must not be inside divergent control flow)",
                                self.threads_per_block
                            ),
                            None,
                        ));
                    }
                    if n != 0 {
                        let block = start / self.threads_per_block;
                        let pending = &self.named[block][0];
                        if !pending.arrived.is_empty() {
                            return Err(self.err(
                                format!(
                                    "sync_threads() uses barrier 0 while warps {} have arrived at it with barrier_arrive and it has not completed",
                                    warp_list(&pending.arrived)
                                ),
                                None,
                            ));
                        }
                        let w = self.warps_per_block;
                        let all: Vec<usize> = (0..w).collect();
                        self.synchronize(block, &all, &all, &[]);
                        self.report.barriers += 1;
                    }
                }
            }
            Op::NamedBarrier { kind, id, threads } => self.named_barrier(*kind, *id, *threads, active)?,
            Op::WarpBarrier { mask } => {
                for (base, lanes) in self.warps_of(active) {
                    let wanted = self.get(*mask, base + lanes[0]) as u32;
                    if let Some(l) =
                        (0..32).find(|l| wanted >> l & 1 == 1 && !lanes.contains(l) && l < &self.warp_size(base))
                    {
                        return Err(
                            self.err(format!("sync_warp() is waiting for lane {l}, which is not active"), Some(base))
                        );
                    }
                }
            }
            Op::Shuffle { result, kind, value, offset, mask } => {
                let mut out = Vec::with_capacity(active.len());
                for &t in active {
                    let lane = self.lane_of(t);
                    let base = t - lane;
                    let off = self.get(*offset, t) as usize & 31;
                    let members = self.get(*mask, t) as u32;
                    let source = match kind {
                        ShuffleKind::Idx => Some(off),
                        ShuffleKind::Up => lane.checked_sub(off),
                        ShuffleKind::Down => Some(lane + off).filter(|&l| l < 32),
                        ShuffleKind::Bfly => Some(lane ^ off),
                    };
                    let bits = match source {
                        None => self.get(*value, t),
                        Some(s) if members >> s & 1 == 1 && active.binary_search(&(base + s)).is_ok() => {
                            self.get(*value, base + s)
                        }
                        Some(s) => {
                            return Err(self.err(format!("shuffle reads lane {s}, which is not active"), Some(t)));
                        }
                    };
                    out.push((t, bits));
                }
                for (t, bits) in out {
                    self.set(*result, t, bits);
                }
            }
            Op::Vote { result, kind, pred, mask } => {
                for (base, lanes) in self.warps_of(active) {
                    let members = self.get(*mask, base + lanes[0]) as u32;
                    let votes: Vec<(usize, bool)> = lanes
                        .iter()
                        .filter(|&&l| members >> l & 1 == 1)
                        .map(|&l| (l, self.get(*pred, base + l) != 0))
                        .collect();
                    let bits = match kind {
                        VoteKind::All => votes.iter().all(|v| v.1) as u64,
                        VoteKind::Any => votes.iter().any(|v| v.1) as u64,
                        VoteKind::Uni => votes.windows(2).all(|w| w[0].1 == w[1].1) as u64,
                        VoteKind::Ballot => votes.iter().filter(|v| v.1).fold(0u64, |acc, v| acc | 1 << v.0),
                    };
                    for &l in &lanes {
                        self.set(*result, base + l, bits);
                    }
                }
            }
            Op::MatchAny { result, value, mask } => {
                for (base, lanes) in self.warps_of(active) {
                    let members = self.get(*mask, base + lanes[0]) as u32;
                    for &l in &lanes {
                        let mine = self.get(*value, base + l);
                        let same = lanes
                            .iter()
                            .filter(|&&o| members >> o & 1 == 1 && self.get(*value, base + o) == mine)
                            .fold(0u64, |acc, &o| acc | 1 << o);
                        self.set(*result, base + l, same);
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
            Op::Atomic { result, kind, buffer, index, value, compare } => {
                let ty = ir.dtype(*value);
                let size = ty.bytes();
                for &t in active {
                    let byte = self.checked_range(*buffer, self.index(*index, t), 1, size, size, t, "atomic")?;
                    let (operand, expected) = (self.get(*value, t), compare.map(|c| self.get(c, t)));
                    let old = self.atomic_update(*buffer, byte, size, t, |old| match kind {
                        AtomicKind::Add => mask(old.wrapping_add(operand), ty.bits()),
                        AtomicKind::And => old & operand,
                        AtomicKind::Or => old | operand,
                        AtomicKind::Xor => old ^ operand,
                        AtomicKind::Min | AtomicKind::Max => {
                            let less = compare_less(ty, operand, old);
                            if less == (*kind == AtomicKind::Min) { operand } else { old }
                        }
                        AtomicKind::Exchange => operand,
                        AtomicKind::CompareExchange => {
                            if Some(old) == expected {
                                operand
                            } else {
                                old
                            }
                        }
                    })?;
                    self.set(*result, t, old);
                }
            }
            Op::MbarrierInit { buffer, index, count } => {
                for &t in active {
                    let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "mbarrier.init")?;
                    let arrivals = self.get(*count, t) as u32;
                    if !(1..1 << 20).contains(&arrivals) {
                        return Err(self.err(
                            format!("an mbarrier count of {} is outside 1 to 2^20 - 1", arrivals as i32),
                            Some(t),
                        ));
                    }
                    let w = self.warps_per_block;
                    let mbarrier = Mbarrier {
                        arrivals_per_phase: arrivals,
                        arrivals_left: arrivals,
                        tx_count: 0,
                        copies: Vec::new(),
                        phase: 0,
                        release: vec![0; w + 1],
                        completed: vec![0; w + 1],
                        seen: vec![Vec::new(); w],
                    };
                    self.mbarriers.insert((self.block_rank(t), buffer.0, byte), mbarrier);
                }
            }
            Op::MbarrierArrive { buffer, index, count, expect_tx, peer, relaxed, .. } => {
                let w = self.warps_per_block;
                // Warps of the mbarrier's own block that arrived (released).
                let mut released: Vec<(usize, usize)> = Vec::new();
                for &t in active {
                    let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "mbarrier.arrive")?;
                    let block = self.mbarrier_block(*peer, t, "mbarrier.arrive")?;
                    let n = count.map_or(1, |c| self.get(c, t) as u32);
                    let tx = expect_tx.map_or(0, |v| self.get(v, t) as u32 as u64);
                    let warp = self.warp_of(t) as usize;
                    // Only arrivals on the warp's own block's mbarrier take
                    // part in its race detection.
                    let release = block == self.block_rank(t) && !relaxed;
                    let clock = &self.clocks[self.block_rank(t) * w + warp];
                    let Some(state) = self.mbarriers.get_mut(&(block, buffer.0, byte)) else {
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
                    if release {
                        join(&mut state.release, clock);
                    }
                    // `.expect_tx` raises the tx-count before the arrival.
                    state.tx_count += tx;
                    state.arrivals_left -= n;
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
                    if release && !released.contains(&(block, warp)) {
                        released.push((block, warp));
                    }
                }
                // Arriving warps move on to a new time of their own.
                for (block, warp) in released {
                    self.clocks[block * w + warp][warp] += 1;
                }
            }
            Op::MbarrierExpectTx { buffer, index, bytes, peer, .. } => {
                for &t in active {
                    let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "mbarrier.expect_tx")?;
                    let block = self.mbarrier_block(*peer, t, "mbarrier.expect_tx")?;
                    let tx = self.get(*bytes, t) as u32 as u64;
                    let Some(state) = self.mbarriers.get_mut(&(block, buffer.0, byte)) else {
                        return Err(self.err("mbarrier.expect_tx on an mbarrier that was never initialised", Some(t)));
                    };
                    state.tx_count += tx;
                }
            }
            Op::MbarrierWait { result, buffer, index, parity, kind } => {
                let what = match kind {
                    MbarrierWaitKind::Try => "mbarrier.try_wait",
                    MbarrierWaitKind::Test => "mbarrier.test_wait",
                };
                let w = self.warps_per_block;
                for &t in active {
                    let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, what)?;
                    let block = self.block_rank(t);
                    let key = (block, buffer.0, byte);
                    self.land_bulk_copies(key, t)?;
                    let warp = self.warp_of(t) as usize;
                    let parity = self.get(*parity, t) & 1;
                    let Some(state) = self.mbarriers.get(&key) else {
                        return Err(self.err(format!("{what} on an mbarrier that was never initialised"), Some(t)));
                    };
                    // The parity names the current phase, which is running,
                    // or the one before it, which is over. Waiting for phase
                    // `asked` only means that if the warp knows phase
                    // `asked - 1` is over: otherwise the mbarrier may still be
                    // in that phase on hardware, where the parity names the
                    // phase before it and the wait succeeds at once.
                    let over = parity != state.phase & 1;
                    let asked = if over { state.phase.saturating_sub(1) } else { state.phase };
                    if state.known_phases(&self.clocks[block * w + warp]) < asked {
                        return Err(self.err(
                            format!(
                                "{what} with parity {parity} asks about phase {asked} of the mbarrier, but warp {warp} \
                                 has not seen phase {} complete: on hardware the mbarrier may still be in that phase, \
                                 where parity {parity} refers to the phase before it, which is over, so the wait succeeds \
                                 at once (lockstep simulation hides this). Wait for phase {} first",
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
                        let clock = &mut self.clocks[block * w + warp];
                        join(clock, &state.completed);
                        let (now, phases) = (clock[warp], state.phase);
                        let seen = &mut state.seen[warp];
                        match seen.last_mut() {
                            Some(last) if last.1 >= phases => {}
                            Some(last) if last.0 == now => last.1 = phases,
                            _ => seen.push((now, phases)),
                        }
                    }
                    self.set(*result, t, over as u64);
                }
            }
            Op::FenceMbarrierInit => {}
            Op::CpAsyncBulk { src, src_index, dst, dst_index, elem, bytes, mbarrier, mbarrier_index } => {
                let size = elem.bytes();
                let mut accesses = Vec::new();
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
                    self.smem_claim_write(*dst, di, count, t, true)?;
                    let data = self.globals[src.0 as usize].as_ref().expect("global buffers are bound")[si..si + count]
                        .to_vec();
                    let (warp, block) = (self.warp_of(t), self.block_rank(t));
                    let Some(state) = self.mbarriers.get_mut(&(block, mbarrier.0, byte)) else {
                        return Err(
                            self.err("cp.async.bulk reports to an mbarrier that was never initialised", Some(t))
                        );
                    };
                    state.copies.push(Pending { dst: *dst, block, dst_byte: di, data, warp });
                    accesses.push((t, si, count));
                }
                for access in accesses {
                    self.account(MemSpace::Global, &[access], false);
                }
            }
            Op::Fence { .. } => {}
            Op::Printf { format, args } => {
                for &t in active {
                    let values: Vec<(DType, u64)> = args.iter().map(|a| (ir.dtype(*a), self.get(*a, t))).collect();
                    self.report.printed.push(format_printf(format, &values));
                }
            }
            Op::InlinePtx { template, inputs, outputs, sim, .. } => {
                let Some(f) = sim else {
                    return Err(self.err(
                        format!("inline PTX `{template}` cannot be simulated; use inline_ptx_with_sim to describe it"),
                        None,
                    ));
                };
                for &t in active {
                    let bits: Vec<u64> = inputs.iter().map(|v| self.get(*v, t)).collect();
                    let produced = f(&bits);
                    if produced.len() != outputs.len() {
                        return Err(self.err(
                            format!("inline PTX simulation returned {} values, not {}", produced.len(), outputs.len()),
                            Some(t),
                        ));
                    }
                    for (o, bits) in outputs.iter().zip(produced) {
                        self.set(*o, t, mask(bits, ir.dtype(*o).bits()));
                    }
                }
            }
            Op::If { cond, then_region, else_region, results } => {
                let (taken, not_taken): (Vec<usize>, Vec<usize>) =
                    active.iter().partition(|&&t| self.get(*cond, t) != 0);
                self.region(then_region, &taken)?;
                self.region(else_region, &not_taken)?;
                for (lanes, region) in [(&taken, then_region), (&not_taken, else_region)] {
                    for &t in lanes.iter() {
                        for (r, y) in results.iter().zip(&region.yields) {
                            let bits = self.get(*y, t);
                            self.set(*r, t, bits);
                        }
                    }
                }
            }
            // The unroll and vectorize hints do not change what a loop
            // computes; `at_least_once` is a promise the program makes.
            Op::For { iv, lower, upper, step, init, args, body, results, opts } => {
                let as_int = |m: &Self, v: ValueId, t: usize| m.get(v, t) as u32 as i32 as i64;
                for &t in active {
                    let s = self.get(*step, t) as u32 as i32;
                    if s <= 0 {
                        return Err(self.err(format!("loop step must be positive, got {s}"), Some(t)));
                    }
                    let (lo, hi) = (as_int(self, *lower, t), as_int(self, *upper, t));
                    if opts.at_least_once && lo >= hi {
                        return Err(self.err(
                            format!(
                                "the loop is marked at_least_once, but runs zero times in this thread \
                                 ({lo}..{hi}); on the GPU the compiler may assume it runs"
                            ),
                            Some(t),
                        ));
                    }
                    for (a, x) in args.iter().zip(init) {
                        let bits = self.get(*x, t);
                        self.set(*a, t, bits);
                    }
                }
                let mut ivs: Vec<i64> = vec![0; self.nthreads];
                for &t in active {
                    ivs[t] = as_int(self, *lower, t);
                }
                let mut running: Vec<usize> =
                    active.iter().copied().filter(|&t| ivs[t] < as_int(self, *upper, t)).collect();
                while !running.is_empty() {
                    for &t in &running {
                        self.set(*iv, t, ivs[t] as i32 as u32 as u64);
                    }
                    self.region(body, &running)?;
                    self.loc = Some(inst.loc);
                    for &t in &running {
                        let next: Vec<u64> = body.yields.iter().map(|y| self.get(*y, t)).collect();
                        for (a, bits) in args.iter().zip(next) {
                            self.set(*a, t, bits);
                        }
                        ivs[t] += as_int(self, *step, t);
                    }
                    running.retain(|&t| ivs[t] < as_int(self, *upper, t));
                }
                for &t in active {
                    for (r, a) in results.iter().zip(args) {
                        let bits = self.get(*a, t);
                        self.set(*r, t, bits);
                    }
                }
            }
            Op::While { init, cond_args, cond, body_args, body, results } => {
                for &t in active {
                    for (a, x) in cond_args.iter().zip(init) {
                        let bits = self.get(*x, t);
                        self.set(*a, t, bits);
                    }
                }
                let mut running: Vec<usize> = active.to_vec();
                let mut iterations = 0u64;
                while !running.is_empty() {
                    iterations += 1;
                    if iterations > self.opts.max_while_iterations {
                        return Err(self.err("while_carry did not terminate", Some(running[0])));
                    }
                    self.region(cond, &running)?;
                    self.loc = Some(inst.loc);
                    let c = cond.yields[0];
                    let (go, stop): (Vec<usize>, Vec<usize>) = running.iter().partition(|&&t| self.get(c, t) != 0);
                    for &t in &stop {
                        for (r, a) in results.iter().zip(cond_args) {
                            let bits = self.get(*a, t);
                            self.set(*r, t, bits);
                        }
                    }
                    for &t in &go {
                        for (b, a) in body_args.iter().zip(cond_args) {
                            let bits = self.get(*a, t);
                            self.set(*b, t, bits);
                        }
                    }
                    self.region(body, &go)?;
                    self.loc = Some(inst.loc);
                    for &t in &go {
                        let next: Vec<u64> = body.yields.iter().map(|y| self.get(*y, t)).collect();
                        for (a, bits) in cond_args.iter().zip(next) {
                            self.set(*a, t, bits);
                        }
                    }
                    running = go;
                }
            }
        }
        Ok(())
    }

    /// Land the bulk copies reporting to one mbarrier, as late as the
    /// simulator can: when thread `t` waits on it after every arrival of
    /// the phase is in. (Until then the phase cannot complete wherever they
    /// land, and any access to their destination is reported.) Their bytes
    /// count down the phase's tx-count, which must hold them all: otherwise
    /// the phase would complete with some still in flight.
    fn land_bulk_copies(&mut self, key: (usize, u32, usize), t: usize) -> R {
        let Some(state) = self.mbarriers.get_mut(&key) else {
            return Ok(());
        };
        if state.arrivals_left > 0 || state.copies.is_empty() {
            return Ok(());
        }
        let landed: u64 = state.copies.iter().map(|c| c.data.len() as u64).sum();
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
        state.tx_count -= landed;
        let copies = std::mem::take(&mut state.copies);
        let (block, w) = (key.0, self.warps_per_block);
        for copy in copies {
            let range = copy.dst_byte..copy.dst_byte + copy.data.len();
            self.shared[copy.dst.0 as usize][copy.block].data[range.clone()].copy_from_slice(&copy.data);
            // The block's bulk-copy agent writes it at a new time of its
            // own, which the phase's completion publishes to its waiters.
            self.bulk_time[block] += 1;
            let time = self.bulk_time[block];
            self.smem_land(copy.dst, copy.block, range, w, time);
            let state = self.mbarriers.get_mut(&key).expect("the mbarrier exists");
            state.release[w] = time;
        }
        self.mbarriers.get_mut(&key).expect("the mbarrier exists").complete_if_done();
        Ok(())
    }

    /// Whether byte `byte` of shared buffer `b` of `block` is the destination
    /// of a bulk copy still in flight (rather than of a `cp.async`).
    fn bulk_copy_writing(&self, b: BufferId, block: usize, byte: usize) -> bool {
        self.mbarriers
            .values()
            .flat_map(|m| &m.copies)
            .any(|c| c.dst == b && c.block == block && (c.dst_byte..c.dst_byte + c.data.len()).contains(&byte))
    }

    /// The block whose mbarrier an operation by thread `t` addresses: its
    /// own, or CTA `peer` of its cluster.
    fn mbarrier_block(&self, peer: Option<ValueId>, t: usize, what: &str) -> R<usize> {
        let Some(peer) = peer else { return Ok(self.block_rank(t)) };
        let rank = self.get(peer, t) as u32 as i32;
        if rank < 0 || rank as usize >= self.blocks.len() {
            return Err(self.err(format!("{what} on CTA {rank} of a cluster of {} CTA(s)", self.blocks.len()), Some(t)));
        }
        Ok(rank as usize)
    }

    /// The warps of `block` in `releasing` publish their clocks (joined into
    /// `release` as well); those in `acquiring` then see all of it, and every
    /// warp involved moves on to a new time of its own.
    fn synchronize(&mut self, block: usize, releasing: &[usize], acquiring: &[usize], release: &[u32]) {
        let w = self.warps_per_block;
        let mut joined = vec![0; w + 1];
        join(&mut joined, release);
        for &warp in releasing {
            join(&mut joined, &self.clocks[block * w + warp]);
        }
        for &warp in acquiring {
            join(&mut self.clocks[block * w + warp], &joined);
        }
        for &warp in releasing.iter().chain(acquiring) {
            let clock = &mut self.clocks[block * w + warp];
            clock[warp] = clock[warp].max(joined[warp]) + 1;
        }
    }

    /// `bar.sync` / `bar.arrive` executed by the threads in `active`.
    fn named_barrier(&mut self, kind: BarrierKind, id: ValueId, threads: Option<ValueId>, active: &[usize]) -> R {
        let what = match kind {
            BarrierKind::Sync => "barrier",
            BarrierKind::Arrive => "barrier_arrive",
            BarrierKind::ArriveUnaligned => "barrier_arrive (unaligned)",
        };
        // (block, barrier id) -> (thread count, arriving warps)
        let mut arrivals: BTreeMap<(usize, usize), (Option<usize>, Vec<usize>)> = BTreeMap::new();
        for (base, lanes) in self.warps_of(active) {
            let warp = self.warp_of(base) as usize;
            if lanes.len() != self.warp_size(base) {
                return Err(self.err(
                    format!(
                        "{what} executed by {} of the {} lanes of warp {warp} (each participating warp must reach it as a whole)",
                        lanes.len(),
                        self.warp_size(base)
                    ),
                    Some(base + lanes[0]),
                ));
            }
            let uniform = |v: ValueId, name: &str| -> R<i32> {
                let first = self.get(v, base + lanes[0]) as u32 as i32;
                match lanes.iter().find(|&&l| self.get(v, base + l) as u32 as i32 != first) {
                    Some(&l) => Err(self.err(
                        format!(
                            "{what}: the lanes of warp {warp} disagree on the {name} ({first} and {})",
                            self.get(v, base + l) as u32 as i32
                        ),
                        Some(base + l),
                    )),
                    None => Ok(first),
                }
            };
            let b = uniform(id, "barrier id")?;
            if !(0..16).contains(&b) {
                return Err(self.err(format!("{what}: barrier id {b} is out of range (0 to 15)"), Some(base)));
            }
            let count = match threads {
                Some(n) => {
                    let n = uniform(n, "thread count")?;
                    let limit = self.warps_per_block * 32;
                    if n <= 0 || n % 32 != 0 || n as usize > limit {
                        return Err(self.err(
                            format!("{what}: a thread count of {n} is not a positive multiple of 32 up to the block's {limit}"),
                            Some(base),
                        ));
                    }
                    Some(n as usize)
                }
                None => None,
            };
            let entry = arrivals.entry((self.block_rank(base), b as usize)).or_insert((count, Vec::new()));
            if entry.0 != count {
                return Err(self.err(
                    format!("{what}: warps arriving at barrier {b} together disagree on its thread count"),
                    Some(base),
                ));
            }
            entry.1.push(warp);
        }
        let w = self.warps_per_block;
        for ((block, b), (count, warps)) in arrivals {
            let expected = count.map_or(w, |n| n / 32);
            let state = self.named[block][b].clone();
            if !state.arrived.is_empty() && state.expected != expected {
                return Err(self.err(
                    format!(
                        "{what}: barrier {b} is used with {} threads while warps {} wait on it with {}",
                        expected * 32,
                        warp_list(&state.arrived),
                        state.expected * 32
                    ),
                    None,
                ));
            }
            if let Some(&again) = warps.iter().find(|w| state.arrived.contains(w)) {
                return Err(self.err(
                    format!("{what}: warp {again} arrives at barrier {b} again before the barrier has completed"),
                    None,
                ));
            }
            let total = state.arrived.len() + warps.len();
            if total > expected {
                return Err(self.err(
                    format!(
                        "{what}: {total} warps arrive at barrier {b}, which completes after {expected} ({} threads)",
                        expected * 32
                    ),
                    None,
                ));
            }
            let release = if state.release.is_empty() { vec![0; w + 1] } else { state.release.clone() };
            if total == expected {
                // Complete: waiting warps acquire everything the arrivals released.
                let acquiring: &[usize] = if kind == BarrierKind::Sync { &warps } else { &[] };
                self.synchronize(block, &warps, acquiring, &release);
                self.named[block][b] = NamedState::default();
                self.report.barriers += 1;
            } else if kind == BarrierKind::Sync {
                let mut here = state.arrived.clone();
                here.extend(&warps);
                return Err(self.err(
                    format!(
                        "barrier {b} needs {expected} warps ({} threads) but only warps {} have arrived, and they cannot continue: \
                         the simulator runs warps in lockstep, so the other arrivals have to come earlier in program order \
                         (with barrier_arrive); if they never come, the kernel deadlocks on hardware too",
                        expected * 32,
                        warp_list(&here)
                    ),
                    None,
                ));
            } else {
                // Arrive and move on: the barrier keeps what these warps release.
                let mut joined = release;
                for &warp in &warps {
                    join(&mut joined, &self.clocks[block * w + warp]);
                }
                self.synchronize(block, &warps, &[], &[]);
                let state = &mut self.named[block][b];
                state.arrived.extend(&warps);
                state.expected = expected;
                state.release = joined;
            }
        }
        Ok(())
    }

    /// Lanes in the warp starting at thread `base` (the last warp of a block
    /// can be short).
    fn warp_size(&self, base: usize) -> usize {
        (self.threads_per_block - base % self.threads_per_block).min(32)
    }

    fn mma(
        &mut self,
        kind: MmaKind,
        d: &[ValueId],
        a: &[ValueId],
        b: &[ValueId],
        c: &[ValueId],
        active: &[usize],
    ) -> R {
        let bases = self.full_warps(active, "mma.sync")?;
        let (m, n, k) = kind.shape_mnk();
        let (ta, tb, tc) = kind.dtypes();
        // TF32 operands: the tensor core ignores the low 13 mantissa bits.
        let tf32 = |x: f64| -> f64 {
            if kind.is_tf32() { f32::from_bits((x as f32).to_bits() & 0xffff_e000) as f64 } else { x }
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

/// Whether `a < b` for an integer type.
fn compare_less(ty: DType, a: u64, b: u64) -> bool {
    compare(CmpPred::Lt, ty, a, b)
}
