//! The simulator's interpreter.
//!
//! Values are kept as raw bits, zero-extended to 64. Memory is byte
//! addressed, so one allocation can be read as several element types and
//! hazards are tracked per byte. The blocks of a cluster run together;
//! without clusters that is one block at a time.
//!
//! Each warp runs on its own, from a continuation over the structured IR: a
//! stack of frames, each a region, the index of its next instruction, the
//! lanes running it and the state of the instruction that owns the region
//! (an `If`'s other lanes, a loop's induction variables or round count).
//! Region ends are where divergent lanes reconverge. A divergent `If` splits
//! the warp's convergence group ([`WarpState`]) into a `then` group and an
//! `else` group with a join record; each is scheduled on its own
//! (independent thread scheduling), and the `then` group waits at the end of
//! its region for the `else` group to finish. A non-divergent `If` runs its
//! taken region in place. A step executes one instruction for one group, so switching warps
//! costs nothing, and the scheduler ([`Machine::schedule`]) is a plain loop:
//! deterministic, and able to say what every warp is doing when none can
//! continue. `bar.sync`, `sync_threads` and `barrier.cluster.wait` block
//! a warp until the barrier completes, and `mbarrier.try_wait` until its phase completes (or, when
//! nothing else can run, until it gives up and returns false, as PTX allows).
//!
//! Shared-memory races are found with vector clocks at warp granularity:
//! each warp keeps the latest time of every warp of its cluster
//! that it is ordered after. `sync_threads`, named barriers (`bar.sync`
//! acquires what `bar.arrive` released) and mbarriers (a successful wait
//! acquires what the arrivals of the completed phase released) merge
//! clocks; an access races with an earlier one by another warp unless the
//! accessing warp's clock has caught up with it. Bulk copies are written by
//! a per-block agent with a clock entry of its own, which the mbarrier
//! phase they complete publishes to its waiters. So are `cp.async` copies
//! that an mbarrier tracks; one that its thread's `cp_async_wait_group`
//! covers also gets the thread's warp as a second writer, so that the wait
//! orders it, and none of the agent's other writes, before the warp.
//!
//! Clocks span the cluster: a slot per warp of each block, and per block
//! [`AGENT_SLOTS`] slots for agents that write on its behalf. Asynchronous
//! operations ([`AsyncOp`]) share one in-flight model whose completion
//! ([`SimOptions::async_completion`]) lands their writes as their agent's.
//!
//! The clocks also say what a warp knows about an mbarrier's phases (those
//! it, or a warp it is ordered after, has seen complete), which is what
//! makes a parity wait meaningful whatever order the warps would run in on
//! hardware.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::rc::Rc;

use super::float::{f64_to_f16, is_nan, ptx_nan};
use super::{AsyncCompletion, ResidentClusters, Schedule, carry, float, int, printf};
use super::{LaunchConfig, SimError, SimOptions, SimReport};
use crate::ir::{
    AtomicKind, AtomicType, BarrierKind, BarrierRedKind, BinaryOp, BufferDecl, BufferId, BulkCompletion, BulkDirection,
    CmpPred, DType, Inst, KernelIr, LaneMask, LdMatrixForm, Loc, MatchKind, MbarrierWaitKind, MemScopeKind, MemSpace,
    MemoryOrder, MmaKind, Op, ReduxKind, Region, SharedMemoryLayout, ShuffleKind, SpecialReg, StMatrixForm,
    Tcgen05Inst, UnaryOp, ValueId, VoteKind,
};
use crate::tensor::AddressSpace;
use crate::value::{bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits};

type R<T = ()> = Result<T, SimError>;

mod tensormap;
// The tensor-memory model.
mod tmem;
// Tcgen05 MMAs and commits.
mod umma;
// Tcgen05.cp.
mod tmem_cp;
mod wgmma;
// Block-scaled mma.sync and the shared block-scale arithmetic.
mod blockscale;
// Cluster launch control.
mod clc;
// External calls and multimem.
mod distributed;
// The simulator's scheduler.
mod scheduler;
// Barriers.
mod barriers;
// Reports on blocked and exited threads.
mod blocked;
// The memory model.
mod memory;
// The instruction interpreter.
mod inst;
// Asynchronous copies.
mod async_copy;
pub(in crate::sim) use distributed::McGroup;
pub use distributed::ld_reduce_word;

/// Vector-clock slots reserved per block for agents that write memory on
/// a block's behalf: the asynchronous-copy agent (`cp.async.bulk`, and
/// `cp.async` copies an mbarrier tracks) is the first; the others are
/// reserved for TMA, multicast and tensor-core agents.
const AGENT_SLOTS: usize = 4;

/// The asynchronous-copy agent's index among a block's agent slots.
const COPY_AGENT: usize = 0;

/// How many times in a row a warp's `mbarrier.try_wait` may give up with
/// nothing visible happening in between before the simulator reports a
/// deadlock.
const GIVE_UPS_WITHOUT_PROGRESS: u32 = 64;

/// Per-byte shared memory state for hazard detection.
#[derive(Clone, Copy, Default)]
struct Meta {
    written: bool,
    /// The last write was strong (ordered, volatile, MMIO or atomic): it
    /// does not race with strong accesses of the same bytes, and a warp that
    /// read it with such a strong load is ordered after it.
    strong: bool,
    /// The access unit of the last write, if strong ([`unit_code`]).
    unit: u8,
    inflight: u16,
    /// The writer of the last write (the clock slot of a warp, or of the
    /// block's asynchronous agent; see [`Machine::slot_of`]) and its own
    /// clock at the time. Clock 0 is ordered before every access.
    w_slot: u16,
    w_clock: u32,
    /// Whether the agent's last write was a `cp.async` that a
    /// `cp.async.mbarrier.arrive` tracked (rather than a bulk copy).
    tracked_cp_async: bool,
    /// A second writer the last write is ordered after, with its clock (0:
    /// none): the warp that issued a tracked `cp.async`, from the
    /// `cp_async_wait_group` that covered it on. Its other readers order
    /// after it through the mbarrier phase that publishes the agent's time.
    alt_slot: u16,
    alt_clock: u32,
}

impl Meta {
    /// Whether the last write is ordered before an access by the warp in
    /// clock slot `slot`, whose vector clock is `clock`: through its writer,
    /// or through the second writer of a tracked `cp.async` copy.
    fn write_ordered_before(&self, slot: usize, clock: &[u32]) -> bool {
        let seen = |writer: u16, time: u32| writer as usize == slot || time <= clock[writer as usize];
        seen(self.w_slot, self.w_clock) || (self.alt_clock > 0 && seen(self.alt_slot, self.alt_clock))
    }
}

/// A read of a byte since its last write: the reader's clock slot, its own
/// clock at the read (0: no read) and, for strong reads, the access unit
/// ([`unit_code`], or [`MIXED_UNITS`] if the warp read it with several),
/// with [`OBSERVED`] if one of them was morally strong with the last write.
#[derive(Clone, Copy, Default)]
struct ReadEpoch {
    slot: u16,
    unit: u8,
    clock: u32,
}

/// [`ReadEpoch::slot`] of a byte whose reads are in [`Reads::sets`].
const INFLATED: u16 = u16::MAX;

/// The reads of each byte of a shared buffer since its last write, in the
/// style of FastTrack: one read epoch per byte, inflated to a small set of
/// them only while reads that nothing orders against each other exist.
/// Memory stays proportional to the bytes, whatever the number of warps.
#[derive(Default)]
struct Reads {
    epochs: Vec<ReadEpoch>,
    sets: HashMap<usize, Vec<ReadEpoch>>,
}

impl Reads {
    fn new(bytes: usize) -> Reads {
        Reads { epochs: vec![ReadEpoch::default(); bytes], sets: HashMap::new() }
    }

    fn is_empty(&self) -> bool {
        self.epochs.is_empty()
    }

    /// The reads of byte `i`.
    fn of(&self, i: usize) -> &[ReadEpoch] {
        let e = &self.epochs[i];
        if e.slot == INFLATED {
            &self.sets[&i]
        } else if e.clock == 0 {
            &[]
        } else {
            std::slice::from_ref(e)
        }
    }

    /// The read of byte `i` by the warp in `slot`, if any.
    fn by(&self, i: usize, slot: usize) -> Option<ReadEpoch> {
        self.of(i).iter().find(|r| r.slot as usize == slot).copied()
    }

    /// Record `read` of byte `i` by a warp whose clock is `clock`. A weak
    /// read replaces a read that is ordered before it (`forget`), which a
    /// later write that is ordered after this read is ordered after too;
    /// strong reads keep what other warps observed.
    fn record(&mut self, i: usize, read: ReadEpoch, clock: &[u32], forget: bool) {
        let e = self.epochs[i];
        let ordered = |r: &ReadEpoch| r.slot == read.slot || (forget && r.clock <= clock[r.slot as usize]);
        if e.slot == INFLATED {
            let set = self.sets.get_mut(&i).expect("an inflated byte has a read set");
            set.retain(|r| !ordered(r));
            set.push(read);
        } else if e.clock == 0 || ordered(&e) {
            self.epochs[i] = read;
        } else {
            self.sets.insert(i, vec![e, read]);
            self.epochs[i] = ReadEpoch { slot: INFLATED, unit: 0, clock: 1 };
        }
    }

    /// Forget the reads of byte `i` (it was written).
    fn clear(&mut self, i: usize) {
        if self.epochs[i].slot == INFLATED {
            self.sets.remove(&i);
        }
        self.epochs[i] = ReadEpoch::default();
    }
}

struct Shared {
    data: Vec<u8>,
    meta: Vec<Meta>,
    /// The weak reads of each byte since its last write. Empty without race
    /// detection.
    reads: Reads,
    /// The same for strong reads, allocated at the first one.
    strong_reads: Reads,
}

/// What a shared-memory write races with: the earlier write (its byte's
/// state), or a read by a clock slot (with that read's unit if both are
/// strong).
type WriteRace = Result<Meta, (usize, Option<u8>)>;

/// The access unit that a strong access of `size` bytes gives its byte at
/// `offset`: the size and that offset. PTX's memory model treats each
/// element of a vector access as one access, and two strong accesses are
/// morally strong (they do not race, and a release and an acquire
/// synchronise) only if they cover the same bytes, that is, if their units
/// match on every byte; accesses that overlap only partly are a mixed-size
/// data race.
fn unit_code(offset: usize, size: usize) -> u8 {
    debug_assert!(size <= 8 && offset < size, "a strong access of {size} bytes");
    ((offset << 4) | size) as u8
}

/// A warp read a byte strongly with different units since its last write,
/// which matches no write's unit.
const MIXED_UNITS: u8 = 0x7f;

/// A warp's strong read of a byte observed the last write (they were
/// morally strong), which orders the warp's later accesses after it.
const OBSERVED: u8 = 0x80;

/// Why two strong accesses to a byte race: their units ([`unit_code`])
/// differ.
fn mixed_sizes(this: &str, unit: u8, other: &str, other_unit: u8) -> String {
    let sized = |u: u8| {
        let bytes = u & 0xf;
        format!("{} {bytes}-byte", if bytes == 8 { "an" } else { "a" })
    };
    let what = if other_unit == MIXED_UNITS {
        format!("{other}s of other sizes")
    } else {
        format!("{} {other}", sized(other_unit))
    };
    format!(
        "{} {this} and {what} that do not cover the same bytes are not morally strong in PTX's memory model (a \
         mixed-size data race)",
        sized(unit)
    )
}

/// What a strong write that published a clock left on a byte of memory:
/// the publishing warp (by block rank and warp) and the clock.
#[derive(Clone)]
struct Released {
    /// The id of the publishing warp's cluster ([`Machine::cluster_ids`]):
    /// clocks are per cluster, so only its own warps can join the clock.
    cluster: u64,
    block: usize,
    warp: usize,
    clock: Rc<[u32]>,
    /// The access unit of the write ([`unit_code`]), which an acquire must
    /// match to join the clock.
    unit: u8,
}

/// The block-rank part of the key of a global byte in
/// [`Machine::releases`]: global memory is shared by every block.
const GLOBAL: usize = usize::MAX;

/// How a write changes the release information of the bytes it writes.
#[derive(Clone)]
enum Publish {
    /// A weak write: whatever was published there is gone.
    Weak,
    /// A strong write that is not a release: it publishes its warp's clock
    /// at its last fence (a release pattern), or keeps what its own warp
    /// released there before (`st.release [M]; st.relaxed [M]`).
    Strong,
    /// A release store, publishing this clock of its warp.
    Release(Rc<[u32]>),
    /// An atomic read-modify-write: the release sequence continues through
    /// it, and it adds its warp's clock at its last fence.
    Rmw,
    /// A release read-modify-write: it continues the release sequence and
    /// heads one of its own, publishing this clock of its warp.
    ReleaseRmw(Rc<[u32]>),
}

/// Which bytes of a global buffer non-coherent loads read and the kernel
/// wrote: one of each makes the non-coherent loads unsound.
struct NcState {
    read: Vec<bool>,
    written: Vec<bool>,
}

/// The register budgets of a block's warpgroups under `setmaxnreg` (PTX ISA,
/// "setmaxnreg"): each warpgroup starts at [`setmaxnreg_initial_budget`]
/// registers per thread; a decrease returns registers to the block's pool,
/// and an increase takes them from it, waiting until it has them.
#[derive(Clone)]
struct BlockRegs<'a> {
    /// Registers in the pool (for the whole block, not per thread).
    free: u32,
    groups: Vec<WarpgroupRegs<'a>>,
}

#[derive(Clone)]
struct WarpgroupRegs<'a> {
    /// Registers per thread.
    budget: u32,
    pending: Option<PendingRegs<'a>>,
    /// Each warp's clock entry at the last completed `setmaxnreg`: (warp
    /// key, clock slot, time). The next one needs every warp to have seen
    /// the others' times (to have synchronized with them since).
    last: Vec<(usize, usize, u32)>,
}

/// A `setmaxnreg` some warps of a warpgroup have executed.
#[derive(Clone)]
struct PendingRegs<'a> {
    increase: bool,
    count: u32,
    inst: &'a Inst,
    /// Warps of the warpgroup (0 to 3) that have executed it.
    arrived: Vec<usize>,
    joined: Vec<(usize, usize, u32)>,
}

/// The per-thread register budget the simulator gives each warpgroup of a block of
/// `threads` threads before any `setmaxnreg`: the 64K registers of an SM shared by the
/// block's threads, at most 255, rounded down to a multiple of 8 (168 for the 384 threads
/// of a kernel with a producer and two consumer warpgroups).
pub fn setmaxnreg_initial_budget(threads: usize) -> u32 {
    ((65536 / threads.max(1)).min(255) / 8 * 8) as u32
}

/// The state of one barrier (`sync_threads`, `bar.sync` / `bar.arrive`) of
/// a block.
#[derive(Clone, Default)]
struct NamedState<'a> {
    /// Warps that have arrived at the current, incomplete use.
    arrived: Vec<usize>,
    /// Those of them that wait for it to complete (`bar.sync`), blocked.
    waiting: Vec<usize>,
    /// The warp count that completes it.
    expected: usize,
    /// The join of the arrived warps' clocks.
    release: Vec<u32>,
    /// Whether the current use is an all-thread barrier (`sync_threads`, or
    /// `bar.sync` without a thread count), and the instruction its first
    /// waiters reached.
    all_threads: bool,
    site: Option<&'a Inst>,
    /// The reduction of a `bar.red` use, the counts of
    /// true and false predicates so far, and the arrived warps' result
    /// values (by warp key).
    red: Option<BarrierRedKind>,
    red_counts: (u32, u32),
    red_results: Vec<(usize, ValueId)>,
}

/// What a frame of a warp's continuation runs: the kernel body, or a
/// region of a structured instruction, with that instruction's state.
enum FrameKind<'a> {
    /// The warp exits at its end.
    Body,
    /// The `then` region of an `If`; its `else` region runs next, for
    /// `else_lanes` (the SIMT stack's order).
    IfThen {
        inst: &'a Inst,
        else_lanes: Vec<usize>,
        /// With independent thread scheduling: the join record of a
        /// divergent `If`, whose `else` lanes run as a convergence group of
        /// their own (`else_lanes` is then empty). At the region's end the
        /// group waits for that one to finish ([`Wait::Join`]).
        split: Option<u64>,
    },
    /// The `else` region of an `If`; at its end both lane sets take their
    /// region's yields as the `If`'s results.
    IfElse {
        inst: &'a Inst,
        then_lanes: Vec<usize>,
        /// The join record when this is the bottom frame of an `else`
        /// group split off at a divergent `If`: the group ends with the
        /// region, waking the `then` group waiting at the join.
        join: Option<u64>,
    },
    /// The body of a `For`, run by the frame's lanes (those still
    /// iterating); `all` entered the loop. `ivs` holds each lane's induction
    /// variable, by lane.
    ForBody {
        inst: &'a Inst,
        all: Vec<usize>,
        ivs: Box<[i64; 32]>,
    },
    /// The condition and the body of a `While`, after `iterations` rounds.
    WhileCond {
        inst: &'a Inst,
        all: Vec<usize>,
        iterations: u64,
    },
    WhileBody {
        inst: &'a Inst,
        all: Vec<usize>,
        iterations: u64,
    },
    /// The traced body of a `Call`; at its end the
    /// lanes take its yields as the call's results.
    Call {
        inst: &'a Inst,
    },
}

/// One level of a warp's continuation: a region, the index of its next
/// instruction, and the lanes that run it.
struct Frame<'a> {
    region: &'a Region,
    next: usize,
    lanes: Vec<usize>,
    kind: FrameKind<'a>,
}

impl Frame<'_> {
    /// Whether the frame has nothing left to run.
    fn finished(&self) -> bool {
        self.lanes.is_empty() || self.next >= self.region.insts.len()
    }
}

/// A convergence group's continuation (innermost frame last).
struct Group<'a> {
    frames: Vec<Frame<'a>>,
}

/// An mbarrier's key in [`Machine::mbarriers`]: (block rank, buffer, byte).
type MbarrierKey = (usize, u32, usize);

/// The arrivals of one step at a `bar.red`: its true
/// and false predicates, and the arriving warps' result values (by key).
type RedPart = (u32, u32, Vec<(usize, ValueId)>);

/// What a blocked warp waits for.
#[derive(Clone, Copy)]
enum Wait<'a> {
    /// Barrier `id` of block `block` (0 is also `sync_threads`'), which it
    /// reached at `inst`.
    Barrier { block: usize, id: usize, inst: &'a Inst },
    /// The completion of phase `phase` of the mbarrier keyed by `key`, at
    /// an `mbarrier.try_wait` (`inst`).
    Mbarrier { key: MbarrierKey, phase: u64, inst: &'a Inst },
    /// The completion of the cluster barrier's current phase, at a
    /// `barrier.cluster.wait` (`inst`).
    Cluster { inst: &'a Inst },
    /// The end of a divergent `If` (`inst`), for the `else` group of join
    /// record `id` to finish (structured reconvergence).
    Join { id: u64, inst: &'a Inst },
    /// The lanes of its member mask that other convergence groups of the
    /// warp hold, at a `.sync` warp collective (`inst`): the instruction
    /// runs once, for all of them, when they have all reached it.
    Rendezvous { inst: &'a Inst },
    /// The rest of warpgroup `group` of block `block` at a `setmaxnreg`
    /// (`inst`), or, for an increase, registers in the block's pool.
    SetMaxNReg { block: usize, group: usize, inst: &'a Inst },
    /// The rest of warpgroup `group` of block `block` at a
    /// `wgmma.mma_async` (`inst`).
    Wgmma { block: usize, group: usize, inst: &'a Inst },
    /// The completion of the `tcgen05.alloc` (`inst`) pending for block
    /// `block` (the even CTA of a pair): the peer's issue, or free columns.
    TmemAlloc { block: usize, inst: &'a Inst },
}

/// What a spinning warp last polled: the mbarrier phase of an
/// `mbarrier.test_wait` (or timed-out `mbarrier.try_wait`) that failed, or
/// the address of an ordered (strong) load.
#[derive(Clone, Copy)]
enum Poll<'a> {
    Mbarrier { key: MbarrierKey, phase: u64, inst: &'a Inst },
    Load { buffer: BufferId, block: usize, byte: usize, inst: &'a Inst },
}

#[derive(Clone, Copy)]
enum Status<'a> {
    Runnable,
    Blocked(Wait<'a>),
    Exited,
}

/// The execution state of a convergence group: lanes of a warp that run
/// together. A warp starts as one group; with independent thread
/// scheduling a divergent `If` splits it into a `then` group and an `else`
/// group, scheduled on their own, which rejoin at the end of the `If`.
/// The group a warp runs is [`Machine::warps`]; the others wait in
/// [`Machine::parked`].
struct WarpState<'a> {
    groups: Vec<Group<'a>>,
    status: Status<'a>,
    /// Whether its frames are known to stand at an instruction (or it has
    /// exited): region ends and loop iterations are handled lazily.
    settled: bool,
    /// Its `mbarrier.try_wait` gave up waiting (the time limit PTX allows):
    /// the next one returns whether the phase is over without blocking.
    timed_out: bool,
    /// The first scheduling round it may run in: a warp that another wakes
    /// starts again in the next round, together with its waker.
    ready_round: u64,
    /// It reached a yield point (a failed `mbarrier.test_wait` or a
    /// `while_carry` back-edge) since [`Schedule::RunToBlock`] gave it the
    /// turn.
    ///
    /// [`Schedule::RunToBlock`]: super::Schedule::RunToBlock
    yielded: bool,
    /// The progress epoch ([`Machine::epoch`]) of its last yield: it spins
    /// if nothing visible has happened since.
    spin_epoch: u64,
    /// What it polled last, for hang reports, and the progress epoch then.
    /// A warp spins (it counts toward a hang) only if it has yielded and
    /// polled since anything visible happened: a long loop that computes
    /// in registers is not a spin.
    poll: Option<Poll<'a>>,
    poll_epoch: u64,
    /// The progress epoch at which its `mbarrier.try_wait` last gave up, and
    /// how many times it has given up since anything visible happened: once
    /// that reaches [`GIVE_UPS_WITHOUT_PROGRESS`], waiting longer cannot help
    /// (a deadlock). A few are allowed, since a warp may count its failed
    /// waits in registers before it does something visible.
    timeout_epoch: Option<(u64, u32)>,
}

impl<'a> WarpState<'a> {
    /// A runnable group with these frames.
    fn new(frames: Vec<Frame<'a>>) -> WarpState<'a> {
        WarpState {
            groups: vec![Group { frames }],
            status: Status::Runnable,
            settled: false,
            timed_out: false,
            ready_round: 0,
            yielded: false,
            spin_epoch: u64::MAX,
            poll: None,
            poll_epoch: u64::MAX,
            timeout_epoch: None,
        }
    }

    /// The lanes (threads) running its next instruction.
    fn lanes(&self) -> &[usize] {
        &self.groups[0].frames.last().expect("a running group has a frame").lanes
    }

    /// Whether it can run in scheduling round `round`.
    fn ready(&self, round: u64) -> bool {
        matches!(self.status, Status::Runnable) && self.ready_round <= round
    }

    /// The join record of the divergent `If` whose `else` group it is.
    fn join(&self) -> Option<u64> {
        match self.groups[0].frames.first().map(|f| &f.kind) {
            Some(FrameKind::IfElse { join, .. }) => *join,
            _ => None,
        }
    }
}

/// SplitMix64: the deterministic generator of seeded schedules.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A cluster's barrier (`barrier.cluster.arrive` / `barrier.cluster.wait`):
/// every thread of the cluster that has not exited arrives once per phase,
/// and a wait blocks until all have, after which the barrier re-arms.
/// Arrivals release at cluster scope (unless relaxed) and waits acquire.
#[derive(Default)]
struct ClusterBarrier {
    /// The current phase.
    phase: u64,
    /// The join of the clocks of the warps that arrived with a release in
    /// the current phase, and in the last completed one.
    release: Vec<u32>,
    completed: Vec<u32>,
    /// The same for every arrival, relaxed ones included, but only of each
    /// arriving warp's own slot: what carries `fence.mbarrier_init`.
    init_release: Vec<u32>,
    init_completed: Vec<u32>,
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

/// Shared memory an asynchronous operation writes: `data` (captured at
/// issue) at byte `dst_byte` of buffer `dst` of block `block`.
struct AsyncWrite {
    dst: BufferId,
    block: usize,
    dst_byte: usize,
    data: Vec<u8>,
}

/// Shared memory an asynchronous operation reads in the background (a bulk
/// copy's or TMA store's source, a WGMMA operand): overwriting it before the
/// operation has read it is a race.
struct AsyncRead {
    buffer: BufferId,
    block: usize,
    bytes: std::ops::Range<usize>,
}

/// What completes an asynchronous operation, and whom its completion
/// orders it before.
#[derive(Clone, Copy, PartialEq)]
enum Trigger {
    /// A `cp.async` of the issuing thread: its covering
    /// `cp_async_wait_group` completes it for the thread or, if a
    /// `cp.async.mbarrier.arrive` tracks it, the arrive-on it owes fires
    /// once it completes, for the waiters of that mbarrier phase.
    CpAsync { tracked: bool },
    /// A bulk copy: `complete_tx` of its bytes on this mbarrier, whose
    /// phase publishes it to its waiters.
    CompleteTx(MbarrierKey),
    /// A bulk copy of the issuing thread's bulk async-groups: the covering
    /// `cp.async.bulk.wait_group.read` releases its source, and the covering
    /// plain `cp.async.bulk.wait_group` (or the CTA's exit) lands its writes.
    BulkGroup,
    /// A warpgroup MMA (one warp's share): the covering
    /// `wgmma.wait_group` of that warp completes it.
    Wgmma,
    /// A `tcgen05.ld` (one warp's): the warp's `tcgen05.wait::ld` lands its
    /// registers.
    TmemLd,
    /// A `tcgen05.mma` of a CTA's tensor-core pipe: it lands as the CTA agent's write when the mbarrier
    /// phase its `tcgen05.commit` owes an arrival is observed (or at
    /// random before).
    Tcgen05Mma,
    /// A test-only operation, completed by hand.
    #[cfg(test)]
    Test,
}

/// One in-flight asynchronous-operation model for every kind of operation:
/// `cp.async` (with `cp.async.mbarrier.arrive{.noinc}`), `cp.async.bulk`, TMA,
/// bulk async-groups and proxy fences, WGMMA groups and tcgen05.
struct AsyncOp {
    /// The issuing thread.
    thread: usize,
    /// The vector-clock slot of the agent that performs it.
    agent: usize,
    /// Shared memory it writes when it completes.
    writes: Vec<AsyncWrite>,
    /// Shared memory it reads until it completes (or, for a bulk copy to
    /// global memory, until the `cp.async.bulk.wait_group.read` that covers
    /// it, which empties the list).
    reads: Vec<AsyncRead>,
    /// Global memory it writes when it completes, with the data captured at
    /// issue: `(buffer, byte, data)`. Reading those bytes before is reported.
    global_writes: Vec<(BufferId, usize, Vec<u8>)>,
    /// For a TMA reduce store: the reduction and element type its global
    /// writes combine with memory by, element-wise, when they land.
    global_reduce: Option<(crate::cpasync::ReductionKind, DType)>,
    /// Register results, `(value, thread, bits)`, written when it completes;
    /// reading one earlier is reported.
    results: Vec<(ValueId, usize, u64)>,
    trigger: Trigger,
    /// `None` in flight; once complete, the agent's time it completed at
    /// (0: completed as the issuing warp's own write).
    done: Option<u32>,
}

impl AsyncOp {
    fn in_flight(&self) -> bool {
        self.done.is_none()
    }

    /// Whether it writes byte `byte` of shared buffer `b` of `block`.
    fn writes_byte(&self, b: BufferId, block: usize, byte: usize) -> bool {
        self.writes
            .iter()
            .any(|w| w.dst == b && w.block == block && (w.dst_byte..w.dst_byte + w.data.len()).contains(&byte))
    }
}

/// The arrive-on a `cp.async.mbarrier.arrive` owes its mbarrier, which it
/// makes once the copies its thread issued before it (still in flight then,
/// by id) have landed.
struct DeferredArrival {
    copies: Vec<usize>,
    /// The agent's time at the arrive, which it publishes too: with
    /// [`super::AsyncCompletion::Random`] the thread's earlier copies may
    /// have landed as agent writes before it (0 otherwise).
    floor: u32,
    /// Owed by a `tcgen05.commit` (its MMAs) rather than by
    /// `cp.async.mbarrier.arrive` (names the source
    /// in the count-mismatch report).
    commit: bool,
}

/// One mbarrier's state. It is keyed by its address, so it needs no storage.
struct Mbarrier {
    arrivals_per_phase: u32,
    arrivals_left: u32,
    /// The current phase's tx-count: bytes of bulk copies it still waits
    /// for. `expect_tx` raises it and landing copies lower it; the PTX ISA
    /// lets it go negative, but copies only land here once every arrival of
    /// the phase is in, and landing more than it holds is reported.
    tx_count: i64,
    /// Bulk copies in flight that report to this mbarrier (by operation id).
    copies: Vec<usize>,
    /// The arrivals `cp.async.mbarrier.arrive` owes the current phase. They
    /// are part of `arrivals_left`, so no other arrival may take their place.
    deferred: Vec<DeferredArrival>,
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
    /// Who initialised it (the warp's clock slot and key), where, and the
    /// warp's own time at the `fence.mbarrier_init` that released the init
    /// to the cluster, if it has executed one since.
    init: (usize, usize, Loc),
    fenced: Option<u32>,
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

/// The first byte address the simulator gives a global buffer.
pub const SIM_GLOBAL_BASE: u64 = 0x7f00_0000_0000;
/// The bytes left unmapped after each global buffer, so that an overrun
/// is reported rather than landing in the next one.
const GLOBAL_GAP: u64 = 1 << 16;
/// Where the block's shared memory appears among generic addresses.
pub const SIM_SHARED_WINDOW: u64 = 0x1000_0000_0000;
/// Where the thread's local memory appears among generic addresses.
pub const SIM_LOCAL_WINDOW: u64 = 0x2000_0000_0000;
/// The span of the shared and local windows.
const WINDOW_BYTES: u64 = 1 << 32;

/// The simulator's address map: the virtual addresses `toint` gives and
/// `make_ptr` resolves. Global buffers get per-launch bases from
/// [`SIM_GLOBAL_BASE`], 256-byte aligned as `cuMemAlloc`'s, with a gap after
/// each; a shared buffer's address is its start in the block's shared
/// memory ([`KernelIr::shared_memory_layout`]), the same small offset in
/// every block, as on the GPU; a per-thread buffer's is its start in the
/// thread's local memory. Generic addresses are global ones, or the shared
/// or local offset plus its window.
pub(super) struct AddrMap {
    /// Each global buffer's base and bytes.
    global: Vec<Option<(u64, u64)>>,
    /// Each per-thread buffer's start in the thread's local memory.
    local: Vec<Option<u64>>,
}

/// For each buffer, the buffer whose shared storage holds it ([`Machine::home`]).
fn overlay_homes(ir: &KernelIr) -> Vec<usize> {
    let mut home: Vec<usize> = (0..ir.buffers.len()).collect();
    for group in &ir.overlays {
        let members: Vec<usize> = group.iter().flatten().map(|b| b.0 as usize).collect();
        if let Some(&first) = members.iter().min() {
            members.iter().for_each(|&m| home[m] = first);
        }
    }
    home
}

/// The bytes of the shared storage of buffer `k` (a home): its own, or its overlay group's
/// largest member's.
fn storage_bytes(ir: &KernelIr, home: &[usize], k: usize, dynamic: usize) -> usize {
    ir.buffers
        .iter()
        .enumerate()
        .filter(|&(m, _)| home[m] == k)
        .map(|(_, b)| b.len.map_or(dynamic, |n| n * b.elem.bytes()))
        .max()
        .unwrap_or(0)
}

impl AddrMap {
    fn new(ir: &KernelIr, globals: &[Option<Vec<u8>>]) -> AddrMap {
        let mut next = SIM_GLOBAL_BASE;
        let global = globals
            .iter()
            .map(|g| {
                let bytes = g.as_ref()?.len() as u64;
                let base = next;
                next = (base + bytes + GLOBAL_GAP).next_multiple_of(256);
                Some((base, bytes))
            })
            .collect();
        let mut at = 0u64;
        let local = ir
            .buffers
            .iter()
            .map(|b| {
                (b.space == MemSpace::Local).then(|| {
                    let start = at.next_multiple_of(b.align.max(16) as u64);
                    at = start + (b.len.unwrap_or(0) * b.elem.bytes()) as u64;
                    start
                })
            })
            .collect();
        AddrMap { global, local }
    }
}

pub(super) struct Machine<'a> {
    ir: &'a KernelIr,
    opts: &'a SimOptions,
    /// The programmatic-dependent-launch model ([`super::SimStream`]): this launch's
    /// global-memory footprint, and the primary's to check pre-wait accesses against.
    pdl: Option<Box<PdlState>>,
    cfg: LaunchConfig,
    globals: Vec<Option<Vec<u8>>>,
    scalars: Vec<(ValueId, u128)>,
    report: SimReport,
    /// The blocks running together, indexed by rank in the cluster.
    blocks: Vec<[u32; 3]>,
    threads_per_block: usize,
    nthreads: usize,
    values: Vec<Vec<u64>>,
    /// The high words of 128-bit values (Int128, Uint128), per thread, for
    /// those values only: the others leave their entry empty.
    values_hi: Vec<Vec<u64>>,
    /// `shared[buffer][block rank]`.
    shared: Vec<Vec<Shared>>,
    locals: Vec<Option<LocalMem>>,
    /// Every asynchronous operation of the launch, by id.
    ops: Vec<AsyncOp>,
    /// Register results of operations in flight, by (value, thread): read
    /// before completion is an error. Empty unless such operations exist,
    /// which keeps the check off the hot path.
    pending_results: HashMap<(ValueId, usize), usize>,
    /// Register values a masked-off lane of a masked load without
    /// pass-through values left undefined, by (value, thread), with the
    /// load's location. Pure operations pass the mark on (a select only
    /// from the operand it picks); anything else that reads one is
    /// reported. Empty unless such a load ran, which keeps the check off
    /// the hot path.
    undefined: HashMap<(ValueId, usize), Loc>,
    /// Each warp's own times at its proxy fences (`fence.proxy.async`), in
    /// order: what lets the async proxy read its earlier generic writes.
    proxy_fences: Vec<Vec<u32>>,
    /// Each thread's bulk copies (to global memory) not yet committed, and
    /// its committed bulk async-groups, oldest first, by id.
    bulk_uncommitted: Vec<Vec<usize>>,
    bulk_groups: Vec<VecDeque<Vec<usize>>>,
    /// Operations in flight that still have global memory to write: global
    /// reads check them when there are any.
    pending_global: usize,
    /// The buffer whose storage holds each shared buffer's bytes: itself, or for an overlaid
    /// buffer the first of its overlay group (every member starts at the group's offset).
    home: Vec<usize>,
    /// The clusters to run, by first block, when not all of them.
    cluster_filter: Option<std::collections::HashSet<[u32; 3]>>,
    /// Whether any operation of the launch reads memory in the background.
    any_async_reads: bool,
    /// Each thread's uncommitted and committed `cp.async` copies, by id.
    uncommitted: Vec<Vec<usize>>,
    groups: Vec<VecDeque<Vec<usize>>>,
    /// Each thread's `cp.async` copies still in flight, by id: what a
    /// `cp.async.mbarrier.arrive` of the thread tracks.
    in_flight: Vec<Vec<usize>>,
    /// Warps per block (the last one may be short).
    warps_per_block: usize,
    /// Blocks per cluster.
    cluster_size: usize,
    /// Clock slots per block: one per warp, then [`AGENT_SLOTS`] for its
    /// agents ([`Machine::slot_of`]).
    stride: usize,
    /// The vector clock of each warp, by `block rank * warps_per_block + warp`.
    /// It has an entry for every warp of the block and, after them, one for
    /// the block's bulk-copy agent (index `warps_per_block`).
    clocks: Vec<Vec<u32>>,
    /// The latest time of each block's bulk-copy agent: every bulk copy that
    /// lands gets a new one.
    bulk_time: Vec<u32>,
    /// The 16 named barriers of each block (barrier 0 is `sync_threads`').
    named: Vec<Vec<NamedState<'a>>>,
    /// The current scheduling round.
    round: u64,
    /// Counts visible progress (see [`SimOptions::max_idle_steps`]).
    epoch: u64,
    /// Warp steps since the last visible progress.
    idle: u64,
    /// A buffer for the lanes of the warps that execute an instruction.
    active: Vec<usize>,
    /// Every warp of the cluster, by `block rank * warps_per_block + warp`:
    /// the convergence group it runs.
    warps: Vec<WarpState<'a>>,
    /// The warp's other convergence groups, by the same index.
    parked: Vec<Vec<WarpState<'a>>>,
    /// The last join record given to a divergent `If`.
    next_join: u64,
    /// The generators of [`super::Schedule::Random`] and
    /// [`super::AsyncCompletion::Random`].
    sched_rng: Option<u64>,
    async_rng: Option<u64>,
    /// Scheduler steps so far.
    steps: u64,
    /// Operations below this id have completed.
    first_live_op: usize,
    /// The lanes of each warp (by block rank, barrier, warp) that have
    /// executed a `barrier.arrive` (unaligned) of the barrier's current use:
    /// the warp arrives once all have.
    unaligned: HashMap<(usize, usize, usize), u32>,
    /// The `setmaxnreg` state of each block that has executed one.
    setmaxnreg: HashMap<usize, BlockRegs<'a>>,
    /// The clusters that have not started yet, in launch order.
    queued: VecDeque<Vec<[u32; 3]>>,
    /// How many clusters at the front of `queued` have
    /// launched (no `try_cancel` can cancel them), and the generator that
    /// launches them ([`SimOptions::clc_launch_seed`]).
    clc_launched: usize,
    clc_rng: Option<u64>,
    /// By resident slot: an id of the cluster running there (unique for
    /// the launch), and the last id given.
    cluster_ids: Vec<u64>,
    next_cluster_id: u64,
    /// The barrier of each running cluster.
    cluster_barriers: Vec<ClusterBarrier>,
    /// By thread: the cluster barrier phase it arrived in and has not yet
    /// waited for.
    cluster_arrival: Vec<Option<u64>>,
    /// By `(block rank, buffer, byte)`.
    mbarriers: HashMap<(usize, u32, usize), Mbarrier>,
    /// The mbarriers invalidated by `mbarrier.inval`
    /// (and not initialised again since), with where.
    invalidated: HashMap<MbarrierKey, Loc>,
    /// Where each warp (by key) that executed `exit` did.
    exit_sites: HashMap<usize, Loc>,
    clock: u64,
    loc: Option<Loc>,
    /// Each thread's carry flag (PTX's `CC.CF`), as the extended-precision
    /// instructions left it in program order, and whether a subtraction set
    /// it (a borrow, which only a subtraction may read); `None` where no
    /// instruction has set it since the thread entered its current block of
    /// straight-line code (PTX keeps it across neither control flow nor
    /// calls).
    carry: Vec<Option<(bool, bool)>>,
    /// The values constants define (a 64-bit division by one keeps the
    /// carry flag; see [`crate::ir::carry_flag_breaker`]).
    constants: std::collections::HashSet<ValueId>,
    /// Where a block's shared memory lives, as the launcher passes it.
    smem: SharedMemoryLayout,
    /// Where each buffer is in its address space ([`AddrMap`]).
    addrs: AddrMap,
    /// For each byte of memory that a strong write published a clock on
    /// (a release store, or a strong write after a fence), what it
    /// published, by `(buffer, block rank or GLOBAL, byte)`. An acquire
    /// load that reads the byte, or a strong load followed by a fence, joins
    /// the clock if it comes from its own block.
    releases: HashMap<(u32, usize, usize), Released>,
    /// What each warp knows of other warps' `fence.mbarrier_init`s besides
    /// its clock: the arrivals at the cluster barriers it waited for,
    /// relaxed ones included (see [`Self::check_remote_mbarrier`]).
    init_clocks: Vec<Vec<u32>>,
    /// The block whose shared memory the access being executed reaches, when
    /// it is another CTA's (distributed shared memory); `None`: the accessing thread's own.
    smem_target: Option<usize>,
    /// For each block, the accesses other CTAs of its cluster made to its
    /// shared memory: the latest own-clock value of each accessing warp's
    /// slot, and where. The block must not exit before they are ordered
    /// before its exit.
    dsmem_log: HashMap<usize, HashMap<usize, (u32, crate::ir::Loc)>>,
    /// Each warp's clock at its last fence, which its strong writes publish
    /// (the PTX memory model's `fence; st.relaxed` release pattern).
    fence_release: Vec<Option<Rc<[u32]>>>,
    /// The clocks released on what each warp's strong loads read since its
    /// last fence, which its next fence acquires (`ld.relaxed; fence`).
    pending_acquire: Vec<Option<Vec<u32>>>,
    /// For each global buffer that non-coherent loads read, the bytes they
    /// read and the bytes the kernel wrote (for the whole launch).
    nc: Vec<Option<NcState>>,
    /// The warpgroup MMAs' state.
    wgmma: wgmma::State<'a>,
    /// Tensor memory.
    tmem: tmem::State<'a>,
    /// Tensor maps written in memory and the tensormap proxy.
    tmaps: tensormap::State,
    /// The multicast buffers' replicas.
    multicast: Vec<distributed::McGroup>,
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
        DType::F32 | DType::TF32 => bf(b),
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
        DType::F32 | DType::TF32 => f32b(x),
        DType::F16 => f2h(x),
        DType::BF16 => f32_to_bf16_bits(x) as u64,
        DType::F64 => (x as f64).to_bits(),
        _ => unreachable!("not a float type"),
    }
}

/// A TF32 operand as the tensor core reads it (warp `mma.sync` and tcgen05
/// `.kind::tf32`): the Float32 value with its low 13
/// mantissa bits dropped (the top 19 bits kept).
pub(super) fn tf32_truncate(x: f64) -> f64 {
    f32::from_bits((x as f32).to_bits() & 0xffff_e000) as f64
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
        // Through Int32, wrapped to 4 bits, as the emitter converts.
        DType::I4 => x as i32 as u64 & 0xf,
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
            // through f32) give the canonical 0x7fff; and f16 -> f32
            // (cvt.f32.f16) gives the canonical 0x7fffffff (checked by the
            // FP8 decodes, which widen through f16).
            let sign = b >> (from.bits() - 1) & 1;
            return match (from, to) {
                (DType::F16, DType::F32) => 0x7fff_ffff,
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

/// Whether `op` is a scalar instruction on 128-bit values, which
/// [`Machine::wide_inst`] runs (loads, stores and the structured control
/// flow move them with the rest).
fn is_wide_op(ir: &KernelIr, op: &Op) -> bool {
    let wide = |v: &ValueId| ir.dtype(*v).bits() == 128;
    match op {
        Op::Const { result, .. }
        | Op::Unary { result, .. }
        | Op::Binary { result, .. }
        | Op::Bitcast { result, .. } => wide(result),
        Op::Select { result, .. } => wide(result),
        Op::Cmp { lhs, .. } => wide(lhs),
        Op::Cast { result, x } => wide(result) || wide(x),
        _ => false,
    }
}

/// Integer semantics at 128 bits.
mod wide {
    use crate::ir::{BinaryOp, CmpPred, DType, UnaryOp};

    pub(super) fn unary(op: UnaryOp, ty: DType, a: u128) -> u128 {
        let x = a as i128;
        match op {
            UnaryOp::Neg => x.wrapping_neg() as u128,
            UnaryOp::Not => !a,
            UnaryOp::Abs if ty.is_signed() => x.wrapping_abs() as u128,
            UnaryOp::Abs => a,
            UnaryOp::Popc => a.count_ones() as u128,
            UnaryOp::Clz => a.leading_zeros() as u128,
            UnaryOp::Ctz => a.trailing_zeros() as u128,
            UnaryOp::Brev => a.reverse_bits(),
            _ => unreachable!("validated"),
        }
    }

    pub(super) fn binary(op: BinaryOp, ty: DType, a: u128, b: u128) -> Result<u128, String> {
        use BinaryOp::*;
        let signed = ty.is_signed();
        let (x, y) = (a as i128, b as i128);
        Ok(match op {
            Add => a.wrapping_add(b),
            Sub => a.wrapping_sub(b),
            Mul => a.wrapping_mul(b),
            Div | Rem | DivU | RemU if b == 0 => return Err("integer division by zero".into()),
            Div if signed => x.wrapping_div(y) as u128,
            Div | DivU => a / b,
            Rem if signed => x.wrapping_rem(y) as u128,
            Rem | RemU => a % b,
            Min if signed => x.min(y) as u128,
            Min => a.min(b),
            Max if signed => x.max(y) as u128,
            Max => a.max(b),
            And => a & b,
            Or => a | b,
            Xor => a ^ b,
            Shl | ShrU | ShrS if b >= 128 => return Err(format!("shift by {y} bits")),
            Shl => a << b,
            ShrU => a >> b,
            ShrS => (x >> b) as u128,
            MulHi => unreachable!("validated"),
        })
    }

    pub(super) fn compare(pred: CmpPred, ty: DType, a: u128, b: u128) -> bool {
        let ord = if ty.is_signed() { (a as i128).cmp(&(b as i128)) } else { a.cmp(&b) };
        match pred {
            CmpPred::Eq => ord.is_eq(),
            CmpPred::Ne => ord.is_ne(),
            CmpPred::Lt => ord.is_lt(),
            CmpPred::Le => ord.is_le(),
            CmpPred::Gt => ord.is_gt(),
            CmpPred::Ge => ord.is_ge(),
        }
    }

    /// A 128-bit value converted to a narrower integer (truncated) or to a
    /// float (rounded to nearest even), or to the other 128-bit type.
    pub(super) fn cast_from(from: DType, to: DType, b: u128) -> u128 {
        if to.is_float() {
            let bits = match (to, from.is_signed()) {
                (DType::F64, true) => (b as i128 as f64).to_bits(),
                (DType::F64, false) => (b as f64).to_bits(),
                (_, true) => super::from_f32(to, b as i128 as f32),
                (_, false) => super::from_f32(to, b as f32),
            };
            return bits as u128;
        }
        if to.bits() == 128 { b } else { super::mask(b as u64, to.bits()) as u128 }
    }

    /// A narrower value converted to a 128-bit integer: sign- or
    /// zero-extended, or a float truncated toward zero (saturating, NaN 0).
    pub(super) fn cast_to(from: DType, to: DType, b: u64) -> u128 {
        let signed_to = to.is_signed();
        if from.is_float() {
            let x = super::to_f64(from, b);
            return if signed_to { x as i128 as u128 } else { x as u128 };
        }
        if from.is_signed() { super::sext(b, from.bits()) as i128 as u128 } else { b as u128 }
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
/// (PTX ISA, "Matrix Fragments for mma.m16n8k16 / m16n8k8 / m16n8k4 / m8n8k4
/// / m16n8k32"). `g` is `groupID` and `t` is `threadID_in_group`. For a
/// sparse kind the column is into the compressed A (the kept half of each
/// row), which [`sparse_dense_col`] maps to the dense column.
fn mma_a_coord(kind: MmaKind, lane: usize, i: usize) -> (usize, usize) {
    use MmaKind::*;
    let (g, t) = (lane >> 2, lane % 4);
    match kind {
        // .f16/.bf16 m16n8k16: rows g (a0, a1, a4, a5) and g + 8; columns
        // 2t + (i & 1), plus 8 for a4..a7. Sparse m16n8k32 holds its
        // compressed 16 x 16 A the same way.
        M16N8K16F32F16 | M16N8K16F16F16 | M16N8K16F32BF16 | SpM16N8K32F32F16 | SpM16N8K32F16F16 | SpM16N8K32F32BF16 => {
            (g + 8 * ((i >> 1) & 1), 2 * t + (i & 1) + 8 * (i >> 2))
        }
        // .f16/.bf16 m16n8k8: rows g (a0, a1) and g + 8 (a2, a3); sparse
        // m16n8k16 holds its compressed 16 x 8 A the same way.
        M16N8K8F32F16 | M16N8K8F16F16 | M16N8K8F32BF16 | SpM16N8K16F32F16 | SpM16N8K16F16F16 | SpM16N8K16F32BF16 => {
            (g + 8 * (i >> 1), 2 * t + (i & 1))
        }
        M16N8K4Tf32 => (g + 8 * i, t),
        // .tf32 m16n8k8: rows g (a0, a2) and g + 8 (a1, a3); columns t (a0,
        // a1) and t + 4 (a2, a3).
        M16N8K8Tf32 => (g + 8 * (i & 1), t + 4 * (i >> 1)),
        M8N8K4F64 => (g, t),
        // .e4m3/.e5m2 m16n8k32: rows g (a0..a3, a8..a11) and g + 8; columns
        // 4t + (i & 3), plus 16 for a8..a15.
        M16N8K32F32E4M3 | M16N8K32F16E4M3 | M16N8K32F32E5M2 | M16N8K32F16E5M2 => {
            (g + 8 * ((i >> 2) & 1), 4 * t + (i & 3) + 16 * (i >> 3))
        }
        // .e4m3/.e5m2 m16n8k16: rows g (a0..a3) and g + 8 (a4..a7).
        M16N8K16F32E4M3 | M16N8K16F16E4M3 | M16N8K16F32E5M2 | M16N8K16F16E5M2 => (g + 8 * (i >> 2), 4 * t + (i & 3)),
    }
}

/// `(k, n)` of value `i` of lane `lane` in an `mma.sync` B fragment.
fn mma_b_coord(kind: MmaKind, lane: usize, i: usize) -> (usize, usize) {
    use MmaKind::*;
    let (g, t) = (lane >> 2, lane % 4);
    match kind {
        // Pairs of rows 2t, 2t + 1, then 8 rows further for each next pair
        // (sparse m16n8k32 has four pairs).
        M16N8K16F32F16 | M16N8K16F16F16 | M16N8K16F32BF16 | SpM16N8K16F32F16 | SpM16N8K16F16F16 | SpM16N8K16F32BF16
        | SpM16N8K32F32F16 | SpM16N8K32F16F16 | SpM16N8K32F32BF16 => (2 * t + (i & 1) + 8 * (i >> 1), g),
        M16N8K8F32F16 | M16N8K8F16F16 | M16N8K8F32BF16 => (2 * t + (i & 1), g),
        // .tf32 m16n8k8: rows t (b0) and t + 4 (b1).
        M16N8K8Tf32 => (t + 4 * i, g),
        M16N8K4Tf32 | M8N8K4F64 => (t, g),
        // .e4m3/.e5m2 m16n8k32: rows 4t + (i & 3), plus 16 for b4..b7.
        M16N8K32F32E4M3 | M16N8K32F16E4M3 | M16N8K32F32E5M2 | M16N8K32F16E5M2 => (4 * t + (i & 3) + 16 * (i >> 2), g),
        M16N8K16F32E4M3 | M16N8K16F16E4M3 | M16N8K16F32E5M2 | M16N8K16F16E5M2 => (4 * t + i, g),
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

/// Where the metadata of the 4-wide chunk `chunk` of row `row` of a sparse
/// A lives, as `(lane, nibble)` of the metadata registers, with sparsity
/// selector `selector` (PTX ISA, "Sparse matrix storage" and the sparse
/// m16n8k16 and m16n8k32 metadata layouts for .f16/.bf16): lane `4g +
/// selector` (m16n8k16) or the pair `4g + 2 * selector + h` (m16n8k32, `h`
/// the half of K) holds rows `g` (nibbles 0 to 3) and `g + 8` (nibbles 4 to
/// 7), one nibble per chunk.
pub(crate) fn sparse_metadata_slot(k: usize, selector: usize, row: usize, chunk: usize) -> (usize, usize) {
    let (g, half) = (row % 8, row / 8);
    if k == 16 {
        (4 * g + selector, 4 * half + chunk)
    } else {
        (4 * g + 2 * selector + chunk / 4, 4 * half + chunk % 4)
    }
}

/// The two indices of a metadata nibble of `mma.sp::ordered_metadata`, the
/// first in the low bits, or `None` if they are not one of the meaningful
/// values (`0b0100`, `0b1000`, `0b1001`, `0b1100`, `0b1101`, `0b1110`: two
/// increasing positions in the chunk).
pub(crate) fn sparse_indices(nibble: u32) -> Option<(usize, usize)> {
    let (lo, hi) = ((nibble & 3) as usize, (nibble >> 2 & 3) as usize);
    (lo < hi).then_some((lo, hi))
}

fn read_le(bytes: &[u8]) -> u64 {
    bytes.iter().rev().fold(0, |acc, &b| (acc << 8) | b as u64)
}

fn write_le(bytes: &mut [u8], bits: u64) {
    for (k, b) in bytes.iter_mut().enumerate() {
        *b = (bits >> (8 * k)) as u8;
    }
}

impl<'a> Machine<'a> {
    pub(super) fn new(
        ir: &'a KernelIr,
        opts: &'a SimOptions,
        cfg: LaunchConfig,
        globals: Vec<Option<Vec<u8>>>,
        scalars: Vec<(ValueId, u128)>,
    ) -> Machine<'a> {
        // Non-coherent loads need memory that nothing writes during the
        // launch: track both for the buffers they read.
        let mut nc_read = vec![false; ir.buffers.len()];
        fn mark(r: &Region, nc_read: &mut [bool]) {
            for inst in &r.insts {
                if let Op::Load { buffer, attrs, .. } = &inst.op
                    && attrs.is_non_coherent()
                {
                    nc_read[buffer.0 as usize] = true;
                }
                inst.op.regions().into_iter().for_each(|r| mark(r, nc_read));
            }
        }
        mark(&ir.body, &mut nc_read);
        let nc = globals
            .iter()
            .zip(&nc_read)
            .map(|(g, &read)| {
                let bytes = g.as_ref().map(Vec::len).filter(|_| read)?;
                Some(NcState { read: vec![false; bytes], written: vec![false; bytes] })
            })
            .collect();
        let addrs = AddrMap::new(ir, &globals);
        Machine {
            ir,
            opts,
            pdl: None,
            cfg,
            globals,
            scalars,
            report: SimReport::default(),
            blocks: Vec::new(),
            threads_per_block: cfg.threads_per_block(),
            nthreads: 0,
            values: Vec::new(),
            values_hi: Vec::new(),
            shared: Vec::new(),
            locals: Vec::new(),
            ops: Vec::new(),
            pending_results: HashMap::new(),
            undefined: HashMap::new(),
            proxy_fences: Vec::new(),
            bulk_uncommitted: Vec::new(),
            bulk_groups: Vec::new(),
            pending_global: 0,
            cluster_filter: None,
            home: overlay_homes(ir),
            any_async_reads: false,
            uncommitted: Vec::new(),
            groups: Vec::new(),
            in_flight: Vec::new(),
            warps_per_block: cfg.threads_per_block().div_ceil(32),
            cluster_size: cfg.cluster.iter().map(|&c| c as usize).product(),
            stride: cfg.threads_per_block().div_ceil(32) + AGENT_SLOTS,
            clocks: Vec::new(),
            bulk_time: Vec::new(),
            named: Vec::new(),
            round: 0,
            epoch: 0,
            idle: 0,
            active: Vec::new(),
            warps: Vec::new(),
            parked: Vec::new(),
            next_join: 0,
            sched_rng: match opts.schedule {
                super::Schedule::Random { seed } => Some(seed),
                _ => None,
            },
            async_rng: match opts.async_completion {
                super::AsyncCompletion::Random(seed) => Some(seed ^ 0x5bd1_e995_a5a5_a5a5),
                _ => None,
            },
            steps: 0,
            first_live_op: 0,
            unaligned: HashMap::new(),
            setmaxnreg: HashMap::new(),
            cluster_barriers: Vec::new(),
            queued: VecDeque::new(),
            clc_launched: 0,
            clc_rng: opts.clc_launch_seed.map(|s| s ^ 0x9e37_79b9_7f4a_7c15),
            cluster_ids: Vec::new(),
            next_cluster_id: 0,
            cluster_arrival: Vec::new(),
            mbarriers: HashMap::new(),
            invalidated: HashMap::new(),
            exit_sites: HashMap::new(),
            clock: 0,
            loc: None,
            carry: Vec::new(),
            constants: constant_values(&ir.body),
            smem: ir.shared_memory_layout(),
            addrs,
            releases: HashMap::new(),
            fence_release: Vec::new(),
            pending_acquire: Vec::new(),
            init_clocks: Vec::new(),
            smem_target: None,
            dsmem_log: HashMap::new(),
            nc,
            wgmma: wgmma::State::new(ir),
            tmem: tmem::State::default(),
            tmaps: tensormap::State::new(ir),
            multicast: Vec::new(),
        }
    }

    /// Run only the clusters whose first block is in `origins` (the clusters of one shape
    /// in a launch whose clusters have mixed shapes).
    pub(super) fn set_cluster_filter(&mut self, origins: std::collections::HashSet<[u32; 3]>) {
        self.cluster_filter = Some(origins);
    }

    /// Record this launch's global-memory footprint and, with `primary`, check its
    /// accesses before `griddepcontrol_wait` against the primary's footprint.
    pub(super) fn set_pdl(&mut self, primary: Option<Rc<PdlPrimary>>) {
        let footprint = self.globals.iter().map(|g| g.as_ref().map(|g| vec![0u8; g.len()])).collect();
        self.pdl = Some(Box::new(PdlState { footprint, primary, waited: HashSet::new() }));
    }

    /// The footprint recorded since [`set_pdl`](Self::set_pdl): per buffer, per byte,
    /// [`PDL_READ`] and [`PDL_WRITTEN`] flags.
    pub(super) fn take_footprint(&mut self) -> Vec<Option<Vec<u8>>> {
        self.pdl.take().map(|p| p.footprint).unwrap_or_default()
    }

    /// One global access by thread `t` in the PDL model.
    fn pdl_access(&mut self, b: BufferId, bytes: std::ops::Range<usize>, t: usize, write: bool) -> R {
        let key = (self.block_of(t), t % self.threads_per_block);
        let pdl = self.pdl.as_mut().expect("the PDL model is on");
        if let Some(f) = pdl.footprint.get_mut(b.0 as usize).and_then(Option::as_mut) {
            let flag = if write { PDL_WRITTEN } else { PDL_READ };
            let end = bytes.end.min(f.len());
            f[bytes.start.min(end)..end].iter_mut().for_each(|x| *x |= flag);
        }
        let Some(primary) = pdl.primary.clone() else { return Ok(()) };
        if pdl.waited.contains(&key) {
            return Ok(());
        }
        let Some(p) = primary.footprint.get(b.0 as usize).and_then(Option::as_ref) else { return Ok(()) };
        let conflict = if write { PDL_READ | PDL_WRITTEN } else { PDL_WRITTEN };
        let Some(i) = bytes.clone().find(|&i| p.get(i).is_some_and(|f| f & conflict != 0)) else { return Ok(()) };
        let d = self.ir.buffer(b);
        let place = format!("`{}`[{}]", d.name, i / d.elem.bytes().max(1));
        let theirs = if p[i] & PDL_WRITTEN != 0 { "writes" } else { "reads" };
        let ours = if write { "writes" } else { "reads" };
        Err(self.err(
            format!(
                "a programmatic dependent launch {ours} {place} before griddepcontrol_wait(), but the primary kernel \
                 `{}` {theirs} it: with use_pdl the dependent's code before the wait may run while any part of the \
                 primary still runs; call griddepcontrol_wait() before this access",
                primary.name
            ),
            Some(t),
        ))
    }

    pub(super) fn finish(mut self) -> (Vec<Option<Vec<u8>>>, SimReport) {
        self.report.steps = self.steps;
        (self.globals, self.report)
    }

    /// Scheduler steps so far.
    pub(super) fn steps(&self) -> u64 {
        self.steps
    }

    /// Group `g` of warp `k`: 0 is the one it runs, then its parked ones.
    fn group(&self, k: usize, g: usize) -> &WarpState<'a> {
        if g == 0 { &self.warps[k] } else { &self.parked[k][g - 1] }
    }

    fn group_mut(&mut self, k: usize, g: usize) -> &mut WarpState<'a> {
        if g == 0 { &mut self.warps[k] } else { &mut self.parked[k][g - 1] }
    }

    /// Every convergence group of every warp, as `(warp, group)`.
    fn all_groups(&self) -> Vec<(usize, usize)> {
        (0..self.warps.len()).flat_map(|k| (0..=self.parked[k].len()).map(move |g| (k, g))).collect()
    }

    /// Make parked group `g` (> 0) of warp `k` the one it runs.
    fn swap_in(&mut self, k: usize, g: usize) {
        std::mem::swap(&mut self.warps[k], &mut self.parked[k][g - 1]);
    }

    fn block_rank(&self, t: usize) -> usize {
        t / self.threads_per_block
    }

    /// The block whose shared memory thread `t`'s access reaches: its own,
    /// or the peer of the distributed-shared-memory access being executed.
    fn smem_block(&self, t: usize) -> usize {
        self.smem_target.unwrap_or_else(|| self.block_rank(t))
    }

    /// Before thread `t` accesses (`what`) the shared memory of CTA `peer`
    /// of its cluster (distributed shared memory):
    /// the launch has clusters, the rank is in the cluster, that CTA has not
    /// exited, and (another CTA's) is known to have started: some arrival of
    /// it at a cluster barrier (or a release it made) is ordered before the
    /// access, as CUDA requires a cluster barrier after the kernel starts.
    /// The access is logged for the exit check of the peer. Returns the
    /// peer's block, which the access then targets ([`Self::smem_target`]).
    fn dsmem_target(&mut self, peer: ValueId, t: usize, what: &str) -> R<usize> {
        self.need_clusters(t, what)?;
        let block = self.mbarrier_block(Some(peer), t, what)?;
        let own = self.block_rank(t);
        if block == own {
            return Ok(block);
        }
        let w = self.warps_per_block;
        let (rank, own_rank) = (block % self.cluster_size, own % self.cluster_size);
        if (0..w).all(|warp| matches!(self.warps[block * w + warp].status, Status::Exited)) {
            return Err(self.err(
                format!(
                    "{what} by CTA {own_rank} reaches the shared memory of CTA {rank}, which has exited: its shared \
                     memory is gone (CUDA requires a cluster barrier before a CTA exits while its peers still use its \
                     shared memory)"
                ),
                Some(t),
            ));
        }
        let k = self.clock_index(t);
        let started = (0..w).any(|warp| {
            let s = self.slot_of(block * w + warp);
            self.clocks[k][s].max(self.init_clocks[k][s]) > 0
        });
        if !started {
            return Err(self.err(
                format!(
                    "{what} by CTA {own_rank} reaches the shared memory of CTA {rank} before anything shows that CTA \
                     {rank} has started: CUDA requires a cluster barrier (cluster_arrive or cluster_arrive_relaxed, \
                     then cluster_wait) after the kernel starts and before the first access to another CTA's shared \
                     memory"
                ),
                Some(t),
            ));
        }
        let slot = self.slot_of(k);
        let now = self.clocks[k][slot];
        if let Some(loc) = self.loc {
            let log = self.dsmem_log.entry(block).or_default();
            if log.get(&slot).is_none_or(|&(c, _)| c <= now) {
                log.insert(slot, (now, loc));
            }
        }
        Ok(block)
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
        let mut message: String = message.into();
        if let Some(path) = self.loc.and_then(|loc| self.ir.scope_path_at(loc)) {
            message.push_str(&format!(" (in profiling scope `{path}`)"));
        }
        SimError {
            message,
            loc: self.loc,
            block: thread.map_or_else(|| self.blocks.first().copied().unwrap_or([0; 3]), |t| self.block_of(t)),
            thread: thread.map(|t| self.thread_coord(t)),
            seed: None,
            steps: 0,
        }
    }

    fn get(&self, v: ValueId, t: usize) -> u64 {
        self.values[v.0 as usize][t]
    }

    /// An arithmetic, comparison or conversion instruction on 128-bit
    /// values ([`is_wide_op`]); the validator allows no others.
    fn wide_inst(&mut self, op: &Op, active: &[usize]) -> R {
        let ir = self.ir;
        match op {
            Op::Const { result, value } => {
                let bits = value.bits128();
                self.lanes128(active, *result, |_, _| Ok(bits))
            }
            Op::Unary { result, op, x } => {
                let ty = ir.dtype(*x);
                self.lanes128(active, *result, |m, t| Ok(wide::unary(*op, ty, m.get128(*x, t))))
            }
            Op::Binary { result, op, lhs, rhs } => {
                let ty = ir.dtype(*lhs);
                self.lanes128(active, *result, |m, t| wide::binary(*op, ty, m.get128(*lhs, t), m.get128(*rhs, t)))
            }
            Op::Cmp { result, pred, lhs, rhs } => {
                let ty = ir.dtype(*lhs);
                self.lanes128(active, *result, |m, t| {
                    Ok(wide::compare(*pred, ty, m.get128(*lhs, t), m.get128(*rhs, t)) as u128)
                })
            }
            Op::Select { result, cond, on_true, on_false } => self.lanes128(active, *result, |m, t| {
                Ok(if m.get(*cond, t) != 0 { m.get128(*on_true, t) } else { m.get128(*on_false, t) })
            }),
            Op::Cast { result, x } => {
                let (from, to) = (ir.dtype(*x), ir.dtype(*result));
                self.lanes128(active, *result, |m, t| {
                    let b = m.get128(*x, t);
                    Ok(if from.bits() == 128 {
                        wide::cast_from(from, to, b)
                    } else {
                        wide::cast_to(from, to, b as u64)
                    })
                })
            }
            Op::Bitcast { result, x } => self.lanes128(active, *result, |m, t| Ok(m.get128(*x, t))),
            _ => unreachable!("validated: {op:?} takes no 128-bit values"),
        }
    }

    fn set(&mut self, v: ValueId, t: usize, bits: u64) {
        let n = self.nthreads;
        let slot = &mut self.values[v.0 as usize];
        if slot.is_empty() {
            slot.resize(n, 0);
        }
        slot[t] = bits;
    }

    /// A value of up to 128 bits: the low word, and the high word of a
    /// 128-bit value (zero for the others).
    fn get128(&self, v: ValueId, t: usize) -> u128 {
        let hi = self.values_hi[v.0 as usize].get(t).copied().unwrap_or(0);
        self.get(v, t) as u128 | (hi as u128) << 64
    }

    /// Set a value of up to 128 bits. Only 128-bit values have a nonzero
    /// high word, so only they get a high-word entry.
    fn set128(&mut self, v: ValueId, t: usize, bits: u128) {
        self.set(v, t, bits as u64);
        let hi = (bits >> 64) as u64;
        let n = self.nthreads;
        let slot = &mut self.values_hi[v.0 as usize];
        if hi != 0 || !slot.is_empty() {
            if slot.is_empty() {
                slot.resize(n, 0);
            }
            slot[t] = hi;
        }
    }

    fn lanes128(&mut self, active: &[usize], result: ValueId, f: impl Fn(&Self, usize) -> Result<u128, String>) -> R {
        for &t in active {
            let bits = f(self, t).map_err(|m| self.err(m, Some(t)))?;
            self.set128(result, t, bits);
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

    /// An `i32` or `i64` buffer index.
    fn index(&self, v: ValueId, t: usize) -> i64 {
        match self.ir.dtype(v) {
            DType::I64 => self.get(v, t) as i64,
            _ => self.get(v, t) as u32 as i32 as i64,
        }
    }

    /// Write each hidden tensor-map parameter's descriptor (the simulator's encoding of
    /// [`KernelSignature::hidden_tensor_map`](crate::KernelSignature::hidden_tensor_map),
    /// from the tensor's address in this launch's address map and the scalar arguments).
    pub(super) fn bind_tensor_maps(&mut self, sig: &crate::KernelSignature) -> Result<(), String> {
        for t in sig.tensor_maps.iter().filter(|t| t.source.is_some()) {
            let src = t.source.as_ref().expect("filtered");
            let crate::ir::Param::Buffer(tensor) = self.ir.args[src.tensor_param] else {
                return Err(format!("parameter {} is not a tensor", src.tensor_param));
            };
            let scalar = |i: usize| match &self.ir.args[i] {
                crate::ir::Param::Scalar { value, .. } => {
                    let bits = self.scalars.iter().find(|(v, _)| v == value)?.1;
                    let ty = self.ir.dtype(*value);
                    Some(if ty.is_signed() {
                        sext(bits as u64, ty.bits())
                    } else {
                        mask(bits as u64, ty.bits()) as i64
                    })
                }
                _ => None,
            };
            let desc = sig.hidden_tensor_map(t, self.address_of(tensor, 0), &scalar)?;
            let crate::ir::Param::ByValue { buffer, .. } = self.ir.args[t.param] else {
                return Err(format!("parameter {} is not by value", t.param));
            };
            self.globals[buffer.0 as usize] = Some(desc.to_sim_bytes().to_vec());
        }
        Ok(())
    }

    /// Every buffer parameter's simulated address is aligned as the kernel
    /// declares (as the CUDA launcher checks the device pointers): always,
    /// since the simulator's bases are 256-byte aligned.
    pub(super) fn check_param_alignment(&self) -> Result<(), String> {
        for p in &self.ir.args {
            if let crate::ir::Param::Buffer(b) = p {
                let d = self.ir.buffer(*b);
                crate::check_pointer_alignment(&d.name, self.address_of(*b, 0), d.align)?;
            }
        }
        Ok(())
    }
}

/// Whether `.sync` warp collectives `a` and `b` can meet: the same
/// instruction kind with the same flavour (lanes of a member mask may reach
/// different instructions on sm_70 and newer).
fn same_collective(a: &Inst, b: &Inst) -> bool {
    match (&a.op, &b.op) {
        (Op::Shuffle { kind: x, .. }, Op::Shuffle { kind: y, .. }) => x == y,
        (Op::Vote { kind: x, .. }, Op::Vote { kind: y, .. }) => x == y,
        (Op::Match { kind: x, .. }, Op::Match { kind: y, .. }) => x == y,
        (Op::Redux { kind: x, abs: xa, nan: xn, .. }, Op::Redux { kind: y, abs: ya, nan: yn, .. }) => {
            (x, xa, xn) == (y, ya, yn) && a.op.operands().len() == b.op.operands().len()
        }
        (Op::WarpBarrier { .. }, Op::WarpBarrier { .. }) => true,
        _ => false,
    }
}

/// The result of a `.sync` warp collective, if it has one.
fn collective_result(inst: &Inst) -> Option<ValueId> {
    match &inst.op {
        Op::Shuffle { result, .. } | Op::Vote { result, .. } | Op::Match { result, .. } | Op::Redux { result, .. } => {
            Some(*result)
        }
        _ => None,
    }
}

/// Whether `a < b` for an integer type.
fn compare_less(ty: DType, a: u64, b: u64) -> bool {
    compare(CmpPred::Lt, ty, a, b)
}

/// One element of a TMA reduce store (`cp.reduce.async.bulk.tensor`) landing: `old` in
/// global memory combined with `x` from shared memory, as `kind` on `elem`. Float adds
/// round to nearest even and keep subnormals: `.f16` and `.bf16` adds are `.noftz`, and
/// the PTX ISA (9.4, `cp.reduce.async.bulk`) states that the `.f32` add's default is
/// `.noftz` too (earlier editions described the `.f32` add as flushing). `inc` and `dec` wrap as PTX's
/// atomics: `old >= x ? 0 : old + 1` and `old == 0 || old > x ? x : old - 1`. `min` and
/// `max` of floats ignore a NaN operand.
fn tma_reduce(kind: crate::cpasync::ReductionKind, elem: DType, old: u64, x: u64) -> u64 {
    use crate::cpasync::ReductionKind as K;
    let bits = elem.bits();
    match kind {
        K::ADD if elem.is_float() => super::float::atomic_add(elem, old, x, false),
        K::ADD => mask(old.wrapping_add(x), bits),
        K::MIN | K::MAX => {
            if elem.is_float() && is_nan(elem, x) {
                return old;
            }
            if elem.is_float() && is_nan(elem, old) {
                return x;
            }
            let less = compare_less(elem, x, old);
            if less == (kind == K::MIN) { x } else { old }
        }
        K::INC => {
            if old >= x {
                0
            } else {
                mask(old + 1, bits)
            }
        }
        K::DEC => {
            if old == 0 || old > x {
                x
            } else {
                old - 1
            }
        }
        K::AND => old & x,
        K::OR => old | x,
        K::XOR => old ^ x,
    }
}

/// The shared-memory allocation unit of sm_80 to sm_120 (PTX ISA,
/// `%total_smem_size`).
const SMEM_ALLOCATION_UNIT: usize = 128;

/// The shared memory reserved for the system per block from sm_80 on (the
/// CUDA programming guide), which `%aggr_smem_size` counts.
const SMEM_SYSTEM_RESERVED: usize = 1024;

/// `%total_smem_size` of a block of a kernel laid out as `smem` whose
/// dynamic region gets `dynamic` bytes: its static and dynamic shared memory,
/// "in multiples of the allocation unit", which an SM89 GPU reports as bytes
/// rounded up to a multiple of the unit (`tests/gpu.rs` checks 19 layouts:
/// static buffers in declaration order with their alignment padding, the
/// dynamic window, and both).
fn total_smem_size(smem: &SharedMemoryLayout, dynamic: usize) -> usize {
    smem.block_bytes(dynamic).next_multiple_of(SMEM_ALLOCATION_UNIT)
}

/// `%aggr_smem_size`: the block's shared memory plus the system's
/// reservation, in bytes. Unlike `%total_smem_size` it is not rounded up to
/// the allocation unit: a B200 reports 3,088 for a block of 2,064 bytes
/// (the one layout measured), where `%total_smem_size` is 2,176.
fn aggr_smem_size(smem: &SharedMemoryLayout, dynamic: usize) -> usize {
    smem.block_bytes(dynamic) + SMEM_SYSTEM_RESERVED
}

/// How a report names byte `i` of shared buffer `d`: by element
/// (`smem1[5]`), or by byte in the dynamic region, whose accesses have the
/// types of the pointers cast from it rather than one of its own.
fn smem_place(d: &BufferDecl, i: usize) -> String {
    if d.len.is_none() {
        format!("byte {i} of the dynamic shared memory region (get_dyn_smem)")
    } else {
        format!("{}[{}]", d.name, i / d.elem.bytes())
    }
}

/// The values defined by constants anywhere in `r`.
fn constant_values(r: &Region) -> std::collections::HashSet<ValueId> {
    let mut out = std::collections::HashSet::new();
    let mut stack = vec![r];
    while let Some(region) = stack.pop() {
        for inst in &region.insts {
            if let Op::Const { result, .. } = inst.op {
                out.insert(result);
            }
            stack.extend(inst.op.regions());
        }
    }
    out
}

/// The lane that a `shfl.sync` executed by lane `lane` reads, from its
/// operand `b` (the source lane or lane offset) and its operand `c` (the
/// segment mask in bits 12:8 and the clamp value in bits 4:0), as the PTX
/// ISA's pseudo-code computes it; `None` when the computed lane is out of
/// range, and the lane keeps its own value.
fn shuffle_source(kind: ShuffleKind, lane: usize, b: u32, c: u32) -> Option<usize> {
    let lane = lane as i32;
    let bval = (b & 0x1f) as i32;
    let cval = (c & 0x1f) as i32;
    let segmask = (c >> 8 & 0x1f) as i32;
    let max_lane = (lane & segmask) | (cval & !segmask);
    let min_lane = lane & segmask;
    let (j, in_range) = match kind {
        ShuffleKind::Up => (lane - bval, lane - bval >= max_lane),
        ShuffleKind::Down => (lane + bval, lane + bval <= max_lane),
        ShuffleKind::Bfly => (lane ^ bval, lane ^ bval <= max_lane),
        ShuffleKind::Idx => (min_lane | (bval & !segmask), (min_lane | (bval & !segmask)) <= max_lane),
    };
    in_range.then_some(j as usize)
}

/// `redux.sync` over the member lanes' raw 32-bit `inputs`, as the PTX ISA
/// defines it: `add` wraps at 32 bits, `min` and `max` compare as signed
/// and `umin` and `umax` as unsigned integers. On Float32, `.abs` reduces
/// the absolute values and `+0 > -0`; without `.NaN` the NaN inputs are
/// ignored (all NaN gives the canonical NaN), with it any NaN input makes
/// the result the canonical NaN.
fn redux(kind: ReduxKind, abs: bool, nan: bool, inputs: &[u32]) -> u32 {
    const CANONICAL_NAN: u32 = 0x7fff_ffff;
    let all = inputs.iter().copied();
    match kind {
        ReduxKind::Add => all.fold(0, u32::wrapping_add),
        ReduxKind::And => all.fold(u32::MAX, |a, b| a & b),
        ReduxKind::Or => all.fold(0, |a, b| a | b),
        ReduxKind::Xor => all.fold(0, |a, b| a ^ b),
        ReduxKind::Max => all.fold(i32::MIN, |a, b| a.max(b as i32)) as u32,
        ReduxKind::Min => all.fold(i32::MAX, |a, b| a.min(b as i32)) as u32,
        ReduxKind::Umax => all.fold(0, u32::max),
        ReduxKind::Umin => all.fold(u32::MAX, u32::min),
        ReduxKind::Fmax | ReduxKind::Fmin => {
            let values: Vec<u32> = all.map(|v| if abs { v & 0x7fff_ffff } else { v }).collect();
            let is_nan = |v: &u32| v & 0x7fff_ffff > 0x7f80_0000;
            if nan && values.iter().any(is_nan) {
                return CANONICAL_NAN;
            }
            // The numbers in order, -0 before +0: negative values reversed
            // below the positive ones.
            let order = |v: &u32| if v >> 31 == 1 { !v } else { v | 0x8000_0000 };
            let numbers = values.iter().filter(|v| !is_nan(v));
            let found = if kind == ReduxKind::Fmax {
                numbers.max_by_key(|v| order(v))
            } else {
                numbers.min_by_key(|v| order(v))
            };
            found.copied().unwrap_or(CANONICAL_NAN)
        }
    }
}

#[cfg(test)]
mod async_tests {
    use super::*;
    use crate::prelude::*;

    /// A machine with one block of 64 threads and a 64-byte shared buffer.
    fn with_machine(f: impl FnOnce(&mut Machine)) {
        let kernel = Kernel::builder("hooks").block_dim([64, 1, 1]).build(|| {
            let smem = make_smem_tensor::<u32>(Layout::col_major(16));
            smem.store(thread_idx().x % 16, 1u32);
        });
        let ir = kernel.ir();
        let opts = SimOptions::default();
        let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
        let mut m = Machine::new(ir, &opts, cfg, vec![None; ir.buffers.len()], Vec::new());
        m.allocate(1);
        m.start_cluster(0, vec![[0, 0, 0]]);
        f(&mut m);
    }

    fn smem(m: &Machine) -> BufferId {
        BufferId(m.ir.buffers.iter().position(|b| b.space == MemSpace::Shared).unwrap() as u32)
    }

    /// Register results: reading one before the operation completes is
    /// reported; afterwards it holds the result.
    #[test]
    fn register_results_are_pending_until_completion() {
        with_machine(|m| {
            let v = ValueId(0);
            let id = m.issue_test_op(3, Vec::new(), Vec::new(), vec![(v, 3, 42)]).unwrap();
            let e = m.check_pending_operands(&[v], &[3]).unwrap_err();
            assert!(e.message.contains("reads the result of an asynchronous operation before waiting for it"), "{e}");
            assert!(m.check_pending_operands(&[v], &[4]).is_ok());
            m.complete_as_agent(id, false);
            assert!(m.check_pending_operands(&[v], &[3]).is_ok());
            assert_eq!(m.get(v, 3), 42);
        });
    }

    /// Background reads: overwriting the source before completion is a race,
    /// as is the block exiting; writes land at completion as the agent's.
    #[test]
    fn async_reads_and_writes() {
        with_machine(|m| {
            let b = smem(m);
            let write = AsyncWrite { dst: b, block: 0, dst_byte: 32, data: vec![7; 8] };
            let read = AsyncRead { buffer: b, block: 0, bytes: 0..16 };
            let id = m.issue_test_op(0, vec![write], vec![read], Vec::new()).unwrap();
            let e = m.write_as(b, 4, 4, 33, 1, false).unwrap_err();
            assert!(
                e.message.contains("data race: warp 1 overwrites smem0[1], which an asynchronous operation"),
                "{e}"
            );
            let e = m.read_as(b, 32, 4, 0, false).unwrap_err();
            assert!(e.message.contains("still writing it"), "{e}");
            for k in 0..2 {
                m.warps[k].status = Status::Exited;
            }
            let e = m.check_block_exit(0).unwrap_err();
            assert!(
                e.message.contains("exits while an asynchronous copy is still writing its shared memory (smem0[8])"),
                "{e}"
            );
            for k in 0..2 {
                m.warps[k].status = Status::Runnable;
            }
            let time = m.complete_as_agent(id, false);
            assert!(m.write_as(b, 4, 4, 33, 1, false).is_ok());
            let meta = m.shared[b.0 as usize][0].meta[32];
            assert_eq!((meta.w_slot as usize, meta.w_clock), (m.agent_slot(0), time));
            assert_eq!(m.shared[b.0 as usize][0].data[32..40], [7; 8]);
        });
    }

    /// The proxy bookkeeping: a generic write needs a proxy fence in the
    /// writing warp, ordered before the async read's issue.
    #[test]
    fn async_proxy_reads_need_a_proxy_fence() {
        with_machine(|m| {
            let b = smem(m);
            m.write_as(b, 0, 4, 0, 5, false).unwrap();
            let e = m.async_proxy_may_read(b, 0, 0..4, 0).unwrap_err();
            assert!(e.contains("without a fence.proxy.async since"), "{e}");
            m.fence_proxy_async(0);
            assert!(m.async_proxy_may_read(b, 0, 0..4, 0).is_ok());
            // Warp 1 has not synchronised with warp 0's fence.
            let e = m.async_proxy_may_read(b, 0, 0..4, 1).unwrap_err();
            assert!(e.contains("nothing orders the fence before this operation's issue"), "{e}");
            m.synchronize(0, &[0], &[1], &[]);
            assert!(m.async_proxy_may_read(b, 0, 0..4, 1).is_ok());
            // A write after the fence needs another.
            m.write_as(b, 0, 4, 0, 6, false).unwrap();
            assert!(m.async_proxy_may_read(b, 0, 0..4, 0).is_err());
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One element of a TMA reduce store, against the PTX ISA's `cp.reduce.async.bulk`
    /// rules: float adds round to nearest even and keep subnormals (`.f32` included),
    /// `inc`/`dec` wrap at the source value, `min`/`max` follow the element's signedness.
    #[test]
    fn tma_reduce_elements() {
        use crate::cpasync::ReductionKind as K;
        let sub = f32::from_bits(1).to_bits() as u64; // the smallest f32 subnormal
        let r = tma_reduce(K::ADD, DType::F32, sub, sub);
        assert_eq!(r, 2, "an f32 add keeps subnormal inputs and results");
        let one = 1.0f32.to_bits() as u64;
        let tiny = (f32::EPSILON / 2.0).to_bits() as u64; // a tie: rounds to even (1.0)
        assert_eq!(tma_reduce(K::ADD, DType::F32, one, tiny), one);
        assert_eq!(tma_reduce(K::ADD, DType::F16, 0x0001, 0x0001), 0x0002, "an f16 add is .noftz");
        assert_eq!(tma_reduce(K::INC, DType::U32, 5, 5), 0);
        assert_eq!(tma_reduce(K::INC, DType::U32, 4, 5), 5);
        assert_eq!(tma_reduce(K::DEC, DType::U32, 0, 5), 5);
        assert_eq!(tma_reduce(K::DEC, DType::U32, 7, 5), 5);
        assert_eq!(tma_reduce(K::DEC, DType::U32, 3, 5), 2);
        assert_eq!(tma_reduce(K::MIN, DType::I32, 1, 0xffff_ffff), 0xffff_ffff, "s32 min: -1 < 1");
        assert_eq!(tma_reduce(K::MIN, DType::U32, 1, 0xffff_ffff), 1);
        assert_eq!(tma_reduce(K::MAX, DType::I64, 1, u64::MAX), 1);
    }

    /// The source lanes of `shfl.sync`, worked from the PTX ISA's
    /// pseudo-code.
    #[test]
    fn shuffle_sources() {
        use ShuffleKind::*;
        // The whole warp: c = 31 (0 for up).
        assert_eq!(shuffle_source(Idx, 9, 3, 31), Some(3));
        assert_eq!(shuffle_source(Idx, 9, 35, 31), Some(3), "b[4:0]");
        assert_eq!(shuffle_source(Up, 1, 2, 0), None);
        assert_eq!(shuffle_source(Up, 2, 2, 0), Some(0));
        assert_eq!(shuffle_source(Up, 31, 1, 31), None, "up with c = 31 keeps every lane's own value");
        assert_eq!(shuffle_source(Down, 27, 5, 31), None);
        assert_eq!(shuffle_source(Down, 26, 5, 31), Some(31));
        assert_eq!(shuffle_source(Bfly, 6, 1, 31), Some(7));
        // Segments of 8 lanes: c = 0x1807 (0x1800 for up).
        assert_eq!(shuffle_source(Idx, 13, 5, 0x1807), Some(13));
        assert_eq!(shuffle_source(Idx, 13, 2, 0x1807), Some(10));
        assert_eq!(shuffle_source(Idx, 13, 10, 0x1807), Some(10), "bits above the segment are ignored");
        assert_eq!(shuffle_source(Up, 9, 2, 0x1800), None);
        assert_eq!(shuffle_source(Up, 10, 2, 0x1800), Some(8));
        assert_eq!(shuffle_source(Down, 14, 2, 0x1807), None);
        assert_eq!(shuffle_source(Down, 13, 2, 0x1807), Some(15));
        // A clamp below the segment's top: lanes past it are out of range.
        assert_eq!(shuffle_source(Idx, 9, 6, 0x1803), None);
        assert_eq!(shuffle_source(Idx, 9, 2, 0x1803), Some(10));
        // A butterfly may leave its segment when the lane is at most maxLane.
        assert_eq!(shuffle_source(Bfly, 3, 8, 0x1807), None);
        assert_eq!(shuffle_source(Bfly, 11, 8, 0x1807), Some(3));
    }

    #[test]
    fn redux_semantics() {
        let ints = [5u32, 0xffff_fffe, 0x8000_0000, 7];
        assert_eq!(redux(ReduxKind::Add, false, false, &ints), 0x8000_000a);
        assert_eq!(redux(ReduxKind::Max, false, false, &ints), 7);
        assert_eq!(redux(ReduxKind::Min, false, false, &ints), 0x8000_0000);
        assert_eq!(redux(ReduxKind::Umax, false, false, &ints), 0xffff_fffe);
        assert_eq!(redux(ReduxKind::Umin, false, false, &ints), 5);
        assert_eq!(redux(ReduxKind::And, false, false, &[0b1100, 0b1010]), 0b1000);
        assert_eq!(redux(ReduxKind::Or, false, false, &[0b1100, 0b1010]), 0b1110);
        assert_eq!(redux(ReduxKind::Xor, false, false, &[0b1100, 0b1010]), 0b0110);
        let f = |x: f32| x.to_bits();
        let floats = [f(-3.0), 0x7fc0_0000, f(2.0), 0x8000_0000];
        assert_eq!(redux(ReduxKind::Fmax, false, false, &floats), f(2.0));
        assert_eq!(redux(ReduxKind::Fmin, false, false, &floats), f(-3.0));
        assert_eq!(redux(ReduxKind::Fmax, true, false, &floats), f(3.0));
        assert_eq!(redux(ReduxKind::Fmin, true, false, &floats), 0);
        assert_eq!(redux(ReduxKind::Fmax, false, true, &floats), 0x7fff_ffff);
        assert_eq!(redux(ReduxKind::Fmin, false, false, &[0x7fc0_0000, 0xffc0_0001]), 0x7fff_ffff);
        assert_eq!(redux(ReduxKind::Fmax, false, false, &[0x8000_0000, 0]), 0, "+0 > -0");
        assert_eq!(redux(ReduxKind::Fmin, false, false, &[0, 0x8000_0000]), 0x8000_0000);
    }
}

/// A 16-byte `ldmatrix` source row as 16 8-bit containers: the bytes
/// themselves, or with `unpack` the 16 4-bit (`.b4x16_p64`, in the low 8
/// bytes) or 6-bit (`.b6x16_p32`, in the low 12 bytes) values packed from
/// the least significant bit, the rest padding, each zero-extended to its
/// byte: a 4-bit value in bits 0 to 3 and a 6-bit value in bits 0 to 5. The
/// PTX ISA calls the source elements unsigned without saying where the
/// padding goes; a B200 gives these bytes, and CuTe C++ names the atoms
/// `SM100_SU4_DU8x16_*_LDSM_N`.
///
/// `mma` with `.kind::f8f6f4` takes E3M2 and E2M3 in the low 6 bits of a
/// byte but E2M1 in its central 4, so CuTe C++ shifts a 4-bit operand
/// loaded this way left by two before the MMA (`fp4_shift_A`,
/// `fp4_shift_B`).
pub(crate) fn ldmatrix_expand(row: u128, unpack: Option<u8>) -> [u8; 16] {
    std::array::from_fn(|e| match unpack {
        None => (row >> (8 * e)) as u8,
        Some(4) => ((row >> (4 * e)) & 0xf) as u8,
        Some(_) => ((row >> (6 * e)) & 0x3f) as u8,
    })
}

#[cfg(test)]
mod ldmatrix_expand_tests {
    use super::ldmatrix_expand;

    /// Hand-built padded rows: 16 4-bit values 0 to 15 in the low 8 bytes
    /// (padding all ones), and 16 6-bit values `4e + 3` in the low 12.
    #[test]
    fn padded_rows_expand_from_the_least_significant_bit() {
        let mut b4 = 0u128;
        for e in 0..16u128 {
            b4 |= e << (4 * e);
        }
        b4 |= u128::MAX << 64;
        assert_eq!(ldmatrix_expand(b4, Some(4)), std::array::from_fn(|e| e as u8));
        let mut b6 = 0u128;
        for e in 0..16u128 {
            b6 |= (4 * e + 3) << (6 * e);
        }
        b6 |= u128::MAX << 96;
        assert_eq!(ldmatrix_expand(b6, Some(6)), std::array::from_fn(|e| 4 * e as u8 + 3));
        let bytes = u128::from_le_bytes(std::array::from_fn(|k| 0xa0 + k as u8));
        assert_eq!(ldmatrix_expand(bytes, None), std::array::from_fn(|k| 0xa0 + k as u8));
    }
}

/// A byte the launch read, in a PDL footprint.
pub(super) const PDL_READ: u8 = 1;
/// A byte the launch wrote, in a PDL footprint.
pub(super) const PDL_WRITTEN: u8 = 2;

/// The primary of a programmatic dependent launch: its name and its global-memory footprint,
/// mapped onto the dependent's buffers.
pub(super) struct PdlPrimary {
    pub name: String,
    pub footprint: Vec<Option<Vec<u8>>>,
}

/// The PDL model's state in one launch.
pub(super) struct PdlState {
    footprint: Vec<Option<Vec<u8>>>,
    primary: Option<Rc<PdlPrimary>>,
    /// The threads (block, thread in block) that have passed `griddepcontrol.wait`.
    waited: HashSet<([u32; 3], usize)>,
}

// ---- TMA im2col and SM100 modes ----

/// A TMA instruction's operands, for the simulator.
struct TmaIssue<'a> {
    desc: ValueId,
    coords: &'a [ValueId],
    mode: &'a crate::ir::TmaMode,
    overrides: Option<&'a crate::ir::TmaOverride>,
}

/// The descriptor-level access of an instruction in `mode` at `start` (with `offsets`).
fn tma_access<'a>(mode: &crate::ir::TmaMode, start: &'a [i64], offsets: &'a [i64]) -> crate::tma::TmaAccess<'a> {
    use crate::ir::TmaMode;
    match mode {
        TmaMode::Tile => crate::tma::TmaAccess::Tile { start },
        TmaMode::Im2col { .. } | TmaMode::Im2colNoOffs => crate::tma::TmaAccess::Im2col { start, offsets },
        TmaMode::Gather4 | TmaMode::Scatter4 => {
            crate::tma::TmaAccess::Rows4 { x: start[0], rows: [start[1], start[2], start[3], start[4]] }
        }
    }
}

/// The low `bits` bits set.
fn mask_bits(bits: usize) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

/// Write the low `bits` bits of `v` at bit `at` of `image`, marking the bytes written.
fn put_bits(image: &mut [u8], written: &mut [bool], at: usize, bits: usize, v: u64) {
    for k in 0..bits {
        let (byte, bit) = ((at + k) / 8, (at + k) % 8);
        image[byte] = (image[byte] & !(1 << bit)) | ((((v >> k) & 1) as u8) << bit);
        written[byte] = true;
    }
}

/// The `bits` bits at bit `at` of `image`.
fn get_bits(image: &[u8], at: usize, bits: usize) -> u64 {
    if at.is_multiple_of(8) && bits.is_multiple_of(8) {
        return image[at / 8..at / 8 + bits / 8].iter().rev().fold(0u64, |a, &b| (a << 8) | u64::from(b));
    }
    (0..bits).fold(0u64, |a, k| a | ((u64::from(image[(at + k) / 8] >> ((at + k) % 8)) & 1) << k))
}
