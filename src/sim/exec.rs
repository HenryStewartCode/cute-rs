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
// m10-tmem-core.
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
/// `cp.async` (with `cp.async.mbarrier.arrive{.noinc}`) and `cp.async.bulk`
/// today, and the hooks for TMA, bulk async-groups and proxy fences (m7),
/// WGMMA groups (m8) and tcgen05 (m10).
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

    pub(super) fn run(&mut self) -> R {
        let [gx, gy, gz] = self.cfg.grid;
        let [cx, cy, cz] = self.cfg.cluster;
        let mut clusters = VecDeque::new();
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
                    if self.cluster_filter.as_ref().is_none_or(|f| f.contains(&[x, y, z])) {
                        clusters.push_back(blocks);
                    }
                }
            }
        }
        let resident = match self.opts.resident_clusters {
            super::ResidentClusters::Count(n) => n.max(1) as usize,
            super::ResidentClusters::Grid => clusters.len(),
        }
        .min(clusters.len());
        self.allocate(resident);
        for c in 0..resident {
            let blocks = clusters.pop_front().expect("a cluster per resident slot");
            self.start_cluster(c, blocks);
        }
        self.queued = clusters;
        self.schedule()
    }

    /// Allocate the state of `resident` clusters that run together; each
    /// cluster that starts in a slot ([`Self::start_cluster`]) resets its
    /// part.
    fn allocate(&mut self, resident: usize) {
        let count = resident * self.cluster_size;
        let w = self.warps_per_block;
        let slots = self.slots();
        self.blocks = vec![[0; 3]; count];
        self.cluster_ids = vec![0; resident];
        self.nthreads = self.threads_per_block * count;
        self.loc = None;
        self.clocks = vec![vec![0; slots]; count * w];
        self.init_clocks = vec![vec![0; slots]; count * w];
        self.fence_release = vec![None; count * w];
        self.pending_acquire = vec![None; count * w];
        self.bulk_time = vec![0; count];
        self.named = vec![vec![NamedState::default(); 16]; count];
        self.values = vec![Vec::new(); self.ir.value_types.len()];
        let race_bytes = |bytes: usize| if self.opts.detect_races { Reads::new(bytes) } else { Reads::default() };
        self.values_hi = vec![Vec::new(); self.ir.value_types.len()];
        // The dynamic region gets the launch's bytes.
        let dynamic = self.cfg.dynamic_smem_bytes as usize;
        self.shared = self
            .ir
            .buffers
            .iter()
            .enumerate()
            .map(|(k, b)| {
                if b.space != MemSpace::Shared || self.home[k] != k {
                    return Vec::new();
                }
                let bytes = storage_bytes(self.ir, &self.home, k, dynamic);
                (0..count)
                    .map(|_| Shared {
                        data: Vec::new(),
                        meta: Vec::new(),
                        reads: race_bytes(bytes),
                        strong_reads: Reads::default(),
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
        self.ops.clear();
        self.pending_results.clear();
        self.undefined.clear();
        self.wgmma.reset(count * w);
        self.tmaps.reset();
        self.proxy_fences = vec![Vec::new(); count * w];
        self.uncommitted = (0..nthreads).map(|_| Vec::new()).collect();
        self.bulk_uncommitted = (0..nthreads).map(|_| Vec::new()).collect();
        self.bulk_groups = (0..nthreads).map(|_| VecDeque::new()).collect();
        self.pending_global = 0;
        self.groups = (0..nthreads).map(|_| VecDeque::new()).collect();
        self.carry = vec![None; nthreads];
        self.in_flight = (0..nthreads).map(|_| Vec::new()).collect();
        self.cluster_arrival = vec![None; nthreads];
        self.warps = Vec::with_capacity(count * w);
        self.cluster_barriers = (0..resident).map(|_| ClusterBarrier::default()).collect();
    }

    /// Start the cluster of `blocks` in resident slot `c`: fresh shared
    /// memory, clocks (which start again with each cluster, so what other
    /// clusters published orders nothing here), barriers and warps.
    fn start_cluster(&mut self, c: usize, blocks: Vec<[u32; 3]>) {
        let (w, tpb, cs) = (self.warps_per_block, self.threads_per_block, self.cluster_size);
        let slots = self.slots();
        let block_range = c * cs..(c + 1) * cs;
        let warp_range = c * cs * w..(c + 1) * cs * w;
        let thread_range = c * cs * tpb..(c + 1) * cs * tpb;
        self.next_cluster_id += 1;
        self.cluster_ids[c] = self.next_cluster_id;
        self.blocks[block_range.clone()].copy_from_slice(&blocks);
        for k in warp_range.clone() {
            self.clocks[k].fill(0);
            let slot = self.slot_of(k);
            self.clocks[k][slot] = 1;
            self.init_clocks[k].fill(0);
            self.fence_release[k] = None;
            self.pending_acquire[k] = None;
            self.proxy_fences[k].clear();
        }
        for block in block_range.clone() {
            self.bulk_time[block] = 0;
            self.named[block] = vec![NamedState::default(); 16];
        }
        self.mbarriers.retain(|key, _| !block_range.contains(&key.0));
        self.releases.retain(|key, _| !block_range.contains(&key.1));
        self.dsmem_log.retain(|block, _| !block_range.contains(block));
        // With every other cluster finished (always, with one resident
        // cluster), nothing that runs refers to earlier asynchronous
        // operations or their mbarriers: forget them, so that memory and the
        // scans over operations stay per cluster rather than per launch.
        let others_done =
            self.warps.iter().enumerate().all(|(k, s)| warp_range.contains(&k) || matches!(s.status, Status::Exited));
        if others_done {
            self.ops.clear();
            self.tmaps.forget_ops();
            self.first_live_op = 0;
            self.pending_results.clear();
            self.any_async_reads = false;
            self.mbarriers.clear();
        }
        self.unaligned.retain(|key, _| !block_range.contains(&key.0));
        self.setmaxnreg.retain(|block, _| !block_range.contains(block));
        self.tmem.start_cluster(block_range.clone());
        self.wgmma.start_cluster(warp_range.clone(), thread_range.clone(), block_range.clone(), others_done);
        let dynamic = self.cfg.dynamic_smem_bytes as usize;
        for (b, decl) in self.ir.buffers.iter().enumerate() {
            if decl.space != MemSpace::Shared || self.home[b] != b {
                continue;
            }
            let bytes = storage_bytes(self.ir, &self.home, b, dynamic);
            for block in block_range.clone() {
                let sh = &mut self.shared[b][block];
                sh.data = vec![0xff; bytes];
                sh.meta = vec![Meta::default(); bytes];
                if !sh.reads.is_empty() {
                    sh.reads = Reads::new(bytes);
                }
                sh.strong_reads = Reads::default();
            }
        }
        for mem in self.locals.iter_mut().flatten() {
            let bytes = mem.per_thread * thread_range.start..mem.per_thread * thread_range.end;
            mem.data[bytes.clone()].fill(0xff);
            mem.written[bytes].fill(false);
        }
        for t in thread_range.clone() {
            self.uncommitted[t].clear();
            self.groups[t].clear();
            self.bulk_uncommitted[t].clear();
            self.bulk_groups[t].clear();
            self.in_flight[t].clear();
            self.carry[t] = None;
            self.cluster_arrival[t] = None;
        }
        for (v, bits) in self.scalars.clone() {
            for t in thread_range.clone() {
                self.set128(v, t, bits);
            }
        }
        let body = &self.ir.body;
        for k in warp_range {
            let start = (k / w) * tpb + (k % w) * 32;
            let end = (start + 32).min((k / w + 1) * tpb);
            let frame = Frame { region: body, next: 0, lanes: (start..end).collect(), kind: FrameKind::Body };
            let state = WarpState::new(vec![frame]);
            if k < self.warps.len() {
                self.warps[k] = state;
                self.parked[k].clear();
            } else {
                self.warps.push(state);
                self.parked.push(Vec::new());
            }
        }
        self.cluster_barriers[c] = ClusterBarrier {
            phase: 0,
            release: vec![0; slots],
            completed: vec![0; slots],
            init_release: vec![0; slots],
            init_completed: vec![0; slots],
        };
        self.report.blocks += cs as u64;
    }

    /// Start the next queued cluster in each resident slot whose cluster
    /// has finished. Returns whether one started.
    fn refill(&mut self) -> bool {
        let per_cluster = self.cluster_size * self.warps_per_block;
        let mut started = false;
        for c in 0..self.cluster_ids.len() {
            if self.queued.is_empty() {
                break;
            }
            let done =
                self.warps[c * per_cluster..(c + 1) * per_cluster].iter().all(|s| matches!(s.status, Status::Exited));
            if done {
                let blocks = self.queued.pop_front().expect("a queued cluster");
                self.clc_launched = self.clc_launched.saturating_sub(1);
                self.start_cluster(c, blocks);
                started = true;
            }
        }
        started
    }

    /// Run the cluster's warps until every one has exited, by the schedule
    /// ([`super::Schedule`]). With `RoundRobin`, each round the scheduler
    /// hands every runnable warp one instruction in turn; warps that stand
    /// at the same instruction one after another in that order execute it
    /// together, which is the same thing, but cheaper (and it lets a barrier
    /// see the warps that reach it together). With `RunToBlock`, each round
    /// one warp runs until it blocks, yields or exits.
    fn schedule(&mut self) -> R {
        let mut next_warp = 0;
        loop {
            self.round += 1;
            self.wake_mbarrier_waiters()?;
            let progress = match self.opts.schedule {
                super::Schedule::RoundRobin => {
                    let mut progress = false;
                    let mut cursor = 0;
                    while let Some(group) = self.next_group(&mut cursor)? {
                        progress = true;
                        self.step(&group)?;
                        self.check_idle()?;
                    }
                    progress
                }
                super::Schedule::RunToBlock => self.run_to_block(&mut next_warp)?,
                super::Schedule::Random { .. } => self.run_random()?,
            };
            let runnable =
                self.all_groups().into_iter().any(|(k, g)| matches!(self.group(k, g).status, Status::Runnable));
            if progress || runnable {
                self.refill();
                continue;
            }
            if self.refill() {
                continue;
            }
            if self.warps.iter().all(|s| matches!(s.status, Status::Exited)) {
                if self.refill() {
                    continue;
                }
                return Ok(());
            }
            // Nothing can run. An `mbarrier.try_wait` may give up after a
            // while (it then returns false); a barrier never does. One that
            // has given up again and again, with nothing visible happening
            // in between, would only give up again.
            let mut timed_out = false;
            let epoch = self.epoch;
            for state in self.warps.iter_mut().chain(self.parked.iter_mut().flatten()) {
                if let Status::Blocked(Wait::Mbarrier { .. }) = state.status {
                    let give_ups = match state.timeout_epoch {
                        Some((e, n)) if e == epoch => n,
                        _ => 0,
                    };
                    if give_ups >= GIVE_UPS_WITHOUT_PROGRESS {
                        continue;
                    }
                    state.status = Status::Runnable;
                    state.timed_out = true;
                    state.timeout_epoch = Some((epoch, give_ups + 1));
                    timed_out = true;
                }
            }
            if !timed_out {
                return Err(self.deadlock());
            }
        }
    }

    /// One round of [`super::Schedule::RunToBlock`]: the first runnable warp
    /// from `*next` on (cyclically) runs until it blocks, yields or exits,
    /// and the next round starts after it. Returns whether a warp ran.
    fn run_to_block(&mut self, next: &mut usize) -> R<bool> {
        let n = self.warps.len();
        for i in 0..n {
            let k = (*next + i) % n;
            self.take_turn(k);
            if self.current_inst(k)?.is_none() {
                continue;
            }
            self.warps[k].yielded = false;
            loop {
                self.step(&[k])?;
                self.check_idle()?;
                let at = self.current_inst(k)?;
                if at.is_none() || self.warps[k].yielded {
                    break;
                }
            }
            *next = k + 1;
            return Ok(true);
        }
        Ok(false)
    }

    /// One step of [`super::Schedule::Random`]: a convergence group drawn
    /// from those that can run runs up to its next scheduling point
    /// ([`Self::scheduling_point`]), its next yield, or until it blocks or
    /// ends. Returns whether one ran.
    fn run_random(&mut self) -> R<bool> {
        let round = self.round;
        let ready: Vec<(usize, usize)> =
            self.all_groups().into_iter().filter(|&(k, g)| self.group(k, g).ready(round)).collect();
        if ready.is_empty() {
            return Ok(false);
        }
        let rng = self.sched_rng.as_mut().expect("a seeded schedule");
        let (k, g) = ready[(splitmix(rng) % ready.len() as u64) as usize];
        if g > 0 {
            self.swap_in(k, g);
        }
        self.warps[k].yielded = false;
        let mut first = true;
        while let Some(inst) = self.front_inst(k)? {
            if !first && (self.warps[k].yielded || self.scheduling_point(&inst.op)) {
                break;
            }
            first = false;
            self.step(&[k])?;
            self.check_idle()?;
        }
        Ok(true)
    }

    /// Whether the seeded schedule may switch groups before `op`: anything
    /// another warp could observe or be affected by (memory other than
    /// local, barriers, asynchronous operations, opaque statements).
    fn scheduling_point(&self, op: &Op) -> bool {
        match op {
            Op::Load { buffer, .. } | Op::Store { buffer, .. } => self.ir.buffer(*buffer).space != MemSpace::Local,
            Op::LdMatrix { .. }
            | Op::StMatrix { .. }
            | Op::SetMaxNReg { .. }
            | Op::Tcgen05 { .. }
            | Op::Tcgen05Ld { .. }
            | Op::Tcgen05St { .. }
            | Op::Tcgen05Mma { .. }
            | Op::Tcgen05Commit { .. }
            | Op::Tcgen05Cp { .. }
            | Op::ClcTryCancel { .. }
            | Op::ClcQueryCancel { .. }
            | Op::Atomic { .. }
            | Op::Red { .. }
            | Op::Barrier
            | Op::NamedBarrier { .. }
            | Op::ClusterArrive { .. }
            | Op::ClusterWait { .. }
            | Op::MbarrierInit { .. }
            | Op::MbarrierArrive { .. }
            | Op::MbarrierExpectTx { .. }
            | Op::MbarrierWait { .. }
            | Op::FenceMbarrierInit
            | Op::GridDepControl { .. }
            | Op::CpAsync { .. }
            | Op::CpAsyncCommitGroup
            | Op::CpAsyncWaitGroup { .. }
            | Op::CpAsyncBulk { .. }
            | Op::CpAsyncBulkCommitGroup
            | Op::CpAsyncBulkWaitGroup { .. }
            | Op::TmaLoad { .. }
            | Op::TmaStore { .. }
            | Op::TmaPrefetch { .. }
            | Op::PrefetchTensormap { .. }
            | Op::TensormapReplace { .. }
            | Op::FenceTensormapRelease
            | Op::FenceTensormapAcquire { .. }
            | Op::TensormapCpFenceproxy { .. }
            | Op::BulkPrefetch { .. }
            | Op::StAsync { .. }
            | Op::FenceProxy { .. }
            | Op::CpAsyncMbarrierArrive { .. }
            | Op::Fence { .. }
            | Op::InlinePtx { .. }
            | Op::MlirOp { .. }
            | Op::Assert { .. }
            | Op::Printf { .. }
            | Op::Wgmma { .. }
            | Op::WgmmaFence { .. }
            | Op::WgmmaCommitGroup
            | Op::WgmmaWait { .. }
            | Op::BarrierRed { .. }
            | Op::MbarrierExt { .. }
            | Op::PrimMarker { .. }
            | Op::Exit
            | Op::StBulk { .. }
            | Op::Tcgen05Shift { .. } => true,
            _ => false,
        }
    }

    /// At a scheduling point, with [`super::AsyncCompletion::Random`]:
    /// complete an operation in flight, drawn at random, one time in four.
    fn complete_at_random(&mut self) {
        // Arrive-ons owed for copies that have all landed (by now, or at
        // their thread's cp_async_wait_group) fire.
        let owed: Vec<MbarrierKey> =
            self.mbarriers.iter().filter(|(_, m)| !m.deferred.is_empty()).map(|(&key, _)| key).collect();
        for key in owed {
            self.fire_ready_arrivals(key);
        }
        let Some(rng) = self.async_rng.as_mut() else { return };
        if !splitmix(rng).is_multiple_of(4) {
            return;
        }
        while self.first_live_op < self.ops.len() && !self.ops[self.first_live_op].in_flight() {
            self.first_live_op += 1;
        }
        let live: Vec<usize> = (self.first_live_op..self.ops.len()).filter(|&id| self.ops[id].in_flight()).collect();
        if live.is_empty() {
            return;
        }
        let rng = self.async_rng.as_mut().expect("checked above");
        let id = live[(splitmix(rng) % live.len() as u64) as usize];
        match self.ops[id].trigger {
            Trigger::CpAsync { .. } => {
                // Owed arrive-ons fire at the next scheduling point.
                self.land_tracked_copy(id);
            }
            Trigger::CompleteTx(key) => self.complete_bulk_copy(id, key),
            Trigger::BulkGroup => self.land_bulk_store(id),
            // Its results land at the warp's wait, so that reading them
            // earlier is always reported.
            Trigger::Wgmma | Trigger::TmemLd => {}
            Trigger::Tcgen05Mma => {
                self.umma_land(id);
            }
            #[cfg(test)]
            Trigger::Test => {}
        }
    }

    /// At the start of warp `k`'s turn (round-robin or run-to-block): if the
    /// group it runs cannot run, or yielded since its last turn, while
    /// another of its groups can, the next of those takes over (the others
    /// keep their order), so groups of one warp that wait for each other
    /// (a spin on a flag the other branch sets) all make progress.
    fn take_turn(&mut self, k: usize) {
        if self.parked[k].is_empty() {
            return;
        }
        let round = self.round;
        if self.warps[k].ready(round) && !self.warps[k].yielded {
            return;
        }
        let Some(i) = self.parked[k].iter().position(|g| g.ready(round)) else { return };
        let next = self.parked[k].remove(i);
        let mut old = std::mem::replace(&mut self.warps[k], next);
        old.yielded = false;
        self.parked[k].push(old);
    }

    /// Something visible happened (see [`SimOptions::max_idle_steps`]).
    fn progress(&mut self) {
        self.epoch += 1;
        self.idle = 0;
    }

    /// Warp `k` reached a yield point.
    fn yield_warp(&mut self, k: usize) {
        let state = &mut self.warps[k];
        state.yielded = true;
        state.spin_epoch = self.epoch;
    }

    /// Whether a runnable warp spins: it has yielded and polled memory
    /// since anything visible last happened.
    fn spins(&self, s: &WarpState) -> bool {
        s.spin_epoch == self.epoch && s.poll_epoch == self.epoch
    }

    /// Report a hang once nothing visible has happened for
    /// [`SimOptions::max_idle_steps`] steps and every warp that has not
    /// exited is blocked or spinning.
    fn check_idle(&mut self) -> R {
        if self.idle <= self.opts.max_idle_steps {
            return Ok(());
        }
        let hung = self.warps.iter().chain(self.parked.iter().flatten()).all(|s| match s.status {
            Status::Runnable => self.spins(s),
            Status::Blocked(_) | Status::Exited => true,
        });
        if hung {
            return Err(self.hang());
        }
        // Some warp is still busy: look again later.
        self.idle = self.opts.max_idle_steps / 2;
        Ok(())
    }

    /// The next warps to step, from warp `*cursor` on, by the schedule
    /// ([`Schedule::RoundRobin`](super::Schedule::RoundRobin): the next
    /// runnable warp, with the runnable warps right after it that stand at
    /// the same instruction), and moves the cursor past them.
    fn next_group(&mut self, cursor: &mut usize) -> R<Option<Vec<usize>>> {
        let mut group: Vec<usize> = Vec::new();
        let mut at: Option<&'a Inst> = None;
        let mut k = *cursor;
        while k < self.warps.len() {
            self.take_turn(k);
            if let Some(inst) = self.current_inst(k)? {
                match at {
                    None => at = Some(inst),
                    Some(i) if std::ptr::eq(i, inst) => {}
                    Some(_) => break,
                }
                group.push(k);
            }
            k += 1;
        }
        *cursor = group.last().map_or(k, |&last| last + 1);
        Ok((!group.is_empty()).then_some(group))
    }

    /// The instruction runnable warp `k` executes next (after handling the
    /// region ends before it), or `None` if it is blocked or has exited.
    /// If the group it runs cannot run, another of its groups that can
    /// takes over.
    fn current_inst(&mut self, k: usize) -> R<Option<&'a Inst>> {
        loop {
            if let Some(inst) = self.front_inst(k)? {
                return Ok(Some(inst));
            }
            let round = self.round;
            match self.parked[k].iter().position(|g| g.ready(round)) {
                Some(i) => self.swap_in(k, i + 1),
                None => return Ok(None),
            }
        }
    }

    /// The instruction the group warp `k` runs executes next, if it can run
    /// (after handling the region ends before it; a group that ends there
    /// hands over to the one waiting for it).
    fn front_inst(&mut self, k: usize) -> R<Option<&'a Inst>> {
        loop {
            if !self.warps[k].ready(self.round) {
                return Ok(None);
            }
            if self.warps[k].settled {
                break;
            }
            self.settle(k)?;
        }
        let frame = self.warps[k].groups[0].frames.last().expect("a running warp has a frame");
        let region: &'a Region = frame.region;
        Ok(Some(&region.insts[frame.next]))
    }

    /// The lanes of warp `k` that run its next instruction.
    fn warp_lanes(&self, k: usize) -> &[usize] {
        &self.warps[k].groups[0].frames.last().expect("a running warp has a frame").lanes
    }

    /// [`Self::inst`] for the lanes of the warps of `group`, in order.
    fn run_inst(&mut self, inst: &'a Inst, group: &[usize]) -> R {
        if let [k] = *group {
            // Lend the warp's lanes to the instruction.
            let frame = self.warps[k].groups[0].frames.last_mut().expect("a running warp has a frame");
            let lanes = std::mem::take(&mut frame.lanes);
            let result = self.inst(inst, &lanes);
            if self.wgmma.enabled && result.is_ok() {
                self.wgmma_note_access(&inst.op, &lanes);
            }
            self.warps[k].groups[0].frames.last_mut().expect("a running warp has a frame").lanes = lanes;
            return result;
        }
        let mut active = std::mem::take(&mut self.active);
        active.clear();
        for &k in group {
            active.extend_from_slice(self.warp_lanes(k));
        }
        let result = self.inst(inst, &active);
        if self.wgmma.enabled && result.is_ok() {
            self.wgmma_note_access(&inst.op, &active);
        }
        self.active = active;
        result
    }

    /// Move warp `k` past the instruction it executed.
    fn advance(&mut self, k: usize) {
        let state = &mut self.warps[k];
        state.groups[0].frames.last_mut().expect("a running warp has a frame").next += 1;
        state.settled = false;
    }

    /// Execute one instruction for the warps of `group`, which all stand
    /// at it.
    fn step(&mut self, group: &[usize]) -> R {
        let frame = self.warps[group[0]].groups[0].frames.last().expect("a running warp has a frame");
        let region: &'a Region = frame.region;
        let inst: &'a Inst = &region.insts[frame.next];
        self.idle += group.len() as u64;
        self.steps += 1;
        if self.async_rng.is_some() && self.scheduling_point(&inst.op) {
            self.complete_at_random();
        }
        let count = |m: &mut Self| {
            m.loc = Some(inst.loc);
            m.report.warp_instructions += group.len() as u64;
            m.clock += 1;
        };
        match &inst.op {
            Op::If { .. } | Op::For { .. } | Op::While { .. } | Op::Call { body: Some(_), .. } => {
                count(self);
                for &k in group {
                    self.enter(k, inst)?;
                }
            }
            Op::Barrier => {
                count(self);
                self.barrier(inst, group, BarrierKind::Sync, None, None, None)?;
            }
            Op::NamedBarrier { kind, id, threads } => {
                count(self);
                self.barrier(inst, group, *kind, Some(*id), *threads, None)?;
            }
            // m11-prims-nvvm-layer.
            Op::BarrierRed { result, kind, pred, id, threads } => {
                count(self);
                self.barrier(inst, group, BarrierKind::Sync, Some(*id), *threads, Some((*kind, *pred, *result)))?;
            }
            Op::Exit => {
                count(self);
                self.exit_warps(inst, group)?;
            }
            Op::SetMaxNReg { increase, count: n } => {
                count(self);
                self.setmaxnreg(inst, group, *increase, *n)?;
            }
            Op::Wgmma { .. } => {
                count(self);
                self.wgmma_issue(inst, group)?;
            }
            Op::Tcgen05 { inst: which, .. } if which.is_warp_collective() => {
                count(self);
                self.tcgen05_issue(inst, group)?;
            }
            Op::ClusterArrive { relaxed, aligned } => {
                count(self);
                self.cluster_arrive(group, *relaxed, *aligned)?;
                for &k in group {
                    self.advance(k);
                }
            }
            Op::ClusterWait { aligned } => {
                count(self);
                let proceed = self.cluster_wait(inst, group, *aligned)?;
                for k in proceed {
                    self.advance(k);
                }
            }
            Op::MbarrierWait { kind: MbarrierWaitKind::Try, .. } => self.try_wait(inst, group)?,
            Op::Shuffle { mask, .. }
            | Op::Vote { mask, .. }
            | Op::Match { mask, .. }
            | Op::Redux { mask, .. }
            | Op::WarpBarrier { mask }
                if group.iter().any(|&k| !self.parked[k].is_empty()) =>
            {
                for &k in group {
                    self.rendezvous(inst, k, *mask)?;
                }
            }
            Op::InlinePtx { template, predicated: false, .. } if template.contains("barrier.cluster.") => {
                self.inline_cluster_barrier(inst, template, group)?
            }
            _ => {
                self.run_inst(inst, group)?;
                for &k in group {
                    self.advance(k);
                }
            }
        }
        Ok(())
    }

    /// Warp `k` executes structured instruction `inst` (`If`, `For` or
    /// `While`): it enters the instruction's first region.
    fn enter(&mut self, k: usize, inst: &'a Inst) -> R {
        self.advance(k);
        let lanes = self.warp_lanes(k).to_vec();
        let frame = match &inst.op {
            Op::If { cond, then_region, else_region, .. } => {
                let (taken, not_taken): (Vec<usize>, Vec<usize>) =
                    lanes.iter().partition(|&&t| self.get(*cond, t) != 0);
                if taken.is_empty() || not_taken.is_empty() {
                    Frame {
                        region: then_region,
                        next: 0,
                        lanes: taken,
                        kind: FrameKind::IfThen { inst, else_lanes: not_taken, split: None },
                    }
                } else {
                    // Divergent: the else lanes become a convergence group
                    // of their own (independent thread scheduling), which
                    // the then group waits for at the end of the If. The
                    // then group runs first, as a SIMT stack would.
                    self.next_join += 1;
                    let id = self.next_join;
                    self.reset_carry(&not_taken);
                    let kind = FrameKind::IfElse { inst, then_lanes: Vec::new(), join: Some(id) };
                    let mut group =
                        WarpState::new(vec![Frame { region: else_region, next: 0, lanes: not_taken, kind }]);
                    group.ready_round = self.warps[k].ready_round;
                    self.parked[k].push(group);
                    Frame {
                        region: then_region,
                        next: 0,
                        lanes: taken,
                        kind: FrameKind::IfThen { inst, else_lanes: Vec::new(), split: Some(id) },
                    }
                }
            }
            // The unroll and vectorize hints do not change what a loop
            // computes; `at_least_once` is a promise the program makes.
            Op::For { lower, upper, step, init, args, body, opts, .. } => {
                let as_int = |m: &Self, v: ValueId, t: usize| m.get(v, t) as u32 as i32 as i64;
                let mut ivs = Box::new([0i64; 32]);
                for &t in &lanes {
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
                        let bits = self.get128(*x, t);
                        self.set128(*a, t, bits);
                        self.carry_pending(*x, *a, t);
                    }
                    ivs[self.lane_of(t)] = lo;
                }
                let running: Vec<usize> =
                    lanes.iter().copied().filter(|&t| ivs[self.lane_of(t)] < as_int(self, *upper, t)).collect();
                self.start_iteration(inst, &running, &ivs);
                Frame { region: body, next: 0, lanes: running, kind: FrameKind::ForBody { inst, all: lanes, ivs } }
            }
            // A call simulated by its traced body: the body runs in the calling lanes.
            Op::Call { body: Some(body), .. } => {
                Frame { region: body, next: 0, lanes: lanes.clone(), kind: FrameKind::Call { inst } }
            }
            Op::While { init, cond_args, cond, .. } => {
                for &t in &lanes {
                    for (a, x) in cond_args.iter().zip(init) {
                        let bits = self.get128(*x, t);
                        self.set128(*a, t, bits);
                        self.carry_pending(*x, *a, t);
                    }
                }
                self.check_iterations(1, &lanes)?;
                Frame {
                    region: cond,
                    next: 0,
                    lanes: lanes.clone(),
                    kind: FrameKind::WhileCond { inst, all: lanes, iterations: 1 },
                }
            }
            _ => unreachable!("not a structured instruction"),
        };
        self.reset_carry(&frame.lanes);
        self.warps[k].groups[0].frames.push(frame);
        Ok(())
    }

    /// Set the induction variable of the `running` lanes of `For`
    /// instruction `inst` for their next iteration.
    fn start_iteration(&mut self, inst: &Inst, running: &[usize], ivs: &[i64; 32]) {
        let Op::For { iv, .. } = &inst.op else { unreachable!("a For") };
        for &t in running {
            self.set(*iv, t, ivs[self.lane_of(t)] as i32 as u32 as u64);
        }
    }

    /// Report a `while_carry` loop that has run more rounds than allowed.
    fn check_iterations(&self, iterations: u64, running: &[usize]) -> R {
        if iterations > self.opts.max_while_iterations {
            return Err(self.err("while_carry did not terminate", running.first().copied()));
        }
        Ok(())
    }

    /// A region is a new block of straight-line code, and the end of a
    /// structured instruction ends one: no carry flag.
    fn reset_carry(&mut self, lanes: &[usize]) {
        for &t in lanes {
            self.carry[t] = None;
        }
    }

    /// Handle the region ends that warp `k` stands at, until it stands at
    /// an instruction or has exited.
    fn settle(&mut self, k: usize) -> R {
        loop {
            let frames = &mut self.warps[k].groups[0].frames;
            if !frames.last().expect("a running warp has a frame").finished() {
                break;
            }
            let mut frame = frames.pop().expect("a running warp has a frame");
            let continued = match &mut frame.kind {
                FrameKind::Body => {
                    self.warps[k].status = Status::Exited;
                    self.warps[k].settled = true;
                    self.progress();
                    let block = k / self.warps_per_block;
                    self.complete_cluster_barrier(block / self.cluster_size);
                    self.land_bulk_stores_at_exit(block);
                    self.check_block_exit(block)?;
                    return self.check_exit(block);
                }
                FrameKind::IfThen { inst, split: Some(id), .. } => {
                    let (inst, id): (&'a Inst, u64) = (inst, *id);
                    if self.parked[k].iter().any(|g| g.join() == Some(id)) {
                        // Wait for the else group at the join.
                        self.warps[k].groups[0].frames.push(frame);
                        self.warps[k].status = Status::Blocked(Wait::Join { id, inst });
                        return Ok(());
                    }
                    let Op::If { then_region, results, .. } = &inst.op else { unreachable!("an If") };
                    for &t in &frame.lanes {
                        for (r, y) in results.iter().zip(&then_region.yields) {
                            let bits = self.get(*y, t);
                            self.set(*r, t, bits);
                            self.carry_pending(*y, *r, t);
                        }
                    }
                    self.reset_carry(&frame.lanes);
                    None
                }
                FrameKind::IfThen { inst, else_lanes, split: None } => {
                    let inst: &'a Inst = inst;
                    let Op::If { else_region: region, .. } = &inst.op else { unreachable!("an If") };
                    let lanes = std::mem::take(else_lanes);
                    self.reset_carry(&lanes);
                    let then_lanes = std::mem::take(&mut frame.lanes);
                    Some(Frame { region, next: 0, lanes, kind: FrameKind::IfElse { inst, then_lanes, join: None } })
                }
                FrameKind::IfElse { inst, then_lanes, join } => {
                    let Op::If { then_region, else_region, results, .. } = &inst.op else { unreachable!("an If") };
                    for (lanes, region) in [(&*then_lanes, then_region), (&frame.lanes, else_region)] {
                        for &t in lanes {
                            for (r, y) in results.iter().zip(&region.yields) {
                                let bits = self.get128(*y, t);
                                self.set128(*r, t, bits);
                                self.carry_pending(*y, *r, t);
                            }
                        }
                    }
                    self.reset_carry(then_lanes);
                    self.reset_carry(&frame.lanes);
                    if let Some(id) = *join {
                        self.end_group(k, id);
                        return Ok(());
                    }
                    None
                }
                FrameKind::Call { inst } => {
                    let Op::Call { body: Some(body), results, .. } = &inst.op else { unreachable!("a Call") };
                    for &t in &frame.lanes {
                        for (r, y) in results.iter().zip(&body.yields) {
                            let bits = self.get128(*y, t);
                            self.set128(*r, t, bits);
                            self.carry_pending(*y, *r, t);
                        }
                    }
                    self.reset_carry(&frame.lanes);
                    None
                }
                FrameKind::ForBody { inst, all, ivs } => {
                    let inst: &'a Inst = inst;
                    let Op::For { upper, step, args, body, results, .. } = &inst.op else { unreachable!("a For") };
                    self.loc = Some(inst.loc);
                    let as_int = |m: &Self, v: ValueId, t: usize| m.get(v, t) as u32 as i32 as i64;
                    let mut running = std::mem::take(&mut frame.lanes);
                    for &t in &running {
                        let next: Vec<u128> = body.yields.iter().map(|y| self.get128(*y, t)).collect();
                        for (a, bits) in args.iter().zip(next) {
                            self.set128(*a, t, bits);
                        }
                        for (a, y) in args.iter().zip(&body.yields) {
                            self.carry_pending(*y, *a, t);
                        }
                        ivs[self.lane_of(t)] += as_int(self, *step, t);
                    }
                    running.retain(|&t| ivs[self.lane_of(t)] < as_int(self, *upper, t));
                    if running.is_empty() {
                        for &t in all.iter() {
                            for (r, a) in results.iter().zip(args) {
                                let bits = self.get128(*a, t);
                                self.set128(*r, t, bits);
                                self.carry_pending(*a, *r, t);
                            }
                        }
                        self.reset_carry(all);
                        None
                    } else {
                        self.start_iteration(inst, &running, ivs);
                        self.reset_carry(&running);
                        frame.lanes = running;
                        frame.next = 0;
                        Some(frame)
                    }
                }
                FrameKind::WhileCond { inst, all, iterations } => {
                    let inst: &'a Inst = inst;
                    let Op::While { cond_args, cond, body_args, body, results, .. } = &inst.op else {
                        unreachable!("a While")
                    };
                    self.loc = Some(inst.loc);
                    let c = cond.yields[0];
                    let (go, stop): (Vec<usize>, Vec<usize>) = frame.lanes.iter().partition(|&&t| self.get(c, t) != 0);
                    for &t in &stop {
                        for (r, a) in results.iter().zip(cond_args) {
                            let bits = self.get128(*a, t);
                            self.set128(*r, t, bits);
                            self.carry_pending(*a, *r, t);
                        }
                    }
                    for &t in &go {
                        for (b, a) in body_args.iter().zip(cond_args) {
                            let bits = self.get128(*a, t);
                            self.set128(*b, t, bits);
                            self.carry_pending(*a, *b, t);
                        }
                    }
                    if go.is_empty() {
                        self.reset_carry(all);
                        None
                    } else {
                        self.reset_carry(&go);
                        let (all, iterations) = (std::mem::take(all), *iterations);
                        Some(Frame {
                            region: body,
                            next: 0,
                            lanes: go,
                            kind: FrameKind::WhileBody { inst, all, iterations },
                        })
                    }
                }
                FrameKind::WhileBody { inst, all, iterations } => {
                    let inst: &'a Inst = inst;
                    let Op::While { cond_args, cond, body, .. } = &inst.op else { unreachable!("a While") };
                    self.loc = Some(inst.loc);
                    for &t in &frame.lanes {
                        let next: Vec<u128> = body.yields.iter().map(|y| self.get128(*y, t)).collect();
                        for (a, bits) in cond_args.iter().zip(next) {
                            self.set128(*a, t, bits);
                        }
                        for (a, y) in cond_args.iter().zip(&body.yields) {
                            self.carry_pending(*y, *a, t);
                        }
                    }
                    let iterations = *iterations + 1;
                    self.check_iterations(iterations, &frame.lanes)?;
                    // A back-edge: spin loops let the other warps run.
                    self.yield_warp(k);
                    self.reset_carry(&frame.lanes);
                    let all = std::mem::take(all);
                    let lanes = std::mem::take(&mut frame.lanes);
                    Some(Frame { region: cond, next: 0, lanes, kind: FrameKind::WhileCond { inst, all, iterations } })
                }
            };
            if let Some(frame) = continued {
                self.warps[k].groups[0].frames.push(frame);
            }
        }
        self.warps[k].settled = true;
        Ok(())
    }

    /// The else group that warp `k` runs, of join record `id`, has finished:
    /// it ends, and the then group waiting for it at the join (or, if that
    /// one has not got there yet, another group) takes over.
    fn end_group(&mut self, k: usize, id: u64) {
        let waiting = self.parked[k]
            .iter()
            .position(|g| matches!(g.status, Status::Blocked(Wait::Join { id: j, .. }) if j == id));
        let i = match waiting {
            Some(i) => {
                let g = &mut self.parked[k][i];
                g.status = Status::Runnable;
                g.settled = false;
                i
            }
            None => self.parked[k].iter().position(|g| g.ready(self.round)).unwrap_or(0),
        };
        self.warps[k] = self.parked[k].remove(i);
    }

    /// Group `k`'s turn at `.sync` warp collective `inst` (member mask
    /// `mask`), with independent thread scheduling: the lanes of the mask
    /// may stand in several convergence groups of the warp. The group
    /// blocks until every lane of the mask that has not exited has reached
    /// the instruction, then it executes once for all of them. Lanes of the
    /// mask that no group holds (they wait at a loop's end or a join) can
    /// never come: executing reports them.
    fn rendezvous(&mut self, inst: &'a Inst, k: usize, mask: ValueId) -> R {
        let lanes = self.warp_lanes(k).to_vec();
        let base = lanes[0] - self.lane_of(lanes[0]);
        let bits = |m: &Self, lanes: &[usize]| lanes.iter().fold(0u32, |acc, &t| acc | 1 << m.lane_of(t));
        let live = if self.warp_size(base) == 32 { u32::MAX } else { (1u32 << self.warp_size(base)) - 1 };
        let needed = lanes.iter().fold(0u32, |acc, &t| acc | self.get(mask, t) as u32) & live;
        let mine = bits(self, &lanes);
        if needed & !mine == 0 {
            self.run_inst(inst, &[k])?;
            self.advance(k);
            return Ok(());
        }
        let held = self.parked[k].iter().fold(0u32, |acc, g| acc | bits(self, g.lanes()));
        if needed & !mine & !held != 0 {
            self.run_inst(inst, &[k])?;
            self.advance(k);
            return Ok(());
        }
        let partners: Vec<usize> = (0..self.parked[k].len())
            .filter(|&i| {
                let g = &self.parked[k][i];
                matches!(g.status, Status::Blocked(Wait::Rendezvous { inst: at }) if same_collective(at, inst))
                    && bits(self, g.lanes()) & !needed == 0
            })
            .collect();
        let present = partners.iter().fold(mine, |acc, &i| acc | bits(self, self.parked[k][i].lanes()));
        if needed & !present != 0 {
            self.warps[k].status = Status::Blocked(Wait::Rendezvous { inst });
            return Ok(());
        }
        // Lanes at another instruction of the same kind (sm_70 lets lanes
        // meet at different ones) run this one, with their own operands,
        // and take their results back.
        let mut all = lanes;
        let mut moved: Vec<(&'a Inst, Vec<usize>)> = Vec::new();
        for &i in &partners {
            let g = &self.parked[k][i];
            all.extend_from_slice(g.lanes());
            let Status::Blocked(Wait::Rendezvous { inst: at }) = g.status else { unreachable!("a partner") };
            if !std::ptr::eq(at, inst) {
                moved.push((at, g.lanes().to_vec()));
            }
        }
        // The partners' registers of this instruction's operands may hold
        // values they use later (an operand defined before the `if`): they
        // are put back afterwards.
        let mut saved: Vec<(ValueId, usize, u64)> = Vec::new();
        for (at, ts) in &moved {
            for (mine, theirs) in inst.op.operands().into_iter().zip(at.op.operands()) {
                for &t in ts {
                    saved.push((mine, t, self.get(mine, t)));
                }
                for &t in ts {
                    let bits = self.get(theirs, t);
                    self.set(mine, t, bits);
                }
            }
        }
        all.sort_unstable();
        self.inst(inst, &all)?;
        for &(v, t, bits) in saved.iter().rev() {
            self.set(v, t, bits);
        }
        for (at, ts) in &moved {
            if let (Some(mine), Some(theirs)) = (collective_result(inst), collective_result(at)) {
                for &t in ts {
                    let bits = self.get(mine, t);
                    self.set(theirs, t, bits);
                }
            }
        }
        self.advance(k);
        for i in partners {
            let g = &mut self.parked[k][i];
            g.status = Status::Runnable;
            g.groups[0].frames.last_mut().expect("a blocked group has a frame").next += 1;
            g.settled = false;
        }
        Ok(())
    }

    /// The warps of `block` that have exited.
    fn exited_warps(&self, block: usize) -> Vec<usize> {
        let w = self.warps_per_block;
        (0..w).filter(|&warp| matches!(self.warps[block * w + warp].status, Status::Exited)).collect()
    }

    /// Threads in the warps `warps` of a block.
    fn threads_in(&self, warps: &[usize]) -> usize {
        warps.iter().map(|&warp| self.warp_size(warp * 32)).sum()
    }

    /// After a warp of `block` exited: if it was the block's last, nothing
    /// asynchronous may still write its shared memory, which is gone (a
    /// bulk copy or `cp.async` in flight, or an arrival that
    /// `cp.async.mbarrier.arrive` owes one of its mbarriers).
    fn check_block_exit(&self, block: usize) -> R {
        let w = self.warps_per_block;
        if !(0..w).all(|warp| matches!(self.warps[block * w + warp].status, Status::Exited)) {
            return Ok(());
        }
        self.tmem_check_exit(block)?;
        let cta = format!("CTA {} ({})", block % self.cluster_size, self.block_name(block));
        // Accesses of other CTAs to its shared memory must be
        // ordered before its exit.
        if let Some(log) = self.dsmem_log.get(&block) {
            let mut entries: Vec<(&usize, &(u32, crate::ir::Loc))> = log.iter().collect();
            entries.sort_by_key(|e| *e.0);
            for (&slot, &(clock, loc)) in entries {
                let known = (0..w).map(|warp| self.clocks[block * w + warp][slot]).max().unwrap_or(0);
                if known < clock {
                    return Err(self.err(
                        format!(
                            "{cta} exits while {} may still access its shared memory (the access at {}:{} is not \
                             ordered before the exit): CUDA requires a cluster barrier (cluster_arrive, cluster_wait) \
                             before a CTA whose shared memory other CTAs use exits",
                            self.slot_name(slot, block),
                            loc.file(),
                            loc.line()
                        ),
                        None,
                    ));
                }
            }
        }
        let mut keys: Vec<&MbarrierKey> = self.mbarriers.keys().filter(|key| key.0 == block).collect();
        keys.sort();
        for key in keys {
            let m = &self.mbarriers[key];
            let name = smem_place(self.ir.buffer(BufferId(key.1)), key.2);
            if !m.copies.is_empty() {
                let n = m.copies.len();
                return Err(self.err(
                    format!(
                        "{cta} exits while {n} bulk cop{} reporting to its mbarrier {name} {} still in flight: they \
                         write shared memory that no longer exists (wait for the mbarrier phase before exiting)",
                        if n == 1 { "y" } else { "ies" },
                        if n == 1 { "is" } else { "are" }
                    ),
                    None,
                ));
            }
            let owed = m.deferred.iter().filter(|d| d.copies.iter().any(|&id| self.ops[id].in_flight())).count();
            if owed > 0 {
                return Err(self.err(
                    format!(
                        "{cta} exits while its mbarrier {name} still waits for {owed} cp.async.mbarrier.arrive \
                         arrival(s) whose copies are in flight (wait for the mbarrier phase before exiting)"
                    ),
                    None,
                ));
            }
        }
        for op in self.ops.iter().filter(|op| op.in_flight()) {
            if let Some(w) = op.writes.iter().find(|w| w.block == block) {
                let place = smem_place(self.ir.buffer(w.dst), w.dst_byte);
                return Err(self.err(
                    format!(
                        "{cta} exits while an asynchronous copy is still writing its shared memory ({place}): wait \
                         for the copy (cp_async_wait_group, or its mbarrier) before exiting"
                    ),
                    None,
                ));
            }
            if let Some(r) = op.reads.iter().find(|r| r.block == block) {
                let place = smem_place(self.ir.buffer(r.buffer), r.bytes.start);
                return Err(self.err(
                    format!(
                        "{cta} exits while an asynchronous operation still reads its shared memory ({place}): wait \
                         for the operation before exiting"
                    ),
                    None,
                ));
            }
        }
        Ok(())
    }

    /// After a warp of `block` exited: an all-thread barrier of the block
    /// that every other warp waits at would complete only because warps
    /// exited instead of reaching it, which is reported.
    fn check_exit(&self, block: usize) -> R {
        for state in &self.named[block] {
            if state.all_threads && !state.waiting.is_empty() {
                let exited = self.exited_warps(block);
                if state.arrived.len() + exited.len() == state.expected {
                    let site = state.site.expect("an all-thread barrier with waiters has a site");
                    return Err(self.exit_error(site, block, &state.arrived, &exited));
                }
            }
        }
        Ok(())
    }

    /// The report of all-thread barrier `site` of `block`, reached by warps
    /// `arrived`, that would complete only because warps `exited` have
    /// exited.
    fn exit_error(&self, site: &Inst, block: usize, arrived: &[usize], exited: &[usize]) -> SimError {
        let what = if matches!(site.op, Op::Barrier) { "sync_threads()" } else { "barrier" };
        SimError {
            loc: Some(site.loc),
            block: self.blocks[block],
            ..self.err(
                format!(
                    "{what} reached by {} of {} threads: warps {} exited without reaching it (barriers must not be \
                     inside divergent control flow)",
                    self.threads_in(arrived),
                    self.threads_per_block,
                    warp_list(exited)
                ),
                None,
            )
        }
    }

    /// The `mbarrier.try_wait` that `group` stands at: warps for which a
    /// lane's phase is not over block until it is (or until the wait times
    /// out, when nothing else can run); the others execute it.
    fn try_wait(&mut self, inst: &'a Inst, group: &[usize]) -> R {
        let mut proceed: Vec<usize> = Vec::new();
        self.loc = Some(inst.loc);
        for &k in group {
            let mut blocker = if self.warps[k].timed_out { None } else { self.wait_blocker(inst, k)? };
            // A seeded schedule sometimes lets the wait time out at once.
            if blocker.is_some()
                && let Some(rng) = self.sched_rng.as_mut()
                && splitmix(rng).is_multiple_of(8)
            {
                blocker = None;
            }
            match blocker {
                Some((key, phase)) => self.warps[k].status = Status::Blocked(Wait::Mbarrier { key, phase, inst }),
                None => proceed.push(k),
            }
        }
        if proceed.is_empty() {
            return Ok(());
        }
        self.run_inst(inst, &proceed)?;
        for k in proceed {
            self.warps[k].timed_out = false;
            self.advance(k);
        }
        Ok(())
    }

    /// Inline PTX holding `barrier.cluster.arrive` and/or
    /// `barrier.cluster.wait` (`inst`), for the warps of `group`: the
    /// cluster barrier's arrive and wait ([`Self::cluster_arrive`],
    /// [`Self::cluster_wait`]), `.relaxed` and `.aligned` as the template
    /// says. Warps that get past it run the instruction's simulation. Woken
    /// at a combined arrive-and-wait, a warp does not arrive again.
    fn inline_cluster_barrier(&mut self, inst: &'a Inst, template: &str, group: &[usize]) -> R {
        self.loc = Some(inst.loc);
        let arrives = template.contains("barrier.cluster.arrive");
        let waits = template.contains("barrier.cluster.wait");
        let relaxed = template.contains("barrier.cluster.arrive.relaxed");
        let aligned = template.contains(".aligned");
        if arrives {
            let arriving: Vec<usize> = group
                .iter()
                .copied()
                .filter(|&k| !waits || self.warp_lanes(k).iter().all(|&t| self.cluster_arrival[t].is_none()))
                .collect();
            if !arriving.is_empty() {
                self.cluster_arrive(&arriving, relaxed, aligned)?;
            }
        }
        let proceed = if waits { self.cluster_wait(inst, group, aligned)? } else { group.to_vec() };
        if proceed.is_empty() {
            return Ok(());
        }
        self.run_inst(inst, &proceed)?;
        for k in proceed {
            self.advance(k);
        }
        Ok(())
    }

    /// Report a cluster barrier instruction (`what`) that only some lanes of
    /// warp `k` execute, if it is `.aligned`.
    fn check_aligned(&self, k: usize, aligned: bool, what: &str) -> R {
        let lanes = self.warp_lanes(k);
        let base = lanes[0] - self.lane_of(lanes[0]);
        if aligned && lanes.len() != self.warp_size(base) {
            return Err(self.err(
                format!(
                    "{what} executed by {} of the {} lanes of warp {}: its .aligned form needs every thread of the \
                     warp to execute the same instruction (the non-aligned form lets threads arrive one by one)",
                    lanes.len(),
                    self.warp_size(base),
                    k % self.warps_per_block
                ),
                Some(lanes[0]),
            ));
        }
        Ok(())
    }

    /// `barrier.cluster.arrive{.relaxed}{.aligned}` by the lanes of the
    /// warps of `group`: each thread arrives in the current phase of its
    /// cluster's barrier (once: arriving again before the phase completes is
    /// reported); a release arrival publishes its warp's clock at cluster
    /// scope, and the warp moves on to a new time of its own.
    fn cluster_arrive(&mut self, group: &[usize], relaxed: bool, aligned: bool) -> R {
        let what = if relaxed { "cluster_arrive_relaxed" } else { "cluster_arrive" };
        let mut clusters: Vec<usize> = Vec::new();
        for &k in group {
            self.check_aligned(k, aligned, what)?;
            let (block, slot) = (k / self.warps_per_block, self.slot_of(k));
            let c = block / self.cluster_size;
            let phase = self.cluster_barriers[c].phase;
            for &t in self.warps[k].groups[0].frames.last().expect("a running warp has a frame").lanes.iter() {
                if let Some(earlier) = self.cluster_arrival[t] {
                    // Whether the earlier phase has completed depends on the
                    // schedule; arriving twice without a wait never is right.
                    return Err(self.err(
                        format!(
                            "{what}: the thread arrives at the cluster barrier again without having waited since its \
                             arrival in phase {earlier} (each thread arrives once per phase, then waits with \
                             cluster_wait before arriving again)"
                        ),
                        Some(t),
                    ));
                }
                self.cluster_arrival[t] = Some(phase);
            }
            let clock = std::mem::take(&mut self.clocks[k]);
            let barrier = &mut self.cluster_barriers[c];
            if !relaxed {
                join(&mut barrier.release, &clock);
            } else if let Some(fence) = &self.fence_release[k] {
                // A fence then a relaxed arrival is a release pattern (the
                // PTX ISA's example for barrier.cluster.arrive.relaxed): it
                // publishes the clock at the fence.
                join(&mut barrier.release, fence);
            }
            barrier.init_release[slot] = barrier.init_release[slot].max(clock[slot]);
            self.clocks[k] = clock;
            self.clocks[k][slot] += 1;
            if !clusters.contains(&c) {
                clusters.push(c);
            }
        }
        self.progress();
        for c in clusters {
            self.complete_cluster_barrier(c);
        }
        Ok(())
    }

    /// `barrier.cluster.wait{.aligned}` by the warps of `group`: a warp
    /// whose lanes' phase has completed acquires what its arrivals released
    /// and goes on (it is returned); one whose phase is still running
    /// blocks. A thread that has not arrived would wait for itself.
    fn cluster_wait(&mut self, inst: &'a Inst, group: &[usize], aligned: bool) -> R<Vec<usize>> {
        let mut proceed = Vec::new();
        for &k in group {
            self.check_aligned(k, aligned, "cluster_wait")?;
            let c = (k / self.warps_per_block) / self.cluster_size;
            let phase = self.cluster_barriers[c].phase;
            let lanes = self.warp_lanes(k).to_vec();
            if let Some(&t) = lanes.iter().find(|&&t| self.cluster_arrival[t].is_none()) {
                return Err(self.err(
                    "cluster_wait by a thread that has not arrived at the cluster barrier in this phase: it would \
                     wait for its own arrival, so the barrier never completes (cluster_arrive first)",
                    Some(t),
                ));
            }
            if lanes.iter().any(|&t| self.cluster_arrival[t] == Some(phase)) {
                self.warps[k].status = Status::Blocked(Wait::Cluster { inst });
                continue;
            }
            let barrier = &self.cluster_barriers[c];
            join(&mut self.clocks[k], &barrier.completed);
            join(&mut self.init_clocks[k], &barrier.init_completed);
            for t in lanes {
                self.cluster_arrival[t] = None;
            }
            proceed.push(k);
        }
        Ok(proceed)
    }

    /// The threads of cluster `c` that have not exited, by thread index.
    fn live_cluster_threads(&self, c: usize) -> impl Iterator<Item = usize> + '_ {
        let tpb = self.threads_per_block;
        let w = self.warps_per_block;
        let blocks = c * self.cluster_size..(c + 1) * self.cluster_size;
        blocks
            .flat_map(move |b| b * tpb..(b + 1) * tpb)
            .filter(move |&t| !matches!(self.warps[(t / tpb) * w + (t % tpb) / 32].status, Status::Exited))
    }

    /// Complete cluster `c`'s barrier phase if some thread has arrived in it
    /// and every thread of the cluster that has not exited has, waking the
    /// warps that wait for it.
    fn complete_cluster_barrier(&mut self, c: usize) {
        let phase = self.cluster_barriers[c].phase;
        let mut any = false;
        for t in self.live_cluster_threads(c) {
            if self.cluster_arrival[t] != Some(phase) {
                return;
            }
            any = true;
        }
        if !any {
            return;
        }
        let barrier = &mut self.cluster_barriers[c];
        barrier.phase += 1;
        barrier.completed = std::mem::replace(&mut barrier.release, vec![0; barrier.completed.len()]);
        barrier.init_completed = std::mem::replace(&mut barrier.init_release, vec![0; barrier.init_completed.len()]);
        self.report.barriers += 1;
        let w = self.warps_per_block;
        let round = self.round;
        let warps = c * self.cluster_size * w..(c + 1) * self.cluster_size * w;
        for state in self.warps[warps.clone()].iter_mut().chain(self.parked[warps].iter_mut().flatten()) {
            if let Status::Blocked(Wait::Cluster { .. }) = state.status {
                state.status = Status::Runnable;
                state.ready_round = round + 1;
            }
        }
    }

    /// The mbarrier phase that warp `k`'s lanes would have to wait for at
    /// `mbarrier.try_wait` `inst`, if any. Lanes whose wait the instruction
    /// reports as a mistake do not block: executing it reports them.
    fn wait_blocker(&mut self, inst: &Inst, k: usize) -> R<Option<(MbarrierKey, u64)>> {
        let Op::MbarrierWait { buffer, index, parity, state: by_state, .. } = &inst.op else {
            unreachable!("an mbarrier wait")
        };
        let w = self.warps_per_block;
        let lanes = self.warp_lanes(k).to_vec();
        let mut blocker = None;
        for t in lanes {
            let Ok(byte) = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "mbarrier.try_wait") else {
                return Ok(None);
            };
            let key = (self.block_rank(t), buffer.0, byte);
            self.fire_deferred_arrivals(key);
            self.land_bulk_copies(key, t)?;
            let Some(state) = self.mbarriers.get(&key) else { return Ok(None) };
            let raw = self.get(*parity, t);
            let over = if *by_state { state.phase > raw } else { raw & 1 != state.phase & 1 };
            let asked = if over { state.phase.saturating_sub(1) } else { state.phase };
            if !*by_state && state.known_phases(&self.clocks[self.block_rank(t) * w + self.warp_of(t) as usize]) < asked
            {
                return Ok(None);
            }
            if !over && blocker.is_none() {
                blocker = Some((key, state.phase));
            }
        }
        Ok(blocker)
    }

    /// Wake the warps blocked at an `mbarrier.try_wait` whose phase has
    /// completed, landing the bulk copies and making the `cp.async`
    /// arrivals that the phase still waits for first.
    fn wake_mbarrier_waiters(&mut self) -> R {
        for (k, g) in self.all_groups() {
            let Status::Blocked(Wait::Mbarrier { key, phase, inst }) = self.group(k, g).status else { continue };
            self.loc = Some(inst.loc);
            let t = self.group(k, g).lanes()[0];
            self.fire_deferred_arrivals(key);
            self.land_bulk_copies(key, t)?;
            if self.mbarriers.get(&key).is_none_or(|m| m.phase != phase) {
                self.group_mut(k, g).status = Status::Runnable;
            }
        }
        Ok(())
    }

    /// `file:line` of an instruction.
    fn at(inst: &Inst) -> String {
        format!("{}:{}", inst.loc.file(), inst.loc.line())
    }

    /// `block (x,y,z)` for block rank `block`.
    fn block_name(&self, block: usize) -> String {
        let [x, y, z] = self.blocks[block];
        format!("block ({x},{y},{z})")
    }

    /// The constructs warp `k` stands in, innermost first, from its frame
    /// stack: `, in for_range at k.rs:40, iteration 3; in the then-branch of
    /// if_ at k.rs:38` (empty at the top level of the kernel).
    fn context(&self, k: usize) -> String {
        self.group_context(k, &self.warps[k])
    }

    /// The frames group `group` of warp `k` stands in, outermost first: its
    /// own and, for an else group, those of the then group it split from
    /// below the divergent `If`.
    fn frames_of<'s>(&'s self, k: usize, group: &'s WarpState<'a>) -> Vec<&'s Frame<'a>> {
        let own = &group.groups[0].frames;
        let Some(id) = group.join() else { return own.iter().collect() };
        let split_at = |g: &'s WarpState<'a>| {
            g.groups[0]
                .frames
                .iter()
                .position(|f| matches!(f.kind, FrameKind::IfThen { split: Some(s), .. } if s == id))
        };
        for g in std::iter::once(&self.warps[k]).chain(&self.parked[k]) {
            if let Some(i) = split_at(g) {
                let mut frames = self.frames_of(k, g);
                // The then group's frames below the If, then its own.
                let below = frames.len() - (g.groups[0].frames.len() - i);
                frames.truncate(below);
                frames.extend(own.iter());
                return frames;
            }
        }
        own.iter().collect()
    }

    /// [`Self::context`] for group `group` of warp `k`.
    fn group_context(&self, k: usize, group: &WarpState<'a>) -> String {
        let mut parts: Vec<String> = Vec::new();
        for frame in self.frames_of(k, group).into_iter().rev() {
            let t = frame.lanes.first().copied();
            match &frame.kind {
                FrameKind::Body => {}
                FrameKind::IfThen { inst, .. } => {
                    parts.push(format!("in the then-branch of if_ at {}", Self::at(inst)))
                }
                FrameKind::IfElse { inst, .. } => {
                    parts.push(format!("in the else-branch of if_ at {}", Self::at(inst)))
                }
                FrameKind::ForBody { inst, ivs, .. } => {
                    let Op::For { lower, step, .. } = &inst.op else { unreachable!("a For") };
                    let iteration = t.map(|t| {
                        let lo = self.get(*lower, t) as u32 as i32 as i64;
                        let step = (self.get(*step, t) as u32 as i32 as i64).max(1);
                        (ivs[self.lane_of(t)] - lo) / step
                    });
                    parts.push(match iteration {
                        Some(i) => format!("in for_range at {}, iteration {i}", Self::at(inst)),
                        None => format!("in for_range at {}", Self::at(inst)),
                    });
                }
                FrameKind::WhileCond { inst, iterations, .. } | FrameKind::WhileBody { inst, iterations, .. } => {
                    parts.push(format!("in while_carry at {}, iteration {}", Self::at(inst), iterations - 1))
                }
                FrameKind::Call { inst } => {
                    let Op::Call { callee, .. } = &inst.op else { unreachable!("a Call") };
                    let symbol = &self.ir.extern_decl(*callee).symbol;
                    parts.push(format!("in the call of `{symbol}` at {}", Self::at(inst)))
                }
            }
        }
        if parts.is_empty() { String::new() } else { format!(", {}", parts.join("; ")) }
    }

    /// What blocked warp `k` waits at: `barrier 1 (k.rs:12)`.
    fn wait_site(&self, wait: &Wait) -> String {
        match wait {
            Wait::Barrier { inst, .. } if matches!(inst.op, Op::Barrier) => {
                format!("sync_threads() ({})", Self::at(inst))
            }
            Wait::Barrier { id, inst, .. } => format!("barrier {id} ({})", Self::at(inst)),
            Wait::Mbarrier { inst, .. } => format!("mbarrier.try_wait ({})", Self::at(inst)),
            Wait::Cluster { inst } => format!("the cluster barrier ({})", Self::at(inst)),
            Wait::Join { inst, .. } => format!("the end of if_ ({})", Self::at(inst)),
            Wait::SetMaxNReg { inst, .. } | Wait::Wgmma { inst, .. } | Wait::TmemAlloc { inst, .. } => {
                format!("{} ({})", inst.op.name(), Self::at(inst))
            }
            Wait::Rendezvous { inst } => {
                let name = match &inst.op {
                    Op::Shuffle { .. } => "shuffle",
                    Op::Vote { kind: VoteKind::All, .. } => "vote_all_sync",
                    Op::Vote { kind: VoteKind::Any, .. } => "vote_any_sync",
                    Op::Vote { kind: VoteKind::Uni, .. } => "vote_uni_sync",
                    Op::Vote { kind: VoteKind::Ballot, .. } => "vote_ballot_sync",
                    Op::Match { .. } => "match_sync",
                    Op::Redux { .. } => "warp_redux_sync",
                    _ => "sync_warp",
                };
                format!("{name} ({})", Self::at(inst))
            }
        }
    }

    /// ` (lanes 0..=15)` for a group that holds part of its warp (whose
    /// thread `t` is), else nothing.
    fn partial_lanes(&self, group: &WarpState, t: usize) -> String {
        let lanes = self.lane_set(group);
        if lanes.len() == self.warp_size(t - self.lane_of(t)) {
            String::new()
        } else {
            format!(" ({} {})", if lanes.len() == 1 { "lane" } else { "lanes" }, warp_list(&lanes))
        }
    }

    /// The lanes (0 to 31) of group `group`.
    fn lane_set(&self, group: &WarpState) -> Vec<usize> {
        group.lanes().iter().map(|&t| self.lane_of(t)).collect()
    }

    /// Where `lanes` of warp `k` are: `lanes 1..=31 wait at mbarrier.try_wait
    /// (k.rs:30), in the else-branch of if_ at k.rs:20`, by the groups that
    /// hold them; lanes no group holds wait to reconverge.
    fn lanes_whereabouts(&self, k: usize, lanes: &[usize]) -> String {
        let mut parts = Vec::new();
        let mut left: Vec<usize> = lanes.to_vec();
        for g in std::iter::once(&self.warps[k]).chain(&self.parked[k]) {
            let mine: Vec<usize> = self.lane_set(g).into_iter().filter(|l| lanes.contains(l)).collect();
            if mine.is_empty() {
                continue;
            }
            left.retain(|l| !mine.contains(l));
            let what = match &g.status {
                Status::Blocked(wait) => format!("wait at {}{}", self.wait_site(wait), self.group_context(k, g)),
                Status::Runnable if self.spins(g) => format!("spin{}", self.group_context(k, g)),
                Status::Runnable => format!("are still running{}", self.group_context(k, g)),
                Status::Exited => "have exited".into(),
            };
            let what =
                if mine.len() == 1 { what.replacen("wait ", "waits ", 1).replacen("are ", "is ", 1) } else { what };
            let what =
                if mine.len() == 1 && what.starts_with("spin") { what.replacen("spin", "spins", 1) } else { what };
            parts.push(format!("{} {} {what}", if mine.len() == 1 { "lane" } else { "lanes" }, warp_list(&mine)));
        }
        if !left.is_empty() {
            parts.push(format!(
                "{} {} took another path and {} at the end of a branch or loop for the others to reconverge, so they \
                 never get there",
                if left.len() == 1 { "lane" } else { "lanes" },
                warp_list(&left),
                if left.len() == 1 { "waits" } else { "wait" }
            ));
        }
        parts.join("; ")
    }

    /// Where warp `k` is, as the singular and plural predicate of a
    /// sentence about it (`has exited` / `have exited`).
    fn whereabouts(&self, k: usize) -> (String, String) {
        if !self.parked[k].is_empty() {
            let t = self.warps[k].lanes()[0];
            let all: Vec<usize> = (0..self.warp_size(t - self.lane_of(t))).collect();
            let site = format!("is divergent ({})", self.lanes_whereabouts(k, &all));
            return (site.clone(), site.replacen("is", "are", 1));
        }
        match &self.warps[k].status {
            Status::Exited => match self.exit_sites.get(&k) {
                Some(at) => {
                    let site = format!("at the exit at {}:{}", at.file(), at.line());
                    (format!("has exited {site}"), format!("have exited {site}"))
                }
                None => ("has exited".into(), "have exited".into()),
            },
            Status::Blocked(wait) => {
                let site = format!("at {}{}", self.wait_site(wait), self.context(k));
                (format!("waits {site}"), format!("wait {site}"))
            }
            Status::Runnable if self.spins(&self.warps[k]) => ("spins".into(), "spin".into()),
            Status::Runnable => ("is still running".into(), "are still running".into()),
        }
    }

    /// `warp 2 waits at ...; warps 0..=3 have exited`: where each of
    /// `warps` (by key) is, warps in the same place together. With
    /// `blocks`, each is named with its block.
    fn whereabouts_of(&self, warps: &[usize], blocks: bool) -> String {
        let w = self.warps_per_block;
        let mut groups: Vec<((String, String, usize), Vec<usize>)> = Vec::new();
        for &k in warps {
            let (one, many) = self.whereabouts(k);
            let block = if blocks { k / w } else { 0 };
            match groups.iter_mut().find(|g| g.0.0 == one && g.0.2 == block) {
                Some(g) => g.1.push(k % w),
                None => groups.push(((one, many, block), vec![k % w])),
            }
        }
        groups
            .into_iter()
            .map(|((one, many, block), list)| {
                let of = if blocks { format!(" of {}", self.block_name(block)) } else { String::new() };
                if list.len() == 1 {
                    format!("warp {}{of} {one}", list[0])
                } else {
                    format!("warps {}{of} {many}", warp_list(&list))
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// How a report names the mbarrier keyed by `key`.
    fn mbarrier_name(&self, key: MbarrierKey) -> String {
        let place = smem_place(self.ir.buffer(BufferId(key.1)), key.2);
        if self.blocks.len() > 1 { format!("{place} of {}", self.block_name(key.0)) } else { place }
    }

    /// `the mbarrier m[0] for phase 2 (parity 0), which still waits for 1
    /// of 2 arrivals, ...`: the state of a phase that warps wait for.
    fn mbarrier_phase(&self, key: MbarrierKey, phase: u64) -> String {
        let name = self.mbarrier_name(key);
        let Some(m) = self.mbarriers.get(&key) else {
            return format!("the mbarrier {name}, which is not initialised");
        };
        if m.phase != phase {
            return format!("the mbarrier {name} for phase {phase} (parity {}), which has completed", phase & 1);
        }
        let mut what = format!(
            "the mbarrier {name} for phase {phase} (parity {}), which still waits for {} of its {} arrival(s)",
            phase & 1,
            m.arrivals_left,
            m.arrivals_per_phase
        );
        if !m.deferred.is_empty() {
            what += &format!(" ({} owed by cp.async.mbarrier.arrive)", m.deferred.len());
        }
        what += &format!(
            ", with a tx-count of {} byte(s) and {} bulk cop{} in flight",
            m.tx_count,
            m.copies.len(),
            if m.copies.len() == 1 { "y" } else { "ies" }
        );
        what
    }

    /// One line per blocked warp, warps in the same state together, and
    /// the location and thread of the first.
    fn blocked_report(&self) -> (Vec<String>, Option<(Loc, usize)>) {
        let w = self.warps_per_block;
        // (tag, the groups (warp, lanes) in that state, the rest of the line)
        type Line = (String, Vec<(usize, String)>, String);
        let mut lines: Vec<Line> = Vec::new();
        let mut first: Option<(Loc, usize)> = None;
        for (k, g) in self.all_groups() {
            let state = self.group(k, g);
            let Status::Blocked(wait) = state.status else { continue };
            let t = state.lanes().first().copied().unwrap_or((k / w) * self.threads_per_block + (k % w) * 32);
            let inst = match wait {
                Wait::Barrier { inst, .. }
                | Wait::Mbarrier { inst, .. }
                | Wait::Cluster { inst }
                | Wait::Join { inst, .. }
                | Wait::SetMaxNReg { inst, .. }
                | Wait::Wgmma { inst, .. }
                | Wait::TmemAlloc { inst, .. }
                | Wait::Rendezvous { inst } => inst,
            };
            first.get_or_insert((inst.loc, t));
            let ctx = self.group_context(k, state);
            let lanes = self.partial_lanes(state, t);
            let (block, rest) = match wait {
                Wait::Barrier { block, id, .. } => {
                    let s = &self.named[block][id];
                    let others: Vec<usize> = s.arrived.iter().copied().filter(|a| !s.waiting.contains(a)).collect();
                    let also = if others.is_empty() {
                        String::new()
                    } else {
                        format!(", and warps {} have arrived without waiting", warp_list(&others))
                    };
                    let missing: Vec<usize> =
                        (0..w).filter(|o| !s.arrived.contains(o)).map(|o| block * w + o).collect();
                    let exited = self.exited_warps(block);
                    let live = missing.len() - exited.len();
                    let mut rest = format!(
                        " at {}{ctx}, which needs {} warps{also}; {}",
                        self.wait_site(&wait),
                        s.expected,
                        self.whereabouts_of(&missing, false)
                    );
                    if !exited.is_empty() && s.arrived.len() + live < s.expected {
                        rest += &format!(
                            ". It can only complete through warps that have exited ({}): PTX may release a barrier \
                             for threads that exited, but that is almost always a bug",
                            warp_list(&exited)
                        );
                    }
                    (block, rest)
                }
                Wait::Mbarrier { key, phase, .. } => {
                    (k / w, format!(" at {}{ctx} on {}", self.wait_site(&wait), self.mbarrier_phase(key, phase)))
                }
                Wait::Cluster { .. } => {
                    let c = (k / w) / self.cluster_size;
                    let phase = self.cluster_barriers[c].phase;
                    let mut missing: Vec<usize> = Vec::new();
                    let mut names: Vec<String> = Vec::new();
                    for o in c * self.cluster_size * w..(c + 1) * self.cluster_size * w {
                        if matches!(self.warps[o].status, Status::Exited) {
                            continue;
                        }
                        let start = (o / w) * self.threads_per_block + (o % w) * 32;
                        let lanes: Vec<usize> = (0..self.warp_size(start))
                            .filter(|&l| self.cluster_arrival[start + l] != Some(phase))
                            .collect();
                        if lanes.is_empty() {
                            continue;
                        }
                        missing.push(o);
                        let partial = if lanes.len() == self.warp_size(start) {
                            String::new()
                        } else {
                            format!(" (lanes {})", warp_list(&lanes))
                        };
                        names.push(format!("warp {}{partial} of {}", o % w, self.block_name(o / w)));
                    }
                    (
                        usize::MAX,
                        format!(
                            " at {}{ctx} for phase {phase}, which {} of the cluster have not arrived at ({})",
                            self.wait_site(&wait),
                            names.join(", "),
                            self.whereabouts_of(&missing, true)
                        ),
                    )
                }
                Wait::SetMaxNReg { block, group, .. } => (block, self.setmaxnreg_wait(block, group, &ctx)),
                Wait::Wgmma { block, group, .. } => (block, self.wgmma_wait_report(block, group, &ctx)),
                Wait::TmemAlloc { block, .. } => (block, self.tmem_wait_report(block, &ctx)),
                Wait::Join { .. } => {
                    let mine = self.lane_set(state);
                    let others: Vec<usize> =
                        (0..self.warp_size(t - self.lane_of(t))).filter(|l| !mine.contains(l)).collect();
                    let rest = format!(
                        " at {}{ctx} for the lanes of its else-branch: {}",
                        self.wait_site(&wait),
                        self.lanes_whereabouts(k, &others)
                    );
                    (k / w, rest)
                }
                Wait::Rendezvous { .. } => {
                    let (Op::Shuffle { mask, .. }
                    | Op::Vote { mask, .. }
                    | Op::Match { mask, .. }
                    | Op::Redux { mask, .. }
                    | Op::WarpBarrier { mask }) = &inst.op
                    else {
                        unreachable!("a .sync warp collective")
                    };
                    let mine = self.lane_set(state);
                    let needed = state.lanes().iter().fold(0u32, |acc, &t| acc | self.get(*mask, t) as u32);
                    let missing: Vec<usize> = (0..self.warp_size(t - self.lane_of(t)))
                        .filter(|&l| needed >> l & 1 == 1 && !mine.contains(&l))
                        .filter(|&l| {
                            !self.parked[k].iter().chain(std::iter::once(&self.warps[k])).any(|o| {
                                matches!(o.status, Status::Blocked(Wait::Rendezvous { inst: at }) if same_collective(at, inst))
                                    && self.lane_set(o).contains(&l)
                            })
                        })
                        .collect();
                    let rest = format!(
                        " at {}{ctx} with member mask {needed:#010x}, for lanes {} of the mask to reach it (every lane \
                         of the mask must execute it; lanes of other branches may, with independent thread \
                         scheduling): {}",
                        self.wait_site(&wait),
                        warp_list(&missing),
                        self.lanes_whereabouts(k, &missing)
                    );
                    (k / w, rest)
                }
            };
            let tag = format!("{block}{rest}");
            match lines.iter_mut().find(|l| l.0 == tag) {
                Some(l) => l.1.push((k, lanes)),
                None => lines.push((tag, vec![(k, lanes)], rest)),
            }
        }
        let out = lines
            .into_iter()
            .map(|(_, warps, rest)| {
                let block = warps[0].0 / w;
                let cluster_wide = warps.iter().any(|&(k, _)| k / w != block);
                let partial = warps.iter().any(|(_, lanes)| !lanes.is_empty());
                let subject = if cluster_wide || partial {
                    let of =
                        |k: usize| if cluster_wide { format!(" of {}", self.block_name(k / w)) } else { String::new() };
                    let list = warps
                        .iter()
                        .map(|(k, lanes)| format!("warp {}{lanes}{}", k % w, of(*k)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    if cluster_wide { list } else { format!("{list} of {}", self.block_name(block)) }
                } else if warps.len() == 1 {
                    format!("warp {} of {}", warps[0].0 % w, self.block_name(block))
                } else {
                    let list: Vec<usize> = warps.iter().map(|(k, _)| k % w).collect();
                    format!("warps {} of {}", warp_list(&list), self.block_name(block))
                };
                let verb = if warps.len() == 1 { "waits" } else { "wait" };
                format!("{subject} {verb}{rest}")
            })
            .collect();
        (out, first)
    }

    /// For hang and deadlock reports, the warps that
    /// executed `exit`, and where.
    fn exit_report(&self) -> Vec<String> {
        let w = self.warps_per_block;
        let mut sites: Vec<(&usize, &Loc)> = self.exit_sites.iter().collect();
        sites.sort_by_key(|e| *e.0);
        sites
            .into_iter()
            .map(|(&k, at)| {
                format!(
                    "warp {} of {} exited at the exit at {}:{}: the arrivals and stores after it never happen",
                    k % w,
                    self.block_name(k / w),
                    at.file(),
                    at.line()
                )
            })
            .collect()
    }

    /// The report when no warp can run and some are blocked: each blocked
    /// warp, where it waits and what it waits for, and where the warps it
    /// waits for are.
    fn deadlock(&self) -> SimError {
        let (mut lines, first) = self.blocked_report();
        lines.extend(self.exit_report());
        let (loc, t) = first.expect("a deadlock has a blocked warp");
        SimError {
            loc: Some(loc),
            ..self.err(
                format!(
                    "the kernel did not terminate: every warp that has not exited is blocked (a deadlock):\n  - {}",
                    lines.join("\n  - ")
                ),
                Some(t),
            )
        }
    }

    /// The report when nothing visible has happened for
    /// [`SimOptions::max_idle_steps`] steps and every warp that has not
    /// exited is blocked or spinning: the spinning warps, their loops and
    /// what they poll, then the blocked warps.
    fn hang(&self) -> SimError {
        let w = self.warps_per_block;
        let mut lines: Vec<String> = Vec::new();
        let mut first: Option<(Loc, usize)> = None;
        let mut polls_global = false;
        for (k, g) in self.all_groups() {
            let state = self.group(k, g);
            if !matches!(state.status, Status::Runnable) {
                continue;
            }
            let t = state.lanes().first().copied().unwrap_or((k / w) * self.threads_per_block + (k % w) * 32);
            let polled = match state.poll {
                Some(Poll::Mbarrier { key, phase, inst }) => {
                    first.get_or_insert((inst.loc, t));
                    format!(
                        ", polling with mbarrier.test_wait ({}) {}",
                        Self::at(inst),
                        self.mbarrier_phase(key, phase)
                    )
                }
                Some(Poll::Load { buffer, block, byte, inst }) => {
                    first.get_or_insert((inst.loc, t));
                    let d = self.ir.buffer(buffer);
                    polls_global |= block == GLOBAL;
                    let place = if block == GLOBAL {
                        format!("{}[{}]", d.name, byte / d.elem.bytes())
                    } else {
                        smem_place(d, byte)
                    };
                    format!(", polling {place} with an ordered load ({})", Self::at(inst))
                }
                None => ", without polling memory".to_string(),
            };
            let lanes = self.partial_lanes(state, t);
            lines.push(format!(
                "warp {}{lanes} of {} spins{}{polled}",
                k % w,
                self.block_name(k / w),
                self.group_context(k, state)
            ));
        }
        let (blocked, first_blocked) = self.blocked_report();
        lines.extend(blocked);
        lines.extend(self.exit_report());
        let (loc, t) = first.or(first_blocked).unwrap_or((self.loc.expect("a step ran"), 0));
        let mut message = format!(
            "the kernel did not terminate: nothing visible has happened for {} warp steps and every warp that has not \
             exited is blocked or spinning (a hang):\n  - {}",
            self.opts.max_idle_steps,
            lines.join("\n  - ")
        );
        if polls_global && let Some(hint) = self.residency_hint() {
            message += &hint;
        }
        SimError { loc: Some(loc), ..self.err(message, Some(t)) }
    }

    /// For hang reports: the clusters of the grid that have not started, if
    /// any, which a spin on global memory may be waiting for.
    fn residency_hint(&self) -> Option<String> {
        if self.queued.is_empty() {
            return None;
        }
        Some(format!(
            "\n{} cluster(s) of the grid have not started: the simulator runs {} cluster(s) at a time \
             (SimOptions::resident_clusters), and a cluster that waits for one that has not started waits forever. \
             If the kernel needs its clusters to run together (a grid-wide barrier), launch it as a cooperative \
             launch (LaunchConfig::with_cooperative(true)), which the GPU runs only with every block resident, or \
             set resident_clusters to ResidentClusters::Grid or a count",
            self.queued.len(),
            self.cluster_ids.len()
        ))
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

    /// The address of byte `byte` of buffer `b` in the buffer's own space
    /// ([`AddrMap`]): a global address, or a shared or local offset.
    fn address_of(&self, b: BufferId, byte: usize) -> u64 {
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
    fn read_only(&self, b: BufferId) -> Option<&'static str> {
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
    fn resolve(&self, space: AddressSpace, addr: u64) -> Result<(BufferId, usize), String> {
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
    fn target(&self, b: BufferId, index: ValueId, size: usize, t: usize, what: &str) -> R<(BufferId, i64)> {
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
    fn cvt_addr(&self, x: u64, from: AddressSpace, to: AddressSpace) -> Result<u64, String> {
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

    fn buffer_bytes(&self, b: BufferId) -> usize {
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

    fn read(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R<u64> {
        self.read_as(b, byte, size, t, false)
    }

    /// A read by thread `t`, strong (ordered, volatile or MMIO) or weak.
    fn read_as(&mut self, b: BufferId, byte: usize, size: usize, t: usize, strong: bool) -> R<u64> {
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
    fn write_as(&mut self, b: BufferId, byte: usize, size: usize, t: usize, bits: u64, strong: bool) -> R {
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
    fn smem_claim_write(
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
    #[allow(clippy::too_many_arguments)]
    fn smem_claim_write_in(
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
    fn smem_land(&mut self, b: BufferId, block: usize, range: std::ops::Range<usize>, writer: usize, clock: u32) {
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
    fn atomic_update(&mut self, b: BufferId, byte: usize, size: usize, t: usize, f: impl FnOnce(u64) -> u64) -> R<u64> {
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
    #[allow(clippy::too_many_arguments)]
    fn atomic(
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
                // A compare-and-swap that fails only reads (m11: so a spin on one is a poll,
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

    fn inst(&mut self, inst: &'a Inst, active: &[usize]) -> R {
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
                    // m7-tma-sm100-modes: `.cp_mask` writes the bytes of each 16-byte chunk
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

    /// Whether no asynchronous operation reads memory in the background
    /// (the common case, which skips [`Self::check_async_reads`]).
    fn pending_reads_free(&self) -> bool {
        !self.any_async_reads
    }

    /// Report a read by `active` of an operand that an asynchronous
    /// operation still has to write (a register result, such as a WGMMA
    /// accumulator).
    fn check_pending_operands(&self, operands: &[ValueId], active: &[usize]) -> R {
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
    fn check_async_reads(&self, b: BufferId, range: std::ops::Range<usize>, t: usize) -> R {
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
    fn fence_proxy_async(&mut self, k: usize) {
        let slot = self.slot_of(k);
        let now = self.clocks[k][slot];
        self.proxy_fences[k].push(now);
        self.clocks[k][slot] += 1;
    }

    /// Before a bulk copy by thread `t` reads bytes `range` of shared buffer
    /// `b` of `block`: no asynchronous copy may still be writing them, and
    /// their last write must be ordered before the issue.
    fn bulk_store_may_read(&self, b: BufferId, block: usize, range: std::ops::Range<usize>, t: usize) -> R {
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
    fn note_bulk_source_read(&mut self, t: usize, b: BufferId, block: usize, range: std::ops::Range<usize>) {
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
    fn check_pending_global(&self, b: BufferId, range: std::ops::Range<usize>, t: usize, what: &str) -> R {
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
    fn land_bulk_store(&mut self, id: usize) {
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
    fn land_bulk_stores_at_exit(&mut self, block: usize) {
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
    fn async_proxy_may_read(
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
    /// effects m7 and m8 add: shared-memory writes, background reads and
    /// register results.
    #[cfg(test)]
    fn issue_test_op(
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
    fn tma_issue_desc(&self, a: &TmaIssue<'_>, t: usize, what: &str) -> R<(crate::tma::TmaDesc, Vec<i64>, Vec<i64>)> {
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
    fn tma_map(&self, desc: ValueId, t: usize, what: &str) -> R<crate::tma::TmaDesc> {
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
    fn check_bulk_mbarrier(&self, key: MbarrierKey, t: usize, what: &str) -> R {
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
    fn async_source_may_read(
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
    fn cluster_targets(&self, mask: Option<ValueId>, t: usize, what: &str) -> R<Vec<usize>> {
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
    fn need_clusters(&self, t: usize, what: &str) -> R {
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
    fn check_copy_mbarrier(&self, key: MbarrierKey, t: usize, what: &str) -> R {
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
    #[allow(clippy::too_many_arguments)]
    fn issue_to_mbarrier(
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
    #[allow(clippy::too_many_arguments)]
    fn tma_load(
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
    fn tma_store(
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
        // box inside the tensor (the PTX ISA's restrictions on tensor copies).
        if start.iter().any(|&c| c < 0) && !access.mode.is_rows4() {
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
    fn land_bulk_copies(&mut self, key: (usize, u32, usize), t: usize) -> R {
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
    fn complete_bulk_copy(&mut self, id: usize, key: MbarrierKey) {
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
    fn complete_as_agent(&mut self, id: usize, tracked: bool) -> u32 {
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
    fn complete(&mut self, id: usize, writer: usize, time: u32) {
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
    fn bulk_copy_writing(&self, b: BufferId, block: usize, byte: usize) -> bool {
        self.ops
            .iter()
            .any(|op| op.in_flight() && matches!(op.trigger, Trigger::CompleteTx(_)) && op.writes_byte(b, block, byte))
    }

    /// Whether byte `byte` of shared buffer `b` of `block` is the
    /// destination of a `cp.async` in flight that a `cp.async.mbarrier.arrive`
    /// tracks.
    fn tracked_copy_writing(&self, b: BufferId, block: usize, byte: usize) -> bool {
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
    fn land_cp_async_for_wait(&mut self, id: usize, t: usize) {
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
    fn land_tracked_copy(&mut self, id: usize) -> u32 {
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
    fn fire_deferred_arrivals(&mut self, key: (usize, u32, usize)) {
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
    fn fire_ready_arrivals(&mut self, key: MbarrierKey) {
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
    fn check_remote_mbarrier(&self, key: MbarrierKey, t: usize, what: &str) -> R {
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
    fn mbarrier_block(&self, peer: Option<ValueId>, t: usize, what: &str) -> R<usize> {
        let Some(peer) = peer else { return Ok(self.block_rank(t)) };
        let rank = self.get(peer, t) as u32 as i32;
        if rank < 0 || rank as usize >= self.cluster_size {
            return Err(self.err(format!("{what} on CTA {rank} of a cluster of {} CTA(s)", self.cluster_size), Some(t)));
        }
        Ok(self.cluster_base(self.block_rank(t)) + rank as usize)
    }

    /// The index of thread `t`'s warp in [`Self::clocks`].
    fn clock_index(&self, t: usize) -> usize {
        self.block_rank(t) * self.warps_per_block + self.warp_of(t) as usize
    }

    /// The clock slot of warp `k` (by key, `block * warps_per_block +
    /// warp`): vector clocks span the warp's cluster, with `stride` slots
    /// per block, its warps' first and then its agents'.
    fn slot_of(&self, k: usize) -> usize {
        let w = self.warps_per_block;
        ((k / w) % self.cluster_size) * self.stride + k % w
    }

    /// The clock slot of `block`'s asynchronous-copy agent.
    fn agent_slot(&self, block: usize) -> usize {
        (block % self.cluster_size) * self.stride + self.warps_per_block + COPY_AGENT
    }

    /// Whether clock slot `slot` is an agent's.
    fn is_agent(&self, slot: usize) -> bool {
        slot % self.stride >= self.warps_per_block
    }

    /// Clock slots per vector clock: every warp and agent of a cluster.
    fn slots(&self) -> usize {
        self.cluster_size * self.stride
    }

    /// How a report by a warp of `block` names clock slot `slot`: `warp 3`
    /// in its own block, `warp 3 of CTA 1` in another of its cluster.
    fn slot_name(&self, slot: usize, block: usize) -> String {
        let (rank, i) = (slot / self.stride, slot % self.stride);
        let what =
            if i >= self.warps_per_block { "the asynchronous-copy agent".to_string() } else { format!("warp {i}") };
        if rank == block % self.cluster_size { what } else { format!("{what} of CTA {rank}") }
    }

    /// The first block of `block`'s cluster.
    fn cluster_base(&self, block: usize) -> usize {
        block - block % self.cluster_size
    }

    /// The key of byte `byte` of buffer `b`, as thread `t` addresses it, in
    /// [`Self::releases`].
    fn release_key(&self, b: BufferId, byte: usize, t: usize) -> (u32, usize, usize) {
        let block = if self.ir.buffer(b).space == MemSpace::Shared { self.smem_block(t) } else { GLOBAL };
        (b.0, block, byte)
    }

    /// The join of the clocks published on the `size` bytes at `byte` of
    /// `b`, which a strong read reads as elements of `elem` bytes, by warps
    /// of thread `t`'s block (the only ones whose clocks it shares). An
    /// element joins what a write published only if that write covered
    /// exactly the element's bytes (a morally strong pair).
    fn released_on(&self, b: BufferId, byte: usize, size: usize, elem: usize, t: usize) -> Option<Vec<u32>> {
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
    fn publish(&mut self, b: BufferId, byte: usize, size: usize, elem: usize, t: usize, how: Publish) {
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
    fn fence_lanes(&mut self, active: &[usize]) {
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
    fn nc_read(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R {
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
    fn nc_write(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R {
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
    fn synchronize(&mut self, block: usize, releasing: &[usize], acquiring: &[usize], release: &[u32]) {
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

    /// `setmaxnreg` by the warps of `group`: each joins its warpgroup's pending one and waits;
    /// once all four warps have, it completes (an increase once the block's pool has the
    /// registers). Checks that whole warps of whole warpgroups execute the same one, that
    /// the warps synchronized since the previous one, and the budget the PTX ISA requires.
    fn setmaxnreg(&mut self, inst: &'a Inst, group: &[usize], increase: bool, count: u32) -> R {
        let w = self.warps_per_block;
        let tpb = self.threads_per_block;
        let what = inst.op.name();
        if !tpb.is_multiple_of(128) {
            return Err(self
                .err(format!("{what} needs whole warpgroups of 128 threads, but the block has {tpb} threads"), None));
        }
        let mut full = Vec::new();
        for &k in group {
            let lanes = self.warp_lanes(k).to_vec();
            let (block, warp) = (k / w, k % w);
            if lanes.len() != 32 {
                return Err(self.err(
                    format!(
                        "{what} {count} executed by {} of the 32 lanes of warp {warp}: every thread of the warpgroup \
                         executes it together (.sync.aligned), outside divergent code",
                        lanes.len()
                    ),
                    Some(lanes[0]),
                ));
            }
            let wg = warp / 4;
            let slot = self.slot_of(k);
            let seen = self.clocks[k].clone();
            let groups = tpb / 128;
            let initial = setmaxnreg_initial_budget(tpb);
            let state = self.setmaxnreg.entry(block).or_insert_with(|| BlockRegs {
                free: 0,
                groups: vec![WarpgroupRegs { budget: initial, pending: None, last: Vec::new() }; groups],
            });
            let g = &mut state.groups[wg];
            if let Some(&(o, _, _)) = g.last.iter().find(|&&(o, s, time)| o != k && seen[s] <= time) {
                return Err(self.err(
                    format!(
                        "{what} {count} by warp {warp} of warpgroup {wg} before it has synchronized with warp {} since \
                         their previous setmaxnreg: separate consecutive setmaxnreg of a warpgroup with a barrier \
                         (a named barrier over the warpgroup, say)",
                        o % w
                    ),
                    Some(lanes[0]),
                ));
            }
            let p =
                g.pending.get_or_insert(PendingRegs { increase, count, inst, arrived: Vec::new(), joined: Vec::new() });
            if (p.increase, p.count) != (increase, count) {
                let other = format!("setmaxnreg.{} {}", if p.increase { "inc" } else { "dec" }, p.count);
                let at = Self::at(p.inst);
                let others = warp_list(&p.arrived.iter().map(|a| 4 * wg + a).collect::<Vec<_>>());
                return Err(self.err(
                    format!(
                        "{what} {count} by warp {warp} while warps {others} of warpgroup {wg} execute {other} ({at}): \
                         the warps of a warpgroup execute the same setmaxnreg"
                    ),
                    Some(lanes[0]),
                ));
            }
            if p.arrived.contains(&(warp % 4)) {
                let missing: Vec<usize> = (0..4).filter(|a| !p.arrived.contains(a)).map(|a| 4 * wg + a).collect();
                return Err(self.err(
                    format!(
                        "{what} {count}: warp {warp} executes it again before warps {} of warpgroup {wg} have executed \
                         the previous one",
                        warp_list(&missing)
                    ),
                    Some(lanes[0]),
                ));
            }
            p.arrived.push(warp % 4);
            p.joined.push((k, slot, seen[slot]));
            if p.arrived.len() == 4 {
                full.push((block, wg));
            }
            self.clocks[k][slot] += 1;
            self.advance(k);
            self.warps[k].status = Status::Blocked(Wait::SetMaxNReg { block, group: wg, inst });
        }
        self.progress();
        for (block, wg) in full {
            self.complete_setmaxnreg(block, wg)?;
        }
        Ok(())
    }

    /// Complete warpgroup `wg`'s pending `setmaxnreg` if every warp has executed it and, for
    /// an increase, the pool has the registers; a decrease then lets waiting increases of
    /// the block go ahead.
    fn complete_setmaxnreg(&mut self, block: usize, wg: usize) -> R {
        let state = self.setmaxnreg.get_mut(&block).expect("a setmaxnreg state");
        let g = &mut state.groups[wg];
        let Some(p) = g.pending.as_ref().filter(|p| p.arrived.len() == 4) else { return Ok(()) };
        let (increase, count, budget, inst) = (p.increase, p.count, g.budget, p.inst);
        let what = format!("setmaxnreg.{} {count}", if increase { "inc" } else { "dec" });
        if increase && budget > count {
            return Err(SimError {
                loc: Some(inst.loc),
                ..self.err(
                    format!(
                        "{what} by warpgroup {wg}, whose budget is already {budget} registers per thread: an increase \
                         needs a budget of at most its count"
                    ),
                    None,
                )
            });
        }
        if !increase && budget < count {
            return Err(SimError {
                loc: Some(inst.loc),
                ..self.err(
                    format!(
                        "{what} by warpgroup {wg}, whose budget is only {budget} registers per thread: a decrease \
                         needs a budget of at least its count"
                    ),
                    None,
                )
            });
        }
        let need = (count.saturating_sub(budget)) * 128;
        if increase && state.free < need {
            // It waits for registers.
            return Ok(());
        }
        if increase {
            state.free -= need;
        } else {
            state.free += (budget - count) * 128;
        }
        let g = &mut state.groups[wg];
        let p = g.pending.take().expect("pending");
        g.budget = count;
        g.last = p.joined.clone();
        let round = self.round;
        for &(k, _, _) in &p.joined {
            let s = &mut self.warps[k];
            s.status = Status::Runnable;
            s.ready_round = round + 1;
        }
        self.progress();
        if !increase {
            let groups = self.setmaxnreg[&block].groups.len();
            for other in (0..groups).filter(|&o| o != wg) {
                self.complete_setmaxnreg(block, other)?;
            }
        }
        Ok(())
    }

    /// For hang and deadlock reports: what a warp waiting at a `setmaxnreg` waits for.
    fn setmaxnreg_wait(&self, block: usize, wg: usize, ctx: &str) -> String {
        let w = self.warps_per_block;
        let Some(state) = self.setmaxnreg.get(&block) else { return String::new() };
        let g = &state.groups[wg];
        let Some(p) = &g.pending else { return String::new() };
        let site = format!("setmaxnreg.{} {} ({})", if p.increase { "inc" } else { "dec" }, p.count, Self::at(p.inst));
        if p.arrived.len() < 4 {
            let missing: Vec<usize> =
                (0..4).filter(|a| !p.arrived.contains(a)).map(|a| block * w + 4 * wg + a).collect();
            return format!(
                " at {site}{ctx}, which warps {} of warpgroup {wg} have not executed: {}",
                warp_list(&missing.iter().map(|k| k % w).collect::<Vec<_>>()),
                self.whereabouts_of(&missing, false)
            );
        }
        let need = (p.count - g.budget) * 128;
        format!(
            " at {site}{ctx}, an increase from {} to {} registers per thread that needs {need} registers from the \
             block's pool, which has {}: no warpgroup releases them (setmaxnreg.dec), so it can never be served",
            g.budget, p.count, state.free
        )
    }

    /// A barrier instruction (`inst`) executed by the warps of `group`:
    /// `sync_threads` (`id` is `None`: barrier 0, for every thread of the
    /// block), or `bar.sync` / `bar.arrive` on barrier `id`, of `threads`
    /// threads (every thread of the block without a count). Arriving warps
    /// that wait block until the barrier completes; the last arrival
    /// completes it and wakes them.
    fn barrier(
        &mut self,
        inst: &'a Inst,
        group: &[usize],
        kind: BarrierKind,
        id: Option<ValueId>,
        threads: Option<ValueId>,
        red: Option<(BarrierRedKind, ValueId, ValueId)>,
    ) -> R {
        let what = match (id, kind) {
            _ if red.is_some() => "barrier_cta_red",
            (None, _) => "sync_threads()",
            (_, BarrierKind::Sync) => "barrier",
            (_, BarrierKind::Arrive) => "barrier_arrive",
            (_, BarrierKind::ArriveUnaligned) => "barrier_arrive (unaligned)",
        };
        let w = self.warps_per_block;
        // (block, barrier id) -> (thread count, arriving warps)
        let mut arrivals: BTreeMap<(usize, usize), (Option<usize>, Vec<usize>)> = BTreeMap::new();
        // A bar.red's true and false predicates, and
        // its arriving warps' result values, by (block, barrier id).
        let mut red_parts: BTreeMap<(usize, usize), RedPart> = BTreeMap::new();
        for &k in group {
            let lanes = self.warp_lanes(k);
            let base = lanes[0] - self.lane_of(lanes[0]);
            let (block, warp) = (k / w, k % w);
            let partial = lanes.len() != self.warp_size(base);
            if partial && kind != BarrierKind::ArriveUnaligned {
                if id.is_none() {
                    let n: usize = group.iter().filter(|&&o| o / w == block).map(|&o| self.warp_lanes(o).len()).sum();
                    return Err(self.err(
                        format!(
                            "sync_threads() reached by {n} of {} threads (barriers must not be inside divergent \
                             control flow)",
                            self.threads_per_block
                        ),
                        None,
                    ));
                }
                return Err(self.err(
                    format!(
                        "{what} executed by {} of the {} lanes of warp {warp} (each participating warp must reach it as a whole)",
                        lanes.len(),
                        self.warp_size(base)
                    ),
                    Some(lanes[0]),
                ));
            }
            let uniform = |v: ValueId, name: &str| -> R<i32> {
                let first = self.get(v, lanes[0]) as u32 as i32;
                match lanes.iter().find(|&&t| self.get(v, t) as u32 as i32 != first) {
                    Some(&t) => Err(self.err(
                        format!(
                            "{what}: the lanes of warp {warp} disagree on the {name} ({first} and {})",
                            self.get(v, t) as u32 as i32
                        ),
                        Some(t),
                    )),
                    None => Ok(first),
                }
            };
            let b = match id {
                Some(id) => uniform(id, "barrier id")?,
                None => 0,
            };
            if !(0..16).contains(&b) {
                return Err(self.err(format!("{what}: barrier id {b} is out of range (0 to 15)"), Some(base)));
            }
            let count = match threads {
                Some(n) => {
                    let n = uniform(n, "thread count")?;
                    let limit = w * 32;
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
            if partial {
                // barrier.arrive (unaligned) waits for every non-exited
                // thread of its warp: the warp arrives once all its lanes
                // have executed one for this barrier, from any instruction.
                let bits = lanes.iter().fold(0u32, |acc, &t| acc | 1 << self.lane_of(t));
                let full = if self.warp_size(base) == 32 { u32::MAX } else { (1u32 << self.warp_size(base)) - 1 };
                let first = lanes[0];
                let seen = self.unaligned.get(&(block, b as usize, warp)).copied().unwrap_or(0);
                if seen & bits != 0 {
                    return Err(self.err(
                        format!(
                            "{what}: lanes {} of warp {warp} arrive at barrier {b} again before the rest of their \
                             warp has arrived",
                            warp_list(&(0..32).filter(|l| (seen & bits) >> l & 1 == 1).collect::<Vec<_>>())
                        ),
                        Some(first),
                    ));
                }
                if seen | bits != full {
                    self.unaligned.insert((block, b as usize, warp), seen | bits);
                    continue;
                }
                self.unaligned.remove(&(block, b as usize, warp));
            }
            let entry = arrivals.entry((block, b as usize)).or_insert((count, Vec::new()));
            if entry.0 != count {
                return Err(self.err(
                    format!("{what}: warps arriving at barrier {b} together disagree on its thread count"),
                    Some(base),
                ));
            }
            entry.1.push(warp);
            if let Some((_, pred, result)) = red {
                let lanes = self.warp_lanes(k).to_vec();
                let part = red_parts.entry((block, b as usize)).or_default();
                for t in lanes {
                    if self.get(pred, t) & 1 == 1 {
                        part.0 += 1;
                    } else {
                        part.1 += 1;
                    }
                }
                part.2.push((k, result));
            }
        }
        for &k in group {
            self.advance(k);
        }
        self.progress();
        let waits = kind == BarrierKind::Sync;
        for ((block, b), (count, warps)) in arrivals {
            let expected = count.map_or(w, |n| n / 32);
            let all_threads = count.is_none() && waits;
            let state = &self.named[block][b];
            let pending = |s: &NamedState| -> Vec<usize> {
                s.arrived.iter().copied().filter(|a| !s.waiting.contains(a)).collect()
            };
            if id.is_none() && !pending(state).is_empty() {
                return Err(self.err(
                    format!(
                        "sync_threads() uses barrier 0 while warps {} have arrived at it with barrier_arrive and it has not completed",
                        warp_list(&pending(state))
                    ),
                    None,
                ));
            }
            if id.is_some() && state.site.is_some_and(|s| matches!(s.op, Op::Barrier)) {
                return Err(self.err(
                    format!(
                        "sync_threads() uses barrier 0 while warps {} arrive at it with {what} before it has \
                         completed (warps {} wait at the sync_threads())",
                        warp_list(&warps),
                        warp_list(&state.waiting)
                    ),
                    None,
                ));
            }
            if !state.arrived.is_empty() && state.expected != expected {
                return Err(self.err(
                    format!(
                        "{what}: barrier {b} is used with {} threads while warps {} wait on it with {} (the warps \
                         arriving at it disagree on its thread count)",
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
            // An all-thread barrier must be reached at one instruction, by
            // every warp of the block.
            if all_threads || state.all_threads {
                if let Some(site) = state.site.filter(|s| !std::ptr::eq(*s, inst)) {
                    return Err(self.err(
                        format!(
                            "{what} reached by {} of {} threads here, while warps {} wait at another one ({}:{}) \
                             (barriers must not be inside divergent control flow)",
                            self.threads_in(&warps),
                            self.threads_per_block,
                            warp_list(&state.waiting),
                            site.loc.file(),
                            site.loc.line()
                        ),
                        None,
                    ));
                }
                let exited = self.exited_warps(block);
                if !exited.is_empty() && total + exited.len() == expected {
                    let mut arrived = state.arrived.clone();
                    arrived.extend(&warps);
                    return Err(self.exit_error(inst, block, &arrived, &exited));
                }
            }
            // A bar.red's warps must all reduce the same way.
            let red_kind = red.map(|r| r.0);
            if !state.arrived.is_empty() && state.red != red_kind {
                let name = |k: Option<BarrierRedKind>| match k {
                    Some(k) => format!("barrier_cta_red ({})", k.name()),
                    None => "a plain barrier".to_string(),
                };
                return Err(self.err(
                    format!(
                        "{what}: warps {} reach barrier {b} with {} while warps {} wait at it with {}: every thread \
                         of a barrier's use executes the same reduction",
                        warp_list(&warps),
                        name(red_kind),
                        warp_list(&state.arrived),
                        name(state.red)
                    ),
                    None,
                ));
            }
            let part = red_parts.remove(&(block, b)).unwrap_or_default();
            let state = self.named[block][b].clone();
            let release = if state.release.is_empty() { vec![0; self.slots()] } else { state.release.clone() };
            if total == expected {
                if let Some(kind) = red_kind {
                    // Every participant gets the reduction of all predicates.
                    let (yes, no) = (state.red_counts.0 + part.0, state.red_counts.1 + part.1);
                    let value = match kind {
                        BarrierRedKind::Popc => yes as u64,
                        BarrierRedKind::And => (no == 0) as u64,
                        BarrierRedKind::Or => (yes > 0) as u64,
                    };
                    for &(k, result) in state.red_results.iter().chain(&part.2) {
                        let lanes = self.warp_lanes(k).to_vec();
                        for t in lanes {
                            self.set(result, t, value);
                        }
                    }
                }
                // Complete: waiting warps acquire everything the arrivals released.
                let mut acquiring = state.waiting.clone();
                if waits {
                    acquiring.extend(&warps);
                }
                self.synchronize(block, &warps, &acquiring, &release);
                for &warp in &state.waiting {
                    let woken = &mut self.warps[block * w + warp];
                    woken.status = Status::Runnable;
                    woken.ready_round = self.round + 1;
                }
                self.named[block][b] = NamedState::default();
                self.report.barriers += 1;
            } else {
                // Arrive (and wait, for bar.sync): the barrier keeps what
                // these warps release.
                let mut joined = release;
                for &warp in &warps {
                    join(&mut joined, &self.clocks[block * w + warp]);
                }
                self.synchronize(block, &warps, &[], &[]);
                let state = &mut self.named[block][b];
                state.arrived.extend(&warps);
                state.expected = expected;
                state.release = joined;
                state.red = red_kind;
                state.red_counts.0 += part.0;
                state.red_counts.1 += part.1;
                state.red_results.extend(part.2);
                if waits {
                    state.waiting.extend(&warps);
                    state.all_threads |= all_threads;
                    state.site.get_or_insert(inst);
                    for &warp in &warps {
                        self.warps[block * w + warp].status = Status::Blocked(Wait::Barrier { block, id: b, inst });
                    }
                }
            }
        }
        Ok(())
    }

    /// The report of `what` on the mbarrier keyed by
    /// `key` if `mbarrier.inval` invalidated it.
    fn invalidated_error(&self, key: MbarrierKey, what: &str, t: usize) -> Option<SimError> {
        let at = self.invalidated.get(&key)?;
        Some(self.err(
            format!(
                "{what} on an mbarrier that mbarrier_inval invalidated at {}:{}: after an inval only mbarrier_init \
                 may use its memory as an mbarrier again",
                at.file(),
                at.line()
            ),
            Some(t),
        ))
    }

    /// The mbarrier operations of [`MbarrierExtKind`].
    fn mbarrier_ext(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        use crate::ir::MbarrierExtKind as K;
        let Op::MbarrierExt { kind, result, buffer, index, value } = &inst.op else { unreachable!("an mbarrier op") };
        let what = kind.ptx();
        self.progress();
        let mut released: Vec<usize> = Vec::new();
        for &t in active {
            let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, what)?;
            let key = (self.block_rank(t), buffer.0, byte);
            if !self.mbarriers.contains_key(&key) {
                if let Some(e) = self.invalidated_error(key, what, t) {
                    return Err(e);
                }
                return Err(self.err(format!("{what} on an mbarrier that was never initialised"), Some(t)));
            }
            if *kind == K::Inval {
                let m = &self.mbarriers[&key];
                if !m.copies.is_empty() || !m.deferred.is_empty() {
                    return Err(self.err(
                        format!(
                            "mbarrier.inval on an mbarrier that {} bulk copies or cp.async arrivals still report to",
                            m.copies.len() + m.deferred.len()
                        ),
                        Some(t),
                    ));
                }
                self.mbarriers.remove(&key);
                self.invalidated.insert(key, inst.loc);
                continue;
            }
            let v = value.map_or(1, |v| self.get(v, t) as u32);
            let k = self.clock_index(t);
            let clock = self.clocks[k].clone();
            let state = self.mbarriers.get_mut(&key).expect("checked above");
            let token = state.phase;
            if *kind == K::CompleteTx {
                state.tx_count -= v as i64;
                state.complete_if_done();
            } else {
                // (arrivals, arrive-drop, expect-tx bytes, must not complete)
                let (n, drop, tx, nocomplete) = match kind {
                    K::ArriveState => (1, false, 0, false),
                    K::ArriveDrop => (1, true, 0, false),
                    K::ArriveDropExpectTx => (1, true, v, false),
                    K::ArriveDropNoComplete => (v, true, 0, true),
                    _ => (v, false, 0, true),
                };
                let (left, owed) = (state.arrivals_left, state.deferred.len() as u32);
                if n == 0 || n > left.saturating_sub(owed) {
                    return Err(self.err(
                        format!(
                            "{what}: {n} arrival(s) at an mbarrier whose phase is waiting for {left} ({owed} of them \
                             owed by asynchronous operations)"
                        ),
                        Some(t),
                    ));
                }
                if nocomplete && n == left && state.tx_count + tx as i64 == 0 {
                    return Err(self.err(
                        format!(
                            "{what} with {n} arrival(s) completes the mbarrier's phase, which a .noComplete arrival \
                             must not do (use mbarrier_arrive for the last arrival)"
                        ),
                        Some(t),
                    ));
                }
                join(&mut state.release, &clock);
                state.tx_count += tx as i64;
                state.arrivals_left -= n;
                if drop {
                    state.arrivals_per_phase -= n.min(state.arrivals_per_phase);
                }
                state.complete_if_done();
                if !released.contains(&k) {
                    released.push(k);
                }
            }
            if let Some(r) = result {
                self.set(*r, t, token);
            }
        }
        for k in released {
            let slot = self.slot_of(k);
            self.clocks[k][slot] += 1;
        }
        Ok(())
    }

    /// The warps of `group` execute `exit` (`inst`)
    /// with all their lanes: each ends as if it had reached the end of the
    /// kernel. An exit by part of a warp is not simulated.
    fn exit_warps(&mut self, inst: &'a Inst, group: &[usize]) -> R {
        for &k in group {
            let lanes = self.warp_lanes(k).to_vec();
            let base = lanes[0] - self.lane_of(lanes[0]);
            if lanes.len() != self.warp_size(base) || !self.parked[k].is_empty() {
                return Err(self.err(
                    format!(
                        "exit executed by {} of the {} lanes of warp {}: the simulator runs an exit only by a whole \
                         warp (exit by part of a warp is not simulated)",
                        lanes.len(),
                        self.warp_size(base),
                        k % self.warps_per_block
                    ),
                    Some(lanes[0]),
                ));
            }
            self.exit_sites.insert(k, inst.loc);
            self.report.exits += 1;
            let state = &mut self.warps[k];
            let frames = &mut state.groups[0].frames;
            frames.truncate(1);
            let body = &mut frames[0];
            body.next = body.region.insts.len();
            state.settled = false;
        }
        Ok(())
    }

    /// Lanes in the warp starting at thread `base` (the last warp of a block
    /// can be short).
    fn warp_size(&self, base: usize) -> usize {
        (self.threads_per_block - base % self.threads_per_block).min(32)
    }

    #[allow(clippy::too_many_arguments)]
    fn mma(
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
/// reservation, in bytes.
fn aggr_smem_size(smem: &SharedMemoryLayout, dynamic: usize) -> usize {
    smem.block_bytes(dynamic).next_multiple_of(SMEM_ALLOCATION_UNIT) + SMEM_SYSTEM_RESERVED
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
/// the least significant bit, the rest padding, each expanded as the PTX
/// ISA's decompression figures show (Figures 198 and 200, the same
/// `.dst_fmt.src_fmt` qualifiers on `tcgen05.cp`): a 4-bit value in bits 2
/// to 5 of its byte and a 6-bit value in bits 0 to 5, the other bits zero.
/// That is the container placement of `mma` with `.kind::f8f6f4` (E2M1 in
/// the central 4 bits, E3M2 and E2M3 in the low 6).
pub(crate) fn ldmatrix_expand(row: u128, unpack: Option<u8>) -> [u8; 16] {
    std::array::from_fn(|e| match unpack {
        None => (row >> (8 * e)) as u8,
        Some(4) => (((row >> (4 * e)) & 0xf) as u8) << 2,
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
        assert_eq!(ldmatrix_expand(b4, Some(4)), std::array::from_fn(|e| (e as u8) << 2));
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

// ---- m7-tma-im2col and m7-tma-sm100-modes ----

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
