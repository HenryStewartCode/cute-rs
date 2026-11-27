//! The instruction set: `LoopOptions`, the `Op` enum and its methods.

use super::*;

/// Hints for a runtime loop: the keyword arguments of CuTe DSL's
/// `cutlass.range(start, stop, step, unroll=, unroll_full=, vectorize=,
/// at_least_once=)`. The default attaches nothing, as in CuTe DSL.
///
/// The unroll and vectorize hints become LLVM loop metadata
/// (`#llvm.loop_annotation`), which NVPTX honours; they never change what the
/// loop computes, and the simulator ignores them. Fields not given keep
/// their defaults: `LoopOptions { vectorize: true, ..LoopOptions::default()
/// }`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LoopOptions {
    /// `unroll=K`: unroll the loop `K` times (`count = K`). 1 keeps it rolled
    /// (`disable = true`, PTX `.pragma "nounroll"`); 0 leaves the factor to
    /// LLVM. A runtime trip count gets a remainder loop.
    pub unroll: Option<u32>,
    /// `unroll_full=True`: unroll the loop completely (`full = true`). LLVM
    /// only does so when it can compute the trip count. Cannot be combined
    /// with `unroll` (CuTe DSL lets `unroll_full` win; this is an error).
    pub unroll_full: bool,
    /// `vectorize=True`: ask LLVM to vectorise the loop (`vectorize =
    /// <disable = false>`). Needs sm_100 or newer, as in CuTe DSL.
    pub vectorize: bool,
    /// `at_least_once=True`: the program promises that the loop runs at
    /// least once in every thread, so the compiler can drop the check
    /// whether to skip it (the MLIR assumes `lower < upper` with
    /// `llvm.intr.assume`). The simulator reports a thread for which it
    /// runs zero times, and the validator constant bounds that do.
    pub at_least_once: bool,
}

impl LoopOptions {
    /// `unroll=count`.
    pub fn unroll(count: u32) -> LoopOptions {
        LoopOptions { unroll: Some(count), ..LoopOptions::default() }
    }

    /// `unroll_full=True`.
    pub fn unroll_full() -> LoopOptions {
        LoopOptions { unroll_full: true, ..LoopOptions::default() }
    }

    /// Whether any hint becomes loop metadata.
    pub fn has_annotation(&self) -> bool {
        self.unroll.is_some() || self.unroll_full || self.vectorize
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    Const {
        result: ValueId,
        value: Literal,
    },
    Unary {
        result: ValueId,
        op: UnaryOp,
        x: ValueId,
    },
    Binary {
        result: ValueId,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    /// `a * b + c` with a single rounding.
    Fma {
        result: ValueId,
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },
    /// A floating-point instruction with explicit PTX modifiers: a rounding
    /// direction, the hardware approximation, flushing subnormal inputs and
    /// results to zero (`ftz`), NaN propagation (`nan`, for `min` and
    /// `max`) or `.xorsign.abs` (`xorsign_abs`, for Float32 `min` and
    /// `max`: the magnitude of the operands' absolute values, with the
    /// exclusive or of their signs). These are `cute.math`'s `approx=`,
    /// `ftz=`, `rounding=`, `full=` and `propagate_nan=` options and its
    /// `fastmath=True` approximations, and `cute.arch.fmax`'s and `fmin`'s
    /// `abs=`, `nan=` and `ftz=`.
    FloatInst {
        result: ValueId,
        func: FloatFn,
        args: Vec<ValueId>,
        mode: FloatMode,
        ftz: bool,
        nan: bool,
        xorsign_abs: bool,
    },
    /// A PTX integer or bit instruction ([`IntFn`]), with its immediates and
    /// modifiers: CuTe DSL's `cute.arch` integer helpers (`bfe`, `lop3`,
    /// `shf`, `prmt`, `mul_wide`, `mul24`, `add_sat_int`, ...). Pure.
    IntInst {
        result: ValueId,
        func: IntFn,
        args: Vec<ValueId>,
    },
    Cmp {
        result: ValueId,
        pred: CmpPred,
        lhs: ValueId,
        rhs: ValueId,
    },
    Select {
        result: ValueId,
        cond: ValueId,
        on_true: ValueId,
        on_false: ValueId,
    },
    /// Numeric conversion to the result's type.
    Cast {
        result: ValueId,
        x: ValueId,
    },
    /// Reinterpret the bits as the result's type (same width).
    Bitcast {
        result: ValueId,
        x: ValueId,
    },
    Special {
        result: ValueId,
        reg: SpecialReg,
    },
    /// Load `results.len()` contiguous elements of the results' type. `index`
    /// counts elements of that type from the start of the buffer. `attrs`
    /// give the ordering and cache hints (plain for tensor loads). With a
    /// `pred` (Boolean), a thread whose flag is false reads nothing: its
    /// results are `pass_thru` (one value per result), or undefined when
    /// `pass_thru` is empty.
    Load {
        results: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
        attrs: MemAttrs,
        pred: Option<ValueId>,
        pass_thru: Vec<ValueId>,
    },
    /// Store `values.len()` contiguous elements of the values' type. `index`
    /// counts elements of that type from the start of the buffer. `attrs`
    /// give the ordering and cache hints (plain for tensor stores). With a
    /// `pred` (Boolean), a thread whose flag is false writes nothing.
    Store {
        values: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
        attrs: MemAttrs,
        pred: Option<ValueId>,
    },
    /// `cp.async` of `elems` contiguous `elem`s from global to shared memory.
    /// Indices count `elem`s. When `pred` is false the destination is
    /// zero-filled instead.
    CpAsync {
        src: BufferId,
        src_index: ValueId,
        dst: BufferId,
        dst_index: ValueId,
        elem: DType,
        elems: u32,
        bypass_l1: bool,
        pred: Option<ValueId>,
    },
    CpAsyncCommitGroup,
    /// Wait until at most `pending` committed groups are still in flight.
    CpAsyncWaitGroup {
        pending: u32,
    },
    /// `ldmatrix.sync.aligned.{shape}.x{num}[.trans].shared.{format}`, the
    /// shape and element format given by `form`. Each lane passes the
    /// address of one 16-byte row. For [`LdMatrixForm::M8n8B16`] `index`
    /// counts 16-bit elements and `results` holds two 16-bit values per
    /// matrix, register-major; for the 8-bit forms `index` counts bytes and
    /// `results` holds the 32-bit destination registers (each four 8-bit
    /// values, the first in the low byte).
    LdMatrix {
        results: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
        num: u32,
        trans: bool,
        form: LdMatrixForm,
    },
    /// `stmatrix.sync.aligned.{shape}.x{num}[.trans].shared.{type}`: each
    /// lane passes the address of one 16-byte row of shared memory (lanes
    /// `8m` to `8m + 7` those of matrix `m`) and one 32-bit register per
    /// matrix, `values[m]`: for [`StMatrixForm::M8n8B16`] two 16-bit elements
    /// (the first in the low half) and `index` counts 16-bit elements; for
    /// [`StMatrixForm::M16n8B8`] four bytes and `index` counts bytes.
    StMatrix {
        values: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
        num: u32,
        trans: bool,
        form: StMatrixForm,
    },
    /// `setmaxnreg.{inc,dec}.sync.aligned.u32 count`: the warpgroup's
    /// per-thread register budget becomes `count` (24 to 256, a multiple of
    /// 8). All four warps of the warpgroup execute it together.
    SetMaxNReg {
        increase: bool,
        count: u32,
    },
    /// `mma.sync`, or with a sparse kind `mma.sp::ordered_metadata.sync`,
    /// whose `e` is the 32-bit metadata register (sparsity selector 0).
    Mma {
        kind: MmaKind,
        d: Vec<ValueId>,
        a: Vec<ValueId>,
        b: Vec<ValueId>,
        c: Vec<ValueId>,
        e: Option<ValueId>,
    },
    /// `bar.sync 0` (`__syncthreads`).
    Barrier,
    /// A named barrier: barrier `id` (0 to 15) completes when `threads`
    /// threads (a multiple of 32; all of the block when `None`) have arrived.
    NamedBarrier {
        kind: BarrierKind,
        id: ValueId,
        threads: Option<ValueId>,
    },
    /// `bar.warp.sync` (`__syncwarp`) over the lanes in `mask`.
    WarpBarrier {
        mask: ValueId,
    },
    /// `shfl.sync.{kind}.b32 result, value, offset, clamp, mask`: read the
    /// 32-bit `value` of another lane of the warp. `offset` is the source
    /// lane or lane offset (its bits 4:0), and `clamp` packs the segment mask
    /// (bits 12:8) and the clamp value (bits 4:0), CuTe DSL's
    /// `mask_and_clamp`.
    Shuffle {
        result: ValueId,
        kind: ShuffleKind,
        value: ValueId,
        offset: ValueId,
        mask: ValueId,
        clamp: ValueId,
    },
    /// `vote.sync` over the lanes in `mask`.
    Vote {
        result: ValueId,
        kind: VoteKind,
        pred: ValueId,
        mask: ValueId,
    },
    /// `match.{any,all}.sync.b{32,64}`: compare a 32- or 64-bit `value`
    /// across the lanes in `mask`, giving a mask of lanes.
    Match {
        result: ValueId,
        kind: MatchKind,
        value: ValueId,
        mask: ValueId,
    },
    /// `elect.sync`: true in exactly one of the lanes in `mask`.
    ElectOne {
        result: ValueId,
        mask: ValueId,
    },
    /// `activemask.b32`.
    ActiveMask {
        result: ValueId,
    },
    /// `value`, which the program promises is the same in every lane of the
    /// warp (CuTe DSL's `make_warp_uniform`): a compiler hint, not an
    /// instruction.
    WarpUniform {
        result: ValueId,
        value: ValueId,
    },
    /// Atomic read-modify-write (`atom{.sem}{.scope}{.space}.op.type`);
    /// `result` is the old value. `index` counts elements of the operand's
    /// type, or of the half-precision element of a packed type (whose
    /// operand is an Int32 holding two). `sem` is relaxed, acquire,
    /// release or acq_rel.
    Atomic {
        result: ValueId,
        kind: AtomicKind,
        ty: AtomicType,
        buffer: BufferId,
        index: ValueId,
        value: ValueId,
        compare: Option<ValueId>,
        sem: MemoryOrder,
        scope: MemoryScope,
        /// Distributed shared memory: the CTA rank, as for
        /// [`MemAttrs::peer`].
        peer: Option<ValueId>,
    },
    /// A reduction (`red{.sem}{.scope}{.space}.op.type`): an atomic
    /// read-modify-write without a result, addressed as [`Op::Atomic`] is.
    /// `sem` is relaxed or release.
    Red {
        kind: AtomicKind,
        ty: AtomicType,
        buffer: BufferId,
        index: ValueId,
        value: ValueId,
        sem: MemoryOrder,
        scope: MemoryScope,
        /// As for [`Op::Atomic`].
        peer: Option<ValueId>,
    },
    /// `mbarrier.init`: an mbarrier waiting for `count` arrivals per phase.
    MbarrierInit {
        buffer: BufferId,
        index: ValueId,
        count: ValueId,
    },
    /// `mbarrier.arrive`: `count` arrivals (1 when `None`), after raising the
    /// bytes of bulk copies the phase waits for by `expect_tx` when given
    /// (`.expect_tx`). With `peer`, on the mbarrier at the same offset in
    /// that CTA of the cluster (`mapa` + `.shared::cluster`). `relaxed` drops
    /// the release ordering.
    MbarrierArrive {
        buffer: BufferId,
        index: ValueId,
        count: Option<ValueId>,
        expect_tx: Option<ValueId>,
        peer: Option<ValueId>,
        relaxed: bool,
        scope: MemScopeKind,
    },
    /// `mbarrier.expect_tx`: make the current phase wait for `bytes` more
    /// bytes of bulk copies, without arriving.
    MbarrierExpectTx {
        buffer: BufferId,
        index: ValueId,
        bytes: ValueId,
        peer: Option<ValueId>,
        scope: MemScopeKind,
    },
    /// `mbarrier.try_wait.parity` / `mbarrier.test_wait.parity`: whether the
    /// phase with this parity is over.
    MbarrierWait {
        result: ValueId,
        buffer: BufferId,
        index: ValueId,
        parity: ValueId,
        kind: MbarrierWaitKind,
        /// `parity` is an i64 state token from an
        /// arrival (`mbarrier.{try,test}_wait`, no `.parity`).
        state: bool,
        /// The `suspendTimeHint` (i32, nanoseconds) of
        /// a `try_wait`.
        time_limit: Option<ValueId>,
    },
    /// `fence.mbarrier_init.release.cluster`: make earlier `mbarrier.init`s
    /// visible to the cluster.
    FenceMbarrierInit,
    /// `griddepcontrol.wait` (`launch_dependents` false): block until every prerequisite
    /// grid of a programmatic dependent launch has completed and its memory is visible; or
    /// `griddepcontrol.launch_dependents`: let the dependent grid be scheduled once every CTA
    /// has issued it or exited. Both are no-ops without programmatic stream serialization.
    GridDepControl {
        launch_dependents: bool,
    },
    /// `barrier.cluster.arrive{.relaxed}{.aligned}`: the thread arrives at
    /// its cluster's barrier, with release semantics at cluster scope unless
    /// `relaxed`. `aligned`: every thread of the warp executes it together.
    ClusterArrive {
        relaxed: bool,
        aligned: bool,
    },
    /// `barrier.cluster.wait{.aligned}`: wait until every thread of the
    /// cluster that has not exited has arrived in the thread's phase
    /// (acquire semantics at cluster scope).
    ClusterWait {
        aligned: bool,
    },
    /// `cp.async.bulk` of `bytes` bytes (a multiple of 16, both addresses
    /// 16-byte aligned) in `direction`, reporting completion as
    /// `completion` says. Indices count `elem`s. `cache_policy` is the
    /// 64-bit operand of `.L2::cache_hint` (G2S and S2G); `mcast_mask` (an
    /// i16) makes a G2S copy `.multicast::cluster`, landing at the same
    /// offset in each CTA of the mask and signalling the mbarrier at the
    /// same offset in each; `peer` (an i32 CTA rank, S2S only) is the CTA
    /// whose shared memory and mbarrier the copy addresses (`mapa`).
    CpAsyncBulk {
        src: BufferId,
        src_index: ValueId,
        dst: BufferId,
        dst_index: ValueId,
        elem: DType,
        bytes: ValueId,
        direction: BulkDirection,
        completion: BulkCompletion,
        cache_policy: Option<ValueId>,
        mcast_mask: Option<ValueId>,
        peer: Option<ValueId>,
        /// `.cp_mask` with this i16 mask (S2G only, sm_100): bit `i` enables byte `i` of each
        /// 16-byte chunk.
        byte_mask: Option<ValueId>,
    },
    /// `cp.async.bulk.commit_group`: the thread's bulk copies issued since
    /// its last commit become a bulk async-group.
    CpAsyncBulkCommitGroup,
    /// `cp.async.bulk.wait_group{.read} pending`: wait until at most
    /// `pending` of the thread's most recent bulk async-groups are pending;
    /// with `read`, only until the others have read their sources.
    CpAsyncBulkWaitGroup {
        pending: u32,
        read: bool,
    },
    /// A TMA tile load:
    /// `cp.async.bulk.tensor.{N}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes[.L2::cache_hint]`
    /// of the box at `coords` (one i32 per dimension of the descriptor, in its
    /// order) of the tensor map at the generic address `desc` (an i64) into
    /// shared memory at element `dst_index` of `dst` (its unswizzled start;
    /// the descriptor's swizzle places the bytes), its bytes counting down
    /// the transaction count of the mbarrier at element `mbar_index` of
    /// `mbar`. `elem` is the element type (the index counts it) and
    /// `box_bytes` the box's bytes the atom was built for.
    TmaLoad {
        dst: BufferId,
        dst_index: ValueId,
        elem: DType,
        desc: ValueId,
        coords: Vec<ValueId>,
        mbar: BufferId,
        mbar_index: ValueId,
        cache_policy: Option<ValueId>,
        box_bytes: u32,
        /// `.multicast::cluster` with this i16 CTA mask: the box lands at the same offset in every CTA
        /// of the mask, and counts down the mbarrier at the same offset in
        /// each.
        mcast_mask: Option<ValueId>,
        /// The access mode: `.tile` unless an im2col or gather4 atom.
        mode: TmaMode,
        /// `.override::*` operands.
        overrides: Option<TmaOverride>,
        /// `.cta_group::2` (a load for the CTA
        /// pair of a 2-CTA tcgen05 MMA): the box lands in the issuing CTA
        /// (or the CTAs of the mask), and its bytes count down the mbarrier
        /// at the same offset in the even CTA of each destination's pair
        /// (the address with its peer bit cleared, CUTLASS's
        /// `Sm100MmaPeerBitMask`).
        two_cta: bool,
    },
    // ---- TMA stores and cluster copies ----
    /// A TMA tile store:
    /// `cp.async.bulk.tensor.{N}d.global.shared::cta.tile.bulk_group[.L2::cache_hint]`
    /// of the box at element `src_index` of `src` (its unswizzled start;
    /// the descriptor's swizzle says where each byte is read) to the box at
    /// `coords` of the tensor map at `desc` (out-of-bounds elements are not
    /// written), tracked by the thread's bulk async-groups. With `reduce`,
    /// `cp.reduce.async.bulk.tensor` combining the box into global memory
    /// with that operation.
    TmaStore {
        src: BufferId,
        src_index: ValueId,
        elem: DType,
        desc: ValueId,
        coords: Vec<ValueId>,
        reduce: Option<crate::cpasync::ReductionKind>,
        cache_policy: Option<ValueId>,
        box_bytes: u32,
        /// The access mode: `.tile`, `.im2col_no_offs` or `.tile::scatter4`.
        mode: TmaMode,
        /// `.override::*` operands.
        overrides: Option<TmaOverride>,
    },
    /// `cp.async.bulk.prefetch.tensor.{N}d.L2.global{mode} [desc, coords]`:
    /// bring the box into L2. A hint. `mode` is `.tile`, `.im2col` or
    /// `.tile::gather4`.
    TmaPrefetch {
        desc: ValueId,
        coords: Vec<ValueId>,
        mode: TmaMode,
    },
    /// `prefetch.tensormap [desc]`: bring the tensor map at the generic
    /// address `desc` into the cache. A hint.
    PrefetchTensormap {
        desc: ValueId,
    },
    // In-kernel tensor-map updates.
    /// `tensormap.replace.tile.{field}{.ss}.b1024.{b32|b64} [addr], {ord,}
    /// value`: replace one field of the 128-byte tensor map at `addr` (an
    /// `i64` global or generic address, or an `i32` shared one, as `space`
    /// says). A generic-proxy write of the whole object.
    TensormapReplace {
        addr: ValueId,
        space: crate::tensor::AddressSpace,
        field: crate::nvgpu::tensormap::TensormapField,
        value: ValueId,
    },
    /// `fence.proxy.tensormap::generic.release.gpu`: the thread's earlier
    /// generic writes of tensor maps become visible to the tensormap proxy
    /// of threads that acquire them.
    FenceTensormapRelease,
    /// `fence.proxy.tensormap::generic.acquire.gpu [addr], 128`: the
    /// thread's later tensormap-proxy reads of the tensor map at the
    /// generic address `addr` see the released writes.
    FenceTensormapAcquire {
        addr: ValueId,
    },
    /// `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned
    /// [dst], [src], 128`: copy the tensor map at the shared address `src`
    /// (`i32`) to the global address `dst` (`i64`) and release it to the
    /// tensormap proxy. Warp-collective.
    TensormapCpFenceproxy {
        dst: ValueId,
        src: ValueId,
    },
    /// `cp.async.bulk.prefetch.L2.global [a], bytes`: bring `bytes` bytes
    /// (a multiple of 16) from element `index` of a global buffer into L2.
    BulkPrefetch {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        bytes: ValueId,
    },
    /// `st.async.shared::cluster.mbarrier::complete_tx::bytes{.v2,.v4}.b32`:
    /// store `values` (one, two or four i32s) at element `index` of `buffer`
    /// in CTA `peer` of the cluster, counting their bytes down the mbarrier
    /// at element `mbar_index` of `mbar` in that CTA (both addresses mapped
    /// with `mapa`).
    StAsync {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        values: Vec<ValueId>,
        mbar: BufferId,
        mbar_index: ValueId,
        peer: ValueId,
    },
    /// `fence.proxy.{kind}`: orders this thread's accesses through one proxy
    /// against its later accesses through another.
    FenceProxy {
        kind: ProxyKind,
    },
    /// `fence.acq_rel.{scope}`, or with `sc` `fence.sc.{scope}` (`membar`).
    Fence {
        scope: FenceScope,
        sc: bool,
    },
    /// `vprintf`.
    Printf {
        format: String,
        args: Vec<ValueId>,
    },
    /// A device assertion (`cute::testing::assert_`, CuTe DSL's
    /// `cute.testing.assert_`): `cf.assert` when assertions are enabled
    /// (`__assertfail` with the message and the op's file and line, then
    /// the thread exits), nothing otherwise.
    Assert {
        cond: ValueId,
        message: String,
    },
    /// Hand-written PTX, as LLVM inline assembly. `template` names the
    /// operands `$0, $1, ...`: the outputs, then the inputs (`$$` is a
    /// literal `$`). `constraints` is LLVM's constraint string: an entry per
    /// output (`=r`), then per input (`r`, or an output's number for an input
    /// tied to that output, which makes the output a read-write operand),
    /// then clobbers (`~{memory}`). With `predicated`, the last input is a
    /// Boolean guarding the template, which is one instruction (`@$n ...`),
    /// and where it is false the outputs tied to an input keep that input's
    /// value and the others are 0. The simulator can only run it if `sim` is
    /// given.
    InlinePtx {
        template: String,
        inputs: Vec<AsmInput>,
        outputs: Vec<ValueId>,
        constraints: String,
        predicated: bool,
        side_effects: bool,
        sim: Option<AsmSim>,
    },
    If {
        cond: ValueId,
        then_region: Region,
        else_region: Region,
        results: Vec<ValueId>,
    },
    /// `for iv in (lower..upper).step_by(step)` with loop-carried values:
    /// `args` are the body's views of the carried values, initialised from
    /// `init` and updated from the body's yields; `results` are the final
    /// values. `opts` holds the unroll and vectorize hints.
    For {
        iv: ValueId,
        lower: ValueId,
        upper: ValueId,
        step: ValueId,
        init: Vec<ValueId>,
        args: Vec<ValueId>,
        body: Region,
        results: Vec<ValueId>,
        opts: LoopOptions,
    },
    /// `while cond(args) { args = body(args) }`. `cond` sees `cond_args` and
    /// yields one boolean; `body` sees `body_args` and yields the next state.
    /// `results` are the state when the condition first fails.
    While {
        init: Vec<ValueId>,
        cond_args: Vec<ValueId>,
        cond: Region,
        body_args: Vec<ValueId>,
        body: Region,
        results: Vec<ValueId>,
    },
    /// `redux.sync`: reduce `value` (Int32, Uint32 or Float32) across the
    /// lanes in `mask`; every member lane gets the result. `abs` and `nan`
    /// are the Float32 modifiers `.abs` and `.NaN`.
    Redux {
        result: ValueId,
        kind: ReduxKind,
        value: ValueId,
        mask: ValueId,
        abs: bool,
        nan: bool,
    },
    /// An extended-precision instruction ([`CarryFn`]): CuTe DSL's
    /// `add_cc`, `addc`, `sub_cc`, `subc`, `mad_cc` and `madc`, and the
    /// carry-in-and-out forms. It reads or writes the carry flag, state that
    /// no SSA value shows, so it is never removed, merged with another or
    /// moved past one: the builder neither CSEs nor deletes it, and the
    /// backend emits it as inline PTX with side effects. A carry-in needs a
    /// carry-out before it in the same region with no control flow between
    /// them (the validator checks).
    CarryInst {
        result: ValueId,
        func: CarryFn,
        args: Vec<ValueId>,
    },
    /// An MLIR operation written by hand (CuTe DSL's raw `cutlass._mlir`
    /// builders inside a `dsl_user_op`), for operations cute-rs lacks:
    /// `text` is the operation without its results, with `%0, %1, ...`
    /// standing for `operands`, and `results` get its results. An address
    /// operand is an LLVM pointer: `!llvm.ptr<1>` into global memory,
    /// `!llvm.ptr<3>` into shared memory and a generic `!llvm.ptr` into a
    /// local buffer. The simulator can only run it if `sim` is given.
    /// Without side effects, it is removed when its results are unused.
    MlirOp {
        text: String,
        operands: Vec<AsmInput>,
        results: Vec<ValueId>,
        side_effects: bool,
        sim: Option<AsmSim>,
    },
    /// `cp.async.mbarrier.arrive{.noinc}.shared.b64`: an arrive-on the
    /// mbarrier once every `cp.async` this thread issued before it has
    /// completed. Without `noinc` the phase's pending count is raised by one
    /// first, so the asynchronous arrival changes nothing on balance; with
    /// it, the mbarrier's count must include the arrival.
    CpAsyncMbarrierArrive {
        buffer: BufferId,
        index: ValueId,
        noinc: bool,
    },
    /// `prefetch{.global}.level [a]`, `prefetch.global.L2::evict_*` (with
    /// `evict`) or `prefetchu.L1` (`uniform`): bring the cache line holding
    /// element `index` (counting `elem`s) of a global or local buffer into a
    /// cache, when `pred` (if given) holds. A hint: nothing is read.
    Prefetch {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        level: PrefetchCacheLevel,
        evict: Option<CacheEvictionPriority>,
        uniform: bool,
        pred: Option<ValueId>,
    },
    /// The bits of `parts` (two or more values of one type) concatenated
    /// into one value of the result's type, part 0 in the low bits: a
    /// bitcast that changes the number of values. Pure.
    Pack {
        result: ValueId,
        parts: Vec<ValueId>,
    },
    /// The bits of `x` split into `results` (two or more values of one
    /// type), result 0 from the low bits: the inverse of [`Op::Pack`]. Pure.
    Unpack {
        results: Vec<ValueId>,
        x: ValueId,
    },
    /// A PTX half-precision instruction ([`HalfFn`]) on 16-bit floats or on
    /// pairs of them packed into Int32s ([`HalfType`]), with CuTe DSL's
    /// modifiers. With a `predicate` the instruction runs only where it is
    /// true (`@p`); elsewhere the result is the first operand (CuTe DSL
    /// leaves it undefined). Pure.
    HalfInst {
        result: ValueId,
        func: HalfFn,
        ty: HalfType,
        args: Vec<ValueId>,
        mods: HalfModifiers,
        predicate: Option<ValueId>,
    },
    /// A PTX conversion with modifiers ([`CvtFn`]): CuTe DSL's narrow
    /// conversion helpers. Pure.
    Cvt {
        result: ValueId,
        func: CvtFn,
        args: Vec<ValueId>,
    },
    /// `x` itself, which the program promises is a multiple of `divby` and,
    /// with `pow2`, a positive power of two (CuTe DSL's `cute.assume`). The
    /// result carries those facts for the builder (which uses them to prove
    /// accesses aligned); breaking the promise is undefined behaviour, which
    /// the simulator reports. The MLIR backend passes a power-of-two
    /// `divby` on to LLVM as `llvm.intr.assume`, and nothing else. It is
    /// pure, so an unused one is removed: with no use, nothing relies on the
    /// promise.
    Assume {
        result: ValueId,
        x: ValueId,
        divby: u64,
        pow2: bool,
    },
    /// The address of element `index` (counting `elem`s) of `buffer`, a
    /// global, shared or per-thread buffer, as an integer in the buffer's
    /// own address space (CuTe DSL's `cute.ptrtoint`): an `i64` global
    /// address, or the `i32` offset in the block's shared-memory window or
    /// the thread's local window.
    PtrToInt {
        result: ValueId,
        buffer: BufferId,
        index: ValueId,
        elem: DType,
    },
    /// The integer address `x` of space `from` as one of space `to`, one of
    /// them generic (`cvta` and `cvta.to`, LLVM's `addrspacecast`): `i64`
    /// for generic and global addresses, `i32` for shared and local ones.
    /// A generic address outside the target space's window is undefined.
    CvtAddr {
        result: ValueId,
        x: ValueId,
        from: crate::tensor::AddressSpace,
        to: crate::tensor::AddressSpace,
    },
    // WGMMA.
    /// `wgmma.mma_async.sync.aligned.m64n{N}k{K}.{acc}.{a}.{b}` (see
    /// [`WgmmaKind`]): `d = a * b + (scale_d ? c : 0)` over a warpgroup, B
    /// read from shared memory through the 64-bit matrix descriptor
    /// `desc_b` (U64), `scale_d` a Boolean (the atom's `ACCUMULATE` field).
    /// `a` is A's 64-bit descriptor (one U64) for a shared-memory A,
    /// or, for the register-sourced form ([`WgmmaKind::a_regs`]), this
    /// thread's A fragment: 8 16-bit or 16 8-bit elements in the PTX ISA's
    /// A-fragment order, which the instruction takes as four 32-bit
    /// registers. `c` and `d` are each thread's `N / 2` accumulators in the
    /// PTX ISA's D-fragment order. The operation is asynchronous: `d` holds
    /// its results only once a [`Op::WgmmaWait`] covers it, and may
    /// meanwhile flow only into other wgmmas' `c`, waits, operand fences and
    /// loop carries.
    Wgmma {
        kind: WgmmaKind,
        d: Vec<ValueId>,
        c: Vec<ValueId>,
        a: Vec<ValueId>,
        desc_b: ValueId,
        scale_d: ValueId,
    },
    /// `wgmma.fence.sync.aligned`, after an operand fence on `inputs` (the
    /// accumulators of the warpgroup's MMAs): `results[i]` is `inputs[i]`,
    /// pinned in registers before the fence, so register writes cannot
    /// sink past it.
    WgmmaFence {
        results: Vec<ValueId>,
        inputs: Vec<ValueId>,
    },
    /// `wgmma.commit_group.sync.aligned`.
    WgmmaCommitGroup,
    /// `wgmma.wait_group.sync.aligned pending`, then an operand fence on
    /// `inputs`: `results[i]` is `inputs[i]` read after the wait.
    /// `complete[i]` says whether the builder knows `inputs[i]` to be the
    /// result of a wgmma in a group the wait completes (always so when
    /// `pending` is 0).
    WgmmaWait {
        pending: u32,
        results: Vec<ValueId>,
        inputs: Vec<ValueId>,
        complete: Vec<bool>,
    },
    // External device-function calls and multimem.
    /// A call of external device function `callee` ([`KernelIr::externs`]):
    /// CuTe DSL's `cute.extern` / `cute.ffi` call, `func.call` in MLIR.
    /// `args` follow the declaration's parameters: values, and for pointer
    /// parameters the address of a buffer element. `body`, when present, is
    /// what the simulator runs in place of the call (the callee traced in
    /// the caller, as a [`DeviceFunction`](crate::ffi::DeviceFunction) or an
    /// extern's DSL simulation), yielding `results`; the backend emits only
    /// the call.
    Call {
        callee: ExternId,
        args: Vec<AsmInput>,
        results: Vec<ValueId>,
        body: Option<Region>,
    },
    /// A `multimem` instruction on a multicast buffer (CuTe DSL's
    /// `cutlass.utils.distributed`): see [`MultimemOp`]. `values` are the
    /// registers stored (`St`), `results` the registers loaded (`LdReduce`).
    Multimem {
        op: MultimemOp,
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        values: Vec<ValueId>,
        results: Vec<ValueId>,
    },
    // Tensor memory.
    /// A `tcgen05` instruction (sm_100), one family for all of them: see
    /// [`Tcgen05Inst`] for what `buffer` and `args` hold. `two_cta` is
    /// `.cta_group::2` (a CTA pair) rather than `.cta_group::1`;
    /// `exclusive` is the sm_107 `.exclusive` form of alloc and dealloc.
    Tcgen05 {
        inst: Tcgen05Inst,
        two_cta: bool,
        exclusive: bool,
        buffer: Option<BufferId>,
        args: Vec<ValueId>,
    },
    // Loads and stores between tensor memory and registers.
    /// `tcgen05.ld.sync.aligned.<shape>.x<num>{.pack::16b}.b32 {results},
    /// [taddr]{, half_split}` (sm_100a): `results` are the elements the
    /// 32-bit registers hold (`shape.registers(num)` registers, each of `32 /
    /// bits` elements of one type of 8, 16 or 32 bits, the first in the low
    /// bits), `taddr` the warp-uniform TMEM address (i32). Warp-collective
    /// and asynchronous: the registers are undefined until the warp's
    /// `tcgen05.wait::ld`. The `.16x32bx2` half split is
    /// [`TmemShape::half_split`].
    ///
    /// With `red`, the `tcgen05.ld.red` form (PTX ISA
    /// 8.8: sm_110a, the sm_103f and sm_110f families), which also reduces
    /// each thread's loaded values to one Float32 ([`TmemLdRed`]); the
    /// results are then Float32 values and `pack` is false. The reduction
    /// is as pending as the data.
    ///
    /// With `sp`, the `.spcompress` form (PTX ISA 9.4:
    /// sm_107a): `results` are only the compressed data ([`TmemLdSpCompress`]).
    Tcgen05Ld {
        results: Vec<ValueId>,
        taddr: ValueId,
        shape: TmemShape,
        num: u32,
        pack: bool,
        red: Option<TmemLdRed>,
        sp: Option<TmemLdSpCompress>,
    },
    /// `tcgen05.st.sync.aligned.<shape>.x<num>{.unpack::16b}.b32 [taddr]{,
    /// half_split}, {values}` (sm_100a): `values` are the registers'
    /// elements, as for [`Op::Tcgen05Ld`]. Warp-collective and asynchronous
    /// until the warp's `tcgen05.wait::st`.
    Tcgen05St {
        values: Vec<ValueId>,
        taddr: ValueId,
        shape: TmemShape,
        num: u32,
        unpack: bool,
    },
    // Tcgen05 MMAs and their commits.
    /// One warp-level issue of `tcgen05.mma.cta_group::1.kind::<kind>
    /// [taddr], desc_a, desc_b, idesc, enable_input_d` (sm_100a): executed by
    /// a converged warp, whose lane `elect` (the Bool of an `elect.sync`)
    /// issues the instruction. `taddr` is the TMEM address of D (i32),
    /// `desc_a` and `desc_b` the UMMA shared-memory descriptors (u64), and
    /// `enable_input_d` (Bool) whether D accumulates (`false`: D = A B). The
    /// instruction descriptor is `kind.idesc()`, fixed when the kernel is
    /// built.
    ///
    /// With `kind.a_tmem`, `desc_a` is A's TMEM address
    /// (i32, the `[a-tmem]` form). `idesc`, when given, is the instruction
    /// descriptor as a runtime u32 (the constant bits OR the runtime
    /// negate bits); `disable_lane`, when not empty, the
    /// `{disable-output-lane}` words (i32, `kind.disable_lane_words()`).
    Tcgen05Mma {
        kind: UmmaKind,
        taddr: ValueId,
        desc_a: ValueId,
        desc_b: ValueId,
        enable_input_d: ValueId,
        elect: ValueId,
        idesc: Option<ValueId>,
        disable_lane: Vec<ValueId>,
        /// The TMEM addresses (i32) of the scale
        /// factors of A and B (`[sfa-tmem]`, `[sfb-tmem]`) of a block-scaled
        /// kind (`kind.scale`); their scale-factor IDs (the byte of each
        /// 32-bit column the first factor is in) are in `idesc`.
        sf: Option<(ValueId, ValueId)>,
        /// The TMEM address (i32) of the sparsity metadata
        /// (`[sp-meta-tmem]`) of a sparse kind (`kind.sparse`); its
        /// selector is in `idesc`'s bits 0-1.
        meta: Option<ValueId>,
        /// The `zero-column-mask-desc` (u64) of a
        /// `.ws` MMA (`kind.prims.ws`).
        zero_col_mask: Option<ValueId>,
    },
    /// `tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one.shared::cluster{.multicast::cluster}.b64
    /// [mbar]{, mask}` (sm_100a): one arrival on the mbarrier at element
    /// `index` of `buffer` once the executing thread's earlier tcgen05
    /// MMAs have completed; with `mask` (an i16 of CTA ranks) on the
    /// mbarrier at the same offset of each CTA it names. Per thread.
    Tcgen05Commit {
        buffer: BufferId,
        index: ValueId,
        mask: Option<ValueId>,
        two_cta: bool,
    },
    // Shared-memory-to-TMEM copies.
    /// One warp-level issue of `tcgen05.cp.cta_group::{1,2}.<shape> [taddr],
    /// s-desc` (sm_100a): executed by a converged warp, whose lane `elect`
    /// (the Bool of an `elect.sync`) issues the instruction. `taddr` is the
    /// destination TMEM address (i32), `desc` the shared-memory matrix
    /// descriptor of the source (u64, K-major core matrices). Asynchronous
    /// on the CTA's tensor-core pipe, in order with its `tcgen05.mma`s, and
    /// completed through `tcgen05.commit`. With `two_cta` the pair's even
    /// CTA issues it and each CTA copies from its own shared memory into its
    /// own TMEM.
    Tcgen05Cp {
        shape: TmemCpShape,
        two_cta: bool,
        taddr: ValueId,
        desc: ValueId,
        elect: ValueId,
    },
    // SM120 warp-level block-scaled mma.sync.
    /// `mma.sync.aligned.<shape>.row.col.kind::<kind>.block_scale.scale_vec::<n>X.f32.<a>.<b>.f32.<sf>
    /// d, a, b, c, sfa, {0, 0}, sfb, {0, 0}` (sm_120a): executed by a converged
    /// warp. `a` and `b` are the operand registers (u32; see
    /// [`BlockScaledMmaKind::operand_registers`] for how values are packed),
    /// `c` and `d` the four f32 accumulators, `sfa` and `sfb` the scale-factor
    /// registers (u32, the scale of K-block `j` in byte `j`). Byte-id and
    /// thread-id are 0, as CUTLASS passes them.
    MmaBlockScaled {
        kind: BlockScaledMmaKind,
        d: Vec<ValueId>,
        a: Vec<ValueId>,
        b: Vec<ValueId>,
        c: Vec<ValueId>,
        sfa: ValueId,
        sfb: ValueId,
    },
    // SM120 warp-level sparse block-scaled mma.
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k{64,128}.row.col.kind::<kind>.block_scale.scale_vec::<n>X.f32.<a>.<b>.f32.<sf>
    /// d, a, b, c, e, 0, sfa, {bid-a, tid-a}, sfb, {bid-b, tid-b}`
    /// (sm_120a): executed by a converged warp. `a` (4 u32 registers, the
    /// compressed half of K), `b` (4), `c` and `d` (4 f32), the metadata
    /// register `e` (sparsity selector 0) and the scale-factor registers
    /// `sfa` and `sfb` (u32); the byte and thread ids are in `kind`.
    MmaSpBlockScaled {
        kind: SpBlockScaledMmaKind,
        d: Vec<ValueId>,
        a: Vec<ValueId>,
        b: Vec<ValueId>,
        c: Vec<ValueId>,
        e: ValueId,
        sfa: ValueId,
        sfb: ValueId,
    },
    // Cluster launch control.
    /// `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{.multicast::cluster::all}.b128
    /// [resp], [mbar]` (sm_100, PTX 8.6; the multicast form sm_100a): take a
    /// still-queued cluster of the grid off the launch queue. The 16-byte
    /// response lands asynchronously at element `index` of `buffer` (of type
    /// `elem`), completing 16 bytes of tx on the mbarrier at element
    /// `mbar_index` of `mbar`; with `multicast`, at the same offsets in every
    /// CTA of the cluster. Per thread.
    ClcTryCancel {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        mbar: BufferId,
        mbar_index: ValueId,
        multicast: bool,
    },
    /// A 16-byte shared-memory load of a try_cancel response at element
    /// `index` of `buffer`, decoded with
    /// `clusterlaunchcontrol.query_cancel`: `results` are the first CTA's x,
    /// y and z and whether a cluster was cancelled (1 or 0), all i32.
    ClcQueryCancel {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        results: [ValueId; 4],
    },
    /// `barrier.cta.red.{popc.u32,and.pred,or.pred}.aligned id{, count}, pred`
    /// (`bar.red`): a `bar.sync` at named barrier `id` that also reduces the
    /// participating threads' predicates `pred` (Bool) and returns the
    /// result to each of them (`result`: U32 for popc, Bool otherwise).
    BarrierRed {
        result: ValueId,
        kind: BarrierRedKind,
        pred: ValueId,
        id: ValueId,
        threads: Option<ValueId>,
    },
    /// The mbarrier operations of [`MbarrierExtKind`] on the mbarrier at
    /// element `index` of shared buffer `buffer`: `value` is the count or the
    /// bytes of the kinds that take one, `result` (i64) the state token of
    /// those that return one.
    MbarrierExt {
        kind: MbarrierExtKind,
        result: Option<ValueId>,
        buffer: BufferId,
        index: ValueId,
        value: Option<ValueId>,
    },
    /// A per-thread marker or hint ([`PrimMarker`]).
    PrimMarker {
        marker: PrimMarker,
    },
    /// `exit`: the executing threads stop.
    Exit,
    /// `st.bulk.weak.shared::cta [addr], size, 0` (sm_100): zero `size` (i64)
    /// bytes of shared memory from element `index` (of type `elem`) of `buffer`.
    StBulk {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        size: ValueId,
    },
    /// `tcgen05.shift.cta_group::{1,2}.down [taddr]` (sm_100a): shift the rows
    /// of the 32-lane subpartitions at TMEM address `taddr` (i32) down by one
    /// lane, the last row of each staying. Issued by one thread.
    Tcgen05Shift {
        taddr: ValueId,
        two_cta: bool,
    },
}

impl Op {
    /// Values defined by this operation (block arguments of nested regions
    /// excluded).
    pub fn results(&self) -> Vec<ValueId> {
        match self {
            Op::Const { result, .. }
            | Op::Unary { result, .. }
            | Op::Binary { result, .. }
            | Op::Fma { result, .. }
            | Op::FloatInst { result, .. }
            | Op::IntInst { result, .. }
            | Op::Cmp { result, .. }
            | Op::Select { result, .. }
            | Op::Cast { result, .. }
            | Op::Bitcast { result, .. }
            | Op::Special { result, .. }
            | Op::Shuffle { result, .. }
            | Op::Vote { result, .. }
            | Op::Match { result, .. }
            | Op::ElectOne { result, .. }
            | Op::ActiveMask { result }
            | Op::WarpUniform { result, .. }
            | Op::MbarrierWait { result, .. }
            | Op::BarrierRed { result, .. }
            | Op::Atomic { result, .. }
            | Op::Redux { result, .. }
            | Op::Pack { result, .. }
            | Op::HalfInst { result, .. }
            | Op::Cvt { result, .. } => vec![*result],
            Op::Load { results, .. } | Op::LdMatrix { results, .. } | Op::Unpack { results, .. } => results.clone(),
            Op::Mma { d, .. } => d.clone(),
            Op::InlinePtx { outputs, .. } => outputs.clone(),
            Op::If { results, .. } | Op::For { results, .. } | Op::While { results, .. } => results.clone(),
            Op::Call { results, .. } | Op::Multimem { results, .. } => results.clone(),
            Op::Store { .. }
            | Op::CpAsync { .. }
            | Op::CpAsyncCommitGroup
            | Op::CpAsyncWaitGroup { .. }
            | Op::Barrier
            | Op::NamedBarrier { .. }
            | Op::WarpBarrier { .. }
            | Op::MbarrierInit { .. }
            | Op::MbarrierArrive { .. }
            | Op::MbarrierExpectTx { .. }
            | Op::FenceMbarrierInit
            | Op::GridDepControl { .. }
            | Op::ClusterArrive { .. }
            | Op::ClusterWait { .. }
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
            | Op::Fence { .. }
            | Op::Printf { .. }
            | Op::Assert { .. }
            | Op::CpAsyncMbarrierArrive { .. }
            | Op::Prefetch { .. }
            | Op::StMatrix { .. }
            | Op::SetMaxNReg { .. }
            | Op::Red { .. } => Vec::new(),
            Op::CarryInst { result, .. } => vec![*result],
            Op::MlirOp { results, .. } => results.clone(),
            Op::Assume { result, .. } | Op::PtrToInt { result, .. } | Op::CvtAddr { result, .. } => vec![*result],
            Op::Wgmma { d, .. } => d.clone(),
            Op::WgmmaFence { results, .. } | Op::WgmmaWait { results, .. } => results.clone(),
            Op::WgmmaCommitGroup
            | Op::Tcgen05 { .. }
            | Op::Tcgen05St { .. }
            | Op::Tcgen05Mma { .. }
            | Op::Tcgen05Commit { .. }
            | Op::Tcgen05Cp { .. }
            | Op::ClcTryCancel { .. } => Vec::new(),
            Op::ClcQueryCancel { results, .. } => results.to_vec(),
            Op::Tcgen05Ld { results, red, sp, .. } => results
                .iter()
                .copied()
                .chain(sp.iter().flat_map(|s| s.mdata.iter().copied()))
                .chain(red.map(|r| r.result))
                .collect(),
            Op::MmaBlockScaled { d, .. } => d.clone(),
            Op::MmaSpBlockScaled { d, .. } => d.clone(),
            Op::MbarrierExt { result, .. } => result.iter().copied().collect(),
            Op::PrimMarker { .. } | Op::Exit | Op::StBulk { .. } | Op::Tcgen05Shift { .. } => Vec::new(),
        }
    }

    /// Values read by this operation itself (not by nested regions).
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Op::Const { .. }
            | Op::Special { .. }
            | Op::ActiveMask { .. }
            | Op::Fence { .. }
            | Op::FenceProxy { .. }
            | Op::CpAsyncBulkCommitGroup
            | Op::CpAsyncBulkWaitGroup { .. }
            | Op::ClusterArrive { .. }
            | Op::ClusterWait { .. }
            | Op::GridDepControl { .. }
            | Op::FenceMbarrierInit => {
                vec![]
            }
            Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } | Op::WarpUniform { value: x, .. } => {
                vec![*x]
            }
            Op::Binary { lhs, rhs, .. } | Op::Cmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::Fma { a, b, c, .. } => vec![*a, *b, *c],
            Op::FloatInst { args, .. } | Op::IntInst { args, .. } => args.clone(),
            Op::Select { cond, on_true, on_false, .. } => vec![*cond, *on_true, *on_false],
            Op::Load { index, attrs, pred, pass_thru, .. } => {
                let mut v = vec![*index];
                v.extend(attrs.cache_policy);
                v.extend(attrs.peer);
                v.extend(pred);
                v.extend(pass_thru);
                v
            }
            Op::LdMatrix { index, .. } => vec![*index],
            Op::Store { values, index, attrs, pred, .. } => {
                let mut v = values.clone();
                v.push(*index);
                v.extend(attrs.cache_policy);
                v.extend(attrs.peer);
                v.extend(pred);
                v
            }
            Op::CpAsync { src_index, dst_index, pred, .. } => {
                let mut v = vec![*src_index, *dst_index];
                v.extend(pred);
                v
            }
            Op::CpAsyncCommitGroup | Op::CpAsyncWaitGroup { .. } | Op::Barrier | Op::SetMaxNReg { .. } => vec![],
            Op::StMatrix { values, index, .. } => {
                let mut v = values.clone();
                v.push(*index);
                v
            }
            Op::NamedBarrier { id, threads, .. } => {
                let mut v = vec![*id];
                v.extend(threads);
                v
            }
            Op::WarpBarrier { mask } | Op::ElectOne { mask, .. } => vec![*mask],
            Op::Shuffle { value, offset, mask, clamp, .. } => vec![*value, *offset, *mask, *clamp],
            Op::Vote { pred, mask, .. } => vec![*pred, *mask],
            Op::Match { value, mask, .. } | Op::Redux { value, mask, .. } => vec![*value, *mask],
            Op::Atomic { index, value, compare, peer, .. } => {
                let mut v = vec![*index, *value];
                v.extend(compare);
                v.extend(peer);
                v
            }
            Op::Red { index, value, peer, .. } => {
                let mut v = vec![*index, *value];
                v.extend(peer);
                v
            }
            Op::MbarrierInit { index, count, .. } => vec![*index, *count],
            Op::MbarrierArrive { index, count, expect_tx, peer, .. } => {
                let mut v = vec![*index];
                v.extend(count);
                v.extend(expect_tx);
                v.extend(peer);
                v
            }
            Op::MbarrierExpectTx { index, bytes, peer, .. } => {
                let mut v = vec![*index, *bytes];
                v.extend(peer);
                v
            }
            Op::MbarrierWait { index, parity, time_limit, .. } => {
                let mut v = vec![*index, *parity];
                v.extend(time_limit);
                v
            }
            Op::BarrierRed { pred, id, threads, .. } => {
                let mut v = vec![*pred, *id];
                v.extend(threads);
                v
            }
            Op::MbarrierExt { index, value, .. } => {
                let mut v = vec![*index];
                v.extend(value);
                v
            }
            Op::PrimMarker { marker: PrimMarker::Nanosleep(t) } => vec![*t],
            Op::PrimMarker { .. } | Op::Exit => vec![],
            Op::StBulk { index, size, .. } => vec![*index, *size],
            Op::Tcgen05Shift { taddr, .. } => vec![*taddr],
            Op::CpAsyncBulk {
                src_index,
                dst_index,
                bytes,
                completion,
                cache_policy,
                mcast_mask,
                peer,
                byte_mask,
                ..
            } => {
                let mut v = vec![*src_index, *dst_index, *bytes];
                if let BulkCompletion::Mbarrier { index, .. } = completion {
                    v.push(*index);
                }
                v.extend(cache_policy);
                v.extend(mcast_mask);
                v.extend(peer);
                v.extend(byte_mask);
                v
            }
            Op::TmaLoad { dst_index, desc, coords, mbar_index, cache_policy, mcast_mask, mode, overrides, .. } => {
                let mut v = vec![*dst_index, *desc];
                v.extend(coords);
                v.push(*mbar_index);
                v.extend(cache_policy);
                v.extend(mcast_mask);
                v.extend(mode.offsets());
                v.extend(overrides.iter().flat_map(TmaOverride::operands));
                v
            }
            Op::TmaStore { src_index, desc, coords, cache_policy, overrides, .. } => {
                let mut v = vec![*src_index, *desc];
                v.extend(coords);
                v.extend(cache_policy);
                v.extend(overrides.iter().flat_map(TmaOverride::operands));
                v
            }
            Op::TmaPrefetch { desc, coords, mode } => {
                let mut v = vec![*desc];
                v.extend(coords);
                v.extend(mode.offsets());
                v
            }
            Op::PrefetchTensormap { desc } => vec![*desc],
            Op::TensormapReplace { addr, value, .. } => vec![*addr, *value],
            Op::FenceTensormapRelease => vec![],
            Op::FenceTensormapAcquire { addr } => vec![*addr],
            Op::TensormapCpFenceproxy { dst, src } => vec![*dst, *src],
            Op::BulkPrefetch { index, bytes, .. } => vec![*index, *bytes],
            Op::StAsync { index, values, mbar_index, peer, .. } => {
                let mut v = vec![*index];
                v.extend(values);
                v.push(*mbar_index);
                v.push(*peer);
                v
            }
            Op::Printf { args, .. } => args.clone(),
            Op::Assert { cond, .. } => vec![*cond],
            Op::InlinePtx { inputs, .. } => inputs.iter().map(|i| i.value()).collect(),
            Op::Call { args, .. } => args.iter().map(|i| i.value()).collect(),
            Op::Multimem { index, values, .. } => std::iter::once(*index).chain(values.iter().copied()).collect(),
            Op::Mma { a, b, c, e, .. } => a.iter().chain(b).chain(c).chain(e).copied().collect(),
            Op::If { cond, .. } => vec![*cond],
            Op::For { lower, upper, step, init, .. } => {
                let mut v = vec![*lower, *upper, *step];
                v.extend(init);
                v
            }
            Op::While { init, .. } => init.clone(),
            Op::CarryInst { args, .. } => args.clone(),
            Op::MlirOp { operands, .. } => operands.iter().map(|i| i.value()).collect(),
            Op::CpAsyncMbarrierArrive { index, .. } => vec![*index],
            Op::Prefetch { index, pred, .. } => {
                let mut v = vec![*index];
                v.extend(pred);
                v
            }
            Op::Pack { parts, .. } => parts.clone(),
            Op::Unpack { x, .. } => vec![*x],
            Op::HalfInst { args, predicate, .. } => {
                let mut v = args.clone();
                v.extend(predicate);
                v
            }
            Op::Cvt { args, .. } => args.clone(),
            Op::Assume { x, .. } | Op::CvtAddr { x, .. } => vec![*x],
            Op::PtrToInt { index, .. } => vec![*index],
            Op::Wgmma { c, a, desc_b, scale_d, .. } => {
                let mut v = c.clone();
                v.extend(a);
                v.extend([*desc_b, *scale_d]);
                v
            }
            Op::WgmmaFence { inputs, .. } | Op::WgmmaWait { inputs, .. } => inputs.clone(),
            Op::WgmmaCommitGroup => vec![],
            Op::Tcgen05 { args, .. } => args.clone(),
            Op::Tcgen05Ld { taddr, .. } => vec![*taddr],
            Op::Tcgen05St { values, taddr, .. } => {
                let mut v = vec![*taddr];
                v.extend(values);
                v
            }
            Op::Tcgen05Mma {
                taddr,
                desc_a,
                desc_b,
                enable_input_d,
                elect,
                idesc,
                disable_lane,
                sf,
                meta,
                zero_col_mask,
                ..
            } => {
                let mut v = vec![*taddr, *desc_a, *desc_b, *enable_input_d, *elect];
                v.extend(idesc);
                v.extend(disable_lane);
                if let Some((a, b)) = sf {
                    v.extend([*a, *b]);
                }
                v.extend(meta);
                v.extend(zero_col_mask);
                v
            }
            Op::MmaSpBlockScaled { a, b, c, e, sfa, sfb, .. } => {
                a.iter().chain(b).chain(c).copied().chain([*e, *sfa, *sfb]).collect()
            }
            Op::MmaBlockScaled { a, b, c, sfa, sfb, .. } => {
                a.iter().chain(b).chain(c).copied().chain([*sfa, *sfb]).collect()
            }
            Op::Tcgen05Commit { index, mask, .. } => {
                let mut v = vec![*index];
                v.extend(mask);
                v
            }
            Op::Tcgen05Cp { taddr, desc, elect, .. } => vec![*taddr, *desc, *elect],
            Op::ClcTryCancel { index, mbar_index, .. } => vec![*index, *mbar_index],
            Op::ClcQueryCancel { index, .. } => vec![*index],
        }
    }

    /// Nested regions.
    pub fn regions(&self) -> Vec<&Region> {
        match self {
            Op::If { then_region, else_region, .. } => vec![then_region, else_region],
            Op::For { body, .. } => vec![body],
            Op::While { cond, body, .. } => vec![cond, body],
            Op::Call { body: Some(body), .. } => vec![body],
            _ => vec![],
        }
    }

    /// Nested regions, mutably.
    pub fn regions_mut(&mut self) -> Vec<&mut Region> {
        match self {
            Op::If { then_region, else_region, .. } => vec![then_region, else_region],
            Op::For { body, .. } => vec![body],
            Op::While { cond, body, .. } => vec![cond, body],
            Op::Call { body: Some(body), .. } => vec![body],
            _ => vec![],
        }
    }

    /// Pure operations can be removed when their results are unused.
    pub fn is_pure(&self) -> bool {
        matches!(
            self,
            Op::Const { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Fma { .. }
                | Op::FloatInst { .. }
                | Op::IntInst { .. }
                | Op::Cmp { .. }
                | Op::Select { .. }
                | Op::Cast { .. }
                | Op::Bitcast { .. }
                | Op::Special { .. }
                | Op::WarpUniform { .. }
                | Op::InlinePtx { side_effects: false, .. }
                | Op::MlirOp { side_effects: false, .. }
                | Op::Pack { .. }
                | Op::Unpack { .. }
                | Op::HalfInst { .. }
                | Op::Cvt { .. }
                | Op::Assume { .. }
                | Op::PtrToInt { .. }
                | Op::CvtAddr { .. }
        ) && !matches!(self, Op::InlinePtx { constraints, .. } if asm_clobbers_memory(constraints))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Op::Const { .. } => "const",
            Op::Unary { .. } => "unary",
            Op::Binary { .. } => "binary",
            Op::Fma { .. } => "fma",
            Op::FloatInst { .. } => "float instruction",
            Op::IntInst { .. } => "integer instruction",
            Op::Cmp { .. } => "cmp",
            Op::Select { .. } => "select",
            Op::Cast { .. } => "cast",
            Op::Bitcast { .. } => "bitcast",
            Op::Special { .. } => "special register",
            Op::Load { .. } => "load",
            Op::Store { .. } => "store",
            Op::CpAsync { .. } => "cp.async",
            Op::CpAsyncCommitGroup => "cp.async.commit_group",
            Op::CpAsyncWaitGroup { .. } => "cp.async.wait_group",
            Op::LdMatrix { .. } => "ldmatrix",
            Op::StMatrix { .. } => "stmatrix",
            Op::SetMaxNReg { increase: true, .. } => "setmaxnreg.inc",
            Op::SetMaxNReg { increase: false, .. } => "setmaxnreg.dec",
            Op::Mma { kind, .. } if kind.is_sparse() => "mma.sp.sync",
            Op::Mma { .. } => "mma.sync",
            Op::Barrier => "barrier",
            Op::NamedBarrier { kind: BarrierKind::Sync, .. } => "bar.sync",
            Op::NamedBarrier { kind: BarrierKind::Arrive, .. } => "bar.arrive",
            Op::NamedBarrier { kind: BarrierKind::ArriveUnaligned, .. } => "barrier.arrive",
            Op::WarpBarrier { .. } => "bar.warp.sync",
            Op::Shuffle { .. } => "shfl.sync",
            Op::Vote { .. } => "vote.sync",
            Op::Match { kind: MatchKind::Any, .. } => "match.any.sync",
            Op::Match { kind: MatchKind::All, .. } => "match.all.sync",
            Op::ElectOne { .. } => "elect.sync",
            Op::ActiveMask { .. } => "activemask",
            Op::WarpUniform { .. } => "make_warp_uniform",
            Op::Atomic { .. } => "atomic",
            Op::Red { .. } => "red",
            Op::MbarrierInit { .. } => "mbarrier.init",
            Op::MbarrierArrive { expect_tx: Some(_), .. } => "mbarrier.arrive.expect_tx",
            Op::MbarrierArrive { .. } => "mbarrier.arrive",
            Op::MbarrierExpectTx { .. } => "mbarrier.expect_tx",
            Op::MbarrierWait { kind: MbarrierWaitKind::Try, .. } => "mbarrier.try_wait",
            Op::MbarrierWait { kind: MbarrierWaitKind::Test, .. } => "mbarrier.test_wait",
            Op::FenceMbarrierInit => "fence.mbarrier_init",
            Op::GridDepControl { launch_dependents: false } => "griddepcontrol.wait",
            Op::GridDepControl { launch_dependents: true } => "griddepcontrol.launch_dependents",
            Op::ClusterArrive { relaxed: false, .. } => "barrier.cluster.arrive",
            Op::ClusterArrive { relaxed: true, .. } => "barrier.cluster.arrive.relaxed",
            Op::ClusterWait { .. } => "barrier.cluster.wait",
            Op::CpAsyncBulk { .. } => "cp.async.bulk",
            Op::TmaLoad { .. } => "cp.async.bulk.tensor",
            Op::TmaStore { reduce: None, .. } => "cp.async.bulk.tensor (store)",
            Op::TmaStore { reduce: Some(_), .. } => "cp.reduce.async.bulk.tensor",
            Op::TmaPrefetch { .. } => "cp.async.bulk.prefetch.tensor",
            Op::PrefetchTensormap { .. } => "prefetch.tensormap",
            Op::TensormapReplace { .. } => "tensormap.replace",
            Op::FenceTensormapRelease => "fence.proxy.tensormap::generic.release",
            Op::FenceTensormapAcquire { .. } => "fence.proxy.tensormap::generic.acquire",
            Op::TensormapCpFenceproxy { .. } => "tensormap.cp_fenceproxy",
            Op::BulkPrefetch { .. } => "cp.async.bulk.prefetch",
            Op::StAsync { .. } => "st.async",
            Op::CpAsyncBulkCommitGroup => "cp.async.bulk.commit_group",
            Op::CpAsyncBulkWaitGroup { read: false, .. } => "cp.async.bulk.wait_group",
            Op::CpAsyncBulkWaitGroup { read: true, .. } => "cp.async.bulk.wait_group.read",
            Op::FenceProxy { kind } => kind.ptx(),
            Op::Fence { .. } => "fence",
            Op::Printf { .. } => "printf",
            Op::Assert { .. } => "assert",
            Op::InlinePtx { .. } => "inline ptx",
            Op::Call { .. } => "call",
            Op::Multimem { .. } => "multimem",
            Op::If { .. } => "if",
            Op::For { .. } => "for",
            Op::While { .. } => "while",
            Op::Redux { .. } => "redux.sync",
            Op::CarryInst { .. } => "carry instruction",
            Op::MlirOp { .. } => "mlir operation",
            Op::CpAsyncMbarrierArrive { noinc: true, .. } => "cp.async.mbarrier.arrive.noinc",
            Op::CpAsyncMbarrierArrive { noinc: false, .. } => "cp.async.mbarrier.arrive",
            Op::Prefetch { uniform: true, .. } => "prefetchu",
            Op::Prefetch { .. } => "prefetch",
            Op::Pack { .. } => "pack",
            Op::Unpack { .. } => "unpack",
            Op::HalfInst { .. } => "half-precision instruction",
            Op::Cvt { .. } => "cvt",
            Op::Assume { .. } => "assume",
            Op::PtrToInt { .. } => "ptrtoint",
            Op::CvtAddr { .. } => "cvta",
            Op::Wgmma { .. } => "wgmma.mma_async",
            Op::WgmmaFence { .. } => "wgmma.fence",
            Op::WgmmaCommitGroup => "wgmma.commit_group",
            Op::WgmmaWait { .. } => "wgmma.wait_group",
            Op::Tcgen05 { inst, .. } => inst.name(),
            Op::Tcgen05Ld { .. } => "tcgen05.ld",
            Op::Tcgen05St { .. } => "tcgen05.st",
            Op::Tcgen05Mma { .. } => "tcgen05.mma",
            Op::MmaBlockScaled { .. } => "mma.block_scale",
            Op::MmaSpBlockScaled { .. } => "mma.sp.block_scale",
            Op::Tcgen05Commit { .. } => "tcgen05.commit",
            Op::Tcgen05Cp { .. } => "tcgen05.cp",
            Op::ClcTryCancel { .. } => "clusterlaunchcontrol.try_cancel",
            Op::ClcQueryCancel { .. } => "clusterlaunchcontrol.query_cancel",
            Op::BarrierRed { .. } => "barrier.cta.red",
            Op::MbarrierExt { kind, .. } => kind.ptx(),
            Op::PrimMarker { marker: PrimMarker::Nanosleep(_) } => "nanosleep",
            Op::PrimMarker { marker: PrimMarker::Pmevent(_) } => "pmevent",
            Op::PrimMarker { marker: PrimMarker::Breakpoint } => "brkpt",
            Op::PrimMarker { marker: PrimMarker::Comment(_) } => "comment",
            Op::Exit => "exit",
            Op::StBulk { .. } => "st.bulk",
            Op::Tcgen05Shift { .. } => "tcgen05.shift",
        }
    }
}
