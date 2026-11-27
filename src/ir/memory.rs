//! Memory-system kinds: TMA modes, proxies, fences and memory orders, cache hints, `MemAttrs` and the barrier kinds.

use super::*;

/// Which way a bulk copy (`cp.async.bulk`) moves data.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BulkDirection {
    /// Global to shared memory (`cp.async.bulk.shared::cluster.global`).
    G2S,
    /// Shared to global memory (`cp.async.bulk.global.shared::cta`).
    S2G,
    /// From this CTA's shared memory to a peer CTA's of its cluster
    /// (`cp.async.bulk.shared::cluster.shared::cta`).
    S2S,
}

// ---- TMA im2col and SM100 modes ----

/// The tensor access mode of a TMA instruction (the PTX ISA's `.load_mode`).
#[derive(Clone, Debug, PartialEq, Default)]
pub enum TmaMode {
    /// `.tile`: the box at the coordinates.
    #[default]
    Tile,
    /// `.im2col` (loads and prefetches): the coordinates are the filter base
    /// `c, w, (h, (d,)) n`, and `offsets` (i16, one per spatial dimension, W
    /// first) are added to each pixel.
    Im2col { offsets: Vec<ValueId> },
    /// `.im2col_no_offs` (stores): im2col coordinates without offsets.
    Im2colNoOffs,
    /// `.tile::gather4` (loads and prefetches): five coordinates, the column
    /// and four rows of a 2-D tensor map.
    Gather4,
    /// `.tile::scatter4` (stores): as [`TmaMode::Gather4`].
    Scatter4,
}

impl TmaMode {
    /// The PTX qualifier.
    pub fn ptx(&self) -> &'static str {
        match self {
            TmaMode::Tile => ".tile",
            TmaMode::Im2col { .. } => ".im2col",
            TmaMode::Im2colNoOffs => ".im2col_no_offs",
            TmaMode::Gather4 => ".tile::gather4",
            TmaMode::Scatter4 => ".tile::scatter4",
        }
    }

    /// The four-row modes.
    pub fn is_rows4(&self) -> bool {
        matches!(self, TmaMode::Gather4 | TmaMode::Scatter4)
    }

    /// The im2col offsets (empty for the other modes).
    pub fn offsets(&self) -> &[ValueId] {
        match self {
            TmaMode::Im2col { offsets } => offsets,
            _ => &[],
        }
    }
}

/// The `.override::global_address` and `.override::global_dim{_stride}`
/// operands of a TMA instruction (sm_107f, PTX 9.4): the tensor's address
/// (an i64), and when `dims` is not empty its extents (one i16 per
/// dimension, the value in the low 8 bits) and strides (`lower_strides`,
/// one i32 per dimension after the first, and the i16 `upper_strides`
/// holding a 4-bit value per dimension after the first, the stride in bytes
/// being `(lower + (upper << 32)) << 4`). The coordinates are then zero.
#[derive(Clone, Debug, PartialEq)]
pub struct TmaOverride {
    pub global_address: ValueId,
    pub dims: Vec<ValueId>,
    pub lower_strides: Vec<ValueId>,
    pub upper_strides: Option<ValueId>,
}

impl TmaOverride {
    /// Its operands.
    pub fn operands(&self) -> Vec<ValueId> {
        let mut v = vec![self.global_address];
        v.extend(&self.dims);
        v.extend(&self.lower_strides);
        v.extend(self.upper_strides);
        v
    }
}

/// How a bulk copy reports its completion.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BulkCompletion {
    /// `.mbarrier::complete_tx::bytes`: its bytes count down the
    /// transaction count of the mbarrier at element `index` of `buffer`.
    Mbarrier { buffer: BufferId, index: ValueId },
    /// `.bulk_group`: the thread's bulk async-groups
    /// (`cp.async.bulk.commit_group`, `cp.async.bulk.wait_group`).
    BulkGroup,
}

/// The proxies a `fence.proxy` orders (CuTe DSL's `fence_proxy` kinds the
/// IR has; `tensormap` and `generic` belong to the tensor-map fences).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ProxyKind {
    /// `fence.proxy.alias`: accesses to the same memory through different
    /// virtual addresses.
    Alias,
    /// `fence.proxy.async`: the generic and async proxies, in every state
    /// space.
    Async,
    /// `fence.proxy.async.global`.
    AsyncGlobal,
    /// `fence.proxy.async.shared::cta`.
    AsyncSharedCta,
    /// `fence.proxy.async.shared::cluster`.
    AsyncSharedCluster,
}

impl ProxyKind {
    /// The PTX instruction, without the semicolon.
    pub fn ptx(self) -> &'static str {
        match self {
            ProxyKind::Alias => "fence.proxy.alias",
            ProxyKind::Async => "fence.proxy.async",
            ProxyKind::AsyncGlobal => "fence.proxy.async.global",
            ProxyKind::AsyncSharedCta => "fence.proxy.async.shared::cta",
            ProxyKind::AsyncSharedCluster => "fence.proxy.async.shared::cluster",
        }
    }

    /// Whether it orders the generic proxy's shared-memory accesses before
    /// the async proxy's (every async kind but `async.global`).
    pub fn orders_async_shared(self) -> bool {
        matches!(self, ProxyKind::Async | ProxyKind::AsyncSharedCta | ProxyKind::AsyncSharedCluster)
    }
}

/// The threads a fence orders against.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FenceScope {
    /// The block.
    Cta,
    /// The block cluster.
    Cluster,
    /// The whole GPU.
    Gpu,
    /// The GPU and the host.
    Sys,
}

impl FenceScope {
    /// The PTX scope qualifier, without the dot (`cta`, `cluster`, `gpu`,
    /// `sys`).
    pub fn ptx(self) -> &'static str {
        match self {
            FenceScope::Cta => "cta",
            FenceScope::Cluster => "cluster",
            FenceScope::Gpu => "gpu",
            FenceScope::Sys => "sys",
        }
    }
}

/// The threads an ordered load or store synchronises with: CuTe DSL's
/// `cute.nvgpu.MemoryScope` (`CTA`, `CLUSTER`, `GPU`, `SYS`) and the
/// `scope=` of `cute.arch.load` and `cute.arch.store` (`"cta"`, `"cluster"`,
/// `"gpu"`, `"sys"`). The scopes of a fence.
pub type MemoryScope = FenceScope;

/// The memory-ordering semantics of a load or store (PTX's `.weak`,
/// `.relaxed`, `.acquire`, `.release`, `.volatile` and `.mmio`): CuTe DSL's
/// `cute.nvgpu.MemoryOrder` and the `sem=` of `cute.arch.load` and
/// `cute.arch.store`.
///
/// Every order but `Weak` and `Constant` makes the access *strong* in the
/// PTX memory model: two strong accesses of the same bytes from threads in
/// each other's scope never race (strong accesses of different sizes that
/// overlap do: a mixed-size data race). A release store (or a strong store
/// after a fence) publishes what its thread has seen, and an acquire load
/// that reads exactly what it wrote (or such a strong load followed by a
/// fence) sees it too. Each element of a vector access is one access.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MemoryOrder {
    /// `.weak`, a plain access with no synchronisation (the default).
    #[default]
    Weak,
    /// `.relaxed.scope`: strong, but orders nothing else.
    Relaxed,
    /// `.acquire.scope` (loads): later accesses of the thread see what the
    /// release it reads from published.
    Acquire,
    /// `.release.scope` (stores): publishes the thread's earlier accesses.
    Release,
    /// Acquire-release: an order for atomics, which loads and stores do not
    /// take.
    AcqRel,
    /// Sequentially consistent. PTX has no `.sc` load or store; this is
    /// LLVM's mapping of a `seq_cst` access: `fence.sc.scope` followed by an
    /// acquire load or a release store.
    Sc,
    /// `.mmio.relaxed.sys`: a scalar access to memory-mapped I/O in global
    /// memory, performed exactly once, with relaxed system-scope semantics.
    Mmio,
    /// A load of memory that no thread writes while the kernel runs, as a
    /// non-coherent load (`ld.global.nc`, like CuTe DSL's `invariant`).
    Constant,
    /// `.volatile`: relaxed with system scope, never cached, and taking no
    /// cache hints.
    Volatile,
}

impl MemoryOrder {
    /// Whether accesses with this order are strong in PTX's memory model.
    pub fn is_strong(self) -> bool {
        !matches!(self, MemoryOrder::Weak | MemoryOrder::Constant)
    }

    /// Whether this order takes a scope (`.relaxed.scope` and so on).
    pub fn is_scoped(self) -> bool {
        matches!(self, MemoryOrder::Relaxed | MemoryOrder::Acquire | MemoryOrder::Release | MemoryOrder::Sc)
    }

    /// CuTe DSL's name, for messages.
    pub fn name(self) -> &'static str {
        match self {
            MemoryOrder::Weak => "weak",
            MemoryOrder::Relaxed => "relaxed",
            MemoryOrder::Acquire => "acquire",
            MemoryOrder::Release => "release",
            MemoryOrder::AcqRel => "acq_rel",
            MemoryOrder::Sc => "sc",
            MemoryOrder::Mmio => "mmio",
            MemoryOrder::Constant => "constant",
            MemoryOrder::Volatile => "volatile",
        }
    }
}

/// An L1 eviction priority (`.L1::evict_normal`, `.L1::evict_first`,
/// `.L1::evict_last`, `.L1::evict_unchanged`, `.L1::no_allocate`): CuTe
/// DSL's `cute.nvgpu.CacheEvictionPriority` and the
/// `level1_eviction_priority=` of `cute.arch.load` and `cute.arch.store`.
/// A prefetch takes `EvictLast` and `EvictNormal` for the L2 cache.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CacheEvictionPriority {
    /// `evict_normal`.
    #[default]
    EvictNormal,
    /// `evict_first`: likely to be evicted first.
    EvictFirst,
    /// `evict_last`: likely to be evicted last.
    EvictLast,
    /// `evict_unchanged`: keep the line's current priority.
    EvictUnchanged,
    /// `no_allocate` (CuTe DSL's `evict_no_allocate` and `NO_ALLOCATE`):
    /// do not allocate the line in L1.
    NoAllocate,
}

impl CacheEvictionPriority {
    /// The PTX name of the priority (`evict_last`, `no_allocate`).
    pub fn name(self) -> &'static str {
        match self {
            CacheEvictionPriority::EvictNormal => "evict_normal",
            CacheEvictionPriority::EvictFirst => "evict_first",
            CacheEvictionPriority::EvictLast => "evict_last",
            CacheEvictionPriority::EvictUnchanged => "evict_unchanged",
            CacheEvictionPriority::NoAllocate => "no_allocate",
        }
    }

    /// The L1 qualifier, without the dot (`L1::evict_last`).
    pub fn l1_ptx(self) -> String {
        format!("L1::{}", self.name())
    }
}

/// A PTX cache operator: `.ca`, `.cg`, `.cs`, `.lu` and `.cv` for loads,
/// `.wb`, `.cg`, `.cs` and `.wt` for stores (the `cop=` of CuTe DSL's
/// `cute.arch.load` and `cute.arch.store`). A performance hint only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CacheOperator {
    /// `.ca`: cache at all levels (loads; the default).
    Ca,
    /// `.cg`: cache in L2, not L1.
    Cg,
    /// `.cs`: streaming, likely to be accessed once.
    Cs,
    /// `.lu`: last use (loads).
    Lu,
    /// `.cv`: do not cache, fetch again (loads).
    Cv,
    /// `.wb`: write back (stores; the default).
    Wb,
    /// `.wt`: write through to system memory (stores).
    Wt,
}

impl CacheOperator {
    /// The PTX qualifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            CacheOperator::Ca => "ca",
            CacheOperator::Cg => "cg",
            CacheOperator::Cs => "cs",
            CacheOperator::Lu => "lu",
            CacheOperator::Cv => "cv",
            CacheOperator::Wb => "wb",
            CacheOperator::Wt => "wt",
        }
    }

    /// Whether loads take it.
    pub fn for_loads(self) -> bool {
        !matches!(self, CacheOperator::Wb | CacheOperator::Wt)
    }

    /// Whether stores take it.
    pub fn for_stores(self) -> bool {
        matches!(self, CacheOperator::Wb | CacheOperator::Cg | CacheOperator::Cs | CacheOperator::Wt)
    }
}

/// How much more data a load asks L2 to fetch (`.L2::64B`, `.L2::128B`,
/// `.L2::256B`): CuTe DSL's `cute.nvgpu.L2PrefetchSize` and the
/// `level_prefetch_size=` of `cute.arch.load`. A performance hint only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum L2PrefetchSize {
    /// `.L2::64B` (`size_64b`, `SIZE_64B`).
    Size64B,
    /// `.L2::128B` (`size_128b`, `SIZE_128B`).
    Size128B,
    /// `.L2::256B` (`size_256b`, `SIZE_256B`), sm_80 and newer.
    Size256B,
}

impl L2PrefetchSize {
    /// The PTX qualifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            L2PrefetchSize::Size64B => "L2::64B",
            L2PrefetchSize::Size128B => "L2::128B",
            L2PrefetchSize::Size256B => "L2::256B",
        }
    }
}

/// The memory attributes of a load or store: its ordering and scope, its
/// cache hints, whether it reads through the non-coherent cache, and an
/// optional L2 cache-policy operand (`.L2::cache_hint`, a 64-bit value from
/// `createpolicy`). The default is a plain weak access, which is what
/// tensor loads and stores are.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MemAttrs {
    pub order: MemoryOrder,
    /// The threads a relaxed, acquire, release or seq_cst access
    /// synchronises with. Weak, volatile and constant accesses have none,
    /// and MMIO is always `Sys`.
    pub scope: MemoryScope,
    /// `.L1::evict_*` (loads and stores).
    pub l1_eviction: Option<CacheEvictionPriority>,
    /// `.ca`, `.cg`, ... (loads and stores).
    pub cop: Option<CacheOperator>,
    /// `.L2::64B` and so on (loads).
    pub l2_prefetch: Option<L2PrefetchSize>,
    /// `.nc`: read through the non-coherent (read-only) cache (loads).
    pub non_coherent: bool,
    /// The 64-bit operand of `.L2::cache_hint`.
    pub cache_policy: Option<ValueId>,
    /// Distributed shared memory: the rank (an
    /// Int32) of the CTA of the cluster whose shared memory the access
    /// reaches, at the offset it has in the buffer of this CTA (`mapa`, then
    /// `.shared::cluster`). Shared buffers only.
    pub peer: Option<ValueId>,
}

impl Default for MemAttrs {
    fn default() -> Self {
        MemAttrs {
            order: MemoryOrder::Weak,
            scope: FenceScope::Sys,
            l1_eviction: None,
            cop: None,
            l2_prefetch: None,
            non_coherent: false,
            cache_policy: None,
            peer: None,
        }
    }
}

impl MemAttrs {
    /// A weak access with no hints, as [`Tensor::load`](crate::Tensor::load)
    /// and `store` make (the scope does not matter).
    pub fn is_plain(&self) -> bool {
        self.order == MemoryOrder::Weak && !self.has_hints()
    }

    /// Whether it carries a cache hint, a cache policy or `.nc`.
    pub fn has_hints(&self) -> bool {
        self.l1_eviction.is_some()
            || self.cop.is_some()
            || self.l2_prefetch.is_some()
            || self.non_coherent
            || self.cache_policy.is_some()
    }

    /// Whether the access is strong ([`MemoryOrder::is_strong`]).
    pub fn is_strong(&self) -> bool {
        self.order.is_strong()
    }

    /// Whether it reads through the non-coherent cache (`.nc`, or
    /// [`MemoryOrder::Constant`]).
    pub fn is_non_coherent(&self) -> bool {
        self.non_coherent || self.order == MemoryOrder::Constant
    }

    /// The PTX qualifiers of the access, as in `.acquire.gpu.L1::evict_last`
    /// (the state space, vector width and type are not included), for
    /// listings and messages.
    pub fn qualifiers(&self) -> String {
        let mut q = String::new();
        match self.order {
            MemoryOrder::Weak | MemoryOrder::Constant => {}
            MemoryOrder::Volatile => q.push_str(".volatile"),
            MemoryOrder::Mmio => q.push_str(".mmio.relaxed.sys"),
            o => q.push_str(&format!(".{}.{}", o.name(), self.scope.ptx())),
        }
        if self.is_non_coherent() {
            q.push_str(".nc");
        }
        if let Some(c) = self.cop {
            q.push_str(&format!(".{}", c.ptx()));
        }
        if let Some(e) = self.l1_eviction {
            q.push_str(&format!(".{}", e.l1_ptx()));
        }
        if self.cache_policy.is_some() {
            q.push_str(".L2::cache_hint");
        }
        if let Some(p) = self.l2_prefetch {
            q.push_str(&format!(".{}", p.ptx()));
        }
        q
    }
}

/// The cache a prefetch fills: CuTe DSL's `cache_level=` of
/// `cute.arch.prefetch` (`"L1"`, the default, or `"L2"`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PrefetchCacheLevel {
    #[default]
    L1,
    L2,
}

impl PrefetchCacheLevel {
    /// The PTX qualifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            PrefetchCacheLevel::L1 => "L1",
            PrefetchCacheLevel::L2 => "L2",
        }
    }
}

/// The threads an mbarrier operation synchronises with (CuTe DSL's
/// `nvvm.MemScopeKind` as the mbarrier operations take it).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MemScopeKind {
    /// The CTA (the default).
    #[default]
    Cta,
    /// The CTA's cluster (SM90 and newer).
    Cluster,
}

/// How an mbarrier phase is waited on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MbarrierWaitKind {
    /// `mbarrier.try_wait`: may suspend the thread for a while (SM90).
    Try,
    /// `mbarrier.test_wait`: returns at once (SM80).
    Test,
}

/// The named-barrier instructions (`bar`, `barrier`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BarrierKind {
    /// `bar.sync id{, count}`: arrive, and wait until the barrier completes.
    Sync,
    /// `bar.arrive id, count`: arrive without waiting.
    Arrive,
    /// `barrier.arrive id, count`: an arrive without `.aligned`, which the
    /// threads of the CTA need not all execute.
    ArriveUnaligned,
}

/// The reduction of a `barrier.cta.red` (`bar.red`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BarrierRedKind {
    /// `.popc.u32`: how many participating threads' predicates are true.
    Popc,
    /// `.and.pred`: whether every participating thread's predicate is true.
    And,
    /// `.or.pred`: whether any participating thread's predicate is true.
    Or,
}

impl BarrierRedKind {
    /// The PTX reduction and type (`popc.u32`, `and.pred`, `or.pred`).
    pub fn ptx(self) -> &'static str {
        match self {
            BarrierRedKind::Popc => "popc.u32",
            BarrierRedKind::And => "and.pred",
            BarrierRedKind::Or => "or.pred",
        }
    }

    /// The reduction's name (`popc`, `and`, `or`).
    pub fn name(self) -> &'static str {
        match self {
            BarrierRedKind::Popc => "popc",
            BarrierRedKind::And => "and",
            BarrierRedKind::Or => "or",
        }
    }
}

/// The extra mbarrier operations of `cute::prims`
/// ([`Op::MbarrierExt`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MbarrierExtKind {
    /// `mbarrier.arrive.shared.b64 state, [addr]`: one arrival that returns
    /// the phase's state token.
    ArriveState,
    /// `mbarrier.arrive_drop.shared.b64 state, [addr]`: one arrival that
    /// also lowers the arrival count of this phase and all later ones.
    ArriveDrop,
    /// `mbarrier.arrive_drop.expect_tx.shared::cta.b64 state, [addr], tx`
    /// (sm_90): raise the tx-count by `value` bytes, then arrive-drop.
    ArriveDropExpectTx,
    /// `mbarrier.arrive_drop.noComplete.shared.b64 state, [addr], count`:
    /// `value` arrive-drops that must not complete the phase.
    ArriveDropNoComplete,
    /// `mbarrier.arrive.noComplete.shared.b64 state, [addr], count`: `value`
    /// arrivals that must not complete the phase.
    ArriveNoComplete,
    /// `mbarrier.complete_tx.shared::cta.b64 [addr], tx` (sm_90): lower the
    /// tx-count by `value` bytes, as if bulk copies had landed.
    CompleteTx,
    /// `mbarrier.inval.shared.b64 [addr]`: the mbarrier's memory becomes
    /// plain shared memory again; any later use but `mbarrier.init` is an error.
    Inval,
}

impl MbarrierExtKind {
    /// The PTX operation, without its qualifiers.
    pub fn ptx(self) -> &'static str {
        match self {
            MbarrierExtKind::ArriveState => "mbarrier.arrive",
            MbarrierExtKind::ArriveDrop => "mbarrier.arrive_drop",
            MbarrierExtKind::ArriveDropExpectTx => "mbarrier.arrive_drop.expect_tx",
            MbarrierExtKind::ArriveDropNoComplete => "mbarrier.arrive_drop.noComplete",
            MbarrierExtKind::ArriveNoComplete => "mbarrier.arrive.noComplete",
            MbarrierExtKind::CompleteTx => "mbarrier.complete_tx",
            MbarrierExtKind::Inval => "mbarrier.inval",
        }
    }

    /// Whether it returns a state token (an i64).
    pub fn has_state(self) -> bool {
        !matches!(self, MbarrierExtKind::CompleteTx | MbarrierExtKind::Inval)
    }

    /// Whether it takes a value: an arrival count or transaction bytes.
    pub fn has_value(self) -> bool {
        matches!(
            self,
            MbarrierExtKind::ArriveDropExpectTx
                | MbarrierExtKind::ArriveDropNoComplete
                | MbarrierExtKind::ArriveNoComplete
                | MbarrierExtKind::CompleteTx
        )
    }
}

/// The per-thread markers and hints of `cute::prims`
/// ([`Op::PrimMarker`]), which the simulator counts and otherwise skips.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum PrimMarker {
    /// `nanosleep.u32 t`: suspend the thread for about `t` nanoseconds (a
    /// hint; the simulator yields).
    Nanosleep(ValueId),
    /// `pmevent.mask m`: trigger the performance-monitor events of the
    /// 16-bit mask.
    Pmevent(u16),
    /// `brkpt`: suspend the thread for a debugger.
    Breakpoint,
    /// A `// text` comment line in the PTX.
    Comment(String),
}
