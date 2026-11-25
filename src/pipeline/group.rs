//! Who takes part in a pipeline's synchronisation: [`Agent`] and
//! [`CooperativeGroup`].

use crate::arch::{WARP_SIZE, block_dim, cluster_size};

use super::BarrierParam;

/// The kind of thread grouping that synchronises (CuTe DSL's
/// `cutlass.pipeline.Agent`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Agent {
    /// Any number of single threads.
    Thread,
    /// Warps of 32 threads.
    Warp,
    /// Whole thread blocks.
    ThreadBlock,
    /// Every thread block of a cluster.
    ThreadBlockCluster,
}

/// `size` agents of one kind, the threads on one side of a pipeline barrier
/// (CuTe DSL's `cutlass.pipeline.CooperativeGroup`). CuTe DSL's deprecated
/// `alignment` argument, which only warns, is left out.
///
/// An mbarrier counts the threads of one side only (the producers, or the
/// consumers), where a [`NamedBarrier`](super::NamedBarrier) counts both.
#[derive(Clone, Copy, Debug)]
pub struct CooperativeGroup {
    agent: Agent,
    size: BarrierParam,
}

impl CooperativeGroup {
    /// `size` agents of kind `agent`: a static count, which must be at
    /// least 1, or an [`Int32`](crate::Int32) computed at run time (not checked, as in
    /// CuTe DSL).
    #[track_caller]
    pub fn new(agent: Agent, size: impl Into<BarrierParam>) -> CooperativeGroup {
        let size = size.into();
        if let BarrierParam::Static(n) = size {
            assert!(n > 0, "cute: CooperativeGroup: the group must have at least one agent, not {n}");
        }
        CooperativeGroup { agent, size }
    }

    /// The kind of agent.
    pub fn agent(&self) -> Agent {
        self.agent
    }

    /// The number of agents.
    pub fn size(&self) -> BarrierParam {
        self.size
    }

    /// The number of threads in the group: `size` threads, `size * 32` for
    /// warps, `size` times the block's threads for thread blocks, and that
    /// times [`cluster_size`] for clusters. Static when it folds: always for
    /// threads and warps of a static size, and for blocks when the kernel
    /// declares its block size
    /// ([`KernelOptions::block_dim`](crate::KernelOptions::block_dim)).
    /// Blocks and clusters read [`block_dim`] (and the cluster's size), so
    /// they need a kernel being built.
    #[track_caller]
    pub fn thread_arrive_count(&self) -> BarrierParam {
        match self.agent {
            Agent::Thread => self.size,
            Agent::Warp => mul(self.size, BarrierParam::Static(WARP_SIZE as i64)),
            Agent::ThreadBlock => mul(self.size, block_threads()),
            Agent::ThreadBlockCluster => mul(mul(self.size, BarrierParam::Dynamic(cluster_size())), block_threads()),
        }
    }
}

/// The threads of a block, from [`block_dim`] (constant when declared).
#[track_caller]
fn block_threads() -> BarrierParam {
    let d = block_dim();
    [d.y, d.z].into_iter().fold(BarrierParam::Dynamic(d.x), |acc, e| mul(acc, BarrierParam::Dynamic(e)))
}

/// The product of two counts, static when both are (constants included).
#[track_caller]
fn mul(a: BarrierParam, b: BarrierParam) -> BarrierParam {
    match (a.as_static(), b.as_static()) {
        (Some(x), Some(y)) => BarrierParam::Static(x * y),
        _ => BarrierParam::Dynamic(a.value() * b.value()),
    }
}
