//! Publishing pipeline barriers after their initialisation
//! ([`pipeline_init_arrive`], [`pipeline_init_wait`]) and synchronising an
//! agent ([`agent_sync`]).

use crate::arch::{cluster_arrive, cluster_arrive_relaxed, cluster_wait, mbarrier_init_fence, sync_threads};
use crate::layout::{Shape, size};

use super::Agent;
use super::mbarrier::portable_waits;

/// The number of CTAs in `cluster_shape_mn`, 1 without one.
#[track_caller]
fn cluster_ctas(cluster_shape_mn: Option<&Shape>) -> usize {
    cluster_shape_mn.map_or(1, |shape| {
        size(shape, &[]).as_static().unwrap_or_else(|| {
            panic!("cute: the cluster shape of a pipeline's initialisation must be static, not {shape}")
        })
    })
}

/// The first half of publishing a pipeline's mbarrier initialisations
/// (CuTe DSL's `pipeline_init_arrive`): `fence.mbarrier_init` (left out
/// below sm_90, see [`MbarrierArray`](super::MbarrierArray)), then, when
/// `cluster_shape_mn` (static) has more than one CTA, an arrival at the
/// cluster barrier: `barrier.cluster.arrive.relaxed` when `is_relaxed`,
/// `barrier.cluster.arrive` otherwise. A single CTA has nothing to arrive
/// at: [`pipeline_init_wait`]'s CTA barrier does the work.
///
/// Every thread of the CTA (of the cluster) calls it after the barriers are
/// made and before they are used, then [`pipeline_init_wait`]; work that
/// does not touch the barriers can go between the two. Pipelines made with
/// their own init sync deferred share one pair of calls this way.
#[track_caller]
pub fn pipeline_init_arrive(cluster_shape_mn: Option<Shape>, is_relaxed: bool) {
    if !portable_waits() {
        mbarrier_init_fence();
    }
    if cluster_ctas(cluster_shape_mn.as_ref()) > 1 {
        if is_relaxed {
            cluster_arrive_relaxed();
        } else {
            cluster_arrive();
        }
    }
}

/// The second half of [`pipeline_init_arrive`] (CuTe DSL's
/// `pipeline_init_wait`): [`sync_threads`] for a single CTA (no
/// `cluster_shape_mn`, or one of size 1), `barrier.cluster.wait` for a
/// cluster. Afterwards every barrier initialised before
/// [`pipeline_init_arrive`] can be used, from any CTA of the cluster.
#[track_caller]
pub fn pipeline_init_wait(cluster_shape_mn: Option<Shape>) {
    if cluster_ctas(cluster_shape_mn.as_ref()) > 1 {
        cluster_wait();
    } else {
        agent_sync(Agent::ThreadBlock, false);
    }
}

/// Synchronise every thread of `agent` (CuTe DSL's `agent_sync`): a CTA
/// barrier ([`sync_threads`]) for [`Agent::ThreadBlock`], and a cluster
/// barrier (`barrier.cluster.arrive`, `.relaxed` when `is_relaxed`, then
/// `barrier.cluster.wait`) for [`Agent::ThreadBlockCluster`]. Single
/// threads cannot be synchronised this way, and warps have no such
/// instruction: [`Agent::Thread`] and [`Agent::Warp`] panic.
#[track_caller]
pub fn agent_sync(agent: Agent, is_relaxed: bool) {
    match agent {
        Agent::ThreadBlock => sync_threads(),
        Agent::ThreadBlockCluster => {
            if is_relaxed {
                cluster_arrive_relaxed();
            } else {
                cluster_arrive();
            }
            cluster_wait();
        }
        Agent::Thread => panic!("cute: agent_sync(Agent::Thread) is not supported"),
        Agent::Warp => panic!(
            "cute: agent_sync(Agent::Warp): there is no warp barrier instruction to sync with; use a named barrier \
             or an mbarrier"
        ),
    }
}
