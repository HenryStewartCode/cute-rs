//! The interface of a pipeline's synchronisation objects ([`SyncObject`]).

use crate::layout::IndexExpr;
use crate::tensor::Ptr;
use crate::value::Int32;

use super::{BarrierParam, CtaGroup, MbarrierArray, NamedBarrier};

/// A synchronisation object a pipeline's full or empty side can use (CuTe
/// DSL's abstract `cutlass.pipeline.SyncObject`). The objects take
/// different arguments, so each names them as associated types: an
/// [`MbarrierArray`] arrives on a stage, maybe on a peer CTA, and waits on
/// a stage and phase; a [`NamedBarrier`] takes nothing. The pipelines keep
/// their concrete types; the trait is for code generic over the object.
///
/// [`MbarrierArray`], [`NamedBarrier`] and the TMA store pipeline's
/// [`TmaStoreFence`](super::TmaStoreFence) implement it.
pub trait SyncObject {
    /// What [`arrive`](Self::arrive) takes.
    type ArriveArgs;
    /// What [`wait`](Self::wait) takes.
    type WaitArgs;
    /// What [`get_barrier`](Self::get_barrier) takes.
    type BarrierArgs;
    /// What [`get_barrier`](Self::get_barrier) returns.
    type Barrier;

    /// Signal the object.
    fn arrive(&self, args: Self::ArriveArgs);
    /// Wait on the object.
    fn wait(&self, args: Self::WaitArgs);
    /// [`arrive`](Self::arrive), then [`wait`](Self::wait).
    fn arrive_and_wait(&self, arrive: Self::ArriveArgs, wait: Self::WaitArgs);
    /// Arrive and leave the barrier for good, where supported (panics
    /// otherwise, as CuTe DSL raises).
    fn arrive_and_drop(&self);
    /// The underlying barrier.
    fn get_barrier(&self, args: Self::BarrierArgs) -> Self::Barrier;
    /// The largest arrival count, or `None` for an object without one.
    fn max(&self) -> Option<u32>;
}

impl SyncObject for MbarrierArray {
    /// The stage, the peer CTA (or multicast mask) and the CTA group, as
    /// [`MbarrierArray::arrive`] takes them.
    type ArriveArgs = (IndexExpr, Option<Int32>, Option<CtaGroup>);
    /// The stage and the phase.
    type WaitArgs = (IndexExpr, Int32);
    /// The stage.
    type BarrierArgs = IndexExpr;
    type Barrier = Ptr<u64>;

    #[track_caller]
    fn arrive(&self, (index, dst, cta_group): Self::ArriveArgs) {
        MbarrierArray::arrive(self, index, dst, cta_group);
    }

    #[track_caller]
    fn wait(&self, (index, phase): Self::WaitArgs) {
        MbarrierArray::wait(self, index, phase);
    }

    #[track_caller]
    fn arrive_and_wait(&self, (index, dst, cta_group): Self::ArriveArgs, (wait_index, phase): Self::WaitArgs) {
        MbarrierArray::arrive(self, index, dst, cta_group);
        MbarrierArray::wait(self, wait_index, phase);
    }

    #[track_caller]
    fn arrive_and_drop(&self) {
        MbarrierArray::arrive_and_drop(self);
    }

    #[track_caller]
    fn get_barrier(&self, index: IndexExpr) -> Ptr<u64> {
        MbarrierArray::get_barrier(self, index)
    }

    fn max(&self) -> Option<u32> {
        Some(MbarrierArray::max(self))
    }
}

impl SyncObject for NamedBarrier {
    type ArriveArgs = ();
    type WaitArgs = ();
    type BarrierArgs = ();
    /// The barrier's id.
    type Barrier = BarrierParam;

    #[track_caller]
    fn arrive(&self, (): ()) {
        NamedBarrier::arrive(self);
    }

    #[track_caller]
    fn wait(&self, (): ()) {
        NamedBarrier::wait(self);
    }

    #[track_caller]
    fn arrive_and_wait(&self, (): (), (): ()) {
        NamedBarrier::arrive_and_wait(self);
    }

    #[track_caller]
    fn arrive_and_drop(&self) {
        NamedBarrier::arrive_and_drop(self);
    }

    fn get_barrier(&self, (): ()) -> BarrierParam {
        NamedBarrier::get_barrier(self)
    }

    fn max(&self) -> Option<u32> {
        Some(NamedBarrier::max(self))
    }
}
