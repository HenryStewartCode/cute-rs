//! The stateful sides of a pipeline: [`PipelineProducer`] and
//! [`PipelineConsumer`], each a pipeline with its own [`PipelineState`],
//! and the handles to one acquired stage they return
//! ([`ProducerHandle`], [`ConsumerHandle`]).

use crate::control::Carry;
use crate::ir::ValueId;
use crate::tensor::Ptr;
use crate::value::{Boolean, Int32};

use super::{BarrierParam, CooperativeGroup, Pipeline, PipelineBarriers, PipelineState};

/// Whether two references name the same pipeline object.
fn same<P>(a: &P, b: &P) -> bool {
    std::ptr::eq(a, b)
}

/// The producer side of a pipeline (CuTe DSL's
/// `cutlass.pipeline.PipelineProducer`): the pipeline, the producer's
/// position in its ring and the producer group. Made by
/// [`Pipeline::make_producer`]; carried through loops ([`Carry`]: the state
/// is what changes).
///
/// [`acquire_and_advance`](Self::acquire_and_advance) waits for a free stage
/// and moves on, returning a [`ProducerHandle`] to the stage it acquired;
/// [`commit`](ProducerHandle::commit) on the handle (or
/// [`commit`](Self::commit) here) signals that it is full. A probe
/// ([`try_acquire`](Self::try_acquire)) gives a token that lets the acquire
/// skip its wait.
pub struct PipelineProducer<'p, P: Pipeline> {
    pipeline: &'p P,
    state: PipelineState,
    group: Option<CooperativeGroup>,
}

/// The consumer side of a pipeline (CuTe DSL's
/// `cutlass.pipeline.PipelineConsumer`), the mirror of
/// [`PipelineProducer`]: [`wait_and_advance`](Self::wait_and_advance) waits
/// for a full stage and moves on, and [`release`](ConsumerHandle::release)
/// on the [`ConsumerHandle`] it returns frees the stage.
pub struct PipelineConsumer<'p, P: Pipeline> {
    pipeline: &'p P,
    state: PipelineState,
    group: Option<CooperativeGroup>,
}

/// A stage a [`PipelineProducer`] acquired (CuTe DSL's producer
/// `ImmutableResourceHandle`): the producer's state at the acquire and the
/// pipeline it came from.
pub struct ProducerHandle<'p, P: Pipeline> {
    pipeline: &'p P,
    state: PipelineState,
}

/// A stage a [`PipelineConsumer`] waited for (CuTe DSL's consumer
/// `ImmutableResourceHandle`).
pub struct ConsumerHandle<'p, P: Pipeline> {
    pipeline: &'p P,
    state: PipelineState,
}

macro_rules! common {
    ($ty:ident) => {
        impl<P: Pipeline> Clone for $ty<'_, P> {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<P: Pipeline> Copy for $ty<'_, P> {}

        impl<P: Pipeline> std::fmt::Debug for $ty<'_, P> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($ty)).field("state", &self.state).finish_non_exhaustive()
            }
        }

        impl<P: Pipeline> Carry for $ty<'_, P> {
            #[track_caller]
            fn flatten(&self, out: &mut Vec<ValueId>) {
                self.state.flatten(out);
            }

            fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
                $ty { state: self.state.rebuild(values), ..*self }
            }

            fn structure_mismatch(&self, other: &Self) -> Option<String> {
                if !same(self.pipeline, other.pipeline) {
                    return Some(format!("a {} of one pipeline became one of another", stringify!($ty)));
                }
                self.state.structure_mismatch(&other.state)
            }
        }
    };
}

common!(PipelineProducer);
common!(PipelineConsumer);
common!(ProducerHandle);
common!(ConsumerHandle);

macro_rules! handle_common {
    ($ty:ident) => {
        impl<'p, P: Pipeline> $ty<'p, P> {
            /// The stage acquired.
            pub fn index(&self) -> Int32 {
                self.state.index()
            }

            /// The stages the participant had passed when it acquired this
            /// one (CuTe DSL's `count`), such as the k tiles loaded so far.
            pub fn count(&self) -> Int32 {
                self.state.count()
            }

            /// The participant's state at the acquire.
            pub fn state(&self) -> PipelineState {
                self.state
            }

            /// The pipeline the handle came from (CuTe DSL's `get_origin`).
            pub fn get_origin(&self) -> &'p P {
                self.pipeline
            }
        }
    };
}

handle_common!(ProducerHandle);
handle_common!(ConsumerHandle);

impl<P: PipelineBarriers> ProducerHandle<'_, P> {
    /// The barrier the producers arrive on for this stage (the full
    /// barrier), for a copy that arrives on it itself.
    #[track_caller]
    pub fn barrier(&self) -> Ptr<u64> {
        self.pipeline.producer_get_barrier(&self.state)
    }
}

impl<P: Pipeline> ProducerHandle<'_, P> {
    /// Signal that the stage is full ([`Pipeline::producer_commit`]).
    #[track_caller]
    pub fn commit(&self) {
        self.pipeline.producer_commit(&self.state);
    }
}

impl<P: PipelineBarriers> ConsumerHandle<'_, P> {
    /// The barrier the consumers arrive on for this stage (the empty
    /// barrier).
    #[track_caller]
    pub fn barrier(&self) -> Ptr<u64> {
        self.pipeline.consumer_get_barrier(&self.state)
    }
}

impl<P: Pipeline> ConsumerHandle<'_, P> {
    /// Signal that the stage is free again ([`Pipeline::consumer_release`]).
    #[track_caller]
    pub fn release(&self) {
        self.pipeline.consumer_release(&self.state);
    }
}

macro_rules! participant_common {
    ($ty:ident, $handle:ident) => {
        impl<'p, P: Pipeline> $ty<'p, P> {
            /// A participant of `pipeline` at `state`, for `group`.
            pub fn new(pipeline: &'p P, state: PipelineState, group: Option<CooperativeGroup>) -> Self {
                $ty { pipeline, state, group }
            }

            /// The pipeline.
            pub fn pipeline(&self) -> &'p P {
                self.pipeline
            }

            /// The current position in the ring.
            pub fn state(&self) -> PipelineState {
                self.state
            }

            /// The group of threads on this side, `None` for a pipeline
            /// without barriers.
            pub fn group(&self) -> Option<CooperativeGroup> {
                self.group
            }

            /// A copy at the same position (CuTe DSL's `clone`; the
            /// participant is `Copy`).
            pub fn clone_participant(&self) -> Self {
                *self
            }

            /// Set the count of stages passed to 0 (CuTe DSL's `reset`).
            #[track_caller]
            pub fn reset(&mut self) {
                self.state.reset_count();
            }

            /// A handle to the current stage, without waiting.
            pub fn current_handle(&self) -> $handle<'p, P> {
                $handle { pipeline: self.pipeline, state: self.state }
            }

            /// Move `count` stages on: a static count of at least 1 (1 as
            /// CuTe DSL's default), or an [`Int32`].
            #[track_caller]
            pub fn advance(&mut self, count: impl Into<BarrierParam>) {
                self.state.advance_by(count);
            }
        }
    };
}

participant_common!(PipelineProducer, ProducerHandle);
participant_common!(PipelineConsumer, ConsumerHandle);

impl<'p, P: Pipeline> PipelineProducer<'p, P> {
    /// Wait for the current stage to be free (unless `token` says it is)
    /// and return a handle to it (CuTe DSL's `acquire`).
    #[track_caller]
    pub fn acquire(&self, token: Option<Boolean>) -> ProducerHandle<'p, P> {
        self.acquire_with(token, P::AcquireOptions::default())
    }

    /// [`acquire`](Self::acquire) with the pipeline's acquire options (CuTe
    /// DSL's keyword arguments).
    #[track_caller]
    pub fn acquire_with(&self, token: Option<Boolean>, options: P::AcquireOptions) -> ProducerHandle<'p, P> {
        self.pipeline.producer_acquire_with(&self.state, token, options);
        self.current_handle()
    }

    /// [`acquire`](Self::acquire), then move to the next stage.
    #[track_caller]
    pub fn acquire_and_advance(&mut self, token: Option<Boolean>) -> ProducerHandle<'p, P> {
        self.acquire_and_advance_with(token, P::AcquireOptions::default())
    }

    /// [`acquire_with`](Self::acquire_with), then move to the next stage.
    #[track_caller]
    pub fn acquire_and_advance_with(
        &mut self,
        token: Option<Boolean>,
        options: P::AcquireOptions,
    ) -> ProducerHandle<'p, P> {
        let handle = self.acquire_with(token, options);
        self.state.advance();
        handle
    }

    /// Whether the current stage is free already, probing once: a token for
    /// [`acquire`](Self::acquire).
    #[track_caller]
    pub fn try_acquire(&self) -> Boolean {
        self.pipeline.producer_try_acquire(&self.state)
    }

    /// Signal that a stage is full: the handle's, or the current one with
    /// `None`. Panics, while the kernel is built, for a handle from another
    /// pipeline.
    #[track_caller]
    pub fn commit(&self, handle: Option<&ProducerHandle<'_, P>>) {
        match handle {
            Some(h) => {
                assert!(
                    same(h.pipeline, self.pipeline),
                    "cute: PipelineProducer::commit: the handle (ResourceHandle) was acquired from another pipeline than this producer's"
                );
                h.commit();
            }
            None => self.pipeline.producer_commit(&self.state),
        }
    }

    /// Before the producer exits, wait until every stage has been released
    /// ([`Pipeline::producer_tail`]); the state moves on by the number of
    /// stages.
    #[track_caller]
    pub fn tail(&mut self) {
        self.pipeline.producer_tail(&mut self.state);
    }
}

impl<'p, P: Pipeline> PipelineConsumer<'p, P> {
    /// Wait for the current stage to be full (unless `token` says it is)
    /// and return a handle to it (CuTe DSL's `wait`).
    #[track_caller]
    pub fn wait(&self, token: Option<Boolean>) -> ConsumerHandle<'p, P> {
        self.pipeline.consumer_wait(&self.state, token);
        self.current_handle()
    }

    /// [`wait`](Self::wait), then move to the next stage.
    #[track_caller]
    pub fn wait_and_advance(&mut self, token: Option<Boolean>) -> ConsumerHandle<'p, P> {
        let handle = self.wait(token);
        self.state.advance();
        handle
    }

    /// Whether the current stage is full already, probing once: a token for
    /// [`wait`](Self::wait).
    #[track_caller]
    pub fn try_wait(&self) -> Boolean {
        self.pipeline.consumer_try_wait(&self.state)
    }

    /// Signal that a stage is free: the handle's, or the current one with
    /// `None`. Panics, while the kernel is built, for a handle from another
    /// pipeline.
    #[track_caller]
    pub fn release(&self, handle: Option<&ConsumerHandle<'_, P>>) {
        match handle {
            Some(h) => {
                assert!(
                    same(h.pipeline, self.pipeline),
                    "cute: PipelineConsumer::release: the handle (ResourceHandle) was acquired from another pipeline than this consumer's"
                );
                h.release();
            }
            None => self.pipeline.consumer_release(&self.state),
        }
    }
}
