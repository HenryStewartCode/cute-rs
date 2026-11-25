//! CUDA events (`CUevent`): markers recorded in a stream, for timing, for ordering work
//! across streams ([`CudaStream::wait_event`]) and as the launch-completion and programmatic
//! events of a launch ([`LaunchBuilder::launch_completion_event`](super::LaunchBuilder::launch_completion_event),
//! [`LaunchBuilder::programmatic_event`](super::LaunchBuilder::programmatic_event); CuTe DSL's
//! `launch_completion_event=` and `programmatic_event=`).

use std::ffi::{c_int, c_uint};

use super::{CuEvent, CudaContext, CudaError, CudaStream, check};

/// `CUDA_ERROR_NOT_READY`: the event's work is still pending.
const CUDA_ERROR_NOT_READY: c_int = 600;

/// The flags an event is created with (`CU_EVENT_*`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EventFlags {
    /// `CU_EVENT_BLOCKING_SYNC`: [`CudaEvent::synchronize`] blocks the thread instead of
    /// spinning.
    pub blocking_sync: bool,
    /// `CU_EVENT_DISABLE_TIMING`: no timestamp (faster; required for the events of a launch
    /// attribute).
    pub disable_timing: bool,
}

impl EventFlags {
    /// `CU_EVENT_DEFAULT`: timing, spin-wait synchronisation.
    pub const DEFAULT: EventFlags = EventFlags { blocking_sync: false, disable_timing: false };
    /// `CU_EVENT_DISABLE_TIMING`.
    pub const DISABLE_TIMING: EventFlags = EventFlags { blocking_sync: false, disable_timing: true };

    fn bits(self) -> c_uint {
        (self.blocking_sync as c_uint) | (self.disable_timing as c_uint) << 1
    }
}

/// A CUDA event, destroyed (`cuEventDestroy`) when dropped.
pub struct CudaEvent<'ctx> {
    pub(super) ctx: &'ctx CudaContext,
    pub(super) raw: CuEvent,
    flags: EventFlags,
}

impl CudaContext {
    /// A new event (`cuEventCreate`).
    pub fn event(&self, flags: EventFlags) -> Result<CudaEvent<'_>, CudaError> {
        self.make_current()?;
        let mut raw: CuEvent = std::ptr::null_mut();
        // SAFETY: a valid out-pointer.
        check(&self.api, unsafe { (self.api.event_create)(&mut raw, flags.bits()) }, "cuEventCreate")?;
        Ok(CudaEvent { ctx: self, raw, flags })
    }
}

impl<'ctx> CudaEvent<'ctx> {
    /// The flags it was created with.
    pub fn flags(&self) -> EventFlags {
        self.flags
    }

    /// The `CUevent` handle, for other CUDA libraries. The event keeps owning it.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        self.raw
    }

    /// Record the event in `stream` (`cuEventRecord`): it completes when the work queued
    /// there before it has.
    pub fn record(&self, stream: &CudaStream<'_>) -> Result<(), CudaError> {
        self.ctx.make_current()?;
        // SAFETY: a live event and stream of this context.
        check(&self.ctx.api, unsafe { (self.ctx.api.event_record)(self.raw, stream.raw) }, "cuEventRecord")
    }

    /// Whether the work before the event's last record has completed (`cuEventQuery`; an
    /// event never recorded is complete).
    pub fn is_complete(&self) -> Result<bool, CudaError> {
        self.ctx.make_current()?;
        // SAFETY: a live event.
        match unsafe { (self.ctx.api.event_query)(self.raw) } {
            0 => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            r => check(&self.ctx.api, r, "cuEventQuery").map(|_| true),
        }
    }

    /// Wait until the event completes (`cuEventSynchronize`).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx.make_current()?;
        // SAFETY: a live event.
        check(&self.ctx.api, unsafe { (self.ctx.api.event_synchronize)(self.raw) }, "cuEventSynchronize")
    }

    /// Milliseconds from `start` to this event (`cuEventElapsedTime`); both must have
    /// completed and have timing.
    pub fn elapsed_ms_since(&self, start: &CudaEvent<'_>) -> Result<f32, CudaError> {
        self.ctx.make_current()?;
        let mut ms = 0f32;
        // SAFETY: a valid out-pointer and two live events.
        check(
            &self.ctx.api,
            unsafe { (self.ctx.api.event_elapsed_time)(&mut ms, start.raw, self.raw) },
            "cuEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for CudaEvent<'_> {
    fn drop(&mut self) {
        let _ = self.ctx.make_current();
        // SAFETY: created by cuEventCreate and destroyed once (the driver defers the release
        // of an event still pending).
        unsafe { (self.ctx.api.event_destroy)(self.raw) };
    }
}

impl CudaStream<'_> {
    /// Make the work queued on this stream from now on wait until `event` completes
    /// (`cuStreamWaitEvent`), which orders work across streams without blocking the host.
    pub fn wait_event(&self, event: &CudaEvent<'_>) -> Result<(), CudaError> {
        self.ctx.make_current()?;
        // SAFETY: a live stream and event of this context.
        check(&self.ctx.api, unsafe { (self.ctx.api.stream_wait_event)(self.raw, event.raw, 0) }, "cuStreamWaitEvent")
    }
}
