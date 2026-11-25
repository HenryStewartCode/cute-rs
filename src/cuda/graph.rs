//! CUDA graphs from stream capture: record a sequence of launches and copies once, then
//! replay it with one call (`torch.cuda.graph()` around a CuTe DSL function in CuTe's
//! `cuda_graphs` notebook, and `cutlass.testing.benchmark(use_cuda_graphs=True)`).
//!
//! Only stream-ordered work can be captured: launches on the capturing stream
//! ([`LaunchBuilder::stream`](super::LaunchBuilder::stream)), device-to-device copies and
//! memsets. Calls that synchronise or allocate (`CudaContext::launch`, which uploads and
//! waits; `alloc`; `synchronize`) fail with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` or
//! invalidate the capture (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`). Setting a function
//! attribute (`cuFuncSetAttribute`, which launches above 48 KB of shared memory need) is
//! allowed.
//!
//! A graph holds the device addresses and argument values it captured: the buffers must
//! outlive every replay. For [`DeviceBuffer`](super::DeviceBuffer)s this is checked: a capture
//! remembers the buffers its launches, copies and memsets use, and
//! [`CudaGraphExec::launch`] fails once one of them has been dropped. Raw addresses
//! ([`DevicePtr`](super::DevicePtr)) and external memory are not tracked: keeping them alive
//! is up to the caller.

use std::collections::HashMap;
use std::ffi::c_int;

use super::{CuDevicePtr, CuGraph, CuGraphExec, CuStream, CudaContext, CudaError, CudaStream, check, err};

/// A captured buffer: its address and its allocation's serial number (an address can be
/// allocated again after a free).
type Captured = (CuDevicePtr, u64);

/// What a context remembers so that a graph replay can refuse buffers freed since the
/// capture: the live allocations, and the buffers each capturing stream has used so far.
#[derive(Default)]
pub(super) struct Tracking {
    serial: u64,
    live: HashMap<CuDevicePtr, u64>,
    captures: HashMap<usize, Vec<Captured>>,
}

impl Tracking {
    pub(super) fn alloc(&mut self, ptr: CuDevicePtr) {
        self.serial += 1;
        self.live.insert(ptr, self.serial);
    }

    pub(super) fn free(&mut self, ptr: CuDevicePtr) {
        self.live.remove(&ptr);
    }

    /// Note that work using the buffers at `ptrs` was queued on `stream`, if it is capturing.
    fn used(&mut self, stream: usize, ptrs: impl IntoIterator<Item = CuDevicePtr>) {
        let Some(log) = self.captures.get_mut(&stream) else { return };
        for ptr in ptrs {
            if let Some(&serial) = self.live.get(&ptr)
                && !log.contains(&(ptr, serial))
            {
                log.push((ptr, serial));
            }
        }
    }

    /// The first of `buffers` that has been freed since it was captured.
    fn first_freed(&self, buffers: &[Captured]) -> Option<CuDevicePtr> {
        buffers.iter().find(|(ptr, serial)| self.live.get(ptr) != Some(serial)).map(|&(ptr, _)| ptr)
    }
}

impl CudaContext {
    /// Note that work using the device buffers at `ptrs` was queued on `stream` (kept only
    /// while the stream is capturing).
    pub(super) fn note_captured(&self, stream: CuStream, ptrs: impl IntoIterator<Item = CuDevicePtr>) {
        self.tracking.borrow_mut().used(stream as usize, ptrs);
    }
}

/// How a capture treats calls from other threads that cannot be captured
/// (`CUstreamCaptureMode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CaptureMode {
    /// Any thread's unsafe call fails while this capture is active (the default, and
    /// PyTorch's).
    #[default]
    Global = 0,
    /// Only this thread's unsafe calls fail.
    ThreadLocal = 1,
    /// No call is refused for being potentially unsafe.
    Relaxed = 2,
}

/// An active stream capture (`cuStreamBeginCapture`). [`end`](CaptureGuard::end) returns
/// the captured graph; dropping the guard ends the capture and discards the graph, so a
/// failure in between leaves the stream usable.
pub struct CaptureGuard<'s, 'ctx> {
    stream: &'s CudaStream<'ctx>,
    ended: bool,
}

impl<'ctx> CudaStream<'ctx> {
    /// Start capturing the work queued on this stream into a graph. The legacy NULL stream
    /// cannot be captured.
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<CaptureGuard<'_, 'ctx>, CudaError> {
        if self.is_default() {
            return Err(err(
                "the legacy NULL stream cannot be captured into a graph: capture a stream from CudaContext::stream",
            ));
        }
        let begin = self.ctx.api.stream_begin_capture.ok_or_else(|| err("this CUDA driver has no stream capture"))?;
        self.ctx.make_current()?;
        // SAFETY: a live, non-NULL stream.
        check(&self.ctx.api, unsafe { begin(self.raw, mode as c_int) }, "cuStreamBeginCapture")?;
        self.ctx.tracking.borrow_mut().captures.insert(self.raw as usize, Vec::new());
        Ok(CaptureGuard { stream: self, ended: false })
    }
}

impl<'s, 'ctx> CaptureGuard<'s, 'ctx> {
    /// The capturing stream: queue the work to capture on it.
    pub fn stream(&self) -> &'s CudaStream<'ctx> {
        self.stream
    }

    /// The graph, and the buffers the capture used.
    fn finish(&mut self) -> Result<(CuGraph, Vec<Captured>), CudaError> {
        self.ended = true;
        let ctx = self.stream.ctx;
        let buffers = ctx.tracking.borrow_mut().captures.remove(&(self.stream.raw as usize)).unwrap_or_default();
        let end = self.stream.ctx.api.stream_end_capture.expect("checked when the capture began");
        self.stream.ctx.make_current()?;
        let mut graph: CuGraph = std::ptr::null_mut();
        // SAFETY: the stream is capturing; a valid out-pointer.
        let r = unsafe { end(self.stream.raw, &mut graph) };
        check(&self.stream.ctx.api, r, "cuStreamEndCapture")?;
        Ok((graph, buffers))
    }

    /// End the capture (`cuStreamEndCapture`) and return the graph.
    pub fn end(mut self) -> Result<CudaGraph<'ctx>, CudaError> {
        let (raw, buffers) = self.finish()?;
        Ok(CudaGraph { ctx: self.stream.ctx, raw, buffers })
    }
}

impl Drop for CaptureGuard<'_, '_> {
    fn drop(&mut self) {
        if !self.ended
            && let Ok((graph, _)) = self.finish()
            && !graph.is_null()
            && let Some(destroy) = self.stream.ctx.api.graph_destroy
        {
            // SAFETY: the graph came from cuStreamEndCapture and is not used again.
            unsafe { destroy(graph) };
        }
    }
}

/// A captured graph (`CUgraph`), destroyed when dropped. [`instantiate`](Self::instantiate)
/// makes it launchable.
pub struct CudaGraph<'ctx> {
    ctx: &'ctx CudaContext,
    raw: CuGraph,
    buffers: Vec<Captured>,
}

impl<'ctx> CudaGraph<'ctx> {
    /// Make an executable graph (`cuGraphInstantiateWithFlags`, no flags).
    pub fn instantiate(&self) -> Result<CudaGraphExec<'ctx>, CudaError> {
        let inst =
            self.ctx.api.graph_instantiate.ok_or_else(|| err("this CUDA driver has no cuGraphInstantiateWithFlags"))?;
        self.ctx.make_current()?;
        let mut exec: CuGraphExec = std::ptr::null_mut();
        // SAFETY: a live graph and a valid out-pointer.
        check(&self.ctx.api, unsafe { inst(&mut exec, self.raw, 0) }, "cuGraphInstantiateWithFlags")?;
        Ok(CudaGraphExec { ctx: self.ctx, raw: exec, buffers: self.buffers.clone() })
    }

    /// The number of nodes in the graph (`cuGraphGetNodes`): one per captured launch,
    /// copy or memset.
    pub fn node_count(&self) -> Result<usize, CudaError> {
        let nodes = self.ctx.api.graph_get_nodes.ok_or_else(|| err("this CUDA driver has no cuGraphGetNodes"))?;
        let mut n = 0usize;
        // SAFETY: a null node array asks for the count only.
        check(&self.ctx.api, unsafe { nodes(self.raw, std::ptr::null_mut(), &mut n) }, "cuGraphGetNodes")?;
        Ok(n)
    }

    /// The `CUgraph` handle; the graph keeps owning it.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        self.raw
    }
}

impl Drop for CudaGraph<'_> {
    fn drop(&mut self) {
        if let Some(destroy) = self.ctx.api.graph_destroy {
            let _ = self.ctx.make_current();
            // SAFETY: destroyed once.
            unsafe { destroy(self.raw) };
        }
    }
}

/// An executable graph (`CUgraphExec`), destroyed when dropped.
pub struct CudaGraphExec<'ctx> {
    ctx: &'ctx CudaContext,
    raw: CuGraphExec,
    buffers: Vec<Captured>,
}

impl CudaGraphExec<'_> {
    /// Queue one replay of the graph on `stream` (`cuGraphLaunch`). Replays on one stream run
    /// in order; [`CudaStream::synchronize`] waits for them. Fails, queueing nothing, if a
    /// device buffer the graph captured has been dropped.
    pub fn launch(&self, stream: &CudaStream<'_>) -> Result<(), CudaError> {
        let launch = self.ctx.api.graph_launch.ok_or_else(|| err("this CUDA driver has no cuGraphLaunch"))?;
        {
            let mut tracking = self.ctx.tracking.borrow_mut();
            if let Some(ptr) = tracking.first_freed(&self.buffers) {
                return Err(err(format!(
                    "this graph captured work on a device buffer (at {ptr:#x}) that has been freed since: the \
                     buffers of a graph must outlive every replay"
                )));
            }
            // A replay captured into another graph hands its buffers on.
            if let Some(log) = tracking.captures.get_mut(&(stream.raw as usize)) {
                log.extend_from_slice(&self.buffers);
            }
        }
        self.ctx.make_current()?;
        // SAFETY: a live executable graph and stream.
        check(&self.ctx.api, unsafe { launch(self.raw, stream.raw) }, "cuGraphLaunch")
    }

    /// The `CUgraphExec` handle; this keeps owning it.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        self.raw
    }
}

impl Drop for CudaGraphExec<'_> {
    fn drop(&mut self) {
        if let Some(destroy) = self.ctx.api.graph_exec_destroy {
            let _ = self.ctx.make_current();
            // SAFETY: destroyed once.
            unsafe { destroy(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tracking;

    #[test]
    fn a_replay_sees_buffers_freed_since_the_capture() {
        let mut t = Tracking::default();
        t.alloc(0x1000);
        t.alloc(0x2000);
        t.used(7, [0x1000]); // stream 7 is not capturing yet
        t.captures.insert(7, Vec::new());
        t.used(7, [0x1000, 0x1000]);
        t.used(8, [0x2000]); // another stream
        let buffers = t.captures.remove(&7).unwrap();
        assert_eq!(buffers.len(), 1);
        assert_eq!(t.first_freed(&buffers), None);
        t.free(0x2000);
        assert_eq!(t.first_freed(&buffers), None);
        t.free(0x1000);
        assert_eq!(t.first_freed(&buffers), Some(0x1000));
        // The address allocated again is not the captured buffer.
        t.alloc(0x1000);
        assert_eq!(t.first_freed(&buffers), Some(0x1000));
    }
}
