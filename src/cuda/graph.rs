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
//! outlive every replay.

use std::ffi::c_int;

use super::{CuGraph, CuGraphExec, CudaContext, CudaError, CudaStream, check, err};

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
        Ok(CaptureGuard { stream: self, ended: false })
    }
}

impl<'s, 'ctx> CaptureGuard<'s, 'ctx> {
    /// The capturing stream: queue the work to capture on it.
    pub fn stream(&self) -> &'s CudaStream<'ctx> {
        self.stream
    }

    fn finish(&mut self) -> Result<CuGraph, CudaError> {
        self.ended = true;
        let end = self.stream.ctx.api.stream_end_capture.expect("checked when the capture began");
        self.stream.ctx.make_current()?;
        let mut graph: CuGraph = std::ptr::null_mut();
        // SAFETY: the stream is capturing; a valid out-pointer.
        let r = unsafe { end(self.stream.raw, &mut graph) };
        check(&self.stream.ctx.api, r, "cuStreamEndCapture")?;
        Ok(graph)
    }

    /// End the capture (`cuStreamEndCapture`) and return the graph.
    pub fn end(mut self) -> Result<CudaGraph<'ctx>, CudaError> {
        let raw = self.finish()?;
        Ok(CudaGraph { ctx: self.stream.ctx, raw })
    }
}

impl Drop for CaptureGuard<'_, '_> {
    fn drop(&mut self) {
        if !self.ended
            && let Ok(graph) = self.finish()
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
        Ok(CudaGraphExec { ctx: self.ctx, raw: exec })
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
}

impl CudaGraphExec<'_> {
    /// Queue one replay of the graph on `stream` (`cuGraphLaunch`). Replays on one stream run
    /// in order; [`CudaStream::synchronize`] waits for them.
    pub fn launch(&self, stream: &CudaStream<'_>) -> Result<(), CudaError> {
        let launch = self.ctx.api.graph_launch.ok_or_else(|| err("this CUDA driver has no cuGraphLaunch"))?;
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
