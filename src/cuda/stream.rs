//! CUDA streams (CuTe DSL's `stream=` launch argument, a `CUstream` or `cudaStream_t`).

use std::ffi::{c_int, c_void};

use super::{CuStream, CudaContext, CudaError, check, err};

/// `CU_STREAM_NON_BLOCKING`: the stream does not synchronise with the legacy NULL stream.
const CU_STREAM_NON_BLOCKING: u32 = 1;
/// `CU_STREAM_CAPTURE_STATUS_*`.
const CAPTURE_STATUS_ACTIVE: c_int = 1;
const CAPTURE_STATUS_INVALIDATED: c_int = 2;

/// A CUDA stream: a queue of launches, copies and memsets that run in order. Work on
/// different streams may overlap.
///
/// [`CudaContext::stream`] creates one (`cuStreamCreate` with `CU_STREAM_NON_BLOCKING`,
/// destroyed when dropped); [`CudaContext::default_stream`] is the legacy NULL stream that
/// launches use when given none, as in CuTe DSL; [`CudaStream::from_raw`] borrows a stream
/// another library owns.
pub struct CudaStream<'ctx> {
    pub(super) ctx: &'ctx CudaContext,
    pub(super) raw: CuStream,
    owned: bool,
}

impl CudaContext {
    /// A new non-blocking stream (`cuStreamCreate(CU_STREAM_NON_BLOCKING)`), destroyed
    /// (`cuStreamDestroy`) when dropped.
    pub fn stream(&self) -> Result<CudaStream<'_>, CudaError> {
        self.make_current()?;
        let mut raw: CuStream = std::ptr::null_mut();
        // SAFETY: a valid out-pointer.
        check(&self.api, unsafe { (self.api.stream_create)(&mut raw, CU_STREAM_NON_BLOCKING) }, "cuStreamCreate")?;
        Ok(CudaStream { ctx: self, raw, owned: true })
    }

    /// The legacy NULL stream (stream 0), which synchronises with every blocking stream of
    /// the context. It cannot be captured into a graph.
    pub fn default_stream(&self) -> CudaStream<'_> {
        CudaStream { ctx: self, raw: std::ptr::null_mut(), owned: false }
    }
}

impl<'ctx> CudaStream<'ctx> {
    /// Use a stream created elsewhere (by PyTorch, cuBLAS, a `cudaStream_t`): it is not
    /// destroyed when this is dropped.
    ///
    /// # Safety
    ///
    /// `raw` must be a live stream of `ctx`'s context (or null, the legacy stream), and must
    /// outlive the returned value.
    pub unsafe fn from_raw(ctx: &'ctx CudaContext, raw: *mut c_void) -> CudaStream<'ctx> {
        CudaStream { ctx, raw, owned: false }
    }

    /// The `CUstream` handle, for other CUDA libraries. The stream keeps owning it.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// The context the stream belongs to.
    pub fn context(&self) -> &'ctx CudaContext {
        self.ctx
    }

    /// Whether this is the legacy NULL stream.
    pub fn is_default(&self) -> bool {
        self.raw.is_null()
    }

    /// Wait for everything queued on the stream (`cuStreamSynchronize`). A kernel's failure
    /// (an illegal address, a device assertion) is reported here.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx.make_current()?;
        // SAFETY: a live stream of the current context.
        check(&self.ctx.api, unsafe { (self.ctx.api.stream_synchronize)(self.raw) }, "the stream's work")
    }

    /// Whether everything queued so far has finished (`cuStreamQuery`), without waiting.
    pub fn is_idle(&self) -> Result<bool, CudaError> {
        const CUDA_ERROR_NOT_READY: c_int = 600;
        self.ctx.make_current()?;
        // SAFETY: a live stream.
        match unsafe { (self.ctx.api.stream_query)(self.raw) } {
            0 => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            r => check(&self.ctx.api, r, "cuStreamQuery").map(|_| true),
        }
    }

    /// Whether the stream is capturing a graph (`cuStreamIsCapturing`). An invalidated
    /// capture is an error that says why.
    pub fn is_capturing(&self) -> Result<bool, CudaError> {
        self.ctx.make_current()?;
        let mut status: c_int = 0;
        // SAFETY: a valid out-pointer and a live stream.
        check(
            &self.ctx.api,
            unsafe { (self.ctx.api.stream_is_capturing)(self.raw, &mut status) },
            "cuStreamIsCapturing",
        )?;
        match status {
            CAPTURE_STATUS_ACTIVE => Ok(true),
            CAPTURE_STATUS_INVALIDATED => {
                Err(err("the stream's graph capture was invalidated by an operation that cannot be captured \
                 (CU_STREAM_CAPTURE_STATUS_INVALIDATED); end the capture and start again"))
            }
            _ => Ok(false),
        }
    }
}

impl Drop for CudaStream<'_> {
    fn drop(&mut self) {
        if self.owned {
            let _ = self.ctx.make_current();
            // SAFETY: created by cuStreamCreate and destroyed once; pending work completes
            // first (the driver defers the release).
            unsafe { (self.ctx.api.stream_destroy)(self.raw) };
        }
    }
}
