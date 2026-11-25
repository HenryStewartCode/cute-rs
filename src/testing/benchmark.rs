//! Benchmarking (CuTe DSL's `cutlass.testing.benchmark` and `get_workspace_count`): the average
//! device time of a launch, measured with a pair of CUDA events around many iterations, with
//! workspaces rotated so that each iteration reads memory the L2 cache does not hold, and CUDA graphs to take launch overhead out.
//!
//! The launch is a closure `|stream, workspace| -> Result<(), _>` that queues one iteration on
//! `stream` with workspace `workspace` (0 to [`BenchmarkOptions::workspace_count`] - 1): the
//! workspaces are the closure's own (one set of buffers each), or made by
//! [`benchmark_with_workspaces`] from a generator. Workspaces are used in turn, the timed
//! iterations continuing where the warmup stopped, as in CuTe DSL.
//!
//! What CuTe DSL also has and cute-rs does not: CUPTI timing (`use_cupti=True`, which times
//! the kernels alone, without the gaps between them) and NVTX ranges (`nvtx_tag=`); the
//! events here time everything queued between them, launch gaps included, which graph mode
//! mostly removes. [`LaunchStats`](crate::cuda::LaunchStats) (from `CudaContext::launch` and
//! `launch_on_device`) is different again: a host timer around one launch and the
//! synchronisation after it, so it includes launch latency and the wait.

/// How [`benchmark`] runs (CuTe DSL's `warmup_iterations=10`, `iterations=100`,
/// `workspace_count=1`, `use_cuda_graphs=False`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkOptions {
    /// Untimed iterations first.
    pub warmup_iterations: usize,
    /// Timed iterations; the result is their average.
    pub iterations: usize,
    /// Workspaces to rotate through (at least 1; see [`get_workspace_count`]).
    pub workspace_count: usize,
    /// Capture the warmup and the timed iterations into two CUDA graphs and time a replay of
    /// the second; needs a stream other than the legacy default stream, and launches that
    /// can be captured.
    pub use_cuda_graphs: bool,
}

impl Default for BenchmarkOptions {
    fn default() -> BenchmarkOptions {
        BenchmarkOptions { warmup_iterations: 10, iterations: 100, workspace_count: 1, use_cuda_graphs: false }
    }
}

impl BenchmarkOptions {
    pub fn new() -> BenchmarkOptions {
        BenchmarkOptions::default()
    }

    pub fn warmup_iterations(mut self, n: usize) -> Self {
        self.warmup_iterations = n;
        self
    }

    pub fn iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    pub fn workspace_count(mut self, n: usize) -> Self {
        self.workspace_count = n;
        self
    }

    pub fn use_cuda_graphs(mut self, on: bool) -> Self {
        self.use_cuda_graphs = on;
        self
    }
}

/// How many workspaces leave the L2 cache without the next one's data (CuTe DSL's
/// `get_workspace_count`): enough distinct workspaces of `one_workspace_bytes` to cover three
/// times the L2 cache of `l2_bytes`, plus one, but no more than there are iterations:
/// `min(warmup + iterations, l2_bytes * 3 / one_workspace_bytes + 1)`. A workspace of 0 bytes
/// needs only one.
pub fn workspace_count_for_l2(l2_bytes: u64, one_workspace_bytes: u64, warmup: usize, iterations: usize) -> usize {
    if one_workspace_bytes == 0 {
        return 1;
    }
    let needed = (l2_bytes.saturating_mul(3) / one_workspace_bytes).saturating_add(1);
    let iters = (warmup + iterations).max(1) as u64;
    needed.min(iters) as usize
}

#[cfg(feature = "cuda")]
pub use gpu::*;

#[cfg(feature = "cuda")]
mod gpu {
    use super::{BenchmarkOptions, workspace_count_for_l2};
    use crate::cuda::{CaptureMode, CudaContext, CudaError, CudaEvent, CudaStream, DeviceAttribute, EventFlags};

    /// Errors of a launch closure and of the benchmark itself.
    pub type BenchError = Box<dyn std::error::Error>;

    /// [`workspace_count_for_l2`] with the L2 cache size of `ctx`'s device
    /// (`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`; CuTe DSL's `get_workspace_count`).
    pub fn get_workspace_count(
        ctx: &CudaContext,
        one_workspace_bytes: usize,
        warmup_iterations: usize,
        iterations: usize,
    ) -> Result<usize, CudaError> {
        let l2 = ctx.device_attribute(DeviceAttribute::L2CacheSize)?.max(0) as u64;
        Ok(workspace_count_for_l2(l2, one_workspace_bytes as u64, warmup_iterations, iterations))
    }

    /// Queue `count` iterations from workspace `next`; the workspace after the last.
    fn run<'c>(
        stream: &CudaStream<'c>,
        launch: &mut dyn FnMut(&CudaStream<'c>, usize) -> Result<(), BenchError>,
        count: usize,
        mut next: usize,
        workspaces: usize,
    ) -> Result<usize, BenchError> {
        for _ in 0..count {
            launch(stream, next)?;
            next = (next + 1) % workspaces;
        }
        Ok(next)
    }

    /// Time `f` (which queues work on `stream`) with two events: milliseconds.
    fn timed<'c>(stream: &CudaStream<'c>, f: impl FnOnce() -> Result<(), BenchError>) -> Result<f64, BenchError> {
        let ctx = stream.context();
        let (start, end): (CudaEvent<'_>, CudaEvent<'_>) =
            (ctx.event(EventFlags::DEFAULT)?, ctx.event(EventFlags::DEFAULT)?);
        start.record(stream)?;
        f()?;
        end.record(stream)?;
        end.synchronize()?;
        Ok(end.elapsed_ms_since(&start)? as f64)
    }

    /// Run `launch` [`warmup_iterations`](BenchmarkOptions::warmup_iterations) times untimed
    /// and [`iterations`](BenchmarkOptions::iterations) times between two CUDA events on
    /// `stream`, rotating through the workspaces, and return the average microseconds per
    /// iteration (CuTe DSL's `cutlass.testing.benchmark`). With
    /// [`use_cuda_graphs`](BenchmarkOptions::use_cuda_graphs), the two runs are captured
    /// into graphs once (the closure is called the same number of times), the warmup graph
    /// replayed, and the replay of the timed graph measured. See the
    /// module documentation for what is timed.
    pub fn benchmark<'c>(
        stream: &CudaStream<'c>,
        options: &BenchmarkOptions,
        mut launch: impl FnMut(&CudaStream<'c>, usize) -> Result<(), BenchError>,
    ) -> Result<f64, BenchError> {
        let workspaces = options.workspace_count;
        if workspaces == 0 {
            return Err("benchmark: there must be a workspace (workspace_count is 0)".into());
        }
        if options.iterations == 0 {
            return Err("benchmark: there is nothing to time (iterations is 0)".into());
        }
        let ms = if options.use_cuda_graphs {
            if stream.is_default() {
                return Err("benchmark: CUDA graphs need a stream other than the legacy default stream \
                            (CudaContext::stream)"
                    .into());
            }
            let capture =
                |count: usize,
                 next: usize,
                 launch: &mut dyn FnMut(&CudaStream<'c>, usize) -> Result<(), BenchError>| {
                    let guard = stream.begin_capture(CaptureMode::ThreadLocal)?;
                    let next = run(stream, launch, count, next, workspaces)?;
                    let graph = guard.end()?;
                    Ok::<_, BenchError>((graph, next))
                };
            let (warmup, next) = capture(options.warmup_iterations, 0, &mut launch)?;
            let (profile, _) = capture(options.iterations, next, &mut launch)?;
            let (warmup, profile) = (warmup.instantiate()?, profile.instantiate()?);
            if options.warmup_iterations > 0 {
                warmup.launch(stream)?;
            }
            timed(stream, || Ok(profile.launch(stream)?))?
        } else {
            let next = run(stream, &mut launch, options.warmup_iterations, 0, workspaces)?;
            timed(stream, || run(stream, &mut launch, options.iterations, next, workspaces).map(|_| ()))?
        };
        Ok(ms / options.iterations as f64 * 1e3)
    }

    /// [`benchmark`] over workspaces made by `generate` (CuTe DSL's `workspace_generator`),
    /// called once for each of the [`workspace_count`](BenchmarkOptions::workspace_count)
    /// workspaces, before anything runs; `launch` gets the workspace of each iteration.
    pub fn benchmark_with_workspaces<'c, W>(
        stream: &CudaStream<'c>,
        options: &BenchmarkOptions,
        mut generate: impl FnMut(usize) -> Result<W, BenchError>,
        mut launch: impl FnMut(&CudaStream<'c>, &W) -> Result<(), BenchError>,
    ) -> Result<f64, BenchError> {
        let workspaces = (0..options.workspace_count).map(&mut generate).collect::<Result<Vec<W>, _>>()?;
        benchmark(stream, options, |s, i| launch(s, &workspaces[i]))
    }
}
