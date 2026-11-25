//! A simulated stream: kernels launched one after another over shared host buffers, with a
//! model of programmatic dependent launch (PDL).
//!
//! On a GPU, a launch with [`LaunchConfig::use_pdl`] may start while the previous kernel in
//! its stream is still running; [`griddepcontrol_wait`](crate::griddepcontrol_wait) blocks a
//! thread until that kernel has completed and its writes are visible. The model is
//! deterministic and conservative: the primary (the previous launch) runs to completion while
//! its global-memory footprint (the bytes it reads and writes) is recorded; then the
//! dependent runs, and an access a thread makes before its wait that conflicts with the
//! footprint (reading or writing what the primary wrote, or writing what it read) is
//! reported at the access's line, since on hardware that code may overlap any part of the
//! primary. Accesses after the wait see all of the primary's writes. A launch without
//! `use_pdl` is not checked, and a wait with no PDL primary is a no-op.

use super::{KernelArg, LaunchConfig, SimError, SimOptions, SimReport, exec};
use crate::builder::Kernel;
use crate::ir::Param;

/// A buffer of a [`SimStream`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SimBuffer(usize);

/// One argument of a [`SimStream::launch`]: one of the stream's buffers, or a value of the
/// launch's own (a scalar, or a buffer only this launch sees).
#[derive(Clone, Debug)]
pub enum StreamArg {
    Buffer(SimBuffer),
    Value(KernelArg),
}

impl From<SimBuffer> for StreamArg {
    fn from(b: SimBuffer) -> Self {
        StreamArg::Buffer(b)
    }
}

impl From<KernelArg> for StreamArg {
    fn from(a: KernelArg) -> Self {
        StreamArg::Value(a)
    }
}

/// Kernels launched in order on shared host buffers, with the PDL model described in the
/// module documentation.
///
/// ```
/// use cute::prelude::*;
///
/// let n = 64;
/// let producer = Kernel::builder("producer").block_dim([64, 1, 1]).build(|| {
///     let x = param_tensor::<f32>("x", Layout::col_major(64));
///     let i = thread_idx().x;
///     x.store(i, i.to::<f32>());
///     griddepcontrol_launch_dependents();
/// });
/// let consumer = Kernel::builder("consumer").block_dim([64, 1, 1]).build(|| {
///     let x = param_tensor::<f32>("x", Layout::col_major(64));
///     let y = param_tensor::<f32>("y", Layout::col_major(64));
///     let i = thread_idx().x;
///     griddepcontrol_wait();
///     y.store(i, x.load(i) * 2.0f32);
/// });
/// let mut stream = SimStream::new();
/// let x = stream.buffer(KernelArg::buffer(&vec![0.0f32; n]));
/// let y = stream.buffer(KernelArg::buffer(&vec![0.0f32; n]));
/// let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
/// stream.launch(&producer, cfg, &[x.into()]).unwrap();
/// stream.launch(&consumer, cfg.with_pdl(true), &[x.into(), y.into()]).unwrap();
/// assert_eq!(stream.get(y).to_vec::<f32>()[3], 6.0);
/// ```
#[derive(Default)]
pub struct SimStream {
    buffers: Vec<KernelArg>,
    opts: SimOptions,
    /// The previous launch: its kernel's name and its footprint per stream buffer.
    last: Option<(String, Vec<Option<Vec<u8>>>)>,
}

impl SimStream {
    /// An empty stream, simulating with the default options.
    pub fn new() -> SimStream {
        SimStream::default()
    }

    /// An empty stream whose launches simulate with `opts`.
    pub fn with_options(opts: SimOptions) -> SimStream {
        SimStream { opts, ..SimStream::default() }
    }

    /// Add a buffer (a buffer [`KernelArg`]) the launches share.
    #[track_caller]
    pub fn buffer(&mut self, data: KernelArg) -> SimBuffer {
        assert!(data.buffer_type().is_some(), "cute: SimStream::buffer: {} is not a buffer", data.describe());
        self.buffers.push(data);
        self.last.as_mut().into_iter().for_each(|(_, f)| f.push(None));
        SimBuffer(self.buffers.len() - 1)
    }

    /// A buffer's contents after the launches so far.
    pub fn get(&self, b: SimBuffer) -> &KernelArg {
        &self.buffers[b.0]
    }

    /// Replace a buffer's contents from the host (between launches: the next PDL launch
    /// does not see it as the previous kernel's write).
    #[track_caller]
    pub fn set(&mut self, b: SimBuffer, data: KernelArg) {
        assert!(
            data.buffer_type() == self.buffers[b.0].buffer_type(),
            "cute: SimStream::set: {} does not match the buffer's type and length",
            data.describe()
        );
        self.buffers[b.0] = data;
    }

    /// Run `kernel` after everything launched before. With [`LaunchConfig::use_pdl`], its
    /// accesses before [`griddepcontrol_wait`](crate::griddepcontrol_wait) are checked
    /// against the previous launch's footprint.
    pub fn launch(&mut self, kernel: &Kernel, cfg: LaunchConfig, args: &[StreamArg]) -> Result<SimReport, SimError> {
        let ir = kernel.ir();
        if args.len() != ir.args.len() {
            return Err(SimError::plain(format!(
                "kernel `{}` takes {} arguments, {} were given",
                kernel.name(),
                ir.args.len(),
                args.len()
            )));
        }
        let mut values: Vec<KernelArg> = args
            .iter()
            .map(|a| match a {
                StreamArg::Buffer(b) => self.buffers[b.0].clone(),
                StreamArg::Value(v) => v.clone(),
            })
            .collect();
        // Which stream buffer each of the kernel's buffers is.
        let mut of_buffer: Vec<Option<usize>> = vec![None; ir.buffers.len()];
        for (p, a) in ir.args.iter().zip(args) {
            if let (Param::Buffer(b), StreamArg::Buffer(s)) = (p, a) {
                of_buffer[b.0 as usize] = Some(s.0);
            }
        }
        let primary = match (&self.last, cfg.use_pdl) {
            (Some((name, footprint)), true) => Some(exec::PdlPrimary {
                name: name.clone(),
                footprint: of_buffer.iter().map(|s| s.and_then(|s| footprint[s].clone())).collect(),
            }),
            _ => None,
        };
        let (report, footprint) = kernel.sim_module().run(cfg, &mut values, &self.opts, Some(primary))?;
        let mut mine: Vec<Option<Vec<u8>>> = vec![None; self.buffers.len()];
        for (b, s) in of_buffer.iter().enumerate() {
            if let (Some(s), Some(Some(f))) = (s, footprint.get(b)) {
                match &mut mine[*s] {
                    Some(m) => m.iter_mut().zip(f).for_each(|(a, b)| *a |= b),
                    slot => *slot = Some(f.clone()),
                }
            }
        }
        for (v, a) in values.into_iter().zip(args) {
            if let StreamArg::Buffer(b) = a {
                self.buffers[b.0] = v;
            }
        }
        self.last = Some((kernel.name().to_string(), mine));
        Ok(report)
    }

    /// The buffers, in the order they were added.
    pub fn into_buffers(self) -> Vec<KernelArg> {
        self.buffers
    }
}
