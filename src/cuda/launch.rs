//! Launching: the one path every launch takes ([`LaunchBuilder`]), from a function, a
//! configuration, a stream and device arguments, packed by the kernel's
//! [`KernelSignature`].

use std::borrow::Cow;
use std::ffi::{c_int, c_uint};

use super::{
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, CuEvent, CuStream,
    CudaContext, CudaError, CudaEvent, CudaFunction, CudaStream, DeviceArg, DeviceAttribute, FuncAttribute,
    LaunchAttribute, LaunchExConfig, check, err,
};
use crate::builder::Kernel;
use crate::log::Level;
use crate::signature::{KernelSignature, LaunchValue};
use crate::sim::LaunchConfig;
use crate::validate::MAX_STATIC_SHARED_BYTES;

/// `CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION` (CUDA 12.8; SM100).
const CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION: c_int = 11;
/// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`: every block resident at once, or the launch fails.
const CU_LAUNCH_ATTRIBUTE_COOPERATIVE: c_int = 2;
/// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`: programmatic dependent launch.
const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: c_int = 6;
/// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT`: `{ CUevent event; int flags; int triggerAtBlockStart; }`.
const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT: c_int = 7;
/// `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`: `{ CUevent event; int flags; }`.
const CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT: c_int = 12;

/// The events a launch records ([`LaunchBuilder::launch_completion_event`],
/// [`LaunchBuilder::programmatic_event`]).
#[derive(Clone, Copy, Default)]
pub(super) struct LaunchEvents {
    /// The event and its record flags.
    completion: Option<(CuEvent, i32)>,
    /// The event, its record flags and `triggerAtBlockStart`.
    programmatic: Option<(CuEvent, i32, bool)>,
}

impl LaunchEvents {
    fn any(&self) -> bool {
        self.completion.is_some() || self.programmatic.is_some()
    }
}

/// A launch attribute holding `words` (32-bit words of its value union).
fn attribute(id: c_int, words: &[u32]) -> LaunchAttribute {
    let mut a = LaunchAttribute { id, pad: 0, value: [0; 16] };
    a.value[..words.len()].copy_from_slice(words);
    a
}

/// A launch attribute holding an event (a pointer, at the start of the 8-byte-aligned value
/// union) followed by `ints`.
fn event_attribute(id: c_int, event: CuEvent, ints: &[i32]) -> LaunchAttribute {
    let p = event as usize as u64;
    let mut words = vec![p as u32, (p >> 32) as u32];
    words.extend(ints.iter().map(|&i| i as u32));
    attribute(id, &words)
}

/// One launch, built up and then [`submit`](Self::submit)ted: the function, its
/// configuration, the stream (the legacy NULL stream unless [`stream`](Self::stream) names
/// one, as in CuTe DSL) and the device arguments. Submitting queues the kernel and returns;
/// nothing is allocated or synchronised, so a launch on a capturing stream is captured.
pub struct LaunchBuilder<'a> {
    ctx: &'a CudaContext,
    function: &'a CudaFunction<'a>,
    sig: Cow<'a, KernelSignature>,
    cfg: LaunchConfig,
    stream: CuStream,
    args: &'a [DeviceArg<'a>],
    events: LaunchEvents,
    /// The kernel's device workspace, when launched with the kernel (m11-cute-ext-tma).
    workspace: Option<crate::experimental::host_runtime::WorkspaceDecl>,
}

impl CudaContext {
    /// A launch of `function`, compiled from `kernel`.
    pub fn launch_builder<'a>(
        &'a self,
        kernel: &'a Kernel,
        function: &'a CudaFunction<'a>,
        cfg: LaunchConfig,
    ) -> LaunchBuilder<'a> {
        LaunchBuilder {
            ctx: self,
            function,
            sig: Cow::Owned(kernel.signature()),
            cfg,
            stream: std::ptr::null_mut(),
            args: &[],
            events: LaunchEvents::default(),
            workspace: kernel.ir().workspace.clone(),
        }
    }
}

impl<'f> CudaFunction<'f> {
    /// A launch of this function, by the signature of the compiled kernel it was loaded
    /// from ([`CudaContext::load`]). A function loaded from a bare image has none: use
    /// [`CudaContext::launch_builder`] with the kernel.
    pub fn launch_builder<'a>(&'a self, cfg: LaunchConfig) -> Result<LaunchBuilder<'a>, CudaError>
    where
        'f: 'a,
    {
        let sig = self.signature.as_ref().ok_or_else(|| {
            err("this function was loaded from a bare image, so it has no kernel signature: use \
                 CudaContext::launch_builder with the kernel")
        })?;
        Ok(LaunchBuilder {
            ctx: self.ctx,
            function: self,
            sig: Cow::Borrowed(sig),
            cfg,
            stream: std::ptr::null_mut(),
            args: &[],
            events: LaunchEvents::default(),
            workspace: None,
        })
    }
}

impl<'a> LaunchBuilder<'a> {
    /// Queue the launch on `stream`.
    pub fn stream(mut self, stream: &'a CudaStream<'_>) -> Self {
        self.stream = stream.raw;
        self
    }

    /// The device arguments, one per kernel parameter.
    pub fn args(mut self, args: &'a [DeviceArg<'a>]) -> Self {
        self.args = args;
        self
    }

    /// Record `event` when every block of the launch has started
    /// (`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`; CuTe DSL's `launch_completion_event=`
    /// and `launch_completion_event_flags=`, `cuEventRecordWithFlags` flags), so work waiting
    /// on it may start while the kernel still runs. The event must have timing disabled
    /// ([`EventFlags::DISABLE_TIMING`](super::EventFlags::DISABLE_TIMING)): driver 595 refuses
    /// a timing event with `CUDA_ERROR_NOT_SUPPORTED`. Accepted on SM89 as well as SM90.
    pub fn launch_completion_event(mut self, event: &'a CudaEvent<'_>, flags: i32) -> Self {
        self.events.completion = Some((event.raw, flags));
        self
    }

    /// Record `event` once every block has triggered it with
    /// [`griddepcontrol_launch_dependents`](crate::griddepcontrol_launch_dependents) or
    /// exited, or as each block starts with `trigger_at_block_start`
    /// (`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT`; CuTe DSL's `programmatic_event=`,
    /// `programmatic_event_flags=` and `programmatic_event_trigger_at_block_start=`). The
    /// trigger needs SM90; on SM89 driver 595 accepts the attribute and records the event as
    /// the blocks exit. The event must have timing disabled.
    pub fn programmatic_event(mut self, event: &'a CudaEvent<'_>, flags: i32, trigger_at_block_start: bool) -> Self {
        self.events.programmatic = Some((event.raw, flags, trigger_at_block_start));
        self
    }

    /// The kernel's signature, which the arguments are checked and packed by.
    pub fn signature(&self) -> &KernelSignature {
        &self.sig
    }

    /// Check and queue the launch (`cuLaunchKernelEx`, or `cuLaunchKernel` on old drivers).
    pub fn submit(self) -> Result<(), CudaError> {
        if let Some(ws) = &self.workspace {
            let scalar = |i: usize| match self.args.get(i) {
                Some(DeviceArg::Scalar(a)) => crate::experimental::host_runtime::scalar_arg(a),
                _ => None,
            };
            let bytes = match self.args.get(ws.param) {
                Some(
                    a @ (DeviceArg::Buffer(_) | DeviceArg::View { .. } | DeviceArg::Slice { .. } | DeviceArg::Ptr(_)),
                ) => match launch_value(ws.param, a, self.sig.params.get(ws.param))? {
                    LaunchValue::Pointer { bytes, .. } => bytes.map(|b| b as u64),
                    _ => None,
                },
                _ => None,
            };
            ws.check_launch(&self.sig.name, &self.cfg, self.args.len(), &scalar, bytes).map_err(err)?;
        }
        let (ctx, stream, args) = (self.ctx, self.stream, self.args);
        submit(ctx, self.function, &self.sig, self.cfg, stream, args, self.events)?;
        // A capture remembers the buffers its launches use, so that a replay can refuse freed ones.
        ctx.note_captured(
            stream,
            args.iter().filter_map(|a| match a {
                DeviceArg::Buffer(buffer) | DeviceArg::View { buffer, .. } | DeviceArg::Slice { buffer, .. } => {
                    Some(buffer.ptr)
                }
                _ => None,
            }),
        );
        Ok(())
    }
}

/// The launch value of a device argument for the parameter `param`.
fn launch_value<'v>(
    i: usize,
    arg: &'v DeviceArg<'_>,
    param: Option<&crate::signature::ParamSignature>,
) -> Result<LaunchValue<'v>, CudaError> {
    Ok(match arg {
        DeviceArg::Slice { buffer, byte_offset, bytes } => {
            if byte_offset + bytes > buffer.bytes {
                return Err(err(format!(
                    "argument {i}: a view of {bytes} bytes at byte {byte_offset} of a buffer of {} bytes",
                    buffer.bytes
                )));
            }
            LaunchValue::Pointer { address: buffer.ptr + *byte_offset as u64, bytes: Some(*bytes) }
        }
        DeviceArg::Ptr(p) => {
            if let Some(crate::signature::ParamSignature {
                name,
                kind: crate::signature::ParamKind::Pointer { elem, .. },
            }) = param
            {
                let mine = p.spec().dtype;
                if mine != *elem && crate::tensor::storage_decl(mine, 0).0 != *elem {
                    return Err(err(format!(
                        "argument {i} (`{name}`) is a pointer to {elem}, but the raw pointer given is to {mine}"
                    )));
                }
            }
            LaunchValue::Pointer { address: p.address(), bytes: p.bytes() }
        }
        DeviceArg::Buffer(b) => LaunchValue::Pointer { address: b.ptr, bytes: Some(b.bytes) },
        DeviceArg::View { buffer, byte_offset } => {
            if *byte_offset > buffer.bytes {
                return Err(err(format!(
                    "argument {i}: a view at byte {byte_offset} of a buffer of {} bytes",
                    buffer.bytes
                )));
            }
            LaunchValue::Pointer { address: buffer.ptr + *byte_offset as u64, bytes: Some(buffer.bytes - byte_offset) }
        }
        DeviceArg::Scalar(s) => LaunchValue::Scalar(s),
        DeviceArg::ByValue(bytes) => LaunchValue::Bytes(bytes),
    })
}

/// The carveout that fits `blocks` blocks of `block_bytes` shared memory on an SM, in
/// percent of the SM's shared memory.
fn carveout_for(blocks: u32, block_bytes: usize, reserved: usize, per_sm: usize) -> i32 {
    if per_sm == 0 {
        return -1;
    }
    let need = blocks as usize * (block_bytes + reserved);
    (need * 100).div_ceil(per_sm).min(100) as i32
}

pub(super) fn submit(
    ctx: &CudaContext,
    function: &CudaFunction<'_>,
    sig: &KernelSignature,
    cfg: LaunchConfig,
    stream: CuStream,
    args: &[DeviceArg<'_>],
    events: LaunchEvents,
) -> Result<(), CudaError> {
    let device = ctx.name();
    // A block or grid that no GPU runs is left to the driver, whose error names it
    // (CUDA_ERROR_INVALID_VALUE); everything the kernel itself requires is checked here.
    let plan = if (1..=1024).contains(&cfg.threads_per_block()) && !cfg.grid.contains(&0) {
        sig.check_config(&cfg, ctx.max_shared_memory_per_block(), Some(&device)).map_err(err)?
    } else {
        sig.smem_plan(&cfg).map_err(err)?
    };
    let clustered = cfg.cluster != [1, 1, 1] || cfg.fallback_cluster.is_some();
    if clustered {
        if ctx.device_attribute(DeviceAttribute::ClusterLaunch).unwrap_or(0) == 0 {
            let (major, minor) = ctx.compute_capability();
            return Err(err(format!(
                "kernel `{}` is launched with clusters of {:?}, but {device} (sm_{major}{minor}) cannot launch \
                 thread block clusters (CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH is 0; clusters need SM90 or newer)",
                sig.name, cfg.cluster
            )));
        }
        if cfg.fallback_cluster.is_some() && ctx.compute_capability().0 < 10 {
            return Err(err(format!(
                "kernel `{}` is launched with a preferred cluster of {:?} and a fallback cluster: preferred \
                 cluster shapes need SM100 or newer, and {device} is sm_{}{}",
                sig.name,
                cfg.cluster,
                ctx.compute_capability().0,
                ctx.compute_capability().1
            )));
        }
    }
    let mut values =
        args.iter().enumerate().map(|(i, a)| launch_value(i, a, sig.params.get(i))).collect::<Result<Vec<_>, _>>()?;
    // Tensor maps: hidden ones are computed from the tensor's device address and the scalar
    // arguments, host-bound ones re-encoded; both by the driver.
    let maps = if sig.tensor_maps.is_empty() {
        Vec::new()
    } else {
        if values.len() != sig.user_param_count() {
            return Err(err(format!(
                "kernel `{}` takes {} arguments, {} were given",
                sig.name,
                sig.user_param_count(),
                values.len()
            )));
        }
        super::tensor_map_encode::launch_tensor_maps(ctx, sig, &values)?
    };
    for (param, enc) in &maps {
        if *param < values.len() {
            values[*param] = LaunchValue::Bytes(&enc.0);
        } else {
            values.push(LaunchValue::Bytes(&enc.0));
        }
    }
    let mut packed = sig.pack(&values).map_err(err)?;
    let mut params = packed.pointers();
    ctx.make_current()?;

    if plan.block_bytes > MAX_STATIC_SHARED_BYTES {
        function.set_attribute_raw(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, plan.launch_bytes as c_int)?;
    }
    let carveout = match (cfg.preferred_smem_carveout, sig.min_blocks_per_sm) {
        (Some(pct), _) => Some(pct),
        (None, Some(n)) if n > 1 => {
            let reserved = ctx.device_attribute(DeviceAttribute::ReservedSharedMemoryPerBlock).unwrap_or(0) as usize;
            let per_sm = ctx.device_attribute(DeviceAttribute::MaxSharedMemoryPerMultiprocessor)? as usize;
            Some(carveout_for(n, plan.block_bytes, reserved, per_sm))
        }
        _ => None,
    };
    if let Some(pct) = carveout {
        function.set_attribute(FuncAttribute::PreferredSharedMemoryCarveout, pct)?;
    }
    if cfg.cooperative {
        check_cooperative(ctx, function, sig, &cfg, plan.launch_bytes)?;
    }

    crate::cute_log!(
        Level::Debug,
        "launching `{}`: grid {:?}, block {:?}, dynamic shared memory {} bytes ({} per block), stream {:?}, \
         cluster {:?}{}",
        sig.name,
        cfg.grid,
        cfg.block,
        plan.launch_bytes,
        plan.block_bytes,
        stream,
        cfg.cluster,
        cfg.fallback_cluster.map_or(String::new(), |f| format!(" (fallback {f:?})"))
    );
    let api = &ctx.api;
    match api.launch_kernel_ex {
        Some(launch_ex) => {
            let mut attrs: Vec<LaunchAttribute> = Vec::new();
            if clustered {
                let mut a = LaunchAttribute { id: CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, pad: 0, value: [0; 16] };
                a.value[..3].copy_from_slice(&cfg.fallback_cluster.unwrap_or(cfg.cluster));
                attrs.push(a);
            }
            if cfg.fallback_cluster.is_some() {
                let mut a =
                    LaunchAttribute { id: CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, pad: 0, value: [0; 16] };
                a.value[..3].copy_from_slice(&cfg.cluster);
                attrs.push(a);
            }
            if cfg.cooperative {
                attrs.push(attribute(CU_LAUNCH_ATTRIBUTE_COOPERATIVE, &[1]));
            }
            if cfg.use_pdl {
                attrs.push(attribute(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, &[1]));
            }
            if let Some((event, flags, at_start)) = events.programmatic {
                attrs.push(event_attribute(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT, event, &[flags, at_start as i32]));
            }
            if let Some((event, flags)) = events.completion {
                attrs.push(event_attribute(CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT, event, &[flags]));
            }
            let config = LaunchExConfig {
                grid: cfg.grid,
                block: cfg.block,
                shared_mem_bytes: plan.launch_bytes as c_uint,
                stream,
                attrs: if attrs.is_empty() { std::ptr::null_mut() } else { attrs.as_mut_ptr() },
                num_attrs: attrs.len() as c_uint,
            };
            // SAFETY: `config`, `attrs` and `params` outlive the call, which copies the
            // argument values.
            let r = unsafe { launch_ex(&config, function.function, params.as_mut_ptr(), std::ptr::null_mut()) };
            check(api, r, "cuLaunchKernelEx")
        }
        None if clustered || cfg.cooperative || cfg.use_pdl || events.any() => Err(err(
            "this driver has no cuLaunchKernelEx, which block clusters, cooperative and programmatic dependent \
             launches and launch events need",
        )),
        None => {
            let ([gx, gy, gz], [bx, by, bz]) = (cfg.grid, cfg.block);
            // SAFETY: params point at live storage matching the signature.
            let r = unsafe {
                (api.launch_kernel)(
                    function.function,
                    gx,
                    gy,
                    gz,
                    bx,
                    by,
                    bz,
                    plan.launch_bytes as c_uint,
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            check(api, r, "cuLaunchKernel")
        }
    }
}

/// A cooperative launch's requirements, checked before the driver's
/// `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`: the device supports it, and every block of the
/// grid fits on the device at once (the occupancy of this block size and dynamic shared
/// memory times the SM count).
fn check_cooperative(
    ctx: &CudaContext,
    function: &CudaFunction<'_>,
    sig: &KernelSignature,
    cfg: &LaunchConfig,
    dynamic_smem: usize,
) -> Result<(), CudaError> {
    let device = ctx.name();
    if ctx.device_attribute(DeviceAttribute::CooperativeLaunch).unwrap_or(0) == 0 {
        return Err(err(format!(
            "kernel `{}` is launched cooperatively, but {device} cannot launch cooperative kernels \
             (CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH is 0)",
            sig.name
        )));
    }
    let threads = cfg.threads_per_block() as u32;
    let per_sm = function.max_active_blocks_per_multiprocessor(threads, dynamic_smem)?;
    let sms = ctx.device_attribute(DeviceAttribute::MultiprocessorCount)? as u64;
    let blocks: u64 = cfg.grid.iter().map(|&g| g as u64).product();
    let fit = per_sm as u64 * sms;
    if blocks > fit {
        return Err(err(format!(
            "kernel `{}` is launched cooperatively with {blocks} blocks (grid {:?}), but at most {fit} can be \
             resident at once on {device}: {per_sm} block(s) of {threads} threads and {dynamic_smem} bytes of \
             dynamic shared memory per SM (cuOccupancyMaxActiveBlocksPerMultiprocessor) times {sms} SMs; a \
             cooperative launch needs every block resident, so use a smaller grid (a persistent kernel that loops \
             over the work)",
            sig.name, cfg.grid
        )));
    }
    Ok(())
}

impl CudaFunction<'_> {
    /// How many blocks of `block_size` threads with `dynamic_smem` bytes of dynamic shared
    /// memory fit on one SM at once (`cuOccupancyMaxActiveBlocksPerMultiprocessor`).
    pub fn max_active_blocks_per_multiprocessor(&self, block_size: u32, dynamic_smem: usize) -> Result<u32, CudaError> {
        self.ctx.make_current()?;
        let mut n: c_int = 0;
        // SAFETY: a valid out-pointer and a loaded function.
        let r = unsafe {
            (self.ctx.api.occupancy_max_active_blocks)(&mut n, self.function, block_size as c_int, dynamic_smem)
        };
        check(&self.ctx.api, r, "cuOccupancyMaxActiveBlocksPerMultiprocessor")?;
        Ok(n.max(0) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::{carveout_for, event_attribute};

    #[test]
    fn carveout_fits_the_blocks() {
        // Two blocks of 40 KB plus 1 KB reserved on a 100 KB SM need 82%.
        assert_eq!(carveout_for(2, 40 * 1024, 1024, 100 * 1024), 82);
        assert_eq!(carveout_for(8, 40 * 1024, 0, 100 * 1024), 100);
        assert_eq!(carveout_for(4, 0, 1024, 0), -1);
    }

    /// `CUlaunchAttribute` is 72 bytes, 8-byte aligned, with its value union at byte 8, so an
    /// event pointer lands on an 8-byte boundary, followed by the flags.
    #[test]
    fn event_attribute_layout() {
        assert_eq!(std::mem::size_of::<super::LaunchAttribute>(), 72);
        assert_eq!(std::mem::align_of::<super::LaunchAttribute>(), 8);
        let a = event_attribute(7, 0x1234_5678_9abc_def0usize as *mut std::ffi::c_void, &[3, 1]);
        assert_eq!((a.id, a.value[0], a.value[1], a.value[2], a.value[3]), (7, 0x9abc_def0, 0x1234_5678, 3, 1));
    }
}
