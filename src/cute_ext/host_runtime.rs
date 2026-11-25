//! cute_ext's host runtime: the device workspace query (CuTe DSL's
//! `AllocationRequirement` and `QueryDeviceWorkspaceFunc`, roadmap part
//! m11-cute-ext-tma). Kernels here are ordinary
//! [`Kernel::build`](crate::Kernel::build) closures, launched as any other.
//!
//! A kernel that needs scratch global memory of a size known only at launch
//! (TMA descriptors it builds on the device, one per CTA and operand;
//! per-CTA partial results) declares each need while it is built with
//! [`request_device_workspace`] (or [`device_workspace_cta_slot`]). The
//! kernel then takes one more buffer argument, after all of its own: the
//! *device workspace*, a `u8` buffer. Before launching, the host asks how
//! large and how aligned it must be with
//! [`Kernel::device_workspace_requirement`] (or [`QueryDeviceWorkspaceFunc`],
//! CuTe DSL's spelling), from the launch configuration and the other
//! arguments, allocates it and passes it last. The simulator and the CUDA
//! launcher refuse a launch without it, or with one smaller than the
//! requirement; the CUDA launcher checks its alignment as it does every
//! pointer's. The workspace's contents are not initialised: what the kernel
//! reads from it, it writes first.
//!
//! The needs are laid out in the order they were requested, each at the
//! next multiple of its alignment; the requirement's alignment is the
//! largest of them (at least 16 bytes).

use crate::Kernel;
use crate::arch::{block_idx, grid_dim};
use crate::builder::{self, user_panic};
use crate::ir::{BufferId, Param, ValueId};
use crate::sim::{KernelArg, LaunchConfig};
use crate::tensor::{AddressSpace, Ptr, param_ptr_aligned};
use crate::value::{Int32, Int64, IntoValue};

/// The name of the workspace parameter.
pub const DEVICE_WORKSPACE_PARAM: &str = "device_workspace";

/// How much device workspace a kernel needs (CuTe DSL's
/// `AllocationRequirement`): bytes, and the alignment of the buffer's start
/// in bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AllocationRequirement {
    pub size_in_bytes: i64,
    pub alignment: i64,
}

/// The size of one workspace need.
#[derive(Clone, Copy, Debug)]
pub enum WorkspaceSize {
    /// A fixed number of bytes.
    Bytes(usize),
    /// Bytes per CTA of the launch's grid.
    PerCta(usize),
    /// Bytes times a scalar argument of the kernel (an
    /// [`Int32`] parameter from [`param_scalar`](crate::param_scalar) or
    /// the like): one descriptor per group of a grouped GEMM, say.
    PerScalar { count: Int32, bytes: usize },
}

/// A need as recorded: its size and alignment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NeedSize {
    Bytes(u64),
    PerCta(u64),
    PerScalar { value: ValueId, bytes: u64 },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct WorkspaceNeed {
    pub(crate) size: NeedSize,
    pub(crate) align: u64,
}

/// The workspace of a kernel being built.
pub(crate) struct WorkspaceBuild {
    pub(crate) buffer: BufferId,
    base: Ptr<u8>,
    needs: Vec<WorkspaceNeed>,
}

impl WorkspaceBuild {
    /// The declaration, with parameter positions from the finished `args`.
    pub(crate) fn finish(self, args: &[Param]) -> WorkspaceDecl {
        let param = args
            .iter()
            .position(|p| matches!(p, Param::Buffer(b) if *b == self.buffer))
            .expect("the workspace is a parameter");
        let scalars = self
            .needs
            .iter()
            .filter_map(|n| match n.size {
                NeedSize::PerScalar { value, .. } => Some(value),
                _ => None,
            })
            .map(|v| {
                let i = args
                    .iter()
                    .position(|p| matches!(p, Param::Scalar { value, .. } if *value == v))
                    .expect("a workspace size's scalar is a parameter");
                (v, i)
            })
            .collect();
        WorkspaceDecl { param, needs: self.needs, scalars }
    }
}

/// A finished kernel's device workspace ([`KernelIr::workspace`](crate::ir::KernelIr::workspace)):
/// its parameter (the last of the kernel's own) and its needs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WorkspaceDecl {
    /// The workspace's parameter index.
    pub param: usize,
    pub(crate) needs: Vec<WorkspaceNeed>,
    /// The parameter index of each scalar a need depends on.
    pub(crate) scalars: Vec<(ValueId, usize)>,
}

impl WorkspaceDecl {
    /// The requirement of a launch of `ctas` CTAs whose scalar argument `i`
    /// is `scalar(i)`.
    pub fn requirement(
        &self,
        ctas: u64,
        scalar: &dyn Fn(usize) -> Option<i64>,
    ) -> Result<AllocationRequirement, String> {
        let mut end = 0u64;
        let mut align = 16u64;
        for n in &self.needs {
            let bytes = match n.size {
                NeedSize::Bytes(b) => b,
                NeedSize::PerCta(b) => b * ctas,
                NeedSize::PerScalar { value, bytes } => {
                    let i = self.scalars.iter().find(|(v, _)| *v == value).map(|&(_, i)| i).expect("resolved");
                    let count = scalar(i).ok_or_else(|| format!("argument {i} (a workspace size) is not a scalar"))?;
                    if count < 0 {
                        return Err(format!("argument {i} sizes the device workspace, but is negative ({count})"));
                    }
                    bytes * count as u64
                }
            };
            end = end.next_multiple_of(n.align) + bytes;
            align = align.max(n.align);
        }
        Ok(AllocationRequirement { size_in_bytes: end as i64, alignment: align as i64 })
    }

    /// Check a launch's workspace argument: `nargs` arguments were given,
    /// the workspace (when given) has `bytes` bytes, if known.
    pub fn check_launch(
        &self,
        kernel: &str,
        cfg: &LaunchConfig,
        nargs: usize,
        scalar: &dyn Fn(usize) -> Option<i64>,
        bytes: Option<u64>,
    ) -> Result<(), String> {
        let req = self.requirement(ctas_of(cfg), scalar)?;
        if nargs == self.param {
            return Err(format!(
                "kernel `{kernel}` takes a device workspace of {} bytes ({}-byte aligned) as its last argument \
                 (argument {}), which is missing: size it with Kernel::device_workspace_requirement",
                req.size_in_bytes, req.alignment, self.param
            ));
        }
        if let Some(b) = bytes
            && b < req.size_in_bytes as u64
        {
            return Err(format!(
                "kernel `{kernel}`'s device workspace (argument {}) has {b} bytes, but this launch needs {}: size it \
                 with Kernel::device_workspace_requirement",
                self.param, req.size_in_bytes
            ));
        }
        Ok(())
    }
}

/// The CTAs of a launch.
fn ctas_of(cfg: &LaunchConfig) -> u64 {
    cfg.grid.iter().map(|&g| g as u64).product()
}

/// The signed value of a scalar argument.
pub(crate) fn scalar_arg(a: &KernelArg) -> Option<i64> {
    a.scalar_bits().map(|(ty, bits)| {
        let w = ty.bits() as u32;
        if ty.is_signed() && w < 128 { ((bits << (128 - w)) as i128 >> (128 - w)) as i64 } else { bits as i64 }
    })
}

/// Check the workspace argument of a simulated or CUDA launch of `kernel`
/// with host arguments `args`.
pub(crate) fn check_launch_args(kernel: &Kernel, cfg: &LaunchConfig, args: &[KernelArg]) -> Result<(), String> {
    let Some(ws) = &kernel.ir().workspace else { return Ok(()) };
    let bytes = args.get(ws.param).and_then(|a| a.buffer_type()).map(|(t, n)| t.storage_bytes(n) as u64);
    ws.check_launch(kernel.name(), cfg, args.len(), &|i| args.get(i).and_then(scalar_arg), bytes)
}

impl Kernel {
    /// The device workspace a launch with `cfg` and `args` needs (CuTe DSL's
    /// `get_aux_func(QueryDeviceWorkspaceFunc, kernel=...)(*args)`), computed
    /// from the grid and the scalar arguments before anything is uploaded.
    /// `args` are the kernel's arguments, with or without the workspace
    /// itself. A kernel without workspace needs 0 bytes.
    pub fn device_workspace_requirement(
        &self,
        cfg: LaunchConfig,
        args: &[KernelArg],
    ) -> Result<AllocationRequirement, String> {
        match &self.ir().workspace {
            None => Ok(AllocationRequirement { size_in_bytes: 0, alignment: 16 }),
            Some(ws) => ws.requirement(ctas_of(&cfg), &|i| args.get(i).and_then(scalar_arg)),
        }
    }
}

/// CuTe DSL's `QueryDeviceWorkspaceFunc`, the auxiliary function that sizes a
/// kernel's device workspace: [`QueryDeviceWorkspaceFunc::call`] is
/// [`Kernel::device_workspace_requirement`].
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryDeviceWorkspaceFunc;

impl QueryDeviceWorkspaceFunc {
    pub fn call(
        &self,
        kernel: &Kernel,
        cfg: LaunchConfig,
        args: &[KernelArg],
    ) -> Result<AllocationRequirement, String> {
        kernel.device_workspace_requirement(cfg, args)
    }
}

/// Declare a need for `size` bytes of the kernel's device workspace, aligned
/// to `alignment` bytes (a power of two), and return a pointer to its start
/// (a global pointer, `alignment`-aligned). The first need adds the
/// workspace parameter, which the kernel takes after its own parameters.
/// A [`WorkspaceSize::PerScalar`] count must be a scalar parameter of the
/// kernel.
#[track_caller]
pub fn request_device_workspace(size: WorkspaceSize, alignment: usize) -> Ptr<u8> {
    let loc = std::panic::Location::caller();
    if !alignment.is_power_of_two() {
        user_panic(
            loc,
            format!("experimental::request_device_workspace: the alignment {alignment} is not a power of two"),
        );
    }
    let size = match size {
        WorkspaceSize::Bytes(b) => NeedSize::Bytes(b as u64),
        WorkspaceSize::PerCta(b) => NeedSize::PerCta(b as u64),
        WorkspaceSize::PerScalar { count, bytes } => {
            let value = count.checked();
            let is_param = builder::with(|b| {
                b.params().iter().any(|p| matches!(p, Param::Scalar { value: v, .. } if *v == value))
            });
            if !is_param {
                user_panic(
                    loc,
                    "experimental::request_device_workspace: a PerScalar count is a scalar parameter of the kernel, \
                     which the host knows before the launch, not a value computed in the kernel"
                        .to_string(),
                );
            }
            NeedSize::PerScalar { value, bytes: bytes as u64 }
        }
    };
    let index = push_need(WorkspaceNeed { size, align: alignment as u64 });
    need_ptr(index)
}

/// Record a workspace need (declaring the workspace at the first) and
/// return its index.
pub(crate) fn push_need(need: WorkspaceNeed) -> usize {
    if builder::with(|b| b.workspace.is_none()) {
        let base = param_ptr_aligned::<u8>(DEVICE_WORKSPACE_PARAM, 16);
        builder::with(|b| b.workspace = Some(WorkspaceBuild { buffer: base.buffer, base, needs: Vec::new() }));
    }
    builder::with(|b| {
        let w = b.workspace.as_mut().expect("the workspace");
        w.needs.push(need);
        let (index, buffer) = (w.needs.len() - 1, w.buffer);
        let decl = b.buffer_mut(buffer);
        decl.align = decl.align.max(need.align as usize);
        index
    })
}

/// The workspace's base pointer (a pointer to the parameter), if declared.
pub(crate) fn workspace_base() -> Option<Ptr<u8>> {
    builder::with(|b| b.workspace.as_ref().map(|w| w.base.clone()))
}

/// The start of workspace need `index`, computed here: the earlier needs'
/// bytes, each aligned.
#[track_caller]
pub(crate) fn need_ptr(index: usize) -> Ptr<u8> {
    let (base, needs) = builder::with(|b| {
        let w = b.workspace.as_ref().expect("the workspace");
        (w.base.clone(), w.needs[..=index].to_vec())
    });
    let mut end: Int64 = 0i64.into_value();
    for n in &needs[..index] {
        end = align_up(end, n.align) + need_bytes(n);
    }
    let need = needs[index];
    let offset = align_up(end, need.align);
    let address = base.toint_i64() + offset;
    crate::raw_ptr::make_ptr::<u8>(address, AddressSpace::Gmem, Some(need.align as usize))
}

/// [`request_device_workspace`] of `bytes_per_cta` bytes per CTA, returning
/// this CTA's slot: the region's start plus `bytes_per_cta` times the CTA's
/// linear index in the grid (`x` fastest). `bytes_per_cta` must be a
/// multiple of `alignment`.
#[track_caller]
pub fn device_workspace_cta_slot(bytes_per_cta: usize, alignment: usize) -> Ptr<u8> {
    let loc = std::panic::Location::caller();
    if !bytes_per_cta.is_multiple_of(alignment.max(1)) {
        user_panic(
            loc,
            format!(
                "experimental::device_workspace_cta_slot: {bytes_per_cta} bytes per CTA is not a multiple of the \
                 alignment {alignment}, so the slots after the first would not be aligned"
            ),
        );
    }
    let region = request_device_workspace(WorkspaceSize::PerCta(bytes_per_cta), alignment);
    let address = region.toint_i64() + linear_cta().to::<i64>() * bytes_per_cta as i64;
    crate::raw_ptr::make_ptr::<u8>(address, AddressSpace::Gmem, Some(alignment))
}

/// The CTA's linear index in the grid.
#[track_caller]
pub(crate) fn linear_cta() -> Int32 {
    let (b, g) = (block_idx(), grid_dim());
    b.x + b.y * g.x + b.z * g.x * g.y
}

#[track_caller]
fn align_up(v: Int64, align: u64) -> Int64 {
    if align <= 1 { v } else { (v + (align as i64 - 1)) / align as i64 * align as i64 }
}

#[track_caller]
fn need_bytes(n: &WorkspaceNeed) -> Int64 {
    match n.size {
        NeedSize::Bytes(b) => (b as i64).into_value(),
        NeedSize::PerCta(b) => {
            let g = grid_dim();
            (g.x * g.y * g.z).to::<i64>() * b as i64
        }
        NeedSize::PerScalar { value, bytes } => Int32::from_id(value).to::<i64>() * bytes as i64,
    }
}
