//! cute_ext's fragment-free TMA operations (roadmap part m11-cute-ext-tma):
//! [`tma_load`], [`tma_store`] and [`tma_reduce_store`] on whole tensors,
//! with their descriptors made automatically.
//!
//! The global operand is a view of a global tensor (a slice, `local_tile`,
//! ... of a kernel argument), whose predication origin says which tensor
//! (the *root*) it views and where its elements lie in it; the shared
//! operand is one buffer, or one stage of one. From the root, the shared
//! buffer's layout and the view's map from the buffer's elements to the
//! root's coordinates (the CTA-V map, which `cta_v_map` overrides), each
//! operation makes a TMA atom ([`make_tiled_tma_atom`](crate::cpasync::make_tiled_tma_atom)'s
//! construction) and so a hidden grid-constant tensor-map parameter, one
//! per distinct root, box layout, operation and internal type in the
//! kernel: two loads of different tiles of the same tensor into buffers of
//! the same layout share one. The launcher encodes it from the root's
//! buffer and the launch's extents and strides, as for any TMA atom. The
//! view's coordinates (which may be runtime values) give each
//! instruction's coordinates; TMA zero-fills the elements of a load outside
//! the root and drops those of a store.
//!
//! As in CuTe DSL's lowering, one elected lane of each warp that calls an
//! operation issues it: call it from one warp (the TMA warp of a
//! warp-specialised kernel). A load with `update_expect_tx` (the default)
//! also adds the buffer's bytes to the mbarrier's transaction count from
//! that lane (`mbarrier.expect_tx`, without an arrival: the arrivals stay
//! the program's, a pipeline's commit or an explicit `mbarrier_arrive`).
//! Under a CuTe DSL-style pipeline, which arms the count itself, pass
//! `update_expect_tx(false)`: counting twice leaves the phase waiting for
//! bytes that never come, which the simulator reports. Stores do not commit
//! a bulk async-group: [`TMAStorePipeline`](super::pipeline::TMAStorePipeline)
//! or `cp_async_bulk_commit_group` does.
//!
//! A root whose base pointer is a value computed in the kernel (read from
//! device memory, as a pointer-array GEMM's are) has no host-encoded
//! descriptor: the kernel builds one in its device workspace
//! ([`host_runtime`](super::host_runtime), 128 bytes per CTA and
//! descriptor) at each call, from a template whose box, swizzle and type
//! are the atom's, rewriting its address, extents and strides
//! ([`update_tma_descriptor`]) and
//! releasing it to the tensormap proxy before the copy.

use std::panic::Location;
use std::rc::Rc;

use super::core::OperationType;
use super::host_runtime::{WorkspaceNeed, linear_cta, need_ptr, push_need, workspace_base};
use crate::arch::{elect_one, mbarrier_expect_tx};
use crate::atom::copy::{CopyArgs, CopyAtom, CopyOp};
use crate::builder::{self, user_panic};
use crate::control::if_;
use crate::cpasync::{
    CtaGroup, ReductionKind, TmaAtomOptions, copy_tensormap, fence_tma_desc_acquire, fence_tma_desc_release,
    tma_partition, update_tma_descriptor,
};
use crate::ir::{DType, MemSpace, Param};
use crate::layout::{IndexExpr, Layout, StrideLeaf, Tree, make_composed_layout, make_layout_image_mask};
use crate::nvgpu::tma_atom::{TmaTile, make_tma_atom_core};
use crate::tensor::{AddressSpace, Engine, Ident, Ptr, Tensor, TensorInner, make_coord_tensor, make_tensor};
use crate::value::{Int16, Numeric};

/// The keyword arguments of [`tma_load_with`] (CuTe DSL's `tma_load(...,
/// *, cta_v_map, tma_operation_type, vmnk_layout, multicast_mode,
/// internal_type, update_expect_tx)`).
#[derive(Clone, Debug)]
pub struct TmaLoadOptions {
    pub cta_v_map: Option<Layout>,
    pub tma_operation_type: Option<OperationType>,
    pub vmnk_layout: Option<Layout>,
    pub multicast_mode: Option<usize>,
    pub internal_type: Option<DType>,
    pub update_expect_tx: bool,
}

impl Default for TmaLoadOptions {
    fn default() -> Self {
        TmaLoadOptions {
            cta_v_map: None,
            tma_operation_type: None,
            vmnk_layout: None,
            multicast_mode: None,
            internal_type: None,
            update_expect_tx: true,
        }
    }
}

impl TmaLoadOptions {
    pub fn new() -> TmaLoadOptions {
        TmaLoadOptions::default()
    }

    /// The map from the shared buffer's elements to the root tensor's
    /// coordinates (a layout with basis strides), instead of the view's.
    pub fn cta_v_map(mut self, map: &Layout) -> Self {
        self.cta_v_map = Some(map.clone());
        self
    }

    /// The TMA operation: [`OperationType::Sm90TmaLoad`] (the default),
    /// `Sm90TmaLoadMulticast`, `Sm100TmaLoad2Sm` or
    /// `Sm100TmaLoad2SmMulticast`.
    pub fn tma_operation_type(mut self, op: OperationType) -> Self {
        self.tma_operation_type = Some(op);
        self
    }

    /// The cluster's CTA layout (a rank in the cluster for each coordinate),
    /// for a multicast.
    pub fn vmnk_layout(mut self, layout: &Layout) -> Self {
        self.vmnk_layout = Some(layout.clone());
        self
    }

    /// The mode of [`vmnk_layout`](Self::vmnk_layout) along which the CTAs
    /// share the tile, for a multicast.
    pub fn multicast_mode(mut self, mode: usize) -> Self {
        self.multicast_mode = Some(mode);
        self
    }

    /// The element type TMA moves the elements as (the tensors keep theirs).
    pub fn internal_type(mut self, t: DType) -> Self {
        self.internal_type = Some(t);
        self
    }

    /// Whether the load adds its bytes to the mbarrier's transaction count
    /// (the default); false under a pipeline that arms it.
    pub fn update_expect_tx(mut self, yes: bool) -> Self {
        self.update_expect_tx = yes;
        self
    }
}

/// The keyword arguments of [`tma_store_with`] and
/// [`tma_reduce_store_with`] (`cta_v_map`, `internal_type`).
#[derive(Clone, Debug, Default)]
pub struct TmaStoreOptions {
    pub cta_v_map: Option<Layout>,
    pub internal_type: Option<DType>,
}

impl TmaStoreOptions {
    pub fn new() -> TmaStoreOptions {
        TmaStoreOptions::default()
    }

    pub fn cta_v_map(mut self, map: &Layout) -> Self {
        self.cta_v_map = Some(map.clone());
        self
    }

    pub fn internal_type(mut self, t: DType) -> Self {
        self.internal_type = Some(t);
        self
    }
}

/// Load the global view `src` into the shared buffer `dst` with TMA,
/// reporting to the mbarrier `mbar` (cute_ext's `tma_load(src, dst, mbar)`).
/// See [`tma_load_with`].
#[track_caller]
pub fn tma_load<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, mbar: &Ptr<u64>) {
    tma_load_with(src, dst, mbar, &TmaLoadOptions::new());
}

/// [`tma_load`] with cute_ext's keyword arguments ([`TmaLoadOptions`]).
///
/// `src` is a view of a global tensor (see the [module](self)) with as many
/// elements as `dst`, a shared-memory buffer (plain or swizzled) laid out
/// as TMA can fill it. Without an operation type the load is
/// `SM90_TMA_LOAD`; the multicast types need `vmnk_layout` and
/// `multicast_mode`, and those keywords need a multicast type. In a
/// multicast the CTAs along `multicast_mode` of the cluster each load
/// their share of the tile and deliver it to all of them, at the same
/// shared-memory offset, counting down each one's mbarrier at the same
/// offset; each CTA's `expect_tx` takes the whole tile. A 2SM load
/// (`SM100_TMA_LOAD_2SM`, `SM100_TMA_LOAD_2SM_MULTICAST`) reports to the
/// mbarrier of the pair's leader (the even cluster rank), so only the
/// leader adds to it, the bytes of both CTAs' tiles.
#[track_caller]
pub fn tma_load_with<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, mbar: &Ptr<u64>, opts: &TmaLoadOptions) {
    let loc = Location::caller();
    let _at = builder::LocationScope::new();
    let f = Fail { loc, what: "experimental::tma_load" };
    let fail = |m: String| -> ! { f.fail(m) };
    let op_type = match opts.tma_operation_type {
        Some(t) => t,
        None => {
            if opts.vmnk_layout.is_some() || opts.multicast_mode.is_some() {
                fail(
                    "vmnk_layout and multicast_mode describe a multicast, so they need a multicast \
                     tma_operation_type (SM90_TMA_LOAD_MULTICAST or SM100_TMA_LOAD_2SM_MULTICAST)"
                        .to_string(),
                );
            }
            OperationType::Sm90TmaLoad
        }
    };
    let (multicast, cta_group) = match op_type {
        OperationType::Sm90TmaLoad => (false, CtaGroup::ONE),
        OperationType::Sm90TmaLoadMulticast => (true, CtaGroup::ONE),
        OperationType::Sm100TmaLoad2Sm => (false, CtaGroup::TWO),
        OperationType::Sm100TmaLoad2SmMulticast => (true, CtaGroup::TWO),
        other => fail(format!(
            "{other} is not a tile load; tma_operation_type is SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST, \
             SM100_TMA_LOAD_2SM or SM100_TMA_LOAD_2SM_MULTICAST"
        )),
    };
    let mcast = if multicast {
        let (Some(vmnk), Some(mode)) = (&opts.vmnk_layout, opts.multicast_mode) else {
            fail(format!(
                "{op_type} shares the tile among the CTAs of a cluster: give vmnk_layout (the cluster's CTA layout) \
                 and multicast_mode (the mode the CTAs that share it run along)"
            ))
        };
        Some(multicast_share(vmnk, mode, f))
    } else {
        if opts.vmnk_layout.is_some() || opts.multicast_mode.is_some() {
            fail(format!("{op_type} does not multicast, so vmnk_layout and multicast_mode do not apply"));
        }
        None
    };
    let op = CopyOp::TmaLoad { multicast, cta_group };
    let num_multicast = mcast.as_ref().map_or(1, |m| m.count);
    let plan = plan_copy(src, dst, op, opts.cta_v_map.as_ref(), opts.internal_type, num_multicast, f);
    let (ts, tg) = match &mcast {
        Some(m) => tma_partition(
            &plan.atom,
            Tree::Leaf(m.coord.clone()),
            &Layout::new(m.count, 1),
            &dst.group_modes(0, dst.rank()),
            &plan.gview.group_modes(0, plan.gview.rank()),
        ),
        None => tma_partition(
            &plan.atom,
            0,
            &Layout::new(1, 0),
            &dst.group_modes(0, dst.rank()),
            &plan.gview.group_modes(0, plan.gview.rank()),
        ),
    };
    let bytes = (dst.size() * T::DTYPE.bits()).div_ceil(8);
    let mut args = CopyArgs::new().tma_bar_ptr(mbar);
    if let Some(m) = &mcast {
        args = args.mcast_mask(m.mask);
    }
    issue(&plan, args, |args| {
        if opts.update_expect_tx {
            if cta_group == CtaGroup::TWO {
                // m11-cute-ext-gemm: a 2SM load completes on the pair's leader
                // (even cluster rank), which receives both CTAs' tiles: the leader
                // alone arms its barrier, for the pair's bytes, so that its phase
                // cannot complete before the peer's tile has landed.
                let leader = (crate::arch::block_idx_in_cluster() % 2).eq(0);
                if_(leader, || if_(elect_one(), || mbarrier_expect_tx(mbar, 2 * bytes as i32)));
            } else {
                if_(elect_one(), || mbarrier_expect_tx(mbar, bytes as i32));
            }
        }
        crate::copy_with(&plan.atom, &tg, &ts, args);
    });
}

/// The deprecated multicast form of [`tma_load_with`] (cute_ext's
/// `tma_load_multicast(src, dst, mbar, *, vmnk_layout, cta_v_map,
/// tma_operation_type, multicast_mode, update_expect_tx)`): `op` must be a
/// multicast operation type.
#[deprecated(note = "use tma_load_with with a multicast tma_operation_type, vmnk_layout and multicast_mode")]
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn tma_load_multicast<T: Numeric>(
    src: &Tensor<T>,
    dst: &Tensor<T>,
    mbar: &Ptr<u64>,
    vmnk_layout: &Layout,
    cta_v_map: Option<&Layout>,
    tma_operation_type: OperationType,
    multicast_mode: usize,
    update_expect_tx: bool,
) {
    if !matches!(tma_operation_type, OperationType::Sm90TmaLoadMulticast | OperationType::Sm100TmaLoad2SmMulticast) {
        user_panic(
            Location::caller(),
            format!(
                "experimental::tma_load_multicast: {tma_operation_type} is not a multicast load \
                 (SM90_TMA_LOAD_MULTICAST, SM100_TMA_LOAD_2SM_MULTICAST)"
            ),
        );
    }
    let mut opts = TmaLoadOptions::new()
        .tma_operation_type(tma_operation_type)
        .vmnk_layout(vmnk_layout)
        .multicast_mode(multicast_mode)
        .update_expect_tx(update_expect_tx);
    opts.cta_v_map = cta_v_map.cloned();
    tma_load_with(src, dst, mbar, &opts);
}

/// Store the shared buffer `src` to the global view `dst` with TMA
/// (cute_ext's `tma_store(src, dst)`). See [`tma_store_with`].
#[track_caller]
pub fn tma_store<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>) {
    tma_store_with(src, dst, &TmaStoreOptions::new());
}

/// [`tma_store`] with `cta_v_map` and `internal_type`: `dst` is a view of
/// a global tensor, `src` a shared buffer with as many elements. Elements
/// outside the root tensor are not written. The store joins the issuing
/// thread's open bulk async-group; commit and wait for it as for any TMA
/// store (before the buffer is rewritten, and before the CTA exits).
#[track_caller]
pub fn tma_store_with<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, opts: &TmaStoreOptions) {
    let loc = Location::caller();
    let _at = builder::LocationScope::new();
    let f = Fail { loc, what: "experimental::tma_store" };
    store_core(src, dst, CopyOp::TmaStore, opts, f);
}

/// Reduce the shared buffer `src` into the global view `dst` with TMA
/// (cute_ext's `tma_reduce_store(src, dst, kind=...)`,
/// `cp.reduce.async.bulk.tensor.<rank>d.global.shared::cta.<kind>.tile`):
/// each element of `dst` inside its root becomes `kind` of itself and the
/// buffer's, atomically. See [`tma_reduce_store_with`].
#[track_caller]
pub fn tma_reduce_store<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, kind: ReductionKind) {
    tma_reduce_store_with(src, dst, kind, &TmaStoreOptions::new());
}

/// [`tma_reduce_store`] with `cta_v_map` and `internal_type`, as
/// [`tma_store_with`] takes them.
#[track_caller]
pub fn tma_reduce_store_with<T: Numeric>(
    src: &Tensor<T>,
    dst: &Tensor<T>,
    kind: ReductionKind,
    opts: &TmaStoreOptions,
) {
    let loc = Location::caller();
    let _at = builder::LocationScope::new();
    let f = Fail { loc, what: "experimental::tma_reduce_store" };
    store_core(src, dst, CopyOp::TmaReduce(kind), opts, f);
}

#[track_caller]
fn store_core<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, op: CopyOp, opts: &TmaStoreOptions, f: Fail) {
    let plan = plan_copy(dst, src, op, opts.cta_v_map.as_ref(), opts.internal_type, 1, f);
    let (ts, tg) = tma_partition(
        &plan.atom,
        0,
        &Layout::new(1, 0),
        &src.group_modes(0, src.rank()),
        &plan.gview.group_modes(0, plan.gview.rank()),
    );
    issue(&plan, CopyArgs::new(), |args| crate::copy_with(&plan.atom, &ts, &tg, args));
}

/// Where a TMA operation's build-time errors are reported: the caller's
/// location and the function's name.
#[derive(Clone, Copy)]
struct Fail {
    loc: &'static Location<'static>,
    what: &'static str,
}

impl Fail {
    fn fail(&self, m: String) -> ! {
        user_panic(self.loc, format!("{}: {m}", self.what))
    }
}

/// A multicast's share: the CTAs along the mode, this CTA's coordinate
/// among them, and the mask of their ranks.
struct Multicast {
    count: usize,
    coord: IndexExpr,
    mask: Int16,
}

#[track_caller]
fn multicast_share(vmnk: &Layout, mode: usize, f: Fail) -> Multicast {
    let fail = |m: String| -> ! { f.fail(m) };
    if mode >= vmnk.rank() {
        fail(format!("multicast_mode {mode} is not a mode of the rank-{} vmnk_layout {vmnk}", vmnk.rank()));
    }
    if !vmnk.is_static() {
        fail(format!("the vmnk_layout {vmnk} is not static: a cluster's shape is known when the kernel is built"));
    }
    let count = vmnk.mode(mode).size();
    let rank = crate::arch::block_idx_in_cluster();
    let coord = vmnk.get_flat_coord(IndexExpr::from(rank));
    let mine = match &coord {
        Tree::Branch(m) => match &m[mode] {
            Tree::Leaf(c) => c.clone(),
            other => fail(format!("the CTA's coordinate {other} in mode {mode} is not flat")),
        },
        Tree::Leaf(c) => c.clone(),
    };
    let mask = make_layout_image_mask(vmnk, coord, mode);
    Multicast { count, coord: mine, mask }
}

/// What a TMA operation issues with: the atom, the view of its TMA tensor
/// matching the global operand, and, for a root on a device-side pointer,
/// how to build its descriptor.
struct CopyPlan<T> {
    atom: CopyAtom<T>,
    gview: Tensor<Ident>,
    device: Option<DeviceDescriptor<T>>,
}

/// A descriptor built in the device workspace: the workspace need of its
/// per-CTA slots and the root tensor it describes.
struct DeviceDescriptor<T> {
    need: usize,
    root: Tensor<T>,
}

/// Issue `body` (given the copy's arguments) on one elected lane of the
/// calling warp; with a device-built descriptor, build it first.
#[track_caller]
fn issue<T: Numeric>(plan: &CopyPlan<T>, args: CopyArgs<'static>, body: impl FnOnce(CopyArgs<'static>)) {
    match &plan.device {
        None => body(args),
        Some(d) => {
            let slot = need_ptr(d.need);
            let slot = crate::raw_ptr::make_ptr::<u64>(
                slot.toint_i64() + linear_cta().to::<i64>() * 128,
                AddressSpace::Gmem,
                Some(128),
            );
            if_(elect_one(), || {
                copy_tensormap(&plan.atom, &slot);
                update_tma_descriptor(&plan.atom, &d.root, &slot);
                fence_tma_desc_release();
                fence_tma_desc_acquire(&slot);
            });
            body(args.tma_desc_ptr(slot.toint_i64()));
        }
    }
}

/// The atom (shared by every operation with the same descriptor) and the
/// TMA-tensor view for a copy between the global view `g` and the shared
/// buffer `s`.
#[track_caller]
fn plan_copy<T: Numeric>(
    g: &Tensor<T>,
    s: &Tensor<T>,
    op: CopyOp,
    cta_v_map: Option<&Layout>,
    internal_type: Option<DType>,
    num_multicast: usize,
    f: Fail,
) -> CopyPlan<T> {
    let fail = |m: String| -> ! { f.fail(m) };
    let buffer = match g.engine {
        Engine::Memory { buffer, space: MemSpace::Global | MemSpace::Raw(AddressSpace::Gmem), .. } => buffer,
        _ => fail(format!("the global operand {g:?} is not in global memory")),
    };
    if !matches!(s.engine, Engine::Memory { space: MemSpace::Shared, .. }) {
        fail(format!("the buffer {s:?} is not in shared memory: TMA copies between global and shared memory"));
    }
    if matches!(s.inner, TensorInner::Layout { .. }) {
        fail("the shared buffer's layout has a layout inner; TMA writes plain or swizzled shared memory".to_string());
    }
    if g.size() != s.size() {
        fail(format!("the global view has {} elements and the shared buffer {}: they must match", g.size(), s.size()));
    }
    let Some(origin) = g.origin.as_ref() else {
        fail("the global operand has no predication origin (a view of a kernel argument)".to_string())
    };
    let (Some(root), Some((coords, _))) = (g.origin_root(), origin.replay()) else {
        fail(format!(
            "the global operand {g:?} was viewed through an operation that loses track of where its elements lie in \
             the tensor it views (a coalesce, a product, a layout set whole): mark the tensor it should describe \
             with experimental::predicated_tensor_origin"
        ))
    };
    if !matches!(root.inner, TensorInner::Identity) {
        fail("the global tensor is swizzled or composed; TMA describes a plain strided tensor".to_string());
    }
    let map = match cta_v_map {
        Some(m) => m.clone(),
        None => coords.layout().clone(),
    };
    let smem =
        make_composed_layout(s.swizzle().unwrap_or_else(crate::layout::Swizzle::identity), 0, s.layout().clone());
    let is_param = builder::with(|b| b.params().iter().any(|p| matches!(p, Param::Buffer(x) if *x == buffer)));
    let key = format!(
        "{buffer:?}|{}|{:?}|{smem}|{map}|{op:?}|{internal_type:?}|{num_multicast}|{is_param}",
        root.layout(),
        root.base_offset()
    );
    let cached = builder::with(|b| b.cute_ext_tma.get(&key).cloned());
    let entry: Rc<TmaEntry<T>> = match cached.and_then(|c| c.downcast::<TmaEntry<T>>().ok()) {
        Some(e) => e,
        None => {
            let loc = Location::caller();
            let rank = s.rank();
            let atom_options = TmaAtomOptions { gmem_coord_tensor: None, internal_type };
            let (gmem, need) = if is_param {
                (root.clone(), None)
            } else {
                // A root on a device-side pointer: a template descriptor on the workspace.
                let need = push_need(WorkspaceNeed { size: super::host_runtime::NeedSize::PerCta(128), align: 128 });
                let base = workspace_base().expect("the workspace");
                let template = template_layout(root.layout(), f);
                (make_tensor(crate::tensor::recast_ptr::<T, u8>(&base, None), template), Some(need))
            };
            let info = make_tma_atom_core(
                loc,
                "experimental TMA",
                op,
                &gmem,
                smem.clone(),
                TmaTile::VMap { map: map.clone(), rank },
                num_multicast,
                &atom_options,
            );
            let stride = info.tma_tensor.layout().stride().clone();
            let e = Rc::new(TmaEntry { atom: info.atom, tma_stride: stride, need });
            builder::with(|b| b.cute_ext_tma.insert(key, e.clone()));
            e
        }
    };
    let tma_tensor = make_coord_tensor(0, Layout::new(root.layout().shape().clone(), entry.tma_stride.clone()));
    let Some(gview) = origin.replay_on(tma_tensor) else { unreachable!("a tracked origin") };
    let device = entry.need.map(|need| DeviceDescriptor { need, root });
    CopyPlan { atom: entry.atom.clone(), gview, device }
}

/// A cached atom: the atom, its TMA tensor's strides, and the workspace need
/// of a device-built descriptor.
struct TmaEntry<T> {
    atom: CopyAtom<T>,
    tma_stride: crate::layout::Stride,
    need: Option<usize>,
}

/// The layout of a template descriptor for a root whose extents and strides
/// may be device values: static ones kept, runtime extents 256 and runtime
/// strides distinct large multiples of 16 bytes (in mode order), so that the
/// TMA dimensions come out in the order the root's would and the device
/// rewrite fills in the real ones.
fn template_layout(root: &Layout, f: Fail) -> Layout {
    let fail = |m: String| -> ! { f.fail(m) };
    let shape = root.shape().map(&mut |e: &IndexExpr| match e.as_static() {
        Some(v) => IndexExpr::Static(v),
        None => IndexExpr::Static(256),
    });
    let mut k = 0usize;
    let stride = root.stride().map(&mut |s: &StrideLeaf| match s {
        StrideLeaf::Int(e) if e.as_static().is_some() => s.clone(),
        StrideLeaf::Int(_) => {
            k += 1;
            StrideLeaf::Int(IndexExpr::Static((1usize << 24) * k))
        }
        StrideLeaf::Basis(_) => fail(format!("the global tensor's layout {root} has a basis stride")),
    });
    Layout::new(shape, stride)
}
