//! cute_ext's memory operations: [`allocate`] (shared memory, registers and
//! pooled tensor memory), [`copy`] with automatic predication, and
//! [`predicated_tensor_origin`].

use crate::Arch;
use crate::arch::{
    alloc_tmem, cluster_sync, dealloc_tmem, relinquish_tmem_alloc_permit, retrieve_tmem_ptr, sync_threads,
    tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync, warp_idx,
};
use crate::atom::copy::AsCopyAtom;
use crate::builder;
use crate::control::if_;
use crate::layout::{Inner, Layout};
use crate::memory::compute_tmem_cols_from_layout;
use crate::tensor::{
    AddressSpace, Ptr, SmemLayout, Tensor, alloc_smem, make_fragment, make_fragment_with_layout,
    make_smem_tensor_aligned, make_tensor, recast_ptr,
};
use crate::value::{Boolean, Numeric};

/// The most TMEM columns one CTA's pooled allocation takes.
const MAX_TMEM_COLUMNS: usize = 512;

/// Allocate a buffer of `T` with `layout` in `space` (cute_ext's
/// `allocate(type, address_space, layout, alignment)`), as a tensor:
///
/// - [`AddressSpace::Smem`]: a shared-memory buffer aligned to `alignment`
///   bytes (a power of two). A swizzled composed layout keeps its swizzle,
///   and `alignment` must be at least the swizzle's period in bytes, so
///   that the swizzle acts on aligned addresses.
/// - [`AddressSpace::Rmem`]: a register fragment with `layout` (static);
///   `alignment` is checked to be a power of two and otherwise has no
///   effect, registers having no address.
/// - [`AddressSpace::Tmem`]: a column range of the kernel's pooled TMEM
///   allocation (`alignment` counts columns here: the range starts at a
///   multiple of it). The pool is allocated by warp 0 at the start of the
///   kernel body (the next power of two of at least 32 columns covering
///   every range), its address published through shared memory before a
///   fence and a barrier, and freed at the end of the body. At most 512
///   columns in all; an allocation inside runtime control flow (an `if_`,
///   a runtime loop) is refused, as the prologue and epilogue run once.
///
/// [`allocate_with`] takes cute_ext's `is2cta`.
#[track_caller]
pub fn allocate<T: Numeric>(space: AddressSpace, layout: impl SmemLayout, alignment: usize) -> Tensor<T> {
    allocate_with(space, layout, alignment, false)
}

/// [`allocate`] with `is2cta`: a TMEM range of a CTA pair's allocation (the
/// `cta_group::2` forms of `tcgen05.alloc` and `tcgen05.dealloc`, the free
/// after a cluster barrier). Every TMEM allocation of a kernel uses the same
/// form; `is2cta` must be false in the other spaces.
#[track_caller]
pub fn allocate_with<T: Numeric>(
    space: AddressSpace,
    layout: impl SmemLayout,
    alignment: usize,
    is2cta: bool,
) -> Tensor<T> {
    let _at = builder::LocationScope::new();
    assert!(alignment.is_power_of_two(), "experimental::allocate: the alignment {alignment} must be a power of two");
    assert!(
        !is2cta || space == AddressSpace::Tmem,
        "experimental::allocate: is2cta applies to tensor memory, not {space} memory"
    );
    let c = layout.into_composed();
    match space {
        AddressSpace::Smem => {
            if let Inner::Swizzle(sw) = &c.inner
                && !sw.is_identity()
            {
                let period_bytes = sw.period() * T::DTYPE.storage_bits() / 8;
                assert!(
                    alignment >= period_bytes,
                    "experimental::allocate: shared buffer with the swizzle {sw} needs an alignment of at least \
                     {period_bytes} bytes (one swizzle period), but has {alignment}"
                );
            }
            make_smem_tensor_aligned::<T>(c, alignment)
        }
        AddressSpace::Rmem => make_fragment_with_layout::<T>(plain_layout(c, "register")),
        AddressSpace::Tmem => allocate_tmem::<T>(plain_layout(c, "tensor-memory"), alignment, is2cta),
        other => panic!(
            "experimental::allocate: buffers are allocated in shared memory, registers or tensor memory, not {other} \
             memory"
        ),
    }
}

#[track_caller]
fn plain_layout(c: crate::layout::ComposedLayout, what: &str) -> Layout {
    assert!(c.is_normal(), "experimental::allocate: a {what} buffer takes a plain layout, not the composed layout {c}");
    c.outer
}

/// The pooled TMEM allocation of a kernel being built: the shared-memory
/// word its address goes through, its base (read where the first range was
/// allocated), the columns handed out, and its form.
pub(crate) struct TmemPool {
    slot: Ptr<i32>,
    base: Ptr<f32>,
    used: usize,
    two_cta: bool,
    arch: Arch,
}

#[track_caller]
fn allocate_tmem<T: Numeric>(layout: Layout, alignment: usize, two_cta: bool) -> Tensor<T> {
    assert!(
        !builder::in_runtime_region(),
        "experimental::allocate: tensor memory is allocated once per kernel, so allocate(tmem) must be called at the \
         top level of the kernel body, outside if_ and runtime loops"
    );
    let cols = compute_tmem_cols_from_layout(&layout, T::DTYPE);
    if builder::with(|b| b.cute_ext_tmem.is_none()) {
        start_tmem_pool(two_cta);
    }
    let (base, offset) = builder::with(|b| {
        let pool = b.cute_ext_tmem.as_mut().expect("the TMEM pool");
        assert!(
            pool.two_cta == two_cta,
            "experimental::allocate: every tensor-memory buffer of a kernel takes the same is2cta (this one {two_cta}, \
             the first {})",
            pool.two_cta
        );
        let offset = pool.used.next_multiple_of(alignment);
        let end = offset + cols;
        assert!(
            end <= MAX_TMEM_COLUMNS,
            "experimental::allocate: the tensor-memory buffers of this kernel need {end} columns, more than the \
             {MAX_TMEM_COLUMNS} a CTA has"
        );
        pool.used = end;
        (pool.base.clone(), offset)
    });
    let ptr = if offset == 0 { base } else { base.offset(offset) };
    make_tensor(recast_ptr::<T, f32>(&ptr, None), layout)
}

/// Start the pool at the first TMEM allocation: the address word, the read
/// of the base (here, at the top level, after the prologue's barrier), and
/// the finish hook that emits the prologue and the epilogue.
#[track_caller]
fn start_tmem_pool(two_cta: bool) {
    let arch = crate::target_arch().unwrap_or(Arch::Sm100a);
    let slot = alloc_smem::<i32>(1, 16);
    let base = retrieve_tmem_ptr::<f32>(16, &slot);
    builder::with(|b| {
        b.cute_ext_tmem = Some(TmemPool { slot, base, used: 0, two_cta, arch });
        b.finish_hooks.push(Box::new(finish_tmem_pool));
    });
}

/// Emit the pool's prologue at the start of the body and its epilogue here,
/// at the end.
fn finish_tmem_pool() {
    let Some(pool) = builder::with(|b| b.cute_ext_tmem.take()) else { return };
    let ncols = pool.used.max(32).next_power_of_two();
    let (two_cta, arch) = (pool.two_cta, pool.arch);
    let slot = pool.slot.clone();
    builder::emit_at_body_start(move || {
        if_(warp_idx().eq(0), || alloc_tmem(ncols as i32, &slot, Some(two_cta), arch));
        tcgen05_fence_before_thread_sync();
        sync_threads();
        tcgen05_fence_after_thread_sync();
    });
    tcgen05_fence_before_thread_sync();
    sync_threads();
    if two_cta {
        // Neither CTA of the pair frees the pair's columns while the other
        // may still use them.
        cluster_sync();
    }
    if_(warp_idx().eq(0), || {
        relinquish_tmem_alloc_permit(Some(two_cta));
        dealloc_tmem(&pool.base, ncols as i32, Some(two_cta), arch);
    });
}

/// Copy `src` to `dst` with `copy_atom` (cute_ext's `copy(src, dst, *,
/// copy_atom)`), predicated automatically: an operand viewed from a
/// predication origin (a global-memory tensor, or one given to
/// [`predicated_tensor_origin`]) copies only the elements inside the
/// origin's shape. The predicate guards each application of the atom by its
/// first element, so an atom's elements must be all inside or all outside
/// (as with a 128-bit copy along a mode whose extent is a multiple of the
/// vector). Operands statically in bounds are copied without a predicate.
#[track_caller]
pub fn copy<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, copy_atom: &impl AsCopyAtom<T>) {
    let _at = builder::LocationScope::new();
    match auto_predicate(src, dst, "experimental::copy") {
        Some(pred) => crate::copy_if(copy_atom, &pred, src, dst),
        None => crate::copy(copy_atom, src, dst),
    }
}

/// Make `tensor` a predication origin (cute_ext's
/// `predicated_tensor_origin`): the automatic predicates of the views made
/// from it, and of itself, bound them by its shape, whatever it was viewed
/// from. A tensor with static extents viewed from nowhere further is
/// always in bounds, so its copies are unpredicated.
#[track_caller]
pub fn predicated_tensor_origin<T>(tensor: &Tensor<T>) -> Tensor<T> {
    tensor.with_fresh_origin()
}

/// The automatic predicate of a copy from `src` to `dst`: a Boolean
/// fragment shaped like `src`, the conjunction of both operands' bounds,
/// or `None` when both are statically in bounds (or have no origin).
#[track_caller]
pub(crate) fn auto_predicate<S, D>(src: &Tensor<S>, dst: &Tensor<D>, func: &str) -> Option<Tensor<bool>> {
    let flags = [origin_flags(src, func, "source"), origin_flags(dst, func, "destination")];
    let mut combined: Option<Vec<Boolean>> = None;
    for f in flags.into_iter().flatten() {
        combined = Some(match combined {
            None => f,
            Some(c) => {
                assert_eq!(c.len(), f.len(), "{func}: the source and destination differ in size");
                c.into_iter().zip(f).map(|(a, b)| a & b).collect()
            }
        });
    }
    let flags = combined?;
    let pred = make_fragment::<bool>(src.shape().clone());
    assert_eq!(pred.size(), flags.len(), "{func}: the source and destination differ in size");
    for (i, f) in flags.into_iter().enumerate() {
        pred.store(i, f);
    }
    Some(pred)
}

/// One flag per element of `t` (in its index order): whether the element is
/// inside its predication origin; `None` without an origin or when every
/// element is statically inside.
#[track_caller]
fn origin_flags<T>(t: &Tensor<T>, func: &str, role: &str) -> Option<Vec<Boolean>> {
    let origin = t.origin.as_ref()?;
    let Some((coords, shape)) = origin.replay() else {
        panic!(
            "{func}: the {role} {t:?} was viewed from a global tensor through an operation that does not keep track \
             of where its elements lie in that tensor (a coalesce, a product, or a layout set whole), so its bounds \
             are unknown; mark the view with experimental::predicated_tensor_origin to predicate it by its own shape"
        )
    };
    let n = t.size();
    assert_eq!(coords.size(), n, "{func}: the {role}'s predication coordinates do not match its size");
    let extents: Option<Vec<usize>> = shape.leaves().iter().map(|e| e.as_static()).collect();
    let statically_inside = |i: usize| {
        let Some(ext) = &extents else { return false };
        let c = coords.coord(i);
        let leaves = c.leaves();
        leaves.len() == ext.len() && leaves.iter().zip(ext).all(|(l, &e)| l.as_static().is_some_and(|v| v < e))
    };
    if (0..n).all(statically_inside) {
        return None;
    }
    Some((0..n).map(|i| crate::elem_less(coords.coord(i), shape.clone())).collect())
}
