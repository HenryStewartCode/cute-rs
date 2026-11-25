//! cute_ext's algorithms: [`partition`], [`partition_and_copy`],
//! [`simt_auto_vec_copy`], [`warp_reduce`] and [`elementwise`].

use super::core::OperationType;
use super::memory::auto_predicate;
use crate::arch::{lane_idx, shuffle_sync_bfly};
use crate::atom::copy::{CopyAtom, CopyOp, ThrCopy, autovec_plan};
use crate::builder;
use crate::ir::MemSpace;
use crate::layout::{IndexExpr, Layout, Tileable, Tiler, Tree, zipped_divide};
use crate::tensor::{Tensor, make_fragment_like};
use crate::value::{Numeric, Value, select};

/// Agent `agent_id`'s slice of `buffer` (cute_ext's `partition(buffer,
/// agent_id, *, layout_tv, tiler)`): `buffer` is divided into `tiler`
/// tiles, each tile is viewed through `layout_tv` (`(agent, value) ->`
/// index in the tile, as a tiled copy's `layout_src_tv_tiled` and
/// `layout_dst_tv_tiled` are), and the agent keeps its values of every
/// tile: `(V, REST)`, where `V` is `layout_tv`'s value mode and `REST` the
/// tile counts, as one mode. The view keeps the buffer's predication
/// origin.
#[track_caller]
pub fn partition<T>(
    buffer: &Tensor<T>,
    agent_id: impl Into<IndexExpr>,
    layout_tv: &Layout,
    tiler: impl Into<Tiler>,
) -> Tensor<T> {
    let agent = agent_id.into();
    assert_eq!(
        layout_tv.rank(),
        2,
        "experimental::partition: layout_tv {layout_tv} must have two modes, (agent, value)"
    );
    let divided = zipped_divide(buffer, tiler);
    let tiler: Tiler = Tree::Branch(vec![Tree::Leaf(Some(layout_tv.clone())), Tree::Leaf(None)]);
    let layout = divided.layout().compose_tiler(&tiler);
    let viewed = divided.map_layout(layout, move |l| l.compose_tiler(&tiler));
    viewed.slice(Tree::Branch(vec![Tree::Branch(vec![Tree::Leaf(Some(agent)), Tree::Leaf(None)]), Tree::Leaf(None)]))
}

/// Whether `t` is per-thread storage: a register fragment or a per-thread
/// array.
fn is_registers<T>(t: &Tensor<T>) -> bool {
    t.is_register() || t.memory_space() == Some(MemSpace::Local)
}

/// Partition `src` and `dst` for `thr_copy`'s thread and copy (cute_ext's
/// `partition_and_copy(thr_copy, src, dst)`). The operands in memory are
/// partitioned by the tiled copy's source and destination TV layouts and
/// tile (`partition_s`, `partition_d`); a register operand (a fragment in
/// the copy's value order, such as an MMA fragment) is retiled for it. An
/// SMEM -> TMEM copy (`tcgen05.cp`) first gives the shared-memory operand
/// its broadcast mode ([`append_s2t_broadcast_mode`](crate::nvgpu::tcgen05::append_s2t_broadcast_mode)).
/// Then:
///
/// - `ldmatrix` and `stmatrix` atoms copy with the atom itself;
/// - copies between registers and shared or global memory go through
///   [`simt_auto_vec_copy`];
/// - global -> shared copies through its asynchronous form (`cp.async`);
/// - anything else through [`copy`](super::copy) with an atom of the tiled
///   copy's operation over `T`.
///
/// Every path predicates automatically, as [`copy`](super::copy) does.
#[track_caller]
pub fn partition_and_copy<T: Numeric>(thr_copy: &ThrCopy<T>, src: &Tensor<T>, dst: &Tensor<T>) {
    let _at = builder::LocationScope::new();
    let tiled = thr_copy.tiled_copy();
    let op = tiled.atom().op();
    let src = match op {
        CopyOp::TmemCp { .. } if src.memory_space() == Some(MemSpace::Shared) => {
            crate::nvgpu::tcgen05::append_s2t_broadcast_mode(op, src)
        }
        _ => src.clone(),
    };
    // A tcgen05 TMEM load's (store's) register operand is
    // the thread's partition of the destination (source) as it is: the
    // TMEM side's values span the warp's lanes, which a retile would
    // broadcast the registers over.
    let tmem_ld = matches!(op, CopyOp::TmemLoad { .. });
    let tmem_st = matches!(op, CopyOp::TmemStore { .. });
    let s = if is_registers(&src) && !tmem_st {
        tiled.retile(&src)
    } else if is_registers(&src) {
        src.clone()
    } else {
        thr_copy.partition_s(&src)
    };
    let d = if is_registers(dst) && !tmem_ld {
        tiled.retile(dst)
    } else if is_registers(dst) {
        dst.clone()
    } else {
        thr_copy.partition_d(dst)
    };
    let memory = |t: &Tensor<T>| matches!(t.memory_space(), Some(MemSpace::Shared | MemSpace::Global));
    let matrix = matches!(
        op,
        CopyOp::LdMatrix { .. }
            | CopyOp::LdMatrix8x16x8b { .. }
            | CopyOp::LdMatrix16x16x8b { .. }
            | CopyOp::LdMatrix16x8x8b { .. }
            | CopyOp::StMatrix { .. }
            | CopyOp::StMatrix16x8x8b { .. }
    );
    if matrix {
        super::copy(&s, &d, tiled);
    } else if (is_registers(&s) && memory(&d)) || (memory(&s) && is_registers(&d)) {
        simt_auto_vec_copy(&s, &d, false);
    } else if s.memory_space() == Some(MemSpace::Global) && d.memory_space() == Some(MemSpace::Shared) {
        simt_auto_vec_copy(&s, &d, true);
    } else {
        super::copy(&s, &d, &CopyAtom::<T>::new(op));
    }
}

/// An automatically vectorised SIMT copy (cute_ext's
/// `simt_auto_vec_copy(src, dst, async_op)`): the widest vectors both
/// operands allow, as [`autovec_copy`](crate::autovec_copy) chooses them.
/// With `async_op`, a global -> shared copy issues `cp.async` (cache
/// "always", `.ca`) of those vectors, which must be 4, 8 or 16 bytes;
/// narrower runs are copied synchronously.
///
/// The copy predicates automatically, as [`copy`](super::copy) does. A
/// predicated copy that is not statically in bounds goes element by
/// element, since a vector could straddle the bound; the asynchronous form
/// then copies synchronously too and zero-fills the elements predicated
/// off, as `cp.async` does.
#[track_caller]
pub fn simt_auto_vec_copy<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, async_op: bool) {
    let _at = builder::LocationScope::new();
    let func = "experimental::simt_auto_vec_copy";
    if async_op {
        assert!(
            src.memory_space() == Some(MemSpace::Global) && dst.memory_space() == Some(MemSpace::Shared),
            "{func}: async_op copies global memory to shared memory, not {} to {}",
            src.memspace(),
            dst.memspace()
        );
    }
    if let Some(pred) = auto_predicate(src, dst, func) {
        crate::basic_copy_if(&pred, src, dst);
        if async_op {
            let zeros = make_fragment_like::<T, T>(src);
            let loc = std::panic::Location::caller();
            let zero = Value::<T>::from_id(builder::with(|b| b.constant(crate::ir::Literal::zero(T::DTYPE), loc)));
            zeros.fill(zero);
            let off = make_fragment_like::<bool, T>(src);
            for i in 0..pred.size() {
                off.store(i, !pred.load(i));
            }
            crate::basic_copy_if(&off, &zeros, dst);
        }
        return;
    }
    let eb = T::DTYPE.storage_bits();
    if async_op
        && let Some((s, d, v)) = autovec_plan(src, dst, 128)
        && v * eb >= 32
    {
        let atom = CopyAtom::<T>::new(CopyOp::CpAsync { bytes: v * eb / 8, bypass_l1: false });
        crate::copy(&atom, &s, &d);
        return;
    }
    crate::autovec_copy(src, dst);
}

/// A [`warp_reduce`] lane map, as the lane (xor) offsets of the bits of the
/// logical lane.
struct ReduceMap {
    /// The xor offset of each bit of the logical lane, low to high.
    offsets: Vec<usize>,
}

#[track_caller]
fn reduce_map(map: &Layout, func: &str) -> ReduceMap {
    let modes = map.flatten().flat_modes();
    for &(n, _) in &modes {
        assert!(n.is_power_of_two(), "{func}: reduction_t_map {map} has a mode of {n} lanes, not a power of two");
    }
    let domain: usize = modes.iter().map(|&(n, _)| n).product();
    assert!(domain == 32, "{func}: reduction_t_map {map} must map the 32 lanes of a warp, but covers {domain}");
    // (logical stride, size, physical stride) of each reducing mode.
    let mut reducing = Vec::new();
    let mut phys = 1;
    for &(n, d) in &modes {
        if d != 0 && n > 1 {
            reducing.push((d, n, phys));
        }
        phys *= n;
    }
    reducing.sort();
    let mut offsets = Vec::new();
    let mut expect = 1;
    for &(d, n, p) in &reducing {
        assert_eq!(
            d, expect,
            "{func}: reduction_t_map {map} must number the lanes of each reduction 0, 1, 2, ... (its non-zero \
             strides compact)"
        );
        for j in 0..n.trailing_zeros() {
            offsets.push(p << j);
        }
        expect *= n;
    }
    ReduceMap { offsets }
}

/// A reduction across the lanes of a warp (cute_ext's `warp_reduce(input,
/// output, reduction_t_map, intra_warp_reduce_type, body)`), with `body` the
/// reduction's binary operation (associative and commutative).
///
/// `reduction_t_map` maps the 32 lanes to the logical lane within each
/// reduction: its modes (flattened) are powers of two, covering 32 lanes;
/// the modes of stride 0 select independent groups, and the others number
/// the lanes of a group compactly. `32:1` is one reduction over the warp;
/// `(8,4):(1,0)` four reductions over lanes 0-7, 8-15, 16-23 and 24-31. A
/// group of `n` lanes is reduced with `log2(n)` butterfly shuffles.
///
/// - [`OperationType::ReduceDownShuffle`]: `input` and `output` have the
///   same size; every lane of a group ends with the reduction of the
///   group's `input` element by element.
/// - [`OperationType::ReduceSwapShuffle`]: a transposing reduce-scatter.
///   `input` holds `n` slices of `output.size()` elements each (slice `s`
///   is elements `s * output.size() ..` in index order), and lane `i` of a
///   group ends with the reduction over the group of slice `i`.
///
/// Every lane of the warp must call it (the shuffles are warp-wide); the
/// simulator checks that the shuffles are reached by the whole warp.
#[track_caller]
pub fn warp_reduce<T: Numeric>(
    input: &Tensor<T>,
    output: &Tensor<T>,
    reduction_t_map: &Layout,
    intra_warp_reduce_type: OperationType,
    body: impl Fn(Value<T>, Value<T>) -> Value<T>,
) {
    let _at = builder::LocationScope::new();
    let func = "experimental::warp_reduce";
    let map = reduce_map(reduction_t_map, func);
    let n = 1usize << map.offsets.len();
    let values: Vec<Value<T>> = (0..input.size()).map(|i| input.load(i)).collect();
    match intra_warp_reduce_type {
        OperationType::ReduceDownShuffle => {
            assert_eq!(
                input.size(),
                output.size(),
                "{func}: a down-shuffle reduction keeps the size: the input has {} elements, the output {}",
                input.size(),
                output.size()
            );
            for (i, mut v) in values.into_iter().enumerate() {
                for &off in &map.offsets {
                    v = body(v, shuffle_sync_bfly(v, off as i32));
                }
                output.store(i, v);
            }
        }
        OperationType::ReduceSwapShuffle => {
            let frag = output.size();
            assert_eq!(
                input.size(),
                n * frag,
                "{func}: a swap-shuffle reduction over {n} lanes takes {n} slices of the output's {frag} elements, {} \
                 in all, but the input has {}",
                n * frag,
                input.size()
            );
            let lane = lane_idx();
            let mut cur = values;
            // The highest logical bit first: lanes with it set keep the
            // upper half of the slices, their partners the lower half.
            for &off in map.offsets.iter().rev() {
                let half = cur.len() / 2;
                let upper = ((lane / off as i32) % 2).eq(1);
                let mut next = Vec::with_capacity(half);
                for j in 0..half {
                    let (lo, hi) = (cur[j], cur[j + half]);
                    let send = select(upper, lo, hi);
                    let keep = select(upper, hi, lo);
                    next.push(body(keep, shuffle_sync_bfly(send, off as i32)));
                }
                cur = next;
            }
            for (i, v) in cur.into_iter().enumerate() {
                output.store(i, v);
            }
        }
        other => {
            panic!("{func}: intra_warp_reduce_type must be REDUCE_DOWN_SHUFFLE or REDUCE_SWAP_SHUFFLE, not {other}")
        }
    }
}

/// Tensors that [`elementwise`] walks together: tuples of one to four
/// tensors, whose values reach the body as a tuple of the same arity.
pub trait ElementwiseTensors {
    /// One element of each tensor.
    type Values;
    /// The tensors' sizes.
    fn sizes(&self) -> Vec<usize>;
    /// Element `i` of each tensor (registers never written read as zero).
    fn load(&self, i: usize) -> Self::Values;
    /// Store element `i` of each tensor.
    fn store(&self, i: usize, values: Self::Values);
}

macro_rules! elementwise_tuple {
    ($($t:ident $i:tt),+) => {
        impl<'a, $($t: Numeric),+> ElementwiseTensors for ($(&'a Tensor<$t>,)+) {
            type Values = ($(Value<$t>,)+);
            fn sizes(&self) -> Vec<usize> {
                vec![$(self.$i.size()),+]
            }
            #[track_caller]
            fn load(&self, i: usize) -> Self::Values {
                ($(self.$i.load_or_zero(i),)+)
            }
            #[track_caller]
            fn store(&self, i: usize, values: Self::Values) {
                $(self.$i.store(i, values.$i);)+
            }
        }
    };
}

elementwise_tuple!(A 0);
elementwise_tuple!(A 0, B 1);
elementwise_tuple!(A 0, B 1, C 2);
elementwise_tuple!(A 0, B 1, C 2, D 3);

/// Apply `body` element by element across `inputs` and `outputs`, tensors
/// of one size (cute_ext's `elementwise(inputs, outputs, body)`): for each
/// element `body` receives the inputs' values and the outputs' current
/// values (zero for registers not yet written), and returns the outputs'
/// new values. The tensors are walked in their index order, unrolled when
/// the kernel is built, so their extents must be static.
///
/// ```
/// use cute::prelude::*;
/// use cute::experimental::elementwise;
///
/// let kernel = Kernel::builder("axpy").block_dim([1, 1, 1]).build(|| {
///     let x = param_tensor::<f32>("x", Layout::col_major(8));
///     let y = param_tensor::<f32>("y", Layout::col_major(8));
///     elementwise((&x,), (&y,), |(x,), (y,)| (x * 2.0f32 + y,));
/// });
/// let mut args = vec![KernelArg::f32(vec![1.0; 8]), KernelArg::f32(vec![3.0; 8])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[1].as_f32(), vec![5.0; 8]);
/// ```
#[track_caller]
pub fn elementwise<I: ElementwiseTensors, O: ElementwiseTensors>(
    inputs: I,
    outputs: O,
    mut body: impl FnMut(I::Values, O::Values) -> O::Values,
) {
    let _at = builder::LocationScope::new();
    let sizes: Vec<usize> = inputs.sizes().into_iter().chain(outputs.sizes()).collect();
    let n = sizes[0];
    assert!(
        sizes.iter().all(|&s| s == n),
        "experimental::elementwise: the tensors differ in size ({sizes:?}); they take one shape"
    );
    for i in 0..n {
        let new = body(inputs.load(i), outputs.load(i));
        outputs.store(i, new);
    }
}
