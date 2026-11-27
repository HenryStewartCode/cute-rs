//! cute_ext's GEMM building blocks (CuTe
//! DSL's `cute.experimental.utils`): reusable kernel fragments over the rest
//! of cute_ext ([`allocate`], [`tma_load`](super::tma_load)
//! and [`tma_store`], the pipeline classes, [`dot`](super::dot)
//! and the tcgen05 TMEM copies).
//!
//! - [`is_preferred_cluster`]: whether the launch gave this block's cluster
//!   the preferred shape (rather than the fallback one).
//! - [`get_cta_v_map_ab`] and [`get_cta_v_map_c`]: the CTA-V maps (buffer
//!   element -> global coordinate) a TMA load of an MMA operand or a TMA
//!   store of an epilogue tile uses, for [`TmaLoadOptions::cta_v_map`](super::TmaLoadOptions::cta_v_map).
//! - [`make_tmem_layout_acc`] and [`make_tmem_layout_a`]: staged
//!   tensor-memory layouts of the accumulator and of A, for
//!   [`allocate`] in tensor memory.
//! - [`make_t2r_rmem_layout`]: the per-thread register layout one epilogue
//!   TMEM load (`tcgen05.ld`) fills.
//! - [`mainloop_mma`]: the tcgen05 mainloop over the stages of a TMA -> UMMA
//!   pipeline; [`mainloop_mma_sm120`]: SM120's warp-level block-scaled inner
//!   K loop.
//! - [`epilogue_tma_store`] (and [`epilogue_tma_store_with`],
//!   [`epilogue_tma_store_drain`]): the TMEM -> registers -> shared memory
//!   -> TMA store epilogue through a [`TMAStorePipeline`], with an optional
//!   amax reduction and the drain order of an [`AccumulatorDrain`].
//!
//! The semantics follow CuTe DSL 4.8.1's documented behaviour; the
//! implementation is cute-rs's own.

use super::accumulator_buffering::AccumulatorDrain;
use super::algorithm::partition_and_copy;
use super::memory::allocate;
use super::pipeline::{AsGenericPipeline, TMAStorePipeline};
use super::pipeline_ops::{is_2sm, is_umma};
use super::tma::tma_store;
use crate::arch::{
    ReduxOptions, atomic_fmax_with, fence_view_async_tmem_load, fmax, lane_idx, make_warp_uniform, thread_idx,
    warp_idx, warp_redux_sync_with,
};
use crate::atom::copy::{CopyOp, TiledCopy, make_copy_atom, make_tiled_copy_d};
use crate::atom::mma::{MmaOp, Operand, TiledMma};
use crate::builder;
use crate::control::{for_range_carry, if_};
use crate::ir::{DType, ReduxKind};
use crate::layout::{IndexExpr, Layout, Shape, Tiler, Tree, flat_divide, make_layout_like};
use crate::nvgpu::tcgen05::{Field, make_tmem_copy};
use crate::tensor::{AddressSpace, Tensor, local_tile};
use crate::tensor_ssa::{ReductionOp, TensorSsa};
use crate::utils::LayoutEnum;
use crate::utils::sm100::{get_tmem_load_op, make_smem_layout_epi};
use crate::value::{Boolean, Float32, Int32, IntoValue, Numeric};

/// The epilogue's threads: one warpgroup.
const THREADS_PER_WARPGROUP: i32 = 128;

/// Whether this block's cluster has the shape `preferred_shape` (CuTe DSL's
/// `cute.experimental.utils.is_preferred_cluster`), rather than the
/// fallback shape of a launch with two
/// ([`LaunchConfig::with_fallback_cluster`](crate::LaunchConfig::with_fallback_cluster)).
/// See [`mixed_cluster_arms`](crate::utils::mixed_cluster_arms) for
/// building a kernel body per shape.
#[track_caller]
pub fn is_preferred_cluster(preferred_shape: [usize; 3]) -> Boolean {
    crate::utils::is_preferred_cluster(preferred_shape)
}

/// The MMA operand a CTA-V map is made for ([`get_cta_v_map_ab`]'s
/// `input_operand`: `'A'`, `'B'`, `'SFA'` or `'SFB'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MmaOperand {
    A,
    B,
    SFA,
    SFB,
}

impl MmaOperand {
    fn name(self) -> &'static str {
        match self {
            MmaOperand::A => "A",
            MmaOperand::B => "B",
            MmaOperand::SFA => "SFA",
            MmaOperand::SFB => "SFB",
        }
    }
}

/// The CTA-V map of a TMA load of MMA operand `input_operand` of
/// `gmem_tensor` (CuTe DSL's `get_cta_v_map_ab(gmem_tensor, mma_tiler_mnk,
/// tiled_mma, input_operand)`): the identity layout of the tensor's shape,
/// composed with the operand's tile of `mma_tiler_mnk` (`(M, K)` for A and
/// SFA, `(N, K)` for B and SFB), partitioned by the MMA (`thrfrg`) and
/// reduced to its value mode, flattened to `(FrgV, Rest_MN, Rest_K)`. That
/// maps each element of the CTA's shared buffer to the global coordinate
/// it holds, in the order the MMA reads it. For an SM120 block-scaled
/// warp MMA, whose operands go through `ldmatrix`, the map is the tile
/// itself.
///
/// Panics for a scale-factor operand of an MMA that takes none, and for a
/// tensor of rank below 2.
#[track_caller]
pub fn get_cta_v_map_ab<T>(
    gmem_tensor: &Tensor<T>,
    mma_tiler_mnk: (usize, usize, usize),
    tiled_mma: &TiledMma,
    input_operand: MmaOperand,
) -> Layout {
    let op = tiled_mma.atom().op();
    let scaled = match op {
        MmaOp::Sm120(_) => true,
        MmaOp::Sm100(kind) => kind.scale.is_some(),
        _ => false,
    };
    if matches!(input_operand, MmaOperand::SFA | MmaOperand::SFB) && !scaled {
        builder::user_panic(
            std::panic::Location::caller(),
            format!(
                "experimental::get_cta_v_map_ab: {op:?} is not a block-scaled MMA, so it has no {} operand to load",
                input_operand.name()
            ),
        );
    }
    let glayout = gmem_tensor.layout();
    assert!(
        glayout.rank() >= 2,
        "experimental::get_cta_v_map_ab: the global tensor {glayout} has rank {}, not at least 2",
        glayout.rank()
    );
    let (which, mn) = match input_operand {
        MmaOperand::A | MmaOperand::SFA => (Operand::A, mma_tiler_mnk.0),
        MmaOperand::B | MmaOperand::SFB => (Operand::B, mma_tiler_mnk.1),
    };
    let tiler: Tiler = (mn, mma_tiler_mnk.2).into();
    let g_tile = crate::nvgpu::tma_atom::tiler_v_map(glayout, &tiler)
        .unwrap_or_else(|e| panic!("experimental::get_cta_v_map_ab: {e}"));
    if let MmaOp::Sm120(_) = op {
        return g_tile;
    }
    // The value mode of the MMA's partition, flattened to (FrgV, Rest_MN, Rest_K).
    let frg = tiled_mma.thrfrg(&g_tile, which).mode(1);
    let rest = frg.mode(1);
    Layout::from_modes([frg.mode(0), rest.mode(0), rest.mode(1)])
}

/// The CTA-V map of a TMA store of an epilogue tile (CuTe DSL's
/// `get_cta_v_map_c(gmem_tensor, epi_tile)`): the identity layout of
/// `gmem_tensor`'s shape composed with `epi_tile`.
#[track_caller]
pub fn get_cta_v_map_c<T>(gmem_tensor: &Tensor<T>, epi_tile: impl Into<Tiler>) -> Layout {
    crate::nvgpu::tma_atom::tiler_v_map(gmem_tensor.layout(), &epi_tile.into())
        .unwrap_or_else(|e| panic!("experimental::get_cta_v_map_c: {e}"))
}

/// `shape` with `extra` appended as a last mode.
fn append_mode(shape: Shape, extra: usize) -> Shape {
    let mut modes = match shape {
        Tree::Branch(m) => m,
        leaf => vec![leaf],
    };
    modes.push(Shape::from(extra));
    Tree::Branch(modes)
}

/// The tensor-memory layout of `acc_stage` accumulator stages of a tcgen05
/// tiled MMA over the `(M, N)` of `mnk_tiler` (CuTe DSL's
/// `make_tmem_layout_acc(tiled_mma, mnk_tiler, acc_stage)`): the layout of
/// `make_fragment_c` of `partition_shape_c((M, N))` with the stages
/// appended, `((M, N), MMA_M, MMA_N, STAGE)` (one CTA's half of a pair's
/// accumulator for a 2SM MMA). Give it to [`allocate`] in
/// tensor memory.
#[track_caller]
pub fn make_tmem_layout_acc(tiled_mma: &TiledMma, mnk_tiler: (usize, usize, usize), acc_stage: usize) -> Layout {
    check_tcgen05(tiled_mma, "make_tmem_layout_acc");
    let shape = append_mode(tiled_mma.partition_shape_c((mnk_tiler.0, mnk_tiler.1)), acc_stage);
    let (_, _, c) = tiled_mma.atom().dtypes();
    match c {
        DType::F32 => tiled_mma.make_fragment_c::<f32>(&shape).layout().clone(),
        DType::I32 => tiled_mma.make_fragment_c::<i32>(&shape).layout().clone(),
        DType::F16 => tiled_mma.make_fragment_c::<crate::F16>(&shape).layout().clone(),
        other => panic!("experimental::make_tmem_layout_acc: no accumulator layout for {other}"),
    }
}

/// The tensor-memory layout of `stage` stages of A for a tcgen05 MMA that
/// reads A from tensor memory (CuTe DSL's `make_tmem_layout_a(tiled_mma,
/// mk_tiler, stage)`): the layout of `make_fragment_a` of
/// `partition_shape_a(mk_tiler)` with the stages appended.
#[track_caller]
pub fn make_tmem_layout_a(tiled_mma: &TiledMma, mk_tiler: (usize, usize), stage: usize) -> Layout {
    check_tcgen05(tiled_mma, "make_tmem_layout_a");
    let shape = append_mode(tiled_mma.partition_shape_a(mk_tiler), stage);
    let MmaOp::Sm100(kind) = tiled_mma.atom().op() else { unreachable!("a tcgen05 MMA") };
    assert!(
        kind.a_tmem,
        "experimental::make_tmem_layout_a: {} reads A from shared memory (a_src=OperandSource.SMEM), not tensor memory",
        kind.ptx()
    );
    // CuTe C++'s `UMMA::tmem_frg_1sm` of A (make_fragment_a's TMEM form).
    let first = shape.mode(0).size();
    let (m, k) = (kind.m as usize, kind.k());
    assert!(
        first == m * k,
        "experimental::make_tmem_layout_a: the partition {shape} is not ((M, K), ...) of the {m} x {k} A tile"
    );
    let rest: Vec<usize> = (1..shape.rank()).map(|i| shape.mode(i).size()).collect();
    crate::nvgpu::tcgen05::mma::tmem_frg_a_layout(m, k, &rest, kind.a.storage_bits())
}

#[track_caller]
fn check_tcgen05(tiled_mma: &TiledMma, what: &str) {
    assert!(
        tiled_mma.atom().is_tcgen05(),
        "experimental::{what}: {:?} is not a tcgen05 MMA, whose operands live in tensor memory",
        tiled_mma.atom().op()
    );
}

/// The register layout of one epilogue TMEM load (CuTe DSL's
/// `make_t2r_rmem_layout(tiled_copy_t2r, gC_mnl_epi, tidx)`): thread
/// `tidx`'s destination partition of `g_c_mnl_epi` (C divided into
/// epilogue subtiles, `(EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, ...)`) at
/// subtile `(0, 0)`, `(T2R, T2R_M, T2R_N)`, as a compact layout of its
/// shape.
#[track_caller]
pub fn make_t2r_rmem_layout<A: Numeric, T>(
    tiled_copy_t2r: &TiledCopy<A>,
    g_c_mnl_epi: &Tensor<T>,
    tidx: impl Into<IndexExpr>,
) -> Layout {
    let part = tiled_copy_t2r.get_slice(tidx).partition_d(g_c_mnl_epi);
    let rest = part.rank().saturating_sub(3);
    make_layout_like(slice_after(&part, 3, &vec![IndexExpr::from(0); rest]).layout())
}

/// `t[(.., .. (nones of them), fixed...)]`.
fn slice_after<T>(t: &Tensor<T>, nones: usize, fixed: &[IndexExpr]) -> Tensor<T> {
    let mut coords: Vec<Tree<Option<IndexExpr>>> = vec![Tree::Leaf(None); nones];
    coords.extend(fixed.iter().cloned().map(|i| Tree::Leaf(Some(i))));
    t.slice(Tree::Branch(coords))
}

/// The pipelines [`mainloop_mma`] consumes: the UMMA side of a TMA -> UMMA
/// (or thread -> UMMA) pipeline.
fn check_mainloop_pipeline<P: AsGenericPipeline, T>(
    p: &P,
    buffer: &Tensor<T>,
    name: &str,
    tiled_mma: &TiledMma,
    loc: &'static std::panic::Location<'static>,
) {
    let raw = p.base().raw_pipeline();
    let fail = |m: String| -> ! { builder::user_panic(loc, format!("experimental::mainloop_mma: {m}")) };
    let umma: Vec<_> = raw.consumer().ops().iter().copied().filter(|&o| is_umma(o)).collect();
    if umma.is_empty() {
        fail(format!(
            "the {name} pipeline feeds {}, not a UMMA operation: give the pipeline the MMA warp consumes",
            raw.consumer()
        ));
    }
    let MmaOp::Sm100(kind) = tiled_mma.atom().op() else { unreachable!("a tcgen05 MMA") };
    if umma.iter().any(|&o| is_2sm(o) != kind.two_cta) {
        fail(format!(
            "the {name} pipeline's MMA ({}) and the tiled MMA ({}) disagree on the CTA group: a 2SM MMA consumes a \
             2SM pipeline, a 1SM MMA a 1SM one",
            raw.consumer(),
            if kind.two_cta { "2SM" } else { "1SM" }
        ));
    }
    let rank = buffer.rank();
    let stages = if rank == 0 { 0 } else { buffer.layout().mode(rank - 1).size() };
    if stages != raw.num_stages() {
        fail(format!(
            "the {name} buffer {} has {stages} stages (its last mode) but its pipeline {}",
            buffer.layout(),
            raw.num_stages()
        ));
    }
}

/// The tcgen05 mainloop of one output tile (CuTe DSL's
/// `mainloop_mma(tiled_mma, a_buffer, b_buffer, acc_buffer, k_tile_start,
/// k_tile_end, mma_inst_tile_k, a_buffer_pipeline, b_buffer_pipeline,
/// ab_buffer_same_pipeline=True, accumulate_to_acc=False)`), run by the MMA
/// warp (the leader CTA's, for a 2SM MMA).
///
/// `a_buffer` and `b_buffer` are the staged operand buffers, `(MMA, MMA_MN,
/// MMA_K, STAGE)` with `MMA_K` at least `mma_inst_tile_k` (shared memory:
/// [`make_smem_layout_a`](crate::utils::sm100::make_smem_layout_a) through
/// [`allocate`]; or A in tensor memory), and `acc_buffer`
/// the accumulator, `((M, N), MMA_M, MMA_N)`. For each k-tile from
/// `k_tile_start` to `k_tile_end` (runtime values), the loop waits for the
/// A stage (and the B stage, from `b_buffer_pipeline`, unless
/// `ab_buffer_same_pipeline`), issues `mma_inst_tile_k` MMAs over its
/// k-blocks ([`dot`](super::dot)), the first without accumulating unless
/// `accumulate_to_acc`, and releases the stage (a `tcgen05.commit` to the
/// stage's empty barrier). Returns the pipelines, advanced; with
/// `ab_buffer_same_pipeline` both are the advanced A pipeline.
///
/// Panics when a pipeline does not feed a UMMA operation of the tiled MMA's
/// CTA group, when a buffer's stage count is not its pipeline's, and when
/// the shared pipelines differ in stages.
#[track_caller]
pub fn mainloop_mma<P: AsGenericPipeline, A: Numeric, B: Numeric, C: Numeric>(
    tiled_mma: &TiledMma,
    a_buffer: &Tensor<A>,
    b_buffer: &Tensor<B>,
    acc_buffer: &Tensor<C>,
    k_tile_start: impl IntoValue<i32>,
    k_tile_end: impl IntoValue<i32>,
    mma_inst_tile_k: usize,
    a_buffer_pipeline: P,
    b_buffer_pipeline: P,
    ab_buffer_same_pipeline: bool,
    accumulate_to_acc: bool,
) -> (P, P) {
    let loc = std::panic::Location::caller();
    let _at = builder::LocationScope::new();
    check_tcgen05(tiled_mma, "mainloop_mma");
    check_mainloop_pipeline(&a_buffer_pipeline, a_buffer, "A", tiled_mma, loc);
    check_mainloop_pipeline(&b_buffer_pipeline, b_buffer, "B", tiled_mma, loc);
    let (ra, rb) = (a_buffer_pipeline.base().raw_pipeline(), b_buffer_pipeline.base().raw_pipeline());
    if ab_buffer_same_pipeline && (ra.num_stages() != rb.num_stages() || ra.consumer() != rb.consumer()) {
        builder::user_panic(
            loc,
            format!(
                "experimental::mainloop_mma: ab_buffer_same_pipeline says A and B share one pipeline, but the A \
                 pipeline ({} stages, {} -> {}) and the B pipeline ({} stages, {} -> {}) differ",
                ra.num_stages(),
                ra.producer(),
                ra.consumer(),
                rb.num_stages(),
                rb.producer(),
                rb.consumer()
            ),
        );
    }
    for (name, rank, kblocks) in [
        ("A", a_buffer.rank(), a_buffer.layout().mode(a_buffer.rank().saturating_sub(2)).size()),
        ("B", b_buffer.rank(), b_buffer.layout().mode(b_buffer.rank().saturating_sub(2)).size()),
    ] {
        if rank < 3 || kblocks < mma_inst_tile_k {
            builder::user_panic(
                loc,
                format!(
                    "experimental::mainloop_mma: the {name} buffer must be (MMA, MMA_MN, MMA_K, STAGE) with MMA_K at \
                     least mma_inst_tile_k ({mma_inst_tile_k}); it has {kblocks} k-blocks in rank {rank}"
                ),
            );
        }
    }
    // The descriptor fragments of shared-memory operands (CuTe DSL's dot
    // takes the buffers themselves).
    let a_frag = if a_buffer.memory_space() == Some(crate::ir::MemSpace::Shared) {
        tiled_mma.make_fragment_a(a_buffer)
    } else {
        a_buffer.clone()
    };
    let b_frag = tiled_mma.make_fragment_b(b_buffer);
    let mut mma = tiled_mma.clone();
    mma.set(Field::ACCUMULATE, accumulate_to_acc);
    let (a, b, _) = for_range_carry(
        k_tile_start,
        k_tile_end,
        1,
        (a_buffer_pipeline, b_buffer_pipeline, mma),
        |_, (mut a, mut b, mut mma)| {
            let (_, a_stage) = a.base_mut().consumer_wait_and_get_stage(None);
            let b_stage =
                if ab_buffer_same_pipeline { a_stage } else { b.base_mut().consumer_wait_and_get_stage(None).1 };
            for k in 0..mma_inst_tile_k {
                let fa = slice_last2(&a_frag, k, a_stage);
                let fb = slice_last2(&b_frag, k, b_stage);
                super::dot(&mma, &fa, &fb, acc_buffer);
                mma.set(Field::ACCUMULATE, true);
            }
            a.base_mut().consumer_release_and_advance();
            if !ab_buffer_same_pipeline {
                b.base_mut().consumer_release_and_advance();
            }
            (a, b, mma)
        },
    );
    let b = if ab_buffer_same_pipeline { a.clone() } else { b };
    (a, b)
}

/// `t[.., .., k, stage]` of a `(MMA, MMA_MN, MMA_K, STAGE)` tensor.
fn slice_last2<T>(t: &Tensor<T>, k: usize, stage: Int32) -> Tensor<T> {
    let nones = t.rank() - 2;
    slice_after(t, nones, &[IndexExpr::from(k), IndexExpr::from(stage)])
}

/// SM120's warp-level block-scaled inner K loop (CuTe DSL's
/// `mainloop_mma_sm120(tiled_mma, tCrA, tCrSFA, tCrB, tCrSFB, acc_buffer,
/// mma_inst_tile_k, accumulate_to_acc=True)`): `mma_inst_tile_k`
/// block-scaled MMAs ([`dot_block_scaled`](super::dot_block_scaled)) over
/// the last mode of the register fragments `t_cr_a`, `t_cr_sfa`, `t_cr_b`
/// and `t_cr_sfb` (`(..., K)`), into the register accumulator
/// `acc_buffer`, which is zeroed first unless `accumulate_to_acc`. The
/// caller waits for and releases the pipeline stage and loads the
/// fragments (`ldmatrix`).
#[track_caller]
pub fn mainloop_mma_sm120<A: Numeric, SA: Numeric, B: Numeric, SB: Numeric>(
    tiled_mma: &TiledMma,
    t_cr_a: &Tensor<A>,
    t_cr_sfa: &Tensor<SA>,
    t_cr_b: &Tensor<B>,
    t_cr_sfb: &Tensor<SB>,
    acc_buffer: &Tensor<f32>,
    mma_inst_tile_k: usize,
    accumulate_to_acc: bool,
) {
    let _at = builder::LocationScope::new();
    assert!(
        matches!(tiled_mma.atom().op(), MmaOp::Sm120(_)),
        "experimental::mainloop_mma_sm120: {:?} is not an SM120 block-scaled warp MMA",
        tiled_mma.atom().op()
    );
    for (name, rank, k) in [
        ("tCrA", t_cr_a.rank(), last_mode(t_cr_a)),
        ("tCrSFA", t_cr_sfa.rank(), last_mode(t_cr_sfa)),
        ("tCrB", t_cr_b.rank(), last_mode(t_cr_b)),
        ("tCrSFB", t_cr_sfb.rank(), last_mode(t_cr_sfb)),
    ] {
        assert!(
            rank >= 2 && k >= mma_inst_tile_k,
            "experimental::mainloop_mma_sm120: {name} must end in a K mode of at least mma_inst_tile_k \
             ({mma_inst_tile_k}) instructions; it has {k} in rank {rank}"
        );
    }
    if !accumulate_to_acc {
        acc_buffer.fill(0.0f32);
    }
    let at = |t_rank: usize, k: usize| {
        let mut c: Vec<Tree<Option<IndexExpr>>> = vec![Tree::Leaf(None); t_rank - 1];
        c.push(Tree::Leaf(Some(IndexExpr::from(k))));
        Tree::Branch(c)
    };
    for k in 0..mma_inst_tile_k {
        super::dot_block_scaled(
            tiled_mma,
            &t_cr_a.slice(at(t_cr_a.rank(), k)),
            &t_cr_sfa.slice(at(t_cr_sfa.rank(), k)),
            &t_cr_b.slice(at(t_cr_b.rank(), k)),
            &t_cr_sfb.slice(at(t_cr_sfb.rank(), k)),
            acc_buffer,
        );
    }
}

fn last_mode<T>(t: &Tensor<T>) -> usize {
    if t.rank() == 0 { 0 } else { t.layout().mode(t.rank() - 1).size() }
}

/// The keyword arguments of [`epilogue_tma_store_with`] (CuTe DSL's
/// `d_major_mode=`, `tid_x_in_group=`, `amax_out=`).
#[derive(Clone, Debug, Default)]
pub struct EpilogueTmaStoreOptions {
    /// D's majorness; by default [`LayoutEnum::from_tensor`] of D.
    pub d_major_mode: Option<LayoutEnum>,
    /// The thread's index in the epilogue warpgroup (0 to 127); by default
    /// `thread_idx().x % 128`.
    pub tid_x_in_group: Option<Int32>,
    /// A one-element Float32 global tensor that receives the maximum of
    /// the accumulator's absolute values (`atomic_fmax`; initialise it to
    /// 0). `None` builds no amax code.
    pub amax_out: Option<Tensor<f32>>,
}

/// The TMA-store epilogue of one output tile (CuTe DSL's
/// `epilogue_tma_store(cta_tile_shape_mnk, use_2cta_instrs,
/// tmem_acc_buffer_staged, gmem_d, cta_d_tile_coord, tma_store_pipeline,
/// tma_store_warp_id, epilogue_op)`), run by the epilogue warpgroup (the
/// four warps of [`TMAStorePipeline`]'s barrier) after the accumulator
/// stage is full.
///
/// `tmem_acc_buffer_staged` is one accumulator stage, the `(M, N)` CTA tile
/// in tensor memory (`acc[.., .., .., stage]` of [`make_tmem_layout_acc`]'s
/// buffer, or that tile as a rank-2 tensor); `gmem_d` is D, `(M, N)` or `(M, N,
/// L)`, and `cta_d_tile_coord` the CTA tile's coordinates in it (`(m, n)`
/// or `(m, n, l)`, as the tile scheduler gives them). The tile is divided
/// into the subtiles of [`compute_epilogue_tile_shape`](crate::utils::compute_epilogue_tile_shape),
/// and for each:
///
/// 1. the TMEM load ([`get_tmem_load_op`]) into registers, and its wait;
/// 2. the conversion to D's type and `epilogue_op` in registers;
/// 3. [`acquire_sync`](TMAStorePipeline::acquire_sync), the store of the
///    registers to the pipeline's stage of a staged shared buffer
///    ([`make_smem_layout_epi`]), [`commit_sync`](TMAStorePipeline::commit_sync);
/// 4. the TMA store of the stage by warp `tma_store_warp_id`, and
///    [`release_advance`](TMAStorePipeline::release_advance).
///
/// Last, [`tail`](TMAStorePipeline::tail). Returns the pipeline (carry it
/// to the next tile). The caller releases the accumulator stage.
///
/// Panics when the pipeline's barrier does not count the epilogue
/// warpgroup's 128 threads or its TMA warp is not `tma_store_warp_id`, and
/// when the accumulator is not the CTA tile.
#[track_caller]
pub fn epilogue_tma_store<Acc: Numeric, D: Numeric>(
    cta_tile_shape_mnk: (usize, usize, usize),
    use_2cta_instrs: bool,
    tmem_acc_buffer_staged: &Tensor<Acc>,
    gmem_d: &Tensor<D>,
    cta_d_tile_coord: &[Int32],
    tma_store_pipeline: TMAStorePipeline,
    tma_store_warp_id: i32,
    epilogue_op: impl Fn(TensorSsa<D>) -> TensorSsa<D>,
) -> TMAStorePipeline {
    epilogue_tma_store_with(
        cta_tile_shape_mnk,
        use_2cta_instrs,
        tmem_acc_buffer_staged,
        gmem_d,
        cta_d_tile_coord,
        tma_store_pipeline,
        tma_store_warp_id,
        epilogue_op,
        &EpilogueTmaStoreOptions::default(),
    )
}

/// [`epilogue_tma_store`] with its keyword arguments
/// ([`EpilogueTmaStoreOptions`]). With `amax_out`, each thread keeps the
/// running maximum of its accumulator values' absolute values over the
/// subtiles, each warp reduces it (`redux.sync` max, NaN-propagating) and
/// lane 0 takes it into `amax_out` atomically; the accumulator must be
/// Float32.
#[track_caller]
pub fn epilogue_tma_store_with<Acc: Numeric, D: Numeric>(
    cta_tile_shape_mnk: (usize, usize, usize),
    use_2cta_instrs: bool,
    tmem_acc_buffer_staged: &Tensor<Acc>,
    gmem_d: &Tensor<D>,
    cta_d_tile_coord: &[Int32],
    tma_store_pipeline: TMAStorePipeline,
    tma_store_warp_id: i32,
    epilogue_op: impl Fn(TensorSsa<D>) -> TensorSsa<D>,
    options: &EpilogueTmaStoreOptions,
) -> TMAStorePipeline {
    let loc = std::panic::Location::caller();
    let _at = builder::LocationScope::new();
    let (pipe, _) = epilogue_core::<Acc, D, super::GenericPipelineBase>(
        loc,
        cta_tile_shape_mnk,
        use_2cta_instrs,
        tmem_acc_buffer_staged,
        gmem_d,
        cta_d_tile_coord,
        tma_store_pipeline,
        tma_store_warp_id,
        &epilogue_op,
        options,
        None,
    );
    pipe
}

/// [`epilogue_tma_store_with`] draining an accumulator view of an
/// [`AccumulatorBufferingPlan`](super::AccumulatorBufferingPlan) (CuTe
/// DSL's `accumulator_drain=`): the subtiles are visited in the drain's
/// order ([`AccumulatorDrain::subtile_index`]: view 0 of an overlapping plan
/// in reverse), and the view is released after the plan's subtile
/// ([`AccumulatorDrain::after_t2r`]), so this function owns the
/// accumulator's one release. Returns the store pipeline and the
/// accumulator pipeline, advanced.
#[track_caller]
pub fn epilogue_tma_store_drain<Acc: Numeric, D: Numeric, P: AsGenericPipeline>(
    cta_tile_shape_mnk: (usize, usize, usize),
    use_2cta_instrs: bool,
    tmem_acc_buffer_staged: &Tensor<Acc>,
    gmem_d: &Tensor<D>,
    cta_d_tile_coord: &[Int32],
    tma_store_pipeline: TMAStorePipeline,
    tma_store_warp_id: i32,
    epilogue_op: impl Fn(TensorSsa<D>) -> TensorSsa<D>,
    accumulator_drain: AccumulatorDrain<P>,
    options: &EpilogueTmaStoreOptions,
) -> (TMAStorePipeline, P) {
    let loc = std::panic::Location::caller();
    let _at = builder::LocationScope::new();
    let (pipe, acc) = epilogue_core(
        loc,
        cta_tile_shape_mnk,
        use_2cta_instrs,
        tmem_acc_buffer_staged,
        gmem_d,
        cta_d_tile_coord,
        tma_store_pipeline,
        tma_store_warp_id,
        &epilogue_op,
        options,
        Some(accumulator_drain),
    );
    (pipe, acc.expect("the drain's pipeline"))
}

/// The accumulator stage as its `(M, N)` tile: a rank-2 tensor, or the
/// `((M, N, ...), 1, 1, ...)` form of [`make_tmem_layout_acc`]'s stage
/// (`acc[.., .., .., stage]`), whose first mode's `(M, N)` is taken.
fn acc_tile<T>(acc: &Tensor<T>, want: (usize, usize)) -> Result<Tensor<T>, String> {
    let l = acc.layout();
    let fixed = |n: usize| (0..n).map(|_| Tree::Leaf(Some(IndexExpr::from(0))));
    let tile = if l.rank() == 2 && l.mode(0).rank() <= 1 && l.mode(1).rank() <= 1 {
        acc.clone()
    } else if l.rank() >= 1
        && l.mode(0).rank() >= 2
        && (1..l.rank()).all(|i| l.mode(i).size() == 1)
        && (2..l.mode(0).rank()).all(|i| l.mode(0).mode(i).size() == 1)
    {
        let mut first = vec![Tree::Leaf(None), Tree::Leaf(None)];
        first.extend(fixed(l.mode(0).rank() - 2));
        let mut coords = vec![Tree::Branch(first)];
        coords.extend(fixed(l.rank() - 1));
        acc.slice(Tree::Branch(coords))
    } else {
        return Err(format!("the accumulator {l} is not one stage's (M, N) tile"));
    };
    let tl = tile.layout();
    if tl.rank() != 2 || (tl.mode(0).size(), tl.mode(1).size()) != want {
        return Err(format!("the accumulator {l} is not the {} x {} CTA tile", want.0, want.1));
    }
    Ok(tile)
}

#[track_caller]
fn epilogue_core<Acc: Numeric, D: Numeric, P: AsGenericPipeline>(
    loc: &'static std::panic::Location<'static>,
    cta_tile_shape_mnk: (usize, usize, usize),
    use_2cta_instrs: bool,
    tmem_acc: &Tensor<Acc>,
    gmem_d: &Tensor<D>,
    cta_d_tile_coord: &[Int32],
    mut pipe: TMAStorePipeline,
    tma_store_warp_id: i32,
    epilogue_op: &dyn Fn(TensorSsa<D>) -> TensorSsa<D>,
    options: &EpilogueTmaStoreOptions,
    mut drain: Option<AccumulatorDrain<P>>,
) -> (TMAStorePipeline, Option<P>) {
    let fail = |m: String| -> ! { builder::user_panic(loc, format!("experimental::epilogue_tma_store: {m}")) };
    // The store pipeline must synchronise exactly the epilogue warpgroup.
    match pipe.barrier().num_threads().as_static() {
        Some(n) if n == THREADS_PER_WARPGROUP as i64 => {}
        Some(n) => fail(format!(
            "the TMAStorePipeline's barrier counts {n} threads, but the epilogue's warpgroup ({THREADS_PER_WARPGROUP} \
             threads, every one storing to shared memory) meets on it: give the pipeline arv_count \
             {THREADS_PER_WARPGROUP}"
        )),
        None => {}
    }
    if pipe.tma_warp_id() != tma_store_warp_id {
        fail(format!(
            "the TMAStorePipeline's TMA warp is {}, but tma_store_warp_id is {tma_store_warp_id}",
            pipe.tma_warp_id()
        ));
    }
    if options.amax_out.is_some() && Acc::DTYPE != DType::F32 {
        fail(format!("amax_out reduces a Float32 accumulator, not {}", Acc::DTYPE));
    }
    let (cta_m, cta_n, cta_k) = cta_tile_shape_mnk;
    let tid: Int32 = options.tid_x_in_group.unwrap_or_else(|| thread_idx().x % THREADS_PER_WARPGROUP);
    let warp = make_warp_uniform(warp_idx());
    let d_major = options.d_major_mode.unwrap_or_else(|| LayoutEnum::from_tensor(gmem_d));
    let shape = [cta_m, cta_n, cta_k];
    let epi_tile =
        crate::utils::compute_epilogue_tile_shape(&shape, use_2cta_instrs, d_major, D::DTYPE, None, None, None);
    let atom_t2r = get_tmem_load_op::<Acc>(&shape, d_major, D::DTYPE, &epi_tile, use_2cta_instrs, None);

    // (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
    let acc = acc_tile(tmem_acc, (cta_m, cta_n)).unwrap_or_else(|m| fail(m));
    let acc_epi = flat_divide(&acc, epi_tile.clone());
    let tiled_copy_t2r = make_tmem_copy(&atom_t2r, &slice_after(&acc_epi, 2, &[0.into(), 0.into()]));
    let bits = D::DTYPE.storage_bits().max(8);
    let tiled_copy_r2s = make_tiled_copy_d(&make_copy_atom::<D>(CopyOp::Universal { bits }), &tiled_copy_t2r);

    // D's CTA tile, divided into subtiles: (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N).
    let d_tile = match (gmem_d.rank(), cta_d_tile_coord.len()) {
        (2, 2 | 3) => local_tile(gmem_d, (cta_m, cta_n), (cta_d_tile_coord[0], cta_d_tile_coord[1])),
        (3, 3) => local_tile(
            &gmem_d.slice((.., .., cta_d_tile_coord[2])),
            (cta_m, cta_n),
            (cta_d_tile_coord[0], cta_d_tile_coord[1]),
        ),
        (r, c) => {
            fail(format!("D has rank {r} and the tile coordinate {c} modes: give (M, N) or (M, N, L) and (m, n[, l])"))
        }
    };
    let d_epi = flat_divide(&d_tile, epi_tile.clone());
    let epi_m = d_epi.layout().mode(2).size();
    if epi_m != 1 {
        fail(format!(
            "the epilogue tile {epi_tile:?} splits the CTA tile's M ({cta_m}) into {epi_m}: it takes whole columns"
        ));
    }
    let subtiles = d_epi.layout().mode(3).size();

    let rmem_layout = make_t2r_rmem_layout(&tiled_copy_t2r, &d_epi, tid);
    let rmem_acc = allocate::<Acc>(AddressSpace::Rmem, rmem_layout.clone(), 32);
    let rmem_d = allocate::<D>(AddressSpace::Rmem, rmem_layout, 32);
    let smem_layout = make_smem_layout_epi(D::DTYPE, d_major, &epi_tile, pipe.get_num_stages());
    let smem_d = allocate::<D>(AddressSpace::Smem, smem_layout, 1024);

    let mut thread_amax = options.amax_out.as_ref().map(|_| Float32::from(0.0f32));
    let thr_t2r = tiled_copy_t2r.get_slice(tid);
    let thr_r2s = tiled_copy_r2s.get_slice(tid);
    for i in 0..subtiles {
        let real: IndexExpr = match &drain {
            Some(d) => d.subtile_index(i, subtiles).into(),
            None => IndexExpr::from(i),
        };
        // TMEM -> registers.
        partition_and_copy(&thr_t2r, &slice_after(&acc_epi, 2, &[0.into(), real.clone()]), &rmem_acc);
        fence_view_async_tmem_load();
        if let Some(d) = drain.as_mut() {
            d.after_t2r(i);
        }
        // The epilogue operation, in registers.
        let acc_vec = rmem_acc.load_ssa();
        if let Some(m) = thread_amax.as_mut() {
            let sub = acc_vec.to::<f32>().abs().reduce_all(ReductionOp::Max, 0.0f32);
            *m = fmax(*m, sub);
        }
        rmem_d.store_ssa(&epilogue_op(acc_vec.to::<D>()));
        // Registers -> shared memory, then the TMA store.
        pipe.acquire_sync();
        let stage = smem_d.slice((.., .., pipe.get_index()));
        partition_and_copy(&thr_r2s, &rmem_d, &stage);
        pipe.commit_sync();
        let dst = slice_after(&d_epi, 2, &[0.into(), real]);
        if_(warp.eq(tma_store_warp_id), || tma_store(&stage, &dst));
        pipe.release_advance();
    }
    if let (Some(m), Some(out)) = (thread_amax, options.amax_out.as_ref()) {
        let warp_amax =
            warp_redux_sync_with(m, ReduxKind::Fmax, u32::MAX, ReduxOptions { nan: true, ..Default::default() });
        if_(lane_idx().eq(0), || {
            atomic_fmax_with(&out.ptr(), warp_amax, Some(false), Default::default());
        });
    }
    pipe.tail();
    (pipe, drain.map(AccumulatorDrain::into_pipeline))
}
