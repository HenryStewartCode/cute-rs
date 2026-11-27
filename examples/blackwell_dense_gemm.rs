// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py
//   (`DenseGemmKernel`: its configuration, `_setup_attributes`, `__call__`, `kernel`,
//   `_compute_stages`, `_compute_grid` and `_compute_num_tmem_alloc_cols`, with their
//   comments and names, and the data of its tensors),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The canonical Blackwell GEMM, ported from CuTe DSL's `dense_gemm.py`: `C = A B^T` with TMA loads, tcgen05 MMAs and a
//! TMA-store epilogue, on one CTA or a CTA pair, in a cluster of any shape.
//!
//! - A `(M, K)` and B `(N, K)` are K-major Float16, the accumulator Float32 in TMEM, C
//!   `(M, N)` Float16 or Float32 in either major.
//! - [`make_trivial_tiled_mma`] builds the MMA from the tile `mma_tiler_mn` (the K of the
//!   tile is four instructions' worth, 64 for Float16); with `use_2cta_instrs` a CTA pair
//!   runs it (`CtaGroup::TWO`) and each CTA owns half of the tile's rows.
//! - [`make_smem_layout_a`], `_b` and [`make_smem_layout_epi`] give the staged shared
//!   memory layouts; the stage counts come from [`DenseGemm::compute_stages`] (the
//!   example's heuristic, filling the shared memory).
//! - [`cluster_shape_to_tma_atom_a`] and `_b` pick the TMA loads of the cluster shape: A is
//!   multicast along N and B along M, with the masks of [`create_tma_multicast_mask`].
//! - [`get_num_tmem_alloc_cols`] sizes the TMEM allocation, made by a [`TmemAllocator`].
//! - `PipelineTmaUmma` carries A and B from TMA to the MMA, `PipelineUmmaAsync` the
//!   accumulator from the MMA to the epilogue, and `PipelineTmaStore` paces the TMA stores of
//!   C.
//!
//! The roles of the warps are separate functions:
//! [`tma_load_warp`] issues the TMA loads of one k-tile, [`mma_warp`] runs the MMAs
//! of one k-tile (on the leader CTA of a pair) and [`epilogue_warps`] drains the accumulator
//! (all four warps). As in the example, warp 0 plays both the TMA and the MMA role, with the
//! loads running `ab_stages - 2` k-tiles ahead of the MMAs ([`for_range_pipelined`] is
//! `cutlass.range(.., prefetch_stages=..)`; the example's hand-written prefetch loop and
//! `try_wait` peeks are what it expands to).
//!
//! Differences from the Python:
//! - the example defines its own `epilogue_tma_store`; the port builds the epilogue from the
//!   packaged one ([`cute::utils::gemm::sm100::epilogue_tma_store`]). The packaged one waits for each TMEM load explicitly (the
//!   simulator requires it) and releases the accumulator from one thread per warp, so the
//!   accumulator pipeline's consumer group counts the four epilogue warps, and its
//!   named barrier is 2 (the example's own is 1; 1 is the TMEM free's here).
//! - the SMEM tiles are allocated with their swizzled layouts at 1024-byte alignment.
//! - the example's `epilogue_op` is a closure argument here too ([`DenseGemm::build_with`]).
//! - the TMA-store path only: the example's non-TMA `epilogue` (`use_tma_store=False`), the
//!   batch mode `L`, the Float8 and Int8 inputs, TF32 and M-major operands, host-side argument
//!   parsing, torch interop and benchmarking are not ported.
//! - `M`, `N` and `K` are launch values and need not be tile multiples: TMA loads fill
//!   what lies outside A and B with zeros and TMA stores skip what lies outside C. `K` and (for
//!   a row-major C) `N` must keep the rows 16-byte aligned (`K` and `N` multiples of 8).
//!
//! No Blackwell GPU is available to cute-rs's tests, so the kernel is verified by the simulator
//! (bit for bit: the data are integers in `[-2, 2)`) and by `ptxas -arch=sm_100a`.
//!
//! Run with `cargo run --release --example blackwell_dense_gemm`.

use cute::cpasync;
use cute::memory::{
    SmemAllocator, SmemPartition, TmemAllocator, TmemAllocatorOptions, get_num_tmem_alloc_cols,
    get_smem_capacity_in_bytes,
};
use cute::nvgpu::{make_tiled_tma_atom_a, make_tiled_tma_atom_b};
use cute::pipeline::{
    self, Agent, CooperativeGroup, NamedBarrier, Pipeline, PipelineConsumer, PipelineProducer, PipelineState,
    PipelineTmaStore, PipelineTmaUmma, PipelineTmaUmmaOptions, PipelineUmmaAsync, PipelineUmmaOptions,
};
use cute::prelude::*;
use cute::tcgen05::{CtaGroup, Field, OperandMajorMode, OperandSource};
use cute::utils::gemm::sm100::{EpilogueConfig, epilogue_tma_store};
use cute::utils::sm100::{
    cluster_shape_to_tma_atom_a, cluster_shape_to_tma_atom_b, make_smem_layout_a, make_smem_layout_b,
    make_smem_layout_epi, make_trivial_tiled_mma,
};
use cute::utils::{LayoutEnum, compute_epilogue_tile_shape};

/// Float16 operands, Float32 accumulation.
pub type AbDtype = F16;
pub const ACC_DTYPE: DType = DType::F32;

/// A C element type: Float16 or Float32.
pub trait CType: Numeric + Copy + PartialEq + std::fmt::Debug + 'static {
    fn from_f(x: f32) -> Self;
    fn to_f(self) -> f32;
}

impl CType for F16 {
    fn from_f(x: f32) -> F16 {
        F16::from_f32(x)
    }
    fn to_f(self) -> f32 {
        self.to_f32()
    }
}

impl CType for f32 {
    fn from_f(x: f32) -> f32 {
        x
    }
    fn to_f(self) -> f32 {
        self
    }
}

/// `DenseGemmKernel`'s constructor arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseGemm {
    pub name: &'static str,
    /// `CtaGroup::TWO` MMAs: a CTA pair per MMA tile.
    pub use_2cta_instrs: bool,
    /// The `(M, N)` of the MMA tile (of the CTA pair with `use_2cta_instrs`).
    pub mma_tiler_mn: (usize, usize),
    /// The cluster `(M, N)`; its M extent is a multiple of 2 with `use_2cta_instrs`.
    pub cluster_shape_mn: (usize, usize),
    /// C's majorness in global memory.
    pub c_layout: LayoutEnum,
    pub threads_per_cta: usize,
}

/// One CTA, a `(128, 128)` tile, no cluster.
pub const DENSE_GEMM_1CTA: DenseGemm = DenseGemm {
    name: "blackwell_dense_gemm_1cta",
    use_2cta_instrs: false,
    mma_tiler_mn: (128, 128),
    cluster_shape_mn: (1, 1),
    c_layout: LayoutEnum::ROW_MAJOR,
    threads_per_cta: 128,
};

/// A CTA pair on a `(256, 128)` tile, in a `(2, 1)` cluster.
pub const DENSE_GEMM_2CTA: DenseGemm = DenseGemm {
    name: "blackwell_dense_gemm_2cta",
    use_2cta_instrs: true,
    mma_tiler_mn: (256, 128),
    cluster_shape_mn: (2, 1),
    ..DENSE_GEMM_1CTA
};

/// A CTA pair in a `(2, 2)` cluster: A is multicast along N and B along M.
pub const DENSE_GEMM_2CTA_MULTICAST: DenseGemm =
    DenseGemm { name: "blackwell_dense_gemm_2cta_multicast", cluster_shape_mn: (2, 2), ..DENSE_GEMM_2CTA };

cute_struct! {
    pub struct SharedStorage(num_ab_stage: usize, num_acc_stage: usize) {
        ab_full_mbar_ptr: MemRange<i64>[num_ab_stage * 2],
        acc_full_mbar_ptr: MemRange<i64>[num_acc_stage * 2],
        tmem_dealloc_mbar: i64,
        tmem_holding_buf: i32,
    }
}

/// What `_setup_attributes` computes from the types and the layout of C.
pub struct Setup {
    pub tiled_mma: TiledMma,
    /// `(M, N, K)` of the MMA tile.
    pub mma_tiler: (usize, usize, usize),
    pub cta_tile_shape_mnk: (usize, usize, usize),
    /// `(V, M, N, K)` of the cluster.
    pub cluster_layout_vmnk: Layout,
    pub epi_tile: Tiler,
    pub num_acc_stage: usize,
    pub num_ab_stage: usize,
    pub num_c_stage: usize,
    pub a_smem_layout_staged: ComposedLayout,
    pub b_smem_layout_staged: ComposedLayout,
    pub c_smem_layout_staged: ComposedLayout,
    pub num_tmem_alloc_cols: usize,
}

impl DenseGemm {
    /// The cta group of the MMA.
    pub fn cta_group(&self) -> CtaGroup {
        if self.use_2cta_instrs { CtaGroup::TWO } else { CtaGroup::ONE }
    }

    /// `_setup_attributes`: the tiled MMA, the tile and cluster shapes, the epilogue
    /// tile, the stage counts, the shared-memory layouts and the TMEM columns.
    pub fn setup(&self, c_dtype: DType) -> Setup {
        let tiled_mma = make_trivial_tiled_mma(
            AbDtype::DTYPE,
            AbDtype::DTYPE,
            OperandMajorMode::K,
            OperandMajorMode::K,
            ACC_DTYPE,
            self.cta_group(),
            self.mma_tiler_mn,
            OperandSource::SMEM,
        );

        // Compute mma/cluster/tile shapes
        let mma_inst_shape_k = tiled_mma.atom().shape_mnk()[2];
        let mma_inst_tile_k = 4;
        let mma_tiler = (self.mma_tiler_mn.0, self.mma_tiler_mn.1, mma_inst_shape_k * mma_inst_tile_k);
        let cta_tile_shape_mnk = (mma_tiler.0 / tiled_mma.thr_id().size(), mma_tiler.1, mma_tiler.2);

        // Compute cluster layout
        let (cm, cn) = self.cluster_shape_mn;
        let cluster_layout_vmnk = Layout::col_major((cm, cn, 1))
            .tiled_divide(&Tree::Branch(vec![Tree::Leaf(Some(tiled_mma.thr_id().clone()))]));

        // Compute epilogue subtile
        let epi_tile = compute_epilogue_tile_shape(
            &[cta_tile_shape_mnk.0, cta_tile_shape_mnk.1],
            self.use_2cta_instrs,
            self.c_layout,
            c_dtype,
            None,
            None,
            None,
        );

        // Setup A/B/C stage count in shared memory
        let smem_capacity = get_smem_capacity_in_bytes(Some(Arch::Sm100a));
        let (num_acc_stage, num_ab_stage, num_c_stage) =
            Self::compute_stages(&tiled_mma, mma_tiler, &epi_tile, c_dtype, self.c_layout, smem_capacity, 1);

        // Compute A/B/C shared memory layout
        let a_smem_layout_staged = make_smem_layout_a(&tiled_mma, mma_tiler, AbDtype::DTYPE, num_ab_stage, None);
        let b_smem_layout_staged = make_smem_layout_b(&tiled_mma, mma_tiler, AbDtype::DTYPE, num_ab_stage, None);
        let c_smem_layout_staged = make_smem_layout_epi(c_dtype, self.c_layout, &epi_tile, num_c_stage);

        // Compute the number of tensor memory allocation columns
        let acc_shape = tiled_mma.partition_shape_c((mma_tiler.0, mma_tiler.1));
        let t_ct_acc_fake = tiled_mma.make_fragment_c::<f32>(&acc_shape);
        let num_tmem_alloc_cols = get_num_tmem_alloc_cols(&[&t_ct_acc_fake], true, Arch::Sm100a);

        Setup {
            tiled_mma,
            mma_tiler,
            cta_tile_shape_mnk,
            cluster_layout_vmnk,
            epi_tile,
            num_acc_stage,
            num_ab_stage,
            num_c_stage,
            a_smem_layout_staged,
            b_smem_layout_staged,
            c_smem_layout_staged,
            num_tmem_alloc_cols,
        }
    }

    /// `_compute_stages`: the accumulator and C stages by default, then as many A/B stages
    /// as fit the shared memory beside the C stages and the barriers' bytes, and the
    /// shared memory left over goes to C.
    pub fn compute_stages(
        tiled_mma: &TiledMma,
        mma_tiler_mnk: (usize, usize, usize),
        epi_tile: &Tiler,
        c_dtype: DType,
        c_layout: LayoutEnum,
        smem_capacity: usize,
        occupancy: usize,
    ) -> (usize, usize, usize) {
        // Default ACC stages and C stages
        let num_acc_stage = 1;
        let num_c_stage = 2;

        // Calculate smem layout and size for one stage of A, B, and C
        let tiler = mma_tiler_mnk;
        let a_one = make_smem_layout_a(tiled_mma, tiler, AbDtype::DTYPE, 1, None);
        let b_one = make_smem_layout_b(tiled_mma, tiler, AbDtype::DTYPE, 1, None);
        let c_one = make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1);
        let bytes =
            |t: DType, l: &ComposedLayout| size_in_bytes_dtype(t, l).as_static().expect("the SMEM layouts are static");
        let ab_bytes_per_stage = bytes(AbDtype::DTYPE, &a_one) + bytes(AbDtype::DTYPE, &b_one);
        let mbar_helpers_bytes = 1024;
        let c_bytes_per_stage = bytes(c_dtype, &c_one);
        let c_bytes = c_bytes_per_stage * num_c_stage;

        // Calculate A/B stages
        let num_ab_stage = (smem_capacity - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)) / ab_bytes_per_stage;

        // Add remaining unused smem to epilogue
        let num_c_stage = num_c_stage
            + (smem_capacity - ab_bytes_per_stage * num_ab_stage - (occupancy + 1) * (mbar_helpers_bytes + c_bytes))
                / ((occupancy + 1) * c_bytes_per_stage);
        (num_acc_stage, num_ab_stage, num_c_stage)
    }

    /// `_compute_grid`: one CTA per `cta_tile` tile of C, rounded up to whole clusters.
    pub fn launch(&self, m: usize, n: usize) -> LaunchConfig {
        let tm = self.mma_tiler_mn.0 / if self.use_2cta_instrs { 2 } else { 1 };
        let tn = self.mma_tiler_mn.1;
        let (cx, cy) = self.cluster_shape_mn;
        let round = |x: usize, c: usize| x.div_ceil(c) * c;
        let grid = [round(m.div_ceil(tm), cx), round(n.div_ceil(tn), cy), 1];
        let launch = LaunchConfig::new(grid.map(|g| g as u32), [self.threads_per_cta as u32, 1, 1]);
        if cx * cy > 1 { launch.with_cluster([cx as u32, cy as u32, 1]) } else { launch }
    }

    /// The kernel with the identity `epilogue_op`: parameters `m`, `n`, `k` (launch
    /// values), then `a`, `b` and `c`.
    pub fn build<C: CType>(&self) -> Kernel {
        self.build_with::<C>(|v| v)
    }

    /// [`build`](Self::build) with an elementwise `epilogue_op` applied to the C-typed
    /// values before they are stored.
    pub fn build_with<C: CType>(&self, epilogue_op: impl Fn(TensorSsa<C>) -> TensorSsa<C> + 'static) -> Kernel {
        let cfg = *self;
        Kernel::builder(cfg.name).block_dim([cfg.threads_per_cta as u32, 1, 1]).target(Arch::Sm100a).build(move || {
            let st = cfg.setup(C::DTYPE);
            let mma_tiler = st.mma_tiler;
            let m_major_c = cfg.c_layout.is_m_major_c();
            let m = param_extent("m", if m_major_c { 8 } else { 1 });
            let n = param_extent("n", 8);
            let k = param_extent("k", 8);
            let a = param_tensor_aligned::<AbDtype>("a", Layout::new((m, k), (k, 1)), 32);
            let b = param_tensor_aligned::<AbDtype>("b", Layout::new((n, k), (k, 1)), 32);
            let c = param_tensor_aligned::<C>(
                "c",
                if m_major_c { Layout::new((m, n), (1, m)) } else { Layout::new((m, n), (n, 1)) },
                32,
            );
            assert_eq!(LayoutEnum::from_tensor(&c), cfg.c_layout, "C is declared in the configuration's layout");

            let tiled_mma = st.tiled_mma.clone();
            let atom_thr_size = tiled_mma.thr_id().size();

            // Setup TMA load for A
            let a_op =
                cluster_shape_to_tma_atom_a((cfg.cluster_shape_mn.0, cfg.cluster_shape_mn.1, 1), tiled_mma.thr_id())
                    .unwrap_or_else(|e| panic!("{e}"));
            let a_smem_layout = st.a_smem_layout_staged.select(&[0, 1, 2]);
            let cluster =
                (cfg.cluster_shape_mn.0 * cfg.cluster_shape_mn.1 > 1).then(|| st.cluster_layout_vmnk.shape().clone());
            let (tma_atom_a, tma_tensor_a) =
                make_tiled_tma_atom_a(a_op, &a, a_smem_layout.clone(), mma_tiler, &tiled_mma, cluster.clone()).into();

            // Setup TMA load for B
            let b_op =
                cluster_shape_to_tma_atom_b((cfg.cluster_shape_mn.0, cfg.cluster_shape_mn.1, 1), tiled_mma.thr_id())
                    .unwrap_or_else(|e| panic!("{e}"));
            let b_smem_layout = st.b_smem_layout_staged.select(&[0, 1, 2]);
            let (tma_atom_b, tma_tensor_b) =
                make_tiled_tma_atom_b(b_op, &b, b_smem_layout.clone(), mma_tiler, &tiled_mma, cluster).into();

            let a_copy_size = size_in_bytes::<AbDtype>(&a_smem_layout).as_static().expect("static SMEM layouts");
            let b_copy_size = size_in_bytes::<AbDtype>(&b_smem_layout).as_static().expect("static SMEM layouts");
            let num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size;

            // Setup store for C
            let (tma_atom_c, tma_tensor_c) = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileS2GOp,
                &c,
                st.c_smem_layout_staged.slice((.., .., 0)),
                st.epi_tile.clone(),
                1,
            )
            .into();

            let mut ecfg =
                EpilogueConfig::new(st.cta_tile_shape_mnk, cfg.c_layout, C::DTYPE, ACC_DTYPE, cfg.use_2cta_instrs);
            ecfg.num_c_stage = st.num_c_stage;

            kernel::<C>(
                &cfg,
                &st,
                tiled_mma,
                &tma_atom_a,
                &tma_tensor_a,
                &tma_atom_b,
                &tma_tensor_b,
                &tma_atom_c,
                &tma_tensor_c,
                num_tma_load_bytes,
                &ecfg,
                &epilogue_op,
            );
        })
    }
}

/// The TMA loads and the MMAs of the mainloop: what the roles of warp 0 share.
pub struct Mainloop<'a> {
    pub tma_atom_a: &'a CopyAtom<AbDtype>,
    pub tma_atom_b: &'a CopyAtom<AbDtype>,
    /// `((atom_v, rest_v), RestK)`: this CTA's tiles of A and B in global memory.
    pub t_ag_a: &'a Tensor<Ident>,
    pub t_bg_b: &'a Tensor<Ident>,
    /// `((atom_v, rest_v), STAGE)`.
    pub t_as_a: &'a Tensor<AbDtype>,
    pub t_bs_b: &'a Tensor<AbDtype>,
    pub a_full_mcast_mask: Option<Int16>,
    pub b_full_mcast_mask: Option<Int16>,
    /// `(MMA, MMA_M, MMA_K, STAGE)` and `(MMA, MMA_N, MMA_K, STAGE)`.
    pub t_cr_a: &'a Tensor<AbDtype>,
    pub t_cr_b: &'a Tensor<AbDtype>,
    /// `(MMA, MMA_M, MMA_N)` in TMEM.
    pub t_ct_acc: &'a Tensor<f32>,
    /// Whether this CTA issues the MMAs (the even CTA of a pair, every CTA otherwise).
    pub is_leader_cta: Int32,
    pub use_2cta_instrs: bool,
}

type AbProducer<'p> = PipelineProducer<'p, PipelineTmaUmma>;
type AbConsumer<'p> = PipelineConsumer<'p, PipelineTmaUmma>;

/// The TMA role: wait for an empty A/B stage and load one k-tile of A and B into it
/// (`cute.copy` with the stage's full barrier and the multicast masks).
pub fn tma_load_warp<'p>(ml: &Mainloop, mut ab_producer: AbProducer<'p>) -> AbProducer<'p> {
    // Conditionally wait for AB buffer empty
    let producer_handle = ab_producer.acquire_and_advance(None);
    let bar = CopyArgs::new().tma_bar_ptr(&producer_handle.barrier());
    let (count, index) = (producer_handle.count(), producer_handle.index());
    // TMA load A/B
    copy_with(
        ml.tma_atom_a,
        &ml.t_ag_a.slice((.., count)),
        &ml.t_as_a.slice((.., index)),
        match ml.a_full_mcast_mask {
            Some(mask) => bar.clone().mcast_mask(mask),
            None => bar.clone(),
        },
    );
    copy_with(
        ml.tma_atom_b,
        &ml.t_bg_b.slice((.., count)),
        &ml.t_bs_b.slice((.., index)),
        match ml.b_full_mcast_mask {
            Some(mask) => bar.mcast_mask(mask),
            None => bar,
        },
    );
    ab_producer
}

/// The MMA role: wait for a full A/B stage and issue its MMAs, `tCtAcc += tCrA * tCrB`,
/// on the leader CTA of a pair; then release the stage (an asynchronous arrive on the
/// empty barrier when the MMAs complete). The accumulate flag is on after the first
/// k-block.
pub fn mma_warp<'p>(ml: &Mainloop, mut ab_consumer: AbConsumer<'p>, tiled_mma: TiledMma) -> (AbConsumer<'p>, TiledMma) {
    let consumer_handle = ab_consumer.current_handle();
    ab_consumer.advance(1);
    let issue = |tiled_mma: &TiledMma| {
        let mut tiled_mma = tiled_mma.clone();
        // Conditionally wait for AB buffer full
        consumer_handle.get_origin().consumer_wait(&consumer_handle.state(), None);
        let num_kblks = ml.t_cr_a.layout().mode(2).size();
        for kblk_idx in 0..num_kblks {
            let kblk_crd = (.., .., kblk_idx, consumer_handle.index());
            gemm(&tiled_mma, ml.t_ct_acc, &ml.t_cr_a.slice(kblk_crd), &ml.t_cr_b.slice(kblk_crd), ml.t_ct_acc);
            // Enable accumulate on tCtAcc after first kblock
            tiled_mma.set(Field::ACCUMULATE, true);
        }
        // Async arrive AB buffer empty
        consumer_handle.release();
    };
    if ml.use_2cta_instrs {
        if_(ml.is_leader_cta.eq(0), || issue(&tiled_mma))
    } else {
        issue(&tiled_mma)
    }
    let mut tiled_mma = tiled_mma;
    tiled_mma.set(Field::ACCUMULATE, true);
    (ab_consumer, tiled_mma)
}

/// The epilogue role (all four warps): TMEM -> registers -> shared memory -> global memory
/// through [`epilogue_tma_store`], waiting for the accumulator and releasing it.
pub fn epilogue_warps<C: CType>(
    ecfg: &EpilogueConfig,
    tidx: Int32,
    warp_idx: Int32,
    mma_tile_coord_mnl: &[Int32],
    thr_mma: &ThrMma,
    tma_atom_c: &CopyAtom<C>,
    t_ct_acc: &Tensor<f32>,
    acc_shape: &Shape,
    tma_tensor_c: &Tensor<Ident>,
    s_c: &Tensor<C>,
    mma_tiler: (usize, usize, usize),
    tiled_mma: &TiledMma,
    epi_tile: &Tiler,
    epilogue_op: &dyn Fn(TensorSsa<C>) -> TensorSsa<C>,
    acc_pipeline: &PipelineUmmaAsync,
    acc_consumer_state: &PipelineState,
    num_acc_stage: usize,
) {
    // (bM, bN, RestM, RestN)
    let g_c_mnl = local_tile(tma_tensor_c, (mma_tiler.0, mma_tiler.1), (.., ..));
    // (MMA, MMA_M, MMA_N, RestM, RestN)
    let t_cg_c = thr_mma.partition_c(&g_c_mnl);
    // The accumulator with a stage mode: ((M, N), MMA_M, MMA_N, STAGE)
    let Tree::Branch(mut modes) = acc_shape.clone() else { panic!("a tuple shape") };
    modes.push(Shape::from(num_acc_stage));
    let staged = tiled_mma.make_fragment_c::<f32>(&Shape::from(Tree::Branch(modes)));
    let t_ct_acc = make_tensor(t_ct_acc.ptr().clone(), staged.layout().clone());

    // Initialize tma store c_pipeline
    let c_pipeline = PipelineTmaStore::create(
        ecfg.num_c_stage,
        CooperativeGroup::new(Agent::Thread, 32 * ecfg.epilogue_warp_id.len() as i32),
    );
    epilogue_tma_store::<f32, C, _, _>(
        ecfg,
        tidx,
        warp_idx,
        tma_atom_c,
        &t_ct_acc,
        s_c,
        &t_cg_c,
        epi_tile,
        Int32::from(0),
        epilogue_op,
        mma_tile_coord_mnl,
        acc_consumer_state,
        acc_pipeline,
        &c_pipeline,
    );
    // Wait for C store complete
    if_(warp_idx.eq(ecfg.epilogue_warp_id[0] as i32), || c_pipeline.producer_tail());
}

/// The kernel of `DenseGemmKernel`.
pub fn kernel<C: CType>(
    cfg: &DenseGemm,
    st: &Setup,
    mut tiled_mma: TiledMma,
    tma_atom_a: &CopyAtom<AbDtype>,
    m_a_mkl: &Tensor<Ident>,
    tma_atom_b: &CopyAtom<AbDtype>,
    m_b_nkl: &Tensor<Ident>,
    tma_atom_c: &CopyAtom<C>,
    m_c_mnl: &Tensor<Ident>,
    num_tma_load_bytes: usize,
    ecfg: &EpilogueConfig,
    epilogue_op: &dyn Fn(TensorSsa<C>) -> TensorSsa<C>,
) {
    let mma_tiler = st.mma_tiler;
    let mma_tiler_mnk = mma_tiler;
    let cluster_layout_vmnk = &st.cluster_layout_vmnk;
    let (num_ab_stage, num_acc_stage) = (st.num_ab_stage, st.num_acc_stage);
    let warp_idx = make_warp_uniform(warp_idx());

    // Prefetch tma descriptor
    if_(warp_idx.eq(0), || {
        cpasync::prefetch_descriptor(tma_atom_a);
        cpasync::prefetch_descriptor(tma_atom_b);
        cpasync::prefetch_descriptor(tma_atom_c);
    });

    let use_2cta_instrs = cluster_layout_vmnk.mode(0).size() == 2;

    // Setup cta/thread coordinates
    // Coords inside cluster
    let (bidx, bidy) = (block_idx().x, block_idx().y);
    let v = cluster_layout_vmnk.mode(0).size();
    let mma_tile_coord_v = bidx % v as i32;
    let is_leader_cta = mma_tile_coord_v;
    let cta_rank_in_cluster = make_warp_uniform(block_idx_in_cluster());
    let block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster);
    // Coords outside cluster
    // (no batch mode: C is two-dimensional)
    let mma_tile_coord_mnl = [bidx / v as i32, bidy];
    // Coords inside cta
    let tidx = thread_idx().x;
    let mma_coord_mnk = (mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], ..);

    //
    // Alloc and init: a+b full/empty, accumulator full, tensor memory dealloc barrier
    //
    let smem = SmemAllocator::new();
    let storage_layout = SharedStorage::layout(num_ab_stage, num_acc_stage);
    let storage =
        SharedStorage::from_layout(&storage_layout, &smem.allocate_struct(&storage_layout, 8, SmemPartition::User));

    // Initialize mainloop ab_pipeline (barrier) and states
    let num_mcast_ctas_a = cluster_layout_vmnk.mode(2).size();
    let num_mcast_ctas_b = cluster_layout_vmnk.mode(1).size();
    let num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1;
    let ab_pipeline = PipelineTmaUmma::create(
        num_ab_stage,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, num_tma_producer),
        num_tma_load_bytes,
        PipelineTmaUmmaOptions {
            barrier_storage: Some(storage.ab_full_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cluster_layout_vmnk.clone()),
            defer_sync: true,
            ..Default::default()
        },
    );
    let (ab_producer, ab_consumer) = ab_pipeline.make_participants();

    // Initialize acc_pipeline (barrier) and states: the packaged epilogue's consumers are
    // its warps.
    let epilogue_warps_count = ecfg.epilogue_warp_id.len();
    let acc_pipeline = PipelineUmmaAsync::create(
        num_acc_stage,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, (v * epilogue_warps_count) as i32),
        PipelineUmmaOptions {
            barrier_storage: Some(storage.acc_full_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cluster_layout_vmnk.clone()),
            defer_sync: true,
            ..Default::default()
        },
    );
    let (acc_producer, acc_consumer) = acc_pipeline.make_participants();

    let tmem_alloc_barrier = NamedBarrier::new(0, cfg.threads_per_cta);
    // Tensor memory dealloc barrier init
    let mut tmem = TmemAllocator::new(
        tmem_alloc_barrier,
        TmemAllocatorOptions {
            alloc_result_dst_smem_ptr: Some(storage.tmem_holding_buf.ptr()),
            is_two_cta: use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr: use_2cta_instrs.then(|| storage.tmem_dealloc_mbar.ptr().cast::<u64>()),
            arch: Arch::Sm100a,
            ..Default::default()
        },
    );

    // Cluster arrive after barrier init
    pipeline::pipeline_init_arrive(Some(cluster_layout_vmnk.shape().clone()), true);

    //
    // Setup smem tensor A/B/C
    //
    // (EPI_TILE_M, EPI_TILE_N, STAGE)
    let s_c = smem.allocate_tensor::<C>(st.c_smem_layout_staged.clone(), 1024, None, SmemPartition::User);
    // (MMA, MMA_M, MMA_K, STAGE)
    let s_a = smem.allocate_tensor::<AbDtype>(st.a_smem_layout_staged.clone(), 1024, None, SmemPartition::User);
    // (MMA, MMA_N, MMA_K, STAGE)
    let s_b = smem.allocate_tensor::<AbDtype>(st.b_smem_layout_staged.clone(), 1024, None, SmemPartition::User);

    //
    // Compute multicast mask for A/B buffer full
    //
    let clustered = cluster_layout_vmnk.size() > 1;
    let a_full_mcast_mask = clustered
        .then(|| cpasync::create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk.clone(), 2));
    let b_full_mcast_mask = clustered
        .then(|| cpasync::create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk.clone(), 1));

    //
    // Local_tile partition global tensors
    //
    // (bM, bK, RestK) of this CTA's row of tiles
    let g_a = local_tile_proj(m_a_mkl, mma_tiler_mnk, mma_coord_mnk, &[true, false, true]);
    // (bN, bK, RestK)
    let g_b = local_tile_proj(m_b_nkl, mma_tiler_mnk, mma_coord_mnk, &[false, true, true]);
    let k_tile_cnt = size(&g_a, &[2]);

    //
    // Partition global tensor for TiledMMA_A/B/C
    //
    let thr_mma = tiled_mma.get_slice(mma_tile_coord_v);
    // (MMA, MMA_M, MMA_K, RestK)
    let t_cg_a = thr_mma.partition_a(&g_a);
    // (MMA, MMA_N, MMA_K, RestK)
    let t_cg_b = thr_mma.partition_b(&g_b);

    //
    // Partition global/shared tensor for TMA load A/B
    //
    let coord = |i: usize| match &block_in_cluster_coord_vmnk {
        Tree::Branch(modes) => modes[i].clone(),
        leaf => leaf.clone(),
    };
    // TMA load A partition_S/D: ((atom_v, rest_v), STAGE) and ((atom_v, rest_v), RestK)
    let (t_as_a, t_ag_a) = cpasync::tma_partition(
        tma_atom_a,
        coord(2),
        &make_layout(cluster_layout_vmnk.mode(2).size(), 1),
        &s_a.group_modes(0, 3),
        &t_cg_a.group_modes(0, 3),
    );
    // TMA load B partition_S/D
    let (t_bs_b, t_bg_b) = cpasync::tma_partition(
        tma_atom_b,
        coord(1),
        &make_layout(cluster_layout_vmnk.mode(1).size(), 1),
        &s_b.group_modes(0, 3),
        &t_cg_b.group_modes(0, 3),
    );

    //
    // Partition shared/tensor memory tensor for TiledMMA_A/B/C
    //
    // (MMA, MMA_M, MMA_K, STAGE)
    let t_cr_a = tiled_mma.make_fragment_a(&s_a);
    // (MMA, MMA_N, MMA_K, STAGE)
    let t_cr_b = tiled_mma.make_fragment_b(&s_b);
    // (MMA, MMA_M, MMA_N)
    let acc_shape = tiled_mma.partition_shape_c((mma_tiler.0, mma_tiler.1));
    // (MMA, MMA_M, MMA_N)
    let t_ct_acc_fake = tiled_mma.make_fragment_c::<f32>(&acc_shape);

    //
    // Cluster wait before tensor memory alloc
    //
    pipeline::pipeline_init_wait(Some(cluster_layout_vmnk.shape().clone()));

    // Alloc tensor memory buffer
    tmem.allocate(st.num_tmem_alloc_cols);

    // Barrier before retrieve tensor memory ptr from shared memory
    tmem.wait_for_alloc();

    let tmem_ptr = tmem.retrieve_ptr::<f32>();
    // (MMA, MMA_M, MMA_N)
    let t_ct_acc = make_tensor(tmem_ptr.clone(), t_ct_acc_fake.layout().clone());

    let ml = Mainloop {
        tma_atom_a,
        tma_atom_b,
        t_ag_a: &t_ag_a,
        t_bg_b: &t_bg_b,
        t_as_a: &t_as_a,
        t_bs_b: &t_bs_b,
        a_full_mcast_mask,
        b_full_mcast_mask,
        t_cr_a: &t_cr_a,
        t_cr_b: &t_cr_b,
        t_ct_acc: &t_ct_acc,
        is_leader_cta,
        use_2cta_instrs,
    };

    //
    // Pipelining TMA load A/B and MMA mainloop
    //
    if_(warp_idx.eq(0), || {
        let acc_producer_state = acc_producer.clone_participant();
        // The accumulator starts overwritten: the first MMA does not accumulate.
        tiled_mma.set(Field::ACCUMULATE, false);
        // Prefetch TMA load A/B `ab_stages - 2` k-tiles ahead of the MMAs (at most the k-tile
        // count), then one TMA load and one MMA per k-tile.
        let _ = for_range_pipelined(
            0,
            k_tile_cnt.clone(),
            1,
            Prefetch::stages(num_ab_stage as i64 - 2).bounded_by(num_ab_stage),
            (ab_producer.clone_participant(), (ab_consumer.clone_participant(), tiled_mma.clone())),
            |_, ab_producer| tma_load_warp(&ml, ab_producer),
            |_, (ab_consumer, tiled_mma)| mma_warp(&ml, ab_consumer, tiled_mma),
        );
        // Async arrive accumulator buffer full
        let commit = || acc_producer_state.current_handle().commit();
        if use_2cta_instrs { if_(is_leader_cta.eq(0), commit) } else { commit() }
    });

    //
    // Epilogue
    //

    // Release tensor memory allocation lock
    tmem.relinquish_alloc_permit();

    // The epilogue waits for the accumulator buffer to be full.
    epilogue_warps::<C>(
        ecfg,
        tidx,
        warp_idx,
        &mma_tile_coord_mnl,
        &thr_mma,
        tma_atom_c,
        &t_ct_acc,
        &acc_shape,
        m_c_mnl,
        &s_c,
        mma_tiler,
        &tiled_mma,
        &st.epi_tile,
        epilogue_op,
        &acc_pipeline,
        &acc_consumer.state(),
        num_acc_stage,
    );

    //
    // Dealloc the tensor memory buffer
    //
    pipeline::sync(1);
    tmem.free(&tmem_ptr, 0);

    //
    // Wait A/B buffer empty: the producer's state after the mainloop is `k_tile_cnt`
    // stages on.
    //
    if_(warp_idx.eq(0), || {
        let mut ab_producer = ab_producer.clone_participant();
        ab_producer.advance(IntoValue::<i32>::into_value(k_tile_cnt.clone()));
        ab_producer.tail();
    });
}

/// Integers in `[-2, 2)`, as `torch.empty(..., dtype=torch.int32).random_(-2, 2)` gives
/// them (from a fixed LCG here).
pub fn small_ints(n: usize, seed: u64) -> Vec<AbDtype> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            AbDtype::from_f32(((s >> 40) % 4) as f32 - 2.0)
        })
        .collect()
}

/// `einsum("mk,nk->mn")` in Float32, converted to C's type (exact for these data), in C's
/// layout; `epilogue_op` is applied to the converted values.
pub fn reference<C: CType>(
    a: &[AbDtype],
    b: &[AbDtype],
    (m, n, k): (usize, usize, usize),
    c_layout: LayoutEnum,
    epilogue_op: impl Fn(C) -> C,
) -> Vec<C> {
    let mut out = vec![C::from_f(0.0); m * n];
    for r in 0..m {
        for col in 0..n {
            let acc: f32 = (0..k).map(|i| a[r * k + i].to_f32() * b[col * k + i].to_f32()).sum();
            let at = if c_layout.is_m_major_c() { col * m + r } else { r * n + col };
            out[at] = epilogue_op(C::from_f(acc));
        }
    }
    out
}

/// Simulate `cfg`'s kernel on an `m x n x k` problem with the options `opts` and check C bit
/// for bit.
pub fn run_with<C: CType>(
    cfg: &DenseGemm,
    (m, n, k): (usize, usize, usize),
    opts: &SimOptions,
    seeds: &[u64],
) -> SimReport {
    let kernel = cfg.build::<C>();
    let a = small_ints(m * k, 1111);
    let b = small_ints(n * k, 2222);
    let c = vec![C::from_f(-7.0); m * n];
    let mut args = vec![
        KernelArg::scalar(m as i32),
        KernelArg::scalar(n as i32),
        KernelArg::scalar(k as i32),
        KernelArg::buffer(&a),
        KernelArg::buffer(&b),
        KernelArg::buffer(&c),
    ];
    let report = kernel
        .simulate_schedules(cfg.launch(m, n), &mut args, opts, seeds)
        .unwrap_or_else(|e| panic!("{m}x{n}x{k}: {e}"));
    let got = args[5].to_vec::<C>();
    let want = reference::<C>(&a, &b, (m, n, k), cfg.c_layout, |x| x);
    assert!(got == want, "{m}x{n}x{k}: C differs from the reference");
    report
}

/// [`run_with`] under the default options.
pub fn run<C: CType>(cfg: &DenseGemm, mnk: (usize, usize, usize)) -> SimReport {
    run_with::<C>(cfg, mnk, &SimOptions::default(), &[])
}

fn main() {
    // The three configurations on problems the simulator can run (the example's default,
    // 8192x8192x8192, is too large to simulate).
    for (cfg, (m, n, k)) in [
        (DENSE_GEMM_1CTA, (200, 136, 200)),
        (DENSE_GEMM_2CTA, (264, 136, 200)),
        (DENSE_GEMM_2CTA_MULTICAST, (264, 264, 136)),
    ] {
        println!("===================================================================");
        println!("Running Blackwell dense GEMM example {} with:", cfg.name);
        println!("  mnk:       {m},{n},{k} (simulated)");
        println!("  tiler/cluster: {:?} {:?}", cfg.mma_tiler_mn, cfg.cluster_shape_mn);
        println!("===================================================================");
        let report = run::<F16>(&cfg, (m, n, k));
        println!("{report}");
        println!("PASS");
    }
}
