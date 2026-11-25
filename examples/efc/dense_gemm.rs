// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/efc/common_dense_gemm_efc.py
//   (`DenseGemmEFC`: its configuration, `_setup_attributes`, `__call__`, `kernel` with its
//   TMA, MMA, epilogue and epilogue-load warps, `epilogue_tmem_copy_and_partition`,
//   `epilogue_smem_copy_and_partition_load`, `epilogue_gmem_copy_and_partition`,
//   `compute_stages`, `_compute_grid` and `compute_num_tmem_alloc_cols`) and
//   examples/python/CuTeDSL/cute/blackwell/efc/common_efc.py
//   (`EFC.JIT` and `EFC.Kernel`: the written and read tensors with the biggest element
//   type, `compute_stage`, `smem_layout`, `create_tma_arguments`, `allocate_smem`, the
//   partitions of the supplemental tensors and their per-subtile loads and stores),
//   with their comments and names,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! `DenseGemmEFC`: the persistent, warp-specialized SM100 GEMM of the EFC examples, whose
//! epilogue is an [`Efc`] (`common_dense_gemm_efc.py`).
//!
//! - A `(M, K)` and B `(N, K)` are K-major Float16, the accumulator Float32 in TMEM; one CTA
//!   or a CTA pair (`use_2cta_instrs`) per MMA tile, in a cluster with A multicast along N
//!   and B along M; the `StaticPersistentTileScheduler` walks the tiles.
//! - Warps 0 to 3 run the epilogue, warp 4 the MMAs, warp 5 the TMA loads of A and B and
//!   warp 6 the TMA loads of the tensors the epilogue reads (224 threads).
//! - Every tensor the epilogue reads gets a TMA load atom and a staged SMEM buffer
//!   (`make_smem_layout_epi` with the C stage count) fed through a `PipelineTmaAsync` by
//!   warp 6, one stage per epilogue subtile; every tensor it writes gets an SMEM buffer
//!   with the D stage count and a TMA store, paced by a `PipelineTmaStore`. The stage
//!   counts include their bytes (`compute_stages`).
//! - Per subtile the epilogue loads the accumulator from TMEM, the read tensors from SMEM,
//!   runs the epilogue function on the Float32 register values
//!   ([`Efc::evaluate_on_device`]), converts what it stores to each tensor's type and
//!   stores it through SMEM with TMA.
//! - The written tensor with the widest type sets the epilogue tile and the majorness of
//!   C/D (`d_name_bigger`), the widest read tensor takes part in the epilogue tile
//!   (`c_name_bigger`); every tensor must have D's majorness.
//!
//! Differences from the Python:
//! - the kernel parameters are `m`, `n`, `k` (launch values, `N` and `K` multiples of 8),
//!   `a`, `b`, then the epilogue's parameters in order (scalars as Float32, tensors as
//!   `(M, N)` tensors); no batch mode `L`, Float16 A/B only;
//! - the read tensors' registers stay in the TMEM load's order (the Python retiles them
//!   with D's register-to-SMEM copy, which pairs the same elements);
//! - each written tensor's SMEM store is chosen for its own type (the Python uses D's);
//! - without read tensors there is no C pipeline and warp 6 idles (the Python releases
//!   stages nobody filled);
//! - the epilogue waits for each TMEM load explicitly (`fence_view_async_tmem_load`), which
//!   the simulator requires, and the TMA-load warp's `try_acquire` peeks are left out.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use cute::cpasync;
use cute::memory::{
    SmemAllocator, SmemPartition, TmemAllocator, TmemAllocatorOptions, get_num_tmem_alloc_cols,
    get_smem_capacity_in_bytes,
};
use cute::nvgpu::{make_tiled_tma_atom_a, make_tiled_tma_atom_b};
use cute::pipeline::{
    self, Agent, CooperativeGroup, NamedBarrier, Pipeline, PipelineBarriers, PipelineState, PipelineTmaAsync,
    PipelineTmaAsyncOptions, PipelineTmaStore, PipelineTmaUmma, PipelineTmaUmmaOptions, PipelineUmmaAsync,
    PipelineUmmaOptions, PipelineUserType, make_pipeline_state,
};
use cute::prelude::*;
use cute::tcgen05::{CtaGroup, Field, OperandMajorMode, OperandSource};
use cute::utils::gemm::sm100::{
    EpilogueConfig, SmemCopyPartition, TmemCopyPartition, epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition, transform_partitioned_tensor_layout,
};
use cute::utils::sm100::{
    cluster_shape_to_tma_atom_a, cluster_shape_to_tma_atom_b, make_smem_layout_a, make_smem_layout_b,
    make_smem_layout_epi, make_trivial_tiled_mma,
};
use cute::utils::{
    LayoutEnum, PersistentTileSchedulerParams, StaticPersistentTileScheduler, WorkTileInfo, compute_epilogue_tile_shape,
};
use cute::{CopyOp, FenceProxyKind, FenceProxyOptions, SharedSpace, fence_proxy_with};

use super::{Efc, EfcError, HostArgs, Param, ParameterAttributes};

/// Float16 operands, Float32 accumulation and epilogue.
pub type AbDtype = F16;

/// The warps responsible for computing the epilogue function and storing the results.
pub const EPILOGUE_WARP_ID: [i32; 4] = [0, 1, 2, 3];
/// The warp responsible for computing the matrix multiplication.
pub const MMA_WARP_ID: i32 = 4;
/// The warp responsible for loading the tensors A & B to feed the MMA.
pub const TMA_WARP_ID: i32 = 5;
/// The warp responsible for loading the auxiliary tensors used in the epilogue.
pub const EPILOGUE_LOAD_WARP_ID: i32 = 6;
pub const THREADS_PER_CTA: usize = 32 * 7;
// Barrier ids for epilogue sync and tmem ptr sync.
const EPILOGUE_SYNC_BAR_ID: u32 = 2;
const TMEM_ALLOC_SYNC_BAR_ID: u32 = 3;

/// `DenseGemmEFC`'s MMA and cluster arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmConfig {
    pub name: &'static str,
    /// `CtaGroup::TWO` MMAs: a CTA pair per MMA tile.
    pub use_2cta_instrs: bool,
    /// The `(M, N)` of the MMA tile (of the pair with `use_2cta_instrs`).
    pub mma_tiler_mn: (usize, usize),
    /// The cluster `(M, N)`.
    pub cluster_shape_mn: (usize, usize),
}

/// One CTA, a `(128, 128)` tile, no cluster.
pub const EFC_GEMM_1CTA: GemmConfig =
    GemmConfig { name: "efc_gemm_1cta", use_2cta_instrs: false, mma_tiler_mn: (128, 128), cluster_shape_mn: (1, 1) };

/// A CTA pair on a `(256, 128)` tile in a `(2, 1)` cluster (the examples' default).
pub const EFC_GEMM_2CTA: GemmConfig =
    GemmConfig { name: "efc_gemm_2cta", use_2cta_instrs: true, mma_tiler_mn: (256, 128), cluster_shape_mn: (2, 1) };

/// A CTA pair in a `(2, 2)` cluster: A multicast along N, B along M.
pub const EFC_GEMM_2CTA_MULTICAST: GemmConfig =
    GemmConfig { name: "efc_gemm_2cta_multicast", cluster_shape_mn: (2, 2), ..EFC_GEMM_2CTA };

/// The GEMM and its epilogue.
#[derive(Clone)]
pub struct DenseGemmEfc {
    pub cfg: GemmConfig,
    pub efc: Efc,
}

/// The tensors that drive the tiling: the written (D) and read (C) ones with the widest
/// types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRoles {
    pub d_name_bigger: String,
    pub d_dtype: DType,
    pub d_layout: LayoutEnum,
    pub c_name_bigger: Option<String>,
    pub c_dtype: Option<DType>,
    pub c_layout: Option<LayoutEnum>,
}

/// What `_setup_attributes` computes.
pub struct Setup {
    pub roles: TensorRoles,
    pub tiled_mma: TiledMma,
    pub mma_tiler: (usize, usize, usize),
    pub cta_tile_shape_mnk: (usize, usize, usize),
    pub cluster_layout_vmnk: Layout,
    pub epi_tile: Tiler,
    pub num_acc_stage: usize,
    pub num_ab_stage: usize,
    pub num_c_stage: usize,
    pub num_d_stage: usize,
    /// The bytes of one SMEM stage of the read and of the written tensors.
    pub smem_size_of_read_tensors: usize,
    pub smem_size_of_written_tensors: usize,
    pub a_smem_layout_staged: ComposedLayout,
    pub b_smem_layout_staged: ComposedLayout,
    pub num_tmem_alloc_cols: usize,
}

cute_struct! {
    pub struct SharedStorage(num_ab_stage: usize, num_acc_stage: usize, num_c_stage: usize) {
        ab_mbar_ptr: MemRange<i64>[num_ab_stage * 2],
        acc_mbar_ptr: MemRange<i64>[num_acc_stage * 2],
        c_mbar_ptr: MemRange<i64>[num_c_stage * 2],
        tmem_dealloc_mbar: i64,
        tmem_holding_buf: i32,
    }
}

/// The static size of a layout in bytes.
fn bytes(dtype: DType, l: &ComposedLayout) -> usize {
    size_in_bytes_dtype(dtype, l).as_static().expect("the SMEM layouts are static")
}

impl DenseGemmEfc {
    pub fn new(cfg: GemmConfig, efc: Efc) -> DenseGemmEfc {
        DenseGemmEfc { cfg, efc }
    }

    fn cta_group(&self) -> CtaGroup {
        if self.cfg.use_2cta_instrs { CtaGroup::TWO } else { CtaGroup::ONE }
    }

    fn atom_ctas(&self) -> usize {
        if self.cfg.use_2cta_instrs { 2 } else { 1 }
    }

    /// The widest written and read tensors (the first of the widest, in parameter order),
    /// and the check that every tensor has D's majorness.
    pub fn tensor_roles(&self) -> Result<TensorRoles, EfcError> {
        let widest = |names: &[String]| {
            let mut best: Option<(String, DType, LayoutEnum)> = None;
            for n in names {
                let (dtype, layout) = self.efc.tensor_param(n);
                if best.as_ref().is_none_or(|b| dtype.bits() > b.1.bits()) {
                    best = Some((n.clone(), dtype, layout));
                }
            }
            best
        };
        let (d_name_bigger, d_dtype, d_layout) =
            widest(&self.efc.written_tensor_names).ok_or(EfcError::NoWrittenTensor)?;
        for n in &self.efc.used_tensor_names {
            let (_, layout) = self.efc.tensor_param(n);
            if layout != d_layout {
                return Err(EfcError::LayoutMismatch { tensor: n.clone(), layout, d_layout });
            }
        }
        let c = widest(&self.efc.read_tensor_names);
        Ok(TensorRoles {
            d_name_bigger,
            d_dtype,
            d_layout,
            c_name_bigger: c.as_ref().map(|c| c.0.clone()),
            c_dtype: c.as_ref().map(|c| c.1),
            c_layout: c.map(|c| c.2),
        })
    }

    /// The bytes of one SMEM stage of the read and of the written tensors (`compute_stage`).
    pub fn smem_sizes(&self, d_layout: LayoutEnum, epi_tile: &Tiler) -> (usize, usize) {
        let (mut read, mut written) = (0, 0);
        for n in &self.efc.used_tensor_names {
            let (dtype, _) = self.efc.tensor_param(n);
            let a = self.efc.parameter_attributes(n);
            let one = bytes(dtype, &make_smem_layout_epi(dtype, d_layout, epi_tile, 1));
            if a.is_read {
                read += one;
            }
            if a.is_written {
                written += one;
            }
        }
        (read, written)
    }

    /// `_setup_attributes`. Panics on a broken EFC rule.
    #[track_caller]
    pub fn setup(&self) -> Setup {
        let roles = self.tensor_roles().unwrap_or_else(|e| panic!("{}: {e}", self.cfg.name));
        let tiled_mma = make_trivial_tiled_mma(
            AbDtype::DTYPE,
            AbDtype::DTYPE,
            OperandMajorMode::K,
            OperandMajorMode::K,
            DType::F32,
            self.cta_group(),
            self.cfg.mma_tiler_mn,
            OperandSource::SMEM,
        );
        let mma_inst_shape_k = tiled_mma.atom().shape_mnk()[2];
        let mma_inst_tile_k = 4;
        let mma_tiler = (self.cfg.mma_tiler_mn.0, self.cfg.mma_tiler_mn.1, mma_inst_shape_k * mma_inst_tile_k);
        let cta_tile_shape_mnk = (mma_tiler.0 / tiled_mma.thr_id().size(), mma_tiler.1, mma_tiler.2);
        let (cm, cn) = self.cfg.cluster_shape_mn;
        let cluster_layout_vmnk = Layout::col_major((cm, cn, 1))
            .tiled_divide(&Tree::Branch(vec![Tree::Leaf(Some(tiled_mma.thr_id().clone()))]));

        let epi_tile = compute_epilogue_tile_shape(
            &[cta_tile_shape_mnk.0, cta_tile_shape_mnk.1],
            self.cfg.use_2cta_instrs,
            roles.d_layout,
            roles.d_dtype,
            roles.c_layout,
            roles.c_dtype,
            None,
        );

        // compute_stages
        let num_acc_stage = 2;
        let num_c_stage = 2;
        let mut num_d_stage = 2;
        let a_one = make_smem_layout_a(&tiled_mma, mma_tiler, AbDtype::DTYPE, 1, None);
        let b_one = make_smem_layout_b(&tiled_mma, mma_tiler, AbDtype::DTYPE, 1, None);
        let (smem_size_of_read_tensors, smem_size_of_written_tensors) = self.smem_sizes(roles.d_layout, &epi_tile);
        let ab_bytes_per_stage = bytes(AbDtype::DTYPE, &a_one) + bytes(AbDtype::DTYPE, &b_one);
        let mbar_helpers_bytes = 1024;
        let c_bytes = smem_size_of_read_tensors * num_c_stage;
        let d_bytes = smem_size_of_written_tensors * num_d_stage;
        let smem_capacity = get_smem_capacity_in_bytes(Some(Arch::Sm100a));
        let occupancy = 1;
        let fixed = mbar_helpers_bytes + c_bytes + d_bytes;
        assert!(
            smem_capacity / occupancy > fixed && (smem_capacity / occupancy - fixed) / ab_bytes_per_stage > 0,
            "{}: the EFC tensors leave no shared memory for the A/B stages",
            self.cfg.name
        );
        let num_ab_stage = (smem_capacity / occupancy - fixed) / ab_bytes_per_stage;
        num_d_stage += (smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy * fixed)
            / (occupancy * smem_size_of_written_tensors);

        let a_smem_layout_staged = make_smem_layout_a(&tiled_mma, mma_tiler, AbDtype::DTYPE, num_ab_stage, None);
        let b_smem_layout_staged = make_smem_layout_b(&tiled_mma, mma_tiler, AbDtype::DTYPE, num_ab_stage, None);

        // compute_num_tmem_alloc_cols
        let acc_shape = tiled_mma.partition_shape_c((mma_tiler.0, mma_tiler.1));
        let t_ct_acc_fake = tiled_mma.make_fragment_c::<f32>(&append_mode(&acc_shape, num_acc_stage));
        let num_tmem_alloc_cols = get_num_tmem_alloc_cols(&[&t_ct_acc_fake], true, Arch::Sm100a);

        Setup {
            roles,
            tiled_mma,
            mma_tiler,
            cta_tile_shape_mnk,
            cluster_layout_vmnk,
            epi_tile,
            num_acc_stage,
            num_ab_stage,
            num_c_stage,
            num_d_stage,
            smem_size_of_read_tensors,
            smem_size_of_written_tensors,
            a_smem_layout_staged,
            b_smem_layout_staged,
            num_tmem_alloc_cols,
        }
    }

    /// The epilogue subtiles of a CTA tile.
    pub fn subtile_count(st: &Setup) -> usize {
        let epi_elems = bytes(DType::F32, &make_smem_layout_epi(DType::F32, st.roles.d_layout, &st.epi_tile, 1)) / 4;
        st.cta_tile_shape_mnk.0 * st.cta_tile_shape_mnk.1 / epi_elems
    }

    /// The CTA tiles of an `(m, n)` problem.
    pub fn num_ctas_mn(&self, m: usize, n: usize) -> (usize, usize) {
        let tm = self.cfg.mma_tiler_mn.0 / self.atom_ctas();
        (m.div_ceil(tm), n.div_ceil(self.cfg.mma_tiler_mn.1))
    }

    fn sched_params(&self, ntiles: impl Into<Shape>) -> PersistentTileSchedulerParams {
        let (cm, cn) = self.cfg.cluster_shape_mn;
        PersistentTileSchedulerParams::new(ntiles, [cm, cn, 1], Default::default())
    }

    /// `_compute_grid`: at most `max_active_clusters` clusters of the persistent schedule.
    pub fn launch(&self, m: usize, n: usize, max_active_clusters: u32) -> LaunchConfig {
        let (tm, tn) = self.num_ctas_mn(m, n);
        let grid = self.sched_params((tm, tn, 1)).get_grid_shape(max_active_clusters);
        let launch = LaunchConfig::new(grid, [THREADS_PER_CTA as u32, 1, 1]);
        let (cm, cn) = self.cfg.cluster_shape_mn;
        if cm * cn > 1 { launch.with_cluster([cm as u32, cn as u32, 1]) } else { launch }
    }

    /// The kernel (`__call__` and `kernel`): parameters `m`, `n`, `k`, `a`, `b`, then the
    /// epilogue's.
    pub fn build(&self) -> Kernel {
        let gemm = self.clone();
        Kernel::builder(gemm.cfg.name).block_dim([THREADS_PER_CTA as u32, 1, 1]).target(Arch::Sm100a).build(move || {
            let st = gemm.setup();
            let m_major = st.roles.d_layout.is_m_major_c();
            let m = param_extent("m", if m_major { 8 } else { 1 });
            let n = param_extent("n", 8);
            let k = param_extent("k", 8);
            let a = param_tensor_aligned::<AbDtype>("a", Layout::new((m, k), (k, 1)), 32);
            let b = param_tensor_aligned::<AbDtype>("b", Layout::new((n, k), (k, 1)), 32);

            // The epilogue's parameters: scalars, and the tensors with their TMA atoms and
            // staged SMEM layouts (`smem_layout`, `create_tma_arguments`).
            let mut scalars = HashMap::new();
            let mut tensors: Vec<Box<dyn Supplemental>> = vec![];
            for p in gemm.efc.params() {
                match p {
                    Param::Scalar(name) => {
                        scalars.insert(name.clone(), param_scalar::<f32>(name));
                    }
                    Param::Tensor { name, dtype, layout } => {
                        let attrs = gemm.efc.parameter_attributes(name);
                        let t = make_supplemental(*dtype, name, *layout, attrs, m, n, &st);
                        if attrs.is_read || attrs.is_written {
                            tensors.push(t);
                        }
                    }
                    Param::Config => unreachable!(),
                }
            }

            let tiled_mma = st.tiled_mma.clone();
            let atom_thr_size = tiled_mma.thr_id().size();
            let (cm, cn) = gemm.cfg.cluster_shape_mn;
            let cluster = (cm * cn > 1).then(|| st.cluster_layout_vmnk.shape().clone());
            // Setup TMA load for A
            let a_op = cluster_shape_to_tma_atom_a((cm, cn, 1), tiled_mma.thr_id()).unwrap_or_else(|e| panic!("{e}"));
            let a_smem_layout = st.a_smem_layout_staged.select(&[0, 1, 2]);
            let (tma_atom_a, tma_tensor_a) =
                make_tiled_tma_atom_a(a_op, &a, a_smem_layout.clone(), st.mma_tiler, &tiled_mma, cluster.clone())
                    .into();
            // Setup TMA load for B
            let b_op = cluster_shape_to_tma_atom_b((cm, cn, 1), tiled_mma.thr_id()).unwrap_or_else(|e| panic!("{e}"));
            let b_smem_layout = st.b_smem_layout_staged.select(&[0, 1, 2]);
            let (tma_atom_b, tma_tensor_b) =
                make_tiled_tma_atom_b(b_op, &b, b_smem_layout.clone(), st.mma_tiler, &tiled_mma, cluster).into();
            let num_tma_load_bytes =
                (bytes(AbDtype::DTYPE, &a_smem_layout) + bytes(AbDtype::DTYPE, &b_smem_layout)) * atom_thr_size;

            // The tile counts of D (`_compute_grid`), for the scheduler.
            let (tm, tn) = (st.cta_tile_shape_mnk.0, st.cta_tile_shape_mnk.1);
            let ntile = |e: Int32, t: usize| IndexExpr::from((e + (t as i32 - 1)) / t as i32);
            let num_ctas_mnl = Shape::from(Tree::Branch(vec![
                Tree::Leaf(ntile(m, tm)),
                Tree::Leaf(ntile(n, tn)),
                Tree::Leaf(IndexExpr::from(1)),
            ]));

            kernel(KernelArgs {
                gemm: &gemm,
                st: &st,
                tma_atom_a: &tma_atom_a,
                tma_tensor_a: &tma_tensor_a,
                tma_atom_b: &tma_atom_b,
                tma_tensor_b: &tma_tensor_b,
                num_tma_load_bytes,
                tile_sched_params: gemm.sched_params(num_ctas_mnl),
                scalars: &scalars,
                tensors,
            });
        })
    }
}

/// `shape` with one more mode of extent `extent`.
fn append_mode(shape: &Shape, extent: usize) -> Shape {
    let Tree::Branch(mut modes) = shape.clone() else { panic!("a tuple shape") };
    modes.push(Shape::from(extent));
    Shape::from(Tree::Branch(modes))
}

/// `t[(.., .. (nones times), fixed...)]`.
fn slice_after<T>(t: &Tensor<T>, nones: usize, fixed: &[IndexExpr]) -> Tensor<T> {
    let mut coords: Vec<Tree<Option<IndexExpr>>> = vec![Tree::Leaf(None); nones];
    coords.extend(fixed.iter().cloned().map(|i| Tree::Leaf(Some(i))));
    t.slice(Tree::Branch(coords))
}

// ---------------------------------------------------------------------------------------
// The supplemental tensors (`EFC.JIT` and `EFC.Kernel`), one generic type per element type
// behind trait objects, so that one kernel mixes element types.
// ---------------------------------------------------------------------------------------

/// What the epilogue warps and the epilogue-load warp share about the tiles.
pub struct PartitionCtx<'a> {
    pub thr_mma: &'a ThrMma,
    pub mma_tiler_mn: (usize, usize),
    pub epi_tile: &'a Tiler,
    pub tidx: Int32,
}

/// What the epilogue warps' TMEM copy fixes for the supplemental tensors.
pub struct EpilogueCtx<'a> {
    pub part: PartitionCtx<'a>,
    pub ecfg: &'a EpilogueConfig,
    pub tiled_copy_t2r: &'a TiledCopy<f32>,
    /// `(T2R, T2R_M, T2R_N)`: one subtile's registers.
    pub t_tr_r_acc_shape: Shape,
}

/// A supplemental tensor in the kernel.
trait Supplemental {
    fn name(&self) -> &str;
    fn is_read(&self) -> bool;
    fn is_written(&self) -> bool;
    /// The TMA tensor of a written tensor (the D partition of the TMEM copy).
    fn written_tma_tensor(&self) -> Option<&Tensor<Ident>>;
    /// The bytes of one stage of a read tensor (0 otherwise).
    fn tma_load_bytes(&self) -> usize;
    fn prefetch_tma_descriptors(&self);
    fn allocate_smem(&mut self, smem: &SmemAllocator);
    /// The epilogue-load warp's partitions (a read tensor).
    fn loader(&self, k: &PartitionCtx) -> Option<Box<dyn Loader>>;
    /// The epilogue warps' partitions.
    fn epilogue_side(&self, k: &EpilogueCtx) -> Box<dyn EpilogueSide>;
}

/// The epilogue-load warp's view of a read tensor.
trait Loader {
    /// The subtiles of a tile.
    fn subtile_count(&self) -> usize;
    /// TMA load subtile `subtile` of tile `(mma_m, mma_n)` into stage `stage` (`load_tensor_subtiles`).
    fn load_subtile(&self, mma_m: Int32, mma_n: Int32, subtile: usize, stage: Int32, bar: &Ptr<u64>);
}

/// The epilogue warps' view of a supplemental tensor.
trait EpilogueSide {
    /// SMEM stage `index` to registers (`load_tensors_from_smem_to_register`).
    fn load_from_smem(&self, index: Int32);
    /// The registers loaded, as Float32.
    fn loaded(&self) -> Option<TensorSsa<f32>>;
    /// The stored value, converted to the tensor's type, into its registers.
    fn store_to_regs(&self, v: &TensorSsa<f32>);
    /// Registers to SMEM buffer `d_buffer` (`store_written_tensors_to_smem`).
    fn store_to_smem(&self, d_buffer: Int32);
    /// TMA store buffer `d_buffer` to subtile `subtile` of tile `(mma_m, mma_n)`
    /// (`tma_store_written_tensors_to_gmem`).
    fn tma_store(&self, d_buffer: Int32, subtile: usize, mma_m: Int32, mma_n: Int32);
}

struct TmaSide<T> {
    atom: CopyAtom<T>,
    tma_tensor: Tensor<Ident>,
    smem_layout_staged: ComposedLayout,
    smem: Option<Tensor<T>>,
}

struct Supp<T> {
    name: String,
    read: Option<TmaSide<T>>,
    written: Option<TmaSide<T>>,
}

fn make_supplemental(
    dtype: DType,
    name: &str,
    layout: LayoutEnum,
    attrs: ParameterAttributes,
    m: Int32,
    n: Int32,
    st: &Setup,
) -> Box<dyn Supplemental> {
    match dtype {
        DType::F16 => Box::new(Supp::<F16>::new(name, layout, attrs, m, n, st)),
        DType::BF16 => Box::new(Supp::<BF16>::new(name, layout, attrs, m, n, st)),
        DType::F32 => Box::new(Supp::<f32>::new(name, layout, attrs, m, n, st)),
        _ => panic!("EFC: tensor `{name}` of {dtype}: only Float16, BFloat16 and Float32 tensors are supported"),
    }
}

impl<T: Numeric> Supp<T> {
    fn new(name: &str, layout: LayoutEnum, attrs: ParameterAttributes, m: Int32, n: Int32, st: &Setup) -> Supp<T> {
        let tensor = param_tensor_aligned::<T>(
            name,
            if layout.is_m_major_c() { Layout::new((m, n), (1, m)) } else { Layout::new((m, n), (n, 1)) },
            16,
        );
        assert_eq!(LayoutEnum::from_tensor(&tensor), layout, "EFC: `{name}` is declared in its layout");
        let side = |stages: usize, load: bool| {
            let smem_layout_staged = make_smem_layout_epi(T::DTYPE, layout, &st.epi_tile, stages);
            let one = smem_layout_staged.slice((.., .., 0));
            let (atom, tma_tensor) = if load {
                cpasync::make_tiled_tma_atom(
                    cpasync::CopyBulkTensorTileG2SOp::default(),
                    &tensor,
                    one,
                    st.epi_tile.clone(),
                    1,
                )
                .into()
            } else {
                cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, &tensor, one, st.epi_tile.clone(), 1)
                    .into()
            };
            TmaSide { atom, tma_tensor, smem_layout_staged, smem: None }
        };
        Supp {
            name: name.to_string(),
            read: attrs.is_read.then(|| side(st.num_c_stage, true)),
            written: attrs.is_written.then(|| side(st.num_d_stage, false)),
        }
    }
}

/// `epilogue_gmem_copy_and_partition`: the TMA partitions `((ATOM_V, REST_V), STAGE)` of
/// the SMEM buffer and `((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN)` of the tensor.
fn gmem_partition<T: Numeric>(side: &TmaSide<T>, k: &PartitionCtx) -> (Tensor<T>, Tensor<Ident>) {
    // (bM, bN, RestM, RestN)
    let g = local_tile(&side.tma_tensor, k.mma_tiler_mn, (.., ..));
    // ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), RestM, RestN)
    let t_cg = transform_partitioned_tensor_layout(&k.thr_mma.partition_c(&g));
    // (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN)
    let g_epi = flat_divide(&t_cg, k.epi_tile.clone());
    let smem = side.smem.as_ref().expect("the SMEM is allocated");
    cpasync::tma_partition(&side.atom, 0, &make_layout(1, 1), &smem.group_modes(0, 2), &g_epi.group_modes(0, 2))
}

/// `((ATOM_V, REST_V), EPI_M * EPI_N)`: the subtiles of tile `(mma_m, mma_n)`.
fn tile_subtiles(partitioned: &Tensor<Ident>, mma_m: Int32, mma_n: Int32) -> Tensor<Ident> {
    let t = slice_after(partitioned, 3, &[mma_m.into(), mma_n.into()]);
    t.group_modes(1, t.rank())
}

impl<T: Numeric> Supplemental for Supp<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_read(&self) -> bool {
        self.read.is_some()
    }

    fn is_written(&self) -> bool {
        self.written.is_some()
    }

    fn written_tma_tensor(&self) -> Option<&Tensor<Ident>> {
        self.written.as_ref().map(|w| &w.tma_tensor)
    }

    fn tma_load_bytes(&self) -> usize {
        self.read.as_ref().map_or(0, |r| bytes(T::DTYPE, &r.smem_layout_staged.slice((.., .., 0))))
    }

    fn prefetch_tma_descriptors(&self) {
        for side in [&self.read, &self.written].into_iter().flatten() {
            cpasync::prefetch_descriptor(&side.atom);
        }
    }

    fn allocate_smem(&mut self, smem: &SmemAllocator) {
        for side in [&mut self.read, &mut self.written].into_iter().flatten() {
            side.smem =
                Some(smem.allocate_tensor::<T>(side.smem_layout_staged.clone(), 1024, None, SmemPartition::User));
        }
    }

    fn loader(&self, k: &PartitionCtx) -> Option<Box<dyn Loader>> {
        let r = self.read.as_ref()?;
        let (b_gs_s_c, b_gs_g_c_partitioned) = gmem_partition(r, k);
        Some(Box::new(SuppLoader { atom: r.atom.clone(), b_gs_s_c, b_gs_g_c_partitioned }))
    }

    fn epilogue_side(&self, k: &EpilogueCtx) -> Box<dyn EpilogueSide> {
        // epilogue_smem_copy_and_partition_load: a plain copy tiled like the TMEM load's
        // registers.
        let read = self.read.as_ref().map(|r| {
            let t_tr_r_c = make_fragment::<T>(k.t_tr_r_acc_shape.clone());
            let atom = make_copy_atom::<T>(CopyOp::Universal { bits: T::DTYPE.bits().max(8) });
            let tiled_copy_s2r = make_tiled_copy_d(&atom, k.tiled_copy_t2r);
            let t_sr_s_c = tiled_copy_s2r.get_slice(k.part.tidx).partition_d(r.smem.as_ref().expect("allocated"));
            let t_sr_r_c = tiled_copy_s2r.retile(&t_tr_r_c);
            ReadRegs { tiled_copy_s2r, t_sr_s_c, t_sr_r_c, t_tr_r_c }
        });
        let written = self.written.as_ref().map(|w| {
            let t_tr_r_d = make_fragment::<T>(k.t_tr_r_acc_shape.clone());
            let SmemCopyPartition { tiled_copy_r2s, t_rs_r_c: t_rs_r_d, t_rs_s_c: t_rs_s_d } =
                epilogue_smem_copy_and_partition::<f32, T>(
                    k.ecfg,
                    k.tiled_copy_t2r,
                    &t_tr_r_d,
                    k.part.tidx,
                    w.smem.as_ref().expect("allocated"),
                );
            let (b_sg_s_d, b_sg_g_d_partitioned) = gmem_partition(w, &k.part);
            WrittenRegs {
                atom: w.atom.clone(),
                t_tr_r_d,
                tiled_copy_r2s,
                t_rs_r_d,
                t_rs_s_d,
                b_sg_s_d,
                b_sg_g_d_partitioned,
            }
        });
        Box::new(SuppEpilogue { read, written })
    }
}

struct SuppLoader<T> {
    atom: CopyAtom<T>,
    b_gs_s_c: Tensor<T>,
    b_gs_g_c_partitioned: Tensor<Ident>,
}

impl<T: Numeric> Loader for SuppLoader<T> {
    fn subtile_count(&self) -> usize {
        let l = self.b_gs_g_c_partitioned.layout();
        l.mode(1).size() * l.mode(2).size()
    }

    fn load_subtile(&self, mma_m: Int32, mma_n: Int32, subtile: usize, stage: Int32, bar: &Ptr<u64>) {
        let b_gs_g_c = tile_subtiles(&self.b_gs_g_c_partitioned, mma_m, mma_n);
        copy_with(
            &self.atom,
            &slice_after(&b_gs_g_c, 1, &[IndexExpr::from(subtile)]),
            &slice_after(&self.b_gs_s_c, 1, &[stage.into()]),
            CopyArgs::new().tma_bar_ptr(bar),
        );
    }
}

struct ReadRegs<T> {
    tiled_copy_s2r: TiledCopy<T>,
    /// `(S2R, S2R_M, S2R_N, STAGE)`.
    t_sr_s_c: Tensor<T>,
    t_sr_r_c: Tensor<T>,
    /// `(T2R, T2R_M, T2R_N)`: the same registers in the TMEM load's order.
    t_tr_r_c: Tensor<T>,
}

struct WrittenRegs<T> {
    atom: CopyAtom<T>,
    t_tr_r_d: Tensor<T>,
    tiled_copy_r2s: TiledCopy<T>,
    t_rs_r_d: Tensor<T>,
    /// `(R2S, R2S_M, R2S_N, PIPE_D)`.
    t_rs_s_d: Tensor<T>,
    /// `((ATOM_V, REST_V), PIPE_D)`.
    b_sg_s_d: Tensor<T>,
    b_sg_g_d_partitioned: Tensor<Ident>,
}

struct SuppEpilogue<T> {
    read: Option<ReadRegs<T>>,
    written: Option<WrittenRegs<T>>,
}

impl<T: Numeric> EpilogueSide for SuppEpilogue<T> {
    fn load_from_smem(&self, index: Int32) {
        if let Some(r) = &self.read {
            copy(&r.tiled_copy_s2r, &slice_after(&r.t_sr_s_c, 3, &[index.into()]), &r.t_sr_r_c);
        }
    }

    fn loaded(&self) -> Option<TensorSsa<f32>> {
        self.read.as_ref().map(|r| r.t_tr_r_c.load_ssa().to::<f32>())
    }

    fn store_to_regs(&self, v: &TensorSsa<f32>) {
        let w = self.written.as_ref().expect("a written tensor");
        w.t_tr_r_d.store_ssa(&v.to::<T>());
    }

    fn store_to_smem(&self, d_buffer: Int32) {
        let w = self.written.as_ref().expect("a written tensor");
        copy(&w.tiled_copy_r2s, &w.t_rs_r_d, &slice_after(&w.t_rs_s_d, 3, &[d_buffer.into()]));
    }

    fn tma_store(&self, d_buffer: Int32, subtile: usize, mma_m: Int32, mma_n: Int32) {
        let w = self.written.as_ref().expect("a written tensor");
        let b_sg_g_d = tile_subtiles(&w.b_sg_g_d_partitioned, mma_m, mma_n);
        copy(
            &w.atom,
            &slice_after(&w.b_sg_s_d, 1, &[d_buffer.into()]),
            &slice_after(&b_sg_g_d, 1, &[IndexExpr::from(subtile)]),
        );
    }
}

// ---------------------------------------------------------------------------------------
// The kernel.
// ---------------------------------------------------------------------------------------

struct KernelArgs<'a> {
    gemm: &'a DenseGemmEfc,
    st: &'a Setup,
    tma_atom_a: &'a CopyAtom<AbDtype>,
    tma_tensor_a: &'a Tensor<Ident>,
    tma_atom_b: &'a CopyAtom<AbDtype>,
    tma_tensor_b: &'a Tensor<Ident>,
    num_tma_load_bytes: usize,
    tile_sched_params: PersistentTileSchedulerParams,
    scalars: &'a HashMap<String, Value<f32>>,
    tensors: Vec<Box<dyn Supplemental>>,
}

/// A role's persistent tile loop: `body` for each of the CTA's tiles, threading `state`.
fn for_each_tile<S: Carry>(
    sched: &StaticPersistentTileScheduler,
    work: WorkTileInfo,
    state: S,
    body: impl FnOnce(WorkTileInfo, S) -> S,
) -> S {
    let (_, _, state) = while_carry(
        (work, sched.clone(), state),
        |(w, _, _)| w.is_valid_tile(),
        |(w, mut sched, state)| {
            let state = body(w, state);
            sched.advance_to_next_work();
            (sched.get_current_work(), sched, state)
        },
    );
    state
}

fn kernel(args: KernelArgs) {
    let KernelArgs {
        gemm: dg,
        st,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        num_tma_load_bytes,
        tile_sched_params,
        scalars,
        mut tensors,
    } = args;
    let efc = &dg.efc;
    let tiled_mma = &st.tiled_mma;
    let mma_tiler = st.mma_tiler;
    let cluster_layout_vmnk = &st.cluster_layout_vmnk;
    let has_reads = !efc.read_tensor_names.is_empty();

    let warp_idx = make_warp_uniform(warp_idx());
    let tidx = thread_idx().x;

    // Prefetch tma descriptor
    if_(warp_idx.eq(TMA_WARP_ID), || {
        cpasync::prefetch_descriptor(tma_atom_a);
        cpasync::prefetch_descriptor(tma_atom_b);
        for t in tensors.iter() {
            t.prefetch_tma_descriptors();
        }
    });

    let bidx = block_idx().x;
    let v = cluster_layout_vmnk.mode(0).size();
    let mma_tile_coord_v = bidx % v as i32;
    let is_leader_cta = mma_tile_coord_v.eq(0);
    let as_leader = |f: &dyn Fn()| if v == 1 { f() } else { if_(is_leader_cta, f) };
    let cta_rank_in_cluster = make_warp_uniform(block_idx_in_cluster());
    let block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster);

    //
    // Alloc and init the shared storage, the A/B buffers and the EFC buffers
    //
    let smem = SmemAllocator::new();
    let storage_layout = SharedStorage::layout(st.num_ab_stage, st.num_acc_stage, st.num_c_stage);
    let storage =
        SharedStorage::from_layout(&storage_layout, &smem.allocate_struct(&storage_layout, 16, SmemPartition::User));
    let s_a = smem.allocate_tensor::<AbDtype>(st.a_smem_layout_staged.clone(), 1024, None, SmemPartition::User);
    let s_b = smem.allocate_tensor::<AbDtype>(st.b_smem_layout_staged.clone(), 1024, None, SmemPartition::User);
    for t in tensors.iter_mut() {
        t.allocate_smem(&smem);
    }
    let tensors = &tensors;

    // As many participants as the number of threads issuing the MMA in the same row and
    // column, not counting twice the same thread.
    let num_tma_producer = cluster_layout_vmnk.mode(1).size() + cluster_layout_vmnk.mode(2).size() - 1;
    let ab_pipeline = PipelineTmaUmma::create(
        st.num_ab_stage,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, num_tma_producer),
        num_tma_load_bytes,
        PipelineTmaUmmaOptions {
            barrier_storage: Some(storage.ab_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cluster_layout_vmnk.clone()),
            defer_sync: true,
            ..Default::default()
        },
    );
    let (ab_producer, ab_consumer) = ab_pipeline.make_participants();

    // One arrival per epilogue warp of each CTA of the pair (released by an elected thread).
    let acc_pipeline = PipelineUmmaAsync::create(
        st.num_acc_stage,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, (v * EPILOGUE_WARP_ID.len()) as i32),
        PipelineUmmaOptions {
            barrier_storage: Some(storage.acc_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cluster_layout_vmnk.clone()),
            defer_sync: true,
            ..Default::default()
        },
    );
    let (acc_producer, acc_consumer) = acc_pipeline.make_participants();

    // The read tensors' pipeline: warp 6 loads, lane 0 of each epilogue warp releases.
    let total_tma_load_bytes: usize = tensors.iter().map(|t| t.tma_load_bytes()).sum();
    let c_pipeline = has_reads.then(|| {
        PipelineTmaAsync::create(
            st.num_c_stage,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, EPILOGUE_WARP_ID.len() as i32),
            total_tma_load_bytes,
            PipelineTmaAsyncOptions {
                barrier_storage: Some(storage.c_mbar_ptr.data_ptr().cast::<u64>()),
                defer_sync: true,
                ..Default::default()
            },
        )
    });

    let tmem_alloc_barrier = NamedBarrier::new(TMEM_ALLOC_SYNC_BAR_ID, 32 * (1 + EPILOGUE_WARP_ID.len()));
    let mut tmem = TmemAllocator::new(
        tmem_alloc_barrier,
        TmemAllocatorOptions {
            alloc_result_dst_smem_ptr: Some(storage.tmem_holding_buf.ptr()),
            allocator_warp_id: EPILOGUE_WARP_ID[0],
            is_two_cta: v > 1,
            two_cta_tmem_dealloc_mbar_ptr: (v > 1).then(|| storage.tmem_dealloc_mbar.ptr().cast::<u64>()),
            arch: Arch::Sm100a,
            ..Default::default()
        },
    );

    pipeline::pipeline_init_arrive(Some(cluster_layout_vmnk.shape().clone()), true);

    // Multicast masks
    let clustered = cluster_layout_vmnk.size() > 1;
    let a_full_mcast_mask = clustered
        .then(|| cpasync::create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk.clone(), 2));
    let b_full_mcast_mask = clustered
        .then(|| cpasync::create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk.clone(), 1));

    // (bM, bK, RestM, RestK) and (bN, bK, RestN, RestK)
    let g_a = local_tile(tma_tensor_a, (mma_tiler.0, mma_tiler.2), (.., ..));
    let g_b = local_tile(tma_tensor_b, (mma_tiler.1, mma_tiler.2), (.., ..));
    let k_tile_cnt = size(&g_a, &[3]);

    let thr_mma = tiled_mma.get_slice(mma_tile_coord_v);
    let t_cg_a = thr_mma.partition_a(&g_a);
    let t_cg_b = thr_mma.partition_b(&g_b);

    let coord = |i: usize| match &block_in_cluster_coord_vmnk {
        Tree::Branch(modes) => modes[i].clone(),
        leaf => leaf.clone(),
    };
    let (t_as_a, t_ag_a) = cpasync::tma_partition(
        tma_atom_a,
        coord(2),
        &make_layout(cluster_layout_vmnk.mode(2).size(), 1),
        &s_a.group_modes(0, 3),
        &t_cg_a.group_modes(0, 3),
    );
    let (t_bs_b, t_bg_b) = cpasync::tma_partition(
        tma_atom_b,
        coord(1),
        &make_layout(cluster_layout_vmnk.mode(1).size(), 1),
        &s_b.group_modes(0, 3),
        &t_cg_b.group_modes(0, 3),
    );

    let t_cr_a = tiled_mma.make_fragment_a(&s_a);
    let t_cr_b = tiled_mma.make_fragment_b(&s_b);
    let acc_shape = tiled_mma.partition_shape_c((mma_tiler.0, mma_tiler.1));
    let t_ct_acc_fake = tiled_mma.make_fragment_c::<f32>(&append_mode(&acc_shape, st.num_acc_stage));

    pipeline::pipeline_init_wait(Some(cluster_layout_vmnk.shape().clone()));

    let tile_sched = StaticPersistentTileScheduler::create(&tile_sched_params, block_idx(), grid_dim());
    let work_tile = tile_sched.initial_work_tile_info();
    let mma_tile_coord_mn = |w: &WorkTileInfo| {
        let t = w.tile_idx();
        (t[0] / v as i32, t[1])
    };

    //
    // Specialized TMA load warp
    //
    if_(warp_idx.eq(TMA_WARP_ID), || {
        let mut ab_producer = for_each_tile(&tile_sched, work_tile, ab_producer.clone_participant(), |w, p| {
            let (mma_m, mma_n) = mma_tile_coord_mn(&w);
            let t_ag_a_slice = t_ag_a.slice((.., mma_m, ..));
            let t_bg_b_slice = t_bg_b.slice((.., mma_n, ..));
            for_range_carry(0, k_tile_cnt.clone(), 1, p, |k_tile, mut p| {
                let handle = p.acquire_and_advance(None);
                let bar = CopyArgs::new().tma_bar_ptr(&handle.barrier());
                let (bar_a, bar_b) = match (a_full_mcast_mask, b_full_mcast_mask) {
                    (Some(a), Some(b)) => (bar.clone().mcast_mask(a), bar.mcast_mask(b)),
                    _ => (bar.clone(), bar),
                };
                copy_with(tma_atom_a, &t_ag_a_slice.slice((.., k_tile)), &t_as_a.slice((.., handle.index())), bar_a);
                copy_with(tma_atom_b, &t_bg_b_slice.slice((.., k_tile)), &t_bs_b.slice((.., handle.index())), bar_b);
                p
            })
        });
        // Wait A/B buffer empty
        ab_producer.tail();
    });

    //
    // Specialized MMA warp
    //
    if_(warp_idx.eq(MMA_WARP_ID), || {
        tmem.wait_for_alloc();
        let tmem_ptr = tmem.retrieve_ptr::<f32>();
        let t_ct_acc_base = make_tensor(tmem_ptr, t_ct_acc_fake.layout().clone());
        let (_, acc_producer) = for_each_tile(
            &tile_sched,
            work_tile,
            (ab_consumer.clone_participant(), acc_producer.clone_participant()),
            |_, (mut ab_consumer, mut acc_producer)| {
                // The states advance on every CTA; the leader CTA of a pair waits and issues.
                let acc_empty = acc_producer.current_handle();
                acc_producer.advance(1);
                as_leader(&|| {
                    acc_empty.get_origin().producer_acquire(&acc_empty.state(), None);
                    let t_ct_acc = t_ct_acc_base.slice((.., .., .., acc_empty.index()));
                    let mut tiled_mma = tiled_mma.clone();
                    tiled_mma.set(Field::ACCUMULATE, false);
                    for_range_carry(
                        0,
                        k_tile_cnt.clone(),
                        1,
                        (ab_consumer.clone_participant(), tiled_mma),
                        |_, (mut ab_consumer, mut tiled_mma)| {
                            let handle = ab_consumer.wait_and_advance(None);
                            let num_k_blocks = t_cr_a.layout().mode(2).size();
                            for k_block_idx in 0..num_k_blocks {
                                let k_block_coord = (.., .., k_block_idx, handle.index());
                                gemm(
                                    &tiled_mma,
                                    &t_ct_acc,
                                    &t_cr_a.slice(k_block_coord),
                                    &t_cr_b.slice(k_block_coord),
                                    &t_ct_acc,
                                );
                                tiled_mma.set(Field::ACCUMULATE, true);
                            }
                            handle.release();
                            (ab_consumer, tiled_mma)
                        },
                    );
                    acc_empty.commit();
                });
                ab_consumer.advance(IntoValue::<i32>::into_value(k_tile_cnt.clone()));
                (ab_consumer, acc_producer)
            },
        );
        // Wait for accumulator buffer empty
        as_leader(&|| acc_producer.clone_participant().tail());
    });

    //
    // Specialized epilogue warps
    //
    if_(warp_idx.lt(MMA_WARP_ID), || {
        tmem.allocate(st.num_tmem_alloc_cols);
        tmem.wait_for_alloc();
        let tmem_ptr = tmem.retrieve_ptr::<f32>();
        let t_ct_acc_base = make_tensor(tmem_ptr.clone(), t_ct_acc_fake.layout().clone());
        // ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE)
        let t_ct_acc = transform_partitioned_tensor_layout(&t_ct_acc_base);

        // The TMEM load, for D's layout and the Float32 epilogue type.
        let mut ecfg = EpilogueConfig::new(
            st.cta_tile_shape_mnk,
            st.roles.d_layout,
            DType::F32,
            DType::F32,
            dg.cfg.use_2cta_instrs,
        );
        ecfg.epilog_sync_bar_id = EPILOGUE_SYNC_BAR_ID;
        ecfg.num_c_stage = st.num_d_stage;
        let d = tensors.iter().find(|t| t.name() == st.roles.d_name_bigger).expect("D is used");
        let g_d = local_tile(d.written_tma_tensor().expect("D is written"), (mma_tiler.0, mma_tiler.1), (.., ..));
        let t_cg_d = transform_partitioned_tensor_layout(&thr_mma.partition_c(&g_d));
        let TmemCopyPartition { tiled_copy_t2r, t_tr_t_acc: t_tr_t_acc_base, t_tr_r_acc } =
            epilogue_tmem_copy_and_partition(&ecfg, tidx, &t_ct_acc, &t_cg_d, &st.epi_tile, dg.cfg.use_2cta_instrs);

        // copy_and_partition_supplemental_rmem_tensors
        let ectx = EpilogueCtx {
            part: PartitionCtx {
                thr_mma: &thr_mma,
                mma_tiler_mn: (mma_tiler.0, mma_tiler.1),
                epi_tile: &st.epi_tile,
                tidx,
            },
            ecfg: &ecfg,
            tiled_copy_t2r: &tiled_copy_t2r,
            t_tr_r_acc_shape: t_tr_r_acc.shape().clone(),
        };
        let sides: Vec<(String, bool, bool, Box<dyn EpilogueSide>)> = tensors
            .iter()
            .map(|t| (t.name().to_string(), t.is_read(), t.is_written(), t.epilogue_side(&ectx)))
            .collect();

        let d_pipeline = PipelineTmaStore::create(
            st.num_d_stage,
            CooperativeGroup::new(Agent::Thread, 32 * EPILOGUE_WARP_ID.len() as i32),
        );
        let epilogue_sync_barrier = NamedBarrier::new(EPILOGUE_SYNC_BAR_ID, 32 * EPILOGUE_WARP_ID.len());
        let fence = || {
            fence_proxy_with(
                FenceProxyKind::AsyncShared,
                FenceProxyOptions { space: Some(SharedSpace::Cta), ..Default::default() },
            )
        };

        let c_state = make_pipeline_state(PipelineUserType::Consumer, st.num_c_stage);
        for_each_tile(
            &tile_sched,
            work_tile,
            (acc_consumer.state(), c_state, Int32::from(0)),
            |w, (acc_state, mut c_state, tiles_done): (PipelineState, PipelineState, Int32)| {
                let (mma_m, mma_n) = mma_tile_coord_mn(&w);
                // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                let t_tr_t_acc = slice_after(&t_tr_t_acc_base, 5, &[acc_state.index().into()]);
                // Wait for accumulator buffer full
                acc_pipeline.consumer_wait(&acc_state, None);
                let t_tr_t_acc = t_tr_t_acc.group_modes(3, t_tr_t_acc.rank());
                // Store accumulator to global memory in subtiles
                let subtile_cnt = t_tr_t_acc.layout().mode(3).size();
                let num_prev_subtiles = tiles_done * subtile_cnt as i32;
                for subtile_idx in 0..subtile_cnt {
                    // Load accumulator from tensor memory buffer to register
                    copy(&tiled_copy_t2r, &slice_after(&t_tr_t_acc, 3, &[IndexExpr::from(subtile_idx)]), &t_tr_r_acc);
                    fence_view_async_tmem_load();
                    // Load the read tensors from shared memory to registers
                    if let Some(c_pipeline) = &c_pipeline {
                        c_pipeline.consumer_wait(&c_state, None);
                        for (_, _, _, s) in &sides {
                            s.load_from_smem(c_state.index());
                        }
                        fence();
                        c_pipeline.consumer_release(&c_state);
                        c_state.advance();
                    }
                    // The epilogue computation
                    let acc_vec = t_tr_r_acc.load_ssa();
                    let loads: HashMap<String, TensorSsa<f32>> =
                        sides.iter().filter_map(|(n, _, _, s)| s.loaded().map(|v| (n.clone(), v))).collect();
                    let stores = efc.evaluate_on_device(acc_vec, &loads, scalars);
                    for (n, _, written, s) in &sides {
                        if *written {
                            s.store_to_regs(stores.get(n).unwrap_or_else(|| panic!("EFC: `{n}` is not stored")));
                        }
                    }
                    // Store the written tensors to shared memory
                    let d_buffer = (num_prev_subtiles + subtile_idx as i32) % (st.num_d_stage as i32);
                    for (_, _, written, s) in &sides {
                        if *written {
                            s.store_to_smem(d_buffer);
                        }
                    }
                    // Fence and barrier to make sure shared memory store is visible to TMA store
                    fence();
                    epilogue_sync_barrier.arrive_and_wait();
                    // TMA store the written tensors to global memory
                    if_(warp_idx.eq(EPILOGUE_WARP_ID[0]), || {
                        for (_, _, written, s) in &sides {
                            if *written {
                                s.tma_store(d_buffer, subtile_idx, mma_m, mma_n);
                            }
                        }
                        d_pipeline.producer_commit();
                        d_pipeline.producer_acquire();
                    });
                    epilogue_sync_barrier.arrive_and_wait();
                }
                // Async arrive accumulator buffer empty
                if_(elect_one(), || acc_pipeline.consumer_release(&acc_state));
                let mut next = acc_state;
                next.advance();
                (next, c_state, tiles_done + 1)
            },
        );

        // Wait for the stores, then dealloc the tensor memory buffer
        if_(warp_idx.eq(EPILOGUE_WARP_ID[0]), || d_pipeline.producer_tail());
        tmem.relinquish_alloc_permit();
        epilogue_sync_barrier.arrive_and_wait();
        tmem.free(&tmem_ptr, 0);
    });

    //
    // Specialized epilogue-load warp: the read tensors' subtiles, one C stage each
    //
    if let Some(c_pipeline) = &c_pipeline {
        if_(warp_idx.eq(EPILOGUE_LOAD_WARP_ID), || {
            let part = PartitionCtx {
                thr_mma: &thr_mma,
                mma_tiler_mn: (mma_tiler.0, mma_tiler.1),
                epi_tile: &st.epi_tile,
                tidx,
            };
            let loaders: Vec<Box<dyn Loader>> = tensors.iter().filter_map(|t| t.loader(&part)).collect();
            let subtile_cnt = loaders[0].subtile_count();
            assert!(
                loaders.iter().all(|l| l.subtile_count() == subtile_cnt),
                "EFC: the read tensors have different subtile counts"
            );
            let state = make_pipeline_state(PipelineUserType::Producer, st.num_c_stage);
            let mut state = for_each_tile(&tile_sched, work_tile, state, |w, mut state: PipelineState| {
                let (mma_m, mma_n) = mma_tile_coord_mn(&w);
                for subtile_idx in 0..subtile_cnt {
                    c_pipeline.producer_acquire(&state, None);
                    let bar = c_pipeline.producer_get_barrier(&state);
                    for l in &loaders {
                        l.load_subtile(mma_m, mma_n, subtile_idx, state.index(), &bar);
                    }
                    state.advance();
                }
                state
            });
            c_pipeline.producer_tail(&mut state);
        });
    }
}

// ---------------------------------------------------------------------------------------
// The host side.
// ---------------------------------------------------------------------------------------

/// Integers in `[-2, 2)` from a fixed LCG (the examples' `random_(-2, 2)`).
pub fn small_ints(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) % 4) as f64 - 2.0
        })
        .collect()
}

/// `A B^T` in `f64`, laid out in `layout`.
pub fn matmul(a: &[f64], b: &[f64], (m, n, k): (usize, usize, usize), layout: LayoutEnum) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for r in 0..m {
        for c in 0..n {
            let acc: f64 = (0..k).map(|i| a[r * k + i] * b[c * k + i]).sum();
            out[if layout.is_m_major_c() { c * m + r } else { r * n + c }] = acc;
        }
    }
    out
}

/// A host buffer of `dtype` from `f64` values.
fn to_buffer(dtype: DType, v: &[f64]) -> KernelArg {
    match dtype {
        DType::F16 => KernelArg::buffer(&v.iter().map(|&x| F16::from_f32(x as f32)).collect::<Vec<_>>()),
        DType::BF16 => KernelArg::buffer(&v.iter().map(|&x| BF16::from_f32(x as f32)).collect::<Vec<_>>()),
        DType::F32 => KernelArg::buffer(&v.iter().map(|&x| x as f32).collect::<Vec<_>>()),
        _ => panic!("EFC: tensors of {dtype} are not supported"),
    }
}

fn from_buffer(dtype: DType, arg: &KernelArg) -> Vec<f64> {
    match dtype {
        DType::F16 => arg.to_vec::<F16>().iter().map(|x| x.to_f32() as f64).collect(),
        DType::BF16 => arg.to_vec::<BF16>().iter().map(|x| x.to_f32() as f64).collect(),
        DType::F32 => arg.to_vec::<f32>().iter().map(|&x| x as f64).collect(),
        _ => unreachable!(),
    }
}

/// A simulated run: the tensors after the kernel, the host reference, and the report.
pub struct EfcRun {
    pub device: HashMap<String, Vec<f64>>,
    pub reference: HostArgs,
    pub report: SimReport,
}

impl DenseGemmEfc {
    /// Simulate an `m x n x k` problem on `inputs` (every scalar and tensor of the
    /// epilogue, tensors in their layouts) with A and B of small integers, at most
    /// `max_active_clusters` clusters, under `opts` and the schedules `seeds`; and evaluate
    /// the epilogue on the host.
    pub fn simulate(
        &self,
        (m, n, k): (usize, usize, usize),
        inputs: &HostArgs,
        max_active_clusters: u32,
        opts: &SimOptions,
        seeds: &[u64],
    ) -> EfcRun {
        let name = self.cfg.name;
        let roles = self.tensor_roles().unwrap_or_else(|e| panic!("{name}: {e}"));
        let kernel = self.build();
        let a = small_ints(m * k, 1111);
        let b = small_ints(n * k, 2222);
        let mut args = vec![
            KernelArg::scalar(m as i32),
            KernelArg::scalar(n as i32),
            KernelArg::scalar(k as i32),
            to_buffer(AbDtype::DTYPE, &a),
            to_buffer(AbDtype::DTYPE, &b),
        ];
        for p in self.efc.params() {
            match p {
                Param::Scalar(s) => args.push(KernelArg::scalar(inputs.scalars[s] as f32)),
                Param::Tensor { name: t, dtype, .. } => {
                    let v = &inputs.tensors[t];
                    assert_eq!(v.len(), m * n, "{name}: tensor `{t}` is not {m}x{n}");
                    args.push(to_buffer(*dtype, v));
                }
                Param::Config => unreachable!(),
            }
        }
        let report = kernel
            .simulate_schedules(self.launch(m, n, max_active_clusters), &mut args, opts, seeds)
            .unwrap_or_else(|e| panic!("{name} {m}x{n}x{k}: {e}"));
        let mut device = HashMap::new();
        for (i, p) in self.efc.params().iter().enumerate() {
            if let Param::Tensor { name: t, dtype, .. } = p {
                device.insert(t.clone(), from_buffer(*dtype, &args[5 + i]));
            }
        }
        let mut reference = inputs.clone();
        self.efc.evaluate_on_host(&matmul(&a, &b, (m, n, k), roles.d_layout), &mut reference);
        EfcRun { device, reference, report }
    }
}

impl EfcRun {
    /// Every tensor of the device equals the reference within `|got - want| <= atol + rtol
    /// |want|` (exactly for zeros).
    #[track_caller]
    pub fn assert_close(&self, what: &str, atol: f64, rtol: f64) {
        for (t, want) in &self.reference.tensors {
            let got = &self.device[t];
            for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - w).abs() <= atol + rtol * w.abs(),
                    "{what}: tensor `{t}` element {i}: device {g}, reference {w} (atol {atol}, rtol {rtol})"
                );
            }
        }
    }
}
