// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py
//   (`HopperWgmmaGemmKernel`: its configuration, `_setup_attributes`,
//   `__call__`, `kernel`, `_compute_stages`, `_make_smem_layouts`,
//   `_compute_grid`, `_make_tma_store_atoms_and_tensors`,
//   `_make_tma_atoms_and_tensors`, `is_valid_dtypes`,
//   `is_valid_tensor_alignment`, the tensor set-up of `run` and the
//   comments),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A batched dense GEMM (`C = A * B`) for NVIDIA Hopper (SM90), ported from
//! the CuTe DSL example `hopper/kernel/dense_gemm/dense_gemm.py`.
//!
//! - Matrix A is `MxKxL`, L the batch dimension; A is row-major ("K") or
//!   column-major ("M").
//! - Matrix B is `NxKxL`; B is row-major ("N") or column-major ("K").
//! - Matrix C is `MxNxL`; C is row-major ("N") or column-major ("M").
//!
//! The kernel:
//!
//! 1. loads A and B tiles from global to shared memory with TMA
//!    (`make_tiled_tma_atom`, `tma_partition`), multicast across the CTAs
//!    of a thread-block cluster that share them (A along N, B along M, the
//!    masks from `make_layout_image_mask` on the cluster layout), into a
//!    multi-stage ring guarded by full and empty mbarriers;
//! 2. multiplies with Hopper's warpgroup MMA (WGMMA): A and B straight from
//!    shared memory through descriptor fragments (`make_fragment_A/B`),
//!    `k_pipe_mmas` groups of MMAs in flight (`warpgroup::wait_group`);
//! 3. stores the accumulators registers → shared memory with `stmatrix`
//!    (`StMatrix8x8x16bOp` through `make_tiled_copy_C_atom` and
//!    `make_tiled_copy_S`), then shared → global with TMA stores in bulk
//!    async-groups, one epilogue tile at a time.
//!
//! cute-rs has no SM90 GPU to run it on: the tests check it in the simulator
//! against a plain Rust GEMM, bit for bit on small integers, and assemble its
//! PTX with `ptxas -arch=sm_90a`.
//!
//! The pipeline is the library's: `PipelineTmaAsync` for the A/B stages
//! (arrival counts, peer-CTA consumer release, `pipeline_init_arrive` /
//! `pipeline_init_wait`), `PipelineTmaStore` for the epilogue, a
//! `cute_struct!` for the shared memory, and the `utils::sm90`,
//! `utils::sm100` (`check_gemm_tma_alignment`) and `utils::gemm` helpers.
//!
//! Differences from the example:
//!
//! * C's staging buffer aliases A's (the example recasts `sA`'s pointer), except
//!   for 8-bit A with 16-bit C: cute-rs's `stmatrix` stores 16-bit
//!   elements, so C gets a struct member of its own there and
//!   `_compute_stages` counts its bytes.
//! * The inputs come from this file's own generator (small integers, or
//!   fractions that round), not `create_gemm_tensor`'s values of -1 to 1,
//!   so that the tests can compare with the earlier hand-pipelined version;
//!   the reference is `utils::gemm::reference_gemm_values`.
//! * `consumers_in_threads` counts the consumers in threads instead of
//!   warps, for the diagnostics test that shows the hang.
//! * The CTA swizzle guards its division by the number of regular cluster
//!   groups: with fewer than eight clusters along M there are none, and the
//!   example's `cluster_id % num_reg_cids` divides by zero (harmless on the
//!   GPU, whose result the tail branch then overrides; the simulator
//!   reports it).
//! * cute-rs's `StMatrix8x8x16bOp` atom takes 16-bit elements only; for
//!   another C type the copy that `make_tiled_copy_C_atom` tiles from is a
//!   128-bit universal atom of C, the same 128 bits per thread that
//!   `stmatrix .x4` stores, and the store itself is `CopyUniversalOp`, as
//!   `sm90_get_smem_store_op` picks.
//! * BFloat16 inputs are accepted (the example's `is_valid_dtypes` lists
//!   Float16 only; the warpgroup MMA takes BFloat16 with Float32
//!   accumulators).
//! * The accumulators are a register fragment (`make_fragment_c`), where the
//!   example's `make_rmem_tensor` gives one of `tCgC`'s shape.
//!
//! The problem size is given at launch (`m`, `n`, `k`, `l` are kernel
//! parameters, as the example marks its tensors' layouts dynamic): one built
//! kernel serves every size whose contiguous extents are 16-byte multiples.
//!
//! Its shared memory is far above 48 KB, so its buffers live in the
//! dynamic window at nonzero offsets; their addresses are the window's
//! pointer plus each start, which LLVM 21, 22 and 23 all compile
//! correctly (`tests/dyn_smem_address.rs`).
//!
//! Run with `cargo run --release --example hopper_gemm` (on the simulator).

use cute::cpasync;
use cute::cute_struct;
use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineBarriers, PipelineState, PipelineTmaAsync, PipelineTmaAsyncOptions,
    PipelineTmaStore, PipelineUserType, make_pipeline_state, pipeline_init_arrive, pipeline_init_wait,
};
use cute::prelude::*;
use cute::testing::CantImplement;
use cute::utils::gemm::{reference_gemm_values, to_memory_order};
use cute::utils::sm90;
use cute::utils::sm100::{GemmMajor, check_gemm_tma_alignment};
use cute::warpgroup::{self, OperandSource};

cute_struct! {
    /// The kernel's shared memory (`SharedStorage`): the mainloop's full and
    /// empty mbarriers (`mainloop_pipeline_array_ptr`, `2 * ab_stage`
    /// of them), then A's and B's staged tiles (`a_elems` and `b_elems`
    /// elements, the swizzled layouts' extents), each aligned to 1024 bytes.
    /// C's staging tiles (`c_elems` elements) take a buffer of their own only
    /// when they cannot alias A's ([`epi_reuses_smem_a`]); the example's
    /// `sC` is A's buffer recast with the epilogue swizzle.
    pub struct SharedStorage<A, B, C>(ab_stage: usize, a_elems: usize, b_elems: usize, c_elems: usize) {
        mainloop_pipeline_array_ptr: MemRange<u64>[2 * ab_stage],
        #[align(1024)]
        s_a: MemRange<A>[a_elems],
        #[align(1024)]
        s_b: MemRange<B>[b_elems],
        #[align(1024)]
        s_c: MemRange<C>[c_elems],
    }
}

// /////////////////////////////////////////////////////////////////////////////
//  Host setup and device kernel launch
// /////////////////////////////////////////////////////////////////////////////

/// The major mode of each operand, as `run`'s `--a_major`, `--b_major` and
/// `--c_major` give them: A `k` or `m`, B `k` or `n`, C `n` or `m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Majors {
    pub a: char,
    pub b: char,
    pub c: char,
}

impl Majors {
    pub const KKN: Majors = Majors { a: 'k', b: 'k', c: 'n' };

    pub fn new(a: char, b: char, c: char) -> Majors {
        assert!(matches!(a, 'k' | 'm') && matches!(b, 'k' | 'n') && matches!(c, 'n' | 'm'), "majors {a} {b} {c}");
        Majors { a, b, c }
    }

    fn layouts(self) -> (LayoutEnum, LayoutEnum, LayoutEnum) {
        let e = |row| if row { LayoutEnum::ROW_MAJOR } else { LayoutEnum::COL_MAJOR };
        (e(self.a == 'k'), e(self.b == 'k'), e(self.c == 'n'))
    }
}

/// This class implements batched matrix multiplication (C = A x B) with
/// support for various data types and architectural features specific to
/// Hopper GPUs (`HopperWgmmaGemmKernel`).
///
/// Supported A/B data types: Float16 (A and B the same; BFloat16 too, see
/// the module docs), Float8E4M3FN/Float8E5M2 (A and B may differ, K-major
/// only), Int8/Uint8 (A and B may differ, K-major only). Accumulators:
/// Float32 or Float16 for floating-point inputs, Int32 for 8-bit integers.
/// CTA tile M 64 or 128, N 64, 128 or 256, K four MMA k-blocks; cluster M
/// and N powers of 2, at most 4 CTAs.
#[derive(Clone, Debug)]
pub struct HopperWgmmaGemmKernel {
    pub acc_dtype: DType,
    pub cluster_shape_mn: (usize, usize),
    pub tile_shape_mnk: (usize, usize, usize),
    /// For large tile size, using two warp groups is preferred because using
    /// only one warp group may result in register spill.
    pub atom_layout_mnk: (usize, usize, usize),
    pub num_mcast_ctas_a: usize,
    pub num_mcast_ctas_b: usize,
    pub is_a_mcast: bool,
    pub is_b_mcast: bool,
    pub occupancy: usize,
    pub mma_warp_groups: usize,
    pub num_threads_per_warp_group: usize,
    pub threads_per_cta: usize,
    pub smem_capacity: usize,
    pub ab_stage: usize,
    pub epi_stage: usize,
    pub epi_tile: (usize, usize),
    pub buffer_align_bytes: usize,
    /// Count the mainloop's consumers in threads, not warps (for the
    /// diagnostics tests): the empty barriers then expect more arrivals than
    /// the signalling lanes make, and the kernel hangs.
    pub consumers_in_threads: bool,
    // Set by `setup` (`__call__` reads them from the tensors).
    pub a_dtype: DType,
    pub b_dtype: DType,
    pub c_dtype: DType,
    pub a_layout: LayoutEnum,
    pub b_layout: LayoutEnum,
    pub c_layout: LayoutEnum,
}

impl HopperWgmmaGemmKernel {
    /// Initializes the configuration for a Hopper dense GEMM kernel
    /// (`__init__`).
    pub fn new(acc_dtype: DType, tile_shape_mn: (usize, usize), cluster_shape_mn: (usize, usize)) -> Self {
        // K dimension is deferred in _setup_attributes
        let tile_shape_mnk = (tile_shape_mn.0, tile_shape_mn.1, 1);
        let atom_layout_mnk = if tile_shape_mnk.0 > 64 && tile_shape_mnk.1 > 128 { (2, 1, 1) } else { (1, 1, 1) };
        let mma_warp_groups = atom_layout_mnk.0 * atom_layout_mnk.1 * atom_layout_mnk.2;
        let num_threads_per_warp_group = 128;
        HopperWgmmaGemmKernel {
            acc_dtype,
            cluster_shape_mn,
            tile_shape_mnk,
            atom_layout_mnk,
            num_mcast_ctas_a: 0,
            num_mcast_ctas_b: 0,
            is_a_mcast: false,
            is_b_mcast: false,
            occupancy: 1,
            mma_warp_groups,
            num_threads_per_warp_group,
            threads_per_cta: mma_warp_groups * num_threads_per_warp_group,
            // utils.get_smem_capacity_in_bytes("sm_90")
            smem_capacity: cute::memory::get_smem_capacity_in_bytes(Some(Arch::Sm90)),
            ab_stage: 0,
            epi_stage: 0,
            epi_tile: (0, 0),
            buffer_align_bytes: 1024,
            consumers_in_threads: false,
            a_dtype: DType::F16,
            b_dtype: DType::F16,
            c_dtype: DType::F16,
            a_layout: LayoutEnum::ROW_MAJOR,
            b_layout: LayoutEnum::ROW_MAJOR,
            c_layout: LayoutEnum::ROW_MAJOR,
        }
    }

    /// The configuration for operands of these types and majors: the type
    /// checks of `__call__`, then `_setup_attributes`.
    pub fn setup(mut self, a_dtype: DType, b_dtype: DType, c_dtype: DType, majors: Majors) -> Self {
        self.a_dtype = a_dtype;
        self.b_dtype = b_dtype;
        self.c_dtype = c_dtype;
        (self.a_layout, self.b_layout, self.c_layout) = majors.layouts();
        let (aw, bw) = (a_dtype.bits(), b_dtype.bits());
        assert!(!(aw == 16 && a_dtype != b_dtype), "TypeError: Type mismatch: {a_dtype} != {b_dtype}");
        assert!(aw == bw, "TypeError: Type width mismatch: {aw} != {bw}");
        assert!(aw == 16 || aw == 8, "TypeError: a_dtype should be float16 or float8");
        self.setup_attributes();
        self
    }

    /// Set up configurations that are dependent on GEMM inputs
    /// (`_setup_attributes`): the tiled MMA, the tile's K, the multicast
    /// CTAs, the epilogue subtile, the stage counts and the shared-memory
    /// layouts (made when the kernel is built, [`Self::make_smem_layouts`]).
    fn setup_attributes(&mut self) {
        // check the cta tile shape
        assert!(matches!(self.tile_shape_mnk.0, 64 | 128), "ValueError: CTA tile shape M must be 64/128");
        assert!(matches!(self.tile_shape_mnk.1, 64 | 128 | 256), "ValueError: CTA tile shape N must be 64/128/256");

        let tiled_mma = self.tiled_mma();
        let mma_inst_shape_k = tiled_mma.tile_size(2);
        let mma_inst_tile_k = 4;
        self.tile_shape_mnk = (self.tile_shape_mnk.0, self.tile_shape_mnk.1, mma_inst_shape_k * mma_inst_tile_k);

        self.num_mcast_ctas_a = self.cluster_shape_mn.1;
        self.num_mcast_ctas_b = self.cluster_shape_mn.0;
        self.is_a_mcast = self.num_mcast_ctas_a > 1;
        self.is_b_mcast = self.num_mcast_ctas_b > 1;

        let is_cooperative = self.atom_layout_mnk == (2, 1, 1);
        self.epi_tile = sm90::compute_tile_shape_or_override(self.tile_shape_mnk, self.c_dtype, is_cooperative, None);

        // Compute stage before compute smem layout
        (self.ab_stage, self.epi_stage) = Self::compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            (self.c_dtype, self.epi_tile),
            self.smem_capacity,
            self.occupancy,
        );
    }

    /// `sm90_utils.make_trivial_tiled_mma(a_dtype, b_dtype, a major, b major,
    /// acc_dtype, atom_layout_mnk, tiler_mn=(64, tile N))`.
    pub fn tiled_mma(&self) -> TiledMma {
        sm90::make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            (64, self.tile_shape_mnk.1),
            OperandSource::SMEM,
        )
    }

    /// `cute.make_layout((*cluster_shape_mn, 1))`.
    pub fn cta_layout_mnk(&self) -> Layout {
        Layout::col_major((self.cluster_shape_mn.0, self.cluster_shape_mn.1, 1))
    }

    /// The kernel's name for its configuration.
    pub fn name(&self) -> String {
        let (tm, tn, _) = self.tile_shape_mnk;
        let (cm, cn) = self.cluster_shape_mn;
        let majors = gemm_majors(self);
        format!(
            "hopper_gemm_{}_{}_{}_{}_{}{}{}_t{tm}x{tn}_c{cm}x{cn}",
            self.a_dtype, self.b_dtype, self.c_dtype, self.acc_dtype, majors.a, majors.b, majors.c,
        )
    }

    /// The kernel (`__call__`, which makes the TMA atoms, and `kernel`): its
    /// parameters are the extents `m`, `n`, `k` and `l`, then the tensors
    /// `a` (`m x k x l`), `b` (`n x k x l`) and `c` (`m x n x l`) in the
    /// configured majors. Launch it with [`Self::launch`].
    pub fn build(&self) -> Kernel {
        use DType as D;
        type E4 = F8E4M3FN;
        type E5 = F8E5M2;
        // The type combinations this example instantiates (each is a separate
        // monomorphisation of the kernel).
        match (self.a_dtype, self.b_dtype, self.c_dtype, self.acc_dtype) {
            (D::F16, D::F16, D::F16, D::F32) => self.build_typed::<F16, F16, F16, f32>(),
            (D::F16, D::F16, D::F16, D::F16) => self.build_typed::<F16, F16, F16, F16>(),
            (D::F16, D::F16, D::F32, D::F32) => self.build_typed::<F16, F16, f32, f32>(),
            (D::BF16, D::BF16, D::BF16, D::F32) => self.build_typed::<BF16, BF16, BF16, f32>(),
            (D::F8E4M3FN, D::F8E4M3FN, D::F16, D::F32) => self.build_typed::<E4, E4, F16, f32>(),
            (D::F8E4M3FN, D::F8E5M2, D::F32, D::F32) => self.build_typed::<E4, E5, f32, f32>(),
            (D::F8E5M2, D::F8E4M3FN, D::F16, D::F16) => self.build_typed::<E5, E4, F16, F16>(),
            (D::F8E5M2, D::F8E5M2, D::F8E5M2, D::F32) => self.build_typed::<E5, E5, E5, f32>(),
            (D::I8, D::I8, D::I32, D::I32) => self.build_typed::<i8, i8, i32, i32>(),
            (D::U8, D::I8, D::F16, D::I32) => self.build_typed::<u8, i8, F16, i32>(),
            (a, b, c, acc) => panic!(
                "hopper_gemm: A {a}, B {b}, C {c}, Acc {acc} is not among the combinations this example instantiates"
            ),
        }
    }

    fn build_typed<A: Numeric, B: Numeric, C: Numeric, Acc: Numeric>(&self) -> Kernel {
        let gemm = self.clone();
        Kernel::builder(self.name()).block_dim([self.threads_per_cta as u32, 1, 1]).target(Arch::Sm90a).build(
            move || {
                let (a_major_k, b_major_k, c_major_n) =
                    (gemm.a_layout.is_k_major_a(), gemm.b_layout.is_k_major_b(), gemm.c_layout.is_n_major_c());
                // The contiguous extents are whole 16-byte vectors
                // (`is_valid_tensor_alignment`), which TMA needs.
                let vec = |d: DType| 128 / d.bits();
                let mut align = (1, 1, 1);
                if a_major_k {
                    align.2 = align.2.max(vec(A::DTYPE))
                } else {
                    align.0 = align.0.max(vec(A::DTYPE))
                }
                if b_major_k {
                    align.2 = align.2.max(vec(B::DTYPE))
                } else {
                    align.1 = align.1.max(vec(B::DTYPE))
                }
                if c_major_n {
                    align.1 = align.1.max(vec(C::DTYPE))
                } else {
                    align.0 = align.0.max(vec(C::DTYPE))
                }
                let m = param_extent("m", align.0);
                let n = param_extent("n", align.1);
                let k = param_extent("k", align.2);
                let l = param_extent("l", 1);
                let (mi, ni, ki) = (IndexExpr::from(m), IndexExpr::from(n), IndexExpr::from(k));
                // (mode0, mode1, l), mode0 or mode1 contiguous, batches packed.
                let tensor_layout = |s0: &IndexExpr, s1: &IndexExpr, mode0_major: bool| {
                    let batch = s0.clone() * s1.clone();
                    let shape = (s0.clone(), s1.clone(), IndexExpr::from(l));
                    if mode0_major {
                        Layout::new(shape, (IndexExpr::from(1), s0.clone(), batch))
                    } else {
                        Layout::new(shape, (s1.clone(), IndexExpr::from(1), batch))
                    }
                };
                let a = param_tensor_aligned::<A>("a", tensor_layout(&mi, &ki, !a_major_k), 16);
                let b = param_tensor_aligned::<B>("b", tensor_layout(&ni, &ki, !b_major_k), 16);
                let c = param_tensor_aligned::<C>("c", tensor_layout(&mi, &ni, !c_major_n), 16);
                gemm.call::<A, B, C, Acc>(&a, &b, &c);
            },
        )
    }

    /// Execute the GEMM operation in steps (`__call__`): the TMA load and
    /// store atoms and tensors, the shared-memory layouts, then the kernel.
    fn call<A: Numeric, B: Numeric, C: Numeric, Acc: Numeric>(&self, a: &Tensor<A>, b: &Tensor<B>, c: &Tensor<C>) {
        let (a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged) = Self::make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        );
        let (tma_atom_a, tma_tensor_a) = Self::make_tma_atoms_and_tensors(
            a,
            &a_smem_layout_staged,
            (self.tile_shape_mnk.0, self.tile_shape_mnk.2),
            self.cluster_shape_mn.1,
        );
        let (tma_atom_b, tma_tensor_b) = Self::make_tma_atoms_and_tensors(
            b,
            &b_smem_layout_staged,
            (self.tile_shape_mnk.1, self.tile_shape_mnk.2),
            self.cluster_shape_mn.0,
        );
        let (tma_atom_c, tma_tensor_c) =
            Self::make_tma_store_atoms_and_tensors(c, &epi_smem_layout_staged, self.epi_tile);
        self.kernel::<A, B, C, Acc>(
            &tma_atom_a,
            &tma_tensor_a,
            &tma_atom_b,
            &tma_tensor_b,
            &tma_atom_c,
            &tma_tensor_c,
            self.tiled_mma(),
            &self.cta_layout_mnk(),
            &a_smem_layout_staged,
            &b_smem_layout_staged,
            &epi_smem_layout_staged,
        );
    }

    /// The launch for an `m x n x l` problem (`_compute_grid` and the
    /// `launch` of `__call__`): one CTA per output tile, rounded up to whole
    /// clusters, `l` along z, in clusters of `cluster_shape_mn`.
    pub fn launch(&self, m: usize, n: usize, l: usize) -> LaunchConfig {
        let grid = Self::compute_grid((m, n, l), self.tile_shape_mnk, self.cluster_shape_mn);
        let cfg = LaunchConfig::new(grid, [self.threads_per_cta as u32, 1, 1]);
        let (cm, cn) = self.cluster_shape_mn;
        cfg.with_cluster([cm as u32, cn as u32, 1])
    }

    //  GPU device kernel
    /// GPU device kernel performing the batched GEMM computation (`kernel`).
    fn kernel<A: Numeric, B: Numeric, C: Numeric, Acc: Numeric>(
        &self,
        tma_atom_a: &CopyAtom<A>,
        m_a_mkl: &Tensor<Ident>,
        tma_atom_b: &CopyAtom<B>,
        m_b_nkl: &Tensor<Ident>,
        tma_atom_c: &CopyAtom<C>,
        m_c_mnl: &Tensor<Ident>,
        tiled_mma: TiledMma,
        cta_layout_mnk: &Layout,
        a_smem_layout_staged: &ComposedLayout,
        b_smem_layout_staged: &ComposedLayout,
        epi_smem_layout_staged: &ComposedLayout,
    ) {
        let warp_idx = make_warp_uniform(warp_idx());

        // /////////////////////////////////////////////////////////////////////
        //  Prefetch Tma desc
        // /////////////////////////////////////////////////////////////////////
        if_(warp_idx.eq(0), || {
            cpasync::prefetch_descriptor(tma_atom_a);
            cpasync::prefetch_descriptor(tma_atom_b);
        });

        // /////////////////////////////////////////////////////////////////////
        //  Get cta/warp/thread idx
        // /////////////////////////////////////////////////////////////////////
        let bidz = block_idx().z;
        let tidx = thread_idx().x;

        let (cidx, cidy) = (cluster_idx().x, cluster_idx().y);
        let (cdimx, cdimy) = (cluster_dim().x, cluster_dim().y);
        let cluster_id = cidx + cdimx * cidy;

        // CTA Swizzle to promote L2 data reuse
        let group_size_m = 8;
        let num_reg_cids = Int32::from(group_size_m) * (cdimx / group_size_m) * cdimy;
        let (cid_m, cid_n) = if_else_yield(
            cluster_id.lt(num_reg_cids),
            || {
                let s_shape = ((group_size_m, cdimx / group_size_m), cdimy);
                let s_stride = ((1, cdimy * group_size_m), group_size_m);
                let s_layout = make_layout(s_shape, s_stride);
                let c = s_layout.get_flat_coord(cluster_id % num_reg_cids);
                (leaf_i32(&c, 0), leaf_i32(&c, 1))
            },
            || {
                // Deal with the tail part
                let tail_size_m = cdimx % group_size_m;
                let tail_layout = make_layout((tail_size_m, cdimy), (1, tail_size_m));
                let tail_cid = cluster_id - num_reg_cids;
                let c = tail_layout.get_flat_coord(tail_cid);
                let cid_m = Int32::from(group_size_m) * (cdimx / group_size_m) + leaf_i32(&c, 0);
                (cid_m, leaf_i32(&c, 1))
            },
        );

        // Get the pid from cluster id
        let bidx_in_cluster = block_in_cluster_idx();
        let pid_m = cid_m * self.cluster_shape_mn.0 as i32 + bidx_in_cluster.x;
        let pid_n = cid_n * self.cluster_shape_mn.1 as i32 + bidx_in_cluster.y;

        let cta_rank_in_cluster = make_warp_uniform(block_idx_in_cluster());
        let cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster);

        // /////////////////////////////////////////////////////////////////////
        // Get mcast mask
        // /////////////////////////////////////////////////////////////////////
        let a_mcast_mask = make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk.clone(), 1);
        let b_mcast_mask = make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk.clone(), 0);

        // A multicast-less atom takes no mask (the example passes 0).
        let a_mcast_mask = self.is_a_mcast.then_some(a_mcast_mask);
        let b_mcast_mask = self.is_b_mcast.then_some(b_mcast_mask);
        let a_smem_layout = a_smem_layout_staged.slice((.., .., 0));
        let b_smem_layout = b_smem_layout_staged.slice((.., .., 0));
        let tma_copy_bytes = static_usize(size_in_bytes::<A>(&a_smem_layout) + size_in_bytes::<B>(&b_smem_layout));

        // /////////////////////////////////////////////////////////////////////
        //  Alloc and init AB full/empty + ACC full mbar (pipeline)
        // /////////////////////////////////////////////////////////////////////
        let a_elems = smem_extent(a_smem_layout_staged);
        let b_elems = smem_extent(b_smem_layout_staged);
        let reuses_a = epi_reuses_smem_a(A::DTYPE, C::DTYPE);
        let c_elems = if reuses_a { 0 } else { smem_extent(epi_smem_layout_staged) };
        let storage = SharedStorage::<A, B, C>::allocate_aligned(
            self.buffer_align_bytes,
            self.ab_stage,
            a_elems,
            b_elems,
            c_elems,
        );

        // Threads per warp group is 128 here, so each warp group has 4 warps.
        // Each warp will constribute to the arrive count with the number of mcast size
        let mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1;
        let num_warps = self.threads_per_cta / 32;
        let consumer_arrive_cnt = if self.consumers_in_threads { self.threads_per_cta } else { mcast_size * num_warps };
        let mainloop_pipeline_producer_group = CooperativeGroup::new(Agent::Thread, 1);
        let mainloop_pipeline_consumer_group = CooperativeGroup::new(Agent::Thread, consumer_arrive_cnt);
        let mainloop_pipeline = PipelineTmaAsync::create(
            self.ab_stage,
            mainloop_pipeline_producer_group,
            mainloop_pipeline_consumer_group,
            tma_copy_bytes,
            PipelineTmaAsyncOptions {
                barrier_storage: Some(storage.mainloop_pipeline_array_ptr.data_ptr().cast::<u64>()),
                cta_layout_vmnk: Some(Layout::col_major((1, self.cluster_shape_mn.0, self.cluster_shape_mn.1, 1))),
                tidx: Some(tidx),
                defer_sync: true,
                ..Default::default()
            },
        );

        //  Cluster arrive after barrier init
        let cluster_size = self.cluster_shape_mn.0 * self.cluster_shape_mn.1;
        let cluster_shape = Layout::col_major((self.cluster_shape_mn.0, self.cluster_shape_mn.1)).shape().clone();
        pipeline_init_arrive(Some(cluster_shape.clone()), true);

        // /////////////////////////////////////////////////////////////////////
        //  Generate smem tensor A/B
        // /////////////////////////////////////////////////////////////////////
        let s_a = storage.s_a.get_tensor_composed(a_smem_layout_staged.clone());
        let s_b = storage.s_b.get_tensor_composed(b_smem_layout_staged.clone());
        // C's staging tiles reuse A's buffer once the mainloop is done (the
        // example's `sC` is `sA`'s pointer recast with the epilogue's
        // layout), unless `stmatrix` would store into an 8-bit buffer: see
        // `epi_reuses_smem_a`.
        let s_c = if reuses_a {
            let s_c_ptr = recast_ptr::<C, A>(&s_a.ptr(), None);
            make_tensor(&s_c_ptr, epi_smem_layout_staged.clone())
        } else {
            storage.s_c.get_tensor_composed(epi_smem_layout_staged.clone())
        };

        // /////////////////////////////////////////////////////////////////////
        //  Local_tile partition global tensors
        // /////////////////////////////////////////////////////////////////////
        let (bm, bn, bk) = self.tile_shape_mnk;
        // (bM, bK, RestK)
        let g_a_mkl = local_tile(m_a_mkl, (bm, bk), (pid_m, .., bidz));
        // (bN, bK, RestK)
        let g_b_nkl = local_tile(m_b_nkl, (bn, bk), (pid_n, .., bidz));
        // (bM, bN)
        let g_c_mnl = local_tile(m_c_mnl, (bm, bn), (pid_m, pid_n, bidz));

        // /////////////////////////////////////////////////////////////////////
        //  Partition global tensor for TiledMMA_A/B/C
        // /////////////////////////////////////////////////////////////////////
        let warp_group_idx = make_warp_uniform(tidx / self.num_threads_per_warp_group as i32);
        let warp_group_thread_layout = make_layout(self.mma_warp_groups, self.num_threads_per_warp_group);
        let thr_mma = tiled_mma.get_slice(crd2idx(warp_group_idx, &warp_group_thread_layout));

        let t_cg_c = thr_mma.partition_c(&g_c_mnl);

        // /////////////////////////////////////////////////////////////////////
        //  Partition shared tensor for TMA load A/B
        // /////////////////////////////////////////////////////////////////////
        //  TMA load A partition_S/D
        let a_cta_layout = make_layout(self.cluster_shape_mn.1, 1);
        let a_cta_crd = cluster_coord_mnk.mode(1).clone();
        let s_a_for_tma_partition = s_a.group_modes(0, 2);
        let g_a_for_tma_partition = g_a_mkl.group_modes(0, 2);
        let (t_as_a, t_ag_a_mkl) = cpasync::tma_partition(
            tma_atom_a,
            a_cta_crd,
            &a_cta_layout,
            &s_a_for_tma_partition,
            &g_a_for_tma_partition,
        );

        // TMA load B partition_S/D
        let b_cta_layout = make_layout(self.cluster_shape_mn.0, 1);
        let b_cta_crd = cluster_coord_mnk.mode(0).clone();
        let s_b_for_tma_partition = s_b.group_modes(0, 2);
        let g_b_for_tma_partition = g_b_nkl.group_modes(0, 2);
        let (t_bs_b, t_bg_b_nkl) = cpasync::tma_partition(
            tma_atom_b,
            b_cta_crd,
            &b_cta_layout,
            &s_b_for_tma_partition,
            &g_b_for_tma_partition,
        );

        // /////////////////////////////////////////////////////////////////////
        //  Make fragments
        // /////////////////////////////////////////////////////////////////////
        let t_cs_a = thr_mma.partition_a(&s_a);
        let t_cs_b = thr_mma.partition_b(&s_b);
        let t_cr_a = tiled_mma.make_fragment_a(&t_cs_a);
        let t_cr_b = tiled_mma.make_fragment_b(&t_cs_b);

        let acc_shape = t_cg_c.shape().clone();
        let accumulators = tiled_mma.make_fragment_c::<Acc>(&acc_shape);

        // /////////////////////////////////////////////////////////////////////
        //  Cluster wait
        // /////////////////////////////////////////////////////////////////////
        // cluster wait for barrier init
        pipeline_init_wait(Some(cluster_shape));

        // /////////////////////////////////////////////////////////////////////
        //  Prefetch
        // /////////////////////////////////////////////////////////////////////
        let k_tile_cnt = Int32::from(size(&g_a_mkl, &[2]));
        let prefetch_k_tile_cnt = imax(imin(Int32::from(self.ab_stage as i32), k_tile_cnt), Int32::from(0));

        // One TMA load of A and B into the producer's stage.
        let tma_load = |state: &PipelineState| {
            // Slice to global/shared memref to current k_tile
            let t_ag_a_k = t_ag_a_mkl.slice((.., state.count()));
            let t_as_a_pipe = t_as_a.slice((.., state.index()));
            let t_bg_b_k = t_bg_b_nkl.slice((.., state.count()));
            let t_bs_b_pipe = t_bs_b.slice((.., state.index()));
            // TMA load A/B
            let bar = mainloop_pipeline.producer_get_barrier(state);
            let args = |mask: Option<Int16>| {
                let args = CopyArgs::new().tma_bar_ptr(&bar);
                match mask {
                    Some(mask) => args.mcast_mask(mask),
                    None => args,
                }
            };
            copy_with(tma_atom_a, &t_ag_a_k, &t_as_a_pipe, args(a_mcast_mask));
            copy_with(tma_atom_b, &t_bg_b_k, &t_bs_b_pipe, args(b_mcast_mask));
        };

        let mainloop_producer_state = make_pipeline_state(PipelineUserType::Producer, self.ab_stage);
        let mainloop_producer_state = if_else_yield(
            warp_idx.eq(0),
            || {
                // Prefetch TMA load
                for_range_carry(0, prefetch_k_tile_cnt, 1, mainloop_producer_state, |_prefetch_idx, mut state| {
                    // Wait for A/B buffers to be empty before loading into
                    // them. Also sets the transaction barrier for the A/B
                    // buffers
                    mainloop_pipeline.producer_acquire(&state, None);
                    tma_load(&state);
                    // Mainloop pipeline's producer commit is a NOP
                    mainloop_pipeline.producer_commit(&state);
                    state.advance();
                    state
                })
            },
            || mainloop_producer_state,
        );

        // /////////////////////////////////////////////////////////////////////
        //  Prologue MMAs
        // /////////////////////////////////////////////////////////////////////
        let k_pipe_mmas: usize = 1;

        let mut mainloop_consumer_read_state = make_pipeline_state(PipelineUserType::Consumer, self.ab_stage);
        let mainloop_consumer_release_state = make_pipeline_state(PipelineUserType::Consumer, self.ab_stage);

        let peek = |state: &PipelineState| {
            if_else_yield(
                state.count().lt(k_tile_cnt),
                || mainloop_pipeline.consumer_try_wait(state),
                || Boolean::constant(true),
            )
        };
        let mut peek_ab_full_status = peek(&mainloop_consumer_read_state);

        let mut tiled_mma = tiled_mma;
        tiled_mma.set(MmaField::ACCUMULATE, false);
        let num_k_blocks = static_usize(size(&t_cr_a, &[2]));
        // The k-blocks of the consumer's stage.
        let mmas = |tiled_mma: &mut TiledMma, accumulators: &Tensor<Acc>, state: &PipelineState, prologue: bool| {
            for k_block_idx in 0..num_k_blocks {
                let k_block_coord = (.., .., k_block_idx, state.index());
                let t_cr_a_1phase = t_cr_a.slice(k_block_coord);
                let t_cr_b_1phase = t_cr_b.slice(k_block_coord);
                gemm(tiled_mma, accumulators, &t_cr_a_1phase, &t_cr_b_1phase, accumulators);
                if prologue {
                    tiled_mma.set(MmaField::ACCUMULATE, true);
                }
            }
        };
        for _k_tile in 0..k_pipe_mmas {
            // Wait for A/B buffer to be ready
            mainloop_pipeline.consumer_wait(&mainloop_consumer_read_state, Some(peek_ab_full_status));

            warpgroup::fence();
            mmas(&mut tiled_mma, &accumulators, &mainloop_consumer_read_state, true);
            warpgroup::commit_group();
            mainloop_consumer_read_state.advance();
            peek_ab_full_status = peek(&mainloop_consumer_read_state);
        }

        // /////////////////////////////////////////////////////////////////////
        //  MAINLOOP
        // /////////////////////////////////////////////////////////////////////
        let epilogue_mma = tiled_mma.clone();
        let init = (
            accumulators,
            tiled_mma,
            mainloop_consumer_read_state,
            mainloop_consumer_release_state,
            mainloop_producer_state,
            peek_ab_full_status,
        );
        let (accumulators, _, _, _, _, _) = for_range_carry_with(
            k_pipe_mmas as i32,
            k_tile_cnt,
            1,
            LoopOptions::unroll(1),
            init,
            |_k_tile, (accumulators, mut tiled_mma, mut read_state, mut release_state, producer_state, peek_status)| {
                // Wait for TMA copies to complete
                mainloop_pipeline.consumer_wait(&read_state, Some(peek_status));
                // WGMMA
                warpgroup::fence();
                mmas(&mut tiled_mma, &accumulators, &read_state, false);
                warpgroup::commit_group();
                // Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
                warpgroup::wait_group(k_pipe_mmas as u32);

                mainloop_pipeline.consumer_release(&release_state);

                read_state.advance();
                release_state.advance();

                let peek_status = peek(&read_state);
                // TMA load
                let producer_state = if_else_yield(
                    warp_idx.eq(0) & producer_state.count().lt(k_tile_cnt),
                    || {
                        let mut producer_state = producer_state;
                        // Wait for A/B buffers to be empty before loading
                        // into them. Also sets the transaction barrier for
                        // the A/B buffers
                        mainloop_pipeline.producer_acquire(&producer_state, None);
                        tma_load(&producer_state);
                        // Mainloop pipeline's producer commit is a NOP
                        mainloop_pipeline.producer_commit(&producer_state);
                        producer_state.advance();
                        producer_state
                    },
                    || producer_state,
                );
                (accumulators, tiled_mma, read_state, release_state, producer_state, peek_status)
            },
        );

        // /////////////////////////////////////////////////////////////////////
        //  EPILOG
        // /////////////////////////////////////////////////////////////////////
        warpgroup::wait_group(0);

        if cluster_size > 1 {
            // Wait for all threads in the cluster to finish, avoid early release of smem
            cluster_arrive();
            cluster_wait();
        } else {
            // For cluster that has a single thread block, it might have more
            // than one warp groups. Wait for all warp groups in the thread
            // block to finish, because smem for tensor A in the mainloop is
            // reused in the epilogue.
            sync_threads();
        }

        self.epilogue::<C, Acc>(&epilogue_mma, &accumulators, &s_c, &g_c_mnl, tma_atom_c, warp_idx, tidx);
    }

    /// The epilogue of `kernel`: registers → shared memory (`stmatrix`) →
    /// global memory (TMA store), one `epi_tile` at a time, `epi_stage`
    /// staging buffers in flight.
    fn epilogue<C: Numeric, Acc: Numeric>(
        &self,
        tiled_mma: &TiledMma,
        accumulators: &Tensor<Acc>,
        s_c: &Tensor<C>,
        g_c_mnl: &Tensor<Ident>,
        tma_atom_c: &CopyAtom<C>,
        warp_idx: Int32,
        tidx: Int32,
    ) {
        // sm90_utils.sm90_get_smem_store_op(c_layout, elem_ty_d=c_dtype,
        // elem_ty_acc=acc_dtype): stmatrix for 16-bit C, else a universal copy.
        let copy_atom_r2s = sm90::get_smem_store_op::<C>(self.c_layout, self.acc_dtype);
        let c16 = C::DTYPE.bits() == 16;
        let stmatrix = CopyOp::StMatrix { num: 4, trans: self.c_layout.is_m_major_c() };

        // StMatrix8x8x16bOp(is_m_major_c, 4) of c_dtype (for C not of 16
        // bits, a 128-bit atom of the same per-thread width: see the module
        // docs).
        let copy_atom_c = make_copy_atom::<C>(if c16 { stmatrix } else { CopyOp::Universal { bits: 128 } });

        let tiled_copy_c_atom = make_tiled_copy_c_atom(&copy_atom_c, tiled_mma);

        let tiled_copy_r2s = make_tiled_copy_s(&copy_atom_r2s, &tiled_copy_c_atom);

        // (R2S, R2S_M, R2S_N, PIPE_D)
        let thr_copy_r2s = tiled_copy_r2s.get_slice(tidx);
        let t_rs_sd = thr_copy_r2s.partition_d(s_c);
        // (R2S, R2S_M, R2S_N)
        let t_rs_racc = tiled_copy_r2s.retile(accumulators);

        // Allocate D registers.
        let rd_shape = thr_copy_r2s.partition_s(s_c).shape().clone();
        let t_rs_rd_shape: Shape = Tree::Branch((0..3).map(|i| rd_shape.mode(i).clone()).collect());
        let t_rs_rd = make_fragment::<Acc>(t_rs_rd_shape.clone());
        let size_t_rs_rd = t_rs_rd.size();

        let sepi_for_tma_partition = s_c.group_modes(0, 2);
        let t_cg_c_for_tma_partition = zipped_divide(g_c_mnl, self.epi_tile);

        let (bsg_sd, bsg_gd) = cpasync::tma_partition(
            tma_atom_c,
            0,
            &make_layout(1, 1),
            &sepi_for_tma_partition,
            &t_cg_c_for_tma_partition,
        );

        let epi_tile_shape = t_cg_c_for_tma_partition.shape().mode(1).clone();
        let epi_tile_num = static_usize(size(&t_cg_c_for_tma_partition, &[1]));
        let epi_tile_n = epi_tile_shape.mode(1).size();
        let epi_tile_layout = make_layout(epi_tile_shape, (epi_tile_n, 1));
        let epi_buffers = static_usize(size(&t_rs_sd, &[3]));

        // Initialize tma store c_pipeline
        let c_producer_group = CooperativeGroup::new(Agent::Thread, self.threads_per_cta);
        let c_pipeline = PipelineTmaStore::create(self.epi_stage, c_producer_group);

        for epi_idx in 0..epi_tile_num {
            // Copy from accumulators to D registers
            for epi_v in 0..size_t_rs_rd {
                t_rs_rd.store(epi_v, t_rs_racc.load(epi_idx * size_t_rs_rd + epi_v));
            }

            // Type conversion
            let t_rs_rd_out = make_fragment::<C>(t_rs_rd_shape.clone());
            let acc_vec = t_rs_rd.load_ssa();
            t_rs_rd_out.store_ssa(&acc_vec.to::<C>());

            // Copy from D registers to shared memory
            let epi_buffer = epi_idx % epi_buffers;
            copy(&tiled_copy_r2s, &t_rs_rd_out, &t_rs_sd.slice((.., .., .., epi_buffer)));

            fence_proxy_with(
                FenceProxyKind::AsyncShared,
                FenceProxyOptions { space: Some(SharedSpace::Cta), ..Default::default() },
            );
            // barrier for sync
            pipeline::sync(1);

            let gmem_coord = epi_tile_layout.get_hier_coord(epi_idx);
            // Copy from shared memory to global memory
            if_(warp_idx.eq(0), || {
                copy(tma_atom_c, &bsg_sd.slice((.., epi_buffer)), &bsg_gd.slice((.., gmem_coord.clone())));
                c_pipeline.producer_commit();
                c_pipeline.producer_acquire();
            });

            pipeline::sync(1);
        }

        if_(warp_idx.eq(0), || c_pipeline.producer_tail());
    }

    /// Computes the number of stages for A/B/C operands based on heuristics
    /// (`_compute_stages`): `(A/B operand stages, epilogue stages)`. The
    /// epilogue's C type and tile are an addition: when C cannot reuse A's
    /// buffer ([`epi_reuses_smem_a`]), its stages take their own bytes.
    pub fn compute_stages(
        tile_shape_mnk: (usize, usize, usize),
        a_dtype: DType,
        b_dtype: DType,
        (c_dtype, epi_tile): (DType, (usize, usize)),
        smem_capacity: usize,
        occupancy: usize,
    ) -> (usize, usize) {
        let epi_stage = 4;
        // epi_smem will reuse smem ab.
        let epi_bytes = if epi_reuses_smem_a(a_dtype, c_dtype) {
            0
        } else {
            epi_tile.0 * epi_tile.1 * epi_stage * c_dtype.bits() / 8
        };

        let (m, n, k) = tile_shape_mnk;
        let ab_bytes_per_stage = m * k * a_dtype.bits() / 8 + n * k * b_dtype.bits() / 8;
        let mbar_helpers_bytes = 1024;

        let ab_stage = (smem_capacity / occupancy - mbar_helpers_bytes - epi_bytes) / ab_bytes_per_stage;
        (ab_stage, epi_stage)
    }

    /// Create shared memory layouts for A, B, and C tensors
    /// (`_make_smem_layouts`).
    pub fn make_smem_layouts(
        tile_shape_mnk: (usize, usize, usize),
        epi_tile: (usize, usize),
        a_dtype: DType,
        a_layout: LayoutEnum,
        b_dtype: DType,
        b_layout: LayoutEnum,
        ab_stage: usize,
        c_dtype: DType,
        c_layout: LayoutEnum,
        epi_stage: usize,
    ) -> (ComposedLayout, ComposedLayout, ComposedLayout) {
        let a_smem_layout_staged = sm90::make_smem_layout_a(a_layout, tile_shape_mnk, a_dtype, ab_stage);
        let b_smem_layout_staged = sm90::make_smem_layout_b(b_layout, tile_shape_mnk, b_dtype, ab_stage);
        let epi_smem_layout_staged = sm90::make_smem_layout_epi(c_dtype, c_layout, epi_tile, epi_stage, None, None);
        (a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged)
    }

    /// Compute grid shape for the output tensor C (`_compute_grid`).
    pub fn compute_grid(
        c_shape_mnl: (usize, usize, usize),
        tile_shape_mnk: (usize, usize, usize),
        cluster_shape_mn: (usize, usize),
    ) -> [u32; 3] {
        let (m, n, l) = c_shape_mnl;
        let tiles = [m.div_ceil(tile_shape_mnk.0), n.div_ceil(tile_shape_mnk.1), l];
        let cluster_shape_mnl = [cluster_shape_mn.0, cluster_shape_mn.1, 1];
        let clusters: Vec<usize> = tiles.iter().zip(cluster_shape_mnl).map(|(&t, c)| t.div_ceil(c)).collect();
        [0, 1, 2].map(|i| (clusters[i] * cluster_shape_mnl[i]) as u32)
    }

    /// Create TMA atoms and tensors for C tensor storage
    /// (`_make_tma_store_atoms_and_tensors`).
    pub fn make_tma_store_atoms_and_tensors<C: Numeric>(
        tensor_c: &Tensor<C>,
        epi_smem_layout_staged: &ComposedLayout,
        epi_tile: (usize, usize),
    ) -> (CopyAtom<C>, Tensor<Ident>) {
        let epi_smem_layout = epi_smem_layout_staged.slice((.., .., 0));
        cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, tensor_c, epi_smem_layout, epi_tile, 1).into()
    }

    /// Create TMA atoms and tensors for input tensors
    /// (`_make_tma_atoms_and_tensors`).
    pub fn make_tma_atoms_and_tensors<T: Numeric>(
        tensor: &Tensor<T>,
        smem_layout_staged: &ComposedLayout,
        smem_tile: (usize, usize),
        mcast_dim: usize,
    ) -> (CopyAtom<T>, Tensor<Ident>) {
        let op: CopyOp = if mcast_dim == 1 {
            cpasync::CopyBulkTensorTileG2SOp::default().into()
        } else {
            cpasync::CopyBulkTensorTileG2SMulticastOp::default().into()
        };

        let smem_layout = smem_layout_staged.slice((.., .., 0));
        cpasync::make_tiled_tma_atom(op, tensor, smem_layout, smem_tile, mcast_dim).into()
    }

    /// Check if the dtypes are valid (`is_valid_dtypes`), with BFloat16
    /// added (see the module docs).
    pub fn is_valid_dtypes(
        a_dtype: DType,
        b_dtype: DType,
        acc_dtype: DType,
        c_dtype: DType,
        a_major: char,
        b_major: char,
    ) -> bool {
        use DType::*;
        let mut is_valid = true;
        let valid_ab_dtypes = [F16, BF16, F8E4M3FN, F8E5M2, U8, I8];
        if !valid_ab_dtypes.contains(&a_dtype) || !valid_ab_dtypes.contains(&b_dtype) {
            return false;
        }
        // make sure a_dtype == b_dtype for Float16
        if a_dtype.bits() == 16 && a_dtype != b_dtype {
            is_valid = false;
        }
        if a_dtype.bits() != b_dtype.bits() {
            is_valid = false;
        }
        let is_int = |d: DType| matches!(d, U8 | I8);
        if is_int(a_dtype) != is_int(b_dtype) {
            is_valid = false;
        }
        // for 8-bit types, this implementation only supports k-major layout
        if (a_dtype.bits() == 8 && a_major != 'k') || (b_dtype.bits() == 8 && b_major != 'k') {
            is_valid = false;
        }
        // Define compatibility mapping between accumulator type and AB type
        let acc_ab_compatibility: &[DType] = match acc_dtype {
            F32 => &[F16, BF16, F8E4M3FN, F8E5M2],
            F16 => &[F16, F8E4M3FN, F8E5M2],
            I32 => &[U8, I8],
            _ => return false,
        };
        if !acc_ab_compatibility.contains(&a_dtype) {
            is_valid = false;
        }
        // Define compatibility mapping between accumulator type and C type
        let acc_c_compatibility: &[DType] = match acc_dtype {
            F32 | F16 => &[F32, F16, BF16, F8E4M3FN, F8E5M2],
            _ => &[F32, F16, I32, I8, U8],
        };
        if !acc_c_compatibility.contains(&c_dtype) {
            is_valid = false;
        }
        is_valid
    }

    /// Check if the tensor alignment is valid (`is_valid_tensor_alignment`):
    /// the contiguous extent of each tensor is a whole number of 16-byte
    /// vectors.
    pub fn is_valid_tensor_alignment(
        m: usize,
        n: usize,
        k: usize,
        l: usize,
        ab_dtype: DType,
        c_dtype: DType,
        majors: Majors,
    ) -> bool {
        Self::can_implement(m, n, k, l, ab_dtype, ab_dtype, c_dtype, majors).is_ok()
    }

    /// Whether the kernel can run a problem (`can_implement`'s alignment
    /// check): `check_gemm_tma_alignment` over A, B and C, the contiguous
    /// extent of each a whole number of TMA's 16-byte units. The error names
    /// the tensor and its extent.
    pub fn can_implement(
        m: usize,
        n: usize,
        k: usize,
        _l: usize,
        a_dtype: DType,
        b_dtype: DType,
        c_dtype: DType,
        majors: Majors,
    ) -> Result<(), CantImplement> {
        let major = |c: char| c.to_string().parse::<GemmMajor>().expect("a major mode");
        check_gemm_tma_alignment(
            m,
            n,
            k,
            a_dtype,
            b_dtype,
            Some(c_dtype),
            major(majors.a),
            major(majors.b),
            Some(major(majors.c)),
            "C",
        )
    }
}

/// Whether the epilogue's C tiles can live in A's shared memory, as the
/// example has them. cute-rs's `stmatrix` (16-bit C) must store into a
/// buffer of 16-bit elements, so with 8-bit A and 16-bit C the C tiles get
/// a buffer of their own (and the A/B stages make room for it).
pub fn epi_reuses_smem_a(a_dtype: DType, c_dtype: DType) -> bool {
    !(c_dtype.bits() == 16 && a_dtype.bits() != 16)
}

/// The elements of shared memory a staged layout spans: its cosize in whole
/// swizzle periods (what `make_smem_tensor` allocates).
fn smem_extent(layout: &ComposedLayout) -> usize {
    layout.cosize().next_multiple_of(layout.swizzle().map_or(1, |sw| sw.period()))
}

/// Mode `i` of a flat runtime coordinate as an Int32.
fn leaf_i32(c: &Coord, i: usize) -> Int32 {
    Int32::from(c.mode(i).as_leaf().expect("a flat coordinate").clone())
}

/// A size known when the kernel is built.
fn static_usize(e: IndexExpr) -> usize {
    e.as_static().expect("a static size")
}

/// The smaller of two Int32s (`cutlass.min`).
fn imin(a: Int32, b: Int32) -> Int32 {
    a.min(b)
}

/// The larger of two Int32s (`cutlass.max`).
fn imax(a: Int32, b: Int32) -> Int32 {
    a.max(b)
}

// /////////////////////////////////////////////////////////////////////////////
//  Host data, reference and run
// /////////////////////////////////////////////////////////////////////////////

/// What the inputs hold: small integers (`run`'s random integers, -2 to 2,
/// or 0 to 4 for unsigned types), whose products and sums every type here
/// represents exactly, or fractions (multiples of 1/64 in [-1, 1)), whose
/// sums round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Data {
    Integers,
    Fractions,
}

/// One problem: `m x n x k`, `l` batches, and the inputs.
#[derive(Clone, Copy, Debug)]
pub struct Problem {
    pub mnkl: (usize, usize, usize, usize),
    pub data: Data,
    pub seed: u64,
}

impl Problem {
    pub fn new(m: usize, n: usize, k: usize, l: usize) -> Problem {
        Problem { mnkl: (m, n, k, l), data: Data::Integers, seed: 1111 }
    }
}

/// Deterministic values for a tensor of `dtype` (`create_and_permute_tensor`'s
/// random initialisation).
fn values(len: usize, dtype: DType, data: Data, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let unsigned = dtype == DType::U8;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match data {
                Data::Integers if unsigned => (state >> 33) as f32 % 5.0,
                Data::Integers => ((state >> 33) % 5) as f32 - 2.0,
                Data::Fractions => ((state >> 33) % 128) as f32 / 64.0 - 1.0,
            }
        })
        .collect()
}

/// A buffer of `values` (already representable) as `dtype`.
pub fn buffer(dtype: DType, values: &[f32]) -> KernelArg {
    match dtype {
        DType::F16 => KernelArg::f16_from_f32(values),
        DType::BF16 => KernelArg::buffer(&values.iter().map(|&x| BF16::from_f32(x)).collect::<Vec<_>>()),
        DType::F8E4M3FN => KernelArg::fp8_from_f32::<F8E4M3FN>(values),
        DType::F8E5M2 => KernelArg::fp8_from_f32::<F8E5M2>(values),
        DType::I8 => KernelArg::buffer(&values.iter().map(|&x| x as i8).collect::<Vec<_>>()),
        DType::U8 => KernelArg::buffer(&values.iter().map(|&x| x as u8).collect::<Vec<_>>()),
        DType::I32 => KernelArg::i32(values.iter().map(|&x| x as i32).collect()),
        DType::F32 => KernelArg::f32(values.to_vec()),
        other => panic!("hopper_gemm: no buffer of {other}"),
    }
}

/// The values of a buffer of `dtype` as f64.
fn read(dtype: DType, arg: &KernelArg) -> Vec<f64> {
    match dtype {
        DType::I32 => arg.to_vec::<i32>().into_iter().map(f64::from).collect(),
        DType::I8 => arg.to_vec::<i8>().into_iter().map(f64::from).collect(),
        DType::U8 => arg.to_vec::<u8>().into_iter().map(f64::from).collect(),
        _ => arg.to_f32().into_iter().map(f64::from).collect(),
    }
}

/// The element offset of `(i, j, b)` in an `(s0, s1, l)` tensor, mode 0 or 1
/// contiguous.
fn offset(i: usize, j: usize, b: usize, s0: usize, s1: usize, mode0_major: bool) -> usize {
    b * s0 * s1 + if mode0_major { i + j * s0 } else { i * s1 + j }
}

/// The arguments of the kernel of `gemm` for `problem`: `m`, `n`, `k`, `l`,
/// then A and B with values from [`values`] and C zeroed, laid out in the
/// configured majors (`create_and_permute_tensor`).
pub fn inputs(gemm: &HopperWgmmaGemmKernel, problem: &Problem) -> Vec<KernelArg> {
    let (m, n, k, l) = problem.mnkl;
    let (a, b) = host_ab(gemm, problem);
    vec![
        KernelArg::ScalarI32(m as i32),
        KernelArg::ScalarI32(n as i32),
        KernelArg::ScalarI32(k as i32),
        KernelArg::ScalarI32(l as i32),
        buffer(gemm.a_dtype, &a),
        buffer(gemm.b_dtype, &b),
        buffer(gemm.c_dtype, &vec![0.0; m * n * l]),
    ]
}

/// The host values of A and B, in memory order.
fn host_ab(gemm: &HopperWgmmaGemmKernel, problem: &Problem) -> (Vec<f32>, Vec<f32>) {
    let (m, n, k, l) = problem.mnkl;
    // As the device holds them (fractions round to the input type).
    let held = |d: DType, len: usize, seed: u64| -> Vec<f32> {
        read(d, &buffer(d, &values(len, d, problem.data, seed))).into_iter().map(|x| x as f32).collect()
    };
    (held(gemm.a_dtype, m * k * l, problem.seed), held(gemm.b_dtype, n * k * l, problem.seed + 1))
}

/// `einsum("mkl,nkl->mnl", a, b)` in f64 (`utils::gemm::reference_gemm_values`
/// over the operands in logical order), in C's memory order.
pub fn reference(gemm: &HopperWgmmaGemmKernel, problem: &Problem) -> Vec<f64> {
    let (m, n, k, l) = problem.mnkl;
    let (a, b) = host_ab(gemm, problem);
    let (a_m, b_n, c_m) = (gemm.a_layout.is_m_major_a(), gemm.b_layout.is_n_major_b(), gemm.c_layout.is_m_major_c());
    // Memory order to logical order, `mn + MN * (k + K * l)`.
    let logical = |mem: &[f32], mn: usize, mode0_major: bool| -> Vec<f32> {
        let mut v = vec![0.0; mem.len()];
        for bt in 0..l {
            for kk in 0..k {
                for i in 0..mn {
                    v[i + mn * (kk + k * bt)] = mem[offset(i, kk, bt, mn, k, mode0_major)];
                }
            }
        }
        v
    };
    let c = reference_gemm_values(&logical(&a, m, a_m), &logical(&b, n, b_n), (m, n, k, l));
    let major = if c_m { OperandMajorMode::MN } else { OperandMajorMode::K };
    to_memory_order(&c, (m, n, l), major)
}

/// Check C in `args` against [`reference`]: on integers, bit for bit after
/// rounding the exact sum to C's type (integer accumulators wrap, which
/// these sizes never reach); on fractions within `|got - want| <= 0.02 +
/// 0.01 |want|`, which covers C's rounding (half an ulp: 2^-11 relative for
/// Float16, 2^-8 for BFloat16, 2^-4 for E4M3 and 2^-3 for E5M2 outputs aside),
/// plus [`f16_accumulator_rounding`] for a Float16 accumulator.
pub fn check(gemm: &HopperWgmmaGemmKernel, problem: &Problem, args: &[KernelArg]) {
    let want = reference(gemm, problem);
    let got = read(gemm.c_dtype, &args[6]);
    let name = gemm.name();
    let acc_rounding = if gemm.acc_dtype == DType::F16 { f16_accumulator_rounding(&want, problem.mnkl.2) } else { 0.0 };
    let mut zeros = 0;
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        match problem.data {
            Data::Integers => {
                let rounded = read(gemm.c_dtype, &buffer(gemm.c_dtype, &[w as f32]))[0];
                assert_eq!(g, rounded, "{name} {:?}: C element {i}: got {g}, want {w}", problem.mnkl);
            }
            Data::Fractions => {
                let tol = if gemm.c_dtype.bits() == 8 { 0.02 + 0.13 * w.abs() } else { 0.02 + 0.01 * w.abs() };
                let tol = tol + acc_rounding;
                assert!((g - w).abs() <= tol, "{name} {:?}: C element {i}: got {g}, want {w}", problem.mnkl);
            }
        }
        zeros += usize::from(w == 0.0);
    }
    // C starts out 0: inputs that leave most of it 0 would test little.
    assert!(zeros * 4 < want.len(), "{name}: {zeros} of the {} elements of C are 0", want.len());
}

/// What a Float16 accumulator's rounding adds to the difference from the f64 reference over
/// `k` products. The accumulator is rounded at every MMA (16 products at a time), by up to
/// half a Float16 step at the running sum's size, and the roundings add up like a random
/// walk: `sqrt(k / 16)` steps, the step taken at the largest value of C. (Measured on an H100
/// at K = 1024 with values up to 50: 0.13 at most, where this gives 0.25. The simulator
/// rounds as the device does, bit for bit; it is the reference that does not round.)
pub fn f16_accumulator_rounding(want: &[f64], k: usize) -> f64 {
    let largest = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    if largest == 0.0 {
        return 0.0;
    }
    let step = 2f64.powi(largest.log2().floor() as i32 - 10);
    (k as f64 / 16.0).sqrt() * step
}

/// Simulate `kernel` (built from `gemm`) on `problem` and check it.
pub fn run(gemm: &HopperWgmmaGemmKernel, kernel: &Kernel, problem: &Problem) -> SimReport {
    let (m, n, k, l) = problem.mnkl;
    let majors = gemm_majors(gemm);
    assert!(
        HopperWgmmaGemmKernel::is_valid_dtypes(
            gemm.a_dtype,
            gemm.b_dtype,
            gemm.acc_dtype,
            gemm.c_dtype,
            majors.a,
            majors.b
        ),
        "unsupported combination of types and majors: A {}, B {}, Acc {}, C {}, {majors:?}",
        gemm.a_dtype,
        gemm.b_dtype,
        gemm.acc_dtype,
        gemm.c_dtype
    );
    assert!(
        HopperWgmmaGemmKernel::is_valid_tensor_alignment(m, n, k, l, gemm.a_dtype, gemm.c_dtype, majors),
        "the contiguous dimension of A/B/C tensors is not 16 bytes aligned"
    );
    let mut args = inputs(gemm, problem);
    let report = kernel
        .simulate(gemm.launch(m, n, l), &mut args)
        .unwrap_or_else(|e| panic!("{} {:?}: {e}", kernel.name(), problem.mnkl));
    check(gemm, problem, &args);
    report
}

/// The majors of a configured kernel.
pub fn gemm_majors(gemm: &HopperWgmmaGemmKernel) -> Majors {
    Majors {
        a: if gemm.a_layout.is_k_major_a() { 'k' } else { 'm' },
        b: if gemm.b_layout.is_k_major_b() { 'k' } else { 'n' },
        c: if gemm.c_layout.is_n_major_c() { 'n' } else { 'm' },
    }
}

fn main() {
    println!("Running Hopper Dense GEMM (in the simulator) with:");
    let (m, n, k, l) = (256, 256, 256, 1);
    let tile_shape_mn = (128, 128);
    let cluster_shape_mn = (2, 1);
    println!("mnkl: ({m}, {n}, {k}, {l})");
    println!("A dtype: Float16, B dtype: Float16, C dtype: Float16, Acc dtype: Float32");
    println!("Matrix majors - A: k, B: k, C: n");
    println!("Tile Shape: {tile_shape_mn:?}, Cluster Shape: {cluster_shape_mn:?}");
    let gemm = HopperWgmmaGemmKernel::new(DType::F32, tile_shape_mn, cluster_shape_mn).setup(
        DType::F16,
        DType::F16,
        DType::F16,
        Majors::KKN,
    );
    println!(
        "tile {:?}, {} A/B stages, epilogue tile {:?} x {} stages, {} threads",
        gemm.tile_shape_mnk, gemm.ab_stage, gemm.epi_tile, gemm.epi_stage, gemm.threads_per_cta
    );
    let kernel = gemm.build();
    let report = run(&gemm, &kernel, &Problem::new(m, n, k, l));
    println!("{}: correct\n{report}\n", kernel.name());

    // Partial tiles, the other majors and a cluster along N.
    let gemm = HopperWgmmaGemmKernel::new(DType::F32, (128, 64), (1, 2)).setup(
        DType::F16,
        DType::F16,
        DType::F16,
        Majors::new('m', 'n', 'm'),
    );
    let kernel = gemm.build();
    run(&gemm, &kernel, &Problem::new(200, 136, 136, 2));
    println!("{} on 200x136x136x2 (partial tiles): correct", kernel.name());

    match Toolchain::discover() {
        Some(tc) => match kernel.compile_ptx(&tc, Arch::Sm90a) {
            Ok(ptx) => {
                let count = |pat: &str| ptx.lines().filter(|l| l.contains(pat)).count();
                println!(
                    "PTX (sm_90a): {} lines; {} wgmma.mma_async, {} multicast TMA loads, {} stmatrix, {} TMA stores",
                    ptx.lines().count(),
                    count("wgmma.mma_async"),
                    count(".multicast::cluster"),
                    count("stmatrix"),
                    count("cp.async.bulk.tensor.3d.global.shared"),
                );
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}
