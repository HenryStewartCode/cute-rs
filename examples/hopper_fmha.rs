// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/hopper/kernel/attention/fmha.py
//   (`HopperFusedMultiHeadAttentionForward`: its configuration, `__call__`,
//   `kernel`, `compute`, `softmax_step`, `tail`, `make_acc_into_op`,
//   `layout_acc_mn`, `reduction_target_n`, `gemm_zero_acc`, the pipeline
//   set-up helpers, the tensor set-up of `run` and the comments) and
//   examples/python/CuTeDSL/helpers/fmha_helpers.py (`FmhaStaticTileScheduler`,
//   `MaskEnum`, `FusedMask`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A fused multi-head attention forward pass (`O = softmax(Q K^T) V`) for
//! NVIDIA Hopper (SM90), ported from the CuTe DSL example
//! `hopper/kernel/attention/fmha.py`.
//!
//! Q, O are `(batch, seqlen_q, heads, head_dim)` and K, V `(batch, seqlen_k,
//! heads_k, head_dim)` row-major Float16 or BFloat16 tensors (`heads` a
//! multiple of `heads_k`: grouped-query attention), with Float32
//! accumulators; the log-sum-exp of each row comes out as `lse`
//! (`(batch, heads, seqlen_q)`).
//!
//! The kernel is warp-specialised, 3 warpgroups of 128 threads:
//!
//! 1. warpgroup 0 (registers lowered with `setmaxregister_decrease`) loads Q
//!    (two tiles of 64 rows), K and V tiles with TMA, from one warp, through
//!    two `PipelineTmaAsync` rings (Q: 2 stages, K and V: 5 stages that
//!    alternate K tile, V tile);
//! 2. warpgroups 1 and 2 (registers raised) each own 64 rows of the CTA's
//!    128: `S = Q K^T` with A and B from shared memory, an online softmax on
//!    the accumulator registers (running max and sum per row, `exp2`, the
//!    mask applied to the tiles that need it), then `O += P V` with `P`
//!    converted to Float16/BFloat16 **in registers** and fed to the second
//!    warpgroup MMA as its A operand (`OperandSource::RMEM`), and V from
//!    shared memory. A `PipelineOrder` between the two warpgroups
//!    (ping-pong) staggers their `Q K^T` MMAs so one's softmax overlaps the
//!    other's MMA; consumers take the V stage's wait token from a
//!    `try_wait` issued before the softmax;
//! 3. the epilogue of each MMA warpgroup normalises `O` and `lse`, stores
//!    `O` registers → shared memory with `stmatrix` and shared → global with
//!    TMA stores paced by `PipelineTmaStore`, on named barriers of the
//!    warpgroup (`pipeline::arrive_and_wait`, `barrier_id` the warpgroup
//!    index).
//!
//! Masks (`MaskEnum`): the residual mask for a K length that is not a
//! multiple of the tile, and the window mask (causal is `window_right = 0`;
//! `WindowMaskInference` aligns the end of Q with the end of K).
//! Scheduling is one CTA per `(M tile, head, batch)` or persistent
//! (`FmhaStaticTileScheduler`: CTAs stride over the linear tile index).
//!
//! cute-rs has no SM90 GPU to run it on: the tests check it in the
//! simulator against a plain Rust attention in f64 and assemble its PTX with
//! `ptxas -arch=sm_90a`.
//!
//! Differences from the example:
//!
//! * The example swaps in a spin-limited wait for its mbarrier waits; that
//!   optimisation is left out, and the library's pipeline waits are used.
//! * Only 16-bit inputs (Float16, BFloat16), Float32 accumulators and the
//!   `RESIDUAL_MASK`, `WINDOW_MASK` and `WINDOW_MASK_INFERENCE` masks are
//!   ported: FP8 (`make_acc_into_op`'s 8-bit shuffle exchange) and the
//!   backward-pass masks are left out.
//! * The accumulator's row/column view (`layout_acc_mn`,
//!   `reduction_target_n`) is the warpgroup C layout's own:
//!   `((4, 8, 4), (2, 2, N / 8)) : ((128, 1, 16), (64, 8, 512))`, so a
//!   thread's value `v0 + 2 v1 + 4 v2` is row `v1` (of 2) and column
//!   `v0 + 8 v2`, and a row is reduced over 4 adjacent lanes.
//! * `lse` is a `(batch, heads, seqlen_q)` buffer, and Q, K, V, O are dense
//!   `(batch, seq, heads, head_dim)` tensors, the layout of the example's
//!   permuted torch tensors.
//!
//! Run with `cargo run --release --example hopper_fmha` (on the simulator).

use cute::cpasync;
use cute::cute_struct;
use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineOrder, PipelineOrderOptions, PipelineTmaAsync, PipelineTmaAsyncOptions,
    PipelineTmaStore, pipeline_init_arrive, pipeline_init_wait,
};
use cute::prelude::*;
use cute::utils::sm90;
use cute::warpgroup::{self, OperandSource};

cute_struct! {
    /// The kernel's shared memory (`SharedStorage`): the barriers of the Q
    /// and K/V rings (full then empty) and of the warpgroup order, the
    /// persistent kernel's O staging, then Q's and K/V's staged tiles, each
    /// aligned to 1024 bytes. V's tiles share K's buffer.
    pub struct SharedStorage<T, O>(
        q_stage: usize,
        kv_stage: usize,
        num_wg: usize,
        o_elems: usize,
        q_elems: usize,
        k_elems: usize
    ) {
        load_q_mbar_ptr: MemRange<u64>[2 * q_stage],
        load_kv_mbar_ptr: MemRange<u64>[2 * kv_stage],
        math_wg_order_barrier: MemRange<u64>[num_wg],
        #[align(1024)]
        s_o: MemRange<O>[o_elems],
        #[align(1024)]
        s_q: MemRange<T>[q_elems],
        #[align(1024)]
        s_k: MemRange<T>[k_elems],
    }
}

// /////////////////////////////////////////////////////////////////////////////
//  Masks (`fmha_helpers.MaskEnum`, `FusedMask`)
// /////////////////////////////////////////////////////////////////////////////

/// The mask types of the forward pass (`MaskEnum`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskEnum {
    /// Residual mask for a K length that is not a multiple of the tile.
    ResidualMask,
    /// Window mask (also causal, and no mask when no window is given).
    WindowMask,
    /// Window mask with the end of Q aligned with the end of K.
    WindowMaskInference,
}

/// The floor quotient of `a` by a positive constant.
fn floor_div(a: Int32, b: i32) -> Int32 {
    let q = a / b;
    let r = a % b;
    select(r.lt(0), q - 1, q)
}

/// `cute.ceil_div(a, b)` of an Int32 by a positive constant.
fn ceil_div_i32(a: Int32, b: i32) -> Int32 {
    floor_div(a + (b - 1), b)
}

/// The mask logic of one problem (`FusedMask`): which K tiles a CTA row
/// visits, how many of them need the mask, and the mask itself. Window
/// sizes are `Some` when the kernel was built with that side of the window.
#[derive(Clone, Copy)]
pub struct FusedMask {
    pub mask_type: MaskEnum,
    pub tile_m: i32,
    pub tile_n: i32,
    pub seqlen_q: Int32,
    pub seqlen_k: Int32,
    pub window_left: Option<Int32>,
    pub window_right: Option<Int32>,
}

impl FusedMask {
    fn is_window(&self) -> bool {
        self.mask_type != MaskEnum::ResidualMask
    }

    /// `seqlen_k - seqlen_q` for the inference mask, else 0.
    fn offset(&self) -> Int32 {
        if self.mask_type == MaskEnum::WindowMaskInference { self.seqlen_k - self.seqlen_q } else { Int32::from(0) }
    }

    /// The first K tile of the M tile `blk_m` (`get_trip_start`).
    pub fn get_trip_start(&self, blk_m: Int32) -> Int32 {
        let mut result = Int32::from(0);
        if let (true, Some(left)) = (self.is_window(), self.window_left) {
            let min_idx_q = blk_m * self.tile_m;
            let idx_k = min_idx_q + self.offset() - left;
            result = floor_div(idx_k, self.tile_n).max(result);
        }
        result
    }

    /// How many K tiles the M tile `blk_m` visits (`get_trip_count`).
    pub fn get_trip_count(&self, blk_m: Int32) -> Int32 {
        let mut result = ceil_div_i32(self.seqlen_k, self.tile_n);
        if let (true, Some(right)) = (self.is_window(), self.window_right) {
            let max_idx_q = (blk_m + 1) * self.tile_m;
            let idx_k = max_idx_q + self.offset() + right;
            let tmp_blocks_k = ceil_div_i32(idx_k, self.tile_n);
            result = result.min(tmp_blocks_k);
        }
        result - self.get_trip_start(blk_m)
    }

    /// The first and last K tile of the leading mask (`get_leading_mask_id`).
    fn leading_mask_id(&self, blk_m: Int32) -> (Int32, Int32) {
        let begin = self.get_trip_start(blk_m);
        let trip_count = self.get_trip_count(blk_m);
        let end = match self.window_left {
            Some(left) => {
                let min_idx_q = (blk_m + 1) * self.tile_m + self.offset() - left;
                (ceil_div_i32(min_idx_q, self.tile_n) - 1).min(trip_count + begin - 1)
            }
            None => begin - 1,
        };
        (begin, end)
    }

    /// The first and last K tile of the trailing mask (`get_trailing_mask_id`).
    fn trailing_mask_id(&self, blk_m: Int32) -> (Int32, Int32) {
        let trip_start = self.get_trip_start(blk_m);
        let trip_count = self.get_trip_count(blk_m);
        let last = trip_count + trip_start - 1;
        match self.window_right {
            Some(right) => {
                let min_idx_q = blk_m * self.tile_m + self.offset() + right;
                (floor_div(min_idx_q, self.tile_n).min(last), last)
            }
            // The last tile is always masked.
            None => (last, last),
        }
    }

    /// The masked tiles before the unmasked ones (`get_masked_leading_count`).
    pub fn get_masked_leading_count(&self, blk_m: Int32) -> Int32 {
        if self.is_window() && (self.window_left.is_some() || self.window_right.is_some()) {
            let (begin, end) = self.leading_mask_id(blk_m);
            (end - begin + 1).max(0)
        } else {
            Int32::from(0)
        }
    }

    /// The masked tiles after the unmasked ones, plus `rem_count`
    /// (`get_masked_trailing_count`).
    pub fn get_masked_trailing_count(&self, blk_m: Int32, rem_count: Int32) -> Int32 {
        let result = if self.is_window() {
            if self.window_left.is_some() || self.window_right.is_some() {
                let (tb, te) = self.trailing_mask_id(blk_m);
                let (_, le) = self.leading_mask_id(blk_m);
                select(tb.le(le), (te - le).max(0), (te - tb + 1).max(0))
            } else {
                Int32::from(0)
            }
        } else {
            select((self.seqlen_k % self.tile_n).ne(0), Int32::from(1), Int32::from(0))
        };
        result + rem_count
    }

    /// The tiles that need no mask (`get_unmasked_trip_count`).
    pub fn get_unmasked_trip_count(&self, blk_m: Int32) -> Int32 {
        self.get_trip_count(blk_m)
            - self.get_masked_leading_count(blk_m)
            - self.get_masked_trailing_count(blk_m, Int32::from(0))
    }

    /// Whether the score at `(index_q, index_k)` is masked out (`apply_mask`
    /// for one element).
    pub fn is_masked(&self, index_q: Int32, index_k: Int32) -> Boolean {
        let offset = self.offset();
        let residual = index_k.ge(self.seqlen_k) | index_q.ge(self.seqlen_q);
        let mut masked = Boolean::constant(false);
        if self.window_left.is_some() || self.window_right.is_some() {
            masked = match (self.window_left, self.window_right) {
                (None, Some(right)) => (index_q + offset + right).lt(index_k),
                (Some(left), None) => (index_q + offset - left).gt(index_k),
                (Some(left), Some(right)) => {
                    let max_k = (index_q + offset + right).min(self.seqlen_k);
                    let min_k = (index_q + offset - left).max(0);
                    index_k.gt(max_k) | index_k.lt(min_k)
                }
                (None, None) => unreachable!(),
            };
            masked = masked | residual;
        }
        if self.mask_type == MaskEnum::ResidualMask {
            masked = masked | residual;
        }
        masked
    }
}

// /////////////////////////////////////////////////////////////////////////////
//  Configuration
// /////////////////////////////////////////////////////////////////////////////

/// This class implements the forward pass of fused multi-head attention
/// for Hopper GPUs (`HopperFusedMultiHeadAttentionForward`): `mma_tiler` is
/// `(M, N, K)`, the MMA's M rows per warpgroup, N keys per tile and the
/// head dimension; the CTA covers `2 * M` rows with two MMA warpgroups.
#[derive(Clone, Debug)]
pub struct HopperFusedMultiHeadAttentionForward {
    pub qk_acc_dtype: DType,
    pub pv_acc_dtype: DType,
    /// `(2 * M, N, K)`.
    pub cta_tiler: (usize, usize, usize),
    pub qk_mma_tiler: (usize, usize, usize),
    pub pv_mma_tiler: (usize, usize, usize),
    pub is_persistent: bool,
    pub mask_type: MaskEnum,
    /// Whether the kernel takes a left (`window_size_left`) and a right window size.
    pub has_window_left: bool,
    pub has_window_right: bool,
    pub num_mma_warp_groups: usize,
    pub num_threads_per_warp_group: usize,
    pub threads_per_cta: usize,
    pub num_regs_load: u32,
    pub num_regs_mma: u32,
    pub buffer_align_bytes: usize,
    pub q_stage: usize,
    pub kv_stage: usize,
    pub epi_stage: usize,
    pub epi_tile: (usize, usize),
    pub q_dtype: DType,
    pub o_dtype: DType,
}

impl HopperFusedMultiHeadAttentionForward {
    /// Initializes the configuration (`__init__`).
    pub fn new(
        qk_acc_dtype: DType,
        pv_acc_dtype: DType,
        mma_tiler: (usize, usize, usize),
        is_persistent: bool,
        mask_type: MaskEnum,
        has_window_left: bool,
        has_window_right: bool,
    ) -> Self {
        let num_mma_warp_groups = 2;
        let cta_tiler = (mma_tiler.0 * num_mma_warp_groups, mma_tiler.1, mma_tiler.2);
        let num_threads_per_warp_group = 128;
        HopperFusedMultiHeadAttentionForward {
            qk_acc_dtype,
            pv_acc_dtype,
            cta_tiler,
            qk_mma_tiler: mma_tiler,
            pv_mma_tiler: (mma_tiler.0, mma_tiler.2, mma_tiler.1),
            is_persistent,
            mask_type,
            has_window_left,
            has_window_right,
            num_mma_warp_groups,
            num_threads_per_warp_group,
            threads_per_cta: (num_mma_warp_groups + 1) * num_threads_per_warp_group,
            num_regs_load: 40 - 2 * 8,
            num_regs_mma: 240,
            buffer_align_bytes: 1024,
            q_stage: 2,
            kv_stage: 5,
            epi_stage: 2,
            epi_tile: (0, 0),
            q_dtype: DType::F16,
            o_dtype: DType::F16,
        }
    }

    /// The configuration for these input and output types: the checks of
    /// `__call__`, `_setup_attributes` and the epilogue tile.
    pub fn setup(mut self, q_dtype: DType, o_dtype: DType) -> Self {
        assert!(matches!(q_dtype, DType::F16 | DType::BF16), "TypeError: in_dtype must be Float16 or BFloat16");
        assert!(matches!(o_dtype, DType::F16 | DType::BF16), "TypeError: out_dtype must be Float16 or BFloat16");
        assert!(self.qk_acc_dtype == DType::F32 && self.pv_acc_dtype == DType::F32, "accumulators must be Float32");
        assert!(matches!(self.qk_mma_tiler.2, 64 | 128), "head dimension must be 64 or 128");
        assert!(self.qk_mma_tiler.0 == 64, "the MMA tile's M must be 64");
        assert!(
            self.qk_mma_tiler.1.is_multiple_of(16) && self.qk_mma_tiler.1 <= 256,
            "the K/V tile's N is a whole number of 16-wide MMA steps"
        );
        self.q_dtype = q_dtype;
        self.o_dtype = o_dtype;
        // The example's epilogue tile of the CTA tile (non-cooperative).
        self.epi_tile = sm90::compute_tile_shape_or_override(self.cta_tiler, o_dtype, false, None);
        self
    }

    /// `make_trivial_tiled_mma` for `Q K^T` (A and B in shared memory).
    pub fn qk_tiled_mma(&self) -> TiledMma {
        sm90::make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            OperandMajorMode::K,
            OperandMajorMode::K,
            self.qk_acc_dtype,
            (1, 1, 1),
            (self.qk_mma_tiler.0, self.qk_mma_tiler.1),
            OperandSource::SMEM,
        )
    }

    /// `make_trivial_tiled_mma` for `P V` (P in registers, V in shared
    /// memory with the head dimension contiguous: MN-major).
    pub fn pv_tiled_mma(&self) -> TiledMma {
        sm90::make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            OperandMajorMode::K,
            OperandMajorMode::MN,
            self.pv_acc_dtype,
            (1, 1, 1),
            (self.pv_mma_tiler.0, self.pv_mma_tiler.1),
            OperandSource::RMEM,
        )
    }

    /// The kernel's name for its configuration.
    pub fn name(&self) -> String {
        format!(
            "hopper_fmha_{}_{}_n{}_d{}_{}_{:?}{}{}",
            self.q_dtype,
            self.o_dtype,
            self.qk_mma_tiler.1,
            self.qk_mma_tiler.2,
            if self.is_persistent { "persistent" } else { "static" },
            self.mask_type,
            if self.has_window_left { "_wl" } else { "" },
            if self.has_window_right { "_wr" } else { "" },
        )
    }

    /// The shared-memory layouts of Q, K, V and O (`__call__`).
    pub fn make_smem_layouts(&self) -> (ComposedLayout, ComposedLayout, ComposedLayout, ComposedLayout) {
        let q = sm90::make_smem_layout_a(LayoutEnum::ROW_MAJOR, self.qk_mma_tiler, self.q_dtype, self.q_stage);
        let k = sm90::make_smem_layout_b(LayoutEnum::ROW_MAJOR, self.qk_mma_tiler, self.q_dtype, self.kv_stage);
        let v = sm90::make_smem_layout_b(LayoutEnum::COL_MAJOR, self.pv_mma_tiler, self.q_dtype, self.kv_stage);
        let o = sm90::make_smem_layout_epi(
            self.o_dtype,
            LayoutEnum::ROW_MAJOR,
            self.epi_tile,
            self.epi_stage,
            Some((self.epi_tile.0, self.epi_tile.1, self.epi_stage, self.num_mma_warp_groups).into()),
            Some(&[0, 1, 2, 3]),
        );
        (q, k, v, o)
    }

    /// The kernel: its parameters are the extents `s_q`, `s_k`, `h_r`
    /// (query heads per key head), `h_k` and `b`, the tensors `q`, `k`, `v`,
    /// `o` and `lse`, the scales `scale_softmax_log2`, `scale_softmax` and
    /// `scale_output`, then `window_size_left` and `window_size_right` when
    /// the kernel was built with them. Launch it with [`Self::launch`].
    pub fn build(&self) -> Kernel {
        use DType as D;
        match (self.q_dtype, self.o_dtype) {
            (D::F16, D::F16) => self.build_typed::<F16, F16>(),
            (D::BF16, D::BF16) => self.build_typed::<BF16, BF16>(),
            (D::F16, D::BF16) => self.build_typed::<F16, BF16>(),
            (D::BF16, D::F16) => self.build_typed::<BF16, F16>(),
            (a, b) => panic!("hopper_fmha: in {a}, out {b} is not among the combinations this example instantiates"),
        }
    }

    fn build_typed<T: Numeric, O: Numeric>(&self) -> Kernel {
        let fmha = self.clone();
        Kernel::builder(self.name()).block_dim([self.threads_per_cta as u32, 1, 1]).target(Arch::Sm90a).build(
            move || {
                let s_q = param_extent("s_q", 1);
                let s_k = param_extent("s_k", 1);
                let h_r = param_extent("h_r", 1);
                let h_k = param_extent("h_k", 1);
                let b = param_extent("b", 1);
                let (sqi, ski, hri, hki, bi) = (
                    IndexExpr::from(s_q),
                    IndexExpr::from(s_k),
                    IndexExpr::from(h_r),
                    IndexExpr::from(h_k),
                    IndexExpr::from(b),
                );
                let d = IndexExpr::from(fmha.cta_tiler.2);
                let one = IndexExpr::from(1);
                let zero = IndexExpr::from(0);
                let hq = hri.clone() * hki.clone();
                // Q and O: (s_q, d, h, b) of the dense (b, s_q, h, d).
                let q_layout = Layout::new(
                    (sqi.clone(), d.clone(), hq.clone(), bi.clone()),
                    (hq.clone() * d.clone(), one.clone(), d.clone(), sqi.clone() * hq.clone() * d.clone()),
                );
                // K: (s_k, d, (h_r, h_k), b), the query heads of a key head
                // sharing it (stride 0), of the dense (b, s_k, h_k, d).
                let k_layout = Layout::new(
                    (ski.clone(), d.clone(), (hri.clone(), hki.clone()), bi.clone()),
                    (
                        hki.clone() * d.clone(),
                        one.clone(),
                        (zero.clone(), d.clone()),
                        ski.clone() * hki.clone() * d.clone(),
                    ),
                );
                // V: (d, s_k, (h_r, h_k), b), the transpose of K's view.
                let v_layout = Layout::new(
                    (d.clone(), ski.clone(), (hri.clone(), hki.clone()), bi.clone()),
                    (
                        one.clone(),
                        hki.clone() * d.clone(),
                        (zero.clone(), d.clone()),
                        ski.clone() * hki.clone() * d.clone(),
                    ),
                );
                let q = param_tensor_aligned::<T>("q", q_layout.clone(), 16);
                let k = param_tensor_aligned::<T>("k", k_layout, 16);
                let v = param_tensor_aligned::<T>("v", v_layout, 16);
                let o = param_tensor_aligned::<O>("o", q_layout, 16);
                // lse: (s_q, h, b) of the (b, h, s_q) buffer.
                let lse = param_tensor::<f32>(
                    "lse",
                    Layout::new(
                        (sqi.clone(), hq.clone(), bi.clone()),
                        (one.clone(), sqi.clone(), sqi.clone() * hq.clone()),
                    ),
                );
                let scale_softmax_log2 = param_scalar::<f32>("scale_softmax_log2");
                let scale_softmax = param_scalar::<f32>("scale_softmax");
                let scale_output = param_scalar::<f32>("scale_output");
                let window_left = fmha.has_window_left.then(|| param_scalar::<i32>("window_size_left"));
                let window_right = fmha.has_window_right.then(|| param_scalar::<i32>("window_size_right"));
                let sizes = Sizes { s_q, s_k, h_r, h_k, b };
                fmha.call::<T, O>(
                    &q,
                    &k,
                    &v,
                    &o,
                    &lse,
                    sizes,
                    (scale_softmax_log2, scale_softmax, scale_output),
                    (window_left, window_right),
                );
            },
        )
    }

    /// The TMA atoms, then the kernel (`__call__`).
    fn call<T: Numeric, O: Numeric>(
        &self,
        q: &Tensor<T>,
        k: &Tensor<T>,
        v: &Tensor<T>,
        o: &Tensor<O>,
        lse: &Tensor<f32>,
        sizes: Sizes,
        scales: (Float32, Float32, Float32),
        window: (Option<Int32>, Option<Int32>),
    ) {
        let (q_smem_layout_staged, k_smem_layout_staged, v_smem_layout_staged, o_smem_layout_staged) =
            self.make_smem_layouts();
        let (tma_atom_q, tma_tensor_q) =
            Self::make_tma_atoms_and_tensors(q, &q_smem_layout_staged, (self.qk_mma_tiler.0, self.qk_mma_tiler.2), 1);
        let (tma_atom_k, tma_tensor_k) =
            Self::make_tma_atoms_and_tensors(k, &k_smem_layout_staged, (self.qk_mma_tiler.1, self.qk_mma_tiler.2), 1);
        // The tile of V is (head dimension, keys).
        let (tma_atom_v, tma_tensor_v) =
            Self::make_tma_atoms_and_tensors(v, &v_smem_layout_staged, (self.pv_mma_tiler.1, self.pv_mma_tiler.2), 1);
        let o_smem_layout = o_smem_layout_staged.slice((.., .., 0, 0));
        let (tma_atom_o, tma_tensor_o): (CopyAtom<O>, Tensor<Ident>) =
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, o, o_smem_layout, self.epi_tile, 1).into();
        let q_copy_bytes = static_usize(size_in_bytes::<T>(&q_smem_layout_staged.slice((.., .., 0))));
        let kv_copy_bytes = static_usize(size_in_bytes::<T>(&k_smem_layout_staged.slice((.., .., 0))));
        self.kernel::<T, O>(&Inputs {
            qk_tiled_mma: self.qk_tiled_mma(),
            pv_tiled_mma: self.pv_tiled_mma(),
            tma_atom_q: &tma_atom_q,
            m_q: &tma_tensor_q,
            tma_atom_k: &tma_atom_k,
            m_k: &tma_tensor_k,
            tma_atom_v: &tma_atom_v,
            m_v: &tma_tensor_v,
            tma_atom_o: &tma_atom_o,
            m_o: &tma_tensor_o,
            m_lse: lse,
            sizes,
            scales,
            window,
            q_smem_layout_staged: &q_smem_layout_staged,
            k_smem_layout_staged: &k_smem_layout_staged,
            v_smem_layout_staged: &v_smem_layout_staged,
            o_smem_layout_staged: &o_smem_layout_staged,
            q_copy_bytes,
            kv_copy_bytes,
        });
    }

    /// Create TMA atoms and tensors for input tensors
    /// (`_make_tma_atoms_and_tensors`).
    pub fn make_tma_atoms_and_tensors<E: Numeric>(
        tensor: &Tensor<E>,
        smem_layout_staged: &ComposedLayout,
        smem_tile: (usize, usize),
        mcast_dim: usize,
    ) -> (CopyAtom<E>, Tensor<Ident>) {
        let op: CopyOp = if mcast_dim == 1 {
            cpasync::CopyBulkTensorTileG2SOp::default().into()
        } else {
            cpasync::CopyBulkTensorTileG2SMulticastOp::default().into()
        };
        let smem_layout = smem_layout_staged.slice((.., .., 0));
        cpasync::make_tiled_tma_atom(op, tensor, smem_layout, smem_tile, mcast_dim).into()
    }

    /// The launch for a problem of `s_q` query rows, `h` query heads and `b`
    /// batches (`compute_grid`): one CTA per `(M tile, head, batch)`, or for
    /// a persistent kernel `min(num_sms, tiles)` CTAs.
    pub fn launch(&self, s_q: usize, h: usize, b: usize, num_sms: usize) -> LaunchConfig {
        let tiles = [s_q.div_ceil(self.cta_tiler.0), h, b];
        let grid = if self.is_persistent {
            [num_sms.min(tiles.iter().product()) as u32, 1, 1]
        } else {
            tiles.map(|t| t as u32)
        };
        LaunchConfig::new(grid, [self.threads_per_cta as u32, 1, 1])
    }

    /// The work tile of iteration `it` of this CTA (`FmhaStaticTileScheduler`):
    /// `(M tile, head, batch)`, and how many tiles the CTA visits. A
    /// persistent CTA starts at its block index and strides by the grid size;
    /// otherwise the block index is the tile.
    fn tile_scheduler(&self, sizes: &Sizes) -> (Int32, impl Fn(Int32) -> (Int32, Int32, Int32)) {
        let hq = sizes.h_r * sizes.h_k;
        let m_tiles = ceil_div_i32(sizes.s_q, self.cta_tiler.0 as i32);
        let persistent = self.is_persistent;
        let grid = grid_dim();
        let num_blocks = m_tiles * hq * sizes.b;
        let bx = block_idx().x;
        let num_persistent_sm = grid.x * grid.y * grid.z;
        // The tiles of this CTA: linear indices bx, bx + P, ... below the total.
        let count = if persistent {
            select(bx.lt(num_blocks), (num_blocks - bx + num_persistent_sm - 1) / num_persistent_sm, Int32::from(0))
        } else {
            Int32::from(1)
        };
        let coords = move |it: Int32| {
            if persistent {
                let lin = bx + it * num_persistent_sm;
                (lin % m_tiles, (lin / m_tiles) % hq, lin / (m_tiles * hq))
            } else {
                (block_idx().x, block_idx().y, block_idx().z)
            }
        };
        (count, coords)
    }

    //  GPU device kernel
    /// GPU device kernel performing the fused attention (`kernel`).
    fn kernel<T: Numeric, O: Numeric>(&self, a: &Inputs<'_, T, O>) {
        let (tile_m, tile_n, head_dim) = self.qk_mma_tiler;
        let tidx = thread_idx().x;
        let warp_idx = make_warp_uniform(warp_idx());
        let warp_group_idx = make_warp_uniform(tidx / self.num_threads_per_warp_group as i32);
        let wg_threads = self.num_threads_per_warp_group as i32;

        let o_elems = if self.is_persistent { smem_extent(a.o_smem_layout_staged) } else { 0 };
        let storage = SharedStorage::<T, O>::allocate_aligned(
            self.buffer_align_bytes,
            self.q_stage,
            self.kv_stage,
            self.num_mma_warp_groups,
            o_elems,
            smem_extent(a.q_smem_layout_staged),
            smem_extent(a.k_smem_layout_staged),
        );

        let warps_per_wg = self.num_threads_per_warp_group / 32;
        let load_q_pipeline = PipelineTmaAsync::create(
            self.q_stage,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, warps_per_wg),
            a.q_copy_bytes,
            PipelineTmaAsyncOptions {
                barrier_storage: Some(storage.load_q_mbar_ptr.data_ptr().cast::<u64>()),
                tidx: Some(tidx),
                defer_sync: true,
                ..Default::default()
            },
        );
        let load_kv_pipeline = PipelineTmaAsync::create(
            self.kv_stage,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, self.num_mma_warp_groups * warps_per_wg),
            a.kv_copy_bytes,
            PipelineTmaAsyncOptions {
                barrier_storage: Some(storage.load_kv_mbar_ptr.data_ptr().cast::<u64>()),
                tidx: Some(tidx),
                defer_sync: true,
                ..Default::default()
            },
        );
        let tma_store_pipeline = PipelineTmaStore::create(self.epi_stage, CooperativeGroup::new(Agent::Thread, 1));
        let math_wg_order_barrier = PipelineOrder::create(
            1,
            self.num_mma_warp_groups,
            warp_group_idx - 1,
            CooperativeGroup::new(Agent::Thread, self.num_threads_per_warp_group),
            PipelineOrderOptions {
                barrier_storage: Some(storage.math_wg_order_barrier.data_ptr().cast::<u64>()),
                defer_sync: true,
                ..Default::default()
            },
        );
        let (load_q_producer, load_q_consumer) = load_q_pipeline.make_participants();
        let (load_kv_producer, load_kv_consumer) = load_kv_pipeline.make_participants();

        let s_q = storage.s_q.get_tensor_composed(a.q_smem_layout_staged.clone());
        let s_k = storage.s_k.get_tensor_composed(a.k_smem_layout_staged.clone());
        // V's tiles are K's buffer with another layout.
        let s_v = storage.s_k.get_tensor_composed(a.v_smem_layout_staged.clone());
        let s_o = if self.is_persistent {
            storage.s_o.get_tensor_composed(a.o_smem_layout_staged.clone())
        } else {
            // O's staging tiles reuse Q's buffer (warpgroup g's tiles lie in
            // its own Q stage).
            let ptr = recast_ptr::<O, T>(&s_q.ptr(), None);
            make_tensor(&ptr, a.o_smem_layout_staged.clone())
        };

        let qk_thr_mma = a.qk_tiled_mma.get_slice(tidx % wg_threads);
        let pv_thr_mma = a.pv_tiled_mma.get_slice(tidx % wg_threads);

        // Fence the mbarrier init and finish the shared-memory set-up.
        pipeline_init_arrive(None, true);
        pipeline_init_wait(None);

        if_(warp_idx.eq(0), || {
            cpasync::prefetch_descriptor(a.tma_atom_q);
            cpasync::prefetch_descriptor(a.tma_atom_k);
            cpasync::prefetch_descriptor(a.tma_atom_v);
            cpasync::prefetch_descriptor(a.tma_atom_o);
        });

        let sizes = &a.sizes;
        let (tile_count, tile_coords) = self.tile_scheduler(sizes);
        let mask = FusedMask {
            mask_type: self.mask_type,
            tile_m: self.cta_tiler.0 as i32,
            tile_n: tile_n as i32,
            seqlen_q: sizes.s_q,
            seqlen_k: sizes.s_k,
            window_left: a.window.0,
            window_right: a.window.1,
        };
        let num_wg = self.num_mma_warp_groups as i32;

        // /////////////////////////////////////////////////////////////////////
        //  Load warpgroup
        // /////////////////////////////////////////////////////////////////////
        if_(warp_group_idx.eq(0), || {
            setmaxregister_decrease(self.num_regs_load);
            let load_q_producer = Var::new(load_q_producer);
            let load_kv_producer = Var::new(load_kv_producer);
            let producer_warp_role = warp_idx % (self.num_threads_per_warp_group as i32 / 32);

            for_range(0, tile_count, 1, |it| {
                let (m_tile, h, b) = tile_coords(it);
                let k_index = mask.get_trip_start(m_tile);
                let fusion_tile_count = mask.get_trip_count(m_tile);
                let k_tile_count = fusion_tile_count * 2;

                // (64, d, M tiles) of this head and batch, grouped for TMA.
                let g_q = local_tile(a.m_q, (tile_m, head_dim), (.., 0, h, b));
                let (t_qs_q, t_qg_q) = cpasync::tma_partition(
                    a.tma_atom_q,
                    0,
                    &make_layout(1, 1),
                    &s_q.group_modes(0, 2),
                    &g_q.group_modes(0, 2),
                );
                let g_k = local_tile(a.m_k, (tile_n, head_dim), (.., 0, h, b));
                let (t_ks_k, t_kg_k) = cpasync::tma_partition(
                    a.tma_atom_k,
                    0,
                    &make_layout(1, 1),
                    &s_k.group_modes(0, 2),
                    &g_k.group_modes(0, 2),
                );
                let g_v = local_tile(a.m_v, (head_dim, tile_n), (0, .., h, b));
                let (t_vs_v, t_vg_v) = cpasync::tma_partition(
                    a.tma_atom_v,
                    0,
                    &make_layout(1, 1),
                    &s_v.group_modes(0, 2),
                    &g_v.group_modes(0, 2),
                );

                if_(producer_warp_role.eq(1), || {
                    let load_q = |q_idx: i32| {
                        let mut p = load_q_producer.get();
                        let handle = p.acquire_and_advance(None);
                        load_q_producer.set(p);
                        copy_with(
                            a.tma_atom_q,
                            &t_qg_q.slice((.., m_tile * num_wg + q_idx)),
                            &t_qs_q.slice((.., handle.index())),
                            CopyArgs::new().tma_bar_ptr(&handle.barrier()),
                        );
                    };
                    let load_kv = |is_k: bool, k_idx: Int32| {
                        let mut p = load_kv_producer.get();
                        let handle = p.acquire_and_advance(None);
                        load_kv_producer.set(p);
                        let args = CopyArgs::new().tma_bar_ptr(&handle.barrier());
                        if is_k {
                            copy_with(
                                a.tma_atom_k,
                                &t_kg_k.slice((.., k_idx)),
                                &t_ks_k.slice((.., handle.index())),
                                args,
                            );
                        } else {
                            copy_with(
                                a.tma_atom_v,
                                &t_vg_v.slice((.., k_idx)),
                                &t_vs_v.slice((.., handle.index())),
                                args,
                            );
                        }
                    };
                    // Load Q0
                    load_q(0);
                    // Load K
                    if_(k_tile_count.gt(0), || load_kv(true, k_index));
                    // Load Q1
                    load_q(1);
                    // Load V
                    if_((k_tile_count - 1).gt(0), || load_kv(false, k_index));
                    // The remaining K, V pairs
                    let remaining = (k_tile_count - 2).max(0);
                    for_range(0, (remaining + 1) / 2, 1, |i| {
                        load_kv(true, k_index + 1 + i);
                        load_kv(false, k_index + 1 + i);
                    });
                });
            });
        });

        // /////////////////////////////////////////////////////////////////////
        //  Compute and epilogue warpgroups
        // /////////////////////////////////////////////////////////////////////
        if_(warp_group_idx.eq(1) | warp_group_idx.eq(2), || {
            setmaxregister_increase(self.num_regs_mma);
            let load_q_consumer = Var::new(load_q_consumer);
            let load_kv_consumer = Var::new(load_kv_consumer);
            let math_wg_order = Var::new(math_wg_order_barrier.clone());
            let wg = warp_group_idx - 1;

            // Mainloop setup QK and PV
            let t_ss_q = qk_thr_mma.partition_a(&s_q);
            let t_ss_k = qk_thr_mma.partition_b(&s_k);
            let t_sr_q = a.qk_tiled_mma.make_fragment_a(&t_ss_q);
            let t_sr_k = a.qk_tiled_mma.make_fragment_b(&t_ss_k);
            let t_os_v = pv_thr_mma.partition_b(&s_v);
            let t_or_v = a.pv_tiled_mma.make_fragment_b(&t_os_v);
            let num_k_blocks_qk = static_usize(size(&t_sr_q, &[2]));
            let num_k_blocks_pv = static_usize(size(&t_or_v, &[2]));

            // This thread's coordinate in each tile of the scores and of O.
            let qk_coords = thread_coords(&qk_thr_mma, tile_m, tile_n);
            let pv_coords = thread_coords(&pv_thr_mma, tile_m, head_dim);

            let qk_acc_shape = a.qk_tiled_mma.partition_shape_c((tile_m, tile_n));
            let pv_acc_shape = a.pv_tiled_mma.partition_shape_c((tile_m, head_dim));
            let rows = ROWS_PER_THREAD;

            let pv_dtype_marker = make_identity_tensor((tile_m, tile_n));

            for_range(0, tile_count, 1, |it| {
                let (m_tile, h, b) = tile_coords(it);
                // Skip the Q stages of the warpgroups before this one.
                if_(wg.eq(1), || {
                    load_q_consumer.update(|mut c| {
                        c.advance(1);
                        c
                    })
                });

                // _wg_coord_0 is the 64-row tile of this warpgroup.
                let wg_row_tile = m_tile * num_wg + wg;
                let q_handle = {
                    let c = load_q_consumer.get();
                    c.wait(None)
                };

                // Allocate PV acc, the running max and sum.
                // (Registers carried through a loop start out filled.)
                let acc_pv_init = a.pv_tiled_mma.make_fragment_c::<f32>(&pv_acc_shape);
                acc_pv_init.fill(0.0f32);
                let s_max_init = make_fragment::<f32>(rows);
                s_max_init.fill(0.0f32);
                let a_sum_init = make_fragment::<f32>(rows);
                a_sum_init.fill(0.0f32);
                let acc_pv = Var::new(acc_pv_init);
                let s_max = Var::new(s_max_init);
                let a_sum = Var::new(a_sum_init);

                let kv_offset = Var::new(mask.get_trip_start(m_tile));
                let masked_leading_count = mask.get_masked_leading_count(m_tile);
                let unmasked_trip_count = mask.get_unmasked_trip_count(m_tile);

                let q_row0 = wg_row_tile * tile_m as i32;
                let q_stage_idx = q_handle.index();
                let tsr_q = t_sr_q.slice((.., .., .., q_stage_idx));

                // One MMA of a stage's worth of Q x K^T from zero accumulators.
                let qk_gemm = |acc: &Tensor<f32>, k_stage: Int32| {
                    let tsr_k = t_sr_k.slice((.., .., .., k_stage));
                    gemm_zero_acc(&mut a.qk_tiled_mma.clone(), &tsr_q, &tsr_k, acc, num_k_blocks_qk);
                };
                // `P V` into `acc_pv`, `accumulate` or from zero.
                let pv_gemm = |p: &Tensor<T>, v_stage: Int32, accumulate: bool| {
                    let tor_v = t_or_v.slice((.., .., .., v_stage));
                    let acc = acc_pv.get();
                    let mut mma = a.pv_tiled_mma.clone();
                    // MMA PV, waited for before the accumulators go back to
                    // the Var (the wait completes the registers it holds).
                    warpgroup::fence();
                    if accumulate {
                        mma.set(MmaField::ACCUMULATE, true);
                        gemm(&mma, &acc, p, &tor_v, &acc);
                    } else {
                        gemm_zero_acc(&mut mma, p, &tor_v, &acc, num_k_blocks_pv);
                    }
                    warpgroup::commit_group();
                    warpgroup::wait_group(0);
                    acc_pv.set(acc);
                };
                // The first tile (always masked) through the softmax's first iteration.
                {
                    let acc_qk = a.qk_tiled_mma.make_fragment_c::<f32>(&qk_acc_shape);
                    let kv_off = kv_offset.get();
                    kv_offset.set(kv_off + 1);
                    let k_handle = {
                        let mut c = load_kv_consumer.get();
                        let h = c.wait_and_advance(None);
                        load_kv_consumer.set(c);
                        h
                    };
                    {
                        let o = math_wg_order.get();
                        o.wait();
                    }
                    // MMA QK
                    warpgroup::fence();
                    qk_gemm(&acc_qk, k_handle.index());
                    warpgroup::commit_group();
                    {
                        let mut o = math_wg_order.get();
                        o.arrive();
                        math_wg_order.set(o);
                    }
                    // Wait for the pipeline MMAs to drain
                    warpgroup::wait_group(0);
                    self.softmax_step(
                        &mask,
                        true,
                        &acc_qk,
                        &qk_coords,
                        (q_row0, kv_off * tile_n as i32),
                        &s_max,
                        &a_sum,
                        None,
                        a.scales.0,
                        true,
                    );
                    let p = self.make_acc_into_op::<T>(&acc_qk, &pv_thr_mma);
                    let v_handle = {
                        let mut c = load_kv_consumer.get();
                        let h = c.wait_and_advance(None);
                        load_kv_consumer.set(c);
                        h
                    };
                    // MMA PV
                    pv_gemm(&p, v_handle.index(), false);
                    k_handle.release();
                    v_handle.release();
                }
                let _ = &pv_dtype_marker;

                // `compute`: the loop over the remaining K tiles, with or
                // without the mask fused into the softmax; returns the count
                // left (0, or a negative count it was given).
                let compute = |fusion: bool, k_tile_count: Int32| -> Int32 {
                    for_range(0, k_tile_count, 1, |_| {
                        let kv_off = kv_offset.get();
                        kv_offset.set(kv_off + 1);
                        let acc_qk = a.qk_tiled_mma.make_fragment_c::<f32>(&qk_acc_shape);
                        let k_handle = {
                            let mut c = load_kv_consumer.get();
                            let h = c.wait_and_advance(None);
                            load_kv_consumer.set(c);
                            h
                        };
                        // MMA QK
                        warpgroup::fence();
                        qk_gemm(&acc_qk, k_handle.index());
                        warpgroup::commit_group();
                        let tok = load_kv_consumer.get().try_wait();
                        // Wait for the pipeline MMAs to drain
                        warpgroup::wait_group(0);
                        self.softmax_step(
                            &mask,
                            fusion,
                            &acc_qk,
                            &qk_coords,
                            (q_row0, kv_off * tile_n as i32),
                            &s_max,
                            &a_sum,
                            Some(&acc_pv),
                            a.scales.0,
                            false,
                        );
                        let p = self.make_acc_into_op::<T>(&acc_qk, &pv_thr_mma);
                        let v_handle = {
                            let mut c = load_kv_consumer.get();
                            let h = c.wait_and_advance(Some(tok));
                            load_kv_consumer.set(c);
                            h
                        };
                        // MMA PV
                        pv_gemm(&p, v_handle.index(), true);
                        k_handle.release();
                        v_handle.release();
                    });
                    k_tile_count.min(0)
                };

                let unmasked = Var::new(unmasked_trip_count - 1);
                if_(masked_leading_count.ge(1), || {
                    compute(true, masked_leading_count - 1);
                    unmasked.set(unmasked_trip_count);
                });
                let k_tile_count = compute(false, unmasked.get());
                let k_tile_count = mask.get_masked_trailing_count(m_tile, k_tile_count);
                // Use fusion in softmax
                compute(true, k_tile_count);

                if self.is_persistent {
                    load_q_consumer.get().release(Some(&q_handle));
                }

                // Wait for the pipeline MMAs to drain
                warpgroup::wait_group(0);

                // acc_pv updated
                let lse = self.tail(&s_max, &a_sum, &acc_pv, a.scales.1, a.scales.2);

                if_(warp_group_idx.eq(1), || {
                    load_q_consumer.update(|mut c| {
                        c.advance(self.num_mma_warp_groups);
                        c
                    })
                });
                if_(warp_group_idx.eq(2), || {
                    load_q_consumer.update(|mut c| {
                        c.advance(self.num_mma_warp_groups - 1);
                        c
                    })
                });

                math_wg_order.get().wait();

                // store log-sum-exp (LSE)
                let o_row0 = wg_row_tile * tile_m as i32;
                for i in 0..rows {
                    // Elements (row i, column 0) of the thread's rows: only
                    // the threads that hold column 0 write.
                    let (row, col) = &pv_coords[2 * i];
                    let global_row = o_row0 + *row;
                    let store_it = col.eq(0) & global_row.lt(sizes.s_q);
                    let lse_i = lse.load(i);
                    if_(store_it, || {
                        a.m_lse.store((global_row, h, b), lse_i);
                    });
                }

                // Epilogue
                self.epilogue::<O>(
                    &a.pv_tiled_mma,
                    &acc_pv.get(),
                    &s_o,
                    a.m_o,
                    a.tma_atom_o,
                    &tma_store_pipeline,
                    (wg_row_tile, h, b),
                    wg,
                    tidx,
                    warp_idx,
                    warp_group_idx,
                );

                {
                    let mut o = math_wg_order.get();
                    o.arrive();
                    math_wg_order.set(o);
                }
            });
            // The stores in flight are done before the shared memory goes.
            if_(warp_idx.eq(4) | warp_idx.eq(8), || tma_store_pipeline.producer_tail());
        });
    }

    /// One step of the online softmax on the scores `acc_qk` of a K tile
    /// (`softmax_step`): the mask (when `fusion`), the row maxima, `exp2`
    /// of the scaled scores, the rescaling of the running sum and of
    /// `acc_pv`, and the row sums. `origin` is the global `(row, column)` of
    /// the tile's first element.
    // `i` indexes several arrays at once (the row of the tile).
    #[allow(clippy::needless_range_loop)]
    fn softmax_step(
        &self,
        mask: &FusedMask,
        fusion: bool,
        acc_qk: &Tensor<f32>,
        coords: &[(Int32, Int32)],
        origin: (Int32, Int32),
        s_max: &Var<Tensor<f32>>,
        a_sum: &Var<Tensor<f32>>,
        acc_pv: Option<&Var<Tensor<f32>>>,
        scale_softmax_log2: Float32,
        is_first_iter: bool,
    ) {
        let rows = ROWS_PER_THREAD;
        let cols = acc_qk.size() / rows;
        // acc_qk_mn[i, j]
        let mn = |i: usize, j: usize| acc_mn_index(i, j);
        if fusion {
            // FusedMask.apply_mask
            let neg_inf = Float32::constant(f32::NEG_INFINITY);
            for (idx, (row, col)) in coords.iter().enumerate() {
                let masked = mask.is_masked(origin.0 + *row, origin.1 + *col);
                acc_qk.store(idx, select(masked, neg_inf, acc_qk.load(idx)));
            }
        }
        let max_t = s_max.get();
        let sum_t = a_sum.get();
        let pv_t = acc_pv.map(|v| v.get());
        let neg_inf = Float32::constant(f32::NEG_INFINITY);
        let zero = Float32::constant(0.0);

        let mut s_max_prev = [zero; ROWS_PER_THREAD];
        for i in 0..rows {
            if is_first_iter {
                // Linear reduction is faster for the first iteration
                let mut m = acc_qk.load(mn(i, 0));
                for j in 1..cols {
                    m = fmax(m, acc_qk.load(mn(i, j)));
                }
                max_t.store(i, m);
            } else {
                s_max_prev[i] = max_t.load(i);
                // Linear reduction is faster here, as well
                let mut m = max_t.load(i);
                for j in 0..cols {
                    m = m.max(acc_qk.load(mn(i, j)));
                }
                max_t.store(i, m);
            }
            // reduce max over the 4 lanes of a row
            let reduced = warp_reduction_max(max_t.load(i), 4);
            max_t.store(i, reduced);

            let s_max_i = max_t.load(i);
            let local_max = select(s_max_i.eq(neg_inf), zero, s_max_i);
            let scale_max = scale_softmax_log2 * local_max;
            for j in 0..cols {
                let x = acc_qk.load(mn(i, j));
                acc_qk.store(mn(i, j), (scale_softmax_log2 * x - scale_max).exp2());
            }

            let mut prev_sum = zero;
            if !is_first_iter {
                let s_max_cur = local_max;
                let scale_pv = ((s_max_prev[i] - s_max_cur) * scale_softmax_log2).exp2();
                sum_t.store(i, sum_t.load(i) * scale_pv);
                let pv = pv_t.as_ref().expect("acc_pv");
                let pv_cols = pv.size() / rows;
                for j in 0..pv_cols {
                    pv.store(mn(i, j), pv.load(mn(i, j)) * scale_pv);
                }
                prev_sum = sum_t.load(i);
            }
            let mut row_sum = zero;
            for j in 0..cols {
                row_sum = row_sum + acc_qk.load(mn(i, j));
            }
            sum_t.store(i, prev_sum + row_sum);
        }
        s_max.set(max_t);
        a_sum.set(sum_t);
        if let (Some(v), Some(t)) = (acc_pv, pv_t) {
            v.set(t);
        }
    }

    /// The scores as the A operand of `P V` (`make_acc_into_op`): the
    /// accumulator's values converted to the input type, in the same order
    /// (a k-block of 16 columns is 8 consecutive values: `v0 + 2 v1 + 4 v2`
    /// with `v2` the 8-column group).
    fn make_acc_into_op<T: Numeric>(&self, acc: &Tensor<f32>, pv_thr: &ThrMma) -> Tensor<T> {
        let p = pv_thr.partition_fragment_a::<T, _>(&make_identity_tensor((self.qk_mma_tiler.0, self.qk_mma_tiler.1)));
        assert_eq!(p.size(), acc.size(), "the P fragment is as large as the scores' accumulator");
        for i in 0..acc.size() {
            p.store(i, acc.load(i).to::<T>());
        }
        p
    }

    /// The final processing of FMHA (`tail`): reduces the sums over the
    /// row's lanes, computes the log-sum-exp of each row, handles zero and
    /// NaN sums, and scales `acc_pv`. Returns the `lse` registers.
    fn tail(
        &self,
        s_max: &Var<Tensor<f32>>,
        a_sum: &Var<Tensor<f32>>,
        acc_pv: &Var<Tensor<f32>>,
        scale_softmax: Float32,
        scale_output: Float32,
    ) -> Tensor<f32> {
        let rows = ROWS_PER_THREAD;
        let (max_t, sum_t, acc) = (s_max.get(), a_sum.get(), acc_pv.get());
        let cols = acc.size() / rows;
        let lse = make_fragment::<f32>(rows);
        let one = Float32::constant(1.0);
        for i in 0..rows {
            sum_t.store(i, warp_reduction_sum(sum_t.load(i), 4));
        }
        for i in 0..rows {
            let sum = sum_t.load(i);
            let bad = sum.eq(0.0f32) | sum.ne(sum);
            let inv_sum = select(bad, one, sum.rcp());
            let lse_i = select(bad, Float32::constant(f32::INFINITY), max_t.load(i) * scale_softmax + sum.ln());
            lse.store(i, lse_i);
            let scale = inv_sum;
            for j in 0..cols {
                acc.store(acc_mn_index(i, j), acc.load(acc_mn_index(i, j)) * (scale * scale_output));
            }
        }
        acc_pv.set(acc);
        lse
    }

    /// The epilogue of one warpgroup: registers → shared memory
    /// (`stmatrix`) → global memory (TMA store), one epilogue tile at a
    /// time, on the warpgroup's named barrier.
    fn epilogue<O: Numeric>(
        &self,
        pv_tiled_mma: &TiledMma,
        acc_pv: &Tensor<f32>,
        s_o: &Tensor<O>,
        m_o: &Tensor<Ident>,
        tma_atom_o: &CopyAtom<O>,
        tma_store_pipeline: &PipelineTmaStore,
        (wg_row_tile, h, b): (Int32, Int32, Int32),
        wg: Int32,
        tidx: Int32,
        warp_idx: Int32,
        warp_group_idx: Int32,
    ) {
        let copy_atom_r2s = sm90::get_smem_store_op::<O>(LayoutEnum::ROW_MAJOR, self.pv_acc_dtype);
        let copy_atom_o = make_copy_atom::<O>(CopyOp::StMatrix { num: 4, trans: false });
        let tiled_copy_o_atom = make_tiled_copy_c_atom(&copy_atom_o, pv_tiled_mma);
        let tiled_copy_r2s = make_tiled_copy_s(&copy_atom_r2s, &tiled_copy_o_atom);

        let thr_copy_r2s = tiled_copy_r2s.get_slice(tidx % self.num_threads_per_warp_group as i32);
        let t_rs_sd = thr_copy_r2s.partition_d(s_o);
        let t_rs_racc = tiled_copy_r2s.retile(acc_pv);

        // Allocate D registers.
        let rd_shape = thr_copy_r2s.partition_s(s_o).shape().clone();
        let t_rs_rd_shape: Shape = Tree::Branch((0..3).map(|i| rd_shape.mode(i).clone()).collect());
        let t_rs_rd = make_fragment::<f32>(t_rs_rd_shape.clone());
        let size_t_rs_rd = t_rs_rd.size();

        let g_d = local_tile(m_o, (self.pv_mma_tiler.0, self.pv_mma_tiler.1), (wg_row_tile, 0, h, b));
        let sepi_for_tma_partition = s_o.group_modes(0, 2);
        let tcgc_for_tma_partition = zipped_divide(&g_d, self.epi_tile);
        let (bsg_sd, bsg_gd) =
            cpasync::tma_partition(tma_atom_o, 0, &make_layout(1, 1), &sepi_for_tma_partition, &tcgc_for_tma_partition);
        let epi_tile_num = static_usize(size(&tcgc_for_tma_partition, &[1]));

        if !self.is_persistent {
            // O's staging tiles are this warpgroup's Q stage: every warp of
            // the warpgroup is done reading Q (the example relies on the
            // MMAs having finished in step).
            cute::pipeline::arrive_and_wait(warp_group_idx, self.num_threads_per_warp_group as i32);
        }
        for epi_idx in 0..epi_tile_num {
            // Copy from accumulators to D registers
            for epi_v in 0..size_t_rs_rd {
                t_rs_rd.store(epi_v, t_rs_racc.load(epi_idx * size_t_rs_rd + epi_v));
            }
            // Type conversion
            let t_rs_rd_out = make_fragment::<O>(t_rs_rd_shape.clone());
            let acc_vec = t_rs_rd.load_ssa();
            t_rs_rd_out.store_ssa(&acc_vec.to::<O>());

            // Copy from D registers to shared memory
            let epi_buffer = epi_idx % self.epi_stage;
            copy(&tiled_copy_r2s, &t_rs_rd_out, &t_rs_sd.slice((.., .., .., epi_buffer, wg)));

            fence_proxy_with(
                FenceProxyKind::AsyncShared,
                FenceProxyOptions { space: Some(SharedSpace::Cta), ..Default::default() },
            );
            cute::pipeline::arrive_and_wait(warp_group_idx, self.num_threads_per_warp_group as i32);

            // only one warp in each warpgroup copy shared memory to global memory
            if_(warp_idx.eq(4) | warp_idx.eq(8), || {
                copy(tma_atom_o, &bsg_sd.slice((.., epi_buffer, wg)), &bsg_gd.slice((.., epi_idx)));
                tma_store_pipeline.producer_commit();
                tma_store_pipeline.producer_acquire();
            });

            cute::pipeline::arrive_and_wait(warp_group_idx, self.num_threads_per_warp_group as i32);
        }
    }
}

/// The values a thread holds per row of the accumulator (the 2 rows of the
/// warpgroup MMA's C layout, `v1`).
const ROWS_PER_THREAD: usize = 2;

/// The accumulator's flat index of element `(i, j)` of the `(rows, columns)`
/// view (`layout_acc_mn`): row `i` is `v1`; the column index `j = v0 + 2 v2`
/// walks `v0` fastest.
fn acc_mn_index(i: usize, j: usize) -> usize {
    (j % 2) + 2 * i + 4 * (j / 2)
}

/// `gemm_zero_acc`: `C = A B` over the k-blocks of `A` (`(MMA, M, K)`) and
/// `B` (`(MMA, N, K)`), the first with a zero accumulator.
fn gemm_zero_acc<A: Numeric, B: Numeric>(
    tiled_mma: &mut TiledMma,
    a: &Tensor<A>,
    b: &Tensor<B>,
    c: &Tensor<f32>,
    num_k_blocks: usize,
) {
    for k_block_idx in 0..num_k_blocks {
        tiled_mma.set(MmaField::ACCUMULATE, k_block_idx != 0);
        gemm(tiled_mma, c, &a.slice((.., .., k_block_idx)), &b.slice((.., .., k_block_idx)), c);
    }
}

/// The `(row, column)` of each element of this thread's accumulator
/// partition of a `(rows, cols)` tile, as the MMA's C layout places them.
fn thread_coords(thr: &ThrMma, rows: usize, cols: usize) -> Vec<(Int32, Int32)> {
    let ident = make_identity_tensor((rows, cols));
    let part = thr.partition_c(&ident);
    (0..part.size())
        .map(|i| {
            let c = part.coord(i);
            (leaf_i32(&c, 0), leaf_i32(&c, 1))
        })
        .collect()
}

/// The kernel's problem sizes.
#[derive(Clone, Copy)]
pub struct Sizes {
    pub s_q: Int32,
    pub s_k: Int32,
    pub h_r: Int32,
    pub h_k: Int32,
    pub b: Int32,
}

/// What `kernel` takes (the example's arguments).
pub struct Inputs<'a, T: Numeric, O: Numeric> {
    pub qk_tiled_mma: TiledMma,
    pub pv_tiled_mma: TiledMma,
    pub tma_atom_q: &'a CopyAtom<T>,
    pub m_q: &'a Tensor<Ident>,
    pub tma_atom_k: &'a CopyAtom<T>,
    pub m_k: &'a Tensor<Ident>,
    pub tma_atom_v: &'a CopyAtom<T>,
    pub m_v: &'a Tensor<Ident>,
    pub tma_atom_o: &'a CopyAtom<O>,
    pub m_o: &'a Tensor<Ident>,
    pub m_lse: &'a Tensor<f32>,
    pub sizes: Sizes,
    /// `scale_softmax_log2`, `scale_softmax`, `scale_output`.
    pub scales: (Float32, Float32, Float32),
    pub window: (Option<Int32>, Option<Int32>),
    pub q_smem_layout_staged: &'a ComposedLayout,
    pub k_smem_layout_staged: &'a ComposedLayout,
    pub v_smem_layout_staged: &'a ComposedLayout,
    pub o_smem_layout_staged: &'a ComposedLayout,
    pub q_copy_bytes: usize,
    pub kv_copy_bytes: usize,
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

// /////////////////////////////////////////////////////////////////////////////
//  Host data, reference and run
// /////////////////////////////////////////////////////////////////////////////

/// One problem: `(batch, s_q, heads, head_dim)` queries against
/// `(batch, s_k, heads_k, head_dim)` keys and values.
#[derive(Clone, Copy, Debug)]
pub struct Problem {
    pub b: usize,
    pub s_q: usize,
    pub s_k: usize,
    pub h: usize,
    pub h_k: usize,
    pub d: usize,
    pub seed: u64,
    pub is_causal: bool,
    pub bottom_right_align: bool,
    pub window: (Option<i32>, Option<i32>),
}

impl Problem {
    pub fn new(b: usize, s_q: usize, s_k: usize, h: usize, h_k: usize, d: usize) -> Problem {
        Problem {
            b,
            s_q,
            s_k,
            h,
            h_k,
            d,
            seed: 1234,
            is_causal: false,
            bottom_right_align: false,
            window: (None, None),
        }
    }

    pub fn causal(self) -> Problem {
        Problem { is_causal: true, ..self }
    }

    /// The window sizes the kernel gets (causal is `window_right = 0`).
    pub fn window_sizes(&self) -> (Option<i32>, Option<i32>) {
        let (l, r) = self.window;
        (l, if self.is_causal { Some(0) } else { r })
    }

    /// The mask type and window sides `run` picks for the problem.
    pub fn mask(&self, tile_n: usize) -> (MaskEnum, bool, bool) {
        let (l, r) = self.window_sizes();
        let mut mask_type = if self.bottom_right_align { MaskEnum::WindowMaskInference } else { MaskEnum::WindowMask };
        if l.is_none() && r.is_none() && !self.s_k.is_multiple_of(tile_n) {
            mask_type = MaskEnum::ResidualMask;
        }
        (mask_type, l.is_some(), r.is_some())
    }

    /// `1 / sqrt(d)`, the softmax scale `run` defaults to.
    pub fn scale_softmax(&self) -> f32 {
        1.0 / (self.d as f32).sqrt()
    }
}

/// The kernel configuration `run` makes for `problem`.
pub fn fmha_for(
    problem: &Problem,
    mma_tiler_mn: (usize, usize),
    in_dtype: DType,
    out_dtype: DType,
    is_persistent: bool,
) -> HopperFusedMultiHeadAttentionForward {
    let (mask_type, wl, wr) = problem.mask(mma_tiler_mn.1);
    HopperFusedMultiHeadAttentionForward::new(
        DType::F32,
        DType::F32,
        (mma_tiler_mn.0, mma_tiler_mn.1, problem.d),
        is_persistent,
        mask_type,
        wl,
        wr,
    )
    .setup(in_dtype, out_dtype)
}

/// Deterministic values in [-1, 1) for a tensor of `len` elements.
fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 33) % 128) as f32 / 64.0 - 1.0
        })
        .collect()
}

/// A buffer of `values` as `dtype`.
pub fn buffer(dtype: DType, values: &[f32]) -> KernelArg {
    match dtype {
        DType::F16 => KernelArg::f16_from_f32(values),
        DType::BF16 => KernelArg::buffer(&values.iter().map(|&x| BF16::from_f32(x)).collect::<Vec<_>>()),
        other => panic!("hopper_fmha: no buffer of {other}"),
    }
}

/// The values of a buffer of `dtype` as f64.
fn read(arg: &KernelArg) -> Vec<f64> {
    arg.to_f32().into_iter().map(f64::from).collect()
}

/// The host values of Q, K and V as the device holds them (rounded to the
/// input type), in memory order.
fn host_qkv(fmha: &HopperFusedMultiHeadAttentionForward, p: &Problem) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let held = |len: usize, seed: u64| -> Vec<f32> {
        read(&buffer(fmha.q_dtype, &values(len, seed))).into_iter().map(|x| x as f32).collect()
    };
    (
        held(p.b * p.s_q * p.h * p.d, p.seed),
        held(p.b * p.s_k * p.h_k * p.d, p.seed + 1),
        held(p.b * p.s_k * p.h_k * p.d, p.seed + 2),
    )
}

/// The arguments of the kernel for `problem`.
pub fn inputs(fmha: &HopperFusedMultiHeadAttentionForward, p: &Problem) -> Vec<KernelArg> {
    let (q, k, v) = host_qkv(fmha, p);
    let log2_e = std::f32::consts::LOG2_E;
    let scale = p.scale_softmax();
    let mut args = vec![
        KernelArg::ScalarI32(p.s_q as i32),
        KernelArg::ScalarI32(p.s_k as i32),
        KernelArg::ScalarI32((p.h / p.h_k) as i32),
        KernelArg::ScalarI32(p.h_k as i32),
        KernelArg::ScalarI32(p.b as i32),
        buffer(fmha.q_dtype, &q),
        buffer(fmha.q_dtype, &k),
        buffer(fmha.q_dtype, &v),
        buffer(fmha.o_dtype, &vec![0.0; p.b * p.s_q * p.h * p.d]),
        KernelArg::f32(vec![0.0; p.b * p.h * p.s_q]),
        KernelArg::ScalarF32(scale * log2_e),
        KernelArg::ScalarF32(scale),
        KernelArg::ScalarF32(1.0),
    ];
    let (l, r) = p.window_sizes();
    if let Some(l) = l.filter(|_| fmha.has_window_left) {
        args.push(KernelArg::ScalarI32(l));
    }
    if let Some(r) = r.filter(|_| fmha.has_window_right) {
        args.push(KernelArg::ScalarI32(r));
    }
    args
}

/// Attention in f64 over the device's inputs (`run_torch_fmha`): the
/// output `(b, s_q, h, d)` and the log-sum-exp `(b, h, s_q)`.
pub fn reference(fmha: &HopperFusedMultiHeadAttentionForward, p: &Problem) -> (Vec<f64>, Vec<f64>) {
    let (q, k, v) = host_qkv(fmha, p);
    let (l, r) = p.window_sizes();
    let offset = if p.bottom_right_align { p.s_k as i64 - p.s_q as i64 } else { 0 };
    let scale = f64::from(p.scale_softmax());
    let mut o = vec![0.0; p.b * p.s_q * p.h * p.d];
    let mut lse = vec![0.0; p.b * p.h * p.s_q];
    let h_r = p.h / p.h_k;
    for b in 0..p.b {
        for h in 0..p.h {
            let hk = h / h_r;
            for i in 0..p.s_q {
                let qrow = &q[((b * p.s_q + i) * p.h + h) * p.d..][..p.d];
                let mut scores = vec![f64::NEG_INFINITY; p.s_k];
                for (j, s) in scores.iter_mut().enumerate() {
                    let (ii, jj) = (i as i64, j as i64);
                    let masked = r.is_some_and(|r| jj > ii + offset + i64::from(r))
                        || l.is_some_and(|l| jj < ii + offset - i64::from(l));
                    if !masked {
                        let krow = &k[((b * p.s_k + j) * p.h_k + hk) * p.d..][..p.d];
                        *s = qrow.iter().zip(krow).map(|(&a, &c)| f64::from(a) * f64::from(c)).sum::<f64>() * scale;
                    }
                }
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = scores.iter().map(|&s| (s - max).exp()).sum();
                lse[(b * p.h + h) * p.s_q + i] = max + sum.ln();
                for (j, &s) in scores.iter().enumerate() {
                    let w = (s - max).exp() / sum;
                    if w != 0.0 {
                        let vrow = &v[((b * p.s_k + j) * p.h_k + hk) * p.d..][..p.d];
                        for (dd, &x) in vrow.iter().enumerate() {
                            o[((b * p.s_q + i) * p.h + h) * p.d + dd] += w * f64::from(x);
                        }
                    }
                }
            }
        }
    }
    (o, lse)
}

/// Check O and the log-sum-exp in `args` against [`reference`] (`|got -
/// want| <= 0.01 + 0.01 |want|` for O, which covers P's and O's rounding to
/// 16 bits, `1e-3 (1 + |want|)` for the Float32 log-sum-exp).
pub fn check(fmha: &HopperFusedMultiHeadAttentionForward, p: &Problem, args: &[KernelArg]) {
    let (want_o, want_lse) = reference(fmha, p);
    let got_o = read(&args[8]);
    let got_lse = args[9].to_f32();
    let name = fmha.name();
    let mut nonzero = 0;
    for (i, (&g, &w)) in got_o.iter().zip(&want_o).enumerate() {
        assert!((g - w).abs() <= 0.01 + 0.01 * w.abs(), "{name} {p:?}: O element {i}: got {g}, want {w}");
        nonzero += usize::from(w != 0.0);
    }
    for (i, (&g, &w)) in got_lse.iter().zip(&want_lse).enumerate() {
        let g = f64::from(g);
        assert!((g - w).abs() <= 1e-3 * (1.0 + w.abs()), "{name} {p:?}: lse element {i}: got {g}, want {w}");
    }
    assert!(nonzero * 2 > want_o.len(), "{name}: most of O is 0");
}

/// Simulate `kernel` (built from `fmha`) on `problem` and check it.
pub fn run(fmha: &HopperFusedMultiHeadAttentionForward, kernel: &Kernel, p: &Problem) -> SimReport {
    assert_eq!(p.d, fmha.cta_tiler.2, "the head dimension is the MMA tile's K");
    assert!(p.h.is_multiple_of(p.h_k), "h must be divisible by h_k");
    let mut args = inputs(fmha, p);
    let report = kernel
        .simulate(fmha.launch(p.s_q, p.h, p.b, SIM_SMS), &mut args)
        .unwrap_or_else(|e| panic!("{} {p:?}: {e}", kernel.name()));
    check(fmha, p, &args);
    report
}

/// The number of SMs the simulated persistent grids assume (a real run reads
/// `HardwareInfo::get_device_multiprocessor_count`).
pub const SIM_SMS: usize = 3;

fn main() {
    println!("Running Hopper FMHA forward (in the simulator) with:");
    let p = Problem::new(1, 128, 200, 2, 1, 64).causal();
    println!("{p:?}");
    let fmha = fmha_for(&p, (64, 128), DType::F16, DType::F16, false);
    println!("{} threads, tile {:?}, epilogue tile {:?}", fmha.threads_per_cta, fmha.cta_tiler, fmha.epi_tile);
    let kernel = fmha.build();
    let report = run(&fmha, &kernel, &p);
    println!("{}: correct\n{report}\n", kernel.name());

    // Persistent, grouped-query heads, a residual tile.
    let p = Problem::new(2, 130, 200, 4, 2, 64);
    let fmha = fmha_for(&p, (64, 128), DType::F16, DType::F16, true);
    let kernel = fmha.build();
    run(&fmha, &kernel, &p);
    println!("{} on {p:?}: correct", kernel.name());

    match Toolchain::discover() {
        Some(tc) => match kernel.compile_ptx(&tc, Arch::Sm90a) {
            Ok(ptx) => {
                let count = |pat: &str| ptx.lines().filter(|l| l.contains(pat)).count();
                println!(
                    "PTX (sm_90a): {} lines; {} wgmma.mma_async, {} try_wait, {} setmaxnreg, {} stmatrix, {} TMA stores",
                    ptx.lines().count(),
                    count("wgmma.mma_async"),
                    count("mbarrier.try_wait.parity"),
                    count("setmaxnreg"),
                    count("stmatrix"),
                    count("cp.async.bulk.tensor.4d.global.shared"),
                );
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}
