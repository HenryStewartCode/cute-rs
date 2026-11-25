// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//
// Portions of this file are translated from FlashInfer v0.7.5:
//
// * `include/flashinfer/attention/blackwell/plan.cuh` (`plan_kernel`, here the host
//   [`Plan::new`]), Copyright (c) 2025 by FlashInfer team, Apache-2.0;
// * the CUTLASS SM100 FMHA forward pass FlashInfer ships in
//   `include/flashinfer/attention/blackwell/`: `kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`
//   (the warp roles and register budgets), `collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`
//   (the TMEM allocation, the pipelines, the MMA, softmax and correction loops),
//   `collective/sm100_fmha_load_tma_warpspecialized.hpp` (the load order),
//   `collective/fmha_fusion.hpp` (`ResidualMask`, `CausalMask`) and
//   `kernel/fmha_tile_scheduler.hpp` (`HostPrecomputedTileScheduler`),
//   Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are met:
//
//   1. Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
//   2. Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
//   3. Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// How some cute-rs calls are put together (the tcgen05 tensors over TMEM, the TMA
// partitions over a `domain_offset` of the varlen tensors) follows NVIDIA CUTLASS 4.8.1's
// `examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py`, under the same
// BSD-3-Clause licence and copyright holder (Copyright (c) 2025 - 2026). The rest of the
// file is the crate's own, under Apache-2.0.

//! FlashInfer's Blackwell (SM100) prefill: the CUTLASS SM100 FMHA forward kernel behind
//! `BatchPrefillWithRaggedKVCacheWrapper(backend="cutlass")`, `fmha_varlen` and
//! `single_prefill_with_kv_cache(backend="cutlass")` on B200, translated to cute-rs.
//!
//! Ragged (varlen) Q, K and V (`[total, heads, 128]`, NHD, f16 or bf16), `qo_indptr` and
//! `kv_indptr` segment offsets, GQA (`num_qo_heads` a multiple of `num_kv_heads`), causal
//! (bottom-right aligned, FlashInfer's convention: query row `i` of a request sees keys
//! `j <= i + kv_len - qo_len`) or not; output `o` in the input type and the base-2
//! log-sum-exp `lse [total_q, num_qo_heads]`.
//!
//! The kernel is FlashInfer's (CUTLASS's) warp-specialised design, one persistent CTA of
//! 512 threads per SM walking the work list [`Plan`] (FlashInfer's `plan_kernel`, here on the
//! host): a work item is a 256-row Q tile of one head of one request, split into two 128-row
//! halves.
//!
//! * warp 13 loads with TMA, Q0, K0, Q1, V0, then K1, V1, K2, V2, ..., through a 2-stage Q
//!   pipeline and 2-stage K and V pipelines (`PipelineTmaUmma`), the varlen offsets applied
//!   with `domain_offset` on the TMA tensors;
//! * warp 12 issues the `tcgen05.mma`s: `S0 = Q0 K^T`, `S1 = Q1 K^T` (A and B in shared
//!   memory) and `O0 += P0 V`, `O1 += P1 V` with `P` read from **tensor memory** (TS MMA),
//!   in CUTLASS's order (`Q0 K1, Q1 K1, P0 V1, Q0 K2, P1 V1, Q1 K2, P0 V2, ...`);
//! * warps 0-3 and 4-7 (two softmax warpgroups, 192 registers) own S0 and S1: each thread
//!   reads its row of S from TMEM (`tcgen05.ld.32x32b`), masks, takes the running row max,
//!   hands `(old max, new max)` to the correction warpgroup through TMEM, computes
//!   `exp2(s * scale - max * scale)` with packed `fma.f32x2`, converts to f16/bf16 pairs and
//!   writes `P` over its own S columns (`tcgen05.st`); an ordered barrier alternates the two
//!   warpgroups' exponentials as CUTLASS's `OrderedSequenceBarrier`;
//! * warps 8-11 (the correction warpgroup) rescale `O` in TMEM when the row max grows, and at
//!   the end of a tile normalise `O` by the row sum and write `O` (16-byte stores, rows past
//!   the request's end skipped) and `lse`;
//! * TMEM (512 columns): S0, S1, O0, O1 of 128 columns each; P0 and P1 sit at columns 32-95
//!   of S0 and S1, and the row statistics at columns 0-1.
//!
//! Differences from FlashInfer: the plan runs on the host; the correction warpgroup stores
//! `O` straight to global memory (CUTLASS stages it in shared memory for a TMA store by an
//! epilogue warp, which with FlashInfer's per-request clipping needs its offset O tensor
//! trick), so there is no epilogue warp; head dim 128, f16/bf16 only (no FP8, no
//! `q/k/v/o_scale`); every request needs `kv_len > 0`, and a causal request
//! `qo_len <= kv_len` (as in prefill, where they are equal).
//!
//! The cute-rs simulator models tcgen05 (MMA, TMEM loads and stores, commits), so the
//! kernel is checked there against an f64 reference, and on a B200 again.

use cute::cpasync;
use cute::memory::{SmemAllocator, SmemPartition, TmemAllocator, TmemAllocatorOptions};
use cute::nvgpu::{make_tiled_tma_atom_a, make_tiled_tma_atom_b};
use cute::pipeline::{
    Agent, CooperativeGroup, NamedBarrier, Pipeline, PipelineAsync, PipelineAsyncOptions, PipelineBarriers,
    PipelineOrder, PipelineOrderOptions, PipelineState, PipelineTmaUmma, PipelineTmaUmmaOptions, PipelineUmmaAsync,
    PipelineUmmaOptions, PipelineUserType, make_pipeline_state,
};
use cute::prelude::*;
use cute::tcgen05::{CtaGroup, Field, Ld32x32bOp, OperandSource, Repetition, St32x32bOp};
use cute::utils::sm100::{Sm100TmaLoadOp, make_smem_layout_a, make_smem_layout_b, make_trivial_tiled_mma};

/// Head dimension (of Q, K and V).
pub const HEAD_DIM: usize = 128;
/// Query rows of a work item (CUTLASS's `TileShapeQK` M): two 128-row halves.
pub const TILE_Q: usize = 256;
/// Query rows of one half, the M of each MMA.
pub const MMA_M: usize = 128;
/// Keys per K/V tile (the N of `Q K^T`, the K of `P V`).
pub const TILE_KV: usize = 128;
/// Threads per CTA: 16 warps.
pub const THREADS: usize = 512;
const Q_STAGES: usize = 2;
const KV_STAGES: usize = 2;
/// Register budgets (CUTLASS's `NumRegsSoftmax`; the others as CuTe DSL's fmha.py).
const REGS_SOFTMAX: u32 = 192;
const REGS_CORRECTION: u32 = 96;
const REGS_OTHER: u32 = 32;
/// TMEM columns (`TmemAllocation`).
const TMEM_S: [i32; 2] = [0, 128];
const TMEM_O: [i32; 2] = [256, 384];
const TMEM_P: [i32; 2] = [32, 160];
const TMEM_V: [i32; 2] = [0, 128];
/// One TMEM lane in a TMEM address.
const LANE: i32 = 1 << 16;
const WARP_MMA: i32 = 12;
const WARP_LOAD: i32 = 13;

/// What a kernel is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub causal: bool,
    /// BFloat16 Q, K, V and O (else Float16).
    pub bf16: bool,
}

impl Config {
    pub fn name(&self) -> String {
        format!(
            "prefill_sm100_{}_{}_h{}x{}",
            if self.bf16 { "bf16" } else { "f16" },
            if self.causal { "causal" } else { "full" },
            self.num_qo_heads,
            self.num_kv_heads
        )
    }

    /// The kernel. Its parameters are in the order of [`p`].
    pub fn kernel(&self) -> Kernel {
        assert!(
            self.num_kv_heads > 0 && self.num_qo_heads.is_multiple_of(self.num_kv_heads),
            "prefill_sm100: num_qo_heads must be a multiple of num_kv_heads"
        );
        if self.bf16 { build::<BF16>(*self) } else { build::<F16>(*self) }
    }
}

/// Kernel argument positions.
pub mod p {
    pub const TOTAL_Q: usize = 0;
    pub const TOTAL_KV: usize = 1;
    pub const BATCH_PLUS_1: usize = 2;
    pub const CTAS_PLUS_1: usize = 3;
    pub const WORK_ITEMS: usize = 4;
    pub const Q: usize = 5;
    pub const K: usize = 6;
    pub const V: usize = 7;
    pub const O: usize = 8;
    pub const LSE: usize = 9;
    pub const QO_INDPTR: usize = 10;
    pub const KV_INDPTR: usize = 11;
    pub const WORK_INDPTR: usize = 12;
    pub const QO_TILE_INDICES: usize = 13;
    pub const HEAD_INDICES: usize = 14;
    pub const BATCH_INDICES: usize = 15;
    pub const SM_SCALE_LOG2: usize = 16;
}

fn ceil_div(a: Int32, b: i32) -> Int32 {
    (a + (b - 1)) / b
}

/// A work item's coordinates and its K/V trip counts (the masks' `get_trip_count` and
/// `get_masked_trip_count`).
#[derive(Clone, Copy)]
struct Work {
    tile: Int32,
    h: Int32,
    h_kv: Int32,
    qs: Int32,
    ql: Int32,
    ks: Int32,
    kl: Int32,
    trip: Int32,
    masked: Int32,
}

fn build<T: Numeric>(cfg: Config) -> Kernel {
    let name = cfg.name();
    Kernel::builder(&name)
        .block_dim([THREADS as u32, 1, 1])
        .min_blocks_per_sm(1)
        .target(Arch::Sm100a)
        .build(move || kernel_body::<T>(cfg))
}

#[allow(clippy::too_many_lines)]
fn kernel_body<T: Numeric>(cfg: Config) {
    let (hq, hkv, d) = (cfg.num_qo_heads, cfg.num_kv_heads, HEAD_DIM);
    let group = (hq / hkv) as i32;
    let total_q = IndexExpr::from(param_extent("total_q", 1));
    let total_kv = IndexExpr::from(param_extent("total_kv", 1));
    let batch1 = IndexExpr::from(param_extent("batch_plus_1", 1));
    let ctas1 = IndexExpr::from(param_extent("ctas_plus_1", 1));
    let works = IndexExpr::from(param_extent("work_items", 1));
    let ix = IndexExpr::from;
    // Q (M, K, L) = (total_q, D, Hq) and K (N, K, L) = (total_kv, D, Hkv), K-major; V as
    // (N, K, L) = (D, total_kv, Hkv), MN-major.
    let q =
        param_tensor_aligned::<T>("q", Layout::new((total_q.clone(), ix(d), ix(hq)), (ix(hq * d), ix(1), ix(d))), 16);
    let k = param_tensor_aligned::<T>(
        "k",
        Layout::new((total_kv.clone(), ix(d), ix(hkv)), (ix(hkv * d), ix(1), ix(d))),
        16,
    );
    let v = param_tensor_aligned::<T>(
        "v",
        Layout::new((ix(d), total_kv.clone(), ix(hkv)), (ix(1), ix(hkv * d), ix(d))),
        16,
    );
    let o = param_tensor_aligned::<T>("o", Layout::new((total_q.clone(), ix(hq * d)), (ix(hq * d), ix(1))), 16);
    let lse = param_tensor::<f32>("lse", Layout::new((total_q, ix(hq)), (ix(hq), ix(1))));
    let qo_indptr = param_tensor::<i32>("qo_indptr", Layout::col_major(batch1.clone()));
    let kv_indptr = param_tensor::<i32>("kv_indptr", Layout::col_major(batch1));
    let work_indptr = param_tensor::<i32>("work_indptr", Layout::col_major(ctas1));
    let qo_tile_indices = param_tensor::<i32>("qo_tile_indices", Layout::col_major(works.clone()));
    let head_indices = param_tensor::<i32>("head_indices", Layout::col_major(works.clone()));
    let batch_indices = param_tensor::<i32>("batch_indices", Layout::col_major(works));
    let sm_scale_log2 = param_scalar::<f32>("sm_scale_log2");

    // The tiled MMAs: Q K^T with both operands in shared memory, P V with P in TMEM.
    let kmaj = OperandMajorMode::K;
    let qk_mma = make_trivial_tiled_mma(
        T::DTYPE,
        T::DTYPE,
        kmaj,
        kmaj,
        DType::F32,
        CtaGroup::ONE,
        (MMA_M, TILE_KV),
        OperandSource::SMEM,
    );
    let pv_mma = make_trivial_tiled_mma(
        T::DTYPE,
        T::DTYPE,
        kmaj,
        OperandMajorMode::MN,
        DType::F32,
        CtaGroup::ONE,
        (MMA_M, HEAD_DIM),
        OperandSource::TMEM,
    );
    let qk_tiler = (MMA_M, TILE_KV, HEAD_DIM);
    let pv_tiler = (MMA_M, HEAD_DIM, TILE_KV);
    let q_smem = make_smem_layout_a(&qk_mma, qk_tiler, T::DTYPE, Q_STAGES, None);
    let k_smem = make_smem_layout_b(&qk_mma, qk_tiler, T::DTYPE, KV_STAGES, None);
    let v_smem = make_smem_layout_b(&pv_mma, pv_tiler, T::DTYPE, KV_STAGES, None);
    let tma_op = Sm100TmaLoadOp::Tile(cpasync::CopyBulkTensorTileG2SOp { cta_group: CtaGroup::ONE });
    let (tma_q, m_q): (CopyAtom<T>, Tensor<Ident>) =
        make_tiled_tma_atom_a(tma_op, &q, q_smem.select(&[0, 1, 2]), qk_tiler, &qk_mma, None).into();
    let (tma_k, m_k): (CopyAtom<T>, Tensor<Ident>) =
        make_tiled_tma_atom_b(tma_op, &k, k_smem.select(&[0, 1, 2]), qk_tiler, &qk_mma, None).into();
    let (tma_v, m_v): (CopyAtom<T>, Tensor<Ident>) =
        make_tiled_tma_atom_b(tma_op, &v, v_smem.select(&[0, 1, 2]), pv_tiler, &pv_mma, None).into();
    let bytes = |l: &ComposedLayout| size_in_bytes::<T>(&l.select(&[0, 1, 2])).as_static().expect("static tile") as u32;
    let (q_bytes, k_bytes, v_bytes) = (bytes(&q_smem), bytes(&k_smem), bytes(&v_smem));

    let tidx = thread_idx().x;
    let warp = make_warp_uniform(warp_idx());
    let cta = block_idx().x;

    let smem = SmemAllocator::new();
    let s_q = smem.allocate_tensor::<T>(q_smem.clone(), 1024, None, SmemPartition::User);
    let s_k = smem.allocate_tensor::<T>(k_smem.clone(), 1024, None, SmemPartition::User);
    let s_v = smem.allocate_tensor::<T>(v_smem.clone(), 1024, None, SmemPartition::User);

    // The pipelines (`PipelineQ/K/V/S/C/O` and `OrderBarrierSoftmax`).
    let one = || CooperativeGroup::new(Agent::Thread, 1);
    let wg = || CooperativeGroup::new(Agent::Thread, 128);
    let tma_pipe =
        |stages: usize, tx: u32| PipelineTmaUmma::create(stages, one(), one(), tx, PipelineTmaUmmaOptions::default());
    let pipe_q = tma_pipe(Q_STAGES, q_bytes);
    let pipe_k = tma_pipe(KV_STAGES, k_bytes);
    let pipe_v = tma_pipe(KV_STAGES, v_bytes);
    let umma_pipe = |stages: usize| PipelineUmmaAsync::create(stages, one(), wg(), PipelineUmmaOptions::default());
    let pipe_s = [umma_pipe(1), umma_pipe(1)];
    let pipe_o = umma_pipe(2);
    let async_pipe = || PipelineAsync::create(1, wg(), wg(), PipelineAsyncOptions::default());
    let pipe_c = [async_pipe(), async_pipe()];
    let order = PipelineOrder::create(1, 2, select(warp.lt(4), 0, 1), wg(), PipelineOrderOptions::default());

    // All 512 TMEM columns, allocated by warp 0 for everyone.
    let mut tmem = TmemAllocator::new(
        NamedBarrier::new(1, THREADS as i32),
        TmemAllocatorOptions { arch: Arch::Sm100a, ..Default::default() },
    );
    tmem.allocate(512);
    tmem.wait_for_alloc();
    tmem.relinquish_alloc_permit();
    let tmem_f32 = tmem.retrieve_ptr::<f32>();
    let tmem_i32 = tmem.retrieve_ptr::<i32>();
    let tmem_t = tmem.retrieve_ptr::<T>();

    // The persistent work list of this CTA (`HostPrecomputedTileScheduler`).
    let w_beg = work_indptr.load(cta);
    let w_end = work_indptr.load(cta + 1);
    let work = |w: Int32| -> Work {
        let tile = qo_tile_indices.load(w);
        let h = head_indices.load(w);
        let b = batch_indices.load(w);
        let qs = qo_indptr.load(b);
        let ql = qo_indptr.load(b + 1) - qs;
        let ks = kv_indptr.load(b);
        let kl = kv_indptr.load(b + 1) - ks;
        let kv_tiles = ceil_div(kl, TILE_KV as i32);
        let (trip, masked) = if cfg.causal {
            // CausalMask: offset_q = kv_len - qo_len.
            let max_q = ceil_div((tile + 1) * TILE_Q as i32 + (kl - ql), TILE_KV as i32);
            let trip = kv_tiles.min(max_q).max(1);
            (trip, trip.min((TILE_Q / TILE_KV) as i32 + 1))
        } else {
            // ResidualMask: the last tile is masked when kv_len is not a multiple of it.
            (kv_tiles, select((kl % TILE_KV as i32).ne(0), 1, 0))
        };
        Work { tile, h, h_kv: h / group, qs, ql, ks, kl, trip, masked }
    };
    // Take a pipeline state and advance the one kept in `v`.
    let take = |v: &Var<PipelineState>| -> PipelineState {
        let s = v.get();
        let mut n = s;
        n.advance();
        v.set(n);
        s
    };

    // MMA fragments.
    let qk_thr = qk_mma.get_slice(0);
    let pv_thr = pv_mma.get_slice(0);
    let t_qa = qk_mma.make_fragment_a(&s_q);
    let t_kb = qk_mma.make_fragment_b(&s_k);
    let t_vb = pv_mma.make_fragment_b(&s_v);
    let acc_s = qk_mma.make_fragment_c::<f32>(&qk_mma.partition_shape_c((MMA_M, TILE_KV))).layout().clone();
    let acc_o = pv_mma.make_fragment_c::<f32>(&pv_mma.partition_shape_c((MMA_M, HEAD_DIM))).layout().clone();
    let frg_p = pv_mma.make_tmem_fragment_a::<T>(&pv_mma.partition_shape_a((MMA_M, TILE_KV))).layout().clone();
    let t_s = [0, 1].map(|i| make_tensor(tmem_f32.clone() + TMEM_S[i], acc_s.clone()));
    let t_o = [0, 1].map(|i| make_tensor(tmem_f32.clone() + TMEM_O[i], acc_o.clone()));
    let t_p = [0, 1].map(|i| make_tensor(tmem_t.clone() + TMEM_P[i] * 2, frg_p.clone()));
    let k_blocks = HEAD_DIM / 16;
    let pv_blocks = TILE_KV / 16;

    // ---------------------------------------------------------------------------------
    // Load warp
    // ---------------------------------------------------------------------------------
    if_(warp.eq(WARP_LOAD), || {
        setmaxregister_decrease(REGS_OTHER);
        cpasync::prefetch_descriptor(&tma_q);
        cpasync::prefetch_descriptor(&tma_k);
        cpasync::prefetch_descriptor(&tma_v);
        let q_st = Var::new(make_pipeline_state(PipelineUserType::Producer, Q_STAGES));
        let k_st = Var::new(make_pipeline_state(PipelineUserType::Producer, KV_STAGES));
        let v_st = Var::new(make_pipeline_state(PipelineUserType::Producer, KV_STAGES));
        let load =
            |pipe: &PipelineTmaUmma, st: &Var<PipelineState>, atom: &CopyAtom<T>, g: Tensor<Ident>, s: &Tensor<T>| {
                let state = take(st);
                pipe.producer_acquire(&state, None);
                let bar = pipe.producer_get_barrier(&state);
                copy_with(atom, &g, &s.slice((.., state.index())), CopyArgs::new().tma_bar_ptr(&bar));
            };
        for_range(w_beg, w_end, 1, |w| {
            let wk = work(w);
            let zero = Int32::from(0);
            let m_q_b = domain_offset((wk.qs, zero, zero), &m_q);
            let g_q = local_tile(&m_q_b, (MMA_M, HEAD_DIM), (.., 0, wk.h));
            let (t_qs, t_qg) = cpasync::tma_partition(
                &tma_q,
                0,
                &make_layout(1, 1),
                &s_q.group_modes(0, 3),
                &qk_thr.partition_a(&g_q).group_modes(0, 3),
            );
            let m_k_b = domain_offset((wk.ks, zero, zero), &m_k);
            let g_k = local_tile(&m_k_b, (TILE_KV, HEAD_DIM), (.., 0, wk.h_kv));
            let (t_ks, t_kg) = cpasync::tma_partition(
                &tma_k,
                0,
                &make_layout(1, 1),
                &s_k.group_modes(0, 3),
                &qk_thr.partition_b(&g_k).group_modes(0, 3),
            );
            let m_v_b = domain_offset((zero, wk.ks, zero), &m_v);
            let g_v = local_tile(&m_v_b, (HEAD_DIM, TILE_KV), (0, .., wk.h_kv));
            let (t_vs, t_vg) = cpasync::tma_partition(
                &tma_v,
                0,
                &make_layout(1, 1),
                &s_v.group_modes(0, 3),
                &pv_thr.partition_b(&g_v).group_modes(0, 3),
            );
            // Q1, K1, Q2, V1, then K2, V2, K3, V3, ...
            load(&pipe_q, &q_st, &tma_q, t_qg.slice((.., wk.tile * 2)), &t_qs);
            load(&pipe_k, &k_st, &tma_k, t_kg.slice((.., 0)), &t_ks);
            load(&pipe_q, &q_st, &tma_q, t_qg.slice((.., wk.tile * 2 + 1)), &t_qs);
            load(&pipe_v, &v_st, &tma_v, t_vg.slice((.., 0)), &t_vs);
            for_range(1, wk.trip, 1, |i| {
                load(&pipe_k, &k_st, &tma_k, t_kg.slice((.., i)), &t_ks);
                load(&pipe_v, &v_st, &tma_v, t_vg.slice((.., i)), &t_vs);
            });
        });
    });

    // ---------------------------------------------------------------------------------
    // MMA warp
    // ---------------------------------------------------------------------------------
    if_(warp.eq(WARP_MMA), || {
        setmaxregister_decrease(REGS_OTHER);
        let consumer = |stages| Var::new(make_pipeline_state(PipelineUserType::Consumer, stages));
        let producer = |stages| Var::new(make_pipeline_state(PipelineUserType::Producer, stages));
        let (q_c, q_r) = (consumer(Q_STAGES), consumer(Q_STAGES));
        let (k_c, k_r) = (consumer(KV_STAGES), consumer(KV_STAGES));
        let (v_c, v_r) = (consumer(KV_STAGES), consumer(KV_STAGES));
        let s_p = [producer(1), producer(1)];
        let o_p = producer(2);
        let wait = |pipe: &PipelineTmaUmma, st: &Var<PipelineState>| -> Int32 {
            let s = take(st);
            pipe.consumer_wait(&s, None);
            s.index()
        };
        let release = |pipe: &PipelineTmaUmma, st: &Var<PipelineState>| pipe.consumer_release(&take(st));
        let acquire = |pipe: &PipelineUmmaAsync, st: &Var<PipelineState>| pipe.producer_acquire(&st.get(), None);
        let commit = |pipe: &PipelineUmmaAsync, st: &Var<PipelineState>| pipe.producer_commit(&take(st));
        // gemm_zero_acc of Q_i K^T into S_i.
        let qk = |i: usize, qi: Int32, ki: Int32| {
            let mut m = qk_mma.clone();
            for kb in 0..k_blocks {
                m.set(Field::ACCUMULATE, kb != 0);
                gemm(&m, &t_s[i], &t_qa.slice((.., .., kb, qi)), &t_kb.slice((.., .., kb, ki)), &t_s[i]);
            }
        };
        // P_i V into O_i, accumulating from the first k-block when `acc`.
        let pv = |i: usize, vi: Int32, acc: Boolean| {
            let mut m = pv_mma.clone();
            for kb in 0..pv_blocks {
                if kb == 0 {
                    m.set(Field::ACCUMULATE, acc);
                } else {
                    m.set(Field::ACCUMULATE, true);
                }
                gemm(&m, &t_o[i], &t_p[i].slice((.., .., kb)), &t_vb.slice((.., .., kb, vi)), &t_o[i]);
            }
        };
        let (no, yes) = (Boolean::constant(false), Boolean::constant(true));
        for_range(w_beg, w_end, 1, |w| {
            let wk = work(w);
            let q0 = wait(&pipe_q, &q_c);
            let k0 = wait(&pipe_k, &k_c);
            // Q1 * K1 -> S1
            acquire(&pipe_s[0], &s_p[0]);
            qk(0, q0, k0);
            commit(&pipe_s[0], &s_p[0]);
            // Q2 * K1 -> S2
            let q1 = wait(&pipe_q, &q_c);
            acquire(&pipe_s[1], &s_p[1]);
            qk(1, q1, k0);
            commit(&pipe_s[1], &s_p[1]);
            release(&pipe_k, &k_r);
            // P1 * V1 -> O1 (the S0 acquire returns P0 to the MMA warp)
            let v0 = wait(&pipe_v, &v_c);
            acquire(&pipe_o, &o_p);
            acquire(&pipe_s[0], &s_p[0]);
            pv(0, v0, no);
            commit(&pipe_o, &o_p);
            let v_idx = Var::new(v0);
            let pv1_acc = Var::new(no);
            for_range(1, wk.trip, 1, |_| {
                // Q1 * Ki -> S1
                let ki = wait(&pipe_k, &k_c);
                qk(0, q0, ki);
                commit(&pipe_s[0], &s_p[0]);
                // P2 * V(i-1) -> O2
                acquire(&pipe_o, &o_p);
                acquire(&pipe_s[1], &s_p[1]);
                pv(1, v_idx.get(), pv1_acc.get());
                pv1_acc.set(yes);
                commit(&pipe_o, &o_p);
                release(&pipe_v, &v_r);
                // Q2 * Ki -> S2
                qk(1, q1, ki);
                commit(&pipe_s[1], &s_p[1]);
                release(&pipe_k, &k_r);
                // P1 * Vi -> O1
                let vi = wait(&pipe_v, &v_c);
                v_idx.set(vi);
                acquire(&pipe_o, &o_p);
                acquire(&pipe_s[0], &s_p[0]);
                pv(0, vi, yes);
                commit(&pipe_o, &o_p);
            });
            release(&pipe_q, &q_r);
            release(&pipe_q, &q_r);
            // P2 * Vi -> O2
            acquire(&pipe_o, &o_p);
            acquire(&pipe_s[1], &s_p[1]);
            pv(1, v_idx.get(), pv1_acc.get());
            commit(&pipe_o, &o_p);
            release(&pipe_v, &v_r);
            // Hand S0 and S1 back to the softmax warpgroups for the final statistics.
            commit(&pipe_s[0], &s_p[0]);
            commit(&pipe_s[1], &s_p[1]);
        });
    });

    // The two idle warps of the fourth warpgroup.
    if_(warp.ge(14), || setmaxregister_decrease(REGS_OTHER));

    // A thread's row of a 128-lane TMEM tensor: `cols` columns from column `col`.
    let t = tidx % 128;
    let lane_base = (warp % 4) * (32 * LANE);
    let row_view = |base: &Ptr<f32>, col: i32, cols: usize| {
        make_tensor(base.clone() + (lane_base + col), Layout::from_modes([Layout::new((cols, 32), (1, LANE))]))
    };
    let row_view_i32 = |col: i32, cols: usize| {
        make_tensor(tmem_i32.clone() + (lane_base + col), Layout::from_modes([Layout::new((cols, 32), (1, LANE))]))
    };
    let rep = |n: usize| match n {
        1 => Repetition::X1,
        2 => Repetition::X2,
        4 => Repetition::X4,
        8 => Repetition::X8,
        16 => Repetition::X16,
        32 => Repetition::X32,
        64 => Repetition::X64,
        _ => Repetition::X128,
    };
    // tcgen05.ld of `n` columns of this thread's row into registers.
    let tmem_load = |col: i32, n: usize| -> Vec<Float32> {
        let frag = make_fragment::<f32>(n);
        copy(&make_copy_atom::<f32>(Ld32x32bOp::new(rep(n))), &row_view(&tmem_f32, col, n), &frag);
        fence_view_async_tmem_load();
        (0..n).map(|i| frag.load(i)).collect()
    };
    let tmem_store = |col: i32, vals: &[Float32]| {
        let frag = make_fragment::<f32>(vals.len());
        for (i, &x) in vals.iter().enumerate() {
            frag.store(i, x);
        }
        copy(&make_copy_atom::<f32>(St32x32bOp::new(rep(vals.len()))), &frag, &row_view(&tmem_f32, col, vals.len()));
    };
    let tmem_store_i32 = |col: i32, vals: &[Int32]| {
        let frag = make_fragment::<i32>(vals.len());
        for (i, &x) in vals.iter().enumerate() {
            frag.store(i, x);
        }
        copy(&make_copy_atom::<i32>(St32x32bOp::new(rep(vals.len()))), &frag, &row_view_i32(col, vals.len()));
    };
    let neg_inf = Float32::from(f32::NEG_INFINITY);
    let pk = PackedF32Options::default();
    let pack2 = |a: Float32, b: Float32| -> Int32 {
        if T::DTYPE == DType::BF16 {
            cvt_f32x2_bf16x2::<Int32>([a, b])
        } else {
            pack::<F16, i32, 2>([a.to::<F16>(), b.to::<F16>()])
        }
    };

    // ---------------------------------------------------------------------------------
    // Softmax warpgroups
    // ---------------------------------------------------------------------------------
    let softmax = |stage: usize| {
        setmaxregister_increase(REGS_SOFTMAX);
        let (pipe_s, pipe_c) = (&pipe_s[stage], &pipe_c[stage]);
        let s_c = Var::new(make_pipeline_state(PipelineUserType::Consumer, 1));
        let c_p = Var::new(make_pipeline_state(PipelineUserType::Producer, 1));
        let order = Var::new(order.clone());
        for_range(w_beg, w_end, 1, |w| {
            let wk = work(w);
            let q_pos = wk.tile * TILE_Q as i32 + (stage * MMA_M) as i32 + t;
            let offset_q = wk.kl - wk.ql;
            let row_max = Var::new(neg_inf);
            let row_sum = Var::new(Float32::from(0.0f32));
            pipe_c.producer_acquire(&c_p.get(), None);
            let step = |apply_mask: bool, j: Int32, final_call: Boolean| {
                let ss = s_c.get();
                pipe_s.consumer_wait(&ss, None);
                let mut s: Vec<Float32> = Vec::with_capacity(TILE_KV);
                for c in 0..TILE_KV / 32 {
                    s.extend(tmem_load(TMEM_S[stage] + (c * 32) as i32, 32));
                }
                if apply_mask {
                    // Column i of the tile is masked when i >= limit: past kv_len, or (causal)
                    // past the row's diagonal, q_pos + kv_len - qo_len.
                    let kv0 = j * TILE_KV as i32;
                    let mut limit = wk.kl - kv0;
                    if cfg.causal {
                        limit = limit.min(q_pos + offset_q + 1 - kv0);
                    }
                    for (i, x) in s.iter_mut().enumerate() {
                        *x = select(limit.le(i as i32), neg_inf, *x);
                    }
                }
                let old_max = row_max.get();
                let mut mx = [old_max; 4];
                for (i, &x) in s.iter().enumerate() {
                    mx[i % 4] = fmax(mx[i % 4], x);
                }
                let new_max = fmax(fmax(mx[0], mx[1]), fmax(mx[2], mx[3]));
                let safe = select(new_max.eq(neg_inf), Float32::from(0.0f32), new_max);
                row_max.set(new_max);
                // (old max, new max) to the correction warpgroup.
                tmem_store(TMEM_V[stage], &[old_max, safe]);
                fence_view_async_tmem_store();
                pipe_c.producer_commit(&take(&c_p));
                let scale = sm_scale_log2;
                let neg = Float32::from(0.0f32) - safe * scale;
                let mut ord = order.get();
                ord.wait();
                let mut words: Vec<Int32> = Vec::with_capacity(TILE_KV / 2);
                for i in (0..TILE_KV).step_by(2) {
                    let (a, b) = fma_packed_f32x2((s[i], s[i + 1]), (scale, scale), (neg, neg), pk);
                    let (a, b) = (cute::math::exp2(a, true), cute::math::exp2(b, true));
                    s[i] = a;
                    s[i + 1] = b;
                    words.push(pack2(a, b));
                    if i == TILE_KV - 10 {
                        ord.arrive();
                    }
                    if i == TILE_KV - 6 {
                        tmem_store_i32(TMEM_P[stage], &words[..32]);
                    }
                }
                order.set(ord);
                tmem_store_i32(TMEM_P[stage] + 32, &words[32..]);
                fence_view_async_tmem_store();
                // P is ready for the tensor core.
                pipe_s.consumer_release(&take(&s_c));
                pipe_c.producer_acquire(&c_p.get(), None);
                let acc_scale = cute::math::exp2(scale * (old_max - safe), true);
                let mut sums = [(Float32::from(0.0f32), Float32::from(0.0f32)); 4];
                for i in (0..TILE_KV).step_by(8) {
                    for (u, sum) in sums.iter_mut().enumerate() {
                        *sum = add_packed_f32x2(*sum, (s[i + 2 * u], s[i + 2 * u + 1]), pk);
                    }
                }
                let s01 = add_packed_f32x2(sums[0], sums[1], pk);
                let s23 = add_packed_f32x2(sums[2], sums[3], pk);
                let tot = add_packed_f32x2(s01, s23, pk);
                let sum = row_sum.get() * acc_scale + (tot.0 + tot.1);
                row_sum.set(sum);
                if_(final_call, || {
                    // Re-acquire S for the final statistics.
                    pipe_s.consumer_wait(&s_c.get(), None);
                    tmem_store(TMEM_V[stage], &[sum, new_max]);
                    fence_view_async_tmem_store();
                });
            };
            let unmasked = wk.trip - wk.masked;
            for_range(0, unmasked, 1, |j| step(false, j, (j + 1).eq(unmasked) & wk.masked.eq(0)));
            for_range(unmasked, wk.trip, 1, |j| step(true, j, (j + 1).eq(wk.trip)));
            pipe_c.producer_commit(&take(&c_p));
            pipe_c.producer_acquire(&c_p.get(), None);
            // The empty step that syncs with the S pipeline.
            pipe_s.consumer_release(&take(&s_c));
        });
    };
    if_(warp.lt(4), || softmax(0));
    if_(warp.ge(4) & warp.lt(8), || softmax(1));

    // ---------------------------------------------------------------------------------
    // Correction warpgroup
    // ---------------------------------------------------------------------------------
    if_(warp.ge(8) & warp.lt(12), || {
        setmaxregister_decrease(REGS_CORRECTION);
        let c_c = [0, 1].map(|_| Var::new(make_pipeline_state(PipelineUserType::Consumer, 1)));
        let o_c = Var::new(make_pipeline_state(PipelineUserType::Consumer, 2));
        let wait_c = |i: usize| pipe_c[i].consumer_wait(&c_c[i].get(), None);
        let release_c = |i: usize| pipe_c[i].consumer_release(&take(&c_c[i]));
        let pk = PackedF32Options::default();
        // correction_rescale: O_i *= scale, 32 columns at a time.
        let rescale = |i: usize, scale: Float32| {
            for c in 0..HEAD_DIM / 32 {
                let col = TMEM_O[i] + (c * 32) as i32;
                let x = tmem_load(col, 32);
                let mut y = Vec::with_capacity(32);
                for j in (0..32).step_by(2) {
                    let (a, b) = mul_packed_f32x2((x[j], x[j + 1]), (scale, scale), pk);
                    y.push(a);
                    y.push(b);
                }
                tmem_store(col, &y);
            }
        };
        let rescale_step = |i: usize| {
            wait_c(i);
            let vstat = tmem_load(TMEM_V[i], 2);
            let scale = cute::math::exp2(sm_scale_log2 * (vstat[0] - vstat[1]), true);
            let should = vote_any_sync(scale.ne(1.0f32));
            let o_st = o_c.get();
            pipe_o.consumer_wait(&o_st, None);
            if_(should, || rescale(i, scale));
            release_c(1 - i);
            fence_view_async_tmem_store();
            pipe_o.consumer_release(&take(&o_c));
        };
        // correction_epilogue: O_i / row_sum to global memory, and lse.
        let epilogue = |i: usize, wk: &Work| {
            wait_c(i);
            let vstat = tmem_load(TMEM_V[i], 2);
            release_c(i);
            pipe_o.consumer_wait(&o_c.get(), None);
            let (sum, mx) = (vstat[0], vstat[1]);
            let inv = Float32::from(1.0f32) / sum;
            let row = wk.tile * TILE_Q as i32 + (i * MMA_M) as i32 + t;
            let in_range = row.lt(wk.ql);
            let base = (wk.qs + row) * (hq * HEAD_DIM) as i32 + wk.h * HEAD_DIM as i32;
            let optr = o.ptr();
            for c in 0..HEAD_DIM / 32 {
                let x = tmem_load(TMEM_O[i] + (c * 32) as i32, 32);
                if_(in_range, || {
                    for j in 0..4 {
                        let vals: [Value<T>; 8] = std::array::from_fn(|e| (x[j * 8 + e] * inv).to::<T>());
                        store_vector::<T, 8>(
                            &optr.offset(base + (c * 32 + j * 8) as i32),
                            vals,
                            StoreOptions::default(),
                        );
                    }
                });
            }
            if_(in_range, || {
                let l = cute::math::log2(sum, true) + sm_scale_log2 * mx;
                lse.store((wk.qs + row, wk.h), l);
            });
            fence_view_async_tmem_load();
            pipe_o.consumer_release(&take(&o_c));
        };
        for_range(w_beg, w_end, 1, |w| {
            let wk = work(w);
            // The first statistics need no correction.
            wait_c(0);
            release_c(0);
            wait_c(1);
            for_range(1, wk.trip, 1, |_| {
                rescale_step(0);
                rescale_step(1);
            });
            release_c(1);
            epilogue(0, &wk);
            epilogue(1, &wk);
        });
    });

    sync_threads();
    tmem.free(&tmem_f32, 512);
}

// -------------------------------------------------------------------------------------
// Host side
// -------------------------------------------------------------------------------------

/// The persistent work list (FlashInfer's `plan_kernel`, on the host): every
/// `(qo_tile, head, request)` item, the Q tiles of a request taken last to first, goes to
/// the CTA ("bucket") with the least work so far, the cost of an item being its K/V length
/// (minus the causal mask's saving).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub work_indptr: Vec<i32>,
    pub qo_tile_indices: Vec<i32>,
    pub head_indices: Vec<i32>,
    pub batch_indices: Vec<i32>,
}

impl Plan {
    pub fn new(qo_lens: &[usize], kv_lens: &[usize], num_qo_heads: usize, num_ctas: usize, causal: bool) -> Plan {
        let mut cost = vec![0f32; num_ctas];
        let mut items: Vec<Vec<(i32, i32, i32)>> = vec![vec![]; num_ctas];
        for head in 0..num_qo_heads {
            for (b, (&ql, &kl)) in qo_lens.iter().zip(kv_lens).enumerate() {
                let tiles = ql.div_ceil(TILE_Q);
                for tile in (0..tiles).rev() {
                    // The lowest-indexed bucket of least cost.
                    let mut best = 0;
                    for i in 1..num_ctas {
                        if cost[i] < cost[best] {
                            best = i;
                        }
                    }
                    cost[best] += if causal { kl as f32 - ((tiles - tile - 1) * TILE_Q) as f32 } else { kl as f32 };
                    items[best].push((tile as i32, head as i32, b as i32));
                }
            }
        }
        let mut plan =
            Plan { work_indptr: vec![0], qo_tile_indices: vec![], head_indices: vec![], batch_indices: vec![] };
        for bucket in items {
            for (tile, head, b) in bucket {
                plan.qo_tile_indices.push(tile);
                plan.head_indices.push(head);
                plan.batch_indices.push(b);
            }
            plan.work_indptr.push(plan.qo_tile_indices.len() as i32);
        }
        plan
    }

    pub fn num_ctas(&self) -> usize {
        self.work_indptr.len() - 1
    }

    pub fn work_items(&self) -> usize {
        self.qo_tile_indices.len()
    }
}

/// A ragged prefill: requests of `qo_lens[b]` queries and `kv_lens[b]` keys, Q `[sum
/// qo_lens, num_qo_heads, 128]`, K and V `[sum kv_lens, num_kv_heads, 128]` (NHD). The data
/// are held as f32 values exactly representable in the problem's element type.
#[derive(Clone, Debug)]
pub struct Problem {
    pub qo_lens: Vec<usize>,
    pub kv_lens: Vec<usize>,
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub causal: bool,
    pub bf16: bool,
    pub sm_scale: f32,
    pub qo_indptr: Vec<i32>,
    pub kv_indptr: Vec<i32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

fn indptr(lens: &[usize]) -> Vec<i32> {
    let mut v = vec![0i32];
    for &l in lens {
        v.push(v.last().unwrap() + l as i32);
    }
    v
}

impl Problem {
    /// Random data in `[-1, 1)` (rounded to the element type).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qo_lens: &[usize],
        kv_lens: &[usize],
        num_qo_heads: usize,
        num_kv_heads: usize,
        causal: bool,
        bf16: bool,
        seed: u64,
    ) -> Problem {
        assert_eq!(qo_lens.len(), kv_lens.len(), "one kv_len per request");
        for (&ql, &kl) in qo_lens.iter().zip(kv_lens) {
            assert!(kl > 0, "prefill_sm100: every request needs kv_len > 0");
            assert!(!causal || ql <= kl, "prefill_sm100: a causal request needs qo_len <= kv_len");
        }
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let round = |x: f32| if bf16 { BF16::from_f32(x).to_f32() } else { F16::from_f32(x).to_f32() };
        let mut fill = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    round(((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
                })
                .collect()
        };
        let (tq, tk): (usize, usize) = (qo_lens.iter().sum(), kv_lens.iter().sum());
        Problem {
            qo_lens: qo_lens.to_vec(),
            kv_lens: kv_lens.to_vec(),
            num_qo_heads,
            num_kv_heads,
            causal,
            bf16,
            sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            qo_indptr: indptr(qo_lens),
            kv_indptr: indptr(kv_lens),
            q: fill(tq * num_qo_heads * HEAD_DIM),
            k: fill(tk * num_kv_heads * HEAD_DIM),
            v: fill(tk * num_kv_heads * HEAD_DIM),
        }
    }

    pub fn config(&self) -> Config {
        Config {
            num_qo_heads: self.num_qo_heads,
            num_kv_heads: self.num_kv_heads,
            causal: self.causal,
            bf16: self.bf16,
        }
    }

    pub fn total_q(&self) -> usize {
        self.qo_lens.iter().sum()
    }

    pub fn total_kv(&self) -> usize {
        self.kv_lens.iter().sum()
    }

    /// The plan over `num_ctas` persistent CTAs.
    pub fn plan(&self, num_ctas: usize) -> Plan {
        Plan::new(&self.qo_lens, &self.kv_lens, self.num_qo_heads, num_ctas, self.causal)
    }

    pub fn launch(&self, plan: &Plan) -> LaunchConfig {
        LaunchConfig::new([plan.num_ctas() as u32, 1, 1], [THREADS as u32, 1, 1])
    }

    /// The attention FLOPs (FlashInfer's `attention_flops` convention: the causal mask
    /// halves the score matrix).
    pub fn flops(&self) -> f64 {
        let per = |ql: usize, kl: usize| {
            if self.causal { (2 * kl - ql) as f64 * ql as f64 / 2.0 } else { ql as f64 * kl as f64 }
        };
        let pairs: f64 = self.qo_lens.iter().zip(&self.kv_lens).map(|(&q, &k)| per(q, k)).sum();
        pairs * self.num_qo_heads as f64 * 4.0 * HEAD_DIM as f64
    }

    fn half_arg(&self, v: &[f32]) -> KernelArg {
        if self.bf16 {
            KernelArg::buffer(&v.iter().map(|&x| BF16::from_f32(x)).collect::<Vec<_>>())
        } else {
            KernelArg::buffer(&v.iter().map(|&x| F16::from_f32(x)).collect::<Vec<_>>())
        }
    }

    /// The kernel arguments (see [`p`]); `o` and `lse` start at zero.
    pub fn args(&self, plan: &Plan) -> Vec<KernelArg> {
        let (tq, hq) = (self.total_q(), self.num_qo_heads);
        vec![
            KernelArg::ScalarI32(tq as i32),
            KernelArg::ScalarI32(self.total_kv() as i32),
            KernelArg::ScalarI32(self.qo_lens.len() as i32 + 1),
            KernelArg::ScalarI32(plan.num_ctas() as i32 + 1),
            KernelArg::ScalarI32(plan.work_items().max(1) as i32),
            self.half_arg(&self.q),
            self.half_arg(&self.k),
            self.half_arg(&self.v),
            self.half_arg(&vec![0.0; tq * hq * HEAD_DIM]),
            KernelArg::f32(vec![0.0; tq * hq]),
            KernelArg::i32(self.qo_indptr.clone()),
            KernelArg::i32(self.kv_indptr.clone()),
            KernelArg::i32(plan.work_indptr.clone()),
            KernelArg::i32(if plan.work_items() == 0 { vec![0] } else { plan.qo_tile_indices.clone() }),
            KernelArg::i32(if plan.work_items() == 0 { vec![0] } else { plan.head_indices.clone() }),
            KernelArg::i32(if plan.work_items() == 0 { vec![0] } else { plan.batch_indices.clone() }),
            KernelArg::ScalarF32(self.sm_scale * std::f32::consts::LOG2_E),
        ]
    }

    /// The f64 reference of one query row: `o [128]` and the base-2 `lse` of row `i` of
    /// request `b`, head `h`.
    pub fn reference_row(&self, b: usize, h: usize, i: usize) -> (Vec<f32>, f32) {
        let (hq, hkv, d) = (self.num_qo_heads, self.num_kv_heads, HEAD_DIM);
        let hk = h / (hq / hkv);
        let (qs, ks) = (self.qo_indptr[b] as usize, self.kv_indptr[b] as usize);
        let (ql, kl) = (self.qo_lens[b], self.kv_lens[b]);
        let visible = if self.causal { (i + kl - ql + 1).min(kl) } else { kl };
        let qrow = &self.q[((qs + i) * hq + h) * d..][..d];
        let s: Vec<f64> = (0..visible)
            .map(|j| {
                let krow = &self.k[((ks + j) * hkv + hk) * d..][..d];
                qrow.iter().zip(krow).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * self.sm_scale as f64
            })
            .collect();
        let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
        let sum: f64 = w.iter().sum();
        let o = (0..d)
            .map(|e| {
                let acc: f64 =
                    w.iter().enumerate().map(|(j, wj)| wj * self.v[((ks + j) * hkv + hk) * d + e] as f64).sum();
                (acc / sum) as f32
            })
            .collect();
        (o, ((mx + sum.ln()) * std::f64::consts::LOG2_E) as f32)
    }

    /// Every `stride`-th query row of every request and head (the first and last rows
    /// always): `(request, head, row)`.
    pub fn sample_rows(&self, stride: usize) -> Vec<(usize, usize, usize)> {
        let mut rows = vec![];
        for (b, &ql) in self.qo_lens.iter().enumerate() {
            for h in 0..self.num_qo_heads {
                let mut i = 0;
                while i < ql {
                    rows.push((b, h, i));
                    i += stride;
                }
                if (ql - 1) % stride != 0 {
                    rows.push((b, h, ql - 1));
                }
            }
        }
        rows
    }

    /// `(max |o - ref|, max |lse - ref|)` over `rows` of the kernel's `o` (as f32,
    /// `[total_q, num_qo_heads, 128]`) and `lse`.
    pub fn check(&self, o: &[f32], lse: &[f32], rows: &[(usize, usize, usize)]) -> (f32, f32) {
        let (hq, d) = (self.num_qo_heads, HEAD_DIM);
        let (mut eo, mut el) = (0f32, 0f32);
        for &(b, h, i) in rows {
            let (want, want_lse) = self.reference_row(b, h, i);
            let r = self.qo_indptr[b] as usize + i;
            let got = &o[(r * hq + h) * d..][..d];
            eo = got.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(eo, f32::max);
            let e = (lse[r * hq + h] - want_lse).abs();
            el = if e.is_nan() { f32::INFINITY } else { el.max(e) };
            if got.iter().any(|x| x.is_nan()) {
                eo = f32::INFINITY;
            }
        }
        (eo, el)
    }

    fn out_f32(&self, arg: &KernelArg) -> Vec<f32> {
        if self.bf16 {
            arg.to_vec::<BF16>().into_iter().map(|x| x.to_f32()).collect()
        } else {
            arg.to_vec::<F16>().into_iter().map(|x| x.to_f32()).collect()
        }
    }
}

/// Run `p` on the simulator over `num_ctas` persistent CTAs: `o` (as f32) and `lse`.
pub fn execute(p: &Problem, num_ctas: usize) -> (Vec<f32>, Vec<f32>) {
    let plan = p.plan(num_ctas);
    let mut args = p.args(&plan);
    let k = p.config().kernel();
    k.simulate(p.launch(&plan), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    (p.out_f32(&args[p::O]), args[p::LSE].as_f32().to_vec())
}

/// [`execute`] checked against the f64 reference on every row: `(max |o - ref|, max |lse -
/// ref|)`.
pub fn simulate(p: &Problem, num_ctas: usize) -> (f32, f32) {
    let (o, lse) = execute(p, num_ctas);
    p.check(&o, &lse, &p.sample_rows(1))
}

/// Running on a B200 (`--features cuda`): the kernel compiled and loaded once, a problem
/// run, checked against the f64 reference and timed from CUDA-graph replays.
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg, HardwareInfo};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{Config, Problem};

    /// A kernel compiled for the device and loaded.
    pub struct Kernels<'c> {
        pub config: Config,
        pub kernel: (Kernel, CudaFunction<'c>),
        /// One persistent CTA per SM.
        pub num_ctas: usize,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &Toolchain, config: Config) -> Result<Kernels<'c>, Box<dyn Error>> {
            let opts = CompileOptions::new().gpu_arch(Arch::Sm100a);
            let k = config.kernel();
            let f = ctx.load(&k.compile_with(tc, &opts)?)?;
            let num_ctas = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            Ok(Kernels { config, kernel: (k, f), num_ctas })
        }
    }

    /// Run `p` once; check `check_rows`-strided rows against the reference when `check > 0`
    /// (`(max |o - ref|, max |lse - ref|)`), and time CUDA-graph replays when `time`
    /// (microseconds).
    #[allow(clippy::type_complexity)]
    pub fn run(
        ctx: &CudaContext,
        ks: &Kernels<'_>,
        p: &Problem,
        check: usize,
        time: bool,
    ) -> Result<(Option<(f32, f32)>, Option<f64>), Box<dyn Error>> {
        if ks.config != p.config() {
            return Err("the kernel's configuration differs from the problem's".into());
        }
        let plan = p.plan(ks.num_ctas);
        let host = p.args(&plan);
        let mut bufs = Vec::new();
        for a in &host {
            if !matches!(a, KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_)) {
                bufs.push(ctx.upload(a)?);
            }
        }
        let mut bi = bufs.iter();
        let args: Vec<DeviceArg> = host
            .iter()
            .map(|a| match a {
                KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => DeviceArg::Scalar(a.clone()),
                _ => DeviceArg::Buffer(bi.next().unwrap()),
            })
            .collect();
        let (k, f) = (&ks.kernel.0, &ks.kernel.1);
        let launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(k, f, p.launch(&plan)).stream(s).args(&args).submit()?;
            Ok(())
        };
        let stream = ctx.stream()?;
        launch(&stream, 0)?;
        stream.synchronize()?;
        // The buffers are the scalar-free arguments, in order: q, k, v, o, lse, ...
        let err = if check > 0 {
            let o_buf = &bufs[3];
            let o: Vec<f32> = if p.bf16 {
                o_buf.to_vec::<u16>()?.into_iter().map(|x| BF16(x).to_f32()).collect()
            } else {
                o_buf.to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect()
            };
            let lse = bufs[4].to_vec::<f32>()?;
            Some(p.check(&o, &lse, &p.sample_rows(check)))
        } else {
            None
        };
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(5).iterations(30).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, launch)?)
        } else {
            None
        };
        Ok((err, t))
    }
}
