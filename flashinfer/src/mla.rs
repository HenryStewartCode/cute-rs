// SPDX-License-Identifier: Apache-2.0
//
// Copyright (c) 2023-2025 by FlashInfer team.
//
// Translated from FlashInfer v0.7.5: the kernel from `include/flashinfer/attention/mla.cuh`
// (`BatchMLAPagedAttentionKernel`, `DevicePersistentMergeStates`), the planner from
// `include/flashinfer/attention/scheduler.cuh` (`MLAPlan`) with `heap.h`'s `MinHeap`.

//! Multi-head latent attention (DeepSeek V2/V3/R1) over a paged cache: FlashInfer's
//! `BatchMLAPagedAttentionWrapper` with the `fa2` backend (what `backend="auto"` picks on
//! SM89), translated from `mla.cuh` and the `MLAPlan` planner of `scheduler.cuh`.
//!
//! Queries come in two parts, `q_nope [*, H, 512]` and `q_pe [*, H, 64]`; the cache is a
//! compressed `ckv [pages, page_size, 512]` (both the keys' "no-PE" half and the values)
//! and `kpe [pages, page_size, 64]`. Logits are `(q_nope . ckv + q_pe . kpe) * sm_scale`,
//! the output is `softmax . ckv` (`[*, H, 512]`, f16) plus FlashInfer's base-2 LSE.
//!
//! * **Plan** ([`Plan::new`], on the host, as FlashInfer's `MLAPlan`): the query heads of
//!   a request are packed (`row = token * H + head`). Clusters of 1 or 2 CTAs (2 when the
//!   average packed length is above 64) take 64- or 128-row tiles, `num_sm / cluster`
//!   clusters in all (a persistent grid). Each tile's KV range is cut into chunks of at
//!   most `kv_len_limit` tokens (FlashInfer's rounding of the average KV per cluster), and
//!   chunks go to the least-loaded cluster (`MinHeap`, cost `2 * tile + kv`). A split tile
//!   writes f16 partial states, and its rows are given to merge CTAs.
//! * **Kernel**: grid `(cluster, num_clusters)`, block `(32, 4, 2)` (two warpgroups of four
//!   warps), one cooperative launch. Each CTA loops over its cluster's work items:
//!   * Q (64 packed rows of 512 + 64) is staged once in shared memory with 16-byte
//!     `cp.async`; KV streams in tiles of 16 tokens (warpgroup 0 copies, through the page
//!     table, zero-filling past `kv_len`), from the last tile down to the first.
//!   * Shared memory uses FlashInfer's 128-byte permuted layout (vector `j` of row `i` at
//!     `i * stride + (j ^ (i % 8))`).
//!   * Warp `y` of both warpgroups computes the scores of rows `16 y ..` against the tile
//!     (`mma.sync m16n8k16`, f16 in, f32 accumulate, `ldmatrix.x4` operands; 36 steps of
//!     16 over the 512 no-PE and 64 PE columns), applies the chunk-end and causal mask
//!     on the boundary tiles, and runs FlashInfer's base-2 online softmax (row maxima over
//!     a lane quad, row sums by an MMA against ones). Warpgroup `z` then accumulates half
//!     `z` of the 512 output columns, `O += P V` with V read by `ldmatrix.x4.trans` from
//!     the same `ckv` tile.
//!   * The output is staged through shared memory for 16-byte stores, to `o`/`lse` or, for
//!     a split tile, to the f16 partial buffer.
//!   * After a grid-wide barrier, each CTA merges the rows the plan gave it (FlashInfer's
//!     `DevicePersistentMergeStates`), eight columns per thread.
//!
//! On a GPU with 100 KB of shared memory per SM (SM89; FlashInfer's `DISPATCH_SMEM_CONFIG`
//! for `>= 92672` bytes) FlashInfer uses 1 stage, 16-token KV tiles and no QK sharding
//! between the warpgroups; this port implements that configuration only (90 KB of shared
//! memory, one CTA per SM).
//!
//! Differences from FlashInfer:
//! * `num_heads` and `page_size` are build-time constants of a kernel (FlashInfer divides
//!   by them with `uint_fastdiv` at run time).
//! * In shared memory a row holds a query's (or token's) no-PE part and then its PE part
//!   (FlashInfer has separate PE arrays, the same size in all), and `Q K^T` takes the PE
//!   part last. With separate arrays, LLVM hoists all 88 `ldmatrix` addresses of the main
//!   loop out of it, a 64-bit register each (cute-rs's shared pointers are generic), and
//!   the kernel spilled 316 bytes; with shared rows the Q and K addresses are four bases
//!   plus constants each, and it spills 24 bytes (outside the main loop).
//! * The page of the next tile's rows is read at the start of an iteration, so the copies
//!   issued at its end do not wait on the page table (as the decode port does). FlashInfer
//!   reads it just before the copies, which puts two dependent memory latencies in each
//!   iteration of its single-stage pipeline.
//! * FlashInfer's `cooperative_groups` grid sync is a monotonic arrival counter in global
//!   memory (a one-`i32` buffer, zeroed once, wrapping after about 2^31 / num_sm launches);
//!   the port flips a phase bit (bit 31) as cooperative groups does, so it never overflows.
//! * The causal KV bound of a cluster uses `ceil(cluster_end / H)` as the planner does;
//!   `mla.cuh` uses the floor, which drops a visible key when a partial query ends a
//!   cluster on a KV-tile boundary (FlashInfer's Python rejects those workloads).
//! * LSE is always written (base 2).
//!
//! Not covered: the 2-stage / 32- and 64-token KV-tile / QK-shard configurations of
//! larger-shared-memory GPUs, FP8 KV, bf16, non-contiguous strides, the SM90 (`fa3`)
//! backend, CUDA-graph re-planning into fixed buffers.
//!
//! `examples/mla.rs` runs it on the simulator and, with `--features cuda` and `GPU=1`, on
//! the GPU against the f64 reference, then benchmarks it on the shapes of FlashInfer's
//! `benchmarks/bench_deepseek_mla.py` (`examples/mla_flashinfer_bench.py`).

use cute::prelude::*;

/// Width of `q_nope`, `ckv` and the output.
pub const HEAD_DIM_CKV: usize = 512;
/// Width of `q_pe` and `kpe`.
pub const HEAD_DIM_KPE: usize = 64;
/// Packed query rows per CTA.
pub const CTA_TILE_Q: usize = 64;
/// KV tokens per tile (FlashInfer's choice for 100 KB of shared memory per SM).
pub const CTA_TILE_KV: usize = 16;
/// Threads per CTA: `(32, 4, 2)`.
pub const NUM_THREADS: usize = 256;
/// 16-byte vectors per row of `ckv` / `q_nope` / `o`, and of `kpe` / `q_pe`.
const VC: usize = HEAD_DIM_CKV / 8;
const VP: usize = HEAD_DIM_KPE / 8;
/// 16-wide MMA tiles of output columns per warpgroup.
const NUM_MMA_D_HALF: usize = HEAD_DIM_CKV / 16 / 2;
/// Elements a vector view may address (as in the other ports).
const BOUND: usize = 1 << 30;
/// 16-byte vectors per shared-memory row: a query's (or token's) no-PE part, then its PE
/// part.
const VR: usize = VC + VP;
/// Shared memory, in 16-byte vectors: the Q tile and the KV tile (the output tile aliases
/// the Q tile, as FlashInfer's union). FlashInfer keeps the PE parts in arrays of their
/// own; here they extend the rows, so that the 36 steps of `Q K^T` are one loop.
const S_Q: usize = 0;
const S_K: usize = S_Q + CTA_TILE_Q * VR;
const S_END: usize = S_K + CTA_TILE_KV * VR;
/// i32 fields per work item and per merge CTA in [`Plan::work`] and [`Plan::merge`].
pub const WORK_FIELDS: usize = 8;
pub const MERGE_FIELDS: usize = 8;

/// What a kernel is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelSpec {
    pub num_heads: usize,
    pub page_size: usize,
    pub causal: bool,
}

impl KernelSpec {
    pub fn name(&self) -> String {
        format!(
            "flashinfer_mla_paged_h{}_p{}{}",
            self.num_heads,
            self.page_size,
            if self.causal { "_causal" } else { "" }
        )
    }
}

/// Parameter positions of the kernel.
pub mod p {
    pub const Q_NOPE: usize = 0;
    pub const Q_PE: usize = 1;
    pub const CKV: usize = 2;
    pub const KPE: usize = 3;
    pub const KV_INDICES: usize = 4;
    pub const WORK_INDPTR: usize = 5;
    pub const WORK: usize = 6;
    pub const MERGE: usize = 7;
    pub const O: usize = 8;
    pub const LSE: usize = 9;
    pub const PARTIAL_O: usize = 10;
    pub const PARTIAL_LSE: usize = 11;
    pub const BARRIER: usize = 12;
    pub const SM_SCALE_LOG2: usize = 13;
}

/// The MLA kernel for `spec`. Parameters are in the order of [`p`]: `q_nope [n, H, 512]`,
/// `q_pe [n, H, 64]`, `ckv [pages, page_size, 512]`, `kpe [pages, page_size, 64]`,
/// `kv_indices`, the plan's `work_indptr [clusters + 1]`, `work [works, 8]` and `merge
/// [num_sm, 8]`, `o [n, H, 512]`, `lse [n, H]`, `partial_o [rows, 512]` (f16),
/// `partial_lse [rows]`, `barrier [1]` (zeroed once) and the scalar `sm_scale_log2`.
pub fn kernel(spec: &KernelSpec) -> Kernel {
    let KernelSpec { num_heads, page_size, causal } = *spec;
    let (h, ps) = (num_heads as i32, page_size as i32);
    let (vc, vp, vr) = (VC as i32, VP as i32, VR as i32);
    Kernel::builder(spec.name()).block_dim([32, 4, 2]).build(|| {
        let q_nope = param_ptr::<F16>("q_nope");
        let q_pe = param_ptr::<F16>("q_pe");
        let ckv = param_ptr::<F16>("ckv");
        let kpe = param_ptr::<F16>("kpe");
        let kv_indices = param_ptr::<i32>("kv_indices");
        let work_indptr = param_ptr::<i32>("work_indptr");
        let work = param_ptr::<i32>("work");
        let merge = param_ptr::<i32>("merge");
        let o = param_ptr::<F16>("o");
        let lse = param_ptr::<f32>("lse");
        let partial_o = param_ptr::<F16>("partial_o");
        let partial_lse = param_ptr::<f32>("partial_lse");
        let barrier = param_ptr::<i32>("barrier");
        let sm_scale_log2 = param_scalar::<f32>("sm_scale_log2");

        let (lane, y, z) = (thread_idx().x, thread_idx().y, thread_idx().z);
        let tid = (z * 4 + y) * 32 + lane;
        let (bx, by) = (block_idx().x, block_idx().y);
        let (gdx, gdy) = (grid_dim().x, grid_dim().y);

        let smem = make_smem_tensor_aligned::<F16>(Layout::col_major(S_END * 8), 16);
        let sv = make_tensor(&smem.ptr(), Layout::col_major((8, S_END)));
        let s_pairs = make_tensor(&smem.ptr(), Layout::col_major((2, S_END * 4)));
        let vecs = |ptr: &Ptr<F16>| make_tensor(ptr, Layout::col_major((8, BOUND / 8)));
        let (qn_g, qp_g, ckv_g, kpe_g, o_g, po_g) =
            (vecs(&q_nope), vecs(&q_pe), vecs(&ckv), vecs(&kpe), vecs(&o), vecs(&partial_o));
        let cp_async = make_copy_atom::<F16>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let ldsm = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false });
        let ldsm_t = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: true });
        let mma = make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)), Layout::col_major((1, 1, 1)));

        // A predicated 16-byte cp.async (zero-filled when off) from global vector `gvec`
        // of `src` to shared vector `svec`.
        let load_vec = |src: &Tensor<F16>, gvec: Int32, svec: Int32, pred: Boolean| {
            let pf = make_fragment::<bool>(1);
            pf.store(0, pred);
            copy_if(&cp_async, &pf, &src.slice((.., gvec)), &sv.slice((.., svec)));
        };
        // `ldmatrix.x4` (optionally transposed) at shared vector `svec`.
        let ldmatrix = |trans: bool, svec: Int32| -> Vec<Value<F16>> {
            let frag = make_fragment::<F16>(8);
            copy(if trans { &ldsm_t } else { &ldsm }, &sv.slice((.., svec)), &frag);
            frag.values()
        };
        // One m16n8k16 MMA: `a` (8 values), `b` (4) and `c` (4) in PTX register order.
        let mma16 = |a: &[Value<F16>], b: &[Value<F16>], c: &[Float32]| -> Vec<Float32> {
            let (fa, fb, fc) =
                (make_fragment::<F16>((8, 1, 1)), make_fragment::<F16>((4, 1, 1)), make_fragment::<f32>((4, 1, 1)));
            fa.store_values(a);
            fb.store_values(b);
            fc.store_values(c);
            gemm(&mma, &fc, &fa, &fb, &fc);
            fc.values()
        };
        let zero = Float32::from(0.0f32);
        let neg_inf = Float32::from(f32::NEG_INFINITY);
        let ones = vec![Value::<F16>::from(F16::from_f32(1.0)); 4];
        let ceil_div = |a: Int32, b: i32| (a + (b - 1)) / b;

        // ---- Phase 1: the cluster's work items.
        let w_begin = work_indptr.load_at(by);
        let w_end = work_indptr.load_at(by + 1);
        for_range(w_begin, w_end, 1, |w| {
            let field = |i: i32| work.load_at(w * WORK_FIELDS as i32 + i);
            let q_indptr = field(0);
            let kv_indptr = field(1);
            let partial_indptr = field(2);
            let q_len = field(3);
            let kv_len = field(4);
            let q_start = field(5);
            let kv_start = field(6);
            let kv_end = field(7);
            // This CTA's first packed row of the request.
            let qo_base = q_start + bx * CTA_TILE_Q as i32;
            let has_partial = partial_indptr.ge(0);
            let partial_row = partial_indptr + bx * CTA_TILE_Q as i32;

            // The previous item's output staging aliases the Q tile.
            sync_threads();

            let main = || {
                // ---- Q: global -> shared, rows past the request zero-filled. (A rolled
                // loop: unrolled, LLVM hoists all 18 shared addresses out of the work loop,
                // where they stay live through the main loop.)
                for_range_with(0, 2, 1, LoopOptions::unroll(1), |mq| {
                    let srow = z * 16 + y * 4 + lane / 8 + 32 * mq;
                    let packed = qo_base + srow;
                    let qi = packed / h;
                    let row = (q_indptr + qi.min(q_len - 1)) * h + packed % h;
                    let x = (lane % 8) ^ (srow % 8);
                    let dst = S_Q as i32 + srow * vr + x;
                    for md in 0..VC / 8 {
                        let m8 = (8 * md) as i32;
                        load_vec(&qn_g, row * vc + m8 + lane % 8, dst + m8, qi.lt(q_len));
                    }
                    load_vec(&qp_g, row * vp + lane % 8, dst + vc, qi.lt(q_len));
                });

                // ---- KV tile `t` (16 tokens from `kv_start + 16 t`): warpgroup 0, one row
                // per 8 lanes, tokens past `kv_len` zero-filled.
                let iter_base = kv_indptr * ps + kv_start;
                let bound = kv_indptr * ps + kv_len;
                let wr_row = y * 4 + lane / 8;
                let wr_x = (lane % 8) ^ (wr_row % 8);
                // The page of this thread's row of tile `t` (read one tile ahead, so the
                // copies never wait on the page table), and the copies of tile `t`.
                let page_of =
                    |t: Int32| kv_indices.load_at((iter_base + t * CTA_TILE_KV as i32 + wr_row).min(bound - 1) / ps);
                let load_kv = |t: Int32, page: Int32| {
                    if_(z.eq(0), || {
                        let it = iter_base + t * CTA_TILE_KV as i32 + wr_row;
                        let ic = it.min(bound - 1);
                        let row = page * ps + ic % ps;
                        let valid = it.lt(bound);
                        let dst = S_K as i32 + wr_row * vr + wr_x;
                        for md in 0..VC / 8 {
                            let m8 = (8 * md) as i32;
                            load_vec(&ckv_g, row * vc + m8 + lane % 8, dst + m8, valid);
                        }
                        load_vec(&kpe_g, row * vp + lane % 8, dst + vc, valid);
                    });
                };

                // ---- Tiles (FlashInfer's bounds): the last one this cluster's rows can see,
                // and the first that needs masking. Tiles run from the last down to 0.
                let cluster_tile = gdx * CTA_TILE_Q as i32;
                let (vis_end, unmasked_end) = if causal {
                    (
                        kv_end.min(kv_len - q_len + ceil_div(q_start + cluster_tile, h)),
                        kv_end.min(kv_len - q_len + q_start / h),
                    )
                } else {
                    (kv_end, kv_end)
                };
                let first_tile = kv_start / CTA_TILE_KV as i32;
                let top = ceil_div(vis_end, CTA_TILE_KV as i32) - 1 - first_tile;
                let mask_from = unmasked_end / CTA_TILE_KV as i32 - first_tile;
                let first = top.max(Int32::from(0));
                load_kv(first, page_of(first));
                cp_async_commit_group();

                // ---- Per-thread ldmatrix addresses (vector indices). The rows a lane reads
                // all have swizzle `lane % 8`, so vector `8 k + c` of its row is at `8 k +
                // (c ^ (lane % 8))`: `[j]` is the address of `c = 2 j + (the lane's matrix
                // column)`, and step `4 k + j` adds `8 k`.
                let q_sw: Vec<Int32> = (0..4)
                    .map(|j| S_Q as i32 + (y * 16 + lane % 16) * vr + ((lane / 16 + 2 * j) ^ (lane % 8)))
                    .collect();
                let k_row = 8 * (lane / 16) + lane % 8;
                let k_sw: Vec<Int32> =
                    (0..4).map(|j| S_K as i32 + k_row * vr + (((lane % 16) / 8 + 2 * j) ^ (lane % 8))).collect();
                let v_sw: Vec<Int32> = (0..4)
                    .map(|j| {
                        S_K as i32
                            + (lane % 16) * vr
                            + z * (2 * NUM_MMA_D_HALF) as i32
                            + ((lane / 16 + 2 * j) ^ (lane % 8))
                    })
                    .collect();
                // The query index of this thread's two accumulator rows (causal mask).
                let acc_q: [Int32; 2] = [0, 1].map(|j| (qo_base + y * 16 + lane / 4 + 8 * j) / h);

                // Online-softmax state: `o[mma_d][8]` over this warpgroup's 256 columns,
                // `m` (unscaled) and `d` per accumulator row.
                let o_state = Var::new(vec![zero; NUM_MMA_D_HALF * 8]);
                let m_state = Var::new(vec![neg_inf; 2]);
                let d_state = Var::new(vec![Float32::from(1.0f32); 2]);

                for_range(0, top + 1, 1, |i| {
                    let t = top - i;
                    let next_page = page_of((t - 1).max(Int32::from(0)));
                    cp_async_wait_group(0);
                    sync_threads();

                    // S = [Q_nope Q_pe] [ckv kpe]^T: an m16n16 tile per warp, 36 steps of
                    // 16 columns, the no-PE part then the PE part.
                    let mut s = vec![zero; 8];
                    for j in 0..VR / 2 {
                        let off = (8 * (j / 4)) as i32;
                        let a = ldmatrix(false, q_sw[j % 4] + off);
                        let b = ldmatrix(false, k_sw[j % 4] + off);
                        for n in 0..2 {
                            let r = mma16(&a, &b[4 * n..4 * n + 4], &s[4 * n..4 * n + 4]);
                            s[4 * n..4 * n + 4].copy_from_slice(&r);
                        }
                    }

                    // Mask the boundary tiles (the chunk end and the causal diagonal).
                    let kv_base = kv_start + t * CTA_TILE_KV as i32;
                    let s = if_else_yield(
                        t.ge(mask_from) | t.eq(0),
                        || {
                            let mut s = s.clone();
                            for (reg, x) in s.iter_mut().enumerate() {
                                let kv_idx = kv_base + (8 * (reg / 4) + reg % 2) as i32 + 2 * (lane % 4);
                                let mut masked = kv_idx.ge(kv_end);
                                if causal {
                                    masked = masked | (kv_idx + q_len).gt(kv_len + acc_q[(reg % 4) / 2]);
                                }
                                *x = select(masked, neg_inf, *x);
                            }
                            s
                        },
                        || s.clone(),
                    );

                    // Online softmax (FlashInfer's `update_mdo_states_`).
                    let (mut o_v, mut m_v, mut d_v) = (o_state.get(), m_state.get(), d_state.get());
                    let mut pr = s;
                    for j in 0..2 {
                        let m_prev = m_v[j];
                        let mut mx = m_prev;
                        for e in [0, 1, 4, 5] {
                            mx = fmax(mx, pr[2 * j + e]);
                        }
                        mx = fmax(mx, shuffle_sync_bfly(mx, 2));
                        mx = fmax(mx, shuffle_sync_bfly(mx, 1));
                        let m_scaled = fmax(mx * sm_scale_log2, Float32::from(-f32::MAX));
                        let o_scale = cute::math::exp2(m_prev * sm_scale_log2 - m_scaled, true);
                        d_v[j] = d_v[j] * o_scale;
                        for md in 0..NUM_MMA_D_HALF {
                            for e in [0, 1, 4, 5] {
                                o_v[md * 8 + 2 * j + e] = o_v[md * 8 + 2 * j + e] * o_scale;
                            }
                        }
                        for e in [0, 1, 4, 5] {
                            pr[2 * j + e] = cute::math::exp2(pr[2 * j + e] * sm_scale_log2 - m_scaled, true);
                        }
                        m_v[j] = mx;
                    }

                    // P in f16, row sums by an MMA against ones, then O += P V on this
                    // warpgroup's half of the columns.
                    let p16: Vec<Value<F16>> = pr.iter().map(|x| x.to::<F16>()).collect();
                    let r = mma16(&p16, &ones, &[d_v[0], zero, d_v[1], zero]);
                    d_v[0] = r[0];
                    d_v[1] = r[2];
                    for md in 0..NUM_MMA_D_HALF {
                        let b = ldmatrix(true, v_sw[md % 4] + (8 * (md / 4)) as i32);
                        for n in 0..2 {
                            let at = md * 8 + 4 * n;
                            let r = mma16(&p16, &b[4 * n..4 * n + 4], &o_v[at..at + 4]);
                            o_v[at..at + 4].copy_from_slice(&r);
                        }
                    }
                    o_state.set(o_v);
                    m_state.set(m_v);
                    d_state.set(d_v);

                    // The next (lower) tile, once every warp is done with this one.
                    if_(t.gt(0), || {
                        sync_threads();
                        load_kv(t - 1, next_page);
                    });
                    cp_async_commit_group();
                });
                cp_async_wait_group(0);
                // The output staging aliases the Q tile other warps may still read.
                sync_threads();

                // ---- Normalise (`normalize_d_`, `finalize_m_`) and write (`write_o`).
                let (mut o_v, m_v, d_v) = (o_state.get(), m_state.get(), d_state.get());
                let empty: Vec<Boolean> = m_v.iter().map(|m| m.eq(neg_inf)).collect();
                let d_rcp: Vec<Float32> =
                    (0..2).map(|j| select(empty[j], zero, cute::math::rcp(d_v[j], true))).collect();
                for (k, ov) in o_v.iter_mut().enumerate() {
                    *ov = *ov * d_rcp[(k % 4) / 2];
                }
                let lse_v: Vec<Float32> = (0..2)
                    .map(|j| select(empty[j], neg_inf, cute::math::log2(d_v[j], true) + m_v[j] * sm_scale_log2))
                    .collect();

                // f16 pairs into this warp's rows (and warpgroup's half) of the Q tile.
                for md in 0..NUM_MMA_D_HALF {
                    let row = y * 16 + lane / 4;
                    let vec0 =
                        S_Q as i32 + row * vr + ((z * (2 * NUM_MMA_D_HALF) as i32 + (2 * md) as i32) ^ (lane / 4));
                    for (n, vec) in [vec0, vec0 ^ 1].into_iter().enumerate() {
                        for j in 0..2 {
                            let at = md * 8 + 4 * n + 2 * j;
                            let pair = (vec + 8 * vr * j as i32) * 4 + lane % 4;
                            let vals: Vec<Value<F16>> = o_v[at..at + 2].iter().map(|x| x.to::<F16>()).collect();
                            s_pairs.slice((.., pair)).store_values(&vals);
                        }
                    }
                }
                sync_warp();

                // 16-byte rows out: to `partial_o` for a split tile, else to `o`.
                for j in 0..4 {
                    let local = y * 16 + lane / 8 + 4 * j;
                    let packed = qo_base + local;
                    let qi = packed / h;
                    let grow = (q_indptr + qi) * h + packed % h;
                    if_(qi.lt(q_len), || {
                        for md in 0..NUM_MMA_D_HALF / 4 {
                            let col = z * (2 * NUM_MMA_D_HALF) as i32 + lane % 8 + (8 * md) as i32;
                            let svec = S_Q as i32 + local * vr + (col ^ (local % 8));
                            let vals = sv.slice((.., svec)).values();
                            if_else(
                                has_partial,
                                || po_g.slice((.., (partial_row + local) * vc + col)).store_values(&vals),
                                || o_g.slice((.., grow * vc + col)).store_values(&vals),
                            );
                        }
                    });
                }
                for (j, lse_j) in lse_v.iter().enumerate() {
                    let local = y * 16 + lane / 4 + (8 * j) as i32;
                    let packed = qo_base + local;
                    let qi = packed / h;
                    if_(qi.lt(q_len) & (lane % 4).eq(0) & z.eq(0), || {
                        if_else(
                            has_partial,
                            || partial_lse.store_at(partial_row + local, *lse_j),
                            || lse.store_at((q_indptr + qi) * h + packed % h, *lse_j),
                        );
                    });
                }
            };

            // An empty KV range (`write_empty_o`): zero output, LSE -inf.
            let empty = || {
                let valid = (q_len * h - qo_base).max(Int32::from(0)).min(Int32::from(CTA_TILE_Q as i32));
                let n = valid * HEAD_DIM_CKV as i32;
                let zero16 = Value::<F16>::from(F16::from_f32(0.0));
                for_range(tid, n, NUM_THREADS as i32, |e| {
                    let pk = e / HEAD_DIM_CKV as i32;
                    let dim = e % HEAD_DIM_CKV as i32;
                    if_else(
                        has_partial,
                        || partial_o.store_at((partial_row + pk) * HEAD_DIM_CKV as i32 + dim, zero16),
                        || {
                            let packed = qo_base + pk;
                            o.store_at(((q_indptr + packed / h) * h + packed % h) * HEAD_DIM_CKV as i32 + dim, zero16);
                        },
                    );
                });
                for_range(tid, valid, NUM_THREADS as i32, |pk| {
                    if_else(
                        has_partial,
                        || partial_lse.store_at(partial_row + pk, neg_inf),
                        || {
                            let packed = qo_base + pk;
                            lse.store_at((q_indptr + packed / h) * h + packed % h, neg_inf);
                        },
                    );
                });
            };
            if_else(kv_end.gt(kv_start), main, empty);
        });

        // ---- Grid-wide barrier (FlashInfer's `grid.sync()`, cooperative groups' scheme): bit 31
        // of the counter is the phase. Every CTA adds 1 except CTA 0, which adds
        // `0x8000_0000 - (n - 1)`, so the low bits return to 0 and the phase bit flips when the
        // last CTA arrives; the waiters leave when the bit differs from the one they saw. The
        // counter never grows, so any number of launches is safe.
        sync_threads();
        if_(tid.eq(0), || {
            fence_acq_rel_gpu();
            let num_ctas = gdx * gdy;
            let first = (bx + by).eq(0);
            let add = select(first, Int32::from(i32::MIN) - (num_ctas - 1), Int32::from(1));
            let old = atomic_add_with(&barrier, add, AtomicOptions::new(MemoryOrder::Relaxed, MemoryScope::Gpu));
            let opts =
                LoadOptions { sem: MemoryOrder::Relaxed, scope: Some(MemoryScope::Gpu), ..LoadOptions::default() };
            let bar = barrier.clone();
            let _ = while_carry(old, |v: Int32| (v ^ old).ge(0), |_v: Int32| load(&bar, opts));
            fence_acq_rel_gpu();
        });
        sync_threads();

        // ---- Phase 2: merge the partial states of this CTA's rows
        // (`DevicePersistentMergeStates`): 64 threads per row, 8 columns each.
        let cta = gdx * by + bx;
        let mfield = |i: i32| merge.load_at(cta * MERGE_FIELDS as i32 + i);
        let (start, end, pstart, pend, stride) = (mfield(0), mfield(1), mfield(2), mfield(3), mfield(4));
        let threads_per_row = (HEAD_DIM_CKV / 8) as i32;
        let col = tid % threads_per_row;
        for_range(tid / threads_per_row, end - start, NUM_THREADS as i32 / threads_per_row, |local| {
            let fin = start + local;
            let m = Var::new(neg_inf);
            let d = Var::new(Float32::from(1.0f32));
            let acc = Var::new(vec![zero; 8]);
            for_range(pstart + local, pend, stride, |pp| {
                let ov: Vec<Float32> = po_g.slice((.., pp * vc + col)).values().iter().map(|x| x.to::<f32>()).collect();
                let l = partial_lse.load_at(pp);
                let (m_prev, d_prev) = (m.get(), d.get());
                let m_new = fmax(m_prev, l);
                // fmax drops the NaN of (-inf) - (-inf), as FlashInfer's `state_t::merge`.
                let f1 = cute::math::exp2(fmax(m_prev - m_new, neg_inf), true);
                let f2 = cute::math::exp2(fmax(l - m_new, neg_inf), true);
                d.set(d_prev * f1 + f2);
                let a: Vec<Float32> = acc.get().iter().zip(&ov).map(|(a, v)| *a * f1 + *v * f2).collect();
                acc.set(a);
                m.set(m_new);
            });
            let dd = d.get();
            let d_rcp = select(dd.gt(zero), Float32::from(1.0f32) / dd, zero);
            let out: Vec<Value<F16>> = acc.get().iter().map(|a| (*a * d_rcp).to::<F16>()).collect();
            o_g.slice((.., fin * vc + col)).store_values(&out);
            if_(col.eq(0), || lse.store_at(fin, m.get() + cute::math::log2(dd, true)));
        });
    })
}

/// FlashInfer's `MinHeap` (`heap.h`) over `std::push_heap` / `std::pop_heap` with the
/// comparator `a.cost > b.cost`, reproduced step for step (libstdc++'s sift-up and
/// `__adjust_heap`) so that ties between clusters break as FlashInfer's do.
struct MinHeap {
    heap: Vec<(usize, f32)>,
}

impl MinHeap {
    fn new(capacity: usize) -> MinHeap {
        MinHeap { heap: (0..capacity).map(|i| (i, 0.0)).collect() }
    }

    fn comp(a: &(usize, f32), b: &(usize, f32)) -> bool {
        a.1 > b.1
    }

    /// libstdc++'s `__push_heap`: sift `value` up from `hole` to at most `top`.
    fn sift_up(&mut self, mut hole: usize, top: usize, value: (usize, f32)) {
        while hole > top {
            let parent = (hole - 1) / 2;
            if !Self::comp(&self.heap[parent], &value) {
                break;
            }
            self.heap[hole] = self.heap[parent];
            hole = parent;
        }
        self.heap[hole] = value;
    }

    fn insert(&mut self, e: (usize, f32)) {
        self.heap.push(e);
        let last = self.heap.len() - 1;
        self.sift_up(last, 0, e);
    }

    fn pop(&mut self) -> (usize, f32) {
        let len = self.heap.len();
        if len > 1 {
            // std::pop_heap: move the top to the back, re-heap [0, len - 1) with the old back.
            let value = self.heap[len - 1];
            self.heap[len - 1] = self.heap[0];
            let n = len - 1;
            let mut hole = 0;
            let mut child = 0;
            while child < (n.saturating_sub(1)) / 2 {
                child = 2 * (child + 1);
                if Self::comp(&self.heap[child], &self.heap[child - 1]) {
                    child -= 1;
                }
                self.heap[hole] = self.heap[child];
                hole = child;
            }
            if n.is_multiple_of(2) && n >= 2 && child == (n - 2) / 2 {
                child = 2 * (child + 1);
                self.heap[hole] = self.heap[child - 1];
                hole = child - 1;
            }
            self.sift_up(hole, 0, value);
        }
        self.heap.pop().expect("pop from an empty heap")
    }
}

/// `ceil(a / b)` for positive `b`.
fn cdiv(a: i64, b: i64) -> i64 {
    (a + b - 1).div_euclid(b)
}

/// `scheduler.cuh`'s `packed_causal_kv_end`.
fn packed_causal_kv_end(
    qo_len: i64,
    kv_len: i64,
    qo_tile_idx: i64,
    cluster_tile_q: i64,
    num_qo_tiles: i64,
    group: i64,
) -> i64 {
    if qo_tile_idx + 1 == num_qo_tiles {
        return kv_len;
    }
    let kv_len_init = kv_len - qo_len;
    (kv_len_init + cdiv((qo_tile_idx + 1) * cluster_tile_q, group)).min(kv_len).max(0)
}

/// FlashInfer's MLA plan (`MLAPlan`): the persistent grid, each cluster's work items and
/// each merge CTA's rows.
#[derive(Clone, Debug)]
pub struct Plan {
    pub num_sm: usize,
    /// CTAs per cluster (FlashInfer's `num_blks_x`): 2 when the average packed query
    /// length is above 64.
    pub cluster_size: usize,
    /// Clusters (`num_blks_y`).
    pub num_clusters: usize,
    /// Most KV tokens per work item.
    pub kv_len_limit: usize,
    /// Cluster `c`'s work items are `work_indptr[c] .. work_indptr[c + 1]`.
    pub work_indptr: Vec<i32>,
    /// Per work item: `q_indptr, kv_indptr, partial_indptr (-1: unsplit), q_len, kv_len,
    /// q_start (packed), kv_start, kv_end`.
    pub work: Vec<i32>,
    /// Per CTA (`num_sm` entries): `packed_start, packed_end, partial_start, partial_end,
    /// partial_stride` and three zeros.
    pub merge: Vec<i32>,
    /// Rows of the partial buffers the plan uses.
    pub partial_rows: usize,
}

impl Plan {
    /// Plan for `qo_indptr`, `kv_indptr` (pages) and `kv_lens` on a GPU with `num_sm` SMs.
    pub fn new(
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        kv_lens: &[usize],
        num_heads: usize,
        causal: bool,
        num_sm: usize,
    ) -> Plan {
        let batch = kv_lens.len();
        let heads = num_heads as i64;
        let reqs: Vec<(usize, i64, i64)> =
            (0..batch).map(|i| (i, (qo_indptr[i + 1] - qo_indptr[i]) as i64, kv_lens[i] as i64)).collect();
        let accum: i64 = reqs.iter().map(|r| r.1 * heads).sum();
        let avg_packed = accum / batch as i64;
        let cluster_size = if avg_packed > 64 { 2 } else { 1 };
        let num_clusters = num_sm / cluster_size;
        let cluster_tile_q = (cluster_size * CTA_TILE_Q) as i64;

        let remaining_of = |qo_len: i64, kv_len: i64, tile: i64, num_tiles: i64| {
            if causal { packed_causal_kv_end(qo_len, kv_len, tile, cluster_tile_q, num_tiles, heads) } else { kv_len }
        };
        let mut total_kv: i64 = 0;
        for &(_, qo_len, kv_len) in &reqs {
            let num_tiles = cdiv(qo_len * heads, cluster_tile_q);
            for tile in (0..num_tiles).rev() {
                total_kv += remaining_of(qo_len, kv_len, tile, num_tiles);
            }
        }
        let f = |x: i64| -> i64 {
            match x {
                ..=8 => 32,
                9..=16 => 64,
                17..=32 => 128,
                33..=64 => 192,
                _ => cdiv(x, 256) * 256,
            }
        };
        let kv_len_limit = f(cdiv(total_kv, num_clusters as i64).max(1));

        let mut heap = MinHeap::new(num_clusters);
        let mut cluster_work: Vec<Vec<[i32; WORK_FIELDS]>> = vec![vec![]; num_clusters];
        let mut merge = vec![0i32; num_sm * MERGE_FIELDS];
        let mut merge_cta = 0usize;
        let mut partial_nnz: i64 = 0;
        for &(i, qo_len, kv_len) in &reqs {
            let packed_len = qo_len * heads;
            let num_tiles = cdiv(packed_len, cluster_tile_q);
            let qo_off = qo_indptr[i] as i64;
            for tile in (0..num_tiles).rev() {
                let mut remaining = remaining_of(qo_len, kv_len, tile, num_tiles);
                let mut kv_start = 0;
                let split = remaining > kv_len_limit;
                let row_tile = cluster_tile_q.min(packed_len - tile * cluster_tile_q);
                if split {
                    let chunks = (remaining * cluster_size as i64 / kv_len_limit).max(1);
                    let row_chunk = cdiv(row_tile, chunks);
                    let mut off = 0;
                    while off < row_tile {
                        let m = &mut merge[merge_cta * MERGE_FIELDS..(merge_cta + 1) * MERGE_FIELDS];
                        m[0] = (qo_off * heads + tile * cluster_tile_q + off) as i32;
                        m[1] = (qo_off * heads + tile * cluster_tile_q + (off + row_chunk).min(row_tile)) as i32;
                        m[2] = (partial_nnz + off) as i32;
                        m[3] = (partial_nnz + cdiv(remaining, kv_len_limit) * row_tile) as i32;
                        m[4] = row_tile as i32;
                        merge_cta += 1;
                        off += row_chunk;
                    }
                }
                let zero_len = remaining == 0;
                while remaining > 0 || zero_len {
                    let (c, cost) = heap.pop();
                    let actual = remaining.min(kv_len_limit);
                    heap.insert((c, cost + (2.0 * cluster_tile_q as f32 + actual as f32)));
                    let partial = if split {
                        let p = partial_nnz;
                        partial_nnz += row_tile;
                        p as i32
                    } else {
                        -1
                    };
                    cluster_work[c].push([
                        qo_indptr[i],
                        kv_indptr[i],
                        partial,
                        qo_len as i32,
                        kv_len as i32,
                        (tile * cluster_tile_q) as i32,
                        kv_start as i32,
                        (kv_start + actual) as i32,
                    ]);
                    remaining -= actual;
                    kv_start += actual;
                    if zero_len {
                        break;
                    }
                }
            }
        }
        assert!(merge_cta <= num_sm, "MLA plan: {merge_cta} merge CTAs for {num_sm} SMs");
        let mut work_indptr = vec![0i32];
        let mut work = vec![];
        for items in &cluster_work {
            for item in items {
                work.extend_from_slice(item);
            }
            work_indptr.push(work_indptr.last().unwrap() + items.len() as i32);
        }
        Plan {
            num_sm,
            cluster_size,
            num_clusters,
            kv_len_limit: kv_len_limit as usize,
            work_indptr,
            work,
            merge,
            partial_rows: partial_nnz as usize,
        }
    }

    pub fn num_works(&self) -> usize {
        self.work.len() / WORK_FIELDS
    }

    /// Whether any tile is split (and merged).
    pub fn split(&self) -> bool {
        self.partial_rows > 0
    }

    pub fn launch(&self) -> LaunchConfig {
        LaunchConfig::new([self.cluster_size as u32, self.num_clusters as u32, 1], [32, 4, 2]).with_cooperative(true)
    }
}

/// A test problem: ragged queries, per-request KV lengths, a page size and a page table.
pub struct Problem {
    pub qo_lens: Vec<usize>,
    pub kv_lens: Vec<usize>,
    pub page_size: usize,
    pub num_heads: usize,
    pub causal: bool,
    pub q_nope: Vec<F16>,
    pub q_pe: Vec<F16>,
    pub ckv: Vec<F16>,
    pub kpe: Vec<F16>,
    pub qo_indptr: Vec<i32>,
    pub kv_indptr: Vec<i32>,
    pub kv_indices: Vec<i32>,
    pub sm_scale: f32,
}

impl Problem {
    /// Random inputs in [-1, 1] and a shuffled page table (two unused pages).
    pub fn new(
        qo_lens: &[usize],
        kv_lens: &[usize],
        page_size: usize,
        num_heads: usize,
        causal: bool,
        seed: u64,
    ) -> Problem {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let pages_per: Vec<usize> = kv_lens.iter().map(|l| l.div_ceil(page_size)).collect();
        let num_pages: usize = pages_per.iter().sum::<usize>() + 2;
        let mut ids: Vec<i32> = (0..num_pages as i32).collect();
        for i in (1..ids.len()).rev() {
            let j = ((rnd() + 1.0) * 0.5 * (i + 1) as f32) as usize % (i + 1);
            ids.swap(i, j);
        }
        let mut p = Self::layout(qo_lens, kv_lens, page_size, num_heads, causal, num_pages, &ids);
        let mut fill = |v: &mut Vec<F16>| v.iter_mut().for_each(|x| *x = F16::from_f32(rnd()));
        fill(&mut p.q_nope);
        fill(&mut p.q_pe);
        fill(&mut p.ckv);
        fill(&mut p.kpe);
        p
    }

    /// A benchmark problem as FlashInfer's `bench_deepseek_mla.py`: `batch` requests of one
    /// query and `seq_len` tokens, page size 1, pages in order; the inputs are a cheap
    /// repeating pattern (their values do not change the timing).
    pub fn bench(batch: usize, seq_len: usize, num_heads: usize) -> Problem {
        let ids: Vec<i32> = (0..(batch * seq_len) as i32).collect();
        let mut p = Self::layout(&vec![1; batch], &vec![seq_len; batch], 1, num_heads, false, batch * seq_len, &ids);
        let table: Vec<F16> = (0..4093).map(|i| F16::from_f32(((i * 37 % 4093) as f32 / 4093.0) * 2.0 - 1.0)).collect();
        for v in [&mut p.q_nope, &mut p.q_pe, &mut p.ckv, &mut p.kpe] {
            let n = v.len();
            v.clear();
            while v.len() < n {
                v.extend_from_slice(&table[..table.len().min(n - v.len())]);
            }
        }
        p
    }

    /// The index arrays and zeroed buffers; request `b`'s pages are the next
    /// `ceil(kv_len / page_size)` entries of `ids`.
    fn layout(
        qo_lens: &[usize],
        kv_lens: &[usize],
        page_size: usize,
        num_heads: usize,
        causal: bool,
        num_pages: usize,
        ids: &[i32],
    ) -> Problem {
        let mut qo_indptr = vec![0i32];
        for &l in qo_lens {
            qo_indptr.push(qo_indptr.last().unwrap() + l as i32);
        }
        let mut kv_indptr = vec![0i32];
        let mut next = 0;
        for &l in kv_lens {
            next += l.div_ceil(page_size);
            kv_indptr.push(next as i32);
        }
        let n = *qo_indptr.last().unwrap() as usize;
        let zero = F16::from_f32(0.0);
        Problem {
            qo_lens: qo_lens.to_vec(),
            kv_lens: kv_lens.to_vec(),
            page_size,
            num_heads,
            causal,
            q_nope: vec![zero; n * num_heads * HEAD_DIM_CKV],
            q_pe: vec![zero; n * num_heads * HEAD_DIM_KPE],
            ckv: vec![zero; num_pages * page_size * HEAD_DIM_CKV],
            kpe: vec![zero; num_pages * page_size * HEAD_DIM_KPE],
            qo_indptr,
            kv_indptr,
            kv_indices: ids[..next].to_vec(),
            sm_scale: 1.0 / ((HEAD_DIM_CKV + HEAD_DIM_KPE) as f32).sqrt(),
        }
    }

    pub fn spec(&self) -> KernelSpec {
        KernelSpec { num_heads: self.num_heads, page_size: self.page_size, causal: self.causal }
    }

    /// Total query tokens.
    pub fn num_tokens(&self) -> usize {
        *self.qo_indptr.last().unwrap() as usize
    }

    pub fn plan(&self, num_sm: usize) -> Plan {
        Plan::new(&self.qo_indptr, &self.kv_indptr, &self.kv_lens, self.num_heads, self.causal, num_sm)
    }

    /// The kernel's arguments, in the order of [`p`].
    pub fn args(&self, plan: &Plan) -> Vec<KernelArg> {
        let rows = self.num_tokens() * self.num_heads;
        let prow = plan.partial_rows.max(1);
        vec![
            KernelArg::F16(self.q_nope.clone()),
            KernelArg::F16(self.q_pe.clone()),
            KernelArg::F16(self.ckv.clone()),
            KernelArg::F16(self.kpe.clone()),
            KernelArg::i32(if self.kv_indices.is_empty() { vec![0] } else { self.kv_indices.clone() }),
            KernelArg::i32(plan.work_indptr.clone()),
            KernelArg::i32(if plan.work.is_empty() { vec![0; WORK_FIELDS] } else { plan.work.clone() }),
            KernelArg::i32(plan.merge.clone()),
            KernelArg::F16(vec![F16::from_f32(0.0); rows * HEAD_DIM_CKV]),
            KernelArg::f32(vec![0.0; rows]),
            KernelArg::F16(vec![F16::from_f32(0.0); prow * HEAD_DIM_CKV]),
            KernelArg::f32(vec![0.0; prow]),
            KernelArg::i32(vec![0]),
            KernelArg::ScalarF32(self.sm_scale * std::f32::consts::LOG2_E),
        ]
    }

    /// Attention in f64 on the f16 inputs: `o [n, H, 512]` and base-2 `lse [n, H]`. A row
    /// that sees no key gets `o = 0`, `lse = -inf`, as FlashInfer's.
    pub fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        let (hh, ps) = (self.num_heads, self.page_size);
        let f = |x: F16| x.to_f32() as f64;
        let n = self.num_tokens();
        let mut o = vec![0f32; n * hh * HEAD_DIM_CKV];
        let mut lse = vec![f32::NEG_INFINITY; n * hh];
        for (b, (&ql, &kl)) in self.qo_lens.iter().zip(&self.kv_lens).enumerate() {
            let row_of = |t: usize| {
                let page = self.kv_indices[self.kv_indptr[b] as usize + t / ps] as usize;
                page * ps + t % ps
            };
            for i in 0..ql {
                let tok = self.qo_indptr[b] as usize + i;
                let visible = if self.causal { (kl + i + 1).saturating_sub(ql).min(kl) } else { kl };
                for hd in 0..hh {
                    let qrow = tok * hh + hd;
                    let s: Vec<f64> = (0..visible)
                        .map(|t| {
                            let r = row_of(t);
                            let nope: f64 = (0..HEAD_DIM_CKV)
                                .map(|e| f(self.q_nope[qrow * HEAD_DIM_CKV + e]) * f(self.ckv[r * HEAD_DIM_CKV + e]))
                                .sum();
                            let pe: f64 = (0..HEAD_DIM_KPE)
                                .map(|e| f(self.q_pe[qrow * HEAD_DIM_KPE + e]) * f(self.kpe[r * HEAD_DIM_KPE + e]))
                                .sum();
                            (nope + pe) * self.sm_scale as f64
                        })
                        .collect();
                    if s.is_empty() {
                        continue;
                    }
                    let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                    let sum: f64 = w.iter().sum();
                    lse[qrow] = ((mx + sum.ln()) * std::f64::consts::LOG2_E) as f32;
                    for e in 0..HEAD_DIM_CKV {
                        let acc: f64 = (0..visible).map(|t| w[t] * f(self.ckv[row_of(t) * HEAD_DIM_CKV + e])).sum();
                        o[qrow * HEAD_DIM_CKV + e] = (acc / sum) as f32;
                    }
                }
            }
        }
        (o, lse)
    }
}

/// Largest `|a - b|`, counting equal infinities as equal.
pub fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| if x == y { 0.0 } else { (x - y).abs() }).fold(0.0, f32::max)
}

/// Simulate `p` under a plan for `num_sm` SMs and return the plan and `(max |o - ref|,
/// max |lse - ref|)`.
pub fn simulate(p: &Problem, num_sm: usize) -> (Plan, f32, f32) {
    let plan = p.plan(num_sm);
    let mut args = p.args(&plan);
    let k = kernel(&p.spec());
    k.simulate(plan.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let o: Vec<f32> = args[p::O].to_vec::<F16>().into_iter().map(|x| x.to_f32()).collect();
    let lse = args[p::LSE].as_f32().to_vec();
    let (want_o, want_lse) = p.reference();
    (plan, max_error(&o, &want_o), max_error(&lse, &want_lse))
}

/// Running on the GPU (`--features cuda`).
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::error::Error;
    use std::rc::Rc;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg, HardwareInfo};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{KernelSpec, Plan, Problem, kernel, max_error};

    type Loaded<'c> = Rc<(Kernel, CudaFunction<'c>)>;

    /// The MLA kernels, compiled and loaded when first needed.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: Toolchain,
        opts: CompileOptions,
        cache: RefCell<HashMap<KernelSpec, Loaded<'c>>>,
        pub num_sm: usize,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: Toolchain) -> Result<Kernels<'c>, Box<dyn Error>> {
            let arch = ctx.best_target(&tc).ok_or("no target for this GPU")?;
            let opts = CompileOptions::new().gpu_arch(arch);
            let num_sm = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            Ok(Kernels { ctx, tc, opts, cache: RefCell::new(HashMap::new()), num_sm })
        }

        /// The kernel for `spec`, compiled on first use.
        pub fn get(&self, spec: &KernelSpec) -> Result<Loaded<'c>, Box<dyn Error>> {
            if let Some(k) = self.cache.borrow().get(spec) {
                return Ok(k.clone());
            }
            let k = kernel(spec);
            let f = self.ctx.load(&k.compile_with(&self.tc, &self.opts)?)?;
            let loaded = Rc::new((k, f));
            self.cache.borrow_mut().insert(*spec, loaded.clone());
            Ok(loaded)
        }
    }

    /// Run `p` once on the GPU; if `check`, compare with the f64 reference; if `time`,
    /// time the launch replayed from a CUDA graph, `iterations` times. Returns the plan,
    /// the errors and the average GPU time in microseconds.
    #[allow(clippy::type_complexity)]
    pub fn run(
        ks: &Kernels<'_>,
        p: &Problem,
        check: bool,
        time: Option<usize>,
    ) -> Result<(Plan, f32, f32, Option<f64>), Box<dyn Error>> {
        let ctx = ks.ctx;
        let plan = p.plan(ks.num_sm);
        let loaded = ks.get(&p.spec())?;
        let (k, f) = (&loaded.0, &loaded.1);
        let host = p.args(&plan);
        let mut bufs = Vec::new();
        for a in &host {
            if !matches!(a, KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_)) {
                bufs.push(ctx.upload(a)?);
            }
        }
        drop(host);
        let scalar = KernelArg::ScalarF32(p.sm_scale * std::f32::consts::LOG2_E);
        let mut args: Vec<DeviceArg> = bufs.iter().map(DeviceArg::Buffer).collect();
        args.push(DeviceArg::Scalar(scalar));
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(k, f, plan.launch()).stream(s).args(&args).submit()?;
            Ok(())
        };
        let stream = ctx.stream()?;
        launch(&stream, 0)?;
        stream.synchronize()?;
        let (mut eo, mut el) = (0.0, 0.0);
        if check {
            use super::p as pp;
            let o: Vec<f32> = bufs[pp::O].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect();
            let lse = bufs[pp::LSE].to_vec::<f32>()?;
            let (want_o, want_lse) = p.reference();
            (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
        }
        let t = match time {
            Some(iterations) => {
                let opts = BenchmarkOptions::new().warmup_iterations(3).iterations(iterations).use_cuda_graphs(true);
                Some(benchmark(&stream, &opts, &mut launch)?)
            }
            None => None,
        };
        Ok((plan, eo, el, t))
    }
}
