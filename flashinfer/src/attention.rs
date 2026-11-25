// SPDX-License-Identifier: Apache-2.0
//
// Copyright (c) 2025 by FlashInfer team.
//
// The planner is translated from FlashInfer v0.7.5's `TwoStageHolisticPlan`
// (`include/flashinfer/attention/scheduler.cuh`, with `heap.h`'s `MinHeap`); the kernel's
// structure (two persistent runners, a grid-wide barrier, then a persistent reduction)
// follows `include/flashinfer/attention/persistent.cuh` and `persistent_template.cuh`.

//! FlashInfer's persistent `BatchAttention` (`flashinfer/attention/_core.py`): one launch
//! runs a mixed batch of prefill and decode requests over a paged KV cache from per-CTA
//! work lists, then merges the split-KV partial states in the same kernel.
//!
//! * **Plan** ([`Plan::new`], on the host, translated from `TwoStageHolisticPlan`): the
//!   grid is `2 * SMs` persistent CTAs. Requests whose packed query length (`qo_len *
//!   group`) is above 16 are task 0 (128-row query tiles), the others task 1 (16-row
//!   tiles). Each task gets a KV chunk length `len_kv_chunk`: the total (causal-trimmed)
//!   KV per CTA over both tasks times the KV heads, rounded up to 128 or a multiple of
//!   256, and halved for task 0 when there are two or more KV heads. Every (query tile,
//!   KV chunk, KV head) goes to the least-loaded CTA (FlashInfer's `MinHeap`, cost `2 *
//!   tile + chunk`, one heap for both tasks, ties broken as libstdc++'s heap does). A tile
//!   whose KV is longer than the chunk is split, and its packed rows become merge rows.
//! * **Kernel** (grid `2 * SMs`, 128 threads, a cooperative launch): each CTA runs its
//!   task-0 work items and then its task-1 items, each the [`prefill`](crate::prefill)
//!   kernel's CTA body (`prefill::body`) on one (request, query tile, KV chunk, KV head),
//!   as FlashInfer's `BlockBatchPagedAttentionPersistent` is its prefill kernel's
//!   mainloop. An unsplit item writes `o` and `lse` directly; a split one writes its f32
//!   partial state. Then a grid-wide barrier (FlashInfer's `grid.sync()`; cooperative
//!   groups' phase-bit counter, as the [`mla`](crate::mla) port) and the persistent
//!   reduction (`BlockBatchReductionPersistent`): each warp of the grid takes (merge row,
//!   KV head) pairs in turn and folds the row's partial states, four columns per lane.
//!
//! Differences from FlashInfer:
//! * Tile shapes. FlashInfer fixes `NUM_MMA_KV` at 4 (128-row tile) and 2 (16-row tile),
//!   64 and 68 KB of shared memory; with `2 * SMs` CTAs, a cooperative launch on a GPU
//!   with 100 KB per SM (SM89) fails ("too many blocks in cooperative launch"), so
//!   FlashInfer's `BatchAttention` does not run on SM89. This port uses the prefill port's
//!   SM89 tiles (32- and 64-token KV tiles, at most 50 KB), two CTAs per SM.
//! * Partial states are f32 in the prefill port's layout (row `o_indptr[b] + q *
//!   num_chunks + chunk`, `num_chunks = ceil(kv_len / len_kv_chunk)`), where FlashInfer's
//!   are f16 `[packed row x chunk, kv_head]`; the merge rows point into it.
//! * The GQA group and page size are build-time constants (as in the prefill port), and
//!   the task bodies are built only in the forms a plan uses (split and/or unsplit).
//! * The merge row count is a scalar argument (FlashInfer reads it from the plan buffer,
//!   for CUDA-graph re-planning, which is not ported).
//!
//! Float16 only (Q, K/V, output), head dim 128, NHD cache, causal or not, any GQA group;
//! no attention variants (soft-cap, window), FP8, profiler or CUDA-graph planning.
//! Causal requests need `kv_len >= qo_len`.
//!
//! Measured on an RTX 2000 Ada (22 SMs, ~224 GB/s), page size 1, CUDA-graph replays, on
//! `bench_batch_attention.py`'s mixes (Qwen-8B heads 32/8, f16), against FlashInfer's other
//! two schedulers in that benchmark (its `BatchAttention` cannot launch on SM89):
//!
//! | mix (decodes x kv + prefill chunk / kv) | this | FlashInfer batch prefill | FlashInfer decode + prefill | FlashInfer POD |
//! |---|---|---|---|---|
//! | 128 x 8192 + 4096 / 8192 (the default) | 30.1 ms | 31.2 ms | 31.6 ms | 22.4 ms |
//! | 128 x 2048 + 1024 / 2048 | 5.91 ms | 5.67 ms | 5.80 ms | 5.34 ms |
//! | 32 x 8192 + 4096 / 8192 | 15.8 ms | 15.6 ms | 16.3 ms | 12.7 ms |
//! | 64 x 4096 + 2 x 2048 / 4096 | 10.2 ms | 10.6 ms | 10.9 ms | - |
//! | 128 x 8192, decodes only | 20.6 ms (209 GB/s) | 20.4 ms | 20.4 ms | - |
//!
//! FlashInfer's order (each CTA's prefill items, then its decode items) runs the
//! compute-bound and bandwidth-bound halves one after the other, so it is no faster than
//! the two halves launched apart; POD, which pairs them on each SM, is 25% faster on the
//! default mix. [`AttentionSpec::alternate`] (the second half of the grid takes its decode
//! items first) recovers part of that: 28.7 ms on the default mix, 5.76 ms on the second.
//!
//! `examples/attention.rs` runs it on the simulator and, with `--features cuda` and
//! `GPU=1`, on the GPU, then benchmarks it on the shapes of FlashInfer's
//! `benchmarks/bench_batch_attention.py` (`examples/attention_flashinfer_bench.py`).

use cute::prelude::*;

use crate::prefill::{self, CtaCoords, KernelSpec, Params, TileConfig};

/// Threads per CTA (both tiles have four warps).
pub const THREADS: usize = 128;
/// Query tile of each task (FlashInfer's `CTA_TILE_Q_SIZES`).
pub const CTA_TILE_Q: [usize; 2] = [128, 16];
/// i32 fields per merge row: tmp row of chunk 0, chunk count, query token, head in group.
pub const MERGE_FIELDS: usize = 4;
/// Elements a vector view may address.
const BOUND: usize = 1 << 30;

/// Parameter positions: the shared tensors, each task's work lists (`TASK[t] + task::*`),
/// then the merge rows, the barrier and the scalars.
pub mod p {
    pub const Q: usize = 0;
    pub const K: usize = 1;
    pub const V: usize = 2;
    pub const QO_INDPTR: usize = 3;
    pub const KV_INDPTR: usize = 4;
    pub const KV_INDICES: usize = 5;
    pub const LAST_PAGE_LEN: usize = 6;
    pub const O_INDPTR: usize = 7;
    pub const O: usize = 8;
    pub const LSE: usize = 9;
    pub const TMP_O: usize = 10;
    pub const TMP_LSE: usize = 11;
    pub const TASK: [usize; 2] = [12, 18];
    pub const MERGE: usize = 24;
    pub const NUM_MERGE: usize = 25;
    pub const BARRIER: usize = 26;
    pub const NUM_KV_HEADS: usize = 27;
    pub const SM_SCALE_LOG2: usize = 28;
    pub const COUNT: usize = 29;

    /// A task's arguments, from `TASK[t]`: CTA `c`'s items are `WORK_INDPTR[c] ..
    /// WORK_INDPTR[c + 1]`; item `w` is request `REQUEST[w]`, query tile `QO_TILE[w]`, KV
    /// chunk `KV_TILE[w]`, `AUX[w] = 2 * kv_head + split`; `CHUNK` is `len_kv_chunk`.
    pub mod task {
        pub const WORK_INDPTR: usize = 0;
        pub const REQUEST: usize = 1;
        pub const QO_TILE: usize = 2;
        pub const KV_TILE: usize = 3;
        pub const AUX: usize = 4;
        pub const CHUNK: usize = 5;
    }
}

/// What one task's bodies are built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskSpec {
    pub tile: TileConfig,
    /// Items that write partial states, and items that write through.
    pub split: bool,
    pub unsplit: bool,
}

/// What a kernel is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionSpec {
    pub tasks: [Option<TaskSpec>; 2],
    pub group: usize,
    pub page_size: usize,
    pub causal: bool,
    /// Run task 1 before task 0 in the second half of the grid (not FlashInfer's order;
    /// overlaps compute- and bandwidth-bound work on an SM).
    pub alternate: bool,
}

impl AttentionSpec {
    pub fn name(&self) -> String {
        let task = |t: &Option<TaskSpec>| match t {
            None => "x".to_string(),
            Some(t) => {
                format!("{}{}{}", t.tile.cta_tile_q, if t.split { "s" } else { "" }, if t.unsplit { "u" } else { "" })
            }
        };
        format!(
            "flashinfer_batch_attention_g{}_p{}{}{}_{}_{}",
            self.group,
            self.page_size,
            if self.causal { "_causal" } else { "" },
            if self.alternate { "_alt" } else { "" },
            task(&self.tasks[0]),
            task(&self.tasks[1])
        )
    }

    /// Whether the kernel merges partial states (after the grid barrier).
    pub fn reduce(&self) -> bool {
        self.tasks.iter().flatten().any(|t| t.split)
    }

    /// Shared memory per CTA, in bytes: the larger task's.
    pub fn smem_bytes(&self) -> usize {
        self.tasks.iter().flatten().map(|t| t.tile.smem_bytes()).max().unwrap_or(16)
    }

    fn body_spec(&self, t: &TaskSpec, partition: bool) -> KernelSpec {
        KernelSpec {
            tile: t.tile,
            group: self.group,
            page_size: self.page_size,
            causal: self.causal,
            partition,
            variant: prefill::Variant::NONE,
        }
    }
}

/// The persistent kernel for `spec`; parameters in the order of [`p`].
pub fn kernel(spec: &AttentionSpec) -> Kernel {
    let spec = *spec;
    let g = spec.group as i32;
    let elems = (spec.smem_bytes() / 2).max(8);
    Kernel::builder(spec.name()).block_dim([THREADS as u32, 1, 1]).build(|| {
        let q = param_ptr::<F16>("q");
        let k_cache = param_ptr::<F16>("k_cache");
        let v_cache = param_ptr::<F16>("v_cache");
        let qo_indptr = param_ptr::<i32>("qo_indptr");
        let kv_indptr = param_ptr::<i32>("kv_indptr");
        let kv_indices = param_ptr::<i32>("kv_indices");
        let kv_last_page_len = param_ptr::<i32>("kv_last_page_len");
        let o_indptr = param_ptr::<i32>("o_indptr");
        let o = param_ptr::<F16>("o");
        let lse = param_ptr::<f32>("lse");
        let tmp_o = param_ptr::<f32>("tmp_o");
        let tmp_lse = param_ptr::<f32>("tmp_lse");
        let tasks: Vec<_> = (0..2)
            .map(|t| {
                let n = |s: &str| format!("t{t}_{s}");
                (
                    param_ptr::<i32>(&n("work_indptr")),
                    param_ptr::<i32>(&n("request_indices")),
                    param_ptr::<i32>(&n("qo_tile_indices")),
                    param_ptr::<i32>(&n("kv_tile_indices")),
                    param_ptr::<i32>(&n("aux")),
                    param_scalar::<i32>(&n("len_kv_chunk")),
                )
            })
            .collect();
        let merge = param_ptr::<i32>("merge");
        let num_merge = param_scalar::<i32>("num_merge");
        let barrier = param_ptr::<i32>("barrier");
        let num_kv_heads = param_scalar::<i32>("num_kv_heads");
        let sm_scale_log2 = param_scalar::<f32>("sm_scale_log2");

        let smem = make_smem_tensor_aligned::<F16>(Layout::col_major(elems), 16);
        let tid = thread_idx().x;
        let (lane, warp) = (tid % 32, tid / 32);
        let cta = block_idx().x;
        let num_ctas = grid_dim().x;

        // ---- Phase 1: this CTA's work items, task 0 then task 1 (FlashInfer's two
        // persistent runners), or, with `alternate`, task 1 first in the second half of the
        // grid.
        let run_task = |t: usize| {
            let (work_indptr, req, qt, kt, aux, chunk) = &tasks[t];
            let Some(ts) = spec.tasks[t] else { return };
            // The prefill body's parameters: the shared tensors and this task's lists. The
            // variant parameters are unused by a plain (no-variant, Float16) body.
            let prm = Params {
                q: q.clone(),
                k_cache: k_cache.clone(),
                v_cache: v_cache.clone(),
                qo_indptr: qo_indptr.clone(),
                kv_indptr: kv_indptr.clone(),
                kv_indices: kv_indices.clone(),
                kv_last_page_len: kv_last_page_len.clone(),
                request_indices: req.clone(),
                qo_tile_indices: qt.clone(),
                kv_tile_indices: kt.clone(),
                o_indptr: o_indptr.clone(),
                o: o.clone(),
                lse: lse.clone(),
                tmp_o: tmp_o.clone(),
                tmp_lse: tmp_lse.clone(),
                num_kv_heads,
                sm_scale_log2,
                kv_chunk_size: *chunk,
                logits_pre_scale: Float32::from(1.0f32),
                window_left: Int32::from(-1),
                alibi_slopes: tmp_lse.clone(),
                custom_mask: recast_ptr::<u8, i32>(&o_indptr, None),
                mask_indptr: o_indptr.clone(),
                v_scale: Float32::from(1.0f32),
            };
            let nwq = ts.tile.num_warps_q as i32;
            let (w_begin, w_end) = (work_indptr.load_at(cta), work_indptr.load_at(cta + 1));
            for_range(w_begin, w_end, 1, |w| {
                // The previous item's output staging aliases the Q tile.
                sync_threads();
                let a = aux.load_at(w);
                // The thread's coordinates, re-derived per item through `a >> 16` (always 0,
                // which the compiler cannot see): otherwise LLVM hoists the body's per-thread
                // address arithmetic out of the work loop and keeps it live across the item,
                // which spills.
                let t = tid + (a >> 16);
                let (lane, warp) = (t % 32, t / 32);
                let at = CtaCoords { lane, wq: warp % nwq, wkv: warp / nwq, kv_head: (a & 0xffff) / 2, cta: w };
                let run = |partition: bool| prefill::body::<F16>(&spec.body_spec(&ts, partition), &prm, &smem, at);
                match (ts.split, ts.unsplit) {
                    (true, true) => if_else((a % 2).ne(0), || run(true), || run(false)),
                    (split, _) => run(split),
                }
            });
            sync_threads();
        };
        if spec.alternate && spec.tasks.iter().all(|t| t.is_some()) {
            // The two CTAs an SM holds are (usually) `c` and `c + num_ctas / 2`: one starts
            // on the compute-bound tiles, the other on the bandwidth-bound ones.
            let flip = select(cta.ge(num_ctas / 2), Int32::from(1), Int32::from(0));
            for_range(0, 2, 1, |pass| {
                let t = pass ^ flip;
                if_(t.eq(0), || run_task(0));
                if_(t.eq(1), || run_task(1));
            });
        } else {
            run_task(0);
            run_task(1);
        }
        if !spec.reduce() {
            return;
        }

        // ---- Grid-wide barrier (FlashInfer's `grid.sync()`, cooperative groups' scheme, as
        // the MLA port): bit 31 of the counter is the phase. Every CTA adds 1 except CTA 0,
        // which adds `0x8000_0000 - (n - 1)`, so the phase bit flips when the last CTA
        // arrives; the counter returns to its old low bits, so any number of launches is
        // safe. TODO: dedupe with mla.rs.
        if_(tid.eq(0), || {
            fence_acq_rel_gpu();
            let add = select(cta.eq(0), Int32::from(i32::MIN) - (num_ctas - 1), Int32::from(1));
            let old = atomic_add_with(&barrier, add, AtomicOptions::new(MemoryOrder::Relaxed, MemoryScope::Gpu));
            let opts =
                LoadOptions { sem: MemoryOrder::Relaxed, scope: Some(MemoryScope::Gpu), ..LoadOptions::default() };
            let bar = barrier.clone();
            let _ = while_carry(old, |v: Int32| (v ^ old).ge(0), |_v: Int32| load(&bar, opts));
            fence_acq_rel_gpu();
        });
        sync_threads();

        // ---- Phase 2: the persistent reduction. Warp `cta * 4 + warp` of the grid takes
        // (merge row, KV head) pairs `i`, `i + 4 * grid`, ...; lane `l` folds columns `4 l
        // .. 4 l + 3` of the row's partial states (FlashInfer's `state_t::merge`).
        let num_qo_heads = num_kv_heads * g;
        let quads = |ptr: &Ptr<f32>| make_tensor(ptr, Layout::col_major((4, BOUND / 4)));
        let tmp_q = quads(&tmp_o);
        let o_q = make_tensor(&o, Layout::col_major((4, BOUND / 4)));
        let zero = Float32::from(0.0f32);
        let neg_inf = Float32::from(f32::NEG_INFINITY);
        let workers = num_ctas * (THREADS / 32) as i32;
        for_range(cta * (THREADS / 32) as i32 + warp, num_merge * num_kv_heads, workers, |i| {
            let (row, kv_head) = (i / num_kv_heads, i % num_kv_heads);
            let field = |f: i32| merge.load_at(row * MERGE_FIELDS as i32 + f);
            let (start, count, token, r) = (field(0), field(1), field(2), field(3));
            let head = kv_head * g + r;
            let m = Var::new(neg_inf);
            let d = Var::new(zero);
            let acc = Var::new(vec![zero; 4]);
            for_range(0, count, 1, |c| {
                let at = (start + c) * num_qo_heads + head;
                let l = tmp_lse.load_at(at);
                let v = tmp_q.slice((.., at * 32 + lane)).values();
                // A row with no visible key in this chunk has lse -inf (and o NaN): skip it.
                let valid = l.gt(neg_inf);
                let (m_prev, d_prev) = (m.get(), d.get());
                let m_new = fmax(m_prev, l);
                let f1 = select(m_prev.gt(neg_inf), cute::math::exp2(m_prev - m_new, true), zero);
                let f2 = select(valid, cute::math::exp2(l - m_new, true), zero);
                d.set(d_prev * f1 + f2);
                let a: Vec<Float32> =
                    acc.get().iter().zip(&v).map(|(a, x)| *a * f1 + select(valid, *x, zero) * f2).collect();
                acc.set(a);
                m.set(m_new);
            });
            let dd = d.get();
            let d_rcp = select(dd.gt(zero), Float32::from(1.0f32) / dd, zero);
            let out: Vec<Value<F16>> = acc.get().iter().map(|a| (*a * d_rcp).to::<F16>()).collect();
            let orow = token * num_qo_heads + head;
            o_q.slice((.., orow * 32 + lane)).store_values(&out);
            if_(lane.eq(0), || lse.store_at(orow, m.get() + cute::math::log2(dd, true)));
        });
    })
}

/// FlashInfer's `MinHeap` (`heap.h`) over `std::push_heap` / `std::pop_heap` with the
/// comparator `a.cost > b.cost`, step for step as libstdc++. TODO: dedupe with mla.rs.
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
            let value = self.heap[len - 1];
            self.heap[len - 1] = self.heap[0];
            let n = len - 1;
            let (mut hole, mut child) = (0, 0);
            while child < n.saturating_sub(1) / 2 {
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

fn cdiv(a: i64, b: i64) -> i64 {
    (a + b - 1).div_euclid(b)
}

/// `scheduler.cuh`'s `packed_causal_kv_end`.
fn packed_causal_kv_end(qo_len: i64, kv_len: i64, tile: i64, tile_q: i64, num_tiles: i64, group: i64) -> i64 {
    if tile + 1 == num_tiles {
        return kv_len;
    }
    (kv_len - qo_len + cdiv((tile + 1) * tile_q, group)).min(kv_len).max(0)
}

/// A work item: request, query tile, KV chunk, KV head, split.
type WorkItem = (i32, i32, i32, i32, bool);

/// One task's share of the plan.
#[derive(Clone, Debug, Default)]
pub struct TaskPlan {
    /// FlashInfer's `len_kv_chunk` for the task, in tokens.
    pub len_kv_chunk: usize,
    /// CTA `c`'s items are `work_indptr[c] .. work_indptr[c + 1]`.
    pub work_indptr: Vec<i32>,
    pub request_indices: Vec<i32>,
    pub qo_tile_indices: Vec<i32>,
    pub kv_tile_indices: Vec<i32>,
    pub kv_heads: Vec<i32>,
    pub split: Vec<bool>,
}

impl TaskPlan {
    pub fn num_works(&self) -> usize {
        self.request_indices.len()
    }
}

/// FlashInfer's `TwoStageHolisticPlan`: the persistent grid, each task's work items per
/// CTA, the partial-state rows and the merge rows.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Persistent CTAs (FlashInfer's `num_blks_y`; clusters are one CTA): 2 per SM.
    pub num_ctas: usize,
    pub tasks: [TaskPlan; 2],
    /// Request `b`'s partial states start at row `o_indptr[b]`.
    pub o_indptr: Vec<i32>,
    /// Merge rows, [`MERGE_FIELDS`] each.
    pub merge: Vec<i32>,
}

impl Plan {
    /// Plan requests of `qo_lens` queries and `kv_lens` tokens on a GPU with `num_sm` SMs.
    pub fn new(
        qo_lens: &[usize],
        kv_lens: &[usize],
        group: usize,
        num_kv_heads: usize,
        causal: bool,
        num_sm: usize,
    ) -> Plan {
        let num_ctas = 2 * num_sm; // head dim < 256: two CTAs per SM
        let (g, kvh) = (group as i64, num_kv_heads as i64);
        let batch = qo_lens.len();
        let task_of = |b: usize| if qo_lens[b] * group > CTA_TILE_Q[1] { 0 } else { 1 };
        let tiles = |b: usize| cdiv((qo_lens[b] * group) as i64, CTA_TILE_Q[task_of(b)] as i64);
        let remaining = |b: usize, tile: i64| {
            let (ql, kl) = (qo_lens[b] as i64, kv_lens[b] as i64);
            if causal { packed_causal_kv_end(ql, kl, tile, CTA_TILE_Q[task_of(b)] as i64, tiles(b), g) } else { kl }
        };
        let total_kv: i64 = (0..batch).map(|b| (0..tiles(b)).map(|t| remaining(b, t)).sum::<i64>()).sum();
        let f = |x: i64| if x <= 128 { 128 } else { cdiv(x, 256) * 256 };
        let limit: [i64; 2] = std::array::from_fn(|t| {
            let l = f(cdiv(total_kv * kvh, num_ctas as i64).max(1));
            if CTA_TILE_Q[t] >= 64 { l / kvh.min(2) } else { l }
        });

        // Partial-state rows: a request with a split tile gets `ceil(kv_len / chunk)` rows
        // per query (the prefill body's layout).
        let mut o_indptr = vec![0i32];
        let num_chunks = |b: usize| cdiv(kv_lens[b] as i64, limit[task_of(b)]);
        for b in 0..batch {
            let split = (0..tiles(b)).any(|t| remaining(b, t) > limit[task_of(b)]);
            let rows = if split { qo_lens[b] as i64 * num_chunks(b) } else { 0 };
            o_indptr.push(o_indptr[b] + rows as i32);
        }

        let mut heap = MinHeap::new(num_ctas);
        let mut merge = vec![];
        let mut qo_indptr = vec![0i64];
        for &l in qo_lens {
            qo_indptr.push(qo_indptr.last().unwrap() + l as i64);
        }
        let tasks = std::array::from_fn(|t| {
            let tile_q = CTA_TILE_Q[t] as i64;
            let chunk = limit[t];
            let mut per_cta: Vec<Vec<WorkItem>> = vec![vec![]; num_ctas];
            for b in (0..batch).filter(|&b| task_of(b) == t) {
                let packed = (qo_lens[b] * group) as i64;
                for tile in 0..tiles(b) {
                    let mut rem = remaining(b, tile);
                    let split = rem > chunk;
                    let num_kv_tiles = if split { cdiv(rem, chunk) } else { 1 };
                    let row_tile = tile_q.min(packed - tile * tile_q);
                    let (mut kv_tile, zero_len) = (0, rem == 0);
                    while rem > 0 || zero_len {
                        let actual = rem.min(chunk);
                        for h in 0..num_kv_heads {
                            let (c, cost) = heap.pop();
                            heap.insert((c, cost + (2.0 * tile_q as f32 + actual as f32)));
                            per_cta[c].push((b as i32, tile as i32, kv_tile, h as i32, split));
                        }
                        rem -= actual;
                        kv_tile += 1;
                        if zero_len || rem == 0 {
                            break;
                        }
                    }
                    if split {
                        for row in 0..row_tile {
                            let p = tile * tile_q + row;
                            let (qi, r) = (p / g, p % g);
                            merge.extend_from_slice(&[
                                (o_indptr[b] as i64 + qi * num_chunks(b)) as i32,
                                num_kv_tiles as i32,
                                (qo_indptr[b] + qi) as i32,
                                r as i32,
                            ]);
                        }
                    }
                }
            }
            let mut tp = TaskPlan { len_kv_chunk: chunk as usize, work_indptr: vec![0], ..TaskPlan::default() };
            for items in &per_cta {
                for &(b, qt, kt, h, s) in items {
                    tp.request_indices.push(b);
                    tp.qo_tile_indices.push(qt);
                    tp.kv_tile_indices.push(kt);
                    tp.kv_heads.push(h);
                    tp.split.push(s);
                }
                tp.work_indptr.push(tp.request_indices.len() as i32);
            }
            tp
        });
        Plan { num_ctas, tasks, o_indptr, merge }
    }

    pub fn num_merge(&self) -> usize {
        self.merge.len() / MERGE_FIELDS
    }

    /// Rows of partial-state scratch (at least one, so buffers are never empty).
    pub fn tmp_rows(&self) -> usize {
        (*self.o_indptr.last().unwrap() as usize).max(1)
    }

    pub fn launch(&self) -> LaunchConfig {
        LaunchConfig::new([self.num_ctas as u32, 1, 1], [THREADS as u32, 1, 1]).with_cooperative(true)
    }
}

/// A mixed batch: ragged queries and a paged NHD cache (a [`prefill::Problem`], Float16,
/// no variants).
pub struct Problem {
    pub batch: prefill::Problem,
}

impl Problem {
    /// Random inputs: requests of `qo_lens` queries over `kv_lens` tokens, pages of
    /// `page_size` (a shuffled page table).
    pub fn new(
        qo_lens: &[usize],
        kv_lens: &[usize],
        page_size: usize,
        num_kv_heads: usize,
        group: usize,
        causal: bool,
        seed: u64,
    ) -> Problem {
        Problem::from_batch(prefill::Problem::new(qo_lens, kv_lens, page_size, num_kv_heads, group, causal, seed))
    }

    /// A problem on a paged prefill problem (checked: paged, Float16, no variants, causal
    /// requests with `kv_len >= qo_len`).
    pub fn from_batch(batch: prefill::Problem) -> Problem {
        assert!(!batch.is_ragged(), "BatchAttention reads a paged cache");
        assert_eq!(batch.kv_dtype, prefill::KvDtype::F16, "Float16 only");
        assert_eq!(batch.variant, prefill::Variant::NONE, "no attention variants");
        if batch.causal {
            assert!(batch.qo_lens.iter().zip(&batch.kv_lens).all(|(q, k)| k >= q), "causal needs kv_len >= qo_len");
        }
        Problem { batch }
    }

    pub fn plan(&self, num_sm: usize) -> Plan {
        let b = &self.batch;
        Plan::new(&b.qo_lens, &b.kv_lens, b.group, b.num_kv_heads, b.causal, num_sm)
    }

    pub fn spec(&self, plan: &Plan) -> AttentionSpec {
        let tasks = std::array::from_fn(|t| {
            let tp = &plan.tasks[t];
            (tp.num_works() > 0).then(|| TaskSpec {
                tile: TileConfig::sm89(CTA_TILE_Q[t]),
                split: tp.split.iter().any(|&s| s),
                unsplit: tp.split.iter().any(|&s| !s),
            })
        });
        AttentionSpec {
            tasks,
            group: self.batch.group,
            page_size: self.batch.page_size,
            causal: self.batch.causal,
            alternate: false,
        }
    }

    /// The kernel's arguments, in the order of [`p`] (a zeroed barrier).
    pub fn args(&self, plan: &Plan) -> Vec<KernelArg> {
        let b = &self.batch;
        let nonempty = |v: Vec<i32>| if v.is_empty() { vec![0] } else { v };
        let rows = plan.tmp_rows();
        let mut args = vec![
            KernelArg::buffer(&b.q),
            KernelArg::buffer(&b.k_cache),
            KernelArg::buffer(&b.v_cache),
            KernelArg::i32(b.qo_indptr.clone()),
            KernelArg::i32(b.kv_indptr.clone()),
            KernelArg::i32(nonempty(b.kv_indices.clone())),
            KernelArg::i32(b.kv_last_page_len.clone()),
            KernelArg::i32(plan.o_indptr.clone()),
            KernelArg::buffer(&vec![F16::from_f32(0.0); b.q.len()]),
            KernelArg::f32(vec![0.0; b.q.len() / prefill::HEAD_DIM]),
            KernelArg::f32(vec![0.0; rows * b.num_qo_heads() * prefill::HEAD_DIM]),
            KernelArg::f32(vec![0.0; rows * b.num_qo_heads()]),
        ];
        for tp in &plan.tasks {
            let aux = tp.kv_heads.iter().zip(&tp.split).map(|(&h, &s)| 2 * h + s as i32).collect();
            args.extend([
                KernelArg::i32(tp.work_indptr.clone()),
                KernelArg::i32(nonempty(tp.request_indices.clone())),
                KernelArg::i32(nonempty(tp.qo_tile_indices.clone())),
                KernelArg::i32(nonempty(tp.kv_tile_indices.clone())),
                KernelArg::i32(nonempty(aux)),
                KernelArg::ScalarI32(tp.len_kv_chunk as i32),
            ]);
        }
        args.extend([
            KernelArg::i32(nonempty(plan.merge.clone())),
            KernelArg::ScalarI32(plan.num_merge() as i32),
            KernelArg::i32(vec![0]),
            KernelArg::ScalarI32(b.num_kv_heads as i32),
            KernelArg::ScalarF32(b.sm_scale * std::f32::consts::LOG2_E),
        ]);
        assert_eq!(args.len(), p::COUNT);
        args
    }

    /// The f64 reference: `o` and base-2 `lse`.
    pub fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        self.batch.reference()
    }

    /// Floating-point operations (`flashinfer.testing.attention_flops`).
    pub fn flops(&self) -> f64 {
        self.batch.flops()
    }

    /// Bytes of Q and the K/V cache, as `bench_batch_attention.py` counts them (the whole
    /// cache tensor: every allocated page).
    pub fn bytes(&self) -> f64 {
        (self.batch.q.len() + self.batch.k_cache.len() + self.batch.v_cache.len()) as f64 * 2.0
    }
}

/// Largest `|a - b|`, counting equal values (and infinities) as equal.
pub fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| if x == y { 0.0 } else { (x - y).abs() }).fold(0.0, f32::max)
}

/// Run `p` under `plan` on the simulator; returns `o` (as f32) and `lse`.
pub fn execute(p: &Problem, plan: &Plan) -> (Vec<f32>, Vec<f32>) {
    execute_spec(p, plan, &p.spec(plan))
}

/// [`execute`] with the kernel built for `spec` (such as [`AttentionSpec::alternate`]).
pub fn execute_spec(p: &Problem, plan: &Plan, spec: &AttentionSpec) -> (Vec<f32>, Vec<f32>) {
    let mut args = p.args(plan);
    let k = kernel(spec);
    k.simulate(plan.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let o: Vec<f32> = args[p::O].to_vec::<F16>().into_iter().map(|x| x.to_f32()).collect();
    (o, args[p::LSE].as_f32().to_vec())
}

/// Simulate `p` under a plan for `num_sm` SMs and check it: the plan and `(max |o - ref|,
/// max |lse - ref|)`.
pub fn simulate(p: &Problem, num_sm: usize) -> (Plan, f32, f32) {
    let plan = p.plan(num_sm);
    let (o, lse) = execute(p, &plan);
    let (want_o, want_lse) = p.reference();
    (plan.clone(), max_error(&o, &want_o), max_error(&lse, &want_lse))
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

    use super::{AttentionSpec, Plan, Problem, kernel, max_error};

    type Loaded<'c> = Rc<(Kernel, CudaFunction<'c>)>;

    /// The persistent kernels, compiled and loaded when first needed.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: Toolchain,
        opts: CompileOptions,
        cache: RefCell<HashMap<AttentionSpec, Loaded<'c>>>,
        pub num_sm: usize,
        /// Build kernels with [`AttentionSpec::alternate`].
        pub alternate: bool,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: Toolchain) -> Result<Kernels<'c>, Box<dyn Error>> {
            let arch = ctx.best_target(&tc).ok_or("no target for this GPU")?;
            let opts = CompileOptions::new().gpu_arch(arch);
            let num_sm = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            Ok(Kernels { ctx, tc, opts, cache: RefCell::new(HashMap::new()), num_sm, alternate: false })
        }

        /// The kernel for `spec`, compiled on first use.
        pub fn get(&self, spec: &AttentionSpec) -> Result<Loaded<'c>, Box<dyn Error>> {
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

    /// Run `p` once on the GPU under FlashInfer's plan; if `check`, compare with the f64
    /// reference; if `time`, time the launch replayed from a CUDA graph, `iterations` times.
    /// Returns the plan, `(max |o - ref|, max |lse - ref|)` and the GPU time (us).
    #[allow(clippy::type_complexity)]
    pub fn run(
        ks: &Kernels<'_>,
        p: &Problem,
        check: bool,
        time: Option<usize>,
    ) -> Result<(Plan, f32, f32, Option<f64>), Box<dyn Error>> {
        let ctx = ks.ctx;
        let plan = p.plan(ks.num_sm);
        let loaded = ks.get(&AttentionSpec { alternate: ks.alternate, ..p.spec(&plan) })?;
        let (k, f) = (&loaded.0, &loaded.1);
        let host = p.args(&plan);
        let mut bufs = Vec::new();
        for a in &host {
            bufs.push(match a {
                KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => None,
                _ => Some(ctx.upload(a)?),
            });
        }
        let args: Vec<DeviceArg> = (0..host.len())
            .map(|i| match &bufs[i] {
                Some(b) => DeviceArg::Buffer(b),
                None => DeviceArg::Scalar(host[i].clone()),
            })
            .collect();
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
            let o: Vec<f32> = bufs[pp::O].as_ref().unwrap().to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect();
            let lse = bufs[pp::LSE].as_ref().unwrap().to_vec::<f32>()?;
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
