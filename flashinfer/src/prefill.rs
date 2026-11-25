// SPDX-License-Identifier: Apache-2.0

//! Batch prefill attention over a paged KV cache, after FlashInfer's
//! `BatchPrefillWithPagedKVCacheKernel` (`include/flashinfer/attention/prefill.cuh`, the
//! `fa2` backend) and its planner (`PrefillSplitQOKVIndptr` in `scheduler.cuh`), with
//! split-KV partial states merged by [`decode::merge_kernel`](crate::decode::merge_kernel)
//! (FlashInfer's `VariableLengthMergeStates`). This is cute-rs's own code: it follows the
//! published design (tile selection, thread geometry, permuted shared-memory layout,
//! fragment layouts, pipeline and softmax), but no FlashInfer source was translated.
//!
//! * **Plan** ([`Plan::new`], on the host, as FlashInfer's `plan()` outside CUDA-graph
//!   mode): the query tile `CTA_TILE_Q` is chosen from the average packed query length
//!   (`qo_len * group`): 128 above 64, 64 above 16, else 16. If the query tiles alone do
//!   not fill the GPU (`2 * SMs / num_kv_heads` CTAs per KV head), the smallest KV chunk,
//!   in pages and at least `128 / page_size`, is found whose `(query tile, KV chunk)`
//!   pairs still fit, and each request's KV is split into such chunks.
//! * **Prefill kernel**: grid `(tiles, num_kv_heads)`, block `(32, NUM_WARPS_Q,
//!   NUM_WARPS_KV)`. A CTA takes one tile of `CTA_TILE_Q` packed query rows (row `p` is
//!   query `p / group` of query head `kv_head * group + p % group`: GQA heads share their
//!   K and V tiles) and one KV chunk of one request, in tiles of `CTA_TILE_KV` tokens.
//!   * Q is staged once in shared memory; K and V tiles stream through one buffer each
//!     with 16-byte `cp.async` copies, the next K loading during this tile's `P V` and
//!     the next V during the next tile's `Q K^T` (two groups in flight). Token `t`'s
//!     row is gathered through the page table: `kv_indices[kv_indptr[b] + t /
//!     page_size]`, entry `t % page_size`, read one tile ahead.
//!   * Shared memory is FlashInfer's permuted layout: rows of 16 16-byte vectors,
//!     vector `j` of row `i` stored at `i * 16 + (j ^ (i % 8))`, so the eight rows an
//!     `ldmatrix` reads, and the rows `cp.async` writes, hit eight different bank groups.
//!   * Each warp owns `NUM_MMA_Q` 16-row slices of Q. `S = Q K^T` is `mma.sync
//!     m16n8k16` (f16 in, f32 accumulate) on fragments loaded with `ldmatrix.x4` (Q
//!     and K) and `ldmatrix.x4.trans` (V); `P` stays in registers, converted to f16,
//!     as the A operand of `P V` (the accumulator layout of two m16n8 tiles is the A
//!     layout of an m16k16).
//!   * The softmax is online and base 2 (`ex2.approx.ftz`), on scores scaled by
//!     `sm_scale * log2(e)`; row maxima are reduced over the four lanes of a row with
//!     butterfly shuffles, and row sums with an MMA against a ones matrix.
//!   * Tiles that cross the causal diagonal or the chunk end are masked
//!     (`kv_idx + qo_len > kv_len + q_idx`, or `kv_idx >= chunk_end`); tiles past the
//!     diagonal are never loaded.
//!   * With `NUM_WARPS_KV > 1` (the 16-row tile) the warps split each KV tile and merge
//!     their states through shared memory at the end.
//!   * Without split-KV it writes `o` (f16, staged through shared memory for 16-byte
//!     stores) and FlashInfer's base-2 `lse = m + log2(d)`; with split-KV each chunk's
//!     normalised output and lse go to f32 scratch, row `o_indptr[b] + q * num_chunks +
//!     chunk`, and the merge kernel folds them (FlashInfer's scratch is f16; f32 lets
//!     the decode port's merge kernel be reused as is).
//!
//! Tile shapes are FlashInfer's for head dim 128 and f16 on a 100 KB-per-SM GPU
//! (RTX 2000 Ada): `CTA_TILE_Q` 128: 4 x 1 warps, `NUM_MMA_Q = 2`, `NUM_MMA_KV = 2`
//! (`CTA_TILE_KV = 32`); 64: 4 x 1 warps, 1, 4 (64); 16: 1 x 4 warps, 1, 1 (64). Each
//! uses at most half the SM's shared memory, so two CTAs share an SM.
//!
//! Unlike FlashInfer, whose kernel takes them at run time, the GQA group size and the
//! page size are build-time constants of a kernel (cute-rs builds kernels at run time,
//! so this costs a build per value, not a template instance per value).
//!
//! The cache layout is NHD: `[num_pages, page_size, num_kv_heads, HEAD_DIM]`; queries
//! and outputs are ragged, `[total_q, num_qo_heads, HEAD_DIM]` with `qo_indptr`.
//!
//! **Ragged KV** (FlashInfer's `BatchPrefillWithRaggedKVCacheWrapper`, `fa2`): a kernel
//! built with `page_size` [`RAGGED`] reads K and V from contiguous `[total_kv,
//! num_kv_heads, HEAD_DIM]` tensors, request `b` owning tokens `kv_indptr[b] ..
//! kv_indptr[b + 1]` (FlashInfer's NHD ragged layout); only the K/V row address and
//! `kv_len` differ from the paged kernel, which shares every other stage. The plan is the
//! paged planner with one-token pages, as FlashInfer's ragged `plan()`.
//! [`single_prefill_with_kv_cache`] (one request, contiguous K/V) is a host wrapper over
//! the ragged kernel, with FlashInfer's `SinglePrefillWithKVCacheDispatched` chunk rule
//! ([`Plan::single`]).
//!
//! **Attention variants** ([`Variant`](crate::decode::Variant) in [`KernelSpec`],
//! build-time, FlashInfer's `DefaultAttention<use_custom_mask, use_sliding_window,
//! use_logits_soft_cap, use_alibi>`; [`Problem::with_soft_cap`], [`Problem::with_window`],
//! [`Problem::with_custom_mask`], [`Problem::with_alibi`]), each built only into its own
//! kernels, so the plain kernels are unchanged:
//! * logits soft-cap: `S = tanh(S * sm_scale / cap)` (hardware `tanh.approx.f32`), then
//!   the softmax scale `cap * log2(e)`;
//! * sliding window `window_left`: query `i` sees tokens with `kv_idx + qo_len +
//!   window_left >= kv_len + i`. As FlashInfer, a CTA starts at the first token any of its
//!   queries sees (`kv_start`), so tiles left of the window are never loaded, and only the
//!   tiles crossing the window's edge are masked; with causal, the planner counts only
//!   `window_left + CTA_TILE_Q` tokens per request;
//! * custom mask: FlashInfer's packed bit mask (little bit order, built with
//!   [`packbits`](crate::quantization)), request `b`'s bits starting at byte
//!   `mask_indptr[b]`, bit `q * kv_len + kv` (every tile is masked, as FlashInfer's
//!   `kCustom`);
//! * ALiBi: `S * sm_scale + slope[head] * (kv_idx - q_idx)`, not combined with soft-cap.
//!
//! **FP8 KV cache** ([`KvDtype`] in [`TileConfig`], build-time; [`Problem::quantized`] with
//! `k_scale`/`v_scale`): E4M3 or E5M2 K and V with Float16 queries and output, as FlashInfer's
//! `kv_data_type=torch.float8_e4m3fn` fa2 prefill. K and V tiles stay FP8 in shared memory
//! (rows of eight 16-byte vectors, the same XOR permutation) and are widened in registers
//! after `ldmatrix` with `cvt.rn.f16x2.{e4m3x2,e5m2x2}` (FlashInfer's `vec_cast`), so the
//! MMAs are the f16 ones. The tiles are half the size, so the shared-memory rule picks
//! longer KV tiles (64 / 128 / 128 tokens for `CTA_TILE_Q` 128 / 64 / 16). Where FlashInfer
//! fixes the 8-bit fragment layout with a lane shuffle per register, this kernel permutes the
//! head dimension instead (a sum, so free): a lane's `ldmatrix` of FP8 K holds elements `4 t ..
//! 4 t + 3`, which Q's shared copy (loaded once, with 4-byte `cp.async`) is stored in the
//! order of; the transposed `ldmatrix` of V gives a lane elements `2 g` and `2 g + 1`, so the
//! two n8 tiles of a slice take the even and odd elements, and a lane's accumulators hold
//! four consecutive elements, written as one 8-byte (f16) or 16-byte (f32 scratch) store.
//! `k_scale` folds into `sm_scale` on the host and `v_scale` scales the normalised output in
//! the kernel (FlashInfer applies it afterwards); paged, ragged and single all share it.
//!
//! Not yet covered: RoPE in the kernel; the HND layout, other head dims; CUDA-graph planning (fixed-size buffers); FP8 queries.
//! Every request must have at least one query and one KV token.
//!
//! `examples/prefill.rs` runs it on the simulator and, with `--features cuda` and `GPU=1`,
//! on the GPU against a reference, then benchmarks it against the same shapes as
//! `examples/prefill_flashinfer_bench.py`.

use cute::prelude::*;

pub use crate::decode::{KvDtype, Variant, alibi_slopes};

/// Head dimension (Q, K and V).
pub const HEAD_DIM: usize = 128;
/// Warps per CTA in every configuration.
const NUM_WARPS: usize = 4;
/// Elements a vector view may address (as in the decode port).
const BOUND: usize = 1 << 30;
/// The `page_size` of a ragged (contiguous K/V, `kv_indptr` in tokens) kernel or problem.
pub const RAGGED: usize = 0;

/// FlashInfer's `FA2DetermineCtaTileQ` for head dim 128 on SM80 and newer.
pub fn fa2_cta_tile_q(avg_packed_qo_len: usize) -> usize {
    if avg_packed_qo_len > 64 {
        128
    } else if avg_packed_qo_len > 16 {
        64
    } else {
        16
    }
}

/// A CTA's tile shape (FlashInfer's `KernelTraits` for one `CTA_TILE_Q`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileConfig {
    /// The K/V cache's element type (FP8 halves the K and V tiles).
    pub kv: KvDtype,
    pub cta_tile_q: usize,
    /// 16-row MMA tiles of Q per warp.
    pub num_mma_q: usize,
    /// 16-token MMA tiles of K/V per warp per iteration.
    pub num_mma_kv: usize,
    pub num_warps_q: usize,
    pub num_warps_kv: usize,
}

impl std::hash::Hash for TileConfig {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.kv as u8, self.cta_tile_q, self.num_mma_q, self.num_mma_kv, self.num_warps_q, self.num_warps_kv)
            .hash(state);
    }
}

impl TileConfig {
    /// FlashInfer's dispatch for `cta_tile_q` (128, 64 or 16), given the device's shared
    /// memory per SM and per block (opt-in): the warp layout, then the largest
    /// `NUM_MMA_KV` in {8, 4, 2, 1} whose K and V tiles fit next to the Q tile in a
    /// half-SM share (or a whole SM if two CTAs never fit), capped at `8 / NUM_MMA_Q`.
    pub fn new(cta_tile_q: usize, max_smem_per_sm: usize, max_smem_per_block: usize) -> TileConfig {
        Self::new_kv(cta_tile_q, max_smem_per_sm, max_smem_per_block, KvDtype::F16)
    }

    /// [`TileConfig::new`] for a K/V cache of type `kv`: FlashInfer's
    /// `max_num_mma_kv_smem` counts the K and V tiles in `sizeof(DTypeKV)` bytes.
    pub fn new_kv(cta_tile_q: usize, max_smem_per_sm: usize, max_smem_per_block: usize, kv: KvDtype) -> TileConfig {
        Self::new_kv_hd(cta_tile_q, max_smem_per_sm, max_smem_per_block, kv, HEAD_DIM)
    }

    /// [`TileConfig::new_kv`] for head dimension `head_dim`.
    pub(crate) fn new_kv_hd(
        cta_tile_q: usize,
        max_smem_per_sm: usize,
        max_smem_per_block: usize,
        kv: KvDtype,
        head_dim: usize,
    ) -> TileConfig {
        let (num_warps_q, num_mma_q) = match cta_tile_q {
            128 => (4, 2),
            64 => (4, 1),
            16 => (1, 1),
            _ => panic!("CTA_TILE_Q must be 128, 64 or 16, not {cta_tile_q}"),
        };
        let num_warps_kv = NUM_WARPS / num_warps_q;
        let fixed = cta_tile_q * head_dim * 2;
        let per_mma_kv = 2 * head_dim * 16 * num_warps_kv * kv.bytes();
        let ctas_per_sm = if max_smem_per_sm >= 2 * (fixed + per_mma_kv) { 2 } else { 1 };
        let budget = (max_smem_per_sm / ctas_per_sm).min(max_smem_per_block);
        let max_mma_kv = ((budget.saturating_sub(fixed)) / per_mma_kv).min(8 / num_mma_q);
        let num_mma_kv = [8, 4, 2, 1].into_iter().find(|&n| n <= max_mma_kv).expect("no KV tile fits shared memory");
        TileConfig { kv, cta_tile_q, num_mma_q, num_mma_kv, num_warps_q, num_warps_kv }
    }

    /// [`TileConfig::new`] for an SM89 (Ada) GPU: 100 KB of shared memory per SM, 99 KB
    /// per block.
    pub fn sm89(cta_tile_q: usize) -> TileConfig {
        TileConfig::new(cta_tile_q, 102400, 101376)
    }

    /// [`TileConfig::sm89`] for a K/V cache of type `kv`.
    pub fn sm89_kv(cta_tile_q: usize, kv: KvDtype) -> TileConfig {
        TileConfig::new_kv(cta_tile_q, 102400, 101376, kv)
    }

    /// 16-byte vectors per row of K or V in shared memory (a row is `head_dim` elements).
    pub(crate) fn kv_vecs_hd(&self, head_dim: usize) -> usize {
        head_dim * self.kv.bytes() / 16
    }

    pub fn cta_tile_kv(&self) -> usize {
        self.num_mma_kv * self.num_warps_kv * 16
    }

    pub fn block(&self) -> [u32; 3] {
        [32, self.num_warps_q as u32, self.num_warps_kv as u32]
    }

    /// Shared memory in f16 elements: Q, K and V tiles, or (aliasing them, as
    /// FlashInfer's union) the cross-warp merge buffers when `NUM_WARPS_KV > 1`.
    fn smem_elems(&self) -> usize {
        self.smem_elems_hd(HEAD_DIM)
    }

    /// [`TileConfig::smem_elems`] for head dimension `head_dim`.
    pub(crate) fn smem_elems_hd(&self, head_dim: usize) -> usize {
        let qkv = self.cta_tile_q * head_dim + 2 * self.cta_tile_kv() * self.kv_vecs_hd(head_dim) * 8;
        let sync = if self.num_warps_kv > 1 {
            2 * (self.num_warps_kv * self.cta_tile_q * head_dim + self.num_warps_kv * self.cta_tile_q * 2)
        } else {
            0
        };
        qkv.max(sync)
    }

    /// Shared memory per CTA, in bytes.
    pub fn smem_bytes(&self) -> usize {
        2 * self.smem_elems()
    }
}

/// What a kernel is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelSpec {
    pub tile: TileConfig,
    /// Query heads per KV head.
    pub group: usize,
    pub page_size: usize,
    pub causal: bool,
    /// The split-KV variant: each chunk's normalised state goes to f32 scratch.
    pub partition: bool,
    /// The attention variant ([`Variant::NONE`] for plain attention).
    pub variant: Variant,
}

impl KernelSpec {
    pub fn name(&self) -> String {
        let layout = if self.page_size == RAGGED { "ragged".to_string() } else { format!("paged_p{}", self.page_size) };
        format!(
            "flashinfer_batch_prefill_{layout}_q{}_g{}{}{}{}{}",
            self.tile.cta_tile_q,
            self.group,
            if self.causal { "_causal" } else { "" },
            if self.partition { "_split" } else { "" },
            self.variant.suffix(),
            match self.tile.kv {
                KvDtype::F16 => "",
                KvDtype::E4M3 => "_e4m3",
                KvDtype::E5M2 => "_e5m2",
            }
        )
    }
}

/// Parameter positions of the prefill kernel (both variants).
pub mod p {
    pub const Q: usize = 0;
    pub const K: usize = 1;
    pub const V: usize = 2;
    pub const QO_INDPTR: usize = 3;
    pub const KV_INDPTR: usize = 4;
    pub const KV_INDICES: usize = 5;
    pub const LAST_PAGE_LEN: usize = 6;
    pub const REQUEST_INDICES: usize = 7;
    pub const QO_TILE_INDICES: usize = 8;
    pub const KV_TILE_INDICES: usize = 9;
    pub const O_INDPTR: usize = 10;
    pub const O: usize = 11;
    pub const LSE: usize = 12;
    pub const TMP_O: usize = 13;
    pub const TMP_LSE: usize = 14;
    pub const MERGE_INDPTR: usize = 15;
    pub const NUM_KV_HEADS: usize = 16;
    pub const SM_SCALE_LOG2: usize = 17;
    pub const KV_CHUNK_SIZE: usize = 18;
    /// The scale before the logits transform ([`Variant::scales`]; unread by the plain
    /// kernel).
    pub const LOGITS_PRE_SCALE: usize = 19;
    /// FlashInfer's `window_left` (negative: no window; read only with a sliding window).
    pub const WINDOW_LEFT: usize = 20;
    /// ALiBi slopes `[num_qo_heads]` (read only with ALiBi).
    pub const ALIBI_SLOPES: usize = 21;
    /// The packed custom mask (bytes) and each request's first byte `[batch + 1]` (read
    /// only with a custom mask).
    pub const CUSTOM_MASK: usize = 22;
    pub const MASK_INDPTR: usize = 23;
    /// The output scale (FlashInfer's `v_scale`; read only by an FP8 kernel).
    pub const V_SCALE: usize = 24;
}

/// The element type of the K/V cache a kernel is built over, and how a pair of its bytes
/// widens (the FP8 formats' hardware `cvt.rn.f16x2.{e4m3x2,e5m2x2}`, exact).
pub trait KvElem: Numeric + Copy {
    const KV: KvDtype;
    /// Two elements widened to Float16 (FP8 only).
    fn widen2(pair: [Value<Self>; 2]) -> [Value<F16>; 2];
}

impl KvElem for F16 {
    const KV: KvDtype = KvDtype::F16;
    fn widen2(_: [Value<Self>; 2]) -> [Value<F16>; 2] {
        unreachable!("a Float16 cache is not widened")
    }
}

fn widen_fp8<F: Fp8Format>(pair: [Value<F>; 2]) -> [Value<F16>; 2] {
    let (lo, hi): (Float16, Float16) = cvt_f8x2_to_f16x2::<F, _>(pair, false);
    [lo, hi]
}

impl KvElem for F8E4M3FN {
    const KV: KvDtype = KvDtype::E4M3;
    fn widen2(pair: [Value<Self>; 2]) -> [Value<F16>; 2] {
        widen_fp8(pair)
    }
}

impl KvElem for F8E5M2 {
    const KV: KvDtype = KvDtype::E5M2;
    fn widen2(pair: [Value<Self>; 2]) -> [Value<F16>; 2] {
        widen_fp8(pair)
    }
}

/// The prefill kernel for `spec`. Parameters are in the order of [`p`]: `q [total_q,
/// num_qo_heads, HEAD_DIM]`, `k_cache`, `v_cache` (NHD pages), `qo_indptr [batch + 1]`,
/// `kv_indptr [batch + 1]` (pages), `kv_indices`, `kv_last_page_len [batch]`, the plan's
/// `request_indices`, `qo_tile_indices` and `kv_tile_indices [ctas]` and `o_indptr
/// [batch + 1]`, `o` (like `q`), `lse [total_q, num_qo_heads]`, `tmp_o [rows, num_qo_heads,
/// HEAD_DIM]`, `tmp_lse [rows, num_qo_heads]`, `merge_indptr` (read only by the merge),
/// and the scalars `num_kv_heads`, `sm_scale_log2` (`sm_scale * log2(e)`) and
/// `kv_chunk_size` (tokens). With `page_size` [`RAGGED`], `k_cache` and `v_cache` are
/// contiguous `[total_kv, num_kv_heads, HEAD_DIM]`, `kv_indptr` counts tokens, and
/// `kv_indices` and `kv_last_page_len` are not read.
pub fn kernel(spec: &KernelSpec) -> Kernel {
    match spec.tile.kv {
        KvDtype::F16 => build::<F16>(spec),
        KvDtype::E4M3 => build::<F8E4M3FN>(spec),
        KvDtype::E5M2 => build::<F8E5M2>(spec),
    }
}

fn build<KV: KvElem>(spec: &KernelSpec) -> Kernel {
    Kernel::builder(spec.name()).block_dim(spec.tile.block()).build(|| {
        let prm = Params::declare::<KV>("");
        let at = CtaCoords {
            lane: thread_idx().x,
            wq: thread_idx().y,
            wkv: thread_idx().z,
            kv_head: block_idx().y,
            cta: block_idx().x,
        };
        let smem = make_smem_tensor_aligned::<F16>(Layout::col_major(spec.tile.smem_elems()), 16);
        body::<KV>(spec, &prm, &smem, at);
    })
}

/// The prefill kernel's parameters, declared in the order of [`p`] (shared with the POD
/// kernel, which declares two sets, under name prefixes).
pub(crate) struct Params {
    pub q: Ptr<F16>,
    pub k_cache: Ptr<F16>,
    pub v_cache: Ptr<F16>,
    pub qo_indptr: Ptr<i32>,
    pub kv_indptr: Ptr<i32>,
    pub kv_indices: Ptr<i32>,
    pub kv_last_page_len: Ptr<i32>,
    pub request_indices: Ptr<i32>,
    pub qo_tile_indices: Ptr<i32>,
    pub kv_tile_indices: Ptr<i32>,
    pub o_indptr: Ptr<i32>,
    pub o: Ptr<F16>,
    pub lse: Ptr<f32>,
    pub tmp_o: Ptr<f32>,
    pub tmp_lse: Ptr<f32>,
    pub num_kv_heads: Int32,
    pub sm_scale_log2: Float32,
    pub kv_chunk_size: Int32,
    pub logits_pre_scale: Float32,
    pub window_left: Int32,
    pub alibi_slopes: Ptr<f32>,
    pub custom_mask: Ptr<u8>,
    pub mask_indptr: Ptr<i32>,
    pub v_scale: Float32,
}

impl Params {
    /// Declare the parameters of [`p`], each named `prefix` + its name; K and V are
    /// declared as `KV` and viewed as Float16 words.
    pub(crate) fn declare<KV: KvElem>(prefix: &str) -> Params {
        let n = |s: &str| format!("{prefix}{s}");
        let q = param_ptr::<F16>(&n("q"));
        let k_cache = recast_ptr::<F16, KV>(&param_ptr::<KV>(&n("k_cache")), None);
        let v_cache = recast_ptr::<F16, KV>(&param_ptr::<KV>(&n("v_cache")), None);
        let qo_indptr = param_ptr::<i32>(&n("qo_indptr"));
        let kv_indptr = param_ptr::<i32>(&n("kv_indptr"));
        let kv_indices = param_ptr::<i32>(&n("kv_indices"));
        let kv_last_page_len = param_ptr::<i32>(&n("kv_last_page_len"));
        let request_indices = param_ptr::<i32>(&n("request_indices"));
        let qo_tile_indices = param_ptr::<i32>(&n("qo_tile_indices"));
        let kv_tile_indices = param_ptr::<i32>(&n("kv_tile_indices"));
        let o_indptr = param_ptr::<i32>(&n("o_indptr"));
        let o = param_ptr::<F16>(&n("o"));
        let lse = param_ptr::<f32>(&n("lse"));
        let tmp_o = param_ptr::<f32>(&n("tmp_o"));
        let tmp_lse = param_ptr::<f32>(&n("tmp_lse"));
        let _merge_indptr = param_ptr::<i32>(&n("merge_indptr"));
        let num_kv_heads = param_scalar::<i32>(&n("num_kv_heads"));
        let sm_scale_log2 = param_scalar::<f32>(&n("sm_scale_log2"));
        let kv_chunk_size = param_scalar::<i32>(&n("kv_chunk_size"));
        let logits_pre_scale = param_scalar::<f32>(&n("logits_pre_scale"));
        let window_left = param_scalar::<i32>(&n("window_left"));
        let alibi_slopes = param_ptr::<f32>(&n("alibi_slopes"));
        let custom_mask = param_ptr::<u8>(&n("custom_mask"));
        let mask_indptr = param_ptr::<i32>(&n("mask_indptr"));
        let v_scale = param_scalar::<f32>(&n("v_scale"));
        Params {
            q,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            request_indices,
            qo_tile_indices,
            kv_tile_indices,
            o_indptr,
            o,
            lse,
            tmp_o,
            tmp_lse,
            num_kv_heads,
            sm_scale_log2,
            kv_chunk_size,
            logits_pre_scale,
            window_left,
            alibi_slopes,
            custom_mask,
            mask_indptr,
            v_scale,
        }
    }
}

/// Where a CTA's work is: the thread's lane and warp coordinates (`threadIdx` of the
/// prefill kernel's `(32, NUM_WARPS_Q, NUM_WARPS_KV)` block), the plan entry `cta` and
/// the KV head.
#[derive(Clone, Copy)]
pub(crate) struct CtaCoords {
    pub lane: Int32,
    pub wq: Int32,
    pub wkv: Int32,
    pub kv_head: Int32,
    pub cta: Int32,
}

/// The prefill kernel's work for one CTA (plan entry `at.cta`, KV head `at.kv_head`), on
/// the shared buffer `smem` of [`TileConfig::smem_bytes`]. Every thread of the CTA calls it.
pub(crate) fn body<KV: KvElem>(spec: &KernelSpec, prm: &Params, smem: &Tensor<F16>, at: CtaCoords) {
    body_hd::<KV>(spec, HEAD_DIM, prm, smem, at)
}

/// [`body`] for head dimension `head_dim` (64 or 128; 64 with a Float16 cache only), on a
/// shared buffer of [`TileConfig::smem_elems_hd`] elements (used by the block-sparse port).
pub(crate) fn body_hd<KV: KvElem>(spec: &KernelSpec, head_dim: usize, prm: &Params, smem: &Tensor<F16>, at: CtaCoords) {
    let KernelSpec { tile, group, page_size, causal, partition, variant } = *spec;
    assert!(!(variant.logits_soft_cap && variant.alibi), "soft-cap with ALiBi is not supported");
    assert!(head_dim == HEAD_DIM || (head_dim == 64 && KV::KV == KvDtype::F16), "head dim {head_dim} not supported");
    // 16-wide MMA tiles along the head dimension, and 16-byte vectors per row of Q or O.
    let (num_mma_d, row_vecs) = (head_dim / 16, head_dim / 8);
    let TileConfig { cta_tile_q, num_mma_q: nq, num_mma_kv: nkv, num_warps_q, num_warps_kv, .. } = tile;
    let cta_tile_kv = tile.cta_tile_kv();
    let total = tile.smem_elems_hd(head_dim);
    let fp8 = KV::KV != KvDtype::F16;
    // 16-byte vectors per row of K or V (8 for FP8: a row is 128 bytes).
    let kvecs = tile.kv_vecs_hd(head_dim);
    let (k_off, v_off) = ((cta_tile_q * row_vecs) as i32, (cta_tile_q * row_vecs + cta_tile_kv * kvecs) as i32);
    let ragged = page_size == RAGGED;
    let (g, ps) = (group as i32, page_size.max(1) as i32);
    // Rows of K/V a thread copies per tile (FlashInfer's `NUM_MMA_KV * 4 / NUM_WARPS_Q`).
    let kv_rows = cta_tile_kv / 16;
    {
        let Params {
            q,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            request_indices,
            qo_tile_indices,
            kv_tile_indices,
            o_indptr,
            o,
            lse,
            tmp_o,
            tmp_lse,
            num_kv_heads,
            sm_scale_log2,
            kv_chunk_size,
            logits_pre_scale,
            window_left: window_left_arg,
            alibi_slopes,
            custom_mask,
            mask_indptr,
            v_scale,
        } = prm;
        let (num_kv_heads, sm_scale_log2, kv_chunk_size) = (*num_kv_heads, *sm_scale_log2, *kv_chunk_size);
        let (logits_pre_scale, window_left_arg, v_scale) = (*logits_pre_scale, *window_left_arg, *v_scale);

        let CtaCoords { lane, wq, wkv, kv_head, cta } = at;
        let warp = wkv * num_warps_q as i32 + wq;
        let req = request_indices.load_at(cta);
        let qo_tile = qo_tile_indices.load_at(cta);
        let kv_tile = kv_tile_indices.load_at(cta);
        let num_qo_heads = num_kv_heads * g;
        let hd = head_dim as i32;

        let qo_begin = qo_indptr.load_at(req);
        let qo_len = qo_indptr.load_at(req + 1) - qo_begin;
        let page_begin = kv_indptr.load_at(req);
        let kv_len = if ragged {
            kv_indptr.load_at(req + 1) - page_begin
        } else {
            (kv_indptr.load_at(req + 1) - page_begin - 1) * ps + kv_last_page_len.load_at(req)
        };
        let ceil_div = |a: Int32, b: i32| (a + (b - 1)) / b;
        let sub_or_zero = |a: Int32, b: Int32| select(a.gt(b), a - b, Int32::from(0));
        // A sliding window (FlashInfer's: a negative `window_left` is the whole request)
        // starts the CTA at the first token any of its queries sees.
        let window_left = select(window_left_arg.ge(0), window_left_arg, kv_len);
        let kv_start = if variant.sliding_window {
            Some(sub_or_zero(kv_len + qo_tile * cta_tile_q as i32 / g, qo_len + window_left))
        } else {
            None
        };
        let (chunk_start, chunk_end) = match (partition, kv_start) {
            (true, None) => ((kv_tile * kv_chunk_size).min(kv_len), ((kv_tile + 1) * kv_chunk_size).min(kv_len)),
            (true, Some(s)) => {
                ((kv_tile * kv_chunk_size + s).min(kv_len), ((kv_tile + 1) * kv_chunk_size + s).min(kv_len))
            }
            (false, None) => (Int32::from(0), kv_len),
            (false, Some(s)) => (s, kv_len),
        };
        let chunk_size = chunk_end - chunk_start;
        // This warp's first packed query row.
        let qo_packed_base = (qo_tile * num_warps_q as i32 + wq) * (nq * 16) as i32;

        // One shared buffer, carved into Q, K and V tiles, viewed as 16-byte vectors
        // (columns of an (8, n) view, which proves each vector's alignment) and as
        // 4-byte pairs.
        let sv = make_tensor(&smem.ptr(), Layout::col_major((8, total / 8)));
        let s_pairs = make_tensor(&smem.ptr(), Layout::col_major((2, total / 2)));
        let vecs = |ptr: &Ptr<F16>| make_tensor(ptr, Layout::col_major((8, BOUND / 8)));
        let (q_g, k_g, v_g, o_g) = (vecs(q), vecs(k_cache), vecs(v_cache), vecs(o));
        let cp_async = make_copy_atom::<F16>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let cp_async4 = make_copy_atom::<F16>(CopyOp::CpAsync { bytes: 4, bypass_l1: false });
        let q_words = make_tensor(q, Layout::col_major((2, BOUND / 2)));
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
        // `ldmatrix.x4` (optionally transposed) at shared vector `svec`: eight f16 values,
        // two per matrix.
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
        // FP8 dequantisation after `ldmatrix`, as FlashInfer's `vec_cast`: the 4-byte words
        // of a fragment (two Float16 slots each, holding four FP8 bytes) split into bytes
        // and widened two at a time with the hardware `cvt` (exact in Float16).
        let fp8_words = |frag: &[Value<F16>]| -> Vec<[Value<KV>; 4]> {
            frag.chunks(2).map(|w| unpack::<KV, u32, 4>(pack::<F16, u32, 2>([w[0], w[1]]))).collect()
        };
        let widen = |a: Value<KV>, b: Value<KV>| KV::widen2([a, b]);

        // ---- Q: global -> shared (warps with wkv = 0), rows past the request zero-filled.
        // With an FP8 cache the head dimension is permuted within each 16-wide MMA slice: a
        // thread's `ldmatrix` of FP8 K gives it elements `4 t .. 4 t + 3` of a 16-byte row
        // (not f16's `2 t`, `2 t + 8`), so K's MMA k-index `2 t + e` is element `4 t + e`
        // and `2 t + 8 + e` is `4 t + 2 + e`. Q's shared copy takes the same order (shared
        // vector `2 md + h`, word `i` is slice `md`'s word `2 i + h`), which leaves Q K^T
        // unchanged, as the contraction is a sum over the head dimension.
        let q_rows_base = wq * (nq * 16) as i32;
        let load_q = || {
            for mq in 0..nq {
                for j in 0..4 {
                    let local = lane / 8 + (mq * 16 + j * 4) as i32;
                    let packed = qo_packed_base + local;
                    let (qi, r) = (packed / g, packed % g);
                    let row = (qo_begin + qi.min(qo_len - 1)) * num_qo_heads + kv_head * g + r;
                    let srow = q_rows_base + local;
                    let swz = (lane % 8) ^ (lane / 8 + (4 * (j % 2)) as i32);
                    for half in 0..(row_vecs / 8) as i32 {
                        let col = lane % 8 + 8 * half;
                        if fp8 {
                            let pf = make_fragment::<bool>(1);
                            pf.store(0, qi.lt(qo_len));
                            let (md, h) = (col / 2, col % 2);
                            for i in 0..4 {
                                let word = row * (head_dim / 2) as i32 + md * 8 + 2 * i + h;
                                let svec = srow * row_vecs as i32 + swz + 8 * half;
                                copy_if(
                                    &cp_async4,
                                    &pf,
                                    &q_words.slice((.., word)),
                                    &s_pairs.slice((.., svec * 4 + i)),
                                );
                            }
                        } else {
                            load_vec(
                                &q_g,
                                row * row_vecs as i32 + col,
                                srow * row_vecs as i32 + swz + 8 * half,
                                qi.lt(qo_len),
                            );
                        }
                    }
                }
            }
        };
        if num_warps_kv == 1 {
            load_q();
        } else {
            if_(wkv.eq(0), load_q);
        }
        cp_async_commit_group();

        // ---- K/V producer: this thread's rows `warp * 4 + lane / 8 + 16 i` of a tile,
        // vectors `lane % 8` and `lane % 8 + 8`.
        let wr_row = warp * 4 + lane / 8;
        let kv_wr_base = wr_row * kvecs as i32 + ((lane % 8) ^ (wr_row % 8));
        // The global vector of each of this thread's rows of tile `iter`, read through the
        // page table or (ragged) at the request's token offset (tokens past the chunk read
        // its last token, a valid address), and whether the row is in the chunk.
        let tile_rows = |iter: Int32| -> Vec<(Int32, Boolean)> {
            (0..kv_rows)
                .map(|i| {
                    let in_chunk = iter * cta_tile_kv as i32 + wr_row + (16 * i) as i32;
                    let t = chunk_start + in_chunk.min(chunk_size - 1);
                    let row =
                        if ragged { page_begin + t } else { kv_indices.load_at(page_begin + t / ps) * ps + t % ps };
                    let gvec = (row * num_kv_heads + kv_head) * kvecs as i32 + lane % 8;
                    (gvec, in_chunk.lt(chunk_size))
                })
                .collect()
        };
        let produce = |src: &Tensor<F16>, base: i32, rows: &[(Int32, Boolean)]| {
            for (i, (gvec, pred)) in rows.iter().enumerate() {
                for half in 0..kvecs / 8 {
                    let svec = kv_wr_base + base + (16 * kvecs * i + 8 * half) as i32;
                    load_vec(src, *gvec + (8 * half) as i32, svec, *pred);
                }
            }
        };
        let rows0 = tile_rows(Int32::from(0));
        produce(&k_g, k_off, &rows0);
        cp_async_commit_group();
        produce(&v_g, v_off, &rows0);
        cp_async_commit_group();

        // ---- Iteration counts (FlashInfer's): tiles up to the causal diagonal of this
        // query tile, and the first tile that needs masking.
        let (num_iterations, mask_iteration) = if causal {
            let last_q = ceil_div((qo_tile + 1) * cta_tile_q as i32, g);
            let first_q = qo_tile * cta_tile_q as i32 / g;
            let reach = chunk_size.min(sub_or_zero(kv_len - qo_len + last_q, chunk_start));
            let unmasked = chunk_size.min(sub_or_zero(kv_len + first_q - qo_len, chunk_start));
            (ceil_div(reach, cta_tile_kv as i32), unmasked / cta_tile_kv as i32)
        } else {
            (ceil_div(chunk_size, cta_tile_kv as i32), chunk_size / cta_tile_kv as i32)
        };
        // With a window, tiles before `window_iteration` cross the window's left edge for
        // some query of the tile.
        let window_iteration = kv_start.map(|_| {
            let last_q = ceil_div((qo_tile + 1) * cta_tile_q as i32, g);
            ceil_div(sub_or_zero(kv_len + last_q, qo_len + window_left + chunk_start), cta_tile_kv as i32)
        });
        let needs_mask = |iter: Int32| -> Boolean {
            if variant.custom_mask {
                Boolean::from(true)
            } else {
                match window_iteration {
                    Some(w) => iter.ge(mask_iteration) | iter.lt(w),
                    None => iter.ge(mask_iteration),
                }
            }
        };

        // ---- Per-thread ldmatrix addresses (vector indices): row `i` of a tile has its
        // vectors XORed with `i % 8`, which for every row these lanes read is `lane % 8`,
        // so moving `2 mma_d` vectors along a row is an XOR with `2 mma_d`.
        let q_rd = (q_rows_base + lane % 16) * row_vecs as i32 + ((lane / 16) ^ (lane % 8));
        let kv_rows_base = wkv * (nkv * 16) as i32;
        let (k_rd, v_rd) = if fp8 {
            // One x4 `ldmatrix` reads, for 16-byte slices `c` and `c + 1` of a row (16 FP8
            // elements each), the keys 0-7 and 8-15: matrix `m` is keys `8 (m % 2) ..`, slice
            // `c + m / 2`.
            let row = kv_rows_base + 8 * ((lane % 16) / 8) + lane % 8;
            let vec = (lane / 16) ^ (lane % 8);
            (k_off + row * kvecs as i32 + vec, v_off + row * kvecs as i32 + vec)
        } else {
            (
                k_off
                    + (kv_rows_base + 8 * (lane / 16) + lane % 8) * row_vecs as i32
                    + (((lane % 16) / 8) ^ (lane % 8)),
                v_off + (kv_rows_base + lane % 16) * row_vecs as i32 + ((lane / 16) ^ (lane % 8)),
            )
        };

        // The packed query rows of this thread's accumulator rows: `[mma_q][j]` is row
        // `lane / 4 + 8 j` of the warp's 16-row tile `mma_q`.
        let acc_rows: Vec<[Int32; 2]> =
            (0..nq).map(|mq| [0, 1].map(|j| qo_packed_base + (mq * 16 + 8 * j) as i32 + lane / 4)).collect();
        // The query index of each accumulator row (for the causal mask).
        let acc_q: Vec<[Int32; 2]> = acc_rows.iter().map(|r| r.map(|p| p / g)).collect();
        // ALiBi: each accumulator row's slope.
        let acc_slope: Vec<[Float32; 2]> = if variant.alibi {
            acc_rows.iter().map(|r| r.map(|p| alibi_slopes.load_at(kv_head * g + p % g))).collect()
        } else {
            vec![]
        };
        let mask_base = if variant.custom_mask { Some(mask_indptr.load_at(req)) } else { None };

        // Online-softmax state: `o[mma_q][mma_d][8]` (two m16n8 tiles: d `16 mma_d + 2
        // (lane % 4) + {0, 1}` and `+ 8`), `m` (unscaled scores) and `d` per row.
        let o_state = Var::new(vec![zero; nq * num_mma_d * 8]);
        let m_state = Var::new(vec![Float32::from(f32::NEG_INFINITY); nq * 2]);
        let d_state = Var::new(vec![Float32::from(1.0f32); nq * 2]);
        let ones = vec![Value::<F16>::from(F16::from_f32(1.0)); 4];

        for_range(0, num_iterations, 1, |iter| {
            // Page-table reads for the next tile, first used after this tile's Q K^T.
            let next_rows = tile_rows(iter + 1);
            cp_async_wait_group(1);
            sync_threads();

            // S = Q K^T: s[mma_q][mma_kv][8], an m16n16 tile per (mma_q, mma_kv).
            let mut s = vec![zero; nq * nkv * 8];
            if fp8 {
                // One x4 ldmatrix per (16 keys, two head-dimension slices); its words are the
                // B fragments of keys 0-7 and 8-15 for slices `md` and `md + 1`.
                for p in 0..num_mma_d / 2 {
                    let a: Vec<[Vec<Value<F16>>; 2]> = (0..nq)
                        .map(|mq| {
                            [0, 1]
                                .map(|h| ldmatrix(false, (q_rd ^ (4 * p + 2 * h) as i32) + (mq * 16 * row_vecs) as i32))
                        })
                        .collect();
                    for mk in 0..nkv {
                        let words = fp8_words(&ldmatrix(false, (k_rd ^ (2 * p) as i32) + (mk * 16 * kvecs) as i32));
                        for h in 0..2 {
                            for n in 0..2 {
                                let w = words[2 * h + n];
                                let [r0, r1] = [widen(w[0], w[1]), widen(w[2], w[3])];
                                let b = [r0[0], r0[1], r1[0], r1[1]];
                                for (mq, a) in a.iter().enumerate() {
                                    let at = (mq * nkv + mk) * 8 + 4 * n;
                                    let r = mma16(&a[h], &b, &s[at..at + 4]);
                                    s[at..at + 4].copy_from_slice(&r);
                                }
                            }
                        }
                    }
                }
            } else {
                for md in 0..num_mma_d {
                    let x = (2 * md) as i32;
                    let a: Vec<Vec<Value<F16>>> =
                        (0..nq).map(|mq| ldmatrix(false, (q_rd ^ x) + (mq * 16 * row_vecs) as i32)).collect();
                    for mk in 0..nkv {
                        let b = ldmatrix(false, (k_rd ^ x) + (mk * 16 * row_vecs) as i32);
                        for (mq, a) in a.iter().enumerate() {
                            let at = (mq * nkv + mk) * 8;
                            for n in 0..2 {
                                let c = s[at + 4 * n..at + 4 * n + 4].to_vec();
                                let r = mma16(a, &b[4 * n..4 * n + 4], &c);
                                s[at + 4 * n..at + 4 * n + 4].copy_from_slice(&r);
                            }
                        }
                    }
                }
            }

            let kv_base = chunk_start + iter * cta_tile_kv as i32 + kv_rows_base;
            // The token of each score: register `reg` of tile `mk`.
            let kv_of = |mk: usize, reg: usize| kv_base + (mk * 16 + 8 * (reg / 4) + reg % 2) as i32 + 2 * (lane % 4);
            // FlashInfer's `LogitsTransform`: soft-cap or ALiBi.
            if variant.logits_soft_cap {
                for x in s.iter_mut() {
                    *x = cute::math::tanh(*x * logits_pre_scale, true);
                }
            } else if variant.alibi {
                for mq in 0..nq {
                    for mk in 0..nkv {
                        for reg in 0..8 {
                            let j = (reg % 4) / 2;
                            let rel = (kv_of(mk, reg) - acc_q[mq][j]).to::<f32>();
                            let at = (mq * nkv + mk) * 8 + reg;
                            s[at] = s[at] * logits_pre_scale + acc_slope[mq][j] * rel;
                        }
                    }
                }
            }

            // Mask the tiles that cross the diagonal, the window's edge or the chunk end
            // (every tile with a custom mask).
            let s = if_else_yield(
                needs_mask(iter),
                || {
                    let mut s = s.clone();
                    #[allow(clippy::needless_range_loop)] // indexes several arrays in step
                    for mq in 0..nq {
                        for mk in 0..nkv {
                            for reg in 0..8 {
                                let kv_idx = kv_of(mk, reg);
                                let q_idx = acc_q[mq][(reg % 4) / 2];
                                let mut masked = kv_idx.ge(chunk_end);
                                if causal {
                                    masked = masked | (kv_idx + qo_len).gt(kv_len + q_idx);
                                }
                                if variant.sliding_window {
                                    masked = masked | (kv_idx + qo_len + window_left).lt(kv_len + q_idx);
                                }
                                if let Some(mb) = mask_base {
                                    // Rows past the request read a valid byte (they are never written).
                                    let off = q_idx.min(qo_len - 1) * kv_len + kv_idx.min(kv_len - 1);
                                    let byte = custom_mask.load_at(mb + off / 8).to::<i32>();
                                    masked = masked | ((byte >> (off % 8)) & 1).eq(0);
                                }
                                let at = (mq * nkv + mk) * 8 + reg;
                                s[at] = select(masked, Float32::from(f32::NEG_INFINITY), s[at]);
                            }
                        }
                    }
                    s
                },
                || s.clone(),
            );

            // Online softmax: new row maxima (over the quad of lanes sharing a row), the
            // rescale of o and d, and P = exp2(s * scale - m * scale).
            let (mut o_v, mut m_v, mut d_v) = (o_state.get(), m_state.get(), d_state.get());
            let mut p = s;
            for mq in 0..nq {
                for j in 0..2 {
                    let m_prev = m_v[mq * 2 + j];
                    let mut mx = m_prev;
                    for mk in 0..nkv {
                        for e in [0, 1, 4, 5] {
                            mx = fmax(mx, p[(mq * nkv + mk) * 8 + 2 * j + e]);
                        }
                    }
                    mx = fmax(mx, shuffle_sync_bfly(mx, 2));
                    mx = fmax(mx, shuffle_sync_bfly(mx, 1));
                    let m_scaled = fmax(mx * sm_scale_log2, Float32::from(-f32::MAX));
                    let o_scale = select(
                        m_prev.eq(Float32::from(f32::NEG_INFINITY)),
                        zero,
                        cute::math::exp2((m_prev - mx) * sm_scale_log2, true),
                    );
                    d_v[mq * 2 + j] = d_v[mq * 2 + j] * o_scale;
                    for md in 0..num_mma_d {
                        for e in [0, 1, 4, 5] {
                            let at = (mq * num_mma_d + md) * 8 + 2 * j + e;
                            o_v[at] = o_v[at] * o_scale;
                        }
                    }
                    for mk in 0..nkv {
                        for e in [0, 1, 4, 5] {
                            let at = (mq * nkv + mk) * 8 + 2 * j + e;
                            p[at] = cute::math::exp2(p[at] * sm_scale_log2 - m_scaled, true);
                        }
                    }
                    m_v[mq * 2 + j] = mx;
                }
            }
            m_state.set(m_v);
            sync_threads();

            // The next K tile loads while this tile's P V runs.
            produce(&k_g, k_off, &next_rows);
            cp_async_commit_group();
            cp_async_wait_group(1);
            sync_threads();

            // P in f16 (the A operand), row sums by an MMA against ones, then O += P V.
            let p16: Vec<Value<F16>> = p.iter().map(|x| x.to::<F16>()).collect();
            for mq in 0..nq {
                for mk in 0..nkv {
                    let at = (mq * nkv + mk) * 8;
                    let r = mma16(&p16[at..at + 8], &ones, &[d_v[mq * 2], zero, d_v[mq * 2 + 1], zero]);
                    d_v[mq * 2] = r[0];
                    d_v[mq * 2 + 1] = r[2];
                }
            }
            for mk in 0..nkv {
                if fp8 {
                    // Transposed x4 ldmatrix over FP8: each word holds keys `2 t` and `2 t + 1`
                    // of head-dimension elements `2 g` and `2 g + 1`, so the two m16n8 tiles of
                    // a 16-wide slice take the even and the odd element: tile `n`'s column `g`
                    // is element `2 g + n`, and an accumulator's pair `(2 t, 2 t + 1)` of tile `n` is
                    // elements `4 t + n` and `4 t + 2 + n`.
                    for p in 0..num_mma_d / 2 {
                        let words = fp8_words(&ldmatrix(true, (v_rd ^ (2 * p) as i32) + (mk * 16 * kvecs) as i32));
                        for h in 0..2 {
                            let md = 2 * p + h;
                            let (wa, wb) = (words[2 * h], words[2 * h + 1]);
                            for n in 0..2 {
                                let (ra, rb) = (widen(wa[n], wa[2 + n]), widen(wb[n], wb[2 + n]));
                                let b = [ra[0], ra[1], rb[0], rb[1]];
                                for mq in 0..nq {
                                    let a = &p16[(mq * nkv + mk) * 8..(mq * nkv + mk) * 8 + 8];
                                    let at = (mq * num_mma_d + md) * 8 + 4 * n;
                                    let r = mma16(a, &b, &o_v[at..at + 4]);
                                    o_v[at..at + 4].copy_from_slice(&r);
                                }
                            }
                        }
                    }
                } else {
                    for md in 0..num_mma_d {
                        let b = ldmatrix(true, (v_rd ^ (2 * md) as i32) + (mk * 16 * row_vecs) as i32);
                        for mq in 0..nq {
                            let a = &p16[(mq * nkv + mk) * 8..(mq * nkv + mk) * 8 + 8];
                            for n in 0..2 {
                                let at = (mq * num_mma_d + md) * 8 + 4 * n;
                                let r = mma16(a, &b[4 * n..4 * n + 4], &o_v[at..at + 4]);
                                o_v[at..at + 4].copy_from_slice(&r);
                            }
                        }
                    }
                }
            }
            o_state.set(o_v);
            d_state.set(d_v);
            sync_threads();

            // The next V tile loads during the next tile's Q K^T.
            produce(&v_g, v_off, &next_rows);
            cp_async_commit_group();
        });
        cp_async_wait_group(0);
        sync_threads();

        let (mut o_v, mut m_v, mut d_v) = (o_state.get(), m_state.get(), d_state.get());
        // Scaled maxima (FlashInfer's `finalize_m`).
        for m in m_v.iter_mut() {
            *m = select(m.eq(Float32::from(f32::NEG_INFINITY)), *m, *m * sm_scale_log2);
        }

        // ---- Merge the NUM_WARPS_KV warps' states of each row through shared memory
        // (aliasing the Q/K/V tiles, which are no longer read).
        if num_warps_kv > 1 {
            let fptr = recast_ptr::<f32, F16>(&smem.ptr(), None);
            let o_elems = num_warps_kv * cta_tile_q * head_dim;
            let so = make_tensor(&fptr, Layout::col_major((8, o_elems / 8)));
            let smd = make_tensor(&fptr.offset(o_elems as i32), Layout::col_major((2, num_warps_kv * cta_tile_q)));
            // o: [warp][mma_q][mma_d][lane][8]; m, d: [warp][mma_q][j][row].
            for mq in 0..nq {
                for md in 0..num_mma_d {
                    let col = ((warp * nq as i32 + mq as i32) * num_mma_d as i32 + md as i32) * 32 + lane;
                    let at = (mq * num_mma_d + md) * 8;
                    so.slice((.., col)).store_values(&o_v[at..at + 8]);
                }
            }
            if_((lane % 4).eq(0), || {
                for mq in 0..nq {
                    for j in 0..2 {
                        let col = ((warp * nq as i32 + mq as i32) * 2 + j as i32) * 8 + lane / 4;
                        smd.slice((.., col)).store_values(&[m_v[mq * 2 + j], d_v[mq * 2 + j]]);
                    }
                }
            });
            sync_threads();
            let neg_inf = Float32::from(f32::NEG_INFINITY);
            for mq in 0..nq {
                let mut scales = [vec![], vec![]];
                for j in 0..2 {
                    let mds: Vec<(Float32, Float32)> = (0..num_warps_kv)
                        .map(|i| {
                            let w = i as i32 * num_warps_q as i32 + wq;
                            let v =
                                smd.slice((.., ((w * nq as i32 + mq as i32) * 2 + j as i32) * 8 + lane / 4)).values();
                            (v[0], v[1])
                        })
                        .collect();
                    let m_new = mds.iter().fold(neg_inf, |a, (mi, _)| fmax(a, *mi));
                    // An all-masked row has m_new = -inf: its scales are 0, not NaN.
                    let scale = |mi: Float32| select(m_new.eq(neg_inf), zero, cute::math::exp2(mi - m_new, true));
                    let mut d_new = zero;
                    for (mi, di) in &mds {
                        let sc = scale(*mi);
                        d_new = d_new + *di * sc;
                        scales[j].push(sc);
                    }
                    m_v[mq * 2 + j] = m_new;
                    d_v[mq * 2 + j] = d_new;
                }
                for md in 0..num_mma_d {
                    let at = (mq * num_mma_d + md) * 8;
                    let mut acc = vec![zero; 8];
                    #[allow(clippy::needless_range_loop)] // indexes several arrays in step
                    for i in 0..num_warps_kv {
                        let w = i as i32 * num_warps_q as i32 + wq;
                        let col = ((w * nq as i32 + mq as i32) * num_mma_d as i32 + md as i32) * 32 + lane;
                        let oi = so.slice((.., col)).values();
                        for (reg, a) in acc.iter_mut().enumerate() {
                            *a = *a + oi[reg] * scales[(reg % 4) / 2][i];
                        }
                    }
                    o_v[at..at + 8].copy_from_slice(&acc);
                }
            }
            // The output is staged where the merge buffers were.
            sync_threads();
        }

        // ---- Normalise and write.
        let inv_d: Vec<Float32> =
            d_v.iter().map(|d| if fp8 { v_scale / *d } else { Float32::from(1.0f32) / *d }).collect();
        for mq in 0..nq {
            for md in 0..num_mma_d {
                for reg in 0..8 {
                    let at = (mq * num_mma_d + md) * 8 + reg;
                    o_v[at] = o_v[at] * inv_d[mq * 2 + (reg % 4) / 2];
                }
            }
        }
        let lse_v: Vec<Float32> = (0..nq * 2).map(|i| cute::math::log2(d_v[i], true) + m_v[i]).collect();
        let write = || {
            if partition {
                // Normalised f32 partial states, straight from the accumulators.
                // (With a causal window, FlashInfer's `min(kv_len, window_left + CTA_TILE_Q)`
                // tokens, as the planner counts them.)
                let kv_span =
                    if variant.sliding_window && causal { kv_len.min(window_left + cta_tile_q as i32) } else { kv_len };
                let num_chunks = (kv_span + kv_chunk_size - 1) / kv_chunk_size;
                let tmp_pairs = make_tensor(tmp_o, Layout::col_major((2, BOUND / 2)));
                let tmp_quads = make_tensor(tmp_o, Layout::col_major((4, BOUND / 4)));
                let row_base = o_indptr.load_at(req);
                for mq in 0..nq {
                    for j in 0..2 {
                        let packed = acc_rows[mq][j];
                        let (qi, r) = (packed / g, packed % g);
                        let row = row_base + qi * num_chunks + kv_tile;
                        let head = kv_head * g + r;
                        if_(qi.lt(qo_len), || {
                            let elem = (row * num_qo_heads + head) * hd + 2 * (lane % 4);
                            for md in 0..num_mma_d {
                                if fp8 {
                                    // Elements `4 t ..= 4 t + 3` of slice `md`: tile 0's and tile 1's
                                    // accumulators interleaved.
                                    let b = (mq * num_mma_d + md) * 8 + 2 * j;
                                    let quad = (row * num_qo_heads + head) * (hd / 4) + (4 * md) as i32 + lane % 4;
                                    tmp_quads.slice((.., quad)).store_values(&[
                                        o_v[b],
                                        o_v[b + 4],
                                        o_v[b + 1],
                                        o_v[b + 5],
                                    ]);
                                    continue;
                                }
                                for n in 0..2 {
                                    let at = (mq * num_mma_d + md) * 8 + 4 * n + 2 * j;
                                    let pair = (elem + (16 * md + 8 * n) as i32) / 2;
                                    tmp_pairs.slice((.., pair)).store_values(&o_v[at..at + 2]);
                                }
                            }
                            if_((lane % 4).eq(0), || {
                                tmp_lse.store_at(row * num_qo_heads + head, lse_v[mq * 2 + j]);
                            });
                        });
                    }
                }
            } else {
                // f16 pairs into this warp's rows of the Q tile, then 16-byte rows out.
                let s_quads = make_tensor(&smem.ptr(), Layout::col_major((4, total / 4)));
                for mq in 0..nq {
                    for md in 0..num_mma_d {
                        let row = q_rows_base + (mq * 16) as i32 + lane / 4;
                        if fp8 {
                            // Elements `4 t ..= 4 t + 3` of slice `md`, one 8-byte store per row.
                            let vec = row * row_vecs as i32 + (((2 * md) as i32 + (lane % 4) / 2) ^ (lane / 4));
                            for j in 0..2 {
                                let b = (mq * num_mma_d + md) * 8 + 2 * j;
                                let quad = (vec + (8 * row_vecs * j) as i32) * 2 + lane % 2;
                                let vals: Vec<Value<F16>> =
                                    [b, b + 4, b + 1, b + 5].iter().map(|&i| o_v[i].to::<F16>()).collect();
                                s_quads.slice((.., quad)).store_values(&vals);
                            }
                            continue;
                        }
                        let vec0 = row * row_vecs as i32 + ((lane / 4) ^ (2 * md) as i32);
                        for (n, vec) in [vec0, vec0 ^ 1].into_iter().enumerate() {
                            for j in 0..2 {
                                let at = (mq * num_mma_d + md) * 8 + 4 * n + 2 * j;
                                let pair = (vec + (8 * row_vecs * j) as i32) * 4 + lane % 4;
                                let vals: Vec<Value<F16>> = o_v[at..at + 2].iter().map(|x| x.to::<F16>()).collect();
                                s_pairs.slice((.., pair)).store_values(&vals);
                            }
                        }
                    }
                }
                sync_warp();
                for mq in 0..nq {
                    for j in 0..4 {
                        let local = lane / 8 + (mq * 16 + j * 4) as i32;
                        let packed = qo_packed_base + local;
                        let (qi, r) = (packed / g, packed % g);
                        let grow = (qo_begin + qi) * num_qo_heads + kv_head * g + r;
                        let srow = q_rows_base + local;
                        let swz = (lane % 8) ^ (lane / 8 + (4 * (j % 2)) as i32);
                        if_(qi.lt(qo_len), || {
                            for half in 0..(row_vecs / 8) as i32 {
                                let svec = srow * row_vecs as i32 + swz + 8 * half;
                                let vals = sv.slice((.., svec)).values();
                                let gvec = grow * row_vecs as i32 + lane % 8 + 8 * half;
                                o_g.slice((.., gvec)).store_values(&vals);
                            }
                        });
                    }
                }
                for mq in 0..nq {
                    for j in 0..2 {
                        let packed = acc_rows[mq][j];
                        let (qi, r) = (packed / g, packed % g);
                        if_(qi.lt(qo_len) & (lane % 4).eq(0), || {
                            lse.store_at((qo_begin + qi) * num_qo_heads + kv_head * g + r, lse_v[mq * 2 + j]);
                        });
                    }
                }
            }
        };
        if num_warps_kv == 1 {
            write();
        } else {
            if_(wkv.eq(0), write);
        }
    }
}

/// FlashInfer's prefill plan: which query tile and KV chunk of which request each CTA
/// takes.
#[derive(Clone, Debug)]
pub struct Plan {
    pub tile: TileConfig,
    pub split: bool,
    /// The KV chunk, in tokens (a whole number of pages).
    pub kv_chunk_size: usize,
    pub request_indices: Vec<i32>,
    pub qo_tile_indices: Vec<i32>,
    pub kv_tile_indices: Vec<i32>,
    /// Query token `i`'s partial states are rows `merge_indptr[i] .. merge_indptr[i + 1]`.
    pub merge_indptr: Vec<i32>,
    /// Request `b`'s partial states start at row `o_indptr[b]`.
    pub o_indptr: Vec<i32>,
}

impl Plan {
    /// Plan for requests of `qo_lens` queries and `kv_pages` pages, when `max_grid` CTAs
    /// run at once (FlashInfer assumes 2 per SM): as FlashInfer outside CUDA-graph mode.
    pub fn new(
        qo_lens: &[usize],
        kv_pages: &[usize],
        page_size: usize,
        group: usize,
        num_kv_heads: usize,
        max_grid: usize,
    ) -> Plan {
        Self::with_min_chunk(qo_lens, kv_pages, page_size, group, num_kv_heads, max_grid, (128 / page_size).max(1))
    }

    /// [`Plan::new`] with another smallest chunk, in pages (tests use 1 to split short
    /// requests).
    pub fn with_min_chunk(
        qo_lens: &[usize],
        kv_pages: &[usize],
        page_size: usize,
        group: usize,
        num_kv_heads: usize,
        max_grid: usize,
        min_chunk_pages: usize,
    ) -> Plan {
        let batch = qo_lens.len();
        let packed: Vec<usize> = qo_lens.iter().map(|l| l * group).collect();
        let cta_tile_q = fa2_cta_tile_q(packed.iter().sum::<usize>() / batch);
        let tile = TileConfig::sm89(cta_tile_q);
        let max_batch_if_split = max_grid / num_kv_heads;
        let max_kv = kv_pages.iter().copied().max().unwrap_or(1).max(1);
        let ctas = |c: usize| {
            packed.iter().zip(kv_pages).map(|(q, k)| q.div_ceil(cta_tile_q) * k.max(&1).div_ceil(c)).sum::<usize>()
        };
        let (mut low, mut high) = (min_chunk_pages.max(1), max_kv);
        while low < high {
            let mid = (low + high) / 2;
            if ctas(mid) > max_batch_if_split {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        let split = low < max_kv;
        let chunk_pages = low;
        let (mut request_indices, mut qo_tile_indices, mut kv_tile_indices) = (vec![], vec![], vec![]);
        let (mut merge_indptr, mut o_indptr) = (vec![0i32], vec![0i32]);
        for b in 0..batch {
            let num_chunks = kv_pages[b].max(1).div_ceil(chunk_pages);
            for qt in 0..packed[b].div_ceil(cta_tile_q) {
                for kt in 0..num_chunks {
                    request_indices.push(b as i32);
                    qo_tile_indices.push(qt as i32);
                    kv_tile_indices.push(kt as i32);
                }
            }
            for _ in 0..qo_lens[b] {
                merge_indptr.push(merge_indptr.last().unwrap() + num_chunks as i32);
            }
            o_indptr.push(o_indptr[b] + (qo_lens[b] * num_chunks) as i32);
        }
        Plan {
            tile,
            split,
            kv_chunk_size: chunk_pages * page_size,
            request_indices,
            qo_tile_indices,
            kv_tile_indices,
            merge_indptr,
            o_indptr,
        }
    }

    /// The plan of FlashInfer's single-request prefill (`SinglePrefillWithKVCacheDispatched`):
    /// the query tile from `qo_len * group`; if the query tiles leave room for `max_chunks =
    /// max_grid / (num_kv_heads * tiles)` chunks, KV chunks of `max(ceil(kv_len /
    /// max_chunks), 256)` tokens, split when there is more than one.
    pub fn single(qo_len: usize, kv_len: usize, group: usize, num_kv_heads: usize, max_grid: usize) -> Plan {
        let cta_tile_q = fa2_cta_tile_q(qo_len * group);
        let q_tiles = (qo_len * group).div_ceil(cta_tile_q);
        let max_chunks = max_grid / (num_kv_heads * q_tiles);
        let chunk = if max_chunks > 0 { kv_len.div_ceil(max_chunks).max(256) } else { kv_len };
        let num_chunks = kv_len.div_ceil(chunk);
        let split = num_chunks > 1;
        let (chunk, num_chunks) = if split { (chunk, num_chunks) } else { (kv_len, 1) };
        let (mut request_indices, mut qo_tile_indices, mut kv_tile_indices) = (vec![], vec![], vec![]);
        // Last query tiles first: under a causal mask they do the most work, and starting
        // them first shortens the tail of the last wave.
        for qt in (0..q_tiles).rev() {
            for kt in 0..num_chunks {
                request_indices.push(0);
                qo_tile_indices.push(qt as i32);
                kv_tile_indices.push(kt as i32);
            }
        }
        Plan {
            tile: TileConfig::sm89(cta_tile_q),
            split,
            kv_chunk_size: chunk,
            request_indices,
            qo_tile_indices,
            kv_tile_indices,
            merge_indptr: (0..=qo_len).map(|i| (i * num_chunks) as i32).collect(),
            o_indptr: vec![0, (qo_len * num_chunks) as i32],
        }
    }

    /// This plan with the tile shape of a K/V cache of type `kv` (the planner's chunks
    /// do not depend on it, as FlashInfer's).
    pub fn for_kv(mut self, kv: KvDtype) -> Plan {
        self.tile = TileConfig::sm89_kv(self.tile.cta_tile_q, kv);
        self
    }

    pub fn num_ctas(&self) -> usize {
        self.request_indices.len()
    }

    /// Rows of split-KV scratch (at least one, so buffers are never empty).
    pub fn tmp_rows(&self) -> usize {
        if self.split { *self.o_indptr.last().unwrap() as usize } else { 1 }
    }
}

/// A test problem: per-request query and KV lengths, a page size and a shuffled page
/// table; or, with `page_size` [`RAGGED`], contiguous K/V with `kv_indptr` in tokens.
pub struct Problem {
    pub qo_lens: Vec<usize>,
    pub kv_lens: Vec<usize>,
    pub page_size: usize,
    pub num_kv_heads: usize,
    pub group: usize,
    pub causal: bool,
    pub q: Vec<F16>,
    pub k_cache: Vec<F16>,
    pub v_cache: Vec<F16>,
    pub qo_indptr: Vec<i32>,
    pub kv_indptr: Vec<i32>,
    pub kv_indices: Vec<i32>,
    pub kv_last_page_len: Vec<i32>,
    pub sm_scale: f32,
    /// The attention variant the kernel is built for, and its run-time values: FlashInfer's
    /// `logits_soft_cap`, `window_left` (negative: none), ALiBi slopes per query head, and
    /// the custom mask (request `b`'s `qo_len * kv_len` bits from `qk_indptr[b]`, packed
    /// little-endian per request from byte `mask_indptr[b]`).
    pub variant: Variant,
    pub logits_soft_cap: f32,
    pub window_left: i32,
    pub alibi_slopes: Vec<f32>,
    pub mask_bits: Vec<bool>,
    pub qk_indptr: Vec<i32>,
    pub packed_mask: Vec<u8>,
    pub mask_indptr: Vec<i32>,
    /// The cache's element type. For FP8, `k_cache`/`v_cache` hold the dequantised values
    /// (exact in Float16, before `k_scale`/`v_scale`), and `k_fp8`/`v_fp8` the stored bytes.
    pub kv_dtype: KvDtype,
    pub k_fp8: Vec<u8>,
    pub v_fp8: Vec<u8>,
    /// FlashInfer's `k_scale`, `v_scale`: K is `k_scale * k_cache`, V `v_scale * v_cache`.
    pub k_scale: f32,
    pub v_scale: f32,
}

impl Problem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qo_lens: &[usize],
        kv_lens: &[usize],
        page_size: usize,
        num_kv_heads: usize,
        group: usize,
        causal: bool,
        seed: u64,
    ) -> Problem {
        assert_eq!(qo_lens.len(), kv_lens.len());
        assert!(qo_lens.iter().chain(kv_lens).all(|&l| l > 0), "every request needs queries and KV tokens");
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let pages_per: Vec<usize> = kv_lens.iter().map(|l| l.div_ceil(page_size)).collect();
        let num_pages: usize = pages_per.iter().sum::<usize>() + 2; // two unused pages
        let mut ids: Vec<i32> = (0..num_pages as i32).collect();
        for i in (1..ids.len()).rev() {
            let j = ((rnd() + 1.0) * 0.5 * (i + 1) as f32) as usize % (i + 1);
            ids.swap(i, j);
        }
        let (mut kv_indptr, mut kv_indices, mut next) = (vec![0i32], vec![], 0);
        for &p in &pages_per {
            kv_indices.extend_from_slice(&ids[next..next + p]);
            next += p;
            kv_indptr.push(next as i32);
        }
        let mut qo_indptr = vec![0i32];
        for &l in qo_lens {
            qo_indptr.push(qo_indptr.last().unwrap() + l as i32);
        }
        let kv_last_page_len = kv_lens.iter().map(|&l| (l - (l.div_ceil(page_size) - 1) * page_size) as i32).collect();
        let total_q: usize = qo_lens.iter().sum();
        let cache = num_pages * page_size * num_kv_heads * HEAD_DIM;
        // Large inputs are filled from a cheap repeating pattern (the f64 reference is
        // only run on small ones).
        let mut fill = |n: usize| -> Vec<F16> {
            if n <= 1 << 22 {
                (0..n).map(|_| F16::from_f32(rnd())).collect()
            } else {
                let block: Vec<F16> = (0..65537).map(|_| F16::from_f32(rnd())).collect();
                (0..n).map(|i| block[i % block.len()]).collect()
            }
        };
        Problem {
            qo_lens: qo_lens.to_vec(),
            kv_lens: kv_lens.to_vec(),
            page_size,
            num_kv_heads,
            group,
            causal,
            q: fill(total_q * num_kv_heads * group * HEAD_DIM),
            k_cache: fill(cache),
            v_cache: fill(cache),
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            ..Problem::no_variant()
        }
    }

    /// A ragged problem (FlashInfer's `BatchPrefillWithRaggedKVCacheWrapper` inputs) with
    /// random contents: `k`, `v` contiguous `[total_kv, num_kv_heads, HEAD_DIM]`.
    pub fn new_ragged(
        qo_lens: &[usize],
        kv_lens: &[usize],
        num_kv_heads: usize,
        group: usize,
        causal: bool,
        seed: u64,
    ) -> Problem {
        // A one-token-page problem has the ragged layout up to its page table, which the
        // ragged problem drops.
        let p = Problem::new(qo_lens, kv_lens, 1, num_kv_heads, group, causal, seed);
        let row = num_kv_heads * HEAD_DIM;
        let gather = |cache: &[F16]| -> Vec<F16> {
            p.kv_indices
                .iter()
                .flat_map(|&pg| cache[pg as usize * row..(pg as usize + 1) * row].iter().copied())
                .collect()
        };
        let (k, v) = (gather(&p.k_cache), gather(&p.v_cache));
        Problem::from_ragged(qo_lens, kv_lens, num_kv_heads, group, causal, p.q, k, v)
    }

    /// A ragged problem on given tensors: `q [sum(qo_lens), num_kv_heads * group,
    /// HEAD_DIM]`, `k`, `v [sum(kv_lens), num_kv_heads, HEAD_DIM]`, scale `1/sqrt(HEAD_DIM)`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_ragged(
        qo_lens: &[usize],
        kv_lens: &[usize],
        num_kv_heads: usize,
        group: usize,
        causal: bool,
        q: Vec<F16>,
        k: Vec<F16>,
        v: Vec<F16>,
    ) -> Problem {
        assert_eq!(qo_lens.len(), kv_lens.len());
        assert!(qo_lens.iter().chain(kv_lens).all(|&l| l > 0), "every request needs queries and KV tokens");
        let indptr = |lens: &[usize]| -> Vec<i32> {
            std::iter::once(0)
                .chain(lens.iter().scan(0i32, |a, &l| {
                    *a += l as i32;
                    Some(*a)
                }))
                .collect()
        };
        let (qo_indptr, kv_indptr) = (indptr(qo_lens), indptr(kv_lens));
        assert_eq!(q.len(), *qo_indptr.last().unwrap() as usize * num_kv_heads * group * HEAD_DIM);
        assert_eq!(k.len(), *kv_indptr.last().unwrap() as usize * num_kv_heads * HEAD_DIM);
        assert_eq!(v.len(), k.len());
        Problem {
            qo_lens: qo_lens.to_vec(),
            kv_lens: kv_lens.to_vec(),
            page_size: RAGGED,
            num_kv_heads,
            group,
            causal,
            q,
            k_cache: k,
            v_cache: v,
            qo_indptr,
            kv_indptr,
            kv_indices: vec![0],
            kv_last_page_len: vec![1; qo_lens.len()],
            sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            ..Problem::no_variant()
        }
    }

    /// An empty problem without a variant (the defaults of the variant fields).
    fn no_variant() -> Problem {
        Problem {
            qo_lens: vec![],
            kv_lens: vec![],
            page_size: 1,
            num_kv_heads: 1,
            group: 1,
            causal: false,
            q: vec![],
            k_cache: vec![],
            v_cache: vec![],
            qo_indptr: vec![0],
            kv_indptr: vec![0],
            kv_indices: vec![],
            kv_last_page_len: vec![],
            sm_scale: 1.0,
            variant: Variant::NONE,
            logits_soft_cap: 0.0,
            window_left: -1,
            alibi_slopes: vec![],
            mask_bits: vec![],
            qk_indptr: vec![0],
            packed_mask: vec![],
            mask_indptr: vec![0],
            kv_dtype: KvDtype::F16,
            k_fp8: vec![],
            v_fp8: vec![],
            k_scale: 1.0,
            v_scale: 1.0,
        }
    }

    /// This problem with its K/V cache quantised to `kv` with scales `k_scale`, `v_scale`
    /// (each element `x` stored as `x / scale` rounded to `kv`), as a caller quantises a
    /// cache for FlashInfer's FP8 prefill (`k_scale`, `v_scale` arguments). The reference
    /// then uses the dequantised cache.
    pub fn quantized(mut self, kv: KvDtype, k_scale: f32, v_scale: f32) -> Problem {
        assert_eq!(self.kv_dtype, KvDtype::F16, "already quantised");
        if kv == KvDtype::F16 {
            return self;
        }
        let quant = |c: &[F16], scale: f32| -> (Vec<F16>, Vec<u8>) {
            c.iter().map(|x| kv.quantize(x.to_f32() / scale)).unzip()
        };
        (self.k_cache, self.k_fp8) = quant(&self.k_cache, k_scale);
        (self.v_cache, self.v_fp8) = quant(&self.v_cache, v_scale);
        (self.kv_dtype, self.k_scale, self.v_scale) = (kv, k_scale, v_scale);
        self
    }

    pub fn is_ragged(&self) -> bool {
        self.page_size == RAGGED
    }

    /// This problem with FlashInfer's logits soft-cap `cap` (Gemma 2).
    pub fn with_soft_cap(mut self, cap: f32) -> Problem {
        (self.variant.logits_soft_cap, self.logits_soft_cap) = (true, cap);
        self
    }

    /// This problem with FlashInfer's sliding window `window_left`.
    pub fn with_window(mut self, window_left: i32) -> Problem {
        (self.variant.sliding_window, self.window_left) = (true, window_left);
        self
    }

    /// This problem with ALiBi, FlashInfer's slopes ([`alibi_slopes`]).
    pub fn with_alibi(mut self) -> Problem {
        (self.variant.alibi, self.alibi_slopes) = (true, alibi_slopes(self.num_qo_heads()));
        self
    }

    /// This problem with a random custom mask from `seed` (each bit set with probability
    /// one half; every query keeps its diagonal token), packed with FlashInfer's
    /// `segment_packbits` (little bit order) into `packed_mask` / `mask_indptr`.
    pub fn with_custom_mask(self, seed: u64) -> Problem {
        let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut bits = vec![];
        for (&qo, &kv) in self.qo_lens.iter().zip(&self.kv_lens) {
            for i in 0..qo {
                let diag = (i + kv).saturating_sub(qo).min(kv - 1);
                for t in 0..kv {
                    st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    bits.push(t == diag || st >> 40 & 1 == 1);
                }
            }
        }
        self.with_mask_bits(bits)
    }

    /// This problem with the custom mask `bits` (request by request, `qo_len * kv_len` bits
    /// each, row-major).
    pub fn with_mask_bits(mut self, bits: Vec<bool>) -> Problem {
        let mut qk_indptr = vec![0i32];
        for (&qo, &kv) in self.qo_lens.iter().zip(&self.kv_lens) {
            qk_indptr.push(qk_indptr.last().unwrap() + (qo * kv) as i32);
        }
        assert_eq!(bits.len(), *qk_indptr.last().unwrap() as usize, "one bit per (query, token)");
        let packer = crate::quantization::Problem { bits, indptr: qk_indptr.clone() };
        self.packed_mask = packer.reference(crate::quantization::Order::Little);
        self.mask_indptr = packer.out_indptr();
        (self.variant.custom_mask, self.mask_bits, self.qk_indptr) = (true, packer.bits, qk_indptr);
        self
    }

    /// Whether query `i` of request `b` sees token `t` (causal, window and custom mask).
    pub fn visible(&self, b: usize, i: usize, t: usize) -> bool {
        let (qo, kv) = (self.qo_lens[b], self.kv_lens[b]);
        (!self.causal || t + qo <= kv + i)
            && self.variant.in_window(self.window_left, qo, kv, i, t)
            && (!self.variant.custom_mask || self.mask_bits[self.qk_indptr[b] as usize + i * kv + t])
    }

    /// The KV pages the planner counts for request `b` under query tile `cta_tile_q`: with
    /// a causal window, FlashInfer's `ceil((window_left + cta_tile_q) / page_size)` at most.
    fn plan_pages(&self, b: usize, cta_tile_q: usize) -> usize {
        let pages = (self.kv_indptr[b + 1] - self.kv_indptr[b]) as usize;
        if self.windowed_causal() {
            pages.min((self.window_left as usize + cta_tile_q).div_ceil(self.page_tokens()))
        } else {
            pages
        }
    }

    fn windowed_causal(&self) -> bool {
        self.variant.sliding_window && self.causal && self.window_left >= 0
    }

    /// Tokens per page (1 for a ragged problem, as FlashInfer's ragged planner).
    pub fn page_tokens(&self) -> usize {
        self.page_size.max(1)
    }

    pub fn num_qo_heads(&self) -> usize {
        self.num_kv_heads * self.group
    }

    pub fn total_q(&self) -> usize {
        *self.qo_indptr.last().unwrap() as usize
    }

    pub fn plan(&self, max_grid: usize) -> Plan {
        self.plan_with_min_chunk(max_grid, (128 / self.page_tokens()).max(1))
    }

    pub fn plan_with_min_chunk(&self, max_grid: usize, min_chunk_pages: usize) -> Plan {
        let packed: usize = self.qo_lens.iter().sum::<usize>() * self.group;
        let cta_tile_q = fa2_cta_tile_q(packed / self.qo_lens.len());
        let pages: Vec<usize> = (0..self.qo_lens.len()).map(|b| self.plan_pages(b, cta_tile_q)).collect();
        Plan::with_min_chunk(
            &self.qo_lens,
            &pages,
            self.page_tokens(),
            self.group,
            self.num_kv_heads,
            max_grid,
            min_chunk_pages,
        )
        .for_kv(self.kv_dtype)
    }

    pub fn spec(&self, plan: &Plan) -> KernelSpec {
        KernelSpec {
            tile: plan.tile,
            group: self.group,
            page_size: self.page_size,
            causal: self.causal,
            partition: plan.split,
            variant: self.variant,
        }
    }

    pub fn launch(&self, plan: &Plan) -> LaunchConfig {
        LaunchConfig::new([plan.num_ctas() as u32, self.num_kv_heads as u32, 1], plan.tile.block())
    }

    /// The merge's launch: one CTA per (query token, query head).
    pub fn merge_launch(&self) -> LaunchConfig {
        LaunchConfig::new([self.total_q() as u32, self.num_qo_heads() as u32, 1], [HEAD_DIM as u32, 1, 1])
    }

    /// The prefill kernel's arguments, in the order of [`p`].
    pub fn args(&self, plan: &Plan) -> Vec<KernelArg> {
        let n_o = self.q.len();
        let rows = plan.tmp_rows();
        vec![
            KernelArg::buffer(&self.q),
            self.cache_arg(&self.k_cache, &self.k_fp8),
            self.cache_arg(&self.v_cache, &self.v_fp8),
            KernelArg::i32(self.qo_indptr.clone()),
            KernelArg::i32(self.kv_indptr.clone()),
            KernelArg::i32(self.kv_indices.clone()),
            KernelArg::i32(self.kv_last_page_len.clone()),
            KernelArg::i32(plan.request_indices.clone()),
            KernelArg::i32(plan.qo_tile_indices.clone()),
            KernelArg::i32(plan.kv_tile_indices.clone()),
            KernelArg::i32(plan.o_indptr.clone()),
            KernelArg::buffer(&vec![F16::from_f32(0.0); n_o]),
            KernelArg::f32(vec![0.0; n_o / HEAD_DIM]),
            KernelArg::f32(vec![0.0; rows * self.num_qo_heads() * HEAD_DIM]),
            KernelArg::f32(vec![0.0; rows * self.num_qo_heads()]),
            KernelArg::i32(plan.merge_indptr.clone()),
            KernelArg::ScalarI32(self.num_kv_heads as i32),
            KernelArg::ScalarF32(self.variant.scales(self.sm_scale * self.k_scale, self.logits_soft_cap).0),
            KernelArg::ScalarI32(plan.kv_chunk_size as i32),
            KernelArg::ScalarF32(self.variant.scales(self.sm_scale * self.k_scale, self.logits_soft_cap).1),
            KernelArg::ScalarI32(self.window_left),
            KernelArg::f32(if self.alibi_slopes.is_empty() { vec![0.0] } else { self.alibi_slopes.clone() }),
            KernelArg::Bytes {
                elem: DType::U8,
                data: if self.packed_mask.is_empty() { vec![0] } else { self.packed_mask.clone() },
            },
            KernelArg::i32(self.mask_indptr.clone()),
            KernelArg::ScalarF32(self.v_scale),
        ]
    }

    fn cache_arg(&self, f16: &[F16], fp8: &[u8]) -> KernelArg {
        match self.kv_dtype {
            KvDtype::F16 => KernelArg::buffer(f16),
            KvDtype::E4M3 => KernelArg::buffer(&fp8.iter().map(|&b| F8E4M3FN(b)).collect::<Vec<_>>()),
            KvDtype::E5M2 => KernelArg::buffer(&fp8.iter().map(|&b| F8E5M2(b)).collect::<Vec<_>>()),
        }
    }

    /// Attention in f64 on the f16 inputs: `o` and base-2 `lse`, like FlashInfer's PyTorch
    /// reference (softmax(q k^T * sm_scale) v over the gathered pages; with `causal`, query
    /// `i` of a request with `qo_len` queries and `kv_len` tokens sees tokens up to `i +
    /// kv_len - qo_len`).
    pub fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        let (hkv, n_qo, ps) = (self.num_kv_heads, self.num_qo_heads(), self.page_tokens());
        let f = |x: F16| x.to_f32() as f64;
        let (sm, cap) = (self.sm_scale as f64 * self.k_scale as f64, self.logits_soft_cap as f64);
        let mut o = vec![0f32; self.q.len()];
        let mut lse = vec![0f32; self.q.len() / HEAD_DIM];
        for (b, (&qo_len, &kv_len)) in self.qo_lens.iter().zip(&self.kv_lens).enumerate() {
            let addr = |t: usize, h: usize| {
                let start = self.kv_indptr[b] as usize;
                let row =
                    if self.is_ragged() { start + t } else { self.kv_indices[start + t / ps] as usize * ps + t % ps };
                (row * hkv + h) * HEAD_DIM
            };
            for i in 0..qo_len {
                let token = self.qo_indptr[b] as usize + i;
                let visible: Vec<usize> = (0..kv_len).filter(|&t| self.visible(b, i, t)).collect();
                for qh in 0..n_qo {
                    let h = qh / self.group;
                    let qrow = (token * n_qo + qh) * HEAD_DIM;
                    let slope = self.alibi_slopes.get(qh).map_or(0.0, |&x| x as f64);
                    let s: Vec<f64> = visible
                        .iter()
                        .map(|&t| {
                            let a = addr(t, h);
                            let qk = (0..HEAD_DIM).map(|e| f(self.q[qrow + e]) * f(self.k_cache[a + e])).sum::<f64>();
                            self.variant.ref_logit(qk, sm, cap, slope, i, t)
                        })
                        .collect();
                    let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                    let sum: f64 = w.iter().sum();
                    lse[token * n_qo + qh] = ((mx + sum.ln()) * std::f64::consts::LOG2_E) as f32;
                    for e in 0..HEAD_DIM {
                        let acc: f64 =
                            visible.iter().zip(&w).map(|(&t, wt)| wt * f(self.v_cache[addr(t, h) + e])).sum();
                        o[qrow + e] = (acc / sum * self.v_scale as f64) as f32;
                    }
                }
            }
        }
        (o, lse)
    }

    /// Floating-point operations, as `flashinfer.testing.attention_flops` counts them
    /// (causal: `(2 kv_len - qo_len) qo_len` score/value pairs per head and dimension).
    pub fn flops(&self) -> f64 {
        let h = self.num_qo_heads() as f64;
        self.qo_lens
            .iter()
            .zip(&self.kv_lens)
            .map(|(&q, &k)| {
                let (q, k) = (q as f64, k as f64);
                if self.causal {
                    (2.0 * k - q) * q * h * 2.0 * HEAD_DIM as f64
                } else {
                    4.0 * q * k * h * HEAD_DIM as f64
                }
            })
            .sum()
    }
}

pub use crate::decode::max_error;

/// The merge kernel's arguments, taken from the prefill kernel's (`merge_indptr` plays
/// the decode merge's `o_indptr`, one "request" per query token).
pub fn merge_args(p: &Problem, args: &[KernelArg]) -> Vec<KernelArg> {
    vec![
        args[p::TMP_O].clone(),
        args[p::TMP_LSE].clone(),
        args[p::MERGE_INDPTR].clone(),
        args[p::O].clone(),
        args[p::LSE].clone(),
        KernelArg::ScalarI32(p.num_qo_heads() as i32),
    ]
}

/// Simulate `p` under a plan for `max_grid` concurrent CTAs (the prefill kernel, then the
/// merge if the plan splits) and return the plan and `(max |o - ref|, max |lse - ref|)`.
pub fn simulate(p: &Problem, max_grid: usize, min_chunk_pages: usize) -> (Plan, f32, f32) {
    let plan = p.plan_with_min_chunk(max_grid, min_chunk_pages);
    let (eo, el) = simulate_plan(p, &plan);
    (plan, eo, el)
}

/// Run `p` under `plan` on the simulator; returns `o` (as f32) and `lse`.
pub fn execute(p: &Problem, plan: &Plan) -> (Vec<f32>, Vec<f32>) {
    let mut args = p.args(plan);
    let k = kernel(&p.spec(plan));
    k.simulate(p.launch(plan), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    if plan.split {
        let mut margs = merge_args(p, &args);
        crate::decode::merge_kernel().simulate(p.merge_launch(), &mut margs).unwrap_or_else(|e| panic!("merge: {e}"));
        args[p::O] = margs[crate::decode::mp::O].clone();
        args[p::LSE] = margs[crate::decode::mp::LSE].clone();
    }
    let o: Vec<f32> = args[p::O].to_vec::<F16>().into_iter().map(|x| x.to_f32()).collect();
    (o, args[p::LSE].as_f32().to_vec())
}

/// [`execute`] checked against the f64 reference: `(max |o - ref|, max |lse - ref|)`.
pub fn simulate_plan(p: &Problem, plan: &Plan) -> (f32, f32) {
    let (o, lse) = execute(p, plan);
    let (want_o, want_lse) = p.reference();
    (max_error(&o, &want_o), max_error(&lse, &want_lse))
}

/// FlashInfer's `single_prefill_with_kv_cache` (NHD, `fa2`) on the simulator: `q [qo_len,
/// num_qo_heads, HEAD_DIM]`, contiguous `k`, `v [kv_len, num_kv_heads, HEAD_DIM]`, the
/// ragged kernel under [`Plan::single`] for `max_grid` concurrent CTAs. Returns `o` (as
/// f32) and the base-2 `lse [qo_len, num_qo_heads]`.
pub fn single_prefill_with_kv_cache(
    q: Vec<F16>,
    k: Vec<F16>,
    v: Vec<F16>,
    num_qo_heads: usize,
    num_kv_heads: usize,
    causal: bool,
    max_grid: usize,
) -> (Vec<f32>, Vec<f32>) {
    let p = single_problem(q, k, v, num_qo_heads, num_kv_heads, causal);
    execute(&p, &p.single_plan(max_grid))
}

/// The ragged one-request problem behind [`single_prefill_with_kv_cache`].
pub fn single_problem(
    q: Vec<F16>,
    k: Vec<F16>,
    v: Vec<F16>,
    num_qo_heads: usize,
    num_kv_heads: usize,
    causal: bool,
) -> Problem {
    assert_eq!(num_qo_heads % num_kv_heads, 0);
    let qo_len = q.len() / (num_qo_heads * HEAD_DIM);
    let kv_len = k.len() / (num_kv_heads * HEAD_DIM);
    Problem::from_ragged(&[qo_len], &[kv_len], num_kv_heads, num_qo_heads / num_kv_heads, causal, q, k, v)
}

impl Problem {
    /// [`Plan::single`] for a one-request problem.
    pub fn single_plan(&self, max_grid: usize) -> Plan {
        assert_eq!(self.qo_lens.len(), 1, "a single-request plan needs one request");
        let kv = if self.windowed_causal() {
            let cta_tile_q = fa2_cta_tile_q(self.qo_lens[0] * self.group);
            self.kv_lens[0].min(self.window_left as usize + cta_tile_q)
        } else {
            self.kv_lens[0]
        };
        Plan::single(self.qo_lens[0], kv, self.group, self.num_kv_heads, max_grid).for_kv(self.kv_dtype)
    }
}

/// Running on the GPU (`--features cuda`): kernels compiled and loaded on first use, and
/// one problem run, checked against the f64 reference and optionally timed.
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

    /// The prefill kernels, compiled and loaded when first needed, and the merge kernel.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: Toolchain,
        opts: CompileOptions,
        cache: RefCell<HashMap<KernelSpec, Loaded<'c>>>,
        pub merge: (Kernel, CudaFunction<'c>),
        /// CTAs the planner assumes run at once: 2 per SM, as FlashInfer's.
        pub max_grid: usize,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: Toolchain) -> Result<Kernels<'c>, Box<dyn Error>> {
            let arch = ctx.best_target(&tc).ok_or("no target for this GPU")?;
            let opts = CompileOptions::new().gpu_arch(arch);
            let merge_k = crate::decode::merge_kernel();
            let merge_f = ctx.load(&merge_k.compile_with(&tc, &opts)?)?;
            let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            Ok(Kernels {
                ctx,
                tc,
                opts,
                cache: RefCell::new(HashMap::new()),
                merge: (merge_k, merge_f),
                max_grid: 2 * sms,
            })
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
    /// time the prefill (and merge) launches replayed from a CUDA graph, `iterations`
    /// times. Returns the plan, the errors and the average GPU time in microseconds.
    #[allow(clippy::type_complexity)]
    pub fn run(
        ks: &Kernels<'_>,
        p: &Problem,
        check: bool,
        time: Option<usize>,
    ) -> Result<(Plan, f32, f32, Option<f64>), Box<dyn Error>> {
        let plan = p.plan(ks.max_grid);
        let (eo, el, t) = run_plan(ks, p, &plan, check, time)?;
        Ok((plan, eo, el, t))
    }

    /// [`run`] under a given plan: the errors (0 unless `check`) and the GPU time.
    pub fn run_plan(
        ks: &Kernels<'_>,
        p: &Problem,
        plan: &Plan,
        check: bool,
        time: Option<usize>,
    ) -> Result<(f32, f32, Option<f64>), Box<dyn Error>> {
        let (out, t) = launch(ks, p, plan, check, time)?;
        let (mut eo, mut el) = (0.0, 0.0);
        if let Some((o, lse)) = out {
            let (want_o, want_lse) = p.reference();
            (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
        }
        Ok((eo, el, t))
    }

    /// FlashInfer's `single_prefill_with_kv_cache` (NHD, `fa2`) on the GPU: the ragged
    /// kernel (and the merge, if [`Plan::single`] splits) on `q [qo_len, num_qo_heads,
    /// HEAD_DIM]` and contiguous `k`, `v [kv_len, num_kv_heads, HEAD_DIM]`. Returns `o`
    /// (as f32) and the base-2 `lse`.
    #[allow(clippy::type_complexity)]
    pub fn single_prefill_with_kv_cache(
        ks: &Kernels<'_>,
        q: Vec<F16>,
        k: Vec<F16>,
        v: Vec<F16>,
        num_qo_heads: usize,
        num_kv_heads: usize,
        causal: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        let p = super::single_problem(q, k, v, num_qo_heads, num_kv_heads, causal);
        let (out, _) = launch(ks, &p, &p.single_plan(ks.max_grid), true, None)?;
        Ok(out.unwrap())
    }

    /// Upload `p`, run it once under `plan`, download `o` and `lse` if `fetch`, and time
    /// it from a CUDA graph if `time` is `Some(iterations)`.
    #[allow(clippy::type_complexity)]
    pub fn launch(
        ks: &Kernels<'_>,
        p: &Problem,
        plan: &Plan,
        fetch: bool,
        time: Option<usize>,
    ) -> Result<(Option<(Vec<f32>, Vec<f32>)>, Option<f64>), Box<dyn Error>> {
        let ctx = ks.ctx;
        let loaded = ks.get(&p.spec(plan))?;
        let (pk, pf) = (&loaded.0, &loaded.1);
        let host = p.args(plan);
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
        use super::p as pp;
        let margs = vec![
            DeviceArg::Buffer(&bufs[pp::TMP_O]),
            DeviceArg::Buffer(&bufs[pp::TMP_LSE]),
            DeviceArg::Buffer(&bufs[pp::MERGE_INDPTR]),
            DeviceArg::Buffer(&bufs[pp::O]),
            DeviceArg::Buffer(&bufs[pp::LSE]),
            DeviceArg::Scalar(KernelArg::ScalarI32(p.num_qo_heads() as i32)),
        ];
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(pk, pf, p.launch(plan)).stream(s).args(&args).submit()?;
            if plan.split {
                ctx.launch_builder(&ks.merge.0, &ks.merge.1, p.merge_launch()).stream(s).args(&margs).submit()?;
            }
            Ok(())
        };
        let stream = ctx.stream()?;
        launch(&stream, 0)?;
        stream.synchronize()?;
        let out = if fetch {
            let o: Vec<f32> = bufs[pp::O].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect();
            Some((o, bufs[pp::LSE].to_vec::<f32>()?))
        } else {
            None
        };
        let t = match time {
            Some(iterations) => {
                let opts = BenchmarkOptions::new().warmup_iterations(3).iterations(iterations).use_cuda_graphs(true);
                Some(benchmark(&stream, &opts, &mut launch)?)
            }
            None => None,
        };
        Ok((out, t))
    }
}
