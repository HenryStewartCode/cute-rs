// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batch decode attention over a paged KV cache, after FlashInfer's
//! `BatchDecodeWithPagedKVCacheKernel` (`include/flashinfer/attention/decode.cuh`,
//! the `fa2` backend), with its split-KV plan (`scheduler.cuh`) and merge step
//! (`cascade.cuh`). It follows FlashInfer's design
//! (thread geometry, cp.async staging, base-2 online softmax, cross-`tz` state merge,
//! the planner's chunk-size rule).
//!
//! * **Plan** ([`Plan::new`], on the host, once per batch layout, as FlashInfer's
//!   `plan()`):
//!   * if `batch * num_kv_heads` already fills the GPU (the occupancy times the SM
//!     count), every request is one CTA;
//!   * otherwise the smallest chunk size, in pages (at least `128 / page_size`), is
//!     found whose chunks still fit in one wave, and each request is split into chunks.
//!     The plan lists each CTA's request and chunk, and where each request's partial
//!     results start (`o_indptr`).
//! * **Decode kernel**: grid `(num_ctas, num_kv_heads)`, block `(BDX, BDY, BDZ)`:
//!   * `BDX = HEAD_DIM / VEC` threads each own `VEC` elements of the head dimension;
//!   * `BDY = GROUP` threads, one per query head sharing the KV head (GQA);
//!   * `BDZ` groups each take a slice of every tile of tokens.
//!
//!   Each iteration stages `TOKENS_PER_ITER` tokens of the CTA's chunk in shared
//!   memory with 16-byte `cp.async` copies (two stages: the next tile loads while this
//!   one is used). A token's address is gathered through the page table:
//!   `kv_indices[kv_indptr[b] + t / page_size]`, then entry `t % page_size`. Each
//!   thread takes the dot product of its query head with each staged token on its
//!   `VEC` elements, reduces it over the `BDX` lanes with a butterfly, and updates a
//!   base-2 online softmax `(m, d, o[VEC])` with hardware `exp2`. The `BDZ` partial
//!   states are then merged through shared memory.
//!   * Without split-KV it writes `o` (Float16) and FlashInfer's base-2 log-sum-exp
//!     `lse = m + log2(d)`.
//!   * With split-KV (the `partition` variant) it writes each chunk's normalised output
//!     and lse to Float32 scratch.
//! * **Merge kernel** (split-KV only): grid `(batch, num_qo_heads)`, `HEAD_DIM` threads.
//!   It folds each request's chunk states with weights `2^(lse_i - max)`, then writes
//!   `o` and `lse`.
//!
//! **Contiguous KV** ([`kernel_ragged`], and [`single_decode_with_kv_cache`] over it): the
//! same kernel built to read K and V from contiguous `[total_kv, num_kv_heads, HEAD_DIM]`
//! tensors, request `b` owning tokens `kv_indptr[b] .. kv_indptr[b + 1]`; only the token
//! address and `kv_len` change. FlashInfer's `SingleDecodeWithKVCacheKernel` has the same
//! geometry for f16, head dim 128, group 4 (`bdz = 128 / (bdx * bdy) = 2`,
//! `tile_size_per_bdx = 1`, 2 stages), and the single-request wrapper follows its
//! `SingleDecodeWithKVCacheDispatched` split rule ([`Plan::single`]): whole up to 256
//! tokens, else chunks of `max(ceil(kv_len / (max_grid / num_kv_heads)), 256)` tokens
//! merged by the merge kernel.
//!
//! The cache layout is FlashInfer's NHD: `[num_pages, page_size, num_kv_heads, HEAD_DIM]`.
//!
//! Configuration (FlashInfer's choices for f16, head dim 128, group 4):
//! `VEC = 8`, `BDX = 16`, `BDY = 4`, `BDZ = 2`, `tile_size_per_bdx = 1`, 2 stages.
//!
//! **FP8 KV** ([`KvDtype`], a build-time choice: [`kernel_variant_kv`],
//! [`Problem::quantized`], [`gpu::Kernels::with_kv`]): an E4M3 or E5M2 cache with Float16
//! queries and output, as FlashInfer's decode with `kv_data_type=torch.float8_e4m3fn`.
//! The vector width doubles to 16 elements (still 16-byte `cp.async`), so `BDX = 8` and
//! `BDZ = 4` (128 threads); staged FP8 is widened two at a time with the hardware
//! `cvt.rn.f16x2.e4m3x2` (FlashInfer's `cast_load`), and the K/V stages are reused for the
//! `BDZ` merge. FlashInfer's `k_scale` folds into `sm_scale` on the host, as its Python
//! does; `v_scale` scales the output in the kernel (FlashInfer applies it afterwards).
//!
//! **Attention variants** ([`Variant`], build-time, FlashInfer's `DefaultAttention<
//! use_custom_mask, use_sliding_window, use_logits_soft_cap, use_alibi>` in
//! `attention/variants.cuh`; [`kernel_full`], [`Problem::with_soft_cap`],
//! [`Problem::with_window`], [`Problem::with_alibi`], [`gpu::Kernels::with_variant`]):
//! * logits soft-cap (Gemma 2): the score is `cap * tanh(s * sm_scale / cap)`, with the
//!   hardware `tanh.approx.f32` as FlashInfer;
//! * sliding window `window_left` (FlashInfer's `kv_idx + window_left + 1 >= kv_len` for a
//!   decode query; negative turns it off): the CTA starts at the window's first token,
//!   and the plan counts only the window's pages, so out-of-window tokens are never read
//!   (FlashInfer's decode kernel masks them instead);
//! * ALiBi: `s * sm_scale + slope[head] * kv_idx` ([`alibi_slopes`]), not combined with
//!   soft-cap.
//!
//! **Head dims and BFloat16** ([`kernel_shape`], [`Problem::new_hd`], [`Problem::bf16`]): head
//! dim 64, 128 or 256 and, build-time, BFloat16 queries, cache and output (FlashInfer's
//! `fa2` instantiations); the geometry follows FlashInfer's `vec = max(16 / sizeof(KV),
//! head_dim / 32)`. The head-dim-128 Float16 kernels are unchanged.
//!
//! The plain kernel is unchanged: each variant's code is only built into its own kernel.
//!
//! Not yet covered:
//! * RoPE in the kernel;
//! * the HND layout and other group sizes;
//! * FP8 caches at head dims other than 128, and FP8 with BFloat16 queries;
//! * CUDA-graph mode (FlashInfer then always splits, into fixed-size buffers).
//!
//! Every request must have at least one token.
//!
//! `examples/batch_decode.rs` runs it on the simulator and, with `--features cuda` and
//! `GPU=1`, on the GPU against a reference, then benchmarks it.

use cute::prelude::*;

/// Head dimension.
pub const HEAD_DIM: usize = 128;
/// Query heads per KV head.
pub const GROUP: usize = 4;
/// Elements per thread along the head dimension (16 bytes of Float16).
pub const VEC: usize = 8;
pub const BDX: usize = HEAD_DIM / VEC;
pub const BDY: usize = GROUP;
pub const BDZ: usize = 2;
/// Tokens per `(ty, tz)` thread row per iteration (FlashInfer's `tile_size_per_bdx`).
pub const TILE: usize = 1;
/// Shared-memory stages of the decode kernel: one tile loads while one is used
/// (FlashInfer's choice). 3 and 4 stages measured no faster on the RTX 2000 Ada, and the
/// extra shared memory cut the blocks per SM.
pub const STAGES: usize = 2;
pub const TOKENS_PER_Z: usize = BDY * TILE;
pub const TOKENS_PER_ITER: usize = BDZ * TOKENS_PER_Z;
pub const STAGE_ELEMS: usize = TOKENS_PER_ITER * HEAD_DIM;
/// The `page_size` of a problem over contiguous (ragged) K/V.
pub const RAGGED: usize = 0;

/// The element type of the K/V cache, a build-time choice (FlashInfer's `kv_data_type`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvDtype {
    /// Float16 (the default).
    #[default]
    F16,
    /// `torch.float8_e4m3fn`.
    E4M3,
    /// `torch.float8_e5m2`.
    E5M2,
}

impl KvDtype {
    /// Bytes per element.
    pub fn bytes(self) -> usize {
        if self == KvDtype::F16 { 2 } else { 1 }
    }
    /// Elements per thread along the head dimension (16 bytes).
    pub fn vec(self) -> usize {
        16 / self.bytes()
    }
    pub fn bdx(self) -> usize {
        self.bdx_hd(HEAD_DIM)
    }
    /// [`KvDtype::bdx`] for head dimension `head_dim` (FlashInfer: `HEAD_DIM / vec_size`).
    pub fn bdx_hd(self, head_dim: usize) -> usize {
        head_dim / self.vec_hd(head_dim)
    }
    /// Elements per thread for head dimension `head_dim`: FlashInfer's
    /// `max(16 / sizeof(DTypeKV), HEAD_DIM / 32)`.
    pub fn vec_hd(self, head_dim: usize) -> usize {
        self.vec().max(head_dim / 32)
    }
    /// `BDZ`: 128 threads in all (FlashInfer's `get_heuristic_num_threads` for group 4).
    pub fn bdz(self) -> usize {
        self.bdz_hd(HEAD_DIM)
    }
    pub fn bdz_hd(self, head_dim: usize) -> usize {
        128 / (self.bdx_hd(head_dim) * BDY)
    }
    pub fn tokens_per_iter(self) -> usize {
        self.tokens_per_iter_hd(HEAD_DIM)
    }
    pub fn tokens_per_iter_hd(self, head_dim: usize) -> usize {
        self.bdz_hd(head_dim) * TOKENS_PER_Z
    }
    pub fn block(self) -> [u32; 3] {
        self.block_hd(HEAD_DIM)
    }
    pub fn block_hd(self, head_dim: usize) -> [u32; 3] {
        [self.bdx_hd(head_dim) as u32, BDY as u32, self.bdz_hd(head_dim) as u32]
    }
    fn suffix(self) -> &'static str {
        match self {
            KvDtype::F16 => "",
            KvDtype::E4M3 => "_e4m3",
            KvDtype::E5M2 => "_e5m2",
        }
    }
    /// `x` rounded to this type (round to nearest even, saturating for FP8), as Float16
    /// (exact: both FP8 formats widen to Float16 exactly), and its stored bits.
    pub fn quantize(self, x: f32) -> (F16, u8) {
        match self {
            KvDtype::F16 => (F16::from_f32(x), 0),
            KvDtype::E4M3 => {
                let q = F8E4M3FN::from_f32(x);
                (F16::from_f32(q.to_f32()), q.0)
            }
            KvDtype::E5M2 => {
                let q = F8E5M2::from_f32(x);
                (F16::from_f32(q.to_f32()), q.0)
            }
        }
    }
}

/// FlashInfer's attention variant flags (`DefaultAttention<use_custom_mask,
/// use_sliding_window, use_logits_soft_cap, use_alibi>`), fixed when a kernel is built.
/// Decode supports all but `custom_mask`; prefill supports all four.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Variant {
    pub custom_mask: bool,
    pub sliding_window: bool,
    pub logits_soft_cap: bool,
    pub alibi: bool,
}

impl Variant {
    /// FlashInfer's default: no variant.
    pub const NONE: Variant =
        Variant { custom_mask: false, sliding_window: false, logits_soft_cap: false, alibi: false };

    /// The kernel-name suffix (empty for [`Variant::NONE`]).
    pub fn suffix(&self) -> String {
        let mut s = String::new();
        for (on, tag) in [
            (self.custom_mask, "_mask"),
            (self.sliding_window, "_win"),
            (self.logits_soft_cap, "_cap"),
            (self.alibi, "_alibi"),
        ] {
            if on {
                s.push_str(tag);
            }
        }
        s
    }

    /// FlashInfer's `sm_scale_log2` and the scale applied before the logits transform,
    /// for softmax scale `sm_scale` and soft-cap `cap`: `(cap log2(e), sm_scale / cap)` with
    /// soft-cap, `(log2(e), sm_scale)` with ALiBi, else `(sm_scale log2(e), sm_scale)`.
    pub fn scales(&self, sm_scale: f32, cap: f32) -> (f32, f32) {
        let log2e = std::f32::consts::LOG2_E;
        assert!(!(self.logits_soft_cap && self.alibi), "soft-cap with ALiBi is not supported");
        if self.logits_soft_cap {
            assert!(cap > 0.0, "the soft-cap must be positive");
            (cap * log2e, sm_scale / cap)
        } else if self.alibi {
            (log2e, sm_scale)
        } else {
            (sm_scale * log2e, sm_scale)
        }
    }

    /// The natural-base logit of a raw dot product `qk` (f64 reference): `qk * sm_scale`,
    /// plus `slope * (kv_idx - q_idx)` with ALiBi, then `cap * tanh(s / cap)` with soft-cap.
    pub fn ref_logit(&self, qk: f64, sm_scale: f64, cap: f64, slope: f64, q_idx: usize, kv_idx: usize) -> f64 {
        let mut s = qk * sm_scale;
        if self.alibi {
            s += slope * (kv_idx as f64 - q_idx as f64);
        }
        if self.logits_soft_cap {
            s = cap * (s / cap).tanh();
        }
        s
    }

    /// Whether token `kv_idx` is inside query `q_idx`'s window (FlashInfer's
    /// `kv_idx + qo_len + window_left >= kv_len + q_idx`; always with a negative
    /// `window_left` or without the variant).
    pub fn in_window(&self, window_left: i32, qo_len: usize, kv_len: usize, q_idx: usize, kv_idx: usize) -> bool {
        !self.sliding_window || window_left < 0 || kv_idx + qo_len + window_left as usize >= kv_len + q_idx
    }
}

/// FlashInfer's `get_alibi_slopes(n_heads)`: `2^(-8 i / n)` for `i = 1..=n` (`n` the
/// largest power of two up to `n_heads`), then `2^(-4 i / n)` for odd `i` for the rest.
pub fn alibi_slopes(n_heads: usize) -> Vec<f32> {
    let n = 1usize << (usize::BITS - 1 - n_heads.leading_zeros());
    let mut m: Vec<f32> = (1..=n).map(|i| 2f64.powf(-8.0 * i as f64 / n as f64) as f32).collect();
    m.extend((0..n_heads - n).map(|j| 2f64.powf(-4.0 * (2 * j + 1) as f64 / n as f64) as f32));
    m
}

/// Parameter positions of the decode kernel (shared by both variants).
pub mod p {
    pub const Q: usize = 0;
    pub const K: usize = 1;
    pub const V: usize = 2;
    pub const KV_INDPTR: usize = 3;
    pub const KV_INDICES: usize = 4;
    pub const LAST_PAGE_LEN: usize = 5;
    pub const REQUEST_INDICES: usize = 6;
    pub const KV_TILE_INDICES: usize = 7;
    pub const O_INDPTR: usize = 8;
    pub const O: usize = 9;
    pub const LSE: usize = 10;
    pub const TMP_O: usize = 11;
    pub const TMP_LSE: usize = 12;
    pub const PAGE_SIZE: usize = 13;
    pub const NUM_KV_HEADS: usize = 14;
    pub const SM_SCALE_LOG2: usize = 15;
    pub const CHUNK_TOKENS: usize = 16;
    /// The output scale (FlashInfer's `v_scale`; 1 for a Float16 cache).
    pub const V_SCALE: usize = 17;
    /// The scale before the logits transform ([`Variant::scales`]; unread by the plain
    /// kernel).
    pub const LOGITS_PRE_SCALE: usize = 18;
    /// FlashInfer's `window_left` (negative: no window; read only with a sliding window).
    pub const WINDOW_LEFT: usize = 19;
    /// ALiBi slopes `[num_qo_heads]` (read only with ALiBi).
    pub const ALIBI_SLOPES: usize = 20;
}

/// The decode kernel. `partition` selects the split-KV variant, which writes each
/// chunk's state to `tmp_o`/`tmp_lse` (row `o_indptr[b] + chunk`) instead of `o`/`lse`.
/// Parameters are in the order of [`p`]: `q [batch, num_kv_heads * GROUP, HEAD_DIM]`,
/// `k_cache`, `v_cache` (NHD pages), `kv_indptr [batch + 1]`, `kv_indices`,
/// `kv_last_page_len [batch]`, `request_indices [ctas]`, `kv_tile_indices [ctas]`,
/// `o_indptr [batch + 1]`, `o` (like `q`), `lse [batch, num_qo_heads]`,
/// `tmp_o [rows, num_qo_heads, HEAD_DIM]`, `tmp_lse [rows, num_qo_heads]`, and the
/// scalars `page_size`, `num_kv_heads`, `sm_scale_log2` (`sm_scale * log2(e)`, times
/// `k_scale` for a scaled FP8 cache), `kv_chunk_tokens` and `v_scale` (the output's scale).
pub fn kernel(partition: bool) -> Kernel {
    kernel_with_stages(partition, STAGES)
}

/// The decode kernel over contiguous (ragged) K/V: `k_cache`, `v_cache` are `[total_kv,
/// num_kv_heads, HEAD_DIM]`, `kv_indptr` counts tokens, and `kv_indices`,
/// `kv_last_page_len` and `page_size` are not read. Parameters as [`kernel`]'s.
pub fn kernel_ragged(partition: bool) -> Kernel {
    kernel_variant(partition, STAGES, true)
}

/// [`kernel`] with `stages` shared-memory stages (at least 2).
pub fn kernel_with_stages(partition: bool, stages: usize) -> Kernel {
    kernel_variant(partition, stages, false)
}

/// The decode kernel: split-KV or not, `stages` shared-memory stages, paged or ragged K/V
/// (a Float16 cache).
pub fn kernel_variant(partition: bool, stages: usize, ragged: bool) -> Kernel {
    kernel_variant_kv(partition, stages, ragged, KvDtype::F16)
}

/// [`kernel_variant`] over a K/V cache of `kv` elements (Float16 queries and output in
/// every case). An FP8 kernel reads 16 elements per thread (`BDX = 8`, `BDZ = 4`), as
/// FlashInfer's `vec_size = max(16 / sizeof(DTypeKV), HEAD_DIM / 32)` does.
pub fn kernel_variant_kv(partition: bool, stages: usize, ragged: bool, kv: KvDtype) -> Kernel {
    kernel_full(partition, stages, ragged, kv, Variant::NONE)
}

/// [`kernel_variant_kv`] built for an attention [`Variant`] (any but `custom_mask`).
pub fn kernel_full(partition: bool, stages: usize, ragged: bool, kv: KvDtype, variant: Variant) -> Kernel {
    kernel_shape(partition, stages, ragged, kv, variant, HEAD_DIM, false)
}

/// [`kernel_full`] for head dimension `head_dim` (64, 128 or 256; FP8 only at 128) and, with
/// `bf16`, BFloat16 queries, cache and output instead of Float16 (`kv` must then be
/// [`KvDtype::F16`], which stands for the unquantised two-byte cache). FlashInfer's
/// geometry: `VEC = max(16 / sizeof(DTypeKV), head_dim / 32)`, `BDX = head_dim / VEC`,
/// `BDZ = 128 / (BDX * BDY)`: 64 is `(8, 4, 4)` and 256 is `(32, 4, 1)`.
pub fn kernel_shape(
    partition: bool,
    stages: usize,
    ragged: bool,
    kv: KvDtype,
    variant: Variant,
    head_dim: usize,
    bf16: bool,
) -> Kernel {
    assert!(!variant.custom_mask, "decode has no custom mask");
    assert!(!(variant.logits_soft_cap && variant.alibi), "soft-cap with ALiBi is not supported");
    assert!(matches!(head_dim, 64 | 128 | 256), "head dim {head_dim} not supported");
    assert!(
        kv == KvDtype::F16 || (head_dim == 128 && !bf16),
        "FP8 K/V is built for head dim 128 and Float16 queries only"
    );
    match (kv, bf16) {
        (KvDtype::F16, false) => build::<F16, 8>(partition, stages, ragged, variant, head_dim),
        (KvDtype::F16, true) => build::<BF16, 8>(partition, stages, ragged, variant, head_dim),
        (KvDtype::E4M3, _) => build::<F8E4M3FN, 16>(partition, stages, ragged, variant, head_dim),
        (KvDtype::E5M2, _) => build::<F8E5M2, 16>(partition, stages, ragged, variant, head_dim),
    }
}

/// The element type of a K/V cache: how a vector of it widens to Float32.
pub trait KvElem: Numeric + Copy {
    const KV: KvDtype;
    /// The type of the queries and the output (Float16, but BFloat16 for a BFloat16 cache).
    type Out: Numeric + Copy;
    fn widen(v: &[Value<Self>]) -> Vec<Float32>;
}

impl KvElem for F16 {
    const KV: KvDtype = KvDtype::F16;
    type Out = F16;
    fn widen(v: &[Value<Self>]) -> Vec<Float32> {
        v.iter().map(|x| x.to::<f32>()).collect()
    }
}

/// FP8 widens two at a time with `cvt.rn.f16x2.{e4m3x2,e5m2x2}` (exact), then to Float32.
fn widen_fp8<F: KvElem + Fp8Format>(v: &[Value<F>]) -> Vec<Float32> {
    v.chunks(2)
        .flat_map(|pair| {
            let (lo, hi): (Float16, Float16) = cvt_f8x2_to_f16x2::<F, _>([pair[0], pair[1]], false);
            [lo.to::<f32>(), hi.to::<f32>()]
        })
        .collect()
}

/// BFloat16 queries, cache and output; its `KV` is the two-byte class, [`KvDtype::F16`].
impl KvElem for BF16 {
    const KV: KvDtype = KvDtype::F16;
    type Out = BF16;
    fn widen(v: &[Value<Self>]) -> Vec<Float32> {
        v.iter().map(|x| x.to::<f32>()).collect()
    }
}

impl KvElem for F8E4M3FN {
    const KV: KvDtype = KvDtype::E4M3;
    type Out = F16;
    fn widen(v: &[Value<Self>]) -> Vec<Float32> {
        widen_fp8(v)
    }
}

impl KvElem for F8E5M2 {
    const KV: KvDtype = KvDtype::E5M2;
    type Out = F16;
    fn widen(v: &[Value<Self>]) -> Vec<Float32> {
        widen_fp8(v)
    }
}

/// The decode kernel over `KV` elements, `VEC` (16 bytes) per thread.
fn build<KV: KvElem, const VEC: usize>(
    partition: bool,
    stages: usize,
    ragged: bool,
    variant: Variant,
    head_dim: usize,
) -> Kernel {
    assert!(stages >= 2, "the decode kernel needs at least 2 stages");
    let kv = KV::KV;
    let bf16 = std::any::TypeId::of::<KV>() == std::any::TypeId::of::<BF16>();
    assert_eq!(VEC, kv.vec_hd(head_dim));
    let (bdx, bdz) = (kv.bdx_hd(head_dim), kv.bdz_hd(head_dim));
    let tokens_per_iter = kv.tokens_per_iter_hd(head_dim);
    let stage_elems = tokens_per_iter * head_dim;
    let layout = if ragged { "ragged" } else { "paged" };
    let base_name = format!(
        "flashinfer_batch_decode_{layout}{}{}{}{}{}",
        if partition { "_split" } else { "" },
        kv.suffix(),
        variant.suffix(),
        if head_dim == HEAD_DIM { String::new() } else { format!("_hd{head_dim}") },
        if bf16 { "_bf16" } else { "" }
    );
    let name = if stages == STAGES { base_name } else { format!("{base_name}_s{stages}") };
    Kernel::builder(&name).block_dim(kv.block_hd(head_dim)).build(|| {
        let q = param_ptr::<KV::Out>("q");
        let k_cache = param_ptr::<KV>("k_cache");
        let v_cache = param_ptr::<KV>("v_cache");
        let kv_indptr = param_ptr::<i32>("kv_indptr");
        let kv_indices = param_ptr::<i32>("kv_indices");
        let kv_last_page_len = param_ptr::<i32>("kv_last_page_len");
        let request_indices = param_ptr::<i32>("request_indices");
        let kv_tile_indices = param_ptr::<i32>("kv_tile_indices");
        let o_indptr = param_ptr::<i32>("o_indptr");
        let o = param_ptr::<KV::Out>("o");
        let lse = param_ptr::<f32>("lse");
        let tmp_o = param_ptr::<f32>("tmp_o");
        let tmp_lse = param_ptr::<f32>("tmp_lse");
        let page_size = param_scalar::<i32>("page_size");
        let num_kv_heads = param_scalar::<i32>("num_kv_heads");
        let sm_scale_log2 = param_scalar::<f32>("sm_scale_log2");
        let chunk_tokens = param_scalar::<i32>("kv_chunk_tokens");
        let v_scale = param_scalar::<f32>("v_scale");
        let logits_pre_scale = param_scalar::<f32>("logits_pre_scale");
        let window_left = param_scalar::<i32>("window_left");
        let alibi_slopes = param_ptr::<f32>("alibi_slopes");

        let (tx, ty, tz) = (thread_idx().x, thread_idx().y, thread_idx().z);
        let kv_head = block_idx().y;
        let b = request_indices.load_at(block_idx().x);
        let chunk = kv_tile_indices.load_at(block_idx().x);
        let hd = head_dim as i32;
        let qo_head = kv_head * GROUP as i32 + ty;
        let num_qo_heads = num_kv_heads * GROUP as i32;
        let qo_row = (b * num_qo_heads + qo_head) * hd + tx * VEC as i32;

        // This thread's query slice, in Float32 registers (16-byte loads: one for a
        // Float16 cache, two for FP8).
        let q_vec: Vec<Float32> = (0..VEC / 8)
            .flat_map(|h| load_vector::<KV::Out, 8>(&q.offset(qo_row + 8 * h as i32), LoadOptions::default()))
            .map(|x| x.to::<f32>())
            .collect();

        // The CTA's tokens: [t_begin, t_begin + chunk_len) of request b.
        let page_begin = kv_indptr.load_at(b);
        let page_end = kv_indptr.load_at(b + 1);
        let kv_len = if ragged {
            page_end - page_begin
        } else {
            (page_end - page_begin - 1) * page_size + kv_last_page_len.load_at(b)
        };
        // A sliding window keeps the last `window_left + 1` tokens: the CTA's chunks start
        // at the window's first token (`kv_start`), and `kv_len` becomes the window's length.
        let (kv_start, kv_len) = if variant.sliding_window {
            let win = select(window_left.ge(0), kv_len.min(window_left + 1), kv_len);
            (Some(kv_len - win), win)
        } else {
            (None, kv_len)
        };
        let t_begin = chunk * chunk_tokens;
        // The chunk's first token in the request.
        let t_first = match kv_start {
            Some(s) => s + t_begin,
            None => t_begin,
        };
        let slope = if variant.alibi { Some(alibi_slopes.load_at(qo_head)) } else { None };
        let chunk_len = (kv_len - t_begin).min(chunk_tokens);
        let num_iters = (chunk_len + (tokens_per_iter as i32 - 1)) / tokens_per_iter as i32;

        // An FP8 kernel keeps K and V in one buffer and reuses it for the merge below, as
        // FlashInfer does: 8 KB rather than 16 KB lets 6 rather than 5 blocks share an SM,
        // so batch 16 (128 CTAs on 22 SMs) runs in one wave.
        let reuse = kv != KvDtype::F16;
        let stage_buf = Layout::col_major(stages * stage_elems);
        let (k_smem, v_smem) = if reuse {
            let kv_smem = make_smem_tensor_aligned::<KV>(Layout::col_major(2 * stages * stage_elems), 16);
            let v_ptr = kv_smem.ptr().offset((stages * stage_elems) as i32);
            (make_tensor(&kv_smem.ptr(), stage_buf.clone()), make_tensor(&v_ptr, stage_buf))
        } else {
            (make_smem_tensor_aligned::<KV>(stage_buf.clone(), 16), make_smem_tensor_aligned::<KV>(stage_buf, 16))
        };
        let atom = make_copy_atom::<KV>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        // 16-byte vectors as columns of `(VEC, n)` views: the layouts prove each column's
        // alignment where a flat runtime offset could not.
        const BOUND: usize = 1 << 30;
        let vecs = |p: &Ptr<KV>| make_tensor(p, Layout::col_major((VEC, BOUND / VEC)));
        let (k_g, v_g) = (vecs(&k_cache), vecs(&v_cache));
        let smem_vecs = |t: &Tensor<KV>| make_tensor(&t.ptr(), Layout::col_major((VEC, stages * stage_elems / VEC)));
        let (k_s, v_s) = (smem_vecs(&k_smem), smem_vecs(&v_smem));
        let hv = (head_dim / VEC) as i32;

        // This thread's token of iteration `iter` (one per `(ty, tz)` row; TILE = 1).
        // Tokens past the chunk's end are clamped to its last token (a valid address);
        // their scores are masked later.
        assert_eq!(TILE, 1);
        let local = tz * TOKENS_PER_Z as i32 + ty;
        let token = |iter: Int32| t_first + (iter * tokens_per_iter as i32 + local).min(chunk_len - 1);
        // (Ragged K/V has no page table: its "page" is unused.)
        let page_of = |iter: Int32| {
            if ragged { Int32::from(0) } else { kv_indices.load_at(page_begin + token(iter) / page_size) }
        };
        // Stage the token of iteration `iter`, whose page is `page`, into `stage`.
        let load_tile = |iter: Int32, page: Int32, stage: Int32| {
            let row = if ragged { page_begin + token(iter) } else { page * page_size + token(iter) % page_size };
            let gvec = (row * num_kv_heads + kv_head) * hv + tx;
            let svec = stage * (stage_elems / VEC) as i32 + local * hv + tx;
            for (g, s) in [(&k_g, &k_s), (&v_g, &v_s)] {
                copy(&atom, &g.slice((.., gvec)), &s.slice((.., svec)));
            }
        };

        // Prologue: the first `stages - 1` tiles in flight, and the page of the next one
        // already in a register, so the copies issued in the loop never wait on a
        // page-table read.
        for st in 0..stages - 1 {
            let it = Int32::from(st as i32);
            load_tile(it, page_of(it), it);
            cp_async_commit_group();
        }
        let next_page = Var::new(page_of(Int32::from(stages as i32 - 1)));

        // Online-softmax state. `m` starts finite, as FlashInfer's does, so a thread whose
        // tokens are all masked keeps `d = 0` without NaNs.
        let m = Var::new(Float32::from(-1.0e30f32));
        let d = Var::new(Float32::from(0.0f32));
        let acc: Vec<Var<Float32>> = (0..VEC).map(|_| Var::new(Float32::from(0.0f32))).collect();

        for_range(0, num_iters, 1, |iter| {
            let stage = iter % stages as i32;
            // Prefetch tile `iter + stages - 1` into the stage freed last iteration (an
            // over-the-end prefetch reads clamped, valid addresses and is never used),
            // then read the page of the tile after it; that read is first needed next
            // iteration, so its latency hides behind this iteration's work.
            let ahead = iter + (stages as i32 - 1);
            load_tile(ahead, next_page.get(), ahead % stages as i32);
            cp_async_commit_group();
            next_page.set(page_of(ahead + 1));
            cp_async_wait_group(stages as u32 - 1);
            sync_threads();

            for i in 0..TOKENS_PER_Z {
                let local = tz * TOKENS_PER_Z as i32 + i as i32;
                let t = iter * tokens_per_iter as i32 + local;
                let base = stage * stage_elems as i32 + local * hd + tx * VEC as i32;
                // 16-byte shared-memory loads of this thread's K and V slices.
                let kv = load_vector::<KV, VEC>(&k_smem.ptr().offset(base), LoadOptions::default());
                let vv = load_vector::<KV, VEC>(&v_smem.ptr().offset(base), LoadOptions::default());
                let mut partial = Float32::from(0.0f32);
                for (qe, ke) in q_vec.iter().zip(KV::widen(&kv)) {
                    partial = partial + *qe * ke;
                }
                let dot = warp_reduction_sum(partial, bdx);
                // FlashInfer's `LogitsTransform`, then the scale to base 2.
                let logit = if variant.logits_soft_cap {
                    cute::math::tanh(dot * logits_pre_scale, true) * sm_scale_log2
                } else if let Some(slope) = slope {
                    let pos = (t_first + t).to::<f32>();
                    (dot * logits_pre_scale + slope * pos) * sm_scale_log2
                } else {
                    dot * sm_scale_log2
                };
                let s = select(t.lt(chunk_len), logit, Float32::from(f32::NEG_INFINITY));
                let m_new = fmax(m.get(), s);
                // `ex2.approx.ftz`, as nvcc emits for FlashInfer: the non-`ftz` form costs
                // extra range checks. (The multiply-adds here fuse into `fma` because the
                // kernels are compiled with contraction, cute-rs's default.)
                let correction = cute::math::exp2(m.get() - m_new, true);
                let p = cute::math::exp2(s - m_new, true);
                d.set(d.get() * correction + p);
                for (a, ve) in acc.iter().zip(KV::widen(&vv)) {
                    a.set(a.get() * correction + p * ve);
                }
                m.set(m_new);
            }
            sync_threads();
        });
        cp_async_wait_group(0);

        // Merge the BDZ partial states of each query head. (For Float16, reusing the K/V
        // stages for these buffers, as FlashInfer does, lets 10 rather than 7 blocks share
        // an SM, but the planner then splits into more chunks and batch 1 measured slower.)
        let o_layout = Layout::col_major(bdz * BDY * head_dim);
        let o_smem = if reuse {
            // Every thread's copies (over-the-end prefetches included) have landed.
            sync_threads();
            make_tensor(&recast_ptr::<f32, KV>(&k_smem.ptr(), None), o_layout)
        } else {
            make_smem_tensor::<f32>(o_layout)
        };
        let md_smem = make_smem_tensor::<f32>(Layout::col_major(bdz * BDY * 2));
        let row = tz * BDY as i32 + ty;
        for (e, a) in acc.iter().enumerate() {
            o_smem.store(row * hd + tx * VEC as i32 + e as i32, a.get());
        }
        if_(tx.eq(0), || {
            md_smem.store(row * 2, m.get());
            md_smem.store(row * 2 + 1, d.get());
        });
        sync_threads();
        if_(tz.eq(0), || {
            let mz: Vec<Float32> = (0..bdz).map(|z| md_smem.load((z as i32 * BDY as i32 + ty) * 2)).collect();
            let m_all = mz.iter().skip(1).fold(mz[0], |x, y| fmax(x, *y));
            let scales: Vec<Float32> = mz.iter().map(|mzz| (*mzz - m_all).exp2()).collect();
            let mut d_all = Float32::from(0.0f32);
            for (z, sc) in scales.iter().enumerate() {
                d_all = d_all + md_smem.load((z as i32 * BDY as i32 + ty) * 2 + 1) * *sc;
            }
            let inv_d = v_scale / d_all;
            let out_lse = m_all + d_all.log2();
            // The output row: the request's, or its chunk's row of the scratch.
            let tmp_row = o_indptr.load_at(b) + chunk;
            let tmp_base = (tmp_row * num_qo_heads + qo_head) * hd + tx * VEC as i32;
            for e in 0..VEC {
                let mut oe = Float32::from(0.0f32);
                for (z, sc) in scales.iter().enumerate() {
                    oe = oe + o_smem.load((z as i32 * BDY as i32 + ty) * hd + tx * VEC as i32 + e as i32) * *sc;
                }
                if partition {
                    tmp_o.store_at(tmp_base + e as i32, oe * inv_d);
                } else {
                    o.store_at(qo_row + e as i32, (oe * inv_d).to::<KV::Out>());
                }
            }
            if_(tx.eq(0), || {
                if partition {
                    tmp_lse.store_at(tmp_row * num_qo_heads + qo_head, out_lse);
                } else {
                    lse.store_at(b * num_qo_heads + qo_head, out_lse);
                }
            });
        });
    })
}

/// Parameter positions of the merge kernel.
pub mod mp {
    pub const TMP_O: usize = 0;
    pub const TMP_LSE: usize = 1;
    pub const O_INDPTR: usize = 2;
    pub const O: usize = 3;
    pub const LSE: usize = 4;
    pub const NUM_QO_HEADS: usize = 5;
}

/// The merge of split-KV states (FlashInfer's variable-length `MergeStates`): grid
/// `(batch, num_qo_heads)`, `HEAD_DIM` threads, one output element each. Request `b`'s
/// states are rows `o_indptr[b] .. o_indptr[b + 1]` of `tmp_o`/`tmp_lse`.
pub fn merge_kernel() -> Kernel {
    merge_kernel_shape(HEAD_DIM, false)
}

/// [`merge_kernel`] for head dimension `head_dim` (`head_dim` threads per block) and, with
/// `bf16`, a BFloat16 output.
pub fn merge_kernel_shape(head_dim: usize, bf16: bool) -> Kernel {
    let name = match (head_dim == HEAD_DIM, bf16) {
        (true, false) => "flashinfer_merge_states".to_string(),
        (_, false) => format!("flashinfer_merge_states_hd{head_dim}"),
        (_, true) => format!("flashinfer_merge_states_hd{head_dim}_bf16"),
    };
    if bf16 { merge_build::<BF16>(&name, head_dim) } else { merge_build::<F16>(&name, head_dim) }
}

fn merge_build<O: Numeric + Copy>(name: &str, head_dim: usize) -> Kernel {
    /// Rows merged per step.
    const MERGE_UNROLL: usize = 8;
    Kernel::builder(name).block_dim([head_dim as u32, 1, 1]).build(|| {
        let tmp_o = param_ptr::<f32>("tmp_o");
        let tmp_lse = param_ptr::<f32>("tmp_lse");
        let o_indptr = param_ptr::<i32>("o_indptr");
        let o = param_ptr::<O>("o");
        let lse = param_ptr::<f32>("lse");
        let num_qo_heads = param_scalar::<i32>("num_qo_heads");
        let (e, b, h) = (thread_idx().x, block_idx().x, block_idx().y);
        let hd = head_dim as i32;
        let m = Var::new(Float32::from(-1.0e30f32));
        let d = Var::new(Float32::from(0.0f32));
        let acc = Var::new(Float32::from(0.0f32));
        let (begin, end) = (o_indptr.load_at(b), o_indptr.load_at(b + 1));
        // MERGE_UNROLL rows per step, their loads issued together so their latencies
        // overlap; rows past the end read the last row and get weight 0.
        let steps = (end - begin + (MERGE_UNROLL as i32 - 1)) / MERGE_UNROLL as i32;
        for_range(0, steps, 1, |step| {
            let rows: Vec<Int32> = (0..MERGE_UNROLL).map(|u| begin + step * MERGE_UNROLL as i32 + u as i32).collect();
            let loads: Vec<(Float32, Float32)> = rows
                .iter()
                .map(|r| {
                    let rc = r.min(end - 1);
                    (tmp_lse.load_at(rc * num_qo_heads + h), tmp_o.load_at((rc * num_qo_heads + h) * hd + e))
                })
                .collect();
            let ss: Vec<Float32> = rows
                .iter()
                .zip(&loads)
                .map(|(r, (s, _))| select(r.lt(end), *s, Float32::from(f32::NEG_INFINITY)))
                .collect();
            let m_new = ss.iter().fold(m.get(), |x, y| fmax(x, *y));
            let correction = (m.get() - m_new).exp2();
            let (mut dn, mut an) = (d.get() * correction, acc.get() * correction);
            for (s, (_, v)) in ss.iter().zip(&loads) {
                let w = (*s - m_new).exp2();
                dn = dn + w;
                an = an + w * *v;
            }
            d.set(dn);
            acc.set(an);
            m.set(m_new);
        });
        o.store_at((b * num_qo_heads + h) * hd + e, (acc.get() / d.get()).to::<O>());
        if_(e.eq(0), || lse.store_at(b * num_qo_heads + h, m.get() + d.get().log2()));
    })
}

/// FlashInfer's decode plan: which chunk of which request each CTA takes.
#[derive(Clone, Debug)]
pub struct Plan {
    pub split: bool,
    pub chunk_pages: usize,
    pub request_indices: Vec<i32>,
    pub kv_tile_indices: Vec<i32>,
    /// Request `b`'s chunk states are rows `o_indptr[b] .. o_indptr[b + 1]`.
    pub o_indptr: Vec<i32>,
}

impl Plan {
    /// Plan for requests of `num_pages` pages each, when at most `max_grid` CTAs run at
    /// once (`blocks per SM * SMs`). As FlashInfer (outside CUDA-graph mode): no split if
    /// `batch * num_kv_heads` fills the GPU; otherwise the smallest chunk of at least
    /// `max(128 / page_size, 1)` pages whose chunks fit in `max_grid`, and no split if
    /// that chunk leaves every request whole.
    pub fn new(num_pages: &[usize], page_size: usize, num_kv_heads: usize, max_grid: usize) -> Plan {
        Self::with_min_chunk(num_pages, num_kv_heads, max_grid, (128 / page_size).max(1))
    }

    /// [`Plan::new`] with another smallest chunk, in pages (tests use 1 to split short
    /// requests).
    pub fn with_min_chunk(num_pages: &[usize], num_kv_heads: usize, max_grid: usize, min_chunk_pages: usize) -> Plan {
        let batch = num_pages.len();
        let max_pages = num_pages.iter().copied().max().unwrap_or(1).max(1);
        let chunks = |c: usize| num_pages.iter().map(|&n| n.max(1).div_ceil(c)).sum::<usize>();
        let chunk_pages = if batch * num_kv_heads >= max_grid {
            max_pages
        } else {
            let (mut low, mut high) = (min_chunk_pages.max(1), max_pages);
            while low < high {
                let mid = (low + high) / 2;
                if chunks(mid) * num_kv_heads > max_grid {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low.min(max_pages)
        };
        let split = chunks(chunk_pages) > batch;
        let chunk_pages = if split { chunk_pages } else { max_pages };
        let (mut request_indices, mut kv_tile_indices, mut o_indptr) = (vec![], vec![], vec![0i32]);
        for (b, &n) in num_pages.iter().enumerate() {
            let c = n.max(1).div_ceil(chunk_pages);
            for t in 0..c {
                request_indices.push(b as i32);
                kv_tile_indices.push(t as i32);
            }
            o_indptr.push(o_indptr[b] + c as i32);
        }
        Plan { split, chunk_pages, request_indices, kv_tile_indices, o_indptr }
    }

    /// The plan of FlashInfer's single-request decode (`SingleDecodeWithKVCacheDispatched`)
    /// for `kv_len` tokens: whole up to 256 tokens, else chunks of `max(ceil(kv_len /
    /// (max_grid / num_kv_heads)), 256)` tokens (split when there is more than one).
    /// Chunks are counted in "pages" of one token (a ragged problem's).
    pub fn single(kv_len: usize, num_kv_heads: usize, max_grid: usize) -> Plan {
        let max_chunks = (max_grid / num_kv_heads).max(1);
        let chunk = if kv_len <= 256 { kv_len } else { kv_len.div_ceil(max_chunks).max(256) };
        let n = kv_len.div_ceil(chunk);
        Plan {
            split: n > 1,
            chunk_pages: chunk,
            request_indices: vec![0; n],
            kv_tile_indices: (0..n as i32).collect(),
            o_indptr: vec![0, n as i32],
        }
    }

    pub fn num_ctas(&self) -> usize {
        self.request_indices.len()
    }
}

/// A test problem: per-request token counts, a page size and a shuffled page table; or,
/// with `page_size` [`RAGGED`], contiguous K/V with `kv_indptr` in tokens.
pub struct Problem {
    pub kv_lens: Vec<usize>,
    pub page_size: usize,
    pub num_kv_heads: usize,
    /// The head dimension (64, 128 or 256).
    pub head_dim: usize,
    /// BFloat16 queries, cache and output ([`Problem::bf16`]). The host vectors below then
    /// hold the BFloat16 values (exact in Float16), uploaded as BFloat16.
    pub bf16: bool,
    pub q: Vec<F16>,
    pub k_cache: Vec<F16>,
    pub v_cache: Vec<F16>,
    pub kv_indptr: Vec<i32>,
    pub kv_indices: Vec<i32>,
    pub kv_last_page_len: Vec<i32>,
    pub sm_scale: f32,
    /// The cache's element type. For FP8, `k_cache`/`v_cache` hold the dequantised values
    /// (exact in Float16, before `k_scale`/`v_scale`), and `k_fp8`/`v_fp8` the stored bytes.
    pub kv_dtype: KvDtype,
    pub k_fp8: Vec<u8>,
    pub v_fp8: Vec<u8>,
    /// FlashInfer's `k_scale`, `v_scale`: K is `k_scale * k_cache`, V `v_scale * v_cache`.
    pub k_scale: f32,
    pub v_scale: f32,
    /// The attention variant the kernel is built for, and its run-time values: FlashInfer's
    /// `logits_soft_cap`, `window_left` (negative: none) and ALiBi slopes per query head.
    pub variant: Variant,
    pub logits_soft_cap: f32,
    pub window_left: i32,
    pub alibi_slopes: Vec<f32>,
}

impl Problem {
    pub fn new(kv_lens: &[usize], page_size: usize, num_kv_heads: usize, seed: u64) -> Problem {
        Self::new_hd(kv_lens, page_size, num_kv_heads, HEAD_DIM, seed)
    }

    /// [`Problem::new`] for head dimension `head_dim` (scale `1/sqrt(head_dim)`).
    pub fn new_hd(kv_lens: &[usize], page_size: usize, num_kv_heads: usize, head_dim: usize, seed: u64) -> Problem {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let batch = kv_lens.len();
        let pages_per: Vec<usize> = kv_lens.iter().map(|l| l.div_ceil(page_size)).collect();
        let num_pages: usize = pages_per.iter().sum::<usize>() + 2; // two unused pages
        // A shuffled page table (a fixed permutation of the page ids).
        let mut ids: Vec<i32> = (0..num_pages as i32).collect();
        for i in (1..ids.len()).rev() {
            let j = ((rnd() + 1.0) * 0.5 * (i + 1) as f32) as usize % (i + 1);
            ids.swap(i, j);
        }
        let mut kv_indptr = vec![0i32];
        let mut kv_indices = vec![];
        let mut next = 0;
        for &p in &pages_per {
            kv_indices.extend_from_slice(&ids[next..next + p]);
            next += p;
            kv_indptr.push(next as i32);
        }
        let kv_last_page_len = kv_lens.iter().map(|&l| (l - (l.div_ceil(page_size) - 1) * page_size) as i32).collect();
        let n_qo = num_kv_heads * GROUP;
        let cache = num_pages * page_size * num_kv_heads * head_dim;
        let mut fill = |n: usize, scale: f32| (0..n).map(|_| F16::from_f32(rnd() * scale)).collect::<Vec<_>>();
        Problem {
            kv_lens: kv_lens.to_vec(),
            page_size,
            num_kv_heads,
            head_dim,
            bf16: false,
            q: fill(batch * n_qo * head_dim, 1.0),
            k_cache: fill(cache, 1.0),
            v_cache: fill(cache, 1.0),
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            kv_dtype: KvDtype::F16,
            k_fp8: vec![],
            v_fp8: vec![],
            k_scale: 1.0,
            v_scale: 1.0,
            variant: Variant::NONE,
            logits_soft_cap: 0.0,
            window_left: -1,
            alibi_slopes: vec![],
        }
    }

    /// This problem with BFloat16 queries, cache and output: the data rounded to BFloat16
    /// (to nearest even; the reference then uses the rounded values), the kernels built
    /// for BFloat16 ([`kernel_shape`]).
    pub fn bf16(mut self) -> Problem {
        assert_eq!(self.kv_dtype, KvDtype::F16, "BFloat16 has no FP8 cache");
        // Rounding a Float16 value to 8 significant bits stays exact in Float16.
        let round = |v: &mut Vec<F16>| {
            for x in v.iter_mut() {
                *x = F16::from_f32(BF16::from_f32(x.to_f32()).to_f32());
            }
        };
        for v in [&mut self.q, &mut self.k_cache, &mut self.v_cache] {
            round(v);
        }
        self.bf16 = true;
        self
    }

    /// This problem with its K/V cache quantised to `kv` with scales `k_scale`,
    /// `v_scale` (each element `x` stored as `x / scale` rounded to `kv`), as a caller
    /// quantises a cache for FlashInfer's FP8 decode. The reference then uses the
    /// dequantised cache.
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

    /// A problem over contiguous K/V (`[sum(kv_lens), num_kv_heads, HEAD_DIM]`) with random
    /// contents.
    pub fn new_ragged(kv_lens: &[usize], num_kv_heads: usize, seed: u64) -> Problem {
        Self::new_ragged_hd(kv_lens, num_kv_heads, HEAD_DIM, seed)
    }

    /// [`Problem::new_ragged`] for head dimension `head_dim`.
    pub fn new_ragged_hd(kv_lens: &[usize], num_kv_heads: usize, head_dim: usize, seed: u64) -> Problem {
        // A one-token-page problem, its pages gathered into token order.
        let p = Problem::new_hd(kv_lens, 1, num_kv_heads, head_dim, seed);
        let row = num_kv_heads * head_dim;
        let gather = |cache: &[F16]| -> Vec<F16> {
            p.kv_indices
                .iter()
                .flat_map(|&pg| cache[pg as usize * row..(pg as usize + 1) * row].iter().copied())
                .collect()
        };
        let (k, v) = (gather(&p.k_cache), gather(&p.v_cache));
        Problem::from_ragged(kv_lens, num_kv_heads, p.q, k, v)
    }

    /// A problem on given tensors: `q [batch, num_kv_heads * GROUP, HEAD_DIM]`, contiguous
    /// `k`, `v [sum(kv_lens), num_kv_heads, HEAD_DIM]`, scale `1/sqrt(HEAD_DIM)`. The head
    /// dimension is `q.len() / (batch * num_kv_heads * GROUP)`.
    pub fn from_ragged(kv_lens: &[usize], num_kv_heads: usize, q: Vec<F16>, k: Vec<F16>, v: Vec<F16>) -> Problem {
        assert!(kv_lens.iter().all(|&l| l > 0), "every request needs a token");
        let kv_indptr: Vec<i32> = std::iter::once(0)
            .chain(kv_lens.iter().scan(0i32, |a, &l| {
                *a += l as i32;
                Some(*a)
            }))
            .collect();
        let head_dim = q.len() / (kv_lens.len() * num_kv_heads * GROUP);
        assert!(matches!(head_dim, 64 | 128 | 256), "head dim {head_dim} not supported");
        assert_eq!(q.len(), kv_lens.len() * num_kv_heads * GROUP * head_dim);
        assert_eq!(k.len(), *kv_indptr.last().unwrap() as usize * num_kv_heads * head_dim);
        assert_eq!(v.len(), k.len());
        Problem {
            kv_lens: kv_lens.to_vec(),
            page_size: RAGGED,
            num_kv_heads,
            head_dim,
            bf16: false,
            q,
            k_cache: k,
            v_cache: v,
            kv_indptr,
            kv_indices: vec![0],
            kv_last_page_len: vec![1; kv_lens.len()],
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            kv_dtype: KvDtype::F16,
            k_fp8: vec![],
            v_fp8: vec![],
            k_scale: 1.0,
            v_scale: 1.0,
            variant: Variant::NONE,
            logits_soft_cap: 0.0,
            window_left: -1,
            alibi_slopes: vec![],
        }
    }

    pub fn is_ragged(&self) -> bool {
        self.page_size == RAGGED
    }

    /// This problem with FlashInfer's logits soft-cap `cap` (Gemma 2).
    pub fn with_soft_cap(mut self, cap: f32) -> Problem {
        (self.variant.logits_soft_cap, self.logits_soft_cap) = (true, cap);
        self
    }

    /// This problem with FlashInfer's sliding window `window_left` (a query sees its last
    /// `window_left + 1` tokens).
    pub fn with_window(mut self, window_left: i32) -> Problem {
        (self.variant.sliding_window, self.window_left) = (true, window_left);
        self
    }

    /// This problem with ALiBi, FlashInfer's slopes ([`alibi_slopes`]).
    pub fn with_alibi(mut self) -> Problem {
        (self.variant.alibi, self.alibi_slopes) = (true, alibi_slopes(self.num_qo_heads()));
        self
    }

    /// The tokens request `b`'s query attends to: `(first, count)` (the last
    /// `window_left + 1` with a window).
    pub fn visible(&self, b: usize) -> (usize, usize) {
        let len = self.kv_lens[b];
        let win = if self.variant.sliding_window && self.window_left >= 0 {
            len.min(self.window_left as usize + 1)
        } else {
            len
        };
        (len - win, win)
    }

    /// Tokens per page (1 for a ragged problem).
    pub fn page_tokens(&self) -> usize {
        self.page_size.max(1)
    }

    pub fn num_qo_heads(&self) -> usize {
        self.num_kv_heads * GROUP
    }

    pub fn plan(&self, max_grid: usize) -> Plan {
        self.plan_with_min_chunk(max_grid, (128 / self.page_tokens()).max(1))
    }

    /// [`Plan::single`] for a one-request ragged problem, over its visible tokens (with a
    /// window, only the window's: the kernel starts at the window's first token, so chunks
    /// planned past the window would be empty).
    pub fn single_plan(&self, max_grid: usize) -> Plan {
        assert!(self.is_ragged() && self.kv_lens.len() == 1, "a single-request plan needs one ragged request");
        Plan::single(self.visible(0).1, self.num_kv_heads, max_grid)
    }

    /// The decode kernel for this problem's layout.
    pub fn kernel(&self, partition: bool) -> Kernel {
        kernel_shape(partition, STAGES, self.is_ragged(), self.kv_dtype, self.variant, self.head_dim, self.bf16)
    }

    /// The plan over each request's visible pages (with a window, only the window's, as
    /// the kernel starts at the window's first token).
    pub fn plan_with_min_chunk(&self, max_grid: usize, min_chunk_pages: usize) -> Plan {
        let pages: Vec<usize> =
            (0..self.kv_lens.len()).map(|b| self.visible(b).1.div_ceil(self.page_tokens())).collect();
        Plan::with_min_chunk(&pages, self.num_kv_heads, max_grid, min_chunk_pages)
    }

    pub fn launch(&self, plan: &Plan) -> LaunchConfig {
        LaunchConfig::new([plan.num_ctas() as u32, self.num_kv_heads as u32, 1], self.kv_dtype.block_hd(self.head_dim))
    }

    pub fn merge_launch(&self) -> LaunchConfig {
        LaunchConfig::new([self.kv_lens.len() as u32, self.num_qo_heads() as u32, 1], [self.head_dim as u32, 1, 1])
    }

    /// Rows of split-KV scratch the plan needs (at least one, so buffers are never empty).
    pub fn tmp_rows(plan: &Plan) -> usize {
        if plan.split { plan.num_ctas() } else { 1 }
    }

    /// The decode kernel's arguments, in the order of [`p`].
    pub fn args(&self, plan: &Plan) -> Vec<KernelArg> {
        let n_o = self.q.len();
        let rows = Self::tmp_rows(plan);
        let hd = self.head_dim;
        vec![
            self.half_arg(&self.q),
            self.cache_arg(&self.k_cache, &self.k_fp8),
            self.cache_arg(&self.v_cache, &self.v_fp8),
            KernelArg::i32(self.kv_indptr.clone()),
            KernelArg::i32(self.kv_indices.clone()),
            KernelArg::i32(self.kv_last_page_len.clone()),
            KernelArg::i32(plan.request_indices.clone()),
            KernelArg::i32(plan.kv_tile_indices.clone()),
            KernelArg::i32(plan.o_indptr.clone()),
            self.half_arg(&vec![F16::from_f32(0.0); n_o]),
            KernelArg::f32(vec![0.0; n_o / hd]),
            KernelArg::f32(vec![0.0; rows * self.num_qo_heads() * hd]),
            KernelArg::f32(vec![0.0; rows * self.num_qo_heads()]),
            KernelArg::ScalarI32(self.page_size as i32),
            KernelArg::ScalarI32(self.num_kv_heads as i32),
            KernelArg::ScalarF32(self.variant.scales(self.sm_scale * self.k_scale, self.logits_soft_cap).0),
            KernelArg::ScalarI32((plan.chunk_pages * self.page_tokens()) as i32),
            KernelArg::ScalarF32(self.v_scale),
            KernelArg::ScalarF32(self.variant.scales(self.sm_scale * self.k_scale, self.logits_soft_cap).1),
            KernelArg::ScalarI32(self.window_left),
            KernelArg::f32(if self.alibi_slopes.is_empty() { vec![0.0] } else { self.alibi_slopes.clone() }),
        ]
    }

    /// A Float16 (or, for a BFloat16 problem, BFloat16) buffer argument of the host values.
    fn half_arg(&self, v: &[F16]) -> KernelArg {
        if self.bf16 {
            KernelArg::buffer(&v.iter().map(|x| BF16::from_f32(x.to_f32())).collect::<Vec<_>>())
        } else {
            KernelArg::buffer(v)
        }
    }

    /// A cache argument of the problem's element type.
    fn cache_arg(&self, f16: &[F16], fp8: &[u8]) -> KernelArg {
        match self.kv_dtype {
            KvDtype::F16 => self.half_arg(f16),
            KvDtype::E4M3 => KernelArg::buffer(&fp8.iter().map(|&b| F8E4M3FN(b)).collect::<Vec<_>>()),
            KvDtype::E5M2 => KernelArg::buffer(&fp8.iter().map(|&b| F8E5M2(b)).collect::<Vec<_>>()),
        }
    }

    /// Attention in f64 on the f16 inputs: `o` and base-2 `lse`, like FlashInfer's
    /// PyTorch reference (softmax(q k^T * sm_scale) v over the gathered pages). For an
    /// FP8 cache, on the dequantised cache: `k_scale * k_cache`, `v_scale * v_cache`.
    pub fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        let (hkv, n_qo, ps) = (self.num_kv_heads, self.num_qo_heads(), self.page_tokens());
        let f = |x: F16| x.to_f32() as f64;
        let mut o = vec![0f32; self.q.len()];
        let hd = self.head_dim;
        let mut lse = vec![0f32; self.q.len() / hd];
        let (sm, cap) = (self.sm_scale as f64 * self.k_scale as f64, self.logits_soft_cap as f64);
        for b in 0..self.kv_lens.len() {
            let (first, n) = self.visible(b);
            let addr = |t: usize, h: usize| {
                let start = self.kv_indptr[b] as usize;
                let row =
                    if self.is_ragged() { start + t } else { self.kv_indices[start + t / ps] as usize * ps + t % ps };
                (row * hkv + h) * hd
            };
            for qh in 0..n_qo {
                let h = qh / GROUP;
                let qrow = (b * n_qo + qh) * hd;
                let slope = self.alibi_slopes.get(qh).map_or(0.0, |&x| x as f64);
                let s: Vec<f64> = (first..first + n)
                    .map(|t| {
                        let a = addr(t, h);
                        let qk = (0..hd).map(|e| f(self.q[qrow + e]) * f(self.k_cache[a + e])).sum::<f64>();
                        self.variant.ref_logit(qk, sm, cap, slope, 0, t)
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let sum: f64 = w.iter().sum();
                lse[b * n_qo + qh] = ((mx + sum.ln()) * std::f64::consts::LOG2_E) as f32;
                for e in 0..hd {
                    let acc: f64 = (0..n).map(|i| w[i] * f(self.v_cache[addr(first + i, h) + e])).sum();
                    o[qrow + e] = (acc / sum * self.v_scale as f64) as f32;
                }
            }
        }
        (o, lse)
    }
}

pub use crate::check::max_error;

/// Simulate `p` under a plan for `max_grid` concurrent CTAs (the decode kernel, then the
/// merge if the plan splits) and return the plan and `(max |o - ref|, max |lse - ref|)`.
pub fn simulate(p: &Problem, max_grid: usize, min_chunk_pages: usize) -> (Plan, f32, f32) {
    let plan = p.plan_with_min_chunk(max_grid, min_chunk_pages);
    let (eo, el) = simulate_plan(p, &plan);
    (plan, eo, el)
}

/// [`execute`] checked against the f64 reference: `(max |o - ref|, max |lse - ref|)`.
pub fn simulate_plan(p: &Problem, plan: &Plan) -> (f32, f32) {
    let (o, lse) = execute(p, plan);
    let (want_o, want_lse) = p.reference();
    (max_error(&o, &want_o), max_error(&lse, &want_lse))
}

/// FlashInfer's `single_decode_with_kv_cache` (NHD) on the simulator: `q [num_qo_heads,
/// HEAD_DIM]` (`num_qo_heads = GROUP * num_kv_heads`), contiguous `k`, `v [kv_len,
/// num_kv_heads, HEAD_DIM]`, the ragged kernel under [`Plan::single`] for `max_grid`
/// concurrent CTAs. Returns `o` (as f32) and the base-2 `lse [num_qo_heads]`.
pub fn single_decode_with_kv_cache(
    q: Vec<F16>,
    k: Vec<F16>,
    v: Vec<F16>,
    num_kv_heads: usize,
    max_grid: usize,
) -> (Vec<f32>, Vec<f32>) {
    let kv_len = k.len() / (num_kv_heads * HEAD_DIM);
    let p = Problem::from_ragged(&[kv_len], num_kv_heads, q, k, v);
    execute(&p, &p.single_plan(max_grid))
}

/// Run `p` under `plan` on the simulator (the decode kernel, then the merge if the plan
/// splits); returns `o` (as f32) and `lse`.
pub fn execute(p: &Problem, plan: &Plan) -> (Vec<f32>, Vec<f32>) {
    let mut args = p.args(plan);
    let k = p.kernel(plan.split);
    k.simulate(p.launch(plan), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    if plan.split {
        let mut margs = vec![
            args[p::TMP_O].clone(),
            args[p::TMP_LSE].clone(),
            args[p::O_INDPTR].clone(),
            args[p::O].clone(),
            args[p::LSE].clone(),
            KernelArg::ScalarI32(p.num_qo_heads() as i32),
        ];
        merge_kernel_shape(p.head_dim, p.bf16)
            .simulate(p.merge_launch(), &mut margs)
            .unwrap_or_else(|e| panic!("merge: {e}"));
        args[p::O] = margs[mp::O].clone();
        args[p::LSE] = margs[mp::LSE].clone();
    }
    let o: Vec<f32> = if p.bf16 {
        args[p::O].to_vec::<BF16>().into_iter().map(|x| x.to_f32()).collect()
    } else {
        args[p::O].to_vec::<F16>().into_iter().map(|x| x.to_f32()).collect()
    };
    (o, args[p::LSE].as_f32().to_vec())
}

/// Running on the GPU (`--features cuda`): the kernels compiled and loaded, and one problem
/// run, checked against the f64 reference and optionally timed.
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg, HardwareInfo};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{KvDtype, Plan, Problem, Variant, kernel_shape, max_error, merge_kernel_shape};

    /// The three kernels, compiled and loaded once, the decode kernels with `stages`
    /// shared-memory stages, for paged or (`ragged`) contiguous K/V.
    pub struct Kernels<'c> {
        pub ragged: bool,
        /// The K/V cache element type the decode kernels read.
        pub kv: KvDtype,
        /// The attention variant the decode kernels are built for.
        pub variant: Variant,
        /// The head dimension, and whether queries, cache and output are BFloat16.
        pub head_dim: usize,
        pub bf16: bool,
        pub decode: (Kernel, CudaFunction<'c>),
        pub split: (Kernel, CudaFunction<'c>),
        pub merge: (Kernel, CudaFunction<'c>),
        /// CTAs of the decode kernel that run at once: blocks per SM times SMs.
        pub max_grid: usize,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &Toolchain, stages: usize) -> Result<Kernels<'c>, Box<dyn Error>> {
            Self::with_layout(ctx, tc, stages, false)
        }

        /// The kernels for contiguous K/V (single decode).
        pub fn new_ragged(ctx: &'c CudaContext, tc: &Toolchain) -> Result<Kernels<'c>, Box<dyn Error>> {
            Self::with_layout(ctx, tc, super::STAGES, true)
        }

        pub fn with_layout(
            ctx: &'c CudaContext,
            tc: &Toolchain,
            stages: usize,
            ragged: bool,
        ) -> Result<Kernels<'c>, Box<dyn Error>> {
            Self::with_kv(ctx, tc, stages, ragged, KvDtype::F16)
        }

        /// [`Kernels::with_layout`] for a K/V cache of `kv` elements.
        pub fn with_kv(
            ctx: &'c CudaContext,
            tc: &Toolchain,
            stages: usize,
            ragged: bool,
            kv: KvDtype,
        ) -> Result<Kernels<'c>, Box<dyn Error>> {
            Self::with_variant(ctx, tc, stages, ragged, kv, Variant::NONE)
        }

        /// [`Kernels::with_kv`] built for an attention [`Variant`].
        pub fn with_variant(
            ctx: &'c CudaContext,
            tc: &Toolchain,
            stages: usize,
            ragged: bool,
            kv: KvDtype,
            variant: Variant,
        ) -> Result<Kernels<'c>, Box<dyn Error>> {
            Self::with_shape(ctx, tc, stages, ragged, kv, variant, super::HEAD_DIM, false)
        }

        /// [`Kernels::with_variant`] for head dimension `head_dim` and, with `bf16`,
        /// BFloat16 queries, cache and output ([`kernel_shape`]).
        #[allow(clippy::too_many_arguments)]
        pub fn with_shape(
            ctx: &'c CudaContext,
            tc: &Toolchain,
            stages: usize,
            ragged: bool,
            kv: KvDtype,
            variant: Variant,
            head_dim: usize,
            bf16: bool,
        ) -> Result<Kernels<'c>, Box<dyn Error>> {
            let arch = ctx.best_target(tc).ok_or("no target for this GPU")?;
            // Contraction (on by default) fuses the softmax's multiply-adds into `fma`, as
            // nvcc does for FlashInfer; without it the main loop runs about a fifth more
            // instructions.
            let opts = CompileOptions::new().gpu_arch(arch);
            let load = |k: Kernel| -> Result<(Kernel, CudaFunction<'c>), Box<dyn Error>> {
                let f = ctx.load(&k.compile_with(tc, &opts)?)?;
                Ok((k, f))
            };
            let (decode, split, merge) = (
                load(kernel_shape(false, stages, ragged, kv, variant, head_dim, bf16))?,
                load(kernel_shape(true, stages, ragged, kv, variant, head_dim, bf16))?,
                load(merge_kernel_shape(head_dim, bf16))?,
            );
            let threads = kv.block_hd(head_dim).iter().product::<u32>();
            let per_sm = split.1.max_active_blocks_per_multiprocessor(threads, 0)? as usize;
            let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            Ok(Kernels { ragged, kv, variant, head_dim, bf16, decode, split, merge, max_grid: per_sm * sms })
        }
    }

    /// Run `p` once on the GPU; if `time`, also benchmark the decode (and merge) launches.
    /// Returns the plan, the errors and the average time in microseconds.
    #[allow(clippy::type_complexity)]
    pub fn run(
        ctx: &CudaContext,
        ks: &Kernels<'_>,
        p: &Problem,
        check: bool,
        time: bool,
    ) -> Result<(Plan, f32, f32, Option<(f64, f64, f64)>), Box<dyn Error>> {
        let plan = p.plan(ks.max_grid);
        let (out, t) = launch(ctx, ks, p, &plan, check, time)?;
        let (mut eo, mut el) = (0.0, 0.0);
        if let Some((o, lse)) = out {
            let (want_o, want_lse) = p.reference();
            (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
        }
        Ok((plan, eo, el, t))
    }

    /// FlashInfer's `single_decode_with_kv_cache` (NHD) on the GPU: the ragged kernels of
    /// `ks` under [`Plan::single`] on `q [num_kv_heads * GROUP, HEAD_DIM]` and contiguous
    /// `k`, `v [kv_len, num_kv_heads, HEAD_DIM]`. Returns `o` (as f32) and the base-2 `lse`.
    #[allow(clippy::type_complexity)]
    pub fn single_decode_with_kv_cache(
        ctx: &CudaContext,
        ks: &Kernels<'_>,
        q: Vec<F16>,
        k: Vec<F16>,
        v: Vec<F16>,
        num_kv_heads: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        let kv_len = k.len() / (num_kv_heads * super::HEAD_DIM);
        let p = Problem::from_ragged(&[kv_len], num_kv_heads, q, k, v);
        let (out, _) = launch(ctx, ks, &p, &p.single_plan(ks.max_grid), true, false)?;
        Ok(out.unwrap())
    }

    /// Upload `p`, run it once under `plan`, download `o` and `lse` if `fetch`, and, if
    /// `time`, time the launches back to back, from a CUDA graph, and the decode kernel
    /// alone from a graph (microseconds).
    #[allow(clippy::type_complexity)]
    pub fn launch(
        ctx: &CudaContext,
        ks: &Kernels<'_>,
        p: &Problem,
        plan: &Plan,
        fetch: bool,
        time: bool,
    ) -> Result<(Option<(Vec<f32>, Vec<f32>)>, Option<(f64, f64, f64)>), Box<dyn Error>> {
        if ks.ragged != p.is_ragged() {
            return Err("the kernels' K/V layout (paged or ragged) differs from the problem's".into());
        }
        if ks.kv != p.kv_dtype {
            return Err("the kernels' K/V element type differs from the problem's".into());
        }
        if (ks.head_dim, ks.bf16) != (p.head_dim, p.bf16) {
            return Err("the kernels' head dimension or Q/O type differs from the problem's".into());
        }
        if ks.variant != p.variant {
            return Err("the kernels' attention variant differs from the problem's".into());
        }
        let host = p.args(plan);
        // Upload every buffer argument; scalars pass through.
        let mut bufs = Vec::new();
        for a in &host {
            if let Some(b) = match a {
                KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => None,
                _ => Some(ctx.upload(a)?),
            } {
                bufs.push(b);
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
        let margs = vec![
            DeviceArg::Buffer(&bufs[super::p::TMP_O]),
            DeviceArg::Buffer(&bufs[super::p::TMP_LSE]),
            DeviceArg::Buffer(&bufs[super::p::O_INDPTR]),
            DeviceArg::Buffer(&bufs[super::p::O]),
            DeviceArg::Buffer(&bufs[super::p::LSE]),
            DeviceArg::Scalar(KernelArg::ScalarI32(p.num_qo_heads() as i32)),
        ];
        let (dk, df) = if plan.split { (&ks.split.0, &ks.split.1) } else { (&ks.decode.0, &ks.decode.1) };
        let launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(dk, df, p.launch(plan)).stream(s).args(&args).submit()?;
            if plan.split {
                ctx.launch_builder(&ks.merge.0, &ks.merge.1, p.merge_launch()).stream(s).args(&margs).submit()?;
            }
            Ok(())
        };
        let stream = ctx.stream()?;
        let mut launch = launch;
        launch(&stream, 0)?;
        stream.synchronize()?;
        let out = if fetch {
            let o: Vec<f32> = if p.bf16 {
                bufs[super::p::O].to_vec::<u16>()?.into_iter().map(|x| BF16(x).to_f32()).collect()
            } else {
                bufs[super::p::O].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect()
            };
            Some((o, bufs[super::p::LSE].to_vec::<f32>()?))
        } else {
            None
        };
        // Two timings: launches issued back to back (what a caller sees, host launch cost
        // included when it exceeds the GPU time), and the same launches replayed from a
        // CUDA graph (GPU time alone).
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100);
            let direct = benchmark(&stream, &opts, &mut launch)?;
            let graph = benchmark(&stream, &opts.clone().use_cuda_graphs(true), &mut launch)?;
            let decode_only = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
                ctx.launch_builder(dk, df, p.launch(plan)).stream(s).args(&args).submit()?;
                Ok(())
            };
            let decode = benchmark(&stream, &opts.use_cuda_graphs(true), decode_only)?;
            Some((direct, graph, decode))
        } else {
            None
        };
        Ok((out, t))
    }
}
