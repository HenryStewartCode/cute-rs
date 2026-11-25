// SPDX-License-Identifier: Apache-2.0

//! XQA decode attention (`flashinfer.xqa`): TensorRT-LLM's GQA decode kernel as FlashInfer
//! ships it (`csrc/xqa/mha.cu`, `csrc/xqa/xqa_wrapper.cu`, Apache-2.0, NVIDIA), its SM80 /
//! SM89 path. All query heads of a KV head's group are processed together as the rows of
//! one 16-row tensor-core tile, over a paged K/V cache that may be FP8 (E4M3) with a scale.
//! This is cute-rs's own code: it follows mha.cu's published design (grid, CTA tile,
//! per-warp token slices, multi-block split with a semaphore and an in-kernel merge by the
//! last CTA, the scale and sink semantics), but no XQA source was translated.
//!
//! * **Grid** `(nb_sub_seq, num_kv_heads, batch)`, as mha.cu's `launchMHA`, with XQA's
//!   multi-block rule `nb_sub_seq = min(max(1, sm_count / (batch * num_kv_heads)),
//!   ceil(max_seq_len / 256))` ([`nb_sub_seq`]).
//! * **CTA tile** (mha.cu's `ctaTile`): 256 tokens, four warps of 64 (`warpTile.x`). CTA
//!   `sub` of a sequence takes the CTA tiles `sub`, `sub + nb_sub_seq`, ... (mha.cu's
//!   `seqIterInit` / `seqStrideIters`).
//! * **Warps.** mha.cu specialises warps: four run `Q K^T` and the softmax, four `P V`, and
//!   `P` passes through shared memory under barriers. Here each of the four warps runs
//!   both on its own 64-token slice, in steps of 16 tokens through a private two-stage
//!   `cp.async` ring, so the main loop has no CTA barrier at all (only `bar.warp.sync`),
//!   and `P` never leaves registers (the accumulator layout of `Q K^T` is the A layout of
//!   `P V`). The warps' states merge through shared memory at the end.
//! * **MMA**: `mma.sync.m16n8k16` with f32 accumulators, f16 or bf16 operands (the input
//!   type); row `r < group` of the 16-row tile is query head `kv_head * group + r`, the
//!   rest are padding (with `group <= 8` the upper eight rows are never computed past the
//!   MMA). K is read from shared memory with plain 16-byte loads: the head dimension is
//!   permuted within each lane's 16-byte chunk (a lane's chunk holds MMA k-slots `2 t`,
//!   `2 t + 1`, `2 t + 8`, `2 t + 9` of consecutive k-steps), and Q is loaded once, from
//!   global memory into registers, in the same order, so `Q K^T` is unchanged. V is read
//!   with `ldmatrix.x4.trans`.
//! * **FP8 K/V** (E4M3): staged as bytes, widened in registers with the hardware
//!   `cvt.rn.f16x2.e4m3x2` (then to bf16 for a bf16 query, exactly), as mha.cu converts the
//!   cache to the input type before its MMAs. For V the transposed `ldmatrix` over bytes
//!   gives a lane keys `2 t`, `2 t + 1` of elements `2 g`, `2 g + 1`, so the two n8 tiles of
//!   a 16-element slice take the even and the odd elements.
//! * **Scales** (mha.cu's): `qk_scale = q_scale * kv_scale / sqrt(HEAD_DIM)` on `Q K^T`
//!   (`kv_scale` only for an FP8 cache), `vo_scale = kv_scale` on the output (likewise).
//!   The softmax is base 2 (`ex2.approx.ftz`).
//! * **Attention sinks** (optional, build-time): one f32 logit per query head, added to
//!   the softmax denominator as `exp(sink - max)` (mha.cu's `addAttentionSinks`).
//! * **Sliding window** (optional, build-time; mha.cu's `SLIDING_WINDOW`): a sequence of
//!   `seq_len > window` tokens attends to its last `window` tokens; whole CTA tiles before
//!   the window are skipped, the rest masked.
//! * **Multi-block**: with `nb_sub_seq > 1` each CTA writes its normalised partial output
//!   and `(max, sum)` to f32 scratch, then counts itself in on a per-(request, KV head)
//!   semaphore; the last CTA merges the partials and resets the semaphore to zero (mha.cu's
//!   scheme), so one launch does all the work and the semaphores need zeroing only once.
//!
//! Layouts (FlashInfer's `xqa()` with `kv_layout="NHD"`): `q`, `out [batch, num_kv_heads *
//! group, HEAD_DIM]`; `k_cache`, `v_cache [num_pages, page_size, num_kv_heads, HEAD_DIM]`;
//! `page_table [batch, max_pages_per_seq]` (i32); `seq_lens [batch]`.
//!
//! Covered: f16 / bf16 queries and output; a K/V cache of the query's type or E4M3 with
//! `kv_scale`; `q_scale`; page sizes 16, 32, 64 and 128 (FlashInfer's); group sizes 1-16;
//! head dim 128; sliding window; attention sinks.
//!
//! Not covered: the SM90 / SM120 kernels (TMA, GMMA, MLA), beam search, speculative decoding
//! (`q_seq_len > 1`, masks), int8 and NVFP4 caches, FP8 output (`rcp_out_scale`), the HND
//! layout, device-side scale tensors, groups above 16, PDL.
//!
//! Timings (RTX 2000 Ada, CUDA graphs, 8 KV heads, page 16, f16, group 4): batch 1 kv 1K/4K/16K
//! 9.3/24.7/321 us (FlashInfer's XQA built for sm_89: 11.7/28.6/323), batch 16 322/1273/5089
//! (324/1282/5164), batch 64 kv 1K 1281 (1287); about 210 GB/s when bandwidth-bound.
//!
//! `examples/xqa.rs` runs it on the simulator and, with `--features cuda` and `GPU=1`, on
//! the GPU against the reference, then benchmarks it; `examples/xqa_flashinfer_bench.py`
//! times FlashInfer's own XQA (built for sm_89) on the same shapes.

use cute::prelude::*;

/// Head dimension.
pub const HEAD_DIM: usize = 128;
/// Warps per CTA.
pub const WARPS: usize = 4;
/// Tokens of a warp's slice of a CTA tile (mha.cu's `warpTile.x`).
pub const WARP_TILE: usize = 64;
/// Tokens per CTA tile (mha.cu's `ctaTile.x`).
pub const CTA_TILE: usize = WARPS * WARP_TILE;
/// Tokens per step of a warp's pipeline.
pub const STEP: usize = 16;
/// The largest group (query heads per KV head): one 16-row MMA tile.
pub const MAX_GROUP: usize = 16;
/// Elements a vector view may address.
const BOUND: usize = 1 << 30;

/// The query (and output) element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputDtype {
    F16,
    BF16,
}

impl InputDtype {
    fn name(self) -> &'static str {
        match self {
            InputDtype::F16 => "f16",
            InputDtype::BF16 => "bf16",
        }
    }
    /// `x` rounded to this type.
    pub fn round(self, x: f32) -> f32 {
        match self {
            InputDtype::F16 => F16::from_f32(x).to_f32(),
            InputDtype::BF16 => BF16::from_f32(x).to_f32(),
        }
    }
}

/// The K/V cache element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheDtype {
    /// The query's type.
    Input,
    /// `torch.float8_e4m3fn`, scaled by `kv_scale`.
    E4M3,
}

/// What a kernel is built for (FlashInfer's JIT flags: `DTYPE`, `CACHE_ELEM_ENUM`,
/// `TOKENS_PER_PAGE`, `HEAD_GRP_SIZE`, `SLIDING_WINDOW`; and attention sinks).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    pub input: InputDtype,
    pub cache: CacheDtype,
    pub group: usize,
    pub page_size: usize,
    pub sliding_window: bool,
    pub sinks: bool,
}

impl Config {
    pub fn new(input: InputDtype, cache: CacheDtype, group: usize, page_size: usize) -> Config {
        Config { input, cache, group, page_size, sliding_window: false, sinks: false }
    }

    pub fn name(&self) -> String {
        format!(
            "flashinfer_xqa_{}_{}_g{}_p{}{}{}",
            self.input.name(),
            match self.cache {
                CacheDtype::Input => self.input.name(),
                CacheDtype::E4M3 => "e4m3",
            },
            self.group,
            self.page_size,
            if self.sliding_window { "_win" } else { "" },
            if self.sinks { "_sinks" } else { "" }
        )
    }

    fn fp8(&self) -> bool {
        self.cache == CacheDtype::E4M3
    }

    /// Bytes per cache element.
    pub fn cache_bytes(&self) -> usize {
        if self.fp8() { 1 } else { 2 }
    }

    /// Accumulator rows that hold query heads: 8, or 16 for a group above 8.
    fn rows(&self) -> usize {
        if self.group > 8 { 16 } else { 8 }
    }

    /// 16-bit words per K or V row (one token of one head).
    fn row_words(&self) -> usize {
        HEAD_DIM * self.cache_bytes() / 2
    }

    /// Shared memory in 16-bit words: the warps' K/V rings, or (aliasing them) the
    /// merge buffers.
    fn smem_words(&self) -> usize {
        let ring = WARPS * STAGES * 2 * STEP * self.row_words();
        let merge = 2 * (WARPS * self.rows() * HEAD_DIM + WARPS * self.rows() * 2);
        ring.max(merge)
    }

    /// Shared memory per CTA, in bytes.
    pub fn smem_bytes(&self) -> usize {
        2 * self.smem_words() + 16
    }

    pub fn block(&self) -> [u32; 3] {
        [32, WARPS as u32, 1]
    }

    fn check(&self) {
        assert!((1..=MAX_GROUP).contains(&self.group), "group {} not in 1..=16", self.group);
        assert!([16, 32, 64, 128].contains(&self.page_size), "page size {} not 16/32/64/128", self.page_size);
    }
}

/// Stages of each warp's K/V ring.
pub const STAGES: usize = 2;

/// XQA's multi-block rule (mha.cu's `launchMHA`): CTAs per sequence.
pub fn nb_sub_seq(batch: usize, num_kv_heads: usize, max_seq_len: usize, sm_count: usize) -> usize {
    (sm_count / (batch * num_kv_heads)).max(1).min(max_seq_len.div_ceil(CTA_TILE).max(1))
}

/// Parameter positions of the kernel.
pub mod p {
    pub const Q: usize = 0;
    pub const K: usize = 1;
    pub const V: usize = 2;
    pub const PAGE_TABLE: usize = 3;
    pub const SEQ_LENS: usize = 4;
    pub const OUT: usize = 5;
    /// f32 `[num_kv_heads * group]` (read only with sinks).
    pub const SINKS: usize = 6;
    /// f32 `[batch, num_kv_heads, nb_sub_seq, group, HEAD_DIM]` (multi-block only).
    pub const SCRATCH_O: usize = 7;
    /// f32 `[batch, num_kv_heads, nb_sub_seq, group, 2]`: `(max, sum)`.
    pub const SCRATCH_ML: usize = 8;
    /// i32 `[batch * num_kv_heads]`, zero before the first launch (the kernel leaves it so).
    pub const SEMAPHORES: usize = 9;
    pub const NUM_KV_HEADS: usize = 10;
    pub const MAX_PAGES: usize = 11;
    /// `q_scale * kv_scale / sqrt(HEAD_DIM) * log2(e)`.
    pub const QK_SCALE_LOG2: usize = 12;
    /// The output scale (`kv_scale` for an FP8 cache, else 1).
    pub const VO_SCALE: usize = 13;
    /// The sliding window in tokens (read only with a window).
    pub const WINDOW: usize = 14;
}

/// The query element type of a kernel.
pub trait InputElem: Numeric + Copy {
    fn from_f16(x: Value<F16>) -> Value<Self>;
    fn zero() -> Value<Self>;
}

impl InputElem for F16 {
    fn from_f16(x: Value<F16>) -> Value<F16> {
        x
    }
    fn zero() -> Value<F16> {
        Value::<F16>::from(F16::from_f32(0.0))
    }
}

impl InputElem for BF16 {
    fn from_f16(x: Value<F16>) -> Value<BF16> {
        // Exact: an E4M3 value fits BFloat16's 8-bit significand and its exponent range.
        x.to::<f32>().to::<BF16>()
    }
    fn zero() -> Value<BF16> {
        Value::<BF16>::from(BF16::from_f32(0.0))
    }
}

/// The XQA kernel for `cfg`. Parameters are in the order of [`p`].
pub fn kernel(cfg: &Config) -> Kernel {
    cfg.check();
    match (cfg.input, cfg.cache) {
        (InputDtype::F16, CacheDtype::Input) => build::<F16, F16>(cfg),
        (InputDtype::F16, CacheDtype::E4M3) => build::<F16, F8E4M3FN>(cfg),
        (InputDtype::BF16, CacheDtype::Input) => build::<BF16, BF16>(cfg),
        (InputDtype::BF16, CacheDtype::E4M3) => build::<BF16, F8E4M3FN>(cfg),
    }
}

fn build<T: InputElem, KV: Numeric + Copy>(cfg: &Config) -> Kernel {
    let cfg = *cfg;
    Kernel::builder(cfg.name()).block_dim(cfg.block()).build(move || body::<T, KV>(&cfg))
}

#[allow(clippy::needless_range_loop)] // fragments are indexed by several loops in step
fn body<T: InputElem, KV: Numeric + Copy>(cfg: &Config) {
    let fp8 = cfg.fp8();
    let group = cfg.group as i32;
    let rows = cfg.rows();
    let hi_rows = rows == 16;
    let ps = cfg.page_size as i32;
    let row_words = cfg.row_words();
    // 16-byte chunks per K/V row, rows per copy instruction of the warp, and copy
    // instructions per 16-token tile.
    let chunks = row_words / 8;
    let rows_per_copy = 32 / chunks;
    let copies = STEP / rows_per_copy;
    // Elements of the cache per 16-byte chunk, chunks of a row each lane reads for `Q K^T`,
    // and MMA k-steps per chunk.
    let chunk_elems = 16 / cfg.cache_bytes();
    let lane_chunks = chunks / 4;
    let ksteps_per_chunk = chunk_elems / 4;

    let q = param_ptr::<T>("q");
    let k_cache = recast_ptr::<T, KV>(&param_ptr::<KV>("k_cache"), None);
    let v_cache = recast_ptr::<T, KV>(&param_ptr::<KV>("v_cache"), None);
    let page_table = param_ptr::<i32>("page_table");
    let seq_lens = param_ptr::<i32>("seq_lens");
    let out = param_ptr::<T>("out");
    let sinks = param_ptr::<f32>("sinks");
    let scratch_o = param_ptr::<f32>("scratch_o");
    let scratch_ml = param_ptr::<f32>("scratch_ml");
    let semaphores = param_ptr::<i32>("semaphores");
    let num_kv_heads = param_scalar::<i32>("num_kv_heads");
    let max_pages = param_scalar::<i32>("max_pages");
    let qk_scale_log2 = param_scalar::<f32>("qk_scale_log2");
    let vo_scale = param_scalar::<f32>("vo_scale");
    let window = param_scalar::<i32>("window");

    let (lane, warp) = (thread_idx().x, thread_idx().y);
    let tid = warp * 32 + lane;
    let (sub, kv_head, b) = (block_idx().x, block_idx().y, block_idx().z);
    let nb_sub = grid_dim().x;
    let (g, t) = (lane / 4, lane % 4);
    let hd = HEAD_DIM as i32;
    let num_qo_heads = num_kv_heads * group;
    let zero = Float32::from(0.0f32);
    let neg_inf = Float32::from(f32::NEG_INFINITY);
    let ceil_div = |a: Int32, d: i32| (a + (d - 1)) / d;

    // ---- This warp's tokens: steps of 16 over its 64-token slice of each CTA tile.
    let seq_len = seq_lens.load_at(b);
    let skip =
        if cfg.sliding_window { select(seq_len.gt(window), seq_len - window, Int32::from(0)) } else { Int32::from(0) };
    let n_tiles = ceil_div(seq_len, CTA_TILE as i32);
    let first_tile = skip / CTA_TILE as i32 + sub;
    let my_tiles = select(n_tiles.gt(first_tile), (n_tiles - first_tile + nb_sub - 1) / nb_sub, Int32::from(0));
    let tile_token = |i: Int32| (first_tile + i * nb_sub) * CTA_TILE as i32 + warp * WARP_TILE as i32;
    let last_base = tile_token(my_tiles - 1);
    let last_steps = ceil_div(seq_len - last_base, STEP as i32).max(Int32::from(0)).min(Int32::from(4));
    let n_steps = select(my_tiles.gt(0), (my_tiles - 1) * 4 + last_steps, Int32::from(0));
    // The first token of step `k` (clamped to the last step, so over-the-end prefetches
    // read valid addresses).
    let step_token = |k: Int32| {
        let k = k.min(n_steps - 1).max(Int32::from(0));
        tile_token(k / 4) + (k % 4) * STEP as i32
    };

    // ---- Shared memory: each warp's ring of `STAGES` (K, V) tiles of 16 rows.
    let smem = make_smem_tensor_aligned::<T>(Layout::col_major(cfg.smem_words()), 16);
    let flag = make_smem_tensor::<i32>(Layout::col_major(1));
    let sv = make_tensor(&smem.ptr(), Layout::col_major((8, cfg.smem_words() / 8)));
    let tile_vecs = (STEP * chunks) as i32;
    let warp_vec_base = warp * (STAGES * 2) as i32 * tile_vecs;
    let k_vec = |stage: Int32| warp_vec_base + stage * 2 * tile_vecs;
    let v_vec = |stage: Int32| warp_vec_base + stage * 2 * tile_vecs + tile_vecs;
    let gvecs = |ptr: &Ptr<T>| make_tensor(ptr, Layout::col_major((8, BOUND / 8)));
    let (k_g, v_g) = (gvecs(&k_cache), gvecs(&v_cache));
    let cp_async = make_copy_atom::<T>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });

    // Swizzles (16-byte chunk `c` of tile row `r`): K rows of odd tokens swap the halves of
    // each eight chunks, so the two tokens a quarter-warp's K loads read hit different
    // banks; V rows XOR the chunk with `r % 8`, so an `ldmatrix`'s eight rows do.
    let k_swz = |r: Int32, c: Int32| c ^ ((r % 2) * 4);
    let v_swz = |r: Int32, c: Int32| c ^ (r % 8);

    // Copy step `k` (its page `page`) into `stage`: this lane's rows `i * rows_per_copy +
    // lane / chunks`, chunk `lane % chunks`; tokens past the sequence are zero-filled.
    let (copy_row, copy_chunk) = (lane / chunks as i32, lane % chunks as i32);
    let load_step = |k: Int32, page: Int32, stage: Int32| {
        let tok0 = step_token(k);
        for i in 0..copies {
            let r = copy_row + (i * rows_per_copy) as i32;
            let tok = tok0 + r;
            let row = page * ps + tok % ps;
            let gvec = (row * num_kv_heads + kv_head) * chunks as i32 + copy_chunk;
            let pf = make_fragment::<bool>(1);
            pf.store(0, tok.lt(seq_len));
            let kd = k_vec(stage) + r * chunks as i32 + k_swz(r, copy_chunk);
            let vd = v_vec(stage) + r * chunks as i32 + v_swz(r, copy_chunk);
            copy_if(&cp_async, &pf, &k_g.slice((.., gvec)), &sv.slice((.., kd)));
            copy_if(&cp_async, &pf, &v_g.slice((.., gvec)), &sv.slice((.., vd)));
        }
    };
    let page_of = |k: Int32| page_table.load_at(b * max_pages + step_token(k) / ps);

    // ---- Q: rows `g` and (group > 8) `g + 8` of the tile, in registers, in the permuted
    // order of K's chunks: k-step `s` of this lane is elements `(t + 4 i) * chunk_elems + 4 u
    // + 0..4` (`i = s / ksteps_per_chunk`, `u = s % ksteps_per_chunk`), held as MMA k-slots
    // `2 t, 2 t + 1, 2 t + 8, 2 t + 9`.
    let q_g = gvecs(&q);
    let load_q_row = |r: Int32| -> Vec<Value<T>> {
        let head = kv_head * group + r.min(Int32::from(group - 1));
        let base = (b * num_qo_heads + head) * (hd / 8);
        let valid = r.lt(group);
        let mut v = Vec::with_capacity(HEAD_DIM);
        for i in 0..lane_chunks {
            for h in 0..chunk_elems / 8 {
                let col = (t + 4 * i as i32) * (chunk_elems / 8) as i32 + h as i32;
                let x = load_vector::<T, 8>(&q_g.slice((.., base + col)).ptr(), LoadOptions::default());
                v.extend(x.iter().map(|e| select(valid, *e, T::zero())));
            }
        }
        v
    };
    let q_lo = load_q_row(g);
    let q_hi = if hi_rows { Some(load_q_row(g + 8)) } else { None };
    let tzero = T::zero();
    let num_ksteps = HEAD_DIM / 16;
    // A fragments per k-step: (row g, k 2t..), (row g + 8, k 2t..), (row g, k 2t + 8..),
    // (row g + 8, k 2t + 8..).
    let a_q: Vec<[Value<T>; 8]> = (0..num_ksteps)
        .map(|s| {
            let at = s * 4;
            let hi = |e: usize| q_hi.as_ref().map_or(tzero, |h| h[at + e]);
            [q_lo[at], q_lo[at + 1], hi(0), hi(1), q_lo[at + 2], q_lo[at + 3], hi(2), hi(3)]
        })
        .collect();

    let mma =
        make_tiled_mma(MmaF16BF16Op::new(<T as Numeric>::DTYPE, DType::F32, (16, 8, 16)), Layout::col_major((1, 1, 1)));
    let mma16 = |a: &[Value<T>], bv: &[Value<T>], c: &[Float32]| -> Vec<Float32> {
        let (fa, fb, fc) =
            (make_fragment::<T>((8, 1, 1)), make_fragment::<T>((4, 1, 1)), make_fragment::<f32>((4, 1, 1)));
        fa.store_values(a);
        fb.store_values(bv);
        fc.store_values(c);
        gemm(&mma, &fc, &fa, &fb, &fc);
        fc.values()
    };
    let ldsm_t = make_copy_atom::<T>(CopyOp::LdMatrix { num: 4, trans: true });
    let ldmatrix_t = |svec: Int32| -> Vec<Value<T>> {
        let frag = make_fragment::<T>(8);
        copy(&ldsm_t, &sv.slice((.., svec)), &frag);
        frag.values()
    };
    // Four FP8 bytes of each pair of 16-bit words.
    let bytes_of = |w: &[Value<T>]| -> Vec<Value<F8E4M3FN>> {
        w.chunks(2).flat_map(|p| unpack::<F8E4M3FN, u32, 4>(pack::<T, u32, 2>([p[0], p[1]]))).collect()
    };
    let widen = |a: Value<F8E4M3FN>, c: Value<F8E4M3FN>| -> [Value<T>; 2] {
        let (lo, hi): (Float16, Float16) = cvt_f8x2_to_f16x2::<F8E4M3FN, _>([a, c], false);
        [T::from_f16(lo), T::from_f16(hi)]
    };
    // The head-dimension element of each accumulator register `[nt][e]` (n8 tile `nt`,
    // register `e`) of `P V`: f16/bf16 tile `nt` covers elements `8 nt ..`; FP8 tile `2 md
    // + n` takes elements `16 md + 2 j + n` for its column `j`.
    let o_dim = |nt: usize, e: usize| -> Int32 {
        let c = (e % 2) as i32;
        if fp8 {
            let (md, n) = ((nt / 2) as i32, (nt % 2) as i32);
            md * 16 + 4 * t + 2 * c + n
        } else {
            nt as i32 * 8 + 2 * t + c
        }
    };

    // ---- Online-softmax state per accumulator row (`g`, `g + 8`): the base-2 maximum, this
    // lane's partial sum, and `o[16 n8 tiles][4]`.
    let o_state = Var::new(vec![zero; 64]);
    let m_state = Var::new(vec![Float32::from(-1.0e30f32); 2]);
    let l_state = Var::new(vec![zero; 2]);

    if_(n_steps.gt(0), || {
        for st in 0..STAGES - 1 {
            let k = Int32::from(st as i32);
            load_step(k, page_of(k), k);
            cp_async_commit_group();
        }
        let next_page = Var::new(page_of(Int32::from(STAGES as i32 - 1)));
        for_range(0, n_steps, 1, |k| {
            let ahead = k + (STAGES as i32 - 1);
            load_step(ahead, next_page.get(), ahead % STAGES as i32);
            cp_async_commit_group();
            next_page.set(page_of(ahead + 1));
            cp_async_wait_group(STAGES as u32 - 1);
            sync_warp();
            let stage = k % STAGES as i32;

            // S = Q K^T over 16 tokens: two n8 tiles (tokens 8 j + g).
            let mut s = [zero; 8];
            for j in 0..2 {
                let r = Int32::from(8 * j as i32) + g;
                let row_base = k_vec(stage) + r * chunks as i32;
                for i in 0..lane_chunks {
                    let c = t + 4 * i as i32;
                    let kw = load_vector::<T, 8>(&sv.slice((.., row_base + k_swz(r, c))).ptr(), LoadOptions::default());
                    for u in 0..ksteps_per_chunk {
                        let bv: Vec<Value<T>> = if fp8 {
                            let by = bytes_of(&kw[2 * u..2 * u + 2]);
                            let (x, y) = (widen(by[0], by[1]), widen(by[2], by[3]));
                            vec![x[0], x[1], y[0], y[1]]
                        } else {
                            kw[4 * u..4 * u + 4].to_vec()
                        };
                        let ks = i * ksteps_per_chunk + u;
                        let r = mma16(&a_q[ks], &bv, &s[4 * j..4 * j + 4]);
                        s[4 * j..4 * j + 4].copy_from_slice(&r);
                    }
                }
            }

            // Mask tokens past the sequence or before the window; base-2 scores.
            let tok0 = step_token(k);
            for j in 0..2 {
                for e in 0..4 {
                    let tok = tok0 + (8 * j + e % 2) as i32 + 2 * t;
                    let mut valid = tok.lt(seq_len);
                    if cfg.sliding_window {
                        valid = valid & tok.ge(skip);
                    }
                    s[4 * j + e] = select(valid, s[4 * j + e] * qk_scale_log2, neg_inf);
                }
            }

            // Online softmax per row (the quad of lanes sharing it), and P in the input type
            // as the A operand of P V: (row g, tokens 2t..), (g + 8, 2t..), (g, 2t + 8..),
            // (g + 8, 2t + 8..).
            let (mut o_v, mut m_v, mut l_v) = (o_state.get(), m_state.get(), l_state.get());
            let mut p = [zero; 8];
            for h in 0..(if hi_rows { 2 } else { 1 }) {
                let idx = [2 * h, 2 * h + 1, 4 + 2 * h, 5 + 2 * h];
                let mut mx = m_v[h];
                for &i in &idx {
                    mx = fmax(mx, s[i]);
                }
                mx = fmax(mx, shuffle_sync_bfly(mx, 2));
                mx = fmax(mx, shuffle_sync_bfly(mx, 1));
                let corr = cute::math::exp2(m_v[h] - mx, true);
                let mut sum = zero;
                for &i in &idx {
                    p[i] = cute::math::exp2(s[i] - mx, true);
                    sum = sum + p[i];
                }
                l_v[h] = l_v[h] * corr + sum;
                for nt in 0..16 {
                    for e in [2 * h, 2 * h + 1] {
                        o_v[nt * 4 + e] = o_v[nt * 4 + e] * corr;
                    }
                }
                m_v[h] = mx;
            }
            let pt: Vec<Value<T>> = p.iter().map(|x| x.to::<T>()).collect();
            let a_p = [pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]];
            let a_p = if hi_rows { a_p } else { [a_p[0], a_p[1], tzero, tzero, a_p[4], a_p[5], tzero, tzero] };

            // O += P V, V's B fragments by transposed ldmatrix.
            let vr = (lane / 8) % 2 * 8 + lane % 8;
            let vrow = v_vec(stage) + vr * chunks as i32;
            if fp8 {
                for pp in 0..HEAD_DIM / 32 {
                    let c = (2 * pp) as i32 + lane / 16;
                    let words = bytes_of(&ldmatrix_t(vrow + v_swz(vr, c)));
                    for h in 0..2 {
                        let (wa, wb) = (&words[8 * h..8 * h + 4], &words[8 * h + 4..8 * h + 8]);
                        for n in 0..2 {
                            let (x, y) = (widen(wa[n], wa[2 + n]), widen(wb[n], wb[2 + n]));
                            let nt = 4 * pp + 2 * h + n;
                            let r = mma16(&a_p, &[x[0], x[1], y[0], y[1]], &o_v[4 * nt..4 * nt + 4]);
                            o_v[4 * nt..4 * nt + 4].copy_from_slice(&r);
                        }
                    }
                }
            } else {
                for d in 0..HEAD_DIM / 16 {
                    let c = (2 * d) as i32 + lane / 16;
                    let bv = ldmatrix_t(vrow + v_swz(vr, c));
                    for n in 0..2 {
                        let nt = 2 * d + n;
                        let r = mma16(&a_p, &bv[4 * n..4 * n + 4], &o_v[4 * nt..4 * nt + 4]);
                        o_v[4 * nt..4 * nt + 4].copy_from_slice(&r);
                    }
                }
            }
            o_state.set(o_v);
            m_state.set(m_v);
            l_state.set(l_v);
            sync_warp();
        });
        cp_async_wait_group(0);
    });

    // ---- Merge the warps' states through shared memory (aliasing the rings).
    let (o_v, m_v, l_v) = (o_state.get(), m_state.get(), l_state.get());
    let l_v: Vec<Float32> = l_v
        .iter()
        .map(|l| {
            let l = *l + shuffle_sync_bfly(*l, 1);
            l + shuffle_sync_bfly(l, 2)
        })
        .collect();
    sync_threads();
    let fsm = recast_ptr::<f32, T>(&smem.ptr(), None);
    let o_elems = WARPS * rows * HEAD_DIM;
    let so = make_tensor(&fsm, Layout::col_major(o_elems));
    let sml = make_tensor(&fsm.offset(o_elems as i32), Layout::col_major(WARPS * rows * 2));
    for h in 0..rows / 8 {
        let r = g + 8 * h as i32;
        for nt in 0..16 {
            for e in [2 * h, 2 * h + 1] {
                so.store((warp * rows as i32 + r) * hd + o_dim(nt, e), o_v[nt * 4 + e]);
            }
        }
        if_(t.eq(0), || {
            sml.store((warp * rows as i32 + r) * 2, m_v[h]);
            sml.store((warp * rows as i32 + r) * 2 + 1, l_v[h]);
        });
    }
    sync_threads();

    // Each thread: row `tid / 16 (+ 8)`, elements `8 (tid % 16) ..`.
    let col = (tid % 16) * 8;
    let task_rows: Vec<Int32> = (0..rows / 8).map(|h| tid / 16 + 8 * h as i32).collect();
    // The CTA's (m, l, o[8]) of row `r` (o unnormalised).
    let cta_state = |r: Int32| -> (Float32, Float32, Vec<Float32>) {
        let ms: Vec<Float32> = (0..WARPS).map(|w| sml.load((w as i32 * rows as i32 + r) * 2)).collect();
        let m = ms.iter().skip(1).fold(ms[0], |a, c| fmax(a, *c));
        let mut l = zero;
        let mut o = vec![zero; 8];
        for (w, mw) in ms.iter().enumerate() {
            let sc = cute::math::exp2(*mw - m, true);
            l = l + sml.load((w as i32 * rows as i32 + r) * 2 + 1) * sc;
            for (e, oe) in o.iter_mut().enumerate() {
                *oe = *oe + so.load((w as i32 * rows as i32 + r) * hd + col + e as i32) * sc;
            }
        }
        (m, l, o)
    };
    // The output row of row `r`: `o / (l + sink) * vo_scale`.
    let store_out = |r: Int32, m: Float32, l: Float32, o: &[Float32]| {
        let head = kv_head * group + r;
        let l = if cfg.sinks {
            l + cute::math::exp2(sinks.load_at(head) * Float32::from(std::f32::consts::LOG2_E) - m, true)
        } else {
            l
        };
        let inv = select(l.gt(zero), vo_scale / l, zero);
        let vals: Vec<Value<T>> = o.iter().map(|x| (*x * inv).to::<T>()).collect();
        let ov = gvecs(&out);
        let v: [Value<T>; 8] = std::array::from_fn(|i| vals[i]);
        store_vector::<T, 8>(
            &ov.slice((.., ((b * num_qo_heads + head) * hd + col) / 8)).ptr(),
            v,
            StoreOptions::default(),
        );
    };

    if_else(
        nb_sub.eq(1),
        || {
            for r in &task_rows {
                if_(r.lt(group), || {
                    let (m, l, o) = cta_state(*r);
                    store_out(*r, m, l, &o);
                });
            }
        },
        || {
            // Partial state to scratch, then the semaphore; the last CTA merges.
            let seq = b * num_kv_heads + kv_head;
            let part = |s: Int32, r: Int32| (seq * nb_sub + s) * group + r;
            for r in &task_rows {
                if_(r.lt(group), || {
                    let (m, l, o) = cta_state(*r);
                    let inv = select(l.gt(zero), Float32::from(1.0f32) / l, zero);
                    let row = part(sub, *r);
                    for (e, oe) in o.iter().enumerate() {
                        scratch_o.store_at(row * hd + col + e as i32, *oe * inv);
                    }
                    if_((tid % 16).eq(0), || {
                        scratch_ml.store_at(row * 2, m);
                        scratch_ml.store_at(row * 2 + 1, l);
                    });
                });
            }
            fence_acq_rel_gpu();
            sync_threads();
            if_(tid.eq(0), || {
                let old = atomic_add_with(
                    &semaphores.offset(seq),
                    1,
                    AtomicOptions::new(MemoryOrder::AcqRel, MemoryScope::Gpu),
                );
                let last = old.eq(nb_sub - 1);
                flag.store(0, select(last, Int32::from(1), Int32::from(0)));
                if_(last, || semaphores.store_at(seq, 0));
            });
            sync_threads();
            if_(flag.load(0).eq(1), || {
                fence_acq_rel_gpu();
                let cg = LoadOptions { cop: Some(CacheOperator::Cg), ..LoadOptions::default() };
                for r in &task_rows {
                    if_(r.lt(group), || {
                        let m = Var::new(Float32::from(-1.0e30f32));
                        let l = Var::new(zero);
                        let o = Var::new(vec![zero; 8]);
                        for_range(0, nb_sub, 1, |s| {
                            let row = part(s, *r);
                            let ms = load(&scratch_ml.offset(row * 2), cg);
                            let ls = load(&scratch_ml.offset(row * 2 + 1), cg);
                            let os: Vec<Float32> =
                                (0..8).map(|e| load(&scratch_o.offset(row * hd + col + e), cg)).collect();
                            let m_new = fmax(m.get(), ms);
                            let (c_old, c_new) =
                                (cute::math::exp2(m.get() - m_new, true), cute::math::exp2(ms - m_new, true) * ls);
                            l.set(l.get() * c_old + c_new);
                            let ov: Vec<Float32> =
                                o.get().iter().zip(&os).map(|(a, x)| *a * c_old + *x * c_new).collect();
                            o.set(ov);
                            m.set(m_new);
                        });
                        store_out(*r, m.get(), l.get(), &o.get());
                    });
                }
            });
        },
    );
}

/// A K/V cache as stored.
#[derive(Clone, Debug)]
pub enum Cache {
    F16(Vec<F16>),
    BF16(Vec<BF16>),
    E4M3(Vec<F8E4M3FN>),
}

impl Cache {
    fn get(&self, i: usize) -> f64 {
        match self {
            Cache::F16(v) => v[i].to_f32() as f64,
            Cache::BF16(v) => v[i].to_f32() as f64,
            Cache::E4M3(v) => v[i].to_f32() as f64,
        }
    }
    fn arg(&self) -> KernelArg {
        match self {
            Cache::F16(v) => KernelArg::buffer(v),
            Cache::BF16(v) => KernelArg::buffer(v),
            Cache::E4M3(v) => KernelArg::buffer(v),
        }
    }
}

/// A test problem: per-request sequence lengths, a shuffled page table, random q and K/V.
pub struct Problem {
    pub cfg: Config,
    pub seq_lens: Vec<usize>,
    pub num_kv_heads: usize,
    /// Pages per row of the page table (`max_seq_len / page_size`).
    pub max_pages: usize,
    /// `[batch, num_kv_heads * group, HEAD_DIM]`, values of the input type.
    pub q: Vec<f32>,
    pub k_cache: Cache,
    pub v_cache: Cache,
    /// `[batch, max_pages]`.
    pub page_table: Vec<i32>,
    pub q_scale: f32,
    /// The FP8 cache's scale (K and V; ignored for a cache of the input type).
    pub kv_scale: f32,
    pub window: usize,
    /// `[num_kv_heads * group]` (with `cfg.sinks`).
    pub sinks: Vec<f32>,
}

impl Problem {
    pub fn new(cfg: Config, seq_lens: &[usize], num_kv_heads: usize, seed: u64) -> Problem {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let batch = seq_lens.len();
        let ps = cfg.page_size;
        let max_pages = seq_lens.iter().map(|l| l.div_ceil(ps)).max().unwrap_or(1).max(1);
        let num_pages = batch * max_pages + 2;
        let mut ids: Vec<i32> = (0..num_pages as i32).collect();
        for i in (1..ids.len()).rev() {
            let j = ((rnd() + 1.0) * 0.5 * (i + 1) as f32) as usize % (i + 1);
            ids.swap(i, j);
        }
        let page_table = ids[..batch * max_pages].to_vec();
        let n_qo = num_kv_heads * cfg.group;
        let q = (0..batch * n_qo * HEAD_DIM).map(|_| cfg.input.round(rnd())).collect();
        let n = num_pages * ps * num_kv_heads * HEAD_DIM;
        let mut cache = || match (cfg.cache, cfg.input) {
            (CacheDtype::E4M3, _) => Cache::E4M3((0..n).map(|_| F8E4M3FN::from_f32(rnd() * 2.0)).collect()),
            (CacheDtype::Input, InputDtype::F16) => Cache::F16((0..n).map(|_| F16::from_f32(rnd())).collect()),
            (CacheDtype::Input, InputDtype::BF16) => Cache::BF16((0..n).map(|_| BF16::from_f32(rnd())).collect()),
        };
        let (k_cache, v_cache) = (cache(), cache());
        let sinks = if cfg.sinks { (0..n_qo).map(|_| rnd() * 4.0).collect() } else { vec![] };
        Problem {
            cfg,
            seq_lens: seq_lens.to_vec(),
            num_kv_heads,
            max_pages,
            q,
            k_cache,
            v_cache,
            page_table,
            q_scale: 1.0,
            kv_scale: 1.0,
            window: 0,
            sinks,
        }
    }

    /// This problem with `q_scale` and (FP8) `kv_scale`.
    pub fn with_scales(mut self, q_scale: f32, kv_scale: f32) -> Problem {
        (self.q_scale, self.kv_scale) = (q_scale, kv_scale);
        self
    }

    /// This problem with a sliding window of `window` tokens (the kernel must be built
    /// with `sliding_window`).
    pub fn with_window(mut self, window: usize) -> Problem {
        assert!(self.cfg.sliding_window, "the config has no sliding window");
        self.window = window;
        self
    }

    pub fn batch(&self) -> usize {
        self.seq_lens.len()
    }

    pub fn num_qo_heads(&self) -> usize {
        self.num_kv_heads * self.cfg.group
    }

    fn kv_scale_eff(&self) -> f32 {
        if self.cfg.fp8() { self.kv_scale } else { 1.0 }
    }

    /// CTAs per sequence for a GPU of `sm_count` SMs.
    pub fn nb_sub(&self, sm_count: usize) -> usize {
        nb_sub_seq(self.batch(), self.num_kv_heads, self.max_pages * self.cfg.page_size, sm_count)
    }

    pub fn launch(&self, nb_sub: usize) -> LaunchConfig {
        LaunchConfig::new([nb_sub as u32, self.num_kv_heads as u32, self.batch() as u32], self.cfg.block())
    }

    /// The kernel's arguments, in the order of [`p`], for `nb_sub` CTAs per sequence.
    pub fn args(&self, nb_sub: usize) -> Vec<KernelArg> {
        let n_o = self.q.len();
        let parts = if nb_sub > 1 { self.batch() * self.num_kv_heads * nb_sub * self.cfg.group } else { 1 };
        let typed = |v: &[f32]| match self.cfg.input {
            InputDtype::F16 => KernelArg::buffer(&v.iter().map(|&x| F16::from_f32(x)).collect::<Vec<_>>()),
            InputDtype::BF16 => KernelArg::buffer(&v.iter().map(|&x| BF16::from_f32(x)).collect::<Vec<_>>()),
        };
        vec![
            typed(&self.q),
            self.k_cache.arg(),
            self.v_cache.arg(),
            KernelArg::i32(self.page_table.clone()),
            KernelArg::i32(self.seq_lens.iter().map(|&l| l as i32).collect()),
            typed(&vec![0.0; n_o]),
            KernelArg::f32(if self.sinks.is_empty() { vec![0.0] } else { self.sinks.clone() }),
            KernelArg::f32(vec![0.0; parts * HEAD_DIM]),
            KernelArg::f32(vec![0.0; parts * 2]),
            KernelArg::i32(vec![0; self.batch() * self.num_kv_heads]),
            KernelArg::ScalarI32(self.num_kv_heads as i32),
            KernelArg::ScalarI32(self.max_pages as i32),
            KernelArg::ScalarF32(
                self.q_scale * self.kv_scale_eff() / (HEAD_DIM as f32).sqrt() * std::f32::consts::LOG2_E,
            ),
            KernelArg::ScalarF32(self.kv_scale_eff()),
            KernelArg::ScalarI32(self.window as i32),
        ]
    }

    /// The output as f32 from the kernel's arguments.
    pub fn output(&self, args: &[KernelArg]) -> Vec<f32> {
        match self.cfg.input {
            InputDtype::F16 => args[p::OUT].to_vec::<F16>().into_iter().map(|x| x.to_f32()).collect(),
            InputDtype::BF16 => args[p::OUT].to_vec::<BF16>().into_iter().map(|x| x.to_f32()).collect(),
        }
    }

    /// Attention in f64 (mha.cu's semantics): `softmax(q k^T * q_scale * kv_scale /
    /// sqrt(d))` over the last `window` tokens (all without a window), a sink logit in the
    /// denominator, times `v * kv_scale`.
    pub fn reference(&self) -> Vec<f32> {
        let (hkv, n_qo, ps, group) = (self.num_kv_heads, self.num_qo_heads(), self.cfg.page_size, self.cfg.group);
        let kvs = self.kv_scale_eff() as f64;
        let scale = self.q_scale as f64 * kvs / (HEAD_DIM as f64).sqrt();
        let mut o = vec![0f32; self.q.len()];
        for b in 0..self.batch() {
            let len = self.seq_lens[b];
            let first = if self.cfg.sliding_window && len > self.window { len - self.window } else { 0 };
            let addr = |tok: usize, h: usize| {
                ((self.page_table[b * self.max_pages + tok / ps] as usize * ps + tok % ps) * hkv + h) * HEAD_DIM
            };
            for qh in 0..n_qo {
                let h = qh / group;
                let qrow = (b * n_qo + qh) * HEAD_DIM;
                let s: Vec<f64> = (first..len)
                    .map(|tok| {
                        let a = addr(tok, h);
                        (0..HEAD_DIM).map(|e| self.q[qrow + e] as f64 * self.k_cache.get(a + e)).sum::<f64>() * scale
                    })
                    .collect();
                let mut mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if self.cfg.sinks {
                    mx = mx.max(self.sinks[qh] as f64);
                }
                let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let mut sum: f64 = w.iter().sum();
                if self.cfg.sinks {
                    sum += (self.sinks[qh] as f64 - mx).exp();
                }
                for e in 0..HEAD_DIM {
                    let acc: f64 =
                        w.iter().enumerate().map(|(i, wi)| wi * self.v_cache.get(addr(first + i, h) + e)).sum();
                    o[qrow + e] = (acc / sum * kvs) as f32;
                }
            }
        }
        o
    }

    /// A tolerance on `|o - ref|` for this problem's types.
    pub fn tolerance(&self) -> f32 {
        let base = match self.cfg.input {
            InputDtype::F16 => 2e-3,
            InputDtype::BF16 => 1.6e-2,
        };
        base * self.kv_scale_eff().max(1.0) * if self.cfg.fp8() { 2.0 } else { 1.0 }
    }
}

pub fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Run `p` on the simulator with `nb_sub` CTAs per sequence; returns the output (f32).
pub fn execute(p: &Problem, nb_sub: usize) -> Vec<f32> {
    let mut args = p.args(nb_sub);
    let k = kernel(&p.cfg);
    k.simulate(p.launch(nb_sub), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.output(&args)
}

/// [`execute`] checked against the f64 reference: `max |o - ref|`.
pub fn simulate(p: &Problem, nb_sub: usize) -> f32 {
    max_error(&execute(p, nb_sub), &p.reference())
}

/// Running on the GPU (`--features cuda`).
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg, HardwareInfo};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{Config, Problem, kernel, max_error};

    /// A kernel compiled and loaded, and the GPU's SM count.
    pub struct XqaKernel<'c> {
        pub cfg: Config,
        pub kernel: Kernel,
        pub func: CudaFunction<'c>,
        pub sm_count: usize,
    }

    impl<'c> XqaKernel<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &Toolchain, cfg: Config) -> Result<XqaKernel<'c>, Box<dyn Error>> {
            let arch = ctx.best_target(tc).ok_or("no target for this GPU")?;
            let k = kernel(&cfg);
            let func = ctx.load(&k.compile_with(tc, &CompileOptions::new().gpu_arch(arch))?)?;
            let sm_count = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            Ok(XqaKernel { cfg, kernel: k, func, sm_count })
        }
    }

    /// Run `p` once (with `nb_sub` CTAs per sequence, XQA's rule if `None`); if `check`,
    /// return `max |o - ref|`; if `time`, the CUDA-graph time in microseconds.
    pub fn run(
        ctx: &CudaContext,
        k: &XqaKernel<'_>,
        p: &Problem,
        nb_sub: Option<usize>,
        check: bool,
        time: bool,
    ) -> Result<(usize, f32, Option<f64>), Box<dyn Error>> {
        if k.cfg != p.cfg {
            return Err("the kernel's config differs from the problem's".into());
        }
        let nb_sub = nb_sub.unwrap_or_else(|| p.nb_sub(k.sm_count));
        let host = p.args(nb_sub);
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
        drop(host);
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(&k.kernel, &k.func, p.launch(nb_sub)).stream(s).args(&args).submit()?;
            Ok(())
        };
        let stream = ctx.stream()?;
        launch(&stream, 0)?;
        stream.synchronize()?;
        let err = if check {
            let o: Vec<f32> = match p.cfg.input {
                super::InputDtype::F16 => {
                    bufs[super::p::OUT].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect()
                }
                super::InputDtype::BF16 => {
                    bufs[super::p::OUT].to_vec::<u16>()?.into_iter().map(|x| f32::from_bits((x as u32) << 16)).collect()
                }
            };
            max_error(&o, &p.reference())
        } else {
            0.0
        };
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            let mut best = f64::MAX;
            for _ in 0..3 {
                best = best.min(benchmark(&stream, &opts, &mut launch)?);
            }
            Some(best)
        } else {
            None
        };
        Ok((nb_sub, err, t))
    }
}
