// SPDX-License-Identifier: Apache-2.0

//! Sorting-free top-k / top-p sampling from probabilities, after FlashInfer's
//! `SamplingFromProbKernel`, `TopKSamplingFromProbKernel`, `TopPSamplingFromProbKernel` and
//! `TopKTopPSamplingFromProbKernel` (`include/flashinfer/sampling.cuh`; Python
//! `flashinfer.sampling.{sampling_from_probs, top_k_sampling_from_probs,
//! top_p_sampling_from_probs, top_k_top_p_sampling_from_probs(filter_apply_order="joint")}`).
//! This is cute-rs's own code: it follows the published design (one block per row, rejection
//! rounds with two pivots, the accept rules, curand's Philox stream per row), but no
//! FlashInfer source was translated.
//!
//! # Algorithm
//!
//! One block of [`THREADS`] threads per row of `probs [batch, d]` (Float32). Each thread owns
//! `VEC = 4` consecutive probabilities (one 16-byte load) of each chunk of `THREADS * 4`.
//! A round, with the window `(low, high)` and the kept mass `q` (initially `0, 1, 1`):
//!
//! 1. **Draw** `u = uniform() * q` from the row's Philox stream.
//! 2. **Inverse-CDF pass** over the chunks, keeping only `x > low`: each chunk's masked
//!    values are scanned (thread, then a shuffle scan per warp, then the warp totals, which
//!    every warp scans again from shared memory, so one barrier per chunk gives every thread
//!    the chunk total and its own prefix). The pass stops at the chunk where the running
//!    mass passes `u`; the token is the first kept element whose inclusive CDF exceeds `u`
//!    (a block-wide minimum). If rounding leaves no such element, the last kept element seen
//!    is taken, and a row with nothing kept is reported invalid (FlashInfer's fallbacks).
//! 3. **Pivot pass** (not for plain sampling): `pivot_0` is the drawn token's probability and
//!    `pivot_1 = (pivot_0 + high) / 2`; one pass sums and counts the elements above each.
//!    * accept the token if the mass above `pivot_0` is `< top_p` and/or fewer than `top_k`
//!      elements are above it;
//!    * otherwise, if `pivot_1` passes the same test, the window becomes `(pivot_0, pivot_1)`
//!      and `q` the mass above `pivot_0`; else `low = pivot_1` and `q` the mass above it.
//!
//!    Rounds repeat while `low < high`.
//!
//! This rejection scheme samples exactly from the renormalised top-k / top-p set, where the
//! set is the tokens with fewer than `k` strictly larger probabilities and less than `p`
//! mass strictly above them (ties are all kept, as in FlashInfer's "joint" filter).
//!
//! All reductions run in a fixed order, so a row's sample depends only on the row, its
//! `top_k`/`top_p`, the seed and the offset (FlashInfer's `deterministic=True`).
//!
//! # Random numbers
//!
//! Philox4x32-10 exactly as curand's `curandStatePhilox4_32_10_t`: block `b` behaves as
//! `curand_init(seed, b, offset, &state)` followed by one `curand_uniform(&state)` per round
//! ([`philox4x32_10`], [`curand_uniform`]), so draw `n` of a row is word `(offset + n) % 4` of
//! the Philox block at counter `((offset + n) / 4, row)`, scaled to `(0, 1]`. The samples
//! are therefore reproducible from `(seed, offset)`; they need not equal FlashInfer's, whose
//! sums round differently.
//!
//! Min-p ([`Mode::MinP`], `min_p_sampling_from_probs`) is the same kernel with a different
//! first stage: block maximum, the mass `q` at or above `pivot = min_p * max`, then the
//! inverse-CDF pass over those elements with `u = uniform * q` (the `top_p` parameter carries
//! `min_p`).
//!
//! # Renorm, mask and softmax ([`Op`], [`op_kernel`])
//!
//! `top_p_renorm_probs`, `top_k_renorm_probs`, `top_k_mask_logits` and `softmax(logits,
//! temperature)` (f32, one block per row; own code after the published semantics, not
//! translated): a four-pass 8-bit radix select of the threshold key (lane-private shared
//! histograms with integer atomics, the scan of the module's sampler) then one or two passes
//! to sum and write; softmax keeps a per-thread online max and sum and merges them once. Kept
//! tokens are those with `key >= threshold`, so ties at the k-th value are all kept. The top-p
//! histogram holds the mass in units of `2^-30`, so rows should sum to less than 4.
//!
//! Below [`SPLIT_BELOW`] rows, [`OpConfig::for_batch`] spreads each row of softmax and the top-k
//! operations over several blocks ([`op_kernels`]: a short pipeline of kernels, one per radix
//! round), which fills the GPU at batch 1.
//!
//! # Top-k selection and speculative sampling
//!
//! [`Op::TopK`] is `flashinfer.top_k(logits, k)` (f32, unsorted values and i32 indices; own
//! code): the radix select of the threshold above, then one pass that writes the elements above
//! it and as many ties as are still needed, claiming slots with shared-memory counters (several
//! blocks per row: a counting pass, one global atomic per block and counter, then the write;
//! [`OpConfig::for_batch`] splits below [`SPLIT_BELOW`] rows). [`spec`] is
//! `chain_speculative_sampling` (see its documentation). Their checks and benchmarks are in
//! `examples/sampling.rs` (`examples/sampling_topk_spec_flashinfer_bench.py` for FlashInfer).
//! The page-table and ragged top-k transforms of `topk.py` are in [`transform`].
//!
//! [`transform`] is `top_k_page_table_transform` / `top_k_ragged_transform`: the same select on
//! a variable-length window of each row, with the indices mapped through a page table or offset.
//! Same device, CUDA-graph timing, `k` 2048, randn f32 scores, batch 1 / 4 / 16 / 64 and 4096 to
//! 131072 scores per row (`examples/topk_flashinfer_bench.py` for FlashInfer 0.7.5, default
//! dispatch; its `dsa_graph_safe` mode does not run on SM89): 0.4x to 1.1x of FlashInfer's time
//! (geometric mean about 0.75x); the slowest cells are 16 and 64 rows of 16384 (1.04x to 1.09x).
//!
//! Not yet covered: `indices` (row gather), per-row seed/offset tensors, `d` that is not a
//! multiple of 4 (FlashInfer narrows its vector width), sampling from logits, f16/bf16 rows
//! for the new operations, FlashInfer's SM90 split-row (cluster)
//! variant and its `top_k_first` radix path.
//!
//! # Performance
//!
//! On an RTX 2000 Ada (SM89), vocab 128512 and batches 1 to 512 from FlashInfer's
//! `bench_sampling.py`, softmax of normal(1) and gumbel(0.1) logits, CUDA-graph timing:
//! within 0.88x to 1.08x of FlashInfer 0.7.5's time at every point (sampling, top-k 10 and
//! 1000, top-p 0.5 and 0.9, joint top-k 50 / top-p 0.9), 4 to 12% faster at batch 1 and 16
//! (one barrier per chunk instead of a reduction then a scan), and level at large batch, where
//! both are bound by memory bandwidth.
//!
//! Same device, CUDA-graph timing, vocab 128512, batches 1 / 16 / 128 (cute-rs / FlashInfer):
//! `top_k` (k 50 to 2048, normal(1) logits) 0.72x at batch 1 (one row over 16 blocks), 0.5x at
//! 16 and 0.55x at 128; `chain_speculative_sampling` with 1 or 4 draft tokens 0.82 / 0.88x at
//! batch 1, 1.04 / 1.15x at 16 and 1.09 / 1.02x at 128. The inverse-CDF pass prefetches the
//! next chunk, which made plain sampling 0.79 to 0.87x at batch 1 and 16.
//!
//! `examples/sampling.rs` runs it on the simulator and, with `--features cuda` and `GPU=1`,
//! on the GPU against the reference, then benchmarks it.

use cute::prelude::*;

/// Probabilities per thread per chunk (one 16-byte load of Float32).
pub const VEC: usize = 4;
/// Threads per block on SM80 and newer (FlashInfer's choice there).
pub const THREADS: usize = 1024;

/// Which sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// `sampling_from_probs`: one inverse-CDF pass, no rounds.
    Plain,
    /// `top_k_sampling_from_probs`.
    TopK,
    /// `top_p_sampling_from_probs`.
    TopP,
    /// `top_k_top_p_sampling_from_probs(..., filter_apply_order="joint")`.
    TopKTopP,
    /// `min_p_sampling_from_probs`: samples among the tokens with `p >= min_p * max(p)`. The
    /// `top_p` parameter of [`kernel`] carries `min_p`.
    MinP,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Plain => "sampling",
            Mode::TopK => "top_k",
            Mode::TopP => "top_p",
            Mode::TopKTopP => "top_k_top_p",
            Mode::MinP => "min_p",
        }
    }
    fn uses_k(self) -> bool {
        matches!(self, Mode::TopK | Mode::TopKTopP)
    }
    fn uses_p(self) -> bool {
        matches!(self, Mode::TopP | Mode::TopKTopP)
    }
}

/// `filter_apply_order` of `top_k_top_p_sampling_from_{probs,logits}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FilterOrder {
    /// Both filters on the whole row: the [`Mode::TopKTopP`] kernel (after a softmax, from logits).
    Joint,
    /// The top-k filter first (`top_k_renorm_probs`, or `top_k_mask_logits` then softmax), then
    /// top-p sampling ([`Mode::TopP`]) from what is left.
    TopKFirst,
}

/// A kernel's compile-time configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    pub mode: Mode,
    /// Threads per block, a multiple of 32 up to 1024.
    pub threads: usize,
    /// `top_k` / `top_p` are per-row arrays (`[batch]`) instead of scalars.
    pub per_row: bool,
    /// `seed` and `offset` are per-row `u64` arrays (`[batch]`) instead of scalars: FlashInfer's
    /// tensor `seed` / `offset` (`seed_arr[bx * seed_stride]`, stride one).
    pub per_seed: bool,
}

impl Config {
    pub fn new(mode: Mode) -> Config {
        Config { mode, threads: THREADS, per_row: false, per_seed: false }
    }
    pub fn per_seed(self, per_seed: bool) -> Config {
        Config { per_seed, ..self }
    }
    pub fn threads(self, threads: usize) -> Config {
        Config { threads, ..self }
    }
    pub fn per_row(self, per_row: bool) -> Config {
        Config { per_row, ..self }
    }
}

// ---------------------------------------------------------------------------------------
// Philox4x32-10, as curand and Random123.

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
/// curand's `CURAND_2POW32_INV`.
const TWO_POW_M32: f32 = 2.328_306_4e-10;

/// The Philox4x32-10 block of `ctr` under `key` (Random123's `philox4x32(ctr, key)` with 10
/// rounds, curand's `curand_Philox4x32_10`).
pub fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (mut c, mut k) = (ctr, key);
    for r in 0..10 {
        if r > 0 {
            k = [k[0].wrapping_add(PHILOX_W0), k[1].wrapping_add(PHILOX_W1)];
        }
        let p0 = PHILOX_M0 as u64 * c[0] as u64;
        let p1 = PHILOX_M1 as u64 * c[2] as u64;
        c = [(p1 >> 32) as u32 ^ c[1] ^ k[0], p1 as u32, (p0 >> 32) as u32 ^ c[3] ^ k[1], p0 as u32];
    }
    c
}

/// Word `n` of curand's Philox stream after `curand_init(seed, subsequence, 0, ..)`, i.e. the
/// `n`-th `curand()` result after `curand_init(seed, subsequence, offset, ..)` with
/// `n = offset + draws`.
pub fn curand_word(seed: u64, subsequence: u64, n: u64) -> u32 {
    let c = n >> 2;
    let ctr = [c as u32, (c >> 32) as u32, subsequence as u32, (subsequence >> 32) as u32];
    philox4x32_10(ctr, [seed as u32, (seed >> 32) as u32])[(n & 3) as usize]
}

/// The `draw`-th `curand_uniform` (from 0) after `curand_init(seed, subsequence, offset, ..)`:
/// a value in `(0, 1]`.
pub fn curand_uniform(seed: u64, subsequence: u64, offset: u64, draw: u64) -> f32 {
    let x = curand_word(seed, subsequence, offset.wrapping_add(draw));
    (x as f32).mul_add(TWO_POW_M32, TWO_POW_M32 / 2.0)
}

/// The device side of [`philox4x32_10`].
fn philox_device(ctr: [Uint32; 4], key: [Uint32; 2]) -> [Uint32; 4] {
    let (mut c, mut k) = (ctr, key);
    for r in 0..10 {
        if r > 0 {
            k = [k[0] + PHILOX_W0, k[1] + PHILOX_W1];
        }
        let (hi0, lo0) = (c[0].mul_hi(PHILOX_M0), c[0] * PHILOX_M0);
        let (hi1, lo1) = (c[2].mul_hi(PHILOX_M1), c[2] * PHILOX_M1);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
    }
    c
}

/// The device side of [`curand_uniform`]: word `n` of the stream `(seed, subsequence)`.
fn uniform_device(seed: Uint64, subsequence: Uint64, n: Uint64) -> Float32 {
    let c = n >> 2u64;
    let ctr = [c.to::<u32>(), (c >> 32u64).to::<u32>(), subsequence.to::<u32>(), (subsequence >> 32u64).to::<u32>()];
    let w = philox_device(ctr, [seed.to::<u32>(), (seed >> 32u64).to::<u32>()]);
    let j = (n & 3u64).to::<u32>();
    let x = select(j.eq(0u32), w[0], select(j.eq(1u32), w[1], select(j.eq(2u32), w[2], w[3])));
    x.to::<f32>() * TWO_POW_M32 + TWO_POW_M32 / 2.0
}

/// A test kernel: `out[b * n + i]` is the `i`-th `curand_uniform` of block `b` after
/// `curand_init(seed, b, offset, ..)`. Parameters `out, n, seed, offset`; one thread per
/// block.
pub fn uniform_kernel() -> Kernel {
    Kernel::builder("flashinfer_philox_uniform").block_dim([1, 1, 1]).build(|| {
        let out = param_ptr::<f32>("out");
        let n = param_scalar::<i32>("n");
        let seed = param_scalar::<u64>("seed");
        let offset = param_scalar::<u64>("offset");
        let b = block_idx().x;
        let sub = b.to::<u64>();
        for_range(0, n, 1, |i| {
            let u = uniform_device(seed, sub, offset + i.to::<u64>());
            out.store_at(b * n + i, u);
        });
    })
}

// ---------------------------------------------------------------------------------------
// The kernel.

/// Parameter positions of [`kernel`]: `probs [batch, d]` (f32), `output [batch]` (i32),
/// `valid [batch]` (i32, 0 or 1), `top_k` (i32 scalar or `[batch]`), `top_p` (f32 scalar
/// or `[batch]`), `d` (i32, a multiple of 4), `seed` and `offset` (u64). `top_k` and
/// `top_p` are passed (and ignored) whatever the mode.
pub mod p {
    pub const PROBS: usize = 0;
    pub const OUTPUT: usize = 1;
    pub const VALID: usize = 2;
    pub const TOP_K: usize = 3;
    pub const TOP_P: usize = 4;
    pub const D: usize = 5;
    pub const SEED: usize = 6;
    pub const OFFSET: usize = 7;
}

/// The warp-inclusive scan of `v` over the lanes (shuffle-up, deterministic).
fn warp_inclusive_scan(v: Float32, lane: Int32) -> Float32 {
    let mut s = v;
    for off in [1, 2, 4, 8, 16] {
        let t = shuffle_sync_up(s, off);
        s = select(lane.ge(off), s + t, s);
    }
    s
}

/// Chunks whose loads are issued together in the streaming passes (a thread has this many
/// 16-byte loads in flight; one at a time leaves a block latency-bound at small batch).
const UNROLL: usize = 4;

/// Visit the chunks of a row, [`UNROLL`] at a time: all of a group's vectors are loaded first,
/// then `f(col, vector)` runs on each. Chunks past the end of the row load as the fill value
/// and have `col >= dv`.
fn for_chunks(
    chunks: Int32,
    nt: i32,
    tid: Int32,
    load: &dyn Fn(Int32) -> [Float32; VEC],
    f: &dyn Fn(Int32, [Float32; VEC]),
) {
    let groups = (chunks + (UNROLL as i32 - 1)) / UNROLL as i32;
    for_range(0, groups, 1, |g| {
        let cols: Vec<Int32> = (0..UNROLL).map(|u| (g * UNROLL as i32 + u as i32) * nt + tid).collect();
        let xs: Vec<[Float32; VEC]> = cols.iter().map(|c| load(*c)).collect();
        for (c, x) in cols.into_iter().zip(xs) {
            f(c, x);
        }
    });
}

/// The pieces of a block that [`Pass::inverse_cdf`] needs: the row geometry, this thread's
/// place and two shared scratch arrays (`red`: 64 floats, `cand`: 32 ints).
struct Pass<'a> {
    d: Int32,
    chunks: Int32,
    nt: i32,
    nwarps: usize,
    tid: Int32,
    warp: Int32,
    lane: Int32,
    red: &'a Ptr<f32>,
    cand: &'a Ptr<i32>,
}

impl Pass<'_> {
    /// The block minimum (`max = false`) or maximum of `x`, through `buf`; uniform.
    fn block_min_max(&self, x: Int32, buf: &Ptr<i32>, max: bool) -> Int32 {
        let (lane, warp, nwarps) = (self.lane, self.warp, self.nwarps);
        let kind = if max { ReduxKind::Max } else { ReduxKind::Min };
        let w = warp_redux_sync(x, kind);
        if_(lane.eq(0), || buf.store_at(warp, w));
        sync_threads();
        let init = if max { i32::MIN } else { i32::MAX };
        let mine = select(lane.lt(nwarps as i32), buf.load_at(lane.min(nwarps as i32 - 1)), Int32::from(init));
        warp_redux_sync(mine, kind)
    }

    /// One inverse-CDF pass over the elements `keep` accepts, for the draw `u`: the token (the
    /// first kept element whose inclusive CDF exceeds `u`), or `d` if none was found, and the
    /// last kept token seen (or -1). `load(col)` gives the vector `col` of the row, zeros past
    /// its end.
    fn inverse_cdf(
        &self,
        load: &dyn Fn(Int32) -> [Float32; VEC],
        keep: &dyn Fn(Float32) -> Boolean,
        u: Float32,
    ) -> (Int32, Int32) {
        let Pass { d, chunks, nt, nwarps, tid, warp, lane, red, cand } = *self;
        let v = VEC as i32;
        let agg = Var::new(Float32::constant(0.0));
        let i = Var::new(Int32::from(0));
        let sel = Var::new(d);
        let last = Var::new(Int32::from(-1));
        // The next chunk's vector is loaded before the current one's scan, so the load's latency
        // overlaps the barrier and the scan.
        let cur = load(tid).map(Var::new);
        while_carry(
            (),
            |()| i.get().lt(chunks) & !agg.get().gt(u),
            |()| {
                let ic = i.get();
                let col = ic * nt + tid;
                let x: [Float32; VEC] = std::array::from_fn(|j| cur[j].get());
                let next = load(col + nt);
                for (c, n) in cur.iter().zip(next) {
                    c.set(n);
                }
                let base_idx = col * v;
                let kept: [Boolean; VEC] = std::array::from_fn(|j| keep(x[j]));
                let m: [Float32; VEC] = std::array::from_fn(|j| select(kept[j], x[j], Float32::constant(0.0)));
                let mut part = [m[0]; VEC];
                for j in 1..VEC {
                    part[j] = part[j - 1] + m[j];
                }
                let incl = warp_inclusive_scan(part[VEC - 1], lane);
                let buf = (ic & 1) * 32;
                if_(lane.eq(31), || red.store_at(buf + warp, incl));
                sync_threads();
                // Every warp scans the warp totals: the chunk total and this warp's prefix.
                let wt = select(
                    lane.lt(nwarps as i32),
                    red.load_at(buf + lane.min(nwarps as i32 - 1)),
                    Float32::constant(0.0),
                );
                let wincl = warp_inclusive_scan(wt, lane);
                let total = shuffle_sync(wincl, nwarps as i32 - 1);
                let wprev = shuffle_sync(wincl, (warp - 1).max(0));
                let wexcl = select(warp.gt(0), wprev, Float32::constant(0.0));
                let a = agg.get();
                if_(((a + total).gt(u)) & sel.get().eq(d), || {
                    let tprev = shuffle_sync_up(incl, 1);
                    let texcl = select(lane.gt(0), tprev, Float32::constant(0.0));
                    let base = a + (wexcl + texcl);
                    let c = Var::new(Int32::from(i32::MAX));
                    for j in (0..VEC).rev() {
                        let hit = kept[j] & (base + part[j]).gt(u);
                        c.set(select(hit, base_idx + j as i32, c.get()));
                    }
                    let found = self.block_min_max(c.get(), cand, false);
                    if_(found.lt(i32::MAX), || sel.set(found));
                });
                let mut l = last.get();
                for (j, &kj) in kept.iter().enumerate() {
                    l = select(kj, base_idx + j as i32, l);
                }
                last.set(l);
                agg.set(a + total);
                i.set(ic + 1);
            },
        );
        (sel.get(), last.get())
    }
}

/// The Philox `(seed, offset)` of `row`: the `seed` / `offset` parameters, scalars or (with
/// `per_seed`) `[batch]` arrays.
fn seed_offset(per_seed: bool, row: Int32) -> (Uint64, Uint64) {
    if per_seed {
        (param_ptr::<u64>("seed").load_at(row), param_ptr::<u64>("offset").load_at(row))
    } else {
        (param_scalar::<u64>("seed"), param_scalar::<u64>("offset"))
    }
}

/// The sampler of `cfg` (see the module documentation; parameters as [`p`]).
pub fn kernel(cfg: Config) -> Kernel {
    let threads = cfg.threads;
    assert!((32..=1024).contains(&threads) && threads.is_multiple_of(32), "threads = {threads}");
    let nwarps = threads / 32;
    let prefix = if cfg.mode == Mode::Plain { String::new() } else { format!("{}_", cfg.mode.name()) };
    let name = format!(
        "flashinfer_{prefix}sampling_from_probs_t{threads}{}{}",
        if cfg.per_row { "_rows" } else { "" },
        if cfg.per_seed { "_seeds" } else { "" }
    );
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let probs = param_ptr::<f32>("probs");
        let output = param_ptr::<i32>("output");
        let valid = param_ptr::<i32>("valid");
        let row = block_idx().x;
        let (top_k, top_p) = if cfg.per_row {
            let ka = param_ptr::<i32>("top_k");
            let pa = param_ptr::<f32>("top_p");
            (ka.load_at(row), pa.load_at(row))
        } else {
            (param_scalar::<i32>("top_k"), param_scalar::<f32>("top_p"))
        };
        let d = param_scalar::<i32>("d");
        let (seed, offset) = seed_offset(cfg.per_seed, row);

        let tid = thread_idx().x;
        let (warp, lane) = (tid >> 5, tid & 31);
        let nt = threads as i32;
        let v = VEC as i32;
        let dv = d / v;
        let chunks = (dv + (nt - 1)) / nt;
        let row_vec = row * dv;
        let sub = row.to::<u64>();

        // Warp totals of a chunk (double-buffered by chunk parity, so one barrier per chunk
        // suffices), candidate tokens, last kept tokens, and the pivot sums and counts.
        let red = make_smem_tensor::<f32>(Layout::col_major(64)).ptr();
        let cand = make_smem_tensor::<i32>(Layout::col_major(32)).ptr();
        let lastv = make_smem_tensor::<i32>(Layout::col_major(32)).ptr();
        let piv_s = make_smem_tensor::<f32>(Layout::col_major(64)).ptr();
        let piv_c = make_smem_tensor::<i32>(Layout::col_major(64)).ptr();

        let pass = Pass { d, chunks, nt, nwarps, tid, warp, lane, red: &red, cand: &cand };
        let block_min_max = |x: Int32, buf: &Ptr<i32>, max: bool| pass.block_min_max(x, buf, max);

        // Load the vector `col` of the row (zeros past its end).
        let load = |col: Int32| -> [Float32; VEC] {
            let at = (row_vec + col.min(dv - 1)) * v;
            let x = load_vector::<f32, VEC>(&probs.offset(at), LoadOptions::default());
            let inb = col.lt(dv);
            x.map(|e| select(inb, e, Float32::constant(0.0)))
        };

        // One inverse-CDF pass over the kept elements (`x > low`, or `x >= low` for min-p).
        let sample_pass = |u: Float32, low: Float32| -> (Int32, Int32) {
            let keep = |x: Float32| if cfg.mode == Mode::MinP { x.ge(low) } else { x.gt(low) };
            pass.inverse_cdf(&load, &keep, u)
        };

        // The fallback when the pass found no token: the last kept one, or invalid.
        let resolve = |sel: Int32, last: Int32, ok: &Var<Int32>| -> Int32 {
            let out = Var::new(sel);
            if_(sel.eq(d), || {
                let lv = block_min_max(last, &lastv, true);
                if_else(lv.lt(0), || ok.set(0), || out.set(lv));
            });
            select(out.get().ge(0) & out.get().lt(d), out.get(), Int32::from(0))
        };

        // Sum `v` over the block (a fresh scratch each call, so no barrier is needed after).
        let block_sum = |v: Float32| -> Float32 {
            let scratch = make_smem_tensor::<f32>(Layout::col_major(32)).ptr();
            let w = warp_reduction_sum(v, 32);
            if_(lane.eq(0), || scratch.store_at(warp, w));
            sync_threads();
            let has = lane.lt(nwarps as i32);
            warp_reduction_sum(select(has, scratch.load_at(lane.min(nwarps as i32 - 1)), Float32::constant(0.0)), 32)
        };

        let ok = Var::new(Int32::from(1));
        let token = if cfg.mode == Mode::MinP {
            // Three passes over the row: its maximum (non-negative floats order as their bits),
            // the mass at or above `pivot = min_p * max`, and the inverse-CDF pass.
            let mx = Var::new(Int32::from(0));
            for_chunks(chunks, nt, tid, &|c| load(c), &|_, x| {
                let mut m = mx.get();
                for e in x {
                    m = m.max(e.bitcast::<i32>());
                }
                mx.set(m);
            });
            let pivot = block_min_max(mx.get(), &cand, true).bitcast::<f32>() * top_p;
            let mass = Var::new(Float32::constant(0.0));
            for_chunks(chunks, nt, tid, &|c| load(c), &|_, x| {
                let mut s = mass.get();
                for e in x {
                    s = s + select(e.ge(pivot), e, Float32::constant(0.0));
                }
                mass.set(s);
            });
            let q = block_sum(mass.get());
            let u = uniform_device(seed, sub, offset) * q;
            let (sel, last) = sample_pass(u, pivot);
            resolve(sel, last, &ok)
        } else if cfg.mode == Mode::Plain {
            let u = uniform_device(seed, sub, offset);
            let (sel, last) = sample_pass(u, Float32::constant(0.0));
            resolve(sel, last, &ok)
        } else {
            let q = Var::new(Float32::constant(1.0));
            let low = Var::new(Float32::constant(0.0));
            let high = Var::new(Float32::constant(1.0));
            let draw = Var::new(offset);
            let token = Var::new(Int32::from(0));
            let done = Var::new(Boolean::from(false));
            while_carry(
                (),
                |()| !done.get() & low.get().lt(high.get()),
                |()| {
                    let u = uniform_device(seed, sub, draw.get()) * q.get();
                    draw.set(draw.get() + 1u64);
                    let (sel, last) = sample_pass(u, low.get());
                    let t = resolve(sel, last, &ok);
                    token.set(t);
                    if_else(
                        ok.get().eq(0),
                        || done.set(Boolean::from(true)),
                        || {
                            let p0 = probs.load_at(row_vec * v + t);
                            let p1 = (p0 + high.get()) * 0.5f32;
                            // Sums and counts above both pivots over the whole row.
                            let acc = Var::new((
                                Float32::constant(0.0),
                                Float32::constant(0.0),
                                Int32::from(0),
                                Int32::from(0),
                            ));
                            for_range(0, chunks, 1, |ic| {
                                let x = load(ic * nt + tid);
                                let (mut s0, mut s1, mut c0, mut c1) = acc.get();
                                for e in x {
                                    let (g0, g1) = (e.gt(p0), e.gt(p1));
                                    // The masses are needed in every mode: they become `q`.
                                    s0 = s0 + select(g0, e, Float32::constant(0.0));
                                    s1 = s1 + select(g1, e, Float32::constant(0.0));
                                    if cfg.mode.uses_k() {
                                        c0 = c0 + select(g0, Int32::from(1), Int32::from(0));
                                        c1 = c1 + select(g1, Int32::from(1), Int32::from(0));
                                    }
                                }
                                acc.set((s0, s1, c0, c1));
                            });
                            let (s0, s1, c0, c1) = acc.get();
                            let (s0, s1) = (warp_reduction_sum(s0, 32), warp_reduction_sum(s1, 32));
                            let (c0, c1) = if cfg.mode.uses_k() {
                                (warp_redux_sync(c0, ReduxKind::Add), warp_redux_sync(c1, ReduxKind::Add))
                            } else {
                                (c0, c1)
                            };
                            if_(lane.eq(0), || {
                                piv_s.store_at(warp * 2, s0);
                                piv_s.store_at(warp * 2 + 1, s1);
                                if cfg.mode.uses_k() {
                                    piv_c.store_at(warp * 2, c0);
                                    piv_c.store_at(warp * 2 + 1, c1);
                                }
                            });
                            sync_threads();
                            let has = lane.lt(nwarps as i32);
                            let li = lane.min(nwarps as i32 - 1) * 2;
                            let zero = Float32::constant(0.0);
                            let s0 = warp_reduction_sum(select(has, piv_s.load_at(li), zero), 32);
                            let s1 = warp_reduction_sum(select(has, piv_s.load_at(li + 1), zero), 32);
                            let ok_at = |s: Float32, c: Int32| -> Boolean {
                                match cfg.mode {
                                    Mode::TopP => s.lt(top_p),
                                    Mode::TopK => c.lt(top_k),
                                    _ => s.lt(top_p) & c.lt(top_k),
                                }
                            };
                            let (c0, c1) = if cfg.mode.uses_k() {
                                let z = Int32::from(0);
                                (
                                    warp_redux_sync(select(has, piv_c.load_at(li), z), ReduxKind::Add),
                                    warp_redux_sync(select(has, piv_c.load_at(li + 1), z), ReduxKind::Add),
                                )
                            } else {
                                (Int32::from(0), Int32::from(0))
                            };
                            if_else(
                                ok_at(s0, c0),
                                || done.set(Boolean::from(true)),
                                || {
                                    if_else(
                                        ok_at(s1, c1),
                                        || {
                                            low.set(p0);
                                            high.set(p1);
                                            q.set(s0);
                                        },
                                        || {
                                            low.set(p1);
                                            q.set(s1);
                                        },
                                    );
                                },
                            );
                        },
                    );
                },
            );
            token.get()
        };
        if_(tid.eq(0), || {
            output.store_at(row, token);
            valid.store_at(row, ok.get());
        });
    })
}

// ---------------------------------------------------------------------------------------
// Sampling from logits (the Gumbel-max trick).

/// Parameter positions of [`logits_kernel`]: `logits [batch, d]` (f32), `output [batch]` (i32),
/// `d` (i32, a multiple of 4), `seed` and `offset` (u64 scalars, or `[batch]` arrays with
/// [`Config::per_seed`]).
pub mod l {
    pub const LOGITS: usize = 0;
    pub const OUTPUT: usize = 1;
    pub const D: usize = 2;
    pub const SEED: usize = 3;
    pub const OFFSET: usize = 4;
}

/// FlashInfer's `1 - epsilon` (keeps the uniform below one).
const GUMBEL_SCALE: f32 = 1.0 - f32::EPSILON;

/// The Gumbel noise of one uniform `x` in `(0, 1]`: `-ln(-ln(x))`, as `-ln2 * log2(-log2(x))`.
pub fn gumbel_noise(x: f32) -> f32 {
    -std::f32::consts::LN_2 * (-(x * GUMBEL_SCALE).log2()).log2()
}

/// `sampling_from_logits` (`SamplingFromLogitsKernel`): `argmax(logits + gumbel)` per row,
/// one block per row. Token `i` of row `b` takes noise from the Philox block of
/// `curand_init(seed, b * d + 4 * (i / 4), offset)`, word `i % 4` of one `curand_uniform4`, so
/// the offset must be a multiple of 4 (as FlashInfer's generator makes it). Ties go to the larger
/// token; a row of `-inf` gives the last token, a token past `d` gives 0 (FlashInfer's rules).
/// Own code after the published design; `cfg.mode`, `cfg.per_row` are ignored.
pub fn logits_kernel(cfg: Config) -> Kernel {
    let threads = cfg.threads;
    assert!((32..=1024).contains(&threads) && threads.is_multiple_of(32), "threads = {threads}");
    let nwarps = threads / 32;
    let name = format!("flashinfer_sampling_from_logits_t{threads}{}", if cfg.per_seed { "_seeds" } else { "" });
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let logits = param_ptr::<f32>("logits");
        let output = param_ptr::<i32>("output");
        let d = param_scalar::<i32>("d");
        let row = block_idx().x;
        let (seed, offset) = seed_offset(cfg.per_seed, row);
        let tid = thread_idx().x;
        let (warp, lane) = (tid >> 5, tid & 31);
        let nt = threads as i32;
        let v = VEC as i32;
        let dv = d / v;
        let chunks = (dv + (nt - 1)) / nt;
        let row_vec = row * dv;
        let red = make_smem_tensor::<f32>(Layout::col_major(64)).ptr();
        let cand = make_smem_tensor::<i32>(Layout::col_major(32)).ptr();
        let cand2 = make_smem_tensor::<i32>(Layout::col_major(32)).ptr();
        let pass = Pass { d, chunks, nt, nwarps, tid, warp, lane, red: &red, cand: &cand };

        let load = |col: Int32| -> [Float32; VEC] {
            let at = (row_vec + col.min(dv - 1)) * v;
            load_vector::<f32, VEC>(&logits.offset(at), LoadOptions::default())
        };
        let ctr0 = (offset >> 2u64).to::<u32>();
        let ctr1 = (offset >> 34u64).to::<u32>();
        let skey = [seed.to::<u32>(), (seed >> 32u64).to::<u32>()];
        let best_v = Var::new(Float32::constant(f32::NEG_INFINITY));
        let best_i = Var::new(Int32::from(0));
        for_chunks(chunks, nt, tid, &|c| load(c), &|col, x| {
            let inb = col.lt(dv);
            let sub = (row * d + col * v).to::<u64>();
            let w = philox_device([ctr0, ctr1, sub.to::<u32>(), (sub >> 32u64).to::<u32>()], skey);
            let mut bv = best_v.get();
            let mut bi = best_i.get();
            for j in 0..VEC {
                let u = w[j].to::<f32>() * TWO_POW_M32 + TWO_POW_M32 / 2.0;
                let l2 = cute::math::log2(Float32::constant(0.0) - cute::math::log2(u * GUMBEL_SCALE, true), true);
                let g = l2 * -std::f32::consts::LN_2;
                let val = select(inb, x[j] + g, Float32::constant(f32::NEG_INFINITY));
                let better = val.ge(bv) & col.lt(chunks * nt);
                bv = select(better, val, bv);
                bi = select(better, col * v + j as i32, bi);
            }
            best_v.set(bv);
            best_i.set(bi);
        });
        // Block argmax: the orderable integer key of the value, then the largest index with it.
        let bits = best_v.get().bitcast::<i32>();
        let key = bits ^ ((bits >> 31) & 0x7FFF_FFFF);
        let top = pass.block_min_max(key, &cand, true);
        let idx = pass.block_min_max(select(key.eq(top), best_i.get(), Int32::from(-1)), &cand2, true);
        if_(tid.eq(0), || output.store_at(row, select(idx.lt(d), idx, Int32::from(0))));
    })
}

/// A `sampling_from_logits` problem: `logits [batch, d]`.
#[derive(Clone, Debug)]
pub struct LogitsProblem {
    pub batch: usize,
    pub d: usize,
    pub logits: Vec<f32>,
    pub seed: u64,
    pub offset: u64,
    /// Per-row seeds and offsets for [`Config::per_seed`] kernels; empty otherwise.
    pub seeds: Vec<u64>,
    pub offsets: Vec<u64>,
}

impl LogitsProblem {
    pub fn new(batch: usize, d: usize, dist: Dist, seed: u64) -> LogitsProblem {
        assert!(d.is_multiple_of(VEC), "d = {d} must be a multiple of {VEC}");
        let logits = (0..(batch * d) as u64).map(|i| dist.logit(i) as f32).collect();
        LogitsProblem { batch, d, logits, seed, offset: 0, seeds: Vec::new(), offsets: Vec::new() }
    }

    pub fn row(&self, b: usize) -> &[f32] {
        &self.logits[b * self.d..(b + 1) * self.d]
    }

    pub fn launch(&self, cfg: Config) -> LaunchConfig {
        LaunchConfig::new([self.batch as u32, 1, 1], [cfg.threads as u32, 1, 1])
    }

    /// The kernel's arguments (see [`l`]).
    pub fn args(&self, cfg: Config) -> Vec<KernelArg> {
        vec![
            KernelArg::buffer(&self.logits),
            KernelArg::buffer(&vec![-1i32; self.batch]),
            KernelArg::ScalarI32(self.d as i32),
            seed_arg(cfg.per_seed, self.batch, self.seed, &self.seeds),
            seed_arg(cfg.per_seed, self.batch, self.offset, &self.offsets),
        ]
    }

    /// The noisy logits of row `b` (`logit + gumbel`, f32, from the same Philox words).
    pub fn perturbed(&self, b: usize) -> Vec<f32> {
        let (seed, offset) =
            (self.seeds.get(b).copied().unwrap_or(self.seed), self.offsets.get(b).copied().unwrap_or(self.offset));
        assert!(offset.is_multiple_of(4));
        self.row(b)
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let sub = (b * self.d + i / 4 * 4) as u64;
                let u = curand_uniform(seed, sub, offset, (i % 4) as u64);
                x + gumbel_noise(u)
            })
            .collect()
    }

    /// The reference token of row `b` (the last of equal maxima) and the gap between the best
    /// and second-best perturbed values (a sample may differ from the reference only within
    /// rounding of that gap).
    pub fn reference(&self, b: usize) -> (usize, f32) {
        let v = self.perturbed(b);
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x >= v[best] {
                best = i;
            }
        }
        let second =
            v.iter().enumerate().filter(|(i, _)| *i != best).map(|(_, &x)| x).fold(f32::NEG_INFINITY, f32::max);
        (best, v[best] - second)
    }

    /// Check the samples: each equals the reference, or is within `1e-3` of its value (the device's
    /// fast `log2` differs from libm in the last bits). Returns the number of rows that differed.
    pub fn check(&self, output: &[i32]) -> usize {
        let mut near = 0;
        for (b, &got) in output.iter().enumerate().take(self.batch) {
            let (want, _) = self.reference(b);
            let got = got as usize;
            if got != want {
                let v = self.perturbed(b);
                assert!(got < self.d && (v[want] - v[got]).abs() < 1e-3, "row {b}: sampled {got}, reference {want}");
                near += 1;
            }
        }
        near
    }
}

/// Run [`logits_kernel`] on the simulator; returns the tokens.
pub fn simulate_logits(p: &LogitsProblem, cfg: Config) -> Vec<i32> {
    let k = logits_kernel(cfg);
    let mut args = p.args(cfg);
    k.simulate(p.launch(cfg), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    args[l::OUTPUT].to_vec::<i32>()
}

// ---------------------------------------------------------------------------------------
// Row filters (top-p / top-k renorm, top-k mask) and softmax.

/// A row operation besides sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// `top_p_renorm_probs`: keep the tokens with less than `top_p` mass strictly above them,
    /// scale them to sum to one, zero the rest.
    TopPRenorm,
    /// `top_k_renorm_probs`: keep the `top_k` largest probabilities (all ties of the last),
    /// scale them to sum to one, zero the rest.
    TopKRenorm,
    /// `top_k_mask_logits`: keep the `top_k` largest logits (all ties of the last), set the
    /// rest to `-inf`.
    TopKMask,
    /// `softmax(logits, temperature)`.
    Softmax,
    /// `flashinfer.top_k(logits, k)`: the `k` largest logits and their indices, unsorted (ties at
    /// the k-th value: some of them). `output` is `[batch, k]` and an extra `indices [batch, k]`
    /// (i32) buffer follows `d`; `k` is the same for every row (scalar).
    TopK,
}

impl Op {
    pub fn name(self) -> &'static str {
        match self {
            Op::TopPRenorm => "top_p_renorm",
            Op::TopKRenorm => "top_k_renorm",
            Op::TopKMask => "top_k_mask",
            Op::Softmax => "softmax",
            Op::TopK => "top_k_select",
        }
    }
    /// Whether the row holds logits (any float) rather than probabilities.
    pub fn takes_logits(self) -> bool {
        matches!(self, Op::TopKMask | Op::Softmax | Op::TopK)
    }
}

/// A row operation's compile-time configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpConfig {
    pub op: Op,
    /// Threads per block: a power of two from 64 to 1024.
    pub threads: usize,
    /// `top_k` and the parameter are per-row arrays instead of scalars.
    pub per_row: bool,
    /// Blocks per row: 1 is [`op_kernel`]; more runs the [`op_kernels`] pipeline (softmax and
    /// the top-k operations), which fills the GPU at small batch.
    pub split: usize,
}

impl OpConfig {
    pub fn new(op: Op) -> OpConfig {
        OpConfig { op, threads: THREADS, per_row: false, split: 1 }
    }
    pub fn split(self, split: usize) -> OpConfig {
        OpConfig { split, ..self }
    }
    /// The best configuration for `batch` rows: rows are split over blocks when there are too
    /// few of them to fill the GPU (`batch` below [`SPLIT_BELOW`]).
    pub fn for_batch(op: Op, batch: usize) -> OpConfig {
        let split = if batch >= SPLIT_BELOW { 1 } else { (SPLIT_BELOW / batch).clamp(1, MAX_SPLIT) };
        OpConfig::new(op).split(split)
    }
    pub fn threads(self, threads: usize) -> OpConfig {
        OpConfig { threads, ..self }
    }
    pub fn per_row(self, per_row: bool) -> OpConfig {
        OpConfig { per_row, ..self }
    }
}

/// Parameter positions of [`op_kernel`]: `input [batch, d]` and `output [batch, d]` (f32),
/// `top_k` (i32 scalar or `[batch]`), `param` (f32 scalar or `[batch]`: `top_p` for
/// [`Op::TopPRenorm`], the temperature for [`Op::Softmax`], else ignored), `d` (a multiple of
/// 4). Buffers are 16-byte aligned.
pub mod q {
    pub const INPUT: usize = 0;
    pub const OUTPUT: usize = 1;
    pub const TOP_K: usize = 2;
    pub const PARAM: usize = 3;
    pub const D: usize = 4;
    /// [`super::Op::TopK`] only.
    pub const INDICES: usize = 5;
}

/// [`OpConfig::for_batch`] splits rows below this batch.
pub const SPLIT_BELOW: usize = 16;
/// The most blocks [`OpConfig::for_batch`] gives a row.
pub const MAX_SPLIT: usize = 16;

const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// The kernel of `cfg`, one block per row.
///
/// **Filters**: a radix select of the threshold key in four passes over the row (an 8-bit digit
/// of the order-preserving key at a time, most significant first). Each pass histograms
/// the digit of the elements whose higher digits equal the prefix chosen so far (shared-memory
/// atomics: element counts for top-k, probability mass for top-p), scans the histogram from
/// the largest digit down, and picks the first bucket where the running count (mass) reaches
/// the remaining target `k` (`p`); that bucket's digit extends the prefix and the target drops
/// by what lies above it. The final prefix is the threshold key: an element is kept iff its key
/// is at least that. Renorm kernels then sum the kept probabilities and scale them; the mask
/// kernel writes `-inf` below the threshold.
///
/// **Softmax**: each thread keeps an online `(max, sum)` over its elements (no barrier in the
/// loop); the block merges them once, then a second pass writes `exp(z - max) / sum`.
pub fn op_kernel(cfg: OpConfig) -> Kernel {
    let threads = cfg.threads;
    assert!(threads.is_power_of_two() && (64..=1024).contains(&threads), "threads = {threads}");
    let nwarps = threads / 32;
    let name = format!("flashinfer_{}_t{threads}{}", cfg.op.name(), if cfg.per_row { "_rows" } else { "" });
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let input = param_ptr::<f32>("input");
        let output = param_ptr::<f32>("output");
        let row = block_idx().x;
        let (top_k, param) = if cfg.per_row {
            let ka = param_ptr::<i32>("top_k");
            let pa = param_ptr::<f32>("param");
            (ka.load_at(row), pa.load_at(row))
        } else {
            (param_scalar::<i32>("top_k"), param_scalar::<f32>("param"))
        };
        let d = param_scalar::<i32>("d");

        let tid = thread_idx().x;
        let (warp, lane) = (tid >> 5, tid & 31);
        let nt = threads as i32;
        let v = VEC as i32;
        let dv = d / v;
        let chunks = (dv + (nt - 1)) / nt;
        let row_vec = row * dv;
        let neg_inf = Float32::constant(f32::NEG_INFINITY);
        let zero = Float32::constant(0.0);

        // Vector `col` of the row (`fill` past its end).
        let load = |col: Int32, fill: Float32| -> [Float32; VEC] {
            let at = (row_vec + col.min(dv - 1)) * v;
            let x = load_vector::<f32, VEC>(&input.offset(at), LoadOptions::default());
            let inb = col.lt(dv);
            x.map(|e| select(inb, e, fill))
        };
        let store = |col: Int32, out: [Float32; VEC]| {
            if_(col.lt(dv), || {
                store_vector::<f32, VEC>(&output.offset((row_vec + col) * v), out, StoreOptions::default())
            });
        };
        // Reduce `v` over the block with `f` (a butterfly per warp, then over the warp results
        // through a fresh scratch, so no barrier is needed after).
        let block_reduce = |v: Float32, f: &dyn Fn(Float32, Float32) -> Float32, id: Float32| -> Float32 {
            let scratch = make_smem_tensor::<f32>(Layout::col_major(32)).ptr();
            let mut w = v;
            for off in [16, 8, 4, 2, 1] {
                w = f(w, shuffle_sync_bfly(w, off));
            }
            if_(lane.eq(0), || scratch.store_at(warp, w));
            sync_threads();
            let mut t = select(lane.lt(nwarps as i32), scratch.load_at(lane.min(nwarps as i32 - 1)), id);
            for off in [16, 8, 4, 2, 1] {
                t = f(t, shuffle_sync_bfly(t, off));
            }
            t
        };

        if cfg.op == Op::Softmax {
            let inv_t = select(param.eq(0.0f32), zero, Float32::constant(1.0) / param);
            let scale = inv_t * LOG2_E;
            let z_of = |col: Int32, x: [Float32; VEC]| -> [Float32; VEC] {
                let inb = col.lt(dv);
                x.map(|e| select(inb, e * scale, neg_inf))
            };
            let safe = |m: Float32| select(m.eq(neg_inf), zero, m);
            // Pass 1: per-thread online max and sum of 2^(z - max).
            let m = Var::new(neg_inf);
            let s = Var::new(zero);
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                let z = z_of(col, x);
                let mut nm = m.get();
                for e in z {
                    nm = fmax(nm, e);
                }
                let nm_safe = safe(nm);
                let mut t = s.get() * cute::math::exp2(m.get() - nm_safe, true);
                for e in z {
                    t = t + cute::math::exp2(e - nm_safe, true);
                }
                s.set(t);
                m.set(nm);
            });
            // Merge across the block: max, then the sums rescaled to it.
            let big_m = block_reduce(m.get(), &|a, b| fmax(a, b), neg_inf);
            let big_m = safe(big_m);
            let total = block_reduce(s.get() * cute::math::exp2(m.get() - big_m, true), &|a, b| a + b, zero);
            let inv = Float32::constant(1.0) / total;
            // Pass 2.
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                store(col, z_of(col, x).map(|e| cute::math::exp2(e - big_m, true) * inv));
            });
            return;
        }

        // The order-preserving key: probabilities are non-negative, so their bits already
        // order; logits need the sign fix.
        let key = |x: Float32| -> Uint32 {
            let b = x.bitcast::<u32>();
            if matches!(cfg.op, Op::TopKMask | Op::TopK) {
                b ^ select((b >> 31u32).eq(1u32), Uint32::from(0xFFFF_FFFFu32), Uint32::from(0x8000_0000u32))
            } else {
                b
            }
        };
        // Histogram weights are integers (shared-memory float atomics compile to compare-and-swap
        // loops): 1 per element for top-k, the mass in units of `2^-30` (rounded) for top-p.
        const MASS_UNIT: f32 = 1073741824.0;
        let weight = |x: Float32| -> Int32 {
            if cfg.op == Op::TopPRenorm { (x * MASS_UNIT + 0.5f32).to::<i32>() } else { Int32::from(1) }
        };

        // Lane-private histograms: bucket `b` of lane `l` is `hist[b * 33 + l]`, so the 32
        // lanes of a warp update 32 different banks whatever their digits (a shared bucket
        // would serialise the whole warp). The scan sums a bucket's 32 lane counters.
        const HIST: usize = 256 * 33;
        let hist = make_smem_tensor::<i32>(Layout::col_major(HIST)).ptr();
        let sel = make_smem_tensor::<i32>(Layout::col_major(4)).ptr();
        let above = make_smem_tensor::<f32>(Layout::col_major(4)).ptr();
        let counters = make_smem_tensor::<i32>(Layout::col_major(2)).ptr();
        if_(tid.eq(0), || {
            counters.store_at(0, 0);
            counters.store_at(1, 0);
            for r in 0..4 {
                sel.store_at(r, 0);
                above.store_at(r, 0.0f32);
            }
        });
        let prefix = Var::new(Uint32::from(0u32));
        let rem = Var::new(if cfg.op == Op::TopPRenorm { param } else { top_k.to::<f32>() });
        for r in 0..4usize {
            // Digit `r` is `(key >> shift) & 255`; the digits before it are `key >> (shift + 8)`.
            let shift = 24 - 8 * r as u32;
            for i in 0..HIST.div_ceil(threads) {
                let at = tid + (i * threads) as i32;
                if_(at.lt(HIST as i32), || hist.store_at(at, 0i32));
            }
            sync_threads();
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                let inb = col.lt(dv);
                for e in x {
                    let k = key(e);
                    let hit = if r == 0 { inb } else { inb & (k >> (shift + 8)).eq(prefix.get()) };
                    let digit = ((k >> shift) & 255u32).to::<i32>();
                    if_(hit, || {
                        let _ = atomic_add(&hist.offset(digit * 33 + lane), weight(e));
                    });
                }
            });
            sync_threads();
            // Scan the buckets from the top: thread `t` owns `bpt` consecutive ones.
            let bpt = (256 / threads).max(1);
            let top = 255 - tid * bpt as i32;
            let hs: Vec<Float32> = (0..bpt)
                .map(|j| {
                    let bucket = top - j as i32;
                    let mut t = hist.load_at(bucket.max(0) * 33);
                    for l in 1..32 {
                        t = t + hist.load_at(bucket.max(0) * 33 + l);
                    }
                    let unit = if cfg.op == Op::TopPRenorm { 1.0 / MASS_UNIT } else { 1.0 };
                    select(bucket.ge(0), t.to::<f32>() * unit, zero)
                })
                .collect();
            let mut sum = hs[0];
            for h in &hs[1..] {
                sum = sum + *h;
            }
            let incl = warp_inclusive_scan(sum, lane);
            let wt = make_smem_tensor::<f32>(Layout::col_major(32)).ptr();
            if_(lane.eq(31), || wt.store_at(warp, incl));
            sync_threads();
            let w = select(lane.lt(nwarps as i32), wt.load_at(lane.min(nwarps as i32 - 1)), zero);
            let wincl = warp_inclusive_scan(w, lane);
            let wexcl = select(warp.gt(0), shuffle_sync(wincl, (warp - 1).max(0)), zero);
            let texcl = select(lane.gt(0), shuffle_sync_up(incl, 1), zero);
            let mut c = wexcl + texcl;
            let target = rem.get();
            for (j, h) in hs.iter().enumerate() {
                let hit = c.lt(target) & (c + *h).ge(target);
                if_(hit, || {
                    sel.store_at(r as i32, top - j as i32);
                    above.store_at(r as i32, c);
                });
                c = c + *h;
            }
            sync_threads();
            prefix.set((prefix.get() << 8u32) | sel.load_at(r as i32).to::<u32>());
            rem.set(target - above.load_at(r as i32));
        }
        let kt = prefix.get();

        if cfg.op == Op::TopK {
            // The elements above the threshold, then as many of those equal to it as are still
            // needed (`rem` of them), each slot claimed with a shared-memory counter.
            let indices = param_ptr::<i32>("indices");
            let ties = rem.get().to::<i32>();
            let n_above = top_k - ties;
            let out_at = row * top_k;
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                let inb = col.lt(dv);
                for (j, &e) in x.iter().enumerate() {
                    let kk = key(e);
                    if_(inb & kk.gt(kt), || {
                        let slot = atomic_add(&counters.offset(0), 1);
                        output.store_at(out_at + slot, e);
                        indices.store_at(out_at + slot, col * v + j as i32);
                    });
                    if_(inb & kk.eq(kt), || {
                        let t = atomic_add(&counters.offset(1), 1);
                        if_(t.lt(ties), || {
                            output.store_at(out_at + n_above + t, e);
                            indices.store_at(out_at + n_above + t, col * v + j as i32);
                        });
                    });
                }
            });
        } else if cfg.op == Op::TopKMask {
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                store(col, x.map(|e| select(key(e).ge(kt), e, neg_inf)));
            });
        } else {
            let acc = Var::new(zero);
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|_, x| {
                let mut a = acc.get();
                for e in x {
                    a = a + select(key(e).ge(kt), e, zero);
                }
                acc.set(a);
            });
            let total = block_reduce(acc.get(), &|a, b| a + b, zero);
            let norm = Float32::constant(1.0) / total.max(1e-8f32);
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                store(col, x.map(|e| select(key(e).ge(kt), e * norm, zero)));
            });
        }
    })
}

// ---------------------------------------------------------------------------------------
// Multi-block rows (small batch).

/// Per-row stride of the integer workspace: four 256-bucket histograms, then the prefix
/// state of each radix round.
const WS_I: usize = 4 * 256 + 8;
/// Per-row stride of the float workspace of `split` blocks: the remaining target of each
/// round (8 slots), then the block partials (two per block).
fn ws_f(split: usize) -> usize {
    8 + 2 * split
}

/// Number of kernels a configuration runs: 1 for `split == 1` ([`op_kernel`]); else softmax
/// takes 2 (partial `(max, sum)` per block, then the write), the top-k and top-p ops 4 histogram rounds
/// plus the apply (mask) or the apply-and-sum and the scale (renorm).
pub fn op_stages(cfg: OpConfig) -> usize {
    match (cfg.split > 1, cfg.op) {
        (false, _) => 1,
        (true, Op::Softmax) => 2,
        (true, Op::TopKMask) => 5,
        (true, Op::TopK) => 5,
        (true, Op::TopKRenorm) => 6,
        (true, Op::TopPRenorm) => 6,
    }
}

/// The kernels of `cfg`, run in order on a `[split, batch]` grid (see [`op_stages`]) with the
/// arguments of [`op_kernel`] followed by `ws_i` (i32, `batch * 1032`, zero before the first
/// launch; later launches leave it zero) and `ws_f` (f32, `batch * (8 + 2 * split)`).
///
/// A block owns a contiguous slice of its row's vectors. **Softmax**: stage 0 writes each
/// block's `(max, sum)`; stage 1 merges the `split` partials (every block redundantly) and
/// writes its slice. **Top-k**: round `r` histograms its slice's digit `r` into a global
/// 256-bucket histogram with atomics; the next stage starts by scanning that histogram
/// (every block redundantly, like [`op_kernel`]'s scan) to extend the prefix and reduce the
/// target. The last scan gives the threshold key; the mask is written in the same stage, the
/// renorm sums its slice first and scales in one more.
pub fn op_kernels(cfg: OpConfig) -> Vec<Kernel> {
    (0..op_stages(cfg)).map(|s| if cfg.split > 1 { split_kernel(cfg, s) } else { op_kernel(cfg) }).collect()
}

fn block_reduce_fn(
    tid_warp: Int32,
    lane: Int32,
    nwarps: usize,
    v: Float32,
    f: &dyn Fn(Float32, Float32) -> Float32,
    id: Float32,
) -> Float32 {
    let scratch = make_smem_tensor::<f32>(Layout::col_major(32)).ptr();
    let mut w = v;
    for off in [16, 8, 4, 2, 1] {
        w = f(w, shuffle_sync_bfly(w, off));
    }
    if_(lane.eq(0), || scratch.store_at(tid_warp, w));
    sync_threads();
    let mut t = select(lane.lt(nwarps as i32), scratch.load_at(lane.min(nwarps as i32 - 1)), id);
    for off in [16, 8, 4, 2, 1] {
        t = f(t, shuffle_sync_bfly(t, off));
    }
    t
}

fn split_kernel(cfg: OpConfig, stage: usize) -> Kernel {
    let (threads, split) = (cfg.threads, cfg.split);
    assert!(threads.is_power_of_two() && (64..=1024).contains(&threads), "threads = {threads}");
    let nwarps = threads / 32;
    let name =
        format!("flashinfer_{}_t{threads}_s{split}_k{stage}{}", cfg.op.name(), if cfg.per_row { "_rows" } else { "" });
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let input = param_ptr::<f32>("input");
        let output = param_ptr::<f32>("output");
        let row = block_idx().y;
        let cta = block_idx().x;
        let (top_k, param) = if cfg.per_row {
            let ka = param_ptr::<i32>("top_k");
            let pa = param_ptr::<f32>("param");
            (ka.load_at(row), pa.load_at(row))
        } else {
            (param_scalar::<i32>("top_k"), param_scalar::<f32>("param"))
        };
        let d = param_scalar::<i32>("d");
        let indices = (cfg.op == Op::TopK).then(|| param_ptr::<i32>("indices"));
        let wsi = param_ptr::<i32>("ws_i");
        let wsf = param_ptr::<f32>("ws_f");

        let tid = thread_idx().x;
        let (warp, lane) = (tid >> 5, tid & 31);
        let nt = threads as i32;
        let v = VEC as i32;
        let dv = d / v;
        // This block's slice of the row's vectors: `[lo, lo + n)`.
        let per = (dv + (split as i32 - 1)) / split as i32;
        let lo = cta * per;
        let n = (dv - lo).min(per).max(1);
        let chunks = (n + (nt - 1)) / nt;
        let row_vec = row * dv + lo;
        let neg_inf = Float32::constant(f32::NEG_INFINITY);
        let zero = Float32::constant(0.0);
        let wi = row * WS_I as i32;
        let wf = row * ws_f(split) as i32;

        let load = |col: Int32, fill: Float32| -> [Float32; VEC] {
            let at = (row_vec + col.min(n - 1)) * v;
            let x = load_vector::<f32, VEC>(&input.offset(at), LoadOptions::default());
            let inb = col.lt(n);
            x.map(|e| select(inb, e, fill))
        };
        let store = |col: Int32, out: [Float32; VEC]| {
            if_(col.lt(n), || {
                store_vector::<f32, VEC>(&output.offset((row_vec + col) * v), out, StoreOptions::default())
            });
        };
        let reduce = |x: Float32, f: &dyn Fn(Float32, Float32) -> Float32, id: Float32| {
            block_reduce_fn(warp, lane, nwarps, x, f, id)
        };

        if cfg.op == Op::Softmax {
            let inv_t = select(param.eq(0.0f32), zero, Float32::constant(1.0) / param);
            let scale = inv_t * LOG2_E;
            let z_of = |col: Int32, x: [Float32; VEC]| -> [Float32; VEC] {
                let inb = col.lt(n);
                x.map(|e| select(inb, e * scale, neg_inf))
            };
            let safe = |m: Float32| select(m.eq(neg_inf), zero, m);
            if stage == 0 {
                let m = Var::new(neg_inf);
                let s = Var::new(zero);
                for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                    let z = z_of(col, x);
                    let mut nm = m.get();
                    for e in z {
                        nm = fmax(nm, e);
                    }
                    let nm_safe = safe(nm);
                    let mut t = s.get() * cute::math::exp2(m.get() - nm_safe, true);
                    for e in z {
                        t = t + cute::math::exp2(e - nm_safe, true);
                    }
                    s.set(t);
                    m.set(nm);
                });
                let big_m = safe(reduce(m.get(), &|a, b| fmax(a, b), neg_inf));
                let total = reduce(s.get() * cute::math::exp2(m.get() - big_m, true), &|a, b| a + b, zero);
                if_(tid.eq(0), || {
                    wsf.store_at(wf + 8 + cta * 2, big_m);
                    wsf.store_at(wf + 9 + cta * 2, total);
                });
            } else {
                let mut gm = neg_inf;
                for i in 0..split as i32 {
                    gm = fmax(gm, wsf.load_at(wf + 8 + i * 2));
                }
                let mut total = zero;
                for i in 0..split as i32 {
                    total =
                        total + wsf.load_at(wf + 9 + i * 2) * cute::math::exp2(wsf.load_at(wf + 8 + i * 2) - gm, true);
                }
                let inv = Float32::constant(1.0) / total;
                for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                    store(col, z_of(col, x).map(|e| cute::math::exp2(e - gm, true) * inv));
                });
            }
            return;
        }

        let key = |x: Float32| -> Uint32 {
            let b = x.bitcast::<u32>();
            if matches!(cfg.op, Op::TopKMask | Op::TopK) {
                b ^ select((b >> 31u32).eq(1u32), Uint32::from(0xFFFF_FFFFu32), Uint32::from(0x8000_0000u32))
            } else {
                b
            }
        };

        // Top-p histograms weigh each element by its mass in units of `2^-30` (as `op_kernel`).
        const MASS_UNIT: f32 = 1073741824.0;
        let top_p = cfg.op == Op::TopPRenorm;
        let weight =
            |x: Float32| -> Int32 { if top_p { (x * MASS_UNIT + 0.5f32).to::<i32>() } else { Int32::from(1) } };
        let target0 = if top_p { param } else { top_k.to::<f32>() };

        // Rounds 1..=4 begin by scanning the previous round's histogram.
        let (prefix, rem_left) = if stage == 0 || stage == 5 {
            (Uint32::from(0u32), zero)
        } else {
            let (p_in, rem_in) = if stage == 1 {
                (Uint32::from(0u32), target0)
            } else {
                (wsi.load_at(wi + 1024 + (stage as i32 - 1)).to::<u32>(), wsf.load_at(wf + (stage as i32 - 1)))
            };
            let hbase = wi + (stage as i32 - 1) * 256;
            let bpt = (256 / threads).max(1);
            let top = 255 - tid * bpt as i32;
            let hs: Vec<Float32> = (0..bpt)
                .map(|j| {
                    let bucket = top - j as i32;
                    let unit = if top_p { 1.0 / MASS_UNIT } else { 1.0 };
                    select(bucket.ge(0), wsi.load_at(hbase + bucket.max(0)).to::<f32>() * unit, zero)
                })
                .collect();
            let sel = make_smem_tensor::<i32>(Layout::col_major(1)).ptr();
            let above = make_smem_tensor::<f32>(Layout::col_major(1)).ptr();
            if_(tid.eq(0), || {
                sel.store_at(0, 0);
                above.store_at(0, 0.0f32);
            });
            sync_threads();
            let mut sum = hs[0];
            for h in &hs[1..] {
                sum = sum + *h;
            }
            let incl = warp_inclusive_scan(sum, lane);
            let wt = make_smem_tensor::<f32>(Layout::col_major(32)).ptr();
            if_(lane.eq(31), || wt.store_at(warp, incl));
            sync_threads();
            let w = select(lane.lt(nwarps as i32), wt.load_at(lane.min(nwarps as i32 - 1)), zero);
            let wincl = warp_inclusive_scan(w, lane);
            let wexcl = select(warp.gt(0), shuffle_sync(wincl, (warp - 1).max(0)), zero);
            let texcl = select(lane.gt(0), shuffle_sync_up(incl, 1), zero);
            let mut c = wexcl + texcl;
            for (j, h) in hs.iter().enumerate() {
                let hit = c.lt(rem_in) & (c + *h).ge(rem_in);
                if_(hit, || {
                    sel.store_at(0, top - j as i32);
                    above.store_at(0, c);
                });
                c = c + *h;
            }
            sync_threads();
            let np = (p_in << 8u32) | sel.load_at(0).to::<u32>();
            let nr = rem_in - above.load_at(0);
            if_(cta.eq(0) & tid.eq(0), || {
                wsi.store_at(wi + 1024 + stage as i32, np.to::<i32>());
                wsf.store_at(wf + stage as i32, nr);
            });
            (np, nr)
        };

        if stage < 4 {
            let r = stage;
            let shift = 24 - 8 * r as u32;
            const HIST: usize = 256 * 33;
            let hist = make_smem_tensor::<i32>(Layout::col_major(HIST)).ptr();
            for i in 0..HIST.div_ceil(threads) {
                let at = tid + (i * threads) as i32;
                if_(at.lt(HIST as i32), || hist.store_at(at, 0i32));
            }
            sync_threads();
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                let inb = col.lt(n);
                for e in x {
                    let k = key(e);
                    let hit = if r == 0 { inb } else { inb & (k >> (shift + 8)).eq(prefix) };
                    let digit = ((k >> shift) & 255u32).to::<i32>();
                    if_(hit, || {
                        let _ = atomic_add(&hist.offset(digit * 33 + lane), weight(e));
                    });
                }
            });
            sync_threads();
            // Fold the lane counters of each bucket into the global histogram.
            for i in 0..256usize.div_ceil(threads) {
                let b = tid + (i * threads) as i32;
                if_(b.lt(256), || {
                    let mut t = hist.load_at(b * 33);
                    for l in 1..32 {
                        t = t + hist.load_at(b * 33 + l);
                    }
                    if_(t.gt(0), || {
                        let _ = atomic_add(&wsi.offset(wi + r as i32 * 256 + b), t);
                    });
                });
            }
            // The slot counters of the top-k emit stage are cleared by the next launch.
            if r == 0 && cfg.op == Op::TopK {
                if_(cta.eq(0) & tid.eq(0), || {
                    wsi.store_at(wi + 1029, 0);
                    wsi.store_at(wi + 1030, 0);
                });
            }
            // Clear the histogram the next round fills (round 3 clears round 0's, for the
            // next launch).
            for i in 0..256usize.div_ceil(threads) {
                let b = tid + (i * threads) as i32;
                if_(cta.eq(0) & b.lt(256), || wsi.store_at(wi + ((r as i32 + 1) & 3) * 256 + b, 0i32));
            }
            return;
        }

        let kt = prefix;
        if let Some(indices) = indices {
            // Count this slice's elements above the threshold and equal to it, claim a range of
            // the row's two slot counters, then write them (ties beyond the `rem_left` still
            // needed are dropped).
            let (cg, ce) = (Var::new(zero), Var::new(zero));
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                let inb = col.lt(n);
                let (mut g, mut e) = (cg.get(), ce.get());
                for &val in &x {
                    let kk = key(val);
                    g = g + select(inb & kk.gt(kt), Float32::constant(1.0), zero);
                    e = e + select(inb & kk.eq(kt), Float32::constant(1.0), zero);
                }
                cg.set(g);
                ce.set(e);
            });
            let (tg, te) = (reduce(cg.get(), &|a, b| a + b, zero), reduce(ce.get(), &|a, b| a + b, zero));
            // [base_gt, base_eq, local_gt, local_eq]
            let st = make_smem_tensor::<i32>(Layout::col_major(4)).ptr();
            if_(tid.eq(0), || {
                st.store_at(0, atomic_add(&wsi.offset(wi + 1029), tg.to::<i32>()));
                st.store_at(1, atomic_add(&wsi.offset(wi + 1030), te.to::<i32>()));
                st.store_at(2, 0);
                st.store_at(3, 0);
            });
            sync_threads();
            let (base_g, base_e) = (st.load_at(0), st.load_at(1));
            let ties = rem_left.to::<i32>();
            let n_above = top_k - ties;
            let out_at = row * top_k;
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                let inb = col.lt(n);
                for (j, &val) in x.iter().enumerate() {
                    let kk = key(val);
                    let at = (lo + col) * v + j as i32;
                    if_(inb & kk.gt(kt), || {
                        let slot = base_g + atomic_add(&st.offset(2), 1);
                        output.store_at(out_at + slot, val);
                        indices.store_at(out_at + slot, at);
                    });
                    if_(inb & kk.eq(kt), || {
                        let t = base_e + atomic_add(&st.offset(3), 1);
                        if_(t.lt(ties), || {
                            output.store_at(out_at + n_above + t, val);
                            indices.store_at(out_at + n_above + t, at);
                        });
                    });
                }
            });
        } else if stage == 4 && cfg.op == Op::TopKMask {
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                store(col, x.map(|e| select(key(e).ge(kt), e, neg_inf)));
            });
        } else if stage == 4 {
            let acc = Var::new(zero);
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|_, x| {
                let mut a = acc.get();
                for e in x {
                    a = a + select(key(e).ge(kt), e, zero);
                }
                acc.set(a);
            });
            let total = reduce(acc.get(), &|a, b| a + b, zero);
            if_(tid.eq(0), || wsf.store_at(wf + 8 + cta, total));
        } else {
            let kt = wsi.load_at(wi + 1024 + 4).to::<u32>();
            let mut total = zero;
            for i in 0..split as i32 {
                total = total + wsf.load_at(wf + 8 + i);
            }
            let norm = Float32::constant(1.0) / total.max(1e-8f32);
            for_chunks(chunks, nt, tid, &|c| load(c, zero), &|col, x| {
                store(col, x.map(|e| select(key(e).ge(kt), e * norm, zero)));
            });
        }
    })
}

// ---------------------------------------------------------------------------------------
// Host side.

/// A sampling problem: `probs [batch, d]` and per-row `top_k`, `top_p`.
#[derive(Clone, Debug)]
pub struct Problem {
    pub batch: usize,
    pub d: usize,
    pub probs: Vec<f32>,
    pub top_k: Vec<i32>,
    pub top_p: Vec<f32>,
    pub seed: u64,
    pub offset: u64,
    /// Per-row seeds and offsets, used (instead of `seed`, `offset`) by [`Config::per_seed`]
    /// kernels; empty otherwise.
    pub seeds: Vec<u64>,
    pub offsets: Vec<u64>,
}

/// A seed or offset argument: the scalar, or a `[batch]` buffer of `rows` (`per_seed`).
fn seed_arg(per_seed: bool, batch: usize, scalar: u64, rows: &[u64]) -> KernelArg {
    if per_seed {
        assert_eq!(rows.len(), batch, "per-row seeds / offsets need one value per row");
        KernelArg::buffer(rows)
    } else {
        KernelArg::Scalar { ty: DType::U64, bits: scalar }
    }
}

/// SplitMix64.
fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform in `(0, 1)` from the hash of `(stream, i)`, as in `sampling_flashinfer_bench.py`.
fn hash_uniform(stream: u64, i: u64) -> f64 {
    ((splitmix(stream.wrapping_mul(0x1_0000_0001).wrapping_add(i)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// The logit distributions of FlashInfer's `bench_sampling.py`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dist {
    /// `randn * std`.
    Normal(f64),
    /// `log(-log(U + 1e-20) + 1e-20) / beta`.
    Gumbel(f64),
}

impl Dist {
    pub fn name(self) -> String {
        match self {
            Dist::Normal(s) => format!("normal({s})"),
            Dist::Gumbel(b) => format!("gumbel({b})"),
        }
    }

    /// Logit `i` of the flattened `[batch, d]` array (Box-Muller from two hashed uniforms).
    pub fn logit(self, i: u64) -> f64 {
        let u1 = hash_uniform(1, i);
        match self {
            Dist::Normal(std) => {
                let u2 = hash_uniform(2, i);
                std * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            }
            Dist::Gumbel(beta) => (-(u1 + 1e-20).ln() + 1e-20).ln() / beta,
        }
    }
}

/// Softmax of each row of `logits`, rounded to f32.
pub fn softmax_rows(logits: &[f64], d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(logits.len());
    for r in logits.chunks(d) {
        let m = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let s: f64 = r.iter().map(|&x| (x - m).exp()).sum();
        out.extend(r.iter().map(|&x| ((x - m).exp() / s) as f32));
    }
    out
}

impl Problem {
    /// Rows of softmax(`dist` logits), with scalar `top_k`, `top_p`.
    pub fn new(batch: usize, d: usize, dist: Dist, top_k: i32, top_p: f32, seed: u64) -> Problem {
        assert!(d.is_multiple_of(VEC), "d = {d} must be a multiple of {VEC}");
        let logits: Vec<f64> = (0..(batch * d) as u64).map(|i| dist.logit(i)).collect();
        Problem::from_probs(batch, d, softmax_rows(&logits, d), top_k, top_p, seed)
    }

    pub fn from_probs(batch: usize, d: usize, probs: Vec<f32>, top_k: i32, top_p: f32, seed: u64) -> Problem {
        assert_eq!(probs.len(), batch * d);
        Problem {
            batch,
            d,
            probs,
            top_k: vec![top_k; batch],
            top_p: vec![top_p; batch],
            seed,
            offset: 0,
            seeds: Vec::new(),
            offsets: Vec::new(),
        }
    }

    pub fn row(&self, b: usize) -> &[f32] {
        &self.probs[b * self.d..(b + 1) * self.d]
    }

    pub fn launch(&self, cfg: Config) -> LaunchConfig {
        LaunchConfig::new([self.batch as u32, 1, 1], [cfg.threads as u32, 1, 1])
    }

    /// The kernel's arguments (see [`p`]). With `cfg.per_row` false, every row must have the
    /// same `top_k` and `top_p`, which are passed as scalars.
    pub fn args(&self, cfg: Config) -> Vec<KernelArg> {
        let (k, pp) = if cfg.per_row {
            (KernelArg::buffer(&self.top_k), KernelArg::buffer(&self.top_p))
        } else {
            assert!(
                self.top_k.iter().all(|&k| k == self.top_k[0]) && self.top_p.iter().all(|&p| p == self.top_p[0]),
                "scalar top_k / top_p need the same value in every row"
            );
            (KernelArg::ScalarI32(self.top_k[0]), KernelArg::ScalarF32(self.top_p[0]))
        };
        vec![
            KernelArg::buffer(&self.probs),
            KernelArg::buffer(&vec![-1i32; self.batch]),
            KernelArg::buffer(&vec![-1i32; self.batch]),
            k,
            pp,
            KernelArg::ScalarI32(self.d as i32),
            seed_arg(cfg.per_seed, self.batch, self.seed, &self.seeds),
            seed_arg(cfg.per_seed, self.batch, self.offset, &self.offsets),
        ]
    }

    /// The seed of row `b`: [`Problem::seeds`] if set, else [`Problem::seed`].
    pub fn seed_of(&self, b: usize) -> u64 {
        self.seeds.get(b).copied().unwrap_or(self.seed)
    }

    /// The offset of row `b`: [`Problem::offsets`] if set, else [`Problem::offset`].
    pub fn offset_of(&self, b: usize) -> u64 {
        self.offsets.get(b).copied().unwrap_or(self.offset)
    }

    /// `min_p * max(row)` of row `b` (the `top_p` field holds `min_p` for [`Mode::MinP`]).
    pub fn min_p_pivot(&self, b: usize) -> f32 {
        self.row(b).iter().cloned().fold(0.0f32, f32::max) * self.top_p[b]
    }

    /// Whether token `t` is in row `b`'s sampling set for `mode`: positive, fewer than `k`
    /// strictly larger probabilities, and less than `p` mass strictly above it (both with
    /// `slack` on the mass, for rounding).
    pub fn in_set(&self, mode: Mode, b: usize, t: usize, slack: f64) -> bool {
        let r = self.row(b);
        let x = r[t];
        if mode == Mode::MinP {
            return x >= self.min_p_pivot(b);
        }
        if x <= 0.0 {
            return false;
        }
        let above: Vec<f32> = r.iter().cloned().filter(|&y| y > x).collect();
        let mass: f64 = above.iter().map(|&y| y as f64).sum();
        let k_ok = !mode.uses_k() || (above.len() as i64) < self.top_k[b] as i64;
        let p_ok = !mode.uses_p() || mass < self.top_p[b] as f64 + slack;
        k_ok && p_ok
    }

    /// The normalised target distribution of row `b` (the set of [`Problem::in_set`],
    /// weighted by probability).
    pub fn target(&self, mode: Mode, b: usize) -> Vec<f64> {
        let w: Vec<f64> =
            (0..self.d).map(|t| if self.in_set(mode, b, t, 0.0) { self.row(b)[t] as f64 } else { 0.0 }).collect();
        let s: f64 = w.iter().sum();
        w.iter().map(|x| x / s).collect()
    }

    /// The f64 emulation of the kernel on row `b` with the same Philox draws: the token
    /// (`None` if the row is invalid) and the smallest margin of any decision it took (the
    /// distance of `u` from the CDF at the chosen boundary, of the masses from `top_p`); a
    /// sample from the GPU or simulator may differ from it only when that margin is within
    /// rounding.
    pub fn reference(&self, mode: Mode, b: usize) -> (Option<usize>, f64) {
        let r = self.row(b);
        if mode == Mode::MinP {
            let pivot = self.min_p_pivot(b);
            let q = r.iter().filter(|&&x| x >= pivot).map(|&x| x as f64).sum::<f64>() as f32;
            let u = (curand_uniform(self.seed_of(b), b as u64, self.offset_of(b), 0) * q) as f64;
            let (mut cum, mut last) = (0.0f64, None);
            for (j, &x) in r.iter().enumerate().filter(|(_, x)| **x >= pivot) {
                let before = cum;
                cum += x as f64;
                last = Some(j);
                if cum > u {
                    return (Some(j), (cum - u).min(u - before));
                }
            }
            return (last, (u - cum).abs());
        }
        let (k, pp) = (self.top_k[b] as i64, self.top_p[b] as f64);
        let (mut q, mut low, mut high) = (1.0f32, 0.0f32, 1.0f32);
        let mut margin = f64::INFINITY;
        let mut draw = 0u64;
        loop {
            let u = (curand_uniform(self.seed_of(b), b as u64, self.offset_of(b), draw) * q) as f64;
            draw += 1;
            let mut cum = 0.0f64;
            let mut sel = None;
            let mut last = None;
            for (j, &x) in r.iter().enumerate() {
                if x > low {
                    let before = cum;
                    cum += x as f64;
                    last = Some(j);
                    if cum > u {
                        margin = margin.min(cum - u).min(u - before);
                        sel = Some(j);
                        break;
                    }
                }
            }
            let t = match sel.or(last) {
                Some(t) => t,
                None => return (None, margin),
            };
            if sel.is_none() {
                margin = margin.min(u - cum);
            }
            if mode == Mode::Plain {
                return (Some(t), margin);
            }
            let p0 = r[t];
            let p1 = (p0 + high) * 0.5;
            let (mut s0, mut s1, mut c0, mut c1) = (0.0f64, 0.0f64, 0i64, 0i64);
            for &x in r {
                if x > p0 {
                    s0 += x as f64;
                    c0 += 1;
                }
                if x > p1 {
                    s1 += x as f64;
                    c1 += 1;
                }
            }
            if mode.uses_p() {
                margin = margin.min((s0 - pp).abs()).min((s1 - pp).abs());
            }
            let ok = |s: f64, c: i64| (!mode.uses_p() || s < pp) && (!mode.uses_k() || c < k);
            if ok(s0, c0) {
                return (Some(t), margin);
            }
            if ok(s1, c1) {
                low = p0;
                high = p1;
                q = s0 as f32;
            } else {
                low = p1;
                q = s1 as f32;
            }
            if low >= high {
                return (Some(t), margin);
            }
        }
    }

    /// Check the samples (`output`, `valid`) of `mode`: each valid sample is in the sampling
    /// set, and equals the f64 emulation unless that one's margin is within `1e-5`. Returns
    /// the number of rows where they differed (all within the margin).
    pub fn check(&self, mode: Mode, output: &[i32], valid: &[i32]) -> usize {
        let mut near = 0;
        for b in 0..self.batch {
            let (want, margin) = self.reference(mode, b);
            let close = margin < 1e-5;
            let got = (valid[b] != 0).then_some(output[b] as usize);
            if got != want {
                assert!(close, "{} row {b}: sampled {got:?}, reference {want:?} (margin {margin:.2e})", mode.name());
                near += 1;
            }
            if let Some(t) = got {
                assert!(t < self.d, "row {b}: token {t} out of range");
                assert!(
                    self.in_set(mode, b, t, if close { 1e-5 } else { 0.0 }) || close,
                    "{} row {b}: token {t} (p = {}) is outside the sampling set",
                    mode.name(),
                    self.row(b)[t]
                );
            }
        }
        near
    }
}

/// Run `cfg` on `p` on the simulator; returns `(output, valid)`.
pub fn simulate(p: &Problem, cfg: Config) -> (Vec<i32>, Vec<i32>) {
    let k = kernel(cfg);
    let mut args = p.args(cfg);
    k.simulate(p.launch(cfg), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    (args[p::OUTPUT].to_vec::<i32>(), args[p::VALID].to_vec::<i32>())
}

/// Simulate and [`Problem::check`]; returns the samples.
pub fn simulate_checked(p: &Problem, cfg: Config) -> Vec<i32> {
    let (out, valid) = simulate(p, cfg);
    p.check(cfg.mode, &out, &valid);
    out
}

/// Pearson's chi-squared statistic of `counts` against `target` probabilities over the
/// cells where the target is positive, and its degrees of freedom. A count in a cell of
/// zero target probability gives infinity.
pub fn chi_squared(counts: &[usize], target: &[f64]) -> (f64, usize) {
    let n: usize = counts.iter().sum();
    let mut chi = 0.0;
    let mut cells = 0;
    for (&c, &t) in counts.iter().zip(target) {
        if t > 0.0 {
            let e = t * n as f64;
            chi += (c as f64 - e).powi(2) / e;
            cells += 1;
        } else if c > 0 {
            return (f64::INFINITY, cells);
        }
    }
    (chi, cells.saturating_sub(1))
}

/// A [`Op`] problem: `input [batch, d]`, per-row `top_k` and `param` (`top_p` or temperature).
#[derive(Clone, Debug)]
pub struct OpProblem {
    pub batch: usize,
    pub d: usize,
    pub input: Vec<f32>,
    pub top_k: Vec<i32>,
    pub param: Vec<f32>,
}

impl OpProblem {
    /// Rows of softmax(`dist` logits) (for the probability operations) or of the logits
    /// themselves (for [`Op::takes_logits`] ones), with scalar `top_k` and `param`.
    pub fn new(op: Op, batch: usize, d: usize, dist: Dist, top_k: i32, param: f32) -> OpProblem {
        assert!(d.is_multiple_of(VEC), "d = {d} must be a multiple of {VEC}");
        let logits: Vec<f64> = (0..(batch * d) as u64).map(|i| dist.logit(i)).collect();
        let input =
            if op.takes_logits() { logits.iter().map(|&x| x as f32).collect() } else { softmax_rows(&logits, d) };
        OpProblem { batch, d, input, top_k: vec![top_k; batch], param: vec![param; batch] }
    }

    pub fn row(&self, b: usize) -> &[f32] {
        &self.input[b * self.d..(b + 1) * self.d]
    }

    pub fn launch(&self, cfg: OpConfig) -> LaunchConfig {
        if cfg.split > 1 {
            assert!(self.d / VEC >= cfg.split, "d too small for {} blocks per row", cfg.split);
            return LaunchConfig::new([cfg.split as u32, self.batch as u32, 1], [cfg.threads as u32, 1, 1]);
        }
        LaunchConfig::new([self.batch as u32, 1, 1], [cfg.threads as u32, 1, 1])
    }

    /// The kernel's arguments (see [`q`]); a multi-block `cfg` ([`op_kernels`]) adds the two
    /// workspace buffers.
    pub fn args(&self, cfg: OpConfig) -> Vec<KernelArg> {
        let (k, p) = if cfg.per_row {
            (KernelArg::buffer(&self.top_k), KernelArg::buffer(&self.param))
        } else {
            assert!(
                self.top_k.iter().all(|&k| k == self.top_k[0]) && self.param.iter().all(|&p| p == self.param[0]),
                "scalar top_k / param need the same value in every row"
            );
            (KernelArg::ScalarI32(self.top_k[0]), KernelArg::ScalarF32(self.param[0]))
        };
        let out_len = if cfg.op == Op::TopK { self.batch * self.top_k[0] as usize } else { self.input.len() };
        let mut args = vec![
            KernelArg::buffer(&self.input),
            KernelArg::buffer(&vec![f32::NAN; out_len]),
            k,
            p,
            KernelArg::ScalarI32(self.d as i32),
        ];
        if cfg.op == Op::TopK {
            args.push(KernelArg::buffer(&vec![-1i32; out_len]));
        }
        if cfg.split > 1 {
            args.push(KernelArg::buffer(&vec![0i32; self.batch * WS_I]));
            args.push(KernelArg::buffer(&vec![0f32; self.batch * ws_f(cfg.split)]));
        }
        args
    }

    /// Check the `values` and `indices` ([`Op::TopK`], `[batch, k]`): each index is in the row
    /// and distinct, each value is the row's value there, and the values are exactly the `k`
    /// largest of the row (as a multiset).
    pub fn check_top_k(&self, values: &[f32], indices: &[i32]) {
        let k = self.top_k[0] as usize;
        for b in 0..self.batch {
            let x = self.row(b);
            let (v, ix) = (&values[b * k..(b + 1) * k], &indices[b * k..(b + 1) * k]);
            let mut seen = vec![false; self.d];
            for (&val, &i) in v.iter().zip(ix) {
                assert!((0..self.d as i32).contains(&i), "row {b}: index {i}");
                assert!(!std::mem::replace(&mut seen[i as usize], true), "row {b}: index {i} twice");
                assert_eq!(val.to_bits(), x[i as usize].to_bits(), "row {b}: value at {i}");
            }
            let mut want = x.to_vec();
            want.sort_by(|a, b| b.total_cmp(a));
            let mut got = v.to_vec();
            got.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(&got[..], &want[..k], "row {b}: not the {k} largest");
        }
    }

    /// The `k`-th largest value of row `b` (`None` if `k` keeps the whole row).
    fn kth(&self, b: usize) -> Option<f32> {
        let k = self.top_k[b];
        if k <= 0 || k as usize >= self.d {
            return None;
        }
        let mut r = self.row(b).to_vec();
        r.sort_by(|a, b| b.total_cmp(a));
        Some(r[k as usize - 1])
    }

    /// Check `got` (the output of `op`) against the f64 reference; returns the largest
    /// relative error (zero for the exact operations). Top-p rows where some token's mass
    /// above it is within `1e-5` of `top_p` may include or drop that token (the kernel sums
    /// in a different order); they are only checked to be consistent and to sum to one.
    pub fn check(&self, op: Op, got: &[f32]) -> f64 {
        let mut worst = 0.0f64;
        for b in 0..self.batch {
            let (x, g) = (self.row(b), &got[b * self.d..(b + 1) * self.d]);
            let fail = |i: usize, what: &str| -> ! {
                panic!("{} row {b} element {i}: {what}: input {} got {}", op.name(), x[i], g[i])
            };
            match op {
                Op::TopK => panic!("check [`Op::TopK`] with `check_top_k`"),
                Op::Softmax => {
                    let t = self.param[b] as f64;
                    let z: Vec<f64> = x.iter().map(|&v| if t == 0.0 { 0.0 } else { v as f64 / t }).collect();
                    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = z.iter().map(|&v| (v - m).exp()).collect();
                    let s: f64 = e.iter().sum();
                    for i in 0..self.d {
                        let want = e[i] / s;
                        let err = (g[i] as f64 - want).abs() / (want + 1e-12);
                        if err > 3e-5 {
                            fail(i, &format!("want {want}"));
                        }
                        worst = worst.max(err);
                    }
                }
                Op::TopKMask => {
                    let kth = self.kth(b);
                    for i in 0..self.d {
                        let want = if kth.is_none_or(|k| x[i] >= k) { x[i] } else { f32::NEG_INFINITY };
                        if g[i].to_bits() != want.to_bits() {
                            fail(i, &format!("want {want}"));
                        }
                    }
                }
                Op::TopKRenorm => {
                    let kth = self.kth(b);
                    let kept = |i: usize| kth.is_none_or(|k| x[i] >= k);
                    let s: f64 = (0..self.d).filter(|&i| kept(i)).map(|i| x[i] as f64).sum();
                    for i in 0..self.d {
                        let want = if kept(i) { x[i] as f64 / s } else { 0.0 };
                        let err = (g[i] as f64 - want).abs() / (want + 1e-12);
                        if err > 1e-5 {
                            fail(i, &format!("want {want}"));
                        }
                        worst = worst.max(err);
                    }
                }
                Op::TopPRenorm => {
                    let p = self.param[b] as f64;
                    let mut order: Vec<usize> = (0..self.d).collect();
                    order.sort_by(|&a, &c| x[c].total_cmp(&x[a]));
                    // Mass strictly above each token (equal values share it).
                    let mut mass_above = vec![0.0f64; self.d];
                    let (mut cum, mut i) = (0.0f64, 0);
                    while i < self.d {
                        let mut j = i;
                        let mut tie = 0.0;
                        while j < self.d && x[order[j]] == x[order[i]] {
                            mass_above[order[j]] = cum;
                            tie += x[order[j]] as f64;
                            j += 1;
                        }
                        cum += tie;
                        i = j;
                    }
                    let near = (0..self.d).any(|i| x[i] > 0.0 && (mass_above[i] - p).abs() < 1e-5);
                    let kept = |i: usize| x[i] > 0.0 && mass_above[i] < p;
                    let s: f64 = (0..self.d).filter(|&i| kept(i)).map(|i| x[i] as f64).sum();
                    if near {
                        let sum: f64 = g.iter().map(|&v| v as f64).sum();
                        assert!((sum - 1.0).abs() < 1e-4, "{} row {b}: sums to {sum}", op.name());
                        for i in 0..self.d {
                            let ambiguous = (mass_above[i] - p).abs() < 1e-5;
                            if !ambiguous && (g[i] > 0.0) != kept(i) {
                                fail(i, "wrong set");
                            }
                        }
                    } else {
                        for i in 0..self.d {
                            let want = if kept(i) { x[i] as f64 / s } else { 0.0 };
                            let err = (g[i] as f64 - want).abs() / (want + 1e-12);
                            if err > 1e-5 {
                                fail(i, &format!("want {want}"));
                            }
                            worst = worst.max(err);
                        }
                    }
                }
            }
        }
        worst
    }
}

/// Run `cfg` on `p` on the simulator and return the output.
pub fn simulate_op(p: &OpProblem, cfg: OpConfig) -> Vec<f32> {
    let mut args = p.args(cfg);
    for k in op_kernels(cfg) {
        k.simulate(p.launch(cfg), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    }
    args[q::OUTPUT].to_vec::<f32>()
}

/// Run [`Op::TopK`] on `p` on the simulator and check it ([`OpProblem::check_top_k`]); returns
/// `(values, indices)`.
pub fn simulate_top_k(p: &OpProblem, cfg: OpConfig) -> (Vec<f32>, Vec<i32>) {
    assert_eq!(cfg.op, Op::TopK);
    let mut args = p.args(cfg);
    for k in op_kernels(cfg) {
        k.simulate(p.launch(cfg), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    }
    let (v, i) = (args[q::OUTPUT].to_vec::<f32>(), args[q::INDICES].to_vec::<i32>());
    p.check_top_k(&v, &i);
    (v, i)
}

/// Simulate and [`OpProblem::check`]; returns the largest relative error.
pub fn simulate_op_checked(p: &OpProblem, cfg: OpConfig) -> f64 {
    p.check(cfg.op, &simulate_op(p, cfg))
}

// ---------------------------------------------------------------------------------------
// Chain speculative sampling.

/// `chain_speculative_sampling` (FlashInfer's `ChainSpeculativeSampling`, `sampling.cuh`;
/// Python `flashinfer.sampling.chain_speculative_sampling`), f32, own code after the
/// published semantics.
///
/// One block per row (a request). Draft token `i` of the row is accepted while
/// `u_i * draft_probs[i][id] < target_probs[i][id]`, `u_i` being draw `i` of the row's Philox
/// stream (`curand_init(seed, row, offset)`, then one `curand_uniform` per draw). Every warp
/// finds the first rejected position `pos` (a warp minimum, 32 draft tokens at a time) and counts
/// the later tokens that the stream's following draws would have accepted (`accepted`), without
/// any barrier. The token at `pos` is then sampled from `relu(target[pos] - draft[pos])`
/// (the bonus position `pos == n` has no draft row, so it samples `target[n]`): one pass for the
/// mass, one inverse-CDF pass with the block scan of the plain sampler. Outputs are the accepted
/// draft ids, the sampled token, then `-1`; `accepted` and `emitted` (`pos`) are added to the
/// caller's counters, as FlashInfer does.
///
/// Draw numbering follows FlashInfer: draws `0..=pos` are the first loop's; the second loop
/// then draws once for each `i` in `pos..n` (the first of those is discarded), so token `i > pos`
/// uses draw `i + 1`, and the sampling draw is `n + 1` if `pos < n`, else `n`.
pub mod spec {
    use super::*;

    /// Parameter positions of [`kernel`]: `draft_probs [batch, n, d]`, `draft_ids [batch, n]`
    /// (i32), `target_probs [batch, n + 1, d]`, `output_ids [batch, n + 1]`, `accepted [batch]`
    /// and `emitted [batch]` (i32, added to), `n` and `d` (i32, `d` a multiple of 4, `n >= 1`),
    /// `seed` and `offset` (u64).
    pub mod p {
        pub const DRAFT_PROBS: usize = 0;
        pub const DRAFT_IDS: usize = 1;
        pub const TARGET_PROBS: usize = 2;
        pub const OUTPUT_IDS: usize = 3;
        pub const ACCEPTED: usize = 4;
        pub const EMITTED: usize = 5;
        pub const N: usize = 6;
        pub const D: usize = 7;
        pub const SEED: usize = 8;
        pub const OFFSET: usize = 9;
    }

    /// The kernel with `threads` threads per block (a multiple of 32 up to 1024).
    pub fn kernel(threads: usize) -> Kernel {
        assert!((32..=1024).contains(&threads) && threads.is_multiple_of(32), "threads = {threads}");
        let nwarps = threads / 32;
        Kernel::builder(format!("flashinfer_chain_speculative_sampling_t{threads}"))
            .block_dim([threads as u32, 1, 1])
            .build(|| {
                let draft = param_ptr::<f32>("draft_probs");
                let ids = param_ptr::<i32>("draft_ids");
                let target = param_ptr::<f32>("target_probs");
                let out = param_ptr::<i32>("output_ids");
                let accepted = param_ptr::<i32>("accepted");
                let emitted = param_ptr::<i32>("emitted");
                let n = param_scalar::<i32>("n");
                let d = param_scalar::<i32>("d");
                let seed = param_scalar::<u64>("seed");
                let offset = param_scalar::<u64>("offset");
                let row = block_idx().x;
                let sub = row.to::<u64>();
                let tid = thread_idx().x;
                let (warp, lane) = (tid >> 5, tid & 31);
                let nt = threads as i32;
                let v = VEC as i32;
                let dv = d / v;
                let chunks = (dv + (nt - 1)) / nt;
                let zero = Float32::constant(0.0);

                // One lane per draft token `i`, 32 at a time: its draft id, `p` and `q` are
                // gathered once and tested with draw `i` (the first loop of FlashInfer's kernel)
                // and draw `i + 1` (its second loop, for the tokens after the first rejection).
                // Groups are visited in order, so a group's lanes beyond the first rejection
                // found so far (this group's or an earlier one's) are exactly the ones that
                // count in the second loop.
                let groups = (n + 31) / 32;
                let pos_var = Var::new(n);
                let extra = Var::new(Int32::from(0));
                for_range(0, groups, 1, |g| {
                    let i = g * 32 + lane;
                    let valid = i.lt(n);
                    let ic = i.min(n - 1);
                    let id = ids.load_at(row * n + ic);
                    let q = target.load_at((row * (n + 1) + ic) * d + id);
                    let p = draft.load_at((row * n + ic) * d + id);
                    let u0 = uniform_device(seed, sub, offset + i.to::<u64>());
                    let u1 = uniform_device(seed, sub, offset + (i + 1).to::<u64>());
                    let rej = select(valid & !(u0 * p).lt(q), i, n);
                    let pos_now = pos_var.get().min(warp_redux_sync(rej, ReduxKind::Min));
                    pos_var.set(pos_now);
                    let hit = select(valid & i.gt(pos_now) & (u1 * p).lt(q), Int32::from(1), Int32::from(0));
                    extra.set(extra.get() + warp_redux_sync(hit, ReduxKind::Add));
                });
                let pos = pos_var.get();
                if_(tid.eq(0), || {
                    accepted.store_at(row, accepted.load_at(row) + pos + extra.get());
                    emitted.store_at(row, emitted.load_at(row) + pos);
                });

                // Sample from `relu(target[pos] - draft[pos])`.
                let has_p = pos.lt(n);
                let q_row = (row * (n + 1) + pos) * dv;
                let p_row = (row * n + pos.min(n - 1)) * dv;
                let load = |col: Int32| -> [Float32; VEC] {
                    let c = col.min(dv - 1);
                    let q = load_vector::<f32, VEC>(&target.offset((q_row + c) * v), LoadOptions::default());
                    // The bonus position has no draft row: read one (cached) vector, not the row.
                    let pc = select(has_p, c, Int32::from(0));
                    let pv = load_vector::<f32, VEC>(&draft.offset((p_row + pc) * v), LoadOptions::default());
                    let inb = col.lt(dv);
                    std::array::from_fn(|j| {
                        let r = fmax(q[j] - select(has_p, pv[j], zero), zero);
                        select(inb, r, zero)
                    })
                };
                let red = make_smem_tensor::<f32>(Layout::col_major(64)).ptr();
                let cand = make_smem_tensor::<i32>(Layout::col_major(32)).ptr();
                let lastv = make_smem_tensor::<i32>(Layout::col_major(32)).ptr();
                let pass = Pass { d, chunks, nt, nwarps, tid, warp, lane, red: &red, cand: &cand };
                let part = Var::new(zero);
                for_chunks(chunks, nt, tid, &load, &|_, x| {
                    let mut a = part.get();
                    for e in x {
                        a = a + e;
                    }
                    part.set(a);
                });
                let mass = block_reduce_fn(warp, lane, nwarps, part.get(), &|a, b| a + b, zero);
                let u =
                    uniform_device(seed, sub, offset + (n + select(has_p, Int32::from(1), Int32::from(0))).to::<u64>())
                        * mass;
                let (sel, last) = pass.inverse_cdf(&load, &|x| x.gt(0.0f32), u);
                // `u` very close to 1 (rounding left the mass short of it): the last positive entry.
                let tok = Var::new(sel);
                if_(sel.eq(d), || tok.set(pass.block_min_max(last, &lastv, true)));
                let t = select(tok.get().ge(0) & tok.get().lt(d), tok.get(), Int32::from(0));

                // Outputs: accepted draft ids, the sampled token, then -1.
                for_range(0, (n + nt) / nt, 1, |g| {
                    let i = g * nt + tid;
                    if_(i.lt(n + 1), || {
                        let draft_id = ids.load_at(row * n + i.min(n - 1));
                        out.store_at(
                            row * (n + 1) + i,
                            select(i.lt(pos), draft_id, select(i.eq(pos), t, Int32::from(-1))),
                        );
                    });
                });
            })
    }

    /// A speculative sampling problem.
    #[derive(Clone, Debug)]
    pub struct Problem {
        pub batch: usize,
        /// Draft tokens per request.
        pub n: usize,
        pub d: usize,
        /// `[batch, n, d]`.
        pub draft_probs: Vec<f32>,
        /// `[batch, n]`, sampled from the draft probabilities.
        pub draft_ids: Vec<i32>,
        /// `[batch, n + 1, d]`.
        pub target_probs: Vec<f32>,
        pub seed: u64,
        pub offset: u64,
    }

    impl Problem {
        /// Draft and target rows are softmaxes of independent `dist` logits (the draft with
        /// std/beta `draft_dist`, the target `dist`); draft ids are drawn from the draft rows.
        pub fn new(batch: usize, n: usize, d: usize, draft_dist: Dist, dist: Dist, seed: u64) -> Problem {
            assert!(d.is_multiple_of(VEC) && n >= 1);
            let rows = |count: usize, dist: Dist, salt: u64| -> Vec<f32> {
                let logits: Vec<f64> = (0..(count * d) as u64).map(|i| dist.logit(i + salt)).collect();
                softmax_rows(&logits, d)
            };
            let draft_probs = rows(batch * n, draft_dist, 1 << 40);
            let target_probs = rows(batch * (n + 1), dist, 0);
            let draft_ids = (0..batch * n)
                .map(|r| {
                    let row = &draft_probs[r * d..(r + 1) * d];
                    let x = hash_uniform(7, r as u64 + seed);
                    let mut c = 0.0f64;
                    let mut pick = d - 1;
                    for (i, &p) in row.iter().enumerate() {
                        c += p as f64;
                        if c > x {
                            pick = i;
                            break;
                        }
                    }
                    pick as i32
                })
                .collect();
            Problem { batch, n, d, draft_probs, draft_ids, target_probs, seed, offset: 0 }
        }

        pub fn launch(&self, threads: usize) -> LaunchConfig {
            LaunchConfig::new([self.batch as u32, 1, 1], [threads as u32, 1, 1])
        }

        /// The kernel's arguments (see [`p`]); the counters start at `accepted0` / `emitted0`.
        pub fn args(&self, accepted0: i32, emitted0: i32) -> Vec<KernelArg> {
            vec![
                KernelArg::buffer(&self.draft_probs),
                KernelArg::buffer(&self.draft_ids),
                KernelArg::buffer(&self.target_probs),
                KernelArg::buffer(&vec![-2i32; self.batch * (self.n + 1)]),
                KernelArg::buffer(&vec![accepted0; self.batch]),
                KernelArg::buffer(&vec![emitted0; self.batch]),
                KernelArg::ScalarI32(self.n as i32),
                KernelArg::ScalarI32(self.d as i32),
                KernelArg::Scalar { ty: DType::U64, bits: self.seed },
                KernelArg::Scalar { ty: DType::U64, bits: self.offset },
            ]
        }

        /// `(draft probability, target probability)` of draft token `i` of row `b`.
        fn pq(&self, b: usize, i: usize) -> (f32, f32) {
            let id = self.draft_ids[b * self.n + i] as usize;
            (self.draft_probs[(b * self.n + i) * self.d + id], self.target_probs[(b * (self.n + 1) + i) * self.d + id])
        }

        /// The reference of row `b`: `(first rejected position, accepted count)`.
        pub fn reference(&self, b: usize) -> (usize, usize) {
            let u = |draw: usize| curand_uniform(self.seed, b as u64, self.offset, draw as u64);
            let accepts = |i: usize, draw: usize| {
                let (p, q) = self.pq(b, i);
                u(draw) * p < q
            };
            let pos = (0..self.n).find(|&i| !accepts(i, i)).unwrap_or(self.n);
            let extra = (pos + 1..self.n).filter(|&i| accepts(i, i + 1)).count();
            (pos, pos + extra)
        }

        /// The distribution the token at `pos` of row `b` is drawn from: `relu(target - draft)`
        /// (unnormalised), and the draw `u * mass` that picks it.
        fn residual(&self, b: usize, pos: usize) -> (Vec<f64>, f64) {
            let t = &self.target_probs[(b * (self.n + 1) + pos) * self.d..][..self.d];
            let r: Vec<f64> = (0..self.d)
                .map(|j| {
                    let dp = if pos < self.n { self.draft_probs[(b * self.n + pos) * self.d + j] } else { 0.0 };
                    (t[j] - dp).max(0.0) as f64
                })
                .collect();
            let draw = self.n + usize::from(pos < self.n);
            let u = curand_uniform(self.seed, b as u64, self.offset, draw as u64) as f64;
            let mass: f64 = r.iter().sum();
            (r, u * mass)
        }

        /// Check the kernel's outputs: positions, counts and the accepted prefix exactly, the
        /// padding, and that the resampled token is the one the draw selects (up to rounding of
        /// the running sum: its interval of the cumulative sum must contain the draw within
        /// `1e-4` of the mass).
        pub fn check(&self, out: &[i32], accepted: &[i32], emitted: &[i32], accepted0: i32, emitted0: i32) {
            for b in 0..self.batch {
                let (pos, acc) = self.reference(b);
                assert_eq!(emitted[b], emitted0 + pos as i32, "row {b}: emitted");
                assert_eq!(accepted[b], accepted0 + acc as i32, "row {b}: accepted");
                let o = &out[b * (self.n + 1)..(b + 1) * (self.n + 1)];
                assert_eq!(&o[..pos], &self.draft_ids[b * self.n..b * self.n + pos], "row {b}: accepted prefix");
                assert!(o[pos + 1..].iter().all(|&t| t == -1), "row {b}: padding {o:?}");
                let (r, x) = self.residual(b, pos);
                let t = o[pos] as usize;
                assert!(t < self.d && r[t] > 0.0, "row {b}: token {t} outside the support");
                let before: f64 = r[..t].iter().sum();
                let tol = 1e-4 * r.iter().sum::<f64>();
                assert!(
                    before <= x + tol && before + r[t] >= x - tol,
                    "row {b}: token {t} for draw {x}: [{before}, {})",
                    before + r[t]
                );
            }
        }

        /// The normalised distribution of the resampled token at `pos` of row `b`.
        pub fn residual_distribution(&self, b: usize, pos: usize) -> Vec<f64> {
            let (r, _) = self.residual(b, pos);
            let s: f64 = r.iter().sum();
            r.iter().map(|x| x / s).collect()
        }
    }

    /// Run [`kernel`] on `p` on the simulator; returns `(output_ids, accepted, emitted)` with
    /// the counters starting from 3 and 5.
    pub fn simulate(p: &Problem, threads: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let k = kernel(threads);
        let mut args = p.args(3, 5);
        k.simulate(p.launch(threads), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
        (args[p::OUTPUT_IDS].to_vec::<i32>(), args[p::ACCEPTED].to_vec::<i32>(), args[p::EMITTED].to_vec::<i32>())
    }

    /// Simulate and [`Problem::check`]; returns the outputs.
    pub fn simulate_checked(p: &Problem, threads: usize) -> Vec<i32> {
        let (out, acc, em) = simulate(p, threads);
        p.check(&out, &acc, &em, 3, 5);
        out
    }

    /// Running on the GPU (`--features cuda`).
    #[cfg(feature = "cuda")]
    pub mod gpu {
        use std::error::Error;

        use cute::cuda::{CudaContext, DeviceArg};
        use cute::prelude::*;
        use cute::testing::{BenchmarkOptions, benchmark};

        use super::{Problem, kernel, p};

        /// Run on `p` once and return `(output_ids, accepted, emitted)` (counters from 3 and
        /// 5); if `time`, also the average GPU time in microseconds from CUDA-graph replays.
        #[allow(clippy::type_complexity)]
        pub fn run(
            ctx: &CudaContext,
            tc: &Toolchain,
            p: &Problem,
            threads: usize,
            time: bool,
        ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>, Option<f64>), Box<dyn Error>> {
            let k = kernel(threads);
            let arch = ctx.best_target(tc).ok_or("no target for this GPU")?;
            let f = ctx.load(&k.compile_with(tc, &CompileOptions::new().gpu_arch(arch))?)?;
            let host = p.args(3, 5);
            let is_scalar = |a: &KernelArg| matches!(a, KernelArg::ScalarI32(_) | KernelArg::Scalar { .. });
            let mut bufs = Vec::new();
            for a in &host {
                if !is_scalar(a) {
                    bufs.push(ctx.upload(a)?);
                }
            }
            let mut bi = bufs.iter();
            let args: Vec<DeviceArg> =
                host.iter()
                    .map(|a| {
                        if is_scalar(a) { DeviceArg::Scalar(a.clone()) } else { DeviceArg::Buffer(bi.next().unwrap()) }
                    })
                    .collect();
            let stream = ctx.stream()?;
            let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
                ctx.launch_builder(&k, &f, p.launch(threads)).stream(s).args(&args).submit()?;
                Ok(())
            };
            launch(&stream, 0)?;
            stream.synchronize()?;
            // Buffers 0..=2 are inputs; the outputs follow (scalars are not buffers).
            let out = bufs[p::OUTPUT_IDS].to_vec::<i32>()?;
            let acc = bufs[p::ACCEPTED].to_vec::<i32>()?;
            let em = bufs[p::EMITTED].to_vec::<i32>()?;
            let t = if time {
                let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
                Some(benchmark(&stream, &opts, &mut launch)?)
            } else {
                None
            };
            Ok((out, acc, em, t))
        }
    }
}

// ---------------------------------------------------------------------------------------
// The compact top-k-first path for logits.

/// Threads of the compact kernels (one block per row).
pub const COMPACT_THREADS: usize = 128;

/// `k` rounded up to the sampler's vector width: the row length of the compact probabilities.
pub fn compact_width(k: usize) -> usize {
    k.next_multiple_of(VEC)
}

/// The softmax of the `k` logits of each row (`vals [batch, k]`, from [`Op::TopK`]) written as
/// `probs [batch, kp]` (`kp = compact_width(k)`, zero past `k`), the `softmax(top_k_mask_logits)`
/// that `top_k_top_p_sampling_from_logits(filter_apply_order="top_k_first")` samples from, over
/// `k` values instead of the whole vocabulary. Parameters `vals`, `probs`, `k`, `kp` (i32); one
/// block of [`COMPACT_THREADS`] per row. Own code.
pub fn compact_softmax_kernel() -> Kernel {
    let threads = COMPACT_THREADS;
    Kernel::builder("flashinfer_compact_softmax").block_dim([threads as u32, 1, 1]).build(|| {
        let vals = param_ptr::<f32>("vals");
        let probs = param_ptr::<f32>("probs");
        let k = param_scalar::<i32>("k");
        let kp = param_scalar::<i32>("kp");
        let row = block_idx().x;
        let tid = thread_idx().x;
        let (warp, lane) = (tid >> 5, tid & 31);
        let nwarps = threads / 32;
        let (zero, neg_inf) = (Float32::constant(0.0), Float32::constant(f32::NEG_INFINITY));
        let iters = (kp + (threads as i32 - 1)) / threads as i32;
        let get = |i: Int32| select(i.lt(k), vals.load_at(row * k + i.min(k - 1)), neg_inf);
        let m = Var::new(neg_inf);
        for_range(0, iters, 1, |it| m.set(fmax(m.get(), get(it * threads as i32 + tid))));
        let big_m = block_reduce_fn(warp, lane, nwarps, m.get(), &|a, b| fmax(a, b), neg_inf);
        let big_m = select(big_m.eq(neg_inf), zero, big_m);
        let s = Var::new(zero);
        for_range(0, iters, 1, |it| {
            let z = (get(it * threads as i32 + tid) - big_m) * LOG2_E;
            s.set(s.get() + cute::math::exp2(z, true));
        });
        let total = block_reduce_fn(warp, lane, nwarps, s.get(), &|a, b| a + b, zero);
        let inv = select(total.eq(0.0f32), zero, Float32::constant(1.0) / total);
        for_range(0, iters, 1, |it| {
            let i = it * threads as i32 + tid;
            let z = (get(i) - big_m) * LOG2_E;
            if_(i.lt(kp), || probs.store_at(row * kp + i, cute::math::exp2(z, true) * inv));
        });
    })
}

/// Map each row's local sample back to its vocabulary token: `out[row] = indices[row * k +
/// local[row]]` (0 if the row was invalid). Parameters `indices [batch, k]`, `local`, `valid`,
/// `out` (i32) and `k`; one block of 32 threads per row.
pub fn gather_kernel() -> Kernel {
    Kernel::builder("flashinfer_gather_token").block_dim([32, 1, 1]).build(|| {
        let indices = param_ptr::<i32>("indices");
        let local = param_ptr::<i32>("local");
        let valid = param_ptr::<i32>("valid");
        let out = param_ptr::<i32>("out");
        let k = param_scalar::<i32>("k");
        let row = block_idx().x;
        if_(thread_idx().x.eq(0), || {
            let t = local.load_at(row).max(0).min(k - 1);
            let tok = indices.load_at(row * k + t);
            out.store_at(row, select(valid.load_at(row).ne(0), tok, Int32::from(0)));
        });
    })
}

// ---------------------------------------------------------------------------------------
// Top-k with an index transform (`top_k_page_table_transform`, `top_k_ragged_transform`).

/// `flashinfer.top_k_page_table_transform` and `top_k_ragged_transform` (`topk.py`; the kernels
/// behind them are `RadixTopKPageTableTransform` / `RadixTopKRaggedTransform` of
/// `include/flashinfer/topk.cuh`), f32 scores, own code after the published semantics: nothing
/// was translated, and the BSD-licensed `topk_varlen` kernels were not read.
///
/// Row `i` of `input [rows, stride]` scores the window `[row_starts[i], row_starts[i] +
/// lengths[i])`. The `k` largest scores of the window (ties at the k-th: some of them) give
/// local indices `idx` (relative to `row_starts[i]`), unsorted, written to `output [rows, k]`:
///
/// * [`Kind::PageTable`]: `table[b, pt_row_starts[i] + (idx >> s)] << s | (idx & (2^s - 1))`
///   with `b = row_to_batch[i]` and page size `2^s`;
/// * [`Kind::Ragged`]: `idx + offsets[i]`.
///
/// A row with `lengths[i] <= k` selects every position in order, and positions from `lengths[i]`
/// on are `-1` (unmapped). The selection is the radix select of [`Op::TopK`] (four 8-bit rounds
/// over the order-preserving key, lane-private shared-memory histograms folded into a global one
/// per row), as the pipeline of [`op_kernels`]: stage `r < 4` scans round `r - 1`'s histogram then
/// histograms digit `r`; stage 4 scans the last and emits the elements above the threshold and as
/// many ties as are needed. A row is spread over `split` blocks (`grid [split, rows]`), which is
/// what fills the GPU at small batch. Scores are read with scalar loads (a window may start at any
/// element), four per thread in flight.
pub mod transform {
    use super::*;

    /// How selected indices are mapped.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Kind {
        PageTable,
        Ragged,
    }

    /// Compile-time configuration.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct TConfig {
        pub kind: Kind,
        /// Threads per block (power of two, 64 to 1024).
        pub threads: usize,
        /// Blocks per row.
        pub split: usize,
        /// `log2` of the page size ([`Kind::PageTable`]).
        pub page_shift: u32,
    }

    impl TConfig {
        pub fn new(kind: Kind) -> TConfig {
            TConfig { kind, threads: 1024, split: 1, page_shift: 0 }
        }
        pub fn threads(self, threads: usize) -> TConfig {
            TConfig { threads, ..self }
        }
        pub fn split(self, split: usize) -> TConfig {
            TConfig { split, ..self }
        }
        pub fn page_shift(self, page_shift: u32) -> TConfig {
            TConfig { page_shift, ..self }
        }
        /// The best configuration found on an RTX 2000 Ada for `rows` rows of at most `max_len`
        /// scores: rows are spread over blocks while there are fewer than [`SPLIT_BELOW`] of them
        /// (at least 4096 scores per block, at most [`MAX_SPLIT`] blocks); rows of 8192 or fewer
        /// scores use 256 threads (512 for 32 rows or more below 32768) (one block per row fuses all the stages into one kernel, and its
        /// fixed costs, the histogram clears and scans, dominate there).
        pub fn for_rows(kind: Kind, rows: usize, max_len: usize) -> TConfig {
            let split = (SPLIT_BELOW / rows.max(1)).min(max_len / 4096).clamp(1, MAX_SPLIT);
            let threads = if max_len <= 8192 {
                256
            } else if max_len < 32768 && rows >= 32 {
                512
            } else {
                1024
            };
            TConfig::new(kind).threads(threads).split(split)
        }
    }

    /// Kernel stages: four histogram rounds, then the emit.
    pub const STAGES: usize = 5;
    /// Per-row stride of the float workspace (the remaining target of each round).
    pub const WS_F: usize = 8;

    /// Scores loaded together per thread.
    const GROUP: usize = 4;

    /// The kernels of `cfg`, in launch order; each runs on a `[split, rows]` grid.
    ///
    /// Parameters, in order: `input` (f32 `[rows, stride]`), `output` (i32 `[rows, k]`),
    /// `lengths`, `row_starts` (i32 `[rows]`); for [`Kind::PageTable`] `table` (i32 `[batch,
    /// pt_stride]`), `row_to_batch` and `pt_row_starts` (`[rows]`), for [`Kind::Ragged`]
    /// `offsets` (`[rows]`); then scalars `top_k`, `stride` (and `pt_stride` for page tables)
    /// and the workspaces `ws_i` (i32 `rows * 1032`, zero before the first launch) and `ws_f`
    /// (f32 `rows * 8`).
    pub fn kernels(cfg: TConfig) -> Vec<Kernel> {
        if cfg.split == 1 {
            vec![stage_kernel(cfg, None)]
        } else {
            (0..STAGES).map(|s| stage_kernel(cfg, Some(s))).collect()
        }
    }

    /// Number of kernels [`kernels`] returns: one for a single block per row (all the stages in
    /// one kernel, with block barriers between them), else [`STAGES`].
    pub fn num_kernels(cfg: TConfig) -> usize {
        if cfg.split == 1 { 1 } else { STAGES }
    }

    /// Shared-memory words of the lane-private histogram: bucket `b` of lane `l` is `b * 33 + l`.
    const HIST: usize = 256 * 33;

    /// Stage `stage` of the pipeline, or (`None`, one block per row) all of them.
    fn stage_kernel(cfg: TConfig, which: Option<usize>) -> Kernel {
        let fused = which.is_none();
        let (threads, split) = (cfg.threads, cfg.split);
        assert!(threads.is_power_of_two() && (64..=1024).contains(&threads), "threads = {threads}");
        let nwarps = threads / 32;
        let page = cfg.kind == Kind::PageTable;
        let name = format!(
            "flashinfer_top_k_{}_t{threads}_s{split}_p{}_k{}",
            if page { "page_table" } else { "ragged" },
            cfg.page_shift,
            which.map_or("all".to_string(), |s| s.to_string())
        );
        Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
            let input = param_ptr::<f32>("input");
            let output = param_ptr::<i32>("output");
            let lengths = param_ptr::<i32>("lengths");
            let row_starts = param_ptr::<i32>("row_starts");
            let table = param_ptr::<i32>("table");
            let (row_to_batch, pt_starts) = if page {
                (Some(param_ptr::<i32>("row_to_batch")), Some(param_ptr::<i32>("pt_row_starts")))
            } else {
                (None, None)
            };
            let top_k = param_scalar::<i32>("top_k");
            let stride = param_scalar::<i32>("stride");
            let pt_stride = page.then(|| param_scalar::<i32>("pt_stride"));
            let wsi = param_ptr::<i32>("ws_i");
            let wsf = param_ptr::<f32>("ws_f");

            let row = block_idx().y;
            let cta = block_idx().x;
            let tid = thread_idx().x;
            let (warp, lane) = (tid >> 5, tid & 31);
            let nt = threads as i32;
            let zero = Float32::constant(0.0);

            let len = lengths.load_at(row);
            let start = row_starts.load_at(row);
            // Rows that keep everything: positions are emitted in order, no selection.
            let keep_all = len.le(top_k);
            // This block's slice `[lo, lo + n)` of the row's window.
            let per = (len + (split as i32 - 1)) / split as i32;
            let lo = cta * per;
            let n = (len - lo).min(per).max(0);
            let n_hist = select(keep_all, Int32::from(0), n);
            let base = row * stride + start + lo;
            let wi = row * WS_I as i32;
            let wf = row * WS_F as i32;

            let key = |x: Float32| -> Uint32 {
                let b = x.bitcast::<u32>();
                b ^ select((b >> 31u32).eq(1u32), Uint32::from(0xFFFF_FFFFu32), Uint32::from(0x8000_0000u32))
            };
            // Visit this block's `nn` scores, [`GROUP`] loads in flight per thread:
            // `f(local index, in range, score)`.
            let for_scores = |nn: Int32, f: &dyn Fn(Int32, Boolean, Float32)| {
                let step = nt * GROUP as i32;
                let groups = (nn + (step - 1)) / step;
                for_range(0, groups, 1, |g| {
                    let ix: Vec<Int32> = (0..GROUP).map(|u| (g * GROUP as i32 + u as i32) * nt + tid).collect();
                    let xs: Vec<Float32> = ix.iter().map(|i| input.load_at(base + i.min(nn - 1).max(0))).collect();
                    for (i, x) in ix.into_iter().zip(xs) {
                        f(i, i.lt(nn), x);
                    }
                });
            };
            let reduce = |x: Float32, f: &dyn Fn(Float32, Float32) -> Float32, id: Float32| {
                block_reduce_fn(warp, lane, nwarps, x, f, id)
            };

            let hist =
                (fused || which.is_some_and(|s| s < 4)).then(|| make_smem_tensor::<i32>(Layout::col_major(HIST)).ptr());
            // Unmapped positions past the window.
            let pad = || {
                let pads = (top_k + (nt - 1)) / nt;
                for_range(0, pads, 1, |c| {
                    let j = c * nt + tid;
                    if_(cta.eq(0) & j.lt(top_k) & j.ge(len), || output.store_at(row * top_k + j, Int32::from(-1)));
                });
            };
            let selecting = len.gt(top_k);

            let run = |stage: usize| {
                // Rounds 1..=4 begin by scanning the previous round's histogram.
                let (prefix, rem_left) = if stage == 0 {
                    (Uint32::from(0u32), zero)
                } else {
                    let target0 = top_k.to::<f32>();
                    let (p_in, rem_in) = if stage == 1 {
                        (Uint32::from(0u32), target0)
                    } else {
                        (wsi.load_at(wi + 1024 + (stage as i32 - 1)).to::<u32>(), wsf.load_at(wf + (stage as i32 - 1)))
                    };
                    let hbase = wi + (stage as i32 - 1) * 256;
                    let bpt = (256 / threads).max(1);
                    let top = 255 - tid * bpt as i32;
                    let hs: Vec<Float32> = (0..bpt)
                        .map(|j| {
                            let bucket = top - j as i32;
                            // One block reads what its own atomics wrote: read at the L2, not the L1.
                            let at = hbase + bucket.max(0);
                            let count = if fused { atomic_add(&wsi.offset(at), 0i32) } else { wsi.load_at(at) };
                            select(bucket.ge(0), count.to::<f32>(), zero)
                        })
                        .collect();
                    let sel = make_smem_tensor::<i32>(Layout::col_major(1)).ptr();
                    let above = make_smem_tensor::<f32>(Layout::col_major(1)).ptr();
                    if_(tid.eq(0), || {
                        sel.store_at(0, 0);
                        above.store_at(0, 0.0f32);
                    });
                    sync_threads();
                    let mut sum = hs[0];
                    for h in &hs[1..] {
                        sum = sum + *h;
                    }
                    let incl = warp_inclusive_scan(sum, lane);
                    let wt = make_smem_tensor::<f32>(Layout::col_major(32)).ptr();
                    if_(lane.eq(31), || wt.store_at(warp, incl));
                    sync_threads();
                    let w = select(lane.lt(nwarps as i32), wt.load_at(lane.min(nwarps as i32 - 1)), zero);
                    let wincl = warp_inclusive_scan(w, lane);
                    let wexcl = select(warp.gt(0), shuffle_sync(wincl, (warp - 1).max(0)), zero);
                    let texcl = select(lane.gt(0), shuffle_sync_up(incl, 1), zero);
                    let mut c = wexcl + texcl;
                    for (j, h) in hs.iter().enumerate() {
                        let hit = c.lt(rem_in) & (c + *h).ge(rem_in);
                        if_(hit, || {
                            sel.store_at(0, top - j as i32);
                            above.store_at(0, c);
                        });
                        c = c + *h;
                    }
                    sync_threads();
                    let np = (p_in << 8u32) | sel.load_at(0).to::<u32>();
                    let nr = rem_in - above.load_at(0);
                    if_(cta.eq(0) & tid.eq(0), || {
                        wsi.store_at(wi + 1024 + stage as i32, np.to::<i32>());
                        wsf.store_at(wf + stage as i32, nr);
                    });
                    (np, nr)
                };

                if stage < 4 {
                    let r = stage;
                    let shift = 24 - 8 * r as u32;
                    let hist = hist.as_ref().unwrap();
                    for i in 0..HIST.div_ceil(threads) {
                        let at = tid + (i * threads) as i32;
                        if_(at.lt(HIST as i32), || hist.store_at(at, 0i32));
                    }
                    sync_threads();
                    for_scores(n_hist, &|_, inb, x| {
                        let k = key(x);
                        let hit = if r == 0 { inb } else { inb & (k >> (shift + 8)).eq(prefix) };
                        let digit = ((k >> shift) & 255u32).to::<i32>();
                        if_(hit, || {
                            let _ = atomic_add(&hist.offset(digit * 33 + lane), 1i32);
                        });
                    });
                    sync_threads();
                    for i in 0..256usize.div_ceil(threads) {
                        let b = tid + (i * threads) as i32;
                        if_(b.lt(256), || {
                            let mut t = hist.load_at(b * 33);
                            for l in 1..32 {
                                t = t + hist.load_at(b * 33 + l);
                            }
                            if_(t.gt(0), || {
                                let _ = atomic_add(&wsi.offset(wi + r as i32 * 256 + b), t);
                            });
                        });
                    }
                    // The slot counters of the emit stage are cleared by the next launch.
                    if r == 0 {
                        if_(cta.eq(0) & tid.eq(0), || {
                            wsi.store_at(wi + 1029, 0);
                            wsi.store_at(wi + 1030, 0);
                        });
                    }
                    // Clear the histogram the next round fills (round 3 clears round 0's).
                    for i in 0..256usize.div_ceil(threads) {
                        let b = tid + (i * threads) as i32;
                        if_(cta.eq(0) & b.lt(256), || wsi.store_at(wi + ((r as i32 + 1) & 3) * 256 + b, 0i32));
                    }
                } else {
                    // Emit: map the local index of a selected score to its output value.
                    let (pt_base, batch_pt) = if page {
                        let b = row_to_batch.as_ref().unwrap().load_at(row);
                        let ps = pt_starts.as_ref().unwrap().load_at(row);
                        (b * pt_stride.unwrap() + ps, Int32::from(0))
                    } else {
                        (Int32::from(0), table.load_at(row))
                    };
                    let mapped = |idx: Int32| -> Int32 {
                        if page {
                            let mask = (1i32 << cfg.page_shift) - 1;
                            let p = table.load_at(pt_base + (idx >> cfg.page_shift as i32));
                            (p << cfg.page_shift as i32) | (idx & mask)
                        } else {
                            idx + batch_pt
                        }
                    };
                    let kt = prefix;
                    let out_at = row * top_k;
                    let ties = rem_left.to::<i32>();
                    let n_above = top_k - ties;
                    // [base_gt, base_eq, local_gt, local_eq]
                    let st = make_smem_tensor::<i32>(Layout::col_major(4)).ptr();
                    if split > 1 {
                        // Count this slice's elements above the threshold and equal to it, then claim a
                        // range of the row's two slot counters.
                        let (cg, ce) = (Var::new(zero), Var::new(zero));
                        for_scores(n_hist, &|_, inb, x| {
                            let kk = key(x);
                            cg.set(cg.get() + select(inb & kk.gt(kt), Float32::constant(1.0), zero));
                            ce.set(ce.get() + select(inb & kk.eq(kt), Float32::constant(1.0), zero));
                        });
                        let (tg, te) = (reduce(cg.get(), &|a, b| a + b, zero), reduce(ce.get(), &|a, b| a + b, zero));
                        if_(tid.eq(0), || {
                            st.store_at(0, atomic_add(&wsi.offset(wi + 1029), tg.to::<i32>()));
                            st.store_at(1, atomic_add(&wsi.offset(wi + 1030), te.to::<i32>()));
                            st.store_at(2, 0);
                            st.store_at(3, 0);
                        });
                    } else {
                        if_(tid.eq(0), || {
                            for s in 0..4 {
                                st.store_at(s, 0);
                            }
                        });
                    }
                    sync_threads();
                    let (base_g, base_e) = (st.load_at(0), st.load_at(1));
                    for_scores(n, &|i, inb, x| {
                        let kk = key(x);
                        let idx = lo + i;
                        if_(inb & keep_all, || output.store_at(out_at + idx, mapped(idx)));
                        if_(inb & selecting & kk.gt(kt), || {
                            let slot = base_g + atomic_add(&st.offset(2), 1);
                            output.store_at(out_at + slot, mapped(idx));
                        });
                        if_(inb & selecting & kk.eq(kt), || {
                            let t = base_e + atomic_add(&st.offset(3), 1);
                            if_(t.lt(ties), || output.store_at(out_at + n_above + t, mapped(idx)));
                        });
                    });
                }
            };
            match which {
                Some(stage) => {
                    if stage == 0 {
                        pad();
                    }
                    run(stage);
                }
                None => {
                    pad();
                    if_(selecting, || {
                        for stage in 0..4 {
                            run(stage);
                            sync_threads();
                        }
                    });
                    run(4);
                }
            }
        })
    }

    /// A transform problem: scores `[rows, stride]`, per-row windows, the page table or offsets.
    #[derive(Clone, Debug)]
    pub struct TProblem {
        pub kind: Kind,
        pub rows: usize,
        pub stride: usize,
        pub k: usize,
        pub input: Vec<f32>,
        pub lengths: Vec<i32>,
        pub row_starts: Vec<i32>,
        /// Page tables: `[batch, pt_stride]` (injective values), `row_to_batch` and
        /// `pt_row_starts` (`[rows]`); `page_shift` is `log2` of the page size.
        pub table: Vec<i32>,
        pub pt_stride: usize,
        pub row_to_batch: Vec<i32>,
        pub pt_row_starts: Vec<i32>,
        pub page_shift: u32,
        /// Ragged: the offsets `[rows]`.
        pub offsets: Vec<i32>,
    }

    impl TProblem {
        /// Scores from `dist` on rows of `stride`, windows `[row_starts[i], + lengths[i])`.
        /// Page tables get `batch` sequences with `row_to_batch[i] = i % batch` and page-table
        /// starts `row_starts >> page_shift` (so the windows stay inside the table); the table
        /// entries are distinct, so a mapped index identifies its position.
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            kind: Kind,
            stride: usize,
            k: usize,
            dist: Dist,
            lengths: Vec<i32>,
            row_starts: Vec<i32>,
            batch: usize,
            page_shift: u32,
        ) -> TProblem {
            let rows = lengths.len();
            assert_eq!(row_starts.len(), rows);
            for (&l, &s) in lengths.iter().zip(&row_starts) {
                assert!(l >= 0 && s >= 0 && (l + s) as usize <= stride, "window {s}+{l} outside {stride}");
            }
            let input: Vec<f32> = (0..(rows * stride) as u64).map(|i| dist.logit(i) as f32).collect();
            let pt_stride = (stride >> page_shift) + 2;
            let table: Vec<i32> = (0..(batch * pt_stride) as i32).map(|i| i * 7 + 3).collect();
            TProblem {
                kind,
                rows,
                stride,
                k,
                input,
                row_to_batch: (0..rows as i32).map(|i| i % batch as i32).collect(),
                pt_row_starts: row_starts.iter().map(|&s| s >> page_shift).collect(),
                table,
                pt_stride,
                page_shift,
                offsets: (0..rows as i32).map(|i| i * 1000 + 5).collect(),
                lengths,
                row_starts,
            }
        }

        pub fn launch(&self, cfg: TConfig) -> LaunchConfig {
            LaunchConfig::new([cfg.split as u32, self.rows as u32, 1], [cfg.threads as u32, 1, 1])
        }

        /// The arguments of [`kernels`] (the output starts at `-2`).
        pub fn args(&self) -> Vec<KernelArg> {
            let mut a = vec![
                KernelArg::buffer(&self.input),
                KernelArg::buffer(&vec![-2i32; self.rows * self.k]),
                KernelArg::buffer(&self.lengths),
                KernelArg::buffer(&self.row_starts),
            ];
            match self.kind {
                Kind::PageTable => {
                    a.push(KernelArg::buffer(&self.table));
                    a.push(KernelArg::buffer(&self.row_to_batch));
                    a.push(KernelArg::buffer(&self.pt_row_starts));
                }
                Kind::Ragged => a.push(KernelArg::buffer(&self.offsets)),
            }
            a.push(KernelArg::ScalarI32(self.k as i32));
            a.push(KernelArg::ScalarI32(self.stride as i32));
            if self.kind == Kind::PageTable {
                a.push(KernelArg::ScalarI32(self.pt_stride as i32));
            }
            a.push(KernelArg::buffer(&vec![0i32; self.rows * WS_I]));
            a.push(KernelArg::buffer(&vec![0f32; self.rows * WS_F]));
            a
        }

        /// Position of the output buffer among [`TProblem::args`].
        pub const OUTPUT: usize = 1;

        /// The score window of row `i`.
        pub fn window(&self, i: usize) -> &[f32] {
            let s = i * self.stride + self.row_starts[i] as usize;
            &self.input[s..s + self.lengths[i] as usize]
        }

        /// What local index `idx` of row `i` maps to.
        pub fn map(&self, i: usize, idx: usize) -> i32 {
            match self.kind {
                Kind::Ragged => idx as i32 + self.offsets[i],
                Kind::PageTable => {
                    let at = self.row_to_batch[i] as usize * self.pt_stride
                        + self.pt_row_starts[i] as usize
                        + (idx >> self.page_shift);
                    (self.table[at] << self.page_shift) | (idx & ((1 << self.page_shift) - 1)) as i32
                }
            }
        }

        /// Check `out` (`[rows, k]`): a row of at most `k` scores maps every position in order and
        /// pads with `-1`; a longer row yields `k` distinct valid positions whose scores are
        /// exactly the `k` largest of its window (as a multiset).
        pub fn check(&self, out: &[i32]) {
            let k = self.k;
            for i in 0..self.rows {
                let (x, o) = (self.window(i), &out[i * k..(i + 1) * k]);
                let len = x.len();
                if len <= k {
                    for (j, &v) in o.iter().enumerate() {
                        let want = if j < len { self.map(i, j) } else { -1 };
                        assert_eq!(v, want, "row {i} position {j} (len {len})");
                    }
                    continue;
                }
                let back: std::collections::HashMap<i32, usize> = (0..len).map(|j| (self.map(i, j), j)).collect();
                assert_eq!(back.len(), len, "mapping not injective");
                let mut seen = vec![false; len];
                let mut got = Vec::with_capacity(k);
                for &v in o {
                    let j = *back.get(&v).unwrap_or_else(|| panic!("row {i}: output {v} is not a mapped index"));
                    assert!(!std::mem::replace(&mut seen[j], true), "row {i}: index {j} twice");
                    got.push(x[j]);
                }
                let mut want = x.to_vec();
                want.sort_by(|a, b| b.total_cmp(a));
                got.sort_by(|a, b| b.total_cmp(a));
                assert_eq!(&got[..], &want[..k], "row {i}: not the {k} largest of {len}");
            }
        }
    }

    /// Run `cfg` on the simulator and check the output; returns it.
    pub fn simulate_checked(p: &TProblem, cfg: TConfig) -> Vec<i32> {
        let mut args = p.args();
        for k in kernels(cfg) {
            k.simulate(p.launch(cfg), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
        }
        let out = args[TProblem::OUTPUT].to_vec::<i32>();
        p.check(&out);
        out
    }
}

/// Running on the GPU (`--features cuda`).
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::collections::HashMap;
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{
        Config, FilterOrder, LogitsProblem, MAX_SPLIT, Mode, Op, OpConfig, OpProblem, Problem, VEC, WS_I, kernel, l,
        logits_kernel, op_kernels, transform, ws_f,
    };

    /// Kernels compiled and loaded on first use, per configuration.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: &'c Toolchain,
        loaded: HashMap<Config, (Kernel, CudaFunction<'c>)>,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &'c Toolchain) -> Kernels<'c> {
            Kernels { ctx, tc, loaded: HashMap::new() }
        }

        pub fn get(&mut self, cfg: Config) -> Result<&(Kernel, CudaFunction<'c>), Box<dyn Error>> {
            if !self.loaded.contains_key(&cfg) {
                let arch = self.ctx.best_target(self.tc).ok_or("no target for this GPU")?;
                let k = kernel(cfg);
                let f = self.ctx.load(&k.compile_with(self.tc, &CompileOptions::new().gpu_arch(arch))?)?;
                self.loaded.insert(cfg, (k, f));
            }
            Ok(&self.loaded[&cfg])
        }

        /// The kernel of `cfg` that [`Kernels::get`] loaded.
        pub fn loaded(&self, cfg: Config) -> &(Kernel, CudaFunction<'c>) {
            &self.loaded[&cfg]
        }
    }

    /// Run `cfg` on `p` once and return `(output, valid)`; if `time`, also the average GPU
    /// time in microseconds of launches replayed from a CUDA graph.
    #[allow(clippy::type_complexity)]
    pub fn run(
        ctx: &CudaContext,
        ks: &mut Kernels<'_>,
        p: &Problem,
        cfg: Config,
        time: bool,
    ) -> Result<(Vec<i32>, Vec<i32>, Option<f64>), Box<dyn Error>> {
        let (k, f) = ks.get(cfg)?;
        let host = p.args(cfg);
        let is_scalar =
            |a: &KernelArg| matches!(a, KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) | KernelArg::Scalar { .. });
        let mut bufs = Vec::new();
        for a in &host {
            if !is_scalar(a) {
                bufs.push(ctx.upload(a)?);
            }
        }
        let mut bi = bufs.iter();
        let args: Vec<DeviceArg> = host
            .iter()
            .map(|a| if is_scalar(a) { DeviceArg::Scalar(a.clone()) } else { DeviceArg::Buffer(bi.next().unwrap()) })
            .collect();
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(k, f, p.launch(cfg)).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let out = bufs[super::p::OUTPUT].to_vec::<i32>()?;
        let valid = bufs[super::p::VALID].to_vec::<i32>()?;
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((out, valid, t))
    }

    /// [`Kernels`] for [`OpConfig`]s.
    pub struct OpKernels<'c> {
        ctx: &'c CudaContext,
        tc: &'c Toolchain,
        loaded: HashMap<OpConfig, Vec<(Kernel, CudaFunction<'c>)>>,
    }

    impl<'c> OpKernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &'c Toolchain) -> OpKernels<'c> {
            OpKernels { ctx, tc, loaded: HashMap::new() }
        }

        pub fn get(&mut self, cfg: OpConfig) -> Result<&[(Kernel, CudaFunction<'c>)], Box<dyn Error>> {
            if !self.loaded.contains_key(&cfg) {
                let arch = self.ctx.best_target(self.tc).ok_or("no target for this GPU")?;
                let mut all = Vec::new();
                for k in op_kernels(cfg) {
                    let f = self.ctx.load(&k.compile_with(self.tc, &CompileOptions::new().gpu_arch(arch))?)?;
                    all.push((k, f));
                }
                self.loaded.insert(cfg, all);
            }
            Ok(&self.loaded[&cfg])
        }

        /// The kernels of `cfg` that [`OpKernels::get`] loaded.
        pub fn loaded(&self, cfg: OpConfig) -> &[(Kernel, CudaFunction<'c>)] {
            &self.loaded[&cfg]
        }
    }

    /// Run `cfg` on `p` once and return the output (and, for [`Op::TopK`], the indices; else an
    /// empty vector); if `time`, also the average GPU time in microseconds of launches replayed
    /// from a CUDA graph.
    #[allow(clippy::type_complexity)]
    pub fn run_op(
        ctx: &CudaContext,
        ks: &mut OpKernels<'_>,
        p: &OpProblem,
        cfg: OpConfig,
        time: bool,
    ) -> Result<(Vec<f32>, Vec<i32>, Option<f64>), Box<dyn Error>> {
        let kernels = ks.get(cfg)?;
        let host = p.args(cfg);
        let is_scalar = |a: &KernelArg| matches!(a, KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_));
        let mut bufs = Vec::new();
        for a in &host {
            if !is_scalar(a) {
                bufs.push(ctx.upload(a)?);
            }
        }
        let mut bi = bufs.iter();
        let args: Vec<DeviceArg> = host
            .iter()
            .map(|a| if is_scalar(a) { DeviceArg::Scalar(a.clone()) } else { DeviceArg::Buffer(bi.next().unwrap()) })
            .collect();
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            for (k, f) in kernels {
                ctx.launch_builder(k, f, p.launch(cfg)).stream(s).args(&args).submit()?;
            }
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let out = bufs[1].to_vec::<f32>()?;
        let indices = if cfg.op == Op::TopK { bufs[super::q::INDICES - 3].to_vec::<i32>()? } else { Vec::new() };
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((out, indices, t))
    }

    /// [`Kernels`] for the top-k transforms ([`transform::TConfig`]).
    pub struct TransformKernels<'c> {
        ctx: &'c CudaContext,
        tc: &'c Toolchain,
        loaded: HashMap<transform::TConfig, Vec<(Kernel, CudaFunction<'c>)>>,
    }

    impl<'c> TransformKernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &'c Toolchain) -> TransformKernels<'c> {
            TransformKernels { ctx, tc, loaded: HashMap::new() }
        }

        pub fn get(&mut self, cfg: transform::TConfig) -> Result<&[(Kernel, CudaFunction<'c>)], Box<dyn Error>> {
            if !self.loaded.contains_key(&cfg) {
                let arch = self.ctx.best_target(self.tc).ok_or("no target for this GPU")?;
                let mut all = Vec::new();
                for k in transform::kernels(cfg) {
                    let f = self.ctx.load(&k.compile_with(self.tc, &CompileOptions::new().gpu_arch(arch))?)?;
                    all.push((k, f));
                }
                self.loaded.insert(cfg, all);
            }
            Ok(&self.loaded[&cfg])
        }
    }

    /// Run a top-k transform once and return the output `[rows, k]`; if `time`, also the average
    /// GPU time in microseconds of launches replayed from a CUDA graph.
    pub fn run_transform(
        ctx: &CudaContext,
        ks: &mut TransformKernels<'_>,
        p: &transform::TProblem,
        cfg: transform::TConfig,
        time: bool,
    ) -> Result<(Vec<i32>, Option<f64>), Box<dyn Error>> {
        let kernels = ks.get(cfg)?;
        let host = p.args();
        let is_scalar = |a: &KernelArg| matches!(a, KernelArg::ScalarI32(_));
        let mut bufs = Vec::new();
        for a in &host {
            if !is_scalar(a) {
                bufs.push(ctx.upload(a)?);
            }
        }
        let mut bi = bufs.iter();
        let args: Vec<DeviceArg> = host
            .iter()
            .map(|a| if is_scalar(a) { DeviceArg::Scalar(a.clone()) } else { DeviceArg::Buffer(bi.next().unwrap()) })
            .collect();
        let launch_cfg = p.launch(cfg);
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            for (k, f) in kernels {
                ctx.launch_builder(k, f, launch_cfg).stream(s).args(&args).submit()?;
            }
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let out = bufs[transform::TProblem::OUTPUT].to_vec::<i32>()?;
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((out, t))
    }

    /// Run `top_k_top_p_sampling_from_probs` (`from_logits` false) or `..._from_logits` on `p`
    /// (its `probs` field holds logits when `from_logits`), in `order`, as the pipeline of
    /// FlashInfer's own fallback: [`Op`] kernels then a sampler, all launched into one stream.
    /// `per_row` selects per-row `top_k` / `top_p`. Returns `(tokens, valid, time)`; `time`
    /// (microseconds) replays the whole chain from a CUDA graph.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn run_filtered(
        ctx: &CudaContext,
        ks: &mut Kernels<'_>,
        oks: &mut OpKernels<'_>,
        p: &Problem,
        from_logits: bool,
        order: FilterOrder,
        per_row: bool,
        time: bool,
    ) -> Result<(Vec<i32>, Vec<i32>, Option<f64>), Box<dyn Error>> {
        let (b, d) = (p.batch, p.d);
        let up = |a: &KernelArg| ctx.upload(a);
        let (inp, t1, t2) = (up(&KernelArg::buffer(&p.probs))?, ctx.alloc::<f32>(b * d)?, ctx.alloc::<f32>(b * d)?);
        let (tokens, valid) = (up(&KernelArg::buffer(&vec![-1i32; b]))?, up(&KernelArg::buffer(&vec![-1i32; b]))?);
        let (kbuf, pbuf, obuf) = (
            up(&KernelArg::buffer(&p.top_k))?,
            up(&KernelArg::buffer(&p.top_p))?,
            up(&KernelArg::buffer(&vec![1.0f32; b]))?,
        );
        let ws_i = up(&KernelArg::buffer(&vec![0i32; b * WS_I]))?;
        let ws_f_buf = up(&KernelArg::buffer(&vec![0f32; b * ws_f(MAX_SPLIT)]))?;
        let ws_i2 = up(&KernelArg::buffer(&vec![0i32; b * WS_I]))?;
        let ws_f2 = up(&KernelArg::buffer(&vec![0f32; b * ws_f(MAX_SPLIT)]))?;
        let scalar_k = || -> DeviceArg<'_> {
            if per_row { DeviceArg::Buffer(&kbuf) } else { DeviceArg::Scalar(KernelArg::ScalarI32(p.top_k[0])) }
        };
        let scalar_p = || -> DeviceArg<'_> {
            if per_row { DeviceArg::Buffer(&pbuf) } else { DeviceArg::Scalar(KernelArg::ScalarF32(p.top_p[0])) }
        };
        let temp = || -> DeviceArg<'_> {
            if per_row { DeviceArg::Buffer(&obuf) } else { DeviceArg::Scalar(KernelArg::ScalarF32(1.0)) }
        };
        let sd = || DeviceArg::Scalar(KernelArg::ScalarI32(d as i32));
        let (sampler_mode, ops): (Mode, Vec<Op>) = match (from_logits, order) {
            (false, FilterOrder::Joint) => (Mode::TopKTopP, vec![]),
            (false, FilterOrder::TopKFirst) => (Mode::TopP, vec![Op::TopKRenorm]),
            (true, FilterOrder::Joint) => (Mode::TopKTopP, vec![Op::Softmax]),
            (true, FilterOrder::TopKFirst) => (Mode::TopP, vec![Op::TopKMask, Op::Softmax]),
        };
        let scfg = Config::new(sampler_mode).per_row(per_row);
        ks.get(scfg)?;
        let cfgs: Vec<OpConfig> = ops.iter().map(|&op| OpConfig::for_batch(op, b).per_row(per_row)).collect();
        for &c in &cfgs {
            oks.get(c)?;
        }
        // Chain: input -> t1 -> t2; the last stage's output is the sampler's input.
        let bufs = [&inp, &t1, &t2];
        let op_args: Vec<Vec<DeviceArg<'_>>> = cfgs
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let param = if c.op == Op::Softmax { temp() } else { scalar_p() };
                let mut a = vec![DeviceArg::Buffer(bufs[i]), DeviceArg::Buffer(bufs[i + 1]), scalar_k(), param, sd()];
                if c.split > 1 {
                    let (wi, wf) = if i == 0 { (&ws_i, &ws_f_buf) } else { (&ws_i2, &ws_f2) };
                    a.push(DeviceArg::Buffer(wi));
                    a.push(DeviceArg::Buffer(wf));
                }
                a
            })
            .collect();
        let sampler_in = bufs[ops.len()];
        let sampler_args = vec![
            DeviceArg::Buffer(sampler_in),
            DeviceArg::Buffer(&tokens),
            DeviceArg::Buffer(&valid),
            scalar_k(),
            scalar_p(),
            sd(),
            DeviceArg::Scalar(KernelArg::Scalar { ty: DType::U64, bits: p.seed }),
            DeviceArg::Scalar(KernelArg::Scalar { ty: DType::U64, bits: p.offset }),
        ];
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            for (c, args) in cfgs.iter().zip(&op_args) {
                let launch =
                    OpProblem { batch: b, d, input: Vec::new(), top_k: Vec::new(), param: Vec::new() }.launch(*c);
                for (k, f) in oks.loaded(*c) {
                    ctx.launch_builder(k, f, launch).stream(s).args(args).submit()?;
                }
            }
            let (k, f) = ks.loaded(scfg);
            ctx.launch_builder(k, f, p.launch(scfg)).stream(s).args(&sampler_args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let (out, ok) = (tokens.to_vec::<i32>()?, valid.to_vec::<i32>()?);
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((out, ok, t))
    }

    /// `top_k_top_p_sampling_from_logits(filter_apply_order="top_k_first")` by FlashInfer's fast
    /// path: [`Op::TopK`] selects the `k` largest logits (unsorted, with their indices), a compact
    /// softmax and the top-p sampler run over those `k` only, and the sample is mapped back to its
    /// token. Scalar `k` (at most 1024) and `top_p` (`p.top_k[0]`, `p.top_p[0]`); ties at the k-th logit keep exactly `k` tokens (some of them).
    /// Returns `(tokens, valid, time)`.
    #[allow(clippy::type_complexity)]
    pub fn run_top_k_first_compact(
        ctx: &CudaContext,
        ks: &mut Kernels<'_>,
        oks: &mut OpKernels<'_>,
        tc: &Toolchain,
        p: &Problem,
        time: bool,
    ) -> Result<(Vec<i32>, Vec<i32>, Option<f64>), Box<dyn Error>> {
        let (b, d) = (p.batch, p.d);
        let k = p.top_k[0] as usize;
        assert!((1..=1024).contains(&k) && k <= d, "k = {k}");
        let kp = super::compact_width(k);
        let arch = ctx.best_target(tc).ok_or("no target for this GPU")?;
        let opts = CompileOptions::new().gpu_arch(arch);
        let (csk, gk) = (super::compact_softmax_kernel(), super::gather_kernel());
        let (csf, gf) = (ctx.load(&csk.compile_with(tc, &opts)?)?, ctx.load(&gk.compile_with(tc, &opts)?)?);
        let tcfg = OpConfig::for_batch(Op::TopK, b);
        oks.get(tcfg)?;
        let threads = (kp / VEC).next_power_of_two().clamp(32, 1024);
        let scfg = Config::new(Mode::TopP).threads(threads);
        ks.get(scfg)?;
        let up = |a: &KernelArg| ctx.upload(a);
        let inp = up(&KernelArg::buffer(&p.probs))?;
        let (vals, idx) = (ctx.alloc::<f32>(b * k)?, up(&KernelArg::buffer(&vec![-1i32; b * k]))?);
        let probs_k = ctx.alloc::<f32>(b * kp)?;
        let (local, valid) = (up(&KernelArg::buffer(&vec![-1i32; b]))?, up(&KernelArg::buffer(&vec![-1i32; b]))?);
        let tokens = up(&KernelArg::buffer(&vec![-1i32; b]))?;
        let ws_i = up(&KernelArg::buffer(&vec![0i32; b * WS_I]))?;
        let ws_f_buf = up(&KernelArg::buffer(&vec![0f32; b * ws_f(MAX_SPLIT)]))?;
        let sc = |x: usize| DeviceArg::Scalar(KernelArg::ScalarI32(x as i32));
        let mut top_args = vec![
            DeviceArg::Buffer(&inp),
            DeviceArg::Buffer(&vals),
            sc(k),
            DeviceArg::Scalar(KernelArg::ScalarF32(0.0)),
            sc(d),
            DeviceArg::Buffer(&idx),
        ];
        if tcfg.split > 1 {
            top_args.push(DeviceArg::Buffer(&ws_i));
            top_args.push(DeviceArg::Buffer(&ws_f_buf));
        }
        let cs_args = [DeviceArg::Buffer(&vals), DeviceArg::Buffer(&probs_k), sc(k), sc(kp)];
        let sampler_args = [
            DeviceArg::Buffer(&probs_k),
            DeviceArg::Buffer(&local),
            DeviceArg::Buffer(&valid),
            DeviceArg::Scalar(KernelArg::ScalarI32(0)),
            DeviceArg::Scalar(KernelArg::ScalarF32(p.top_p[0])),
            sc(kp),
            DeviceArg::Scalar(KernelArg::Scalar { ty: DType::U64, bits: p.seed }),
            DeviceArg::Scalar(KernelArg::Scalar { ty: DType::U64, bits: p.offset }),
        ];
        let g_args = [
            DeviceArg::Buffer(&idx),
            DeviceArg::Buffer(&local),
            DeviceArg::Buffer(&valid),
            DeviceArg::Buffer(&tokens),
            sc(k),
        ];
        let top_launch =
            OpProblem { batch: b, d, input: Vec::new(), top_k: Vec::new(), param: Vec::new() }.launch(tcfg);
        let rows = LaunchConfig::new([b as u32, 1, 1], [super::COMPACT_THREADS as u32, 1, 1]);
        let gather = LaunchConfig::new([b as u32, 1, 1], [32, 1, 1]);
        let sampler_launch = LaunchConfig::new([b as u32, 1, 1], [threads as u32, 1, 1]);
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            for (kn, f) in oks.loaded(tcfg) {
                ctx.launch_builder(kn, f, top_launch).stream(s).args(&top_args).submit()?;
            }
            ctx.launch_builder(&csk, &csf, rows).stream(s).args(&cs_args).submit()?;
            let (sk, sf) = ks.loaded(scfg);
            ctx.launch_builder(sk, sf, sampler_launch).stream(s).args(&sampler_args).submit()?;
            ctx.launch_builder(&gk, &gf, gather).stream(s).args(&g_args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let (out, ok) = (tokens.to_vec::<i32>()?, valid.to_vec::<i32>()?);
        let t = if time {
            let o = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &o, &mut launch)?)
        } else {
            None
        };
        Ok((out, ok, t))
    }

    /// Run [`logits_kernel`] on `p` once and return the tokens; if `time`, also the average GPU
    /// time in microseconds of launches replayed from a CUDA graph.
    pub fn run_logits(
        ctx: &CudaContext,
        cfg: Config,
        tc: &Toolchain,
        p: &LogitsProblem,
        time: bool,
    ) -> Result<(Vec<i32>, Option<f64>), Box<dyn Error>> {
        let arch = ctx.best_target(tc).ok_or("no target for this GPU")?;
        let k = logits_kernel(cfg);
        let f = ctx.load(&k.compile_with(tc, &CompileOptions::new().gpu_arch(arch))?)?;
        let host = p.args(cfg);
        let is_scalar = |a: &KernelArg| matches!(a, KernelArg::ScalarI32(_) | KernelArg::Scalar { .. });
        let mut bufs = Vec::new();
        for a in &host {
            if !is_scalar(a) {
                bufs.push(ctx.upload(a)?);
            }
        }
        let mut bi = bufs.iter();
        let args: Vec<DeviceArg> = host
            .iter()
            .map(|a| if is_scalar(a) { DeviceArg::Scalar(a.clone()) } else { DeviceArg::Buffer(bi.next().unwrap()) })
            .collect();
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(&k, &f, p.launch(cfg)).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let out = bufs[l::OUTPUT].to_vec::<i32>()?;
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((out, t))
    }
}
