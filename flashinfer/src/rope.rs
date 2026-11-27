// SPDX-License-Identifier: Apache-2.0

//! Rotary position embedding with a cos/sin cache, and the append of new tokens to a paged
//! KV cache, after FlashInfer's `BatchQKApplyRotaryPosIdsCosSinCache`
//! (`include/flashinfer/pos_enc.cuh`, Python `flashinfer.apply_rope_with_cos_sin_cache_inplace`)
//! and `AppendPagedKVCache` (`include/flashinfer/page.cuh`, `flashinfer.append_paged_kv_cache`).
//! It follows FlashInfer's design (thread geometry, vector
//! widths, the launch planner).
//!
//! # RoPE
//!
//! `query [nnz, num_q_heads * head_dim]` and `key [nnz, num_k_heads * head_dim]` are rotated in
//! place by the cache row of each token's position: `cos_sin_cache [max_pos, head_dim]` holds,
//! as f32, `head_dim / 2` cosines and then `head_dim / 2` sines. `rotary_dim = head_dim`.
//! * NeoX (`is_neox`): element `i` pairs with `i + head_dim / 2`;
//! * GPT-J (interleaved): element `2j` pairs with `2j + 1`.
//!
//! Geometry (f16/bf16, head dim 64 or 128): `VEC = 8` elements per thread (16 bytes),
//! `BDX = head_dim / 8` threads per head, `max(128, BDX) / BDX` tokens per block (`BDY`), one
//! token per `ty`. A thread loads its token's cos/sin once, then rotates its `VEC` elements of
//! every head. Three launch shapes, as FlashInfer's:
//! * [`Variant::HeadParallel`]: grid `(token blocks, q + k heads)`, one head per block, when
//!   the token blocks do not fill the GPU;
//! * [`Variant::Unsplit`]: grid `(token blocks, 1)`, each block loops over every head;
//! * [`Variant::Split`]: grid `(token blocks, g)`, each block loops over every `g`-th head
//!   (`g` chosen by FlashInfer's wave-cost model, [`plan`]).
//!
//! # Paged KV append
//!
//! [`append_kernel`]: tokens `i < nnz` of `append_k`/`append_v [nnz, num_kv_heads, head_dim]`
//! go to request `batch_indices[i]`, position `positions[i]` (see
//! [`batch_indices_positions`]), i.e. entry `pos % page_size` of page
//! `kv_indices[kv_indptr[b] + pos / page_size]`, in an NHD cache
//! `[num_pages, 2, page_size, num_kv_heads, head_dim]` (K then V of each page). Block
//! `(head_dim / 8, num_kv_heads)`, a persistent grid of `occupancy * SMs` blocks (capped by
//! the token count) that stride over the tokens, 16 bytes per thread per K and V. As in
//! FlashInfer, inputs and cache share one element type ([`CacheElem`]): Float16, BFloat16,
//! or an FP8 cache (E4M3, E5M2) taking FP8 inputs, 16 per thread.
//!
//! # Frequency RoPE
//!
//! [`freq_kernel`] is `apply_rope_pos_ids_inplace` / `apply_llama31_rope_pos_ids_inplace`:
//! frequencies computed in the kernel, `rotary_dim <= head_dim`. The `(indptr, offsets)`
//! forms (`apply_rope_inplace`, `apply_llama31_rope_inplace`) are the same kernel with the
//! positions expanded by [`pos_ids_from_indptr`].
//!
//! Not covered: strided (non-contiguous) q/k, head dims other
//! than 64/128 for RoPE, the `pos_ids`-free and quantising variants, HND layout.

use cute::prelude::*;

/// Elements per thread (16 bytes of a 16-bit float).
pub const VEC: usize = 8;

/// The 16-bit floating-point element types.
pub trait Half: Numeric + Copy {
    const NAME: &'static str;
    /// The tolerance of one rounding of a value of magnitude 2 or less.
    const TOL: f32;
    fn from_f32(x: f32) -> Self;
    fn to_f32(self) -> f32;
    /// The element of the 16 stored bits.
    fn from_bits16(bits: u16) -> Self;
}

impl Half for F16 {
    const NAME: &'static str = "f16";
    const TOL: f32 = 2.5e-3;
    fn from_f32(x: f32) -> F16 {
        F16::from_f32(x)
    }
    fn to_f32(self) -> f32 {
        F16::to_f32(self)
    }
    fn from_bits16(bits: u16) -> F16 {
        F16(bits)
    }
}

impl Half for BF16 {
    const NAME: &'static str = "bf16";
    const TOL: f32 = 2e-2;
    fn from_f32(x: f32) -> BF16 {
        BF16::from_f32(x)
    }
    fn to_f32(self) -> f32 {
        BF16::to_f32(self)
    }
    fn from_bits16(bits: u16) -> BF16 {
        BF16(bits)
    }
}

/// The element types of a paged KV cache for the append: the [`Half`] types and FP8
/// (E4M3, E5M2).
pub trait CacheElem: Numeric + Copy {
    const CACHE_NAME: &'static str;
    fn from_f32_rounded(x: f32) -> Self;
    fn value(self) -> f32;
    /// The element of its little-endian bytes (`Numeric::BYTES` of them).
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

impl<T: Half> CacheElem for T {
    const CACHE_NAME: &'static str = T::NAME;
    fn from_f32_rounded(x: f32) -> T {
        T::from_f32(x)
    }
    fn value(self) -> f32 {
        self.to_f32()
    }
    fn from_le_bytes(bytes: &[u8]) -> T {
        T::from_bits16(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
}

macro_rules! fp8_cache_elem {
    ($t:ident, $name:literal) => {
        impl CacheElem for $t {
            const CACHE_NAME: &'static str = $name;
            fn from_f32_rounded(x: f32) -> $t {
                $t::from_f32(x)
            }
            fn value(self) -> f32 {
                self.to_f32()
            }
            fn from_le_bytes(bytes: &[u8]) -> $t {
                $t(bytes[0])
            }
        }
    };
}
fp8_cache_elem!(F8E4M3FN, "e4m3");
fp8_cache_elem!(F8E5M2, "e5m2");

/// Position / index types.
pub trait IndexType: Numeric + Copy {
    const NAME: &'static str;
    fn from_i32(x: i32) -> Self;
    fn to_i64(self) -> i64;
}

impl IndexType for i32 {
    const NAME: &'static str = "i32";
    fn from_i32(x: i32) -> i32 {
        x
    }
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl IndexType for i64 {
    const NAME: &'static str = "i64";
    fn from_i32(x: i32) -> i64 {
        x as i64
    }
    fn to_i64(self) -> i64 {
        self
    }
}

/// Which launch shape a RoPE kernel has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    HeadParallel,
    Unsplit,
    Split,
}

impl Variant {
    fn tag(self) -> &'static str {
        match self {
            Variant::HeadParallel => "headpar",
            Variant::Unsplit => "unsplit",
            Variant::Split => "split",
        }
    }
}

/// Threads along the head dimension.
pub fn bdx(head_dim: usize) -> usize {
    head_dim / VEC
}

/// Tokens per block.
pub fn bdy(head_dim: usize) -> usize {
    128usize.max(bdx(head_dim)) / bdx(head_dim)
}

/// Parameter positions of the RoPE kernels.
pub mod p {
    pub const Q: usize = 0;
    pub const K: usize = 1;
    pub const COS_SIN_CACHE: usize = 2;
    pub const POS_IDS: usize = 3;
    pub const NNZ: usize = 4;
    pub const NUM_Q_HEADS: usize = 5;
    pub const NUM_K_HEADS: usize = 6;
}

/// The RoPE kernel (in place) for element type `T`, position type `I`, `head_dim` (64 or
/// 128) and the layout. Parameters, in the order of [`p`]: `q`, `k` (contiguous `[nnz,
/// heads * head_dim]`), `cos_sin_cache [max_pos, head_dim]` (f32), `pos_ids [nnz]`, and the
/// scalars `nnz`, `num_q_heads`, `num_k_heads`.
pub fn kernel<T: Half, I: IndexType>(head_dim: usize, interleave: bool, variant: Variant) -> Kernel {
    assert!(head_dim == 64 || head_dim == 128, "the RoPE kernel supports head dim 64 and 128");
    let name = format!(
        "flashinfer_rope_cos_sin_{}_{}_{}_{}_{head_dim}",
        if interleave { "gptj" } else { "neox" },
        variant.tag(),
        T::NAME,
        I::NAME
    );
    let (bx, by) = (bdx(head_dim), bdy(head_dim));
    Kernel::builder(&name).block_dim([bx as u32, by as u32, 1]).build(|| {
        let q = param_ptr::<T>("q");
        let k = param_ptr::<T>("k");
        let cache = param_ptr::<f32>("cos_sin_cache");
        let pos_ids = param_ptr::<I>("pos_ids");
        let nnz = param_scalar::<i32>("nnz");
        let num_q_heads = param_scalar::<i32>("num_q_heads");
        let num_k_heads = param_scalar::<i32>("num_k_heads");

        let (tx, ty) = (thread_idx().x, thread_idx().y);
        let hd = head_dim as i32;
        let hdv = (head_dim / VEC) as i32;
        let idx = block_idx().x * by as i32 + ty;
        if_(idx.lt(nnz), || {
            let pos = pos_ids.load_at(idx).to::<i32>();
            let row = pos * hd;
            let half = hd / 2;
            // This thread's cos and sin, per element of its vector.
            let (cos, sin): (Vec<Float32>, Vec<Float32>) = if interleave {
                // Element 2j and 2j + 1 share entry j: this thread's four entries start at
                // `tx * 4` (FlashInfer loads eight and uses the first four).
                let c = load_vector::<f32, 4>(&cache.offset(row + tx * 4), LoadOptions::default());
                let s = load_vector::<f32, 4>(&cache.offset(row + half + tx * 4), LoadOptions::default());
                let dup = |v: [Float32; 4]| (0..VEC).map(|i| v[i / 2]).collect::<Vec<_>>();
                (dup(c), dup(s))
            } else {
                let start = (tx % (bx as i32 / 2)) * VEC as i32;
                let c0 = load_vector::<f32, 4>(&cache.offset(row + start), LoadOptions::default());
                let c1 = load_vector::<f32, 4>(&cache.offset(row + start + 4), LoadOptions::default());
                let s0 = load_vector::<f32, 4>(&cache.offset(row + half + start), LoadOptions::default());
                let s1 = load_vector::<f32, 4>(&cache.offset(row + half + start + 4), LoadOptions::default());
                (c0.into_iter().chain(c1).collect(), s0.into_iter().chain(s1).collect())
            };
            // NeoX: the first half of the head takes `x * cos - partner * sin`, the second
            // `x * cos + partner * sin`, so the sin carries the sign.
            let first_half = tx.lt(bx as i32 / 2);
            let sin: Vec<Float32> =
                if interleave { sin } else { sin.into_iter().map(|s| select(first_half, -s, s)).collect() };
            let partner_tx = select(first_half, tx + bx as i32 / 2, tx - bx as i32 / 2);

            let apply = |x: &Ptr<T>, row_vecs: Int32, h: Int32| {
                let vec_of = |t: Int32| (idx * row_vecs + h * hdv + t) * VEC as i32;
                let xv = load_vector::<T, VEC>(&x.offset(vec_of(tx)), LoadOptions::default());
                let xf: Vec<Float32> = xv.iter().map(|v| v.to::<f32>()).collect();
                let out: Vec<Float32> = if interleave {
                    (0..VEC)
                        .map(|i| {
                            let partner = xf[i ^ 1];
                            // Even elements subtract, odd add.
                            if i % 2 == 0 {
                                xf[i] * cos[i] - partner * sin[i]
                            } else {
                                xf[i] * cos[i] + partner * sin[i]
                            }
                        })
                        .collect()
                } else {
                    let pv = load_vector::<T, VEC>(&x.offset(vec_of(partner_tx)), LoadOptions::default());
                    (0..VEC).map(|i| xf[i] * cos[i] + pv[i].to::<f32>() * sin[i]).collect()
                };
                let ov: [Value<T>; VEC] = std::array::from_fn(|i| out[i].to::<T>());
                if !interleave {
                    // The partner's half was read from the buffer this call overwrites: order
                    // every lane's read of its row before any lane's store (independent thread
                    // scheduling does not keep a warp in lockstep). The mask is the row's active lanes.
                    let width = bdx(head_dim) as u32;
                    let row_lanes = Uint32::from((1u32 << width) - 1) << (lane_idx().to::<u32>() / width * width);
                    sync_warp_with(row_lanes);
                }
                store_vector::<T, VEC>(&x.offset(vec_of(tx)), ov, StoreOptions::default());
            };
            let q_row_vecs = num_q_heads * hdv;
            let k_row_vecs = num_k_heads * hdv;
            match variant {
                Variant::HeadParallel => {
                    let h = block_idx().y;
                    if_else(h.lt(num_q_heads), || apply(&q, q_row_vecs, h), || apply(&k, k_row_vecs, h - num_q_heads));
                }
                Variant::Unsplit => {
                    for_range(0, num_q_heads, 1, |h| apply(&q, q_row_vecs, h));
                    for_range(0, num_k_heads, 1, |h| apply(&k, k_row_vecs, h));
                }
                Variant::Split => {
                    let (h0, step) = (block_idx().y, grid_dim().y);
                    for_range(h0, num_q_heads, step, |h| apply(&q, q_row_vecs, h));
                    for_range(h0, num_k_heads, step, |h| apply(&k, k_row_vecs, h));
                }
            }
        });
    })
}

/// FlashInfer's `RopeSplitCost`: waves times (steps + 1).
fn split_cost(nblks: usize, resident: usize, g: usize, heads: usize) -> u64 {
    let waves = (nblks * g).div_ceil(resident) as u64;
    waves * (heads.div_ceil(g) as u64 + 1)
}

/// A launch decision: the kernel variant and the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub variant: Variant,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

/// FlashInfer's launch planner for one GPU (without its SM 10.7 two-heads variant):
/// `resident_unsplit` and `resident_split` are the CTAs that run at once (blocks per SM
/// times SMs) of the unsplit and the split kernel.
pub fn plan(
    head_dim: usize,
    nnz: usize,
    num_q_heads: usize,
    num_k_heads: usize,
    resident_unsplit: usize,
    resident_split: usize,
) -> Plan {
    let nblks = nnz.div_ceil(bdy(head_dim)).max(1);
    let heads = num_q_heads + num_k_heads;
    let block = [bdx(head_dim) as u32, bdy(head_dim) as u32, 1];
    if nblks < resident_unsplit {
        return Plan { variant: Variant::HeadParallel, grid: [nblks as u32, heads as u32, 1], block };
    }
    // RopeGridSplit: the split minimising the wave cost.
    let (mut best, mut best_cost) = (1usize, u64::MAX);
    for g in 1..=heads.min(128) {
        let c = split_cost(nblks, resident_split, g, heads);
        if c < best_cost {
            (best, best_cost) = (g, c);
        }
    }
    let unsplit_cost = split_cost(nblks, resident_unsplit, 1, heads);
    let multi_wave = nblks > resident_unsplit;
    let unsplit = resident_unsplit == 0
        || if multi_wave { unsplit_cost * 10 <= best_cost * 11 } else { unsplit_cost <= best_cost };
    if unsplit {
        Plan { variant: Variant::Unsplit, grid: [nblks as u32, 1, 1], block }
    } else {
        Plan { variant: Variant::Split, grid: [nblks as u32, best as u32, 1], block }
    }
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

/// A RoPE test problem: random q and k, random positions and a base-10000 cos/sin cache.
pub struct Problem<T: Half, I: IndexType = i32> {
    pub head_dim: usize,
    pub interleave: bool,
    pub nnz: usize,
    pub num_q_heads: usize,
    pub num_k_heads: usize,
    pub q: Vec<T>,
    pub k: Vec<T>,
    pub cos_sin_cache: Vec<f32>,
    pub pos_ids: Vec<I>,
}

impl<T: Half, I: IndexType> Problem<T, I> {
    pub fn new(
        nnz: usize,
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
        interleave: bool,
        max_pos: usize,
        seed: u64,
    ) -> Self {
        let mut rnd = lcg(seed);
        let half = head_dim / 2;
        let mut cache = vec![0f32; max_pos * head_dim];
        for pos in 0..max_pos {
            for j in 0..half {
                let freq = 1.0f64 / 10000f64.powf(2.0 * j as f64 / head_dim as f64);
                let a = pos as f64 * freq;
                cache[pos * head_dim + j] = a.cos() as f32;
                cache[pos * head_dim + half + j] = a.sin() as f32;
            }
        }
        let pos_ids =
            (0..nnz).map(|_| I::from_i32(((rnd() + 1.0) * 0.5 * max_pos as f32) as i32 % max_pos as i32)).collect();
        let q = (0..nnz * num_q_heads * head_dim).map(|_| T::from_f32(rnd())).collect();
        let k = (0..nnz * num_k_heads * head_dim).map(|_| T::from_f32(rnd())).collect();
        Problem { head_dim, interleave, nnz, num_q_heads, num_k_heads, q, k, cos_sin_cache: cache, pos_ids }
    }

    /// The kernel's arguments, in the order of [`p`].
    pub fn args(&self) -> Vec<KernelArg> {
        vec![
            KernelArg::buffer(&self.q),
            KernelArg::buffer(&self.k),
            KernelArg::f32(self.cos_sin_cache.clone()),
            KernelArg::buffer(&self.pos_ids),
            KernelArg::ScalarI32(self.nnz as i32),
            KernelArg::ScalarI32(self.num_q_heads as i32),
            KernelArg::ScalarI32(self.num_k_heads as i32),
        ]
    }

    /// The rotated `x [nnz, heads * head_dim]`, in f64 from the exact f16/bf16 inputs and the
    /// f32 cache, as f32 (unrounded to `T`).
    pub fn reference(&self, x: &[T], heads: usize) -> Vec<f32> {
        let (hd, half) = (self.head_dim, self.head_dim / 2);
        let mut out = vec![0f32; x.len()];
        for t in 0..self.nnz {
            let pos = self.pos_ids[t].to_i64() as usize;
            let c = |j: usize| self.cos_sin_cache[pos * hd + j] as f64;
            let s = |j: usize| self.cos_sin_cache[pos * hd + half + j] as f64;
            for h in 0..heads {
                let base = (t * heads + h) * hd;
                let f = |i: usize| x[base + i].to_f32() as f64;
                for j in 0..half {
                    let (i0, i1) = if self.interleave { (2 * j, 2 * j + 1) } else { (j, j + half) };
                    out[base + i0] = (f(i0) * c(j) - f(i1) * s(j)) as f32;
                    out[base + i1] = (f(i1) * c(j) + f(i0) * s(j)) as f32;
                }
            }
        }
        out
    }
}

pub use crate::check::max_error;

/// Simulate `p` with the launch shape `variant` (the split one with `split` head groups)
/// and return the max errors against the reference, for q and k.
pub fn simulate<T: Half, I: IndexType>(p: &Problem<T, I>, variant: Variant, split: usize) -> (f32, f32) {
    let (bx, by) = (bdx(p.head_dim) as u32, bdy(p.head_dim) as u32);
    let nblks = p.nnz.div_ceil(by as usize) as u32;
    let grid = match variant {
        Variant::HeadParallel => [nblks, (p.num_q_heads + p.num_k_heads) as u32, 1],
        Variant::Unsplit => [nblks, 1, 1],
        Variant::Split => [nblks, split as u32, 1],
    };
    let mut args = p.args();
    let k = kernel::<T, I>(p.head_dim, p.interleave, variant);
    k.simulate(LaunchConfig::new(grid, [bx, by, 1]), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let q: Vec<f32> = args[self::p::Q].to_vec::<T>().into_iter().map(|x| x.to_f32()).collect();
    let kk: Vec<f32> = args[self::p::K].to_vec::<T>().into_iter().map(|x| x.to_f32()).collect();
    (max_error(&q, &p.reference(&p.q, p.num_q_heads)), max_error(&kk, &p.reference(&p.k, p.num_k_heads)))
}

// ---------------------------------------------------------------------------------------
// RoPE with on-the-fly frequencies (`apply_rope*`, `apply_llama31_rope*`)
// ---------------------------------------------------------------------------------------

/// Parameter positions of the frequency RoPE kernels.
pub mod fp {
    pub const Q: usize = 0;
    pub const K: usize = 1;
    pub const POS_IDS: usize = 2;
    pub const NNZ: usize = 3;
    pub const NUM_Q_HEADS: usize = 4;
    pub const NUM_K_HEADS: usize = 5;
    pub const RCP_SCALE: usize = 6;
    pub const RCP_THETA: usize = 7;
    pub const SMOOTH_A: usize = 8;
    pub const SMOOTH_B: usize = 9;
}

/// The frequency schedule of a RoPE call: `rotary_dim` leading elements of each head are
/// rotated by `pos * freq_j`, `freq_j = rope_theta^(-2j / rotary_dim) / rope_scale`, or,
/// with [`FreqConfig::llama31`], Llama 3.1's smoothed blend of the scaled and unscaled
/// frequency.
#[derive(Clone, Copy, Debug)]
pub struct FreqConfig {
    pub rotary_dim: usize,
    pub rope_scale: f32,
    pub rope_theta: f32,
    /// `(low_freq_factor, high_freq_factor, old_context_length)`.
    pub llama31: Option<(f32, f32, f32)>,
}

impl FreqConfig {
    /// `apply_rope*`'s defaults: scale 1, theta 1e4.
    pub fn new(rotary_dim: usize) -> Self {
        FreqConfig { rotary_dim, rope_scale: 1.0, rope_theta: 1e4, llama31: None }
    }

    /// `apply_llama31_rope*`'s defaults: scale 8, theta 5e5, factors 1 and 4, context 8192.
    pub fn llama31(rotary_dim: usize) -> Self {
        FreqConfig { rotary_dim, rope_scale: 8.0, rope_theta: 5e5, llama31: Some((1.0, 4.0, 8192.0)) }
    }

    /// FlashInfer's `(smooth_a, smooth_b)`: both 0 without Llama 3.1 smoothing.
    pub fn smooth(&self) -> (f32, f32) {
        match self.llama31 {
            None => (0.0, 0.0),
            Some((lo, hi, old)) => {
                (old / (2.0 * std::f32::consts::PI * hi - 2.0 * std::f32::consts::PI * lo), -1.0 / (hi / lo - 1.0))
            }
        }
    }

    /// The scalar arguments (reciprocal scale and theta, smooth a and b) of the kernel.
    pub fn args(&self) -> [KernelArg; 4] {
        let (a, b) = self.smooth();
        [
            KernelArg::ScalarF32(1.0 / self.rope_scale),
            KernelArg::ScalarF32(1.0 / self.rope_theta),
            KernelArg::ScalarF32(a),
            KernelArg::ScalarF32(b),
        ]
    }

    /// Frequency `j` (`0 <= j < rotary_dim / 2`), in f64.
    pub fn freq(&self, j: usize) -> f64 {
        let f = (self.rope_theta as f64).powf(-2.0 * j as f64 / self.rotary_dim as f64);
        let (a, b) = self.smooth();
        let smooth = (f * a as f64 + b as f64).clamp(0.0, 1.0);
        (1.0 - smooth) * (f / self.rope_scale as f64) + smooth * f
    }
}

/// The per-token positions of `apply_rope`'s `(indptr, offsets)` form: token `i` of request
/// `b` has position `offsets[b] + i - indptr[b]`.
pub fn pos_ids_from_indptr(indptr: &[i32], offsets: &[i32]) -> Vec<i32> {
    let mut pos = vec![];
    for b in 0..offsets.len() {
        pos.extend((0..indptr[b + 1] - indptr[b]).map(|i| offsets[b] + i));
    }
    pos
}

/// RoPE (in place) with frequencies computed in the kernel: `__powf` once per element, then
/// `__sincosf` of `pos * freq` per token, as FlashInfer's `BatchQKApplyRotaryPosIds`.
/// Elements at or past `rotary_dim` (a multiple of 16, at most `head_dim`) are left
/// unchanged. Parameters, in the order of [`fp`]: `q`, `k` (contiguous), `pos_ids [nnz]`, the
/// scalars `nnz`, `num_q_heads`, `num_k_heads`, then the f32 `rope_rcp_scale`,
/// `rope_rcp_theta`, `smooth_a` and `smooth_b` ([`FreqConfig::args`]). The launch shapes are
/// [`Variant`]'s, planned by [`plan`].
pub fn freq_kernel<T: Half, I: IndexType>(
    head_dim: usize,
    rotary_dim: usize,
    interleave: bool,
    variant: Variant,
) -> Kernel {
    assert!(head_dim == 64 || head_dim == 128, "the RoPE kernel supports head dim 64 and 128");
    assert!(
        rotary_dim.is_multiple_of(16) && rotary_dim <= head_dim,
        "rotary_dim must be a multiple of 16, at most head_dim"
    );
    let name = format!(
        "flashinfer_rope_freq_{}_{}_{}_{}_{head_dim}_{rotary_dim}",
        if interleave { "gptj" } else { "neox" },
        variant.tag(),
        T::NAME,
        I::NAME
    );
    let (bx, by) = (bdx(head_dim), bdy(head_dim));
    let rbx = (rotary_dim / VEC) as i32;
    Kernel::builder(&name).block_dim([bx as u32, by as u32, 1]).build(|| {
        let q = param_ptr::<T>("q");
        let k = param_ptr::<T>("k");
        let pos_ids = param_ptr::<I>("pos_ids");
        let nnz = param_scalar::<i32>("nnz");
        let num_q_heads = param_scalar::<i32>("num_q_heads");
        let num_k_heads = param_scalar::<i32>("num_k_heads");
        let rcp_scale = param_scalar::<f32>("rope_rcp_scale");
        let rcp_theta = param_scalar::<f32>("rope_rcp_theta");
        let smooth_a = param_scalar::<f32>("smooth_a");
        let smooth_b = param_scalar::<f32>("smooth_b");

        let (tx, ty) = (thread_idx().x, thread_idx().y);
        let hdv = (head_dim / VEC) as i32;
        let idx = block_idx().x * by as i32 + ty;
        if_(idx.lt(nnz) & tx.lt(rbx), || {
            let pos = pos_ids.load_at(idx).to::<i32>().to::<f32>();
            // This thread's frequencies, then cos and sin of `pos * freq` per element.
            let mut cos = vec![];
            let mut sin = vec![];
            for i in 0..VEC as i32 {
                let e: Int32 =
                    if interleave { tx * (VEC as i32 / 2) + i / 2 } else { (tx * VEC as i32) % (rbx * 4) + i };
                let expo = (e * 2).to::<f32>() / rotary_dim as f32;
                let f = cute::math::pow(rcp_theta, expo, true);
                let smooth = fmax(0.0f32, fmin(1.0f32, f * smooth_a + smooth_b));
                let f = (smooth * -1.0f32 + 1.0f32) * (f * rcp_scale) + smooth * f;
                let (s, c) = cute::math::sincos(pos * f, true);
                cos.push(c);
                sin.push(s);
            }
            let first_half = tx.lt(rbx / 2);
            let sin: Vec<Float32> =
                if interleave { sin } else { sin.into_iter().map(|s| select(first_half, -s, s)).collect() };
            let partner_tx = select(first_half, tx + rbx / 2, tx - rbx / 2);

            let apply = |x: &Ptr<T>, row_vecs: Int32, h: Int32| {
                let vec_of = |t: Int32| (idx * row_vecs + h * hdv + t) * VEC as i32;
                let xv = load_vector::<T, VEC>(&x.offset(vec_of(tx)), LoadOptions::default());
                let xf: Vec<Float32> = xv.iter().map(|v| v.to::<f32>()).collect();
                let out: Vec<Float32> = if interleave {
                    (0..VEC)
                        .map(|i| {
                            let partner = xf[i ^ 1];
                            if i % 2 == 0 {
                                xf[i] * cos[i] - partner * sin[i]
                            } else {
                                xf[i] * cos[i] + partner * sin[i]
                            }
                        })
                        .collect()
                } else {
                    let pv = load_vector::<T, VEC>(&x.offset(vec_of(partner_tx)), LoadOptions::default());
                    (0..VEC).map(|i| xf[i] * cos[i] + pv[i].to::<f32>() * sin[i]).collect()
                };
                let ov: [Value<T>; VEC] = std::array::from_fn(|i| out[i].to::<T>());
                if !interleave {
                    // The partner's half was read from the buffer this call overwrites: order
                    // every lane's read of its row before any lane's store (independent thread
                    // scheduling does not keep a warp in lockstep). The mask is the row's active lanes.
                    let (width, active) = (bdx(head_dim) as u32, rbx as u32);
                    let row_lanes = Uint32::from((1u32 << active) - 1) << (lane_idx().to::<u32>() / width * width);
                    sync_warp_with(row_lanes);
                }
                store_vector::<T, VEC>(&x.offset(vec_of(tx)), ov, StoreOptions::default());
            };
            let q_row_vecs = num_q_heads * hdv;
            let k_row_vecs = num_k_heads * hdv;
            match variant {
                Variant::HeadParallel => {
                    let h = block_idx().y;
                    if_else(h.lt(num_q_heads), || apply(&q, q_row_vecs, h), || apply(&k, k_row_vecs, h - num_q_heads));
                }
                Variant::Unsplit => {
                    for_range(0, num_q_heads, 1, |h| apply(&q, q_row_vecs, h));
                    for_range(0, num_k_heads, 1, |h| apply(&k, k_row_vecs, h));
                }
                Variant::Split => {
                    let (h0, step) = (block_idx().y, grid_dim().y);
                    for_range(h0, num_q_heads, step, |h| apply(&q, q_row_vecs, h));
                    for_range(h0, num_k_heads, step, |h| apply(&k, k_row_vecs, h));
                }
            }
        });
    })
}

impl<T: Half, I: IndexType> Problem<T, I> {
    /// Arguments of [`freq_kernel`], in the order of [`fp`].
    pub fn freq_args(&self, cfg: &FreqConfig) -> Vec<KernelArg> {
        let mut v = vec![
            KernelArg::buffer(&self.q),
            KernelArg::buffer(&self.k),
            KernelArg::buffer(&self.pos_ids),
            KernelArg::ScalarI32(self.nnz as i32),
            KernelArg::ScalarI32(self.num_q_heads as i32),
            KernelArg::ScalarI32(self.num_k_heads as i32),
        ];
        v.extend(cfg.args());
        v
    }

    /// The rotated `x` for [`freq_kernel`], in f64 (as f32, unrounded to `T`): the first
    /// `rotary_dim` elements of each head rotated, the rest copied.
    pub fn freq_reference(&self, cfg: &FreqConfig, x: &[T], heads: usize) -> Vec<f32> {
        let (hd, rd) = (self.head_dim, cfg.rotary_dim);
        let mut out: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
        for t in 0..self.nnz {
            let pos = self.pos_ids[t].to_i64() as f64;
            for h in 0..heads {
                let base = (t * heads + h) * hd;
                let f = |i: usize| x[base + i].to_f32() as f64;
                for j in 0..rd / 2 {
                    let (c, s) = ((pos * cfg.freq(j)).cos(), (pos * cfg.freq(j)).sin());
                    let (i0, i1) = if self.interleave { (2 * j, 2 * j + 1) } else { (j, j + rd / 2) };
                    out[base + i0] = (f(i0) * c - f(i1) * s) as f32;
                    out[base + i1] = (f(i1) * c + f(i0) * s) as f32;
                }
            }
        }
        out
    }
}

/// Simulate [`freq_kernel`] on `p` and return the max errors for q and k against
/// [`Problem::freq_reference`].
pub fn simulate_freq<T: Half, I: IndexType>(
    p: &Problem<T, I>,
    cfg: &FreqConfig,
    variant: Variant,
    split: usize,
) -> (f32, f32) {
    let (bx, by) = (bdx(p.head_dim) as u32, bdy(p.head_dim) as u32);
    let nblks = p.nnz.div_ceil(by as usize) as u32;
    let grid = match variant {
        Variant::HeadParallel => [nblks, (p.num_q_heads + p.num_k_heads) as u32, 1],
        Variant::Unsplit => [nblks, 1, 1],
        Variant::Split => [nblks, split as u32, 1],
    };
    let mut args = p.freq_args(cfg);
    let k = freq_kernel::<T, I>(p.head_dim, cfg.rotary_dim, p.interleave, variant);
    k.simulate(LaunchConfig::new(grid, [bx, by, 1]), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let q: Vec<f32> = args[fp::Q].to_vec::<T>().into_iter().map(|x| x.to_f32()).collect();
    let kk: Vec<f32> = args[fp::K].to_vec::<T>().into_iter().map(|x| x.to_f32()).collect();
    (
        max_error(&q, &p.freq_reference(cfg, &p.q, p.num_q_heads)),
        max_error(&kk, &p.freq_reference(cfg, &p.k, p.num_k_heads)),
    )
}

// ---------------------------------------------------------------------------------------
// Paged KV append
// ---------------------------------------------------------------------------------------

/// Parameter positions of the append kernel.
pub mod ap {
    pub const APPEND_K: usize = 0;
    pub const APPEND_V: usize = 1;
    pub const KV_CACHE: usize = 2;
    pub const KV_INDPTR: usize = 3;
    pub const KV_INDICES: usize = 4;
    pub const BATCH_INDICES: usize = 5;
    pub const POSITIONS: usize = 6;
    pub const NNZ: usize = 7;
    /// Three arguments: the page size's [`KernelArg::fast_divmod`].
    pub const PAGE_SIZE: usize = 8;
    pub const PAGE_STRIDE_VECS: usize = 11;
    pub const V_OFFSET_VECS: usize = 12;
}

/// Elements per thread of the append over `T`: 16 bytes (FlashInfer's
/// `vec_size = max(16 / sizeof(DType), head_dim / 32)` for head dims up to 256).
pub fn append_vec<T: CacheElem>() -> usize {
    16 / T::BYTES
}

/// The paged-KV append kernel for `num_kv_heads` heads of `head_dim` (a multiple of 16
/// bytes, at most 1024 threads in all). As FlashInfer's, inputs and cache have one element
/// type `T`: a Float16/BFloat16 cache, or an FP8 cache taking FP8 inputs. Parameters, in
/// the order of [`ap`]: `append_k`, `append_v` (`[nnz, num_kv_heads, head_dim]`),
/// `kv_cache` (NHD, K of each page then V), `kv_indptr`, `kv_indices`, `batch_indices`,
/// `positions` (i32), `nnz`, the page size's fast divmod (three arguments), the page stride
/// and the V offset within a page, both in [`append_vec`]-element vectors.
pub fn append_kernel<T: CacheElem>(num_kv_heads: usize, head_dim: usize) -> Kernel {
    match T::BYTES {
        2 => append_kernel_vec::<T, 8>(num_kv_heads, head_dim),
        1 => append_kernel_vec::<T, 16>(num_kv_heads, head_dim),
        _ => unreachable!("16- and 8-bit elements only"),
    }
}

fn append_kernel_vec<T: CacheElem, const VEC: usize>(num_kv_heads: usize, head_dim: usize) -> Kernel {
    assert!(head_dim.is_multiple_of(VEC) && head_dim / VEC * num_kv_heads <= 1024);
    let name = format!("flashinfer_append_paged_kv_cache_{}_h{num_kv_heads}_d{head_dim}", T::CACHE_NAME);
    let bx = head_dim / VEC;
    Kernel::builder(&name).block_dim([bx as u32, num_kv_heads as u32, 1]).build(|| {
        let append_k = param_ptr::<T>("append_k");
        let append_v = param_ptr::<T>("append_v");
        let cache = param_ptr::<T>("kv_cache");
        let kv_indptr = param_ptr::<i32>("kv_indptr");
        let kv_indices = param_ptr::<i32>("kv_indices");
        let batch_indices = param_ptr::<i32>("batch_indices");
        let positions = param_ptr::<i32>("positions");
        let nnz = param_scalar::<i32>("nnz");
        let page_size = param_fast_divmod("page_size");
        let page_stride_vecs = param_scalar::<i32>("page_stride_vecs");
        let v_offset_vecs = param_scalar::<i32>("v_offset_vecs");

        let (tx, head) = (thread_idx().x, thread_idx().y);
        let hdv = bx as i32;
        let nh = num_kv_heads as i32;
        for_range_with(block_idx().x, nnz, grid_dim().x, LoopOptions::unroll(4), |i| {
            let b = batch_indices.load_at(i);
            let pos = positions.load_at(i);
            let (page_iter, entry) = page_size.divmod(kv_indptr.load_at(b) * page_size.divisor + pos);
            let page = kv_indices.load_at(page_iter);
            let dst = (page * page_stride_vecs + (entry * nh + head) * hdv + tx) * VEC as i32;
            let src = ((i * nh + head) * hdv + tx) * VEC as i32;
            let k_in = load_vector::<T, VEC>(&append_k.offset(src), LoadOptions::default());
            let v_in = load_vector::<T, VEC>(&append_v.offset(src), LoadOptions::default());
            store_vector::<T, VEC>(&cache.offset(dst), k_in, StoreOptions::default());
            store_vector::<T, VEC>(&cache.offset(dst + v_offset_vecs * VEC as i32), v_in, StoreOptions::default());
        });
    })
}

/// `batch_indices` and `positions` of the appended tokens (FlashInfer's
/// `get_batch_indices_positions`): token `i` of request `b` (`append_indptr[b] <= i <
/// append_indptr[b + 1]`) goes to position `seq_lens[b] - append_len[b] + (i -
/// append_indptr[b])`.
pub fn batch_indices_positions(append_indptr: &[i32], seq_lens: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let (mut batch, mut pos) = (vec![], vec![]);
    for b in 0..seq_lens.len() {
        let len = append_indptr[b + 1] - append_indptr[b];
        for j in 0..len {
            batch.push(b as i32);
            pos.push(seq_lens[b] - len + j);
        }
    }
    (batch, pos)
}

/// An append test problem: requests whose caches hold `seq_lens` tokens after the append
/// of `append_lens` new ones, with a shuffled page table.
pub struct AppendProblem<T: CacheElem> {
    pub page_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_pages: usize,
    pub append_k: Vec<T>,
    pub append_v: Vec<T>,
    /// `[num_pages, 2, page_size, num_kv_heads, head_dim]`, random before the append.
    pub kv_cache: Vec<T>,
    pub kv_indptr: Vec<i32>,
    pub kv_indices: Vec<i32>,
    pub batch_indices: Vec<i32>,
    pub positions: Vec<i32>,
}

impl<T: CacheElem> AppendProblem<T> {
    pub fn new(
        seq_lens: &[usize],
        append_lens: &[usize],
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seed: u64,
    ) -> Self {
        let mut rnd = lcg(seed);
        let pages_per: Vec<usize> = seq_lens.iter().map(|l| l.div_ceil(page_size)).collect();
        let num_pages = pages_per.iter().sum::<usize>() + 2;
        let mut ids: Vec<i32> = (0..num_pages as i32).collect();
        for i in (1..ids.len()).rev() {
            let j = ((rnd() + 1.0) * 0.5 * (i + 1) as f32) as usize % (i + 1);
            ids.swap(i, j);
        }
        let (mut kv_indptr, mut kv_indices, mut next) = (vec![0i32], vec![], 0);
        for &n in &pages_per {
            kv_indices.extend_from_slice(&ids[next..next + n]);
            next += n;
            kv_indptr.push(next as i32);
        }
        let mut append_indptr = vec![0i32];
        for &a in append_lens {
            append_indptr.push(append_indptr.last().unwrap() + a as i32);
        }
        let seq32: Vec<i32> = seq_lens.iter().map(|&l| l as i32).collect();
        let (batch_indices, positions) = batch_indices_positions(&append_indptr, &seq32);
        let nnz = batch_indices.len();
        let row = num_kv_heads * head_dim;
        let append_k = (0..nnz * row).map(|_| T::from_f32_rounded(rnd())).collect();
        let append_v = (0..nnz * row).map(|_| T::from_f32_rounded(rnd())).collect();
        let kv_cache = (0..num_pages * 2 * page_size * row).map(|_| T::from_f32_rounded(rnd())).collect();
        AppendProblem {
            page_size,
            num_kv_heads,
            head_dim,
            num_pages,
            append_k,
            append_v,
            kv_cache,
            kv_indptr,
            kv_indices,
            batch_indices,
            positions,
        }
    }

    pub fn nnz(&self) -> usize {
        self.batch_indices.len()
    }

    /// Elements of one (K or V) half page.
    pub fn half_page(&self) -> usize {
        self.page_size * self.num_kv_heads * self.head_dim
    }

    /// The kernel's arguments, in the order of [`ap`].
    pub fn args(&self) -> Vec<KernelArg> {
        let [d0, d1, d2] = KernelArg::fast_divmod(self.page_size as i32);
        vec![
            KernelArg::buffer(&self.append_k),
            KernelArg::buffer(&self.append_v),
            KernelArg::buffer(&self.kv_cache),
            KernelArg::i32(self.kv_indptr.clone()),
            KernelArg::i32(self.kv_indices.clone()),
            KernelArg::i32(self.batch_indices.clone()),
            KernelArg::i32(self.positions.clone()),
            KernelArg::ScalarI32(self.nnz() as i32),
            d0,
            d1,
            d2,
            KernelArg::ScalarI32((2 * self.half_page() / append_vec::<T>()) as i32),
            KernelArg::ScalarI32((self.half_page() / append_vec::<T>()) as i32),
        ]
    }

    pub fn launch(&self, grid_x: usize) -> LaunchConfig {
        LaunchConfig::new(
            [grid_x as u32, 1, 1],
            [(self.head_dim / append_vec::<T>()) as u32, self.num_kv_heads as u32, 1],
        )
    }

    /// The cache after the append, on the host.
    pub fn reference(&self) -> Vec<T> {
        let mut cache = self.kv_cache.clone();
        let (h, d, ps) = (self.num_kv_heads, self.head_dim, self.page_size);
        for i in 0..self.nnz() {
            let (b, pos) = (self.batch_indices[i] as usize, self.positions[i] as usize);
            let page = self.kv_indices[self.kv_indptr[b] as usize + pos / ps] as usize;
            let dst = ((page * 2) * ps + pos % ps) * h * d;
            let src = i * h * d;
            cache[dst..dst + h * d].copy_from_slice(&self.append_k[src..src + h * d]);
            let dst_v = dst + self.half_page();
            cache[dst_v..dst_v + h * d].copy_from_slice(&self.append_v[src..src + h * d]);
        }
        cache
    }
}

/// Simulate the append with `grid_x` blocks; the number of cache elements that differ from
/// the reference.
pub fn simulate_append<T: CacheElem>(p: &AppendProblem<T>, grid_x: usize) -> usize {
    let mut args = p.args();
    let k = append_kernel::<T>(p.num_kv_heads, p.head_dim);
    k.simulate(p.launch(grid_x), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let got = args[ap::KV_CACHE].to_vec::<T>();
    let want = p.reference();
    got.iter().zip(&want).filter(|(a, b)| a.value().to_bits() != b.value().to_bits()).count()
}

/// Running on the GPU (`--features cuda`): kernels compiled and loaded, problems run and
/// checked, and timed from CUDA-graph replays.
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg, HardwareInfo};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{
        AppendProblem, CacheElem, FreqConfig, Half, IndexType, Problem, Variant, ap, append_kernel, append_vec, bdx,
        bdy, fp, freq_kernel, kernel, max_error, plan,
    };

    fn compile<'c>(
        ctx: &'c CudaContext,
        tc: &Toolchain,
        k: Kernel,
    ) -> Result<(Kernel, CudaFunction<'c>), Box<dyn Error>> {
        let arch = ctx.best_target(tc).ok_or("no target for this GPU")?;
        let f = ctx.load(&k.compile_with(tc, &CompileOptions::new().gpu_arch(arch))?)?;
        Ok((k, f))
    }

    fn upload<'c>(
        ctx: &'c CudaContext,
        host: &[KernelArg],
    ) -> Result<Vec<Option<cute::cuda::DeviceBuffer<'c>>>, Box<dyn Error>> {
        host.iter()
            .map(|a| match a {
                KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) | KernelArg::Scalar { .. } => Ok(None),
                _ => Ok(Some(ctx.upload(a)?)),
            })
            .collect()
    }

    fn device_args<'a>(host: &[KernelArg], bufs: &'a [Option<cute::cuda::DeviceBuffer<'_>>]) -> Vec<DeviceArg<'a>> {
        host.iter()
            .zip(bufs)
            .map(|(a, b)| match b {
                Some(b) => DeviceArg::Buffer(b),
                None => DeviceArg::Scalar(a.clone()),
            })
            .collect()
    }

    /// The three RoPE launch shapes of one configuration, compiled and loaded once.
    pub struct RopeKernels<'c, T: Half, I: IndexType> {
        pub head_dim: usize,
        pub interleave: bool,
        pub unsplit: (Kernel, CudaFunction<'c>),
        pub split: (Kernel, CudaFunction<'c>),
        pub head_parallel: (Kernel, CudaFunction<'c>),
        pub resident_unsplit: usize,
        pub resident_split: usize,
        _t: std::marker::PhantomData<(T, I)>,
    }

    impl<'c, T: Half, I: IndexType> RopeKernels<'c, T, I> {
        pub fn new(
            ctx: &'c CudaContext,
            tc: &Toolchain,
            head_dim: usize,
            interleave: bool,
        ) -> Result<Self, Box<dyn Error>> {
            let build = |v| compile(ctx, tc, kernel::<T, I>(head_dim, interleave, v));
            let (unsplit, split, head_parallel) =
                (build(Variant::Unsplit)?, build(Variant::Split)?, build(Variant::HeadParallel)?);
            let threads = (bdx(head_dim) * bdy(head_dim)) as u32;
            let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            let resident_unsplit = unsplit.1.max_active_blocks_per_multiprocessor(threads, 0)? as usize * sms;
            let resident_split = split.1.max_active_blocks_per_multiprocessor(threads, 0)? as usize * sms;
            Ok(RopeKernels {
                head_dim,
                interleave,
                unsplit,
                split,
                head_parallel,
                resident_unsplit,
                resident_split,
                _t: std::marker::PhantomData,
            })
        }
    }

    /// Run `p` once, check it, and optionally time it (graph replays, microseconds). Returns
    /// the launch plan, the max errors for q and k, and the time.
    #[allow(clippy::type_complexity)]
    pub fn run_rope<T: Half, I: IndexType>(
        ctx: &CudaContext,
        ks: &RopeKernels<'_, T, I>,
        p: &Problem<T, I>,
        check: bool,
        time: bool,
    ) -> Result<(super::Plan, f32, f32, Option<f64>), Box<dyn Error>> {
        let pl = plan(p.head_dim, p.nnz, p.num_q_heads, p.num_k_heads, ks.resident_unsplit, ks.resident_split);
        let (kk, f) = match pl.variant {
            Variant::HeadParallel => (&ks.head_parallel.0, &ks.head_parallel.1),
            Variant::Unsplit => (&ks.unsplit.0, &ks.unsplit.1),
            Variant::Split => (&ks.split.0, &ks.split.1),
        };
        let host = p.args();
        let bufs = upload(ctx, &host)?;
        let args = device_args(&host, &bufs);
        let cfg = LaunchConfig::new(pl.grid, pl.block);
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(kk, f, cfg).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let (mut eq, mut ek) = (0.0, 0.0);
        if check {
            let q: Vec<f32> = bufs[super::p::Q]
                .as_ref()
                .unwrap()
                .to_vec::<u16>()?
                .into_iter()
                .map(T::from_bits16)
                .map(|x| x.to_f32())
                .collect();
            let k: Vec<f32> = bufs[super::p::K]
                .as_ref()
                .unwrap()
                .to_vec::<u16>()?
                .into_iter()
                .map(T::from_bits16)
                .map(|x| x.to_f32())
                .collect();
            eq = max_error(&q, &p.reference(&p.q, p.num_q_heads));
            ek = max_error(&k, &p.reference(&p.k, p.num_k_heads));
        }
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(20).iterations(200).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((pl, eq, ek, t))
    }

    /// The three launch shapes of one frequency-RoPE configuration, compiled and loaded once.
    pub struct FreqKernels<'c, T: Half, I: IndexType> {
        pub cfg: FreqConfig,
        pub head_dim: usize,
        pub unsplit: (Kernel, CudaFunction<'c>),
        pub split: (Kernel, CudaFunction<'c>),
        pub head_parallel: (Kernel, CudaFunction<'c>),
        pub resident_unsplit: usize,
        pub resident_split: usize,
        _t: std::marker::PhantomData<(T, I)>,
    }

    impl<'c, T: Half, I: IndexType> FreqKernels<'c, T, I> {
        pub fn new(
            ctx: &'c CudaContext,
            tc: &Toolchain,
            head_dim: usize,
            interleave: bool,
            cfg: FreqConfig,
        ) -> Result<Self, Box<dyn Error>> {
            let build = |v| compile(ctx, tc, freq_kernel::<T, I>(head_dim, cfg.rotary_dim, interleave, v));
            let (unsplit, split, head_parallel) =
                (build(Variant::Unsplit)?, build(Variant::Split)?, build(Variant::HeadParallel)?);
            let threads = (bdx(head_dim) * bdy(head_dim)) as u32;
            let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            let resident_unsplit = unsplit.1.max_active_blocks_per_multiprocessor(threads, 0)? as usize * sms;
            let resident_split = split.1.max_active_blocks_per_multiprocessor(threads, 0)? as usize * sms;
            Ok(FreqKernels {
                cfg,
                head_dim,
                unsplit,
                split,
                head_parallel,
                resident_unsplit,
                resident_split,
                _t: std::marker::PhantomData,
            })
        }
    }

    /// Run `p` through the frequency kernel once, check it against [`Problem::freq_reference`],
    /// and optionally time it (graph replays, microseconds).
    #[allow(clippy::type_complexity)]
    pub fn run_freq<T: Half, I: IndexType>(
        ctx: &CudaContext,
        ks: &FreqKernels<'_, T, I>,
        p: &Problem<T, I>,
        check: bool,
        time: bool,
    ) -> Result<(super::Plan, f32, f32, Option<f64>), Box<dyn Error>> {
        let pl = plan(p.head_dim, p.nnz, p.num_q_heads, p.num_k_heads, ks.resident_unsplit, ks.resident_split);
        let (kk, f) = match pl.variant {
            Variant::HeadParallel => (&ks.head_parallel.0, &ks.head_parallel.1),
            Variant::Unsplit => (&ks.unsplit.0, &ks.unsplit.1),
            Variant::Split => (&ks.split.0, &ks.split.1),
        };
        let host = p.freq_args(&ks.cfg);
        let bufs = upload(ctx, &host)?;
        let args = device_args(&host, &bufs);
        let cfg = LaunchConfig::new(pl.grid, pl.block);
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(kk, f, cfg).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let (mut eq, mut ek) = (0.0, 0.0);
        if check {
            let read = |i: usize| -> Result<Vec<f32>, Box<dyn Error>> {
                Ok(bufs[i]
                    .as_ref()
                    .unwrap()
                    .to_vec::<u16>()?
                    .into_iter()
                    .map(T::from_bits16)
                    .map(|x| x.to_f32())
                    .collect())
            };
            eq = max_error(&read(fp::Q)?, &p.freq_reference(&ks.cfg, &p.q, p.num_q_heads));
            ek = max_error(&read(fp::K)?, &p.freq_reference(&ks.cfg, &p.k, p.num_k_heads));
        }
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(20).iterations(200).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((pl, eq, ek, t))
    }

    /// The append kernel, compiled and loaded once, with its persistent-grid size.
    pub struct AppendKernels<'c, T: CacheElem> {
        pub kernel: (Kernel, CudaFunction<'c>),
        pub num_sms: usize,
        pub blocks_per_sm: usize,
        _t: std::marker::PhantomData<T>,
    }

    impl<'c, T: CacheElem> AppendKernels<'c, T> {
        pub fn new(
            ctx: &'c CudaContext,
            tc: &Toolchain,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<Self, Box<dyn Error>> {
            let kernel = compile(ctx, tc, append_kernel::<T>(num_kv_heads, head_dim))?;
            let threads = (head_dim / append_vec::<T>() * num_kv_heads) as u32;
            let blocks_per_sm = kernel.1.max_active_blocks_per_multiprocessor(threads, 0)? as usize;
            let num_sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            Ok(AppendKernels { kernel, num_sms, blocks_per_sm, _t: std::marker::PhantomData })
        }

        /// FlashInfer's grid: `min(blocks per SM, ceil(nnz / SMs)) * SMs` blocks.
        pub fn grid(&self, nnz: usize) -> usize {
            self.blocks_per_sm.min(nnz.div_ceil(self.num_sms)).max(1) * self.num_sms
        }
    }

    /// Run the append once, count the mismatching cache elements, optionally time it
    /// (microseconds, graph replays).
    pub fn run_append<T: CacheElem>(
        ctx: &CudaContext,
        ks: &AppendKernels<'_, T>,
        p: &AppendProblem<T>,
        check: bool,
        time: bool,
    ) -> Result<(usize, Option<f64>), Box<dyn Error>> {
        let host = p.args();
        let bufs = upload(ctx, &host)?;
        let args = device_args(&host, &bufs);
        let cfg = p.launch(ks.grid(p.nnz()));
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(&ks.kernel.0, &ks.kernel.1, cfg).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let mut bad = 0;
        if check {
            let got = bufs[ap::KV_CACHE].as_ref().unwrap().to_vec::<u8>()?;
            let want = p.reference();
            bad = got
                .chunks(T::BYTES)
                .zip(&want)
                .filter(|(g, w)| T::from_le_bytes(g).value().to_bits() != w.value().to_bits())
                .count();
        }
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(20).iterations(200).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((bad, t))
    }
}
