// SPDX-License-Identifier: Apache-2.0

//! RMSNorm and fused add + RMSNorm, after FlashInfer's `RMSNormKernel` and
//! `FusedAddRMSNormKernel` (`include/flashinfer/norm.cuh`).
//! It follows FlashInfer's design (one block per row, 16-byte vector accesses, the block
//! size rule, a warp shuffle then a shared-memory reduction, the residual sum stashed in
//! shared memory).
//!
//! * **Geometry** ([`num_warps`]): one block per row; `vec = 8` elements (16 bytes of
//!   Float16 or BFloat16) per thread per round; `block_size = min(1024, d / 8)`, rounded up
//!   to whole warps, so a row of `d = 4096` gets 512 threads; `rounds = ceil(d / (8 *
//!   threads))` is a runtime loop. The block size is a compile-time property in cute-rs,
//!   so a kernel is built per `(dtype, num_warps)`; `d` itself is a runtime scalar.
//! * **[`kernel`]`(Rms, ..)`** `rmsnorm(input [batch, d], weight [d]) -> out`: pass 1
//!   accumulates the row's squares in Float32 and reduces them (a butterfly over the
//!   warp, one value per warp through shared memory, a second butterfly in warp 0);
//!   pass 2 re-reads the row and writes `x * rsqrt(sum / d + eps) * w`.
//! * **[`kernel`]`(FusedAdd, ..)`**: pass 1 loads `input` and `residual`, forms
//!   `x = input + residual` in Float32, writes `residual = T(x)` and stashes `x` (Float32) in
//!   dynamic shared memory; pass 2 reads `x` back from shared memory and writes `input =
//!   x * rms * w`. The row is read from global memory once.
//!
//! Float16 and BFloat16 data, Float32 accumulation, `eps` a runtime scalar.
//!
//! * **Gemma variants** ([`Variant::GemmaRms`], [`Variant::GemmaFusedAdd`]): the same kernels
//!   with the weight read as `weight + 1` (FlashInfer's `weight_bias = 1`).
//! * **[`Variant::LayerNorm`]** (`layernorm(input, gamma, beta, eps)`, after the semantics of
//!   `generalLayerNorm` in `norm.cuh`): `(x - mean) * rsqrt(var + eps) * gamma +
//!   beta` per row with Float32 `gamma` and `beta`. The variance is the two-pass `E[(x -
//!   mean)^2]`. The one-block-per-row kernel stashes the row in Float32 shared memory as
//!   the fused kernel does and reduces twice; the multi-row kernel keeps the row in registers.
//!
//! * **Quantising variants** ([`Variant::RmsQuant`], [`Variant::FusedAddQuant`], after
//!   `RMSNormQuantKernel` / `FusedAddRMSNormQuantKernel`): the same two passes, with
//!   the output `fp8_e4m3(clamp(x * rms * w * (1 / scale), +-448))` written as 8 bytes per
//!   thread; `scale` is a one-element Float32 buffer, its reciprocal taken after the
//!   reduction. The fused one keeps `input` and updates `residual`.
//! * **[`Variant::QkRms`]** (`QKRMSNorm`, FlashInfer's `rmsnorm` on 3-D input; see
//!   [`qk_kernel`]): `[tokens, heads, d]` with token and head strides, `tpr` lanes per row (16 for
//!   `d = 128`), several rows per block, as FlashInfer's CuTe DSL kernel.
//!
//! Not yet covered: `d` that is not a multiple of 8 (FlashInfer narrows the vector width to
//! `gcd(8, d)`), strides on the 2-D kernels (FlashInfer takes `stride_input` and
//! `stride_output`), E5M2 output, `layernorm_quant`, the FP8 block-quant, SiLU, DiT and
//! RoPE-fused norms, programmatic dependent launch.
//!
//! `examples/norm.rs` runs it on the simulator and, with `--features cuda` and `GPU=1`,
//! on the GPU against a reference, then benchmarks it.

use cute::prelude::*;

/// Elements per thread per round (16 bytes of a 16-bit type).
pub const VEC: usize = 8;

/// Which kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// `out = rmsnorm(input) * weight`.
    Rms,
    /// `residual += input; input = rmsnorm(residual) * weight`.
    FusedAdd,
    /// `out = rmsnorm(input) * (weight + 1)` (`gemma_rmsnorm`).
    GemmaRms,
    /// `residual += input; input = rmsnorm(residual) * (weight + 1)` (`gemma_fused_add_rmsnorm`).
    GemmaFusedAdd,
    /// `out = (input - mean) * rsqrt(var + eps) * gamma + beta` with Float32 `gamma`, `beta`
    /// (`layernorm`). Parameters `input, gamma, beta, output, d, eps` (+ `batch` for the
    /// multi-row geometry).
    LayerNorm,
    /// `rmsnorm_quant`: `out = fp8_e4m3(clamp(rmsnorm(input) * weight / scale, +-448))`;
    /// parameters `input, weight, out (E4M3), scale (one Float32), d, eps`.
    RmsQuant,
    /// `fused_add_rmsnorm_quant`: `residual += input`, then
    /// `out = fp8_e4m3(clamp(rmsnorm(residual) * weight / scale))`; `input` is left alone.
    /// Parameters `input, residual, weight, out (E4M3), scale, d, eps`.
    FusedAddQuant,
    /// `rmsnorm` of a 3-D `[batch, heads, d]` input (FlashInfer's `QKRMSNorm`), `tpr` lanes per
    /// `(token, head)` ([`qk_config`]); built by [`qk_kernel`], parameters `input, weight, out, d, batch,
    /// heads, in_n, in_h, out_n, out_h, eps` with the strides in 8-element vectors.
    QkRms,
}

impl Variant {
    /// Adds the input to a residual first.
    pub fn is_fused(self) -> bool {
        matches!(self, Variant::FusedAdd | Variant::GemmaFusedAdd | Variant::FusedAddQuant)
    }
    /// Writes a scaled E4M3 output.
    pub fn is_quant(self) -> bool {
        matches!(self, Variant::RmsQuant | Variant::FusedAddQuant)
    }
    /// The constant added to the weight (1 for the Gemma variants).
    pub fn weight_bias(self) -> f32 {
        if matches!(self, Variant::GemmaRms | Variant::GemmaFusedAdd) { 1.0 } else { 0.0 }
    }
    fn tag(self) -> &'static str {
        match self {
            Variant::Rms => "rmsnorm",
            Variant::FusedAdd => "fused_add_rmsnorm",
            Variant::GemmaRms => "gemma_rmsnorm",
            Variant::GemmaFusedAdd => "gemma_fused_add_rmsnorm",
            Variant::LayerNorm => "layernorm",
            Variant::RmsQuant => "rmsnorm_quant",
            Variant::FusedAddQuant => "fused_add_rmsnorm_quant",
            Variant::QkRms => "qk_rmsnorm",
        }
    }
}

/// The element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    F16,
    BF16,
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::BF16 => "bf16",
        }
    }

    /// Round `x` to this type.
    pub fn round(self, x: f32) -> f32 {
        match self {
            Dtype::F16 => F16::from_f32(x).to_f32(),
            Dtype::BF16 => BF16::from_f32(x).to_f32(),
        }
    }

    /// Output tolerance (absolute plus relative): one rounding of the type, with margin.
    pub fn tolerance(self) -> f64 {
        match self {
            Dtype::F16 => 2e-3,
            Dtype::BF16 => 1.6e-2,
        }
    }

    pub fn buffer(self, v: &[f32]) -> KernelArg {
        match self {
            Dtype::F16 => KernelArg::buffer(&v.iter().map(|&x| F16::from_f32(x)).collect::<Vec<_>>()),
            Dtype::BF16 => KernelArg::buffer(&v.iter().map(|&x| BF16::from_f32(x)).collect::<Vec<_>>()),
        }
    }

    pub fn read(self, a: &KernelArg) -> Vec<f32> {
        match self {
            Dtype::F16 => a.to_vec::<F16>().into_iter().map(|x| x.to_f32()).collect(),
            Dtype::BF16 => a.to_vec::<BF16>().into_iter().map(|x| x.to_f32()).collect(),
        }
    }
}

/// Warps per block for rows of `d` elements, as FlashInfer: `min(1024, d / 8)` threads,
/// rounded up to whole warps.
pub fn num_warps(d: usize) -> usize {
    assert!(d >= VEC && d.is_multiple_of(VEC), "d = {d} must be a positive multiple of {VEC}");
    (d / VEC).min(1024).div_ceil(32)
}

/// Floats of shared memory in front of the stashed row in the fused kernel (one per warp,
/// rounded up to 4 so the row's vector accesses stay 16-byte aligned).
pub fn smem_head(num_warps: usize) -> usize {
    num_warps.div_ceil(4) * 4
}

/// Dynamic shared memory the launch needs, in bytes.
pub fn smem_bytes(variant: Variant, d: usize) -> usize {
    match variant {
        Variant::Rms | Variant::GemmaRms | Variant::RmsQuant | Variant::QkRms => 0,
        Variant::FusedAdd | Variant::GemmaFusedAdd | Variant::LayerNorm | Variant::FusedAddQuant => {
            (smem_head(num_warps(d)) + d) * 4
        }
    }
}

/// Sum `v` over the block: a butterfly over each warp, one value per warp through
/// `red[..num_warps]`, a butterfly of those in warp 0, broadcast through `red[0]`.
fn block_sum(v: Float32, red: &Ptr<f32>, tid: Int32, num_warps: usize) -> Float32 {
    let (warp, lane) = (tid >> 5, tid & 31);
    let v = warp_reduction_sum(v, 32);
    if_(lane.eq(0), || red.store_at(warp, v));
    sync_threads();
    if_(warp.eq(0), || {
        let w = if_else_yield(lane.lt(num_warps as i32), || red.load_at(lane), || Float32::constant(0.0));
        red.store_at(0, warp_reduction_sum(w, 32));
    });
    sync_threads();
    red.load_at(0)
}

/// Parameter order of [`kernel`]: `Rms` takes `input, weight, out, d, eps`; `FusedAdd`
/// takes `input, residual, weight, d, eps`. Buffers are `[batch, d]` row-major (`weight`
/// is `[d]`) and 16-byte aligned; `d` is a multiple of 8.
pub fn kernel(dtype: Dtype, variant: Variant, num_warps: usize) -> Kernel {
    if variant == Variant::LayerNorm {
        return match dtype {
            Dtype::F16 => layernorm_row_of::<F16>(dtype, num_warps),
            Dtype::BF16 => layernorm_row_of::<BF16>(dtype, num_warps),
        };
    }
    match dtype {
        Dtype::F16 => kernel_of::<F16>(dtype, variant, num_warps),
        Dtype::BF16 => kernel_of::<BF16>(dtype, variant, num_warps),
    }
}

fn kernel_of<T: Numeric>(dtype: Dtype, variant: Variant, num_warps: usize) -> Kernel {
    assert!((1..=32).contains(&num_warps));
    let threads = 32 * num_warps;
    let name = format!("flashinfer_{}_{}_w{num_warps}", variant.tag(), dtype.name());
    let bias = variant.weight_bias();
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let fused = variant.is_fused();
        let input = param_ptr::<T>("input");
        let residual = fused.then(|| param_ptr::<T>("residual"));
        let weight = param_ptr::<T>("weight");
        let quant = variant.is_quant();
        let output = (!fused && !quant).then(|| param_ptr::<T>("output"));
        let output8 = quant.then(|| param_ptr::<F8E4M3FN>("output"));
        let scale = quant.then(|| param_ptr::<f32>("scale"));
        let d = param_scalar::<i32>("d");
        let eps = param_scalar::<f32>("eps");

        let tid = thread_idx().x;
        let row = block_idx().x;
        let nt = threads as i32;
        let v = VEC as i32;
        // Row length in vectors, and the round count. Every global offset below is `vec
        // index * 8`, which cute-rs proves 8-element aligned for the 16-byte accesses.
        let dv = d / v;
        let rounds = (dv + (nt - 1)) / nt;
        let row_vec = row * dv;

        let (red, stash) = if fused {
            let smem = get_dyn_smem::<f32>(Some(16));
            let stash = smem.offset(smem_head(num_warps) as i32);
            (smem, Some(stash))
        } else {
            (make_smem_tensor::<f32>(Layout::col_major(32)).ptr(), None)
        };

        // Pass 1: the sum of squares.
        let sum_sq = Var::new(Float32::constant(0.0));
        for_range(0, rounds, 1, |i| {
            let col = i * nt + tid;
            if_(col.lt(dv), || {
                let at = (row_vec + col) * v;
                let x = load_vector::<T, VEC>(&input.offset(at), LoadOptions::default());
                if let (Some(residual), Some(stash)) = (&residual, &stash) {
                    let r = load_vector::<T, VEC>(&residual.offset(at), LoadOptions::default());
                    let mut s = sum_sq.get();
                    let mut xs = Vec::with_capacity(VEC);
                    for (xe, re) in x.iter().zip(r.iter()) {
                        let e = xe.to::<f32>() + re.to::<f32>();
                        s = s + e * e;
                        xs.push(e);
                    }
                    sum_sq.set(s);
                    // The residual keeps `T(x)`; the Float32 sum goes to shared memory.
                    let rv: [Value<T>; VEC] = std::array::from_fn(|k| xs[k].to::<T>());
                    store_vector::<T, VEC>(&residual.offset(at), rv, StoreOptions::default());
                    let sbase = col * v;
                    for h in 0..2 {
                        let part: [Value<f32>; 4] = std::array::from_fn(|k| xs[h * 4 + k]);
                        store_vector::<f32, 4>(&stash.offset(sbase + 4 * h as i32), part, StoreOptions::default());
                    }
                } else {
                    let mut s = sum_sq.get();
                    for xe in &x {
                        let e = xe.to::<f32>();
                        s = s + e * e;
                    }
                    sum_sq.set(s);
                }
            });
        });
        // Pass 2 reads what this thread alone wrote or read in pass 1 (its own stash, the
        // row, the weight), so round 0's loads are issued before the block reduction and
        // their latency hides behind it. (FlashInfer loads after the reduction.) A thread
        // with no column in round 0 (`dv < threads`) reads the last one, unused.
        let load_x_w = |col: Int32| {
            let at = (row_vec + col) * v;
            let w = load_vector::<T, VEC>(&weight.offset(col * v), LoadOptions::default());
            let xs: Vec<Float32> = if let Some(stash) = &stash {
                (0..2)
                    .flat_map(|h| load_vector::<f32, 4>(&stash.offset(col * v + 4 * h), LoadOptions::default()))
                    .collect()
            } else {
                load_vector::<T, VEC>(&input.offset(at), LoadOptions::default()).iter().map(|x| x.to::<f32>()).collect()
            };
            (xs, w)
        };
        let (xs0, w0) = load_x_w(tid.min(dv - 1));
        let total = block_sum(sum_sq.get(), &red, tid, num_warps);
        let rms_rcp = cute::math::rsqrt(total / d.to::<f32>() + eps, true);
        let scale_inv = scale.as_ref().map(|s| Float32::constant(1.0) / s.load_at(0));

        // Scale by `rms_rcp * weight` and write (`rmsnorm` writes `output`; the fused
        // kernel overwrites `input`; the quantising kernels write E4M3 `output`).
        let finish = |col: Int32, xs: &[Float32], w: &[Value<T>; VEC]| {
            let vals: Vec<Float32> = (0..VEC)
                .map(|k| {
                    let wk = w[k].to::<f32>();
                    xs[k] * rms_rcp * if bias != 0.0 { wk + bias } else { wk }
                })
                .collect();
            let at = (row_vec + col) * v;
            if let (Some(out8), Some(inv)) = (&output8, scale_inv) {
                store_fp8(out8, at, &vals, inv);
            } else {
                let out: [Value<T>; VEC] = std::array::from_fn(|k| vals[k].to::<T>());
                let dst = output.as_ref().unwrap_or(&input);
                store_vector::<T, VEC>(&dst.offset(at), out, StoreOptions::default());
            }
        };
        if_(tid.lt(dv), || finish(tid, &xs0, &w0));
        for_range(1, rounds, 1, |i| {
            let col = i * nt + tid;
            if_(col.lt(dv), || {
                let (xs, w) = load_x_w(col);
                finish(col, &xs, &w);
            });
        });
    })
}

/// How a row is spread over threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Config {
    /// One block per row of `32 * num_warps` threads ([`kernel`]); `norm.cuh`'s geometry.
    Row { num_warps: usize },
    /// `threads / tpr` rows per block, `tpr` threads per row, each thread owning `nvb`
    /// 16-byte vectors per row, all loaded before any is used ([`kernel_rows`]); the
    /// geometry of FlashInfer's CuTe DSL norm kernels (`flashinfer/norm/kernels/rmsnorm.py`).
    Rows { tpr: usize, threads: usize, nvb: usize },
}

impl Config {
    /// FlashInfer's CuTe DSL choice for rows of `d` elements: `tpr` by `d` (32 up to 3072,
    /// 64 up to 6144, 128 up to 16384, else 256), 128 threads per block (256 above 16384),
    /// and enough vectors per thread to cover the row. Rows of 128 or fewer use the
    /// one-block-per-row kernel.
    pub fn for_d(d: usize) -> Config {
        assert!(d >= VEC && d.is_multiple_of(VEC), "d = {d} must be a positive multiple of {VEC}");
        if d <= 128 {
            return Config::Row { num_warps: num_warps(d) };
        }
        let tpr = match d {
            ..=3072 => 32,
            3073..=6144 => 64,
            6145..=16384 => 128,
            _ => 256,
        };
        let threads = if d <= 16384 { 128 } else { 256 };
        Config::Rows { tpr, threads, nvb: (d / VEC).div_ceil(tpr) }
    }

    /// The faster geometry for `variant` on `batch` rows of `d`, as measured on an RTX 2000
    /// Ada (22 SMs): the multi-row one when rows are long (the single block per row is then
    /// a latency chain at one block per SM) or the batch is large; the one-block-per-row one
    /// for short rows and small batches, where the multi-row kernel has too few threads to
    /// cover the latency.
    pub fn choose(variant: Variant, batch: usize, d: usize) -> Config {
        // Measured on the same GPU, `tpr` of 128 or more beats the `for_d` choice for plain
        // `Rms` rows of 3072..=8192, and `FusedAddQuant` is faster still with fewer
        // registers per thread. At `tpr = 32` that kernel holds 12 vectors a thread
        // (200+ registers, two blocks per SM, six waves for 989 rows). A sweep of `tpr` and
        // block size over the benchmark shapes gives the table below. The `Rms` kernel at
        // 8192 with `tpr = 128` takes 125 registers, four blocks per SM, so 99 rows need two
        // waves; wider rows use fewer registers and all 99 rows sit on the GPU together.
        if (3072..=8192).contains(&d) {
            let tuned = |tpr: usize| Config::Rows { tpr, threads: tpr.max(128), nvb: (d / VEC).div_ceil(tpr) };
            match variant {
                Variant::Rms if batch < 256 && d == 8192 => return tuned(if batch >= 64 { 512 } else { 256 }),
                Variant::Rms if d <= 4096 => return tuned(128),
                Variant::FusedAddQuant if batch <= 19 => return tuned(512),
                Variant::FusedAddQuant if batch < 256 && d <= 4096 => return tuned(if d <= 3072 { 128 } else { 256 }),
                Variant::FusedAddQuant if batch >= 256 && d <= 3072 => return tuned(64),
                _ => {}
            }
        }
        let rows = match variant {
            Variant::Rms | Variant::GemmaRms | Variant::RmsQuant => d >= 4096 || batch >= 256,
            Variant::FusedAddQuant => d >= 8192 || batch >= 256,
            Variant::FusedAdd | Variant::GemmaFusedAdd | Variant::LayerNorm | Variant::QkRms => {
                d >= 8192 || (d >= 4096 && batch >= 256)
            }
        };
        if rows { Config::for_d(d) } else { Config::row(d) }
    }

    /// `norm.cuh`'s one-block-per-row geometry for rows of `d` elements.
    pub fn row(d: usize) -> Config {
        Config::Row { num_warps: num_warps(d) }
    }

    pub fn kernel(self, dtype: Dtype, variant: Variant) -> Kernel {
        if variant == Variant::QkRms {
            let Config::Rows { tpr, threads, nvb } = self else { panic!("QkRms needs a Rows config ([`qk_config`])") };
            return qk_kernel(dtype, tpr, threads, nvb);
        }
        match self {
            Config::Row { num_warps } => kernel(dtype, variant, num_warps),
            Config::Rows { tpr, threads, nvb } => kernel_rows(dtype, variant, tpr, threads, nvb),
        }
    }
}

/// The multi-row kernel: parameters as [`kernel`] plus a trailing `batch` scalar. Block
/// `b` takes rows `b * rows .. (b + 1) * rows`, `rows = threads / tpr`; thread `t` works
/// on row `t / tpr`, vectors `lane + j * tpr` for `j < nvb` (`lane = t % tpr`). Every
/// vector of the row is loaded before the reduction (the weight too), kept in registers
/// across it, and reused after it, so each row is read once. Rows of one warp reduce in
/// the warp; wider rows reduce one value per warp through shared memory.
pub fn kernel_rows(dtype: Dtype, variant: Variant, tpr: usize, threads: usize, nvb: usize) -> Kernel {
    if variant == Variant::LayerNorm {
        return match dtype {
            Dtype::F16 => layernorm_rows_of::<F16>(dtype, tpr, threads, nvb),
            Dtype::BF16 => layernorm_rows_of::<BF16>(dtype, tpr, threads, nvb),
        };
    }
    match dtype {
        Dtype::F16 => kernel_rows_of::<F16>(dtype, variant, tpr, threads, nvb),
        Dtype::BF16 => kernel_rows_of::<BF16>(dtype, variant, tpr, threads, nvb),
    }
}

fn kernel_rows_of<T: Numeric>(dtype: Dtype, variant: Variant, tpr: usize, threads: usize, nvb: usize) -> Kernel {
    assert!(tpr >= 32 && tpr.is_power_of_two() && threads.is_multiple_of(tpr) && nvb >= 1);
    let rows = threads / tpr;
    let wpr = tpr / 32;
    let name = format!("flashinfer_{}_rows_{}_t{tpr}x{rows}_v{nvb}", variant.tag(), dtype.name());
    let bias = variant.weight_bias();
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let fused = variant.is_fused();
        let input = param_ptr::<T>("input");
        let residual = fused.then(|| param_ptr::<T>("residual"));
        let weight = param_ptr::<T>("weight");
        let quant = variant.is_quant();
        let output = (!fused && !quant).then(|| param_ptr::<T>("output"));
        let output8 = quant.then(|| param_ptr::<F8E4M3FN>("output"));
        let scale = quant.then(|| param_ptr::<f32>("scale"));
        let d = param_scalar::<i32>("d");
        let eps = param_scalar::<f32>("eps");
        let batch = param_scalar::<i32>("batch");

        let tid = thread_idx().x;
        let (row_in_block, lane) = (tid / tpr as i32, tid % tpr as i32);
        let row = block_idx().x * rows as i32 + row_in_block;
        let row_ok = row.lt(batch);
        let v = VEC as i32;
        let dv = d / v;
        // Rows past the batch and columns past the row read valid addresses (the last row,
        // the last column) and are masked in the sum and the stores.
        let row_vec = row.min(batch - 1) * dv;
        let cols: Vec<Int32> = (0..nvb).map(|j| lane + (j * tpr) as i32).collect();
        let oks: Vec<Boolean> = cols.iter().map(|c| c.lt(dv)).collect();
        let at: Vec<Int32> = cols.iter().map(|c| (row_vec + c.min(dv - 1)) * v).collect();

        // Issue every load first.
        let ws: Vec<[Value<T>; VEC]> = cols
            .iter()
            .map(|c| load_vector::<T, VEC>(&weight.offset(c.min(dv - 1) * v), LoadOptions::default()))
            .collect();
        let xs: Vec<[Value<T>; VEC]> =
            at.iter().map(|a| load_vector::<T, VEC>(&input.offset(*a), LoadOptions::default())).collect();
        let rs: Vec<[Value<T>; VEC]> = match &residual {
            Some(r) => at.iter().map(|a| load_vector::<T, VEC>(&r.offset(*a), LoadOptions::default())).collect(),
            None => vec![],
        };
        // The values pass 2 scales: the input, or in the fused kernel the Float32 sum.
        let vals: Vec<Vec<Float32>> = (0..nvb)
            .map(|j| {
                (0..VEC)
                    .map(|k| {
                        let x = xs[j][k].to::<f32>();
                        if fused { x + rs[j][k].to::<f32>() } else { x }
                    })
                    .collect()
            })
            .collect();
        let mut sum = Float32::constant(0.0);
        for (j, vs) in vals.iter().enumerate() {
            let mut part = Float32::constant(0.0);
            for e in vs {
                part = part + *e * *e;
            }
            sum = sum + select(oks[j], part, Float32::constant(0.0));
        }
        if let Some(residual) = &residual {
            for (j, vs) in vals.iter().enumerate() {
                let rv: [Value<T>; VEC] = std::array::from_fn(|k| vs[k].to::<T>());
                if_(oks[j] & row_ok, || store_vector::<T, VEC>(&residual.offset(at[j]), rv, StoreOptions::default()));
            }
        }

        let sum = warp_reduction_sum(sum, 32);
        let total = if wpr == 1 {
            sum
        } else {
            let red = make_smem_tensor::<f32>(Layout::col_major(rows * wpr)).ptr();
            let (warp, l) = (lane >> 5, lane & 31);
            if_(l.eq(0), || red.store_at(row_in_block * wpr as i32 + warp, sum));
            sync_threads();
            let mut t = red.load_at(row_in_block * wpr as i32);
            for w in 1..wpr {
                t = t + red.load_at(row_in_block * wpr as i32 + w as i32);
            }
            t
        };
        let rms_rcp = cute::math::rsqrt(total / d.to::<f32>() + eps, true);

        let dst = output.as_ref().unwrap_or(&input);
        let scale_inv = scale.as_ref().map(|s| Float32::constant(1.0) / s.load_at(0));
        for j in 0..nvb {
            let vs: Vec<Float32> = (0..VEC)
                .map(|k| {
                    let wk = ws[j][k].to::<f32>();
                    vals[j][k] * rms_rcp * if bias != 0.0 { wk + bias } else { wk }
                })
                .collect();
            if let (Some(out8), Some(inv)) = (&output8, scale_inv) {
                if_(oks[j] & row_ok, || store_fp8(out8, at[j], &vs, inv));
            } else {
                let out: [Value<T>; VEC] = std::array::from_fn(|k| vs[k].to::<T>());
                if_(oks[j] & row_ok, || store_vector::<T, VEC>(&dst.offset(at[j]), out, StoreOptions::default()));
            }
        }
    })
}

/// Largest finite E4M3 value; FlashInfer clamps to it before the cast.
pub const E4M3_MAX: f32 = 448.0;

/// Scale `vals` by `scale_inv`, clamp to `+-448` and store them as 8 E4M3 bytes at `at`.
fn store_fp8(out: &Ptr<F8E4M3FN>, at: Int32, vals: &[Float32], scale_inv: Float32) {
    let q: [Value<F8E4M3FN>; VEC] = std::array::from_fn(|k| {
        let y = vals[k] * scale_inv;
        y.max(Float32::constant(-E4M3_MAX)).min(Float32::constant(E4M3_MAX)).to::<F8E4M3FN>()
    });
    store_vector::<F8E4M3FN, VEC>(&out.offset(at), q, StoreOptions::default());
}

/// The geometry [`Variant::QkRms`] uses for head dimension `d`: `tpr = min(32, d / 8)` rounded
/// up to a power of two (at least 4) threads per row, 128 threads per block, and `nvb`
/// 16-byte vectors per thread; `d = 128` gets 16 threads per row and 8 rows per block.
pub fn qk_config(d: usize) -> Config {
    assert!(d >= VEC && d.is_multiple_of(VEC), "d = {d} must be a positive multiple of {VEC}");
    let tpr = (d / VEC).next_power_of_two().clamp(4, 32);
    Config::Rows { tpr, threads: 128, nvb: (d / VEC).div_ceil(tpr) }
}

/// `QKRMSNorm` (`norm.cuh`): `rmsnorm` of a `[batch, heads, d]` input with arbitrary
/// token and head strides, as FlashInfer's `rmsnorm` does for 3-D input (Q and K head norms),
/// in the multi-row geometry of its CuTe DSL `QKRMSNormKernel`: `tpr` threads (at most one
/// warp) per `(token, head)` row, `threads / tpr` rows per block, every vector loaded before
/// the reduction and kept in registers, a butterfly over the `tpr` lanes (no shared memory).
/// Parameters: `input, weight, output, d, batch, heads, in_n, in_h, out_n, out_h, eps`, the
/// strides counted in 8-element vectors (all element strides must be multiples of 8).
pub fn qk_kernel(dtype: Dtype, tpr: usize, threads: usize, nvb: usize) -> Kernel {
    match dtype {
        Dtype::F16 => qk_of::<F16>(dtype, tpr, threads, nvb),
        Dtype::BF16 => qk_of::<BF16>(dtype, tpr, threads, nvb),
    }
}

fn qk_of<T: Numeric>(dtype: Dtype, tpr: usize, threads: usize, nvb: usize) -> Kernel {
    assert!((4..=32).contains(&tpr) && tpr.is_power_of_two() && threads.is_multiple_of(tpr) && nvb >= 1);
    let rows = threads / tpr;
    let name = format!("flashinfer_qk_rmsnorm_{}_t{tpr}x{rows}_v{nvb}", dtype.name());
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let input = param_ptr::<T>("input");
        let weight = param_ptr::<T>("weight");
        let output = param_ptr::<T>("output");
        let d = param_scalar::<i32>("d");
        let batch = param_scalar::<i32>("batch");
        let heads = param_scalar::<i32>("heads");
        let in_n = param_scalar::<i32>("in_n");
        let in_h = param_scalar::<i32>("in_h");
        let out_n = param_scalar::<i32>("out_n");
        let out_h = param_scalar::<i32>("out_h");
        let eps = param_scalar::<f32>("eps");

        let tid = thread_idx().x;
        let (row_in_block, lane) = (tid / tpr as i32, tid % tpr as i32);
        let job = block_idx().x * rows as i32 + row_in_block;
        let ok = job.lt(batch * heads);
        // Rows past the last read the last row and are masked in the stores.
        let job_c = job.min(batch * heads - 1);
        let (n, h) = (job_c / heads, job_c % heads);
        let (in_base, out_base) = (n * in_n + h * in_h, n * out_n + h * out_h);
        let v = VEC as i32;
        let dv = d / v;
        let cols: Vec<Int32> = (0..nvb).map(|j| lane + (j * tpr) as i32).collect();
        let oks: Vec<Boolean> = cols.iter().map(|c| c.lt(dv)).collect();
        let cc: Vec<Int32> = cols.iter().map(|c| c.min(dv - 1)).collect();

        let ws: Vec<[Value<T>; VEC]> =
            cc.iter().map(|c| load_vector::<T, VEC>(&weight.offset(*c * v), LoadOptions::default())).collect();
        let xs: Vec<[Value<T>; VEC]> = cc
            .iter()
            .map(|c| load_vector::<T, VEC>(&input.offset((in_base + *c) * v), LoadOptions::default()))
            .collect();
        let mut sum = Float32::constant(0.0);
        for (j, x) in xs.iter().enumerate() {
            let mut part = Float32::constant(0.0);
            for e in x {
                let e = e.to::<f32>();
                part = part + e * e;
            }
            sum = sum + select(oks[j], part, Float32::constant(0.0));
        }
        let rms_rcp = cute::math::rsqrt(warp_reduction_sum(sum, tpr) / d.to::<f32>() + eps, true);
        for j in 0..nvb {
            let out: [Value<T>; VEC] =
                std::array::from_fn(|k| (xs[j][k].to::<f32>() * rms_rcp * ws[j][k].to::<f32>()).to::<T>());
            if_(oks[j] & ok, || {
                store_vector::<T, VEC>(&output.offset((out_base + cc[j]) * v), out, StoreOptions::default())
            });
        }
    })
}

/// Load the `VEC` Float32 values of `ptr` starting at `at` (a multiple of 4): two 16-byte loads.
fn load_f32_vec(ptr: &Ptr<f32>, at: Int32) -> Vec<Float32> {
    (0..2).flat_map(|h| load_vector::<f32, 4>(&ptr.offset(at + 4 * h), LoadOptions::default())).collect()
}

/// [`Variant::LayerNorm`] with one block per row (parameters `input, gamma, beta, output, d,
/// eps`): pass 1 sums the row and stashes it in Float32 shared memory, the block reduces it to
/// the mean, pass 2 sums `(x - mean)^2` from the stash, and pass 3 writes
/// `(x - mean) * rsqrt(var + eps) * gamma + beta`.
fn layernorm_row_of<T: Numeric>(dtype: Dtype, num_warps: usize) -> Kernel {
    assert!((1..=32).contains(&num_warps));
    let threads = 32 * num_warps;
    let name = format!("flashinfer_layernorm_{}_w{num_warps}", dtype.name());
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let input = param_ptr::<T>("input");
        let gamma = param_ptr::<f32>("gamma");
        let beta = param_ptr::<f32>("beta");
        let output = param_ptr::<T>("output");
        let d = param_scalar::<i32>("d");
        let eps = param_scalar::<f32>("eps");

        let tid = thread_idx().x;
        let row = block_idx().x;
        let nt = threads as i32;
        let v = VEC as i32;
        let dv = d / v;
        let rounds = (dv + (nt - 1)) / nt;
        let row_vec = row * dv;
        let red = get_dyn_smem::<f32>(Some(16));
        let stash = red.offset(smem_head(num_warps) as i32);
        let d_f = d.to::<f32>();

        let sum = Var::new(Float32::constant(0.0));
        for_range(0, rounds, 1, |i| {
            let col = i * nt + tid;
            if_(col.lt(dv), || {
                let x = load_vector::<T, VEC>(&input.offset((row_vec + col) * v), LoadOptions::default());
                let xs: Vec<Float32> = x.iter().map(|e| e.to::<f32>()).collect();
                let mut s = sum.get();
                for e in &xs {
                    s = s + *e;
                }
                sum.set(s);
                for h in 0..2 {
                    let part: [Value<f32>; 4] = std::array::from_fn(|k| xs[h * 4 + k]);
                    store_vector::<f32, 4>(&stash.offset(col * v + 4 * h as i32), part, StoreOptions::default());
                }
            });
        });
        let mean = block_sum(sum.get(), &red, tid, num_warps) / d_f;
        sync_threads(); // `red` is read by every thread before it is written again
        let sq = Var::new(Float32::constant(0.0));
        for_range(0, rounds, 1, |i| {
            let col = i * nt + tid;
            if_(col.lt(dv), || {
                let mut s = sq.get();
                for e in load_f32_vec(&stash, col * v) {
                    let c = e - mean;
                    s = s + c * c;
                }
                sq.set(s);
            });
        });
        let rstd = cute::math::rsqrt(block_sum(sq.get(), &red, tid, num_warps) / d_f + eps, true);
        for_range(0, rounds, 1, |i| {
            let col = i * nt + tid;
            if_(col.lt(dv), || {
                let (xs, g, b) =
                    (load_f32_vec(&stash, col * v), load_f32_vec(&gamma, col * v), load_f32_vec(&beta, col * v));
                let out: [Value<T>; VEC] = std::array::from_fn(|k| ((xs[k] - mean) * rstd * g[k] + b[k]).to::<T>());
                store_vector::<T, VEC>(&output.offset((row_vec + col) * v), out, StoreOptions::default());
            });
        });
    })
}

/// [`Variant::LayerNorm`] with `threads / tpr` rows per block (parameters as the row kernel
/// plus `batch`): the row stays in registers across the two reductions, as [`kernel_rows`].
fn layernorm_rows_of<T: Numeric>(dtype: Dtype, tpr: usize, threads: usize, nvb: usize) -> Kernel {
    assert!(tpr >= 32 && tpr.is_power_of_two() && threads.is_multiple_of(tpr) && nvb >= 1);
    let rows = threads / tpr;
    let wpr = tpr / 32;
    let name = format!("flashinfer_layernorm_rows_{}_t{tpr}x{rows}_v{nvb}", dtype.name());
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let input = param_ptr::<T>("input");
        let gamma = param_ptr::<f32>("gamma");
        let beta = param_ptr::<f32>("beta");
        let output = param_ptr::<T>("output");
        let d = param_scalar::<i32>("d");
        let eps = param_scalar::<f32>("eps");
        let batch = param_scalar::<i32>("batch");

        let tid = thread_idx().x;
        let (row_in_block, lane) = (tid / tpr as i32, tid % tpr as i32);
        let row = block_idx().x * rows as i32 + row_in_block;
        let row_ok = row.lt(batch);
        let v = VEC as i32;
        let dv = d / v;
        let d_f = d.to::<f32>();
        let row_vec = row.min(batch - 1) * dv;
        let cols: Vec<Int32> = (0..nvb).map(|j| lane + (j * tpr) as i32).collect();
        let oks: Vec<Boolean> = cols.iter().map(|c| c.lt(dv)).collect();
        let cc: Vec<Int32> = cols.iter().map(|c| c.min(dv - 1)).collect();

        let xs: Vec<Vec<Float32>> = cc
            .iter()
            .map(|c| {
                load_vector::<T, VEC>(&input.offset((row_vec + *c) * v), LoadOptions::default())
                    .iter()
                    .map(|e| e.to::<f32>())
                    .collect()
            })
            .collect();
        // The sum over the row of `f(x)` for each of its elements, reduced over its `tpr` threads.
        let row_sum = |f: &dyn Fn(Float32) -> Float32| -> Float32 {
            let mut sum = Float32::constant(0.0);
            for (j, x) in xs.iter().enumerate() {
                let mut part = Float32::constant(0.0);
                for e in x {
                    part = part + f(*e);
                }
                sum = sum + select(oks[j], part, Float32::constant(0.0));
            }
            let sum = warp_reduction_sum(sum, 32);
            if wpr == 1 {
                return sum;
            }
            let red = make_smem_tensor::<f32>(Layout::col_major(rows * wpr)).ptr();
            let (warp, l) = (lane >> 5, lane & 31);
            if_(l.eq(0), || red.store_at(row_in_block * wpr as i32 + warp, sum));
            sync_threads();
            let mut t = red.load_at(row_in_block * wpr as i32);
            for w in 1..wpr {
                t = t + red.load_at(row_in_block * wpr as i32 + w as i32);
            }
            t
        };
        let mean = row_sum(&|e| e) / d_f;
        let rstd = cute::math::rsqrt(row_sum(&|e| (e - mean) * (e - mean)) / d_f + eps, true);
        for j in 0..nvb {
            let (g, b) = (load_f32_vec(&gamma, cc[j] * v), load_f32_vec(&beta, cc[j] * v));
            let out: [Value<T>; VEC] = std::array::from_fn(|k| ((xs[j][k] - mean) * rstd * g[k] + b[k]).to::<T>());
            if_(oks[j] & row_ok, || {
                store_vector::<T, VEC>(&output.offset((row_vec + cc[j]) * v), out, StoreOptions::default())
            });
        }
    })
}

/// A test problem: `[batch, d]` rows of `input` and `residual`, and a `weight` of `d`,
/// every value exactly representable in `dtype`.
pub struct Problem {
    pub dtype: Dtype,
    pub batch: usize,
    pub d: usize,
    pub eps: f32,
    pub input: Vec<f32>,
    pub residual: Vec<f32>,
    pub weight: Vec<f32>,
    /// How rows map to threads; [`Config::for_d`] unless changed.
    pub config: Config,
    /// The quantising variants' scale (`out = x / scale` before the E4M3 cast).
    pub scale: f32,
    /// [`Variant::QkRms`]: heads per token (`batch` is the token count) and the element
    /// strides `[in_n, in_h, out_n, out_h]` of input and output; `Problem::new` makes the
    /// 2-D case (one head, `[d, d, d, d]`).
    pub heads: usize,
    pub strides: [usize; 4],
}

impl Problem {
    pub fn new(dtype: Dtype, batch: usize, d: usize, seed: u64) -> Problem {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut fill = |n: usize, scale: f32, shift: f32| {
            (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let u = (state >> 33) as f32 / (1u64 << 31) as f32;
                    dtype.round((u * 2.0 - 1.0) * scale + shift)
                })
                .collect::<Vec<_>>()
        };
        Problem {
            dtype,
            batch,
            d,
            eps: 1e-6,
            input: fill(batch * d, 3.0, 0.0),
            residual: fill(batch * d, 3.0, 0.0),
            weight: fill(d, 1.0, 1.0),
            config: Config::for_d(d),
            scale: 0.02,
            heads: 1,
            strides: [d; 4],
        }
    }

    /// A [`Variant::QkRms`] problem: `batch` tokens of `heads` heads of `d`, the input a
    /// slice of a packed `[batch, heads + 2, d]` (Q, K, V) tensor (token stride `(heads + 2)
    /// * d`, head stride `d`), the output a dense `[batch, heads, d]`.
    pub fn qk(dtype: Dtype, batch: usize, heads: usize, d: usize, seed: u64) -> Problem {
        let mut p = Problem::new(dtype, batch * (heads + 2), d, seed);
        p.batch = batch;
        p.heads = heads;
        p.strides = [(heads + 2) * d, d, heads * d, d];
        p.config = qk_config(d);
        p
    }

    pub fn launch(&self, variant: Variant) -> LaunchConfig {
        match self.config {
            Config::Row { num_warps } => LaunchConfig::new([self.batch as u32, 1, 1], [32 * num_warps as u32, 1, 1])
                .with_dynamic_smem(smem_bytes(variant, self.d) as u32),
            Config::Rows { tpr, threads, .. } => {
                let rows_total = if variant == Variant::QkRms { self.batch * self.heads } else { self.batch };
                LaunchConfig::new([rows_total.div_ceil(threads / tpr) as u32, 1, 1], [threads as u32, 1, 1])
            }
        }
    }

    /// The kernel's arguments (see [`kernel`]).
    pub fn args(&self, variant: Variant) -> Vec<KernelArg> {
        let t = self.dtype;
        let scalars = [KernelArg::ScalarI32(self.d as i32), KernelArg::ScalarF32(self.eps)];
        let mut a = match variant {
            Variant::Rms | Variant::GemmaRms => {
                vec![t.buffer(&self.input), t.buffer(&self.weight), t.buffer(&vec![0.0; self.input.len()])]
            }
            Variant::FusedAdd | Variant::GemmaFusedAdd => {
                vec![t.buffer(&self.input), t.buffer(&self.residual), t.buffer(&self.weight)]
            }
            Variant::LayerNorm => vec![
                t.buffer(&self.input),
                KernelArg::buffer(&self.weight),
                KernelArg::buffer(&self.beta()),
                t.buffer(&vec![0.0; self.input.len()]),
            ],
            Variant::RmsQuant => {
                vec![t.buffer(&self.input), t.buffer(&self.weight), self.out8(), KernelArg::buffer(&[self.scale])]
            }
            Variant::FusedAddQuant => vec![
                t.buffer(&self.input),
                t.buffer(&self.residual),
                t.buffer(&self.weight),
                self.out8(),
                KernelArg::buffer(&[self.scale]),
            ],
            Variant::QkRms => {
                let [in_n, in_h, out_n, out_h] = self.strides.map(|x| {
                    assert!(x.is_multiple_of(VEC), "QkRms strides must be multiples of {VEC}");
                    (x / VEC) as i32
                });
                return vec![
                    t.buffer(&self.input),
                    t.buffer(&self.weight),
                    t.buffer(&vec![0.0; self.batch * self.strides[2]]),
                    KernelArg::ScalarI32(self.d as i32),
                    KernelArg::ScalarI32(self.batch as i32),
                    KernelArg::ScalarI32(self.heads as i32),
                    KernelArg::ScalarI32(in_n),
                    KernelArg::ScalarI32(in_h),
                    KernelArg::ScalarI32(out_n),
                    KernelArg::ScalarI32(out_h),
                    KernelArg::ScalarF32(self.eps),
                ];
            }
        };
        a.extend(scalars);
        if matches!(self.config, Config::Rows { .. }) {
            a.push(KernelArg::ScalarI32(self.batch as i32));
        }
        a
    }

    /// A zeroed E4M3 output buffer for the quantising variants.
    fn out8(&self) -> KernelArg {
        KernelArg::fp8_from_f32::<F8E4M3FN>(&vec![0.0; self.input.len()])
    }

    /// The LayerNorm `beta`: a fixed pattern in `[-0.5, 0.5]` (`gamma` is `weight`).
    pub fn beta(&self) -> Vec<f32> {
        (0..self.d).map(|k| ((k * 37 % 17) as f32 - 8.0) / 16.0).collect()
    }

    /// `out = x * rsqrt(mean(x^2) + eps) * (w + bias)` per row of `x`, in f64.
    fn norm(&self, x: &[f64], bias: f64) -> Vec<f64> {
        let d = self.d;
        let mut y = vec![0.0; x.len()];
        for (xr, yr) in x.chunks(d).zip(y.chunks_mut(d)) {
            let scale = 1.0 / (xr.iter().map(|v| v * v).sum::<f64>() / d as f64 + self.eps as f64).sqrt();
            for k in 0..d {
                yr[k] = xr[k] * scale * (self.weight[k] as f64 + bias);
            }
        }
        y
    }

    /// The f64 reference of `rmsnorm`.
    pub fn reference_rms(&self) -> Vec<f64> {
        self.norm(&self.input.iter().map(|&v| v as f64).collect::<Vec<_>>(), 0.0)
    }

    /// The f64 reference of `gemma_rmsnorm` (`weight + 1`).
    pub fn reference_gemma_rms(&self) -> Vec<f64> {
        self.norm(&self.input.iter().map(|&v| v as f64).collect::<Vec<_>>(), 1.0)
    }

    /// The f64 reference of `layernorm` with `gamma = weight` and [`Problem::beta`].
    pub fn reference_layernorm(&self) -> Vec<f64> {
        let (d, beta) = (self.d, self.beta());
        let mut y = Vec::with_capacity(self.input.len());
        for xr in self.input.chunks(d) {
            let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = xr.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let rstd = 1.0 / (var + self.eps as f64).sqrt();
            y.extend((0..d).map(|k| (xr[k] as f64 - mean) * rstd * self.weight[k] as f64 + beta[k] as f64));
        }
        y
    }

    /// The f64 reference of `rmsnorm_quant`: the normalised rows divided by `scale` and clamped
    /// to `+-448`, before the E4M3 rounding.
    pub fn reference_rms_quant(&self) -> Vec<f64> {
        self.quant_ref(self.reference_rms())
    }

    /// The f64 reference of `fused_add_rmsnorm_quant`: the pre-rounding quantised rows and the
    /// new `residual`.
    pub fn reference_fused_quant(&self) -> (Vec<f64>, Vec<f64>) {
        let (out, res) = self.reference_fused();
        (self.quant_ref(out), res)
    }

    fn quant_ref(&self, y: Vec<f64>) -> Vec<f64> {
        let m = E4M3_MAX as f64;
        y.into_iter().map(|v| (v / self.scale as f64).clamp(-m, m)).collect()
    }

    /// The f64 reference of [`Variant::QkRms`]: dense `[batch, heads, d]`.
    pub fn reference_qk(&self) -> Vec<f64> {
        let (d, [in_n, in_h, ..]) = (self.d, self.strides);
        let mut y = Vec::with_capacity(self.batch * self.heads * d);
        for n in 0..self.batch {
            for h in 0..self.heads {
                let row = &self.input[n * in_n + h * in_h..][..d];
                let scale =
                    1.0 / (row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / d as f64 + self.eps as f64).sqrt();
                y.extend((0..d).map(|k| row[k] as f64 * scale * self.weight[k] as f64));
            }
        }
        y
    }

    /// Check E4M3 `got` (decoded) against the pre-rounding `want`: within one E4M3 step (an f32
    /// kernel and an f64 reference can round a boundary value either way), and at least 99% of
    /// the codes equal to the rounding of `want`. Returns the fraction of differing codes.
    pub fn check_fp8(&self, what: &str, got: &[f32], want: &[f64]) -> f64 {
        let mut diff = 0usize;
        for (i, (&g, &r)) in got.iter().zip(want).enumerate() {
            let e = (g as f64 - r).abs();
            assert!(
                e <= 0.13 * r.abs() + 0.004,
                "{what}: element {i}: {g} against {r} ({} batch {} d {})",
                self.dtype.name(),
                self.batch,
                self.d
            );
            diff += (F8E4M3FN::from_f32(r as f32).to_f32() != g) as usize;
        }
        let frac = diff as f64 / got.len() as f64;
        assert!(frac < 0.01, "{what}: {:.2}% of E4M3 codes differ from the reference", frac * 100.0);
        frac
    }

    /// The f64 reference of `fused_add_rmsnorm`: the normalised rows (the new `input`) and
    /// the new `residual`, the sum rounded to the type.
    pub fn reference_fused(&self) -> (Vec<f64>, Vec<f64>) {
        self.fused_with_bias(0.0)
    }

    /// The f64 reference of `gemma_fused_add_rmsnorm`.
    pub fn reference_gemma_fused(&self) -> (Vec<f64>, Vec<f64>) {
        self.fused_with_bias(1.0)
    }

    fn fused_with_bias(&self, bias: f64) -> (Vec<f64>, Vec<f64>) {
        let sum: Vec<f64> = self.input.iter().zip(&self.residual).map(|(&a, &b)| a as f64 + b as f64).collect();
        let res = sum.iter().map(|&s| self.dtype.round(s as f32) as f64).collect();
        (self.norm(&sum, bias), res)
    }

    /// Check `got` against `want` (absolute plus relative tolerance, as
    /// [`Dtype::tolerance`]); returns the largest error.
    pub fn check(&self, what: &str, got: &[f32], want: &[f64]) -> f64 {
        let tol = self.dtype.tolerance();
        let mut worst = 0.0f64;
        for (i, (&g, &r)) in got.iter().zip(want).enumerate() {
            let e = (g as f64 - r).abs() / (1.0 + r.abs());
            assert!(
                e <= tol,
                "{what}: element {i}: {g} against {r} ({} batch {} d {})",
                self.dtype.name(),
                self.batch,
                self.d
            );
            worst = worst.max(e);
        }
        worst
    }
}

/// Simulate both kernels on `p` and check them against the f64 references; returns the
/// largest scaled error of each (rmsnorm, fused output, fused residual).
pub fn simulate(p: &Problem) -> (f64, f64, f64) {
    let k = p.config.kernel(p.dtype, Variant::Rms);
    let mut args = p.args(Variant::Rms);
    k.simulate(p.launch(Variant::Rms), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let e_rms = p.check("rmsnorm", &p.dtype.read(&args[2]), &p.reference_rms());

    let k = p.config.kernel(p.dtype, Variant::FusedAdd);
    let mut args = p.args(Variant::FusedAdd);
    k.simulate(p.launch(Variant::FusedAdd), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let (want_out, want_res) = p.reference_fused();
    let e_out = p.check("fused output", &p.dtype.read(&args[0]), &want_out);
    let e_res = p.check("fused residual", &p.dtype.read(&args[1]), &want_res);
    (e_rms, e_out, e_res)
}

/// Simulate `variant` on `p` and check it against its f64 reference; returns the largest
/// scaled error of the output and of the residual (zero when the variant has none).
pub fn simulate_variant(p: &Problem, variant: Variant) -> (f64, f64) {
    let k = p.config.kernel(p.dtype, variant);
    let mut args = p.args(variant);
    k.simulate(p.launch(variant), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    match variant {
        Variant::Rms => (p.check("rmsnorm", &p.dtype.read(&args[2]), &p.reference_rms()), 0.0),
        Variant::GemmaRms => (p.check("gemma_rmsnorm", &p.dtype.read(&args[2]), &p.reference_gemma_rms()), 0.0),
        Variant::LayerNorm => (p.check("layernorm", &p.dtype.read(&args[3]), &p.reference_layernorm()), 0.0),
        Variant::RmsQuant => (p.check_fp8("rmsnorm_quant", &args[2].to_f32(), &p.reference_rms_quant()), 0.0),
        Variant::FusedAddQuant => {
            let (out, res) = p.reference_fused_quant();
            (
                p.check_fp8("fused_add_rmsnorm_quant", &args[3].to_f32(), &out),
                p.check("residual", &p.dtype.read(&args[1]), &res),
            )
        }
        Variant::QkRms => (p.check("qk_rmsnorm", &p.dtype.read(&args[2]), &p.reference_qk()), 0.0),
        Variant::FusedAdd | Variant::GemmaFusedAdd => {
            let (out, res) = if variant == Variant::FusedAdd { p.reference_fused() } else { p.reference_gemma_fused() };
            (
                p.check("fused output", &p.dtype.read(&args[0]), &out),
                p.check("fused residual", &p.dtype.read(&args[1]), &res),
            )
        }
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

    use super::{Config, Dtype, Problem, Variant};

    /// Kernels compiled and loaded on first use, per `(dtype, variant, config)`.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: &'c Toolchain,
        loaded: HashMap<(Dtype, Variant, Config), (Kernel, CudaFunction<'c>)>,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &'c Toolchain) -> Kernels<'c> {
            Kernels { ctx, tc, loaded: HashMap::new() }
        }

        pub fn get(
            &mut self,
            dtype: Dtype,
            variant: Variant,
            config: Config,
        ) -> Result<&(Kernel, CudaFunction<'c>), Box<dyn Error>> {
            if !self.loaded.contains_key(&(dtype, variant, config)) {
                let arch = self.ctx.best_target(self.tc).ok_or("no target for this GPU")?;
                let k = config.kernel(dtype, variant);
                let f = self.ctx.load(&k.compile_with(self.tc, &CompileOptions::new().gpu_arch(arch))?)?;
                self.loaded.insert((dtype, variant, config), (k, f));
            }
            Ok(&self.loaded[&(dtype, variant, config)])
        }
    }

    /// Run `variant` on `p` once; if `check`, compare with the f64 reference and return the
    /// largest scaled error; if `time`, also return the average GPU time in microseconds
    /// of launches replayed from a CUDA graph.
    pub fn run(
        ctx: &CudaContext,
        ks: &mut Kernels<'_>,
        p: &Problem,
        variant: Variant,
        check: bool,
        time: bool,
    ) -> Result<(f64, Option<f64>), Box<dyn Error>> {
        let (k, f) = ks.get(p.dtype, variant, p.config)?;
        let host = p.args(variant);
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
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(k, f, p.launch(variant)).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let mut err = 0.0;
        if check {
            let read = |i: usize| -> Result<Vec<f32>, Box<dyn Error>> {
                Ok(match p.dtype {
                    Dtype::F16 => bufs[i].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
                    Dtype::BF16 => {
                        bufs[i].to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
                    }
                })
            };
            match variant {
                Variant::Rms => err = p.check("rmsnorm", &read(2)?, &p.reference_rms()),
                Variant::GemmaRms => err = p.check("gemma_rmsnorm", &read(2)?, &p.reference_gemma_rms()),
                Variant::LayerNorm => err = p.check("layernorm", &read(3)?, &p.reference_layernorm()),
                Variant::QkRms => err = p.check("qk_rmsnorm", &read(2)?, &p.reference_qk()),
                Variant::RmsQuant | Variant::FusedAddQuant => {
                    let at = if variant == Variant::RmsQuant { 2 } else { 3 };
                    let got: Vec<f32> = bufs[at].to_vec::<u8>()?.into_iter().map(|b| F8E4M3FN(b).to_f32()).collect();
                    if variant == Variant::RmsQuant {
                        err = p.check_fp8("rmsnorm_quant", &got, &p.reference_rms_quant());
                    } else {
                        let (out, res) = p.reference_fused_quant();
                        err = p.check_fp8("fused_add_rmsnorm_quant", &got, &out).max(p.check(
                            "residual",
                            &read(1)?,
                            &res,
                        ));
                    }
                }
                Variant::FusedAdd | Variant::GemmaFusedAdd => {
                    let (out, res) =
                        if variant == Variant::FusedAdd { p.reference_fused() } else { p.reference_gemma_fused() };
                    err = p.check("fused output", &read(0)?, &out).max(p.check("fused residual", &read(1)?, &res));
                }
            }
        }
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((err, t))
    }
}
