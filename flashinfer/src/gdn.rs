// SPDX-License-Identifier: Apache-2.0

//! Gated Delta Net (GDN) decode and prefill. Single-token decode, after FlashInfer's
//! `gated_delta_rule_decode_pretranspose` (`flashinfer/gdn_decode.py`, kernel
//! `flashinfer/gdn_kernels/gdn_decode_pretranspose.py`, both Apache-2.0) and its published
//! reference `decode_delta_rule` (`tests/gdn/reference_delta_rule.py`): the recurrent state
//! update of Qwen3-Next-style linear-attention layers. This is cute-rs's own code written
//! after that design; no FlashInfer source was translated.
//!
//! For each `(batch b, v-head hv)`, with the state `S` of shape `[V, K]` (v-major, K
//! contiguous: FlashInfer's "pretranspose" layout), `h = hv / (HV / H)` the shared q/k head
//! and Float32 arithmetic throughout:
//!
//! ```text
//! g     = exp(-exp(A_log[hv]) * softplus(a[b, hv] + dt_bias[hv]))     (softplus: beta 1, above 20 left as is)
//! beta  = sigmoid(b[b, hv])
//! q, k  = l2_normalize(q[b, h]), l2_normalize(k[b, h])                (if use_qk_l2norm; rsqrt(sum + 1e-6))
//! q    *= scale
//! S    *= g                                  (decay)
//! u     = (v - S k) * beta                   (delta rule, per row of S)
//! S    += u k^T                              (rank-one update, written back)
//! o     = S q
//! ```
//!
//! * **Geometry**: FlashInfer's small-batch kernel has 128 threads (4 warps), a warp owning a
//!   state row at a time with its 32 lanes holding `K / 32` consecutive elements each, `q` and
//!   `k` in registers and two butterfly reductions per row (`S k`, then `S q`). Here the lane
//!   keeps `VS = min(16 bytes / sizeof(state), K / 32)` consecutive elements per vector,
//!   `K / (32 VS)` vectors strided by 32 lanes. FlashInfer streams the state through shared
//!   memory with a two-stage `cp.async` pipeline, with 8 blocks per state when the batch is small;
//!   this kernel loads the rows straight into registers, `unroll` rows in flight per warp, and
//!   splits a state over `V / rows_per_block` blocks when `batch * HV` is small ([`Plan`]).
//! * **Target**: FlashInfer's GDN decode kernels are written for SM90+ (its tests skip below
//!   it); this one uses only plain loads, stores and warp shuffles and runs on SM80+.
//!
//! Dtypes: `q`, `k`, `v`, `a`, `b`, output Float16 or BFloat16 ([`Dtype`]); state Float32
//! (FlashInfer's legacy path) or BFloat16 / Float16 ([`StateDtype`]); `A_log`, `dt_bias`
//! Float32. Shapes: `q`, `k` `[batch, 1, H, K]`, `v`, output `[batch, 1, HV, V]`, `a`, `b`
//! `[batch, 1, HV]`, state `[slots, HV, V, K]` (`slots == batch` without indices), all
//! contiguous. `state_indices` (build-time [`Options::has_indices`]) gives each batch entry's
//! slot in the pool; an optional `output_state_indices` (`has_out_indices`) the slot written; a
//! negative index is padding: the state is untouched and the output is zero (FlashInfer's
//! float32-path semantics).
//!
//! **Multi-token decode** (`Options::tokens > 1`, FlashInfer's `gated_delta_rule_mtp`): the
//! recurrence runs over the tokens with each row's state in registers, so the state is read
//! once and written once whatever the token count. `q`, `k` are `[batch, T, H, K]`, `v` and the
//! output `[batch, T, HV, V]`, `a`, `b` `[batch, T, HV]`. `Options::cache_steps > 0` writes the
//! state after every token into the intermediate-state cache `[batch, cache_steps, HV, V, K]`
//! (FlashInfer's `intermediate_states_buffer`; per batch entry, not per pool slot; steps beyond
//! `T` are left alone), and `update_state = false` is `disable_state_update` (the pool is read
//! only). With a pool, a padded entry is skipped entirely.
//!
//! Not covered: the per-token pool scatter (`ssm_state_indices`), tree verification
//! (`verify_parents`), the ReplaySSM caches and commit, the non-transposed `[K, V]` state
//! layout, `K`/`V` other than multiples of 32 / 4, the bf16 `-1` to slot-0 redirect, strided
//! inputs and caches, offsets beyond 2^31 elements, the Rubin and `cake_gdn` backends, and
//! the SM90 TMA kernels.
//!
//! **Prefill** ([`prefill_kernel`], FlashInfer's `chunk_gated_delta_rule` in
//! `flashinfer/gdn_prefill.py`; its SM8x kernel is `gdn_kernels/delta_rule_dsl/delta_rule_sm80.py`
//! with `collective_inverse_hmma.py`, Apache-2.0 under the repository licence). Own code after
//! that design: chunks of 64 tokens, one block of 4 warps per `(sequence, v head, 64-row V
//! band)`, the `[V, K]` state band in Float32 registers (16 rows a warp, MMA accumulator
//! layout) and `mma.sync m16n8k16` throughout, so it runs on SM80+. Per chunk, with `G` the
//! cumulative log decay and `L = I + A`, `A[t][s] = beta_t exp(G_t - G_s) k_t.k_s` (s < t):
//!
//! ```text
//! R^T = (V^T - S K^T diag(exp G)) diag(beta)       Y^T = R^T L^-T    (Y = FlashInfer's new V)
//! O^T = scale (S Q^T diag(exp G) + Y^T M^T)         M[t][s] = exp(G_t - G_s) q_t.k_s (s <= t)
//! S   = exp(G_C) S + Y^T diag(exp(G_C - G)) K
//! ```
//!
//! Where FlashInfer inverts all of `L` (8x8 Gauss blocks merged by MMAs up to 64x64), this
//! kernel inverts only the four 16x16 diagonal blocks (a column a thread, Float32) and solves
//! `Y` block by block with MMAs against the off-diagonal blocks. `L`, its block inverses and
//! `M` are Float16 MMA operands, as FlashInfer's inverse; the state enters `S K^T` / `S Q^T` in
//! the input type. Inputs: `q`, `k` `[T, H, K]`, `v` and the output `[T, HV, V]` (f16/bf16),
//! `g` (the linear-domain decay alpha) and `beta` `[T, HV]` Float32, `cu_seqlens` Int32,
//! states `[N or slots, HV, V, K]` in f32/bf16/f16 (initial and final in separate buffers),
//! optional `state_indices` / `output_state_indices` pools (a negative read row starts from
//! zero; a negative write row and an empty sequence write nothing), optional in-kernel q/k L2
//! norm (rounded back to the input type), GVA. `K` is 128 (FlashInfer's SM80 kernel too) and
//! `V` a multiple of 64. Not ported: GQA (`H > HV`), context parallelism (`use_cp`), state
//! checkpoints, `prefix_lens`, log-domain and in-kernel gates (`gate_domain`, `A_log`,
//! `beta_is_logit`), FP8 states, int64 `cu_seqlens`, and the SM90+/cuDNN/Cake backends.
//!
//! `examples/gdn.rs` runs both on the simulator and, with `--features cuda` and `GPU=1`, on
//! the GPU against the reference, then benchmarks them (`ONLY=prefill`, `SKIP_CHECKS`).

use cute::prelude::*;

pub use crate::mamba::StateDtype;
pub use crate::norm::Dtype;

/// Threads per block (4 warps).
pub const THREADS: usize = 128;
/// Warps per block.
pub const WARPS: usize = 4;

/// Build-time options.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Options {
    /// L2-normalise `q` and `k` (FlashInfer's `use_qk_l2norm`).
    pub l2norm: bool,
    /// `state_indices` is given: the state is a pool and each batch entry names its slot.
    pub has_indices: bool,
    /// `output_state_indices` is given (with `has_indices`): the new state goes to another slot.
    pub has_out_indices: bool,
    /// Tokens per sequence: 1 is single-token decode, more are FlashInfer's `gated_delta_rule_mtp`
    /// (the recurrence runs over the tokens with the state in registers).
    pub tokens: usize,
    /// Write the final state back (`!disable_state_update`).
    pub update_state: bool,
    /// Steps per batch entry of the intermediate-state cache `[batch, cache_steps, HV, V, K]`
    /// (at least `tokens`), into which the state after every token is written; 0 for no cache.
    pub cache_steps: usize,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            l2norm: true,
            has_indices: false,
            has_out_indices: false,
            tokens: 1,
            update_state: true,
            cache_steps: 0,
        }
    }
}

/// How a launch is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Plan {
    /// State rows (of `V`) per block.
    pub rows_per_block: usize,
    /// Rows a warp handles at once (all their state loads first).
    pub unroll: usize,
}

impl Plan {
    /// Split a state over several blocks when `batch * hv` leaves the SMs short of blocks.
    pub fn new(batch: usize, hv: usize, v: usize, num_sms: usize) -> Plan {
        assert!(v.is_multiple_of(WARPS), "V = {v} must be a multiple of {WARPS}");
        let states = batch * hv;
        let mut rows_per_block = v;
        while rows_per_block > 8 && states * (v / rows_per_block) < 4 * num_sms && rows_per_block.is_multiple_of(8) {
            rows_per_block /= 2;
        }
        let per_warp = rows_per_block / WARPS;
        let unroll = [4, 2, 1].into_iter().find(|u| per_warp.is_multiple_of(*u)).unwrap();
        Plan { rows_per_block, unroll }
    }
}

fn c(v: f32) -> Float32 {
    Float32::from(v)
}

/// Parameter order of [`kernel`]: `state, q, k, v, a, b, A_log, dt_bias, out, state_indices,
/// output_state_indices, intermediate_states, num_q_heads, num_v_heads, scale`. Absent index tensors still take a
/// (one-element) buffer.
pub fn kernel(input: Dtype, state: StateDtype, opts: Options, kdim: usize, vdim: usize, plan: Plan) -> Kernel {
    match (input, state) {
        (Dtype::F16, StateDtype::F16) => kernel_of::<F16, F16>(input, state, opts, kdim, vdim, plan),
        (Dtype::F16, StateDtype::BF16) => kernel_of::<F16, BF16>(input, state, opts, kdim, vdim, plan),
        (Dtype::F16, StateDtype::F32) => kernel_of::<F16, f32>(input, state, opts, kdim, vdim, plan),
        (Dtype::BF16, StateDtype::F16) => kernel_of::<BF16, F16>(input, state, opts, kdim, vdim, plan),
        (Dtype::BF16, StateDtype::BF16) => kernel_of::<BF16, BF16>(input, state, opts, kdim, vdim, plan),
        (Dtype::BF16, StateDtype::F32) => kernel_of::<BF16, f32>(input, state, opts, kdim, vdim, plan),
    }
}

fn kernel_of<T: Numeric, S: Numeric>(
    input: Dtype,
    state_dtype: StateDtype,
    opts: Options,
    kdim: usize,
    vdim: usize,
    plan: Plan,
) -> Kernel {
    let vs = state_dtype.vec(kdim);
    assert!(vs >= 1 && kdim.is_multiple_of(32 * vs), "K = {kdim} must be a multiple of {}", 32 * vs);
    let nv = kdim / (32 * vs);
    let Plan { rows_per_block: rpb, unroll } = plan;
    assert!(rpb.is_multiple_of(WARPS) && vdim.is_multiple_of(rpb));
    let rpw = rpb / WARPS;
    assert!(rpw.is_multiple_of(unroll));
    assert!(!opts.has_out_indices || opts.has_indices);
    assert!(opts.tokens >= 1 && (opts.cache_steps == 0 || opts.cache_steps >= opts.tokens));
    let o = opts;
    let name = format!(
        "flashinfer_gdn_{}_{}_k{kdim}_v{vdim}_r{rpb}_u{unroll}_{}{}{}_t{}_s{}_u{}",
        input.name(),
        state_dtype.name(),
        o.l2norm as u8,
        o.has_indices as u8,
        o.has_out_indices as u8,
        o.tokens,
        o.cache_steps,
        o.update_state as u8,
    );
    Kernel::builder(&name).block_dim([THREADS as u32, 1, 1]).build(|| {
        let state = param_ptr::<S>("state");
        let q = param_ptr::<T>("q");
        let k = param_ptr::<T>("k");
        let v = param_ptr::<T>("v");
        let a = param_ptr::<T>("a");
        let b = param_ptr::<T>("b");
        let a_log = param_ptr::<f32>("A_log");
        let dt_bias = param_ptr::<f32>("dt_bias");
        let out = param_ptr::<T>("out");
        let indices = param_ptr::<i32>("state_indices");
        let out_indices = param_ptr::<i32>("output_state_indices");
        let cache = param_ptr::<S>("intermediate_states");
        let hq = param_scalar::<i32>("num_q_heads");
        let hv = param_scalar::<i32>("num_v_heads");
        let scale = param_scalar::<f32>("scale");

        let tid = thread_idx().x;
        let lane = tid & 31;
        let warp = tid >> 5;
        let batch = block_idx().x;
        let vhead = block_idx().y;
        let row_off = block_idx().z * rpb as i32;
        let qhead = vhead / (hv / hq);

        let slot_in = if o.has_indices { indices.load_at(batch) } else { batch };
        let valid = if o.has_indices { slot_in.ge(0) } else { Boolean::constant(true) };
        let load_slot = if o.has_indices { slot_in.max(0) } else { slot_in };
        let store_slot = if o.has_out_indices { out_indices.load_at(batch).max(0) } else { load_slot };
        let store_ok = if o.has_out_indices { valid & out_indices.load_at(batch).ge(0) } else { valid };

        // Per token: the gates (every lane computes the two scalars) and this lane's slice of q and
        // k, `nv` vectors of `vs` elements at columns `(lane + 32 j) * vs`.
        let nt = o.tokens;
        let tok = |t: usize| batch * nt as i32 + t as i32;
        let mut decays = Vec::with_capacity(nt);
        let mut betas = Vec::with_capacity(nt);
        let mut qrs: Vec<Vec<Float32>> = Vec::with_capacity(nt);
        let mut krs: Vec<Vec<Float32>> = Vec::with_capacity(nt);
        for t in 0..nt {
            let ab = tok(t) * hv + vhead;
            let x = a.load_at(ab).to::<f32>() + dt_bias.load_at(vhead);
            // softplus(x) = log(1 + exp(x)), x itself above 20.
            let e = cute::math::exp2(x * c(std::f32::consts::LOG2_E), true);
            let sp = select(x.le(20.0f32), cute::math::log2(e + c(1.0), true) * c(std::f32::consts::LN_2), x);
            decays.push(cute::math::exp2(
                -(cute::math::exp2(a_log.load_at(vhead) * c(std::f32::consts::LOG2_E), true))
                    * sp
                    * c(std::f32::consts::LOG2_E),
                true,
            ));
            let bx = b.load_at(ab).to::<f32>();
            betas.push(cute::math::div(
                c(1.0),
                c(1.0) + cute::math::exp2(bx * c(-std::f32::consts::LOG2_E), true),
                true,
            ));
            let qk_base = (tok(t) * hq + qhead) * kdim as i32;
            let load_k = |p: &Ptr<T>| -> Vec<Float32> {
                (0..nv).flat_map(|j| load_n::<T>(&p.offset(qk_base + (lane + 32 * j as i32) * vs as i32), vs)).collect()
            };
            let mut qr = load_k(&q);
            let mut kr = load_k(&k);
            if o.l2norm {
                let sq = qr.iter().fold(c(0.0), |s, x| s + *x * *x);
                let sk = kr.iter().fold(c(0.0), |s, x| s + *x * *x);
                let sq = warp_reduction_sum(sq, 32);
                let sk = warp_reduction_sum(sk, 32);
                let iq = (sq + c(1e-6)).rsqrt();
                let ik = (sk + c(1e-6)).rsqrt();
                qr = qr.iter().map(|x| *x * iq).collect();
                kr = kr.iter().map(|x| *x * ik).collect();
            }
            qrs.push(qr.iter().map(|x| *x * scale).collect());
            krs.push(kr);
        }

        let sv = (kdim / vs) as i32;
        let state_in = (load_slot * hv + vhead) * vdim as i32 + row_off;
        let state_out = (store_slot * hv + vhead) * vdim as i32 + row_off;
        let vo = |t: usize| (tok(t) * hv + vhead) * vdim as i32 + row_off;
        let cache_base = |t: usize| ((batch * o.cache_steps as i32 + t as i32) * hv + vhead) * vdim as i32 + row_off;
        let first_row = warp * rpw as i32;
        let at = |base: Int32, r: &Int32, j: usize| ((base + *r) * sv + lane + 32 * j as i32) * vs as i32;
        let run = |r0: Int32| {
            let rows: Vec<Int32> = (0..unroll).map(|i| r0 + i as i32).collect();
            let vvals: Vec<Vec<Float32>> =
                (0..nt).map(|t| rows.iter().map(|r| v.load_at(vo(t) + *r).to::<f32>()).collect()).collect();
            // All state loads first.
            let mut cur: Vec<Vec<Float32>> = rows
                .iter()
                .map(|r| (0..nv).flat_map(|j| load_n::<S>(&state.offset(at(state_in, r, j)), vs)).collect())
                .collect();
            for t in 0..nt {
                // Per row: `S k` (all lanes need it), the delta `u` and the new row.
                let mut partial: Vec<Float32> = Vec::with_capacity(unroll);
                for (i, row) in cur.iter_mut().enumerate() {
                    let hs: Vec<Float32> = row.iter().map(|s| *s * decays[t]).collect();
                    let mut hk = c(0.0);
                    for (h, kk) in hs.iter().zip(&krs[t]) {
                        hk = hk + *h * *kk;
                    }
                    let hk = warp_reduction_sum(hk, 32);
                    let u = (vvals[t][i] - hk) * betas[t];
                    let mut oq = c(0.0);
                    for ((n, h), (kk, qq)) in row.iter_mut().zip(&hs).zip(krs[t].iter().zip(&qrs[t])) {
                        *n = *h + *kk * u;
                        oq = oq + *n * *qq;
                    }
                    partial.push(oq);
                }
                // The outputs `S q` of the rows together: a reduce-scatter leaves row
                // `lane >> (5 - m)`'s sum in lanes whose low `5 - m` bits are zero, with fewer
                // shuffles than one butterfly each.
                let (sums, idx_shift) = reduce_scatter(partial, lane);
                let res = select(valid, sums, 0.0f32);
                if_((lane & ((1 << idx_shift) - 1)).eq(0), || {
                    out.store_at(vo(t) + rows[0] + (lane >> idx_shift as i32), res.to::<T>())
                });
                if o.cache_steps > 0 {
                    for (r, row) in rows.iter().zip(&cur) {
                        for j in 0..nv {
                            let ptr = cache.offset(at(cache_base(t), r, j));
                            let vals = &row[j * vs..(j + 1) * vs];
                            if o.has_indices {
                                if_(valid, || store_n::<S>(&ptr, vals));
                            } else {
                                store_n::<S>(&ptr, vals);
                            }
                        }
                    }
                }
            }
            if o.update_state {
                for (r, row) in rows.iter().zip(&cur) {
                    for j in 0..nv {
                        let ptr = state.offset(at(state_out, r, j));
                        let vals = &row[j * vs..(j + 1) * vs];
                        if o.has_indices {
                            if_(store_ok, || store_n::<S>(&ptr, vals));
                        } else {
                            store_n::<S>(&ptr, vals);
                        }
                    }
                }
            }
        };
        if rpw == unroll {
            run(first_row);
        } else {
            for_range(0, rpw as i32 / unroll as i32, 1, |i| run(first_row + i * unroll as i32));
        }
    })
}

/// Sum each of `vals` (a power of two of them, at most 32) over the 32 lanes of a warp, leaving
/// value `lane >> shift` in the lanes whose low `shift` bits are zero; returns the lane's sum
/// and `shift` (`5 - log2(vals.len())`). A single value is a plain butterfly.
fn reduce_scatter(mut vals: Vec<Float32>, lane: Int32) -> (Float32, u32) {
    assert!(vals.len().is_power_of_two() && vals.len() <= 32);
    let mut offset = 16;
    while vals.len() > 1 {
        let half = vals.len() / 2;
        let upper = (lane & offset).ne(0);
        vals = (0..half)
            .map(|i| {
                let keep = select(upper, vals[i + half], vals[i]);
                let send = select(upper, vals[i], vals[i + half]);
                keep + shuffle_sync_bfly(send, offset)
            })
            .collect();
        offset /= 2;
    }
    let mut sum = vals[0];
    let mut shift = 0;
    while offset > 0 {
        sum = sum + shuffle_sync_bfly(sum, offset);
        offset /= 2;
        shift += 1;
    }
    (sum, shift)
}

/// Load `n` contiguous elements at `ptr` (offset a multiple of `n`) as Float32.
fn load_n<T: Numeric>(ptr: &Ptr<T>, n: usize) -> Vec<Float32> {
    fn go<T: Numeric, const N: usize>(ptr: &Ptr<T>) -> Vec<Float32> {
        load_vector::<T, N>(ptr, LoadOptions::default()).iter().map(|v| v.to::<f32>()).collect()
    }
    match n {
        1 => go::<T, 1>(ptr),
        2 => go::<T, 2>(ptr),
        4 => go::<T, 4>(ptr),
        8 => go::<T, 8>(ptr),
        _ => panic!("vector of {n}"),
    }
}

/// Store `vals`, converted to `T`, as one vector access.
fn store_n<T: Numeric>(ptr: &Ptr<T>, vals: &[Float32]) {
    fn go<T: Numeric, const N: usize>(ptr: &Ptr<T>, vals: &[Float32]) {
        let v: [Value<T>; N] = std::array::from_fn(|i| vals[i].to::<T>());
        store_vector::<T, N>(ptr, v, StoreOptions::default());
    }
    match vals.len() {
        1 => go::<T, 1>(ptr, vals),
        2 => go::<T, 2>(ptr, vals),
        4 => go::<T, 4>(ptr, vals),
        8 => go::<T, 8>(ptr, vals),
        n => panic!("vector of {n}"),
    }
}

/// A test problem; every input value is exactly representable in its dtype.
pub struct Problem {
    pub input: Dtype,
    pub state: StateDtype,
    pub opts: Options,
    pub batch: usize,
    /// q and k heads.
    pub hq: usize,
    /// v heads (and state heads).
    pub hv: usize,
    pub k: usize,
    pub v: usize,
    pub scale: f32,
    /// Slots in the state pool (`batch` unless there are indices).
    pub slots: usize,
    /// The slot read for each batch entry (with `opts.has_indices`); negative is padding.
    pub indices: Vec<i32>,
    /// The slot written for each batch entry (with `opts.has_out_indices`).
    pub out_indices: Vec<i32>,
    pub state0: Vec<f32>,
    pub q: Vec<f32>,
    pub kk: Vec<f32>,
    pub vv: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// The SM count the launch plan assumes.
    pub num_sms: usize,
    /// Override the launch plan's `(rows_per_block, unroll)` (for tuning).
    pub plan_override: Option<(usize, usize)>,
}

impl Problem {
    /// `(batch, hq, hv, k, v)`; `scale` is `1 / sqrt(k)`.
    pub fn new(
        input: Dtype,
        state: StateDtype,
        opts: Options,
        (batch, hq, hv, k, v): (usize, usize, usize, usize, usize),
        seed: u64,
    ) -> Problem {
        assert!(hv.is_multiple_of(hq));
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut uni = move |lo: f32, hi: f32| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((rng >> 33) as f32 / (1u64 << 31) as f32)
        };
        let reads = if opts.has_indices { batch + 3 } else { batch };
        // Written slots live above the read ones, so no block reads what another writes.
        let slots = if opts.has_out_indices { 2 * reads } else { reads };
        // A shuffled choice of distinct slots, with every fifth one padded.
        let indices: Vec<i32> = if opts.has_indices {
            (0..batch).map(|i| if i % 5 == 3 { -1 } else { ((i * 7 + 2) % reads) as i32 }).collect()
        } else {
            vec![]
        };
        // Written slots: the read slot moved above the read range; padding stays padding.
        let out_indices: Vec<i32> = if opts.has_out_indices {
            indices.iter().map(|&i| if i < 0 { -1 } else { i + reads as i32 }).collect()
        } else {
            vec![]
        };
        let mut mk =
            |n: usize, lo: f32, hi: f32, r: &dyn Fn(f32) -> f32| (0..n).map(|_| r(uni(lo, hi))).collect::<Vec<f32>>();
        let (ri, rs) = (|x| input.round(x), |x| state.round(x));
        let state0 = mk(slots * hv * v * k, -1.0, 1.0, &rs);
        let nt = opts.tokens;
        let q = mk(batch * nt * hq * k, -1.0, 1.0, &ri);
        let kk = mk(batch * nt * hq * k, -1.0, 1.0, &ri);
        let vv = mk(batch * nt * hv * v, -1.0, 1.0, &ri);
        let a = mk(batch * nt * hv, -0.5, 0.5, &ri);
        let b = mk(batch * nt * hv, -3.0, 3.0, &ri);
        let a_log = mk(hv, -0.3, 0.3, &|x| x);
        let dt_bias = mk(hv, -0.3, 0.3, &|x| x);
        Problem {
            input,
            state,
            opts,
            batch,
            hq,
            hv,
            k,
            v,
            scale: 1.0 / (k as f32).sqrt(),
            slots,
            indices,
            out_indices,
            state0,
            q,
            kk,
            vv,
            a,
            b,
            a_log,
            dt_bias,
            num_sms: 22,
            plan_override: None,
        }
    }

    pub fn plan(&self) -> Plan {
        if let Some((rows_per_block, unroll)) = self.plan_override {
            return Plan { rows_per_block, unroll };
        }
        Plan::new(self.batch, self.hv, self.v, self.num_sms)
    }

    pub fn kernel(&self) -> Kernel {
        kernel(self.input, self.state, self.opts, self.k, self.v, self.plan())
    }

    pub fn launch(&self) -> LaunchConfig {
        let p = self.plan();
        LaunchConfig::new(
            [self.batch as u32, self.hv as u32, (self.v / p.rows_per_block) as u32],
            [THREADS as u32, 1, 1],
        )
    }

    /// Elements of the intermediate-state cache.
    pub fn cache_len(&self) -> usize {
        self.batch * self.opts.cache_steps * self.hv * self.v * self.k
    }

    /// The kernel's arguments (see [`kernel`]); the state is argument 0, the output 8, the
    /// intermediate-state cache 11.
    pub fn args(&self) -> Vec<KernelArg> {
        assert!(self.slots * self.hv * self.v * self.k < 1 << 31, "offsets are 32-bit");
        let idx = |on: bool, v: &Vec<i32>| KernelArg::i32(if on { v.clone() } else { vec![0] });
        vec![
            self.state.buffer(&self.state0),
            self.input.buffer(&self.q),
            self.input.buffer(&self.kk),
            self.input.buffer(&self.vv),
            self.input.buffer(&self.a),
            self.input.buffer(&self.b),
            KernelArg::f32(self.a_log.clone()),
            KernelArg::f32(self.dt_bias.clone()),
            self.input.buffer(&vec![0.0; self.batch * self.opts.tokens * self.hv * self.v]),
            idx(self.opts.has_indices, &self.indices),
            idx(self.opts.has_out_indices, &self.out_indices),
            self.state.buffer(&vec![0.0; self.cache_len().max(1)]),
            KernelArg::ScalarI32(self.hq as i32),
            KernelArg::ScalarI32(self.hv as i32),
            KernelArg::ScalarF32(self.scale),
        ]
    }

    /// The new state pool, the output and the intermediate-state cache in f64:
    /// `(state, out, cache)`; FlashInfer's `decode_delta_rule` applied token by token, with the
    /// state in `[V, K]` order.
    pub fn reference(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (hq, hv, kd, vd, nt) = (self.hq, self.hv, self.k, self.v, self.opts.tokens);
        let steps = self.opts.cache_steps;
        let mut state: Vec<f64> = self.state0.iter().map(|&x| x as f64).collect();
        let mut out = vec![0.0f64; self.batch * nt * hv * vd];
        let mut cache = vec![0.0f64; self.cache_len()];
        for b in 0..self.batch {
            let slot = if self.opts.has_indices { self.indices[b] } else { b as i32 };
            let out_slot = if self.opts.has_out_indices { self.out_indices[b] } else { slot };
            if slot < 0 {
                continue;
            }
            let write = out_slot >= 0 && self.opts.update_state;
            for h in 0..hv {
                let g = h / (hv / hq);
                for r in 0..vd {
                    let base = |sl: i32| ((sl as usize * hv + h) * vd + r) * kd;
                    let mut cur: Vec<f64> = (0..kd).map(|i| state[base(slot) + i]).collect();
                    for t in 0..nt {
                        let tok = b * nt + t;
                        let x = self.a[tok * hv + h] as f64 + self.dt_bias[h] as f64;
                        let sp = if x <= 20.0 { x.exp().ln_1p() } else { x };
                        let decay = (-(self.a_log[h] as f64).exp() * sp).exp();
                        let beta = 1.0 / (1.0 + (-(self.b[tok * hv + h] as f64)).exp());
                        let norm = |src: &[f32]| -> Vec<f64> {
                            let s = &src[(tok * hq + g) * kd..(tok * hq + g + 1) * kd];
                            let n = if self.opts.l2norm {
                                s.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt().max(1e-12)
                            } else {
                                1.0
                            };
                            s.iter().map(|x| *x as f64 / n).collect()
                        };
                        let qv: Vec<f64> = norm(&self.q).iter().map(|x| x * self.scale as f64).collect();
                        let kv = norm(&self.kk);
                        let hrow: Vec<f64> = cur.iter().map(|s| s * decay).collect();
                        let hk: f64 = hrow.iter().zip(&kv).map(|(a, b)| a * b).sum();
                        let u = (self.vv[(tok * hv + h) * vd + r] as f64 - hk) * beta;
                        cur = hrow.iter().zip(&kv).map(|(h, k)| h + k * u).collect();
                        out[(tok * hv + h) * vd + r] = cur.iter().zip(&qv).map(|(a, b)| a * b).sum();
                        if steps > 0 {
                            let at = (((b * steps + t) * hv + h) * vd + r) * kd;
                            cache[at..at + kd].copy_from_slice(&cur);
                        }
                    }
                    if write {
                        let at = base(out_slot);
                        state[at..at + kd].copy_from_slice(&cur);
                    }
                }
            }
        }
        (state, out, cache)
    }

    /// Check the new state, output and cache against the reference (absolute plus relative);
    /// returns the largest scaled `(state, out)` errors (the cache counts as state).
    pub fn check(&self, state: &[f32], out: &[f32], cache: &[f32]) -> (f64, f64) {
        let (rs, ro, rc) = self.reference();
        let scaled = |got: &[f32], want: &[f64], tol: f64, what: &str| {
            let mut worst = 0.0f64;
            for (i, (&g, &r)) in got.iter().zip(want).enumerate() {
                let e = (g as f64 - r).abs() / (1.0 + r.abs());
                assert!(e <= tol, "{what} {i}: {g} against {r} ({:?})", self.opts);
                worst = worst.max(e);
            }
            worst
        };
        let ts = self.state.tolerance() + 5e-5;
        let to = self.input.tolerance() + 5e-4 + self.state.tolerance();
        let cache_err = scaled(&cache[..rc.len()], &rc, ts, "cache");
        (scaled(state, &rs, ts, "state").max(cache_err), scaled(out, &ro, to, "out"))
    }

    /// Bytes moved: the state read (and written with `update_state`), the cache written, `q`,
    /// `k`, `v`, `out`, `a`, `b`.
    pub fn bytes(&self) -> f64 {
        let o = &self.opts;
        let one = self.batch * self.hv * self.v * self.k * self.state.bytes();
        let state = one * (1 + o.update_state as usize + if o.cache_steps > 0 { o.tokens } else { 0 });
        let io = self.batch * o.tokens * (2 * self.hq * self.k + 2 * self.hv * self.v + 2 * self.hv) * 2;
        (state + io) as f64
    }
}

/// Simulate the kernel on `p` and check it against the f64 reference; returns the largest
/// scaled `(state, out)` errors (the intermediate-state cache counts as state).
pub fn simulate(p: &Problem) -> (f64, f64) {
    let k = p.kernel();
    let mut args = p.args();
    k.simulate(p.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.check(&p.state.read(&args[0]), &p.input.read(&args[8]), &p.state.read(&args[11]))
}

// ---------------------------------------------------------------------------------------------
// Prefill: `chunk_gated_delta_rule`.
// ---------------------------------------------------------------------------------------------

/// Tokens per chunk of the prefill kernel.
pub const CHUNK: usize = 64;
/// State rows (of `V`) per prefill block: four warps of 16 MMA rows.
pub const BLK_V: usize = 64;
/// The prefill kernel's head size `K` (FlashInfer's SM80 kernel has `D = 128` only).
pub const PREFILL_K: usize = 128;

/// Build-time options of the prefill kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefillOptions {
    /// L2-normalise `q` and `k` (`use_qk_l2norm_in_kernel`; rounded back to the input type).
    pub l2norm: bool,
    /// `initial_state` is given (otherwise every sequence starts from zero).
    pub has_initial_state: bool,
    /// Write the final state (`output_final_state`).
    pub output_final_state: bool,
    /// `state_indices`: both states are pools and sequence `i` uses row `state_indices[i]`
    /// (negative: start from zero, write nothing).
    pub has_indices: bool,
    /// `output_state_indices` (with `has_indices`): the final state goes to that row instead.
    pub has_out_indices: bool,
}

impl Default for PrefillOptions {
    fn default() -> PrefillOptions {
        PrefillOptions {
            l2norm: true,
            has_initial_state: true,
            output_final_state: true,
            has_indices: false,
            has_out_indices: false,
        }
    }
}

// Shared-memory map. Input-type buffer (elements): K (two stages), Q, the block's V band
// and the output staging tile, all `[token][dim]` with 16-byte vectors XOR-swizzled by
// `row % 8`.
const S_K: i32 = 0;
const S_Q: i32 = 2 * 8192;
const S_V: i32 = 3 * 8192;
const S_O: i32 = S_V + 4096;
const T_ELEMS: usize = 8192 * 3 + 4096 * 2;
// Float16 buffer: the gated `-A` (off-diagonal blocks; the diagonal blocks hold the
// inverted 16x16 blocks) and the gated, causal `QK`, both `[t][s]`.
const S_L: i32 = 0;
const S_M: i32 = 4096;
const H_ELEMS: usize = 8192;
// Float32 buffer: the cumulative log2 gates and beta (two stages each) and the strictly
// lower diagonal 16x16 blocks of `A` (rows padded to keep the four blocks on other banks).
const F_LG: i32 = 0;
const F_BETA: i32 = 128;
const F_ADIAG: i32 = 256;
const ADIAG_STRIDE: i32 = 260;
const F_ELEMS: usize = 256 + 4 * 260;

/// Parameter order of [`prefill_kernel`]: `q, k, v, g, beta, cu_seqlens, initial_state,
/// final_state, out, state_indices, output_state_indices, num_q_heads, num_v_heads, scale`.
/// Absent tensors still take a (one-element) buffer. Grid `(V / 64, HV, num_seqs)`, 128
/// threads.
pub fn prefill_kernel(input: Dtype, state: StateDtype, opts: PrefillOptions, vdim: usize) -> Kernel {
    match (input, state) {
        (Dtype::F16, StateDtype::F16) => prefill_kernel_of::<F16, F16>(input, state, opts, vdim),
        (Dtype::F16, StateDtype::BF16) => prefill_kernel_of::<F16, BF16>(input, state, opts, vdim),
        (Dtype::F16, StateDtype::F32) => prefill_kernel_of::<F16, f32>(input, state, opts, vdim),
        (Dtype::BF16, StateDtype::F16) => prefill_kernel_of::<BF16, F16>(input, state, opts, vdim),
        (Dtype::BF16, StateDtype::BF16) => prefill_kernel_of::<BF16, BF16>(input, state, opts, vdim),
        (Dtype::BF16, StateDtype::F32) => prefill_kernel_of::<BF16, f32>(input, state, opts, vdim),
    }
}

fn b2i(b: Boolean) -> Int32 {
    select(b, 1i32, 0i32)
}

/// Vector index of column vector `c` of row `row` in a swizzled tile of `vpr` vectors a row
/// starting at vector `base`.
fn sw(base: Int32, vpr: i32, row: Int32, c: Int32) -> Int32 {
    base + row * vpr + (c ^ (row & 7))
}

fn prefill_kernel_of<T: Numeric, S: Numeric>(
    input: Dtype,
    state_dtype: StateDtype,
    o: PrefillOptions,
    vdim: usize,
) -> Kernel {
    assert!(vdim.is_multiple_of(BLK_V), "V = {vdim} must be a multiple of {BLK_V}");
    assert!(!o.has_out_indices || o.has_indices);
    let kd = PREFILL_K as i32;
    let name = format!(
        "flashinfer_gdn_prefill_{}_{}_v{vdim}_{}{}{}{}{}",
        input.name(),
        state_dtype.name(),
        o.l2norm as u8,
        o.has_initial_state as u8,
        o.output_final_state as u8,
        o.has_indices as u8,
        o.has_out_indices as u8,
    );
    let ab = match input {
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
    };
    Kernel::builder(&name).block_dim([THREADS as u32, 1, 1]).build(|| {
        const BOUND: usize = 1 << 30;
        let q = param_ptr::<T>("q");
        let k = param_ptr::<T>("k");
        let v = param_ptr::<T>("v");
        let g = param_ptr::<f32>("g");
        let beta = param_ptr::<f32>("beta");
        let cu = param_ptr::<i32>("cu_seqlens");
        let state_in = param_ptr::<S>("initial_state");
        let state_out = param_ptr::<S>("final_state");
        let out = param_ptr::<T>("out");
        let indices = param_ptr::<i32>("state_indices");
        let out_indices = param_ptr::<i32>("output_state_indices");
        let hq = param_scalar::<i32>("num_q_heads");
        let hv = param_scalar::<i32>("num_v_heads");
        let scale = param_scalar::<f32>("scale");

        let tid = thread_idx().x;
        let lane = tid & 31;
        let warp = tid >> 5;
        let band = block_idx().x;
        let vhead = block_idx().y;
        let seq = block_idx().z;
        let qhead = vhead / (hv / hq);
        let bos = cu.load_at(seq);
        let len = cu.load_at(seq + 1) - bos;
        let zero = c(0.0);

        let st = make_smem_tensor_aligned::<T>(Layout::col_major(T_ELEMS), 16);
        let sh = make_smem_tensor_aligned::<F16>(Layout::col_major(H_ELEMS), 16);
        let sf = make_smem_tensor_aligned::<f32>(Layout::col_major(F_ELEMS), 16);
        let tv = make_tensor(&st.ptr(), Layout::col_major((8, T_ELEMS / 8)));
        let hvv = make_tensor(&sh.ptr(), Layout::col_major((8, H_ELEMS / 8)));
        let gvec = |p: &Ptr<T>| make_tensor(p, Layout::col_major((8, BOUND / 8)));
        let (qg, kg, vg) = (gvec(&q), gvec(&k), gvec(&v));
        let cp16 = make_copy_atom::<T>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let ldsm = make_copy_atom::<T>(CopyOp::LdMatrix { num: 4, trans: false });
        let ldsm_t = make_copy_atom::<T>(CopyOp::LdMatrix { num: 4, trans: true });
        let ldsm_h = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false });
        let mma = make_tiled_mma(MmaF16BF16Op::new(ab, DType::F32, (16, 8, 16)), Layout::col_major((1, 1, 1)));
        let mma_h =
            make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)), Layout::col_major((1, 1, 1)));

        let load_vec = |src: &Tensor<T>, gv: Int32, sv: Int32, pred: Boolean| {
            let pf = make_fragment::<bool>(1);
            pf.store(0, pred);
            copy_if(&cp16, &pf, &src.slice((.., gv)), &tv.slice((.., sv)));
        };
        // `ldmatrix.x4` at vector `sv` of the input-type (`ld_t`) or Float16 (`ld_h`) buffer.
        let ld_t = |trans: bool, sv: Int32| -> Vec<Value<T>> {
            let frag = make_fragment::<T>(8);
            copy(if trans { &ldsm_t } else { &ldsm }, &tv.slice((.., sv)), &frag);
            frag.values()
        };
        let ld_h = |sv: Int32| -> Vec<Value<F16>> {
            let frag = make_fragment::<F16>(8);
            copy(&ldsm_h, &hvv.slice((.., sv)), &frag);
            frag.values()
        };
        // One m16n8k16 MMA: `a` (8 values), `b` (4) and `c` (4) in PTX register order.
        let mma_t = |a: &[Value<T>], b: &[Value<T>], cc: &[Float32]| -> Vec<Float32> {
            let (fa, fb, fc) =
                (make_fragment::<T>((8, 1, 1)), make_fragment::<T>((4, 1, 1)), make_fragment::<f32>((4, 1, 1)));
            fa.store_values(a);
            fb.store_values(b);
            fc.store_values(cc);
            gemm(&mma, &fc, &fa, &fb, &fc);
            fc.values()
        };
        let mma_f16 = |a: &[Value<F16>], b: &[Value<F16>], cc: &[Float32]| -> Vec<Float32> {
            let (fa, fb, fc) =
                (make_fragment::<F16>((8, 1, 1)), make_fragment::<F16>((4, 1, 1)), make_fragment::<f32>((4, 1, 1)));
            fa.store_values(a);
            fb.store_values(b);
            fc.store_values(cc);
            gemm(&mma_h, &fc, &fa, &fb, &fc);
            fc.values()
        };
        // 16x16 = two m16n8 tiles: `acc[0..4]` and `acc[4..8]` against the two halves of `b`.
        let mma2_t = |a: &[Value<T>], b: &[Value<T>], acc: &mut [Float32]| {
            let lo = mma_t(a, &b[0..4], &acc[0..4]);
            let hi = mma_t(a, &b[4..8], &acc[4..8]);
            acc[0..4].copy_from_slice(&lo);
            acc[4..8].copy_from_slice(&hi);
        };
        let mma2_h = |a: &[Value<F16>], b: &[Value<F16>], acc: &mut [Float32]| {
            let lo = mma_f16(a, &b[0..4], &acc[0..4]);
            let hi = mma_f16(a, &b[4..8], &acc[4..8]);
            acc[0..4].copy_from_slice(&lo);
            acc[4..8].copy_from_slice(&hi);
        };
        let to_t = |x: &[Float32]| -> Vec<Value<T>> { x.iter().map(|v| v.to::<T>()).collect() };
        let to_h = |x: &[Float32]| -> Vec<Value<F16>> { x.iter().map(|v| v.to::<F16>()).collect() };
        let exp2 = |x: Float32| cute::math::exp2(x, true);

        // Lane patterns of `ldmatrix.x4`: P1 (rows `lane % 16`, vector `lane / 16`) reads an
        // A operand from `[m][k]` or, transposed, a B operand from `[k][n]`; P2 (rows
        // `lane % 8 + 8 (lane / 16)`, vector `(lane / 8) % 2`) a B operand from `[n][k]` or,
        // transposed, an accumulator-layout tile from `[n][m]`.
        // Accumulator coordinates: row `gr + 8 a`, column `2 q4 + b` of each m16n8 tile.
        let (gr, q4) = (lane >> 2, lane & 3);

        // The state slots.
        let slot_in = if o.has_indices { indices.load_at(seq) } else { seq };
        let read_ok = if o.has_indices { slot_in.ge(0) } else { Boolean::constant(true) };
        let slot_out = if o.has_out_indices { out_indices.load_at(seq) } else { slot_in };
        let write_ok = if o.has_indices { slot_out.ge(0) } else { Boolean::constant(true) };

        if_(len.gt(0), || {
            let nchunks = (len + (CHUNK as i32 - 1)) / CHUNK as i32;
            let row0 = band * BLK_V as i32 + warp * 16 + gr;
            // This thread's state elements, accumulator layout: tile `n` (columns `8 n ..`),
            // value `2 a + b` at row `row0 + 8 a`, column `8 n + 2 q4 + b`.
            let state_at = |slot: Int32, n: usize, a: usize| {
                ((slot * hv + vhead) * vdim as i32 + row0 + 8 * a as i32) * kd + 8 * n as i32 + 2 * q4
            };
            let mut s0: Vec<Float32> = Vec::with_capacity(64);
            if o.has_initial_state {
                let slot = slot_in.max(0);
                for n in 0..16 {
                    for a in 0..2 {
                        for x in load_n::<S>(&state_in.offset(state_at(slot, n, a)), 2) {
                            s0.push(select(read_ok, x, 0.0f32));
                        }
                    }
                }
            } else {
                s0 = vec![zero; 64];
            }
            let s_state = Var::new(s0);

            // Global -> shared for chunk `cc`: Q, K (stage `cc % 2`) and the V band, rows
            // past the sequence zero-filled.
            // `with_k`: K (to stage `cc % 2`); `with_qv`: Q and the V band.
            let issue_loads = |cc: Int32, tid: Int32, with_k: bool, with_qv: bool| {
                let kbase = S_K / 8 + (cc & 1) * 1024;
                let t0 = cc * CHUNK as i32;
                for it in 0..8 {
                    let row = (tid >> 4) + 8 * it;
                    let col = tid & 15;
                    let ok = (t0 + row).lt(len);
                    let tok = bos + select(ok, t0 + row, t0);
                    let gq = (tok * hq + qhead) * (kd / 8) + col;
                    if with_qv {
                        load_vec(&qg, gq, sw(Int32::from(S_Q / 8), 16, row, col), ok);
                    }
                    if with_k {
                        load_vec(&kg, gq, sw(kbase, 16, row, col), ok);
                    }
                }
                for it in 0..if with_qv { 4 } else { 0 } {
                    let row = (tid >> 3) + 16 * it;
                    let col = tid & 7;
                    let ok = (t0 + row).lt(len);
                    let tok = bos + select(ok, t0 + row, t0);
                    let gv = (tok * hv + vhead) * (vdim as i32 / 8) + band * (BLK_V as i32 / 8) + col;
                    load_vec(&vg, gv, sw(Int32::from(S_V / 8), 8, row, col), ok);
                }
                cp_async_commit_group();
            };
            // The gates of chunk `cc` for tokens `2 lane`, `2 lane + 1`: `[alpha0, beta0,
            // alpha1, beta1]`; alpha 1 and beta 0 past the sequence.
            let gates = |cc: Int32, lane: Int32| -> Vec<Float32> {
                (0..2)
                    .flat_map(|i| {
                        let r = cc * CHUNK as i32 + 2 * lane + i;
                        let ok = r.lt(len);
                        let at = (bos + r.min(len - 1)) * hv + vhead;
                        [select(ok, g.load_at(at), 1.0f32), select(ok, beta.load_at(at), 0.0f32)]
                    })
                    .collect()
            };
            issue_loads(Int32::from(0), tid, true, true);
            let gate_var = Var::new(gates(Int32::from(0), lane));
            // The thread index, re-read from shared memory every chunk: derived from
            // `thread_idx()`, LLVM hoists every lane-dependent shared address out of the
            // chunk loop, where they stay live and spill.
            let sidx = make_smem_tensor::<i32>(Layout::col_major(THREADS));
            sidx.store(tid, tid);
            let tid0 = tid;

            for_range(0, nchunks, 1, |cc| {
                let tid = sidx.load(tid0);
                let lane = tid & 31;
                let warp = tid >> 5;
                let (p1r, p1v) = (lane & 15, lane >> 4);
                let (p2r, p2v) = ((lane & 7) + ((lane >> 4) << 3), (lane >> 3) & 1);
                let (gr, q4) = (lane >> 2, lane & 3);
                let stage = cc & 1;
                let kbase = S_K / 8 + stage * 1024;
                let lgb = F_LG + stage * CHUNK as i32;
                let btb = F_BETA + stage * CHUNK as i32;

                // ---- Gates: inclusive scan of log2(alpha) over the chunk (each warp; warp
                // 0 publishes it).
                let gv = gate_var.get();
                let l0 = cute::math::log2(gv[0].max(1e-30f32), true);
                let l1 = cute::math::log2(gv[2].max(1e-30f32), true);
                let pair = l0 + l1;
                let mut inc = pair;
                for off in [1, 2, 4, 8, 16] {
                    let up = shuffle_sync_up(inc, off);
                    inc = inc + select(lane.ge(off), up, 0.0f32);
                }
                if_(warp.eq(0), || {
                    sf.store(lgb + 2 * lane, inc - l1);
                    sf.store(lgb + 2 * lane + 1, inc);
                    sf.store(btb + 2 * lane, gv[1]);
                    sf.store(btb + 2 * lane + 1, gv[3]);
                });
                cp_async_wait_group(0);
                sync_threads();
                // Next chunk's K to the other stage (last read by the previous chunk).
                if_((cc + 1).lt(nchunks), || issue_loads(cc + 1, tid, true, false));

                // ---- Optional L2 norm of the Q and K rows, in place (rounded to the input
                // type, as FlashInfer): half a warp per row.
                if o.l2norm {
                    for it in 0..16 {
                        let row = warp * 32 + 2 * it + (lane >> 4);
                        let col = lane & 15;
                        let is_k = row.ge(64);
                        let base = select(is_k, kbase, S_Q / 8);
                        let at = sw(base, 16, row & 63, col) * 8;
                        let x: Vec<Float32> = load_vector::<T, 8>(&st.ptr().offset(at), LoadOptions::default())
                            .iter()
                            .map(|v| v.to::<f32>())
                            .collect();
                        let mut ss = x.iter().fold(zero, |s, v| s + *v * *v);
                        for off in [8, 4, 2, 1] {
                            ss = ss + shuffle_sync_bfly(ss, off);
                        }
                        let inv = (ss + c(1e-6)).rsqrt();
                        let y: [Value<T>; 8] = std::array::from_fn(|i| (x[i] * inv).to::<T>());
                        store_vector::<T, 8>(&st.ptr().offset(at), y, StoreOptions::default());
                    }
                    sync_threads();
                }

                // ---- The lower 16x16 tiles of K K^T and Q K^T (ten each), five per warp,
                // gated: `A[t][s] = beta_t exp(G_t - G_s) k_t.k_s` (s < t) and
                // `M[t][s] = exp(G_t - G_s) q_t.k_s` (s <= t).
                // All five tiles' MMAs first (ten independent accumulator chains).
                let tiles: Vec<(Boolean, Int32, Int32)> = (0..5)
                    .map(|rr| {
                        let tile = warp + 4 * rr;
                        let is_q = tile.ge(10);
                        let p = select(is_q, tile - 10, tile);
                        let ti = b2i(p.ge(1)) + b2i(p.ge(3)) + b2i(p.ge(6));
                        (is_q, ti, p - ((ti * (ti + 1)) >> 1))
                    })
                    .collect();
                let mut accs = vec![vec![zero; 8]; 5];
                for kk in 0..8 {
                    for (&(is_q, ti, tj), acc) in tiles.iter().zip(accs.iter_mut()) {
                        let abase = select(is_q, Int32::from(S_Q / 8), kbase);
                        let a = ld_t(false, sw(abase, 16, ti * 16 + p1r, p1v + 2 * kk));
                        let b = ld_t(false, sw(kbase, 16, tj * 16 + p2r, p2v + 2 * kk));
                        mma2_t(&a, &b, acc);
                    }
                }
                for (&(is_q, ti, tj), acc) in tiles.iter().zip(&accs) {
                    let diag = ti.eq(tj);
                    for a in 0..2 {
                        let tl = gr + 8 * a as i32;
                        let t = ti * 16 + tl;
                        let lg_t = sf.load(lgb + t);
                        let beta_t = sf.load(btb + t);
                        for n in 0..2 {
                            let sl = 8 * n as i32 + 2 * q4;
                            let vals: Vec<(Float32, Float32)> = (0..2)
                                .map(|b| {
                                    let s = tj * 16 + sl + b;
                                    let x = acc[4 * n + 2 * a + b as usize];
                                    let decay = exp2(lg_t - sf.load(lgb + s));
                                    let kk = select(s.lt(t), beta_t * decay * x, 0.0f32);
                                    let qk = select(s.le(t), decay * x, 0.0f32);
                                    (kk, qk)
                                })
                                .collect();
                            // Float16 pair at `[t][tj * 16 + sl ..]` of `-A` or `M`.
                            if_(is_q | !diag, || {
                                let base = select(is_q, Int32::from(S_M / 8), Int32::from(S_L / 8));
                                let at = sw(base, 8, t, tj * 2 + n as i32) * 8 + 2 * q4;
                                let pair: [Value<F16>; 2] =
                                    std::array::from_fn(|b| select(is_q, vals[b].1, -vals[b].0).to::<F16>());
                                store_vector::<F16, 2>(&sh.ptr().offset(at), pair, StoreOptions::default());
                            });
                            if_(!is_q & diag, || {
                                let at = F_ADIAG + ti * ADIAG_STRIDE + tl * 16 + sl;
                                sf.store(at, vals[0].0);
                                sf.store(at + 1, vals[1].0);
                            });
                        }
                    }
                }
                sync_threads();

                // ---- Invert the four unit lower-triangular diagonal blocks `I + A_jj`
                // (threads 0-63, one column each, forward substitution in Float32) into the
                // diagonal blocks of the `-A` buffer.
                if_(tid.lt(64), || {
                    let blk = tid >> 4;
                    let col = tid & 15;
                    let base = F_ADIAG + blk * ADIAG_STRIDE;
                    let mut x: Vec<Float32> = Vec::with_capacity(16);
                    for r in 0..16usize {
                        let mut acc = select(col.eq(r as i32), 1.0f32, 0.0f32);
                        for s4 in 0..r.div_ceil(4) {
                            let row = load_vector::<f32, 4>(
                                &sf.ptr().offset(base + (r * 16 + 4 * s4) as i32),
                                LoadOptions::default(),
                            );
                            for (e, a) in row.iter().enumerate() {
                                let s = 4 * s4 + e;
                                if s < r {
                                    acc = acc - *a * x[s];
                                }
                            }
                        }
                        x.push(acc);
                    }
                    for (r, xr) in x.iter().enumerate() {
                        let row = blk * 16 + r as i32;
                        let colg = blk * 16 + col;
                        sh.store(sw(Int32::from(S_L / 8), 8, row, colg >> 3) * 8 + (col & 7), xr.to::<F16>());
                    }
                });

                // ---- Per warp (16 state rows): `S K^T` and `S Q^T` over the chunk.
                let s_cur = s_state.get();
                let mut sk = vec![zero; 32];
                let mut oi = vec![zero; 32];
                for kk in 0..8 {
                    let a = to_t(&s_cur[8 * kk..8 * kk + 8]);
                    for nb in 0..4 {
                        let bk = ld_t(false, sw(kbase, 16, Int32::from(16 * nb) + p2r, p2v + 2 * kk as i32));
                        let bq =
                            ld_t(false, sw(Int32::from(S_Q / 8), 16, Int32::from(16 * nb) + p2r, p2v + 2 * kk as i32));
                        mma2_t(&a, &bk, &mut sk[8 * nb as usize..8 * nb as usize + 8]);
                        mma2_t(&a, &bq, &mut oi[8 * nb as usize..8 * nb as usize + 8]);
                    }
                }
                // The gates of this thread's 16 columns (token `8 n + 2 q4 + b`).
                let col_t = |n: usize, b: usize| 8 * n as i32 + 2 * q4 + b as i32;
                // (Read where used: kept in registers across the barrier they would spill.)
                let lg_at = |n: usize, b: usize| sf.load(lgb + col_t(n, b));
                // `R = beta (V^T - gamma S K^T)`; V^T read transposed in accumulator layout.
                let mut r = vec![zero; 32];
                for tb in 0..4 {
                    let vt = ld_t(true, sw(Int32::from(S_V / 8), 8, Int32::from(16 * tb) + p2r, p2v + 2 * warp));
                    for nn in 0..2 {
                        let n = 2 * tb + nn;
                        for b in 0..2 {
                            let gam = exp2(lg_at(n, b));
                            let bt = sf.load(btb + col_t(n, b));
                            for a in 0..2 {
                                let (e, at) = (4 * nn + 2 * a + b, 4 * n + 2 * a + b);
                                r[at] = bt * (vt[e].to::<f32>() - gam * sk[at]);
                                oi[at] = oi[at] * gam;
                            }
                        }
                    }
                }
                sync_threads();

                // ---- Next chunk's loads (Q and V are dead, K goes to the other stage).
                if_((cc + 1).lt(nchunks), || issue_loads(cc + 1, tid, false, true));
                gate_var.set(gates(cc + 1, lane));

                // ---- Solve `(I + A) Y = R` block by block: `Y_j^T = (R_j^T - sum_k Y_k^T
                // A_jk^T) X_jj^T` (Float16 operands).
                let mut y = vec![zero; 32];
                for j in 0..4 {
                    let mut acc: Vec<Float32> = r[8 * j..8 * j + 8].to_vec();
                    for kb in 0..j {
                        let a = to_h(&y[8 * kb..8 * kb + 8]);
                        let b =
                            ld_h(sw(Int32::from(S_L / 8), 8, Int32::from(16 * j as i32) + p2r, p2v + 2 * kb as i32));
                        mma2_h(&a, &b, &mut acc);
                    }
                    let a = to_h(&acc);
                    let b = ld_h(sw(Int32::from(S_L / 8), 8, Int32::from(16 * j as i32) + p2r, p2v + 2 * j as i32));
                    let mut yj = vec![zero; 8];
                    mma2_h(&a, &b, &mut yj);
                    y[8 * j..8 * j + 8].copy_from_slice(&yj);
                }
                // The state update's A operands, `Y^T diag(gamma_C / gamma)`, and Y in Float16.
                let gc = sf.load(lgb + (CHUNK as i32 - 1));
                let mut ya: Vec<Value<T>> = Vec::with_capacity(32);
                for n in 0..8 {
                    let f: Vec<Float32> = (0..2).map(|b| exp2(gc - lg_at(n, b))).collect();
                    for idx in 0..4 {
                        ya.push((y[4 * n + idx] * f[idx % 2]).to::<T>());
                    }
                }
                let yh: Vec<Value<F16>> = to_h(&y);
                // ---- `O^T = scale (gamma S Q^T + Y^T M^T)`, staged to shared `[t][v]`.
                for jt in 0..4 {
                    let mut acc: Vec<Float32> = oi[8 * jt..8 * jt + 8].to_vec();
                    for sb in 0..=jt {
                        let b =
                            ld_h(sw(Int32::from(S_M / 8), 8, Int32::from(16 * jt as i32) + p2r, p2v + 2 * sb as i32));
                        mma2_h(&yh[8 * sb..8 * sb + 8], &b, &mut acc);
                    }
                    for (e, x) in acc.iter().enumerate() {
                        let (n, idx) = (2 * jt + e / 4, e % 4);
                        let t = col_t(n, idx % 2);
                        let vl = warp * 16 + gr + 8 * (idx / 2) as i32;
                        st.store(sw(Int32::from(S_O / 8), 8, t, vl >> 3) * 8 + (vl & 7), (*x * scale).to::<T>());
                    }
                }
                // ---- `S = gamma_C S + Y^T diag(gamma_C / gamma) K`.
                let dc = exp2(gc);
                let mut s_new: Vec<Float32> = s_cur.iter().map(|x| *x * dc).collect();
                for sb in 0..4 {
                    let a = &ya[8 * sb..8 * sb + 8];
                    for kb in 0..8 {
                        let b = ld_t(true, sw(kbase, 16, Int32::from(16 * sb as i32) + p1r, p1v + 2 * kb));
                        mma2_t(a, &b, &mut s_new[8 * kb as usize..8 * kb as usize + 8]);
                    }
                }
                s_state.set(s_new);
                sync_threads();

                // ---- O: shared -> global, 16-byte stores of the live rows.
                for it in 0..4 {
                    let row = (tid >> 3) + 16 * it;
                    let col = tid & 7;
                    let t = cc * CHUNK as i32 + row;
                    let vals = load_vector::<T, 8>(
                        &st.ptr().offset(sw(Int32::from(S_O / 8), 8, row, col) * 8),
                        LoadOptions::default(),
                    );
                    let at = ((bos + t) * hv + vhead) * vdim as i32 + band * BLK_V as i32 + col * 8;
                    if_(t.lt(len), || store_vector::<T, 8>(&out.offset(at), vals, StoreOptions::default()));
                }
            });

            if o.output_final_state {
                let s_fin = s_state.get();
                if_(write_ok, || {
                    for n in 0..16 {
                        for a in 0..2 {
                            let at = 4 * n + 2 * a;
                            store_n::<S>(&state_out.offset(state_at(slot_out, n, a)), &s_fin[at..at + 2]);
                        }
                    }
                });
            }
        });
    })
}

/// A prefill test problem; every input value is exactly representable in its dtype.
pub struct PrefillProblem {
    pub input: Dtype,
    pub state: StateDtype,
    pub opts: PrefillOptions,
    /// Tokens of each sequence (`cu_seqlens` is their prefix sum; zero is allowed).
    pub seqlens: Vec<usize>,
    /// q and k heads.
    pub hq: usize,
    /// v heads (and state heads).
    pub hv: usize,
    pub v: usize,
    pub scale: f32,
    /// Rows of the state pools (`seqlens.len()` without indices).
    pub slots: usize,
    /// The row each sequence reads (with `opts.has_indices`); negative starts from zero.
    pub indices: Vec<i32>,
    /// The row each sequence writes (with `opts.has_out_indices`).
    pub out_indices: Vec<i32>,
    /// Initial state pool `[slots, HV, V, K]`.
    pub state0: Vec<f32>,
    /// `[T, H, K]`.
    pub q: Vec<f32>,
    /// `[T, H, K]`.
    pub kk: Vec<f32>,
    /// `[T, HV, V]`.
    pub vv: Vec<f32>,
    /// The decay `alpha` (FlashInfer's linear-domain `g`), `[T, HV]` Float32.
    pub g: Vec<f32>,
    /// `[T, HV]` Float32.
    pub beta: Vec<f32>,
}

impl PrefillProblem {
    /// `seqlens`, `(hq, hv, v)`; `K` is [`PREFILL_K`] and `scale` is `1 / sqrt(K)`.
    pub fn new(
        input: Dtype,
        state: StateDtype,
        opts: PrefillOptions,
        seqlens: &[usize],
        (hq, hv, v): (usize, usize, usize),
        seed: u64,
    ) -> PrefillProblem {
        assert!(hv.is_multiple_of(hq) && v.is_multiple_of(BLK_V));
        let k = PREFILL_K;
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut uni = move |lo: f32, hi: f32| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((rng >> 33) as f32 / (1u64 << 31) as f32)
        };
        let n = seqlens.len();
        let reads = if opts.has_indices { n + 3 } else { n };
        let slots = if opts.has_out_indices { 2 * reads } else { reads };
        let indices: Vec<i32> = if opts.has_indices {
            (0..n).map(|i| if i % 5 == 3 { -1 } else { ((i * 7 + 2) % reads) as i32 }).collect()
        } else {
            vec![]
        };
        // Written rows above the read ones; a padded read still writes.
        let out_indices: Vec<i32> =
            if opts.has_out_indices { (0..n).map(|i| (reads + (i * 5 + 1) % reads) as i32).collect() } else { vec![] };
        let total: usize = seqlens.iter().sum();
        let mut mk =
            |n: usize, lo: f32, hi: f32, r: &dyn Fn(f32) -> f32| (0..n).map(|_| r(uni(lo, hi))).collect::<Vec<f32>>();
        let (ri, rs) = (|x| input.round(x), |x| state.round(x));
        let state0 = mk(slots * hv * v * k, -0.5, 0.5, &rs);
        let q = mk(total * hq * k, -1.0, 1.0, &ri);
        // Without the L2 norm, keys of about unit norm keep the recurrence stable.
        let kr = if opts.l2norm { 1.0 } else { 0.15 };
        let kk = mk(total * hq * k, -kr, kr, &ri);
        let vv = mk(total * hv * v, -1.0, 1.0, &ri);
        let g = mk(total * hv, 0.85, 1.0, &|x| x);
        let beta = mk(total * hv, 0.0, 1.0, &|x| x);
        PrefillProblem {
            input,
            state,
            opts,
            seqlens: seqlens.to_vec(),
            hq,
            hv,
            v,
            scale: 1.0 / (k as f32).sqrt(),
            slots,
            indices,
            out_indices,
            state0,
            q,
            kk,
            vv,
            g,
            beta,
        }
    }

    pub fn total(&self) -> usize {
        self.seqlens.iter().sum()
    }

    pub fn cu_seqlens(&self) -> Vec<i32> {
        let mut cu = vec![0i32];
        for l in &self.seqlens {
            cu.push(cu.last().unwrap() + *l as i32);
        }
        cu
    }

    pub fn kernel(&self) -> Kernel {
        prefill_kernel(self.input, self.state, self.opts, self.v)
    }

    pub fn launch(&self) -> LaunchConfig {
        LaunchConfig::new([(self.v / BLK_V) as u32, self.hv as u32, self.seqlens.len() as u32], [THREADS as u32, 1, 1])
    }

    /// The kernel's arguments (see [`prefill_kernel`]); the final state is argument 7, the
    /// output 8. The final-state pool starts zeroed.
    pub fn args(&self) -> Vec<KernelArg> {
        let k = PREFILL_K;
        assert!(self.slots * self.hv * self.v * k < 1 << 31, "offsets are 32-bit");
        assert!(self.total() * self.hv.max(self.hq) * self.v.max(k) < 1 << 31, "offsets are 32-bit");
        let idx = |on: bool, v: &Vec<i32>| KernelArg::i32(if on { v.clone() } else { vec![0] });
        let pool = self.slots * self.hv * self.v * k;
        vec![
            self.input.buffer(&self.q),
            self.input.buffer(&self.kk),
            self.input.buffer(&self.vv),
            KernelArg::f32(self.g.clone()),
            KernelArg::f32(self.beta.clone()),
            KernelArg::i32(self.cu_seqlens()),
            self.state.buffer(if self.opts.has_initial_state { &self.state0 } else { &[0.0][..] }),
            self.state.buffer(&vec![0.0; if self.opts.output_final_state { pool } else { 1 }]),
            self.input.buffer(&vec![0.0; self.total() * self.hv * self.v]),
            idx(self.opts.has_indices, &self.indices),
            idx(self.opts.has_out_indices, &self.out_indices),
            KernelArg::ScalarI32(self.hq as i32),
            KernelArg::ScalarI32(self.hv as i32),
            KernelArg::ScalarF32(self.scale),
        ]
    }

    /// The final-state pool and the output in f64: the token-by-token recurrence of
    /// [`Problem::reference`] (with `alpha` and `beta` given), from each sequence's
    /// initial state.
    pub fn reference(&self) -> (Vec<f64>, Vec<f64>) {
        let (hq, hv, kd, vd) = (self.hq, self.hv, PREFILL_K, self.v);
        let total = self.total();
        let o = self.opts;
        let norm = |src: &[f32]| -> Vec<f64> {
            src.chunks(kd)
                .flat_map(|s| {
                    let n =
                        if o.l2norm { (s.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() + 1e-6).sqrt() } else { 1.0 };
                    s.iter().map(move |x| *x as f64 / n)
                })
                .collect()
        };
        let (qn, kn) = (norm(&self.q), norm(&self.kk));
        let pool = if o.output_final_state { self.slots * hv * vd * kd } else { 0 };
        let mut state = vec![0.0f64; pool];
        let mut out = vec![0.0f64; total * hv * vd];
        let mut bos = 0;
        for (i, &len) in self.seqlens.iter().enumerate() {
            let slot = if o.has_indices { self.indices[i] } else { i as i32 };
            let out_slot = if o.has_out_indices { self.out_indices[i] } else { slot };
            for h in 0..hv {
                let gh = h / (hv / hq);
                for r in 0..vd {
                    let mut cur: Vec<f64> = if o.has_initial_state && slot >= 0 {
                        let at = ((slot as usize * hv + h) * vd + r) * kd;
                        self.state0[at..at + kd].iter().map(|x| *x as f64).collect()
                    } else {
                        vec![0.0; kd]
                    };
                    for t in bos..bos + len {
                        let alpha = self.g[t * hv + h] as f64;
                        let beta = self.beta[t * hv + h] as f64;
                        let kv = &kn[(t * hq + gh) * kd..(t * hq + gh + 1) * kd];
                        let qv = &qn[(t * hq + gh) * kd..(t * hq + gh + 1) * kd];
                        let hrow: Vec<f64> = cur.iter().map(|s| s * alpha).collect();
                        let hk: f64 = hrow.iter().zip(kv).map(|(a, b)| a * b).sum();
                        let u = (self.vv[(t * hv + h) * vd + r] as f64 - hk) * beta;
                        cur = hrow.iter().zip(kv).map(|(h, k)| h + k * u).collect();
                        out[(t * hv + h) * vd + r] =
                            cur.iter().zip(qv).map(|(a, b)| a * b).sum::<f64>() * self.scale as f64;
                    }
                    if o.output_final_state && len > 0 && out_slot >= 0 {
                        let at = ((out_slot as usize * hv + h) * vd + r) * kd;
                        state[at..at + kd].copy_from_slice(&cur);
                    }
                }
            }
            bos += len;
        }
        (state, out)
    }

    /// Tolerances (absolute plus relative) of `(state, out)`: the state and the chunk's
    /// intermediates go through the input type (or Float16) as MMA operands (measured errors
    /// are about a tenth of this).
    pub fn tolerances(&self) -> (f64, f64) {
        let t = 0.5 * self.input.tolerance() + self.state.tolerance();
        (t, t)
    }

    /// Check the final state and output against the reference; returns the largest scaled
    /// `(state, out)` errors.
    pub fn check(&self, state: &[f32], out: &[f32]) -> (f64, f64) {
        let (rs, ro) = self.reference();
        let (ts, to) = self.tolerances();
        let scaled = |got: &[f32], want: &[f64], tol: f64, what: &str| {
            let mut worst = 0.0f64;
            for (i, (&g, &r)) in got.iter().zip(want).enumerate() {
                let e = (g as f64 - r).abs() / (1.0 + r.abs());
                assert!(e <= tol, "{what} {i}: {g} against {r} ({:?} {:?})", self.opts, self.seqlens);
                worst = worst.max(e);
            }
            worst
        };
        (scaled(&state[..rs.len()], &rs, ts, "state"), scaled(out, &ro, to, "out"))
    }

    /// Bytes moved: q, k, v, out, the gates, and the states read and written.
    pub fn bytes(&self) -> f64 {
        let t = self.total();
        let io = t * (2 * self.hq * PREFILL_K + 2 * self.hv * self.v) * 2 + t * self.hv * 8;
        let one = self.seqlens.len() * self.hv * self.v * PREFILL_K * self.state.bytes();
        let st = one * (self.opts.has_initial_state as usize + self.opts.output_final_state as usize);
        (io + st) as f64
    }

    /// Tensor-core FLOPs of the chunked form as FlashInfer counts the useful work: per
    /// 64-token chunk and v head, `K K^T`, `Q K^T` (lower halves), `S K^T`, `S Q^T`, the
    /// state update (`V K` each) and the triangular solve and `Y M^T` (lower halves).
    pub fn flops(&self) -> f64 {
        let (c, k, v) = (CHUNK as f64, PREFILL_K as f64, self.v as f64);
        let chunks: f64 = self.seqlens.iter().map(|l| l.div_ceil(CHUNK) as f64).sum();
        let per = 2.0 * (c * c * k + 3.0 * c * v * k + c * c * v);
        chunks * self.hv as f64 * per
    }
}

/// Simulate the prefill kernel on `p` and check it against the f64 reference; returns the
/// largest scaled `(state, out)` errors.
pub fn simulate_prefill(p: &PrefillProblem) -> (f64, f64) {
    let k = p.kernel();
    let mut args = p.args();
    k.simulate(p.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.check(&p.state.read(&args[7]), &p.input.read(&args[8]))
}

/// Running on the GPU (`--features cuda`).
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::collections::HashMap;
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{Plan, PrefillProblem, Problem};

    /// Kernels compiled and loaded on first use.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: &'c Toolchain,
        loaded: HashMap<String, (Kernel, CudaFunction<'c>)>,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &'c Toolchain) -> Kernels<'c> {
            Kernels { ctx, tc, loaded: HashMap::new() }
        }

        pub fn get(&mut self, p: &Problem) -> Result<&(Kernel, CudaFunction<'c>), Box<dyn Error>> {
            self.load(p.kernel())
        }

        /// Compile and load `k` unless a kernel of its name is loaded.
        pub fn load(&mut self, k: Kernel) -> Result<&(Kernel, CudaFunction<'c>), Box<dyn Error>> {
            let key = k.name().to_string();
            if !self.loaded.contains_key(&key) {
                let arch = self.ctx.best_target(self.tc).ok_or("no target for this GPU")?;
                let f = self.ctx.load(&k.compile_with(self.tc, &CompileOptions::new().gpu_arch(arch))?)?;
                self.loaded.insert(key.clone(), (k, f));
            }
            Ok(&self.loaded[&key])
        }
    }

    /// The largest scaled `(state, out)` errors, and the time in microseconds.
    pub type RunResult = ((f64, f64), Option<f64>);

    /// Run `p` once; if `check`, compare with the f64 reference and return the largest scaled
    /// `(state, out)` errors; if `time`, also return the average GPU time in microseconds of
    /// launches replayed from a CUDA graph (each updates the state in place, as a decoder).
    pub fn run(
        ctx: &CudaContext,
        ks: &mut Kernels<'_>,
        p: &Problem,
        check: bool,
        time: bool,
    ) -> Result<RunResult, Box<dyn Error>> {
        let _: Plan = p.plan();
        let (k, f) = ks.get(p)?;
        let host = p.args();
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
            ctx.launch_builder(k, f, p.launch()).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let mut err = (0.0, 0.0);
        if check {
            let read_state = |buf: &cute::cuda::DeviceBuffer| -> Result<Vec<f32>, Box<dyn Error>> {
                Ok(match p.state {
                    super::StateDtype::F16 => buf.to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
                    super::StateDtype::BF16 => {
                        buf.to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
                    }
                    super::StateDtype::F32 => buf.to_vec::<f32>()?,
                })
            };
            let state = read_state(&bufs[0])?;
            let cache = read_state(&bufs[11])?;
            let out: Vec<f32> = match p.input {
                super::Dtype::F16 => bufs[8].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
                super::Dtype::BF16 => {
                    bufs[8].to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
                }
            };
            err = p.check(&state, &out, &cache);
        }
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((err, t))
    }

    fn read_floats(buf: &cute::cuda::DeviceBuffer, half: Option<super::Dtype>) -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(match half {
            Some(super::Dtype::F16) => buf.to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
            Some(super::Dtype::BF16) => {
                buf.to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
            }
            None => buf.to_vec::<f32>()?,
        })
    }

    /// Run the prefill problem `p` once; if `check`, compare with the f64 reference and return
    /// the largest scaled `(state, out)` errors; if `time`, also return the average GPU time in
    /// microseconds of launches replayed from a CUDA graph (the initial and final states are
    /// separate buffers, so every replay does the same work).
    pub fn run_prefill(
        ctx: &CudaContext,
        ks: &mut Kernels<'_>,
        p: &PrefillProblem,
        check: bool,
        time: bool,
    ) -> Result<RunResult, Box<dyn Error>> {
        let (k, f) = ks.load(p.kernel())?;
        let host = p.args();
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
            ctx.launch_builder(k, f, p.launch()).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let mut err = (0.0, 0.0);
        if check {
            let half = |s: super::StateDtype| match s {
                super::StateDtype::F16 => Some(super::Dtype::F16),
                super::StateDtype::BF16 => Some(super::Dtype::BF16),
                super::StateDtype::F32 => None,
            };
            let state = read_floats(&bufs[7], half(p.state))?;
            let out = read_floats(&bufs[8], Some(p.input))?;
            err = p.check(&state, &out);
        }
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(5).iterations(20).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((err, t))
    }
}
