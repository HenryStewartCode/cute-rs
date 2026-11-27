// SPDX-License-Identifier: Apache-2.0

//! Kimi Delta Attention (KDA) decode, after FlashInfer's `recurrent_kda`
//! (`flashinfer/kda_decode.py`, kernel `flashinfer/kda_kernels/recurrent_kda.py`, both
//! Apache-2.0) and its published reference `naive_recurrent_kda` (`tests/kda/test_recurrent_kda.py`).
//! It follows that design and is modelled on [`crate::gdn`]'s decode
//! kernel.
//!
//! KDA is Gated Delta Net with a per-channel gate: for each `(batch b, v-head hv)`, with the
//! state `S` of shape `[V, K]` (K contiguous), `h = hv / (HV / H)` the shared q/k head and
//! Float32 arithmetic throughout:
//!
//! ```text
//! g[k]  = exp(g_log[k])                                       (precomputed gate), or in the kernel:
//!         exp(-exp(A_log[h]) * softplus(g[k] + dt_bias[h K + k]))        ([`Gate::Softplus`])
//!         exp(lower_bound * sigmoid(exp(A_log[h]) * (g[k] + dt_bias[h K + k])))   ([`Gate::LowerBound`])
//! beta  = beta[b, hv]                                         (or its sigmoid, `beta_is_logit`)
//! q, k  = l2_normalize(q[b, h]), l2_normalize(k[b, h])        (if l2norm; rsqrt(sum + 1e-6))
//! q    *= scale
//! S    *= g (per column)                                      (decay)
//! u     = (v - S k) * beta                                    (delta rule, per row of S)
//! S    += u k^T                                               (rank-one update)
//! o     = S q
//! ```
//!
//! `A_log` is `[H]` and `dt_bias` `[H * K]`, indexed by the q/k head as FlashInfer's kernel does.
//!
//! * **Geometry** as [`crate::gdn`]: 128 threads, a warp owning state rows with its 32 lanes
//!   holding `K / 32` consecutive elements (16 bytes of state where possible), the state in
//!   registers, `unroll` rows in flight, a reduce-scatter for the `S q` outputs, and a state
//!   split over blocks when `batch * HV` is small. FlashInfer's one-warp kernel (at least 128
//!   sequence-heads) and its grouped-CTA kernel keep the same state-in-registers design; here
//!   the per-channel gate is also a register vector per lane, so a token costs one extra
//!   `K`-wide vector load per `(b, hv)` and a multiply per state element.
//! * **Dtypes**: `q`, `k`, `v`, `g`, `beta`, output Float16 or BFloat16 ([`Dtype`]); state
//!   Float32, BFloat16 (FlashInfer's) or Float16 ([`StateDtype`]); `A_log`, `dt_bias` Float32.
//!   Shapes: `q`, `k` `[batch, T, H, K]`, `v`, output `[batch, T, HV, V]`, `g` `[batch, T, HV,
//!   K]`, `beta` `[batch, T, HV]`, state `[slots, HV, V, K]`, all contiguous.
//! * **State pool**: `state_indices` (`has_indices`) names each batch entry's slot, optionally
//!   `output_state_indices` (`has_out_indices`) where the new state goes; a negative index is
//!   padding: the state is untouched and the output zero.
//! * **Multi-token decode** (`tokens > 1`, FlashInfer's `num_spec_tokens = T - 1`): the
//!   recurrence runs over the tokens with each row in registers, so the state is read once and
//!   written once. With `per_token_slots` the pool indices are `[batch, T]` and token `t`'s
//!   state is written to slot `indices[b, t]` (FlashInfer's checkpoint slots; token 0's slot is
//!   also the one read; a negative slot skips that token's state write and zeroes its output).
//!   `update_state = false` reads the pool only.
//!
//! * **Packed varlen** (`varlen`, FlashInfer's `cu_seqlens`): the inputs and output are packed
//!   `[total, ...]`, sequence `b` owning rows `cu_seqlens[b]..cu_seqlens[b + 1]` (at most `tokens`
//!   of them); one block column per sequence as before. Tokens past the sequence's length write no
//!   output and no state, as FlashInfer's `token_t < seq_len` guard. With `tokens > 1` the slots
//!   are per token (`per_token_slots`). **`has_accepted`** is `num_accepted_tokens`: the read
//!   slot is `indices[b, clamp(accepted[b] - 1, 0, tokens - 1)]`; a negative read slot reads
//!   slot 0, as FlashInfer. Both only change addressing: the same kernel body.
//!
//! Not covered: `initial_state_source`, the
//! frozen-verify kernels (`disable_state_update` with correction/kg caches), the Cake and
//! Blackwell backends, the `q`/`k` head count unequal to a divisor of `HV` and strided inputs.

use cute::prelude::*;

pub use crate::gdn::Plan;
pub use crate::mamba::StateDtype;
pub use crate::norm::Dtype;

/// Threads per block (4 warps).
pub const THREADS: usize = crate::gdn::THREADS;
/// Warps per block.
pub const WARPS: usize = crate::gdn::WARPS;

/// How the per-channel decay is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gate {
    /// `g` is the log-space decay; the kernel takes `exp(g)`.
    Precomputed,
    /// `exp(-exp(A_log) * softplus(g + dt_bias))`.
    Softplus,
    /// `exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))`; `lower_bound` is a kernel scalar.
    LowerBound,
}

/// Build-time options.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Options {
    /// L2-normalise `q` and `k` (FlashInfer's `use_qk_l2norm_in_kernel`).
    pub l2norm: bool,
    pub gate: Gate,
    /// Add `dt_bias` to `g` before the in-kernel gate (ignored for [`Gate::Precomputed`]).
    pub has_dt_bias: bool,
    /// `beta` is a logit: the kernel takes its sigmoid.
    pub beta_is_logit: bool,
    /// `state_indices` is given: the state is a pool and each batch entry names its slot.
    pub has_indices: bool,
    /// `output_state_indices` is given (with `has_indices`).
    pub has_out_indices: bool,
    /// Tokens per sequence.
    pub tokens: usize,
    /// Write the state back.
    pub update_state: bool,
    /// `state_indices` is `[batch, tokens]` and every token's state goes to its own slot.
    pub per_token_slots: bool,
    /// Packed variable-length input (FlashInfer's `cu_seqlens`): `q`, `k`, `v`, `g`, `beta` and
    /// the output hold `cu_seqlens[batch]` tokens in all, sequence `b` owning rows
    /// `cu_seqlens[b]..cu_seqlens[b + 1]` (at most `tokens` of them; the grid still has one block
    /// column per sequence). Tokens past a sequence's length write neither output nor state. With
    /// `tokens > 1` this needs `per_token_slots`.
    pub varlen: bool,
    /// `num_accepted_tokens` `[batch]` is given (with `per_token_slots`, `tokens > 1`): the state
    /// is read from slot `indices[b, clamp(accepted[b] - 1, 0, tokens - 1)]` instead of column 0.
    pub has_accepted: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            l2norm: true,
            gate: Gate::Precomputed,
            has_dt_bias: false,
            beta_is_logit: false,
            has_indices: false,
            has_out_indices: false,
            tokens: 1,
            update_state: true,
            per_token_slots: false,
            varlen: false,
            has_accepted: false,
        }
    }
}

fn c(v: f32) -> Float32 {
    Float32::from(v)
}

/// Parameter order of [`kernel`]: `state, q, k, v, g, beta, A_log, dt_bias, out, state_indices,
/// output_state_indices, num_q_heads, num_v_heads, scale, lower_bound, cu_seqlens,
/// num_accepted_tokens`. Absent index tensors, `A_log`, `dt_bias`, `cu_seqlens` and
/// `num_accepted_tokens` still take a (one-element) buffer.
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
    assert!(!opts.per_token_slots || (opts.has_indices && !opts.has_out_indices));
    assert!(opts.tokens >= 1);
    assert!(!opts.has_accepted || (opts.per_token_slots && opts.tokens > 1));
    assert!(!opts.varlen || opts.tokens == 1 || opts.per_token_slots);
    let o = opts;
    let name = format!(
        "flashinfer_kda_{}_{}_k{kdim}_v{vdim}_r{rpb}_u{unroll}_{}{}{}{}{}{}{}{}{}_t{}_u{}",
        input.name(),
        state_dtype.name(),
        o.l2norm as u8,
        o.gate as u8,
        o.has_dt_bias as u8,
        o.beta_is_logit as u8,
        o.has_indices as u8,
        o.has_out_indices as u8,
        o.per_token_slots as u8,
        o.varlen as u8,
        o.has_accepted as u8,
        o.tokens,
        o.update_state as u8,
    );
    Kernel::builder(&name).block_dim([THREADS as u32, 1, 1]).build(|| {
        let state = param_ptr::<S>("state");
        let q = param_ptr::<T>("q");
        let k = param_ptr::<T>("k");
        let v = param_ptr::<T>("v");
        let g = param_ptr::<T>("g");
        let beta = param_ptr::<T>("beta");
        let a_log = param_ptr::<f32>("A_log");
        let dt_bias = param_ptr::<f32>("dt_bias");
        let out = param_ptr::<T>("out");
        let indices = param_ptr::<i32>("state_indices");
        let out_indices = param_ptr::<i32>("output_state_indices");
        let hq = param_scalar::<i32>("num_q_heads");
        let hv = param_scalar::<i32>("num_v_heads");
        let scale = param_scalar::<f32>("scale");
        let lower_bound = param_scalar::<f32>("lower_bound");
        let cu_seqlens = param_ptr::<i32>("cu_seqlens");
        let accepted = param_ptr::<i32>("num_accepted_tokens");

        let tid = thread_idx().x;
        let lane = tid & 31;
        let warp = tid >> 5;
        let batch = block_idx().x;
        let vhead = block_idx().y;
        let row_off = block_idx().z * rpb as i32;
        let qhead = vhead / (hv / hq);
        let nt = o.tokens;
        // Row of token `t` in the packed inputs and whether the sequence has that token.
        let (toks, has_tok): (Vec<Int32>, Vec<Boolean>) = if o.varlen {
            let start = cu_seqlens.load_at(batch);
            let len = cu_seqlens.load_at(batch + 1) - start;
            (0..nt)
                .map(|t| {
                    let ok = len.ge(t as i32 + 1);
                    (select(ok, start + t as i32, 0i32), ok)
                })
                .unzip()
        } else {
            ((0..nt).map(|t| batch * nt as i32 + t as i32).collect(), vec![Boolean::constant(true); nt])
        };
        let tok = |t: usize| toks[t];
        // Slot of token `t` in `state_indices` (`[batch, tokens]` with `per_token_slots`).
        let sidx = |t: usize| batch * nt as i32 + t as i32;

        // Slots: read slot, per-token (or final) write slots and their validity.
        let read_col =
            if o.has_accepted { (accepted.load_at(batch) - 1).max(0).min(nt as i32 - 1) } else { Int32::constant(0) };
        let slot_in = if o.has_indices {
            indices.load_at(if o.per_token_slots { sidx(0) + read_col } else { batch })
        } else {
            batch
        };
        let valid = if o.has_indices { slot_in.ge(0) } else { Boolean::constant(true) };
        let load_slot = if o.has_indices { slot_in.max(0) } else { slot_in };
        let (store_slot, store_ok) = if o.has_out_indices {
            let s = out_indices.load_at(batch);
            (s.max(0), valid & s.ge(0))
        } else {
            (load_slot, valid)
        };
        // Per token, with `per_token_slots`: the token's slot and whether it is real.
        let tslots: Vec<(Int32, Boolean)> = (0..nt)
            .map(|t| {
                if o.per_token_slots {
                    let s = indices.load_at(sidx(t));
                    (s.max(0), s.ge(0))
                } else {
                    (store_slot, valid)
                }
            })
            .collect();

        // Per token: beta, this lane's slice of q and k, and of the per-channel gate: `nv`
        // vectors of `vs` elements at columns `(lane + 32 j) * vs`.
        let col = |j: usize| (lane + 32 * j as i32) * vs as i32;
        let load_cols = |p: &Ptr<T>, base: Int32| -> Vec<Float32> {
            (0..nv).flat_map(|j| load_n::<T>(&p.offset(base + col(j)), vs)).collect()
        };
        let a_exp = if o.gate == Gate::Precomputed {
            c(0.0)
        } else {
            cute::math::exp2(a_log.load_at(qhead) * c(std::f32::consts::LOG2_E), true)
        };
        let bias: Vec<Float32> = if o.gate != Gate::Precomputed && o.has_dt_bias {
            let n = vs.min(4);
            (0..nv)
                .flat_map(|j| {
                    let base = qhead * kdim as i32 + col(j);
                    (0..vs / n)
                        .flat_map(|h| load_n::<f32>(&dt_bias.offset(base + (h * n) as i32), n))
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            vec![]
        };
        let mut gates: Vec<Vec<Float32>> = Vec::with_capacity(nt);
        let mut betas = Vec::with_capacity(nt);
        let mut qrs: Vec<Vec<Float32>> = Vec::with_capacity(nt);
        let mut krs: Vec<Vec<Float32>> = Vec::with_capacity(nt);
        for t in 0..nt {
            let bx = beta.load_at(tok(t) * hv + vhead).to::<f32>();
            betas.push(if o.beta_is_logit {
                cute::math::div(c(1.0), c(1.0) + cute::math::exp2(bx * c(-std::f32::consts::LOG2_E), true), true)
            } else {
                bx
            });
            let gx = load_cols(&g, (tok(t) * hv + vhead) * kdim as i32);
            gates.push(
                gx.iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let x = if bias.is_empty() { *x } else { *x + bias[i] };
                        match o.gate {
                            Gate::Precomputed => cute::math::exp2(x * c(std::f32::consts::LOG2_E), true),
                            Gate::Softplus => {
                                // softplus(x) = log(1 + exp(x)), x itself above 20.
                                let e = cute::math::exp2(x * c(std::f32::consts::LOG2_E), true);
                                let sp = select(
                                    x.le(20.0f32),
                                    cute::math::log2(e + c(1.0), true) * c(std::f32::consts::LN_2),
                                    x,
                                );
                                cute::math::exp2(-a_exp * sp * c(std::f32::consts::LOG2_E), true)
                            }
                            Gate::LowerBound => {
                                let sig = cute::math::div(
                                    c(1.0),
                                    c(1.0) + cute::math::exp2(-a_exp * x * c(std::f32::consts::LOG2_E), true),
                                    true,
                                );
                                cute::math::exp2(lower_bound * sig * c(std::f32::consts::LOG2_E), true)
                            }
                        }
                    })
                    .collect(),
            );
            let qk_base = (tok(t) * hq + qhead) * kdim as i32;
            let mut qr = load_cols(&q, qk_base);
            let mut kr = load_cols(&k, qk_base);
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
        let state_at = |slot: Int32| (slot * hv + vhead) * vdim as i32 + row_off;
        let vo = |t: usize| (tok(t) * hv + vhead) * vdim as i32 + row_off;
        let first_row = warp * rpw as i32;
        let at = |base: Int32, r: &Int32, j: usize| ((base + *r) * sv + lane + 32 * j as i32) * vs as i32;
        let write_rows = |rows: &[Int32], cur: &[Vec<Float32>], slot: Int32, ok: Boolean| {
            for (r, row) in rows.iter().zip(cur) {
                for j in 0..nv {
                    let ptr = state.offset(at(state_at(slot), r, j));
                    let vals = &row[j * vs..(j + 1) * vs];
                    if o.has_indices || o.varlen {
                        if_(ok, || store_n::<S>(&ptr, vals));
                    } else {
                        store_n::<S>(&ptr, vals);
                    }
                }
            }
        };
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
                    let hs: Vec<Float32> = row.iter().zip(&gates[t]).map(|(s, gk)| *s * *gk).collect();
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
                // The outputs `S q` of the rows together: a reduce-scatter.
                let (sums, idx_shift) = reduce_scatter(partial, lane);
                let (slot_t, ok_t) = tslots[t];
                let live = if o.per_token_slots { ok_t } else { valid };
                let res = select(live, sums, 0.0f32);
                if_((lane & ((1 << idx_shift) - 1)).eq(0) & has_tok[t], || {
                    out.store_at(vo(t) + rows[0] + (lane >> idx_shift as i32), res.to::<T>())
                });
                if o.per_token_slots && o.update_state {
                    write_rows(&rows, &cur, slot_t, ok_t & has_tok[t]);
                }
            }
            if o.update_state && !o.per_token_slots {
                write_rows(&rows, &cur, store_slot, store_ok & has_tok[0]);
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
    /// v heads (and state and gate heads).
    pub hv: usize,
    pub k: usize,
    pub v: usize,
    pub scale: f32,
    pub lower_bound: f32,
    /// Slots in the state pool.
    pub slots: usize,
    /// Read slots (`[batch]`), or `[batch, tokens]` with `per_token_slots`; negative is padding.
    pub indices: Vec<i32>,
    /// Written slots (with `has_out_indices`).
    pub out_indices: Vec<i32>,
    pub state0: Vec<f32>,
    pub q: Vec<f32>,
    pub kk: Vec<f32>,
    pub vv: Vec<f32>,
    pub g: Vec<f32>,
    pub beta: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// Tokens of each sequence with `varlen` (`cu_seqlens` is their prefix sums); else empty.
    pub seqlens: Vec<usize>,
    /// `num_accepted_tokens` (with `has_accepted`); values outside `1..=tokens` are clamped.
    pub accepted: Vec<i32>,
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
        shape: (usize, usize, usize, usize, usize),
        seed: u64,
    ) -> Problem {
        assert!(!opts.varlen, "use Problem::new_varlen");
        Self::build(input, state, opts, shape, vec![], seed)
    }

    /// A packed variable-length problem (`opts.varlen`): `shape.0` is the number of sequences,
    /// `seqlens[b] <= opts.tokens` the tokens of sequence `b` (zero allowed).
    pub fn new_varlen(
        input: Dtype,
        state: StateDtype,
        opts: Options,
        shape: (usize, usize, usize, usize, usize),
        seqlens: &[usize],
        seed: u64,
    ) -> Problem {
        assert!(opts.varlen && seqlens.len() == shape.0 && seqlens.iter().all(|&l| l <= opts.tokens));
        Self::build(input, state, opts, shape, seqlens.to_vec(), seed)
    }

    fn build(
        input: Dtype,
        state: StateDtype,
        opts: Options,
        (batch, hq, hv, k, v): (usize, usize, usize, usize, usize),
        seqlens: Vec<usize>,
        seed: u64,
    ) -> Problem {
        assert!(hv.is_multiple_of(hq));
        let nt = opts.tokens;
        let total = if opts.varlen { seqlens.iter().sum() } else { batch * nt };
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut uni = move |lo: f32, hi: f32| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((rng >> 33) as f32 / (1u64 << 31) as f32)
        };
        let reads = if opts.per_token_slots {
            batch * nt
        } else if opts.has_indices {
            batch + 3
        } else {
            batch
        };
        // Written slots live above the read ones, so no block reads what another writes.
        let slots = if opts.has_out_indices { 2 * reads } else { reads };
        let indices: Vec<i32> = if opts.per_token_slots {
            // A shuffled permutation of the slots; every seventh later token is padding.
            (0..batch * nt).map(|i| if i % nt != 0 && i % 7 == 3 { -1 } else { ((i * 5 + 1) % reads) as i32 }).collect()
        } else if opts.has_indices {
            (0..batch).map(|i| if i % 5 == 3 { -1 } else { ((i * 7 + 2) % reads) as i32 }).collect()
        } else {
            vec![]
        };
        let out_indices: Vec<i32> = if opts.has_out_indices {
            indices.iter().map(|&i| if i < 0 { -1 } else { i + reads as i32 }).collect()
        } else {
            vec![]
        };
        let mut mk =
            |n: usize, lo: f32, hi: f32, r: &dyn Fn(f32) -> f32| (0..n).map(|_| r(uni(lo, hi))).collect::<Vec<f32>>();
        let (ri, rs) = (|x| input.round(x), |x| state.round(x));
        let state0 = mk(slots * hv * v * k, -1.0, 1.0, &rs);
        let q = mk(total * hq * k, -1.0, 1.0, &ri);
        let kk = mk(total * hq * k, -1.0, 1.0, &ri);
        let vv = mk(total * hv * v, -1.0, 1.0, &ri);
        let (glo, ghi) = if opts.gate == Gate::Precomputed { (-1.5, 0.0) } else { (-1.5, 1.5) };
        let g = mk(total * hv * k, glo, ghi, &ri);
        let beta = if opts.beta_is_logit { mk(total * hv, -3.0, 3.0, &ri) } else { mk(total * hv, 0.0, 1.0, &ri) };
        // Mostly in range, with one zero and one above `tokens` (both clamped).
        let accepted: Vec<i32> = if opts.has_accepted {
            (0..batch).map(|i| [1, 2, 0, nt as i32 + 2, nt as i32][i % 5]).collect()
        } else {
            vec![]
        };
        let a_log = mk(hq, -0.3, 0.3, &|x| x);
        let dt_bias = mk(hq * k, -0.3, 0.3, &|x| x);
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
            lower_bound: -5.0,
            slots,
            indices,
            out_indices,
            state0,
            q,
            kk,
            vv,
            g,
            beta,
            a_log,
            dt_bias,
            seqlens,
            accepted,
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

    /// Packed token offsets `[batch + 1]` (with `varlen`).
    pub fn cu_seqlens(&self) -> Vec<i32> {
        std::iter::once(0)
            .chain(self.seqlens.iter().scan(0, |s, &l| {
                *s += l as i32;
                Some(*s)
            }))
            .collect()
    }

    /// Tokens of the packed inputs and output.
    pub fn total_tokens(&self) -> usize {
        if self.opts.varlen { self.seqlens.iter().sum() } else { self.batch * self.opts.tokens }
    }

    /// The kernel's arguments (see [`kernel`]); the state is argument 0, the output 8.
    pub fn args(&self) -> Vec<KernelArg> {
        assert!(self.slots * self.hv * self.v * self.k < 1 << 31, "offsets are 32-bit");
        let idx = |on: bool, v: &Vec<i32>| KernelArg::i32(if on { v.clone() } else { vec![0] });
        vec![
            self.state.buffer(&self.state0),
            self.input.buffer(&self.q),
            self.input.buffer(&self.kk),
            self.input.buffer(&self.vv),
            self.input.buffer(&self.g),
            self.input.buffer(&self.beta),
            KernelArg::f32(self.a_log.clone()),
            KernelArg::f32(self.dt_bias.clone()),
            self.input.buffer(&vec![0.0; self.total_tokens().max(1) * self.hv * self.v]),
            idx(self.opts.has_indices, &self.indices),
            idx(self.opts.has_out_indices, &self.out_indices),
            KernelArg::ScalarI32(self.hq as i32),
            KernelArg::ScalarI32(self.hv as i32),
            KernelArg::ScalarF32(self.scale),
            KernelArg::ScalarF32(self.lower_bound),
            KernelArg::i32(if self.opts.varlen { self.cu_seqlens() } else { vec![0] }),
            KernelArg::i32(if self.opts.has_accepted { self.accepted.clone() } else { vec![0] }),
        ]
    }

    /// The new state pool and the output in f64: `(state, out)`; FlashInfer's
    /// `naive_recurrent_kda` token by token, with the state in `[V, K]` order and the gates of
    /// `naive_kda_gate` / `naive_kda_lowerbound_gate`.
    #[allow(clippy::needless_range_loop)]
    pub fn reference(&self) -> (Vec<f64>, Vec<f64>) {
        let (hq, hv, kd, vd, nt) = (self.hq, self.hv, self.k, self.v, self.opts.tokens);
        let o = &self.opts;
        let mut state: Vec<f64> = self.state0.iter().map(|&x| x as f64).collect();
        let mut out = vec![0.0f64; self.total_tokens() * hv * vd];
        let cu = if o.varlen { self.cu_seqlens() } else { vec![] };
        for b in 0..self.batch {
            let (first, len) = if o.varlen { (cu[b] as usize, self.seqlens[b]) } else { (b * nt, nt) };
            let col = if o.has_accepted { (self.accepted[b] - 1).clamp(0, nt as i32 - 1) as usize } else { 0 };
            let slot_of = |t: usize| if o.per_token_slots { self.indices[b * nt + t] } else { 0 };
            let slot = if o.per_token_slots {
                self.indices[b * nt + col].max(0)
            } else if o.has_indices {
                self.indices[b]
            } else {
                b as i32
            };
            let out_slot = if o.has_out_indices { self.out_indices[b] } else { slot };
            if slot < 0 {
                continue;
            }
            for h in 0..hv {
                let qh = h / (hv / hq);
                for r in 0..vd {
                    let base = |sl: i32| ((sl as usize * hv + h) * vd + r) * kd;
                    let mut cur: Vec<f64> = (0..kd).map(|i| self.state0[base(slot) + i] as f64).collect();
                    for t in 0..len {
                        let tok = first + t;
                        let beta = self.beta[tok * hv + h] as f64;
                        let beta = if o.beta_is_logit { 1.0 / (1.0 + (-beta).exp()) } else { beta };
                        let norm = |src: &[f32]| -> Vec<f64> {
                            let s = &src[(tok * hq + qh) * kd..(tok * hq + qh + 1) * kd];
                            let n = if o.l2norm {
                                (s.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() + 1e-6).sqrt()
                            } else {
                                1.0
                            };
                            s.iter().map(|x| *x as f64 / n).collect()
                        };
                        let qv: Vec<f64> = norm(&self.q).iter().map(|x| x * self.scale as f64).collect();
                        let kv = norm(&self.kk);
                        let a = (self.a_log[qh] as f64).exp();
                        let hrow: Vec<f64> = (0..kd)
                            .map(|i| {
                                let mut x = self.g[(tok * hv + h) * kd + i] as f64;
                                if o.gate != Gate::Precomputed && o.has_dt_bias {
                                    x += self.dt_bias[qh * kd + i] as f64;
                                }
                                let log_g = match o.gate {
                                    Gate::Precomputed => x,
                                    Gate::Softplus => -a * if x <= 20.0 { x.exp().ln_1p() } else { x },
                                    Gate::LowerBound => self.lower_bound as f64 / (1.0 + (-a * x).exp()),
                                };
                                cur[i] * log_g.exp()
                            })
                            .collect();
                        let hk: f64 = hrow.iter().zip(&kv).map(|(a, b)| a * b).sum();
                        let u = (self.vv[(tok * hv + h) * vd + r] as f64 - hk) * beta;
                        cur = hrow.iter().zip(&kv).map(|(h, k)| h + k * u).collect();
                        let live = !o.per_token_slots || slot_of(t) >= 0;
                        if live {
                            out[(tok * hv + h) * vd + r] = cur.iter().zip(&qv).map(|(a, b)| a * b).sum();
                        }
                        if o.per_token_slots && o.update_state && live {
                            let at = base(slot_of(t));
                            state[at..at + kd].copy_from_slice(&cur);
                        }
                    }
                    if !o.per_token_slots && o.update_state && out_slot >= 0 && len > 0 {
                        let at = base(out_slot);
                        state[at..at + kd].copy_from_slice(&cur);
                    }
                }
            }
        }
        (state, out)
    }

    /// Check the new state and output against the reference (absolute plus relative); returns
    /// the largest scaled `(state, out)` errors.
    pub fn check(&self, state: &[f32], out: &[f32]) -> (f64, f64) {
        let (rs, ro) = self.reference();
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
        (scaled(state, &rs, ts, "state"), scaled(out, &ro, to, "out"))
    }

    /// Bytes moved: the state read (and written with `update_state`, once per token with
    /// `per_token_slots`), `q`, `k`, `v`, `g`, `out`, `beta`.
    pub fn bytes(&self) -> f64 {
        let o = &self.opts;
        let one = self.batch * self.hv * self.v * self.k * self.state.bytes();
        let writes = if !o.update_state {
            0
        } else if o.per_token_slots {
            o.tokens
        } else {
            1
        };
        let io = self.total_tokens() * (2 * self.hq * self.k + 3 * self.hv * self.v + self.hv) * 2;
        (one * (1 + writes) + io) as f64
    }
}

/// Simulate the kernel on `p` and check it against the f64 reference; returns the largest
/// scaled `(state, out)` errors.
pub fn simulate(p: &Problem) -> (f64, f64) {
    let k = p.kernel();
    let mut args = p.args();
    k.simulate(p.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.check(&p.state.read(&args[0]), &p.input.read(&args[8]))
}

#[cfg(feature = "cuda")]
pub mod gpu {
    use std::collections::HashMap;
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::Problem;

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
            let k = p.kernel();
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
            let state: Vec<f32> = match p.state {
                super::StateDtype::F16 => bufs[0].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
                super::StateDtype::BF16 => {
                    bufs[0].to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
                }
                super::StateDtype::F32 => bufs[0].to_vec::<f32>()?,
            };
            let out: Vec<f32> = match p.input {
                super::Dtype::F16 => bufs[8].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
                super::Dtype::BF16 => {
                    bufs[8].to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
                }
            };
            err = p.check(&state, &out);
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
