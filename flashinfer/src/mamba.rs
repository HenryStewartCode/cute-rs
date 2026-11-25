// SPDX-License-Identifier: Apache-2.0

//! Mamba-2 `selective_state_update`, after FlashInfer's single-token "simple" kernel
//! (`include/flashinfer/mamba/kernel_selective_state_update_stp.cuh`, the only algorithm
//! FlashInfer picks below SM90) and its launcher, and its multi-token (MTP) update. This is cute-rs's own code written after
//! the published design; no FlashInfer source was translated.
//!
//! For each `(batch, head, row d)` of the state `[cache, nheads, dim, dstate]`:
//!
//! ```text
//! dt'   = dt[b, h] + dt_bias[h]            (softplus'd, above 20 left as is, if dt_softplus)
//! dA    = exp(A[h] * dt')
//! s'[n] = s[n] * dA + B[g, n] * dt' * x[d]  (written back to the state)
//! out   = sum_n s'[n] * C[g, n] + D[h] * x[d]       (times silu(z[d]) with z)
//! ```
//!
//! with `g = h / (nheads / ngroups)`, in Float32.
//!
//! * **Geometry**: FlashInfer's: 128 threads (4 warps); a warp owns a row at a time, its
//!   32 lanes each holding `VS = min(16 bytes / sizeof(state), dstate / 32)` consecutive
//!   state elements, `dstate / (32 VS)` vectors per lane, and the row's `out` is a
//!   butterfly over the warp. A launch with `batch * nheads < 2 * SMs` tiles the dim:
//!   grid `(batch, nheads, dim / 4)`, one row per warp; otherwise grid `(batch, nheads)`
//!   and each warp walks `dim / 4` rows.
//! * **Multi-token**: 8 lanes share a row, as FlashInfer's MTP kernel does (4 rows per warp
//!   at once): a token's reduction is then 3 shuffles shared by 4 rows rather than 5 per
//!   row, which was the cost of the 32-lane form (17.5 us against FlashInfer's 14.0 us at
//!   batch 1, 6 tokens; 10.7 us now).
//! * **Difference**: a lane keeps its `B` and `C` slice in registers for every row it
//!   handles (FlashInfer stages `B`, `C`, `x` and `z` in shared memory and syncs), so the
//!   kernel has no shared memory and no barrier. A walk handles `unroll` rows at once, all
//!   state loads before the math, for memory-level parallelism.
//!
//! Dtypes: `x`, `B`, `C`, `z`, `out` Float16 or BFloat16 ([`Dtype`]); state Float16,
//! BFloat16 or Float32 ([`StateDtype`]); `dt`, `A`, `D`, `dt_bias` Float32 (FlashInfer's
//! defaults). `z`, `D`, `dt_bias`, `state_batch_indices` (with `pad_slot_id`: such a slot
//! reads as zeros and is not written), `dt_softplus` and `update_state` are build-time
//! options ([`Options`]). Tensors are contiguous: `x`, `z`, `out` `[batch, nheads, dim]`,
//! `dt` `[batch, nheads]`, `B`, `C` `[batch, ngroups, dstate]` (each with a `tokens` axis after
//! `batch` in the multi-token form, `Options::tokens > 1`), `A`, `D`, `dt_bias`
//! `[nheads]` (FlashInfer's `(nheads, dim, dstate)` `A` with zero strides, `dt` over `dim`).
//!
//! * **Sequence options** ([`Options`]), the semantics of FlashInfer's MTP "simple" kernel:
//!   `cu_seqlens` (`varlen`: packed `[total_tokens, ..]` inputs, `tokens` the longest sequence,
//!   a token past a sequence leaves the state alone, an empty sequence is skipped);
//!   `num_accepted_tokens` (the state is read from column `max(accepted - 1, 0)` of 2D
//!   `[batch, tokens]` indices); `dst_state_batch_indices` (written per token for several
//!   tokens, per sequence for one; `pad_slot_id` skipped); and the MTP cache
//!   (`intermediate_states` `[cache, tokens, nheads, dim, dstate]` at
//!   `intermediate_state_indices[b]`, or the state slot, written for every token, the main
//!   state left alone). The state is already in registers, so each is a per-token store.
//! * **int16 state** (`Options::int16`, FlashInfer's `state_scale`): a Float32 decode scale per
//!   `(slot, head, row)`; the row's new maximum (a butterfly over its lanes) sets the next
//!   scale, `max|s| / 32767`, and the stored value is `round(s * 32767 / max|s|)`. Its
//!   `Problem::state` is [`StateDtype::F16`], a 2-byte stand-in (the enum is shared with
//!   [`crate::gdn`] and [`crate::kda`]).
//!
//! Not yet covered: stochastic rounding (Philox), strided inputs, the SM90 TMA kernels
//! (vertical, horizontal), `dim` not a multiple of 4, `dstate` not a multiple of 32, and
//! offsets beyond 2^31 elements.
//!
//! `examples/mamba.rs` runs it on the simulator and, with `--features cuda` and `GPU=1`, on
//! the GPU against a reference, then benchmarks it.

use cute::prelude::*;

pub use crate::norm::Dtype;

/// Threads per block (4 warps).
pub const THREADS: usize = 128;
/// Warps per block.
pub const WARPS: usize = 4;

/// The element type of the state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StateDtype {
    F16,
    BF16,
    F32,
}

impl StateDtype {
    pub fn name(self) -> &'static str {
        match self {
            StateDtype::F16 => "f16",
            StateDtype::BF16 => "bf16",
            StateDtype::F32 => "f32",
        }
    }

    pub fn bytes(self) -> usize {
        if self == StateDtype::F32 { 4 } else { 2 }
    }

    /// Round `x` to this type.
    pub fn round(self, x: f32) -> f32 {
        match self {
            StateDtype::F16 => Dtype::F16.round(x),
            StateDtype::BF16 => Dtype::BF16.round(x),
            StateDtype::F32 => x,
        }
    }

    /// Relative rounding error of one store, with margin.
    pub fn tolerance(self) -> f64 {
        match self {
            StateDtype::F16 => 2e-3,
            StateDtype::BF16 => 1.6e-2,
            StateDtype::F32 => 1e-5,
        }
    }

    pub fn buffer(self, v: &[f32]) -> KernelArg {
        match self {
            StateDtype::F16 => Dtype::F16.buffer(v),
            StateDtype::BF16 => Dtype::BF16.buffer(v),
            StateDtype::F32 => KernelArg::f32(v.to_vec()),
        }
    }

    pub fn read(self, a: &KernelArg) -> Vec<f32> {
        match self {
            StateDtype::F16 => Dtype::F16.read(a),
            StateDtype::BF16 => Dtype::BF16.read(a),
            StateDtype::F32 => a.to_vec::<f32>(),
        }
    }

    /// Elements per lane per vector: 16 bytes at most, and `dstate / 32` at most.
    pub fn vec(self, dstate: usize) -> usize {
        (16 / self.bytes()).min(dstate / 32)
    }
}

/// Build-time options.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Options {
    /// `z` is given: the output is multiplied by `silu(z)`.
    pub has_z: bool,
    /// `D` is given.
    pub has_d: bool,
    /// `dt_bias` is given.
    pub has_dt_bias: bool,
    /// `state_batch_indices` is given (with the `pad_slot_id`).
    pub has_indices: bool,
    /// Apply softplus to `dt`.
    pub dt_softplus: bool,
    /// Write the new state back (FlashInfer's `update_state`).
    pub update_state: bool,
    /// Tokens per sequence: 1 is the single-token kernel, more are FlashInfer's multi-token
    /// (MTP) update, which runs the recurrence over the tokens with the state in registers
    /// and writes only the last state.
    pub tokens: usize,
    /// `dst_state_batch_indices` is given: the state is read from `state_batch_indices` and
    /// written to these slots (`[batch]` for one token, `[batch, tokens]` for several, one
    /// write per token; `pad_slot_id` entries are skipped). Exclusive with `intermediate`.
    pub has_dst: bool,
    /// `cu_seqlens` is given: the inputs are packed `[total_tokens, ..]` and `tokens` is the
    /// longest sequence (FlashInfer's `cache_steps`); a sequence of no tokens is skipped.
    pub varlen: bool,
    /// `num_accepted_tokens` is given (needs `has_indices`, whose indices are then
    /// `[batch, tokens]`): the state is read from column `max(accepted - 1, 0)`.
    pub has_accepted: bool,
    /// The MTP cache: every token's state goes to `intermediate_states`
    /// `[cache, tokens, nheads, dim, dstate]` at `[intermediate_state_indices[b], t]` and the
    /// main state is not written.
    pub intermediate: bool,
    /// `intermediate_state_indices` `[batch]` is given (else the state slot is used).
    pub istate_indices: bool,
    /// The state is block-scaled int16 (FlashInfer's `state_scale`): one Float32 decode scale
    /// per `(slot, head, row)`, the stored value being `round(state * 32767 / max|row|)`.
    /// `state` of [`Problem`] must then be [`StateDtype::F16`] (a 2-byte stand-in).
    pub int16: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            has_z: false,
            has_d: true,
            has_dt_bias: true,
            has_indices: false,
            dt_softplus: true,
            update_state: true,
            tokens: 1,
            has_dst: false,
            varlen: false,
            has_accepted: false,
            intermediate: false,
            istate_indices: false,
            int16: false,
        }
    }
}

/// How a launch is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Plan {
    /// Rows of the dim per block (`dim` when untiled).
    pub rows_per_block: usize,
    /// Rows a warp handles at once (all their state loads first).
    pub unroll: usize,
    /// Lanes that share a row (a divisor of 32): `32 / lanes` rows per warp at once, and
    /// the row's reduction is `log2(lanes)` shuffles. FlashInfer's multi-token kernel uses 8.
    pub lanes: usize,
}

impl Plan {
    /// FlashInfer's rule: tile the dim (4 rows per block) when `batch * nheads` is below
    /// two blocks per SM.
    pub fn new(batch: usize, nheads: usize, dim: usize, tokens: usize, num_sms: usize) -> Plan {
        assert!(dim.is_multiple_of(WARPS), "dim = {dim} must be a multiple of {WARPS}");
        let rows_per_block = if batch * nheads < 2 * num_sms { WARPS } else { dim };
        let per_warp = rows_per_block / WARPS;
        // Rows in flight per lane group: 8 with several tokens, else 4; 16 measured slower.
        let max_unroll = if tokens > 1 { 8 } else { 4 };
        // Several tokens: 8 lanes per row (FlashInfer's), so each token's reduction is 3
        // shuffles for 4 rows at once instead of 5 per row. Measured: 32 lanes 17.5 us,
        // 16 lanes 12.6 us, 8 lanes 10.7 us at batch 1, 6 tokens.
        let lanes = if tokens > 1 { 8 } else { 32 };
        // Fewer rows per warp at once if the block's rows do not divide.
        let lanes = (0..6).map(|i| lanes << i).find(|l| *l == 32 || per_warp.is_multiple_of(32 / l)).unwrap();
        let groups = 32 / lanes;
        let unroll = [8, 4, 2, 1]
            .into_iter()
            .filter(|u| *u <= max_unroll)
            .find(|u| per_warp.is_multiple_of(u * groups))
            .unwrap();
        Plan { rows_per_block, unroll, lanes }
    }
}

fn c(v: f32) -> Float32 {
    Float32::from(v)
}

/// Parameter order of [`kernel`]: `state, x, dt, A, B, C, D, dt_bias, z, out,
/// state_batch_indices, dst_state_batch_indices, cu_seqlens, num_accepted_tokens,
/// intermediate_states, intermediate_state_indices, state_scale, intermediate_state_scales,
/// nheads, ngroups, pad_slot_id`. Absent optional tensors still take
/// a (one-element) buffer.
pub fn kernel(input: Dtype, state: StateDtype, opts: Options, dim: usize, dstate: usize, plan: Plan) -> Kernel {
    if opts.int16 {
        return match input {
            Dtype::F16 => kernel_of::<F16, i16>(input, state, opts, dim, dstate, plan),
            Dtype::BF16 => kernel_of::<BF16, i16>(input, state, opts, dim, dstate, plan),
        };
    }
    match (input, state) {
        (Dtype::F16, StateDtype::F16) => kernel_of::<F16, F16>(input, state, opts, dim, dstate, plan),
        (Dtype::F16, StateDtype::BF16) => kernel_of::<F16, BF16>(input, state, opts, dim, dstate, plan),
        (Dtype::F16, StateDtype::F32) => kernel_of::<F16, f32>(input, state, opts, dim, dstate, plan),
        (Dtype::BF16, StateDtype::F16) => kernel_of::<BF16, F16>(input, state, opts, dim, dstate, plan),
        (Dtype::BF16, StateDtype::BF16) => kernel_of::<BF16, BF16>(input, state, opts, dim, dstate, plan),
        (Dtype::BF16, StateDtype::F32) => kernel_of::<BF16, f32>(input, state, opts, dim, dstate, plan),
    }
}

fn kernel_of<T: Numeric, S: Numeric>(
    input: Dtype,
    state_dtype: StateDtype,
    opts: Options,
    dim: usize,
    dstate: usize,
    plan: Plan,
) -> Kernel {
    let Plan { rows_per_block: rpb, unroll, lanes } = plan;
    assert!(lanes.is_power_of_two() && lanes <= 32);
    let groups = 32 / lanes;
    let vs = (16 / state_dtype.bytes()).min(dstate / lanes);
    assert!(vs >= 1 && dstate.is_multiple_of(lanes * vs), "dstate = {dstate} must be a multiple of {}", lanes * vs);
    let nv = dstate / (lanes * vs);
    assert!(rpb.is_multiple_of(WARPS) && dim.is_multiple_of(rpb));
    let rpw = rpb / WARPS;
    let step = unroll * groups;
    assert!(rpw.is_multiple_of(step));
    let tiled = rpb < dim;
    let o = opts;
    assert!(!(o.has_dst && o.intermediate), "dst_state_batch_indices and the intermediate cache are exclusive");
    assert!(!o.has_accepted || o.has_indices, "num_accepted_tokens needs state_batch_indices");
    assert!(!o.istate_indices || o.intermediate);
    assert!(!o.int16 || state_dtype == StateDtype::F16, "int16 state: pass StateDtype::F16");
    let scaled = o.int16;
    let name = format!(
        "flashinfer_ssu_{}_{}_d{dim}_n{dstate}_r{rpb}_u{unroll}_l{lanes}_{}{}{}{}{}{}_t{}_{}{}{}{}{}{}",
        input.name(),
        if scaled { "i16" } else { state_dtype.name() },
        o.has_z as u8,
        o.has_d as u8,
        o.has_dt_bias as u8,
        o.has_indices as u8,
        o.dt_softplus as u8,
        o.update_state as u8,
        o.tokens,
        o.has_dst as u8,
        o.varlen as u8,
        o.has_accepted as u8,
        o.intermediate as u8,
        o.istate_indices as u8,
        o.int16 as u8,
    );
    Kernel::builder(&name).block_dim([THREADS as u32, 1, 1]).build(|| {
        let state = param_ptr::<S>("state");
        let x = param_ptr::<T>("x");
        let dt = param_ptr::<f32>("dt");
        let a = param_ptr::<f32>("A");
        let bm = param_ptr::<T>("B");
        let cm = param_ptr::<T>("C");
        let dd = param_ptr::<f32>("D");
        let dt_bias = param_ptr::<f32>("dt_bias");
        let z = param_ptr::<T>("z");
        let out = param_ptr::<T>("out");
        let indices = param_ptr::<i32>("state_batch_indices");
        let dst_indices = param_ptr::<i32>("dst_state_batch_indices");
        let cu_seqlens = param_ptr::<i32>("cu_seqlens");
        let accepted = param_ptr::<i32>("num_accepted_tokens");
        let istate = param_ptr::<S>("intermediate_states");
        let istate_indices = param_ptr::<i32>("intermediate_state_indices");
        let state_scale = param_ptr::<f32>("state_scale");
        let istate_scale = param_ptr::<f32>("intermediate_state_scales");
        let nheads = param_scalar::<i32>("nheads");
        let ngroups = param_scalar::<i32>("ngroups");
        let pad_slot_id = param_scalar::<i32>("pad_slot_id");

        let tid = thread_idx().x;
        let lane = tid & 31;
        let member = lane & (lanes as i32 - 1);
        let group_in_warp = lane / lanes as i32;
        let warp = tid >> 5;
        let batch = block_idx().x;
        let head = block_idx().y;
        let dim_off = if tiled { block_idx().z * rpb as i32 } else { Int32::from(0) };
        let group = head / (nheads / ngroups);

        let nt = o.tokens;
        // The sequence's first token and length: `cu_seqlens`, or `tokens` each.
        let (bos, seq_len) = if o.varlen {
            let b0 = cu_seqlens.load_at(batch);
            (b0, cu_seqlens.load_at(batch + 1) - b0)
        } else {
            (batch * nt as i32, Int32::from(nt as i32))
        };
        // The state is read from the column `max(accepted - 1, 0)` of the indices.
        let init = if o.has_accepted { (accepted.load_at(batch) - 1).max(0) } else { Int32::from(0) };
        let idx_cols = if o.has_accepted { nt as i32 } else { 1 };
        let state_batch = if o.has_indices { indices.load_at(batch * idx_cols + init) } else { batch };
        let valid = if o.has_indices { state_batch.ne(pad_slot_id) } else { Boolean::constant(true) };
        let load_slot = if o.has_indices { state_batch.max(0) } else { state_batch };
        let valid_opt = o.has_indices.then_some(valid);
        let both = |a: Option<Boolean>, b: Option<Boolean>| match (a, b) {
            (Some(a), Some(b)) => Some(a & b),
            (a, b) => a.or(b),
        };
        let act = |t: usize| o.varlen.then(|| seq_len.gt(t as i32));

        let body = || {
            // Per token: dt (with bias and softplus), dA, and this lane's B and C slice, `nv`
            // vectors of `vs` elements at columns `(member + lanes j) * vs`, reused for every row.
            let d_value = if o.has_d { Some(dd.load_at(head)) } else { None };
            // Tokens past the sequence read the last one (and are masked below).
            let tok = |t: usize| if o.varlen { bos + (seq_len - 1).min(t as i32) } else { bos + t as i32 };
            let mut d_as = Vec::with_capacity(nt);
            let mut db_regs: Vec<Vec<Float32>> = Vec::with_capacity(nt);
            let mut c_regs: Vec<Vec<Float32>> = Vec::with_capacity(nt);
            for t in 0..nt {
                let mut dt_value = dt.load_at(tok(t) * nheads + head);
                if o.has_dt_bias {
                    dt_value = dt_value + dt_bias.load_at(head);
                }
                if o.dt_softplus {
                    // `__logf(1 + __expf(x))`, and x itself above 20.
                    let e = cute::math::exp2(dt_value * c(std::f32::consts::LOG2_E), true);
                    let sp = cute::math::log2(e + c(1.0), true) * c(std::f32::consts::LN_2);
                    dt_value = select(dt_value.le(20.0f32), sp, dt_value);
                }
                d_as.push(cute::math::exp2(a.load_at(head) * dt_value * c(std::f32::consts::LOG2_E), true));
                let bc_base = (tok(t) * ngroups + group) * dstate as i32;
                let load_bc = |p: &Ptr<T>| -> Vec<Float32> {
                    (0..nv)
                        .flat_map(|j| {
                            load_n::<T>(&p.offset(bc_base + (member + lanes as i32 * j as i32) * vs as i32), vs)
                        })
                        .collect()
                };
                // `B * dt`, once.
                db_regs.push(load_bc(&bm).iter().map(|b| *b * dt_value).collect());
                c_regs.push(load_bc(&cm));
            }
            // Where each token's state goes. The destination slot of token `t` (uniform).
            let dst_slots: Vec<Int32> = (0..nt)
                .map(|t| if o.has_dst { dst_indices.load_at(batch * nt as i32 + t as i32) } else { Int32::from(0) })
                .collect();
            let icache = if o.istate_indices { istate_indices.load_at(batch) } else { state_batch };

            let row_base = |t: usize| (tok(t) * nheads + head) * dim as i32 + dim_off;
            let first_row = warp * rpw as i32;
            let sv = (dstate / vs) as i32;
            // The element offset of `r`'s vector `j` in `slot`.
            let at = |slot: Int32, r: &Int32, j: usize| {
                (((slot * nheads + head) * dim as i32 + dim_off + *r) * sv + member + lanes as i32 * j as i32)
                    * vs as i32
            };
            let scale_at = |slot: Int32, r: &Int32| (slot * nheads + head) * dim as i32 + dim_off + *r;
            // Store `cur` as `r`'s state in `slot` of `ptr` (with the scale in `sptr`), if `cond`.
            let write =
                |ptr: &Ptr<S>, sptr: &Ptr<f32>, slot: Int32, r: &Int32, cur: &[Float32], cond: Option<Boolean>| {
                    // int16: scale the row so its largest value is 32767 (all of the row's lanes agree).
                    let (vals, scale) = if scaled {
                        let mx = cur.iter().skip(1).fold(cur[0].abs(), |m, v| m.max(v.abs()));
                        let mx = warp_reduction_max(mx, lanes);
                        let enc = select(mx.eq(0.0f32), c(1.0), c(32767.0) / mx);
                        let q: Vec<Float32> =
                            cur.iter().map(|v| (*v * enc).max(-32767.0f32).min(32767.0f32).rint()).collect();
                        (q, Some(c(1.0) / enc))
                    } else {
                        (cur.to_vec(), None)
                    };
                    let go = || {
                        for j in 0..nv {
                            store_n::<S>(&ptr.offset(at(slot, r, j)), &vals[j * vs..(j + 1) * vs]);
                        }
                        if let Some(sc) = scale {
                            if_(member.eq(0), || sptr.store_at(scale_at(slot, r), sc));
                        }
                    };
                    match cond {
                        Some(cn) => if_(cn, go),
                        None => go(),
                    }
                };
            let run = |r0: Int32| {
                // The rows `r0 .. r0 + unroll` of this warp (relative to the block's tile).
                let rows: Vec<Int32> = (0..unroll).map(|k| r0 + (k * groups) as i32 + group_in_warp).collect();
                let xs: Vec<Vec<Float32>> =
                    rows.iter().map(|r| (0..nt).map(|t| x.load_at(row_base(t) + *r).to::<f32>()).collect()).collect();
                // All state loads first.
                let olds: Vec<Vec<Float32>> = rows
                    .iter()
                    .map(|r| (0..nv).flat_map(|j| load_n::<S>(&state.offset(at(load_slot, r, j)), vs)).collect())
                    .collect();
                for ((r, xrow), old) in rows.iter().zip(&xs).zip(&olds) {
                    let decode = if scaled { Some(state_scale.load_at(scale_at(load_slot, r))) } else { None };
                    let mut cur: Vec<Float32> = old
                        .iter()
                        .map(|s| {
                            let s = decode.map_or(*s, |d| *s * d);
                            if o.has_indices { select(valid, s, 0.0f32) } else { s }
                        })
                        .collect();
                    for t in 0..nt {
                        let mut acc = c(0.0);
                        for (i, s) in cur.iter_mut().enumerate() {
                            let ns = *s * d_as[t] + db_regs[t][i] * xrow[t];
                            acc = acc + ns * c_regs[t][i];
                            // Past the sequence the state stays.
                            *s = match act(t) {
                                Some(on) => select(on, ns, *s),
                                None => ns,
                            };
                        }
                        let mut res = warp_reduction_sum(acc, lanes);
                        if let Some(dv) = d_value {
                            res = res + dv * xrow[t];
                        }
                        if o.has_z {
                            let zv = z.load_at(row_base(t) + *r).to::<f32>();
                            // `z * sigmoid(z)`, `__fdividef(1, 1 + __expf(-z))`.
                            let e = cute::math::exp2(zv * c(-std::f32::consts::LOG2_E), true);
                            res = res * cute::math::div(zv, e + c(1.0), true);
                        }
                        let store_out = both(Some(member.eq(0)), act(t)).unwrap();
                        if_(store_out, || out.store_at(row_base(t) + *r, res.to::<T>()));
                        // The state of this token: the cache, a destination slot, or (after the
                        // last token) the state itself.
                        if o.intermediate {
                            let slot = icache * nt as i32 + t as i32;
                            write(&istate, &istate_scale, slot, r, &cur, both(valid_opt, act(t)));
                        } else if o.has_dst {
                            if nt == 1 && !o.update_state {
                                continue;
                            }
                            let cond = both(both(valid_opt, act(t)), Some(dst_slots[t].ne(pad_slot_id)));
                            write(&state, &state_scale, dst_slots[t], r, &cur, cond);
                        } else if o.update_state && t == nt - 1 {
                            write(&state, &state_scale, state_batch, r, &cur, valid_opt);
                        }
                    }
                }
            };
            if rpw == step {
                run(first_row);
            } else {
                for_range(0, (rpw / step) as i32, 1, |i| run(first_row + i * step as i32));
            }
        };
        if o.varlen {
            if_(seq_len.gt(0), body);
        } else {
            body();
        }
    })
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

/// A test problem; every value is exactly representable in its dtype.
pub struct Problem {
    pub input: Dtype,
    pub state: StateDtype,
    pub opts: Options,
    /// Sequences.
    pub batch: usize,
    pub nheads: usize,
    pub dim: usize,
    pub dstate: usize,
    pub ngroups: usize,
    /// Slots in the state cache (`batch` unless there are indices).
    pub cache: usize,
    /// The slot of each sequence (with `opts.has_indices`), `[batch]` or `[batch, tokens]` with
    /// `num_accepted_tokens`; negative is `pad_slot_id`.
    pub indices: Vec<i32>,
    pub pad_slot_id: i32,
    /// The destination slots (with `opts.has_dst`), `[batch]` or `[batch, tokens]`.
    pub dst: Vec<i32>,
    /// Tokens of each sequence (with `opts.varlen`; `cu_seqlens` is their running sum).
    pub seqlens: Vec<usize>,
    /// Accepted tokens of each sequence (with `opts.has_accepted`).
    pub accepted: Vec<i32>,
    /// Slots of the intermediate cache (with `opts.intermediate`) and each sequence's slot.
    pub icache: usize,
    pub istate_indices: Vec<i32>,
    /// The stored state (the integers, for int16) and, for int16, the decode scales.
    pub state0: Vec<f32>,
    pub scale0: Vec<f32>,
    pub x: Vec<f32>,
    pub dt: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub z: Vec<f32>,
    /// The SM count the launch plan assumes.
    pub num_sms: usize,
    /// At most this many rows in flight per warp (overrides [`Plan`]'s choice downward).
    pub unroll_cap: usize,
}

/// What a launch left in its buffers, decoded to values (see [`Problem::decode`]).
pub struct Outputs {
    pub state: Vec<f32>,
    pub out: Vec<f32>,
    pub istate: Vec<f32>,
}

impl Problem {
    pub fn new(
        input: Dtype,
        state: StateDtype,
        opts: Options,
        (batch, nheads, dim, dstate, ngroups): (usize, usize, usize, usize, usize),
        seed: u64,
    ) -> Problem {
        assert!(nheads.is_multiple_of(ngroups));
        let nt = opts.tokens;
        assert!(!opts.has_accepted || opts.has_indices);
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut uni = move |lo: f32, hi: f32| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((rng >> 33) as f32 / (1u64 << 31) as f32)
        };
        // Source slots `0 .. batch * cols` (distinct); destinations come after them, so no block
        // writes a slot another reads.
        let cols = if opts.has_accepted { nt } else { 1 };
        let sources = batch * cols;
        let dst_cols = if nt > 1 { nt } else { 1 };
        let cache = if opts.has_indices || opts.has_dst { sources + batch * dst_cols + 3 } else { batch };
        let indices: Vec<i32> = if opts.has_indices {
            (0..sources).map(|i| if (i / cols + i % cols) % 5 == 3 { -1 } else { i as i32 }).collect()
        } else {
            vec![]
        };
        let dst: Vec<i32> = if opts.has_dst {
            (0..batch * dst_cols)
                .map(|i| match i % 4 {
                    3 => -1,
                    // In place, when a one-token sequence's own slot is known.
                    0 if nt == 1 && opts.has_indices && indices[i] >= 0 => indices[i],
                    _ => (sources + i) as i32,
                })
                .collect()
        } else {
            vec![]
        };
        let seqlens: Vec<usize> =
            if opts.varlen { (0..batch).map(|b| (b * 3 + 1) % (nt + 1)).collect() } else { vec![] };
        let accepted: Vec<i32> =
            if opts.has_accepted { (0..batch).map(|b| (b % (nt + 1)) as i32).collect() } else { vec![] };
        let icache = match (opts.intermediate, opts.istate_indices || !opts.has_indices) {
            (false, _) => 0,
            (true, true) => batch + 2,
            (true, false) => cache,
        };
        let istate_indices: Vec<i32> =
            if opts.istate_indices { (0..batch).map(|b| (batch + 1 - b) as i32).collect() } else { vec![] };
        let total = if opts.varlen { seqlens.iter().sum() } else { batch * nt };
        let mut v =
            |n: usize, lo: f32, hi: f32, r: &dyn Fn(f32) -> f32| (0..n).map(|_| r(uni(lo, hi))).collect::<Vec<f32>>();
        let (ri, rs) = (
            |x| input.round(x),
            |x: f32| if opts.int16 { x.round_ties_even().clamp(-32767.0, 32767.0) } else { state.round(x) },
        );
        let i16s = opts.int16;
        let state0 = v(cache * nheads * dim * dstate, -1.0, 1.0, &|x| rs(if i16s { x * 32767.0 } else { x }));
        let scale0 = if i16s { v(cache * nheads * dim, 0.5 / 32767.0, 1.0 / 32767.0, &|x| x) } else { vec![] };
        let x = v(total * nheads * dim, -1.0, 1.0, &ri);
        let dt = v(total * nheads, -1.5, 1.0, &|x| x);
        let a = v(nheads, -2.0, -0.1, &|x| x);
        let b = v(total * ngroups * dstate, -1.0, 1.0, &ri);
        let c = v(total * ngroups * dstate, -1.0, 1.0, &ri);
        let d = v(nheads, -1.0, 1.0, &|x| x);
        let dt_bias = v(nheads, -0.5, 0.5, &|x| x);
        let z = v(total * nheads * dim, -3.0, 3.0, &ri);
        Problem {
            input,
            state,
            opts,
            batch,
            nheads,
            dim,
            dstate,
            ngroups,
            cache,
            indices,
            pad_slot_id: -1,
            dst,
            seqlens,
            accepted,
            icache,
            istate_indices,
            state0,
            scale0,
            x,
            dt,
            a,
            b,
            c,
            d,
            dt_bias,
            z,
            num_sms: 22,
            // A store per token keeps more rows' state live: 4 rows spill (500 bytes, 1.7x
            // slower on the cache); 2 do not.
            unroll_cap: if nt > 1 && (opts.intermediate || opts.has_dst) { 2 } else { usize::MAX },
        }
    }

    /// A state-typed buffer of `v` (stored integers if int16).
    pub fn state_buffer(&self, v: &[f32]) -> KernelArg {
        if self.opts.int16 {
            KernelArg::buffer(&v.iter().map(|&x| x as i16).collect::<Vec<_>>())
        } else {
            self.state.buffer(v)
        }
    }

    /// A state buffer's values.
    pub fn read_state(&self, a: &KernelArg) -> Vec<f32> {
        if self.opts.int16 { a.to_vec::<i16>().into_iter().map(|x| x as f32).collect() } else { self.state.read(a) }
    }

    /// Tokens in all sequences.
    pub fn total_tokens(&self) -> usize {
        if self.opts.varlen { self.seqlens.iter().sum() } else { self.batch * self.opts.tokens }
    }

    /// `cu_seqlens`.
    pub fn cu_seqlens(&self) -> Vec<i32> {
        std::iter::once(0)
            .chain(self.seqlens.iter().scan(0, |acc, n| {
                *acc += *n as i32;
                Some(*acc)
            }))
            .collect()
    }

    fn istate_len(&self) -> usize {
        self.icache * self.opts.tokens * self.nheads * self.dim * self.dstate
    }

    pub fn plan(&self) -> Plan {
        let mut plan = Plan::new(self.batch, self.nheads, self.dim, self.opts.tokens, self.num_sms);
        while plan.unroll > self.unroll_cap.max(1) {
            plan.unroll /= 2;
        }
        plan
    }

    pub fn kernel(&self) -> Kernel {
        kernel(self.input, self.state, self.opts, self.dim, self.dstate, self.plan())
    }

    pub fn launch(&self) -> LaunchConfig {
        let p = self.plan();
        let tiles = (self.dim / p.rows_per_block) as u32;
        LaunchConfig::new([self.batch as u32, self.nheads as u32, tiles], [THREADS as u32, 1, 1])
    }

    /// The kernel's arguments (see [`kernel`]); the state is argument 0, the output 9, the
    /// intermediate states 14, the state scales 16 and the intermediate scales 17.
    pub fn args(&self) -> Vec<KernelArg> {
        assert!(
            self.cache.max(self.icache * self.opts.tokens) * self.nheads * self.dim * self.dstate < 1 << 31,
            "offsets are 32-bit"
        );
        let one = || vec![0.0f32];
        let opt = |on: bool, v: &Vec<f32>| if on { v.clone() } else { one() };
        let ints = |on: bool, v: Vec<i32>| KernelArg::i32(if on { v } else { vec![0] });
        let scaled = self.opts.int16;
        vec![
            self.state_buffer(&self.state0),
            self.input.buffer(&self.x),
            KernelArg::f32(self.dt.clone()),
            KernelArg::f32(self.a.clone()),
            self.input.buffer(&self.b),
            self.input.buffer(&self.c),
            KernelArg::f32(opt(self.opts.has_d, &self.d)),
            KernelArg::f32(opt(self.opts.has_dt_bias, &self.dt_bias)),
            self.input.buffer(&opt(self.opts.has_z, &self.z)),
            self.input.buffer(&vec![0.0; self.total_tokens() * self.nheads * self.dim]),
            ints(self.opts.has_indices, self.indices.clone()),
            ints(self.opts.has_dst, self.dst.clone()),
            ints(self.opts.varlen, self.cu_seqlens()),
            ints(self.opts.has_accepted, self.accepted.clone()),
            self.state_buffer(&vec![0.0; self.istate_len().max(1)]),
            ints(self.opts.istate_indices, self.istate_indices.clone()),
            KernelArg::f32(opt(scaled, &self.scale0)),
            KernelArg::f32(vec![
                0.0;
                if scaled { self.icache * self.opts.tokens * self.nheads * self.dim } else { 0 }.max(1)
            ]),
            KernelArg::ScalarI32(self.nheads as i32),
            KernelArg::ScalarI32(self.ngroups as i32),
            KernelArg::ScalarI32(self.pad_slot_id),
        ]
    }

    /// Values from stored integers and their per-row decode scales (identity unless int16).
    pub fn decode(&self, stored: &[f32], scales: &[f32]) -> Vec<f32> {
        if !self.opts.int16 {
            return stored.to_vec();
        }
        stored.iter().enumerate().map(|(i, v)| v * scales[i / self.dstate]).collect()
    }

    /// The new state, the output and the intermediate states in f64.
    pub fn reference(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (nh, dim, ds) = (self.nheads, self.dim, self.dstate);
        let nt = self.opts.tokens;
        let mut state: Vec<f64> = self.decode(&self.state0, &self.scale0).iter().map(|&v| v as f64).collect();
        let mut istate = vec![0.0f64; self.istate_len()];
        let cu = if self.opts.varlen { self.cu_seqlens() } else { vec![] };
        let mut out = vec![0.0f64; self.total_tokens() * nh * dim];
        #[allow(clippy::needless_range_loop)]
        for b in 0..self.batch {
            let (bos, len) = if self.opts.varlen { (cu[b] as usize, self.seqlens[b]) } else { (b * nt, nt) };
            if len == 0 {
                continue;
            }
            let init = if self.opts.has_accepted { (self.accepted[b] - 1).max(0) as usize } else { 0 };
            let cols = if self.opts.has_accepted { nt } else { 1 };
            let slot = if self.opts.has_indices { self.indices[b * cols + init] } else { b as i32 };
            let pad = self.opts.has_indices && slot == self.pad_slot_id;
            let icache = if self.opts.istate_indices { self.istate_indices[b] } else { slot };
            for h in 0..nh {
                let g = h / (nh / self.ngroups);
                for r in 0..dim {
                    let row = |s: i32| ((s.max(0) as usize * nh + h) * dim + r) * ds;
                    let mut cur: Vec<f64> = (0..ds).map(|n| if pad { 0.0 } else { state[row(slot) + n] }).collect();
                    for t in 0..len {
                        let tok = bos + t;
                        let mut dt = self.dt[tok * nh + h] as f64;
                        if self.opts.has_dt_bias {
                            dt += self.dt_bias[h] as f64;
                        }
                        if self.opts.dt_softplus && dt <= 20.0 {
                            dt = dt.exp().ln_1p();
                        }
                        let da = (self.a[h] as f64 * dt).exp();
                        let xv = self.x[(tok * nh + h) * dim + r] as f64;
                        let mut acc = if self.opts.has_d { self.d[h] as f64 * xv } else { 0.0 };
                        for (n, s) in cur.iter_mut().enumerate() {
                            let bv = self.b[(tok * self.ngroups + g) * ds + n] as f64;
                            let cv = self.c[(tok * self.ngroups + g) * ds + n] as f64;
                            *s = *s * da + bv * dt * xv;
                            acc += *s * cv;
                        }
                        if self.opts.has_z {
                            let zv = self.z[(tok * nh + h) * dim + r] as f64;
                            acc *= zv / (1.0 + (-zv).exp());
                        }
                        out[(tok * nh + h) * dim + r] = acc;
                        if pad {
                            continue;
                        }
                        if self.opts.intermediate {
                            let at = (icache as usize * nt + t) * nh * dim * ds + (h * dim + r) * ds;
                            istate[at..at + ds].copy_from_slice(&cur);
                        } else if self.opts.has_dst {
                            let dst = self.dst[b * if nt > 1 { nt } else { 1 } + t];
                            if dst != self.pad_slot_id && (nt > 1 || self.opts.update_state) {
                                state[row(dst)..row(dst) + ds].copy_from_slice(&cur);
                            }
                        } else if t == len - 1 && self.opts.update_state {
                            state[row(slot)..row(slot) + ds].copy_from_slice(&cur);
                        }
                    }
                }
            }
        }
        (state, out, istate)
    }

    /// Decode what the buffers hold after a launch: the raw `state`, `state_scale`, `out`,
    /// `intermediate_states` and `intermediate_state_scales` values.
    pub fn outputs(&self, state: &[f32], scales: &[f32], out: &[f32], istate: &[f32], iscales: &[f32]) -> Outputs {
        Outputs {
            state: self.decode(state, scales),
            out: out.to_vec(),
            istate: self.decode(istate, iscales)[..self.istate_len()].to_vec(),
        }
    }

    /// Check the new state, the output and the intermediate states against the reference
    /// (absolute plus relative); returns the largest scaled `(state, out)` errors (the state's
    /// including the intermediate states).
    pub fn check(&self, got: &Outputs) -> (f64, f64) {
        let (rs, ro, ri) = self.reference();
        let scaled = |got: &[f32], want: &[f64], tol: f64, what: &str| {
            assert_eq!(got.len(), want.len(), "{what}");
            let mut worst = 0.0f64;
            for (i, (&g, &r)) in got.iter().zip(want).enumerate() {
                let e = (g as f64 - r).abs() / (1.0 + r.abs());
                assert!(e <= tol, "{what} {i}: {g} against {r} ({:?})", self.opts);
                worst = worst.max(e);
            }
            worst
        };
        let st = if self.opts.int16 { 1.5e-4 } else { self.state.tolerance() };
        let ts = st + 2e-5;
        let to = self.input.tolerance() + 2e-4 + st;
        let es = scaled(&got.state, &rs, ts, "state").max(scaled(&got.istate, &ri, ts, "intermediate state"));
        (es, scaled(&got.out, &ro, to, "out"))
    }

    /// Bytes moved: the state read and written, `x`, `z`, `out`, `B`, `C` (a sequence's state
    /// is read and written once; the cache or destination writes add one write per token).
    pub fn bytes(&self) -> f64 {
        let row = self.nheads * self.dim * self.dstate * if self.opts.int16 { 2 } else { self.state.bytes() };
        let nt = self.opts.tokens;
        let writes = if self.opts.intermediate || (self.opts.has_dst && nt > 1) { nt } else { 1 };
        let state = (1 + writes) * self.batch * row;
        let io = (nt * self.batch * self.nheads * self.dim * (2 + self.opts.has_z as usize)
            + 2 * nt * self.batch * self.ngroups * self.dstate)
            * 2;
        (state + io) as f64
    }
}

/// Simulate the kernel on `p` and check it against the f64 reference; returns the largest
/// scaled `(state, out)` errors.
pub fn simulate(p: &Problem) -> (f64, f64) {
    let k = p.kernel();
    let mut args = p.args();
    k.simulate(p.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let scales = |a: &KernelArg| if p.opts.int16 { a.to_vec::<f32>() } else { vec![] };
    p.check(&p.outputs(
        &p.read_state(&args[0]),
        &scales(&args[16]),
        &p.input.read(&args[9]),
        &p.read_state(&args[14]),
        &scales(&args[17]),
    ))
}

/// Running on the GPU (`--features cuda`).
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::collections::HashMap;
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{Plan, Problem};

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
            let read = |i: usize| -> Result<Vec<f32>, Box<dyn Error>> {
                if p.opts.int16 {
                    return Ok(bufs[i].to_vec::<i16>()?.into_iter().map(|x| x as f32).collect());
                }
                Ok(match p.state {
                    super::StateDtype::F16 => bufs[i].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
                    super::StateDtype::BF16 => {
                        bufs[i].to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
                    }
                    super::StateDtype::F32 => bufs[i].to_vec::<f32>()?,
                })
            };
            let out: Vec<f32> = match p.input {
                super::Dtype::F16 => bufs[9].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
                super::Dtype::BF16 => {
                    bufs[9].to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
                }
            };
            let scales = |i: usize| -> Result<Vec<f32>, Box<dyn Error>> {
                Ok(if p.opts.int16 { bufs[i].to_vec::<f32>()? } else { vec![] })
            };
            err = p.check(&p.outputs(&read(0)?, &scales(16)?, &out, &read(14)?, &scales(17)?));
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
