// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::kda` on the inputs of `reference/cases_kda.py`.
//!
//! The port's `Problem` makes its own data; every field is replaced by the case's tensors, so
//! the kernel sees exactly what FlashInfer saw. The state pool is written back whole.

use flashinfer::kda::{Dtype, Gate, Options, Problem, StateDtype};

use crate::case::{Case, Gpu, Res, launch};

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let input = match case.str("inp") {
        "f16" => Dtype::F16,
        "bf16" => Dtype::BF16,
        other => return Err(format!("input dtype {other}").into()),
    };
    let state = match case.str("st") {
        "f16" => StateDtype::F16,
        "bf16" => StateDtype::BF16,
        "f32" => StateDtype::F32,
        other => return Err(format!("state dtype {other}").into()),
    };
    let gate = match case.str("gate") {
        "pre" => Gate::Precomputed,
        "softplus" => Gate::Softplus,
        "lower_bound" => Gate::LowerBound,
        other => return Err(format!("gate {other}").into()),
    };
    let tokens = case.int("tokens");
    let has_idx = case.flag("idx");
    let varlen = case.flag("varlen");
    // Several tokens without given indices: FlashInfer's checkpoint slots are `b * T + t`.
    let implicit = tokens > 1 && !case.has("indices");
    let per_token = case.flag("per_token") || implicit;
    let opts = Options {
        l2norm: case.flag("l2norm"),
        gate,
        has_dt_bias: case.has("dt_bias"),
        beta_is_logit: case.flag("logit"),
        has_indices: has_idx || per_token,
        has_out_indices: false,
        tokens,
        update_state: true,
        per_token_slots: per_token,
        varlen,
        has_accepted: case.flag("accepted"),
    };
    let (batch, hq, hv) = (case.int("batch"), case.int("hq"), case.int("hv"));
    let shape = (batch, hq, hv, 128, 128);
    let mut p = if varlen {
        let cu = case.tensor("cu_seqlens").usizes();
        let lens: Vec<usize> = cu.windows(2).map(|w| w[1] - w[0]).collect();
        Problem::new_varlen(input, state, opts, shape, &lens, 0)
    } else {
        Problem::new(input, state, opts, shape, 0)
    };
    p.scale = case.float("scale") as f32;
    p.lower_bound = case.float("lower_bound") as f32;
    let st = case.tensor("state");
    p.slots = st.shape[0];
    p.state0 = st.f32s();
    p.q = case.tensor("q").f32s();
    p.kk = case.tensor("k").f32s();
    p.vv = case.tensor("v").f32s();
    p.g = case.tensor("g").f32s();
    p.beta = case.tensor("beta").f32s();
    if case.has("A_log") {
        p.a_log = case.tensor("A_log").f32s();
    }
    if case.has("dt_bias") {
        p.dt_bias = case.tensor("dt_bias").f32s();
    }
    if case.has("indices") {
        p.indices = case.tensor("indices").i32s();
    } else if implicit {
        p.indices = (0..(batch * tokens) as i32).collect();
    }
    if opts.has_accepted {
        p.accepted = case.tensor("accepted").i32s();
    }
    let args = launch(gpu, &p.kernel(), p.launch(), p.args())?;
    case.write("out", &cute::host::arg_bytes(&args[8]))?;
    case.write("state", &cute::host::arg_bytes(&args[0]))
}
