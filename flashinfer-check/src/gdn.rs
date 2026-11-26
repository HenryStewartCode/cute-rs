// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::gdn` on the inputs of `reference/cases_gdn.py`: the decode kernels (state `[V, K]` and
//! `[K, V]`, one token or several, pools, the intermediate-state cache and the per-token scatter) and
//! the chunked prefill.
//!
//! The port's problem structs make their own data; every field is replaced by the case's tensors, so
//! the kernel sees exactly what FlashInfer saw. State pools and caches start from the file's contents
//! and are written back whole.

use flashinfer::gdn::{Dtype, Options, PrefillOptions, PrefillProblem, Problem, StateDtype};

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
    match case.str("fn") {
        "prefill" => prefill(gpu, case, input, state),
        _ => decode(gpu, case, input, state),
    }
}

fn decode(gpu: &Gpu, case: &Case, input: Dtype, state: StateDtype) -> Res<()> {
    let fun = case.str("fn");
    let tokens = case.int("tokens");
    let cache_steps = case.int("cache");
    let scatter = case.flag("scatter");
    // `gated_delta_rule_mtp` always has read indices; so does a pool.
    let has_indices = case.flag("pool") || fun == "mtp";
    let opts = Options {
        l2norm: case.flag("l2norm"),
        has_indices,
        has_out_indices: case.flag("out_idx"),
        tokens,
        update_state: !case.flag("no_update"),
        cache_steps,
        kv_layout: fun == "kv",
        per_token_slots: scatter,
    };
    let shape = (case.int("batch"), case.int("hq"), case.int("hv"), 128, 128);
    let mut p = Problem::new(input, state, opts, shape, 0);
    p.scale = case.float("scale") as f32;
    let st = case.tensor("state");
    p.slots = st.shape[0];
    p.state0 = st.f32s();
    p.q = case.tensor("q").f32s();
    p.kk = case.tensor("k").f32s();
    p.vv = case.tensor("v").f32s();
    p.a = case.tensor("a").f32s();
    p.b = case.tensor("b").f32s();
    p.a_log = case.tensor("A_log").f32s();
    p.dt_bias = case.tensor("dt_bias").f32s();
    if has_indices {
        p.indices = case.tensor("indices").i32s();
    }
    if opts.has_out_indices {
        p.out_indices = case.tensor("out_indices").i32s();
    }
    if scatter {
        p.ssm_indices = case.tensor("ssm_indices").i32s();
    }
    let mut args = p.args();
    if cache_steps > 0 {
        // The cache starts from FlashInfer's contents: steps it does not write stay as they were.
        args[11] = state.buffer(&case.tensor("cache").f32s());
    }
    let args = launch(gpu, &p.kernel(), p.launch(), args)?;
    case.write("out", &cute::host::arg_bytes(&args[8]))?;
    case.write("state", &cute::host::arg_bytes(&args[0]))?;
    if cache_steps > 0 {
        case.write("cache", &cute::host::arg_bytes(&args[11]))?;
    }
    Ok(())
}

fn prefill(gpu: &Gpu, case: &Case, input: Dtype, state: StateDtype) -> Res<()> {
    let pool = case.flag("pool");
    let opts = PrefillOptions {
        l2norm: case.flag("l2norm"),
        has_initial_state: case.flag("initial"),
        output_final_state: true,
        has_indices: pool,
        has_out_indices: case.flag("out_idx"),
    };
    let cu = case.tensor("cu_seqlens").usizes();
    let lens: Vec<usize> = cu.windows(2).map(|w| w[1] - w[0]).collect();
    let shape = (case.int("hq"), case.int("hv"), 128);
    let mut p = PrefillProblem::new(input, state, opts, &lens, shape, 0);
    p.scale = case.float("scale") as f32;
    let out0 = case.tensor("state_out");
    p.slots = out0.shape[0];
    if opts.has_initial_state {
        p.state0 = case.tensor("state").f32s();
    }
    p.q = case.tensor("q").f32s();
    p.kk = case.tensor("k").f32s();
    p.vv = case.tensor("v").f32s();
    p.g = case.tensor("g").f32s();
    p.beta = case.tensor("beta").f32s();
    if pool {
        p.indices = case.tensor("indices").i32s();
    }
    if opts.has_out_indices {
        p.out_indices = case.tensor("out_indices").i32s();
    }
    let mut args = p.args();
    // The final-state buffer starts from the file's contents, so rows nobody writes are checked too.
    args[7] = state.buffer(&out0.f32s());
    let args = launch(gpu, &p.kernel(), p.launch(), args)?;
    case.write("out", &cute::host::arg_bytes(&args[8]))?;
    case.write("state_out", &cute::host::arg_bytes(&args[7]))
}
