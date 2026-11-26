// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::mamba` on the inputs of `reference/cases_mamba.py`.
//!
//! The port's `Problem` makes its own data; every field is replaced by the case's tensors, so
//! the kernel sees exactly what FlashInfer saw. The pool, the intermediate cache and the scales
//! start from the file's contents and are written back whole.

use flashinfer::mamba::{Dtype, Options, Problem, StateDtype};

use crate::case::{Case, Gpu, Res, launch};

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let input = match case.str("inp") {
        "f16" => Dtype::F16,
        "bf16" => Dtype::BF16,
        other => return Err(format!("input dtype {other}").into()),
    };
    let int16 = case.flag("int16");
    let state = match case.str("st") {
        "f16" => StateDtype::F16,
        "bf16" => StateDtype::BF16,
        "f32" => StateDtype::F32,
        other => return Err(format!("state dtype {other}").into()),
    };
    let istate = case.str("istate").to_string();
    let varlen = case.flag("varlen");
    let opts = Options {
        has_z: case.flag("z"),
        has_d: case.flag("d"),
        has_dt_bias: case.flag("bias"),
        has_indices: case.flag("idx"),
        dt_softplus: case.flag("softplus"),
        update_state: !case.flag("no_update"),
        tokens: case.int("tokens"),
        has_dst: case.flag("dst"),
        varlen,
        has_accepted: case.flag("accepted"),
        intermediate: !istate.is_empty(),
        istate_indices: istate == "idx",
        int16,
    };
    let shape = (case.int("batch"), case.int("nheads"), case.int("dim"), case.int("dstate"), case.int("ngroups"));
    let mut p = Problem::new(input, state, opts, shape, 0);
    p.pad_slot_id = case.i64("pad") as i32;
    let st = case.tensor("state");
    p.cache = st.shape[0];
    p.state0 = st.i64s_or_f32s();
    if int16 {
        p.scale0 = case.tensor("state_scale").f32s();
    }
    p.x = case.tensor("x").f32s();
    p.dt = case.tensor("dt").f32s();
    p.a = case.tensor("A").f32s();
    p.b = case.tensor("B").f32s();
    p.c = case.tensor("C").f32s();
    if opts.has_d {
        p.d = case.tensor("D").f32s();
    }
    if opts.has_dt_bias {
        p.dt_bias = case.tensor("dt_bias").f32s();
    }
    if opts.has_z {
        p.z = case.tensor("z").f32s();
    }
    if opts.has_indices {
        p.indices = case.tensor("indices").i32s();
    }
    if opts.has_dst {
        p.dst = case.tensor("dst").i32s();
    }
    if opts.has_accepted {
        p.accepted = case.tensor("accepted").i32s();
    }
    if varlen {
        let cu = case.tensor("cu_seqlens").usizes();
        p.seqlens = cu.windows(2).map(|w| w[1] - w[0]).collect();
    }
    let mut args_override = Vec::new();
    if opts.intermediate {
        let ib = case.tensor("istate");
        p.icache = ib.shape[0];
        args_override.push((14, p.state_buffer(&ib.i64s_or_f32s())));
        if opts.istate_indices {
            p.istate_indices = case.tensor("istate_indices").i32s();
        }
        if int16 {
            args_override.push((17, cute::prelude::KernelArg::f32(case.tensor("istate_scale").f32s())));
        }
    }
    let mut args = p.args();
    for (i, a) in args_override {
        args[i] = a;
    }
    let args = launch(gpu, &p.kernel(), p.launch(), args)?;
    let bytes = |i: usize| cute::host::arg_bytes(&args[i]);
    case.write("out", &bytes(9))?;
    case.write("state", &bytes(0))?;
    if int16 {
        case.write("state_scale", &bytes(16))?;
    }
    if opts.intermediate {
        case.write("istate", &bytes(14))?;
        if int16 {
            case.write("istate_scale", &bytes(17))?;
        }
    }
    Ok(())
}

/// A state tensor's values as Float32: the stored integers for an int16 state.
trait Values {
    fn i64s_or_f32s(&self) -> Vec<f32>;
}

impl Values for crate::case::Tensor {
    fn i64s_or_f32s(&self) -> Vec<f32> {
        if self.dtype == "int16" { self.i64s().into_iter().map(|v| v as f32).collect() } else { self.f32s() }
    }
}
