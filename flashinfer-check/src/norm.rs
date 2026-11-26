// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::norm` on the inputs of `reference/cases_norm.py`.

use cute::prelude::*;
use flashinfer::norm::{Config, Dtype, Problem, Variant, qk_config};

use crate::case::{Case, Gpu, Res, launch};

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let dtype = match case.str("dtype") {
        "f16" => Dtype::F16,
        "bf16" => Dtype::BF16,
        other => return Err(format!("dtype {other}").into()),
    };
    let variant = match case.str("variant") {
        "rms" => Variant::Rms,
        "gemma_rms" => Variant::GemmaRms,
        "fused_add" => Variant::FusedAdd,
        "gemma_fused_add" => Variant::GemmaFusedAdd,
        "layernorm" => Variant::LayerNorm,
        "rms_quant" => Variant::RmsQuant,
        "fused_add_quant" => Variant::FusedAddQuant,
        "qk_rms" => Variant::QkRms,
        other => return Err(format!("variant {other}").into()),
    };
    let (batch, d) = (case.int("batch"), case.int("d"));
    let mut p = if variant == Variant::QkRms {
        let mut p = Problem::qk(dtype, batch, case.int("heads"), d, 0);
        p.config = qk_config(d);
        p
    } else {
        let mut p = Problem::new(dtype, batch, d, 0);
        p.config = Config::choose(variant, batch, d);
        p
    };
    p.eps = case.float("eps") as f32;
    p.input = case.tensor("input").f32s();
    if case.has("residual") {
        p.residual = case.tensor("residual").f32s();
    }
    if case.has("weight") {
        p.weight = case.tensor("weight").f32s();
    }
    if let Some(scale) = case.get("scale") {
        p.scale = scale.parse()?;
    }
    let mut args = p.args(variant);
    if variant == Variant::LayerNorm {
        // `input, gamma, beta, output, ...`: the problem makes its own gamma and beta.
        args[1] = KernelArg::buffer(&case.tensor("gamma").f32s());
        args[2] = KernelArg::buffer(&case.tensor("beta").f32s());
    }
    let args = launch(gpu, &p.config.kernel(dtype, variant), p.launch(variant), args)?;
    let bytes = |i: usize| cute::host::arg_bytes(&args[i]);
    match variant {
        Variant::Rms | Variant::GemmaRms | Variant::QkRms | Variant::RmsQuant => case.write("out", &bytes(2)),
        Variant::LayerNorm => case.write("out", &bytes(3)),
        Variant::FusedAdd | Variant::GemmaFusedAdd => {
            case.write("input", &bytes(0))?;
            case.write("residual", &bytes(1))
        }
        Variant::FusedAddQuant => {
            case.write("out", &bytes(3))?;
            case.write("residual", &bytes(1))
        }
    }
}
