// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::activation` on the inputs of `reference/cases_activation.py`.

use flashinfer::activation::{Act, Problem};
use flashinfer::norm::Dtype;

use crate::case::{Case, Gpu, Res, launch};

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let dtype = match case.str("dtype") {
        "f16" => Dtype::F16,
        "bf16" => Dtype::BF16,
        other => return Err(format!("dtype {other}").into()),
    };
    let act = match case.str("act") {
        "silu" => Act::Silu,
        "gelu" => Act::Gelu,
        "gelu_tanh" => Act::GeluTanh,
        other => return Err(format!("act {other}").into()),
    };
    let mut p = Problem::new(dtype, act, case.int("tokens"), case.int("d"), 0);
    p.input = case.tensor("input").f32s();
    let args = launch(gpu, &p.kernel(), p.launch(), p.args())?;
    case.write("out", &cute::host::arg_bytes(&args[1]))
}
