// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::cascade` on the inputs of `reference/cases_cascade.py`.

use flashinfer::cascade::{Op, Problem};
use flashinfer::norm::Dtype;

use crate::case::{Case, Gpu, Res, launch};

/// The two states of `merge_state` as a `Problem`'s `[seq, 2, heads, head_dim]` and `[seq, 2, heads]`.
fn interleave(a: &[f32], b: &[f32], chunk: usize) -> Vec<f32> {
    a.chunks(chunk).zip(b.chunks(chunk)).flat_map(|(x, y)| x.iter().chain(y)).copied().collect()
}

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let dtype = match case.str("dtype") {
        "f16" => Dtype::F16,
        "bf16" => Dtype::BF16,
        other => return Err(format!("dtype {other}").into()),
    };
    let (seq, heads, hd) = (case.int("seq"), case.int("heads"), case.int("head_dim"));
    let op = case.str("op");
    let sets = if op == "merge_states" { case.int("sets") } else { 2 };
    let mut p = Problem::new(dtype, seq, heads, hd, sets, 0);
    let kind = match op {
        "merge_state" => Op::MergeState,
        "merge_state_in_place" => Op::InPlace { masked: false },
        "merge_state_in_place_mask" => Op::InPlace { masked: true },
        "merge_states" => Op::MergeStates,
        other => return Err(format!("op {other}").into()),
    };
    match kind {
        Op::MergeState => {
            p.v = interleave(&case.tensor("v_a").f32s(), &case.tensor("v_b").f32s(), heads * hd);
            p.s = interleave(&case.tensor("s_a").f32s(), &case.tensor("s_b").f32s(), heads);
        }
        Op::InPlace { masked } => {
            p.v = interleave(&case.tensor("v").f32s(), &case.tensor("v_other").f32s(), heads * hd);
            p.s = interleave(&case.tensor("s").f32s(), &case.tensor("s_other").f32s(), heads);
            if masked {
                p.mask = case.tensor("mask").i64s().into_iter().map(|m| m as u8).collect();
            }
        }
        Op::MergeStates => {
            p.v = case.tensor("v").f32s();
            p.s = case.tensor("s").f32s();
        }
    }
    let (args, vi, si) = p.args(kind);
    let args = launch(gpu, &p.kernel(kind), p.launch(kind), args)?;
    case.write("v", &cute::host::arg_bytes(&args[vi]))?;
    case.write("s", &cute::host::arg_bytes(&args[si]))
}
