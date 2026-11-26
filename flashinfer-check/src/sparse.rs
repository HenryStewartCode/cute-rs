// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::sparse` on the inputs of `reference/cases_sparse.py`: block-sparse attention in BSR
//! form (with and without a custom mask) and with variable blocks.

use flashinfer::sparse::Problem;
use flashinfer::sparse::gpu::{Kernels, launch};

use crate::case::{Case, Gpu, Res, Tensor};
use cute::prelude::*;

fn f16s(t: &Tensor) -> Vec<F16> {
    t.f32s().into_iter().map(F16::from_f32).collect()
}

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let (hq, hkv, hd) = (case.int("hq"), case.int("hkv"), case.int("head_dim"));
    let qkv = (f16s(&case.tensor("q")), f16s(&case.tensor("k")), f16s(&case.tensor("v")));
    let p = match case.str("kind") {
        "bsr" => {
            let mask: Option<Vec<bool>> =
                case.flag("mask").then(|| case.tensor("mask_bits").i64s().into_iter().map(|b| b != 0).collect());
            Problem::from_bsr(
                (case.int("M"), case.int("N"), case.int("R"), case.int("C")),
                &case.tensor("indptr").i32s(),
                &case.tensor("indices").i32s(),
                hq,
                hkv,
                hd,
                qkv,
                mask.as_deref(),
            )
        }
        "variable" => {
            let mask: Vec<bool> = case.tensor("block_mask").i64s().into_iter().map(|b| b != 0).collect();
            Problem::from_variable_blocks(
                &mask,
                &case.tensor("row_sz").usizes(),
                &case.tensor("col_sz").usizes(),
                hq,
                hkv,
                hd,
                qkv,
            )
        }
        other => return Err(format!("kind {other}").into()),
    };
    let ks = Kernels::new(&gpu.ctx, gpu.tc.clone())?;
    let plan = p.plan(ks.max_grid);
    let (out, _) = launch(&ks, &p, &plan, true, None)?;
    let (o, lse) = out.ok_or("no output")?;
    case.write_f32("o", &p.output_layout(&o))?;
    // Variable blocks: the kernel's rows are (KV head, query token, group member).
    let lse = match p.hnd {
        None => lse,
        Some((n_kv, qo)) => {
            let g = p.group;
            let mut out = vec![0f32; lse.len()];
            for h in 0..n_kv {
                for gi in 0..g {
                    for t in 0..qo {
                        out[(h * g + gi) * qo + t] = lse[(h * qo + t) * g + gi];
                    }
                }
            }
            out
        }
    };
    case.write_f32("lse", &lse)
}
