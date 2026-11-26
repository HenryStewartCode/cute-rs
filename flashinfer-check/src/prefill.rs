// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::prefill` on the inputs of `reference/cases_prefill.py`: batch prefill over a paged
//! cache, ragged batch prefill and single prefill, with their variants and FP8 caches.

use flashinfer::prefill::gpu::{Kernels, launch};
use flashinfer::prefill::{KvDtype, Problem};

use crate::case::{Case, Gpu, Res, Tensor};
use cute::prelude::*;

fn f16s(t: &Tensor) -> Vec<F16> {
    t.f32s().into_iter().map(F16::from_f32).collect()
}

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let (hd, hq, hkv) = (case.int("head_dim"), case.int("hq"), case.int("hkv"));
    let causal = case.flag("causal");
    let kv = match case.str("kv") {
        "f16" => KvDtype::F16,
        "e4m3" => KvDtype::E4M3,
        "e5m2" => KvDtype::E5M2,
        other => return Err(format!("kv {other}").into()),
    };
    let (q, k, v) = (case.tensor("q"), case.tensor("k"), case.tensor("v"));
    let mut p = match case.str("kind") {
        "paged" => {
            let (qo_indptr, kv_indptr) = (case.tensor("qo_indptr").i32s(), case.tensor("kv_indptr").i32s());
            let (qo_lens, kv_lens) = (case.ints("qo_lens"), case.ints("kv_lens"));
            let mut p = Problem::new_hd(&qo_lens, &kv_lens, case.int("page_size"), hkv, hq / hkv, causal, hd, 0);
            p.q = f16s(&q);
            // The page table, lengths and cache contents are FlashInfer's own.
            (p.qo_indptr, p.kv_indptr) = (qo_indptr, kv_indptr);
            p.kv_indices = case.tensor("kv_indices").i32s();
            p.kv_last_page_len = case.tensor("last").i32s();
            p.k_cache = f16s(&k);
            p.v_cache = f16s(&v);
            p
        }
        "ragged" => {
            let (qo_lens, kv_lens) = (case.ints("qo_lens"), case.ints("kv_lens"));
            Problem::from_ragged(&qo_lens, &kv_lens, hkv, hq / hkv, causal, f16s(&q), f16s(&k), f16s(&v))
        }
        "single" => Problem::from_ragged(
            &case.ints("qo_lens"),
            &case.ints("kv_lens"),
            hkv,
            hq / hkv,
            causal,
            f16s(&q),
            f16s(&k),
            f16s(&v),
        ),
        other => return Err(format!("kind {other}").into()),
    };
    if kv != KvDtype::F16 {
        // The stored FP8 bytes go to the kernel as they are (the host vectors above only give the sizes); the
        // scales are FlashInfer's `k_scale`, `v_scale`.
        (p.kv_dtype, p.k_fp8, p.v_fp8) = (kv, k.bytes.clone(), v.bytes.clone());
        (p.k_scale, p.v_scale) = (case.float("k_scale") as f32, case.float("v_scale") as f32);
    }
    if case.str("dtype") == "bf16" {
        p = p.bf16();
    }
    match case.str("variant") {
        "none" => {}
        "softcap" => p = p.with_soft_cap(case.float("cap") as f32),
        "window" => p = p.with_window(case.i64("window") as i32),
        "alibi" => p = p.with_alibi(),
        "mask" => p = p.with_mask_bits(case.tensor("mask").i64s().into_iter().map(|b| b != 0).collect()),
        other => return Err(format!("variant {other}").into()),
    }
    let ks = Kernels::new(&gpu.ctx, gpu.tc.clone())?;
    let plan = if case.str("kind") == "single" { p.single_plan(ks.max_grid) } else { p.plan(ks.max_grid) };
    let (out, _) = launch(&ks, &p, &plan, true, None)?;
    let (o, lse) = out.ok_or("no output")?;
    case.write_f32("o", &o)?;
    case.write_f32("lse", &lse)
}
