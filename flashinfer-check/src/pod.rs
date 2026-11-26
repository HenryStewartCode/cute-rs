// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::pod` on the inputs of `reference/cases_pod.py`: a prefill and a paged decode batch in one
//! launch.

use flashinfer::pod::Problem;
use flashinfer::pod::gpu::{Kernels, launch};
use flashinfer::prefill::Problem as Half;

use crate::case::{Case, Gpu, Res, Tensor};
use cute::prelude::*;

fn f16s(t: &Tensor) -> Vec<F16> {
    t.f32s().into_iter().map(F16::from_f32).collect()
}

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let (hq, hkv, ps) = (case.int("hq"), case.int("hkv"), case.int("page_size"));
    let (qo, kv) = (case.int("qo_len"), case.int("kv_len"));
    let prefill = Half::from_ragged(
        &[qo],
        &[kv],
        hkv,
        hq / hkv,
        case.flag("causal"),
        f16s(&case.tensor("q_p")),
        f16s(&case.tensor("k_p")),
        f16s(&case.tensor("v_p")),
    );
    // The decode half is FlashInfer's own page table and cache: `[pages, 2, page, kv heads, head dim]`
    // holds K then V of each page, which the port reads as two NHD caches.
    let decode_kv = case.ints("decode_kv");
    let mut decode = Half::new(&vec![1; decode_kv.len()], &decode_kv, ps, hkv, hq / hkv, false, 0);
    let cache = f16s(&case.tensor("cache"));
    let page = ps * hkv * 128;
    let pages = cache.len() / (2 * page);
    let (mut kc, mut vc) = (Vec::with_capacity(pages * page), Vec::with_capacity(pages * page));
    for pg in 0..pages {
        kc.extend_from_slice(&cache[2 * pg * page..(2 * pg + 1) * page]);
        vc.extend_from_slice(&cache[(2 * pg + 1) * page..(2 * pg + 2) * page]);
    }
    (decode.q, decode.k_cache, decode.v_cache) = (f16s(&case.tensor("q_d")), kc, vc);
    decode.kv_indptr = case.tensor("indptr").i32s();
    decode.kv_indices = case.tensor("indices").i32s();
    decode.kv_last_page_len = case.tensor("last").i32s();
    let p = Problem::from_parts(prefill, decode);
    let ks = Kernels::new(&gpu.ctx, gpu.tc.clone())?;
    let plans = p.plans(ks.max_grid);
    let (out, _) = launch(&ks, &p, &plans, true, None)?;
    let ((o_p, _), (o_d, _)) = out.ok_or("no output")?;
    case.write_f32("o_p", &o_p)?;
    case.write_f32("o_d", &o_d)
}
