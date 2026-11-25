// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's ragged-KV batch prefill (`BatchPrefillWithRaggedKVCacheWrapper`) and
//! `single_prefill_with_kv_cache` ([`flashinfer::prefill`]): simulated on small problems,
//! then, with `--features cuda` and `GPU=1`, checked on the GPU against the f64 reference
//! and timed from CUDA graphs on the shapes of `prefill_ragged_flashinfer_bench.py`: the
//! `prefill` example's batch {1, 8} x qo_len = kv_len {512, 2048, 8192}, and one request
//! of qo_len = kv_len {1024, 8192, 32768}; causal, 32 query / 8 KV heads, f16, NHD.
//!
//! `cargo run -p cute-flashinfer --example prefill_ragged`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example prefill_ragged`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
#[cfg(feature = "cuda")]
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::prefill::gpu::{Kernels, run, run_plan, single_prefill_with_kv_cache as gpu_single};
use flashinfer::prefill::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let ks = Kernels::new(&ctx, tc)?;
    println!("device: {}, planner grid: {} CTAs", ctx.name(), ks.max_grid);
    let ok = |eo: f32, el: f32| if eo < 2e-3 && el < 1e-3 { Ok(()) } else { Err("out of tolerance") };
    for (qo, kv, heads, group, causal) in [
        (vec![3usize, 1, 4], vec![70usize, 9, 130], 2usize, 4usize, true),
        (vec![9, 13, 30], vec![37, 13, 300], 8, 4, false),
        (vec![150, 70, 333], vec![160, 75, 400], 8, 1, true),
        (vec![256, 100], vec![256, 1000], 8, 4, true),
    ] {
        let p = Problem::new_ragged(&qo, &kv, heads, group, causal, 11);
        let (plan, eo, el, _) = run(&ks, &p, true, None)?;
        println!(
            "ragged qo {qo:?} kv {kv:?} heads {}/{heads} causal {causal}: tile {} split {}: |o| {eo:.2e}, |lse| {el:.2e}",
            heads * group,
            plan.tile.cta_tile_q,
            plan.split
        );
        ok(eo, el)?;
    }
    for (qo, kv, causal) in [(1usize, 3000usize, false), (77, 1500, true), (600, 600, true)] {
        let p = Problem::new_ragged(&[qo], &[kv], 8, 4, causal, 13);
        let (o, lse) = gpu_single(&ks, p.q.clone(), p.k_cache.clone(), p.v_cache.clone(), 32, 8, causal)?;
        let (want_o, want_lse) = p.reference();
        let (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
        println!(
            "single qo {qo} kv {kv} causal {causal}: split {}: |o| {eo:.2e}, |lse| {el:.2e}",
            p.single_plan(ks.max_grid).split
        );
        ok(eo, el)?;
    }
    println!("api     batch  qo=kv  tile  split  ctas     gpu(us)   TFLOPS");
    let row = |api: &str, p: &Problem, plan: &Plan, t: f64| {
        println!(
            "{api:6}  {:5}  {:5}  {:4}  {:5}  {:4}  {t:10.1}  {:7.2}",
            p.qo_lens.len(),
            p.qo_lens[0],
            plan.tile.cta_tile_q,
            plan.split,
            plan.num_ctas(),
            p.flops() / (t * 1e-6) / 1e12
        );
    };
    for batch in [1usize, 8] {
        for n in [512usize, 2048, 8192] {
            let p = Problem::new_ragged(&vec![n; batch], &vec![n; batch], 8, 4, true, 3);
            let iterations = if batch * n * n <= 8 * 2048 * 2048 { 100 } else { 20 };
            let (plan, _, _, t) = run(&ks, &p, false, Some(iterations))?;
            row("ragged", &p, &plan, t.unwrap());
        }
    }
    for n in [1024usize, 8192, 32768] {
        let p = Problem::new_ragged(&[n], &[n], 8, 4, true, 3);
        let plan = p.single_plan(ks.max_grid);
        let iterations = [100, 20, 5][[1024, 8192, 32768].iter().position(|&x| x == n).unwrap()];
        let (_, _, t) = run_plan(&ks, &p, &plan, false, Some(iterations))?;
        row("single", &p, &plan, t.unwrap());
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    for (qo, kv, heads, group) in
        [(vec![3usize, 1, 4], vec![70usize, 9, 130], 2usize, 4usize), (vec![9, 13], vec![37, 13], 1, 4)]
    {
        for causal in [false, true] {
            let p = Problem::new_ragged(&qo, &kv, heads, group, causal, 7);
            for grid in [heads, 1000] {
                let (plan, eo, el) = simulate(&p, grid, 16);
                println!(
                    "ragged qo {qo:?} kv {kv:?} heads {}/{heads} causal {causal}: tile {} split {}: |o| {eo:.2e}, |lse| {el:.2e}",
                    heads * group,
                    plan.tile.cta_tile_q,
                    plan.split
                );
                assert!(eo < 2e-3 && el < 1e-3, "out of tolerance");
            }
        }
    }
    for causal in [false, true] {
        let p = Problem::new_ragged(&[5], &[300], 2, 4, causal, 9);
        let (o, lse) =
            single_prefill_with_kv_cache(p.q.clone(), p.k_cache.clone(), p.v_cache.clone(), 8, 2, causal, 1000);
        let (want_o, want_lse) = p.reference();
        let (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
        println!(
            "single qo 5 kv 300 causal {causal} (split {}): |o| {eo:.2e}, |lse| {el:.2e}",
            p.single_plan(1000).split
        );
        assert!(eo < 2e-3 && el < 1e-3, "out of tolerance");
    }
}
