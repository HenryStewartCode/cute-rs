// SPDX-License-Identifier: Apache-2.0

//! XQA decode ([`flashinfer::xqa`]): simulated on small problems (one CTA per sequence and
//! multi-block), then lowered to PTX if a toolchain is found. With `--features cuda` and
//! `GPU=1` it runs on the GPU instead: checks against the f64 reference, then CUDA-graph
//! timings at batch 1/16/64 and kv 1K/4K/16K (8 KV heads, page 16, head dim 128), beside
//! `flashinfer::decode` on the group-4 shapes it supports. Compare with
//! `xqa_flashinfer_bench.py` (FlashInfer's own XQA, built for sm_89).
//!
//! `cargo run -p cute-flashinfer --example xqa`; on a GPU,
//! `GPU=1 cargo run --release -p cute-flashinfer --features cuda --example xqa`.
//! `SHAPES=quick` times only kv 1K/4K; `CHECK=0` skips the checks;
//! `TYPES=0,2` times only those rows of the type list.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::xqa::gpu::{XqaKernel, run};
use flashinfer::xqa::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    use flashinfer::decode;
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    println!("device: {}", ctx.name());
    let win_sinks = {
        let mut c = Config::new(InputDtype::F16, CacheDtype::Input, 8, 16);
        (c.sliding_window, c.sinks) = (true, true);
        c
    };
    // Correctness: every type pair, groups, page sizes, multi-block, window + sinks
    // (`CHECK=0` skips it).
    let checks = if std::env::var("CHECK").as_deref() == Ok("0") {
        vec![]
    } else {
        vec![
            (Config::new(InputDtype::F16, CacheDtype::Input, 4, 16), vec![1usize, 37, 300, 600, 4097], 2usize, 0usize),
            (Config::new(InputDtype::F16, CacheDtype::E4M3, 8, 16), vec![77, 1000, 2049], 8, 0),
            (Config::new(InputDtype::BF16, CacheDtype::Input, 8, 32), vec![513, 9, 3000], 4, 0),
            (Config::new(InputDtype::BF16, CacheDtype::E4M3, 4, 64), vec![1500, 64], 8, 0),
            (Config::new(InputDtype::F16, CacheDtype::Input, 16, 128), vec![700, 129], 2, 0),
            (win_sinks, vec![5000, 90, 1100], 2, 1024),
        ]
    };
    for (cfg, lens, heads, window) in checks {
        let k = XqaKernel::new(&ctx, &tc, cfg)?;
        for nb_sub in [None, Some(1), Some(5)] {
            let mut p = Problem::new(cfg, &lens, heads, 11).with_scales(0.9, 0.5);
            if cfg.sliding_window {
                p = p.with_window(window);
            }
            let (n, err, _) = run(&ctx, &k, &p, nb_sub, true, false)?;
            println!("check {} batch {} nb_sub {n}: max |o - ref| = {err:.2e}", cfg.name(), lens.len());
            if err >= p.tolerance() {
                return Err("out of tolerance".into());
            }
        }
    }

    let quick = std::env::var("SHAPES").as_deref() == Ok("quick");
    let kv_lens: &[usize] = if quick { &[1024, 4096] } else { &[1024, 4096, 16384] };
    // `TYPES=1,2` picks rows of the type list (0: f16/f16, 1: f16/e4m3, 2: bf16/bf16).
    let types: Option<Vec<usize>> =
        std::env::var("TYPES").ok().map(|v| v.split(',').map(|x| x.parse().expect("TYPES")).collect());
    let heads = 8;
    // flashinfer::decode on the same shapes (group 4, f16 queries).
    let mut dks: Vec<(CacheDtype, decode::gpu::Kernels<'_>)> = vec![];
    for (cache, kv) in [(CacheDtype::Input, decode::KvDtype::F16), (CacheDtype::E4M3, decode::KvDtype::E4M3)] {
        dks.push((cache, decode::gpu::Kernels::with_kv(&ctx, &tc, decode::STAGES, false, kv)?));
    }
    println!("q     kv    group  batch  kv_len  nb_sub  xqa(us)  GB/s   decode(us)");
    for (ti, (input, cache)) in [
        (InputDtype::F16, CacheDtype::Input),
        (InputDtype::F16, CacheDtype::E4M3),
        (InputDtype::BF16, CacheDtype::Input),
    ]
    .into_iter()
    .enumerate()
    {
        if types.as_ref().is_some_and(|t| !t.contains(&ti)) {
            continue;
        }
        for group in [4usize, 8] {
            let cfg = Config::new(input, cache, group, 16);
            let k = XqaKernel::new(&ctx, &tc, cfg)?;
            for batch in [1usize, 16, 64] {
                for &kv_len in kv_lens {
                    let p = Problem::new(cfg, &vec![kv_len; batch], heads, 3);
                    let (n, _, t) = run(&ctx, &k, &p, None, false, true)?;
                    let t = t.unwrap();
                    drop(p);
                    let gbs = (2 * batch * kv_len * heads * HEAD_DIM * cfg.cache_bytes()) as f64 / (t * 1e-6) / 1e9;
                    let dec = match dks.iter().find(|(c, _)| *c == cache) {
                        Some((_, ks)) if group == decode::GROUP && input == InputDtype::F16 => {
                            let kv =
                                if cache == CacheDtype::E4M3 { decode::KvDtype::E4M3 } else { decode::KvDtype::F16 };
                            let dp = decode::Problem::new(&vec![kv_len; batch], 16, heads, 3).quantized(kv, 1.0, 1.0);
                            let (_, _, _, dt) = decode::gpu::run(&ctx, ks, &dp, false, true)?;
                            format!("{:9.1}", dt.unwrap().1)
                        }
                        _ => "        -".to_string(),
                    };
                    println!(
                        "{:5} {:5} {group:5}  {batch:5}  {kv_len:6}  {n:6}  {t:7.1}  {gbs:5.0}  {dec}",
                        format!("{input:?}").to_lowercase(),
                        if cache == CacheDtype::E4M3 {
                            "e4m3".to_string()
                        } else {
                            format!("{input:?}").to_lowercase()
                        }
                    );
                }
            }
        }
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    for (cfg, lens, heads) in [
        (Config::new(InputDtype::F16, CacheDtype::Input, 4, 16), vec![1usize, 37, 300, 600], 2usize),
        (Config::new(InputDtype::BF16, CacheDtype::E4M3, 8, 32), vec![333, 17], 1),
    ] {
        let p = Problem::new(cfg, &lens, heads, 7).with_scales(1.0, 0.5);
        for nb_sub in [1, 3] {
            let err = simulate(&p, nb_sub);
            println!("{} batch {lens:?} nb_sub {nb_sub}: max |o - ref| = {err:.2e}", cfg.name());
            assert!(err < p.tolerance(), "out of tolerance");
        }
    }
    match Toolchain::discover() {
        Some(tc) => {
            for cfg in [
                Config::new(InputDtype::F16, CacheDtype::Input, 8, 16),
                Config::new(InputDtype::F16, CacheDtype::E4M3, 8, 16),
                Config::new(InputDtype::BF16, CacheDtype::Input, 4, 16),
            ] {
                let k = kernel(&cfg);
                let ptx = k
                    .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                    .unwrap_or_else(|e| panic!("{e}"))
                    .ptx()
                    .to_string();
                // `PTX_DIR=dir` also writes each kernel's PTX there, for `ptxas -v` or SASS.
                if let Some(dir) = std::env::var_os("PTX_DIR") {
                    std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", k.name())), &ptx)
                        .unwrap_or_else(|e| panic!("{e}"));
                }
                println!(
                    "PTX {}: {} bytes, {} mma, {} ldmatrix, {} cp.async, {} ex2.approx",
                    k.name(),
                    ptx.len(),
                    ptx.matches("mma.sync").count(),
                    ptx.matches("ldmatrix").count(),
                    ptx.matches("cp.async.cg").count(),
                    ptx.matches("ex2.approx").count()
                );
            }
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}
