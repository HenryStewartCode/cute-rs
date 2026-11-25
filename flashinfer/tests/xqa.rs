// SPDX-License-Identifier: Apache-2.0

//! XQA decode on the simulator against the f64 reference, with cute-rs's race and
//! `cp.async` checks: f16 / bf16 queries, a cache of the query type or E4M3 with scales,
//! groups 4, 8 and 16, page sizes, one CTA per sequence and the multi-block path (partials
//! merged by the last CTA), sliding window and attention sinks.

use cute::prelude::*;
use flashinfer::xqa::*;

fn check(p: &Problem, nb_sub: usize) {
    let err = simulate(p, nb_sub);
    assert!(
        err < p.tolerance(),
        "{} nb_sub {nb_sub}: |o - ref| {err:.2e} (tolerance {:.1e})",
        p.cfg.name(),
        p.tolerance()
    );
}

/// Lengths cover one token, partial steps, pages and CTA tiles, and more than one tile.
#[test]
fn f16_whole_and_multi_block() {
    let cfg = Config::new(InputDtype::F16, CacheDtype::Input, 4, 16);
    let p = Problem::new(cfg, &[1, 37, 300, 600], 2, 7);
    check(&p, 1);
    check(&p, 3);
}

#[test]
fn bf16_group8_page32() {
    let cfg = Config::new(InputDtype::BF16, CacheDtype::Input, 8, 32);
    let p = Problem::new(cfg, &[260, 5], 1, 3).with_scales(0.7, 1.0);
    check(&p, 1);
    check(&p, 2);
}

#[test]
fn e4m3_cache_with_scales() {
    for input in [InputDtype::F16, InputDtype::BF16] {
        let cfg = Config::new(input, CacheDtype::E4M3, 8, 64);
        let p = Problem::new(cfg, &[333, 17], 1, 5).with_scales(1.3, 0.5);
        check(&p, 1);
        check(&p, 2);
    }
}

#[test]
fn group16_page128() {
    let cfg = Config::new(InputDtype::F16, CacheDtype::Input, 16, 128);
    let p = Problem::new(cfg, &[140, 513], 1, 9);
    check(&p, 1);
    check(&p, 3);
}

#[test]
fn sliding_window_and_sinks() {
    let mut cfg = Config::new(InputDtype::F16, CacheDtype::Input, 4, 16);
    cfg.sliding_window = true;
    cfg.sinks = true;
    let p = Problem::new(cfg, &[700, 90, 300], 1, 11).with_window(300);
    check(&p, 1);
    check(&p, 2);
}

/// The kernel lowers to tensor-core MMAs, transposed `ldmatrix`, 16-byte `cp.async` and
/// the fast `exp2`.
#[test]
fn lowers_to_mma_and_cp_async() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for cache in [CacheDtype::Input, CacheDtype::E4M3] {
        let cfg = Config::new(InputDtype::F16, cache, 8, 16);
        let ptx =
            kernel(&cfg).compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap().ptx().to_string();
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"), "{}", cfg.name());
        assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.trans"), "{}", cfg.name());
        assert!(ptx.contains("cp.async.cg.shared.global"), "{}", cfg.name());
        assert!(ptx.contains("ex2.approx.ftz.f32"), "{}", cfg.name());
    }
}
