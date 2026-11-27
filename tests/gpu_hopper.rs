// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Hopper examples on an sm_90 GPU: the WGMMA dense GEMM (clusters, multicast, every
//! major mode, Float16, BFloat16, FP8 and 8-bit integers, batches), the persistent GEMM with
//! the device's real cluster count, and the FMHA forward (causal, grouped-query, persistent),
//! each checked against the same CPU references as the simulator tests (`tests/hopper_gemm.rs`,
//! `tests/hopper_persistent_gemm.rs`, `tests/hopper_fmha.rs`), on problems larger than the
//! simulator can run. The `#[ignore]`d `bench_*` tests time the GEMM and the FMHA at full size.
//!
//! Skipped on anything but sm_90 (set `CUTE_RS_EXPECT_SM=90` to make that a failure):
//!
//! ```text
//! cargo test --release --features cuda --test gpu_hopper
//! cargo test --release --features cuda --test gpu_hopper -- --ignored --nocapture bench_
//! ```

// hopper_persistent_gemm loads its own copy of hopper_gemm; this one gives its reference.
#![allow(clippy::duplicate_mod)]
// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/hopper_gemm.rs"]
mod hopper_gemm;

#[path = "../examples/hopper_persistent_gemm.rs"]
mod hopper_persistent_gemm;

#[path = "../examples/hopper_fmha.rs"]
mod hopper_fmha;

#[path = "support/gpu_device.rs"]
mod gpu_device;

use cute::prelude::*;
use gpu_device::{Device, device, report};
use hopper_gemm::{Data, HopperWgmmaGemmKernel, Majors, Problem};

fn hopper(what: &str) -> Option<Device> {
    device(9, what)
}

fn gemm(
    ab: (DType, DType),
    c: DType,
    acc: DType,
    majors: Majors,
    tile_mn: (usize, usize),
    cluster_mn: (usize, usize),
) -> HopperWgmmaGemmKernel {
    HopperWgmmaGemmKernel::new(acc, tile_mn, cluster_mn).setup(ab.0, ab.1, c, majors)
}

fn f16(majors: Majors, tile_mn: (usize, usize), cluster_mn: (usize, usize)) -> HopperWgmmaGemmKernel {
    gemm((DType::F16, DType::F16), DType::F16, DType::F32, majors, tile_mn, cluster_mn)
}

/// Build `g` once, then run and check each problem on the device.
fn check_all(dev: &Device, g: &HopperWgmmaGemmKernel, problems: &[(usize, usize, usize, usize)], data: Data) {
    let kernel = g.build();
    let loaded = dev.load(&kernel);
    for &(m, n, k, l) in problems {
        let problem = Problem { data, ..Problem::new(m, n, k, l) };
        let mut args = hopper_gemm::inputs(g, &problem);
        loaded.launch(g.launch(m, n, l), &mut args);
        hopper_gemm::check(g, &problem, &args);
    }
}

#[test]
fn gemm_float16_clusters_on_device() {
    let Some(dev) = hopper("gemm_float16_clusters_on_device") else { return };
    let problems = [(128, 128, 64, 1), (200, 136, 72, 1), (1024, 1024, 1024, 1), (1000, 776, 520, 1)];
    for (tile, cluster) in [((128, 128), (1, 1)), ((128, 128), (2, 1)), ((128, 128), (1, 2)), ((64, 64), (2, 2))] {
        check_all(&dev, &f16(Majors::KKN, tile, cluster), &problems, Data::Integers);
    }
}

#[test]
fn gemm_float16_every_major_mode_on_device() {
    let Some(dev) = hopper("gemm_float16_every_major_mode_on_device") else { return };
    for a in ['k', 'm'] {
        for b in ['k', 'n'] {
            for c in ['n', 'm'] {
                let majors = Majors::new(a, b, c);
                check_all(
                    &dev,
                    &f16(majors, (64, 128), (2, 1)),
                    &[(128, 136, 72, 1), (512, 512, 256, 1)],
                    Data::Integers,
                );
                check_all(&dev, &f16(majors, (128, 64), (1, 2)), &[(136, 128, 64, 1)], Data::Integers);
            }
        }
    }
}

#[test]
fn gemm_two_warpgroups_and_batches_on_device() {
    let Some(dev) = hopper("gemm_two_warpgroups_and_batches_on_device") else { return };
    check_all(&dev, &f16(Majors::KKN, (128, 256), (1, 1)), &[(200, 264, 72, 1), (1024, 1024, 512, 1)], Data::Integers);
    check_all(&dev, &f16(Majors::new('m', 'n', 'm'), (128, 256), (2, 1)), &[(256, 256, 64, 1)], Data::Integers);
    check_all(&dev, &f16(Majors::new('m', 'k', 'n'), (64, 64), (2, 1)), &[(128, 64, 64, 3)], Data::Integers);
    check_all(&dev, &f16(Majors::KKN, (128, 128), (1, 1)), &[(136, 72, 80, 2), (512, 384, 256, 4)], Data::Integers);
}

#[test]
fn gemm_other_types_on_device() {
    use DType::{BF16, F8E4M3FN as E4, F8E5M2 as E5, F16, F32, I8, I32, U8};
    let Some(dev) = hopper("gemm_other_types_on_device") else { return };
    let cases = [
        (gemm((BF16, BF16), BF16, F32, Majors::KKN, (128, 128), (2, 1)), (256, 128, 128, 1)),
        (gemm((BF16, BF16), BF16, F32, Majors::new('m', 'n', 'm'), (64, 128), (1, 2)), (72, 200, 64, 1)),
        (gemm((F16, F16), F16, F16, Majors::KKN, (128, 128), (1, 2)), (128, 200, 128, 1)),
        (gemm((F16, F16), F32, F32, Majors::new('k', 'k', 'm'), (128, 128), (2, 1)), (200, 128, 64, 1)),
        (gemm((E4, E4), F16, F32, Majors::KKN, (128, 128), (1, 1)), (512, 512, 512, 1)),
        (gemm((E4, E5), F32, F32, Majors::KKN, (128, 128), (2, 1)), (256, 128, 144, 1)),
        (gemm((E5, E4), F16, F16, Majors::new('k', 'k', 'm'), (64, 128), (1, 2)), (64, 256, 128, 1)),
        (gemm((E5, E5), E5, F32, Majors::KKN, (128, 64), (2, 1)), (200, 64, 128, 1)),
        (gemm((I8, I8), I32, I32, Majors::KKN, (128, 128), (2, 1)), (512, 256, 384, 1)),
        (gemm((U8, I8), F16, I32, Majors::KKN, (64, 64), (1, 2)), (64, 128, 160, 1)),
    ];
    for (g, problem) in &cases {
        check_all(&dev, g, &[*problem], Data::Integers);
    }
}

#[test]
fn gemm_fractions_on_device() {
    let Some(dev) = hopper("gemm_fractions_on_device") else { return };
    let cases = [
        f16(Majors::KKN, (128, 128), (2, 1)),
        gemm((DType::BF16, DType::BF16), DType::BF16, DType::F32, Majors::KKN, (128, 128), (1, 2)),
        gemm((DType::F16, DType::F16), DType::F16, DType::F16, Majors::new('m', 'n', 'n'), (128, 64), (1, 1)),
        gemm((DType::F8E4M3FN, DType::F8E5M2), DType::F32, DType::F32, Majors::KKN, (128, 128), (1, 1)),
    ];
    for g in &cases {
        check_all(&dev, g, &[(136, 256, 256, 1), (768, 512, 1024, 1)], Data::Fractions);
    }
}

/// The persistent GEMM with as many clusters as the device holds, and with fewer (so each
/// cluster runs several tiles), checked with the dense port's reference.
#[test]
fn persistent_gemm_on_device() {
    let Some(dev) = hopper("persistent_gemm_on_device") else { return };
    for (tile, cluster, swizzle, along_m) in
        [((128, 128), (2, 1), 2, true), ((128, 128), (1, 2), 1, false), ((128, 256), (1, 1), 4, true)]
    {
        let p =
            hopper_persistent_gemm::HopperWgmmaGemmPersistentKernel::new(DType::F32, tile, cluster, swizzle, along_m)
                .setup(DType::F16, DType::F16, DType::F16, hopper_persistent_gemm::Majors::KKN);
        // The same configuration as the dense port sees it, for its reference and check.
        let twin = f16(Majors::KKN, tile, cluster);
        let device_clusters = p.max_active_clusters_on_device().expect("max active clusters");
        let kernel = p.build();
        let loaded = dev.load(&kernel);
        for (m, n, k, l) in [(384, 256, 128, 1), (1024, 1024, 512, 1), (2000, 1032, 264, 2)] {
            for clusters in [device_clusters, 3] {
                let problem = hopper_persistent_gemm::Problem::new(m, n, k, l);
                let mut args = p.inputs(&problem);
                loaded.launch(p.launch(m, n, l, clusters), &mut args);
                hopper_gemm::check(&twin, &Problem::new(m, n, k, l), &args[..7]);
            }
        }
    }
}

/// The FMHA forward: plain, causal, grouped-query and persistent, against the CPU's
/// attention and log-sum-exp.
#[test]
fn fmha_on_device() {
    use hopper_fmha::{Problem as Attn, check, fmha_for, inputs};
    let Some(dev) = hopper("fmha_on_device") else { return };
    let sms = dev.sm_count() as usize;
    let cases = [
        (Attn::new(1, 128, 200, 2, 1, 64).causal(), (64, 128), false),
        (Attn::new(2, 130, 200, 4, 2, 64), (64, 128), true),
        (Attn::new(1, 300, 300, 2, 1, 64).causal(), (64, 128), true),
        (Attn::new(2, 1024, 1024, 8, 2, 64), (64, 128), true),
        (Attn::new(1, 512, 777, 4, 4, 64).causal(), (64, 64), false),
    ];
    for (p, tile, persistent) in &cases {
        let fmha = fmha_for(p, *tile, DType::F16, DType::F16, *persistent);
        let kernel = fmha.build();
        let mut args = inputs(&fmha, p);
        dev.run(&kernel, fmha.launch(p.s_q, p.h, p.b, sms), &mut args);
        check(&fmha, p, &args);
    }
}

/// Float16 GEMM throughput at 8192^3 (fractions; C unchecked: the tests above check the same
/// kernels). Compare with CUTLASS's `hopper` example or cuBLAS on the same GPU.
#[test]
#[ignore = "a timing, for the hardware runs"]
fn bench_gemm() {
    let Some(dev) = hopper("bench_gemm") else { return };
    let (m, n, k) = (8192, 8192, 8192);
    for (tile, cluster) in [((128, 128), (2, 1)), ((128, 256), (2, 1)), ((128, 256), (1, 2))] {
        let g = f16(Majors::KKN, tile, cluster);
        let kernel = g.build();
        let loaded = dev.load(&kernel);
        let mut args = hopper_gemm::inputs(&g, &Problem { data: Data::Fractions, ..Problem::new(m, n, k, 1) });
        let s = loaded.best_of(10, g.launch(m, n, 1), &mut args);
        report(&g.name(), &format!("{m}x{n}x{k}"), 2.0 * (m * n * k) as f64, s);
    }
    let p = hopper_persistent_gemm::HopperWgmmaGemmPersistentKernel::new(DType::F32, (128, 256), (2, 1), 4, true)
        .setup(DType::F16, DType::F16, DType::F16, hopper_persistent_gemm::Majors::KKN);
    let kernel = p.build();
    let loaded = dev.load(&kernel);
    let mut args = p.inputs(&hopper_persistent_gemm::Problem::new(m, n, k, 1));
    let clusters = p.max_active_clusters_on_device().expect("max active clusters");
    let s = loaded.best_of(10, p.launch(m, n, 1, clusters), &mut args);
    report(&p.name(), &format!("{m}x{n}x{k}"), 2.0 * (m * n * k) as f64, s);
}

/// FMHA forward throughput (Float16, head dim 64), not causal and causal.
#[test]
#[ignore = "a timing, for the hardware runs"]
fn bench_fmha() {
    use hopper_fmha::{Problem as Attn, fmha_for, inputs};
    let Some(dev) = hopper("bench_fmha") else { return };
    let sms = dev.sm_count() as usize;
    for p in [Attn::new(4, 4096, 4096, 16, 16, 64), Attn::new(4, 4096, 4096, 16, 16, 64).causal()] {
        let fmha = fmha_for(&p, (64, 128), DType::F16, DType::F16, true);
        let kernel = fmha.build();
        let loaded = dev.load(&kernel);
        let mut args = inputs(&fmha, &p);
        let s = loaded.best_of(10, fmha.launch(p.s_q, p.h, p.b, sms), &mut args);
        let full = 4.0 * (p.b * p.h * p.s_q * p.s_k * p.d) as f64;
        let flops = if p.is_causal { full / 2.0 } else { full };
        report(
            &fmha.name(),
            &format!("b{}h{}s{}d{}{}", p.b, p.h, p.s_q, p.d, if p.is_causal { "c" } else { "" }),
            flops,
            s,
        );
    }
}
