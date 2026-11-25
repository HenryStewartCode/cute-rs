// SPDX-License-Identifier: MIT OR Apache-2.0

//! The datacenter Blackwell examples on an sm_100 GPU (B200, GB200; sm_103 too): the
//! tutorial FP16 GEMMs (one CTA, warp-specialized CTA pairs, the static persistent and the CLC
//! schedulers, preferred and fallback clusters), the canonical dense GEMM (1 CTA, a CTA pair,
//! pairs with multicast; Float16 and Float32 C, N- and M-major), the NVFP4 block-scaled GEMM
//! (UE4M3 and UE8M0 scale factors), the mixed-input GEMM (Int4 with and without scales) and
//! the TMA copy, each checked bit for bit against the same CPU references as the simulator
//! tests (`tests/examples.rs`, `tests/mixed_input.rs`), on problems larger than the simulator
//! can run. The `#[ignore]`d `bench_*` tests time the GEMMs at full size.
//!
//! Skipped on anything but sm_10x (set `CUTE_RS_EXPECT_SM=100` to make that a failure):
//!
//! ```text
//! cargo test --release --features cuda --test gpu_blackwell
//! cargo test --release --features cuda --test gpu_blackwell -- --ignored --nocapture bench_
//! ```

#[allow(dead_code)]
#[path = "../examples/blackwell_fp16_gemm.rs"]
mod blackwell_fp16_gemm;

#[allow(dead_code)]
#[path = "../examples/blackwell_dense_gemm.rs"]
mod blackwell_dense_gemm;

#[allow(dead_code)]
#[path = "../examples/blackwell_nvfp4_gemm.rs"]
mod blackwell_nvfp4_gemm;

#[allow(dead_code)]
#[path = "../examples/blackwell_mixed_input_gemm.rs"]
mod blackwell_mixed_input_gemm;

#[allow(dead_code)]
#[path = "../examples/tma_copy.rs"]
mod tma_copy;

#[path = "support/gpu_device.rs"]
mod gpu_device;

use cute::prelude::*;
use gpu_device::{Device, device, report};

fn blackwell(what: &str) -> Option<Device> {
    device(10, what)
}

/// `m`, `n`, `k`, then A, B and C, as every GEMM example here takes them.
fn mnk_abc<T: Numeric, C: Numeric>((m, n, k): (usize, usize, usize), a: &[T], b: &[T], c: &[C]) -> Vec<KernelArg> {
    vec![
        KernelArg::scalar(m as i32),
        KernelArg::scalar(n as i32),
        KernelArg::scalar(k as i32),
        KernelArg::buffer(a),
        KernelArg::buffer(b),
        KernelArg::buffer(c),
    ]
}

fn first_difference<T: PartialEq>(got: &[T], want: &[T]) -> Option<usize> {
    assert_eq!(got.len(), want.len());
    (0..got.len()).find(|&i| got[i] != want[i])
}

#[test]
fn fp16_gemm_0_on_device() {
    use blackwell_fp16_gemm::{FP16_GEMM_0, IoDtype, reference, small_ints};
    let Some(dev) = blackwell("fp16_gemm_0_on_device") else { return };
    let cfg = FP16_GEMM_0;
    let kernel = cfg.build();
    let loaded = dev.load(&kernel);
    for (m, n, k) in [(256, 512, 256), (1024, 1024, 512), (2048, 1024, 1024)] {
        let (a, b, c) = (small_ints(m * k, 1111), small_ints(n * k, 2222), small_ints(m * n, 3333));
        let mut args = mnk_abc((m, n, k), &a, &b, &c);
        loaded.launch(cfg.launch(m, n), &mut args);
        let got = args[5].to_vec::<IoDtype>();
        if let Some(i) = first_difference(&got, &reference(&a, &b, m, n, k)) {
            panic!("fp16_gemm_0 {m}x{n}x{k}: C({}, {}) differs", i / n, i % n);
        }
    }
}

/// The warp-specialized tutorial GEMMs: CTA pairs with a TMA-store epilogue, the static
/// persistent scheduler (with the device's cluster count, and with one cluster), CLC, and
/// preferred clusters with a fallback, on whole and ragged problems.
#[test]
fn fp16_gemm_ws_on_device() {
    use blackwell_fp16_gemm::{FP16_GEMM_2, FP16_GEMM_3, FP16_GEMM_3_1, FP16_GEMM_4, FP16_GEMM_4_STATIC, IoDtype};
    use blackwell_fp16_gemm::{reference, small_ints};
    let Some(dev) = blackwell("fp16_gemm_ws_on_device") else { return };
    // Inputs and references once per problem: the CPU GEMM dominates otherwise.
    let problems: Vec<_> = [(256, 256, 128), (264, 136, 72), (512, 512, 64), (2048, 2048, 1024), (1000, 1528, 520)]
        .into_iter()
        .map(|(m, n, k)| {
            let (a, b) = (small_ints(m * k, 1111), small_ints(n * k, 2222));
            let want = reference(&a, &b, m, n, k);
            ((m, n, k), a, b, want)
        })
        .collect();
    for cfg in [FP16_GEMM_2, FP16_GEMM_3, FP16_GEMM_3_1, FP16_GEMM_4, FP16_GEMM_4_STATIC] {
        let (cx, cy, cz) = cfg.cluster_shape_mnk;
        let device_clusters = dev.max_active_clusters((cx * cy * cz) as u32);
        let kernel = cfg.build();
        let loaded = dev.load(&kernel);
        for ((m, n, k), a, b, want) in &problems {
            let (m, n, k) = (*m, *n, *k);
            for clusters in [device_clusters, 1] {
                let mut args = mnk_abc((m, n, k), a, b, &vec![IoDtype::from_f32(-7.0); m * n]);
                loaded.launch(cfg.launch(m, n, clusters), &mut args);
                let got = args[5].to_vec::<IoDtype>();
                if let Some(i) = first_difference(&got, want) {
                    panic!("{} {m}x{n}x{k} ({clusters} clusters): C({}, {}) differs", cfg.name, i / n, i % n);
                }
            }
        }
    }
}

fn dense<C: blackwell_dense_gemm::CType>(
    dev: &Device,
    cfg: &blackwell_dense_gemm::DenseGemm,
    mnk: (usize, usize, usize),
) {
    use blackwell_dense_gemm::{reference, small_ints};
    let (m, n, k) = mnk;
    let kernel = cfg.build::<C>();
    let (a, b) = (small_ints(m * k, 1111), small_ints(n * k, 2222));
    let mut args = mnk_abc(mnk, &a, &b, &vec![C::from_f(-7.0); m * n]);
    dev.run(&kernel, cfg.launch(m, n), &mut args);
    let got = args[5].to_vec::<C>();
    if let Some(i) = first_difference(&got, &reference::<C>(&a, &b, mnk, cfg.c_layout, |x| x)) {
        panic!("{} {m}x{n}x{k}: C element {i} differs", cfg.name);
    }
}

#[test]
fn dense_gemm_on_device() {
    use blackwell_dense_gemm::{DENSE_GEMM_1CTA, DENSE_GEMM_2CTA, DENSE_GEMM_2CTA_MULTICAST, DenseGemm};
    let Some(dev) = blackwell("dense_gemm_on_device") else { return };
    for cfg in [DENSE_GEMM_1CTA, DENSE_GEMM_2CTA, DENSE_GEMM_2CTA_MULTICAST] {
        for mnk in [(200, 136, 200), (264, 264, 136), (2048, 2048, 1024), (1000, 1528, 520)] {
            dense::<F16>(&dev, &cfg, mnk);
            dense::<f32>(&dev, &cfg, mnk);
        }
    }
    for cfg in [
        DenseGemm { c_layout: LayoutEnum::COL_MAJOR, name: "blackwell_dense_gemm_1cta_m_major", ..DENSE_GEMM_1CTA },
        DenseGemm { c_layout: LayoutEnum::COL_MAJOR, name: "blackwell_dense_gemm_2cta_m_major", ..DENSE_GEMM_2CTA },
    ] {
        for mnk in [(136, 72, 72), (1024, 512, 256)] {
            dense::<F16>(&dev, &cfg, mnk);
        }
    }
}

/// The NVFP4 GEMM: E2M1 operands with UE4M3 and UE8M0 scale factors, decoded by the CPU.
#[test]
fn nvfp4_gemm_on_device() {
    use blackwell_nvfp4_gemm::{AbDtype, CDtype, NVFP4_GEMM_0, NVFP4_GEMM_0_UE8M0, Problem, typed};
    let Some(dev) = blackwell("nvfp4_gemm_on_device") else { return };
    for cfg in [NVFP4_GEMM_0, NVFP4_GEMM_0_UE8M0] {
        let kernel = cfg.build();
        let loaded = dev.load(&kernel);
        for (m, n, k) in [(256, 512, 512), (128, 256, 1280), (1024, 1024, 2048)] {
            let p = Problem::new(&cfg, m, n, k);
            let (sfa, sfb) = (p.sf_image(&p.sfa, m, cfg.sf_vec_size), p.sf_image(&p.sfb, n, cfg.sf_vec_size));
            let sf = if cfg.sf_dtype == DType::F8E4M3FN { typed::<F8E4M3FN> } else { typed::<F8E8M0FNU> };
            let mut args = vec![
                KernelArg::scalar(m as i32),
                KernelArg::scalar(n as i32),
                KernelArg::scalar(k as i32),
                typed::<AbDtype>(&p.a),
                typed::<AbDtype>(&p.b),
                sf(&sfa),
                sf(&sfb),
                KernelArg::buffer(&vec![CDtype::from_f32(0.0); m * n]),
            ];
            loaded.launch(cfg.launch(m, n), &mut args);
            let got = args[7].to_vec::<CDtype>();
            if let Some(i) = first_difference(&got, &p.reference(&cfg)) {
                panic!("{} {m}x{n}x{k}: C({}, {}) differs", cfg.name, i / n, i % n);
            }
        }
    }
}

/// The mixed-input GEMM: Int4 A converted, or converted and scaled, K- and M-major, one CTA
/// and a CTA pair.
#[test]
fn mixed_input_gemm_on_device() {
    use blackwell_mixed_input_gemm::*;
    let Some(dev) = blackwell("mixed_input_gemm_on_device") else { return };
    for cfg in [
        MIXED_K_MAJOR_CONVERT,
        MIXED_K_MAJOR_SCALE,
        MIXED_M_MAJOR_CONVERT,
        MIXED_M_MAJOR_SCALE,
        MIXED_M_MAJOR_2CTA,
        MIXED_M_MAJOR_2CTA_SCALE,
    ] {
        for mnk in [(128, 64, 256), (256, 64, 256), (512, 256, 1024), (2048, 1024, 2048)] {
            if !cfg.can_implement(mnk) {
                continue;
            }
            let (m, n, _) = mnk;
            let kernel = cfg.build(mnk);
            let p = Problem::new(&cfg, mnk, 0x5eed + m as u64);
            let (a, scale) = (p.a_device(&cfg), p.scale_device(&cfg));
            let b: Vec<BF16> = p.b.iter().map(|&x| BF16::from_f32(x)).collect();
            let mut args = vec![KernelArg::buffer(&a)];
            if !scale.is_empty() {
                args.push(KernelArg::buffer(&scale));
            }
            args.push(KernelArg::buffer(&b));
            args.push(KernelArg::buffer(&vec![-7.0f32; m * n]));
            dev.run(&kernel, cfg.launch(m, n), &mut args);
            let got = args.last().unwrap().to_vec::<f32>();
            if let Some(i) = first_difference(&got, &p.reference(&cfg)) {
                panic!(
                    "{} {mnk:?}: C({}, {}) is {}, the reference {}",
                    cfg.name,
                    i / n,
                    i % n,
                    got[i],
                    p.reference(&cfg)[i]
                );
            }
        }
    }
}

#[test]
fn tma_copy_on_device() {
    let Some(dev) = blackwell("tma_copy_on_device") else { return };
    let copy = tma_copy::Sm100SimpleCopyKernel::new();
    let kernel = copy.build();
    let loaded = dev.load(&kernel);
    for (m, n) in [(tma_copy::DEFAULT_M, tma_copy::DEFAULT_N), (200, 144), (4096, 4096)] {
        let mut args = tma_copy::args(m, n);
        loaded.launch(copy.launch(m, n), &mut args);
        tma_copy::check(&args);
    }
}

/// GEMM throughput at 8192^3: the dense GEMM's three configurations, the persistent and CLC
/// tutorial GEMMs, and NVFP4 (C unchecked: the tests above check the same kernels). Compare
/// with CUTLASS's Blackwell examples or cuBLAS on the same GPU.
#[test]
#[ignore = "a timing, for the hardware runs"]
fn bench_gemm() {
    let Some(dev) = blackwell("bench_gemm") else { return };
    let (m, n, k) = (8192, 8192, 8192);
    let flops = 2.0 * (m * n * k) as f64;
    let shape = format!("{m}x{n}x{k}");
    {
        use blackwell_dense_gemm::*;
        let (a, b) = (small_ints(m * k, 1), small_ints(n * k, 2));
        for cfg in [DENSE_GEMM_1CTA, DENSE_GEMM_2CTA, DENSE_GEMM_2CTA_MULTICAST] {
            let kernel = cfg.build::<F16>();
            let loaded = dev.load(&kernel);
            let mut args = mnk_abc((m, n, k), &a, &b, &vec![F16::from_f32(0.0); m * n]);
            report(cfg.name, &shape, flops, loaded.best_of(10, cfg.launch(m, n), &mut args));
        }
    }
    {
        use blackwell_fp16_gemm::*;
        let (a, b) = (small_ints(m * k, 1), small_ints(n * k, 2));
        for cfg in [FP16_GEMM_3, FP16_GEMM_3_1, FP16_GEMM_4] {
            let (cx, cy, cz) = cfg.cluster_shape_mnk;
            let clusters = dev.max_active_clusters((cx * cy * cz) as u32);
            let kernel = cfg.build();
            let loaded = dev.load(&kernel);
            let mut args = mnk_abc((m, n, k), &a, &b, &vec![IoDtype::from_f32(0.0); m * n]);
            report(cfg.name, &shape, flops, loaded.best_of(10, cfg.launch(m, n, clusters), &mut args));
        }
    }
    {
        use blackwell_nvfp4_gemm::*;
        for cfg in [NVFP4_GEMM_0, NVFP4_GEMM_0_UE8M0] {
            let p = Problem::new(&cfg, m, n, k);
            let (sfa, sfb) = (p.sf_image(&p.sfa, m, cfg.sf_vec_size), p.sf_image(&p.sfb, n, cfg.sf_vec_size));
            let sf = if cfg.sf_dtype == DType::F8E4M3FN { typed::<F8E4M3FN> } else { typed::<F8E8M0FNU> };
            let kernel = cfg.build();
            let loaded = dev.load(&kernel);
            let mut args = vec![
                KernelArg::scalar(m as i32),
                KernelArg::scalar(n as i32),
                KernelArg::scalar(k as i32),
                typed::<AbDtype>(&p.a),
                typed::<AbDtype>(&p.b),
                sf(&sfa),
                sf(&sfb),
                KernelArg::buffer(&vec![CDtype::from_f32(0.0); m * n]),
            ];
            report(cfg.name, &shape, flops, loaded.best_of(10, cfg.launch(m, n), &mut args));
        }
    }
}
