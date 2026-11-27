// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! More datacenter Blackwell examples on an sm_100 GPU (B200, GB200; sm_103 too), one
//! configuration family each: the custom-epilogue (EFC) dense GEMM, the persistent GEMM built
//! from cute_ext's pieces, and the Task Scheduling GEMM. Each is checked bit for bit against the
//! same CPU reference as its simulator test (`tests/efc_examples.rs`, `tests/cute_ext_gemm.rs`,
//! `tests/ts_gemm.rs`).
//!
//! Skipped on anything but sm_10x (set `CUTE_RS_EXPECT_SM=100` to make that a failure):
//!
//! ```text
//! cargo test --release --features cuda --test gpu_blackwell_more
//! ```

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/custom_epilogue_dense_gemm.rs"]
mod custom;

#[path = "../examples/cute_ext_dense_gemm_persistent.rs"]
mod persistent;

#[path = "../examples/ts_gemm_simple.rs"]
mod ts_gemm_simple;

#[path = "support/gpu_device.rs"]
mod gpu_device;

use custom::efc::dense_gemm::{AbDtype, DenseGemmEfc, EFC_GEMM_1CTA, EFC_GEMM_2CTA_MULTICAST, matmul, small_ints};
use custom::efc::{HostArgs, Param};
use cute::prelude::*;
use cute::utils::LayoutEnum;
use gpu_device::{Device, device};

fn blackwell(what: &str) -> Option<Device> {
    device(10, what)
}

/// A host buffer of `dtype` from `f64` values.
fn to_buffer(dtype: DType, v: &[f64]) -> KernelArg {
    match dtype {
        DType::F16 => KernelArg::buffer(&v.iter().map(|&x| F16::from_f32(x as f32)).collect::<Vec<_>>()),
        DType::BF16 => KernelArg::buffer(&v.iter().map(|&x| BF16::from_f32(x as f32)).collect::<Vec<_>>()),
        DType::F32 => KernelArg::buffer(&v.iter().map(|&x| x as f32).collect::<Vec<_>>()),
        _ => panic!("tensors of {dtype} are not supported"),
    }
}

fn from_buffer(dtype: DType, arg: &KernelArg) -> Vec<f64> {
    match dtype {
        DType::F16 => arg.to_vec::<F16>().iter().map(|x| x.to_f32() as f64).collect(),
        DType::BF16 => arg.to_vec::<BF16>().iter().map(|x| x.to_f32() as f64).collect(),
        DType::F32 => arg.to_vec::<f32>().iter().map(|&x| x as f64).collect(),
        _ => unreachable!(),
    }
}

/// The custom epilogue (two read, two written tensors) on one CTA with mixed element types and
/// on a 2x2 cluster of CTA pairs with multicast, on ragged problems.
#[test]
fn custom_efc_gemm_on_device() {
    let Some(dev) = blackwell("custom_efc_gemm_on_device") else { return };
    let mixed = custom::Dtypes { c: DType::F16, d: DType::F32, x: DType::BF16, y: DType::F16 };
    for (cfg, mnk, t) in [
        (EFC_GEMM_1CTA, (200, 136, 192), mixed),
        (EFC_GEMM_1CTA, (1024, 512, 256), custom::FLOAT32),
        (EFC_GEMM_2CTA_MULTICAST, (264, 264, 136), custom::FLOAT32),
        (EFC_GEMM_2CTA_MULTICAST, (1024, 1024, 256), custom::FLOAT32),
    ] {
        let (m, n, k) = mnk;
        let gemm = DenseGemmEfc::new(cfg, custom::epilogue(t, LayoutEnum::ROW_MAJOR).unwrap());
        let roles = gemm.tensor_roles().unwrap();
        let mut inputs = HostArgs::default();
        for (name, v) in [("alpha", 1.0), ("beta", 2.0), ("x_factor", 3.0)] {
            inputs.scalars.insert(name.into(), v);
        }
        inputs.tensors.insert("C".into(), small_ints(m * n, 44));
        inputs.tensors.insert("X".into(), small_ints(m * n, 55));
        inputs.tensors.insert("D".into(), vec![-7.0; m * n]);
        inputs.tensors.insert("Y".into(), vec![-7.0; m * n]);

        let (a, b) = (small_ints(m * k, 1111), small_ints(n * k, 2222));
        let mut args = vec![
            KernelArg::scalar(m as i32),
            KernelArg::scalar(n as i32),
            KernelArg::scalar(k as i32),
            to_buffer(AbDtype::DTYPE, &a),
            to_buffer(AbDtype::DTYPE, &b),
        ];
        for p in gemm.efc.params() {
            match p {
                Param::Scalar(s) => args.push(KernelArg::scalar(inputs.scalars[s] as f32)),
                Param::Tensor { name, dtype, .. } => args.push(to_buffer(*dtype, &inputs.tensors[name])),
                Param::Config => unreachable!(),
            }
        }
        let (cm, cn) = cfg.cluster_shape_mn;
        let clusters = dev.max_active_clusters((cm * cn) as u32);
        dev.run(&gemm.build(), gemm.launch(m, n, clusters), &mut args);

        let mut want = inputs.clone();
        gemm.efc.evaluate_on_host(&matmul(&a, &b, mnk, roles.d_layout), &mut want);
        for (i, p) in gemm.efc.params().iter().enumerate() {
            let Param::Tensor { name, dtype, .. } = p else { continue };
            let got = from_buffer(*dtype, &args[5 + i]);
            if let Some(j) = (0..got.len()).find(|&j| got[j] != want.tensors[name][j]) {
                panic!(
                    "{} {mnk:?}: tensor `{name}` element {j} is {}, the reference {}",
                    cfg.name, got[j], want.tensors[name][j]
                );
            }
        }
    }
}

/// The persistent GEMM from cute_ext's pieces: 1SM, a CTA pair, and a pair with multicast, on
/// whole and ragged problems, with the device's cluster count and with one cluster.
#[test]
fn cute_ext_persistent_gemm_on_device() {
    use persistent::{ONE_SM, TWO_SM, TWO_SM_MULTICAST, reference, small_ints};
    let Some(dev) = blackwell("cute_ext_persistent_gemm_on_device") else { return };
    for cfg in [ONE_SM, TWO_SM, TWO_SM_MULTICAST] {
        let (cm, cn) = cfg.cluster_shape_mn;
        let device_clusters = dev.max_active_clusters((cm * cn) as u32);
        let kernel = cfg.build();
        let loaded = dev.load(&kernel);
        for (m, n, k) in [(512, 384, 256), (300, 200, 136), (1024, 1024, 512)] {
            let (a, b) = (small_ints(m * k, 1111), small_ints(n * k, 2222));
            let want = reference(&a, &b, m, n, k);
            for clusters in [device_clusters, 1] {
                let mut args = vec![
                    KernelArg::scalar(m as i32),
                    KernelArg::scalar(n as i32),
                    KernelArg::scalar(k as i32),
                    KernelArg::buffer(&a),
                    KernelArg::buffer(&b),
                    KernelArg::f32(vec![-7.0; m * n]),
                ];
                loaded.launch(cfg.launch(m, n, clusters), &mut args);
                let got = args[5].to_vec::<f32>();
                if let Some(i) = (0..got.len()).find(|&i| got[i] as f64 != want[i]) {
                    panic!("{} {m}x{n}x{k} ({clusters} clusters): D({}, {}) differs", cfg.name, i / n, i % n);
                }
            }
        }
    }
}

/// The Task Scheduling GEMM in FP16 and BF16, on tile-aligned and ragged problems.
#[test]
fn ts_gemm_on_device() {
    use ts_gemm_simple as g;
    let Some(dev) = blackwell("ts_gemm_on_device") else { return };
    fn check<T: g::IoDtype>(dev: &Device, (m, n, k): (usize, usize, usize)) {
        let (av, bv) = (g::small_ints(m * k, 3), g::small_ints(n * k, 7));
        let to_t = |v: &[f32]| v.iter().map(|&x| T::from_f32(x)).collect::<Vec<T>>();
        let kernel = g::build::<T>(m, n, k, None);
        let mut args = vec![
            KernelArg::buffer(&to_t(&av)),
            KernelArg::buffer(&to_t(&bv)),
            KernelArg::buffer(&vec![T::from_f32(0.0); m * n]),
        ];
        dev.run(&kernel, g::launch(m, n), &mut args);
        let (want, _) = g::reference(&av, &bv, m, n, k);
        for (i, d) in args[2].to_vec::<T>().into_iter().enumerate() {
            assert_eq!(
                d.to_f32(),
                g::round::<T>(want[i] as f32),
                "D[{}, {}] of {m}x{n}x{k} {}",
                i / n,
                i % n,
                T::DTYPE
            );
        }
    }
    for mnk in [(256, 512, 256), (200, 300, 136), (1024, 1024, 512)] {
        check::<F16>(&dev, mnk);
    }
    for mnk in [(128, 256, 64), (130, 264, 72)] {
        check::<BF16>(&dev, mnk);
    }
}
