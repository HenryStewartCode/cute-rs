// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/efc/custom_epilogue_dense_gemm.py
//   (its epilogue `epilogue(efc_config, C, D, alpha, beta, X, x_factor, Y)`, the default
//   types and scalars, and the tensors of `run`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The custom EFC GEMM (`custom_epilogue_dense_gemm.py`): an
//! epilogue with two read and two written tensors,
//!
//! ```text
//! Y = acc
//! D = relu(acc * alpha + C * beta) + X * x_factor
//! ```
//!
//! on [`DenseGemmEfc`](efc::dense_gemm::DenseGemmEfc). All the loads come before the
//! stores. The tensors are Float32 by default, as in the example; the types can be mixed.
//!
//! Run with `cargo run --release --example custom_epilogue_dense_gemm`.

#[allow(dead_code)]
#[path = "efc/mod.rs"]
pub mod efc;

use cute::prelude::*;
use cute::utils::LayoutEnum;
use efc::dense_gemm::{DenseGemmEfc, EFC_GEMM_1CTA, EFC_GEMM_2CTA_MULTICAST, EfcRun, GemmConfig, small_ints};
use efc::{Efc, EfcError, HostArgs};

/// The element types of C, D, X and Y.
#[derive(Clone, Copy, Debug)]
pub struct Dtypes {
    pub c: DType,
    pub d: DType,
    pub x: DType,
    pub y: DType,
}

/// The example's defaults: Float32 everywhere.
pub const FLOAT32: Dtypes = Dtypes { c: DType::F32, d: DType::F32, x: DType::F32, y: DType::F32 };

/// The epilogue, with every tensor in `layout`.
pub fn epilogue(t: Dtypes, layout: LayoutEnum) -> Result<Efc, EfcError> {
    Efc::builder()
        .tensor_with_layout("C", t.c, layout)
        .tensor_with_layout("D", t.d, layout)
        .scalar("alpha")
        .scalar("beta")
        .tensor_with_layout("X", t.x, layout)
        .scalar("x_factor")
        .tensor_with_layout("Y", t.y, layout)
        .build(|efc_config| {
            // All the loads happen before any store.
            // Store the accumulator to Y
            efc_config.store("Y", efc_config.accum());
            // Compute the result with alpha, beta scaling and X factor
            let result = (efc_config.accum() * efc_config.scalar("alpha")
                + efc_config.load("C") * efc_config.scalar("beta"))
            .relu()
                + efc_config.load("X") * efc_config.scalar("x_factor");
            efc_config.store("D", result);
        })
}

/// Simulate `cfg` with the custom epilogue on an `m x n x k` problem (`alpha` 1, `beta` 2,
/// `x_factor` 3; C and X small integers), under `opts` and the schedules `seeds`.
pub fn run(
    cfg: GemmConfig,
    (m, n, k): (usize, usize, usize),
    t: Dtypes,
    layout: LayoutEnum,
    max_active_clusters: u32,
    opts: &SimOptions,
    seeds: &[u64],
) -> EfcRun {
    let gemm = DenseGemmEfc::new(cfg, epilogue(t, layout).unwrap());
    let mut inputs = HostArgs::default();
    for (name, v) in [("alpha", 1.0), ("beta", 2.0), ("x_factor", 3.0)] {
        inputs.scalars.insert(name.into(), v);
    }
    inputs.tensors.insert("C".into(), small_ints(m * n, 44));
    inputs.tensors.insert("X".into(), small_ints(m * n, 55));
    inputs.tensors.insert("D".into(), vec![-7.0; m * n]);
    inputs.tensors.insert("Y".into(), vec![-7.0; m * n]);
    gemm.simulate((m, n, k), &inputs, max_active_clusters, opts, seeds)
}

fn main() {
    for (cfg, mnk) in [(EFC_GEMM_1CTA, (200, 136, 64)), (EFC_GEMM_2CTA_MULTICAST, (264, 264, 64))] {
        println!("Running the custom EFC GEMM {} {mnk:?}", cfg.name);
        let run = run(cfg, mnk, FLOAT32, LayoutEnum::ROW_MAJOR, 2, &SimOptions::default(), &[]);
        run.assert_close(cfg.name, 0.0, 0.0);
        println!("{}", run.report);
        println!("PASS");
    }
}
