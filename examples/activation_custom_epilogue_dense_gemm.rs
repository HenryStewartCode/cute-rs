// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/efc/activation_custom_epilogue_dense_gemm.py
//   (`create_epilogue_function`, its epilogue, and the tensors of `create_arguments`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The activation EFC GEMM (`activation_custom_epilogue_dense_gemm.py`), after the Ada FP8 GEMM epilogue:
//!
//! ```text
//! Aux = (alpha * scale_a * scale_b) * acc + (beta * scale_c) * C + bias
//! D   = activation(Aux)
//! ```
//!
//! for each activation function of [`efc::ACTIVATION_FUNCTIONS`] (leaky ReLU with the
//! negative slope `leaky_relu_alpha`). C, Aux and D are Float16 by default, as in the
//! example. The device's tanh, sigmoid, SiLU and GELU are Float32 formulas (GELU the tanh
//! approximation), the host's PyTorch's (GELU with `erf`), so those are compared within a
//! tolerance.
//!
//! Run with `cargo run --release --example activation_custom_epilogue_dense_gemm`.

#[path = "efc/mod.rs"]
pub mod efc;

use cute::prelude::*;
use efc::dense_gemm::{DenseGemmEfc, EFC_GEMM_1CTA, EfcRun, GemmConfig, small_ints};
use efc::{ACTIVATION_FUNCTIONS, Efc, EfcError, HostArgs};

/// The epilogue with activation `activation_name`. Panics for an unknown activation.
pub fn create_epilogue_function(
    activation_name: &str,
    c_dtype: DType,
    aux_dtype: DType,
    d_dtype: DType,
) -> Result<Efc, EfcError> {
    assert!(ACTIVATION_FUNCTIONS.contains(&activation_name), "unsupported activation `{activation_name}`");
    let activation_name = activation_name.to_string();
    Efc::builder()
        .tensor("C", c_dtype)
        .tensor("Aux", aux_dtype)
        .scalar("alpha")
        .scalar("beta")
        .scalar("bias")
        .scalar("scale_a")
        .scalar("scale_b")
        .scalar("scale_c")
        .tensor("D", d_dtype)
        .scalar("leaky_relu_alpha")
        .build(move |efc_config| {
            let s = |name: &str| efc_config.scalar(name);
            // Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias
            let aux_val = (s("alpha") * s("scale_a") * s("scale_b")) * efc_config.accum()
                + (s("beta") * s("scale_c")) * efc_config.load("C")
                + s("bias");
            efc_config.store("Aux", &aux_val);
            // D = activation(Aux); only leaky_relu takes the extra parameter
            efc_config.store("D", efc_config.activation(&activation_name, &aux_val, s("leaky_relu_alpha")));
        })
}

/// The tolerance `(atol, rtol)` of an activation against the host for Float16 D: exact for
/// the piecewise-linear ones (on data whose values Float16 holds, and a power-of-two
/// slope), one Float16 ulp for the Float32 transcendental formulas, plus 1e-3 for GELU's
/// tanh approximation.
pub fn tolerance(activation_name: &str) -> (f64, f64) {
    match activation_name {
        "identity" | "relu" | "leaky_relu" | "hardswish" => (0.0, 0.0),
        "gelu" => (1e-3, 1.0 / 1024.0),
        _ => (1e-6, 1.0 / 1024.0),
    }
}

/// Simulate `cfg` with activation `activation_name` on an `m x n x k` problem of Float16
/// tensors, with scalars that keep Aux exact in Float16.
pub fn run(
    cfg: GemmConfig,
    activation_name: &str,
    (m, n, k): (usize, usize, usize),
    leaky_relu_alpha: f64,
    max_active_clusters: u32,
    opts: &SimOptions,
) -> EfcRun {
    let efc = create_epilogue_function(activation_name, DType::F16, DType::F16, DType::F16).unwrap();
    let gemm = DenseGemmEfc::new(cfg, efc);
    let mut inputs = HostArgs::default();
    for (name, v) in [
        ("alpha", 0.0625),
        ("beta", 0.5),
        ("bias", 0.25),
        ("scale_a", 1.0),
        ("scale_b", 1.0),
        ("scale_c", 1.0),
        ("leaky_relu_alpha", leaky_relu_alpha),
    ] {
        inputs.scalars.insert(name.into(), v);
    }
    inputs.tensors.insert("C".into(), small_ints(m * n, 33));
    inputs.tensors.insert("Aux".into(), vec![-7.0; m * n]);
    inputs.tensors.insert("D".into(), vec![-7.0; m * n]);
    gemm.simulate((m, n, k), &inputs, max_active_clusters, opts, &[])
}

fn main() {
    for activation in ACTIVATION_FUNCTIONS {
        println!("Running the activation EFC GEMM with {activation}");
        let run = run(EFC_GEMM_1CTA, activation, (128, 136, 64), 0.25, 2, &SimOptions::default());
        let (atol, rtol) = tolerance(activation);
        run.assert_close(activation, atol, rtol);
        println!("PASS");
    }
}
