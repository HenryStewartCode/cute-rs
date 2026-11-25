// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/efc/synthetic_custom_epilogue_dense_gemm.py
//   (`meta_epilogue`, its epilogue computation, and the tensors and arguments of `run`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The synthetic EFC GEMM (`synthetic_custom_epilogue_dense_gemm.py`): an epilogue generated for any number of read and written tensors, on
//! [`DenseGemmEfc`](efc::dense_gemm::DenseGemmEfc).
//!
//! With parameters `alpha`, `beta`, `read_t0..` and `written_t0..`, the epilogue sums
//! `beta + read_ti * alpha`, doubles the sum where it is at least 1 (`where` and
//! `full_like`), then stores `t = t * alpha + read + 5000` to each written tensor in turn,
//! starting from the accumulator. With no read tensor it is `D = alpha * acc + beta + 5000`.
//!
//! The parameter names are generated (the Python's `create_named_epilogue`): the signature
//! is a list here anyway. The example's 8192x8192x8192 default is too large to simulate;
//! `main` simulates small problems and compares with the host evaluation.
//!
//! Run with `cargo run --release --example synthetic_custom_epilogue_dense_gemm`.

#[allow(dead_code)]
#[path = "efc/mod.rs"]
pub mod efc;

use cute::prelude::*;
use efc::dense_gemm::{DenseGemmEfc, EFC_GEMM_1CTA, EFC_GEMM_2CTA, EfcRun, GemmConfig, small_ints};
use efc::{Efc, EfcError, HostArgs};

/// The epilogue of `read_tensors` read and `written_tensors` written tensors of `t_dtype`.
pub fn meta_epilogue(read_tensors: usize, written_tensors: usize, t_dtype: DType) -> Result<Efc, EfcError> {
    let mut sig = Efc::builder().scalar("alpha").scalar("beta");
    for i in 0..read_tensors {
        sig = sig.tensor(&format!("read_t{i}"), t_dtype);
    }
    for i in 0..written_tensors {
        sig = sig.tensor(&format!("written_t{i}"), t_dtype);
    }
    sig.build(move |efc_config| {
        let alpha = efc_config.scalar("alpha");
        let mut read = efc_config.scalar("beta");
        for i in 0..read_tensors {
            read = read + efc_config.load(&format!("read_t{i}")) * &alpha;
        }
        if read_tensors > 0 {
            // Can use some CuTe/PyTorch-like functions exposed under efc_config for
            // portability:
            read = efc_config.where_(&read.lt(1.0), &read, &read * efc_config.full_like(&read, 2.0));
        }
        let mut t = efc_config.accum();
        for i in 0..written_tensors {
            t = t * &alpha + &read + 5000.0;
            efc_config.store(&format!("written_t{i}"), &t);
        }
    })
}

/// Simulate `cfg` with the synthetic epilogue on an `m x n x k` problem: read tensors of
/// small integers, written ones filled with -7.
#[allow(clippy::too_many_arguments)]
pub fn run(
    cfg: GemmConfig,
    (m, n, k): (usize, usize, usize),
    read_tensors: usize,
    written_tensors: usize,
    t_dtype: DType,
    (alpha, beta): (f64, f64),
    max_active_clusters: u32,
    opts: &SimOptions,
) -> EfcRun {
    let gemm = DenseGemmEfc::new(cfg, meta_epilogue(read_tensors, written_tensors, t_dtype).unwrap());
    let mut inputs = HostArgs::default();
    inputs.scalars.insert("alpha".into(), alpha);
    inputs.scalars.insert("beta".into(), beta);
    for i in 0..read_tensors {
        inputs.tensors.insert(format!("read_t{i}"), small_ints(m * n, 10 + i as u64));
    }
    for i in 0..written_tensors {
        inputs.tensors.insert(format!("written_t{i}"), vec![-7.0; m * n]);
    }
    gemm.simulate((m, n, k), &inputs, max_active_clusters, opts, &[])
}

fn main() {
    for (cfg, mnk, reads, writes) in [(EFC_GEMM_1CTA, (200, 136, 64), 2, 3), (EFC_GEMM_2CTA, (264, 136, 64), 1, 1)] {
        println!("Running the synthetic EFC GEMM {} {mnk:?}: {reads} read and {writes} written tensors", cfg.name);
        let run = run(cfg, mnk, reads, writes, DType::F32, (2.0, 1.0), 2, &SimOptions::default());
        run.assert_close(cfg.name, 0.0, 0.0);
        println!("{}", run.report);
        println!("PASS");
    }
}
