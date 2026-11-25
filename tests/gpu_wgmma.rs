// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host- and device-side parts of the warpgroup MMAs
//! that run on any GPU: the shared-memory matrix descriptors `make_fragment_a/b` builds,
//! held to the PTX ISA's canonical layouts worked out by hand (the MMAs themselves need
//! sm_90a: `tests/wgmma.rs` simulates them and assembles them for sm_90a). Needs the
//! `cuda` feature; skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_wgmma
//! ```

mod feature_kernels;

use cute::cuda::CudaContext;
use cute::prelude::*;

#[test]
fn descriptors_on_the_gpu() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let case = feature_kernels::wgmma_descriptors();
    let mut args = feature_kernels::inputs_with_len(&case.kernel, true, 256);
    case.kernel.run_cuda(&ctx, &tc, case.cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    (case.check)(&args);
}
