// SPDX-License-Identifier: Apache-2.0

//! `silu_and_mul`, `gelu_and_mul` and `gelu_tanh_and_mul` on the simulator (with cute-rs's
//! race and alignment checks) against the f64 reference, for Float16 and BFloat16, and the
//! PTX they lower to.

use cute::prelude::*;
use flashinfer::activation::*;

const ACTS: [Act; 3] = [Act::Silu, Act::Gelu, Act::GeluTanh];

/// One block per row (`tokens >= 2 * SMs`), a partial last vector round (`d = 2056`), and
/// a row shorter than the block (`d = 64`).
#[test]
fn one_block_per_row_matches_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for act in ACTS {
            for (tokens, d) in [(5usize, 64usize), (3, 2056), (4, 4096)] {
                let mut p = Problem::new(dtype, act, tokens, d, 5);
                p.num_sms = 2;
                assert_eq!(p.plan().split, 1);
                simulate(&p);
            }
        }
    }
}

/// Rows split over several blocks, with and without two vectors loaded together, and a
/// split that leaves some threads with no vector at all.
#[test]
fn split_rows_match_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for act in ACTS {
            for (tokens, d, sms) in [(1usize, 4096usize, 4usize), (1, 14336, 1), (2, 2056, 5), (3, 1544, 8), (1, 8, 4)]
            {
                let mut p = Problem::new(dtype, act, tokens, d, 9);
                p.num_sms = sms;
                simulate(&p);
            }
        }
    }
    assert_eq!(plan(1, 4096, 4), Plan { threads: 256, split: 2, unroll: 1 });
    assert_eq!(plan(1, 14336, 1), Plan { threads: 256, split: 4, unroll: 2 });
    assert_eq!(plan(100, 4096, 22), Plan { threads: 256, split: 1, unroll: 1 });
    assert_eq!(plan(1, 14336, 22), Plan { threads: 256, split: 7, unroll: 1 });
}

/// 16-byte global accesses and the fast transcendentals.
#[test]
fn lowers_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for act in ACTS {
        let ptx = kernel(Dtype::F16, act, 256, 2)
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
            .unwrap()
            .ptx()
            .to_string();
        assert!(ptx.contains("ld.global.v4.b32"), "{ptx}");
        assert!(ptx.contains("st.global.v4.b32"), "{ptx}");
        match act {
            Act::Silu => assert!(ptx.contains("ex2.approx.ftz"), "{ptx}"),
            Act::GeluTanh => assert!(ptx.contains("tanh.approx.f32"), "{ptx}"),
            Act::Gelu => {}
        }
    }
}
