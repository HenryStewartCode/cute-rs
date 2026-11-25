// SPDX-License-Identifier: Apache-2.0

//! Kimi Delta Attention decode on the simulator (with cute-rs's race and alignment checks)
//! against the f64 reference: dtypes, gate modes, GVA head ratios, split and unsplit launches,
//! state pools with padding, multi-token decode and per-token checkpoint slots.

use cute::prelude::*;
use flashinfer::kda::*;

/// (batch, hq, hv, k, v)
const SMALL: (usize, usize, usize, usize, usize) = (3, 2, 4, 128, 16);

#[test]
fn dtypes_match_reference() {
    for input in [Dtype::F16, Dtype::BF16] {
        for state in [StateDtype::F16, StateDtype::BF16, StateDtype::F32] {
            let mut p = Problem::new(input, state, Options::default(), SMALL, 3);
            p.num_sms = 1; // 12 states >= 4 blocks: untiled
            assert_eq!(p.plan().rows_per_block, 16);
            simulate(&p);
            p.num_sms = 100; // split
            assert!(p.plan().rows_per_block < 16);
            simulate(&p);
        }
    }
}

#[test]
fn gate_modes_match_reference() {
    let base = Options::default();
    for opts in [
        Options { gate: Gate::Softplus, ..base },
        Options { gate: Gate::Softplus, has_dt_bias: true, beta_is_logit: true, ..base },
        Options { gate: Gate::LowerBound, has_dt_bias: true, ..base },
        Options { gate: Gate::LowerBound, l2norm: false, beta_is_logit: true, ..base },
    ] {
        for state in [StateDtype::F32, StateDtype::BF16] {
            let mut p = Problem::new(Dtype::BF16, state, opts, SMALL, 5);
            p.num_sms = 1;
            simulate(&p);
        }
    }
}

#[test]
fn pool_indices_match_reference() {
    let base = Options::default();
    for opts in [Options { has_indices: true, ..base }, Options { has_indices: true, has_out_indices: true, ..base }] {
        for state in [StateDtype::F32, StateDtype::BF16] {
            for num_sms in [1, 100] {
                let mut p = Problem::new(Dtype::F16, state, opts, (7, 2, 4, 128, 16), 11);
                p.num_sms = num_sms;
                simulate(&p);
            }
        }
    }
}

/// Kimi-Linear's head count (equal q/k and v heads), a GVA ratio, K of 256 (two vectors a lane)
/// and 64 (half-width vectors).
#[test]
fn shapes_match_reference() {
    for (shape, state, gate) in [
        ((2usize, 4usize, 4usize, 128usize, 32usize), StateDtype::BF16, Gate::Softplus),
        ((2, 2, 4, 128, 8), StateDtype::F32, Gate::Precomputed),
        ((2, 2, 2, 256, 8), StateDtype::F32, Gate::LowerBound),
        ((2, 2, 4, 256, 8), StateDtype::BF16, Gate::Softplus),
        ((1, 1, 2, 64, 12), StateDtype::F32, Gate::Precomputed),
    ] {
        let opts = Options { gate, has_dt_bias: gate != Gate::Precomputed, ..Options::default() };
        let mut p = Problem::new(Dtype::BF16, state, opts, shape, 9);
        p.num_sms = 1;
        simulate(&p);
    }
}

#[test]
fn lowers_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else { return };
    let opts = Options { gate: Gate::Softplus, has_dt_bias: true, ..Options::default() };
    let p = Problem::new(Dtype::BF16, StateDtype::F32, opts, (32, 16, 16, 128, 128), 1);
    let ptx = p
        .kernel()
        .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
        .unwrap_or_else(|e| panic!("{e}"))
        .ptx()
        .to_string();
    assert!(ptx.contains("ld.global.v4.b32"), "f32 state loads are 16 bytes per lane");
    assert!(ptx.contains("st.global.v4.b32"));
    assert!(!ptx.contains("bar.sync"), "no shared memory, no barrier");
}

/// Several tokens per sequence with the state in registers, `disable_state_update`, pools,
/// padding, and per-token checkpoint slots.
#[test]
fn multi_token_matches_reference() {
    let base = Options { gate: Gate::LowerBound, has_dt_bias: true, beta_is_logit: true, ..Options::default() };
    for (opts, state) in [
        (Options { tokens: 3, ..base }, StateDtype::F32),
        (Options { tokens: 2, update_state: false, ..base }, StateDtype::BF16),
        (Options { tokens: 4, has_indices: true, ..base }, StateDtype::F32),
        (Options { tokens: 3, has_indices: true, has_out_indices: true, ..base }, StateDtype::F16),
        (Options { tokens: 4, has_indices: true, per_token_slots: true, ..base }, StateDtype::BF16),
        (Options { tokens: 3, has_indices: true, per_token_slots: true, update_state: false, ..base }, StateDtype::F32),
        (Options { tokens: 5, l2norm: false, update_state: false, gate: Gate::Precomputed, ..base }, StateDtype::F32),
    ] {
        for num_sms in [1, 100] {
            let mut p = Problem::new(Dtype::F16, state, opts, (6, 2, 4, 128, 16), 17);
            p.num_sms = num_sms;
            simulate(&p);
        }
    }
}

/// Eight rows in flight per warp.
#[test]
fn eight_rows_in_flight_match_reference() {
    let opts = Options { tokens: 3, gate: Gate::Softplus, ..Options::default() };
    let mut p = Problem::new(Dtype::BF16, StateDtype::F32, opts, (2, 2, 2, 128, 64), 21);
    p.plan_override = Some((64, 8)); // 16 rows per warp
    simulate(&p);
}
