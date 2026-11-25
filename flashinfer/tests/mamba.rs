// SPDX-License-Identifier: Apache-2.0

//! `selective_state_update` on the simulator (with cute-rs's race and alignment checks)
//! against the f64 reference: every dtype pair, the tiled and untiled launches, and the
//! optional inputs.

use cute::prelude::*;
use flashinfer::mamba::*;

const SMALL: (usize, usize, usize, usize, usize) = (3, 4, 8, 64, 2);

#[test]
fn dtypes_match_reference() {
    for input in [Dtype::F16, Dtype::BF16] {
        for state in [StateDtype::F16, StateDtype::BF16, StateDtype::F32] {
            let mut p = Problem::new(input, state, Options::default(), SMALL, 3);
            p.num_sms = 2; // 12 >= 4 blocks: untiled
            assert_eq!(p.plan().rows_per_block, 8);
            simulate(&p);
            p.num_sms = 100; // tiled
            assert_eq!(p.plan().rows_per_block, 4);
            simulate(&p);
        }
    }
}

#[test]
fn optional_inputs_match_reference() {
    let base = Options::default();
    for opts in [
        Options { has_z: true, ..base },
        Options { has_d: false, has_dt_bias: false, dt_softplus: false, ..base },
        Options { has_indices: true, ..base },
        Options { has_indices: true, has_z: true, ..base },
        Options { update_state: false, ..base },
    ] {
        for num_sms in [2, 100] {
            let mut p = Problem::new(Dtype::BF16, StateDtype::BF16, opts, (6, 4, 16, 128, 4), 5);
            p.num_sms = num_sms;
            simulate(&p);
        }
    }
}

/// `dstate` 128 and 256 (one vector of 8 or 4 per lane, two or more rounds), and `dim` 64
/// walked four rows at a time.
#[test]
fn shapes_match_reference() {
    for (shape, state) in [
        ((2usize, 2usize, 64usize, 128usize, 1usize), StateDtype::F16),
        ((2, 2, 16, 256, 2), StateDtype::BF16),
        ((2, 2, 16, 256, 2), StateDtype::F32),
        ((1, 2, 12, 32, 1), StateDtype::F32),
    ] {
        let mut p = Problem::new(Dtype::F16, state, Options::default(), shape, 9);
        p.num_sms = 1;
        simulate(&p);
    }
}

#[test]
fn lowers_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else { return };
    let p = Problem::new(Dtype::BF16, StateDtype::BF16, Options::default(), (128, 128, 64, 128, 8), 1);
    let ptx = p
        .kernel()
        .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
        .unwrap_or_else(|e| panic!("{e}"))
        .ptx()
        .to_string();
    assert!(ptx.contains("ld.global.v4.b16"), "state loads are 8 bytes per lane");
    assert!(ptx.contains("st.global.v2.b32"));
    assert!(!ptx.contains("bar.sync"), "no shared memory, no barrier");
}

/// The multi-token update: `tokens` steps of the recurrence with the state in registers.
#[test]
fn multi_token_matches_reference() {
    let base = Options::default();
    for (tokens, opts, state) in [
        (3usize, base, StateDtype::BF16),
        (4, Options { has_z: true, has_indices: true, ..base }, StateDtype::F32),
        (2, Options { has_d: false, dt_softplus: false, update_state: false, ..base }, StateDtype::F16),
    ] {
        for num_sms in [2, 100] {
            let mut p = Problem::new(Dtype::F16, state, Options { tokens, ..opts }, (5, 4, 16, 128, 2), 13);
            p.num_sms = num_sms;
            simulate(&p);
        }
    }
}

/// Eight rows in flight per warp (multi-token, `dim` 64 untiled).
#[test]
fn eight_rows_in_flight_match_reference() {
    let mut p = Problem::new(
        Dtype::BF16,
        StateDtype::BF16,
        Options { tokens: 2, ..Options::default() },
        (2, 2, 128, 128, 1),
        21,
    );
    p.num_sms = 1;
    assert_eq!(p.plan().unroll, 8);
    simulate(&p);
}
