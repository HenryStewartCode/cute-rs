// SPDX-License-Identifier: Apache-2.0

//! Gated Delta Net decode on the simulator (with cute-rs's race and alignment checks)
//! against the f64 reference: dtypes, GVA head ratios, split and unsplit launches, state
//! pools with padding.

use cute::prelude::*;
use flashinfer::gdn::*;

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
fn without_l2norm_matches_reference() {
    let p = Problem::new(Dtype::BF16, StateDtype::F32, Options { l2norm: false, ..Options::default() }, SMALL, 5);
    simulate(&p);
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

/// Qwen3-Next's head layout (16 q/k heads, 32 v heads) at a smaller V, a K of 256 (two
/// vectors per lane) and one of 64 (half-width vectors).
#[test]
fn shapes_match_reference() {
    for (shape, state) in [
        ((2usize, 4usize, 8usize, 128usize, 32usize), StateDtype::F32),
        ((2, 2, 2, 256, 8), StateDtype::F32),
        ((2, 2, 4, 256, 8), StateDtype::BF16),
        ((1, 1, 2, 64, 12), StateDtype::F32),
    ] {
        let mut p = Problem::new(Dtype::BF16, state, Options::default(), shape, 9);
        p.num_sms = 1;
        simulate(&p);
    }
}

#[test]
fn lowers_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else { return };
    let p = Problem::new(Dtype::BF16, StateDtype::F32, Options::default(), (32, 16, 32, 128, 128), 1);
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

/// FlashInfer's `gated_delta_rule_mtp`: several tokens per sequence with the state in
/// registers, the intermediate-state cache (also with a cache longer than the tokens) and
/// `disable_state_update`, with pools and padding.
#[test]
fn multi_token_matches_reference() {
    let base = Options::default();
    for (opts, state) in [
        (Options { tokens: 3, ..base }, StateDtype::F32),
        (Options { tokens: 4, cache_steps: 4, ..base }, StateDtype::F32),
        (Options { tokens: 2, cache_steps: 5, update_state: false, ..base }, StateDtype::BF16),
        (Options { tokens: 4, cache_steps: 4, has_indices: true, ..base }, StateDtype::F32),
        (Options { tokens: 3, has_indices: true, has_out_indices: true, cache_steps: 3, ..base }, StateDtype::F16),
        (Options { tokens: 5, l2norm: false, update_state: false, ..base }, StateDtype::F32),
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
    let opts = Options { tokens: 3, cache_steps: 3, ..Options::default() };
    let mut p = Problem::new(Dtype::BF16, StateDtype::F32, opts, (2, 2, 2, 128, 64), 21);
    p.plan_override = Some((64, 8)); // 16 rows per warp
    simulate(&p);
}

/// Prefill (`chunk_gated_delta_rule`): varlen sequences (three chunks with a partial last
/// one, a short one, an empty one), GVA, initial and final state, every dtype pair.
#[test]
fn prefill_dtypes_match_reference() {
    for input in [Dtype::F16, Dtype::BF16] {
        for state in [StateDtype::F16, StateDtype::BF16, StateDtype::F32] {
            let p = PrefillProblem::new(input, state, PrefillOptions::default(), &[150, 0, 5], (1, 2, 64), 3);
            let (es, eo) = simulate_prefill(&p);
            println!("prefill {} {}: state {es:.2e} out {eo:.2e}", input.name(), state.name());
        }
    }
    // Two value bands.
    let p = PrefillProblem::new(Dtype::BF16, StateDtype::F32, PrefillOptions::default(), &[80], (1, 1, 128), 9);
    simulate_prefill(&p);
}

/// Prefill options: no L2 norm, no initial state, no final state, state pools with padding
/// and separate output rows; two q/k heads.
#[test]
fn prefill_options_match_reference() {
    let base = PrefillOptions::default();
    for opts in [
        PrefillOptions { l2norm: false, ..base },
        PrefillOptions { has_initial_state: false, ..base },
        PrefillOptions { output_final_state: false, ..base },
        PrefillOptions { has_indices: true, ..base },
        PrefillOptions { has_indices: true, has_out_indices: true, ..base },
    ] {
        let p = PrefillProblem::new(Dtype::F16, StateDtype::F32, opts, &[64, 3, 70], (2, 4, 64), 5);
        let (es, eo) = simulate_prefill(&p);
        println!("prefill {opts:?}: state {es:.2e} out {eo:.2e}");
    }
}

/// FlashInfer's non-transposed `[K, V]` state (`gated_delta_rule_decode`), all state dtypes,
/// single- and multi-token, with pools and padding, split and unsplit launches.
#[test]
fn kv_layout_matches_reference() {
    let base = Options { kv_layout: true, ..Options::default() };
    for (opts, state, shape) in [
        (base, StateDtype::F32, (3usize, 2usize, 4usize, 128usize, 64usize)),
        (base, StateDtype::BF16, (3, 2, 4, 128, 64)),
        (base, StateDtype::F16, (2, 2, 2, 128, 128)),
        (Options { tokens: 3, ..base }, StateDtype::F32, (3, 2, 4, 128, 64)),
        // The tile kernel (f32 state, K = 128, few states): pools with padding, no l2 norm, no write-back.
        (Options { has_indices: true, has_out_indices: true, ..base }, StateDtype::F32, (7, 2, 4, 128, 64)),
        (Options { l2norm: false, ..base }, StateDtype::F32, (3, 2, 4, 128, 32)),
        (Options { update_state: false, ..base }, StateDtype::F32, (3, 2, 4, 128, 64)),
        (Options { has_indices: true, has_out_indices: true, ..base }, StateDtype::BF16, (7, 2, 4, 128, 64)),
        (Options { tokens: 2, update_state: false, ..base }, StateDtype::F32, (3, 2, 4, 64, 32)),
        (Options { tokens: 2, has_indices: true, ..base }, StateDtype::F32, (6, 2, 2, 256, 32)),
    ] {
        for num_sms in [1, 100] {
            let mut p = Problem::new(Dtype::BF16, state, opts, shape, 31);
            p.num_sms = num_sms;
            simulate(&p);
        }
    }
}

/// FlashInfer's per-token pool scatter (`ssm_state_indices`): every token's state to its own
/// slot, the read slot left alone; with a pool (padding), and in the `[K, V]` layout.
#[test]
fn per_token_scatter_matches_reference() {
    let base = Options { per_token_slots: true, ..Options::default() };
    for (opts, state) in [
        (Options { tokens: 3, ..base }, StateDtype::F32),
        (Options { tokens: 4, has_indices: true, ..base }, StateDtype::BF16),
        (Options { tokens: 2, has_indices: true, kv_layout: true, ..base }, StateDtype::F32),
        (Options { tokens: 3, kv_layout: true, ..base }, StateDtype::F16),
    ] {
        for num_sms in [1, 100] {
            let mut p = Problem::new(Dtype::F16, state, opts, (6, 2, 4, 128, 64), 37);
            p.num_sms = num_sms;
            simulate(&p);
        }
    }
}
