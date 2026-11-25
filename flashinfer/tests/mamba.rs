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

/// `dst_state_batch_indices`: one write slot per sequence (one token, in place for some) and
/// per token (several), with pad entries skipped.
#[test]
fn dst_state_batch_indices_match_reference() {
    let base = Options { has_indices: true, has_dst: true, ..Options::default() };
    for (tokens, opts) in [(1usize, base), (1, Options { update_state: false, ..base }), (3, base)] {
        for num_sms in [2, 100] {
            let mut p = Problem::new(Dtype::F16, StateDtype::F32, Options { tokens, ..opts }, (6, 4, 16, 128, 2), 17);
            p.num_sms = num_sms;
            simulate(&p);
        }
    }
    // Destinations without source indices (the read slot is the sequence).
    let o = Options { has_dst: true, tokens: 2, ..Options::default() };
    simulate(&Problem::new(Dtype::BF16, StateDtype::BF16, o, (4, 2, 8, 64, 1), 3));
}

/// Packed varlen sequences (some empty, some shorter than `tokens`), with and without the
/// optional inputs.
#[test]
fn varlen_matches_reference() {
    let base = Options { varlen: true, tokens: 4, ..Options::default() };
    for opts in [
        base,
        Options { has_z: true, has_indices: true, ..base },
        Options { has_indices: true, has_dst: true, ..base },
        Options { intermediate: true, ..base },
    ] {
        for num_sms in [2, 100] {
            let mut p = Problem::new(Dtype::F16, StateDtype::BF16, opts, (7, 4, 16, 128, 2), 23);
            p.num_sms = num_sms;
            assert!(p.seqlens.contains(&0) && p.seqlens.iter().any(|&n| n < 4), "{:?}", p.seqlens);
            simulate(&p);
        }
    }
}

/// `num_accepted_tokens`: the state is read from column `accepted - 1` of the 2D indices.
#[test]
fn num_accepted_tokens_match_reference() {
    let base = Options { has_indices: true, has_accepted: true, tokens: 3, ..Options::default() };
    for opts in [
        base,
        Options { varlen: true, ..base },
        Options { has_dst: true, ..base },
        Options { intermediate: true, ..base },
    ] {
        let mut p = Problem::new(Dtype::BF16, StateDtype::F32, opts, (6, 4, 16, 64, 2), 29);
        p.num_sms = 2;
        assert!(p.accepted.contains(&0) && p.accepted.contains(&3));
        simulate(&p);
    }
}

/// The MTP intermediate-state cache, indexed by the state slot or by its own indices.
#[test]
fn intermediate_states_match_reference() {
    let base = Options { intermediate: true, tokens: 4, ..Options::default() };
    for opts in [
        base,
        Options { istate_indices: true, has_indices: true, ..base },
        Options { has_z: true, update_state: false, ..base },
    ] {
        for num_sms in [2, 100] {
            let mut p = Problem::new(Dtype::F16, StateDtype::F16, opts, (6, 4, 16, 128, 2), 31);
            p.num_sms = num_sms;
            simulate(&p);
        }
    }
}

/// The block-scaled int16 state, alone and with the destination and cache modes.
#[test]
fn int16_state_matches_reference() {
    let base = Options { int16: true, ..Options::default() };
    for (opts, tokens) in [
        (base, 1usize),
        (Options { has_indices: true, has_dst: true, ..base }, 1),
        (Options { has_z: true, ..base }, 3),
        (Options { intermediate: true, istate_indices: true, has_indices: true, ..base }, 3),
        (Options { varlen: true, has_dst: true, ..base }, 3),
    ] {
        for num_sms in [2, 100] {
            let mut p = Problem::new(Dtype::BF16, StateDtype::F16, Options { tokens, ..opts }, (5, 4, 16, 128, 2), 37);
            p.num_sms = num_sms;
            simulate(&p);
        }
    }
}

/// The per-token stores (the cache, the destinations) are vector accesses like the state's.
#[test]
fn per_token_stores_are_vectors() {
    let Some(tc) = Toolchain::discover() else { return };
    let base = Options { tokens: 6, ..Options::default() };
    for opts in [
        Options { intermediate: true, istate_indices: true, ..base },
        Options { has_indices: true, has_dst: true, ..base },
        Options { int16: true, ..base },
    ] {
        let p = Problem::new(Dtype::BF16, StateDtype::F16, opts, (16, 128, 64, 128, 8), 1);
        let ptx = p.kernel().compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap().ptx().to_string();
        if let Some(dir) = std::env::var_os("DUMP_PTX") {
            std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", p.kernel().name())), &ptx).unwrap();
        }
        let vec_stores = ptx.matches("st.global.v").count();
        let scalar_stores = ptx.matches("st.global.b16").count() + ptx.matches("st.global.b32").count();
        println!("{opts:?}: {vec_stores} vector stores, {scalar_stores} scalar");
        assert!(vec_stores >= 6, "{opts:?}");
    }
}
