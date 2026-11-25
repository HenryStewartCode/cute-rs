// SPDX-License-Identifier: Apache-2.0

//! `merge_state`, `merge_state_in_place` and `merge_states` on the simulator (with cute-rs's
//! race and alignment checks) against the f64 reference, for Float16 and BFloat16, with
//! empty states (`s = -inf`) in the data, and the PTX they lower to.

use cute::prelude::*;
use flashinfer::cascade::*;

#[test]
fn merge_state_matches_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (seq, heads, hd) in [(3usize, 2usize, 128usize), (2, 5, 64), (2, 1, 256)] {
            let p = Problem::new(dtype, seq, heads, hd, 2, 3);
            simulate(&p, Op::MergeState);
            simulate(&p, Op::InPlace { masked: false });
            simulate(&p, Op::InPlace { masked: true });
        }
    }
}

/// Fewer sets than positions: the simple kernel, with an odd and an even number of sets.
#[test]
fn merge_states_matches_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (seq, heads, hd, sets) in
            [(5usize, 3usize, 128usize, 3usize), (5, 2, 64, 4), (3, 2, 128, 1), (4, 1, 128, 0)]
        {
            let p = Problem::new(dtype, seq, heads, hd, sets, 7);
            assert!(!p.large());
            simulate(&p, Op::MergeStates);
        }
    }
}

/// At least as many sets as positions: the `cp.async` pipeline, with a partial last round
/// of `128 / bdx` sets, fewer sets than stages, and more rounds than stages.
#[test]
fn merge_states_large_matches_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (seq, heads, hd, sets) in
            [(1usize, 2usize, 128usize, 3usize), (2, 2, 128, 21), (1, 1, 64, 70), (2, 2, 256, 2), (1, 2, 128, 100)]
        {
            let p = Problem::new(dtype, seq, heads, hd, sets, 11);
            assert!(p.large());
            simulate(&p, Op::MergeStates);
        }
    }
}

/// Merging all-empty states gives 0 and `-inf`, not NaN.
#[test]
fn empty_states_stay_empty() {
    let mut p = Problem::new(Dtype::F16, 2, 2, 128, 2, 1);
    p.s.iter_mut().for_each(|s| *s = f32::NEG_INFINITY);
    for op in [Op::MergeState, Op::MergeStates] {
        let (v, s) = (0.0f64, 0.0f64);
        let (ev, es) = simulate(&p, op);
        assert!(ev == v && es == s);
    }
}

/// 16-byte global accesses and the fast transcendentals.
#[test]
fn lowers_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
    for k in [
        merge_state_kernel(Dtype::F16, 128, 32),
        merge_state_in_place_kernel(Dtype::F16, 128, 32, true),
        merge_states_kernel(Dtype::F16, 128, 32),
        merge_states_large_kernel(Dtype::F16, 128),
    ] {
        let ptx = k.compile_with(&tc, &opts).unwrap().ptx().to_string();
        assert!(ptx.contains("ld.global.v4.b32") || ptx.contains("cp.async.cg.shared.global"), "{}: {ptx}", k.name());
        assert!(ptx.contains("ex2.approx.ftz"), "{}: {ptx}", k.name());
        assert!(ptx.contains("lg2.approx.ftz"), "{}: {ptx}", k.name());
    }
}
