//! `make_tiled_copy_c_atom` with atoms of one and of several values.
//!
//! The copy covers the C elements of the first `atom` values of each thread's
//! accumulator fragment, and repeats over the rest of C. For an atom of one
//! value, the repeats of an m16n8 accumulator run over the row block first
//! (rows `g`, `g + 8`) and then over the column offset (`2t`, `2t + 1`), while
//! the fragment runs c0 (row g, col 2t), c1 (col 2t + 1), c2 (row g + 8, col
//! 2t), c3. So `retile(acc)` followed by `copy` pairs fragment slot `i` with
//! repeat `i`, which swaps c1 and c2. `retile` is a port of C++ CuTe's, which
//! states the same precondition (the fragment's values are the copy's first
//! value ids), but the one-value case has not been run against C++. The
//! epilogues use an atom of a full row pair, for which the two orders agree. A
//! one-value copy has to pick the fragment slot for each repeat itself, as the
//! tests below do.

use cute::prelude::*;

const M: usize = 16;
const N: usize = 8;

/// Fills each thread's accumulator fragment with the value of the C element it
/// holds (read through the MMA's own C partition of an input), copies it out
/// through `make_tiled_copy_c_atom(atom, mma)` and checks `out == input`.
fn roundtrip(name: &'static str, bits: usize, row_major: bool) {
    // With one value per call the repeats run row block first, the fragment column offset first.
    let one_value = bits == 32;
    let layout = if row_major { Layout::row_major((M, N)) } else { Layout::col_major((M, N)) };
    let kernel = Kernel::builder(name).block_dim([32, 1, 1]).build(move || {
        let src = param_tensor::<f32>("src", layout.clone());
        let out = param_tensor::<f32>("out", layout.clone());
        let t = thread_idx().x;
        let mma = make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)), Layout::col_major(1));
        let thr = mma.get_slice(t);
        let held = thr.partition_c(&src);
        let acc = thr.partition_fragment_c::<f32, _>(&out);
        for i in 0..acc.size() {
            acc.store(i, held.load(i));
        }
        let c = make_tiled_copy_c_atom(&make_copy_atom::<f32>(CopyOp::Universal { bits }), &mma);
        let dst = c.get_slice(t).partition_d(&out);
        if one_value {
            let pieces = make_fragment_like::<f32, f32>(&dst);
            for r in 0..4 {
                pieces.store(r, acc.load((r % 2) * 2 + r / 2));
            }
            copy(&c, &pieces, &dst);
        } else {
            copy(&c, &c.retile(&acc), &dst);
        }
    });
    let want: Vec<f32> = (0..M * N).map(|k| k as f32 + 1.0).collect();
    let mut args = vec![KernelArg::buffer(&want), KernelArg::buffer(&vec![0.0f32; M * N])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let got = args[1].to_vec::<f32>();
    for k in 0..M * N {
        assert_eq!(got[k], want[k], "{name}: element {k}");
    }
}

#[test]
fn one_value_atom_row_major() {
    roundtrip("c_atom_one_row", 32, true);
}

#[test]
fn one_value_atom_col_major() {
    roundtrip("c_atom_one_col", 32, false);
}

#[test]
fn two_value_atom_row_major() {
    roundtrip("c_atom_two_row", 64, true);
}

/// A two-value atom needs the two values contiguous in memory, so a column-major
/// C is rejected rather than stored wrongly.
#[test]
fn two_value_atom_col_major_is_rejected() {
    let msg = std::panic::catch_unwind(|| roundtrip("c_atom_two_col", 64, false))
        .err()
        .and_then(|e| e.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(msg.contains("contiguous destination"), "{msg}");
}
