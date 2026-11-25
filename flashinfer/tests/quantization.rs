// SPDX-License-Identifier: Apache-2.0

//! `packbits` and `segment_packbits` on the simulator (with cute-rs's race checks) against
//! the exact `numpy.packbits` semantics.

use flashinfer::quantization::*;

#[test]
fn packbits_matches_reference() {
    for order in [Order::Big, Order::Little] {
        // Whole and partial bytes, partial blocks, and more than one block (2048 bits each).
        for n in [1usize, 7, 8, 9, 100, 2048, 2049, 5000] {
            let p = Problem::new(&[n], n as u64);
            assert_eq!(simulate_pack(&p, order), 0, "{order:?} n {n}");
        }
    }
}

#[test]
fn segment_packbits_matches_reference() {
    for order in [Order::Big, Order::Little] {
        // The docs' example, unaligned starts, an empty segment, segments past one pass.
        for lens in [vec![4usize, 3, 4], vec![5, 0, 13, 8], vec![1, 3000, 17, 2048, 2049], vec![8, 8, 8]] {
            let p = Problem::new(&lens, 3);
            assert_eq!(simulate_segment(&p, order), 0, "{order:?} {lens:?}");
        }
    }
}

#[test]
fn doc_example() {
    // x = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1], segments [0, 4, 7, 11]:
    // ['0b10110000', '0b100000', '0b11010000'].
    let bits = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1].map(|b| b == 1);
    let p = Problem { bits: bits.to_vec(), indptr: vec![0, 4, 7, 11] };
    assert_eq!(p.reference(Order::Big), vec![0b10110000, 0b00100000, 0b11010000]);
    assert_eq!(simulate_segment(&p, Order::Big), 0);
}

fn modes() -> [QuantMode; 3] {
    [
        QuantMode { column_major: false, ue8m0: false },
        QuantMode { column_major: true, ue8m0: false },
        QuantMode { column_major: true, ue8m0: true },
    ]
}

#[test]
fn group_quant_matches_reference() {
    // Every output type and input type, group 128 (FlashInfer's), all three scale layouts;
    // a token count that leaves a partial block (16 groups per block) and one that does not.
    for dtype in [Dtype::F16, Dtype::BF16] {
        for out in [Out8::E4M3, Out8::E5M2, Out8::Int8] {
            for mode in modes() {
                for (tokens, hidden) in [(1usize, 128usize), (3, 384), (5, 512), (4, 4 * 128 * 4)] {
                    let p = GroupQuantProblem::new(dtype, out, tokens, hidden, 128, mode, 7);
                    assert_eq!(simulate_group_quant(&p), 0, "{dtype:?} {out:?} {mode:?} {tokens}x{hidden}");
                }
            }
        }
    }
}

#[test]
fn group_quant_other_group_sizes() {
    for gs in [64usize, 256] {
        for mode in modes() {
            let p = GroupQuantProblem::new(Dtype::BF16, Out8::E4M3, 7, gs * 3, gs, mode, 11);
            assert_eq!(simulate_group_quant(&p), 0, "group {gs} {mode:?}");
        }
    }
}

#[test]
fn group_quant_reference_round_trips() {
    // The reference itself: dequantizing stays within about half a step of the type
    // (in scale units: the top binade of E4M3 has steps of 32, of E5M2 8192).
    let m = QuantMode { column_major: false, ue8m0: false };
    for (out, bound) in [(Out8::E4M3, 16.0f32), (Out8::E5M2, 4096.0), (Out8::Int8, 0.5)] {
        let p = GroupQuantProblem::new(Dtype::F16, out, 4, 256, 128, m, 5);
        let (q, s) = p.reference();
        let err = p.dequant_error(&q, &s);
        assert!(err <= bound * 1.001, "{out:?}: {err} > {bound}");
    }
}
