//! The SM90 epilogue helpers, the TMA alignment preflight checks and the
//! torch-free GEMM host tensors (roadmap part m9-gemm-host-helpers).

use std::str::FromStr;

use cute::backend::Arch;
use cute::prelude::*;
use cute::utils::gemm::*;
use cute::utils::sm90::{compute_tile_shape_or_override, get_smem_store_op, sm90_get_smem_store_op};
use cute::utils::sm100::*;

const MN: OperandMajorMode = OperandMajorMode::MN;
const K: OperandMajorMode = OperandMajorMode::K;

#[test]
fn rng_is_seeded_and_uniform_enough() {
    let (mut a, mut b, mut c) = (Rng::new(7), Rng::new(7), Rng::new(8));
    let xs: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
    assert_eq!(xs, (0..16).map(|_| b.next_u64()).collect::<Vec<_>>());
    assert_ne!(xs, (0..16).map(|_| c.next_u64()).collect::<Vec<_>>());
    let mut counts = [0usize; 3];
    for _ in 0..3000 {
        counts[a.below(3)] += 1;
    }
    assert!(counts.iter().all(|&n| n > 800), "{counts:?}");
}

#[test]
fn builders_have_the_strides_of_each_majorness() {
    for (major, strides) in [(MN, (1, 6, 6 * 5)), (K, (5, 1, 6 * 5))] {
        for l in [1, 3] {
            let t = create_gemm_tensor(6, 5, l, major, DType::F16, &mut Rng::new(1)).unwrap();
            assert_eq!(t.shape(), (6, 5, l));
            assert_eq!(t.strides(), strides);
            assert_eq!(t.layout, Layout::new((6, 5, l), (strides.0, strides.1, strides.2)));
            let vals = t.data.to_f32();
            assert_eq!(vals.len(), 6 * 5 * l);
            for li in 0..l {
                for k in 0..5 {
                    for m in 0..6 {
                        let x = t.value(m, k, li);
                        assert!([-1.0, 0.0, 1.0].contains(&x));
                        assert_eq!(vals[t.offset(m, k, li)], x, "({m},{k},{li})");
                    }
                }
            }
            assert!(t.values().contains(&-1.0) || l == 1);
        }
    }
}

#[test]
fn builders_cover_the_gemm_element_types() {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::F8E4M3FN, DType::F8E5M2, DType::I8, DType::I32] {
        let t = create_gemm_tensor(8, 4, 2, K, dtype, &mut Rng::new(3)).unwrap();
        for x in t.values() {
            assert!([-1.0, 0.0, 1.0].contains(x), "{dtype}: {x}");
        }
        for v in [-1.0, 0.0, 1.0] {
            assert!(t.values().contains(&v), "{dtype} never draws {v}");
        }
        assert_eq!(t.host_tensor().element_type(), dtype);
    }
    let u = create_gemm_tensor(8, 4, 1, K, DType::U8, &mut Rng::new(3)).unwrap();
    assert!(u.values().iter().all(|x| [0.0, 1.0, 255.0].contains(x)));
    assert!(create_gemm_tensor(8, 4, 1, K, DType::F64, &mut Rng::new(3)).is_err());
}

#[test]
fn same_seed_same_values_in_either_majorness() {
    let a = create_gemm_tensor(7, 9, 2, K, DType::F32, &mut Rng::new(42)).unwrap();
    let b = create_gemm_tensor(7, 9, 2, MN, DType::F32, &mut Rng::new(42)).unwrap();
    assert_eq!(a.values(), b.values());
    assert_ne!(a.data, b.data);
}

#[test]
fn host_tensor_marks_the_leading_dimension_dynamic() {
    let k = create_gemm_tensor(8, 16, 2, K, DType::F16, &mut Rng::new(1)).unwrap();
    assert_eq!(get_gemm_tensor(&k).layout_string(), "(?,?,?):(?{i64},1,?{i64})");
    let mn = create_gemm_tensor(8, 16, 2, MN, DType::F16, &mut Rng::new(1)).unwrap();
    assert_eq!(mn.host_tensor().layout_string(), "(?,?,?):(1,?{i64},?{i64})");
}

#[test]
fn get_gemm_tensors_shapes() {
    let (a, b, d) =
        get_gemm_tensors(4, 6, 8, 2, (K, MN, K), (DType::F16, DType::F16, DType::F32), &mut Rng::new(5)).unwrap();
    assert_eq!((a.shape(), b.shape(), d.shape()), ((4, 8, 2), (6, 8, 2), (4, 6, 2)));
    assert_eq!((a.major, b.major, d.major), (K, MN, K));
    assert_eq!((a.dtype, d.dtype), (DType::F16, DType::F32));
    assert_ne!(a.values(), b.values().get(..a.values().len()).unwrap_or(&[]));
}

#[test]
fn fp4_operands_are_bytes_that_decode_to_minus_one_zero_one() {
    for major in [MN, K] {
        let t = create_gemm_tensor(8, 6, 2, major, DType::F4E2M1FN, &mut Rng::new(9)).unwrap();
        // One byte per logical element (the 2x over-allocation).
        assert_eq!(t.data.to_vec::<u8>().len(), 8 * 6 * 2);
        assert!(t.data.to_vec::<u8>().iter().all(|b| FP4_BYTES.contains(b)));
        assert!(t.values().iter().all(|x| [-1.0, 0.0, 1.0].contains(x)));
        for v in [-1.0, 0.0, 1.0] {
            assert!(t.values().contains(&v));
        }
    }
}

#[test]
fn decode_float4e2m1fn_reads_nibbles_of_the_first_half() {
    // 1 x 8 x 1: eight logical elements; elements 0..8 read bytes 0..4's nibbles.
    let bytes = [0x20u8, 0xA2, 0x0A, 0x76, 0xFF, 0xFF, 0xFF, 0xFF];
    let got = decode_float4e2m1fn(&bytes, (1, 8, 1));
    assert_eq!(got, [0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 4.0, 6.0]);
    // Every code once through the whole LUT.
    let all: Vec<u8> = (0..8u8).map(|i| (2 * i) | ((2 * i + 1) << 4)).chain([0; 8]).collect();
    let got = decode_float4e2m1fn(&all, (1, 16, 1));
    assert_eq!(&got[..], &FP4_E2M1_LUT[..]);
}

#[test]
fn decode_float4e2m1fn_orders_by_l_mn_k() {
    // 2 x 2 x 2: logical p = l*4 + m*2 + k holds nibble p of the bytes taken in (L, MN, K) order.
    let (mn, k, l) = (2, 2, 2);
    // Bytes in logical (mn fastest) order; the reader goes through (L, MN, K) order.
    let mut logical = vec![0u8; 8];
    for li in 0..l {
        for m in 0..mn {
            for kk in 0..k {
                logical[m + mn * (kk + k * li)] = ((li * 4 + m * 2 + kk) as u8) << 1;
            }
        }
    }
    let got = decode_float4e2m1fn(&logical, (mn, k, l));
    for li in 0..l {
        for m in 0..mn {
            for kk in 0..k {
                let p = li * 4 + m * 2 + kk;
                // Brute force: byte p / 2 of the (L, MN, K) order.
                let b = p / 2;
                let (bl, bm, bk) = (b / 4, b / 2 % 2, b % 2);
                let byte = logical[bm + mn * (bk + k * bl)];
                assert_eq!(got[m + mn * (kk + k * li)], FP4_E2M1_LUT[(byte >> (4 * (p % 2)) & 0xF) as usize]);
            }
        }
    }
}

/// The index map evaluated one element at a time, from the formula.
fn brute_offset(m: usize, k: usize, l: usize, vec: usize, mn: usize, kk: usize) -> usize {
    let k_tile = 4 * vec;
    let k_tiles = kk.div_ceil(k_tile);
    let mn_tiles = mn.div_ceil(128);
    (m / 128) * 512 * k_tiles
        + (k / k_tile) * 512
        + 16 * (m % 32)
        + 4 * ((m % 128) / 32)
        + (k / vec) % 4
        + l * 512 * mn_tiles * k_tiles
}

#[test]
fn scale_factors_round_trip_against_the_index_map() {
    for (mn, k, l, vec) in [(128, 64, 1, 16), (100, 70, 2, 16), (130, 33, 2, 32), (257, 129, 3, 16), (5, 3, 1, 16)] {
        let n = scale_factor_len(mn, k, l, vec);
        // Padded shape: MN to 128, ceil(K / vec) to 4.
        assert_eq!(n, mn.div_ceil(128) * 128 * k.div_ceil(vec).div_ceil(4) * 4 * l);
        let packed: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dense = unpack_scale_factors(&packed, vec, mn, k, l);
        for li in 0..l {
            for kk in 0..k {
                for m in 0..mn {
                    let want = brute_offset(m, kk, li, vec, mn, k);
                    assert_eq!(dense[m + mn * (kk + k * li)], want as f32, "({m},{kk},{li}) {mn}x{k}x{l} vec {vec}");
                }
            }
        }
        // Pack is the inverse on the addressed positions.
        let repacked = pack_scale_factors(&dense, vec, mn, k, l);
        assert_eq!(unpack_scale_factors(&repacked, vec, mn, k, l), dense);
        // The map is injective on blocks: distinct (m, k / vec, l) give distinct offsets.
        let mut seen = std::collections::HashSet::new();
        for li in 0..l {
            for kb in 0..k.div_ceil(vec) {
                for m in 0..mn {
                    assert!(seen.insert(scale_factor_offset((m, kb * vec, li), vec, (mn, k))));
                }
            }
        }
    }
}

#[test]
fn create_scale_factor_tensor_values_and_types() {
    for sf_dtype in [DType::F8E4M3FN, DType::F8E8M0FNU, DType::F8E5M3FNU] {
        let (mn, k, l, vec) = (130, 70, 2, 16);
        let sf = create_scale_factor_tensor(mn, k, l, vec, sf_dtype, &mut Rng::new(11)).unwrap();
        assert_eq!(sf.packed.len(), scale_factor_len(mn, k, l, vec));
        assert_eq!(sf.reference.len(), mn * k * l);
        assert!(sf.packed.iter().all(|x| [1.0, 2.0, 4.0].contains(x)));
        for v in [1.0, 2.0, 4.0] {
            assert!(sf.packed.contains(&v));
        }
        // The stored bytes decode back to the f32 values.
        assert_eq!(sf.data.to_f32(), sf.packed, "{sf_dtype}");
        assert_eq!(sf.reference, unpack_scale_factors(&sf.packed, vec, mn, k, l));
    }
}

#[test]
fn reference_gemm_accumulates_in_f64() {
    let a = create_gemm_tensor(5, 7, 2, K, DType::F16, &mut Rng::new(1)).unwrap();
    let b = create_gemm_tensor(3, 7, 2, MN, DType::F16, &mut Rng::new(2)).unwrap();
    let d = reference_gemm(&a, &b);
    assert_eq!(d.len(), 5 * 3 * 2);
    for l in 0..2 {
        for n in 0..3 {
            for m in 0..5 {
                let want: f64 = (0..7).map(|k| a.value(m, k, l) as f64 * b.value(n, k, l) as f64).sum();
                assert_eq!(d[m + 5 * (n + 3 * l)], want);
            }
        }
    }
    // A long reduction of values that f32 would round.
    let big = reference_gemm_values(&[0.1f32; 4], &[1.0; 4], (1, 1, 4, 1));
    assert_eq!(big, [4.0 * 0.1f32 as f64]);
    let mem = to_memory_order(&d, (5, 3, 2), K);
    // D is (5, 3, 2) and K-major: (m, n, l) is at 3 m + n + 15 l.
    assert_eq!(mem[3 * 2 + 1 + 15], d[2 + 5 * 4]);
    assert_eq!(mem[3 * 4 + 2], d[4 + 5 * 2]);
}

#[test]
fn tma_aligned_contiguous_elements() {
    for (dtype, want) in [
        (DType::F32, 4),
        (DType::F16, 8),
        (DType::BF16, 8),
        (DType::F8E4M3FN, 16),
        (DType::I8, 16),
        (DType::F4E2M1FN, 32),
    ] {
        assert_eq!(get_tma_aligned_contiguous_elements(dtype).unwrap(), want, "{dtype}");
    }
    let err = get_tma_aligned_contiguous_elements(DType::F6E2M3FN).unwrap_err();
    assert_eq!(err.to_string(), "[alignment] FP6 operands are refused: their TMA alignment rule is not checked yet");
    assert!(get_tma_aligned_contiguous_elements(DType::F6E3M2FN).is_err());
}

#[test]
fn tma_tensor_alignment_messages() {
    check_tma_tensor_alignment("A TMA load", DType::F16, 64, "K").unwrap();
    let err = check_tma_tensor_alignment("A TMA load", DType::F16, 100, "K").unwrap_err();
    assert_eq!(
        err.to_string(),
        "[alignment] A TMA load: K=100 Float16 elements is not a whole number of 16-byte TMA units \
         (use a multiple of 8)"
    );
    assert!(check_tma_tensor_alignment("A TMA load", DType::F6E2M3FN, 64, "K").is_err());
}

#[test]
fn gemm_tma_alignment_passes_and_fails() {
    use GemmMajor::*;
    let (f16, f32_, fp8) = (DType::F16, DType::F32, DType::F8E4M3FN);
    check_gemm_tma_alignment(128, 256, 64, f16, f16, Some(f32_), K, K, Some(N), "D").unwrap();
    check_gemm_tma_alignment(128, 256, 64, f16, f16, None, K, K, None, "D").unwrap();
    // The contiguous dimensions only: M = 3 is fine for a K-major A.
    check_gemm_tma_alignment(3, 256, 64, f16, f16, None, K, K, None, "D").unwrap();
    let msg = |r: Result<(), cute::testing::CantImplement>| r.unwrap_err().to_string();
    assert_eq!(
        msg(check_gemm_tma_alignment(128, 256, 100, f16, f16, None, K, K, None, "D")),
        "[alignment] A TMA load: K=100 Float16 elements is not a whole number of 16-byte TMA units \
         (use a multiple of 8)"
    );
    assert_eq!(
        msg(check_gemm_tma_alignment(100, 256, 64, f16, f16, None, M, K, None, "D")),
        "[alignment] A TMA load: M=100 Float16 elements is not a whole number of 16-byte TMA units \
         (use a multiple of 8)"
    );
    assert_eq!(
        msg(check_gemm_tma_alignment(128, 20, 64, fp8, fp8, None, K, N, None, "D")),
        "[alignment] B TMA load: N=20 Float8E4M3FN elements is not a whole number of 16-byte TMA units \
         (use a multiple of 16)"
    );
    assert_eq!(
        msg(check_gemm_tma_alignment(128, 258, 64, f16, f16, Some(f32_), K, K, Some(N), "C")),
        "[alignment] C TMA store: N=258 Float32 elements is not a whole number of 16-byte TMA units \
         (use a multiple of 4)"
    );
    assert_eq!(
        msg(check_gemm_tma_alignment(130, 256, 64, f16, f16, Some(f16), K, K, Some(M), "D")),
        "[alignment] D TMA store: M=130 Float16 elements is not a whole number of 16-byte TMA units \
         (use a multiple of 8)"
    );
    assert_eq!(
        msg(check_gemm_tma_alignment(128, 256, 64, DType::F6E2M3FN, f16, None, K, K, None, "D")),
        "[alignment] FP6 operands are refused: their TMA alignment rule is not checked yet"
    );
    // Invalid majors.
    assert_eq!(
        msg(check_gemm_tma_alignment(128, 256, 64, f16, f16, None, N, K, None, "D")),
        "[alignment] a_major must be m or k, not n"
    );
    assert_eq!(
        msg(check_gemm_tma_alignment(128, 256, 64, f16, f16, None, K, M, None, "D")),
        "[alignment] b_major must be n or k, not m"
    );
    assert_eq!(
        msg(check_gemm_tma_alignment(128, 256, 64, f16, f16, Some(f16), K, K, Some(K), "D")),
        "[alignment] d_major must be m or n, not k"
    );
    assert_eq!(
        msg(check_gemm_tma_alignment(128, 256, 64, f16, f16, Some(f16), K, K, None, "D")),
        "[alignment] d_major must be m or n, not None"
    );
    // CuTe DSL's strings.
    assert_eq!(
        msg(check_gemm_tma_alignment_str(128, 256, 64, f16, f16, None, "x", "k", None, "D")),
        "[alignment] a_major must be m or k, not x"
    );
    check_gemm_tma_alignment_str(128, 256, 64, f16, f16, Some(f16), "k", "k", Some("m"), "D").unwrap();
    assert_eq!("m".parse::<GemmMajor>().unwrap(), M);
    assert_eq!(GemmMajor::from_str("K").unwrap(), K);
    assert!("q".parse::<GemmMajor>().is_err());
}

#[test]
fn int8_mma_is_rejected_on_sm_107_only() {
    for target in [Arch::Sm107, Arch::Sm107a, Arch::Sm107f] {
        for dtype in [DType::I8, DType::U8] {
            let err = check_int8_mma_arch(dtype, Some(target)).unwrap_err();
            assert_eq!(err.to_string(), format!("{target} has no tcgen05 MMA for Int8 or Uint8 operands"));
        }
        // Other types are fine on it.
        check_int8_mma_arch(DType::F16, Some(target)).unwrap();
        check_int8_mma_arch(DType::F8E4M3FN, Some(target)).unwrap();
    }
    for target in [Arch::Sm100a, Arch::Sm103a, Arch::Sm110a, Arch::Sm120a, Arch::Sm90a] {
        check_int8_mma_arch(DType::I8, Some(target)).unwrap();
    }
    // Strings, as CUTE_DSL_ARCH holds them.
    for target in ["sm_107", "SM_107a", " sm_107f ", "107"] {
        assert!(check_int8_mma_target(DType::U8, target).is_err(), "{target}");
    }
    for target in ["sm_100", "sm_10", "sm_1070x_no", "sm_90a"] {
        let r = check_int8_mma_target(DType::U8, target);
        assert_eq!(r.is_err(), target.starts_with("sm_107"), "{target}");
    }
}

/// `CUTE_DSL_ARCH` is process-wide: one test sets and restores it.
#[test]
fn int8_mma_arch_from_the_environment() {
    let saved = std::env::var("CUTE_DSL_ARCH").ok();
    // SAFETY: no other test in this binary touches the environment.
    unsafe {
        std::env::set_var("CUTE_DSL_ARCH", "sm_107a");
        let err = check_int8_mma_arch(DType::I8, None).unwrap_err();
        assert_eq!(err.to_string(), "sm_107a has no tcgen05 MMA for Int8 or Uint8 operands");
        check_int8_mma_arch(DType::F16, None).unwrap();
        // An argument wins over the environment.
        check_int8_mma_arch(DType::I8, Some(Arch::Sm100a)).unwrap();
        std::env::set_var("CUTE_DSL_ARCH", "sm_100a");
        check_int8_mma_arch(DType::I8, None).unwrap();
        assert!(check_int8_mma_arch(DType::I8, Some(Arch::Sm107)).is_err());
        match saved {
            Some(v) => std::env::set_var("CUTE_DSL_ARCH", v),
            None => std::env::remove_var("CUTE_DSL_ARCH"),
        }
    }
}

#[test]
fn epilogue_tile_shapes() {
    let t = |tile, ty, coop, over| compute_tile_shape_or_override(tile, ty, coop, over);
    // Cooperative: (min(128, M), min(32, N)).
    assert_eq!(t((128, 256, 64), DType::F16, true, None), (128, 32));
    assert_eq!(t((256, 16, 64), DType::F16, true, None), (128, 16));
    assert_eq!(t((64, 128, 64), DType::F32, true, None), (64, 32));
    // Ping-pong: (min(64, M), min(32 or 64, N)).
    assert_eq!(t((128, 256, 64), DType::F16, false, None), (64, 32));
    assert_eq!(t((128, 256, 64), DType::F8E4M3FN, false, None), (64, 64));
    assert_eq!(t((128, 256, 64), DType::I8, false, None), (64, 64));
    assert_eq!(t((32, 48, 64), DType::F8E5M2, false, None), (32, 48));
    assert_eq!(t((128, 256, 64), DType::F32, false, None), (64, 32));
    // An override wins, whatever the rest.
    assert_eq!(t((128, 256, 64), DType::F16, true, Some((16, 8))), (16, 8));
    assert_eq!(t((8, 8, 8), DType::I8, false, Some((128, 128))), (128, 128));
}

#[test]
fn smem_store_op_picks_stmatrix_for_16_bit_d() {
    for (layout, trans) in [(LayoutEnum::ROW_MAJOR, false), (LayoutEnum::COL_MAJOR, true)] {
        let f16 = get_smem_store_op::<F16>(layout, DType::F32);
        assert_eq!(f16.op(), CopyOp::StMatrix { num: 4, trans }, "{layout}");
        let bf16 = get_smem_store_op::<BF16>(layout, DType::F32);
        assert_eq!(bf16.op(), CopyOp::StMatrix { num: 4, trans }, "{layout}");
        assert_eq!(bf16.value_type(), DType::BF16);
        assert_eq!(sm90_get_smem_store_op::<F16>(layout, DType::F16).op(), f16.op());
        assert!(matches!(get_smem_store_op::<f32>(layout, DType::F32).op(), CopyOp::Universal { .. }));
        assert!(matches!(get_smem_store_op::<F8E4M3FN>(layout, DType::F32).op(), CopyOp::Universal { .. }));
        assert!(matches!(get_smem_store_op::<F8E5M2>(layout, DType::F16).op(), CopyOp::Universal { .. }));
    }
}
