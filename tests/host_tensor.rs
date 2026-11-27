//! Host tensor helpers (`cute::host`), the dtype conversion kernel (`cute::testing::convert`
//! on the simulator) and dtype interop. The GPU half is `tests/gpu_host_tensor.rs`.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/sgemm.rs"]
mod sgemm;

use std::sync::Mutex;

use cute::external::DLDataType;
use cute::external::dlpack::type_code;
use cute::host::HostTensor;
use cute::host::*;
use cute::layout::Layout;
use cute::params::HostTensor as ParamTensor;
use cute::prelude::*;
use cute::testing::{CONVERT_DTYPES, convert_kernels_built, convert_sim, elem_per_copy};

/// Conversions share the kernel cache, whose size one test reads.
static CONVERT: Mutex<()> = Mutex::new(());

/// The pair sweep's types: Float32, Float16, BFloat16, both FP8s, both FP6s, FP4, Int4, Int8.
const SWEEP: [DType; 10] = [
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::F8E4M3FN,
    DType::F8E5M2,
    DType::F6E2M3FN,
    DType::F6E3M2FN,
    DType::F4E2M1FN,
    DType::I4,
    DType::I8,
];

fn offsets(layout: &Layout) -> Vec<usize> {
    cute::host::HostTensor::<u8>::zeros(layout.clone()).offsets()
}

/// A launch tensor of `dtype` over `layout` whose logical element `i` is `values[i]` (rounded
/// on the host).
fn tensor_of(dtype: DType, layout: &Layout, values: &[f64]) -> ParamTensor {
    let offs = offsets(layout);
    let mut bits = vec![0u64; layout.cosize()];
    for (o, &v) in offs.iter().zip(values) {
        bits[*o] = encode_f64(dtype, v);
    }
    ParamTensor::new(KernelArg::Bytes { elem: dtype, data: pack_elements(dtype, &bits) }, layout.clone())
}

fn zeros_of(dtype: DType, layout: &Layout) -> ParamTensor {
    let data = vec![0u8; dtype.storage_bytes(layout.cosize())];
    ParamTensor::new(KernelArg::Bytes { elem: dtype, data }, layout.clone())
}

/// The raw bits of a launch tensor's elements in logical order.
fn logical_bits(t: &ParamTensor) -> Vec<u64> {
    let layout = t.layout();
    let all = unpack_elements(t.element_type(), &arg_bytes(t.data()), layout.cosize());
    offsets(&layout).into_iter().map(|o| all[o]).collect()
}

/// Source values: multiples of a quarter in [-7.5, 7.5], in a seeded order.
fn quarters(n: usize, seed: u64) -> Vec<f64> {
    quarters_within(n, seed, 30)
}

/// Multiples of a quarter up to `limit / 4` in magnitude.
fn quarters_within(n: usize, seed: u64, limit: usize) -> Vec<f64> {
    let mut rng = cute::utils::gemm::Rng::new(seed);
    (0..n).map(|_| (rng.below(2 * limit + 1) as f64 - limit as f64) / 4.0).collect()
}

/// Every pair of the sweep's types, a row-major and a column-major leading mode, bit for bit
/// against the host conversion.
#[test]
fn convert_matches_the_host_for_every_pair() {
    let _g = CONVERT.lock().unwrap_or_else(|e| e.into_inner());
    let layouts = [Layout::row_major((4, 32)), Layout::col_major((32, 4)), Layout::new((16, 3, 2), (1, 16, 48))];
    let mut pairs = 0;
    for &src in &SWEEP {
        for &dst in &SWEEP {
            for layout in &layouts {
                let n = layout.size();
                // A float outside the range of an integer destination converts to an
                // unspecified value, so those stay within Int4's range (after rounding).
                let values = quarters_within(n, 7 + pairs, if dst == DType::I4 { 13 } else { 30 });
                let s = tensor_of(src, layout, &values);
                let mut d = zeros_of(dst, layout);
                convert_sim(&s, &mut d).unwrap_or_else(|e| panic!("{src} -> {dst} on {layout}: {e}"));
                let got = logical_bits(&d);
                let want: Vec<u64> =
                    logical_bits(&s).into_iter().map(|b| encode_f64(dst, decode_f64(src, b))).collect();
                assert_eq!(got, want, "{src} -> {dst} on {layout}");
                pairs += 1;
            }
        }
    }
    assert_eq!(pairs, 300);
}

#[test]
fn convert_handles_other_ranks_and_strides() {
    let _g = CONVERT.lock().unwrap_or_else(|e| e.into_inner());
    // Rank 1, a leading mode that is not the first, and a destination with a padded stride.
    let cases =
        [(Layout::new(64, 1), Layout::new(64, 1)), (Layout::new((3, 40), (40, 1)), Layout::new((3, 40), (48, 1)))];
    for (i, (sl, dl)) in cases.iter().enumerate() {
        let values = quarters(sl.size(), 3);
        let s = tensor_of(DType::F32, sl, &values);
        let mut d = zeros_of(DType::F16, dl);
        convert_sim(&s, &mut d).unwrap();
        let want: Vec<u64> = values.iter().map(|&v| encode_f64(DType::F16, v)).collect();
        assert_eq!(logical_bits(&d), want, "case {i}");
    }
    // 5 x 9 tiles of 16 (45 threads): no block-size multiple.
    let l = Layout::new((5, 9, 32), (288, 32, 1));
    let values = quarters(l.size(), 4);
    let s = tensor_of(DType::F32, &l, &values);
    let mut d = zeros_of(DType::F8E4M3FN, &l);
    convert_sim(&s, &mut d).unwrap();
    let want: Vec<u64> = values.iter().map(|&v| encode_f64(DType::F8E4M3FN, v)).collect();
    assert_eq!(logical_bits(&d), want);
}

fn refusal(src: &ParamTensor, dst: &mut ParamTensor) -> String {
    convert_sim(src, dst).expect_err("the conversion should be refused").0
}

#[test]
fn convert_diagnostics() {
    let _g = CONVERT.lock().unwrap_or_else(|e| e.into_inner());
    let row = Layout::row_major((4, 32));
    let src = tensor_of(DType::F32, &row, &quarters(128, 1));

    // No mode of stride 1 and extent above 1.
    let strided = Layout::new((4, 32), (64, 2));
    let mut d = zeros_of(DType::F16, &strided);
    let m = refusal(&tensor_of(DType::F32, &strided, &quarters(128, 1)), &mut d);
    assert!(m.contains("leading mode must be unique") && m.contains("has 0"), "{m}");

    // Two of them.
    let two = Layout::new((4, 4), (1, 1));
    let mut d = zeros_of(DType::F16, &two);
    let m = refusal(&tensor_of(DType::F32, &two, &quarters(16, 1)), &mut d);
    assert!(m.contains("leading mode must be unique") && m.contains("has 2"), "{m}");

    // The leading extent is not a multiple of the vector (4 for an 8-bit type).
    let odd = Layout::row_major((4, 30));
    let mut d = zeros_of(DType::F8E5M2, &odd);
    let m = refusal(&tensor_of(DType::F32, &odd, &quarters(120, 1)), &mut d);
    assert!(m.contains("leading extent 30") && m.contains("multiple of 4"), "{m}");
    // 16 elements for a 6-bit type.
    let l = Layout::row_major((4, 24));
    let mut d = zeros_of(DType::F6E2M3FN, &l);
    let m = refusal(&tensor_of(DType::F32, &l, &quarters(96, 1)), &mut d);
    assert!(m.contains("multiple of 16"), "{m}");

    // Rank, shape, leading mode and type mismatches.
    let mut d = zeros_of(DType::F16, &Layout::row_major(128));
    assert!(refusal(&src, &mut d).contains("ranks differ"));
    let mut d = zeros_of(DType::F16, &Layout::row_major((8, 16)));
    assert!(refusal(&src, &mut d).contains("shapes differ"));
    let mut d = zeros_of(DType::F16, &Layout::col_major((4, 32)));
    assert!(refusal(&src, &mut d).contains("leading mode"));
    let mut d = zeros_of(DType::F8E5M3FNU, &row);
    assert!(refusal(&src, &mut d).contains("FloatNV8E5M3FNU"));
    assert!(CONVERT_DTYPES.contains(&DType::F32) && !CONVERT_DTYPES.contains(&DType::F8E5M3FNU));
    assert_eq!(elem_per_copy(DType::F8E5M2, DType::F16), 4);
}

#[test]
fn convert_builds_each_kernel_once() {
    let _g = CONVERT.lock().unwrap_or_else(|e| e.into_inner());
    let l = Layout::row_major((6, 34));
    let s = tensor_of(DType::F32, &l, &quarters(l.size(), 9));
    let mut d = zeros_of(DType::BF16, &l);
    convert_sim(&s, &mut d).unwrap();
    let built = convert_kernels_built();
    for _ in 0..3 {
        convert_sim(&s, &mut d).unwrap();
    }
    assert_eq!(convert_kernels_built(), built);
    // Another destination type is another kernel.
    let mut d2 = zeros_of(DType::F16, &l);
    convert_sim(&s, &mut d2).unwrap();
    assert_eq!(convert_kernels_built(), built + 1);
}

// ---- host tensors ------------------------------------------------------------------------

#[test]
fn permuted_tensors_have_strided_views() {
    let t = create_and_permute_tensor::<f32>(&[2, 3, 4], None, TensorInit::RANDOM, 1);
    assert_eq!((t.shape(), t.strides()), (vec![2, 3, 4], vec![12, 4, 1]));
    assert_eq!(t.leading_dim(), Some(2));
    let p = create_and_permute_tensor::<f32>(&[2, 3, 4], Some(&[2, 0, 1]), TensorInit::RANDOM, 1);
    assert_eq!((p.shape(), p.strides()), (vec![4, 2, 3], vec![1, 12, 4]));
    assert_eq!(p.leading_dim(), Some(0));
    // The same memory, seen permuted.
    assert_eq!(p.data(), t.data());
    assert_eq!(p.get(&[3, 1, 2]), t.get(&[1, 2, 3]));
    assert_eq!(p.numel(), 24);
    let text =
        std::panic::catch_unwind(|| create_and_permute_tensor::<f32>(&[2, 3], Some(&[0, 0]), TensorInit::ZERO, 1));
    assert!(text.is_err());
}

#[test]
fn matrix_strides_follow_the_major_mode() {
    let m0 = matrix::<f32>(3, 4, 5, true, TensorInit::RANDOM, 2);
    assert_eq!((m0.shape(), m0.strides()), (vec![4, 5, 3], vec![1, 4, 20]));
    assert_eq!(m0.leading_dim(), Some(0));
    let m1 = matrix::<f32>(3, 4, 5, false, TensorInit::RANDOM, 2);
    assert_eq!((m1.shape(), m1.strides()), (vec![4, 5, 3], vec![5, 1, 20]));
    assert_eq!(m1.leading_dim(), Some(1));
    // Of two modes with stride 1, the one that is not trivial is the leading one.
    let one = matrix::<f32>(1, 1, 8, true, TensorInit::ZERO, 2);
    assert_eq!((one.strides(), one.leading_dim()), (vec![1, 1, 8], Some(1)));
    assert_eq!(get_leading_dim(&[4, 4], &[8, 2]), None);
    assert_eq!(get_leading_dim(&[1, 4], &[1, 1]), Some(1));
}

#[test]
fn random_values_are_integers_in_range() {
    let t = create_and_permute_tensor::<f32>(&[64, 64], None, TensorInit::Random { min: -3, max: 5 }, 11);
    let mut seen = [false; 8];
    for &v in t.data() {
        assert!(v.fract() == 0.0 && (-3.0..5.0).contains(&v), "{v}");
        seen[(v + 3.0) as usize] = true;
    }
    assert!(seen.iter().all(|&s| s), "every integer of the range is drawn: {seen:?}");
    // The default is -2..2, and the upper bound is excluded.
    let d = create_and_permute_tensor::<f32>(&[4096], None, TensorInit::RANDOM, 5);
    assert!(d.data().iter().all(|&v| (-2.0..2.0).contains(&v)));
    // Narrow types round the integers (1.0 and -2.0 are all FP4 needs here).
    let q = create_and_permute_tensor::<F4E2M1FN>(&[256], None, TensorInit::RANDOM, 5);
    assert!(q.to_f64().iter().all(|v| [-2.0, -1.0, 0.0, 1.0].contains(v)));
}

#[test]
fn scalar_is_exact_and_gaussian_follows_its_moments() {
    let t = create_and_permute_tensor::<F16>(&[3, 5], None, TensorInit::Scalar(1.5), 0);
    assert!(t.to_f64().iter().all(|&v| v == 1.5));
    let z = create_and_permute_tensor::<f32>(&[7], None, TensorInit::Skip, 0);
    assert_eq!(z.data(), &[0.0; 7]);
    let g = create_and_permute_tensor::<f32>(&[200, 100], None, TensorInit::GAUSSIAN, 42);
    let n = g.numel() as f64;
    let mean = g.to_f64().iter().sum::<f64>() / n;
    let var = g.to_f64().iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    assert!(mean.abs() < 0.03 && (var.sqrt() - 1.0).abs() < 0.03, "mean {mean}, std {}", var.sqrt());
    let h = create_and_permute_tensor::<f32>(
        &[200, 100],
        None,
        TensorInit::Gaussian { mean: 3.0, std: 0.5, scale: 2.0 },
        42,
    );
    let mean = h.to_f64().iter().sum::<f64>() / n;
    let std = (h.to_f64().iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    assert!((mean - 6.0).abs() < 0.05 && (std - 1.0).abs() < 0.05, "mean {mean}, std {std}");
}

#[test]
fn values_are_deterministic_per_seed() {
    let make = |seed| matrix::<F16>(2, 8, 8, false, TensorInit::RANDOM, seed);
    assert_eq!(make(1), make(1));
    assert_ne!(make(1).data(), make(2).data());
    let a = create_and_permute_tensor::<f32>(&[100], None, TensorInit::GAUSSIAN, 9);
    assert_eq!(a, create_and_permute_tensor::<f32>(&[100], None, TensorInit::GAUSSIAN, 9));
    // Pinned values: the generator is SplitMix64, so these never change.
    let p = create_and_permute_tensor::<f32>(&[4], None, TensorInit::Random { min: 0, max: 1000 }, 0);
    assert_eq!(p.data(), &[535.0, 700.0, 679.0, 444.0]);
}

#[test]
fn host_tensors_round_trip_through_kernel_arguments() {
    // convert kernels are counted by convert_builds_each_kernel_once.
    let _g = CONVERT.lock().unwrap_or_else(|e| e.into_inner());
    let t = matrix::<F16>(2, 6, 4, true, TensorInit::RANDOM, 3);
    let back = HostTensor::<F16>::from_param(&t.to_param());
    assert_eq!(back.data(), t.data());
    assert_eq!(back.layout(), t.layout());
    let i4 = create_and_permute_tensor::<I4>(&[4, 16], None, TensorInit::Random { min: -8, max: 8 }, 3);
    let back = HostTensor::<I4>::from_kernel_arg(&i4.to_kernel_arg(), i4.layout().clone());
    assert_eq!(back.data()[..64], i4.data()[..64]);
    // Element type conversions on the host.
    let q: HostTensor<F8E4M3FN> = t.convert();
    assert_eq!(q.to_f64(), t.to_f64());
    // Reading memory.
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let r = unsafe { as_tensor(v.as_ptr(), &[2, 3]) };
    assert_eq!((r.strides(), r.get(&[1, 2])), (vec![3, 1], 6.0));
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let r = unsafe { as_tensor_dyn(bytes.as_ptr(), &[3, 2], DType::F32) };
    assert_eq!((r.element_type(), r.data().to_vec::<f32>()), (DType::F32, v.to_vec()));
}

#[test]
fn convert_cute_tensor_fills_narrow_types_through_the_device_kernel() {
    let _g = CONVERT.lock().unwrap_or_else(|e| e.into_inner());
    let f = matrix::<f32>(2, 4, 32, false, TensorInit::Random { min: -4, max: 5 }, 6);
    for dtype in
        [DType::I4, DType::F8E4M3FN, DType::F8E5M2, DType::F8E8M0FNU, DType::F6E2M3FN, DType::F6E3M2FN, DType::F4E2M1FN]
    {
        let t = cute_tensor_like(&f, dtype, false, None).unwrap_or_else(|e| panic!("{dtype}: {e}"));
        let want: Vec<u64> = f.to_f64().into_iter().map(|v| encode_f64(dtype, v)).collect();
        assert_eq!(logical_bits(&t), want, "{dtype}");
    }
    // The NV E5M3 scale type and the wide types are rounded on the host.
    for dtype in [DType::F8E5M3FNU, DType::F16, DType::BF16, DType::F32, DType::I8, DType::I32] {
        let t = cute_tensor_like(&f, dtype, true, Some(16)).unwrap();
        let want: Vec<u64> = f.to_f64().into_iter().map(|v| encode_f64(dtype, v)).collect();
        assert_eq!(logical_bits(&t), want, "{dtype}");
        assert_eq!(t.element_type(), dtype);
    }
    // A dynamic layout keeps the leading dimension static.
    let t = cute_tensor_like(&f, DType::F16, true, Some(16)).unwrap();
    assert_eq!(t.layout_string(), "(?,?,?):(?{i64},1,?{i64})");
}

#[test]
fn prepare_tensors_for_gemm_makes_k_major_operands() {
    // convert kernels are counted by convert_builds_each_kernel_once.
    let _g = CONVERT.lock().unwrap_or_else(|e| e.into_inner());
    let g = prepare_tensors_for_gemm((8, 16, 32, 2), DType::F8E4M3FN, DType::I4, DType::F16, 5).unwrap();
    assert_eq!((g.a.element_type(), g.b.element_type(), g.c.element_type()), (DType::F8E4M3FN, DType::I4, DType::F16));
    let shapes = |t: &ParamTensor| t.layout().static_shape().flat_leaves();
    let strides = |t: &ParamTensor| t.layout().static_stride().flat_leaves();
    assert_eq!((shapes(&g.a), strides(&g.a)), (vec![8, 32, 2], vec![32, 1, 256]));
    assert_eq!((shapes(&g.b), strides(&g.b)), (vec![16, 32, 2], vec![32, 1, 512]));
    assert_eq!((shapes(&g.c), strides(&g.c)), (vec![8, 16, 2], vec![16, 1, 128]));
    assert_eq!(operand_major_mode(&g.a.layout()), Some(OperandMajorMode::K));
    // The references hold what the tensors hold.
    for (t, r) in [(&g.a, &g.a_ref), (&g.b, &g.b_ref), (&g.c, &g.c_ref)] {
        let dt = t.element_type();
        let got: Vec<f64> = logical_bits(t).into_iter().map(|b| decode_f64(dt, b)).collect();
        assert_eq!(got, r.to_f64(), "{dt}");
    }
    assert!(g.c_ref.data().iter().all(|&v| v == 0.0));
    assert!(g.a_ref.data().iter().any(|&v| v != 0.0));
}

/// `sgemm` (the tutorial's `C = alpha A B^T + beta C` with M-major A and B), driven by
/// `matrix()` tensors instead of hand-built vectors.
#[test]
fn sgemm_with_matrix_operands() {
    let (m, n, k) = (128, 128, 16);
    let a = matrix::<f32>(1, m, k, true, TensorInit::Random { min: -5, max: 6 }, 1);
    let b = matrix::<f32>(1, n, k, true, TensorInit::Random { min: -6, max: 7 }, 2);
    let c0 = matrix::<f32>(1, m, n, true, TensorInit::Random { min: 0, max: 5 }, 3);
    let (alpha, beta) = (1.5f32, -0.5f32);
    let kernel = sgemm::build(m, n, k);
    let mut args = vec![
        a.to_kernel_arg(),
        b.to_kernel_arg(),
        c0.to_kernel_arg(),
        KernelArg::ScalarF32(alpha),
        KernelArg::ScalarF32(beta),
    ];
    let cfg = LaunchConfig::new([1, 1, 1], [256, 1, 1]);
    kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    let c = HostTensor::<f32>::from_kernel_arg(&args[2], c0.layout().clone());
    for i in 0..m {
        for j in 0..n {
            let dot: f32 = (0..k).map(|l| a.get(&[i, l, 0]) * b.get(&[j, l, 0])).sum();
            let want = alpha * dot + beta * c0.get(&[i, j, 0]);
            let got = c.get(&[i, j, 0]);
            assert!((got - want).abs() <= 1e-3 * want.abs().max(1.0), "C[{i},{j}] = {got}, expected {want}");
        }
    }
}

// ---- dtype interop -------------------------------------------------------------------------

#[test]
fn dtype_mapping_round_trips() {
    let mut mapped = 0;
    for dtype in DType::ALL {
        let Some(dl) = dl_data_type(dtype) else {
            assert_eq!(dtype, DType::F8E5M3FNU, "{dtype} has no DLPack type");
            continue;
        };
        let back = dtype_from_dl(dl).unwrap_or_else(|e| panic!("{dtype}: {e}"));
        if dtype == DType::TF32 {
            assert_eq!(back, DType::F32);
        } else {
            assert_eq!(back, dtype, "{dtype} via {dl:?}");
        }
        mapped += 1;
    }
    assert_eq!(mapped, DType::ALL.len() - 1);
    // Spot checks against dlpack.h.
    let t = |d: DType| dl_data_type(d).map(|d| (d.code, d.bits, d.lanes));
    assert_eq!(t(DType::F16), Some((2, 16, 1)));
    assert_eq!(t(DType::BF16), Some((4, 16, 1)));
    assert_eq!(t(DType::Bool), Some((6, 8, 1)));
    assert_eq!(t(DType::I8), Some((0, 8, 1)));
    assert_eq!(t(DType::U8), Some((1, 8, 1)));
    assert_eq!(t(DType::F8E4M3FN), Some((type_code::KDL_FLOAT8_E4M3FN, 8, 1)));
    assert_eq!(t(DType::F4E2M1FNx2), Some((type_code::KDL_FLOAT4_E2M1FN, 4, 2)));
    assert_eq!(t(DType::F6E3M2FNx4), Some((type_code::KDL_FLOAT6_E3M2FN, 6, 4)));
}

#[test]
fn dtype_mapping_rejects_unknown_codes() {
    let bad = |code, bits, lanes| dtype_from_dl(DLDataType { code, bits, lanes }).unwrap_err().0;
    assert!(bad(99, 8, 1).contains("no cute-rs element type"));
    assert!(bad(type_code::KDL_FLOAT, 24, 1).contains("no cute-rs element type"));
    assert!(bad(type_code::KDL_COMPLEX, 64, 1).contains("no cute-rs element type"));
    assert!(bad(type_code::KDL_FLOAT, 32, 4).contains("lanes"));
    assert!(bad(type_code::KDL_FLOAT4_E2M1FN, 4, 4).contains("lanes"));
}

#[test]
fn torch_names_round_trip() {
    for dtype in DType::ALL {
        match torch_dtype_name(dtype) {
            Some(name) => {
                assert!(name.starts_with("torch."), "{name}");
                let back = dtype_from_torch_name(name).unwrap();
                assert_eq!(back, if dtype == DType::TF32 { DType::F32 } else { dtype });
            }
            None => assert!(
                matches!(
                    dtype,
                    DType::I128
                        | DType::U128
                        | DType::F8E4M3
                        | DType::F8E4M3B11FNUZ
                        | DType::F8E5M3FNU
                        | DType::F6E2M3FN
                        | DType::F6E3M2FN
                        | DType::F4E2M1FN
                        | DType::F6E2M3FNx4
                        | DType::F6E3M2FNx4
                ),
                "{dtype}"
            ),
        }
    }
    assert_eq!(torch_dtype_name(DType::F16), Some("torch.float16"));
    assert_eq!(dtype_from_torch_name("torch.float4_e2m1fn_x2"), Some(DType::F4E2M1FNx2));
    assert_eq!(dtype_from_torch_name("torch.complex64"), None);
    assert!(describe_dtype(DType::BF16).contains("torch.bfloat16") && describe_dtype(DType::BF16).contains("code 4"));
}
