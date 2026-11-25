//! DLPack tensors (`from_dlpack`, the implicit conversion, the DLPack structs and dtype
//! codes), external tensors on the simulator, and the cache keys of specs and pointers.
//! The managed tensors are built here over host memory, with deleters that count calls.

#[path = "support/dlpack_alloc.rs"]
mod dlpack_alloc;

use cute::external::dlpack::{self, DLDataType, DLDevice, device_type, type_code};
use cute::prelude::*;
use dlpack_alloc::{Owned, f32_type, managed};

type AddParams<H> = (DlTensor<f32, H>, DlTensor<f32, H>, DlTensor<f32, H>);

fn cpu() -> DLDevice {
    DLDevice { device_type: device_type::KDL_CPU, device_id: 0 }
}

fn floats(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.5 - 3.0).collect()
}

/// `c = a + b` over `m x n` tensors of any layout, one element per thread in a grid-stride
/// loop, on DLPack host values `H`.
fn add_kernel<H: DlpackSource>(host: &(H, H, H)) -> TypedKernel<AddParams<H>> {
    Kernel::builder("dl_add").block_dim([32, 1, 1]).jit_for(
        host,
        |a: DlTensor<f32, H>, b: DlTensor<f32, H>, c: DlTensor<f32, H>| {
            let m: Int32 = size(a.tensor(), &[0]).into_value();
            let n: Int32 = size(a.tensor(), &[1]).into_value();
            for_range(thread_idx().x, m * n, 32, |i| {
                let (r, k) = (i / n, i % n);
                c.store((r, k), a.load((r, k)) + b.load((r, k)));
            });
        },
    )
}

fn cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [32, 1, 1])
}

#[test]
fn struct_layouts_match_dlpack_h() {
    use std::mem::{align_of, size_of};
    assert_eq!((size_of::<dlpack::DLDevice>(), size_of::<DLDataType>()), (8, 4));
    assert_eq!((size_of::<dlpack::DLTensor>(), align_of::<dlpack::DLTensor>()), (48, 8));
    assert_eq!(size_of::<dlpack::DLManagedTensor>(), 64);
    assert_eq!(size_of::<dlpack::DLManagedTensorVersioned>(), 80);
    assert_eq!(std::mem::offset_of!(dlpack::DLManagedTensorVersioned, dl_tensor), 32);
    assert_eq!(std::mem::offset_of!(dlpack::DLTensor, byte_offset), 40);
    assert_eq!((dlpack::DLPACK_MAJOR_VERSION, dlpack::DLPACK_MINOR_VERSION), (1, 3));
}

#[test]
fn dtype_codes_map_to_cute_types() {
    let t = |code, bits, lanes| dlpack::dtype_of(DLDataType { code, bits, lanes }, 0);
    assert_eq!(t(type_code::KDL_FLOAT, 32, 1), Ok((DType::F32, 1)));
    assert_eq!(t(type_code::KDL_BFLOAT, 16, 1), Ok((DType::BF16, 1)));
    assert_eq!(t(type_code::KDL_INT, 64, 1), Ok((DType::I64, 1)));
    assert_eq!(t(type_code::KDL_UINT, 8, 1), Ok((DType::U8, 1)));
    assert_eq!(t(type_code::KDL_BOOL, 8, 1), Ok((DType::Bool, 1)));
    assert_eq!(t(type_code::KDL_FLOAT8_E4M3FN, 8, 1), Ok((DType::F8E4M3FN, 1)));
    assert_eq!(t(type_code::KDL_FLOAT8_E5M2, 8, 1), Ok((DType::F8E5M2, 1)));
    assert_eq!(t(type_code::KDL_FLOAT8_E8M0FNU, 8, 1), Ok((DType::F8E8M0FNU, 1)));
    assert_eq!(t(type_code::KDL_FLOAT6_E3M2FN, 6, 1), Ok((DType::F6E3M2FN, 1)));
    assert_eq!(t(type_code::KDL_FLOAT4_E2M1FN, 4, 2), Ok((DType::F4E2M1FN, 2)));
    let e = t(type_code::KDL_FLOAT, 32, 4).unwrap_err();
    assert!(e.0.contains("4 lanes of Float32"), "{e}");
    let e = t(type_code::KDL_FLOAT8_E5M2FNUZ, 8, 1).unwrap_err();
    assert!(e.0.contains("code 13 with 8 bits"), "{e}");
    let padded = dlpack::DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED;
    let e = dlpack::dtype_of(DLDataType { code: type_code::KDL_FLOAT4_E2M1FN, bits: 4, lanes: 1 }, padded).unwrap_err();
    assert!(e.0.contains("padded"), "{e}");
}

/// Compact, transposed, sliced (byte_offset) and broadcast (stride 0) tensors: layouts and
/// element types, and each deleter runs once, when the last clone goes.
#[test]
fn layouts_of_host_tensors() {
    let (t, count) = managed(Owned::f32(&floats(12)), f32_type(), &[3, 4], None, 0, 0, cpu(), true);
    let x = from_dlpack(t.clone(), FromDlpack::default()).unwrap();
    assert_eq!((x.element_type(), x.layout_string()), (DType::F32, "(3,4):(4,1)".to_string()));
    assert_eq!(x.leading_dim(), Some(1));
    assert_eq!(x.span_bytes(), 48);
    drop(t);
    assert_eq!(count.get(), 0, "the ExternalTensor keeps the DLPack object alive");
    let host = x.host_tensor().unwrap();
    assert_eq!(host.to_vec::<f32>(), floats(12));
    drop(x);
    assert_eq!(count.get(), 1);

    let (t, count) = managed(Owned::f32(&floats(12)), f32_type(), &[4, 3], Some(&[1, 4]), 0, 0, cpu(), true);
    let x = from_dlpack(t, FromDlpack::default()).unwrap();
    assert_eq!(x.layout_string(), "(4,3):(1,4)");
    assert_eq!(x.leading_dim(), Some(0));
    drop(x);
    assert_eq!(count.get(), 1);

    // A slice starting at element 2 (byte_offset 8) of a 2 x 4 view with row stride 5.
    let (t, count) = managed(Owned::f32(&floats(16)), f32_type(), &[2, 4], Some(&[5, 1]), 8, 0, cpu(), true);
    let x = from_dlpack(t, FromDlpack::default()).unwrap();
    assert_eq!(x.layout_string(), "(2,4):(5,1)");
    assert_eq!(x.span_bytes(), 9 * 4);
    assert_eq!(x.host_tensor().unwrap().to_vec::<f32>(), floats(16)[2..11].to_vec());
    drop(x);
    assert_eq!(count.get(), 1);

    // A row broadcast down 3 rows.
    let (t, count) = managed(Owned::f32(&floats(4)), f32_type(), &[3, 4], Some(&[0, 1]), 0, 0, cpu(), true);
    let x = from_dlpack(t, FromDlpack::default()).unwrap();
    assert_eq!(x.layout_string(), "(3,4):(0,1)");
    let x = x.mark_layout_dynamic(None).unwrap();
    assert_eq!(x.layout_string(), "(?,?):(0,1)", "a broadcast stride stays 0");
    drop(x);
    assert_eq!(count.get(), 1);
}

#[test]
fn marking_external_tensors_dynamic() {
    let (t, _) = managed(Owned::f32(&floats(24)), f32_type(), &[2, 3, 4], None, 0, 0, cpu(), true);
    let x = from_dlpack(t.clone(), FromDlpack::default()).unwrap();
    let d = x.clone().mark_layout_dynamic(None).unwrap();
    assert_eq!(d.layout_string(), "(?,?,?):(?{i64},?{i64},1)");
    assert_eq!(d.dynamic_shapes_mask(), [true, true, true]);
    assert_eq!(d.dynamic_strides_mask(), [true, true, false]);
    let e = x.clone().mark_layout_dynamic(Some(0)).unwrap_err();
    assert!(e.0.contains("strides[leading_dim] == 1"), "{e}");
    let c = x.clone().mark_compact_shape_dynamic(0, None, 2).unwrap();
    assert_eq!(c.layout_string(), "(?{div=2},3,4):(12,4,1)");
    assert_eq!(c.dynamic_shapes_mask(), [true, false, false]);
    let c = x.clone().mark_compact_shape_dynamic(2, None, 4).unwrap();
    assert_eq!(c.layout_string(), "(2,3,?{div=4}):(?{i64 div=12},?{i64 div=4},1)");
    let x32 = from_dlpack(t.clone(), FromDlpack { use_32bit_stride: true, ..Default::default() }).unwrap();
    assert_eq!(x32.mark_layout_dynamic(None).unwrap().layout_string(), "(?,?,?):(?,?,1)");
    // The implicit conversion: every mode dynamic but the unit leading stride.
    assert_eq!(t.to_external().unwrap().layout_string(), "(?,?,?):(?{i64},?{i64},1)");
    let tf32 = from_dlpack(t, FromDlpack { force_tf32: true, ..Default::default() }).unwrap();
    assert_eq!(tf32.element_type(), DType::TF32);
}

/// Element-wise addition on host DLPack tensors on the simulator, converted explicitly
/// (static layouts) and implicitly (all dynamic): the output is written back into the
/// producer's memory.
#[test]
fn the_simulator_runs_on_host_dlpack_tensors() {
    let (a, b) = (floats(12), floats(12).iter().map(|x| x * 3.0).collect::<Vec<_>>());
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    for implicit in [false, true] {
        let mk = |v: &[f32]| managed(Owned::f32(v), f32_type(), &[3, 4], None, 0, 0, cpu(), true);
        let ((ta, ca), (tb, cb), (tc, cc)) = (mk(&a), mk(&b), mk(&[0.0; 12]));
        if implicit {
            let mut host = (ta, tb, tc);
            let k = add_kernel(&host);
            assert!(k.param_names().contains(&"arg0_shape_0".to_string()), "{:?}", k.param_names());
            k.simulate(cfg(), &mut host).unwrap_or_else(|e| panic!("{e}"));
        } else {
            let ext = |t| from_dlpack(t, FromDlpack::default()).unwrap();
            let mut host = (ext(ta), ext(tb), ext(tc));
            let k = add_kernel(&host);
            assert_eq!(k.param_names(), ["arg0", "arg1", "arg2"]);
            k.simulate(cfg(), &mut host).unwrap_or_else(|e| panic!("{e}"));
        }
        assert_eq!((ca.get(), cb.get(), cc.get()), (1, 1, 1), "each deleter once");
        assert_eq!(cc.data_f32(), want, "implicit: {implicit}");
    }
}

#[test]
fn errors_name_the_problem_and_deleters_still_run() {
    // Wrong device for the simulator.
    let cuda = DLDevice { device_type: device_type::KDL_CUDA, device_id: 0 };
    let mk = |dev| managed(Owned::f32(&floats(12)), f32_type(), &[3, 4], None, 0, 0, dev, true);
    let ((ta, ca), (tb, cb), (tc, cc)) = (mk(cpu()), mk(cuda), mk(cpu()));
    let mut host = (ta, tb, tc);
    let k = add_kernel(&host);
    let e = k.simulate(cfg(), &mut host).unwrap_err();
    assert!(e.message.contains("argument `arg1` is CUDA device memory (device 0)"), "{e}");
    drop(host);
    assert_eq!((ca.get(), cb.get(), cc.get()), (1, 1, 1));

    // Wrong lanes.
    let lanes4 = DLDataType { code: type_code::KDL_FLOAT, bits: 32, lanes: 4 };
    let (t, count) = managed(Owned::f32(&floats(12)), lanes4, &[3], None, 0, 0, cpu(), true);
    let e = from_dlpack(t, FromDlpack::default()).unwrap_err();
    assert!(e.0.contains("4 lanes of Float32"), "{e}");
    assert_eq!(count.get(), 1, "the deleter ran on the error path");

    // A misaligned start.
    let (t, count) = managed(Owned::f32(&floats(12)), f32_type(), &[2], None, 4, 0, cpu(), true);
    let e = from_dlpack(t, FromDlpack { assumed_align: Some(16), ..Default::default() }).unwrap_err();
    assert!(e.0.contains("not a multiple of the assumed alignment of 16 bytes"), "{e}");
    assert_eq!(count.get(), 1);

    // A read-only output.
    let ro = dlpack::DLPACK_FLAG_BITMASK_READ_ONLY;
    let mk = |flags| managed(Owned::f32(&floats(12)), f32_type(), &[3, 4], None, 0, flags, cpu(), true);
    let ((ta, _), (tb, _), (tc, cc)) = (mk(ro), mk(ro), mk(ro));
    let mut host = (ta, tb, tc);
    let k = add_kernel(&host);
    let e = k.simulate(cfg(), &mut host).unwrap_err();
    assert!(
        e.message.contains("argument `arg2`: the tensor is read-only")
            && e.message.contains("kernel `dl_add` writes it"),
        "{e}"
    );
    drop(host);
    assert_eq!(cc.get(), 1);

    // A read-only tensor written through a raw pointer made from its address.
    let (t, count) = mk(ro);
    let mut host = t;
    let k = Kernel::builder("dl_raw_write").block_dim([32, 1, 1]).jit_for(&host, |a: DlTensor<f32>| {
        let raw = make_ptr::<f32>(a.ptr().toint(), AddressSpace::Gmem, Some(4));
        raw.store(1.0f32);
    });
    let e = k.simulate(cfg(), &mut host).unwrap_err();
    assert!(e.message.contains("the tensor is read-only") && e.message.contains("writes it"), "{e}");
    drop(host);
    assert_eq!(count.get(), 1);

    // An unsupported version.
    let (t, count) = dlpack_alloc::managed_version(Owned::f32(&floats(4)), f32_type(), &[4], cpu(), 2);
    let e = from_dlpack(t, FromDlpack::default()).unwrap_err();
    assert!(e.0.contains("DLPack version 2.3 is not supported"), "{e}");
    assert_eq!(count.get(), 1);
}

/// An unversioned DLManagedTensor, and a borrowed DLTensor, which is never deleted.
#[test]
fn unversioned_and_borrowed_tensors() {
    let (t, count) = managed(Owned::f32(&floats(6)), f32_type(), &[6], None, 0, 0, cpu(), false);
    assert!(t.version().is_none() && t.is_owned());
    let x = from_dlpack(t, FromDlpack::default()).unwrap();
    assert_eq!(x.layout_string(), "(6):(1)");
    drop(x);
    assert_eq!(count.get(), 1);
    let (t, count) = managed(Owned::f32(&floats(6)), f32_type(), &[6], None, 0, 0, cpu(), true);
    // SAFETY: `t` keeps the DLTensor alive for the borrow, which ends first.
    let b = unsafe { DlpackTensor::borrowed(t.dl_tensor()) };
    assert!(!b.is_owned());
    let x = from_dlpack(b, FromDlpack::default()).unwrap();
    assert_eq!(x.host_tensor().unwrap().to_vec::<f32>(), floats(6));
    drop(x);
    assert_eq!(count.get(), 0, "a borrowed view does not delete");
    drop(t);
    assert_eq!(count.get(), 1);
}

/// A packed `float4_e2m1fn_x2` tensor is seen with its logical FP4 shape, and a `uint8`
/// tensor can be relabelled Float4E2M1FN.
#[test]
fn fp4_packing() {
    let bytes: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(37)).collect();
    let fp4x2 = DLDataType { code: type_code::KDL_FLOAT4_E2M1FN, bits: 4, lanes: 2 };
    let (t, _) = managed(Owned::bytes(&bytes), fp4x2, &[4, 8], None, 0, 0, cpu(), true);
    let x = from_dlpack(t, FromDlpack::default()).unwrap();
    assert_eq!((x.element_type(), x.layout_string()), (DType::F4E2M1FN, "(4,16):(16,1)".to_string()));
    assert_eq!(x.span_bytes(), 32);
    let host = x.host_tensor().unwrap();
    assert_eq!(host.data(), &KernelArg::Bytes { elem: DType::F4E2M1FN, data: bytes.clone() });
    let u8t = DLDataType { code: type_code::KDL_UINT, bits: 8, lanes: 1 };
    let (t, _) = managed(Owned::bytes(&bytes), u8t, &[4, 8], None, 0, 0, cpu(), true);
    let x = from_dlpack(t, FromDlpack::default()).unwrap().with_element_type(DType::F4E2M1FN).unwrap();
    assert_eq!((x.element_type(), x.layout_string()), (DType::F4E2M1FN, "(4,8):(8,1)".to_string()));
    assert_eq!(x.span_bytes(), 16);
    let e = x.with_element_type(DType::F32).unwrap_err();
    assert!(e.0.contains("the new type is wider"), "{e}");
}

#[test]
fn cache_keys() {
    let n = sym_int32(4, None);
    let a = make_fake_tensor::<f32>((n, 8), (1, n));
    assert_eq!(a.cache_key(), a.clone().cache_key(), "equal specs hash equal");
    let m = sym_int32(4, None);
    let b = make_fake_tensor::<f32>((m, 8), (1, m));
    assert_eq!(a.cache_key(), b.cache_key(), "the same sharing pattern of fresh SymInts");
    let c = make_fake_tensor::<f32>((sym_int32(4, None), 8), (1, sym_int32(4, None)));
    assert_ne!(a.cache_key(), c.cache_key(), "one SymInt in two places is not two SymInts");
    let d = make_fake_tensor::<f32>((sym_int32(8, None), 8), (1, sym_int32(4, None)));
    assert_ne!(c.cache_key(), d.cache_key(), "a different divisibility");
    assert_ne!(a.cache_key(), a.clone().with_assumed_align(16).cache_key(), "a different alignment");
    let s = make_fake_tensor::<f32>((8, 8), (1, 8));
    assert_ne!(s.cache_key(), a.cache_key(), "static and dynamic leaves differ");
    // Two SymInts of equal width and divisibility are equal values with different keys.
    let (x, y) = (sym_int32(4, None), sym_int32(4, None));
    assert_eq!(x, y);
    assert_ne!(x.cache_key(), y.cache_key());
    assert_eq!(x.cache_key(), x.cache_key());
    // Sharing across specs: A and C sharing M key differently from separate Ms, together.
    let mm = sym_int32(1, None);
    let (a1, c1) = (make_fake_tensor::<f32>((mm, 8), (8, 1)), make_fake_tensor::<f32>((mm, 8), (8, 1)));
    let (a2, c2) = (
        make_fake_tensor::<f32>((sym_int32(1, None), 8), (8, 1)),
        make_fake_tensor::<f32>((sym_int32(1, None), 8), (8, 1)),
    );
    assert_eq!((a1.cache_key(), c1.cache_key()), (a2.cache_key(), c2.cache_key()), "each alone cannot tell");
    assert_ne!(TensorSpec::cache_keys(&[&a1, &c1]), TensorSpec::cache_keys(&[&a2, &c2]));
    // Specs are hashable: a HashSet of them.
    let set: std::collections::HashSet<TensorSpec> = [a.clone(), a.clone(), s.clone()].into_iter().collect();
    assert_eq!(set.len(), 2);
    // Pointer descriptors: CuTe DSL's (dtype, addr_space, assumed_align).
    let p = PointerSpec::new(DType::F16, cute::tensor::AddressSpace::Gmem, None);
    assert_eq!(p.assumed_align, 2);
    assert_eq!(p.cache_key(), PointerSpec::new(DType::F16, cute::tensor::AddressSpace::Gmem, Some(2)));
    assert_ne!(p.cache_key(), PointerSpec::new(DType::F16, cute::tensor::AddressSpace::Gmem, Some(16)));
    let e = PointerSpec::new(DType::F32, cute::tensor::AddressSpace::Gmem, Some(16)).check_address(0x1004).unwrap_err();
    assert!(e.contains("0x1004") && e.contains("16 bytes"), "{e}");
    // DLPack tensors marked the same way key the same.
    let (t, _) = managed(Owned::f32(&floats(12)), f32_type(), &[3, 4], None, 0, 0, cpu(), true);
    let k1 = t.to_external().unwrap().cache_key();
    let k2 = t.to_external().unwrap().cache_key();
    assert_eq!(k1, k2);
    assert_ne!(k1, from_dlpack(t, FromDlpack::default()).unwrap().cache_key());
}

#[allow(dead_code)]
#[path = "../examples/call_bypass_dlpack.rs"]
mod call_bypass_dlpack;

/// `call_bypass_dlpack.py` on the simulator, on host buffers for the pointers.
#[test]
fn call_bypass_dlpack_example() {
    let c = call_bypass_dlpack::run(16, 8, 5, 2);
    assert_eq!(c.len(), 16 * 8 * 2);
}
