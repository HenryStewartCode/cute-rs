// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/dsl_tutorials/dataclass_immutable.py and
//   examples/python/CuTeDSL/dsl_tutorials/tvm_ffi/compile_with_fake_tensor.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   examples/python/CuTeDSL/experimental/primitives/tutorial/08_mlir_value_tree_debug.py,
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   media/docs/pythonDSL/guides/framework_integration.rst (test vectors),
//     Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The kernel argument protocol: the parameters each spec declares, the
//! host values a launch checks, `TensorSpec`, `HostTensor` and its
//! `mark_layout_dynamic` and `mark_compact_shape_dynamic` (the cases of CuTe
//! DSL's framework-integration guide), `SymInt`, the fake-tensor
//! constructors, typed entry points, struct arguments and native structs.
//! The kernels that run at several sizes are in `feature_kernels/params.rs`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::ir::Param;
use cute::prelude::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>().cloned().or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string())).unwrap()
}

/// Each parameter: `name: buffer of T (align A, len L)` or `name: T div D`.
fn signature(k: &Kernel) -> Vec<String> {
    let ir = k.ir();
    ir.args
        .iter()
        .map(|p| match p {
            Param::Buffer(b) => {
                let d = ir.buffer(*b);
                let len = d.len.map_or("?".to_string(), |n| n.to_string());
                format!("{}: buffer of {} (align {}, len {len})", d.name, d.elem, d.align)
            }
            Param::Scalar { name, value, multiple_of } => format!("{name}: {} div {multiple_of}", ir.dtype(*value)),
            Param::ByValue { buffer, .. } => format!("{}: by value", ir.buffer(*buffer).name),
        })
        .collect()
}

fn cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [32, 1, 1])
}

// ---------------------------------------------------------------------------
// The protocol, TensorSpec, layouts as arguments
// ---------------------------------------------------------------------------

type Signature = (Tensor<F16>, Tensor<f32>, Float32, Layout, Shape, Int32, FastDivmod, Option<Ptr<u8>>);

/// The parameter list (names, order, types, divisibility, alignment) is what
/// each spec implies: the buffer, then each dynamic leaf once.
#[test]
fn declared_parameters_follow_the_specs() {
    let (m, ld) = (sym_int32(8, Some("M")), sym_int64(16, None));
    let a = TensorSpec::new::<F16>((m, (4, sym_int32(2, None))), (ld, (1, 4))).with_assumed_align(32);
    let b = TensorSpec::new::<f32>((16,), (1,));
    let l = LayoutSpec::new((sym_int32(1, None), 8), (sym_int64(8, None), 1));
    let shape: SpecTuple = (m, 3).into();
    let spec = (a, b, ScalarSpec::Plain, l, shape, ScalarSpec::Sym(m), (), Some(PtrSpec { assumed_align: 64 }));
    let k = Kernel::jit("signature", spec, |_: Signature| {});
    assert_eq!(
        signature(&k),
        [
            "arg0: buffer of f16 (align 32, len ?)",
            "arg0_shape_0: i32 div 8",
            "arg0_shape_1_1: i32 div 2",
            "arg0_stride_0: i64 div 16",
            "arg1: buffer of f32 (align 4, len 16)",
            "arg2: f32 div 1",
            "arg3_shape_0: i32 div 1",
            "arg3_stride_0: i64 div 8",
            // `M` again (arg4's leaf, arg5): the tensor's extent.
            "arg6_divisor: i32 div 1",
            "arg6_multiplier: u32 div 1",
            "arg6_shift: u32 div 1",
            "arg7: buffer of u8 (align 64, len ?)",
        ]
    );
    // A struct's fields are named after it.
    let k = Kernel::jit("fields", feature_spec(), |_: Pair| {});
    assert_eq!(signature(&k), ["x: buffer of i32 (align 4, len ?)", "x_shape_0: i32 div 1", "y: i32 div 1"]);
    let k = Kernel::jit("nested", (ScalarSpec::Plain, feature_spec()), |_: Int32, _: Pair| {});
    assert_eq!(
        signature(&k)[1..],
        ["arg1_x: buffer of i32 (align 4, len ?)", "arg1_x_shape_0: i32 div 1", "arg1_y: i32 div 1"]
    );
}

kernel_params! {
    struct Pair(PairSpec, PairHost) {
        x: Tensor<i32>,
        y: Int32,
        k: Constexpr<usize>,
    }
}

fn feature_spec() -> PairSpec {
    PairSpec { x: make_fake_compact_tensor::<i32>((sym_int32(1, None),), None, false), y: ScalarSpec::Plain, k: 4 }
}

/// `TensorSpec` prints as CuTe DSL's `TypedTensor`, checks tensors
/// (`isinstance`), and `param_tensor` takes it.
#[test]
fn tensor_spec_text_and_isinstance() {
    let s = TensorSpec::new::<f32>((16,), (1,));
    assert_eq!(s.to_string(), "Tensor<f32, (16,), (1,)>");
    assert_eq!((s.element_type(), s.memspace(), s.assumed_align()), (DType::F32, AddressSpace::Generic, 4));
    let n = sym_int32(4, None);
    let d = TensorSpec::new::<F16>((n, 8), (8, 1));
    assert_eq!(d.to_string(), "Tensor<f16, (?{div=4},8), (8,1)>");
    assert_eq!(d.layout_string(), "(?{div=4},8):(8,1)");
    let _ = Kernel::build("isinstance", || {
        let t = param_tensor::<F16>("t", &d);
        assert!(d.isinstance(&t));
        assert!(!s.isinstance(&param_tensor::<f32>("u", Layout::col_major(8))));
        assert!(s.isinstance(&param_tensor::<f32>("v", Layout::col_major((16,)))));
        // A runtime extent with less divisibility, or a static one, is not of the type.
        let m = param_extent("m", 2);
        assert!(!d.isinstance(&param_tensor::<F16>("w", Layout::new((m, 8), (8, 1)))));
        assert!(d.isinstance(&param_tensor::<F16>("x", Layout::new((m * 2, 8), (8, 1)))));
        assert!(!d.isinstance(&param_tensor::<F16>("y", Layout::new((4, 8), (8, 1)))));
        assert!(!TensorSpec::new::<f32>((n, 8), (8, 1)).isinstance(&t));
    });
}

/// A layout parameter on its own (`param_layout`), in shape and stride
/// forms, and a composed layout with a dynamic offset and outer layout.
#[test]
fn layouts_shapes_and_composed_layouts_as_parameters() {
    let (m, n) = (sym_int32(1, None), sym_int32(2, None));
    let compact = LayoutSpec::col_major((m, 4));
    let composed = ComposedSpec {
        inner: Inner::Swizzle(Swizzle::new(2, 2, 3)),
        offset: n.into(),
        outer: LayoutSpec::new((8, n), (n, 1)),
    };
    let k = Kernel::builder("layout_params").block_dim([32, 1, 1]).jit(
        (compact, composed, PtrSpec::default()),
        |l: Layout, c: ComposedLayout, out: Ptr<i32>| {
            let o = make_tensor(&out, Layout::col_major(32));
            let i = thread_idx().x;
            if_(i.lt(8), || {
                // The compact layout's strides come from its extents.
                o.store(i, Int32::from(crd2idx((i % 2, 1), &l)));
                o.store(i + 8, Int32::from(c.offset_of(&Coord::from((i, 1)))));
            });
        },
    );
    assert_eq!(k.param_names(), ["arg0_shape_0", "arg1_offset", "arg2"]);
    let host_c = ComposedLayout {
        inner: Inner::Swizzle(Swizzle::new(2, 2, 3)),
        offset: Tree::Leaf(IndexExpr::Static(6)),
        outer: Layout::new((8, 6), (6, 1)),
    };
    let mut host = (Layout::col_major((5, 4)), host_c.clone(), KernelArg::buffer(&[0i32; 32]));
    k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut host).unwrap();
    let out = host.2.to_vec::<i32>();
    for i in 0..8 {
        assert_eq!(out[i] as usize, i % 2 + 5, "compact {i}");
        assert_eq!(out[i + 8] as usize, host_c.offset_of(&Coord::from((i, 1))).as_static().unwrap(), "composed {i}");
    }
    // The host's static parts are checked: a strided host layout for a compact spec.
    let mut bad = (Layout::new((5, 4), (4, 1)), host_c, KernelArg::buffer(&[0i32; 32]));
    let e = k.pack(&bad.clone()).unwrap_err();
    assert_eq!(e.path, "arg0");
    assert!(e.message.contains("compact column-major"), "{e}");
    // The swizzle is part of the type.
    bad.0 = Layout::col_major((5, 4));
    bad.1.inner = Inner::Swizzle(Swizzle::new(3, 2, 3));
    assert!(k.simulate(cfg(), &mut bad).unwrap_err().message.contains("inner function"));
    // Shapes and coordinates: `param_shape` on its own.
    let _ = Kernel::build("shape_param", || {
        let s = param_shape("s", &(sym_int32(1, None), (3, sym_int64(1, None))).into());
        assert_eq!(s.to_string(), "(?,(3,?{i64}))");
    });
}

fn diag<P: KernelParam>(k: &TypedKernel<P>, host: &P::Host) -> ArgError {
    k.pack(host).expect_err("expected the launch to be refused")
}

/// Host values that do not match the spec are refused at launch, by the
/// simulator and the launcher alike (both pack through the spec), naming the
/// parameter path.
#[test]
fn mismatched_host_values_name_the_parameter() {
    let (m, ld) = (sym_int32(4, None), sym_int64(1, None));
    let spec = make_fake_tensor::<f32>((m, 8), (ld, 1)).with_assumed_align(16);
    let k = Kernel::jit("diag", (spec.clone(), Some(spec)), |_: Tensor<f32>, _: Option<Tensor<f32>>| {});
    let good = |m: usize| HostTensor::from_slice(&vec![0f32; m * 8], Layout::new((m, 8), (8, 1)));
    assert!(k.pack(&(good(4), Some(good(4)))).is_ok());

    // Wrong element type.
    let i32s = HostTensor::from_slice(&[0i32; 32], Layout::new((4, 8), (8, 1)));
    let e = diag(&k, &(i32s, Some(good(4))));
    assert_eq!(e.to_string(), "argument `arg0`: expected a tensor of Float32, got one of Int32");
    // A static extent or stride that differs.
    let e = diag(&k, &(HostTensor::from_slice(&[0f32; 36], Layout::new((4, 9), (9, 1))), Some(good(4))));
    assert_eq!(e.to_string(), "argument `arg0_shape_1`: the kernel was built for the static extent 8, launched with 9");
    let e = diag(&k, &(HostTensor::from_slice(&[0f32; 64], Layout::new((4, 8), (16, 2))), Some(good(4))));
    assert_eq!(
        e.to_string(),
        "argument `arg0_stride_1`: the kernel was built for the static stride 1, launched with 2"
    );
    // Divisibility.
    let e = diag(&k, &(good(6), Some(good(6))));
    assert_eq!(e.to_string(), "argument `arg0_shape_0`: ?{div=4} must be a multiple of 4, got 6");
    // Too few elements.
    let short = HostTensor::from_slice(&[0f32; 30], Layout::new((4, 8), (8, 1)));
    let e = diag(&k, &(short, Some(good(4))));
    assert_eq!(e.to_string(), "argument `arg0`: a tensor of layout (4,8):(8,1) needs 32 elements, the buffer has 30");
    // A buffer whose start is not as aligned as the kernel assumes.
    let e = diag(&k, &(good(4).with_assumed_align(4), Some(good(4))));
    assert_eq!(
        e.to_string(),
        "argument `arg0`: the kernel assumes a 16-byte-aligned start, but the host tensor promises only 4 bytes"
    );
    // Some versus None.
    let e = diag(&k, &(good(4), None));
    assert_eq!(e.to_string(), "argument `arg1`: the kernel was built with a value (Some), launched with None");
    let none = Kernel::jit("diag_none", (good(4).spec().clone(), None), |_: Tensor<f32>, _: Option<Tensor<f32>>| {});
    let e = diag(&none, &(good(4), Some(good(4))));
    assert_eq!(e.to_string(), "argument `arg1`: the kernel was built without a value (None), launched with Some");
    // A different value for the SymInt the two tensors share.
    let e = diag(&k, &(good(4), Some(good(8))));
    assert_eq!(
        e.to_string(),
        "argument `arg1_shape_0`: got 8, but `arg0_shape_0` is the same SymInt ?{div=4} and got 4"
    );
    // A tuple of the wrong arity: a shape, a layout, a Vec.
    let t = Kernel::jit(
        "arity",
        ((sym_int32(1, None), 2).into(), vec![ScalarSpec::Plain; 2]),
        |_: Shape, _: Vec<Int32>| {},
    );
    let e = diag(&t, &((1, 2, 3).into(), vec![1, 2]));
    assert_eq!(e.to_string(), "argument `arg0`: the kernel takes a tuple of profile (?,2), got (1,2,3)");
    let e = diag(&t, &((1, 2).into(), vec![1, 2, 3]));
    assert_eq!(e.to_string(), "argument `arg1`: the kernel takes 2 elements, got 3");
    let l = Kernel::jit("arity_layout", LayoutSpec::new((sym_int32(1, None), 2), (1, 4)), |_: Layout| {});
    let e = diag(&l, &Layout::new((2, 2, 2), (1, 2, 4)));
    assert_eq!(e.to_string(), "argument `arg0`: the kernel takes a layout of shape (?,2), got (2,2,2):(1,2,4)");
    // The simulator reports the same.
    let mut host = (good(4), None);
    let e = k.simulate(cfg(), &mut host).unwrap_err();
    assert_eq!(e.message, "argument `arg1`: the kernel was built with a value (Some), launched with None");
}

/// A static stride of a mode of extent 1 takes whatever stride the host
/// tensor has there (`mark_compact_shape_dynamic` makes it 0), as it never
/// moves an offset: the kernel of a K-major `(m, k)` runs with `k = 1`.
/// Other static strides are still checked.
#[test]
fn static_stride_of_a_unit_extent_takes_any_host_stride() {
    let k_major = |m: usize, k: usize| {
        HostTensor::from_slice(&vec![0f32; m * k], Layout::new((m, k), (k, 1)))
            .mark_layout_dynamic(Some(1))
            .and_then(|t| t.mark_compact_shape_dynamic(1, Some(&[0, 1]), 1))
            .unwrap()
    };
    let k = Kernel::jit_for("unit_extent", &k_major(4, 8), |_: Tensor<f32>| {});
    assert_eq!(k.spec().layout_string(), "(?,?):(?{i64},1)");
    // Whatever host stride the unit mode gets (0 or its compact 1), it packs.
    assert!(k.pack(&k_major(4, 1)).is_ok());
    let e = diag(&k, &HostTensor::from_slice(&[0f32; 64], Layout::new((4, 8), (16, 2))));
    assert_eq!(
        e.to_string(),
        "argument `arg0_stride_1`: the kernel was built for the static stride 1, launched with 2"
    );
}

/// A misaligned device buffer is the CUDA launcher's to catch; in the
/// simulator the declared alignment is checked on every access. The spec's
/// alignment is the buffer's.
#[test]
fn spec_alignment_reaches_the_kernel() {
    let spec = TensorSpec::new::<f32>((sym_int32(1, None),), (1,)).with_assumed_align(64);
    let k = Kernel::jit("aligned", spec, |_: Tensor<f32>| {});
    assert_eq!(signature(&k)[0], "arg0: buffer of f32 (align 64, len ?)");
    let mlir = k.to_mlir();
    assert!(mlir.contains("llvm.align = 64"), "{mlir}");
    assert!(
        build_panic(|| {
            let _ = TensorSpec::new::<f32>((4,), (1,)).with_assumed_align(2);
        })
        .contains("not a power of two of at least the 4 bytes of f32")
    );
}

// ---------------------------------------------------------------------------
// Mark_layout_dynamic, mark_compact_shape_dynamic, SymInt
// ---------------------------------------------------------------------------

/// A host tensor of `shape:stride` (no data needed to mark it).
fn ht(shape: impl Into<IntTuple>, stride: impl Into<IntTuple>) -> HostTensor {
    let (shape, stride): (IntTuple, IntTuple) = (shape.into(), stride.into());
    HostTensor::new(KernelArg::buffer::<f32>(&[]), Layout::new(&shape, &stride))
}

/// `(8,4,16,2):(2,16,64,1)`, the guide's `a`.
fn a() -> HostTensor {
    ht((8, 4, 16, 2), (2, 16, 64, 1))
}

/// `(1,4,1,32,1):(1,1,1,4,1)`, the guide's `b`.
fn b() -> HostTensor {
    ht((1, 4, 1, 32, 1), (1, 1, 1, 4, 1))
}

/// `(2,2):(8,2)`, the guide's `c`.
fn c() -> HostTensor {
    ht((2, 2), (8, 2))
}

/// `(3,4,2,5):(5,0,0,1)`, the guide's broadcast `d`.
fn d() -> HostTensor {
    ht((3, 4, 2, 5), (5, 0, 0, 1))
}

fn text(t: Result<HostTensor, SpecError>) -> String {
    match t {
        Ok(t) => t.layout_string(),
        Err(e) => e.to_string(),
    }
}

/// The guide's `mark_layout_dynamic` cases t0 to t8. Its printed results
/// show 32-bit strides for t0 to t4 (`?`), as `use_32bit_stride` gives; by
/// default strides are 64-bit, as its t5 shows (`?{i64}`).
#[test]
fn mark_layout_dynamic_guide_cases() {
    let w = |t: HostTensor| t.with_32bit_stride(true);
    assert_eq!(text(w(a()).mark_layout_dynamic(None)), "(?,?,?,?):(?,?,?,1)");
    assert_eq!(text(a().mark_layout_dynamic(None)), "(?,?,?,?):(?{i64},?{i64},?{i64},1)");
    assert_eq!(text(w(b()).mark_layout_dynamic(Some(0))), "(?,?,?,?,?):(1,?,?,?,?)");
    assert_eq!(text(w(b()).mark_layout_dynamic(Some(2))), "(?,?,?,?,?):(?,?,1,?,?)");
    assert_eq!(text(w(c()).mark_layout_dynamic(None)), "(?,?):(?,?)");
    assert_eq!(text(w(d()).mark_layout_dynamic(None)), "(?,?,?,?):(?,0,0,1)");
    assert_eq!(text(b().mark_layout_dynamic(None)), "(?,?,?,?,?):(?{i64},1,?{i64},?{i64},?{i64})");
    assert_eq!(
        text(ht((1, 1, 1), (1, 1, 1)).mark_layout_dynamic(None)),
        "Can't deduce the leading dimension from layout: several modes have stride 1 and none of them has size > 1; \
         specify leading_dim"
    );
    assert_eq!(text(a().mark_layout_dynamic(Some(1))), "Expected strides[leading_dim] == 1, but got 16");
    assert_eq!(text(b().mark_layout_dynamic(Some(3))), "Expected strides[leading_dim] == 1, but got 4");
    let huge = ht((1_000_000_000, 1_000_000_000), (1_000_000_000, 1)).with_32bit_stride(true);
    assert_eq!(
        text(huge.clone().mark_layout_dynamic(None)),
        "Layout in HostTensor has int32 overflow risk. Please set use_32bit_stride to false."
    );
    assert_eq!(text(huge.with_32bit_stride(false).mark_layout_dynamic(None)), "(?,?):(?{i64},1)");
    // The guide's other examples: a negative leading_dim, (1,5,1):(1,1,1).
    assert_eq!(text(w(ht((2, 2, 3, 4), (2, 1, 4, 12))).mark_layout_dynamic(Some(1))), "(?,?,?,?):(?,1,?,?)");
    assert_eq!(
        text(w(ht((2, 2, 3, 4), (2, 1, 4, 12))).mark_layout_dynamic(Some(-1))),
        "Expected strides[leading_dim] == 1, but got 12"
    );
    assert_eq!(text(w(ht((2, 2, 3, 1), (2, 1, 4, 1))).mark_layout_dynamic(Some(-1))), "(?,?,?,?):(?,?,?,1)");
    assert_eq!(text(w(ht((1, 5, 1), (1, 1, 1))).mark_layout_dynamic(None)), "(?,?,?):(?,1,?)");
    assert!(text(ht(((2, 2), 4), ((1, 2), 4)).mark_layout_dynamic(None)).contains("flat layouts only"));
}

/// The guide's `mark_compact_shape_dynamic` cases t0 to t13 (32-bit strides
/// for its printed results, as above).
#[test]
fn mark_compact_shape_dynamic_guide_cases() {
    let w = |t: HostTensor| t.with_32bit_stride(true);
    let t0 = w(a()).mark_compact_shape_dynamic(0, None, 2);
    assert_eq!(text(t0), "(?{div=2},4,16,2):(2,?{div=4},?{div=16},1)");
    assert_eq!(text(a().mark_compact_shape_dynamic(0, None, 2)), "(?{div=2},4,16,2):(2,?{i64 div=4},?{i64 div=16},1)");
    assert_eq!(text(w(a()).mark_compact_shape_dynamic(1, None, 2)), "(8,?{div=2},16,2):(2,16,?{div=32},1)");
    let t2 = w(a()).mark_compact_shape_dynamic(1, None, 2).and_then(|t| t.mark_compact_shape_dynamic(3, None, 2));
    assert_eq!(text(t2.clone()), "(8,?{div=2},16,?{div=2}):(?{div=2},?{div=16},?{div=32},1)");
    assert_eq!(
        text(w(b()).mark_compact_shape_dynamic(2, Some(&[3, 0, 2, 4, 1]), 1)),
        "(1,4,?,32,1):(0,1,4,?{div=4},0)"
    );
    assert_eq!(text(w(b()).mark_compact_shape_dynamic(2, Some(&[2, 3, 4, 0, 1]), 1)), "(1,4,?,32,1):(0,1,128,4,0)");
    assert_eq!(
        text(t2.unwrap().mark_compact_shape_dynamic(3, Some(&[0, 1, 2, 3]), 5)),
        "The stride_order is not consistent with the last stride_order"
    );
    assert_eq!(
        text(a().mark_compact_shape_dynamic(3, Some(&[0, 1, 2, 3]), 5)),
        "The stride_order is not consistent with the deduced stride_order"
    );
    assert_eq!(
        text(b().mark_compact_shape_dynamic(0, None, 4)),
        "The layout could not be deduced, please specify the stride_order explicitly"
    );
    assert_eq!(
        text(b().mark_compact_shape_dynamic(30, Some(&[3, 0, 2, 4, 1]), 5)),
        "Expected mode value to be in range [0, 5), but got 30"
    );
    assert_eq!(
        text(b().mark_compact_shape_dynamic(3, Some(&[2, 1, 2, 3, 4]), 5)),
        "Expected stride_order to contain all the dimensions of the tensor, but it doesn't contain 0."
    );
    assert_eq!(
        text(b().mark_compact_shape_dynamic(3, Some(&[0, 1, 2, 3, 4, 5]), 5)),
        "Expected stride_order to have 5 elements, but got 6."
    );
    // b.dim_order() is (3,2,4,0,1).
    assert_eq!(
        text(b().mark_compact_shape_dynamic(0, Some(&[3, 2, 4, 0, 1]), 4)),
        "The shape(1) of mode(0) is not divisible by the divisibility(4)"
    );
    assert_eq!(
        text(b().mark_compact_shape_dynamic(0, Some(&[2, 1, 3, 0, 4]), 1)),
        "The stride_order is not consistent with the layout"
    );
    let huge = ht((1_000_000_000, 1_000_000_000), (1_000_000_000, 1)).with_32bit_stride(true);
    assert_eq!(
        text(huge.mark_compact_shape_dynamic(0, None, 1)),
        "Layout in HostTensor has int32 overflow risk. Please set use_32bit_stride to false."
    );
    // The guide's deduction examples, and a non-compact layout.
    assert_eq!(a().mark_compact_shape_dynamic(0, None, 1).unwrap().stride_order(), Some(&[2, 1, 0, 3][..]));
    assert_eq!(
        ht((2, 2, 3, 4), (2, 1, 4, 12)).mark_compact_shape_dynamic(0, None, 1).unwrap().stride_order(),
        Some(&[3, 2, 0, 1][..])
    );
    assert_eq!(
        text(ht((4, 2), (1, 8)).mark_compact_shape_dynamic(0, None, 1)),
        "The stride_order is not consistent with the layout"
    );
}

/// `SymInt` printing, identity and arithmetic (CuTe DSL's rules).
#[test]
fn symint_text_and_arithmetic() {
    let s = sym_int32(8, None);
    assert_eq!(s.to_string(), "?{div=8}");
    assert_eq!(sym_int64(1, None).to_string(), "?{i64 div=1}");
    assert_eq!(sym_int(64, 16, Some("LD")).to_string(), "LD ?{i64 div=16}");
    assert_eq!((s.width(), s.divisibility(), s.symbol(), s.dtype()), (32, 8, None, DType::I32));
    // Equality compares width, divisibility and symbol; identity is separate.
    let t = sym_int32(8, None);
    assert!(s == t && !s.same(&t) && s.same(&{ s }));
    assert!(sym_int32(8, Some("M")) != s);
    // * multiplies the divisibility, % gives 0 or the gcd, // divides it when exact.
    assert_eq!((s * 3).divisibility(), 24);
    assert_eq!((3 * s).divisibility(), 24);
    assert_eq!((s * sym_int64(4, None)).to_string(), "?{i64 div=32}");
    assert_eq!(s % 4, SymRem::Zero);
    let SymRem::Sym(r) = s % 12 else { panic!("8 does not divide by 12") };
    assert_eq!(r.divisibility(), 4);
    let SymRem::Sym(r) = s % sym_int64(3, None) else { panic!() };
    assert_eq!(r.to_string(), "?{i64 div=1}");
    assert_eq!(20 % s, 4);
    assert_eq!((s / 4).divisibility(), 2);
    assert_eq!((s / 3).divisibility(), 1);
    assert_eq!((s / sym_int32(2, None)).divisibility(), 1);
    assert!(!(s * 2).same(&s), "arithmetic gives a fresh SymInt");
    assert!(
        build_panic(|| {
            let _ = sym_int(16, 1, None);
        })
        .contains("unsupported width 16")
    );
}

/// The five specs of CuTe DSL's `compile_with_fake_tensor.py`, and the
/// fake-tensor constructors' examples.
#[test]
fn fake_tensor_specs() {
    let a = make_fake_compact_tensor::<F16>((3, 4), Some(&[1, 0]), false);
    assert_eq!(a.to_string(), "Tensor<f16, (3,4), (4,1)>");
    let n = sym_int32(8, None);
    let a = make_fake_compact_tensor::<F16>((3, n), Some(&[1, 0]), false);
    assert_eq!(a.to_string(), "Tensor<f16, (3,?{div=8}), (?{i64 div=8},1)>");
    let a = make_fake_tensor::<F16>((3, n), (4, 1));
    assert_eq!(a.to_string(), "Tensor<f16, (3,?{div=8}), (4,1)>");
    let a = make_fake_tensor::<F16>((3, n), (sym_int32(8, None), 1));
    assert_eq!(a.to_string(), "Tensor<f16, (3,?{div=8}), (?{div=8},1)>");
    let a = make_fake_tensor::<F16>((3, n), (sym_int64(8, None), 1));
    assert_eq!(a.to_string(), "Tensor<f16, (3,?{div=8}), (?{i64 div=8},1)>");
    assert_eq!((a.memspace(), a.assumed_align()), (AddressSpace::Gmem, 2));
    // make_fake_compact_tensor's docstring examples.
    let x = make_fake_compact_tensor::<f32>((100, sym_int32(8, None)), Some(&[1, 0]), true);
    assert_eq!(x.layout_string(), "(100,?{div=8}):(?{div=8},1)");
    let y = make_fake_compact_tensor::<f32>((8, 3, 2), None, false);
    assert_eq!(y.layout_string(), "(8,3,2):(1,8,24)");
    // One SymInt in several places is one parameter.
    let (m, ld) = (sym_int64(1, Some("M")), sym_int64(16, Some("LD")));
    let x = make_fake_tensor::<F16>((m, 128), (ld, 1));
    let k = Kernel::jit("fake", (x.clone(), x, ScalarSpec::Sym(m)), |_: Tensor<F16>, _: Tensor<F16>, _: Int64| {});
    assert_eq!(
        signature(&k),
        [
            "arg0: buffer of f16 (align 2, len ?)",
            "arg0_shape_0: i64 div 1",
            "arg0_stride_0: i64 div 16",
            "arg1: buffer of f16 (align 2, len ?)",
        ]
    );
}

/// Launch-time checks of `SymInt`s: one value for every occurrence, the
/// divisibility, and the static modes of a marked tensor.
#[test]
fn symint_launch_checks() {
    let n = sym_int32(4, Some("N"));
    let t = make_fake_compact_tensor::<i32>((n,), None, false);
    let k = Kernel::builder("sym_checks").block_dim([32, 1, 1]).jit(
        (t.clone(), t, ScalarSpec::Sym(n)),
        |x: Tensor<i32>, y: Tensor<i32>, len: Int32| {
            let i = thread_idx().x;
            if_(i.lt(len), || y.store(i, x.load(i) + len));
        },
    );
    let h = |n: usize| HostTensor::from_slice(&vec![1i32; n], Layout::col_major((n,)));
    let mut ok = (h(8), h(8), 8);
    k.simulate(cfg(), &mut ok).unwrap();
    assert_eq!(ok.1.to_vec::<i32>(), [9; 8]);
    // Different values for one SymInt.
    let e = k.simulate(cfg(), &mut (h(8), h(12), 8)).unwrap_err();
    assert_eq!(
        e.message,
        "argument `arg1_shape_0`: got 12, but `arg0_shape_0` is the same SymInt `N` (?{div=4}) and got 8"
    );
    let e = k.simulate(cfg(), &mut (h(8), h(8), 4)).unwrap_err();
    assert_eq!(e.message, "argument `arg2`: got 4, but `arg0_shape_0` is the same SymInt `N` (?{div=4}) and got 8");
    // A divisibility violation.
    let e = k.simulate(cfg(), &mut (h(6), h(6), 6)).unwrap_err();
    assert_eq!(e.message, "argument `arg0_shape_0`: `N` (?{div=4}) must be a multiple of 4, got 6");
    // A static mode that differs from a marked tensor's spec.
    let host =
        HostTensor::from_slice(&[0f32; 32], Layout::row_major((4, 8))).mark_compact_shape_dynamic(0, None, 2).unwrap();
    assert_eq!(host.layout_string(), "(?{div=2},8):(8,1)");
    let k = Kernel::jit_for("marked", &host, |_: Tensor<f32>| {});
    let e = k.pack(&HostTensor::from_slice(&[0f32; 36], Layout::row_major((4, 9)))).unwrap_err();
    assert_eq!(e.to_string(), "argument `arg0_shape_1`: the kernel was built for the static extent 8, launched with 9");
    assert!(k.pack(&HostTensor::from_slice(&[0f32; 48], Layout::row_major((6, 8)))).is_ok());
}

// ---------------------------------------------------------------------------
// Typed entry points, Constexpr, struct arguments, native structs
// ---------------------------------------------------------------------------

kernel_params! {
    /// CuTe DSL's `dataclass_immutable.py`: one static attribute, two tensors.
    pub struct CopyParams(CopyParamsSpec, CopyParamsHost) {
        pub num_threads: usize,
        pub src: Tensor<f32>,
        pub dst: Tensor<f32>,
    }
}

/// The port of `dataclass_immutable.py`: fake tensors over one 64-bit
/// `SymInt`, compiled once, run on real host tensors.
#[test]
fn dataclass_immutable_port() {
    let n = sym_int64(1, None);
    let fake = make_fake_compact_tensor::<f32>((n,), Some(&[0]), false);
    let spec = (CopyParamsSpec { num_threads: 128, src: fake.clone(), dst: fake }, ScalarSpec::Sym(n));
    let threads = spec.0.num_threads as u32;
    let k = Kernel::builder("memcpy_kernel").block_dim([threads, 1, 1]).jit(spec, |params: CopyParams, len: Int64| {
        let idx = block_idx().x * params.num_threads as i32 + thread_idx().x;
        if_(idx.to::<i64>().lt(len), || params.dst.store(idx, params.src.load(idx)));
    });
    assert_eq!(k.param_names(), ["arg0_src", "arg0_src_shape_0", "arg0_dst"]);
    for len in [1024usize, 1000] {
        let src: Vec<f32> = (0..len).map(|i| i as f32 * 0.25 - 3.0).collect();
        let t = |v: &[f32]| HostTensor::from_slice(v, Layout::col_major((len,)));
        let mut host = (CopyParamsHost { num_threads: 128, src: t(&src), dst: t(&vec![0.0; len]) }, len as i64);
        let blocks = len.div_ceil(128) as u32;
        k.simulate(LaunchConfig::new([blocks, 1, 1], [threads, 1, 1]), &mut host).unwrap();
        assert_eq!(host.0.dst.to_vec::<f32>(), src);
    }
    // The constant field is checked at launch.
    let t = |n: usize| HostTensor::from_slice(&vec![0f32; n], Layout::col_major((n,)));
    let e = k.pack(&(CopyParamsHost { num_threads: 64, src: t(8), dst: t(8) }, 8)).unwrap_err();
    assert_eq!(
        e.to_string(),
        "argument `arg0_num_threads`: the kernel was built with the constant 128, launched with 64"
    );
}

kernel_params! {
    /// `08_mlir_value_tree_debug.py`'s `Coord2D`.
    struct Coord2D(Coord2DSpec, Coord2DHost) {
        x: Float32,
        y: Float32,
    }
}

kernel_params! {
    /// Its `Particle`: two nested structs.
    struct Particle(ParticleSpec, ParticleHost) {
        position: Coord2D,
        velocity: Coord2D,
    }
}

kernel_params! {
    /// Its `Accumulator`, with its configuration as constants.
    struct Accumulator(AccumulatorSpec, AccumulatorHost) {
        running_sum: Float32,
        count: Int32,
        current_max: Float32,
        block_size: usize,
        use_fast_math: bool,
    }
}

impl Coord2D {
    fn add(&self, o: &Coord2D) -> Coord2D {
        Coord2D { x: self.x + o.x, y: self.y + o.y }
    }
    fn scale(&self, s: f32) -> Coord2D {
        Coord2D { x: self.x * s, y: self.y * s }
    }
}

/// The struct-through-loop pattern of `08_mlir_value_tree_debug.py`: nested
/// structs and a struct with constant fields carried through runtime loops;
/// a body that changes a constant field is refused, naming the field.
#[test]
fn structs_through_loops() {
    let spec = (
        ParticleSpec {
            position: Coord2DSpec { x: ScalarSpec::Plain, y: ScalarSpec::Plain },
            velocity: Coord2DSpec { x: ScalarSpec::Plain, y: ScalarSpec::Plain },
        },
        AccumulatorSpec {
            running_sum: ScalarSpec::Plain,
            count: ScalarSpec::Plain,
            current_max: ScalarSpec::Plain,
            block_size: 4,
            use_fast_math: true,
        },
        PtrSpec::default(),
    );
    let k = Kernel::builder("value_tree").block_dim([32, 1, 1]).jit(
        spec,
        |p: Particle, acc: Accumulator, out: Ptr<f32>| {
            let o = make_tensor(&out, Layout::col_major(5));
            let p = for_range_carry(0, 3, 1, p, |_, p| Particle {
                position: p.position.add(&p.velocity),
                velocity: p.velocity.scale(0.5),
            });
            let acc = for_range_carry(0, acc.block_size as i32, 1, acc, |i, a| {
                let v = i.to::<f32>() * 1.5;
                Accumulator {
                    running_sum: a.running_sum + v,
                    count: a.count + 1,
                    current_max: fmax(a.current_max, v),
                    ..a
                }
            });
            if_(thread_idx().x.eq(0), || {
                o.store(0, p.position.x);
                o.store(1, p.position.y);
                o.store(2, acc.running_sum / acc.count.to::<f32>());
                o.store(3, acc.current_max);
                o.store(4, p.velocity.x);
            });
        },
    );
    assert_eq!(
        k.param_names(),
        [
            "arg0_position_x",
            "arg0_position_y",
            "arg0_velocity_x",
            "arg0_velocity_y",
            "arg1_running_sum",
            "arg1_count",
            "arg1_current_max",
            "arg2"
        ]
    );
    let c = |x, y| Coord2DHost { x, y };
    let mut host = (
        ParticleHost { position: c(0.0, 1.0), velocity: c(4.0, -2.0) },
        AccumulatorHost { running_sum: 0.0, count: 0, current_max: -1.0, block_size: 4, use_fast_math: true },
        KernelArg::buffer(&[0f32; 5]),
    );
    k.simulate(cfg(), &mut host).unwrap();
    // Position: 0 + 4 + 2 + 1 = 7, 1 - 2 - 1 - 0.5 = -2.5; velocity 4 / 8.
    assert_eq!(host.2.to_vec::<f32>(), [7.0, -2.5, (0.0 + 1.5 + 3.0 + 4.5) / 4.0, 4.5, 0.5]);

    // A loop body that changes a constant field.
    let e = build_panic(|| {
        let _ = Kernel::jit("changed", spec_acc(), |acc: Accumulator| {
            let _ = for_range_carry(0, 2, 1, acc, |_, a| Accumulator { block_size: 8, ..a });
        });
    });
    assert!(e.contains("field block_size: the constant 4 became 8"), "{e}");
}

fn spec_acc() -> AccumulatorSpec {
    AccumulatorSpec {
        running_sum: ScalarSpec::Plain,
        count: ScalarSpec::Plain,
        current_max: ScalarSpec::Plain,
        block_size: 4,
        use_fast_math: true,
    }
}

/// `Constexpr` parameters declare nothing, are fixed when building, and are
/// checked at launch; host values for the wrong signature do not compile
/// (see `Kernel::jit`'s documentation), and a raw argument list of the
/// wrong length is refused as before.
#[test]
fn constexpr_and_signature_checks() {
    let k = Kernel::builder("constexpr").block_dim([32, 1, 1]).jit(
        (ScalarSpec::Plain, 3usize, PtrSpec::default()),
        |x: Int32, tile: Constexpr<usize>, out: Ptr<i32>| {
            let o = make_tensor(&out, Layout::col_major(*tile));
            let i = thread_idx().x;
            if_(i.lt(*tile as i32), || o.store(i, x * *tile as i32 + i));
        },
    );
    assert_eq!(k.param_names(), ["arg0", "arg2"]);
    let mut host = (5, 3, KernelArg::buffer(&[0i32; 3]));
    k.simulate(cfg(), &mut host).unwrap();
    assert_eq!(host.2.to_vec::<i32>(), [15, 16, 17]);
    let e = k.simulate(cfg(), &mut (5, 4, KernelArg::buffer(&[0i32; 4]))).unwrap_err();
    assert_eq!(e.message, "argument `arg1`: the kernel was built with the constant 3, launched with 4");
    // The untyped path still checks against the declared parameters.
    let e = k.kernel().simulate(cfg(), &mut [KernelArg::scalar(5i32)]).unwrap_err();
    assert_eq!(e.message, "kernel `constexpr` takes 2 arguments, 1 were given");
    let e = k.kernel().simulate(cfg(), &mut [KernelArg::scalar(5.0f32), KernelArg::buffer(&[0i32; 3])]).unwrap_err();
    assert_eq!(e.message, "argument 0 (`arg0`) must be a i32 scalar, got a f32 scalar");
    // Call-site typing: plain host values give plain scalars and constants.
    let k = Kernel::jit_for("from_host", &(1i32, 2.5f32, 7usize), |_: Int32, _: Float32, n: usize| assert_eq!(n, 7));
    assert_eq!(signature(&k), ["arg0: i32 div 1", "arg1: f32 div 1"]);
    let k = Kernel::jit_for("shape_from_host", &IntTuple::from((4, 8)), |s: Shape| assert_eq!(s.to_string(), "(?,?)"));
    assert_eq!(k.param_names(), ["arg0_0", "arg0_1"]);
}

kernel_params! {
    #[native_struct(zero_init = true, packed = false)]
    struct Native(NativeSpec, NativeHost) {
        a: Int8,
        b: Float32,
        c: Int16,
        n: Constexpr<usize>,
        d: Int64,
    }
}

kernel_params! {
    #[native_struct(zero_init = false, packed = true)]
    struct Packed(PackedSpec, PackedHost) {
        a: Int8,
        b: Float32,
        inner: Native,
    }
}

kernel_params! {
    #[native_struct(zero_init = true, packed = false)]
    struct Outer(OuterSpec, OuterHost) {
        a: Int8,
        inner: Native,
        z: Int16,
    }
}

fn native_spec() -> NativeSpec {
    NativeSpec { a: ScalarSpec::Plain, b: ScalarSpec::Plain, c: ScalarSpec::Plain, n: 2, d: ScalarSpec::Plain }
}

/// Native structs: the C layout (natural and packed; constants take no
/// space), zero initialisation, and flattening into scalar parameters.
#[test]
fn native_structs() {
    let l = Native::native_layout(&native_spec());
    let fields: Vec<(&str, usize)> = l.fields.iter().map(|f| (f.name.as_str(), f.offset)).collect();
    assert_eq!(fields, [("a", 0), ("b", 4), ("c", 8), ("d", 16)]);
    assert_eq!((l.size, l.align, l.packed), (24, 8, false));
    let p = Packed::native_layout(&PackedSpec { a: ScalarSpec::Plain, b: ScalarSpec::Plain, inner: native_spec() });
    let fields: Vec<(&str, usize)> = p.fields.iter().map(|f| (f.name.as_str(), f.offset)).collect();
    // A nested native struct keeps its own layout (padding included) at
    // offset 5, its alignment being 1 inside a packed struct.
    assert_eq!(fields, [("a", 0), ("b", 1), ("inner_a", 5), ("inner_b", 9), ("inner_c", 13), ("inner_d", 21)]);
    assert_eq!((p.size, p.align), (29, 1));
    // Inside a natural struct it sits at its own alignment (8).
    let o = Outer::native_layout(&OuterSpec { a: ScalarSpec::Plain, inner: native_spec(), z: ScalarSpec::Plain });
    let fields: Vec<(&str, usize)> = o.fields.iter().map(|f| (f.name.as_str(), f.offset)).collect();
    assert_eq!(fields, [("a", 0), ("inner_a", 8), ("inner_b", 12), ("inner_c", 16), ("inner_d", 24), ("z", 32)]);
    assert_eq!((o.size, o.align), (40, 8));
    const { assert!(Native::ZERO_INIT && !Native::PACKED && !Packed::ZERO_INIT && Packed::PACKED) };
    let k = Kernel::builder("native").block_dim([32, 1, 1]).jit(
        (native_spec(), PtrSpec::default()),
        |s: Native, out: Ptr<i64>| {
            let o = make_tensor(&out, Layout::col_major(3));
            let z = Native::make(&native_spec());
            if_(thread_idx().x.eq(0), || {
                o.store(0, s.a.to::<i64>() + s.c.to::<i64>() + s.d);
                o.store(1, z.a.to::<i64>() + z.c.to::<i64>() + z.d + *z.n as i64);
                o.store(2, s.b.to::<i64>());
            });
        },
    );
    assert_eq!(k.param_names(), ["arg0_a", "arg0_b", "arg0_c", "arg0_d", "arg1"]);
    let mut host = (NativeHost { a: -3, b: 7.9, c: 300, n: 2, d: 1 << 40 }, KernelArg::buffer(&[0i64; 3]));
    k.simulate(cfg(), &mut host).unwrap();
    assert_eq!(host.1.to_vec::<i64>(), [(1 << 40) + 297, 2, 7]);
    // A tensor is not a native-struct field.
    assert!(
        build_panic(|| {
            let _ = Kernel::build("not_native", || {
                let _ = <Tensor<f32> as KernelParam>::native_init(&TensorSpec::new::<f32>((4,), (1,)), true);
            });
        })
        .contains("cannot be a field of a native struct")
    );
}

/// `mark_compact_shape_dynamic` on a mode of extent 1: the spec keeps its
/// compact stride (the guide's t3 and t4), and so does the host tensor, so
/// a kernel built from it takes it.
#[test]
fn marked_unit_mode_keeps_its_compact_stride() {
    let mark = |order: &[usize]| {
        HostTensor::from_slice(&[0f32; 128], Layout::new((1, 4, 1, 32, 1), (1, 1, 1, 4, 1)))
            .mark_compact_shape_dynamic(2, Some(order), 1)
            .unwrap()
    };
    let t3 = mark(&[3, 0, 2, 4, 1]);
    assert_eq!(t3.layout().to_string(), "(1,4,1,32,1):(0,1,4,4,0)");
    let t4 = mark(&[2, 3, 4, 0, 1]);
    assert_eq!(t4.layout().to_string(), "(1,4,1,32,1):(0,1,128,4,0)");
    for t in [t3, t4] {
        let k = Kernel::jit_for("unit_mode", &t, |_: Tensor<f32>| {});
        assert!(k.pack(&t).is_ok(), "{}", k.pack(&t).unwrap_err());
    }
}
