//! `TensorSsa`: the instructions whole-tensor operations emit, and the
//! build panics for their misuse. The kernels themselves are feature
//! kernels (`tests/feature_kernels/tensor_ssa.rs`).

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;

mod feature_kernels;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn toolchain() -> Option<Toolchain> {
    let tc = Toolchain::discover();
    if tc.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    tc
}

fn ptx(kernel: &Kernel, tc: &Toolchain) -> String {
    kernel.compile_ptx(tc, Arch::Sm80).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()))
}

fn count(text: &str, pattern: &str) -> usize {
    text.lines().filter(|l| l.contains(pattern)).count()
}

/// A row-major 4 x 4 Float32 tile loads as four 16-byte vectors (through a
/// row-major fragment and autovec_copy), and stores the same way.
#[test]
fn load_ssa_of_a_row_major_tile_is_four_vector_loads() {
    let Some(tc) = toolchain() else { return };
    let kernel = Kernel::builder("load_ssa_vectors").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::row_major((128, 4)));
        let y = param_tensor::<f32>("y", Layout::row_major((128, 4)));
        let t = thread_idx().x;
        let v = local_tile(&x, (4, 4), (t, 0)).load_ssa();
        local_tile(&y, (4, 4), (t, 0)).store_ssa(&(v + 1.0));
    });
    let p = ptx(&kernel, &tc);
    assert_eq!(count(&p, "ld.global.v4."), 4, "{p}");
    assert_eq!(count(&p, "st.global.v4."), 4, "{p}");
}

/// One scalar operation per element, whatever the operands: a scalar or
/// broadcast operand repeats a value, not an instruction.
#[test]
fn one_scalar_op_per_element() {
    let kernel = Kernel::builder("one_op_per_element").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((4, 8)));
        let c = param_tensor::<f32>("c", Layout::col_major(4));
        let y = param_tensor::<f32>("y", Layout::col_major((4, 8)));
        let v = x.load_ssa();
        let col = c.load_ssa();
        let s = Value::constant(2.0f32);
        y.store_ssa(&((v * s) + col));
    });
    let mlir = kernel.to_mlir();
    assert_eq!(count(&mlir, "arith.mulf"), 32, "{mlir}");
    assert_eq!(count(&mlir, "arith.addf"), 32, "{mlir}");
}

/// Broadcasting only repeats values: a broadcast operand gives the same
/// IR as the repeated values written out.
#[test]
fn broadcasting_emits_nothing() {
    let build = |explicit: bool| {
        Kernel::builder("broadcast").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<i32>("x", Layout::col_major(4));
            let y = param_tensor::<i32>("y", Layout::col_major((4, 8)));
            let v = x.load_ssa();
            let wide = if explicit {
                let vs = v.values();
                TensorSsa::new((4, 8), (0..32).map(|k| vs[k % 4]).collect())
            } else {
                v.broadcast_to((4, 8))
            };
            y.store_ssa(&wide);
        })
    };
    assert_eq!(build(false).to_mlir(), build(true).to_mlir());
}

#[test]
fn shapes_that_do_not_broadcast_are_refused() {
    let msg = build_panic(|| {
        Kernel::build("no_broadcast", || {
            let _ = full::<i32>((4, 5), 1) + full::<i32>((3, 5), 2);
        });
    });
    assert!(msg.contains("(4,5) and (3,5)") && msg.contains("do not broadcast"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("no_broadcast_to", || {
            let _ = full::<i32>((4, 2), 1).broadcast_to((4, 3));
        });
    });
    assert!(msg.contains("(4,2) cannot be broadcast to (4,3)"), "{msg}");
}

#[test]
fn store_ssa_needs_equal_sizes() {
    let msg = build_panic(|| {
        Kernel::build("store_size", || {
            let y = param_tensor::<f32>("y", Layout::col_major(8));
            y.store_ssa(&full::<f32>((2, 3), 1.0));
        });
    });
    assert!(msg.contains("Tensor::store_ssa") && msg.contains("has 8 elements") && msg.contains("has 6"), "{msg}");
}

#[test]
fn narrowing_store_needs_to() {
    let msg = build_panic(|| {
        Kernel::build("store_narrow", || {
            let y = param_tensor::<F16>("y", Layout::col_major(8));
            y.store_ssa(&full::<f32>(8, 1.0));
        });
    });
    assert!(msg.contains("Float32 tensor cannot be stored to Float16") && msg.contains(".to::<Float16>()"), "{msg}");
    // A widening store of the same kind converts.
    Kernel::build("store_wide", || {
        let y = param_tensor::<f32>("y", Layout::col_major(8));
        y.store_ssa(&full::<F16>(8, F16::ONE));
    });
}

#[test]
fn tensor_value_metadata() {
    Kernel::build("ssa_metadata", || {
        let t = full::<F16>((4, (2, 3)), F16::ONE);
        assert_eq!(format!("{t:?}"), "tensor_value<vector<24xf16> o (4,(2,3))>");
        assert_eq!((t.size(), t.rank(), t.dtype()), (24, 2, DType::F16));
        let z: TensorSsa<i32> = zeros_like(&t);
        assert_eq!(z.values()[5].as_constant(), Some(0));
        let o: TensorSsa<f32> = ones_like(&param_tensor::<i8>("x", Layout::col_major((2, 2))));
        assert_eq!(o.shape(), &IntTuple::from((2, 2)));
        let e = empty_like::<u8>(&o);
        assert_eq!(e.size(), 4);
        let f = full_like::<f64>(&e, 2.5);
        assert_eq!(f.reshape((1, 4)).shape(), &IntTuple::from((1, 4)));
    });
}

/// A row maximum is `max.NaN.f32`, a float sum one sequential chain from
/// the initial value (each `addf` takes the previous one's result), and
/// `exp2` with fastmath `ex2.approx.ftz.f32`.
#[test]
fn reductions_are_sequential_and_propagate_nan() {
    let kernel = Kernel::builder("ssa_reduce_ptx").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::row_major((64, 8)));
        let out = param_tensor::<f32>("out", Layout::col_major((2, 2, 32)));
        let t = thread_idx().x;
        let tile = local_tile(&x, (2, 8), (t, 0)).load_ssa();
        let m = tile.reduce(ReductionOp::Max, f32::NEG_INFINITY, (.., 0));
        let s = (tile - m.clone()).exp2(true).reduce(ReductionOp::Add, 0.0, (.., 0));
        out.slice((.., 0, t)).store_ssa(&m);
        out.slice((.., 1, t)).store_ssa(&s);
    });
    let mlir = kernel.to_mlir();
    let adds: Vec<&str> = mlir.lines().filter(|l| l.contains("arith.addf")).collect();
    assert_eq!(adds.len(), 16, "{mlir}");
    // Two chains of eight: each add's first operand is the previous add's result.
    for chain in adds.chunks(8) {
        for pair in chain.windows(2) {
            let result = pair[0].trim().split(" = ").next().unwrap();
            let lhs = pair[1].split("arith.addf ").nth(1).unwrap().split(',').next().unwrap();
            assert_eq!(lhs, result, "{mlir}");
        }
    }
    let Some(tc) = toolchain() else { return };
    let p = ptx(&kernel, &tc);
    assert_eq!(count(&p, "max.NaN.f32"), 16, "{p}");
    assert_eq!(count(&p, "ex2.approx.ftz.f32"), 16, "{p}");
}

#[test]
fn reduction_profiles_must_be_weakly_congruent() {
    let msg = build_panic(|| {
        Kernel::build("bad_profile", || {
            let _ = full::<f32>((4, 8), 1.0).reduce(ReductionOp::Add, 0.0, (.., (0, 0)));
        });
    });
    assert!(msg.contains("profile (_,(0,0)) is not weakly congruent with the shape (4,8)"), "{msg}");
}

#[test]
fn boolean_tensors_are_not_ordered() {
    let msg = build_panic(|| {
        Kernel::build("bool_lt", || {
            let m = full::<bool>(4, true);
            let _ = m.lt(&m);
        });
    });
    assert!(msg.contains("TensorSsa::lt: Boolean tensors are compared only with eq and ne"), "{msg}");
}

#[test]
fn where_needs_a_tensor() {
    let msg = build_panic(|| {
        Kernel::build("where_scalars", || {
            let c = full::<bool>(4, true);
            let _: TensorSsa<f32> = where_(&c, 1.0, 2.0);
        });
    });
    assert!(msg.contains("where_: neither x nor y is a tensor"), "{msg}");
}

/// Eight Float32 to E4M3 are four `cvt.rn.satfinite.e4m3x2.f32`; the Int8
/// to BFloat16 fast path sign-extends with `dp4a` and rounds pairs, and the
/// Int4 one is `lop3`, `prmt` and `fma.rn.bf16x2`.
#[test]
fn conversions_use_the_wide_instructions() {
    let fp8 = Kernel::builder("ssa_to_fp8").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((8, 32)));
        let y = param_tensor::<F8E4M3FN>("y", Layout::col_major((8, 32)));
        let t = thread_idx().x;
        y.slice((.., t)).store_ssa(&x.slice((.., t)).load_ssa().to::<F8E4M3FN>());
    });
    let ints = Kernel::builder("ssa_int_to_bf16").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<i8>("x", Layout::col_major((8, 32)));
        let q = param_tensor::<u32>("q", Layout::col_major(32));
        let y = param_tensor::<BF16>("y", Layout::col_major((16, 32)));
        let t = thread_idx().x;
        local_tile(&y.slice((.., t)), 8, 0).store_ssa(&x.slice((.., t)).load_ssa().to::<BF16>());
        let nibbles = TensorSsa::new(1, vec![q.load(t)]).bitcast::<I4>();
        local_tile(&y.slice((.., t)), 8, 1).store_ssa(&nibbles.to::<BF16>());
    });
    let Some(tc) = toolchain() else { return };
    let p = fp8.compile_ptx(&tc, Arch::Sm89).unwrap();
    assert_eq!(count(&p, "cvt.rn.satfinite.e4m3x2.f32"), 4, "{p}");
    let p = ptx(&ints, &tc);
    assert_eq!(count(&p, "dp4a.s32.s32"), 8, "{p}");
    assert_eq!(count(&p, "cvt.rn.satfinite.bf16x2.f32"), 4, "{p}");
    assert!(count(&p, "lop3.b32") >= 1 && count(&p, "prmt.b32") >= 1, "{p}");
    assert_eq!(count(&p, "fma.rn.bf16x2"), 4, "{p}");
}

#[test]
fn bitcasts_must_divide_the_bits() {
    let msg = build_panic(|| {
        Kernel::build("bitcast_bits", || {
            let _ = full::<u8>(3, 1u8).bitcast::<u32>();
        });
    });
    assert!(msg.contains("3 Uint8 values are 24 bits, which Uint32 values of 32 bits cannot hold"), "{msg}");
}

#[test]
fn fp4_conversions_need_pairs() {
    let msg = build_panic(|| {
        Kernel::build("fp4_odd", || {
            let _ = full::<f32>(3, 1.0).to::<F4E2M1FN>();
        });
    });
    assert!(msg.contains("convert to Float4E2M1FN in pairs") && msg.contains("even, not 3"), "{msg}");
}

#[test]
fn ue5m3_waits_for_its_conversion() {
    let msg = build_panic(|| {
        Kernel::build("ue5m3", || {
            let _ = full::<f32>(4, 1.0).to::<F8E5M3FNU>();
        });
    });
    assert!(msg.contains("cvt_f32x4_to_fnv8e5m3x4"), "{msg}");
}

/// The simulator prints `print_tensor`'s text as `vprintf` would.
#[test]
fn print_tensor_text() {
    let case = feature_kernels::ssa_print();
    let x = [0.5f32, -1.25, 2.0, 3.0, 4.0, -0.0];
    let y = [1i32, -2, 3, 40, 5, -600];
    let mut args = vec![KernelArg::buffer(&x), KernelArg::buffer(&y)];
    let report = case.kernel.simulate(case.cfg, &mut args).unwrap();
    let text: String = report.printed.concat();
    let want = "\
tensor(raw_ptr<Float32, gmem, align(16)> o (2,3):(3,1), data=
       [[ 0.5000, -1.2500,  2.0000],
        [ 3.0000,  4.0000, -0.0000]])
tensor(raw_ptr<Int32, gmem, align(16)> o (2,3):(1,2), data=
       [[   1,    3,    5],
        [  -2,   40, -600]])
tensor(raw_ptr<Float16, smem, align(128)> o (2,2):(2,1), data=
       [[ 0.5000, -1.2500],
        [ 3.0000,  4.0000]])
tensor(rmem<Uint8> o 3:1, data=
       [   7,  200,    0])
tensor(rmem<Int64> o (2,2,2):(1,2,4), data=
       [[[-3000, -1000],
         [-2000,    0]],

        [[1000, 3000],
         [2000, 4000]]])
tensor(coord<(2,3)> o (2,2):(1@0,1@1), data=
       [[(2,3), (2,4)],
        [(3,3), (3,4)]])
tensor(rmem<Boolean> o (2,3):(1,2), data=
  (0,0): 0
  (1,0): 1
  (0,1): 0
  (1,1): 1
  (0,2): 1
  (1,2): 0
)
tensor(rmem<BFloat16> o 2:1, data=
       [ 1.5000, -0.2500])
tensor(rmem<Float64> o 2:1, data=
       [10000000000.0000, -2.0000])
tensor(rmem<Uint64> o 2:1, data=
       [18446744073709551615,    3])
tensor(rmem<Int8> o 2:1, data=
       [-128,  127])
tensor(rmem<Uint16> o 2:1, data=
       [65535,    0])
tensor(rmem<Int128> o 1:1, data=
       [0xffffffffffffffffffffffffffffffff])
tensor(rmem<Float8E4M3> o 2:1, data=
       [240.0000, -0.5000])
tensor(rmem<Int4> o 2:1, data=
       [  -3,    5])
";
    assert_eq!(text, want);
}

/// Booleans are bytes in memory but `i1` in MLIR, whose vectors LLVM
/// bit-packs: a multi-element copy of Booleans (an explicit 32-bit
/// universal atom, a G2R/R2G atom with `bits = 0`, `autovec_copy`) must
/// move one byte per element, never one packed byte for four.
#[test]
fn boolean_copies_move_a_byte_per_element() {
    let Some(tc) = toolchain() else { return };
    let kernel = Kernel::builder("bool_copies").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<bool>("x", Layout::col_major((4, 32)));
        let y = param_tensor::<bool>("y", Layout::col_major((4, 32)));
        let z = param_tensor::<bool>("z", Layout::col_major((4, 32)));
        let t = thread_idx().x;
        let (xs, ys, zs) = (x.slice((.., t)), y.slice((.., t)), z.slice((.., t)));
        let frag = make_fragment_like::<bool, bool>(&xs);
        copy(&make_copy_atom::<bool>(CopyOp::Universal { bits: 32 }), &xs, &frag);
        copy(&make_copy_atom::<bool>(CopyOp::Universal { bits: 32 }), &frag, &ys);
        let g2r = make_copy_atom::<bool>(CopyOp::G2R { bits: 0, attrs: G2RAttrs::default() });
        let frag2 = make_fragment_like::<bool, bool>(&xs);
        copy(&g2r, &xs, &frag2);
        let r2g = make_copy_atom::<bool>(CopyOp::R2G { bits: 0, attrs: R2GAttrs::default() });
        copy(&r2g, &frag2, &zs);
    });
    let mlir = kernel.to_mlir();
    assert!(!mlir.contains("vector<4xi1>"), "{mlir}");
    let p = ptx(&kernel, &tc);
    assert_eq!(count(&p, "ld.global.u8") + count(&p, "ld.global.b8") + count(&p, "ld.global.s8"), 8, "{p}");
    assert_eq!(count(&p, "st.global.u8") + count(&p, "st.global.b8") + count(&p, "st.global.s8"), 8, "{p}");
}

/// A reduction over several leaves folds them in CuTe DSL's order
/// (`vector.multi_reduction` of the row-major vector): lexicographic over
/// the reduced leaves, last fastest. Taking 1e8, then 1, then -1e8 (k
/// fastest in x(0, (j, k))) loses the 1 and sums to zero; a
/// colexicographic order (1e8, -1e8, 1) would keep it.
#[test]
fn multi_leaf_reductions_fold_lexicographically() {
    let kernel = Kernel::builder("reduce_order").block_dim([1, 1, 1]).build(|| {
        let out = param_tensor::<f32>("out", Layout::col_major(1));
        // Colexicographic values of (1, (2, 2)): (0,0), (1,0), (0,1), (1,1).
        let vals = [1e8f32, -1e8, 1.0, 0.0].map(Value::constant).to_vec();
        let x = TensorSsa::new((1, (2, 2)), vals);
        out.store_ssa(&x.reduce(ReductionOp::Add, 0.0, (.., 1)));
    });
    let mut out = [f32::NAN];
    let mut args = vec![KernelArg::buffer(&out)];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    out[0] = args[0].to_vec::<f32>()[0];
    assert_eq!(out[0], 0.0);
}

/// A rank-1 shape broadcasts with its one-mode tuple form: a reduction's
/// `(4,)` result meets a `4` tensor.
#[test]
fn a_leaf_meets_its_one_mode_tuple() {
    Kernel::build("leaf_tuple", || {
        let x = param_tensor::<f32>("x", Layout::col_major((4, 8)));
        let c = param_tensor::<f32>("c", Layout::col_major(4));
        let rows = x.load_ssa().reduce(ReductionOp::Add, 0.0, (.., 1));
        assert_eq!(rows.shape(), &IntTuple::from((4,)));
        let col = c.load_ssa();
        let s = rows.clone() + col.clone();
        assert_eq!(s.size(), 4);
        let s = col + rows.clone();
        assert_eq!(s.size(), 4);
        let _ = full::<f32>(4, 1.0).broadcast_to((4,));
        let _ = rows.broadcast_to(4);
        let _ = where_(&full::<bool>(4, true), &rows, 0.0);
    });
}
