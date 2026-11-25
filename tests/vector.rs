//! `Vector` (CuTe DSL's `cutlass.Vector`) and `cute::vector`: build panics
//! for misuse and `print_nd_vector`'s text. The kernels themselves are
//! feature kernels (`tests/feature_kernels/vector.rs`).

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// NumPy broadcasting aligns on the right: a (3,) vector does not meet a
/// (3, 4) one, where a (4,) one does.
#[test]
fn mismatched_broadcast_shapes_are_refused() {
    Kernel::build("vector_broadcast_ok", || {
        let a = vector::full::<f32>((3, 4), 1.0);
        let b = vector::full::<f32>(4, 2.0);
        assert_eq!((a + b).shape(), &[3, 4]);
    });
    let e = build_panic(|| {
        Kernel::build("vector_broadcast", || {
            let a = vector::full::<f32>((3, 4), 1.0);
            let b = vector::full::<f32>(3, 2.0);
            let _ = a + b;
        });
    });
    assert!(e.contains("Vector add: the vector shapes [3, 4] and [3] do not broadcast"), "{e}");
    let e = build_panic(|| {
        Kernel::build("vector_broadcast_to", || {
            let _ = vector::full::<f32>(3, 1.0).broadcast_to((3, 4));
        });
    });
    assert!(e.contains("cannot be broadcast to [3, 4]"), "{e}");
}

#[test]
fn to_vector_of_a_nested_shape_needs_force_flatten() {
    let e = build_panic(|| {
        Kernel::build("to_vector_nested", || {
            let t = full::<f32>(((2, 2), 3), 1.0);
            let _ = t.to_vector(false);
        });
    });
    assert!(e.contains("has the nested shape ((2,2),3); pass force_flatten = true"), "{e}");
}

#[test]
fn shuffles_move_32_bit_vectors_only() {
    let e = build_panic(|| {
        Kernel::build("shuffle_64", || {
            let v = vector::full::<f32>(2, 1.0);
            let _ = shuffle_sync(v, 0);
        });
    });
    assert!(e.contains("moves a vector of 32 bits as one word") && e.contains("2 Float32 elements are 64 bits"), "{e}");
    let e = build_panic(|| {
        Kernel::build("shuffle_tensor_64", || {
            let _ = shuffle_sync_bfly(full::<F16>(4, 1.0), 1);
        });
    });
    assert!(e.contains("moves a tensor value of 32 bits"), "{e}");
}

#[test]
fn indexing_and_slicing_are_checked() {
    let e = build_panic(|| {
        Kernel::build("vector_get", || {
            let _ = vector::full::<f32>((2, 3), 1.0).get(1);
        });
    });
    assert!(e.contains("has 1 indices, but vector<2x3xf32> has 2 dimensions"), "{e}");
    let e = build_panic(|| {
        Kernel::build("vector_slice", || {
            let _ = vector::full::<f32>((2, 3), 1.0).slice((.., 2..4));
        });
    });
    assert!(e.contains("the range 2..4 is out of range"), "{e}");
    let e = build_panic(|| {
        Kernel::build("vector_bitcast", || {
            let _ = vector::full::<F16>((2, 3), 1.0).bitcast::<u32>();
        });
    });
    assert!(e.contains("innermost dimension of vector<2x3xf16> is 48 bits"), "{e}");
}

#[test]
fn print_nd_vector_text() {
    let kernel = Kernel::builder("print_nd_vector").block_dim([1, 1, 1]).build(|| {
        let v = Vector::<f32>::new((2, 3), (0..6).map(|i| Float32::constant(i as f32 * 0.5)).collect());
        vector::print_nd_vector(&v);
        vector::print_nd_vector(&vector::create_mask(4, 2));
    });
    let report = kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut []).unwrap();
    assert_eq!(
        report.printed.concat(),
        "vector<2x3xf32> = [[ 0.0000,  0.5000,  1.0000], [ 1.5000,  2.0000,  2.5000]]\nvector<4xi1> = [1, 1, 0, 0]\n"
    );
}
