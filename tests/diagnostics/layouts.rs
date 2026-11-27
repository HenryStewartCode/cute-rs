//! Layout, tensor and shape-arithmetic misuse, dynamic extents, assumptions and basis strides.

use super::*;
/// `increment_coord` needs a value in every leaf and a congruent shape.
#[test]
fn increment_coord_misuse() {
    let msg = build_panic(|| {
        Kernel::build("increment_underscore", || {
            let t = thread_idx().x;
            let _ = increment_coord((t, ..), (4, 8));
        });
    });
    assert!(msg.contains("increment_coord: coordinate (") && msg.contains(",_) has a `..`"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("increment_incongruent", || {
            let t = thread_idx().x;
            let _ = increment_coord((t, 1), (4, (2, 4)));
        });
    });
    assert!(msg.contains("is not congruent with shape (4,(2,4))"), "{msg}");
    // A leaf is not congruent with a 1-tuple, nor a pair with a leaf.
    let msg = build_panic(|| {
        let _ = increment_coord(1, (4,));
    });
    assert!(msg.contains("coordinate 1 is not congruent with shape (4)"), "{msg}");
    let msg = build_panic(|| {
        let _ = increment_coord((1, 2), 8);
    });
    assert!(msg.contains("coordinate (1,2) is not congruent with shape 8"), "{msg}");
}

/// `tile_to_shape` with an order takes a permutation of the target's modes,
/// and the target needs at least the block's rank.
#[test]
fn tile_to_shape_misuse() {
    let block = Layout::row_major((8, 64));
    for order in [&[0, 0][..], &[1, 2], &[0], &[0, 1, 2]] {
        let msg = build_panic(|| {
            tile_to_shape_ordered(&block, (64, 128), order);
        });
        assert!(msg.contains(&format!("order {order:?} is not a permutation of 0..2")), "{order:?}: {msg}");
    }
    let atom = make_composed_layout(make_swizzle(3, 3, 3), 0, block.clone());
    let msg = build_panic(|| {
        tile_to_shape_swizzled_ordered(&atom, (64, 128, 2), &[2, 0, 2]);
    });
    assert!(msg.contains("order [2, 0, 2] is not a permutation of 0..3"), "{msg}");
    let msg = build_panic(|| {
        tile_to_shape(&block, (64,));
    });
    assert!(msg.contains("block (8,64):(64,1) has rank 2, more than the rank-1 target shape (64)"), "{msg}");
}

/// `is_major` with a mode the stride does not have names itself, the mode and
/// the layout or stride, rather than failing inside the tuple code.
#[test]
fn is_major_mode_out_of_range() {
    let msg = build_panic(|| {
        Layout::row_major((4, 8)).is_major(5);
    });
    assert!(msg.contains("is_major: mode 5 is out of range for the rank-2 layout (4,8):(8,1)"), "{msg}");
    let msg = build_panic(|| {
        Layout::col_major(8).is_major(1);
    });
    assert!(msg.contains("is_major: mode 1 is out of range for the rank-1 layout 8:1"), "{msg}");
    let msg = build_panic(|| {
        is_major(2, &IntTuple::from((8, (1, 2))));
    });
    assert!(msg.contains("is_major: mode 2 is out of range for the rank-2 stride (8,(1,2))"), "{msg}");
    // A runtime stride shows as `?`, in a bare stride and in a layout.
    let msg = build_panic(|| {
        Kernel::build("is_major_runtime_stride", || {
            let layout = Layout::with_strides((4, 8), (thread_idx().x.as_index(), 1));
            is_major(3, layout.stride());
        });
    });
    assert!(msg.contains("is_major: mode 3 is out of range for the rank-2 stride (?,1)"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("is_major_runtime_layout_stride", || {
            Layout::with_strides((4, 8), (thread_idx().x.as_index(), 1)).is_major(5);
        });
    });
    assert!(msg.contains("is_major: mode 5 is out of range for the rank-2 layout (4,8):(?,1)"), "{msg}");
    // A mode that is an empty tuple, or starts with one, has no first stride.
    let msg = build_panic(|| {
        "((),8):((),1)".parse::<Layout>().unwrap().is_major(0);
    });
    assert!(msg.contains("is_major: mode 0 of the layout ((),8):((),1) is the empty tuple ()"), "{msg}");
    let msg = build_panic(|| {
        "(((),2),1):(((),1),2)".parse::<Layout>().unwrap().is_major(0);
    });
    assert!(
        msg.contains(
            "is_major: mode 0 of the layout (((),2),1):(((),1),2) has no first stride: it starts with an empty tuple"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        is_major(1, &"(1,((),2))".parse::<IntTuple>().unwrap());
    });
    assert!(msg.contains("is_major: mode 1 of the stride (1,((),2)) has no first stride: it starts with"), "{msg}");
}

/// `front` of a tuple that has no first leaf says where the empty tuple is.
#[test]
fn front_of_an_empty_tuple() {
    let msg = build_panic(|| {
        IntTuple::tuple([]).front();
    });
    assert!(msg.contains("front: the empty tuple () has no first leaf"), "{msg}");
    let msg = build_panic(|| {
        "(((),2),1)".parse::<IntTuple>().unwrap().front();
    });
    assert!(msg.contains("front: the tuple has no first leaf: its mode [0, 0] is the empty tuple ()"), "{msg}");
}

/// `leading_dim` needs a shape and a stride with the same nesting.
#[test]
fn leading_dim_incongruent() {
    let msg = build_panic(|| {
        leading_dim(&IntTuple::from((4, 8)), &IntTuple::from((1,)));
    });
    assert!(msg.contains("leading_dim: shape (4,8) and stride (1) are not congruent"), "{msg}");
    let msg = build_panic(|| {
        leading_dim(&IntTuple::from((4, (2, 2))), &IntTuple::from((1, 4)));
    });
    assert!(msg.contains("leading_dim: shape (4,(2,2)) and stride (1,4) are not congruent"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("leading_dim_incongruent", || {
            let stride = Coord::from((1usize, thread_idx().x.as_index(), 4usize));
            leading_dim(&IntTuple::from((4, 8)), &stride);
        });
    });
    assert!(msg.contains("leading_dim: shape (4,8) and stride (1,?,4) are not congruent"), "{msg}");
}

/// The `usize` sizes that Rust loops unroll over refuse a layout whose size
/// is only known at run time, naming the dynamic leaf and the `IndexExpr`
/// query to use instead.
#[test]
fn static_sizes_of_dynamic_layouts() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_size", || {
            let m = thread_idx().x;
            let _ = Layout::col_major((m, 8)).size();
        });
    });
    assert!(
        msg.contains("Layout::size: layout (?,8):(1,?) has the dynamic extent %")
            && msg.contains("at mode [0], so its size is only known at run time; use cute::size(&layout, &[])"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_cosize", || {
            let k = thread_idx().x;
            let _ = Layout::with_strides((4, 8), (k, 1)).cosize();
        });
    });
    assert!(
        msg.contains("Layout::cosize: layout (4,8):(?,1) has the dynamic stride %")
            && msg.contains("so its cosize is only known at run time; use cute::cosize(&layout, &[])"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_shape_size", || {
            let m = thread_idx().x;
            let _ = Shape::from((4, (m, 2))).size();
        });
    });
    assert!(
        msg.contains("size: shape (4,(?,2)) has the dynamic extent %")
            && msg.contains("at mode [1, 0], so its size is only known at run time; use cute::size(&shape, &[])"),
        "{msg}"
    );
    // A runtime stride on a size-1 mode does not count.
    Kernel::build("static_cosize", || {
        let k = thread_idx().x;
        assert_eq!(Layout::with_strides((1, 8), (k, 1)).cosize(), 8);
    });
}

/// Thread-value layouts, and so the tiled copies built from them, are static.
#[test]
fn thread_value_layouts_must_be_static() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_thread_layout", || {
            let m = thread_idx().x;
            let _ = make_layout_tv(&Layout::col_major((m, 4)), &Layout::col_major(4));
        });
    });
    assert!(
        msg.contains("make_layout_tv: the thread layout: layout (?,4):(1,?) has the dynamic extent %")
            && msg.contains("at mode [0]; thread-value layouts are static"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_value_layout", || {
            let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
            let k = thread_idx().x;
            let _ = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::with_strides((2, 4), (1, k)));
        });
    });
    assert!(
        msg.contains("make_layout_tv: the value layout: layout (2,4):(1,?) has the dynamic stride %")
            && msg.contains("at mode [1]; thread-value layouts are static"),
        "{msg}"
    );
}

/// `upcast` counts offsets in larger units, which needs static strides.
#[test]
fn upcast_needs_a_static_layout() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_upcast", || {
            let k = thread_idx().x;
            let _ = Layout::with_strides((4, 8), (k, 1)).upcast(2);
        });
    });
    assert!(
        msg.contains("upcast: layout (4,8):(?,1) has the dynamic stride %")
            && msg.contains("at mode [0]; upcast needs a static layout"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_upcast_extent", || {
            let m = thread_idx().x;
            let _ = Layout::col_major((8, m)).upcast(4);
        });
    });
    assert!(msg.contains("upcast: layout (8,?):(1,8) has the dynamic extent %"), "{msg}");
}

/// What the algebra cannot do with runtime leaves is refused with a message
/// naming the operation and the leaf, as the CuTe dialect refuses it.
#[test]
fn layout_algebra_limits_with_runtime_leaves() {
    // A product replicates a static block; only the tiler may be dynamic.
    let msg = build_panic(|| {
        Kernel::build("dynamic_block", || {
            let m = thread_idx().x;
            let _ = logical_product(&Layout::col_major((m, 8)), Layout::col_major((2, 4)));
        });
    });
    assert!(
        msg.contains("logical_product: layout (?,8):(1,?) has the dynamic extent %")
            && msg.contains("a product replicates a static layout (only the tiler may be dynamic)"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_raked_block", || {
            let k = thread_idx().x;
            let _ = raked_product(&Layout::with_strides(4, k), &Layout::col_major(2));
        });
    });
    assert!(msg.contains("raked_product: layout 4:? has the dynamic stride %"), "{msg}");
    // A tiler with a runtime stride must have a single mode.
    let msg = build_panic(|| {
        Kernel::build("dynamic_stride_tiler", || {
            let (m, k) = (thread_idx().x, block_idx().x);
            let _ = logical_divide(&Layout::row_major((64, 64)), Layout::new((m, 4), (1, k)));
        });
    });
    assert!(
        msg.contains("logical_divide: the tiler (?,4):(1,?) of layout (64,64):(64,1) has the runtime stride %")
            && msg.contains("and 2 modes after filtering; CuTe divides by a tiler with a runtime stride only when it"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_stride_complement", || {
            let k = thread_idx().x;
            let _ = complement(&Layout::with_strides((4, 8), (1, k)), 64);
        });
    });
    assert!(
        msg.contains("complement: layout (4,8):(1,?) has the runtime stride %")
            && msg.contains("only when it has a single mode, and this one has 2 after filtering"),
        "{msg}"
    );
    // Two runtime extents that cannot be ordered when the kernel is built are
    // compared at run time, but a static stride that does not divide the
    // static extent it lands in is refused, as by the static algebra, naming
    // the mode of a by-mode tiler.
    Kernel::build("unordered_extents", || {
        let (m, k, s) = (thread_idx().x, block_idx().x, block_idx().y);
        let c = composition(&Layout::new((m, 4), (1, s)), &Layout::new(k, 1));
        assert!(!c.shape().is_static());
    });
    let msg = build_panic(|| {
        Kernel::build("incompatible_mode", || {
            let k = thread_idx().x;
            let _ = logical_divide(&Layout::with_strides((8, (6, 8)), (k, (8, 1))), (.., Layout::new(4, 1)));
        });
    });
    assert!(
        msg.contains(
            "composition (6,8):(8,1) ∘ (4,16):(1,4) (mode [1] of layout (8,(6,8)):(?,(8,1))) is not defined: "
        ) && msg.contains("stride 4 is not compatible with mode size 6"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_left_inverse", || {
            let k = thread_idx().x;
            let _ = left_inverse(&Layout::with_strides((4, 8), (k, 1)));
        });
    });
    assert!(
        msg.contains("left_inverse: layout (4,8):(?,1) has the runtime stride %")
            && msg.contains("at mode [0]; a left inverse needs static strides"),
        "{msg}"
    );
}

/// Buffers sized when the kernel is built (fragments, shared memory,
/// per-thread arrays) are sized from their layouts, so those must be static.
/// (A tensor parameter with runtime extents is a buffer sized at launch.)
#[test]
fn buffers_need_static_layouts() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_stack_array", || {
            let m = thread_idx().x;
            let _ = make_rmem_tensor::<f32>(Layout::col_major((m, 8)));
        });
    });
    assert!(
        msg.contains("make_rmem_tensor: layout (?,8):(1,?) has the dynamic extent %")
            && msg.contains("a per-thread array is sized from its layout"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_fragment", || {
            let m = thread_idx().x;
            let _ = make_fragment::<f32>((4, m));
        });
    });
    assert!(
        msg.contains("make_fragment: layout (4,?):(1,4) has the dynamic extent %")
            && msg.contains("register fragments have static shapes"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_smem", || {
            let k = thread_idx().x;
            let _ = make_smem_tensor::<f32>(Layout::with_strides((4, 8), (k, 1)));
        });
    });
    assert!(
        msg.contains("make_smem_tensor: layout (4,8):(?,1) has the dynamic stride %")
            && msg.contains("a shared-memory allocation is sized from its layout"),
        "{msg}"
    );
}

/// `size`, `shape` and `cosize` take mode paths that stay inside the tuple
/// and stop at a leaf, as in the CuTe dialect.
#[test]
fn mode_paths_of_size_queries() {
    let l = Layout::new((2, (3, 4)), (12, (1, 3)));
    let msg = build_panic(|| {
        size(&l, &[2]);
    });
    assert!(msg.contains("size: mode path [2] does not fit: (2,(3,4)) has rank 2, so it has no mode 2"), "{msg}");
    let msg = build_panic(|| {
        cosize(&l, &[0, 0]);
    });
    assert!(
        msg.contains(
            "cosize: mode path [0, 0] does not fit: mode [0] of (2,(3,4)) is a leaf, so the path cannot go on"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        shape(&l, &[1, 5]);
    });
    assert!(
        msg.contains("shape: mode path [1, 5] does not fit: mode [1] of (2,(3,4)) has rank 2, so it has no mode 5"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_mode_path", || {
            let m = thread_idx().x;
            size(&Layout::col_major((m, 8)), &[0, 1]);
        });
    });
    assert!(msg.contains("size: mode path [0, 1] does not fit: mode [0] of (?,8) is a leaf"), "{msg}");
}

/// A `?` in a parsed layout string is a placeholder for exercising the
/// algebra; an offset that depends on it cannot be computed in a kernel.
#[test]
fn parsed_placeholders_cannot_be_evaluated() {
    let msg = build_panic(|| {
        Kernel::build("placeholder", || {
            let out = param_tensor::<i32>("out", Layout::col_major(4));
            let l: Layout = "(?,4):(1,?)".parse().unwrap();
            out.store(0usize, Int32::from(l.offset_at(5)));
        });
    });
    assert!(
        msg.contains("which does not belong to kernel `placeholder`")
            && msg.contains("a `?` in a parsed layout string is a placeholder and cannot be evaluated in a kernel"),
        "{msg}"
    );
}

thread_local! {
    /// Where the last panic on this thread happened, as `file:line`.
    static PANIC_AT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// [`build_panic`], also returning where the panic was raised. A panic hook,
/// installed once for the whole test binary, records the location of each
/// panic on the panicking thread and then prints it as usual.
fn panic_location(f: impl FnOnce()) -> (String, String) {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let at = info.location().map(|l| format!("{}:{}", l.file(), l.line()));
            PANIC_AT.with(|p| *p.borrow_mut() = at);
            default(info);
        }));
    });
    let msg = build_panic(f);
    let at = PANIC_AT.with(|p| p.borrow_mut().take()).expect("the panic hook recorded a location");
    (msg, at)
}

/// What a panic's message names, and code that panics so.
type PanicCase = (&'static str, Box<dyn FnOnce()>);

/// The layout algebra's misuse messages point at the caller's line, not
/// into the library, including those raised in the middle of a composition
/// or of a by-mode divide, and name the function: a coordinate that does
/// not fit a layout's shape names the mode where it does not.
#[test]
fn layout_panics_report_the_callers_line() {
    let cases: Vec<PanicCase> = vec![
        (
            "Layout::size",
            Box::new(|| {
                Kernel::build("size_location", || {
                    let _ = Layout::col_major((thread_idx().x, 8)).size();
                });
            }),
        ),
        (
            "Layout::cosize",
            Box::new(|| {
                Kernel::build("cosize_location", || {
                    let _ = Layout::with_strides((4, 8), (thread_idx().x, 1)).cosize();
                });
            }),
        ),
        (
            "size: shape",
            Box::new(|| {
                Kernel::build("shape_size_location", || {
                    let _ = Shape::from((4, thread_idx().x)).size();
                });
            }),
        ),
        (
            "composition (6,8):(8,1) ∘ 4:2 is not defined",
            Box::new(|| {
                let _ = composition(&Layout::row_major((6, 8)), &Layout::new(4, 2));
            }),
        ),
        (
            "(mode [1] of layout (8,(6,8)):(1,(8,1))) is not defined",
            Box::new(|| {
                let _ = logical_divide(&Layout::new((8, (6, 8)), (1, (8, 1))), (.., 4));
            }),
        ),
        (
            "crd2idx: coordinate (1,2,3) has rank 3 but layout (4,8):(1,4) has rank 2",
            Box::new(|| {
                let _ = crd2idx((1usize, 2usize, 3usize), &make_layout((4, 8), (1, 4)));
            }),
        ),
        (
            "crd2idx: coordinate ((1,2),3) does not fit layout (4,8):(1,4): its mode [0], (1,2), has rank 2 where \
             the shape has the extent 4",
            Box::new(|| {
                let _ = crd2idx(((1usize, 2usize), 3usize), &make_layout((4, 8), (1, 4)));
            }),
        ),
        (
            "Layout::offset: coordinate (1,(2,3,4)) does not fit layout (4,(2,4)):(1,(4,8)): its mode [1], (2,3,4), \
             has rank 3 where the shape has (2,4), of rank 2",
            Box::new(|| {
                let _ = Layout::col_major((4, (2, 4))).offset(&Coord::from((1usize, (2usize, 3usize, 4usize))));
            }),
        ),
        (
            "Layout::crd2idx: coordinate (1,2,3) has rank 3",
            Box::new(|| {
                let _ = Layout::col_major((4, 8)).crd2idx(&IntTuple::from((1, 2, 3)));
            }),
        ),
        (
            "Layout::slice: coordinate (1,_,2) has rank 3 but layout (4,8):(1,4) has rank 2",
            Box::new(|| {
                let _ = Layout::col_major((4, 8)).slice(&(1usize, .., 2usize).into());
            }),
        ),
        (
            "Tensor::load: coordinate (1,?) has rank 2 but layout 32:1 has rank 1",
            Box::new(|| {
                Kernel::build("load_location", || {
                    let x = param_tensor::<f32>("x", Layout::col_major(32));
                    let _ = x.load((1usize, thread_idx().x));
                });
            }),
        ),
        (
            "logical_divide: the tiler (?,4):(1,?) of mode [1] (64:1) of layout (64,64):(64,1) has the runtime stride",
            Box::new(|| {
                Kernel::build("divide_location", || {
                    let (m, k) = (thread_idx().x, block_idx().x);
                    let _ = zipped_divide(&Layout::row_major((64, 64)), (.., Layout::new((m, 4), (1, k))));
                });
            }),
        ),
        (
            "Tensor::size: tensor (?,8):(1,?)",
            Box::new(|| {
                Kernel::build("tensor_location", || {
                    let _ = make_tensor(&param_ptr::<f32>("x"), Layout::col_major((thread_idx().x, 8))).size();
                });
            }),
        ),
        (
            "copy: the source tensor",
            Box::new(|| {
                Kernel::build("copy_location", || {
                    let x = make_tensor(&param_ptr::<f32>("x"), Layout::row_major((thread_idx().x, 8)));
                    let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
                    copy(&atom, &x, &make_fragment::<f32>((4, 8)));
                });
            }),
        ),
        (
            "partition_fragment_c: tensor",
            Box::new(|| {
                Kernel::build("fragment_location", || {
                    let c = make_tensor(&param_ptr::<f32>("c"), Layout::col_major((thread_idx().x, 16)));
                    let mma = make_tiled_mma(MmaOp::UniversalFma(DType::F32), Layout::col_major((4, 8, 1)));
                    let _ = mma.get_slice(0usize).partition_fragment_c::<f32, _>(&c);
                });
            }),
        ),
        (
            "elem_less: (?,1) and (3,4,5) are not congruent",
            Box::new(|| {
                Kernel::build("elem_less_location", || {
                    let _ = elem_less((thread_idx().x, 1), (3, 4, 5));
                });
            }),
        ),
        (
            "ceil_div: cannot divide (4,8) by (2,4,2)",
            Box::new(|| {
                let _ = ceil_div((4, 8), (2, 4, 2));
            }),
        ),
    ];
    for (what, f) in cases {
        let (msg, at) = panic_location(f);
        assert!(msg.contains(what), "expected {what:?} in: {msg}");
        assert!(at.starts_with("tests/diagnostics/layouts.rs:"), "the panic for {what:?} is reported at {at}: {msg}");
    }
}

/// Tensors take runtime extents: `make_tensor` with a layout of runtime
/// extents, and dividing or composing a tensor by a tiler with runtime
/// extents, or with runtime strides that are not provably multiples of the
/// tensor's extents (which gives it runtime extents from composition,
/// whether a divide or a composition gives it), build. A runtime tiler of a
/// tiled copy is still refused by make_tiled_copy itself, and the static
/// upper bound idiom of examples/runtime_gemm.rs still works.
#[test]
fn tensors_take_runtime_extents() {
    Kernel::build("dynamic_tensor", || {
        let x = param_ptr::<f32>("x");
        let m = thread_idx().x;
        let t = make_tensor(&x, Layout::col_major((m, 8)));
        assert_eq!(t.shape().to_string(), "(?,8)");
        assert!(!size(&t, &[]).is_static());
        assert_eq!(size(&t, &[1]), IndexExpr::Static(8));
    });
    Kernel::build("dynamic_tile", || {
        let x = param_tensor::<f32>("x", Layout::col_major((64, 64)));
        let m = thread_idx().x;
        let tile = local_tile(&x, (Layout::new(m, 1), Layout::new(8, 1)), (0, 0));
        assert_eq!(tile.layout().to_string(), "(?,8):(1,64)");
    });
    for compose in [false, true] {
        Kernel::build("runtime_stride_tile", || {
            let x = param_tensor::<f32>("x", Layout::row_major((6, 8)));
            let s = param_scalar::<i32>("s");
            let t = if compose { x.compose(Layout::new(2, s * 8)) } else { logical_divide(&x, Layout::new(2, s)) };
            assert!(!t.shape().is_static(), "{t:?}");
        });
    }
    // A runtime stride that is provably a multiple of the rows steps over
    // them when the kernel is built, so the extent stays static.
    Kernel::build("multiple_stride_tile", || {
        let x = param_tensor::<f32>("x", Layout::row_major((6, 8)));
        let k = param_extent("k", 6);
        assert_eq!(x.compose(Layout::new(2, k)).shape(), &Shape::from(2));
    });
    // A runtime tiler of a tiled copy is refused by make_tiled_copy itself.
    let msg = build_panic(|| {
        Kernel::build("dynamic_copy_tiler", || {
            let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
            let m = thread_idx().x;
            let _ = make_tiled_copy(&atom, &Layout::col_major((32, 4)), (Layout::new(m, 1), Layout::new(4, 1)));
        });
    });
    assert!(
        msg.contains("make_tiled_copy: the tiler: layout ?:1 has the dynamic extent %")
            && msg.contains("a tiled copy covers a static tile"),
        "{msg}"
    );
    // Static extents with runtime strides, as in examples/runtime_gemm.rs.
    Kernel::build("runtime_strides", || {
        let x = param_ptr::<f32>("x");
        let k = thread_idx().x;
        let t = make_tensor(&x, Layout::with_strides((64, 64), (k, 1)));
        let tile = local_tile(&t, (8, 8), (1, 2));
        assert_eq!(tile.size(), 64);
    });
}

/// A refused composition names the right-hand side's stride or extent as
/// written, with what is left of it after the modes before when that is
/// different, and the elements a mode holds at the stride that split it.
#[test]
fn composition_refusals_name_the_right_hand_values() {
    let a = Layout::new((2, 3, 5), (1, 100, 1000));
    // A stride smaller than the mode it lands in that does not divide it is
    // C++ CuTe's weak divisibility condition; the answer is kept where it is
    // the composition, as here (offsets 0 and 4 of `a` are 0 and 200).
    assert_eq!(composition(&a, &Layout::new(2, 4)), Layout::new(2, 200));
    let msg = build_panic(|| {
        let _ = composition(&a, &Layout::new(4, 4));
    });
    assert!(
        msg.contains(
            "composition (2,3,5):(1,100,1000) ∘ 4:4 is not defined: stride 4 (2 after the modes before it) is not \
             compatible with mode size 3"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        let _ = composition(&a, &Layout::new(8, 1));
    });
    assert!(msg.contains("shape 8 (4 after the modes before it) is not compatible with mode size 3"), "{msg}");
    let msg = build_panic(|| {
        let _ = composition(&Layout::row_major((6, 8)), &Layout::new(4, 2));
    });
    assert!(
        msg.contains(
            "composition (6,8):(8,1) ∘ 4:2 is not defined: shape 4 is not compatible with mode size 6 at stride 2, \
             which has 3 elements"
        ),
        "{msg}"
    );
    // Inside a divide, the tiler's complement is the right-hand side.
    let msg = build_panic(|| {
        let _ = logical_divide(&Layout::row_major((6, 8)), Layout::new(2, 4));
    });
    assert!(
        msg.contains("composition (6,8):(8,1) ∘ (2,(4,4)):(4,(1,8)) is not defined: stride 8 is not compatible"),
        "{msg}"
    );
}

/// A runtime stride on the right-hand side of a composition, or in the
/// tiler of a divide, must not be 0 at launch, as in C++ CuTe: the extents
/// composition computes at run time divide by it, and the simulator reports
/// the division by zero. A static stride of 0 is fine.
#[test]
fn runtime_tiler_strides_must_not_be_zero() {
    let a = Layout::row_major((6, 8));
    assert_eq!(composition(&a, &Layout::new(2, 0)), Layout::new(2, 0));
    // Each thread stores the offset of index t % n, a runtime index into the
    // runtime extents, of (6,8):(8,1) composed with or divided by 2:s; at s =
    // 3 that is the static result with 2:3.
    type Op = fn(&Layout, &Layout) -> Layout;
    let ops: [(&str, Op, i32, &str); 2] = [
        ("compose_runtime_stride", |a, b| composition(a, b), 2, "2:24"),
        ("divide_runtime_stride", |a, b| logical_divide(a, b), 48, "(2,(3,8)):(24,(8,1))"),
    ];
    for (name, op, n, static_result) in ops {
        let kernel = Kernel::builder(name).block_dim([64, 1, 1]).build(|| {
            let s = param_scalar::<i32>("s");
            let out = param_tensor::<i32>("out", Layout::col_major(64));
            let t = thread_idx().x;
            out.store(t, Int32::from(crd2idx(t % n, &op(&Layout::row_major((6, 8)), &Layout::new(2, s)))));
        });
        let run = |s: i32| {
            let mut args = vec![KernelArg::ScalarI32(s), KernelArg::i32(vec![0; 64])];
            kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).map(|_| args[1].to_vec::<i32>())
        };
        let want = op(&a, &Layout::new(2, 3));
        assert_eq!(want.to_string(), static_result);
        let want: Vec<i32> = (0..64).map(|t| want.call(t % n as usize) as i32).collect();
        assert_eq!(run(3).unwrap(), want, "{name}");
        let e = run(0).unwrap_err();
        assert!(e.message.contains("division by zero"), "{name}: {e}");
    }
}

/// A tensor of runtime extents `(m, 8)`, in global memory.
fn dynamic_tensor(name: &str) -> Tensor<f32> {
    let m = param_extent(&format!("{name}_m"), 1);
    make_tensor(&param_ptr::<f32>(name), Layout::col_major((m, 8)))
}

/// The `usize` `Tensor::size` and the loops over every element refuse a
/// tensor whose size is only known at run time, naming the dynamic leaf and
/// the `IndexExpr` query or the for_range to use instead.
#[test]
fn static_sizes_of_dynamic_tensors() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_tensor_size", || {
            let _ = dynamic_tensor("x").size();
        });
    });
    assert!(
        msg.contains("Tensor::size: tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("at mode [0], so its size is only known at run time; use cute::size(&tensor, &[])"),
        "{msg}"
    );
    type Visit = fn(&Tensor<f32>);
    let visits: [(&str, Visit); 4] = [
        ("Tensor::fill", |t| t.fill(1.0)),
        ("Tensor::values", |t| drop(t.values())),
        ("Tensor::map_in_place", |t| t.map_in_place(|v| v + 1.0)),
        ("Tensor::store_values", |t| t.store_values(&[])),
    ];
    for (what, visit) in visits {
        let msg = build_panic(|| {
            Kernel::build("dynamic_tensor_visit", || visit(&dynamic_tensor("x")));
        });
        assert!(
            msg.contains(&format!("{what}: tensor (?,8):(1,?) has the dynamic extent %"))
                && msg.contains(&format!("but {what} visits every element when the kernel is built"))
                && msg.contains("write a for_range over the rest mode"),
            "{what}: {msg}"
        );
    }
}

/// Register fragments have static shapes: a fragment shaped like a tensor
/// of runtime extents, an MMA fragment of a whole tensor of runtime extents
/// rather than of a tile, and a fragment divided by a tiler with runtime
/// extents are refused, naming the dynamic extent.
#[test]
fn fragments_of_dynamic_shapes() {
    let msg = build_panic(|| {
        Kernel::build("fragment_like_dynamic", || {
            let _ = make_fragment_like::<f32, _>(&dynamic_tensor("x"));
        });
    });
    assert!(
        msg.contains("make_fragment_like: tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("register fragments have static shapes")
            && msg.contains("make the fragment from one tile")
            && msg.contains("for_range over the rest mode"),
        "{msg}"
    );
    // A conversion makes a fragment too, and says so under its own name.
    let msg = build_panic(|| {
        Kernel::build("converted_dynamic", || {
            let _ = dynamic_tensor("x").to::<F16>();
        });
    });
    assert!(
        msg.starts_with("Tensor::to: tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("register fragments have static shapes"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("mma_fragment_dynamic", || {
            let m = param_extent("m", 1);
            let c = make_tensor(&param_ptr::<f32>("c"), Layout::col_major((m, 16)));
            let mma = make_tiled_mma(MmaOp::UniversalFma(DType::F32), Layout::col_major((4, 8, 1)));
            let _ = mma.get_slice(thread_idx().x).partition_fragment_c::<f32, _>(&c);
        });
    });
    assert!(msg.contains("partition_fragment_c: tensor (1,?,2):(0,4,?{div=8}) has the dynamic extent ((%"), "{msg}");
    assert!(msg.contains("at mode [1]") && msg.contains("register fragments have static shapes"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("fragment_dynamic_tiler", || {
            let frag = make_fragment::<f32>((8, 4));
            let _ = local_tile(&frag, (Layout::new(param_extent("m", 1), 1), 4), (0, 0));
        });
    });
    assert!(
        msg.contains(
            "cute: a register fragment cannot take the layout ((?,4),(?,1)):((1,8),(?,0)), which has the dynamic"
        ) && msg.contains("register fragments have static shapes (a fragment is an array of registers)"),
        "{msg}"
    );
}

/// Copies and gemm unroll the per-thread chunks and the rest modes when the
/// kernel is built: a copy whose per-thread chunk is dynamic, or whose rest
/// mode is, an element-wise copy of runtime extents, gemm on an operand of
/// runtime extents and retile of one are refused, naming the leaf and what
/// must be static.
#[test]
fn copies_unroll_static_extents() {
    let atom = || make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
    // The per-thread chunk (mode 0) of the source has a runtime extent.
    let msg = build_panic(|| {
        Kernel::build("dynamic_chunk", || {
            let m = param_extent("m", 4);
            let src = make_tensor(&param_ptr::<f32>("x"), Layout::col_major((m, 2)));
            copy(&atom(), &src, &make_fragment::<f32>((8, 2)));
        });
    });
    assert!(
        msg.contains("copy: the source tensor (?{div=4},2):(1,?{div=4}) has the dynamic extent %")
            && msg.contains("at mode [0], in its mode 0, the chunk of values each thread copies")
            && msg.contains("splits mode 0 into the Universal { bits: 128 } atom's chunks when the kernel is built, so it must be static"),
        "{msg}"
    );
    // A rest mode of the partition of a whole tensor of runtime extents: one
    // tile at a time is copied, its rest mode sliced away by a for_range.
    let msg = build_panic(|| {
        Kernel::build("dynamic_rest", || {
            let m = param_extent("m", 1);
            let x = make_tensor(&param_ptr::<f32>("x"), Layout::row_major((m, 16)));
            let tiled = make_tiled_copy_tv(&atom(), &Layout::row_major((8, 4)), &Layout::row_major((1, 4)));
            let thr = tiled.get_slice(thread_idx().x);
            let src = thr.partition_s(&x);
            let dst = make_fragment::<f32>((4, 4, 1));
            copy(&tiled, &src, &dst);
        });
    });
    assert!(
        msg.contains("copy: the source tensor ((4,1),?,1):((1,0),128,0) has the dynamic extent ((%")
            && msg.contains("at mode [1], a rest mode that copy unrolls when the kernel is built")
            && msg.contains("write a for_range over that mode and copy one slice of it per iteration"),
        "{msg}"
    );
    // The same copy of one tile per iteration of a for_range builds.
    Kernel::build("dynamic_rest_by_tile", || {
        let m = param_extent("m", 8);
        let x = make_tensor(&param_ptr::<f32>("x"), Layout::row_major((m, 16)));
        let tiled = make_tiled_copy_tv(&atom(), &Layout::row_major((8, 4)), &Layout::row_major((1, 4)));
        let thr = tiled.get_slice(thread_idx().x);
        let src = thr.partition_s(&x);
        for_range(0, size(&src, &[1]), 1, |i| {
            let dst = make_fragment::<f32>((4, 1));
            copy(&tiled, &src.slice((.., i, ..)), &dst);
        });
    });
    // copy_if names itself and the predicate.
    let msg = build_panic(|| {
        Kernel::build("dynamic_copy_if", || {
            let src = dynamic_tensor("x");
            let pred = make_fragment::<bool>(1);
            copy_if(&make_copy_atom::<f32>(CopyOp::Universal { bits: 32 }), &pred, &src, &make_fragment::<f32>(8));
        });
    });
    assert!(msg.contains("copy_if: the source tensor (?,8):(1,?) has the dynamic extent %"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("dynamic_copy_if_destination", || {
            let pred = make_fragment::<bool>(8);
            pred.fill(true);
            let src = make_fragment::<f32>(8);
            src.fill(1.0);
            copy_if(&make_copy_atom::<f32>(CopyOp::Universal { bits: 32 }), &pred, &src, &dynamic_tensor("y"));
        });
    });
    assert!(
        msg.contains("copy_if: the destination tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("in its mode 0, the chunk of values each thread copies"),
        "{msg}"
    );
    // autovec_copy copies every element with unrolled instructions.
    let msg = build_panic(|| {
        Kernel::build("dynamic_elementwise", || autovec_copy(&make_fragment::<f32>((4, 8)), &dynamic_tensor("y")));
    });
    assert!(
        msg.contains("autovec_copy: the destination tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("but autovec_copy copies every element with instructions unrolled when the kernel")
            && msg.contains("write a for_range over the rest mode"),
        "{msg}"
    );
    // basic_copy and basic_copy_if of runtime size loop at run time, which
    // a register fragment cannot take part in.
    let msg = build_panic(|| {
        Kernel::build("dynamic_basic_copy", || basic_copy(&make_fragment::<f32>((4, 8)), &dynamic_tensor("y")));
    });
    assert!(
        msg.contains(
            "basic_copy: the source tensor is a register fragment, but the source (4,8):(1,4) and \
                      destination (?,8):(1,?) do not both have static shapes"
        ) && msg.contains("a register fragment is indexed with static coordinates only"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_basic_copy_if", || {
            let y = dynamic_tensor("y");
            let pred = make_fragment::<bool>((4, 8));
            pred.fill(true);
            basic_copy_if(&pred, &dynamic_tensor("x"), &y);
        });
    });
    assert!(msg.contains("basic_copy_if: the predicate tensor is a register fragment"), "{msg}");
    // gemm unrolls its modes.
    let msg = build_panic(|| {
        Kernel::build("dynamic_gemm", || {
            let mma = make_tiled_mma(MmaOp::UniversalFma(DType::F32), Layout::col_major((4, 8, 1)));
            let thr = mma.get_slice(thread_idx().x);
            let k = param_extent("k", 1);
            let a = make_tensor(&param_ptr::<f32>("a"), Layout::col_major((4, k)));
            let b = make_tensor(&param_ptr::<f32>("b"), Layout::col_major((8, k)));
            let acc =
                thr.partition_fragment_c::<f32, _>(&make_tensor(&param_ptr::<f32>("c"), Layout::col_major((4, 8))));
            gemm(&mma, &acc, &thr.partition_a(&a), &thr.partition_b(&b), &acc);
        });
    });
    assert!(
        msg.contains("gemm: operand A ((1,1,?):(0,0,4)) has the dynamic extent %")
            && msg.contains(
                "gemm unrolls the per-thread values and the MMA_M, MMA_N and MMA_K modes when the kernel is built"
            ),
        "{msg}"
    );
    // retile regroups a fragment's values.
    let msg = build_panic(|| {
        Kernel::build("dynamic_retile", || {
            let tiled = make_tiled_copy_tv(&atom(), &Layout::row_major((8, 4)), &Layout::row_major((1, 4)));
            let _ = tiled.retile(&dynamic_tensor("x"));
        });
    });
    assert!(msg.contains("retile: tensor (?,8):(1,?) has the dynamic extent %"), "{msg}");
}

/// Coordinate tensors: elem_less needs congruent tuples, a swizzled tensor
/// takes only a shift that is provably non-negative, and an access before
/// the start of a tensor shifted by a negative domain_offset is reported
/// when it is not predicated off (a predicated-off cp.async there is not).
#[test]
fn coordinate_tensor_misuse() {
    // The CuTe dialect's incongruent elem_less operands.
    let msg = build_panic(|| {
        Kernel::build("elem_less_ranks", || {
            let _ = elem_less((3, 4), (3, 4, 5));
        });
    });
    assert!(msg.contains("elem_less: (3,4) and (3,4,5) are not congruent"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("elem_less_nesting", || {
            let _ = elem_less(((1, 2), 3), (4, 5));
        });
    });
    assert!(msg.contains("elem_less: ((1,2),3) and (4,5) are not congruent"), "{msg}");
    // A negative shift of a swizzled tensor.
    let msg = build_panic(|| {
        Kernel::build("swizzled_shift", || {
            let s = make_smem_tensor::<f32>(make_composed_layout(make_swizzle(2, 2, 3), 0, Layout::row_major((8, 32))));
            let r = IndexExpr::signed(param_scalar::<i32>("r"));
            let _ = domain_offset((0, r), &s);
        });
    });
    assert!(
        msg.contains("domain_offset: the shift %") && msg.contains("swizzled by S<2,2,3> may be negative"),
        "{msg}"
    );
    // A non-negative shift of a swizzled tensor keeps the swizzle.
    Kernel::build("swizzled_shift_ok", || {
        let s = make_smem_tensor::<f32>(make_composed_layout(make_swizzle(2, 2, 3), 0, Layout::row_major((8, 32))));
        let shifted = domain_offset((1, 4), &s);
        assert_eq!((shifted.base_offset(), shifted.swizzle()), (&IndexExpr::Static(36), s.swizzle()));
    });
    // A load before the shifted origin is out of bounds, at its line.
    let line = std::cell::Cell::new(0);
    let kernel = Kernel::builder("before_origin").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let y = param_tensor::<f32>("y", Layout::col_major(32));
        let r = IndexExpr::signed(param_scalar::<i32>("r"));
        let shifted = domain_offset(r, &x);
        let t = thread_idx().x;
        line.set(line!() + 1);
        y.store(t, shifted.load(t));
    });
    let mut args = vec![KernelArg::f32(vec![1.0; 64]), KernelArg::f32(vec![0.0; 32]), KernelArg::ScalarI32(-3)];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.message.contains("out of bounds for `x` (64 elements)"), "{e}");
    let loc = e.loc.expect("the load's location");
    assert_eq!((loc.file(), loc.line()), ("tests/diagnostics/layouts.rs", line.get()), "{e}");
    let mut args = vec![KernelArg::f32(vec![1.0; 64]), KernelArg::f32(vec![0.0; 32]), KernelArg::ScalarI32(3)];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    // cp.async before the origin: zero-filled when predicated off, reported
    // when not.
    let kernel = Kernel::builder("cp_async_before_origin").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let y = param_tensor::<f32>("y", Layout::col_major(32));
        let r = IndexExpr::signed(param_scalar::<i32>("r"));
        let all = param_scalar::<i32>("all");
        let shifted = domain_offset(r.clone(), &x);
        let coords = domain_offset(r, &make_identity_tensor(64));
        let smem = make_smem_tensor::<f32>(Layout::col_major(32));
        let atom = make_copy_atom::<f32>(CopyOp::CpAsync { bytes: 4, bypass_l1: false });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(1));
        let t = thread_idx().x;
        let thr = tiled.get_slice(t);
        let src = thr.partition_s(&local_tile(&shifted, 32, 0));
        let c = thr.partition_s(&local_tile(&coords, 32, 0));
        let pred = make_predicate(&c, |c| elem_less(-1, c) | all.ne(0));
        copy_if(&tiled, &pred, &src, &thr.partition_d(&smem));
        cp_async_commit_group();
        cp_async_wait_group(0);
        sync_threads();
        y.store(t, smem.load(t));
    });
    let input: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let mut args = vec![KernelArg::f32(input.clone()), KernelArg::f32(vec![7.0; 32]), KernelArg::ScalarI32(-3)];
    args.push(KernelArg::ScalarI32(0));
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<f32> = (0..32).map(|t| if t < 3 { 0.0 } else { (t - 3) as f32 }).collect();
    assert_eq!(args[1].as_f32(), want);
    let mut args = vec![KernelArg::f32(input), KernelArg::f32(vec![7.0; 32]), KernelArg::ScalarI32(-3)];
    args.push(KernelArg::ScalarI32(1));
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.message.contains("cp.async source read") && e.message.contains("out of bounds for `x`"), "{e}");
}

/// The shape arithmetic's misuse: ceil_div and round_up rank errors,
/// shape_div of static leaves of which neither divides the other, and the
/// widths of basic_copy_if.
#[test]
fn shape_arithmetic_misuse() {
    let msg = build_panic(|| {
        let _ = ceil_div((4, 8), (2, 4, 2));
    });
    assert!(
        msg.contains("ceil_div: cannot divide (4,8) by (2,4,2): the tiler (2,4,2) has rank 3, more than the rank 2"),
        "{msg}"
    );
    let msg = build_panic(|| {
        let _ = ceil_div(7, 0);
    });
    assert!(msg.contains("ceil_div: cannot divide 7 by 0: the tiler has an extent of 0"), "{msg}");
    let msg = build_panic(|| {
        let _ = round_up((9, 10), (4, 8, 2));
    });
    assert!(msg.contains("round_up: cannot round (9,10) up to a multiple of (4,8,2): (4,8,2) has rank 3"), "{msg}");
    let msg = build_panic(|| {
        let _ = round_up(9, (4, 8));
    });
    assert!(msg.contains("the integer 9 needs an integer to round to, not the tuple (4,8)"), "{msg}");
    let msg = build_panic(|| {
        let _ = round_up(Shape::Branch(vec![]), 4);
    });
    assert!(msg.contains("round_up: cannot round () up to a multiple of 4: () is the empty tuple"), "{msg}");
    let msg = build_panic(|| {
        let _ = shape_div(7, 3);
    });
    assert!(msg.contains("shape_div: cannot divide 7 by 3: neither 7 nor 3 divides the other"), "{msg}");
    let msg = build_panic(|| {
        let _ = shape_div((4, 6), (2, 4, 1));
    });
    assert!(msg.contains("shape_div: cannot divide (4,6) by (2,4,1): (4,6) has rank 2 and (2,4,1) rank 3"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("copy_if_widths", || {
            let pred = make_fragment::<bool>(8);
            pred.fill(true);
            basic_copy_if(&pred, &make_fragment::<f32>(8), &make_fragment::<f64>(8));
        });
    });
    assert!(
        msg.contains(
            "basic_copy_if: the source elements (f32, 32 bits) and the destination elements (f64, 64 bits) \
                      differ in width"
        ),
        "{msg}"
    );
    // Dividing or masking an index that may be negative.
    let msg = build_panic(|| {
        Kernel::build("negative_division", || {
            let r = IndexExpr::signed(param_scalar::<i32>("r"));
            let _ = (r + 8usize) / 4usize;
        });
    });
    assert!(
        msg.contains("cute: index expression (%")
            && msg.contains("may be negative (it holds a negative or signed term)"),
        "{msg}"
    );
    let msg = build_panic(|| {
        let _ = IndexExpr::from(-3) % 2usize;
    });
    assert!(msg.contains("cute: index expression -3 % 2: the operand -3 may be negative"), "{msg}");
}

/// Accesses past a runtime extent index the buffer like any other: the
/// simulator reports the one that falls outside the buffer, at its line.
#[test]
fn overruns_of_runtime_extents_are_reported() {
    let line = std::cell::Cell::new(0);
    let kernel = Kernel::builder("runtime_overrun").block_dim([32, 1, 1]).build(|| {
        let n = param_extent("n", 1);
        let x = param_tensor::<f32>("x", Layout::col_major(n));
        let t = thread_idx().x;
        line.set(line!() + 1);
        x.store(t, 1.0);
    });
    let mut args = vec![KernelArg::ScalarI32(20), KernelArg::f32(vec![0.0; 20])];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.message.contains("out of bounds for `x` (20 elements)"), "{e}");
    let loc = e.loc.expect("the store's location");
    assert_eq!((loc.file(), loc.line()), ("tests/diagnostics/layouts.rs", line.get()), "{e}");
    // In bounds, it runs.
    let mut args = vec![KernelArg::ScalarI32(32), KernelArg::f32(vec![0.0; 32])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].as_f32(), vec![1.0; 32]);
}

/// `rank` takes a mode path as `size`, `shape` and `cosize` do, and refuses
/// one that leaves the tuple or goes on past a leaf, naming the function.
#[test]
fn mode_paths_of_rank() {
    let l = Layout::new((4, (3, 2)), (2, (8, 1)));
    let msg = build_panic(|| {
        rank(&l, &[2]);
    });
    assert!(msg.contains("rank: mode path [2] does not fit: (4,(3,2)) has rank 2, so it has no mode 2"), "{msg}");
    let msg = build_panic(|| {
        rank(&make_composed_layout(make_swizzle(2, 3, 3), 0, l.clone()), &[0, 0]);
    });
    assert!(
        msg.contains("rank: mode path [0, 0] does not fit: mode [0] of (4,(3,2)) is a leaf, so the path cannot go on"),
        "{msg}"
    );
    let msg = build_panic(|| {
        rank(&IntTuple::from(8), &[1]);
    });
    assert!(msg.contains("rank: mode path [1] does not fit: 8 has rank 1, so it has no mode 1"), "{msg}");
}

/// Breaking an assumption is undefined behaviour on the GPU: the simulator
/// reports the value, the promised divisor and the `assume` line.
#[test]
fn violated_assumption_names_its_line() {
    let mut line = 0;
    let kernel = Kernel::builder("broken_assume").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        line = line!() + 1;
        let even = assume(t * 2 + 2, Some(4), false);
        out.store(t, even);
    });
    let mut args = vec![KernelArg::buffer(&[0i32; 32])];
    let err = simulate_err(&kernel, 32, &mut args);
    assert_eq!(err.loc.map(|l| l.line()), Some(line), "{err}");
    assert_eq!(err.thread, Some([0, 0, 0]), "{err}");
    assert!(err.message.contains("the value 2 is not a multiple of 4"), "{err}");
    let kernel = Kernel::builder("broken_pow2").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        out.store(t, assume(t + 1, None, true));
    });
    let err = simulate_err(&kernel, 32, &mut [KernelArg::buffer(&[0i32; 32])]);
    assert_eq!(err.thread, Some([2, 0, 0]), "{err}");
    assert!(err.message.contains("the value 3 is not a power of two"), "{err}");
}

/// `assume` checks its arguments while the kernel is built: a divisor of 0,
/// `pow2` with a divisor that is not a power of two, and a static value
/// that breaks its own promise.
#[test]
fn bad_assumptions_are_refused_at_build_time() {
    let msg = build_panic(|| {
        Kernel::build("divby_zero", || {
            let _ = assume(param_scalar::<i32>("n"), Some(0), false);
        });
    });
    assert!(msg.contains("divby must be a positive integer"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("pow2_divby", || {
            let _ = assume(param_scalar::<i32>("n"), Some(12), true);
        });
    });
    assert!(msg.contains("with pow2, divby must be a power of two, not 12"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("static_breaks", || {
            let _ = assume(Int32::from(18), Some(4), false);
        });
    });
    assert!(msg.contains("the static value 18 is not a multiple of 4"), "{msg}");
    let msg = build_panic(|| {
        let _ = assume(IndexExpr::Static(24), Some(8), true);
    });
    assert!(msg.contains("the static value 24 is not a power of two"), "{msg}");
    let msg = build_panic(|| {
        let _ = assume(-8i32, None, true);
    });
    assert!(msg.contains("the static value -8 is not a power of two"), "{msg}");
}

/// Layouts with scaled-basis strides map coordinates to tuples, so what
/// needs offsets refuses them, naming the basis stride.
#[test]
fn basis_strides_are_refused_where_offsets_are_needed() {
    let id = make_identity_layout((4, 8));
    let msg = build_panic(|| {
        id.offset(&(1, 2).into());
    });
    assert!(
        msg.contains(
            "Layout::offset: layout (4,8):(1@0,1@1) has the basis stride 1@0 at mode [0], so it maps a \
                      coordinate to a tuple, not an offset; use Layout::eval"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        crd2idx((1, 2), &id);
    });
    assert!(msg.contains("crd2idx: layout (4,8):(1@0,1@1) has the basis stride 1@0"), "{msg}");
    let msg = build_panic(|| {
        id.slice(&(.., 1).into());
    });
    assert!(
        msg.contains("Layout::slice: layout (4,8):(1@0,1@1) has the basis stride 1@0")
            && msg.contains("slice_and_offset"),
        "{msg}"
    );
    let msg = build_panic(|| {
        make_composed_layout(make_swizzle(2, 3, 3), 0, id.clone());
    });
    assert!(
        msg.contains("make_composed_layout: invalid composition S<2,3,3> o 0 o (4,8):(1@0,1@1)")
            && msg.contains("basis stride 1@0 at mode [0]"),
        "{msg}"
    );
    let msg = build_panic(|| {
        id.upcast(2);
    });
    assert!(msg.contains("upcast: layout (4,8):(1@0,1@1) has the basis stride 1@0 at mode [0]"), "{msg}");
    let msg = build_panic(|| {
        make_layout_like(&id);
    });
    assert!(msg.contains("make_layout_like: layout (4,8):(1@0,1@1) has the basis stride"), "{msg}");
    // The inverses and complements of basis layouts take integer scales
    // and no integer strides besides 0.
    let msg = build_panic(|| {
        "(4,8):(1@0,4)".parse::<Layout>().unwrap().right_inverse();
    });
    assert!(msg.contains("right_inverse: layout (4,8):(1@0,4) mixes basis strides with the integer stride 4"), "{msg}");
    let msg = build_panic(|| {
        "(2,3):(2@0,3@0)".parse::<Layout>().unwrap().left_inverse();
    });
    assert!(msg.contains("left_inverse: layout (2,3):(2@0,3@0) is not injective"), "{msg}");
    // Arithmetic tuples that are not defined.
    let msg = build_panic(|| {
        "(2,3):(1@0,2)".parse::<Layout>().unwrap().eval(&(1, 1).into());
    });
    assert!(msg.contains("Layout::eval: layout (2,3):(1@0,2) at (1,1) is not defined: the integer stride 2"), "{msg}");
    let msg = build_panic(|| {
        "(2,3):(1@2,1@0@2)".parse::<Layout>().unwrap().eval(&(1, 1).into());
    });
    assert!(msg.contains("stride 1@0@2: a mode path goes on through the integer 1"), "{msg}");
    let msg = build_panic(|| {
        "4:1/4@0".parse::<Layout>().unwrap().eval(&2.into());
    });
    assert!(msg.contains("2 * 1/4 is 1/2, not a whole number"), "{msg}");
    let msg = build_panic(|| {
        complement(&"(2,3):(1@0,2)".parse::<Layout>().unwrap(), 12);
    });
    assert!(msg.contains("complement: layout (2,3):(1@0,2) mixes basis strides with the integer stride 2"), "{msg}");
}

/// As in CuTe DSL, a basis element whose scale is a ratio cannot be scaled.
#[test]
fn scaling_a_ratio_basis_is_refused() {
    let msg = build_panic(|| {
        let _ = StrideLeaf::from(ScaledBasis::new(Ratio::new(1, 2), 0)) * 4usize;
    });
    assert!(
        msg.contains(
            "cannot scale the basis element 1/2@0 by 4: scaling a basis element whose scale is a ratio is \
                      not supported"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        let _ = Ratio::new(1, 0);
    });
    assert!(msg.contains("Ratio::new(1, 0): the denominator must not be 0"), "{msg}");
    let msg = build_panic(|| {
        composition(&Layout::col_major((8, 8)), &"2:1/2@0".parse::<Layout>().unwrap());
    });
    assert!(
        msg.contains("composition (8,8):(1,8) ∘ 2:1/2@0 is not defined: the basis stride 1/2@0 has a ratio"),
        "{msg}"
    );
    let msg = build_panic(|| {
        composition(&Layout::col_major((8, 8)), &"2:1@1@1".parse::<Layout>().unwrap());
    });
    assert!(
        msg.contains(
            "composition (8,8):(1,8) ∘ 2:1@1@1 is not defined: the basis stride 1@1@1 names mode [1, 1], but \
                      mode [1] of the left-hand side is a leaf"
        ),
        "{msg}"
    );
}
