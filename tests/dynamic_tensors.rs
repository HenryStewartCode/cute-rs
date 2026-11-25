//! Tensors whose extents are runtime values. Every tiling and partitioning
//! function is applied to tensors of launch-time extents and checked, after
//! evaluating the extents at many values, two ways:
//!
//! * where the tilers divide the extents, against the same function on the
//!   tensors of those extents as static numbers: the same shape, base offset
//!   and swizzle, and the same offset at every element;
//! * at any extents, against the static-upper-bound idiom (a static bound as
//!   the extent, the same runtime strides): the same offset at every
//!   coordinate of the tensor's own, smaller, domain. Tiles of the bound
//!   are whole, so this also covers the positions of a partial tile past
//!   the runtime extent, which a kernel must predicate away.
//!
//! The size queries fold to static values where they can, and evaluate to
//! the static answers. And `runtime_gemm` with runtime extents computes the
//! static-bound kernel's bits on the simulator.

#[allow(dead_code)]
#[path = "../examples/runtime_gemm.rs"]
mod runtime_gemm;

use cute::ir::ValueId;
use cute::layout::idx2crd;
use cute::prelude::*;

/// Extents the tilers below divide (with at least two tiles of each, so
/// that the tile coordinates below are inside), and extents they do not.
const DIVISIBLE: [(usize, usize); 6] = [(16, 16), (16, 32), (32, 48), (64, 16), (16, 64), (48, 32)];
const ANY: [(usize, usize); 9] = [(1, 4), (5, 12), (13, 20), (8, 36), (31, 4), (2, 100), (17, 16), (40, 52), (64, 64)];

/// The static upper bound of the static-upper-bound layouts: a multiple of
/// every tile.
const BOUND: usize = 1 << 10;

/// A runtime extent: the value and its id, which evaluation looks up.
fn extent(name: &str, multiple_of: usize) -> (IndexExpr, ValueId) {
    let value = param_extent(name, multiple_of).as_index();
    let IndexExpr::Dynamic(id) = value else { panic!("an extent parameter is a runtime value") };
    (value, id)
}

/// Evaluate `e` with each runtime value `ids[i]` at `values[i]`.
fn eval(e: &IndexExpr, ids: &[ValueId], values: &[usize]) -> usize {
    e.eval(&|v| values[ids.iter().position(|&id| id == v).unwrap_or_else(|| panic!("{v} is not an extent"))])
}

/// `dynamic`, evaluated at the extents, is `want`: the same shape, base
/// offset and swizzle, and the same offset at every element.
fn assert_same_tensor<T>(what: &str, dynamic: &Tensor<T>, want: &Tensor<T>, ids: &[ValueId], values: &[usize]) {
    let shape = dynamic.shape().map(&mut |e| eval(e, ids, values));
    let want_shape = want.shape().map(&mut |e| eval(e, ids, values));
    assert_eq!(shape, want_shape, "{what} at {values:?}: the shape of {dynamic:?} (static {want:?})");
    assert_eq!(dynamic.swizzle(), want.swizzle(), "{what}: swizzle");
    let base = eval(dynamic.base_offset(), ids, values);
    let want_base = eval(want.base_offset(), ids, values);
    assert_eq!(base, want_base, "{what} at {values:?}: the base offset of {dynamic:?} (static {want:?})");
    for i in 0..want_shape.size() {
        let got = eval(&dynamic.layout().offset_at(i), ids, values);
        let expected = eval(&want.layout().offset_at(i), ids, values);
        assert_eq!(got, expected, "{what} at {values:?}: the offset of element {i} of {dynamic:?} (static {want:?})");
    }
}

/// `dynamic`, evaluated at the extents, maps every coordinate of its own
/// domain to the offset `bound` maps it to; the two shapes have the same
/// rank and nesting, and `bound`'s domain holds `dynamic`'s.
fn assert_same_on_domain<T>(what: &str, dynamic: &Tensor<T>, bound: &Tensor<T>, ids: &[ValueId], values: &[usize]) {
    let shape = dynamic.shape().map(&mut |e| eval(e, ids, values));
    let bound_shape = bound.layout().static_shape();
    assert!(shape.congruent(&bound_shape), "{what} at {values:?}: {dynamic:?} and {bound:?} are not congruent");
    let fits = shape.flat_leaves().iter().zip(bound_shape.flat_leaves()).all(|(&s, b)| s <= b);
    assert!(fits, "{what} at {values:?}: the domain of {dynamic:?} is not inside that of {bound:?}");
    let base = eval(dynamic.base_offset(), ids, values);
    assert_eq!(base, eval(bound.base_offset(), ids, values), "{what} at {values:?}: the base offset of {dynamic:?}");
    for i in 0..shape.size() {
        let coord = idx2crd(i, &shape);
        let got = eval(&dynamic.layout().offset(&coord), ids, values);
        let expected = eval(&bound.layout().offset(&coord), ids, values);
        assert_eq!(got, expected, "{what} at {values:?}: the offset of {coord:?} in {dynamic:?} (bound {bound:?})");
    }
}

/// The layouts every operation is tried on, of extents `(m, n)` and a
/// leading dimension `ld`: row-major and column-major matrices, a
/// hierarchical one and one with a separate leading dimension.
fn layouts(m: IndexExpr, n: IndexExpr, ld: IndexExpr) -> Vec<Layout> {
    vec![
        Layout::row_major((m.clone(), n.clone())),
        Layout::col_major((m.clone(), n.clone())),
        Layout::new((m.clone(), (2usize, n.clone())), (1usize, (m.clone(), m.clone() * 2usize))),
        Layout::new((m, n), (1usize, ld)),
    ]
}

const SOURCES: [&str; 4] = ["row-major", "column-major", "hierarchical", "leading-dimension"];

/// The source layouts with static upper bounds as their extents and the
/// strides of the runtime ones.
fn bound_layouts(m: IndexExpr, n: IndexExpr, ld: IndexExpr) -> Vec<Layout> {
    let b = || IndexExpr::Static(BOUND);
    vec![
        Layout::new((b(), b()), (n, 1usize)),
        Layout::new((b(), b()), (1usize, m.clone())),
        Layout::new((b(), (2usize, b())), (1usize, (m.clone(), m * 2usize))),
        Layout::new((b(), b()), (1usize, ld)),
    ]
}

type Op = Box<dyn Fn(&Tensor<f32>) -> Tensor<f32>>;

/// The tiling and partitioning operations tried on a tensor of rank 2; `r`
/// is a runtime value, 1 when evaluated.
fn operations(r: IndexExpr) -> Vec<(String, Op)> {
    let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
    let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((8, 4)), &Layout::row_major((1, 4)));
    let mma = make_tiled_mma(make_mma_atom(MmaOp::UniversalFma(DType::F32)), Layout::col_major((4, 8, 1)));
    let tensor_core =
        make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)), Layout::col_major((2, 2, 1)));
    let mut ops: Vec<(String, Op)> = vec![
        ("local_tile at a tile".into(), Box::new(|t| local_tile(t, (8, 4), (1, 2)))),
        ("local_tile keeping the rest".into(), Box::new(|t| local_tile(t, (8, 4), (1, ..)))),
        ("local_tile keeping both rests".into(), Box::new(|t| local_tile(t, (4, 8), (.., ..)))),
        ("local_tile at a runtime tile".into(), Box::new(move |t| local_tile(t, (8, 16), (r.clone(), 0)))),
        ("zipped_divide".into(), Box::new(|t| zipped_divide(t, (4, 8)))),
        ("logical_divide".into(), Box::new(|t| logical_divide(t, (Layout::new(2, 1), ..)))),
        ("tiled_divide".into(), Box::new(|t| tiled_divide(t, (8, 4)))),
        ("flat_divide".into(), Box::new(|t| flat_divide(t, (8, 4)))),
        ("composition".into(), Box::new(|t| t.compose((Layout::new(4, 2), Layout::new(2, 1))))),
        ("slice".into(), Box::new(|t| t.slice((.., 3)))),
        ("group_modes".into(), Box::new(|t| t.group_modes(0, 2))),
    ];
    for thread in [0usize, 5, 31] {
        let thr = tiled.get_slice(thread);
        let (s, d) = (thr.clone(), thr);
        ops.push((format!("partition_s, thread {thread}"), Box::new(move |t| s.partition_s(t))));
        ops.push((
            format!("partition_d of a band, thread {thread}"),
            Box::new(move |t| d.partition_d(&local_tile(t, (8, 16), (1, ..)))),
        ));
        let tm = mma.get_slice(thread);
        let (a, b, c) = (tm.clone(), tm.clone(), tm);
        ops.push((format!("partition_a, thread {thread}"), Box::new(move |t| a.partition_a(t))));
        ops.push((format!("partition_b, thread {thread}"), Box::new(move |t| b.partition_b(t))));
        ops.push((format!("partition_c, thread {thread}"), Box::new(move |t| c.partition_c(t))));
        let tc = tensor_core.get_slice(thread * 3 % 128);
        ops.push((format!("tensor-core partition_c, thread {thread}"), Box::new(move |t| tc.partition_c(t))));
        let threads = Layout::col_major((4, 8));
        let lp = threads.clone();
        ops.push((format!("local_partition, thread {thread}"), Box::new(move |t| local_partition(t, &lp, thread))));
        ops.push((
            format!("local_partition_proj, thread {thread}"),
            Box::new(move |t| local_partition_proj(t, &threads, thread % 32, &[false, true])),
        ));
    }
    ops
}

/// Where the tilers divide the extents, every operation on every source
/// layout of runtime extents agrees with the operation on the static
/// layouts of the extents' values.
#[test]
fn tiles_and_partitions_of_divisible_runtime_extents_match_static_ones() {
    Kernel::build("divisible_tiles", || {
        let (m, m_id) = extent("m", 1);
        let (n, n_id) = extent("n", 4);
        let (ld, ld_id) = extent("ld", 1);
        let (r, r_id) = extent("r", 1);
        let (p, ids) = (param_ptr::<f32>("x"), [m_id, n_id, ld_id, r_id]);
        let ops = operations(r);
        for (i, dynamic) in layouts(m, n, ld).into_iter().enumerate() {
            let dynamic = make_tensor(&p, dynamic);
            for (mv, nv) in DIVISIBLE {
                let values = [mv, nv, mv + 3, 1];
                let fixed = make_tensor(&p, layouts(mv.into(), nv.into(), (mv + 3).into()).swap_remove(i));
                let name = SOURCES[i];
                assert_same_tensor(name, &dynamic, &fixed, &ids, &values);
                for (op, f) in &ops {
                    assert_same_tensor(&format!("{op} of the {name} tensor"), &f(&dynamic), &f(&fixed), &ids, &values);
                }
            }
        }
    });
}

/// At any extents, every operation on every source layout of runtime
/// extents maps each coordinate of its result as the operation on the
/// static-upper-bound layout does, and its rest modes count the tiles.
#[test]
fn tiles_and_partitions_of_any_runtime_extents_follow_the_static_bound() {
    Kernel::build("any_tiles", || {
        let (m, m_id) = extent("m", 1);
        let (n, n_id) = extent("n", 4);
        let (ld, ld_id) = extent("ld", 1);
        let (r, r_id) = extent("r", 1);
        let (p, ids) = (param_ptr::<f32>("x"), [m_id, n_id, ld_id, r_id]);
        let ops = operations(r);
        let dynamic = layouts(m.clone(), n.clone(), ld.clone());
        let bound = bound_layouts(m, n, ld);
        for (i, (dynamic, bound)) in dynamic.into_iter().zip(bound).enumerate() {
            let (dynamic, bound) = (make_tensor(&p, dynamic), make_tensor(&p, bound));
            for (mv, nv) in ANY {
                let values = [mv, nv, mv + 3, 1];
                let name = SOURCES[i];
                assert_same_on_domain(name, &dynamic, &bound, &ids, &values);
                for (op, f) in &ops {
                    assert_same_on_domain(
                        &format!("{op} of the {name} tensor"),
                        &f(&dynamic),
                        &f(&bound),
                        &ids,
                        &values,
                    );
                }
                // The rest modes count the tiles: mode 1 of the hierarchical
                // layout has 2n elements.
                let rests = local_tile(&dynamic, (8, 4), (.., ..));
                let counts = [size(&rests, &[2]), size(&rests, &[3])].map(|e| eval(&e, &ids, &values));
                let columns = if name == "hierarchical" { 2 * nv } else { nv };
                assert_eq!(counts, [mv.div_ceil(8), columns.div_ceil(4)], "{name} at {values:?}");
            }
        }
    });
}

/// A GEMM's tiles: `local_tile_proj` of `(m, k)`, `(n, k)` and `(m, n)`
/// tensors of runtime extents keeps the rest mode of K tiles, whose extent
/// is the number of K tiles, and the tiles and their MMA partitions agree
/// with the static ones.
#[test]
fn gemm_tiles_of_runtime_extents() {
    Kernel::build("gemm_tiles", || {
        let (m, m_id) = extent("m", 1);
        let (n, n_id) = extent("n", 1);
        let (k, k_id) = extent("k", 8);
        let ids = [m_id, n_id, k_id];
        let (a, b, c) = (param_ptr::<f32>("a"), param_ptr::<f32>("b"), param_ptr::<f32>("c"));
        let tensors = |m: IndexExpr, n: IndexExpr, k: IndexExpr| {
            (
                make_tensor(&a, Layout::row_major((m.clone(), k.clone()))),
                make_tensor(&b, Layout::row_major((n.clone(), k))),
                make_tensor(&c, Layout::row_major((m, n))),
            )
        };
        let tiler = (32, 16, 8);
        let tiles = |(a, b, c): &(Tensor<f32>, Tensor<f32>, Tensor<f32>)| {
            (
                local_tile_proj(a, tiler, (1, 2, ..), &[true, false, true]),
                local_tile_proj(b, tiler, (1, 2, ..), &[false, true, true]),
                local_tile_proj(c, tiler, (1, 2, ..), &[true, true, false]),
            )
        };
        let (ga, gb, gc) = tiles(&tensors(m, n, k));
        // The number of K tiles is the extent of the rest mode.
        let k_tiles = size(&ga, &[2]);
        assert!(!k_tiles.is_static());
        assert_eq!(size(&gb, &[2]), k_tiles);
        assert_eq!((shape(&ga, &[0]), rank(&gc, &[])), (Shape::from(32), 2));
        let mma = make_tiled_mma(make_mma_atom(MmaOp::UniversalFma(DType::F32)), Layout::col_major((8, 4, 1)));
        for (mv, nv, kv) in [(64, 64, 8), (96, 80, 40), (64, 48, 64), (40, 48, 8)] {
            let values = [mv, nv, kv];
            assert_eq!(eval(&k_tiles, &ids, &values), kv.div_ceil(8));
            let (fa, fb, fc) = tiles(&tensors(mv.into(), nv.into(), kv.into()));
            assert_same_tensor("A's tiles", &ga, &fa, &ids, &values);
            assert_same_tensor("B's tiles", &gb, &fb, &ids, &values);
            assert_same_tensor("C's tile", &gc, &fc, &ids, &values);
            for thread in [0usize, 7, 31] {
                let thr = mma.get_slice(thread);
                assert_same_tensor("partition_a", &thr.partition_a(&ga), &thr.partition_a(&fa), &ids, &values);
                assert_same_tensor("partition_b", &thr.partition_b(&gb), &thr.partition_b(&fb), &ids, &values);
                assert_same_tensor("partition_c", &thr.partition_c(&gc), &thr.partition_c(&fc), &ids, &values);
                // One K tile, at a runtime index, has static extents.
                let one = thr.partition_a(&ga).slice((.., .., .., thread_idx().x % 4));
                assert!(one.shape().is_static(), "{one:?}");
            }
        }
    });
}

/// `size`, `shape`, `cosize`, `rank` and `depth` of tensors: static where
/// the extents (and, for `cosize`, the strides) they depend on are, so Rust
/// loops still unroll over them, and runtime values otherwise, which
/// evaluate to the static answers.
#[test]
fn size_queries_of_tensors() {
    Kernel::build("size_queries", || {
        let (m, m_id) = extent("m", 1);
        let (n, n_id) = extent("n", 4);
        let ids = [m_id, n_id];
        let p = param_ptr::<f32>("x");
        let x = make_tensor(
            &p,
            Layout::new((m.clone(), (4usize, n.clone())), (1usize, (m.clone() * 2usize, m.clone() * 8usize))),
        );
        let tiles = local_tile(&make_tensor(&p, Layout::row_major((m, n))), (8, 4), (.., ..));
        // Static parts fold.
        assert_eq!(size(&x, &[1, 0]), IndexExpr::Static(4));
        assert_eq!(size(&tiles, &[0]), IndexExpr::Static(8));
        assert_eq!(size(&tiles, &[1]), IndexExpr::Static(4));
        assert_eq!(cosize(&tiles, &[1]), IndexExpr::Static(4));
        assert_eq!((rank(&x, &[]), rank(&x, &[1]), rank(&x, &[1, 1]), rank(&tiles, &[])), (2, 2, 1, 4));
        assert_eq!((depth(&x), x.depth(), depth(&tiles), tiles.depth()), (2, 2, 1, 1));
        assert_eq!(shape(&x, &[1, 0]), Shape::from(4));
        assert!(!size(&x, &[]).is_static() && !size(&tiles, &[2]).is_static());
        // A tile of the tiles has static extents, so static sizes; its
        // cosize depends on the runtime row stride.
        let tile = tiles.slice((.., .., 1, 2));
        assert_eq!((tile.size(), size(&tile, &[])), (32, IndexExpr::Static(32)));
        assert!(!cosize(&tile, &[]).is_static());
        for (mv, nv) in [(1, 4), (8, 12), (13, 20), (64, 36)] {
            let values = [mv, nv];
            let at = |e: IndexExpr| eval(&e, &ids, &values);
            assert_eq!(at(size(&x, &[])), mv * 4 * nv);
            assert_eq!(at(size(&x, &[0])), mv);
            assert_eq!(at(size(&x, &[1])), 4 * nv);
            assert_eq!(at(size(&x, &[1, 1])), nv);
            assert_eq!(shape(&x, &[1]).map(&mut |e| at(e.clone())), IntTuple::from((4, nv)));
            // One past the largest offset: 1 + (m-1) + 3*2m + (n-1)*8m.
            assert_eq!(at(cosize(&x, &[])), mv + 6 * mv + (nv - 1) * 8 * mv);
            assert_eq!(at(cosize(&x, &[1])), 1 + 6 * mv + (nv - 1) * 8 * mv);
            assert_eq!(at(cosize(&x, &[0])), mv);
            assert_eq!(at(size(&tiles, &[2])), mv.div_ceil(8));
            assert_eq!(at(size(&tiles, &[3])), nv.div_ceil(4));
            assert_eq!(at(cosize(&tile, &[])), 7 * nv + 4);
        }
    });
}

/// A swizzled tensor's cosize is its layout's, as is the swizzled layout's
/// it is made from, as in CuTe (the allocation is still rounded up to whole
/// swizzle periods).
#[test]
fn cosize_of_a_swizzled_tensor() {
    Kernel::build("swizzled_cosize", || {
        let layout = make_composed_layout(make_swizzle(2, 3, 3), 0, Layout::row_major((8, 20)));
        let smem = make_smem_tensor::<F16>(layout.clone());
        assert_eq!(cosize(&layout, &[]), IndexExpr::Static(160));
        assert_eq!(cosize(&smem, &[]), IndexExpr::Static(160));
        assert_eq!(cosize(&smem.slice((0, ..)), &[]), IndexExpr::Static(20));
        assert_eq!(cosize(&smem, &[0]), IndexExpr::Static(141));
    });
}

/// An identity tensor's codomain is coordinates, so its cosize is CuTe's
/// `size(coshape)`: the product, over the coordinate's modes, of one past
/// the largest coordinate. A tile, partition or slice counts only the
/// coordinates it spans, wherever they start.
#[test]
fn cosize_of_identity_tensors() {
    let st = IndexExpr::Static;
    Kernel::build("identity_cosize", || {
        let id = make_identity_tensor((8, 8));
        assert_eq!((cosize(&id, &[]), cosize(&id, &[0]), cosize(&id, &[1])), (st(64), st(8), st(8)));
        let id3 = make_identity_tensor((4, 6, 5));
        assert_eq!((cosize(&id3, &[]), cosize(&id3, &[1]), cosize(&id3, &[2])), (st(120), st(6), st(5)));
        assert_eq!(cosize(&make_identity_tensor(12), &[]), st(12));
        // A nested mode: ((2,3),4) holds the hierarchical coordinates
        // ((0..2, 0..3), 0..4), and its mode [0, 1] the coordinates
        // ((0, 0..3), 0), so a coshape of ((1,3),1).
        let nested = make_identity_tensor(((2, 3), 4));
        assert_eq!((cosize(&nested, &[]), cosize(&nested, &[0, 1])), (st(24), st(3)));
        // Each thread's 4 coordinates (r, c..c+4) of a 128-bit tiled copy.
        let atom = make_copy_atom::<i32>(CopyOp::Universal { bits: 128 });
        let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((8, 2)), &Layout::row_major((1, 4)));
        for thread in [0usize, 7, 15] {
            assert_eq!(cosize(&tiled.get_slice(thread).partition_s(&id), &[]), st(4), "thread {thread}");
        }
        assert_eq!(cosize(&tiled.get_slice(thread_idx().x).partition_s(&id), &[]), st(4));
        // A tile, a row, and the tiles of a partial last tile, which hold
        // coordinates past the tensor's (16 x 24 of them).
        let tile = local_tile(&make_identity_tensor((64, 64)), (8, 4), (5, 3));
        assert_eq!((cosize(&tile, &[]), cosize(&id.slice((3, ..)), &[])), (st(32), st(8)));
        let tiles = zipped_divide(&make_identity_tensor((13, 20)), (8, 8));
        assert_eq!((cosize(&tiles, &[]), cosize(&tiles, &[1])), (st(16 * 24), st(9 * 17)));
        // A tile of a tensor of runtime extents has a static shape, which an
        // identity tensor (or a layout) takes as it is.
        let x = make_tensor(&param_ptr::<f32>("x"), Layout::row_major((param_extent("m", 1), 32)));
        let tile = local_tile(&x, (8, 4), (0, 1));
        let coords = make_identity_tensor(tile.shape());
        assert_eq!((coords.shape(), cosize(&coords, &[])), (tile.shape(), st(32)));
        assert_eq!(Layout::col_major(tile.shape()), Layout::col_major((8, 4)));
    });
}

/// A tensor parameter whose cosize is only known at launch (runtime extents
/// or strides) is a buffer sized at launch, as `param_ptr` declares one; a
/// layout whose cosize is static, even with a runtime extent of stride 0,
/// still sizes its buffer.
#[test]
fn tensor_parameters_of_runtime_extents_are_sized_at_launch() {
    let kernel = Kernel::build("launch_sized", || {
        let (m, _) = extent("m", 1);
        let k = param_extent("k", 1);
        let x = param_tensor::<f32>("x", Layout::row_major((m.clone(), 8)));
        assert_eq!(x.layout().to_string(), "(?,8):(8,1)");
        let _ = param_tensor::<f32>("y", Layout::col_major((4, 8)));
        let _ = param_tensor::<f32>("z", Layout::with_strides((4, 8), (k, 1)));
        let _ = param_tensor::<f32>("w", Layout::new(m, 0));
    });
    let lens: Vec<_> = kernel.ir().buffers.iter().map(|b| (b.name.as_str(), b.len)).collect();
    assert_eq!(lens, [("x", None), ("y", Some(32)), ("z", None), ("w", Some(1))]);
}

/// `runtime_gemm` with `M`, `N` and `K` as the runtime extents of its
/// global tensors (`build_dynamic`), rather than static upper bounds with
/// runtime strides (`build`): one built kernel at several problem sizes,
/// correct on random small integers, and bit for bit the static-bound
/// kernel's result on random values whose sums round. The launch checks the
/// tile multiples its alignment proofs rely on, now `m`'s too.
#[test]
fn runtime_extents_gemm_matches_static_bounds() {
    let (bounded, dynamic) = (runtime_gemm::build(), runtime_gemm::build_dynamic());
    for (m, n, k) in [(128, 128, 32), (256, 128, 96), (128, 256, 160), (384, 128, 64)] {
        let (launch, mut args, a, b) = runtime_gemm::problem(m, n, k);
        bounded.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
        runtime_gemm::check(m, n, k, &a, &b, args[2].as_f32());
        let (report, c) = runtime_gemm::run_dynamic(&dynamic, m, n, k);
        assert_eq!(report.mma_instructions, (m / 128 * (n / 128) * 4 * 32 * (k / 16)) as u64, "{m}x{n}x{k}");
        let same = c.iter().zip(args[2].as_f32()).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same, "{m}x{n}x{k}: the runtime-extent kernel's result differs from the static-bound kernel's");
    }
    for (m, n, k) in [(128, 128, 64), (256, 128, 160)] {
        let (a, b) = (runtime_gemm::random_values(m * k, 3), runtime_gemm::random_values(n * k, 4));
        // These sums round: adding the products in the opposite order
        // changes some of them.
        let reordered = (0..m * n).filter(|&e| {
            let (i, j) = (e / n, e % n);
            let products: Vec<f32> = (0..k).map(|l| a[i * k + l] * b[j * k + l]).collect();
            products.iter().sum::<f32>().to_bits() != products.iter().rev().sum::<f32>().to_bits()
        });
        assert!(reordered.count() * 2 > m * n, "{m}x{n}x{k}: the random values do not round");
        let (launch, mut args) = runtime_gemm::problem_on(m, n, k, &a, &b);
        let mut dynamic_args = runtime_gemm::with_m(m, args.clone());
        bounded.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
        dynamic.simulate(launch, &mut dynamic_args).unwrap_or_else(|e| panic!("{e}"));
        let (c, dynamic_c) = (args[2].as_f32(), dynamic_args[2].as_f32());
        assert!(c.iter().any(|&x| x != 0.0), "{m}x{n}x{k}: no result");
        let same = c.iter().zip(dynamic_c).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same, "{m}x{n}x{k}: the kernels' results differ on random values");
    }
    let (launch, mut args, ..) = runtime_gemm::problem_dynamic(128, 128, 32);
    args[3] = KernelArg::ScalarI32(100);
    let e = dynamic.simulate(launch, &mut args).unwrap_err();
    assert!(e.message.contains("must be a multiple of 128"), "{e}");
}

/// Coordinate tensors: identity tensors of ranks 1 to 5, nested and with
/// runtime extents, hold their own coordinates; their tiles, partitions and
/// slices hold the coordinates that `idx2crd` gives for the offsets of the
/// same tiles, partitions and slices of a compact memory tensor; and
/// `domain_offset` shifts memory and coordinate tensors, by negative
/// runtime amounts too.
#[test]
fn coordinate_tensors() {
    Kernel::build("coordinate_tensors", || {
        let (m, mid) = extent("m", 1);
        let (n, nid) = extent("n", 1);
        let r = IndexExpr::signed(param_scalar::<i32>("r"));
        let IndexExpr::Signed(rid) = r else { panic!("a signed runtime value") };
        let env = |v: ValueId| -> usize {
            if v == mid {
                16
            } else if v == nid {
                24
            } else {
                panic!("no value for {v:?}")
            }
        };
        let eval = |c: &Coord| -> IntTuple { c.map(&mut |e| e.eval(&env)) };
        // Identity tensors hold their coordinates, congruent with the shape.
        let shapes: [Shape; 6] = [
            Shape::from(7),
            Shape::from((3, m.clone())),
            Shape::from(((2, 1), 3)),
            Shape::from((2, (3, n.clone()), 4)),
            Shape::from((2, 3, 2, m.clone(), 2)),
            Shape::from(((2, m.clone()), (3, (2, n.clone())))),
        ];
        for shape in shapes {
            let id = make_identity_tensor(shape.clone());
            let static_shape: IntTuple = shape.map(&mut |e| e.eval(&env));
            for i in 0..static_shape.size() {
                let c = id.coord(i);
                assert!(c.congruent(&shape), "{shape} at {i}: {c}");
                let want: IntTuple = idx2crd(i, &static_shape).map(&mut |e| e.as_static().unwrap());
                assert_eq!(eval(&c), want, "{shape} at {i}");
            }
        }
        let first: Vec<String> = (0..3).map(|i| make_identity_tensor(((2, 1), 3)).coord(i).to_string()).collect();
        assert_eq!(first, ["((0,0),0)", "((1,0),0)", "((0,0),1)"]);

        // Tiles, partitions and slices, against a compact memory tensor of
        // the same runtime extents (16 x 24, which the tiles divide).
        let id = make_identity_tensor((m.clone(), n.clone()));
        let x = make_tensor(&param_ptr::<f32>("x"), Layout::col_major((m.clone(), n.clone())));
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
        let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((2, 4)), &Layout::row_major((2, 2)));
        type View = fn(&Tensor<Ident>, &Tensor<f32>, &TiledCopy<f32>) -> (Tensor<Ident>, Tensor<f32>);
        let views: [(&str, usize, View); 5] = [
            ("local_tile", 32, |c, x, _| (local_tile(c, (4, 8), (1, 2)), local_tile(x, (4, 8), (1, 2)))),
            ("local_partition", 4, |c, x, _| {
                let thr = Layout::row_major((2, 4));
                (
                    local_partition(&local_tile(c, (4, 8), (3, 1)), &thr, 5),
                    local_partition(&local_tile(x, (4, 8), (3, 1)), &thr, 5),
                )
            }),
            ("partition_s", 4, |c, x, t| {
                let thr = t.get_slice(3);
                (thr.partition_s(&local_tile(c, (4, 8), (2, 0))), thr.partition_s(&local_tile(x, (4, 8), (2, 0))))
            }),
            ("zipped_divide", 384, |c, x, _| (zipped_divide(c, (4, 8)), zipped_divide(x, (4, 8)))),
            ("slice", 16, |c, x, _| (c.slice((.., 5)), x.slice((.., 5)))),
        ];
        for (what, size, view) in views {
            let (c, t) = view(&id, &x, &tiled);
            for i in 0..size {
                let crd = idx2crd(i, &c.shape().map(&mut |e| e.eval(&env)));
                let offset = (t.base_offset().clone() + t.layout().offset(&crd)).eval(&env);
                let want: IntTuple = idx2crd(offset, &tup((16, 24))).map(&mut |e| e.as_static().unwrap());
                assert_eq!(eval(&c.coord(crd.clone())), want, "{what} at {i}: {crd}");
            }
        }

        // domain_offset: a memory tensor's base offset grows by the offset
        // of the shift, a coordinate tensor's origin by the shift.
        let shifted = domain_offset((2, 3), &x);
        assert_eq!((shifted.layout(), shifted.base_offset().eval(&env)), (x.layout(), 2 + 3 * 16));
        assert_eq!(eval(&domain_offset((2, 3), &id).coord((1, 1))), tup((3, 4)));
        let signed_env = |v: ValueId| -> i64 { if v == rid { -3 } else { env(v) as i64 } };
        let back = domain_offset((0, r.clone()), &x);
        assert_eq!(back.base_offset().eval_signed(&signed_env), -48);
        let c = domain_offset((1, r), &id).coord((2, 3));
        assert_eq!(c.flat_leaves().iter().map(|e| e.eval_signed(&signed_env)).collect::<Vec<_>>(), [3, 0]);
    });
}

fn tup(t: impl Into<IntTuple>) -> IntTuple {
    t.into()
}

/// elem_less compares congruent tuples leaf by leaf, folding what is known
/// when the kernel is built: the CuTe dialect's examples, nested tuples, a
/// negative constant against a non-negative runtime value, and runtime
/// extents.
#[test]
fn elem_less_compares_hierarchically() {
    Kernel::build("elem_less_cases", || {
        let folded = |b: Boolean| b.as_constant();
        // dialect: elem_less of int tuples, shapes, and a coordinate against
        // a shape.
        assert_eq!(folded(elem_less((3, 4), (5, 6))), Some(true));
        assert_eq!(folded(elem_less(Shape::from((3, 4)), Shape::from((5, 6)))), Some(true));
        assert_eq!(folded(elem_less(Coord::from((3, 4)), Shape::from((5, 6)))), Some(true));
        assert_eq!(folded(elem_less((3, 6), (5, 6))), Some(false));
        assert_eq!(folded(elem_less(((1, 2), 3), ((2, 3), 4))), Some(true));
        assert_eq!(folded(elem_less(((1, 3), 3), ((2, 3), 4))), Some(false));
        // -1 is below every non-negative value, and a runtime extent is
        // compared at run time.
        let (m, _) = extent("m", 1);
        assert_eq!(folded(elem_less((0, -1), (5, m.clone()))), Some(true));
        let t = thread_idx().x;
        assert_eq!(folded(elem_less((t, 2), (m.clone(), 4))), None);
        // A coordinate tensor shifted by an Int32 K residue, passed as it
        // is rather than through IndexExpr::signed, may hold negative
        // coordinates: the residue predicate is computed at run time, and
        // signed, instead of folding to true.
        let r = param_scalar::<i32>("r");
        let c = domain_offset((0, r), &make_identity_tensor((8, 8))).coord((1, 2));
        let k = c.flat_leaves()[1].clone();
        assert!(!k.is_non_negative(), "{k:?}");
        assert_eq!(folded(elem_less((0, -1), (m, k))), None);
    });
}
