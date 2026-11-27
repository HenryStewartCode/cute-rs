//! Layout algebra tests: worked examples from the CuTe documentation plus
//! brute-force property checks on generated layouts. Test vectors taken
//! from CUTLASS 4.8.1 say where they come from: "layout algebra doc" marks
//! the worked examples of media/docs/cpp/cute/02_layout_algebra.md,
//! "notebook" the results printed in
//! examples/python/CuTeDSL/cute/notebooks/cute_layout_algebra.ipynb, and
//! "dialect" and "cutegen" the vectors of
//! media/docs/cutlass_compiler/cute_dialect.rst and
//! cutlass_compiler/cutegen/test. THIRD-PARTY-NOTICES.md lists those
//! sources.

use super::*;

fn l(shape: impl Into<IntTuple>, stride: impl Into<IntTuple>) -> Layout {
    let (shape, stride): (IntTuple, IntTuple) = (shape.into(), stride.into());
    Layout::new(shape, stride)
}

#[test]
fn display_and_size() {
    // The layout of the dialect's nested cute.slice and cute.dice examples.
    let a = l(((2, 4), 3), ((1, 2), 8));
    assert_eq!(a.to_string(), "((2,4),3):((1,2),8)");
    assert_eq!(a.size(), 24);
    assert_eq!(a.cosize(), 1 + 1 + 6 + 16);
    assert_eq!(Layout::row_major((4, 8)).to_string(), "(4,8):(8,1)");
    assert_eq!(Layout::col_major((4, 8)).to_string(), "(4,8):(1,4)");
}

#[test]
fn coordinates() {
    // layout algebra doc: (B, B*) of its 1-D logical divide example.
    let a = l((4, (2, 3)), (2, (1, 8)));
    assert_eq!(a.crd2idx(&(1, (1, 2)).into()), 2 + 1 + 16);
    // 1-D index 13: colexicographic (1, 3) -> (1, (1, 1)).
    assert_eq!(a.idx2crd(13).to_string(), "(1,(1,1))");
    assert_eq!(a.call(13), 2 + 1 + 8);
    // A 1-D index into a hierarchical mode.
    assert_eq!(a.crd2idx(&(1, 3).into()), 2 + 1 + 8);
}

#[test]
fn slicing() {
    // dialect: its nested cute.slice examples on this layout, with other
    // coordinates.
    let a = l(((2, 4), 3), ((1, 2), 8));
    let (s, off) = a.slice(&(.., 2).into());
    assert_eq!(s.to_string(), "((2,4)):((1,2))");
    assert_eq!(off, IndexExpr::Static(16));
    let (s, off) = a.slice(&((1, ..), ..).into());
    assert_eq!(s.to_string(), "(4,3):(2,8)");
    assert_eq!(off, IndexExpr::Static(1));
}

#[test]
fn coalesce_examples() {
    // layout algebra doc.
    assert_eq!(l((2, (1, 6)), (1, (6, 2))).coalesce().to_string(), "12:1");
    // Modes that do not merge, size-1 modes only, and a profile.
    assert_eq!(l((4, 2), (1, 8)).coalesce().to_string(), "(4,2):(1,8)");
    assert_eq!(l((1, 1), (3, 5)).coalesce().to_string(), "1:0");
    let profile: IntTuple = (1, 1).into();
    assert_eq!(l(((2, 2), (3, 4)), ((1, 2), (4, 12))).coalesce_profile(&profile).to_string(), "(4,12):(1,4)");
}

#[test]
fn composition_examples() {
    // layout algebra doc: its three composition examples.
    assert_eq!(composition(&l((6, 2), (8, 2)), &l((4, 3), (3, 1))).to_string(), "((2,2),3):((24,2),8)");
    assert_eq!(composition(&l(20, 2), &l((5, 4), (4, 1))).to_string(), "(5,4):(8,2)");
    assert_eq!(composition(&l((10, 2), (16, 4)), &l((5, 4), (1, 5))).to_string(), "(5,(2,2)):(16,(80,4))");
}

/// A mode with at least as many elements as are left to place holds them,
/// whether or not they divide it, as in C++ CuTe's `composition_impl`, which
/// keeps `min(next_shape, rest_shape)`. A stride must still divide the
/// extents it steps over or lands in.
#[test]
fn composition_where_a_mode_holds_the_rest() {
    let a = l((6, 8), (8, 1));
    let cases = [
        // C++ CuTe and the CuTe dialect: (6,8):(8,1) ∘ 4:1 is 4:8.
        (a.clone(), l(4, 1), l(4, 8)),
        // Stride 2 leaves 3 elements of the 6, and 2 of them fit.
        (a.clone(), l(2, 2), l(2, 16)),
        // Each mode of a tuple on its own: 4:1 is held, 8:6 steps over the 6.
        (a.clone(), l((4, 8), (1, 6)), l((4, 8), (8, 1))),
        // The whole first mode, then part of the next (C++ CuTe: (2,4):(100,1)).
        (l((2, 6, 5), (100, 1, 10)), l(8, 1), l((2, 4), (100, 1))),
    ];
    for (a, b, c) in cases {
        assert_eq!(composition(&a, &b), c, "({a}) o ({b})");
        for i in 0..b.size() {
            assert_eq!(c.call(i), a.call(b.call(i)), "({a}) o ({b}) at {i}");
        }
    }
    // Refused, as by C++ CuTe: stride 2 leaves 3 elements of the 6, and 4 is
    // neither at most 3 nor a multiple of it; after the whole 2, the 4
    // elements left neither fit in the 3 nor are a multiple of it.
    assert!(std::panic::catch_unwind(|| composition(&a, &l(4, 2))).is_err());
    assert!(std::panic::catch_unwind(|| composition(&l((2, 3, 5), (1, 100, 1000)), &l(8, 1))).is_err());
    // Refused, although C++ CuTe's weak divisibility condition takes it:
    // dividing by 4 composes with (4,12):(1,4), whose stride 4 lands inside
    // the 6 without dividing it. For the 12:4 part C++ CuTe gives
    // (2,6):(32,1), which maps index 2 to 1, where the composition maps it
    // to a(8) = 17.
    assert!(std::panic::catch_unwind(|| logical_divide(&a, 4)).is_err());
    assert_eq!(a.call(8), 17);
}

#[test]
fn complement_examples() {
    // layout algebra doc, except the cotarget 72.
    assert_eq!(complement(&l(4, 1), 24).to_string(), "6:4");
    assert_eq!(complement(&l(6, 4), 24).to_string(), "4:1");
    assert_eq!(complement(&l((4, 6), (1, 4)), 24).size(), 1);
    assert_eq!(complement(&l((4, 6), (1, 4)), 72).to_string(), "3:24");
    assert_eq!(complement(&l((2, 2), (1, 6)), 24).to_string(), "(3,2):(2,12)");
}

#[test]
fn divide_examples() {
    // layout algebra doc: the 1-D example.
    let a = l((4, 2, 3), (2, 1, 8));
    assert_eq!(logical_divide(&a, l(4, 2)).to_string(), "((2,2),(2,3)):((4,1),(2,8))");

    // The 2-D example: the layout and tiler of the layout algebra doc, with
    // the notebook's results.
    let a = l((9, (4, 8)), (59, (13, 1)));
    let tiler: Tiler = (l(3, 3), l((2, 4), (1, 8))).into();
    assert_eq!(logical_divide(&a, &tiler).to_string(), "((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))");
    assert_eq!(zipped_divide(&a, &tiler).to_string(), "((3,(2,4)),(3,(2,2))):((177,(13,2)),(59,(26,1)))");
    assert_eq!(tiled_divide(&a, &tiler).to_string(), "((3,(2,4)),3,(2,2)):((177,(13,2)),59,(26,1))");
    assert_eq!(flat_divide(&a, &tiler).to_string(), "(3,(2,4),3,(2,2)):(177,(13,2),59,(26,1))");

    // Tiling a row-major 8x8 matrix into 4x2 tiles with integer tilers.
    let m = Layout::row_major((8, 8));
    assert_eq!(zipped_divide(&m, (4, 2)).to_string(), "((4,2),(2,4)):((8,1),(32,2))");
}

#[test]
fn product_examples() {
    // layout algebra doc: the 1-D example.
    assert_eq!(logical_product(&l((2, 2), (4, 1)), l(6, 1)).to_string(), "((2,2),(2,3)):((4,1),(2,8))");
    // The 2-D example: the layout algebra doc's 2x5 row-major block over a
    // 3x4 column-major tiler, with the notebook's results.
    let block = l((2, 5), (5, 1));
    let tiler = l((3, 4), (1, 3));
    assert_eq!(blocked_product(&block, &tiler).to_string(), "((2,3),(5,4)):((5,10),(1,30))");
    assert_eq!(raked_product(&block, &tiler).to_string(), "((3,2),(4,5)):((10,5),(30,1))");
}

#[test]
fn tv_layout_example() {
    // The thread and value layouts of the CuTe DSL example elementwise_add.py
    // (examples/python/CuTeDSL/cute/ampere/kernel/elementwise): 4x32
    // threads, 4x4 values each.
    let thr = Layout::row_major((4, 32));
    let val = Layout::row_major((4, 4));
    let (tiler, tv) = make_layout_tv(&thr, &val);
    assert_eq!(tiler.to_string(), "(16,128)");
    assert_eq!(tv.to_string(), "((32,4),(4,4)):((64,4),(16,1))");
}

#[test]
fn tile_to_shape_example() {
    let atom = l((8, 8), (8, 1));
    let t = tile_to_shape(&atom, (16, 32));
    assert_eq!(t.shape().product_each().to_string(), "(16,32)");
    // Each 8x8 block stays row-major and blocks are laid out column-major.
    assert_eq!(t.crd2idx(&(1, 0).into()), 8);
    assert_eq!(t.crd2idx(&(8, 0).into()), 64);
    assert_eq!(t.crd2idx(&(0, 8).into()), 128);
}

#[test]
fn coord_of_inverts_thread_layouts() {
    let thr = Layout::row_major((16, 16));
    for t in 0..256 {
        let c = thr.coord_of(t);
        let crd: Vec<usize> = c.flat_leaves().iter().map(|e| e.as_static().unwrap()).collect();
        assert_eq!(thr.crd2idx(&crd.into()), t);
    }
}

/// A small deterministic generator, so property tests need no dependencies.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next() % xs.len() as u64) as usize]
    }

    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            xs.swap(i, j);
        }
    }
}

/// A random bijection onto `[0, size)`: a compact layout with its modes
/// permuted.
fn random_bijection(rng: &mut Rng) -> Layout {
    let rank = 1 + (rng.next() % 4) as usize;
    let shapes: Vec<usize> = (0..rank).map(|_| rng.pick(&[1, 2, 3, 4, 8])).collect();
    let mut order: Vec<usize> = (0..rank).collect();
    rng.shuffle(&mut order);
    let mut strides = vec![0; rank];
    let mut running = 1;
    for &i in &order {
        strides[i] = running;
        running *= shapes[i];
    }
    Layout::new(shapes, strides)
}

fn random_layout(rng: &mut Rng) -> Layout {
    let rank = 1 + (rng.next() % 3) as usize;
    let shapes: Vec<usize> = (0..rank).map(|_| rng.pick(&[1, 2, 3, 4, 6, 8])).collect();
    let strides: Vec<usize> = (0..rank).map(|_| rng.pick(&[0, 1, 2, 3, 4, 8, 16, 24])).collect();
    Layout::new(shapes, strides)
}

#[test]
fn property_coalesce_preserves_function() {
    let mut rng = Rng(0x1234_5678);
    for _ in 0..500 {
        let a = random_layout(&mut rng);
        let c = a.coalesce();
        assert_eq!(a.size(), c.size());
        for i in 0..a.size() {
            assert_eq!(a.call(i), c.call(i), "coalesce({a}) = {c} differs at {i}");
        }
    }
}

/// An injective layout whose modes do not overlap (a subset of a permuted
/// compact layout), possibly with broadcast (stride-0) modes mixed in. Such
/// layouts are the ones by-mode composition is pointwise-correct for.
fn random_injective(rng: &mut Rng) -> Layout {
    let full = random_bijection(rng);
    let mut modes: Vec<Layout> = full.modes().into_iter().filter(|_| !rng.next().is_multiple_of(3)).collect();
    if modes.is_empty() {
        modes.push(full.mode(0));
    }
    if rng.next().is_multiple_of(4) {
        let at = (rng.next() as usize) % (modes.len() + 1);
        modes.insert(at, Layout::new(rng.pick(&[2, 3]), 0));
    }
    Layout::from_modes(modes)
}

#[test]
fn property_composition_is_function_composition() {
    let mut rng = Rng(0xdead_beef);
    let mut checked = 0;
    for _ in 0..3000 {
        let a = random_bijection(&mut rng);
        let b = random_injective(&mut rng);
        // Composition is defined on the domain of `a`.
        if b.cosize() > a.size() {
            continue;
        }
        let Ok(c) = std::panic::catch_unwind(|| composition(&a, &b)) else { continue };
        assert_eq!(c.size(), b.size());
        for i in 0..b.size() {
            assert_eq!(c.call(i), a.call(b.call(i)), "({a}) o ({b}) = {c} differs at {i}");
        }
        checked += 1;
    }
    assert!(checked > 200, "too few compositions were defined ({checked})");
}

/// A random static layout of at most `max_size` elements: flat or nested,
/// with extents that often do not divide one another and strides with gaps,
/// zeros and overlaps.
fn random_static_layout(rng: &mut Rng, max_size: usize, extents: &[usize], strides: &[usize]) -> Layout {
    let shape = loop {
        let t = random_tree(rng, 2, 1, extents);
        if t.size() <= max_size {
            break t;
        }
    };
    let stride = shape.map(&mut |_| rng.pick(strides));
    Layout::new(shape, stride)
}

/// `c` is `a ∘ b`. For a single mode `b`, `c(i) == a(b(i))` for every `i`
/// in `b`'s domain. CuTe composes with a tuple mode by mode, so `c` then has
/// `b`'s rank and each of its modes is `a` composed with that mode of `b`.
/// (That makes `c(i) == a(b(i))` on the whole of a tuple `b` only when
/// `b`'s modes do not interfere through `a`, as for the injective `b` and
/// bijective `a` of `property_composition_is_function_composition`:
/// `(8,6):(1,6) ∘ (4,8):(1,1)` is `(4,8):(1,1)` in C++ CuTe and cute-rs
/// alike, which maps index 23 to 8 where `a(b(23))` is `a(8) = 6`.)
#[track_caller]
fn assert_composition(a: &Layout, b: &Layout, c: &Layout) {
    assert_eq!(c.size(), b.size(), "({a}) o ({b}) = {c}");
    if b.shape().is_leaf() {
        for i in 0..b.size() {
            assert_eq!(c.call(i), a.call(b.call(i)), "({a}) o ({b}) = {c} differs at {i}");
        }
    } else {
        assert_eq!(c.rank(), b.rank(), "({a}) o ({b}) = {c}");
        for m in 0..b.rank() {
            assert_composition(a, &b.mode(m), &c.mode(m));
        }
    }
}

/// Every static composition cute-rs accepts is function composition (mode
/// by mode, as CuTe defines it for a tuple), for any static left-hand side,
/// not just bijections, and right-hand sides whose offsets stay in its
/// domain. The second part composes with `s:r` where `r` divides the extent
/// `t` of the first mode of the coalesced left-hand side and `s` is smaller
/// than `t / r` without dividing it: the mode holds the `s` elements (C++
/// CuTe's `min`), which cute-rs used to refuse.
#[test]
fn property_static_composition_is_function_composition() {
    let mut rng = Rng(0x5eed_c0de);
    let (mut checked, mut single, mut refused) = (0, 0, 0);
    for _ in 0..6000 {
        let a = random_static_layout(&mut rng, 1024, &[1, 2, 3, 4, 5, 6, 8, 12], &[0, 1, 1, 2, 3, 4, 6, 8, 12, 24, 96]);
        let b = random_static_layout(&mut rng, 256, &[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 1, 1, 2, 3, 4, 6, 8, 12]);
        if b.cosize() > a.size() {
            continue;
        }
        match std::panic::catch_unwind(|| composition(&a, &b)) {
            Ok(c) => {
                assert_composition(&a, &b, &c);
                checked += 1;
                single += usize::from(b.shape().is_leaf());
            }
            Err(_) => refused += 1,
        }
    }
    assert!(
        checked > 1000 && single > 300 && refused > 100,
        "{checked} compositions checked ({single} with a single mode), {refused} refused"
    );

    let mut held = 0;
    for _ in 0..4000 {
        let a = random_static_layout(&mut rng, 1024, &[2, 3, 4, 5, 6, 8, 12], &[0, 1, 2, 3, 5, 7, 12]);
        let flat = a.coalesce().flat_modes();
        if flat.len() < 2 {
            continue;
        }
        let (t, d0) = flat[0];
        let r = rng.pick(&(1..t).filter(|r| t.is_multiple_of(*r)).collect::<Vec<_>>());
        let sizes: Vec<usize> = (2..t / r).filter(|s| !(t / r).is_multiple_of(*s)).collect();
        if sizes.is_empty() {
            continue;
        }
        let b = Layout::new(rng.pick(&sizes), r);
        let c = composition(&a, &b);
        assert_eq!(c, Layout::new(b.size(), d0 * r), "({a}) o ({b})");
        assert_composition(&a, &b, &c);
        held += 1;
    }
    assert!(held > 300, "too few compositions held the rest in a mode ({held})");
}

#[test]
fn property_complement_completes_a_bijection() {
    let mut rng = Rng(42);
    for _ in 0..500 {
        let full = random_bijection(&mut rng);
        let keep: Vec<usize> = (0..full.rank()).filter(|_| rng.next().is_multiple_of(2)).collect();
        let a = if keep.is_empty() { full.mode(0) } else { full.select(&keep) };
        let c = complement(&a, full.size());
        let joint = Layout::from_modes([a.clone(), c.clone()]);
        let mut seen = vec![false; full.size()];
        assert_eq!(joint.size(), full.size(), "({a}, {c})");
        for i in 0..joint.size() {
            let o = joint.call(i);
            assert!(o < full.size() && !seen[o], "({a}, {c}) is not a bijection");
            seen[o] = true;
        }
    }
}

#[test]
fn property_inverses() {
    let mut rng = Rng(7);
    for _ in 0..500 {
        let a = random_bijection(&mut rng);
        let r = right_inverse(&a);
        assert_eq!(r.size(), a.size(), "right_inverse({a}) = {r}");
        for i in 0..r.size() {
            assert_eq!(a.call(r.call(i)), i);
        }
        let li = left_inverse(&a);
        for i in 0..a.size() {
            assert_eq!(li.call(a.call(i)), i);
        }
    }
}

#[test]
fn property_tv_layouts_are_bijections_onto_the_tile() {
    let mut rng = Rng(99);
    for _ in 0..200 {
        let thr = Layout::row_major((rng.pick(&[1, 2, 4, 8]), rng.pick(&[1, 2, 4])));
        let val = if rng.next().is_multiple_of(2) {
            Layout::row_major((rng.pick(&[1, 2, 4]), rng.pick(&[1, 2, 4])))
        } else {
            Layout::col_major((2, 2))
        };
        let (tiler, tv) = make_layout_tv(&thr, &val);
        assert_eq!(tiler.size(), thr.size() * val.size());
        let mut seen = vec![false; tiler.size()];
        for t in 0..thr.size() {
            for v in 0..val.size() {
                let o = tv.crd2idx(&(t, v).into());
                assert!(!seen[o]);
                seen[o] = true;
            }
        }
    }
}

#[test]
fn swizzled_expr_matches_static() {
    let sw = make_swizzle(2, 3, 3);
    for i in 0..256 {
        assert_eq!(sw.apply_expr(IndexExpr::Static(i)).as_static(), Some(sw.apply(i)));
    }
}

#[test]
fn max_common_vector_and_layout() {
    let col = Layout::col_major((4, 8));
    let row = Layout::new((4, 8), (8, 1));
    assert_eq!(max_common_vector(&col, &col), 32);
    assert_eq!(max_common_vector(&col, &row), 1);
    let gapped = Layout::new((4, 8), (1, 8));
    assert_eq!(max_common_vector(&col, &gapped), 4);
    assert_eq!(max_common_layout(&col, &gapped), Layout::new(4, 1));
    assert_eq!(max_common_layout(&col, &row), Layout::new(1, 0));
}

#[test]
fn products_match_their_divides() {
    let (block, tiler) = (Layout::col_major((2, 4)), Layout::col_major((3, 5)));
    let zipped = zipped_product(&block, &tiler);
    assert_eq!(zipped.rank(), 2);
    assert_eq!(zipped.size(), block.size() * tiler.size());
    assert_eq!(tiled_product(&block, &tiler).rank(), 3);
    assert_eq!(flat_product(&block, &tiler).rank(), 4);
    // Every position is reached exactly once.
    let flat = flat_product(&block, &tiler);
    let mut seen = vec![false; flat.size()];
    (0..flat.size()).for_each(|i| seen[flat.call(i)] = true);
    assert!(seen.into_iter().all(|s| s));
}

#[test]
fn ordered_layouts() {
    assert_eq!(make_ordered_layout((2, 3), &[1, 0]), Layout::new((2, 3), (3, 1)));
    assert_eq!(make_layout_like(&Layout::new((2, 3), (30, 10))), Layout::new((2, 3), (3, 1)));
    assert_eq!(dice(&Layout::col_major((2, 3, 4)), &[true, false, true]).rank(), 2);
    assert!(is_congruent(&Layout::col_major((2, 3)), &Layout::row_major((5, 7))));
    assert_eq!(append_ones(&Layout::new(4, 1), 3).rank(), 3);
    assert_eq!(prepend_ones(&Layout::new((4, 2), (1, 4)), 3).mode(0).size(), 1);
    assert_eq!(round_up(9, 4), Shape::from(12));
}

#[test]
fn symbolic_strides_survive_the_algebra() {
    let k = crate::IndexExpr::Dynamic(crate::ir::ValueId(0));
    let a = Layout::with_strides((1024, 1024), (k.clone(), 1));
    let tile = a.zipped_divide(&(128, 32).into());
    let at = |m: usize, kk: usize| tile.offset(&((m, kk), (0, 0)).into());
    assert_eq!(at(3, 5), k.clone() * 3usize + 5usize);
    // The tile index moves by whole tiles along each mode.
    let rest = |b: usize, t: usize| tile.offset(&((0, 0), (b, t)).into());
    assert_eq!(rest(2, 1), k * 256usize + 32usize);
}

// ---------------------------------------------------------------------------
// Tuple helpers: parsing, wrap/unwrap, find, congruence, majorness,
// increment_coord, nullspace and tile_to_shape with an order. Vectors marked
// "dialect" come from media/docs/cutlass_compiler/cute_dialect.rst and those
// marked "cutegen" from cutlass_compiler/cutegen/test (CUTLASS 4.8.1).

use crate::ir::{Op, ValueId};
use crate::{Int32, Kernel, increment_coord, param_tensor, thread_idx};

fn lay(s: &str) -> Layout {
    s.parse().unwrap_or_else(|e| panic!("{e}"))
}

fn tup(s: &str) -> IntTuple {
    s.parse().unwrap_or_else(|e| panic!("{e}"))
}

/// CuTe gives the size-1 modes of compact layouts stride 0, and cute-rs's
/// compact strides do not; a size-1 mode's stride never matters. Zero them
/// on both sides before comparing with a layout CuTe printed.
fn unit_modes_zeroed(layout: &Layout) -> Layout {
    let shape = layout.static_shape();
    let shapes = shape.flat_leaves();
    let strides = layout.static_stride().flat_leaves();
    let zeroed: Vec<usize> = shapes.iter().zip(&strides).map(|(&s, &d)| if s == 1 { 0 } else { d }).collect();
    Layout::new(shape.clone(), Tree::unflatten(&zeroed, &shape))
}

#[track_caller]
fn assert_cute_eq(actual: &Layout, cute: &str) {
    assert_eq!(unit_modes_zeroed(actual), unit_modes_zeroed(&lay(cute)), "{actual} differs from CuTe's {cute}");
}

/// A random tree up to `depth` levels deep with leaves from `values`; tuples
/// have `min_rank..=3` modes.
fn random_tree(rng: &mut Rng, depth: usize, min_rank: usize, values: &[usize]) -> IntTuple {
    if depth == 0 || rng.next().is_multiple_of(3) {
        return Tree::Leaf(rng.pick(values));
    }
    let n = min_rank + (rng.next() % (4 - min_rank as u64)) as usize;
    Tree::Branch((0..n).map(|_| random_tree(rng, depth - 1, min_rank, values)).collect())
}

/// A random hierarchical layout with some zero and some unit strides.
fn random_nested_layout(rng: &mut Rng) -> Layout {
    let shape = random_tree(rng, 2, 1, &[1, 2, 3, 4]);
    let stride = shape.map(&mut |_| rng.pick(&[0, 0, 1, 1, 2, 3, 5, 8]));
    Layout::new(shape, stride)
}

/// Every leaf's path in depth-first order, as `Tree::find_if` reports them.
fn leaf_paths<T>(t: &Tree<T>) -> Vec<Vec<usize>> {
    fn walk<T>(t: &Tree<T>, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        match t {
            Tree::Leaf(_) => out.push(prefix.clone()),
            Tree::Branch(children) => {
                for (i, c) in children.iter().enumerate() {
                    prefix.push(i);
                    walk(c, prefix, out);
                    prefix.pop();
                }
            }
        }
    }
    if t.is_leaf() {
        return vec![vec![0]];
    }
    let mut out = Vec::new();
    walk(t, &mut Vec::new(), &mut out);
    out
}

#[test]
fn parse_tuples_and_layouts() {
    assert_eq!(tup("(2,(3,4))"), IntTuple::from((2, (3, 4))));
    assert_eq!(tup("8"), Tree::Leaf(8));
    assert_eq!(tup("(3)"), Tree::Branch(vec![Tree::Leaf(3)]));
    assert_eq!(tup("(3,)"), tup("(3)"));
    assert_eq!(tup("()"), Tree::Branch(vec![]));
    assert_eq!(tup(" ( _2 , ( 3 ,4 ) ) "), tup("(2,(3,4))"));
    assert_eq!(lay("(2,(3,4)):(1,(2,6))"), l((2, (3, 4)), (1, (2, 6))));
    assert_eq!(lay("8:1"), l(8, 1));
    assert_eq!(lay("(_64,_8):(_1,_64)"), l((64, 8), (1, 64)));
    assert_eq!(lay("():()").size(), 1);
}

#[test]
fn parse_errors_say_where() {
    let cases = [
        ("(2,3", 4, "expected `,` or `)`"),
        ("(2,,3)", 3, "expected a non-negative integer or `(`"),
        ("(-1)", 1, "expected a non-negative integer or `(`"),
        ("", 0, "unexpected end"),
        ("_", 1, "expected digits"),
        ("(2,3) x", 6, "unexpected text after the end"),
        ("99999999999999999999999", 19, "integer too large"),
    ];
    for (s, at, what) in cases {
        let e = s.parse::<IntTuple>().unwrap_err();
        assert_eq!(e.position(), at, "{s:?}: {e}");
        assert!(e.to_string().contains(what), "{s:?}: {e}");
    }
    let layouts = [
        ("(2,3)", 5, "expected `:`"),
        ("(2,3):(1)", 6, "shape (2,3) and stride (1) are not congruent"),
        ("(2,3):(1,2) x", 12, "unexpected text after the end"),
        ("(2,3):(1,2", 10, "expected `,` or `)`"),
    ];
    for (s, at, what) in layouts {
        let e = s.parse::<Layout>().unwrap_err();
        assert_eq!(e.position(), at, "{s:?}: {e}");
        assert!(e.to_string().contains(what), "{s:?}: {e}");
    }
}

#[test]
fn property_parse_round_trips() {
    let mut rng = Rng(0x9a75e);
    for _ in 0..500 {
        let t = random_tree(&mut rng, 3, 0, &[0, 1, 7, 64, 1 << 20]);
        assert_eq!(t.to_string().parse::<IntTuple>().as_ref(), Ok(&t), "{t}");
        let a = random_nested_layout(&mut rng);
        assert_eq!(a.to_string().parse::<Layout>().as_ref(), Ok(&a), "{a}");
    }
}

#[test]
fn wrap_and_unwrap() {
    // cutegen's wrap_if_leaf, which is CuTe DSL's wrap.
    assert_eq!(tup("3").wrap(), tup("(3)"));
    assert_eq!(tup("(3)").wrap(), tup("(3)"));
    assert_eq!(tup("(3,5)").wrap(), tup("(3,5)"));
    assert_eq!(tup("()").wrap(), tup("()"));
    // A 1-tuple of a leaf or of a tuple is stripped, a wider tuple kept, and
    // nested 1-tuples are stripped all the way.
    assert_eq!(tup("(5)").unwrap(), tup("5"));
    assert_eq!(tup("((4,6,8))").unwrap(), tup("(4,6,8)"));
    assert_eq!(tup("(4,6,8)").unwrap(), tup("(4,6,8)"));
    assert_eq!(tup("(((7)))").unwrap(), tup("7"));
    assert_eq!(tup("((1,(2)))").unwrap(), tup("(1,(2))"), "only the outer 1-tuples are stripped");
    assert_eq!(tup("()").unwrap(), tup("()"));
    // Any leaf type: coordinates, slice coordinates and tilers.
    let c: Coord = Tree::Branch(vec![Tree::Leaf(IndexExpr::Static(4))]);
    assert_eq!(c.unwrap(), Coord::from(4usize));
    let s: SliceCoord = (.., 2).into();
    assert_eq!(s.clone().wrap(), s);
    let tiler: Tiler = Tree::Branch(vec![Tree::Leaf(Some(l(4, 1)))]);
    assert_eq!(tiler.unwrap(), Tiler::from(l(4, 1)));
}

#[test]
fn find_and_find_if() {
    // The position of 7 in (5,7): a top-level mode.
    assert_eq!(tup("(5,7)").find_if(|&v, _| v == 7), Some(vec![1]));
    let t = tup("(2,(3,(4,5)),6)");
    assert_eq!(t.find(5), Some(vec![1, 1, 1]));
    assert_eq!(t.get(&[1, 1, 1]), &Tree::Leaf(5));
    assert_eq!(t.find_mode(5), Some(1), "hierarchical=False gives the top-level mode");
    assert_eq!(t.find(2), Some(vec![0]));
    assert_eq!(t.find(6), Some(vec![2]));
    assert_eq!(t.find(7), None);
    assert_eq!(t.find_mode(7), None);
    // Depth first: the 1 inside mode 0 comes before mode 1.
    assert_eq!(tup("((2,1),1)").find(1), Some(vec![0, 1]));
    // A leaf is its own mode 0; the empty tuple has nothing.
    assert_eq!(tup("7").find(7), Some(vec![0]));
    assert_eq!(tup("7").find_mode_if(|_, _| true), Some(0));
    assert_eq!(tup("()").find_if(|_, _| true), None);
    // The predicate sees each leaf's position.
    assert_eq!(t.find_if(|_, path| path.len() == 3), Some(vec![1, 1, 0]));
    assert_eq!(t.find_if(|&v, path| v > 2 && path[0] == 2), Some(vec![2]));
    // find skips runtime leaves and `..`.
    let k = IndexExpr::Dynamic(ValueId(1));
    let c: Coord = (k.clone(), 1, (k, 3)).into();
    assert_eq!(c.find(1), Some(vec![1]));
    assert_eq!(c.find(3), Some(vec![2, 1]));
    let s: SliceCoord = (.., 2, (.., 0)).into();
    assert_eq!(s.find(0), Some(vec![2, 1]));
    assert_eq!(s.find_mode(2), Some(1));
}

#[test]
fn property_find_is_the_first_matching_leaf() {
    let mut rng = Rng(0xf1d);
    for _ in 0..500 {
        let t = random_tree(&mut rng, 3, 0, &[1, 2, 3, 4, 5]);
        let paths = leaf_paths(&t);
        let mut visited = Vec::new();
        assert_eq!(
            t.find_if(|_, p| {
                visited.push(p.to_vec());
                false
            }),
            None
        );
        assert_eq!(visited, paths, "find_if visits the leaves of {t} depth first");
        for x in 0..7 {
            let want = paths.iter().find(|p| t.get(p) == &Tree::Leaf(x)).cloned();
            assert_eq!(t.find(x), want, "find({x}) in {t}");
            assert_eq!(t.find_mode(x), want.map(|p| p[0]));
        }
    }
}

#[test]
fn congruence() {
    // cutegen TupleTest.Congruence.
    let (t1, t2, t2a, t3, t3a) = (tup("(0)"), tup("0"), tup("7"), tup("(15,7)"), tup("(16,8)"));
    let t4 = tup("(3,7,(4,5,(1,2,(3,4))),9)");
    assert!(is_congruent(&t1, &t1));
    assert!(!is_congruent(&t1, &t2));
    assert!(is_congruent(&t2, &t2a));
    assert!(!is_congruent(&t2, &t3));
    assert!(is_congruent(&t3, &t3a));
    assert!(is_congruent(&t4, &t4.clone()));
    assert!(!is_congruent(&t3, &t4));
    // cutegen TupleTest.WeakCongruence: (a, b, a weakly congruent to b).
    let (a0, b0, a1, b1, a2, a3) = ("1", "2", "(3)", "(4)", "(5,6)", "(7,(8,9))");
    let b4 = "((10,11),(12,13))";
    let weak = [
        (a0, a0, true),
        (b0, b0, true),
        (a0, b0, true),
        (a0, a1, true),
        (b0, b1, true),
        (a0, b1, true),
        (b0, a1, true),
        (a1, a0, false),
        (b1, b0, false),
        (a1, b0, false),
        (b1, a0, false),
        (a1, a1, true),
        (b1, b1, true),
        (a1, b1, true),
        (a0, a2, true),
        (a1, a2, false),
        (a2, a2, true),
        (a0, a3, true),
        (a1, a3, false),
        (a2, a3, true),
        (a3, b1, false),
        (a3, a2, false),
        (a3, a3, true),
        (a0, b4, true),
        (a1, b4, false),
        (a2, b4, true),
        (a3, b4, true),
        ("(3,4)", "5", false),
        ("5", "(3,4)", true),
    ];
    for (a, b, want) in weak {
        assert_eq!(is_weakly_congruent(&tup(a), &tup(b)), want, "is_weakly_congruent({a}, {b})");
    }
    // Layouts, swizzled layouts, tensors and tuples of other leaf types are
    // compared by their shapes.
    let sw = make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((8, 64)));
    assert!(is_congruent(&Layout::col_major((2, (3, 4))), &tup("(5,(6,7))")));
    assert!(is_congruent(&sw, &l((1, 1), (0, 0))));
    assert!(is_weakly_congruent(&Layout::col_major(4), &Layout::col_major((2, 2))));
    assert!(!is_weakly_congruent(&Layout::col_major((2, 2)), &Layout::col_major(4)));
    assert!(is_weakly_congruent(&tup("(1,1)"), &sw));
    let c: Coord = (IndexExpr::Dynamic(ValueId(0)), (1, 2)).into();
    assert!(is_congruent(&c, &tup("(4,(2,3))")));
    assert!(!is_congruent(&c, &tup("(4,6)")));
    Kernel::build("congruent_tensor", || {
        let x = param_tensor::<f32>("x", Layout::col_major((4, (2, 2))));
        assert!(is_congruent(&x, &tup("(1,(1,1))")));
        assert!(is_weakly_congruent(&tup("(1,1)"), &x));
        assert!(!is_congruent(&x, &x.layout().flatten()));
    });
}

/// `is_congruent` used to take `&Layout`, so deref coercion let callers pass
/// a reference to a reference, a `&mut` or a smart pointer. Those calls still
/// compile and give the same answers, for `is_weakly_congruent` as well.
#[test]
fn congruence_through_references() {
    fn congruent(a: &Layout, b: &Layout) -> bool {
        is_congruent(&a, &b)
    }
    fn weakly_congruent(a: &Layout, b: &Layout) -> bool {
        is_weakly_congruent(&a, &b)
    }
    fn through_mut(a: &mut Layout, b: &mut Layout) -> (bool, bool) {
        (is_congruent(a, b), is_weakly_congruent(a, b))
    }
    let (a, b, c) = (lay("(2,3):(1,2)"), lay("(4,5):(5,1)"), lay("(2,(3,4)):(1,(2,6))"));
    assert!(congruent(&a, &b) && !congruent(&a, &c) && !congruent(&c, &a));
    assert!(weakly_congruent(&a, &c) && !weakly_congruent(&c, &a));
    assert!(is_congruent(&&&a, &&b) && !is_congruent(&&a, &&&c));
    assert!(is_weakly_congruent(&&&a, &&c) && !is_weakly_congruent(&&c, &&&a));
    assert_eq!(through_mut(&mut a.clone(), &mut c.clone()), (false, true));
    assert_eq!(through_mut(&mut c.clone(), &mut a.clone()), (false, false));
    // Smart pointers, which deref coercion also saw through.
    assert!(is_weakly_congruent(&Box::new(a.clone()), &std::rc::Rc::new(c.clone())));
    assert!(!is_weakly_congruent(&std::sync::Arc::new(c.clone()), &Box::new(&a)));
    let cow: std::borrow::Cow<'_, Layout> = std::borrow::Cow::Borrowed(&a);
    let cell = std::cell::RefCell::new(c.clone());
    assert!(is_weakly_congruent(&cow, &cell.borrow()));
    assert!(!is_congruent(&cell.borrow_mut(), &cow));
    // Unsized pointer types, which deref coercion saw through as well: a
    // `dyn Deref<Target = Layout>` behind a reference or in a box, and a
    // generic `P: Deref<Target = Layout> + ?Sized`.
    use std::ops::Deref;
    fn through_dyn(a: &dyn Deref<Target = Layout>, b: &dyn Deref<Target = Layout>) -> (bool, bool) {
        (is_congruent(a, b), is_weakly_congruent(a, b))
    }
    fn through_unsized<P: Deref<Target = Layout> + ?Sized>(a: &P, b: &P) -> (bool, bool) {
        (is_congruent(a, b), is_weakly_congruent(a, b))
    }
    let (rc_a, box_c) = (std::rc::Rc::new(a.clone()), Box::new(c.clone()));
    assert_eq!(through_dyn(&rc_a, &box_c), (false, true));
    assert_eq!(through_dyn(&box_c, &rc_a), (false, false));
    assert_eq!(through_dyn(&&b, &rc_a), (true, true));
    assert_eq!(through_unsized::<dyn Deref<Target = Layout>>(&rc_a, &box_c), (false, true));
    assert_eq!(through_unsized::<dyn Deref<Target = Layout>>(&box_c, &&a), (false, false));
    let boxed: [Box<dyn Deref<Target = Layout> + '_>; 3] = [Box::new(&b), Box::new(rc_a.clone()), Box::new(box_c)];
    assert!(is_congruent(&boxed[0], &boxed[1]) && !is_congruent(&boxed[1], &boxed[2]));
    assert!(is_weakly_congruent(&boxed[1], &boxed[2]) && !is_weakly_congruent(&boxed[2], &boxed[0]));
    // As function pointers of the old type.
    let f: fn(&Layout, &Layout) -> bool = is_congruent;
    let g: fn(&Layout, &Layout) -> bool = is_weakly_congruent;
    assert!(f(&a, &b) && !f(&a, &c) && g(&a, &c) && !g(&c, &a));
    // References to the other things with a shape.
    let sw = make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((8, 64)));
    assert!(is_congruent(&&tup("(1,1)"), &&sw) && !is_weakly_congruent(&&&c, &&sw));
    Kernel::build("congruent_through_references", || {
        let t = param_tensor::<f32>("t", Layout::col_major((2, (3, 4))));
        assert!(is_congruent(&t.layout(), &c) && !is_congruent(&t.layout(), &&a));
        assert!(is_weakly_congruent(&&a, &t.layout()) && !is_weakly_congruent(&&t, &a));
    });
}

/// Replace random sub-trees of `t` with leaves.
fn coarsen(rng: &mut Rng, t: &IntTuple) -> IntTuple {
    match t {
        Tree::Branch(_) if rng.next().is_multiple_of(4) => Tree::Leaf(1),
        Tree::Branch(cs) => Tree::Branch(cs.iter().map(|c| coarsen(rng, c)).collect()),
        leaf => leaf.clone(),
    }
}

/// Replace the first leaf of `t` with a tuple.
fn refine_first_leaf(t: &IntTuple) -> IntTuple {
    match t {
        Tree::Leaf(_) => tup("(2,2)"),
        Tree::Branch(cs) => {
            let mut cs = cs.clone();
            if let Some(i) = cs.iter().position(|c| !c.leaves().is_empty()) {
                cs[i] = refine_first_leaf(&cs[i]);
            }
            Tree::Branch(cs)
        }
    }
}

#[test]
fn property_weak_congruence_is_a_partial_order() {
    let mut rng = Rng(0xc0c0);
    for _ in 0..500 {
        let c = random_tree(&mut rng, 3, 0, &[1, 2]);
        let b = coarsen(&mut rng, &c);
        let a = coarsen(&mut rng, &b);
        assert!(is_weakly_congruent(&a, &b) && is_weakly_congruent(&b, &c) && is_weakly_congruent(&a, &c));
        assert!(is_weakly_congruent(&c, &c));
        // Antisymmetry: weakly congruent both ways means congruent.
        assert_eq!(is_weakly_congruent(&c, &b), is_congruent(&b, &c), "{b} vs {c}");
        let other = random_tree(&mut rng, 3, 0, &[1, 2]);
        assert_eq!(
            is_weakly_congruent(&other, &c) && is_weakly_congruent(&c, &other),
            is_congruent(&other, &c),
            "{other} vs {c}"
        );
        if !c.leaves().is_empty() {
            let finer = refine_first_leaf(&c);
            assert!(is_weakly_congruent(&c, &finer), "{c} vs {finer}");
            assert!(!is_weakly_congruent(&finer, &c), "{finer} vs {c}");
        }
    }
}

#[test]
fn majorness() {
    // The stride-1 mode of a row-major (5,3):(3,1).
    assert_eq!(leading_dim(&tup("(5,3)"), &tup("(3,1)")), Some(vec![1]));
    let col = Layout::col_major((4, 8));
    assert_eq!(col.leading_dim(), Some(vec![0]));
    assert!(col.is_major(0) && !col.is_major(1));
    let row = Layout::row_major((4, 8));
    assert_eq!(row.leading_dim(), Some(vec![1]));
    assert!(!row.is_major(0) && row.is_major(1));
    // A unit stride on a size-1 mode is not the leading dimension, so there
    // is none when every unit stride is on one, as there is none without a
    // unit stride.
    assert_eq!(lay("(1,8):(1,1)").leading_dim(), Some(vec![1]));
    assert_eq!(lay("(1,1):(1,1)").leading_dim(), None);
    assert_eq!(leading_dim(&tup("(1,1)"), &tup("(1,1)")), None);
    assert_eq!(lay("((1,1),4):((1,1),2)").leading_dim(), None);
    assert_eq!(lay("(4,8):(2,8)").leading_dim(), None);
    assert_eq!(lay("8:1").leading_dim(), Some(vec![0]));
    // Nested modes: the position is a path, and is_major looks at a mode's
    // first stride only.
    let nested = lay("((2,4),8):((8,1),16)");
    assert_eq!(nested.leading_dim(), Some(vec![0, 1]));
    assert!(!nested.is_major(0) && !nested.is_major(1));
    assert!(lay("((2,4),8):((1,2),16)").is_major(0));
    assert_eq!(tup("((3,4),5)").front(), &3);
    assert_eq!(tup("7").front(), &7);
    // Runtime strides are never major, even though they might be 1.
    let k = IndexExpr::Dynamic(ValueId(0));
    let a = Layout::with_strides((4, 8), (k.clone(), 1));
    assert_eq!(a.leading_dim(), Some(vec![1]));
    assert!(!a.is_major(0) && a.is_major(1));
    let b = Layout::with_strides((4, 8), (1, k.clone()));
    assert_eq!(b.leading_dim(), Some(vec![0]));
    assert!(b.is_major(0) && !b.is_major(1));
    assert_eq!(Layout::with_strides((4, 8), (k.clone(), k)).leading_dim(), None);
    assert!(is_major(1, &tup("(4,(1,2))")));
}

#[test]
fn property_leading_dim_is_the_first_unit_stride() {
    let mut rng = Rng(0x1ead);
    for _ in 0..500 {
        let a = random_nested_layout(&mut rng);
        let paths = leaf_paths(a.shape());
        let (shapes, strides) = (a.static_shape().flat_leaves(), a.static_stride().flat_leaves());
        let want = (0..paths.len()).find(|&i| strides[i] == 1 && shapes[i] != 1).map(|i| paths[i].clone());
        assert_eq!(a.leading_dim(), want, "{a}");
        for m in 0..a.rank() {
            assert_eq!(a.is_major(m), a.static_stride().get(&[m]).leaves()[0] == &1, "{a} mode {m}");
        }
    }
}

/// increment_coord on static coordinates, which fold to static results.
fn inc(coord: &str, shape: &str) -> IntTuple {
    let c: Coord = (&tup(coord)).into();
    increment_coord(c, tup(shape)).map(&mut |e| e.as_static().expect("static coordinates fold"))
}

#[test]
fn increment_coord_examples() {
    // dialect.
    assert_eq!(inc("(1,2)", "(4,8)"), tup("(2,2)"));
    assert_eq!(inc("(3,2)", "(4,8)"), tup("(0,3)"));
    // A carry out of the first leaf.
    assert_eq!(inc("(3,0,0)", "(4,2,5)"), tup("(0,1,0)"));
    // cutegen LayoutTest.IncrementCoord: carries, the end marker, rank 1.
    assert_eq!(inc("(0,0,0)", "(3,3,3)"), tup("(1,0,0)"));
    assert_eq!(inc("(2,1,1)", "(3,3,3)"), tup("(0,2,1)"));
    assert_eq!(inc("(2,2,1)", "(3,3,3)"), tup("(0,0,2)"));
    assert_eq!(inc("(2,2,2)", "(3,3,3)"), tup("(0,0,3)"));
    assert_eq!(inc("(0,0,3)", "(3,3,3)"), tup("(1,0,3)"));
    assert_eq!(inc("(1)", "(3)"), tup("(2)"));
    assert_eq!(inc("1", "3"), tup("2"));
    // cutegen: hierarchical shapes.
    assert_eq!(inc("((0,0),(1,0))", "((3,3),(3,3))"), tup("((1,0),(1,0))"));
    assert_eq!(inc("((2,2),(2,0))", "((3,3),(3,3))"), tup("((0,0),(0,1))"));
    assert_eq!(inc("((2),(1),(0,0))", "((3),(3),(2,2))"), tup("((0),(2),(0,0))"));
    assert_eq!(inc("(0)", "(3)"), tup("(1)"));
    assert_eq!(inc("((1,1),(1,1))", "((2,2),(2,2))"), tup("((0,0),(0,2))"));
    assert_eq!(inc("((1,1),(1))", "((2,2),(2))"), tup("((0,0),(2))"));
    // cutegen: a size-1 mode stays 0 and passes the carry on.
    assert_eq!(inc("(0,0,0)", "(3,1,8)"), tup("(1,0,0)"));
    assert_eq!(inc("(2,0,0)", "(3,1,8)"), tup("(0,0,1)"));
    // Out of range, as cutegen implements the dialect op: a leaf at or past
    // its extent wraps and carries, even without an incoming carry.
    assert_eq!(inc("(5,0)", "(4,8)"), tup("(0,1)"));
    assert_eq!(inc("(0,3,1)", "(2,3,4)"), tup("(1,0,2)"));
    assert_eq!(inc("(0,3,0)", "(2,1,4)"), tup("(1,0,0)"));
}

#[test]
fn property_increment_coord_counts_colexicographically() {
    let mut rng = Rng(0x1c0);
    let mut checked = 0;
    while checked < 300 {
        let shape = random_tree(&mut rng, 3, 1, &[1, 1, 2, 3, 4]);
        if shape.size() > 96 {
            continue;
        }
        checked += 1;
        let at = |i: usize| idx2crd(i, &shape);
        // Past the end, idx2crd keeps counting in the last leaf, as does
        // increment_coord.
        for i in 0..2 * shape.size() + 2 {
            assert_eq!(increment_coord(at(i), shape.clone()), at(i + 1), "increment of {:?} in {shape}", at(i));
        }
    }
}

#[test]
fn increment_coord_folds_what_it_can() {
    let kernel = Kernel::build("increment_structure", || {
        let out = param_tensor::<i32>("out", Layout::col_major(12));
        let t = thread_idx().x;
        let d = t.as_index();
        let s = |v: usize| IndexExpr::Static(v);
        let mut results = Vec::new();
        // cutegen: (?,1,8) from (0,0,0) is (?,0,?). A runtime leaf costs
        // two selects; the size-1 mode is a static 0.
        let c = increment_coord((t, 0usize, t), (3, 1, 8));
        assert!(!c.mode(0).as_leaf().unwrap().is_static() && !c.mode(2).as_leaf().unwrap().is_static());
        assert_eq!(c.mode(1), &Tree::Leaf(s(0)));
        results.push(c);
        // A static leaf that does not wrap leaves the rest alone.
        let c = increment_coord((1usize, t), (4, 8));
        assert_eq!(c, Coord::from((s(2), d.clone())));
        results.push(c);
        // A runtime leaf that provably stays in range needs no select.
        let c = increment_coord((0usize, d.clone() % 4usize, 5usize), (2, 4, 8));
        assert_eq!(c, Coord::from((s(1), d.clone() % 4usize, s(5))));
        results.push(c);
        // A static carry into the last leaf.
        let c = increment_coord((3usize, t), (4, 8));
        assert_eq!(c, Coord::from((s(0), d + 1usize)));
        results.push(c);
        for (i, leaf) in results.iter().flat_map(|c| c.flat_leaves()).enumerate() {
            out.store(i, Int32::from(leaf));
        }
    });
    assert_eq!(kernel.ir().count_ops(|op| matches!(op, Op::Select { .. })), 2);
}

#[test]
fn nullspace_examples() {
    assert_eq!(nullspace(&lay("(4,(2,3)):(0,(1,0))")), lay("(4,3):(1,8)"));
    assert_eq!(nullspace(&lay("8:1")), lay("1:0"));
    assert_eq!(nullspace(&lay("4:0")), lay("4:1"));
    assert_eq!(lay("((2,2),4):((0,1),0)").nullspace(), lay("(2,4):(1,4)"));
    // The layouts of C++ CuTe's test/unit/cute/core/nullspace.cpp.
    assert_eq!(nullspace(&lay("(2,2,2):(0,0,0)")), lay("(2,2,2):(1,2,4)"));
    assert_eq!(nullspace(&lay("(7,5,16):(0,0,0)")), lay("(7,5,16):(1,7,35)"));
    assert_eq!(nullspace(&lay("(2,2,2):(1,0,2)")), lay("2:2"));
    assert_eq!(nullspace(&lay("(7,5,16):(3,1,0)")), lay("16:35"));
    // CuTe DSL's strides: the product of the extents before the mode, size-1
    // modes included. C++ CuTe's compact_major gives a size-1 mode stride 0
    // instead, so there these are (2,1):(1,0), 1:0, (1,4):(0,1) and
    // (1,2):(0,3).
    assert_eq!(nullspace(&lay("(2,1,3):(0,0,1)")), lay("(2,1):(1,2)"));
    assert_eq!(nullspace(&lay("(1):(0)")), lay("1:1"));
    assert_eq!(nullspace(&lay("(1,4):(0,0)")), lay("(1,4):(1,1)"));
    assert_eq!(nullspace(&lay("(3,(1,2)):(1,(0,0))")), lay("(1,2):(3,3)"));
    // CuTe DSL raises for a shape that is a single integer with a zero
    // stride, and cute-rs takes C++'s answer: the integer 1 gets stride 0, so
    // 1:0 is its own nullspace, and so the nullspace of every integer-shaped
    // layout of size 1 is 1:0.
    assert_eq!(nullspace(&lay("1:0")), lay("1:0"));
    assert_eq!(nullspace(&lay("1:7")), lay("1:0"));
    assert_eq!(nullspace(&lay("3:0")), lay("3:1"));
    // So nullspace(nullspace(L)) is exactly 1:0 for a layout without zero
    // strides, such as a compact one.
    for l in [lay("8:1"), Layout::col_major((4, 8)), Layout::row_major((2, (3, 1)))] {
        assert_eq!(nullspace(&nullspace(&l)), lay("1:0"), "{l}");
    }
    // A runtime stride is never zero.
    let k = IndexExpr::Dynamic(ValueId(0));
    assert_eq!(Layout::with_strides((4, 2), (k.clone(), 0)).nullspace(), lay("2:4"));
    assert_eq!(Layout::with_strides((4, 2), (0, k)).nullspace(), lay("4:1"));
}

#[test]
fn property_nullspace_is_the_kernel() {
    let mut rng = Rng(0x5eed);
    for _ in 0..500 {
        let a = random_nested_layout(&mut rng);
        let n = nullspace(&a);
        assert_eq!(n.size() * a.filter_zeros().size(), a.size(), "nullspace({a}) = {n}");
        // CuTe's postcondition nullspace(n) == 1:0, exactly: n's strides are
        // products of extents, never 0, unless n is 1:0, which is its own
        // nullspace.
        assert_eq!(nullspace(&n), l(1, 0), "nullspace({a}) = {n}");
        // n enumerates, once each, exactly the indices a sends to 0.
        let mut reached: Vec<usize> = (0..n.size()).map(|i| n.call(i)).collect();
        reached.sort();
        reached.dedup();
        assert_eq!(reached.len(), n.size(), "nullspace({a}) = {n} is not injective");
        let zeros: Vec<usize> = (0..a.size()).filter(|&j| a.call(j) == 0).collect();
        assert_eq!(reached, zeros, "nullspace({a}) = {n}");
        // Its modes are a's zero-stride modes, flat and in order, each with
        // the product of the extents before it as its stride, except that a
        // shape that is the integer 1 takes C++'s stride 0.
        let (shapes, strides) = (a.static_shape().flat_leaves(), a.static_stride().flat_leaves());
        let stride_of = |i: usize| if a.static_shape() == Tree::Leaf(1) { 0 } else { shapes[..i].iter().product() };
        let modes: Vec<(usize, usize)> =
            (0..shapes.len()).filter(|&i| strides[i] == 0).map(|i| (shapes[i], stride_of(i))).collect();
        assert_eq!(n, Layout::from_flat(&modes), "nullspace({a}) = {n}");
    }
}

#[test]
fn tile_to_shape_dialect_examples() {
    // dialect: without an order, then rounded up, swizzled, and ordered.
    assert_cute_eq(&tile_to_shape(&lay("(3,2):(1,3)"), (6, 8)), "((3,2),(2,4)):((1,6),(3,12))");
    assert_cute_eq(&tile_to_shape(&lay("(3,3):(1,3)"), (5, 8)), "((3,2),(3,3)):((1,9),(3,18))");
    let atom = make_composed_layout(make_swizzle(3, 4, 3), 0, lay("(3,2):(1,3)"));
    let tiled = tile_to_shape_swizzled(&atom, (6, 8));
    assert_eq!(tiled.swizzle().unwrap(), make_swizzle(3, 4, 3));
    assert_cute_eq(&tiled.outer, "((3,2),(2,4)):((1,6),(3,12))");
    let block = lay("(128,128):(128,1)");
    let natural = "((128,8),(128,4)):((128,16384),(1,131072))";
    assert_cute_eq(&tile_to_shape_ordered(&block, (1024, 512), &[0, 1]), natural);
    assert_cute_eq(&tile_to_shape(&block, (1024, 512)), natural);
    assert_cute_eq(&tile_to_shape_ordered(&block, (1024, 512), &[1, 0]), "((128,8),(128,4)):((128,65536),(1,16384))");
    let swizzled =
        tile_to_shape_swizzled_ordered(&make_composed_layout(make_swizzle(3, 4, 3), 0, block), (1024, 512), &[1, 0]);
    assert_eq!(swizzled.swizzle().unwrap(), make_swizzle(3, 4, 3));
    assert_cute_eq(&swizzled.outer, "((128,8),(128,4)):((128,65536),(1,16384))");
}

#[test]
fn tile_to_shape_cutegen_examples() {
    // cutegen TileToShapeTest.Basic: (block, target, column-major result,
    // row-major result). cutegen also takes a relative order such as
    // (2,1,3); the dialect, and so cute-rs, takes the permutation it stands
    // for, here (1,0,2).
    let cases = [
        (
            "(64,8):(1,64)",
            "((128,1),(16,1),1)",
            "((64,2),(8,2),(1,1)):((1,512),(64,1024),(0,0))",
            "((64,2),(8,2),(1,1)):((1,1024),(64,512),(0,0))",
        ),
        (
            "(64,8):(1,64)",
            "((64,1),(16,1),1)",
            "((64,1),(8,2),(1,1)):((1,0),(64,512),(0,0))",
            "((64,1),(8,2),(1,1)):((1,0),(64,512),(0,0))",
        ),
        (
            "(8,64):(64,1)",
            "((128,1),(16,4),1)",
            "((8,16),(64,1),(1,1)):((64,512),(1,0),(0,0))",
            "((8,16),(64,1),(1,1)):((64,512),(1,0),(0,0))",
        ),
        (
            "(8,64):(64,1)",
            "((64,1),(16,4),1)",
            "((8,8),(64,1),(1,1)):((64,512),(1,0),(0,0))",
            "((8,8),(64,1),(1,1)):((64,512),(1,0),(0,0))",
        ),
        (
            "(8,64):(64,1)",
            "(64,128,2)",
            "((8,8),(64,2),(1,2)):((64,512),(1,4096),(0,8192))",
            "((8,8),(64,2),(1,2)):((64,2048),(1,1024),(0,512))",
        ),
        (
            "(8,64):(64,1)",
            "((16,4),(64,2),2)",
            "((8,8),(64,2),(1,2)):((64,512),(1,4096),(0,8192))",
            "((8,8),(64,2),(1,2)):((64,2048),(1,1024),(0,512))",
        ),
        (
            "(16,32):(1,16)",
            "(32,64,(2,1))",
            "((16,2),(32,2),(1,2)):((1,512),(16,1024),(0,2048))",
            "((16,2),(32,2),(1,2)):((1,2048),(16,1024),(0,512))",
        ),
    ];
    for (block, target, col, row) in cases {
        let (block, target) = (lay(block), tup(target));
        assert_cute_eq(&tile_to_shape(&block, target.clone()), col);
        assert_cute_eq(&tile_to_shape_ordered(&block, target.clone(), &[0, 1, 2]), col);
        assert_cute_eq(&tile_to_shape_ordered(&block, target, &[2, 1, 0]), row);
    }
    let block = lay("((32,4),(32,4)):((16,4),(0,1))");
    let ordered = [
        ([1, 0, 2], "(((32,4),8),((32,4),2),(1,2)):(((16,4),1024),((0,1),512),(0,8192))"),
        ([0, 1, 2], "(((32,4),8),((32,4),2),(1,2)):(((16,4),512),((0,1),4096),(0,8192))"),
        ([2, 1, 0], "(((32,4),8),((32,4),2),(1,2)):(((16,4),2048),((0,1),1024),(0,512))"),
    ];
    for (order, want) in ordered {
        assert_cute_eq(&tile_to_shape_ordered(&block, (1024, 256, 2), &order), want);
    }
}

#[test]
fn property_tile_to_shape_ordered() {
    let mut rng = Rng(0x7117);
    let mut checked = 0;
    while checked < 200 {
        let block = random_bijection(&mut rng);
        if block.size() > 64 {
            continue;
        }
        checked += 1;
        let r = block.rank() + (rng.next() % 2) as usize;
        let block_sizes: Vec<usize> = (0..r).map(|i| if i < block.rank() { block.mode(i).size() } else { 1 }).collect();
        // Targets the block divides, and targets it does not.
        let target: Vec<usize> = block_sizes
            .iter()
            .map(|&b| (b * rng.pick(&[1, 2, 3])).saturating_sub(rng.pick(&[0, 0, 1])).max(1))
            .collect();
        let mut order: Vec<usize> = (0..r).collect();
        rng.shuffle(&mut order);
        let t = tile_to_shape_ordered(&block, target.clone(), &order);
        let repeats: Vec<usize> = (0..r).map(|i| target[i].div_ceil(block_sizes[i])).collect();
        assert_eq!(t, blocked_product(&block.pad_to_rank(r), &make_ordered_layout(repeats.clone(), &order)));
        // Mode i is (block_i, repeats_i), covering the target.
        for i in 0..r {
            assert_eq!(t.mode(i).static_shape(), IntTuple::from((block_sizes[i], repeats[i])), "{t}");
            assert!(t.mode(i).size() >= target[i] && t.mode(i).size() < target[i] + block_sizes[i]);
        }
        // The copy at tile coordinate c starts at size(block) * offset(c),
        // where the copies are numbered along the mode with order 0 first.
        let tile_stride: Vec<usize> =
            (0..r).map(|i| (0..r).filter(|&j| order[j] < order[i]).map(|j| repeats[j]).product()).collect();
        let tiles: usize = repeats.iter().product();
        for b in 0..block.size() {
            let bc = block.idx2crd(b).flat_leaves();
            for c in 0..tiles {
                let mut rest = c;
                let tc: Vec<usize> = repeats
                    .iter()
                    .map(|&n| {
                        let x = rest % n;
                        rest /= n;
                        x
                    })
                    .collect();
                let coord: IntTuple =
                    Tree::Branch((0..r).map(|i| (bc.get(i).copied().unwrap_or(0), tc[i]).into()).collect());
                let copy: usize = (0..r).map(|i| tc[i] * tile_stride[i]).sum();
                assert_eq!(t.crd2idx(&coord), block.call(b) + block.size() * copy, "{t} at {coord}, order {order:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic leaves: extents and strides that are runtime values, printed `?`.
// Vectors marked "dialect" are the dynamic examples of
// media/docs/cutlass_compiler/cute_dialect.rst (CUTLASS 4.8.1). A `?` in a
// parsed layout is a fresh runtime value; `dv(i)` is a named one.

fn dv(i: u32) -> IndexExpr {
    IndexExpr::Dynamic(ValueId(1000 + i))
}

fn st(v: usize) -> IndexExpr {
    IndexExpr::Static(v)
}

fn parse_shape(s: &str) -> Shape {
    s.parse().unwrap_or_else(|e| panic!("{e}"))
}

/// `actual` prints as CuTe prints `cute`, `?` for each runtime leaf.
#[track_caller]
fn shows(actual: &Layout, cute: &str) {
    assert_eq!(actual.to_string(), cute, "{actual:?}");
}

/// The static layout a dynamic one stands for, with `env` giving the
/// runtime values.
fn eval_layout(l: &Layout, env: &dyn Fn(ValueId) -> usize) -> Layout {
    let eval = |t: &Shape| -> IntTuple { t.map(&mut |e| e.eval(env)) };
    Layout::new(eval(l.shape()), eval(&l.stride().as_int().expect("integer strides")))
}

/// `a` and `b` are the same function on the same domain.
#[track_caller]
fn assert_same_function(a: &Layout, b: &Layout, what: &str) {
    assert_eq!(a.size(), b.size(), "{what}: {a} and {b} differ in size");
    for i in 0..a.size() {
        assert_eq!(a.call(i), b.call(i), "{what}: {a} and {b} differ at {i}");
    }
}

#[test]
fn parse_and_print_dynamic_leaves() {
    // CuTe DSL's from_dlpack layout of a row-major matrix.
    let a = lay("(?,?):(?,1)");
    assert_eq!(a.to_string(), "(?,?):(?,1)");
    assert!(!a.is_static() && lay("(2,3):(1,2)").is_static());
    // Each `?` is a runtime value of its own.
    let dynamic: Vec<IndexExpr> = a
        .shape()
        .flat_leaves()
        .into_iter()
        .chain(a.stride().as_int().unwrap().flat_leaves())
        .filter(|e| !e.is_static())
        .collect();
    assert_eq!(dynamic.len(), 3);
    assert!(dynamic[0] != dynamic[1] && dynamic[1] != dynamic[2] && dynamic[0] != dynamic[2]);
    assert_ne!(lay("?:1"), lay("?:1"));
    assert_eq!(lay(" ( ? , _8 ) : ( 1 , ? ) ").to_string(), "(?,8):(1,?)");
    // Shapes parse too, and convert from and to static tuples.
    assert_eq!(parse_shape("(?,(3,?))").to_string(), "(?,(3,?))");
    assert_eq!(parse_shape("(2,(3,4))"), Shape::from(tup("(2,(3,4))")));
    assert_eq!(parse_shape("(2,(3,4))").as_static(), Some(tup("(2,(3,4))")));
    assert_eq!(parse_shape("(2,(?,4))").as_static(), None);
    // An IntTuple is static.
    let e = "(2,?)".parse::<IntTuple>().unwrap_err();
    assert_eq!(e.position(), 3, "{e}");
    assert!(e.to_string().contains("`?` (a runtime value) cannot be a leaf of an IntTuple"), "{e}");
    let e = "(2,!):(1,2)".parse::<Layout>().unwrap_err();
    assert_eq!(e.position(), 3, "{e}");
    assert!(e.to_string().contains("expected a non-negative integer, `?` or `(`"), "{e}");
    // Display prints `?`; Debug prints the expressions.
    let l = Layout::new((dv(0), 8), (8, 1));
    assert_eq!(l.to_string(), "(?,8):(8,1)");
    assert_eq!(format!("{l:?}"), "(%1000,8):(8,1)");
    let t = IndexExpr::ceil_div(dv(0), st(128));
    assert_eq!((t.to_string(), format!("{t:?}")), ("?".to_string(), "((%1000 + 127) / 128)".to_string()));
    // Layout::new and make_layout take any mix of leaves.
    assert_eq!(make_layout((dv(0), 8usize), (8, 1)), l);
    assert_eq!(Layout::with_strides((dv(0), 8), (8, 1)), l);
}

#[test]
fn property_parse_round_trips_with_dynamic_leaves() {
    let mut rng = Rng(0x3f00d);
    for _ in 0..300 {
        let t = random_nested_layout(&mut rng);
        let text = t.to_string();
        let marked: String =
            text.chars().map(|c| if c.is_ascii_digit() && rng.next().is_multiple_of(3) { '?' } else { c }).collect();
        // Merge runs of digits that now hold a `?` into one `?`.
        let mut cleaned = String::new();
        for c in marked.chars() {
            let prev = cleaned.chars().last();
            match (prev, c) {
                (Some('?'), '?' | '0'..='9') => {}
                (Some('0'..='9'), '?') => {
                    while cleaned.chars().last().is_some_and(|p| p.is_ascii_digit()) {
                        cleaned.pop();
                    }
                    cleaned.push('?');
                }
                _ => cleaned.push(c),
            }
        }
        let parsed = lay(&cleaned);
        assert_eq!(parsed.to_string(), cleaned);
        assert_eq!(parsed.to_string().parse::<Layout>().unwrap().to_string(), cleaned);
    }
}

#[test]
fn coalesce_dynamic_dialect_examples() {
    // dialect.
    shows(&coalesce(&lay("(?,4):(1,4)")), "(?,4):(1,4)");
    shows(&coalesce(&lay("(4,5):(?,4)")), "(4,5):(?,4)");
    shows(&coalesce(&lay("(?):(?)")), "?:?");
    shows(&coalesce(&lay("(4,5,?):(1,4,?)")), "(20,?):(1,?)");
    shows(&coalesce(&lay("(4,5,?,3,2):(1,4,?,1,3)")), "(20,?,6):(1,?,1)");
    shows(&lay("(4,?):(1,4)").coalesce_profile(&tup("(1,1)")), "(4,?):(1,4)");
    // The runtime leaves come through as they are.
    let a = Layout::new((4, 5, dv(0), 3, 2), (1, 4, dv(1), 1, 3));
    assert_eq!(a.coalesce(), Layout::new((20, dv(0), 6), (1, dv(1), 1)));
    // A merge whose condition cute-rs can prove happens even with runtime
    // leaves: a compact column-major layout of runtime extents is one mode,
    // whatever the order of the factors.
    let (m, n, p) = (dv(0), dv(1), dv(2));
    assert_eq!(Layout::col_major((m.clone(), n.clone())).coalesce(), Layout::new(m.clone() * n.clone(), 1));
    let reordered = Layout::new((m.clone(), n.clone(), p.clone()), (1, m.clone(), n.clone() * m.clone()));
    assert_eq!(reordered.coalesce().rank(), 1);
    assert_eq!(Layout::new((4, 5), (m.clone(), m.clone() * 4usize)).coalesce(), Layout::new(20, m.clone()));
    // CuTe keeps (4,?):(1,4) apart; it is (4*?):1 either way, a multiple
    // of 4 (printed as CuTe DSL prints a dynamic integer's divisibility).
    assert_eq!(lay("(4,?):(1,4)").coalesce().to_string(), "?{div=4}:1");
    // Only structure is compared: k * 2 and k + k are not known to be equal.
    let k = dv(3);
    assert_eq!(Layout::new((2, 3), (k.clone(), k.clone() + k.clone())).coalesce().rank(), 2);
}

#[test]
fn divides_with_dynamic_extents() {
    // A from_dlpack-style row-major (M, K) matrix, (?,?):(?,1), divided into
    // 128x32 tiles: the tiles are static, the rest modes ceil_div(M, 128)
    // and ceil_div(K, 32).
    let (m, k, ld) = (dv(0), dv(1), dv(2));
    let a = Layout::new((m.clone(), k.clone()), (ld.clone(), 1));
    let z = zipped_divide(&a, (128, 32));
    shows(&z, "((128,32),(?,?)):((?,1),(?{div=128},32))");
    let rest = (IndexExpr::ceil_div(m.clone(), st(128)), IndexExpr::ceil_div(k.clone(), st(32)));
    assert_eq!(z, Layout::new(((128, 32), rest.clone()), ((ld.clone(), 1), (ld.clone() * 128usize, 32))));
    assert_eq!(z.mode(0).size(), 128 * 32, "the tile is static");
    assert_eq!(size(&z, &[1]), rest.0.clone() * rest.1.clone());
    shows(&logical_divide(&a, (128, 32)), "((128,?),(32,?)):((?,?{div=128}),(1,32))");
    shows(&tiled_divide(&a, (128, 32)), "((128,32),?,?):((?,1),?{div=128},32)");
    shows(&flat_divide(&a, (128, 32)), "(128,32,?,?):(?,1,?{div=128},32)");
    // A tile of the whole layout, by a 1-D tiler.
    shows(&logical_divide(&lay("?:1"), 64), "(64,?):(1,64)");
    // A dynamic tiler extent: the tile is dynamic, the rest ceil_div.
    let t = dv(3);
    let d = logical_divide(&Layout::new(m.clone(), 1), Layout::new(t.clone(), 1));
    assert_eq!(d, Layout::new((t.clone(), IndexExpr::ceil_div(m.clone(), t.clone())), (1, t.clone())));
    // A tile count that divides exactly folds: ceil_div(4m, 4) is m.
    let exact = logical_divide(&Layout::new(m.clone() * 4usize, 1), 4);
    assert_eq!(exact, Layout::new((4, m.clone()), (1, 4)));
    // dialect: a layout tiler with runtime strides is taken when it has a
    // single mode, as ?:?, and refused with two, as (?,?):(?,?). The tiler's
    // extent and stride meet the static extents of (6,8):(8,1) at run time,
    // so the divide is the static one wherever that is defined.
    let (tv, sv) = (dv(6), dv(7));
    let d = logical_divide(&lay("(6,8):(8,1)"), Layout::new(tv.clone(), sv.clone()));
    shows(&d, "((?,?),((?,?),(?,?))):((?{div=8},?),((8,1),(?{div=8},?)))");
    for (t, s) in [(3, 2), (2, 3), (6, 1), (3, 1), (8, 6), (4, 12), (2, 24), (1, 48)] {
        let static_d = logical_divide(&lay("(6,8):(8,1)"), lay(&format!("{t}:{s}")));
        let env = |v: ValueId| if v == ValueId(1006) { t } else { s };
        assert_same_function(&eval_layout(&d, &env), &static_d, &format!("{d:?} at {t}:{s}"));
    }
    let refused = std::panic::catch_unwind(|| logical_divide(&lay("(6,8):(8,1)"), lay("(?,?):(?,?)")));
    assert!(refused.is_err(), "a two-mode tiler with runtime strides has no complement");
    // Tile extents that are runtime values give rest modes of ceil_div.
    let (t0, t1) = (dv(4), dv(5));
    let z = zipped_divide(&a, (Layout::new(t0.clone(), 1), Layout::new(t1.clone(), 1)));
    let rest = (IndexExpr::ceil_div(m.clone(), t0.clone()), IndexExpr::ceil_div(k.clone(), t1.clone()));
    assert_eq!(z, Layout::new(((t0.clone(), t1.clone()), rest), ((ld.clone(), 1), (ld.clone() * t0, t1))));
}

#[test]
fn products_with_a_dynamic_tiler_dialect_examples() {
    // dialect: a static block and a tiler with runtime extents.
    let (a, b) = (lay("(3,4):(4,1)"), lay("(?,?):(1,2)"));
    shows(&logical_product(&a, &b), "((3,4),(?,?)):((4,1),(12,24))");
    shows(&blocked_product(&a, &b), "((3,?),(4,?)):((4,12),(1,24))");
    shows(&raked_product(&a, &b), "((?,3),(?,4)):((12,4),(24,1))");
    shows(&zipped_product(&a, &b), "((3,4),(?,?)):((4,1),(12,24))");
    shows(&tiled_product(&a, &b), "((3,4),?,?):((4,1),12,24)");
    shows(&flat_product(&a, &b), "(3,4,?,?):(4,1,12,24)");
    // The tiler's runtime extents are carried through unchanged.
    let tiler = Layout::new((dv(0), dv(1)), (1, 2));
    assert_eq!(blocked_product(&a, &tiler), Layout::new(((3, dv(0)), (4, dv(1))), ((4, 12), (1, 24))));
    // A runtime tiler stride scales the tiler's strides.
    let k = dv(2);
    let p = logical_product(&lay("4:1"), Layout::new(3, k.clone()));
    assert_eq!(p, Layout::new((4, 3), (1, k.clone() * 4usize)));
}

#[test]
fn complement_with_dynamic_leaves() {
    // A runtime cotarget: the last mode covers ceil_div(cotarget, cosize).
    let n = dv(0);
    assert_eq!(complement(&lay("4:1"), n.clone()), Layout::new(IndexExpr::ceil_div(n.clone(), st(4)), 4));
    shows(&complement(&lay("(2,2):(1,6)"), n.clone()), "(3,?):(2,12)");
    assert_eq!(complement(&lay("(2,2):(1,6)"), 24), lay("(3,2):(2,12)"));
    // A runtime stride in a single-mode layout, as in C++ CuTe:
    // complement(s:d, c) is (d, ceil_div(c, s*d)):(1, s*d).
    let d = dv(1);
    let c = complement(&Layout::new(4, d.clone()), 64);
    let sd = d.clone() * 4usize;
    assert_eq!(c, Layout::new((d.clone(), IndexExpr::ceil_div(st(64), sd.clone())), (1, sd)));
    // A runtime extent.
    let m = dv(2);
    assert_eq!(complement(&Layout::new(m.clone(), 1), 64), Layout::new(IndexExpr::ceil_div(st(64), m.clone()), m));
    // Stride-0 and size-1 modes do not count, even next to runtime leaves.
    shows(&complement(&Layout::new((4, dv(3)), (1, 0)), 16), "4:4");
}

#[test]
fn composition_with_dynamic_leaves() {
    // Runtime strides on the left scale the result.
    let k = dv(0);
    let c = composition(&Layout::new(20, k.clone()), &lay("(5,4):(4,1)"));
    assert_eq!(c, Layout::new((5, 4), (k.clone() * 4usize, k.clone())));
    assert_eq!(composition(&lay("?:2"), &lay("(5,4):(4,1)")), lay("(5,4):(8,2)"));
    // A runtime extent on the left is assumed large and divisible: 64
    // elements of ((8,?),?) fill the 8 and continue into the ?.
    shows(&composition(&lay("(8,?,?):(1,?,?)"), &lay("64:1")), "(8,8):(1,?)");
    // A runtime extent on the right is not assumed to be anything: where it
    // meets the static 4 rows, the extents are computed at run time, as in
    // C++ CuTe. That is the static composition for every m it is defined for.
    let m = dv(1);
    let c = composition(&lay("(4,8):(8,1)"), &Layout::new(m.clone(), 1));
    let want = Layout::new((IndexExpr::min(m.clone(), st(4)), IndexExpr::ceil_div(m.clone(), st(4))), (8, 1));
    assert_eq!(c, want);
    for mv in [1, 2, 4, 8, 12, 16, 36] {
        let static_c = composition(&lay("(4,8):(8,1)"), &lay(&format!("{mv}:1")));
        assert_same_function(&eval_layout(&c, &|_| mv), &static_c, &format!("{c:?} at m = {mv}"));
    }
    // A runtime extent that is provably a multiple of 4 uses the rows whole.
    let c = composition(&lay("(4,8):(8,1)"), &Layout::new(m.clone() * 8usize, 1));
    assert_eq!(c, Layout::new((4, m.clone() * 2usize), (8, 1)));
    // A runtime stride on the right steps over a static extent when it is
    // provably a multiple of it...
    let s = dv(2);
    let c = composition(&lay("(4,8):(1,4)"), &Layout::new(3, s.clone() * 4usize));
    assert_eq!(c, Layout::new(3, s.clone() * 4usize));
    // ...and otherwise at run time: 2:s on (6,8):(8,1) is 2:24 for s = 3,
    // where taking s to be a multiple of 6 would give 2:1.
    let c = composition(&lay("(6,8):(8,1)"), &Layout::new(2, s.clone()));
    assert!(!c.is_static());
    for sv in [1, 3, 6, 12, 18, 30] {
        let static_c = composition(&lay("(6,8):(8,1)"), &lay(&format!("2:{sv}")));
        assert_same_function(&eval_layout(&c, &|_| sv), &static_c, &format!("{c:?} at s = {sv}"));
    }
    // Two runtime values that are not provably multiples of one another are
    // compared at run time too.
    let (k, t) = (dv(3), dv(4));
    let c = composition(&Layout::new((k.clone(), 4), (1, dv(5))), &Layout::new(t.clone(), 1));
    assert_eq!(c, Layout::new((IndexExpr::min(k.clone(), t.clone()), IndexExpr::ceil_div(t, k)), (1, dv(5))));
}

#[test]
fn inverses_with_dynamic_leaves() {
    // right_inverse follows static strides only: a runtime stride ends it.
    assert_eq!(right_inverse(&Layout::new((4, 8), (1, dv(0)))), lay("4:1"));
    // A compact layout of runtime extents coalesces to one mode first.
    let (m, n) = (dv(1), dv(2));
    assert_eq!(right_inverse(&Layout::col_major((m.clone(), 4))), Layout::new(m.clone() * 4usize, 1));
    // A runtime extent ends the chain after its mode.
    assert_eq!(right_inverse(&Layout::new((4, m.clone()), (1, 4))), Layout::new(m.clone() * 4usize, 1));
    assert_eq!(right_inverse(&Layout::new((m.clone(), 4), (4, 1))), Layout::new((4, m.clone()), (m.clone(), 1)));
    // left_inverse takes runtime extents with static strides.
    let l = Layout::new((n.clone(), 4), (4, 1));
    assert_eq!(left_inverse(&l), Layout::new((4, n.clone()), (n.clone(), 1)));
}

#[test]
fn size_and_cosize_queries() {
    // dialect: size and cosize, with mode paths.
    assert_eq!(size(&tup("(1,(2,4),9)"), &[1]), st(8));
    assert_eq!(size(&lay("(4,(16,32)):(1,(4,64))"), &[]), st(2048));
    let l = lay("(4,3,2):(1,4,12)");
    assert_eq!(cosize(&l, &[]), st(24));
    assert_eq!(cosize(&l, &[1]), st(9));
    assert_eq!((l.size(), l.cosize()), (24, 24));
    // A nested layout; nested paths; a leaf shape is its own mode 0.
    let a = lay("(2,(3,4)):(12,(1,3))");
    assert_eq!(cosize(&a, &[]), st(24));
    assert_eq!(cosize(&a, &[1, 1]), st(10));
    assert_eq!(size(&a, &[1, 0]), st(3));
    assert_eq!(size(&lay("8:1"), &[0]), st(8));
    assert_eq!(cosize(&lay("8:3"), &[0]), st(22));
    assert_eq!(cosize(&lay("(0,4):(1,1)"), &[]), st(0), "an empty layout has no offsets");
    // dialect: dynamic extents make sizes dynamic.
    let (m, n, k) = (dv(0), dv(1), dv(2));
    let d = Layout::new((m.clone(), (3, n.clone())), (1, (m.clone(), k.clone())));
    assert_eq!(size(&d, &[]), m.clone() * 3usize * n.clone());
    assert_eq!(size(&d, &[1, 0]), st(3));
    assert_eq!(size(&d, &[1]).to_string(), "?{div=3}");
    assert_eq!(d.shape().product_each().to_string(), "(?,?{div=3})");
    assert_eq!(d.shape().product().to_string(), "?{div=3}");
    assert_eq!(size(&d.shape().clone(), &[1, 1]), n.clone());
    // cosize is 1 + (m-1) + 2m + (n-1)k.
    let c = cosize(&d, &[]);
    for (mv, nv, kv) in [(1, 1, 1), (5, 2, 7), (3, 4, 9)] {
        let env = |v: ValueId| [mv, nv, kv][(v.0 - 1000) as usize];
        assert_eq!(c.eval(&env), 1 + (mv - 1) + 2 * mv + (nv - 1) * kv, "m={mv} n={nv} k={kv}");
        assert_eq!(cosize(&d, &[1]).eval(&env), 1 + 2 * mv + (nv - 1) * kv);
    }
    // Static leaves fold next to runtime ones: a runtime stride on a size-1
    // mode does not count, and the cosize of a compact layout of runtime
    // extents is its size.
    assert_eq!(cosize(&Layout::new((1, 8), (k.clone(), 1)), &[]), st(8));
    assert_eq!(Layout::new((1, 8), (k.clone(), 1)).cosize(), 8);
    let compact = Layout::col_major((m.clone(), n.clone()));
    assert!(cosize(&compact, &[]).same_value(&size(&compact, &[])), "{:?}", cosize(&compact, &[]));
    // IndexExpr does not collect like terms, so a larger one is only equal
    // in value.
    let compact = Layout::col_major((m.clone(), n.clone(), 4));
    for (mv, nv) in [(1, 1), (3, 5), (8, 2)] {
        let env = |v: ValueId| if v.0 == 1000 { mv } else { nv };
        assert_eq!(cosize(&compact, &[]).eval(&env), 4 * mv * nv);
    }
    // shape at a mode path keeps the leaves: runtime values in a layout's
    // shape, integers in an IntTuple. A leaf shape is its own mode 0.
    assert_eq!(shape(&a, &[]), *a.shape());
    assert_eq!(shape(&a, &[1]), Shape::from(tup("(3,4)")));
    assert_eq!(shape(&a, &[1, 0]), Tree::Leaf(st(3)));
    assert_eq!(shape(&tup("(1,(2,4),9)"), &[1, 1]), tup("4"));
    assert_eq!(shape(&lay("8:1"), &[0]), Tree::Leaf(st(8)));
    assert_eq!(shape(&d, &[1]), Shape::from((3, n.clone())));
    assert_eq!(shape(&d, &[1]).to_string(), "(3,?)");
    assert_eq!(size(&shape(&d, &[1]), &[]), size(&d, &[1]));
    // Swizzled layouts: the outer layout's size, shape and cosize, as in
    // CuTe (swizzle_layout.hpp's cosize).
    let sw = make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((8, 64)));
    assert_eq!((size(&sw, &[]), cosize(&sw, &[])), (st(512), st(512)));
    assert_eq!((size(&sw, &[0]), cosize(&sw, &[0])), (st(8), st(449)));
    assert_eq!(shape(&sw, &[1]), Tree::Leaf(st(64)));
    // Tensors have shapes.
    Kernel::build("size_of_tensors", || {
        let t = param_tensor::<f32>("t", Layout::col_major((4, (2, 3))));
        assert_eq!((size(&t, &[]), size(&t, &[1]), size(&t, &[1, 1])), (st(24), st(6), st(3)));
        assert_eq!(shape(&t, &[1]), Shape::from((2, 3)));
        assert_eq!(t.shape().mode(1).size(), 6);
    });
}

#[test]
fn rank_and_depth_queries() {
    // CuTe DSL's rank and depth: a leaf has rank 1 and depth 0, and the
    // empty tuple rank 0 and depth 1.
    assert_eq!((rank(&tup("8"), &[]), depth(&tup("8"))), (1, 0));
    assert_eq!((rank(&tup("()"), &[]), depth(&tup("()"))), (0, 1));
    assert_eq!(rank(&tup("8"), &[0]), 1, "a leaf is its own mode 0");
    // Mode paths into a nested tuple; the depth is the deepest nesting.
    let t = tup("(1,(2,(3,4)),9)");
    assert_eq!((rank(&t, &[]), rank(&t, &[1]), rank(&t, &[1, 1]), rank(&t, &[1, 1, 0])), (3, 2, 2, 1));
    assert_eq!((depth(&t), depth(&tup("(1,2,3)")), depth(&tup("((),5)"))), (3, 1, 2));
    // Layouts and shapes, whose ranks and depths are static even where
    // their extents are runtime values.
    let a = lay("(4,(3,2)):(2,(8,1))");
    assert_eq!((rank(&a, &[]), rank(&a, &[1]), depth(&a), a.rank(), a.depth()), (2, 2, 2, 2, 2));
    let d = Layout::new((dv(0), (3, dv(1))), (1, (dv(0), dv(2))));
    assert_eq!((rank(&d, &[]), rank(&d, &[1]), rank(&d, &[1, 1]), depth(&d)), (2, 2, 1, 2));
    assert_eq!((rank(d.shape(), &[1]), depth(d.shape())), (2, 2));
    // Swizzled layouts have their layout's.
    let sw = make_composed_layout(make_swizzle(3, 3, 3), 0, lay("((8,2),64):((64,512),1)"));
    assert_eq!((rank(&sw, &[]), rank(&sw, &[0]), depth(&sw), sw.rank(), sw.depth()), (2, 2, 2, 2, 2));
    let flat = make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((8, 64)));
    assert_eq!((rank(&flat, &[1]), depth(&flat), flat.rank(), flat.depth()), (1, 1, 2, 1));
}

#[test]
fn coordinates_with_dynamic_leaves() {
    // dialect crd2idx(coord, shape), which is crd2idx with the column-major
    // layout of the shape, and layout_eval.
    assert_eq!(crd2idx((1, 2), &Layout::col_major((4, 8))), st(9));
    let (i, j, m, s) = (dv(0), dv(1), dv(2), dv(3));
    assert_eq!(crd2idx((i.clone(), j.clone()), &Layout::col_major((4, 8))), i.clone() + j.clone() * 4usize);
    assert_eq!(crd2idx((1, 2), &Layout::col_major((m.clone(), 8))), m.clone() * 2usize + 1usize);
    assert_eq!(crd2idx((1, 2), &Layout::new((4, 8), (1, s.clone()))), s.clone() * 2usize + 1usize);
    assert_eq!(crd2idx((1, 2), &Layout::new((4, 8), (1, s.clone()))).to_string(), "?");
    // dialect idx2crd.
    assert_eq!(idx2crd(9, &tup("(4,8)")), Coord::from((1, 2)));
    assert_eq!(idx2crd(i.clone(), &tup("(4,8)")), Coord::from((i.clone() % 4usize, i.clone() / 4usize)));
    let c = idx2crd(9, &Shape::from((m.clone(), 8)));
    assert_eq!(c.to_string(), "(?,?)");
    assert_eq!(c, Coord::from((st(9) % m.clone(), st(9) / m.clone())));
    // A 1-D index into a hierarchical mode of runtime extents divides by
    // them at run time.
    let l = Layout::new(((m.clone(), 3), 8), ((1, s.clone()), s.clone() * 3usize));
    let off = crd2idx((10, 2), &l);
    for (mv, sv) in [(1, 1), (4, 5), (7, 9), (11, 12)] {
        let env = |v: ValueId| if v.0 == 1002 { mv } else { sv };
        // The index splits over (m, 3) with the last extent unbounded.
        let want = 10 % mv + 10 / mv * sv + 2 * 3 * sv;
        assert_eq!(off.eval(&env), want, "m={mv} s={sv}");
        assert_eq!(eval_layout(&l, &env).crd2idx(&tup("(10,2)")), want);
    }
    // dialect slice: the kept modes keep their runtime strides.
    let (kept, off) = Layout::new((3, 5, 7), (s.clone(), 2, 4)).slice(&(.., 0, ..).into());
    assert_eq!((kept, off), (Layout::new((3, 7), (s.clone(), 4)), st(0)));
    let (kept, off) = Layout::new((m.clone(), 3, 4), (1, s.clone(), dv(4))).slice(&(.., 1, ..).into());
    assert_eq!((kept, off), (Layout::new((m.clone(), 4), (1, dv(4))), s.clone()));
    // A runtime coordinate slices too.
    let (kept, off) = l.slice(&(.., j.clone()).into());
    assert_eq!((kept, off), (Layout::new(((m.clone(), 3),), ((1, s.clone()),)), j.clone() * s.clone() * 3usize));
}

#[test]
fn tuple_helpers_with_dynamic_extents() {
    // leading_dim takes a dynamic extent as "not 1".
    assert_eq!(Layout::new((dv(0), 4), (1, 8)).leading_dim(), Some(vec![0]));
    assert_eq!(Layout::new((1, dv(0)), (1, 1)).leading_dim(), Some(vec![1]));
    // nullspace with runtime extents: the strides after them are runtime.
    let (m, n) = (dv(1), dv(2));
    let ns = nullspace(&Layout::new((m.clone(), 4, n.clone()), (0, 1, 0)));
    assert_eq!(ns, Layout::new((m.clone(), n.clone()), (1, m.clone() * 4usize)));
    // tile_to_shape onto a runtime target.
    let t = tile_to_shape(&lay("(8,8):(8,1)"), (m.clone(), n.clone()));
    let (rm, rn) = (IndexExpr::ceil_div(m.clone(), st(8)), IndexExpr::ceil_div(n.clone(), st(8)));
    assert_eq!(t, Layout::new(((8, rm.clone()), (8, rn)), ((8, 64), (1, rm * 64usize))));
    // make_ordered_layout, dialect.
    shows(&make_ordered_layout(parse_shape("(4,3,?,2)"), &[2, 1, 3, 4]), "(4,3,?,2):(3,1,12,?{div=12})");
    // increment_coord compares against runtime extents (a compare and two
    // selects per leaf that can wrap), and folds static ones.
    let kernel = Kernel::build("increment_dynamic_extent", || {
        let out = param_tensor::<i32>("out", Layout::col_major(4));
        let t = thread_idx().x;
        let extent = t + 2;
        let c = increment_coord((1usize, t), (extent, 8));
        assert!(!c.mode(0).as_leaf().unwrap().is_static());
        let s = increment_coord((1usize, t), (4, extent));
        assert_eq!(s, Coord::from((st(2), t.as_index())));
        for (i, leaf) in c.flat_leaves().into_iter().chain(s.flat_leaves()).enumerate() {
            out.store(i, Int32::from(leaf));
        }
    });
    assert_eq!(kernel.ir().count_ops(|op| matches!(op, Op::Select { .. })), 2);
}

/// A divide, as the property tests run them.
type DivideOp = fn(&Layout, &Tiler) -> Layout;

/// A product, as the property tests run them.
type ProductOp = fn(&Layout, &Layout) -> Layout;

/// The first value id [`dynamize`] hands out.
const DYN: u32 = 5000;

/// `t` with each leaf, with probability `percent`, replaced by a fresh
/// runtime value standing for it; `values` records what each stands for.
fn dynamize(rng: &mut Rng, t: &IntTuple, percent: u64, values: &mut Vec<usize>) -> Shape {
    t.map(&mut |&v| {
        if rng.next() % 100 < percent {
            values.push(v);
            IndexExpr::Dynamic(ValueId(DYN + values.len() as u32 - 1))
        } else {
            st(v)
        }
    })
}

/// `template` with runtime values standing for some of its extents and
/// strides.
fn dynamize_layout(rng: &mut Rng, template: &Layout, percent: u64, values: &mut Vec<usize>) -> Layout {
    let shape = dynamize(rng, &template.static_shape(), percent, values);
    let stride = dynamize(rng, &template.static_stride(), percent, values);
    Layout::new(shape, stride)
}

fn env_of(values: &[usize]) -> impl Fn(ValueId) -> usize + '_ {
    move |v| values[(v.0 - DYN) as usize]
}

/// Queries, offsets, coordinates, slices and coalesce on layouts with
/// runtime leaves, evaluated at the values they stand for, agree with the
/// same operations on the static layout of those values.
#[test]
fn property_dynamic_layouts_evaluate_like_their_static_values() {
    let mut rng = Rng(0xd7a1);
    for _ in 0..400 {
        let template = random_nested_layout(&mut rng);
        let mut values = Vec::new();
        let d = dynamize_layout(&mut rng, &template, 40, &mut values);
        let n = values.len() as u32;
        let env = env_of(&values);
        assert_eq!(eval_layout(&d, &env), template);
        assert_eq!(size(&d, &[]).eval(&env), template.size(), "{d:?}");
        assert_eq!(cosize(&d, &[]).eval(&env), template.cosize(), "{d:?}");
        for m in 0..d.rank() {
            assert_eq!(size(&d, &[m]).eval(&env), template.mode(m).size(), "{d:?} mode {m}");
            assert_eq!(cosize(&d, &[m]).eval(&env), template.mode(m).cosize(), "{d:?} mode {m}");
        }
        for i in 0..template.size() {
            assert_eq!(d.offset_at(i).eval(&env), template.call(i), "{d:?} at {i}");
            assert_eq!(idx2crd(i, d.shape()).map(&mut |e| e.eval(&env)), template.idx2crd(i), "{d:?} at {i}");
        }
        // A runtime index stands for each static one.
        let mut values_i = values.clone();
        values_i.push(0);
        let index = IndexExpr::Dynamic(ValueId(DYN + n));
        let offset = d.offset(&Tree::Leaf(index.clone()));
        let coord = idx2crd(index, d.shape());
        for i in 0..template.size() {
            values_i[n as usize] = i;
            let env_i = env_of(&values_i);
            assert_eq!(offset.eval(&env_i), template.call(i), "{d:?} at runtime {i}");
            assert_eq!(coord.map(&mut |e| e.eval(&env_i)), template.idx2crd(i));
        }
        // coalesce: the same function, merging no more than for the values.
        let c = coalesce(&d);
        assert_same_function(&eval_layout(&c, &env), &template, &format!("coalesce({d:?}) = {c:?}"));
        assert!(c.flatten().rank() >= coalesce(&template).flatten().rank(), "coalesce({d:?}) = {c:?}");
        // A slice at static or runtime coordinates of some top-level modes.
        let mut values_s = values.clone();
        let picks: Vec<SliceCoord> = (0..d.rank())
            .map(|m| match rng.next() % 3 {
                0 => Tree::Leaf(None),
                1 => Tree::Leaf(Some(st(rng.next() as usize % template.mode(m).size()))),
                _ => {
                    values_s.push(rng.next() as usize % template.mode(m).size());
                    Tree::Leaf(Some(IndexExpr::Dynamic(ValueId(DYN + values_s.len() as u32 - 1))))
                }
            })
            .collect();
        let env_s = env_of(&values_s);
        let static_picks: SliceCoord =
            Tree::Branch(picks.iter().map(|p| p.map(&mut |c| c.as_ref().map(|e| st(e.eval(&env_s))))).collect());
        let (kept, off) = d.slice(&Tree::Branch(picks));
        let (want_kept, want_off) = template.slice(&static_picks);
        assert_eq!((eval_layout(&kept, &env_s), off.eval(&env_s)), (want_kept, want_off.as_static().unwrap()), "{d:?}");
    }
}

/// By-mode divides of layouts with runtime extents and strides, by tiles
/// that divide the extents, agree with the static ones exactly (up to the
/// strides of size-1 modes, which never matter).
#[test]
fn property_dynamic_divides_match_the_static_ones() {
    let mut rng = Rng(0xd1d);
    for _ in 0..300 {
        let rank = 1 + (rng.next() % 3) as usize;
        let shapes: Vec<usize> = (0..rank).map(|_| rng.pick(&[1, 2, 3, 4, 6, 8, 12])).collect();
        let strides: Vec<usize> = (0..rank).map(|_| rng.pick(&[0, 1, 2, 3, 5, 8, 13])).collect();
        let template = Layout::new(shapes.clone(), strides);
        let mut values = Vec::new();
        let d = dynamize_layout(&mut rng, &template, 50, &mut values);
        let env = env_of(&values);
        let tile = |rng: &mut Rng, s: usize| {
            let divisors: Vec<usize> = (1..=s).filter(|n| s.is_multiple_of(*n)).collect();
            Tree::Leaf(Some(Layout::new(rng.pick(&divisors), 1)))
        };
        let tiler: Tiler = Tree::Branch(shapes.iter().map(|&s| tile(&mut rng, s)).collect());
        let ops: [(&str, DivideOp); 4] = [
            ("logical_divide", Layout::logical_divide),
            ("zipped_divide", Layout::zipped_divide),
            ("tiled_divide", Layout::tiled_divide),
            ("flat_divide", Layout::flat_divide),
        ];
        for (name, op) in ops {
            let (dynamic, want) = (op(&d, &tiler), op(&template, &tiler));
            let got = eval_layout(&dynamic, &env);
            assert_eq!(unit_modes_zeroed(&got), unit_modes_zeroed(&want), "{name}({d:?}, {tiler:?}) = {dynamic:?}");
        }
        // `..` leaves a mode as it is.
        let partial: Tiler = Tree::Branch(
            shapes
                .iter()
                .map(|&s| if rng.next().is_multiple_of(2) { Tree::Leaf(None) } else { tile(&mut rng, s) })
                .collect(),
        );
        let got = eval_layout(&d.logical_divide(&partial), &env);
        assert_eq!(unit_modes_zeroed(&got), unit_modes_zeroed(&template.logical_divide(&partial)));
    }
}

/// Products of a static block with a tiler of runtime extents and strides
/// agree with the static ones. A compact block's complement is a single
/// mode, so this holds for any values.
#[test]
fn property_products_with_dynamic_tilers_match_the_static_ones() {
    let mut rng = Rng(0x9e0d);
    let mut checked = 0;
    while checked < 250 {
        let block = random_bijection(&mut rng);
        if block.size() > 32 {
            continue;
        }
        checked += 1;
        let template = random_layout(&mut rng);
        let mut values = Vec::new();
        let tiler = dynamize_layout(&mut rng, &template, 50, &mut values);
        let env = env_of(&values);
        let ops: [(&str, ProductOp); 6] = [
            ("logical_product", |a, b| logical_product(a, b)),
            ("blocked_product", blocked_product),
            ("raked_product", raked_product),
            ("zipped_product", zipped_product),
            ("tiled_product", tiled_product),
            ("flat_product", flat_product),
        ];
        for (name, op) in ops {
            let (dynamic, want) = (op(&block, &tiler), op(&block, &template));
            let mut got = eval_layout(&dynamic, &env);
            if name == "raked_product" {
                // raked_product coalesces each mode, which a runtime extent
                // that is 1 at these values does not let it do when it is built.
                got = Layout::from_modes(got.modes().iter().map(Layout::coalesce));
            }
            assert_eq!(unit_modes_zeroed(&got), unit_modes_zeroed(&want), "{name}({block}, {tiler:?}) = {dynamic:?}");
        }
    }
}

/// A block with gaps: the tiler's runtime extents are assumed to be
/// multiples of the gaps' extents, which they are here (multiples of 24).
#[test]
fn property_products_of_blocks_with_gaps() {
    let mut rng = Rng(0x6a95);
    let mut checked = 0;
    while checked < 100 {
        let block = random_injective(&mut rng);
        if block.size() > 8 || block.cosize() > 32 || block.cosize() == block.size() {
            continue;
        }
        let template = Layout::col_major((rng.pick(&[24, 48, 72]), rng.pick(&[1, 2, 3])));
        // Gaps that do not divide the tiler make the product undefined.
        let Ok(want) = std::panic::catch_unwind(|| logical_product(&block, &template)) else { continue };
        checked += 1;
        let mut values = Vec::new();
        let tiler =
            Layout::new(dynamize(&mut rng, &template.static_shape(), 70, &mut values), template.stride().clone());
        let env = env_of(&values);
        let dynamic = logical_product(&block, &tiler);
        assert_same_function(&eval_layout(&dynamic, &env), &want, &format!("{block} x {tiler:?} = {dynamic:?}"));
    }
}

/// Composition with runtime extents on the left, standing for large values
/// (1728, a multiple of every stride and extent of the right-hand side), and
/// strides computed from them, agrees with the static composition wherever
/// that is defined.
#[test]
fn property_composition_with_dynamic_extents() {
    let mut rng = Rng(0xc0de);
    let mut checked = 0;
    for _ in 0..3000 {
        if checked >= 300 {
            break;
        }
        // A permuted compact layout; some modes get the extent 1728.
        let rank = 1 + (rng.next() % 4) as usize;
        let mut order: Vec<usize> = (0..rank).collect();
        rng.shuffle(&mut order);
        let mut values = Vec::new();
        let mut shapes = Vec::new();
        let mut static_shapes = Vec::new();
        for _ in 0..rank {
            if rng.next().is_multiple_of(3) {
                values.push(1728);
                shapes.push(IndexExpr::Dynamic(ValueId(DYN + values.len() as u32 - 1)));
                static_shapes.push(1728);
            } else {
                let s = rng.pick(&[1, 2, 3, 4, 8]);
                shapes.push(st(s));
                static_shapes.push(s);
            }
        }
        let a = make_ordered_layout(shapes, &order);
        let a_static = make_ordered_layout(static_shapes, &order);
        let env = env_of(&values);
        assert_eq!(eval_layout(&a, &env), a_static);
        let b = random_injective(&mut rng);
        if b.cosize() > 64 || b.cosize() > a_static.size() {
            continue;
        }
        let Ok(want) = std::panic::catch_unwind(|| composition(&a_static, &b)) else { continue };
        let got = composition(&a, &b);
        assert_same_function(&eval_layout(&got, &env), &want, &format!("({a:?}) o ({b}) = {got:?}"));
        checked += 1;
    }
    assert!(checked >= 300, "too few compositions were defined ({checked})");
}

/// Runtime strides on the left scale the composition. A runtime stride can
/// also keep two modes from merging (it is not known to make them
/// contiguous), and then a composition that crosses them is refused; where
/// it is defined, it agrees with the static one.
#[test]
fn property_composition_with_dynamic_strides() {
    let mut rng = Rng(0x57d);
    let (mut checked, mut refused) = (0, 0);
    for _ in 0..3000 {
        let a = random_bijection(&mut rng);
        let b = random_injective(&mut rng);
        if b.cosize() > a.size() {
            continue;
        }
        let Ok(want) = std::panic::catch_unwind(|| composition(&a, &b)) else { continue };
        let mut values = Vec::new();
        let d = Layout::new(a.shape().clone(), dynamize(&mut rng, &a.static_stride(), 50, &mut values));
        let env = env_of(&values);
        let Ok(got) = std::panic::catch_unwind(|| composition(&d, &b)) else {
            refused += 1;
            continue;
        };
        assert_same_function(&eval_layout(&got, &env), &want, &format!("({d:?}) o ({b}) = {got:?}"));
        checked += 1;
    }
    assert!(checked > 200 && checked > 2 * refused, "{checked} compositions checked, {refused} refused");
}

/// complement with a runtime cotarget, or runtime extents, agrees with the
/// static one.
#[test]
fn property_complement_with_dynamic_leaves() {
    let mut rng = Rng(0xc0c1);
    for _ in 0..400 {
        let a = random_injective(&mut rng);
        let cotarget = 1 + rng.next() as usize % (2 * a.cosize() + 8);
        let want = complement(&a, cotarget);
        let mut values = vec![cotarget];
        let got = complement(&a, IndexExpr::Dynamic(ValueId(DYN)));
        let got = eval_layout(&got, &env_of(&values));
        assert_same_function(&got, &want, &format!("complement({a}, ?={cotarget}) = {got:?}"));
        // A runtime extent is a generic one, so it stands for one of at least
        // 2 (a static 1 would be dropped).
        let shape = a.static_shape().map(&mut |&s| {
            if s > 1 && rng.next().is_multiple_of(2) {
                values.push(s);
                IndexExpr::Dynamic(ValueId(DYN + values.len() as u32 - 1))
            } else {
                st(s)
            }
        });
        let d = Layout::new(shape, a.stride().clone());
        let env = env_of(&values);
        let got = complement(&d, cotarget);
        assert_same_function(&eval_layout(&got, &env), &want, &format!("complement({d:?}, {cotarget}) = {got:?}"));
    }
}

/// right_inverse ignores runtime strides, so it still inverts what it
/// covers; with a runtime extent on the outermost mode, both inverses are
/// complete.
#[test]
fn property_inverses_with_dynamic_leaves() {
    let mut rng = Rng(0x1a5);
    for _ in 0..400 {
        let a = random_bijection(&mut rng);
        let mut values = Vec::new();
        let d = Layout::new(a.shape().clone(), dynamize(&mut rng, &a.static_stride(), 30, &mut values));
        let r = right_inverse(&d);
        let env = env_of(&values);
        let r = eval_layout(&r, &env);
        assert!(r.size() <= a.size());
        for i in 0..r.size() {
            assert_eq!(a.call(r.call(i)), i, "right_inverse({d:?}) = {r}");
        }
        // The outermost mode (largest stride) with a runtime extent.
        let strides = a.static_stride().flat_leaves();
        let shapes = a.static_shape().flat_leaves();
        let Some(outer) = (0..shapes.len()).filter(|&i| shapes[i] > 1).max_by_key(|&i| strides[i]) else { continue };
        let values = vec![shapes[outer]];
        let shape: Vec<IndexExpr> = (0..shapes.len())
            .map(|i| if i == outer { IndexExpr::Dynamic(ValueId(DYN)) } else { st(shapes[i]) })
            .collect();
        let d = Layout::new(Tree::unflatten(&shape, &a.static_shape()), a.stride().clone());
        let env = env_of(&values);
        let r = eval_layout(&right_inverse(&d), &env);
        assert_eq!(r.size(), a.size(), "right_inverse({d:?}) = {r}");
        let l = eval_layout(&left_inverse(&d), &env);
        for i in 0..a.size() {
            assert_eq!(a.call(r.call(i)), i, "right_inverse({d:?}) = {r}");
            assert_eq!(l.call(a.call(i)), i, "left_inverse({d:?}) = {l}");
        }
    }
}

/// Dividing all of a layout with runtime extents by a static tiler splits
/// the rest over the layout's modes, as C++ CuTe does (it takes the
/// complement of the tiler in the layout's shape), so the rest modes line up
/// with the layout's modes. The divides agree with the static ones at
/// extents that the tile divides.
#[test]
fn whole_layout_divides_with_dynamic_extents() {
    let (m, n, ld) = (dv(0), dv(1), dv(2));
    let env3 = |mv: usize, nv: usize, lv: usize| move |v: ValueId| [mv, nv, lv][(v.0 - 1000) as usize];
    let rm = |t: usize| IndexExpr::ceil_div(m.clone(), st(t));
    // CuTe DSL's (?,?):(?,1), by 32.
    let a = make_layout((m.clone(), n.clone()), (ld.clone(), 1));
    let d = logical_divide(&a, 32);
    assert_eq!(d, Layout::new((32, (rm(32), n.clone())), (ld.clone(), (ld.clone() * 32usize, 1))));
    for (mv, nv, lv) in [(32, 1, 1), (64, 3, 5), (96, 2, 7), (32, 4, 4)] {
        let want = logical_divide(&Layout::new((mv, nv), (lv, 1)), 32);
        assert_same_function(&eval_layout(&d, &env3(mv, nv, lv)), &want, &format!("{d:?} at ({mv},{nv}):({lv},1)"));
    }
    // Column-major with a runtime leading dimension.
    let b = make_layout((m.clone(), n.clone()), (1, ld.clone()));
    let d = logical_divide(&b, 32);
    assert_eq!(d, Layout::new((32, (rm(32), n.clone())), (1, (32, ld.clone()))));
    for (mv, nv, lv) in [(32, 2, 40), (64, 3, 64), (96, 1, 100)] {
        let want = logical_divide(&Layout::new((mv, nv), (1, lv)), 32);
        assert_same_function(&eval_layout(&d, &env3(mv, nv, lv)), &want, &format!("{d:?} at ({mv},{nv}):(1,{lv})"));
    }
    // One runtime extent and a static one.
    let c = make_layout((m.clone(), 8), (1, ld.clone()));
    let d = logical_divide(&c, 64);
    assert_eq!(d, Layout::new((64, (rm(64), 8)), (1, (64, ld.clone()))));
    for (mv, lv) in [(64, 64), (128, 130), (192, 200)] {
        let want = logical_divide(&Layout::new((mv, 8), (1, lv)), 64);
        assert_same_function(&eval_layout(&d, &env3(mv, 0, lv)), &want, &format!("{d:?} at ({mv},8):(1,{lv})"));
    }
    // By mode, with a hierarchical mode of a runtime extent.
    let e = Layout::new(((m.clone(), 2), n.clone()), ((2, 1), m.clone() * 2usize));
    let z = zipped_divide(&e, (8, 4));
    let rn = IndexExpr::ceil_div(n.clone(), st(4));
    let want_z = Layout::new(((8, 4), ((rm(8), 2), rn)), ((2, m.clone() * 2usize), ((16, 1), m.clone() * 8usize)));
    assert_eq!(z, want_z);
    for (mv, nv) in [(8, 4), (16, 8), (24, 12)] {
        let want = zipped_divide(&Layout::new(((mv, 2), nv), ((2, 1), 2 * mv)), (8, 4));
        assert_same_function(&eval_layout(&z, &env3(mv, nv, 0)), &want, &format!("{z:?} at m={mv} n={nv}"));
    }
    // The rest mode of a zipped divide, divided again.
    let f = Layout::row_major((m.clone(), n.clone()));
    let d = logical_divide(&zipped_divide(&f, (128, 64)).mode(1), 4);
    for (mv, nv) in [(512, 64), (1024, 128), (2048, 192)] {
        let want = logical_divide(&zipped_divide(&Layout::row_major((mv, nv)), (128, 64)).mode(1), 4);
        assert_same_function(&eval_layout(&d, &env3(mv, nv, 0)), &want, &format!("{d:?} at m={mv} n={nv}"));
    }
}

/// Compositions whose right-hand side has runtime extents and strides that
/// stand for small values, multiples of nothing in particular: wherever the
/// static composition is defined, the dynamic one agrees with it. It is
/// refused only where a static stride does not fit the static extents it
/// steps over, which the static composition lets through for a mode of
/// extent 1 when the stride is smaller than the extent, or where the static
/// composition takes a stride smaller than a mode that it does not divide,
/// which it keeps only after checking every value. (A left-hand side
/// with runtime strides is another matter: they can keep its modes from
/// merging, and the divisibility conditions then apply to the unmerged
/// modes.)
#[test]
fn property_composition_with_runtime_right_hand_sides() {
    /// Whether C++ CuTe's walk of `a ∘ b` takes a stride smaller than a mode
    /// of `a` that it does not divide: the static composition is then kept
    /// only after checking every value, which a runtime extent or stride
    /// prevents.
    fn takes_indivisible_stride(a: &Layout, b: &Layout) -> bool {
        let lhs: Vec<usize> = a.coalesce().flat_modes().into_iter().map(|(s, _)| s).collect();
        b.flat_modes().into_iter().filter(|&(_, d)| d != 0).any(|(s, d)| {
            let (mut rest_shape, mut rest_stride) = (s, d);
            for &cs in &lhs[..lhs.len() - 1] {
                let next_shape = cs.div_ceil(rest_stride);
                if next_shape > 1 && rest_shape > 1 {
                    if !cs.is_multiple_of(rest_stride) {
                        return true;
                    }
                    rest_shape /= next_shape.min(rest_shape).max(1);
                }
                rest_stride = rest_stride.div_ceil(cs);
            }
            false
        })
    }
    let mut rng = Rng(0x2b5);
    let (mut checked, mut refused, mut at_run_time) = (0, 0, 0);
    for _ in 0..4000 {
        let a = random_bijection(&mut rng);
        let b = random_injective(&mut rng);
        if b.cosize() > a.size() {
            continue;
        }
        let Ok(want) = std::panic::catch_unwind(|| composition(&a, &b)) else { continue };
        let mut values = Vec::new();
        let shape = dynamize(&mut rng, &b.static_shape(), 50, &mut values);
        let runtime_extents = values.clone();
        // A stride of 0 stays static: a runtime stride is never taken to be 0.
        let stride = b.static_stride().map(&mut |&d| {
            if d != 0 && rng.next().is_multiple_of(2) {
                values.push(d);
                IndexExpr::Dynamic(ValueId(DYN + values.len() as u32 - 1))
            } else {
                st(d)
            }
        });
        let rb = Layout::new(shape, stride);
        let Ok(got) = std::panic::catch_unwind(|| composition(&a, &rb)) else {
            assert!(
                runtime_extents.contains(&1) || takes_indivisible_stride(&a, &b),
                "({a}) o ({rb:?}) was refused, but ({a}) o ({b}) = {want}"
            );
            refused += 1;
            continue;
        };
        assert_same_function(&eval_layout(&got, &env_of(&values)), &want, &format!("({a}) o ({rb:?}) = {got:?}"));
        // Runtime strides alone that made the extents runtime values.
        at_run_time += usize::from(!got.shape().is_static() && rb.shape().is_static());
        checked += 1;
    }
    assert!(checked > 300 && checked > 10 * refused, "{checked} compositions checked, {refused} refused");
    assert!(at_run_time > 20, "too few runtime strides needed runtime extents ({at_run_time})");
}

/// Divides of all of a layout with runtime extents (multiples of 16, as the
/// large-and-divisible assumption needs) and runtime strides, by static
/// tilers, agree with the static divides wherever those are defined.
#[test]
fn property_whole_layout_divides_with_dynamic_extents() {
    let mut rng = Rng(0x7d1);
    let (mut checked, mut refused, mut tries) = (0, 0, 0);
    while checked < 200 {
        tries += 1;
        assert!(tries < 20_000, "too few divides were defined ({checked})");
        let rank = 2 + (rng.next() % 2) as usize;
        let shapes: Vec<usize> = (0..rank).map(|_| rng.pick(&[1, 2, 3, 4, 8, 16, 32, 48])).collect();
        let mut order: Vec<usize> = (0..rank).collect();
        rng.shuffle(&mut order);
        let mut strides = vec![0; rank];
        let mut running = 1;
        for &i in &order {
            strides[i] = running;
            running *= shapes[i];
        }
        let template = Layout::new(shapes.clone(), strides);
        let tiler: Tiler = (&Layout::new(rng.pick(&[2, 4, 8]), rng.pick(&[1, 1, 2]))).into();
        let Ok(want) = std::panic::catch_unwind(|| template.logical_divide(&tiler)) else { continue };
        let mut values = Vec::new();
        let mut shape = Vec::new();
        for &s in &shapes {
            if s.is_multiple_of(16) && !rng.next().is_multiple_of(4) {
                values.push(s);
                shape.push(IndexExpr::Dynamic(ValueId(DYN + values.len() as u32 - 1)));
            } else {
                shape.push(st(s));
            }
        }
        if values.is_empty() {
            continue;
        }
        // Runtime strides, in half the cases, can keep modes from merging that
        // the static layout merges; the static divide of the unmerged modes
        // can then be undefined, and so refused.
        let percent = if rng.next().is_multiple_of(2) { 0 } else { 40 };
        let d = Layout::new(shape, dynamize(&mut rng, &template.static_stride(), percent, &mut values));
        let env = env_of(&values);
        let Ok(got) = std::panic::catch_unwind(|| d.logical_divide(&tiler)) else {
            assert!(!d.stride().is_static(), "{d:?} / {tiler:?} was refused, but {template} / {tiler:?} = {want}");
            refused += 1;
            continue;
        };
        assert_same_function(&eval_layout(&got, &env), &want, &format!("{d:?} / {tiler:?} = {got:?}"));
        let zipped = d.zipped_divide(&tiler);
        assert_same_function(&eval_layout(&zipped, &env), &template.zipped_divide(&tiler), &format!("{d:?}"));
        checked += 1;
    }
    assert!(checked > 4 * refused, "{checked} divides checked, {refused} refused");
}

/// `max_common_vector` and `max_common_layout` take the static part of a
/// contiguous run whose length is a runtime value. C++ CuTe keeps
/// `(8,m):(1,8)` apart and finds 8; cute-rs coalesces it to `(8*m):1`,
/// whose largest static divisor is 8.
#[test]
fn max_common_with_runtime_extents() {
    let (m, ld) = (dv(0), dv(1));
    let col = Layout::col_major((8, m.clone()));
    assert_eq!(max_common_vector(&col, &col), 8);
    assert_eq!(max_common_layout(&col, &col), Layout::new(8, 1));
    let row = Layout::row_major((m.clone(), 8));
    assert_eq!(max_common_vector(&row, &row), 8);
    assert_eq!(max_common_layout(&row, &row), Layout::new(8, m.clone()));
    // A runtime stride keeps the modes apart.
    let strided = Layout::new((8, m.clone()), (1, ld.clone()));
    assert_eq!(max_common_vector(&strided, &strided), 8);
    assert_eq!(max_common_layout(&strided, &strided), Layout::new(8, 1));
    // A runtime extent with no static factor gives nothing more than one
    // element, as does a run that is not contiguous in both.
    let bare = Layout::col_major(m.clone());
    assert_eq!((max_common_vector(&bare, &bare), max_common_layout(&bare, &bare)), (1, Layout::new(1, 0)));
    assert_eq!(max_common_vector(&col, &Layout::row_major((8, m.clone()))), 1);
    // What is known about a runtime extent counts.
    Kernel::build("max_common_extent", || {
        let k = crate::param_extent("k", 4);
        let col = Layout::col_major((8, k));
        assert_eq!(max_common_vector(&col, &col), 32);
        assert_eq!(max_common_layout(&col, &col), Layout::new(32, 1));
    });
}

// Scaled-basis strides, arithmetic tuples and identity layouts. "cutegen"
// marks the vectors of cg_scaled_basis_test.cpp and cg_ratio_test.cpp.

fn stride_of(s: &str) -> Stride {
    s.parse().unwrap_or_else(|e| panic!("{e}"))
}

fn crd(s: &str) -> Coord {
    Coord::from(tup(s))
}

#[test]
fn basis_elements_print_and_parse() {
    assert_eq!(E(0).to_string(), "1@0");
    assert_eq!(E([1, 0]).to_string(), "1@0@1");
    assert_eq!(E([]), StrideLeaf::Int(st(1)));
    assert_eq!((2 * E(0)).to_string(), "2@0");
    assert_eq!((E([0, 1]) * 3usize).to_string(), "3@1@0");
    // cutegen: an integer mode and a path of one index are the same basis.
    assert_eq!(ScaledBasis::new(2, [1]), ScaledBasis::new(2, 1));
    assert_eq!(ScaledBasis::new(64, [2, 3, 4, 5, 6, 7]).to_string(), "64@7@6@5@4@3@2");
    assert_eq!(ScaledBasis::new(Ratio::new(1, 4), [2, 3, 4, 5, 6, 7]).to_string(), "1/4@7@6@5@4@3@2");
    // cutegen: strings parse back, innermost mode first.
    assert_eq!(stride_of("1@2@2"), Stride::from(ScaledBasis::new(1, [2, 2])));
    assert_eq!(stride_of("2@2"), Stride::from(ScaledBasis::new(2, 2)));
    assert_eq!(stride_of("64@7@6@5@4@3@2"), Stride::from(ScaledBasis::new(64, [2, 3, 4, 5, 6, 7])));
    assert_eq!(stride_of("2/4@7@6@5@4@3@2"), Stride::from(ScaledBasis::new(Ratio::new(2, 4), [2, 3, 4, 5, 6, 7])));
    for bad in ["64@7@6@", "64-7@6", "1/4", "1/0@0"] {
        assert!(bad.parse::<Stride>().is_err(), "{bad}");
    }
    let dynamic = lay("((?,?),?,?,?,?):((?@1,?@0),?@2,?@0,?@1,?@2)");
    assert_eq!(dynamic.to_string(), "((?,?),?,?,?,?):((?@1,?@0),?@2,?@0,?@1,?@2)");
    assert!(!dynamic.stride().is_static() && stride_of("(1@0,0)").is_static());
    // A runtime scale.
    let m = IndexExpr::Dynamic(ValueId(7));
    assert_eq!((m * E(1)).to_string(), "?@1");
    assert_eq!(make_layout((4, 8), (2 * E(0), E(1))).to_string(), "(4,8):(2@0,1@1)");
}

#[test]
fn identity_layouts() {
    // dialect: make_identity_layout.
    assert_eq!(make_identity_layout((4, 2)).to_string(), "(4,2):(1@0,1@1)");
    assert_eq!(make_identity_layout((2, (3, 4), 5)).to_string(), "(2,(3,4),5):(1@0,(1@0@1,1@1@1),1@2)");
    assert_eq!(make_identity_layout(parse_shape("(?,3)")).to_string(), "(?,3):(1@0,1@1)");
    assert_eq!(make_identity_layout(4).to_string(), "4:1");
    // cutegen: make_basis_like.
    assert_eq!(make_basis_like(&tup("(2,2,2)")), stride_of("(1@0,1@1,1@2)"));
    assert_eq!(make_basis_like(&tup("(4)")), stride_of("(1@0)"));
    // Every coordinate maps to itself, as a 1-D index and as a coordinate.
    for shape in ["(4,3)", "(2,(3,4),5)", "((2,2),(3,1))", "6"] {
        let shape = tup(shape);
        let id = make_identity_layout(&shape);
        for i in 0..shape.size() {
            let c = idx2crd(i, &shape);
            assert_eq!(id.eval(&Tree::Leaf(st(i))), c, "{id} at {i}");
            assert_eq!(id.eval(&c), c, "{id} at {c}");
        }
        assert_eq!(id.cosize(), shape.size());
    }
}

#[test]
fn basis_layouts_evaluate_to_tuples() {
    // CuTe DSL's ScaledBasis example.
    assert_eq!(lay("(4,8):(2@0,1@1)").eval(&crd("(2,3)")), crd("(4,3)"));
    // dialect: layout_eval.
    assert_eq!(lay("(4,8):(1@0,1@1)").eval(&crd("(1,2)")), crd("(1,2)"));
    assert_eq!(lay("(4,8):(1@1,1@0)").eval(&crd("(1,2)")), crd("(2,1)"));
    assert_eq!(lay("((2,3),4):((1@0,1@1),1@2)").eval(&crd("(1,2)")), crd("(1,0,2)"));
    assert_eq!(lay("((2,3),4):((1@1@0,1@0@0),1@1)").eval(&crd("0")), crd("((0,0),0)"));
    // cutegen: SparseScaledBasis, LayoutAndCoordinateStatic.
    assert_eq!(lay("(3,5):(2@1,1@1)").eval(&crd("(1,1)")), crd("(0,3)"));
    assert_eq!(lay("(4,1,3):(0,0,64@0)").eval(&crd("5")), crd("(64)"));
    assert_eq!(lay("(((2),1),1,1,2):(((0),0),0,0,64@0)").eval(&crd("10")), crd("(320)"));
    assert_eq!(lay("(10,20,30):(1@0,1@1,20@1)").eval(&crd("(1,2,3)")), crd("(1,62)"));
    assert_eq!(lay("((64,128),2):((1@0,1@1),64@1)").eval(&crd("(65,1)")), crd("(1,65)"));
    assert_eq!(lay("((128,16),1,4):((64,1),0,16)").eval(&crd("(0,0,0)")), crd("0"));
    // Integer layouts give their offset.
    assert_eq!(lay("(4,8):(1,4)").eval(&crd("(1,2)")), crd("9"));
    // cutegen: a runtime coordinate leaves the other entries static.
    let m = IndexExpr::Dynamic(ValueId(9));
    let t = lay("(10,20,30):(1@0,1@1,20@1)").eval(&Tree::tuple([Tree::Leaf(m.clone()), crd("2"), crd("3")]));
    assert!(!t.mode(0).as_leaf().unwrap().is_static() && t.mode(1) == &crd("62"), "{t:?}");
    let t = lay("((10,10),20,30):((1@0,10@0),1@1,1@2)").eval(&Tree::tuple([Tree::Leaf(m.clone()), crd("2"), crd("3")]));
    assert_eq!(t.rank(), 3);
    assert!(!t.mode(0).as_leaf().unwrap().is_static() && t.mode(1) == &crd("2") && t.mode(2) == &crd("3"));
    // A ratio scale: 5 * 1/5 is 1.
    assert_eq!(lay("10:1/5@0").eval(&crd("5")), crd("(1)"));
}

#[test]
fn basis_slices() {
    // cutegen: slice_and_offset of basis layouts.
    let l = lay("((64,128),2):((1@0,1@1),64@1)");
    let c1: SliceCoord = ((.., ..), 1).into();
    let c2: SliceCoord = (.., 1).into();
    let (l1, off1) = l.slice_and_offset(&c1);
    let (l2, off2) = l.slice_and_offset(&c2);
    assert_eq!(l1.to_string(), "(64,128):(1@0,1@1)");
    assert_eq!(l2.to_string(), "((64,128)):((1@0,1@1))");
    assert_eq!((off1.clone(), off2), (crd("(0,64)"), crd("(0,64)")));
    let ma = lay("((64,128),1,1,2):((1@1,1@0),0,0,64@0)");
    assert_eq!(ma.slice_and_offset(&0.into()).1, crd("(0,0)"));
    assert_eq!(ma.slice_and_offset(&(0, 0, 0, 1).into()).1, crd("(64,0)"));
    let gemm = lay("((1,(1,1)),((128,16),(1,4,?,?,?))):((0,(0,0)),((1@1,1@0),(0,16@0,128@1,64@0,1@2)))");
    let (s, o) = gemm.slice_and_offset(&((0, (0, 0)), (.., (.., .., .., .., ..))).into());
    assert_eq!(s.to_string(), "((128,16),1,4,?,?,?):((1@1,1@0),0,16@0,128@1,64@0,1@2)");
    assert_eq!(o, crd("(0,0,0)"));
    let nested = lay("(((128,64),1),1):(((1@0,1@1),1@0),64@0)");
    assert_eq!(nested.slice_and_offset(&((.., 0), 0).into()).1, crd("(0,0)"));
    // An integer layout's offset is a leaf, as Layout::slice gives it.
    let (s, o) = lay("(4,8):(8,1)").slice_and_offset(&(.., 2).into());
    assert_eq!((s.to_string(), o), ("(4):(8)".to_string(), crd("2")));
}

#[test]
fn basis_composition_coalesce_and_cosize() {
    // cutegen: basis strides on the right-hand side pick a mode.
    let a = lay("(4,3,5):(3,1,12)");
    shows(&composition(&a, &lay("2:1@1")), "2:1");
    shows(&composition(&a, &lay("2:1@2")), "2:12");
    shows(&composition(&a, &lay("2:2@2")), "2:24");
    let e = lay("(4,(2,3)):(2,(1,8))");
    shows(&composition(&e, &lay("2:2@1@1")), "2:16");
    shows(&composition(&e, &lay("2:2@1")), "2:8");
    // cutegen: basis strides on the left-hand side are scaled.
    let tma = lay("(512,64,2):(1@1,1@0,64@0)");
    shows(&composition(&tma, &lay("16:1")), "16:1@1");
    shows(&composition(&tma, &lay("(4,16):(512,2)")), "(4,16):(1@0,2@1)");
    shows(&coalesce(&tma), "(512,128):(1@1,1@0)");
    shows(&coalesce(&lay("(2,2):(1@0,2@1)")), "(2,2):(1@0,2@1)");
    shows(&coalesce(&lay("(2,1,3):(1@0,0,2@0)")), "6:1@0");
    // cutegen: CoalesceTest's basis strides.
    shows(&coalesce(&lay("(2,4):(1@1,2@1)")), "8:1@1");
    shows(&coalesce(&lay("(2,1,6,1):(1@1@1,1@3@2,2@1@1,1@3@2)")), "12:1@1@1");
    shows(&coalesce(&lay("(2,1,6,1):(1@1@1,1@3@2,2@0@1,1@3@2)")), "(2,6):(1@1@1,2@0@1)");
    shows(&coalesce(&lay("((2,2),(2,2)):((1@0,1@1),(2@1,2@0))")), "(2,4,2):(1@0,1@1,2@0)");
    // cutegen: CompositionTest's basis strides on the right-hand side.
    shows(&composition(&lay("(4,4):(4,1)"), &lay("(4,4):(1@1,1@0)")), "(4,4):(1,4)");
    shows(&composition(&lay("(4,(2,3)):(6,(3,1))"), &lay("(2,4):(1@1@1,1@0)")), "(2,4):(1,6)");
    // cutegen: the cosize of a basis layout is the size of its coshape.
    assert_eq!(lay("(10,(10,10),10):(1@1,(1@0,10@0),1@2)").cosize(), 100 * 10 * 10);
    assert_eq!(cosize(&make_identity_layout((4, 8)), &[1]), st(8));
    let m = IndexExpr::Dynamic(ValueId(11));
    assert_eq!(cosize(&make_identity_layout((m.clone(), 3)), &[]), m * 3usize);
}

#[test]
fn basis_divides_and_products() {
    // dialect: products of an identity layout.
    let id = make_identity_layout((4, 3));
    shows(&complement(&id, 120), "(30,40):(4@0,3@1)");
    shows(&blocked_product(&id, &lay("(2,5):(1,2)")), "((4,2),(3,5)):((1@0,4@0),(1@1,8@0))");
    shows(&flat_product(&id, &lay("(2,5):(1,2)")), "(4,3,2,5):(1@0,1@1,4@0,8@0)");
    // dialect: a divide of a TMA-like basis layout.
    shows(&flat_divide(&lay("(128,64):(1@1,1@0)"), lay("2:64")), "(2,64,64):(64@1,1@1,1@0)");
    shows(&zipped_divide(&make_identity_layout((8, 8)), (4, 4)), "((4,4),(2,2)):((1@0,1@1),(4@0,4@1))");
    // A basis layout that covers its cotarget has a trivial complement
    // (the dialect's complement with no cotarget gives `(1):(3@0)`).
    shows(&complement(&lay("3:1@0"), 3), "(1):(0)");
    // cutegen: ScaledBasisWithCoTarget.
    shows(&complement(&lay("3:1@0"), 6), "(2):(3@0)");
    // One bucket for each leaf of the coprofile, as cutegen extends the
    // complement with an integer cotarget: a mode that no stride walks, or
    // one before a named mode, is covered whole.
    shows(&complement(&lay("(4,1):(1@0,1@1)"), 8), "(2,8):(4@0,1@1)");
    shows(&complement(&lay("3:4@2"), 24), "(24,24,(4,2)):(1@0,1@1,(1@2,12@2))");
    shows(&complement(&lay("(2,5,3):(4@1,5@0,16@1)"), 100), "((5,4),(4,2,3)):((1@0,25@0),(1@1,8@1,48@1))");
    shows(&blocked_product(&make_identity_layout((4, 1)), &lay("(2,5):(1,2)")), "((4,2),(1,5)):((1@0,4@0),(1@1,8@0))");
}

#[test]
fn property_tiled_identity_layouts() {
    // Tiling an identity layout and evaluating it agrees with idx2crd of the
    // same tiling of the compact layout.
    let mut rng = Rng(0xba5e_ba5e);
    for _ in 0..200 {
        let rank = 1 + (rng.next() % 3) as usize;
        let shape: Vec<usize> = (0..rank).map(|_| rng.pick(&[2, 4, 6, 8, 12])).collect();
        let tiles: Vec<usize> = shape.iter().map(|&s| rng.pick(&[1, 2, s])).collect();
        let shape = IntTuple::from(shape);
        let tiler = Tiler::from(&IntTuple::from(tiles));
        let id = make_identity_layout(&shape);
        let compact = Layout::col_major(&shape);
        for (tiled_id, tiled) in [
            (zipped_divide(&id, tiler.clone()), zipped_divide(&compact, tiler.clone())),
            (logical_divide(&id, tiler.clone()), logical_divide(&compact, tiler.clone())),
            (flat_divide(&id, tiler.clone()), flat_divide(&compact, tiler.clone())),
        ] {
            for i in 0..shape.size() {
                let want = idx2crd(tiled.call(i), &shape);
                assert_eq!(tiled_id.eval(&Tree::Leaf(st(i))), want, "{tiled_id} vs {tiled} at {i}");
            }
        }
    }
}

#[test]
fn ratios() {
    // cutegen: equality is as written, until reduced.
    assert_ne!(Ratio::new(4, 8), Ratio::new(1, 2));
    assert_eq!(Ratio::new(1, 2), Ratio::new(1, 2));
    assert_ne!(Ratio::new(99, -11), Ratio::new(-99, 11));
    assert!(!Ratio::new(4, 8).is_integral() && Ratio::new(16, 4).is_integral() && Ratio::new(99, -11).is_integral());
    assert_eq!(Ratio::new(4, 8).reduced(), Ratio::new(1, 2));
    assert_eq!(Ratio::new(99, -11).reduced(), Ratio::new(-9, 1));
    assert_eq!(Ratio::new(0, 4).reduced(), Ratio::new(0, 1));
    // Products, as CuTe DSL's Ratio multiplies, unreduced.
    assert_eq!(Ratio::new(1, 2) * Ratio::new(8, 4), Ratio::new(8, 8));
    assert_eq!(Ratio::new(1, 2) * 8, Ratio::new(8, 2));
    assert_eq!(11 * Ratio::new(2, 6), Ratio::new(22, 6));
    assert_eq!((Ratio::new(-11, 3).to_int(), Ratio::new(11, 3).to_int(), Ratio::new(1, 4).to_f64()), (-4, 3, 0.25));
    assert_eq!(Ratio::new(1, 4).to_string(), "1/4");
}

#[test]
fn tuple_predicates_and_basis_helpers() {
    // cutegen: has_scaled_basis.
    assert!(!stride_of("(1,2,3)").has_scaled_basis());
    assert!(stride_of("1@0").has_scaled_basis() && stride_of("(1,(1@0))").has_scaled_basis());
    assert!(make_identity_layout((2, 3)).has_scaled_basis() && !lay("(2,3):(1,2)").has_scaled_basis());
    // has_underscore.
    assert!(SliceCoord::from((1, ..)).has_underscore() && !SliceCoord::from((1, 2)).has_underscore());
    assert!(Tiler::from((4, ..)).has_underscore() && !Tiler::from((4, 8)).has_underscore());
    // is_static over shapes, strides, coordinates and layouts.
    assert!(parse_shape("(2,3)").is_static() && !parse_shape("(2,?)").is_static());
    assert!(stride_of("(1@0,2)").is_static() && !stride_of("(?@0,2)").is_static());
    assert!(lay("(2,3):(1@0,1@1)").is_static() && !lay("(?,3):(1@0,1@1)").is_static());
    // basis_value and basis_get.
    assert_eq!(basis_value(&(4 * E(1))), BasisScale::Int(st(4)));
    assert_eq!(basis_value(&StrideLeaf::Int(st(5))), BasisScale::Int(st(5)));
    let t = tup("((2,3),(4,5))");
    assert_eq!(basis_get(&E([1, 0]), &t), tup("4"));
    assert_eq!(basis_get(&E(0), &t), tup("(2,3)"));
    assert_eq!(basis_get(&StrideLeaf::Int(st(1)), &t), t);
    let l = lay("((2,3),(4,5)):((1,2),(6,24))");
    shows(&basis_get(&E([1, 1]), &l), "5:24");
}

// ---------------------------------------------------------------------------
// Shape and index arithmetic over dynamic leaves, equality, the coordinate
// of an offset, by-mode products and signed index terms. "dialect" marks
// the examples of media/docs/cutlass_compiler/cute_dialect.rst.

#[test]
fn ceil_div_shape_div_and_round_up_dialect_examples() {
    let s = |t: &str| parse_shape(t);
    // dialect: ceil_div.
    assert_eq!(ceil_div(7, 3), s("3"));
    assert_eq!(ceil_div((8, 10), (3, 4)), s("(3,3)"));
    assert_eq!(ceil_div((8, 10, 6), (3, 4)), s("(3,3,6)"));
    assert_eq!(ceil_div((8, 16), lay("((4,8)):((1,4))")), s("4"));
    // The tuple / integer and integer / tuple cases.
    assert_eq!(ceil_div((4, 5, 6), 40), s("(1,1,3)"));
    assert_eq!(ceil_div(100, (4, 8)), s("4"));
    // A Tiler of layouts gives zipped_divide's rest mode.
    assert_eq!(ceil_div((13, 20), Tiler::from((4, 8))), s("(4,3)"));
    // dialect: shape_div.
    assert_eq!(shape_div(12, 4), s("3"));
    assert_eq!(shape_div((8, 10), (2, 5)), s("(4,2)"));
    assert_eq!(shape_div((4, 5, 6), 40), s("(1,1,3)"));
    assert_eq!(shape_div(4, 12), s("1"));
    assert_eq!(shape_div(24, (2, 3)), s("4"));
    // round_up pads the divisor with 1s.
    assert_eq!(round_up((9, 10, 11), (4, 8)), s("(12,16,11)"));
    assert_eq!(round_up(9, 4), s("12"));
    // Runtime leaves give runtime results, which evaluate as integers do.
    let (m, n) = (dv(0), dv(1));
    let env = |v: ValueId| [13, 100][(v.0 - 1000) as usize];
    let grid = ceil_div(Shape::from((m.clone(), n.clone())), (4, 8));
    assert_eq!(grid.to_string(), "(?,?)");
    assert_eq!(grid.map(&mut |e| e.eval(&env)), tup("(4,13)"));
    assert_eq!(ceil_div(Shape::from((m.clone(), 16)), (4, 8)).map(&mut |e| e.eval(&env)), tup("(4,2)"));
    assert_eq!(round_up(m.clone(), 8).map(&mut |e| e.eval(&env)), tup("16"));
    assert_eq!(shape_div(Shape::from((n.clone(), 6)), 4).map(&mut |e| e.eval(&env)), tup("(25,6)"));
    assert_eq!(shape_div(Shape::from((4, n)), 40).map(&mut |e| e.eval(&env)), tup("(1,10)"));
    // A tile with a runtime extent: a layout divides the whole input, a
    // tuple mode by mode.
    assert_eq!(ceil_div((64, 64), Tiler::from(m.clone())).map(&mut |e| e.eval(&env)), tup("316"));
    assert_eq!(ceil_div((64, 64), Tiler::from((m, 16))).map(&mut |e| e.eval(&env)), tup("(5,4)"));
}

#[test]
fn equal_folds_or_compares_at_run_time() {
    Kernel::build("equal_folds", || {
        let folded = |b: crate::Boolean| b.as_constant();
        // dialect: equal.
        assert_eq!(folded(equal(&parse_shape("(3,4)"), &parse_shape("(3,4)"))), Some(true));
        assert_eq!(folded(equal(&parse_shape("(3,4)"), &parse_shape("(3,5)"))), Some(false));
        assert_eq!(folded(equal(&lay("(4,8):(1,4)"), &lay("(4,8):(1,4)"))), Some(true));
        // Different profiles, and basis strides of different modes.
        assert_eq!(folded(equal(&lay("(4,8):(1,4)"), &lay("((2,2),8):((1,2),4)"))), Some(false));
        assert_eq!(folded(equal(&lay("4:1@0"), &lay("4:1@1"))), Some(false));
        let m = Int32::from(thread_idx().x);
        let a = Layout::new((IndexExpr::from(m), 8), (1, IndexExpr::from(m)));
        // The same expression folds; a static leaf that differs decides.
        assert_eq!(folded(equal(&a, &a.clone())), Some(true));
        let b = Layout::new((IndexExpr::from(m), 4), (1, IndexExpr::from(m)));
        assert_eq!(folded(equal(&a, &b)), Some(false));
        // m * 2 and m + m are compared at run time, where they are equal.
        let x = Layout::new(IndexExpr::from(m * 2), 1);
        let y = Layout::new(IndexExpr::from(m + m), 1);
        assert_ne!(x, y);
        assert_eq!(folded(equal(&x, &y)), None);
    });
}

#[test]
fn hier_and_flat_coords() {
    // dialect: idx2crd of 9 in (4,8).
    assert_eq!(idx2crd(9, &tup("(4,8)")), Coord::from(tup("(1,2)")));
    assert_eq!(idx2crd(dv(0), &tup("(4,8)")).to_string(), "(?,?)");
    assert_eq!(idx2crd(9, &parse_shape("(?,8)")).to_string(), "(?,?)");
    assert_eq!(Layout::col_major((4, 8)).get_hier_coord(9), Coord::from(tup("(1,2)")));
    // A row-major layout maps 9 back to (1,1), and a size-1 leaf gives 0.
    assert_eq!(lay("(4,8):(8,1)").get_hier_coord(9), Coord::from(tup("(1,1)")));
    assert_eq!(lay("(4,1,2):(1,0,4)").get_hier_coord(6), Coord::from(tup("(2,0,1)")));
    // Hierarchical coordinates, and the flat coordinate of each mode.
    let h = lay("((2,3),4):((12,1),3)");
    for i in 0..24 {
        let c = h.get_hier_coord(i);
        assert_eq!(h.offset(&c), st(i), "{h} at {i}: {c}");
        let flat = h.get_flat_coord(i);
        let (inner, outer) = (c.mode(0).leaves(), c.mode(1).leaves());
        let want = Coord::from((inner[0].clone() + inner[1].clone() * 2usize, outer[0].clone()));
        assert_eq!(flat, want, "{h} at {i}");
    }
    // A runtime index and extent stay runtime values.
    let l = Layout::new((4, dv(0)), (dv(0), 1));
    let c = l.get_hier_coord(dv(1));
    assert_eq!(c.to_string(), "(?,?)");
    let env = |v: ValueId| [5, 13][(v.0 - 1000) as usize];
    assert_eq!(c.map(&mut |e| e.eval(&env)), tup("(2,3)"));
}

#[test]
fn property_hier_coord_inverts_compact_layouts() {
    let mut rng = Rng(0xc00d);
    for _ in 0..300 {
        let mut template = random_bijection(&mut rng);
        if template.rank() >= 2 && rng.next().is_multiple_of(2) {
            // Nest the first two modes.
            let mut modes = vec![Layout::from_modes([template.mode(0), template.mode(1)])];
            modes.extend((2..template.rank()).map(|i| template.mode(i)));
            template = Layout::from_modes(modes);
        }
        for i in 0..template.size() {
            let c = template.get_hier_coord(i);
            assert!(c.congruent(&template.shape().clone()), "{template} at {i}: {c}");
            assert_eq!(template.offset(&c), st(i), "{template} at {i}: {c}");
        }
        // The same with runtime extents and strides, evaluated.
        let mut values = Vec::new();
        let d = dynamize_layout(&mut rng, &template, 50, &mut values);
        let env = env_of(&values);
        for i in 0..template.size() {
            let c = d.get_hier_coord(i).map(&mut |e| e.eval(&env));
            assert_eq!(template.offset(&Coord::from(&c)), st(i), "{d} at {i}: {c}");
        }
    }
}

#[test]
fn by_mode_logical_products() {
    let block = lay("(2,3):(1,2)");
    // A layout is the whole tiler.
    assert_eq!(logical_product(&block, lay("4:1")), logical_product(&block, Tiler::from(&lay("4:1"))));
    // An integer is n:1, and `..` keeps the mode.
    shows(&logical_product(&block, (4, ..)), "((2,4),3):((1,2),2)");
    shows(&logical_product(&block, (.., 2)), "(2,(3,2)):(1,(2,1))");
    shows(&logical_product(&lay("((2,2),3):((1,2),4)"), ((2, ..), ..)), "(((2,2),2),3):(((1,2),2),4)");
    // A runtime tile extent.
    let p = logical_product(&block, Tiler::from((dv(0), ..)));
    shows(&p, "((2,?),3):((1,2),2)");
}

#[test]
fn property_signed_expressions_evaluate_like_i64() {
    let mut rng = Rng(0x5167);
    for _ in 0..500 {
        let values: Vec<i64> = (0..3).map(|_| (rng.next() % 41) as i64 - 20).collect();
        let env = |v: ValueId| values[(v.0 - 1000) as usize];
        let leaf = |rng: &mut Rng| -> (IndexExpr, i64) {
            match rng.next() % 3 {
                0 => {
                    let i = (rng.next() % 3) as u32;
                    (IndexExpr::Signed(ValueId(1000 + i)), values[i as usize])
                }
                1 => {
                    let c = (rng.next() % 21) as i64 - 10;
                    (IndexExpr::from(c), c)
                }
                _ => {
                    let c = (rng.next() % 9) as usize;
                    (st(c), c as i64)
                }
            }
        };
        let (mut e, mut want) = leaf(&mut rng);
        for _ in 0..4 {
            let (x, xv) = leaf(&mut rng);
            match rng.next() % 4 {
                0 => (e, want) = (e + x, want + xv),
                // A difference of two provably non-negative values must not
                // be negative, so those add a negation instead.
                1 if e.is_non_negative() && x.is_non_negative() => (e, want) = (e + -x, want - xv),
                1 => (e, want) = (e - x, want - xv),
                2 => (e, want) = (e * x, want * xv),
                _ => (e, want) = (-e, -want),
            }
        }
        assert_eq!(e.eval_signed(&env), want, "{e:?}");
        if let Some(c) = e.as_static_signed() {
            assert_eq!(c, want, "{e:?} folds to {c}");
        }
        if e.is_non_negative() {
            assert!(want >= 0, "{e:?} claims to be non-negative but is {want}");
        }
    }
    // Folds of negative constants.
    assert_eq!(IndexExpr::from(-3) + 5usize, st(2));
    assert_eq!((dv(0) + IndexExpr::from(-3)) + 3usize, dv(0));
    assert_eq!(dv(0) - dv(0), st(0));
    assert_eq!(IndexExpr::Signed(ValueId(7)) - IndexExpr::Signed(ValueId(7)), st(0));
    assert!(!IndexExpr::from(-1).is_non_negative() && !(dv(0) - IndexExpr::Signed(ValueId(7))).is_non_negative());
    assert_eq!(IndexExpr::from(-1) * -2, st(2));
    // A bound survives adding a negative constant.
    assert_eq!((dv(0) % 8usize + IndexExpr::from(-3)).bound(), Some(5));
}

// ---------------------------------------------------------------------------
// Basis layouts as tilers of logical_divide, their left and right inverses,
// and the complement with no cotarget. "cutegen" marks the vectors of
// cg_logical_divide_test.cpp, cg_left_inverse_test.cpp,
// cg_right_inverse_test.cpp and cg_complement_test.cpp in
// cutlass_compiler/cutegen/test.

#[test]
fn basis_tilers_in_logical_divide() {
    let t = lay("(8,8):(9,1)");
    // cutegen: LogicalDivideTest's basis-stride tilers.
    for (tiler, want) in [
        ("4:1@0", "(4,(2,8)):(9,(36,1))"),
        ("4:1@1", "(4,(8,2)):(1,(9,4))"),
        ("4:2@0", "(4,(2,8)):(18,(9,1))"),
        ("4:2@1", "(4,(8,2)):(2,(9,1))"),
        ("(4,4):(1@1,2@0)", "((4,4),(2,2)):((1,18),(9,4))"),
        ("(5,7):(3@1,2@0)", "((5,7),(2,3)):((3,18),(9,1))"),
    ] {
        assert_cute_eq(&logical_divide(&t, lay(tiler)), want);
    }
    // cutegen: ScaledBasis and ScaledBasisLowerRankTiler (zipped).
    // (cutegen gives the mode of extent 1 the stride 64@1, which never
    // matters.)
    shows(
        &zipped_divide(&lay("(64,(64,2)):(1@1,(1@0,64@1))"), lay("(64,(16,2)):(1@0,(1@1,64@1))")),
        "((64,(16,2)),(1,4)):((1@1,(1@0,64@1)),(0,16@0))",
    );
    shows(&zipped_divide(&lay("(128,64):(1@1,1@0)"), lay("2:64@0")), "(2,(64,64)):(64@1,(1@1,1@0))");
    // cutegen: ScaledBasisIdentityTilerStaticEquivalence, an identity-layout
    // tiler zipped-divides as its shape does.
    let target = lay("(256,384):(1,256)");
    let by_identity = zipped_divide(&target, make_identity_layout((128, 128)));
    assert_cute_eq(&by_identity, &zipped_divide(&target, (128, 128)).to_string());
    let id = make_identity_layout((4, 2));
    assert_cute_eq(&zipped_divide(&t, id), &zipped_divide(&t, (4, 2)).to_string());
}

#[test]
fn inverses_of_basis_layouts() {
    // cutegen: the basis cases of RightInverseTest and LeftInverseTest.
    for (layout, right, left) in [
        ("(4,5):(1@0,1@1)", "(4,5):(1,4)", "(4,5):(1,4)"),
        ("(4,5):(1@1,1@0)", "(5,4):(4,1)", "(5,4):(4,1)"),
        ("(4,5):(1@1,1@1@4)", "(1,4,1,1,(1,5)):(0,1,0,0,(0,4))", "(1,4,1,1,(1,5)):(0,1,0,0,(0,4))"),
        ("(4,5):(2@0,2@1)", "(1,1):(0,0)", "((2,4),(2,5)):((0,1),(0,4))"),
        (
            "((2,2,2),(2,2,2)):((1@0,2@1,4@0),(1@1,2@0,4@1))",
            "((2,2,2),(2,2,2)):((1,16,4),(8,2,32))",
            "((2,2,2),(2,2,2)):((1,16,4),(8,2,32))",
        ),
        ("((4,2),4):((1@1,4@0),1@0)", "((4,2),4):((8,4),1)", "((4,2),4):((8,4),1)"),
        ("((4,8),(2,2)):((2@1,1@0),(1@1,8@0))", "((8,2),(2,4)):((4,64),(32,1))", "((8,2),(2,4)):((4,64),(32,1))"),
        ("(4,1,5):(1@0,0,1@1)", "(4,5):(1,4)", "(4,5):(1,4)"),
        // cutegen's right-inverse test expects (1,16):(0,1) here, though its
        // comment repeats the left inverse, ((2,64),(32,4)):((0,64),(1,16)).
        ("(((16,4),64),1,1):(((1@1,32@1),2@0),0,0)", "(1,16):(0,1)", "((2,64),(32,4)):((0,64),(1,16))"),
    ] {
        let l = lay(layout);
        assert_cute_eq(&l.right_inverse(), right);
        assert_cute_eq(&l.left_inverse(), left);
    }
    // cutegen: strides that do not chain shorten the right inverse, and
    // leave gaps in the left one.
    let l = lay("((2,2,2),(2,2,2)):((1@0,2@1,5@0),(1@1,2@0,5@1))");
    assert_cute_eq(&l.right_inverse(), "((2,2),(2,2)):((1,16),(8,2))");
    let l = lay("((2,2,2),(2,2,2)):((1@0,2@1,5@0),(1@1,2@0,4@1))");
    assert_cute_eq(&l.right_inverse(), "((2,2),(2,2,2)):((1,16),(8,2,32))");
    let l = lay("((2,2,2),(2,2,2)):((1@0,2@1,6@0),(1@1,2@0,6@1))");
    assert_cute_eq(&l.left_inverse(), "((2,3,2),(2,3,2)):((1,16,4),(8,2,32))");
    let l = lay("((2,2,2),(2,2,2)):((1@0,2@1,6@0),(1@1,2@0,4@1))");
    assert_cute_eq(&l.left_inverse(), "((2,3,2),(2,2,2)):((1,16,4),(8,2,32))");
    // The left inverse maps every tuple the layout gives back to its index.
    for s in ["((4,2),4):((1@1,4@0),1@0)", "(4,5):(2@0,2@1)", "((4,8),(2,2)):((2@1,1@0),(1@1,8@0))"] {
        let (l, inv) = (lay(s), lay(s).left_inverse());
        for i in 0..l.size() {
            let c = l.eval(&Tree::Leaf(st(i)));
            assert_eq!(inv.offset(&c), st(i), "{l} at {i}: {c}");
        }
    }
}

#[test]
fn complement_with_no_cotarget() {
    // cutegen: ComplementTest.Basic, with no cotarget.
    for (layout, want) in [
        ("1:0", "1:1"),
        ("1:2", "1:1"),
        ("(1,1):(0,0)", "1:1"),
        ("5:1", "1:5"),
        ("5:3", "(3,1):(1,15)"),
        ("4:0", "1:1"),
        ("4:2", "(2,1):(1,8)"),
        ("4:4", "(4,1):(1,16)"),
        ("(2,4):(1,2)", "1:8"),
        ("(2,3):(1,2)", "1:6"),
        ("(2,4):(1,4)", "(2,1):(2,16)"),
        ("(8,4):(4,1)", "1:32"),
        ("(2,4,6):(4,1,8)", "1:48"),
        ("(2,4,8):(8,1,64)", "(2,4,1):(4,16,512)"),
        ("(2,4,8):(32,0,2)", "(2,2,1):(1,16,64)"),
        ("(2,4,4,4,2):(32,0,2,0,512)", "(2,4,8,1):(1,8,64,1024)"),
        ("((2,2),(2,2)):((1,4),(8,32))", "(2,2,1):(2,16,64)"),
        ("(2,(3,4)):(3,(1,6))", "1:24"),
        ("(4,2):(1,16)", "(4,1):(4,32)"),
        ("3:1@0", "(1):(3@0)"),
        ("3:4@2", "(1,1,(4,1)):(1@0,1@1,(1@2,12@2))"),
        ("(2,5,3):(4@1,5@0,16@1)", "((5,1),(4,2,1)):((1@0,25@0),(1@1,8@1,48@1))"),
        ("(2,1,5,3):(4@1,0,5@0,16@1)", "((5,1),(4,2,1)):((1@0,25@0),(1@1,8@1,48@1))"),
        ("(2,3,5):(4@1,5@0,7@1@2)", "((5,1),(4,1),(1,(7,1))):((1@0,15@0),(1@1,8@1),(1@0@2,(1@1@2,35@1@2)))"),
        ("(2,3,5):(4@1,0,7@1@2)", "(1,(4,1),(1,(7,1))):(1@0,(1@1,8@1),(1@0@2,(1@1@2,35@1@2)))"),
        // The weak cases, whose strides are not multiples of each other.
        ("(4,2):(1,6)", "1:12"),
        ("(4,2):(1,10)", "(2,1):(4,20)"),
        ("(2,4):(11,1)", "(2,1):(4,22)"),
    ] {
        let c = lay(layout).complement_no_cotarget();
        if c.has_scaled_basis() { shows(&c, want) } else { assert_cute_eq(&c, want) }
    }
}

// recast_layout: the dialect's examples, vectors worked from
// include/cute/layout.hpp and include/cute/swizzle_layout.hpp, and
// properties.

#[test]
fn recast_layout_dialect_examples() {
    // Upcast x4, downcast x4 and the identity (dialect).
    assert_eq!(recast_layout(32, 8, &l((32, 4), (1, 32))), l((8, 4), (1, 8)));
    assert_eq!(recast_layout(8, 32, &l((8, 4), (1, 8))), l((32, 4), (1, 32)));
    assert_eq!(recast_layout(32, 32, &l((8, 4), (1, 8))), l((8, 4), (1, 8)));
    // A swizzle's base moves by log2 of the ratio (dialect: S<3,5,4> o 0 o
    // (32,?):(1,?) upcast 32/8 is S<3,3,4> o 0 o (8,?):(1,?)).
    let src: Layout = "(32,?):(1,?)".parse().unwrap();
    let composed = make_composed_layout(Swizzle::new(3, 5, 4), 0, src);
    let r = recast_layout(32, 8, &composed);
    assert_eq!(r.swizzle().unwrap(), Swizzle::new(3, 3, 4));
    assert_eq!(r.outer.to_string(), "(8,?):(1,?)");
    // A downcast of a dynamic layout keeps its runtime leaves runtime.
    let dynamic: Layout = "(?,4):(1,?)".parse().unwrap();
    // (?,4):(1,?), the runtime leaves now known to be multiples of 4.
    assert_eq!(recast_layout(8, 32, &dynamic).to_string(), "(?{div=4},4):(1,?{div=4})");
}

#[test]
fn recast_layout_worked_vectors() {
    // Upcast by 4 of a stride that divides 4: extent ceil_div(6, 4 / 2) = 3,
    // stride ceil_div(2, 4) = 1.
    assert_eq!(l((6, 5), (2, 12)).upcast_by(4), l((3, 5), (1, 3)));
    // Odd extents round up; stride 0 stays; a negative stride keeps its sign.
    let signed = |strides: [i32; 3]| -> Stride {
        Tree::Branch(strides.iter().map(|&d| Tree::Leaf(StrideLeaf::Int(IndexExpr::from(d)))).collect())
    };
    let a = Layout::new((7, 3, 2), signed([1, 0, -8]));
    assert_eq!(a.upcast_by(2), Layout::new((4, 3, 2), signed([1, 0, -4])));
    // The gcd case: 24-bit to 16-bit elements is an upcast by 2 then a
    // downcast by 3.
    assert_eq!(recast_layout(16, 24, &l((12, 4), (1, 12))), l((18, 4), (1, 18)));
    // Nested modes keep their profile.
    assert_eq!(recast_layout(16, 8, &l(((2, 8), 3), ((1, 2), 16))), l(((1, 8), 3), ((1, 1), 8)));
    // Downcast multiplies every non-unit stride, and the stride -1 extent.
    let signed2 = |strides: [i32; 2]| -> Stride {
        Tree::Branch(strides.iter().map(|&d| Tree::Leaf(StrideLeaf::Int(IndexExpr::from(d)))).collect())
    };
    assert_eq!(Layout::new((4, 3), signed2([-1, 4])).downcast(2), Layout::new((8, 3), signed2([-1, 8])));
    // Swizzles: an upcast past the base shrinks the bits (CuTe clamps B).
    assert_eq!(Swizzle::new(3, 1, 3).upcast(8), Swizzle::new(1, 0, 3));
    assert_eq!(Swizzle::new(2, 1, 3).upcast(16), Swizzle::new(0, 0, 3));
    assert_eq!(Swizzle::new(3, 3, 3).downcast(2), Swizzle::new(3, 4, 3));
    let c = make_composed_layout(Swizzle::new(3, 3, 3), 0, Layout::row_major((8, 64)));
    let d = recast_layout(8, 16, &c);
    assert_eq!((d.swizzle().unwrap(), d.outer.clone()), (Swizzle::new(3, 4, 3), l((8, 128), (128, 1))));
    assert_eq!(recast_layout(16, 8, &d), c);
}

#[test]
#[should_panic(expected = "divisibility condition")]
fn recast_layout_checks_divisibility() {
    let _ = l((4, 3), (1, 6)).upcast_by(4);
}

#[test]
#[should_panic(expected = "needs a mode of stride 1")]
fn downcast_needs_adjacent_elements() {
    let _ = l((4, 3), (2, 8)).downcast(2);
}

#[test]
#[should_panic(expected = "basis stride")]
fn recast_layout_refuses_basis_strides() {
    let _ = recast_layout(16, 8, &make_identity_layout((4, 4)));
}

/// Downcasting multiplies the size and the cosize by the ratio, and an
/// upcast by it undoes a downcast; a recast there and back is the identity
/// when the strides allow it.
#[test]
fn recast_layout_properties() {
    let shapes = [vec![4usize, 3], vec![2, 8, 3], vec![16], vec![6, 4, 2]];
    for shape in shapes {
        for r in [2usize, 4, 8] {
            // Column-major and with the unit mode last.
            let col = Layout::col_major(IntTuple::from(shape.clone()));
            let row = Layout::row_major(IntTuple::from(shape.clone()));
            for a in [col, row] {
                let d = a.downcast(r);
                assert_eq!(d.size(), a.size() * r, "{a} by {r}");
                assert_eq!(d.cosize(), a.cosize() * r, "{a} by {r}");
                assert_eq!(d.upcast_by(r), a, "{a} by {r}");
                assert_eq!(recast_layout(8 * r, 8, &recast_layout(8, 8 * r, &a)), a, "{a} by {r}");
                // Element i of the narrow view is byte i % r of wide element
                // i / r, for the unit mode.
                let unit = Layout::col_major(shape[0]);
                let du = unit.downcast(r);
                for i in 0..du.size() {
                    assert_eq!(du.call(i), unit.call(i / r) * r + i % r);
                }
            }
        }
    }
}

// Layout-like helpers. "cutegen" marks the vectors of cg_layout_test.cpp,
// cg_composed_layout_test.cpp and cg_swizzle_test.cpp; "layout.hpp" the
// example in the comment above C++ make_layout_like.

/// `make_layout_like` keeps the hierarchy and static-0 strides, orders the
/// other modes by their strides, and counts a runtime stride as larger
/// than every static one.
#[test]
fn make_layout_like_follows_cute() {
    let lay = |sh: Shape, d: Stride| Layout::new(sh, d);
    // cutegen MakeLayoutLike.
    assert_eq!(make_layout_like(&l((2, 2, 2, 2), (0, 2, 4, 1))), l((2, 2, 2, 2), (0, 2, 4, 1)));
    // layout.hpp and cutegen: (2,3,4,5):(_0,42,_1,_0) -> (2,3,4,5):(_0,4,_1,_0).
    assert_eq!(make_layout_like(&l((2, 3, 4, 5), (0, 42, 1, 0))), l((2, 3, 4, 5), (0, 4, 1, 0)));
    let k = dv(0);
    let stride = |v: Vec<StrideLeaf>| Stride::Branch(v.into_iter().map(Tree::Leaf).collect());
    let i = |v: usize| StrideLeaf::Int(st(v));
    let runtime = lay(Shape::from((2, 3, 4, 5)), stride(vec![i(0), StrideLeaf::Int(k.clone()), i(1), i(0)]));
    assert_eq!(make_layout_like(&runtime), l((2, 3, 4, 5), (0, 4, 1, 0)));
    // cutegen: nested modes with a runtime stride.
    let nested = Layout::new(
        Shape::from(((2, 3), 4, 5)),
        Stride::Branch(vec![stride(vec![i(0), StrideLeaf::Int(k.clone())]), Tree::Leaf(i(1)), Tree::Leaf(i(0))]),
    );
    assert_eq!(make_layout_like(&nested), l(((2, 3), 4, 5), ((0, 4), 1, 0)));
    let deeper = Layout::new(
        Shape::from(((4, (2, 3)), 5)),
        Stride::Branch(vec![
            Stride::Branch(vec![Tree::Leaf(i(1)), stride(vec![i(0), StrideLeaf::Int(k.clone())])]),
            Tree::Leaf(i(0)),
        ]),
    );
    assert_eq!(make_layout_like(&deeper), l(((4, (2, 3)), 5), ((1, (0, 4)), 0)));
    // cutegen ComposedLayoutTest.MakeLayoutLike: the layout before the swizzle.
    let composed = make_composed_layout(make_swizzle(3, 5, 4), 0, l((2, 3), (3, 1)));
    assert_eq!(make_layout_like(&composed), l((2, 3), (3, 1)));
    // Two runtime strides keep their left-to-right order; a size-1 mode gets
    // stride 0, as CuTe's compact strides give it.
    let two = Layout::new((4, 8, 1), (dv(1), dv(2), 3));
    assert_eq!(make_layout_like(&two), l((4, 8, 1), (1, 4, 0)));
    // Runtime extents make the larger strides runtime products.
    let m = dv(3);
    let rows = Layout::new((m.clone(), 8), (8, 1));
    assert_eq!(make_layout_like(&rows), Layout::new((m.clone(), 8), (8, 1)));
    let cols = Layout::new((8, m.clone()), (1, 8));
    assert_eq!(make_layout_like(&Layout::new((m.clone(), 8), (1, 64))), Layout::new((m.clone(), 8), (1, m.clone())));
    assert_eq!(make_layout_like(&cols), cols);
}

/// C++ `make_fragment_like(layout)`: mode 0 compact column-major (keeping
/// its zeros), the other modes ordered by their strides after it.
#[test]
fn make_fragment_like_layout_follows_cute() {
    assert_eq!(make_fragment_like_layout(&l((4, 8), (8, 1))), l((4, 8), (1, 4)));
    assert_eq!(make_fragment_like_layout(&l(((2, 4), 3, 5), ((0, 1), 40, 2))), l(((2, 4), 3, 5), ((0, 1), 20, 4)));
    // Rest modes of equal order share a stride, as C++ compact_order gives.
    assert_eq!(make_fragment_like_layout(&l((4, 2, 3), (1, 0, 0))), l((4, 2, 3), (1, 4, 4)));
    // Rank 1, or a runtime extent: the compact column-major layout.
    assert_eq!(make_fragment_like_layout(&l(8, 4)), l(8, 1));
    assert_eq!(make_fragment_like_layout(&l((2, 4), (4, 1)).group(0, 2)), l(((2, 4),), ((1, 2),)));
    let m = dv(0);
    assert_eq!(make_fragment_like_layout(&Layout::new((8, m.clone()), (m.clone(), 1))), Layout::col_major((8, m)));
    // A composed layout gives its layout's.
    let composed = make_composed_layout(make_swizzle(2, 2, 3), 0, l((8, 4), (4, 1)));
    assert_eq!(make_fragment_like_layout(&composed), l((8, 4), (1, 8)));
}

/// C++ `max_alignment` of a layout, a swizzle and a composed layout.
#[test]
fn max_alignment_follows_cute() {
    // cutegen LayoutTest.MaxAlignment.
    assert_eq!(l((1, 8), (1, 1)).max_alignment(), 8);
    assert_eq!(l((4, 2, (8, 2)), (512, 1, (8, 8192))).max_alignment(), 2);
    assert_eq!(l(((1, (2,)),), ((1, (0,)),)).max_alignment(), 1);
    assert_eq!(l((8, 16), (1, 4)).max_alignment(), 4);
    assert_eq!(l((8, 16), (1, 6)).max_alignment(), 2);
    assert_eq!(make_composed_layout(make_swizzle(3, 4, 4), 0, l((32, 1), (1, 32))).max_alignment(), 16);
    // cutegen SwizzleTest.
    assert_eq!(make_swizzle(3, 4, 3).max_alignment(), 16);
    assert_eq!(Swizzle::new(0, 1, 2).max_alignment(), 2);
    // make_swizzle canonicalises the identity to S<0,4,3>, as CuTe DSL does.
    assert_eq!(make_swizzle(0, 1, 2), Swizzle::new(0, 4, 3));
    // Compact layouts are aligned to their size; a row-major tile to all
    // of it. A runtime stride or extent counts with what divides it, which
    // for these unconstrained values is 1.
    assert_eq!(Layout::col_major((4, 8)).max_alignment(), 32);
    assert_eq!(Layout::row_major((4, 8)).max_alignment(), 32);
    assert_eq!(l(8, 2).max_alignment(), 1);
    assert_eq!(Layout::new((8, 128), (dv(0), 1)).max_alignment(), 1);
    // Scaled-basis strides give 1.
    assert_eq!(make_identity_layout((4, 8)).max_alignment(), 1);
    // Coalescing merges (4,?):(1,4) into (4*?):1, as C++ coalesce merges
    // it; the runtime extent 4*? is a multiple of 4 (C++ CuTe, which takes
    // it as 1, gives 1).
    assert_eq!(Layout::new((4, dv(1)), (1, 4)).max_alignment(), 4);
    assert_eq!(Layout::new((4, dv(1)), (1, 8)).max_alignment(), 4);
}

/// `max_common_layout` and `max_common_vector` with a runtime row stride:
/// the brief's worked case, and the postcondition `a(b.get_1d_coord(n))`.
#[test]
fn max_common_with_runtime_strides() {
    let k = dv(0);
    let a = Layout::with_strides((8, 128), (k.clone(), 1));
    let b = l((8, 128), (128, 1));
    assert_eq!(max_common_vector(&a, &b), 128);
    let common = max_common_layout(&a, &b);
    assert_eq!(common, l(128, 8));
    for n in 0..128 {
        assert_eq!(b.call(common.call(n)), n);
        assert_eq!(a.offset_at(common.call(n)), a.offset_at(common.call(0)) + st(n));
    }
    // Two runtime strides: the inverse skips them.
    let c = Layout::with_strides((4, 8, 2), (1, k.clone(), dv(1)));
    assert_eq!(max_common_layout(&c, &Layout::col_major((4, 8, 2))), l(4, 1));
    // The fragment make_layout_like gives copies the whole tile at once.
    assert_eq!(max_common_vector(&a, &make_layout_like(&a)), 128 * 8 / 8);
}

// ---------------------------------------------------------------------------
// General composed layouts (cutegen's cg_composed_layout_test.cpp and the
// cute_dialect.rst examples).
// ---------------------------------------------------------------------------

fn composed(s: &str) -> ComposedLayout {
    s.parse().unwrap_or_else(|e| panic!("{s}: {e}"))
}

/// cutegen ComposedLayoutTest.ToFromString: CuTe's printed form, read back.
#[test]
fn composed_layouts_print_and_parse() {
    let aob = make_composed_layout(Layout::col_major((4, 5)), 2, Layout::col_major((2, 3)));
    assert_eq!(aob.to_string(), "(4,5):(1,4) o 2 o (2,3):(1,2)");
    assert_eq!(composed("(4,5):(1,4) o 2 o (2,3):(1,2)"), aob);
    let sw = make_composed_layout(make_swizzle(3, 5, 4), 2, Layout::col_major((2, 3)));
    assert_eq!(sw.to_string(), "S<3,5,4> o 2 o (2,3):(1,2)");
    assert_eq!(composed("S<3,5,4> o 2 o (2,3):(1,2)"), sw);
    let basis = make_composed_layout(Layout::col_major((4, 5)), (0, 0), make_identity_layout((2, 3)));
    assert_eq!(basis.to_string(), "(4,5):(1,4) o (0,0) o (2,3):(1@0,1@1)");
    assert_eq!(composed("(4,5):(1,4) o (0,0) o (2,3):(1@0,1@1)"), basis);
    // Dynamic leaves, as cutegen's CountDynamicScalars reads them.
    assert_eq!(composed("S<3,5,4> o ? o (2,?):(1,?)").to_string(), "S<3,5,4> o ? o (2,?):(1,?)");
    // Swizzles print as CuTe's S<b,m,s>, a negative shift too.
    assert_eq!("S<2,4,-3>".parse::<Swizzle>().unwrap(), Swizzle::new(2, 4, -3));
    assert_eq!(Swizzle::new(2, 4, -3).to_string(), "S<2,4,-3>");
    for bad in ["S<3,4,2>", "S<3,4>", "S<3,4,3> o 0 o (2,3):(1@0,1@1)", "S<3,4,3> o (0,0) o 8:1"] {
        assert!(bad.parse::<ComposedLayout>().is_err() || bad.parse::<Swizzle>().is_err(), "{bad}");
    }
}

/// cutegen SliceStaticSwizzledLayouts: a slice the swizzle acts on affinely
/// decays to a normal layout, one it does not keeps the swizzle with the
/// slice's offset moved inside.
#[test]
fn slicing_swizzled_layouts() {
    let master = make_composed_layout(make_swizzle(3, 4, 4), 0, Layout::col_major((64, 128)));
    let (sliced, offset) = master.slice_and_offset((.., 16));
    assert!(sliced.is_normal());
    let off = offset.as_leaf().unwrap().as_static().unwrap();
    for i in 0..64 {
        assert_eq!(sliced.call(i), sliced.outer.call(i));
        assert_eq!(master.offset_of(&(i, 16).into()).as_static().unwrap(), off + sliced.outer.call(i));
    }
    let (sliced, offset) = master.slice_and_offset((32, ..));
    assert!(!sliced.is_normal());
    assert_eq!(sliced.to_string(), "S<3,4,4> o 32 o (128):(64)");
    assert_eq!(offset, Tree::Leaf(st(0)));
    for i in 0..128 {
        assert_eq!(master.offset_of(&(32, i).into()).as_static().unwrap(), sliced.call(i));
    }
    let l = composed("S<2,1,2> o 0 o ((2,4),(2,2)):((1,8),(2,4))");
    let (sl, o) = l.slice_and_offset(((.., 3), ..));
    assert!(sl.is_normal());
    assert_eq!(o, Tree::Leaf(st(30)));
    let want = Layout::new(
        (2, 4),
        (Tree::Leaf(StrideLeaf::Int(st(1))), Tree::Leaf(StrideLeaf::Int(IndexExpr::from_signed(-2)))),
    );
    assert_eq!(sl.outer.coalesce(), want);
    let c: SliceCoord = ((0, ..), (.., 0)).into();
    let (sl, o) = l.slice_and_offset(c.clone());
    assert!(!sl.is_normal());
    assert_eq!((sl.inner.clone(), sl.offset.clone(), o), (l.inner.clone(), Tree::Leaf(st(0)), Tree::Leaf(st(0))));
    assert_eq!(sl.outer, l.outer.slice(&c).0);
}

/// cutegen SliceAffineLayouts: a layout inner keeps the offset inside, and
/// a coordinate with no `_` gives the value.
#[test]
fn slicing_layout_inner_composed_layouts() {
    let lay = make_composed_layout(Layout::new((3, 8), (1, 3)), 0, Layout::new((2, 3), (3, 1)));
    let (slice, offset) = lay.slice_and_offset((0, 2));
    assert_eq!(
        (slice.inner.clone(), slice.offset.clone(), offset),
        (lay.inner.clone(), Tree::Leaf(st(2)), Tree::Leaf(st(2)))
    );
    let (slice, offset) = lay.slice_and_offset((1, ..));
    assert_eq!((slice.offset.clone(), offset), (Tree::Leaf(st(3)), Tree::Leaf(st(0))));
    assert_eq!(slice.outer.to_string(), "(3):(1)");
    // slice_ (CuTe DSL) moves the offset in too.
    assert_eq!(lay.slice((1, ..)), slice);
}

/// cutegen Get, Select, MakeLayoutLike, Upcast, Downcast and CoSize.
#[test]
fn composed_layout_modes_recasts_and_cosize() {
    let a = make_swizzle(3, 5, 4);
    let aob = make_composed_layout(a, 2, Layout::col_major((2, 3)));
    assert_eq!(aob.get(&[0]), make_composed_layout(a, 2, Layout::new(2, 1)));
    assert_eq!(basis_get(&E(0), &aob), aob.get(&[0]));
    assert_eq!(aob.select(&[0]).mode(0), make_composed_layout(a, 2, Layout::new(2, 1)));
    assert_eq!(aob.select(&[0, 1]), aob);
    assert_eq!(aob.select(&[]).rank(), 0);
    let aob = make_composed_layout(a, 2, Layout::new((2, 3), (3, 1)));
    assert_eq!(make_layout_like(&aob), Layout::new((2, 3), (3, 1)));
    let lay = composed("8:1 o 6 o (2,4):(1,2)");
    assert_eq!(recast_layout(2, 1, &lay).to_string(), "4:1 o 3 o (1,4):(1,1)");
    assert_eq!(recast_layout(1, 2, &lay).to_string(), "16:1 o 12 o (4,4):(1,4)");
    assert_eq!(recast_layout(1, 2, &composed("S<3,4,3> o 6 o (2,4):(1,2)")).to_string(), "S<3,5,3> o 12 o (4,4):(1,4)");
    let outer = Layout::new((10, (10, 10), 10), (10, (1, 200), 2000));
    let lay = make_composed_layout(a, 2, outer.clone());
    assert_eq!(cosize(&lay, &[]), cosize(&outer, &[]));
    assert_eq!(lay.cosize(), outer.cosize());
    let affine = make_composed_layout(Layout::col_major(16), 2, outer);
    assert!(std::panic::catch_unwind(|| affine.cosize()).is_err(), "a layout inner has no cosize");
    // A normal composed layout decays to its outer layout.
    let normal = make_composed_layout(make_swizzle(0, 0, 0), 0, Layout::col_major((4, 4)));
    assert_eq!(normal.as_layout(), Some(&Layout::col_major((4, 4))));
    assert_eq!(normal.inner, Inner::Swizzle(Swizzle::new(0, 4, 3)), "make_swizzle(0, ..) is S<0,4,3>");
    assert_eq!(make_composed_layout(a, 0, Layout::col_major(4)).as_layout(), None);
}

/// Swizzles with a negative shift XOR low bits into high ones; every
/// swizzle is an involution that keeps its period's blocks, and the byte and
/// element forms convert by the element's width.
#[test]
fn swizzle_shifts_and_units() {
    let sw = Swizzle::new(1, 0, -1);
    assert_eq!((sw.y_mask(), sw.z_mask(), sw.apply(1), sw.apply(2)), (1, 2, 3, 2));
    for sw in [Swizzle::new(3, 4, 3), Swizzle::new(2, 4, -3), Swizzle::new(3, 1, -4), Swizzle::new(1, 2, 5)] {
        let period = sw.period();
        for i in 0..2 * period {
            assert_eq!(sw.apply(sw.apply(i)), i, "{sw} at {i}");
            assert_eq!(sw.apply(i) / period, i / period, "{sw} at {i}");
            assert_eq!(sw.apply(i) % (1 << sw.base), i % (1 << sw.base), "{sw} at {i}");
        }
    }
    assert_eq!(make_swizzle(3, 4, 3).to_elements(16), Ok(make_swizzle(3, 3, 3)));
    assert_eq!(make_swizzle(3, 4, 3).to_elements(8), Ok(make_swizzle(3, 4, 3)));
    assert_eq!(make_swizzle(3, 4, 3).to_elements(4), Ok(make_swizzle(3, 5, 3)));
    assert!(make_swizzle(3, 1, 3).to_elements(32).is_err(), "finer than a 4-byte element");
    assert!(make_swizzle(3, 4, 3).to_elements(6).is_err());
    assert_eq!(make_swizzle(3, 3, 3).to_bytes(16), make_swizzle(3, 4, 3));
    // For 2^k-byte elements, Swizzle<B,M,S> on elements is Swizzle<B,M+k,S>
    // on bytes.
    let (elem, bytes) = (make_swizzle(2, 3, 3), make_swizzle(2, 4, 3));
    for i in 0..1024 {
        assert_eq!(elem.apply(i) * 2, bytes.apply(i * 2));
    }
    // The swizzle portions.
    let c = make_composed_layout(make_swizzle(3, 4, 3), 0, Layout::row_major((8, 64)));
    assert_eq!(
        (get_swizzle_portion(&c), get_nonswizzle_portion(&c)),
        (make_swizzle(3, 4, 3), Layout::row_major((8, 64)))
    );
    let l = Layout::col_major(8);
    assert_eq!((get_swizzle_portion(&l), get_nonswizzle_portion(&l)), (Swizzle::new(0, 4, 3), l.clone()));
    let affine = make_composed_layout(l.clone(), 0, l.clone());
    assert_eq!((get_swizzle_portion(&affine), get_nonswizzle_portion(&affine)), (Swizzle::identity(), l));
}

/// Brute force: the algebra on a composed layout acts on its outer layout,
/// so composition, the divides, slicing and the products evaluate like
/// `inner(offset + outer(c))` with the outer layout's own results.
#[test]
fn composed_layout_algebra_brute_force() {
    let inners: Vec<Inner> =
        vec![make_swizzle(2, 2, 3).into(), Swizzle::new(2, 4, -3).into(), Layout::new((8, 64), (64, 1)).into()];
    for inner in inners {
        let c = make_composed_layout(inner.clone(), 4, Layout::row_major((8, 32)));
        let f = |i: usize| c.call(i);
        // Composition: (c o b)(i) = c(b(i)).
        let b = Layout::new((4, 8), (2, 32));
        let cb = composition(&c, &b);
        assert!((0..b.size()).all(|i| cb.call(i) == f(b.call(i))), "{c} o {b}");
        // Divides: the divide of c by a tiler is c composed with the divide
        // of the identity of its domain.
        let id = Layout::col_major(c.outer.shape().clone());
        for tiler in [
            Tiler::from(Layout::new(4, 1)),
            Tree::Branch(vec![Tree::Leaf(Some(Layout::new(2, 1))), Tree::Leaf(Some(Layout::new(8, 4)))]),
        ] {
            let d = logical_divide(&c, tiler.clone());
            let di = logical_divide(&id, tiler.clone());
            assert!((0..d.size()).all(|i| d.call(i) == f(di.call(i))), "{c} / {tiler:?}");
            let z = zipped_divide(&c, tiler.clone());
            let zi = zipped_divide(&id, tiler);
            assert!((0..z.size()).all(|i| z.call(i) == f(zi.call(i))));
        }
        // Slicing: the slice at the kept coordinates is c at the full one.
        for r in 0..8 {
            let s = c.slice((r, ..));
            assert!((0..32).all(|j| s.call(j) == c.offset_of(&(r, j).into()).as_static().unwrap()), "{c} at row {r}");
            let (s, o) = c.slice_and_offset((r, ..));
            let o = o.as_leaf().unwrap().as_static().unwrap();
            assert!(
                (0..32).all(|j| o + s.call(j) == c.offset_of(&(r, j).into()).as_static().unwrap()),
                "{c} at row {r}"
            );
        }
        // Products and tile_to_shape keep the inner and the offset over the
        // outer layout's product.
        let t = Layout::col_major((2, 2));
        for (p, q) in [
            (blocked_product(&c, &t), blocked_product(&c.outer, &t)),
            (raked_product(&c, &t), raked_product(&c.outer, &t)),
            (logical_product(&c, t.clone()), logical_product(&c.outer, t.clone())),
            (tile_to_shape(&c, (16, 64)), tile_to_shape(&c.outer, (16, 64))),
            (coalesce(&c), coalesce(&c.outer)),
        ] {
            assert_eq!((p.inner.clone(), p.offset.clone(), p.outer.clone()), (c.inner.clone(), c.offset.clone(), q));
        }
        assert_eq!(crd2idx((3, 5), &c), c.offset_of(&(3, 5).into()));
        assert_eq!(c.group_modes(0, 2).rank(), 1);
    }
}

/// Calls recorded from CuTe DSL 4.8.1 (gpu-results/cutedsl-check-20261126/
/// layouts.tsv) that cute-rs used to answer differently or refuse: the
/// strides of extent-1 modes in composition and the operations built on it,
/// C++ CuTe's `coalesce_x` of the left-hand side, its weak stride condition
/// where the result is exact, the floor division of `complement`, the
/// complement in the divided layout's shape, `left_inverse`,
/// `make_ordered_layout`, the products' padding and raked_product's
/// coalescing, and `make_layout_tv`'s thread-value layout. `ERR` marks a
/// call cute-rs refuses: CuTe DSL refuses the first, and the others are
/// ill-posed (a non-injective complement, a composition C++ CuTe gets
/// wrong).
#[test]
fn cutedsl_recorded_vectors() {
    let p = |s: &str| -> Layout { s.parse().unwrap_or_else(|e| panic!("{s}: {e}")) };
    let tilers = |s: &str| -> Tiler { Tree::Branch(s.split(';').map(|t| Tree::Leaf(Some(p(t)))).collect()) };
    let cases: &[(&str, &[&str], &str)] = &[
        ("composition", &["(8,2):(2,16)", "(6,1,1):(1,6,6)"], "(6,1,1):(2,12,12)"),
        ("composition", &["(3,6,2):(1,6,3)", "(2,1):(1,1)"], "(2,1):(1,3)"),
        ("composition", &["(2,1):(2,1)", "(2,(2,5)):(4,(12,2))"], "(2,(2,5)):(2,(6,1))"),
        ("composition", &["1:2", "(3,(3,1),5,4):(320,(960,2),32,4)"], "(3,(3,1),5,4):(640,(1920,4),64,8)"),
        ("composition", &["(8,2):(2,1)", "(2,3):(3,1)"], "(2,3):(6,2)"),
        ("composition", &["(3,6,3):(12,12,3)", "1:4"], "ERR"),
        ("composition", &["((3,8),4):((8,1),24)", "((2,2),3):((2,2),0)"], "ERR"),
        ("complement", &["(3,2):(2,16)", "21"], "(2,2):(1,6)"),
        ("complement", &["1:2", "1"], "1:0"),
        ("complement", &["(2,2,3,3):(6,4,1,2)", "24"], "ERR"),
        ("logical_divide", &["(2,4,1):(8,8,0)", "8:4"], "((2,4),(2,2)):((16,0),(8,8))"),
        ("logical_divide", &["(3,5,2):(4,16,160)", "2:1"], "ERR"),
        ("logical_divide", &["3:3", "((1,2),2):((2,8),2)"], "(((1,2),2),(2,2)):(((6,24),6),(3,12))"),
        ("zipped_divide", &["(4,2,(2,1)):(1,8,(4,32))", "2:1;2:1;2:2"], "((2,2,2),(2,1,2)):((1,8,32),(2,0,4))"),
        ("tiled_divide", &["((2,1),2,2,5):((20,2),10,1,2)", "2:2"], "((2),2,2,2,5):((2),20,10,1,2)"),
        ("left_inverse", &["4:3"], "(3,4):(0,1)"),
        ("left_inverse", &["(3,2,6):(12,48,1)"], "(12,4,2):(6,1,3)"),
        ("left_inverse", &["(2,3,2):(4,1,3)"], "(4,2):(2,1)"),
        ("left_inverse", &["8:0"], "1:0"),
        ("logical_product", &["5:1", "1:1"], "(5,1):(1,0)"),
        ("logical_product", &["(4,6):(28,4)", "(4):(2)"], "((4,6),((2,2))):((28,4),((2,112)))"),
        ("blocked_product", &["(6,1):(2,12)", "(1,4):(4,1)"], "((6,1),(1,(2,2))):((2,24),(12,(1,12)))"),
        ("blocked_product", &["1:6", "(4):(1)"], "((1,(4))):((6,(1)))"),
        ("raked_product", &["(4,6):(6,1)", "(4):(2)"], "((4,4),6):((48,6),1)"),
        (
            "raked_product",
            &["(5,4,3):(1,15,5)", "((4,2),3):((12,60),2)"],
            "((4,2,5),(3,4),3):((720,3600,1),(120,15),5)",
        ),
        ("zipped_product", &["(2):(1)", "(8,3,3):(27,9,3)"], "((2),(8,3,3)):((1),(54,18,6))"),
        ("tiled_product", &["1:4", "(2,3,3):(9,3,1)"], "(1,2,3,3):(4,9,3,1)"),
        ("flat_product", &["(4,2):(2,8)", "2:1"], "(4,2,2):(2,8,1)"),
        ("flat_product", &["(2,3):(4,1)", "3:2"], "(2,3,3):(4,1,16)"),
        ("make_layout_tv", &["(2):(1)", "1:1"], "((2),(2,1):(1,0))"),
    ];
    for &(op, a, want) in cases {
        let call = || match op {
            "composition" => p(a[0]).compose(&p(a[1])).to_string(),
            "complement" => p(a[0]).complement(a[1].parse::<usize>().unwrap()).to_string(),
            "logical_divide" => p(a[0]).logical_divide(&Tree::Leaf(Some(p(a[1])))).to_string(),
            "zipped_divide" => p(a[0]).zipped_divide(&tilers(a[1])).to_string(),
            "tiled_divide" => p(a[0]).tiled_divide(&tilers(a[1])).to_string(),
            "left_inverse" => p(a[0]).left_inverse().to_string(),
            "logical_product" => logical_product(&p(a[0]), p(a[1])).to_string(),
            "blocked_product" => blocked_product(&p(a[0]), &p(a[1])).to_string(),
            "raked_product" => raked_product(&p(a[0]), &p(a[1])).to_string(),
            "zipped_product" => zipped_product(&p(a[0]), &p(a[1])).to_string(),
            "tiled_product" => tiled_product(&p(a[0]), &p(a[1])).to_string(),
            "flat_product" => flat_product(&p(a[0]), &p(a[1])).to_string(),
            "make_layout_tv" => {
                let (tiler, tv) = make_layout_tv(&p(a[0]), &p(a[1]));
                format!("({tiler},{tv})")
            }
            _ => unreachable!(),
        };
        let got = std::panic::catch_unwind(call).map(|s| s.replace(' ', "")).unwrap_or_else(|_| "ERR".to_string());
        assert_eq!(got, want, "{op}({})", a.join(", "));
    }
    // make_ordered_layout gives an extent of 1 stride 0.
    assert_eq!(
        make_ordered_layout(vec![1usize, 8, 4, 5, 4], &[1, 3, 4, 2, 0]).to_string().replace(' ', ""),
        "(1,8,4,5,4):(0,20,160,4,1)"
    );
}
