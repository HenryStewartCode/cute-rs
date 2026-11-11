//! Layout algebra tests: worked examples from the CuTe documentation plus
//! brute-force property checks on generated layouts.

use super::*;

fn l(shape: impl Into<IntTuple>, stride: impl Into<IntTuple>) -> Layout {
    Layout::new(shape, stride)
}

#[test]
fn display_and_size() {
    let a = l(((2, 4), 3), ((1, 2), 8));
    assert_eq!(a.to_string(), "((2,4),3):((1,2),8)");
    assert_eq!(a.size(), 24);
    assert_eq!(a.cosize(), 1 + 1 + 6 + 16);
    assert_eq!(Layout::row_major((4, 8)).to_string(), "(4,8):(8,1)");
    assert_eq!(Layout::col_major((4, 8)).to_string(), "(4,8):(1,4)");
}

#[test]
fn coordinates() {
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
    assert_eq!(l((2, (1, 6)), (1, (6, 2))).coalesce().to_string(), "12:1");
    assert_eq!(l((4, 2), (1, 8)).coalesce().to_string(), "(4,2):(1,8)");
    assert_eq!(l((1, 1), (3, 5)).coalesce().to_string(), "1:0");
    let profile: IntTuple = (1, 1).into();
    assert_eq!(l(((2, 2), (3, 4)), ((1, 2), (4, 12))).coalesce_profile(&profile).to_string(), "(4,12):(1,4)");
}

#[test]
fn composition_examples() {
    assert_eq!(composition(&l((6, 2), (8, 2)), &l((4, 3), (3, 1))).to_string(), "((2,2),3):((24,2),8)");
    assert_eq!(composition(&l(20, 2), &l((5, 4), (4, 1))).to_string(), "(5,4):(8,2)");
    assert_eq!(composition(&l((10, 2), (16, 4)), &l((5, 4), (1, 5))).to_string(), "(5,(2,2)):(16,(80,4))");
}

#[test]
fn complement_examples() {
    assert_eq!(complement(&l(4, 1), 24).to_string(), "6:4");
    assert_eq!(complement(&l(6, 4), 24).to_string(), "4:1");
    assert_eq!(complement(&l((4, 6), (1, 4)), 24).size(), 1);
    assert_eq!(complement(&l((4, 6), (1, 4)), 72).to_string(), "3:24");
    assert_eq!(complement(&l((2, 2), (1, 6)), 24).to_string(), "(3,2):(2,12)");
}

#[test]
fn divide_examples() {
    let a = l((4, 2, 3), (2, 1, 8));
    assert_eq!(logical_divide(&a, l(4, 2)).to_string(), "((2,2),(2,3)):((4,1),(2,8))");

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
    assert_eq!(logical_product(&l((2, 2), (4, 1)), &l(6, 1)).to_string(), "((2,2),(2,3)):((4,1),(2,8))");
    let block = l((2, 5), (5, 1));
    let tiler = l((3, 4), (1, 3));
    assert_eq!(blocked_product(&block, &tiler).to_string(), "((2,3),(5,4)):((5,10),(1,30))");
    assert_eq!(raked_product(&block, &tiler).to_string(), "((3,2),(4,5)):((10,5),(30,1))");
}

#[test]
fn tv_layout_example() {
    // The CuTe DSL elementwise tutorial: 4x32 threads, 4x4 values each.
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
    assert_eq!(round_up(9, 4), 12);
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
