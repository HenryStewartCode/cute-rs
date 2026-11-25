//! Shape and index arithmetic over runtime leaves, coordinate tensors and
//! signed index terms: `equal`, `ceil_div`, `round_up`, `shape_div`, the
//! coordinates of runtime offsets, by-mode products, `basic_copy` and
//! `basic_copy_if` of runtime size, `domain_offset` by a negative runtime
//! amount and `elem_less` against launch-time extents.

use cute::prelude::*;

use super::{Case, T, launch};

/// The columns of [`shape_arith`].
const SHAPE_ARITH_COLS: usize = 19;
/// The columns of the launch-sized tensors [`shape_arith`] copies, one per
/// thread of the first 16.
const COPY_COLS: usize = 16;

/// Shape arithmetic over runtime leaves: `equal` of layouts compared at
/// run time, `ceil_div`, `round_up` and `shape_div` of runtime extents, the
/// hierarchical and flat coordinates of runtime offsets in layouts of
/// runtime extents, a by-mode `logical_product` with a runtime tile
/// extent, `basic_copy` and `basic_copy_if` over columns of launch-sized
/// tensors (runtime loops), and offsets with a negative runtime term: a
/// memory tensor and an identity tensor shifted by `domain_offset`, and
/// `elem_less` against -1.
pub fn shape_arith() -> Case {
    let kernel = Kernel::builder("shape_arith").block_dim([T as u32, 1, 1]).build(|| {
        let n = param_extent("n", 1);
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, SHAPE_ARITH_COLS)));
        let src = param_tensor::<i32>("src", Layout::col_major((n, COPY_COLS)));
        let dst = param_tensor::<i32>("dst", Layout::col_major((n, COPY_COLS)));
        let flags = param_tensor::<bool>("flags", Layout::col_major((n, COPY_COLS)));
        let picked = param_tensor::<u32>("picked", Layout::col_major((n, COPY_COLS)));
        let t = thread_idx().x;
        let x = (xs.load(t) % 1024u32 + t.to::<u32>()).to::<i32>();
        let (m, m2, xm) = (x % 7 + 1, x / 7 % 7 + 1, x % 100);
        let flag = |p: Boolean| IndexExpr::from(select::<i32>(p, 1, 0));
        let mut values: Vec<IndexExpr> = Vec::new();

        // equal at run time: (m,8):(1,m) against (m2,8):(1,m2), and m * 2
        // against m + m, which are different expressions.
        let a = Layout::new((m, 8), (1, m));
        let b = Layout::new((m2, 8), (1, m2));
        values.push(flag(equal(&a, &b)));
        values.push(flag(equal(&Layout::new(m * 2, 1), &Layout::new(m + m, 1))));

        // ceil_div, round_up and shape_div of runtime extents.
        values.extend(ceil_div((xm, n), (4, 8)).flat_leaves());
        values.extend(round_up(xm, 8).flat_leaves());
        values.extend(shape_div((m, 6), 4).flat_leaves());

        // The coordinates of a runtime offset in compact layouts of a
        // runtime extent.
        let h = Layout::new((4, (m, 3)), (1, (4, m * 4)));
        let i = x % (m * 12);
        values.extend(h.get_hier_coord(i).flat_leaves());
        values.extend(h.get_flat_coord(i).flat_leaves());
        values.extend(Layout::new((m, 8), (8, 1)).get_hier_coord(x % (m * 8)).flat_leaves());

        // A by-mode product with a runtime tile extent: mode 0 of
        // (2,3):(1,2) repeated k times.
        let k = x % 5 + 1;
        let p = logical_product(&Layout::new((2, 3), (1, 2)), (k, ..));
        values.push(p.offset(&((x % 2, x / 2 % k), x % 3).into()));

        // A memory tensor shifted by the K residue n - 8 * ceil(n / 8), which
        // is negative.
        let k_tiles = (n + 7) / 8;
        let r = IndexExpr::signed(n - k_tiles * 8);
        let shifted = domain_offset(r.clone(), &xs);
        values.push(IndexExpr::from(shifted.load(t % 32 + 8).to::<i32>()));

        // An identity tensor of a runtime extent, shifted the same way, and
        // CuTe's residue predicate elem_less((m, -1), (M, k)).
        let c = domain_offset((0, r), &make_identity_tensor((n, 8))).coord((t % 8, t / 8 % 8));
        let (c0, c1) = (c.mode(0).clone(), c.mode(1).clone());
        values.extend(c.flat_leaves());
        values.push(flag(elem_less((c0, -1), (n, c1))));

        assert_eq!(values.len(), SHAPE_ARITH_COLS);
        for (j, v) in values.into_iter().enumerate() {
            out.store((t, j), Int32::from(v));
        }

        // Copies of runtime size: each of the first 16 threads copies its
        // column with a runtime loop.
        if_(t.lt(COPY_COLS as i32), || {
            basic_copy(&src.slice((.., t)), &dst.slice((.., t)));
            let picks = picked.slice((.., t));
            for_range(0, n, 1, |i| picks.store(i, 7u32.into_value()));
            basic_copy_if(&flags.slice((.., t)), &src.slice((.., t)), &picks);
        });
    });
    fn check(args: &[KernelArg]) {
        let KernelArg::ScalarI32(n) = args[0] else { panic!("shape_arith takes n first") };
        let n = n as usize;
        let (xs, out) = (args[1].to_vec::<u32>(), args[2].to_vec::<i32>());
        let (src, dst) = (args[3].to_vec::<i32>(), args[4].to_vec::<i32>());
        let (flags, picked) = (args[5].to_vec::<bool>(), args[6].to_vec::<u32>());
        for i in 0..n * COPY_COLS {
            assert_eq!(dst[i], src[i], "dst[{i}]");
            assert_eq!(picked[i], if flags[i] { src[i] as u32 } else { 7 }, "picked[{i}]");
        }
        let r = n as i64 - 8 * n.div_ceil(8) as i64;
        for t in 0..T {
            let x = (xs[t] % 1024) as usize + t;
            let (m, m2, xm) = (x % 7 + 1, x / 7 % 7 + 1, x % 100);
            let (i, j, k) = (x % (12 * m), x % (8 * m), x % 5 + 1);
            let (c0, c1) = ((t % 8) as i64, (t / 8 % 8) as i64 + r);
            let want: [i64; SHAPE_ARITH_COLS] = [
                (m == m2) as i64,
                1,
                xm.div_ceil(4) as i64,
                n.div_ceil(8) as i64,
                (xm.div_ceil(8) * 8) as i64,
                m.div_ceil(4) as i64,
                6usize.div_ceil(4usize.div_ceil(m)) as i64,
                (i % 4) as i64,
                (i / 4 % m) as i64,
                (i / (4 * m) % 3) as i64,
                (i % 4) as i64,
                (i / 4) as i64,
                (j / 8 % m) as i64,
                (j % 8) as i64,
                (x % 2 + 2 * (x / 2 % k) + 2 * (x % 3)) as i64,
                xs[((t % 32 + 8) as i64 + r) as usize] as i32 as i64,
                c0,
                c1,
                ((c0 as usize) < n && c1 > -1) as i64,
            ];
            let got: Vec<i64> =
                out[t * SHAPE_ARITH_COLS..(t + 1) * SHAPE_ARITH_COLS].iter().map(|&v| v as i64).collect();
            assert_eq!(got, want, "thread {t}: x={x}");
        }
    }
    Case { name: "shape_arith", kernel, cfg: launch(1), small_only: false, check }
}

/// The tile of [`coord_tensor_copy`]: rows, and columns of K.
const CC_BM: usize = 16;
const CC_BK: usize = 8;
/// Threads of [`coord_tensor_copy`], each copying 4 elements of a tile.
const CC_THREADS: usize = 32;

/// The predicated `cp.async` copy of the Ampere SGEMM's K residue: the
/// (m, k) tensor `a` of launch-time extents is tiled with `local_tile`,
/// and it and its identity tensor are shifted by `(0, k - 8 * k_tiles, 0)`,
/// a negative runtime amount, so the first K tile starts before the
/// tensor. Each tile is copied to shared memory with the predicate
/// `elem_less((row, -1), (m, col))`, which zero-fills the rows past m and
/// the columns before 0, then stored to `y`.
pub fn coord_tensor_copy() -> Case {
    let kernel = Kernel::builder("coord_tensor_copy").block_dim([CC_THREADS as u32, 1, 1]).build(|| {
        let m = param_extent("m", 1);
        let k = param_extent("k", 1);
        let a = param_tensor::<f32>("a", Layout::col_major((m, k)));
        let k_tiles = (k + (CC_BK as i32 - 1)) / CC_BK as i32;
        let y = param_tensor::<f32>("y", Layout::col_major((CC_BM, CC_BK, k_tiles)));
        let residue = IndexExpr::signed(k - k_tiles * CC_BK as i32);
        let bid = block_idx().x;
        let g_a = domain_offset((0, residue.clone(), 0), &local_tile(&a, (CC_BM, CC_BK), (bid, ..)));
        let id = make_identity_tensor(a.shape());
        let c_a = domain_offset((0, residue, 0), &local_tile(&id, (CC_BM, CC_BK), (bid, ..)));
        let smem = make_smem_tensor::<f32>(Layout::col_major((CC_BM, CC_BK)));
        let atom = make_copy_atom::<f32>(CopyOp::CpAsync { bytes: 4, bypass_l1: false });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major((16, 2)), &Layout::col_major((1, 4)));
        let t = thread_idx().x;
        let thr = tiled.get_slice(t);
        let t_s = thr.partition_d(&smem);
        for_range(0, k_tiles, 1, |j| {
            let t_g = thr.partition_s(&g_a.slice((.., .., j)));
            let t_c = thr.partition_s(&c_a.slice((.., .., j)));
            let pred = make_predicate(&t_c, |c| elem_less((c.mode(0).clone(), -1), (m, c.mode(1).clone())));
            copy_if(&tiled, &pred, &t_g, &t_s);
            cp_async_commit_group();
            cp_async_wait_group(0);
            sync_threads();
            for v in 0..4 {
                let e = t * 4 + v;
                let (r, c) = (e % CC_BM as i32, e / CC_BM as i32);
                y.store((r, c, j), smem.load((r, c)));
            }
            sync_threads();
        });
    });
    fn check(args: &[KernelArg]) {
        let (KernelArg::ScalarI32(m), KernelArg::ScalarI32(k)) = (&args[0], &args[1]) else {
            panic!("coord_tensor_copy takes the extents m and k first")
        };
        let (m, k) = (*m as usize, *k as usize);
        let (a, y) = (args[2].to_vec::<f32>(), args[3].to_vec::<f32>());
        let r = k as i64 - (CC_BK * k.div_ceil(CC_BK)) as i64;
        for j in 0..k.div_ceil(CC_BK) {
            for c in 0..CC_BK {
                for row in 0..CC_BM {
                    let gk = (c + CC_BK * j) as i64 + r;
                    let want = if row < m && gk >= 0 { a[row + m * gk as usize] } else { 0.0 };
                    let got = y[row + CC_BM * (c + CC_BK * j)];
                    assert_eq!(got.to_bits(), want.to_bits(), "y[{row}, {c}, {j}] (k column {gk})");
                }
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [CC_THREADS as u32, 1, 1]);
    Case { name: "coord_tensor_copy", kernel, cfg, small_only: false, check }
}
