// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's SM80/SM89 GEMM entry points: `SegmentGEMMWrapper.run` (the LoRA segment GEMM)
//! and `bmm_fp8`. This is cute-rs's own code written after FlashInfer's published design
//! (`include/flashinfer/gemm/group_gemm.cuh`, `flashinfer/gemm/gemm_base.py`); no FlashInfer
//! source was translated.
//!
//! **Segment GEMM** ([`segment_gemm_kernel`], [`SegmentProblem`]): `x` is `[rows, d_in]`, cut
//! into `batch_size` segments by `seg_indptr` (i64, `batch_size + 1` entries); segment `i` is
//! multiplied by weight `weight_indices[i]` (or weight `i` without indices) of `weights`,
//! `[num_weights, d_in, d_out]` (row-major) or `[num_weights, d_out, d_in]` (column-major), and
//! the result is **added** to `y`, `[rows, d_out]`. FlashInfer's SM80 path runs CUTLASS's
//! grouped GEMM with `LinearCombination(alpha = 1, beta = 1)` and `C = D = y`, so `run`
//! accumulates into `out` (zero when `out` is not given). Float16 and BFloat16, f32
//! accumulation. The CTA tile is CUTLASS's as FlashInfer instantiates it: 128 x 128 x 32 with
//! 2 x 2 warps of 64 x 64 and `mma.sync.m16n8k16`; a `cp.async` pipeline of [`Tile::stages`]
//! stages, `Swizzle<2,3,3>` K-major shared-memory atoms (and `Swizzle<3,3,3>` N-major ones with
//! `ldmatrix.trans` for row-major weights). Where CUTLASS runs FlashInfer's four persistent
//! CTAs over a host-built problem list, this kernel needs no argument-building pass: it launches
//! one CTA per (M tile, N tile) for an upper bound of `rows / BM + batch_size` M tiles, and
//! each CTA finds its segment by scanning `seg_indptr`; the CTAs past the last tile exit.
//!
//! **`bmm_fp8`** ([`bmm_fp8_kernel`], [`BmmFp8Problem`]): `A` `[b, m, k]` row-major and `B`
//! `[b, k, n]` column-major (stored `[b, n, k]`), both E4M3 or both E5M2; `out = (A @ B) *
//! a_scale * b_scale` in BFloat16 or Float16, the scales being one-element f32 device tensors.
//! On SM89 FlashInfer 0.7.5 runs it through cuBLASLt (its `cutlass` and `cute-dsl` backends
//! are SM100+), so the comparison is against cuBLASLt. The kernel is the same mainloop with
//! `mma.sync.m16n8k32` E4M3/E5M2 (f32 accumulators), `ldmatrix` of 8-bit matrices, and a
//! [`Tile`] chosen per shape ([`BmmFp8Problem::tile`]): 16-row tiles for decode-like `m`.
//!
//! Not yet covered: `d_in`, `d_out`, `k`, `n` that are not multiples of the CTA tile (FlashInfer
//! needs multiples of 8), the `seg_lens` form (pass `seg_indptr`), FP8 segment GEMM (SM90 only in
//! FlashInfer), `bmm_fp8`'s bias, and tensors beyond 2^31 elements (offsets are 32-bit).
//!
//! `examples/gemm.rs` runs both on the simulator and, with `--features cuda` and `GPU=1`, on
//! the GPU against the f64 reference, then benchmarks them.

use cute::prelude::*;

pub use crate::norm::Dtype;

/// An FP8 input type of `bmm_fp8`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fp8 {
    E4M3,
    E5M2,
}

impl Fp8 {
    pub fn name(self) -> &'static str {
        match self {
            Fp8::E4M3 => "e4m3",
            Fp8::E5M2 => "e5m2",
        }
    }

    /// `values` rounded to this type (to nearest even, saturated).
    pub fn buffer(self, values: &[f32]) -> KernelArg {
        match self {
            Fp8::E4M3 => KernelArg::fp8_from_f32::<F8E4M3FN>(values),
            Fp8::E5M2 => KernelArg::fp8_from_f32::<F8E5M2>(values),
        }
    }

    pub fn round(self, values: &[f32]) -> Vec<f32> {
        self.buffer(values).to_f32()
    }

    /// The largest finite value.
    pub fn max(self) -> f32 {
        match self {
            Fp8::E4M3 => 448.0,
            Fp8::E5M2 => 57344.0,
        }
    }
}

/// A CTA tile: `bm x bn x bk`, `stages` pipeline stages, `wm x wn` warps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub stages: usize,
    pub wm: usize,
    pub wn: usize,
}

impl Tile {
    /// FlashInfer's CUTLASS grouped-GEMM tile (`GemmShape<128, 128, 32>`, warps of 64 x 64),
    /// with three stages (CUTLASS's choice on SM89 is two).
    pub const SEGMENT: Tile = Tile { bm: 128, bn: 128, bk: 32, stages: 3, wm: 2, wn: 2 };

    pub fn threads(&self) -> usize {
        32 * self.wm * self.wn
    }

    pub fn name(&self) -> String {
        let Tile { bm, bn, bk, stages, wm, wn } = *self;
        format!("{bm}x{bn}x{bk}_s{stages}_w{wm}x{wn}")
    }

    fn validate(&self, width: usize) {
        let vec = 128 / width;
        let threads = self.threads();
        let tpr = self.bk / vec;
        assert!(self.bk.is_multiple_of(vec) && threads.is_multiple_of(tpr), "bad tile {self:?}");
        assert!(self.bm.is_multiple_of(threads / tpr) && self.bn.is_multiple_of(threads / tpr), "bad tile {self:?}");
        assert!(self.bm.is_multiple_of(16 * self.wm) && self.bn.is_multiple_of(16 * self.wn), "bad tile {self:?}");
        assert!(self.stages >= 2, "bad tile {self:?}");
    }
}

/// A static upper bound on the rows of a global tensor whose rows hold `cols` elements:
/// the largest multiple of 128 rows that keeps every offset below 2^31.
fn row_bound(cols: usize) -> usize {
    ((1usize << 31) - 1) / cols / 128 * 128
}

/// The K-major shared-memory atom for elements of `width` bits and a K tile of `bk`: eight
/// rows of `min(bk, 128 bytes)` elements, and the swizzle that makes `ldmatrix` conflict-free
/// (`Swizzle<2,3,3>` for 16 bits and BK 32, `Swizzle<3,4,4>` for 8 bits and BK 128).
fn k_major_smem(width: usize, rows: usize, bk: usize, stages: usize) -> ComposedLayout {
    let row = bk.min(128 * 8 / width);
    let log2 = |x: usize| x.trailing_zeros();
    let bits = log2(row * width / 128).min(3);
    let base = log2(128 / width);
    let atom = Layout::row_major((8, row));
    make_composed_layout(make_swizzle(bits, base, base as i32), 0, tile_to_shape(&atom, (rows, bk, stages)))
}

/// The N-major shared-memory atom for 16-bit elements: 64 contiguous N values (128 bytes) by
/// eight K, `Swizzle<3,3,3>`.
fn n_major_smem(rows: usize, bk: usize, stages: usize) -> ComposedLayout {
    let atom = Layout::col_major((64, 8));
    make_composed_layout(make_swizzle(3, 3, 3), 0, tile_to_shape(&atom, (rows, bk, stages)))
}

/// The tiled MMA of `tile` for inputs `ab`: m16n8k16 (16-bit) or m16n8k32 (FP8), f32
/// accumulators, `wm x wn` warps, each warp's N doubled (`permutation_mnk`) so that B is
/// loaded four 8x8 matrices at a time.
fn tiled_mma(ab: DType, tile: Tile) -> TiledMma {
    let fp8 = ab.bits() == 8;
    let warps = Layout::col_major((tile.wm, tile.wn, 1));
    let mma = if fp8 {
        make_tiled_mma(MmaFP8Op::new(ab, DType::F32, (16, 8, 32)), warps)
    } else {
        make_tiled_mma(MmaF16BF16Op::new(ab, DType::F32, (16, 8, 16)), warps)
    };
    let perm = |n: usize| Some(Layout::new(n, 1));
    mma.with_permutation_mnk([perm(16 * tile.wm), perm(16 * tile.wn), perm(if fp8 { 32 } else { 16 })])
}

/// The CTA's mainloop: `g_a` `(BM, BK, k)` (K-major, rows at or past `a_rows` zero-filled)
/// times `g_b` `(BN, BK, k)` (K-major, or N-major if `b_n_major`), over `k_tiles` K tiles.
/// Returns this thread's MMA slice and its f32 accumulators.
fn mainloop<AB: Numeric>(
    tile: Tile,
    k_tiles: usize,
    g_a: &Tensor<AB>,
    a_rows: Int32,
    g_b: &Tensor<AB>,
    b_n_major: bool,
) -> (ThrMma, Tensor<f32>) {
    let Tile { bm, bn, bk, stages, .. } = tile;
    let width = AB::DTYPE.bits();
    let fp8 = width == 8;
    assert!(!(fp8 && b_n_major), "FP8 B must be K-major");
    assert!(!b_n_major || bn.is_multiple_of(64), "N-major B needs BN a multiple of 64");
    tile.validate(width);
    let threads = tile.threads();
    let tid = thread_idx().x;

    let s_a = make_smem_tensor::<AB>(k_major_smem(width, bm, bk, stages));
    let s_b = if b_n_major {
        make_smem_tensor::<AB>(n_major_smem(bn, bk, stages))
    } else {
        make_smem_tensor::<AB>(k_major_smem(width, bn, bk, stages))
    };

    // Global -> shared: 16-byte `cp.async`, `bk / vec` threads per K-major row; N-major B has
    // `bn / vec` threads per K row.
    let vec = 128 / width;
    let cp = make_copy_atom::<AB>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
    let tpr = bk / vec;
    let k_major_copy =
        || make_tiled_copy_tv(&cp, &Layout::row_major((threads / tpr, tpr)), &Layout::row_major((1, vec)));
    let g2s_a = k_major_copy();
    let g2s_b = if b_n_major {
        let tpc = (bn / vec).min(threads);
        make_tiled_copy_tv(&cp, &Layout::col_major((tpc, threads / tpc)), &Layout::col_major((vec, 1)))
    } else {
        k_major_copy()
    };
    let thr_a = g2s_a.get_slice(tid);
    let thr_b = g2s_b.get_slice(tid);
    let t_a_g = thr_a.partition_s(g_a); // (CPY, CPY_M, CPY_K, k)
    let t_a_s = thr_a.partition_d(&s_a); // (CPY, CPY_M, CPY_K, stage)
    let t_b_g = thr_b.partition_s(g_b);
    let t_b_s = thr_b.partition_d(&s_b);
    let c_a = thr_a.partition_s(&make_identity_tensor((bm, bk)));
    let p_a = make_predicate(&c_a, |c| elem_less(c.mode(0).clone(), a_rows));

    let mma = tiled_mma(AB::DTYPE, tile);
    let thr_mma = mma.get_slice(tid);
    let r_a = thr_mma.partition_fragment_a::<AB, _>(&s_a.slice((.., .., 0)));
    let r_b = thr_mma.partition_fragment_b::<AB, _>(&s_b.slice((.., .., 0)));
    let acc = thr_mma.partition_fragment_c::<f32, _>(&make_identity_tensor((bm, bn)));
    acc.fill(0.0);

    let (ld_a, ld_b) = if fp8 {
        let op = CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None };
        (op, op)
    } else {
        (CopyOp::LdMatrix { num: 4, trans: false }, CopyOp::LdMatrix { num: 4, trans: b_n_major })
    };
    let s2r_a = make_tiled_copy_a(&make_copy_atom::<AB>(ld_a), &mma);
    let s2r_b = make_tiled_copy_b(&make_copy_atom::<AB>(ld_b), &mma);
    let thr_s2r_a = s2r_a.get_slice(tid);
    let thr_s2r_b = s2r_b.get_slice(tid);
    let t_a_sv = thr_s2r_a.partition_s(&s_a);
    let t_b_sv = thr_s2r_b.partition_s(&s_b);
    let r_a_v = thr_s2r_a.retile_d(&r_a);
    let r_b_v = thr_s2r_b.retile_d(&r_b);
    let k_blocks = r_a.shape().mode(2).size();

    let issue = |k_tile: Int32, stage: Int32| {
        copy_if(&g2s_a, &p_a, &t_a_g.slice((.., .., .., k_tile)), &t_a_s.slice((.., .., .., stage)));
        copy(&g2s_b, &t_b_g.slice((.., .., .., k_tile)), &t_b_s.slice((.., .., .., stage)));
    };
    for s in 0..stages - 1 {
        if s < k_tiles {
            issue(Int32::from(s as i32), Int32::from(s as i32));
        }
        cp_async_commit_group();
    }
    let acc = for_range_carry(0, k_tiles, 1, acc, |k_tile, acc| {
        cp_async_wait_group((stages - 2) as u32);
        sync_threads();
        let next = k_tile + (stages - 1) as i32;
        if_(next.lt(k_tiles), || issue(next, next % stages as i32));
        cp_async_commit_group();
        let stage = k_tile % stages as i32;
        for kb in 0..k_blocks {
            copy(&s2r_a, &t_a_sv.slice((.., .., kb, stage)), &r_a_v.slice((.., .., kb)));
            copy(&s2r_b, &t_b_sv.slice((.., .., kb, stage)), &r_b_v.slice((.., .., kb)));
            gemm(&mma, &acc, &r_a.slice((.., .., kb)), &r_b.slice((.., .., kb)), &acc);
        }
        acc
    });
    cp_async_wait_group(0);
    (thr_mma, acc)
}

/// Store `f(acc[i], i)` to `g_c` `(BM, BN)` in `O`, rows at or past `rows` skipped; with
/// `accumulate`, `f` gets the old value of `C` too (as f32), else 0.
fn epilogue<O: Numeric>(
    thr_mma: &ThrMma,
    acc: &Tensor<f32>,
    g_c: &Tensor<O>,
    tile: Tile,
    rows: Int32,
    accumulate: bool,
    f: impl Fn(Float32, Float32) -> Float32,
) {
    let t_c_g = thr_mma.partition_c(g_c);
    let c_c = thr_mma.partition_c(&make_identity_tensor((tile.bm, tile.bn)));
    let pred = make_predicate(&c_c, |c| elem_less(c.mode(0).clone(), rows));
    let atom = make_copy_atom::<O>(CopyOp::Universal { bits: 32 });
    let old = make_fragment_like::<O, _>(acc);
    if accumulate {
        old.fill(Float32::from(0.0f32).to::<O>());
        copy_if(&atom, &pred, &t_c_g, &old);
    }
    let out = make_fragment_like::<O, _>(acc);
    for i in 0..acc.size() {
        let c = if accumulate { old.load(i).to::<f32>() } else { Float32::from(0.0f32) };
        out.store(i, f(acc.load(i), c).to::<O>());
    }
    copy_if(&atom, &pred, &out, &t_c_g);
}

// ---------------------------------------------------------------------------------------------
// Segment GEMM
// ---------------------------------------------------------------------------------------------

/// The segment GEMM for `d_in x d_out` weights in `dtype`. Parameters: `x`, `w`, `y`,
/// `seg_indptr` (i64), `weight_indices` (i64; read only if `use_weight_indices != 0`),
/// `batch_size`. Launch it with [`segment_launch`].
pub fn segment_gemm_kernel(dtype: Dtype, d_in: usize, d_out: usize, weight_column_major: bool, tile: Tile) -> Kernel {
    match dtype {
        Dtype::F16 => segment_of::<F16>(dtype, d_in, d_out, weight_column_major, tile),
        Dtype::BF16 => segment_of::<BF16>(dtype, d_in, d_out, weight_column_major, tile),
    }
}

/// The tile for `rows` rows in `batch_size` segments (measured on an RTX 2000 Ada):
/// FlashInfer's [`Tile::SEGMENT`], or for segments of at most 64 rows on average, where the
/// GEMM is bound by reading the weights, 32 x 64 x 64 tiles of 1 x 2 warps (more CTAs, less
/// padding).
pub fn segment_tile(rows: usize, batch_size: usize) -> Tile {
    if rows <= 64 * batch_size.max(1) {
        Tile { bm: 32, bn: 64, bk: 64, stages: 4, wm: 1, wn: 2 }
    } else {
        Tile::SEGMENT
    }
}

/// The M tiles of a group in the segment GEMM's tile order.
pub const GROUP_M: usize = 4;

/// The launch of [`segment_gemm_kernel`] for `rows` rows in `batch_size` segments: an upper
/// bound on the M tiles (a multiple of [`GROUP_M`]) times the N tiles, in one dimension.
pub fn segment_launch(rows: usize, batch_size: usize, d_out: usize, tile: Tile) -> LaunchConfig {
    let m_tiles = (rows.div_ceil(tile.bm) + batch_size).next_multiple_of(GROUP_M);
    LaunchConfig::new([(m_tiles * (d_out / tile.bn)) as u32, 1, 1], [tile.threads() as u32, 1, 1])
}

fn segment_of<T: Numeric>(dtype: Dtype, k: usize, n: usize, col_major: bool, tile: Tile) -> Kernel {
    let Tile { bm, bn, bk, .. } = tile;
    assert!(k.is_multiple_of(bk) && n.is_multiple_of(bn), "d_in {k} / d_out {n} must be multiples of {bk} / {bn}");
    let layout = if col_major { "col" } else { "row" };
    let name = format!("flashinfer_segment_gemm_{}_k{k}_n{n}_{layout}_{}", dtype.name(), tile.name());
    Kernel::builder(&name).block_dim([tile.threads() as u32, 1, 1]).build(|| {
        let x = param_ptr::<T>("x");
        let w = param_ptr::<T>("w");
        let y = param_ptr::<T>("y");
        let seg_indptr = param_ptr::<i64>("seg_indptr");
        let weight_indices = param_ptr::<i64>("weight_indices");
        let batch = param_scalar::<i32>("batch_size");
        let use_indices = param_scalar::<i32>("use_weight_indices");

        // This CTA's segment and the first row of its M tile: the segments' tiles are
        // numbered one after another.
        // The CTAs walk groups of `GROUP_M` M tiles across every N tile, M fastest, so that
        // the CTAs resident together share their rows of `x` and their weight tiles in L2.
        let id = block_idx().x;
        let per_group = (GROUP_M * (n / bn)) as i32;
        let within = id % per_group;
        let tile_id = id / per_group * GROUP_M as i32 + within % GROUP_M as i32;
        let n_tile = within / GROUP_M as i32;
        let seg = Var::new(Int32::from(-1));
        let row0 = Var::new(Int32::from(0));
        let rows = Var::new(Int32::from(0));
        let first = Var::new(Int32::from(0));
        for_range(0, batch, 1, |s| {
            let lo = load(&seg_indptr.offset(s), LoadOptions::default()).to::<i32>();
            let hi = load(&seg_indptr.offset(s + 1), LoadOptions::default()).to::<i32>();
            let tiles = (hi - lo + (bm as i32 - 1)) / bm as i32;
            let f = first.get();
            if_(tile_id.ge(f) & tile_id.lt(f + tiles), || {
                let r0 = lo + (tile_id - f) * bm as i32;
                seg.set(s);
                row0.set(r0);
                rows.set(hi - r0);
            });
            first.set(f + tiles);
        });
        let seg = seg.get();
        if_(seg.ge(0), || {
            let (row0, rows) = (row0.get(), rows.get());
            let widx = Var::new(seg);
            if_(use_indices.ne(0), || widx.set(load(&weight_indices.offset(seg), LoadOptions::default()).to::<i32>()));
            let widx = widx.get();
            let x_all = make_tensor(&x, Layout::row_major((row_bound(k), k)));
            let y_all = make_tensor(&y, Layout::row_major((row_bound(n), n)));
            let g_a =
                local_tile_proj(&domain_offset((row0, 0), &x_all), (bm, bn, bk), (0, n_tile, ..), &[true, false, true]);
            let g_c =
                local_tile_proj(&domain_offset((row0, 0), &y_all), (bm, bn, bk), (0, n_tile, ..), &[true, true, false]);
            // B as `(N, K)`: weight `widx` is `(d_out, d_in)` row-major (column-major weights)
            // or `(d_in, d_out)` row-major, which is `(N, K)` column-major.
            let w_all = if col_major {
                domain_offset((widx * n as i32, 0), &make_tensor(&w, Layout::row_major((row_bound(k), k))))
            } else {
                domain_offset((0, widx * k as i32), &make_tensor(&w, Layout::col_major((n, row_bound(n)))))
            };
            let g_b = local_tile_proj(&w_all, (bm, bn, bk), (0, n_tile, ..), &[false, true, true]);
            let (thr_mma, acc) = mainloop(tile, k / bk, &g_a, rows, &g_b, !col_major);
            epilogue(&thr_mma, &acc, &g_c, tile, rows, true, |a, c| a + c);
        });
    })
}

/// A segment GEMM problem with its host data.
pub struct SegmentProblem {
    pub dtype: Dtype,
    pub d_in: usize,
    pub d_out: usize,
    pub weight_column_major: bool,
    pub seg_lens: Vec<usize>,
    pub num_weights: usize,
    pub weight_indices: Option<Vec<usize>>,
    /// `[rows, d_in]`, rounded to `dtype`.
    pub x: Vec<f32>,
    /// `[num_weights, d_in, d_out]` or (column-major) `[num_weights, d_out, d_in]`.
    pub w: Vec<f32>,
    /// `y`'s value before the call (it is accumulated into).
    pub y0: Vec<f32>,
    pub tile: Tile,
}

fn uniform(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

impl SegmentProblem {
    /// Random `x`, `w` and `y0` (`y0` zero unless `accumulate`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dtype: Dtype,
        seg_lens: &[usize],
        d_in: usize,
        d_out: usize,
        weight_column_major: bool,
        num_weights: usize,
        weight_indices: Option<Vec<usize>>,
        accumulate: bool,
        seed: u64,
    ) -> SegmentProblem {
        let rows: usize = seg_lens.iter().sum();
        assert!(
            weight_indices.as_ref().is_none_or(|w| w.len() == seg_lens.len() && w.iter().all(|&i| i < num_weights))
        );
        assert!(weight_indices.is_some() || num_weights >= seg_lens.len());
        let round = |v: Vec<f32>| v.into_iter().map(|x| dtype.round(x)).collect::<Vec<_>>();
        let x = round(uniform(rows * d_in, seed, 1.0));
        let w = round(uniform(num_weights * d_in * d_out, seed + 1, 1.0));
        let y0 = if accumulate { round(uniform(rows * d_out, seed + 2, 4.0)) } else { vec![0.0; rows * d_out] };
        SegmentProblem {
            dtype,
            d_in,
            d_out,
            weight_column_major,
            seg_lens: seg_lens.to_vec(),
            num_weights,
            weight_indices,
            x,
            w,
            y0,
            tile: segment_tile(rows, seg_lens.len()),
        }
    }

    pub fn rows(&self) -> usize {
        self.seg_lens.iter().sum()
    }

    pub fn seg_indptr(&self) -> Vec<i64> {
        let mut v = vec![0i64];
        for &l in &self.seg_lens {
            v.push(v.last().unwrap() + l as i64);
        }
        v
    }

    pub fn kernel(&self) -> Kernel {
        segment_gemm_kernel(self.dtype, self.d_in, self.d_out, self.weight_column_major, self.tile)
    }

    pub fn launch(&self) -> LaunchConfig {
        segment_launch(self.rows(), self.seg_lens.len(), self.d_out, self.tile)
    }

    /// `x, w, y, seg_indptr, weight_indices, batch_size, use_weight_indices`.
    pub fn args(&self) -> Vec<KernelArg> {
        let indices: Vec<i64> = match &self.weight_indices {
            Some(w) => w.iter().map(|&i| i as i64).collect(),
            None => vec![0],
        };
        vec![
            self.dtype.buffer(&self.x),
            self.dtype.buffer(&self.w),
            self.dtype.buffer(&self.y0),
            KernelArg::buffer(&self.seg_indptr()),
            KernelArg::buffer(&indices),
            KernelArg::ScalarI32(self.seg_lens.len() as i32),
            KernelArg::ScalarI32(self.weight_indices.is_some() as i32),
        ]
    }

    pub fn flops(&self) -> f64 {
        2.0 * (self.rows() * self.d_in * self.d_out) as f64
    }

    pub fn reference(&self) -> Vec<f64> {
        let (k, n) = (self.d_in, self.d_out);
        let indptr = self.seg_indptr();
        let mut y: Vec<f64> = self.y0.iter().map(|&v| v as f64).collect();
        for s in 0..self.seg_lens.len() {
            let wi = self.weight_indices.as_ref().map_or(s, |w| w[s]);
            let w = &self.w[wi * k * n..(wi + 1) * k * n];
            for r in indptr[s] as usize..indptr[s + 1] as usize {
                let xr = &self.x[r * k..(r + 1) * k];
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for (l, &xv) in xr.iter().enumerate() {
                        let wv = if self.weight_column_major { w[j * k + l] } else { w[l * n + j] };
                        sum += xv as f64 * wv as f64;
                    }
                    y[r * n + j] += sum;
                }
            }
        }
        y
    }

    /// The largest error of `got` against the reference, in units of the tolerance (one
    /// rounding of `dtype` plus f32 accumulation): at most 1 passes.
    pub fn check(&self, got: &[f32]) -> f64 {
        scaled_error(got, &self.reference(), self.dtype.tolerance(), self.d_in)
    }
}

fn scaled_error(got: &[f32], want: &[f64], rel: f64, k: usize) -> f64 {
    assert_eq!(got.len(), want.len());
    let abs = rel * (k as f64).sqrt() * 0.05;
    got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).abs() / (abs + rel * w.abs())).fold(0.0, f64::max)
}

/// Simulate `p` (with cute-rs's race and alignment checks) and return [`SegmentProblem::check`].
pub fn simulate_segment(p: &SegmentProblem) -> f64 {
    let k = p.kernel();
    let mut args = p.args();
    k.simulate(p.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.check(&p.dtype.read(&args[2]))
}

// ---------------------------------------------------------------------------------------------
// bmm_fp8
// ---------------------------------------------------------------------------------------------

/// `bmm_fp8` for `k x n` B in `fp8`, output `out`. Parameters: `a`, `b`, `d`, `a_scale`,
/// `b_scale` (f32), `m`. Launch it with [`bmm_fp8_launch`].
pub fn bmm_fp8_kernel(fp8: Fp8, out: Dtype, k: usize, n: usize, tile: Tile) -> Kernel {
    match (fp8, out) {
        (Fp8::E4M3, Dtype::BF16) => bmm_of::<F8E4M3FN, BF16>(fp8, out, k, n, tile),
        (Fp8::E4M3, Dtype::F16) => bmm_of::<F8E4M3FN, F16>(fp8, out, k, n, tile),
        (Fp8::E5M2, Dtype::BF16) => bmm_of::<F8E5M2, BF16>(fp8, out, k, n, tile),
        (Fp8::E5M2, Dtype::F16) => bmm_of::<F8E5M2, F16>(fp8, out, k, n, tile),
    }
}

pub fn bmm_fp8_launch(batch: usize, m: usize, n: usize, tile: Tile) -> LaunchConfig {
    LaunchConfig::new([m.div_ceil(tile.bm) as u32, (n / tile.bn) as u32, batch as u32], [tile.threads() as u32, 1, 1])
}

fn bmm_of<AB: Numeric, O: Numeric>(fp8: Fp8, out: Dtype, k: usize, n: usize, tile: Tile) -> Kernel {
    let Tile { bm, bn, bk, .. } = tile;
    assert!(k.is_multiple_of(bk) && n.is_multiple_of(bn), "k {k} / n {n} must be multiples of {bk} / {bn}");
    let name = format!("flashinfer_bmm_fp8_{}_{}_k{k}_n{n}_{}", fp8.name(), out.name(), tile.name());
    Kernel::builder(&name).block_dim([tile.threads() as u32, 1, 1]).build(|| {
        let a = param_ptr::<AB>("a");
        let b = param_ptr::<AB>("b");
        let d = param_ptr::<O>("d");
        let a_scale = param_ptr::<f32>("a_scale");
        let b_scale = param_ptr::<f32>("b_scale");
        let m = param_scalar::<i32>("m");
        let bid = block_idx();
        let row0 = bid.z * m + bid.x * bm as i32;
        let rows = m - bid.x * bm as i32;
        let a_all = make_tensor(&a, Layout::row_major((row_bound(k), k)));
        let b_all = make_tensor(&b, Layout::row_major((row_bound(k), k)));
        let d_all = make_tensor(&d, Layout::row_major((row_bound(n), n)));
        let g_a =
            local_tile_proj(&domain_offset((row0, 0), &a_all), (bm, bn, bk), (0, bid.y, ..), &[true, false, true]);
        let g_b = local_tile_proj(
            &domain_offset((bid.z * n as i32, 0), &b_all),
            (bm, bn, bk),
            (0, bid.y, ..),
            &[false, true, true],
        );
        let g_c =
            local_tile_proj(&domain_offset((row0, 0), &d_all), (bm, bn, bk), (0, bid.y, ..), &[true, true, false]);
        let (thr_mma, acc) = mainloop(tile, k / bk, &g_a, rows, &g_b, false);
        let scale = load(&a_scale, LoadOptions::default()) * load(&b_scale, LoadOptions::default());
        epilogue(&thr_mma, &acc, &g_c, tile, rows, false, |x, _| x * scale);
    })
}

/// A `bmm_fp8` problem with its host data.
pub struct BmmFp8Problem {
    pub fp8: Fp8,
    pub out: Dtype,
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// `[batch, m, k]`, FP8 values.
    pub a: Vec<f32>,
    /// `[batch, n, k]` (B column-major), FP8 values.
    pub b: Vec<f32>,
    pub a_scale: f32,
    pub b_scale: f32,
    pub tile: Tile,
}

impl BmmFp8Problem {
    /// Random inputs quantized as FlashInfer's example does: scaled so that the largest
    /// magnitude is the type's maximum, with the inverse scales as `a_scale`, `b_scale`.
    pub fn new(fp8: Fp8, out: Dtype, batch: usize, m: usize, n: usize, k: usize, seed: u64) -> BmmFp8Problem {
        let quantize = |v: Vec<f32>| {
            let amax = v.iter().fold(1e-12f32, |a, &x| a.max(x.abs()));
            let s = fp8.max() / amax;
            (fp8.round(&v.iter().map(|&x| x * s).collect::<Vec<_>>()), 1.0 / s)
        };
        let (a, a_scale) = quantize(uniform(batch * m * k, seed, 1.0));
        let (b, b_scale) = quantize(uniform(batch * n * k, seed + 1, 1.0));
        BmmFp8Problem { fp8, out, batch, m, n, k, a, b, a_scale, b_scale, tile: Self::tile(m, n) }
    }

    /// The tile for `m x n` (measured on an RTX 2000 Ada): 16 x 128 x 128 with 1 x 4 warps for
    /// `m <= 16`, 64 x 128 x 64 with 2 x 2 warps and four stages for `m <= 64`, else
    /// 128 x 128 x 64 (the CuTe DSL FP8 example's); 64-wide N tiles when `n` is not a multiple
    /// of 128.
    pub fn tile(m: usize, n: usize) -> Tile {
        let bn = if n.is_multiple_of(128) { 128 } else { 64 };
        if m <= 16 {
            Tile { bm: 16, bn, bk: 128, stages: 4, wm: 1, wn: 4 }
        } else if m <= 64 {
            Tile { bm: 64, bn, bk: 64, stages: 4, wm: 2, wn: 2 }
        } else {
            Tile { bm: 128, bn, bk: 64, stages: 3, wm: 2, wn: 2 }
        }
    }

    pub fn kernel(&self) -> Kernel {
        bmm_fp8_kernel(self.fp8, self.out, self.k, self.n, self.tile)
    }

    pub fn launch(&self) -> LaunchConfig {
        bmm_fp8_launch(self.batch, self.m, self.n, self.tile)
    }

    /// `a, b, d, a_scale, b_scale, m`.
    pub fn args(&self) -> Vec<KernelArg> {
        vec![
            self.fp8.buffer(&self.a),
            self.fp8.buffer(&self.b),
            self.out.buffer(&vec![0.0; self.batch * self.m * self.n]),
            KernelArg::f32(vec![self.a_scale]),
            KernelArg::f32(vec![self.b_scale]),
            KernelArg::ScalarI32(self.m as i32),
        ]
    }

    pub fn flops(&self) -> f64 {
        2.0 * (self.batch * self.m * self.n * self.k) as f64
    }

    pub fn reference(&self) -> Vec<f64> {
        let (m, n, k) = (self.m, self.n, self.k);
        let scale = self.a_scale as f64 * self.b_scale as f64;
        let mut d = Vec::with_capacity(self.batch * m * n);
        for bi in 0..self.batch {
            for i in 0..m {
                let a = &self.a[(bi * m + i) * k..(bi * m + i + 1) * k];
                for j in 0..n {
                    let b = &self.b[(bi * n + j) * k..(bi * n + j + 1) * k];
                    d.push(a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum::<f64>() * scale);
                }
            }
        }
        d
    }

    /// The largest error against the reference in units of the tolerance: at most 1 passes.
    /// The tolerance is 2.5% of `|reference| + 0.05 * typical`, whatever the output type: SM89's
    /// FP8 `mma.sync` does not accumulate in full f32 (results come out about 1% off the exact
    /// sum on an RTX 2000 Ada, with Float16 or BFloat16 output alike), as cuBLASLt's do.
    pub fn check(&self, got: &[f32]) -> f64 {
        // The products of two FP8 values, typically `max^2 / 9` before scaling: the
        // tolerance is relative to the typical result, `sqrt(k) / 3` units.
        let want = self.reference();
        let typical = (self.k as f64).sqrt() / 3.0;
        let rel = 2.5e-2;
        got.iter()
            .zip(&want)
            .map(|(&g, &w)| (g as f64 - w).abs() / (rel * (w.abs() + 0.05 * typical)))
            .fold(0.0, f64::max)
    }
}

/// Simulate `p` and return [`BmmFp8Problem::check`].
pub fn simulate_bmm_fp8(p: &BmmFp8Problem) -> f64 {
    let k = p.kernel();
    let mut args = p.args();
    k.simulate(p.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.check(&p.out.read(&args[2]))
}

#[cfg(feature = "cuda")]
pub mod gpu {
    use std::error::Error;

    use cute::cuda::{CudaContext, DeviceArg};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{BmmFp8Problem, Dtype, SegmentProblem};

    /// Run `kernel` on `host` once; read back argument `out` as `dtype`; if `time`, return the
    /// average GPU time in microseconds of launches replayed from a CUDA graph.
    fn run_args(
        ctx: &CudaContext,
        tc: &Toolchain,
        kernel: &Kernel,
        launch: LaunchConfig,
        host: &[KernelArg],
        out: (usize, Dtype),
        time: bool,
    ) -> Result<(Vec<f32>, Option<f64>), Box<dyn Error>> {
        let arch = ctx.best_target(tc).ok_or("no target for this GPU")?;
        // FP8 `mma.sync` lowers through NVVM only from LLVM 23; older toolchains get inline PTX.
        let old = kernel.min_llvm_version().is_some_and(|need| tc.llvm_version().is_some_and(|have| have < need));
        let f = ctx.load(&kernel.compile_with(tc, &CompileOptions::new().gpu_arch(arch).mma_inline_ptx(old))?)?;
        let mut bufs = Vec::new();
        for a in host {
            if !matches!(a, KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_)) {
                bufs.push(ctx.upload(a)?);
            }
        }
        let mut bi = bufs.iter();
        let mut out_buf = None;
        let args: Vec<DeviceArg> = host
            .iter()
            .enumerate()
            .map(|(i, a)| match a {
                KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => DeviceArg::Scalar(a.clone()),
                _ => {
                    let b = bi.next().unwrap();
                    if i == out.0 {
                        out_buf = Some(b);
                    }
                    DeviceArg::Buffer(b)
                }
            })
            .collect();
        let stream = ctx.stream()?;
        let mut go = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(kernel, &f, launch).stream(s).args(&args).submit()?;
            Ok(())
        };
        go(&stream, 0)?;
        stream.synchronize()?;
        let b = out_buf.unwrap();
        let got: Vec<f32> = match out.1 {
            Dtype::F16 => b.to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
            Dtype::BF16 => b.to_vec::<u16>()?.into_iter().map(|x| f32::from_bits((x as u32) << 16)).collect(),
        };
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(50).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut go)?)
        } else {
            None
        };
        Ok((got, t))
    }

    /// Run `p` once; if `check`, return the scaled error against the f64 reference; if
    /// `time`, the CUDA-graph GPU time in microseconds. (Timed launches keep accumulating
    /// into `y`, which changes nothing about the time.)
    pub fn run_segment(
        ctx: &CudaContext,
        tc: &Toolchain,
        p: &SegmentProblem,
        check: bool,
        time: bool,
    ) -> Result<(f64, Option<f64>), Box<dyn Error>> {
        let (got, t) = run_args(ctx, tc, &p.kernel(), p.launch(), &p.args(), (2, p.dtype), time)?;
        Ok((if check { p.check(&got) } else { 0.0 }, t))
    }

    /// Run `p` once; as [`run_segment`].
    pub fn run_bmm_fp8(
        ctx: &CudaContext,
        tc: &Toolchain,
        p: &BmmFp8Problem,
        check: bool,
        time: bool,
    ) -> Result<(f64, Option<f64>), Box<dyn Error>> {
        let (got, t) = run_args(ctx, tc, &p.kernel(), p.launch(), &p.args(), (2, p.out), time)?;
        Ok((if check { p.check(&got) } else { 0.0 }, t))
    }
}
