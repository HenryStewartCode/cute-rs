// SPDX-License-Identifier: MIT OR Apache-2.0

//! Im2col TMA loads and stores (roadmap part m7-tma-im2col), held to a plain Rust im2col
//! of NHWC and NDHWC tensors: each kernel loads the GEMM operand `A` of an implicit-GEMM
//! forward convolution tile by tile (`(pixels, channels)` boxes of the output pixels
//! `(q, p, (z,) n)` and the filter taps `(c, s, r, (t,))`, padding and the pixels past
//! the last image zero-filled), and writes it back through the shared-memory layout; a
//! store kernel writes a `(pixels, channels)` tile through an `.im2col_no_offs`
//! descriptor into an NHWC output. The shapes are CuTe DSL's convolution examples'
//! (`dense_implicit_gemm_fprop.py`) scaled down. They need sm_90 and run in the
//! simulator.

use cute::prelude::*;
use cute::warpgroup::{SmemLayoutAtomKind, make_smem_layout_atom};

use super::Case;

/// Threads of every kernel here: warp 0 issues, all write back.
const THREADS: usize = 128;

/// A convolution's geometry: extents and parameters per spatial dimension, W first.
#[derive(Clone, Debug)]
pub struct Conv {
    pub n: usize,
    /// W, H, D (as many as the convolution has).
    pub extent: Vec<usize>,
    pub c: usize,
    /// S, R, T.
    pub filter: Vec<usize>,
    pub pad_lo: Vec<usize>,
    pub pad_hi: Vec<usize>,
    pub stride: Vec<usize>,
    pub dilation: Vec<usize>,
}

impl Conv {
    /// The output extent per spatial dimension (Q, P, Z).
    pub fn out(&self) -> Vec<usize> {
        (0..self.extent.len())
            .map(|i| {
                (self.extent[i] + self.pad_lo[i] + self.pad_hi[i] - (self.filter[i] - 1) * self.dilation[i] - 1)
                    / self.stride[i]
                    + 1
            })
            .collect()
    }

    /// GEMM rows: output pixels.
    pub fn m(&self) -> usize {
        self.n * self.out().iter().product::<usize>()
    }

    /// GEMM columns: filter taps times channels.
    pub fn k(&self) -> usize {
        self.c * self.filter.iter().product::<usize>()
    }

    /// The input's layout `((w, h, d, n), c)`, densely packed (NHWC / NDHWC).
    fn input_layout(&self) -> Layout {
        let mut shape: Vec<usize> = self.extent.clone();
        shape.push(self.n);
        let mut stride = Vec::new();
        let mut s = self.c;
        for e in &shape {
            stride.push(s);
            s *= e;
        }
        let tup = |v: &[usize]| Tree::Branch(v.iter().map(|&x| Tree::Leaf(IndexExpr::Static(x))).collect());
        Layout::new(
            Tree::Branch(vec![tup(&shape), Tree::Leaf(IndexExpr::Static(self.c))]),
            Tree::Branch(vec![tup(&stride), Tree::Leaf(IndexExpr::Static(1))]),
        )
    }

    fn options(&self) -> cpasync::Im2colOptions {
        let i = |v: &[usize]| v.iter().map(|&x| x as i32).collect::<Vec<_>>();
        cpasync::Im2colOptions::conv(
            &i(&self.filter),
            &i(&self.pad_lo),
            &i(&self.pad_hi),
            &i(&self.stride),
            &i(&self.dilation),
        )
    }
}

/// The GEMM operand `A` of `conv` over the NHWC/NDHWC `input` in plain Rust: row
/// `m = (q, p, z, n)` (q fastest), column `k = (c, s, r, t)` (c fastest), zero where
/// the filter tap lands in the padding; `rows` rows, those past the last pixel zero.
pub fn im2col_reference(conv: &Conv, input: &[u16], rows: usize) -> Vec<u16> {
    let (out, k) = (conv.out(), conv.k());
    let mut a = vec![0u16; rows * k];
    for m in 0..rows.min(conv.m()) {
        let mut rest = m;
        let mut o = Vec::new();
        for e in &out {
            o.push(rest % e);
            rest /= e;
        }
        let n = rest;
        for kk in 0..k {
            let c = kk % conv.c;
            let mut tap = kk / conv.c;
            let mut idx = 0usize;
            let mut inside = true;
            let mut scale = conv.c;
            let mut pos = Vec::new();
            for (i, &oi) in o.iter().enumerate() {
                let s = tap % conv.filter[i];
                tap /= conv.filter[i];
                let x = (oi * conv.stride[i] + s * conv.dilation[i]) as i64 - conv.pad_lo[i] as i64;
                inside &= x >= 0 && (x as usize) < conv.extent[i];
                pos.push(x);
            }
            if !inside {
                continue;
            }
            idx += c;
            for (i, &p) in pos.iter().enumerate() {
                idx += p as usize * scale;
                scale *= conv.extent[i];
            }
            idx += n * scale;
            a[m * k + kk] = input[idx];
        }
    }
    a
}

/// A kernel loading the convolution's operand `A` in `(tile_m, tile_k)` tiles through
/// `kind`'s SMEM layout atom (one tile per block: `x` the M tile, `y` the K tile) and
/// writing each back to `out`, a row-major `(rows, K)` matrix (`rows` M rounded up to
/// the tile).
fn load_kernel(name: &'static str, conv: Conv, tile_m: usize, tile_k: usize, kind: SmemLayoutAtomKind) -> Kernel {
    let rows = conv.m().div_ceil(tile_m) * tile_m;
    let k = conv.k();
    Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).build(move || {
        let x = param_tensor_aligned::<F16>("x", conv.input_layout(), 16);
        let out = param_tensor::<F16>("out", Layout::new((rows, k), (k, 1)));
        let smem = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(kind), (tile_m, tile_k));
        let op = cpasync::CopyBulkTensorIm2ColG2SOp::default();
        let (atom, tma_tensor) =
            cpasync::make_im2col_tma_atom(op, &x, smem.clone(), (tile_m, tile_k), &conv.options()).into();
        let s = make_smem_tensor_aligned::<F16>(smem, 1024);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let (bx, by) = (block_idx().x, block_idx().y);
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let g = local_tile(&tma_tensor, (tile_m, tile_k), (bx, by));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        if_(warp_idx().eq(0), || {
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, (tile_m * tile_k * 2) as i32));
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar));
        });
        mbarrier_wait(&mbar, 0);
        for_range(0, (tile_m * tile_k / THREADS) as i32, 1, |j| {
            let idx = j * THREADS as i32 + t;
            let (m, kk) = (idx / tile_k as i32, idx % tile_k as i32);
            out.store((bx * tile_m as i32 + m, by * tile_k as i32 + kk), s.load((m, kk)));
        });
    })
}

/// A load case: the kernel, its grid, and the check against [`im2col_reference`].
fn load_case(name: &'static str, conv: Conv, tile_m: usize, tile_k: usize, kind: SmemLayoutAtomKind) -> Case {
    let grid = [conv.m().div_ceil(tile_m) as u32, (conv.k() / tile_k) as u32, 1];
    let kernel = load_kernel(name, conv, tile_m, tile_k, kind);
    let cfg = LaunchConfig::new(grid, [THREADS as u32, 1, 1]);
    // The check needs the geometry: the cases are named after it.
    let check: fn(&[KernelArg]) = match name {
        "tma_im2col_2d_fprop" => |args| check_load(&conv_2d(), 64, args),
        "tma_im2col_2d_strided_dilated" => |args| check_load(&conv_2d_strided(), 32, args),
        "tma_im2col_3d_fprop" => |args| check_load(&conv_3d(), 64, args),
        _ => unreachable!("an im2col load case"),
    };
    Case { name, kernel, cfg, small_only: false, check }
}

fn check_load(conv: &Conv, tile_m: usize, args: &[KernelArg]) {
    let rows = conv.m().div_ceil(tile_m) * tile_m;
    let x: Vec<u16> = args[0].to_vec::<F16>().iter().map(|v| v.0).collect();
    let out: Vec<u16> = args[1].to_vec::<F16>().iter().map(|v| v.0).collect();
    let want = im2col_reference(conv, &x, rows);
    let k = conv.k();
    for m in 0..rows {
        for kk in 0..k {
            assert_eq!(out[m * k + kk], want[m * k + kk], "A({m}, {kk}) of {conv:?}");
        }
    }
}

/// A 3x3 convolution with padding 1 over a 2-image, 5x6, 32-channel NHWC tensor.
pub fn conv_2d() -> Conv {
    Conv {
        n: 2,
        extent: vec![6, 5],
        c: 32,
        filter: vec![3, 3],
        pad_lo: vec![1, 1],
        pad_hi: vec![1, 1],
        stride: vec![1, 1],
        dilation: vec![1, 1],
    }
}

/// A 3x3 convolution with stride 2, dilation 2 and uneven padding (2 before, 1 after)
/// over a 2-image, 7x9, 16-channel NHWC tensor.
pub fn conv_2d_strided() -> Conv {
    Conv {
        n: 2,
        extent: vec![9, 7],
        c: 16,
        filter: vec![3, 3],
        pad_lo: vec![2, 2],
        pad_hi: vec![1, 1],
        stride: vec![2, 2],
        dilation: vec![2, 2],
    }
}

/// A 3x2x2 (SxRxT) convolution over a 3x4x5 (DxHxW), 16-channel NDHWC image, padded
/// differently per dimension.
pub fn conv_3d() -> Conv {
    Conv {
        n: 1,
        extent: vec![5, 4, 3],
        c: 16,
        filter: vec![3, 2, 2],
        pad_lo: vec![1, 0, 1],
        pad_hi: vec![1, 1, 0],
        stride: vec![1, 1, 1],
        dilation: vec![1, 1, 1],
    }
}

/// [`conv_2d`]'s operand in 64x32 tiles through the 64-byte swizzle (M = 60, so the last
/// four rows of the tile are past the last image).
pub fn tma_im2col_2d_fprop() -> Case {
    load_case("tma_im2col_2d_fprop", conv_2d(), 64, 32, SmemLayoutAtomKind::K_SW64)
}

/// [`conv_2d_strided`]'s operand in 32x16 tiles through the 32-byte swizzle.
pub fn tma_im2col_2d_strided_dilated() -> Case {
    load_case("tma_im2col_2d_strided_dilated", conv_2d_strided(), 32, 16, SmemLayoutAtomKind::K_SW32)
}

/// [`conv_3d`]'s operand in 64x16 tiles through the 32-byte swizzle.
pub fn tma_im2col_3d_fprop() -> Case {
    load_case("tma_im2col_3d_fprop", conv_3d(), 64, 16, SmemLayoutAtomKind::K_SW32)
}

/// The value the store kernel writes at output pixel `m`, channel `k`: small integers,
/// exact in f16.
fn store_value(m: usize, k: usize) -> f32 {
    ((m * 5 + k * 3) % 53) as f32 - 26.0
}

/// The output of a 2-image, 5x6, 32-channel NHWC convolution (M = 60 pixels), written
/// by an im2col store (`.im2col_no_offs`) from a 64x32 SMEM tile through the 64-byte
/// swizzle: the tile's last four pixels are past the last image and are not written.
pub fn tma_im2col_store() -> Case {
    const N: usize = 2;
    const P: usize = 5;
    const Q: usize = 6;
    const C: usize = 32;
    const TM: usize = 64;
    let kernel = Kernel::builder("tma_im2col_store").block_dim([THREADS as u32, 1, 1]).build(|| {
        let layout = Layout::new(((Q, P, N), C), ((C, Q * C, P * Q * C), 1));
        let out = param_tensor_aligned::<F16>("out", layout, 16);
        let smem = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW64), (TM, C));
        let op = cpasync::CopyBulkTensorIm2ColS2GOp;
        let (atom, tma_tensor) =
            cpasync::make_im2col_tma_atom(op, &out, smem.clone(), (TM, C), &cpasync::Im2colOptions::new()).into();
        let s = make_smem_tensor_aligned::<F16>(smem, 1024);
        let t = thread_idx().x;
        for_range(0, (TM * C / THREADS) as i32, 1, |j| {
            let idx = j * THREADS as i32 + t;
            let (m, k) = (idx / C as i32, idx % C as i32);
            s.store((m, k), ((m * 5 + k * 3) % 53 - 26).to::<f32>().to::<F16>());
        });
        fence_view_async_shared();
        sync_threads();
        let g = local_tile(&tma_tensor, (TM, C), (0, 0));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        if_(warp_idx().eq(0), || {
            copy(&atom, &ts, &tg);
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, false);
        });
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<F16>();
        for m in 0..N * P * Q {
            for k in 0..C {
                assert_eq!(out[m * C + k].to_f32(), store_value(m, k), "pixel {m}, channel {k}");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_im2col_store", kernel, cfg, small_only: false, check }
}

/// The im2col cases, for the sm_90 list.
pub fn tma_im2col_cases_sm90() -> Vec<Case> {
    vec![tma_im2col_2d_fprop(), tma_im2col_2d_strided_dilated(), tma_im2col_3d_fprop(), tma_im2col_store()]
}
