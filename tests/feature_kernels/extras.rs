// SPDX-License-Identifier: MIT OR Apache-2.0

//! The layout extras: `autovec_copy` bounded by a layout's `max_alignment`
//! of a divisibility-tagged runtime extent, the general `make_tensor` and a
//! gather through a composed layout with a closure inner, held to plain
//! Rust.

use cute::prelude::*;

use super::Case;

/// Threads of the extras kernels.
const THREADS: usize = 4;
/// Rows each thread's tile has.
const ROWS: usize = 4;
/// Columns of the column-major matrix.
const COLS: usize = 2;

/// Each thread's `ROWS x COLS` tile of a column-major `m x COLS` matrix,
/// whose column stride `m` is a `param_extent` promised a multiple of 4,
/// copied into registers and back with `autovec_copy`. The common run is one
/// column of the tile, and the tile's `max_alignment` is 4 only because `m`
/// is known to be a multiple of 4 (it would be 1 otherwise), so each
/// column moves as one 16-byte vector.
pub fn autovec_max_alignment() -> Case {
    let kernel = Kernel::builder("autovec_max_alignment").block_dim([THREADS as u32, 1, 1]).build(|| {
        let m = param_extent("m", 4);
        let x = param_tensor::<f32>("x", Layout::col_major((m, COLS)));
        let y = param_tensor::<f32>("y", Layout::col_major((m, COLS)));
        let t = thread_idx().x;
        let tile = |g: &Tensor<f32>| local_tile(g, (ROWS, COLS), (t, 0));
        assert_eq!(tile(&x).layout().max_alignment(), 4);
        let frag = make_fragment_like::<f32, f32>(&tile(&x));
        autovec_copy(&tile(&x), &frag);
        frag.map_in_place(|v| v * 2.0 + 1.0);
        autovec_copy(&frag, &tile(&y));
    });
    fn check(args: &[KernelArg]) {
        let KernelArg::ScalarI32(m) = args[0] else { panic!("autovec_max_alignment takes m first") };
        let (x, y) = (args[1].to_vec::<f32>(), args[2].to_vec::<f32>());
        for c in 0..COLS {
            for r in 0..THREADS * ROWS {
                let i = r + c * m as usize;
                assert_eq!(y[i], x[i] * 2.0 + 1.0, "autovec_max_alignment: element ({r},{c})");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "autovec_max_alignment", kernel, cfg, small_only: false, check }
}

/// Elements of the `make_tensor` kernel's output per thread.
const FORMS: usize = 4;

/// `make_tensor` over a pointer with a shape (column-major), over a shared
/// pointer with a composed layout whose inner is a layout (a transpose),
/// and over an origin (a coordinate tensor), each thread writing what it
/// reads.
pub fn make_tensor_forms() -> Case {
    const N: usize = 32;
    let kernel = Kernel::builder("make_tensor_forms").block_dim([N as u32, 1, 1]).build(|| {
        let x = param_ptr::<i32>("x");
        let out = param_tensor::<i32>("out", Layout::col_major((FORMS, N)));
        let t = thread_idx().x;
        // A shape: (8, 4) column-major over x.
        let by_shape = make_tensor(&x, (8, 4));
        let (r, c) = (t % 8, t / 8);
        out.store((0, t), by_shape.load((r, c)));
        // A composed layout: (8,4):(1,8) read through the transpose
        // (4,8):(8,1) inner, over shared memory.
        let smem = alloc_smem::<i32>(N, 16);
        make_tensor(&smem, Layout::col_major(N)).store(t, x.load_at(t) + 1);
        sync_threads();
        let inner = make_layout((8, 4), (4, 1));
        let transposed = make_tensor(&smem, make_composed_layout(inner, 0, Layout::col_major((8, 4))));
        out.store((1, t), transposed.load((r, c)));
        // An origin: the coordinate tensor (t, 100) + (8,4):(1@0, 1@1).
        let coords = make_tensor((t, 100), make_identity_layout((8, 4)));
        let at = coords.coord((r, c));
        let as_int = |e: &Coord| -> Int32 { e.as_leaf().expect("an integer").clone().into_value() };
        out.store((2, t), as_int(at.mode(0)));
        out.store((3, t), as_int(at.mode(1)));
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for t in 0..N {
            let (r, c) = (t % 8, t / 8);
            let at = |k: usize| out[k + FORMS * t];
            assert_eq!(at(0), x[r + 8 * c], "make_tensor_forms: shape, thread {t}");
            assert_eq!(at(1), x[4 * r + c] + 1, "make_tensor_forms: composed, thread {t}");
            assert_eq!((at(2), at(3)), ((t + r) as i32, (100 + c) as i32), "make_tensor_forms: origin, thread {t}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]);
    Case { name: "make_tensor_forms", kernel, cfg, small_only: false, check }
}

/// Elements the gather kernel reads.
pub const GATHER: usize = 16;

/// The `composed_layout.ipynb` gather: a composed layout whose inner
/// closure reads an index tensor (`offset_tensor[c]`), over
/// `make_layout(16)`. Each thread `i` reads `data[offset[i]]` through it.
/// Its slice of a tile ([`FnComposedLayout::slice`]) gives the same
/// offsets.
pub fn fn_composed_gather() -> Case {
    let kernel = Kernel::builder("fn_composed_gather").block_dim([GATHER as u32, 1, 1]).build(|| {
        let offsets = param_tensor::<i32>("offsets", Layout::col_major(GATHER));
        let data = param_tensor::<i32>("data", Layout::col_major(256));
        let out = param_tensor::<i32>("out", Layout::col_major((GATHER, 2)));
        // Keep the indices in 0..256.
        let inner = |c: Coord| -> Int32 { offsets.load(c) & 255 };
        let gather = make_composed_layout_fn(inner, 0, make_layout(GATHER, 1));
        let i = thread_idx().x;
        out.store((i, 0), data.load(IndexExpr::from(gather.call(i))));
        // Mode 1 of zipped_divide(gather, 4) fixed at i / 4: the tile of i.
        let tile = zipped_divide(&gather, 4).slice((.., i / 4));
        out.store((i, 1), data.load(IndexExpr::from(tile.call(i % 4))));
    });
    fn check(args: &[KernelArg]) {
        let (offsets, data, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        for i in 0..GATHER {
            let want = data[(offsets[i] & 255) as usize];
            assert_eq!((out[i], out[i + GATHER]), (want, want), "fn_composed_gather: element {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [GATHER as u32, 1, 1]);
    Case { name: "fn_composed_gather", kernel, cfg, small_only: false, check }
}

/// Every layout-extras kernel, which the SM89 GPU runs too.
pub fn extras_cases() -> Vec<Case> {
    vec![autovec_max_alignment(), make_tensor_forms(), fn_composed_gather()]
}
