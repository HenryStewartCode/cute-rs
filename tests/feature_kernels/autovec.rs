// SPDX-License-Identifier: MIT OR Apache-2.0

//! `autovec_copy` on `max_common_layout`: tiles and the fragments
//! `make_fragment_like` gives them, held to plain Rust.

use cute::prelude::*;

use super::Case;

/// Rows per thread of the row-major kernels.
pub const AUTOVEC_ROWS: usize = 4;
/// Columns of the row-major kernels.
pub const AUTOVEC_COLS: usize = 8;
/// Threads of the autovec kernels.
const THREADS: usize = 32;

/// `2x + 1`, the change every autovec kernel makes in registers.
fn bump(frag: &Tensor<f32>) {
    frag.map_in_place(|v| v * 2.0 + 1.0);
}

fn check_bumped(name: &str, x: &[f32], y: &[f32], at: impl Iterator<Item = usize>) {
    for i in at {
        assert_eq!(y[i], x[i] * 2.0 + 1.0, "{name}: element {i}");
    }
}

/// Each thread's `AUTOVEC_ROWS x AUTOVEC_COLS` row-major tile copied into
/// the fragment `make_fragment_like` gives it (row-major too) and back with
/// `autovec_copy`: the common run is the whole tile, so 16-byte vectors
/// (the buffers' alignment) move it, where a column-major fragment would
/// copy one element at a time.
pub fn autovec_row_major() -> Case {
    let kernel = Kernel::builder("autovec_row_major").block_dim([THREADS as u32, 1, 1]).build(|| {
        let shape = (THREADS * AUTOVEC_ROWS, AUTOVEC_COLS);
        let x = param_tensor::<f32>("x", Layout::row_major(shape));
        let y = param_tensor::<f32>("y", Layout::row_major(shape));
        let t = thread_idx().x;
        let tile = |g: &Tensor<f32>| local_tile(g, (AUTOVEC_ROWS, AUTOVEC_COLS), (t, 0));
        let frag = make_fragment_like::<f32, f32>(&tile(&x));
        autovec_copy(&tile(&x), &frag);
        bump(&frag);
        autovec_copy(&frag, &tile(&y));
    });
    fn check(args: &[KernelArg]) {
        let (x, y) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        check_bumped("autovec_row_major", &x, &y, 0..x.len());
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "autovec_row_major", kernel, cfg, small_only: false, check }
}

/// Rows `4t..4t+4` of a tile whose row stride `k` is a runtime value (a
/// `param_extent` that is a multiple of 8): the common run is one row, and
/// the stride's known divisor proves each row's vectors aligned.
pub fn autovec_runtime_stride() -> Case {
    const ROWS: usize = 4;
    let kernel = Kernel::builder("autovec_runtime_stride").block_dim([ROWS as u32, 1, 1]).build(|| {
        let k = param_extent("k", 8);
        let layout = Layout::with_strides((ROWS * ROWS, AUTOVEC_COLS), (k, 1));
        let x = param_tensor::<f32>("x", layout.clone());
        let y = param_tensor::<f32>("y", layout);
        let t = thread_idx().x;
        let tile = |g: &Tensor<f32>| local_tile(g, (ROWS, AUTOVEC_COLS), (t, 0));
        let frag = make_fragment_like::<f32, f32>(&tile(&x));
        autovec_copy(&tile(&x), &frag);
        bump(&frag);
        autovec_copy(&frag, &tile(&y));
    });
    fn check(args: &[KernelArg]) {
        let KernelArg::ScalarI32(k) = args[0] else { panic!("autovec_runtime_stride takes k first") };
        let (x, y) = (args[1].to_vec::<f32>(), args[2].to_vec::<f32>());
        let at = (0..ROWS * ROWS).flat_map(|r| (0..AUTOVEC_COLS).map(move |c| r * k as usize + c));
        check_bumped("autovec_runtime_stride", &x, &y, at);
    }
    let cfg = LaunchConfig::new([1, 1, 1], [ROWS as u32, 1, 1]);
    Case { name: "autovec_runtime_stride", kernel, cfg, small_only: false, check }
}

/// Columns of [`autovec_swizzled`]'s shared tile.
pub const SWIZZLED_COLS: usize = 32;

/// A row-major `THREADS x 32` tile through swizzled shared memory
/// (`S<3,2,3>`, runs of four elements): each thread reads its row into
/// registers with `autovec_copy`, which the swizzle's runs hold to
/// 16-byte vectors.
pub fn autovec_swizzled() -> Case {
    let kernel = Kernel::builder("autovec_swizzled").block_dim([THREADS as u32, 1, 1]).build(|| {
        let shape = (THREADS, SWIZZLED_COLS);
        let x = param_tensor::<f32>("x", Layout::row_major(shape));
        let y = param_tensor::<f32>("y", Layout::row_major(shape));
        let smem = make_smem_tensor::<f32>(make_composed_layout(make_swizzle(3, 2, 3), 0, Layout::row_major(shape)));
        let t = thread_idx().x;
        // Fill the tile column by column, so that a thread writes other rows.
        for c in 0..SWIZZLED_COLS {
            smem.store((t, c), x.load((t, c)));
        }
        sync_threads();
        let row = smem.slice((t, ..));
        let frag = make_fragment_like::<f32, f32>(&row);
        autovec_copy(&row, &frag);
        bump(&frag);
        autovec_copy(&frag, &y.slice((t, ..)));
    });
    fn check(args: &[KernelArg]) {
        let (x, y) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        check_bumped("autovec_swizzled", &x, &y, 0..x.len());
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "autovec_swizzled", kernel, cfg, small_only: false, check }
}

/// [`autovec_row_major`] with L1 eviction priorities
/// (`autovec_copy_with`): `evict_first` loads and `no_allocate` stores, as
/// CopyG2ROp and CopyR2GOp atoms of the same width.
pub fn autovec_evict() -> Case {
    let kernel = Kernel::builder("autovec_evict").block_dim([THREADS as u32, 1, 1]).build(|| {
        let shape = (THREADS * AUTOVEC_ROWS, AUTOVEC_COLS);
        let x = param_tensor::<f32>("x", Layout::row_major(shape));
        let y = param_tensor::<f32>("y", Layout::row_major(shape));
        let t = thread_idx().x;
        let tile = |g: &Tensor<f32>| local_tile(g, (AUTOVEC_ROWS, AUTOVEC_COLS), (t, 0));
        let frag = make_fragment_like::<f32, f32>(&tile(&x));
        let with = |l1c_evict_priority| AutovecOptions { l1c_evict_priority };
        autovec_copy_with(&tile(&x), &frag, with(CacheEvictionPriority::EvictFirst));
        bump(&frag);
        autovec_copy_with(&frag, &tile(&y), with(CacheEvictionPriority::NoAllocate));
    });
    fn check(args: &[KernelArg]) {
        let (x, y) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        check_bumped("autovec_evict", &x, &y, 0..x.len());
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "autovec_evict", kernel, cfg, small_only: false, check }
}

/// One row of eight `f32` per thread in 32-byte-aligned buffers
/// (`param_tensor_aligned`): `autovec_copy` issues 32-byte vectors, which
/// LLVM splits into two 16-byte accesses below sm_100.
pub fn autovec_256() -> Case {
    let kernel = Kernel::builder("autovec_256").block_dim([THREADS as u32, 1, 1]).build(|| {
        let shape = (THREADS, 8);
        let x = param_tensor_aligned::<f32>("x", Layout::row_major(shape), 32);
        let y = param_tensor_aligned::<f32>("y", Layout::row_major(shape), 32);
        let t = thread_idx().x;
        let frag = make_fragment_like::<f32, f32>(&x.slice((t, ..)));
        autovec_copy(&x.slice((t, ..)), &frag);
        bump(&frag);
        autovec_copy(&frag, &y.slice((t, ..)));
    });
    fn check(args: &[KernelArg]) {
        let (x, y) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        check_bumped("autovec_256", &x, &y, 0..x.len());
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "autovec_256", kernel, cfg, small_only: false, check }
}

/// Every autovec kernel, which the SM89 GPU runs too.
pub fn autovec_cases() -> Vec<Case> {
    vec![autovec_row_major(), autovec_runtime_stride(), autovec_swizzled(), autovec_evict(), autovec_256()]
}
