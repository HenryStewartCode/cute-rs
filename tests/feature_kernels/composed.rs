// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tensors over CuTe's general composed layouts (`inner o offset o outer`
//! with a layout inner) and over swizzled pointers, held to plain Rust.

use cute::prelude::*;

use super::Case;

const N: usize = 64;

/// A shared-memory tensor over `(8,8):(8,1) o 0 o (8,8):(1,8)`: element `c`
/// of the column-major outer layout lands at the transposed offset. Each
/// thread writes its element through a slice (a column, whose offset moves
/// into the composed offset), and reads the buffer back through a plain
/// view of the same pointer.
pub fn composed_layout_inner() -> Case {
    let kernel = Kernel::builder("composed_layout_inner").block_dim([N as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(N));
        let y = param_tensor::<f32>("y", Layout::col_major(N));
        let composed = make_composed_layout(Layout::new((8, 8), (8, 1)), 0, Layout::col_major((8, 8)));
        let s = make_smem_tensor::<f32>(composed);
        let t = thread_idx().x;
        let column = s.slice((.., t / 8));
        column.store(t % 8, x.load(t));
        sync_threads();
        let plain = make_tensor(&s.ptr(), Layout::col_major(N));
        y.store(t, plain.load(t));
    });
    fn check(args: &[KernelArg]) {
        let (x, y) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for t in 0..N {
            assert_eq!(y[(t % 8) * 8 + t / 8], x[t], "composed_layout_inner: element {t}");
        }
    }
    Case {
        name: "composed_layout_inner",
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]),
        small_only: false,
        check,
    }
}

/// The swizzle `S<2,4,3>` on bytes, carried by a shared-memory pointer
/// (`recast_ptr`): on f32 elements it is `S<2,2,3>`. Each thread writes a
/// column of an `(8, 64)` row-major tile, then copies its row out with
/// `autovec_copy` (in vectors of the swizzle's four-element runs), and the
/// raw buffer goes out through the pointer with its swizzle dropped.
pub fn pointer_swizzle() -> Case {
    let kernel = Kernel::builder("pointer_swizzle").block_dim([N as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::row_major((8, N)));
        let y = param_tensor::<f32>("y", Layout::row_major((8, N)));
        let raw = param_tensor::<f32>("raw", Layout::col_major(8 * N));
        let base = alloc_smem::<f32>(8 * N, 512);
        let s = make_tensor(&recast_ptr::<f32, _>(&base, Some(make_swizzle(2, 4, 3))), Layout::row_major((8, N)));
        let t = thread_idx().x;
        for i in 0..8 {
            s.store((i, t), x.load((i, t)));
        }
        sync_threads();
        let row = t % 8;
        let frag = make_fragment_like::<f32, f32>(&s.slice((row, ..)));
        autovec_copy(&s.slice((row, ..)), &frag);
        if_(t.lt(8), || autovec_copy(&frag, &y.slice((row, ..))));
        let plain = make_tensor(&recast_ptr::<f32, _>(&s.ptr(), None), Layout::col_major(8 * N));
        for j in 0..8 {
            raw.store(t + N as i32 * j, plain.load(t + N as i32 * j));
        }
    });
    fn check(args: &[KernelArg]) {
        let (x, y, raw) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>(), args[2].to_vec::<f32>());
        assert_eq!(x, y, "pointer_swizzle: rows");
        let sw = Swizzle::new(2, 2, 3);
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(raw[sw.apply(i)], v, "pointer_swizzle: element {i}");
        }
    }
    Case {
        name: "pointer_swizzle",
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]),
        small_only: false,
        check,
    }
}
