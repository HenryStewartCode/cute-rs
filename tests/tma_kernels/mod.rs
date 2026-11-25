//! Kernels with tensor-map and grid-constant parameters, shared by the simulator tests
//! (`tests/tma_desc.rs`) and the GPU tests (`tests/gpu_tma.rs`).

#![allow(dead_code)]

use cute::prelude::*;

/// A hidden descriptor over `a`, a row-major `(m, 64)` f16 matrix with a runtime row count
/// and leading dimension; thread 0 copies the descriptor's 16 words to `out` and its
/// generic address to `addr`.
pub fn hidden_desc_kernel() -> Kernel {
    Kernel::builder("hidden_desc").block_dim([32, 1, 1]).build(|| {
        let m = param_extent("m", 1);
        let ld = param_extent("ld", 8);
        let a = param_tensor::<F16>("a", Layout::new((m, 64), (ld, 1)));
        let out = param_tensor::<u64>("out", Layout::col_major((16,)));
        let addr = param_tensor::<i64>("addr", Layout::col_major((1,)));
        let opts = TensorMapOptions::new().swizzle(TensorMapSwizzle::s128b);
        let desc = tma_descriptor(&a, &[8, 64], None, &opts);
        if_(thread_idx().x.eq(0), || {
            for k in 0..16 {
                out.store(k, desc.load_word(k));
            }
            addr.store(0, desc.get_ptr());
        });
    })
}

pub fn hidden_desc_args(m: i32, ld: i32) -> Vec<KernelArg> {
    let rows = m.max(1) as usize;
    vec![
        KernelArg::scalar(m),
        KernelArg::scalar(ld),
        KernelArg::f16_from_f32(&vec![0.0; rows * ld.max(64) as usize]),
        KernelArg::buffer(&[0u64; 16]),
        KernelArg::buffer(&[0i64; 1]),
    ]
}

/// A host-bound descriptor: thread 0 copies its 16 words to `out`.
pub fn explicit_desc_kernel() -> Kernel {
    Kernel::builder("explicit_desc").block_dim([32, 1, 1]).build(|| {
        let desc = param_tensormap("desc");
        let out = param_tensor::<u64>("out", Layout::col_major((16,)));
        if_(thread_idx().x.eq(0), || {
            for k in 0..16 {
                out.store(k, desc.load_word(k));
            }
        });
    })
}

/// A 128-byte grid-constant parameter whose bytes thread `t < 16` copies to `out[t]`
/// through the parameter's generic address (`cvta.param`, a generic load).
pub fn grid_constant_copy_kernel() -> Kernel {
    Kernel::builder("grid_constant_copy").block_dim([32, 1, 1]).build(|| {
        let p = param_bytes("blob", 128, 64, true);
        let out = param_tensor::<u64>("out", Layout::col_major((16,)));
        let t = thread_idx().x;
        if_(t.lt(16), || {
            let base = p.address();
            let at = make_ptr::<u64>(base + t.to::<i64>() * 8, AddressSpace::Generic, Some(8));
            out.store(t, at.load());
        });
    })
}

pub fn grid_constant_blob() -> Vec<u8> {
    (0..128u32).map(|i| (i * 37 + 11) as u8).collect()
}

/// The descriptor's generic address handed to (test-only) inline PTX that copies it.
pub fn desc_address_to_inline_ptx_kernel() -> Kernel {
    Kernel::builder("desc_inline_ptx").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::row_major((128, 32)));
        let out = param_tensor::<u64>("out", Layout::col_major((1,)));
        let desc = tma_descriptor(&a, &[16, 32], None, &TensorMapOptions::new());
        let moved: Uint64 =
            InlinePtx::new("mov.b64 {$w0}, {$r0};").write::<u64>().read(desc.get_ptr()).sim(|r, _| vec![r[0]]).emit();
        if_(thread_idx().x.eq(0), || out.store(0, moved));
    })
}

/// A TMA atom (roadmap part m7-tma-atom) over `a`, a row-major `(m, k)` f16 matrix with
/// runtime extents, for 128x64 tiles in the 128-byte-swizzled K-major layout: thread 0
/// copies the atom's hidden descriptor's 16 words to `out`.
pub fn tma_atom_desc_kernel() -> Kernel {
    Kernel::builder("tma_atom_desc").block_dim([32, 1, 1]).build(|| {
        let m = param_extent("m", 1);
        let k = param_extent("k", 8);
        let a = param_tensor::<F16>("a", Layout::new((m, k), (k, 1)));
        let out = param_tensor::<u64>("out", Layout::col_major((16,)));
        let smem = tile_to_shape_swizzled(
            &make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::new((8, 64), (64, 1))),
            (128, 64),
        );
        let info = cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SOp::default(), &a, smem, (128, 64), 1);
        let desc = info.atom.tma_info().unwrap().descriptor().clone();
        if_(thread_idx().x.eq(0), || {
            for w in 0..16 {
                out.store(w, desc.load_word(w));
            }
        });
    })
}

pub fn tma_atom_desc_args(m: i32, k: i32) -> Vec<KernelArg> {
    vec![
        KernelArg::scalar(m),
        KernelArg::scalar(k),
        KernelArg::f16_from_f32(&vec![0.0; (m.max(1) * k.max(1)) as usize]),
        KernelArg::buffer(&[0u64; 16]),
    ]
}
