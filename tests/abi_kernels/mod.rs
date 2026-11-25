//! Kernels for the launch ABI tests (`tests/launch_abi.rs` on the simulator, `tests/gpu_launch.rs`
//! on a GPU): a by-value struct, module globals, launch-time shared memory, shared-memory
//! overlays, preferred and fallback clusters, and launch bounds.
//!
//! `globals_kernel` ports the module-global part of CUTLASS 4.8.1's
//! `examples/python/CuTeDSL/notebooks/2_primitives/01_array_concepts.ipynb`, and
//! `static_smem_kernel` the kernels of `examples/python/CuTeDSL/dsl_tutorials/dynamic_smem_size.py`,
//! Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//! BSD-3-Clause; see THIRD-PARTY-NOTICES.md.

#![allow(dead_code)]

use cute::prelude::*;

/// A C struct of every scalar width at natural alignment: `a` at 0, `b` at 4, `c` at 8, `d` at
/// 16, `e` at 24; 32 bytes aligned to 8.
pub fn struct_layout() -> NativeLayout {
    NativeLayout::new(
        vec![
            ("a".into(), DType::I8),
            ("b".into(), DType::I32),
            ("c".into(), DType::I64),
            ("d".into(), DType::F32),
            ("e".into(), DType::F64),
        ],
        false,
    )
}

pub const STRUCT_C: i64 = -(1 << 40) - 7;

/// Reads every field of a by-value struct and writes them out widened.
pub fn struct_kernel() -> Kernel {
    Kernel::builder("byval_struct").block_dim([32, 1, 1]).build(|| {
        let s = param_struct("s", &struct_layout());
        let ints = param_tensor::<i64>("ints", Layout::col_major((3,)));
        let floats = param_tensor::<f64>("floats", Layout::col_major((2,)));
        if_(thread_idx().x.eq(0), || {
            ints.store(0, s.field::<i8>("a").to::<i64>());
            ints.store(1, s.field::<i32>("b").to::<i64>());
            ints.store(2, s.field::<i64>("c"));
            floats.store(0, s.field::<f32>("d").to::<f64>());
            floats.store(1, s.field::<f64>("e"));
        });
    })
}

/// The struct's bytes for `struct_kernel`.
pub fn struct_bytes() -> Vec<u8> {
    struct_layout()
        .pack(&[
            KernelArg::scalar(-5i8),
            KernelArg::scalar(123_456i32),
            KernelArg::scalar(STRUCT_C),
            KernelArg::scalar(1.25f32),
            KernelArg::scalar(std::f64::consts::PI),
        ])
        .expect("the values match the fields")
}

pub fn struct_args() -> Vec<KernelArg> {
    vec![KernelArg::by_value(struct_bytes()), KernelArg::buffer(&[0i64; 3]), KernelArg::buffer(&[0f64; 2])]
}

pub fn check_struct(args: &[KernelArg]) {
    assert_eq!(args[1].to_vec::<i64>(), [-5, 123_456, STRUCT_C]);
    assert_eq!(args[2].to_vec::<f64>(), [1.25, std::f64::consts::PI]);
}

/// A 128-byte grid-constant parameter (a TMA descriptor's shape), of which the kernel reads the
/// first and last 8 bytes.
pub fn grid_constant_kernel() -> Kernel {
    Kernel::builder("grid_constant_desc").block_dim([32, 1, 1]).build(|| {
        let desc = param_bytes("desc", 128, 64, true);
        let out = param_tensor::<u64>("out", Layout::col_major((2,)));
        if_(thread_idx().x.eq(0), || {
            out.store(0, desc.load::<u64>(0));
            out.store(1, desc.load::<u64>(120));
        });
    })
}

/// The constant table and the launch counter of the notebook: `out[i] = inp[i] * gamma[i]`,
/// and thread 0 bumps `run_count`, which persists across launches, and reports it in `seen`.
pub fn globals_kernel() -> Kernel {
    Kernel::builder("scale_kernel").block_dim([4, 1, 1]).build(|| {
        let inp = param_tensor::<f32>("inp", Layout::col_major((4,)));
        let out = param_tensor::<f32>("out", Layout::col_major((4,)));
        let seen = param_tensor::<i32>("seen", Layout::col_major((1,)));
        let gamma = module_global::<f32>(Some("gamma"), 4, AddressSpace::Cmem, Some(&[1.0, 1.5, 2.0, 2.5]), false);
        let run_count = module_global::<i32>(Some("run_count"), 1, AddressSpace::Gmem, None, false);
        let i = thread_idx().x;
        out.store(i, inp.load(i) * gamma.load(i));
        if_(i.eq(0), || {
            run_count.store(0, run_count.load(0) + 1);
            seen.store(0, run_count.load(0));
        });
    })
}

pub fn globals_args() -> Vec<KernelArg> {
    vec![KernelArg::f32(vec![2.0, 4.0, 6.0, 8.0]), KernelArg::f32(vec![0.0; 4]), KernelArg::i32(vec![0])]
}

/// A read-only global (an LLVM constant) added to the input.
pub fn readonly_kernel() -> Kernel {
    Kernel::builder("readonly_global").block_dim([4, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major((4,)));
        let offset = module_global::<i32>(Some("offset"), 4, AddressSpace::Gmem, Some(&[10, 20, 30, 40]), true);
        let i = thread_idx().x;
        out.store(i, offset.load(i) + i);
    })
}

/// The tutorial's `kernel_static`: shared allocations of several kinds (a struct of 64 floats,
/// an int and a byte; 512 bytes aligned to 64; 128 ints; a 32x16 half tile), and the launch's
/// dynamic shared memory reported in `out`.
pub fn static_smem_kernel() -> Kernel {
    Kernel::builder("kernel_static").build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major((1,)));
        let values = make_smem_tensor_aligned::<f32>(Layout::col_major((64,)), 4);
        let counter = make_smem_tensor_aligned::<i32>(Layout::col_major((1,)), 4);
        let flag = make_smem_tensor_aligned::<i8>(Layout::col_major((1,)), 1);
        let raw = make_smem_tensor_aligned::<u8>(Layout::col_major((512,)), 64);
        let ints = make_smem_tensor_aligned::<i32>(Layout::col_major((128,)), 4);
        let tile = make_smem_tensor_aligned::<F16>(Layout::col_major((32, 16)), 16);
        values.store(0, Float32::from(1.0));
        counter.store(0, Int32::from(1));
        flag.store(0, Value::<i8>::from(1i8));
        raw.store(0, Value::<u8>::from(1u8));
        ints.store(0, Int32::from(1));
        tile.store((0, 0), Value::<F16>::from(F16::from_f32(1.0)));
        out.store(0, dynamic_smem_size());
    })
}

/// Where `region_kernel`'s dynamic region starts: after its 160 bytes of build-time buffers.
pub const REGION_START: u32 = 160;

/// A build-time buffer, then a dynamic region whose bytes the launch sets: each thread writes
/// five times its index into the region where it fits, and `out` reports the region's bytes
/// (`dynamic_smem_size`: the launch's dynamic shared memory less the buffer in front).
pub fn region_kernel() -> Kernel {
    Kernel::builder("region_kernel").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major((65,)));
        let fixed = make_smem_tensor_aligned::<f32>(Layout::col_major((40,)), 16);
        let region = get_dyn_smem::<i32>(Some(16));
        let i = thread_idx().x;
        if_(i.lt(40), || fixed.store(i, Float32::from(0.0)));
        let fits = (i * 4).lt(dynamic_smem_size());
        if_(fits, || region.offset(i).store(i * 5));
        sync_threads();
        if_(fits, || out.store(i + 1, region.offset(i).load()));
        if_(i.eq(0), || out.store(0, dynamic_smem_size()));
    })
}

/// Two alternatives overlaid at one offset: 64 floats, then 32 ints in the same bytes, with a
/// barrier between the phases (or without one, `racy`).
pub fn overlay_kernel(racy: bool) -> Kernel {
    Kernel::builder(if racy { "overlay_racy" } else { "overlay" }).block_dim([64, 1, 1]).build(|| {
        let out_f = param_tensor::<f32>("out_f", Layout::col_major((64,)));
        let out_i = param_tensor::<i32>("out_i", Layout::col_major((32,)));
        let (a, b) = smem_overlay(
            || make_smem_tensor::<f32>(Layout::col_major((64,))),
            || make_smem_tensor::<i32>(Layout::col_major((32,))),
        );
        let i = thread_idx().x;
        a.store(i, i.to::<f32>());
        sync_threads();
        out_f.store(i, a.load(63 - i));
        if !racy {
            sync_threads();
        }
        if_(i.lt(32), || b.store(i, i * 3));
        sync_threads();
        if_(i.lt(32), || out_i.store(i, b.load(31 - i)));
    })
}

/// Each block's cluster coordinates: `out[3b..3b+3]` = cluster index, rank in the cluster and
/// cluster size, along x.
pub fn cluster_kernel() -> Kernel {
    Kernel::builder("cluster_shapes").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major((24,)));
        let b = block_idx().x;
        if_(thread_idx().x.eq(0), || {
            out.store(b * 3, cluster_idx().x);
            out.store(b * 3 + 1, block_in_cluster_idx().x);
            out.store(b * 3 + 2, cluster_size());
        });
    })
}

/// A kernel with launch bounds: a declared block, or `max_threads`, and `min_blocks_per_sm`.
pub fn bounds_kernel(max_threads: Option<[u32; 3]>, min_blocks: Option<u32>) -> Kernel {
    let mut options = Kernel::builder("bounds");
    options = match max_threads {
        Some(m) => options.max_threads(m),
        None => options.block_dim([128, 1, 1]),
    };
    if let Some(n) = min_blocks {
        options = options.min_blocks_per_sm(n);
    }
    options.build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major((128,)));
        let i = thread_idx().x;
        out.store(i, i * 2);
    })
}
