// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/notebooks/async_pipeline.ipynb (the shape of
//   its SharedStorage), a notebook with no header of its own: the README.md of
//   its folder carries
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   and CUTLASS's LICENSE.txt
//     Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Kernels with shared-storage structs (`cute_struct!`, `cute_union!`):
//! CuTe DSL's `async_pipeline` notebook's `SharedStorage` carrying an
//! mbarrier hand-off, scalar members stored with conversion, and a union
//! written through one member and read through others.

use cute::prelude::*;

use super::{Case, T, bounded_spin, check_no_spin_gave_up, launch};

const STAGES: usize = 3;
const ITERS: usize = 8;

cute_struct! {
    /// The `async_pipeline` notebook's shared storage: a full and an empty
    /// mbarrier per stage, and a 1024-byte-aligned staging buffer (here 32
    /// floats per stage, one per lane).
    pub struct PipelineStorage(stages: usize) {
        tma_mbar_ptr: MemRange<u64>[2 * stages],
        #[align(1024)]
        staging_buffer: MemRange<f32>[stages * 32],
    }
}

/// A producer warp hands `ITERS` rows of 32 floats to a consumer warp
/// through the struct's `STAGES`-deep ring of staging slots, with a full
/// and an empty mbarrier per slot.
pub fn struct_mbarrier_pipeline() -> Case {
    let kernel = Kernel::builder("struct_mbarrier_pipeline").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<f32>("src", Layout::col_major(ITERS * 32));
        let out = param_tensor::<f32>("out", Layout::col_major(ITERS * 32));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let s = PipelineStorage::allocate(STAGES);
        let bars = s.tma_mbar_ptr.data_ptr();
        let full = |stage: Int32| bars.offset(stage);
        let empty = |stage: Int32| bars.offset(stage + STAGES as i32);
        if_(t.eq(0), || {
            for i in 0..2 * STAGES {
                mbarrier_init(&bars.offset(i), 32);
            }
        });
        sync_threads();
        let staging = s.staging_buffer.get_tensor(Layout::col_major((32, STAGES)));
        let misses = Var::new(Uint32::from(0u32));
        let spin = |bar: Ptr<u64>, parity: Int32| {
            let missed = bounded_spin(&bar, parity);
            misses.update(|m| m + select(missed, 1u32, 0u32));
        };
        for_range(0, ITERS as i32, 1, |i| {
            let stage = i % STAGES as i32;
            let parity = (i / STAGES as i32) % 2;
            if_else(
                warp_idx().eq(0),
                || {
                    spin(full(stage), parity);
                    out.store(i * 32 + lane, staging.load((lane, stage)) * 2.0);
                    mbarrier_arrive(&empty(stage));
                },
                || {
                    // A fresh mbarrier counts the phase before its first as complete.
                    spin(empty(stage), 1 - parity);
                    staging.store((lane, stage), src.load(i * 32 + lane) + 1.0);
                    mbarrier_arrive(&full(stage));
                },
            );
        });
        gave_up.store(t, misses.get());
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for (i, (&x, &y)) in src.iter().zip(&out).enumerate() {
            assert_eq!(y.to_bits(), ((x + 1.0) * 2.0).to_bits(), "element {i}");
        }
    }
    Case { name: "struct_mbarrier_pipeline", kernel, cfg: launch(1), small_only: false, check }
}

cute_struct! {
    pub struct Scalars {
        flag: bool,
        byte: i8,
        half: F16,
        alpha: f32,
        count: i32,
        wide: u64,
        unused: MemRange<u16>[3],
        #[align(16)]
        tail: MemRange<u32>[4],
    }
}

/// Scalar members stored with conversion (CuTe DSL's `s.x = value`) and
/// read back, and member addresses (`ptrtoint`, of a member never accessed
/// as its own type too) against the layout's offsets.
pub fn struct_scalar_members() -> Case {
    let kernel = Kernel::builder("struct_scalar_members").block_dim([T as u32, 1, 1]).build(|| {
        let ints = param_tensor::<i32>("ints", Layout::col_major(4));
        let floats = param_tensor::<f32>("floats", Layout::col_major(2));
        let out = param_tensor::<u32>("out", Layout::col_major(10));
        let s = Scalars::allocate();
        if_(thread_idx().x.eq(0), || {
            s.alpha.store_from(ints.load(0));
            s.count.store_from(floats.load(0) * 1000.0);
            s.half.store_from(floats.load(1) * 1000.0);
            s.wide.store_from(ints.load(1));
            s.flag.store_from(ints.load(2));
            s.byte.store_from(ints.load(3));
            s.tail.store(3, 7u32);
        });
        sync_threads();
        if_(thread_idx().x.eq(T as i32 - 1), || {
            let wide = s.wide.load();
            let base = s.base().toint_i64();
            let at = |p: Int64| (p - base).to::<u32>();
            out.store(0, s.alpha.load().bitcast::<u32>());
            out.store(1, s.count.load().bitcast::<u32>());
            out.store(2, s.half.load().bitcast::<u16>().to::<u32>());
            out.store(3, wide.to::<u32>());
            out.store(4, (wide >> 32u64).to::<u32>());
            out.store(5, s.flag.load().to::<u32>());
            out.store(6, s.byte.load().to::<i32>().bitcast::<u32>());
            out.store(7, at(s.wide.ptr().toint_i64()));
            out.store(8, at(s.unused.data_ptr().toint_i64()));
            out.store(9, s.tail.load(3) + at(s.tail.data_ptr().toint_i64()));
        });
    });
    fn check(args: &[KernelArg]) {
        let (ints, floats, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<f32>(), args[2].to_vec::<u32>());
        let wide = ints[1] as i64 as u64;
        let want = [
            (ints[0] as f32).to_bits(),
            (floats[0] * 1000.0) as i32 as u32,
            F16::from_f32(floats[1] * 1000.0).0 as u32,
            wide as u32,
            (wide >> 32) as u32,
            (ints[2] != 0) as u32,
            ints[3] as i8 as i32 as u32,
            16,
            24,
            7 + 32,
        ];
        assert_eq!(out, want);
    }
    Case { name: "struct_scalar_members", kernel, cfg: launch(1), small_only: false, check }
}

cute_union! {
    pub struct Alias(n: usize) {
        words: MemRange<u32>[n],
        halves: MemRange<u16>[2 * n],
        bytes: MemRange<u8>[4 * n],
    }
}

cute_struct! {
    pub struct WithUnion(n: usize) {
        head: u32,
        #[align(16)]
        u: Alias(n),
    }
}

/// A union nested in a struct, written through its words and read back
/// through its halves and bytes (as `reinterpreted_memory`, through members
/// at the same offset).
pub fn struct_union_alias() -> Case {
    let kernel = Kernel::builder("struct_union_alias").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, 4)));
        let t = thread_idx().x;
        let s = WithUnion::allocate(T);
        s.u.words.get_tensor(Layout::col_major(T)).store(t, xs.load(t));
        if_(t.eq(0), || s.head.store(5u32));
        sync_threads();
        let other = (t + 32) % 64;
        out.store((t, 0), s.u.halves.load(other * 2).to::<u32>());
        out.store((t, 1), s.u.halves.load(other * 2 + 1).to::<u32>());
        out.store((t, 2), s.u.bytes.load(other * 4 + 3).to::<u32>());
        out.store((t, 3), s.head.load() + s.u.bytes.byte_offset().unwrap() as u32);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            let x = xs[(t + 32) % 64];
            assert_eq!(out[t * 4..t * 4 + 4], [x & 0xffff, x >> 16, x >> 24, 5 + 16], "thread {t}");
        }
    }
    Case { name: "struct_union_alias", kernel, cfg: launch(1), small_only: false, check }
}

pub fn struct_cases() -> Vec<Case> {
    vec![struct_mbarrier_pipeline(), struct_scalar_members(), struct_union_alias()]
}
