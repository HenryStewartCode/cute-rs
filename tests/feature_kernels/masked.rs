// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/basic_types.py (the
//   `masked_load_store` kernel and its expected values),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Masked whole-tensor loads and stores, gather and scatter along a mode,
//! and the `cutlass.Pointer` methods on `Ptr`, held to plain Rust.

use cute::prelude::*;
use cute::tensor::{gather, scatter};

use super::Case;

/// Threads of these kernels: one warp.
const THREADS: usize = 32;

fn cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1])
}

/// `basic_types.py`'s `masked_load_store`, per thread: eight floats preset
/// to -5, four read with an all-true mask and stored, then the four plus 10
/// stored through the mask `[T, F, T, F]` four further on:
/// `[s0, s1, s2, s3, s0 + 10, -5, s2 + 10, -5]`.
pub fn masked_load_store() -> Case {
    let kernel = Kernel::builder("masked_load_store").block_dim([THREADS as u32, 1, 1]).build(|| {
        let src = param_tensor::<f32>("src", Layout::col_major(4 * THREADS));
        let dst = param_tensor::<f32>("dst", Layout::col_major(8 * THREADS));
        let t = thread_idx().x;
        let src_ptr = src.ptr() + t * 4;
        let dst_ptr = dst.ptr() + t * 8;
        dst_ptr.store_vec(&vector::full::<f32>(4, -5.0), 16);
        (&dst_ptr + 4).store_vec(&vector::full::<f32>(4, -5.0), 16);
        let load_mask = Vector::<bool>::from_elements([true, true, true, true]);
        let store_mask = Vector::<bool>::from_elements([true, false, true, false]);
        let values = src_ptr.masked_load(&load_mask, None, 16);
        dst_ptr.store_vec(&values, 16);
        let shifted = values + 10.0;
        (dst_ptr + 4).masked_store(&shifted, &store_mask, 16);
    });
    fn check(args: &[KernelArg]) {
        let (src, dst) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for t in 0..THREADS {
            let s = &src[t * 4..t * 4 + 4];
            let want = [s[0], s[1], s[2], s[3], s[0] + 10.0, -5.0, s[2] + 10.0, -5.0];
            assert_eq!(dst[t * 8..t * 8 + 8], want, "masked_load_store, thread {t}");
        }
    }
    Case { name: "masked_load_store", kernel, cfg: cfg(), small_only: false, check }
}

/// Masked `TensorSsa` loads with and without `pass_thru` from global
/// memory, shared memory and a fragment, a masked store, and an all-false
/// mask on a tile past the end of a buffer (which touches no memory).
/// Lane `i` of thread `t` is on when `i < n[t] & 7`.
pub fn masked_tensor_ssa() -> Case {
    let kernel = Kernel::builder("masked_tensor_ssa").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((8, THREADS)));
        let n = param_tensor::<i32>("n", Layout::col_major(THREADS));
        let out = param_tensor::<f32>("out", Layout::col_major((8, 6, THREADS)));
        let t = thread_idx().x;
        let k = n.load(t) & 7;
        let lanes = TensorSsa::new(8, (0..8).map(Int32::constant).collect());
        let mask = lanes.lt(k);
        let src = x.slice((.., t));
        let dst = |j: i32| out.slice((.., j, t));
        // No pass_thru: the masked-off lanes are undefined, selected away.
        let a = src.load_ssa_masked(&mask, None);
        dst(0).store_ssa(&where_(&mask, &a, -1.0));
        dst(1).store_ssa(&src.load_ssa_masked(&mask, Some(&full::<f32>(8, -2.0))));
        let smem = make_smem_tensor::<f32>(Layout::col_major((8, THREADS)));
        smem.slice((.., t)).store_ssa(&src.load_ssa());
        dst(2).store_ssa(&smem.slice((.., t)).load_ssa_masked(&!mask.clone(), Some(&full::<f32>(8, -3.0))));
        let frag = make_fragment_like::<f32, f32>(&src);
        frag.store_ssa(&(src.load_ssa() * 2.0));
        dst(3).store_ssa(&frag.load_ssa_masked(&mask, Some(&full::<f32>(8, -4.0))));
        dst(4).store_ssa(&full::<f32>(8, -5.0));
        dst(4).store_ssa_masked(&(src.load_ssa() * 3.0), &mask);
        // A tile just past the end of x, with a runtime all-false mask.
        let past = make_tensor(&(x.ptr() + 8 * THREADS), Layout::col_major(8));
        let none = lanes.gt(k + 100);
        let p = past.load_ssa_masked(&none, Some(&full::<f32>(8, 6.0)));
        past.store_ssa_masked(&p, &none);
        dst(5).store_ssa(&p);
    });
    fn check(args: &[KernelArg]) {
        let (x, n, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<i32>(), args[2].to_vec::<f32>());
        for t in 0..THREADS {
            let k = (n[t] & 7) as usize;
            let o = |j: usize, i: usize| out[t * 48 + j * 8 + i];
            for i in 0..8 {
                let v = x[t * 8 + i];
                let on = i < k;
                assert_eq!(o(0, i), if on { v } else { -1.0 }, "gmem, no pass_thru, lane {i}, thread {t}");
                assert_eq!(o(1, i), if on { v } else { -2.0 }, "gmem, pass_thru, lane {i}, thread {t}");
                assert_eq!(o(2, i), if !on { v } else { -3.0 }, "smem, lane {i}, thread {t}");
                assert_eq!(o(3, i), if on { v * 2.0 } else { -4.0 }, "fragment, lane {i}, thread {t}");
                assert_eq!(o(4, i), if on { v * 3.0 } else { -5.0 }, "masked store, lane {i}, thread {t}");
                assert_eq!(o(5, i), 6.0, "past the end, lane {i}, thread {t}");
            }
        }
    }
    Case { name: "masked_tensor_ssa", kernel, cfg: cfg(), small_only: false, check }
}

/// Outputs per thread of `gather_scatter`.
const GS: usize = 160;

/// Gathers along modes 0 and 1 of a 2-D and a 3-D tensor with Int32, Int64
/// and Uint8 index tensors, a gather from a register fragment, and
/// scatters (duplicates: the last in colexicographic order wins) to global
/// memory and to a fragment.
pub fn gather_scatter() -> Case {
    let kernel = Kernel::builder("gather_scatter").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x2 = param_tensor::<f32>("x2", Layout::col_major((8, 4, THREADS)));
        let x3 = param_tensor::<f32>("x3", Layout::col_major((4, 3, 2, THREADS)));
        let i32s = param_tensor::<i32>("i32s", Layout::col_major((3, 4, THREADS)));
        let i64s = param_tensor::<i64>("i64s", Layout::col_major((8, 2, THREADS)));
        let u8s = param_tensor::<u8>("u8s", Layout::col_major((2, 3, 2, THREADS)));
        let u8b = param_tensor::<u8>("u8b", Layout::col_major((4, 2, 2, THREADS)));
        let out = param_tensor::<f32>("out", Layout::col_major((GS, THREADS)));
        let t = thread_idx().x;
        let a = x2.slice((.., .., t));
        let b = x3.slice((.., .., .., t));
        let o = out.ptr() + t * GS as i32;
        let put = |at: usize, v: &TensorSsa<f32>| make_tensor(&(&o + at), Layout::col_major(v.size())).store_ssa(v);
        let i0 = i32s.slice((.., .., t)).load_ssa() & 7;
        let i1 = i64s.slice((.., .., t)).load_ssa() & 3i64;
        let i2 = u8s.slice((.., .., .., t)).load_ssa() & 3u8;
        let i3 = u8b.slice((.., .., .., t)).load_ssa() % 3u8;
        let g0 = gather(&a, 0, &i0);
        put(0, &g0);
        put(12, &gather(&a, 1, &i1));
        put(28, &gather(&b, 0, &i2));
        put(40, &gather(&b, 1, &i3));
        let frag = make_fragment_like::<f32, f32>(&a);
        frag.store_ssa(&a.load_ssa());
        put(56, &gather(&frag, 0, &i0));
        let region = make_tensor(&(&o + 68), Layout::col_major((8, 4)));
        region.store_ssa(&full::<f32>((8, 4), -1.0));
        scatter(&region, 0, &i0, &(g0.clone() + 100.0));
        let region3 = make_tensor(&(&o + 100), Layout::col_major((4, 3, 2)));
        region3.store_ssa(&full::<f32>((4, 3, 2), -1.0));
        let data3 = gather(&b, 1, &i3) * 2.0;
        scatter(&region3, 1, &i3, &data3);
        let rfrag = make_fragment::<f32>((8, 4));
        rfrag.store_ssa(&full::<f32>((8, 4), -1.0));
        scatter(&rfrag, 0, &i0, &g0);
        put(124, &rfrag.load_ssa());
    });
    fn check(args: &[KernelArg]) {
        let (x2, x3) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        let (i32s, i64s) = (args[2].to_vec::<i32>(), args[3].to_vec::<i64>());
        let (u8s, u8b, out) = (args[4].to_vec::<u8>(), args[5].to_vec::<u8>(), args[6].to_vec::<f32>());
        for t in 0..THREADS {
            let a = |i: usize, j: usize| x2[t * 32 + i + 8 * j];
            let b = |i: usize, j: usize, k: usize| x3[t * 24 + i + 4 * j + 12 * k];
            let o = &out[t * GS..(t + 1) * GS];
            let i0 = |i: usize, j: usize| (i32s[t * 12 + i + 3 * j] & 7) as usize;
            let i1 = |i: usize, j: usize| (i64s[t * 16 + i + 8 * j] & 3) as usize;
            let i2 = |i: usize, j: usize, k: usize| (u8s[t * 12 + i + 2 * j + 6 * k] & 3) as usize;
            let i3 = |i: usize, j: usize, k: usize| (u8b[t * 16 + i + 4 * j + 8 * k] % 3) as usize;
            let mut scattered = [-1.0f32; 32];
            let mut rscattered = [-1.0f32; 32];
            for j in 0..4 {
                for i in 0..3 {
                    let g = a(i0(i, j), j);
                    assert_eq!(o[i + 3 * j], g, "gather mode 0 ({i}, {j}), thread {t}");
                    assert_eq!(o[56 + i + 3 * j], g, "fragment gather ({i}, {j}), thread {t}");
                    scattered[i0(i, j) + 8 * j] = g + 100.0;
                    rscattered[i0(i, j) + 8 * j] = g;
                }
            }
            for j in 0..2 {
                for i in 0..8 {
                    assert_eq!(o[12 + i + 8 * j], a(i, i1(i, j)), "gather mode 1 ({i}, {j}), thread {t}");
                }
            }
            let mut scattered3 = [-1.0f32; 24];
            for k in 0..2 {
                for j in 0..3 {
                    for i in 0..2 {
                        let want = b(i2(i, j, k), j, k);
                        assert_eq!(o[28 + i + 2 * j + 6 * k], want, "3-D gather mode 0, thread {t}");
                    }
                }
                for j in 0..2 {
                    for i in 0..4 {
                        let g = b(i, i3(i, j, k), k);
                        assert_eq!(o[40 + i + 4 * j + 8 * k], g, "3-D gather mode 1, thread {t}");
                        scattered3[i + 4 * i3(i, j, k) + 12 * k] = g * 2.0;
                    }
                }
            }
            assert_eq!(o[68..100], scattered, "scatter mode 0, thread {t}");
            assert_eq!(o[100..124], scattered3, "3-D scatter mode 1, thread {t}");
            assert_eq!(o[124..156], rscattered, "fragment scatter, thread {t}");
        }
    }
    Case { name: "gather_scatter", kernel, cfg: cfg(), small_only: false, check }
}

/// Outputs per thread of `pointer_api`.
const PTR_OUT: usize = 28;

/// The `cutlass.Pointer` methods on `Ptr`: `ptr[i]`, 4 x Float32 vector
/// loads and stores with alignment 16, pointer arithmetic, a masked gather,
/// and ordered, volatile, non-coherent and cache-hinted accesses.
pub fn pointer_api() -> Case {
    let kernel = Kernel::builder("pointer_api").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(8 * THREADS));
        let offs = param_tensor::<i32>("offs", Layout::col_major(4 * THREADS));
        let out = param_tensor::<f32>("out", Layout::col_major(PTR_OUT * THREADS));
        let t = thread_idx().x;
        let p = x.ptr() + t * 8;
        let o = out.ptr() + t * PTR_OUT as i32;
        o.store_at(0, p.load_at(3));
        let v = p.load_vec(4, 16);
        (&o + 4).store_vec(&(v * 2.0), 16);
        let mut q = &p + 6usize;
        q -= 2usize;
        o.store_at(8, (&q - 1usize).load());
        q += 1usize;
        o.store_at(9, q.load());
        let lanes = Vector::<i32>::from_elements((0..4).map(|i| offs.load(t * 4 + i)));
        let mask = (lanes.clone() & 8).ne(0);
        let gathered = p.masked_gather(&(lanes & 7), &mask, Some(&vector::full::<f32>(4, -1.0)), 4);
        (&o + 10).masked_store(&gathered, &vector::full::<bool>(4, true), 4);
        let gpu = Some(MemoryScope::Gpu);
        let acquire = LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..Default::default() };
        let streaming = LoadOptions { cop: Some(CacheOperator::Cs), ..Default::default() };
        let invariant = LoadOptions { sem: MemoryOrder::Constant, ..Default::default() };
        let hinted = LoadOptions {
            level1_eviction_priority: Some(CacheEvictionPriority::EvictLast),
            level_prefetch_size: Some(L2PrefetchSize::Size128B),
            ..Default::default()
        };
        let volatile = LoadOptions { sem: MemoryOrder::Volatile, ..Default::default() };
        o.store_at(14, p.load_with(acquire));
        o.store_at(15, (&p + 1).load_with(streaming));
        o.store_at(16, (&p + 2).load_with(invariant));
        o.store_at(17, p.load_vec_with(4, 16, hinted).get(1));
        let release = StoreOptions { sem: MemoryOrder::Release, scope: gpu, ..Default::default() };
        let write_through = StoreOptions { cop: Some(CacheOperator::Wt), ..Default::default() };
        (&o + 18).store_with((&p + 4).load(), release);
        (&o + 19).store_with((&p + 5).load(), write_through);
        let evict =
            StoreOptions { level1_eviction_priority: Some(CacheEvictionPriority::EvictLast), ..Default::default() };
        (&o + 20).store_vec_with(&(&p + 4).load_vec(4, 16), 16, evict);
        o.store_at(24, (&p + 7).load_with(volatile));
    });
    fn check(args: &[KernelArg]) {
        let (x, offs, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<i32>(), args[2].to_vec::<f32>());
        for t in 0..THREADS {
            let p = &x[t * 8..t * 8 + 8];
            let o = &out[t * PTR_OUT..(t + 1) * PTR_OUT];
            assert_eq!(o[0], p[3], "ptr[3], thread {t}");
            for k in 0..4 {
                assert_eq!(o[4 + k], p[k] * 2.0, "load_vec/store_vec {k}, thread {t}");
                let off = offs[t * 4 + k];
                let want = if off & 8 != 0 { p[(off & 7) as usize] } else { -1.0 };
                assert_eq!(o[10 + k], want, "masked_gather {k}, thread {t}");
                assert_eq!(o[20 + k], p[4 + k], "hinted store_vec {k}, thread {t}");
            }
            assert_eq!(o[8..10], [p[3], p[5]], "pointer arithmetic, thread {t}");
            assert_eq!(o[14..20], [p[0], p[1], p[2], p[1], p[4], p[5]], "ordered and hinted, thread {t}");
            assert_eq!(o[24], p[7], "volatile, thread {t}");
        }
    }
    Case { name: "pointer_api", kernel, cfg: cfg(), small_only: false, check }
}

/// `Ptr` vector accesses with cluster scope (relaxed loads and stores),
/// which need sm_90.
pub fn pointer_cluster_scope() -> Case {
    let kernel = Kernel::builder("pointer_cluster_scope").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(4 * THREADS));
        let out = param_tensor::<u32>("out", Layout::col_major(4 * THREADS));
        let t = thread_idx().x;
        let cluster = Some(MemoryScope::Cluster);
        let relaxed = LoadOptions { sem: MemoryOrder::Relaxed, scope: cluster, ..Default::default() };
        let v = (x.ptr() + t * 4).load_vec_with(4, 16, relaxed);
        let release = StoreOptions { sem: MemoryOrder::Relaxed, scope: cluster, ..Default::default() };
        (out.ptr() + t * 4).store_vec_with(&(v + 1u32), 16, release);
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for (i, (&a, &b)) in x.iter().zip(&out).enumerate() {
            assert_eq!(b, a.wrapping_add(1), "element {i}");
        }
    }
    Case { name: "pointer_cluster_scope", kernel, cfg: cfg(), small_only: false, check }
}

/// The masked, gather and pointer kernels of every target.
pub fn masked_cases() -> Vec<Case> {
    vec![masked_load_store(), masked_tensor_ssa(), gather_scatter(), pointer_api()]
}
