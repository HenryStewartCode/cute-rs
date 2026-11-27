// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// This kernel follows the design of a CuTe DSL example of NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// and takes its tile size, its MMA, copy and shared-memory layouts and its
// swizzle from it. Those parts are under the BSD-3-Clause licence, and the rest
// of the file is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md
// has the licence text and describes what is taken.

//! A tensor-core GEMM for SM80 (Ampere), following the design of CuTe DSL's
//! `ampere/tensorop_gemm` example.
//!
//! `C = A * B^T` with f16 (or bf16) inputs and f32 accumulation, A `(M, K)`
//! and B `(N, K)` both K-major (CuTe's "TN" GEMM), C `(M, N)` row-major.
//!
//! The example's FP8 path runs on SM89 (Ada): Float8E4M3FN or Float8E5M2
//! inputs (`GemmConfig::fp8`), `MmaFP8Op` of shape 16x8x32 with Float32
//! accumulators, a 128x128x64 tile, `LdMatrix8x16x8bOp(False, 4)` for both
//! K-major operands, and C in Float32 or FP8 (`GemmConfig { c, .. }`).
//!
//! * 128 threads (4 warps, 2 x 2) compute a `BM x BN` tile of C with
//!   `mma.sync.m16n8k16`.
//! * A `stages`-deep `cp.async` pipeline streams `BM x BK` and `BN x BK`
//!   tiles of A and B into shared memory while earlier tiles are consumed.
//! * Shared memory is swizzled (`Swizzle<2,3,3>` over 8 x 32 atoms) so that
//!   `ldmatrix` reads are free of bank conflicts; the simulator's report
//!   shows the difference against an unswizzled layout.
//! * Fragments move from shared memory to registers with `ldmatrix`, using
//!   tiled copies derived from the MMA's own operand layouts.
//! * The K loop runs on the GPU and carries the 128 accumulators per thread.
//!
//! Run with `cargo run --release --example tensorop_gemm`.

use cute::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct GemmConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub stages: usize,
    pub swizzle: bool,
    /// Put the swizzle on the shared-memory pointers (`recast_ptr`, on
    /// bytes: `Swizzle<2,4,3>` for f16, CuTe DSL's values) rather than in a
    /// composed layout (on elements: `Swizzle<2,3,3>`). The two address the
    /// same bytes; the pointer form needs its buffers aligned to the
    /// swizzle's period (512 bytes), so it allocates them with
    /// `alloc_smem`.
    pub ptr_swizzle: bool,
    /// Keep the A and B stages in one shared-memory struct, [`GemmStorage`]
    /// (CuTe DSL's `@cute.struct class SharedStorage`), each 1024-byte
    /// aligned, rather than in two buffers of their own: the tensors come
    /// from the members' `get_tensor_composed`, and `cp.async` and
    /// `ldmatrix` go through the struct's byte buffer.
    pub smem_struct: bool,
    /// The type of A and B: `DType::F16` or `DType::BF16`, or for the FP8
    /// path (SM89) `DType::F8E4M3FN` or `DType::F8E5M2`.
    pub ab: DType,
    /// Rotate the mainloop with [`for_range_pipelined`] (prefetch distance
    /// `stages - 1`) rather than by hand. Same results and the same
    /// instructions per K tile.
    pub pipelined: bool,
    /// The type of C: `DType::F32`, or with FP8 inputs also an FP8 type
    /// (the accumulators converted with `.satfinite`, as `.to()` does).
    pub c: DType,
    /// Trace the hand-rotated mainloop with IKET events, as CuTe DSL's `fp16_gemm_4_iket.py`
    /// does its GEMM: `range_push`/`range_pop` around the prologue, each K tile's wait and its
    /// issue, and the epilogue (`IketOptions`; off by default, and then the calls emit nothing).
    pub iket: Option<IketOptions>,
}

impl GemmConfig {
    pub fn new(m: usize, n: usize, k: usize) -> GemmConfig {
        GemmConfig {
            m,
            n,
            k,
            bm: 128,
            bn: 128,
            bk: 32,
            stages: 3,
            swizzle: true,
            ptr_swizzle: false,
            smem_struct: false,
            ab: DType::F16,
            c: DType::F32,
            pipelined: false,
            iket: None,
        }
    }

    /// The example's FP8 configuration: `ab` (`DType::F8E4M3FN` or
    /// `DType::F8E5M2`) inputs, its 128x128x64 tile, Float32 C.
    pub fn fp8(m: usize, n: usize, k: usize, ab: DType) -> GemmConfig {
        GemmConfig { bk: 64, ab, ..GemmConfig::new(m, n, k) }
    }

    fn is_fp8(&self) -> bool {
        matches!(self.ab, DType::F8E4M3FN | DType::F8E5M2)
    }

    /// The kernel's name: `tensorop_gemm`, then the input type unless f16,
    /// the C type unless f32, and `_unswizzled`.
    fn name(&self) -> String {
        let ty = |t: DType| match t {
            DType::BF16 => "_bf16",
            DType::F8E4M3FN => "_e4m3",
            DType::F8E5M2 => "_e5m2",
            _ => "",
        };
        let c = if self.c == DType::F32 { String::new() } else { format!("_c{}", ty(self.c)) };
        let swizzle = match (self.swizzle, self.ptr_swizzle) {
            (false, _) => "_unswizzled",
            (true, true) => "_ptr_swizzle",
            (true, false) => "",
        };
        let storage = if self.smem_struct { "_struct" } else { "" };
        let pipelined = if self.pipelined { "_pipelined" } else { "" };
        format!("tensorop_gemm{}{c}{swizzle}{storage}{pipelined}", ty(self.ab))
    }
}

pub fn build(cfg: GemmConfig) -> Kernel {
    assert!(
        cfg.c == DType::F32 || (cfg.is_fp8() && matches!(cfg.c, DType::F8E4M3FN | DType::F8E5M2)),
        "tensorop_gemm writes f32 C, or FP8 C from FP8 inputs, not {} from {}",
        cfg.c,
        cfg.ab
    );
    let f32_c = |acc: &Tensor<f32>, c: &Tensor<f32>| autovec_copy(acc, c);
    match (cfg.ab, cfg.c) {
        (DType::F16, _) => build_typed::<F16, f32>(cfg, f32_c),
        (DType::BF16, _) => build_typed::<BF16, f32>(cfg, f32_c),
        (DType::F8E4M3FN, DType::F32) => build_typed::<F8E4M3FN, f32>(cfg, f32_c),
        (DType::F8E4M3FN, DType::F8E4M3FN) => build_typed::<F8E4M3FN, F8E4M3FN>(cfg, narrow_c),
        (DType::F8E4M3FN, _) => build_typed::<F8E4M3FN, F8E5M2>(cfg, narrow_c),
        (DType::F8E5M2, DType::F32) => build_typed::<F8E5M2, f32>(cfg, f32_c),
        (DType::F8E5M2, DType::F8E4M3FN) => build_typed::<F8E5M2, F8E4M3FN>(cfg, narrow_c),
        (DType::F8E5M2, _) => build_typed::<F8E5M2, F8E5M2>(cfg, narrow_c),
        (other, _) => panic!("tensorop_gemm takes f16, bf16 or FP8 inputs, not {other}"),
    }
}

/// The FP8 C epilogue: the accumulators converted (`epilogue_op(..).to(c_dtype)`,
/// rounded to nearest even and saturated) into a fragment, then stored.
fn narrow_c<C: Numeric>(acc: &Tensor<f32>, c: &Tensor<C>) {
    let out = make_fragment_like::<C, _>(acc);
    for i in 0..acc.size() {
        out.store(i, acc.load(i).to::<C>());
    }
    autovec_copy(&out, c);
}

/// The K-major shared-memory atom and swizzle of `tensorop_gemm.py`'s
/// `_make_smem_layout_AB` for elements of `width` bits, BK `bk` and 128-bit
/// copies: rows of `min(bk, 128 bytes)` elements, eight of them, and
/// `Swizzle<bits, base, shift>` with `bits = min(log2(row bits / 128), 3)`
/// and `base = shift = log2(128 / width)`, in elements (the example gives
/// `base` in bytes, `log2(16)`, which is `log2(128 / width)` elements). For
/// f16 and BK 32 that is `(8, 32)` with `Swizzle<2,3,3>`; for FP8 and BK 64
/// `(8, 64)` with `Swizzle<2,4,4>`.
fn smem_atom_ab(width: usize, bk: usize) -> (Layout, (u32, u32, u32)) {
    let row = bk.min(128 * 8 / width);
    let log2 = |x: usize| x.trailing_zeros();
    let bits = log2(row * width / 128).min(3);
    let base = log2(128 / width);
    (Layout::row_major((8, row)), (bits, base, base))
}

cute_struct! {
    /// The kernel's shared memory with [`GemmConfig::smem_struct`]: the
    /// `stages` tiles of A and of B, `a_elems` and `b_elems` elements (the
    /// swizzled layouts' extents), each aligned to 1024 bytes as CuTe DSL's
    /// examples align their operand buffers (`Align[MemRange[..], 1024]`).
    pub struct GemmStorage<AB>(a_elems: usize, b_elems: usize) {
        #[align(1024)]
        s_a: MemRange<AB>[a_elems],
        #[align(1024)]
        s_b: MemRange<AB>[b_elems],
    }
}

/// The kernel for inputs of type `AB` (`F16`, `BF16` or an FP8 type) and C
/// of type `C`.
fn build_typed<AB: Numeric, C: Numeric>(cfg: GemmConfig, epilogue: impl Fn(&Tensor<f32>, &Tensor<C>)) -> Kernel {
    let GemmConfig { m, n, k, bm, bn, bk, stages, swizzle, .. } = cfg;
    assert!(
        m.is_multiple_of(bm) && n.is_multiple_of(bn) && k.is_multiple_of(bk),
        "problem size must be a multiple of the tile size"
    );
    let fp8 = cfg.is_fp8();
    // The instruction's K: 16 for f16 and bf16, 32 for FP8.
    let mma_k = if fp8 { 32 } else { 16 };
    assert!(bk.is_multiple_of(mma_k), "BK ({bk}) is not a whole number of MMA k-blocks of {mma_k}");
    let width = AB::DTYPE.bits();
    let (atom, (sw_bits, sw_base, sw_shift)) = smem_atom_ab(width, bk);
    assert!(bk.is_multiple_of(atom.mode(1).size()), "BK ({bk}) is not a whole number of shared-memory atom rows");
    let k_tiles = k / bk;
    let name = cfg.name();

    let mut builder = Kernel::builder(&name).block_dim([128, 1, 1]);
    if let Some(options) = cfg.iket {
        builder = builder.iket(options);
    }
    builder.build(|| {
        let a = param_tensor::<AB>("a", Layout::row_major((m, k)));
        let b = param_tensor::<AB>("b", Layout::row_major((n, k)));
        let c = param_tensor::<C>("c", Layout::row_major((m, n)));
        let tid = thread_idx().x;
        let bid = block_idx();

        let cta = (bid.x, bid.y, ..);
        let g_a = local_tile_proj(&a, (bm, bn, bk), cta, &[true, false, true]); // (BM, BK, k)
        let g_b = local_tile_proj(&b, (bm, bn, bk), cta, &[false, true, true]); // (BN, BK, k)
        let g_c = local_tile_proj(&c, (bm, bn, bk), cta, &[true, true, false]); // (BM, BN)

        // Shared memory: `stages` K-major tiles of A and B (`sA_layout`,
        // `sB_layout`, from `_make_smem_layout_AB`). For f16 each 8 x 32
        // atom is a row-major block of eight 64-byte rows; the swizzle XORs
        // the row (bits 6..8 of the offset) into the 16-byte chunk index
        // (bits 3..5) so the eight rows `ldmatrix` reads hit eight different
        // bank groups. FP8's 8 x 64 atoms (64-byte rows again) XOR bits 8..10
        // into bits 4..6, as the example chooses.
        let s_a_layout = tile_to_shape(&atom, (bm, bk, stages));
        let s_b_layout = tile_to_shape(&atom, (bn, bk, stages));
        let (s_a, s_b) = if cfg.smem_struct {
            // One struct holding both: the composed layouts as above (the
            // identity swizzle unswizzled), each member sized to its
            // layout's extent in whole swizzle periods.
            let sw = if swizzle { make_swizzle(sw_bits, sw_base, sw_shift as i32) } else { Swizzle::identity() };
            let (c_a, c_b) = (make_composed_layout(sw, 0, s_a_layout), make_composed_layout(sw, 0, s_b_layout));
            let extent = |c: &ComposedLayout| c.cosize().next_multiple_of(if swizzle { sw.period() } else { 1 });
            let storage = GemmStorage::<AB>::allocate(extent(&c_a), extent(&c_b));
            (storage.s_a.get_tensor_composed(c_a), storage.s_b.get_tensor_composed(c_b))
        } else if swizzle && cfg.ptr_swizzle {
            // The same swizzle on bytes, carried by the pointers (CuTe DSL's
            // `make_tensor(recast_ptr(ptr, swizzle_), outer)`).
            let sw = make_swizzle(sw_bits, sw_base, sw_shift as i32).to_bytes(width);
            let tile = |layout: Layout| {
                let len = layout.cosize().next_multiple_of(sw.period() * 8 / width);
                let ptr = recast_ptr::<AB, AB>(&alloc_smem::<AB>(len, sw.period()), Some(sw));
                make_tensor(&ptr, layout)
            };
            (tile(s_a_layout), tile(s_b_layout))
        } else if swizzle {
            let sw = make_swizzle(sw_bits, sw_base, sw_shift as i32);
            (
                make_smem_tensor::<AB>(make_composed_layout(sw, 0, s_a_layout)),
                make_smem_tensor::<AB>(make_composed_layout(sw, 0, s_b_layout)),
            )
        } else {
            (make_smem_tensor::<AB>(s_a_layout), make_smem_tensor::<AB>(s_b_layout))
        };

        // Global -> shared with 16-byte cp.async: 32 x 4 threads, each moving
        // 8 contiguous K values (16 for FP8).
        let g2s_atom = make_copy_atom::<AB>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let g2s = make_tiled_copy_tv(&g2s_atom, &Layout::row_major((32, 4)), &Layout::row_major((1, 128 / width)));
        let thr_g2s = g2s.get_slice(tid);
        let t_a_g = thr_g2s.partition_s(&g_a); // (CPY, CPY_M, CPY_K, k)
        let t_a_s = thr_g2s.partition_d(&s_a); // (CPY, CPY_M, CPY_K, stage)
        let t_b_g = thr_g2s.partition_s(&g_b);
        let t_b_s = thr_g2s.partition_d(&s_b);

        // 2 x 2 warps of m16n8k16 tensor-core MMAs with f32 accumulators;
        // for FP8 m16n8k32 with the example's `permutation_mnk` (32, 32, 32),
        // which doubles each warp's N so that B, too, is loaded four
        // matrices at a time.
        let mma = if fp8 {
            let perm = |n: usize| Some(Layout::new(n, 1));
            make_tiled_mma(MmaFP8Op::new(AB::DTYPE, DType::F32, (16, 8, 32)), Layout::col_major((2, 2, 1)))
                .with_permutation_mnk([perm(32), perm(32), perm(32)])
        } else {
            make_tiled_mma(MmaF16BF16Op::new(AB::DTYPE, DType::F32, (16, 8, 16)), Layout::col_major((2, 2, 1)))
        };
        let thr_mma = mma.get_slice(tid);
        let r_a = thr_mma.partition_fragment_a::<AB, _>(&s_a.slice((.., .., 0))); // (MMA, MMA_M, MMA_K)
        let r_b = thr_mma.partition_fragment_b::<AB, _>(&s_b.slice((.., .., 0))); // (MMA, MMA_N, MMA_K)
        let t_c_g = thr_mma.partition_c(&g_c); // (MMA, MMA_M, MMA_N)
        let acc = thr_mma.partition_fragment_c::<f32, _>(&g_c);
        acc.fill(0.0);

        // Shared -> registers with ldmatrix, laid out exactly like the MMA's
        // A and B fragments: `LdMatrix8x16x8bOp(False, 4)` for K-major FP8
        // (the example's `CopyUniversalOp` of 8 bits serves M- or N-major
        // FP8, which this TN GEMM does not take).
        let (s2r_a, s2r_b) = if fp8 {
            let ldsm = make_copy_atom::<AB>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None });
            (make_tiled_copy_a(&ldsm, &mma), make_tiled_copy_b(&ldsm, &mma))
        } else {
            (
                make_tiled_copy_a(&make_copy_atom::<AB>(CopyOp::LdMatrix { num: 4, trans: false }), &mma),
                make_tiled_copy_b(&make_copy_atom::<AB>(CopyOp::LdMatrix { num: 2, trans: false }), &mma),
            )
        };
        let thr_s2r_a = s2r_a.get_slice(tid);
        let thr_s2r_b = s2r_b.get_slice(tid);
        let t_a_s_view = thr_s2r_a.partition_s(&s_a); // (CPY, CPY_M, CPY_K, stage)
        let t_b_s_view = thr_s2r_b.partition_s(&s_b);
        let r_a_view = thr_s2r_a.retile_d(&r_a);
        let r_b_view = thr_s2r_b.retile_d(&r_b);
        let k_blocks = r_a.shape().mode(2).size();

        // The MMAs of K tile `k_tile`, from its stage.
        let mmas = |acc: &Tensor<f32>, stage: Int32| {
            for kb in 0..k_blocks {
                copy(&s2r_a, &t_a_s_view.slice((.., .., kb, stage)), &r_a_view.slice((.., .., kb)));
                copy(&s2r_b, &t_b_s_view.slice((.., .., kb, stage)), &r_b_view.slice((.., .., kb)));
                gemm(&mma, acc, &r_a.slice((.., .., kb)), &r_b.slice((.., .., kb)), acc);
            }
        };
        let acc = if cfg.pipelined {
            // The same rotation by `for_range_pipelined` (CuTe DSL's
            // `cutlass.range(k_tiles, prefetch_stages=stages - 1)`): the
            // producer loads tile `k_tile` into its stage, after waiting for
            // the tile the consumer reads next and the barrier (which also
            // retires the reads of the stage it refills); the prologue's
            // loads need neither. The drain, which loads nothing, waits for
            // its tile itself, `k_tiles - 1 - k_tile` groups newer.
            let ahead = stages - 1;
            let (_, acc) = for_range_pipelined(
                0,
                k_tiles as i32,
                1,
                Prefetch::stages(ahead as i64).bounded_by(stages),
                ((), acc),
                |k_tile, ()| {
                    if k_tile.as_constant().is_none_or(|k| k as usize >= ahead) {
                        cp_async_wait_group((stages - 2) as u32);
                        sync_threads();
                    }
                    let stage = k_tile % stages as i32;
                    copy(&g2s, &t_a_g.slice((.., .., .., k_tile)), &t_a_s.slice((.., .., .., stage)));
                    copy(&g2s, &t_b_g.slice((.., .., .., k_tile)), &t_b_s.slice((.., .., .., stage)));
                    cp_async_commit_group();
                },
                |k_tile, acc| {
                    if let Some(k) = k_tile.as_constant() {
                        cp_async_wait_group((k_tiles - 1 - k as usize) as u32);
                        sync_threads();
                    }
                    mmas(&acc, k_tile % stages as i32);
                    acc
                },
            );
            acc
        } else {
            // Prologue: start loading the first `stages - 1` K tiles.
            range_push("prologue", None);
            for s in 0..stages - 1 {
                if s < k_tiles {
                    copy(&g2s, &t_a_g.slice((.., .., .., s)), &t_a_s.slice((.., .., .., s)));
                    copy(&g2s, &t_b_g.slice((.., .., .., s)), &t_b_s.slice((.., .., .., s)));
                }
                cp_async_commit_group();
            }
            range_pop();

            for_range_carry(0, k_tiles, 1, acc, |k_tile, acc| {
                // Tile `k_tile` has landed once at most `stages - 2` newer
                // groups are still in flight; the barrier publishes every
                // thread's copies and retires the previous iteration's reads of
                // the stage about to be refilled.
                range_push("wait", Some(k_tile.into()));
                cp_async_wait_group((stages - 2) as u32);
                sync_threads();
                range_pop();

                range_push("issue", Some(k_tile.into()));
                let next = k_tile + (stages - 1) as i32;
                if_(next.lt(k_tiles), || {
                    let stage = next % stages as i32;
                    copy(&g2s, &t_a_g.slice((.., .., .., next)), &t_a_s.slice((.., .., .., stage)));
                    copy(&g2s, &t_b_g.slice((.., .., .., next)), &t_b_s.slice((.., .., .., stage)));
                });
                cp_async_commit_group();
                range_pop();

                let stage = k_tile % stages as i32;
                for kb in 0..k_blocks {
                    copy(&s2r_a, &t_a_s_view.slice((.., .., kb, stage)), &r_a_view.slice((.., .., kb)));
                    copy(&s2r_b, &t_b_s_view.slice((.., .., kb, stage)), &r_b_view.slice((.., .., kb)));
                    gemm(&mma, &acc, &r_a.slice((.., .., kb)), &r_b.slice((.., .., kb)), &acc);
                }
                acc
            })
        };

        // Epilogue: registers -> global, two contiguous f32 per store (FP8
        // C converts the accumulators first).
        range_push("epilogue", None);
        epilogue(&acc, &t_c_g);
        range_pop();
    })
}

/// Simulate `cfg` on small-integer inputs and check against a CPU GEMM.
pub fn run(cfg: GemmConfig) -> SimReport {
    let kernel = build(cfg);
    let GemmConfig { m, n, k, .. } = cfg;
    let (a, b) = (small_integers(m * k, 1), small_integers(n * k, 2));
    let mut args = vec![input(cfg.ab, &a), input(cfg.ab, &b), input(cfg.c, &vec![0.0; m * n])];
    let launch = LaunchConfig::new([(m / cfg.bm) as u32, (n / cfg.bn) as u32, 1], [128, 1, 1]);
    let report =
        kernel.simulate_all_orders(launch, &mut args, &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    // Exact sums, and for FP8 C their FP8 rounding.
    let c = args[2].to_f32();
    let rounded = |x: f32| match cfg.c {
        DType::F32 => x,
        _ => input(cfg.c, &[x]).to_f32()[0],
    };
    let mut zeros = 0;
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum();
            assert_eq!(c[i * n + j], rounded(expected), "C[{i},{j}]");
            zeros += usize::from(expected == 0.0);
        }
    }
    // C starts out 0: inputs that leave most of it 0 would let a kernel that
    // writes little of it pass.
    assert!(zeros * 4 < m * n, "{zeros} of the {} elements of C are 0: the inputs test little", m * n);
    report
}

/// A buffer of `values` as `ty` (f16, bf16, an FP8 type or f32): the FP8
/// values built from f32 with `KernelArg::fp8_from_f32`, the analogue of
/// the example's `create_cute_tensor_for_fp8`, rounded to nearest even and
/// saturated.
pub fn input(ty: DType, values: &[f32]) -> KernelArg {
    match ty {
        DType::F16 => KernelArg::f16_from_f32(values),
        DType::BF16 => KernelArg::buffer(&values.iter().map(|&x| BF16::from_f32(x)).collect::<Vec<_>>()),
        DType::F8E4M3FN => KernelArg::fp8_from_f32::<F8E4M3FN>(values),
        DType::F8E5M2 => KernelArg::fp8_from_f32::<F8E5M2>(values),
        _ => KernelArg::f32(values.to_vec()),
    }
}

/// `len` integers from -8 to 8, from a generator seeded with `seed`. They
/// are exact in f16, bf16, E4M3 and E5M2, and their sums of products are exact in f32
/// (below 2^24 for any `k` up to 2^18), so the result is exact; they are
/// random, so an element read from or written to the wrong place shows.
pub fn small_integers(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 17) as f32 - 8.0
        })
        .collect()
}

fn main() {
    let cfg = GemmConfig::new(256, 256, 256);
    let report = run(cfg);
    println!("tensorop_gemm {}x{}x{}: correct\n{report}\n", cfg.m, cfg.n, cfg.k);

    let unswizzled = run(GemmConfig { swizzle: false, ..cfg });
    println!(
        "without the swizzle, ldmatrix and shared-memory traffic is {:.2}x serialised by bank conflicts (vs {:.2}x)\n",
        unswizzled.bank_conflict_factor(),
        report.bank_conflict_factor()
    );

    let fp8 = GemmConfig::fp8(256, 256, 256, DType::F8E4M3FN);
    let report = run(fp8);
    println!("tensorop_gemm Float8E4M3FN {}x{}x{} (Float32 C): correct\n{report}\n", fp8.m, fp8.n, fp8.k);
    let unswizzled = run(GemmConfig { swizzle: false, ..fp8 });
    println!(
        "FP8, without the swizzle: {:.2}x serialised by bank conflicts (vs {:.2}x with the example's Swizzle<2,4,4>)\n",
        unswizzled.bank_conflict_factor(),
        report.bank_conflict_factor()
    );
    run(GemmConfig { c: DType::F8E5M2, ab: DType::F8E5M2, ..fp8 });
    println!("tensorop_gemm Float8E5M2 with Float8E5M2 C: correct\n");

    match Toolchain::discover() {
        Some(tc) => match build(cfg).compile_ptx(&tc, Arch::Sm80) {
            Ok(ptx) => {
                let count = |pat: &str| ptx.lines().filter(|l| l.contains(pat)).count();
                println!(
                    "PTX: {} lines; {} mma.sync, {} ldmatrix, {} cp.async per K-tile iteration",
                    ptx.lines().count(),
                    count("mma.sync.aligned.m16n8k16"),
                    count("ldmatrix.sync"),
                    count("cp.async.cg"),
                );
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
    // The FP8 kernel for SM89: nvvm.mma.sync needs LLVM 23, else inline PTX.
    if let Some(tc) = Toolchain::discover() {
        let kernel = build(fp8);
        let old = kernel.min_llvm_version().is_some_and(|need| tc.llvm_version().is_some_and(|have| have < need));
        let options = CompileOptions::new().gpu_arch(Arch::Sm89).mma_inline_ptx(old);
        match kernel.compile_with(&tc, &options) {
            Ok(compiled) => {
                let count = |pat: &str| compiled.ptx().lines().filter(|l| l.contains(pat)).count();
                println!(
                    "FP8 PTX (sm_89): {} mma.sync.m16n8k32 e4m3, {} ldmatrix per K-tile iteration",
                    count("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"),
                    count("ldmatrix.sync"),
                );
            }
            Err(e) => eprintln!("FP8 PTX compilation failed: {e}"),
        }
    }
}
