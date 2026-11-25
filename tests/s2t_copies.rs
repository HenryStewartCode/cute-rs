// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The scale-factor copy (filter_zeros of both sides, make_s2t_copy over the TMEM tensor,
// get_slice(0), partition_S / partition_D and get_s2t_smem_desc_tensor) follows NVIDIA
// CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! Shared-memory-to-TMEM copies (roadmap part m10-s2t-copies): every `tcgen05.cp` shape
//! with `cta_group::1` on the simulator (zeroed TMEM, a copy issued by a converged warp, a
//! commit and an mbarrier wait, then `tcgen05.ld` read-back compared with plain Rust,
//! including the replicated lanes of the broadcast forms); `cta_group::2` in a `(2, 1, 1)`
//! cluster; a scale-factor tile copied through make_s2t_copy and get_s2t_smem_desc_tensor,
//! and through block_copy; append_s2t_broadcast_mode; the simulator's misuse reports; and
//! the sm_100a PTX (assembled by `ptxas` when it is on PATH).

use cute::block::block_copy;
use cute::ir::TmemCpShape;
use cute::prelude::*;
use cute::tcgen05::{
    self, Cp2x64x128b0123Op, Cp2x64x128b0213Op, Cp4x32x128bOp, Cp4x256bOp, Cp128x128bOp, Cp128x256bOp, CtaGroup,
    Ld32x32bOp, Repetition, St32x32bOp,
};

/// A TMEM lane's stride in 32-bit elements.
const LANE: usize = 1 << 16;

#[derive(Clone, Copy, Default, PartialEq)]
enum Fault {
    #[default]
    None,
    NoProxyFence,
    OverwriteInFlight,
    InsideElectOne,
}

/// The source rows (PTX ISA) written to each TMEM lane: the expected lane map, written out
/// independently of the library's.
fn lanes_of(shape: TmemCpShape, row: usize) -> Vec<usize> {
    use TmemCpShape::*;
    match shape {
        S128x256b | S128x128b => vec![row],
        S4x256b => vec![32 * row],
        S64x128bWarpx2_02_13 => vec![row, row + 64],
        S64x128bWarpx2_01_23 if row < 32 => vec![row, row + 32],
        S64x128bWarpx2_01_23 => vec![row + 32, row + 64],
        S32x128bWarpx4 => vec![row, row + 32, row + 64, row + 96],
    }
}

fn op_of(shape: TmemCpShape, group: CtaGroup) -> CopyOp {
    use TmemCpShape::*;
    match shape {
        S128x256b => Cp128x256bOp::new(group).into(),
        S128x128b => Cp128x128bOp::new(group).into(),
        S4x256b => Cp4x256bOp::new(group).into(),
        S32x128bWarpx4 => Cp4x32x128bOp::new(group).into(),
        S64x128bWarpx2_02_13 => Cp2x64x128b0213Op::new(group).into(),
        S64x128bWarpx2_01_23 => Cp2x64x128b0123Op::new(group).into(),
    }
}

/// Source rows (in shared memory) and 32-bit words per row.
fn geometry(shape: TmemCpShape) -> (usize, usize) {
    use TmemCpShape::*;
    let rows = match shape {
        S128x256b | S128x128b | S4x256b => 128,
        S64x128bWarpx2_02_13 | S64x128bWarpx2_01_23 => 64,
        S32x128bWarpx4 => 32,
    };
    (rows, shape.row_bits() / 32)
}

/// The canonical K-major tile (no swizzle) of `rows` rows of `u` 16-byte units, in Uint32s:
/// core matrices of 8 rows of 16 bytes, the next along a row 128 bytes on (LBO), the next
/// 8 rows `128 u` bytes on (SBO).
fn tile_modes(rows: usize, u: usize) -> (Layout, Layout) {
    (Layout::new((8, rows / 8), (4, 32 * u)), Layout::new((4, u), (1, 32)))
}

/// The atom's values on the shared-memory side (as partition_s would give them): the core
/// matrices, then the repeat and broadcast modes.
fn atom_smem_values(shape: TmemCpShape) -> Layout {
    use TmemCpShape::*;
    let u = shape.row_bits() / 128;
    let (rows, k) = match shape {
        S64x128bWarpx2_01_23 => tile_modes(32, u),
        _ => tile_modes(geometry(shape).0, u),
    };
    let sbo = 32 * u;
    let mut modes = vec![rows, k];
    match shape {
        S32x128bWarpx4 => modes.push(Layout::new(4, 0)),
        S64x128bWarpx2_02_13 => modes.push(Layout::new(2, 0)),
        S64x128bWarpx2_01_23 => modes.extend([Layout::new(2, 4 * sbo), Layout::new(2, 0)]),
        _ => {}
    }
    Layout::from_modes(modes)
}

/// The atom's TMEM footprint in Uint32s (CuTe C++'s ValID upcast to 32 bits).
fn atom_tmem_values(shape: TmemCpShape) -> Layout {
    use TmemCpShape::*;
    match shape {
        S128x256b | S4x256b => Layout::new((128, 8), (LANE, 1)),
        S128x128b => Layout::new((128, 4), (LANE, 1)),
        S32x128bWarpx4 => Layout::new((32, 4, 4), (LANE, 1, 32 * LANE)),
        S64x128bWarpx2_02_13 => Layout::new((64, 4, 2), (LANE, 1, 64 * LANE)),
        S64x128bWarpx2_01_23 => Layout::new((32, 4, 2, 2), (LANE, 1, 64 * LANE, 32 * LANE)),
    }
}

fn rep(n: usize) -> Repetition {
    match n {
        4 => Repetition::X4,
        8 => Repetition::X8,
        _ => Repetition::X16,
    }
}

/// Each warp's 32 lanes of `cols` columns of the TMEM at `tmem`, through `.32x32b` loads
/// (`load`) or zeroing stores.
fn quadrant(tmem: &Ptr<u32>, cols: usize) -> Tensor<u32> {
    let w = make_warp_uniform(warp_idx()) % 4;
    make_tensor(tmem.clone() + w * (32 * LANE) as i32, Layout::from_modes([Layout::new((cols, 32), (1, LANE))]))
}

fn zero_quadrant(tmem: &Ptr<u32>, cols: usize) {
    let frag = make_fragment::<u32>(cols);
    for e in 0..cols {
        frag.store(e, Uint32::from(0u32));
    }
    copy(&make_copy_atom::<u32>(St32x32bOp::new(rep(cols))), &frag, &quadrant(tmem, cols));
    fence_view_async_tmem_store();
}

fn read_quadrant(tmem: &Ptr<u32>, cols: usize, out: &Tensor<u32>, base: Int32) {
    let frag = make_fragment::<u32>(cols);
    copy(&make_copy_atom::<u32>(Ld32x32bOp::new(rep(cols))), &quadrant(tmem, cols), &frag);
    fence_view_async_tmem_load();
    let lane = (make_warp_uniform(warp_idx()) % 4) * 32 + lane_idx();
    for e in 0..cols {
        out.store(base + lane * cols as i32 + e as i32, frag.load(e));
    }
}

fn sync(two: bool) {
    if two {
        cluster_arrive();
        cluster_wait();
    } else {
        sync_threads();
    }
}

/// One copy of `shape`: each CTA fills its shared tile from `g` (its own part), zeroes 8
/// columns of its TMEM, warp 0 of the (even) CTA copies and commits, and every warp waits
/// and reads its quadrant back into `out`.
fn shape_kernel(shape: TmemCpShape, two: bool, fault: Fault) -> Kernel {
    let name = format!("s2t_{}{}", shape.ptx().replace(['.', ':'], "_"), if two { "_pair" } else { "" });
    Kernel::builder(&name).block_dim([128, 1, 1]).target(Arch::Sm100a).build(move || {
        let ncta = if two { 2 } else { 1 };
        let (rows, words) = geometry(shape);
        let u = words / 4;
        let g = param_tensor::<u32>("g", Layout::col_major(ncta * rows * words));
        let out = param_tensor::<u32>("out", Layout::col_major(ncta * 128 * words));
        let (rm, km) = tile_modes(rows, u);
        let s = make_smem_tensor_aligned::<u32>(Layout::from_modes([rm, km]), 128);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let warp = make_warp_uniform(warp_idx());
        let rank = if two { block_idx_in_cluster() } else { Int32::from(0) };
        let group = if two { CtaGroup::TWO } else { CtaGroup::ONE };
        if_(warp.eq(0), || alloc_tmem(32, &slot, two.then_some(true), Arch::Sm100a));
        for_range(0, (rows * words / 128) as i32, 1, |j| {
            let idx = j * 128 + t;
            s.store((idx / words as i32, idx % words as i32), g.load(rank * (rows * words) as i32 + idx));
        });
        if fault != Fault::NoProxyFence {
            fence_view_async_shared();
        }
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync(two);
        let tmem = retrieve_tmem_ptr::<u32>(16, &slot);
        zero_quadrant(&tmem, 8);
        sync(two);
        let atom = make_copy_atom::<u32>(op_of(shape, group));
        let part = make_tensor(
            s.ptr(),
            Layout::from_modes([Layout::from_modes([atom_smem_values(shape), Layout::new(1, 0)])]),
        );
        let desc = tcgen05::get_s2t_smem_desc_tensor(&atom, &part);
        let dst = make_tensor(tmem.clone(), Layout::from_modes([atom_tmem_values(shape)]));
        if_(warp.eq(0) & rank.eq(0), || {
            if fault == Fault::InsideElectOne {
                if_(elect_one(), || copy(&atom, &desc, &dst));
            } else {
                copy(&atom, &desc, &dst);
            }
            let mask = two.then(|| Int16::from(0b11i16));
            if_(elect_one(), || tcgen05::commit(&mbar, mask, group));
        });
        if fault == Fault::OverwriteInFlight {
            s.store((t % rows as i32, 0), g.load(0));
        }
        mbarrier_wait(&mbar, 0);
        read_quadrant(&tmem, words, &out, rank * (128 * words) as i32);
        fence_view_async_tmem_load();
        sync(two);
        if_(warp.eq(0), || {
            relinquish_tmem_alloc_permit(two.then_some(true));
            dealloc_tmem(&tmem, 32, two.then_some(true), Arch::Sm100a);
        });
    })
}

fn source(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| i.wrapping_mul(2654435761) ^ 0x5a5a_0000).collect()
}

fn run_shape(shape: TmemCpShape, two: bool) {
    let (rows, words) = geometry(shape);
    let ncta = if two { 2 } else { 1 };
    let g = source(ncta * rows * words);
    let mut args = vec![KernelArg::buffer(&g), KernelArg::buffer(&vec![0u32; ncta * 128 * words])];
    let cfg = if two {
        LaunchConfig::new([2, 1, 1], [128, 1, 1]).with_cluster([2, 1, 1])
    } else {
        LaunchConfig::new([1, 1, 1], [128, 1, 1])
    };
    shape_kernel(shape, two, Fault::None)
        .simulate_all_orders(cfg, &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{shape:?} (pair {two}): {e}"));
    let got = args[1].to_vec::<u32>();
    let mut want = vec![0u32; ncta * 128 * words];
    for cta in 0..ncta {
        let read = match shape {
            TmemCpShape::S4x256b => 4,
            _ => rows,
        };
        for r in 0..read {
            for l in lanes_of(shape, r) {
                for c in 0..words {
                    want[cta * 128 * words + l * words + c] = g[cta * rows * words + r * words + c];
                }
            }
        }
    }
    assert_eq!(got, want, "{shape:?} (pair {two})");
}

#[test]
fn every_shape_copies_with_one_cta() {
    for shape in TmemCpShape::ALL {
        run_shape(shape, false);
    }
}

#[test]
fn a_cta_pair_copies_into_both_ctas() {
    for shape in [TmemCpShape::S128x256b, TmemCpShape::S32x128bWarpx4, TmemCpShape::S64x128bWarpx2_01_23] {
        run_shape(shape, true);
    }
}

fn report_for(fault: Fault) -> String {
    let shape = TmemCpShape::S128x128b;
    let (rows, words) = geometry(shape);
    let g = source(rows * words);
    let mut args = vec![KernelArg::buffer(&g), KernelArg::buffer(&vec![0u32; 128 * words])];
    match shape_kernel(shape, false, fault).simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args) {
        Ok(_) => panic!("the fault was not reported"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_missing_proxy_fence_is_reported() {
    let e = report_for(Fault::NoProxyFence);
    assert!(e.contains("tcgen05.cp") && e.contains("fence.proxy.async"), "{e}");
}

#[test]
fn overwriting_the_source_in_flight_is_reported() {
    let e = report_for(Fault::OverwriteInFlight);
    assert!(e.contains("still reads"), "{e}");
}

#[test]
fn a_copy_inside_elect_one_is_reported() {
    let e = report_for(Fault::InsideElectOne);
    assert!(e.contains("executed by 1 of the 32 lanes") && e.contains("elect_one"), "{e}");
}

// The scale-factor tile of a 128-row MMA with 16-element scale vectors: in shared memory
// BlockScaledBasicChunk's ((32,4),(16,4)):((16,4),(0,1)) bytes, two K tiles; in TMEM the
// layout CuTe C++'s tmem_sf_frg gives (each 32-row block replicated in the four lane
// quadrants), read back after a Cp4x32x128bOp copy.

/// The shared-memory SF layout: `(((32,4),(16,4)), KT):(((16,4),(0,1)), 512)`.
fn sf_smem_layout() -> Layout {
    let chunk = utils::BlockScaledBasicChunk::new(16, OperandMajorMode::K);
    Layout::from_modes([chunk.layout().clone(), Layout::new(2, 512)])
}

/// tmem_sf_frg's layout for MMA_MN 128, 16-element vectors, 4 per MMA, two K tiles, in
/// bytes: `((((32,4),4),(16,4)), KT)` mapping MN `(r, q)`, replica `p`, `(v, s)` to lane
/// `r + 32 p` and byte `4 q + s` of the TMEM (4 columns per K tile).
fn sf_tmem_layout() -> Layout {
    let dp = 4 * LANE;
    let mn = Layout::new(((32, 4), 4), ((dp, 4), 32 * dp));
    let k = Layout::new((16, 4), (0, 1));
    Layout::from_modes([Layout::from_modes([mn, k]), Layout::new(2, 16)])
}

fn sf_kernel(via_block_copy: bool) -> Kernel {
    Kernel::builder(if via_block_copy { "s2t_sf_block_copy" } else { "s2t_sf" })
        .block_dim([128, 1, 1])
        .target(Arch::Sm100a)
        .build(move || {
            let g = param_tensor::<u8>("g", Layout::col_major(1024));
            let out = param_tensor::<u32>("out", Layout::col_major(128 * 8));
            let s = make_smem_tensor_aligned::<u8>(sf_smem_layout(), 128);
            let raw = make_tensor(s.ptr(), Layout::col_major(1024));
            let slot = alloc_smem::<u32>(1, 4);
            let mbar = alloc_smem::<u64>(1, 8);
            let t = thread_idx().x;
            let warp = make_warp_uniform(warp_idx());
            if_(warp.eq(0), || alloc_tmem(32, &slot, None, Arch::Sm100a));
            for_range(0, 8, 1, |j| raw.store(j * 128 + t, g.load(j * 128 + t)));
            fence_view_async_shared();
            if_(t.eq(0), || mbarrier_init(&mbar, 1));
            mbarrier_init_fence();
            sync_threads();
            let tmem32 = retrieve_tmem_ptr::<u32>(16, &slot);
            zero_quadrant(&tmem32, 8);
            sync_threads();
            let tmem = retrieve_tmem_ptr::<u8>(16, &slot);
            let tsf = make_tensor(tmem, sf_tmem_layout());
            let atom = make_copy_atom::<u8>(Cp4x32x128bOp::new(CtaGroup::ONE));
            let tiled = tcgen05::make_s2t_copy(&atom, &tsf.with_layout(tsf.layout().filter_zeros()));
            if_(warp.eq(0), || {
                if via_block_copy {
                    block_copy(&tiled, &s, &tsf, CopyArgs::new());
                } else {
                    let thr = tiled.get_slice(0);
                    let compact = s.with_layout(s.layout().filter_zeros());
                    let src = thr.partition_s(&tcgen05::append_s2t_broadcast_mode(atom.op(), &compact));
                    let desc = tcgen05::get_s2t_smem_desc_tensor(&tiled, &src);
                    let dst = thr.partition_d(&tsf.with_layout(tsf.layout().filter_zeros()));
                    copy(&tiled, &desc, &dst);
                }
                if_(elect_one(), || tcgen05::commit(&mbar, None, CtaGroup::ONE));
            });
            mbarrier_wait(&mbar, 0);
            read_quadrant(&tmem32, 8, &out, Int32::from(0));
            fence_view_async_tmem_load();
            sync_threads();
            if_(warp.eq(0), || {
                relinquish_tmem_alloc_permit(None);
                dealloc_tmem(&tmem32, 32, None, Arch::Sm100a);
            });
        })
}

#[test]
fn scale_factor_tiles_land_where_the_tmem_layout_says() {
    let g: Vec<u8> = (0..1024u32).map(|i| (i * 37 + 11) as u8).collect();
    let (smem, tmem) = (sf_smem_layout(), sf_tmem_layout());
    // Every coordinate of the TMEM layout holds the shared-memory byte of the same (MN, K)
    // coordinate (the replica mode aside).
    let mut want = vec![0u32; 128 * 8];
    for kt in 0..2 {
        for s in 0..4 {
            for v in 0..16 {
                for m in 0..128 {
                    let byte = g[smem.call(m + 128 * (v + 16 * s + 64 * kt))];
                    for p in 0..4 {
                        let off = tmem.call(m + 128 * p + 512 * (v + 16 * s + 64 * kt));
                        let (lane, b) = (off / (4 * LANE), off % (4 * LANE));
                        let w = &mut want[lane * 8 + b / 4];
                        *w = (*w & !(0xff << (8 * (b % 4)))) | (byte as u32) << (8 * (b % 4));
                    }
                }
            }
        }
    }
    for via in [false, true] {
        let mut args = vec![KernelArg::buffer(&g), KernelArg::buffer(&vec![0u32; 128 * 8])];
        sf_kernel(via)
            .simulate_all_orders(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args, &SimOptions::default())
            .unwrap_or_else(|e| panic!("block_copy {via}: {e}"));
        assert_eq!(args[1].to_vec::<u32>(), want, "block_copy {via}");
    }
}

#[test]
fn make_s2t_copy_partitions_the_scale_factors() {
    // Build-time: the partitions of nvfp4_gemm_0.py's SFA copy shape.
    let _ = Kernel::builder("s2t_partitions").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
        let s = make_smem_tensor_aligned::<u8>(sf_smem_layout(), 128);
        let slot = alloc_smem::<u32>(1, 4);
        let tsf = make_tensor(retrieve_tmem_ptr::<u8>(16, &slot), sf_tmem_layout());
        let atom = make_copy_atom::<u8>(Cp4x32x128bOp::new(CtaGroup::ONE));
        let tiled = tcgen05::make_s2t_copy(&atom, &tsf.with_layout(tsf.layout().filter_zeros()));
        let thr = tiled.get_slice(0);
        let compact = s.with_layout(s.layout().filter_zeros());
        let src = thr.partition_s(&tcgen05::append_s2t_broadcast_mode(atom.op(), &compact));
        let dst = thr.partition_d(&tsf.with_layout(tsf.layout().filter_zeros()));
        // One atom (32 rows x 16 bytes, broadcast 4) per K tile.
        assert_eq!(dst.layout().mode(0).size(), 2048, "{}", dst.layout());
        assert_eq!(size(&dst, &[]).as_static(), Some(4096), "{}", dst.layout());
        let desc = tcgen05::get_s2t_smem_desc_tensor(&tiled, &src);
        // Descriptors step by the K tile's 512 bytes (32 units).
        assert_eq!(desc.layout().mode(desc.layout().rank() - 1).static_stride().flat_leaves().last(), Some(&32));
    });
}

#[test]
fn a_broadcast_copy_without_its_broadcast_mode_is_refused() {
    let e = panic_text(|| {
        let _ = Kernel::builder("s2t_no_bcast").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
            let s = make_smem_tensor_aligned::<u8>(sf_smem_layout(), 128);
            let slot = alloc_smem::<u32>(1, 4);
            let tsf = make_tensor(retrieve_tmem_ptr::<u8>(16, &slot), sf_tmem_layout());
            let atom = make_copy_atom::<u8>(Cp4x32x128bOp::new(CtaGroup::ONE));
            let tiled = tcgen05::make_s2t_copy(&atom, &tsf.with_layout(tsf.layout().filter_zeros()));
            let src = tiled.get_slice(0).partition_s(&s.with_layout(s.layout().filter_zeros()));
            let _ = tcgen05::get_s2t_smem_desc_tensor(&tiled, &src);
        });
    });
    assert!(e.contains("append_s2t_broadcast_mode"), "{e}");
}

#[test]
fn broadcast_modes_are_appended() {
    let _ = Kernel::builder("s2t_bcast").block_dim([32, 1, 1]).target(Arch::Sm100a).build(|| {
        let s = make_smem_tensor_aligned::<u8>(Layout::new(((32, 16), 2), ((16, 1), 512)), 128);
        let b = tcgen05::append_s2t_broadcast_mode(Cp4x32x128bOp::new(CtaGroup::ONE), &s);
        let want = Layout::from_modes([
            Layout::from_modes([Layout::new((32, 4), (16, 0)), Layout::new(16, 1)]),
            Layout::new(2, 512),
        ]);
        assert_eq!(b.layout(), &want, "{}", b.layout());
        let b2 = tcgen05::append_s2t_broadcast_mode(Cp2x64x128b0213Op::new(CtaGroup::ONE), &s);
        assert_eq!(b2.layout().mode(0).mode(0), Layout::new((32, 2), (16, 0)));
        let same = tcgen05::append_s2t_broadcast_mode(Cp128x128bOp::new(CtaGroup::ONE), &s);
        assert_eq!(same.layout(), s.layout());
    });
    let e = panic_text(|| {
        let _ = Kernel::builder("s2t_bcast_bad").block_dim([32, 1, 1]).target(Arch::Sm100a).build(|| {
            let s = make_smem_tensor_aligned::<u8>(Layout::new((512, 2), (1, 512)), 128);
            let _ = tcgen05::append_s2t_broadcast_mode(Cp4x32x128bOp::new(CtaGroup::ONE), &s);
        });
    });
    assert!(e.contains("(MN, K)"), "{e}");
    let e = panic_text(|| {
        let _ = Kernel::builder("s2t_bcast_op").block_dim([32, 1, 1]).target(Arch::Sm100a).build(|| {
            let s = make_smem_tensor_aligned::<u8>(Layout::new(((32, 16), 2), ((16, 1), 512)), 128);
            let _ = tcgen05::append_s2t_broadcast_mode(Ld32x32bOp::new(Repetition::X1), &s);
        });
    });
    assert!(e.contains("SMEM-to-TMEM"), "{e}");
}

fn panic_text(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let e = std::panic::catch_unwind(f).expect_err("expected a panic");
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
fn s2t_ops_report_their_broadcast_factor() {
    assert_eq!(Cp4x32x128bOp::new(CtaGroup::ONE).broadcast_factor(), 4);
    assert_eq!(Cp2x64x128b0123Op::new(CtaGroup::TWO).broadcast_factor(), 2);
    assert_eq!(Cp2x64x128b0213Op::default().broadcast_factor(), 2);
    assert_eq!(Cp128x256bOp::default().broadcast_factor(), 1);
    assert_eq!(Cp4x256bOp::default().broadcast_factor(), 1);
}

#[test]
fn s2t_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    let mut kernels: Vec<(Kernel, String)> = TmemCpShape::ALL
        .into_iter()
        .map(|s| (shape_kernel(s, false, Fault::None), format!("tcgen05.cp.cta_group::1.{}", s.ptx())))
        .collect();
    kernels.push((
        shape_kernel(TmemCpShape::S32x128bWarpx4, true, Fault::None),
        "tcgen05.cp.cta_group::2.32x128b.warpx4".into(),
    ));
    for (kernel, want) in kernels {
        let name = kernel.ir().name.clone();
        let compiled = kernel
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptx = compiled.ptx();
        assert!(ptx.contains(&want), "{name}: no `{want}` in the PTX");
        let Some(ptxas) = &ptxas else { continue };
        let dir = std::env::temp_dir().join(format!("cute-s2t-{}-{name}", std::process::id()));
        let (src, cubin) = (dir.with_extension("ptx"), dir.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{name}: ptxas: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn s2t_mlir_text() {
    let text = shape_kernel(TmemCpShape::S128x256b, false, Fault::None).to_mlir();
    assert!(text.contains("@p tcgen05.cp.cta_group::1.128x256b [$0], $1;"), "{text}");
    assert!(text.contains("nvvm.tcgen05.commit"), "{text}");
}

/// block_copy with a TMA tile load: a 64 x 64 Float16 tile into a 128-byte-swizzled shared
/// tile, then copied out by every thread.
#[test]
fn block_copy_loads_a_tma_tile() {
    let kernel = Kernel::builder("block_copy_tma").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64), (64, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((64, 64), (64, 1)));
        let layout = tile_to_shape_swizzled(
            &cute::warpgroup::make_smem_layout_atom::<F16>(cute::warpgroup::SmemLayoutAtomKind::K_SW128),
            (64, 64),
        );
        let s = make_smem_tensor_aligned::<F16>(layout.clone(), 1024);
        let mbar = alloc_smem::<u64>(1, 8);
        let (atom, tma_tensor) =
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SOp::default(), &a, layout, (64, 64), 1).into();
        let g = local_tile(&tma_tensor, (64, 64), (0, 0));
        let t = thread_idx().x;
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        if_(warp_idx().eq(0), || {
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, 64 * 64 * 2));
            block_copy(&atom, &g.group_modes(0, 2), &s.group_modes(0, 2), CopyArgs::new().tma_bar_ptr(&mbar));
        });
        mbarrier_wait(&mbar, 0);
        for_range(0, 32, 1, |j| {
            let idx = j * 128 + t;
            let (r, c) = (idx / 64, idx % 64);
            out.store((r, c), s.load((r, c)));
        });
    });
    let a: Vec<F16> = (0..4096).map(|i| F16::from_f32((i % 251) as f32)).collect();
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![F16::from_f32(0.0); 4096])];
    kernel.simulate_all_orders(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args, &SimOptions::default()).unwrap();
    assert_eq!(args[1].to_vec::<F16>(), a);
}
