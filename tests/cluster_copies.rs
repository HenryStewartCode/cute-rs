//! Copies across the CTAs of a cluster (roadmap part m7-cluster-copies): multicast
//! masks over every CTA rank of 2x1, 2x2 and 4x2 clusters, the feature kernels under
//! seeded schedules and random completion, the simulator's reports of misuse, and the
//! MLIR and PTX the copies lower to.

mod feature_kernels;

use cute::prelude::*;

/// `make_layout_image_mask` and `create_tma_multicast_mask` on the simulator, over every
/// CTA rank of 2x1, 2x2 and 4x2 cluster layouts, with runtime and static coordinates,
/// against plain Rust.
#[test]
fn image_masks_match_rust_over_every_cta() {
    for (cm, cn) in feature_kernels::MASK_CLUSTERS {
        let kernel = feature_kernels::image_masks(cm, cn);
        let size = cm * cn;
        let mut args = vec![KernelArg::buffer(&vec![0i16; (2 + 2 * size) * size])];
        let cfg = LaunchConfig::new([size as u32, 1, 1], [32, 1, 1]).with_cluster([size as u32, 1, 1]);
        kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{cm}x{cn}: {e}"));
        assert_eq!(args[0].to_vec::<i16>(), feature_kernels::image_masks_reference(cm, cn), "{cm}x{cn}");
    }
}

/// Every cluster-copy kernel under the built-in schedules and 8 seeded ones with random
/// completion: each matches plain Rust, and the report counts the copies into peers.
#[test]
fn cluster_copies_match_rust_under_seeded_schedules() {
    let seeds: Vec<u64> = (0..8).collect();
    for case in feature_kernels::cluster_copy_cases_sm90() {
        let mut args = feature_kernels::inputs_with_len(&case.kernel, true, 256);
        let report = case
            .kernel
            .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
        assert!(report.cluster_copies > 0, "{}: {report}", case.name);
        let mut args = feature_kernels::inputs_with_len(&case.kernel, true, 256);
        case.kernel
            .simulate_schedules(case.cfg, &mut args, &SimOptions::default(), &seeds)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
    }
}

fn build_panic(f: impl FnOnce()) -> String {
    let err =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// Build a kernel around `body` and expect it to panic with every one of `want`.
#[track_caller]
fn expect_build_panic(body: impl FnOnce() + 'static, want: &[&str]) {
    let msg = build_panic(move || {
        Kernel::builder("misuse").block_dim([64, 1, 1]).build(body);
    });
    for w in want {
        assert!(msg.contains(w), "expected {w:?} in: {msg}");
    }
}

/// What is refused when the kernel is built: a multicast load or bulk copy without its
/// mask, a peer bulk copy without its CTA rank or with a cache hint, and `st.async` of
/// three values, of 16-bit values or to a global pointer.
#[test]
fn cluster_copy_misuse_is_refused_when_built() {
    expect_build_panic(
        || {
            let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64), (64, 1)), 16);
            let smem = Layout::new((64, 64), (64, 1));
            let mc = cpasync::CopyBulkTensorTileG2SMulticastOp::default();
            let (atom, tma) = cpasync::make_tiled_tma_atom(mc, &a, smem.clone(), (64, 64), 2).into();
            let s = make_smem_tensor_aligned::<F16>(smem, 128);
            let mbar = alloc_smem::<u64>(1, 8);
            let (ts, tg) =
                cpasync::tma_partition(&atom, 0, &make_layout(2, 1), &s.group_modes(0, 2), &tma.group_modes(0, 2));
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar));
        },
        &["needs `mcast_mask`", "CopyArgs::new().mcast_mask(mask)"],
    );
    expect_build_panic(
        || {
            let x = param_tensor_aligned::<f32>("x", Layout::col_major(64), 16);
            let s = make_smem_tensor_aligned::<f32>(Layout::col_major(64), 128);
            let mbar = alloc_smem::<u64>(1, 8);
            let atom = make_copy_atom::<f32>(cpasync::CopyBulkG2SMulticastOp);
            copy_with(&atom, &x, &s, CopyArgs::new().mbar_ptr(&mbar));
        },
        &["needs `mcast_mask`"],
    );
    expect_build_panic(
        || {
            let s = make_smem_tensor_aligned::<f32>(Layout::col_major(64), 128);
            let r = make_smem_tensor_aligned::<f32>(Layout::col_major(64), 128);
            let mbar = alloc_smem::<u64>(1, 8);
            copy_with(&make_copy_atom::<f32>(cpasync::CopyBulkS2SOp), &s, &r, CopyArgs::new().mbar_ptr(&mbar));
        },
        &["needs `cta_rank`", "CopyArgs::new().cta_rank(rank)"],
    );
    expect_build_panic(
        || {
            let s = make_smem_tensor_aligned::<f32>(Layout::col_major(64), 128);
            let r = make_smem_tensor_aligned::<f32>(Layout::col_major(64), 128);
            let mbar = alloc_smem::<u64>(1, 8);
            let args = CopyArgs::new().mbar_ptr(&mbar).cta_rank(1).cache_policy(0i64);
            copy_with(&make_copy_atom::<f32>(cpasync::CopyBulkS2SOp), &s, &r, args);
        },
        &["takes no `cache_policy` argument"],
    );
    expect_build_panic(
        || {
            let s = make_smem_tensor_aligned::<i32>(Layout::col_major(64), 128);
            let mbar = alloc_smem::<u64>(1, 8);
            let v = Int32::from(1);
            store_async_dsmem(&s.ptr(), [v, v, v], &mbar, 1);
        },
        &["st.async stores one, two or four values, not 3"],
    );
    expect_build_panic(
        || {
            let s = make_smem_tensor_aligned::<F16>(Layout::col_major(64), 128);
            let mbar = alloc_smem::<u64>(1, 8);
            store_async_dsmem(&s.ptr(), [F16::ZERO.into_value()], &mbar, 1);
        },
        &["st.async stores 32-bit values (Int32, Uint32 or Float32), not Float16"],
    );
    expect_build_panic(
        || {
            let s = make_smem_tensor_aligned::<i32>(Layout::col_major(64), 128);
            let mbar = alloc_smem::<u64>(1, 8);
            let v = Int32::from(1);
            store_async_dsmem(&s.ptr().offset(2), [v, v, v, v], &mbar, 1);
        },
        &["is at byte 8 of its buffer, which is not 16-byte aligned"],
    );
}

/// The ways the cluster copies can go wrong at run time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Misuse {
    None,
    /// A multicast mask with a bit beyond the cluster of two.
    MaskBeyond,
    /// The receiving CTA's `expect_tx` counts only its own share of the tile.
    ExpectOwnShare,
    /// A thread reads the multicast tile before waiting on its mbarrier.
    ReadBeforeWait,
    /// No cluster barrier between the mbarrier initialisation and the copies.
    NoInitPublish,
    /// The launch has no clusters.
    NoClusters,
}

/// A 64x64 f16 tile shared by the two CTAs of a cluster (a multicast TMA load, each
/// CTA loading half and multicasting it), copied to `out` after the wait, with
/// `misuse` built in.
fn multicast_misuse(misuse: Misuse) -> Kernel {
    Kernel::builder("multicast_misuse").block_dim([64, 1, 1]).build(move || {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64), (64, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((64, 64, 2), (64, 1, 4096)));
        let smem = cute::warpgroup::make_smem_layout_atom::<F16>(cute::warpgroup::SmemLayoutAtomKind::K_SW128);
        let smem = tile_to_shape_swizzled(&smem, (64, 64));
        let mc = cpasync::CopyBulkTensorTileG2SMulticastOp::default();
        let (atom, tma) = cpasync::make_tiled_tma_atom(mc, &a, smem.clone(), (64, 64), 2).into();
        let s = make_smem_tensor_aligned::<F16>(smem, 1024);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let rank = if misuse == Misuse::NoClusters { block_idx().x } else { block_idx_in_cluster() };
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        if !matches!(misuse, Misuse::NoInitPublish | Misuse::NoClusters) {
            cluster_arrive_relaxed();
            cluster_wait();
        }
        let (ts, tg) =
            cpasync::tma_partition(&atom, rank, &make_layout(2, 1), &s.group_modes(0, 2), &tma.group_modes(0, 2));
        let mask = if misuse == Misuse::MaskBeyond { 0b111i16 } else { 0b11i16 };
        let bytes = if misuse == Misuse::ExpectOwnShare { 4096 } else { 8192 };
        if_(warp_idx().eq(0), || {
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, bytes));
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar).mcast_mask(mask));
        });
        if matches!(misuse, Misuse::ExpectOwnShare | Misuse::ReadBeforeWait) {
            // Both shares are issued before either CTA reads or waits.
            cluster_arrive();
            cluster_wait();
        }
        if misuse == Misuse::ReadBeforeWait {
            out.store((t, 0, rank), s.load((t, 0)));
        }
        mbarrier_wait(&mbar, 0);
        for_range(0, 64, 1, |j| out.store((t, j, rank), s.load((t, j))));
        if misuse != Misuse::NoClusters {
            cluster_arrive();
            cluster_wait();
        }
    })
}

fn run_multicast_misuse(misuse: Misuse) -> Result<Vec<F16>, String> {
    let kernel = multicast_misuse(misuse);
    let a: Vec<F16> = (0..64 * 64).map(|i| F16::from_f32((i % 13) as f32)).collect();
    let mut args = vec![KernelArg::F16(a), KernelArg::buffer(&vec![F16::ZERO; 2 * 64 * 64])];
    let cfg = LaunchConfig::new([2, 1, 1], [64, 1, 1]);
    let cfg = if misuse == Misuse::NoClusters { cfg } else { cfg.with_cluster([2, 1, 1]) };
    kernel.simulate(cfg, &mut args).map_err(|e| e.to_string())?;
    Ok(args[1].to_vec::<F16>())
}

/// The multicast kernel without misuse gives both CTAs the tile (each CTA's share lands
/// in the other at the same offset, through the 128-byte swizzle); each misuse is
/// reported.
#[test]
fn multicast_misuse_is_reported() {
    let out = run_multicast_misuse(Misuse::None).unwrap();
    for r in 0..2 {
        for (i, v) in out[r * 4096..(r + 1) * 4096].iter().enumerate() {
            assert_eq!(v.to_f32(), (i % 13) as f32, "CTA {r}, element {i}");
        }
    }
    for (misuse, want) in [
        (Misuse::MaskBeyond, &["the CTA mask 0x0007 has bits beyond the cluster's 2 CTA(s)"][..]),
        (
            Misuse::ExpectOwnShare,
            &["2 bulk copies of 8192 bytes", "transaction count is 4096", "expect_tx too small"][..],
        ),
        (Misuse::ReadBeforeWait, &["while a bulk copy is still writing it"][..]),
        (Misuse::NoInitPublish, &["before its initialisation was published to CTA", "cluster barrier"][..]),
        (Misuse::NoClusters, &["the launch has no clusters", "with_cluster"][..]),
    ] {
        let e = run_multicast_misuse(misuse).expect_err(&format!("{misuse:?} should be reported"));
        for w in want {
            assert!(e.contains(w), "{misuse:?}: expected {w:?} in: {e}");
        }
    }
}

/// Misuse of the peer copies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PeerMisuse {
    None,
    /// `cta_rank` 5 in a cluster of two.
    RankOutside,
    /// The sender exits while its copy still reads its shared memory.
    SenderExits,
    /// The receiver exits before the sender's copy (no cluster barrier at the end).
    ReceiverExits,
    /// `st.async` into the receiver, whose phase expects too few bytes.
    StAsyncTooMany,
}

/// CTA 0 of a cluster of two sends 64 floats to CTA 1 (a `CopyBulkS2SOp` copy, or four
/// `st.async`s of four values), which copies them to `out`.
fn peer_misuse(misuse: PeerMisuse) -> Kernel {
    Kernel::builder("peer_misuse").block_dim([64, 1, 1]).build(move || {
        let out = param_tensor::<f32>("out", Layout::col_major(64));
        let send = make_smem_tensor_aligned::<f32>(Layout::col_major(64), 128);
        let recv = make_smem_tensor_aligned::<f32>(Layout::col_major(64), 128);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let rank = block_idx_in_cluster();
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        send.store(t, t.to::<f32>() + 0.5f32);
        fence_view_async_shared();
        cluster_arrive();
        cluster_wait();
        let st_async = misuse == PeerMisuse::StAsyncTooMany;
        if_(rank.eq(0), || {
            // Give the receiver time to get ahead.
            let spin = for_range_carry(0, 200, 1, Int32::from(0), |i, acc| acc + i);
            out.store(63, spin.to::<f32>() * 0.0f32 + out.load(63));
            if_(t.eq(0), || {
                let peer = if misuse == PeerMisuse::RankOutside { 5 } else { 1 };
                if st_async {
                    for k in 0..16 {
                        let v = [send.load(k * 4), send.load(k * 4 + 1), send.load(k * 4 + 2), send.load(k * 4 + 3)];
                        store_async_dsmem(&recv.ptr().offset(k * 4), v, &mbar, peer);
                    }
                } else {
                    let atom = make_copy_atom::<f32>(cpasync::CopyBulkS2SOp);
                    copy_with(&atom, &send, &recv, CopyArgs::new().mbar_ptr(&mbar).cta_rank(peer));
                }
            });
        });
        if st_async {
            // Every st.async is issued before the receiver waits.
            if_(rank.eq(1) & t.eq(0), || mbarrier_arrive_and_expect_tx(&mbar, 128));
            cluster_arrive();
            cluster_wait();
        }
        if_(rank.eq(1), || {
            if misuse != PeerMisuse::ReceiverExits {
                if !st_async {
                    if_(t.eq(0), || mbarrier_arrive_and_expect_tx(&mbar, 256));
                }
                if misuse == PeerMisuse::SenderExits {
                    let spin = for_range_carry(0, 400, 1, Int32::from(0), |i, acc| acc + i);
                    out.store(62, spin.to::<f32>() * 0.0f32 + out.load(62));
                }
                mbarrier_wait(&mbar, 0);
                out.store(t, recv.load(t));
            }
        });
        if misuse == PeerMisuse::None || misuse == PeerMisuse::StAsyncTooMany {
            cluster_arrive();
            cluster_wait();
        }
    })
}

fn run_peer_misuse(misuse: PeerMisuse) -> Result<Vec<f32>, String> {
    let kernel = peer_misuse(misuse);
    let mut args = vec![KernelArg::buffer(&vec![0f32; 64])];
    let cfg = LaunchConfig::new([2, 1, 1], [64, 1, 1]).with_cluster([2, 1, 1]);
    kernel.simulate(cfg, &mut args).map_err(|e| e.to_string())?;
    Ok(args[0].to_vec::<f32>())
}

/// The peer copy without misuse delivers the data; each misuse is reported: a CTA rank
/// outside the cluster, a sender that exits while its copy still reads its shared memory,
/// a receiver that exits before a peer signals its mbarrier, and `st.async`s that land
/// more bytes than the receiving phase expects.
#[test]
fn peer_copy_misuse_is_reported() {
    let out = run_peer_misuse(PeerMisuse::None).unwrap();
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, i as f32 + 0.5, "element {i}");
    }
    for (misuse, want) in [
        (PeerMisuse::RankOutside, &["on CTA 5 of a cluster of 2 CTA(s)"][..]),
        (PeerMisuse::SenderExits, &["exits while an asynchronous operation still reads its shared memory"][..]),
        (PeerMisuse::ReceiverExits, &["of CTA 1, which has exited", "cluster barrier before a CTA exits"][..]),
        (PeerMisuse::StAsyncTooMany, &["16 bulk copies of 256 bytes", "transaction count is 128"][..]),
    ] {
        let e = run_peer_misuse(misuse).expect_err(&format!("{misuse:?} should be reported"));
        for w in want {
            assert!(e.contains(w), "{misuse:?}: expected {w:?} in: {e}");
        }
    }
}

/// The cluster copies lower to inline PTX (`mapa` for the peer forms, an `h` operand for
/// the mask) in the MLIR, and every installed `mlir-opt` (LLVM 21 and newer) compiles
/// them for sm_90 and sm_90a; `st.async` raises the PTX ISA to 8.1.
#[test]
fn cluster_copies_lower_to_inline_ptx() {
    let cases = [
        (
            feature_kernels::multicast_gemm_loads(2, 2).kernel,
            &[
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$3, $4}], [$2], $5;",
                "\"r,l,r,r,r,h,~{memory}\"",
            ][..],
            &["mbarrier::complete_tx::bytes.multicast::cluster"][..],
        ),
        (
            feature_kernels::bulk_multicast().kernel,
            &[
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1], $2, [$3], $4;",
            ][..],
            &["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"][..],
        ),
        (
            feature_kernels::s2s_ring().kernel,
            &[
                "mapa.shared::cluster.u64 rdst, $0, $4;",
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [rdst], [$1], $2, [rbar];",
            ][..],
            &["mapa.shared::cluster.u64", "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"][..],
        ),
        (
            feature_kernels::cluster_reduce_st_async().kernel,
            &["st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [raddr], {$3, $4, $5, $6}, [rbar];"][..],
            &["st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32", ".version 8."][..],
        ),
    ];
    for (kernel, mlir_patterns, _) in &cases {
        let mlir = kernel.to_mlir();
        for p in *mlir_patterns {
            assert!(mlir.contains(p), "{}: {p}\n{mlir}", kernel.name());
        }
    }
    for tc in installed_toolchains() {
        let who = tc.mlir_opt.display().to_string();
        for (kernel, _, ptx_patterns) in &cases {
            for arch in [Arch::Sm90, Arch::Sm90a] {
                let ptx = kernel.compile_ptx(&tc, arch).unwrap_or_else(|e| panic!("{who}: {}: {e}", kernel.name()));
                for p in *ptx_patterns {
                    assert!(ptx.contains(p), "{who}: {}: {p}\n{ptx}", kernel.name());
                }
                if kernel.name() == "cluster_reduce_st_async" {
                    assert!(!ptx.contains(".version 8.0"), "{who}: st.async needs PTX 8.1\n{ptx}");
                }
            }
        }
    }
}

/// The installed `mlir-opt`s of LLVM 21 and newer: the discovered one and every
/// Homebrew or `/workspace` LLVM, each once.
fn installed_toolchains() -> Vec<Toolchain> {
    let mut paths: Vec<std::path::PathBuf> = Toolchain::discover().into_iter().map(|t| t.mlir_opt).collect();
    for root in ["/opt/homebrew/opt", "/usr/local/opt", "/workspace"] {
        let Ok(entries) = std::fs::read_dir(root) else { continue };
        let mut dirs: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("llvm")))
            .collect();
        dirs.sort();
        paths.extend(dirs.into_iter().map(|d| d.join("bin").join("mlir-opt")));
    }
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for p in paths.into_iter().filter(|p| p.is_file()) {
        let canonical = std::fs::canonicalize(&p).unwrap_or(p.clone());
        if seen.contains(&canonical) {
            continue;
        }
        seen.push(canonical);
        let version = std::process::Command::new(&p).arg("--version").output().ok();
        let major = version.and_then(|o| {
            let text = String::from_utf8_lossy(&o.stdout).into_owned();
            let at = text.find("version ")? + "version ".len();
            text[at..].split('.').next()?.trim().parse::<u32>().ok()
        });
        if major.is_some_and(|m| m >= 21) {
            out.push(Toolchain::new(p));
        }
    }
    out
}
