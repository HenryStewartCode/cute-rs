//! TMA tile loads, runtime atom fields and `cute.copy`'s keyword arguments: the feature kernels under seeded schedules and random
//! completion, the simulator's reports of misuse, the build-time checks of
//! `CopyArgs`, `with_` and the MMA fields, and the MLIR the loads lower to.

mod feature_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;

/// Every TMA load kernel under the built-in schedules and 8 seeded ones with random
/// completion: each matches plain Rust, and the report counts its TMA loads.
#[test]
fn tma_loads_match_rust_under_seeded_schedules() {
    let seeds: Vec<u64> = (0..8).collect();
    for case in feature_kernels::tma_load_cases_sm90() {
        for small in [true, false] {
            if case.small_only && !small {
                continue;
            }
            let mut args = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            let report = case
                .kernel
                .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
            assert!(report.tma_instructions > 0 && report.tma_bytes > 0, "{}: {report}", case.name);
            let mut args = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            case.kernel
                .simulate_schedules(case.cfg, &mut args, &SimOptions::default(), &seeds)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
        }
    }
}

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// Build a kernel around `body` and expect it to panic with every one of `want`.
#[track_caller]
fn expect_build_panic(body: impl FnOnce() + 'static, want: &[&str]) {
    let msg = build_panic(move || {
        Kernel::builder("misuse").block_dim([128, 1, 1]).build(body);
    });
    for w in want {
        assert!(msg.contains(w), "expected {w:?} in: {msg}");
    }
}

/// The 128-byte-swizzled 64x64 f16 tile of the single-tile kernels.
fn sw128_tile() -> ComposedLayout {
    tile_to_shape_swizzled(
        &cute::warpgroup::make_smem_layout_atom::<F16>(cute::warpgroup::SmemLayoutAtomKind::K_SW128),
        (64, 64),
    )
}

/// The unswizzled (16-byte interleaved) 64x64 f16 tile.
fn inter_tile() -> ComposedLayout {
    tile_to_shape_swizzled(
        &cute::warpgroup::make_smem_layout_atom::<F16>(cute::warpgroup::SmemLayoutAtomKind::K_INTER),
        (64, 64),
    )
}

/// A `(64, 64)` K-major f16 parameter, its TMA atom for the 128-byte-swizzled tile,
/// and the tile's partitions over `s`.
fn tma_setup(s: &Tensor<F16>) -> (CopyAtom<F16>, Tensor<F16>, Tensor<Ident>) {
    tma_setup_for(s, sw128_tile())
}

/// [`tma_setup`] for the shared-memory layout `layout`.
fn tma_setup_for(s: &Tensor<F16>, layout: ComposedLayout) -> (CopyAtom<F16>, Tensor<F16>, Tensor<Ident>) {
    let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64), (64, 1)), 16);
    let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
    let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(g2s, &a, layout, (64, 64), 1).into();
    let g = local_tile(&tma_tensor, (64, 64), (0, 0));
    let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
    (atom, ts, tg)
}

/// What `cute.copy` refuses when the kernel is built: the TMA load's mbarrier, fields
/// an atom does not take, a field bound and given again, `cute.copy`'s own arguments
/// bound to an atom, the block API's multicast, data-source TMA loads,
/// a box start that is not 128-byte aligned, auxiliary operands, coordinates for a
/// data copy, a zero unroll factor and a predicated TMA load.
#[test]
fn copy_args_are_checked_when_the_kernel_is_built() {
    let smem = || make_smem_tensor_aligned::<F16>(sw128_tile(), 1024);
    expect_build_panic(
        move || {
            let (atom, ts, tg) = tma_setup(&smem());
            copy(&atom, &tg, &ts);
        },
        &["a TMA load", "needs `tma_bar_ptr`"],
    );
    expect_build_panic(
        || {
            let x = param_tensor::<f32>("x", Layout::col_major(4));
            let y = make_fragment::<f32>(4);
            let mbar = alloc_smem::<u64>(1, 8);
            let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
            copy_with(&atom, &x, &y, CopyArgs::new().tma_bar_ptr(&mbar));
        },
        &["takes no `tma_bar_ptr` argument", "a TMA load's field"],
    );
    expect_build_panic(
        move || {
            let (atom, ts, tg) = tma_setup(&smem());
            let mbar = alloc_smem::<u64>(1, 8);
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar).byte_mask(0x7fffi16));
        },
        &["takes no `byte_mask` argument", "CopyBulkS2GByteMaskOp"],
    );
    expect_build_panic(
        move || {
            let (atom, ..) = tma_setup(&smem());
            let p = make_fragment::<bool>(1);
            let p: &'static Tensor<bool> = Box::leak(Box::new(p));
            atom.with_(CopyArgs::new().pred(p));
        },
        &["pred and unroll_factor are arguments of one copy"],
    );
    expect_build_panic(
        move || {
            let (atom, ts, tg) = tma_setup(&smem());
            let mbar = alloc_smem::<u64>(1, 8);
            let bound = atom.with_(CopyArgs::new().tma_bar_ptr(&mbar));
            copy_with(&bound, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar));
        },
        &["`tma_bar_ptr` was bound to the atom by with_ and is given again"],
    );
    expect_build_panic(
        || {
            let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
            atom.with_(CopyArgs::new().cache_policy(0i64));
        },
        &["takes no `cache_policy` field"],
    );
    expect_build_panic(
        || {
            let _ = CopyArgs::new().tma_multicast((2, 1), 'M');
        },
        &["tma_multicast=", "not supported yet", "mcast_mask"],
    );
    expect_build_panic(
        move || {
            let s = smem();
            let (atom, ts, _) = tma_setup(&s);
            let a = param_tensor::<F16>("b", Layout::new((64, 64), (64, 1)));
            let mbar = alloc_smem::<u64>(1, 8);
            copy_with(&atom, &a.group_modes(0, 2), &ts, CopyArgs::new().tma_bar_ptr(&mbar));
        },
        &["the source is the TMA tensor's partition", "not a data tensor"],
    );
    expect_build_panic(
        || {
            let buf = make_smem_tensor_aligned::<F16>(Layout::col_major(2 * 4096), 1024);
            let s = make_tensor(&buf.ptr().offset(32), inter_tile());
            let (atom, ts, tg) = tma_setup_for(&s, inter_tile());
            let mbar = alloc_smem::<u64>(1, 8);
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar));
        },
        &["box 0 starts at byte 64", "not 128-byte aligned"],
    );
    // An atom for the 128-byte-swizzled tile with an unswizzled destination: the box
    // would land permuted.
    expect_build_panic(
        || {
            let s = make_smem_tensor_aligned::<F16>(inter_tile(), 1024);
            let (atom, ts, tg) = tma_setup(&s);
            let mbar = alloc_smem::<u64>(1, 8);
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar));
        },
        &["descriptor swizzles shared memory with s128b", "is swizzled by"],
    );
    expect_build_panic(
        move || {
            let (atom, ts, tg) = tma_setup(&smem());
            let idx = param_tensor::<i32>("idx", Layout::col_major(64));
            let mbar = alloc_smem::<u64>(1, 8);
            copy_aux(&atom, (&tg, &[AuxOperand::from(&idx)]), (&ts, &[]), CopyArgs::new().tma_bar_ptr(&mbar));
        },
        &["takes no auxiliary operands", "gather4/scatter4"],
    );
    expect_build_panic(
        || {
            let c = make_identity_tensor(4);
            let y = make_fragment::<f32>(4);
            copy(&make_copy_atom::<f32>(CopyOp::Universal { bits: 128 }), &c, &y);
        },
        &["the source is a coordinate tensor, which only TMA copies take"],
    );
    expect_build_panic(
        || {
            let x = param_tensor::<f32>("x", Layout::col_major((4, 4)));
            let y = param_tensor::<f32>("y", Layout::col_major((4, 4)));
            copy_with(
                &make_copy_atom::<f32>(CopyOp::Universal { bits: 128 }),
                &x,
                &y,
                CopyArgs::new().unroll_factor(0),
            );
        },
        &["unroll_factor=0", "unrolled 1 or more times"],
    );
    expect_build_panic(
        move || {
            let (atom, ts, tg) = tma_setup(&smem());
            let mbar = alloc_smem::<u64>(1, 8);
            let p = make_fragment::<bool>(1);
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar).pred(&p));
        },
        &["a TMA load is not predicated"],
    );
}

/// The ways a TMA load can go wrong at run time, each built into the single-tile
/// kernel.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Misuse {
    None,
    ExpectTooSmall,
    ReadBeforeWait,
    Misaligned,
    SwizzleMisaligned,
    PastAllocation,
    GenericWriteWithoutFence,
    DoubleIssue,
    NoInitFence,
    Uninitialised,
}

/// The single-tile load of a 64x64 f16 tile into 128-byte-swizzled shared memory,
/// with `misuse` built in. `off` (a launch value) offsets the tile in its buffer for
/// the misalignment cases.
fn misuse_kernel(misuse: Misuse) -> Kernel {
    Kernel::builder("tma_misuse").block_dim([128, 1, 1]).build(move || {
        let off = param_scalar::<i32>("off");
        let out = param_tensor::<F16>("out", Layout::new((64, 64), (64, 1)));
        let s = match misuse {
            Misuse::SwizzleMisaligned => {
                // The mbarrier first: the tile then starts at byte 128, 128-byte but not
                // 1024-byte aligned.
                let _first = alloc_smem::<u64>(2, 8);
                make_smem_tensor::<F16>(sw128_tile())
            }
            Misuse::Misaligned | Misuse::PastAllocation => {
                let buf = make_smem_tensor_aligned::<F16>(Layout::col_major(4096 + 256), 1024);
                make_tensor(&buf.ptr().offset(off), inter_tile())
            }
            _ => make_smem_tensor_aligned::<F16>(sw128_tile(), 1024),
        };
        let mbar = alloc_smem::<u64>(1, 8);
        let plain = matches!(misuse, Misuse::Misaligned | Misuse::PastAllocation);
        let (atom, ts, tg) = tma_setup_for(&s, if plain { inter_tile() } else { sw128_tile() });
        let t = thread_idx().x;
        if misuse != Misuse::Uninitialised {
            if_(t.eq(0), || mbarrier_init(&mbar, 1));
        }
        if misuse != Misuse::NoInitFence {
            mbarrier_init_fence();
        }
        if misuse == Misuse::GenericWriteWithoutFence {
            for_range(0, 32, 1, |j| s.store((t % 64, (t / 64) * 32 + j), F16::ZERO));
        }
        sync_threads();
        let issuers = if misuse == Misuse::DoubleIssue { 2 } else { 1 };
        let bytes = if misuse == Misuse::ExpectTooSmall { 4096 } else { 8192 };
        if misuse != Misuse::Uninitialised {
            if_(warp_idx().eq(0), || if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, bytes)));
        }
        if_(warp_idx().lt(issuers), || copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar)));
        if misuse == Misuse::ReadBeforeWait {
            out.store((t % 64, t / 64), s.load((t % 64, t / 64)));
        }
        if misuse == Misuse::Uninitialised {
            return;
        }
        mbarrier_wait(&mbar, 0);
        for_range(0, 32, 1, |j| {
            let (m, k) = (t % 64, (t / 64) * 32 + j);
            out.store((m, k), s.load((m, k)));
        });
    })
}

fn run_misuse(misuse: Misuse, off: i32) -> Result<Vec<F16>, String> {
    let kernel = misuse_kernel(misuse);
    let a: Vec<F16> = (0..64 * 64).map(|i| F16::from_f32((i % 13) as f32)).collect();
    let mut args = vec![KernelArg::ScalarI32(off), KernelArg::buffer(&vec![F16::ZERO; 64 * 64]), KernelArg::F16(a)];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).map_err(|e| e.to_string())?;
    Ok(args[1].to_vec::<F16>())
}

/// The misuse kernel without misuse loads the tile; each misuse is reported: an
/// `expect_tx` too small for the box, a read of the stage before the wait, a box start
/// that is not 128-byte aligned or not at the 128-byte swizzle's 1024-byte pattern, a
/// box running past its buffer, generic writes the TMA overwrites without a proxy
/// fence, a copy issued by two warps (one elected lane each, so the box lands twice:
/// an extra user-level election would not hide it), an mbarrier whose initialisation
/// was not published with `fence.mbarrier_init`, and one never initialised.
#[test]
fn tma_misuse_is_reported() {
    let out = run_misuse(Misuse::None, 0).unwrap();
    for (i, v) in out.iter().enumerate() {
        assert_eq!(v.to_f32(), (i % 13) as f32, "element {i}");
    }
    for (misuse, off, want) in [
        (
            Misuse::ExpectTooSmall,
            0,
            &["1 bulk copy of 8192 bytes", "transaction count is 4096", "expect_tx too small"][..],
        ),
        (Misuse::ReadBeforeWait, 0, &["reads smem0[", "while a bulk copy is still writing it"][..]),
        (Misuse::Misaligned, 32, &["shared address 0x40", "not 128-byte aligned"][..]),
        (Misuse::SwizzleMisaligned, 0, &["shared address 0x80", "1024-byte pattern"][..]),
        (Misuse::PastAllocation, 1024, &["runs past the end of `smem0`"][..]),
        (
            Misuse::GenericWriteWithoutFence,
            0,
            &["through the generic proxy, without a fence.proxy.async", "fence_view_async_shared"][..],
        ),
        (Misuse::DoubleIssue, 0, &["bulk cop"][..]),
        (Misuse::NoInitFence, 0, &["before its initialisation", "mbarrier_init_fence"][..]),
        (
            Misuse::Uninitialised,
            0,
            &["cp.async.bulk.tensor (a TMA load) reports to the mbarrier", "never initialised"][..],
        ),
    ] {
        let e = run_misuse(misuse, off).expect_err(&format!("{misuse:?} should be reported"));
        for w in want {
            assert!(e.contains(w), "{misuse:?}: expected {w:?} in: {e}");
        }
    }
}

/// `tma_desc_ptr` replaces the atom's descriptor: copying with `a`'s atom and the
/// address of `b`'s descriptor (the same box over another tensor) loads `b`'s tile; a
/// cache policy rides along (a hint the simulator ignores).
#[test]
fn tma_desc_ptr_overrides_the_descriptor() {
    let kernel = Kernel::builder("tma_desc_override").block_dim([128, 1, 1]).build(|| {
        let s = make_smem_tensor_aligned::<F16>(sw128_tile(), 1024);
        let (atom, ts, tg) = tma_setup(&s);
        let b = param_tensor_aligned::<F16>("b", Layout::new((64, 64), (64, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((64, 64), (64, 1)));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom_b, _) = cpasync::make_tiled_tma_atom(g2s, &b, sw128_tile(), (64, 64), 1).into();
        let desc_b = atom_b.tma_info().unwrap().descriptor().get_ptr();
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        if_(warp_idx().eq(0), || {
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, 8192));
            let args = CopyArgs::new().tma_bar_ptr(&mbar).tma_desc_ptr(desc_b).cache_policy(0i64);
            copy_with(&atom, &tg, &ts, args);
        });
        mbarrier_wait(&mbar, 0);
        for_range(0, 32, 1, |j| {
            let (m, k) = (t % 64, (t / 64) * 32 + j);
            out.store((m, k), s.load((m, k)));
        });
    });
    let a: Vec<F16> = (0..64 * 64).map(|i| F16::from_f32((i % 7) as f32)).collect();
    let b: Vec<F16> = (0..64 * 64).map(|i| F16::from_f32((i % 11) as f32 + 100.0)).collect();
    let mut args = vec![KernelArg::F16(a), KernelArg::F16(b.clone()), KernelArg::buffer(&vec![F16::ZERO; 64 * 64])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap();
    let out = args[2].to_vec::<F16>();
    for (i, (x, y)) in b.iter().zip(&out).enumerate() {
        assert_eq!(x.0, y.0, "element {i}");
    }
}

/// The double issue is reported under every schedule: as a copy issued twice while the
/// first is in flight, as more bytes than the transaction count, or as the second
/// overwriting the first's bytes without a wait.
#[test]
fn double_issue_is_reported_under_every_schedule() {
    let kernel = misuse_kernel(Misuse::DoubleIssue);
    let a: Vec<F16> = (0..64 * 64).map(|i| F16::from_f32((i % 13) as f32)).collect();
    let mut opts: Vec<SimOptions> =
        Schedule::ALL.iter().map(|&schedule| SimOptions { schedule, ..SimOptions::default() }).collect();
    opts.extend((0..8).map(|seed| SimOptions {
        schedule: Schedule::Random { seed },
        async_completion: AsyncCompletion::Random(seed),
        ..SimOptions::default()
    }));
    let mut issued_twice = 0;
    for o in opts {
        let mut args =
            vec![KernelArg::ScalarI32(0), KernelArg::buffer(&vec![F16::ZERO; 64 * 64]), KernelArg::F16(a.clone())];
        let e = kernel
            .simulate_with(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args, &o)
            .expect_err("reported")
            .to_string();
        assert!(
            e.contains("the copy was issued twice")
                || e.contains("transaction count is 8192")
                || e.contains("which a bulk copy wrote"),
            "{:?}: {e}",
            o.schedule
        );
        issued_twice += e.contains("the copy was issued twice") as usize;
    }
    assert!(issued_twice > 0, "some schedule issues both copies before either lands");
}

/// `copy_if` with one flag per application of the atom (CuTe DSL's `(ATOM_REST,
/// REST...)` profile): 4 applications of a 4-element atom, every other one off.
#[test]
fn copy_if_takes_one_flag_per_application() {
    let kernel = Kernel::builder("pred_per_application").block_dim([1, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(16));
        let y = param_tensor::<f32>("y", Layout::col_major(16));
        let p = make_fragment::<bool>(4);
        for a in 0..4 {
            p.store(a, Boolean::from(a % 2 == 0));
        }
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        copy_if(&atom, &p, &x, &y);
    });
    let mut args =
        vec![KernelArg::buffer(&(0..16).map(|i| i as f32).collect::<Vec<_>>()), KernelArg::buffer(&[-1.0f32; 16])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    let y = args[1].to_vec::<f32>();
    for (i, &v) in y.iter().enumerate() {
        let want = if (i / 4) % 2 == 0 { i as f32 } else { -1.0 };
        assert_eq!(v, want, "element {i}");
    }
}

/// `unroll_factor` loops over the rest mode of memory operands (here with a runtime
/// extent, which the fully unrolled copy refuses), and copies register fragments fully
/// unrolled, without a loop.
#[test]
fn unroll_factor_loops_over_memory_rest_modes() {
    let kernel = Kernel::builder("unrolled_copy").block_dim([1, 1, 1]).build(|| {
        let n = param_extent("n", 1);
        let x = param_tensor::<f32>("x", Layout::col_major((4, n)));
        let y = param_tensor::<f32>("y", Layout::col_major((4, n)));
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        copy_with(&atom, &x, &y, CopyArgs::new().unroll_factor(2));
    });
    assert!(kernel.to_mlir().contains("scf.for"), "a loop over the rest mode");
    let n = 7;
    let mut args = vec![
        KernelArg::ScalarI32(n),
        KernelArg::buffer(&(0..4 * n).map(|i| i as f32).collect::<Vec<_>>()),
        KernelArg::buffer(&vec![0.0f32; 4 * n as usize]),
    ];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    assert_eq!(args[2].to_vec::<f32>(), (0..4 * n).map(|i| i as f32).collect::<Vec<_>>());
    let fragment = Kernel::builder("unrolled_fragment").block_dim([1, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((4, 3)));
        let y = param_tensor::<f32>("y", Layout::col_major((4, 3)));
        let f = make_fragment::<f32>((4, 3));
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        copy_with(&atom, &x, &f, CopyArgs::new().unroll_factor(2));
        copy(&atom, &f, &y);
    });
    assert!(!fragment.to_mlir().contains("scf.for"), "a fragment copies without a loop");
    let mut args =
        vec![KernelArg::buffer(&(0..12).map(|i| i as f32).collect::<Vec<_>>()), KernelArg::buffer(&[0.0f32; 12])];
    fragment.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].to_vec::<f32>(), (0..12).map(|i| i as f32).collect::<Vec<_>>());
}

/// MMA runtime fields: set, get and with_ on an atom that takes `ACCUMULATE` (a WGMMA
/// atom), and a `TiledMma` carried through
/// `for_range_carry`, its field set inside the loop and read in the next iteration
/// and after it. An atom refuses the fields it does not take.
#[test]
fn mma_fields_are_loop_carried() {
    let kernel = Kernel::builder("mma_fields").block_dim([1, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(6));
        let atom = make_mma_atom(warpgroup::MmaF16BF16Op::new(
            DType::F16,
            DType::F32,
            (64, 8),
            warpgroup::OperandSource::SMEM,
            OperandMajorMode::K,
            OperandMajorMode::K,
        ));
        let mut mma = make_tiled_mma(atom, Layout::col_major((1, 1, 1)));
        assert_eq!(mma.fields(), &[MmaField::ACCUMULATE]);
        out.store(0, mma.get(MmaField::ACCUMULATE).to::<i32>());
        mma.set(MmaField::ACCUMULATE, false);
        let mma = for_range_carry(0, 4, 1, mma, |i, mut m| {
            out.store(i + 1, m.get(MmaField::ACCUMULATE).to::<i32>());
            m.set(MmaField::ACCUMULATE, true);
            m
        });
        out.store(
            5,
            mma.with_(MmaField::ACCUMULATE, false).get(MmaField::ACCUMULATE).to::<i32>()
                + mma.get(MmaField::ACCUMULATE).to::<i32>() * 10,
        );
    });
    let mut args = vec![KernelArg::buffer(&[-1i32; 6])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), vec![0, 0, 1, 1, 1, 10]);
    expect_build_panic(
        || {
            let mut mma = make_tiled_mma(make_mma_atom(MmaKind::M16N8K16F32F16), Layout::col_major((1, 1, 1)));
            mma.set(MmaField::ACCUMULATE, true);
        },
        &["Field.ACCUMULATE is not one of its fields", "it has no runtime fields"],
    );
    expect_build_panic(
        || {
            let atom = make_mma_atom(MmaKind::M16N8K16F32F16).__admit_fields_for_tests(&[MmaField::SFA]);
            let _ = atom.get(MmaField::SFA);
        },
        &["arrive with m10"],
    );
}

/// The MLIR of a TMA load: inline PTX with 32-bit shared addresses for the destination
/// and the mbarrier, the descriptor's 64-bit generic address, a register per
/// coordinate and a memory clobber; given a policy, the `.L2::cache_hint` form with one
/// more (64-bit) operand; three coordinates for a batched tensor. It compiles for sm_90a
/// with every installed `mlir-opt` (LLVM 21 and newer).
#[test]
fn tma_loads_lower_to_inline_ptx() {
    let tile = feature_kernels::tma_tile_k_sw128().kernel;
    let mlir = tile.to_mlir();
    let inst =
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4}], [$2];";
    assert!(mlir.contains(inst), "{mlir}");
    assert!(mlir.contains("\"r,l,r,r,r,~{memory}\""), "{mlir}");
    let batched = feature_kernels::tma_batched().kernel.to_mlir();
    assert!(batched.contains("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5}], [$2];"));
    let hinted = Kernel::builder("tma_cache_hint").block_dim([32, 1, 1]).build(|| {
        let s = make_smem_tensor_aligned::<F16>(sw128_tile(), 1024);
        let (atom, ts, tg) = tma_setup(&s);
        let mbar = alloc_smem::<u64>(1, 8);
        if_(thread_idx().x.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let policy = param_scalar::<i64>("policy");
        copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar).cache_policy(policy));
        mbarrier_wait(&mbar, 0);
    });
    let hinted_mlir = hinted.to_mlir();
    assert!(
        hinted_mlir.contains(".tile.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1, {$3, $4}], [$2], $5;"),
        "{hinted_mlir}"
    );
    assert!(hinted_mlir.contains("\"r,l,r,r,r,l,~{memory}\""), "{hinted_mlir}");
    for tc in installed_toolchains() {
        let who = tc.mlir_opt.display().to_string();
        for (name, k) in [("tile", &tile), ("hinted", &hinted)] {
            let ptx = tc.compile_ptx(&k.to_mlir(), Arch::Sm90a).unwrap_or_else(|e| panic!("{who}: {name}: {e}"));
            assert!(
                ptx.contains("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"),
                "{who}: {name}: {ptx}"
            );
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

/// A TMA load into another CTA's shared memory (a `map_dsmem_ptr` tensor) is
/// refused rather than landing in this CTA's.
#[test]
fn tma_load_refuses_a_dsmem_destination() {
    expect_build_panic(
        || {
            let s = make_smem_tensor_aligned::<F16>(inter_tile(), 1024);
            let d = make_tensor(map_dsmem_ptr(&s.ptr(), 1), inter_tile());
            let (atom, ts, tg) = tma_setup_for(&d, inter_tile());
            let mbar = alloc_smem::<u64>(1, 8);
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar));
        },
        &["writes shared memory", "dsmem"],
    );
}
