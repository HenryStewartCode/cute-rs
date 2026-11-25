//! TMA stores and reduce stores, L2 cache hints and prefetches (roadmap part
//! m7-tma-store): the feature kernels under seeded schedules and random completion, a
//! split-K reduction from CTAs outside a cluster with known inputs, the simulator's
//! reports of misuse, the build-time checks, and the MLIR and PTX the operations lower to.

mod feature_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::cpasync::{self, CacheHintSm90, ReductionKind};
use cute::prelude::*;
use cute::warpgroup::{SmemLayoutAtomKind, make_smem_layout_atom};

/// Every TMA store, reduction and prefetch kernel under the built-in schedules and 8
/// seeded ones with random completion: each matches plain Rust, and the report counts
/// its stores and prefetches.
#[test]
fn tma_stores_match_rust_under_seeded_schedules() {
    let seeds: Vec<u64> = (0..8).collect();
    for case in feature_kernels::tma_store_cases_sm90() {
        let mut args = feature_kernels::inputs_with_len(&case.kernel, true, 256);
        let report = case
            .kernel
            .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
        if case.name == "bulk_atom_cache_hints" {
            assert_eq!((report.prefetches, report.cache_hints), (2, 4), "{}: {report}", case.name);
        } else if case.name == "tma_prefetch_k_loop" {
            assert_eq!((report.prefetches, report.tensormap_prefetches), (4, 32), "{}: {report}", case.name);
        } else {
            assert!(report.tma_stores > 0 && report.tma_store_bytes > 0, "{}: {report}", case.name);
        }
        if case.name.starts_with("tma_store") {
            assert_eq!(report.cache_hints, report.tma_stores, "{}: {report}", case.name);
        }
        let mut args = feature_kernels::inputs_with_len(&case.kernel, true, 256);
        case.kernel
            .simulate_schedules(case.cfg, &mut args, &SimOptions::default(), &seeds)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
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

/// The 128-byte-swizzled 64x64 f16 tile, two stages.
fn staged_tile() -> ComposedLayout {
    tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64, 2))
}

/// What is refused when the kernel is built: a reduction kind on an element type PTX
/// does not take (naming the PTX rule), `cute.prefetch` with an atom that is neither a
/// TMA load nor a bulk copy, a TMA store's operands swapped, a predicate, a field the
/// store does not take, and the non-tensor bulk copies' missing fields.
#[test]
fn tma_store_misuse_is_refused_when_built() {
    for (kind, ty, rule) in [
        (ReductionKind::INC, "f32", ".inc and .dec take .u32"),
        (ReductionKind::MIN, "f32", ".min and .max take .f16, .bf16, .u32, .s32, .u64 and .s64"),
        (ReductionKind::XOR, "f16", ".and, .or and .xor take .b32 and .b64"),
    ] {
        let want = [format!("does not take {}", if ty == "f32" { "Float32" } else { "Float16" }), rule.to_string()];
        let msg = build_panic(move || {
            Kernel::builder("bad_reduce").block_dim([128, 1, 1]).build(move || {
                let op = cpasync::CopyReduceBulkTensorTileS2GOp { reduction_kind: kind };
                let smem = Layout::new((32, 32), (32, 1));
                if ty == "f32" {
                    let acc = param_tensor_aligned::<f32>("acc", Layout::new((32, 32), (32, 1)), 16);
                    let (atom, tma) = cpasync::make_tiled_tma_atom(op, &acc, smem.clone(), (32, 32), 1).into();
                    let s = make_smem_tensor_aligned::<f32>(smem, 128);
                    let (ts, tg) = cpasync::tma_partition(
                        &atom,
                        0,
                        &make_layout(1, 0),
                        &s.group_modes(0, 2),
                        &tma.group_modes(0, 2),
                    );
                    copy(&atom, &ts, &tg);
                } else {
                    let acc = param_tensor_aligned::<F16>("acc", Layout::new((32, 32), (32, 1)), 16);
                    let (atom, tma) = cpasync::make_tiled_tma_atom(op, &acc, smem.clone(), (32, 32), 1).into();
                    let s = make_smem_tensor_aligned::<F16>(smem, 128);
                    let (ts, tg) = cpasync::tma_partition(
                        &atom,
                        0,
                        &make_layout(1, 0),
                        &s.group_modes(0, 2),
                        &tma.group_modes(0, 2),
                    );
                    copy(&atom, &ts, &tg);
                }
            });
        });
        for w in &want {
            assert!(msg.contains(w.as_str()), "{kind:?} on {ty}: expected {w:?} in: {msg}");
        }
    }
    expect_build_panic(
        || {
            let x = param_tensor::<f32>("x", Layout::col_major(128));
            let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
            prefetch_atom(&atom, &x);
        },
        &["prefetch_atom: cute.prefetch takes TMA load atoms", "non-tensor bulk-copy atoms", "Universal"],
    );
    let store_setup = || {
        let out = param_tensor_aligned::<F16>("out", Layout::new((128, 128), (128, 1)), 16);
        let (atom, tma) =
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, &out, staged_tile(), (64, 64), 1).into();
        let s = make_smem_tensor_aligned::<F16>(staged_tile(), 1024);
        let g = local_tile(&tma, (64, 64), (0, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        (atom, ts, tg)
    };
    expect_build_panic(
        move || {
            let (atom, ts, tg) = store_setup();
            copy(&atom, &tg.slice((.., 0)), &ts.slice((.., 0)));
        },
        &["a TMA store reads shared memory: the source is the SMEM partition, not coordinates"],
    );
    expect_build_panic(
        move || {
            let (atom, ts, tg) = store_setup();
            let mbar = alloc_smem::<u64>(1, 8);
            copy_with(&atom, &ts.slice((.., 0)), &tg.slice((.., 0)), CopyArgs::new().tma_bar_ptr(&mbar));
        },
        &["takes no `tma_bar_ptr` argument", "it is a TMA load's field"],
    );
    expect_build_panic(
        || {
            let x = param_tensor_aligned::<f32>("x", Layout::col_major(256), 16);
            let s = make_smem_tensor_aligned::<f32>(Layout::col_major(256), 128);
            copy(&make_copy_atom::<f32>(cpasync::CopyBulkG2SOp), &x, &s);
        },
        &["needs `mbar_ptr`", "CopyArgs::new().mbar_ptr(&mbarrier)"],
    );
    expect_build_panic(
        || {
            let x = param_tensor_aligned::<f32>("x", Layout::col_major(256), 16);
            let s = make_smem_tensor_aligned::<f32>(Layout::col_major(256), 128);
            let mbar = alloc_smem::<u64>(1, 8);
            copy_with(
                &make_copy_atom::<f32>(cpasync::CopyBulkG2SOp),
                &x,
                &s,
                CopyArgs::new().mbar_ptr(&mbar).cta_rank(1),
            );
        },
        &["takes no `cta_rank` argument"],
    );
}

/// The ways the store epilogue can go wrong at run time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Misuse {
    None,
    /// Rewrites a stage without `cp_async_bulk_wait_group(1, true)` first.
    ReuseBeforeRead,
    /// No `fence_view_async_shared` between the plain stores and the TMA store.
    NoProxyFence,
    /// Reads the stored output before the full wait.
    ReadBeforeWait,
}

/// Three 64x64 tiles of a `(64, 192)` f16 output through two stages, as the feature
/// kernel's epilogue, with `misuse` built in.
fn epilogue_misuse(misuse: Misuse) -> Kernel {
    Kernel::builder("tma_store_misuse").block_dim([128, 1, 1]).build(move || {
        let out = param_tensor_aligned::<F16>("out", Layout::new((64, 192), (192, 1)), 16);
        let peek = param_tensor::<F16>("peek", Layout::col_major(128));
        let (atom, tma) =
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, &out, staged_tile(), (64, 64), 1).into();
        let s = make_smem_tensor_aligned::<F16>(staged_tile(), 1024);
        let g = local_tile(&tma, (64, 64), (0, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let t = thread_idx().x;
        for j in 0..3usize {
            let st = j % 2;
            for_range(0, 32, 1, |k| {
                let idx = k * 128 + t;
                s.store((idx / 64, idx % 64, st as i32), (idx % 7).to::<f32>().to::<F16>());
            });
            if misuse != Misuse::NoProxyFence {
                fence_view_async_shared();
            }
            cute::pipeline::sync(1);
            if_(warp_idx().eq(0), || {
                copy(&atom, &ts.slice((.., st)), &tg.slice((.., j)));
                cp_async_bulk_commit_group();
                if misuse != Misuse::ReuseBeforeRead {
                    cp_async_bulk_wait_group(1, true);
                }
            });
            cute::pipeline::sync(1);
        }
        if misuse == Misuse::ReadBeforeWait {
            peek.store(t, out.load((0, t)));
        }
        if_(warp_idx().eq(0), || cp_async_bulk_wait_group(0, false));
    })
}

fn run_epilogue_misuse(misuse: Misuse) -> Result<Vec<F16>, String> {
    let kernel = epilogue_misuse(misuse);
    let mut args = vec![KernelArg::buffer(&vec![F16::ZERO; 64 * 192]), KernelArg::buffer(&vec![F16::ZERO; 128])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).map_err(|e| e.to_string())?;
    Ok(args[0].to_vec::<F16>())
}

/// The epilogue without misuse stores its tiles; each misuse is reported: a stage
/// rewritten while the store of it still reads it (no `.read` wait), generic writes the
/// TMA store reads without a proxy fence, and a read of the stored region before the
/// full wait.
#[test]
fn tma_store_misuse_is_reported() {
    let out = run_epilogue_misuse(Misuse::None).unwrap();
    for m in 0..64 {
        for n in 0..192 {
            let idx = m * 64 + n % 64;
            assert_eq!(out[m * 192 + n].to_f32(), (idx % 7) as f32, "element ({m}, {n})");
        }
    }
    for (misuse, want) in [
        (Misuse::ReuseBeforeRead, &["data race", "which an asynchronous operation issued by", "still reads"][..]),
        (
            Misuse::NoProxyFence,
            &["a TMA store) reads shared memory through the async proxy", "without a fence.proxy.async"][..],
        ),
        (
            Misuse::ReadBeforeWait,
            &["reads element", "of `out`", "is still writing", "cp_async_bulk_wait_group(n, false)"][..],
        ),
    ] {
        let e = run_epilogue_misuse(misuse).expect_err(&format!("{misuse:?} should be reported"));
        for w in want {
            assert!(e.contains(w), "{misuse:?}: expected {w:?} in: {e}");
        }
    }
}

/// Split-K: four CTAs outside any cluster reduce-add their `(32, 32)` f32 partials into
/// an accumulator with known contents, which ends up holding their sum.
#[test]
fn split_k_reduce_add_from_several_ctas() {
    let kernel = Kernel::builder("split_k").block_dim([128, 1, 1]).build(|| {
        let acc = param_tensor_aligned::<f32>("acc", Layout::new((32, 32), (32, 1)), 16);
        let smem = Layout::new((32, 32), (32, 1));
        let op = cpasync::CopyReduceBulkTensorTileS2GOp { reduction_kind: ReductionKind::ADD };
        let (atom, tma) = cpasync::make_tiled_tma_atom(op, &acc, smem.clone(), (32, 32), 1).into();
        let s = make_smem_tensor_aligned::<f32>(smem, 128);
        let (ts, tg) =
            cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &tma.group_modes(0, 2));
        let (t, k) = (thread_idx().x, block_idx().x);
        for_range(0, 8, 1, |j| {
            let idx = j * 128 + t;
            s.store((idx / 32, idx % 32), ((idx + k * 3) % 11).to::<f32>() * 0.5f32);
        });
        fence_view_async_shared();
        sync_threads();
        if_(warp_idx().eq(0), || {
            copy_with(&atom, &ts, &tg, CopyArgs::new().cache_policy(CacheHintSm90::EVICT_NORMAL));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, true);
        });
    });
    let init: Vec<f32> = (0..1024).map(|i| (i % 5) as f32).collect();
    let mut args = vec![KernelArg::buffer(&init)];
    let report = kernel.simulate(LaunchConfig::new([4, 1, 1], [128, 1, 1]), &mut args).unwrap();
    assert_eq!((report.tma_stores, report.cache_hints), (4, 4), "{report}");
    let got = args[0].to_vec::<f32>();
    for i in 0..1024 {
        let want = init[i] + (0..4).map(|k| ((i + k * 3) % 11) as f32 * 0.5).sum::<f32>();
        assert_eq!(got[i], want, "element {i}");
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

/// TMA stores and reductions, the prefetches and the cache-hinted bulk copies lower to
/// inline PTX: in the MLIR, and in the PTX of every installed `mlir-opt` (LLVM 21 and
/// newer) for sm_90a.
#[test]
fn tma_stores_lower_to_inline_ptx() {
    let cases = [
        (
            feature_kernels::tma_store_epilogue().kernel,
            &[
                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint [$1, {$2, $3}], [$0], $4;",
                "fence.proxy.async.shared::cta",
            ][..],
            &[
                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint",
                "cp.async.bulk.commit_group",
            ][..],
        ),
        (
            feature_kernels::tma_reduce_cases().remove(0).kernel,
            &["cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [$1, {$2, $3}], [$0];"][..],
            &["cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"][..],
        ),
        (
            feature_kernels::tma_prefetch_k_loop().kernel,
            &["cp.async.bulk.prefetch.tensor.2d.L2.global.tile [$0, {$1, $2}];", "prefetch.tensormap [$0];"][..],
            &["cp.async.bulk.prefetch.tensor.2d.L2.global.tile", "prefetch.tensormap"][..],
        ),
        (
            feature_kernels::bulk_atom_cache_hints().kernel,
            &[
                "cp.async.bulk.prefetch.L2.global [$0], $1;",
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;",
                "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [$0], [$1], $2, $3;",
            ][..],
            &[
                "cp.async.bulk.prefetch.L2.global",
                "mbarrier::complete_tx::bytes.L2::cache_hint",
                "bulk_group.L2::cache_hint",
            ][..],
        ),
    ];
    let reductions: Vec<String> = feature_kernels::tma_reduce_cases().into_iter().map(|c| c.kernel.to_mlir()).collect();
    for op in ["add", "min", "max", "inc", "dec", "xor"] {
        let pattern = format!("global.shared::cta.{op}.tile.bulk_group");
        assert!(reductions.iter().any(|m| m.contains(&pattern)), "{pattern}");
    }
    for (kernel, mlir_patterns, _) in &cases {
        let mlir = kernel.to_mlir();
        for p in *mlir_patterns {
            assert!(mlir.contains(p), "{}: {p}\n{mlir}", kernel.name());
        }
    }
    for tc in installed_toolchains() {
        let who = tc.mlir_opt.display().to_string();
        for (kernel, _, ptx_patterns) in &cases {
            let ptx = tc
                .compile_ptx(&kernel.to_mlir(), Arch::Sm90a)
                .unwrap_or_else(|e| panic!("{who}: {}: {e}", kernel.name()));
            for p in *ptx_patterns {
                assert!(ptx.contains(p), "{who}: {}: {p}\n{ptx}", kernel.name());
            }
        }
    }
}
