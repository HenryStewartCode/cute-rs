// SPDX-License-Identifier: MIT OR Apache-2.0

//! The warpgroup MMA variants: FP8, 8-bit integer and
//! register-sourced A MMAs in the simulator, `make_trivial_tiled_mma`, diagnostics and
//! the MLIR and PTX.

mod feature_kernels;

use cute::prelude::*;
use feature_kernels::{inputs_with_len, wgmma_variant_cases};

/// Every variant feature kernel in the simulator, under every built-in schedule, on small
/// inputs (and the integer ones on random bytes too).
#[test]
fn variant_kernels_match_rust() {
    for case in wgmma_variant_cases() {
        for small in [true, false] {
            if !small && case.small_only {
                continue;
            }
            let mut args = inputs_with_len(&case.kernel, small, 256);
            case.kernel
                .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
        }
    }
}

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::utils::sm90::{make_trivial_tiled_mma, sm90_make_trivial_tiled_mma};
use cute::warpgroup::{self, MmaF8Op, MmaI8Op, OperandSource, SmemLayoutAtomKind};
use feature_kernels::operand_layout;

#[track_caller]
fn expect_panic(f: impl FnOnce(), want: &[&str]) {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    for w in want {
        assert!(msg.contains(w), "expected {w:?} in: {msg}");
    }
}

/// `(m, n)`-colexicographic index of `CuTe`'s layout `shape : stride` at the
/// flat coordinates `coord` (the layouts of `include/cute/atom/mma_traits_sm90_gmma.hpp`,
/// evaluated by hand here, apart from cute-rs's layout algebra).
fn dot(coord: &[usize], stride: &[usize]) -> usize {
    coord.iter().zip(stride).map(|(c, s)| c * s).sum()
}

/// The `(thread, value)` of a tiled MMA's TV layout, flattened as `t + threads * v`,
/// and the index it should give: `atom(t % 128, v)` (an index into the atom's `(MN, K)`
/// or `(M, N)` tile) moved to warpgroup `t / 128`'s place in the tile (stacked along M
/// for A and C; B is shared).
fn check_tv(
    tv: &Layout,
    threads: usize,
    values: usize,
    atom: impl Fn(usize, usize) -> usize,
    atom_rows: usize,
    tile_rows: usize,
    stacked: bool,
    what: &str,
) {
    assert_eq!(tv.size(), threads * values, "{what}: {tv}");
    for t in 0..threads {
        for v in 0..values {
            let i = atom(t % 128, v);
            let (r, c) = (i % atom_rows, i / atom_rows);
            let r = if stacked { r + atom_rows * (t / 128) } else { r };
            assert_eq!(tv.call(t + threads * v), r + tile_rows * c, "{what}: thread {t}, value {v}");
        }
    }
}

/// `make_trivial_tiled_mma` for the configurations CuTe DSL's Hopper `dense_gemm`
/// builds (tiler `(64, N)` for CTA tiles of N 64, 128 and 256; two warpgroups along M
/// for the 128 x 256 tile) and its attention kernel's `P V` (A in registers): the
/// operation it picks and its A, B and C TV layouts against CuTe's GMMA layouts
/// (`ABLayout`, `ALayout_64x16`, `ALayout_64x32`, `CLayout_64xN`).
#[test]
fn trivial_tiled_mma_matches_the_gmma_layouts() {
    use DType::*;
    let (k, smem, rmem) = (OperandMajorMode::K, OperandSource::SMEM, OperandSource::RMEM);
    let configs = [
        (F16, F16, F32, (1, 1, 1), 64, smem, "m64n64k16.f32.f16.f16"),
        (BF16, BF16, F32, (1, 1, 1), 128, smem, "m64n128k16.f32.bf16.bf16"),
        (F16, F16, F16, (1, 1, 1), 256, smem, "m64n256k16.f16.f16.f16"),
        (F16, F16, F32, (2, 1, 1), 256, smem, "m64n256k16.f32.f16.f16"),
        (F8E4M3FN, F8E5M2, F32, (2, 1, 1), 256, smem, "m64n256k32.f32.e4m3.e5m2"),
        (F8E5M2, F8E5M2, F16, (1, 1, 1), 128, smem, "m64n128k32.f16.e5m2.e5m2"),
        (I8, U8, I32, (2, 1, 1), 128, smem, "m64n128k32.s32.s8.u8"),
        (U8, U8, I32, (1, 1, 1), 64, smem, "m64n64k32.s32.u8.u8"),
        (F16, F16, F32, (1, 1, 1), 64, rmem, "m64n64k16.f32.f16.f16"),
        (F8E4M3FN, F8E4M3FN, F32, (2, 1, 1), 128, rmem, "m64n128k32.f32.e4m3.e4m3"),
        (I8, I8, I32, (1, 1, 1), 32, rmem, "m64n32k32.s32.s8.s8"),
    ];
    for (a, b, acc, atoms, n, src, ptx) in configs {
        let mma = make_trivial_tiled_mma(a, b, k, k, acc, atoms, (64, n), src);
        let alias = sm90_make_trivial_tiled_mma(a, b, k, k, acc, atoms, (64, n), src);
        let what = format!("{ptx} {atoms:?} {src:?}");
        let MmaOp::Sm90(kind) = mma.atom().op() else { panic!("{what}: not a warpgroup MMA") };
        assert_eq!(alias.atom().op(), mma.atom().op());
        assert_eq!(kind.ptx(), format!("wgmma.mma_async.sync.aligned.{ptx}"), "{what}");
        assert_eq!(kind.a_regs, src == rmem, "{what}");
        let kk = 256 / a.bits();
        let w = atoms.0;
        let threads = 128 * w;
        assert_eq!(mma.atom().shape_mnk(), [64, n, kk], "{what}");
        assert_eq!((mma.num_threads(), mma.tile_size(0), mma.tile_size(1), mma.tile_size(2)), (threads, 64 * w, n, kk));
        // A: every thread sees its warpgroup's 64 x K (ABLayout), or holds a register
        // fragment (ALayout_64x16 or ALayout_64x32).
        let lane = |t: usize| [t % 4, (t / 4) % 8, t / 32];
        if src == smem {
            check_tv(&mma.layout_a_tv(), threads, 64 * kk, |_, v| v, 64, 64 * w, true, &format!("{what} A"));
        } else {
            let (vals, tstride, vstride): (Vec<usize>, [usize; 3], [usize; 3]) = if kk == 16 {
                (vec![2, 2, 2], [128, 1, 16], [64, 8, 512])
            } else {
                (vec![4, 2, 2], [256, 1, 16], [64, 8, 1024])
            };
            let a_atom = |t: usize, v: usize| {
                let vc = [v % vals[0], (v / vals[0]) % vals[1], v / (vals[0] * vals[1])];
                dot(&lane(t), &tstride) + dot(&vc, &vstride)
            };
            check_tv(&mma.layout_a_tv(), threads, 64 * kk / 128, a_atom, 64, 64 * w, true, &format!("{what} A"));
        }
        // B: every thread sees all of N x K.
        check_tv(&mma.layout_b_tv(), threads, n * kk, |_, v| v, n, n, false, &format!("{what} B"));
        // C: CLayout_64xN = ((4, 8, 4), (2, 2, N / 8)) : ((128, 1, 16), (64, 8, 512)).
        let c_atom =
            |t: usize, v: usize| dot(&lane(t), &[128, 1, 16]) + dot(&[v % 2, (v / 2) % 2, v / 4], &[64, 8, 512]);
        check_tv(&mma.layout_c_tv(), threads, n / 2, c_atom, 64, 64 * w, true, &format!("{what} C"));
    }
}

/// The operations' parameter checks, and `make_trivial_tiled_mma`'s type dispatch.
#[test]
fn parameters_are_checked() {
    use DType::*;
    let (k, mn, smem, rmem) = (OperandMajorMode::K, OperandMajorMode::MN, OperandSource::SMEM, OperandSource::RMEM);
    // FP8 and INT8 operands are K-major.
    expect_panic(
        || {
            let _ = MmaF8Op::new(F8E4M3FN, F8E4M3FN, F32, (64, 64), smem, mn, k);
        },
        &["'a_major_mode' is MN", "K-major"],
    );
    expect_panic(
        || {
            let _ = MmaF8Op::new(F8E4M3FN, F8E5M2, F16, (64, 64), smem, k, mn);
        },
        &["'b_major_mode' is MN"],
    );
    expect_panic(
        || {
            let _ = MmaI8Op::new(I8, U8, I32, (64, 64), smem, mn, k);
        },
        &["MmaI8Op", "'a_major_mode' is MN"],
    );
    // Types, shapes.
    expect_panic(
        || {
            let _ = MmaF8Op::new(F16, F8E4M3FN, F32, (64, 64), smem, k, k);
        },
        &["'a_dtype' is f16"],
    );
    expect_panic(
        || {
            let _ = MmaF8Op::new(F8E4M3FN, F8E4M3FN, BF16, (64, 64), smem, k, k);
        },
        &["'acc_dtype' is bf16"],
    );
    expect_panic(
        || {
            let _ = MmaF8Op::new(F8E4M3FN, F8E4M3FN, F32, (64, 64, 16), smem, k, k);
        },
        &["K is 16", "K = 32"],
    );
    expect_panic(
        || {
            let _ = MmaI8Op::new(I8, I16, I32, (64, 64), smem, k, k);
        },
        &["'b_dtype' is i16"],
    );
    expect_panic(
        || {
            let _ = MmaI8Op::new(I8, I8, F32, (64, 64), smem, k, k);
        },
        &["'acc_dtype' is f32", "Int32"],
    );
    // N: 8, 24 or a multiple of 16 for INT8.
    for bad in [40, 56, 72, 248] {
        expect_panic(
            || {
                let _ = MmaI8Op::new(U8, U8, I32, (64, bad), smem, k, k);
            },
            &[&format!("N is {bad}"), "8, 24"],
        );
    }
    for good in [8, 16, 24, 32, 48, 240, 256] {
        assert_eq!(MmaI8Op::new(U8, I8, I32, (64, good), smem, k, k).shape_mnk(), (64, good, 32));
    }
    expect_panic(
        || {
            let _ = MmaF8Op::new(F8E5M2, F8E5M2, F32, (64, 12), smem, k, k);
        },
        &["N is 12"],
    );
    // A in registers is K-major.
    expect_panic(
        || {
            let _ = warpgroup::MmaF16BF16Op::new(BF16, F32, (64, 64), rmem, mn, k);
        },
        &["OperandSource::RMEM", "'a_major_mode' MN"],
    );
    // make_trivial_tiled_mma: pairs no warpgroup MMA takes.
    for (a, b) in [(F16, BF16), (F16, F8E4M3FN), (F32, F32), (F8E4M3FN, I8), (I8, F16), (TF32, TF32)] {
        expect_panic(
            || {
                let _ = make_trivial_tiled_mma(a, b, k, k, F32, (1, 1, 1), (64, 64), smem);
            },
            &["TypeError"],
        );
    }
}

/// The kernels need sm_90a; the mixed-sign integer MMAs PTX ISA 8.4, the others 8.0.
#[test]
fn targets() {
    for case in wgmma_variant_cases() {
        for arch in [Arch::Sm90, Arch::Sm100a] {
            let e = cute::validate::check_target(case.kernel.ir(), arch).unwrap_err().to_string();
            assert!(e.contains("wgmma") && e.contains("sm_90a"), "{} on {arch}: {e}", case.name);
        }
        let ptx = cute::validate::check_target(case.kernel.ir(), Arch::Sm90a).unwrap();
        let mixed = ["s8_u8", "u8_s8"].iter().any(|m| case.name.contains(m));
        assert_eq!(ptx, if mixed { 84 } else { 80 }, "{}", case.name);
    }
}

fn build(name: &str, body: impl FnOnce() + 'static) -> Kernel {
    Kernel::builder(name).block_dim([128, 1, 1]).build(body)
}

/// A register-sourced `64 x 8 x 16` MMA whose A comes from `ldmatrix`; `after_fence`
/// runs between the fence and the MMA.
fn rs_small(after_fence: impl FnOnce(&Tensor<F16>) + 'static) -> Kernel {
    build("rs_small", move || {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, 16), (16, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((8, 16), (16, 1)), 16);
        let c = param_tensor::<f32>("c", Layout::new((64, 8), (8, 1)));
        let k = OperandMajorMode::K;
        let mma =
            make_trivial_tiled_mma(DType::F16, DType::F16, k, k, DType::F32, (1, 1, 1), (64, 8), OperandSource::RMEM);
        let sa = make_smem_tensor_aligned::<F16>(Layout::new((64, 16), (16, 1)), 128);
        let sb = make_smem_tensor_aligned::<F16>(operand_layout::<F16>(SmemLayoutAtomKind::K_INTER, 8, 16, 1), 128);
        let t = thread_idx().x;
        for_range(0, 8, 1, |j| {
            let i = j * 128 + t;
            sa.store((i % 64, i / 64), a.load((i % 64, i / 64)));
        });
        sb.store((t % 8, t / 8), b.load((t % 8, t / 8)));
        fence_view_async_shared();
        sync_threads();
        let thr = mma.get_slice(t);
        let fa = thr.partition_fragment_a::<F16, _>(&sa);
        let ldsm = make_tiled_copy_a(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
        copy(&ldsm, &ldsm.get_slice(t).partition_s(&sa), &ldsm.retile(&fa));
        let tcrb = mma.make_fragment_b(&thr.partition_b(&sb));
        let tcgc = thr.partition_c(&c);
        let acc = mma.make_fragment_c::<f32>(&tcgc);
        warpgroup::fence();
        after_fence(&fa);
        gemm(&mma, &acc, &fa, &tcrb, &acc);
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        for i in 0..acc.size() {
            tcgc.store(i, acc.load(i));
        }
    })
}

/// The A fragment written (or read) after the fence and before the MMA is reported;
/// the well-formed kernel runs. The A registers go through the fence's and the wait's
/// operand fences.
#[test]
fn register_a_diagnostics() {
    let good = rs_small(|_| {});
    let mut args = inputs_with_len(&good, true, 256);
    good.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap();
    let (a, b, c) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<f32>());
    for (i, &got) in c.iter().enumerate() {
        let (m, n) = (i / 8, i % 8);
        let want: f32 = (0..16).map(|kk| a[m * 16 + kk].to_f32() * b[n * 16 + kk].to_f32()).sum();
        assert_eq!(got, want, "c[{m}][{n}]");
    }
    let fences: Vec<(usize, usize)> = good
        .ir()
        .body
        .insts
        .iter()
        .filter_map(|i| match &i.op {
            cute::ir::Op::WgmmaFence { inputs, .. } | cute::ir::Op::WgmmaWait { inputs, .. } => {
                Some((inputs.len(), inputs.iter().filter(|&&v| good.ir().dtype(v) == DType::F16).count()))
            }
            _ => None,
        })
        .collect();
    // The fence: 4 accumulators and the 8 A registers; the wait: 4 and the 8 A registers.
    assert_eq!(fences, vec![(12, 8), (12, 8)]);
    let bad = rs_small(|fa| fa.store(0, fa.load(1) + fa.load(2)));
    let mut args = inputs_with_len(&bad, true, 256);
    let e = bad.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap_err().to_string();
    assert!(e.contains("A fragment registers") && e.contains("after the warp's last warpgroup::fence"), "{e}");
}

/// The MLIR of the variants: the 8-bit forms' instruction names and immediates (none
/// for integers), the register form's four A registers, packed from the fragment, and
/// the A registers in the operand fences.
#[test]
fn mlir_of_the_variants() {
    let cases = wgmma_variant_cases();
    let mlir = |name: &str| cases.iter().find(|c| c.name == name).unwrap().kernel.to_mlir();
    let f8 = mlir("wgmma_f8_e4m3_e5m2_f32_n128");
    let list = (0..64).map(|i| format!("${i}")).collect::<Vec<_>>().join(", ");
    let want = format!("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e5m2 {{{list}}}, $64, $65, p, 1, 1;");
    assert!(f8.contains(&want), "{want} missing:\n{f8}");
    let i8 = mlir("wgmma_i8_s8_s8_n8");
    assert!(i8.contains("wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 {$0, $1, $2, $3}, $4, $5, p;"), "{i8}");
    assert!(i8.contains("\"=r,=r,=r,=r,l,l,r,0,1,2,3\""), "{i8}");
    let f16acc = mlir("wgmma_f8_e5m2_e4m3_f16_n96");
    assert!(f16acc.contains("m64n96k32.f16.e5m2.e4m3 {$0, "), "{f16acc}");
    let rs = mlir("wgmma_rs_ldmatrix_f16_n64");
    let list = (0..32).map(|i| format!("${i}")).collect::<Vec<_>>().join(", ");
    let want = format!(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {{{list}}}, {{$32, $33, $34, $35}}, $36, p, 1, 1, 0;"
    );
    assert!(rs.contains(&want), "{want} missing:\n{rs}");
    assert!(rs.contains("=f,r,r,r,r,l,r,0,"), "{rs}");
    let rs8 = mlir("wgmma_rs_ldmatrix_u8_s8_n16");
    assert!(rs8.contains("m64n16k32.s32.u8.s8 {$0, $1, $2, $3, $4, $5, $6, $7}, {$8, $9, $10, $11}, $12, p;"), "{rs8}");
    assert!(rs8.contains("vector<4xi8> to i32"), "{rs8}");
}

/// The PTX of every variant with each installed `mlir-opt`, and, when `ptxas` is
/// installed, its assembly for sm_90a without the warning that ptxas serialised the MMAs.
#[test]
fn ptx_and_ptxas_for_sm_90a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for case in wgmma_variant_cases() {
        let ptx = case.kernel.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert!(ptx.contains(".target sm_90a") && ptx.contains("wgmma.mma_async.sync.aligned"), "{}", case.name);
        let mixed = ["s8_u8", "u8_s8"].iter().any(|m| case.name.contains(m));
        assert!(ptx.contains(if mixed { ".version 8.4" } else { ".version 8.0" }), "{}: {ptx}", case.name);
        let Some(ptxas) = find_ptxas() else { continue };
        let path = std::env::temp_dir().join(format!("cute-wgmma-variants-{}-{}.ptx", std::process::id(), case.name));
        std::fs::write(&path, &ptx).unwrap();
        let out = std::process::Command::new(&ptxas)
            .args(["-arch=sm_90a", "-v", "-o", "/dev/null"])
            .arg(&path)
            .output()
            .expect("running ptxas");
        let _ = std::fs::remove_file(&path);
        let log = String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{}: ptxas failed:\n{log}", case.name);
        assert!(!log.contains("serialized"), "{}: ptxas serialised the warpgroup MMAs:\n{log}", case.name);
        eprintln!("{}: {}", case.name, log.lines().filter(|l| l.contains("registers")).collect::<Vec<_>>().join(" "));
    }
}

/// `ptxas` on `PATH`.
fn find_ptxas() -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("ptxas")).find(|p| p.is_file())
}

/// Accumulators a loop carries before any MMA has written them (a main loop with no
/// MMAs issued ahead of it, `k_pipe_mmas = 0`) start as zeros instead of being refused.
#[test]
fn unwritten_accumulators_can_be_carried() {
    let case = feature_kernels::wgmma_pipelined(2, 0);
    let mut args = inputs_with_len(&case.kernel, true, 256);
    case.kernel.simulate(case.cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    (case.check)(&args);
}

/// A block whose last warpgroup is partial may issue warpgroup MMAs from its whole
/// warpgroups (the PTX ISA needs the issuing warpgroup complete); one issued by the partial warpgroup is reported.
#[test]
fn partial_last_warpgroup() {
    let kernel = |issue_all: bool| {
        Kernel::builder("partial").block_dim([160, 1, 1]).build(move || {
            let t = thread_idx().x;
            let body = move || {
                let a = param_tensor_aligned::<F16>("a", Layout::new((64, 16), (16, 1)), 16);
                let b = param_tensor_aligned::<F16>("b", Layout::new((8, 16), (16, 1)), 16);
                let c = param_tensor::<f32>("c", Layout::new((64, 8), (8, 1)));
                let k = OperandMajorMode::K;
                let mma = make_trivial_tiled_mma(
                    DType::F16,
                    DType::F16,
                    k,
                    k,
                    DType::F32,
                    (1, 1, 1),
                    (64, 8),
                    OperandSource::SMEM,
                );
                let sa =
                    make_smem_tensor_aligned::<F16>(operand_layout::<F16>(SmemLayoutAtomKind::K_INTER, 64, 16, 1), 128);
                let sb =
                    make_smem_tensor_aligned::<F16>(operand_layout::<F16>(SmemLayoutAtomKind::K_INTER, 8, 16, 1), 128);
                (a, b, c, mma, sa, sb)
            };
            let (a, b, c, mma, sa, sb) = body();
            if_(t.lt(128), || {
                for_range(0, 8, 1, |j| {
                    let i = j * 128 + t;
                    sa.store((i % 64, i / 64), a.load((i % 64, i / 64)));
                });
                sb.store((t % 8, t / 8), b.load((t % 8, t / 8)));
            });
            fence_view_async_shared();
            sync_threads();
            let work = || {
                let thr = mma.get_slice(t % 128);
                let (fa, fb) = (mma.make_fragment_a(&thr.partition_a(&sa)), mma.make_fragment_b(&thr.partition_b(&sb)));
                let tcgc = thr.partition_c(&c);
                let acc = mma.make_fragment_c::<f32>(&tcgc);
                warpgroup::fence();
                gemm(&mma, &acc, &fa, &fb, &acc);
                warpgroup::commit_group();
                warpgroup::wait_group(0);
                for i in 0..acc.size() {
                    tcgc.store(i, acc.load(i));
                }
            };
            if issue_all { work() } else { if_(t.lt(128), work) }
        })
    };
    let cfg = LaunchConfig::new([1, 1, 1], [160, 1, 1]);
    let good = kernel(false);
    let mut args = inputs_with_len(&good, true, 256);
    good.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    let (a, b, c) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<f32>());
    for (i, &got) in c.iter().enumerate() {
        let want: f32 = (0..16).map(|kk| a[(i / 8) * 16 + kk].to_f32() * b[(i % 8) * 16 + kk].to_f32()).sum();
        assert_eq!(got, want, "c[{i}]");
    }
    let bad = kernel(true);
    let mut args = inputs_with_len(&bad, true, 256);
    let e = bad.simulate(cfg, &mut args).unwrap_err().to_string();
    assert!(e.contains("warpgroup 1 is not complete"), "{e}");
}
