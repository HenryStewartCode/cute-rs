// SPDX-License-Identifier: MIT OR Apache-2.0

//! Warpgroup MMAs: the feature kernels in the simulator,
//! descriptors, diagnostics and the MLIR.

mod feature_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::ir::Op;
use cute::prelude::*;
use cute::warpgroup::{self, OperandSource, SmemLayoutAtomKind};
use feature_kernels::{inputs_with_len, operand_layout, wgmma_cases};

/// Every WGMMA feature kernel in the simulator, under every built-in schedule.
#[test]
fn wgmma_kernels_match_rust() {
    for case in wgmma_cases() {
        let mut args = inputs_with_len(&case.kernel, true, 256);
        case.kernel
            .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
    }
}

/// The descriptors of every shared-memory layout atom against the hand-worked fields,
/// in the simulator (`tests/gpu_wgmma.rs` runs the kernel on a GPU).
#[test]
fn descriptors_match_the_canonical_layouts() {
    let case = feature_kernels::wgmma_descriptors();
    let mut args = inputs_with_len(&case.kernel, true, 256);
    case.kernel.simulate(case.cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    (case.check)(&args);
}

fn panic_message(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[track_caller]
fn expect_panic(f: impl FnOnce(), want: &[&str]) {
    let msg = panic_message(f);
    for w in want {
        assert!(msg.contains(w), "expected {w:?} in: {msg}");
    }
}

/// `MmaF16BF16Op`'s parameter checks, as CuTe DSL's (each message names the parameter
/// and what it allows).
#[test]
fn op_parameters_are_checked() {
    use DType::*;
    use OperandMajorMode::K;
    let op = |ab, acc, shape: (usize, usize, usize), src| {
        move || {
            warpgroup::MmaF16BF16Op::new(ab, acc, shape, src, K, K);
        }
    };
    let smem = OperandSource::SMEM;
    expect_panic(op(F16, F32, (128, 64, 16), smem), &["M is 128", "M = 64"]);
    expect_panic(op(F16, F32, (64, 4, 16), smem), &["N is 4", "multiple of 8 from 8 to 256"]);
    expect_panic(op(F16, F32, (64, 12, 16), smem), &["N is 12", "multiple of 8 from 8 to 256"]);
    expect_panic(op(F16, F32, (64, 264, 16), smem), &["N is 264"]);
    expect_panic(op(BF16, F16, (64, 64, 16), smem), &["BFloat16", "Float32 only", "'acc_dtype' is f16"]);
    expect_panic(op(F32, F32, (64, 64, 16), smem), &["'ab_dtype' is f32", "Float16 or BFloat16"]);
    expect_panic(op(F16, F64, (64, 64, 16), smem), &["'acc_dtype' is f64"]);
    expect_panic(op(F16, F32, (64, 64, 32), smem), &["K is 32", "K = 16"]);
    // A in registers is K-major.
    expect_panic(
        || {
            warpgroup::MmaF16BF16Op::new(F16, F32, (64, 64), OperandSource::RMEM, OperandMajorMode::MN, K);
        },
        &["'a_src' is OperandSource::RMEM", "'a_major_mode' MN", "K-major"],
    );
    warpgroup::MmaF16BF16Op::new(F16, F32, (64, 64), OperandSource::RMEM, K, K);
    // Every legal N, with or without K.
    for n in (8..=256).step_by(8) {
        let o = warpgroup::MmaF16BF16Op::new(F16, F32, (64, n), smem, K, OperandMajorMode::MN);
        assert_eq!(o.shape_mnk(), (64, n, 16));
        assert_eq!(o.kind().ptx(), format!("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.f16.f16"));
        assert!(o.kind().trans_b && !o.kind().trans_a);
    }
    let atom = make_mma_atom(warpgroup::MmaF16BF16Op::new(BF16, F32, (64, 256, 16), smem, K, K));
    assert_eq!((atom.shape_mnk(), atom.num_threads()), ([64, 256, 16], 128));
    assert_eq!(atom.fields(), &[MmaField::ACCUMULATE]);
    assert_eq!(atom.layout_c().to_string(), "((4,8,4),(2,2,32)):((128,1,16),(64,8,512))");
}

/// The kernels need sm_90a exactly: plain sm_90 and the later chips have no wgmma.
#[test]
fn wgmma_needs_sm_90a() {
    let k = wgmma_cases().remove(0).kernel;
    for arch in [Arch::Sm90, Arch::Sm100a, Arch::Sm120a] {
        let e = cute::validate::check_target(k.ir(), arch).unwrap_err().to_string();
        assert!(e.contains("wgmma") && e.contains("sm_90a"), "{arch}: {e}");
    }
    assert!(cute::validate::check_target(k.ir(), Arch::Sm90a).is_ok());
}

/// A `64 x 8 x 16` problem: A and B in K_INTER shared memory filled by plain stores
/// (with `fence_view_async_shared` when `proxy_fence`), the MMA, its fragments, the
/// accumulators and C's partition.
struct Small {
    mma: TiledMma,
    sa: Tensor<F16>,
    tcra: Tensor<F16>,
    tcrb: Tensor<F16>,
    acc: Tensor<f32>,
    tcgc: Tensor<f32>,
}

fn small(proxy_fence: bool) -> Small {
    use SmemLayoutAtomKind::K_INTER;
    let a = param_tensor_aligned::<F16>("a", Layout::new((64, 16), (16, 1)), 16);
    let b = param_tensor_aligned::<F16>("b", Layout::new((8, 16), (16, 1)), 16);
    let c = param_tensor::<f32>("c", Layout::new((64, 8), (8, 1)));
    let mma = feature_kernels::wgmma_op::<F16, f32>(8, K_INTER, K_INTER);
    let sa = make_smem_tensor_aligned::<F16>(operand_layout::<F16>(K_INTER, 64, 16, 1), 128);
    let sb = make_smem_tensor_aligned::<F16>(operand_layout::<F16>(K_INTER, 8, 16, 1), 128);
    let t = thread_idx().x;
    for_range(0, 8, 1, |j| {
        let i = j * 128 + t;
        sa.store((i % 64, i / 64), a.load((i % 64, i / 64)));
    });
    sb.store((t % 8, t / 8), b.load((t % 8, t / 8)));
    if proxy_fence {
        fence_view_async_shared();
    }
    sync_threads();
    let thr = mma.get_slice(t);
    let (tcra, tcrb) = (mma.make_fragment_a(&thr.partition_a(&sa)), mma.make_fragment_b(&thr.partition_b(&sb)));
    let tcgc = thr.partition_c(&c);
    let acc = mma.make_fragment_c::<f32>(&tcgc);
    Small { mma, sa, tcra, tcrb, acc, tcgc }
}

fn store_c(s: &Small) {
    for i in 0..s.acc.size() {
        s.tcgc.store(i, s.acc.load(i));
    }
}

fn build(name: &str, body: impl FnOnce() + 'static) -> Kernel {
    Kernel::builder(name).block_dim([128, 1, 1]).build(body)
}

/// Simulate `kernel` (one warpgroup) and return the error.
fn sim_error(kernel: &Kernel) -> String {
    let mut args = inputs_with_len(kernel, true, 256);
    kernel.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).expect_err("expected an error").to_string()
}

/// The well-formed small kernel runs; each misuse is reported.
#[test]
fn simulator_diagnostics() {
    let good = build("good", || {
        let s = small(true);
        warpgroup::fence();
        gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        store_c(&s);
    });
    let mut args = inputs_with_len(&good, true, 256);
    good.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap();

    // No warpgroup::fence before the first MMA.
    let k = build("no_fence", || {
        let s = small(true);
        gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        store_c(&s);
    });
    let e = sim_error(&k);
    assert!(e.contains("without a warpgroup::fence (wgmma.fence) before it"), "{e}");

    // An accumulator written after the fence.
    let k = build("write_after_fence", || {
        let mut s = small(true);
        warpgroup::fence();
        s.acc.fill(s.tcgc.load(0) * 0.0f32);
        s.mma.set(MmaField::ACCUMULATE, true);
        gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        store_c(&s);
    });
    let e = sim_error(&k);
    assert!(e.contains("after the warp's last warpgroup::fence"), "{e}");

    // The shared-memory operand overwritten before the wait.
    let k = build("overwrite", || {
        let s = small(true);
        warpgroup::fence();
        gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
        warpgroup::commit_group();
        if_(thread_idx().x.eq(0), || s.sa.store((0, 0), F16::from_f32(1.0)));
        warpgroup::wait_group(0);
        store_c(&s);
    });
    let e = sim_error(&k);
    assert!(e.contains("overwrites") && e.contains("still reads"), "{e}");

    // Shared memory written by plain stores without fence_view_async_shared.
    let k = build("no_proxy_fence", || {
        let s = small(false);
        warpgroup::fence();
        gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        store_c(&s);
    });
    let e = sim_error(&k);
    assert!(e.contains("fence.proxy.async") && e.contains("fence_view_async_shared"), "{e}");

    // Three warps of the warpgroup issue the MMA; the fourth never does.
    let k = build("three_warps", || {
        let s = small(true);
        if_(warp_idx().lt(3), || {
            let acc = s.mma.make_fragment_c::<f32>(&s.tcgc);
            warpgroup::fence();
            gemm(&s.mma, &acc, &s.tcra, &s.tcrb, &acc);
            warpgroup::commit_group();
            warpgroup::wait_group(0);
            for i in 0..acc.size() {
                s.tcgc.store(i, acc.load(i));
            }
        });
    });
    let e = sim_error(&k);
    assert!(e.contains("which warps 3 of warpgroup 0 have not issued"), "{e}");

    // A read before the wait that the build-time check cannot see: the kernel's
    // wait_group(1) (which completes the first of two groups) turned into
    // wait_group(2) in the IR.
    let k = build("early_read", || {
        let s = small(true);
        let acc2 = s.mma.make_fragment_c::<f32>(&s.tcgc);
        warpgroup::fence();
        gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
        warpgroup::commit_group();
        gemm(&s.mma, &acc2, &s.tcra, &s.tcrb, &acc2);
        warpgroup::commit_group();
        warpgroup::wait_group(1);
        store_c(&s);
        warpgroup::wait_group(0);
        for i in 0..acc2.size() {
            s.tcgc.store(i, s.tcgc.load(i) + acc2.load(i));
        }
    });
    let mut args = inputs_with_len(&k, true, 256);
    k.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).expect("the kernel as built is correct");
    let mut ir = k.ir().clone();
    for inst in &mut ir.body.insts {
        if let Op::WgmmaWait { pending, .. } = &mut inst.op
            && *pending == 1
        {
            *pending = 2;
        }
    }
    let e = sim_error(&Kernel::from_ir(ir).unwrap());
    assert!(e.contains("reads an accumulator of a wgmma.mma_async") && e.contains("before a wgmma.wait_group"), "{e}");
}

/// A fragment made outside an `if_` whose MMA, fence and wait are all inside it: the
/// registers the branch wrote are rebound there (the fragment is filled before the
/// branch, which does not zero it), and a fence inside an unrelated branch leaves the
/// fragment alone.
#[test]
fn accumulators_from_outside_a_branch() {
    let k = build("in_if", || {
        let s = small(true);
        s.acc.fill(0.0f32);
        if_(thread_idx().x.lt(1000), || {
            warpgroup::fence();
            gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
            warpgroup::commit_group();
            warpgroup::wait_group(0);
            store_c(&s);
        });
    });
    let mut args = inputs_with_len(&k, true, 256);
    k.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap();
    let (a, b, c) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<f32>());
    for (i, &got) in c.iter().enumerate() {
        let (m, n) = (i / 8, i % 8);
        let want: f32 = (0..16).map(|kk| a[m * 16 + kk].to_f32() * b[n * 16 + kk].to_f32()).sum();
        assert_eq!(got, want, "c[{m}][{n}]");
    }
    let k = build("fence_in_other_if", || {
        let s = small(true);
        if_(thread_idx().x.lt(1000), warpgroup::fence);
        warpgroup::fence();
        gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        store_c(&s);
    });
    let mut args = inputs_with_len(&k, true, 256);
    k.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap();
}

/// The validator: a read of an accumulator before the wait that completes it, and
/// accumulators left in flight.
#[test]
fn build_time_diagnostics() {
    let e = panic_message(|| {
        build("read_before_wait", || {
            let s = small(true);
            warpgroup::fence();
            gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
            warpgroup::commit_group();
            store_c(&s);
            warpgroup::wait_group(0);
        });
    });
    assert!(
        e.contains("reads an accumulator of the warpgroup MMA at") && e.contains("before a wgmma.wait_group"),
        "{e}"
    );
    // wait_group(1) leaves the newest group in flight.
    let e = panic_message(|| {
        build("wait_one", || {
            let s = small(true);
            warpgroup::fence();
            gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
            warpgroup::commit_group();
            warpgroup::wait_group(1);
            store_c(&s);
            warpgroup::wait_group(0);
        });
    });
    assert!(e.contains("reads an accumulator"), "{e}");
    // An uncommitted MMA is not covered by wait_group(0).
    let e = panic_message(|| {
        build("uncommitted", || {
            let s = small(true);
            warpgroup::fence();
            gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
            warpgroup::wait_group(0);
            store_c(&s);
        });
    });
    assert!(e.contains("reads an accumulator"), "{e}");
    let e = panic_message(|| {
        build("dropped", || {
            let s = small(true);
            warpgroup::fence();
            gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
            warpgroup::commit_group();
        });
    });
    assert!(e.contains("dropped while it may still be in flight"), "{e}");
    // Two accumulators: wait_group(1) completes the first group, whose results may be
    // read.
    build("two_groups", || {
        let s = small(true);
        let acc2 = s.mma.make_fragment_c::<f32>(&s.tcgc);
        warpgroup::fence();
        gemm(&s.mma, &s.acc, &s.tcra, &s.tcrb, &s.acc);
        warpgroup::commit_group();
        gemm(&s.mma, &acc2, &s.tcra, &s.tcrb, &acc2);
        warpgroup::commit_group();
        warpgroup::wait_group(1);
        store_c(&s);
        warpgroup::wait_group(0);
        store_c(&Small { acc: acc2, ..s });
    });
    // Fragments: a swizzle that is no GMMA layout, a buffer aligned below the swizzle's
    // repeat, a register operand.
    expect_panic(
        || {
            build("unaligned", || {
                let mma =
                    feature_kernels::wgmma_op::<F16, f32>(64, SmemLayoutAtomKind::K_SW128, SmemLayoutAtomKind::K_SW128);
                let sa =
                    make_smem_tensor_aligned::<F16>(operand_layout::<F16>(SmemLayoutAtomKind::K_SW128, 64, 64, 1), 128);
                let _ = mma.make_fragment_a(&mma.get_slice(thread_idx().x).partition_a(&sa));
            });
        },
        &["aligned to 128 bytes", "make_smem_tensor_aligned(.., 1024)"],
    );
    expect_panic(
        || {
            build("not_canonical", || {
                let mma =
                    feature_kernels::wgmma_op::<F16, f32>(64, SmemLayoutAtomKind::K_SW128, SmemLayoutAtomKind::K_SW128);
                let sa = make_smem_tensor_aligned::<F16>(Layout::new((64, 64), (1, 64)), 1024);
                let _ = mma.make_fragment_a(&mma.get_slice(thread_idx().x).partition_a(&sa));
            });
        },
        &["not a canonical GMMA layout"],
    );
    expect_panic(
        || {
            build("registers", || {
                let mma =
                    feature_kernels::wgmma_op::<F16, f32>(64, SmemLayoutAtomKind::K_SW128, SmemLayoutAtomKind::K_SW128);
                let r = make_fragment::<F16>((64, 64));
                let _ = mma.make_fragment_a(&mma.get_slice(thread_idx().x).partition_a(&r));
            });
        },
        &["not in shared memory"],
    );
}

/// The MLIR: the NVVM fence, commit and wait operations, the MMA as inline assembly
/// with its accumulators tied, the descriptors and scale-d operands and the transpose
/// immediates, and the operand fences around them.
#[test]
fn mlir_of_a_warpgroup_mma() {
    let case = wgmma_cases().into_iter().find(|c| c.name == "wgmma_n8_ksw128_mninter").unwrap();
    let mlir = case.kernel.to_mlir();
    for want in [
        "nvvm.wgmma.fence.aligned",
        "nvvm.wgmma.commit.group.sync.aligned",
        "nvvm.wgmma.wait.group.sync.aligned 0",
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {$0, $1, $2, $3}, $4, $5, p, 1, 1, 0, 1;",
        "setp.ne.b32 p, $6, 0;",
        "\"=f,=f,=f,=f,l,l,r,0,1,2,3\"",
        // The operand fences: empty statements with side effects over the accumulators.
        "llvm.inline_asm has_side_effects \"\", \"=f,=f,=f,=f,0,1,2,3\"",
    ] {
        assert!(mlir.contains(want), "{want:?} missing from:\n{mlir}");
    }
    assert_eq!(case.kernel.ir().count_ops(|op| matches!(op, Op::Wgmma { .. })), 4);
    // Float16 accumulators: packed pairs in the instruction and the fences.
    let f16 = wgmma_cases().into_iter().find(|c| c.name == "wgmma_n128_f16acc_ksw128_mnsw128").unwrap().kernel;
    let mlir = f16.to_mlir();
    assert!(mlir.contains("wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 {$0, "), "{mlir}");
    assert!(mlir.contains("=r,=r,") && mlir.contains("vector<2xf16> to i32"), "{mlir}");
    // A bf16 MMA with a transposed A.
    let bf16 = wgmma_cases().into_iter().find(|c| c.name == "wgmma_n128_bf16_mnsw64_ksw64").unwrap().kernel;
    assert!(bf16.to_mlir().contains("m64n128k16.f32.bf16.bf16 {"), "bf16");
    assert!(bf16.to_mlir().contains(", p, 1, 1, 1, 0;"), "trans-a");
}

/// The PTX of every WGMMA kernel with each installed `mlir-opt`, and, when `ptxas` is
/// installed, its assembly for sm_90a without the warning that ptxas serialised the
/// MMAs (which it gives when it cannot prove the accumulators untouched between them).
#[test]
fn ptx_and_ptxas_for_sm_90a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for case in wgmma_cases() {
        let ptx = case.kernel.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert!(ptx.contains(".target sm_90a") && ptx.contains("wgmma.mma_async.sync.aligned"), "{}", case.name);
        assert!(ptx.contains("wgmma.fence.sync.aligned") && ptx.contains("wgmma.commit_group.sync.aligned"));
        let Some(ptxas) = find_ptxas() else { continue };
        let path = std::env::temp_dir().join(format!("cute-wgmma-{}-{}.ptx", std::process::id(), case.name));
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
