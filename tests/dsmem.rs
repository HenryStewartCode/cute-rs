// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distributed shared memory (roadmap part m8-dsmem): the feature kernels in the
//! simulator, diagnostics, and the MLIR and PTX.

mod feature_kernels;

use cute::prelude::*;
use feature_kernels::{dsmem_cases_sm90, inputs_with_len};

/// Every DSMEM feature kernel in the simulator, under every built-in schedule, on small
/// and random inputs.
#[test]
fn dsmem_kernels_match_rust() {
    for case in dsmem_cases_sm90() {
        for small in [true, false] {
            let mut args = inputs_with_len(&case.kernel, small, 256);
            case.kernel
                .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
        }
    }
}

use std::panic::{AssertUnwindSafe, catch_unwind};

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

/// A two-CTA cluster kernel with a 128-element Int32 `buf` in shared memory per CTA and
/// an output; `body` gets the buffer, the thread and the CTA rank.
fn pair(name: &str, body: impl FnOnce(&Tensor<i32>, &Tensor<i32>, Int32, Int32) + 'static) -> Kernel {
    Kernel::builder(name).block_dim([128, 1, 1]).build(move || {
        let out = param_tensor::<i32>("out", Layout::col_major(256));
        let buf = make_smem_tensor_aligned::<i32>(Layout::col_major(128), 16);
        body(&buf, &out, thread_idx().x, block_idx_in_cluster());
    })
}

fn sim_error(k: &Kernel) -> String {
    let mut args = inputs_with_len(k, true, 256);
    let cfg = LaunchConfig::new([2, 1, 1], [128, 1, 1]).with_cluster([2, 1, 1]);
    k.simulate(cfg, &mut args).expect_err("expected an error").to_string()
}

fn remote(buf: &Tensor<i32>, rank: impl IntoValue<i32>) -> Tensor<i32> {
    make_tensor(map_dsmem_ptr(&buf.ptr(), rank), Layout::col_major(128))
}

/// The simulator's checks: a peer rank outside the cluster, an access before a cluster
/// barrier shows the peer has started, a CTA exiting while its peer may still access its
/// shared memory, a race between CTAs, a misaligned `st.async`; and the well-formed
/// kernel.
#[test]
fn simulator_diagnostics() {
    let good = pair("good", |buf, out, t, rank| {
        buf.store(t, t);
        cluster_arrive();
        cluster_wait();
        out.store(rank * 128 + t, remote(buf, 1 - rank).load(t));
        cluster_arrive();
        cluster_wait();
    });
    let mut args = inputs_with_len(&good, true, 256);
    let cfg = LaunchConfig::new([2, 1, 1], [128, 1, 1]).with_cluster([2, 1, 1]);
    good.simulate_all_orders(cfg, &mut args, &SimOptions::default()).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), (0..256).map(|i| i % 128).collect::<Vec<i32>>());

    let e = sim_error(&pair("rank", |buf, _, t, _| {
        cluster_arrive();
        cluster_wait();
        remote(buf, 5).store(t, t);
        cluster_arrive();
        cluster_wait();
    }));
    assert!(e.contains("on CTA 5 of a cluster of 2"), "{e}");

    let e = sim_error(&pair("early", |buf, _, t, rank| {
        remote(buf, 1 - rank).store(t, t);
        cluster_arrive();
        cluster_wait();
    }));
    assert!(e.contains("before anything shows that CTA") && e.contains("cluster barrier"), "{e}");

    // CTA 1 leaves after the first barrier while CTA 0 writes its shared memory.
    let e = sim_error(&pair("exit", |buf, _, t, rank| {
        cluster_arrive();
        cluster_wait();
        if_(rank.eq(0), || remote(buf, 1).store(t, t));
    }));
    assert!(e.contains("exits while") || e.contains("which has exited"), "{e}");

    let e = sim_error(&pair("race", |buf, out, t, rank| {
        buf.store(t, 0);
        cluster_arrive();
        cluster_wait();
        if_(rank.eq(0), || remote(buf, 1).store(t, t));
        if_(rank.eq(1), || out.store(t, buf.load(t)));
        cluster_arrive();
        cluster_wait();
    }));
    assert!(e.contains("data race"), "{e}");

    // An st.async of four values at a runtime offset that is only 4-byte aligned.
    let e = sim_error(&pair("misaligned", |buf, _, t, rank| {
        let mbar = alloc_smem::<u64>(1, 8);
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        cluster_arrive();
        cluster_wait();
        if_(t.eq(0), || {
            let v = [t, t, t, t];
            store_async_dsmem(&buf.ptr().offset(t + 1), v, &mbar, 1 - rank);
        });
        cluster_arrive();
        cluster_wait();
    }));
    assert!(e.contains("misaligned st.async"), "{e}");
}

/// What the builder refuses: `map_dsmem_ptr` of a pointer that is not to the CTA's own
/// shared memory, `ss` and `shared_space` that do not match the pointer, instructions that do
/// not reach another CTA's shared memory, a `CopyDsmemStoreOp` whose mbarrier is not the
/// destination CTA's, and a misaligned `st.async` vector known when the kernel is built.
#[test]
fn build_time_diagnostics() {
    let build = |f: fn()| move || drop(Kernel::builder("k").block_dim([128, 1, 1]).build(f));
    expect_panic(
        build(|| {
            let g = param_tensor::<i32>("g", Layout::col_major(4));
            let _ = map_dsmem_ptr(&g.ptr(), 1);
        }),
        &["map_dsmem_ptr: smem_ptr is a gmem pointer"],
    );
    expect_panic(
        build(|| {
            let s = make_smem_tensor::<i32>(Layout::col_major(4));
            let _ = map_dsmem_ptr(&map_dsmem_ptr(&s.ptr(), 1), 0);
        }),
        &["smem_ptr is a dsmem pointer"],
    );
    expect_panic(
        build(|| {
            let s = make_smem_tensor::<i32>(Layout::col_major(4));
            let p = map_dsmem_ptr(&s.ptr(), 1);
            let _ = load(&p, LoadOptions { ss: Some(SharedSpace::Cta), ..Default::default() });
        }),
        &["ss=SharedSpace::Cta", "use ss=SharedSpace::Cluster"],
    );
    expect_panic(
        build(|| {
            let s = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
            let r = make_fragment::<i32>(4);
            let attrs = S2RAttrs { shared_space: SharedSpace::Cluster, ..Default::default() };
            copy(&make_copy_atom::<i32>(CopyOp::S2R { bits: 128, attrs }), &s, &r);
        }),
        &["shared_space=SharedSpace::Cluster", "not smem memory"],
    );
    expect_panic(
        build(|| {
            let g = param_tensor::<i32>("g", Layout::col_major(4));
            let s = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
            let d = make_tensor(map_dsmem_ptr(&s.ptr(), 1), Layout::col_major(4));
            copy(&make_copy_atom::<i32>(CopyOp::CpAsync { bytes: 16, bypass_l1: false }), &g, &d);
        }),
        &["in another CTA's shared memory", "cannot reach"],
    );
    expect_panic(
        build(|| {
            let s = make_smem_tensor_aligned::<i32>(Layout::col_major(512), 16);
            let mbar = alloc_smem::<u64>(1, 8);
            let r = make_fragment::<i32>(4);
            r.fill(0);
            let d = make_tensor(map_dsmem_ptr(&s.ptr(), 1), Layout::col_major(4));
            let atom = make_copy_atom::<i32>(CopyOp::DsmemStore { bits: 128 });
            copy_with(&atom, &r, &d, CopyArgs::new().mbar_ptr(&mbar));
        }),
        &["mbar_ptr is the mbarrier of the CTA the store writes to", "it is this CTA's"],
    );
    expect_panic(
        build(|| {
            let s = make_smem_tensor_aligned::<i32>(Layout::col_major(512), 16);
            let mbar = alloc_smem::<u64>(1, 8);
            let r = make_fragment::<i32>(4);
            r.fill(0);
            let d = make_tensor(map_dsmem_ptr(&s.ptr().offset(1), 1), Layout::col_major(4));
            let atom = make_copy_atom::<i32>(CopyOp::DsmemStore { bits: 128 });
            copy_with(&atom, &r, &d, CopyArgs::new().mbar_ptr(&map_dsmem_ptr(&mbar, 1)));
        }),
        &["cannot prove", "aligned to the store's 4 elements"],
    );
    expect_panic(|| drop(make_copy_atom::<i32>(CopyOp::DsmemStore { bits: 96 })), &["num_bits_per_copy 96"]);
    // A masked load or store of a DSMEM tensor reaches the peer.
    let k = Kernel::builder("k").block_dim([32, 1, 1]).build(|| {
        let s = make_smem_tensor::<i32>(Layout::col_major(8));
        let d = make_tensor(map_dsmem_ptr(&s.ptr(), 1), Layout::col_major(8));
        let mask = TensorSsa::new(8, (0..8).map(Int32::constant).collect()).lt(thread_idx().x);
        let v = d.load_ssa_masked(&mask, None);
        d.store_ssa_masked(&v, &mask);
    });
    let dump = k.dump();
    assert_eq!(dump.matches("in CTA").count(), 16, "{dump}");
    let mlir = k.to_mlir();
    assert_eq!(mlir.matches("mapa.shared::cluster.u64").count(), 16, "{mlir}");
    // The integer address of a DSMEM pointer is not modelled yet.
    expect_panic(
        build(|| {
            let s = make_smem_tensor::<i32>(Layout::col_major(4));
            let _ = map_dsmem_ptr(&s.ptr(), 1).toint();
        }),
        &["toint of the pointer", "another CTA's shared memory"],
    );
}

/// DSMEM accesses need sm_90 (`.shared::cluster`, PTX 7.8; `st.async` PTX 8.1).
#[test]
fn dsmem_needs_sm_90() {
    for case in dsmem_cases_sm90() {
        let e = cute::validate::check_target(case.kernel.ir(), Arch::Sm80).unwrap_err().to_string();
        assert!(e.contains("sm_90"), "{}: {e}", case.name);
        cute::validate::check_target(case.kernel.ir(), Arch::Sm90).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    }
}

/// The MLIR maps addresses with inline `mapa`, and the PTX of every DSMEM kernel (with
/// each installed `mlir-opt`) accesses `.shared::cluster`; `ptxas` assembles it for
/// sm_90 when installed.
#[test]
fn mlir_ptx_and_ptxas_for_sm_90() {
    let cases = dsmem_cases_sm90();
    let ring = &cases.iter().find(|c| c.name == "dsmem_ring_2x1").unwrap().kernel;
    let mlir = ring.to_mlir();
    assert!(mlir.contains("mapa.shared::cluster.u64 $0, $1, $2;") && mlir.contains("!llvm.ptr<7>"), "{mlir}");
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let want: &[(&str, &[&str])] = &[
        ("dsmem_ring_2x1", &["st.shared::cluster.b32", "ld.shared::cluster.b32"]),
        (
            "dsmem_cluster_reduce_2x1",
            &["red.relaxed.cluster.shared::cluster.add.s32", "atom.relaxed.cluster.shared::cluster.add"],
        ),
        (
            "dsmem_s2r_r2s_2x1",
            &[
                "st.relaxed.cluster.shared::cluster.v4.b32",
                "ld.acquire.cluster.shared::cluster.v4.b32",
                "st.release.cluster.shared::cluster.b32",
            ],
        ),
        ("dsmem_store_f16_b64_4x1", &["st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32"]),
        ("dsmem_store_u8_b128_2x1", &["st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32"]),
    ];
    for case in &cases {
        let ptx = case.kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert!(ptx.contains("mapa.shared::cluster"), "{}: {ptx}", case.name);
        if let Some((_, strings)) = want.iter().find(|(n, _)| *n == case.name) {
            for s in *strings {
                assert!(ptx.contains(s), "{}: {s} missing from:\n{ptx}", case.name);
            }
        }
        let Some(ptxas) = find_ptxas() else { continue };
        let path = std::env::temp_dir().join(format!("cute-dsmem-{}-{}.ptx", std::process::id(), case.name));
        std::fs::write(&path, &ptx).unwrap();
        let out = std::process::Command::new(&ptxas)
            .args(["-arch=sm_90", "-o", "/dev/null"])
            .arg(&path)
            .output()
            .expect("running ptxas");
        let _ = std::fs::remove_file(&path);
        assert!(out.status.success(), "{}: ptxas failed:\n{}", case.name, String::from_utf8_lossy(&out.stderr));
    }
}

/// `ptxas` on `PATH`.
fn find_ptxas() -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("ptxas")).find(|p| p.is_file())
}
