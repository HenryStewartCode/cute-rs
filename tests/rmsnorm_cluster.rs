// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cluster RMSNorm port: the
//! simulator against plain Rust for one, two and four CTAs per cluster on
//! Float16, BFloat16 and Float32, the simulator's diagnostics of a broken
//! cluster reduction, and the sm_90 PTX (with `ptxas` where installed).

#[allow(dead_code)]
#[path = "../examples/rmsnorm_cluster.rs"]
mod rmsnorm_cluster;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;
use rmsnorm_cluster::{Misuse, RMSNormConfig, rmsnorm, rmsnorm_with_misuse, run_bf16, run_f16, run_f32};

/// `(N, SM version, cluster_n)` for 16-bit elements: every threads-per-row
/// step of the example's table on one CTA (8, 16, 32, 64 and 128 threads
/// per row; 16, 8, 4, 2 and 1 rows per CTA), then clusters of two and four
/// CTAs. N is ragged (not a whole number of tiles) except for 4096.
const HALF: [(usize, u32, usize); 9] = [
    (40, 90, 1),
    (120, 90, 1),
    (1000, 90, 1),
    (4096, 80, 1),
    (12000, 90, 1),
    (20000, 80, 1),
    (20000, 90, 2),
    (16392, 90, 2),
    (40000, 90, 4),
];

/// The same for Float32, whose clusters start at twice the row length.
const SINGLE: [(usize, u32, usize); 6] =
    [(40, 90, 1), (1000, 90, 1), (12000, 90, 1), (20000, 90, 1), (40000, 90, 2), (80000, 90, 4)];

/// Row counts for one built kernel: one and two rows of a cluster (the
/// simulation stays short), more when a CTA holds several rows, with a
/// partial last CTA.
fn rows(cfg: &RMSNormConfig) -> Vec<usize> {
    match cfg.rows_per_block {
        1 => vec![1, 2],
        r => vec![1, r + 1, 2 * r - 1],
    }
}

fn config(dtype: DType, (n, sm, cluster_n): (usize, u32, usize)) -> RMSNormConfig {
    let cfg = RMSNormConfig::new(dtype, n, true, sm);
    assert_eq!(cfg.cluster_n, cluster_n, "{dtype} N = {n} on SM{sm}: the example's cluster size");
    cfg
}

#[test]
fn float16_matches_rust() {
    for c in HALF {
        let cfg = config(DType::F16, c);
        let kernel = rmsnorm::<F16>(cfg);
        for m in rows(&cfg) {
            run_f16(&kernel, &cfg, m);
        }
    }
}

#[test]
fn bfloat16_matches_rust() {
    for c in HALF {
        let cfg = config(DType::BF16, c);
        let kernel = rmsnorm::<BF16>(cfg);
        for m in rows(&cfg) {
            run_bf16(&kernel, &cfg, m);
        }
    }
}

#[test]
fn float32_matches_rust() {
    for c in SINGLE {
        let cfg = config(DType::F32, c);
        let kernel = rmsnorm::<f32>(cfg);
        for m in rows(&cfg) {
            run_f32(&kernel, &cfg, m);
        }
    }
}

/// Without a weight, and with the precise `rsqrt`, on one CTA and on a
/// cluster of two.
#[test]
fn without_weight_and_precise() {
    for (n, sm) in [(1000, 90), (20000, 90)] {
        let cfg = RMSNormConfig { has_weight: false, ..RMSNormConfig::new(DType::F16, n, true, sm) };
        run_f16(&rmsnorm::<F16>(cfg), &cfg, 3);
        let cfg = RMSNormConfig::new(DType::F16, n, true, sm).precise();
        run_f16(&rmsnorm::<F16>(cfg), &cfg, 2);
    }
}

/// The cluster reduction under the simulator's built-in schedules: the
/// result is the same whatever order the CTAs and warps run in.
#[test]
fn cluster_of_two_under_every_schedule() {
    let cfg = RMSNormConfig::new(DType::F16, 20000, true, 90);
    let kernel = rmsnorm::<F16>(cfg);
    let mut args = rmsnorm_cluster::inputs(&cfg, 1, F16::from_f32);
    kernel.simulate_all_orders(cfg.launch(1), &mut args, &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    rmsnorm_cluster::check(kernel.name(), &cfg, &args, F16::to_f32, 4e-3);
}

/// The configuration's derived values for the example's own sizes.
#[test]
fn configuration_follows_the_example() {
    let cfg = RMSNormConfig::new(DType::BF16, 32768, true, 90);
    assert_eq!(
        (cfg.cluster_n, cfg.n_per_cta, cfg.threads_per_row, cfg.num_threads, cfg.rows_per_block),
        (2, 16384, 128, 128, 1)
    );
    assert_eq!((cfg.vec_size, cfg.num_vec_blocks, cfg.cols_per_tile, cfg.warps_per_row), (8, 16, 16384, 4));
    assert_eq!(cfg.smem_size_in_bytes(), 16384 * 2 + 4 * 2 * 4 + 8);
    let cfg = RMSNormConfig::new(DType::BF16, 4096, true, 100);
    assert_eq!((cfg.cluster_n, cfg.threads_per_row, cfg.rows_per_block, cfg.num_vec_blocks), (1, 64, 2, 8));
    for (n, want) in [(16384, 1), (16385 + 7, 2), (65536, 4), (131072, 8), (131080, 16)] {
        assert_eq!(RMSNormConfig::compute_cluster_n(n, DType::F16, 90), want, "N = {n}");
        assert_eq!(RMSNormConfig::compute_cluster_n(n, DType::F16, 89), 1);
        assert_eq!(RMSNormConfig::compute_cluster_n(2 * n, DType::F32, 90), want, "Float32 N = {}", 2 * n);
    }
    // The ceiling of N / cluster_n: the floor's tiles would miss a column.
    let cfg = RMSNormConfig::new(DType::F16, 16392, true, 90);
    assert_eq!((cfg.cluster_n, cfg.n_per_cta, cfg.num_vec_blocks, cfg.cols_per_tile), (2, 8196, 9, 9216));
    let e = catch_unwind(|| RMSNormConfig::new(DType::F16, 1001, true, 90)).unwrap_err();
    let msg = e.downcast_ref::<String>().unwrap();
    assert!(msg.contains("whole number of 128-bit vectors of 8 elements"), "{msg}");
}

fn misuse_error(misuse: Misuse) -> String {
    let cfg = RMSNormConfig::new(DType::F16, 20000, true, 90);
    let kernel = rmsnorm_with_misuse::<F16>(cfg, misuse);
    let mut args = rmsnorm_cluster::inputs(&cfg, 1, F16::from_f32);
    let r = catch_unwind(AssertUnwindSafe(|| kernel.simulate(cfg.launch(1), &mut args)));
    match r {
        Ok(Ok(_)) => panic!("{misuse:?}: the simulator accepted the kernel"),
        Ok(Err(e)) => e.to_string(),
        Err(p) => p.downcast_ref::<String>().cloned().unwrap_or_default(),
    }
}

/// A phase whose `expect_tx` misses one peer's partial: the transaction
/// count runs out before the last `st.async` lands.
#[test]
fn expect_tx_short_of_the_peers_partials_is_reported() {
    let e = misuse_error(Misuse::ExpectTxShort);
    assert!(e.contains("transaction count is") && e.contains("mbarrier_expect_tx too small"), "{e}");
}

/// A CTA that leaves without waiting for its mbarrier or arriving at the
/// cluster barrier, while its peers still store into its shared memory.
#[test]
fn cta_exiting_before_the_cluster_barrier_is_reported() {
    let e = misuse_error(Misuse::ExitBeforeClusterBarrier);
    // Which message depends on whether a peer's store lands before or after
    // the CTA exits.
    assert!(e.contains("exits while") || e.contains("which has exited"), "{e}");
}

/// A cluster kernel needs sm_90; a single-CTA one runs on sm_80.
#[test]
fn cluster_kernels_need_sm_90() {
    let cluster = rmsnorm::<F16>(RMSNormConfig::new(DType::F16, 20000, true, 90));
    let e = cute::validate::check_target(cluster.ir(), Arch::Sm80).unwrap_err().to_string();
    assert!(e.contains("sm_90"), "{e}");
    let single = rmsnorm::<F16>(RMSNormConfig::new(DType::F16, 20000, true, 80));
    cute::validate::check_target(single.ir(), Arch::Sm80).unwrap();
}

/// The MLIR maps the reduction slots and mbarrier with `mapa` and stores
/// with `st.async`; the sm_90a PTX of every cluster kernel (with an
/// installed `mlir-opt`) has the cluster barrier, `mapa` and `st.async`,
/// and `ptxas -arch=sm_90a` assembles it when installed.
#[test]
fn cluster_ptx_for_sm_90a() {
    let kernels: Vec<Kernel> = vec![
        rmsnorm::<F16>(RMSNormConfig::new(DType::F16, 20000, true, 90)),
        rmsnorm::<BF16>(RMSNormConfig::new(DType::BF16, 40000, true, 90)),
        rmsnorm::<f32>(RMSNormConfig::new(DType::F32, 80000, false, 90)),
    ];
    let mlir = kernels[0].to_mlir();
    for s in ["mapa.shared::cluster", "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32"] {
        assert!(mlir.contains(s), "{s} missing from the MLIR:\n{mlir}");
    }
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for kernel in &kernels {
        let name = kernel.name();
        let ptx = kernel.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{name}: {e}"));
        for s in [
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32",
            "mapa.shared::cluster",
            "barrier.cluster.arrive.relaxed",
            "barrier.cluster.wait",
            "mbarrier.arrive.expect_tx",
            "cp.async.ca.shared.global",
            "fence.mbarrier_init.release.cluster",
        ] {
            assert!(ptx.contains(s), "{name}: {s} missing from the PTX:\n{ptx}");
        }
        let Some(ptxas) = find_ptxas() else { continue };
        let path = std::env::temp_dir().join(format!("cute-rmsnorm-{}-{name}.ptx", std::process::id()));
        std::fs::write(&path, &ptx).unwrap();
        let out = std::process::Command::new(&ptxas)
            .args(["-arch=sm_90a", "-o", "/dev/null"])
            .arg(&path)
            .output()
            .expect("running ptxas");
        let _ = std::fs::remove_file(&path);
        assert!(out.status.success(), "{name}: ptxas failed:\n{}", String::from_utf8_lossy(&out.stderr));
    }
}

/// `ptxas` on `PATH`.
fn find_ptxas() -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("ptxas")).find(|p| p.is_file())
}
