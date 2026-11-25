// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Hopper fused multi-head attention forward port: the simulator against a plain Rust attention in f64 for
//! non-causal, residual-mask, causal and windowed masks, head dimensions 64
//! and 128, sequence lengths that are not multiples of the tile,
//! grouped-query heads, batches, the persistent scheduler and random
//! schedules with the race detector on; the kernel's instructions; and the
//! sm_90a PTX with `ptxas` where installed.

#[allow(dead_code)]
#[path = "../examples/hopper_fmha.rs"]
mod hopper_fmha;

use cute::prelude::*;
use hopper_fmha::{MaskEnum, Problem, SIM_SMS, fmha_for, run};

const F16: DType = DType::F16;

/// Build once, then simulate and check each problem (same configuration).
fn check_all(tile_n: usize, dtype: DType, persistent: bool, problems: &[Problem]) {
    let first = &problems[0];
    let fmha = fmha_for(first, (64, tile_n), dtype, dtype, persistent);
    let kernel = fmha.build();
    for p in problems {
        assert_eq!(fmha_for(p, (64, tile_n), dtype, dtype, persistent).name(), fmha.name(), "same kernel for {p:?}");
        run(&fmha, &kernel, p);
    }
}

/// No mask: K lengths that are multiples of the tile, the example's tile
/// (N 128, head dimension 64), one and several K tiles, whole and partial M
/// tiles (rows past the sequence are not stored).
#[test]
fn no_mask_whole_tiles() {
    check_all(
        128,
        F16,
        false,
        &[
            Problem::new(1, 128, 128, 1, 1, 64),
            Problem::new(1, 128, 384, 1, 1, 64),
            Problem::new(1, 200, 256, 2, 2, 64),
        ],
    );
}

/// The residual mask: K lengths that are not multiples of the tile.
#[test]
fn residual_mask() {
    let p = Problem::new(1, 130, 200, 2, 1, 64);
    assert_eq!(p.mask(128).0, MaskEnum::ResidualMask);
    check_all(128, F16, false, &[p, Problem::new(1, 64, 129, 1, 1, 64), Problem::new(1, 70, 33, 1, 1, 64)]);
}

/// Causal masks, with the top-left and the bottom-right alignment.
#[test]
fn causal() {
    check_all(
        128,
        F16,
        false,
        &[
            Problem::new(1, 128, 128, 1, 1, 64).causal(),
            Problem::new(1, 300, 300, 2, 1, 64).causal(),
            Problem::new(1, 200, 100, 1, 1, 64).causal(),
        ],
    );
    // The end of Q aligned with the end of K.
    let aligned = Problem { bottom_right_align: true, ..Problem::new(1, 100, 300, 2, 2, 64).causal() };
    assert_eq!(aligned.mask(128).0, MaskEnum::WindowMaskInference);
    check_all(128, F16, false, &[aligned, Problem { s_q: 256, s_k: 256, ..aligned }]);
}

/// Sliding windows: right only, left only, both sides.
#[test]
fn windows() {
    let base = Problem::new(1, 300, 300, 1, 1, 64);
    check_all(128, F16, false, &[Problem { window: (None, Some(40)), ..base }]);
    check_all(128, F16, false, &[Problem { window: (Some(40), None), ..base }]);
    check_all(128, F16, false, &[Problem { window: (Some(150), Some(20)), ..base }]);
    let aligned =
        Problem { window: (Some(30), Some(10)), bottom_right_align: true, ..Problem::new(1, 100, 300, 1, 1, 64) };
    check_all(128, F16, false, &[aligned]);
}

/// Head dimension 128 and BFloat16, a smaller K/V tile.
#[test]
fn head_dim_128_bfloat16_and_tile_64() {
    check_all(128, F16, false, &[Problem::new(1, 128, 256, 1, 1, 128).causal()]);
    check_all(64, F16, false, &[Problem::new(1, 128, 200, 2, 1, 64)]);
    check_all(128, DType::BF16, false, &[Problem::new(1, 128, 200, 1, 1, 64)]);
    check_all(128, DType::BF16, false, &[Problem::new(1, 100, 128, 1, 1, 64).causal()]);
}

/// Grouped-query heads (queries per key head: 2, 4) and batches.
#[test]
fn grouped_query_and_batches() {
    check_all(128, F16, false, &[Problem::new(2, 130, 130, 4, 2, 64)]);
    check_all(128, F16, false, &[Problem::new(1, 64, 128, 4, 1, 64).causal()]);
}

/// The persistent scheduler, with fewer CTAs than tiles, for each mask.
#[test]
fn persistent() {
    check_all(128, F16, true, &[Problem::new(2, 200, 256, 2, 1, 64), Problem::new(1, 128, 128, 1, 1, 64)]);
    check_all(128, F16, true, &[Problem::new(1, 300, 300, 2, 2, 64).causal()]);
    check_all(128, F16, true, &[Problem::new(2, 130, 200, 4, 2, 64)]);
    let w = Problem { window: (Some(50), Some(20)), ..Problem::new(1, 260, 260, 2, 1, 64) };
    check_all(128, F16, true, &[w]);
    // Head dimension 128 persistent (O's staging has its own buffer).
    check_all(128, F16, true, &[Problem::new(1, 200, 200, 2, 1, 128).causal()]);
    let tiles = 2 * 2 * 3;
    assert!(tiles > SIM_SMS);
}

/// `problem` under every built-in schedule of the simulator and seeded
/// random ones, with the race detector on.
fn under_schedules(p: &Problem, persistent: bool, seeds: &[u64]) {
    let fmha = fmha_for(p, (64, 128), F16, F16, persistent);
    let kernel = fmha.build();
    let mut args = hopper_fmha::inputs(&fmha, p);
    kernel
        .simulate_all_orders(fmha.launch(p.s_q, p.h, p.b, SIM_SMS), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{} {p:?}: {e}", kernel.name()));
    hopper_fmha::check(&fmha, p, &args);
    let mut args = hopper_fmha::inputs(&fmha, p);
    kernel
        .simulate_schedules(fmha.launch(p.s_q, p.h, p.b, SIM_SMS), &mut args, &SimOptions::default(), seeds)
        .unwrap_or_else(|e| panic!("{} {p:?}: {e}", kernel.name()));
    hopper_fmha::check(&fmha, p, &args);
}

/// The ping-pong order barrier, the K/V ring and the Q ring hold under
/// every schedule: more K/V tiles than stages, so each stage is refilled.
#[test]
fn schedules_and_races() {
    under_schedules(&Problem::new(1, 128, 128 * 6, 1, 1, 64), false, &[1, 2]);
}

/// The persistent kernel's tile loop under every schedule: a CTA runs two
/// tiles, so the pipelines' states and the order barrier carry over.
#[test]
fn persistent_schedules_and_races() {
    under_schedules(&Problem::new(1, 128, 256, 4, 1, 64), true, &[1, 2]);
}

/// What the configuration derives, against the example's numbers.
#[test]
fn configuration_follows_the_example() {
    let p = Problem::new(1, 128, 128, 1, 1, 128);
    let fmha = fmha_for(&p, (64, 128), F16, F16, false);
    assert_eq!(fmha.cta_tiler, (128, 128, 128));
    assert_eq!((fmha.qk_mma_tiler, fmha.pv_mma_tiler), ((64, 128, 128), (64, 128, 128)));
    assert_eq!((fmha.q_stage, fmha.kv_stage, fmha.epi_stage), (2, 5, 2));
    assert_eq!((fmha.threads_per_cta, fmha.num_regs_load, fmha.num_regs_mma), (384, 24, 240));
    assert_eq!(fmha.epi_tile, (64, 32));
    // CTA tile M is twice the MMA tile M.
    let fmha = fmha_for(&p, (64, 64), F16, F16, false);
    assert_eq!(fmha.cta_tiler, (128, 64, 128));
    assert_eq!(fmha.launch(300, 4, 2, 132).grid, [3, 4, 2]);
    assert_eq!(fmha_for(&p, (64, 64), F16, F16, true).launch(300, 4, 2, 5).grid, [5, 1, 1]);
    assert_eq!(fmha_for(&p, (64, 64), F16, F16, true).launch(300, 4, 2, 132).grid, [24, 1, 1]);
    // run()'s mask choice.
    assert_eq!(Problem::new(1, 64, 128, 1, 1, 64).mask(128), (MaskEnum::WindowMask, false, false));
    assert_eq!(Problem::new(1, 64, 129, 1, 1, 64).mask(128), (MaskEnum::ResidualMask, false, false));
    assert_eq!(Problem::new(1, 64, 129, 1, 1, 64).causal().mask(128), (MaskEnum::WindowMask, false, true));
}

/// The kernel's instructions: the register-A WGMMA, the ping-pong order
/// barrier's waits, TMA loads and stores, `stmatrix`, and it needs sm_90a.
#[test]
fn kernel_ir_and_target() {
    use cute::ir::Op;
    let p = Problem::new(1, 128, 128, 1, 1, 64);
    let kernel = fmha_for(&p, (64, 128), F16, F16, false).build();
    let ir = kernel.ir();
    assert!(ir.count_ops(|op| matches!(op, Op::Wgmma { .. })) > 0);
    assert!(ir.count_ops(|op| matches!(op, Op::StMatrix { .. })) > 0);
    assert!(ir.count_ops(|op| matches!(op, Op::TmaLoad { .. })) > 0);
    let e = cute::validate::check_target(ir, Arch::Sm90).unwrap_err().to_string();
    assert!(e.contains("sm_90a"), "{e}");
    cute::validate::check_target(ir, Arch::Sm90a).unwrap();
}

/// The sm_90a PTX of each configuration with each installed `mlir-opt`, and,
/// when `ptxas` is installed, its assembly for sm_90a with `-v`.
#[test]
fn ptx_and_ptxas_for_sm_90a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let causal = |d: usize| Problem::new(1, 256, 256, 2, 1, d).causal();
    let kernels = [
        fmha_for(&Problem::new(1, 128, 128, 1, 1, 64), (64, 128), F16, F16, false),
        fmha_for(&causal(64), (64, 128), F16, F16, true),
        fmha_for(&causal(128), (64, 128), DType::BF16, DType::BF16, false),
        fmha_for(&Problem::new(1, 130, 200, 2, 1, 128), (64, 64), F16, F16, true),
        fmha_for(&Problem { window: (Some(40), Some(8)), ..causal(64) }, (64, 128), F16, F16, false),
    ];
    for fmha in &kernels {
        let kernel = fmha.build();
        let name = kernel.name().to_string();
        let ptx = kernel.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{name}: {e}"));
        for pattern in [
            "wgmma.mma_async.sync.aligned",
            "stmatrix.sync.aligned",
            "fence.proxy.async.shared::cta",
            "cp.async.bulk.commit_group",
            "cp.async.bulk.wait_group.read",
            "mbarrier.try_wait.parity",
            "setmaxnreg.dec.sync.aligned.u32",
            "setmaxnreg.inc.sync.aligned.u32",
            "cp.async.bulk.tensor.4d.shared::cluster.global",
            "cp.async.bulk.tensor.4d.global.shared::cta",
            "ex2.approx",
        ] {
            assert!(ptx.contains(pattern), "{name}: no {pattern}");
        }
        // Both forms of WGMMA: A and B from descriptors (`Q K^T`), and A as
        // four registers (`P V`).
        let wgmma: Vec<&str> = ptx.lines().filter(|l| l.starts_with("wgmma.mma_async.sync.aligned.m64n")).collect();
        let reg_a = wgmma.iter().filter(|l| l.contains("}, {%r")).count();
        assert!(reg_a > 0 && reg_a < wgmma.len(), "{name}: {reg_a} register-A of {} WGMMAs", wgmma.len());
        let Some(ptxas) = find_ptxas() else { continue };
        let path = std::env::temp_dir().join(format!("cute-hopper-fmha-{}-{name}.ptx", std::process::id()));
        std::fs::write(&path, &ptx).unwrap();
        let out = std::process::Command::new(&ptxas)
            .args(["-arch=sm_90a", "-v", "-o", "/dev/null"])
            .arg(&path)
            .output()
            .expect("running ptxas");
        let _ = std::fs::remove_file(&path);
        let log = String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{name}: ptxas failed:\n{log}");
        assert!(!log.contains("serialized"), "{name}: ptxas serialised the warpgroup MMAs:\n{log}");
        let info = |w: &str| log.lines().filter(|l| l.contains(w)).map(str::trim).collect::<Vec<_>>().join(" ");
        eprintln!("{name}: {} {}", info("registers"), info("spill"));
        eprintln!(
            "{name}: PTX: {} wgmma.mma_async, {} try_wait, {} setmaxnreg, {} stmatrix",
            ptx.matches("wgmma.mma_async.sync.aligned").count(),
            ptx.matches("mbarrier.try_wait.parity").count(),
            ptx.matches("setmaxnreg").count(),
            ptx.matches("stmatrix.sync.aligned").count(),
        );
    }
}

/// `ptxas` on `PATH`.
fn find_ptxas() -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("ptxas")).find(|p| p.is_file())
}
