// SPDX-License-Identifier: MIT OR Apache-2.0

//! cute_ext's GEMM building blocks on the simulator,
//! against plain Rust: the persistent SM100 GEMM port (`examples/cute_ext_dense_gemm_persistent.rs`)
//! with 1SM on a (1, 1) cluster, 2SM on (2, 1), 2SM with multicast on (2, 2), a ragged
//! problem, `get_cta_v_map_ab`'s maps given to the loads, random data within Float32's bound,
//! preferred and fallback clusters (every cluster preferred, every one fallback, mixed), and
//! seeded schedules with the race detector; `epilogue_tma_store_drain` over an overlapping
//! accumulator plan with `amax_out`; `mainloop_mma_sm120` on one SM120 warp; the layouts;
//! sm_100a and sm_120a PTX (assembled when `ptxas` is installed); diagnostics last.

#[allow(dead_code)]
#[path = "../examples/cute_ext_dense_gemm_persistent.rs"]
mod gemm;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::experimental::{
    self, AccumulatorBufferingKind, AccumulatorBufferingOptions, AccumulatorBufferingPlan, EpilogueTmaStoreOptions,
    GenericPipeline, MmaOperand, OperationType, TMAStorePipeline, TMAToUMMAPipeline, plan_accumulator_buffering,
};
use cute::prelude::*;
use cute::tcgen05::{CtaGroup, OperandSource, Repetition, St32x32bOp, make_tmem_copy};
use cute::utils::sm100::{make_smem_layout_a, make_smem_layout_b, make_trivial_tiled_mma};
use cute::utils::{LayoutEnum, compute_epilogue_tile_shape};
use gemm::{ONE_SM, ONE_SM_FALLBACK, PersistentDenseGemmConfig, TWO_SM, TWO_SM_MULTICAST};

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn in_kernel(threads: u32, f: impl FnOnce() + 'static) -> String {
    build_panic(move || {
        Kernel::builder("diag").block_dim([threads, 1, 1]).target(Arch::Sm100a).build(f);
    })
}

// ---------------------------------------------------------------------------------------------
// (1) The persistent GEMM
// ---------------------------------------------------------------------------------------------

#[test]
fn persistent_gemm_configurations_match_rust() {
    // 2 x 2 CTA tiles on one or two persistent clusters; 2 k-tiles.
    let r = gemm::run(&ONE_SM, (256, 128, 128), 2, true);
    assert_eq!(r.tcgen05_mma_instructions, 4 * 2 * 4);
    let r = gemm::run(&TWO_SM, (256, 128, 128), 2, true);
    assert_eq!(r.tcgen05_mma_instructions, 2 * 2 * 4);
    gemm::run(&TWO_SM_MULTICAST, (256, 128, 128), 1, true);
    // Ragged in M, N and K, with fewer clusters than tiles.
    gemm::run(&ONE_SM, (200, 100, 72), 1, true);
    gemm::run(&TWO_SM_MULTICAST, (300, 92, 40), 1, true);
}

#[test]
fn explicit_cta_v_maps_and_random_data() {
    let explicit = |c: PersistentDenseGemmConfig| PersistentDenseGemmConfig { explicit_cta_v_map: true, ..c };
    gemm::run(&explicit(ONE_SM), (256, 128, 128), 2, true);
    gemm::run(&explicit(TWO_SM_MULTICAST), (256, 128, 128), 1, true);
    gemm::run(&ONE_SM, (128, 64, 256), 1, false);
}

#[test]
fn preferred_and_fallback_clusters() {
    let two_sm_fallback = PersistentDenseGemmConfig {
        name: "cute_ext_dense_gemm_2sm_fallback",
        fallback_cluster_shape_mn: Some((2, 1)),
        ..TWO_SM_MULTICAST
    };
    for cfg in [ONE_SM_FALLBACK, two_sm_fallback] {
        for cluster_shapes in [ClusterShapes::Preferred, ClusterShapes::Fallback, ClusterShapes::Mixed { seed: 3 }] {
            let opts = SimOptions { cluster_shapes, ..SimOptions::default() };
            gemm::run_with(&cfg, (256, 200, 64), 2, true, &opts);
        }
    }
}

#[test]
fn seeded_schedules_with_the_race_detector() {
    for seed in 0..2 {
        let opts = SimOptions {
            schedule: Schedule::Random { seed },
            async_completion: AsyncCompletion::Random(seed),
            ..SimOptions::default()
        };
        gemm::run_with(&TWO_SM_MULTICAST, (256, 128, 128), 1, true, &opts);
        gemm::run_with(&ONE_SM, (200, 100, 72), 1, true, &opts);
    }
}

// ---------------------------------------------------------------------------------------------
// (2) The epilogue with an accumulator drain and amax
// ---------------------------------------------------------------------------------------------

const ROUNDS: usize = 4;
const EPI_N: usize = 64;

fn drain_plan() -> AccumulatorBufferingPlan {
    let one =
        Layout::from_modes([Layout::new((128, EPI_N), (1usize << 16, 1usize)), Layout::new(1, 0), Layout::new(1, 0)]);
    let epi_tile =
        compute_epilogue_tile_shape(&[128, EPI_N], false, LayoutEnum::ROW_MAJOR, DType::F32, None, None, None);
    plan_accumulator_buffering(
        &one,
        DType::F32,
        &AccumulatorBufferingOptions {
            epilogue_tile: Some(epi_tile),
            tmem_capacity_columns: 96,
            allow_overlap: true,
            ..Default::default()
        },
    )
}

/// Warps 4-7 write each round's 128 x 64 tile of `input` into the view the plan hands them
/// (`tcgen05.st`); warps 0-3 drain it with `epilogue_tma_store_drain` (warp 0 storing) into
/// row block `round` of `d`, doubling it, and take the amax into `amax`.
fn drain_kernel(plan: AccumulatorBufferingPlan) -> Kernel {
    Kernel::builder("cute_ext_gemm_drain").block_dim([256, 1, 1]).target(Arch::Sm100a).build(move || {
        let input = param_tensor::<f32>("input", Layout::row_major((ROUNDS, 128, EPI_N)));
        let d = param_tensor_aligned::<f32>("d", Layout::row_major((ROUNDS * 128, EPI_N)), 16);
        let amax = param_tensor::<f32>("amax", Layout::col_major(1));
        let buf = experimental::allocate::<f32>(AddressSpace::Tmem, plan.storage_layout().clone(), 32);
        let pipe = GenericPipeline::create(
            OperationType::Sm100CopyR2t,
            OperationType::Sm100CopyT2r,
            128,
            128,
            plan.synchronization_depth(),
            None,
        );
        let store = TMAStorePipeline::new(2, 128, 1, 0);
        let t = thread_idx().x;
        if_else(
            warp_idx().ge(4),
            || {
                let mut p = for_range_carry(0, ROUNDS as i32, 1, pipe.clone(), |r, mut p| {
                    let view = plan.producer_acquire_and_get_view(&mut p, &buf);
                    let m0 = view.layout().mode(0);
                    let tile = make_tensor(view.ptr(), Layout::from_modes([m0.mode(0), m0.mode(1)]));
                    let src = input.slice((r, .., ..));
                    let st = make_tmem_copy(&make_copy_atom::<f32>(St32x32bOp::new(Repetition::X32)), &tile);
                    let thr = st.get_slice(t - 128);
                    let frag = make_fragment_like::<f32, _>(&thr.partition_s(&src));
                    basic_copy(&thr.partition_s(&src), &frag);
                    copy(&st, &frag, &thr.partition_d(&tile));
                    p.producer_commit_and_advance(false, false);
                    p
                });
                p.producer_tail();
            },
            || {
                let opts = EpilogueTmaStoreOptions { amax_out: Some(amax.clone()), ..Default::default() };
                for_range_carry(0, ROUNDS as i32, 1, (pipe.clone(), store), |r, (mut c, store)| {
                    let (view, drain) = plan.consumer_wait_and_get_view(&mut c, &buf);
                    let (store, c) = experimental::epilogue_tma_store_drain(
                        (128, EPI_N, 64),
                        false,
                        &view,
                        &d,
                        &[r, Int32::from(0)],
                        store,
                        0,
                        |x| x * 2.0f32,
                        drain,
                        &opts,
                    );
                    (c, store)
                });
            },
        );
    })
}

#[test]
fn epilogue_drains_an_overlapping_plan_with_amax() {
    let plan = drain_plan();
    assert_eq!(plan.kind(), AccumulatorBufferingKind::Overlapping, "{plan}");
    let kernel = drain_kernel(plan);
    let input: Vec<f32> = (0..ROUNDS * 128 * EPI_N).map(|i| (i % 1999) as f32 - 1500.0).collect();
    for seed in 0..3 {
        let opts = SimOptions {
            schedule: Schedule::Random { seed },
            async_completion: AsyncCompletion::Random(seed),
            ..SimOptions::default()
        };
        let mut args =
            vec![KernelArg::f32(input.clone()), KernelArg::f32(vec![0.0; input.len()]), KernelArg::f32(vec![0.0])];
        kernel
            .simulate_with(LaunchConfig::new([1, 1, 1], [256, 1, 1]), &mut args, &opts)
            .unwrap_or_else(|e| panic!("{e}"));
        let want: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
        assert_eq!(args[1].as_f32(), want.as_slice());
        assert_eq!(args[2].as_f32(), &[1500.0]);
    }
}

// ---------------------------------------------------------------------------------------------
// (3) mainloop_mma_sm120
// ---------------------------------------------------------------------------------------------

type Fp4 = cute::F4E2M1FN;
type Ue8 = cute::F8E8M0FNU;

const FP4_VALUES: [f32; 15] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
const SCALES: [f32; 3] = [0.5, 1.0, 2.0];

fn sm120_atom() -> MmaAtom {
    cute::warp::MmaMXF4Op::new(DType::F4E2M1FN, DType::F32, DType::F8E8M0FNU).into()
}

/// The scale factors of an `(mn, k)` operand: one per 32 elements of K.
fn sf_layout(mn: usize, k: usize) -> Layout {
    let nsf = k / 32;
    Layout::new((mn, (32, nsf)), (nsf, (0, 1)))
}

/// One warp: fragments of A, B and their scales for two MMAs along K, the accumulator
/// prefilled with 5, then `mainloop_mma_sm120`.
fn sm120_kernel(accumulate: bool) -> Kernel {
    let atom = sm120_atom();
    let [m, n, ka] = atom.shape_mnk();
    let k = 2 * ka;
    Kernel::builder("cute_ext_gemm_sm120").block_dim([32, 1, 1]).target(Arch::Sm120a).build(move || {
        let a = param_tensor::<u8>("a", Layout::new((m, k), (k, 1)));
        let b = param_tensor::<u8>("b", Layout::new((n, k), (k, 1)));
        let sfa = param_tensor::<Ue8>("sfa", sf_layout(m, k));
        let sfb = param_tensor::<Ue8>("sfb", sf_layout(n, k));
        let c = param_tensor::<f32>("c", Layout::new((m, n), (n, 1)));
        let mma = make_tiled_mma(atom.clone(), Layout::col_major((1, 1, 1)));
        let thr = mma.get_slice(lane_idx());
        let (ta, tb) = (thr.partition_a(&a), thr.partition_b(&b));
        let (fa, fb) = (thr.partition_fragment_a::<Fp4, _>(&a), thr.partition_fragment_b::<Fp4, _>(&b));
        for (frag, part) in [(&fa, &ta), (&fb, &tb)] {
            for i in 0..frag.size() {
                let [lo, _]: [Value<Fp4>; 2] = unpack(part.load(i));
                frag.store(i, lo);
            }
        }
        let fsfa = cute::utils::sm120::partition_fragment_sfa(&sfa, &thr);
        let fsfb = cute::utils::sm120::partition_fragment_sfb(&sfb, &thr);
        let (psfa, psfb) =
            (cute::utils::sm120::partition_sfa(&sfa, &thr), cute::utils::sm120::partition_sfb(&sfb, &thr));
        for i in 0..psfa.size() {
            fsfa.store(i, psfa.load(i));
        }
        for i in 0..psfb.size() {
            fsfb.store(i, psfb.load(i));
        }
        let acc = thr.partition_fragment_c::<f32, _>(&c);
        acc.fill(5.0f32);
        experimental::mainloop_mma_sm120(&mma, &fa, &fsfa, &fb, &fsfb, &acc, 2, accumulate);
        let tc = thr.partition_c(&c);
        for i in 0..acc.size() {
            tc.store(i, acc.load(i));
        }
    })
}

fn lcg(s: &mut u64) -> usize {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (*s >> 33) as usize
}

#[test]
fn sm120_inner_loop_matches_rust() {
    let [m, n, ka] = sm120_atom().shape_mnk();
    let k = 2 * ka;
    let mut s = 29u64;
    let mut pick = |set: &[f32]| set[lcg(&mut s) % set.len()];
    let a: Vec<f32> = (0..m * k).map(|_| pick(&FP4_VALUES)).collect();
    let b: Vec<f32> = (0..n * k).map(|_| pick(&FP4_VALUES)).collect();
    let sfa: Vec<f32> = (0..m * k / 32).map(|_| pick(&SCALES)).collect();
    let sfb: Vec<f32> = (0..n * k / 32).map(|_| pick(&SCALES)).collect();
    let byte = |x: &f32| Fp4::from_f32(*x).raw_bits() as u8;
    let nsf = k / 32;
    for accumulate in [false, true] {
        let kernel = sm120_kernel(accumulate);
        let mut args = vec![
            KernelArg::buffer(&a.iter().map(byte).collect::<Vec<u8>>()),
            KernelArg::buffer(&b.iter().map(byte).collect::<Vec<u8>>()),
            KernelArg::buffer(&sfa.iter().map(|&x| Ue8::from_f32(x)).collect::<Vec<_>>()),
            KernelArg::buffer(&sfb.iter().map(|&x| Ue8::from_f32(x)).collect::<Vec<_>>()),
            KernelArg::f32(vec![0.0; m * n]),
        ];
        let r = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.mma_instructions, 2);
        let got = args[4].as_f32();
        for i in 0..m {
            for j in 0..n {
                let start = if accumulate { 5.0 } else { 0.0 };
                let want = (0..k).fold(start, |acc: f32, kk| {
                    acc + a[i * k + kk] * sfa[i * nsf + kk / 32] * b[j * k + kk] * sfb[j * nsf + kk / 32]
                });
                assert_eq!(got[i * n + j], want, "accumulate {accumulate}: C({i}, {j})");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// (4) The layouts
// ---------------------------------------------------------------------------------------------

#[test]
fn tmem_layouts_and_cta_v_maps() {
    Kernel::builder("layouts").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
        let mma = make_trivial_tiled_mma(
            DType::F16,
            DType::F16,
            OperandMajorMode::K,
            OperandMajorMode::K,
            DType::F32,
            CtaGroup::ONE,
            (128, 64),
            OperandSource::SMEM,
        );
        let acc = experimental::make_tmem_layout_acc(&mma, (128, 64, 64), 3);
        assert_eq!((acc.rank(), acc.mode(0).size(), acc.mode(3).size()), (4, 128 * 64, 3), "{acc}");
        let ts = make_trivial_tiled_mma(
            DType::F16,
            DType::F16,
            OperandMajorMode::K,
            OperandMajorMode::K,
            DType::F32,
            CtaGroup::ONE,
            (128, 64),
            OperandSource::TMEM,
        );
        let a = experimental::make_tmem_layout_a(&ts, (128, 64), 2);
        assert_eq!((a.size(), a.mode(a.rank() - 1).size()), (128 * 64 * 2, 2), "{a}");
        let x = param_tensor::<F16>("x", Layout::new((256, 128), (128, 1)));
        let map = experimental::get_cta_v_map_ab(&x, (128, 64, 64), &mma, MmaOperand::A);
        assert_eq!(map.size(), 128 * 64, "{map}");
        let y = param_tensor::<f32>("y", Layout::new((256, 128), (128, 1)));
        let c = experimental::get_cta_v_map_c(&y, (128, 32));
        assert_eq!(c.shape().to_string(), "(128,32)", "{c}");
    });
}

// ---------------------------------------------------------------------------------------------
// (5) sm_100a and sm_120a PTX
// ---------------------------------------------------------------------------------------------

/// Compile `kernel` for `arch` and assemble it when `ptxas` is installed; returns the PTX.
fn assemble(kernel: &Kernel, tc: &Toolchain, arch: Arch, flag: &str) -> String {
    let ptx = kernel.compile_ptx(tc, arch).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    let ptxas = std::env::var("CUTE_PTXAS").unwrap_or_else(|_| "ptxas".into());
    let dir = std::env::temp_dir().join(format!("cute_ext_gemm_{}_{}", kernel.name(), std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("k.ptx");
    std::fs::write(&file, &ptx).unwrap();
    match std::process::Command::new(&ptxas)
        .arg(format!("-arch={flag}"))
        .arg(&file)
        .arg("-o")
        .arg(dir.join("k.cubin"))
        .output()
    {
        Ok(out) => assert!(out.status.success(), "{}: ptxas: {}", kernel.name(), String::from_utf8_lossy(&out.stderr)),
        Err(_) => eprintln!("skipping the assembly: no ptxas found"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    ptx
}

#[test]
fn kernels_compile_for_sm100a_and_sm120a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for cfg in [ONE_SM, TWO_SM_MULTICAST, ONE_SM_FALLBACK] {
        let ptx = assemble(&cfg.build(), &tc, Arch::Sm100a, "sm_100a");
        for want in ["tcgen05.mma", "cp.async.bulk.tensor", "tcgen05.ld", "tcgen05.commit"] {
            assert!(ptx.contains(want), "{}: no {want}", cfg.name);
        }
    }
    let ptx = assemble(&drain_kernel(drain_plan()), &tc, Arch::Sm100a, "sm_100a");
    assert!(ptx.contains("redux.sync") && ptx.contains("cp.async.bulk.tensor"), "drain kernel");
    let inline = tc.llvm_version().is_some_and(|v| v < 22);
    let options = CompileOptions::new().gpu_arch(Arch::Sm120a).mma_inline_ptx(inline);
    let compiled = sm120_kernel(false).compile_with(&tc, &options).unwrap_or_else(|e| panic!("sm120: {e}"));
    assert!(compiled.ptx().contains(".block_scale"), "sm120: no block-scaled MMA");
    let ptxas = std::env::var("CUTE_PTXAS").unwrap_or_else(|_| "ptxas".into());
    let dir = std::env::temp_dir().join(format!("cute_ext_gemm_sm120_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.ptx"), compiled.ptx()).unwrap();
    if let Ok(out) = std::process::Command::new(&ptxas)
        .arg("-arch=sm_120a")
        .arg(dir.join("k.ptx"))
        .arg("-o")
        .arg(dir.join("k.cubin"))
        .output()
    {
        assert!(out.status.success(), "sm120: ptxas: {}", String::from_utf8_lossy(&out.stderr));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------------
// (6) Diagnostics
// ---------------------------------------------------------------------------------------------

/// An epilogue kernel whose store pipeline's barrier counts `arv_count` threads.
fn epilogue_with(arv_count: usize, warp: i32) {
    let mma = make_trivial_tiled_mma(
        DType::F16,
        DType::F16,
        OperandMajorMode::K,
        OperandMajorMode::K,
        DType::F32,
        CtaGroup::ONE,
        (128, 64),
        OperandSource::SMEM,
    );
    let d = param_tensor_aligned::<f32>("d", Layout::new((128, 64), (64, 1)), 16);
    let acc = experimental::allocate::<f32>(
        AddressSpace::Tmem,
        experimental::make_tmem_layout_acc(&mma, (128, 64, 64), 1),
        16,
    );
    let pipe = TMAStorePipeline::new(1, arv_count, 1, 0);
    experimental::epilogue_tma_store(
        (128, 64, 64),
        false,
        &acc.slice((.., .., .., 0)),
        &d,
        &[Int32::from(0), Int32::from(0)],
        pipe,
        warp,
        |x| x,
    );
}

/// A mainloop over buffers of `stages` stages with `mma` and a pipeline of `pipe_stages`
/// stages of `kind`.
fn mainloop_with(two_sm_mma: bool, kind: OperationType, stages: usize, pipe_stages: usize, b_stages: usize) {
    let group = if two_sm_mma { CtaGroup::TWO } else { CtaGroup::ONE };
    let tiler = if two_sm_mma { (256, 64) } else { (128, 64) };
    let mma = make_trivial_tiled_mma(
        DType::F16,
        DType::F16,
        OperandMajorMode::K,
        OperandMajorMode::K,
        DType::F32,
        group,
        tiler,
        OperandSource::SMEM,
    );
    let mnk = (tiler.0, tiler.1, 64);
    let a = experimental::allocate::<F16>(
        AddressSpace::Smem,
        make_smem_layout_a(&mma, mnk, DType::F16, stages, None),
        1024,
    );
    let b = experimental::allocate::<F16>(
        AddressSpace::Smem,
        make_smem_layout_b(&mma, mnk, DType::F16, stages, None),
        1024,
    );
    let acc = experimental::allocate::<f32>(AddressSpace::Tmem, experimental::make_tmem_layout_acc(&mma, mnk, 1), 16);
    let p = TMAToUMMAPipeline::create(pipe_stages, kind, None, None);
    let q = TMAToUMMAPipeline::create(b_stages, kind, None, None);
    experimental::mainloop_mma(&mma, &a, &b, &acc.slice((.., .., .., 0)), 0, 2, 4, p, q, true, false);
}

#[test]
fn diagnostics() {
    // The store pipeline's barrier does not cover the epilogue warpgroup.
    let e = in_kernel(160, || epilogue_with(160, 0));
    assert!(e.contains("counts 160 threads") && e.contains("arv_count 128"), "{e}");
    let e = in_kernel(160, || epilogue_with(128, 1));
    assert!(e.contains("TMA warp is 0, but tma_store_warp_id is 1"), "{e}");
    // Mismatched A/B pipelines.
    let e = in_kernel(192, || mainloop_with(false, OperationType::Sm100Mma1SmSs, 2, 3, 3));
    assert!(e.contains("has 2 stages (its last mode) but its pipeline 3"), "{e}");
    let e = in_kernel(192, || mainloop_with(false, OperationType::Sm100Mma1SmSs, 2, 2, 3));
    assert!(e.contains("has 2 stages (its last mode) but its pipeline 3"), "{e}");
    let e = build_panic(|| {
        Kernel::builder("diag").block_dim([192, 1, 1]).target(Arch::Sm100a).build(|| {
            let mma = make_trivial_tiled_mma(
                DType::F16,
                DType::F16,
                OperandMajorMode::K,
                OperandMajorMode::K,
                DType::F32,
                CtaGroup::ONE,
                (128, 64),
                OperandSource::SMEM,
            );
            let mnk = (128, 64, 64);
            let a = experimental::allocate::<F16>(
                AddressSpace::Smem,
                make_smem_layout_a(&mma, mnk, DType::F16, 2, None),
                1024,
            );
            let b = experimental::allocate::<F16>(
                AddressSpace::Smem,
                make_smem_layout_b(&mma, mnk, DType::F16, 3, None),
                1024,
            );
            let acc =
                experimental::allocate::<f32>(AddressSpace::Tmem, experimental::make_tmem_layout_acc(&mma, mnk, 1), 16);
            let p = TMAToUMMAPipeline::create(2, OperationType::Sm100Mma1SmSs, None, None);
            let q = TMAToUMMAPipeline::create(3, OperationType::Sm100Mma1SmSs, None, None);
            experimental::mainloop_mma(&mma, &a, &b, &acc.slice((.., .., .., 0)), 0, 2, 4, p, q, true, false);
        });
    });
    assert!(e.contains("ab_buffer_same_pipeline") && e.contains("differ"), "{e}");
    let e = build_panic(|| {
        Kernel::builder("diag").block_dim([192, 1, 1]).target(Arch::Sm100a).build(|| {
            let mma = make_trivial_tiled_mma(
                DType::F16,
                DType::F16,
                OperandMajorMode::K,
                OperandMajorMode::K,
                DType::F32,
                CtaGroup::ONE,
                (128, 64),
                OperandSource::SMEM,
            );
            let mnk = (128, 64, 64);
            let a = experimental::allocate::<F16>(
                AddressSpace::Smem,
                make_smem_layout_a(&mma, mnk, DType::F16, 2, None),
                1024,
            );
            let b = experimental::allocate::<F16>(
                AddressSpace::Smem,
                make_smem_layout_b(&mma, mnk, DType::F16, 2, None),
                1024,
            );
            let acc =
                experimental::allocate::<f32>(AddressSpace::Tmem, experimental::make_tmem_layout_acc(&mma, mnk, 1), 16);
            let pair = Layout::col_major((2usize, 1usize, 1usize, 1usize));
            let p = TMAToUMMAPipeline::create(
                2,
                OperationType::Sm100Mma2SmSs,
                Some(OperationType::Sm100TmaLoad2Sm),
                Some(&pair),
            );
            experimental::mainloop_mma(&mma, &a, &b, &acc.slice((.., .., .., 0)), 0, 2, 4, p.clone(), p, true, false);
        });
    });
    assert!(e.contains("disagree on the CTA group"), "{e}");
    // A CTA-V map for an operand the MMA does not have.
    let e = in_kernel(128, || {
        let mma = make_trivial_tiled_mma(
            DType::F16,
            DType::F16,
            OperandMajorMode::K,
            OperandMajorMode::K,
            DType::F32,
            CtaGroup::ONE,
            (128, 64),
            OperandSource::SMEM,
        );
        let x = param_tensor::<F16>("x", Layout::new((256, 128), (128, 1)));
        experimental::get_cta_v_map_ab(&x, (128, 64, 64), &mma, MmaOperand::SFA);
    });
    assert!(e.contains("not a block-scaled MMA") && e.contains("SFA"), "{e}");
}
