//! The TMA pipelines on the simulator: the
//! `PipelineTmaAsync` and `PipelineTmaStore` feature kernels under the built-in and
//! seeded schedules with random async completion, the signalling lanes of
//! `init_empty_barrier_arrive_signal`, the multicast consumer count, the
//! `TmaStoreFence`, and what each emits.

mod feature_kernels;

use cute::ir::Op;
use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineTmaAsync, PipelineTmaAsyncOptions, PipelineTmaStore, SyncObject,
    TmaStoreFence, init_empty_barrier_arrive_signal,
};
use cute::prelude::*;

/// Every TMA pipeline kernel under the built-in schedules and 8 seeded ones with
/// random completion: each matches plain Rust.
#[test]
fn tma_pipelines_match_rust_under_seeded_schedules() {
    let seeds: Vec<u64> = (0..8).collect();
    for case in feature_kernels::tma_pipeline_cases_sm90() {
        let mut args = feature_kernels::inputs_with_len(&case.kernel, true, 256);
        let report = case
            .kernel
            .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
        if case.name.starts_with("tma_store") {
            assert_eq!(report.tma_stores, 6, "{}: {report}", case.name);
        } else {
            assert!(report.tma_instructions > 0, "{}: {report}", case.name);
        }
        let mut args = feature_kernels::inputs_with_len(&case.kernel, true, 256);
        case.kernel
            .simulate_schedules(case.cfg, &mut args, &SimOptions::default(), &seeds)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
    }
}

const MODES: [(u32, u32); 3] = [(1, 0), (0, 1), (1, 1)];

/// For a `(cm, cn)` cluster (`(1, cm, cn, 1)` column-major), every thread of every CTA
/// writes the destination rank and signalling flag of
/// `init_empty_barrier_arrive_signal` for each multicast mode.
fn signal_kernel(cm: usize, cn: usize) -> Kernel {
    Kernel::builder(format!("arrive_signal_{cm}x{cn}")).block_dim([64, 1, 1]).build(move || {
        let size = cm * cn;
        let out = param_tensor::<i32>("out", Layout::col_major(size * 64 * 6));
        let vmnk = Layout::col_major((1usize, cm, cn, 1usize));
        let t = thread_idx().x;
        let at = (block_idx_in_cluster() * 64 + t) * 6;
        for (i, mode) in MODES.into_iter().enumerate() {
            let (dst, sig) = init_empty_barrier_arrive_signal(&vmnk, t, mode);
            out.store(at + (2 * i) as i32, dst);
            out.store(at + (2 * i + 1) as i32, sig.to::<i32>());
        }
    })
}

/// [`signal_kernel`]'s output in plain Rust: lane `l` signals CTA `l % size` when
/// `l < size` and that CTA shares the multicast mode's coordinates.
fn signal_reference(cm: usize, cn: usize) -> Vec<i32> {
    let size = cm * cn;
    let mut out = Vec::new();
    for rank in 0..size {
        for t in 0..64 {
            let lane = t % 32;
            let dst = lane % size;
            let (same_m, same_n) = (dst % cm == rank % cm, dst / cm == rank / cm);
            for (m, n) in MODES {
                let shares = match (m, n) {
                    (1, 0) => same_m,
                    (0, 1) => same_n,
                    _ => same_m || same_n,
                };
                out.push(dst as i32);
                out.push(i32::from(lane < size && shares));
            }
        }
    }
    out
}

/// The signalling lanes of 2x1, 2x2, 4x2 and 1x4 clusters for each multicast mode,
/// against plain Rust: lane `i` signals CTA `i` when it shares this CTA's M
/// coordinate (multicast along M), N coordinate (along N), or either (both).
#[test]
fn signalling_lanes_follow_the_multicast_modes() {
    for (cm, cn) in [(2, 1), (2, 2), (4, 2), (1, 4)] {
        let kernel = signal_kernel(cm, cn);
        let size = (cm * cn) as u32;
        let mut args = vec![KernelArg::buffer(&vec![0i32; size as usize * 64 * 6])];
        let cfg = LaunchConfig::new([cm as u32, cn as u32, 1], [64, 1, 1]).with_cluster([cm as u32, cn as u32, 1]);
        kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{cm}x{cn}: {e}"));
        assert_eq!(args[0].to_vec::<i32>(), signal_reference(cm, cn), "{cm}x{cn}");
    }
}

/// With one CTA only lane 0 of each warp signals, on its own CTA: the flag folds to
/// `lane < 1` and no coordinate is computed.
#[test]
fn a_single_cta_signals_from_lane_zero() {
    let kernel = Kernel::builder("arrive_signal_1").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(64 * 2));
        let t = thread_idx().x;
        let (dst, sig) =
            init_empty_barrier_arrive_signal(&Layout::col_major((1usize, 1usize, 1usize, 1usize)), t, (1, 1));
        out.store(t * 2, dst);
        out.store(t * 2 + 1, sig.to::<i32>());
    });
    assert_eq!(kernel.ir().count_ops(|op| matches!(op, Op::Special { .. })), 1, "only thread_idx is read");
    let mut args = vec![KernelArg::buffer(&[0i32; 128])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
    let want: Vec<i32> = (0..64).flat_map(|t| [0, i32::from(t % 32 == 0)]).collect();
    assert_eq!(args[0].to_vec::<i32>(), want);
}

/// The empty barriers' arrival count: the consumer group as given, or with
/// `enable_multicast_signaling` one arrival for each warp of consumers and each CTA multicasting to this one
/// (`(mcast_m + mcast_n - overlap) * warps`).
#[test]
fn multicast_signaling_counts_warps_times_partners() {
    let count = |cm: usize, cn: usize, mode: (u32, u32), auto: bool, consumers: CooperativeGroup| {
        let got = std::cell::Cell::new(0);
        Kernel::builder("count").block_dim([384, 1, 1]).build(|| {
            let p = PipelineTmaAsync::create(
                2,
                CooperativeGroup::new(Agent::Thread, 1),
                consumers,
                1024,
                PipelineTmaAsyncOptions {
                    cta_layout_vmnk: Some(Layout::col_major((1usize, cm, cn, 1usize))),
                    mcast_mode_mn: mode,
                    enable_multicast_signaling: auto,
                    defer_sync: true,
                    ..Default::default()
                },
            );
            got.set(p.sync_object_empty().arrive_count().as_static().unwrap());
            assert_eq!(p.sync_object_full().tx_count().as_static(), Some(1024));
            assert_eq!(p.consumer_group().unwrap().agent(), if auto { Agent::Thread } else { consumers.agent() });
        });
        got.get()
    };
    let wg = |n| CooperativeGroup::new(Agent::Thread, n);
    assert_eq!(count(1, 1, (1, 1), false, wg(4)), 4);
    assert_eq!(count(1, 1, (1, 1), true, wg(128)), 4);
    assert_eq!(count(2, 1, (1, 1), true, wg(128)), 8);
    assert_eq!(count(2, 2, (1, 1), true, wg(256)), 3 * 8);
    // Along M only: the cluster's N extent; along N only: its M extent.
    assert_eq!(count(2, 4, (1, 0), true, wg(128)), 4 * 4);
    assert_eq!(count(2, 4, (0, 1), true, wg(128)), 2 * 4);
    assert_eq!(count(2, 2, (1, 1), true, CooperativeGroup::new(Agent::Warp, 8)), 3 * 8);
    assert_eq!(count(2, 2, (1, 1), false, CooperativeGroup::new(Agent::Warp, 1)), 32);
}

/// What the pipelines emit: `PipelineTmaAsync`'s acquire arms the full barrier with
/// expect_tx in an elected lane and its commit emits nothing; a cluster pipeline's
/// release arrives through `mapa` at cluster scope and its init is a relaxed cluster
/// arrive and a wait; `TmaStoreFence` and `PipelineTmaStore` are bulk groups only.
#[test]
fn pipelines_emit_the_expected_operations() {
    let kernel = Kernel::builder("emit").block_dim([64, 1, 1]).build(|| {
        let p = PipelineTmaAsync::create(
            2,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, 2),
            256,
            PipelineTmaAsyncOptions {
                cta_layout_vmnk: Some(Layout::col_major((1usize, 2usize, 1usize, 1usize))),
                ..Default::default()
            },
        );
        let state = cute::pipeline::make_pipeline_state(cute::pipeline::PipelineUserType::Producer, 2);
        p.producer_acquire(&state, None);
        p.consumer_release(&state);
    });
    let ir = kernel.ir();
    assert_eq!(ir.count_ops(|op| matches!(op, Op::MbarrierArrive { expect_tx: Some(_), peer: None, .. })), 1);
    assert_eq!(
        ir.count_ops(|op| matches!(
            op,
            Op::MbarrierArrive {
                expect_tx: None,
                peer: Some(_),
                relaxed: false,
                scope: cute::ir::MemScopeKind::Cluster,
                ..
            }
        )),
        1
    );
    assert_eq!(ir.count_ops(|op| matches!(op, Op::ClusterArrive { relaxed: true, .. })), 1);
    assert_eq!(ir.count_ops(|op| matches!(op, Op::ClusterWait { .. })), 1);
    // producer_commit adds no operation.
    let ops = |commit: bool| {
        Kernel::builder("commit")
            .block_dim([32, 1, 1])
            .build(move || {
                let p = PipelineTmaAsync::create(
                    2,
                    CooperativeGroup::new(Agent::Thread, 1),
                    CooperativeGroup::new(Agent::Thread, 1),
                    256,
                    PipelineTmaAsyncOptions::default(),
                );
                if commit {
                    p.producer_commit(&cute::pipeline::make_pipeline_state(
                        cute::pipeline::PipelineUserType::Producer,
                        2,
                    ));
                }
            })
            .ir()
            .count_ops(|_| true)
    };
    assert_eq!(ops(true), ops(false));
    let kernel = Kernel::builder("emit_store").block_dim([32, 1, 1]).build(|| {
        let fence = TmaStoreFence::new(3);
        SyncObject::arrive_and_wait(&fence, (), ());
        fence.tail();
        let p = PipelineTmaStore::create(2, CooperativeGroup::new(Agent::Warp, 1));
        p.producer_commit();
        p.producer_acquire();
        p.producer_tail();
        assert_eq!(SyncObject::max(&fence), None);
    });
    let ir = kernel.ir();
    assert_eq!(ir.count_ops(|op| matches!(op, Op::CpAsyncBulkCommitGroup)), 2);
    let waits =
        |n: u32| ir.count_ops(|op| matches!(op, Op::CpAsyncBulkWaitGroup { pending, read: true } if *pending == n));
    assert_eq!((waits(0), waits(1), waits(2)), (2, 1, 1));
}
