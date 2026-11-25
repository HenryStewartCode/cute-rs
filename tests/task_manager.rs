// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The copy kernels follow NVIDIA CUTLASS 4.8.1's BSD-3-Clause Task Scheduling
// tutorials (01_copy_grid_stride.py, 02_copy_tma.py,
// 01_copy_persistent_skip_tile.py); see examples/ts_copy_*.rs,
// tests/ts_kernels/mod.rs and THIRD-PARTY-NOTICES.md.

//! Task Scheduling part 2: tasks, the
//! TaskManager's static checks, kernel emission on the simulator (with seeded
//! interleavings), PDL barrier resources, IKET ranges per step, and the
//! diagnostics.

#[allow(dead_code)]
#[path = "../examples/ts_copy_grid_stride.rs"]
mod ts_copy_grid_stride;
#[allow(dead_code)]
#[path = "../examples/ts_copy_persistent_skip_tile.rs"]
mod ts_copy_persistent_skip_tile;
#[allow(dead_code)]
mod ts_kernels;

use std::path::PathBuf;
use std::process::Command;

use cute::pipeline::{Agent, CooperativeGroup};
use cute::prelude::*;
use cute::task_scheduling::*;
use cute::task_scheduling::{Schedule, schedule};
use ts_kernels::*;

fn nothing(_: &StageInfo, _: &WorkArgs) -> WorkResults {
    WorkResults::none()
}

// ---------------------------------------------------------------------------
// The ported examples and the two-task pipelined copy on the simulator
// ---------------------------------------------------------------------------

#[test]
fn grid_stride_example_on_the_simulator() {
    use ts_copy_grid_stride as gs;
    gs::run(gs::NUM_ENTRIES, gs::NUM_BLOCKS, gs::NUM_WARPS);
    // Unrolled, under the built-in orders and two seeded interleavings.
    let src = gs::source(300);
    let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![0i16; 300])];
    let kernel = gs::build(300, 2, 2, None);
    TaskManager::verify_interleavings(&kernel, gs::launch(2, 2), &mut args, &[1, 2]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<i16>(), src);
}

#[test]
fn persistent_skip_tile_example_on_the_simulator() {
    use ts_copy_persistent_skip_tile as ps;
    ps::run(ps::ROWS, ps::LOGICAL_TILES, ps::OVERLAUNCH_TILES, ps::CTAS);
    let a = ps::matrix(2, 2 * ps::TILE);
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![0.0f32; a.len()])];
    let kernel = ps::build(2, 2, 3);
    TaskManager::verify_interleavings(&kernel, ps::launch(2), &mut args, &[7]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<f32>(), a);
}

#[test]
fn two_task_pipelined_copy_on_the_simulator() {
    let kernel = managed_staged_copy(false);
    let a = ramp_i32(ST_ROWS * ST_COLS);
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![0i32; ST_ROWS * ST_COLS])];
    let cfg = LaunchConfig::new([ST_BLOCKS as u32, 1, 1], [ST_THREADS, 1, 1]);
    TaskManager::verify_interleavings(&kernel, cfg, &mut args, &[3, 4]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<i32>(), a);
}

/// Without a pipeline the store warps read rows the load warp may not have
/// written: the static checks pass, the simulator reports it in the store
/// task's `read_smem` step.
#[test]
fn racy_schedule_is_caught_by_the_interleavings() {
    let kernel = managed_staged_copy(true);
    let a = ramp_i32(ST_ROWS * ST_COLS);
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![0i32; ST_ROWS * ST_COLS])];
    let cfg = LaunchConfig::new([1, 1, 1], [ST_THREADS, 1, 1]);
    let e = TaskManager::verify_interleavings(&kernel, cfg, &mut args, &[5]).unwrap_err();
    assert_eq!(e.task.as_deref(), Some("StoreTask"), "{e}");
    assert!(e.step.as_deref().is_some_and(|s| s.contains("smemResource.read_smem")), "{e}");
    assert!(e.loc.is_some(), "{e}");
}

// ---------------------------------------------------------------------------
// IKET
// ---------------------------------------------------------------------------

#[test]
fn iket_trace_has_one_range_per_task_step() {
    use ts_copy_grid_stride as gs;
    let kernel = gs::build(64, 1, 1, Some(IketOptions { capacity: 64, clock: IketClock::Clock64 }));
    let src = gs::source(64);
    let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&[0i16; 64])];
    let mut module = kernel.sim_module();
    module.simulate(gs::launch(1, 1), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<i16>(), src);
    let words = module.read_global(IKET_GLOBAL).unwrap().to_vec::<u32>();
    let trace = kernel.decode_iket(&words).unwrap_or_else(|e| panic!("{e}"));
    assert!(trace.unmatched.is_empty() && !trace.overflowed, "{:?}", trace.unmatched);
    let events: Vec<String> = trace.warp_records(0, 0).iter().map(|r| format!("{:?}:{}", r.kind, r.event)).collect();
    let mut want = vec!["Start:CopyTask".to_string()];
    for _ in 0..2 {
        for step in ["#0 inputGmemResource.get_item", "#1 outputGmemResource.set_item"] {
            want.push(format!("Start:{step}"));
            want.push(format!("End:{step}"));
        }
    }
    want.push("End:CopyTask".into());
    assert_eq!(events, want);
    assert!(kernel.symbols().to_json(&[]).contains("CopyTask"));
}

// ---------------------------------------------------------------------------
// PDL barrier resources
// ---------------------------------------------------------------------------

const PDL_N: usize = 64;

/// The primary (`y = 2x`, then `launch_griddep`) or the dependent (`z = y +
/// 1`, after `wait_griddep`). Both wait first: a launch must follow a wait.
/// With `launch_first`, the primary launches before it waits.
fn pdl_kernel(primary: bool, launch_first: bool) -> Kernel {
    let name = if primary { "ts_pdl_primary" } else { "ts_pdl_dependent" };
    Kernel::builder(name).block_dim([PDL_N as u32, 1, 1]).target(Arch::Sm90a).build(move || {
        let src = param_tensor::<f32>("src", Layout::col_major(PDL_N));
        let dst = param_tensor::<f32>("dst", Layout::col_major(PDL_N));
        let v = TaskLocalVariable::<Float32>::new("v", || Float32::constant(0.0), "the element");
        let input = MemoryResource::new("input").work(Work::consumer("load").returns(&v).run({
            let v = v.clone();
            move |_, _| WorkResults::of(&v, src.load(thread_idx().x))
        }));
        let output =
            MemoryResource::new("output").work(Work::producer("store").input::<Float32>("v").run(move |_, args| {
                let x: Float32 = args.get("v");
                dst.store(thread_idx().x, if primary { x * 2.0f32 } else { x + 1.0f32 });
                WorkResults::none()
            }));
        let wait = PdlWaitBarrier::new("pdl_wait");
        let launch = PdlLaunchBarrier::new("pdl_launch");
        let res: Vec<&dyn Resource> = vec![&wait, &input, &output, &launch];
        let s = schedule("pdl_schedule", &res, |_, h| {
            let (wait, input, output, launch) = (h[0], h[1], h[2], h[3]);
            if primary && launch_first {
                launch.call("launch_griddep");
            }
            wait.call("wait_griddep");
            let x = input.call("load").one();
            output.call_with("store", [("v", &x)]);
            if primary && !launch_first {
                launch.call("launch_griddep");
            }
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let dst: &[&dyn Resource] = if primary { &[&output, &launch] } else { &[&output] };
        let task = Task::new("PdlTask", &[&wait, &input], dst, 0, (PDL_N / 32) as u32, s);
        let graph = ResourceDependencyGraph::new()
            .depends_on(&input, [DependencyGraphControlEdge::new(&wait)])
            .depends_on(&output, [&input]);
        let mut tm =
            TaskManager::new(vec![task], graph).resource(wait).resource(input).resource(output).resource(launch);
        tm.setup_resources_and_tasks().unwrap_or_else(|e| panic!("{e}"));
        tm.run().unwrap_or_else(|e| panic!("{e}"));
    })
}

fn find_ptxas() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("ptxas")).find(|p| p.is_file())
}

#[test]
fn pdl_pair_through_the_stream_model_and_ptxas() {
    let (primary, dependent) = (pdl_kernel(true, false), pdl_kernel(false, false));
    let x: Vec<f32> = (0..PDL_N).map(|i| i as f32 - 7.0).collect();
    let mut stream = SimStream::new();
    let xb = stream.buffer(KernelArg::buffer(&x));
    let yb = stream.buffer(KernelArg::buffer(&[0.0f32; PDL_N]));
    let zb = stream.buffer(KernelArg::buffer(&[0.0f32; PDL_N]));
    let cfg = LaunchConfig::new([1, 1, 1], [PDL_N as u32, 1, 1]);
    stream.launch(&primary, cfg, &[xb.into(), yb.into()]).unwrap_or_else(|e| panic!("{e}"));
    stream.launch(&dependent, cfg.with_pdl(true), &[yb.into(), zb.into()]).unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<f32> = x.iter().map(|v| v * 2.0 + 1.0).collect();
    assert_eq!(stream.get(zb).to_vec::<f32>(), want);

    let mlir = primary.to_mlir();
    assert!(mlir.contains("griddepcontrol.launch_dependents"), "{mlir}");
    assert!(dependent.to_mlir().contains("griddepcontrol.wait"));
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping the PTX: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let options = CompileOptions::new().gpu_arch(Arch::Sm90a);
    let compiled: Vec<_> =
        [&primary, &dependent].map(|k| k.compile_with(&toolchain, &options).unwrap_or_else(|e| panic!("{e}"))).into();
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping the assembly: no ptxas on PATH");
        return;
    };
    for k in &compiled {
        let stem = std::env::temp_dir().join(format!("cute-ts-pdl-{}-{}", std::process::id(), k.name()));
        let (source, cubin) = (stem.with_extension("ptx"), stem.with_extension("cubin"));
        std::fs::write(&source, k.ptx()).unwrap();
        let out = Command::new(&ptxas).arg("-arch=sm_90a").arg(&source).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&cubin);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }
}

#[test]
fn launch_dependents_before_any_wait_is_refused() {
    let e = std::panic::catch_unwind(|| pdl_kernel(true, true)).expect_err("the build should fail");
    let msg = e.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("pdl_launch.launch_griddep") && msg.contains("before any wait_griddep"), "{msg}");
}

// ---------------------------------------------------------------------------
// Diagnostics (no kernel needed)
// ---------------------------------------------------------------------------

fn group(n: i32) -> CooperativeGroup {
    CooperativeGroup::new(Agent::Thread, n)
}

/// A staged resource between a one-warp producer task and a four-warp
/// consumer task, with a source and a destination resource.
struct Pair {
    src: MemoryResource,
    smem: MemoryResource,
    dst: MemoryResource,
}

fn pair(consumer_threads: i32) -> Pair {
    let v = TaskLocalVariable::<Int32>::scalar("v", 0, "");
    let w = TaskLocalVariable::<Int32>::scalar("w", 0, "");
    let cfg = PipelineConfig::create_async_async_pipeline_cfg(2, group(32), group(consumer_threads), None);
    Pair {
        src: MemoryResource::new("src").work(Work::consumer("get").returns(&v).run(nothing)),
        smem: MemoryResource::new("smem")
            .with_pipeline_config(cfg)
            .work(Work::producer("fill").input::<Int32>("v").run(nothing))
            .work(Work::consumer("drain").returns(&w).run(nothing)),
        dst: MemoryResource::new("dst").work(Work::producer("put").input::<Int32>("w").run(nothing)),
    }
}

fn producer(p: &Pair, rows: i64) -> Schedule {
    schedule("producer", &[&p.src, &p.smem], |s, h| {
        s.domain_loop(0, rows, 1, 1, |_| {
            let v = h[0].call("get").one();
            h[1].acquire();
            h[1].call_with("fill", [("v", &v)]);
            h[1].commit();
        });
    })
    .unwrap()
}

fn consumer(p: &Pair, rows: i64) -> Schedule {
    schedule("consumer", &[&p.smem, &p.dst], |s, h| {
        s.domain_loop(0, rows, 1, 1, |_| {
            h[0].wait();
            let w = h[0].call("drain").one();
            h[0].release();
            h[1].call_with("put", [("w", &w)]);
        });
    })
    .unwrap()
}

fn manager(
    p: Pair,
    prod_rows: i64,
    cons_rows: i64,
    graph: impl FnOnce(&Pair) -> ResourceDependencyGraph,
) -> TaskManager {
    let tasks = vec![
        Task::new("Producer", &[&p.src], &[&p.smem], 4, 1, producer(&p, prod_rows)),
        Task::new("Consumer", &[&p.smem], &[&p.dst], 0, 4, consumer(&p, cons_rows)),
    ];
    let g = graph(&p);
    TaskManager::new(tasks, g).resource(p.src).resource(p.smem).resource(p.dst)
}

fn good_graph(p: &Pair) -> ResourceDependencyGraph {
    ResourceDependencyGraph::new().depends_on(&p.smem, [&p.src]).depends_on(&p.dst, [&p.smem])
}

#[test]
fn a_consistent_pair_verifies() {
    manager(pair(128), 4, 4, good_graph).verify().unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn diagnostics_cyclic_graph() {
    let e = manager(pair(128), 4, 4, |p| good_graph(p).depends_on(&p.src, [&p.dst])).verify().unwrap_err();
    assert!(e.has(VerifyIssueKind::CyclicGraph), "{e}");
    assert!(e.to_string().contains("cycle"), "{e}");
}

#[test]
fn diagnostics_unbalanced_commit_and_release() {
    let e = manager(pair(128), 4, 3, good_graph).verify().unwrap_err();
    assert!(e.has(VerifyIssueKind::UnbalancedPipeline), "{e}");
    assert!(e.to_string().contains("4 iterations") && e.to_string().contains("3 iterations"), "{e}");
}

#[test]
fn diagnostics_thread_count_mismatch() {
    let e = manager(pair(96), 4, 4, good_graph).verify().unwrap_err();
    assert!(e.has(VerifyIssueKind::ThreadCount), "{e}");
    assert!(e.to_string().contains("96") && e.to_string().contains("128"), "{e}");
}

#[test]
fn diagnostics_variable_flow_and_graph_coverage() {
    // The data edge smem <- src is missing (only a control edge).
    let e = manager(pair(128), 4, 4, |p| {
        ResourceDependencyGraph::new()
            .depends_on(&p.smem, [DependencyGraphControlEdge::new(&p.src)])
            .depends_on(&p.dst, [&p.smem])
    })
    .verify()
    .unwrap_err();
    assert!(e.has(VerifyIssueKind::VariableFlow), "{e}");
    // `dst` left out of the graph.
    let e = manager(pair(128), 4, 4, |p| ResourceDependencyGraph::new().depends_on(&p.smem, [&p.src]))
        .verify()
        .unwrap_err();
    assert!(e.has(VerifyIssueKind::BadGraph), "{e}");
}

#[test]
fn diagnostics_roles_and_warps() {
    let p = pair(128);
    // The consumer declares smem as a destination: its waits are on the wrong side.
    let tasks = vec![
        Task::new("Producer", &[&p.src], &[&p.smem], 4, 1, producer(&p, 4)),
        Task::new("Consumer", &[], &[&p.smem, &p.dst], 1, 4, consumer(&p, 4)),
    ];
    let g = good_graph(&p);
    let e = TaskManager::new(tasks, g).resource(p.src).resource(p.smem).resource(p.dst).verify().unwrap_err();
    assert!(e.has(VerifyIssueKind::RoleMismatch), "{e}");
    assert!(e.has(VerifyIssueKind::BadTask), "{e}"); // warps 1..5 overlap warp 4
}

#[test]
fn diagnostics_wait_before_init() {
    // run() before setup: the barriers would be used uninitialised.
    let e = manager(pair(128), 4, 4, good_graph).run().unwrap_err();
    assert!(e.has(VerifyIssueKind::BarrierInit), "{e}");
    // A resource the tasks use but the manager does not own is never created.
    let p = pair(128);
    let tasks = vec![
        Task::new("Producer", &[&p.src], &[&p.smem], 4, 1, producer(&p, 4)),
        Task::new("Consumer", &[&p.smem], &[&p.dst], 0, 4, consumer(&p, 4)),
    ];
    let e = TaskManager::new(tasks, good_graph(&p)).resource(p.src).resource(p.dst).verify().unwrap_err();
    assert!(e.has(VerifyIssueKind::BarrierInit), "{e}");
}

#[test]
fn diagnostics_pdl_ordering() {
    let wait = PdlWaitBarrier::new("pdl_wait");
    let out = MemoryResource::new("out").work(Work::producer("write").run(nothing));
    let s = schedule("s", &[&wait, &out], |_, h| {
        h[1].call("write");
        h[0].call("wait_griddep");
    })
    .unwrap();
    let task = Task::new("T", &[&wait], &[&out], 0, 1, s);
    let g = ResourceDependencyGraph::new().depends_on(&out, [DependencyGraphControlEdge::new(&wait)]);
    let e = TaskManager::new(vec![task], g).resource(wait).resource(out).verify().unwrap_err();
    assert!(e.has(VerifyIssueKind::PdlOrdering), "{e}");
}

/// A launch in the consumer task is fine when the producer task waited
/// before committing the stage the consumer waits on.
#[test]
fn pdl_launch_ordered_through_a_pipeline() {
    let p = pair(128);
    let (wait, launch) = (PdlWaitBarrier::new("pdl_wait"), PdlLaunchBarrier::new("pdl_launch"));
    let build = |producer_waits: bool| {
        let prod = schedule("producer", &[&p.src, &p.smem, &wait], |s, h| {
            if producer_waits {
                h[2].call("wait_griddep");
            }
            s.domain_loop(0, 4, 1, 1, |_| {
                let v = h[0].call("get").one();
                h[1].acquire();
                h[1].call_with("fill", [("v", &v)]);
                h[1].commit();
            });
        })
        .unwrap();
        let cons = schedule("consumer", &[&p.smem, &p.dst, &launch], |s, h| {
            s.domain_loop(0, 4, 1, 1, |d| {
                h[0].wait();
                let w = h[0].call("drain").one();
                h[0].release();
                h[1].call_with("put", [("w", &w)]);
                d.first_iter(|| {
                    h[2].call("launch_griddep");
                });
            });
        })
        .unwrap();
        let src: &[&dyn Resource] = if producer_waits { &[&p.src, &wait] } else { &[&p.src] };
        let tasks = vec![
            Task::new("Producer", src, &[&p.smem], 4, 1, prod),
            Task::new("Consumer", &[&p.smem], &[&p.dst, &launch], 0, 4, cons),
        ];
        let mut g = good_graph(&p);
        if producer_waits {
            g = g.depends_on(&p.src, [DependencyGraphControlEdge::new(&wait)]);
        }
        // The manager owns fresh resources with the same names (verify only
        // looks at their descriptions).
        let fresh = pair(128);
        let tm = TaskManager::new(tasks, g)
            .resource(fresh.src)
            .resource(fresh.smem)
            .resource(fresh.dst)
            .resource(PdlLaunchBarrier::new("pdl_launch"));
        if producer_waits { tm.resource(PdlWaitBarrier::new("pdl_wait")) } else { tm }
    };
    build(true).verify().unwrap_or_else(|e| panic!("{e}"));
    let e = build(false).verify().unwrap_err();
    assert!(e.has(VerifyIssueKind::PdlLaunchBeforeWait), "{e}");
}

#[test]
fn launch_barrier_is_never_downstream() {
    let launch = PdlLaunchBarrier::new("pdl_launch");
    let out = MemoryResource::new("out").work(Work::producer("write").run(nothing));
    let s = schedule("s", &[&out], |_, h| {
        h[0].call("write");
    })
    .unwrap();
    let task = Task::new("T", &[], &[&out], 0, 1, s);
    let g = ResourceDependencyGraph::new().depends_on(&launch, [&out]);
    let e = TaskManager::new(vec![task], g).resource(launch).resource(out).verify().unwrap_err();
    assert!(e.has(VerifyIssueKind::BadGraph), "{e}");
}
