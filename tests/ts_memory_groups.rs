// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The UMMA kernel's TMEM epilogue (a `tcgen05.ld.32x32b.x64` copy of each
// accumulator stage, `fence_view_async_tmem_load`, then the release) and the
// MMA issue loop follow NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// (as tests/umma_pipelines.rs does). That portion is under the BSD-3-Clause
// licence, and the rest of the file is cute-rs's own, under MIT OR Apache-2.0.
// THIRD-PARTY-NOTICES.md has the licence text.

//! Task Scheduling part 3 (m11-ts-memory-groups): the declarative SMEM and
//! TMEM allocators on the host (offsets, alignment, reordering, barrier
//! placement, alias groups), the manager's new checks (alias races, group
//! arrival counts, TMEM capacity, barrier storage, group misuse), the ported
//! `02_copy_tma.py` and `01_merge_copy.py` on the simulator with their sm_90a
//! MLIR and PTX, and a TMEM + `TSPipelineAsyncUmma` / `TSPipelineUmmaUmma`
//! kernel on the simulator with its sm_100a PTX. With `CUTE_TS_PTX_DIR` set,
//! the PTX is also written there (for `ptxas` elsewhere).

#[path = "../examples/ts_copy_tma.rs"]
mod ts_copy_tma;

#[path = "../examples/ts_merge_copy.rs"]
mod ts_merge_copy;

use cute::pipeline::{Agent, CooperativeGroup};
use cute::prelude::*;
use cute::task_scheduling::*;

fn threads(n: u32) -> CooperativeGroup {
    CooperativeGroup::new(Agent::Thread, n)
}

fn single() -> Layout {
    Layout::col_major((1usize, 1usize, 1usize, 1usize))
}

fn nothing(_: &StageInfo, _: &WorkArgs) -> WorkResults {
    WorkResults::none()
}

fn async_cfg(stages: usize, producers: u32, consumers: u32) -> PipelineConfig {
    PipelineConfig::create_async_async_pipeline_cfg(stages, threads(producers), threads(consumers), Some(single()))
}

// ---------------------------------------------------------------------------
// The allocators on the host
// ---------------------------------------------------------------------------

#[test]
fn smem_layout_offsets_alignment_barriers_and_reordering() {
    let x = SmemAllocation::new::<u8>("x", 100, 16);
    let y = SmemAllocation::new::<u8>("y", 64, 128);
    let z = SmemAllocation::new::<f32>("z", 10, 1024);
    let r1 = MemoryResource::new("r1")
        .with_pipeline_config(async_cfg(2, 32, 32))
        .smem_requirement(x.clone())
        .smem_requirement(y.clone());
    let r2 = MemoryResource::new("r2").smem_requirement(z.clone());
    let fixed = MemoryResource::new("fixed").with_pipeline_config(async_cfg(3, 32, 32));
    let mut alloc = SmemAllocator::new();
    alloc.add_resource(&r1);
    alloc.add_resource(&r2);
    let layout = alloc.compute_layout().unwrap().clone();
    // Largest alignment first, then declaration order: z, y, x.
    assert_eq!((z.offset(), y.offset(), x.offset()), (0, 128, 192));
    assert_eq!(layout.data_bytes, 292);
    assert_eq!(layout.alignment, 1024);
    // r1's 2 x 2 mbarriers after the data, 8-byte aligned.
    assert_eq!(layout.barriers, vec![BarrierRegion { owner: "r1".into(), group: false, count: 4, offset: 296 }]);
    assert_eq!(layout.total_bytes, 296 + 32);
    assert!(alloc.places_barriers_of("r1") && !alloc.places_barriers_of("fixed"));
    assert_eq!(alloc.barrier_smem_bytes(), 32);
    assert_eq!(layout.region("y").unwrap().owner.as_deref(), Some("r1"));
    let text = layout.to_string();
    assert!(text.contains("4 mbarriers of pipeline `r1`") && text.contains("z"), "{text}");
    let _ = fixed;

    // A group's (N + 1) x S barriers replace its members' own.
    let a = MemoryResource::new("a").with_pipeline_config(async_cfg(4, 32, 128));
    let b = MemoryResource::new("b").with_pipeline_config(async_cfg(4, 32, 128));
    let g = PipelineGroup::new("g", &[&a, &b], PipelineGroupMode::Merge);
    assert_eq!((g.num_barriers_per_stage(), g.num_barriers()), (3, 12));
    let f = PipelineGroup::new("f", &[&a, &b], PipelineGroupMode::FusedMerge);
    assert_eq!(f.num_barriers(), 8);
    let mut alloc = SmemAllocator::new();
    alloc.add_resource(&a);
    alloc.add_resource(&b);
    alloc.add_pipeline_group(&g);
    let l = alloc.compute_layout().unwrap();
    assert_eq!(l.barriers, vec![BarrierRegion { owner: "g".into(), group: true, count: 12, offset: 0 }]);
    assert!(alloc.places_barriers_of("a"));

    // Bad alignments and capacity.
    let mut alloc = SmemAllocator::new();
    alloc.add(&SmemAllocation::new::<u8>("odd", 8, 48));
    assert!(matches!(alloc.compute_layout(), Err(AllocError::BadAlignment { .. })));
    let mut alloc = SmemAllocator::new().with_capacity(1000);
    alloc.add(&SmemAllocation::bytes("big", 1001));
    let e = alloc.compute_layout().unwrap_err();
    assert!(matches!(e, AllocError::OverCapacity { needed: 1001, capacity: 1000, .. }), "{e}");
}

#[test]
fn alias_groups_share_bytes_and_tmem_layout() {
    let a = SmemAllocation::bytes("a", 256);
    let b = SmemAllocation::bytes("b", 128);
    let c = SmemAllocation::bytes("c", 512);
    let solo = SmemAllocation::new::<u8>("solo", 16, 16);
    let mut alloc = SmemAllocator::new();
    for x in [&a, &b, &c, &solo] {
        alloc.add(x);
    }
    alloc.add_alias_group(vec![vec![a.clone(), b.clone()], vec![c.clone()]]);
    let l = alloc.compute_layout().unwrap();
    // One 512-byte block (the larger phase) at 0; a then b in phase 0, c in phase 1.
    assert_eq!((a.offset(), b.offset(), c.offset(), solo.offset()), (0, 256, 0, 512));
    assert_eq!(l.data_bytes, 528);
    assert_eq!(l.region("c").unwrap().alias_group, Some(0));

    let acc = TmemAllocation::new("acc", 256);
    let sf = TmemAllocation::new("sf", 32);
    let r = MemoryResource::new("r").tmem_requirement(acc.clone()).tmem_requirement(sf.clone());
    let mut t = TmemAllocator::new();
    t.add_resource(&r);
    let l = t.compute_layout().unwrap();
    assert_eq!((acc.offset(), sf.offset(), l.used_columns, l.alloc_columns), (0, 256, 288, 512));
    let mut over = TmemAllocator::new();
    over.add(&TmemAllocation::new("big", 384));
    over.add(&TmemAllocation::new("more", 256));
    assert!(matches!(over.compute_layout(), Err(AllocError::OverCapacity { needed: 640, capacity: 512, .. })));
}

// ---------------------------------------------------------------------------
// The manager's checks
// ---------------------------------------------------------------------------

/// A two-task pipeline `src -> stage -> dst` with the given stage resource.
fn pair(
    stage: MemoryResource,
    alloc: Option<SmemAllocator>,
    extra: Vec<Box<dyn Fn(TaskManager) -> TaskManager>>,
) -> TaskManager {
    let src = MemoryResource::new("src").work(Work::consumer("get").run(nothing));
    let dst = MemoryResource::new("dst").work(Work::producer("put").run(nothing));
    let load = schedule("load", &[&src, &stage], |s, r| {
        s.domain_loop(0, 4, 1, 1, |_| {
            r[0].call("get");
            r[1].acquire();
            r[1].call("fill");
            r[1].commit();
        });
    })
    .unwrap();
    let store = schedule("store", &[&stage, &dst], |s, r| {
        s.domain_loop(0, 4, 1, 1, |_| {
            r[0].wait();
            r[0].call("read");
            r[0].release();
            r[1].call("put");
        });
    })
    .unwrap();
    let graph = ResourceDependencyGraph::new().depends_on(&stage, [&src]).depends_on(&dst, [&stage]);
    let tasks =
        vec![Task::new("Load", &[&src], &[&stage], 4, 1, load), Task::new("Store", &[&stage], &[&dst], 0, 4, store)];
    let mut tm = TaskManager::new(tasks, graph).resource(src).resource(stage).resource(dst);
    if let Some(a) = alloc {
        tm = tm.smem_allocator(a);
    }
    for f in extra {
        tm = f(tm);
    }
    tm
}

fn stage_resource(name: &str) -> MemoryResource {
    MemoryResource::new(name)
        .with_pipeline_config(async_cfg(2, 32, 128))
        .work(Work::producer("fill").run(nothing))
        .work(Work::consumer("read").run(nothing))
}

fn kinds(tm: &TaskManager) -> Vec<VerifyIssueKind> {
    match tm.verify() {
        Ok(()) => vec![],
        Err(e) => e.issues.iter().map(|i| i.kind).collect(),
    }
}

fn messages(tm: &TaskManager, kind: VerifyIssueKind) -> String {
    tm.verify()
        .err()
        .map(|e| e.issues.iter().filter(|i| i.kind == kind).map(|i| i.message.clone()).collect::<Vec<_>>().join("\n"))
        .unwrap_or_default()
}

#[test]
fn diagnostics_barrier_storage_and_tmem_capacity() {
    let stage = stage_resource("stage");
    // Registered: fine.
    let mut alloc = SmemAllocator::new();
    alloc.add_resource(&stage);
    assert_eq!(kinds(&pair(stage, Some(alloc), vec![])), vec![]);
    // An allocator that does not place the pipeline's barriers.
    let tm = pair(stage_resource("stage"), Some(SmemAllocator::new()), vec![]);
    assert_eq!(kinds(&tm), vec![VerifyIssueKind::BarrierStorage]);
    assert!(messages(&tm, VerifyIssueKind::BarrierStorage).contains("`stage`"));
    // TMEM over capacity.
    let mut t = TmemAllocator::new();
    t.add(&TmemAllocation::new("acc", 512));
    t.add(&TmemAllocation::new("extra", 32));
    let tm = pair(stage_resource("stage"), None, vec![Box::new(move |tm: TaskManager| tm.tmem_allocator(t.clone()))]);
    assert_eq!(kinds(&tm), vec![VerifyIssueKind::TmemCapacity]);
    assert!(messages(&tm, VerifyIssueKind::TmemCapacity).contains("544"));
}

#[test]
fn diagnostics_aliased_allocations_need_ordered_owners() {
    // Two resources whose allocations share bytes; `second` follows `first`
    // through a control edge in one manager and not in the other.
    let build = |ordered: bool| {
        let (a, b) = (SmemAllocation::bytes("first_buf", 256), SmemAllocation::bytes("second_buf", 128));
        let first = MemoryResource::new("first").smem_requirement(a.clone()).work(Work::consumer("use").run(nothing));
        let second = MemoryResource::new("second").smem_requirement(b.clone()).work(Work::producer("use").run(nothing));
        let mut alloc = SmemAllocator::new();
        alloc.add_resource(&first);
        alloc.add_resource(&second);
        alloc.add_alias_group(vec![vec![a], vec![b]]);
        let s = schedule("s", &[&first, &second], |_, r| {
            r[0].call("use");
            r[1].call("use");
        })
        .unwrap();
        let edge = if ordered { DependencyGraphControlEdge::new(&first) } else { DependencyGraphEdge::new(&first) };
        let graph = ResourceDependencyGraph::new().depends_on(&second, [edge]);
        TaskManager::new(vec![Task::new("T", &[&first], &[&second], 0, 1, s)], graph)
            .resource(first)
            .resource(second)
            .smem_allocator(alloc)
    };
    assert_eq!(kinds(&build(true)), vec![]);
    let tm = build(false);
    assert_eq!(kinds(&tm), vec![VerifyIssueKind::SmemAlias]);
    let msg = messages(&tm, VerifyIssueKind::SmemAlias);
    for w in ["[0, 128)", "`first_buf`", "`second_buf`", "`first`", "`second`"] {
        assert!(msg.contains(w), "{w} in {msg}");
    }
}

/// The merge copy's resources and store schedule on the host, with
/// `consumers_b` threads in `smem_b`'s consumer group and the store task
/// releasing through the group (or each member itself with `per_member`).
fn merge_manager(consumers_b: u32, per_member: bool, explicit: Option<PipelineConfig>, register: bool) -> TaskManager {
    let src = MemoryResource::new("src").work(Work::consumer("get").run(nothing));
    let dst = MemoryResource::new("dst").work(Work::producer("put").run(nothing));
    let a = stage_resource("smem_a");
    let b = MemoryResource::new("smem_b")
        .with_pipeline_config(async_cfg(2, 32, consumers_b))
        .work(Work::producer("fill").run(nothing))
        .work(Work::consumer("read").run(nothing));
    let mut g = PipelineGroup::new("ab", &[&a, &b], PipelineGroupMode::Merge);
    if let Some(cfg) = explicit {
        g = g.with_pipeline_config(cfg);
    }
    let load = |name: &str, m: &MemoryResource| {
        schedule(name, &[&src, m], |s, r| {
            s.domain_loop(0, 4, 1, 1, |_| {
                r[0].call("get");
                r[1].acquire();
                r[1].call("fill");
                r[1].commit();
            });
        })
        .unwrap()
    };
    let store = schedule("store", &[&a, &b, &dst, &g], |s, r| {
        s.domain_loop(0, 4, 1, 1, |_| {
            r[0].wait();
            r[1].wait();
            r[2].call("put");
            if per_member {
                r[0].release();
                r[1].release();
            } else {
                r[3].release();
            }
        });
    })
    .unwrap();
    let mut alloc = SmemAllocator::new();
    alloc.add_resource(&a);
    alloc.add_resource(&b);
    if register {
        alloc.add_pipeline_group(&g);
    }
    let graph = ResourceDependencyGraph::new().depends_on(&a, [&src]).depends_on(&b, [&src]).depends_on(&dst, [&a, &b]);
    let tasks = vec![
        Task::new("LoadA", &[&src], &[&a], 4, 1, load("la", &a)),
        Task::new("LoadB", &[&src], &[&b], 5, 1, load("lb", &b)),
        Task::new("Store", &[&a, &b], &[&dst], 0, 4, store),
    ];
    TaskManager::new(tasks, graph).resource(src).resource(a).resource(b).resource(g).resource(dst).smem_allocator(alloc)
}

#[test]
fn diagnostics_pipeline_groups() {
    assert_eq!(kinds(&merge_manager(128, false, None, true)), vec![]);
    // The members' consumer counts disagree with the shared empty barrier's.
    let tm = merge_manager(64, false, None, true);
    assert!(kinds(&tm).contains(&VerifyIssueKind::GroupArrivalCount), "{:?}", kinds(&tm));
    assert!(messages(&tm, VerifyIssueKind::GroupArrivalCount).contains("`smem_b`"));
    // An explicit group configuration counting other arrivals.
    let tm = merge_manager(128, false, Some(async_cfg(2, 32, 96)), true);
    assert_eq!(kinds(&tm), vec![VerifyIssueKind::GroupArrivalCount, VerifyIssueKind::GroupArrivalCount]);
    // Releasing the members one by one would arrive twice on the shared barrier.
    let tm = merge_manager(128, true, None, true);
    assert!(kinds(&tm).iter().all(|k| *k == VerifyIssueKind::BadGroup), "{:?}", kinds(&tm));
    assert!(messages(&tm, VerifyIssueKind::BadGroup).contains("through the group"));
    // A group nobody gave barrier storage.
    let tm = merge_manager(128, false, None, false);
    assert_eq!(kinds(&tm), vec![VerifyIssueKind::BarrierStorage]);
    // Structural problems: mismatched stages.
    let a = stage_resource("x");
    let b = MemoryResource::new("y").with_pipeline_config(async_cfg(3, 32, 128));
    let g = PipelineGroup::new("xy", &[&a, &b], PipelineGroupMode::Fork);
    assert!(g.structural_problems().iter().any(|p| p.contains("stages")));
    // A fused merge's full barrier counts the members' producers summed.
    let f = PipelineGroup::new("f", &[&a, &stage_resource("z")], PipelineGroupMode::FusedMerge);
    assert_eq!(f.pipeline_config().unwrap().producer_group.thread_arrive_count().as_static(), Some(64));
    assert!(f.arrival_count_problems().is_empty());
    let f = f.with_pipeline_config(async_cfg(2, 32, 128));
    assert!(f.arrival_count_problems()[0].contains("add up to 64"));
}

#[test]
fn schedules_understand_groups() {
    let a = stage_resource("a");
    let b = stage_resource("b");
    let g = PipelineGroup::new("g", &[&a, &b], PipelineGroupMode::Merge);
    let ok = schedule("ok", &[&a, &b, &g], |_, r| {
        r[0].wait();
        r[1].wait();
        r[2].release();
    });
    assert!(ok.is_ok(), "{}", ok.unwrap_err());
    let err = schedule("bad", &[&a, &g], |_, r| {
        r[1].wait();
        r[1].release();
        r[0].wait();
    })
    .unwrap_err()
    .to_string();
    for w in ["takes only release", "closes no member", "a.wait is never released"] {
        assert!(err.contains(w), "{w} in {err}");
    }
}

// ---------------------------------------------------------------------------
// The ported tutorials
// ---------------------------------------------------------------------------

fn toolchain() -> Option<Toolchain> {
    let tc = Toolchain::discover();
    if tc.is_none() {
        eprintln!("skipping the PTX: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    tc
}

/// The kernel's PTX for `arch` (written to `CUTE_TS_PTX_DIR` when set), with
/// the instructions it must contain.
fn check_ptx(tc: &Toolchain, kernel: &Kernel, arch: Arch, want: &[&str]) {
    let compiled = kernel.compile_with(tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{e}"));
    let ptx = compiled.ptx();
    for w in want {
        assert!(ptx.contains(w), "{}: no `{w}` in the PTX", kernel.name());
    }
    if let Some(dir) = std::env::var_os("CUTE_TS_PTX_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", kernel.name())), ptx).unwrap();
    }
}

#[test]
fn copy_tma_port_on_the_simulator_with_sm90a_ptx() {
    use ts_copy_tma as t;
    t::run(t::ROWS, t::COLS, false);
    t::run(t::ROWS, t::COLS, true);
    let kernel = t::build(4, 128, true);
    let src = t::source(4, 128);
    let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![F16::from_f32(0.0); 512])];
    TaskManager::verify_interleavings(&kernel, t::launch(128), &mut args, &[3]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<F16>(), src);
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("mbarrier"), "no mbarrier in the MLIR");
    if let Some(tc) = toolchain() {
        check_ptx(
            &tc,
            &kernel,
            Arch::Sm90a,
            &["cp.async.bulk.tensor", "mbarrier.arrive.expect_tx", "mbarrier.try_wait"],
        );
    }
}

#[test]
fn merge_copy_port_on_the_simulator_with_sm90a_ptx() {
    use ts_merge_copy as m;
    m::run(m::N_ELEMENTS, true);
    m::run(512, false);
    let kernel = m::build(512, true);
    let mut args = m::args(512);
    TaskManager::verify_interleavings(&kernel, m::launch(), &mut args, &[5]).unwrap_or_else(|e| panic!("{e}"));
    let (a, b) = m::inputs(512);
    assert_eq!((args[2].to_vec::<F16>(), args[3].to_vec::<F16>()), (a, b));
    if let Some(tc) = toolchain() {
        check_ptx(&tc, &kernel, Arch::Sm90a, &["cp.async.bulk.tensor", "mbarrier.arrive"]);
    }
}

/// Fork (one producer warp fills both members and commits them once; one
/// consumer warp per member) and fused merge (one producer warp per member,
/// each committing to the shared full barrier, whose count is their sum; one
/// consumer warp waits on both and releases once): `ITERS` iterations of 32
/// `i32`s per member through 2 stages, on the simulator (SM89-level).
fn group_kernel(mode: PipelineGroupMode) -> Kernel {
    const ITERS: i64 = 6;
    let name = if mode == PipelineGroupMode::Fork { "ts_fork" } else { "ts_fused_merge" };
    Kernel::builder(name).block_dim([96, 1, 1]).target(Arch::Sm89).build(move || {
        let outs = [
            param_tensor::<i32>("x_out", Layout::col_major(32 * ITERS as usize)),
            param_tensor::<i32>("y_out", Layout::col_major(32 * ITERS as usize)),
        ];
        let fork = mode == PipelineGroupMode::Fork;
        let member = |i: usize| {
            let name = ["x", "y"][i];
            let buf = SmemAllocation::new::<i32>(&format!("{name}_buf"), 64, 16);
            let out = outs[i].clone();
            MemoryResource::new(name)
                .with_pipeline_config(async_cfg(2, 32, 32))
                .smem_requirement(buf.clone())
                .work(Work::producer("fill").run({
                    let buf = buf.clone();
                    move |si, _| {
                        let s = buf.ptr::<i32>(&si.context).offset(si.stage_idx.expect("staged") * 32);
                        s.offset(lane_idx()).store(si.loop_offset * 100 + lane_idx() + i as i32 * 10_000);
                        WorkResults::none()
                    }
                }))
                .work(Work::consumer("drain").run(move |si, _| {
                    let s = buf.ptr::<i32>(&si.context).offset(si.stage_idx.expect("staged") * 32);
                    out.store(si.loop_offset * 32 + lane_idx(), s.offset(lane_idx()).load());
                    WorkResults::none()
                }))
        };
        let (x, y) = (member(0), member(1));
        let src = MemoryResource::new("src").work(Work::consumer("get").run(nothing));
        let g = PipelineGroup::new("xy", &[&x, &y], mode);
        let mut alloc = SmemAllocator::new();
        alloc.add_resource(&x);
        alloc.add_resource(&y);
        alloc.add_pipeline_group(&g);
        let tasks = if fork {
            let p = schedule("p", &[&x, &y, &g, &src], |s, r| {
                s.domain_loop(0, ITERS, 1, 1, |_| {
                    r[3].call("get");
                    for m in [r[0], r[1]] {
                        m.acquire();
                        m.call("fill");
                    }
                    r[2].commit();
                });
            })
            .unwrap_or_else(|e| panic!("{e}"));
            let c = |m: &MemoryResource| {
                schedule("c", &[m], |s, r| {
                    s.domain_loop(0, ITERS, 1, 1, |_| {
                        r[0].wait();
                        r[0].call("drain");
                        r[0].release();
                    });
                })
                .unwrap_or_else(|e| panic!("{e}"))
            };
            vec![
                Task::new("Cx", &[&x], &[], 0, 1, c(&x)),
                Task::new("Cy", &[&y], &[], 1, 1, c(&y)),
                Task::new("P", &[&src], &[&x, &y], 2, 1, p),
            ]
        } else {
            let p = |m: &MemoryResource| {
                schedule("p", &[m, &src], |s, r| {
                    s.domain_loop(0, ITERS, 1, 1, |_| {
                        r[1].call("get");
                        r[0].acquire();
                        r[0].call("fill");
                        r[0].commit();
                    });
                })
                .unwrap_or_else(|e| panic!("{e}"))
            };
            let c = schedule("c", &[&x, &y, &g], |s, r| {
                s.domain_loop(0, ITERS, 1, 1, |_| {
                    r[0].wait();
                    r[1].wait();
                    r[0].call("drain");
                    r[1].call("drain");
                    r[2].release();
                });
            })
            .unwrap_or_else(|e| panic!("{e}"));
            vec![
                Task::new("C", &[&x, &y], &[], 0, 1, c),
                Task::new("Px", &[&src], &[&x], 1, 1, p(&x)),
                Task::new("Py", &[&src], &[&y], 2, 1, p(&y)),
            ]
        };
        let graph = ResourceDependencyGraph::new().depends_on(&x, [&src]).depends_on(&y, [&src]);
        let mut tm =
            TaskManager::new(tasks, graph).resource(src).resource(x).resource(y).resource(g).smem_allocator(alloc);
        tm.setup_resources_and_tasks().unwrap_or_else(|e| panic!("{e}"));
        tm.run().unwrap_or_else(|e| panic!("{e}"));
    })
}

#[test]
fn fork_and_fused_merge_groups_on_the_simulator() {
    for mode in [PipelineGroupMode::Fork, PipelineGroupMode::FusedMerge] {
        let kernel = group_kernel(mode);
        let mut args = vec![KernelArg::buffer(&[0i32; 192]), KernelArg::buffer(&[0i32; 192])];
        TaskManager::verify_interleavings(&kernel, LaunchConfig::new([1, 1, 1], [96, 1, 1]), &mut args, &[1, 2])
            .unwrap_or_else(|e| panic!("{mode:?}: {e}"));
        for (i, out) in args.iter().enumerate() {
            let want: Vec<i32> = (0..192).map(|k| (k / 32) * 100 + k % 32 + i as i32 * 10_000).collect();
            assert_eq!(out.to_vec::<i32>(), want, "{mode:?} member {i}");
        }
    }
}

// ---------------------------------------------------------------------------
// TMEM, TSPipelineAsyncUmma and TSPipelineUmmaUmma
// ---------------------------------------------------------------------------

mod umma {
    use super::*;
    use cute::memory::compute_tmem_cols_from_layout;
    use cute::tcgen05::{
        self, CtaGroup, Field, Ld32x32bOp, MmaF16BF16Op, OperandSource, Repetition, SmemLayoutAtomKind, make_tmem_copy,
    };

    /// C = A B^T, A `(M, K)`, B `(N, K)` K-major f16, C `(M, N)` row-major f32,
    /// in `N_TILES` output tiles of `(M, NT)`, each from two k-tiles of `KT`.
    pub const M: usize = 128;
    pub const NT: usize = 64;
    pub const KT: usize = 64;
    pub const N_TILES: usize = 2;
    pub const N: usize = NT * N_TILES;
    pub const K: usize = 2 * KT;
    pub const THREADS: u32 = 224;

    fn mma_op() -> TiledMma {
        let op = MmaF16BF16Op::new(
            DType::F16,
            DType::F32,
            (M, NT),
            CtaGroup::ONE,
            OperandSource::SMEM,
            OperandMajorMode::K,
            OperandMajorMode::K,
        );
        make_tiled_mma(op, Layout::col_major((1, 1, 1)))
    }

    fn smem_layout(mn: usize) -> ComposedLayout {
        let atom = tcgen05::make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128);
        tile_to_shape_swizzled(&atom, (mn, KT))
    }

    /// An `AsyncUmma` resource holding k-tile `k` of A and of B: the loader
    /// warp copies them in and commits; an MMA warp consumes them.
    fn operands(k: usize, a: &Tensor<F16>, b: &Tensor<F16>) -> (MemoryResource, SmemAllocation, SmemAllocation) {
        let sa = SmemAllocation::new::<F16>(&format!("a{k}"), M * KT, 1024);
        let sb = SmemAllocation::new::<F16>(&format!("b{k}"), NT * KT, 1024);
        let cfg = PipelineConfig::create_async_umma_pipeline_cfg(1, threads(32), threads(1), single());
        let (a, b) = (a.clone(), b.clone());
        let r = MemoryResource::new(&format!("ab{k}"))
            .with_pipeline_config(cfg)
            .smem_requirement(sa.clone())
            .smem_requirement(sb.clone())
            .work(Work::producer("load").run({
                let (sa, sb) = (sa.clone(), sb.clone());
                move |si, _| {
                    let atom = make_copy_atom::<F16>(CopyOp::Universal { bits: 128 });
                    let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((4, 8)), &Layout::row_major((1, 8)));
                    let thr = tiled.get_slice(lane_idx());
                    let ga = local_tile(&a, (M, KT), (0, k as i32));
                    let gb = local_tile(&b, (NT, KT), (si.loop_offset, k as i32));
                    let ta = make_tensor(&sa.ptr::<F16>(&si.context), smem_layout(M));
                    let tb = make_tensor(&sb.ptr::<F16>(&si.context), smem_layout(NT));
                    copy(&tiled, &thr.partition_s(&ga), &thr.partition_d(&ta));
                    copy(&tiled, &thr.partition_s(&gb), &thr.partition_d(&tb));
                    // The MMAs read through the async proxy.
                    fence_view_async_shared();
                    WorkResults::none()
                }
            }));
        (r, sa, sb)
    }

    /// Issue the MMAs of one k-tile into accumulator stage `stage`.
    fn issue(si: &StageInfo, acc: &TmemAllocation, sa: &SmemAllocation, sb: &SmemAllocation, accumulate: bool) {
        let mma = mma_op();
        let acc_layout = mma.make_fragment_c::<f32>(&mma.partition_shape_c((M, NT))).layout().clone();
        let cols = compute_tmem_cols_from_layout(&acc_layout, DType::F32) as i32;
        let stage = si.stage_idx.expect("an accumulator stage");
        let acc = make_tensor(acc.ptr::<f32>(&si.context).offset(stage * cols), acc_layout);
        let thr = mma.get_slice(0);
        let ta = make_tensor(&sa.ptr::<F16>(&si.context), smem_layout(M));
        let tb = make_tensor(&sb.ptr::<F16>(&si.context), smem_layout(NT));
        let fa = mma.make_fragment_a(&thr.partition_a(&ta));
        let fb = mma.make_fragment_b(&thr.partition_b(&tb));
        let mut m = mma.clone();
        m.set(Field::ACCUMULATE, accumulate);
        for kb in 0..KT / 16 {
            gemm(&m, &acc, &fa.slice((.., .., kb)), &fb.slice((.., .., kb)), &acc);
            m.set(Field::ACCUMULATE, true);
        }
    }

    /// Warps 0-3 read accumulator stages back and store them; warp 4 loads
    /// the operands; warp 5 (MMA1) computes the first k-tile into a TMEM
    /// accumulator and hands it to warp 6 (MMA2) through a `UmmaUmma`
    /// pipeline; MMA2 adds the second k-tile and hands the result to the
    /// epilogue through a `UmmaAsync` pipeline. The accumulators (two
    /// stages) come from the TaskManager's TMEM allocator.
    pub fn kernel() -> Kernel {
        Kernel::builder("ts_umma_handoff").block_dim([THREADS, 1, 1]).target(Arch::Sm100a).build(|| {
            let a = param_tensor_aligned::<F16>("a", Layout::new((M, K), (K, 1)), 16);
            let b = param_tensor_aligned::<F16>("b", Layout::new((N, K), (K, 1)), 16);
            let c = param_tensor::<f32>("c", Layout::new((M, N), (N, 1)));
            let (ab0, sa0, sb0) = operands(0, &a, &b);
            let (ab1, sa1, sb1) = operands(1, &a, &b);
            let mma = mma_op();
            let acc_layout = mma.make_fragment_c::<f32>(&mma.partition_shape_c((M, NT))).layout().clone();
            let stage_cols = compute_tmem_cols_from_layout(&acc_layout, DType::F32);
            let acc = TmemAllocation::new("acc", 2 * stage_cols);
            let handoff = MemoryResource::new("acc_handoff")
                .with_pipeline_config(PipelineConfig::create_umma_umma_pipeline_cfg(
                    2,
                    threads(1),
                    threads(1),
                    single(),
                ))
                .tmem_requirement(acc.clone())
                .work(Work::producer("mma_first").run({
                    let acc = acc.clone();
                    move |si, _| {
                        issue(si, &acc, &sa0, &sb0, false);
                        WorkResults::none()
                    }
                }));
            let acc_out = MemoryResource::new("acc_out")
                .with_pipeline_config(PipelineConfig::create_umma_async_pipeline_cfg(
                    2,
                    threads(1),
                    threads(128),
                    single(),
                ))
                .work(Work::producer("mma_second").run({
                    let acc = acc.clone();
                    move |si, _| {
                        issue(si, &acc, &sa1, &sb1, true);
                        WorkResults::none()
                    }
                }))
                .work(Work::consumer("epilogue").run(move |si, _| {
                    let stage = si.stage_idx.expect("an accumulator stage");
                    let t =
                        make_tensor(acc.ptr::<f32>(&si.context).offset(stage * stage_cols as i32), acc_layout.clone());
                    let block = t.layout().mode(0);
                    let tile = make_tensor(t.ptr(), Layout::from_modes([block.mode(0), block.mode(1)]));
                    let ctile = local_tile(&c, (M, NT), (0, si.loop_offset));
                    let tiled = make_tmem_copy(&make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X64)), &tile);
                    let thr = tiled.get_slice(thread_idx().x);
                    let (src, dst) = (thr.partition_s(&tile), thr.partition_d(&ctile));
                    let frag = make_fragment_like::<f32, _>(&dst);
                    copy(&tiled, &src, &frag);
                    fence_view_async_tmem_load();
                    basic_copy(&frag, &dst);
                    WorkResults::none()
                }));

            let n = N_TILES as i64;
            let load = schedule("load", &[&ab0, &ab1], |s, r| {
                s.domain_loop(0, n, 1, 1, |_| {
                    for ab in [r[0], r[1]] {
                        ab.acquire();
                        ab.call("load");
                        ab.commit();
                    }
                });
            })
            .unwrap_or_else(|e| panic!("{e}"));
            let first = schedule("mma1", &[&ab0, &handoff], |s, r| {
                s.domain_loop(0, n, 1, 1, |_| {
                    r[0].wait();
                    r[1].acquire();
                    r[1].call("mma_first");
                    r[1].commit();
                    r[0].release();
                });
            })
            .unwrap_or_else(|e| panic!("{e}"));
            let second = schedule("mma2", &[&handoff, &ab1, &acc_out], |s, r| {
                s.domain_loop(0, n, 1, 1, |_| {
                    r[0].wait();
                    r[1].wait();
                    r[2].acquire();
                    r[2].call("mma_second");
                    r[2].commit();
                    r[1].release();
                    r[0].release();
                });
            })
            .unwrap_or_else(|e| panic!("{e}"));
            let epilogue = schedule("epilogue", &[&acc_out], |s, r| {
                s.domain_loop(0, n, 1, 1, |_| {
                    r[0].wait();
                    r[0].call("epilogue");
                    r[0].release();
                });
            })
            .unwrap_or_else(|e| panic!("{e}"));
            let tasks = vec![
                Task::new("Epilogue", &[&acc_out], &[], 0, 4, epilogue),
                Task::new("Load", &[], &[&ab0, &ab1], 4, 1, load),
                Task::new("Mma1", &[&ab0], &[&handoff], 5, 1, first),
                Task::new("Mma2", &[&handoff, &ab1], &[&acc_out], 6, 1, second),
            ];
            let graph =
                ResourceDependencyGraph::new().depends_on(&handoff, [&ab0]).depends_on(&acc_out, [&handoff, &ab1]);
            let mut smem = SmemAllocator::new();
            let mut tmem = TmemAllocator::new();
            for r in [&ab0, &ab1, &handoff, &acc_out] {
                smem.add_resource(r);
                tmem.add_resource(r);
            }
            let mut tm = TaskManager::new(tasks, graph)
                .resource(ab0)
                .resource(ab1)
                .resource(handoff)
                .resource(acc_out)
                .smem_allocator(smem)
                .tmem_allocator(tmem);
            tm.setup_resources_and_tasks().unwrap_or_else(|e| panic!("{e}"));
            tm.run().unwrap_or_else(|e| panic!("{e}"));
        })
    }

    fn small_ints(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                ((s >> 40) % 9) as f32 - 4.0
            })
            .collect()
    }

    pub fn inputs() -> (Vec<f32>, Vec<f32>, Vec<KernelArg>) {
        let (av, bv) = (small_ints(M * K, 3), small_ints(N * K, 7));
        let a: Vec<F16> = av.iter().map(|&x| F16::from_f32(x)).collect();
        let b: Vec<F16> = bv.iter().map(|&x| F16::from_f32(x)).collect();
        (av, bv, vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0f32; M * N])])
    }

    pub fn reference(av: &[f32], bv: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; M * N];
        for r in 0..M {
            for col in 0..N {
                out[r * N + col] = (0..K).map(|k| av[r * K + k] * bv[col * K + k]).sum();
            }
        }
        out
    }
}

#[test]
fn tmem_allocator_and_umma_handoffs_on_the_simulator_with_sm100a_ptx() {
    let kernel = umma::kernel();
    let (av, bv, mut args) = umma::inputs();
    let cfg = LaunchConfig::new([1, 1, 1], [umma::THREADS, 1, 1]);
    let report = TaskManager::verify_interleavings(&kernel, cfg, &mut args, &[7]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.tcgen05_mma_instructions, (umma::N_TILES * 2 * umma::KT / 16) as u64);
    assert_eq!(args[2].to_vec::<f32>(), umma::reference(&av, &bv));
    if let Some(tc) = toolchain() {
        check_ptx(
            &tc,
            &kernel,
            Arch::Sm100a,
            &[
                "tcgen05.alloc.cta_group::1",
                "tcgen05.dealloc.cta_group::1",
                "tcgen05.relinquish_alloc_permit",
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one",
                "tcgen05.mma",
            ],
        );
    }
}
