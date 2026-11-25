// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The captured schedules are those of NVIDIA CUTLASS 4.8.1's BSD-3-Clause
// Task Scheduling tutorials (examples/python/CuTeDSL/experimental/
// task_scheduling/blackwell/tutorial/01_copy_basics_ts/01_copy_grid_stride.py,
// 02_copy_tma.py, 03_copy_tma_conditional.py and
// 03_persistent_scheduling_dynamic_domain_ts/01_copy_persistent_skip_tile.py):
// their resources' work names and variables and the order of their schedule
// steps. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. See THIRD-PARTY-NOTICES.md.

//! Task Scheduling part 1 (roadmap part m11-ts-resources): schedule capture
//! of the copy tutorials compared with expected listings, the capture
//! diagnostics, pipeline configuration checks, and hand-written
//! warp-specialised kernels calling resources directly on the simulator.

#[allow(dead_code)]
mod ts_kernels;

use cute::pipeline::{Agent, CooperativeGroup, PipelineOp};
use cute::prelude::*;
use cute::task_scheduling::*;
use cute::task_scheduling::{Schedule, schedule};
use cute::utils::WorkTileInfo;
use ts_kernels::*;

fn nothing(_: &StageInfo, _: &WorkArgs) -> WorkResults {
    WorkResults::none()
}

fn group(n: i32) -> CooperativeGroup {
    CooperativeGroup::new(Agent::Thread, n)
}

/// The resources of 02_copy_tma.py / 03_copy_tma_conditional.py, with
/// bodies that do nothing (capture never runs them).
struct TmaResources {
    input: MemoryResource,
    smem: MemoryResource,
    output: MemoryResource,
    trace: MemoryResource,
}

fn tma_resources() -> TmaResources {
    let gmem_idx = TaskLocalVariable::<Int32>::scalar("gmem_idx", 0, "GMEM row coordinate");
    let smem_val = TaskLocalVariable::<Float16>::new("smem_val", || Float16::constant(F16(0)), "SMEM value");
    let is_highlight = TaskLocalVariable::<Boolean>::scalar("is_highlight", false, "highlight row");
    let cfg = PipelineConfig::create_tma_async_pipeline_cfg(1, 128 * 2, group(1), group(4 * 32));
    TmaResources {
        input: MemoryResource::new("inputGmemResource")
            .work(Work::consumer("compute_coords").returns(&gmem_idx).run(nothing)),
        smem: MemoryResource::new("smemResource")
            .with_pipeline_config(cfg)
            .smem_requirement(SmemAllocation::new::<F16>("smem_data", 128, 128))
            .work(Work::producer("init_load_state").auxiliary().run(nothing))
            .work(Work::consumer("init_read_state").auxiliary().run(nothing))
            .work(Work::consumer("read_smem").returns(&smem_val).run(nothing))
            .work(Work::consumer("is_highlight_tile").auxiliary().returns(&is_highlight).run(nothing))
            .work(Work::producer("tma_load").input::<Int32>("gmem_idx").run(nothing)),
        output: MemoryResource::new("outputGmemResource")
            .work(Work::producer("store").input::<Float16>("smem_val").run(nothing)),
        trace: MemoryResource::new("traceGmemResource")
            .work(Work::producer("mark_begin").run(nothing))
            .work(Work::producer("record_heartbeat").run(nothing))
            .work(Work::producer("mark_highlight").run(nothing))
            .work(Work::producer("mark_end").run(nothing)),
    }
}

fn rows() -> Bound {
    Bound::symbol("num_rows")
}

#[test]
fn capture_grid_stride() {
    let item = TaskLocalVariable::<Int16>::scalar("item", 0, "input element");
    let input = MemoryResource::new("inputGmemResource").work(Work::consumer("get_item").returns(&item).run(nothing));
    let output =
        MemoryResource::new("outputGmemResource").work(Work::producer("set_item").input::<Int16>("data").run(nothing));
    let s = schedule("schedule_fn", &[&input, &output], |s, r| {
        let (input, output) = (r[0], r[1]);
        s.domain_loop(Bound::symbol("start"), Bound::symbol("num_entries"), Bound::symbol("step"), 2, |_| {
            let res = input.call("get_item").one();
            output.call_with("set_item", [("data", &res)]);
        });
    })
    .unwrap();
    assert_eq!(
        s.to_string(),
        "schedule schedule_fn [inputGmemResource, outputGmemResource]
  domain_loop(start, num_entries, step, unroll=2)
    #0 loop inputGmemResource.get_item -> item
    #1 loop outputGmemResource.set_item(data <- #0.item)"
    );
    assert!(s.warnings.is_empty());
}

#[test]
fn capture_tma_load_store_and_padding() {
    let r = tma_resources();
    let load = schedule("load_schedule", &[&r.input, &r.smem], |s, h| {
        let (input, smem) = (h[0], h[1]);
        smem.call("init_load_state");
        s.domain_loop(0, rows(), 1, 1, |_| {
            let gmem_idx = input.call("compute_coords").one();
            smem.try_acquire();
            smem.acquire();
            smem.call_with("tma_load", [("gmem_idx", &gmem_idx)]);
            smem.commit();
        });
    })
    .unwrap();
    assert_eq!(
        load.to_string(),
        "schedule load_schedule [inputGmemResource, smemResource]
  #0 head smemResource.init_load_state [aux]
  domain_loop(0, num_rows, 1)
    #1 loop inputGmemResource.compute_coords -> gmem_idx
    #2 loop smemResource.try_acquire
    #3 loop smemResource.acquire
    #4 loop smemResource.tma_load(gmem_idx <- #1.gmem_idx)
    #5 loop smemResource.commit"
    );
    let store = schedule("store_schedule", &[&r.smem, &r.output], |s, h| {
        let (smem, output) = (h[0], h[1]);
        smem.call("init_read_state");
        s.domain_loop(0, rows(), 1, 1, |_| {
            smem.try_wait();
            smem.wait();
            let smem_val = smem.call("read_smem").one();
            smem.release();
            output.call_with("store", [("smem_val", &smem_val)]);
        });
    })
    .unwrap();
    assert_eq!(
        store.to_string(),
        "schedule store_schedule [smemResource, outputGmemResource]
  #0 head smemResource.init_read_state [aux]
  domain_loop(0, num_rows, 1)
    #1 loop smemResource.try_wait
    #2 loop smemResource.wait
    #3 loop smemResource.read_smem -> smem_val
    #4 loop smemResource.release
    #5 loop outputGmemResource.store(smem_val <- #3.smem_val)"
    );
    let steps = store.steps();
    assert_eq!(steps[3].stage, ScheduleStage::ConsumerWork);
    assert_eq!(steps[0].stage, ScheduleStage::ConsumerAuxWork);
    assert_eq!(steps[5].inputs[0].source, Source::Step { step: 3, variable: "smem_val".into() });

    let padding = schedule("padding_schedule", &[], |s, _| s.domain_loop(0, rows(), 1, 1, |_| {})).unwrap();
    assert_eq!(padding.to_string(), "schedule padding_schedule []\n  domain_loop(0, num_rows, 1)");
}

#[test]
fn capture_tma_conditional_guards() {
    let r = tma_resources();
    let load = schedule("load_schedule", &[&r.input, &r.smem, &r.trace], |s, h| {
        let (input, smem, trace) = (h[0], h[1], h[2]);
        smem.call("init_load_state");
        s.domain_loop(0, rows(), 1, 1, |d| {
            d.first_iter(|| {
                trace.call("mark_begin");
            });
            d.every(4, 0, || {
                trace.call("record_heartbeat");
            });
            let gmem_idx = input.call("compute_coords").one();
            smem.try_acquire();
            smem.acquire();
            smem.call_with("tma_load", [("gmem_idx", &gmem_idx)]);
            smem.commit();
        });
    })
    .unwrap();
    assert_eq!(
        load.to_string(),
        "schedule load_schedule [inputGmemResource, smemResource, traceGmemResource]
  #0 head smemResource.init_load_state [aux]
  domain_loop(0, num_rows, 1)
    first_iter
      #1 loop-first traceGmemResource.mark_begin
    every(4)
      #2 loop traceGmemResource.record_heartbeat
    #3 loop inputGmemResource.compute_coords -> gmem_idx
    #4 loop smemResource.try_acquire
    #5 loop smemResource.acquire
    #6 loop smemResource.tma_load(gmem_idx <- #3.gmem_idx)
    #7 loop smemResource.commit"
    );
    let store = schedule("store_schedule", &[&r.smem, &r.output, &r.trace], |s, h| {
        let (smem, output, trace) = (h[0], h[1], h[2]);
        smem.call("init_read_state");
        s.domain_loop(0, rows(), 1, 1, |d| {
            smem.try_wait();
            smem.wait();
            let smem_val = smem.call("read_smem").one();
            smem.release();
            let is_highlight = smem.call("is_highlight_tile").one();
            s.when_true(&is_highlight, || {
                trace.call("mark_highlight");
            });
            output.call_with("store", [("smem_val", &smem_val)]);
            d.last_iter(|| {
                trace.call("mark_end");
            });
        });
    })
    .unwrap();
    assert_eq!(
        store.to_string(),
        "schedule store_schedule [smemResource, outputGmemResource, traceGmemResource]
  #0 head smemResource.init_read_state [aux]
  domain_loop(0, num_rows, 1)
    #1 loop smemResource.try_wait
    #2 loop smemResource.wait
    #3 loop smemResource.read_smem -> smem_val
    #4 loop smemResource.release
    #5 loop smemResource.is_highlight_tile [aux] -> is_highlight
    when_true(smemResource.is_highlight_tile -> is_highlight)
      #6 loop traceGmemResource.mark_highlight
    #7 loop outputGmemResource.store(smem_val <- #3.smem_val)
    last_iter
      #8 loop-last traceGmemResource.mark_end"
    );
    // The guard counts as a use of `is_highlight`: no warning.
    assert!(store.warnings.is_empty(), "{:?}", store.warnings);
    // Iteration predicates as CuTe DSL evaluates them.
    assert!(every(4, 0).fires(8, 10) && !every(4, 1).fires(4, 10) && every(4, 1).fires(5, 10));
    assert!(LAST_ITER.fires(9, 10) && !LAST_ITER.fires(0, 0) && FIRST_ITER.fires(0, 1));
}

fn skip_tile_resources() -> (MemoryResource, WorkQueue) {
    let copy = MemoryResource::new("CopyResource").work(Work::producer("copy_tile_row").run(nothing));
    let wq = WorkQueue::new(
        "WorkQueue",
        TileSchedulerConfig::create_static_persistent_tile_scheduler_params(persistent_params()),
    )
    .with_skip_if(|wt: &WorkTileInfo| wt.tile_idx()[0].ge(3));
    (copy, wq)
}

#[test]
fn capture_persistent_skip_tile() {
    let (copy, wq) = skip_tile_resources();
    let s = schedule("copy_schedule", &[&copy, &wq], |s, h| {
        let (copy_res, wq) = (h[0], h[1]);
        s.work_tile_loop_skipping(wq, |w| {
            w.skippable(|| {
                s.domain_loop(0, rows(), 1, 1, |_| {
                    copy_res.call("copy_tile_row");
                });
            });
            wq.try_wait();
            wq.wait();
            wq.call("get_and_advance_work_tile");
            wq.release();
        });
    })
    .unwrap();
    assert_eq!(
        s.to_string(),
        "schedule copy_schedule [CopyResource, WorkQueue]
  work_tile_loop(WorkQueue, skip_if=WorkQueue.skip_work_tile_if)
    skippable
      domain_loop(0, num_rows, 1)
        #0 loop CopyResource.copy_tile_row
    #1 tail WorkQueue.try_wait
    #2 tail WorkQueue.wait
    #3 tail WorkQueue.get_and_advance_work_tile -> work_tile
    #4 tail WorkQueue.release"
    );
    // WorkQueue outputs feed the persistent loop, not routes: no warning.
    assert!(s.warnings.is_empty());
    assert!(s.has_work_tile_loop());
    assert!(s.resources[1].work_queue && !s.resources[1].pipelined);
}

fn err(r: Result<Schedule, ScheduleError>) -> ScheduleError {
    match r {
        Ok(s) => panic!("expected an error, captured:\n{s}"),
        Err(e) => e,
    }
}

#[test]
fn diagnostics_unknown_work_and_unproduced_variable() {
    let r = tma_resources();
    // A typo gets a suggestion; a variable name is pointed at its producer.
    let e = err(schedule("typo", &[&r.input, &r.smem], |s, h| {
        s.domain_loop(0, rows(), 1, 1, |_| {
            h[0].call("compute_cords");
            h[1].call("smem_val");
        });
    }));
    assert!(e.has(IssueKind::UnknownWork));
    assert_eq!(e.issues.len(), 2, "{e}");
    assert!(e.issues[0].message.contains("did you mean `compute_coords`?"), "{e}");
    assert!(e.issues[1].message.contains("one of its variables"), "{e}");

    // `store` reads `smem_val` that no step produced, and smemResource's
    // variable is not mirrored into a schedule where it is not pipelined.
    let smem_val = TaskLocalVariable::<Float16>::uninitialized("smem_val", "");
    let plain = MemoryResource::new("plain").variable(&smem_val);
    let e = err(schedule("unproduced", &[&plain, &r.output], |s, h| {
        s.domain_loop(0, rows(), 1, 1, |_| {
            h[1].call_with("store", [("smem_val", &smem_val)]);
        });
    }));
    assert!(e.has(IssueKind::UnproducedVariable), "{e}");
    assert!(e.issues[0].message.contains("read before any step produces it"), "{e}");

    // A missing input, and a handle from a call that produces nothing.
    let e = err(schedule("missing", &[&r.smem, &r.output], |s, h| {
        let aux = h[0].call("init_read_state").one();
        s.domain_loop(0, rows(), 1, 1, |_| {
            h[1].call("store");
            h[1].call_with("store", [("smem_val", &aux)]);
        });
    }));
    assert_eq!(e.issues.iter().filter(|i| i.kind == IssueKind::UnproducedVariable).count(), 2, "{e}");

    // A variable of a pipelined resource is mirrored from the peer task.
    let smem_var = TaskLocalVariable::<Float16>::uninitialized("peer_val", "");
    let piped = MemoryResource::new("pipedSmem")
        .with_pipeline_config(PipelineConfig::create_async_async_pipeline_cfg(2, group(32), group(32), None))
        .variable(&smem_var);
    let ok = schedule("mirrored", &[&piped, &r.output], |s, h| {
        s.domain_loop(0, rows(), 1, 1, |_| {
            h[1].call_with("store", [("smem_val", &smem_var)]);
        });
    })
    .unwrap();
    assert_eq!(
        ok.steps()[0].inputs[0].source,
        Source::Peer { resource: "pipedSmem".into(), variable: "peer_val".into() }
    );

    // Wrong type, unknown input, stale handle.
    let e = err(schedule("bad_args", &[&r.input, &r.smem, &r.output], |s, h| {
        s.domain_loop(0, rows(), 1, 1, |_| {
            let idx = h[0].call("compute_coords").one();
            h[2].call_with("store", [("smem_val", &idx)]);
            let _again = h[0].call("compute_coords").one();
            h[1].call_with("tma_load", [("gmem_idx", &idx), ("gmem_ix", &idx)]);
        });
    }));
    let msgs: Vec<&str> = e.issues.iter().map(|i| i.message.as_str()).collect();
    assert!(e.issues.iter().all(|i| i.kind == IssueKind::BadArgument), "{e}");
    assert!(msgs.iter().any(|m| m.contains("is a f16") && m.contains("holds a i32")), "{e}");
    assert!(msgs.iter().any(|m| m.contains("is stale")), "{e}");
    assert!(msgs.iter().any(|m| m.contains("did you mean `gmem_idx`?")), "{e}");
}

#[test]
fn diagnostics_unpaired_pipeline_ops() {
    let r = tma_resources();
    let e = err(schedule("unpaired", &[&r.smem], |s, h| {
        let smem = h[0];
        s.domain_loop(0, rows(), 1, 1, |_| {
            smem.acquire();
            smem.call("init_load_state");
        });
        s.domain_loop(0, rows(), 1, 1, |_| {
            smem.commit();
            smem.try_wait();
            smem.release();
        });
    }));
    assert!(e.issues.iter().all(|i| i.kind == IssueKind::UnpairedPipelineOp || i.kind == IssueKind::BadStructure));
    let msgs: Vec<&str> = e.issues.iter().map(|i| i.message.as_str()).collect();
    assert!(msgs.iter().any(|m| m.contains("smemResource.acquire is never committed")), "{e}");
    assert!(msgs.iter().any(|m| m.contains("smemResource.commit without a preceding acquire")), "{e}");
    assert!(msgs.iter().any(|m| m.contains("smemResource.try_wait is followed by release, not wait")), "{e}");
    // The two domain loops are also a structural problem? No: sibling loops are allowed.
    assert!(!e.has(IssueKind::BadStructure), "{e}");
    // Acquiring twice.
    let e = err(schedule("twice", &[&r.smem], |s, h| {
        s.domain_loop(0, rows(), 1, 1, |_| {
            h[0].acquire();
            h[0].acquire();
            h[0].commit();
        });
    }));
    assert!(e.issues[0].message.contains("acquire while the previous acquire is not yet committed"), "{e}");
}

#[test]
fn diagnostics_guards_and_structure() {
    let r = tma_resources();
    let (copy, wq) = skip_tile_resources();
    // An iteration predicate outside a loop, negated, or with period 0; a
    // guard on a value that is not a boolean or integer; nested guards.
    let e = err(schedule("guards", &[&r.input, &r.smem, &r.trace, &wq], |s, h| {
        let (input, smem, trace, wq) = (h[0], h[1], h[2], h[3]);
        s.when_true(FIRST_ITER, || {});
        s.domain_loop(0, rows(), 1, 1, |d| {
            s.when_false(LAST_ITER, || {});
            s.when_true(every(0, 0), || {});
            let tile = wq.call("init_work_tile").named("work_tile");
            s.when_true(&tile, || {});
            let init = smem.call("init_read_state").one();
            s.when_true(&init, || {});
            d.first_iter(|| {
                d.every(2, 0, || {
                    trace.call("mark_end");
                })
            });
            let _ = input;
        });
    }));
    let guard_issues: Vec<&Issue> = e.issues.iter().filter(|i| i.kind == IssueKind::UnsupportedGuard).collect();
    assert_eq!(guard_issues.len(), 6, "{e}");
    assert!(guard_issues[0].message.contains("needs a domain_loop"), "{e}");
    assert!(guard_issues[1].message.contains("cannot be negated"), "{e}");
    assert!(guard_issues[2].message.contains("positive period"), "{e}");
    assert!(guard_issues[3].message.contains("`work_tile` is a WorkTileInfo"), "{e}");
    assert!(guard_issues[4].message.contains("produces no value to test"), "{e}");
    assert!(guard_issues[5].message.contains("do not nest"), "{e}");

    // Placement: a work queue step in a skippable region, skippable without
    // a skip predicate, a domain loop outside skippable in a skipping loop,
    // nested domain loops, a second work_tile_loop, work after it.
    let e = err(schedule("structure", &[&copy, &wq], |s, h| {
        let (copy_res, wq) = (h[0], h[1]);
        s.work_tile_loop_skipping(wq, |w| {
            w.skippable(|| wq.release());
            s.domain_loop(0, rows(), 1, 1, |_| s.domain_loop(0, 1, 1, 1, |_| {}));
        });
        s.work_tile_loop(wq, |w| w.skippable(|| {}));
        copy_res.call("copy_tile_row");
    }));
    let msgs: Vec<&str> = e.issues.iter().map(|i| i.message.as_str()).collect();
    assert!(msgs.iter().any(|m| m.contains("cannot be skippable")), "{e}");
    assert!(msgs.iter().any(|m| m.contains("must be inside skippable()")), "{e}");
    assert!(msgs.iter().any(|m| m.contains("nested in another domain_loop")), "{e}");
    assert!(msgs.iter().any(|m| m.contains("at most one work_tile_loop")), "{e}");
    assert!(msgs.iter().any(|m| m.contains("needs a work_tile_loop with a skip predicate")), "{e}");
    assert!(msgs.iter().any(|m| m.contains("copy_tile_row after the work_tile_loop")), "{e}");
    // work_tile_loop over a resource that is not a work queue.
    let e = err(schedule("not_wq", &[&copy], |s, h| s.work_tile_loop(h[0], |_| {})));
    assert!(e.issues[0].message.contains("needs a WorkQueue"), "{e}");
}

#[test]
fn unrouted_output_is_a_warning() {
    let r = tma_resources();
    let s =
        schedule("unrouted", &[&r.input], |s, h| s.domain_loop(0, rows(), 1, 1, |_| drop(h[0].call("compute_coords"))))
            .unwrap();
    assert_eq!(s.warnings, ["inputGmemResource.compute_coords produces `gmem_idx`, which no later step reads"]);
}

#[test]
#[should_panic(expected = "name of a pipeline operation")]
fn work_named_like_a_pipeline_op_is_refused() {
    let _ = MemoryResource::new("r").work(Work::producer("commit").run(nothing));
}

#[test]
fn pipeline_configs() {
    let layout = Layout::col_major(1);
    let cfgs = [
        PipelineConfig::create_async_async_pipeline_cfg(4, group(32), group(64), None),
        PipelineConfig::create_tma_async_pipeline_cfg(4, 1024, group(1), group(128)),
        PipelineConfig::create_tma_umma_pipeline_cfg(4, 1024, group(1), group(1), layout.clone()),
        PipelineConfig::create_umma_async_pipeline_cfg(2, group(1), group(128), layout.clone()),
        PipelineConfig::create_async_umma_pipeline_cfg(2, group(128), group(1), layout.clone()),
        PipelineConfig::create_umma_umma_pipeline_cfg(2, group(1), group(1), layout.clone()),
        PipelineConfig::create_clc_fetch_async_pipeline_cfg(1, 16, group(1), group(32), None),
    ];
    let kinds: Vec<PipelineType> = cfgs.iter().map(|c| c.pipeline_type).collect();
    assert_eq!(
        kinds,
        [
            PipelineType::AsyncAsync,
            PipelineType::TmaAsync,
            PipelineType::TmaUmma,
            PipelineType::UmmaAsync,
            PipelineType::AsyncUmma,
            PipelineType::UmmaUmma,
            PipelineType::ClcFetchAsync
        ]
    );
    for c in &cfgs {
        c.validate().unwrap();
    }
    let base = cfgs[0].clone();
    // Interleaving: strides must divide the stages; split roles need the advance flags.
    assert!(base.clone().with_interleave_stride(2).validate().is_ok());
    assert!(base.clone().with_interleave_stride(3).validate().unwrap_err().0.contains("does not divide"));
    let split = base.clone().with_interleave_stride([2, 1, 1, 1]);
    assert!(split.validate().unwrap_err().0.contains("advance_on_acquire"));
    assert!(split.with_advance_on_acquire(true).validate().is_ok());
    assert!(base.clone().with_interleave_stride([1, 1, 4, 2]).validate().unwrap_err().0.contains("advance_on_wait"));
    assert_eq!(base.clone().with_interleave_stride([2, 2, 4, 4]).interleave_stride.max(), 4);
    // Producer operations and signalling sets.
    assert!(base.clone().with_producer_op(PipelineOp::AsyncLoad).validate().is_ok());
    assert!(cfgs[1].clone().with_producer_op(PipelineOp::AsyncLoad).validate().is_err());
    assert!(base.clone().with_producer_op(PipelineOp::TmaLoad).validate().is_err());
    let leaders = SignalingThreads::CTA_LEADER | SignalingThreads::TASK_WARP_LEADER;
    assert!(base.clone().with_signaling_threads(leaders, SignalingThreads::ALL).validate().is_ok());
    let bad = SignalingThreads::ALL | SignalingThreads::CTA_LEADER;
    assert!(base.clone().with_signaling_threads(SignalingThreads::ALL, bad).validate().is_err());
    assert_eq!(base.clone().with_consumer_wait_signaling_threads(leaders).consumer_wait_threads(), leaders);
    // The UMMA pipelines need a cluster layout.
    let mut no_layout = cfgs[3].clone();
    no_layout.cta_layout_vmnk = None;
    assert!(no_layout.validate().unwrap_err().0.contains("cluster layout"));
    // Tile scheduler configs.
    let ts = TileSchedulerConfig::create_static_persistent_tile_scheduler_params(persistent_params());
    assert_eq!(ts.tile_scheduler_type(), TileSchedulerType::StaticPersistent);
    // Memory requirements reach the allocator hooks.
    let r = tma_resources();
    assert_eq!(r.smem.get_smem_requirements()[0].size_bytes(), 256);
    assert!(r.smem.get_tmem_requirements().is_empty());
}

#[test]
fn sim_grid_stride_copy() {
    let kernel = grid_stride_copy();
    let src = ramp_i16(GS_ENTRIES);
    let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![0i16; GS_ENTRIES])];
    kernel.simulate(LaunchConfig::new([GS_BLOCKS, 1, 1], [GS_THREADS, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].to_vec::<i16>(), src);
}

#[test]
fn sim_staged_copy() {
    let kernel = staged_copy();
    let a = ramp_i32(ST_ROWS * ST_COLS);
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![0i32; ST_ROWS * ST_COLS])];
    kernel.simulate(LaunchConfig::new([ST_BLOCKS as u32, 1, 1], [ST_THREADS, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].to_vec::<i32>(), a);
}

#[test]
fn sim_persistent_skip_copy() {
    let kernel = persistent_skip_copy();
    let a = ramp_i32(PS_ROWS * PS_COLS);
    let mut args = vec![
        KernelArg::buffer(&a),
        KernelArg::buffer(&vec![0i32; PS_ROWS * PS_COLS]),
        KernelArg::buffer(&[0i32; PS_LAUNCH_TILES]),
    ];
    kernel.simulate(LaunchConfig::new(PS_GRID, [PS_TILE as u32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].to_vec::<i32>(), a);
    assert_eq!(args[2].to_vec::<i32>(), [1, 1, 1, 2, 2]);
}

/// The work queue over cluster launch control (sm_100a, simulator only):
/// warp 1 is the scheduler (`acquire`, `fetch_work_tile`, `commit`), both
/// warps consume (`wait`, `get_and_advance_work_tile`, `release`) through a
/// one-stage `ClcFetchAsync` pipeline; every tile is visited exactly once.
#[test]
fn sim_clc_work_queue() {
    use cute::utils::ClcDynamicPersistentTileSchedulerParams;
    const NTILES: usize = 12;
    let params = ClcDynamicPersistentTileSchedulerParams::new((4, 3, 1), [1, 1, 1], Default::default());
    let grid = params.get_grid_shape();
    let kernel = Kernel::builder("ts_clc_queue").block_dim([64, 1, 1]).target(Arch::Sm100a).build(move || {
        let hits = param_tensor::<i32>("hits", Layout::col_major(NTILES));
        let response = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
        let cfg = PipelineConfig::create_clc_fetch_async_pipeline_cfg(
            1,
            16,
            group(1),
            group(64),
            Some(Layout::col_major((1usize, 1usize, 1usize, 1usize))),
        );
        let mut wq = WorkQueue::new(
            "WorkQueue",
            TileSchedulerConfig::create_clc_dynamic_persistent_tile_scheduler_params(params.clone(), &response.ptr()),
        )
        .with_pipeline_config(cfg);
        wq.create();
        let (first, _) = wq.init_work_tile(&StageInfo::new(0));
        let consume = |t: WorkTileInfo, mut c: cute::pipeline::PipelineState| {
            wq.try_wait(&c);
            wq.wait(&c);
            let next = wq.get_and_advance_work_tile(&wq.consumer_stage(&c, StageInfo::new(0).with_work_tile(t)));
            wq.release(&mut c);
            (next, c)
        };
        if_else(
            warp_idx().eq(0),
            || {
                while_carry(
                    (first, wq.consumer_state()),
                    |(t, _)| t.is_valid_tile(),
                    |(t, c)| {
                        let [m, n, _] = t.tile_idx();
                        if_(thread_idx().x.eq(0), || {
                            let i = m + n * 4;
                            hits.store(i, hits.load(i) + 1);
                        });
                        consume(t, c)
                    },
                );
            },
            || {
                let (_, mut p, _) = while_carry(
                    (first, wq.producer_state(), wq.consumer_state()),
                    |(t, _, _)| t.is_valid_tile(),
                    |(t, mut p, c)| {
                        wq.acquire(&p);
                        wq.fetch_work_tile(&wq.producer_stage(&p, StageInfo::new(0).with_work_tile(t)));
                        wq.commit(&mut p);
                        let (next, c) = consume(t, c);
                        (next, p, c)
                    },
                );
                wq.producer_tail(&mut p);
            },
        );
    });
    let mut args = vec![KernelArg::buffer(&[0i32; NTILES])];
    kernel.simulate(LaunchConfig::new(grid, [64, 1, 1]).with_cluster([1, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), [1; NTILES]);
    assert!(kernel.to_mlir().contains("clusterlaunchcontrol"), "no CLC query in the MLIR");
}
