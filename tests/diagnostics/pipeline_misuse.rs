//! Pipeline primitives and shared-memory partitions: misuse diagnostics.

use super::{build_panic, simulate_err};
use cute::memory::{SmemAllocator, SmemPartition, get_smem_capacity_in_bytes};
use cute::pipeline::{
    Agent, CooperativeGroup, MbarrierArray, PipelineOp, PipelineUserType, agent_sync, alloc_reserved_mbarrier,
    make_pipeline_state,
};
use cute::prelude::*;

type Misuse = (&'static str, Box<dyn FnOnce()>);

fn warp() -> CooperativeGroup {
    CooperativeGroup::new(Agent::Warp, 1)
}

fn array(op: PipelineOp) -> MbarrierArray {
    MbarrierArray::new(&alloc_reserved_mbarrier(2, 1), 2, (op, warp()), 0)
}

/// Static misuse of the pipeline states, groups and mbarrier arrays is
/// refused while the kernel is built.
#[test]
fn pipeline_primitives_refuse_bad_arguments() {
    let cases: Vec<Misuse> = vec![
        (
            "PipelineState::advance_by: count must be >= 1, not 0",
            Box::new(|| make_pipeline_state(PipelineUserType::Consumer, 3).advance_by(0)),
        ),
        (
            "CooperativeGroup: the group must have at least one agent, not 0",
            Box::new(|| {
                CooperativeGroup::new(Agent::Thread, 0);
            }),
        ),
        (
            "an mbarrier array needs at least one stage, not 0",
            Box::new(|| {
                MbarrierArray::new(&alloc_smem::<u64>(1, 8), 0, (PipelineOp::AsyncThread, warp()), 0);
            }),
        ),
        (
            "each mbarrier must count at least one arrival, not 0",
            Box::new(|| {
                let none = CooperativeGroup::new(Agent::Thread, Int32::from(0));
                MbarrierArray::new(&alloc_smem::<u64>(1, 8), 1, (PipelineOp::AsyncThread, none), 0);
            }),
        ),
        (
            "a TMA load's transaction byte count cannot be negative (-16)",
            Box::new(|| {
                MbarrierArray::new(&alloc_smem::<u64>(1, 8), 1, (PipelineOp::TmaLoad, warp()), -16);
            }),
        ),
        ("a TCGen05Mma arrival needs a CtaGroup", Box::new(|| array(PipelineOp::TCGen05Mma).arrive(0, None, None))),
        (
            "an mbarrier array cannot arrive for PipelineOp::TmaStore",
            Box::new(|| array(PipelineOp::TmaStore).arrive(0, None, None)),
        ),
        (
            "MbarrierArray::arrive_and_drop is not supported yet",
            Box::new(|| array(PipelineOp::AsyncThread).arrive_and_drop()),
        ),
        ("agent_sync(Agent::Thread) is not supported", Box::new(|| agent_sync(Agent::Thread, false))),
        ("agent_sync(Agent::Warp): there is no warp barrier instruction", Box::new(|| agent_sync(Agent::Warp, false))),
    ];
    for (want, body) in cases {
        let msg = build_panic(|| {
            Kernel::builder("pipeline_misuse").block_dim([64, 1, 1]).build(body);
        });
        assert!(msg.contains(want), "expected `{want}` in: {msg}");
    }
}

/// A pipeline built without a target waits as CuTe DSL does, with
/// `mbarrier.try_wait`: compiling it for sm_89 fails at the line that
/// waits.
#[test]
fn pipeline_without_a_target_needs_sm90() {
    let line = std::cell::Cell::new(0);
    let kernel = Kernel::builder("untargeted_pipeline").block_dim([64, 1, 1]).build(|| {
        let full = array(PipelineOp::AsyncThread);
        sync_threads();
        if_(warp_idx().eq(1), || full.arrive(0, None, None));
        full.wait(0, 0);
        line.set(line!() - 1);
    });
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm89).unwrap_err();
    assert!(e.message.contains("mbarrier.try_wait needs sm_90 or newer"), "{e}");
    let loc = e.loc.expect("a source location");
    assert!(loc.file().ends_with("pipeline_misuse.rs") && loc.line() == line.get(), "{e} (want line {})", line.get());
    cute::validate::check_target(kernel.ir(), Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
}

/// Without the init sync, nothing stops warp 1 from waiting on an
/// mbarrier before warp 0 has initialised it: here warp 0 is held at a
/// named barrier until warp 1 has polled. The simulator reports the
/// poll.
#[test]
fn waiting_before_the_pipeline_init_is_reported() {
    let kernel = Kernel::builder("no_init_sync").block_dim([64, 1, 1]).target(Arch::Sm89).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(64));
        let hold = cute::pipeline::NamedBarrier::new(1, 64);
        if_(warp_idx().eq(0), || hold.arrive_and_wait());
        let full = array(PipelineOp::AsyncThread);
        // Missing: pipeline_init_arrive(None, false); pipeline_init_wait(None);
        let done = if_else_yield(
            warp_idx().eq(1),
            || {
                let done = full.try_wait(0, 0);
                hold.arrive();
                done
            },
            || Boolean::from(true),
        );
        out.store(thread_idx().x, done.to::<i32>());
    });
    let mut args = vec![KernelArg::buffer(&[0i32; 64])];
    let e = simulate_err(&kernel, 64, &mut args);
    assert!(e.message.contains("on an mbarrier that was never initialised"), "{e}");
    assert!(e.to_string().contains("pipeline_misuse.rs"), "{e}");
}

/// The allocator's limits, a capacity query with no target to go by,
/// and a kernel too big for its target.
#[test]
fn smem_allocator_and_capacity_misuse() {
    let msg = build_panic(|| {
        Kernel::build("align_2048", || {
            SmemAllocator::new().allocate_array::<u32>(4, 2048, SmemPartition::User);
        });
    });
    assert!(msg.contains("an alignment of 2048 bytes is more than the 1024 bytes shared memory allows"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("no_elements", || {
            SmemAllocator::new().allocate_array::<u32>(0, 4, SmemPartition::Reserved);
        });
    });
    assert!(msg.contains("allocate_array: num_elems is 0"), "{msg}");
    if std::env::var("CUTE_RS_ARCH").is_err() {
        let msg = build_panic(|| {
            get_smem_capacity_in_bytes(None);
        });
        assert!(msg.contains("unsupported compute capability: no architecture was given"), "{msg}");
    }
    let msg = build_panic(|| {
        Kernel::builder("too_big_for_sm89").target(Arch::Sm89).build(|| {
            alloc_smem::<u8>(120 * 1024, 16).store(1u8);
        });
    });
    assert!(
        msg.contains(
            "the kernel uses 122880 bytes of shared memory, more than the 101376 bytes (99 KiB) a block can have \
             on sm_89"
        ),
        "{msg}"
    );
}
