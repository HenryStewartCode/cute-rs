//! The pipeline primitives (`cute::pipeline`) and shared-memory partitions
//! and capacities (`cute::memory`) at build time: layouts, partition sizes,
//! capacities, target checks, and the IR the mbarrier arrays build.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::ir::{Op, Region, SmemPartition};
use cute::memory::{SmemAllocator, get_kernel_smem_size, get_smem_capacity_in_bytes};
use cute::pipeline::{
    Agent, BarrierParam, CooperativeGroup, MbarrierArray, NamedBarrier, PipelineOp, PipelineUserType, StageState,
    SyncObject, alloc_reserved_mbarrier, make_pipeline_state,
};
use cute::prelude::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// A reserved buffer declared after user buffers is laid out first, at
/// offset 0, and the user buffers follow it with their alignment padding.
#[test]
fn reserved_buffers_come_first() {
    let kernel = Kernel::build("reserved_first", || {
        let smem = SmemAllocator::new();
        let a = smem.allocate_array::<u8>(3, 1, SmemPartition::User);
        let r = smem.allocate_array::<u64>(2, 8, SmemPartition::Reserved);
        let b = smem.allocate_array::<f32>(1, 16, SmemPartition::User);
        let r2 = smem.allocate_bytes(3, 4, SmemPartition::Reserved);
        a.store(1u8);
        r.store(2u64);
        b.store(3.0f32);
        r2.store(4u8);
    });
    let ir = kernel.ir();
    let layout = ir.shared_memory_layout();
    let starts: Vec<usize> = layout.starts.iter().flatten().copied().collect();
    // Declaration order: a, r, b, r2. Reserved: r at 0, r2 at 16 (to 19); then a at 19, b at 32.
    assert_eq!(starts, [19, 0, 32, 16]);
    assert_eq!((layout.reserved_bytes, layout.static_bytes), (19, 36));
    assert_eq!(layout.partition_bytes(SmemPartition::Reserved, false), 19);
    assert_eq!(layout.partition_bytes(SmemPartition::Reserved, true), 19);
    assert_eq!(layout.partition_bytes(SmemPartition::User, false), 17);
    assert_eq!(layout.partition_bytes(SmemPartition::User, true), 36);
    let partitions: Vec<SmemPartition> = ir.buffers.iter().map(|b| b.partition).collect();
    use SmemPartition::{Reserved, User};
    assert_eq!(partitions, [User, Reserved, User, Reserved]);
    assert_eq!(get_kernel_smem_size(&kernel), ir.shared_memory_bytes());
}

/// Without reserved buffers nothing moves: the layout is the one before
/// partitions, and every buffer is in the user partition.
#[test]
fn kernels_without_reserved_buffers_keep_their_layout() {
    let kernel = Kernel::build("user_only", || {
        alloc_smem::<u8>(5, 1).store(1u8);
        make_smem_tensor::<f32>(Layout::col_major(8)).store(0, 1.0f32);
        alloc_smem::<u64>(1, 8).store(2u64);
    });
    let layout = kernel.ir().shared_memory_layout();
    assert_eq!(layout.starts.iter().flatten().copied().collect::<Vec<_>>(), [0, 128, 160]);
    assert_eq!((layout.reserved_bytes, layout.static_bytes), (0, 168));
    assert_eq!(layout.partition_bytes(SmemPartition::User, false), 168);
}

/// Partition-size queries made before, between and inside branches all get
/// the final sizes, which count allocations made after them.
#[test]
fn partition_queries_see_later_allocations() {
    let kernel = Kernel::builder("late_queries").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(4));
        let first = cute::memory::calculate_total_usage();
        if_(thread_idx().x.eq(0), || {
            out.store(0, first);
            out.store(1, cute::memory::calculate_partition_size(SmemPartition::Reserved, false));
            out.store(2, cute::memory::calculate_partition_size(SmemPartition::User, false));
        });
        let bars = alloc_reserved_mbarrier(3, 2);
        let data = alloc_smem::<u32>(10, 4);
        if_(thread_idx().x.eq(0), || {
            bars.store(0u64);
            data.store(0u32);
            out.store(3, cute::memory::calculate_total_usage());
        });
    });
    let mut args = vec![KernelArg::buffer(&[0i32; 4])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), [88, 48, 40, 88]);
}

/// `alloc_reserved_mbarrier` is `num_stages * multiplier` 8-byte words,
/// 8-byte aligned, in the reserved partition.
#[test]
fn reserved_mbarriers() {
    let kernel = Kernel::build("reserved_mbarriers", || {
        alloc_smem::<u8>(1, 1).store(0u8);
        alloc_reserved_mbarrier(4, 3).store(0u64);
    });
    let b = &kernel.ir().buffers[1];
    assert_eq!((b.len, b.align, b.partition, b.elem), (Some(12), 8, SmemPartition::Reserved, DType::U64));
    assert_eq!(kernel.ir().shared_memory_layout().starts[1], Some(0));
}

/// Every architecture's capacity, as CuTe DSL's table has it.
#[test]
fn smem_capacity_of_every_architecture() {
    for arch in Arch::ALL {
        let kib = match arch.version() {
            (8, 0) | (8, 7) => 163,
            (8, 6) | (8, 9) | (12, 0) | (12, 1) => 99,
            (10, 7) => 327,
            (9, 0) | (10, 0) | (10, 3) | (11, 0) => 227,
            v => panic!("no capacity known for {v:?}"),
        };
        assert_eq!(arch.smem_capacity_bytes(), kib * 1024, "{arch}");
        assert_eq!(get_smem_capacity_in_bytes(Some(arch)), kib * 1024, "{arch}");
        assert_eq!(SmemAllocator::capacity_in_bytes(Some(arch)), kib * 1024, "{arch}");
    }
    assert_eq!(cute::validate::MAX_SHARED_BYTES, 327 * 1024);
    // During a build, `None` is the declared target.
    Kernel::builder("capacity").target(Arch::Sm86).build(|| {
        assert_eq!(get_smem_capacity_in_bytes(None), 99 * 1024);
    });
}

/// `check_target` holds the build-time shared memory to the target's
/// capacity.
#[test]
fn check_target_uses_the_capacity() {
    let kernel = |kib: usize| {
        Kernel::build("big_smem", move || {
            alloc_smem::<u8>(kib * 1024, 16).store(1u8);
        })
    };
    let k120 = kernel(120);
    for arch in [Arch::Sm89, Arch::Sm120, Arch::Sm86] {
        let e = cute::validate::check_target(k120.ir(), arch).unwrap_err();
        assert!(e.message.contains("122880 bytes of shared memory") && e.message.contains(arch.chip()), "{e}");
    }
    for arch in [Arch::Sm80, Arch::Sm90, Arch::Sm100a] {
        cute::validate::check_target(k120.ir(), arch).unwrap_or_else(|e| panic!("{arch}: {e}"));
    }
    let k230 = kernel(230);
    let e = cute::validate::check_target(k230.ir(), Arch::Sm90).unwrap_err();
    assert!(e.message.contains("more than the 232448 bytes (227 KiB) a block can have on sm_90"), "{e}");
    cute::validate::check_target(k230.ir(), Arch::Sm107).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(get_kernel_smem_size(&k230), 230 * 1024);
}

/// The swizzle of a pointer and of a layout cannot be combined; a swizzled
/// pointer alone is fine.
#[test]
fn allocate_tensor_swizzles() {
    let msg = build_panic(|| {
        Kernel::build("two_swizzles", || {
            let layout = make_composed_layout(make_swizzle(2, 2, 3), 0, Layout::row_major((8, 32)));
            SmemAllocator::new().allocate_tensor::<u16>(layout, 128, Some(make_swizzle(2, 2, 3)), SmemPartition::User);
        });
    });
    assert!(msg.contains("the layout is swizzled already"), "{msg}");
    Kernel::build("pointer_swizzle", || {
        let t = SmemAllocator::new().allocate_tensor::<u16>(
            Layout::row_major((8, 32)),
            128,
            Some(make_swizzle(2, 2, 3)),
            SmemPartition::Reserved,
        );
        assert!(t.swizzle().is_some());
        t.store((1, 2), 7u16);
    });
}

/// Every `mbarrier.init` of an array runs in warp 0's elected lane: inside
/// an `if` on the warp, inside an `if` on `elect_one`, once per stage. A
/// recast array initialises nothing.
#[test]
fn mbarrier_arrays_initialise_each_stage_once() {
    fn inits_at_depth(r: &Region, depth: usize, out: &mut Vec<usize>) {
        for inst in &r.insts {
            if matches!(inst.op, Op::MbarrierInit { .. }) {
                out.push(depth);
            }
            let nested = depth + matches!(inst.op, Op::If { .. }) as usize;
            for region in inst.op.regions() {
                inits_at_depth(region, nested, out);
            }
        }
    }
    let kernel = Kernel::builder("inits").block_dim([64, 1, 1]).build(|| {
        let storage = alloc_reserved_mbarrier(4, 2);
        let group = CooperativeGroup::new(Agent::ThreadBlock, 1);
        let full = MbarrierArray::new(&storage, 4, (PipelineOp::AsyncThread, group), 0);
        assert_eq!(full.arrive_count().as_static(), Some(64));
        let _empty = MbarrierArray::new(&storage.offset(4), 4, (PipelineOp::AsyncLoad, group), 0);
        let recast = full.recast_to_new_op_type(PipelineOp::TmaLoad);
        assert_eq!((recast.op(), recast.num_stages(), recast.max()), (PipelineOp::TmaLoad, 4, (1 << 20) - 1));
    });
    let mut depths = Vec::new();
    inits_at_depth(&kernel.ir().body, 0, &mut depths);
    assert_eq!(depths, [2; 8]);
}

/// The groups' thread counts: static for threads, warps and declared
/// blocks, a runtime value for clusters.
#[test]
fn cooperative_group_thread_counts() {
    Kernel::builder("counts").block_dim([32, 2, 1]).build(|| {
        let count = |agent, size: i32| CooperativeGroup::new(agent, size).thread_arrive_count().as_static();
        assert_eq!(count(Agent::Thread, 5), Some(5));
        assert_eq!(count(Agent::Warp, 3), Some(96));
        assert_eq!(count(Agent::ThreadBlock, 2), Some(128));
        assert_eq!(count(Agent::ThreadBlockCluster, 1), None);
        let runtime = CooperativeGroup::new(Agent::Warp, thread_idx().x + 1);
        assert!(matches!(runtime.thread_arrive_count(), BarrierParam::Dynamic(_)));
    });
    Kernel::build("undeclared_block", || {
        let group = CooperativeGroup::new(Agent::ThreadBlock, 1);
        assert_eq!(group.thread_arrive_count().as_static(), None);
    });
}

/// `make_pipeline_state` starts producers, and the producer-consumer kind,
/// at phase 1 and consumers at phase 0.
#[test]
fn pipeline_state_starting_phases() {
    Kernel::build("phases", || {
        let phase = |kind| make_pipeline_state(kind, 4).phase().as_constant();
        assert_eq!(phase(PipelineUserType::Producer), Some(1));
        assert_eq!(phase(PipelineUserType::ProducerConsumer), Some(1));
        assert_eq!(phase(PipelineUserType::Consumer), Some(0));
        let s = make_pipeline_state(PipelineUserType::Consumer, 4);
        assert_eq!((s.stages(), s.index().as_constant(), s.count().as_constant()), (4, Some(0), Some(0)));
        assert_eq!(StageState::index(&s).as_constant(), Some(0));
    });
}

/// Code generic over [`SyncObject`] works with mbarrier arrays and named
/// barriers alike.
#[test]
fn sync_objects_are_generic() {
    fn signal_and_wait<S: SyncObject>(s: &S, arrive: S::ArriveArgs, wait: S::WaitArgs) -> Option<u32> {
        s.arrive_and_wait(arrive, wait);
        s.max()
    }
    let kernel = Kernel::builder("generic_sync").block_dim([64, 1, 1]).target(Arch::Sm89).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(64));
        let group = CooperativeGroup::new(Agent::ThreadBlock, 1);
        let bars = MbarrierArray::new(&alloc_reserved_mbarrier(1, 1), 1, (PipelineOp::AsyncThread, group), 0);
        sync_threads();
        let max = signal_and_wait(&bars, (0.into(), None, None), (0.into(), Int32::from(0)));
        assert_eq!(max, Some((1 << 20) - 1));
        let named = NamedBarrier::new(1, 64);
        assert_eq!(signal_and_wait(&named, (), ()), Some(4095));
        assert!(matches!(SyncObject::get_barrier(&named, ()), BarrierParam::Static(1)));
        out.store(thread_idx().x, bars.test_wait(0, 0).to::<u32>());
    });
    let mut args = vec![KernelArg::buffer(&[0u32; 64])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<u32>(), [1; 64]);
}
