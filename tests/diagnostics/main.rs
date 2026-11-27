//! Kernels with deliberate bugs: the simulator and the validator must catch
//! each one and point at the offending line.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;

fn simulate_err(kernel: &Kernel, block: u32, args: &mut [KernelArg]) -> SimError {
    match kernel.simulate(LaunchConfig::new([1, 1, 1], [block, 1, 1]), args) {
        Ok(_) => panic!("expected `{}` to fail in the simulator", kernel.name()),
        Err(e) => e,
    }
}

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// Stage 64 floats through shared memory with cp.async, then read them back.
fn staged_copy(wait: bool) -> Kernel {
    Kernel::builder("staged_copy").block_dim([32, 1, 1]).build(|| {
        let src = param_tensor::<f32>("src", Layout::col_major(128));
        let dst = param_tensor::<f32>("dst", Layout::col_major(128));
        let smem = make_smem_tensor::<f32>(Layout::col_major(128));
        let t = thread_idx().x;
        let atom = make_copy_atom::<f32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
        let thr = tiled.get_slice(t);
        copy(&tiled, &thr.partition_s(&src), &thr.partition_d(&smem));
        cp_async_commit_group();
        if wait {
            cp_async_wait_group(0);
        }
        sync_threads();
        let i = (t + 1) % 32 * 4;
        dst.store(i, smem.load(i));
    })
}

mod arithmetic;
mod barriers;
mod basics;
mod carry_and_asm;
mod errors;
mod layouts;
mod memory_model;
mod types_and_targets;
mod vars_and_loops;

#[path = "../feature_kernels/mod.rs"]
mod feature_kernels;

// ---- Pipeline primitives and shared-memory partitions ----

mod pipeline_misuse;

// ---- PipelineAsync and the static persistent tile schedulers ----

mod pipeline_async_misuse;

mod tile_scheduler_misuse;

mod cpasync_order_misuse;

mod grouped_scheduler_misuse;

/// Tensor maps edited in the kernel: a TMA copy
/// through a descriptor written without the tensormap-proxy release or acquire, or with
/// them in the wrong order, through bytes that hold no descriptor, a divergent
/// `tensormap.cp_fenceproxy`, and `tensormap.replace` for plain sm_90.
mod tensormap_misuse;

// ---- PipelineTmaAsync and PipelineTmaStore ----

mod tma_pipeline_misuse;

// Misuse of tensor-memory allocation.
mod tmem_misuse;

// Software-pipelined loops.
mod range_prefetch_misuse;
