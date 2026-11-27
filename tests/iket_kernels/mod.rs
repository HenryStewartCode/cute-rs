//! Kernels instrumented with IKET events, shared by `profiling.rs` (simulator) and
//! `gpu_iket.rs` (GPU).
// Shared by several test binaries; each uses part of it.
#![allow(dead_code)]

use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineAsync, PipelineAsyncOptions};
use cute::prelude::*;

pub const ITEMS: usize = 4;
const STAGES: usize = 2;

/// A producer warp (0) and a consumer warp (1) hand `ITEMS` rows through a named
/// `PipelineAsync`. With `iket` on, the producer keeps a sentinel-token range ("fill") open
/// across loop iterations and marks each row; the consumer pushes and pops "wait" and
/// "consume" ranges. Both warps sit in iket `WarpScope`s. With `iket` off, the same body
/// builds with the event calls emitting nothing.
pub fn traced_handoff(iket: Option<IketOptions>) -> Kernel {
    let mut opts = Kernel::builder("traced_handoff").block_dim([64, 1, 1]).target(Arch::Sm89);
    if let Some(o) = iket {
        opts = opts.iket(o);
    }
    opts.build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(ITEMS * 32));
        let sums = param_tensor::<u32>("sums", Layout::col_major(32));
        let lane = lane_idx();
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let options = PipelineAsyncOptions { name: "handoff".to_string(), ..PipelineAsyncOptions::default() };
        let pipeline = PipelineAsync::create(STAGES, warp, warp, options);
        let data = make_smem_tensor::<u32>(Layout::col_major(STAGES * 32));
        let (producer, consumer) = pipeline.make_participants();
        let items = ITEMS as i32;
        if_else(
            warp_idx().eq(0),
            || {
                let scope = WarpScope::with("producer", 0, 0, true);
                let start = (sentinel_token("fill"), producer);
                let (token, mut p) = for_range_carry(0, items, 1, start, |k, (token, mut p)| {
                    range_end(token, Some(k.into()));
                    let handle = p.acquire_and_advance(None);
                    data.store(handle.index() * 32 + lane, src.load(k * 32 + lane));
                    mark("produced", Some(k.into()));
                    handle.commit();
                    (range_start("fill", Some(k.into())), p)
                });
                range_end(token, Some(Int32::constant(items).into()));
                p.tail();
                scope.close();
            },
            || {
                let scope = WarpScope::with("consumer", 1, 1, true);
                let zero = Uint32::from(0u32);
                let (sum, _) = for_range_carry(0, items, 1, (zero, consumer), |k, (sum, mut c)| {
                    range_push("wait", None);
                    let handle = c.wait_and_advance(None);
                    range_pop();
                    range_push("consume", None);
                    let x = data.load(handle.index() * 32 + lane);
                    handle.release();
                    range_pop();
                    mark("consumed", Some(k.into()));
                    (sum + x, c)
                });
                sums.store(lane, sum);
                scope.close();
            },
        );
    })
}

pub fn handoff_args() -> Vec<KernelArg> {
    let src: Vec<u32> = (0..ITEMS * 32).map(|i| (i as u32 * 7919) % 1000).collect();
    vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 32])]
}

pub fn handoff_cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [64, 1, 1])
}

pub fn handoff_sums() -> Vec<u32> {
    let src: Vec<u32> = (0..ITEMS * 32).map(|i| (i as u32 * 7919) % 1000).collect();
    (0..32).map(|l| (0..ITEMS).map(|k| src[k * 32 + l]).sum()).collect()
}

/// A warp's records as `kind:event` strings, in order.
pub fn warp_events(trace: &IketTrace, warp: u32) -> Vec<String> {
    trace.warp_records(0, warp).iter().map(|r| format!("{:?}:{}", r.kind, r.event)).collect()
}
