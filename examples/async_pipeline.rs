// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/notebooks/async_pipeline.ipynb
//   (the kernels `synced_producer_consumer`, `async_pipeline_kernel` and
//   `async_pipeline_staged_kernel`, their `SharedStorage` structs and launches,
//   and the host data and outputs of the notebook's cells),
//     Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Warp specialisation with an async pipeline, ported from CuTe DSL's
//! `async_pipeline.ipynb` tutorial. Every kernel runs 64 threads: warp 0
//! produces, warp 1 consumes.
//!
//! * [`Variant::Synced`] (`synced_producer_consumer`): a hand-off through a
//!   one-element staging buffer, with a `sync_threads` before and after
//!   each read.
//! * [`Variant::OneStage`] (`async_pipeline_kernel`): a one-stage
//!   `PipelineAsync` made from a `SharedStorage` struct's mbarriers, driven
//!   by its participants (`acquire_and_advance` and `commit`,
//!   `wait_and_advance` and `release`, the producer's `tail`).
//! * [`Variant::Staged`] (`async_pipeline_staged_kernel`): the same with
//!   five stages; thread 0 then copies the staging buffer out, which ends
//!   as `[5, 6, 7, 3, 4]` after eight items.
//! * [`Variant::TryTokens`]: the notebook's "Try Acquire/Wait" section,
//!   which describes `try_acquire` and `try_wait` without code: the staged
//!   kernel with each acquire and wait given the token of a probe made
//!   just before it.
//!
//! The notebook zeroes the staging buffer from every thread, which two warps
//! writing the same words makes a race; here warp 0 alone zeroes it.
//!
//! The pipelines need `mbarrier.try_wait` (sm_90) as CuTe DSL builds them;
//! built for sm_89 ([`build`]'s `arch`) their waits spin on
//! `mbarrier.test_wait` instead, which the SM89 GPU runs.
//!
//! Run with `cargo run --release --example async_pipeline`.

use cute::cute_struct;
use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineAsync, PipelineAsyncOptions};
use cute::prelude::*;

/// Items handed over (`res = torch.zeros((8,))`).
pub const ITEMS: usize = 8;
/// Stages of the staged kernels (`staging = torch.zeros((5,))`).
pub const STAGES: usize = 5;
/// The seeds [`run`] simulates under, besides the built-in schedules.
pub const SEEDS: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

cute_struct! {
    /// `run_synced_producer_consumer`'s `SharedStorage`: a one-element
    /// staging buffer aligned to 1024 bytes.
    pub struct SyncedStorage() {
        #[align(1024)]
        staging_buffer: MemRange<f32>[1],
    }
}

cute_struct! {
    /// The pipelines' `SharedStorage`: `2 * stages` mbarriers (full, then
    /// empty) and a staging buffer of `stages` elements aligned to 1024
    /// bytes.
    pub struct SharedStorage(stages: usize) {
        tma_mbar_ptr: MemRange<i64>[2 * stages],
        #[align(1024)]
        staging_buffer: MemRange<f32>[stages],
    }
}

/// The notebook's kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Synced,
    OneStage,
    Staged,
    TryTokens,
}

impl Variant {
    pub const ALL: [Variant; 4] = [Variant::Synced, Variant::OneStage, Variant::Staged, Variant::TryTokens];

    /// The staging buffer's elements.
    pub fn stages(self) -> usize {
        match self {
            Variant::Synced | Variant::OneStage => 1,
            Variant::Staged | Variant::TryTokens => STAGES,
        }
    }

    /// Whether the kernel writes its staging buffer out.
    pub fn has_staging_output(self) -> bool {
        matches!(self, Variant::Staged | Variant::TryTokens)
    }

    fn name(self) -> &'static str {
        match self {
            Variant::Synced => "synced_producer_consumer",
            Variant::OneStage => "async_pipeline_kernel",
            Variant::Staged => "async_pipeline_staged_kernel",
            Variant::TryTokens => "async_pipeline_try_kernel",
        }
    }
}

/// Zero the staging buffer. The notebook has every thread fill it; two
/// warps writing the same words between barriers is a race the simulator
/// reports, so only warp 0 does here.
fn fill_from_warp_0(warp_idx: Int32, staging_smem: &Tensor<f32>) {
    if_(warp_idx.eq(0), || staging_smem.fill(0.0));
}

/// The kernel of `variant`, built for `arch`.
pub fn build(variant: Variant, arch: Arch) -> Kernel {
    Kernel::builder(variant.name()).block_dim([64, 1, 1]).target(arch).build(move || {
        let res = param_tensor::<f32>("res", Layout::col_major(ITEMS));
        let warp_idx = make_warp_uniform(warp_idx());
        let items = ITEMS as i32;
        if variant == Variant::Synced {
            let storage = SyncedStorage::allocate_aligned(64);
            let staging_smem = storage.staging_buffer.get_tensor(Layout::col_major(1));
            fill_from_warp_0(warp_idx, &staging_smem);
            sync_threads();
            for_range(0, items, 1, |i| {
                if_(warp_idx.eq(0), || staging_smem.store(0, i.to::<f32>() * 1.0));
                // Enter the critical region.
                sync_threads();
                if_(warp_idx.eq(1), || res.store(i, staging_smem.load(0)));
                // Leave it.
                sync_threads();
            });
            return;
        }
        let stages = variant.stages();
        let staging = variant.has_staging_output().then(|| param_tensor::<f32>("staging", Layout::col_major(stages)));
        let storage = SharedStorage::allocate_aligned(64, stages);
        // Warp 0 and warp 1, as 32 threads each.
        let producer_group = CooperativeGroup::new(Agent::Thread, 32);
        let consumer_group = CooperativeGroup::new(Agent::Thread, 32);
        let options = PipelineAsyncOptions {
            barrier_storage: Some(storage.tma_mbar_ptr.data_ptr().cast::<u64>()),
            ..Default::default()
        };
        let pipeline = PipelineAsync::create(stages, producer_group, consumer_group, options);
        let staging_smem = storage.staging_buffer.get_tensor(Layout::col_major(stages));
        fill_from_warp_0(warp_idx, &staging_smem);
        sync_threads();
        let (producer, consumer) = pipeline.make_participants();
        let try_tokens = variant == Variant::TryTokens;
        // Producer warp.
        if_(warp_idx.eq(0), || {
            let mut producer = for_range_carry(0, items, 1, producer, |i, mut producer| {
                // Wait until a stage is free (with a probe's token first, for
                // the try variant).
                let token = try_tokens.then(|| producer.try_acquire());
                let handle = producer.acquire_and_advance(token);
                staging_smem.store(handle.index(), i.to::<f32>() * 1.0);
                // The data is ready.
                handle.commit();
                producer
            });
            // Keep the block alive until every stage is released.
            producer.tail();
        });
        // Consumer warp.
        if_(warp_idx.eq(1), || {
            for_range_carry(0, items, 1, consumer, |i, mut consumer| {
                let token = try_tokens.then(|| consumer.try_wait());
                let handle = consumer.wait_and_advance(token);
                res.store(i, staging_smem.load(handle.index()));
                // The stage may be written again.
                handle.release();
                consumer
            });
        });
        if let Some(staging) = &staging {
            if_(thread_idx().x.eq(0), || {
                for s in 0..stages {
                    staging.store(s, staging_smem.load(s));
                }
            });
        }
    })
}

/// The notebook's launch: one block of two warps.
pub fn launch() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [64, 1, 1])
}

/// The notebook's host data: zeros for `res` (and `staging`).
pub fn args(variant: Variant) -> Vec<KernelArg> {
    let mut args = vec![KernelArg::f32(vec![0.0; ITEMS])];
    if variant.has_staging_output() {
        args.push(KernelArg::f32(vec![0.0; STAGES]));
    }
    args
}

/// The notebook's outputs: `res = [0, 1, ..., 7]`, and for the staged
/// kernels `staging = [5, 6, 7, 3, 4]` (item `i` was last written to stage
/// `i % 5`).
pub fn check(variant: Variant, args: &[KernelArg]) {
    let want: Vec<f32> = (0..ITEMS).map(|i| i as f32).collect();
    assert_eq!(args[0].as_f32(), &want[..], "{variant:?}: res");
    if variant.has_staging_output() {
        assert_eq!(args[1].as_f32(), &[5.0, 6.0, 7.0, 3.0, 4.0][..], "{variant:?}: staging");
    }
}

/// Simulate every kernel, built for SM89 and SM90, under every schedule
/// and [`SEEDS`], and check.
pub fn run() {
    for variant in Variant::ALL {
        for arch in [Arch::Sm89, Arch::Sm90] {
            let mut args = args(variant);
            build(variant, arch)
                .simulate_schedules(launch(), &mut args, &SimOptions::default(), &SEEDS)
                .unwrap_or_else(|e| panic!("{variant:?} for {arch:?}: {e}"));
            check(variant, &args);
        }
    }
}

fn main() {
    run();
    let mut args = args(Variant::Staged);
    build(Variant::Staged, Arch::Sm89).simulate(launch(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    println!("res = {:?}, staging = {:?}", args[0].as_f32(), args[1].as_f32());
    println!("async_pipeline: PASS");
}
