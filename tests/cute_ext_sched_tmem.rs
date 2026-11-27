// SPDX-License-Identifier: MIT OR Apache-2.0

//! cute_ext's scheduler pipelines, DSMEM cluster mailbox and TMEM accumulator buffering: host tests of the b32 payload codec and of the
//! accumulator planner (hand-computed tables, and the configuration of CuTe DSL's
//! `cute_ext/blackwell/dense_gemm/dense_block_scaled_gemm.py` with its defaults), then on the
//! simulator against plain Rust: a CLCPipeline handing out tiles on a small grid, a
//! GroupedGemmSchedulerPipeline feeding grouped-GEMM tiles to three consumer warps, a mailbox
//! exchange between the two CTAs of a pair with mixed field types and storage alignments, and
//! disjoint and overlapping accumulator buffering in a TMEM producer/consumer kernel (an early
//! release is reported by the race detector); MLIR and, where `ptxas` is installed, sm_100a
//! PTX; diagnostics last.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::experimental::{
    self, AccumulatorBufferingKind, AccumulatorBufferingOptions, AccumulatorBufferingPlan, AccumulatorOverlapPolicy,
    AccumulatorPlanSelection, ArrivalCountOptions, B32Payload, B32PayloadCodec, CLCPipeline, CTA_PAIR_PEER_LOCALE,
    ClusterMailbox, DecodedPayload, GenericPipeline, GroupedGemmSchedulerPipeline, OperationType, PipelineClass, Side,
    TmemAllocationRequest, plan_accumulator_buffering, plan_accumulator_buffering_columns,
};
use cute::memory::compute_tmem_cols_from_layout;
use cute::prelude::*;
use cute::tcgen05::{CtaGroup, Ld32x32bOp, OperandSource, Repetition, St32x32bOp, make_tmem_copy};
use cute::utils::{
    LayoutEnum, PersistentTileSchedulerParams, StaticPersistentGroupTileScheduler, create_initial_search_state,
};
use cute::{AnyValue, cute_struct};

use OperationType::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn in_kernel(f: impl FnOnce() + 'static) -> String {
    build_panic(move || {
        Kernel::builder("diag").block_dim([128, 1, 1]).target(Arch::Sm100a).build(f);
    })
}

/// Seeded schedules with random async completion, plus the built-in ones.
fn schedules(seeds: u64) -> Vec<SimOptions> {
    let mut runs: Vec<SimOptions> =
        Schedule::ALL.iter().map(|&schedule| SimOptions { schedule, ..SimOptions::default() }).collect();
    for seed in 0..seeds {
        runs.push(SimOptions {
            schedule: Schedule::Random { seed },
            async_completion: AsyncCompletion::Random(seed),
            ..SimOptions::default()
        });
    }
    runs
}

// ---------------------------------------------------------------------------------------------
// (1) The b32 payload codec on the host
// ---------------------------------------------------------------------------------------------

/// The payload the 16-byte-aligned mailbox carries: nine words of every supported type.
struct Msg {
    a: Int32,
    b: Int64,
    c: Float16,
    d: Boolean,
    e: Float32,
    f: Int8,
    g: BFloat16,
    h: Int16,
}

impl B32Payload for Msg {
    const NAME: &'static str = "Msg";
    fn fields() -> Vec<(&'static str, DType)> {
        use DType::*;
        vec![("a", I32), ("b", I64), ("c", F16), ("d", Bool), ("e", F32), ("f", I8), ("g", BF16), ("h", I16)]
    }
    fn to_fields(&self) -> Vec<(&'static str, AnyValue)> {
        vec![
            ("a", self.a.into()),
            ("b", self.b.into()),
            ("c", self.c.into()),
            ("d", self.d.into()),
            ("e", self.e.into()),
            ("f", self.f.into()),
            ("g", self.g.into()),
            ("h", self.h.into()),
        ]
    }
    fn from_fields(p: &DecodedPayload) -> Self {
        Msg {
            a: p.get("a"),
            b: p.get("b"),
            c: p.get("c"),
            d: p.get("d"),
            e: p.get("e"),
            f: p.get("f"),
            g: p.get("g"),
            h: p.get("h"),
        }
    }
}

cute_struct! {
    /// The payload of the 4-byte-aligned mailbox, declared as a shared-storage struct.
    struct Small { p: f32, q: i8, r: BF16 }
}

cute_struct! {
    /// Three mailboxes' storage at 16-, 8- and 4-byte aligned offsets (0, 40 and 60).
    struct Mail {
        #[align(16)] m16: MemRange<i32>[9],
        pad: i32,
        #[align(8)] m8: MemRange<i32>[4],
        pad2: i32,
        m4: MemRange<i32>[3],
    }
}

fn codec8() -> B32PayloadCodec {
    B32PayloadCodec::new(&[("x", DType::I64), ("y", DType::F32), ("z", DType::I16)], Some("Wide"))
}

#[test]
fn codec_words_chunks_and_sizes() {
    let msg = experimental::b32_payload::<Msg>();
    assert_eq!(msg.name(), "Msg");
    assert_eq!(msg.field_names(), ["a", "b", "c", "d", "e", "f", "g", "h"]);
    assert_eq!((msg.storage_words(), msg.tx_bytes(), msg.required_alignment()), (9, 36, 16));
    let words: Vec<(usize, usize)> = msg.fields().iter().map(|f| (f.word, f.words)).collect();
    assert_eq!(words, [(0, 1), (1, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]);
    assert_eq!(experimental::dsmem_payload::<Msg>(), msg);
    // Chunks by the base alignment: v4 at 16-byte aligned words with four left, v2 at 8.
    let v = |c: Vec<(usize, Vec<usize>)>| c.into_iter().map(|(n, w)| (n, w[0])).collect::<Vec<_>>();
    assert_eq!(v(msg.chunks_for_alignment(16)), [(4, 0), (4, 4), (1, 8)]);
    assert_eq!(v(msg.chunks_for_alignment(32)), [(4, 0), (4, 4), (1, 8)]);
    assert_eq!(v(msg.chunks_for_alignment(8)), [(2, 0), (2, 2), (2, 4), (2, 6), (1, 8)]);
    assert_eq!(v(msg.chunks_for_alignment(4)), (0..9).map(|i| (1, i)).collect::<Vec<_>>());
    let wide = codec8();
    assert_eq!((wide.storage_words(), wide.tx_bytes(), wide.required_alignment()), (4, 16, 16));
    assert_eq!(v(wide.chunks_for_alignment(8)), [(2, 0), (2, 2)]);
    let one = B32PayloadCodec::new(&[("t", DType::I32)], None);
    assert_eq!((one.name(), one.required_alignment()), ("B32PayloadCodec", 4));
    let two = B32PayloadCodec::new(&[("t", DType::I64)], None);
    assert_eq!(two.required_alignment(), 8);
    // From a shared-storage struct: its scalar members, named after it.
    let small = B32PayloadCodec::from_struct(&Small::layout());
    assert_eq!(small.name(), "Small");
    assert_eq!(small.field_names(), ["p", "q", "r"]);
    assert_eq!(small.storage_words(), 3);
    // The struct's offsets: 0, 40 and 60.
    let l = Mail::layout();
    let off: Vec<usize> = ["m16", "m8", "m4"].iter().map(|n| l.member(n).offset).collect();
    assert_eq!(off, [0, 40, 60]);
}

// ---------------------------------------------------------------------------------------------
// (2) The accumulator planner on the host
// ---------------------------------------------------------------------------------------------

fn plan_cols(
    c: usize,
    s: usize,
    aux: &[TmemAllocationRequest],
    cap: usize,
    depth: usize,
    max_depth: usize,
    overlap: bool,
    overlap_align: usize,
) -> AccumulatorPlanSelection {
    plan_accumulator_buffering_columns(
        c,
        s,
        aux,
        cap,
        depth,
        max_depth,
        overlap,
        overlap_align,
        4,
        AccumulatorOverlapPolicy::MinimumLegalFit,
    )
}

/// (kind, storage depth, sync depth, stride, physical, overlap, release ordinal, offsets, packed)
type Row = (AccumulatorBufferingKind, usize, usize, usize, usize, usize, usize, Vec<usize>, usize);

fn row(s: &AccumulatorPlanSelection) -> Row {
    (
        s.kind,
        s.storage_depth,
        s.synchronization_depth,
        s.stage_stride_columns,
        s.physical_columns,
        s.overlap_columns,
        s.release_after_t2r_ordinal,
        s.allocation_offsets.clone(),
        s.packed_columns,
    )
}

#[test]
fn planner_matches_hand_computed_tables() {
    use AccumulatorBufferingKind::*;
    let sf = || vec![TmemAllocationRequest::new("sfa", 16, 4), TmemAllocationRequest::new("sfb", 16, 4)];
    let cases: Vec<(AccumulatorPlanSelection, Row)> = vec![
        // Two 128-column views fit: disjoint, released after the last of four 32-column subtiles.
        (plan_cols(128, 32, &[], 512, 2, 2, false, 1), (Disjoint, 2, 2, 128, 256, 0, 3, vec![0], 256)),
        // 2 x 256 + 32 columns of scale factors do not fit; no overlap allowed: one view, the
        // requests packed after it.
        (plan_cols(256, 64, &sf(), 512, 2, 2, false, 1), (Disjoint, 1, 1, 256, 256, 0, 3, vec![0, 256, 272], 288)),
        // With overlap: the smallest overlap whose footprint fits is 32 (480 + 32 = 512), released
        // after subtile 0 (ceil(32 / 64) - 1).
        (plan_cols(256, 64, &sf(), 512, 2, 2, true, 1), (Overlapping, 2, 1, 224, 480, 32, 0, vec![0, 480, 496], 512)),
        // Strides must be multiples of 64: overlap 64 (stride 192), footprint 448.
        (plan_cols(256, 64, &sf(), 512, 2, 2, true, 64), (Overlapping, 2, 1, 192, 448, 64, 0, vec![0, 448, 464], 480)),
        // An overlap of 40 columns drained in 16-column subtiles releases after subtile 2.
        (plan_cols(64, 16, &[], 88, 2, 2, true, 1), (Overlapping, 2, 1, 24, 88, 40, 2, vec![0], 88)),
        // Deeper disjoint buffering when asked for and allowed.
        (plan_cols(100, 32, &[], 512, 4, 4, false, 1), (Disjoint, 4, 4, 100, 400, 0, 3, vec![0], 400)),
        (plan_cols(100, 32, &[], 512, 4, 3, false, 1), (Disjoint, 3, 3, 100, 300, 0, 3, vec![0], 300)),
        (plan_cols(200, 32, &[], 512, 4, 4, false, 1), (Disjoint, 2, 2, 200, 400, 0, 6, vec![0], 400)),
        // A wanted depth of 1 never overlaps.
        (plan_cols(256, 64, &sf(), 512, 2, 1, true, 1), (Disjoint, 1, 1, 256, 256, 0, 3, vec![0, 256, 272], 288)),
        // Requests at their own alignments: 128 -> 128 (align 1), 133 -> 136 (align 8).
        (
            plan_cols(
                64,
                64,
                &[TmemAllocationRequest::new("a", 5, 1), TmemAllocationRequest::new("b", 3, 8)],
                512,
                2,
                2,
                false,
                1,
            ),
            (Disjoint, 2, 2, 64, 128, 0, 0, vec![0, 128, 136], 139),
        ),
    ];
    for (k, (got, want)) in cases.iter().enumerate() {
        assert_eq!(row(got), *want, "case {k}");
    }
    // The accumulator alignment pads the start (an alignment of 8 columns still starts at 0).
    let s = plan_accumulator_buffering_columns(
        32,
        8,
        &[TmemAllocationRequest::new("x", 4, 32)],
        512,
        2,
        2,
        false,
        1,
        8,
        AccumulatorOverlapPolicy::MinimumLegalFit,
    );
    assert_eq!((s.allocation_offsets.clone(), s.packed_columns), (vec![0, 64], 68));
}

/// The one-view accumulator layout of a `(M, N)` f32 tile in TMEM (lane stride 2^16).
fn acc_layout(m: usize, n: usize) -> Layout {
    Layout::from_modes([Layout::new((m, n), (1usize << 16, 1usize)), Layout::new(1, 0), Layout::new(1, 0)])
}

#[test]
fn planner_builds_the_storage_layout() {
    let opts = |cap: usize, overlap: bool| AccumulatorBufferingOptions {
        epilogue_tile: Some(Tiler::from(&IntTuple::from((128, 8)))),
        tmem_capacity_columns: cap,
        allow_overlap: overlap,
        ..Default::default()
    };
    let plan = plan_accumulator_buffering(&acc_layout(128, 32), DType::F32, &opts(48, true));
    assert_eq!(row(plan.selection()), (AccumulatorBufferingKind::Overlapping, 2, 1, 16, 48, 16, 1, vec![0], 48));
    assert!(plan.uses_overlap());
    let l = plan.storage_layout();
    assert_eq!(l.rank(), 4, "{l}");
    assert_eq!(l.mode(3).size(), 2);
    assert_eq!(l.call(128 * 32), 16, "view 1 starts 16 columns on: {l}");
    let plan = plan_accumulator_buffering(&acc_layout(128, 32), DType::F32, &opts(512, false));
    assert_eq!(row(plan.selection()), (AccumulatorBufferingKind::Disjoint, 2, 2, 32, 64, 0, 3, vec![0], 64));
    assert_eq!(plan.storage_layout().call(128 * 32), 32);
    // A 16-bit accumulator, one element per 32-bit column (strides in 16-bit elements: lanes
    // 2^17, columns 2): 64 columns, drained 16 at a time.
    let opts16 = AccumulatorBufferingOptions {
        epilogue_tile: Some(Tiler::from(&IntTuple::from((128, 16)))),
        ..Default::default()
    };
    let acc16 =
        Layout::from_modes([Layout::new((128, 64), (1usize << 17, 2usize)), Layout::new(1, 0), Layout::new(1, 0)]);
    let plan = plan_accumulator_buffering(&acc16, DType::F16, &opts16);
    assert_eq!((plan.one_stage_columns(), plan.release_after_t2r_ordinal()), (64, 3));
    assert_eq!(plan.storage_layout().call(128 * 64), 128, "64 columns on, in 16-bit elements");
    // A request from a layout: 128 lanes x 4 bytes of 8-bit scale factors (strides in bytes:
    // lanes 2^18) take one column.
    let r = TmemAllocationRequest::from_layout("sf", &Layout::new((128, 4), (1usize << 18, 1usize)), DType::U8, 16);
    assert_eq!((r.columns, r.alignment_columns), (1, 4));
}

/// The configuration of `dense_block_scaled_gemm.py` with its defaults: MXF8 (Float8E4M3FN A
/// and B, Float8E8M0FNU factors of 32), a 128 x 128 MMA (one CTA), K tile 4 x 32, Float16 D
/// (N-major), Float32 accumulators, SFA and SFB in TMEM at 16 bytes, overlap allowed.
#[test]
fn planner_for_the_block_scaled_gemm_example() {
    // The layout helpers need a kernel being built.
    Kernel::builder("block_scaled_plan").block_dim([128, 1, 1]).target(Arch::Sm100a).build(block_scaled_plan_check);
}

fn block_scaled_plan_check() {
    let mma = cute::utils::sm100::make_blockscaled_trivial_tiled_mma(
        DType::F8E4M3FN,
        DType::F8E4M3FN,
        OperandMajorMode::K,
        OperandMajorMode::K,
        DType::F8E8M0FNU,
        32,
        CtaGroup::ONE,
        (128, 128),
        OperandSource::SMEM,
    );
    let tiler = (128, 128, 128);
    let acc = mma.make_fragment_c::<f32>(&mma.partition_shape_c((128, 128))).layout().clone();
    let sfa_smem = cute::utils::make_smem_layout_sfa(&mma, tiler, 32, 1);
    let sfb_smem = cute::utils::make_smem_layout_sfb(&mma, tiler, 32, 1);
    let sfa = cute::utils::make_tmem_layout_sfa(&mma, tiler, 32, &sfa_smem.select(&[0, 1, 2]));
    let sfb = cute::utils::make_tmem_layout_sfb(&mma, tiler, 32, &sfb_smem.select(&[0, 1, 2]));
    let epi = cute::utils::compute_epilogue_tile_shape(
        &[128, 128, 128],
        false,
        LayoutEnum::ROW_MAJOR,
        DType::F16,
        None,
        None,
        None,
    );
    // One factor per row per MMA: the four k-blocks share the bytes of four columns (CUTLASS's
    // tmem_sf_frg tiles them into the atom's free bytes).
    let cols = |l: &Layout| compute_tmem_cols_from_layout(l, DType::F8E8M0FNU);
    assert_eq!((cols(&sfa), cols(&sfb)), (4, 4), "{sfa} / {sfb}");
    // K-block `kk` starts at byte `kk` of the first column (its scale-factor ID).
    let block = sfa.mode(0).size();
    assert_eq!((0..4).map(|kk| sfa.call(block * kk)).collect::<Vec<_>>(), [0, 1, 2, 3], "{sfa}");
    assert_eq!(compute_tmem_cols_from_layout(&acc, DType::F32), 128);
    let aux = vec![TmemAllocationRequest::new("sfa", cols(&sfa), 4), TmemAllocationRequest::new("sfb", cols(&sfb), 4)];
    let plan = plan_accumulator_buffering(
        &acc,
        DType::F32,
        &AccumulatorBufferingOptions {
            epilogue_tile: Some(epi),
            auxiliary_tmem_allocations: aux,
            allow_overlap: true,
            ..Default::default()
        },
    );
    // The epilogue tile is 128 x 32: four 32-column subtiles. 2 x 128 accumulator columns, SFA
    // at 256 and SFB at 260 fit easily: two disjoint views.
    assert_eq!(
        row(plan.selection()),
        (AccumulatorBufferingKind::Disjoint, 2, 2, 128, 256, 0, 3, vec![0, 256, 260], 264),
        "{plan}"
    );
}

// ---------------------------------------------------------------------------------------------
// (3) CLCPipeline on a small grid
// ---------------------------------------------------------------------------------------------

const CLC_TILES: usize = 7;
const CLC_THREADS: u32 = 96;

/// Each CTA starts on its own tile; the scheduler warp (warp 0) asks for the next tile through
/// a CLCPipeline of `stages` stages and decodes the responses with the two consumer warps.
/// Every warp records each tile it is handed (the CTA that ran it, plus 1) at `out[tile][warp]`.
fn clc_kernel(stages: usize) -> Kernel {
    Kernel::builder("cute_ext_sched_clc").block_dim([CLC_THREADS, 1, 1]).target(Arch::Sm100a).build(move || {
        let out = param_tensor::<i32>("out", Layout::row_major((CLC_TILES, 3)));
        let resp = make_smem_tensor_aligned::<i32>(Layout::col_major(4 * stages), 16);
        let pipe = CLCPipeline::create(stages, CLC_THREADS as i32, &resp);
        assert_eq!(pipe.raw_pipeline().class(), PipelineClass::ClcFetchAsync);
        let (lane, w) = (lane_idx(), warp_idx());
        let me = block_idx().x;
        let record = move |tile: Int32| if_(lane.eq(0), || out.store((tile, w), me + 1));
        let next = |p: &mut CLCPipeline<i32>| {
            let (_, i) = p.consumer_wait_and_get_stage(None);
            let (x, _, _, valid) = clc_response(&p.get_response_ptr(i));
            p.consumer_release_and_advance();
            (x, valid.ne(0))
        };
        if_else(
            w.eq(0),
            || {
                let (_, _, mut p) = while_carry(
                    (me, Boolean::from(true), pipe.clone()),
                    |(_, v, _)| v,
                    |(tile, _, mut p)| {
                        record(tile);
                        p.producer_acquire();
                        p.issue_next();
                        p.producer_commit_and_advance(false, false);
                        let (x, v) = next(&mut p);
                        (x, v, p)
                    },
                );
                p.producer_tail();
            },
            || {
                while_carry(
                    (me, Boolean::from(true), pipe.clone()),
                    |(_, v, _)| v,
                    |(tile, _, mut p)| {
                        record(tile);
                        let (x, v) = next(&mut p);
                        (x, v, p)
                    },
                );
            },
        );
    })
}

#[test]
fn clc_pipeline_hands_every_tile_out_once() {
    for stages in [1, 2] {
        let kernel = clc_kernel(stages);
        for opts in schedules(2) {
            let mut args = vec![KernelArg::buffer(&[0i32; CLC_TILES * 3])];
            kernel
                .simulate_with(LaunchConfig::new([CLC_TILES as u32, 1, 1], [CLC_THREADS, 1, 1]), &mut args, &opts)
                .unwrap_or_else(|e| panic!("{stages} stages, {:?}: {e}", opts.schedule));
            let got = args[0].to_vec::<i32>();
            for t in 0..CLC_TILES {
                let r = &got[t * 3..t * 3 + 3];
                assert!(r[0] > 0 && r.iter().all(|&x| x == r[0]), "tile {t}: {got:?}");
            }
            // The simulator keeps one CTA resident, so CTA 0 cancels and runs every other tile.
            let ran: Vec<i32> = got.iter().step_by(3).copied().collect();
            assert_eq!(ran, vec![1; CLC_TILES], "{stages} stages, {:?}", opts.schedule);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// (4) GroupedGemmSchedulerPipeline
// ---------------------------------------------------------------------------------------------

/// (M, N, K) of each group; an empty group in the middle.
const GROUPS: [[i32; 4]; 4] = [[256, 128, 64, 1], [128, 384, 64, 1], [0, 128, 64, 1], [300, 200, 64, 1]];
const CTA_TILE: [usize; 3] = [128, 128, 64];
const SCHED_CTAS: u32 = 2;
const MAX_TILES: usize = 8;
/// Fields per record: valid, linear index, group, tile M, tile N.
const FIELDS: usize = 5;

/// Every group's tiles in order: (group, tile along M, tile along N), M fastest.
fn group_tiles() -> Vec<(i32, i32, i32)> {
    let mut out = Vec::new();
    for (g, s) in GROUPS.iter().enumerate() {
        let (cm, cn) = ((s[0] + 127) / 128, (s[1] + 127) / 128);
        for i in 0..cm * cn {
            out.push((g as i32, i % cm, i / cm));
        }
    }
    out
}

/// Warp 0 runs the grouped tile scheduler and publishes each tile's record into a two-stage
/// SMEM ring (then a terminating record); warps 1-3 read every record and write it to
/// `out[cta][k][warp - 1]`.
fn grouped_kernel() -> Kernel {
    let total = group_tiles().len();
    Kernel::builder("cute_ext_sched_grouped").block_dim([128, 1, 1]).target(Arch::Sm100a).build(move || {
        let shapes = param_tensor::<i32>("shapes", Layout::row_major((GROUPS.len(), 4)));
        let out = param_tensor::<i32>("out", Layout::row_major((SCHED_CTAS as usize, MAX_TILES, 3, FIELDS)));
        let slots = make_smem_tensor::<i32>(Layout::row_major((2, FIELDS)));
        // The scheduler warp's 32 threads commit; each consumer warp releases once.
        let pipe = GroupedGemmSchedulerPipeline::create(2, 32, 3);
        assert_eq!(pipe.raw_pipeline().class(), PipelineClass::Async);
        let lane = lane_idx();
        let params = PersistentTileSchedulerParams::new((1, 1, total), [1, 1, 1], Default::default());
        if_else(
            warp_idx().eq(0),
            || {
                let mut sched = StaticPersistentGroupTileScheduler::create(
                    &params,
                    block_idx(),
                    grid_dim(),
                    CTA_TILE,
                    create_initial_search_state(),
                    GROUPS.len(),
                    &shapes,
                    false,
                );
                let work = sched.initial_work_tile_info();
                let publish = |p: &mut GroupedGemmSchedulerPipeline, fields: [Int32; FIELDS]| {
                    let (_, i) = p.producer_acquire_and_get_stage(None);
                    if_(lane.eq(0), || {
                        for (j, f) in fields.into_iter().enumerate() {
                            slots.store((i, j as i32), f);
                        }
                    });
                    p.producer_commit_and_advance(false, false);
                };
                let (_, _, mut p) = while_carry(
                    (work, sched, pipe.clone()),
                    |(w, _, _)| w.is_valid_tile(),
                    |(work, mut sched, mut p)| {
                        let r = work.group_search_result();
                        let linear = sched.scheduler().current_work_linear_idx();
                        publish(&mut p, [Int32::from(1), linear, r.group_idx, r.cta_tile_idx_m, r.cta_tile_idx_n]);
                        sched.advance_to_next_work();
                        let work = sched.get_current_work();
                        (work, sched, p)
                    },
                );
                publish(&mut p, [Int32::from(0); FIELDS]);
                p.producer_tail();
            },
            || {
                let (cta, w) = (block_idx().z, warp_idx() - 1);
                while_carry(
                    (Boolean::from(true), Int32::from(0), pipe.clone()),
                    |(v, _, _)| v,
                    |(_, k, mut p)| {
                        p.consumer_wait();
                        let (_, i) = p.get_consumer_stage();
                        let rec: Vec<Int32> = (0..FIELDS as i32).map(|j| slots.load((i, j))).collect();
                        p.consumer_release(true, false);
                        let next = p.increment_state(&p.consumer_state());
                        p.set_consumer_state(next);
                        let valid = rec[0].eq(1);
                        if_(valid & lane.eq(0) & k.lt(MAX_TILES as i32), || {
                            for (j, f) in rec.iter().enumerate() {
                                out.store((cta, k, w, j as i32), *f);
                            }
                        });
                        (valid, k + 1, p)
                    },
                );
            },
        );
    })
}

#[test]
fn grouped_scheduler_pipeline_feeds_three_warps() {
    let tiles = group_tiles();
    let kernel = grouped_kernel();
    let shapes: Vec<i32> = GROUPS.iter().flatten().copied().collect();
    for opts in schedules(2) {
        let mut args = vec![
            KernelArg::buffer(&shapes),
            KernelArg::buffer(&vec![-1i32; SCHED_CTAS as usize * MAX_TILES * 3 * FIELDS]),
        ];
        kernel
            .simulate_with(LaunchConfig::new([1, 1, SCHED_CTAS], [128, 1, 1]), &mut args, &opts)
            .unwrap_or_else(|e| panic!("{:?}: {e}", opts.schedule));
        let got = args[1].to_vec::<i32>();
        let at = |c: usize, k: usize, w: usize| &got[((c * MAX_TILES + k) * 3 + w) * FIELDS..][..FIELDS];
        let mut seen = vec![0; tiles.len()];
        for c in 0..SCHED_CTAS as usize {
            for k in 0..MAX_TILES {
                let r = at(c, k, 0);
                for w in 1..3 {
                    assert_eq!(at(c, k, w), r, "CTA {c}, record {k}: the warps disagree");
                }
                if r[0] == -1 {
                    continue;
                }
                let linear = r[1] as usize;
                // Static persistent order: CTA c takes linear tiles c, c + 2, ...
                assert_eq!(linear, c + k * SCHED_CTAS as usize, "{r:?}");
                assert_eq!((r[2], r[3], r[4]), tiles[linear], "tile {linear}");
                seen[linear] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// (5) The cluster mailbox between the two CTAs of a pair
// ---------------------------------------------------------------------------------------------

const ROUNDS: i32 = 4;
/// The output words per round: Msg (9), Wide (4), Small (3).
const MAIL_WORDS: usize = 16;

/// The value of field `j` (of `dtype`) in round `k` from CTA `src`.
fn field_value(dtype: DType, j: i32, k: Int32, src: Int32) -> AnyValue {
    let base = k * 16 + j + src * 256;
    match dtype {
        DType::I32 => (base * 3 - 50).into(),
        DType::I64 => (base.to::<i64>() * 0x1_0000_0001i64 - 3i64).into(),
        DType::F32 => (base.to::<f32>() * 0.25f32 - 2.0f32).into(),
        DType::F16 => (base.to::<f32>() * 0.5f32 - 8.0f32).to::<F16>().into(),
        DType::BF16 => ((base % 64).to::<f32>() - 30.0f32).to::<BF16>().into(),
        DType::Bool => (base % 3).eq(0).into(),
        DType::I8 => (base % 200 - 100).to::<i8>().into(),
        DType::I16 => (base * -37).to::<i16>().into(),
        _ => unreachable!(),
    }
}

/// The host's words for field `j` in round `k` from CTA `src` (as the kernel writes them).
fn field_words(dtype: DType, j: i32, k: i32, src: i32) -> Vec<i32> {
    let base = k * 16 + j + src * 256;
    match dtype {
        DType::I32 => vec![base * 3 - 50],
        DType::I64 => {
            let v = (base as i64).wrapping_mul(0x1_0000_0001) - 3;
            vec![v as i32, (v >> 32) as i32]
        }
        DType::F32 => vec![(base as f32 * 0.25 - 2.0).to_bits() as i32],
        DType::F16 => vec![(base as f32 * 0.5 - 8.0).to_bits() as i32],
        DType::BF16 => vec![((base % 64) as f32 - 30.0).to_bits() as i32],
        DType::Bool => vec![(base % 3 == 0) as i32],
        DType::I8 => vec![base % 200 - 100],
        DType::I16 => vec![base * -37],
        _ => unreachable!(),
    }
}

/// A decoded field as the words the kernel writes: Int64 as its two halves, floats as their
/// Float32 bits, the rest converted to Int32.
fn out_words(v: AnyValue) -> Vec<Int32> {
    match v.dtype() {
        DType::I64 => {
            let x = v.cast::<i64>();
            vec![(x & 0xFFFF_FFFFi64).to::<i32>(), (x >> 32i64).to::<i32>()]
        }
        DType::F32 => vec![v.cast::<f32>().bitcast::<i32>()],
        DType::F16 => vec![v.cast::<F16>().to::<f32>().bitcast::<i32>()],
        DType::BF16 => vec![v.cast::<BF16>().to::<f32>().bitcast::<i32>()],
        DType::Bool => vec![v.cast::<bool>().to::<i32>()],
        DType::I8 => vec![v.cast::<i8>().to::<i32>()],
        DType::I16 => vec![v.cast::<i16>().to::<i32>()],
        DType::I32 => vec![v.cast::<i32>()],
        _ => unreachable!(),
    }
}

fn codecs() -> [B32PayloadCodec; 3] {
    [experimental::b32_payload::<Msg>(), codec8(), B32PayloadCodec::from_struct(&Small::layout())]
}

/// Two CTAs of a cluster each send `ROUNDS` messages to the other through three mailboxes (16-,
/// 8- and 4-byte aligned storage: v4, v2 and scalar stores) over a one-stage
/// CTA_PAIR_PEER_LOCALE pipeline: warp 0 sends, warp 1 receives and writes the decoded fields.
fn mailbox_kernel() -> Kernel {
    Kernel::builder("cute_ext_sched_mailbox").block_dim([64, 1, 1]).target(Arch::Sm100a).build(|| {
        let out = param_tensor::<i32>("out", Layout::row_major((2, ROUNDS as usize, MAIL_WORDS)));
        let mail = Mail::allocate();
        let [c16, c8, c4] = codecs();
        let boxes = [
            ClusterMailbox::new(&c16, mail.m16.clone()),
            ClusterMailbox::new(&c8, mail.m8.clone()),
            ClusterMailbox::new(&c4, mail.m4.clone()),
        ];
        let count = |side| {
            experimental::get_arrival_count(
                1,
                StShared,
                LdShared,
                side,
                &ArrivalCountOptions::new().num_threads_per_cta(32).elect_one_sync(true).locale(CTA_PAIR_PEER_LOCALE),
            )
        };
        let (pc, cc) = (count(Side::Producer), count(Side::Consumer));
        assert_eq!((pc.as_constant(), cc.as_constant()), (Some(1), Some(1)));
        let pipe = GenericPipeline::create(StShared, LdShared, pc, cc, 1, Some(CTA_PAIR_PEER_LOCALE));
        let rank = block_idx_in_cluster();
        let peer = rank ^ 1;
        let lane = lane_idx();
        if_(warp_idx().eq(0), || {
            let mut p = for_range_carry(0, ROUNDS, 1, pipe.clone(), |k, mut p| {
                let (tok, _) = p.producer_acquire_and_get_stage(None);
                let mut j = 0;
                let mut vals = |c: &B32PayloadCodec| -> Vec<(String, AnyValue)> {
                    c.fields()
                        .iter()
                        .map(|f| {
                            j += 1;
                            (f.name.clone(), field_value(f.dtype, j - 1, k, rank))
                        })
                        .collect()
                };
                let v16 = vals(&c16);
                let msg = Msg {
                    a: v16[0].1.cast(),
                    b: v16[1].1.cast(),
                    c: v16[2].1.cast(),
                    d: v16[3].1.cast(),
                    e: v16[4].1.cast(),
                    f: v16[5].1.cast(),
                    g: v16[6].1.cast(),
                    h: v16[7].1.cast(),
                };
                boxes[0].send_payload(&tok, peer, &msg);
                for (b, c) in boxes[1..].iter().zip([&c8, &c4]) {
                    let v = vals(c);
                    let fields: Vec<(&str, AnyValue)> = v.iter().map(|(n, x)| (n.as_str(), *x)).collect();
                    b.send_to_pipeline_stage(&tok, peer, &fields);
                }
                p.producer_commit_and_advance(true, false);
                p
            });
            p.producer_tail();
        });
        if_(warp_idx().eq(1), || {
            for_range_carry(0, ROUNDS, 1, pipe.clone(), |k, mut c| {
                c.consumer_wait();
                // The first mailbox through the payload type, the others by field.
                let m: Msg = boxes[0].recv_as();
                let mut words: Vec<Int32> = m.to_fields().into_iter().flat_map(|(_, v)| out_words(v)).collect();
                for b in &boxes[1..] {
                    let p = b.recv();
                    words.extend(p.values().iter().flat_map(|(_, v)| out_words(*v)));
                }
                assert_eq!(words.len(), MAIL_WORDS);
                if_(lane.eq(0), || {
                    for (i, w) in words.iter().enumerate() {
                        out.store((rank, k, i as i32), *w);
                    }
                });
                c.consumer_release(true, false);
                let next = c.increment_state(&c.consumer_state());
                c.set_consumer_state(next);
                c
            });
        });
        cluster_sync();
    })
}

#[test]
fn mailbox_exchanges_mixed_payloads_between_a_pair() {
    let kernel = mailbox_kernel();
    let mut want = Vec::new();
    for dst in 0..2 {
        let src = dst ^ 1;
        for k in 0..ROUNDS {
            let mut j = 0;
            for c in codecs() {
                for f in c.fields() {
                    want.extend(field_words(f.dtype, j, k, src));
                    j += 1;
                }
            }
        }
    }
    for opts in schedules(3) {
        let mut args = vec![KernelArg::buffer(&vec![0i32; 2 * ROUNDS as usize * MAIL_WORDS])];
        kernel
            .simulate_with(LaunchConfig::new([2, 1, 1], [64, 1, 1]).with_cluster([2, 1, 1]), &mut args, &opts)
            .unwrap_or_else(|e| panic!("{:?}: {e}", opts.schedule));
        assert_eq!(args[0].to_vec::<i32>(), want, "{:?}", opts.schedule);
    }
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("st.async") || mlir.contains("st_async"), "no st.async in the MLIR");
}

// ---------------------------------------------------------------------------------------------
// (6) Accumulator buffering in a TMEM producer/consumer kernel
// ---------------------------------------------------------------------------------------------

const ACC_N: usize = 32;
const SUB: usize = 8;
const ACC_ROUNDS: usize = 6;

/// How the consumer drains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drain {
    /// As the plan says.
    Planned,
    /// Forwards, releasing after the first subtile: the producer's next view overwrites
    /// columns not yet read.
    EarlyForward,
}

fn acc_plan(capacity: usize) -> AccumulatorBufferingPlan {
    plan_accumulator_buffering(
        &acc_layout(128, ACC_N),
        DType::F32,
        &AccumulatorBufferingOptions {
            epilogue_tile: Some(Tiler::from(&IntTuple::from((128, SUB)))),
            tmem_capacity_columns: capacity,
            allow_overlap: true,
            ..Default::default()
        },
    )
}

/// Warps 0-3 write each round's 128 x 32 tile of `input` into the view the plan's producer
/// side hands them (`tcgen05.st`, then an R2T commit); warps 4-7 drain the consumer view in
/// 8-column subtiles (`tcgen05.ld`) in the drain's order, releasing through it, and store the
/// tile to `output`.
fn acc_kernel(plan: AccumulatorBufferingPlan, drain: Drain) -> Kernel {
    Kernel::builder("cute_ext_sched_acc").block_dim([256, 1, 1]).target(Arch::Sm100a).build(move || {
        let input = param_tensor::<f32>("input", Layout::row_major((ACC_ROUNDS, 128, ACC_N)));
        let output = param_tensor::<f32>("output", Layout::row_major((ACC_ROUNDS, 128, ACC_N)));
        let buf = experimental::allocate::<f32>(AddressSpace::Tmem, plan.storage_layout().clone(), 32);
        let pipe = GenericPipeline::create(Sm100CopyR2t, Sm100CopyT2r, 128, 128, plan.synchronization_depth(), None);
        let tile_of = |view: &Tensor<f32>| {
            let m0 = view.layout().mode(0);
            make_tensor(view.ptr(), Layout::from_modes([m0.mode(0), m0.mode(1)]))
        };
        let t = thread_idx().x;
        if_else(
            warp_idx().lt(4),
            || {
                let mut p = for_range_carry(0, ACC_ROUNDS as i32, 1, pipe.clone(), |r, mut p| {
                    let view = plan.producer_acquire_and_get_view(&mut p, &buf);
                    let tile = tile_of(&view);
                    let src = input.slice((r, .., ..));
                    let st = make_tmem_copy(&make_copy_atom::<f32>(St32x32bOp::new(Repetition::X32)), &tile);
                    let thr = st.get_slice(t);
                    let frag = make_fragment_like::<f32, _>(&thr.partition_s(&src));
                    basic_copy(&thr.partition_s(&src), &frag);
                    copy(&st, &frag, &thr.partition_d(&tile));
                    p.producer_commit_and_advance(false, false);
                    p
                });
                p.producer_tail();
            },
            || {
                let ct = t - 128;
                for_range_carry(0, ACC_ROUNDS as i32, 1, pipe.clone(), |r, mut c| {
                    let (view, mut d) = plan.consumer_wait_and_get_view(&mut c, &buf);
                    let tile = tile_of(&view);
                    let dst = output.slice((r, .., ..));
                    let count = ACC_N / SUB;
                    for o in 0..count {
                        let j = match drain {
                            Drain::Planned => d.subtile_index(o, count),
                            Drain::EarlyForward => Int32::from(o as i32),
                        };
                        let sub_l = Layout::new((128, SUB), (1usize << 16, 1usize));
                        let sub = make_tensor(tile.ptr().offset(j * SUB as i32), sub_l);
                        let dsub = make_tensor(dst.ptr().offset(j * SUB as i32), Layout::new((128, SUB), (ACC_N, 1)));
                        let ld = make_tmem_copy(&make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X8)), &sub);
                        let thr = ld.get_slice(ct);
                        let frag = make_fragment_like::<f32, _>(&thr.partition_d(&dsub));
                        copy(&ld, &thr.partition_s(&sub), &frag);
                        match drain {
                            Drain::Planned => {
                                d.after_t2r(o);
                            }
                            Drain::EarlyForward if o == 0 => {
                                d.after_t2r(d.plan().release_after_t2r_ordinal());
                            }
                            Drain::EarlyForward => {}
                        }
                        fence_view_async_tmem_load();
                        basic_copy(&frag, &thr.partition_d(&dsub));
                    }
                    d.into_pipeline()
                });
            },
        );
    })
}

fn acc_args() -> (Vec<KernelArg>, Vec<f32>) {
    let input: Vec<f32> = (0..ACC_ROUNDS * 128 * ACC_N).map(|i| i as f32 - 7000.0).collect();
    (vec![KernelArg::f32(input.clone()), KernelArg::f32(vec![0.0; input.len()])], input)
}

#[test]
fn accumulator_buffering_drains_disjoint_and_overlapping_views() {
    for (cap, kind) in [(512, AccumulatorBufferingKind::Disjoint), (48, AccumulatorBufferingKind::Overlapping)] {
        let plan = acc_plan(cap);
        assert_eq!(plan.kind(), kind);
        let kernel = acc_kernel(plan.clone(), Drain::Planned);
        for opts in schedules(3) {
            let (mut args, want) = acc_args();
            kernel
                .simulate_with(LaunchConfig::new([1, 1, 1], [256, 1, 1]), &mut args, &opts)
                .unwrap_or_else(|e| panic!("{kind:?} {:?}: {e}", opts.schedule));
            assert_eq!(args[1].as_f32(), want.as_slice(), "{kind:?}");
        }
    }
}

/// Releasing the overlapping view after its first (forward) subtile lets the next round's
/// view overwrite shared columns the epilogue has not read: the race detector reports it.
#[test]
fn accumulator_early_release_is_a_race() {
    let kernel = acc_kernel(acc_plan(48), Drain::EarlyForward);
    let mut reported = false;
    for opts in schedules(8) {
        let (mut args, _) = acc_args();
        if let Err(e) = kernel.simulate_with(LaunchConfig::new([1, 1, 1], [256, 1, 1]), &mut args, &opts) {
            let e = e.to_string();
            assert!(e.contains("TMEM") && e.contains("tcgen05."), "{e}");
            reported = true;
        }
    }
    assert!(reported, "no schedule reported the early release");
}

// ---------------------------------------------------------------------------------------------
// (7) Compiling for sm_100a
// ---------------------------------------------------------------------------------------------

fn assemble(kernel: &Kernel, tc: &Toolchain) -> String {
    let ptx = kernel.compile_ptx(tc, Arch::Sm100a).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    let ptxas = std::env::var("CUTE_PTXAS").unwrap_or_else(|_| "ptxas".into());
    let dir = std::env::temp_dir().join(format!("cute_ext_sched_{}_{}", kernel.name(), std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("k.ptx");
    std::fs::write(&file, &ptx).unwrap();
    match std::process::Command::new(&ptxas).arg("-arch=sm_100a").arg(&file).arg("-o").arg(dir.join("k.cubin")).output()
    {
        Ok(out) => assert!(out.status.success(), "{}: ptxas: {}", kernel.name(), String::from_utf8_lossy(&out.stderr)),
        Err(_) => eprintln!("skipping the assembly: no ptxas found"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    ptx
}

#[test]
fn kernels_compile_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = assemble(&clc_kernel(2), &tc);
    assert!(ptx.contains("clusterlaunchcontrol.try_cancel") && ptx.contains("mbarrier.expect_tx"), "{ptx}");
    assemble(&grouped_kernel(), &tc);
    let ptx = assemble(&mailbox_kernel(), &tc);
    assert!(ptx.contains("st.async") && ptx.contains(".v4.b32") && ptx.contains(".v2.b32"), "{ptx}");
    let ptx = assemble(&acc_kernel(acc_plan(48), Drain::Planned), &tc);
    assert!(ptx.contains("tcgen05.ld") && ptx.contains("tcgen05.st"), "{ptx}");
}

// ---------------------------------------------------------------------------------------------
// (8) Diagnostics
// ---------------------------------------------------------------------------------------------

#[test]
fn diagnostics() {
    // A mailbox send with a stage token of a CTA-locale pipeline.
    let e = in_kernel(|| {
        let mail = Mail::allocate();
        let mb = ClusterMailbox::new(&codec8(), mail.m8.clone());
        let mut p = GenericPipeline::create(StShared, LdShared, 1, 1, 1, None);
        let (tok, _) = p.producer_acquire_and_get_stage(None);
        let x = Int64::from(1i64);
        mb.send_to_pipeline_stage(
            &tok,
            1,
            &[("x", x.into()), ("y", Float32::from(1.0).into()), ("z", Int16::from(1i16).into())],
        );
    });
    assert!(e.contains("CTA_PAIR_PEER_LOCALE") && e.contains("!lir.pipeline<1, ST_SHARED -> LD_SHARED>"), "{e}");
    // Unsupported payload types, wrong fields, wrong storage.
    let e = build_panic(|| {
        B32PayloadCodec::new(&[("ok", DType::I32), ("bad", DType::F64)], None);
    });
    assert!(e.contains("`bad`") && e.contains("Float64") && e.contains("no b32 encoding"), "{e}");
    let e = build_panic(|| {
        B32PayloadCodec::new(&[], None);
    });
    assert!(e.contains("at least one field"), "{e}");
    let e = in_kernel(|| {
        codec8().encode_fields(&[("x", Int64::from(1i64).into()), ("w", Int32::from(1).into())]);
    });
    assert!(e.contains("no value for [\"y\", \"z\"]") && e.contains("[\"w\"] are not fields"), "{e}");
    let e = in_kernel(|| {
        let mail = Mail::allocate();
        ClusterMailbox::new(&codec8(), mail.m16.clone());
    });
    assert!(e.contains("takes 4 b32 words") && e.contains("holds 9"), "{e}");
    cute_struct! {
        struct Nested { a: i32, r: MemRange<i32>[2] }
    }
    let e = build_panic(|| {
        B32PayloadCodec::from_struct(&Nested::layout());
    });
    assert!(e.contains("member `r`") && e.contains("scalars"), "{e}");
    // A plan over the TMEM capacity.
    let e = build_panic(|| {
        plan_cols(600, 32, &[], 512, 2, 2, true, 1);
    });
    assert!(e.contains("600 columns") && e.contains("capacity of 512"), "{e}");
    let e = build_panic(|| {
        plan_cols(500, 32, &[TmemAllocationRequest::new("sf", 16, 4)], 512, 2, 2, false, 1);
    });
    assert!(e.contains("sf (16 columns, aligned to 4)") && e.contains("need 516"), "{e}");
    // Overlap is never chosen without allow_overlap: a plan that only fits overlapping falls
    // back to one view, and an overlapping selection whose numbers break the invariants (an
    // overlap on a two-stage pipeline, an overlap of a whole view) is refused.
    let no = plan_cols(256, 64, &[], 480, 2, 2, false, 1);
    assert_eq!((no.kind, no.storage_depth, no.overlap_columns), (AccumulatorBufferingKind::Disjoint, 1, 0));
    let good = plan_cols(256, 64, &[], 480, 2, 2, true, 1);
    assert_eq!((good.kind, good.overlap_columns), (AccumulatorBufferingKind::Overlapping, 32));
    let layout = acc_layout(128, 256);
    let e = build_panic(|| {
        AccumulatorBufferingPlan::new(
            AccumulatorPlanSelection { synchronization_depth: 2, ..good.clone() },
            layout.clone(),
        );
    });
    assert!(e.contains("two views on a one-stage pipeline"), "{e}");
    let e = build_panic(|| {
        AccumulatorBufferingPlan::new(
            AccumulatorPlanSelection {
                overlap_columns: 0,
                stage_stride_columns: 256,
                physical_columns: 512,
                ..good.clone()
            },
            layout.clone(),
        );
    });
    assert!(e.contains("more than 0 and less than"), "{e}");
    let e = build_panic(|| {
        AccumulatorBufferingPlan::new(AccumulatorPlanSelection { overlap_columns: 8, ..no.clone() }, layout.clone());
    });
    assert!(e.contains("disjoint views share no columns"), "{e}");
    let e = build_panic(|| {
        plan_accumulator_buffering(&acc_layout(128, 32), DType::F32, &AccumulatorBufferingOptions::default());
    });
    assert!(e.contains("give the epilogue tile or the epilogue subtile columns"), "{e}");
    let e = build_panic(|| {
        TmemAllocationRequest::from_layout("sf", &acc_layout(128, 4), DType::U8, 6);
    });
    assert!(e.contains("6 bytes") && e.contains("multiple of"), "{e}");
    // A view taken through a pipeline of the wrong depth.
    let e = in_kernel(|| {
        let plan = acc_plan(512);
        let buf = experimental::allocate::<f32>(AddressSpace::Tmem, plan.storage_layout().clone(), 32);
        let mut p = GenericPipeline::create(Sm100CopyR2t, Sm100CopyT2r, 128, 128, 1, None);
        plan.producer_acquire_and_get_view(&mut p, &buf);
    });
    assert!(e.contains("plan's pipeline has 2 stages, this one 1"), "{e}");
    // issue_next outside acquire/commit.
    let e = in_kernel(|| {
        let resp = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
        let p = CLCPipeline::create(1, 32, &resp);
        p.issue_next();
    });
    assert!(e.contains("issue_next") && e.contains("no stage is acquired"), "{e}");
    let e = in_kernel(|| {
        let resp = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
        let mut p = CLCPipeline::create(1, 32, &resp);
        p.producer_acquire();
        p.issue_next();
        p.producer_commit(false, false);
        p.issue_next();
    });
    assert!(e.contains("no stage is acquired"), "{e}");
    let e = in_kernel(|| {
        let resp = make_smem_tensor_aligned::<i32>(Layout::col_major(6), 16);
        CLCPipeline::create(2, 32, &resp);
    });
    assert!(e.contains("not 2 compact 16-byte slots"), "{e}");
}
