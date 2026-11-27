# Blackwell and Rubin (SM100, SM103, SM107, SM120)

## Blackwell layouts and UMMA descriptors

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.SmemLayoutAtomKind`, `tcgen05.make_smem_layout_atom`, `make_sparse_smem_layout_atom` | `tcgen05::SmemLayoutAtomKind` (with `MN_SW128_32B`), `tcgen05::make_smem_layout_atom(_dtype)`, `make_sparse_smem_layout_atom(_dtype)`. `MN_SW128_32B` has CuTe C++'s 4 rows (CuTe DSL's has 8; both tile alike K-first, and only the 4-row atom stays canonical when tiled MN-first) |
| `tcgen05.tile_to_mma_shape` | `tcgen05::tile_to_mma_shape(atom, shape, order)` (0-based order, static shapes) |
| `tcgen05.make_umma_smem_desc`, `helpers.int_to_smem_descriptor`, `smem_descriptor_to_int` | `tcgen05::make_umma_smem_desc` (build-time fields ORed with the runtime start address; sm_107's 15-bit fields from the target; `next_src` unsupported), `umma_desc_fields`, `SmemDescriptor`, `UmmaLayoutType`, `UmmaDescEncoding`, `UmmaOperandMap` (the PTX-ISA decoder for the simulator), `int_to_smem_descriptor`, `smem_descriptor_to_int` |
| `cutlass.utils.BlockScaledBasicChunk`, `Sm103BlockScaledBasicChunk`, `tile_atom_to_shape_SF` | `utils::BlockScaledBasicChunk`, `Sm103BlockScaledBasicChunk`, `tile_atom_to_shape_sf` (static shapes) |
| `cutlass.utils.sm100.compute_acc_tmem_cols_per_stage`, `compute_epilogue_tile_size`, `cutlass.utils.compute_epilogue_tile_shape` | `utils::sm100::compute_acc_tmem_cols_per_stage`, `compute_epilogue_tile_size`, `utils::compute_epilogue_tile_shape` |
| `cute.arch.get_max_tmem_alloc_cols`, `get_min_tmem_alloc_cols`, `is_tmem_allocation_exclusive`, `SM100_TMEM_*` | the same names, taking an `Arch` (`a`/`f` targets map to their base) |

Dynamic shared-memory addresses: a buffer in the dynamic window is addressed as the window's pointer plus its start (LLVM 21's `--expand-strided-metadata` dropped a `memref.view`'s offset), so kernels over 48 KB compile correctly with LLVM 21, 22 and 23 (`tests/dyn_smem_address.rs`).

## Tensor memory

| CuTe DSL | cute-rs |
|---|---|
| `cute.make_ptr(dtype, addr, AddressSpace.tmem)`, `cute.make_tensor(tmem_ptr, layout)` | `make_ptr::<T>(addr, AddressSpace::Tmem, align)` and `make_tensor`: offsets count elements of `T` and move the 32-bit address by `offset >> log2(32 / bits(T))` (CuTe C++'s `tmem_ptr`), so a lane is `1 << 16` elements of a 32-bit type and `1 << 17` of a 16-bit one. Slicing, partitioning, composition and recasts work; loads, stores and fills are refused (only tcgen05 copies and MMAs touch TMEM) |
| `cute.arch.alloc_tmem(num_columns, smem_ptr, is_two_cta, arch=)`, `dealloc_tmem(tmem_ptr, num_columns, is_two_cta, arch=)` | `alloc_tmem(n, &slot, is_two_cta, Arch)`, `dealloc_tmem(&ptr, n, is_two_cta, Arch)`: LLVM's convergent `nvvm.tcgen05.alloc`/`dealloc`; a constant count is checked (a power of two from 32 to 512; on sm_107 more than 512 is the exclusive form, inline PTX) |
| `cute.arch.relinquish_tmem_alloc_permit(is_two_cta)` | `relinquish_tmem_alloc_permit(is_two_cta)` |
| `cute.arch.retrieve_tmem_ptr(element_type, alignment, ptr)` | `retrieve_tmem_ptr::<T>(alignment, &slot)`; the slot is a 32-bit shared `Ptr` or a `cute.struct` scalar member (`TmemAddressSlot`) |
| `cutlass.utils.find_tmem_tensor_col_offset` | `find_tmem_tensor_col_offset(&tensor)` (static layouts) |

The simulator gives each CTA its own TMEM and checks the PTX rules: warp-collective issue with
uniform operands, one `cta_group` per kernel, CTA pairs (`cta_group::2`, even and odd
`%cluster_ctarank`) that both allocate the same columns, frees that match a live allocation's
start, count, group and exclusiveness, nothing left allocated at exit, no allocation after
relinquishing the permit, `retrieve_tmem_ptr` of a live allocation, and the address write as the
allocating warp's (readers need a barrier). Allocations take the lowest free range aligned to
their size. Targets: sm_100a and sm_110a, or the sm_100f/sm_110f families from PTX 8.8; the
exclusive forms the sm_107f family with PTX 9.4.

## TMEM loads and stores

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.Ld16x64bOp`, `Ld16x128bOp`, `Ld16x256bOp`, `Ld16x32bx2Op`, `Ld32x32bOp(repeat, pack)`; `tcgen05.Repetition`, `tcgen05.Pack` | `tcgen05::Ld32x32bOp::new(Repetition::X64).with_pack(Pack::Pack16bIn32b)` and peers, into `CopyOp::TmemLoad`; `make_copy_atom::<T>(op)` (8-, 16- or 32-bit `T`; packing 16-bit) and `copy(atom, tmem, fragment)`: one convergent `nvvm.tcgen05.ld` per atom, into a register fragment (`make_fragment`, not `make_rmem_tensor`). Repetitions up to 128 registers per thread; the `.16x32bx2` half split is as many columns as the first access spans |
| `tcgen05.St..Op(repeat, unpack)`, `tcgen05.Unpack` | `tcgen05::St32x32bOp::new(r).with_unpack(Unpack::Unpack32bIn16b)` and peers (`CopyOp::TmemStore`), `nvvm.tcgen05.st`, from a fragment or `make_rmem_tensor` |
| `tcgen05.make_tmem_copy(atom, tmem_tensor)` | `tcgen05::make_tmem_copy(&atom, &tmem)`: CuTe C++'s, the atom over the four warps' lane quadrants and the tensor's columns (128 threads) |
| `cute.arch.fence_view_async_tmem_load()`, `fence_view_async_tmem_store()`, `fence_view_async_tmem_op(kind)` | the same names (`TmemOpKind::Load`/`Store`): `nvvm.tcgen05.wait <load>`/`<store>` |
| (the compiler's `tcgen05.fence::before/after_thread_sync`) | `tcgen05_fence_before_thread_sync()`, `tcgen05_fence_after_thread_sync()`; in kernels with TMEM loads or stores the backend also puts `before` ahead of every barrier, named-barrier arrive, mbarrier arrive and cluster arrive, and `after` behind every barrier sync, mbarrier wait and cluster wait (PTX ISA 9.7.18.6), so ported code needs none |

The simulator checks warp-collective issue (32 lanes, one address), the lane quadrant of each
warp, that every cell is in a live allocation and (for loads) was written, by a write ordered
before the load; a load's registers are pending until the warp's `wait::ld` (reading one earlier
is reported), and a store is in flight until its `wait::st` (another warp's load, a racing
store and `dealloc_tmem` before it are reported). `.unpack::16b` leaves zero in the columns'
high halves in the simulator (undefined on hardware).

## TMEM load-reduce, packed f32x2 and Rubin packed math

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.LdRed32x32bOp(repeat, redOp, nan)`, `LdRed16x32bx2Op(repeat, redOp, nan, half_split_off)`, `tcgen05.TmemLoadRedOp` (`MAX`, `MIN`, `MAXABS`, `MINABS`), called as `cute.copy_atom_call(atom, src, [data, stat])` | `tcgen05::LdRed32x32bOp::new(Repetition::X64, TmemLoadRedOp::Max).with_nan(true)` and `LdRed16x32bx2Op` (into `CopyOp::TmemLoadRed`), `make_copy_atom::<f32>(op)` and `copy_atom_call(&atom, &tmem_view, &[&data, &stat])`: one `nvvm.tcgen05.ld.red` (LLVM 23) per call, loading Float32 values (`.x2` to `.x128`) into `data` and each thread's `min`/`max` (of absolute values for the `ABS` forms; NaNs ignored unless `nan`, which makes any NaN the result; -0 < +0) into the first element of `stat`. Both are pending until `fence_view_async_tmem_load()`. `copy` refuses these atoms. The reduction runs over the values each thread loaded: `.16x32bx2` threads `i` and `i + 16` share a lane but hold different columns, so each reduces its own. Targets: sm_110a, sm_103f and sm_110f (so sm_103a), PTX 8.8 (9.0 for sm_110), not sm_100a |
| `cute.arch.add_packed_f32x2`, `sub_packed_f32x2`, `mul_packed_f32x2`, `fma_packed_f32x2(a, b[, c], rnd='rn', ftz=None)`, `exp_packed_f32x2` (deprecated) | the same names on `(Float32, Float32)` with `PackedF32Options { rnd: Option<RoundingMode>, ftz }` (`None` is `rnd='none'`; fma needs a direction): inline `add/sub/mul/fma{.rnd}{.ftz}.f32x2` on 64-bit register pairs (no NVVM op has them), sm_100 and PTX 8.6. Each element is the scalar f32 operation; the simulator uses m1b's float arithmetic for both lanes. `exp_packed_f32x2` is a `mul_packed_f32x2` by log2(e) and two `ex2.approx.ftz.f32` |
| `add_packed_{f16x2,bf16x2}_f32x2_f32x2`, `sub_packed_{f16x2,bf16x2}_f32x2_f32x2`, `add_packed_f32x2_{f16x2,bf16x2}_f32x2`, `sub_packed_f32x2_{f16x2,bf16x2}_f32x2`, `fma_packed_f32x2_{f16x2,bf16x2}_f32x2_f32x2` | the same names; the half operand is an `Int32` pair or a 2-tuple (`Half2`), results of the first kind are `(Float16, Float16)` / `(BFloat16, BFloat16)` (one rounding of the exact Float32 sum, to nearest even), the others `(Float32, Float32)` with `rnd: Option<RoundingMode>`; inline `add.rn.f16x2.f32x2`, `add{.rnd}.f32x2.f16x2`, `fma{.rnd}.f32x2.f16x2` and the `bf16x2` forms. Rubin only: the sm_107f family, PTX 9.4 |
| `cute.arch.cvt_f32x4_to_fnv8e5m3x4(src)` | `cvt_f32x4_to_fnv8e5m3x4([Float32; 4]) -> [FloatNV8E5M3FNU; 4]`: two inline `cvt.rp.satfinite.ue5m3x2.f32` (first source into the upper byte, so element 0 lands in the low byte), rounding toward +infinity, saturating to `0xfe` (114688), a NaN giving `0xff`; the sign is ignored, as for the UE8M0 pair conversions. sm_107f, PTX 9.4 |

Toolchain ceilings (what each form was checked with): the `.f32x2` forms are PTX 8.6, so
`ptxas -arch=sm_100a` assembles them (`tests/ldred_packed.rs`); `tcgen05.ld.red` is PTX 8.8,
which `ptxas` 12.8 lacks, so it is checked with LLVM 23's `mlir-opt` (`nvvm.tcgen05.ld.red`
verifies and lowers to `tcgen05.ld.red.sync.aligned.32x32b.x4.max.abs.NaN.f32`; LLVM 21 and 22
have no such op); the mixed-precision packed forms and `.ue5m3x2` are PTX 9.4, which no
installed LLVM or `ptxas` has, so they are checked in the simulator and as MLIR text only, and
their PTX spelling follows the PTX ISA's pattern for the neighbouring mixed forms, not a
checked assembler.

## SM120 block-scaled warp MMAs

`cute::warp::{MmaMXF4Op, MmaMXF4NVF4Op, MmaMXF8Op, MmaMXF8F6F4Op}` issue the GeForce Blackwell
`mma.sync ... .block_scale` forms (sm_120a/sm_121a with PTX 8.7, the sm_120f family with 8.8):
MXF4 and NVFP4 at m16n8k64, MXF8 and the mixed FP4 x FP8 pairs (`MXF8F6F4_SUPPORTED_PAIRS`) at
m16n8k32, all accumulating in Float32. `gemm` takes each operand with its scale factors,
`gemm(&mma, &d, (&a, &sfa), (&b, &sfb), &c)` (CuTe DSL's `[a, sfa]` lists), or reads them from the
atom's `Field::SFA`/`SFB` (`set_tensor`). `cute::utils::sm120` has `get_permutation_mnk`,
`thrfrg_sfa/sfb`, `partition_fragment_sfa/sfb` (plus `partition_sfa/sfb` for the matching source
view), `get_layout_sfa_tv/sfb_tv` and `sm120_get_smem_store_op`; `cute::utils` has
`sm120_make_smem_layout_sfa/sfb`. The FP4 operand of the mixed form is an `F4E2M1FN` fragment like
the others; `gemm` puts it in the 8-bit containers the instruction reads (CuTe DSL kernels shift
the bytes themselves). The MMA lowers to `nvvm.mma.block_scale` (LLVM 22) or, with
`CompileOptions::mma_inline_ptx`, to inline PTX (LLVM 21). No available GPU runs SM120: the
simulator (`src/sim/exec/blockscale.rs`, whose block-scale arithmetic tcgen05 can reuse) is held
to plain Rust, and the PTX is assembled with `ptxas -arch=sm_120a`.

## Cluster launch control

| CuTe DSL | cute-rs |
|---|---|
| `cute.arch.issue_clc_query(mbar_ptr, clc_response_ptr, multicast=True)` | `issue_clc_query(&mbar_ptr, &response_ptr, multicast)`: `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{.multicast::cluster::all}.b128` as inline PTX (LLVM 21 has no NVVM op); sm_100 and PTX 8.6, the multicast form sm_100a-class targets. The response pointer is 16-byte aligned shared memory of any byte-sized type |
| `cute.arch.clc_response(result_addr) -> (x, y, z, is_valid)` | `clc_response(&ptr) -> (Int32, Int32, Int32, Int32)`: a 16-byte `ld.shared.b128` and `query_cancel.is_canceled` / `get_first_ctaid::{x,y,z}` in one inline-PTX statement, so no Int128 type is needed |
| `cutlass.pipeline.PipelineClcFetchAsync.create(*, num_stages, producer_group, consumer_group, tx_count, barrier_storage=None, producer_mask=None, consumer_mask=None, cta_layout_vmnk=None, defer_sync=False, name='')` | `pipeline::PipelineClcFetchAsync::create(n, producer_group, consumer_group, tx_count, PipelineClcFetchAsyncOptions { barrier_storage, cta_layout_vmnk, defer_sync, name })` (the masks are left out: CuTe DSL overrides both). The scheduler warp of CTA 0 acquires (CTA 0's empty barrier), then lane `i` arms CTA `i`'s full barrier with the 16 expected bytes (`PipelineOp::ClcLoad`); `producer_commit` is a no-op; consumers release to CTA 0's empty barrier (at cluster scope in a cluster), so its count covers the whole cluster's consumers |
| `ClcDynamicPersistentTileSchedulerParams(problem_shape_ntile_mnl, cluster_shape_mnk, swizzle_size=1, raster_along_m=True)`, `get_grid_shape()`; `ClcDynamicPersistentTileScheduler.create(params, block_idx, grid_dim, clc_response_ptr, insert_fence=True)`, `initial_work_tile_info`, `get_current_work`, `work_tile_info_from_clc_response`, `advance_to_next_work(mbarrier_addr)`, `num_tiles_executed` | `utils::ClcDynamicPersistentTileSchedulerParams::new(tiles, cluster, ClcDynamicPersistentTileSchedulerOptions { swizzle_size, raster_along_m, fallback_cluster_shape_mnk })` (the fallback shape's schedule is kept, `fallback()`, and `create` takes it in the fallback arm of `mixed_cluster_arms`) and `utils::ClcDynamicPersistentTileScheduler::create(&params, block_idx(), grid_dim(), &response_ptr, insert_fence)`, the same methods; `Carry` with only the tile count. The visiting order is the static scheduler's (`swizzle_and_rasterize`, FastDivmod for N-major). Deviation: a swizzled grid launches the padded layout's clusters (CuTe DSL's unpadded count leaves tiles unvisited when the swizzle does not divide the clusters), whose padding tiles the kernel skips |
| `cute.experimental.utils.is_preferred_cluster(shape)`; scheduler params' `fallback_cluster_shape_mnk=`, `_select_active_params`, `runtime_cluster_dims()`; the mixed-cluster expansion pass and `smem_merge_branch_allocs=True` | `utils::is_preferred_cluster([x, y, z])` (`%cluster_nctaid` compared mode by mode) and `utils::mixed_cluster_arms(preferred, Some(fallback), \|arm\| body)`: no compiler pass; the body closure is built twice, under `if_else(is_preferred_cluster(..))`, each time with the arm's static `ClusterArm::cluster_shape_mnk`, so cluster-shape math stays static per arm (shared memory both arms use is allocated before the call). While an arm is built, `current_cluster_arm()` names it: `ClcDynamicPersistentTileScheduler::create` picks the params' fallback schedule in the fallback arm (the build-time `_select_active_params`), and schedulers, or anything checked with `check_cluster_shape_in_arm(what, shape)`, built for another shape panic. `check_mixed_cluster_shapes(preferred, fallback, grid)` checks the tutorial's constraints (equal depth, the fallback divides the preferred shape, the preferred shape divides the grid). The simulator's CLC queue hands each cluster only clusters of its own shape |

The simulator keeps the grid's clusters that have not started in a queue: `try_cancel` removes the
first one that has not launched (each at most once; a cancelled cluster never starts) and answers
"not cancelled" once it is empty. `SimOptions::clc_launch_seed` launches clusters at the front of
the queue at random before each query, so which are left to cancel changes; `SimReport::clc_cancelled`
counts the cancellations. The response is the simulator's own 16-byte encoding (x, y, z, then a tag
word with the cancelled bit), delivered as an async-proxy write that completes 16 bytes of tx on
each target CTA's mbarrier, as the simulator's other in-flight copies do; a multicast query after a CTA of the
cluster has exited is reported. `tests/clc_scheduling.rs` runs a persistent kernel whose clusters
claim more tiles through CLC for several grids, cluster shapes, swizzles, resident-cluster counts
and seeds (each tile once, cancelled clusters never run, the static scheduler's order), plus the
misuse reports; Blackwell CLC cannot run on the SM89 pod, so the PTX is assembled with
`ptxas -arch=sm_100a`.

## MMA-aware TMA atoms and the first Blackwell GEMM

| CuTe DSL | cute-rs |
|---|---|
| `cute.nvgpu.make_tiled_tma_atom_A(op, gmem_tensor, smem_layout, mma_tiler_mnk, tiled_mma, cluster_shape_vmnk=None, *, internal_type=None)`, `make_tiled_tma_atom_B(...)` -> `TmaInfo` | `nvgpu::make_tiled_tma_atom_a(op, &gmem, smem_layout, mma_tiler_mnk, &tiled_mma, cluster_shape_vmnk)` and `_b` (`Option<Shape>`, e.g. `Some((1, 2, 2, 1).into())`); `_a_with`/`_b_with(.., &TmaAtomOptions)` for `internal_type`. `op` is `CopyBulkTensorTileG2SOp` or `...MulticastOp`; `smem_layout` one stage `(MMA, MMA_M, MMA_K)` of `make_smem_layout_a/b` or the staged layout (first stage used). C++ CuTe's `make_tma_atom_A/B_sm100`: the CTA tile is the value mode of the tcgen05 `TiledMma`'s `thrfrg_A/B` of the tile's identity layout, `(MMA, MMA_M, MMA_K)`, so the box matches the SMEM layout's and `tma_partition` takes `group_modes(0, 3)` of the SMEM tensor and of `thr_mma.partition_a(gA)`. The multicast count is the cluster's N mode for A and M mode for B (1 for the non-multicast op or without a cluster). The tiled MMA must be tcgen05 |
| `cute.nvgpu.make_tiled_tma_atom_C(op, gmem_tensor, smem_layout, mma_tiler_mnk, tiled_mma, *, internal_type=None)` | `nvgpu::make_tiled_tma_atom_c(op, &gmem, smem_layout, mma_tiler_mnk, &tiled_mma)` with `CopyBulkTensorTileS2GOp` or `CopyReduceBulkTensorTileS2GOp`: the `(TILE_M, TILE_N)` tile as `thrfrg_C` partitions it; no multicast |
| `cute.nvgpu.make_im2col_tma_atom_A(op, gmem_tensor, smem_layout, mma_tiler_mnk, tiled_mma, filter_trs, upper_padding_dhw, lower_padding_dhw, stride_dhw, dilation_dhw, cluster_shape_vmnk=None, *, internal_type=None)` | `nvgpu::make_im2col_tma_atom_a(op, &gmem, smem_layout, mma_tiler_mnk, &tiled_mma, &conv, cluster_shape_vmnk)`, `conv` from `cpasync::Im2colOptions::conv(filter_srt, lower_padding_whd, upper_padding_whd, stride_whd, dilation_whd)` (W first: CuTe DSL's tuples reversed; its `internal_type` too, its `num_multicast` replaced by the cluster's N mode): `make_im2col_tma_atom` with the MMA's `(MMA, MMA_M, MMA_K)` tile of the GEMM's `(pixels, channels)` |

`fp16_gemm_0.py` is ported as the `blackwell_fp16_gemm` example (see Examples). What the
Python host function builds (tiled MMA, SMEM layouts, TMA atoms) is built at the start of the
kernel's build closure. Porting it needed one change beyond the API: the epilogue waits for each
TMEM load with `fence_view_async_tmem_load()` before reading its registers, which the simulator
requires to be explicit. `tests/tma_mma_atoms.rs` compares the atoms with `make_tiled_tma_atom`
and `make_im2col_tma_atom` on the explicit CTA tile and checks the multicast counts for `(2, 1)`
and `(2, 2)` clusters; `tests/examples.rs` simulates a 256x512x256 GEMM (C bit for bit, the
tcgen05.mma count and the global bytes read) and assembles its PTX for sm_100a.

## CTA pairs: 2-SM tcgen05 MMAs and 2-CTA TMA loads

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.MmaF16BF16Op(..., cta_group=tcgen05.CtaGroup.TWO, ...)`, and `MmaTF32Op`, `MmaI8Op`, `MmaF8F6F4Op`, `MmaFP8Op` with `CtaGroup.TWO`; `make_trivial_tiled_mma(..., CtaGroup.TWO, ...)` | the same constructors with `CtaGroup::TWO`: M of 128 or 256, N a multiple of 16 from 16 to 256 (32 for an 8-bit MN-major B), A in shared memory. The atom's `thr_id` is the pair (size 2) and A, B and C are split in row halves (C++'s `SM100_MMA_*_2x1SM_SS` traits), so `tiled_mma.get_slice(rank_in_pair)` partitions each CTA's half, `partition_shape_C` is `((M/2, N), 1, 1)`, and `make_fragment_C` lays it out as `UMMA::tmem_frg_2sm` (PTX Layout A for M = 256; Layout B for M = 128: columns `N/2..N` in lanes 64 to 127) |
| `cute.gemm(tiled_mma, ...)` with a pair MMA | `gemm` as for one CTA, called by a warp of the even CTA only (`tcgen05.mma.cta_group::2`); it reads A and B at the descriptors' addresses in both CTAs' shared memory and writes D into both CTAs' TMEM |
| `tcgen05.commit(mbar, mask, cta_group=CtaGroup.TWO)` | `tcgen05::commit(&mbar, Some(mask), CtaGroup::TWO)`; the mask names the CTAs whose mbarrier (at the same offset) hears of the completion, both CTAs of the pair for a pair MMA |
| `cpasync.CopyBulkTensorTileG2SOp(cta_group=CtaGroup.TWO)`, `CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)` (TiledTmaLoadEnum `sm_100_2sm`), `make_tiled_tma_atom_A/B` with a pair's tiled MMA | the same ops; `make_tiled_tma_atom_a/b` give each CTA's half box (the MMA's value mode). The load is `cp.async.bulk.tensor....cta_group::2` and signals the mbarrier at the same offset in the pair's even CTA (C++'s `Sm100MmaPeerBitMask`), so one CTA expects both halves' bytes. Refused for an sm_90 target; `.cta_group::2` loads need sm_100a, sm_101a, sm_110a or their f families |
| `alloc_tmem`/`dealloc_tmem`/`relinquish_tmem_alloc_permit(is_two_cta=True)`, `TmemAllocator(is_two_cta=True)` | as before: both CTAs of the pair allocate together |

The simulator runs a pair MMA as one instruction of the pair's tensor core: issued by an odd
CTA it is reported; its operand reads in the peer CTA are checked like the issuer's (each byte
written and ordered before the issue, which for the peer's loads means through the leader's
mbarrier), and the peer's operand bytes stay read until the MMA completes, so overwriting them
is reported; D's cells in both CTAs land at one time of both CTAs' tensor-core agents, which a
warp of either CTA observes by waiting on its own mbarrier of the commit. A commit without the
peer's bit leaves the peer's waiters in a deadlock, and a pair allocation made by one CTA only
never completes (also a deadlock). The MMAs and commits join the alloc, dealloc and relinquish
instructions in the check that all tcgen05 instructions of a kernel use one `cta_group`.
`tests/cta_pair.rs` runs a `(2, 1, 1)` cluster tile with raw mbarriers (M = 256 and 128, N =
128 and 256, f16 and bf16) against plain Rust, the five misuses, the atoms' split, and the
sm_100a PTX through `ptxas`. Not modelled yet: pair MMAs with A in TMEM (CUTLASS's duplicated
TMEM A) and `tcgen05.cp` with `cta_group::2`; gather4 and override loads
refuse `CtaGroup::TWO`.

The 2-CTA pipelines are simulated by `tests/cta_pair_pipelines.rs`:
the `blackwell_fp16_gemm` example in `fp16_gemm_1.py`'s configuration and on `(2, 2, 1)` and
`(4, 1, 1)` clusters (multicast in both modes, each multicast tile read from global memory
once per cluster), bit for bit; the pipelines' masks on two cluster shapes; a 2-CTA
warp-specialised kernel through `PipelineTmaUmma` and `PipelineUmmaAsync` with every built-in
schedule, and its misuses (the full barrier armed on the odd CTA, a deadlock with the pair's
copies in flight; a TMEM-sync mask without the peer, a deadlock of the peer's epilogue); a
Mamba2-style `PipelineTmaMultiConsumersAsync` kernel (a TMA warp, the MMA warp and two async
consumer warps reading each stage) with one-lane and per-lane signalling, and async
consumers counted per warp arriving from every lane (more arrivals than the barrier counts,
naming the `tcgen05.commit` whose arrival is owed). Its sm_100a PTX is assembled by `ptxas`.
Not covered: a multi-consumer kernel on a cluster (its signalling is checked, not run).

## Shared-memory-to-TMEM copies

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.Cp128x256bOp`, `Cp128x128bOp`, `Cp4x256bOp`, `Cp4x32x128bOp`, `Cp2x64x128b0213Op`, `Cp2x64x128b0123Op` (`cta_group`, `broadcast_factor`) | `tcgen05::Cp128x256bOp::new(CtaGroup::ONE)` and peers (`broadcast_factor()`), into `CopyOp::TmemCp`; `make_copy_atom::<T>(op)` (8-, 16- or 32-bit `T`). `copy(atom, desc_tensor, tmem)` is called by a converged warp and elects the lane that issues each `tcgen05.cp.cta_group::{1,2}.<shape>` (inline PTX; do not wrap it in `elect_one`); completion through `tcgen05::commit` |
| `tcgen05.make_s2t_copy(atom, tmem_tensor)`, `tcgen05.get_s2t_smem_desc_tensor(atom, smem_tensor)` | `tcgen05::make_s2t_copy(&atom, &tmem)` (CuTe C++'s `make_utccp_copy`, after `filter_zeros`; one thread per CTA of the group, `get_slice(0)`) and `tcgen05::get_s2t_smem_desc_tensor(&tiled, &partitioned_smem)` (`get_utccp_smem_desc_tensor`: the core matrices checked as a canonical K-major UMMA layout, descriptors in 16-byte steps; 15-bit fields on sm_107) |
| `tcgen05.append_s2t_broadcast_mode(op, smem_tensor)` | `tcgen05::append_s2t_broadcast_mode(op, &smem)`: `((MN..., b:0), K)` for the broadcast ops, the tensor unchanged otherwise; panics (CuTe DSL's `ValueError`) for a non-S2T op or a mode 0 that is not `(MN, K)`. A broadcast copy whose partitioned source lacks the zero-stride mode is refused, naming it |
| `cutlass.block.block_copy(tiled_copy, src, dst, **kwargs)`, `cutlass.utils.block.block_copy` (deprecated) | `cute::block::block_copy(&atom_or_tiled_copy, &src, &dst, CopyArgs)` (`cute::utils::block::block_copy` deprecated): TMA atoms through `tma_partition` for one CTA, S2T tiled copies through slice 0 and the descriptor view (a broadcast copy's source gets its broadcast mode back after `filter_zeros`). `tma_multicast` is not supported |

The simulator (`src/sim/exec/tmem_cp.rs`) decodes each descriptor with `UmmaOperandMap`, writes
TMEM by the PTX shapes (the broadcast forms replicate each row into 2 or 4 lane quadrants),
queues the copy on the CTA's tensor-core pipe with the MMAs (completed by `tcgen05.commit` and
an observed mbarrier phase), keeps the source in flight until then, requires `fence.proxy.async`
after generic writes of the source, and reports a copy issued by fewer than 32 lanes (inside
`elect_one`, say). The decompressing source formats (`.b8x16.b6x16_p32`, `.b4x16_p64`) are
reached only through CuTe DSL's experimental `nvvm_wrapper.tcgen05_cp`, not its copy atoms, and
are not modelled.

## SM100 epilogue helpers

| CuTe DSL | cute-rs |
|---|---|
| `utils.sm100.get_tmem_load_op(cta_tile_shape, layout_d, elem_ty_d, elem_ty_acc, epi_tile, use_2cta_instrs, *, tmem_warp_shape_mn=None)` | `utils::sm100::get_tmem_load_op::<Acc>(&cta_tile, layout_d, d_dtype, &epi_tile, use_2cta, tmem_warp_shape_mn) -> CopyAtom<Acc>`, and `select_tmem_load_op(.., acc_dtype, ..) -> TmemLoadChoice { shape, repeat, pack }` for the choice as data. Ported from CuTe C++'s `sm100_get_tmem_load_op` and `TMEM::op_repeater`: the tile is split by the TMEM warp layout (`(2, 2)` for a CTA pair with 64 rows per CTA, else `(4, 1)`); a warp's part must have 16 or 32 datapaths. `32x32b` for 32 datapaths (`16x256b` for M-major 16-bit and 8-bit D), `16x128b`, `16x256b` or `16x32bx2` for 16, `.pack::16b` for 16-bit accumulators, the widest repetition the columns allow. Panics (not `ValueError`) with its own messages for 8 datapaths, a warp layout that does not divide the tile, 6- and 4-bit D with 16 datapaths, and unserved width pairs |
| `utils.sm100.get_smem_store_op(layout_d, elem_ty_d, elem_ty_acc, tiled_tmem_load)` | `utils::sm100::get_smem_store_op::<D, Acc>(layout_d, acc_dtype, &tiled_tmem_load)`, and `select_smem_store_op(layout_d, d_bits, acc_bits, &load_op) -> CopyOp`: `StMatrix8x8x16bOp` (4 matrices for a repeated `16x256b` load of 16-bit D, a repeated `16x128b` load of N-major 32-bit D or a packed `16x128b` load of 16-bit accumulators; 2 for `.x1`; transposed for M-major D), `StMatrix16x8x8bOp` for M-major 8-bit D from `16x256b`, else a plain store. The simulator checks the fragments against the load bit for bit. `StMatrix` takes 32-bit elements (one per register, untransposed only: an extension of the `StMatrix` atom, with its f32 shared-memory buffer) for N-major Float32 D |
| `utils.sm100.get_smem_layout_atom_epi(layout, element_type, epi_tile)`, `make_smem_layout_epi(epi_dtype, epi_layout, epi_tile, epi_stage)` | `utils::sm100::get_smem_layout_atom_epi(layout, dtype, &epi_tile)` (the M or N extent of the right inverse of each tile mode, through `get_smem_layout_atom_ab`) and `make_smem_layout_epi(dtype, layout, &epi_tile, stages)` (Boolean is Int8; the staged layout in the order `(1, 0, 2)` for M-major, `(0, 1, 2)` for N-major). M-major Float32 gets `MN_SW128_32B`, which `make_tiled_tma_atom` cannot express for a TMA store yet (deferred); direct stores and N-major Float32 work |
| `utils.gemm.sm100.transform_partitioned_tensor_layout(tensor)` | `utils::gemm::sm100::transform_partitioned_tensor_layout(&t)`: `((A_M, A_N), MMA_M, MMA_N, ..)` to `((A_M, MMA_M), (A_N, MMA_N), ..)`; the accumulator's trailing extent-1 mode of the atom is dropped |
| `utils.gemm.sm100.epilogue_tmem_copy_and_partition(gemm_kernel, tidx, tAcc, tCgC, epi_tile, use_2cta_instrs)`, `epilogue_smem_copy_and_partition(gemm_kernel, tiled_copy_t2r, tTR_rC, tidx, sC)` | `utils::gemm::sm100::epilogue_tmem_copy_and_partition(&cfg, tidx, &t_acc, &t_cg_c, &epi_tile, use_2cta)` -> `TmemCopyPartition { tiled_copy_t2r, t_tr_t_acc, t_tr_r_acc }` and `epilogue_smem_copy_and_partition(&cfg, &tiled_copy_t2r, &t_tr_r_c, tidx, &s_c)` -> `SmemCopyPartition { tiled_copy_r2s, t_rs_r_c, t_rs_s_c }`. `gemm_kernel`, which CuTe DSL duck-types, is `EpilogueConfig` (`cta_tile_shape_mnk`, `c_layout`, `c_dtype`, `acc_dtype`, `use_2cta_instrs`, `tmem_warp_shape_mn`, `epilog_sync_bar_id`, `epilogue_warp_id`, `num_c_stage`, `iter_acc_early_release_in_epilogue`; `EpilogueConfig::new` gives the usual four warps on barrier 2) |
| `utils.gemm.sm100.epilogue_tma_store(gemm_kernel, epi_tidx, warp_idx, tma_atom_c, tCtAcc_base, sC, tCgC_base, epi_tile, num_tiles_executed, epilogue_op, mma_tile_coord_mnl, acc_consumer_state, acc_pipeline, c_pipeline)` | `utils::gemm::sm100::epilogue_tma_store::<Acc, C, _, _>(&cfg, tidx, warp_idx, &atom, &t_ct_acc, &s_c, &t_cg_c, &epi_tile, num_tiles_executed, &op, &[tile_m, tile_n], &state, &acc_pipeline, &c_pipeline) -> PipelineState` (any `Pipeline`; the state is the next). `epilogue_op` is `&dyn Fn(TensorSsa<C>) -> TensorSsa<C>`. The loop is unrolled when the kernel is built; each subtile's `tcgen05.ld` is followed by an explicit wait (`fence_view_async_tmem_load`), which cute-rs requires; the accumulator stage is released by one elected thread of each warp, so the accumulator pipeline's consumer group counts the epilogue warps. The caller runs `c_pipeline.producer_tail()` |
| `utils.gemm.sm100.epilogue(gemm_kernel, epi_tidx, tCtAcc_base, tCgC_base, epi_tile, epilogue_op, mma_tile_coord_mnl, acc_consumer_state, acc_pipeline, tCcC_base=None, mC_mnl=None, overlapping_accum=False)` | `utils::gemm::sm100::epilogue::<Acc, C, _>(&cfg, tidx, &t_ct_acc, &t_cg_c, &epi_tile, &op, &coord, &state, &pipeline, Some(&t_cc_c), Some(&c), overlapping)`: direct stores, `R2G` atoms of the widest vector (at most 128 bits where CuTe DSL allows 256; the pointer's alignment and the registers' and C's common run bound it) with no-allocate L1 eviction; one predicate per vector from its first element's coordinate against `mC.shape` (so the contiguous extent must be a whole number of vectors); `overlapping_accum` reads the stage from the phase, drains the subtiles backwards on phase 0 and releases after `iter_acc_early_release_in_epilogue` |

`examples/blackwell_fp16_gemm.rs` gained a hook for them: `Fp16GemmConfig::build_with(acc_consumer_arrivals, make_c, epilogue)` takes the declaration of C (and anything built from it, such as a TMA store atom) and a closure that drains the accumulator, handed an `EpilogueCtx` (the TMEM tensor, `ThrMma`, tile coordinates, the shared-memory allocator and the accumulator pipeline and state); `build()` is the tutorial's own epilogue. `tests/epilogue_helpers.rs` drains `fp16_gemm_0` (1 CTA, M = 128 and 64) and `fp16_gemm_1` (a CTA pair, 2x2 warps for M = 128) through both epilogues for Float16, BFloat16 and Float32 D in both majors with an identity and a scaling operation, bit for bit against the CPU; the host tests list the choices worked from the C++ builder, and the sm_100a PTX is assembled with `ptxas`.

## Block-scaled tcgen05 MMAs

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.MmaMXF8F6F4Op(a_dtype, b_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)`, `MmaMXF8Op(ab_dtype, ...)` (deprecated) | `tcgen05::MmaMXF8F6F4Op::new(...)`, `tcgen05::MmaMXF8Op::new(...)`: `tcgen05.mma.cta_group::{1,2}.kind::mxf8f6f4.block_scale`, K = 32, UE8M0 per 32 elements, FP6/FP4 from the padded image (Uint8 tensors) |
| `tcgen05.MmaMXF4Op(instruction_shape, cta_group, a_src)`, `MmaMXF4NVF4Op(sf_dtype, ...)` | `tcgen05::MmaMXF4Op::new(...)`, `tcgen05::MmaMXF4NVF4Op::new(sf_dtype, ...)`: `.kind::mxf4.block_scale.scale_vec::2X` / `.kind::mxf4nvf4.block_scale.scale_vec::4X`, K = 64, packed Float4E2M1FN tensors, K-major; sm_100a, sm_103a, sm_107a, sm_110a only (not the `f` families) |
| `tcgen05.SM103MmaMXF4Op`, `SM103MmaMXF4NVF4Op` | `tcgen05::SM103MmaMXF4Op::new(...)`, `tcgen05::SM103MmaMXF4NVF4Op::new(sf_dtype, ...)`: K = 96, `.block32` / `.block16`, sm_103a and sm_107a |
| `cute.gemm(atom, d, [a, sfa], [b, sfb], c)`, `atom.set(tcgen05.Field.SFA/SFB, ptr)` | `gemm(&mma, &d, (&a, &sfa), (&b, &sfb), &c)` or `mma.set_tensor(Field::SFA, &sfa)`: `sfa`/`sfb` TMEM tensors of the scale-factor type (sliced at each MMA's `(MN, K)` block when they have those modes); the byte of the column the first factor is in becomes the descriptor's scale-factor ID. `tcgen05::tmem_sf_frg_layout(mn, sf_vec_size, nsf, rest)` gives the TMEM layout. No `DISABLE_OUTPUT_LANE` |
| `cutlass.utils.make_blockscaled_trivial_tiled_mma(a_dtype, b_dtype, a_leading_mode, b_leading_mode, sf_dtype, sf_vec_size, cta_group, mma_tiler_mn, a_source)`, `sm103_make_blockscaled_trivial_tiled_mma(sf_dtype, sf_vec_size, cta_group, mma_tiler_mn, a_source)` | `utils::sm100::make_blockscaled_trivial_tiled_mma(...)`, `utils::sm100::sm103_make_blockscaled_trivial_tiled_mma(...)`; unlike CuTe DSL, a `sf_dtype` other than UE8M0 with 32-element vectors (and an F8F6F4 `sf_vec_size` other than 32) panics rather than being ignored |

The simulator reads each CTA's scale factors from its TMEM (SFA for its rows, SFB for all N; in
a pair each CTA its own): row `32 c + l` in lane `l` of all four sub-partitions (which must
agree), column `c`, the factors in consecutive bytes from the scale-factor ID, continuing four
columns on past a column's fourth byte (the K = 96 `.block16` placement is cute-rs's reading of
CUTLASS's `sm107_tmem_sf_frg`, not checked against the PTX ISA's figure). A scale-factor cell
written by another warp's `tcgen05.cp` still in flight, or landed without the MMA's warp
observing its completion, is reported. The sm_103a forms need PTX 8.8, so they are checked
through mlir-opt only.

## Rubin (SM107) tcgen05 MMAs and collector buffers

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.SM107MmaFP8Op(a_dtype, b_dtype, acc_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode, a_collector_op, b_collector_op)` | `tcgen05::SM107MmaFP8Op::new(...)`: `.kind::f8f6f4`, E4M3/E5M2 each, Float16 or Float32 accumulator, K = 64 (the default; instruction-descriptor bit 29) or 32; M 64/128 on one CTA, 128/256 on a pair; MN-major A and B allowed |
| `tcgen05.SM107BlockScaledMmaMXF8F6F4Op(a_dtype, b_dtype, sf_dtype, sf_vec_size, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode, a_collector_op, b_collector_op)` | `tcgen05::SM107BlockScaledMmaMXF8F6F4Op::new(...)`: `.kind::mxf8f6f4.block_scale`, K = 64 (bit 31), UE8M0 per 32 elements; M = 128 on one CTA, 256 on a pair. FloatNV8E5M3FNU operands are accepted by the constructor but `gemm` refuses them: no published source gives their descriptor format code |
| `tcgen05.SM107MmaMXF4NVF4Op(sf_dtype, sf_vec_size, instruction_shape, cta_group, a_src, a_collector_op, b_collector_op)` | `tcgen05::SM107MmaMXF4NVF4Op::new(...)`: `.kind::mxf4nvf4.block_scale.block16` / `.block32` (CUTLASS's spelling), K = 128 (descriptor K-size 2: bit 3), UE8M0, UE4M3 or UE5M3 (`FloatNV8E5M3FNU`, scale format 2) per 16 or 32 elements; a row's eight factors take two four-column groups in TMEM |
| `tcgen05.CollectorOp.DISCARD/LASTUSE/FILL/USE` | `tcgen05::CollectorOp::{DISCARD, LASTUSE, FILL, USE}` (also `cute::CollectorOp`); `.collector::a::<op>` / `.collector::b::<op>` after the kind's qualifiers, nothing for `DISCARD`. `op.with_collectors(a, b)` changes them (CuTe C++'s `MMA_Traits::with`) |

All three are the sm_107f family's (sm_107f and sm_107a, PTX ISA 9.4); `check_target` refuses
them elsewhere, naming the K or collector form. Their shared-memory descriptors use the sm_107
encoding (15-bit start address and LBO) whatever the kernel's target, and their K-major tiles are
64 bytes of K. **No installed toolchain has sm_107** (LLVM 23 stops at PTX 9.3, ptxas 12.8 at PTX
8.7), so the MMAs are inline PTX checked on the simulator and as MLIR text (verified with
mlir-opt); nothing is assembled, and the PTX spellings and descriptor bits follow CUTLASS 4.8.1's
`mma_sm107_umma.hpp` and `mma_sm107_desc.hpp` without an assembler's confirmation. The simulator
keeps each CTA's A and B collector buffers: `FILL` stores the operand read from shared memory,
`USE`/`LASTUSE` read it from the buffer (`LASTUSE` empties it, as does any `DISCARD` MMA), and a
`USE` with nothing filled, of another descriptor or form, or of a tile whose shared-memory bytes
changed since the `FILL` is reported. An A collector with A in TMEM panics when built (cute-rs's
reading of the PTX ISA, which could not be checked); `.ashift` is not modelled.
`tests/umma_rubin.rs` simulates each op and K exactly against plain Rust (UE5M3 codes from the
host codec), a B FILL/USE/LASTUSE sequence against the same MMAs without reuse, and the
collector, target and build-time diagnostics.

## Sparse tcgen05 MMAs

| CuTe DSL | cute-rs |
|---|---|
| `cute.get_sparse_elem_type(num_logical, num_phys, elem_type)`, `cute.make_sparse_elem(num_logical, elems)` | `get_sparse_elem_type(n, p, DType)` gives a build-time `SparseElemType` (fractional `width()`, not a `DType`); `make_sparse_elem(n, &elems)`. `recast_layout_to_sparse` / `recast_layout_from_sparse` rescale layouts; `recast_tensor_sparse::<T, U>(&t, ty)`, `recast_ptr_sparse` and `SmemAllocator::allocate_sparse_tensor` give a `SparseTensor` (logical layout over the physical pointer, `physical()` for the data); its `load`/`store` panic when the kernel is built |
| `tcgen05.MmaF16BF16SparseOp(ab_dtype, acc_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode, sparse_metadata_format)`, `MmaFP8SparseOp(...)` | `tcgen05::MmaF16BF16SparseOp::new(...)` (K = 32), `tcgen05::MmaFP8SparseOp::new(...)` (K = 64; E4M3 or E5M2): `tcgen05.mma.sp.cta_group::{1,2}.kind::{f16,f8f6f4}`, sparse flag in the descriptor. M = 128 on one CTA, 256 on a pair; A from shared memory only. A is partitioned and described compressed (`(M, K/2)` tiles, plain tensors of the stored type) |
| `cute.gemm(atom, d, [a, e], b, c)`, `atom.set(tcgen05.Field.METADATA, e)` | `gemm(&mma, &d, (&a, &e), &b, &c)` or `mma.set_tensor(Field::METADATA, &e)` (or `gemm_sparse`): `e` a TMEM tensor (sliced per MMA when it has `(MMA_M, MMA_K)` modes); its column's lowest bit becomes the descriptor's sparse selector. A missing metadata panics when built |
| `cutlass.utils.make_sparse_a_dtype`, `make_sparse_e_dtype`, `make_tensor_e_atom_layout`, `make_sparse_gmem_layout_e`, `make_sparse_smem_layout_e`, `make_sparse_tmem_layout_e`, `make_sparse_smem_layout_a` | `utils::sm100::` the same names; the E layouts count logical (one-bit) metadata elements (recast to Uint8 for byte tensors; one 128 x 128 atom is a 128x128b `tcgen05.cp`); `make_sparse_smem_layout_a` returns the compressed layout in stored elements |

The simulator reads A compressed and expands it with each CTA's metadata from its TMEM, by its
own formula for the 128-row metadata layout (`sparse_meta_place`; the PTX ISA's figures were not
available to check it against, it agrees with CUTLASS's `tmem_e_frg`); a field not naming two
increasing positions of four, a metadata address not at lane 0 of an even column, and the usual
TMEM ordering faults are reported. The block-scaled sparse forms, TF32/I8/FP4 sparse MMAs and
M = 64 are not modelled.

## Scale-factor SMEM/TMEM layouts and the NVFP4 tutorial GEMM

| CuTe DSL | cute-rs |
|---|---|
| `blockscaled_layout.make_smem_layout_sf(tile_shape, sf_vec_size, num_stages)` | `utils::make_smem_layout_sf(...)`: `BlockScaledBasicChunk` tiled over the tile K-chunk first, plus a stage mode |
| `blockscaled_layout.make_smem_layout_sfa/sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, num_stages)` | `utils::make_smem_layout_sfa/sfb(...)`: `(((128, MN_inst / 128), (sf_vec_size, nsf)), MMA_MN, MMA_K, STAGE)`, SFB rounded up to 128 rows |
| `blockscaled_layout.make_tmem_layout_sfa/sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, smem_layout)` | `utils::make_tmem_layout_sfa/sfb(...)`: `(MMA, MMA_MN, MMA_K)` in 8-bit TMEM element offsets, CuTe C++'s `UMMA::tmem_sf_frg` (mode 0 `((((32, MN/32), 4), (VS, NSF)))`, the `4` being the sub-partition replicas). CuTe DSL computes it in a closed-source op; cute-rs rederives it (the derivation is in `src/utils/blockscaled_layout.rs`) and checks it on the simulator. One CTA's SFB only: a CTA pair's (the 2x2 data path) panics |
| `sm103_make_smem_layout_sfa/sfb` (of `sm103_dense_blockscaled_gemm_persistent.py`) | `utils::sm103_make_smem_layout_sfa/sfb(...)` |
| `tcgen05.make_s2t_copy(atom, filter_zeros(tCtSFA))` on the SF layouts | the same, with `tcgen05::append_s2t_broadcast_mode` on the shared-memory side before `partition_S` (cute-rs spells out `Cp4x32x128bOp`'s broadcast) |

`nvfp4_gemm_0.py` is ported as the `blackwell_nvfp4_gemm` example (see Examples). The scale
factors' TMA atoms move 8-bit elements rather than the tutorial's `internal_type=Int16` (the same
boxes); with runtime `K` the scale factors' 32-row run and their K-chunk mode, contiguous in
global memory, stay two TMA dimensions (the planner folds runs into one dimension only over static
extents). `tests/examples.rs` simulates both scale-factor types (one problem wrapping the 4-stage
ring) bit for bit against a CPU reference with random factors per row and vector, and assembles the
sm_100a PTX with `ptxas`; `tests/umma_blockscaled.rs` checks the layouts against placements
written out by hand.

## Mixed-input GEMM helpers

| CuTe DSL | cute-rs |
|---|---|
| `cutlass.utils.mixed_input_helpers.TransformMode` | `utils::mixed_input_helpers::TransformMode::{ConvertOnly, ConvertScale}` (`from_granularity(gm, gk)`) |
| `get_transform_a_source(a_major)`, `get_tma_atom_kind(mcast, use_2cta_instrs, is_b)` | the same names; the TMA kind is a `utils::sm100::Sm100TmaLoadOp` (2-CTA only for B) |
| `get_copy_atom_a_transform(mma_dtype, use_2cta_instrs, transform_a_source, a_smem_shape, a_dtype)` | `get_copy_atom_a_transform::<M, A>(use_2cta, source, &a_smem_shape) -> TransformCopyAtom<M, A>` (`Tmem(CopyAtom<M>)` or `Smem(CopyAtom<A>)`), and `select_copy_op_a_transform` for the choice as a `CopyOp` |
| `is_valid_scale_granularity`, `is_valid_tensor_alignment`, `is_valid_mma_tiler_and_cluster_shape`, `get_divisibility(contiguous_dim_size, upper_bound=128)`, `is_shuffle_a` | the same names; majors are `OperandMajorMode` (C's a `LayoutEnum`); `is_shuffle_a` takes the toolchain's support as a `bool` (below). `get_divisibility` is `mixed_input_helpers::get_divisibility`, not the prelude's `get_divisibility` (CuTe DSL's `cute.get_divisibility`) |
| `get_gmem_layout_scale`, `get_smem_layout_scale`, `compute_smem_layout` | the same names (`get_smem_layout_scale` returns the tile and the per-stage and staged `ComposedLayout`s and panics with its own messages) |
| `scale_tma_partition`, `transform_partition`, `scale_partition` | the same names; `transform_partition` returns a `TransformPartition { src_copy_a, dst_copy_a, t_as_a_input, t_a_transform }`, `scale_partition` a `ScalePartition`. `scale_tma_partition` drops the stride-0 (repeated) scales from the tile before `tma_partition`, so each stage is one TMA copy |
| `epilog_gmem_copy_and_partition`, `epilog_smem_copy_and_partition`, `epilog_tmem_copy_and_partition` | the first ported (an `EpilogGmemPartition`); the other two are `utils::gemm::sm100::epilogue_smem/tmem_copy_and_partition` under these names |
| `cvt_tensor_a(src, dtype, shuffle)`, `cvt_tensor_a_mxf8(src, dtype)`, `cvt_tensor_a_biased(src, dtype)`, `cvt_i4x8_to_mxf8x8_biased`, `cvt_i4_mxf8_biased_intrinsic`, `store_transformed_a` | the same names (`cvt_tensor_a::<D>(&src, shuffle)`); the biased conversion's `lop3`/`prmt` are cute-rs's `lop3`/`prmt` instructions rather than inline PTX. `shuffle_a_host` is the host's shuffle (`mixed_input_host_utils.py`) |
| the grouped variants' `ContiguousGGSearchState`, `contiguous_group_search`, ... | re-exported from `utils` |

CuTe DSL takes the shuffled Int4-to-BFloat16 conversion (the scaled `.s2f6x2` cvt) when its target CUDA
is 13.1 or newer. cute-rs has no target CUDA version: the condition is the PTX ISA the toolchain emits,
9.1 or newer (`SHUFFLE_MIN_PTX_ISA`, `shuffle_supported_by(ptx)`; LLVM 23), passed by the caller to
`is_shuffle_a` and `cvt_tensor_a`. By default nothing is shuffled, which every supported toolchain
compiles. `tests/mixed_input.rs` checks the selectors and checks on the host, the conversions for all 16
Int4 values (natural, shuffled, E4M3 and bias-encoded) on the simulator against plain Rust, and the
reduced GEMM (see `blackwell_mixed_input_gemm` above). `PipelineAsyncUmma`'s producers now arrive on
the leader CTA of a pair at cluster scope, which the simulator needs to order the peer's writes before
the leader's MMA. Int4 tensors move through TMA as the packed 4-bit encoding (`16U4_ALIGN8B`).

## Rubin cutlass.utils helpers and the TMEM load with sparse compression

| CuTe DSL | cute-rs |
|---|---|
| `cutlass.utils.rubin_helpers.canonicalize_rubin_resource_arch(arch)` | `utils::sm107::canonicalize_rubin_resource_arch("sm_107a") -> Result<Arch, String>` (`Arch::Sm107` for `sm_107`, `sm_107a`, `sm_107f`; anything else is an error). The Rubin resources are already per-arch: `Arch::smem_capacity_bytes()` is 327 KiB (`validate::MAX_SHARED_BYTES` admits it) and `arch::get_max_tmem_alloc_cols` is 576 |
| `rubin_helpers.compute_epilogue_tile_shape(mma_op, cta_tile_shape, use_2cta_instrs, layout_d, elem_ty_d, layout_c=, elem_ty_c=)` | `utils::sm107::compute_epilogue_tile_shape(&MmaOp, ...)`: the `(128, 64)` / `(128, 32)` tiles of the FP4 `SM107MmaMXF4NVF4Op` on a 128 or 256 by 256 CTA-pair tile (as a `(128:1, n:1)` tiler), else SM100's. The table is CuTe DSL 4.8.1's; the CUTLASS C++ builder it comes from (`sm107_builder.inl`) is not in the 4.8.1 tree cute-rs reads, so it is not checked against it |
| `rubin_helpers.make_trivial_tiled_mma(a_dtype, b_dtype, a_leading_mode, b_leading_mode, acc_dtype, cta_group, mma_tiler, atom_layout_mnk, permutation_mnk, a_source, a_collector_op, b_collector_op)` | `utils::sm107::make_trivial_tiled_mma(...)` (`mma_tiler` is `(M, N, K)`; `permutation_mnk` an `Option<[Option<Layout>; 3]>`): `MmaF16BF16Op` (K 16), `MmaTF32Op` (K 8) and `SM107MmaFP8Op` (K 32 or 64) with the collector ops; the first two panic on a collector other than `DISCARD` |
| `TiledMmaWrapper.with_bkeep(accumulate=None)`, `with_breuse(accumulate=None)`, `rubin_helpers.make_blockscaled_trivial_tiled_mma(...)` | `TiledMma::with_bkeep(Option<bool>)` (B collector `FILL`) and `with_breuse(Option<bool>)` (`LASTUSE`) on any Rubin tiled MMA, the thread layout, permutations and fields copied (panics for a non-Rubin MMA); `utils::sm107::make_blockscaled_trivial_tiled_mma(...)`: FP4 pairs give `SM107MmaMXF4NVF4Op` (K 128), the 4/6/8-bit mix `SM107BlockScaledMmaMXF8F6F4Op` (K 64). The `USE` form is `op.with_collectors(a, b)` on the atom |
| `rubin_helpers.make_tensor_e_atom_layout`, `make_sparse_gmem_layout_e`, `make_sparse_smem_layout_e`, `make_sparse_tmem_layout_e` | `utils::sm107::` the same names: Rubin's FP4 metadata atom is `(128, 128)` (SM100's `(128, 256)`), the shared-memory layout tiles it over `(M, 384)` per stage (`tiled_mma` is unused, as in CuTe DSL), the TMEM layout is SM100's |
| `tcgen05.LdSPCompress32x32bOp(repeat, redOp)`, `LdRedSPCompress32x32bOp(repeat, redOp, nan)`, `cute.copy_atom_call(atom, src, [cdata, mdata, red])` | `tcgen05::LdSPCompress32x32bOp::new(Repetition::X128, TmemLoadRedOp::MaxAbs)` and `LdRedSPCompress32x32bOp::new(..).with_nan(true)` (into `CopyOp::TmemLoadSpCompress`), `make_copy_atom::<f32>(op)` and `copy_atom_call_sp(&atom, &tmem_view, &cdata, &mdata, red)` (`red` given exactly for the `.red` form; the destinations differ in type, so it is a function of its own beside `copy_atom_call`): `tcgen05.ld{.red}.spcompress.32x32b.x4` to `.x128` of Float32 values. Of every four values a thread loads it keeps the two the row op prefers (larger for `Max`, smaller for `Min`, by absolute value for the `Abs` forms; a NaN least preferred, with `nan` most; the lower position wins a tie), in order, into `cdata` (`x/2` Float32), and writes the 2:4 metadata into `mdata` (`max(x/32, 1)` Uint32: the 4-bit field of group `g` at bit `4 (g % 8)` of register `g / 8`, first kept position in bits 0-1, second in bits 2-3: the sparse MMAs'), and the `.red` form the reduction over all loaded values to `red`. `copy` refuses these atoms. sm_107a only, PTX 9.4 |

`tests/rubin_extras.rs` checks the helpers' choices on the host; runs spcompress in the simulator
for `.x4` to `.x128`, all four row ops, `nan` on and off, rows of NaNs and ties, against plain
Rust; compresses a dense Float32 tile in TMEM and multiplies it with the sparse MMA (E4M3, K = 128,
the metadata staged in TMEM straight from the registers) exactly equal to the pruned dense GEMM;
runs a GEMM tile of two Rubin MMAs from `make_trivial_tiled_mma` with `with_bkeep`/`with_breuse`
equal to the same MMAs without reuse; and checks the sm_107a target rule and the MLIR text (inline
PTX, verified by mlir-opt). **The spcompress instruction's PTX is cute-rs's reading**: CuTe DSL
4.8.1 gives only the operation's parameters, the instruction's qualifier spellings
(`.sp2of4`, `.idx4`, the order of `.abs` and `.NaN`) and its choice rule (the row op, ties, NaN)
are guesses that no assembler or hardware has checked, and `sm107_tmem_sf_frg` (the Rubin
scale-factor TMEM layout) is not ported (the tests place scale factors by hand).
