# Hopper (SM90)

## TMA descriptors, stmatrix and setmaxnreg

The TMA instructions that use
these descriptors came with m7-tma-load and later parts (below).

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cutlass.GridConstant[T]`, `cutlass.grid_constant` | `param_bytes(name, size, align, true)` (one `.param .align A .b8 k_param_i[N]`, `llvm.byval` and `nvvm.grid_constant` set on the lowered `llvm.func`); `StructParam::address()` is its generic address in place (`cvta.param.u64`, PTX 7.7). The simulator keeps it read-only |
| a TMA atom's `CUtensorMap` (`TmaTrait`), passed as a 128-byte `__grid_constant__` | `tma_descriptor(&tensor, &box_dims, stride_order, &TensorMapOptions)`: a hidden parameter appended after the kernel's own (launches pass the same arguments), described by a `TmaDescSpec` whose extents, byte strides and start are expressions over constants and scalar parameters. `KernelSignature::hidden_tensor_map` turns it into a `TmaDesc` for each launch; the CUDA launcher encodes that with `cuTensorMapEncodeTiled` from the tensor's device address, the simulator with its own 128-byte encoding (`TmaDesc::to_sim_bytes`: magic `CuTM`, version, fields) from its address map. `TmaDescriptor::get_ptr()` is the descriptor's generic address |
| `cutlass.experimental.cuda.create_tensor_map_tiled(global_address, dtype, global_dims, global_strides, box_dims, traversal_strides=, interleave=, swizzle=, l2_promotion=, oob_fill=, tma_format=)`, `create_tensor_map_tiled_from_view(tensor, box_dims, stride_order=, ...)`, `TensorMap` (`element_type`, `box_volume`, `shared_storage_dtype`, `shared_storage_bytes`, `global_tx_bytes`, `get_ptr`) | `create_tensor_map_tiled(address, dtype, &dims, &strides_16b, &box, &TensorMapOptions)` (strides of dimensions 1 and up in 16-byte units, TMA order), `create_tensor_map_tiled_from_view(&TensorMapView, &box, stride_order, &opts)`, `TensorMap` with the same queries; also `cute::cuda::tensor_map`. A kernel takes one with `param_tensormap(name)`, a launch with `KernelArg::tensor_map(&map)`; `TmaDescriptor::get_ptr()` is `get_ptr` |
| `TensorMapDataType`, `TensorMapDataFormat`, `TensorMapInterleave`, `TensorMapSwizzle`, `TensorMapL2Promotion`, `TensorMapFloatOOBFill` | the same names and values (the driver's codes). The FP4/FP6 encodings (`f416u4_align8b`, `B4X16`, ...) are supported; the `s128b_atom_*` swizzles exist and are refused until m10-umma-layouts |
| `cute.nvgpu.warp.StMatrix8x8x16bOp(transpose, num_matrices)` | `CopyOp::StMatrix { num, trans }`: `stmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16` (sm_90, PTX 7.8), thread-value layouts of CuTe C++'s `SM90_U32x{1,2,4}_STSM_N` / `SM90_U16x{2,4,8}_STSM_T`. Works with `make_tiled_copy_c_atom` and `make_tiled_copy_s`, as Hopper epilogues use it |
| `cute.nvgpu.warp.StMatrix16x8x8bOp(transpose=True, num_matrices)` | `CopyOp::StMatrix16x8x8b { num, trans: true }`: `stmatrix.sync.aligned.m16n8.x{1,2,4}.trans.shared.b8` (sm_100a, sm_101a/sm_110a, sm_120a; the f families from PTX 8.8), CuTe C++'s `SM100_U8x{4,8,16}_STSM_T`. `CopyOp::LdMatrix16x8x8b` loads it back |
| `cute.arch.setmaxregister_increase(n)`, `setmaxregister_decrease(n)`, `warpgroup_reg_alloc`, `warpgroup_reg_dealloc` (deprecated) | the same names (`#[deprecated]` for the last two): `setmaxnreg.{inc,dec}.sync.aligned.u32 n`, `n` a multiple of 8 from 24 to 256 checked when the kernel is built, sm_90a, sm_100a and sm_120a (the f families from PTX 8.8). The kernel must declare its block (`.reqntid`) or `max_threads` (`.maxntid`) |

* **Checks.** `TmaDesc::validate` (and `check_tensor_map`, which the build runs on what it
  knows) applies `cuTensorMapEncodeTiled`'s documented rules: rank 1 to 5 (3 to 5 with an
  interleave), a 16-byte-aligned address (32 with the 32-byte interleave), extents 1 to
  2^32, strides multiples of 16 bytes (32) below 2^40, box extents 1 to 256, an inner box of
  a multiple of 16 bytes no wider than the swizzle span, element strides 1 to 8, the 32-byte
  interleave only with the 32-byte swizzle, the NaN fill only for floating-point encodings.
  Static violations fail `Kernel::build` at the user's line; launch-time ones fail the
  launch (or `simulate`), naming the hidden parameter and its tensor.
* **The pure model** (`cute::tma`): `TmaDesc::box_elements` (the box's elements with
  their coordinates, shared-memory offsets and global addresses, `None` out of bounds,
  traversal strides counted), `TensorMapSwizzle::apply` (the 32B/64B/128B swizzles XOR the
  16-byte chunk index with address bits 7 and up of the absolute shared-memory address) and
  the simulator's encoding are what later TMA parts use. An interleaved box is traversed
  like a plain one.
* **On the SM89 GPU** (driver 595, CUDA 12.8; checked 2026-11-16): `cuTensorMapEncodeTiled`
  encodes nothing below compute capability 9.0 (`CUDA_ERROR_NOT_SUPPORTED` for every valid
  descriptor), but checks a few fields first: a zero extent, box or element stride and a
  rank outside 1 to 5 give `CUDA_ERROR_INVALID_VALUE`. `tests/gpu_tma.rs` holds the
  validator to that (and to every case on sm_90 or newer), and checks that a launch of a
  kernel with a tensor map on SM89 fails before the launch, naming the parameter. The
  by-value grid-constant ABI runs there: a kernel copies a 128-byte grid constant's bytes
  through its in-place address.
* **stmatrix** emission follows the LLVM version (`MlirOptions::llvm_major`, which
  `compile_with` reads from `mlir-opt --version`; `to_mlir` writes the newest syntax): LLVM
  22 and newer take `nvvm.stmatrix` with shape and element-type attributes and lower it to
  the convergent `llvm.nvvm.stmatrix.*` intrinsics; LLVM 21's `nvvm.stmatrix` (only
  `.m8n8.b16`) lowers to inline assembly that is not convergent. `.m16n8` needs LLVM 22, or
  inline PTX with `CompileOptions::mma_inline_ptx(true)`. The simulator places each
  register's elements by the PTX ISA's fragment tables (full warps only), with race and
  bank-conflict accounting.
* **setmaxnreg in the simulator**: every warpgroup starts at
  `sim::setmaxnreg_initial_budget(threads)` registers per thread (65536 / threads, at most
  255, rounded down to a multiple of 8: 168 for 384 threads) with an empty pool; a decrease
  returns registers to the block's pool, an increase waits until the pool has them. It
  reports a `setmaxnreg` by part of a warp or warpgroup, warps of a warpgroup disagreeing,
  a warp issuing the next one before synchronizing with the others since the previous one,
  an increase above (decrease below) the budget, and an increase that can never be served
  (a deadlock naming the warps).
* **Not yet**: TVM-FFI callables refuse kernels with hidden tensor maps; an AOT export
  lists a hidden tensor map as an ordinary 128-byte by-value parameter that the C caller
  encodes.

### TMA atoms

Building and partitioning TMA atoms; the next section issues the loads.

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cpasync.CopyBulkTensorTileG2SOp(cta_group)`, `CopyBulkTensorTileG2SMulticastOp(cta_group)`, `CopyBulkTensorTileS2GOp()`, `CopyReduceBulkTensorTileS2GOp(reduction_kind)`, `CtaGroup`, `ReductionKind` | the same names in `cute::cpasync` (`CtaGroup::{ONE, TWO}`, `ReductionKind::{ADD, MIN, MAX, INC, DEC, AND, OR, XOR}`), converting into `CopyOp::TmaLoad { multicast, cta_group }`, `CopyOp::TmaStore` and `CopyOp::TmaReduce(kind)`. `CtaGroup::TWO` is refused until m10-cta-pair |
| `cpasync.make_tiled_tma_atom(op, gmem_tensor, smem_layout, cta_tiler, num_multicast=1, *, gmem_coord_tensor=None, internal_type=None)` -> `TmaInfo` (`.atom`, `.tma_tensor`, `.smem_layout`; unpacks to `(atom, tma_tensor)`) | `cpasync::make_tiled_tma_atom(op, &gmem, smem_layout, cta_tiler, num_multicast)` -> `TmaInfo { atom, tma_tensor, smem_layout }` (`let (atom, t) = info.into()`); `make_tiled_tma_atom_with(.., &TmaAtomOptions)` takes the keyword arguments (`gmem_coord_tensor`, `internal_type`). C++ CuTe's algorithm (`make_tma_atom`): the box is the SMEM layout's largest contiguous run along unit steps of the global modes, merged up to 256 elements, other modes appended with extent 1, folded to at most 5 dimensions, split across the multicast; the swizzle is the layout's restated on bytes (`Swizzle<1|2|3,4,3>`). A staged layout (one more mode than the tiler) uses its first stage. The atom registers a hidden descriptor parameter (`atom.tma_info().unwrap().descriptor()`), one thread issues it (`thr_id` `1:0`) and it moves the whole box (`num_val_src`). The TMA tensor is a coordinate tensor over the global shape with scaled-basis strides (`(m,k):(1@1,1@0)` for a K-major tile) |
| `cpasync.tma_partition(atom, cta_coord, cta_layout, smem_tensor, gmem_tensor)` -> `(tXsX, tXgX)` | `cpasync::tma_partition(&atom, cta_coord, &cta_layout, &smem, &gmem)` -> `(tXsX, tXgX)`, mode 0 `(TMA, TMA_Iter)`, both views starting at the CTA's multicast share. The list form for gather4/scatter4 is `tma_partition_indexed`, and for the override copies `tma_partition_override` |

* **Checks.** A non-contiguous SMEM box, a swizzle TMA cannot express, a run longer than
  256 elements (C++ CuTe rejects it too), more than five dimensions that cannot fold, an
  SMEM layout whose majorness, rank or size does not match the tile, a box that cannot be
  split across the multicast (or whose per-CTA share would not start 128-byte aligned), a
  tile walking a broadcast (stride-0) mode, a global tensor that is not a kernel buffer
  parameter, and a `tma_partition` CTA layout whose cosize is not the atom's multicast (or
  a rank outside it, or a signed one) fail `Kernel::build` at the user's line, saying why.
* **Evidence** (`tests/tma_atom.rs`): for the K- and M-major GMMA tiles of f16, bf16 and f32
  (every atom, 128x64 and 64x64), staged, batched, multicast (2 and 4), merged and folded
  modes, dense_gemm.py's A/B loads and epilogue store and tma_v0.py, the descriptor model
  (`TmaDesc::box_elements` and the address-based swizzle, boxes 1024-byte aligned) places
  every global element of a tile where the CuTe layouts put it in shared memory. On the
  SM89 GPU the driver checks the 48 GMMA descriptors (`tests/gpu_tma.rs`).

### TMA loads, copy keyword arguments and atom fields

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cute.copy(tma_atom, tAgA[(None, k)], tAsA[(None, s)], tma_bar_ptr=mbar, cache_policy=p, tma_desc_ptr=d)` | `copy_with(&atom, &tAgA.slice((.., k)), &tAsA.slice((.., s)), CopyArgs::new().tma_bar_ptr(&mbar).cache_policy(p).tma_desc_ptr(d))`: one `cp.async.bulk.tensor.Nd.shared::cluster.global.tile.mbarrier::complete_tx::bytes[.L2::cache_hint]` per box (sm_90, PTX 8.0), as inline PTX (32-bit shared addresses, the descriptor's generic address, one register per coordinate, a memory clobber). The coordinates are the TMA tensor's element at each box's start, the destination the box's unswizzled start. As in CuTe DSL the copy elects one lane of each warp that executes it: call it from one warp, unelected; `mbarrier_init`, `mbarrier_init_fence` and `expect_tx` stay the program's |
| `cute.copy(atom, src, dst, *, pred=, unroll_factor=, **kwargs)` | `copy_with(&atom, &src, &dst, CopyArgs { .. })` (or `&CopyArgs`; `CopyOptions` is the same struct). `pred` takes one flag per source element or one per atom application (`(ATOM_REST, REST...)`). `unroll_factor` loops over the rest modes (which may then be runtime extents) with `LoopUnroll(count)` when both operands are memory or coordinates, and unrolls fully otherwise. `copy` accepts a coordinate tensor (`Tensor<Ident>`) as a TMA load's source |
| `cute.copy(atom, [src, aux...], [dst, aux...])` | `copy_aux(&atom, (&src, &[AuxOperand::from(&t)]), (&dst, &[]), args)`; the gather4/scatter4 atoms take the index tensor's partition and the override atoms the residue's. `tma_multicast=` (the block API) panics, pointing to m10-s2t-copies |
| `atom.with_(tma_bar_ptr=..., ...)`, `tiled_copy.with_(...)` | `atom.with_(CopyArgs::new().tma_bar_ptr(&mbar))`, `tiled_copy.with_(..)`: the fields every later copy with it uses; giving a bound field again panics |
| `mma.set(field, v)`, `mma.get(field)`, `mma.with_(...)`, `warpgroup.Field`, `tcgen05.Field`, `warp.Field` | `MmaAtom`/`TiledMma::set(MmaField::ACCUMULATE, v)`, `get(field) -> Boolean`, `with_(field, v)`; `MmaField::{ACCUMULATE, NEGATE_A, NEGATE_B, SFA, SFB, METADATA, DISABLE_OUTPUT_LANE}`. `TiledMma` and `MmaAtom` are `Carry`: a field set inside `for_range_carry` is carried, as CuTe DSL's K loops need. No `mma.sync` atom takes a field; WGMMA is the first |

* **Fields per atom.** A TMA load takes `tma_bar_ptr` (needed), `cache_policy`,
  `tma_desc_ptr` and `mcast_mask` (ignored, as CuTe DSL ignores it for the plain load);
  `CopyOp::G2R`/`R2G` take `cache_policy`; the TMA stores, multicast loads and bulk copies
  take theirs (next sections); `byte_mask` belongs to m7-tma-sm100-modes. Any other field, a
  missing `tma_bar_ptr`, a data tensor as a TMA load's source, a predicated TMA load, a
  CTA-pair load
 and a box start at a static offset that is not 128-byte aligned fail
  `Kernel::build`.
* **Simulator.** The load decodes the descriptor (its address resolved like any generic
  address), reads the box element by element (out-of-bounds elements zero-filled, NaN with
  `nan_request_zero_fma`, all counted in the transaction bytes), and places its 16-byte
  chunks at the destination's shared address through the descriptor's address-based
  swizzle; the bytes land as bulk copies do, counting down the mbarrier's tx-count.
  `SimReport::tma_instructions` and `tma_bytes` count them. It reports reads of the stage
  before the phase completes, a box larger than `expect_tx`, a destination not 128-byte
  aligned (the PTX ISA's rule) or, with a swizzle, not at a multiple of the swizzle's
  pattern (256, 512 or 1024 bytes: the hardware swizzles by address, so the box would not
  land where the CuTe layout's offset-based swizzle puts it), a box past its buffer, generic
  writes the load overwrites without `fence.proxy.async`, an mbarrier never initialised or
  whose initialisation was not published with `fence.mbarrier_init`, and a copy issued
  twice (two warps executing it).
* **Shared-memory alignment.** `make_smem_tensor` keeps its 128 bytes; use
  `make_smem_tensor_aligned(layout, 1024)` for 128-byte-swizzled TMA destinations. Above
  48 KB the buffers live in the dynamic window, whose `.extern .shared .align` is the
  largest buffer alignment, so the 1024 bytes hold there too.
* **Evidence** (`tests/tma_load.rs`, feature kernels in `tests/feature_kernels/tma_load.rs`):
  single tiles through every K-major atom and the M-major 128-byte one, a K loop through a
  two-stage ring with phase-tracked mbarriers and a checksum, edge tiles, a batched
  `(M, K, L)` tensor, runtime extents and stride, `unroll_factor` over four stages in a
  64 KB kernel, each matched against Rust under the built-in and seeded schedules; the
  misuse reports; the MLIR, compiled to PTX for sm_90a by every installed `mlir-opt`
  (LLVM 21, 22, 23) and assembled by `ptxas -arch=sm_90a` (`tests/ptxas.rs`). Nothing here
  runs on the SM89 GPU.

### TMA stores, reductions, L2 cache hints and prefetches

| CuTe DSL | cute-rs |
|---|---|
| `cpasync.CopyBulkTensorTileS2GOp()`, `cute.copy(tma_atom_c, bSG_sD[(None, stage)], bSG_gD[(None, tile)], tma_desc_ptr=, cache_policy=)` | `copy_with(&atom, &tSs.slice((.., st)), &tGg.slice((.., j)), args)` (source the SMEM partition, destination the TMA tensor's): `cp.async.bulk.tensor.Nd.global.shared::cta.tile.bulk_group[.L2::cache_hint]` per box (sm_90, PTX 8.0), inline PTX, elected as the load is; complete it with `cp_async_bulk_commit_group` / `cp_async_bulk_wait_group(n, read)` |
| `cpasync.CopyReduceBulkTensorTileS2GOp(reduction_kind=cute.ReductionKind.ADD\|MIN\|MAX\|INC\|DEC\|AND\|OR\|XOR)` | `CopyReduceBulkTensorTileS2GOp { reduction_kind: ReductionKind::ADD }`: `cp.reduce.async.bulk.tensor.Nd.global.shared::cta.{op}.tile.bulk_group`. The element type is checked when the kernel is built against the PTX ISA's table: `add` on f16, bf16 (`.noftz`), f32, u32, s32, u64; `min`/`max` on f16, bf16, u32, s32, u64, s64; `inc`/`dec` on u32; `and`/`or`/`xor` on 32- and 64-bit integers. (CuTe DSL's deprecated `ReductionOp` spelling has no counterpart: the enum is the only form) |
| `cache_policy=Int64` on TMA loads, stores and reductions, `CopyBulkG2SOp`, `CopyBulkS2GOp`, `CopyG2ROp`/`CopyR2GOp`; C++'s `TMA::CacheHintSm90` | `CopyArgs::cache_policy(p)` (or bound with `with_`); `cpasync::CacheHintSm90::{EVICT_NORMAL, EVICT_FIRST, EVICT_LAST}` are the createpolicy values C++ CuTe passes (CuTe DSL has no `createpolicy` wrapper). The G2R/R2G forms are `ld`/`st.global.L2::cache_hint` (sm_80, PTX 7.4) and run on the SM89 GPU |
| `cpasync.prefetch_descriptor(tma_atom)`, `cute.arch.prefetch(addr, tensormap=True)` | `cpasync::prefetch_descriptor(&atom)`, `prefetch(&ptr, PrefetchOptions { tensormap: true, .. })`, `prefetch_tensormap(addr)`: `prefetch.tensormap [a]` on the generic address (sm_90), issued by every thread that executes it |
| `cute.prefetch(atom, src)` | `prefetch_atom(&atom, &src)` (`prefetch` is `cute.arch.prefetch`): a TMA load atom with its TMA-tensor partition (every box of every rest mode, `cp.async.bulk.prefetch.tensor.Nd.L2.global.tile`, elected), or a bulk-copy atom with the global source (`cp.async.bulk.prefetch.L2.global [a], size`, not elected). An im2col atom prefetches `.im2col` boxes; a gather4 atom's rows go through `prefetch_atom_aux(&atom, &tAgA, &tAgI)` (`.tile::gather4`, sm_100a); other atoms are refused |

* **Simulator.** A TMA store reads its box through the descriptor's address-based swizzle
  (the inverse of a load's placement) and writes only the box's elements inside the tensor.
  The shared memory is read in the background until the covering
  `cp_async_bulk_wait_group(n, true)` (a write to it before then is a race), the global
  writes land at the covering plain wait or the CTA's exit (a read of them before then is
  reported), and generic writes it reads need `fence_view_async_shared` first. A reduction
  lands as element-wise read-modify-writes: float adds round to nearest even and keep
  subnormals (f16/bf16 are `.noftz`, and the PTX ISA gives `.f32` the same default),
  `inc`/`dec` wrap as the atomics do. Prefetches check their descriptor, coordinates or range and are counted.
  `SimReport::tma_stores`, `tma_store_bytes`, `prefetches`, `tensormap_prefetches` and
  `cache_hints` count them.
* **Evidence** (`tests/tma_store.rs`, `tests/feature_kernels/tma_store.rs`): an epilogue
  through a two-stage buffer (plain stores, proxy fence, named barrier, one warp storing with
  a cache hint and waiting `.read` before each reuse) over a `(128, 256)` tensor and a
  clipped `(100, 200)` one; reductions from the four CTAs of a cluster (f32, f16 and u32
  adds, min, max, inc, dec, xor) after a TMA store of the initial tile; a split-K add from
  CTAs outside a cluster; a K loop prefetched by one `prefetch_atom` with a descriptor
  prefetch; the bulk atoms with hints and a bulk prefetch; the misuse reports; the MLIR
  compiled to PTX by LLVM 21 and 23. `copy_atom_cache_hints` runs on the SM89 GPU.

### Multicast masks, multicast and peer-CTA copies, st.async

| CuTe DSL | cute-rs |
|---|---|
| `cute.make_layout_image_mask(lay, coord, mode)` | `make_layout_image_mask(&lay, coord, mode) -> Int16`: the layout sliced at `coord` with `mode` kept; static layouts (cosize at most 16) give the bits when the kernel is built and shift them by the slice's (possibly runtime) offset, runtime strides set each bit at run time |
| `cpasync.create_tma_multicast_mask(cta_layout_vmnk, cta_coord_vmnk, mcast_mode)` | `cpasync::create_tma_multicast_mask(&vmnk, coord, mode)` (both rank 4) |
| `cpasync.CopyBulkTensorTileG2SMulticastOp()`, `make_tiled_tma_atom(.., num_multicast=k)`, `cute.copy(.., tma_bar_ptr=, mcast_mask=)` | the same with `CopyArgs::new().tma_bar_ptr(&mbar).mcast_mask(mask)` (needed): `...tile.mbarrier::complete_tx::bytes.multicast::cluster [smem], [tmap, coords], [mbar], mask` with an `h` operand. Each CTA loads its share (`tma_partition` with its rank among the `k`), which lands at the same offset in every CTA of the mask and counts down each one's mbarrier, so each receiving CTA expects the whole tile |
| `cpasync.CopyBulkG2SOp()`, `CopyBulkG2SMulticastOp()`, `cute.copy(atom, g, s, mbar_ptr=, mcast_mask=, cache_policy=)` | `make_copy_atom::<T>(cpasync::CopyBulkG2SOp)` / `CopyBulkG2SMulticastOp` (`CopyOp::BulkG2S { bits, multicast }`): `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes[.multicast::cluster][.L2::cache_hint]`, not elected |
| `cpasync.CopyBulkS2SOp()`, `cute.copy(atom, s, d, mbar_ptr=, cta_rank=)` | `CopyOp::BulkS2S { bits }`: `mapa` on the destination and mbarrier, then `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes`, not elected |
| `cute.arch.store_async_dsmem(smem_ptr, value, mbar_ptr, peer_cta_rank)` | `store_async_dsmem(&ptr, [v; 1\|2\|4], &mbar, peer)` (Int32, Uint32 or Float32, bit for bit): `mapa` on both pointers, `st.async.shared::cluster.mbarrier::complete_tx::bytes{.v2,.v4}.b32` (sm_90, PTX 8.1, which it raises the module to); the pointer aligned to the store's bytes |

* **Simulator.** Vector clocks already span the cluster. A multicast copy is one copy per
  destination CTA, written by that CTA's copy agent and landed by its mbarrier phase, so
  only its waiters see the data; a peer copy and `st.async` are copies aimed at the peer the
  same way (`st.async` is a generic-proxy write, needing no proxy fence). A peer copy reads
  its source in the background until it lands, and records that read as the destination's
  agent's, so the sender may overwrite it only once the peer's completion is ordered before
  (typically the peer's arrival on a sender's mbarrier). `SimReport::cluster_copies` counts
  copies into another CTA. Reported: mask bits beyond the cluster, a zero mask, a peer rank
  outside the cluster, a peer mbarrier used before its initialisation was published
  (`fence.mbarrier_init` and a cluster barrier), a phase whose `expect_tx` does not cover
  every multicast share, reads before the local wait, a CTA that exits while copies still
  read or write its shared memory or after which a peer signals it, and any of these copies
  in a launch without clusters.
* **Evidence** (`tests/cluster_copies.rs`, `tests/feature_kernels/cluster_copies.rs`): masks
  over every CTA of 2x1, 2x2 and 4x2 clusters with runtime and static coordinates; GEMM
  loads in 2x1 and 2x2 clusters with A multicast along N and B along M (checksums); a bulk
  G2S multicast; an S2S ring; a cluster reduction with `st.async` modelled on CuTe DSL's
  `cluster_reduce`; under the built-in and seeded schedules; the misuse reports; the MLIR
  compiled to PTX for sm_90 and sm_90a by LLVM 21 and 23.

### In-kernel tensor-map updates

Descriptors in memory that kernels modify, for grouped and pointer-array GEMMs whose groups
each need their own tensor map.

| CuTe DSL | cute-rs |
|---|---|
| `cpasync.copy_tensormap(tma_atom, tensormap_ptr)` | `cpasync::copy_tensormap(&atom, &ptr)`: sixteen 64-bit loads of the atom's grid-constant descriptor and stores to `ptr` (global, shared or generic, any element type), by every thread that executes it |
| `cpasync.update_tma_descriptor(tma_atom, gmem_tensor, tma_desc_ptr)` | `cpasync::update_tma_descriptor(&atom, &tensor, &ptr)`: `tensormap.replace.tile.{global_address,global_dim,global_stride}{.global\|.shared::cta}.b1024.{b64\|b32}` for the address, each TMA dimension's extent and dimensions 1 and up's byte strides, from the tensor's modes in the order the atom's tensor had them (runtime pointers, extents and strides allowed; folded dimensions refused). sm_90a, sm_100a, sm_110a, sm_120a (the f families from PTX 8.8), PTX 8.3 |
| the experimental `tensormap_replace(field, ptr, new_value=, ord=)` | `cpasync::tensormap_replace(&ptr, TensormapField::{GlobalAddress, GlobalDim(ord), GlobalStride(ord)}, value)`. A stride is in bytes, ordinal 0 being dimension 1's (C++ CuTe's numbering; `ptxas` from CUDA 12.5 takes bytes, older ones took 16-byte units). The box, element strides, rank, element type and swizzle fields are not exposed |
| `cpasync.fence_tma_desc_acquire(ptr)`, `fence_tma_desc_release()`, `cp_fence_tma_desc_release(gptr, sptr)` | the same names: `fence.proxy.tensormap::generic.acquire.gpu [p], 128`, `fence.proxy.tensormap::generic.release.gpu` and `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [g], [s], 128` (sm_90, PTX 8.3) |
| `cute.copy(..., tma_desc_ptr=ptr)` | `CopyArgs::tma_desc_ptr(ptr.toint_i64())`: the copy reads the descriptor at that generic address |
| `cutlass.utils.TensorMapManager(tensormap_update_mode, bytes_per_tensormap)`, `TensorMapUpdateMode.{GMEM, SMEM}` (`cutlass/tensor_utils/tensormap.py`) | `cute::utils::{TensorMapManager, TensorMapUpdateMode}`: `get_tensormap_ptr(&ptr, AddressSpace::Gmem)` (an aligned raw pointer), `init_tensormap_from_atom(&atom, &dst, warp_id)` (that warp's elected lane copies, then `sync_warp`), `fence_tensormap_initialization()` (`fence.acq_rel.cta` in GMEM mode), `fence_tensormap_update(&ptr)` (the acquire) and `update_tensormap(&[&tensors], &[&atoms], &[gmem_ptrs], warp_id, &[smem_ptrs])`: SMEM mode edits the shared copies, waits for the bulk groups (commit, `wait_group.read 0`) and publishes with `tensormap.cp_fenceproxy`; GMEM mode edits in place, then `sync_warp` and the release fence. The tensors and atoms share one element type. The workspace is a kernel buffer of `descriptors x 128` bytes, 128-byte aligned (`param_tensor_aligned::<u64>(.., 128)`), or shared memory |

* **Inline PTX.** LLVM 21 has no `nvvm.tensormap.replace` (LLVM 23 has) and no LLVM has
  `tensormap.cp_fenceproxy`, so all of these are inline assembly, as CuTe DSL emits them.
  `tensormap.cp_fenceproxy` is warp-collective (`.sync.aligned`) and LLVM does not treat
  inline assembly as convergent: call it from a converged warp, outside code whose branches
  LLVM could duplicate it into.
* **Simulator.** A descriptor in memory holds the simulator's 128-byte encoding, so a copy,
  a replace and a TMA read round-trip; `tensormap.replace` decodes the object, changes the
  field and writes it back. The tensormap proxy is a third proxy beside the generic and
  async ones: each generic write to a descriptor's bytes in global memory (stores, atomics,
  replaces, `cp_fenceproxy`'s copy) is recorded per warp, and a TMA load or store through
  the descriptor needs, for each, a release by the writing warp after it and an acquire of
  the address by the issuing thread that the release happens before (in that warp's order
  or through the cluster's vector clocks; for writes by another cluster, a release and an
  acquire). It reports a missing release, a missing acquire, an acquire not ordered after
  the release, descriptor bytes that do not decode (uninitialised memory, a replace before
  the copy), a descriptor written in the kernel that `cuTensorMapEncodeTiled` would refuse,
  a divergent `cp_fenceproxy`, a generic write to a descriptor that a TMA store still reads
  (before its bulk group's `wait_group.read`), and `tensormap.replace` compiled for plain
  sm_90. Prefetches are not checked, nor the acquire of descriptors written before the
  launch; a release by another warp than the writer's is reported (conservatively).
* **Evidence** (`tests/tensormap.rs`, `tests/feature_kernels/tensormap.rs`,
  `tests/diagnostics.rs`): after CuTe DSL's `tma_tensormap_replace.py`, a descriptor copied
  to a global workspace, pointed at a later row of a second tensor and shrunk to fewer rows
  (zero-filled past them), then TMA-loaded through; a grouped copy where each CTA rewrites
  its load and store descriptors with `TensorMapManager` in both modes for four groups of
  different shapes (partial tiles) and TMA-loads and stores every tile; all under the
  built-in and seeded schedules; the misuse reports; PTX for sm_90a from LLVM 21 and 23,
  assembled by `ptxas`. Nothing here has run on an SM90 GPU.

### Im2col and the SM100 TMA modes

| CuTe DSL | cute-rs |
|---|---|
| `cpasync.CopyBulkTensorIm2ColG2SOp(cta_group)`, `CopyBulkTensorIm2ColG2SMulticastOp(cta_group)`, `CopyBulkTensorIm2ColS2GOp()` | the same structs (`CopyOp::TmaLoadMode`/`TmaStoreMode` with `TmaOpMode::Im2col`): `cp.async.bulk.tensor.{3,4,5}d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [smem], [tmap, {c, w, (h, (d,)) n}], [mbar], {offsets}` and `.global.shared::cta.im2col_no_offs.bulk_group` (sm_90, PTX 8.0), inline PTX |
| `cpasync.make_im2col_tma_atom(op, gmem_tensor, smem_layout, cta_tiler, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt, num_multicast, *, internal_type)` | `cpasync::make_im2col_tma_atom(op, &gmem, smem_layout, cta_tiler, &Im2colOptions)`; `Im2colOptions::conv(filter_srt, lower_padding, upper_padding, stride, dilation)` derives the corners (`-lower_padding`, `upper_padding - (filter - 1) * dilation`) and offsets. The tensor is `((w, (h, (d,)) n), c)`, channels contiguous; the SMEM tile `(pixels, channels)` K-major. C++ CuTe's `make_tma_atom_im2col`: channels per pixel = the SMEM layout's contiguous run (at most 256), pixels per column its next run (split among a multicast). The TMA tensor has the GEMM shape `(Q*P*Z*N, (C, S, R, T))`, M linear (C++ composes its multi-mode M with a linear layout the same way); a copy turns each box's `(m, c, s, r, t)` into `w = lower_w + q * stride_w` and offsets `lower_srt + s * stride_srt` |
| `nvgpu.make_im2col_tma_atom_A(...)` | `nvgpu::make_im2col_tma_atom_a` (see the MMA-aware TMA atoms) |
| `cute.prefetch(atom, src)` with an im2col atom | `prefetch_atom`: `cp.async.bulk.prefetch.tensor.Nd.L2.global.im2col` |
| `cpasync.CopyBulkTensor2DGather4G2SOp(cta_group)`, `...MulticastOp`, `CopyBulkTensor2DScatter4S2GOp()` | the same structs; atoms from `make_tiled_tma_atom_with(op, &data, smem, tiler, 1, &TmaAtomOptions::new().gmem_coord_tensor(&gI))`, `gI` an `Int32` tensor of the data's shape, broadcast (stride 0) along the contiguous mode, its gather mode at least 4. `tma_partition_indexed(&atom, coord, &cta_layout, &sA, &gA, &gI)` -> `(tAsA, tAgA, tAgI)`; `copy_aux(&atom, (&tAgA, &[(&tAgI).into()]), (&tAsA, &[]), args)` loads (scatter: the index operand on the destination). Each instruction moves four rows of `box_dims[0]` elements (`box_dims[1]` is 1): `cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4` (sm_100a, the sm_100f family from PTX 8.8) and `.global.shared::cta.tile::scatter4.bulk_group` (sm_100a, sm_110a and their families); the elected lane reads the four indices. The multicast form writes each box to the mask's CTAs whole (`num_multicast` 1) |
| `cpasync.CopyBulkS2GByteMaskOp()` with `cute.copy(..., byte_mask=)` | `CopyBulkS2GByteMaskOp` with `CopyArgs::byte_mask(mask)`: `cp.async.bulk.global.shared::cta.bulk_group.cp_mask [dst], [src], size, mask` (sm_100, PTX 8.6); bit `i` writes byte `i` of each 16-byte chunk |
| `cpasync.CopyBulkTensorTileG2SOverrideOp(cta_group)`, `CopyBulkTensorTileG2SMulticastOverrideOp(cta_group, no_fully_oob_tile)`, `CopyBulkTensorTileS2GOverrideOp(no_fully_oob_tile)` | the same structs, atoms from `make_tiled_tma_atom` (CuTe DSL's public maker refuses them); `tma_partition_override(&atom, coord, &cta_layout, &sA, &gA, &residue)` with a residue coordinate tensor (origin the extents, strides negated) and `copy_aux(&atom, (&tAgA, &[(&tAr).into()]), ...)`. Each box gives `.override::global_address.override::global_dim_stride` its address (`ptrtoint`), the remaining extents clamped to the box (8-bit, so box dimensions are at most 255) and the strides (`(lower + upper << 32) << 4`) at coordinates 0, per the PTX ISA. sm_107f only (PTX 9.4): no toolchain compiles it, so it is simulated and checked as MLIR text; `no_fully_oob_tile` is recorded only. With a multicast the mask is `.multicast::cluster::32b` |
| `internal_type=` (`make_tiled_tma_atom`, `make_im2col_tma_atom`) | `TmaAtomOptions::internal_type`, `Im2colOptions::internal_type`: C++ CuTe's `TMA::to_CUtensorMapDataType` and CuTe DSL's `use_unpack`. FP4 defaults to the packed `f416u4_align8b` (`.b4x16`, an `F4E2M1FN` SMEM tensor), FP6 to `f416u6_align16b`; an 8-bit internal type for FP4/FP6 unpacks into 16-byte slots (`f416u4_align16b` `.b4x16_p64`, `f416u6_align16b` `.b6x16_p32`), loaded into a `u8` SMEM tensor of one element per value; the padding is left unwritten. Other internal types of the element's width move as theirs. The transaction counts the packed global bytes (`global_tx_bytes`), the SMEM the padded ones (`shared_storage_bytes`) |
| `TensorMapDataFormat.B4X16`, `B4X16_P64`, `B6X16_P32` | supported for FP4 and FP6 tensors (`create_tensor_map_tiled`) |

* **Descriptors**: `TmaDesc::im2col` (`Im2colParams`: corners, channels per pixel, pixels
  per column) makes an im2col descriptor, checked as `cuTensorMapEncodeIm2col` documents
  (`check_im2col`: rank 3 to 5, corners of 16/8/5 bits by rank, a non-empty bounding box,
  1 to 256 channels whose bytes are a multiple of 16 within the swizzle, 1 to 1024 pixels;
  cute-rs takes no interleave and no sub-byte encoding there) and encoded with it by the
  launcher. The sub-byte encodings follow the driver's and the PTX ISA's rules (16-value
  groups; the padded ones a 128-value inner box, inner extent a multiple of 128, 32-byte
  address and strides, no swizzle or 128B, no NaN fill; packed FP4 an even extent), and a
  kernel with one needs sm_100 to sm_11x (`check_target`).
* **Simulator**: `TmaDesc::access_elements` models each access mode from the PTX ISA alone:
  im2col walks `pixels_per_column` pixels through the bounding box (`lower_corner` to
  `extent - 1 + upper_corner`, W fastest, by each traversal stride, wrapping to the next
  row, plane and image), reads each at its position plus the offsets, and fills padding
  and images past N; gather4/scatter4 four rows; the sub-byte encodings bit by bit into
  packed or padded slots. It reports a filter base outside the bounding box, offsets out
  of range, a tile access with an im2col descriptor (and the reverse), an attribute
  override with non-zero coordinates, an unaligned overriding address, negative store
  coordinates and an im2col store box reaching outside the tensor.
* **Verified**: the im2col tiles of a 2-D (3x3, padding 1), a strided and dilated 2-D and a
  3-D convolution against a plain Rust im2col, and an im2col store, in the simulator;
  gather4 (rows past the tensor zero-filled), scatter4, byte masks, packed and padded
  FP4/FP6 images against a hand-written packer, and override loads and stores over ragged
  edges; `ptxas` 12.8 assembles the im2col kernels for sm_90, sm_90a and sm_100a and the
  SM100 ones for sm_100a; `mlir-opt` 21, 22 and 23 compile them. On the SM89 host,
  `cuTensorMapEncodeIm2col` and `cuTensorMapEncodeTiled` with the 16U4/16U6 data types
  return `CUDA_ERROR_NOT_SUPPORTED` (as every tensor map there) for each descriptor the
  validator accepts, and the driver's invalid-value rejections (0 channels, 0 pixels,
  rank 6) are the validator's too: the SM89 driver (595) does not reject the 16U4/16U6
  types as invalid values, but whether it encodes them needs an SM90 or SM100 GPU.

### The TMA APIs at a glance (m7)

| CuTe DSL | cute-rs |
|---|---|
| `cpasync.make_tiled_tma_atom`, `TmaInfo` | `cpasync::make_tiled_tma_atom` -> `TmaInfo { atom, tma_tensor, smem_layout }` (TMA atoms, above) |
| `cpasync.tma_partition` | `cpasync::tma_partition` |
| `CopyBulkTensorTileG2SOp`, `CopyBulkTensorTileG2SMulticastOp`, `CopyBulkTensorTileS2GOp`, `CopyReduceBulkTensorTileS2GOp`, `CopyBulkG2SOp`, `CopyBulkG2SMulticastOp`, `CopyBulkS2GOp`, `CopyBulkS2SOp` | the same names in `cute::cpasync` |
| `cute.copy(...)` keywords (`tma_bar_ptr`, `mcast_mask`, `cache_policy`, `tma_desc_ptr`, `mbar_ptr`, `cta_rank`), `atom.with_(...)` | `CopyArgs` builders and `CopyAtom::with_` / `TiledCopy::with_` |
| `cpasync.create_tma_multicast_mask`, `cute.make_layout_image_mask` | `cpasync::create_tma_multicast_mask`, `make_layout_image_mask` |
| `cpasync.prefetch_descriptor`, `cute.prefetch` | `cpasync::prefetch_descriptor`, `prefetch_atom` |
| `cute.arch.fence_proxy`, `fence_view_async_shared`, `cp_async_bulk_commit_group`, `cp_async_bulk_wait_group`, `store_async_dsmem` | the same names (`fence_proxy_with` for the keyword arguments) |
| `cpasync.copy_tensormap`, `update_tma_descriptor`, `fence_tma_desc_acquire`, `fence_tma_desc_release`, `cp_fence_tma_desc_release`, `TensorMapManager` | the same names (in-kernel tensor-map updates, above) |
| `cutlass.experimental.cuda.create_tensor_map_tiled`, `create_tensor_map_tiled_from_view` | the same names in `cute::tensor_map` (TMA descriptors, above) |

The TMA tutorials `tma_v0.py` and `tma_v1.py` are ported as the `tma_copy` and
`tma_transpose` examples (see Examples). Porting them found no missing API; what differs
is listed in the examples' headers (`@cute.struct` storage as `alloc_smem` and
`make_smem_tensor_aligned`, `sm100_utils.make_smem_layout` written out from
`utils::sm90::get_smem_layout_atom`, and an explicit wait for the TMA store's bulk group).

## Warpgroup MMAs

The sm_90a warpgroup MMA from shared-memory descriptors. There
is no SM90 GPU to run it on: it is held to a plain Rust GEMM in the simulator, whose model
of the instruction comes from the PTX ISA (descriptor fields, canonical layouts, swizzle by
address, D-fragment layout) rather than from cute-rs's layout algebra, its MLIR is checked
with `mlir-opt` 21 to 23 and its PTX assembled with `ptxas -arch=sm_90a`; the descriptors
themselves are built and checked on a GPU (`tests/gpu_wgmma.rs`).

| CuTe DSL (Python) | cute-rs |
|---|---|
| `warpgroup.MmaF16BF16Op(ab_dtype, acc_dtype, instruction_shape, a_src, a_major_mode, b_major_mode)` | `warpgroup::MmaF16BF16Op::new(ab, acc, (64, n), OperandSource::SMEM, a_major, b_major)` (`(64, n, 16)` too): `wgmma.mma_async.sync.aligned.m64n{N}k16.{f32,f16}.{f16,bf16}.{f16,bf16}`, N a multiple of 8 from 8 to 256, Float16 accumulating in Float16 or Float32 and BFloat16 in Float32, either operand K- or MN-major (the transpose immediates). The atom spans 128 threads (`ABLayout`, `CLayout_64xN`) and takes `MmaField::ACCUMULATE` (scale-d; false until set). `OperandSource::RMEM` (A in registers): see m8-wgmma-variants below. Kernels that use it compile for sm_90a only |
| `tiled_mma.make_fragment_A(tCsA)`, `make_fragment_B`, `make_fragment_C` | `tiled_mma.make_fragment_a(&tCsA)`, `make_fragment_b`, `make_fragment_c::<T>(&tCgC or layout or shape)`: for WGMMA, A and B become tensors of 64-bit shared-memory matrix descriptors, one per MMA tile (`(1, MMA_M, MMA_K, stages...)`, strides in 16-byte units, `Tensor::smem_desc(coord)` reads one); otherwise register fragments. The tile must be a canonical GMMA layout (from `warpgroup::make_smem_layout_atom` atoms; the swizzle may sit in the layout or the pointer) in a buffer aligned to the swizzle's repeat (`make_smem_tensor_aligned(.., 1024)` for 128-byte swizzles). `ThrMma::partition_fragment_a/b/c` give the same |
| `cute.gemm(mma, d, a, b, c)` on WGMMA fragments | `gemm(&mma, &acc, &frag_a.slice((.., .., k)), &frag_b.slice((.., .., k)), &acc)`: one MMA per `(MMA_M, MMA_N, MMA_K)` tile, each with the atom's current `ACCUMULATE` |
| `warpgroup.fence()`, `commit_group()`, `wait_group(n)` | `warpgroup::fence()`, `commit_group()`, `wait_group(n)`: `nvvm.wgmma.fence.aligned`, `nvvm.wgmma.commit.group.sync.aligned`, `nvvm.wgmma.wait.group.sync.aligned n` |

* **Accumulators are SSA values.** The MMA is inline assembly with its accumulators tied
  to its outputs (LLVM has no intrinsic for it); its results become valid only at the wait
  that covers it. `fence()` and `wait_group(n)` pass every accumulator fragment of the
  current region (made by `make_fragment_c`/`partition_fragment_c`, or used by a WGMMA
  `gemm`) through an operand fence (CUTLASS's `warpgroup_fence_operand`: an empty statement
  reading and writing them) and rebind the fragment to its outputs, so that LLVM cannot
  move register writes past the fence or reads before the wait. `fence()` also zeroes the
  registers not yet written. Of a fragment created outside an open `if_` or loop, only
  the registers written inside it (by an MMA there, say) are rebound, and none zeroed: fill
  it before the `if_`, or carry it through a loop (`for_range_carry`), as the dense GEMM's
  main loop does.
* **Build-time check.** An MMA's results (and what they flow into through fences, waits
  that do not complete them, loop carries and `if_` results) may only be the accumulators
  of another MMA, fence or wait inputs, or loop and region carries; any other read before
  the completing wait, and accumulators left in flight (never waited for), fail
  `Kernel::build`. Which groups a `wait_group(n)` completes is tracked in build order
  (results from an earlier loop iteration count as in flight when `n > 0`).
* **Simulator.** The four warps of a warpgroup meet at each MMA (a warp that never comes is
  reported as a hang); its descriptors and scale-d must agree in all 128 threads. The
  operands are read through the decoded descriptors at issue (written, ordered before the
  issue in every warp, generic writes followed by `fence_view_async_shared`, no copy still
  writing them), and stay read until the wait: overwriting them is reported. Each
  accumulator starts from C (or 0) and adds its 16 products in order of K in Float32, then
  rounds once to the accumulator type; the hardware's order is undocumented, so only exact
  sums (small integers) are known to match it. The results land at the warp's covering
  `wgmma.wait_group`; reading them earlier, an MMA whose accumulators another instruction
  accessed since the warp's last `wgmma.fence` (the PTX ISA's rule; chained MMAs of the
  same shape need none) and an MMA without any fence are reported. A descriptor with a
  non-zero base offset is refused (cute-rs builds none).

## WGMMA variants and distributed shared memory

Neither runs on the SM89 GPU: they are held
to plain Rust in the simulator (whose model comes from the PTX ISA), their MLIR is checked
with `mlir-opt` and their PTX assembled with `ptxas` for sm_90a and sm_90.

| CuTe DSL (Python) | cute-rs |
|---|---|
| `warpgroup.MmaF8Op(a_dtype, b_dtype, acc_dtype, instruction_shape, a_src, a_major_mode, b_major_mode)` | `warpgroup::MmaF8Op::new(a, b, acc, (64, n), src, K, K)`: `wgmma...m64n{N}k32.{f32,f16}.{e4m3,e5m2}.{e4m3,e5m2}`, A and B each Float8E4M3FN or Float8E5M2, K-major only (MN-major is refused) |
| `warpgroup.MmaI8Op(...)` | `warpgroup::MmaI8Op::new(a, b, Int32, (64, n), src, K, K)`: `wgmma...m64n{N}k32.s32.{s8,u8}.{s8,u8}`, N 8, 24 or a multiple of 16; mixed signedness needs PTX 8.4 (the kernel's PTX version is raised); sums wrap (no `.satfinite`) |
| `OperandSource.RMEM` | `OperandSource::RMEM` on any of the three: A comes from a register fragment (`ALayout_64x16` / `ALayout_64x32`, four stacked `mma.sync` A fragments) that `make_fragment_a`/`partition_fragment_a` make and the kernel fills (`ldmatrix` through `make_tiled_copy_a`, or a converted accumulator) before `warpgroup::fence()`; A must be K-major. The fence passes the A registers through its operand fence, and `wait_group` keeps the A registers of the MMAs it completes live until then, so LLVM cannot reuse them while the tensor cores read them (an MMA issued inside a loop body and completed after it is not covered: a follow-up) |
| `cutlass.utils.sm90.make_trivial_tiled_mma(...)` (`sm90_make_trivial_tiled_mma`) | `cute::utils::sm90::make_trivial_tiled_mma(a, b, a_major, b_major, acc, atom_layout_mnk, tiler_mn, a_source)` (and `sm90_make_trivial_tiled_mma`): F16/BF16 pairs, FP8 pairs and 8-bit integer pairs; anything else panics (`TypeError`) |
| `cute.arch.map_dsmem_ptr(smem_ptr, cta_rank)` | `map_dsmem_ptr(&ptr, rank)`: a pointer into CTA `rank`'s shared memory (memspace `Dsmem`), keeping type, alignment and swizzle. Tensors made from it are read and written with `ld`/`st` on `.shared::cluster` after `mapa`; `cute::load`/`store` (`ss` `None` or `Cluster`), the atomics and `red`, mbarrier arrivals on a mapped mbarrier, `CopyOp::Universal`, `S2R`/`R2S` and `DsmemStore` copies take them; other instructions refuse them |
| `CopyS2ROp()`, `CopyR2SOp()` with `memory_order`, `memory_scope`, `shared_space` | `CopyOp::S2R { bits, attrs: S2RAttrs { .. } }`, `CopyOp::R2S { bits, attrs: R2SAttrs { .. } }` (`bits` 0: the widest vector) |
| `cpasync.CopyDsmemStoreOp()`, `cute.copy(atom, src, dst, mbar_ptr=)` | `CopyOp::DsmemStore { bits }` (0, 32, 64, 128), `copy_with(&atom, &regs, &dsmem_dst, CopyArgs::new().mbar_ptr(&map_dsmem_ptr(&mbar, rank)))`: `st.async...mbarrier::complete_tx::bytes{.v2,.v4}.b32` |

* **FP8 accumulation.** The simulator adds each MMA's 32 exact products to C in Float32 in
  order of K and rounds once; Hopper accumulates FP8 with less internal precision, so
  hardware results match only within a tolerance unless the sums are exact.
* **Simulator, DSMEM.** An access reaches the peer's shared memory, with the cluster-wide
  vector clocks: races between CTAs are reported, and the peer must be known to have
  started (a cluster barrier, relaxed or not, or a release chain before the first access),
  not to have exited, and must not exit before every access to its shared memory is
  ordered before its exit (a cluster barrier). A rank outside the cluster is reported.
* **Follow-ups.** `red.async` with mbarrier completion; the integer address of a DSMEM
  pointer (`toint`); A registers of MMAs that a loop carries in flight.

## The Hopper dense GEMM and its APIs at a glance

This ports `dense_gemm.py` as `examples/hopper_gemm.rs`, keeping its
class, method and variable names, and completes milestone m8. Porting it needed no new API;
m9-hopper-dense-gemm then swapped in m9's `PipelineTmaAsync`, `PipelineTmaStore`, `cute_struct!`
and helpers for the hand-written versions (the differences that remain are listed in its header).
Every m8 API, and where it is described:

| CuTe DSL | cute-rs | See |
|---|---|---|
| `warpgroup.MmaF16BF16Op`, `MmaF8Op`, `MmaI8Op` | `warpgroup::MmaF16BF16Op::new(..)`, `MmaF8Op::new(..)`, `MmaI8Op::new(..)` | Warpgroup MMAs; WGMMA variants |
| `warpgroup.OperandSource.SMEM` / `.RMEM` | `warpgroup::OperandSource::SMEM` / `RMEM` | WGMMA variants |
| `warpgroup.Field.ACCUMULATE`, `tiled_mma.set(field, v)` | `MmaField::ACCUMULATE`, `TiledMma::set(MmaField::ACCUMULATE, v)` (carried through loops) | TMA loads, copy keyword arguments and atom fields |
| `warpgroup.SmemLayoutAtomKind`, `warpgroup.make_smem_layout_atom(kind, dtype)` | `warpgroup::SmemLayoutAtomKind::K_SW128` (etc.), `make_smem_layout_atom::<T>(kind)`, `make_smem_layout_atom_dtype(kind, dtype)` | CuTe DSL → cute-rs |
| `warpgroup.fence()`, `commit_group()`, `wait_group(n)` | `warpgroup::fence()`, `commit_group()`, `wait_group(n)` | Warpgroup MMAs |
| `tiled_mma.make_fragment_A/B/C` | `TiledMma::make_fragment_a/b/c` (descriptor fragments for WGMMA) | Warpgroup MMAs |
| `warp.StMatrix8x8x16bOp(transpose, n)`, `warp.StMatrix16x8x8bOp(True, n)` | `CopyOp::StMatrix { num, trans }`, `CopyOp::StMatrix16x8x8b { num, trans: true }` | TMA descriptors, stmatrix and setmaxnreg |
| `cute.make_tiled_copy_C_atom`, `make_tiled_copy_S` | `make_tiled_copy_c_atom`, `make_tiled_copy_s` | TMA descriptors, stmatrix and setmaxnreg |
| `cute.arch.setmaxregister_increase/decrease` | `setmaxregister_increase/decrease` | TMA descriptors, stmatrix and setmaxnreg |
| `cute.arch.map_dsmem_ptr(ptr, rank)` | `map_dsmem_ptr(&ptr, rank)` | WGMMA variants and distributed shared memory |
| `CopyS2ROp`, `CopyR2SOp` (`shared_space=`) | `CopyOp::S2R { bits, attrs }`, `CopyOp::R2S { bits, attrs }` | WGMMA variants and distributed shared memory |
| `cpasync.CopyDsmemStoreOp` | `CopyOp::DsmemStore { bits }` | WGMMA variants and distributed shared memory |
| `cutlass.utils.sm90`: `get_smem_layout_atom`, `make_smem_layout_a/b/epi`, `make_trivial_tiled_mma` | `cute::utils::sm90::{get_smem_layout_atom, make_smem_layout_a, make_smem_layout_b, make_smem_layout_epi, make_trivial_tiled_mma}` | CuTe DSL → cute-rs; WGMMA variants |
| `cutlass.utils.LayoutEnum` | `LayoutEnum` (`from_tensor`, `sm90_mma_major_mode`, `is_k_major_a`, ...) | CuTe DSL → cute-rs |
| `cute.nvgpu.OperandMajorMode` | `OperandMajorMode::{K, MN}` | CuTe DSL → cute-rs |

What the port uses from m7 is in "The TMA APIs at a glance (m7)" above. Its
`compute_tile_shape_or_override` and `sm90_get_smem_store_op` (`cutlass.utils.sm90`) are
computed inline, as m9 brings them.

## TMA pipelines

The Hopper mainloop and epilogue pipelines, in `cute::pipeline`. Simulated (single CTA,
2x1, 1x2 and 2x2 clusters, seeded schedules) and assembled for sm_90a; no SM90 GPU runs
them yet.

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cutlass.pipeline.PipelineTmaAsync.create(*, num_stages, producer_group, consumer_group, tx_count, barrier_storage=None, cta_layout_vmnk=None, tidx=None, mcast_mode_mn=(1, 1), enable_multicast_signaling=False, defer_sync=False, name='')` | `pipeline::PipelineTmaAsync::create(n, producer_group, consumer_group, tx_count, PipelineTmaAsyncOptions { barrier_storage, cta_layout_vmnk, tidx, mcast_mode_mn, enable_multicast_signaling, defer_sync, name })`, implementing `Pipeline` and `PipelineBarriers`: `producer_acquire` waits on the empty barrier, then the elected lane arms the full barrier with `mbarrier.arrive.expect_tx`; `producer_commit` is a no-op (the TMA's `complete_tx` completes the phase); `producer_get_barrier` is the TMA's `tma_bar_ptr`; `consumer_release` arrives from the signalling lanes only (lane 0 of each warp for one CTA, so count consumers in warps times multicast partners, or set `enable_multicast_signaling`). A dynamic `cta_layout_vmnk` and `mcast_mode_mn` `(0, 0)` are refused. Init: `fence.mbarrier_init` and a CTA barrier, or a relaxed cluster arrive and wait. Deviation: a peer release arrives at cluster scope (`.release.cluster`), where CuTe DSL uses the CTA-scope default |
| `PipelineTmaAsync.init_empty_barrier_arrive_signal(cta_layout_vmnk, tidx, mcast_mode_mn)` | `pipeline::init_empty_barrier_arrive_signal(&layout, tidx, mode) -> (Int32, Boolean)` |
| `cutlass.pipeline.PipelineTmaStore.create(*, num_stages, producer_group)`; `producer_acquire()`, `producer_commit()`, `producer_tail()` | `pipeline::PipelineTmaStore::create(n, producer_group)` with inherent, state-free methods (not the `Pipeline` trait, whose methods take a state): `cp.async.bulk.wait_group.read n-1`, `cp.async.bulk.commit_group`, `wait_group.read 0`; `consumer_wait` / `consumer_release` panic |
| `cutlass.pipeline.TmaStoreFence(num_stages)` | `pipeline::TmaStoreFence::new(n)`, also a `SyncObject`: `arrive` (commit_group), `wait` (wait_group.read n-1), `arrive_and_wait`, `tail` (wait_group.read 0); `get_barrier`, `max` and `arrive_and_drop` panic (the trait's `max` is `None`) |

## GEMM host helpers

SM90 epilogue helpers, the Blackwell TMA alignment preflight checks and torch-free GEMM
host tensors, under `cute::utils::{sm90, sm100, gemm}`. The checks return a
`testing::CantImplement` with CuTe DSL's message; the host tensors need no GPU.

| CuTe DSL (Python) | cute-rs |
|---|---|
| `utils.sm90.get_smem_store_op(layout_d, elem_ty_d, elem_ty_acc)` (`sm90_get_smem_store_op`) | `utils::sm90::get_smem_store_op::<D>(layout_d, acc_dtype)` (and `sm90_get_smem_store_op`): `StMatrix { num: 4, trans: is_m_major_c }` for a 16-bit D, `CopyOp::Universal` otherwise |
| `utils.sm90.compute_tile_shape_or_override(tile_shape_mnk, element_type, is_cooperative, epi_tile_override)` | `utils::sm90::compute_tile_shape_or_override(tile_mnk, dtype, is_cooperative, Option<(m, n)>)` |
| `utils.sm100.get_tma_aligned_contiguous_elements(elem_ty)`, `check_tma_tensor_alignment(...)`, `check_gemm_tma_alignment(...)` | `utils::sm100::{get_tma_aligned_contiguous_elements, check_tma_tensor_alignment, check_gemm_tma_alignment}`: `Result<_, CantImplement>`, CuTe DSL's messages; the majors are `GemmMajor::{M, N, K}` (`FromStr` for `"m"`/`"n"`/`"k"`), or `check_gemm_tma_alignment_str` for CuTe DSL's strings |
| `utils.sm100.check_int8_mma_arch(ab_dtype, target_arch=None)` | `utils::sm100::check_int8_mma_arch(dtype, Option<Arch>)`: Int8/Uint8 fail on sm_107; the target is the argument, else `CUTE_DSL_ARCH`, else the detected GPU (feature `cuda`), else unchecked; `check_int8_mma_target(dtype, "sm_107")` takes a string |
| `utils.gemm.create_gemm_tensor_torch(...)`, `get_gemm_tensor(...)`, `get_gemm_tensors(...)` | `utils::gemm::{create_gemm_tensor, get_gemm_tensor, get_gemm_tensors}` with a seeded `Rng`: `(MN, K, L)` operands of {-1, 0, 1} with MN-major `(1, MN, MN K)` or K-major `(K, 1, MN K)` strides; a `GemmTensor` holds the `KernelArg`, the `Layout` and the logical values; `host_tensor()` marks the leading dimension dynamic. FP4 operands are one byte per element, as in CuTe DSL |
| `create_scale_factor_tensor(...)`, `unpack_scale_factors(...)`, `decode_float4e2m1fn(...)` | `utils::gemm::{create_scale_factor_tensor, unpack_scale_factors, pack_scale_factors, scale_factor_offset, decode_float4e2m1fn}`: from the explicit index map (to be cross-checked with `BlockScaledBasicChunk` in the m10 block-scaled port) |
| (the examples' torch reference) | `utils::gemm::{reference_gemm, reference_gemm_values, to_memory_order}`: `D = A B^T` with f64 accumulation |
