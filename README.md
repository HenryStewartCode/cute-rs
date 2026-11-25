# cute-rs

A CuTe-style GPU kernel DSL for Rust.

You write kernels as ordinary Rust, using the same building blocks as NVIDIA's
[CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) and CuTe DSL:
hierarchical layouts and their algebra, tensors over global, shared and
register memory, and copy and MMA atoms tiled across threads. A built kernel can be

* **simulated on the CPU** by a checking simulator. It catches shared-memory
  races (ordered by `sync_threads`, named barriers, mbarriers, and flags
  with release and acquire semantics or fences), missing
  `cp.async` waits, bulk copies outside their mbarrier's transaction count,
  `cp.async` copies an mbarrier tracks whose count leaves no room for them,
  mbarrier parity waits that could pass early on hardware, out-of-bounds and
  misaligned accesses, non-coherent loads of memory the kernel writes,
  divergent barriers and barriers that cannot complete, warp collectives run
  outside their member masks or without all of their lanes, and reports each
  one with the Rust source line,
* **lowered to MLIR** using only upstream dialects (`gpu`, `nvgpu`, `nvvm`,
  `scf`, `arith`, `memref`, `vector`), with CuTe DSL's `cute.math` library
  expanded into exactly rounded operations, so that no libdevice is
  needed and the GPU computes the simulator's bits,
* **compiled to PTX** with a stock `mlir-opt` (no CUDA toolkit needed), with
  CuTe DSL's compile options: an optimization level, line information that
  maps PTX and SASS back to the Rust source, and the same option strings and
  environment variables ([Compile options](#compile-options)),
* **run on an NVIDIA GPU** through the CUDA driver API (`cuda` feature).

```rust
use cute::prelude::*;

const N: usize = 1024;

let kernel = Kernel::builder("scale").block_dim([256, 1, 1]).build(|| {
    let x = param_tensor::<f32>("x", Layout::col_major(N));
    let y = param_tensor::<f32>("y", Layout::col_major(N));
    let i = block_idx().x * 256 + thread_idx().x;
    y.store(i, x.load(i) * 2.0);
});

let mut args = vec![KernelArg::f32((0..N).map(|i| i as f32).collect()), KernelArg::f32(vec![0.0; N])];
kernel.simulate(LaunchConfig::new([4, 1, 1], [256, 1, 1]), &mut args).unwrap();
assert_eq!(args[1].as_f32()[10], 20.0);

println!("{}", kernel.to_mlir());
```

## Status

cute-rs covers the features of CuTe DSL 4.8.1, apart from three left out by design: the
`frontend-next` tracing frontend, and `cute-ext`'s compiler pipeline and entry points
(cute-rs builds kernels from Rust closures instead). Every feature is tested on the
simulator, and the generated PTX is assembled by `ptxas` for each architecture it targets
(except SM107, which no released CUDA toolkit assembles yet).

What has run on real GPUs, and what has not:

| Architecture | Hardware tested | Status |
| --- | --- | --- |
| SM80 / SM89 (Ampere, Ada) | an SM89 GPU | The full GPU suite passes |
| SM120 (consumer Blackwell) | RTX 5090 | The full GPU suite passes, including TMA, the SM120 MMAs and the block-scaled warp MMAs |
| SM90 (Hopper) | none yet | WGMMA, the Hopper GEMMs and FMHA are checked by the simulator and `ptxas` only |
| SM100 / SM103 / SM107 (datacenter Blackwell, Rubin) | none yet | tcgen05, tensor memory, UMMA and CTA pairs are checked by the simulator and `ptxas` only |

`tests/gpu_hopper.rs` and `tests/gpu_blackwell.rs` run the Hopper and datacenter-Blackwell
examples on those GPUs against CPU references, and time them (the `#[ignore]`d `bench_*`
tests). `scripts/gpu-validate.sh` does a whole validation pass on a GPU machine in one go:
it installs Rust and `mlir-opt`, runs the architecture's suite, then the full suite, then the
timings (with cuBLAS alongside when PyTorch is installed), and packs the results.
`scripts/pod-run.sh` drives it on a rented machine over SSH (`start`, `status`, `fetch`).

Known gaps besides the Hopper and datacenter-Blackwell hardware:

* No performance comparison with CuTe DSL yet; the tests check results, not speed.
* The JAX (`xla_ffi`) and TVM-FFI bindings are tested against hand-built call frames, not
  under JAX or `libtvm_ffi` themselves.
* A CTA pair's B scale factors (the 2x2 data path of `make_tmem_layout_sfb`) are not laid
  out yet.
* The JIT disk cache refuses a directory other users can write to, which includes network
  volumes that report mode 777; point `CUTE_RS_CACHE_DIR` at local disk there.

The [Limitations](#limitations) section lists each feature's smaller gaps.

## Examples

| Example | What it shows |
|---|---|
| [`elementwise_add`](examples/elementwise_add.rs) | The CuTe DSL tutorial. A naive kernel, then a tiled one built from `make_layout_tv`, with 128-bit vector copies and an identity tensor predicating a partial last tile. Then `elementwise_add.py` itself on `mark_layout_dynamic()` host tensors (`Kernel::jit_for`): one compiled kernel for every `M x N`, `zipped_divide` by the TV tiler, an identity tensor over the runtime `mC.shape` and `elem_less` against it, TensorSSA `a + b`, and the grid `size(gC, mode=[1])` computed on the host. |
| [`ampere_sgemm`](examples/ampere_sgemm.rs) | CuTe DSL's `sgemm.py`: an f32 SIMT GEMM with runtime `M`, `N` and `K` (any sizes, not tile multiples), for each major mode of A, B and C. Host tensors marked with `mark_layout_dynamic(leading_dim)` and `mark_compact_shape_dynamic(mode, divisibility)`; `max_alignment` of mode 0 picks 16-byte or 4-byte `cp.async`; a 3-stage pipeline; the K residue first, through `domain_offset((0, residue_k, 0))` of the tiles and identity tensors and `elem_less((m, -1), (M, k))`; a 16x16 `UniversalFma` `TiledMma` with `permutation_mnk`; the k-block loop as `for_range_with(.., LoopOptions::unroll_full())`, which LLVM unrolls (512 FMAs per K tile, no local memory); an epilogue predicated with the M and N residues; the grid `ceil_div(mC.shape, (bM, bN))`. Bit-exact against an in-order FMA chain, on the simulator and on the SM89 GPU. |
| [`sgemm`](examples/sgemm.rs) | CuTe's `sgemm_2`: a SIMT GEMM with tiled copies into shared memory, a 16×16 `UniversalFma` `TiledMma`, a GPU-side K loop carrying the accumulators, and an α/β epilogue. |
| [`tensorop_gemm`](examples/tensorop_gemm.rs) | An SM80 tensor-core GEMM on f16 or bf16 inputs. 4 warps of `mma.sync.m16n8k16`, a 3-stage `cp.async` pipeline into swizzled shared memory, and `ldmatrix` copies derived from the MMA's own layouts. Bank conflicts are measured with and without the swizzle. Its FP8 path for SM89 (`GemmConfig::fp8`, `tensorop_gemm.py`'s): Float8E4M3FN or Float8E5M2 inputs from `KernelArg::fp8_from_f32`, `MmaFP8Op` 16x8x32 with the example's `permutation_mnk` (32, 32, 32), a 128x128x64 tile, `LdMatrix8x16x8bOp(False, 4)` for A and B, the swizzle of `_make_smem_layout_AB` (`Swizzle<2,4,4>`, which leaves `ldmatrix` 2-way bank-conflicted: the simulator reports 2.00x), and Float32 or FP8 C. Bit-exact on small integers, and on the SM89 GPU within the f32 summation bound plus 2^-13 of the magnitude per FP8 `mma.sync` on random data (SM89 accumulates FP8 products with fewer bits than Float32; the simulator sums in Float32). |
| [`cute_ext_memcpy`](examples/cute_ext_memcpy.rs) | CuTe DSL's `cute_ext/ampere/memcpy_simt_universal_copy.py`: `D = A + addend` over an `M x N x L` Float16 tensor with runtime extents, through `cute::experimental`'s register `allocate`, `partition` by an explicit TV layout and 128-bit `copy`, which predicates against the argument tensors' shape by itself (the default (136, 7, 9) leaves partial tiles). Bit-exact on the simulator and the SM89 GPU; `ld.global.v4`/`st.global.v4` in PTX. |
| [`cute_ext_sgemm`](examples/cute_ext_sgemm.rs) | CuTe DSL's `cute_ext/ampere/sgemm.py`: a one-warp SM80 GEMM in the fragment-free style, with shared and register buffers from `allocate`, 32-bit `cp.async` copies of partitions marked with `predicated_tensor_origin`, `partition_and_copy` with `ldmatrix`, `dot` with the 16x8x8 `mma.sync` and a predicated universal-copy epilogue. Exact on small integers and within the f32 summation bound on random data, on the simulator and the SM89 GPU. |
| [`cute_ext_dense_gemm_persistent`](examples/cute_ext_dense_gemm_persistent.rs) | CuTe DSL's `cute_ext/blackwell/dense_gemm/dense_gemm_persistent.py`: a persistent warp-specialised SM100 GEMM (Float16 in, Float32 accumulator and D) from `allocate`, `tma_load` into a `TMAToUMMAPipeline`, `mainloop_mma`, a `UMMAtoAsyncPipeline` to `SM100_COPY_T2R` consumers, `epilogue_tma_store` through a `TMAStorePipeline` and the static persistent tile scheduler; 1SM or 2SM MMAs, TMA multicast, preferred and fallback clusters (one body per cluster shape with `mixed_cluster_arms`). Bit-exact on small integers and within the f32 bound on random data on the simulator (1SM (1,1), 2SM (2,1), 2SM multicast (2,2), ragged problems, mixed clusters); `ptxas -arch=sm_100a` only, no Blackwell GPU. |
| [`single_warp_pipeline`](examples/single_warp_pipeline.rs) | The kernel this project started from: one warp with a double-buffered `cp.async` pipeline, unrolled at build time, using `ldmatrix.trans` for an N-major B. |
| [`runtime_gemm`](examples/runtime_gemm.rs) | `tensorop_gemm` with `M`, `N` and `K` only known at launch: one compiled kernel for every problem size. The global tensors have static upper bounds as their extents and runtime strides (`build`), or their runtime extents (`build_dynamic`, CuTe DSL's `(?,?):(?,1)`), whose rest mode counts the K tiles; the two give the same bits. |
| [`persistent_gemm`](examples/persistent_gemm.rs) | `tensorop_gemm`'s mainloop in a persistent kernel: a `while_carry` over the work tiles of a `StaticPersistentTileScheduler` (cluster 1x1x1), with a grid smaller than the tile count sized by `get_grid_shape` (on a GPU, from `HardwareInfo::get_max_active_clusters(1)`). Runtime `M`, `N` and `K`; unswizzled and swizzled schedules, skipping the swizzle's padding tiles. Bit-exact on small integers on the simulator and the SM89 GPU |
| [`block_reduce`](examples/block_reduce.rs) | CuTe DSL's `block_smem_reduce.py`: a block reduces each row of its tile (add, max or min of Int32, Float32, Float16 or BFloat16): a per-lane fold, `redux.sync` or a shuffle tree per warp, and `red` (or `atomic_fmax`/`atomic_fmin`, or a shared-memory combine) across warps. |
| [`redux_sync`](examples/redux_sync.rs) | Kernels 1 to 4 of CuTe DSL's `redux_sync.py`: the eight integer `redux.sync` operations next to a shuffle tree, the Float32 forms with `abs` (sm_100a) next to a `shuffle_sync_down` abs-max, and per-row abs-max then FP8 quantisation (the MXFP8 pattern: `redux.sync.max.abs.f32` on sm_100a or the shuffle tree on SM89, scale `amax / 448`, `rcp.approx.ftz`, the satfinite E4M3 `cvt`), checked with the example's tolerances and on the SM89 GPU against the simulator. |
| [`fp4_x2_tensor`](examples/fp4_x2_tensor.rs) | The recast half of CuTe DSL's `tvm_ffi/fp4_x2_tensor.py`: `torch.float4_e2m1fn_x2` bytes as a `Float4E2M1FNx2` tensor and as a Uint8 tensor, both recast (`recast_tensor`) to logical Float4E2M1FN tensors and decoded to Float16 eight values per thread, with integer operations (every target; run on the SM89 GPU) or `cvt_f4e2m1x8_to_f16x8` (sm_100a and sm_120a: simulated and assembled). A `TvmFfiFunction` (m6) takes such a tensor through `from_dlpack`; the TVM-FFI export of the example (`aot_export.py`) belongs to m11. |
| [`red`](examples/red.rs) | CuTe DSL's `red.py`: every thread reduces one element into global sum and maximum cells with `red`. |
| [`warp_named_barrier`](examples/warp_named_barrier.rs) | CuTe DSL's `warp_named_barrier.py`: a producer warp fills a ring of 2 or 4 shared-memory stages through full/empty mbarriers (empty ones pre-arrived, parity `(k / STAGES) & 1`) and flushes with barrier 0 over its 32 threads; a consumer warp accumulates and does `bar.arrive(1, 64)`; an epilogue warp does `bar.sync(1, 64)` and stores. Runtime tile count; SM89 (`mbarrier_test_wait`) and SM90 (`try_wait`, init fence) builds, simulated under every schedule and seeds. |
| [`pipeline_software_fill`](examples/pipeline_software_fill.rs) | CuTe DSL's `pipeline_software_fill.py`: a producer warp fills stages with direct stores, a consumer warp writes them back to global memory, full/empty mbarrier arrays with the pre-signal trick. |
| [`cp_async_mbarrier_arrive`](examples/cp_async_mbarrier_arrive.rs) | CuTe DSL's `cp_async_mbarrier_arrive_cta.py` (a producer warp's 16-byte `cp.async.cg` and `cp_async_mbarrier_arrive_noinc`, a consumer warp waiting on the mbarrier of 32 arrivals) and `cp_async_mbarrier_arrive_cluster.py` (SM90, a 2-CTA cluster: producer, forwarder and consumer warps, local mbarriers with `noinc` arrivals fanning in to CTA 0's leader mbarrier at cluster scope, CTA 0 releasing both CTAs' empty mbarriers; each CTA writes its own tile instead of the example's peer read). |
| [`async_pipeline`](examples/async_pipeline.rs) | CuTe DSL's `async_pipeline.ipynb` tutorial: a producer and a consumer warp hand eight values over a `sync_threads`-guarded one-element buffer, then through a one-stage and a five-stage `PipelineAsync` made from a `cute_struct!` `SharedStorage` (mbarriers and a 1024-byte-aligned staging buffer) and driven by its participants, then with `try_acquire` / `try_wait` tokens. The notebook's outputs (`res = [0, ..., 7]`, `staging = [5, 6, 7, 3, 4]`) on the simulator for SM89 and SM90 and on the SM89 GPU (`mbarrier.test_wait` spins) |
| [`grouped_gemm`](examples/grouped_gemm.rs) | A persistent SM80 `mma.sync` grouped GEMM over concatenated A, B and C buffers with a per-group offset table and a `(group_count, 4)` problem-shape tensor, each tile's group found by `StaticPersistentGroupTileScheduler`'s warp-wide search (direct or cached shapes; up to 37 groups, empty ones included), and an MoE-style layer over one concatenated token tensor whose scheduling warp runs `contiguous_group_search` and hands tile records to a compute warp through a `PipelineAsync`. Exact against the CPU on the simulator and the SM89 GPU |
| [`cooperative_launch`](examples/cooperative_launch.rs) | CuTe DSL's `cooperative_launch.py`: a grid-wide barrier from release adds and acquire spins on one global word, crossed ten times by 8 blocks. Launched cooperatively (`LaunchConfig::with_cooperative(true)`): the simulator runs the whole grid resident (without the attribute, one block at a time, it reports a hang with a hint to launch cooperatively), and the GPU launch sets `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` after an occupancy check. |
| [`call_bypass_dlpack`](examples/call_bypass_dlpack.rs) | CuTe DSL's `call_bypass_dlpack.py`: a kernel on three Float16 pointers assumed 32-byte aligned and runtime `m`, `n`, `k`, `l`, building its own ordered layouts over them (a one-element-per-thread GEMM stands in for the tutorial's `TensorOpGemm`). Simulated on host buffers; on the GPU it runs on raw device pointers (`DevicePtr::from_raw`). |
| [`programmatic_dependent_launch`](examples/programmatic_dependent_launch.rs) | CuTe DSL's `programmatic_dependent_launch.py`: `w = u + v` then `y = w + x` in one stream, the second launched with `use_pdl`, loading `x` before `griddepcontrol_wait()` and `w` after it. Simulated through a `SimStream`, whose PDL model checks the pre-wait accesses; assembled for sm_90 (the SM89 GPU has no griddepcontrol). |
| [`elementwise_apply`](examples/elementwise_apply.rs) | CuTe DSL's `elementwise_apply.py`: a generic `C = op(inputs...)` kernel with `op` a Rust closure over `TensorSsa` values; 4x64 threads by 16 rows of 16 bytes (`recast_layout`), `make_layout_tv`, `zipped_divide` composed with the block remap, `load_ssa` and `store_ssa` as 128-bit `ld.global.v4`/`st.global.v4` (100% global efficiency). `add`, `sub`, `mul`, `truediv`, `floordiv` and `leaky_relu` on Float32, `add` and `mul` on Float16 and BFloat16, bit-exact against a CPU reference and on the SM89 GPU. Static `M` and `N` (CuTe DSL's output is dynamic: m5). |
| [`cta_norm`](examples/cta_norm.rs) | CuTe DSL's `cta_norm.py`: LayerNorm and RMSNorm with one CTA per row on Float16, BFloat16 and Float32: 128-bit copies of 8 elements per thread under per-chunk predicates, Float32 statistics with a butterfly warp reduction and a shared-memory CTA reduction, precise `rsqrt`, and the TensorSSA epilogue `(x - mean) * factor * weight + bias` promoted to Float32 and converted back. Within a documented tolerance of an f64 reference, bit-exact on the SM89 GPU. Chunks load with `load_ssa` in an `if_else_yield` (a register written in a branch is yielded out of it), and `eps` is a constant. |
| [`cuda_graphs`](examples/cuda_graphs.rs) | CuTe DSL's `cuda_graphs` notebook: launches on a stream captured into a CUDA graph and replayed, timed against launching one by one (needs `--features cuda`). |
| [`export_to_c`](examples/export_to_c.rs) | CuTe DSL's `dsl_tutorials/export/export_to_c.py`: two kernels on tensors of launch-time shape exported to C (header, source with the PTX embedded, manifest, and an object file when `cc` and `cuda.h` are found), whose object files are linked into one shared library. |
| [`load_exported`](examples/load_exported.rs) | CuTe DSL's `load_in_python.py`: the exports of `export_to_c` loaded back with `CudaContext::load_exported` and launched with checked arguments, without their Rust code or `mlir-opt` (needs `--features cuda`). |
| [`run_on_gpu`](examples/run_on_gpu.rs) | Runs the tensor-core GEMM on a real GPU and checks it (needs `--features cuda`). |
| [`elementwise_add_autotune`](examples/elementwise_add_autotune.rs) | CuTe DSL's `elementwise_add_autotune.py`: the tiled element-wise add of runtime `M x N` over candidate tiles (thread layouts and 32-, 64- or 128-bit copies, and an unbuildable `copy_bits = 1`), each checked on the simulator; with `--features cuda`, `cute::testing::tune` times them on the GPU, skipping what cannot be built or implemented, and an `Autotuner` keyed on the spec and `M` and `N` tunes each new size once. |
| [`tvm_ffi_add`](examples/tvm_ffi_add.rs) | CuTe DSL's `jit_and_use_in_torch.py` and `error_reporting.py` as a Rust `cdylib`: `b = a + 1` exported as the TVM-FFI functions `__tvm_ffi_add_one` (dynamic extent) and `__tvm_ffi_add_one_static` (`(10,)`), which `tvm_ffi.load_module` loads and calls on torch tensors (`examples/tvm_ffi/*.py`), refusing a wrong shape, dtype or device with a Python `ValueError`. |
| [`xla_cutlass_call`](examples/xla_cutlass_call.rs) | The XLA FFI handler for JAX as a Rust `cdylib` (CuTe DSL's `cutlass.jax.cutlass_call`; `cute::xla_ffi_export!()` and a registered target `add_one`), loaded by `python/cute_rs_jax` and `examples/xla/cutlass_call_basic.py` inside `jax.jit` and `jax.export`. Not run against a real JAX here (none is installed): the handler is tested from Rust with hand-built call frames (`tests/gpu_xla_ffi.rs`). |
| [`tma_copy`](examples/tma_copy.rs) | CuTe DSL's TMA tutorial `tma_v0.py` (`Sm100SimpleCopyKernel`): a TMA load atom and a TMA store atom over one shared-memory layout, `tma_partition` with `make_layout(1)` and `group_modes`, `elect_one` around `mbarrier_init` and `mbarrier_expect_tx`, `mbarrier_init_fence`, the load with `tma_bar_ptr`, `mbarrier_arrive` and `mbarrier_wait`, then the TMA store and its bulk async-group. Runtime `M` and `N` (partial edge tiles zero-filled and clipped). Simulated bit for bit and assembled for sm_90; no SM90 GPU has run it. |
| [`tma_transpose`](examples/tma_transpose.rs) | CuTe DSL's TMA tutorial `tma_v1.py` (`Sm100MatrixTransposeKernelV1`): a warp-specialised transpose. A TMA-load warp, four transpose warps doing S2R and R2S through a universal `make_tiled_copy_tv` copy between a K-major and an M-major 128-byte-swizzled tile, `fence_proxy` (`async.shared`, `cta`) and a `NamedBarrier`, an mbarrier handing the tile to the TMA-store warp. Simulated bit for bit under seeded schedules and assembled for sm_90. |
| [`blackwell_fp16_gemm`](examples/blackwell_fp16_gemm.rs) | CuTe DSL's Blackwell tutorial `fp16_gemm_0.py`: a 1-CTA tcgen05 GEMM `C = A B^T` (K-major Float16 A and B, Float32 accumulator in TMEM, Float16 C) in `(128, 256, 64)` tiles. Warp 0 issues the TMA loads (`make_tiled_tma_atom_A/B`) through a 4-stage `PipelineTmaUmma` and the `128x256x16` MMAs (`gemm` with `Field::ACCUMULATE`); all four warps run the epilogue through a 1-stage `PipelineUmmaAsync`: `make_tmem_copy` with `Ld32x32bOp(Repetition::X64)` in four subtiles, Float16 conversion and `autovec_copy`. `TmemAllocator` (512 columns, named barrier 1), `cute_struct!` shared storage, `SmemAllocator`. Runtime `M`, `N`, `K`; the configuration is a value (`Fp16GemmConfig`) for the tutorial's later steps. `FP16_GEMM_1` is `fp16_gemm_1.py` (m10-cta-pair-pipelines): a `(2, 1, 1)` cluster whose CTA pair runs `256x256x16` `CtaGroup::TWO` MMAs on `(256, 256, 64)` tiles with 7 stages, multicast TMA loads with `create_tma_multicast_mask` masks, the even CTA alone issuing the MMAs, `TmemAllocator` with `is_two_cta`, both pipelines given `cta_layout_vmnk`, and the producer tails; other clusters (`(2, 2, 1)`, `(4, 1, 1)`) only change `cluster_shape_mnk`. The warp-specialized steps are `Fp16GemmWsConfig` (m10-port-clc-clusters; epilogue warps 0-3, MMA warp 4, TMA warp 5, CLC scheduler warp 6, the packaged `epilogue_tma_store`): `FP16_GEMM_2` (`fp16_gemm_2.py`, warp specialization and a TMA-store epilogue, 6 stages), `FP16_GEMM_3` (`fp16_gemm_3.py`, `StaticPersistentTileScheduler`, 2 accumulator stages), `FP16_GEMM_3_1` (`fp16_gemm_3_1.py`, `ClcDynamicPersistentTileScheduler` with a 1-stage `PipelineClcFetchAsync` and the response in the shared storage; `Scheduling` is `use_clc_dynamic_scheduler`) and `FP16_GEMM_4` (`fp16_gemm_4.py`, preferred `(2, 4, 1)` and fallback `(2, 1, 1)` clusters with CLC, the body built per shape by `utils::mixed_cluster_arms`; `FP16_GEMM_4_STATIC` with the static scheduler). Runtime `M`, `N`, `K` (TMA zero-fills and clips ragged and padding tiles). Simulated bit for bit, with every tile computed once: persistent grids smaller and larger than the tile count, CLC under several resident counts and launch seeds, and `fp16_gemm_4` with all-preferred, all-fallback and mixed clusters; assembled for sm_100a with `ptxas`. No Blackwell GPU has run any of it: preferred clusters and CLC have run only in the simulator. |
| [`blackwell_dense_gemm`](examples/blackwell_dense_gemm.rs) | CuTe DSL's `blackwell/kernel/dense_gemm/dense_gemm.py` (`DenseGemmKernel`, m10-port-dense-gemm): `C = A B^T` with K-major Float16 A and B, a Float32 TMEM accumulator and a Float16 or Float32 C in either major, on one CTA or a CTA pair in a cluster of any shape. `make_trivial_tiled_mma`, `make_smem_layout_a/b/epi`, stage counts from `_compute_stages`, `cluster_shape_to_tma_atom_A/B` with multicast masks, `get_num_tmem_alloc_cols` and `TmemAllocator`, `PipelineTmaUmma`, `PipelineUmmaAsync` and `PipelineTmaStore`; the warp roles are separate functions (`tma_load_warp`, `mma_warp`, `epilogue_warps`) that the CLC and EFC ports reuse. The epilogue is the packaged `epilogue_tma_store` (m10-epilogue-helpers) instead of the example's own. Runtime `M`, `N`, `K` that need not be tile multiples (`K` and `N` multiples of 8). Simulated bit for bit against a CPU reference for a 1-CTA, a 2x1 pair and a 2x2 multicast configuration, race-free under several schedules, and assembled with `ptxas -arch=sm_100a`; no Blackwell GPU has run it. Not ported: the non-TMA epilogue, the batch mode, Float8/Int8/TF32 inputs and M-major operands. |
| [`blackwell_nvfp4_gemm`](examples/blackwell_nvfp4_gemm.rs) | CuTe DSL's Blackwell tutorial `nvfp4_gemm_0.py`: a 1-CTA block-scaled tcgen05 GEMM `C = (A SFA) (B SFB)^T` with packed Float4E2M1FN A and B (K-major), one 8-bit scale factor per 16 K elements (UE4M3, `NVFP4_GEMM_0`, or UE8M0, `NVFP4_GEMM_0_UE8M0`) laid out by `tile_atom_to_shape_sf`, Float32 accumulator in TMEM and Float16 C, in `(128, 256, 256)` tiles of `128x256x64` `MmaMXF4NVF4Op` MMAs. Warp 0 issues TMA loads of A, B, SFA and SFB (shared-memory layouts from `make_smem_layout_a/b/sfa/sfb`) through a 4-stage `PipelineTmaUmma`, copies each stage's scale factors into TMEM after the accumulator (`find_tmem_tensor_col_offset`, `make_tmem_layout_sfa/sfb`, `Cp4x32x128bOp` through `make_s2t_copy` and `get_s2t_smem_desc_tensor`) and sets them per k-block in `Field::SFA`/`SFB`; the epilogue loads the accumulator with `Ld32x32bOp(Repetition::X128)`. Runtime `M`, `N`, `K`. Simulated bit for bit against a CPU reference that decodes FP4 and the scale factors, and assembled for sm_100a; no Blackwell GPU has run it. The 2-CTA step (`nvfp4_gemm_1.py`) is not ported. |
| [`blackwell_mixed_input_gemm`](examples/blackwell_mixed_input_gemm.rs) | CuTe DSL's `blackwell/kernel/mixed_input_gemm/mixed_input_gemm.py` (`MixedInputGemmKernel`, m10-mixed-input), reduced: `C = (convert(A) [* scale]) B^T` with Int4 A, BFloat16 B and scales, Float32 accumulator and C. The example's warp roles (epilogue 0-3, MMA 4, TMA 5, scale TMA 6, transform 8-11); the transform warps convert A with `cvt_tensor_a` and store it into TMEM (`tcgen05.st.32x32b.x8`, a TS-UMMA) for K-major A or into shared memory for M-major A, handing it over through `PipelineAsyncUmma`; `PipelineTmaAsync` for A and the scales, `PipelineTmaUmma` for B; ConvertOnly and ConvertScale (one scale per row and K granule, its own TMA); one CTA, or a CTA pair with B's 2-CTA load (M-major A). One tile per CTA, fixed stage counts and tile-multiple `M`, `N`, `K` fixed at build; simulated bit for bit against a CPU reference, race-free under several schedules, and assembled with `ptxas -arch=sm_100a`; no Blackwell GPU has run it. Not ported: the persistent scheduler, the grouped variants, Int8 A, the non-TMA epilogue and a CTA pair with K-major A (a 2-CTA TS-UMMA, which cute-rs does not model). |
| [`synthetic_custom_epilogue_dense_gemm`](examples/synthetic_custom_epilogue_dense_gemm.rs), [`activation_custom_epilogue_dense_gemm`](examples/activation_custom_epilogue_dense_gemm.rs), [`custom_epilogue_dense_gemm`](examples/custom_epilogue_dense_gemm.rs) | CuTe DSL's Blackwell EFC examples (`blackwell/efc/`, m10-port-efc): the epilogue fusion configuration framework (`common_efc.py`) and its persistent, warp-specialized SM100 GEMM `DenseGemmEFC` (`common_dense_gemm_efc.py`), as the shared example module `examples/efc` (example-side code, not a crate API). An epilogue is one Rust closure over named Float32 scalars and `(M, N)` tensors (`efc_config.accum()`, `load`, `store`, `where_`, `full_like`, and identity, relu, leaky_relu, tanh, sigmoid, silu, hardswish and gelu on its values), run in three phases: an analysis of which tensors it reads and writes, which gives each read tensor a TMA load through a `PipelineTmaAsync` fed by an epilogue-load warp and each written one a TMA store through `PipelineTmaStore`, with their SMEM in the stage counts; the kernel's epilogue on register subtiles; and the CPU reference. The signature is declared (`efc_config` first); the rules (that parameter, a written tensor, one majorness) are `EfcError`s. 1-CTA and CTA-pair MMAs in clusters with A/B multicast, the static persistent scheduler, mixed Float16/BFloat16/Float32 tensors. The three examples (a synthetic epilogue of any number of read and written tensors with `where`/`full_like`, `D = activation(Aux)` for each activation, and the multi-input, multi-output `Y = acc; D = relu(alpha acc + beta C) + x_factor X`) are simulated against the host evaluation on ragged problems (exact on small integers; one Float16 ulp for the transcendental activations, plus 1e-3 for GELU's tanh approximation), race-free, and assembled with `ptxas -arch=sm_100a` (one TMA load per read tensor and one TMA store per written tensor per subtile); no Blackwell GPU has run them. Not ported: batch mode, Float8/Int8/TF32 inputs, M-major A/B, CLI parsing and benchmarking; M-major Float32 EFC tensors need TMA's SW128_32B swizzle mode, which cute-rs lacks. |
| [`rmsnorm_cluster`](examples/rmsnorm_cluster.rs) | CuTe DSL's `blackwell/kernel/rmsnorm/rmsnorm.py` with its helper `reduce.py`: RMSNorm whose long rows are split across a thread-block cluster on SM90+ (the example's table: 2 to 16 CTAs past 16K Float16/BFloat16 or 32K Float32 elements) and run on one CTA on SM80. 128-bit `cp.async` to shared memory under per-vector predicates, TensorSSA `x * x` folded per thread, then `row_reduce`: a warp butterfly, `block_reduce` through shared memory, or `cluster_reduce`, where each warp `st.async`s its partial into every CTA's reduction buffer (`mapa`, `mbarrier::complete_tx`) and each CTA waits on its own mbarrier's `expect_tx`; `rsqrt`, a cluster barrier, and the weighted output. One kernel per `N` serves every runtime `M`. Simulated for clusters of 1, 2 and 4 on Float16, BFloat16 and Float32 against an f64 reference, assembled for sm_90a, and run single-CTA on the SM89 GPU (bit-exact with the precise `rsqrt`, within one ulp with the example's `fastmath`). |
| [`hopper_gemm`](examples/hopper_gemm.rs) | CuTe DSL's `hopper/kernel/dense_gemm/dense_gemm.py` (`HopperWgmmaGemmKernel`): a batched SM90 GEMM with TMA loads multicast across clusters of up to 4 CTAs (A along N, B along M, masks from `make_layout_image_mask`), a multi-stage ring on `PipelineTmaAsync` (consumer warps release a stage on every CTA that shares it, through peer-CTA arrivals) with the shared memory in a `cute_struct!`, warpgroup MMAs from shared-memory descriptor fragments with `k_pipe_mmas` groups in flight, and an epilogue of `stmatrix` (`make_tiled_copy_C_atom`, `make_tiled_copy_S`), `fence_proxy` and TMA stores in bulk groups paced by `PipelineTmaStore`. Float16, BFloat16, FP8 and 8-bit integer inputs, every major mode the example allows, one or two warpgroups, runtime M, N, K and L. Simulated against a plain Rust GEMM (bit-exact on small integers) for clusters of 1x1, 2x1, 1x2 and 2x2 and assembled with `ptxas -arch=sm_90a`; it has run only on the simulator and through `ptxas`, no SM90 GPU has run it. |
| [`hopper_persistent_gemm`](examples/hopper_persistent_gemm.rs) | CuTe DSL's `hopper/kernel/dense_gemm/dense_gemm_persistent.py` (`HopperWgmmaGemmPersistentKernel`), on `hopper_gemm`'s configuration, layouts and TMA atoms: warp-specialised roles (a DMA warpgroup that lowers its registers with `setmaxregister_decrease(40)` and loads with TMA multicast, and one or two MMA warpgroups that raise theirs and run WGMMAs and the `PipelineTmaStore` epilogue, the first MMA warp issuing the stores), a persistent grid of at most `max_active_clusters` clusters (`cuda::HardwareInfo` on a GPU) walking `StaticPersistentTileScheduler` tiles with a swizzle size and raster order, mainloop pipeline states carried across tiles, and the epilogue buffer picked from the tiles executed. Simulated against a plain Rust GEMM (bit-exact on small integers) for fewer clusters than tiles, swizzle 1 and 2, rastering along M and N, clusters of 1x1, 2x1, 1x2 and 2x2, partial tiles and random scheduler seeds with the race detector on; resetting the consumer states per tile is reported as a phase error. Assembled with `ptxas -arch=sm_90a` where installed; no SM90 GPU has run it. |
| [`hopper_fmha`](examples/hopper_fmha.rs) | CuTe DSL's `hopper/kernel/attention/fmha.py` (`HopperFusedMultiHeadAttentionForward`) and `helpers/fmha_helpers.py` (`FmhaStaticTileScheduler`, `MaskEnum`, `FusedMask`): the SM90 warp-specialised attention forward, Float16 or BFloat16 Q/K/V with Float32 accumulators, head dimension 64 or 128, grouped-query heads. A load warpgroup (`setmaxregister_decrease`) feeds Q (2 stages) and alternating K/V tiles (5 stages) through two `PipelineTmaAsync` rings with TMA; two MMA warpgroups (`setmaxregister_increase`) run `Q K^T` from shared-memory descriptors, an online softmax on the accumulators (`exp2`, running max and sum, row reductions by 4-lane shuffles) and `P V` with `P` converted in registers as the register-A WGMMA operand, ping-ponged by a `PipelineOrder`, with the V stage's wait taken from a `try_wait` token; the epilogue normalises O and the log-sum-exp, stores O with `stmatrix` and TMA stores on `PipelineTmaStore` and per-warpgroup named barriers. Residual, causal and sliding-window masks (including the end-aligned inference form), whole and partial tiles, static and persistent scheduling. Simulated against an f64 Rust attention (partial tiles, GQA, batches, every mask, random schedules with the race detector) and assembled with `ptxas -arch=sm_90a` for wgmma (descriptor and register-A forms), `try_wait`, `setmaxnreg` and TMA; no SM90 GPU has run it. Not ported: FP8 and the example's `try_wait`-with-time-limit mbarrier wait (the library's pipeline waits are used); the backward-pass masks. Differences: the non-persistent epilogue adds a warpgroup barrier before reusing Q's shared memory (the simulator's race detector asks for it), and `lse` is a `(batch, heads, seqlen_q)` buffer. |

```bash
cargo run --release --example tensorop_gemm
```

On a Mac with Homebrew LLVM the output is:

```text
tensorop_gemm 256x256x256: correct
blocks:              4
warp instructions:   36960
mma.sync:            8192
barriers:            32
global read/written: 524288 / 262144 bytes
global efficiency:   100% (24576 sectors, 24576 ideal)
smem bank conflicts: 1.00x (8192 wavefronts, 8192 ideal)

without the swizzle, ldmatrix and shared-memory traffic is 4.00x serialised by bank conflicts (vs 1.00x)

PTX: 1018 lines; 64 mma.sync, 24 ldmatrix, 24 cp.async per K-tile iteration
```

## CuTe DSL → cute-rs

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cute.make_layout(shape, stride=...)` | `make_layout(shape, stride)`, `Layout::col_major(shape)`, `Layout::row_major(shape)` |
| `cute.make_layout((cutlass.Int32(m), 8), stride=(8, 1))`: a static `int` or a dynamic `Int32` leaf, printed `?` | `make_layout((m, 8), (8, 1))` with `m` an `Int32`: each leaf a `usize`, an `Int32` or an `IndexExpr`. `Display` prints `(?,8):(8,1)`, `{:?}` the expressions; `"(?,?):(?,1)".parse::<Layout>()` makes each `?` a fresh placeholder (for the algebra, not for kernels) |
| `cute.is_static(x)` | `x.is_static()` on layouts, shapes, coordinates, strides (a basis element counts when its scale is static) and `IndexExpr`s; `shape.as_static()` gives an `IntTuple` |
| `cute.ScaledBasis(value, mode)`, `cute.E(mode)`, `scale * E(i)` | `ScaledBasis::new(value, mode)`, `E(mode)` (a `StrideLeaf`; `E([])` is the integer 1), `2 * E(0)`, `m * E(1)` with `m` an `Int32`, `E(0) * n`. A stride leaf is a `StrideLeaf`, `Int(IndexExpr)` or `Basis(ScaledBasis)`; a mode is an index or a path, outermost first (`E([1, 0])`), printed innermost first (`1@0@1`), and `"(2,3):(1@0,1@1)".parse::<Layout>()` reads it. As in CuTe DSL, scaling a basis element whose scale is a `Ratio` panics |
| `cute.crd2idx(coord, layout)` of a layout with basis strides | `layout.eval(&coord)`, the arithmetic tuple as a `Coord`: at `(2,3)`, `(4,8):(2@0,1@1)` gives `(4,3)`. `layout.slice_and_offset(&coord)` slices with a tuple offset. `crd2idx`, `Layout::offset` and `Layout::slice` return offsets, so they refuse basis strides, as do swizzles, `upcast` (the recast of a layout) and `make_layout_like`. Composition (either side), coalesce, the divides (a tiler of basis strides picks the modes it tiles: `(8,8):(9,1) / 4:1@0` is `(4,(2,8)):(9,(36,1))`), the products, `complement` (per leaf of the coprofile; `layout.complement_no_cotarget()` takes none, as cutegen's one-argument `complement` does), `right_inverse` and `left_inverse` (from tuples back to indices: `(4,5):(1@0,1@1)` gives `(4,5):(1,4)`) and `cosize` (`size(coshape)`) take basis strides |
| `cute.Ratio(n, d)`, `is_integral`, `reduced`, `*`, `to(Ratio / float / int)` | `Ratio::new(n, d)`, `is_integral()`, `reduced()`, `*` by a `Ratio` or an `i64`, `to_f64()`, `to_int()` (rounding down). A basis scale is a `BasisScale`, `Int(IndexExpr)` or `Ratio(Ratio)` |
| `cute.basis_value(e)`, `cute.basis_get(e, t)` | `basis_value(&e)`, a `BasisScale`; `basis_get(&e, &t)` for a tuple or a layout |
| `cute.make_identity_layout(shape)` | `make_identity_layout(shape)`: `shape:make_basis_like(shape)`, so `(2,(3,4))` gets `(1@0,(1@0@1,1@1@1))`, and an integer shape `4:1`; extents may be runtime values |
| `cute.has_scaled_basis(x)`, `cute.has_underscore(x)` | `x.has_scaled_basis()` on strides and layouts; `x.has_underscore()` on a `SliceCoord` or a `Tiler` (a `..` leaf) |
| `cute.is_integer(x)`, `is_int_tuple(x)`, `is_valid_leaf(x)` | no function: the Rust types answer them when the kernel is compiled. An integer is a `usize`, `Int32` or `IndexExpr`, an int tuple an `IntTuple`, `Shape` or `Coord`, and the valid leaves of each tree are its leaf type (`IndexExpr` for shapes, `StrideLeaf` for strides, `Option<IndexExpr>` for slice coordinates) |
| `cute.size(x, mode=[...])`, `cute.cosize(x, mode=[...])`, `cute.shape(x, mode=...)` | `size(&x, &[..])` and `cosize(&x, &[..])`, an `IndexExpr` that is static when it folds (a `for_range` takes it as its trip count: `size(&tiles, &[2])` is the number of tiles in a rest mode), and `shape(&x, &[..])`, the sub-shape there. `x` is a tuple, a layout, a swizzled layout or a tensor (for `cosize`, a layout, a swizzled layout, whose cosize is its outer layout's, as in CuTe (`make_smem_tensor` still rounds its allocation up to whole swizzle periods), or a tensor, whose cosize is its layout's, as in CuTe: for an identity tensor, CuTe's `size(coshape)`, the product over its modes of one past the largest coordinate), and the path is CuTe's hierarchical mode. For a loop unrolled in Rust, `layout.size()`, `layout.cosize()`, `shape.size()` and `tensor.size()` are `usize`s that panic, naming the leaf and the `IndexExpr` form, when the answer is only known at run time |
| `cute.rank(x, mode=[...])`, `cute.depth(x)` | `rank(&x, &[..])` and `depth(&x)`, on the same `x` as `size` (and the methods `x.rank()` and `x.depth()`): `usize`s, static for runtime extents too |
| `cute.crd2idx(coord, layout)`, `cute.idx2crd(idx, shape)` | `crd2idx(coord, &layout)`, an `IndexExpr`, and `idx2crd(idx, &shape)`, a `Coord`, dividing by runtime extents where needed (`layout.crd2idx(&c)` and `layout.idx2crd(i)` for static answers) |
| `cute.composition`, `complement`, `coalesce`, `right_inverse`, `left_inverse` | same names. With runtime leaves: `composition` takes the left-hand side's dynamic extents to be large and divisible, and computes the extents with `min` and `ceil_div` at run time where a right-hand runtime value meets an extent it is not provably a multiple of, as C++ CuTe does; `coalesce` also merges what it can prove (`(m,4):(1,m)` to `(m*4):1`, where CuTe keeps the modes); `right_inverse` skips runtime strides and `left_inverse` takes runtime extents, C++ CuTe's rules where the CuTe dialect refuses dynamic layouts |
| `cute.logical_divide`, `zipped_divide`, `tiled_divide`, `flat_divide` | same names, on layouts and tensors (a register fragment's extents stay static); a runtime extent divided by a static tile gives a rest mode of extent `ceil_div(extent, tile)`, a tiler of the whole of a layout splits the rest over the layout's modes (C++ CuTe's complement in `shape(coalesce(layout))`), and a tiler with a runtime stride must have a single mode, as in CuTe |
| `cute.logical_product`, `blocked_product`, `raked_product` | same names; the block is static and the tiler may have runtime extents and strides, as in CuTe. `logical_product` also takes a tile, mode by mode as CuTe DSL's does: an integer `n` is `n:1` (an `Int32` too), `..` keeps the mode, and a tuple applies to the matching modes (`logical_product(&block, (n, ..))`) |
| `cute.tile_to_shape(atom, trg_shape, order)` | `tile_to_shape_ordered(&atom, shape, &order)`, `tile_to_shape_swizzled_ordered` for a swizzled atom (`tile_to_shape`, `tile_to_shape_swizzled` place the copies column-major); a target the atom does not divide is rounded up, as in the CuTe dialect |
| `cute.nvgpu.OperandMajorMode.K` / `.MN`, `OutputMajorMode.M` / `.N` (`warpgroup.OperandMajorMode` is a deprecated alias) | `OperandMajorMode::K`, `OperandMajorMode::MN`, `OutputMajorMode::M`, `OutputMajorMode::N` (in `cute::nvgpu`, re-exported at the top and as `warpgroup::OperandMajorMode`); `"mn".parse()` reads them as CuTe DSL's enums read strings. The per-operation rules (MN-major only for 16-bit WGMMA operands in shared memory, and the tcgen05 ones) belong to the MMA operations, which come with m8-wgmma-core and m10 |
| `warpgroup.SmemLayoutAtomKind.{MN,K}_{INTER,SW32,SW64,SW128}`, `warpgroup.make_smem_layout_atom(kind, element_type)` | `warpgroup::SmemLayoutAtomKind::K_SW128` (etc.), `warpgroup::make_smem_layout_atom::<F16>(kind)` or `make_smem_layout_atom_dtype(kind, DType::F16)`: C++ CuTe's `GMMA::Layout_*_Atom`, 8 rows of 128 to 1024 contiguous bits, `(c,8):(1,c)` (MN) or `(8,c):(c,1)` (K). CuTe's swizzle `Swizzle<B,4,3>` acts on bytes and cute-rs's composed layouts swizzle element offsets, so the atom carries the same swizzle restated for the element (`kind.byte_swizzle().to_elements(bits)`): `S<3,3,3> o 0 o (8,64):(64,1)` for f16 `K_SW128`, `S<3,4,3>` for 8-bit types, `S<3,5,3>` for 4-bit ones; 6-bit floats and Booleans are refused. Tiled to `(128, 64)` the K-major SW128 atom is the PTX ISA's canonical WGMMA layout with a 1024-byte stride byte offset (checked element by element in `tests/sm90_layouts.rs`) |
| `cutlass.utils.LayoutEnum.ROW_MAJOR` / `COL_MAJOR`, `mma_major_mode()`, `sm90_mma_major_mode()`, `is_k_major_a()` (and `is_m_major_a`, `is_n_major_b`, `is_k_major_b`, `is_n_major_c`, `is_m_major_c`), `LayoutEnum.from_tensor(t)` | `LayoutEnum::ROW_MAJOR` (in `cute::utils`, re-exported at the top), the same methods, `LayoutEnum::from_tensor(&t)`, `from_layout(&layout)` and `from_leading_dim(..)`: the first entry of the leading dimension, 0 column-major, 1 row-major; anything else panics (CuTe DSL's `ValueError`) |
| `cutlass.utils.sm90` (`hopper_helpers`): `get_smem_layout_atom(layout, element_type, major_mode_size)`, `make_smem_layout_a(a_layout, mma_tiler_mnk, a_dtype, num_stages)`, `make_smem_layout_b(...)`, `make_smem_layout_epi(epi_dtype, epi_layout, epi_tile, epi_stage, smem_trg_shape=None, smem_order=None)` | `cute::utils::sm90::get_smem_layout_atom(LayoutEnum::ROW_MAJOR, DType::F16, 64)` (SW128, SW64, SW32 or INTER for a major extent that is a multiple of 1024, 512 or 256 bits; panics unless the extent is a multiple of 8), `make_smem_layout_a(LayoutEnum::ROW_MAJOR, (128, 128, 64), DType::F16, 4)`, `make_smem_layout_b`, `make_smem_layout_epi(DType::F16, layout, (m, n), stages, None, None)`: staged `ComposedLayout`s (f16 K-major A of `dense_gemm.py`'s default tile: `S<3,3,3> o 0 o ((8,16),(64,1),(1,4)):((64,512),(1,0),(0,8192))`) for `make_smem_tensor` and a struct's `MemRange::get_tensor_composed`. `get_smem_store_op` comes with m9's epilogue helpers |
| `PersistentTileSchedulerParams(problem_shape_ntile_mnl, cluster_shape_mnk, swizzle_size=1, raster_along_m=True)`, `get_grid_shape(max_active_clusters)` (the 4.8.1 examples' helper; the wheel's copy is deprecated) | `utils::PersistentTileSchedulerParams::new(shape, [cm, cn, 1], PersistentTileSchedulerOptions { swizzle_size, raster_along_m })` (static or runtime tile counts); in the kernel, `param(name, cluster, options)` declares the tile counts and, unswizzled, two host-prepared `FastDivmod` divisors, which the host passes with `kernel_args()`; `get_grid_shape(n)` on the host. A cluster deeper than 1 along K and a swizzle size of 0 are refused. `with_fallback_cluster_shape_mnk(shape)` is the wheel's `fallback_cluster_shape_mnk=` (m10-port-clc-clusters): a scheduler made from it may be created in either arm of `mixed_cluster_arms`, and the fallback arm keeps the preferred shape's work decomposition (deviation: CuTe DSL's fallback clone strides by fallback clusters from the same work index, which revisits tiles in a grid sized for preferred clusters, so its tutorial passes the preferred params to both arms) |
| `StaticPersistentTileScheduler.create(params, block_idx, grid_dim)`, `initial_work_tile_info`, `get_current_work`, `advance_to_next_work(advance_count=1)`, `num_tiles_executed`; `WorkTileInfo.tile_idx`, `is_valid_tile` | `utils::StaticPersistentTileScheduler::create(&params, block_idx(), grid_dim())`, the same methods (`advance_to_next_work_by(n)` for a count); `Carry` with only the work index and tile count. Unswizzled schedules decode with `FastDivmod` (M major when rastering along M), swizzled ones with `get_flat_coord` of `swizzle_and_rasterize`'s layout (shared with m10's CLC scheduler); padding tiles of a swizzle are valid work, as in CuTe DSL. `WorkTileInfo` (`tile_idx() -> [Int32; 3]`, `is_valid_tile()`) is `Carry`. `examples/persistent_gemm.rs` runs on it |
| `StaticPersistentRuntimeTileScheduler.create(params, block_idx, grid_dim, inner_mode=1)` | `utils::StaticPersistentRuntimeTileScheduler::create(&params, block_idx(), grid_dim(), inner_mode)`: every tile valid, decoded with `get_hier_coord` on `(INT32_MAX, n):(n, 1)` (N fastest) or `(m, INT32_MAX):(1, m)`, tile `(m, n, 0)`; an inner mode other than 0 and 1 is refused |
| `StaticPersistentGroupTileScheduler.create(params, block_idx, grid_dim, cluster_tile_shape_mnk, initial_search_state, group_count, problem_shape_mnkl, use_cached_problem_shapes=False)`, `prefetch_problem_shapes`, `get_current_work`, `delinearize_z`; `GroupedWorkTileInfo`, `GroupSearchResult`, `GroupedGemmGroupSearchState`, `create_initial_search_state()`; the deprecated `GroupedGemmTileSchedulerHelper(group_count, params, cluster_tile_shape_mnk, search_state)` (`delinearize_z`, `search_cluster_tile_count_k`) | `utils::StaticPersistentGroupTileScheduler::create(&params, block_idx(), grid_dim(), tile, create_initial_search_state(), group_count, &shapes, cached)` (the examples' helper always caches; the wheel's copy switches), `get_current_work(&mut self)` and `delinearize_z` update the search state. The search is warp-collective (scan by `shuffle_sync_up`, `popc` of `vote_ballot_sync`, `shuffle_sync` broadcasts, a 32-group window): every lane of the warp must call it. Work tiles (11 values), search states (4), results (7) and the scheduler are `Carry`; `#[deprecated] GroupedGemmTileSchedulerHelper` wraps the same search and stops past the last group |
| `cutlass.utils.mixed_input_helpers`: `ContiguousGGSearchState`, `create_initial_contiguous_group_search_state()`, `contiguous_group_search(cluster_tile_shape_mnk, group_count, linear_idx, search_state, cumsum, search_mode)`, `ContiguousGroupWorkTileInfo`, `make_contiguous_group_work_tile_info(group_count, sTile_info)` | `utils::` the same names: the search walks `cumsum` (`group_count + 1` boundaries from 0, so `cur_group_idx` is the group plus one) from the carried state; the work tile reads a 4-entry shared-memory record with `autovec_copy`, `is_valid_tile()` being `group_idx < group_count`. Both `Carry` |
| `cute.nullspace(layout)` | `nullspace(&layout)`, `layout.nullspace()`, with CuTe DSL's strides: the product of the extents before each mode (C++ CuTe gives size-1 modes stride 0). For an integer shape with stride 0, where CuTe DSL raises, C++'s result: `nullspace(1:0)` is `1:0`, `nullspace(4:0)` is `4:1` |
| `cute.is_congruent(a, b)`, `is_weakly_congruent(a, b)` | the same names, on tuples, layouts, swizzled layouts and tensors |
| `cute.leading_dim(shape, stride)`, `is_major(mode, stride)` | the same names, and `layout.leading_dim()`, `layout.is_major(mode)`; a position is a path such as `vec![1, 0]` |
| `cute.find(t, x)`, `find_if(t, pred_fn)` | `t.find(x)`, `t.find_if(\|leaf, path\| ..)`, returning the path; `find_mode`, `find_mode_if` for `hierarchical=False` |
| `cute.wrap(x)`, `unwrap(x)`, `front(x)`, `get(x, mode=[i, j])` | `x.wrap()`, `x.unwrap()`, `x.front()`, `x.get(&[i, j])` on any `Tree` |
| `cute.increment_coord(coord, shape)` | `increment_coord(coord, shape)` |
| `cute.make_layout_tv(thr, val)` | `make_layout_tv(&thr, &val)` |
| `cute.make_swizzle(b, m, s)`, `make_composed_layout(inner, offset, outer)` | `make_swizzle(b, m, s)` (a negative shift too, `\|s\| >= b`; `make_swizzle(0, m, s)` is the identity `S<0,4,3>`, as in CuTe DSL; printed `S<b,m,s>`) and `make_composed_layout(inner, offset, outer)`, CuTe's general `ComposedLayout`, `c -> inner(offset + outer(c))`, with a `Swizzle` or a `Layout` inner (a swizzle over an outer layout of basis strides is refused). `inner()`, `offset()`, `outer()`, `is_normal()` (the identity swizzle and offset 0: `as_layout()` decays it to the outer layout), `max_alignment()`, `eval`/`offset_of`/`call` and `crd2idx`. The algebra acts on the outer layout and keeps the rest: `composition`, the divides and products, `tile_to_shape(_ordered)`, `coalesce`, `get`, `select`, `group_modes`, `size`, `shape`, `rank`, `cosize` (the outer layout's, as in CuTe; a layout inner's is refused); `slice` moves the slice's offset into the composed offset (the swizzle does not distribute over `+`), and `slice_and_offset` decays a slice the swizzle acts on affinely to its layout, as cutegen's does. `Layout`, `ComposedLayout` and `Tensor` share it through `Tileable` (`LayoutLike`). Composed layouts and swizzles print and parse as CuTe's (`"S<3,4,3> o 0 o (8,64):(64,1)".parse::<ComposedLayout>()`), and `basis_get` takes them. A tensor holds the general composed layout: `make_smem_tensor` and `make_tensor` take one with a layout inner too (`tensor.inner_layout()`, `tensor.composed_layout()`); copies move its elements one at a time |
| `cute.get_swizzle_portion(layout)`, `get_nonswizzle_portion(layout)` | the same names: a composed layout's swizzle (the identity `S<0,4,3>` for a layout or a layout inner) and its outer layout (a layout itself) |
| `tensor[None, 0]` | `tensor.slice((.., 0))` |
| `tensor[i, j]`, `tensor[i, j] = v` | `tensor.load((i, j))`, `tensor.store((i, j), v)` |
| `cute.local_tile(t, tiler, coord, proj=...)` | `local_tile`, `local_tile_proj`, also on tensors of runtime extents: a tile has the tiler's static extents, and a rest mode kept with `..` has the number of tiles, `ceil_div(m, BM)`, as its extent |
| `cute.local_partition` | `local_partition`, `local_partition_proj` (also on tensors of runtime extents) |
| `cute.make_fragment`, `make_fragment_like(tensor, dtype)` (`make_rmem_tensor_like`) | `make_fragment`, `make_fragment_like::<T, _>(&tensor)`: the layout is `make_layout_like` of the tensor's (a row-major tile gets a row-major fragment, so `autovec_copy` between them stays wide); a coordinate tensor gives a compact column-major one. MMA fragments (`partition_fragment_a/b/c`) and `to` stay compact column-major |
| `cute.make_fragment_like(layout)`, `make_fragment_like(layout, dtype)` (a layout or a composed layout) | `make_fragment_like_layout(&layout)` (C++ `make_fragment_like`: mode 0 compact column-major keeping its zeros, the other modes ordered by their strides), `make_fragment_like_dtype::<T>(&layout)` for the register fragment |
| `cutlass.AddressSpace` (`cute.AddressSpace`): `generic`, `gmem`, `smem`, `cmem`, `rmem`, `tmem`, `dsmem` | `AddressSpace::Generic`, `Gmem`, ..., with NVVM's numbers (`AddressSpace::Smem as u32 == 3`); `From<MemSpace>` and `TryFrom<AddressSpace> for MemSpace`. Nothing builds `Generic`, `Cmem`, `Tmem` or `Dsmem` yet |
| `tensor.memspace`, `ptr.memspace`, `tensor.stride`, `tensor.element_type`, `tensor.dtype`, `tensor.iterator`, `ptr.max_alignment`, `cute.leading_dim(tensor.shape, tensor.stride)` | `tensor.memspace()` (register fragments and `make_rmem_tensor` arrays are `Rmem`; a coordinate tensor panics, as CuTe DSL raises), `ptr.memspace()`, `tensor.stride()` (panics for a swizzled tensor; `tensor.layout().stride()` is the stride before the swizzle), `element_type()`, `dtype()`, `tensor.iterator()` (a `TensorIterator`: `Ptr`, the `Coord` origin of a coordinate tensor, or `Registers`), `ptr.max_alignment()` (bytes: the buffer's alignment and what divides the offset), `tensor.leading_dim()` |
| `cute.make_identity_tensor(shape)` | `make_identity_tensor(shape)`: the coordinate tensor of origin 0 over `make_identity_layout(shape)`, of any rank and nesting, with runtime extents too (`make_identity_tensor(m_a.shape())`, as CuTe DSL's `make_identity_tensor(mA.shape)`); `t.coord(c)` is the hierarchical coordinate, congruent with the shape, whatever tiles and partitions were taken of it |
| `cute.make_tensor(origin, make_layout(shape, stride=(1*E(0), 1*E(1))))`, a coordinate tensor | `make_tensor(origin, layout)` (an integer, an `IndexExpr`, a `Coord` or a tuple origin) or `make_coord_tensor(origin, layout)`: an arithmetic-tuple origin (an integer or a tuple, runtime and negative leaves allowed) plus the tuple the layout gives; slicing, the divides, `local_tile`, `local_partition`, `partition_s/d`, `make_predicate` and `copy_if` take it as they take memory tensors, and `t.origin()` is the origin |
| `cute.domain_offset(coord, tensor)` | `domain_offset(coord, &tensor)`: a memory tensor's base offset grows by `crd2idx(coord, layout)` (its swizzle kept), a coordinate tensor's origin by the tuple `coord` maps to. The shift may be negative, the K residue `IndexExpr::signed(k - bk * k_tiles)` of the Ampere SGEMM (a coordinate tensor takes every runtime leaf of a shift or origin as signed, so a plain `Int32` residue works there too); a swizzled tensor takes only provably non-negative shifts, and a register fragment only static ones. The simulator reports a predicated-on access before the shifted start, and does not check a predicated-off `cp.async` (zero-fill) there |
| `cute.elem_less(lhs, rhs)` | `elem_less(lhs, rhs)`, a `Boolean`, on any mix of coordinates, shapes and integer tuples of static or runtime leaves (`elem_less(c, t.shape())`); the two must be congruent, and are compared leaf by leaf, signed where a leaf may be negative (`elem_less((m, -1), (M, k))`). A leaf that is known when the kernel is built folds, and so does a result whose every leaf folds |
| `cute.ceil_div(input, tiler)`, `cute.shape_div(a, b)`, `cute.round_up(a, b)` | the same names, on integers and tuples of static or runtime leaves (`ceil_div((m, n), (128, 128))` sizes a grid at run time), with the CuTe dialect's tuple / tuple, tuple / integer and integer / tuple rules. `ceil_div` with a `Layout` or a `Tiler` of layouts gives the rest mode of `zipped_divide`. `shape_div` checks the divisibility of static leaves only, as C++ CuTe does |
| `layout == other` (`Layout.__eq__`), `cute.equal` | `==` on layouts is equality as built (two runtime leaves are equal only as the same expression); `equal(&a, &b)` is a `Boolean` for layouts, shapes and strides, folding different profiles and static leaves, and otherwise comparing the leaves at run time (so `m * 2` and `m + m` are equal there). Swizzled layouts compare only when static |
| `layout.get_hier_coord(idx)`, `layout.get_flat_coord(idx)` | the same names, C++ CuTe's `idx2crd(idx, shape, stride)`: the coordinate that maps to the offset `idx`, hierarchical or one per top-level mode, for runtime indices, extents and strides; meaningful only for compact layouts, as in CuTe |
| `cute.make_copy_atom(CopyUniversalOp(), T, num_bits_per_copy=128)` | `make_copy_atom::<T>(CopyOp::Universal { bits: 128 })` |
| `cpasync.CopyG2SOp`, `warp.LdMatrix8x8x16bOp(transpose, 4)` | `CopyOp::CpAsync { .. }`, `CopyOp::LdMatrix { num: 4, trans }` |
| `warp.LdMatrix8x16x8bOp(num_matrices=4)` (no `unpack_bits`) | `CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None }` on 8-bit elements (FP8, Int8, Uint8): `ldmatrix.m8n8.b16` on a 16-bit view of the bytes, each lane supplying one 16-byte row and receiving 4 consecutive bytes of each matrix, so `make_tiled_copy_a/b` of an `MmaFP8Op` fill its K-major fragments. Non-transposed only |
| `warp.LdMatrix8x16x8bOp(num_matrices, unpack_bits=4\|6)` | `CopyOp::LdMatrix8x16x8b { num, unpack_bits: Some(4 \| 6) }`: `ldmatrix.m8n16.x{1,2,4}.shared.b8x16.b4x16_p64` / `.b6x16_p32`, each 16-byte row holding 16 4-bit (6-bit) values in its low 8 (12) bytes and padding (TMA's `16U4_ALIGN16B` / `16U6_ALIGN16B`), each value arriving in an 8-bit container (4 bits in bits 2 to 5, 6 bits in bits 0 to 5); thread-value layouts of CuTe C++'s `SM100_SU{4,6}_DU8x16_x{1,2,4}_LDSM_N`. sm_100a, sm_101a/sm_110a and sm_120a (PTX 8.6; the f families from 8.8), as `nvvm.ldmatrix` (LLVM 22, or inline PTX with `CompileOptions::mma_inline_ptx(true)`) |
| `warp.LdMatrix16x16x8bOp(num_matrices=1\|2, transpose=True, unpack_bits)` | `CopyOp::LdMatrix16x16x8b { num, trans: true, unpack_bits }`: `ldmatrix.m16n16.x{1,2}.trans.shared.b8` (or `.b8x16.b4x16_p64` / `.b6x16_p32`), with the PTX ISA's fragment of the transposed 16x16 matrix as the thread-value layout; `trans: false` is refused, as in CuTe DSL. Targets as above |
| `warp.LdMatrix16x8x8bOp(num_matrices=2\|4, transpose=True, unpack_bits)` | `CopyOp::LdMatrix16x8x8b { num, trans: true, unpack_bits }`: `ldmatrix.m16n16.x{num/2}.trans` with the address and byte permutations and thread-value layouts of CuTe C++'s `SM100_U8x8_LDSM_T` / `SM100_U8x16_LDSM_T` (laid out like `stmatrix.m16n8.trans`). Targets as above |
| `cute.make_tiled_copy(atom, layout_tv, tiler_mn)` | `make_tiled_copy(&atom, &layout_tv, tiler_mn)` |
| `cute.make_tiled_copy_tv`, `make_tiled_copy_A/B/C`, `make_tiled_copy_S/D`, `make_tiled_copy_C_atom` | `make_tiled_copy_tv`, `make_tiled_copy_a/b/c`, `make_tiled_copy_s/d`, `make_tiled_copy_c_atom` |
| `thr_copy.partition_S/D`, `tiled_copy.retile(t)` | `partition_s`, `partition_d` (also of tensors of runtime extents: the modes after a thread's values count the copy's tiles), `tiled_copy.retile(&t)` (or `thr_copy.retile_s/retile_d`) |
| `atom.value_type`, `thr_id`, `layout_src_tv`, `layout_dst_tv`; `tiled_copy.layout_tv_tiled`, `tiler_mn`, `layout_src_tv_tiled`, `layout_dst_tv_tiled`, `size` | the same names, as methods |
| `cute.copy(atom, src, dst, pred=p)`, `cute.autovec_copy` | `copy`, `copy_if`, `autovec_copy`, unrolled when the kernel is built: a copy whose per-thread chunk or rest mode has a runtime extent is refused, naming the leaf (slice one tile per iteration of a `for_range` instead). `autovec_copy` is CuTe DSL's: the vector is `max_common_layout(src, dst)`, bounded by both layouts' `max_alignment`, both pointers' alignment (a swizzle's run counted) and 256 bits, then proved aligned at each vector's offset (halved until it is), else scalar. Boolean tensors copy one element at a time (MLIR's `i1` vectors are bit-packed in memory, where a Boolean is a byte) |
| `cute.autovec_copy(src, dst, l1c_evict_priority=)` | `autovec_copy_with(&src, &dst, AutovecOptions { l1c_evict_priority })`: a `CopyOp::G2R` or `CopyOp::R2G` of the same width carrying it; other pairs of memory spaces refuse a priority (cute-rs has no `CopyS2ROp`/`CopyR2SOp`) |
| 32-byte (256-bit) vectors, `ld.global.v8.b32` (sm_100, PTX ISA 8.8) | `autovec_copy` (and `CopyOp::Universal { bits: 256 }`, `load_vector::<f32, 8>`) between registers and global memory of 32- or 64-bit elements whose buffer is 32-byte aligned: `param_tensor_aligned(name, layout, 32)` (checked by the launcher; the simulator checks each access). Compiled for sm_100 and newer the PTX ISA version is raised to 8.8; LLVM splits them into two 16-byte accesses for older targets |
| `cute.basic_copy(src, dst)`, `cute.basic_copy_if(pred, src, dst)` | the same names: unrolled when the shapes are static (the sizes must match), otherwise a runtime loop over `size(dst)` (with a branch per element for `basic_copy_if`), which a register fragment cannot take part in (use `make_rmem_tensor`). `basic_copy_if` copies between element types of the same width (`bool` as 8 bits) |
| signed index terms (`Int32(-1)` in a coordinate) | `IndexExpr::signed(v)` for a runtime `Int32` or `Int64` of either sign, and negative `i32`/`i64` constants (`IndexExpr::from(-1)`); `e.is_non_negative()`, `e.eval_signed(env)`. They add and multiply with wrapping arithmetic and compare signed |
| `cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), T, num_bits_per_copy=, memory_order=, memory_scope=, l2_prefetch_size=, l1c_evict_priority=, load_cache_mode=, shared_space=, invariant=)`, and `CopyR2GOp()` with `store_cache_mode=` | `make_copy_atom::<T>(CopyOp::G2R { bits, attrs: G2RAttrs { memory_order: MemoryOrder::Acquire, memory_scope: MemoryScope::Gpu, ..Default::default() } })`, `CopyOp::R2G { bits, attrs: R2GAttrs { .. } }`. `bits: 0` (`num_bits_per_copy=0`) uses the widest vector each copy can prove aligned. The register side is a fragment or a `make_rmem_tensor` |
| `cute.nvgpu.MemoryOrder`, `MemoryScope`, `L2PrefetchSize`, `CacheEvictionPriority`, `LoadCacheMode`, `StoreCacheMode`, `SharedSpace` | the same names (`MemoryOrder::Acquire`, `CacheEvictionPriority::NoAllocate`, `L2PrefetchSize::Size64B`; `L2PrefetchSize.NONE` is `None`) |
| `cute.copy(atom, src, dst, pred=p, cache_policy=policy)` | `copy_with(&atom, &src, &dst, CopyOptions { pred: Some(&p), cache_policy: Some(policy) })` |
| `cute.make_cotiled_copy(atom, atom_layout_tv, data_layout)` | `make_cotiled_copy(&atom, &atom_layout_tv, &data_layout)` |
| `cute.nvgpu.warp.MmaF16BF16Op(ab_dtype, acc_dtype, shape_mnk)`, `MmaTF32Op(shape_mnk)` | `MmaF16BF16Op::new(ab_dtype, acc_dtype, shape_mnk)`, `MmaTF32Op::new(shape_mnk)` (or `MmaOp::Sm80(MmaKind::..)`) |
| `cute.nvgpu.warp.MmaFP8Op(ab_dtype, acc_dtype, shape_mnk)` | `MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 32))`: Float8E4M3FN or Float8E5M2 A and B, Float16 or Float32 accumulators, m16n8k16 or m16n8k32; `mma.sync...e4m3.e4m3` on sm_89, at PTX 8.4 (m16n8k32 with f32) or 8.7 (the rest), as `nvvm.mma.sync`, which needs the `mlir-opt` of LLVM 23 (`CompileOptions::mma_inline_ptx(true)` emits inline PTX instead) |
| `cute.nvgpu.warp.MmaF16BF16SparseOp(ab_dtype, acc_dtype, shape_mnk, SparseMetadataFormat.TID)`, `cute.gemm(atom, d, [a, e], b, c)` | `MmaF16BF16SparseOp::new(ab, acc, shape_mnk, SparseMetadataFormat::Tid)`, `gemm_sparse(&mma, &d, &a, &e, &b, &c)`: A is the compressed `(M, K/2)` tile (`partition_a`, `partition_fragment_a`), `e` a Uint32 metadata fragment (`partition_fragment_e`, one register per MMA, filled by the kernel; `sparse_compress` and `sparse_metadata_lane` build it on the host). `mma.sp::ordered_metadata.sync` with sparsity selector 0 (sm_80, PTX 8.5), as `nvvm.mma.sp.sync` (LLVM 22) |
| `cute.make_mma_atom(op)`, `cute.make_tiled_mma(op_or_atom, atom_layout)` | `make_mma_atom(op)`, `make_tiled_mma(op_or_atom, atom_layout)` |
| `thr_mma.partition_A/B/C`, `make_fragment_A/B/C` | `partition_a/b/c`, `partition_fragment_a/b/c` |
| `cute.gemm(mma, d, a, b, c)` | `gemm(&mma, &d, &a, &b, &c)` |
| `cutlass.Int8` to `Int64`, `Uint8` to `Uint64`, `Float16`, `BFloat16`, `Float32`, `Float64`, `Boolean` | the same names |
| `cutlass.Int4`, `Int128`, `Uint128` | `Int4` (host `I4`, -8 to 7), `Int128`, `Uint128` (host `i128`, `u128`). Int128 and Uint128 have arithmetic (division and remainder too, which LLVM expands inline), shifts, comparisons, `min`/`max`, integer conversions, 16-byte loads and stores, 128-bit scalar parameters (`KernelArg::Scalar128`) and buffers; the simulator keeps their high words in a table of their own. Int4 lives in a sign-extended `i8` in MLIR (a 4-bit pattern in the simulator), converts to and from every type (out-of-range sources wrap at 4 bits; floats through Int32) and its arithmetic wraps at 4 bits (CuTe DSL keeps `Int4 + Int4` an Int4) |
| `cutlass.TFloat32` | `TFloat32` (host `TF32`), stored as f32 bits: `x.to::<TF32>()` is `cvt.rna.tf32.f32`, back to Float32 is free, arithmetic on it is computed in Float32 (with a Float32 result), and `MmaTF32Op` fragments may hold it (the same registers as the Float32 path) |
| `cutlass.Float8E4M3FN`, `Float8E5M2` | `Float8E4M3FN`, `Float8E5M2` (host `F8E4M3FN`, `F8E5M2`, with `from_f32`, `to_f32` and `MAX`, `MIN`, `NAN`), stored as bytes (`i8` in MLIR). `to()` into them is `cvt.rn.satfinite.{e4m3x2,e5m2x2}` (from Float16 its f16x2 form, from the other types through Float32) and so saturates: E4M3FN to ±448, E5M2 to ±57344 instead of infinity; out of them, `cvt.rn.f16x2.{e4m3x2,e5m2x2}` (exact). sm_89 (PTX 8.1, which it raises sm_89 kernels to) or sm_90 and newer (PTX 7.8). They are storage types: no arithmetic of their own; comparisons convert to Float32. The pair forms are `cvt_f32x2_to_f8x2::<F>([a, b], relu)`, `cvt_f16x2_to_f8x2` and `cvt_f8x2_to_f16x2` |
| `cutlass.Float8E4M3`, `Float8E4M3B11FNUZ` | `Float8E4M3`, `Float8E4M3B11FNUZ` (host types with exact codecs): kernels read them and convert from them (integer bit manipulation, as PTX has no `cvt`); a conversion into them is refused, as CuTe DSL refuses it |
| `cutlass.Float8E8M0FNU`, `cutlass.FloatNV8E5M3FNU` | `Float8E8M0FNU` (UE8M0) and `FloatNV8E5M3FNU` (UE5M3), with host codecs. `to::<F8E8M0FNU>()` rounds up and saturates (as `cvt.rp.satfinite.ue8m0x2.f32`, with integer operations, so on every target); `cvt_f32x2_to_ue8m0x2`, `cvt_bf16x2_to_ue8m0x2` and `cvt_ue8m0x2_to_bf16x2` are the sm_100a/sm_101a/sm_120a (or family) `cvt`s. UE5M3 is a storage type here (its `cvt` needs PTX 9.4). CuTe DSL's UE4M3 scale factors are Float8E4M3FN values |
| `cutlass.Float4E2M1FN`, `Float6E2M3FN`, `Float6E3M2FN` | `Float4E2M1FN`, `Float6E2M3FN`, `Float6E3M2FN` (host `F4E2M1FN`, `F6E2M3FN`, `F6E3M2FN`, the low bits of a byte, with exact codecs: FP4 magnitudes 0 to 6, FP6 up to 7.5 and 28, no infinity or NaN). Storage types in sub-byte tensors (FP4 two per byte, FP6 four per three bytes, so FP6 vectors are 16 values, 96 bits); in registers the low bits of an `i8`. `to::<f32>()` (and the other types) from them decodes with integer operations on every target; `to()` into them is refused, as CuTe DSL refuses it: convert pairs with `cvt_f32x2_to_f4e2m1x2([a, b], relu)` and `cvt_f32x2_to_f6x2::<F6E2M3FN>([a, b], relu)` (`cvt.rn.satfinite{.relu}.{e2m1x2,e2m3x2,e3m2x2}.f32`: saturating, NaN to `+MAX_NORM`), and back with `cvt_f6x2_to_f16x2(src, relu)`; sm_100a, sm_101a and sm_120a (PTX 8.6) or their families (PTX 8.8) |
| `cutlass.Float4E2M1FNx2`, `Float6E2M3FNx4`, `Float6E3M2FNx4`, `_element_precision_width` | `Float4E2M1FNx2` (host `F4E2M1FNx2`, a byte), `Float6E2M3FNx4` and `Float6E3M2FNx4` (host `F6E2M3FNx4`, `F6E3M2FNx4`, three bytes in a `u32`), with `new(lanes)` and `lanes()`; tensors of them count packed units (FP4x2 a byte, read and written as one; FP6x4 in groups of four units, 96 bits). `recast_tensor` turns them into the lane type and back (FP4x2 to twice as many FP4 values, FP6x4 to four times as many FP6, in memory and in registers). `DType::element_precision_width`, `packed_lanes`; they are refused as `to()` operands |
| `cute.arch.cvt_f4e2m1_f16`, `cvt_f4e2m1x2_to_f16x2`, `cvt_f4e2m1x4_to_f16x4`, `cvt_f4e2m1x8_to_f16x8` | the same names, on arrays of `Float4E2M1FN` (`cvt_f4e2m1x2_to_f16x2_with(src, relu)` for `.relu`): `cvt.rn.f16x2.e2m1x2` per pair, lane 0 from the low nibble into the low half; sm_100a, sm_101a and sm_120a. (`cvt_f4e2m1_f16_intrinsic`, in CuTe DSL's `__all__` but not defined in 4.8.1, is left out) |
| the tensor helpers `is_fp8_dtype` and (a DLPack workaround) `create_cute_tensor_for_fp8` | `dtype.is_fp8()`; FP8 buffers are bytes, so `KernelArg::fp8_from_f32::<F8E4M3FN>(&xs)` (host round to nearest even, saturating) and `arg.to_f32()` read and write them |
| type metadata: `T.width`, `T.bytes`, `T.n_bytes(n)`, `T.signed`, `T.min`, `T.max`, `T.zero`, `T.inf`, `T.nan`, `T.exponent_width`, `T.mantissa_width`, `T.is_float`, `T.is_integer`, `T.recast_width(w)`, `cutlass.dtype("Float16")` | `T::WIDTH`, `T::BYTES`, `T::SIGNED`, `T::IS_FLOAT`, `T::IS_INTEGER` (consts, usable in `128 / T::WIDTH`), and on `DType`: `bits`, `bytes`, `n_bytes`, `min_value`, `max_value`, `zero`, `inf`, `nan` (as literals; `None` where the type has none), `exponent_width`, `mantissa_width`, `recast_width`, `cute_name` and `DType::from_cute_name("Float16")`, all read from one table row per type (`DType::info`). Host `F16` and `BF16` have `MIN`, `MAX`, `INFINITY`, `NAN`, `ZERO`, `ONE` |
| `cutlass.Boolean(x)` | `x.to::<bool>()`: `x != 0` for integers, an unordered `x != 0.0` for floats (NaN is true); `Boolean + Boolean` is computed in Int32 and converted back, as CuTe DSL 4.8.1 does; `-b` does not compile |
| implicit promotion of two values (`a + b`, `a < b` with different types) | the same rules (`Promote`, `DType::promote`): an integer meeting a float becomes that float at the wider width, floats widen (Float64 > Float32 > TFloat32, Float16 > BFloat16 at equal width), mixed-signedness integers go unsigned when the unsigned type is at least as wide; pairs CuTe DSL refuses (two FP8 formats, FP8 with an 8-bit integer) do not compile. Rust literals still adopt the other operand's type (`x_f16 + 1.0` stays Float16; CuTe DSL gives Float32); a TFloat32 result is a Float32; integer `/` and `%` truncate (CuTe DSL's integer `/` is `true_div(a, b)`, a Float32) |
| `cutlass.range` (runtime loop) / `range_constexpr` | `for_range`, `for_range_carry` / a Rust `for` loop |
| `cutlass.range(start, stop, step, unroll=K)` | `for_range_with(start, stop, step, LoopOptions::unroll(K), body)` (and `for_range_carry_with`): LLVM unroll metadata, `count = K`; 1 keeps the loop rolled |
| `cutlass.range(..., unroll_full=True)` | `LoopOptions::unroll_full()` (needs a trip count LLVM can compute; giving `unroll` too is an error, where CuTe DSL lets `unroll_full` win) |
| `cutlass.range(..., vectorize=True)` | `LoopOptions { vectorize: true, ..LoopOptions::default() }` (sm_100 and newer, checked when compiling) |
| `cutlass.range(..., at_least_once=True)` | `LoopOptions { at_least_once: true, ..LoopOptions::default() }`: an `llvm.intr.assume` that `start < stop`, which the simulator checks |
| `cutlass.range(start, stop, step, prefetch_stages=S)` | `for_range_pipelined(start, stop, step, Prefetch::stages(S), (p0, c0), produce, consume)`: a helper, not a loop attribute. CuTe DSL only tags the loop and NVIDIA's closed compiler rotates canonical producer/consumer bodies; upstream LLVM has no such pass, so cute-rs takes the producer and the consumer as closures and emits the prologue, steady state and drain itself (runtime trip counts too). `Prefetch::bounded_by(num_stages)` rejects `S >= num_stages` (a self-deadlock when one thread produces and consumes) |
| `cutlass.while_generate` | `while_carry` |
| dynamic `if` / `cutlass.const_expr` | `if_`, `if_else`, `if_else_yield` / a Rust `if` |
| variables assigned inside a dynamic `if`/`for`/`while` (carried automatically by the preprocessor) | `let v = Var::new(init)`, `v.get()`, `v.set(x)`, `v.update(f)`: carried through loops and returned from branches automatically |
| a dataclass carried through runtime control flow or held in such a variable | `carry_struct!` (declares the struct, or implements `Carry` for an existing one) |
| `cute.arch.thread_idx()`, `sync_threads()`, `sync_warp()`, `cp_async_wait_group(n)` | the same names |
| `cute.arch.sync_warp(mask)` | `sync_warp_with(mask)` (`bar.warp.sync`; every lane of the mask must execute it) |
| `cute.arch.barrier(barrier_id=i, number_of_threads=n)`, `barrier_arrive(barrier_id=i, number_of_threads=n)` | `barrier(i, n)`, `barrier_arrive(i, n)` |
| `cute.arch.barrier(barrier_id=i)` (every thread of the CTA), `barrier_arrive(..., aligned=False)` | `pipeline::sync(i)`, `barrier_arrive_unaligned(i, n)` |
| `cutlass.pipeline.NamedBarrier(barrier_id, num_threads)` and its methods; `pipeline.arrive`, `arrive_unaligned`, `wait`, `wait_unaligned`, `arrive_and_wait`, `sync` | `pipeline::NamedBarrier::new(id, n)` (each a constant or a runtime `Int32`), the same methods; `pipeline::arrive(id, n)` and so on |
| `cutlass.pipeline.PipelineState(stages, count, index, phase)`, `PipelineUserType`, `make_pipeline_state(type, stages)`; `clone`, `advance(count=1)`, `reverse`, `reset_count` | `pipeline::PipelineState::new(..)`, `make_pipeline_state(PipelineUserType::Producer, n)` (producers and `ProducerConsumer` at phase 1, as CuTe DSL's code does); `advance()` (selects, not a branch), `advance_by(n)` with a static count of at least 1 or an `Int32`, `reverse()`, `reset_count()`. `Copy` and `Carry` (count, index, phase) |
| `PerStagePipelineState(num_stages)`, `next`, `mark_valid`, `mark_invalid`, `reset_count`; `compute_k_tile` | `pipeline::PerStagePipelineState::new(n)` (phases and counts in per-thread arrays, only the index carried), the same methods; `compute_k_tile(idx, count, offset, n)`. Both states implement `StageState` (`index()`, `phase()`), which the pipelines take |
| `Agent`, `CooperativeGroup(agent, size, alignment=None)` | `pipeline::Agent`, `CooperativeGroup::new(agent, size)` (a constant or an `Int32`; the deprecated `alignment` is left out), `thread_arrive_count()`: static for threads, warps and declared blocks |
| `MbarrierArray(barrier_storage, num_stages, (op, group), tx_count, mbarrier_layout, name)`, `PipelineOp`, `MbarrierLayout`; `arrive(index, dst, cta_group)`, `wait`, `try_wait`, `test_wait`, `arrive_and_wait`, `get_barrier`, `max`, `recast_to_new_op_type` | `pipeline::MbarrierArray::new(&storage, n, (PipelineOp::TmaLoad, group), tx)` (`with_options` for the layout and name): initialised from warp 0's elected lane; `arrive` dispatches on the op (`AsyncThread`, with `dst` a peer CTA; `TmaLoad`; `ClcLoad`; `AsyncLoad`; `TCGen05Mma` panics until m10's `tcgen05.commit`); `arrive_and_wait` arrives then waits (CuTe DSL 4.8.1's calls the named-barrier functions) |
| `SyncObject` | `pipeline::SyncObject`, a trait with associated argument types, for `MbarrierArray` and `NamedBarrier` |
| `pipeline_init_arrive(cluster_shape_mn, is_relaxed)`, `pipeline_init_wait(cluster_shape_mn)`, `agent_sync(group, is_relaxed)` | the same names in `pipeline`, with `Option<Shape>` for the cluster shape (static) |
| `alloc_reserved_mbarrier(num_stages, multiplier=2)` | `pipeline::alloc_reserved_mbarrier(n, multiplier)`: 8-byte words in the reserved partition |
| `cutlass.pipeline.PipelineAsync.create(*, num_stages, producer_group, consumer_group, barrier_storage=None, producer_mask=None, consumer_mask=None, defer_sync=False, name='')`; `producer_acquire(state, try_acquire_token=None)`, `producer_try_acquire`, `producer_commit`, `consumer_wait(state, try_wait_token=None)`, `consumer_try_wait`, `consumer_release`, `producer_tail`, `producer_get_barrier`, `consumer_get_barrier`, `make_producer`, `make_consumer`, `make_participants` | `pipeline::PipelineAsync::create(n, producer_group, consumer_group, PipelineAsyncOptions { barrier_storage, producer_mask, consumer_mask, defer_sync, name })`: full barriers at the storage (`alloc_reserved_mbarrier(n, 2)` without one), empty ones `n` further on; unless `defer_sync`, `fence.mbarrier_init` (not below sm_90) and a CTA barrier. The methods are the `pipeline::Pipeline` trait's, shared with the specialised pipelines, taking any `StageState` and an `Option<Boolean>` token (`None` waits; `Some(t)` waits under `!t`; a constant folds); acquire options are an associated type (`()` here); the barrier accessors are the `PipelineBarriers` trait's. `producer_tail(&mut state)` advances the state, as CuTe DSL's does. Below sm_90 the waits spin on `mbarrier.test_wait` and the probes are one `test_wait`, so it runs on the SM89 GPU |
| `PipelineProducer`: `clone`, `reset`, `current_handle`, `acquire(token, **kwargs)`, `advance(count=1)`, `acquire_and_advance(token)`, `try_acquire`, `commit(handle=None)`, `tail`; the handle's `index`, `count`, `barrier`, `commit()` | `pipeline::PipelineProducer<'p, P: Pipeline>` (`Copy`, `Carry`: the state): the same methods, `acquire_with(token, options)` for the keyword arguments, `advance(n)` with a static count or an `Int32`; `ProducerHandle` with `index()`, `count()`, `barrier()` (for pipelines with barriers), `commit()`. `commit(Some(&handle))` checks at build time that the handle came from the same pipeline object and panics otherwise. CuTe DSL kernels mutate participants inside `if warp_idx == 0:`; in cute-rs keep each participant inside its branch and loop, or return it with `if_else_yield` |
| `PipelineConsumer`: `clone`, `reset`, `current_handle`, `wait(token)`, `advance`, `wait_and_advance(token)`, `try_wait`, `release(handle=None)`; the handle's `release()` | `pipeline::PipelineConsumer<'p, P>` and `ConsumerHandle`, the mirror of the producer's |
| `cutlass.pipeline.PipelineEmpty` (`create(*args, **kwargs)`, no-op methods, no `get_barrier`) | `pipeline::PipelineEmpty`, a unit struct implementing `Pipeline` with no barriers: the probes return a constant true, `producer_tail` does nothing, and it has one notional stage so that participants can be made from it. It does not implement `PipelineBarriers`, so asking it for a barrier does not compile. One kernel body generic over `P: Pipeline` serves both |
| `cutlass.pipeline.PipelineCpAsync.create(*, barrier_storage, num_stages, producer_group, consumer_group, producer_mask=None, consumer_mask=None, defer_sync=False, name='')` | `pipeline::PipelineCpAsync::create(n, producer_group, consumer_group, PipelineCpAsyncOptions { .. })` (the options are `PipelineAsyncOptions`; `barrier_storage: None` allocates reserved storage). Built on `PipelineAsync` with `PipelineOp::AsyncLoad` full barriers: a commit is `cp.async.mbarrier.arrive.noinc`, so the full count is the producer group's threads, and each producer thread issues its stage's `cp.async`s, then commits exactly once. Implements `Pipeline` and `PipelineBarriers`; runs on the SM89 GPU |
| `cutlass.pipeline.PipelineOrder.create(*, depth, length, group_id, producer_group, barrier_storage=None, defer_sync=False, name='')`; `wait(state=None)`, `arrive(state=None)`, `get_barrier_for_current_stage_idx(group_id, state=None)` | `pipeline::PipelineOrder::create(depth, length, group_id, producer_group, PipelineOrderOptions { barrier_storage, defer_sync, name })` (`group_id` a constant or a runtime `Int32`): `wait()`, `arrive()` (advances its own state), `wait_with(&state)`, `arrive_with(state) -> PipelineState` (the advanced state), `get_barrier_for_current_stage_idx(group_id, Option<&state>)`. `Carry` (its state). Without storage it reserves `alloc_reserved_mbarrier(depth * length, 2)`, twice what it uses, as CuTe DSL's default does. Runs on the SM89 GPU |
| `cutlass.memory.SmemPartition`, `SmemAllocator()`; `allocate(bytes \| Numeric \| struct, byte_alignment, partition)`, `allocate_array`, `allocate_tensor(..., swizzle=)`, `calculate_partition_size(partition, cumulative)`, `calculate_total_usage()` | `memory::SmemPartition::{Reserved, User}`, `memory::SmemAllocator::new()`: `allocate_bytes`, `allocate::<T>`, `allocate_struct`, `allocate_array::<T>`, `allocate_tensor::<T>(layout, align, swizzle, partition)`; reserved buffers are laid out first. The size queries return an `Int32` set when the build ends, so later allocations count (`memory::calculate_partition_size`, `calculate_total_usage`) |
| `get_smem_capacity_in_bytes(cc)` (`SMEM_CAPACITY_MAP`), `get_kernel_smem_size(kernel)` | `memory::get_smem_capacity_in_bytes(Option<Arch>)` (`None`: the declared target), `Arch::smem_capacity_bytes()`, `memory::get_kernel_smem_size(&kernel)`; `check_target` refuses a kernel whose shared memory exceeds its target's capacity |
| `cute.arch.warp_idx()`, `make_warp_uniform`, `warpsize`, `WARP_SIZE`, `WARPS_PER_WARPGROUP`, `THREADS_PER_WARPGROUP`, `FULL_MASK` | the same names |
| `cute.arch.shuffle_sync`, `shuffle_sync_up`, `shuffle_sync_down`, `shuffle_sync_bfly` (value, offset) | the same names: the whole warp and every lane (`mask_and_clamp` 31, or 0 for up) |
| `cute.arch.shuffle_sync_op(value, offset, mask, mask_and_clamp, kind)`, and the partials with `mask=` and `mask_and_clamp=` | `shuffle_sync_op(value, offset, mask, mask_and_clamp, ShuffleKind::Idx)`, `shuffle_sync_with(value, offset, mask, mask_and_clamp)` (and `shuffle_sync_up_with`, `_down_with`, `_bfly_with`): segments of `width` lanes are `((32 - width) << 8) \| (width - 1)`, or `(32 - width) << 8` for up. Every Numeric moves: 8- and 16-bit values as their bits zero-extended to 32 (so a NaN keeps its payload; CuTe DSL converts Float16 and BFloat16 through Float32, which returns the canonical NaN), 64-bit values as two shuffles |
| `cute.arch.warp_reduction(val, op, threads_in_group=32)`, `warp_reduction_max`, `warp_reduction_sum` | `warp_reduction(val, op, threads_in_group)`, `warp_reduction_max(val, threads_in_group)`, `warp_reduction_sum(val, threads_in_group)`: a butterfly unrolled at build time. `max` is CuTe DSL's: `fmax` with its default modes on Float32 (a NaN is ignored), and on Float16, BFloat16 and Float64 `cutlass.max`, which is `arith.maximumf` there (IEEE 754 `maximum`: a NaN in the group gives NaN, here the quiet NaN of `math::max_with`'s `propagate_nan`) |
| `cute.arch.warp_redux_sync(value, kind, mask_and_clamp=FULL_MASK, abs=, nan=)` | `warp_redux_sync(value, ReduxKind::Add)`, `warp_redux_sync_with(value, kind, mask, ReduxOptions { abs, nan })`: Int32 and Uint32 (`max`/`min` are `umax`/`umin` on a Uint32) from sm_80; Float32 `fmax`/`fmin` on sm_100a or the sm_100f family, as `nvvm.redux.sync` with the mask in a register |
| `cute.arch.match_sync(mask, value, kind="any")` | `match_sync(mask, value, MatchKind::Any)` (or `MatchKind::All`) on Int32, Uint32, Int64 and Uint64; `match.all`'s predicate is the result being nonzero |
| `cute.arch.fmax(a, b, abs=, nan=, ftz=)`, `fmin` | `fmax(a, b)`, `fmax_with(a, b, MinMaxOptions { abs, nan, ftz })`, `fmin`, `fmin_with`: `abs` is `max.xorsign.abs.f32` (sm_86 and PTX 7.2), `nan` is `max.NaN.f32` (sm_80); the defaults are `Value::max` and `Value::min` |
| `cute.arch.add_packed_f16x2(a, b, rnd=, ftz=, sat=, predicate=)`, `sub_packed_f16x2`, `add_packed_bf16x2(a, b, rnd=, predicate=)`, `sub_packed_bf16x2` | the same names, with `HalfOptions { rn, ftz, sat, predicate, .. }`: `add_packed_f16x2(a, b, HalfOptions { sat: true, ..HalfOptions::default() })`. A pair is an `Int32` holding its bits (element 0 low) or a 2-tuple of values (`Half2`), and the result has its form. The bf16x2 forms need sm_90 and PTX 7.8 |
| `cute.arch.fma_packed_f16x2(a, b, c, rnd=, ftz=, sat=, relu=, oob=)`, `fma_packed_bf16x2(a, b, c, rnd=, relu=, oob=)`, `fma_f16`, `fma_bf16` | the same names, with `HalfOptions { ftz, sat, relu, oob, .. }` and CuTe DSL's checks (`ftz` and `sat` are Float16 only, `sat` and `relu` exclude each other, `oob` excludes both): `fma.rn{.ftz}{.sat}.f16x2`, `fma.rn{.relu}.bf16` (sm_80), `fma.rn.oob{.relu}.f16x2` (sm_90, PTX 8.1) |
| `cute.arch.exp2_packed_f16x2(a, predicate=)`, `exp2_packed_bf16x2` | `exp2_packed_f16x2(a, predicate)`, with `predicate` an `Option<Boolean>`: `ex2.approx.f16x2` (sm_75) and `ex2.approx.ftz.bf16x2` (sm_90, PTX 7.8) |
| `cute.arch.cvt_f32_bf16`, `cvt_f32x2_bf16x2(src_vec2)` | the same names; `cvt_f32x2_bf16x2([x, y])` is `cvt.rn.satfinite.bf16x2.f32` (PTX 8.1, which raises sm_80's and sm_89's version) and returns an `Int32` or a 2-tuple, as the caller asks |
| `cute.arch.cvt_f32_tf32` | `cvt_f32_tf32(x)`, the TF32 bits as an `Int32` (`cvt.rna.tf32.f32`); `x.to_tf32()` gives them as a Float32 and `x.to::<TF32>()` as a TFloat32, either for `MmaTF32Op` fragments |
| `cute.recast_layout` (new width, old width, layout) | `recast_layout(new_bits, old_bits, &layout)` on a `Layout` or a `ComposedLayout` (its inner, offset and outer layout recast together, as cutegen's `upcast` and `downcast`) (C++ CuTe's semantics: identity, upcast with `ceil_div`, downcast, or both through the gcd; the swizzle's base moves by `log2` of the ratio, its bits clamped), with `Layout::upcast_by(n)` and `Layout::downcast(n)` (`Layout::upcast` keeps the CUTLASS 3.x form tiled copies use) |
| `cute.recast_tensor(src, dtype)` | `recast_tensor::<U, _>(&src)` or `src.recast::<U>()`: a memory tensor keeps its storage, with its pointer cast (`Ptr::cast`, which counts bits for sub-byte types) and its layout recast (Boolean counts as 8 bits); every static non-zero stride must be a multiple of the widening ratio and the unit-stride mode's extent too, or the build panics. A register fragment is repacked (`Op::Pack`/`Op::Unpack`, element 0 in the low bits; equal widths are a bitcast). `swizzle_` (reserved in CuTe DSL) has no counterpart |
| sub-byte tensors (`cutlass.Int4` in memory) | `param_tensor::<I4>`, `make_smem_tensor::<I4>`, `alloc_smem::<I4>`, `make_rmem_tensor::<I4>`, `make_tensor` over a `Ptr<I4>`: byte buffers (`ceil(n * bits / 8)` bytes; shared memory in whole bytes) whose offsets count Int4 values, packed with element 0 in the low bits. They are read and written in vectors of a multiple of 32 bits (at most 128): `copy` (`Universal`, `G2R`, `R2G`, `cp.async`), `copy_if`, `autovec_copy` and `load_vector`/`store_vector` load or store 32-bit words and unpack or pack them; a scalar `load`/`store` is refused when the kernel is built, as CuTe DSL refuses dereferencing. Register fragments hold one value per register. `KernelArg::buffer(&[I4])` packs and `to_vec::<I4>()` unpacks |
| `cute.arch.cvt_i8x4_to_f32x4(src_vec4)`, `cvt_i8x2_to_f32x2` | the same names, on an `[Int8; 4]` (`[Int8; 2]`) or the values packed into an Int32 or Uint32 (Int16 or Uint16), returning `[Float32; 4]` (`[Float32; 2]`): `dp4a.s32.s32` and `cvt.rn.f32.s32` |
| `cute.arch.cvt_i8_bf16`, `cvt_i8x2_to_bf16x2`, `cvt_i8x4_to_bf16x4`, `cvt_i8_bf16_intrinsic(vec_i8, length)` | the same names (`cvt_i8_bf16_intrinsic(&values, length)` takes a slice); the intrinsic takes CuTe DSL's Ampere path on every target, and `cvt_i8_bf16_intrinsic_with(&values, length, I8ToBf16Path::S2f6x2)` its sm_100a-class path (PTX 9.1). `cvt_i8_bf16` needs sm_90, `cvt_i8x2_to_bf16x2` and `cvt_i8x4_to_bf16x4` an sm_100a-class target |
| `cute.arch.cvt_i4_bf16`, `cvt_i4x2_to_bf16x2`, `cvt_i4x4_to_bf16x4`, `cvt_i4x8_to_bf16x8(src, with_shuffle=)`, `cvt_i4_bf16_intrinsic(vec_i4, length, with_shuffle=)` | the same names, on arrays (a slice for the intrinsic) of `Int4` and a `with_shuffle: bool`: by default `lop3` (LUT `0x6A`), `prmt` (`0xF4F0` to `0xF7F3`) and `fma.rn.bf16x2` (`0x83808380`, `0xC308C308`), exact for every int4 value; with `with_shuffle` (input in the order `(0, 2, 1, 3, 4, 6, 5, 7)`) the scaled `.s2f6x2` cvt with scale `0x8181`, sm_100a-class targets and PTX 9.1. CuTe DSL allows the intrinsic only on sm_100a-class targets; cute-rs allows the default path on every target (it needs only sm_80), and the SM89 GPU runs it |
| `cute.arch.cvt_i4x8_to_mxf8x8(src_vec8, dst_type=)`, `cvt_i4_mxf8_intrinsic(vec_i4, length, dst_type)` | `cvt_i4x8_to_mxf8x8::<F>(src)`, `cvt_i4_mxf8_intrinsic::<F>(&values, length)` (`length` a multiple of 8), `F` any 8-bit type (typically `F8E4M3FN`, or `i8` for raw bytes): three `prmt` per four values over E4M3 tables of 0..7 and -8..-1, exact; every target |
| `cute.arch.sext_unpacked_i4x4_to_i8x4(src_vec4)`, `sext_unpacked_i4_i8_intrinsic(vec_unpacked_i4, length)` | the same names, on an `[Int8; 4]` or its word as an Int32 or Uint32 (the result has the argument's form), and on a slice |
| a bitcast of a vector of 16- or 8-bit values to a 32-bit register (`llvm.bitcast` in CuTe DSL's helpers) | `pack([lo, hi])` and `let [lo, hi]: [Float16; 2] = unpack(bits)`, between any widths that add up |
| `cute.arch.vote_*_sync`, `elect_one`, `activemask`, `lanemask_*`, `clock`, `clock64`, `globaltimer`, `globaltimer_lo` | the same names |
| `cute.printf(fmt, *args, end="\n")`, `cute.printf(*values)` | `printf(fmt, &[a.into(), ...])` appends nothing (one `vprintf` call, as before: write the newline in the format); `printf_end(fmt, args, end)` takes CuTe DSL's `end` (its default is `printf_end(fmt, args, "\n")`); `printf_values(&[..])` is the format-less form (`{}, {}, ...` and a newline). Operands are `PrintArg`s: numbers (`Value<T>`, `AnyValue`, Rust literals: `bool`, `i32`, `f32`, `f64`, ...), pointers (`%p` of the address; shared ones are their 32-bit shared-window address, zero-extended), and as text with their runtime leaves filled in, layouts, composed layouts, integer tuples, shapes, strides (scaled-basis leaves `1@0`), coordinates (`_` for a free mode), tiles and tensors (printed as `print_tensor` prints them: header and data). `{}` resolves at build time to `%d`/`%u`/`%lld`/`%llu`/`%f`/`%p` by the operand's type, and mixes with C conversions (flags `- + space 0 #`, width, precision, `hh h l ll`, `d i u o x X c e E f F g G p %%`). Operands are widened in the IR as C's variadic promotion widens them (Boolean, 8- and 16-bit integers to 32 bits, narrow floats to Float64; Float32 is left to `gpu.printf`'s lowering). Where CuTe DSL leaves a conversion that does not fit its operand undefined, cute-rs refuses it at the call's line, as it refuses a count mismatch, `%s`, a lone `%` and a swizzled-layout (PISL) tensor; more than 32 operands are split over several calls. The simulator formats as the driver does (glibc's `printf` on the host: `%p` of 0 is `(nil)`); `tests/gpu.rs` compares the two on an SM89 |
| `cute.arch.popc`, `clz`, `brev`, `bfind`, `log2_of_pow2_int` | the same names (`clz` returns a Uint32, as in CuTe DSL; `bfind` returns a Uint32 for every operand type, see Limitations; the methods `x.popc()`, `clz()`, `ctz()` and `brev()` keep the operand's type) |
| `cute.arch.bfe(value, start, length)`, `bfi(replacement, value, start, length)` | the same names and operand order (the replacement first, as in PTX: `bfi`'s docstring example swaps them) |
| `cute.arch.lop3(a, b, c, lut)` | `lop3(a, b, c, lut)` with a `u8` table, which `lop3_lut(\|a, b, c\| a ^ (b & c))` computes from a closure |
| `cute.arch.shf(a, b, shift, kind="l")` (`"r"`, `"clamp_left"`, `"clamp_right"`) | `shf(a, b, shift, ShfKind::Left)` (`Right`, `ClampLeft`, `ClampRight`); `a` is the low word, as in PTX |
| `cute.arch.prmt(src, src_reg_shifted, prmt_indices)` | `prmt`; the other PTX modes are a cute-rs extension: `prmt_with(a, b, c, PrmtMode::F4e)` |
| `cute.arch.mul_hi`, `mul_wide`, `mul24(a, b, hi=False)`, `mad24(a, b, c, hi=False)`, `add_sat_int`, `sub_sat_int` | the same names, with `hi` explicit: `mul24(a, b, false)` (`mul_hi` is also a method) |
| `cute.arch.add_cc`, `addc`, `sub_cc`, `subc`, `mad_cc`, `madc` | the same names (`add.cc.u32`, `addc.u64`, `mad.lo.cc.s32`, ...). CuTe DSL's `addc`, `subc` and `madc` read the carry flag without writing it, so its chains stop at two words; the carry-in-and-out forms are cute-rs extensions: `addc_cc`, `subc_cc`, `madc_cc` (`addc.cc.u32`, ...), and `add_with_carry(a, b, cin)` and `sub_with_borrow(a, b, bin)` take and return the carry as a value |
| `cute.arch.inline_ptx(ptx_code, write_only_types=[T], read_only_args=[x, ptr], read_write_args=[y], predicate=p)` | `InlinePtx::new(ptx_code).write::<T>().read(x).read_ptr(&ptr).read_write(y).predicate(p).emit()`, the same `{$wN}`, `{$rN}` and `{$rwN}` names, with CuTe DSL's operand classes: a read-only integer constant is an immediate and a shared-memory pointer a 32-bit shared address (a Float32 or Boolean constant stays a register). It returns the outputs (write-only and read-write, in the order declared): a value for one, a tuple for several. CuTe DSL returns the write-only outputs alone and drops the read-write operands' final values, so returning those is a cute-rs extension, as are `.sim(f)`, which gives the simulator a Rust equivalent, and `.pure()`, which drops the side effects |
| `@cutlass.dsl_user_op` | a `#[track_caller]` Rust function: the operations it builds record its caller's source line ([custom operations](src/arch/inline.rs)) |
| `llvm.inline_asm(result_type, operands, asm, constraints, has_side_effects=)` inside a `dsl_user_op` | `InlineAsm::new(asm, constraints).output::<T>().arg(x).arg_ptr(&ptr).side_effects().memory_clobber().emit()`, with GCC's `+r` read-write operands translated to an output plus a tied input; a pointer is passed as its address (`l`, or `r` for shared memory) |
| raw `nvvm.*`, `llvm.*`, `arith.*` builders inside a `dsl_user_op` | `MlirOp::new("nvvm.read.ptx.sreg.nsmid : i32").result::<i32>().emit()`: one MLIR operation written as text, `%0, %1, ...` for its `.operand(x)`s and `.operand_ptr(&ptr)`s (an `!llvm.ptr<1>`, `!llvm.ptr<3>` or generic `!llvm.ptr`) |
| `cute.arch.atomic_add`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_max`, `atomic_min`, `atomic_exch`, `atomic_cas(ptr, cmp=, val=)` with `sem=` and `scope=` | the same names on a `Ptr` (relaxed at GPU scope, PTX's and CuTe DSL's default), and `atomic_add_with(&ptr, v, AtomicOptions::new(MemoryOrder::AcqRel, MemoryScope::Sys))` and the other `_with` forms; `atomic_cas(&ptr, cmp, val)`. `atomic_add` takes 32- and 64-bit integers, Float32, Float64, Float16 and BFloat16 (sm_90); a two-element f16/bf16 vector is `atomic_add_packed(&ptr, pair)` on a `Half2` pair. Each is one PTX `atom` with explicit semantics and scope |
| `cute.arch.atomic_fmax`, `atomic_fmin(ptr, val, sign_bit=, sem=, scope=)`, the deprecated `atomic_max_float32(ptr, value, positive_only=True)` | `atomic_fmax(&ptr, v)`, `atomic_fmax_with(&ptr, v, sign_bit, opts)`, `atomic_fmin`, `atomic_fmin_with`, `atomic_max_float32(&ptr, v, positive_only)` (deprecated): integer atomics on the bits, chosen at run time or by `sign_bit` |
| `cute.arch.red(ptr, val, op=, dtype=, sem=, scope=)` | `red(&ptr, v, RedOp::Add, AtomicType::U32, opts)` (`dtype` may be a `DType`: `DType::I32` is `s32`), `red_packed(&ptr, pair, RedOp::Add, opts)` for `f16x2` and `bf16x2`; `Umin` and `Umax` make a signed dtype unsigned, as in CuTe DSL, and an add on `s64` is the same add on `u64` |
| `fence_acq_rel_*` | the same names |
| `cute.arch.load(ptr, dtype, sem="acquire", scope="gpu", level1_eviction_priority=, cop=, ss=, level_prefetch_size=)`; a vector `dtype` | `load(&ptr, LoadOptions { sem: MemoryOrder::Acquire, scope: Some(MemoryScope::Gpu), ..Default::default() })` (the strings are enums: `cop: Some(CacheOperator::Cg)`, `level1_eviction_priority: Some(CacheEvictionPriority::EvictLast)`, `level_prefetch_size: Some(L2PrefetchSize::Size128B)`; `scope: None` is `.sys`); `load_vector::<T, N>` |
| `cute.arch.store(ptr, val, level1_eviction_priority=, cop=, ss=, sem="release", scope=)` | `store(&ptr, val, StoreOptions { sem: MemoryOrder::Release, .. })`, `store_vector` |
| `cute.arch.prefetch(addr, cache_level="L2", evict_priority=, predicate=, uniform=)` | `prefetch(&ptr, PrefetchOptions { cache_level: Some(PrefetchCacheLevel::L2), evict_priority, predicate, uniform })` |
| `cute.arch.mbarrier_init`, `mbarrier_arrive`, `mbarrier_test_wait` | the same names (SM80 and newer) |
| `cute.arch.mbarrier_init_fence`, `mbarrier_expect_tx`, `mbarrier_arrive_and_expect_tx`, `mbarrier_try_wait`, `mbarrier_conditional_try_wait`, `mbarrier_wait`, `cp_async_bulk` | the same names (SM90 and newer) |
| `cute.arch.fence_proxy(kind, space=, use_intrinsic=)` | `fence_proxy(FenceProxyKind::Alias)` (sm_70, PTX 7.5; runs on the SM89 GPU), `fence_proxy_with(FenceProxyKind::AsyncShared, FenceProxyOptions { space: Some(SharedSpace::Cta), .. })`; `Async`, `AsyncGlobal` (sm_90). `AsyncShared` needs a space and the others take none; `Tensormap` and `Generic` are refused, naming `fence_tma_desc_acquire`/`fence_tma_desc_release` (m7-tensormap-update). `use_intrinsic` is accepted and ignored: the fence is inline PTX with a memory clobber, as NVVM's operation lowers differently between LLVM releases |
| `cute.arch.fence_view_async_shared()` | `fence_view_async_shared()`: `fence.proxy.async.shared::cta`, after generic shared-memory writes that a bulk copy or TMA store will read |
| `cute.arch.cp_async_bulk_commit_group()`, `cp_async_bulk_wait_group(group, read=)` | `cp_async_bulk_commit_group()`, `cp_async_bulk_wait_group(group, read)` (SM90): per-thread bulk async-groups; `group` must be known when the kernel is built (a runtime count is refused) |
| `cpasync.CopyBulkS2GOp()`, `make_copy_atom(op, dtype, num_bits_per_copy=)`, `cpasync.group_bulk_copy_modes(src, dst)` | `make_copy_atom::<T>(cpasync::CopyBulkS2GOp)` (the whole of mode 0 per copy) or `CopyBulkS2GOp.num_bits_per_copy(bits)` (`CopyOp::BulkS2G { bits }`), `copy(&atom, &smem, &gmem)` on one elected thread, `cpasync::group_bulk_copy_modes(&src, &dst)`: `cp.async.bulk.global.shared::cta.bulk_group` (SM90). The experimental primitive `cp_async_bulk_global_shared_cta(dst, src, size)` is `cp_async_bulk_global_shared_cta(&dst, &src, bytes)`. Sizes are multiples of 16 bytes at 16-byte-aligned addresses: constant sizes and static offsets are checked when the kernel is built, runtime ones by the simulator. `cache_policy=` adds `.L2::cache_hint` (m7-tma-store) |
| `mbarrier_arrive(ptr, peer_cta_rank_in_cluster, arrive_count, relaxed=, scope=)`, and the same keywords of the expect-tx operations | `mbarrier_arrive_with(&ptr, count, MbarrierOptions { peer_cta_rank_in_cluster, relaxed, scope })`, `mbarrier_arrive_and_expect_tx_with`, `mbarrier_expect_tx_with` |
| `cute.make_tensor(iterator, layout_or_shape)` | `make_tensor(iterator, layout)`: the iterator (`TensorEngine`) is a `Ptr` (by reference or value), giving a tensor of its type, or an origin, giving a coordinate tensor; the layout (`impl Into<LayoutOrShape>`) is a `Layout`, a `ComposedLayout` (a normal one decays to its outer layout) or a shape, which becomes `make_layout(shape)` (column-major, runtime extents allowed). A composed layout with a function inner is refused. Base-DSL pointers (`cutlass.Pointer` re-made at its `max_alignment`) wait for m11 and SMEM matrix descriptors for m8 |
| `make_ptr`, `recast_ptr(ptr, swizzle_, dtype)`, `cute.arch.alloc_smem` | `make_tensor`, `tensor.ptr()`, `ptr.cast::<U>()`, `alloc_smem`; `recast_ptr::<U, _>(&ptr, Some(swizzle))` gives the pointer a swizzle (on bytes, as CuTe DSL's are; `None` drops it), and `make_tensor` on a swizzled pointer gives a swizzled tensor, whose `ptr()` carries the swizzle back (`ptr.swizzle()`). The swizzle acts on absolute addresses, so the buffer must be aligned to its period (`alloc_smem(n, 1024)` for `S<3,4,3>`); a byte swizzle that would split an element, a buffer that is not aligned (global parameters are 16-byte aligned) and a swizzled pointer with a swizzled composed layout are refused when the kernel is built. `tensorop_gemm`'s `GemmConfig { ptr_swizzle: true, .. }` puts its `Swizzle<2,4,3>` (bytes) on the pointers and compiles to the PTX of the composed form, bar the buffers' alignment. `make_ptr` from an integer address is the next row; `SmemAllocator.allocate(Struct)` is a struct's `allocate` (the rows below), its other methods have no counterpart yet; `make_tensor(&ptr, Layout::row_major((m, k)))` takes runtime extents (shared memory is sized when the kernel is built, so stays static) |
| `@cute.struct class SharedStorage: ...`, `SmemAllocator().allocate(SharedStorage)`, `SharedStorage.__sizeof__()`, `__alignof__()`, `size_in_bytes()` | `cute_struct! { pub struct SharedStorage(stages: usize) { mbar: MemRange<u64>[2 * stages], #[align(1024)] staging: MemRange<f32>[stages], alpha: f32, inner: Inner(stages) } }`: a view type over one byte buffer of shared memory, its members at the next multiple of their alignment (a scalar's and a range's: the element size, at least 1; a nested struct's: its own), its alignment the largest and its size rounded up to it. CuTe DSL sizes members when the kernel is traced, which is when cute-rs builds it, so the macro's parameters (`stages`) are build-time `usize`s, and the struct may take `Numeric` type parameters (`struct Storage<AB>(n: usize)`). `SharedStorage::allocate(stages)` (`allocate_aligned(byte_alignment, stages)`: the larger alignment, at most 1024), `at(&byte_ptr, stages)`, `layout(stages)` (a `StructLayout` with `size_of()`, `align_of()`, `size_in_bytes()`, `members()`, `field_table()` of `name:size:offset`), `size_of(stages)`, `align_of(stages)`, `s.base()`. Scalar members are `ScalarMember`s: `s.alpha.ptr()`, `load()`, `store(1.5f32)` and `store_from(x)`, which converts as CuTe DSL's `s.alpha = x` does; nested structs are their own views (`s.inner.mbar`). The buffer records the field table, which `Kernel::dump` prints (`struct SharedStorage { mbar:64:0, staging:16:1024, .. }`). `StructLayout::builder(name)` builds the same layouts without the macro (`scalar::<T>(name, align)`, `mem_range::<T>(name, n, align)`, `nested(name, &layout, align)`, returning field handles whose `at(&base)` gives the views). Empty structs, duplicate members and non-power-of-two alignments are refused when the kernel is built. `tensorop_gemm`'s `GemmConfig { smem_struct: true, .. }` keeps its A and B stages in one (`GemmStorage`) |
| `cute.struct.MemRange[dtype, size]`: `data_ptr()`, `get_tensor(layout, swizzle=None, dtype=None)`, `r[i]`, `r[i] = v` | `mbar: MemRange<u64>[n]`, a `MemRange<T>` view: `data_ptr()`, `len()`, `size_in_bytes()` (a Boolean counts as a byte; sub-byte ranges must fill whole bytes), `get_tensor(layout)`, `get_tensor_composed(composed_layout)` (a swizzle on element offsets, the range at a multiple of its period), `get_tensor_swizzled(layout, swizzle)` (a swizzle on byte addresses, through a swizzled pointer: CuTe DSL's `swizzle=`; the range must start at a multiple of its period) and `get_tensor_as::<U>(layout)` (`dtype=`): CuTe DSL's refused combination, a composed layout with a swizzle, has no spelling. Each checks that the layout fits the range and refuses a size-0 range, an address marker (`MemRange<u64>[0]`, whose `data_ptr()` is still its address). `load(i)`, `store(i, v)` and `store_from(i, v)`: a static index is checked against the length when the kernel is built, a runtime one by the simulator. The members are typed views of the struct's byte buffer, which MLIR accesses through a `memref.view` per element type (mbarriers as u64, `cp.async`, `ldmatrix`, atomics, bulk copies and `ptrtoint` too) |
| `cute.struct.Align[member, alignment]` | `#[align(1024)]` before a member (scalar, range or nested struct): its offset rounds up to the larger of its natural alignment and 1024, a power of two of at most 1024 (shared memory's limit). `make_smem_tensor` keeps its 128-byte alignment; `make_smem_tensor_aligned(layout, 1024)` takes another |
| `@cute.union class U: ...` | `cute_union! { pub struct U { halves: MemRange<F16>[4096], words: MemRange<f32>[1024], #[align(16)] tag: u64 } }`: the same members and macro, every one at offset 0, the alignment the largest and the size the largest rounded up to it (8192 bytes, 16-byte aligned here). Members alias: the simulator's per-byte race checks see through them. Nested in a struct as any struct is (`u: U()`) |
| a framework tensor with a dynamic layout (`from_dlpack(...)`, `(?,?):(?,1)`) | a `DlTensor` parameter (the `from_dlpack` rows below), or `param_tensor("a", Layout::row_major((m, k)))`, with `m` and `k` from `param_extent`: a buffer sized at launch; or a `TensorSpec` with `SymInt` leaves, and a `HostTensor` marked dynamic at launch ([Kernel arguments](#kernel-arguments)) |
| `cute.make_rmem_tensor` | `make_rmem_tensor` (indexable at runtime) |
| `cute.max_common_layout`, `max_common_vector`, `dice`, `make_layout_like`, `make_ordered_layout`, `zipped_product`, `tiled_product`, `flat_product` | same names; a common run whose length is a runtime value counts for the largest static number known to divide it (8 for `(8,m):(1,8)`, as in C++ CuTe). `max_common_*` and `make_layout_like` take layouts, composed layouts and tensors (a `LayoutSource`; the layout before any swizzle), and runtime strides: `max_common_layout` of `(8,128):(k,1)` and `(8,128):(128,1)` is `128:8`. `make_layout_like` keeps the hierarchy and static-0 strides, counts runtime strides as larger than every static one, and gives size-1 modes stride 0, as C++ `compact_order` does |
| C++ `max_alignment(layout)` (CuTe DSL's `layout.max_alignment`) | `layout.max_alignment()`, `swizzle.max_alignment()` (`2^M`), `composed.max_alignment()` (the gcd of inner, offset and outer): C++'s algorithm (coalesce, `logical_divide(f, right_inverse(f))`), with runtime leaves counted by what divides them, as CuTe DSL counts their divisibility: a runtime extent as its known divisor (`Layout::col_major((m, k)).mode(0)` with `m = param_extent("m", 16)` gives 16, as `sgemm.py`'s `mA.layout[0].max_alignment % 16` reads it) and a runtime stride as 0, bounding the result by its divisor; scaled-basis strides give 1. `autovec_copy` bounds its vector by it |
| `cute.size_in_bytes(dtype, layout)` | `size_in_bytes::<T>(&layout)` and `size_in_bytes_dtype(dtype, &layout)`: `ceil_div(cosize * bits, 8)` as an `IndexExpr` (static when it folds, runtime for a runtime cosize; sub-byte types round up), a swizzled composed layout counting its outer layout's cosize and a layout inner its inner layout's, `None` 0; a slice of a staged composed layout (`staged.slice((.., .., 0))`) sizes one stage |
| `cute.viz.print_latex(x, color=, render_func=)`, `print_latex_tv(layout_tv, tile_mn, color=, palette=, title=, axis_labels=, render_func=)`, `PALETTES`, `Band`, `tikz_color_rainbow`, `tikz_escape` (and the deprecated `cutlass.utils.print_latex`) | `cute::viz`: `print_latex(&layout)` (a `Layout` or `ComposedLayout`, static, of rank 1 or 2) and `print_latex_tv(&tv, tile)` return a standalone LaTeX document as a `String`; `print_latex_with`/`print_latex_tv_with` take `LatexOptions`/`LatexTvOptions` (a colour function, a `Palette`, a title, axis labels, `body_only` for the `tikzpicture` alone in place of `render_func`), and `write_latex`/`write_latex_tv` write into any `fmt::Write`. `Palette::from_name` and `PALETTES` name `pastel`, `rainbow`, `rainbow_dual`, `white`, `bw` and `bw_dual` (`Palette::Custom` takes a function giving `Band`s); `tikz_color_bwx8`, `tikz_color_tv`, `tikz_color_white`, `tikz_color_rainbow` and `tikz_escape`. Ported from C++ CuTe's `print_latex.hpp`; the palettes' colours are cute-rs's own. A runtime or rank-3 layout is refused. The deprecated `cutlass.utils` alias has no counterpart |
| `cute.make_composed_layout(callable, offset, outer)` | `make_composed_layout_fn(closure, offset, outer)`, a `FnComposedLayout`: `call(c)` runs `closure(offset + outer(c))` when the kernel is built (so a gather closure can load an index tensor), it prints as `(fn o offset o outer)` (`named("gather")` names the inner), and slicing, the divides and the products act on the outer layout as for `ComposedLayout`. It has no memory form: `make_tensor` refuses it, as CuTe DSL does |
| `cute.math.exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `erfc`, `pow`, `fpowi`, `ipowi`, `cbrt`, `sqrt`, `rsqrt`, `rcp`, `copysign`, `rem`, `round`, `roundeven`, `floor`, `ceil`, `trunc`, `abs`/`absf`, `absi`, `neg`, `add`, `sub`, `mul`, `div`, `fma`, `min`, `max`, `clamp`, `isnan`, `isinf`, `isfinite`, `isnormal` (also `cute.exp2` and so on) | the same names in `cute::math`, with `fastmath` as the last parameter: `math::exp(x, true)` is `cute.math.exp(x, fastmath=True)` |
| `cute.TensorSSA`, `TensorSSA(value, shape, dtype)`, `.shape`, `.dtype`, `.reshape(shape)` | `TensorSsa<T>`: a static shape (`IntTuple`) and one SSA value per element in colexicographic order, scalarised (no vector type in the IR), immutable and cheap to clone. `TensorSsa::new(shape, values)`, `values()`, `shape()`, `size()`, `rank()`, `dtype()`, `reshape(shape)` (same size, same order); `{:?}` prints `tensor_value<vector<24xf16> o (4,(2,3))>`. It crosses `for_range_carry`, `if_else_yield` and `while_carry` (`Carry`), keeping its shape |
| `tensor.load()`, `tensor.store(v)`, `t[None, i] = v` | `tensor.load_ssa()` and `tensor.store_ssa(&v)` (`load`/`store` name the scalar access): a register fragment directly; a tensor in memory through a fragment like it (`make_fragment_like`, whose layout follows its strides) and `autovec_copy`, so a row-major 4x4 Float32 tile is four 16-byte loads. The shape must be static and sub-byte elements must fill whole 32-bit words. `store_ssa` needs equal sizes (not shapes), and converts implicitly only to a wider type of the same kind that holds every value exactly (CuTe DSL's `_cvt_to_dest` also converts to equally wide types and to wider ones that round or overflow, such as Float16 to BFloat16 or Int8 to Uint16, and stores the bits of another kind of the same width; cute-rs asks for `.to()` or `.bitcast()`). Slice assignment is `t.slice((.., i)).store_ssa(&v)`. A coordinate tensor has no `load_ssa` (it does not compile); a swizzled one loads and stores whole (CuTe DSL refuses composed layouts there) |
| `cute.full(shape, v, dtype)`, `full_like(a, v, dtype)`, `zeros_like`, `ones_like`, `empty_like` | `full::<T>(shape, v)`, `full_like::<T>(&a, v)`, `zeros_like::<T>(&a)`, `ones_like`, `empty_like` (0, as in CuTe DSL): `a` a `TensorSsa` or a `Tensor` of static shape, `v` a host number or a value converted with `.to()`. The element type is a type parameter (inferred or named, where CuTe DSL defaults to `a`'s). `make_fragment_like::<T, _>(&tensor_ssa)` gives the compact column-major fragment of its shape |
| `a + b`, `a - b`, `a * b`, `a / b`, `a % b`, `-a`, `abs(a)` on tensors | the same operators on `TensorSsa`, with a tensor, a `Value` or a host number of the tensor's type on either side (as `Value`'s operators take them, so unsuffixed literals infer the same way): `TensorSsa<T> op X` has the element type `Value<T> op X` has, with CuTe DSL's promotion (`Int8 + Int32` tensors give Int32, a Float16 tensor times a Float32 value Float32; a Rust literal adopts the tensor's type, where CuTe DSL makes it Float32). One scalar operation per element. Integer `/` truncates as `Value`'s does (`t.true_div(x)` is CuTe DSL's `/`, a Float32 result), `%` truncates (`remsi`), float `%` is `math::rem` (`remf`: the dividend's sign), `-x` flips the sign bit (`-0.0` from `+0.0`). Boolean tensors have no arithmetic (CuTe DSL's `+`/`-` in Int32) |
| `a // b`, `a ** b` on tensors | `a.floor_div(b)` (signed integers round toward -infinity, `floordivsi`; unsigned divide; floats `floor(a / b)`), `a.pow(b, fastmath)` (`math::pow` for floats, `math::ipowi` for integers), `a.fpowi(n)` (a float base, an Int32 exponent) |
| `tensor_ssa.broadcast_to(shape)`, implicit broadcasting (`_infer_broadcast_shape`) | `t.broadcast_to(shape)`, and every operator, comparison, `where_`, `apply_op` and two-operand math function: shapes are padded **on the right** with size-1 modes (column-major: a `(4,)` tensor meets `(4, 5)` as `(4, 1)`, unlike NumPy), must then be congruent, and each leaf equal or 1. cute-rs also lets a 1 stand against a nested mode (`(2, 1)` with `(2, (2, 2))`), and a one-mode tuple against its mode (a reduction's `(4,)` with a `4`). Values are repeated when the kernel is built, no instruction is emitted |
| `tensor_ssa[crd]`, `tensor_ssa[None, i]` | `t.get(crd)`: one element through `crd2idx` of the compact column-major layout; a static coordinate picks a value, a runtime one a balanced select tree (`ceil(log2 n)` deep) up to `SELECT_TREE_MAX` (32) elements, and above it a stack array (`make_rmem_tensor`) loaded at the offset. `t.slice((.., i))` with static coordinates: the shape `slice_(shape, crd)` gives, kept nested modes flattened as `Layout::slice` does (`(4,(5,4))` by `(None,(None,1))` is `(4,5)`) |
| `a < b`, `<=`, `>`, `>=`, `==`, `!=` on tensors | `a.lt(b)`, `le`, `gt`, `ge`, `eq`, `ne` (Rust's comparison traits return `bool`), after promotion and broadcasting: a `TensorSsa<bool>`. Float `ne` is unordered (true for NaN); Boolean tensors take only `eq` and `ne` (the build panics otherwise) |
| `a & b`, `a \| b`, `a ^ b`, `~a`, `a << b`, `a >> b` on tensors | `&`, `\|`, `^`, `!`, `<<`, `>>` on integer and Boolean (`& \| ^ !`) tensors, with tensor, value or host right-hand sides, promotion and broadcasting; `>>` is arithmetic for signed types and logical for unsigned; an oversized shift is reported per element by the simulator (LLVM's poison) |
| `cute.where(cond, x, y)` | `where_(&cond, x, y)` (`where` is a keyword): `cond` a Boolean tensor, `x` and `y` tensors, values or host numbers of one type (no promotion; a mismatch or a non-Boolean condition does not compile), at least one a tensor (else the build panics); scalars become tensors of that shape, the three broadcast, one `select` per element |
| `cute.any_(x)`, `cute.all_(x)` | `any_(&x)`, `all_(&x)`: `x != 0` per element (NaN true, `-0.0` false), OR- or AND-folded in a balanced tree, a `Boolean` |
| `tensor_ssa.reduce(op, init_val, reduction_profile)`, `cute.ReductionOp.ADD/MUL/MAX/MIN` | `t.reduce_all(op, init)` (a depth-0 profile: a `Value`) and `t.reduce(op, init, profile)` (a `TensorSsa`; `..` keeps a mode, an index reduces it, a leaf against a nested mode reduces all of it; the profile must be weakly congruent; the result's shape is `slice_`'s, flattened), `ReductionOp::{Add, Mul, Max, Min}`. Each output is `init` folded with its elements in one sequential chain (colexicographic for `reduce_all`; for `reduce`, lexicographic over the reduced leaves, last fastest, as CuTe DSL's `vector.multi_reduction` of the row-major vector takes them) (no reassociation), so the simulator and the GPU agree bit for bit. `Max`/`Min` propagate NaN (`max.NaN.f32` on Float32, `math::max_with`'s expansion on Float64, Float16 and BFloat16); integer `Max`/`Min` are a cute-rs extension (4.8.1 takes floats there) |
| `tensor_ssa.apply_op(op, other, flip=False)`, `cutlass.max`/`min` on tensors | `t.apply_op(other, flip, \|a, b\| ...)` with the operators' promotion and broadcasting (a scalar is broadcast, then `flip` swaps); `t.map(\|x\| ...)` for one operand; `maximum`, `minimum` (NaN-propagating, as `arith.maximumf`; `Value::max` ignores a NaN) and `t.max(x)`, `t.min(x)` |
| `cutlass.Vector[T, N]`, `Vector.from_elements(elems, dtype)`, `vec[i]`, `vec[:, 1:3]`, `vec.to(dtype)`, `vec.bitcast(dtype)`, `vec.reduce(op)` | `Vector<T>` (`cute::Vector`): a row-major n-D shape fixed when the kernel is built and one SSA value per element, the same scalarised machinery as `TensorSsa`. `Vector::from_elements([1.0, 2.0])`, `Vector::new(shape, values)`, `shape()`, `size()`, `rank()`, `values()` (row-major), `reshape`, `flatten`, `map`; `v.get(i)` / `v.get((i, j))` (a runtime `Int32` index of a 1-D vector picks through a select tree, or a stack array above 32 elements); `v.slice((.., 1..3))` with Rust ranges and indices (an index drops the dimension); no element assignment (immutable, as CuTe DSL raises `TypeError`). `+ - * / % & \| ^ << >>`, unary `-` and `!` with a vector, `Value` or host number, with `Value`'s promotion; `lt`/`le`/`gt`/`ge`/`eq`/`ne`, `abs`, `fma`, `apply_op`. Broadcasting is NumPy's: aligned on the right, padded with 1s on the left (the opposite of `TensorSsa`'s). `to::<U>()` converts as `TensorSsa::to`; `bitcast::<U>()` scales the innermost dimension, element 0 in the low bits (`(2, 4)` Float16 is `(2, 2)` Uint32). `reduce(ReductionOp::{Add, Mul, Max, Min})` folds in row-major order from element 0 (`reduce_with(op, acc)` from an accumulator), Max/Min NaN-propagating as `TensorSsa`'s (cute-rs's documented choice: the wheel's combining kind cannot be checked). It crosses regions (`Carry`); `{:?}` prints `vector<2x4xf32>` |
| `cutlass.vector.full`, `full_like`, `zeros_like`, `ones_like`, `empty_like`, `broadcast_to`, `where`, `any_`, `all_`, `create_mask`, `outerproduct`, `cutlass.print_nd_vector` | `cute::vector::full::<T>(shape, v)` and friends (in the `vector` module, so they do not collide with `TensorSsa`'s `full`; `empty_like` is zeros), `vector::broadcast_to(&v, shape)`, `vector::where_(&cond, x, y)`, `vector::any_`, `vector::all_`, `vector::create_mask(len, bound)` (lane `i` is `i < bound`, the bound a host number or a runtime `Int32`), `vector::outerproduct(&a, &b)` (`(m, n)` of `a[i] * b[j]`) and `vector::outerproduct_acc(&a, &b, &acc)` (fused for floats, as `vector.outerproduct` lowers to `vector.fma`), and `vector::print_nd_vector(&v)` (`printf` calls built here: `vector<2x3xf32> = [[...], [...]]`, elements formatted as `print_tensor` formats them) |
| `cute.TensorSSA(value, shape, dtype)` of a vector, `TensorSSA.from_vector(v, shape=)`, `.to_vector(force_flatten=)`, `.to_elements()` | `TensorSsa::from_flat_vector(&v, shape)` (the flat vector's order, unpermuted), `TensorSsa::from_vector(&v, shape)` (row-major to column-major: the element at row-major coordinate `(i0, ..., ik)` over the leaves of `shape`, by default the vector's own, lands at the colexicographic index of the same coordinate, so `v[i, j]` of a `(2, 3)` vector is element `i + 2j`), `t.to_vector(force_flatten)` (the reverse, the vector's shape the tensor's leaves; a nested shape needs `force_flatten`, or the build panics, as CuTe DSL raises), `t.to_elements()` (colexicographic) |
| `cute.arch.shuffle_sync*` of a 32-bit `TensorSSA` | `shuffle_sync`, `shuffle_sync_up`, `shuffle_sync_down`, `shuffle_sync_bfly` and their `_with` forms take a `Value`, or a `TensorSsa` or `Vector` of exactly 32 bits (two Float16, four Int8: bitcast to a Uint32, one `shfl.sync`, bitcast back to the same shape; the `Shuffle` trait). Any other width panics when the kernel is built |
| `tensor.load(mask=, pass_thru=)`, `tensor.store(data, mask=)` | `t.load_ssa_masked(&mask, pass_thru)` and `t.store_ssa_masked(&data, &mask)`: a Boolean `TensorSsa` of one flag per element (sizes must match). In memory each element is one access guarded by its flag: an `Op::Load`/`Op::Store` with a predicate (`pred`, and the load's `pass_thru` values), lowered to an `scf.if` around the access yielding `pass_thru` or a frozen `ub.poison` (an arbitrary value, so no later use is undefined behaviour to LLVM), which NVPTX turns into a branch over each scalar access, as it scalarises `llvm.masked.load`. A constant-true flag is a plain access and a constant-false store is dropped. A register fragment selects. Masked-off lanes touch no memory: the simulator skips their bounds, alignment and race checks and sector accounting, so an all-false mask past the end of a buffer is fine. Without `pass_thru` the lanes are undefined: pure operations pass the mark on, a select only from the operand it picks, and any other use (a store, an address, a branch, a shuffle, `printf`, a loop carry, ...) is reported naming the masked load's line |
| `cutlass.cute.tensor.gather(input, mode, index)`, `scatter(output, mode, index, data)` | `cute::tensor::gather(&t, mode, &index)` and `cute::tensor::scatter(&t, mode, &index, &data)` (not in the prelude, as CuTe DSL does not re-export them): element `(a, b, c)` read from `(index(a, b, c), b, c)` for mode 0, with an integer `TensorSsa` index (Int32, Int64, Uint8, ...), CuTe DSL's checks (static shapes, mode in range, equal ranks, flat shapes, no swizzle, the index's other modes no larger than the tensor's, no sub-byte elements). One access per element at a runtime offset; a register fragment through a select tree over the mode (at most 32 elements, else the build points at `make_rmem_tensor`). Scatter stores in colexicographic order of the index, so of duplicates the last wins. Out-of-range indices are undefined (CuTe DSL); the simulator reports those that leave the buffer |
| `cute.make_ptr(dtype, value, mem_space, assumed_align=)`, `cutlass.inttoptr(value, mem_space, dtype)` | `make_ptr::<T>(value, space, assumed_align)` and `inttoptr::<T>(value, space)`: a pointer at an integer address (an `Int64`, `Uint64`, `Int32` or `Uint32`) in `AddressSpace::Generic`, `Gmem`, `Smem` (an offset in the block's shared window), `Rmem` (an offset in the thread's local window) or `Tmem` (an `Int32`; loads and stores through it are refused, for m10's tcgen05). `assumed_align` defaults to the element's size and must be a power of two dividing it or a multiple of it; it is passed on as `cute.assume` (`llvm.intr.assume`), the simulator checks it, and vector proofs (`autovec_copy`, `load_vec`) take the pointer's alignment from it. Loads, stores, atomics, reductions and the copies built on them go through the address (`llvm.inttoptr`, `llvm.load`/`llvm.store`; generic addresses take generic instructions); `cp.async`, `ldmatrix`, bulk copies, mbarriers and prefetches take pointers into buffers and refuse a raw one, naming it. Constant and distributed shared memory wait for m8. A pointer array built on the device (`ptrs.store(j, p.toint_i64())`, then `make_ptr(ptrs.load(j), ...)`) is `tests/feature_kernels/raw_ptr.rs`'s `device_pointer_array` |
| `Pointer.toint()`, `cute.ptrtoint`, `llvm_ptr`/`to_llvm_ptr()`/`raw_ptr()` as asm operands | `ptr.toint()` (an `AnyValue`: an `Int64` for global and generic pointers, an `Int32` for the others, as CuTe DSL types it), `toint_i64()` and `toint_i32()`; a buffer's pointer lowers with `memref.extract_aligned_pointer_as_index` (shared addresses truncated to the 32-bit window, per-thread ones cast to the local window), a raw pointer's is its address. A `&Ptr` or `Ptr` converts into an `AnyValue`, so it is an operand of `inline_ptx`, `InlinePtx` and `printf`: an `l` register for a 64-bit address, `r` for a 32-bit one (in place of `llvm_ptr`, since the IR has no LLVM pointer values) |
| `Pointer.tospace(space)`, `ptr & mask`, `Pointer.align(min_align)`, `Pointer.alignment` | `ptr.tospace(space)`: itself for its own space, otherwise one side must be generic (else refused when the kernel is built), `cvta`/`cvta.to` through LLVM's `addrspacecast`, keeping the element type and alignment; the simulator reports a generic address outside the target window at the user's line. `&ptr & mask` (`i64`, `u64`, `i32`, `u32` or a runtime integer of the address's width): `toint() & mask` as a pointer of the same space, a constant mask clearing the low `k` bits proving `2^k`-byte alignment. `ptr.align(min_align)`: a power of two (else refused; TMEM refused), itself when already aligned, a static offset rounded up inside an aligned buffer, otherwise `(toint + a - 1) & !(a - 1)` remade with `assumed_align = a`. `ptr.alignment()` is the buffer's declared or the raw address's assumed alignment reduced by what divides the offset; `ptr.max_alignment()` also counts a swizzle's run. `ptr.is_raw()` tells raw pointers apart |
| `cute.runtime.make_ptr(dtype, value, mem_space, assumed_align=)` and `nullptr(dtype)` as launch arguments (host side) | `DeviceArg::Ptr(DevicePtr)` (cuda feature): unsafe `DevicePtr::from_raw(dtype, address, assumed_align, bytes)` (the alignment defaults to the element's size and the address must be a multiple of it, as `make_ptr` asserts; `bytes`, when given, bounds-checks the launch) and `DevicePtr::null(dtype)` for a parameter the launch leaves unused; `DeviceBuffer::view(byte_offset, bytes)` is a borrowed view (`DeviceArg::Slice`). The launch checks each address against the parameter's declared alignment, naming the address and the alignment, and a raw pointer's element type against the parameter's. The descriptor, `PointerSpec { dtype, mem_space, assumed_align }`, is `Hash + Eq` with `cache_key()` (CuTe DSL's `__cache_key__`). The simulator has no device addresses: raw device pointers are CUDA-only (its own `make_ptr` addresses are the row above) |
| Host-side read and write of a raw device address (no CuTe DSL counterpart; in-kernel reads are `make_ptr` + loads) | cuda feature, unsafe: `CudaContext::read_raw::<T>(&DevicePtr, byte_offset, count)` and `write_raw` (`cuMemcpyDtoH`/`HtoD`), `CudaStream::read_raw_async` and `write_raw_async`. A null pointer, an address not aligned for `T` and a range outside `DevicePtr::bytes()` (when known) are refused first; an address the driver cannot access returns a `CudaError` naming it, and the context stays usable |
| `cute.runtime.from_dlpack(t, assumed_align=None, use_32bit_stride=False, force_tf32=False)` and the result's `mark_layout_dynamic(leading_dim=)`, `mark_compact_shape_dynamic(mode, stride_order=, divisibility=)`, `element_type = T`, `__cache_key__`, `data_ptr`, `dynamic_shapes_mask`, `dynamic_strides_mask`, `leading_dim` | `from_dlpack(t, FromDlpack { assumed_align, use_32bit_stride, force_tf32 })` on a `DlpackTensor` gives an `ExternalTensor` without copying: a static layout from the shape and strides (compact row-major when null; negative strides refused), `data + byte_offset` as its start, checked against the alignment (the element's size by default), the dtype from DLPack's codes (`kDLInt`, `kDLUInt`, `kDLFloat`, `kDLBfloat`, `kDLBool`, the float8 e4m3/e4m3fn/e4m3b11fnuz/e5m2/e8m0fnu, float6 and float4 codes of DLPack 1.3; vector lanes refused except packed `float4_e2m1fn_x2`, seen with its logical FP4 shape, `(128, 128)` as `(128, 256)`, and the packed FP6 x4 forms). Then `mark_layout_dynamic(leading_dim)`, `mark_compact_shape_dynamic(..)`, `with_element_type(T)` (narrow types over a wider storage type), `cache_key()`, `data_ptr()`, `dynamic_shapes_mask()`, `dynamic_strides_mask()`, `leading_dim()`. `DlpackTensor` is the unsafe newtype over `*mut DLManagedTensor` (`from_managed`) or `*mut DLManagedTensorVersioned` (`from_versioned`; the read-only, is-copied and sub-byte-padded flags are read), whose deleter runs exactly once when the last clone goes, error paths included; `DlpackTensor::borrowed(*const DLTensor)` never deletes. The `#[repr(C)]` structs (`DLDevice`, `DLDataType`, `DLTensor`, `DLManagedTensor`, `DLManagedTensorVersioned`) and codes are in `cute::external::dlpack`. `enable_tvm_ffi=True` has no counterpart: a TVM-FFI callable ([Runtime, interop and testing](#runtime-interop-and-testing)) takes the framework's DLPack tensor itself |
| A torch, numpy or jax array passed straight to a `@cute.jit` function (`TensorAdapter`: `from_dlpack(t).mark_layout_dynamic()`) | a `DlTensor<T, H>` parameter (a `Tensor<T>` in the kernel) whose host value is any `DlpackSource`: a `DlpackTensor` (the default `H`) gets the implicit conversion, every extent and stride dynamic but a unit leading stride, and an `ExternalTensor` keeps its own layout. `kDLCUDA` memory is passed to a CUDA launch by address (on the context's device; another ordinal is refused); host memory (`kDLCPU`, `kDLCUDAHost`) is copied in and written back, on the simulator or the GPU, and the simulator refuses device memory. A read-only tensor given for a buffer the kernel writes is refused. Another crate's tensor type becomes a host value by implementing `DlpackSource` through its DLPack export (`cute::external`'s documentation shows how); cute-rs implements it for no foreign type, since it takes no dependencies |
| `from_dlpack(..., assumed_align=N)` on a parameter | `param_ptr_aligned::<T>(name, align)` (any power of two of at least the element's size) and `param_tensor_aligned(name, layout, align)` (a power of two of at least 16); `param_ptr` and `param_tensor` keep 16 bytes as their default. The parameter carries it (`llvm.align`), and every launch checks the device pointer with `check_pointer_alignment` (the CUDA launcher; the simulator checks its own 256-byte-aligned addresses) |
| `cutlass.Pointer`: `ptr[i]`, `ptr.load(alignment, count=N, is_volatile=, is_invariant=, ordering=, syncscope=)`, `ptr.store(v, alignment=, ...)`, `nvvm_load_ext`/`nvvm_store_ext`, `masked_load`, `masked_store`, `masked_gather`, `ptr + n`, `ptr - n`, `dtype`, `alignment`, `memspace`, `max_alignment` | on `Ptr<T>`: `load_at(i)`/`store_at(i, v)`; `load_vec(count, alignment)` (a 1-D `Vector`) and `store_vec(&v, alignment)`, moved in vectors of `alignment` bytes (at most 16 and the buffer's declared alignment; trusted as CuTe DSL's `llvm.load` trusts it, and the simulator checks every access); `load_with(LoadOptions)`/`store_with(v, StoreOptions)` and `load_vec_with`/`store_vec_with`, the ordered, volatile (`MemoryOrder::Volatile`), invariant (`MemoryOrder::Constant`: `ld.global.nc`), cache-operator, eviction and L2-prefetch forms of `load`/`store` (the combinations they refuse are refused; cluster scope needs sm_90); `masked_load(&mask, pass_thru, alignment)`, `masked_store(&v, &mask, alignment)` (lanes sharing one flag value move as one guarded vector, constant-true lanes plainly) and `masked_gather(&offsets, &mask, pass_thru, alignment)`; `+`, `+=` by element counts, `-`, `-=` by a static count no larger than the static part of the offset (offsets are non-negative until m5); `dtype()`, `alignment()`, `memspace()`, `max_alignment()`; and the integer half above (`toint`, `inttoptr`, `&` masking, `tospace`, `align`, asm operands), which completes the pointer/integer API. `load_swizzled`/`store_swizzled`/`apply_swizzle` (swizzled pointer accesses) are not implemented |
| `cute.math.*` of a `TensorSSA` | the same names as `TensorSsa` methods (`t.exp2(fastmath)`, `t.isnan(false)` a Boolean tensor, `t.atan2(x, fastmath)`, `t.copysign`, `t.fma(b, c, fastmath)`, `t.clamp(lo, hi, fastmath)`): the scalar function per element, broadcasting the operands; the free functions keep their scalar signatures |
| `tensor_ssa.to(dtype)` | `t.to::<U>()`: `Value::to` per element, plus Float32 (and Float16) to Float8E4M3FN/E5M2 two at a time (`cvt.rn.satfinite.e4m3x2.f32`, same bits), Int8 to BFloat16 by `cvt_i8_bf16_intrinsic` and Int4 to BFloat16 by `cvt_i4_bf16_intrinsic` (CuTe DSL's exact Ampere fast paths, on every target; Uint8 converts plainly), and Float4E2M1FN, Float6E2M3FN and Float6E3M2FN in pairs (`cvt.rn.satfinite.{e2m1x2,e2m3x2,e3m2x2}.f32`, sm_100a/sm_120a; an odd element count panics). FloatNV8E5M3FNU waits for `cvt_f32x4_to_fnv8e5m3x4` (m10) and panics, naming it. `Tensor::to` converts the same way |
| `tensor_ssa.bitcast(dtype)` | `t.bitcast::<U>()`: the bits kept, the element count scaled, the shape flattened to `(n,)`, element 0 in the low bits (`vector.bitcast`'s order): four Float32 are eight Float16. Equal widths per element; otherwise one width divides the other, lowered as `pack`/`unpack` (bit concatenation in the simulator), sub-byte types with their integer packing; bit counts that do not divide panic |
| `cute.print_tensor(t, verbose=False)` | `print_tensor(&t, verbose)` for memory tensors (loaded), fragments, `TensorSsa` values and coordinate tensors: a header (`tensor(raw_ptr<Float32, gmem, align(16)> o (2,3):(3,1), data=`; `rmem<..>` for registers, `coord<origin>` for coordinates) and torch-style brackets, mode 0 down and mode 1 across, rank-2 blocks of the higher modes in an outer bracket (C++ CuTe's order); floats `%7.4f` (narrow types as doubles: `vprintf` has no half), integers `%4d`/`%4u` (`%4lld`/`%4llu` for 64 bits), 128-bit integers in hex; `verbose` puts each element on a line after its coordinate. Every thread prints (wrap it in `if_(thread_idx().x.eq(0), ..)`) |
| `cute.testing.assert_(cond, msg)`, `cute.testing.RuntimeAssertion(num_assertions, init_value=, disable=)` with `store` and `verify` | `testing::assert_(cond, msg)` (checked by the simulator by default; compiled in as `cf.assert` only with `enable_assertions`), `testing::RuntimeAssertion::new(n).init_value(..).disable(..)` on the host and `testing::DeviceAssertions` in the kernel (`store(idx, pred, msg)`), `verify(&kernel)` giving an `AssertionFailure` (message and Rust location). See "Errors and diagnostics" |
| `DSLUserCodeError`, `DSLRuntimeError`, `CompilerDiagnosticError`, `DSLCudaRuntimeError`; `CUTE_DSL_SHOW_STACKTRACE`, `CUTE_DSL_FILTER_STACKTRACE` | `cute::Error` (from every error type, `loc()`, `{:#}`/`render()` with a code frame); `RUST_BACKTRACE=1`. Compiler warnings and remarks are `Diagnostic`s on `CompiledKernel`, register and memory use a `ResourceUsage` |
| `cute.math.sincos(x)` | `math::sincos(x, fastmath)`, a `(sin, cos)` tuple |
| `approx=`, `ftz=`, `rounding=cute.RoundingMode.ZERO`, `full=`, `propagate_nan=` | the `_with` functions and `MathOptions`: `math::exp2_with(x, MathOptions::APPROX.with_ftz())`, `math::add_with(x, y, MathOptions::rounding(RoundingMode::Zero))` |
| `cute.arch.exp2(x)` (deprecated) | `math::exp2(x, true)` (`ex2.approx.ftz.f32`) |
| `@cute.kernel` | `Kernel::build(name, \|\| { ... })` |
| `cute.compile[cute.OptLevel(2), cute.GenerateLineInfo(True), cute.GPUArch("sm_90a")](fn, ...)` | `kernel.compile_with(&toolchain, &CompileOptions::new().opt_level(2).generate_line_info(true).gpu_arch(Arch::Sm90a))` |
| `cute.compile(fn, ..., options="--opt-level 2 --gpu-arch sm_90a")` | `CompileOptions::parse("--opt-level 2 --gpu-arch sm_90a")?` |
| `CUTE_DSL_COMPILER_OPT`, `CUTE_DSL_ARCH`, `CUTE_DSL_LINEINFO` | `CUTE_RS_COMPILER_OPT`, `CUTE_RS_ARCH`, `CUTE_RS_LINEINFO` (`CompileOptions::with_env`; `Kernel::run_cuda` reads them). Unlike in CuTe DSL, explicit options win over them ([Compile options](#compile-options)). |
| a compiled function's `__ptx__` | `CompiledKernel::ptx()`, with `arch()`, `ptx_version()`, `opt_level()` and `line_info()` |
| `compiled.export_to_c(file_path, file_name, function_prefix='')` (`<name>.h` and `<name>.o`; `<prefix>_Kernel_Module_t`, `_Kernel_Module_Load`, `_Kernel_Module_Unload`, the wrapper; `--host-target`) | `compiled.export_c(dir, name, prefix)` (`export_c_with` for `ExportOptions`: the C compiler, the `cuda.h` directory, no object file), in `cute::aot`: `<name>.h` (version macros, `<prefix>_Args_t` with one field per kernel parameter, `<prefix>_Launch_Config_t`, `<prefix>_Kernel_Module_t`, `<prefix>_Kernel_Module_Load`, `_Unload` and `<prefix>_Kernel_Launch`), `<name>.c` (the PTX and cubin as byte arrays and the functions), `<name>.o` when `cc` and `cuda.h` are found (else a note), and the manifest `<name>.sig` with `<name>.ptx` and `<name>.cubin`. The C uses the driver API (`CUresult`, `cuModuleLoadData`, `cuLaunchKernelEx`, `-lcuda`) rather than the runtime API's `cudaError_t` and `cudaLibrary_t`, and loads into the current context rather than onto every device; there is no cross-compiling host target. See "Ahead-of-time export" below |
| `cute.runtime.load_module(path)`, `find_runtime_libraries()` | `CudaContext::load_exported("<name>.sig")` (cuda feature), or `load_exported_kernel(&ExportedKernel::from_parts(manifest, ptx, cubin))` for files embedded with `include_str!`/`include_bytes!`: a `CudaFunction` with the exported `KernelSignature`, so `launch_builder` checks and packs arguments without the kernel's IR. Loading checks the format version, the target and the PTX version. libcuda is the only runtime library, so `find_runtime_libraries` has no counterpart |
| `cute.arch.get_dyn_smem(element_type, alignment=None)`, `get_dyn_smem_size()` | `get_dyn_smem::<T>(alignment)` (`None` for the element's size; the region starts at a multiple of 16 at least, after the shared memory sized at build time; tensors over the region take static extents, like all shared memory), `get_dyn_smem_size()` |
| `.launch(..., smem=None\|n)` | `LaunchConfig::with_smem(n)` (`smem_bytes`): the block's shared memory in all, which must cover what the kernel allocates when it is built (CuTe DSL warns when it does not; cute-rs refuses, as such a launch overlaps memory), the rest going to the `get_dyn_smem` region; no size (the default) is the build-time allocations plus `with_dynamic_smem(n)`, the region's bytes. Above the device's opt-in maximum the launch fails naming both numbers; the launcher passes it as `sharedMemBytes` and raises `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` above 48 KB. CuTe DSL's compiler attribute `smem.config_smem_size` has no counterpart: cute-rs's emitter places the buffers itself (`KernelIr::shared_memory_layout`) |
| `cute.arch.dynamic_smem_size()`, `total_smem_size()`, `aggr_smem_size()` | the same names; `dynamic_smem_size()` reads the same as `get_dyn_smem_size()`, the bytes `with_dynamic_smem` gave the region (what `total_smem_size` counts is under Limitations) |
| `cute.arch.cluster_arrive(aligned=)`, `cluster_arrive_relaxed(aligned=)`, `cluster_wait()` | the same names, with an `_aligned` suffix for `aligned=True`; cute-rs adds `cluster_wait_aligned`, and `cluster_sync` as in CuTe C++ (sm_90; `.relaxed` needs PTX 8.0) |
| `.launch(stream=...)` with a `cuda.bindings.driver.CUstream` or a `cudaStream_t` (`torch.cuda.current_stream().cuda_stream`) | `CudaStream` (cuda feature): `CudaContext::stream()` (`cuStreamCreate` with `CU_STREAM_NON_BLOCKING`, destroyed on drop), `default_stream()` (the legacy NULL stream, the default as in CuTe DSL), unsafe `CudaStream::from_raw(&ctx, raw)` for another library's stream (not destroyed), `as_raw()`, `synchronize()`, `is_idle()`. `ctx.launch_builder(&kernel, &function, cfg).stream(&s).args(&args).submit()` (or `function.launch_builder(cfg)` with the compiled kernel's signature, or `ctx.launch_on(&s, ..)`) queues without allocating or synchronising. Stream-ordered copies: `upload_async`/`download_async` between `DeviceBuffer`s and page-locked `PinnedBuffer`s (`alloc_pinned`, `pinned_from`), which a `Pending` keeps borrowed until the stream is synchronised; `unsafe` `upload_raw_async`/`download_raw_async` for other host memory; `copy_async` and `memset_async` on the device. The simulator has no streams |
| `cutlass.cuda`: `initialize_cuda_context`, `get_device_info`/`DeviceInfo.pretty_str`, `get_device_attribute`, `get_driver_version`, `stream_create`/`stream_sync`/`stream_destroy`/`default_stream`, `allocate`/`deallocate`, `memcpy_h2d`/`memcpy_d2h`, `load_cubin_module(_data)`/`get_kernel_function`, `set_kernel_attribute` | `CudaContext::new`, `device_info()` (a `DeviceInfo` that prints itself: name, compute capability, SMs, L2, shared memory per block and per SM, global memory, driver), `device_attribute(DeviceAttribute::..)`, `driver_version()`, `CudaContext::device_count()`, the streams above, `alloc::<T>(n)`/`alloc_bytes(n)` (no upload), `alloc_async(bytes, &stream)`/`DeviceBuffer::free_async` (memory pools), `upload_slice`, `DeviceBuffer::write`/`read`/`copy_from` at byte offsets, `memset`, `to_vec`, `load_cubin`/`load_fatbin`/`load_module_file`, `CudaFunction::set_attribute`/`get_attribute(FuncAttribute::..)`, and raw handles (`as_raw()` on the context, function, buffer, stream and graphs). The `CUlibrary` API is not wrapped |
| `cutlass.utils.HardwareInfo(device_id=0)`: `get_max_active_clusters(cluster_size, stream=None)`, `get_l2_cache_size_in_bytes`, `get_device_multiprocessor_count` | `cuda::HardwareInfo::new(id)` (an id past the device count is an error), the same methods returning `Result`: the empty kernel is an embedded PTX module, with all the opt-in shared memory, `cuOccupancyAvailableDynamicSMemPerBlock`, `cuOccupancyMaxActiveBlocksPerMultiprocessor` and `cuOccupancyMaxActiveClusters` (a green-context stream narrows it); cluster sizes outside 1 to 32 and drivers before CUDA 11.8 are errors. Before sm_90, a cluster of 1 is blocks per SM times SMs and larger clusters are errors. `CudaFunction::max_active_clusters(block, dynamic_smem, cluster, stream)` asks the same of a real kernel |
| CUDA graphs: a compiled function called on a stream inside `torch.cuda.graph()`, `cutlass.testing.benchmark(use_cuda_graphs=True)` | `stream.begin_capture(CaptureMode::{Global, ThreadLocal, Relaxed})` returns a guard (dropping it ends the capture); `end()` gives a `CudaGraph` (`node_count()`), `instantiate()` a `CudaGraphExec`, `exec.launch(&stream)` replays it; all destroyed on drop. Launches on the capturing stream, `copy_async` and `memset_async` are captured, and setting a function attribute is allowed; a call that cannot be captured (`CudaContext::launch`, which uploads and waits) fails with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` or `_INVALIDATED` and a note saying why. A graph keeps the device addresses it captured: the buffers must outlive its replays |
| CuTe DSL's JIT cache (`JitCacheDict`, `CUTE_DSL_JIT_CACHE_MAX_ELEMS`, `fn(..., no_cache=True)`, `CUTE_DSL_NO_CACHE`) and file cache (`CUTE_DSL_CACHE_DIR`, `CUTE_DSL_DISABLE_FILE_CACHING`) | `Kernel::compile_with` caches compiled kernels in memory, process-wide and thread-safe, keyed on the MLIR text (with locations only under line information, so moving code keeps the key), the signature, target, PTX version and output-changing options (`CompileOptions::cache_key()`), the lowering pipeline, the `mlir-opt` and `ptxas` paths, versions, sizes and modification times and the crate version; bounded LRU-style by `CUTE_RS_JIT_CACHE_MAX_ELEMS` (1024). On disk, the PTX, cubin and `ptxas` log go to `CUTE_RS_CACHE_DIR` (default `temp_dir()/cute-rs-<user>`; on by default, as CuTe DSL's is; created mode 0700 and, before each use, required to be a real directory owned by the user that no one else can write, or the disk cache is skipped with a warning), named by a SHA-256 of the key plus the running executable's identity, with the full key stored and compared on read, a SHA-256 checksum, a temporary-file-and-rename write, and LRU eviction past `CUTE_RS_FILE_CACHE_MAX_ENTRIES` (256) or `CUTE_RS_FILE_CACHE_MAX_BYTES` (64 MiB); `CUTE_RS_DISABLE_FILE_CACHING` turns it off. Each `CudaContext` caches loaded modules by a hash of the image and JIT options, so `run_cuda` twice loads once (`modules_loaded()`, `module_cache_len()`, `clear_module_cache()`). `CompileOptions::no_cache(true)`, `CUTE_RS_NO_CACHE`, stage hooks and kept artifacts bypass the caches; hits and misses are logged and counted (`cache_stats()`) |
| `__cache_key__()` of `from_dlpack` tensors, `make_ptr` pointers and `make_fake_tensor` specs | `TensorSpec::cache_key()` (dtype, memory space, alignment, static leaves, and each dynamic leaf's width, divisibility and the first position of its `SymInt` in the spec), `ExternalTensor::cache_key()`, `PointerSpec::cache_key()`, `SymInt::cache_key()` (its identity: equal `SymInt`s made separately key differently); `TensorSpec`, `SymInt` (by value, as `==` compares), `Arch` and `CompileOptions` are `Hash`. `KernelCache<K>` memoizes a built and compiled kernel per user key: `cache.get_or_compile(key, \|\| Ok((kernel, compiled)))` |
| `cutlass.register_library_unload_callback(cb)`, `unregister_library_unload_callback(cb)` | `cute::cuda::register_library_unload_callback(\|module\| ..)` returns an `UnloadCallbackToken` for `unregister_library_unload_callback(token)`. Callbacks get the raw `CUmodule` just before cute-rs unloads a module (when its last function and its cache entry are gone: on eviction, `clear_module_cache` or the context's drop); a panicking callback is logged and skipped. `CudaFunction::as_raw()` gives the module handle. cute-rs wraps no `CUlibrary` API, so callbacks get a `CUmodule` |
| `.launch(use_pdl=True)`, `cute.arch.griddepcontrol_wait()`, `griddepcontrol_launch_dependents()` | `LaunchConfig::with_pdl(true)` (`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`) and `griddepcontrol_wait()`, `griddepcontrol_launch_dependents()`, emitted as CuTe DSL emits them, inline PTX with side effects and a memory clobber (sm_90, PTX 7.8; the validator names the line on older targets). On the SM89 GPU, driver 595 accepts the attribute and runs the kernels in order. The simulator's `SimStream` runs kernels in order over shared buffers (`stream.buffer(arg)`, `stream.launch(&kernel, cfg, &[StreamArg])`, `stream.get(b)`): a launch with `use_pdl` is checked against the previous launch's global-memory footprint, and an access before the thread's `griddepcontrol_wait()` that reads what the previous kernel wrote, or writes what it read or wrote, is reported at its line. In a single launch both are no-ops |
| `.launch(launch_completion_event=, launch_completion_event_flags=, programmatic_event=, programmatic_event_flags=, programmatic_event_trigger_at_block_start=)`, CUDA events | `LaunchBuilder::launch_completion_event(&event, flags)` (`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`) and `programmatic_event(&event, flags, trigger_at_block_start)` (`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT`). `CudaEvent` (cuda feature): `ctx.event(EventFlags { blocking_sync, disable_timing })` (`cuEventCreate`, destroyed on drop), `record(&stream)`, `is_complete()` (`cuEventQuery`), `synchronize()`, `elapsed_ms_since(&start)` (`cuEventElapsedTime`), `as_raw()`; `CudaStream::wait_event(&event)` (`cuStreamWaitEvent`). On SM89 with driver 595 both launch events are accepted and complete (the programmatic one as the blocks exit); a launch event with timing is refused with `CUDA_ERROR_NOT_SUPPORTED`. The simulator has no events |
| `.launch(cooperative=True)` | `LaunchConfig::with_cooperative(true)` (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`): the launcher checks `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH` and that the grid fits at once (`CudaFunction::max_active_blocks_per_multiprocessor(threads, dynamic_smem)`, `cuOccupancyMaxActiveBlocksPerMultiprocessor`, times the SM count), with an error naming both numbers before the driver's `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`. The simulator runs a cooperative launch with the whole grid resident; a grid that spins on another block without it is reported as a hang with a hint to launch cooperatively. CuTe DSL has no `grid.sync`: `cooperative_launch` builds the grid barrier |
| `.launch(cluster=[...], fallback_cluster=[...])` | `LaunchConfig::with_fallback_cluster(preferred, fallback)`: the preferred shape (`CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION`, SM100) and the guaranteed one (`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`); the grid must divide by the preferred shape and the preferred by the fallback in every mode, and a fallback without a cluster is refused. The simulator runs every cluster at the preferred shape, every one at the fallback, or a seeded mix (`SimOptions::cluster_shapes`: `ClusterShapes::{Preferred, Fallback, Mixed { seed }}`), and the cluster registers report each cluster's own shape. No cluster directive (`.reqnctapercluster`, `.explicitcluster`) is emitted: the shape comes from the launch. A clustered launch on a device without clusters (`CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH` 0, before SM90) fails before the driver, as does a preferred shape before SM100 |
| `.launch(max_number_threads=[x, y, z], min_blocks_per_mp=n)` | `KernelOptions::max_threads([x, y, z])` (`nvvm.maxntid`, PTX `.maxntid`) and `min_blocks_per_sm(n)` (`nvvm.minctasm`, `.minnctapersm`). A declared `block_dim` gives `.reqntid` (`nvvm.reqntid`), CuTe DSL's default, unless `max_threads` is set (PTX forbids both); launches with a larger block are refused by the simulator and the launcher |
| `preferred_smem_carveout=pct`, `smem_merge_branch_allocs=True`, `hint_smem_base_uniform` | `LaunchConfig::with_smem_carveout(pct)` (-1 default, 0 most L1, 100 most shared memory; `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`, read back with `CudaFunction::get_attribute`); without one and with `min_blocks_per_sm(n > 1)`, the launcher sets a carveout that fits `n` blocks' shared memory on an SM. `smem_overlay(\|\| alloc_a, \|\| alloc_b)`: two shared buffers at one offset (each alternative allocates one buffer), as the layout, the emitter's byte views and the simulator all place them, so the race checker sees overlaid accesses as the same bytes. `KernelOptions::hint_smem_base_uniform(_)` is accepted and ignored, like `make_warp_uniform` |
| `cute.arch.block_idx_in_cluster()`, `cluster_size()` | the same names: the linear rank in the cluster (x fastest) and the cluster's block count; `block_in_cluster_idx()` is the 3-D index |
| `cute.arch.cp_async_mbarrier_arrive_noinc(mbar_ptr)`, `cutlass.experimental.primitives.cp_async_mbarrier_arrive(addr, noinc=False)` | `cp_async_mbarrier_arrive_noinc(&ptr)` (also for `noinc=True`), `cp_async_mbarrier_arrive(&ptr)`; SM80 and newer, where `cute.arch.cp_async_mbarrier_arrive_noinc` asks for SM90 |
| `cute.assume(src, divby=None, pow2=None)` | `assume(src, divby, pow2)` (`divby: Option<usize>`, `pow2: bool`) on an integer value, an `IndexExpr` or a host integer. The facts flow through arithmetic (a sum keeps the gcd, a product the product, a floor division by a constant the quotient when it divides) and prove vector accesses aligned; a static value is checked when the kernel is built, a runtime one by the simulator, and a power-of-two `divby` reaches LLVM as `llvm.intr.assume`. Dynamic integers print their facts as CuTe DSL does: `?{div=8}`, `?{i64 div=4}` |
| `cute.get_divisibility(x)` | `get_divisibility(&x)`: a static value's own value, else the largest known divisor (1 when nothing is known) |
| `cute.fast_divmod_create_divisor_v2(d)`, `FastDivmodDivisorV2`: `divmod(n, d)`, `n // d`, `n % d`, `d.divisor` | `FastDivmod::new(d)` in the kernel (one 64-bit division), or `FastDivmod::host(d)` passed with `param_fast_divmod("d")` and `KernelArg::fast_divmod(d)`: `d.divmod(n)`, `n / d`, `n % d`, `d.divisor`, for `0 <= n < 2^31` (CUTLASS's `FastDivmod`) |
| 64-bit dynamic shapes and strides (`cute.sym_int64`, `from_dlpack`'s 64-bit strides), printed `?{i64}` | an `Int64` leaf (`IndexExpr::from(x)`, or `param_extent_i64(name, multiple_of)`): the offsets it enters are computed in 64 bits, the rest stays 32-bit |
| `cute.Tensor[dtype, shape, stride]` (`TypedTensor(dtype, shape, stride, memspace=, assumed_align=)`): `element_type`, `shape`, `stride`, `memspace`, `assumed_align`, `isinstance(t)`, its text | `TensorSpec::new::<T>(shape, stride)` (`TensorSpec::of(dtype, ..)`), `.with_memspace(..)`, `.with_assumed_align(n)` (default: the element's size, as CuTe DSL's), with the same accessors and `spec.isinstance(&tensor)`; prints `Tensor<f32, (16,), (1,)>`. Leaves are static integers or `SymInt`s. `param_tensor::<T>(name, &spec)` declares it. `mlir_type` has no counterpart: the kernel's IR carries the parameters |
| `cutlass.base_dsl` `JitArgument` (`__c_pointers__`, `__get_mlir_types__`, `__new_from_mlir_values__`), `DynamicExpression` (`__extract_mlir_values__`), `extract_mlir_values`/`new_from_mlir_values`, `cutlass.register_jit_arg_adapter` | the `KernelParam` trait: `declare(name, &spec)` in the kernel, `pack(name, &spec, &host, &mut ArgPack)` and `write_back` on the host, and the loop half `carry_flatten`/`carry_rebuild` (`Carry` is CuTe DSL's `DynamicExpression`). Implemented for `Value<T>`, `Tensor<T>`, `Ptr<T>`, layouts, shapes and composed layouts, `FastDivmod`, tuples, arrays, `Vec`, `Option`, `Constexpr` and plain Rust constants. A third-party type implements the trait (a newtype where Rust's orphan rule needs one) instead of registering an adapter |
| `cute.Layout`, `cute.ComposedLayout` and Shape/Stride/Coord tuples as kernel arguments | `param_layout(name, &LayoutSpec::new(shape, stride))` or `LayoutSpec::col_major(shape)` (strides computed in the kernel), `param_shape(name, &spec)`, `param_composed_layout(name, &ComposedSpec { inner, offset, outer })`; as `KernelParam`s, `Layout`, `Shape` and `ComposedLayout` take a static host layout, tuple or composed layout. Each dynamic leaf is one scalar parameter with its width and divisibility; the static parts are checked at launch. Static layouts still work by closure capture |
| `Tensor.mark_layout_dynamic(leading_dim=None)` | `host.mark_layout_dynamic(leading_dim)` on a `HostTensor` (`Option<i32>`, negative counting from the end), returning it or a `SpecError` with CuTe DSL's message; flat layouts only. Shapes are `Int32`, strides `Int64` unless `with_32bit_stride(true)` (the guide's printed t0 to t4 show 32-bit strides; its t5 the default `?{i64}`) |
| `Tensor.mark_compact_shape_dynamic(mode, stride_order=None, divisibility=1)` | `host.mark_compact_shape_dynamic(mode, Some(&order), divisibility)`, the order outermost first as `dim_order()`; dependent strides become `SymInt` leaves with the derived divisibility (`(?{div=2},4,16,2):(2,?{i64 div=4},?{i64 div=16},1)`), size-1 strides 0, and the order is remembered for chained calls. Every error of the guide, with its message. For a tensor that is not compact, `assume` in the kernel |
| `cute.sym_int(width=32, divisibility=1, symbol=None)`, `sym_int32`, `sym_int64`; `SymInt` `*`, `%`, `//`, its text | `sym_int(width, divisibility, symbol)`, `sym_int32(divisibility, symbol)`, `sym_int64(..)`: a `Copy` handle with an identity (`s.same(&t)`; `==` compares width, divisibility and symbol, as CuTe DSL's). `s * n`, `s % n` (`SymRem::Zero` or a `SymInt` of the gcd), `n % s`, `s / n`, `s / t`, each a fresh `SymInt`; prints `?{div=8}`, `M ?{i64 div=16}`. One `SymInt` anywhere in a kernel's specs (extents, strides, a `ScalarSpec::Sym` scalar) is one parameter, and a launch checks that every occurrence got the same value and that it meets the divisibility (CuTe DSL's TVM-FFI path checks neither) |
| `runtime.make_fake_tensor(dtype, shape, stride, memspace=, assumed_align=)`, `make_fake_compact_tensor(dtype, shape, stride_order=, use_32bit_stride=)` | `make_fake_tensor::<T>(shape, stride)` and `make_fake_compact_tensor::<T>(shape, stride_order, use_32bit_stride)` (the order gives each mode's rank from the innermost, as there), `TensorSpec`s in `Gmem` |
| `@cute.jit` / `@cute.kernel` with annotated parameters, all runtime unless `Constexpr` (`NumericMeta.from_python`, int tuples as dynamic `IntTuple`s, `None` for an `Optional`) | `Kernel::jit(name, spec, \|x: Int32, t: Tensor<f32>, s: Shape, o: Option<Ptr<f32>>\| ..)` (or `Kernel::builder(..).jit`), a `TypedKernel<P>`: `simulate(cfg, &mut host)`, `run_cuda`, `launch` and `pack` take host values of the matching types (`i32`, `f32`, `HostTensor`, `IntTuple`, `Option`, tuples), checked and packed, with outputs taken back. `Kernel::jit_for(name, &host, body)` takes the spec from the host values (CuTe DSL's call-site typing: a plain scalar per number, a dynamic `Int32` per integer of a shape). Tuples and arrays replace `*args`; a host value of the wrong count or type does not compile |
| `cutlass.Constexpr[T]` | `Constexpr<T>` (or a plain Rust value): no parameter; the kernel sees the value it was built with, and a launch checks the host passes the same |
| NamedTuple and dataclass (frozen or not) kernel arguments, nested tuples and lists (`tree_flatten`, `DefaultDataclassAdapter`) | `kernel_params! { struct P(PSpec, PHost) { a: Tensor<f32>, n: Int32, tile: usize } }`: runtime fields become parameters named after the field, plain Rust fields stay build-time constants; it also declares `PSpec` and `PHost` and implements `Carry`, so the struct flows through `for_range_carry` and `Var` (a constant changed in a loop body is refused, naming the field) |
| a kernel argument passed by value (`native_struct` as one `.param`, a TMA descriptor as a grid constant) | `param_struct(name, &NativeLayout)` (fields read with `.field::<T>(path)`; `NativeLayout::pack(&values)` and `KernelArg::by_value(bytes)` on the host, `DeviceArg::ByValue` on the device) and `param_bytes(name, size, align, grid_constant)`: one `.param .align A .b8 k_param_i[N]`, from `llvm.byval` (and `nvvm.grid_constant`) set by a stage hook. The kernel cannot write it (a store is refused at the user's line) |
| the launcher half of argument marshaling (`tree_flatten`, `JitArgAdapter`, the C pointers of a JIT function) | `KernelSignature` (`Kernel::signature()`, `CompiledKernel::signature()`, `CudaFunction::signature()`): each parameter's kind (`ParamKind::{Pointer, Scalar, ByValue}`), dtype, length, divisibility and alignment; shared memory, `block_dim`, `max_threads`, `min_blocks_per_sm`, module globals, target and PTX version. The simulator, `CudaContext::launch` and the device launches all check against it, and `KernelSignature::pack(&[LaunchValue])` is the one packing function: dtype, divisibility, 64-bit leaves, pointer alignment and reach, and by-value size, each error naming the parameter (leaf). `DeviceArg::View { buffer, byte_offset }` passes a pointer into a buffer |
| module-scope `cutlass.Array(dtype, n, space=gmem\|cmem, name=, init=, readonly=, extern=)`, `write_to_symbol` | `module_global::<T>(name, n, AddressSpace::Gmem\|Cmem, init, readonly)`: a public `memref.global` in address space 1 or 4 (`.visible .global` / `.visible .const`, with its initializer; an LLVM constant when `readonly`), read through a generic memref. Names are C identifiers (generated when `None`); at most 64 KB of constant data; no device stores to constant or read-only globals. On the GPU, `CudaFunction::global(name)` reads and writes it (`cuModuleGetGlobal`; read-only ones refuse writes); in the simulator, `kernel.sim_module()` keeps globals across launches (`simulate`, `read_global`, `write_global`), and `Kernel::simulate` starts from their initial values. `extern_global` fails: extern globals need device-code linking (extern-ffi) |
| `@cute.native_struct(zero_init=True, packed=False)`, `make_native_struct` | `kernel_params!` with a first attribute `#[native_struct(zero_init = true, packed = false)]`: `S::make(&spec)` zeroes the fields (without `zero_init` their contents are unspecified, and zero, as cute-rs has no undefined value), `S::native_layout(&spec)` is the C layout (natural or packed, constants left out, a nested native struct placed whole with its own alignment and padding). As a kernel argument it is flattened into one scalar per field; the by-value `.param` struct waits for kernel-argument marshaling (m6), and returning one from a device function for device functions (m11) |
| `cutlass.experimental.task_scheduling` part 1: `MemoryResource(name=, pipeline_config=)` with `@producer_work` / `@consumer_work(returns=, work_attrs=WorkAttr.AUXILIARY)` methods, `get_smem_requirements()` / `get_tmem_requirements()`, `StageInfo`, `TaskLocalVariable(dtype, default, docs)` / `.uninitialized()`, `PipelineConfig.create_{async_async,tma_async,tma_umma,umma_async,async_umma,umma_umma,clc_fetch_async}_pipeline_cfg` (interleave strides, signalling threads, advance flags), `TileSchedulerConfig.create_{static_persistent,clc_dynamic_persistent}_tile_scheduler_params`, `WorkQueue` (`create_tile_scheduler`, `initial_work_tile_info`, `skip_work_tile_if`, `init_work_tile`, `get_and_advance_work_tile`, `fetch_work_tile`, `producer_tail`), `@schedule`, `domain_loop(start, end, step, unroll=)` with `first_iter()` / `last_iter()` / `every(n, start=)`, `work_tile_loop(wq, skip_if=)` with `skippable()`, `when_true` / `when_false`, `ScheduleError` | `cute::task_scheduling`: `MemoryResource::new(name).with_pipeline_config(cfg).work(Work::consumer("get_item").returns(&item).run(\|stage_info, args\| ...))` (work methods are named closures over a `StageInfo` and named `WorkArgs`, returning `WorkResults`; `.auxiliary()`, `.input::<T>(name)`), the `Resource` trait (name, pipeline config, requirement hooks, `create()` materialising m9/m10's pipeline), `run_work`, `try_acquire` / `acquire` / `commit` and `try_wait` / `wait` / `release` over explicit `PipelineState`s (a probe's result is the next wait's token), `TaskLocalVariable<T>` (typed handle; `TaskLocals` stores it in `Var`s), `StageInfo` (`Carry`), `WorkQueue::new(name, cfg).with_skip_if(\|tile\| ...)`. `schedule(name, &[&res..], \|s, handles\| ...)` records an ordered tree (`Schedule`, printable as a listing) without emitting IR: `handle.call("work")` / `call_with("work", [("arg", &flow)])` return `Flow` handles, `s.domain_loop(..)`, `s.work_tile_loop_skipping(wq, \|w\| w.skippable(..))`, `s.when_true(cond, ..)`; each step gets its head / loop / tail `Phase`. Capture reports every problem as an `Issue` (unknown work with a did-you-mean, unproduced or unmirrored variables, wrong or stale inputs, unpaired acquire/commit and wait/release, unsupported guards, misplaced loops). The copy tutorials' schedules are compared with listings; the resources run in hand-written warp-specialised kernels on the simulator and the SM89 GPU (`AsyncAsync` staging, static work queue) and, for the CLC work queue, on the simulator. Tasks, the TaskManager and the allocators come in parts 2 and 3 |
| `cutlass.experimental.task_scheduling` part 2: `Task(name, src_resources, dst_resources, warp_idx, num_warps, schedule)` with a `setmaxnreg` budget, `TaskManager(tasks, resource_dependency_graph, iket_enable_profiling)`, `setup_resources_and_tasks()`, `run()`, `DependencyGraphEdge` / `DependencyGraphControlEdge`, the manager's checks, `PdlWaitBarrier` (`wait_griddep`) / `PdlLaunchBarrier` (`launch_griddep`); tutorials `01_copy_grid_stride.py`, `01_copy_persistent_skip_tile.py` | `cute::task_scheduling`: `Task::new(name, &[&src..], &[&dst..], warp_idx, num_warps, schedule).setmaxnreg_increase(n)`, `TaskManager::new(tasks, ResourceDependencyGraph::new().depends_on(&r, [&up]))` owning its resources (`.resource(r)`, `.iket_enable_profiling(true)`, `.bind(name, value)` for symbolic bounds), `verify()` (also without a kernel: warp ranges, roles, acyclic and backed graph, pipeline sides, commit/release balance across loop nests, cooperative-group thread counts, variable flow along data edges, PDL wait/launch ordering, including through pipelines) returning every `VerifyIssue`, `setup_resources_and_tasks()` (one barrier publication for all pipelines), `run()` (warp dispatch, task-local `Var` storage, pipeline states, `for_range` / `while_carry` loops, skippable tiles, iteration guards, producer tails, profiling scope and IKET range per task and step), `TaskManager::verify_interleavings(&kernel, cfg, args, seeds)` reporting deadlocks and races with task, step and schedule line. Examples `ts_copy_grid_stride`, `ts_copy_persistent_skip_tile` (simulator and SM89 GPU); a two-task pipelined copy; a PDL pair through `SimStream` and sm_90a ptxas. Seeded rather than exhaustive interleavings |
| `cutlass.experimental.task_scheduling` part 3: `memory.SmemAllocation(name, size_bytes, alignment, dtype, count)` / `TmemAllocation(name, num_columns)` with `.offset`, `SmemAllocator` (`add`, `add_resource`, `add_pipeline_group`, `add_alias_group`, `compute_layout`, barrier placement into `PipelineConfig.barrier_ptr`), `TmemAllocator`, `ResourceContext(smem_base, tmem_ptr_i32)`, `TaskManager(..., smem_allocator=)`; `PipelineGroup(name, members, mode)` with `PipelineGroupMode.{Merge, Fork, FusedMerge}` and `group.release()`; `pipeline.TSPipelineTmaAsync` (task-warp-leader arming), `TSPipelineAsyncUmma`, `TSPipelineUmmaUmma`; tutorials `02_copy_tma.py`, `07_group_pipeline_ts/01_merge_copy.py` | `cute::task_scheduling`: `SmemAllocation::new::<T>(name, count, alignment)` / `SmemAllocation::bytes(name, n)` and `TmemAllocation::new(name, cols)`, whose offsets are shared by clones (`offset()`, and `ptr::<T>(&stage_info.context)` for the typed view in a kernel); `SmemAllocator` (`add_resource`, `add_pipeline_group`, `add_alias_group(phases)`, `compute_layout()` / `plan()` returning an `SmemLayout`: largest alignment first, alias groups sharing one block, then the barriers of pipelines without storage and of groups; capacity and alignment errors) and `TmemAllocator` (columns, power-of-two allocation, 512-column capacity), given to `TaskManager::smem_allocator` / `tmem_allocator`: the prologue allocates the unified block, patches the barrier pointers and allocates TMEM from warp 0 (`ResourceContext::smem_base` / `tmem_ptr_i32` in every work call), `run()` frees TMEM after every task. `PipelineGroup::new(name, &[&a, &b], PipelineGroupMode::Merge)` (`with_pipeline_config` for explicit counts) is a barrier-only resource: a merge's consumer waits on each member and `group.release()`s once, a fork's producer `group.commit()`s once, a fused merge shares both barriers with summed producer counts; capture, verification (pipeline balance through the group) and emission understand groups. `TsPipeline::create_tma_async` / `create_async_umma` / `create_umma_umma` (also `ResourcePipeline::Ts`, which `UmmaUmma`, `AsyncUmma`, and `TmaAsync` resources with `TASK_WARP_LEADER` producers or a consumer group counting the consuming threads become). New checks: `SmemAlias` (shared bytes whose owners no control-edge path orders, naming both), `GroupArrivalCount`, `TmemCapacity`, `BarrierStorage`, `BadGroup`, `SmemLayout`. Examples `ts_copy_tma`, `ts_merge_copy` (simulator, sm_90a MLIR/PTX, ptxas); a TMEM + `TSPipelineAsyncUmma` / `TSPipelineUmmaUmma` hand-off GEMM (simulator, sm_100a ptxas). One CTA only for TS pipelines and groups (no 2-CTA commits or multicast masks) |
| `cutlass.experimental.task_scheduling` end to end: the tutorial `02_gemm_simple_ts/01_fp16_bf16_gemm_3.py` (`GmemAbResource` / `SmemAbResource` (`TmaUmma`) / `TmemCResource` (`UmmaAsync`) / `GmemDResource`, `LoadTask` / `MmaTask` / `StoreTask` / `PaddingTask` with `num_registers=`, `PipelineConfig.create_tma_umma_pipeline_cfg` / `create_umma_async_pipeline_cfg`, `SmemAllocator` / `TmemAllocator`, `prims.Tcgen05SmemDesc.build`, `Tcgen05InstrDesc.build`, `tcgen05_mma`, `tcgen05_ld`, `elect_sync`, `prefetch_tensormap`); constexpr work arguments (`load_subtile(subtile_idx=i)`) | Example `ts_gemm_simple` (FP16 and BF16, `128x256` tiles, 3 A/B stages, 2 accumulator stages): the TMA loads are `cpasync::make_tiled_tma_atom` over the 128-byte-swizzled layout, `build_descriptors` returns `prims::Tcgen05SmemDesc` values as task-local `u64`s, `mma` issues `prims::tcgen05_mma` with a `Var<bool>` `scale_d`, `load_subtile` returns a `Vector<f32>` (`tcgen05.ld.32x32b.x32`), and `M`, `N`, `K` may be ragged (TMA zero fill, guarded stores; `K` a multiple of 8). The TaskManager's TMEM allocator replaces the tutorial's explicit `tcgen05_alloc` / barrier / `tcgen05_dealloc`, and TmemC's barriers are placed by the `SmemAllocator` (cute-rs requires it). Gaps the port closed in the TS layer: `Work::constexpr_input(name)` with `ArgSource::constexpr(i)` and `WorkArgs::constexpr(name)`, `Vector<T>` task-local values, and register budgets on tasks smaller than a warpgroup when the warpgroup's tasks agree (one `setmaxnreg` per warpgroup before the dispatch). Verified on the simulator (bit for bit on small integers, a ragged size, within the FP32 summation bound on random data), `TaskManager::verify` and seeded interleavings, an IKET trace with one range per task step, and sm_100a PTX (`tcgen05.mma`, `tcgen05.ld`, `cp.async.bulk.tensor`, `setmaxnreg`) assembled by `ptxas` (CUDA 12.8); no hardware run (no Blackwell GPU) |

## Kernel arguments

A kernel's parameters can be declared one by one (`param_tensor`,
`param_scalar`, `param_extent`, ...) and passed as a flat list of
`KernelArg`s, or through the kernel argument protocol in `cute::params`,
which keeps the two sides in step. A type implementing `KernelParam`
declares its parameters from a build-time spec, and checks and packs a
host value against the same spec at launch:

```rust
use cute::prelude::*;

// One runtime extent shared by both tensors, a multiple of 4.
let n = sym_int32(4, Some("n"));
let t = make_fake_compact_tensor::<f32>((n,), None, false);
let k = Kernel::builder("axpy").block_dim([64, 1, 1]).jit(
    (ScalarSpec::Plain, t.clone(), t),
    |a: Float32, x: Tensor<f32>, y: Tensor<f32>| {
        let i = block_idx().x * 64 + thread_idx().x;
        if_(i.lt(size(&x, &[0])), || y.store(i, a * x.load(i) + y.load(i)));
    },
);
assert_eq!(k.param_names(), ["arg0", "arg1", "arg1_shape_0", "arg2"]);

let host = |v: Vec<f32>| HostTensor::from_slice(&v, Layout::col_major((v.len(),)));
let mut args = (2.0f32, host(vec![1.0; 8]), host(vec![0.5; 8]));
k.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
assert_eq!(args.2.to_vec::<f32>(), [2.5; 8]);
// A length that breaks the divisibility is refused before the launch.
let mut bad = (2.0f32, host(vec![1.0; 6]), host(vec![0.5; 6]));
assert!(k.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut bad).is_err());
```

* Parameter names are built from the name and each field's or leaf's path:
  a tensor `a` is the buffer `a` followed by its dynamic leaves `a_shape_0`,
  `a_stride_1`; element `i` of a `Kernel::jit` signature is `arg{i}`, a
  struct's field `f` is `f` (or `{name}_f` under a name). A `SymInt` is one
  parameter, named where it first appears. `kernel.param_names()` lists them.
* The kernel's IR stays the single list of parameters: `TypedKernel::pack`
  (and `cute::pack_args`) checks that the packed arguments are for exactly
  those parameters, in order, before the simulator or the CUDA launcher sees
  them; both then check the flat list as before.
* At launch the host value is checked against the spec, naming the
  parameter: the element type, static extents and strides, the
  divisibility and width of each dynamic leaf, one value per `SymInt`, the
  buffer's length, the alignment a `HostTensor` promises
  (`with_assumed_align`) against the spec's, `Some` against `None`, tuple
  and `Vec` arities, and constants.
* The ABI is buffers plus scalars, as before: structs and native structs are
  flattened into scalar parameters (by-value structs and descriptors come
  with kernel-argument marshaling).

## How it works

```text
Rust kernel code ──► builder ──► IR ──► validate ──┬──► simulator (CPU, checking)
 (layouts, tensors,   (thread-     (structured,      │
  atoms, Value ops)    local)       SSA)             └──► MLIR ──► mlir-opt ──► PTX ──► CUDA driver
                                                                (two stages,
                                                                 hooks between)
```

* **Layout algebra runs at build time,** as it does in CuTe. Each leaf of a
  shape, a stride or a coordinate is a static integer or a runtime value,
  printed `?` as in CuTe DSL: an `IndexExpr`, a symbolic integer expression
  that folds constants and simplifies `div`/`mod` against known strides and
  bounds. The algebra works on layouts such as `(?,?):(?,1)`, following
  CuTe's rules for dynamic leaves. Composition and the divides take the
  dynamic extents of the layout being divided to be large and divisible, so
  `zipped_divide` of an `(M, K)` layout by `(128, 32)` has static `(128,32)`
  tiles and rest modes of extent `ceil_div(M, 128)` and `ceil_div(K, 32)`,
  and dividing all of an `(M, N)` layout by `32` gives rest modes of extent
  `ceil_div(M, 32)` and `N`. A runtime extent or stride of the right-hand
  side (a tiler's, say) is not assumed to be anything: where it meets an
  extent it is not provably a multiple of, the extents are computed at run
  time with a `min` and a `ceil_div`, as in C++ CuTe, so `(4,8):(8,1) ∘ m:1`
  is `(min(4,m),ceil_div(m,4)):(8,1)`. Coalesce merges only what it can
  prove; complement and the inverses follow static strides. Only what can't
  be resolved becomes instructions. `layout.size()` is a `usize` for loops
  unrolled in Rust, `size(&layout, &[])` an `IndexExpr`.
* **Tensors take runtime extents too,** as CuTe DSL's framework tensors do.
  A tensor parameter of launch-time extents,
  `param_tensor("a", Layout::row_major((m, k)))` with `m` and `k` from
  `param_extent` (or `make_tensor` on a pointer), tiles with `local_tile`
  into tiles of the tiler's static extents and rest modes of runtime extent
  that count the tiles, and partitions among threads with tiled copies and
  MMAs. A `for_range` walks a rest mode, its trip count the mode's size
  (`size(&tiles, &[2])`), and the copies and MMAs inside work on one tile.
  What the kernel unrolls when it is built stays static and is refused,
  naming the leaf, when it meets a runtime extent: the per-thread chunks and
  rest modes of copies and MMAs, register fragments, loops over every
  element, and shared memory. Offsets are `IndexExpr`s as before, so the
  simulator's bounds checks find accesses past a runtime extent, and a
  partial tile needs a predicate. Host tensors carry their runtime shapes
  as CuTe DSL's framework tensors do: `HostTensor::mark_layout_dynamic` and
  `mark_compact_shape_dynamic` make modes dynamic (with a divisibility),
  `Kernel::jit_for` builds the kernel for the resulting specs, and every
  launch packs the buffers with their extents and strides, so one compiled
  kernel serves every size the specs admit. `domain_offset` shifts tiles and
  coordinate tensors by runtime amounts (the negative K residue too),
  `elem_less` compares against runtime extents, and `max_alignment` counts
  what is known to divide them (`ampere_sgemm` and `elementwise_add`'s
  `dynamic` show all of it). The static-upper-bound idiom, a static bound
  as the extent with runtime strides
  ([`Layout::with_strides`](src/layout/layout.rs)), is now optional; it
  still works, with the same code (`runtime_gemm` shows both).
* **Values are SSA.** Arithmetic on `Value<T>` records IR through a
  thread-local builder, so operators like `a * b + c` work. Register
  fragments are build-time arrays of SSA values, so they must be indexed
  statically, like CuTe register tensors. State that changes inside
  runtime control flow lives in a `Var`, a mutable variable that the
  builder puts into SSA form as the kernel is built (in the style of Braun
  et al.): a `Var` touched in a loop body becomes a loop-carried value
  (`scf.for` `iter_args`, or the arguments of an `scf.while`), and one set in
  a branch becomes a result of the `scf.if`, through every level of nesting.
  This is what CuTe DSL's preprocessor does for the variables assigned in a
  dynamic region. A `Var` holds anything implementing `Carry` (values,
  tuples, register tensors, structs declared with `carry_struct!`), and its
  type cannot change: neither its flattened types nor the layouts of the
  register tensors in it. State can also be threaded explicitly, through
  `for_range_carry`, `while_carry` and `if_else_yield`, which check the
  same. Either way, a value computed (or read from a `Var`) inside a region
  belongs to it. A register fragment is not carried by itself: one created
  outside a runtime loop that the loop reads and then writes is reported,
  because the write would not reach the next iteration (CuTe DSL's register
  tensors are memory); keep it in a `Var`.
* **The builder cleans up.** Pure operations are CSE'd across enclosing
  regions and dead code is removed. Each instruction remembers the
  `#[track_caller]` location of the user code that created it.
* **The validator** checks SSA scoping (a value computed inside an `if_` or
  loop body can't leak out), types and instruction constraints, such as a
  named barrier's id (0 to 15) and thread count (a multiple of 32, at most
  the block), and that every access's buffer is declared aligned to the
  access's size (the builder proves only the offset from its start, and
  PTX requires each access to be aligned). Compiling for an architecture
  checks that every instruction
  exists there, naming the first that does not with its source line, and
  raises the PTX ISA version when an instruction needs a newer one than the
  architecture's minimum (`mbarrier.test_wait.parity` needs 7.1 on sm_80,
  `max.xorsign.abs` 7.2 on sm_86). Architecture-specific instructions, which
  newer architectures do not inherit, list the targets that have them
  ([`validate::ArchSpecific`](src/validate.rs)): Float32 `redux.sync` is
  refused for anything but sm_100a with "redux.sync.max.f32 needs sm_100a
  (or the sm_100f family with PTX 8.8); the kernel is being compiled for
  sm_100". A requirement is a set of alternatives, each an exact chip
  (`sm_90a`), a family (`sm_100f` or a later chip of its family, CuTe DSL's
  `is_family_of`) or a baseline (`sm_100` or newer), with its own minimum
  PTX ISA version; the target gets the version of the alternative it
  satisfies (the FP4 `cvt` is PTX 8.6 on sm_100a, 8.7 on sm_120a and 8.8
  on sm_103f), and a refusal lists them all. `validate::SM90A_ONLY` and
  `validate::SETMAXNREG` are the exact-chip lists m8's WGMMA and
  `setmaxnreg` will use.
  Kernels with more than 48 KB of shared memory are compiled to allocate it
  dynamically, and so are kernels with a dynamic region (`get_dyn_smem`),
  whose size the launch sets (`LaunchConfig::with_dynamic_smem`): one layout,
  `KernelIr::shared_memory_layout`, places the buffers sized at build time
  and then the region in the dynamic window for the MLIR emitter, the CUDA
  launcher and the simulator. The validator bounds the build-time part;
  launches check it plus the region's bytes. The window is an external
  array, `__cute_dynamic_smem` (`.extern .shared .align 16 .b8
  __cute_dynamic_smem[];`, as aligned as its most aligned buffer), so such
  a kernel has no static shared memory. MLIR's `gpu.dynamic_shared_memory`
  would leave one byte: its global is internal, and LLVM's NVPTX backend
  demotes it into the kernel as a `.shared .b8`, which counts against the
  48 KB and opt-in limits and in `%total_smem_size`.
* **`cute::math` needs no libdevice.** CuTe DSL lowers `cute.math` functions
  in their default, precise mode to libdevice calls, which need the CUDA
  toolkit. cute-rs expands each one at build time into adds, multiplies,
  fmas, divisions and square roots rounded to nearest, comparisons, selects
  and integer operations, ported from FreeBSD's msun (fdlibm). Every
  operation is exactly rounded, so the GPU computes the simulator's bits:
  `tests/gpu.rs` checks every function on every type (the unary ones on
  every bit pattern of Float16 and BFloat16). The accuracy, measured against
  a double-double reference, is at most 1 ulp for most functions, 1.5 for
  `sin`, `cos` (Float32) and `atan2` and 2.5 for `erfc`, as good as or
  better than libdevice's documented bounds; the table is in
  [`src/math/mod.rs`](src/math/mod.rs). `fastmath=True` uses the hardware
  approximations (`ex2.approx.ftz.f32` and so on), as CuTe DSL and
  `-use_fast_math` do, each within a documented bound, and the
  `approx=`/`ftz=`/`rounding=`/`full=` options choose one PTX instruction,
  as in CuTe DSL; the simulator computes directed roundings exactly. `abs`,
  `-x`/`neg` and `copysign` on floats are bit operations on the sign
  (`and` and `xor` in inline PTX), so NaNs keep their payloads on the GPU
  as in the simulator: as integer operations or `fabs`/`fneg`, LLVM would
  emit `abs.f32` or `neg.f32` in some places, which return the canonical
  NaN.
* **CuTe DSL's integer and bit instructions are single PTX instructions.**
  `bfe`, `bfi`, `lop3`, `shf`, `prmt`, `mul_wide`, `mul24`, `mad24`,
  `add_sat_int` and `sub_sat_int` are each one instruction of inline PTX
  without side effects, so LLVM may CSE, hoist or delete them like any
  arithmetic, and the simulator runs them from the PTX ISA's definitions
  (`mul_hi` is LLVM's own `mul.hi`). The GPU test found two ways in which
  `bfe` and `bfi` on an SM89 GPU depart from the ISA, which takes the low
  byte of the start and length: the 64-bit forms use all 32 bits, and
  `ptxas` folds a constant `bfe.s32` that starts past bit 31 to -1 and a
  constant `bfi.b32` field longer than 31 bits as if its length were taken
  mod 32, where the hardware follows the ISA. So cute-rs masks the 64-bit
  forms' start and length to their low byte, clamps a `bfe.s32` start to 31
  (the same field: the replicated sign bit) and cuts a `bfi.b32` field at
  the msb, in arithmetic that LLVM folds when they are constants. The
  start and length of `bfe.u32` and the length of `bfe.s32` reach `ptxas`
  unchanged: the hardware and `ptxas`'s folding both take their low byte,
  also when it is 0 and higher bits are set (0x100, 0x200). A one-off
  sweep on the GPU then gave the ISA's result for every start and length
  from 0 to 255 (and some larger values), as constants, as runtime values
  and one of each; `tests/gpu.rs` keeps the edge cases.
* **Carry chains go through PTX's carry flag.** `add_cc`, `addc`, `sub_cc`,
  `subc`, `mad_cc`, `madc` and the `_cc` carry-in-and-out forms are each one
  instruction of inline PTX with side effects, as in CuTe DSL: the flag
  (`CC.CF`) is state no SSA value shows, and LLVM neither deletes nor
  reorders side-effecting inline assembly, while nothing else it emits
  touches the flag (only these instructions can, and NVPTX uses them for
  integers wider than 64 bits alone, whose arithmetic the validator keeps
  out of a chain). The builder never merges or removes
  them either. PTX keeps the flag only through straight-line code, so the
  validator rejects a carry-in instruction with no carry-out before it in
  its block, or with a runtime `if`, a loop, a `printf` (a call), a
  hand-written MLIR operation, inline PTX that uses the flag or branches,
  or a 64-bit integer division or remainder by a runtime value between
  them (LLVM lowers that division, from O1 on, with a branch to a 32-bit
  division for when both operands fit; one by a constant becomes a
  multiplication), naming both lines, and a subtraction reading an
  addition's carry or the reverse (see Limitations). The warp collectives
  (shuffles, with constant or runtime segment masks and clamps, votes,
  `redux.sync`, `match.sync`, `sync_warp_with`) are each one convergent
  instruction that neither branches nor touches the flag, so a chain runs
  through them. It also runs through the special-register reads (the
  shared-memory sizes, a block's rank in its cluster and the cluster's size
  among them; `dynamic_smem_size` then subtracts the region's start with
  ordinary arithmetic) and `cp.async.mbarrier.arrive`, single instructions
  that neither branch nor touch the flag either. The simulator keeps a
  carry flag per thread, in program order, from the PTX ISA's definitions,
  and forgets it where the validator's rule says PTX loses it (one rule
  serves both), and `tests/mlir.rs` checks in the PTX of the carry feature
  kernels, at O0 to O3, that the only carry instructions are the kernel's
  own, in its order, each carry-in after a carry-out in the same stretch of
  straight-line code.
* **Hand-written PTX names its operands.** `InlinePtx` takes CuTe DSL's
  `{$wN}`, `{$rN}` and `{$rwN}` names and translates them to LLVM's
  positional `$N` (write-only outputs, then read-write ones, then the
  inputs), escaping any other `$` (a PTX label), and emits `llvm.inline_asm`
  rather than `nvvm.inline_ptx`: LLVM 21's `nvvm.inline_ptx` does not
  rewrite named operands (`mlir-opt` aborts with "Bad $ operand number"),
  and LLVM 23's misnumbers a statement with both read-only and read-write
  operands. The operand classes are the ones LLVM 23's lowering of
  `nvvm.inline_ptx` gives CuTe DSL: each operand's constraint comes from its
  type, except that a read-only integer constant is the immediate `n`
  (`lop3`'s table must be one; LLVM prints it sign-extended), a Float32
  constant stays an `f` register (NVVM's `n` is invalid there, which CuTe
  DSL works around with a `mov.f32`) and a Boolean constant a `b` register
  (LLVM would print `true` as -1); a read-write operand is an output plus
  an input tied to it; a predicate becomes the template's `@$n` guard; a
  `Ptr` becomes its address, computed as the mbarrier operations' are: 64
  bits for global memory and register arrays, and 32 bits for shared
  memory (`r`, as CUTLASS's `cast_smem_ptr_to_uint` gives it, which
  `ld.shared` takes and `mapa.shared::cluster.u32` needs). The kernel build
  checks that every name has an operand and every operand a name.
  `InlineAsm` keeps a hand-written constraint string, which the validator
  checks against the operands (count, order, empty entries, that each code
  is one NVPTX takes, each type's register class, tied types, that
  immediates are integer constants, every `$n`): LLVM 21 would reject the
  rest or abort, and LLVM 23's `mlir-opt` prints LLVM's error for a
  constraint NVPTX does not know but exits with success, leaving the
  statement out of the PTX, so a `Toolchain` also fails when `mlir-opt`
  prints an error of LLVM's (a line starting `error:`; MLIR's own located
  diagnostics are left to the exit status, since MLIR prints one for a
  conversion it then does another way). A memory clobber (from
  `.memory_clobber()` or in the constraint string) also gives the
  statement side effects, because MLIR models inline assembly's effects
  only through them and would delete a store without outputs. `MlirOp`
  takes values and pointers, which become LLVM pointers in their buffers'
  address spaces. A custom operation, CuTe DSL's
  `@dsl_user_op`, is a Rust function marked `#[track_caller]` that builds
  these (or any other operations): everything it builds records the line
  that called it, so the validator's and the simulator's errors and the
  PTX line information point there. Closures cannot be `#[track_caller]`,
  so what it builds inside an `if_` or loop body records its own lines
  ([`cute::arch::inline`](src/arch/inline.rs) has an example).
* **Atomics and reductions are PTX `atom` and `red` with explicit semantics
  and scope.** Each is one instruction of inline assembly with side effects,
  `atom.relaxed.gpu.global.add.u32` or `red.release.cta.shared.add.noftz.f16x2`,
  never an LLVM `atomicrmw` or `cmpxchg`: LLVM 21 drops `atomicrmw`'s
  ordering and scope and `cmpxchg`'s scope, which LLVM 23 keeps, so the
  memory semantics of one kernel would depend on the installed LLVM.
  `tests/mlir.rs` checks that every installed `mlir-opt` emits the same
  instructions. The validator holds each to PTX's tables (no
  `red.max.f32`, no acquire `red`, the bit types for the bit operations)
  and refuses BFloat16 and cluster scope below sm_90. The simulator treats
  each element (each half of a packed pair) as a strong read and write:
  atomics of the same size never race with each other (atomics of different sizes on overlapping bytes do) but do race with plain accesses; a release
  atomic publishes its warp's clock, an acquire one joins what it reads,
  and later atomics continue the release sequence. Float adds round to
  nearest even; a Float32 add on global memory flushes subnormal inputs and
  results to zero (PTX's `atom.add.f32` without `.noftz`), on shared memory
  it keeps them, as do the half-precision adds (`.noftz`); the SM89 GPU
  matches bit for bit.
* **Ordered and cache-hinted accesses carry one set of memory attributes.**
  `load`, `store` and the G2R and R2G copy atoms put them on the IR's loads
  and stores: the ordering (weak, relaxed, acquire, release, seq_cst,
  volatile, MMIO or constant) and its scope, an L1 eviction priority, a
  cache operator, an L2 prefetch size, the non-coherent cache and an L2
  cache policy. The validator holds every access to PTX's rules for `ld` and
  `st` (which forms combine, which need global memory, and the SM and PTX
  version each needs; `ptxas` agrees with each rule). A scalar relaxed,
  acquire, release, seq_cst or volatile access without hints is an LLVM
  atomic or volatile load or store with a sync scope, which LLVM orders and
  lowers to `ld.acquire.gpu` and so on; everything else is PTX `ld` or `st`
  of inline assembly with side effects. A weak access is one instruction,
  with narrow elements packed into 16- or 32-bit words. A strong one keeps
  its element type (`ld.acquire.gpu.global.v4.b8`), since PTX's memory
  model treats each element of a vector access as an access of its own,
  which another thread's access must cover exactly to synchronise with it
  or not race; PTX has no vector of more than four 8- or 16-bit elements, so
  a strong vector of eight or sixteen takes one instruction per four. PTX
  has no seq_cst load or store, so seq_cst is
  `fence.sc` and then an acquire load or a release store, as LLVM maps it,
  and `MemoryOrder::Constant` is a non-coherent load (`ld.global.nc`), as
  `invariant` is (CuTe DSL's own lowering of `SC` and `CONSTANT` is not
  public). `prefetch` is inline PTX on every target, because LLVM 21 to 23
  select their prefetch intrinsics only for sm_90 with PTX 8.0.
* **CuTe DSL's packed half-precision operations and narrow conversions are
  single PTX instructions too,** on pairs of Float16 or BFloat16 values packed
  into 32 bits (`add.rn.ftz.sat.f16x2`, `fma.rn.relu.bf16x2`,
  `ex2.approx.f16x2`, `cvt.rn.satfinite.bf16x2.f32`, `cvt.rna.tf32.f32`,
  `dp4a.s32.s32`, ...), as pure inline PTX. The simulator computes them from
  the PTX ISA's definitions with exact arithmetic: each result is the exact
  value rounded once (also for `fma` on Float16 and BFloat16 values, which
  is `fma.rn.bf16` on the GPU rather than an f32 fma rounded twice), then
  `.ftz`, `.sat` and `.relu` as the ISA describes them; an SM89 GPU gives the
  same bits for every Float16 value against a set of partners, with every
  option, and for every BFloat16 value in fma. `.satfinite` clamps
  infinities and other magnitudes above the largest finite BFloat16 to that
  value with their sign (`0x7f7f`, `0xff7f`), as the ISA specifies. Where
  the PTX ISA is silent the simulator follows that GPU: NaN results are the
  canonical NaN (`0x7fff`), `.sat` and `.relu` turn `-0` into `+0`,
  `.satfinite` keeps NaN as `0x7fff`, and `cvt.rna.tf32.f32` clears a
  NaN's low 13 bits (a NaN with its payload only there becomes an
  infinity). A predicated operation (`predicate=`) does not run where the
  predicate is false, and the result there is its first operand, the
  instruction's result being tied to it (CuTe DSL leaves it undefined).
  `ex2.approx.f16x2` is approximate: the simulator gives the correctly
  rounded value, and the GPU one of the two Float16 values on either side
  of the exact one, so within 1 ulp (0.9999 ulp at worst on SM89, over
  every input). Pairs and words of narrower values are built and taken
  apart with `pack` and `unpack` (`Op::Pack`, `Op::Unpack`): a vector of the
  values bitcast to the whole, which `mma.sync` on BFloat16 uses too.
* **Compiling** runs MLIR's `gpu-lower-to-nvvm-pipeline` as the passes it
  expands to, which a `Toolchain` asks its `mlir-opt` for
  (`--dump-pass-pipeline`) once per target and optimization level, since
  LLVM 21, 22 and 23 differ in their pass options; if that output cannot be
  parsed, MLIR 21's list is used. The
  passes run in two `mlir-opt` invocations split at the LLVM dialect: the
  first ends with the kernel as `llvm.func` inside its `gpu.module`, the
  second runs `gpu-to-llvm` and `gpu-module-to-binary`. Stage hooks can edit
  the module in between ([`LoweredModule`](src/backend/stage.rs)). At default
  options the PTX is exactly what the single pipeline produces
  (`Toolchain::compile_ptx_reference` keeps it for comparison). An LLVM
  that predates the PTX ISA version a kernel needs (9.1 needs LLVM 23)
  ignores the request and writes its own `.version`; compiling refuses that
  PTX, naming the version needed and the `mlir-opt`.
* **Copies check what they emit.** A vectorised copy (`Universal { bits:
  128 }`, `cp.async`, `ldmatrix`) is emitted only when the per-thread chunk
  is provably contiguous and aligned, including under a swizzle. Otherwise
  the kernel build fails with a message saying why.
* **The simulator** runs each warp of a block (or block cluster) on its
  own, with per-lane masks for divergent code, over byte-addressed memory.
  A warp's state is a continuation over the structured IR (a stack of
  regions, each with the index of its next instruction and the lanes
  running it), and a deterministic schedule (`SimOptions::schedule`)
  interleaves the warps: `Schedule::RoundRobin` (the default) gives every
  warp that can run one instruction in turn, ordered by block and warp,
  and warps that run the same code execute together, as if in lockstep;
  `Schedule::RunToBlock` runs one warp until it blocks, exits or yields
  (at a failed `mbarrier_test_wait` and at every `while_carry` back-edge,
  so spin loops let the others run), the most sequential interleaving.
  `Schedule::Random { seed }` draws the next group to run at every
  scheduling point (each shared or global memory access, atomic, barrier,
  mbarrier and cluster-barrier operation, fence, asynchronous issue and
  wait, opaque inline statement and `printf`, and each `while_carry`
  back-edge) from a SplitMix64 generator, and with the same seed a
  blocking `mbarrier_try_wait` sometimes returns false at once (PTX's time
  limit); `AsyncCompletion::Random(seed)` completes copies in flight at
  scheduling points too, and a failed `mbarrier_test_wait` then forces
  nothing. The same seed replays the same run: `SimReport` and `SimError`
  record the seed and the step count, and a failure's message says to
  simulate again with that seed (`SimOptions { schedule: Schedule::Random
  { seed }, async_completion: AsyncCompletion::Random(seed), .. }`) to
  replay it. `Kernel::simulate_all_orders` runs a kernel under every
  built-in schedule and requires the same results, and
  `Kernel::simulate_schedules(cfg, args, opts, seeds)` adds a seeded run
  per seed and reports the first output element that differs, with the two
  schedules (`` `out`[0] is 1 under Schedule::RoundRobin with
  AsyncCompletion::Late and 2 under Schedule::Random { seed: 3 } ``), which
  catches atomics and flag protocols whose result depends on timing; the
  example and feature suites run that way (`CUTE_SIM_SEEDS` seeds, 20 by
  default). Warps diverge with independent thread scheduling at the
  granularity of structured control flow: a divergent `if_` splits the
  warp's convergence group into a then-group and an else-group, each
  scheduled on its own (round-robin runs the then-group first and switches
  when it blocks or spins, as a SIMT stack would until it has to), and they
  rejoin at the end of the `if_`; lanes that leave a loop early wait at its
  end. So lane 0 can wait on an mbarrier that lanes 1 to 31 of its own warp
  arrive at in the else-branch. The `.aligned` collectives (`bar.sync`,
  `bar.arrive`, `sync_threads`, `mma.sync`, `ldmatrix`, the aligned cluster
  barriers) need every lane of the warp in one group, and report otherwise;
  the `.sync` ones with a member mask (`shfl.sync`, `vote.sync`,
  `match.sync`, `redux.sync`, `bar.warp.sync`) rendezvous: a group waits
  until every lane of the mask has reached a collective of the same kind,
  possibly in several groups and at different instructions (as sm_70
  allows), then it executes once for all of them, and a deadlock report
  names the lanes of the mask that are stuck elsewhere. The unaligned
  `barrier_arrive_unaligned` counts a warp's arrival once all its lanes have
  executed one for that barrier, from any instruction. Vector clocks stay
  per warp, so accesses within a warp remain synchronous for the race
  checker. `sync_threads` and `bar.sync` block a warp until the barrier
  completes, and `mbarrier_try_wait` until its phase completes (or, when
  nothing else can run, until it gives up and returns false, as PTX allows
  it to), so warp-specialised kernels, consumers that wait before their
  producers arrive and spin-waits on `mbarrier_test_wait` or on a flag run
  as on hardware. When every warp that has not exited is blocked (an
  `mbarrier_try_wait` that has given up 64 times in a row with nothing
  visible happening in between counts as blocked), the simulator reports a deadlock:
  each blocked warp (block and warp), the instruction and line it waits
  at, the loops and branches it stands in (`in for_range at k.rs:40,
  iteration 3; in the then-branch of if_ at k.rs:38`), and what it waits
  for: a barrier's arrived and expected warps and where each missing warp
  is (blocked elsewhere, at which line, or exited; a counted barrier that
  only exited warps could complete is flagged, since PTX's exit rule may
  release it but it is almost always a bug), or an mbarrier's phase, the
  parity naming it, the arrivals left, the tx-count and the bulk copies in
  flight. When nothing visible (a memory write, an atomic, an arrival, a
  barrier completing, a copy landing, a warp exiting, a `printf`) has
  happened for `SimOptions::max_idle_steps` warp steps and every warp is
  blocked or spinning (looping while it polls an mbarrier or an ordered
  load; a loop that only computes in registers does not count), it
  reports a hang instead, naming the spinning
  warps, their loops and what they poll (an mbarrier phase, or the address
  of an ordered load); `max_while_iterations` stays as a per-loop
  backstop. The cluster barrier (`cluster_arrive`, `cluster_arrive_relaxed`,
  their `_aligned` forms, `cluster_wait` and `cluster_sync`, lowered to the
  convergent `nvvm.cluster.arrive` / `.wait`; inline PTX holding
  `barrier.cluster` acts the same in the simulator) follows PTX's
  `barrier.cluster`: every thread of the cluster that has not exited
  arrives once per phase (the non-aligned forms let a warp's threads arrive
  one by one), a wait blocks until all have, arrivals release and waits
  acquire at cluster scope (`.relaxed` orders nothing by itself, but a
  fence before it makes a release pattern, as in the PTX ISA's example), and a thread that
  arrives twice without waiting, waits without arriving, or executes an
  aligned form with part of its warp is reported. Vector clocks span the
  cluster (a slot per warp of each CTA, plus reserved slots per CTA for
  agents such as the bulk-copy engine), and shared-memory reads are kept
  as FastTrack-style read epochs (one per byte, a small set only while
  unordered readers coexist), so the checks scale to 16 CTAs of 32 warps
  and reports name the CTA when it differs (`warp 3 of CTA 1`). A
  releasing arrival on a peer CTA's mbarrier at cluster scope publishes
  the arriving warp's clock to that phase's waiters; a CTA-scope or relaxed
  one publishes nothing to them. A peer's arrive or `expect_tx` on an
  mbarrier is reported unless the initialisation was published to it:
  `mbarrier_init_fence` by the initialising thread, then a cluster barrier
  (a relaxed arrival will do, as in the PTX ISA's pattern) or a
  release/acquire chain. A remote operation on a CTA that has exited, and a
  CTA exiting with asynchronous copies into its shared memory still in
  flight, are reported. `SimOptions::resident_clusters` runs several
  clusters (or the whole grid, `ResidentClusters::Grid`) together, the rest
  waiting in a queue, so cooperative grids and persistent kernels with
  flags between clusters run; a hang that polls global memory while
  clusters are queued suggests raising it. Clocks stay per cluster.
  `ldmatrix` and `mma.sync` are implemented from the PTX ISA's fragment
  tables, independently of the layout algebra, so a correct simulated GEMM
  is real evidence that the layouts are right. The warp collectives follow
  the PTX ISA's pseudo-code (`shfl.sync`'s segment mask and clamp,
  `redux.sync`'s wrapping sums and NaN rules, `match.sync`, `min` and
  `max` with `.xorsign.abs`), and each (shuffles, votes, `redux.sync`,
  `match.sync` and `sync_warp_with`) checks what the ISA leaves undefined:
  a lane executing one outside its member mask, a lane of the mask that
  does not execute it with the same mask, and a shuffle reading a lane
  outside the mask or one that is not active. The lanes a block's short
  last warp lacks count as exited, as the ISA says. Races in shared memory are
  found with a vector clock per warp: `sync_threads`, named barriers
  (`bar.sync` acquires what `bar.arrive` released) and mbarriers (a
  successful wait acquires what the phase's arrivals released) order
  accesses. Each block has 16 named barriers, counted in whole warps.
  Bulk copies land when a thread waits (or is blocked) on their mbarrier
  after the phase's arrivals are in, and count down its transaction count:
  a phase that completes with copies still in flight, or copies that
  exceed what `expect_tx` announced, are reported, and the landed data is ordered only
  before threads that waited for the phase. `cp.async` copies that
  `cp_async_mbarrier_arrive` makes an mbarrier track land the same way (or
  at their thread's covering `cp_async_wait_group`, if that comes first, and
  never twice), and an mbarrier count that leaves no room for their
  arrivals, which would complete the phase with copies in flight, is
  reported. These copies share one model of in-flight asynchronous
  operations (the one TMA, WGMMA and tcgen05 plug into): each records its
  issuing thread, the agent performing it (a clock slot), its writes,
  background reads and register results, and what completes it (its
  thread's `cp.async` group, a `complete_tx` on an mbarrier, the arrive-on
  that `cp.async.mbarrier.arrive` owes). `SimOptions::async_completion`
  says when they complete: `Late` (the default) as late as anything allows,
  which exposes every premature access, or `Early`, at issue, which a
  correct kernel must also survive (copies landing before their
  `expect_tx`, for one); `simulate_all_orders` runs both. Overwriting
  memory an operation still reads, reading a register result before the
  operation completes, and a block exiting with an operation still writing
  or reading its shared memory are reported. The proxies are modelled:
  every shared-memory byte knows whether its last write was generic (a
  store, an atomic) or asynchronous (a bulk copy landing), and the async
  proxy (a bulk copy reading or overwriting shared memory) may only touch
  bytes the generic proxy wrote if the writing warp executed
  `fence.proxy.async` (unscoped or `.shared`) after the write and that fence
  is ordered before the issue (by a barrier or an mbarrier); a missing or
  unordered fence is reported, naming both lines. Bulk copies to global
  memory belong to their thread's bulk async-groups: the source is
  captured at issue but stays in use until the covering
  `cp_async_bulk_wait_group(n, true)` (or plain wait), so writing it
  earlier is a race, and the global bytes land at the covering plain wait
  (or when the block exits, once the source is read); reading or writing
  them before is reported. The dynamic shared-memory region gets the launch's bytes,
  checked for bounds and races like other shared memory. A parity wait is
  reported when the waiting warp has not seen the previous phase complete,
  since on hardware the parity could then name an older phase and pass at
  once, which the simulator's one schedule might not show; and when the
  mbarrier has completed two or more phases beyond the last one the warp
  knows of, the wait is reported as phase aliasing (the parity can no
  longer tell those phases apart: typically a producer that does not wait
  for an empty barrier), which a schedule that runs the producer ahead
  (run-to-block, or a seed) shows. Flags order
  accesses as the PTX memory model says: a release store publishes its warp's clock on the
  bytes it writes and an acquire load that reads them joins it; a
  `fence.acq_rel` before a relaxed store, or after a relaxed load, makes a
  release or acquire pattern of them; release sequences continue through
  atomics; and strong accesses (relaxed, acquire, release, volatile, MMIO
  and atomic) of the same bytes never race with each other, while plain
  accesses and strong ones of other sizes that overlap them (a mixed-size
  data race) race with them unless something orders them. As in PTX's
  memory model, each element of a vector access is one access, and an
  acquire synchronises with a release only if it reads exactly the bytes of
  one of the release's elements. A non-coherent load of memory that the
  kernel writes during the launch, or a write of memory such a load read,
  is reported. It reports bank-conflict wavefronts
  and global-memory sectors, so the effect of a swizzle or vectorisation is
  visible without a profiler.

## Compile options

`CompileOptions` is CuTe DSL's `CompileOptions`, with its three sources:

* builder methods, for CuTe's typed options:
  `CompileOptions::new().gpu_arch(Arch::Sm90a).opt_level(2).generate_line_info(true)`;
* `CompileOptions::parse`, for its option strings. It accepts CuTe's
  argparse flags (`--opt-level 2`, `--opt-level=2`, and unambiguous
  prefixes such as `--opt 2`) and the compact tokens of
  `CUTE_DSL_COMPILER_OPT` (`opt-level=2`, `preserve-line-info`,
  `cubin-chip=sm_80`, `warnings{nvvm}`, `iket`), separated by spaces or
  commas, with shell-style quoting. As with CuTe's argparse, a following
  word is a flag's value unless it looks like an option: a word that starts
  with `-` is a value only if it is a negative number (`--opt-level -1`) or
  contains a space (`--nvdisasm-options '-g -c'`), or if it follows
  `--ptxas-options` written out in full, which CuTe special-cases (`-v`).
  Otherwise write `--flag=-value`;
* the environment, through `CompileOptions::with_env` (or `from_env_with`
  and `with_env_from`, which take the lookup as a function, so tests need no
  `std::env::set_var`). The variables are `CUTE_RS_` plus CuTe's suffix, so a
  machine that also has CuTe DSL installed is unaffected. Explicit options
  override the environment. That is a deliberate difference from CuTe DSL,
  which applies `CUTE_DSL_COMPILER_OPT` on top of the explicit options and
  lets `CUTE_DSL_LINEINFO=1` switch line information on even over an
  explicit `GenerateLineInfo(False)`: in cute-rs, options a program sets
  cannot be changed from outside it. Among the variables, CuTe's order is
  kept: a target in `CUTE_RS_COMPILER_OPT` (`cubin-chip=`) beats
  `CUTE_RS_ARCH`, and `CUTE_RS_LINEINFO` can only switch line information
  on (`CUTE_RS_LINEINFO=0` leaves what the option string sets). A variable
  that is set but invalid is reported even when another one takes
  precedence.

`Kernel::compile_with(&toolchain, &options)` compiles with exactly the
options it is given and returns a `CompiledKernel` (name, target, PTX ISA
version, options and PTX); it needs a target. `Kernel::compile_ptx` is a
wrapper for the defaults. `Kernel::run_cuda` compiles at the defaults plus
the environment for the GPU it runs on, and `run_cuda_with` takes options.
`CudaContext::load(&compiled)` loads with `cuModuleLoadDataEx` and the JIT
options the kernel asks for (`CompiledKernel::jit_options`).

| CuTe DSL | Spellings | cute-rs |
|---|---|---|
| `OptLevel(n)` | `--opt-level n`, `opt-level=n` | `opt_level(n)`, 0 to 3 (default 3; others are rejected). Sets LLVM's level (`nvvm-attach-target O=n`) and the driver JIT's `CU_JIT_OPTIMIZATION_LEVEL`: 0, 1 and 2 map to themselves, 3 to 4, the JIT default that matches `ptxas -O3`. At O0 the 128-bit accesses stay 128-bit but use generic addresses (LLVM's address-space inference is an optimization). |
| `GenerateLineInfo(True)` | `--generate-line-info`, `preserve-line-info`, `CUTE_DSL_LINEINFO=1` | `generate_line_info(true)`, `CUTE_RS_LINEINFO=1`. The PTX gets `.file` and `.loc` with the Rust file, line and column of the code that built each instruction (helpers' files included), the JIT keeps them (`CU_JIT_GENERATE_LINE_INFO`), and `ptxas -lineinfo` plus `nvdisasm -g -c` show them. The instructions are unchanged. |
| `GPUArch("sm_90a")` | `--gpu-arch sm_90a`, `cubin-chip=sm_90a`, `CUTE_DSL_ARCH` | `gpu_arch(Arch::Sm90a)`, `CUTE_RS_ARCH` (`Arch` parses every CuTe DSL name, `sm_101*` as Thor's `sm_110*`). The kernel's target (`KernelOptions::target`) is the default, and another one is refused. |
| `cutlass.base_dsl.Arch` (`sm_80` ... `sm_121f`, `from_string`, `to_string`, `is_family_of`, `can_run_binary_built_for`, `AmpereArchs()` ... `RubinArchs()`), `get_arch_enum()` | `Arch`, all 24 targets (`sm_101`, `sm_101a`, `sm_101f` parse to the `Sm110` ones, as in CuTe DSL), with `major()`, `minor()`, `suffix()`, `version()`, `FromStr`/`Display`, `is_family_of`, `can_run_binary_built_for`, `Arch::AMPERE` ... `Arch::RUBIN`, and `min_ptx()` (each target's first PTX ISA version: 7.4 for sm_87, 7.8 for sm_89 and sm_90, 8.0 for sm_90a, 8.6 for sm_100/100a, 8.7 for sm_120/120a, 8.8 for the `f` targets, sm_103 and sm_121, 9.0 for sm_110, 9.4 for sm_107). `Arch` is not ordered (sm_90 and sm_90a would compare equal): compare `version()`s. `sm_88`, in PTX ISA 9.0, is not in CuTe DSL 4.8.1's enum, so not here. `KernelOptions::target(arch)` builds a kernel for a target: `target_arch()` reads it in the body (else `CUTE_RS_ARCH`), and the kernel is checked against it when the build ends, so a missing instruction fails at the user's line. `Arch::from_compute_capability`, `CudaContext::arch()` (CC 9.0 is sm_90a, 10.3 sm_103a, ...), `CudaContext::best_target(&toolchain)` and the pure `best_target(cc, driver_ptx, &support)`: the most specific target the device runs that the toolchain emits and the driver loads, falling back from the chip to the family to the baseline (CC 10.3 with a CUDA 12.8 driver gets sm_100); `run_cuda` uses it. `Toolchain::targets()` (`llc -mcpu=help` next to `mlir-opt`, else a table by LLVM version: LLVM 21 has PTX 8.8 and no sm_110, LLVM 22 sm_110 and PTX 9.0, LLVM 23 PTX 9.3, none sm_107), `Toolchain::llvm_version()`, `CudaContext::driver_version()` and `driver_ptx_ceiling` (PTX 8.7 for CUDA 12.8, 9.N for 13.N). Compiling fails, naming the numbers and the instruction that raised the version, when the toolchain emits too old a PTX or lacks the target (sm_107 today), and loading when the driver takes too old a PTX. Thor compiles as `sm_101a` (PTX 8.8 at most) where the LLVM has no sm_110, noted on the `CompiledKernel` (`target_name()`, `note()`); whether a CUDA 13 driver still loads `sm_101a` PTX is not checked here. `cute::VERSION`, and `CUTE_DSL_VERSION` (4.8.1, the `cutlass.__version__` cute-rs follows) |
| `HostTarget` | `--host-target <preset or 'llvm -mtriple=...'>` | No host object is built: the host program is Rust, which `cargo build --target <triple>` cross-compiles, and the PTX does not depend on the host. A spec naming this host is accepted; any other is rejected with that advice. |
| `EnablePYIR`, `FrontendNext`, `DisableCuteExtCompile` | `--enable-pyir`, `enable-pyir`, `CUTE_DSL_ENABLE_PYIR` (the other two are typed only) | `enable_pyir`, `frontend_next`, `disable_cute_ext_compile`, `CUTE_RS_ENABLE_PYIR`: accepted, with no effect (a debug log line says so). They select NVIDIA's other Python front ends and its `cute_ext` pipeline; cute-rs has one builder and one pipeline. |
| `debug{launch-check}` | compact token | Accepted, with no effect. Every launch checks its configuration and arguments against the kernel's `KernelSignature` already (the block shape it was built for, `max_threads`, clusters and fallback clusters, shared memory against the device, buffer types, lengths and alignment, scalars, by-value sizes); a block or grid no GPU runs is left to the driver on device launches, which rejects it. |
| `PtxasOptions`, `KeepPTX`, `KeepCUBIN`, `KeepSASS`, `NvdisasmOptions`, `DumpDir` | `--ptxas-options`, `--keep-ptx`, `--keep-cubin`, `--keep-sass`, `--nvdisasm-options`, `--dump-dir`, `ptx-options=`, `dump-ptx-path`, `dump-cubin-path`, `CUTE_DSL_KEEP`, `CUTE_DSL_DUMP_DIR`, `CUTE_DSL_PRINT_IR`, and the deprecated `CUTE_DSL_KEEP_IR`, `CUTE_DSL_KEEP_PTX`, `CUTE_DSL_KEEP_CUBIN` and `CUTE_DSL_KEEP_SASS` | `ptxas_options("--maxrregcount=128")`, `keep(Artifact, bool)` (`keep_ptx`, `keep_cubin`, `keep_sass`, `keep_all`), `nvdisasm_options` (default `-g -c`), `dump_dir`, `print_ir`, `use_ptxas`, and the same spellings with `CUTE_RS_`: `CUTE_RS_KEEP` (comma-separated `ir`, `mlir`, `llvm`, `ptx`, `cubin`, `sass`, `all`; CuTe's `ir-debug` is `mlir`), `CUTE_RS_DUMP_DIR`, `CUTE_RS_PRINT_IR`, `CUTE_RS_KEEP_IR` (the MLIR) and the other `CUTE_RS_KEEP_*` switches. See "PTX assembler, cubins and artifacts" below. |
| `EnableAssertions`, `RemarkFilter`, `RemarkOutput`, `WarningsFilter`, warnings and remarks | `--enable-assertions`, `--remark-filter`, `--remark-output`, `warnings`, `warnings{cat}`, `remarks`, `remarks{cat}`, `CUTE_DSL_ENABLE_ASSERTIONS`, `CUTE_DSL_WARNINGS_AS_ERRORS`, `CUTE_DSL_WARNINGS_IGNORE` | `enable_assertions(true)` (`CUTE_RS_ENABLE_ASSERTIONS`) compiles `testing::assert_` in as `cf.assert`; `warnings(&[cats])`, `remarks(&[cats])` (none for all; the spellings accumulate), `warning_filter`, `remark_filter` (`\|`-separated categories or message substrings: no regex engine), `remark_output(path)` (YAML), `warnings_as_errors` (`CUTE_RS_WARNINGS_AS_ERRORS`, a `CompileError::Warnings`) and `warnings_ignore` (`CUTE_RS_WARNINGS_IGNORE`). See "Errors and diagnostics" below. |
| `EnableTVMFFI` | `--enable-tvm-ffi`, `CUTE_DSL_ENABLE_TVM_FFI` | `enable_tvm_ffi(true)`, `CUTE_RS_ENABLE_TVM_FFI`: recorded on the options (`get_enable_tvm_ffi`), with the same PTX. The callable is the Rust wrapper `cute::tvm_ffi::TvmFfiFunction`, which `TvmFfiFunction::compile` builds with the option set and `tvm_ffi_export!` exports from a `cdylib` ([Runtime, interop and testing](#runtime-interop-and-testing)). |
| `LinkLibraries`, `RDC`, `DeviceTarget` | `--link-libraries`, `--rdc` (`DeviceTarget` is typed only) | `--link-libraries` and `--rdc` are rejected until m11-extern-ffi-device-fn. `DeviceTarget` has no string spelling, and cute-rs has no builder method for it until that part. |
| IKET | `iket` | Rejected until m11-profiling-iket. |
| `FlattenLocsOutputJsonPath` | `--flatten-locs-output-json-path` | Rejected: cute-rs locations are single Rust file:line:column positions, with no location chains to flatten. |
| raw MLIR pass options | other compact tokens | Rejected: there is no NVIDIA pass pipeline to forward them to. |

Turning an option off (`enable-assertions=0`, `CUTE_RS_DEBUG=0`) asks for
nothing and is accepted. There is no global fast-math or
flush-to-zero switch: precision is chosen per operation.

### PTX assembler, cubins and artifacts

* **ptxas.** `Toolchain::cuda_tools()` finds the CUDA toolkit's `ptxas`, `nvdisasm`
  and `cuobjdump`: `ptxas` in `$CUTE_RS_PTXAS` (a path; a value that names no
  file turns `ptxas` off and does not search), `$CUDA_HOME/bin`,
  `$CUDA_PATH/bin`, `PATH` and `/usr/local/cuda/bin`; the other two next to it,
  then in the same places. Its release comes from `ptxas --version` (release
  12.8 assembles up to PTX ISA 8.7, `ptxas_max_ptx`); a kernel that needs newer
  PTX fails with a message naming the toolkit. `Kernel::compile_with` runs
  `ptxas -arch=<chip> -O<n> [-lineinfo] -v <your flags> -o <cubin> <ptx>` in a
  per-process temporary directory (removed afterwards) when something asks for
  it: `use_ptxas(true)`, `ptxas_options`, or a kept cubin or SASS. Otherwise the
  PTX is left to the driver's JIT (`compile_ptx` never runs `ptxas`). The JIT
  can be a newer assembler than the toolkit, and CUDA 12.8's `ptxas`
  (V12.8.93) miscompiles two instructions that the feature kernels use and
  that the JIT of driver 595 gets right:
  `sub.sat.s32 d, x, INT_MIN` becomes `add.sat.s32 d, x, INT_MIN` from `-O1`
  up (`int_multiplies`), and `st.b32 [a], %rd`, where a `cvt.f64` wrote the
  64-bit `%rd`, stores the double converted to an unsigned integer
  (`F2I.U32.F64.TRUNC` in the SASS) instead of its low 32 bits, which the PTX
  ISA says are chopped (`casts` at `-O0`, where LLVM splits 8-byte stores
  into such pairs). The PTX is valid; `tests/cubin.rs` holds the JIT to the
  simulator on both and checks that `ptxas` differs only in these two ways.
  Use a newer toolkit if you assemble kernels with these instructions. The cubin is `CompiledKernel::cubin()` and the `-v` report
  `ptxas_log()` (parsed into `resource_usage()`); a `-O<n>` in your flags replaces
  cute-rs's. `CompileError::Ptxas` carries a rejection with the command line. `ptxas` ignores `--maxrregcount` for a kernel with a known block size
  (its `.maxntid` launch bound takes precedence), so build the kernel without
  `block_dim` to cap registers.
* **Without a toolkit** (the Mac, or a GPU machine with only the driver) the
  flags that the driver's JIT has an option for are mapped onto
  `cuModuleLoadDataEx` (`jit_options_for_ptxas_flags`): `--maxrregcount=N` is
  `CU_JIT_MAX_REGISTERS`, `-O<n>` is `CU_JIT_OPTIMIZATION_LEVEL` (3 is the JIT's
  4), `-lineinfo` and `--generate-line-info` are `CU_JIT_GENERATE_LINE_INFO`, and
  `-v` is `CU_JIT_LOG_VERBOSE`. Any other flag is an error that names `ptxas`,
  at compile time on every machine. `CudaContext::link_cubin` (the driver's
  `cuLink*` linker) makes a cubin from PTX on a GPU machine, and
  `attach_cubin` gives it to a `CompiledKernel`; `run_cuda_with` uses it when
  a cubin or SASS is kept and there is no `ptxas`.
* **Loading.** `CudaContext::load_module(ModuleImage::{Ptx, Cubin, Fatbin},
  &jit_options, entry)` takes PTX (a NUL is appended) or binary images as bytes,
  and always attaches the JIT's error log, so a failure reads `cuModuleLoadDataEx
  failed: CUDA_ERROR_INVALID_PTX` followed by the assembler's own message; the
  info log is `CudaFunction::jit_log()`. `CudaContext::load(&compiled)` prefers
  the cubin when the device can run a binary built for the cubin's target
  (`Arch::can_run_binary_built_for` with `CompiledKernel::cubin_target`: the
  kernel's target for `ptxas`, the linking device for `attach_cubin`) and
  falls back to the PTX otherwise.
* **Ahead-of-time export.** `CompiledKernel::export_c(dir, name, prefix)`
  (`cute::aot`) writes a kernel that runs without cute-rs: a C header and
  source (an empty prefix is the name), an object file when a C compiler
  (`$CC`, else `cc`) and `cuda.h` (`$CUDA_HOME/include`, `$CUDA_PATH/include`,
  `/usr/local/cuda/include`) are found, and a manifest. Every file is stamped
  with the export format (`EXPORT_FORMAT_VERSION`, 1), the cute-rs version, the
  target and the PTX version (the header's `<PREFIX>_CUTE_RS_EXPORT_FORMAT`,
  `_CUTE_RS_VERSION`, `_TARGET`, `_PTX_VERSION`). `<prefix>_Kernel_Module_Load`
  loads into the current context (retaining the first device's primary context
  when none is current): the cubin when the device runs it as a binary, else
  the PTX after checking the driver's version against it; it fails with
  `CUDA_ERROR_NO_BINARY_FOR_GPU` on a device that cannot run the target.
  `<prefix>_Kernel_Launch(&module, &config, &args)` makes the launcher's checks
  (a declared block size, `max_threads`, whole clusters, pointer alignment,
  the bytes behind a pointer of known length through `cuMemGetAddressRange`,
  scalar divisibility), raises the function's dynamic shared-memory limit past
  48 KB, and queues the kernel with `cuLaunchKernelEx` (cluster shape as an
  attribute; `cuLaunchKernel` with pre-12 headers, which have no clusters). A
  failed check returns `CUDA_ERROR_INVALID_VALUE` with the message in
  `module.error`. Not carried over: the JIT options (the PTX is JIT-compiled at
  the driver's default, which is `-O3`), the shared-memory carveout for
  `min_blocks_per_sm`, and fallback clusters. The manifest is line-based text:
  `cute-rs-export 1`, `crate`, `cubin_target`, the image files and the
  `KernelSignature` (`to_text`/`from_text`: name, target, PTX version, block
  requirements, shared-memory sizes, and each parameter's kind, type, length,
  divisibility, alignment and by-value size, and the module globals), which
  `from_text` reads strictly, naming the line of anything malformed.
  `ExportedKernel::choose_image(cc, driver_ptx)` is the loading decision
  without a GPU.
* **Artifacts.** `keep(Artifact::{Ir, Mlir, Llvm, Ptx, Cubin, Sass}, true)` (or
  `CUTE_RS_KEEP`, `--keep-ptx`, ...) writes them to `dump_dir` (default: the
  working directory; created if needed) as
  `<kernel>_<chip>_<hash>.<ext>` (`.ir.txt`, `.mlir`, `.llvm.mlir`, `.ptx`,
  `.cubin`, `.sass`), where the hash is of the PTX, the optimization level and
  the `ptxas` and `nvdisasm` flags, so the variants of one kernel (several
  `GemmConfig`s named `tensorop_gemm`, two optimization levels, one PTX with
  and without `--maxrregcount`) keep their own files. `ir` is `Kernel::dump`, `mlir` the emitted MLIR with
  locations, `llvm` the module after the first lowering stage. They are also on
  the `CompiledKernel`: `ir()`, `mlir()`, `llvm()`, `ptx()`, `cubin()`,
  `sass()`. SASS is `nvdisasm` on the cubin with `nvdisasm_options` (`-g -c`), or
  `cuobjdump -sass`, and carries Rust file and line numbers when line information
  is on. `CompiledKernel::artifacts()` says, per requested artifact, where it
  went or why it could not be made or written (also on stderr and in the log):
  nothing is skipped silently, and the Mac reports cubin and SASS as unavailable.
  `CUTE_RS_PRINT_IR=1` (or `print_ir(true)`) prints the MLIR to standard output.
* **Debug mode and logging.** `CUTE_RS_DEBUG=1` (`CompileOptions::debug(true)`)
  turns on line information, locations in the MLIR and every artifact. IR
  validation and launch checks run every time without it, and `RUST_BACKTRACE=1`
  gives full backtraces, so CuTe's `CUTE_DSL_DEBUG` has nothing more to raise;
  `CUTE_DSL_PRINT_AFTER_PREPROCESSOR` has no counterpart (`Kernel::dump` prints
  the recorded IR). `cute::log` is a dependency-free logger: `CUTE_RS_LOG_TO_CONSOLE=1`
  (stderr), `CUTE_RS_LOG_TO_FILE=<path>` (appended), `CUTE_RS_LOG_LEVEL` (`error`,
  `warn`, `info`, `debug`, `trace`, or Python's numbers; default `debug`), or
  `log::set_level`, `log::log_to_console`, `log::log_to_file` from a program. It
  logs the kernel build, each pipeline stage with its time (info), `ptxas` and
  `nvdisasm` command lines, module loads and launches with grid, block, shared
  memory and attributes (debug), and unavailable artifacts (warn). Off, a log call
  is one relaxed atomic load: nothing is formatted.


Line information gives the kernel a debug-info subprogram at its
`Kernel::build` call, in a compile unit of emission kind
`DebugDirectivesOnly`, so that NVPTX emits only `.file` and `.loc`
(`LineTablesOnly` would add DWARF sections and `.target ..., debug`, which
`ptxas` refuses for optimized code). MLIR takes an operation's file from its
scope, so operations built in another file get a lexical-block-file scope
for it. A stage hook gets the module as text, with helpers for the edits the
MLIR dialects cannot make earlier: `set_argument_attribute` (argument
attributes of an `llvm.func`, such as `llvm.byval`, `llvm.align` or
`nvvm.grid_constant`) and `set_global_attribute` (attributes of an
`llvm.mlir.global`, such as the `alignment` of `@__cute_dynamic_smem`, the
dynamic shared memory, which LLVM 21 to 23 emit as `.extern .shared .align
16 .b8 __cute_dynamic_smem[];`).

### Errors and diagnostics

`cute::Error` gathers every error cute-rs reports (CuTe DSL's
`DSLUserCodeError`, `DSLRuntimeError`, `CompilerDiagnosticError` and
`DSLCudaRuntimeError`): `ValidationError`, `SimError`, `CompileError`,
`OptionError`, `CudaError` (with the `cuda` feature) and `AssertionFailure`
convert into it, so `?` collects them. `Error::loc()` is an owned
`SourceLoc` (file, line, column): the user's line for validation,
simulation and assertion errors, and the first `file:line:col: error` of
`mlir-opt`'s output for a compile failure (a kernel compiled without line
information is lowered again with locations when `mlir-opt` fails, so the
error names the line that built the operation). `Display` is unchanged;
`{:#}` and `Error::render()` add a code frame of that line, read as given
or under `CARGO_MANIFEST_DIR` (none when the file cannot be read). Misuse
found inside the builder (mismatched operand types, a duplicate parameter
name, a Var whose type changes) panics with a message that starts with the
user's `file:line:col`. Rust's own backtraces take the place of CuTe DSL's
stack-trace switches: `RUST_BACKTRACE=1` for `CUTE_DSL_SHOW_STACKTRACE`
(there are no DSL frames to filter, so nothing stands for
`CUTE_DSL_FILTER_STACKTRACE`). A `CudaError` has the driver's code and
both its name and description (`cuLaunchKernelEx failed:
CUDA_ERROR_INVALID_VALUE: invalid argument`).

Compiler diagnostics are `Diagnostic`s (a `Severity` of error, warning or
remark, a category, a message and a location), asked for with
`CompileOptions::warnings`/`remarks` or `warnings{cat}`/`remarks{cat}` and
kept on `CompiledKernel::diagnostics()` (and printed to stderr). The
categories:

* `ptx`: `ptxas -v`'s report, parsed into `CompiledKernel::resource_usage()`
  (registers, stack frame, spill stores and loads, shared and constant
  bytes; `ResourceUsage::from_ptxas_log`, and `from_jit_log` for the
  driver JIT's verbose info log, `-v` in the ptxas options when ptxas does
  not run). Spills are a warning and local memory a remark, both at the
  kernel's `Kernel::build` line; `remarks{ptx}` also gives the summary. On
  a GPU, `CudaFunction::resource_usage()` reads `cuFuncGetAttribute`
  (registers, local, shared and constant bytes, the block limit, the PTX
  and binary versions) whatever compiled the kernel.
* `target`: the PTX ISA raises and target fallbacks, as remarks.
* `loop`: LLVM's loop remarks are unavailable: `mlir-opt` has no
  `-pass-remarks` option (only `--lto-pass-remarks-*`, which the NVPTX
  lowering does not run), so `remarks{loop}` gives a remark saying so.
* `nvvm`: accepted; cute-rs runs no NVVM checkers. The protocol bugs CuTe
  DSL's static checkers find (`experimental/compiler_diagnostic/show_error.py`'s
  mbarrier arrival counts, say) are found by the simulator at run time
  instead, so that static checker is not built.
* `sim`: the simulator's non-fatal findings, in `SimReport::diagnostics`
  (failed assertions when `SimOptions::assertions` is off).

Device assertions: `cute::testing::assert_(cond, msg)` is `Op::Assert`
(the validator requires a Boolean). The simulator checks it by default
(`SimOptions::assertions`; a failure is a `SimError` with the line, block,
thread and message). Compiled code includes it only with
`enable_assertions(true)` or `CUTE_RS_ENABLE_ASSERTIONS=1`, as a `cf.assert`
with a plain `loc("file":line:col)` even under line information (a fused
location loses `__assertfail`'s file and makes LLVM reject the module),
which lowers to `__assertfail(msg, file, line, ...)`, resolved by the
driver's JIT and by `ptxas` as `vprintf` is. The driver prints
`file:line: (unknown): block: [..], thread: [..] Assertion `msg` failed.`
on stderr and the launch fails with `CUDA_ERROR_ASSERT`, whose message says
so. That error is sticky for the whole process: `CudaContext::reset(&mut
self)` (`cuDevicePrimaryCtxReset`) resets the primary context, but on an
SM89 with driver 595 retaining it again still fails with
`CUDA_ERROR_ASSERT`, so kernels that may assert belong in a child process
(`tests/gpu_assert.rs` does that). `testing::RuntimeAssertion::new(n)`
(`init_value`, `disable`) is CuTe DSL's `RuntimeAssertion`: a host-owned
Boolean buffer of `n` flags, a `KernelParam` whose device side is
`DeviceAssertions` (`DeviceAssertions::param(name, spec)` for
`Kernel::build` kernels; `store(idx, pred, msg)` records the message and
line in the kernel's table, `Kernel::assertion_records()`), and
`verify(&kernel)` returns the first stored flag that is false as an
`AssertionFailure` with its message and Rust location. `disable` declares
no parameter.

## Runtime, interop and testing

What CuTe DSL offers around a compiled kernel, and where it is in cute-rs. Everything that
launches needs the `cuda` feature.

| CuTe DSL | cute-rs |
|---|---|
| Compile options: `cute.compile[...]`, `options="..."`, `CompileOptions` | `CompileOptions` (builder methods, `CompileOptions::parse` for the option strings, `with_env`), every option and spelling in [Compile options](#compile-options), including `--enable-tvm-ffi` |
| Environment variables `CUTE_DSL_*` | `CUTE_RS_*` with CuTe's suffix: `COMPILER_OPT`, `ARCH`, `LINEINFO`, `KEEP` (and the deprecated `KEEP_IR`, `KEEP_PTX`, `KEEP_CUBIN`, `KEEP_SASS`), `DUMP_DIR`, `PRINT_IR`, `DEBUG`, `ENABLE_ASSERTIONS`, `WARNINGS_AS_ERRORS`, `WARNINGS_IGNORE`, `ENABLE_TVM_FFI`, `ENABLE_PYIR` (no effect); the caches' `JIT_CACHE_MAX_ELEMS`, `NO_CACHE`, `CACHE_DIR`, `DISABLE_FILE_CACHING`, `FILE_CACHE_MAX_ENTRIES`, `FILE_CACHE_MAX_BYTES`; the logger's `LOG_TO_CONSOLE`, `LOG_TO_FILE`, `LOG_LEVEL`; and `CUTE_RS_PTXAS` for the assembler. Explicit options win over them. `CUTE_DSL_LOG_AUTOTUNE` has no counterpart (`Tuned::trials` has every configuration's time or reason) |
| `.launch(grid=, block=, cluster=, smem=, stream=, use_pdl=, cooperative=, fallback_cluster=, launch_completion_event=, programmatic_event=, preferred_smem_carveout=, ...)`, `max_number_threads=`, `min_blocks_per_mp=` | `LaunchConfig::new(grid, block)` with `with_cluster`, `with_smem`, `with_dynamic_smem`, `with_pdl`, `with_cooperative`, `with_fallback_cluster`, `with_smem_carveout`; `LaunchBuilder::stream`, `launch_completion_event`, `programmatic_event`; `KernelOptions::max_threads`, `min_blocks_per_sm` (the rows of [CuTe DSL → cute-rs](#cute-dsl--cute-rs)) |
| Launch-time argument checks (the JIT function's marshaling) | `KernelSignature::pack`, the single check-and-pack path of the simulator, the CUDA launcher, exported kernels and TVM-FFI calls; by-value parameters and module globals |
| `cutlass.cuda` (contexts, streams, memory, modules), CUDA graphs, CUDA events | `CudaContext`, `CudaStream` (`from_raw` for a framework's stream), `DeviceBuffer`, `PinnedBuffer`, `CudaGraph` and `CudaGraphExec` from `begin_capture`, `CudaEvent` |
| `cute.runtime.make_ptr`, `nullptr`, `from_dlpack`, a framework tensor passed to a `@cute.jit` function, `make_fake_tensor`, `make_fake_compact_tensor` | `DevicePtr::from_raw`, `DevicePtr::null`, `from_dlpack`, `DlTensor` parameters over a `DlpackTensor` (CuTe DSL's implicit `mark_layout_dynamic()`), `make_fake_tensor`, `make_fake_compact_tensor` |
| The JIT and file caches, `__cache_key__` | `Kernel::compile_with`'s in-memory and on-disk caches, `KernelCache`, `TensorSpec::cache_key`/`cache_keys`, `ExternalTensor::cache_key`, `PointerSpec::cache_key` |
| `export_to_c`, `cute.runtime.load_module` | `CompiledKernel::export_c`, `ExportedKernel`, `CudaContext::load_exported` |
| `cute.compile(..., options="--enable-tvm-ffi")` and the TVM-FFI callable (`__tvm_ffi_<name>`, the `TVMFFIAny` safe call, `TVMFFIEnvGetStream`, `TVMFFIErrorSetRaisedFromCStrParts`) | `cute::tvm_ffi`: `TvmFfiFunction::compile(typed_kernel, &toolchain, &options, \|host\| Ok(launch_config))`, `TvmFfiFunction::new` (a `CompiledKernel`) or `from_exported` (an `ExportedKernel`, so no `mlir-opt` at call time), and `cute::tvm_ffi_export!(name, make())`, which exports `__tvm_ffi_<name>` with the safe-call signature from a `cdylib`. A call decodes `kTVMFFIDLTensorPtr` and `kTVMFFITensor` arguments as borrowed DLPack tensors (CUDA memory only) and `kTVMFFIInt`/`Bool`/`Float` as scalars (`FromTvmFfi`; Rust tuples of parameters are runs of consecutive arguments); checks them against the kernel's specs (dtype, rank, static extents, the divisibility and width of dynamic ones, one `SymInt` shared by several tensors) and the packed values against its `KernelSignature` (alignment); takes the environment's stream for the tensors' device; launches; and reports a failure as a `TypeError`, `ValueError` or `RuntimeError` naming the argument and the signature (``add_one(a: Tensor, b: Tensor): argument 0 (`a`): ...``), returning -1. The runtime functions are looked up with `dlsym` in the process (`import tvm_ffi` loads `libtvm_ffi` globally), so the library links nothing; `set_runtime_override` installs stubs. The ABI follows apache-tvm-ffi 0.1.14's `c_api.h` (`TVM_FFI_VERSION`): nvidia-cutlass-dsl 4.8.1 does not pin apache-tvm-ffi, and 0.1.14.post0 is the release installed beside CuTe DSL on the test machine. Not supported: TVM-FFI `Array`, `Map` and dataclass arguments, defaults and keyword arguments (CuTe DSL's Python wrapper resolves those; a cute-rs callable takes its positional arguments), and the in-kernel checks CuTe DSL compiles into the callable, which cute-rs runs on the host before the launch |
| `tvm_ffi.register_global_func(name, compiled)`, `compiled.export_to_c(..., function_name=)` for a TVM-FFI callable | `cute::tvm_ffi::register_global_function(name, safe_call, override)` (`TVMFFIFunctionCreate` and `TVMFFIFunctionSetGlobal` found with `dlsym` in the process; an error when `libtvm_ffi` is not loaded), and `cute::tvm_ffi_register_all!(add_one, "ns.name" => other)`, which generates the exported `cute_tvm_ffi_register_all()` (the number of failures) to call after `import tvm_ffi`; `cute::tvm_ffi_register_on_load!()` adds a Linux `.init_array` constructor that calls it at load. The `__tvm_ffi_<name>` symbol is `tvm_ffi_export!`'s, the exported PTX or cubin form is `TvmFfiFunction::from_exported` (m6 AOT). Registration is not run against a real `libtvm_ffi` in cute-rs' tests, and the C signatures follow `c_api.h` from memory |
| `cutlass.jax.cutlass_call(fn, output_shape_dtype=, input_spec=, output_spec=, input_output_aliases=, allow_cuda_graph=, ...)`, the XLA FFI target it lowers to | `cute::xla_ffi`: `handle_call_frame(*mut XlaCallFrame) -> *mut XlaError`, with `cute::xla_ffi_export!()` generating `cute_xla_cutlass_call`, `cute_xla_cutlass_call_nograph` and the stage bundle `cute_xla_cutlass_call_bundle` in a `cdylib`; `register_cutlass_call(target, |spec| Ok(XlaTarget::new(callable).alias(0, 0)))` where the builder gets the call's `TensorSpec`s (dtype, dims, XLA minor-to-major layout) and attributes and returns a `TvmFfiFunction` (compiled, or `from_exported` for the AOT form); compiled kernels are cached per thread by target, specs and attributes (`call_cache_stats`, `release_compile_cache`). The execute stage reads `XLA_FFI_Buffer`s and scalar, string and integer-array attributes (`cute_target`, `arg_m2m_<i>`/`ret_m2m_<i>` layouts, `device`; other scalars are kernel arguments), launches on `XLA_FFI_Stream_Get`'s stream, checks declared aliases, and reports errors through `XLA_FFI_Error_Create` (invalid argument, not found, failed precondition, internal). The struct layouts are from memory of `c_api.h`, which is not in the repository, and tested only against the same structs. Sharding, typed state and futures are out of scope |
| `cutlass.jax` helpers: `get_cutlass_call_ffi_name`, `get_cutlass_call_ffi_version`, `jax_to_cutlass_dtype`/`cutlass_to_jax_dtype`, `jax_to_cutlass_layout_order`/`cutlass_to_jax_layout_order`, `register_ffi`, `is_ffi_registered`, `get_export_disabled_safety_checks`, `release_compile_cache` | the same names in `cute::xla_ffi` (dtype codes of XLA's `PrimitiveType`, layouts as inverse permutations, `strides_of_layout`) and mirrored in `python/cute_rs_jax/__init__.py` (plain files, no build step; `register_ffi(library)` registers the handler with `jax.ffi.register_ffi_target`, `cutlass_call(target, output_shape_dtype, ...)` wraps `jax.ffi.ffi_call`, `get_export_disabled_safety_checks()` lists the custom-call targets `jax.export` must accept). The Python files are tested for their converters (`tests/python/test_cute_rs_jax.py`; the jax test skips without `CUTE_RS_XLA_LIB`); the names differ from NVIDIA's runtime library targets (`cute_rs_cutlass_call_v2`). `cutlass.jax.from_dlpack` and `cutlass.jax.testing` have no counterpart beyond `cute::host` |
| `cutlass.testing.benchmark(callable, warmup_iterations=10, iterations=100, stream=, workspace_generator=, workspace_count=1, use_cuda_graphs=False)` | `cute::testing::benchmark(&stream, &BenchmarkOptions { warmup_iterations, iterations, workspace_count, use_cuda_graphs }, \|stream, workspace\| launch)`: the average microseconds per iteration between two `CudaEvent`s, the workspaces used in turn through warmup and timed iterations; graph mode captures both runs once (on a stream other than the legacy one) and times the replay. `benchmark_with_workspaces` takes a generator, called once per workspace. No CUPTI timing (`use_cupti=`) or NVTX ranges (`nvtx_tag=`): the events time everything queued between them. `LaunchStats::kernel_seconds` is different again: a host timer around one launch and the synchronisation after it, launch latency and the wait included |
| `cutlass.testing.get_workspace_count(one_workspace_bytes, warmup_iterations, iterations)` | `get_workspace_count(&ctx, bytes, warmup, iterations)` with the device's L2 size, and the pure `workspace_count_for_l2(l2, bytes, warmup, iterations)`: `min(warmup + iterations, 3 * L2 / bytes + 1)` |
| `cutlass.testing.tune(func, params_dict=, kernel_arguments=, warmup_iterations=, iterations=, stream=)`, `CantImplementError` | `cute::testing::tune(&stream, configs, &options, \|cfg\| Ok(launch_closure))` (the configurations are any iterator: `params_dict`'s product is a `flat_map`), returning a `Tuned { config, time_us, trials }`; a build that panics or returns an error (`CantImplement`) and launches that fail are skipped and recorded. `tune_by(configs, \|cfg\| time)` is the same over any measure (the simulator's, say) |
| `cute.testing.autotune_jit(params_dict=, update_on_change=, warmup_iterations=, iterations=)` | `Autotuner::new(configs).update_on_change(&["M", "N"]).options(..)`, then `get_or_tune(&stream, &TuneKey::new(&spec.cache_key()).part("M", m).part("N", n), build)` (or `get_or_tune_by`): the winner is kept per cache key and watched part, unwatched parts are ignored, and a kernel revisited goes through the compile cache |

## Requirements

* Rust 1.88 or newer (edition 2024).
* For PTX: an MLIR/LLVM installation with the NVPTX backend. LLVM 21, 22
  and 23 are tested; Homebrew's works (`brew install llvm`), and on
  Debian/Ubuntu `mlir-opt` comes from the `mlir-<version>-tools` package on
  apt.llvm.org.
  `Toolchain::discover()` looks at `$CUTE_MLIR_OPT`, `$LLVM_DIR/bin`, `PATH`,
  Homebrew and `/usr/lib/llvm-*`.
* To run on a GPU: Linux, an NVIDIA driver, and an SM80 or newer GPU. Build
  with `--features cuda`. `libcuda` is loaded at runtime, so building needs no
  CUDA installation.

## Testing

```bash
cargo test
```

* `src/layout/tests.rs` covers the layout algebra: CuTe's documented examples
  and the test vectors of CUTLASS 4.8.1's CuTe dialect documentation and
  `cutlass_compiler` tests, written as layout strings, which parse
  (`"(2,(3,4)):(1,(2,6))".parse::<Layout>()`, and `"(?,?):(?,1)"` with
  runtime leaves). Brute-force property tests cover coalesce, composition,
  complement, inverses, TV layouts, nullspace, `tile_to_shape` with an order,
  `increment_coord`, `find` and weak congruence, and evaluate the algebra on
  layouts with runtime extents and strides at random values, against the
  same operations on the static layouts of those values, including
  compositions whose right-hand runtime values stand for small values that
  are multiples of nothing in particular.
* `tests/launch_pdl.rs` covers griddepcontrol (validator, MLIR, sm_90 PTX and `ptxas`), the
  `SimStream` PDL model (a dependent reading the producer's output or overwriting its input
  before the wait is caught, a correct pair passes) and `programmatic_dependent_launch`;
  `tests/dlpack.rs` builds DLPack managed tensors with counting deleters and covers
  layouts, marking, dtype codes, errors, FP4 packing, the simulator on host tensors, cache
  keys and `call_bypass_dlpack`; `tests/jit_cache.rs` covers the in-memory and on-disk
  compile caches (a re-run of the test binary hits the disk) and `KernelCache`. With the
  `cuda` feature, `tests/gpu_pdl_events.rs` (PDL, events, launch events, cooperative
  launch), `tests/gpu_external.rs` (DLPack device tensors, raw pointers, views, null
  pointers) and `tests/gpu_jit_cache.rs` (the module cache and unload callbacks) run on
  the GPU.
* `tests/tvm_ffi.rs` calls exported TVM-FFI functions through their C ABI with a stub
  runtime: the ABI's layout, `kTVMFFIDLTensorPtr` and tensor-object arguments, scalars, and
  every check failing with its exception kind and message. `tests/autotune.rs` covers
  `workspace_count_for_l2`, `tune_by`'s skipping and selection, and the `Autotuner`'s cache.
  With the `cuda` feature, `tests/gpu_tvm_ffi.rs` runs the exported functions on device
  memory (on the environment's stream, proved by capturing the call into a graph, and from
  an export), and with `CUTE_RS_TVM_FFI_LIB` set to the built `libtvm_ffi_add.so` a C
  program compiled with `cc` calls the library with its own stubs, and
  `CUTE_RS_TVM_FFI_PYTHON` (a Python with `tvm_ffi` and `torch`) runs
  `examples/tvm_ffi/*.py` on torch tensors; `tests/gpu_benchmark.rs` ports the
  `benchmark_autotune` notebook's walkthrough: stable times eager and from graphs,
  workspace rotation with the device's count, graphs' lower overhead on a small kernel,
  `tune` skipping the unbuildable and unimplementable tiles, and the `Autotuner` tuning
  again only when `M` or `N` changes.
* `tests/examples.rs` simulates every example and checks it against a CPU
  reference. The GEMMs are bit-exact. The runtime-shape ports build each
  kernel once and run it at several sizes, partial tiles and (for
  `ampere_sgemm`) K below the K tile included.
* `tests/ops.rs` checks every scalar operation and control-flow construct
  against Rust semantics, and `tests/vars.rs` checks the IR that `Var`s
  turn into (loop-carried values and `if` results) and more `Var`
  patterns, simulated against Rust.
* `tests/math.rs` sweeps every `cute::math` function on every type
  (signed zeros, infinities, NaN, subnormals, extremes, known hard cases,
  uniform and random samples; `CUTE_MATH_SWEEP=n` inputs per function, 2^14
  by default) against an independent double-double reference
  (`tests/math_ref`) and asserts the documented bounds in precise and fast
  mode, and checks the rounding modes, `ftz` (also on `abs`, `min`, `max`
  and `clamp`), `min`/`max`/`clamp`, the classifications, `ipowi`, `absi`,
  the exact functions, and `abs`, `neg` and `copysign` as bit operations on
  raw patterns (NaN payloads, signaling NaNs).
* `tests/schedules.rs` holds the seeded schedules and independent thread
  scheduling to planted bugs and divergent warps: a seed replays its run,
  an `atomic_cas` race whose winner writes its id is reported by
  `simulate_schedules` with the element and the two schedules, a producer
  without back-pressure is reported as phase aliasing, a consumer that
  polls `mbarrier_test_wait` once and reads anyway is caught only by
  random completion, lane 0 waits on an mbarrier that lanes 1 to 31 of
  its warp arrive at in the else-branch, divergent unaligned
  `barrier.arrive`s complete a warp's arrival, a shuffle meets across the
  two branches of an `if` (each lane sending its own branch's operand), a
  vote that lanes which left a loop never reach is reported, and a
  divergent `mma.sync` and an intra-warp deadlock are reported with each
  group's lanes; a failing seed replays with the same step count.
* `tests/diagnostics.rs` builds deliberately broken kernels and checks that
  each bug is caught (races, missing waits, OOB, misalignment, divergent
  barriers, barriers that cannot complete, mbarrier parity waits that pass
  early, bulk copies missing from `expect_tx`, escaping values, a `Var` or
  loop state whose type or register-tensor layout changes, a `Var`'s
  alignment or constant relied on inside a loop that carries it, a register
  fragment a runtime loop reads and then writes, a loop marked
  `at_least_once` that runs zero times, a vectorize hint below sm_100,
  `cute.math` options CuTe DSL rejects or PTX lacks, integer instructions on
  types they do not take, warp collectives (votes too) run outside their
  member masks or without all of their lanes, a shuffle reading a lane past
  a short last warp, `redux.sync` kinds on types they do not
  take, Float32 `redux.sync` for targets other than sm_100a, `fmax` with
  `abs` for sm_80, a `warp_reduction` group of 12 lanes, half-precision
  options CuTe DSL rejects (`sat` with `relu`, `ftz` on BFloat16, `oob`
  with `ftz`, a predicate on fma), bf16x2 `add` for sm_89, the s2f6x2
  conversion for sm_100, unsupported `cvt_i8_bf16_intrinsic` and
  `sext_unpacked_i4_i8_intrinsic` lengths, `pack` widths that do not add
  up, carry chains that
  start nowhere, are broken by control flow, a call, inline PTX or a 64-bit
  division by a runtime value, or switch between adding and subtracting,
  inline PTX names without operands and operands without names, positional
  operands in a named template, a predicated template of two instructions
  or with a guard of its own, constraint strings that do not match their
  operands or use codes NVPTX does not take, immediates that are not
  integer constants, 32-bit addresses outside shared memory, hand-written
  MLIR naming operands it lacks, inline statements without a simulation,
  static-only layout operations on runtime extents and strides, a runtime
  tiler stride that is 0 at launch, a `Tensor::size`, a register fragment,
  a copy's per-thread chunk or rest mode, a gemm operand or an element-wise
  loop with a runtime extent, an identity tensor made with one or divided
  into one, a store past a runtime extent, accesses past the dynamic shared-memory
  region's launch size, shared memory larger than any GPU gives a block,
  `cp.async` tracked by an mbarrier whose count has no room for the tracked
  arrivals, or read by a warp that has not waited for its phase (or for its
  own `cp.async` group), a producer's ordinary stores read as if the
  tracked arrival had released them, each combination of memory ordering
  and cache hints that PTX's `ld`, `st` and `prefetch` refuse, a cluster
  scope below sm_90, non-coherent loads of memory the kernel writes, an
  access wider than its buffer's declared alignment, strong accesses of
  different sizes to the same bytes, …). It also holds the race checker to
  the PTX memory model on flag hand-offs: release and acquire, the fence
  patterns, `st.release; st.relaxed`, release sequences through atomics and
  copy atoms' memory orders are accepted, and each broken form (a relaxed
  load or store where the pattern needs more, a missing fence, data written
  after the release, another warp's store in between, a plain flag, an
  acquire of part of a release) is reported.
* `tests/atoms.rs` checks copy-atom and tiled-copy introspection and the
  layouts of CuTe's tiled-copy constructors and `retile`, and
  `make_cotiled_copy` against layouts worked by hand from CuTe's C++
  algorithm.
* `tests/dynamic_tensors.rs` applies every tiling and partitioning function
  (`local_tile`, `local_tile_proj`, `local_partition`, the divides, the
  partitions of tiled copies and MMAs, slicing, composition) to tensors of
  runtime extents, evaluates the results at many extents, and holds them
  to the same functions on static tensors where the tilers divide the
  extents, and to the static-upper-bound layouts (static bounds, the same
  runtime strides) on the tensors' own domains at any extents, partial
  tiles included. It also checks that the size queries on tensors fold
  where they can and evaluate to the static answers (an identity tensor's
  cosize is the size of its coshape, as in CuTe), and that `runtime_gemm`
  with runtime extents is correct on random small integers and gives the
  static-bound kernel's bits, there and on random values whose sums round,
  at several problem sizes from one built kernel.
* `tests/int_facts.rs` checks how divisibility facts flow through value and
  index arithmetic (`assume`, `get_divisibility`, floor division, the
  `?{div=N}` and `?{i64}` printing), that an assumption is what lets a
  vector copy through and costs no PTX instruction, that `i64` leaves make
  64-bit offsets in the MLIR, the PTX and the simulator while the thread
  arithmetic stays 32-bit, and `FastDivmod` prepared on the host; the
  feature kernels `assume_vector_copy`, `int64_strides` and `fast_divmod`
  run in the simulator and on the GPU, and `src/fast_divmod.rs` tests the
  formula on every divisor up to 2^16 and on random ones up to 2^31 - 1.
* `tests/mlir.rs` verifies every example with `mlir-opt` and checks its
  sm_80 PTX for the expected instructions, as well as every `mma.sync` kind,
  the named barriers, the SM80 and SM90 mbarrier operations, the effect of
  the loop hints (`unroll=4` gives four copies of the body plus a
  remainder, `unroll=1` `.pragma "nounroll"`, `unroll_full` a loop without a
  branch, `at_least_once` a loop without the zero-trip check in front), and
  every `cute::math` function on every type in both modes: precise PTX
  contains no approximation and no libdevice call, fast PTX the
  approximations the documentation names, and float `abs`, `neg` and
  `copysign` are `and.bN` and `xor.bN`, never `abs.f32` or `neg.f32`, and
  that each integer and bit instruction appears in the PTX as pure inline
  assembly (`lop3` with its table as the immediate), that the warp
  collectives are their PTX instructions: one `shfl.sync` per 32-bit word
  with the segment masks and clamps as immediates, every `redux.sync`
  kind, `match.{any,all}.sync` on `.b32` and `.b64`, `bar.warp.sync` with
  constant and runtime masks, `max` and `min` in all eight modifier
  combinations (PTX 7.2 on sm_86, refused on sm_80), and Float32
  `redux.sync` for sm_100a (PTX 8.6), refused for every other target; that
  carry chains keep their order in the PTX at O0 to O3, with nothing
  between a carry-out and its carry-in that touches the flag (also
  around 64-bit divisions by constants, warp collectives, the
  special-register reads and `cp.async.mbarrier.arrive`), that named
  inline PTX, raw inline assembly and hand-written MLIR are emitted as
  translated (operand numbers, constraints, integer constants as
  immediates, 32-bit shared addresses, guards, side effects, stores that a
  memory clobber keeps between a store and the reload it forces, pointer
  operands of MLIR operations), that an error of LLVM's that `mlir-opt`
  prints (an unknown constraint) fails the compile on every installed
  `mlir-opt`, even where it exits with success, that each memory ordering,
  scope and cache hint of `load`, `store` and the G2R and R2G copy atoms
  reaches the PTX as the qualifiers it names (plain ordered accesses as
  LLVM atomics, PTX 7.4 for the hints and 8.2 for MMIO, strong vectors of
  narrow elements with their element type), and that each `prefetch` form
  is one instruction. The packed half-precision operations are their instructions with every
  modifier, the predicated ones `@p`-guarded with the result tied to the
  first operand, `fma` on BFloat16 values is `fma.rn.bf16`, the bf16x2,
  `ex2.approx.ftz.bf16x2` and `.oob` forms are refused below sm_90 (`.oob`
  raising sm_90's PTX version to 8.1), `cvt.rn.satfinite.bf16x2.f32` takes
  element 1 first and raises the version to 8.1 on sm_80 and sm_89, the
  int8 conversions are `dp4a.s32.s32` against one-hot masks and
  `cvt.rn.f32.s32`, and the s2f6x2 conversions are refused for all but
  sm_100a and sm_120a and verified as MLIR (and as PTX 9.1 when `mlir-opt`
  is LLVM 23). It also compiles
  kernels for each of sm_80, sm_86, sm_89, sm_90, sm_100 and sm_120. For
  the compile options it checks that the split pipeline gives every example
  and feature kernel, and a kernel for each of `cute::math`'s instruction
  forms and a vectorized loop, the PTX of the single pipeline, and so does
  MLIR 21's fallback pass list on every installed `mlir-opt` (at O0 to O2
  too, for the examples); that 128-bit accesses, `mma.sync`, `ldmatrix`,
  `cp.async` and inline PTX survive at O0 to O2; that line information maps
  a known statement and a helper's operation to their Rust lines, locates
  every operation and changes no instruction; and that stage hooks change
  what they edit and nothing else. These tests are skipped when no
  `mlir-opt` is found; when `ptxas` is installed (or named by
  `$CUTE_PTXAS`), the sm_100a and sm_120 PTX of the vectorize hint is
  assembled with it. The tests that run every installed `mlir-opt` skip
  copies of a tile keep their 128-bit accesses, behind compares with the
  runtime extents, and that `runtime_gemm`
  with runtime extents issues the static-bound kernel's `mma.sync`,
  `ldmatrix` and `cp.async` instructions in no more PTX.
  those older than LLVM 21 (such as Ubuntu's LLVM 19), whose pipelines lack
  options cute-rs uses; the version comes from `mlir-opt --version`.
* `tests/artifacts.rs` runs a fake `ptxas` script (the exact command line,
  the temporary directory removed, the old-toolkit and rejection errors), the
  flag-to-JIT mapping, and kept artifacts in a temporary directory (the Mac
  reports cubin and SASS as unavailable), the file names of variants of one
  kernel and debug mode; with a real `ptxas` it checks the cubin, that
  `--maxrregcount=32` lowers the register count `-v` reports, and that
  `nvdisasm` shows the Rust lines. `tests/logging.rs` writes each log level to
  a file and checks that nothing is formatted when logging is off.
  `tests/cubin.rs` (feature `cuda`) runs five kernels through the `ptxas` cubin
  and through the PTX JIT with identical results, `--maxrregcount` on the GPU,
  the driver-linker cubin without `ptxas`, the JIT info and error logs,
  deliberately invalid inline PTX, and the two `ptxas` 12.8 miscompiles.
* `tests/ptxas.rs` compiles registered kernels for targets no available GPU
  runs (the SM90 feature kernels, with the cluster registers and
  `aggr_smem_size`, for sm_90, sm_90a, sm_100a and sm_120a; the integer and
  bit instructions, the carry chains, the hand-written PTX and MLIR,
  `dynamic_smem_size`, `get_dyn_smem_size` and `total_smem_size`,
  `cp.async.mbarrier.arrive`, the ordered and cache-hinted loads and stores,
  prefetches and G2R, R2G and cotiled copies, and every warp-collective
  feature kernel (shuffles, reductions, `redux.sync`, `match.sync`,
  `bar.warp.sync`, short warps) for every target; `fmax` and `fmin` for
  every target from sm_86 on; the FP8 conversions for every target from
  sm_89 on; Float32 `redux.sync` and the UE8M0 conversions for sm_100a; and the
  packed half-precision, exp2 and conversion kernels for every target,
  their SM90 forms from sm_90 on; and the kernels over tensors of runtime
  extents, `dynamic_tiles` and `runtime_gemm` with runtime extents, for every
  target; the SM90 builds of the warp-specialised examples and the
  `cp_async_mbarrier_arrive` cluster kernel from sm_90 on) and
  assembles their PTX with NVIDIA's `ptxas`, failing on any error. It is
  skipped unless `ptxas` is on `PATH`.
* `tests/gpu.rs` runs every example on a real GPU and on the simulator and
  compares every output (the warp-specialised examples in their SM89
  builds; their SM90 builds and the cluster kernel are simulated and
  assembled only). Results must match bit for bit, except that
  `mma.sync` kernels on data that has to round only need to agree within the
  f32 summation error bound, because the GPU rounds differently from the
  simulator. It also runs every `cute::math` function on the GPU (2^16
  inputs per function and type, every bit pattern of the 16-bit types for
  the unary functions): precise mode, the rounding modes, the `ftz` forms,
  `absi`, and `abs`, `neg` and `copysign` (on NaN payloads, alone in a
  kernel, on products, with constant signs and with their bits used as
  integers) must give the simulator's bits, and fast mode and the explicit
  approximations must stay within their documented bounds. It also runs
  every kernel at O0 with line information, loads a kernel with each JIT
  option, holds the kernels the simulator cannot run (inline PTX and MLIR
  operations through pointers, memory clobbers, `nvvm.read.ptx.sreg.nsmid`
  written as MLIR) to plain Rust, and, when the CUDA toolkit is installed,
  assembles the SM90 kernels with `ptxas` for sm_90a and sm_100a and
  checks with `nvdisasm -g -c` that the SASS shows the Rust lines. The packed half-precision
  operations run over every Float16 value against 32 partners with every
  option (and every BFloat16 value for BFloat16 fma), the Float32
  conversions over half a million bit patterns, and must give the
  simulator's bits; `ex2.approx.f16x2` runs on every input and must give
  one of the two Float16 values on either side of the exact result (the
  exact result itself when it is a Float16, and the correctly rounded one
  when that is zero or infinite). A
  kernel over tensors of runtime extents (`feature_kernels::dynamic_tiles`)
  and `runtime_gemm` with runtime extents are each compiled once and run
  at several problem sizes, against plain Rust, the simulator and, for the
  GEMM, the static-bound kernel's bits, on random small integers and on
  random values whose sums round. It runs
  a dynamic shared-memory region behind static buffers at launch sizes from
  512 bytes to the device's opt-in limit, through a window of exactly 48 KB
  (and checks that a word more than the limit is refused), and reads the
  shared-memory size registers, and the raw `%dynamic_smem_size` the launch
  passes, for 19 layouts, static, in the dynamic window and with a region,
  among them windows of exactly 128 bytes and 48 KB. The feature kernels
  it runs that spin on an mbarrier give up after a bounded number of polls
  and count it, so a lost arrival fails the test instead of hanging the GPU.
  Run it with `cargo test --features cuda --test gpu`. It is skipped when
  there is no GPU. Kernels run at the defaults plus the environment, so
  `CUTE_RS_COMPILER_OPT=--opt-level=0` runs the whole suite at O0.
* `tests/tensor_ssa.rs` checks what `TensorSsa` emits (a row-major 4x4
  Float32 `load_ssa` is four 16-byte loads, one scalar operation per
  element, nothing for broadcasting, sequential `max.NaN.f32` and add
  chains for reductions, the paired FP8 `cvt` and the Int8 and Int4 to
  BFloat16 fast paths), the build panics of its misuse, and
  `print_tensor`'s text. Its kernels (`feature_kernels::tensor_ssa_cases`:
  loads and stores, arithmetic and comparisons of every type, promotion,
  broadcasting, indexing, reductions, the conversion matrix, the Float64
  double-rounding regression, bitcasts and printing) run in
  `tests/features.rs` and `tests/gpu.rs` with the rest.
* `tests/numeric_types.rs` holds the type table to CuTe DSL's metadata, the
  generated promotion table to an independent implementation of CuTe
  DSL's rules (and mixed-type arithmetic to its types, statically), and
  every narrow float codec to a search over the format's encodings, for
  every encoding and an f32 sweep (ties, subnormals, the largest finite
  values, overflow, signed zeros, infinities, NaNs) in each rounding mode,
  with and without `.satfinite` and `.relu`.
* `tests/features.rs` runs a kernel for every scalar type, operation and
  intrinsic, for `Var`s, `carry_struct!` structs and the loop hints, for
  layouts with runtime extents and strides evaluated at runtime coordinates,
  and for tensors of launch-time extents tiled with `local_tile`, walked by
  a `for_range` over the runtime number of tiles and copied by a tiled copy
  under a bounds predicate (`tests/feature_kernels`, also at many extents
  from one built kernel), and holds the simulator to a reference in
  plain Rust (the carry chains against u128 arithmetic: 128-bit sums and
  differences of four u32 words, 256-bit ones of four u64 words with
  carries through every word, a 64x64->128 product from `mad_cc` and
  `madc`, chains inside loops and branches, and chains around
  divisions, warp collectives, the special-register reads and
  `cp.async.mbarrier.arrive` that keep the flag). The GPU test then holds the
  GPU to the simulator on the same kernels, except those needing SM90
  (`feature_kernels::sm90`), which are simulated and compiled for sm_90
  only, and Float32 `redux.sync` and the UE8M0 pairs
  (`feature_kernels::sm100a`), simulated and compiled for sm_100a; `fmax`
  and `fmin` (`feature_kernels::sm86`) run on GPUs from sm_86 on, and the
  FP8 conversions (`feature_kernels::sm89`: every Float16 value to E4M3FN
  and E5M2 with and without `.relu`, every byte back to Float16, an f32
  sweep with ties, NaNs, infinities and the saturation boundaries through
  each conversion, and every byte of every 8-bit format to Float32) from
  sm_89 on. The type-system kernels (Boolean and TFloat32 conversions, an
  MMA on TFloat32 fragments, mixed-type promotion, Int128 arithmetic at its
  edge values with 128-bit scalars and buffers, Int4 over all 16 values) are
  in `feature_kernels::all`. The warp collectives are checked against the PTX
  ISA's pseudo-code written again in the tests: shuffles of Int8, Uint16,
  Float16, BFloat16, Int32, Float32, Int64 and Float64 values in every
  kind, over the whole warp, in segments of 8 and 16 lanes, past the warp's
  ends and with member masks in divergent code (NaN payloads must arrive
  unchanged), and with per-lane operands from the data (also over 200
  random seeds); butterfly reductions over every group size; every integer
  `redux.sync` kind; `match.any` and `match.all` on 32- and 64-bit values;
  all of them over random partitions of the warp; `sync_warp` with
  half-warp masks; `fmax` and `fmin` in all eight combinations of `abs`,
  `nan` and `ftz` on every pair of signed zeros, infinities, NaNs (with
  payloads, negative, signaling), subnormals and extremes; and Float32
  `redux.sync` with `abs` and `nan`. The packed and scalar half-precision operations (every option, the tuple
  forms, predicates), the conversions and `pack`/`unpack` are checked
  against references written again from the PTX ISA in plain Rust (exact
  results from an f64 sum and its error), on tables of signed zeros,
  subnormals, extremes, ties, infinities and NaNs, on the input data and on
  random bits over many seeds; `exp2_packed_f16x2` and the int8 and int4
  conversions on every input. The simulator's own unit tests hold the
  half-precision instructions to such a reference on every Float16 and
  BFloat16 value against a set of partners, for each operation and option. The integer and bit kernels spread
  their data over every bit (the generated 64-bit inputs are below 2^48), so
  64-bit `mul_hi`, `bfe`, `bfi`, `clz` and `bfind` see negative values and
  set high bits, and they try `bfe` and `bfi` at the ISA's edges as
  constants, at runtime and one of each, including starts and lengths whose
  low byte is 0 (0x100, 0x200). Hand-offs through flags in every ordered
  form, every cache hint on every element type, strong vectors of 8- and
  16-bit elements, the prefetches, tiled G2R and R2G copies with each
  memory order and hint, and copies built by `make_cotiled_copy` run there
  too, and so on the GPU.

## TMA descriptors, stmatrix and setmaxnreg

Roadmap parts m7-tma-descriptor and m8-stmatrix-setmaxnreg. The TMA instructions that use
these descriptors came with m7-tma-load and later parts (below).

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cutlass.GridConstant[T]`, `cutlass.grid_constant` | `param_bytes(name, size, align, true)` (one `.param .align A .b8 k_param_i[N]`, `llvm.byval` and `nvvm.grid_constant` set on the lowered `llvm.func`); `StructParam::address()` is its generic address in place (`cvta.param.u64`, PTX 7.7). The simulator keeps it read-only |
| a TMA atom's `CUtensorMap` (`TmaTrait`), passed as a 128-byte `__grid_constant__` | `tma_descriptor(&tensor, &box_dims, stride_order, &TensorMapOptions)`: a hidden parameter appended after the kernel's own (launches pass the same arguments), described by a `TmaDescSpec` whose extents, byte strides and start are expressions over constants and scalar parameters. `KernelSignature::hidden_tensor_map` turns it into a `TmaDesc` for each launch; the CUDA launcher encodes that with `cuTensorMapEncodeTiled` from the tensor's device address, the simulator with its own 128-byte encoding (`TmaDesc::to_sim_bytes`: magic `CuTM`, version, fields) from its address map. `TmaDescriptor::get_ptr()` is the descriptor's generic address |
| `cutlass.experimental.cuda.create_tensor_map_tiled(global_address, dtype, global_dims, global_strides, box_dims, traversal_strides=, interleave=, swizzle=, l2_promotion=, oob_fill=, tma_format=)`, `create_tensor_map_tiled_from_view(tensor, box_dims, stride_order=, ...)`, `TensorMap` (`element_type`, `box_volume`, `shared_storage_dtype`, `shared_storage_bytes`, `global_tx_bytes`, `get_ptr`) | `create_tensor_map_tiled(address, dtype, &dims, &strides_16b, &box, &TensorMapOptions)` (strides of dimensions 1 and up in 16-byte units, TMA order), `create_tensor_map_tiled_from_view(&TensorMapView, &box, stride_order, &opts)`, `TensorMap` with the same queries; also `cute::cuda::tensor_map`. A kernel takes one with `param_tensormap(name)`, a launch with `KernelArg::tensor_map(&map)`; `TmaDescriptor::get_ptr()` is `get_ptr` |
| `TensorMapDataType`, `TensorMapDataFormat`, `TensorMapInterleave`, `TensorMapSwizzle`, `TensorMapL2Promotion`, `TensorMapFloatOOBFill` | the same names and values (the driver's codes). The FP4/FP6 encodings (`f416u4_align8b`, `B4X16`, ...) are supported (m7-tma-sm100-modes); the `s128b_atom_*` swizzles exist and are refused until m10-umma-layouts |
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

### TMA atoms (m7-tma-atom)

Building and partitioning TMA atoms; the next section issues the loads.

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cpasync.CopyBulkTensorTileG2SOp(cta_group)`, `CopyBulkTensorTileG2SMulticastOp(cta_group)`, `CopyBulkTensorTileS2GOp()`, `CopyReduceBulkTensorTileS2GOp(reduction_kind)`, `CtaGroup`, `ReductionKind` | the same names in `cute::cpasync` (`CtaGroup::{ONE, TWO}`, `ReductionKind::{ADD, MIN, MAX, INC, DEC, AND, OR, XOR}`), converting into `CopyOp::TmaLoad { multicast, cta_group }`, `CopyOp::TmaStore` and `CopyOp::TmaReduce(kind)`. `CtaGroup::TWO` is refused until m10-cta-pair |
| `cpasync.make_tiled_tma_atom(op, gmem_tensor, smem_layout, cta_tiler, num_multicast=1, *, gmem_coord_tensor=None, internal_type=None)` -> `TmaInfo` (`.atom`, `.tma_tensor`, `.smem_layout`; unpacks to `(atom, tma_tensor)`) | `cpasync::make_tiled_tma_atom(op, &gmem, smem_layout, cta_tiler, num_multicast)` -> `TmaInfo { atom, tma_tensor, smem_layout }` (`let (atom, t) = info.into()`); `make_tiled_tma_atom_with(.., &TmaAtomOptions)` takes the keyword arguments (`gmem_coord_tensor`, `internal_type`; m7-tma-sm100-modes). C++ CuTe's algorithm (`make_tma_atom`): the box is the SMEM layout's largest contiguous run along unit steps of the global modes, merged up to 256 elements, other modes appended with extent 1, folded to at most 5 dimensions, split across the multicast; the swizzle is the layout's restated on bytes (`Swizzle<1|2|3,4,3>`). A staged layout (one more mode than the tiler) uses its first stage. The atom registers a hidden descriptor parameter (`atom.tma_info().unwrap().descriptor()`), one thread issues it (`thr_id` `1:0`) and it moves the whole box (`num_val_src`). The TMA tensor is a coordinate tensor over the global shape with scaled-basis strides (`(m,k):(1@1,1@0)` for a K-major tile) |
| `cpasync.tma_partition(atom, cta_coord, cta_layout, smem_tensor, gmem_tensor)` -> `(tXsX, tXgX)` | `cpasync::tma_partition(&atom, cta_coord, &cta_layout, &smem, &gmem)` -> `(tXsX, tXgX)`, mode 0 `(TMA, TMA_Iter)`, both views starting at the CTA's multicast share. The list form for gather4/scatter4 is `tma_partition_indexed`, and for the override copies `tma_partition_override` (m7-tma-sm100-modes) |

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

### TMA loads, copy keyword arguments and atom fields (m7-tma-load)

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cute.copy(tma_atom, tAgA[(None, k)], tAsA[(None, s)], tma_bar_ptr=mbar, cache_policy=p, tma_desc_ptr=d)` | `copy_with(&atom, &tAgA.slice((.., k)), &tAsA.slice((.., s)), CopyArgs::new().tma_bar_ptr(&mbar).cache_policy(p).tma_desc_ptr(d))`: one `cp.async.bulk.tensor.Nd.shared::cluster.global.tile.mbarrier::complete_tx::bytes[.L2::cache_hint]` per box (sm_90, PTX 8.0), as inline PTX (32-bit shared addresses, the descriptor's generic address, one register per coordinate, a memory clobber). The coordinates are the TMA tensor's element at each box's start, the destination the box's unswizzled start. As in CuTe DSL the copy elects one lane of each warp that executes it: call it from one warp, unelected; `mbarrier_init`, `mbarrier_init_fence` and `expect_tx` stay the program's |
| `cute.copy(atom, src, dst, *, pred=, unroll_factor=, **kwargs)` | `copy_with(&atom, &src, &dst, CopyArgs { .. })` (or `&CopyArgs`; `CopyOptions` is the same struct). `pred` takes one flag per source element or one per atom application (`(ATOM_REST, REST...)`). `unroll_factor` loops over the rest modes (which may then be runtime extents) with `LoopUnroll(count)` when both operands are memory or coordinates, and unrolls fully otherwise. `copy` accepts a coordinate tensor (`Tensor<Ident>`) as a TMA load's source |
| `cute.copy(atom, [src, aux...], [dst, aux...])` | `copy_aux(&atom, (&src, &[AuxOperand::from(&t)]), (&dst, &[]), args)`; the gather4/scatter4 atoms take the index tensor's partition and the override atoms the residue's (m7-tma-sm100-modes). `tma_multicast=` (the block API) panics, pointing to m10-s2t-copies |
| `atom.with_(tma_bar_ptr=..., ...)`, `tiled_copy.with_(...)` | `atom.with_(CopyArgs::new().tma_bar_ptr(&mbar))`, `tiled_copy.with_(..)`: the fields every later copy with it uses; giving a bound field again panics |
| `mma.set(field, v)`, `mma.get(field)`, `mma.with_(...)`, `warpgroup.Field`, `tcgen05.Field`, `warp.Field` | `MmaAtom`/`TiledMma::set(MmaField::ACCUMULATE, v)`, `get(field) -> Boolean`, `with_(field, v)`; `MmaField::{ACCUMULATE, NEGATE_A, NEGATE_B, SFA, SFB, METADATA, DISABLE_OUTPUT_LANE}`. `TiledMma` and `MmaAtom` are `Carry`: a field set inside `for_range_carry` is carried, as CuTe DSL's K loops need. No `mma.sync` atom takes a field; WGMMA (m8-wgmma-core) is the first |

* **Fields per atom.** A TMA load takes `tma_bar_ptr` (needed), `cache_policy`,
  `tma_desc_ptr` and `mcast_mask` (ignored, as CuTe DSL ignores it for the plain load);
  `CopyOp::G2R`/`R2G` take `cache_policy`; the TMA stores, multicast loads and bulk copies
  take theirs (next sections); `byte_mask` belongs to m7-tma-sm100-modes. Any other field, a
  missing `tma_bar_ptr`, a data tensor as a TMA load's source, a predicated TMA load, a
  CTA-pair load
  (m10-cta-pair) and a box start at a static offset that is not 128-byte aligned fail
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

### TMA stores, reductions, L2 cache hints and prefetches (m7-tma-store)

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

### Multicast masks, multicast and peer-CTA copies, st.async (m7-cluster-copies)

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

### In-kernel tensor-map updates (m7-tensormap-update)

Descriptors in memory that kernels modify, for grouped and pointer-array GEMMs whose groups
each need their own tensor map.

| CuTe DSL | cute-rs |
|---|---|
| `cpasync.copy_tensormap(tma_atom, tensormap_ptr)` | `cpasync::copy_tensormap(&atom, &ptr)`: sixteen 64-bit loads of the atom's grid-constant descriptor and stores to `ptr` (global, shared or generic, any element type), by every thread that executes it |
| `cpasync.update_tma_descriptor(tma_atom, gmem_tensor, tma_desc_ptr)` | `cpasync::update_tma_descriptor(&atom, &tensor, &ptr)`: `tensormap.replace.tile.{global_address,global_dim,global_stride}{.global\|.shared::cta}.b1024.{b64\|b32}` for the address, each TMA dimension's extent and dimensions 1 and up's byte strides, from the tensor's modes in the order the atom's tensor had them (runtime pointers, extents and strides allowed; folded dimensions refused). sm_90a, sm_100a, sm_110a, sm_120a (the f families from PTX 8.8), PTX 8.3 |
| the experimental `tensormap_replace(field, ptr, new_value=, ord=)` | `cpasync::tensormap_replace(&ptr, TensormapField::{GlobalAddress, GlobalDim(ord), GlobalStride(ord)}, value)`. A stride is in bytes, ordinal 0 being dimension 1's (C++ CuTe's numbering; `ptxas` from CUDA 12.5 takes bytes, older ones took 16-byte units). The box, element strides, rank, element type and swizzle fields are not exposed |
| `cpasync.fence_tma_desc_acquire(ptr)`, `fence_tma_desc_release()`, `cp_fence_tma_desc_release(gptr, sptr)` | the same names: `fence.proxy.tensormap::generic.acquire.gpu [p], 128`, `fence.proxy.tensormap::generic.release.gpu` and `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [g], [s], 128` (sm_90, PTX 8.3) |
| `cute.copy(..., tma_desc_ptr=ptr)` | `CopyArgs::tma_desc_ptr(ptr.toint_i64())` (m7-tma-load): the copy reads the descriptor at that generic address |
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

### Im2col and the SM100 TMA modes (m7-tma-im2col, m7-tma-sm100-modes)

| CuTe DSL | cute-rs |
|---|---|
| `cpasync.CopyBulkTensorIm2ColG2SOp(cta_group)`, `CopyBulkTensorIm2ColG2SMulticastOp(cta_group)`, `CopyBulkTensorIm2ColS2GOp()` | the same structs (`CopyOp::TmaLoadMode`/`TmaStoreMode` with `TmaOpMode::Im2col`): `cp.async.bulk.tensor.{3,4,5}d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [smem], [tmap, {c, w, (h, (d,)) n}], [mbar], {offsets}` and `.global.shared::cta.im2col_no_offs.bulk_group` (sm_90, PTX 8.0), inline PTX |
| `cpasync.make_im2col_tma_atom(op, gmem_tensor, smem_layout, cta_tiler, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt, num_multicast, *, internal_type)` | `cpasync::make_im2col_tma_atom(op, &gmem, smem_layout, cta_tiler, &Im2colOptions)`; `Im2colOptions::conv(filter_srt, lower_padding, upper_padding, stride, dilation)` derives the corners (`-lower_padding`, `upper_padding - (filter - 1) * dilation`) and offsets. The tensor is `((w, (h, (d,)) n), c)`, channels contiguous; the SMEM tile `(pixels, channels)` K-major. C++ CuTe's `make_tma_atom_im2col`: channels per pixel = the SMEM layout's contiguous run (at most 256), pixels per column its next run (split among a multicast). The TMA tensor has the GEMM shape `(Q*P*Z*N, (C, S, R, T))`, M linear (C++ composes its multi-mode M with a linear layout the same way); a copy turns each box's `(m, c, s, r, t)` into `w = lower_w + q * stride_w` and offsets `lower_srt + s * stride_srt` |
| `nvgpu.make_im2col_tma_atom_A(...)` | `nvgpu::make_im2col_tma_atom_a` (m10-port-fp16-gemm; see the MMA-aware TMA atoms) |
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

## Warpgroup MMAs (m8-wgmma-core)

Roadmap part m8-wgmma-core: the sm_90a warpgroup MMA from shared-memory descriptors. There
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

## Several kernels in one module (m6b-multi-kernel)

A CuTe DSL `@cute.jit` host function may launch several `@cute.kernel`s; `cute.compile` puts
them all into one GPU module (one PTX and cubin), which the compiled function loads once and
looks each kernel up in by name. In cute-rs that unit is a `KernelModule`:

| CuTe DSL (Python) | cute-rs |
|---|---|
| a `@cute.jit` function launching kernels `a` and `b` | `KernelModule::new("name", [a, b])?`: names must differ, kernels built for different targets (`KernelOptions::target`) cannot share a module, and module globals of one name must be declared alike (else `ModuleError`). `KernelModule::from(kernel)` is a module of one |
| `cute.compile(host_fn, ...)` | `module.compile_with(&toolchain, &options)` → `CompiledModule`: one `gpu.module` (`to_mlir`), one `mlir-opt` lowering, one `ptxas` run, one PTX with an `.entry` per kernel; `kernels()`/`kernel(name)` give each kernel's `CompiledKernel` (the module's images under its own name, signature and diagnostics). A module of one kernel compiles exactly as `Kernel::compile_with` (same MLIR, PTX, cubin and cache key) |
| the compiled function loading its module and finding each kernel | `ctx.load_kernel_module(&compiled)` → `CudaModule`: one `cuModuleLoadDataEx`, `function(name)` per kernel (launched with the usual checks), `global(name)` for the shared module globals |
| module-level `cutlass.Array` globals shared by the kernels | one `__device__`/`__constant__` symbol per name for the whole module: a global one kernel writes, the next reads (on the GPU, the simulator and through the export) |
| simulating the host function | `module.sim_module()` → `SimKernelModule::simulate(name, cfg, args)` over shared globals (`read_global`, `write_global`) |
| `export_to_c` of the compiled function | `compiled.export_c(dir, name, prefix)`: one header and source with `<p>_Module_t`, `<p>_Module_Load`/`_Unload`, and `<p>_<kernel>_Args_t`/`<p>_<kernel>_Launch` per kernel (macros `<P>_<KERNEL>_KERNEL_NAME` and `_INDEX`); the manifest is export format 2 (`module` and `kernels` lines, a `signature` section per kernel). A module of one kernel exports byte for byte as the kernel does (format 1) |
| `cute.runtime.load_module` | `ExportedModule::read(path)` (formats 1 and 2), `ctx.load_exported_module(path)` |
| `--enable-tvm-ffi` on a function with several kernels | `TvmFfiFunction::from_module(typed_kernel, &compiled, config)` (or `from_exported_module`), one `tvm_ffi_export!` per kernel: on each thread and device the module is loaded once for all of them (`tvm_ffi::shared_module_loads`) |
| the JIT cache keyed on the compiled function | the compile caches key a module on every kernel's IR (its MLIR), signature, the options and the toolchain: changing any kernel recompiles it |

What a module shares: the target, the PTX ISA version (the newest any kernel needs), the
optimization level, line information and every other compile option, and the module
globals. What stays per kernel: parameters, block size (`.reqntid`/`.maxntid`),
`.minnctapersm`, shared memory and its dynamic window (one `extern .shared` declaration, at
the strictest alignment any kernel asks for). Two kernels' static shared buffers of one name
become two symbols (the later kernel's is renamed `<kernel>_<buffer>` in the PTX); with line
information the kernels share one debug compile unit. `examples/split_kv_decode.rs` (ported
from CuTe DSL's MLA decode: a split-KV kernel, then `reduction_kernel`) and
`examples/tvm_ffi_module.rs` (two TVM-FFI symbols sharing a module's globals) show it;
`tests/multi_kernel.rs` and `tests/gpu_multi_kernel.rs` test it.

## WGMMA variants and distributed shared memory (m8-wgmma-variants, m8-dsmem)

Roadmap parts m8-wgmma-variants and m8-dsmem. Neither runs on the SM89 GPU: they are held
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

## The Hopper dense GEMM and the m8 APIs at a glance (m8-port-hopper)

Roadmap part m8-port-hopper ports `dense_gemm.py` as `examples/hopper_gemm.rs`, keeping its
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

## TMA pipelines (m9-pipeline-tma)

The Hopper mainloop and epilogue pipelines, in `cute::pipeline`. Simulated (single CTA,
2x1, 1x2 and 2x2 clusters, seeded schedules) and assembled for sm_90a; no SM90 GPU runs
them yet.

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cutlass.pipeline.PipelineTmaAsync.create(*, num_stages, producer_group, consumer_group, tx_count, barrier_storage=None, cta_layout_vmnk=None, tidx=None, mcast_mode_mn=(1, 1), enable_multicast_signaling=False, defer_sync=False, name='')` | `pipeline::PipelineTmaAsync::create(n, producer_group, consumer_group, tx_count, PipelineTmaAsyncOptions { barrier_storage, cta_layout_vmnk, tidx, mcast_mode_mn, enable_multicast_signaling, defer_sync, name })`, implementing `Pipeline` and `PipelineBarriers`: `producer_acquire` waits on the empty barrier, then the elected lane arms the full barrier with `mbarrier.arrive.expect_tx`; `producer_commit` is a no-op (the TMA's `complete_tx` completes the phase); `producer_get_barrier` is the TMA's `tma_bar_ptr`; `consumer_release` arrives from the signalling lanes only (lane 0 of each warp for one CTA, so count consumers in warps times multicast partners, or set `enable_multicast_signaling`). A dynamic `cta_layout_vmnk` and `mcast_mode_mn` `(0, 0)` are refused. Init: `fence.mbarrier_init` and a CTA barrier, or a relaxed cluster arrive and wait. Deviation: a peer release arrives at cluster scope (`.release.cluster`), where CuTe DSL uses the CTA-scope default |
| `PipelineTmaAsync.init_empty_barrier_arrive_signal(cta_layout_vmnk, tidx, mcast_mode_mn)` | `pipeline::init_empty_barrier_arrive_signal(&layout, tidx, mode) -> (Int32, Boolean)` |
| `cutlass.pipeline.PipelineTmaStore.create(*, num_stages, producer_group)`; `producer_acquire()`, `producer_commit()`, `producer_tail()` | `pipeline::PipelineTmaStore::create(n, producer_group)` with inherent, state-free methods (not the `Pipeline` trait, whose methods take a state): `cp.async.bulk.wait_group.read n-1`, `cp.async.bulk.commit_group`, `wait_group.read 0`; `consumer_wait` / `consumer_release` panic |
| `cutlass.pipeline.TmaStoreFence(num_stages)` | `pipeline::TmaStoreFence::new(n)`, also a `SyncObject`: `arrive` (commit_group), `wait` (wait_group.read n-1), `arrive_and_wait`, `tail` (wait_group.read 0); `get_barrier`, `max` and `arrive_and_drop` panic (the trait's `max` is `None`) |

## GEMM host helpers (m9-gemm-host-helpers)

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

## Blackwell layouts and UMMA descriptors (m10-umma-layouts)

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.SmemLayoutAtomKind`, `tcgen05.make_smem_layout_atom`, `make_sparse_smem_layout_atom` | `tcgen05::SmemLayoutAtomKind` (with `MN_SW128_32B`), `tcgen05::make_smem_layout_atom(_dtype)`, `make_sparse_smem_layout_atom(_dtype)`. `MN_SW128_32B` has CuTe C++'s 4 rows (CuTe DSL's has 8; both tile alike K-first, and only the 4-row atom stays canonical when tiled MN-first) |
| `tcgen05.tile_to_mma_shape` | `tcgen05::tile_to_mma_shape(atom, shape, order)` (0-based order, static shapes) |
| `tcgen05.make_umma_smem_desc`, `helpers.int_to_smem_descriptor`, `smem_descriptor_to_int` | `tcgen05::make_umma_smem_desc` (build-time fields ORed with the runtime start address; sm_107's 15-bit fields from the target; `next_src` unsupported), `umma_desc_fields`, `SmemDescriptor`, `UmmaLayoutType`, `UmmaDescEncoding`, `UmmaOperandMap` (the PTX-ISA decoder for the simulator), `int_to_smem_descriptor`, `smem_descriptor_to_int` |
| `cutlass.utils.BlockScaledBasicChunk`, `Sm103BlockScaledBasicChunk`, `tile_atom_to_shape_SF` | `utils::BlockScaledBasicChunk`, `Sm103BlockScaledBasicChunk`, `tile_atom_to_shape_sf` (static shapes) |
| `cutlass.utils.sm100.compute_acc_tmem_cols_per_stage`, `compute_epilogue_tile_size`, `cutlass.utils.compute_epilogue_tile_shape` | `utils::sm100::compute_acc_tmem_cols_per_stage`, `compute_epilogue_tile_size`, `utils::compute_epilogue_tile_shape` |
| `cute.arch.get_max_tmem_alloc_cols`, `get_min_tmem_alloc_cols`, `is_tmem_allocation_exclusive`, `SM100_TMEM_*` | the same names, taking an `Arch` (`a`/`f` targets map to their base) |

Dynamic shared-memory addresses: a buffer in the dynamic window is addressed as the window's pointer plus its start (LLVM 21's `--expand-strided-metadata` dropped a `memref.view`'s offset), so kernels over 48 KB compile correctly with LLVM 21, 22 and 23 (`tests/dyn_smem_address.rs`).

## Tensor memory (m10-tmem-core)

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

## TMEM loads and stores (m10-tmem-ld-st)

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

## TMEM load-reduce, packed f32x2 and Rubin packed math (m10-ldred-packed-math)

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

## SM120 block-scaled warp MMAs (m10-sm120-blockscaled)

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

## Cluster launch control (m10-clc-scheduling)

| CuTe DSL | cute-rs |
|---|---|
| `cute.arch.issue_clc_query(mbar_ptr, clc_response_ptr, multicast=True)` | `issue_clc_query(&mbar_ptr, &response_ptr, multicast)`: `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{.multicast::cluster::all}.b128` as inline PTX (LLVM 21 has no NVVM op); sm_100 and PTX 8.6, the multicast form sm_100a-class targets. The response pointer is 16-byte aligned shared memory of any byte-sized type |
| `cute.arch.clc_response(result_addr) -> (x, y, z, is_valid)` | `clc_response(&ptr) -> (Int32, Int32, Int32, Int32)`: a 16-byte `ld.shared.b128` and `query_cancel.is_canceled` / `get_first_ctaid::{x,y,z}` in one inline-PTX statement, so no Int128 type is needed |
| `cutlass.pipeline.PipelineClcFetchAsync.create(*, num_stages, producer_group, consumer_group, tx_count, barrier_storage=None, producer_mask=None, consumer_mask=None, cta_layout_vmnk=None, defer_sync=False, name='')` | `pipeline::PipelineClcFetchAsync::create(n, producer_group, consumer_group, tx_count, PipelineClcFetchAsyncOptions { barrier_storage, cta_layout_vmnk, defer_sync, name })` (the masks are left out: CuTe DSL overrides both). The scheduler warp of CTA 0 acquires (CTA 0's empty barrier), then lane `i` arms CTA `i`'s full barrier with the 16 expected bytes (`PipelineOp::ClcLoad`); `producer_commit` is a no-op; consumers release to CTA 0's empty barrier (at cluster scope in a cluster), so its count covers the whole cluster's consumers |
| `ClcDynamicPersistentTileSchedulerParams(problem_shape_ntile_mnl, cluster_shape_mnk, swizzle_size=1, raster_along_m=True)`, `get_grid_shape()`; `ClcDynamicPersistentTileScheduler.create(params, block_idx, grid_dim, clc_response_ptr, insert_fence=True)`, `initial_work_tile_info`, `get_current_work`, `work_tile_info_from_clc_response`, `advance_to_next_work(mbarrier_addr)`, `num_tiles_executed` | `utils::ClcDynamicPersistentTileSchedulerParams::new(tiles, cluster, ClcDynamicPersistentTileSchedulerOptions { swizzle_size, raster_along_m, fallback_cluster_shape_mnk })` (the fallback shape's schedule is kept, `fallback()`, and `create` takes it in the fallback arm of `mixed_cluster_arms`) and `utils::ClcDynamicPersistentTileScheduler::create(&params, block_idx(), grid_dim(), &response_ptr, insert_fence)`, the same methods; `Carry` with only the tile count. The visiting order is the static scheduler's (`swizzle_and_rasterize`, FastDivmod for N-major). Deviation: a swizzled grid launches the padded layout's clusters (CuTe DSL's unpadded count leaves tiles unvisited when the swizzle does not divide the clusters), whose padding tiles the kernel skips |
| `cute.experimental.utils.is_preferred_cluster(shape)`; scheduler params' `fallback_cluster_shape_mnk=`, `_select_active_params`, `runtime_cluster_dims()`; the mixed-cluster expansion pass and `smem_merge_branch_allocs=True` | `utils::is_preferred_cluster([x, y, z])` (`%cluster_nctaid` compared mode by mode) and `utils::mixed_cluster_arms(preferred, Some(fallback), \|arm\| body)` (m10-port-clc-clusters): no compiler pass; the body closure is built twice, under `if_else(is_preferred_cluster(..))`, each time with the arm's static `ClusterArm::cluster_shape_mnk`, so cluster-shape math stays static per arm (shared memory both arms use is allocated before the call). While an arm is built, `current_cluster_arm()` names it: `ClcDynamicPersistentTileScheduler::create` picks the params' fallback schedule in the fallback arm (the build-time `_select_active_params`), and schedulers, or anything checked with `check_cluster_shape_in_arm(what, shape)`, built for another shape panic. `check_mixed_cluster_shapes(preferred, fallback, grid)` checks the tutorial's constraints (equal depth, the fallback divides the preferred shape, the preferred shape divides the grid). The simulator's CLC queue hands each cluster only clusters of its own shape |

The simulator keeps the grid's clusters that have not started in a queue: `try_cancel` removes the
first one that has not launched (each at most once; a cancelled cluster never starts) and answers
"not cancelled" once it is empty. `SimOptions::clc_launch_seed` launches clusters at the front of
the queue at random before each query, so which are left to cancel changes; `SimReport::clc_cancelled`
counts the cancellations. The response is the simulator's own 16-byte encoding (x, y, z, then a tag
word with the cancelled bit), delivered as an async-proxy write that completes 16 bytes of tx on
each target CTA's mbarrier, as m2's in-flight copies do; a multicast query after a CTA of the
cluster has exited is reported. `tests/clc_scheduling.rs` runs a persistent kernel whose clusters
claim more tiles through CLC for several grids, cluster shapes, swizzles, resident-cluster counts
and seeds (each tile once, cancelled clusters never run, the static scheduler's order), plus the
misuse reports; Blackwell CLC cannot run on the SM89 pod, so the PTX is assembled with
`ptxas -arch=sm_100a`.

## MMA-aware TMA atoms and the first Blackwell GEMM (m10-port-fp16-gemm)

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

## CTA pairs: 2-SM tcgen05 MMAs and 2-CTA TMA loads (m10-cta-pair)

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.MmaF16BF16Op(..., cta_group=tcgen05.CtaGroup.TWO, ...)`, and `MmaTF32Op`, `MmaI8Op`, `MmaF8F6F4Op`, `MmaFP8Op` with `CtaGroup.TWO`; `make_trivial_tiled_mma(..., CtaGroup.TWO, ...)` | the same constructors with `CtaGroup::TWO`: M of 128 or 256, N a multiple of 16 from 16 to 256 (32 for an 8-bit MN-major B), A in shared memory. The atom's `thr_id` is the pair (size 2) and A, B and C are split in row halves (C++'s `SM100_MMA_*_2x1SM_SS` traits), so `tiled_mma.get_slice(rank_in_pair)` partitions each CTA's half, `partition_shape_C` is `((M/2, N), 1, 1)`, and `make_fragment_C` lays it out as `UMMA::tmem_frg_2sm` (PTX Layout A for M = 256; Layout B for M = 128: columns `N/2..N` in lanes 64 to 127) |
| `cute.gemm(tiled_mma, ...)` with a pair MMA | `gemm` as for one CTA, called by a warp of the even CTA only (`tcgen05.mma.cta_group::2`); it reads A and B at the descriptors' addresses in both CTAs' shared memory and writes D into both CTAs' TMEM |
| `tcgen05.commit(mbar, mask, cta_group=CtaGroup.TWO)` | `tcgen05::commit(&mbar, Some(mask), CtaGroup::TWO)`; the mask names the CTAs whose mbarrier (at the same offset) hears of the completion, both CTAs of the pair for a pair MMA |
| `cpasync.CopyBulkTensorTileG2SOp(cta_group=CtaGroup.TWO)`, `CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)` (TiledTmaLoadEnum `sm_100_2sm`), `make_tiled_tma_atom_A/B` with a pair's tiled MMA | the same ops; `make_tiled_tma_atom_a/b` give each CTA's half box (the MMA's value mode). The load is `cp.async.bulk.tensor....cta_group::2` and signals the mbarrier at the same offset in the pair's even CTA (C++'s `Sm100MmaPeerBitMask`), so one CTA expects both halves' bytes. Refused for an sm_90 target; `.cta_group::2` loads need sm_100a, sm_101a, sm_110a or their f families |
| `alloc_tmem`/`dealloc_tmem`/`relinquish_tmem_alloc_permit(is_two_cta=True)`, `TmemAllocator(is_two_cta=True)` | as before (m10-tmem-core): both CTAs of the pair allocate together |

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
TMEM A) and `tcgen05.cp` with `cta_group::2` (m10-s2t-copies); gather4 and override loads
refuse `CtaGroup::TWO`.

The 2-CTA pipelines (m10-cta-pair-pipelines) are simulated by `tests/cta_pair_pipelines.rs`:
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

## Shared-memory-to-TMEM copies (m10-s2t-copies)

| CuTe DSL | cute-rs |
|---|---|
| `tcgen05.Cp128x256bOp`, `Cp128x128bOp`, `Cp4x256bOp`, `Cp4x32x128bOp`, `Cp2x64x128b0213Op`, `Cp2x64x128b0123Op` (`cta_group`, `broadcast_factor`) | `tcgen05::Cp128x256bOp::new(CtaGroup::ONE)` and peers (`broadcast_factor()`), into `CopyOp::TmemCp`; `make_copy_atom::<T>(op)` (8-, 16- or 32-bit `T`). `copy(atom, desc_tensor, tmem)` is called by a converged warp and elects the lane that issues each `tcgen05.cp.cta_group::{1,2}.<shape>` (inline PTX; do not wrap it in `elect_one`); completion through `tcgen05::commit` |
| `tcgen05.make_s2t_copy(atom, tmem_tensor)`, `tcgen05.get_s2t_smem_desc_tensor(atom, smem_tensor)` | `tcgen05::make_s2t_copy(&atom, &tmem)` (CuTe C++'s `make_utccp_copy`, after `filter_zeros`; one thread per CTA of the group, `get_slice(0)`) and `tcgen05::get_s2t_smem_desc_tensor(&tiled, &partitioned_smem)` (`get_utccp_smem_desc_tensor`: the core matrices checked as a canonical K-major UMMA layout, descriptors in 16-byte steps; 15-bit fields on sm_107) |
| `tcgen05.append_s2t_broadcast_mode(op, smem_tensor)` | `tcgen05::append_s2t_broadcast_mode(op, &smem)`: `((MN..., b:0), K)` for the broadcast ops, the tensor unchanged otherwise; panics (CuTe DSL's `ValueError`) for a non-S2T op or a mode 0 that is not `(MN, K)`. A broadcast copy whose partitioned source lacks the zero-stride mode is refused, naming it |
| `cutlass.block.block_copy(tiled_copy, src, dst, **kwargs)`, `cutlass.utils.block.block_copy` (deprecated) | `cute::block::block_copy(&atom_or_tiled_copy, &src, &dst, CopyArgs)` (`cute::utils::block::block_copy` deprecated): TMA atoms through `tma_partition` for one CTA, S2T tiled copies through slice 0 and the descriptor view (a broadcast copy's source gets its broadcast mode back after `filter_zeros`). `tma_multicast` is not supported yet (deferred to m11) |

The simulator (`src/sim/exec/tmem_cp.rs`) decodes each descriptor with `UmmaOperandMap`, writes
TMEM by the PTX shapes (the broadcast forms replicate each row into 2 or 4 lane quadrants),
queues the copy on the CTA's tensor-core pipe with the MMAs (completed by `tcgen05.commit` and
an observed mbarrier phase), keeps the source in flight until then, requires `fence.proxy.async`
after generic writes of the source, and reports a copy issued by fewer than 32 lanes (inside
`elect_one`, say). The decompressing source formats (`.b8x16.b6x16_p32`, `.b4x16_p64`) are
reached only through CuTe DSL's experimental `nvvm_wrapper.tcgen05_cp`, not its copy atoms, and
are not modelled.

## SM100 epilogue helpers (m10-epilogue-helpers)

| CuTe DSL | cute-rs |
|---|---|
| `utils.sm100.get_tmem_load_op(cta_tile_shape, layout_d, elem_ty_d, elem_ty_acc, epi_tile, use_2cta_instrs, *, tmem_warp_shape_mn=None)` | `utils::sm100::get_tmem_load_op::<Acc>(&cta_tile, layout_d, d_dtype, &epi_tile, use_2cta, tmem_warp_shape_mn) -> CopyAtom<Acc>`, and `select_tmem_load_op(.., acc_dtype, ..) -> TmemLoadChoice { shape, repeat, pack }` for the choice as data. Ported from CuTe C++'s `sm100_get_tmem_load_op` and `TMEM::op_repeater`: the tile is split by the TMEM warp layout (`(2, 2)` for a CTA pair with 64 rows per CTA, else `(4, 1)`); a warp's part must have 16 or 32 datapaths. `32x32b` for 32 datapaths (`16x256b` for M-major 16-bit and 8-bit D), `16x128b`, `16x256b` or `16x32bx2` for 16, `.pack::16b` for 16-bit accumulators, the widest repetition the columns allow. Panics (not `ValueError`) with its own messages for 8 datapaths, a warp layout that does not divide the tile, 6- and 4-bit D with 16 datapaths, and unserved width pairs |
| `utils.sm100.get_smem_store_op(layout_d, elem_ty_d, elem_ty_acc, tiled_tmem_load)` | `utils::sm100::get_smem_store_op::<D, Acc>(layout_d, acc_dtype, &tiled_tmem_load)`, and `select_smem_store_op(layout_d, d_bits, acc_bits, &load_op) -> CopyOp`: `StMatrix8x8x16bOp` (4 matrices for a repeated `16x256b` load of 16-bit D, a repeated `16x128b` load of N-major 32-bit D or a packed `16x128b` load of 16-bit accumulators; 2 for `.x1`; transposed for M-major D), `StMatrix16x8x8bOp` for M-major 8-bit D from `16x256b`, else a plain store. The simulator checks the fragments against the load bit for bit. `StMatrix` takes 32-bit elements (one per register, untransposed only: a new extension of the m8 atom, with its f32 shared-memory buffer) for N-major Float32 D |
| `utils.sm100.get_smem_layout_atom_epi(layout, element_type, epi_tile)`, `make_smem_layout_epi(epi_dtype, epi_layout, epi_tile, epi_stage)` | `utils::sm100::get_smem_layout_atom_epi(layout, dtype, &epi_tile)` (the M or N extent of the right inverse of each tile mode, through `get_smem_layout_atom_ab`) and `make_smem_layout_epi(dtype, layout, &epi_tile, stages)` (Boolean is Int8; the staged layout in the order `(1, 0, 2)` for M-major, `(0, 1, 2)` for N-major). M-major Float32 gets `MN_SW128_32B`, which `make_tiled_tma_atom` cannot express for a TMA store yet (deferred); direct stores and N-major Float32 work |
| `utils.gemm.sm100.transform_partitioned_tensor_layout(tensor)` | `utils::gemm::sm100::transform_partitioned_tensor_layout(&t)`: `((A_M, A_N), MMA_M, MMA_N, ..)` to `((A_M, MMA_M), (A_N, MMA_N), ..)`; the accumulator's trailing extent-1 mode of the atom is dropped |
| `utils.gemm.sm100.epilogue_tmem_copy_and_partition(gemm_kernel, tidx, tAcc, tCgC, epi_tile, use_2cta_instrs)`, `epilogue_smem_copy_and_partition(gemm_kernel, tiled_copy_t2r, tTR_rC, tidx, sC)` | `utils::gemm::sm100::epilogue_tmem_copy_and_partition(&cfg, tidx, &t_acc, &t_cg_c, &epi_tile, use_2cta)` -> `TmemCopyPartition { tiled_copy_t2r, t_tr_t_acc, t_tr_r_acc }` and `epilogue_smem_copy_and_partition(&cfg, &tiled_copy_t2r, &t_tr_r_c, tidx, &s_c)` -> `SmemCopyPartition { tiled_copy_r2s, t_rs_r_c, t_rs_s_c }`. `gemm_kernel`, which CuTe DSL duck-types, is `EpilogueConfig` (`cta_tile_shape_mnk`, `c_layout`, `c_dtype`, `acc_dtype`, `use_2cta_instrs`, `tmem_warp_shape_mn`, `epilog_sync_bar_id`, `epilogue_warp_id`, `num_c_stage`, `iter_acc_early_release_in_epilogue`; `EpilogueConfig::new` gives the usual four warps on barrier 2) |
| `utils.gemm.sm100.epilogue_tma_store(gemm_kernel, epi_tidx, warp_idx, tma_atom_c, tCtAcc_base, sC, tCgC_base, epi_tile, num_tiles_executed, epilogue_op, mma_tile_coord_mnl, acc_consumer_state, acc_pipeline, c_pipeline)` | `utils::gemm::sm100::epilogue_tma_store::<Acc, C, _, _>(&cfg, tidx, warp_idx, &atom, &t_ct_acc, &s_c, &t_cg_c, &epi_tile, num_tiles_executed, &op, &[tile_m, tile_n], &state, &acc_pipeline, &c_pipeline) -> PipelineState` (any `Pipeline`; the state is the next). `epilogue_op` is `&dyn Fn(TensorSsa<C>) -> TensorSsa<C>`. The loop is unrolled when the kernel is built; each subtile's `tcgen05.ld` is followed by an explicit wait (`fence_view_async_tmem_load`), which cute-rs requires; the accumulator stage is released by one elected thread of each warp, so the accumulator pipeline's consumer group counts the epilogue warps. The caller runs `c_pipeline.producer_tail()` |
| `utils.gemm.sm100.epilogue(gemm_kernel, epi_tidx, tCtAcc_base, tCgC_base, epi_tile, epilogue_op, mma_tile_coord_mnl, acc_consumer_state, acc_pipeline, tCcC_base=None, mC_mnl=None, overlapping_accum=False)` | `utils::gemm::sm100::epilogue::<Acc, C, _>(&cfg, tidx, &t_ct_acc, &t_cg_c, &epi_tile, &op, &coord, &state, &pipeline, Some(&t_cc_c), Some(&c), overlapping)`: direct stores, `R2G` atoms of the widest vector (at most 128 bits where CuTe DSL allows 256; the pointer's alignment and the registers' and C's common run bound it) with no-allocate L1 eviction; one predicate per vector from its first element's coordinate against `mC.shape` (so the contiguous extent must be a whole number of vectors); `overlapping_accum` reads the stage from the phase, drains the subtiles backwards on phase 0 and releases after `iter_acc_early_release_in_epilogue` |

`examples/blackwell_fp16_gemm.rs` gained a hook for them: `Fp16GemmConfig::build_with(acc_consumer_arrivals, make_c, epilogue)` takes the declaration of C (and anything built from it, such as a TMA store atom) and a closure that drains the accumulator, handed an `EpilogueCtx` (the TMEM tensor, `ThrMma`, tile coordinates, the shared-memory allocator and the accumulator pipeline and state); `build()` is the tutorial's own epilogue. `tests/epilogue_helpers.rs` drains `fp16_gemm_0` (1 CTA, M = 128 and 64) and `fp16_gemm_1` (a CTA pair, 2x2 warps for M = 128) through both epilogues for Float16, BFloat16 and Float32 D in both majors with an identity and a scaling operation, bit for bit against the CPU; the host tests list the choices worked from the C++ builder, and the sm_100a PTX is assembled with `ptxas`.

## Block-scaled tcgen05 MMAs (m10-umma-blockscaled)

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

## Rubin (SM107) tcgen05 MMAs and collector buffers (m10-rubin-umma)

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

## Sparse tcgen05 MMAs (m10-sparse-umma)

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

## Scale-factor SMEM/TMEM layouts and the NVFP4 tutorial GEMM (m10-port-nvfp4-gemm)

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

## Mixed-input GEMM helpers (m10-mixed-input)

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

## Rubin cutlass.utils helpers and the TMEM load with sparse compression (m10-rubin-extras)

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

## Typed MMA descriptors and the byte-address swizzle (m11-prims-descriptors-swizzle)

| CuTe DSL | cute-rs |
|---|---|
| `cutlass.experimental.primitives`: `WgmmaSmemDesc`, `WgmmaSwizzle` | `prims::WgmmaSmemDesc::{build(start, lbo_bytes, sbo_bytes, base_offset, swizzle), with_start_address, advance_start_address, start_address, leading_byte_offset, stride_byte_offset, base_offset, layout_type}`, `prims::WgmmaSwizzle`. `build` returns `Result` (offsets multiples of 16 that fit, base offset only with a swizzle); the start address is a constant, a `Value<u32>` or a shared-memory `&Ptr` |
| `Tcgen05SmemDesc`, `Tcgen05SmemSwizzle` | `prims::Tcgen05SmemDesc::build(start, &Tcgen05SmemDescFields)` (15-bit start and LBO, version 1, LBO mode, K-segment bit) and the same getters plus `version`, `leading_dim_mode`, `k_segment_offset`; `Tcgen05SmemSwizzle` is `tcgen05::UmmaLayoutType` |
| `Tcgen05InstrDesc`, `Tcgen05MxInstrDesc`, `Tcgen05MxOmmaInstrDesc`, `Tcgen05OutputFormat`, `Tcgen05InputFormat`, `MxScaleFormat` | `prims::Tcgen05InstrDesc`, `Tcgen05MxInstrDesc`, `Tcgen05MxOmmaInstrDesc`, each with `build(&..Fields)` (field ranges) and `for_mma(..)` (the types a kind takes and the M/N shapes of each CTA group), field getters and `set_sf_ids`/`set_sparse_id2`; `Tcgen05InputFormat` is a code newtype with the named constants. `sparse_format` (bit 6) is not exposed (reserved in the PTX ISA and CuTe C++); `sparsity_version` defaults to 0 as in CuTe C++ (CuTe DSL defaults to 1) |
| `TmemAddr`, `TmemAddr.from_row_col` | `prims::TmemAddr::{from_row_col, row_id, col_id, value}` (`as_ptr` is not ported: make a TMEM pointer with `make_ptr`) |
| `cutlass.Swizzle(bbits, mbase, sshift)`, `.from_name`, `.to(TensorMapSwizzle)`, `.to(Tcgen05SmemSwizzle)` | `prims::ByteSwizzle::{new, from_name, apply, period, to_tensor_map_swizzle, to_tcgen05_smem_swizzle}` (and `TryFrom`), with `to_elements(elem_bits)` / `from_elements(swizzle, elem_bits)` to and from cute-rs's element-unit `Swizzle` |
| `apply_swizzle`, `load_swizzled(ptr, sw, alignment, count=)`, `store_swizzled(ptr, value, sw, alignment)` | `prims::apply_swizzle`, `prims::load_swizzled(&ptr, sw, alignment, count) -> ScalarOrVector`, `prims::store_swizzled` (shared memory, whole-byte elements; 16-byte granules move as vectors, `ld.shared.v4`/`st.shared.v4`) |

Every value is a build-time constant (`Bits::Static`) when its inputs are, and a kernel value otherwise. The WGMMA operand fragments, `make_umma_smem_desc` and `UmmaKind::idesc` now encode through these builders, so they are the single implementation. `make_smem_tensor` keeps its 128-byte alignment (a buffer declared under-aligned stays so, and the swizzle functions, the descriptor builders and the simulator report it); allocate swizzled buffers at the swizzle's period with `make_smem_tensor_aligned` or `alloc_smem`. TMA now takes the SM100 128-byte swizzle with 32-byte atoms (`S<2,5,2>` on bytes, `TensorMapSwizzle::s128b_atom_32b` = `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` = 4), which `make_smem_layout_epi` picks for M-major Float32 epilogue tiles: the atom planner, the descriptor and the simulator accept it, and targets before sm_100 refuse it (`s128b_atom_32b_flip_8b` and `s128b_atom_64b` are still refused).

## Primitives-only MMA forms (m11-prims-mma-forms)

| CuTe DSL (`cutlass.experimental.primitives`) | cute-rs |
|---|---|
| `Tcgen05MMAKind`, `Tcgen05MMACollectorBBuffer`, `Tcgen05MMACollectorOp`, `Tcgen05MMAScaleVecSize` (1X/2X/4X) | `prims::Tcgen05MmaKind`, `Tcgen05MmaCollectorBBuffer` (`B0`..`B3`), `Tcgen05MmaCollectorOp` (= `tcgen05::CollectorOp`), `Tcgen05MmaScaleVec` |
| `tcgen05_mma(kind, cta_group, d, a, b, idesc, enable_input_d, *, collector_op, b_collector_op, a_shift, scale_input_d, write_disable_mask)` | `prims::tcgen05_mma(kind, cta_group, d, a, b, idesc, enable_input_d, Tcgen05MmaOptions { .. })`: `.collector::a::*`, `.ashift`, `scale-input-d` (an immediate), the disable-output-lane words; `b_collector_op` is the sm_107 form of m10-rubin-umma |
| `tcgen05_mma_ws(kind, d, a, b, idesc, enable_input_d, *, collector_b_buffer, collector_op, col_b_zero_mask)`, `tcgen05_mma_ws_sp(..., sparse_metadata, *, ...)` | `prims::tcgen05_mma_ws`, `prims::tcgen05_mma_ws_sp` with `Tcgen05MmaWsOptions`; `prims::tmem_frg_ws_layout` is the `.ws` accumulator layout (Layouts G, E, D) |
| `tcgen05_mma_sp_block_scale(kind, cta_group, d, a, b, idesc, enable_input_d, sparse_metadata, scale_a, scale_b, *, scale_vec_size, collector_op, b_collector_op, a_shift)` | `prims::tcgen05_mma_sp_block_scale(..., Tcgen05MmaSpBlockScaleOptions { .. })`: K = 64 `mxf8f6f4` (1X), K = 128 `mxf4` (2X) and `mxf4nvf4` (4X) |
| `mma_sp_block_scale(res, shape, scale_vec_size, block_scale_format, kind, ...)` | `prims::mma_sp_block_scale(shape, scale_vec, sf_dtype, MmaBlockScaleKind, (a_type, b_type), a, b, c, metadata, selector, ScaleOperand { reg, byte_id, thread_id } x2) -> [Float32; 4]` |

D, A (in TMEM) and the metadata/scale-factor operands take a TMEM pointer, a `TmemAddr` or an `Int32`; A and B in shared memory a `Tcgen05SmemDesc` or a `u64`. The instruction descriptor must be a build-time constant (its shape and format fields choose the instruction); its negate bits, scale-factor IDs and sparse selector pass through. The tcgen05 forms are issued by every thread that runs them (wrap them in `elect_one`), as inline PTX (LLVM 21 has no `nvvm.tcgen05.mma*`); the warp form is inline PTX too, not convergent to LLVM, so keep it out of divergent code. The simulator models the `.ws` data paths, the four B collector buffers and the A collector (`SimOptions::collector_source_check`, on by default, reports a collected operand changed in shared memory), `.ashift` per 32-lane sub-partition, `scale-input-d`, FP4 pairwise sparsity, and the warp form's byte and thread ids (cute-rs's reading of the ISA tables). Not modelled: a nonzero zero-column mask and the `.ws` max-shift field (reported), TF32 `.ws.sp` (1:2 sparsity, refused), A in TMEM for `.ws` with M of 32 or 64 (refused). A compile whose PTX declares an `llvm.*` external function (an intrinsic the backend did not know) fails with `CompileError::Unsupported`.

## The primitives NVVM layer (m11-prims-nvvm-layer)

| CuTe DSL (`cutlass.experimental.primitives`) | cute-rs |
|---|---|
| every `nvvm_wrapper` name, the enums, `print_runtime`, `dialect` | the table in the docs of `prims::nvvm`: each name's cute-rs function, or why it is not supported (`read_sreg_hw`, `trace_mark`, `wmma_load`/`wmma_store`, the `sync_restrict` fences, the mixed-type packed math and the scaled `cvt_packfloat` forms) |
| `barrier_cta_red(pred, barrier_id, kind, *, thread_count)`, `barrier_cta_red_aligned` | `prims::barrier_cta_red(pred, id, BarrierRedux, Option<Int32>) -> Uint32`, and the typed `barrier_cta_red_popc` / `_and` / `_or` (always `.aligned`; LLVM 22's `barrier.cta.red` intrinsics, `bar.red` inline PTX with LLVM 21) |
| `mbarrier_arrive` (state), `mbarrier_test_wait`, `mbarrier_try_wait`, `mbarrier_wait` | `prims::nvvm::{mbarrier_arrive, mbarrier_test_wait, mbarrier_try_wait, mbarrier_wait}` (not in the `prims::*` glob, whose names the crate root has); the waits take a parity (`Int32`) or a state token (`Int64`) |
| `mbarrier_arrive_drop`, `mbarrier_arrive_drop_expect_tx`, `mbarrier_arrive_drop_nocomplete`, `mbarrier_arrive_nocomplete`, `mbarrier_complete_tx`, `mbarrier_inval`, `mbarrier_try_wait_parity`, `mbarrier_try_wait_timelimit` | the same names in `prims` |
| `nanosleep`, `pmevent(event_id= / mask=)`, `breakpoint`, `ptx_add_comment`, `exit`, `memory_barrier`, `fence_sc_cluster` | `prims::nanosleep`, `pmevent(id)` / `pmevent_mask(mask)`, `breakpoint`, `ptx_add_comment`, `exit`, `memory_barrier(FenceScope)` (as `fence.sc`), `fence_sc_cluster` |
| `dot_accumulate_4way`, `dot_accumulate_2way`, `elect_sync(membermask=)`, `cvta_to`, `mov_b32`, `st_bulk`, `tcgen05_shift` | the same names in `prims` (`dp4a`/`dp2a` on packed `u32`s; `cvta_to` on pointers only; `tcgen05_shift` is not simulated) |

## Limitations

* Launch ABI and attributes: a by-value struct's fields are read at multiples of their
  size, so a packed layout's misaligned fields cannot be read; `smem_overlay` takes two
  alternatives of one shared buffer each; under `ClusterShapes::Mixed` the simulator runs the
  preferred clusters first and the fallback ones after, so clusters of different shapes never
  run together (a cooperative grid mixing them cannot be simulated); extern module globals
  and the `CUlibrary` API are not supported.
* Tensors in global memory take runtime extents and strides, but what a
  kernel unrolls when it is built must be static: a copy's per-thread chunk
  and its rest modes, gemm's modes, the elements `fill`, `values`,
  `store_values`, `map_in_place` and `autovec_copy` visit, and the shapes of
  register fragments (`make_fragment_like`, `to`,
  `partition_fragment_a/b/c`): each refuses a runtime extent, naming it. A
  runtime number of tiles takes a `for_range` over the rest mode with one
  tile per iteration; `basic_copy` and `basic_copy_if` of runtime extents
  are runtime loops, which a register fragment cannot take part in, and
  `TensorSsa` values and `load_ssa`/`store_ssa` have static shapes. A tensor parameter whose
  cosize is only known at launch is a buffer sized at launch, whose length
  launches do not check (the simulator reports the accesses past it).
  Index expressions are non-negative except for signed terms
  (`IndexExpr::signed`, negative constants), which only add and multiply:
  dividing, taking the remainder of, masking or shifting a value that may
  be negative is refused when the kernel is built. `upcast`, thread-value layouts, the tilers of tiled
  copies, the block a product replicates and the buffers sized when the
  kernel is built (shared memory, fragments, stack arrays) must be static. A
  layout's dynamic extents are taken to be large and divisible when it is
  composed or divided. Where a composition cannot place a right-hand
  runtime value, or two runtime values, when the kernel is built, it
  computes the extents at run time with `min` and `ceil_div`, as C++ CuTe
  does; like C++ CuTe's, the result is only right for values that satisfy
  the static algebra's divisibility conditions on the modes as cute-rs sees
  them (runtime strides keep modes from merging). A runtime stride on the
  right-hand side of a composition, or in the tiler of a divide or a
  product, must be non-zero at launch, as in C++ CuTe: those runtime extents
  divide by it, unchecked (the simulator reports the division by zero; a
  static stride of 0 is fine). Static composition refuses the cases that
  C++ CuTe's weak divisibility condition lets through, where a stride
  splits a mode it does not divide, since C++ CuTe's result is then often
  not the composition (dividing `(6,8):(8,1)` by 4, say). Runtime leaves are
  compared as expressions (up to the order of a product's factors), so
  `m * 2` and `m + m` are different leaves. Offsets are 32-bit unless a
  64-bit leaf (an `Int64` stride or extent, `param_extent_i64`) or a static
  value beyond `i32` enters them; then that offset is computed in 64 bits.
  The simulator holds whole buffers in memory, so it only runs tensors much
  smaller than 2^31 elements. Layout strings parse static integers, `?` and
  basis strides (`1@0`, `?@1`, `1/4@0`), and composed layouts and swizzles
  (`S<3,4,3> o 0 o (8,64):(64,1)`); negative strides do not parse.
* Swizzle units: a composed layout's swizzle acts on what its outer layout
  gives, element offsets, while a pointer's swizzle acts on byte addresses,
  as in CuTe DSL (whose Ampere GEMM notes that its `base_bits` are bytes,
  and whose Hopper GMMA atoms attach `Swizzle<B,4,3>` for every element
  type through the pointer). For `2^k`-byte elements `Swizzle<B,M,S>` on
  elements is `Swizzle<B,M+k,S>` on bytes (`Swizzle::to_bytes`,
  `to_elements`). cute-rs evaluates a pointer's swizzle on the offset from
  the start of its buffer, which equals CuTe's absolute-address form because
  the buffer is checked to be aligned to the swizzle's period; a swizzled
  composed layout on a pointer needs the pointer a whole number of periods
  into its buffer. Slicing a swizzled tensor keeps the swizzle (the slice's
  offset goes inside it) rather than decaying it, so existing kernels
  compile as before.
* `autovec_copy` and `max_common_layout` read a swizzled tensor's layout
  before its swizzle (CuTe DSL composes the swizzle into the layout); the
  swizzle bounds the vector through the pointer's alignment instead, and
  every vector is proved within one of its runs. `max_alignment` counts a
  runtime stride as 0 bounded by its known divisor, where C++ CuTe assumes
  runtime strides large multiples of anything (0) and CuTe DSL's rule is
  not public; the per-vector proofs are unchanged.
  The eviction priority of `autovec_copy_with` also applies when the copy
  falls back to single elements, where CuTe DSL's `basic_copy` drops it.
  32-byte vectors are only issued for plain accesses (inline PTX with
  attributes stops at 16 bytes), and `ptxas` from CUDA 12.8 (PTX ISA 8.7)
  cannot assemble their sm_100 form, so that form is checked as LLVM's PTX.
* Tensor-core MMAs are scalar FMA and SM80 `mma.sync`: f16 and bf16
  m16n8k16 and m16n8k8 (f32 accumulators, or f16 for f16 inputs), TF32
  m16n8k4 and m16n8k8, and f64 m8n8k4. TF32 operands are f32 registers whose
  low mantissa bits the tensor core ignores (round them first with
  `cvt_f32_tf32` or `Value::to_tf32`, `cvt.rna.tf32.f32`), or TFloat32
  values (`x.to::<TF32>()`), the same registers. FP8 `mma.sync`
  (`MmaFP8Op`, sm_89) and the f16/bf16 sparse `mma.sp` (`MmaF16BF16SparseOp`)
  are implemented; block-scaled and sub-byte `mma.sync` are not. SM100 `tcgen05` MMAs, TMEM copies and the Blackwell pipelines are implemented, simulated and assembled by `ptxas -arch=sm_100a`, but have not run on hardware (no Blackwell GPU is available; the `blackwell_fp16_gemm` example is the end-to-end check) (SM90
  WGMMA and TMA copies are: see their sections and the TMA limitations below). The FP8 MMAs' `nvvm.mma.sync` needs the `mlir-opt`
  of LLVM 23 and `nvvm.mma.sp.sync` that of LLVM 22: with an older one
  `compile_with` fails with `CompileError::Unsupported` (it reads `mlir-opt
  --version` once), unless `CompileOptions::mma_inline_ptx(true)` asks for
  inline PTX. Neither form is marked convergent (LLVM's `llvm.nvvm.mma.*`
  intrinsics are only `memory(none)`; the inline PTX has side effects, so
  LLVM does not sink or speculate it, but may duplicate it into both sides
  of a branch), so keep MMAs out of code LLVM could make divergent. The
  SM89 GPU test runs the FP8 MMAs through that inline PTX (the pod's LLVM is
  22); `tests/mlir.rs` assembles the `nvvm` form with LLVM 23. The sparse
  operand is a plain compressed fragment plus a Uint32 metadata fragment the
  kernel fills: CuTe DSL's sparse element types (m10) are not implemented,
  and only the TID metadata format (sparsity selector 0) exists. The
  simulator reports metadata that is not one of the six ordered index
  pairs. The 8-bit `ldmatrix` shapes (`.m16n16`, `.m8n16`) have run on no
  GPU (they need sm_100a or sm_120a): the simulator follows the PTX ISA's
  fragment figures, and the 4- and 6-bit unpacking places each value as
  the PTX ISA's decompression figures for the same `.b8x16.b4x16_p64` /
  `.b6x16_p32` qualifiers of `tcgen05.cp` show (4 bits in bits 2 to 5, 6
  bits in bits 0 to 5, the rest zero; the `ldmatrix` section does not say),
  which is how `mma.kind::f8f6f4` takes them. The signed `.m8n16.s8.s4`
  form (PTX 9.4) is not in CuTe DSL and is left out.
* Conversions to BFloat16 go through Float32 from every type but Float32
  (`cvt.bf16.f64` and the integer forms need sm_90, so the emitter routes
  them through Float32 on every target): from Float64, from the 64-bit
  integers and from 32-bit integers beyond 2^24 that is two roundings,
  which the simulator computes as emitted. Float64 to Float16 is one
  rounding (`cvt.rn.f16.f64`), in the simulator too; `tests/gpu.rs` holds
  the GPU to it on values where the two differ.
* The narrow types: FP8 values are storage types, converted with `to()`
  (which saturates, as `.satfinite` does) and never computed on; a
  conversion into Float8E4M3, Float8E4M3B11FNUZ or FloatNV8E5M3FNU is
  refused (PTX has no `cvt` into the first two; the UE5M3 `cvt` needs PTX
  9.4). FP8 from Float64 or wide integers converts through Float32, so it
  can round twice. The bf16x2-to-FP8 `cvt` (PTX 9.1) and the `.relu` form of
  the bf16x2-to-UE8M0 `cvt` (which ptxas 12.8 refuses) are left out; the
  UE8M0 `cvt`s are checked only by `ptxas` for sm_100a. Int4 values have no
  bit counts. Sub-byte tensors (Int4, FP4, FP6) are only read and written
  in whole vectors of a multiple of 32 bits: no scalar access, no atomics,
  and a copy whose layout breaks a 32-bit word into pieces is refused (CuTe
  DSL would fall back to element-wise access, which hardware does with
  byte read-modify-writes). Two threads storing different values of one
  word without ordering race, which the simulator reports in shared memory.
  `recast_tensor` repacks a register fragment only for widths in a whole
  ratio (not FP6 to bytes). The FP4 and FP6 `cvt`s (sm_100a, sm_101a,
  sm_120a) are checked by the simulator against the codecs, by `mlir-opt`
  and by `ptxas` for sm_100a and sm_120a; no GPU here runs them (FP4 and
  FP6 storage, packing and recasts run on SM89). Their f16x2/bf16x2 sources
  and bf16x2 results (newer PTX) and the x4 forms are left out. FP6 data
  in the one-value-per-byte layout tcgen05 and TMA use is a Uint8 tensor;
  the pair conversions take and give values, which `pack`/`unpack` put in
  bytes. Int128 and Uint128 have no printf conversion,
  inline-assembly register class, shuffle, atomic, packing or `mul_hi`, and
  no conversion to or from floats (LLVM has no NVPTX lowering; convert
  through Int64). Implicit promotion applies to two runtime values: a Rust
  literal takes the other operand's type, so `x_u8 + 1` is still a Uint8,
  unlike CuTe DSL; and a call whose operand type was inferred from the
  other operand (`a + b.bitcast()`, `a.lt(b.bitcast())`) may now need the
  type named (`b.bitcast::<u32>()`), since several right-hand types are
  possible.
* mbarriers work from SM80 (`mbarrier_init`, `mbarrier_arrive`,
  `mbarrier_test_wait`). The SM90 forms (arrival counts, `expect_tx`,
  relaxed and cluster-scope arrivals, peer-CTA arrivals through `mapa`,
  `try_wait`, `fence.mbarrier_init`) and bulk copies are implemented, along
  with block clusters in the simulator and the launcher, but no SM90 GPU
  has run them yet; their PTX assembles with `ptxas` for sm_90a and sm_100a.
  `mbarrier_wait` uses `try_wait`, so it needs SM90 as in CuTe DSL (spin on
  `mbarrier_test_wait` on SM80). Cluster barriers are missing.
  `block_idx_in_cluster`, `cluster_size` and `aggr_smem_size` are simulated
  and assemble for sm_90 and newer, but no SM90 GPU has run them.
  `Kernel::compile_with` and `compile_ptx` check a kernel against its
  target; `Toolchain::compile_ptx` compiles raw MLIR unchecked, at the
  default options. Bulk copies to global memory, bulk async-groups and
  `fence.proxy.async` are simulated and assemble for sm_90a (the port of
  CuTe DSL's `cp_async_bulk.py` among them), but no SM90 GPU has run them;
  only `fence.proxy.alias` runs on the SM89 GPU. The simulator checks the
  global bytes of a pending bulk store against plain loads and stores only
  (not `cp.async` or bulk-copy sources, atomics or other bulk stores), and
  models the async proxy's shared-memory reads, not its global ones
  (`fence.proxy.async.global` orders nothing in it). The `nvgpu` module
  has `cpasync`, the operand major modes and `warpgroup` (m8); the `warp`
  atoms are `CopyOp` variants.
* `cp_async_mbarrier_arrive_noinc` and `cp_async_mbarrier_arrive` run on
  SM80 and newer (tested on SM89). The arrivals they owe are made, and the
  copies land, when a thread waits on the mbarrier and they are all its
  phase still misses, whether the consumer's wait comes before or after
  the producer's copies in program order. The tracked copies land as
  writes of the block's asynchronous agent, whose one clock the bulk copies share, so a
  warp that waited for one of their phases is also ordered after what the
  agent wrote before it. A thread's covering `cp_async_wait_group` orders
  its own tracked copies before its warp, and before the warps that
  synchronize with it afterwards, and nothing else the agent wrote.
* The warp-specialised example ports (`warp_named_barrier`,
  `pipeline_software_fill`, `cp_async_mbarrier_arrive`,
  `cooperative_launch`) follow CuTe DSL's experimental `prims` front end
  through `cute::arch`. Their SM90 builds (`mbarrier_try_wait`,
  `fence.mbarrier_init`) and the 2-CTA cluster kernel have been simulated,
  compiled with `mlir-opt` for sm_90/sm_90a and assembled with `ptxas`, but
  not run on an SM90 GPU. The cluster kernel writes each CTA's own tile
  where the example reads its peer's through distributed shared memory
  (`mapa` loads came with m8-dsmem; the port keeps its own-tile form).
* The simulator reports an mbarrier that runs two phases ahead of a parity
  wait (phase aliasing) only under a schedule that lets it: when nothing
  orders a phase's completion after a waiter's wait for the phase before it
  (one mbarrier reused for several hand-offs, with no `empty` mbarrier
  holding the producer back), a GPU can complete both before the waiter
  polls, and the wait can spin forever, while the round-robin schedule runs
  the waiter in time and passes. Run-to-block and most seeds of
  `simulate_schedules` show it (`tests/schedules.rs`). Three hand-offs from
  a producer warp through one mbarrier deadlocked on all of 200 runs at O3
  on an SM89 GPU; `tests/feature_kernels`' `cp_async_mbarrier` gives each
  hand-off its own mbarrier, and `cp_async_pipeline` reuses its mbarriers
  behind `empty` ones.
* Shared memory sized at launch: the dynamic region starts at a multiple of
  its alignment within the dynamic window, whose PTX declaration asks for
  the largest alignment of its buffers. On an SM89 GPU the window starts at
  shared address 0 for every layout tested (up to `.align 4096`); what
  other GPUs do with large alignments is unchecked. In a kernel with a
  region, or with more than 48 KB of build-time buffers, those buffers are
  in the window too, in front of the region, and the launch passes them to
  the driver as dynamic shared memory: the raw `%dynamic_smem_size`
  (readable with `inline_ptx`) exceeds `LaunchConfig::dynamic_smem_bytes`
  by the region's start (`SharedMemoryLayout::dynamic_start`), which
  `dynamic_smem_size()` and `get_dyn_smem_size()` subtract, and
  `total_smem_size()` counts the whole window, with the alignment padding
  in front of the region. On an SM89 GPU `%total_smem_size` is a byte
  count (not a count of allocation units): static plus dynamic shared
  memory rounded up to 128 bytes, with static buffers placed in
  declaration order, which the simulator returns for every layout tested;
  but LLVM can shrink a static buffer it proves partly unused (one indexed
  only by constants), which the GPU then counts smaller than the simulator
  does. `aggr_smem_size()` is simulated as that plus a 1 KB system
  reservation, from the PTX ISA and the CUDA guide, unchecked on a GPU.
* Of CuTe DSL's compile options, the optimization level, line information
  and the target are implemented. Some are accepted and do nothing (the
  front-end selectors `EnablePYIR`, `FrontendNext` and
  `DisableCuteExtCompile`, `debug{launch-check}`, and a `--host-target`
  naming this host). A few are rejected for good: a `--host-target` for
  another host (cross-compile with `cargo build --target` instead),
  `--flatten-locs-output-json-path` and raw MLIR pass options. The rest are
  rejected until the parts that add them land (see
  [Compile options](#compile-options)).
  Cubins and SASS need a CUDA toolkit with `ptxas` (and `nvdisasm`), or on a
  GPU machine the driver's linker for the cubin; without them the driver JIT
  compiles the PTX and a requested cubin or SASS is reported as unavailable.
* Kernels have no recursion or unstructured control flow, and call only
  external device functions (`ExternFn`, `DeviceFunction`; see
  [External device functions](#external-device-functions-and-multicast-memory)):
  no callbacks into the kernel. Multi-GPU execution of the multimem
  instructions is untested (one GPU cannot make multicast memory).
  `InlinePtx`, `InlineAsm` and `inline_ptx` reach any instruction the
  DSL doesn't have, and `MlirOp` any MLIR operation that is one line
  without regions, takes scalar values and pointers and gives scalar
  results (no vectors or structs), though the simulator can only run them
  when given a Rust equivalent, and only `mlir-opt` checks a hand-written
  operation (when the kernel is compiled). The simulation of a statement
  with pointer operands sees 0 for them: a kernel whose results depend on
  memory read through one can be compiled but not simulated faithfully,
  and the simulator does not see memory written through one (nor check it
  for races). `InlinePtx` passes an integer constant as an immediate, as
  CuTe DSL does; a template that needs a constant in a register (PTX takes
  an immediate almost everywhere) can compute it at run time. A predicate
  guards one instruction (PTX's `@p`), and
  where it is false a write-only output is 0, where CuTe DSL leaves the
  register undefined. LLVM does not treat inline PTX as convergent and may
  duplicate it into both sides of a branch, so it should not hold warp- or
  CTA-collective instructions such as `barrier.cluster` (the DSL's own
  barriers use convergent NVVM intrinsics).
* A carry chain must stay in one block of straight-line code, as PTX
  requires: carries cross control flow only as values (`add_with_carry`,
  `sub_with_borrow`), and a 64-bit integer division or remainder by a
  runtime value, which LLVM lowers with a branch, must come before or after
  the chain (one by a constant, or a 32-bit one, may sit inside it). The
  validator's list of what breaks a chain is what LLVM 21 to 23 are known
  to lower with branches or calls, and `tests/mlir.rs` checks the carry
  kernels' PTX, but LLVM is not bound to it: a version that lowered another
  operation with a branch, or hoisted the identical first instruction of
  chains in both arms of an `if_else` above the branch, would leave a branch
  inside a chain unseen. A chain cannot switch between adding (adds and
  multiply-adds) and subtracting either: the PTX ISA describes one flag for
  carries and borrows, but the GPU test found that on an SM89 GPU (`ptxas`
  12.8) a borrow is the inverted carry, so `subc` after an `add_cc`
  subtracts one minus the carry and `addc` after a `sub_cc` adds one minus
  the borrow (`tests/gpu.rs` keeps a check of this, written as inline PTX).
  The validator rejects such a chain (CuTe DSL does not check), and the
  simulator, which follows the ISA for the chains it accepts, refuses one
  too. The simulator's carry flag is the extended-precision instructions'
  alone: inline PTX that uses the flag cannot continue a chain (the
  validator rejects a carry-in after it), and its own chains must be whole.
  CuTe DSL's `mad_cc` and `madc` are the `.lo` forms, and so are cute-rs's:
  there is no `mad.hi.cc` (use `mul_hi` and an `addc`).
* `bfe`, `bfi` and `shf` take the start, length and shift amount as an
  Int32 value or a constant; CuTe DSL also takes a Uint32, which needs
  `.bitcast()` here. `prmt` takes Int32 or Uint32 operands of one type and
  returns that type, where CuTe DSL converts each operand to Int32. CuTe
  DSL's `bfe` and `bfi` are the bare instruction, whose results on an SM89
  GPU differ from the PTX ISA's for the starts and lengths described above;
  cute-rs's follow the ISA, at the cost of up to four integer instructions
  when the start or length is a runtime value (none for constants). The
  workaround was found with `ptxas` 12.8 on SM89; other GPUs have not run
  it. `bfind` is computed from `clz` rather than with PTX's `bfind`, and
  returns a Uint32 for every operand type (CuTe DSL returns the same bits
  as an Int32 for a Uint32 operand); `log2_of_pow2_int` is computed from
  `ctz` rather than with CuTe DSL's `brev.b32` and `bfind.shiftamt.u32`,
  with the same results.
* `make_warp_uniform` emits no instruction: CuTe DSL lowers it to a hint
  for NVIDIA's compiler, which upstream LLVM has no equivalent for. The
  simulator checks that the value is warp-uniform.
* `Vector` and `TensorSsa` are scalarised: the IR has no vector type, so a
  vector operation is one scalar operation per element (MLIR's
  `vector.reduction`, `vector.outerproduct` and `vector.bitcast` become
  their element-wise expansions; the results are the same bits).
  `cutlass.Pointer`'s swizzled accesses (`load_swizzled`,
  `store_swizzled`) are not implemented; its integer conversions are.
* Raw pointers ([`make_ptr`](src/raw_ptr.rs)) address memory through
  the simulator's address map: global buffers at per-launch bases from
  `SIM_GLOBAL_BASE` (256-byte aligned, a gap after each), shared buffers at
  their start in the block's shared memory (the same offset in every
  block), per-thread buffers at their start in the thread's local memory,
  and generic shared and local addresses in windows of their own. An
  address in no buffer is reported as a wild address; one past a buffer's
  end (within its gap) is out of bounds for that buffer. The addresses
  differ from the GPU's, so kernels should not write them to outputs.
  Boolean raw pointers are refused (point at `Uint8` flags), and so are
  constant-memory and distributed-shared-memory ones (m8).
* The simulator marks the masked-off results of a masked load without
  `pass_thru` undefined and reports their first use by anything other than
  a pure operation or a select that picks another operand; a value an
  `if_else_yield` or a loop yields out of its region loses the mark (it
  reads as 0), and a loop's initial carried values count as a use. NVPTX
  guards each masked lane with a branch rather than a predicated `ld`/`st`
  (as it does for `llvm.masked.load`).
* Shuffles move scalars and 32-bit `TensorSsa`/`Vector` values (as one
  word); other vector widths are refused. Float16 and BFloat16 values are shuffled as
  their bits, so a NaN keeps its payload where CuTe DSL's conversion
  through Float32 returns the canonical NaN; every other value is the same.
  With independent thread scheduling, two shuffles of the same kind and
  mask in the two branches of an `if` meet, as on Volta and newer GPUs; a
  lane of the mask that takes a path without one is a deadlock that names
  it. `vote_*_sync` and
  `elect_one` always use the whole warp's mask (CuTe DSL's votes take a
  `mask=`); votes are checked like the other collectives, while
  `elect_one` picks the lowest active lane.
* Float32 `redux.sync` (sm_100a, or the sm_100f family from PTX 8.8) has
  been simulated, compiled with `mlir-opt` and assembled with `ptxas` 12.8
  for sm_100a, but no GPU that has it has run it. Its member mask is a
  register (CuTe DSL's inline PTX takes an immediate). It compiles for
  sm_100a and for sm_100f and the `a` and `f` targets of sm_103 and sm_107.
* Ordered and cache-hinted loads and stores are for global and shared
  memory: per-thread (local) memory takes only plain accesses
  (`ld.volatile.local` needs PTX 9.1), Booleans take no attributes, and
  another CTA's shared memory (`ss=SharedSpace::Cluster`, `shared_space=
  SharedSpace.CLUSTER`) is reached through `map_dsmem_ptr` (m8-dsmem). The newer PTX forms are
  missing: `.L2::evict_*` priorities with 256-bit `.v8.b32` and `.v4.b64`
  accesses (sm_100, PTX 8.8), `.b128` accesses (PTX 8.3), `.mmio` with
  acquire or release (PTX 9.3), `.unified` and `.proxy::readonly`. A strong
  vector of eight or sixteen 8- or 16-bit elements is one instruction per
  four elements, as PTX has no single one for it (so a volatile one is more
  than one volatile instruction). `load`'s `sem` also takes
  `MemoryOrder::Sc`, `Volatile`, `Mmio` and `Constant`, which CuTe DSL's
  `cute.arch.load` does not, and its `scope: None` means `.sys`.
  CuTe DSL's lowering of `MemoryOrder.SC` and `CONSTANT` is not public;
  cute-rs emits `fence.sc` with an acquire load or release store, and
  `ld.global.nc`. Nor is that of `LoadCacheMode.NONE` and
  `StoreCacheMode.NONE`, which cute-rs takes to mean no cache operator,
  like the defaults `ALWAYS` and `WRITE_BACK` (PTX's `.ca` and `.wb`), so
  `.cv` is reachable through `load`'s `cop` but not through a G2R atom.
  G2R and R2G atoms copy between global memory and registers (fragments,
  or `make_rmem_tensor`s, whose per-thread accesses stay plain), at most
  128 bits at a time; CuTe DSL's `CopyS2ROp` and `CopyR2SOp` are not
  implemented. The tensormap form of `prefetch` (`tensormap=`,
  `in_param_space=`) waits for milestone m7.
* The vector forms of the atomics and reductions (`.v2`, `.v4`, `.v8` of
  f32, f16 and bf16, sm_90) wait for TensorSSA vectors (m4); the
  `.L2::cache_hint`, `inc`, `dec` and 128-bit forms are not implemented.
  Shared-memory atomics are emitted as `.shared` (which is `.shared::cta`)
  so that sm_80's PTX 7.0 takes them; `.shared::cluster` atomics on another
  CTA's memory are not implemented. The simulator ignores atomics' scope.
  `atomic_fmax` and `atomic_fmin` are wrong for NaNs with the sign bit set,
  as CuTe DSL's are, and silently wrong when `sign_bit` is given and does
  not match the value.
* Flags order accesses in the simulator between the warps of a block,
  which is where its race checker looks: global memory is not
  race-checked, and flags between CTAs work but order nothing for the
  checker. Its schedules run a producer that comes first in program
  order before a consumer reads it, so an
  acquire that reads a release orders the consumer's later accesses even
  if the kernel never checks the flag's value. `fence.sc` acts as a
  `fence.acq_rel` (the order among SC fences is not modelled), and two
  patterns are not recognised, which can only add race reports: an
  acquire pattern `ld.relaxed [M]; ld.acquire [M]` whose acquire reads a
  later value than its relaxed load, and a release pattern `st.release
  [M]; st.relaxed [M]` with another warp's store to `M` in between. Nor
  are strong reads of one byte by one warp at different sizes between two
  writes of it, which count as reads of a size that no write matches. The
  non-coherent load check is per byte, over the whole launch.
* `make_cotiled_copy` needs layouts with static strides. It checks that
  every thread's values are at addresses the data holds, which is stricter
  than CuTe: CuTe checks only the value mode (the offsets of one thread's
  values), and accepts a thread mode that reaches beyond the data, whose
  threads its composition then wraps around onto other threads' elements.
* The packed half-precision operations and the conversions take scalars,
  arrays and words; CuTe DSL's take and return MLIR vectors, which come
  with TensorSSA (m4), and its int4 helpers sub-byte tensors. `.oob`
  (fma's out-of-bounds modifier) gives `+0` where a multiplicand is the
  "OOB NaN" that TMA's NaN fill writes, a special NaN whose encoding
  neither the PTX ISA nor the CUDA documentation gives: the simulator takes
  every NaN multiplicand for it, as CUTLASS's host reference does
  (`guarded_multiply_add`), where a GPU may return NaN for other NaNs. No
  available GPU runs `.oob`, the bf16x2 `add` and `sub`,
  `ex2.approx.ftz.bf16x2` or `cvt.rn.bf16.s8` (sm_90): they are simulated
  and assembled with `ptxas` 12.8 for sm_90 and newer. The s2f6x2
  int8-to-BFloat16 conversions need PTX 9.1 (CUDA 13.1), which only LLVM 23
  emits and `ptxas` 12.8 cannot assemble, so they are simulated and checked
  as MLIR (and as PTX text with LLVM 23; LLVM 21 and 22 are refused, see
  Compiling above); `cvt_i8_bf16_intrinsic` uses them only when
  asked (`I8ToBf16Path::S2f6x2`), where CuTe DSL picks them for sm_100a-class
  targets, since a kernel's target is chosen after it is built (m6).
  `exp2_packed_bf16x2` is held to nothing but the simulator's correctly
  rounded value, as no GPU here runs it (the PTX ISA's bound is 2^-7
  relative).
* Loop hints become LLVM loop metadata. `LoopOptions::vectorize` is
  accepted from sm_100 on, as in CuTe DSL, but LLVM (21 to 23) does not
  vectorise loops for NVPTX, so the PTX keeps scalar `add.rn.f32`; CuTe
  DSL leaves the packed f32x2 code to NVIDIA's compiler, and cute-rs has no
  packed f32x2 arithmetic yet. CuTe DSL's `at_least_once` attribute is
  only read by NVIDIA's compiler; cute-rs emits the promise as an
  `llvm.intr.assume` of the bounds instead, which the simulator checks (as
  the validator does for constant bounds). `prefetch_stages` (automatic
  software pipelining) is not implemented.
* A `while_carry` condition can read `Var`s but not set the ones created
  outside it (in CuTe DSL the condition is an expression). `Var::get`
  returns a copy: a register tensor read from a `Var` and changed is stored
  in it only when set back (`Var::update` does both).
* Inside a runtime loop, a `Var` is a loop-carried value even if the loop
  only reads it, so what the builder knows about its value (a constant, a
  divisor or a bound, which index arithmetic and alignment proofs use) does
  not hold there: CuTe DSL's preprocessor sees in advance which variables a
  loop assigns, and the builder cannot. Read the `Var` before the loop and
  use that value inside to keep it; the errors that the loss causes (an
  alignment a copy cannot prove, a register index that is not static) name
  the `Var` and say so. After an `if` or a loop, what holds for every value
  the result can take is kept.
* The build-time constant fields of a struct given `Carry` by
  `carry_struct!`'s second form are part of its type, as a dataclass's
  static fields are in CuTe DSL, but changing them is not detected: a
  loop's or branch's result keeps the constants it came in with, and a
  `Var` those of its initial value.
* Register fragments are build-time arrays of SSA values, not memory as in
  CuTe DSL, so a runtime loop does not carry them. A loop like
  `for_range(0, n, 1, |i| f.store(0, f.load(0) + i))` on a fragment `f`
  from outside the loop would add each `i` to the value from before the
  loop, so it is reported when the kernel is built; keep `f` in a `Var`
  (`acc.update(|f| ...)`) or carry it with `for_range_carry`. A predicated
  copy into such a fragment is not reported: the registers it skips keep
  the value from before the loop rather than the previous iteration's,
  which differs from CuTe DSL only when the predicate changes between
  iterations and the kernel uses the skipped registers (reporting it would
  reject the usual pattern of computing on every register and storing
  under the same predicate).
* The hardware approximations (`Value::exp2`, `log2`, `sin`, `cos`, `rsqrt`
  and `rcp`, and `cute::math`'s fast mode and `approx=`) are not bit for
  bit: the simulator computes the function they approximate, and
  `tests/gpu.rs` holds the GPU to the PTX ISA's error bounds instead. The
  PTX ISA documents no bound for `rcp.approx.ftz.f64` and
  `rsqrt.approx[.ftz].f64`; the simulator follows the ISA's description of
  them and the GPU test allows 2^-19 relative.
* `cute::math` functions take scalars; their TensorSSA forms are
  `TensorSsa` methods of the same names (`t.exp2(true)`), or
  `map_in_place` on a register tensor.
  Precise mode does not give libdevice's bits, only the documented
  accuracy (`erfc` is 2.5 ulp, the others at most 1.5). Expansions are
  long (`sin` and `pow` are a few hundred instructions, and huge
  trigonometric arguments take a divergent Payne-Hanek branch), and
  Float32 `cbrt` uses Float64 arithmetic, as msun's does. `fastmath=True`
  on `add`, `sub`, `mul` and `fma` changes nothing (CuTe DSL's compiler may
  contract and reorder them).
* The simulator returns the NaNs an SM89 GPU returns where the PTX ISA
  leaves them unspecified (the canonical NaN in single and half precision;
  in double precision the first NaN operand, quieted, or
  `0xfff8000000000000`), but a double-precision instruction with two NaN
  operands may return either on the GPU. LLVM also removes arithmetic that
  cannot change a number, such as `x * 1.0`, which on the GPU then passes a
  NaN argument through unchanged where the simulator returns the
  instruction's NaN, so a NaN's bits can differ where LLVM simplifies
  arithmetic on it. `abs`, `neg` and `copysign` are bit operations LLVM
  cannot change, and `cute::math`'s expansions return one canonical NaN, so
  neither is affected.
* The simulator tracks hazards at warp granularity, and in shared memory
  only. Accesses within a warp, across its convergence groups too, are
  treated as synchronous. Its schedules are deterministic (a seeded one for
  its seed): the race checker's vector clocks do not depend on the order,
  but results that do (a racy kernel's, or the order of `printf` output
  across warps) are those of the interleaving run (`simulate_all_orders`
  and `simulate_schedules` compare them). Independent thread scheduling is
  modelled at the granularity of structured control flow: groups split at
  a divergent `if_` and rejoin at its end, so lanes never reconverge
  earlier (or later) than that, and a loop's early leavers wait at its
  end. Every thread of a warp must reach an aligned named barrier
  together; the unaligned `barrier.arrive` may be reached by a warp's lanes
  at different instructions. `mbarrier_try_wait` fails spuriously only
  under a seeded schedule (otherwise it returns false only when no warp can
  run). Mbarrier waits a schedule hides are reported when they could pass
  early, and phase aliasing when the schedule runs the producer ahead. The
  bulk copies of a block share one clock, so waiting for one copy also
  orders a reader after the copies that landed before it.
* The `cuda` launcher has only been run on one GPU, an RTX 2000 Ada (SM89).
  PTX for other architectures assembles with `ptxas` but has not been run.
* What m6 could only check with the simulator and `ptxas`, since the test GPU is an SM89:
  the overlap of programmatic dependent launches (the driver accepts `use_pdl` on SM89 and
  runs the kernels in order; the `SimStream` model checks the accesses before
  `griddepcontrol_wait`), programmatic events triggered by `griddepcontrol_launch_dependents`
  (on SM89 the event is recorded as the blocks exit), preferred clusters (SM100; the
  simulator runs preferred, fallback and mixed shapes, and the launcher refuses them before
  SM100), grid-constant by-value parameters (`nvvm.grid_constant`: emitted, assembled and
  simulated, not launched on a GPU), and the
  sm_100f, sm_103, sm_107, sm_110, sm_120f and sm_121 targets (compiled and, where the
  toolchain has them, assembled; sm_107 is emitted by no LLVM yet).
* Pipelines (m9): CuTe DSL's pipelines need sm_90 (`mbarrier.try_wait`, `fence.mbarrier_init`).
  A kernel built for a target below sm_90 (`KernelOptions::target(Arch::Sm89)`, or
  `CUTE_RS_ARCH`) instead gets `MbarrierArray` waits that spin on `mbarrier.test_wait`
  (sm_80), `try_wait` probes with it, and `pipeline_init_arrive` leaves the init fence out (the
  CTA barrier of `pipeline_init_wait` publishes CTA-local mbarriers), so the mbarrier-only
  pipelines run on SM80 to SM89. Without a target, they emit what CuTe DSL emits, and compiling
  for sm_89 fails at the line that waits. The `TCGen05Mma` arrival waits for m10's
  `tcgen05.commit`; the cluster, peer-CTA and TMA forms are checked with the simulator and
  `ptxas` only.
* TMA (m7): every TMA operation (descriptors built on the host and hidden per atom, tile
  loads, multicast loads, stores and reduce stores, bulk copies, prefetches, in-kernel
  descriptor copies and `tensormap.replace` updates, the tensormap-proxy fences, and the
  ported tutorials) is verified only by the simulator (against plain Rust), by `mlir-opt`
  (LLVM 21, 22 and 23) and by `ptxas` 12.8 for sm_90/sm_90a: no SM90 GPU has run any of
  it, and the SM89 GPU's driver refuses to encode tensor maps. The simulator models TMA from
  the PTX ISA with its own descriptor encoding, so it cannot catch a difference between that
  and the driver's opaque format (`tensormap.replace` field units, say). Only the tile mode
  without interleave is simulated. The im2col, gather4/scatter4, byte-mask and FP4/FP6
  forms (m7-tma-im2col, m7-tma-sm100-modes) are simulated and assembled by `ptxas` (sm_90/
  sm_90a and sm_100a); the sm_107f override copies are simulated and checked as MLIR text
  only (no toolchain has sm_107f). Refused: CTA-pair loads (`.cta_group`; m10-cta-pair),
  `.im2col::w`/`.im2col::w::128`, the `s128b_atom_*` swizzles (m10-umma-layouts), the
  multicast split of a gather4 box, the FP6 store form `.b6p2x16`, and the sm_107f-only
  `.im2col_no_offs::w` and `.report_mechanism`. `tensormap.replace`
  needs an architecture-specific target (sm_90a, sm_100a, sm_110a, sm_120a or their
  families); its fields other than the address, extents and strides are not exposed.
* SM90 (m8): WGMMA (`warpgroup`), `stmatrix`, `setmaxnreg`, distributed shared memory and,
  with m7, TMA are verified only by the simulator (whose models come from the PTX ISA,
  independently of cute-rs's layout code, and are held to plain Rust) and by `ptxas` 12.8
  for sm_90/sm_90a, `examples/hopper_gemm.rs` among them: no SM90 GPU has run any of it
  (the SM89 GPU runs only the host-side descriptor encodings, `cp.async` paths and
  mbarriers). `wgmma.mma_async` is inline assembly with every LLVM version (LLVM has no
  intrinsic for it), its accumulators tied and fenced so LLVM cannot move their reads and
  writes across it; on LLVM 21 `stmatrix` is inline assembly too, which LLVM does not treat
  as convergent (LLVM 22 and 23 lower `nvvm.stmatrix` to the convergent intrinsic). LLVM 21
  also drops the offset of a buffer in the dynamic shared-memory window when it takes its
  address, so kernels above 48 KB whose buffers sit at nonzero offsets, such as
  `hopper_gemm`'s TMA destinations and WGMMA descriptors, need LLVM 22 or newer. The
  simulator adds a WGMMA's products in order of K in Float32; the hardware's order is
  undocumented, so only exact sums are known to match it.
* TVM-FFI callables take positional DLPack tensors and scalars only (no TVM-FFI `Array`,
  `Map` or dataclass objects, defaults or keyword arguments), check their arguments on the
  host rather than in compiled code, and build one callable (and load one module) per
  calling thread. They have been called from C and from Python (`tvm_ffi` 0.1.14 with
  torch) on the SM89 GPU. `benchmark` times with CUDA events only (no CUPTI or NVTX).
* Tensor memory (m10-tmem-core) is modelled per CTA: each simulated CTA has its own 128 x
  576 cells, so co-resident CTAs never share an SM's TMEM (nor contend for its columns or
  the allocation permit). An allocation that can never be served is reported as a deadlock.
  The sm_107 exclusive forms are inline PTX checked only as MLIR text and in the simulator
  (no installed LLVM or ptxas knows PTX 9.4); their qualifier order is unverified.
* Blackwell (m10): every `tcgen05` instruction, TMEM operation, UMMA MMA, `tcgen05.cp`, CLC
  and CTA-pair form is verified by the simulator (whose models come from the PTX ISA and are held
  to plain Rust) and, for sm_100a and sm_120a, by `ptxas` 12.8; **none has run on a Blackwell
  GPU** (the only GPU here is SM89), so timing, throughput, the hardware's accumulation order and
  any instruction-level behaviour the PTX ISA does not state are unchecked. The toolchain
  ceilings: ptxas 12.8 stops at PTX 8.7 and LLVM 21 at PTX 8.8 (no sm_110), LLVM 22 knows sm_110
  and PTX 9.0 and LLVM 23 PTX 9.3, and none knows sm_107 (PTX 9.4). So everything of the sm_107
  family (the Rubin MMAs and collector buffers, `tcgen05.ld.red.spcompress` and
  `.spcompress`, the exclusive TMEM allocation, the sm_107 TMA override copies, the Rubin packed
  math and UE5M3 conversions), and of PTX 8.8 or later (`tcgen05.ld.red`, the sm_103a K = 96
  MMAs, sm_110a forms) are simulated and checked as MLIR text or LLVM 23 PTX only, never
  assembled. Not modelled: the block-scaled sparse MMAs, TF32/I8/FP4 sparse MMAs, M = 64 sparse,
  `.ashift`, the Rubin `SM107MmaF16BF16SparseOp`/`SM107MmaFP8SparseOp` (listed in CuTe DSL's
  `__all__` but undefined in 4.8.1), and `sm107_tmem_sf_frg`.

## Layout of the crate

```text
src/
  layout/      IntTuple trees, Layout and the algebra, IndexExpr, swizzles,
               closure-inner composed layouts, size_in_bytes
  viz.rs       LaTeX TikZ drawings of layouts and TV layouts (cute.viz)
  print.rs     printf with {} placeholders: PrintArg, the format resolution
  diagnostics.rs  cute::Error, SourceLoc and code frames, Diagnostic,
               ResourceUsage and the ptxas/JIT log parser
  testing.rs   device assertions: assert_, RuntimeAssertion; testing/:
               benchmark, get_workspace_count, tune and Autotuner
  tvm_ffi.rs   TVM-FFI callables: the c_api.h ABI, TvmFfiFunction and
               tvm_ffi_export!
  value.rs     Value<T>, numeric types, operators
  numeric.rs   TFloat32, the FP8 and scale-factor formats and Int4: host
               types and the exact narrow-float codecs
  promote.rs   CuTe DSL's implicit promotion table for mixed-type operators
  convert.rs   Value::to between every pair of types (Boolean, TFloat32
               and FP8 conversions expanded to cvt and integer operations)
  control.rs   if_/for_range/while_carry, loop hints, Var, the Carry trait
               and carry_struct!
  arch.rs      thread/block indices, barriers, mbarriers, cp.async groups,
               warp collectives (shuffles, reductions, redux.sync, match),
               integer and bit instructions, fmax and fmin, carry chains,
               shared-memory size registers, cluster rank,
               cp.async.mbarrier.arrive, ordered and cache-hinted loads and
               stores, prefetch, bulk copies and bulk async-groups, proxy
               fences
  arch/inline.rs  hand-written PTX (named operands), inline assembly and
               MLIR, and custom operations
  arch/        packed and scalar half-precision operations (half.rs), narrow
               conversions (cvt.rs)
  pipeline.rs  cutlass.pipeline's NamedBarrier; pipeline/: pipeline states,
               cooperative groups, MbarrierArray, SyncObject, init sync,
               the Pipeline trait, PipelineAsync, its participants and
               PipelineEmpty; PipelineCpAsync and PipelineOrder
  memory.rs    cutlass.memory: SmemPartition, SmemAllocator, partition
               sizes and shared-memory capacities
  tensor.rs    Tensor, slicing, local_tile/partition, fragments, the dynamic
               shared-memory region
  tensor_ssa/  TensorSsa (CuTe DSL's TensorSSA): load_ssa/store_ssa,
               operators, broadcasting, indexing, reductions, conversions,
               bitcast and print_tensor
  vector.rs    Vector (cutlass.Vector) and cute::vector (cutlass.vector),
               TensorSsa interop
  masked.rs    masked loads and stores, gather/scatter, the cutlass.Pointer
               methods on Ptr
  raw_ptr.rs   make_ptr, inttoptr, toint, tospace, align and masking of
               pointers
  elementwise.rs  the flat element arrays, broadcast maps and folds under it
  fast_divmod.rs  FastDivmod (CUTLASS's fast_math.h)
  params/      the kernel argument protocol (KernelParam, ArgPack),
               TensorSpec and HostTensor with mark_layout_dynamic and
               mark_compact_shape_dynamic, SymInt, layout parameters,
               typed entry points (Kernel::jit), kernel_params! and native
               structs
  math/        cute.math: msun-derived expansions, fast and instruction forms
  atom/        copy atoms and TiledCopy, MMA atoms and TiledMma
  nvgpu/       cute.nvgpu's operations by submodule (cpasync: CopyBulkS2GOp,
               group_bulk_copy_modes; the operand major modes; warpgroup's
               SM90 shared-memory layout atoms)
  utils/       cutlass.utils: LayoutEnum, the sm90 shared-memory layout
               helpers (make_smem_layout_a/b/epi) and the static persistent
               tile schedulers; the grouped GEMM scheduler and the
               contiguous group search
  smem_struct.rs  cute.struct and cute.union: shared-storage structs
               (StructLayout, cute_struct!, cute_union!, MemRange views)
  builder.rs   Kernel construction (thread-local builder, CSE, DCE)
  ir.rs        the IR
  validate.rs  IR verification
  sim/         the CPU simulator, and SimStream with its PDL model
  backend/     MLIR emission, compile options, the lowering pipeline and
               PTX compilation; cache.rs: the in-memory and on-disk
               compile caches and KernelCache
  external/    PointerSpec, DLPack (the dlpack.h structs, DlpackTensor),
               from_dlpack, ExternalTensor and DlTensor
  cuda.rs      CUDA driver launcher (feature "cuda"); cuda/event.rs:
               CudaEvent; cuda/module.rs: shared loaded modules and unload
               callbacks
  aot.rs       ahead-of-time export to C and the export manifest
examples/      the kernels above
tests/         integration tests
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

The precise `cute::math` expansions in `src/math/` are ported from FreeBSD's
msun (Sun's fdlibm). Each of those files keeps Sun Microsystems' permissive
copyright notice, which applies to that code.

Parts of cute-rs are ported from NVIDIA CUTLASS's CuTe C++ library, the
layout code of its CuTe compiler, its `fast_math.h` and its BSD-3-Clause example programs (see
[Third-party code](#third-party-code)). NVIDIA's copyright notices and
BSD-3-Clause license, which apply to that code, are in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

cute-rs is an independent project. It is not affiliated with or endorsed by
NVIDIA.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

## Third-party code

Parts of cute-rs port algorithms and tables from NVIDIA's CuTe C++ library
(`include/cute` in CUTLASS 4.8.1), from `include/cutlass/fast_math.h`, from the layout code of its CuTe compiler
(`cutlass_compiler/cutegen`) and from its example programs: the CuTe tutorial
`examples/cute/tutorial/sgemm_2.cu` and the CuTe DSL examples in
`examples/python/CuTeDSL`, which `examples/sgemm.rs`,
`examples/elementwise_add.rs`, `examples/ampere_sgemm.rs`,
`examples/tensorop_gemm.rs` and the other examples port or follow.
All of these are under the BSD-3-Clause license.
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) has NVIDIA's copyright
notices and the license text, and lists the files and functions each port
comes from; each file with ported code also says so at its top.

NVIDIA's Python CuTe DSL package, `python/CuTeDSL` in CUTLASS, is under
NVIDIA's EULA rather than BSD-3-Clause. cute-rs took only API names and
semantics from it: no code, docstrings or message text.

## External device functions and multicast memory

(Roadmap lane m11-extern-distributed.)

**External calls** (`cute::ffi`, CuTe DSL's `cute.extern` / `cute.ffi`): an
`ExternFn` declares a device function by its typed signature (values, and
pointers into generic, global or shared memory); calling it in a kernel adds a
`func.func private` prototype (one per symbol: another signature is an error)
and a `func.call`. Its code comes from LLVM bitcode (`ExternFn::source(BitCode)`,
or `CompileOptions::link_libraries`, CuTe DSL's `cute.LinkLibraries`), linked
by the lowering pipeline's `nvvm-attach-target` (`l=`) before optimization, or
from another PTX module linked at load time (`CudaContext::load_linked`).
Overloads (`ExternOverloads`), generic parameters and compile-time integer
parameters mangle the symbol as CuTe DSL's `default_name_mangler` does
(`cute::ffi::mangle`, MLIR type spellings such as `f32` and `!llvm.ptr<1>`).

* Calls are **inlined by default** (CuTe DSL's `inline=True`);
  `inline(false)` keeps a PTX `call` (`func.call ... {no_inline}`).
* Every prototype is **`convergent` by default** (`convergent(false)` opts
  out): the callee may hold barriers or warp collectives, so LLVM must not
  duplicate a call or sink it into divergent code.
* The simulator runs a call by a DSL body traced at the call
  (`ExternFn::sim_body`, which can read and write through pointer arguments;
  a `DeviceFunction`'s calls use its own body) or a raw-bits closure
  (`ExternFn::sim`); without either, simulation fails at the call.

**Standalone device functions** (`DeviceFunction`, CuTe DSL's
`cute.compile[cute.DeviceTarget]`): built from a closure with value and
pointer parameters and at most one result, compiled to the PTX of an
`extern "C" __device__` function (`.visible .func`), and to relocatable
device code with `ptxas -c` (`compile_relocatable`, CuTe DSL's `cute.RDC`).
They allocate no shared memory and assume no block size; barriers are allowed
but need convergent callers.

**Multicast memory** (`cute::utils::distributed`, CuTe DSL's
`cutlass.utils.distributed`): `param_tensor_multicast` and
`param_ptr_multicast` flag a parameter's buffer as a multimem address, which
only the multimem instructions may access: `multimem_ld_reduce` and its
fifteen typed variants, `multimem_st`, and `multimem_red_add1` (sm_90, PTX
8.1; `.acc::f32` PTX 8.2; FP8 sm_100a-class targets). The flag locks
(`red_add1`, `spin_lock_atom_cas_{relaxed,acquire}_wait`,
`spin_lock_ld_lt_relaxed_wait`) need co-resident blocks (a cooperative launch).
The simulator gives a multicast parameter one replica per simulated GPU
(`MulticastReplicas`, `Kernel::simulate_multicast`), adding replicas in rank
order (the hardware's order is unspecified); `cuda::MulticastGroup` makes
multicast memory on GPUs that support it. `examples/all_reduce_two_shot_multimem.rs`
ports CuTe DSL's two-shot all-reduce and runs one rank's grid in the simulator.
Kernels calling functions linked at load time, and device functions, cannot be
exported with `export_c` (their PTX alone does not load); one symbol must not be
called both inlined and not inlined in a kernel.

### Blackwell pipelines and the TMEM allocator (m10-umma-pipelines)

| CuTe DSL | cute-rs |
| --- | --- |
| `cutlass.pipeline.PipelineTmaUmma.create(*, num_stages, producer_group, consumer_group, tx_count, barrier_storage=None, cta_layout_vmnk=None, mcast_mode_mn=(1,1), enable_multicast_signaling=False, defer_sync=False, name='')`; `producer_acquire(state, token, *, expected_tx=None)` | `pipeline::PipelineTmaUmma::create(n, producer, consumer, tx, PipelineTmaUmmaOptions { .. })`; the `Pipeline` trait (`make_participants`, the handles), `PipelineBarriers`; `expected_tx` is the acquire option (`acquire_with(token, Some(tx))`). The leader CTA arms the full barrier, releases are elected `tcgen05.commit`s |
| `PipelineAsyncUmma.create(...)`, `PipelineUmmaAsync.create(...)` (`producer_tail` on the even CTA) | `pipeline::PipelineAsyncUmma::create` and `PipelineUmmaAsync::create` with `PipelineUmmaOptions`. The 2-CTA masks and `cta_group::2` are computed from the `V` mode with CuTe DSL's formulas (leader from `block_idx.x % V`, the others from the cluster rank), checked on `(2, 2, 1)` and `(4, 1, 1)` clusters and simulated in `tests/cta_pair_pipelines.rs` (m10-cta-pair-pipelines). Without pairs (`V = 1`) a CTA is its own peer. `PipelineUmmaAsync::producer_tail` advances the state on every CTA (CuTe DSL only inside its leader branch) |
| `PipelineTmaMultiConsumersAsync.create(*, num_stages, producer_group, consumer_group_umma, consumer_group_async, tx_count, barrier_storage=None, cta_layout_vmnk=None, mcast_mode_mn=(1,1), tidx=None, enable_multicast_signaling=False, defer_sync=False, force_deprecated_per_lane_signaling=None, name='')`; `consumer_release(state, op_type=PipelineOp.TCGen05Mma)`; `make_participants()` -> `(producer, umma_consumer, None)`; `_init_empty_barrier_arrive_signal_2sm` | `pipeline::PipelineTmaMultiConsumersAsync::create(n, producer, umma, async_, tx, PipelineTmaMultiConsumersAsyncOptions { .. })` (m10-cta-pair-pipelines); `consumer_release_op(&state, PipelineOp::..)` and `ConsumerHandle::release_op(op)` (the `Pipeline` trait's `consumer_release` is the MMA's); `make_participants` returns the first two; `init_empty_barrier_arrive_signal_2sm`. One empty-barrier array counts both groups, viewed as `TCGen05Mma` and `AsyncThread` (`sync_object_empty_umma`/`_async`). The deprecated per-lane signalling is `per_lane_signaling: Option<bool>` (`None` auto-detects it as CuTe DSL does and logs a warning instead of a `DeprecationWarning`; refused in a cluster). Async signalling lanes need a `sync_warp` after the warp's reads |
| `cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A(cluster_shape_mnk, atom_thr_id)`, `_B`, `_SFB` | `utils::sm100::cluster_shape_to_tma_atom_a/b/sfb(cluster_shape, &thr_id) -> Result<Sm100TmaLoadOp, String>` (m10-cta-pair-pipelines); `Sm100TmaLoadOp::{Tile, Multicast}` converts into a `CopyOp`. Errors (CuTe DSL's `ValueError`s) for dynamic shapes, an M extent that is not a multiple of the SM count, and SM counts other than 1 and 2 |
| `tiled_mma.thr_id` | `TiledMma::thr_id()` and `MmaAtom::thr_id()` (`1:0`, or `2:1` for a CTA pair) |
| `cutlass.memory.TmemAllocator(...)` (and the deprecated `cutlass.utils.TmemAllocator`): `check_valid_num_columns`, `allocate`, `wait_for_alloc`, `retrieve_ptr(dtype)`, `reserve`, `relinquish_alloc_permit`, `free(ptr, num_columns=0)` | `memory::TmemAllocator::new(named_barrier, TmemAllocatorOptions { .. })` with an `Arch` for the target string; `retrieve_ptr::<T>()`; the column count is a build-time number (`allocate` and `free` take `&mut self`); not `Carry` |
| `TmemBufferPool`: `base_ptr`, `total_cols`, `current_offset`, `remaining_cols`, `allocate(size, dtype)`, `allocate_tensor(layout, dtype)` | `memory::TmemBufferPool`, `allocate::<T>(cols or &Layout)`, `allocate_tensor::<T>(&layout)` |
| `compute_tmem_cols_from_layout(layout, dtype)`, `get_num_tmem_alloc_cols(tensors, rounding=True, *, arch)` (and the deprecated `cutlass.utils` one) | `memory::compute_tmem_cols_from_layout(&layout, DType)`, `memory::get_num_tmem_alloc_cols(&[&t1, &t2], rounding, arch)` (`utils::get_num_tmem_alloc_cols` for sm_100) |
| `tcgen05.is_tmem_load(atom)`, `is_tmem_store`, `get_tmem_copy_properties(atom)`, `find_tmem_tensor_col_offset(t)` | the same names in `tcgen05` (`TmemPacking` for the Pack/Unpack result), and `CopyOp::is_tmem_load`, `is_tmem_store`, `tmem_copy_properties`. `find_tmem_tensor_col_offset` takes static layouts only |

The simulator reports an `mbarrier.arrive` (a pipeline's release) by a warp whose
`tcgen05.ld` has not completed (`fence_view_async_tmem_load` missing before the release).

## Host tensor helpers (m11-host-tensor-helpers)

Torch-free versions of `cutlass.torch` and `cute.testing.convert`, under `cute::host` and
`cute::testing`. Everything is seeded and dependency-free.

| CuTe DSL (Python) | cute-rs |
|---|---|
| `cute.testing.convert(src, dst)` | `testing::convert(&ctx, &src, &dst)` on `host::DeviceTensor`s (feature `cuda`) and `testing::convert_sim(&src, &mut dst)` on launch tensors (`params::HostTensor`), the same kernel. Same shape and one leading mode (extent above 1, stride 1) in both, the same mode; each thread converts `elem_per_copy` elements (16 for a 6-bit type, 8 for a 4-bit one, 4 for an 8-bit one, else 2) and the leading extent must divide by it. The kernel is built once per (shape, strides, source, destination) and cached. Types: Float32, Float16, BFloat16, Float8E4M3FN, Float8E5M2, Float8E8M0FNU, Float6E2M3FN, Float6E3M2FN, Float4E2M1FN, Int4, Int8, Uint8, Int32 (`testing::CONVERT_DTYPES`). FP4 and FP6 destinations use the paired `cvt` instructions (sm_100a, sm_120a), so on a GPU they compile only there; the simulator runs them everywhere |
| `cutlass.torch.TensorInitType` with `ScalarInitConfig`, `RandomInitConfig`, `GaussianInitConfig` | `host::TensorInit::{Skip, Scalar(v), Random { min, max }, Gaussian { mean, std, scale }}` with the defaults `TensorInit::{ZERO, RANDOM, GAUSSIAN}`; `Random` draws integers in `min..max` (the upper bound excluded) as f32, `Gaussian` is `normal(mean, std) * scale` |
| `cutlass.torch.create_and_permute_torch_tensor(shape, dtype, permute_order, init_type, init_config, device)` | `host::create_and_permute_tensor::<T>(&shape, permute_order, init, seed)`: a `host::HostTensor<T>` (the elements in memory order plus a `Layout`), created row-major and permuted into a strided view; the values are drawn in memory order as f32, then converted to `T` |
| `cutlass.torch.matrix(l, mode0, mode1, is_mode0_major, dtype, init_type, init_config, device)` | `host::matrix::<T>(l, mode0, mode1, is_mode0_major, init, seed)`: shape `(mode0, mode1, l)`, strides `(1, mode0, mode0 mode1)` or `(mode1, 1, mode0 mode1)` |
| `cutlass.torch.get_leading_dim(t)` | `host::get_leading_dim(&shape, &strides)` and `HostTensor::leading_dim()`: the first stride-1 mode of extent above 1 (else the first stride-1 mode), `None` without one |
| `cutlass.torch.cute_tensor_like(data_ref, cutlass_dtype, is_dynamic_layout, assumed_align)` | `host::cute_tensor_like(&data_ref, dtype, is_dynamic_layout, Some(align))`: a `params::HostTensor` over its own copy of the values |
| `cutlass.torch.convert_cute_tensor(f32_tensor, cute_tensor, dtype, is_dynamic_layout)` | `host::convert_cute_tensor(&f32_src, &mut dst)`: Int4, FP8, E8M0, FP6 and FP4 go through `convert_sim`; the NV E5M3 scale type (which `convert` lacks, as `TensorSsa::to` does) and the wide types are rounded on the host |
| `cutlass.torch.prepare_tensors_for_gemm(mnkl, a_dtype, b_dtype, c_dtype)` | `host::prepare_tensors_for_gemm((m, n, k, l), a_dtype, b_dtype, c_dtype, seed)`: K-major A `(m, k, l)` and B `(n, k, l)`, N-major zero C, as launch tensors with the leading dimension dynamic, plus the operands' rounded values as `HostTensor<f32>` references |
| `cutlass.torch.as_tensor(pointer, shape, dtype)` | `unsafe host::as_tensor(ptr, &shape)` and `as_tensor_dyn(ptr, &shape, dtype)`: row-major copies of raw host memory |
| `cutlass.torch.dtype(torch_dtype)`, torch dtype names | `host::{dl_data_type, dtype_from_dl}` (DLPack `{code, bits, lanes}` of every type, packed views as lanes), `host::{torch_dtype_name, dtype_from_torch_name}` (`"torch.float16"`, ...), `host::describe_dtype` |
| `cutlass.torch.default_stream()`, `current_stream()` | `cuda::CudaContext::default_stream()`; a stream another library owns is `unsafe CudaStream::from_raw(&ctx, raw)`, passed to `LaunchBuilder::stream` (see the `cute::host` docs) |

The helpers keep the existing `KernelArg` and `params::HostTensor` constructors working
(`HostTensor::to_param()`, `to_kernel_arg()` and `from_kernel_arg()` convert).

Limitations: there is no torch or `tch` interop (a Rust tensor crate goes through DLPack,
`from_dlpack`; an optional `tch` feature reading torch's current stream is future work);
the random generator is cute-rs's own SplitMix64 with Box-Muller normals, so a seed does
not give torch's values; `TensorInit::Skip` zeroes instead of leaving memory
uninitialised; `convert` needs flat layouts and the same leading mode in both tensors, and a
float outside an integer destination's range converts to an unspecified value (the host
helpers saturate).
| `cutlass.utils.profiling`: `register_symbol`, `get_symbol_registry`, `dump_kernel_symbols`, `reset_symbol_registry`, `register_barrier` (called by every pipeline `create(name=...)`), `dump_profiling_metadata` | `profiling::{register_symbol, get_symbol_registry, dump_kernel_symbols, reset_symbol_registry, register_barrier}` act on a per-kernel registry in the builder; `Kernel::{symbols, kernel_symbols_json, dump_profiling_metadata(dir, extra)}` (`kernel_symbols.json`, nothing written without barriers or warps) |
| `Scope(name or names, iket=, **metadata)`, `.switch`, `.close`, `WarpScope(name, warp_start=, warp_end=)` | `Scope::{new, path, with(names, iket, metadata)}` (RAII guard, `switch`, `close`) and `scope(name, \|\| ..)`, `WarpScope::{new, with}`. Instructions record the scope (`Inst::scope`, `KernelIr::scopes`); MLIR gets nested NameLocs (`loc("a"("b"("f.rs":1:1)))`), IR dumps print `// scope a/b`, simulator errors end with `(in profiling scope `a/b`)`. A scope opened in an `if_`/`for_range` body must close there |
| `cute.experimental.iket`: `mark`, `range_start`/`range_end`, `range_push`/`range_pop`, `sentinel_token`, `dag(name).edge` | `mark`, `range_start`, `range_end`, `range_push`, `range_pop`, `sentinel_token`, `dag`, with `Payload` (i8..u64, f32, pointer) and `RangeToken` (implements `Carry`). cute-rs has its own lowering (NVIDIA's IKET pass is closed): `KernelOptions::iket(IketOptions { capacity, clock })` makes each event an 8-word record appended by the warp's lowest active lane to the module global `IKET_GLOBAL`, reserved with one atomic add (overflow sets a flag). Off by default, and then the calls emit nothing (identical MLIR and PTX). `Kernel::decode_iket(&words)` gives an `IketTrace` (records, paired `spans`, `unmatched`, `dropped`; `write_chrome_trace` for Perfetto); the simulator stamps events with its instruction counter, so traces are deterministic |

## cute_ext foundations (m11-cute-ext-core)

`cutlass.cute.experimental` (cute_ext) is `cute::experimental`: build-time library functions
used in ordinary `Kernel::build` closures (no separate entry point), with builder hooks in
place of cute_ext's compiler passes. The submodules mirror CuTe DSL's (`core`, `memory`,
`algorithm`, `math`, `pipeline_ops` and `pipeline` (m11-cute-ext-pipelines; tests in
`tests/cute_ext_pipelines.rs`), `cluster_mailbox` and `accumulator_buffering`
(m11-cute-ext-sched-tmem; tests in `tests/cute_ext_sched_tmem.rs`), `host_runtime`, `utils`
(the GEMM building blocks, m11-cute-ext-gemm; tests in `tests/cute_ext_gemm.rs`), and the
scaffolding of later parts: `iket`). cute_ext's Blackwell kernels (the GEMM building blocks and
`cute_ext_dense_gemm_persistent`) are verified only on the simulator and with `ptxas` (sm_100a,
sm_120a): no SM100 or SM120 GPU is available to cute-rs's tests.

| CuTe DSL | cute-rs |
|---|---|
| `OperationTypeEnum` (the values cute_ext uses) | `experimental::OperationType` (`Sm90TmaLoad` ... `ReduceSwapShuffle`), with `name()` (`"SM90_TMA_LOAD"`), `from_name`, `ALL` |
| `cute_ext.allocate(type, address_space, layout, alignment, is2cta=False)` | `experimental::allocate::<T>(space, layout, alignment)` and `allocate_with(.., is2cta)`. `Smem`: an aligned shared buffer; a swizzled composed layout keeps its swizzle and needs an alignment of at least one swizzle period (refused otherwise). `Rmem`: a register fragment with the layout. `Tmem`: a column range (`alignment` in columns) of one pooled allocation per kernel: a finish hook inserts at the start of the body warp 0's `tcgen05.alloc` of the next power of two of at least 32 columns covering every range, with the address published through shared memory, a fence and a barrier, and at the end a fence, a barrier (and a cluster barrier for `is2cta`, which uses the `cta_group::2` forms), `relinquish_alloc_permit` and `dealloc`. More than 512 columns, or an `allocate(tmem)` inside `if_` or a runtime loop, is refused |
| Automatic predication (`cute_ext.copy` predicates against the kernel argument's shape), `predicated_tensor_origin(tensor)` | Every global-memory tensor (a parameter, or one made on a global pointer) is a predication origin: views made from it (slices, the divides, `composition`, `compose`, `group_modes`, `local_tile`, `local_partition`, `domain_offset`, `experimental::partition`, `ThrCopy::partition_s/d`, `ThrMma::partition_a/b/c`) record themselves (`Tileable::map_layout`), and the copies here replay them on an identity tensor over the origin's shape to predicate with `elem_less`; statically in-bounds views copy unpredicated. `predicated_tensor_origin(&t)` starts a new origin. A view made by an operation that does not record itself (`coalesce`, the products, `Tileable::with_layout`) has lost its origin, and the copies refuse it, naming `predicated_tensor_origin` |
| `cute_ext.copy(src, dst, *, copy_atom)` | `experimental::copy(&src, &dst, &atom)`: `copy_if` with the automatic predicate (each atom application guarded by its first element, so a vector must lie wholly inside or outside), plain `copy` when none is needed |
| `cute_ext.partition(buffer, agent_id, *, layout_tv, tiler)` | `experimental::partition(&buffer, agent, &layout_tv, tiler)`: `zipped_divide` by the tiler, the tile composed with the TV layout, the agent's slice: `(V, REST)` |
| `cute_ext.partition_and_copy(thr_copy, src, dst)` | `experimental::partition_and_copy(&thr_copy, &src, &dst)`: memory operands through `partition_s`/`partition_d`, register operands retiled; SMEM -> TMEM sources get `append_s2t_broadcast_mode`; `ldmatrix`/`stmatrix` atoms copy with the atom, register <-> memory through `simt_auto_vec_copy`, global -> shared through its `cp.async` form, the rest through `copy` with `make_copy_atom(op)` |
| `cute_ext.simt_auto_vec_copy(src, dst, async_op=False)` | `experimental::simt_auto_vec_copy(&src, &dst, async_op)`: `autovec_copy`'s vectors; `async_op` issues `cp.async` (cache always) of 4-16 bytes. A predicated copy that is not statically in bounds goes element by element (and the async form synchronously, zero-filling the elements predicated off), so no vector straddles the bound |
| `cute_ext.dot(mma_atom, a, b, c)`, `dot_block_scaled(mma_atom, a, sfa, b, sfb, c)`, `dot_sparse(mma_atom, a, e, b, c)` | `experimental::dot(&mma, &a, &b, &c)`, `dot_block_scaled(&mma, &a, &sfa, &b, &sfb, &c)`, `dot_sparse(&mma, &a, &e, &b, &c)`, taking a `TiledMma` or a bare `MmaAtom` (`AsTiledMma`, a one-atom tiled MMA): operands below rank 3 get trailing unit modes, the tile counts are checked, then `gemm`/`gemm_sparse` accumulate into `c`. A block-scaled or sparse atom in `dot`, or a plain one in the others, is refused, naming the right call |
| `cute_ext.warp_reduce(input, output, reduction_t_map, intra_warp_reduce_type, body)` | `experimental::warp_reduce(&input, &output, &map, OperationType::ReduceDownShuffle or ReduceSwapShuffle, \|a, b\| ..)`: the map's (flattened) modes are powers of two covering 32 lanes; stride-0 modes select groups, the others number a group's lanes compactly, and their lane strides are the `shfl.bfly` offsets. Down: every lane gets its group's reduction element by element. Swap: a transposing reduce-scatter by recursive halving, leaving lane `i` with slice `i` (input `n x frag`, output `frag`). The shuffles are warp-wide, so the simulator's shuffle checks catch a divergent call |
| `cute_ext.elementwise(inputs, outputs, body)` | `experimental::elementwise((&a, &b), (&c,), \|(x, y), (c,)\| (..,))`: tuples of one to four tensors of one size (`ElementwiseTensors`), the body getting the inputs' values and the outputs' current ones (zero for registers not yet written) |
| `@cute.experimental.kernel`, `jit`, `compile` | Ordinary `Kernel::build`/`jit_for` closures: the hooks above run when the body returns |
| `cute_ext.tma_load(src, dst, mbar, *, cta_v_map, tma_operation_type, vmnk_layout, multicast_mode, internal_type, update_expect_tx)`, `tma_load_multicast(...)` (deprecated), `tma_store(src, dst, *, cta_v_map, internal_type)`, `tma_reduce_store(src, dst, *, kind, ...)` (m11-cute-ext-tma) | `experimental::tma_load(&g, &s, &mbar)` / `tma_load_with(.., &TmaLoadOptions)`, `tma_load_multicast` (`#[deprecated]`), `tma_store(&s, &g)` / `tma_store_with(.., &TmaStoreOptions)`, `tma_reduce_store(&s, &g, ReductionKind)` / `tma_reduce_store_with`. The global operand is a view of a predication origin (a kernel argument, or a tensor marked with `predicated_tensor_origin`); its root tensor, the buffer's layout and the view's CTA-V map (or `cta_v_map`) make a TMA atom, so one hidden tensor-map parameter per distinct (root, box layout, operation, internal type), encoded at launch; the view's coordinates give the instructions'. One elected lane of the calling warp issues; a load's `update_expect_tx` (default) adds the buffer's bytes with `mbarrier.expect_tx` (no arrival; pass `false` under a pipeline, else the simulator reports the phase stuck owing bytes). Multicast types need `vmnk_layout` and `multicast_mode` (a `vmnk_layout` mode index): the CTAs along it split the box (`tma_partition`) and deliver to `make_layout_image_mask` of it; multicast keywords without an op type are refused. Stores don't commit a bulk group. A root on a device-side pointer gets its descriptor built in the device workspace each call (template on the workspace parameter, `copy_tensormap` + `update_tma_descriptor` + tensormap-proxy fences, 128 bytes per CTA) |
| `cute_ext.TMAStorePipeline(stages, arv_count, barrier_id, tma_warp_id, index=0)`: `acquire_sync`, `commit_sync`, `release_advance`, `tail`, `get_num_stages`, `get_index` | `experimental::TMAStorePipeline::new(stages, arv_count, barrier_id, tma_warp_id)` / `with_index`, the same methods (`release_advance(&mut self)`); over `PipelineTmaStore` and `NamedBarrier`; implements `Carry` (the index is loop-carried) |
| `cute_ext.create_pipeline(stage, producer, consumer, producer_arv_count, consumer_arv_count, arrival_mask=None, *, locale=None)`, `create_pipeline_with_mask`, `PipelineLocale.cta()/cluster(full, empty)`, `MbarrierOwner` | `experimental::create_pipeline(stage, producer, consumer, p_count, c_count, mask, locale) -> (RawPipeline, PipelineState, PipelineState)`, `create_pipeline_with_mask`, `PipelineLocale::cta()/cluster(..)`, `MbarrierOwner::{All, CtaPairLeader, CtaPairPeer, Rank(n)}`. A side is one `OperationType` or a set (`OperationTypes`; duplicates refused, kinds sharing one way of arriving). The pair maps to an m9/m10 class (`PipelineClass`: TMA -> UMMA `PipelineTmaUmma`, TMA -> threads `PipelineTmaAsync`, threads -> UMMA `PipelineAsyncUmma`, UMMA -> threads `PipelineUmmaAsync`, threads -> threads `PipelineAsync`, `SM100_LAUNCH_CONTROL -> LD_SHARED` `PipelineClcFetchAsync`; other pairs refused) whose full/empty `MbarrierArray`s the ops use (functions over the classes, not new IR ops). `mask` is `ArrivalMask` (an `Int16`, none, or the legacy `multicast` flag: `true` refused); it is the UMMA commits' multicast mask, required when the ops span CTAs (2SM or multicast). Thread/TMA arrivals go to the barrier's owner (default each CTA's own; in a CTA pair the side towards UMMA goes to the even rank). Both states start at stage 0, phase 0: the phase is the ring's round, so `get_pipeline_produce_phase` and `get_pipeline_consume_phase` agree; the producer waits on the empty barrier's other parity |
| `cute_ext.get_arrival_count(stage, producer, consumer, *, side, num_threads_per_cta=1, op_subset, elect_one_sync, elect_leader_cta, fan_in, locale)` | `experimental::get_arrival_count(stage, producer, consumer, Side::Producer/Consumer, &ArrivalCountOptions)`: `1 x arrives_per_cta x fan_in`; thread kinds count `num_threads_per_cta` (one per warp with `elect_one_sync`, which needs a multiple of 32), TMA/CLC/UMMA one per CTA; `fan_in` from the barrier's owner (2 for a pair's leader, 1 with `elect_leader_cta`; the cluster size for a rank), required for a UMMA side spanning CTAs. Static when its inputs are; tests check it against each class's own counts |
| `producer_acquire`, `producer_commit(*, elect_one_sync, elect_leader_cta)`, `consumer_wait`, `consumer_release(..)`, `consumer_release_elect_one_sync`, `producer_try_acquire`/`consumer_try_wait(*, token)`, `pipeline_advance_iterator`, `create_pipeline_state_at`, `get_pipeline_produce/consume_phase`, `get_pipeline_produce/consume_stage`, `get_mbarrier`, `mbarrier_expect_tx(mbar, tx, ctaId, elect_one_sync)`, `pipeline_tail`, `consumer_tail` | Same names in `experimental`, plus `producer_acquire_with`/`consumer_wait_with(.., token)` (CuTe DSL's `skipWait` operand). Commits/releases arrive by kind: TMA and CLC one elected plain arrive (the loads add their bytes; CLC producers call `mbarrier_expect_tx`), UMMA an elected `tcgen05.commit` (`cta_group::2` for 2SM kinds, masked), thread kinds per thread or one per warp after `__syncwarp`, with the kind's fence first (`ST_SHARED` -> UMMA `fence.proxy.async.shared::cta`, `R2T` `tcgen05.wait::st`, `T2R` `tcgen05.wait::ld`, `LD_SHARED` after CLC `fence.proxy.async`). A true token skips the probe. `StageToken` holds the stage's barrier (full for the producer, empty for the consumer). `pipeline_tail` waits, for each stage from the state, for the previous round's releases (only on the empty barriers' owner) |
| `GenericPipelineBase` (`producer_acquire`, `producer_try_acquire`, `get_producer/consumer_stage`, `get_producer/consumer_phase`, `producer_acquire_and_get_stage/phase`, `producer_commit(_and_advance)`, `consumer_try_wait`, `consumer_wait(_and_get_stage/phase)`, `consumer_release(_and_advance)`, `producer/consumer_tail`, `increment_state`, `create_state_at`), `GenericPipeline.create(*, producer, consumer, producer_arv_count, consumer_arv_count, num_stages, locale)` | `experimental::GenericPipelineBase` with the same methods (`&mut self`; those returning `self` return `&mut Self`; `consumer_wait(state)` is `consumer_wait_at`), keeping both states; `Carry` (the two states), so it goes through `for_range_carry`/`while_carry`. `GenericPipeline::create(producer, consumer, p_count, c_count, num_stages, locale)`. The classes below deref to it |
| `TMAToUMMAPipeline.create(*, num_stages, mma_operation_type, tma_operation_type=SM90_TMA_LOAD, cluster_layout_vmnk=None)`, `create_with_mask`, `TMAToAsyncPipeline.create(*, num_stages, consumer, consumer_arv_count)`, `AsyncToUMMAPipeline.create(*, num_stages, producer, producer_arv_count, mma_operation_type)`, `UMMAtoAsyncPipeline.create(*, num_stages, consumer, consumer_arv_count, mma_operation_type, cluster_layout_vmnk=None)`, `create_with_mask` | `experimental::TMAToUMMAPipeline::create(num_stages, mma, tma, layout)` (`SM100_TMA_LOAD_2SM`: mask = `make_layout_image_mask` over mode 0 at the CTA's coordinate, the leader alone commits; a multicast load takes `create_with_mask`: A/B multicast masks of the CTA and its pair peer, `fan_in` = size(M) + size(N) - 1; `cluster_layout_vmnk` required for both, a 2SM MMA needs the 2SM load), `TMAToAsyncPipeline::create`, `AsyncToUMMAPipeline::create`, `UMMAtoAsyncPipeline::create(..)` (2SM: the pair mask, consumers of both CTAs release to the leader, so `consumer_arv_count` counts the pair's threads). Commits and releases fix their flags as CuTe DSL's classes do (one elected thread; the leader CTA for a 2SM load) |
| `create_circular_buffer_pipeline(pipeline, state, stages, count_per_stage, count_per_iteration)`, `circular_buffer_pipeline_consume(.., *, elect_leader_cta)`, `circular_buffer_pipeline_consumer_release(.., *, elect_one_sync, elect_leader_cta)`, `circular_buffer_pipeline_advance_iterator`, `TMAToUMMACircularPipeline.create(*, num_stages, mma_operation_type, count_per_stage, count_per_iteration, tma_operation_type, cluster_layout_vmnk)` | Same names; `CircularBufferPipelineState` (`Carry`). Defined here: position `p` in units on a ring of `stages x C` units (C = `count_per_stage`, I = `count_per_iteration`, one dividing the other, I at most the ring); consume waits on the full barrier of each stage `[p, p + I)` overlaps that no earlier iteration waited on (the current stage at offset 0 when I <= C, the I / C stages from it when I > C); release arrives on the empty barrier of each stage whose last unit is in the range; advance moves `p` by I, the phase flipping on wrap. `stage_index()`/`unit_offset()` are cute-rs additions for indexing. `TMAToUMMACircularPipeline`'s `consumer_wait/release(_and_advance)` use them; `get_consumer_stage`, `consumer_wait_and_get_stage` and `consumer_tail` panic. Simulated with (4, 1), (2, 2), (1, 2) (and (4, 2), (2, 4)) on thread kinds and in a TMA -> UMMA GEMM; a stage released early is refilled under the reader and the race detector reports it |
| `GroupedGemmSchedulerPipeline.create(*, num_stages, producer_arv_count, consumer_arv_count)`, `CLCPipeline.create(*, num_stages, consumer_arv_count, response_buffer)` (`producer_commit`, `get_response_ptr(stage_idx)`, `issue_next()`) (m11-cute-ext-sched-tmem) | `experimental::GroupedGemmSchedulerPipeline::create(stages, p_count, c_count)`: `SREG_CTAID -> LD_SHARED` (`PipelineAsync`), the base's `consumer_wait`, `consumer_release(elect_one_sync, elect_leader_cta)`, `producer_commit_and_advance`; the scheduler warp publishes `GroupedWorkTileInfo` fields into a shared ring. `experimental::CLCPipeline::<T>::create(stages, c_count, &response_buffer)` (compact shared buffer of one 16-byte slot per stage): `SM100_LAUNCH_CONTROL -> LD_SHARED` with locale `cluster(all, 0)`, so consumers of every CTA release to CTA 0 (`c_count` counts the cluster's threads); full barriers count `get_arrival_count(.., producer, num_threads_per_cta=1)` = 1. `issue_next` (one elected thread: `mbarrier.expect_tx` of 16 bytes on the stage's full barrier of each cluster rank through `mapa`, then the multicast `try_cancel` into `get_response_ptr(stage)`) is refused outside `producer_acquire*` ... `producer_commit` when the kernel is built; `producer_commit` arrives once on every rank's full barrier. `GenericPipelineBase::set_producer_state/set_consumer_state` stand for CuTe DSL's state assignment. Simulated against plain Rust: CLC on a 7-CTA grid with 1 and 2 stages, the grouped scheduler (4 groups, one empty, 2 persistent CTAs) feeding three warps; sm_100a PTX through `ptxas` |
| `B32PayloadCodec(fields, name=None)` (`field_names`, `chunks_for_alignment`, `encode_fields`, `decode_from_b32_ptr`, `storage_words`, `tx_bytes`, `required_alignment`, `from_struct`), `b32_payload(cls)`, `dsmem_payload`, `ClusterMailbox(payload, storage)` (`send_to_pipeline_stage(stage_token, peer_rank, **fields)`, `recv()`), `CTA_PAIR_PEER_LOCALE` | `experimental::B32PayloadCodec::new(&[(name, DType)], name)`, `from_struct(&StructLayout)` (scalar members of a `cute_struct!`), `from_payload::<P>()`; `b32_payload::<P>()`/`dsmem_payload::<P>()` for a Rust type implementing `B32Payload` (`NAME`, `fields`, `to_fields`, `from_fields`). Boolean/Int8/Int16/Int32/Float16/BFloat16/Float32 take one word (narrow ints and 16-bit float bits sign-extended), Int64 two (low first); other types refused. `encode_fields(&[(name, AnyValue)])` converts values to their field's type and names missing/unknown/repeated fields; `decode_from_b32_ptr` gives a `DecodedPayload` (`get::<T>`, `get_any`, `values`). `ClusterMailbox::new(&codec, MemRange<i32>)` (length checked), `send_to_pipeline_stage(&token, peer, &fields)` / `send_payload(&token, peer, &p)` (the token must come from a `CTA_PAIR_PEER_LOCALE` pipeline: `StageToken::locale()`; one elected thread: `mbarrier.expect_tx(tx_bytes)` on the peer, then `st.async` v4/v2/scalar chunks by the storage's alignment), `recv()`/`recv_as::<P>()`. `AnyValue::dtype()`/`cast::<T>()` added. Simulated: the two CTAs of a pair exchange 4 rounds through 16-, 8- and 4-byte aligned mailboxes of mixed types |
| `plan_accumulator_buffering(*, one_stage_layout, accumulator_dtype, epilogue_tile, epilogue_subtile_columns, auxiliary_tmem_allocations, tmem_capacity_columns=512, desired_disjoint_depth=2, max_auto_disjoint_depth=2, allow_overlap=False, overlap_alignment_columns=1, accumulator_alignment_bytes=16, overlap_policy)`, `AccumulatorBufferingPlan` (`producer_acquire_and_get_view`, `consumer_wait_and_get_view`, `uses_overlap`), `AccumulatorDrain`, `TmemAllocationRequest(name, columns, alignment_columns)`/`from_layout`, `AccumulatorBufferingKind`, `AccumulatorOverlapPolicy` | `experimental::plan_accumulator_buffering(&layout, dtype, &AccumulatorBufferingOptions)` (the same defaults), the integer planner `plan_accumulator_buffering_columns(..)`, `AccumulatorBufferingPlan` (`new(selection, layout)` checks the kind's invariants; getters; `storage_layout()` appends `storage_depth` views `stride x stride[0][1]` apart), `AccumulatorPlanSelection`, `TmemAllocationRequest::{new, from_layout}`. Order: disjoint depth `min(desired, max_auto)` down to 2, then (with `allow_overlap`) the smallest overlap whose stride is aligned, footprint fits and release point lies in the drain, then one view, else refused; requests packed after the accumulator at their alignments. `producer_acquire_and_get_view(&mut pipe, &buf)` / `consumer_wait_and_get_view(&mut pipe, &buf) -> (view, AccumulatorDrain<P>)` select the last mode by stage index (disjoint) or the round's parity (overlapping: cute-rs's producer and consumer phases agree, so no flip) and check the pipeline's depth; `AccumulatorDrain` (`Carry`) has `subtile_index(ordinal, count)` (view 0 of an overlapping plan in reverse), `after_t2r(ordinal)` (the early `consumer_release_and_advance`) and `into_pipeline()`. Host tables and the `dense_block_scaled_gemm.py` defaults; simulated R2T -> T2R kernels with disjoint and overlapping plans, and an early release the race detector reports |
| `cute_ext.host_runtime.AllocationRequirement`, `QueryDeviceWorkspaceFunc`, `compiled_fn.get_aux_func(QueryDeviceWorkspaceFunc, kernel=k)(*args)` | `experimental::AllocationRequirement { size_in_bytes, alignment }`, `Kernel::device_workspace_requirement(cfg, &args)` (or `QueryDeviceWorkspaceFunc.call(&kernel, cfg, &args)`). Kernels declare needs with `request_device_workspace(WorkspaceSize::{Bytes, PerCta, PerScalar{count, bytes}}, align)` or `device_workspace_cta_slot(bytes, align)`; the workspace is a trailing `u8` buffer parameter `device_workspace` the caller passes last. The simulator and CUDA launcher refuse a missing or too-small workspace (the launcher checks its alignment like any pointer's) |
| `cute.experimental.utils` (m11-cute-ext-gemm): `is_preferred_cluster(shape)`, `get_cta_v_map_ab(gmem_tensor, mma_tiler_mnk, tiled_mma, input_operand)`, `get_cta_v_map_c(gmem_tensor, epi_tile)`, `make_tmem_layout_acc(tiled_mma, mnk_tiler, acc_stage)`, `make_tmem_layout_a(tiled_mma, mk_tiler, stage)`, `make_t2r_rmem_layout(tiled_copy_t2r, gC_mnl_epi, tidx)`, `mainloop_mma(tiled_mma, a_buffer, b_buffer, acc_buffer, k_tile_start, k_tile_end, mma_inst_tile_k, a_buffer_pipeline, b_buffer_pipeline, ab_buffer_same_pipeline=True, accumulate_to_acc=False)`, `mainloop_mma_sm120(tiled_mma, tCrA, tCrSFA, tCrB, tCrSFB, acc_buffer, mma_inst_tile_k, accumulate_to_acc=True)`, `epilogue_tma_store(cta_tile_shape_mnk, use_2cta_instrs, tmem_acc_buffer_staged, gmem_d, cta_d_tile_coord, tma_store_pipeline, tma_store_warp_id, epilogue_op, d_major_mode=, tid_x_in_group=, amax_out=, accumulator_drain=)` | The same names in `experimental` (and `experimental::utils`). `get_cta_v_map_ab(&g, mnk, &mma, MmaOperand::{A, B, SFA, SFB})`: the tile of the identity layout partitioned by the MMA (`thrfrg`), its value mode `(FrgV, Rest_MN, Rest_K)` (the tile itself for SM120 warp MMAs); a scale-factor operand of an MMA without scales is refused. `make_tmem_layout_acc/a`: `make_fragment_c/a`'s TMEM layouts with the stages appended, for `allocate` in TMEM. `mainloop_mma(..) -> (P, P)` over any convenience pipeline (`AsGenericPipeline`): per k-tile (runtime bounds) `consumer_wait_and_get_stage`, `mma_inst_tile_k` `dot`s on the shared buffers' descriptor fragments (`ACCUMULATE` off for the first unless `accumulate_to_acc`), `consumer_release_and_advance`; with `ab_buffer_same_pipeline` both results are the advanced A pipeline. Refused: a pipeline not feeding a UMMA of the tiled MMA's CTA group, a buffer whose stage count is not its pipeline's, shared pipelines that differ. `mainloop_mma_sm120` zeroes the accumulator unless `accumulate_to_acc`, then `dot_block_scaled` over the fragments' last mode. `epilogue_tma_store(..) -> TMAStorePipeline`, `epilogue_tma_store_with(.., &EpilogueTmaStoreOptions { d_major_mode, tid_x_in_group, amax_out })`, `epilogue_tma_store_drain(.., AccumulatorDrain<P>, &opts) -> (TMAStorePipeline, P)`: `compute_epilogue_tile_shape`'s subtiles of the `(M, N)` stage, each `tcgen05.ld` (`get_tmem_load_op`, then its wait) into an `allocate`d register buffer, `epilogue_op` on the `TensorSsa` in D's type, `acquire_sync`, the store to the pipeline's stage of a `make_smem_layout_epi` buffer (`partition_and_copy` of `make_tiled_copy_D`), `commit_sync`, `tma_store` by `tma_store_warp_id`, `release_advance`; `tail` at the end. `amax_out`: the running max of `abs` per thread, `redux.sync` max (NaN-propagating) per warp, `atomic_fmax` from lane 0. A drain visits the subtiles in its order and releases after the plan's subtile. Refused: a pipeline barrier not counting the warpgroup's 128 threads, a TMA warp other than `tma_store_warp_id`, an accumulator that is not the CTA tile. Fixes found here: `tma_load` of a 2SM kind now arms only the pair leader's barrier, with both CTAs' bytes (the peer's expect_tx went to its own unused barrier, so the leader's phase could complete before the peer's tile landed), and `partition_and_copy` passes a `tcgen05.ld`/`st` register operand through unretiled. Ported example: `cute_ext_dense_gemm_persistent` (`dense_gemm_persistent.py`); `dense_gemm_ptr_array.py` and `dense_gemm_persistent_dynamic.py` are not ported |

