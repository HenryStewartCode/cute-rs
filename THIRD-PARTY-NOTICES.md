# Third-party notices

cute-rs is an independent Rust implementation of NVIDIA's CuTe and CuTe DSL;
it is not affiliated with or endorsed by NVIDIA. Most of it is original, but
some files port algorithms, tables and example programs from these parts of
[NVIDIA CUTLASS 4.8.1](https://github.com/NVIDIA/cutlass/tree/v4.8.1), each of
which is under the BSD-3-Clause licence:

* CuTe, the C++ template library in `include/cute`, and CUTLASS's
  `include/cutlass/fast_math.h`. A few formulas come from
  earlier CUTLASS releases, whose `include/cute/layout.hpp` later changed:
  `left_inverse` and `upcast` follow CUTLASS 3.8.0 (and 3.0 to 3.7), and
  `logical_product` follows
  the formula that CUTLASS 4.8.1's CuTe documentation
  (`media/docs/cpp/cute/02_layout_algebra.md`) gives, which `include/cute`
  used up to CUTLASS 4.7.1;
* `cutlass_compiler/cutegen`, the layout library of NVIDIA's CuTe compiler:
  `tile_to_shape` with an order, `increment_coord`, the rules of
  `composition_impl` for runtime values, and scaled-basis strides and
  ratios;
* example programs: the CuTe tutorial `examples/cute/tutorial/sgemm_2.cu`,
  and the CuTe DSL examples in `examples/python/CuTeDSL`.

The ported code was rewritten in Rust. The table below lists, for each
cute-rs file, the NVIDIA files it ports from and what it ports; each of these
files also says so in a header at its top. NVIDIA's copyright notices and the
licence follow the table.

NVIDIA's Python CuTe DSL package, `python/CuTeDSL` in CUTLASS, is not under
the BSD-3-Clause licence but under the NVIDIA End User License Agreement.
cute-rs takes only API names and semantics from it: no code, docstrings or
message text. The CuTe DSL examples and notebooks in `examples/python/CuTeDSL`
and the documentation in `media/docs`, which the table and the test data
below draw on, are BSD-3-Clause like the rest of CUTLASS.

The simulator's `ldmatrix` and `mma.sync` semantics follow the PTX ISA
documentation.

`src/external/dlpack.rs` writes out the C ABI of DLPack's `dlpack.h` (version 1.3,
Apache-2.0, "Copyright (c) 2017 - by Contributors"): the field order and types of
`DLPackVersion`, `DLDevice`, `DLDataType`, `DLTensor`, `DLManagedTensor` and
`DLManagedTensorVersioned`, and the numbers of its device types, data type codes and
flags. These are interface facts restated in Rust; no other part of DLPack is used.
DLPack's licence, the Apache License 2.0, is the text in `LICENSE-APACHE`.

`src/tvm_ffi.rs` writes out part of the C ABI of Apache TVM FFI (apache-tvm-ffi 0.1.14,
`include/tvm/ffi/c_api.h` and `include/tvm/ffi/extra/c_env_api.h`; Apache-2.0, "Apache TVM
FFI, Copyright 2024-present The Apache Software Foundation. This product includes software
developed at The Apache Software Foundation (http://www.apache.org/)."): the layouts of
`TVMFFIAny` and `TVMFFIObject`, the numbers of `TVMFFITypeIndex`, the safe-call signature,
the signatures of `TVMFFIErrorSetRaisedFromCStrParts` and `TVMFFIEnvGetStream`, the
`__tvm_ffi_` symbol prefix, and the place of a tensor object's `DLTensor` after its header.
These are interface facts restated in Rust; no other part of TVM FFI is used, and nothing
of it is linked. Its licence is the text in `LICENSE-APACHE`.

`src/math/` ports functions from Sun Microsystems' fdlibm as FreeBSD's msun
library carries it and, for some Float32 functions, as newlib does. Each of
those files keeps, word for word, the Sun notices of the msun and newlib files
it ports, and says which functions come from which file; the
[last section](#sun-microsystems-notices) lists them.

## What is ported

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/layout/layout.rs` | `include/cute/layout.hpp` | `coalesce` (`bw_coalesce`), `coalesce` with a target profile (`coalesce(layout, trg_profile)`, `Layout::coalesce_profile`), `filter_zeros`, `filter`, `composition` (`composition_impl`, with its runtime `ceil_div` and `min` for dynamic values), `complement` (`complement_prefix`, the rank-1 rule for a runtime stride, and the complement in a cotarget shape that `logical_divide` takes), `right_inverse` (it skips runtime strides), `with_shape`, `nullspace`, `append`, `prepend`, `group`, `select`, `logical_divide`, `zipped_divide`, `tiled_divide` and `flat_divide` (all three with `tile_unzip`, as the helper `zip2_by`), `blocked_product`, `raked_product`, `zipped_product`, `tiled_product`, `flat_product`, `tile_to_shape`, `max_common_layout`, `max_common_vector`, `dice`, `make_ordered_layout`, `make_layout_like` (`make_layout(shape, compact_order(filter_zeros(stride, shape), stride))`), `make_fragment_like` of a layout (`make_fragment_like_layout`), `max_alignment` of a layout (with runtime leaves counted by their divisors), `slice_and_offset` (`Layout::slice`), `cosize` |
| | `include/cute/layout.hpp` of CUTLASS 3.8.0 | `left_inverse` (the `right_inverse` of the layout and its complement, as CUTLASS 3.0 to 3.8 compute it; later releases build the inverse directly) and `upcast` (with `shape_div`, as CUTLASS 3.0 to 3.8 compute it; later releases round up with `ceil_div`) |
| | `media/docs/cpp/cute/02_layout_algebra.md` | `logical_product`, with the formula given there, `(A, complement(A, size(A)*cosize(B)) ∘ B)`, which `include/cute/layout.hpp` used up to CUTLASS 4.7.1 (4.8.0 and later take `complement(A)` without a cotarget); `blocked_product` and `raked_product` build on it |
| | `include/cute/stride.hpp` | `crd2idx`, `idx2crd`, `compact_major` (`compact_col_major` and `compact_row_major` of a `Shape`, without its stride 0 for size-1 modes), `compact_order` (in `make_ordered_layout`, and the general one, with runtime orders after the static ones, in `make_layout_like` and `make_fragment_like_layout`) |
| | `include/cute/int_tuple.hpp` | `product`, `product_each`, `size` and `shape` with a mode path, `congruent` and `weakly_congruent` (`is_congruent`, `is_weakly_congruent`), and the `ceil_div` of a tuple by an integer (in the complement in a cotarget shape) |
| | `include/cute/underscore.hpp` | `slice` and `detail::lift_slice` (the modes that `Layout::slice` keeps) |
| | `include/cute/algorithm/tuple_algorithms.hpp` | `zip2_by` (`detail::zip2_by`), the helper with which `zipped_divide`, `tiled_divide` and `flat_divide` do the job of `tile_unzip` |
| | `include/cute/atom/copy_atom.hpp` | the thread-value layout that `make_tiled_copy(copy_atom, thr_layout, val_layout)` builds (`make_layout_tv`) |
| | `cutlass_compiler/cutegen/include/cutegen/layout.hpp` | `tile_to_shape` with an order, rounding the target shape up (`tile_to_shape_ordered`, which `tile_to_shape` calls), `composition_impl`'s rules for dynamic leaves and for a basis stride on the right-hand side (it picks a mode of the left-hand side), `layout_eval` of basis strides (`Layout::eval`, with the walk over every mode that gives a tuple its nesting, and `slice_and_offset`), the `cosize` of a basis layout (the size of its coshape), the complement of a basis layout bucket by bucket of its coprofile, the complement with no cotarget (the one-argument `complement`, with `coprofile` and `coalesce_z`), `left_inverse` and `right_inverse` of basis layouts, and the divide by a tiler of basis strides (`product_divide` with `complement_extend`) |
| `src/layout/basis.rs` | `include/cute/numeric/arithmetic_tuple.hpp` | `ScaledBasis` and `E`, the sum of an arithmetic tuple and a scaled basis element (`tuple_add`), `basis_value`, `basis_get`, `make_basis_like`, and `make_identity_layout` (`shape:make_basis_like(shape)`) |
| | `include/cute/numeric/integral_ratio.hpp` | `R<N, D>` as `Ratio`, its reduction and products |
| | `cutlass_compiler/cutegen/include/cutegen/scaled_basis.hpp` | `scaled_basis_t`, a scale (static, dynamic or a ratio) and a mode path, its innermost-first printing, and `basis_get` |
| | `cutlass_compiler/cutegen/include/cutegen/ratio.hpp` | `ratio`, its `is_integral`, `reduced` (with a positive denominator) and the products of ratios and integers (`scalar_mul`, reducing a whole result to an integer scale) |
| `src/layout/tree.rs` | `include/cute/int_tuple.hpp` | `shape_div` of two integers, `congruent`, `weakly_congruent`, and `size` and `product_each` of an `IntTuple` |
| | `include/cute/stride.hpp` | `compact_major` (`compact_col_major` and `compact_row_major` of an `IntTuple`, without its stride 0 for size-1 modes) |
| | `include/cute/algorithm/tuple_algorithms.hpp` | `front`, `wrap`, `unwrap`, `flatten`, `unflatten` |
| `src/layout/swizzle.rs` | `include/cute/swizzle.hpp` | `Swizzle<B,M,S>` and its `apply`, with negative shifts (`yyy_msk`, `zzz_msk`, `shiftr` by a signed amount) |
| | `include/cute/swizzle_layout.hpp` | `max_alignment` of a swizzle (`2^M`) and of a composed layout (the gcd of its parts) |
| | `include/cute/layout_composed.hpp` | `ComposedLayout` (`inner o offset o outer`), `make_composed_layout`, the algebra acting on the outer layout (composition, the divides and products, `tile_to_shape`, `coalesce`, `get`, `select`, `group_modes`), and slicing with the offset moved inside |
| | `include/cute/swizzle_layout.hpp` | `get_swizzle_portion` and `get_nonswizzle_portion`, and the `cosize` of a swizzled composed layout (its outer layout's) |
| | `cutlass_compiler/cutegen/include/cutegen/composed_layout.hpp` | `composed_layout_t`: its printing and parsing, `is_normal_layout`, `slice_and_offset` (a layout inner's offset kept inside; a swizzled slice the swizzle acts on affinely decaying to its layout), and `upcast`/`downcast` of the offset |
| `src/layout/recast.rs` | `include/cute/layout.hpp` | `upcast` (with `ceil_div`, as CUTLASS 4.8.1 computes it: `Layout::upcast_by`), `downcast` and `recast_layout` of a layout |
| | `include/cute/swizzle_layout.hpp` | `upcast`, `downcast` and `recast_layout` of a swizzle (the base moved by `log2` of the ratio, the bits clamped) |
| | `include/cute/layout_composed.hpp` | `upcast`, `downcast` and `recast_layout` of a composed layout (the inner, the offset and the outer layout recast together) |
| `src/layout/shape_arith.rs` | `include/cute/int_tuple.hpp` | `ceil_div`, `round_up` and `shape_div` of tuples |
| | `include/cute/stride.hpp` | `idx2crd(idx, shape, stride)` (`Layout::get_hier_coord`) and `crd2crd` (in `Layout::get_flat_coord`) |
| | `include/cute/layout.hpp` | `get_hier_coord` and `get_flat_coord` of a layout |
| `src/tensor.rs` | `include/cute/tensor_impl.hpp` | `local_tile` (with `inner_partition`, and the projection form), `local_partition` (with `outer_partition`, and the projection form), `domain_offset`, and `make_identity_tensor` (a coordinate tensor over `make_identity_layout`) |
| | `include/cute/int_tuple.hpp` | `elem_less` of hierarchical tuples |
| | `cutlass_compiler/cutegen/include/cutegen/layout.hpp` | `increment_coord` (`detail::increment_coord_impl`) |
| `src/atom/copy.rs` | `include/cute/atom/copy_traits.hpp` | the one-thread layouts of `UniversalCopy` (`ThrID`, `SrcLayout`, `DstLayout` and `RefLayout` of `Copy_Traits<UniversalCopy>`), counted in elements instead of bits |
| | `include/cute/atom/copy_traits_sm80.hpp` | the one-thread layouts of `cp.async` (`ThrID`, `SrcLayout`, `DstLayout` and `RefLayout` of the `Copy_Traits` of `SM80_CP_ASYNC_CACHEALWAYS` and `SM80_CP_ASYNC_CACHEGLOBAL`, which their `_ZFILL` forms share), counted in elements instead of bits |
| | `include/cute/atom/copy_traits_sm75.hpp` | the `ldmatrix` source, destination and reference layouts of `SM75_U32x1/x2/x4_LDSM_N` and `SM75_U16x2/x4/x8_LDSM_T`, converted from bits to 16-bit elements, and those of `SM75_U32x1/x2/x4_LDSM_N` converted to 8-bit elements for `LdMatrix8x16x8b` |
| | `include/cute/atom/copy_traits_sm100.hpp` | the `ldmatrix` source, destination and reference layouts of `SM100_U8x8_LDSM_T` and `SM100_U8x16_LDSM_T` (for `LdMatrix16x8x8b`) and of `SM100_SU4_DU8x16_x1/x2/x4_LDSM_N` and `SM100_SU6_DU8x16_x1/x2/x4_LDSM_N` (for `LdMatrix8x16x8b` with `unpack_bits`), converted from bits to 8-bit elements |
| | `include/cute/arch/copy_sm100.hpp` | the byte permutation of `SM100_U8x8_LDSM_T` and `SM100_U8x16_LDSM_T` (destination register `w` byte `q` from `ldmatrix.m16n16` register `2 (w / 2) + q / 2`, byte `2 (w % 2) + q % 2`) |
| | `include/cute/atom/copy_atom.hpp` | `make_tiled_copy` (`make_tiled_copy_impl`), `make_tiled_copy_tv` (`make_tiled_copy(copy_atom, thr_layout, val_layout)`), `make_tiled_copy_a/b/c` (`make_tiled_copy_A/B/C`), `make_tiled_copy_s/d` (`make_tiled_copy_S/D`), `make_tiled_copy_c_atom` (`make_tiled_copy_C_atom`), `make_cotiled_copy`, the tiled source and destination layouts (`get_layoutS_TV`, `get_layoutD_TV`), `retile`, `tile2thrfrg`, `tidfrg_S`, `tidfrg_D`, and the thread slice's `partition_S/D` and `retile_S/D` |
| | `include/cute/layout.hpp` | `detail::coalesce_x`, which `make_cotiled_copy` applies to the left operand of its compositions, as CuTe's `composition` does |
| `src/atom/mma.rs` | `include/cute/atom/mma_traits.hpp` | the `UniversalFMA` atom (`MMA_Traits<UniversalFMA>`: its 1x1x1 shape, one-thread `ThrID`, and A, B and C layouts) |
| | `include/cute/atom/mma_traits_sm80.hpp` | the A, B and C thread-value layouts of the SM80 `mma.sync` atoms (`SM80_16x8_Row`, `SM80_8x8_Row`, `SM80_8x4` and the `MMA_Traits` of the 16x8x8, 16x8x16, TF32 and 8x8x4 operations) |
| | `include/cute/atom/mma_traits_sm89.hpp` | the A and B thread-value layouts of the SM89 FP8 `mma.sync` atoms (`MMA_Traits<SM89_16x8x32_*_TN>`) |
| | `include/cute/atom/mma_atom.hpp` | `make_tiled_mma`, `thrfrg_A/B/C`, `get_layoutA/B/C_TV`, `tile_size_mnk`, `permutation_mnk` (`with_permutation_mnk`), and the thread slice's `partition_A/B/C` and `partition_fragment_A/B/C` |
| | `include/cute/algorithm/gemm.hpp` | the serpentine order in which `gemm` visits the MMA tiles |
| `src/nvgpu/warpgroup.rs` | `include/cute/atom/mma_traits_sm90_gmma.hpp` | the SM90 shared-memory layout atoms `GMMA::Layout_{MN,K}_{INTER,SW32,SW64,SW128}_Atom` (`make_smem_layout_atom`): 8 rows of 128, 256, 512 or 1024 contiguous bits, MN-major `(c,8):(1,c)` or K-major `(8,c):(c,1)`, under `Swizzle<0..3,4,3>` on bytes |
| `src/viz.rs` | `include/cute/util/print_latex.hpp` | `print_latex` of a layout of rank at most 2 and `print_latex_tv` of a thread-value layout (the standalone TikZ document, the cell, grid and label nodes), and the colourings `TikzColor_White`, `TikzColor_BWx8` and `TikzColor_TV` |
| `src/fast_divmod.rs` | `include/cutlass/fast_math.h` | `FastDivmod` (its constructor's multiplier and shift, and `fast_divmod`'s quotient and remainder) and `find_log2` |
| `examples/sgemm.rs` | `examples/cute/tutorial/sgemm_2.cu` | the kernel of `gemm_nt` and `gemm_device`: the 128x128x8 tiles, the column-major shared-memory tiles, the 128-bit `UniversalCopy` tiled copies of 32x8 threads by 4x1 values, the 16x16x1 `UniversalFMA` tiled MMA, the `local_tile` projections, the partitions, the accumulators and the alpha/beta epilogue. Its main loop copies straight into shared memory, as `sgemm_1.cu`'s does, without `sgemm_2.cu`'s register prefetch |
| `examples/elementwise_add.rs` | `examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py` | the tiled kernel (`elementwise_add_kernel` and `elementwise_add`): the 4x32 thread and 4x4 value layouts, `make_layout_tv`, the 128-bit tiled copy, the partitions, the register fragments, and the identity tensor and `elem_less` predicate for a partial tile; and the dynamic kernel, which ports the example whole: `mark_layout_dynamic()` on the host tensors, the ordered thread and value layouts, the `zipped_divide` of every tensor and of the identity tensor over `mC.shape` by the TV tiler, the block slice by the linear block index, the element-wise tiled copies, the per-element `elem_less` predicate against the runtime shape, the TensorSSA `a + b`, and the grid `size(gC, mode=[1])` |
| | `examples/python/CuTeDSL/cute/notebooks/elementwise_add.ipynb` | the naive kernel follows its first kernel: one element per thread, in blocks of 256 threads, at the row and column of the global thread index |
| `examples/tensorop_gemm.rs` | `examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py` | follows its design, with its 128x128x32 tile, 2x2x1 warps of 16x8x16 F16/BF16 MMAs with F32 accumulators, `cp.async` tiled copies of 32x4 threads by 1x8 values, 8x32 row-major shared-memory atom swizzled with `Swizzle<2,3,3>` (its `make_swizzle(2, 4, 3)` counts bytes), `ldmatrix` copies made from the MMA, and multistage pipeline (`stages - 1` tiles in flight, then `cp_async_wait_group(stages - 2)` and a barrier for each tile); and its FP8 path: the Float8E4M3FN and Float8E5M2 inputs, 128x128x64 tile, 16x8x32 `MmaFP8Op` with `permutation_mnk` (32, 32, 32), `LdMatrix8x16x8bOp(False, 4)` for K-major A and B, the shared-memory atom and swizzle of `_make_smem_layout_AB` (8 rows of `min(BK, 128 bytes)`, swizzle bits `log2(row bits / 128)` capped at 3, base and shift `log2(128 / width)` elements), and the conversion of the accumulators to FP8 C. It uses 3 stages instead of 4, and has no register pipelining, predication or shared-memory epilogue |
| `examples/ampere_sgemm.rs` | `examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/sgemm.py` | the kernel and its configuration (`SGemm.__init__`, `__call__` and `kernel`): the 128x128x8 tile, 3 stages and 256 threads; the padded shared-memory layouts; the `cp.async` tiled copies, 4-wide or scalar by `max_alignment` of mode 0 for an M- or N-major operand and scalar for a K-major one; the 16x16x1 `MmaUniversalOp` atom layout for either major mode of C with its 4-element `permutation_mnk`; the `local_tile` projections, the K residue and `domain_offset`, the identity tensors and the `elem_less` predicates with and without the residue; the multistage pipeline with its read and write indices, the cleared predicates for fewer tiles than stages, the named barrier and the `unroll_full` k-block loop with its register prefetch; the epilogue predicated with the M and N residues; and from `run`, the marking of the host tensors (`mark_layout_dynamic(leading_dim)` and `mark_compact_shape_dynamic(mode, divisibility)`), the major modes `a_major`, `b_major` and `c_major`, and the grid `ceil_div(mC.shape, (bM, bN))`. The divisibility of each leading extent, the inputs and the bit-exact reference are cute-rs's own |
| `examples/runtime_gemm.rs` | `examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py` | the design that `examples/tensorop_gemm.rs` follows, whose kernel it repeats with problem sizes known only at launch |
| `examples/persistent_gemm.rs` | `examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py` | the design that `examples/tensorop_gemm.rs` follows, whose mainloop it runs on each tile a persistent tile scheduler hands out |
| `examples/block_reduce.rs` | `examples/python/CuTeDSL/experimental/primitives/reduction/block_smem_reduce.py` | the kernel (`_kernel`) and its helpers: the 4-row tiles of 128 threads, the per-lane fold, `redux.sync` for Int32 and the butterfly shuffle tree otherwise, the shared-memory `red` (or `atomic_fmax`/`atomic_fmin`, or the warp-result combine for Float16 and BFloat16 max and min), the identities, and the input data of `run` |
| `examples/redux_sync.rs` | `examples/python/CuTeDSL/experimental/primitives/redux_sync.py` | kernels 1 to 4 (`integer_reductions_kernel`, `float_reductions_kernel`, `redux_vs_shuffle_kernel`, `absmax_quantize_kernel` with its `_FP8_MAX`, guard and steps), the input data of kernels 1 to 3, the shape of kernel 4's (`_DEFAULT_NUM_ROWS` rows of 32) and the tolerances of `verify_absmax_quantize`; the shuffle-tree reference of kernel 1, kernel 4's shuffle-tree fallback and its normal inputs are cute-rs's own |
| `examples/fp4_x2_tensor.rs` | `examples/python/CuTeDSL/cute/tvm_ffi/fp4_x2_tensor.py` | the kernel's two tensors (a Float4E2M1FN tensor and the Uint8 tensor holding the same `float4_e2m1fn_x2` bytes, recast to Float4E2M1FN) and the 16 x 16 random bytes of `torch_float4_x2_tensor`; the decoding to Float16 is cute-rs's own |
| `examples/elementwise_add_autotune.rs` | `examples/python/CuTeDSL/cute/ampere/tutorial/elementwise_add_autotune.py` | the kernel (`elementwise_add_kernel`) and host function (`elementwise_add_autotune`): the 4x32 ordered thread layout and the value layout of 4 rows of `copy_bits / 32` elements, `make_layout_tv`, the `zipped_divide` of the tensors and of the identity tensor by the TV tiler, the block slice, the tiled copies, the per-element `elem_less` predicate and the TensorSSA sum, and the grid `size(gC, mode=[1])`; the tuned `copy_bits` of 64 and 128 and the unimplementable 1 (`tune_class`, `can_implement`), `update_on_change=["M", "N"]`, and the problem sizes 1024 x 512 and 1024 x 1024. The other thread layouts, the 32-bit candidate, the alignment and divisibility of the specs and the simulator checks are cute-rs's own |
| | `examples/python/CuTeDSL/cute/notebooks/benchmark_autotune.ipynb` | the 1024 x 1024 problem and the memory-throughput report |
| `examples/tvm_ffi_add.rs`, `examples/tvm_ffi/jit_and_use_in_torch.py`, `examples/tvm_ffi/error_reporting.py` | `examples/python/CuTeDSL/dsl_tutorials/tvm_ffi/jit_and_use_in_torch.py` and `error_reporting.py` | the kernel (`device_add_one`, `add_one`: one thread adding 1 to each element of `a` into `b`), its launch, the layouts it is compiled for (dynamic, and the static `(10,)`), and the scripts' steps: `b = a + 1` on 10-element Float32 torch tensors and again on `a + 1`, and the call with a 5-element tensor that fails. The shared extent of `a` and `b`, the further error cases and the C harness are cute-rs's own |
| `tests/gpu_benchmark.rs` | `examples/python/CuTeDSL/cute/notebooks/benchmark_autotune.ipynb` | the walkthrough's steps: autotune the element-wise add over `copy_bits` with `update_on_change=["M", "N"]`, tune it with `tune`, and benchmark it on 1024 x 1024 with a workspace generator, 10 warmup and 100 timed iterations, reporting the memory throughput |
| `examples/red.rs` | `examples/python/CuTeDSL/experimental/primitives/red.py` | the kernel, its launch and the input data and initialised outputs of `run` |
| `examples/warp_named_barrier.rs` | `examples/python/CuTeDSL/experimental/primitives/warp_named_barrier.py` | the kernel (`kernel`): the three warps, the full/empty mbarrier ring with its pre-signalled empty barriers and `(k // STAGES) & 1` parity, the producer's flush on barrier 0 over 32 threads, the consumer's `bar.arrive` and the epilogue's `bar.sync` on barrier 1 over 64 threads; its launch and the input data of `run`. The SM80/SM89 form (`mbarrier_test_wait` spins, no init fence) is cute-rs's own |
| `examples/pipeline_software_fill.rs` | `examples/python/CuTeDSL/experimental/primitives/pipeline_software_fill.py` | the kernel (`kernel`): the producer and consumer warps, the full/empty mbarrier arrays with the pre-signal trick, the producer's flush and elected arrivals; its launch and the input data of `run`. The SM80/SM89 form is cute-rs's own |
| `examples/cp_async_mbarrier_arrive.rs` | `examples/python/CuTeDSL/experimental/primitives/cp_async_mbarrier_arrive_cta.py` | the kernel (`kernel`): the producer warp's 16-byte `cp.async.cg` per lane and `cp.async.mbarrier.arrive.noinc`, the mbarrier of 32 arrivals and the consumer warp's wait and copy out; its launch and the input data of `run` |
| | `examples/python/CuTeDSL/experimental/primitives/cp_async_mbarrier_arrive_cluster.py` | the kernel (`kernel`): the cluster of two CTAs with producer, forwarder and consumer warps, the local, leader and empty mbarriers and their counts, the inits with `fence.mbarrier_init` and a relaxed cluster barrier, the forwarder's elected cluster-scope arrival on CTA 0's leader mbarrier, and CTA 0's release of both empty mbarriers; and its launch. Its peer shared-memory read is replaced by each CTA writing its own tile, which is cute-rs's own |
| `examples/cooperative_launch.rs` | `examples/python/CuTeDSL/dsl_tutorials/cooperative_launch.py` | the device side of `GlobalBarrier` (the phase bit over an arrival counter, `arrive` with its release adds, `wait` with its acquire spin, `arrive_and_wait`), the kernel's ten crossings (`cooperative_kernel`) and the grid and block of `run_cooperative_kernel`. The per-crossing counts that check the barrier are cute-rs's own |
| `examples/programmatic_dependent_launch.rs` | `examples/python/CuTeDSL/dsl_tutorials/programmatic_dependent_launch.py` | the kernel (`elementwise_add_kernel`: the tiled copies, the ten-fold loads, `griddepcontrol_launch_dependents` in the first kernel and the second's loads of B before `griddepcontrol_wait` and of A after it), the host function `elementwise_add` (the 4x32 by 4x4 TV layout, the `zipped_divide`s, the launch with `use_pdl`) and the two launches of `run_pdl_example`. The simulated stream that checks the pair is cute-rs's own |
| `examples/call_bypass_dlpack.rs` | `examples/python/CuTeDSL/dsl_tutorials/call_bypass_dlpack.py` | the wrapper `tensor_op_gemm_wrapper` (the pointer parameters, `m` and `n` assumed multiples of 8, the ordered layouts of A, B and C over the pointers) and the memory orders and 32-byte-aligned `make_ptr` pointers of `run_tensor_op_gemm_wrapper`. The one-element-per-thread GEMM standing in for `TensorOpGemm` is cute-rs's own |
| `examples/elementwise_apply.rs` | `examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_apply.py` | the kernel (`elementwise_apply_kernel` and `elementwise_apply`): the 4x64 thread layout, the 16 x 16-byte value layout recast to the element type, `make_layout_tv`, the `zipped_divide` of every tensor composed with the block remap (`remap_block`), the block and thread slices, the identity-tensor predicate it leaves unused, and the whole-tensor load, operation and store; the operations of its test (`add`, `sub`, `mul`, `truediv`, `floordiv` with zeros replaced by 1.2, `leaky_relu` with alpha 0.01). The sizes and input data are cute-rs's own |
| `examples/cta_norm.rs` | `examples/python/CuTeDSL/cute/hopper/kernel/cta_norm/cta_norm.py` | the kernel (`CtaNorm.kernel`, `warp_reduce`, `cta_reduce`, `apply_layernorm`, `apply_rmsnorm`): one CTA per row, the 128-bit tiled copy of 8 elements per thread over a linear thread layout, the row tile and partitions, the per-chunk predicate, the Float32 statistics with the butterfly warp reduction and the shared-memory CTA reduction, `rsqrt(var / N + eps)`, and the TensorSSA epilogues. The sizes, the input data and the tolerances are cute-rs's own |
| `tests/support/printf_cases.rs`, `tests/printf.rs` | `examples/python/CuTeDSL/notebooks/1_dsl_features/01_hello_world.ipynb` | the device printing of `hello_kernel`: thread 0's greeting with its block and thread (its format string, and CuTe DSL's default `end="\n"`), and the elementwise add with its tail guard |
| `tests/gpu.rs` | `examples/python/CuTeDSL/experimental/compiler_diagnostic/ptxas_spill_cases.py` | its two kernels: `_global_reverse_register_spill_kernel` (64 values per thread loaded, a CTA barrier, stored reversed; 32 threads) and `_dynamic_index_local_memory_kernel` (a 256-entry per-thread array filled from memory, a CTA barrier, then indexed by a loaded value masked to 255); compiled with `--maxrregcount=16` as `show_remark.py` compiles for spills |
| | `examples/python/CuTeDSL/experimental/compiler_diagnostic/cuda_launch_error.py` | its two launch errors of a one-element copy kernel: a block of 4096 threads and 1 GiB (1073741824 bytes) of dynamic shared memory |
| `examples/cute_ext_memcpy.rs` | `examples/python/CuTeDSL/cute_ext/ampere/memcpy_simt_universal_copy.py` | the kernel (`memcpy_simt_universal_copy_kernel`): the 128 x 64 `zipped_divide` tiles and the block slice, the ((8,1),(1,8)) register buffer, the 16 x 8 by 8 TV layout over a 128 x 8 tiler, the 128-bit universal copies in and out and `buffer + addend`; the host function's grid (the rest modes of `tiled_divide`) and 128-thread blocks; the M-contiguous dynamic tensors marked with `mark_layout_dynamic(leading_dim=0)` and `mark_compact_shape_dynamic(mode=0, divisibility=8)` at 16-byte alignment, the addend 5.0 and the default shape (136, 7, 9). The input data and the other sizes are cute-rs's own |
| `examples/cute_ext_sgemm.rs` | `examples/python/CuTeDSL/cute_ext/ampere/sgemm.py` | the kernel (`sgemm_kernel`): the 16 x 8, 8 x 8 and 16 x 8 `zipped_divide` tiles and block slices, the shared and register buffers (`partition_shape_A/B/C` of the 16x8x8 `MmaF16BF16Op` tiled MMA), the 32-bit `CopyG2SOp` tiled copies with their thread and value layouts, the `partition`s by `layout_dst_tv_tiled`/`layout_src_tv_tiled`, the K loop with `predicated_tensor_origin`, the copies, the `cp.async` commit and wait and the barriers, `partition_and_copy` with `LdMatrix8x8x16bOp` (one matrix) tiled copies for A and B, `dot`, and the universal-copy epilogue through `make_tiled_copy_C`; the host function's grid and one-warp blocks; K-major A and B, N-major Float32 D. The epilogue copies 32 bits instead of 16, the static layouts replace `iterator.align(4)`, and the problem sizes, input data and checks are cute-rs's own |
| `examples/cute_ext_dense_gemm_persistent.rs` | `examples/python/CuTeDSL/cute_ext/blackwell/dense_gemm/dense_gemm_persistent.py` | the class `PersistentDenseGemmKernel`: its configuration (`mn_tiler`, `use_2cta_instrs`, `cluster_shape_mn`, `fallback_cluster_shape_mn`, the mainloop, accumulator and TMA-store stages, `swizzle_size`, `raster_along_m`) and the constructor's checks; `__call__`'s CTA tile, scheduler params and grid, 192-thread blocks, cluster and fallback cluster; `kernel`'s tiled MMA (`make_trivial_tiled_mma`), `mma_inst_tile_k = 4` and MNK tiler, the shared and TMEM buffers (`make_smem_layout_a/b`, `make_tmem_layout_acc`, `allocate`), the cluster layout and leader CTA, the operation-type selection (1SM/2SM, multicast or not), the `TMAToUMMAPipeline`, `UMMAtoAsyncPipeline` (256 or 128 consumer arrivals) and `TMAStorePipeline` (2 stages, 128 threads, barrier 1, warp 0), the warp roles (epilogue 0-3, MMA 4, TMA 5), the `local_tile`/`partition_A/B` views, the TMA warp's loop (`producer_acquire_and_get_stage`, `get_mbarrier`, `tma_load` with multicast modes 2 and 1, `producer_commit_and_advance`), the MMA warp's loop (`mainloop_mma`) and the epilogue warps' loop (`epilogue_tma_store`, `consumer_release_and_advance`), with their names; and `compare`'s reference GEMM. The tails, the per-shape arms, the problem sizes, data and checks are cute-rs's own |
| `examples/cuda_graphs.rs` | `examples/python/CuTeDSL/cute/notebooks/cuda_graphs.ipynb` | its flow: a one-thread kernel launched on a stream, a graph of two launches captured and replayed, then a graph of 100 launches timed against 100 launches on the stream, with warm-up runs, and the speedup printed. The kernel counts its runs with an atomic instead of printing, and the host clock replaces CUDA events |
| `examples/export_to_c.rs` | `examples/python/CuTeDSL/dsl_tutorials/export/export_to_c.py` | its flow: two kernels, one printing a tensor and one setting `a[0] = b[0] + 1`, on tensors whose first extent is a launch-time multiple of 16 and whose second is any launch-time size, each exported under a file name and prefix, and the object files linked into `libexport_example.so`. The print kernel prints the layout and first element instead of the whole tensor |
| `examples/load_exported.rs` | `examples/python/CuTeDSL/dsl_tutorials/export/load_in_python.py` | its flow: the two exports loaded back and launched on device tensors, and the check that `a[0] == b[0] + 1`. It loads the manifests rather than the object files or the shared library |
| `tests/abi_kernels/mod.rs`, `tests/launch_abi.rs`, `tests/gpu_launch.rs` | `examples/python/CuTeDSL/notebooks/2_primitives/01_array_concepts.ipynb` | the module-global example: the constant table `gamma` (1.0, 1.5, 2.0, 2.5), the global counter `run_count`, `scale_kernel` (the input scaled by the table, the counter bumped by thread 0 and reported in `seen`) with its inputs and expected outputs, and the counter climbing across launches |
| | `examples/python/CuTeDSL/dsl_tutorials/dynamic_smem_size.py` | `kernel_static`'s shared allocations (a struct of 64 Float32, an Int32 and an Int8; 512 bytes aligned to 64; 128 Int32; a 32x16 Float16 tile) and its launches with the size inferred and with `smem=16`, below the allocations |
| `tests/prims_nvvm.rs` | `examples/python/CuTeDSL/experimental/primitives/barrier_cta_red.py` | the kernel (`kernel`): the three predicate reductions on named barriers 0, 1 and 2 (`tidx == 0` with or, `tidx < THREADS` and `tidx != 0` with and), the thread counts 32, 64 and 128 and the expected `[1, 1, 0]` |
| | `examples/python/CuTeDSL/experimental/primitives/mbarrier_lifecycle.py` | the kernel (`kernel`): init, the init fence and a CTA barrier, an arrival's state token waited on with a time limit of 10,000,000, the 1/0 result and the inval; the re-initialisation after the inval is cute-rs's own |
| `tests/feature_kernels/vector.rs` | `examples/python/CuTeDSL/experimental/primitives/basic_types.py` | the kernels `vector_2d_slice`, `vector_scalar_broadcast` and `vector_outerproduct` and their expected values |
| | `examples/python/CuTeDSL/notebooks/2_primitives/02_vector_concepts.ipynb` | its vector operations: `where(u > zeros, u, zeros)`, the square, the absolute value, `vec * A + B` and the four reductions |
| | `examples/python/CuTeDSL/experimental/primitives/reduction/thread_vector_reduce.py` | the reductions of a thread's vector with add, max and min over Float16, BFloat16, Float32 and Int32 |
| `tests/feature_kernels/masked.rs` | `examples/python/CuTeDSL/experimental/primitives/basic_types.py` | the `masked_load_store` kernel (the all-true load mask, the `[T, F, T, F]` store mask, the `+ 10` and the preset -5) and its expected values |
| `tests/feature_kernels/mod.rs` | `cutlass_compiler/cutegen/include/cutegen/layout.hpp` | the reference `increment` of the `increment_coords` case (`detail::increment_coord_impl`) |
| `tests/feature_kernels/structs.rs` | `examples/python/CuTeDSL/cute/notebooks/async_pipeline.ipynb` | the shape of its `SharedStorage` struct: `tma_mbar_ptr`, a `MemRange` of Int64 mbarriers, and `staging_buffer`, a Float32 `MemRange` under `Align[.., 1024]`. The hand-off kernel, its stage count and its data are cute-rs's own |
| `tests/feature_kernels/bulk.rs` | `examples/python/CuTeDSL/experimental/primitives/cp_async_bulk.py` | the kernel (`kernel`) of the `cp_async_bulk_round_trip` case: the producer and consumer warps with elected lanes, the two-stage ring with full and empty mbarriers (the empty ones pre-arrived, parity `(k // NUM_STAGES) & 1`), the bulk load completing on `full`, and the bulk store with `cp_async_bulk_commit_group` and `cp_async_bulk_wait_group(0)` before the `empty` arrival. Its tile, stage and block counts are cute-rs's own |
| `tests/feature_kernels/cvt_ldmatrix.rs` | `include/cute/atom/copy_traits_sm100.hpp` | the destination layout of `SM100_U8x8_LDSM_T` and `SM100_U8x16_LDSM_T` (in bits), from which the `LdMatrix16x8x8b` check reads its expected values |
| `src/params/tensor.rs` | `media/docs/pythonDSL/guides/framework_integration.rst` | the rules of `mark_layout_dynamic` (leading-dimension deduction, kept unit and zero strides) and `mark_compact_shape_dynamic` (stride-order deduction and checks, propagated divisibility, size-1 strides canonicalised to 0), and the wording of their error messages |
| `tests/kernel_params.rs` | `examples/python/CuTeDSL/dsl_tutorials/dataclass_immutable.py`, `examples/python/CuTeDSL/experimental/primitives/tutorial/08_mlir_value_tree_debug.py` | the memcpy kernel and its frozen-dataclass parameters (`CopyParams`: `num_threads`, `src`, `dst`, over one `sym_int64` extent), and the `Coord2D`, `Particle` and `Accumulator` structs carried through loops |

`examples/run_on_gpu.rs` and some tests compile the example files as they
are, with their headers.

Some tests also use data from CUTLASS 4.8.1, as test vectors only, and a
comment at each says where it comes from:

* `src/layout/tests.rs`: the inputs and expected results of
  `TileToShapeTest.Basic`, `TupleTest.Congruence`, `TupleTest.WeakCongruence`,
  `LayoutTest.IncrementCoord`, `LayoutTest.MaxAlignment`,
  `LayoutTest.MakeLayoutLike`, `ComposedLayoutTest.MakeLayoutLike`, the
  `ToFromString`, `SliceStaticSwizzledLayouts`, `SliceAffineLayouts`, `Get`,
  `Select`, `Upcast`, `Downcast` and `CoSize` cases of `ComposedLayoutTest`, the
  `max_alignment` cases of `SwizzleTest`, the `ScaledBasisTest` cases, the `RatioTest`
  cases, the basis-stride cases of `CoalesceTest`, `CompositionTest` and
  `ComplementTest`, the no-cotarget cases of `ComplementTest.Basic`, the
  basis-stride cases of `LogicalDivideTest` and the `Coordinate` cases of
  `LeftInverseTest` and `RightInverseTest` (in `cg_tile_to_shape_test.cpp`,
  `cg_tuple_test.cpp`, `cg_layout_test.cpp`, `cg_composed_layout_test.cpp`,
  `cg_swizzle_test.cpp`, `cg_scaled_basis_test.cpp`,
  `cg_ratio_test.cpp`, `cg_coalesce_test.cpp`, `cg_composition_test.cpp`,
  `cg_complement_test.cpp`, `cg_logical_divide_test.cpp`,
  `cg_left_inverse_test.cpp` and `cg_right_inverse_test.cpp` of
  `cutlass_compiler/cutegen/test`);
  the layouts of `test/unit/cute/core/nullspace.cpp`; the `make_layout_like`
  example in the comment above it in `include/cute/layout.hpp`; the examples of
  `media/docs/cutlass_compiler/cute_dialect.rst`; the worked examples of
  `media/docs/cpp/cute/02_layout_algebra.md` (coalesce, composition,
  complement, divide and product); the divide and product results printed in
  `examples/python/CuTeDSL/cute/notebooks/cute_layout_algebra.ipynb`; and the
  thread and value layouts of `elementwise_add.py` (above). A few doc
  examples in `src/layout/layout.rs` use layouts from `cute_dialect.rst`.
* `tests/dynamic_tensors.rs` and `tests/diagnostics.rs`: the `elem_less`,
  `ceil_div` and `shape_div` examples and error cases of
  `media/docs/cutlass_compiler/cute_dialect.rst`.
* `tests/kernel_params.rs`: the `mark_layout_dynamic` cases t0 to t8 and
  `mark_compact_shape_dynamic` cases t0 to t13 (layouts, printed results and
  error messages) and the other examples of
  `media/docs/pythonDSL/guides/framework_integration.rst`, and the five
  fake-tensor specs of `examples/python/CuTeDSL/dsl_tutorials/tvm_ffi/compile_with_fake_tensor.py`.
* `tests/atoms.rs`: the `SM75_U32x4_LDSM_N` layouts of
  `include/cute/atom/copy_traits_sm75.hpp` and the one-thread `ThrID` of
  `UniversalCopy` in `include/cute/atom/copy_traits.hpp`, the tables
  `src/atom/copy.rs` ports.

### stmatrix atoms (m8-stmatrix-setmaxnreg)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/atom/copy.rs` | `include/cute/atom/copy_traits_sm90.hpp` | the `stmatrix` source, destination and reference layouts of `SM90_U32x1/x2/x4_STSM_N` and `SM90_U16x2/x4/x8_STSM_T` (the `SM75` `ldmatrix` layouts with source and destination swapped), in 16-bit elements |
| | `include/cute/atom/copy_traits_sm100.hpp` | the `stmatrix` source, destination and reference layouts of `SM100_U8x4/x8/x16_STSM_T`, converted from bits to 8-bit elements |
| `tests/feature_kernels/stmatrix.rs` | `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py` and `dense_gemm_persistent.py` | follows their design: the register-to-shared epilogue copy made with `make_tiled_copy_C_atom` of `StMatrix8x8x16bOp` (four matrices) and `make_tiled_copy_S`, and the warp-specialised register split (the producer warpgroup at 40 registers, the consumers at 232) |

### Warpgroup MMA (m8-wgmma-core)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/nvgpu/warpgroup.rs` | `include/cute/atom/mma_traits_sm90_gmma.hpp` | `make_gmma_desc`: the canonical K-major and MN-major layouts per layout type (`INTERLEAVE`, `B32`, `B64`, `B128`), the checks of a tile against them, which strides become the leading and stride byte offsets (swapped between the interleaved and swizzled MN-major forms; 1 for the swizzled K-major leading offset), base offset 0, and the per-element advance of `DescriptorIterator` in 16-byte units |
| | `include/cute/arch/mma_sm90_desc.hpp` | the `GmmaDescriptor` bit fields (start address bits 0-13, leading byte offset 16-29, stride byte offset 32-45, base offset 49-51, layout type 62-63) and the `LayoutType` values |
| `src/atom/mma.rs` | `include/cute/atom/mma_traits_sm90_gmma.hpp` | the `SM90_64xNx16_*_SS` traits' `ABLayout<64, 16>`, `ABLayout<N, 16>` and `CLayout_64xN` thread-value layouts over a 128-thread warpgroup |
| `src/backend/mlir/wgmma.rs` | `include/cute/atom/mma_traits_sm90_gmma.hpp` | `warpgroup_fence_operand`: an empty side-effecting statement that reads and writes each accumulator register (Float16 accumulators as 32-bit pairs) |
| `tests/feature_kernels/wgmma.rs` | `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py` | follows its design: the main loop that issues `k_pipe_mmas` k-tiles of MMAs before the loop, then fences, issues, commits and waits with `wait_group(k_pipe_mmas)` before releasing the stage, with `ACCUMULATE` false before the first k-block; and the shared-memory layout orders (K-major tiled along M/N first, MN-major along K first) |

### WGMMA variants and distributed shared memory (m8-wgmma-variants, m8-dsmem)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/atom/mma.rs` | `include/cute/atom/mma_traits_sm90_gmma.hpp` | the register-A traits' (`SM90_64xNxK_*_RS`) `ALayout_64x16` and `ALayout_64x32` thread-value layouts, and the K = 32 `ABLayout` of the 8-bit types |
| `tests/feature_kernels/dsmem.rs` | `examples/python/CuTeDSL/experimental/primitives/mapa.py` and `experimental/primitives/reduction/cluster_smem_reduce.py` | follows their design: a ring of CTAs writing and reading each other's shared memory through `mapa` between cluster barriers, and a reduction with `red` on the root CTA's shared memory between non-relaxed cluster barriers |

`mapa.py` and `cluster_smem_reduce.py` carry
`Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`; they and
the `include/cute` files are under the BSD-3-Clause licence below.

### TMA atoms (m7-tma-atom)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/nvgpu/tma_atom.rs` | `include/cute/atom/copy_traits_sm90_tma.hpp` | `make_tma_atom` and `make_tma_copy_atom` (the CTA tile's map `make_identity_layout(shape(gtensor)).compose(cta_tiler)`, the one-thread atom moving the whole box), `construct_tma_gbasis` (inverting the SMEM layout, keeping the leading unit-step runs, `coalesce_256`, appending the missing modes with extent 1, grouping past rank 5), `fill_tma_gmem_shape_stride` (the gcd recurrence for folded modes), the box, multicast split, TMA basis strides and L2 promotion of `make_tma_copy_desc`, and `tma_partition` (the SMEM run tiled over the tile, divided by the box, offset by the CTA's multicast share) |
| | `include/cute/atom/copy_traits_sm90_tma_swizzle.hpp` | `get_tma_swizzle_bits` for `Swizzle<B,4,3>` (B 0 to 3) |

### TMA stores and cluster copies (m7-tma-store, m7-cluster-copies)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/nvgpu/cpasync.rs` | `include/cute/arch/copy_sm90_desc.hpp` | the three 64-bit L2 cache-policy values of `TMA::CacheHintSm90` (`EVICT_NORMAL`, `EVICT_FIRST`, `EVICT_LAST`) |
| `tests/feature_kernels/tma_store.rs`, `tests/feature_kernels/cluster_copies.rs` | `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py` | follows its design: the TMA-store epilogue (shared-memory writes, `fence.proxy.async`, a named barrier, one warp storing and waiting `.read` before reusing a stage) and the multicast of A along N and B along M with masks from the cluster layout |
| `tests/feature_kernels/tma_pipelines.rs`, `tests/feature_kernels/gemm_helpers.rs` | `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py` | follows its design: the TMA mainloop pipeline with multicast A/B loads and its consumer arrival count (`mcast_size * num_warps`), the TMA-store epilogue through a staging ring, and the accumulator-to-shared-memory epilogue store |
| `tests/feature_kernels/cluster_copies.rs` | `examples/python/CuTeDSL/cute/blackwell/kernel/reduce/reduce.py` | follows the design of `cluster_reduce`: each CTA stores its partial into every CTA's shared memory with `st.async`, and each waits on its own mbarrier armed with the cluster's bytes |

`dense_gemm.py` and `reduce.py` carry
`Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`,
`copy_sm90_desc.hpp` the `include/cute` line below; all three are under the
BSD-3-Clause licence below.

### Im2col and the SM100 TMA modes (m7-tma-im2col, m7-tma-sm100-modes)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/nvgpu/tma_im2col.rs` | `include/cute/atom/copy_traits_sm90_im2col.hpp` | `make_tma_atom_im2col` (the box's channels and pixels from the SMEM layout's first two contiguous runs over the `(pixels, channels)` tile, the multicast split of the pixels) and `make_im2col_tma_copy_desc` (the descriptor's `(C, W, H, D, N)` extents and byte strides, traversal strides from `stride_whd`, the corners, and the GEMM extents `q/p/z = (w/h/d + upper corner - lower corner - 1) / stride + 1` and `s/r/t = 1 + (upper padding - upper corner) / stride_srt`, with the coordinate offsets `lower_corner_whd` and `lower_srt`) |
| `src/nvgpu/tma_atom.rs` | `include/cute/arch/copy_sm90_desc.hpp` | the FP4/FP6 cases of `TMA::to_CUtensorMapDataType` (`16U4_ALIGN8B` for FP4, `16U6_ALIGN16B` for FP6, the `16U4_ALIGN16B`/`16U6_ALIGN16B` unpacking forms) in `resolve_internal_type` |

Both files carry the `include/cute` copyright line below and are under the BSD-3-Clause
licence below.

### Async pipelines and grouped GEMM schedulers (m9-pipeline-cp-async-order, m9-grouped-gemm-schedulers)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `examples/async_pipeline.rs` | `examples/python/CuTeDSL/cute/notebooks/async_pipeline.ipynb` | the kernels `synced_producer_consumer`, `async_pipeline_kernel` and `async_pipeline_staged_kernel`, their `SharedStorage` structs and launches, and the notebook's host data and outputs |
| `src/utils/grouped_tile_scheduler.rs` | `examples/python/CuTeDSL/helpers/grouped_gemm_persistent_tile_scheduler.py` | the warp-wide group search of `_group_search` and `_prefix_sum` (the 32-group window, the scan, ballot and broadcasts), the double-buffered problem shapes of `prefetch_problem_shapes`, and the tile decode of `delinearize_z` and `_compute_cta_tile_coord` |

The notebook has no header of its own and is covered by CUTLASS's top-level
`LICENSE.txt` (`Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`);
`helpers/grouped_gemm_persistent_tile_scheduler.py` carries
`Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
Both are under the BSD-3-Clause licence below.

### Several kernels in one module (m6b-multi-kernel)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `examples/split_kv_decode.rs` | `examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp16.py` | the host function's two launches from one compiled function (the split-KV kernel, then the reduction), and `reduction_kernel`: the first warp's global log-sum-exp over the splits (warp max, `-inf` guard, warp sum of `exp2`, `log2`), the per-split scales `exp2(lse_split - lse)` in shared memory, the barrier, and the per-thread sum of the splits' partial outputs with those scales. The split kernel (a SIMT online softmax), the sizes and the data are cute-rs's own |

`mla_decode_fp16.py` carries
`Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
and is under the BSD-3-Clause licence below.

### In-kernel tensor-map updates and the TMA tutorials (m7-tensormap-update, m7-port-tma-tutorials)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `tests/feature_kernels/tensormap.rs` | `examples/python/CuTeDSL/experimental/primitives/tma/tma_tensormap_replace.py` | the descriptor-replacement kernel: copy a descriptor to a global workspace, replace its global address with a later row and its second extent with the active rows, release, acquire and TMA-load the box (the rows past the active ones zero-filled) |
| `examples/tma_copy.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v0.py` | the kernel `Sm100SimpleCopyKernel` (its configuration, atoms, barrier protocol, `tma_partition` calls and indexing), its comments and `tma_partition` diagrams, and the default problem of `run_tma_copy` |
| `examples/tma_transpose.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v1.py` | the kernel `Sm100MatrixTransposeKernelV1` (its warp roles, named barriers, mbarrier hand-offs, transposed destination view, shared-memory layouts and universal tiled copy), its comments, and the default problem of `run_transpose` |

`tma_tensormap_replace.py` carries
`Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`,
`tma_v0.py` `Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
and `tma_v1.py` `Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`;
all three are under the BSD-3-Clause licence below.

### Cluster RMSNorm (m8-port-rmsnorm-cluster)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `examples/rmsnorm_cluster.rs` | `examples/python/CuTeDSL/cute/blackwell/kernel/rmsnorm/rmsnorm.py` | `RMSNormConfig` (the cluster-size, threads-per-row and thread-count tables, the TV layout and the shared-memory size), the kernel of `RMSNormKernel` (shared-memory allocation, cluster initialisation, identity-tensor tiles and partitions, the `cp.async`, weight and store tiled copies, `predicate_k`, the two passes and the cluster barrier between them), `rmsnorm_ref`, and their comments |
| `examples/rmsnorm_cluster.rs` | `examples/python/CuTeDSL/cute/blackwell/kernel/reduce/reduce.py` | `row_reduce`, `block_reduce`, `cluster_reduce` (the hierarchical buffer, the expected transaction bytes, the `st.async` of each warp's partial to every CTA and the final warp reduction), `store_shared_remote`, `elem_pointer`, and their comments |

`rmsnorm.py` and `reduce.py` both carry
`Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
and are under the BSD-3-Clause licence below. (`reduce.py` notes that its
cluster primitives are inspired by Quack, https://github.com/Dao-AILab/quack;
no Quack code is ported.)

### Hopper dense GEMM (m8-port-hopper)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `examples/hopper_gemm.rs` | `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py` | `HopperWgmmaGemmKernel`: its configuration (`__init__`, `_setup_attributes`: the warpgroup layout, the tile's K, the multicast CTAs, the epilogue tile), `__call__` (the type checks, the TMA load and store atoms, the grid and the launch), the `kernel` (descriptor prefetch, the CTA swizzle, the multicast masks, the mainloop pipeline's arrival counts and protocol, the TMA partitions, the descriptor fragments, the prologue MMAs and the main loop with `k_pipe_mmas`, the cluster barrier, the `stmatrix` epilogue through `make_tiled_copy_C_atom` and `make_tiled_copy_S`, the proxy fence, named barriers and TMA stores in bulk groups), `_compute_stages`, `_make_smem_layouts`, `_compute_grid`, `_make_tma_store_atoms_and_tensors`, `_make_tma_atoms_and_tensors`, `is_valid_dtypes`, `is_valid_tensor_alignment`, the tensor set-up and reference of `run`, and their comments |
| `tests/hopper_gemm.rs`, `tests/mlir.rs` (`hopper_gemm_in_ptx`) | the same | nothing beyond what they call in `examples/hopper_gemm.rs` |

`dense_gemm.py` carries
`Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
and is under the BSD-3-Clause licence below.

### Blackwell layouts and UMMA descriptors (m10-umma-layouts)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/nvgpu/tcgen05/layouts.rs` | `include/cute/atom/mma_traits_sm100.hpp` | `Layout_MN_SW128_32B_Atom`, the `*_SpAtom` sparse atoms (`blocked_product` of `(1,S)` into the dense atom) and `UMMA::tile_to_mma_shape` |
| `src/nvgpu/tcgen05/smem_desc.rs` | `include/cute/arch/mma_sm100_desc.hpp` | `UMMA::SmemDescriptor`'s bit fields (and the sm_107 15-bit variant), `UMMA::LayoutType`, and `make_umma_desc`'s canonical-form checks and LBO/SBO choice |
| `src/utils/blockscaled_layout.rs` | `include/cutlass/detail/sm100_blockscaled_layout.hpp`, `include/cutlass/detail/sm103_blockscaled_layout.hpp` | the scale-factor atoms `Sm1xxBlockScaledBasicChunk`, `Sm103BlockScaledBasicChunk` and `tile_atom_to_shape_SFA/SFB` |
| `src/utils/sm100.rs`, `src/utils/epilogue_tile.rs` | `include/cutlass/epilogue/collective/builders/sm100_builder.inl` | `sm100_dense_compute_tile_shape_or_override` (automatic epilogue tile) and `sm100_tmem_warps` |
| `src/utils/sm100_epilogue.rs` | `include/cutlass/epilogue/collective/builders/sm100_builder.inl`, `include/cute/atom/copy_traits_sm100.hpp` | `sm100_get_tmem_load_op` (the TMEM load's shape, packing and repetition), `TMEM::op_repeater`, `sm100_get_smem_store_op` (the `stmatrix` variants compatible with the load) and `sm100_get_epilogue_smem_swizzle_layout_atom` |
| `tests/epilogue_helpers.rs` | `include/cutlass/epilogue/collective/builders/sm100_builder.inl`, `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py`, `fp16_gemm_1.py` | the choices worked by hand for the host tests, and the mainloop the epilogues drain (through `examples/blackwell_fp16_gemm.rs`) |

These files carry
`Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
and are under the BSD-3-Clause licence below.

### TMEM loads and stores (m10-tmem-ld-st)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/nvgpu/tcgen05/copy.rs` | `include/cute/atom/copy_traits_sm100.hpp`, `include/cute/arch/copy_sm100.hpp` | the `SM100_TMEM_LOAD_*`/`SM100_TMEM_STORE_*` copy traits' `ValID`, source, destination and reference layouts (as formulas in the shape, repetition and packing, in bits, converted to elements), the `.16x32bx2` half-split offsets, and `make_tmem_copy` |
| `tests/feature_kernels/tmem_ldst.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py` | the epilogue's split of the TMEM accumulator into subtiles with `zipped_divide` and one `make_tmem_copy` per subtile |

These files carry
`Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
(`fp16_gemm_0.py`: `Copyright (c) 2024 - 2026 ...`) and are under the BSD-3-Clause licence below.

### Mixed-input GEMM helpers (m10-mixed-input)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `src/utils/mixed_input_helpers.rs` | `examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/mixed_input_gemm.py`, `mixed_input_host_utils.py` | the kernel's use of the transform, scale and scale-TMA partitions that the helpers package (the TMEM and shared-memory store tiled copies, the source copy for the scales, the stride-0 register view of the scales), and the host's shuffle order of Int4 A |
| `examples/blackwell_mixed_input_gemm.rs` | `examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/mixed_input_gemm.py`, `mixed_input_host_utils.py` | `MixedInputGemmKernel`, reduced: its warp roles and named barriers, `_setup_attributes`, `__call__` (the TMA atoms of A, the scales and B, the transaction sizes, the shared storage) and `kernel` (the five pipelines with their arrival counts, the TMEM allocation and the transformed A's place after the accumulator, the TMA partitions, the TMA, scale-TMA, transform and MMA warps' loops), their comments and names, and the reference (A converted, scaled per row and K granule, times B). The persistent scheduler, sub-tiling, `try_wait` peeks, stage computation, the inputs and the CPU check are cute-rs's own |

These files carry
`Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
and are under the BSD-3-Clause licence below.

## Licence

The NVIDIA files named above are all under the same BSD-3-Clause licence, and
carry these copyright lines:

* `Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`:
  `include/cute` of CUTLASS 4.8.1, and `examples/cute/tutorial/sgemm_1.cu`
  and `sgemm_2.cu`;
* `Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`:
  `include/cute/layout.hpp` of CUTLASS 3.8.0;
* `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`:
  `cutlass_compiler/cutegen`, its headers and its tests, and the CuTe DSL examples `experimental/primitives/reduction/block_smem_reduce.py`,
  `experimental/primitives/redux_sync.py`, `experimental/primitives/red.py`,
  `experimental/primitives/warp_named_barrier.py`,
  `experimental/primitives/pipeline_software_fill.py`,
  `experimental/primitives/cp_async_mbarrier_arrive_cta.py`,
  `experimental/primitives/cp_async_mbarrier_arrive_cluster.py`,
  `experimental/primitives/cp_async_bulk.py`,
  `experimental/primitives/basic_types.py`,
  `experimental/primitives/reduction/thread_vector_reduce.py`,
  `experimental/primitives/tutorial/08_mlir_value_tree_debug.py`,
  `experimental/compiler_diagnostic/ptxas_spill_cases.py`,
  `experimental/compiler_diagnostic/show_remark.py` and
  `experimental/compiler_diagnostic/cuda_launch_error.py`;
* `Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`:
  the CuTe DSL examples `elementwise_add.py`, `sgemm.py`, `elementwise_apply.py`,
  `cta_norm.py`, `tensorop_gemm.py`,
  `dsl_tutorials/cooperative_launch.py`, `dsl_tutorials/programmatic_dependent_launch.py`,
  `dsl_tutorials/call_bypass_dlpack.py`, `dsl_tutorials/dataclass_immutable.py`,
  `dsl_tutorials/tvm_ffi/compile_with_fake_tensor.py`, `dsl_tutorials/export/export_to_c.py`,
  `dsl_tutorials/export/load_in_python.py`, `tvm_ffi/fp4_x2_tensor.py`,
  `ampere/tutorial/elementwise_add_autotune.py`,
  `dsl_tutorials/tvm_ffi/jit_and_use_in_torch.py` and
  `dsl_tutorials/tvm_ffi/error_reporting.py`, and the
  `README.md` of `examples/python/CuTeDSL/cute/notebooks`, the folder of the
  notebooks named above;
* `Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`:
  `include/cutlass/fast_math.h`, `test/unit/cute/core/nullspace.cpp`, `media/docs/cpp/cute/02_layout_algebra.md`,
  and CUTLASS's top-level `LICENSE.txt`, which covers the files that have no
  header of their own, such as `cute_dialect.rst`, `framework_integration.rst`
  and the notebooks.

Each of those files gives the licence after its copyright line:

```text
Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Sun Microsystems notices

The `src/math/` files port these files of FreeBSD's msun (`lib/msun/src`)
and newlib (`newlib/libm/math`), and keep their notices:

| cute-rs file | msun files | newlib files | Notices |
|---|---|---|---|
| `src/math/exp.rs` | `e_exp.c`, `e_expf.c`, `s_expm1.c`, `s_expm1f.c`, `e_sinh.c`, `e_sinhf.c`, `e_cosh.c`, `e_coshf.c`, `s_tanh.c`, `s_tanhf.c` | | A, B, C |
| `src/math/trig.rs` | `s_sin.c`, `s_cos.c`, `s_tan.c`, `k_sin.c`, `k_cos.c`, `k_tan.c`, `e_rem_pio2.c`, `k_rem_pio2.c` (its table of 2/pi) | `kf_sin.c`, `kf_cos.c`, `kf_tan.c`, `ef_rem_pio2.c`, `sf_sin.c`, `sf_cos.c`, `sf_tan.c` | A, B, D |
| `src/math/log.rs` | `e_log.c`, `e_logf.c`, `k_log.h`, `k_logf.h`, `e_log2.c`, `e_log2f.c`, `e_log10.c`, `e_log10f.c`, `s_log1p.c`, `s_log1pf.c`, `s_asinh.c`, `s_asinhf.c`, `e_acosh.c`, `e_acoshf.c`, `e_atanh.c`, `e_atanhf.c` | | A, B |
| `src/math/invtrig.rs` | `e_asin.c`, `e_acos.c`, `e_acosf.c`, `s_atan.c`, `s_atanf.c`, `e_atan2.c`, `e_atan2f.c` | `ef_asin.c` | A, B |
| `src/math/pow.rs` | `e_pow.c`, `e_powf.c` | | A, C |
| `src/math/misc.rs` | `s_cbrt.c`, `s_cbrtf.c`, `e_fmod.c` | | A, B |
| `src/math/erf.rs` | `s_erf.c`, `s_erff.c` | | A |

Notice A, in the msun files `e_expf.c`, `s_expm1.c`, `s_expm1f.c`,
`e_sinhf.c`, `e_coshf.c`, `s_tanh.c`, `s_tanhf.c`, `s_sin.c`, `s_cos.c`,
`s_tan.c`, `e_logf.c`, `k_logf.h`, `e_log2f.c`, `e_log10f.c`, `s_log1p.c`,
`s_log1pf.c`, `s_asinh.c`, `s_asinhf.c`, `e_acoshf.c`, `e_atanhf.c`,
`e_acosf.c`, `s_atan.c`, `s_atanf.c`, `e_atan2f.c`, `e_powf.c`, `s_cbrt.c`,
`s_cbrtf.c`, `s_erf.c` and `s_erff.c`, and in all the newlib files above:

```text
====================================================
Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

Developed at SunPro, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this
software is freely granted, provided that this notice
is preserved.
====================================================
```

Notice B, in `e_sinh.c`, `e_cosh.c`, `k_sin.c`, `k_cos.c`, `k_rem_pio2.c`,
`e_rem_pio2.c`, `e_log.c`, `k_log.h`, `e_log2.c`, `e_log10.c`, `e_acosh.c`,
`e_atanh.c`, `e_asin.c`, `e_acos.c`, `e_atan2.c` and `e_fmod.c`:

```text
====================================================
Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this
software is freely granted, provided that this notice
is preserved.
====================================================
```

Notice C, in `e_exp.c` and `e_pow.c`:

```text
====================================================
Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.

Permission to use, copy, modify, and distribute this
software is freely granted, provided that this notice
is preserved.
====================================================
```

Notice D, in `k_tan.c`:

```text
====================================================
Copyright 2004 Sun Microsystems, Inc.  All Rights Reserved.

Permission to use, copy, modify, and distribute this
software is freely granted, provided that this notice
is preserved.
====================================================
```

The float files also credit their authors: "Conversion to float by Ian Lance
Taylor, Cygnus Support, ian@cygnus.com." (`e_expf.c`, `s_expm1f.c`,
`e_sinhf.c`, `e_coshf.c`, `s_tanhf.c`, `e_logf.c`, `s_log1pf.c`,
`s_asinhf.c`, `e_acoshf.c`, `e_atanhf.c`, `e_acosf.c`, `s_atanf.c`,
`e_atan2f.c`, `e_powf.c`, `s_cbrtf.c`, `s_erff.c` and the newlib files), and
`s_cbrt.c` and `e_rem_pio2.c` say "Optimized by Bruce D. Evans.", `s_cbrtf.c`
"Debugged and optimized by Bruce D. Evans."

### Ports of roadmap lane m11-extern-distributed

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `examples/all_reduce_two_shot_multimem.rs` | `examples/python/CuTeDSL/cute/blackwell/kernel/distributed/all_reduce_two_shot_multimem.py` | the kernel (`all_reduce_multimem_kernel`): the block's tile from the rank's chunk of the `zipped_divide` tiles, the 4 x 32 thread layout with blocks of rows of 4 Float32, the `multimem_ld_reduce_4xf32` and `multimem_st_4xb32` loop, `sync_threads`, and thread 0's `multimem_red_add1` (release, system scope) on its linear block's multicast flag and `spin_lock_atom_cas_relaxed_wait` (expecting `world_size`, resetting to 0) on its unicast flag; the tile height, the inputs and the one-rank simulation are cute-rs's own |

That file carries `Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
reserved.` and is under the BSD-3-Clause licence above.

### Hopper persistent GEMM (m9-hopper-persistent-gemm)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `examples/hopper_persistent_gemm.rs` | `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py` | `HopperWgmmaGemmPersistentKernel`: its configuration (`__init__`: the DMA and MMA warpgroups, the load and store warps, the register requirements, the epilogue barrier), `__call__`, the `kernel` (the warp-specialised roles, the persistent tile loops of the DMA warp and the MMA warpgroups, the pipeline states carried across tiles, the epilogue's buffer index and TMA stores), `_compute_stages`, `_compute_grid`, and their comments |
| `tests/hopper_persistent_gemm.rs`, `tests/mlir.rs` (`hopper_persistent_gemm_in_ptx`) | the same | nothing beyond what they call in `examples/hopper_persistent_gemm.rs` |

### tcgen05 UMMA F16/BF16 and tcgen05.commit (m10-umma-f16)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/nvgpu/tcgen05/mma.rs` | `include/cute/arch/mma_sm100_umma.hpp`, `include/cute/atom/mma_traits_sm100.hpp`, `include/cute/atom/mma_traits_sm100_frag.hpp` | `SM100_MMA_F16BF16_SS`'s shape checks and elected issue (`setp` of the scale into a predicate), the `MMA_Traits<SM100_MMA_F16BF16_SS>` layouts, and `UMMA::tmem_frg_1sm` (the M = 64 and M = 128 accumulator atoms, tiled and restrided to TMEM addresses) |
| `src/atom/mma.rs` (`MmaOp::Sm100`) | `include/cute/atom/mma_traits_sm100.hpp` | the `ThrID`, `ALayout`, `BLayout` and `CLayout` of the SM100 UMMA traits |
| `src/ir.rs` (`UmmaKind::idesc`) | `include/cute/arch/mma_sm100_desc.hpp` | `UMMA::InstrDescriptor`'s bit fields |
| `tests/umma.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py` | the kernel structure (TMEM allocation, the accumulator from `partition_shape_C` and `make_fragment_C` over the TMEM pointer, the k-block loop setting `ACCUMULATE`, the commit, the `make_tmem_copy` epilogue) |

Those files carry `Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
reserved.` (the tutorial `2025 - 2026`) and are under the BSD-3-Clause licence above.

### tcgen05 UMMA kinds TF32, I8 and F8F6F4 (m10-umma-dense-kinds)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/nvgpu/tcgen05/mma_dense.rs` | `include/cute/arch/mma_sm100_umma.hpp` | the shape checks of `SM100_MMA_TF32_SS`, `SM100_MMA_S8_SS` and `SM100_MMA_F8F6F4_SS` |
| `src/ir.rs` (`UmmaDataKind::operand_format`) | `include/cute/arch/mma_sm100_desc.hpp` | `UMMA::to_UMMAFormat`'s TF32, `S8Format` and `MXF8F6F4Format` codes |
| `tests/umma_kinds.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py` | the kernel structure, as in `tests/umma.rs` |

Those files carry `Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
reserved.` (the tutorial `2025 - 2026`) and are under the BSD-3-Clause licence above.

### Hopper fused multi-head attention forward (m9-hopper-fmha)

| cute-rs file | NVIDIA source | What it ports |
|---|---|---|
| `examples/hopper_fmha.rs` | `examples/python/CuTeDSL/cute/hopper/kernel/attention/fmha.py` | `HopperFusedMultiHeadAttentionForward`: its configuration (`__init__`, `_setup_attributes`: the two MMA warpgroups, the register budgets, the stage counts), `__call__` (the K and V views, the staged shared-memory layouts, the TMA atoms), the `kernel` (the load warp's Q, K, V order, the compute warpgroups' mainloop, the ping-pong order barrier, the log-sum-exp store and the `stmatrix` plus TMA store epilogue), `compute`, `softmax_step`, `tail`, `make_acc_into_op`, `layout_acc_mn`, `reduction_target_n`, `gemm_zero_acc`, the pipeline set-up helpers, `_make_tma_atoms_and_tensors`, the tensor set-up and reference of `run`, and their comments |
| `examples/hopper_fmha.rs` | `examples/python/CuTeDSL/helpers/fmha_helpers.py` | `FmhaStaticTileScheduler` (the persistent linear tile walk and grid), `compute_grid`, `MaskEnum` and `FusedMask` (`get_trip_start`, `get_trip_count`, the leading and trailing mask ranges, `get_masked_*_count`, `get_unmasked_trip_count`, `apply_mask`) |
| `tests/hopper_fmha.rs` | the same | nothing beyond what they call in `examples/hopper_fmha.rs` |

`fmha.py` and `fmha_helpers.py` carry
`Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
and are under the BSD-3-Clause licence above.

### SM120 warp-level block-scaled MMA (m10-sm120-blockscaled)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/atom/mma.rs` (`MmaOp::Sm120`) | `include/cute/atom/mma_traits_sm120.hpp`, `include/cute/atom/mma_traits_sm80.hpp` | the `ThrID`, `ALayout`, `BLayout` and `CLayout` of `MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS>` and `SM120_16x8x32_TN_VS` (the latter those of `SM80_16x8x32_S32S8S8S32_TN`) |
| `src/utils/sm120.rs` | `include/cute/atom/mma_traits_sm120.hpp` | the scale-factor atoms `SFALayout` and `SFBLayout` |

Those files carry `Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
reserved.` and are under the BSD-3-Clause licence above.

### tcgen05 TMEM-sourced A and runtime MMA modifiers (m10-umma-kinds)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/nvgpu/tcgen05/mma.rs` | `include/cute/arch/mma_sm100_umma.hpp`, `include/cute/atom/mma_traits_sm100.hpp`, `include/cute/atom/mma_traits_sm100_frag.hpp` | the `_TS` forms' rule that A from TMEM is K-major, their `[a-tmem]` issue with the `{disable-output-lane}` words, and `UMMA::tmem_frg_1sm<T, T, TmemAllocMode::NonInterleaved>` (the A fragment: the tile stride of 2 for M = 64) |
| `tests/umma_ts.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py` | the kernel structure, as in `tests/umma.rs` |

Those files carry `Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
reserved.` (`fp16_gemm_0.py`: 2025 - 2026) and are under the BSD-3-Clause licence above.

### Blackwell pipelines and the TMEM allocator (m10-umma-pipelines)

| cute-rs file | NVIDIA source | What it ports |
| --- | --- | --- |
| `src/pipeline/umma_pipeline.rs` | `include/cutlass/pipeline/sm100_pipeline.hpp` | `PipelineTmaUmmaAsync`, `PipelineAsyncUmma` and `PipelineUmmaConsumerAsync`: which barrier each side arrives on and how (the leader-CTA `arrive_and_expect_tx`, the `tcgen05.commit` releases and commits), the multicast arrival masks and the accumulator pipeline's producer tail |
| `src/memory/tmem_alloc.rs` | `include/cute/arch/tmem_allocator_sm100.hpp` | `Allocator1Sm` and `Allocator2Sm`: allocate, `release_allocation_lock` and free, and the 2-SM free's handshake on the peer CTA's mbarrier |
| `tests/umma_pipelines.rs` | `examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py` | the warp-specialised kernel structure (TMA, MMA and epilogue warps, the two pipelines, the TMEM allocator's call sequence) |
| `tests/cute_ext_pipelines.rs` | `examples/python/CuTeDSL/cute_ext/blackwell/dense_gemm/input_transform_gemm.py` | the warp-specialised GEMM's structure: a TMA or transform warp feeding the MMA warp and the MMA warp feeding four epilogue warps, each through a cute_ext convenience pipeline (`TMAToUMMAPipeline`, `AsyncToUMMAPipeline`, `UMMAtoAsyncPipeline`) with `producer_acquire_and_get_stage`, `get_mbarrier`, `producer_commit_and_advance`, `consumer_wait_and_get_stage` and `consumer_release_and_advance` |

Those files carry `Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
reserved.` (the example `2025 - 2026`) and are under the BSD-3-Clause licence above.

### Cluster launch control (m10-clc-scheduling)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/utils/clc_tile_scheduler.rs` | `examples/python/CuTeDSL/helpers/dynamic_persistent_tile_scheduler.py` | `ClcDynamicPersistentTileSchedulerParams` (grid shape, FastDivmod divisors for N-major order) and `ClcDynamicPersistentTileScheduler` (`create`, `_swizzle_and_rasterize`, `work_tile_info_from_clc_response` with its proxy fence, `initial_work_tile_info`, `advance_to_next_work`) |
| `src/pipeline/clc_pipeline.rs` | `include/cutlass/pipeline/sm100_pipeline.hpp` | `PipelineCLCFetchAsync`'s protocol: the producer's empty wait and per-CTA full-barrier arming, the no-op commit, consumers releasing to the producer CTA, and the producer tail |

Those files carry `Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
reserved.` (the Python helper `2025 - 2026`) and are under the BSD-3-Clause licence above.

### MMA-aware TMA atoms and the first Blackwell tutorial GEMM (m10-port-fp16-gemm)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/nvgpu/tma_sm100_mma.rs` | `include/cute/atom/copy_traits_sm100_tma.hpp` | `make_tma_atom_A_sm100`, `make_tma_atom_B_sm100` and `make_tma_copy_C_sm100`: the CTA value map `layout<1>(mma.thrfrg_X(make_identity_layout(shape(gtensor)).compose(tiler)))(_, repeat<rank>(_))` over the MMA tiler's MK, NK or MN modes, and the multicast count from the cluster's N mode (A) or M mode (B) |
| `examples/blackwell_fp16_gemm.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py` | the kernel `kernel` and `host_function`: the configuration (`(128, 256, 64)` tiler, `128x256x16` instruction, 4 A/B stages, 1 accumulator stage, 128 threads), the shared storage, the TMEM allocation sequence, the two pipelines, the partitioning for MMA, TMA and the TMEM epilogue (four subtiles of `Ld32x32bOp(x64)`), the mainloop and the data of `run_dense_gemm` (integers in `[-2, 2)`), with its comments and names |
| `examples/blackwell_fp16_gemm.rs` (`FP16_GEMM_1`) | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py` | the 2-CTA configuration (`(2, 1, 1)` cluster, `256x256x16` instruction, `(256, 256, 64)` tiler, 7 A/B stages), the cluster layout from `tiled_divide` by the MMA's `thr_id`, the MMA coordinate from `block_idx`, the transaction bytes and consumer counts, the multicast TMA atoms and masks, the 2-CTA TMEM allocator, the leader CTA's share of the mainloop, the producer tails and the grid, with its comments |
| `examples/blackwell_fp16_gemm.rs` (`FP16_GEMM_2`, `Fp16GemmWsConfig`) | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_2.py` | the warp-specialized configuration (6 A/B stages, 2 C stages, 128 epilogue threads), the warp roles (epilogue warps 0-3, MMA warp 4, TMA warp 5), the TMEM allocation by epilogue warp 0 behind named barrier 2 for the MMA and epilogue warps, the epilogue's named barrier 1, the epilogue tile and staged SMEM layout, the TMA store atom, the TMA warp's producer tail, with their comments and names |
| `examples/blackwell_fp16_gemm.rs` (`FP16_GEMM_3`) | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_3.py` | the static persistent configuration (2 accumulator stages), `compute_grid`, the cluster arrive/wait around the partitioning, the partitions with rest modes, the per-role persistent scheduling loops, the staged accumulator and the accumulator producer's tail, with their comments |
| `examples/blackwell_fp16_gemm.rs` (`FP16_GEMM_3_1`) | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_3_1.py` | the CLC configuration (`use_clc_dynamic_scheduler`, `num_clc_stage`, the 16-byte response), the shared storage's `clc_mbar_ptr` and aligned `clc_response`, the CLC pipeline's consumer count, the scheduler warp's loop and the roles' response reads, the 224-thread block, with their comments |
| `examples/blackwell_fp16_gemm.rs` (`FP16_GEMM_4`) | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_4.py` | the preferred `(2, 4, 1)` and fallback `(2, 1, 1)` cluster configuration and its constraints, `kernel`'s two arms over `cluster_specific_kernel`, the per-shape cluster layouts, TMA atoms and scheduler params of `host_function` and `compute_grid`, with their comments |
| `tests/examples.rs` (`blackwell_fp16_gemm_2` to `_4`) | the same | nothing beyond what they call in `examples/blackwell_fp16_gemm.rs` |
| `examples/blackwell_dense_gemm.rs` | `examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py` | `DenseGemmKernel`: the configuration (`__init__`, `_setup_attributes`: the tile's K as four instructions, the CTA tile, the cluster layout from `tiled_divide`, the multicast counts, the epilogue tile, the stage counts and shared-memory layouts, the TMEM columns), `__call__` (the TMA load and store atoms, the transaction bytes, the grid and the launch), the `kernel` (descriptor prefetch, the coordinates, the shared storage, the pipelines' arrival counts, the multicast masks, the partitions, the prefetching mainloop and the leader CTA's MMAs, the epilogue and TMEM free, the producer tail), `_compute_stages`, `_compute_grid`, `_compute_num_tmem_alloc_cols` and the data of its tensors, with their comments and names |
| `tests/examples.rs` (`blackwell_dense_gemm_*`) | the same | nothing beyond what they call in `examples/blackwell_dense_gemm.rs` |

`copy_traits_sm100_tma.hpp` carries `Copyright (c) 2021 - 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.`, `fp16_gemm_0.py` (and `fp16_gemm_2.py`, `fp16_gemm_3.py`, `fp16_gemm_3_1.py` and
`fp16_gemm_4.py`) `Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`; all
are under the BSD-3-Clause licence above. `src/utils/mixed_cluster.rs` (m10-port-clc-clusters) is
cute-rs's own: it follows only the semantics of CuTe DSL's `is_preferred_cluster` and its schedulers'
fallback params.

### The EFC epilogue-fusion GEMMs (m10-port-efc)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
|---|---|---|
| `examples/efc/mod.rs` | `examples/python/CuTeDSL/cute/blackwell/efc/common_efc.py` | the EFC framework: `ACTIVATION_FUNCTIONS`, `EFC.Phase`, `Tensor.ParameterAttributes`, the parameter analysis (`analyze_epilogue_with_arguments`: the used, read and written tensor names), the configuration's `accum`, `load`, `store`, `maximum`, `minimum`, `where`, `full_like` and the activation functions' formulas (identity, relu, leaky_relu, tanh, sigmoid, silu, hardswish, the tanh-approximated gelu), `foreach_*_tensor` and `evaluate_on_cpu`, with their names |
| `examples/efc/dense_gemm.rs` | `examples/python/CuTeDSL/cute/blackwell/efc/common_dense_gemm_efc.py` and `common_efc.py` | `DenseGemmEFC`: the configuration (warp roles 0-3 epilogue, 4 MMA, 5 TMA, 6 epilogue load; barrier ids), `_setup_attributes`, `__call__` (the A/B TMA atoms, the transaction bytes, the grid), `kernel` (the shared storage with the C pipeline's barriers, the A/B, accumulator and C pipelines and their arrival counts, the TMEM allocator, the persistent TMA, MMA, epilogue and epilogue-load loops, the subtile loop with its loads, fences, barriers, TMA stores and D buffer index), `epilogue_tmem_copy_and_partition`, `epilogue_smem_copy_and_partition_load`, `epilogue_gmem_copy_and_partition`, `compute_stages`, `_compute_grid` and `compute_num_tmem_alloc_cols`; and `EFC.JIT`/`EFC.Kernel`: the widest written and read tensors, `compute_stage`, `smem_layout`, `create_tma_arguments`, `allocate_smem` and the per-tensor partitions, loads and stores, with their comments and names |
| `examples/synthetic_custom_epilogue_dense_gemm.rs` | `examples/python/CuTeDSL/cute/blackwell/efc/synthetic_custom_epilogue_dense_gemm.py` | `meta_epilogue` and its computation, and the tensors and scalars of `run` |
| `examples/activation_custom_epilogue_dense_gemm.rs` | `examples/python/CuTeDSL/cute/blackwell/efc/activation_custom_epilogue_dense_gemm.py` | `create_epilogue_function` and its epilogue (the Ada FP8 GEMM pattern), and the tensors of `create_arguments` |
| `examples/custom_epilogue_dense_gemm.rs` | `examples/python/CuTeDSL/cute/blackwell/efc/custom_epilogue_dense_gemm.py` | its epilogue `epilogue(efc_config, C, D, alpha, beta, X, x_factor, Y)`, the default types and scalars, and the tensors of `run` |
| `tests/efc_examples.rs` | the same | nothing beyond what they call in the examples |

The five files carry `Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
and are under the BSD-3-Clause licence above. EFC's Python under `python/CuTeDSL` was not used.

### CTA pairs (m10-cta-pair)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/atom/mma.rs` | `include/cute/atom/mma_traits_sm100.hpp` | the `MMA_Traits<SM100_MMA_*_2x1SM_SS>` layouts: `ThrID` of 2 and the A, B and C layouts split in row halves |
| `src/nvgpu/tcgen05/mma.rs` | `include/cute/arch/mma_sm100_umma.hpp`, `include/cute/atom/mma_traits_sm100_frag.hpp` | the 2x1SM shape checks (M of 128 or 256) and `UMMA::tmem_frg_2sm` (the M = 128 "4x1" and M = 64 "2x2" interleaved accumulator atoms, tiled and restrided to TMEM addresses) |
| `tests/cta_pair.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py` | the structure of the 2-CTA kernel: the pair's tiled MMA sliced by the CTA's rank in the pair, `CtaGroup.TWO` TMA atoms reporting to the leader's mbarrier, the leader alone issuing the MMAs, and the commit to both CTAs |
| `tests/cta_pair_pipelines.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py`, `examples/python/CuTeDSL/cute/blackwell/kernel/attention/mamba2_ssd/mamba2_ssd.py` | the 2-CTA kernel's structure (cluster layout, leader-only MMAs, pipelines with `cta_layout_vmnk`, tails after the epilogue) and the roles of the multi-consumer kernel (one TMA warp feeding the MMA warp and ordinary consumer warps from the same stage) |

`mma_traits_sm100.hpp` carries `Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All
rights reserved.`, `mma_sm100_umma.hpp` `Copyright (c) 2023 - 2026 NVIDIA CORPORATION &
AFFILIATES. All rights reserved.`, `mma_traits_sm100_frag.hpp` `Copyright (c) 2026 NVIDIA CORPORATION &
AFFILIATES. All rights reserved.`, and `fp16_gemm_1.py` `Copyright (c) 2024 - 2026 NVIDIA
CORPORATION & AFFILIATES. All rights reserved.`; all are under the BSD-3-Clause licence above.

### Shared-memory-to-TMEM copies (m10-s2t-copies)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/nvgpu/tcgen05/s2t.rs` | `include/cute/arch/copy_sm100.hpp`, `include/cute/atom/copy_traits_sm100.hpp`, `include/cute/atom/mma_traits_sm100.hpp` | the `SM100_UTCCP_*` instructions and their copy traits' `ValID` (bits to TMEM addresses, with the repeat and broadcast modes), thread and source/destination layouts, `make_utccp_copy`, and `get_utccp_smem_desc_tensor` (the core-matrix view of the partitioned source and its recast to 16-byte units) |
| `tests/s2t_copies.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py` | the scale-factor S2T sequence: `filter_zeros` of both sides, `make_s2t_copy` over the TMEM tensor, `get_slice(0)`, `partition_S`/`partition_D` and `get_s2t_smem_desc_tensor` |

`copy_sm100.hpp` and `copy_traits_sm100.hpp` carry `Copyright (c) 2023 - 2026 NVIDIA CORPORATION &
AFFILIATES. All rights reserved.`, `mma_traits_sm100.hpp` `Copyright (c) 2022 - 2026 NVIDIA
CORPORATION & AFFILIATES. All rights reserved.`, and `nvfp4_gemm_0.py` `Copyright (c) 2025 - 2026
NVIDIA CORPORATION & AFFILIATES. All rights reserved.`; all are under the BSD-3-Clause licence above.

### Block-scaled tcgen05 MMAs (m10-umma-blockscaled)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/nvgpu/tcgen05/mma_blockscaled.rs` | `include/cute/arch/mma_sm100_umma.hpp`, `include/cute/atom/mma_traits_sm100_frag.hpp` | the shape checks and PTX forms of `SM100_MMA_MXF8F6F4_SS`, `SM100_MMA_MXF8F6F4_2x1SM_SS`, `SM100_MMA_MXF4_SS` and `SM100_MMA_MXF4_2x1SM_SS`; the atom of `UMMA::tmem_sf_frg` (the TMEM scale-factor layout) |
| `src/ir.rs` (`UmmaKind::idesc` for block-scaled kinds, `UmmaDataKind::operand_format`) | `include/cute/arch/mma_sm100_desc.hpp` | `InstrDescriptorBlockScaled`'s fields, `make_runtime_instr_desc_block_scaled`'s scale-factor IDs, and `MXF4Format::E2M1` |
| `tests/umma_blockscaled.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py`, `fp16_gemm_1.py` | the kernel structure: scale factors staged into TMEM with a `Cp4x32x128bOp` copy, the k-block loop of block-scaled MMAs, the commit and the TMEM-load epilogue; the CTA pair's leader issuing for both CTAs |

`mma_sm100_umma.hpp`, `mma_traits_sm100_frag.hpp` and `mma_sm100_desc.hpp` carry `Copyright (c)
2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`, and `nvfp4_gemm_0.py` and
`fp16_gemm_1.py` `Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
reserved.`; all are under the BSD-3-Clause licence above.

### Rubin (SM107) tcgen05 MMAs (m10-rubin-umma)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/nvgpu/tcgen05/mma_sm107.rs` | `include/cute/arch/mma_sm107_umma.hpp`, `include/cute/atom/mma_traits_sm107.hpp` | the shape checks and PTX forms of `SM107_MMA_F8F6F4_SS`, `SM107_MMA_MXF8F6F4_SS`, `SM107_MMA_MXF4NVF4_SS` and their 2x1SM forms (K = 64 and 128, `.block16`/`.block32`, `.collector::b::fill`/`lastuse`), and the K and N rules of their traits |
| `src/ir.rs` (`UmmaKind::idesc` and `ptx` for the Rubin forms) | `include/cute/arch/mma_sm107_desc.hpp`, `include/cute/arch/mma_sm100_desc.hpp` | `Sm107MXF8F6F4InstrDescriptor`'s and `Sm107MXF4NVF4InstrDescriptor`'s K-size bits and UE5M3 scale format, and `make_sm107_instr_desc`'s K bit |
| `tests/umma_rubin.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py`, `fp16_gemm_1.py` | the kernel structure: scale factors staged into TMEM with a `Cp4x32x128bOp` copy, the k-block loop of block-scaled MMAs, the commit and the TMEM-load epilogue; the CTA pair's leader issuing for both CTAs |

`mma_sm107_umma.hpp`, `mma_traits_sm107.hpp` and `mma_sm107_desc.hpp` carry `Copyright (c) 2026
NVIDIA CORPORATION & AFFILIATES. All rights reserved.`, `mma_sm100_desc.hpp` `Copyright (c) 2023 -
2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`, and
`nvfp4_gemm_0.py` and `fp16_gemm_1.py` `Copyright (c) 2025 - 2026 NVIDIA CORPORATION &
AFFILIATES. All rights reserved.`; all are under the BSD-3-Clause licence above.

### Sparse tcgen05 MMAs (m10-sparse-umma)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/nvgpu/tcgen05/mma_sparse.rs` | `include/cute/arch/mma_sm100_umma.hpp` | the shape checks and PTX forms of `SM100_MMA_F16BF16_SS_SPARSE`, `SM100_MMA_F16BF16_2x1SM_SS_SPARSE`, `SM100_MMA_F8F6F4_SS_SPARSE` and `SM100_MMA_F8F6F4_2x1SM_SS_SPARSE` |
| `src/atom/mma.rs` (`gemm_tcgen05`'s metadata operand), `src/ir.rs` (`UmmaKind::idesc` sparse flag) | `include/cute/atom/mma_traits_sm100.hpp`, `include/cute/arch/mma_sm100_desc.hpp` | the sparse `MMA_Traits`' split of the metadata TMEM address (its lowest column bit into the descriptor's `sparse_id2_`), and `InstrDescriptor`'s `sparse_flag_` and `sparse_id2_` fields |
| `src/utils/sm100_sparse.rs` | `include/cute/atom/mma_traits_sm100_frag.hpp` | `UMMA::tmem_e_frg` (the metadata TMEM layouts per A width) in `make_sparse_tmem_layout_e` |
| `tests/umma_sparse.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py`, `fp16_gemm_1.py`, `examples/python/CuTeDSL/utils/sparse_utils.py` | the kernel structure (operands staged in shared memory, a `tcgen05.cp` into TMEM, the k-block loop, the commit and the TMEM-load epilogue; the pair's leader issuing for both CTAs) and the six 2:4 metadata codes |

`mma_sm100_umma.hpp`, `mma_traits_sm100_frag.hpp` and `mma_sm100_desc.hpp` carry `Copyright (c)
2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`, `mma_traits_sm100.hpp`
`Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`, and
`nvfp4_gemm_0.py`, `fp16_gemm_1.py` and `sparse_utils.py` `Copyright (c) 2025 - 2026 NVIDIA
CORPORATION & AFFILIATES. All rights reserved.`; all are under the BSD-3-Clause licence above.

### Scale-factor SMEM/TMEM layouts and the NVFP4 tutorial GEMM (m10-port-nvfp4-gemm)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/utils/blockscaled_layout.rs` (`make_tmem_layout_sfa/sfb`) | `include/cute/atom/mma_traits_sm100_frag.hpp` | the atom of `UMMA::tmem_sf_frg` (its 4x1 data path) and its `tiled_product` over the MMA_MN and MMA_K modes, composed with the TMEM restride |
| `src/utils/blockscaled_layout.rs` (`sm103_make_smem_layout_sfa/sfb`) | `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py` | `sm103_make_smem_layout_sfa` and `sm103_make_smem_layout_sfb`: the tilers, the `k_divisor`, the 256-row atom and the stage stride |
| `examples/blackwell_nvfp4_gemm.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py` | the kernel `Sm100BlockScaledDenseGemmKernel.kernel` and its `__call__`: the configuration (`(128, 256)` tiler, `K = 64` instruction, 4 k-blocks per tile, `sf_vec_size` 16, UE4M3 scale factors, 4 A/B stages, 1 accumulator stage, 128 threads, 512 TMEM columns), the shared storage, the scale-factor tensors and TMA atoms, the pipelines, the partitioning for MMA, TMA and the S2T copies, the SFA/SFB placement after the accumulator, the mainloop with its `Field.SFA`/`SFB` settings and the `Ld32x32bOp(x128)` epilogue, with its comments and names |

`mma_traits_sm100_frag.hpp` carries `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All
rights reserved.`, and `sm103_dense_blockscaled_gemm_persistent.py` and `nvfp4_gemm_0.py`
`Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`; all are under
the BSD-3-Clause licence above.

### Rubin helpers and the TMEM load with sparse compression (m10-rubin-extras)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/utils/sm107.rs` (`TiledMma::with_bkeep`, `with_breuse`, the Rubin trivial tiled MMAs) | `include/cute/atom/mma_traits_sm107.hpp`, `include/cute/arch/mma_sm107_umma.hpp` | the MMA operations the trivial tiled MMAs choose among and `MMA_Traits::with(b_reuse)`, the B collector variants |
| `tests/rubin_extras.rs` | `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py`, `examples/python/CuTeDSL/utils/sparse_utils.py` | the sparse-GEMM kernel structure (staged operands, metadata in TMEM, k-block loop of MMAs, commit and TMEM-load epilogue) and the 2:4 metadata codes |

`mma_traits_sm107.hpp` and `mma_sm107_umma.hpp` carry `Copyright (c) 2026 NVIDIA CORPORATION &
AFFILIATES. All rights reserved.`, and `fp16_gemm_1.py` and `sparse_utils.py` `Copyright (c) 2025 -
2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`; all are under the BSD-3-Clause licence
above. The Rubin epilogue tile table, the spcompress operations and their simulation take only
names and semantics from CuTe DSL's Python (`rubin_helpers.py`, `tcgen05/copy.py`).

### XLA FFI call frame (m11-framework-exports)

`src/xla_ffi.rs` declares, with hand-written `#[repr(C)]` structs, the part of the XLA FFI C ABI
(`xla/ffi/api/c_api.h` of OpenXLA, Apache-2.0: `XLA_FFI_CallFrame`, `XLA_FFI_Buffer`, `XLA_FFI_Api`
up to `XLA_FFI_Stream_Get`, the attribute and error structs, and the `XLA_FFI_DataType`, stage and
error-code numbers) that a handler reads. Only the layouts and numbers (an interface) are
reproduced, from memory; no OpenXLA source is copied. The JAX shim in `python/cute_rs_jax` uses
the names of `cutlass.jax` (EULA) for its helpers and nothing else of it.

### Typed MMA descriptors (m11-prims-descriptors-swizzle)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/prims/descriptors.rs` | `include/cute/arch/mma_sm90_desc.hpp`, `include/cute/arch/mma_sm100_desc.hpp`, `include/cute/arch/mma_sm100_umma.hpp` | `GmmaDescriptor`'s, `UMMA::SmemDescriptor`'s, `UMMA::InstrDescriptor`'s and `UMMA::InstrDescriptorBlockScaled`'s bit fields and format codes, and the M/N shape rules of the SM100 MMA structs |

`mma_sm90_desc.hpp`, `mma_sm100_desc.hpp` and `mma_sm100_umma.hpp` carry `Copyright (c) 2023 - 2026
NVIDIA CORPORATION & AFFILIATES. All rights reserved.` and are under the BSD-3-Clause licence above.
The descriptor type names, the `Tcgen05MxOmmaInstrDesc` layout and the base DSL's swizzle
(`src/prims/swizzle.rs`) take only names and semantics from CuTe DSL's Python
(`experimental/primitives/descriptors.py`, `base_dsl/swizzle.py`).

### Primitives-only MMA forms (m11-prims-mma-forms)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `src/prims/mma.rs` | `include/cute/atom/mma_traits_sm100_frag.hpp`, `include/cute/atom/mma_traits_sm100.hpp`, `include/cute/arch/mma_sm120_sparse.hpp` | `UMMA::tmem_frg_ws` (the weight-stationary M = 32 "1x4", M = 64 "2x2" and M = 128 accumulator atoms, tiled and restrided) in `tmem_frg_ws_layout`; the K and scale-factor vector sizes of `SM100_MMA_MXF8F6F4_SS_SPARSE` and `SM100_MMA_MXF4NVF4_SS_SPARSE`; the operand registers of `SM120_SPARSE_16x8x64_TN_VS` and `SM120_SPARSE_16x8x128_TN_VS` |
| `src/sim/exec/umma.rs` (the `.ws` placements) | `include/cute/atom/mma_traits_sm100_frag.hpp` | the data-path placements of `UMMA::tmem_frg_ws` (D) and the replicated metadata of `UMMA::tmem_e_frg_ws` |
| `tests/prims_mma.rs` (the `slicek_*` test) | `examples/python/CuTeDSL/experimental/primitives/tcgen05/slicek_m32n8k128.py` | the six variants' operand layouts (the compact 32-byte-swizzled A, the 128-byte-swizzled logical A of the TMA variant, A in TMEM), the expanded and N-group B descriptors, the MMA sequences with `.ashift`, the A collector and the write-disable masks, and the epilogue's stripe selection and four-lane reduction |

`mma_traits_sm100_frag.hpp` carries `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All
rights reserved.`, `mma_traits_sm100.hpp` `Copyright (c) 2022 - 2026 NVIDIA CORPORATION &
AFFILIATES. All rights reserved.`, `mma_sm120_sparse.hpp` `Copyright (c) 2025 - 2026 NVIDIA
CORPORATION & AFFILIATES. All rights reserved.` and `slicek_m32n8k128.py` `Copyright (c) 2026
NVIDIA CORPORATION & AFFILIATES. All rights reserved.`; all are under the BSD-3-Clause licence
above. The function and enum names and the semantics of the modifiers (`.ashift` per
sub-partition, the collector buffers, `scale-input-d`) take only names and semantics from CuTe
DSL's Python (`experimental/primitives/nvvm_wrapper.py`).

### Task Scheduling resources and schedule capture (m11-ts-resources)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `tests/task_scheduling.rs` | `examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/01_copy_basics_ts/01_copy_grid_stride.py`, `02_copy_tma.py`, `03_copy_tma_conditional.py`, `03_persistent_scheduling_dynamic_domain_ts/01_copy_persistent_skip_tile.py` | the tutorials' resource, work and variable names and the order of the steps of their schedules (`schedule_fn`, `load_schedule`, `store_schedule`, `padding_schedule`, `copy_schedule`), captured and compared with listings |
| `tests/ts_kernels/mod.rs`, `tests/gpu_task_scheduling.rs` | the same tutorials | the structure of the hand-written kernels: the grid-stride copy, a load warp handing rows of a CTA tile through a staged shared-memory resource to four storing warps, and a persistent copy that skips over-launched tiles |

The tutorials carry `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` and
are under the BSD-3-Clause licence above. `src/task_scheduling/` takes only names and semantics from
CuTe DSL's Python (`cutlass/experimental/task_scheduling`: `resources.py`, `schedule_builder.py`,
`enums.py`), under the NVIDIA EULA; its code, listing format and messages are written independently.

### Task Scheduling tasks and the TaskManager (m11-ts-task-manager)

| cute-rs file | CUTLASS 4.8.1 file | What it ports |
| --- | --- | --- |
| `examples/ts_copy_grid_stride.rs` | `examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/01_copy_basics_ts/01_copy_grid_stride.py` | the resources `InputGmemResource` / `OutputGmemResource`, the schedule `schedule_fn`, the task `CopyTask`, the dependency graph and the TaskManager calls of `gmem_copy_kernel` |
| `examples/ts_copy_persistent_skip_tile.rs` | `examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/03_persistent_scheduling_dynamic_domain_ts/01_copy_persistent_skip_tile.py` | the resources `OversubscribedCopyWorkQueue` / `PersistentCopyResource`, the schedule `copy_schedule`, the task `PersistentCopyTask`, the dependency graph, the TaskManager calls and the over-launched tile grid |
| `tests/ts_kernels/mod.rs` (`managed_staged_copy`), `tests/task_manager.rs`, `tests/gpu_task_scheduling.rs` | `01_copy_basics_ts/02_copy_tma.py` and the two tutorials above | the load and store tasks of the staged copy and their schedules, run through the TaskManager |
| `examples/ts_copy_tma.rs`, `tests/ts_memory_groups.rs` | `examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/01_copy_basics_ts/02_copy_tma.py` | the resources `InputGmemResource` / `SmemResource` / `OutputGmemResource`, the schedules `load_schedule` / `store_schedule` / `padding_schedule`, the tasks, the `SmemAllocator` calls and the TaskManager calls of `tma_copy_kernel` |
| `examples/ts_merge_copy.rs`, `tests/ts_memory_groups.rs` | `examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/07_group_pipeline_ts/01_merge_copy.py` | the resources `GmemAResource` / `GmemBResource` / `SmemAResource` / `SmemBResource` / `StoreResource`, the Merge `PipelineGroup` `ab_sync`, the schedules, the tasks, the `SmemAllocator` calls and the TaskManager calls of `merge_copy_kernel` |
| `tests/ts_memory_groups.rs` (the UMMA hand-off kernel) | `examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py` | the MMA issue loop and the TMEM epilogue (`tcgen05.ld`, `fence_view_async_tmem_load`, release) |
| `examples/ts_gemm_simple.rs`, `tests/ts_gemm.rs` | `examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/02_gemm_simple_ts/01_fp16_bf16_gemm_3.py` | the configuration, the resources `GmemAbResource` / `SmemAbResource` / `TmemCResource` / `GmemDResource` and their work methods (coordinates, TMA loads, SMEM descriptors, the tcgen05 MMA loop with `scale_d`, the T2R subtile load and the vectorised store with its scalar tail), the schedules `load_schedule` / `mma_schedule` / `store_schedule` / `padding_schedule`, the tasks with their register budgets, the allocator and TaskManager calls of `_create_gemm_pipeline`, the descriptor prefetch, the TMA boxes and the grid |

`src/task_scheduling/{task,manager,verify,pdl}.rs` (and the compile-time work arguments and shared register budgets of `resource.rs`, `schedule.rs`, `manager.rs` and `verify.rs`) take only names and semantics from CuTe DSL's
Python (`task.py`, `task_manager.py`, `exhaustive_checker.py`, `resources.py`), under the NVIDIA
EULA; their code and messages are written independently.
