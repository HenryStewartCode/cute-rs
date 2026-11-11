# cute-rs

A CuTe-style GPU kernel DSL for Rust.

You write kernels as ordinary Rust, using the same building blocks as NVIDIA's
[CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) and CuTe DSL:
hierarchical layouts and their algebra, tensors over global, shared and
register memory, and copy and MMA atoms tiled across threads. A built kernel can be

* **simulated on the CPU** by a checking simulator. It catches shared-memory
  races (ordered by `sync_threads`, named barriers and mbarriers), missing
  `cp.async` waits, bulk copies outside their mbarrier's transaction count,
  mbarrier parity waits that could pass early on hardware, out-of-bounds and
  misaligned accesses, divergent barriers and barriers that cannot complete,
  and reports each one with the Rust source line,
* **lowered to MLIR** using only upstream dialects (`gpu`, `nvgpu`, `nvvm`,
  `scf`, `arith`, `memref`, `vector`), with CuTe DSL's `cute.math` library
  expanded into exactly rounded operations, so that no libdevice is
  needed and the GPU computes the simulator's bits,
* **compiled to PTX** with a stock `mlir-opt` (no CUDA toolkit needed),
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

## Examples

| Example | What it shows |
|---|---|
| [`elementwise_add`](examples/elementwise_add.rs) | The CuTe DSL tutorial. A naive kernel, then a tiled one built from `make_layout_tv`, with 128-bit vector copies and an identity tensor predicating a partial last tile. |
| [`sgemm`](examples/sgemm.rs) | CuTe's `sgemm_2`: a SIMT GEMM with tiled copies into shared memory, a 16×16 `UniversalFma` `TiledMma`, a GPU-side K loop carrying the accumulators, and an α/β epilogue. |
| [`tensorop_gemm`](examples/tensorop_gemm.rs) | An SM80 tensor-core GEMM on f16 or bf16 inputs. 4 warps of `mma.sync.m16n8k16`, a 3-stage `cp.async` pipeline into swizzled shared memory, and `ldmatrix` copies derived from the MMA's own layouts. Bank conflicts are measured with and without the swizzle. |
| [`single_warp_pipeline`](examples/single_warp_pipeline.rs) | The kernel this project started from: one warp with a double-buffered `cp.async` pipeline, unrolled at build time, using `ldmatrix.trans` for an N-major B. |
| [`runtime_gemm`](examples/runtime_gemm.rs) | `tensorop_gemm` with `M`, `N` and `K` only known at launch: one compiled kernel for every problem size. |
| [`run_on_gpu`](examples/run_on_gpu.rs) | Runs the tensor-core GEMM on a real GPU and checks it (needs `--features cuda`). |

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
| `cute.composition`, `complement`, `coalesce`, `right_inverse`, `left_inverse` | same names |
| `cute.logical_divide`, `zipped_divide`, `tiled_divide`, `flat_divide` | same names (layouts and tensors) |
| `cute.logical_product`, `blocked_product`, `raked_product`, `tile_to_shape` | same names |
| `cute.make_layout_tv(thr, val)` | `make_layout_tv(&thr, &val)` |
| `cute.make_swizzle(b, m, s)`, `make_composed_layout` | `make_swizzle`, `make_composed_layout` |
| `tensor[None, 0]` | `tensor.slice((.., 0))` |
| `tensor[i, j]`, `tensor[i, j] = v` | `tensor.load((i, j))`, `tensor.store((i, j), v)` |
| `cute.local_tile(t, tiler, coord, proj=...)` | `local_tile`, `local_tile_proj` |
| `cute.local_partition` | `local_partition`, `local_partition_proj` |
| `cute.make_fragment`, `make_fragment_like` | `make_fragment`, `make_fragment_like` |
| `cute.make_identity_tensor`, `cute.elem_less` | `make_identity_tensor`, `elem_less` |
| `cute.make_copy_atom(CopyUniversalOp(), T, num_bits_per_copy=128)` | `make_copy_atom::<T>(CopyOp::Universal { bits: 128 })` |
| `cpasync.CopyG2SOp`, `warp.LdMatrix8x8x16bOp(transpose, 4)` | `CopyOp::CpAsync { .. }`, `CopyOp::LdMatrix { num: 4, trans }` |
| `cute.make_tiled_copy(atom, layout_tv, tiler_mn)` | `make_tiled_copy(&atom, &layout_tv, tiler_mn)` |
| `cute.make_tiled_copy_tv`, `make_tiled_copy_A/B/C`, `make_tiled_copy_S/D`, `make_tiled_copy_C_atom` | `make_tiled_copy_tv`, `make_tiled_copy_a/b/c`, `make_tiled_copy_s/d`, `make_tiled_copy_c_atom` |
| `thr_copy.partition_S/D`, `tiled_copy.retile(t)` | `partition_s`, `partition_d`, `tiled_copy.retile(&t)` (or `thr_copy.retile_s/retile_d`) |
| `atom.value_type`, `thr_id`, `layout_src_tv`, `layout_dst_tv`; `tiled_copy.layout_tv_tiled`, `tiler_mn`, `layout_src_tv_tiled`, `layout_dst_tv_tiled`, `size` | the same names, as methods |
| `cute.copy(atom, src, dst, pred=p)`, `cute.autovec_copy` | `copy`, `copy_if`, `autovec_copy` |
| `cute.nvgpu.warp.MmaF16BF16Op(ab_dtype, acc_dtype, shape_mnk)`, `MmaTF32Op(shape_mnk)` | `MmaF16BF16Op::new(ab_dtype, acc_dtype, shape_mnk)`, `MmaTF32Op::new(shape_mnk)` (or `MmaOp::Sm80(MmaKind::..)`) |
| `cute.make_mma_atom(op)`, `cute.make_tiled_mma(op_or_atom, atom_layout)` | `make_mma_atom(op)`, `make_tiled_mma(op_or_atom, atom_layout)` |
| `thr_mma.partition_A/B/C`, `make_fragment_A/B/C` | `partition_a/b/c`, `partition_fragment_a/b/c` |
| `cute.gemm(mma, d, a, b, c)` | `gemm(&mma, &d, &a, &b, &c)` |
| `cutlass.Int8` to `Int64`, `Uint8` to `Uint64`, `Float16`, `BFloat16`, `Float32`, `Float64`, `Boolean` | the same names |
| `cutlass.range` (runtime loop) / `range_constexpr` | `for_range`, `for_range_carry` / a Rust `for` loop |
| `cutlass.range(start, stop, step, unroll=K)` | `for_range_with(start, stop, step, LoopOptions::unroll(K), body)` (and `for_range_carry_with`): LLVM unroll metadata, `count = K`; 1 keeps the loop rolled |
| `cutlass.range(..., unroll_full=True)` | `LoopOptions::unroll_full()` (needs a trip count LLVM can compute; giving `unroll` too is an error, where CuTe DSL lets `unroll_full` win) |
| `cutlass.range(..., vectorize=True)` | `LoopOptions { vectorize: true, ..LoopOptions::default() }` (sm_100 and newer, checked when compiling) |
| `cutlass.range(..., at_least_once=True)` | `LoopOptions { at_least_once: true, ..LoopOptions::default() }`: an `llvm.intr.assume` that `start < stop`, which the simulator checks |
| `cutlass.while_generate` | `while_carry` |
| dynamic `if` / `cutlass.const_expr` | `if_`, `if_else`, `if_else_yield` / a Rust `if` |
| variables assigned inside a dynamic `if`/`for`/`while` (carried automatically by the preprocessor) | `let v = Var::new(init)`, `v.get()`, `v.set(x)`, `v.update(f)`: carried through loops and returned from branches automatically |
| a dataclass carried through runtime control flow or held in such a variable | `carry_struct!` (declares the struct, or implements `Carry` for an existing one) |
| `cute.arch.thread_idx()`, `sync_threads()`, `sync_warp()`, `cp_async_wait_group(n)` | the same names |
| `cute.arch.barrier(barrier_id=i, number_of_threads=n)`, `barrier_arrive(barrier_id=i, number_of_threads=n)` | `barrier(i, n)`, `barrier_arrive(i, n)` |
| `cute.arch.barrier(barrier_id=i)` (every thread of the CTA), `barrier_arrive(..., aligned=False)` | `pipeline::sync(i)`, `barrier_arrive_unaligned(i, n)` |
| `cutlass.pipeline.NamedBarrier(barrier_id, num_threads)` and its methods; `pipeline.arrive`, `arrive_unaligned`, `wait`, `wait_unaligned`, `arrive_and_wait`, `sync` | `pipeline::NamedBarrier::new(id, n)` (each a constant or a runtime `Int32`), the same methods; `pipeline::arrive(id, n)` and so on |
| `cute.arch.warp_idx()`, `make_warp_uniform`, `warpsize`, `WARP_SIZE`, `WARPS_PER_WARPGROUP`, `THREADS_PER_WARPGROUP`, `FULL_MASK` | the same names |
| `cute.arch.shuffle_sync*`, `vote_*_sync`, `match_sync`, `elect_one`, `activemask`, `lanemask_*`, `clock`, `popc`, `clz`, `brev`, `mul_hi`, `printf` | the same names (bit counts and `mul_hi` are methods) |
| `cute.arch.atomic_add` and friends, `fence_acq_rel_*` | the same names, on a `Ptr` |
| `cute.arch.mbarrier_init`, `mbarrier_arrive`, `mbarrier_test_wait` | the same names (SM80 and newer) |
| `cute.arch.mbarrier_init_fence`, `mbarrier_expect_tx`, `mbarrier_arrive_and_expect_tx`, `mbarrier_try_wait`, `mbarrier_conditional_try_wait`, `mbarrier_wait`, `cp_async_bulk` | the same names (SM90 and newer) |
| `mbarrier_arrive(ptr, peer_cta_rank_in_cluster, arrive_count, relaxed=, scope=)`, and the same keywords of the expect-tx operations | `mbarrier_arrive_with(&ptr, count, MbarrierOptions { peer_cta_rank_in_cluster, relaxed, scope })`, `mbarrier_arrive_and_expect_tx_with`, `mbarrier_expect_tx_with` |
| `cute.make_tensor`, `make_ptr`, `recast_ptr`, `cute.arch.alloc_smem` | `make_tensor`, `tensor.ptr()`, `ptr.cast::<U>()`, `alloc_smem` |
| `cute.make_rmem_tensor` | `make_rmem_tensor` (indexable at runtime) |
| `cute.max_common_layout`, `max_common_vector`, `dice`, `make_layout_like`, `make_ordered_layout`, `zipped_product`, `tiled_product`, `flat_product` | same names |
| `cute.math.exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `erfc`, `pow`, `fpowi`, `ipowi`, `cbrt`, `sqrt`, `rsqrt`, `rcp`, `copysign`, `rem`, `round`, `roundeven`, `floor`, `ceil`, `trunc`, `abs`/`absf`, `absi`, `neg`, `add`, `sub`, `mul`, `div`, `fma`, `min`, `max`, `clamp`, `isnan`, `isinf`, `isfinite`, `isnormal` (also `cute.exp2` and so on) | the same names in `cute::math`, with `fastmath` as the last parameter: `math::exp(x, true)` is `cute.math.exp(x, fastmath=True)` |
| `cute.math.sincos(x)` | `math::sincos(x, fastmath)`, a `(sin, cos)` tuple |
| `approx=`, `ftz=`, `rounding=cute.RoundingMode.ZERO`, `full=`, `propagate_nan=` | the `_with` functions and `MathOptions`: `math::exp2_with(x, MathOptions::APPROX.with_ftz())`, `math::add_with(x, y, MathOptions::rounding(RoundingMode::Zero))` |
| `cute.arch.exp2(x)` (deprecated) | `math::exp2(x, true)` (`ex2.approx.ftz.f32`) |
| `@cute.kernel` | `Kernel::build(name, \|\| { ... })` |

## How it works

```text
Rust kernel code ──► builder ──► IR ──► validate ──┬──► simulator (CPU, checking)
 (layouts, tensors,   (thread-     (structured,      │
  atoms, Value ops)    local)       SSA)             └──► MLIR ──► mlir-opt ──► PTX ──► CUDA driver
```

* **Layout algebra runs at build time,** as it does in CuTe. Coordinates and
  strides can depend on runtime values through `IndexExpr`, a symbolic integer
  expression that folds constants and simplifies `div`/`mod` against known
  strides and bounds. Only what can't be resolved becomes instructions.
  Shapes are static; a mode whose extent is only known at launch is declared
  with a static upper bound and a runtime stride
  ([`Layout::with_strides`](src/layout/layout.rs)).
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
  the block). Compiling for an architecture checks that every instruction
  exists there, naming the first that does not with its source line, and
  raises the PTX ISA version when an instruction needs a newer one than the
  architecture's minimum (`mbarrier.test_wait.parity` needs 7.1 on sm_80).
  Kernels with more than 48 KB of shared memory are compiled to allocate it
  dynamically.
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
* **Copies check what they emit.** A vectorised copy (`Universal { bits:
  128 }`, `cp.async`, `ldmatrix`) is emitted only when the per-thread chunk
  is provably contiguous and aligned, including under a swizzle. Otherwise
  the kernel build fails with a message saying why.
* **The simulator** runs each block (or block cluster) in lockstep with
  per-thread masks, over byte-addressed memory.
  `ldmatrix` and `mma.sync` are implemented from the PTX ISA's fragment
  tables, independently of the layout algebra, so a correct simulated GEMM
  is real evidence that the layouts are right. Races in shared memory are
  found with a vector clock per warp: `sync_threads`, named barriers
  (`bar.sync` acquires what `bar.arrive` released) and mbarriers (a
  successful wait acquires what the phase's arrivals released) order
  accesses. Each block has 16 named barriers, counted in whole warps.
  Bulk copies land when a thread waits on their mbarrier after the phase's
  arrivals are in, and count down its transaction count: a phase that
  completes with copies still in flight, or copies that exceed what
  `expect_tx` announced, are reported, and the landed data is ordered only
  before threads that waited for the phase. A parity wait is reported when
  the waiting warp has not seen the previous phase complete, since on
  hardware the parity could then name an older phase and pass at once,
  which lockstep execution would hide. It reports bank-conflict wavefronts
  and global-memory sectors, so the effect of a swizzle or vectorisation is
  visible without a profiler.

## Requirements

* Rust 1.88 or newer (edition 2024).
* For PTX: an MLIR/LLVM installation with the NVPTX backend. LLVM 21 from
  Homebrew works (`brew install llvm`); on Debian/Ubuntu, `mlir-opt` comes
  from the `mlir-<version>-tools` package on apt.llvm.org.
  `Toolchain::discover()` looks at `$CUTE_MLIR_OPT`, `$LLVM_DIR/bin`, `PATH`,
  Homebrew and `/usr/lib/llvm-*`.
* To run on a GPU: Linux, an NVIDIA driver, and an SM80 or newer GPU. Build
  with `--features cuda`. `libcuda` is loaded at runtime, so building needs no
  CUDA installation.

## Testing

```bash
cargo test
```

* `src/layout/tests.rs` covers the layout algebra: CuTe's documented examples,
  plus brute-force property tests for coalesce, composition, complement,
  inverses and TV layouts.
* `tests/examples.rs` simulates every example and checks it against a CPU
  reference. The GEMMs are bit-exact.
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
* `tests/diagnostics.rs` builds deliberately broken kernels and checks that
  each bug is caught (races, missing waits, OOB, misalignment, divergent
  barriers, barriers that cannot complete, mbarrier parity waits that pass
  early, bulk copies missing from `expect_tx`, escaping values, a `Var` or
  loop state whose type or register-tensor layout changes, a `Var`'s
  alignment or constant relied on inside a loop that carries it, a register
  fragment a runtime loop reads and then writes, a loop marked
  `at_least_once` that runs zero times, a vectorize hint below sm_100,
  `cute.math` options CuTe DSL rejects or PTX lacks, …).
* `tests/atoms.rs` checks copy-atom and tiled-copy introspection and the
  layouts of CuTe's tiled-copy constructors and `retile`.
* `tests/mlir.rs` verifies every example with `mlir-opt` and checks its
  sm_80 PTX for the expected instructions, as well as every `mma.sync` kind,
  the named barriers, the SM80 and SM90 mbarrier operations, the effect of
  the loop hints (`unroll=4` gives four copies of the body plus a
  remainder, `unroll=1` `.pragma "nounroll"`, `unroll_full` a loop without a
  branch, `at_least_once` a loop without the zero-trip check in front), and
  every `cute::math` function on every type in both modes: precise PTX
  contains no approximation and no libdevice call, fast PTX the
  approximations the documentation names, and float `abs`, `neg` and
  `copysign` are `and.bN` and `xor.bN`, never `abs.f32` or `neg.f32`. It
  also compiles kernels for each of sm_80, sm_86, sm_89, sm_90, sm_100 and
  sm_120. These tests are skipped when no `mlir-opt` is found; when `ptxas`
  is installed (or named by `$CUTE_PTXAS`), the sm_100a and sm_120 PTX of
  the vectorize hint is assembled with it.
* `tests/gpu.rs` runs every example on a real GPU and on the simulator and
  compares every output. Results must match bit for bit, except that
  `mma.sync` kernels on data that has to round only need to agree within the
  f32 summation error bound, because the GPU rounds differently from the
  simulator. It also runs every `cute::math` function on the GPU (2^16
  inputs per function and type, every bit pattern of the 16-bit types for
  the unary functions): precise mode, the rounding modes, the `ftz` forms,
  `absi`, and `abs`, `neg` and `copysign` (on NaN payloads, alone in a
  kernel, on products, with constant signs and with their bits used as
  integers) must give the simulator's bits, and fast mode and the explicit
  approximations must stay within their documented bounds. Run it with
  `cargo test --features cuda --test gpu`. It is skipped when there is no
  GPU.
* `tests/features.rs` runs a kernel for every scalar type, operation and
  intrinsic, and for `Var`s, `carry_struct!` structs and the loop hints
  (`tests/feature_kernels`),
  and holds the simulator to a reference in plain Rust. The GPU test then
  holds the GPU to the simulator on the same kernels, except those needing
  SM90 (`feature_kernels::sm90`), which are simulated and compiled for
  sm_90 only.

## Limitations

* Layouts have static shapes with optional runtime strides, which covers
  problem sizes known at launch (`runtime_gemm`) but not tiles whose size
  depends on a runtime value. Offsets are 32-bit, so a tensor has at most
  2^31 elements.
* Tensor-core MMAs are scalar FMA and SM80 `mma.sync`: f16 and bf16
  m16n8k16 and m16n8k8 (f32 accumulators, or f16 for f16 inputs), TF32
  m16n8k4 and m16n8k8, and f64 m8n8k4. TF32 operands are f32 registers whose
  low mantissa bits the tensor core ignores; there is no `TFloat32` type or
  `cvt.rna.tf32`. FP8, sparse and block-scaled `mma.sync`, SM90 WGMMA, TMA
  and SM100 `tcgen05` are not implemented.
* mbarriers work from SM80 (`mbarrier_init`, `mbarrier_arrive`,
  `mbarrier_test_wait`). The SM90 forms (arrival counts, `expect_tx`,
  relaxed and cluster-scope arrivals, peer-CTA arrivals through `mapa`,
  `try_wait`, `fence.mbarrier_init`) and bulk copies are implemented, along
  with block clusters in the simulator and the launcher, but no SM90 GPU
  has run them yet; their PTX assembles with `ptxas` for sm_90a and sm_100a.
  `mbarrier_wait` uses `try_wait`, so it needs SM90 as in CuTe DSL (spin on
  `mbarrier_test_wait` on SM80). `cp.async.mbarrier.arrive` and cluster
  barriers are missing. Only `Kernel::compile_ptx` checks a kernel against
  its target; `Toolchain::compile_ptx` compiles raw MLIR unchecked.
* Kernels are single functions: no calls, recursion or unstructured control
  flow. `inline_ptx` reaches any instruction the DSL doesn't have, though the
  simulator can only run it when given a Rust equivalent. LLVM does not
  treat inline PTX as convergent and may duplicate it into both sides of a
  branch, so it should not hold warp- or CTA-collective instructions such
  as `barrier.cluster` (the DSL's own barriers use convergent NVVM
  intrinsics).
* `make_warp_uniform` emits no instruction: CuTe DSL lowers it to a hint
  for NVIDIA's compiler, which upstream LLVM has no equivalent for. The
  simulator checks that the value is warp-uniform.
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
* `cute::math` functions take scalars (apply them to a register tensor
  with `map_in_place`); CuTe DSL's TensorSSA forms are not implemented.
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
  only. Accesses within a warp are treated as synchronous. It runs threads in
  lockstep, so a kernel in which one warp must spin until another makes
  progress later in program order is reported as non-terminating, and a
  named barrier whose remaining arrivals come later in program order (say,
  consumers waiting in the `then` branch of an `if_else` whose `else` branch
  arrives) is reported as unable to complete; arriving first works. Every
  thread of a warp must reach a named barrier together, including the
  unaligned `barrier.arrive`. Of the mbarrier waits that lockstep hides, it
  reports those that could pass early, but not a wait whose mbarrier could
  already be two phases further on. The bulk copies of a block share one
  clock, so waiting for one copy also orders a reader after the copies that
  landed before it.
* The `cuda` launcher has only been run on one GPU, an RTX 2000 Ada (SM89).
  PTX for other architectures assembles with `ptxas` but has not been run.

## Layout of the crate

```text
src/
  layout/      IntTuple trees, Layout and the algebra, IndexExpr, swizzles
  value.rs     Value<T>, numeric types, operators
  control.rs   if_/for_range/while_carry, loop hints, Var, the Carry trait
               and carry_struct!
  arch.rs      thread/block indices, barriers, mbarriers, cp.async groups,
               warp collectives
  pipeline.rs  cutlass.pipeline's NamedBarrier
  tensor.rs    Tensor, slicing, local_tile/partition, fragments
  math/        cute.math: msun-derived expansions, fast and instruction forms
  atom/        copy atoms and TiledCopy, MMA atoms and TiledMma
  builder.rs   Kernel construction (thread-local builder, CSE, DCE)
  ir.rs        the IR
  validate.rs  IR verification
  sim/         the CPU simulator
  backend/     MLIR emission and PTX compilation
  cuda.rs      CUDA driver launcher (feature "cuda")
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

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
