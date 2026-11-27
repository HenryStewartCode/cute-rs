# Testing

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
* `tests/diagnostics/` builds deliberately broken kernels and checks that
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

## Checking the layout algebra against CuTe DSL

[`cutedsl-check/`](../cutedsl-check) asks NVIDIA's CuTe DSL and cute-rs the same questions.
`reference/gen_layouts.py` draws seeded random calls of 41 layout operations, 500 of each
(`coalesce`, `composition`, `complement`, the inverses, the four divides, the six products,
`tile_to_shape`, `make_layout_tv`, `recast_layout`, slicing, swizzled layouts and the rest),
on layouts that are bijections, injections with gaps, or arbitrary, and records what CuTe DSL
4.8.1 prints for each. `cutedsl-check layouts` makes the same calls in cute-rs and compares
the text.

Two seeds, 20,500 calls each. The second was recorded after the fixes described below, so
nothing was tuned on it:

| | Seed 20261126 | Seed 20261127 |
| --- | ---: | ---: |
| The same answer, character for character | 18,475 | 18,470 |
| Both refuse | 1,552 | 1,569 |
| CuTe DSL refuses, cute-rs answers | 53 | 35 |
| Ill-posed: CuTe DSL answers, cute-rs refuses | 420 | 426 |
| cute-rs refuses a well-posed call | 0 | 0 |
| Different answers | 0 | 0 |

"Both refuse" includes the calls that end CuTe DSL's process with a floating-point exception
(543 and 559, products and divides of overlapping layouts).

A call counts as ill-posed only when cute-rs refuses it and a rule that looks at the call's
inputs alone (`cutedsl-check/src/ill_posed.rs`) says why. There are four rules:

* the complement of a layout that is not injective (73 and 96 calls). C++ CuTe refuses it with
  a static assertion; CuTe DSL returns extents of 0, such as `(0,0,0,2):(3,6,8,12)`;
* a composition for which CuTe builds a layout that is not the composition: some index gets an
  offset other than `lhs(rhs(i))` (192 and 181 calls). CuTe's divisibility condition lets
  through a stride that is smaller than a mode and does not divide it. cute-rs accepts such a
  stride only when the result is the composition, which it checks value by value, and refuses
  the rest; C++ CuTe and CuTe DSL return the wrong layout;
* an upcast by a ratio that a stride neither divides nor is divided by (87 and 82), and a
  downcast of a layout with no stride-1 mode (68 and 67). Both are static assertions in C++
  CuTe; CuTe DSL returns rounded values.

The first run found real differences in cute-rs, now fixed. It refused 771 calls that CuTe
DSL answers: `complement` and `composition` wanted strides that divide each other, where CuTe
only needs them not to overlap, and every divide and product inherited that. It answered 143
differently (`left_inverse`, and compositions that run past the left-hand side's domain). And
it printed 2,205 answers differently though they were the same map: the stride of an extent-1
mode, and products padded with extent-1 modes. `cutedsl_recorded_vectors` in
`src/layout/tests.rs` keeps 31 of the recorded calls in the ordinary test suite.

The recorded answers are the output of NVIDIA's package and are not in this repository. To
record them again (Linux, no GPU needed) and compare:

```bash
pip install nvidia-cutlass-dsl==4.8.1
python cutedsl-check/reference/gen_layouts.py --seed 20261126 layouts.tsv
cargo run --release -p cute-rs-cutedsl-check -- layouts layouts.tsv --show 3
```

The generator draws static layouts with non-negative integer strides. Runtime extents,
negative strides and scaled-basis strides are not compared, and nor is anything beyond the
layout algebra (tensors, atoms, kernels).

## Checking the simulator against a GPU

On a machine with a GPU, every simulated kernel can also be run on the device and
compared. With the `cuda` feature, set `CUTE_RS_CROSSCHECK_GPU` to a log file:

```bash
CUTE_RS_CROSSCHECK_GPU=/tmp/crosscheck.log cargo test --release --features cuda
```

Each successful `Kernel::simulate` (or `simulate_with`) then compiles the kernel for GPU 0,
runs it there on the arguments as they were before the simulation, and appends one line to
the log: `SAME` when every buffer equals the simulator's bit for bit, `DIFF` with the
argument, the number of differing elements and (for floats) the largest difference, `SKIP`
when the device cannot run the kernel (with the reason), or `FAIL` when the launch failed. A
`RUN` line is written before each launch, so a kernel that never returns on the device is
the last line of the log (the run waits on it: there is no time limit, so end that test and
run the rest). Nothing panics, and the tests' own results are unchanged, so the
whole simulator suite becomes a check of the simulator's model of that GPU.
(`cute::cuda::crosscheck` describes what can legitimately differ.)

`CUTE_RS_CROSSCHECK_DUMP=<directory>` also saves the buffers of every kernel that differs
(before the run, from the device and from the simulator). With
`CUTE_RS_CROSSCHECK_REPLAY=<directory>` such a directory stands in for the device on any
machine: a simulated kernel that has a saved run with the same inputs is compared with what
the device gave then, which is how a correction to the simulator is checked without the GPU.

## Checking the FlashInfer ports against FlashInfer

[`flashinfer-check/`](../flashinfer-check) gives FlashInfer itself and the `cute-flashinfer`
ports the same inputs on the same GPU and compares what they return, so the expected output
is FlashInfer's and not a reference written beside the port. On an RTX 2000 Ada (SM89) with
FlashInfer 0.7.5, all 1,092 outputs over 18 modules agree (584 of them bit for bit). It found
three faults in the ports, since fixed: NaN from single decode with a sliding window, NaN
from a causal prefill split into chunks, and the large-vocabulary top-k sampling path drawing
another token than FlashInfer's.
[Its README](../flashinfer-check/README.md#results) has the table, the rules and what is not
compared.

## Results on a B200

One rented B200 (sm_100a, driver 595.112, LLVM 22.1, `ptxas` 12.8), on 2026-11-25.

**Device tests.** `tests/gpu_blackwell.rs` and `tests/gpu_blackwell_more.rs` pass: the dense
GEMM (one CTA, a CTA pair, a pair with multicast), the tutorial GEMMs (plain, warp-specialised,
persistent, with cluster launch control and with preferred and fallback clusters), the mixed-input and NVFP4 GEMMs, the
custom-epilogue, `cute_ext` persistent and task-scheduling GEMMs and the TMA copy.
`tests/gpu_blackwell_soak.rs` passes too: every configuration launched 200 times on one
shape and once on about ten more, each result equal to the CPU reference.

**Simulator against device.** The test suite was run with `CUTE_RS_CROSSCHECK_GPU`
(80 of the 119 test files: the GPU-only ones ran on their own, and the last 19, which cover features already run on an SM89 GPU, were left out). Of the simulated kernels the
device could run, 15,140 runs gave exactly the simulator's buffers and 382 did not; 2,522
were skipped (built for another architecture, mostly sm_110a, sm_90a and sm_120a) and 3
failed to launch (the scatter4 store twice, and a shared-memory size above the device's
limit). Most of the differences are of kinds that are not errors (the order a
cluster scheduler works in, approximate instructions, NaN payloads, rounding inside the
tensor cores, memory read without being written, TMA descriptor bytes). The ones that were
errors are in [limitations.md](limitations.md): `ldmatrix`'s 4-bit unpacking, `fma` with
`oob` on NaN operands, `aggr_smem_size` and a negative coordinate in the scatter4 store are
corrected (the first three checked against the device's saved outputs with
`CUTE_RS_CROSSCHECK_REPLAY`), and a CTA pair's block-scaled MMA with M = 128 is refused
until its scale-factor layout is known. One more became a rule: an MN-major tcgen05 operand must pair
`MN_SW128_32B` with 32-bit elements and the other MN-major layouts with narrower ones (the
device read the other pairings as zeros), and the descriptor now refuses the rest.

**Speed** at 8192 x 8192 x 8192, Float16 operands and result, Float32 accumulator. cute-rs
is the fastest of 10 launches, timed on the host; CuTe DSL 4.8.1 is the same example run by
NVIDIA's package, an average over 20 launches from CUDA events (Python's launch overhead
included); cuBLAS is `torch.matmul`.

| Kernel | cute-rs | CuTe DSL | cute-rs takes longer by |
|---|---|---|---|
| Tutorial GEMM, static persistent scheduler (`fp16_gemm_3`) | 0.68-0.70 ms (1,580-1,608 TFLOP/s) | 0.53 ms | 29-31% |
| Tutorial GEMM, cluster launch control (`fp16_gemm_3_1`) | 0.69-0.70 ms | 0.60 ms | 15-16% |
| Tutorial GEMM, preferred and fallback clusters (`fp16_gemm_4`) | 0.67-0.68 ms | 0.56 ms | 20-21% |
| Dense GEMM, one CTA | 1.43-1.45 ms | 1.03 ms | 39-42% |
| Dense GEMM, CTA pair | 1.55-1.57 ms | 0.85 ms | 82-83% |
| Dense GEMM, CTA pair with multicast | 1.72-1.75 ms | 0.97 ms | 78-81% |
| cuBLAS | 0.70 ms (1,571 TFLOP/s) | | |

The NVFP4 GEMM takes 0.46-0.52 ms in cute-rs (about 2,100 TFLOP/s); CuTe DSL's example
reports 0.21 ms, but with a cold cache and a different iteration count, so the two are not
like for like. Raising LLVM's loop-unrolling limit (which unrolls the TMA load loops)
changed none of the cute-rs times. Why CuTe DSL's builds are faster is not known yet.

## Results on an H100

One rented H100 (sm_90, driver 595.112, LLVM 22.1, `ptxas` 12.8), on 2026-11-26: the first
time the Hopper kernels ran on a Hopper GPU.

**Device tests.** `tests/gpu_hopper.rs` passes: the WGMMA dense GEMM (clusters, every major
mode, two warpgroups, batches, Float16, BFloat16, FP8 and 8-bit integers, whole numbers and
fractions), the persistent GEMM with the device's cluster count, and the FMHA forward. The
first run of the whole suite gave 1,950 passes and 4 failures, none of them a fault in a
kernel:

* A GEMM with a Float16 accumulator over K = 1024 was outside the test's tolerance. The
  simulator and the device give the same bits for it; it is the f64 reference that does not
  round at every MMA. The tolerance now allows for that rounding, and a simulator test covers
  the long K (`float16_accumulator_over_a_long_k`).
* The driver refused the packed 4- and 6-bit tensor-map types, which a test expected it to
  take: they came with sm_100. The test now expects the refusal below sm_100, and the error
  says which GPU they need.
* `MulticastGroup::new` failed for one device. The driver makes a team of 2, 4 or 8 devices
  and refuses a team of one (the error a single B200 gave), so a team of one is now refused
  with that reason.
* A timing-stability test saw one slow launch while another job was using the GPU. It passed
  when run again.

With these corrected, the four pass on the device.

**Simulator against device.** The whole test suite, the FlashInfer ports included, was run
with `CUTE_RS_CROSSCHECK_GPU`. Of 20,370 kernel runs the device completed, 18,998 gave
exactly the simulator's buffers and 1,239 did not; 133 failed to launch (120 with the packed
4- and 6-bit tensor-map types and 10 with a preferred cluster shape, which need sm_100; 2
multicast kernels on one GPU; 1 above the device's shared-memory limit) and 2,049 were
skipped (built for another architecture, or using an instruction this GPU lacks). Of the
kernels that are Hopper's own (WGMMA, the GEMMs, the FMHA), 597 runs were identical and 49
differed.

None of the differences is an error. They are of these kinds:

* approximate instructions (`ex2.approx`, `lg2.approx`, `tanh.approx`, `rsqrt.approx`, the
  fast `sin`, `cos`, `tan` and `pow`) and what is computed from them. 769 of the differences
  are in the FlashInfer ports, and all but their top-k and routing ones (below) are of this
  kind: a few steps of the output type, 0.004 at most. So are the FMHA's 48;
* rounding inside the tensor cores: `mma.sync` with a Float32 accumulator differs from the
  simulator's sum in the last digits, and one FP8 WGMMA GEMM by up to 0.007 on values up to
  25. The Float16 and BFloat16 WGMMA kernels gave the simulator's bits, on fractions too;
* values that hold addresses or driver encodings: tensor-map descriptor bytes, and WGMMA
  descriptors (the device's shared memory starts 1,024 bytes up);
* results that depend on the order threads run in: pipeline tokens, a compare-and-swap race,
  the order of an unsorted top-k or routing result, and which of several tied scores a top-k
  selection keeps;
* NaN payloads.

One kernel never finishes on the device: `no_back_pressure` of `tests/schedules.rs`, which is
faulty on purpose (its producer laps the consumer, which the test shows the simulator
reporting). The cross-check has no time limit, so that test file was run test by test with
one.

**Speed** at 8192 x 8192 x 8192, Float16 operands and result, Float32 accumulator, timed as
on the B200; cuBLAS is `torch.matmul` and the attention reference is PyTorch's
`scaled_dot_product_attention`. CuTe DSL was not timed here.

| Kernel | cute-rs | Reference |
|---|---|---|
| Dense GEMM, cluster 2 x 1 | 2.23 ms (494 TFLOP/s) | cuBLAS 1.47 ms (748 TFLOP/s) |
| Dense GEMM, cluster 1 x 2 | 2.21 ms (499 TFLOP/s) | |
| Persistent GEMM | 2.15 ms (511 TFLOP/s) | |
| FMHA, causal, batch 4, 16 heads, 4,096 tokens, head dimension 64 | 0.69 ms (199 TFLOP/s) | PyTorch 0.52 ms (262 TFLOP/s) |

So cute-rs's Hopper GEMM takes about 50% longer than cuBLAS, and its FMHA about 30% longer
than PyTorch's attention.
