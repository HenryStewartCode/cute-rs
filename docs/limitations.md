# Limitations

* What an H100 (sm_90) showed, from the same run of the simulator's tests
  ([testing.md](testing.md#results-on-an-h100)): no error in a kernel or in the simulator.
  Three things it settled: the Float16 and BFloat16 WGMMA kernels give the simulator's bits,
  on inexact sums too, while an FP8 WGMMA differs in the last digits (0.007 at most on values
  up to 25); the packed 4- and 6-bit tensor-map types need sm_100; and multicast memory needs
  a team of at least two devices.
* What a B200 (sm_100a) showed that the simulator did not, found by running the simulator's
  tests with `CUTE_RS_CROSSCHECK_GPU` ([testing.md](testing.md#results-on-a-b200)):
  * Still open: a CTA pair's block-scaled MMA with M = 128 (the PTX ISA's Layout B, the 2x2
    data path, as `MmaMXF4NVF4Op::new(.., (128, 128), CtaGroup::TWO, ..)` uses). With B's
    scale factors staged as for M = 256, the device gave the columns of the second half of
    B the factors of the first. Where that layout wants them is not known, so the simulator
    refuses such an MMA, and `make_tmem_layout_sfb` refuses a pair. One CTA, and a pair with
    M = 256, agree with the device.
  * Corrected in the simulator, and checked against the device's saved outputs:
    `ldmatrix`'s 4-bit unpacking zero-extends (it had the value two bits higher);
    `fma` with `oob` gives zero only for the out-of-bounds NaN `0x7fff` that TMA's NaN fill
    writes, and the canonical NaN for any other NaN (it gave zero for every NaN);
    `aggr_smem_size` is not rounded to the allocation unit.
  * Corrected, but not run on the device again: a TMA store takes no negative coordinate.
    The scatter4 test kernel had one, which the simulator let through for scatter4 and the
    device stopped on with an illegal-instruction error (the PTX ISA allows it on sm_107f
    only). The simulator now reports it.
  * These differ by nature and are not errors: which cluster cluster launch control cancels
    first, and which cluster shape the device picks; how often a `try_wait` succeeds at once;
    a collector `USE` after its source changed (the device reloads the operand, which the
    simulator reports by default); tensor memory read without having been written;
    approximate instructions (`ex2.approx` and the fast maths) and NaN payloads; sums whose
    order depends on timing, and tensor-core sums of values that round; the bytes of a TMA
    descriptor; a dynamic shared-memory size above what the device allows.
* A tcgen05 operand in shared memory that is MN-major takes `MN_SW128_32B` exactly when its
  elements are 32 bits wide. A B200 reads the other pairings as zeros, so the descriptor
  refuses them when the kernel is built.
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
  kernel fills: CuTe DSL's sparse element types are not implemented,
  and only the TID metadata format (sparsity selector 0) exists. The
  simulator reports metadata that is not one of the six ordered index
  pairs. The 8-bit `ldmatrix` shapes (`.m16n16`, `.m8n16`) agree with a
  B200 (they need sm_100a or sm_120a), following the PTX ISA's fragment
  figures. The 4- and 6-bit unpacking zero-extends each value into its
  byte, as the device does. `mma.kind::f8f6f4` takes a 4-bit value two
  bits higher, so bytes loaded that way are shifted left by two before
  such an MMA (CuTe C++'s `fp4_shift_A`/`fp4_shift_B`; cute-rs has no
  helper for it, as `gemm` places the values of a typed FP4 fragment
  itself). The signed `.m8n16.s8.s4`
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
  has `cpasync`, the operand major modes and `warpgroup`; the `warp`
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
  (`mapa` loads are supported; the port keeps its own-tile form).
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
  [Compile options](compile-options.md)).
  Cubins and SASS need a CUDA toolkit with `ptxas` (and `nvdisasm`), or on a
  GPU machine the driver's linker for the cubin; without them the driver JIT
  compiles the PTX and a requested cubin or SASS is reported as unavailable.
* Kernels have no recursion or unstructured control flow, and call only
  external device functions (`ExternFn`, `DeviceFunction`; see
  [External device functions](runtime.md#external-device-functions-and-multicast-memory)):
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
* Raw pointers ([`make_ptr`](../src/raw_ptr.rs)) address memory through
  the simulator's address map: global buffers at per-launch bases from
  `SIM_GLOBAL_BASE` (256-byte aligned, a gap after each), shared buffers at
  their start in the block's shared memory (the same offset in every
  block), per-thread buffers at their start in the thread's local memory,
  and generic shared and local addresses in windows of their own. An
  address in no buffer is reported as a wild address; one past a buffer's
  end (within its gap) is out of bounds for that buffer. The addresses
  differ from the GPU's, so kernels should not write them to outputs.
  Boolean raw pointers are refused (point at `Uint8` flags), and so are
  constant-memory and distributed-shared-memory ones.
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
  SharedSpace.CLUSTER`) is reached through `map_dsmem_ptr`. The newer PTX forms are
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
  implemented. The tensormap form of `prefetch` (`tensormap=`) is
  `PrefetchOptions::tensormap`; `in_param_space=` is not needed, as the address is generic.
* The vector forms of the atomics and reductions (`.v2`, `.v4`, `.v8` of
  f32, f16 and bf16, sm_90) are not implemented, nor are the
  `.L2::cache_hint`, `inc`, `dec` and 128-bit forms.
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
  arrays and words; CuTe DSL's take and return MLIR vectors, and its int4 helpers sub-byte tensors. `.oob`
  (fma's out-of-bounds modifier) gives `+0` where a multiplicand is the
  "OOB NaN" that TMA's NaN fill writes, a special NaN whose encoding
  neither the PTX ISA nor the CUDA documentation gives: the simulator takes
  it to be `0x7fff`, the pattern the fill writes on a GPU, and gives the
  canonical NaN for any other NaN multiplicand, as a B200 does (CUTLASS's
  host reference, `guarded_multiply_add`, zeroes every NaN). Whether the
  device also takes `0xffff` for it is not known. A B200 has run `.oob`,
  the bf16x2 `add` and `sub` and `ex2.approx.ftz.bf16x2` (sm_90) against
  the simulator; `ex2.approx` differs from the simulator's exactly rounded
  value in the last digit of some results, as an approximation may. The s2f6x2
  int8-to-BFloat16 conversions need PTX 9.1 (CUDA 13.1), which only LLVM 23
  emits and `ptxas` 12.8 cannot assemble, so they are simulated and checked
  as MLIR (and as PTX text with LLVM 23; LLVM 21 and 22 are refused, see
  Compiling above); `cvt_i8_bf16_intrinsic` uses them only when
  asked (`I8ToBf16Path::S2f6x2`), where CuTe DSL picks them for sm_100a-class
  targets, since a kernel's target is chosen after it is built.
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
* What can be checked only with the simulator and `ptxas`, since the test GPU is an SM89:
  the overlap of programmatic dependent launches (the driver accepts `use_pdl` on SM89 and
  runs the kernels in order; the `SimStream` model checks the accesses before
  `griddepcontrol_wait`), programmatic events triggered by `griddepcontrol_launch_dependents`
  (on SM89 the event is recorded as the blocks exit), preferred clusters (SM100; the
  simulator runs preferred, fallback and mixed shapes, and the launcher refuses them before
  SM100), grid-constant by-value parameters (`nvvm.grid_constant`: emitted, assembled and
  simulated, not launched on a GPU), and the
  sm_100f, sm_103, sm_107, sm_110, sm_120f and sm_121 targets (compiled and, where the
  toolchain has them, assembled; sm_107 is emitted by no LLVM yet).
* Pipelines: CuTe DSL's pipelines need sm_90 (`mbarrier.try_wait`, `fence.mbarrier_init`).
  A kernel built for a target below sm_90 (`KernelOptions::target(Arch::Sm89)`, or
  `CUTE_RS_ARCH`) instead gets `MbarrierArray` waits that spin on `mbarrier.test_wait`
  (sm_80), `try_wait` probes with it, and `pipeline_init_arrive` leaves the init fence out (the
  CTA barrier of `pipeline_init_wait` publishes CTA-local mbarriers), so the mbarrier-only
  pipelines run on SM80 to SM89. Without a target, they emit what CuTe DSL emits, and compiling
  for sm_89 fails at the line that waits. The `TCGen05Mma` arrival is `tcgen05.commit`, which has no SM89 form; the cluster, peer-CTA and TMA forms are checked with the simulator and
  `ptxas` only.
* TMA: every TMA operation (descriptors built on the host and hidden per atom, tile
  loads, multicast loads, stores and reduce stores, bulk copies, prefetches, in-kernel
  descriptor copies and `tensormap.replace` updates, the tensormap-proxy fences, and the
  ported tutorials) is verified only by the simulator (against plain Rust), by `mlir-opt`
  (LLVM 21, 22 and 23) and by `ptxas` 12.8 for sm_90/sm_90a: no SM90 GPU has run any of
  it, and the SM89 GPU's driver refuses to encode tensor maps. The simulator models TMA from
  the PTX ISA with its own descriptor encoding, so it cannot catch a difference between that
  and the driver's opaque format (`tensormap.replace` field units, say). Only the tile mode
  without interleave is simulated. The im2col, gather4/scatter4, byte-mask and FP4/FP6
  forms are simulated and assembled by `ptxas` (sm_90/
  sm_90a and sm_100a); the sm_107f override copies are simulated and checked as MLIR text
  only (no toolchain has sm_107f). Refused: CTA-pair loads (`.cta_group`),
  `.im2col::w`/`.im2col::w::128`, the `s128b_atom_*` swizzles, the
  multicast split of a gather4 box, the FP6 store form `.b6p2x16`, and the sm_107f-only
  `.im2col_no_offs::w` and `.report_mechanism`. `tensormap.replace`
  needs an architecture-specific target (sm_90a, sm_100a, sm_110a, sm_120a or their
  families); its fields other than the address, extents and strides are not exposed.
* SM90: WGMMA (`warpgroup`), `stmatrix`, `setmaxnreg`, distributed shared memory and
  TMA are verified only by the simulator (whose models come from the PTX ISA,
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
  undocumented. On an H100 the Float16 and BFloat16 kernels gave the same bits, on inexact
  sums too; an FP8 kernel did not (see the top of this page).
* TVM-FFI callables take positional DLPack tensors and scalars only (no TVM-FFI `Array`,
  `Map` or dataclass objects, defaults or keyword arguments), check their arguments on the
  host rather than in compiled code, and build one callable (and load one module) per
  calling thread. They have been called from C and from Python (`tvm_ffi` 0.1.14 with
  torch) on the SM89 GPU. `benchmark` times with CUDA events only (no CUPTI or NVTX).
* Tensor memory is modelled per CTA: each simulated CTA has its own 128 x
  576 cells, so co-resident CTAs never share an SM's TMEM (nor contend for its columns or
  the allocation permit). An allocation that can never be served is reported as a deadlock.
  The sm_107 exclusive forms are inline PTX checked only as MLIR text and in the simulator
  (no installed LLVM or ptxas knows PTX 9.4); their qualifier order is unverified.
* Blackwell: every `tcgen05` instruction, TMEM operation, UMMA MMA, `tcgen05.cp`, CLC
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
* `retile` with a one-value `make_tiled_copy_c_atom` copy: the copy's repeats over an m16n8
  accumulator run over the row block first, while the fragment holds c0, c1, c2, c3, so
  `retile(acc)` followed by `copy` stores c1 and c2 in each other's places. Pick the fragment
  slot for each repeat yourself (as `tests/tiled_copy_c_atom.rs` does) or use an atom of two
  values or more. Not compared with C++ CuTe on this case.
