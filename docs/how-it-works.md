# How it works

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
  ([`Layout::with_strides`](../src/layout/layout.rs)), is now optional; it
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
  ([`validate::ArchSpecific`](../src/validate.rs)): Float32 `redux.sync` is
  refused for anything but sm_100a with "redux.sync.max.f32 needs sm_100a
  (or the sm_100f family with PTX 8.8); the kernel is being compiled for
  sm_100". A requirement is a set of alternatives, each an exact chip
  (`sm_90a`), a family (`sm_100f` or a later chip of its family, CuTe DSL's
  `is_family_of`) or a baseline (`sm_100` or newer), with its own minimum
  PTX ISA version; the target gets the version of the alternative it
  satisfies (the FP4 `cvt` is PTX 8.6 on sm_100a, 8.7 on sm_120a and 8.8
  on sm_103f), and a refusal lists them all. `validate::SM90A_ONLY` and
  `validate::SETMAXNREG` are the exact-chip lists that WGMMA and
  `setmaxnreg` use.
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
  [`src/math/mod.rs`](../src/math/mod.rs). `fastmath=True` uses the hardware
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
  ([`cute::arch::inline`](../src/arch/inline.rs) has an example).
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
  the module in between ([`LoweredModule`](../src/backend/stage.rs)). At default
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
