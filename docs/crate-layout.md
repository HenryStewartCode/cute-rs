# Layout of the crate

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
  ir.rs        the IR (the module root: ValueId, BufferId, Loc; re-exports the
               ir/ files: types, scalar, memory, mma, numeric, asm, op,
               tcgen05, kernel, externs)
  validate.rs  IR verification
  sim/         the CPU simulator, and SimStream with its PDL model;
               exec.rs and exec/: the interpreter (exec.rs holds the
               machine's state and construction; exec/ the scheduler,
               barriers, blocked-thread reports, memory model, instruction
               interpreter and asynchronous copies, and one file each for
               tensormaps, tmem, umma, tmem_cp, wgmma, blockscale, clc and
               distributed)
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
tests/         integration tests; diagnostics/ is one test binary split by
               topic (main.rs and topic modules), and feature_kernels/ holds
               the kernels several binaries share (mod.rs re-exports its
               topic files)
```
