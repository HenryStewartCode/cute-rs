# Runtime and interop

## Runtime, interop and testing

What CuTe DSL offers around a compiled kernel, and where it is in cute-rs. Everything that
launches needs the `cuda` feature.

| CuTe DSL | cute-rs |
|---|---|
| Compile options: `cute.compile[...]`, `options="..."`, `CompileOptions` | `CompileOptions` (builder methods, `CompileOptions::parse` for the option strings, `with_env`), every option and spelling in [Compile options](compile-options.md), including `--enable-tvm-ffi` |
| Environment variables `CUTE_DSL_*` | `CUTE_RS_*` with CuTe's suffix: `COMPILER_OPT`, `ARCH`, `LINEINFO`, `KEEP` (and the deprecated `KEEP_IR`, `KEEP_PTX`, `KEEP_CUBIN`, `KEEP_SASS`), `DUMP_DIR`, `PRINT_IR`, `DEBUG`, `ENABLE_ASSERTIONS`, `WARNINGS_AS_ERRORS`, `WARNINGS_IGNORE`, `ENABLE_TVM_FFI`, `ENABLE_PYIR` (no effect); the caches' `JIT_CACHE_MAX_ELEMS`, `NO_CACHE`, `CACHE_DIR`, `DISABLE_FILE_CACHING`, `FILE_CACHE_MAX_ENTRIES`, `FILE_CACHE_MAX_BYTES`; the logger's `LOG_TO_CONSOLE`, `LOG_TO_FILE`, `LOG_LEVEL`; and `CUTE_RS_PTXAS` for the assembler. Explicit options win over them. `CUTE_DSL_LOG_AUTOTUNE` has no counterpart (`Tuned::trials` has every configuration's time or reason) |
| `.launch(grid=, block=, cluster=, smem=, stream=, use_pdl=, cooperative=, fallback_cluster=, launch_completion_event=, programmatic_event=, preferred_smem_carveout=, ...)`, `max_number_threads=`, `min_blocks_per_mp=` | `LaunchConfig::new(grid, block)` with `with_cluster`, `with_smem`, `with_dynamic_smem`, `with_pdl`, `with_cooperative`, `with_fallback_cluster`, `with_smem_carveout`; `LaunchBuilder::stream`, `launch_completion_event`, `programmatic_event`; `KernelOptions::max_threads`, `min_blocks_per_sm` (the rows of [CuTe DSL → cute-rs](cutedsl-mapping.md)) |
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

## Several kernels in one module

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

## External device functions and multicast memory


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

### Blackwell pipelines and the TMEM allocator

| CuTe DSL | cute-rs |
| --- | --- |
| `cutlass.pipeline.PipelineTmaUmma.create(*, num_stages, producer_group, consumer_group, tx_count, barrier_storage=None, cta_layout_vmnk=None, mcast_mode_mn=(1,1), enable_multicast_signaling=False, defer_sync=False, name='')`; `producer_acquire(state, token, *, expected_tx=None)` | `pipeline::PipelineTmaUmma::create(n, producer, consumer, tx, PipelineTmaUmmaOptions { .. })`; the `Pipeline` trait (`make_participants`, the handles), `PipelineBarriers`; `expected_tx` is the acquire option (`acquire_with(token, Some(tx))`). The leader CTA arms the full barrier, releases are elected `tcgen05.commit`s |
| `PipelineAsyncUmma.create(...)`, `PipelineUmmaAsync.create(...)` (`producer_tail` on the even CTA) | `pipeline::PipelineAsyncUmma::create` and `PipelineUmmaAsync::create` with `PipelineUmmaOptions`. The 2-CTA masks and `cta_group::2` are computed from the `V` mode with CuTe DSL's formulas (leader from `block_idx.x % V`, the others from the cluster rank), checked on `(2, 2, 1)` and `(4, 1, 1)` clusters and simulated in `tests/cta_pair_pipelines.rs`. Without pairs (`V = 1`) a CTA is its own peer. `PipelineUmmaAsync::producer_tail` advances the state on every CTA (CuTe DSL only inside its leader branch) |
| `PipelineTmaMultiConsumersAsync.create(*, num_stages, producer_group, consumer_group_umma, consumer_group_async, tx_count, barrier_storage=None, cta_layout_vmnk=None, mcast_mode_mn=(1,1), tidx=None, enable_multicast_signaling=False, defer_sync=False, force_deprecated_per_lane_signaling=None, name='')`; `consumer_release(state, op_type=PipelineOp.TCGen05Mma)`; `make_participants()` -> `(producer, umma_consumer, None)`; `_init_empty_barrier_arrive_signal_2sm` | `pipeline::PipelineTmaMultiConsumersAsync::create(n, producer, umma, async_, tx, PipelineTmaMultiConsumersAsyncOptions { .. })`; `consumer_release_op(&state, PipelineOp::..)` and `ConsumerHandle::release_op(op)` (the `Pipeline` trait's `consumer_release` is the MMA's); `make_participants` returns the first two; `init_empty_barrier_arrive_signal_2sm`. One empty-barrier array counts both groups, viewed as `TCGen05Mma` and `AsyncThread` (`sync_object_empty_umma`/`_async`). The deprecated per-lane signalling is `per_lane_signaling: Option<bool>` (`None` auto-detects it as CuTe DSL does and logs a warning instead of a `DeprecationWarning`; refused in a cluster). Async signalling lanes need a `sync_warp` after the warp's reads |
| `cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A(cluster_shape_mnk, atom_thr_id)`, `_B`, `_SFB` | `utils::sm100::cluster_shape_to_tma_atom_a/b/sfb(cluster_shape, &thr_id) -> Result<Sm100TmaLoadOp, String>`; `Sm100TmaLoadOp::{Tile, Multicast}` converts into a `CopyOp`. Errors (CuTe DSL's `ValueError`s) for dynamic shapes, an M extent that is not a multiple of the SM count, and SM counts other than 1 and 2 |
| `tiled_mma.thr_id` | `TiledMma::thr_id()` and `MmaAtom::thr_id()` (`1:0`, or `2:1` for a CTA pair) |
| `cutlass.memory.TmemAllocator(...)` (and the deprecated `cutlass.utils.TmemAllocator`): `check_valid_num_columns`, `allocate`, `wait_for_alloc`, `retrieve_ptr(dtype)`, `reserve`, `relinquish_alloc_permit`, `free(ptr, num_columns=0)` | `memory::TmemAllocator::new(named_barrier, TmemAllocatorOptions { .. })` with an `Arch` for the target string; `retrieve_ptr::<T>()`; the column count is a build-time number (`allocate` and `free` take `&mut self`); not `Carry` |
| `TmemBufferPool`: `base_ptr`, `total_cols`, `current_offset`, `remaining_cols`, `allocate(size, dtype)`, `allocate_tensor(layout, dtype)` | `memory::TmemBufferPool`, `allocate::<T>(cols or &Layout)`, `allocate_tensor::<T>(&layout)` |
| `compute_tmem_cols_from_layout(layout, dtype)`, `get_num_tmem_alloc_cols(tensors, rounding=True, *, arch)` (and the deprecated `cutlass.utils` one) | `memory::compute_tmem_cols_from_layout(&layout, DType)`, `memory::get_num_tmem_alloc_cols(&[&t1, &t2], rounding, arch)` (`utils::get_num_tmem_alloc_cols` for sm_100) |
| `tcgen05.is_tmem_load(atom)`, `is_tmem_store`, `get_tmem_copy_properties(atom)`, `find_tmem_tensor_col_offset(t)` | the same names in `tcgen05` (`TmemPacking` for the Pack/Unpack result), and `CopyOp::is_tmem_load`, `is_tmem_store`, `tmem_copy_properties`. `find_tmem_tensor_col_offset` takes static layouts only |

The simulator reports an `mbarrier.arrive` (a pipeline's release) by a warp whose
`tcgen05.ld` has not completed (`fence_view_async_tmem_load` missing before the release).

## Host tensor helpers

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
