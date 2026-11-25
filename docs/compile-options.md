# Compile options

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
| `EnableTVMFFI` | `--enable-tvm-ffi`, `CUTE_DSL_ENABLE_TVM_FFI` | `enable_tvm_ffi(true)`, `CUTE_RS_ENABLE_TVM_FFI`: recorded on the options (`get_enable_tvm_ffi`), with the same PTX. The callable is the Rust wrapper `cute::tvm_ffi::TvmFfiFunction`, which `TvmFfiFunction::compile` builds with the option set and `tvm_ffi_export!` exports from a `cdylib` ([Runtime, interop and testing](runtime.md#runtime-interop-and-testing)). |
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
