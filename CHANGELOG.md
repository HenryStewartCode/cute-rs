# Changelog

No version has been tagged yet; 0.1.0 will be the first.

## Unreleased

### Fixed

- A tiled MMA's thread-value layouts write the thread mode flat, `(4,8,4,2)`, as CuTe does (it
  was nested, `((4,8,4),2)`: the same map).
- `partition_fragment_a` of a tcgen05 MMA with A in tensor memory gives the tensor-memory
  fragment (it gave a register fragment).
- The layout algebra, where a comparison with CuTe DSL 4.8.1 on 20,500 recorded calls showed
  cute-rs differing (`cutedsl-check/`):
  - `complement` divides with floor and refuses only overlapping modes, as C++ CuTe does; it
    used to refuse any strides that do not divide each other, and so did every divide and
    product built on it;
  - `composition` follows C++ CuTe's walk over static values: an extent-1 mode keeps the stride
    the walk reaches (it was 0), and a stride smaller than a mode that does not divide it is
    accepted when the result is the composition, checked value by value;
  - `left_inverse` is built directly, as CUTLASS 4.8.1 builds it (`left_inverse(4:3)` is
    `(3,4):(0,1)`);
  - `zipped_product`, `tiled_product` and `flat_product` no longer pad to a common rank, and
    `raked_product` coalesces each mode;
  - `logical_divide` takes the complement in the layout's coalesced shape;
  - `make_ordered_layout`, `make_layout_tv` and `max_common_layout` give an extent-1 mode
    stride 0.
- `cute-flashinfer`: the error measures of the GEMM and MoE tests no longer drop a NaN, and
  the modules share one `max_error` (`flashinfer::check`).
- About twenty statements in the documentation and comments that a feature was not built yet
  when it is (the tensor-map form of `prefetch`, `CtaGroup::TWO` loads, `param_struct`,
  `utils::sm90`, the byte-masked bulk store and others).
- Launch scalars with a `multiple_of` requirement are checked as signed values (an `Int32`
  of -1 used to pass as a multiple of 3).
- `jit_options()` keeps the `ptxas` flags the driver's JIT can take when another flag has no
  JIT equivalent, instead of dropping them all.
- A value that is always zero (`x * 0`) counts as divisible by anything in alignment checks.
- A CUDA graph replay fails, instead of using freed device memory, once a `DeviceBuffer`
  its capture used has been dropped.
- A tcgen05 MN-major shared-memory operand is refused unless it pairs `MN_SW128_32B` with
  32-bit elements and the other MN-major layouts with narrower ones: a B200 reads the other
  pairings as zeros, which the simulator did not show.
- The Hopper and Blackwell device tests compile a kernel for its own target (the sm_90 TMA
  copy failed to compile on an sm_100 GPU).
- The simulator, where a B200 disagreed with it: `ldmatrix`'s 4-bit unpacking zero-extends
  each value (it was two bits higher); `fma` with `oob` gives zero only for the out-of-bounds
  NaN `0x7fff` and the canonical NaN for any other; `aggr_smem_size` is not rounded up to the
  allocation unit; a scatter4 store with a negative coordinate is reported, like any other
  store's; and a CTA pair's block-scaled MMA with M = 128 is refused, as its scale-factor
  layout for B is not the one modelled.
- `cute-flashinfer`, three faults that the check against FlashInfer found:
  - single decode with a sliding window returned NaN when the cache was longer than the
    window (the plan split the whole cache, but the kernel reads only the window);
  - a causal prefill split into chunks returned NaN for the queries that see nothing of one
    of their chunks, as after a cached prefix (such a chunk's share is now zero);
  - top-k then top-p sampling on a vocabulary of 65,536 or more (k up to 256) now sorts the
    kept values before it draws, as FlashInfer does, so it returns FlashInfer's token for
    the same seed (`compact_sort_kernel`).
- `cute-flashinfer`: the `max_error` helpers of `decode`, `prefill`, `attention`, `mla`,
  `rope`, `sparse` and `xqa` return NaN when an output is NaN. They dropped it, so a NaN
  output passed the tests that used them; that hid the prefill fault above.
- What the first run on an H100 showed (no fault in a kernel):
  - the Hopper GEMM check allows for a Float16 accumulator's rounding over a long K
    (`f16_accumulator_rounding`): at K = 1024 the device and the simulator, which agree bit
    for bit, were outside the old tolerance;
  - `MulticastGroup::new` refuses a team of one device with the reason (the driver makes
    teams of two or more, and gave a bare invalid-value error for one);
  - a tensor map of a packed 4- or 6-bit type on a GPU below sm_100 fails with the GPU it
    needs, and the driver test expects that refusal.

### Added

- `cutedsl-check atoms`: the same comparison with CuTe DSL for atoms and partitioning, on 9,604
  recorded cases. `docs/testing.md` has the table and what still differs.
- `cutedsl-check/`: records NVIDIA CuTe DSL's answers to seeded random layout-algebra calls
  and replays them in cute-rs. On two seeds (41,000 calls) cute-rs gives the same answer on
  every call both libraries answer; `docs/testing.md` has the table.
- Results of the first run on an H100 (`docs/testing.md`): the Hopper device tests pass, the
  simulator agrees with the device on 18,998 of 20,370 kernel runs with no difference that is
  an error, and the Hopper GEMM and FMHA are timed against cuBLAS and PyTorch.
- `flashinfer-check/`: a check of the `cute-flashinfer` ports against FlashInfer itself.
  FlashInfer's Python package and the ports are given the same inputs on one GPU and their
  outputs compared under a rule per output (the same bytes, one representable value apart,
  or a tolerance for sums). On an RTX 2000 Ada, all 1,092 outputs over 18 modules agree,
  after the three `cute-flashinfer` fixes listed above, which it led to.
- On-device tests for the Blackwell custom-epilogue, `cute_ext` persistent and
  task-scheduling GEMMs (`tests/gpu_blackwell_more.rs`; they need an sm_100 GPU).
- Tests of `make_tiled_copy_c_atom` with one- and two-value atoms.
- `CUTE_RS_CROSSCHECK_GPU=<log file>` (with the `cuda` feature): every simulated kernel is
  also run on GPU 0 and compared with the simulator, one `SAME`/`DIFF`/`SKIP`/`FAIL` line each.
  `CUTE_RS_CROSSCHECK_DUMP=<directory>` saves the buffers of the kernels that differ, and
  `CUTE_RS_CROSSCHECK_REPLAY=<directory>` compares the simulator with such saved runs on a
  machine without the GPU.
- `tests/gpu_blackwell_soak.rs`: each Blackwell GEMM launched 200 times and on about ten
  shapes, every result checked (they need an sm_100 GPU; all pass on a B200).
- TF32 and 8-bit tcgen05 tiles with MN-major operands in the tile tests.

### Changed

- The README says how the project was built and separates what has been compared with
  CuTe DSL, FlashInfer and real GPUs from what only its own tests cover.
- Comments and documentation no longer cite the project's internal milestone names.
- Lint allowances: `too_many_arguments` and `type_complexity` are allowed once, in
  `Cargo.toml`, and unused-code allowances once per shared test module, each with its reason.
- `scripts/pod-run.sh --worktree` takes the paths to leave behind from `POD_EXCLUDE`.
- Floating-point multiply-adds contract into `fma` by default
  (`CompileOptions::fp_contract`); bit-for-bit comparisons with the simulator turn it off.
- The README is a short introduction; the reference material moved to `docs/`.
- The errors of compile options that cannot be set by option string no longer name internal
  work items.
- `scripts/gpu-validate.sh` takes `LLVM_VERSION` to choose the LLVM it installs.
- The task-scheduling copy examples (`ts_copy_tma`, `ts_merge_copy`) set the tutorials'
  per-task register counts and prefetch the TMA descriptor.
