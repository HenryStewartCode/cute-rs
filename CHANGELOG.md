# Changelog

No version has been tagged yet; 0.1.0 will be the first.

## Unreleased

### Fixed

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

### Added

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

- Floating-point multiply-adds contract into `fma` by default
  (`CompileOptions::fp_contract`); bit-for-bit comparisons with the simulator turn it off.
- The README is a short introduction; the reference material moved to `docs/`.
- The errors of compile options that are not available yet no longer name internal work
  items.
- `scripts/gpu-validate.sh` takes `LLVM_VERSION` to choose the LLVM it installs.
- The task-scheduling copy examples (`ts_copy_tma`, `ts_merge_copy`) set the tutorials'
  per-task register counts and prefetch the TMA descriptor.
