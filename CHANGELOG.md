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

### Added

- On-device tests for the Blackwell custom-epilogue, `cute_ext` persistent and
  task-scheduling GEMMs (`tests/gpu_blackwell_more.rs`; they need an sm_100 GPU).
- Tests of `make_tiled_copy_c_atom` with one- and two-value atoms.

### Changed

- Floating-point multiply-adds contract into `fma` by default
  (`CompileOptions::fp_contract`); bit-for-bit comparisons with the simulator turn it off.
- The README is a short introduction; the reference material moved to `docs/`.
- The errors of compile options that are not available yet no longer name internal work
  items.
- `scripts/gpu-validate.sh` takes `LLVM_VERSION` to choose the LLVM it installs.
