# cute-rs

[![CI](https://github.com/HenryStewartCode/cute-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/HenryStewartCode/cute-rs/actions/workflows/ci.yml)

A CuTe-style GPU kernel DSL for Rust.

You write GPU kernels as ordinary Rust, using the same building blocks as NVIDIA's
[CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) and CuTe DSL: hierarchical
layouts and their algebra, tensors over global, shared and register memory, and copy and
MMA atoms tiled across threads. A built kernel can be

* **simulated on the CPU** by a checking simulator, which reports shared-memory races,
  missing waits, out-of-bounds and misaligned accesses and barriers that cannot complete,
  each with the Rust source line;
* **lowered to MLIR** using only upstream dialects;
* **compiled to PTX** with a stock `mlir-opt` (no CUDA toolkit needed);
* **run on an NVIDIA GPU** through the CUDA driver API (the `cuda` feature).

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

## Getting started

You need:

* Rust 1.88 or newer (edition 2024).
* To produce PTX: `mlir-opt` from an LLVM with the NVPTX backend. LLVM 22 or newer is
  recommended (some Hopper and Blackwell kernels need it; 21 works for the rest). On macOS
  `brew install llvm`; on Debian or Ubuntu the `mlir-<version>-tools` package from
  apt.llvm.org. Set `CUTE_MLIR_OPT` if it is not found on its own.
* To run on a GPU: Linux, an NVIDIA driver and an SM80 or newer GPU. `libcuda` is loaded at
  run time, so building needs no CUDA installation.

```bash
cargo test                                        # the simulator, MLIR and PTX tests
cargo run --example elementwise_add               # an example, on the simulator
cargo run --example run_on_gpu --features cuda    # the same pipeline on a real GPU
```

To use it from another crate (the library is called `cute`):

```toml
[dependencies]
cute-rs = { git = "https://github.com/HenryStewartCode/cute-rs" }
```

## Status

cute-rs covers the features of CuTe DSL 4.8.1, apart from three left out by design: the
`frontend-next` tracing frontend, and `cute-ext`'s compiler pipeline and entry points
(cute-rs builds kernels from Rust closures instead). Every feature is tested on the
simulator, and the generated PTX is assembled by `ptxas` for each architecture it targets
(except SM107, which no released CUDA toolkit assembles yet).

What has run on real GPUs, and what has not:

| Architecture | Hardware tested | Status |
| --- | --- | --- |
| SM80 / SM89 (Ampere, Ada) | an SM89 GPU | The full GPU suite passes |
| SM120 (consumer Blackwell) | RTX 5090 | The full GPU suite passes, including TMA, the SM120 MMAs and the block-scaled warp MMAs |
| SM90 (Hopper) | none yet | WGMMA, the Hopper GEMMs and FMHA are checked by the simulator and `ptxas` only |
| SM100 (datacenter Blackwell) | B200 | The Blackwell device tests pass (tcgen05 GEMMs with and without CTA pairs, mixed-input, NVFP4, custom epilogues, task scheduling), as does a 200-launch stress test of each; the simulator's test suite was compared with the device kernel by kernel ([docs/testing.md](docs/testing.md#results-on-a-b200)) |
| SM103 / SM107 (Blackwell Ultra, Rubin) | none yet | Checked by the simulator and `ptxas` only (SM107 by the simulator only) |

`tests/gpu_hopper.rs` and `tests/gpu_blackwell.rs` run the Hopper and datacenter-Blackwell
examples on those GPUs against CPU references, and time them (the `#[ignore]`d `bench_*`
tests). With `CUTE_RS_CROSSCHECK_GPU` set, every kernel the simulator tests run is also run
on the GPU and compared. `scripts/gpu-validate.sh` does a whole validation pass on a GPU machine in one go:
it installs Rust and `mlir-opt`, runs the architecture's suite, then the full suite, then the
timings (with cuBLAS alongside when PyTorch is installed), and packs the results.
`scripts/pod-run.sh` drives it on a rented machine over SSH (`start`, `status`, `fetch`).

Known gaps besides the Hopper and datacenter-Blackwell hardware:

* Speed: on a B200 at 8192 x 8192 x 8192 in Float16, the tutorial tcgen05 GEMMs run at about
  1,600 TFLOP/s, level with cuBLAS, but CuTe DSL's own builds of the same kernels are 15 to 30%
  faster, and 40 to 80% faster for the dense GEMM example
  ([docs/testing.md](docs/testing.md#results-on-a-b200)). Hopper is not timed yet.
* On the B200 a few simulated kernels disagree with the device: a CTA pair's NVFP4 scale
  factors for B, `ldmatrix`'s 4-bit unpacking, `fma` with `oob` on NaN operands and the
  scatter4 TMA store. [docs/limitations.md](docs/limitations.md) has the details.
* The JAX (`xla_ffi`) and TVM-FFI bindings are tested against hand-built call frames, not
  under JAX or `libtvm_ffi` themselves.
* A CTA pair's B scale factors (the 2x2 data path of `make_tmem_layout_sfb`) are not laid
  out yet.
* Multicast memory (`MulticastGroup`) fails to create a group on a single B200
  (`cuMulticastCreate` returns an invalid-value error); it has not run on a multi-GPU machine.
* The JIT disk cache refuses a directory other users can write to, which includes network
  volumes that report mode 777; point `CUTE_RS_CACHE_DIR` at local disk there.

[docs/limitations.md](docs/limitations.md) lists each feature's smaller gaps.

## Examples

A few places to start; [docs/examples.md](docs/examples.md) describes all 42.

| Example | What it shows |
|---|---|
| [`elementwise_add`](examples/elementwise_add.rs) | CuTe DSL's first tutorial: a naive kernel, then a tiled one with vector copies. |
| [`sgemm`](examples/sgemm.rs) | CuTe's `sgemm_2`: a GEMM with tiled copies into shared memory and no tensor cores. |
| [`tensorop_gemm`](examples/tensorop_gemm.rs) | An SM80 tensor-core GEMM: `mma.sync`, a `cp.async` pipeline and `ldmatrix`. |
| [`single_warp_pipeline`](examples/single_warp_pipeline.rs) | The kernel this project started from: one warp and a double-buffered `cp.async` pipeline. |
| [`tma_copy`](examples/tma_copy.rs) | CuTe DSL's TMA tutorial: a TMA load and a TMA store over one shared-memory layout. |
| [`hopper_gemm`](examples/hopper_gemm.rs) | CuTe DSL's Hopper dense GEMM: TMA loads, clusters and warpgroup MMAs. |
| [`blackwell_fp16_gemm`](examples/blackwell_fp16_gemm.rs) | CuTe DSL's Blackwell tutorial GEMM: tcgen05 MMAs with the accumulator in tensor memory. |
| [`run_on_gpu`](examples/run_on_gpu.rs) | Compiles the tensor-core GEMM, runs it on a real GPU and checks it (needs `--features cuda`). |

The [`flashinfer/`](flashinfer) crate (`cute-flashinfer`) is a larger body of kernels written
with cute-rs: ports of FlashInfer's attention, sampling and normalization kernels.

## How it works

A kernel is a Rust closure passed to `Kernel::build`. Running the closure records the kernel
as an intermediate representation; layouts, tilings and atoms are worked out while it runs,
the way C++ CuTe works them out at compile time. From that one representation:

1. a **validator** checks it against the target architecture;
2. the **simulator** runs it on the CPU, thread by thread, and checks every access;
3. the **MLIR emitter** writes it in upstream dialects, which `mlir-opt` lowers to PTX;
4. the **CUDA launcher** (the `cuda` feature) loads the PTX and runs it on the device.

[docs/how-it-works.md](docs/how-it-works.md) goes through each step.

## Documentation

| | |
|---|---|
| [How it works](docs/how-it-works.md) | Kernel arguments, the builder, the simulator and the code generator |
| [CuTe DSL to cute-rs](docs/cutedsl-mapping.md) | Every CuTe DSL feature and its cute-rs counterpart: start here if you know CuTe DSL |
| [Examples](docs/examples.md) | What each example shows |
| [Compile options](docs/compile-options.md) | Optimization levels, line information, option strings and environment variables |
| [Runtime and interop](docs/runtime.md) | Streams, graphs, export to C, TVM-FFI, JAX, external functions and host tensors |
| [Testing](docs/testing.md) | What the test suites cover and how to run them on a GPU |
| [Hopper](docs/hopper.md) | TMA, warpgroup MMAs, distributed shared memory and the pipelines |
| [Blackwell and Rubin](docs/blackwell.md) | tcgen05 MMAs, tensor memory, CTA pairs, cluster launch control and block scaling |
| [The primitives layer](docs/primitives.md) | Typed descriptors and the instruction-level API |
| [cute_ext](docs/cute-ext.md) | The experimental extension layer |
| [Limitations](docs/limitations.md) | Each feature's known gaps |
| [Layout of the crate](docs/crate-layout.md) | Where things are in `src/` |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

The precise `cute::math` expansions in `src/math/` are ported from FreeBSD's
msun (Sun's fdlibm). Each of those files keeps Sun Microsystems' permissive
copyright notice, which applies to that code.

Parts of cute-rs are ported from NVIDIA CUTLASS's CuTe C++ library, the
layout code of its CuTe compiler, its `fast_math.h` and its BSD-3-Clause example programs (see
[Third-party code](#third-party-code)). NVIDIA's copyright notices and
BSD-3-Clause license, which apply to that code, are in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

cute-rs is an independent project. It is not affiliated with or endorsed by
NVIDIA.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

## Third-party code

Parts of cute-rs port algorithms and tables from NVIDIA's CuTe C++ library
(`include/cute` in CUTLASS 4.8.1), from `include/cutlass/fast_math.h`, from the layout code of its CuTe compiler
(`cutlass_compiler/cutegen`) and from its example programs: the CuTe tutorial
`examples/cute/tutorial/sgemm_2.cu` and the CuTe DSL examples in
`examples/python/CuTeDSL`, which `examples/sgemm.rs`,
`examples/elementwise_add.rs`, `examples/ampere_sgemm.rs`,
`examples/tensorop_gemm.rs` and the other examples port or follow.
All of these are under the BSD-3-Clause license.
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) has NVIDIA's copyright
notices and the license text, and lists the files and functions each port
comes from; each file with ported code also says so at its top.

NVIDIA's Python CuTe DSL package, `python/CuTeDSL` in CUTLASS, is under
NVIDIA's EULA rather than BSD-3-Clause. cute-rs took only API names and
semantics from it: no code, docstrings or message text.
