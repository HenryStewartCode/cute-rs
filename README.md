# cute-rs

[![CI](https://github.com/HenryStewartCode/cute-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/HenryStewartCode/cute-rs/actions/workflows/ci.yml)

A CuTe-style GPU kernel DSL for Rust.

> **Experimental, and mostly machine-written.** Nearly all of cute-rs was written by an AI
> model (Anthropic's Claude, through Claude Code), directed by one person. At about 285,000
> lines it is more than anyone has read line by line, so judge it by what it has been checked
> against, not by its size. [What has been checked](#what-has-been-checked) lists the
> comparisons with NVIDIA's CuTe DSL, with FlashInfer and with real GPUs, and what has not
> been checked. Reviews and bug reports are welcome.

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

## What has been checked

cute-rs has a counterpart for each feature of CuTe DSL 4.8.1 except three left out by design:
the `frontend-next` tracing frontend, and `cute-ext`'s compiler pipeline and entry points
(cute-rs builds kernels from Rust closures instead).
[docs/cutedsl-mapping.md](docs/cutedsl-mapping.md) lists them one by one. Having a counterpart
is not the same as being right, and how well each part has been checked varies. This section
says how.

### Against things written independently of cute-rs

| What | Compared with | Result |
| --- | --- | --- |
| The layout algebra | NVIDIA's CuTe DSL 4.8.1, given the same 20,500 seeded random calls ([`cutedsl-check/`](cutedsl-check)) | Two seeds, 41,000 calls. cute-rs gives CuTe DSL's answer character for character on 36,945 and both refuse 3,121. cute-rs answers 88 that CuTe DSL refuses, and refuses 846 that are ill-posed, where CuTe DSL returns a layout that is not what the operation defines (the complement of an overlapping layout, say). No call gets a different answer. The first run found 771 calls that cute-rs wrongly refused and 143 it answered differently; those are fixed, and the second seed was recorded after the fixes ([docs/testing.md](docs/testing.md#checking-the-layout-algebra-against-cute-dsl)). |
| Atoms and partitioning: which thread holds which element | CuTe DSL 4.8.1, on 9,604 recorded cases over the MMA and copy atoms of five architectures, tiled MMAs and copies, and tensor partitioning | 7,875 identical and 549 refused by both. 762 are the same assignment written differently, and 180 differ in ways that keep every element on the same thread (register order within a thread, shape nesting, the sparse operand's form). cute-rs refuses 5 that CuTe DSL accepts. Two faults of form were found and fixed ([docs/testing.md](docs/testing.md#checking-atoms-and-partitioning-against-cute-dsl)). |
| The FlashInfer ports ([`flashinfer/`](flashinfer)) | FlashInfer itself, on the same inputs, on an SM89 GPU ([`flashinfer-check/`](flashinfer-check)) | All 1,092 outputs agree within the rule stated for each, 584 of them bit for bit. The first run found three faults in the ports, which are fixed. |
| The simulator | Real GPUs: each kernel the simulator's tests run is also run on the device, and the buffers compared ([docs/testing.md](docs/testing.md#checking-the-simulator-against-a-gpu)) | On a B200, 15,140 of 15,522 kernel runs give the simulator's bytes exactly; on an H100, 18,998 of 20,237. The rest differ in approximate instructions, last-digit rounding in the tensor cores, addresses, thread order and ties. The B200 run also found places where the simulator was wrong; they are corrected, except one that is refused instead ([docs/limitations.md](docs/limitations.md)). |
| The generated PTX | NVIDIA's assembler, `ptxas` | It assembles for every target architecture except SM107, which no released CUDA toolkit assembles yet. |

### On real GPUs

| Architecture | Hardware tested | Status |
| --- | --- | --- |
| SM80 / SM89 (Ampere, Ada) | an SM89 GPU | The full GPU suite passes |
| SM120 (consumer Blackwell) | RTX 5090 | The full GPU suite passes, including TMA, the SM120 MMAs and the block-scaled warp MMAs |
| SM90 (Hopper) | H100 | The Hopper device tests pass (the WGMMA GEMMs with clusters, every major mode, Float16, BFloat16, FP8 and 8-bit integers, the persistent GEMM and the FMHA forward) ([docs/testing.md](docs/testing.md#results-on-an-h100)) |
| SM100 (datacenter Blackwell) | B200 | The Blackwell device tests pass (tcgen05 GEMMs with and without CTA pairs, mixed-input, NVFP4, custom epilogues, task scheduling), as does a 200-launch stress test of each ([docs/testing.md](docs/testing.md#results-on-a-b200)) |
| SM103 / SM107 (Blackwell Ultra, Rubin) | none | Never run. Checked by the simulator and `ptxas` only (SM107 by the simulator only) |

The device tests compare each kernel's output with a CPU reference. `scripts/gpu-validate.sh`
does a whole validation pass on a GPU machine in one go (it installs Rust and `mlir-opt`, runs
the architecture's suite, then the full suite, then the timings, and packs the results), and
`scripts/pod-run.sh` drives it on a rented machine over SSH.

### By its own tests only

The other tests, about 1,850 of them, were written together with the code they test. They
compare kernels with CPU references and `cute::math` with a double-double reference, and they
catch regressions, but they are weaker evidence than the comparisons above: the three faults
in the FlashInfer ports had passed them. In particular:

* Apart from the layout algebra and the atoms' layouts, whether a feature behaves as CuTe DSL's does has been judged
  from CuTe DSL's documentation and examples, not by running the two side by side. The
  layout comparison itself draws static layouts with non-negative strides only.
* The JAX (`xla_ffi`) and TVM-FFI bindings are tested against hand-built call frames, not
  under JAX or `libtvm_ffi` themselves.
* Multicast memory (`MulticastGroup`) has never run: it needs two or more NVLink-connected
  GPUs, and the machines used so far had one.

### Known gaps

* Speed: on a B200 at 8192 x 8192 x 8192 in Float16, the tutorial tcgen05 GEMMs run at about
  1,600 TFLOP/s, level with cuBLAS, but CuTe DSL's own builds of the same kernels are 15 to 30%
  faster, and 40 to 80% faster for the dense GEMM example
  ([docs/testing.md](docs/testing.md#results-on-a-b200)). On an H100 the Hopper GEMMs reach about
  500 TFLOP/s where cuBLAS reaches 750, and the FMHA 199 TFLOP/s where PyTorch's attention
  reaches 262 ([docs/testing.md](docs/testing.md#results-on-an-h100)).
* A CTA pair's block-scaled MMA with M = 128: where it wants B's scale factors is not known,
  so the simulator refuses it, and the 2x2 data path of `make_tmem_layout_sfb` is not laid
  out yet.
* The JIT disk cache refuses a directory other users can write to, which includes network
  volumes that report mode 777; point `CUTE_RS_CACHE_DIR` at local disk there.

[docs/limitations.md](docs/limitations.md) lists each feature's smaller gaps.

## Size, and where to start reading

| Part | Lines |
| --- | --- |
| The library, `src/` | 143,000 |
| Its tests, `tests/` | 88,000 |
| The examples, `examples/` | 22,000 |
| The FlashInfer ports and their check, `flashinfer/` and `flashinfer-check/` | 38,000 |

The public history starts with one large commit because the project was developed in a
private repository and first published, as a snapshot, on 24 November 2026.

Good places to start reading are [`src/layout/layout.rs`](src/layout/layout.rs) (the layout
algebra, the part most directly comparable with CuTe), [`examples/tensorop_gemm.rs`](examples/tensorop_gemm.rs)
(a complete tensor-core kernel), [`src/sim/`](src/sim) (the simulator) and
[`src/backend/mlir.rs`](src/backend/mlir.rs) (code generation).
[docs/crate-layout.md](docs/crate-layout.md) maps the rest.

If you know CuTe, CUTLASS or CUDA and something here looks wrong, please open an issue. A
kernel that the simulator accepts and a GPU does not, or a call that CuTe DSL answers
differently, is exactly the kind of report that is wanted.

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
[`flashinfer-check/`](flashinfer-check) runs them and FlashInfer itself on the same inputs and
compares the outputs ([results](flashinfer-check/README.md#results)).

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
| [Testing](docs/testing.md) | What the test suites cover, the comparisons with CuTe DSL, FlashInfer and real GPUs, and how to run them |
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
