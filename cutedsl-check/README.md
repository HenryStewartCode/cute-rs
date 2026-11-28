# cutedsl-check

Asks NVIDIA's CuTe DSL and cute-rs the same questions and compares the answers. So far it
covers the layout algebra, and atoms and partitioning (which thread holds which element).

## How it works

1. `reference/gen_layouts.py` draws seeded random calls of 41 layout operations (500 of each
   by default) and records what CuTe DSL prints for each: a layout, a number, a tuple, `ERR`
   where it raises an error, or `CRASH` where the call ends its Python process. The cases are
   drawn before CuTe DSL is asked anything, so a seed always gives the same calls.
2. `cutedsl-check layouts FILE` makes the same calls in cute-rs, prints each answer the way
   CuTe DSL does, and compares the two texts.

```bash
pip install nvidia-cutlass-dsl==4.8.1      # Linux; no GPU needed
python cutedsl-check/reference/gen_layouts.py --seed 20261126 layouts.tsv
cargo run --release -p cute-rs-cutedsl-check -- layouts layouts.tsv --show 3
```

The recorded answers are the output of NVIDIA's package and are not kept in this repository.

## Reading the table

Each call lands in one column:

| Column | Meaning |
| --- | --- |
| same | Both answer, with the same text |
| both refuse | Both refuse the call (CuTe DSL with an error or a crash) |
| same map | Both answer with layouts that print differently but send every index to the same offset |
| DSL refuses | CuTe DSL raises an error; cute-rs answers |
| DSL crashes | The call ends CuTe DSL's process; cute-rs answers |
| ill-posed | CuTe DSL answers and cute-rs refuses, and a rule in `src/ill_posed.rs` that looks only at the call's inputs says the call is ill-posed |
| we refuse | CuTe DSL answers and cute-rs refuses, with no such rule |
| differ | Both answer, differently |

The exit status is 1 if any call is in "same map", "we refuse" or "differ". `--show N` prints
up to N examples of each column but the first two.

The ill-posed rules are: the complement of a layout that is not injective; a composition for
which CuTe builds a layout that is not the composition (some index gets an offset other than
`lhs(rhs(i))`); an upcast by a ratio that a stride neither divides nor is divided by; and a
downcast of a layout with no stride-1 mode. `src/ill_posed.rs` says for each which assertion
of C++ CuTe refuses it, or that none does.

## Results

With CuTe DSL 4.8.1, 500 calls of each operation:

| | Seed 20261126 | Seed 20261127 |
| --- | ---: | ---: |
| same | 18,475 | 18,470 |
| both refuse | 1,552 | 1,569 |
| same map | 0 | 0 |
| DSL refuses | 53 | 35 |
| DSL crashes | 0 | 0 |
| ill-posed | 420 | 426 |
| we refuse | 0 | 0 |
| differ | 0 | 0 |

The first run, on the first seed, had 771 calls in "we refuse", 143 in "differ" and 2,205 in
"same map". Those were faults in cute-rs and are fixed; the second seed was recorded after the
fixes. [docs/testing.md](../docs/testing.md#checking-the-layout-algebra-against-cute-dsl) says
what they were.

## Atoms and partitioning

`reference/gen_atoms.py` records, for the MMA atoms of sm_80, sm_89, sm_90a, sm_100a and
sm_120a, the copy atoms, tiled MMAs and tiled copies, and tensor partitioning: the
thread-value layouts, tilers, fragment layouts and what `partition_*` gives a thread. It
traces each architecture with `CUTE_DSL_ARCH`, so it needs no GPU. `cutedsl-check atoms
FILE` replays the cases in cute-rs.

```bash
python cutedsl-check/reference/gen_atoms.py atoms.tsv
cargo run --release -p cute-rs-cutedsl-check -- atoms atoms.tsv --show 3
```

With CuTe DSL 4.8.1 (seed 20261127, 9,604 cases). This check does not pass clean:

| | Cases |
| --- | ---: |
| same | 7,875 |
| both refuse | 549 |
| same map | 762 |
| DSL refuses | 63 |
| ill-posed | 170 |
| we refuse | 5 |
| differ | 180 |

No difference found moves an element to another thread. What remains:

* **same map (762):** the same assignment written differently. An extent-1 mode has stride 0
  in one and a product in the other; for the single-thread mode of the tcgen05 atoms and for
  WGMMA's C with N = 8, cute-rs follows C++ CuTe, which CuTe DSL differs from; the value mode
  of the transposed `ldmatrix` and `stmatrix` atoms is nested differently.
* **differ (180):** the order of a fragment's rest modes (73: cute-rs lays a fragment out
  column-major, CuTe DSL by the partition's strides, so a thread holds the same elements in
  another register order); `partition_shape_*` under a nested permutation (64: cute-rs follows
  C++ CuTe); the sparse A operand of tcgen05, which cute-rs partitions in its compressed
  `(M, K/2)` form (32); and the tiler of a tiled copy with an extent-1 mode (11).
* **we refuse (5):** `tcgen05.MmaMXF8F6F4Op` with an 8-bit MN-major B and N = 216. cute-rs
  wants N to be a multiple of 16; CuTe DSL and C++ CuTe take a multiple of 8.
* **ill-posed (170):** six rules in `cutedsl-check/src/ill_posed.rs`, most of them a tiled copy
  that holds too few values for its atom, which C++ CuTe refuses with an assertion.

Two faults of form were found in cute-rs and fixed: a tiled MMA's thread mode was nested
(`((4,8,4),2)` for `(4,8,4,2)`), and `partition_fragment_a` of a tcgen05 MMA with A in tensor
memory gave a register fragment instead of the tensor-memory one.

Not recorded, because CuTe DSL needs real tensors or descriptors to build them: descriptor
fragments in shared memory, the TMA atoms and `make_tmem_copy`.

## What it does not cover

The generator draws static layouts with non-negative integer strides. Runtime extents,
negative strides and scaled-basis strides are not compared. Beyond the layout algebra and
the atoms, nothing is compared: not what kernels compute, and not the code generated.
