# cutedsl-check

Asks NVIDIA's CuTe DSL and cute-rs the same questions and compares the answers. So far it
covers the layout algebra.

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

## What it does not cover

The generator draws static layouts with non-negative integer strides. Runtime extents,
negative strides and scaled-basis strides are not compared, and nor is anything beyond the
layout algebra.
