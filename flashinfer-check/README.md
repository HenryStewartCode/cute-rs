# cute-flashinfer-check

Checks the kernels of [`cute-flashinfer`](../flashinfer) against FlashInfer itself: both are
given the same inputs on the same GPU, and their outputs are compared.

`cute-flashinfer`'s own tests compare each kernel with a CPU reference written beside it. That
catches a kernel that disagrees with its author's reading of FlashInfer, but not a reading that
is itself wrong. This check has no reference of its own: the expected output is whatever
FlashInfer returns.

## How it works

1. `reference/gen.py` runs FlashInfer (the Python package) on seeded inputs and saves, for each
   case, the inputs and FlashInfer's outputs as raw files.
2. `fi-check` (this crate's program) reads each case's inputs, runs the cute-rs port of the same
   function on the GPU, and saves its outputs beside FlashInfer's.
3. `reference/compare.py` compares the two sets of outputs and prints a table.

Every output has a rule saying how close the two must be:

| Rule | Meaning | Used for |
| --- | --- | --- |
| `exact` | the same bytes | integer results (sampled tokens, indices, packed bits), copies and scatters, results documented as bit-exact |
| `ulp:1` | at most one representable value apart | a short computation rounded once (element-wise maths, a row norm, FP8 outputs) |
| `close:A:R` | `abs(out - ref) <= A + R * abs(ref)` | sums of many terms, whose order differs between the two (attention outputs, GEMMs, recurrent state) |

The comparison prints the largest difference of every output whether or not it passes, so a
loose rule cannot hide a large error.

## Running it

It needs an NVIDIA GPU, FlashInfer and PyTorch (for the reference), and `mlir-opt` (for cute-rs).

```bash
python flashinfer-check/reference/gen.py cases            # FlashInfer's outputs
cargo run --release -p cute-flashinfer-check --features cuda -- cases   # the ports' outputs
python flashinfer-check/reference/compare.py cases report.md
```

`gen.py cases norm rope` writes only those modules' cases; `fi-check cases norm_` runs only the
cases whose name contains `norm_`; `compare.py cases --only norm_` compares only those.

## Results

FlashInfer 0.7.5.post1 (PyTorch 2.8.0) against `cute-flashinfer`, on an RTX 2000 Ada (SM89),
26 November 2026: 631 cases, 1,092 outputs, all within their rule.

| Module | Cases | Outputs | Within the rule | Bit-identical | Differ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `activation` | 30 | 30 | 30 | 20 | 0 |
| `attention` | 10 | 20 | 20 | 1 | 0 |
| `cascade` | 36 | 72 | 72 | 72 | 0 |
| `decode` | 36 | 72 | 72 | 9 | 0 |
| `gdn` | 51 | 109 | 109 | 13 | 0 |
| `gemm` | 24 | 24 | 24 | 24 | 0 |
| `kda` | 28 | 56 | 56 | 11 | 0 |
| `mamba` | 47 | 109 | 109 | 26 | 0 |
| `mla` | 13 | 26 | 26 | 1 | 0 |
| `moe` | 20 | 28 | 28 | 23 | 0 |
| `norm` | 58 | 82 | 82 | 64 | 0 |
| `pod` | 7 | 14 | 14 | 2 | 0 |
| `prefill` | 67 | 134 | 134 | 46 | 0 |
| `quantization` | 62 | 108 | 108 | 108 | 0 |
| `rope` | 46 | 86 | 86 | 84 | 0 |
| `sampling` | 65 | 74 | 74 | 65 | 0 |
| `sparse` | 17 | 34 | 34 | 15 | 0 |
| `xqa` | 14 | 14 | 14 | 0 | 0 |
| Total | 631 | 1,092 | 1,092 | 584 | 0 |

The attention families are rarely bit-identical because the two libraries add the same terms
in a different order. Their largest differences are a few steps of the output type: 0.004 in
a Float16 output and 0.016 in a BFloat16 one, on values above 1.

### Faults it found

The first run had five outputs outside their rule, from two faults in the ports; a third
turned up while those were being fixed. All three are fixed, and their cases now pass.

* **Single decode with a sliding window gave NaN** when the cache was longer than the window
  (`decode_single_f16_hd128_window_n4000`). The plan split the whole cache into chunks, but
  the kernel reads only the window, so the chunks outside it were empty and divided zero by
  zero. The plan now covers the window only, as the batch plan did.
* **A causal prefill split into chunks gave NaN** for the queries that see nothing of one of
  their chunks: after a cached prefix, the early queries cannot see the last chunk (the six
  `prefill_*_c_prefix*` cases, added for it). An unseen chunk's share was zero divided by
  zero, and the merge kept the NaN; it is now zero. The port's own tests ran such plans, but
  their error measure dropped NaN, so they passed.
* **Top-k then top-p sampling on a large vocabulary drew another token** (the four
  `sampling_kp_*_top_k_first_*x128256_*` cases; vocabulary of 65,536 or more and k of 256 or
  less). The token was always one of the top k and the distribution was right, but it was
  not the token FlashInfer draws from the same seed: FlashInfer sorts the k kept values
  before it draws, and the port kept them in the order its selection left them. It now sorts
  them too.

### Where the rule is not the obvious one

* **`gelu`** is held to one step of the output type or 4e-6, whichever is larger. The two
  libraries approximate `erf` differently; where the input is below about -3.7 the result is
  under 0.002 and neither is accurate to its last digits (each is up to 1.3e-6 from a Float64
  calculation, and they are up to 1.9e-6 apart). `silu` and `gelu_tanh` are bit-identical.
* **GDN decode with Float16 inputs** uses the BFloat16 tolerance: FlashInfer converts q, k
  and v to BFloat16 before it computes, and the port does not.
* **GDN chunked prefill's final state** uses the looser of the input's and the state's
  tolerance: both libraries multiply 16-bit operands, so a Float32 state is no more accurate
  than they are.
* **Padded state slots (index -1).** FlashInfer documents a padded row's output as zero on
  some paths and undefined on others, and in practice returns values computed from another
  slot. The cases put a pad only where FlashInfer promises a result; in the KDA pool cases
  the reference's padded rows are set to zero, which is what its documentation says and what
  the port returns. The state pool is always compared whole.
* **A draw that lands on a boundary.** Sampled tokens are compared exactly, and match. Two
  libraries that add the probabilities in a different order can still pick neighbouring
  tokens when a random number falls within Float32 rounding of the boundary between them; one
  such row turned up while the cases were being written (about 7e-9 from the boundary).

### What the reference is, where it is not the public function

* **`xqa`**: `flashinfer.xqa` refuses GPUs below SM90, so the reference is FlashInfer's own
  XQA kernel built for SM89 through its JIT and called directly.
* **`attention`**: FlashInfer's `BatchAttention` cannot launch on this GPU (too many blocks
  for a cooperative launch), so the reference is its `BatchPrefillWithPagedKVCacheWrapper` on
  the same requests, which computes the same attention.
* **`top_k`** and the top-k transforms return unordered sets, so both sides are sorted before
  the comparison, and the inputs have no ties.
* **MoE routing** rows are sorted by expert index on both sides.

### Not compared

* `prefill_sm100`: it needs a Blackwell GPU.
* Calls FlashInfer refuses, though the port has them: Float16 inputs or state in `recurrent_kda`,
  a Float16 state in `gated_delta_rule_decode_pretranspose`, and the GDN multi-token cache with
  exactly T steps (9 cases; `compare.py` lists them).
* Paths FlashInfer cannot run on SM89: the MoE routing methods other than DeepSeek-V3's
  (`trtllm_gen_routing` refuses this GPU), `per_token_group_quant_8bit` with groups of 64 or
  256 (its backend for them is not installed here), and `bmm_fp8` with E5M2 operands.
* A pad inside KDA's per-token checkpoint columns other than the last: the two disagree, and
  FlashInfer's documentation does not say what should happen.
* Ties: the top-k and routing inputs are made without them.

## Adding cases

A module has two files: `reference/cases_<module>.py`, whose `cases()` yields `common.Case`s
(parameters, inputs, and FlashInfer's outputs with their rules), and `src/<module>.rs`, whose
`run` reads a case and writes the port's outputs. `cases_norm.py` and `src/norm.rs` are the
shortest pair to copy from.
