# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.activation`: `silu_and_mul`, `gelu_and_mul` and `gelu_tanh_and_mul`.

Element-wise maths rounded once to the output type, so FlashInfer and the port may land on
neighbouring values: `ulp:1`.

`gelu` is `x * 0.5 * (1 + erf(x / sqrt(2)))`, and the two libraries approximate `erf`
differently. Where `x` is below about -3.7 the sum `1 + erf` nearly cancels, the result is
below 0.002, and neither is accurate to its last digits: against a Float64 calculation each is
up to about 1.3e-6 out, and they are up to 1.9e-6 from each other. So `gelu` is held to one
step of the output type or 4e-6, whichever is larger.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
SHAPES = ((1, 4096), (3, 1000), (64, 4096), (257, 1536), (1024, 14336))
DTYPES = ((torch.float16, "f16"), (torch.bfloat16, "bf16"))
# One step of the type as a relative tolerance (2^-10 and 2^-7), and the erf difference.
GELU_RULE = {"f16": "close:4e-6:1e-3", "bf16": "close:4e-6:8e-3"}
FUNCS = (
    ("silu", flashinfer.silu_and_mul),
    ("gelu", flashinfer.gelu_and_mul),
    ("gelu_tanh", flashinfer.gelu_tanh_and_mul),
)


def cases():
    for dt, tag in DTYPES:
        for tokens, d in SHAPES:
            x = torch.randn(tokens, 2 * d, dtype=dt, device=DEV) * 2
            for act, fn in FUNCS:
                c = Case(f"activation_{act}_{tag}_{tokens}x{d}", "activation", act=act, dtype=tag, tokens=tokens, d=d)
                c.input("input", x)
                c.output("out", fn(x), GELU_RULE[tag] if act == "gelu" else "ulp:1")
                yield c
