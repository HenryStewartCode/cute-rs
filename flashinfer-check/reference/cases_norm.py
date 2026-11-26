# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.norm`: the RMS norms, their Gemma, quantising and 3-D forms, and layer norm.

Each sums a row's squares in Float32 in its own order and rounds the result to the output
type, so FlashInfer and the port may land on neighbouring values: `ulp:1`.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
EPS = 1e-6
SHAPES = ((1, 1024), (19, 3072), (99, 4096), (7, 8192))
DTYPES = ((torch.float16, "f16"), (torch.bfloat16, "bf16"))


def cases():
    for dt, tag in DTYPES:
        for batch, d in SHAPES:
            x = torch.randn(batch, d, dtype=dt, device=DEV) * 2
            r = torch.randn(batch, d, dtype=dt, device=DEV) * 2
            w = torch.randn(d, dtype=dt, device=DEV) * 0.5 + 1
            shape = dict(dtype=tag, batch=batch, d=d, eps=EPS)

            def case(variant, **extra):
                c = Case(f"norm_{variant}_{tag}_{batch}x{d}", "norm", variant=variant, **shape, **extra)
                c.input("weight", w)
                return c

            for variant, fn in (("rms", flashinfer.rmsnorm), ("gemma_rms", flashinfer.gemma_rmsnorm)):
                c = case(variant)
                c.input("input", x)
                c.output("out", fn(x, w, EPS), "ulp:1")
                yield c

            for variant, fn in (
                ("fused_add", flashinfer.fused_add_rmsnorm),
                ("gemma_fused_add", flashinfer.gemma_fused_add_rmsnorm),
            ):
                c = case(variant)
                xs, rs = c.input("input", x).clone(), c.input("residual", r).clone()
                fn(xs, rs, w, EPS)
                c.output("input", xs, "ulp:1")
                # The residual is one Float16 or BFloat16 sum per element.
                c.output("residual", rs, "exact")
                yield c

            if dt == torch.bfloat16:  # FlashInfer's layernorm takes BFloat16 only
                gamma = torch.randn(d, dtype=torch.float32, device=DEV) * 0.5 + 1
                beta = torch.randn(d, dtype=torch.float32, device=DEV) * 0.5
                c = Case(f"norm_layernorm_{tag}_{batch}x{d}", "norm", variant="layernorm", **shape)
                c.input("input", x)
                c.input("gamma", gamma)
                c.input("beta", beta)
                c.output("out", flashinfer.layernorm(x, gamma, beta, EPS), "ulp:1")
                yield c

            scale = 0.02
            st = torch.tensor([scale], dtype=torch.float32, device=DEV)
            c = case("rms_quant", scale=scale)
            c.input("input", x)
            out = torch.empty(batch, d, dtype=torch.float8_e4m3fn, device=DEV)
            flashinfer.norm.rmsnorm_quant(out, x, w, st, EPS)
            c.output("out", out, "ulp:1")
            yield c

            c = case("fused_add_quant", scale=scale)
            xs, rs = c.input("input", x).clone(), c.input("residual", r).clone()
            out = torch.empty(batch, d, dtype=torch.float8_e4m3fn, device=DEV)
            flashinfer.norm.fused_add_rmsnorm_quant(out, xs, rs, w, st, EPS)
            c.output("out", out, "ulp:1")
            c.output("residual", rs, "exact")
            yield c

        # rmsnorm of a 3-D input: the Q or K heads of a packed [batch, heads + 2, d] tensor.
        for batch, heads, d in ((5, 32, 128), (19, 8, 128), (3, 7, 520)):
            qkv = torch.randn(batch, heads + 2, d, dtype=dt, device=DEV) * 2
            w = torch.randn(d, dtype=dt, device=DEV) * 0.5 + 1
            c = Case(f"norm_qk_rms_{tag}_{batch}x{heads}x{d}", "norm", variant="qk_rms", dtype=tag, batch=batch, heads=heads, d=d, eps=EPS)
            c.input("input", qkv)
            c.input("weight", w)
            out = torch.empty(batch, heads, d, dtype=dt, device=DEV)
            flashinfer.rmsnorm(qkv[:, :heads, :], w, EPS, out=out)
            c.output("out", out, "ulp:1")
            yield c
