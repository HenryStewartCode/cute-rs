# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.fused_moe`: `cutlass_fused_moe` (SM80/SM89 path) and `fused_topk_deepseek` routing.

`cutlass_fused_moe` runs SwiGLU experts in Float16 and BFloat16 on precomputed routing (softmax,
top-k, renormalised): one token, a few, a few hundred, and a layer where one expert gets no
token. Expert weights are scaled by 1/sqrt(fan-in), so outputs are of order 1; sums of many
terms use `close`. Experts are kept small so a case's files stay under 50 MB.

Of the routing methods the port lists, only DeepSeek-V3's `fused_topk_deepseek` exists in
FlashInfer for SM89: the others (`trtllm_gen_routing`) refuse compute capability 8.9. Selected
experts are `exact`; FlashInfer may order a row's experts differently from the port, so both
sides sort each row by expert index together with its weights.
"""
import torch
import torch.nn.functional as F

import flashinfer.fused_moe as fused_moe
from common import Case

DEV = "cuda"
DTYPES = ((torch.float16, "f16", "close:1e-3:2e-3"), (torch.bfloat16, "bf16", "close:8e-3:1.6e-2"))
# (name, hidden, inter, experts, top_k, tokens, experts that get no token)
MOE = (
    ("small1", 256, 384, 4, 2, 1, 0),
    ("small37", 256, 384, 4, 2, 37, 0),
    ("mid3", 1024, 512, 8, 2, 3, 0),
    ("mid300", 1024, 512, 8, 2, 300, 0),
    ("empty_expert", 1024, 512, 8, 2, 20, 2),
    ("e32k4", 512, 256, 32, 4, 9, 0),
)
# (name, experts, n_group, topk_group, top_k, tokens, logits type)
ROUTES = (
    ("dsv3_256_t1", 256, 8, 4, 8, 1),
    ("dsv3_256_t7", 256, 8, 4, 8, 7),
    ("dsv3_256_t300", 256, 8, 4, 8, 300),
    ("g4_64_t33", 64, 4, 2, 4, 33),
)


def routing(tokens, experts, k, dead):
    logits = torch.randn(tokens, experts, device=DEV)
    if dead:  # the last `dead` experts are never chosen
        logits[:, experts - dead :] = -1e9
    w, sel = torch.topk(F.softmax(logits, dim=-1), k, dim=-1)
    return sel.to(torch.int32), (w / w.sum(-1, keepdim=True)).float()


def moe_cases():
    for dt, tag, rule in DTYPES:
        for name, h, inter, e, k, tokens, dead in MOE:
            x = torch.randn(tokens, h, device=DEV).to(dt)
            w1 = (torch.randn(e, 2 * inter, h, device=DEV) / h**0.5).to(dt)
            w2 = (torch.randn(e, h, inter, device=DEV) / inter**0.5).to(dt)
            sel, scales = routing(tokens, e, k, dead)
            c = Case(
                f"moe_fused_{name}_{tag}", "moe", op="fused", dtype=tag, hidden=h, inter=inter, experts=e, top_k=k,
                tokens=tokens,
            )
            c.input("x", x)
            c.input("selected", sel)
            c.input("scales", scales)
            c.input("w1", w1)
            c.input("w2", w2)
            out = torch.empty(tokens, h, dtype=dt, device=DEV)
            fused_moe.cutlass_fused_moe(x, sel, scales, w1, w2, dt, quant_scales=None, output=out)
            c.output("out", out, rule)
            yield c


def distinct_rows(tokens, experts, n_group, topk_group, dt, bias):
    """Logits whose selection keys are all different within a row (and whose group scores are)."""
    logits = torch.empty(tokens, experts, device=DEV)
    todo = torch.arange(tokens, device=DEV)
    while todo.numel():
        trial = (torch.randn(todo.numel(), experts, device=DEV) * 1.2).to(dt)
        b = trial.float().sigmoid() + bias.float()
        top2 = b.view(-1, n_group, experts // n_group).topk(2, dim=-1).values.sum(-1)
        ok = torch.ones(todo.numel(), dtype=torch.bool, device=DEV)
        for key in (b, top2):
            s = key.sort(-1).values
            ok &= (s[:, 1:] != s[:, :-1]).all(-1)
        logits[todo[ok]] = trial[ok].float()
        todo = todo[~ok]
    return logits.to(dt)


def route_cases():
    for dt, tag in ((torch.float32, "f32"), (torch.bfloat16, "bf16")):
        for name, e, n_group, topk_group, k, tokens in ROUTES:
            bias = (torch.randn(e, device=DEV) * 0.05).to(dt)
            logits = distinct_rows(tokens, e, n_group, topk_group, dt, bias)
            scaling = 2.5
            c = Case(
                f"moe_route_{name}_{tag}", "moe", op="route_deepseek", dtype=tag, experts=e, n_group=n_group,
                topk_group=topk_group, top_k=k, tokens=tokens, scaling=scaling,
            )
            c.input("logits", logits)
            c.input("bias", bias)
            # The weights come back in the logits' type.
            vals = torch.empty(tokens, k, dtype=dt, device=DEV)
            idx = torch.empty(tokens, k, dtype=torch.int32, device=DEV)
            fused_moe.fused_topk_deepseek(logits, bias, n_group, topk_group, k, scaling, vals, idx)
            # Each row in expert order, so the comparison does not depend on how a library orders it.
            idx, order = idx.sort(-1)
            vals = vals.gather(1, order)
            c.output("indices", idx, "exact")
            # Float32 weights are a short computation (1e-6 close); BFloat16 ones are that rounded once.
            c.output("weights", vals, "close:1e-6:1e-5" if dt == torch.float32 else "ulp:1")
            yield c


def cases():
    yield from moe_cases()
    yield from route_cases()
