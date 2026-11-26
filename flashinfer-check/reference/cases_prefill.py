# SPDX-License-Identifier: Apache-2.0
"""`flashinfer` batch prefill (paged and ragged KV) and `single_prefill_with_kv_cache`, `fa2` backend.

Float16 and BFloat16, head dim 64/128/256, grouped queries, causal or not, ragged lengths (one-token
requests, lengths off the tile and page), long requests that make the planner split the KV and merge
(also causal ones after a cached prefix, where a query cannot see every chunk),
soft-cap, sliding window, custom mask, ALiBi and an FP8 KV cache. Attention outputs add up many terms:
`close` rules, on the output and on the base-2 log-sum-exp.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16}
FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
O_RULE = {"f16": "close:1e-3:2e-3", "bf16": "close:8e-3:1.6e-2"}
LSE_RULE = "close:1e-4:1e-4"
K_SCALE, V_SCALE = 0.7, 1.3
HQ, HKV = 32, 8
# Queries scaled so that attention is peaked and the outputs are of order 0.1 to 1.
QS = 2.0


def indptr(lens):
    return torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=DEV)


def rnd(*shape, dt, scale=1.0):
    return (torch.randn(*shape, device=DEV) * scale).to(dt)


def mask_bits(qo_lens, kv_lens):
    """One random mask per request, flattened; every query keeps its diagonal token."""
    parts = []
    for qo, kv in zip(qo_lens, kv_lens):
        m = torch.rand(qo, kv, device=DEV) > 0.5
        diag = (torch.arange(qo, device=DEV) + kv - qo).clamp(max=kv - 1)
        m[torch.arange(qo, device=DEV), diag] = True
        parts.append(m.flatten())
    return torch.cat(parts)


def variant_kwargs(c, variant, qo_lens, kv_lens, opts):
    """Plan arguments of `variant`, with the case's parameters and inputs recorded."""
    kw = {}
    if variant == "softcap":
        kw["logits_soft_cap"] = opts["cap"]
    elif variant == "window":
        kw["window_left"] = opts["window"]
    elif variant == "alibi":
        kw["pos_encoding_mode"] = "ALIBI"
    elif variant == "mask":
        bits = c.input("mask", mask_bits(qo_lens, kv_lens))
        kw["custom_mask"] = bits
    return kw


def new_case(kind, tag, hd, qo_lens, kv_lens, causal, variant, kv_tag, hq, hkv, suffix, **extra):
    name = f"prefill_{kind}_{tag}_{kv_tag}_d{hd}_{variant}_{'c' if causal else 'n'}_{suffix}"
    return Case(name, "prefill", kind=kind, dtype=tag, kv=kv_tag, head_dim=hd, hq=hq, hkv=hkv, causal=causal,
                variant=variant, qo_lens=qo_lens, kv_lens=kv_lens, k_scale=K_SCALE, v_scale=V_SCALE, **extra)


def kv_tensor(shape, dt, kv_tag):
    """K or V in the cache's type; FP8 stores x / scale, FlashInfer multiplies the scale back."""
    x = torch.randn(*shape, device=DEV)
    return x.to(dt) if kv_tag == "f16" else x.to(FP8[kv_tag])


def run_kwargs(kv_tag):
    return {} if kv_tag == "f16" else dict(k_scale=K_SCALE, v_scale=V_SCALE)


def finish(c, o, lse, tag):
    c.output("o", o, O_RULE[tag])
    c.output("lse", lse, LSE_RULE)
    return c


def paged(tag, hd, qo_lens, kv_lens, causal, suffix, ps=16, hq=HQ, hkv=HKV, variant="none", kv_tag="f16", **opts):
    dt = DTYPES[tag]
    c = new_case("paged", tag, hd, qo_lens, kv_lens, causal, variant, kv_tag, hq, hkv, suffix, page_size=ps, **opts)
    pages = [-(-n // ps) for n in kv_lens]
    num_pages = sum(pages) + 3
    ids = torch.randperm(num_pages, dtype=torch.int32, device=DEV)
    kv_indptr = indptr(pages)
    kv_indices = ids[: sum(pages)].contiguous()
    last = torch.tensor([(n - 1) % ps + 1 for n in kv_lens], dtype=torch.int32, device=DEV)
    qo_indptr = indptr(qo_lens)
    qscale = 3.0 if variant == "softcap" else QS
    q = c.input("q", rnd(sum(qo_lens), hq, hd, dt=dt, scale=qscale))
    k = c.input("k", kv_tensor((num_pages, ps, hkv, hd), dt, kv_tag))
    v = c.input("v", kv_tensor((num_pages, ps, hkv, hd), dt, kv_tag))
    for n, t in (("qo_indptr", qo_indptr), ("kv_indptr", kv_indptr), ("kv_indices", kv_indices), ("last", last)):
        c.input(n, t)
    kw = variant_kwargs(c, variant, qo_lens, kv_lens, opts)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    w.plan(qo_indptr, kv_indptr, kv_indices, last, hq, hkv, hd, ps, causal=causal, q_data_type=dt,
           kv_data_type=dt if kv_tag == "f16" else FP8[kv_tag], **kw)
    o, lse = w.run(q, (k, v), return_lse=True, **run_kwargs(kv_tag))
    return finish(c, o, lse, tag)


def ragged(tag, hd, qo_lens, kv_lens, causal, suffix, hq=HQ, hkv=HKV, variant="none", kv_tag="f16", **opts):
    dt = DTYPES[tag]
    c = new_case("ragged", tag, hd, qo_lens, kv_lens, causal, variant, kv_tag, hq, hkv, suffix, **opts)
    qscale = 3.0 if variant == "softcap" else QS
    q = c.input("q", rnd(sum(qo_lens), hq, hd, dt=dt, scale=qscale))
    k = c.input("k", kv_tensor((sum(kv_lens), hkv, hd), dt, kv_tag))
    v = c.input("v", kv_tensor((sum(kv_lens), hkv, hd), dt, kv_tag))
    qo_indptr, kv_indptr = indptr(qo_lens), indptr(kv_lens)
    c.input("qo_indptr", qo_indptr)
    c.input("kv_indptr", kv_indptr)
    kw = variant_kwargs(c, variant, qo_lens, kv_lens, opts)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    w.plan(qo_indptr, kv_indptr, hq, hkv, hd, causal=causal, q_data_type=dt,
           kv_data_type=dt if kv_tag == "f16" else FP8[kv_tag], **kw)
    o, lse = w.run(q, k, v, return_lse=True, **run_kwargs(kv_tag))
    return finish(c, o, lse, tag)


def single(tag, hd, qo, kv, causal, suffix, hq=HQ, hkv=HKV, kv_tag="f16"):
    dt = DTYPES[tag]
    c = new_case("single", tag, hd, [qo], [kv], causal, "none", kv_tag, hq, hkv, suffix)
    q = c.input("q", rnd(qo, hq, hd, dt=dt, scale=QS))
    k = c.input("k", kv_tensor((kv, hkv, hd), dt, kv_tag))
    v = c.input("v", kv_tensor((kv, hkv, hd), dt, kv_tag))
    o, lse = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal, backend="fa2", return_lse=True,
                                                     **run_kwargs(kv_tag))
    return finish(c, o, lse, tag)


# Requests of different lengths: a one-token query, lengths off the tile (16/64/128) and page.
CAUSAL = ([77, 1, 130, 33], [300, 257, 130, 40])
FULL = ([50, 1, 19], [200, 333, 16])
VAR = ([77, 1, 130], [300, 257, 130])


def cases():
    for hd in (64, 128, 256):
        for tag in ("f16", "bf16"):
            yield paged(tag, hd, *CAUSAL, True, "ragged4")
            yield paged(tag, hd, *FULL, False, "ragged3")
    # Few query tiles and a long KV: the planner splits the KV, and the merge runs.
    yield paged("f16", 128, [1, 16], [3500, 2200], False, "split")
    yield paged("f16", 128, [4, 1], [3999, 2500], True, "split")
    yield paged("bf16", 64, [8], [3000], True, "split")
    yield paged("f16", 128, [1024, 512], [1024, 1536], True, "real")
    yield paged("f16", 128, *CAUSAL, True, "g1", hq=8, hkv=8)
    yield paged("f16", 128, *CAUSAL, True, "g8", hq=32, hkv=4)
    yield paged("f16", 128, *CAUSAL, True, "p1", ps=1)
    yield paged("f16", 128, *CAUSAL, True, "p64", ps=64)

    yield paged("f16", 128, *VAR, True, "v", variant="softcap", cap=10.0)
    yield paged("bf16", 64, *VAR, False, "v", variant="softcap", cap=5.0)
    yield paged("f16", 128, *VAR, True, "v", variant="window", window=100)
    yield paged("f16", 128, *VAR, False, "v", variant="window", window=64)
    yield paged("bf16", 256, *VAR, True, "v", variant="window", window=48)
    yield paged("f16", 128, *VAR, False, "v", variant="mask")
    yield paged("bf16", 128, *VAR, False, "v", variant="mask")
    yield paged("f16", 128, *VAR, True, "v", variant="alibi")

    for kv_tag in ("e4m3", "e5m2"):
        yield paged("f16", 128, *CAUSAL, True, "ragged4", kv_tag=kv_tag)
        yield paged("f16", 128, *FULL, False, "ragged3", kv_tag=kv_tag)
        yield ragged("f16", 128, *CAUSAL, True, "ragged4", kv_tag=kv_tag)
        yield single("f16", 128, 130, 400, True, "s", kv_tag=kv_tag)
    yield paged("f16", 128, [1, 16], [3500, 2200], False, "split", kv_tag="e4m3")

    for hd in (64, 128, 256):
        for tag in ("f16", "bf16"):
            yield ragged(tag, hd, *CAUSAL, True, "ragged4")
            yield ragged(tag, hd, *FULL, False, "ragged3")
    yield ragged("f16", 128, [1, 16], [3500, 2200], False, "split")
    yield ragged("f16", 128, *VAR, True, "v", variant="softcap", cap=10.0)
    yield ragged("f16", 128, *VAR, False, "v", variant="mask")

    for hd in (64, 128, 256):
        yield single("f16", hd, 200, 200, True, "s")
        yield single("f16", hd, 1, 777, False, "s")
    yield single("bf16", 128, 130, 400, True, "s")
    yield single("bf16", 64, 33, 1000, False, "s")
    yield single("f16", 128, 8, 4000, True, "split")
    # Causal queries after a cached prefix, few enough tiles that the planner splits the KV: the
    # early queries see nothing of the last chunks, which must then count for nothing in the merge.
    # (Appended last, so the cases above keep their random inputs.)
    yield paged("f16", 128, [100], [400], True, "prefix", hq=8, hkv=8)
    yield paged("f16", 128, [150, 70], [160, 75], True, "prefix2", hq=4, hkv=1)
    yield paged("bf16", 64, [300], [900], True, "prefix", hq=4, hkv=2)
    yield paged("f16", 128, [100], [400], True, "prefix", hq=8, hkv=8, kv_tag="e4m3")
    yield ragged("f16", 128, [200, 60], [500, 130], True, "prefix", hq=4, hkv=2)
    yield single("f16", 128, 700, 3000, True, "prefix")
