# SPDX-License-Identifier: Apache-2.0
"""The port of `flashinfer.BatchAttention`: mixed prefill and decode requests over one paged cache.

FlashInfer's own `BatchAttention` kernel cannot launch on this GPU (SM89), so the reference is
`BatchPrefillWithPagedKVCacheWrapper` on the same requests (a decode request is a prefill with one
query token); the cases say so with `reference=batch_prefill`. Float16, head dim 128: attention outputs
add up many terms, so `close:1e-3:2e-3` on the output and `close:1e-4:1e-4` on the base-2 lse.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
DT = torch.float16
HD = 128
O_RULE, LSE_RULE = "close:1e-3:2e-3", "close:1e-4:1e-4"
# Queries scaled so that attention is peaked and the outputs are of order 0.1 to 1.
QS = 2.0


def mixed_case(name, qo_lens, kv_lens, ps, hq, hkv, causal):
    c = Case(f"attention_{name}", "attention", reference="batch_prefill", qo_lens=qo_lens, kv_lens=kv_lens,
             page_size=ps, hq=hq, hkv=hkv, causal=causal)
    pages = [-(-n // ps) for n in kv_lens]
    num_pages = sum(pages) + 3
    ids = torch.randperm(num_pages, dtype=torch.int32, device=DEV)
    qo_indptr = torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32, device=DEV)
    kv_indptr = torch.tensor([0] + list(torch.tensor(pages).cumsum(0)), dtype=torch.int32, device=DEV)
    kv_indices = ids[: sum(pages)].contiguous()
    last = torch.tensor([(n - 1) % ps + 1 for n in kv_lens], dtype=torch.int32, device=DEV)
    q = c.input("q", (torch.randn(sum(qo_lens), hq, HD, device=DEV) * QS).to(DT))
    cache = c.input("cache", torch.randn(num_pages, 2, ps, hkv, HD, device=DEV).to(DT))
    for n, t in (("qo_indptr", qo_indptr), ("kv_indptr", kv_indptr), ("kv_indices", kv_indices), ("last", last)):
        c.input(n, t)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    w.plan(qo_indptr, kv_indptr, kv_indices, last, hq, hkv, HD, ps, causal=causal, q_data_type=DT, kv_data_type=DT)
    o, lse = w.run(q, cache, return_lse=True)
    c.output("o", o, O_RULE)
    c.output("lse", lse, LSE_RULE)
    return c


MIX = ([1, 1, 1, 1, 300, 17, 1, 64], [500, 1, 333, 2048, 300, 100, 17, 1000])


def cases():
    yield mixed_case("mix_p16_causal", *MIX, 16, 32, 8, True)
    yield mixed_case("mix_p16_full", *MIX, 16, 32, 8, False)
    yield mixed_case("mix_p1_causal", *MIX, 1, 32, 8, True)
    yield mixed_case("mix_p64_causal", *MIX, 64, 32, 8, True)
    yield mixed_case("mix_group1_p16", *MIX, 16, 8, 8, True)
    yield mixed_case("mix_group8_p16", *MIX, 16, 32, 4, True)
    yield mixed_case("decodes_only_p16", [1] * 24, [64 + 211 * i for i in range(24)], 16, 32, 8, True)
    yield mixed_case("one_prefill_p16", [700], [700], 16, 32, 8, True)
    yield mixed_case("serving_p16", [1] * 32 + [512], [2048] * 32 + [1024], 16, 32, 8, True)
    yield mixed_case("long_decode_p16", [1, 1, 1024], [4000, 3500, 1024], 16, 64, 8, True)
