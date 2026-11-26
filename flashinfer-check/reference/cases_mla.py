# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.mla.BatchMLAPagedAttentionWrapper` (DeepSeek MLA, head dims 512 + 64, Float16).

Decode-like batches (one query token per request) and prefill-like ones (several), causal or
not, several head counts and page sizes, ragged lengths. The output adds up many terms:
`close:1e-3:2e-3`; the base-2 log-sum-exp is Float32: `close:1e-4:1e-4`.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
CKV, KPE = 512, 64
WS = None


def workspace():
    global WS
    if WS is None:
        WS = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=DEV)
    return WS


def mla_case(name, heads, page, qo_lens, kv_lens, causal, qmul=2.0):
    batch = len(kv_lens)
    pages = [-(-n // page) for n in kv_lens]
    num_pages = sum(pages) + 2
    qo_indptr = torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32, device=DEV)
    kv_indptr = torch.tensor([0] + list(torch.tensor(pages).cumsum(0)), dtype=torch.int32, device=DEV)
    kv_indices = torch.randperm(num_pages, dtype=torch.int32, device=DEV)[: int(kv_indptr[-1])].contiguous()
    kv_len_arr = torch.tensor(kv_lens, dtype=torch.int32, device=DEV)
    n = int(qo_indptr[-1])
    h = torch.float16
    q_nope = (torch.randn(n, heads, CKV, device=DEV) * qmul).to(h)
    q_pe = (torch.randn(n, heads, KPE, device=DEV) * qmul).to(h)
    ckv = torch.randn(num_pages, page, CKV, device=DEV).to(h)
    kpe = torch.randn(num_pages, page, KPE, device=DEV).to(h)
    assert ckv.numel() * 2 < 45e6, name
    sm_scale = 1.0 / (CKV + KPE) ** 0.5
    c = Case(f"mla_{name}", "mla", heads=heads, page_size=page, causal=causal, qo_lens=qo_lens, kv_lens=kv_lens,
             sm_scale=sm_scale)
    for nm, t in (("q_nope", q_nope), ("q_pe", q_pe), ("ckv", ckv), ("kpe", kpe), ("qo_indptr", qo_indptr),
                  ("kv_indptr", kv_indptr), ("kv_indices", kv_indices)):
        c.input(nm, t)
    w = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace(), backend="fa2")
    w.plan(qo_indptr, kv_indptr, kv_indices, kv_len_arr, heads, CKV, KPE, page, causal, sm_scale, h, h)
    o, lse = w.run(q_nope, q_pe, ckv, kpe, return_lse=True)
    c.output("o", o, "close:1e-3:2e-3")
    c.output("lse", lse, "close:1e-4:1e-4")
    return c


MIXED = [1, 17, 300, 1000, 2049, 64]


def cases():
    # Decode-like: one query token per request.
    yield mla_case("decode_h128_p1", 128, 1, [1] * 6, MIXED, False)
    yield mla_case("decode_h64_p1_b1", 64, 1, [1], [3000], False)
    yield mla_case("decode_h16_p16", 16, 16, [1] * 6, MIXED, False)
    yield mla_case("decode_h16_p16_causal", 16, 16, [1] * 6, MIXED, True)
    yield mla_case("decode_h32_p64", 32, 64, [1] * 6, MIXED, False)
    yield mla_case("decode_h128_p1_b1_len1", 128, 1, [1], [1], False)
    yield mla_case("decode_h8_p1_batch40", 8, 1, [1] * 40, [100 + 25 * i for i in range(40)], False)
    # Prefill-like: several query tokens per request.
    for causal in (False, True):
        tag = "causal" if causal else "full"
        yield mla_case(f"prefill_h16_p16_{tag}", 16, 16, [16, 40, 128, 3], [64, 200, 500, 3], causal)
        yield mla_case(f"prefill_h64_p1_{tag}", 64, 1, [8, 1, 33], [100, 700, 33], causal)
        yield mla_case(f"prefill_h128_p16_{tag}", 128, 16, [5, 20], [300, 20], causal)
