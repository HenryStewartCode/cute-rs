# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.decode`: `BatchDecodeWithPagedKVCacheWrapper` (plan, then run) and `single_decode_with_kv_cache`.

Float16 and BFloat16, head dim 64/128/256, 4 query heads per KV head (the port's group), FP8
(E4M3/E5M2) caches with `k_scale`/`v_scale`, soft-cap, sliding window and ALiBi, ragged lengths
(some of one token next to some of thousands, so the port's split-KV path and merge run).
Outputs add up many terms: `close`, the output's rule by type; the base-2 log-sum-exp is Float32.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
GROUP = 4
F16, BF16, E4, E5 = (torch.float16, "f16"), (torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "e4m3"), (torch.float8_e5m2, "e5m2")
OUT_RULE = {"f16": "close:1e-3:2e-3", "bf16": "close:8e-3:1.6e-2", "e4m3": "close:1e-3:2e-3", "e5m2": "close:1e-3:2e-3"}
LSE_RULE = "close:1e-4:1e-4"
# One workspace for every wrapper; FlashInfer's plan keeps its own copy of what it needs.
WS = None


def workspace():
    global WS
    if WS is None:
        WS = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    return WS


def page_table(lens, ps):
    pages = [-(-n // ps) for n in lens]
    num_pages = sum(pages) + 2  # two pages no request uses
    perm = torch.randperm(num_pages, dtype=torch.int32, device=DEV)
    indptr = torch.tensor([0] + list(torch.tensor(pages).cumsum(0)), dtype=torch.int32, device=DEV)
    indices = perm[: int(indptr[-1])].contiguous()
    last = torch.tensor([n - (p - 1) * ps for n, p in zip(lens, pages)], dtype=torch.int32, device=DEV)
    return num_pages, indptr, indices, last


def meta(op, tag, kv_tag, hd, hkv, lens, ps, cap, window, alibi, k_scale, v_scale):
    return dict(op=op, dtype=tag, kv_dtype=kv_tag, head_dim=hd, kv_heads=hkv, kv_lens=lens, page_size=ps,
                soft_cap=cap, window=window, alibi=alibi, k_scale=k_scale, v_scale=v_scale,
                sm_scale=hd ** -0.5)


def batch_case(name, qdt, kvdt, hd, hkv, lens, ps, cap=0.0, window=-1, alibi=False, k_scale=1.0, v_scale=1.0,
               qmul=1.5):
    qtype, qtag = qdt
    kvtype, kvtag = kvdt
    num_pages, indptr, indices, last = page_table(lens, ps)
    hqo = hkv * GROUP
    q = (torch.randn(len(lens), hqo, hd, device=DEV) * qmul).to(qtype)
    k = torch.randn(num_pages, ps, hkv, hd, device=DEV).to(kvtype)
    v = torch.randn(num_pages, ps, hkv, hd, device=DEV).to(kvtype)
    assert k.numel() * k.element_size() < 45e6, name
    c = Case(f"decode_batch_{name}", "decode",
             **meta("batch", qtag, kvtag, hd, hkv, lens, ps, cap, window, alibi, k_scale, v_scale))
    for n, t in (("q", q), ("k_cache", k), ("v_cache", v), ("kv_indptr", indptr), ("kv_indices", indices),
                 ("kv_last_page_len", last)):
        c.input(n, t)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace(), "NHD", use_tensor_cores=False, backend="fa2")
    w.plan(indptr, indices, last, hqo, hkv, hd, ps,
           pos_encoding_mode="ALIBI" if alibi else "NONE", window_left=window,
           logits_soft_cap=cap if cap > 0 else None, q_data_type=qtype, kv_data_type=kvtype)
    kw = {}
    if kvtype != qtype:
        kw = dict(k_scale=k_scale, v_scale=v_scale)
    o, lse = w.run(q, (k, v), return_lse=True, **kw)
    c.output("o", o, OUT_RULE[qtag])
    c.output("lse", lse, LSE_RULE)
    return c


def single_case(name, qdt, hd, hkv, n, cap=0.0, window=-1, alibi=False, qmul=1.5):
    qtype, qtag = qdt
    hqo = hkv * GROUP
    q = (torch.randn(hqo, hd, device=DEV) * qmul).to(qtype)
    k = torch.randn(n, hkv, hd, device=DEV).to(qtype)
    v = torch.randn(n, hkv, hd, device=DEV).to(qtype)
    c = Case(f"decode_single_{name}", "decode", **meta("single", qtag, qtag, hd, hkv, [n], 0, cap, window, alibi, 1.0, 1.0))
    for nm, t in (("q", q), ("k", k), ("v", v)):
        c.input(nm, t)
    o, lse = flashinfer.single_decode_with_kv_cache(
        q, k, v, "NHD", "ALIBI" if alibi else "NONE", window_left=window,
        logits_soft_cap=cap if cap > 0 else None, return_lse=True)
    c.output("o", o, OUT_RULE[qtag])
    c.output("lse", lse, LSE_RULE)
    return c


# Lengths of one token next to thousands, and none a multiple of the page.
MIXED = [1, 7, 300, 1500, 4096, 33, 2049, 1]
SHORT = [1, 7, 21, 40, 3, 16, 17, 64]


def cases():
    # Float16 and BFloat16 at each head dim; 32 query heads over 8 KV heads at 128.
    yield batch_case("f16_hd128_mixed_p16", F16, F16, 128, 8, MIXED, 16)
    yield batch_case("bf16_hd128_mixed_p16", BF16, BF16, 128, 8, MIXED, 16)
    yield batch_case("f16_hd128_short_p4", F16, F16, 128, 2, SHORT, 4)
    yield batch_case("f16_hd64_mixed_p8", F16, F16, 64, 4, [100, 1, 5000, 64, 2000, 9], 8)
    yield batch_case("bf16_hd64_mixed_p16", BF16, BF16, 64, 8, [1, 3000, 17, 800, 1], 16)
    yield batch_case("f16_hd256_mixed_p16", F16, F16, 256, 2, [1, 17, 1025, 3000, 600], 16)
    yield batch_case("bf16_hd256_mixed_p32", BF16, BF16, 256, 4, [5, 2048, 1, 333], 32)
    # A realistic batch: many requests of a few hundred to a thousand tokens.
    yield batch_case("f16_hd128_batch24_p16", F16, F16, 128, 8, [200 + 37 * i for i in range(24)], 16)
    yield batch_case("bf16_hd128_batch24_p64", BF16, BF16, 128, 8, [150 + 41 * i for i in range(24)], 64)
    # Batch one, and a page size that is not a power of two.
    yield batch_case("f16_hd128_b1_p16", F16, F16, 128, 8, [4096], 16)
    yield batch_case("f16_hd128_b1_len37_p16", F16, F16, 128, 8, [37], 16)
    yield batch_case("bf16_hd128_b1_p1", BF16, BF16, 128, 4, [1000], 1)
    yield batch_case("f16_hd128_mixed_p7", F16, F16, 128, 4, MIXED, 7)
    # FP8 caches with scales (16-bit queries and output).
    for kv in (E4, E5):
        yield batch_case(f"f16_{kv[1]}_hd128_mixed_p16", F16, kv, 128, 8, MIXED, 16, k_scale=0.5, v_scale=2.0)
        yield batch_case(f"f16_{kv[1]}_hd128_batch1_p16", F16, kv, 128, 8, [3000], 16, k_scale=1.5, v_scale=0.75)
    yield batch_case("f16_e4m3_hd128_short_p4", F16, E4, 128, 2, SHORT, 4, k_scale=1.0, v_scale=1.0)
    # Variants.
    yield batch_case("f16_hd128_cap_mixed", F16, F16, 128, 8, MIXED, 16, cap=4.0, qmul=3.0)
    yield batch_case("f16_hd128_window_mixed", F16, F16, 128, 8, MIXED, 16, window=100)
    yield batch_case("f16_hd128_window_short", F16, F16, 128, 2, SHORT, 4, window=10)
    yield batch_case("f16_hd128_cap_window", F16, F16, 128, 8, MIXED, 16, cap=5.0, window=500, qmul=3.0)
    yield batch_case("bf16_hd64_cap_window", BF16, BF16, 64, 8, [1, 3000, 17, 800, 1], 16, cap=5.0, window=64, qmul=3.0)
    yield batch_case("f16_hd128_alibi_mixed", F16, F16, 128, 8, MIXED, 16, alibi=True)
    yield batch_case("f16_hd128_alibi_short", F16, F16, 128, 2, SHORT, 4, alibi=True)

    # single_decode_with_kv_cache.
    yield single_case("f16_hd128_n5000", F16, 128, 8, 5000)
    yield single_case("f16_hd128_n200", F16, 128, 8, 200)
    yield single_case("f16_hd128_n1", F16, 128, 8, 1)
    yield single_case("bf16_hd128_n3001", BF16, 128, 8, 3001)
    yield single_case("f16_hd64_n777", F16, 64, 8, 777)
    yield single_case("bf16_hd64_n1", BF16, 64, 8, 1)
    yield single_case("f16_hd256_n2000", F16, 256, 4, 2000)
    yield single_case("bf16_hd256_n300", BF16, 256, 4, 300)
    yield single_case("f16_hd128_cap_n4000", F16, 128, 8, 4000, cap=4.0, qmul=3.0)
    yield single_case("f16_hd128_window_n4000", F16, 128, 8, 4000, window=300)
    yield single_case("f16_hd128_alibi_n3000", F16, 128, 8, 3000, alibi=True)
