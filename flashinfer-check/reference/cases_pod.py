# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.PODWithPagedKVCacheWrapper`: one prefill (contiguous K/V) and a paged decode batch in one launch.

Float16, head dim 128. Both halves' outputs are compared (the wrapper does not return the
log-sum-exp, so that is not checked): attention outputs add up many terms, so `close:1e-3:2e-3`.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
DT = torch.float16
HD = 128
O_RULE = "close:1e-3:2e-3"
# Queries scaled so that attention is peaked and the outputs are of order 0.1 to 1.
QS = 2.0


def pod_case(name, qo, kv, decode_kv, ps, hq, hkv, causal):
    c = Case(f"pod_{name}", "pod", qo_len=qo, kv_len=kv, decode_kv=decode_kv, page_size=ps, hq=hq, hkv=hkv,
             causal=causal)
    q_p = c.input("q_p", (torch.randn(qo, hq, HD, device=DEV) * QS).to(DT))
    k_p = c.input("k_p", torch.randn(kv, hkv, HD, device=DEV).to(DT))
    v_p = c.input("v_p", torch.randn(kv, hkv, HD, device=DEV).to(DT))
    batch = len(decode_kv)
    q_d = c.input("q_d", (torch.randn(batch, hq, HD, device=DEV) * QS).to(DT))
    pages = [-(-n // ps) for n in decode_kv]
    num_pages = sum(pages) + 3
    cache = c.input("cache", torch.randn(num_pages, 2, ps, hkv, HD, device=DEV).to(DT))
    ids = torch.randperm(num_pages, dtype=torch.int32, device=DEV)
    indptr = torch.tensor([0] + list(torch.tensor(pages).cumsum(0)), dtype=torch.int32, device=DEV)
    indices = ids[: sum(pages)].contiguous()
    last = torch.tensor([(n - 1) % ps + 1 for n in decode_kv], dtype=torch.int32, device=DEV)
    for n, t in (("indptr", indptr), ("indices", indices), ("last", last)):
        c.input(n, t)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    pod = flashinfer.PODWithPagedKVCacheWrapper(ws, kv_layout="NHD")
    pod.plan(indptr, indices, last, hq, hkv, HD, ps, q_data_type=DT, kv_data_type=DT)
    o_p, o_d = pod.run(q_p, k_p, v_p, q_d, cache, causal_p=causal)
    c.output("o_p", o_p, O_RULE)
    c.output("o_d", o_d, O_RULE)
    return c


def cases():
    yield pod_case("512_dec8x2048_p16", 512, 512, [2048] * 8, 16, 32, 8, True)
    yield pod_case("77_300_ragged_p16", 77, 300, [1, 17, 300, 1000, 257], 16, 32, 8, True)
    yield pod_case("noncausal_p16", 130, 400, [64, 500, 33], 16, 32, 8, False)
    yield pod_case("1_token_p1", 1, 256, [100, 4, 1], 1, 32, 8, True)
    yield pod_case("p64_group8", 256, 1024, [700, 1500, 64, 65], 64, 32, 4, True)
    yield pod_case("group1_p16", 200, 200, [300, 40], 16, 8, 8, True)
    yield pod_case("long_decode_p16", 1024, 1024, [4096] * 4, 16, 32, 8, True)
