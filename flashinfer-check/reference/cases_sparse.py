# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.sparse`: `BlockSparseAttentionWrapper` (BSR, with and without a custom mask) and
`VariableBlockSparseAttentionWrapper`, `fa2` backend, Float16.

Attention outputs add up many terms: `close:1e-3:2e-3` on the output, `close:1e-4:1e-4` on the base-2
log-sum-exp. FlashInfer's CUDA-core path for `R * group < 4` is not ported, so no case has one.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
DT = torch.float16
O_RULE, LSE_RULE = "close:1e-3:2e-3", "close:1e-4:1e-4"
# Queries scaled so that attention is peaked and the outputs are of order 0.1 to 1.
QS = 2.0


def random_bsr(mb, nb, density, empty_rows=()):
    """BSR indptr and indices: each block kept with probability `density`, at least one per row."""
    indptr, indices = [0], []
    for i in range(mb):
        if i in empty_rows:
            cols = []
        else:
            keep = torch.rand(nb) < density
            keep[torch.randint(0, nb, (1,))] = True
            cols = keep.nonzero().flatten().tolist()
        indices += cols
        indptr.append(len(indices))
    return (torch.tensor(indptr, dtype=torch.int32, device=DEV), torch.tensor(indices, dtype=torch.int32, device=DEV))


def bsr_case(name, m, n, r, c, hq, hkv, hd, density, mask=False, empty_rows=()):
    mb, nb = m // r, n // c
    cs = Case(f"sparse_bsr_{name}", "sparse", kind="bsr", M=m, N=n, R=r, C=c, hq=hq, hkv=hkv, head_dim=hd,
              mask=mask)
    indptr, indices = random_bsr(mb, nb, density, empty_rows)
    nnz = indices.numel()
    q = cs.input("q", (torch.randn(m, hq, hd, device=DEV) * QS).to(DT))
    k = cs.input("k", torch.randn(n, hkv, hd, device=DEV).to(DT))
    v = cs.input("v", torch.randn(n, hkv, hd, device=DEV).to(DT))
    cs.input("indptr", indptr)
    cs.input("indices", indices)
    bits = None
    if mask:
        bits = torch.rand(nnz, r, c, device=DEV) < 0.5
        # Every query row keeps a token in its block row's first block.
        for i in range(mb):
            if indptr[i + 1] > indptr[i]:
                bits[int(indptr[i]), :, 0] = True
        cs.input("mask_bits", bits)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    w = flashinfer.sparse.BlockSparseAttentionWrapper(ws, backend="fa2")
    w.plan(indptr, indices, m, n, r, c, hq, hkv, hd, mask=bits, q_data_type=DT)
    o, lse = w.run(q, k, v, return_lse=True)
    cs.output("o", o, O_RULE)
    cs.output("lse", lse, LSE_RULE)
    return cs


def sizes(total, blocks, ragged):
    """`blocks` sizes summing to `total`: equal, or random (each at least 1)."""
    if not ragged:
        s = [total // blocks] * blocks
        s[-1] = total - sum(s[:-1])
        return s
    cuts = sorted(torch.randperm(total - 1)[: blocks - 1].add(1).tolist())
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def variable_case(name, qo, kv, hq, hkv, hd, mb, nb, density, ragged):
    cs = Case(f"sparse_variable_{name}", "sparse", kind="variable", qo_len=qo, kv_len=kv, hq=hq, hkv=hkv,
              head_dim=hd, mb=mb, nb=nb)
    rows = torch.tensor([sizes(qo, mb, ragged) for _ in range(hkv)], dtype=torch.int32)
    cols = torch.tensor([sizes(kv, nb, ragged) for _ in range(hkv)], dtype=torch.int32)
    mask = torch.rand(hkv, mb, nb) < density
    mask[torch.arange(hkv)[:, None], torch.arange(mb)[None, :], torch.randint(0, nb, (hkv, mb))] = True
    q = cs.input("q", (torch.randn(hq, qo, hd, device=DEV) * QS).to(DT))
    k = cs.input("k", torch.randn(hkv, kv, hd, device=DEV).to(DT))
    v = cs.input("v", torch.randn(hkv, kv, hd, device=DEV).to(DT))
    cs.input("block_mask", mask.to(DEV))
    cs.input("row_sz", rows.to(DEV))
    cs.input("col_sz", cols.to(DEV))
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    w = flashinfer.sparse.VariableBlockSparseAttentionWrapper(ws, backend="fa2")
    w.plan(mask.to(DEV), rows.to(DEV), cols.to(DEV), hq, hkv, hd, q_data_type=DT)
    o, lse = w.run(q, k, v, return_lse=True)
    cs.output("o", o, O_RULE)
    cs.output("lse", lse, LSE_RULE)
    return cs


def cases():
    # name, M, N, R, C, query heads, KV heads, head dim, density
    yield bsr_case("realistic_64x64_d128", 2048, 2048, 64, 64, 32, 8, 128, 0.3)
    yield bsr_case("64x64_d64", 1024, 1024, 64, 64, 32, 8, 64, 0.5)
    yield bsr_case("16x16_d128", 512, 512, 16, 16, 32, 8, 128, 0.4)
    yield bsr_case("128x32_d64", 1024, 768, 128, 32, 16, 4, 64, 0.5)
    yield bsr_case("32x16_wide_d64", 384, 1024, 32, 16, 16, 4, 64, 0.3)
    yield bsr_case("8x128_d128", 256, 1536, 8, 128, 32, 8, 128, 0.4)
    yield bsr_case("2x8_d64", 64, 256, 2, 8, 16, 4, 64, 0.4)
    yield bsr_case("16x16_group1_d128", 512, 512, 16, 16, 8, 8, 128, 0.4)
    yield bsr_case("16x16_group8_d64", 512, 512, 16, 16, 32, 4, 64, 0.4)
    yield bsr_case("64x64_empty_row", 512, 512, 64, 64, 32, 8, 128, 0.5, empty_rows=(2,))
    yield bsr_case("16x16_mask_d128", 512, 512, 16, 16, 32, 8, 128, 0.4, mask=True)
    yield bsr_case("64x64_mask_d64", 512, 512, 64, 64, 32, 8, 64, 0.5, mask=True)
    yield bsr_case("32x16_mask_d128", 384, 512, 32, 16, 16, 4, 128, 0.4, mask=True)

    yield variable_case("uniform_d128", 1024, 1024, 32, 32, 128, 8, 16, 0.4, False)
    yield variable_case("ragged_d64", 700, 900, 32, 8, 64, 7, 11, 0.4, True)
    yield variable_case("ragged_group4_d128", 512, 640, 16, 4, 128, 10, 9, 0.5, True)
    yield variable_case("group1_d64", 400, 400, 8, 8, 64, 5, 6, 0.5, True)
