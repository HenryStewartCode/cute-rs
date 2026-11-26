# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.rope` and `flashinfer.page`: the five in-place RoPE functions and `append_paged_kv_cache`.

RoPE is a short fixed computation per element rounded once to Float16 or BFloat16: `ulp:1`.
The append is a pure scatter: `exact`, on the whole cache so that untouched slots count too.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
F16, BF16 = (torch.float16, "f16"), (torch.bfloat16, "bf16")
# (dtype, head_dim, interleave, (tokens, q heads, k heads), rotary_dim fraction of head_dim)
COMBOS = (
    (F16, 128, False, (1, 8, 8), 1),
    (BF16, 128, False, (7, 32, 8), 1),
    (F16, 64, True, (129, 16, 4), 1),
    (BF16, 64, True, (1024, 32, 8), 0.5),
    (F16, 128, True, (33, 8, 8), 0.5),
    (BF16, 128, False, (1024, 32, 8), 0.25),
    (F16, 64, False, (300, 4, 4), 0.5),
    (BF16, 128, True, (100, 16, 16), 1),
)
MAX_POS = 4096


def rand(shape, dt):
    return torch.randn(*shape, dtype=dt, device=DEV)


def ragged(nnz):
    """indptr and offsets of three requests that share `nnz` tokens (the last may be empty-free)."""
    a = max(1, nnz // 3)
    b = max(1, nnz // 2)
    cuts = sorted({0, min(a, nnz), min(a + b, nnz), nnz})
    indptr = torch.tensor(cuts, dtype=torch.int32, device=DEV)
    offsets = torch.randint(0, 2000, (len(cuts) - 1,), dtype=torch.int32, device=DEV)
    return indptr, offsets


def cos_sin_cache(hd):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32, device=DEV) / hd))
    fr = torch.outer(torch.arange(MAX_POS, dtype=torch.float32, device=DEV), inv)
    return torch.cat([fr.cos(), fr.sin()], -1)


def cases():
    for i, ((dt, tag), hd, inter, (n, hq, hk), frac) in enumerate(COMBOS):
        shape = dict(dtype=tag, head_dim=hd, interleave=inter, nnz=n, hq=hq, hk=hk)
        rd = int(hd * frac)
        name = f"{tag}_{hd}_{'gptj' if inter else 'neox'}_{n}x{hq}x{hk}"

        q, k = rand((n, hq, hd), dt), rand((n, hk, hd), dt)
        pos = torch.randint(0, MAX_POS, (n,), dtype=torch.int32, device=DEV)
        indptr, offsets = ragged(n)

        # cos/sin cache: query and key are flattened [nnz, heads * head_dim]; rotary_dim = head_dim.
        c = Case(f"rope_cache_{name}", "rope", op="cache", rotary_dim=hd, **shape)
        qs, ks = c.input("q", q).clone(), c.input("k", k).clone()
        c.input("positions", pos)
        cache = cos_sin_cache(hd)
        c.input("cos_sin_cache", cache)
        qf, kf = qs.view(n, hq * hd), ks.view(n, hk * hd)
        flashinfer.apply_rope_with_cos_sin_cache_inplace(pos.long(), qf, kf, hd, cache, not inter)
        c.output("q", qs, "ulp:1")
        c.output("k", ks, "ulp:1")
        yield c

        for op, llama in (("pos_ids", False), ("pos_ids_llama31", True)):
            c = Case(f"rope_{op}_{name}_r{rd}", "rope", op=op, rotary_dim=rd, **shape)
            qs, ks = c.input("q", q).clone(), c.input("k", k).clone()
            c.input("pos_ids", pos)
            fn = flashinfer.apply_llama31_rope_pos_ids_inplace if llama else flashinfer.apply_rope_pos_ids_inplace
            fn(qs, ks, pos, rotary_dim=rd, interleave=inter)
            c.output("q", qs, "ulp:1")
            c.output("k", ks, "ulp:1")
            yield c

        for op, llama in (("indptr", False), ("indptr_llama31", True)):
            c = Case(f"rope_{op}_{name}_r{rd}", "rope", op=op, rotary_dim=rd, **shape)
            qs, ks = c.input("q", q).clone(), c.input("k", k).clone()
            c.input("indptr", indptr)
            c.input("offsets", offsets)
            fn = flashinfer.apply_llama31_rope_inplace if llama else flashinfer.apply_rope_inplace
            fn(qs, ks, indptr, offsets, rotary_dim=rd, interleave=inter)
            c.output("q", qs, "ulp:1")
            c.output("k", ks, "ulp:1")
            yield c

    # append_paged_kv_cache: (cache dtype, tag), page size, kv heads, head_dim, seq lens after, append lens
    for dt, tag, ps, h, hd, seq_lens, append in (
        (torch.float16, "f16", 16, 8, 128, (45, 8, 25, 22), (45, 8, 25, 22)),
        (torch.bfloat16, "bf16", 1, 4, 64, (5, 1, 9), (2, 1, 3)),
        (torch.float16, "f16", 7, 2, 128, (100, 13, 1), (60, 13, 1)),
        (torch.bfloat16, "bf16", 64, 8, 128, (700, 300), (1, 300)),
        (torch.float8_e4m3fn, "e4m3", 16, 8, 128, (45, 8, 25, 22), (20, 8, 25, 1)),
        (torch.float8_e5m2, "e5m2", 32, 4, 64, (33, 64, 5), (33, 2, 5)),
    ):
        pages = [-(-s // ps) for s in seq_lens]
        num_pages = sum(pages) + 2
        ids = torch.randperm(num_pages, dtype=torch.int32, device=DEV)
        kv_indptr = torch.tensor([0] + list(torch.tensor(pages).cumsum(0)), dtype=torch.int32, device=DEV)
        kv_indices = ids[: int(kv_indptr[-1])].contiguous()
        last = torch.tensor([(s - 1) % ps + 1 for s in seq_lens], dtype=torch.int32, device=DEV)
        app_indptr = torch.tensor([0] + list(torch.tensor(append).cumsum(0)), dtype=torch.int32, device=DEV)
        nnz = int(app_indptr[-1])
        seq = flashinfer.get_seq_lens(kv_indptr, last, ps)
        bi, pos = flashinfer.get_batch_indices_positions(app_indptr, seq, nnz)
        ak = (torch.randn(nnz, h, hd, device=DEV) * 2).to(dt)
        av = (torch.randn(nnz, h, hd, device=DEV) * 2).to(dt)
        # Random contents, so slots the append does not touch are checked too.
        cache = (torch.randn(num_pages, 2, ps, h, hd, device=DEV) * 2).to(dt)
        c = Case(f"rope_append_{tag}_p{ps}_{h}x{hd}_{len(seq_lens)}req", "rope", op="append", dtype=tag,
                 page_size=ps, kv_heads=h, head_dim=hd, nnz=nnz)
        for n, t in (("append_k", ak), ("append_v", av), ("cache", cache), ("kv_indptr", kv_indptr),
                     ("kv_indices", kv_indices), ("batch_indices", bi), ("positions", pos)):
            c.input(n, t)
        out = cache.clone()
        flashinfer.append_paged_kv_cache(ak, av, bi, pos, out, kv_indices, kv_indptr, last)
        c.output("cache", out, "exact")
        yield c
