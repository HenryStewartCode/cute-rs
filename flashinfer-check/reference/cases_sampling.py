# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.sampling` and `flashinfer.topk`: the samplers, renorm / mask / softmax, top-k, speculative sampling
and the top-k index transforms.

Seeds: every sampler is given `seed` and `offset` explicitly (an int, or per-row int64 tensors), so FlashInfer takes
nothing from a torch generator and the port is handed the same values. Sampled tokens must be `exact`.

Logits are a random permutation of normal quantiles, so every value in a row is distinct: the k-th largest is
unambiguous and no tie rule can change which elements are kept. FlashInfer's `top_k` and the transforms return
unsorted sets; `top_k` is asked for `sorted=True`, and the transforms' rows are sorted here and in the port's
runner before the comparison.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
S = flashinfer.sampling
SOFT_RULE = "close:1e-6:1e-5"
SQRT2 = 2.0**0.5


def logits_of(batch, d, scale=2.0):
    """`[batch, d]` Float32 logits, distinct in each row, normal(0, scale) in distribution."""
    perm = torch.stack([torch.randperm(d, device=DEV) for _ in range(batch)]).double()
    q = torch.erfinv(2.0 * (perm + 0.5) / d - 1.0) * SQRT2 * scale
    return q.float().contiguous()


def probs_of(batch, d, scale=2.0):
    return torch.softmax(logits_of(batch, d, scale), dim=-1).float().contiguous()


def seeds(c, batch, per_row, base):
    """The case's seed and offset: scalars in the meta, or per-row int64 tensors saved as inputs."""
    if per_row:
        g = torch.Generator().manual_seed(base)
        seed = torch.randint(1, 1 << 40, (batch,), generator=g, dtype=torch.int64).to(DEV)
        off = (torch.randint(0, 64, (batch,), generator=g, dtype=torch.int64) * 4).to(DEV)
        c.input("seed_arr", seed)
        c.input("offset_arr", off)
        return seed, off
    seed, off = 1000 + base, 4 * (base % 7)
    c.meta["seed"], c.meta["offset"] = seed, off
    return seed, off


def per_row_ks(batch, d):
    return torch.tensor([min(d, 1 + (b * 97) % 1500) for b in range(batch)], dtype=torch.int32, device=DEV)


def per_row_ps(batch, lo=0.15):
    return torch.tensor([lo + (0.99 - lo) * b / max(batch - 1, 1) for b in range(batch)], dtype=torch.float32, device=DEV)


def sampler_cases():
    n = 0
    for batch, d in ((1, 32000), (5, 128256), (24, 128256)):
        for per_row_seed in (False, True):
            if per_row_seed != (batch == 5):
                continue
            n += 1
            c = Case(f"sampling_sample_{batch}x{d}_{'arr' if per_row_seed else 'scalar'}", "sampling",
                     fn="sample", batch=batch, d=d, per_seed=per_row_seed)
            probs = c.input("probs", probs_of(batch, d))
            seed, off = seeds(c, batch, per_row_seed, n)
            c.output("out", S.sampling_from_probs(probs, seed=seed, offset=off), "exact")
            yield c

    # top-k / top-p / min-p from probabilities, scalar and per-row parameters.
    for batch, d, k, p, per_row in (
        (1, 32000, 50, 0.9, False),
        (5, 128256, 1000, 0.5, False),
        (19, 128256, 10, 0.95, False),
        (6, 128256, 0, 0.0, True),
    ):
        for fn in ("top_k", "top_p", "top_k_top_p_joint", "min_p"):
            n += 1
            tag = "arr" if per_row else "scalar"
            c = Case(f"sampling_{fn}_{batch}x{d}_{tag}", "sampling", fn=fn, batch=batch, d=d, per_row=per_row)
            probs = c.input("probs", probs_of(batch, d))
            seed, off = seeds(c, batch, False, n)
            if per_row:
                kk = c.input("top_k_arr", per_row_ks(batch, d))
                pp = c.input("top_p_arr", per_row_ps(batch, 0.15 if fn != "min_p" else 0.02))
            else:
                kk, pp = k, p
                c.meta["top_k"], c.meta["top_p"] = k, (0.1 if fn == "min_p" else p)
                if fn == "min_p":
                    pp = 0.1
            if fn == "top_k":
                out = S.top_k_sampling_from_probs(probs, kk, seed=seed, offset=off)
            elif fn == "top_p":
                out = S.top_p_sampling_from_probs(probs, pp, seed=seed, offset=off)
            elif fn == "min_p":
                out = S.min_p_sampling_from_probs(probs, pp, seed=seed, offset=off)
            else:
                out = S.top_k_top_p_sampling_from_probs(probs, kk, pp, filter_apply_order="joint", seed=seed, offset=off)
            c.output("out", out, "exact")
            yield c

    # Per-row seed and offset arrays on the filtered samplers.
    for fn in ("top_p",):
        batch, d = 7, 32000
        n += 1
        c = Case(f"sampling_{fn}_{batch}x{d}_seedarr", "sampling", fn=fn, batch=batch, d=d, per_row=False,
                 per_seed=True, top_k=40, top_p=0.8)
        probs = c.input("probs", probs_of(batch, d))
        seed, off = seeds(c, batch, True, n)
        if fn == "top_p":
            out = S.top_p_sampling_from_probs(probs, 0.8, seed=seed, offset=off)
        else:
            out = S.top_k_top_p_sampling_from_probs(probs, 40, 0.8, filter_apply_order="joint", seed=seed, offset=off)
        c.output("out", out, "exact")
        yield c

    # Sampling from logits (Gumbel-max).
    for batch, d, per_row_seed in ((1, 32000, False), (5, 128256, False), (6, 128256, True)):
        n += 1
        c = Case(f"sampling_sample_logits_{batch}x{d}_{'arr' if per_row_seed else 'scalar'}", "sampling",
                 fn="sample_logits", batch=batch, d=d, per_seed=per_row_seed)
        logits = c.input("logits", logits_of(batch, d))
        seed, off = seeds(c, batch, per_row_seed, n)
        c.output("out", S.sampling_from_logits(logits, seed=seed, offset=off), "exact")
        yield c

    # top_k_top_p from probs and logits, both filter orders. Vocab 128256 with k <= 256 takes FlashInfer's compact
    # fast path (radix top-k, then top-p over the k kept); 32000 and larger k take the renorm / mask fallback.
    for src in ("probs", "logits"):
        for order, batch, d, k, p, per_row in (
            ("joint", 4, 32000, 50, 0.9, False),
            ("joint", 3, 128256, 200, 0.7, False),
            ("top_k_first", 1, 32000, 50, 0.9, False),
            ("top_k_first", 19, 32000, 300, 0.8, False),
            ("top_k_first", 4, 128256, 1000, 0.9, False),
            ("top_k_first", 6, 32000, 0, 0.0, True),
            ("top_k_first", 4, 128256, 50, 0.9, False),
            ("top_k_first", 20, 128256, 20, 0.95, False),
        ):
            n += 1
            tag = "arr" if per_row else f"k{k}"
            c = Case(f"sampling_kp_{src}_{order}_{batch}x{d}_{tag}", "sampling", fn="top_k_top_p_" + src, order=order,
                     batch=batch, d=d, per_row=per_row, top_k=k, top_p=p)
            x = c.input("x", probs_of(batch, d) if src == "probs" else logits_of(batch, d))
            seed, off = seeds(c, batch, False, n)
            kk, pp = k, p
            if per_row:
                kk = c.input("top_k_arr", per_row_ks(batch, d))
                pp = c.input("top_p_arr", per_row_ps(batch))
            fn = S.top_k_top_p_sampling_from_probs if src == "probs" else S.top_k_top_p_sampling_from_logits
            c.output("out", fn(x, kk, pp, filter_apply_order=order, seed=seed, offset=off), "exact")
            yield c


def op_cases():
    for batch, d, k, p, per_row in (
        (1, 32000, 50, 0.9, False),
        (4, 128256, 1000, 0.5, False),
        (19, 32000, 0, 0.0, True),
    ):
        tag = "arr" if per_row else "scalar"
        for fn in ("top_p_renorm", "top_k_renorm", "top_k_mask"):
            c = Case(f"sampling_{fn}_{batch}x{d}_{tag}", "sampling", fn=fn, batch=batch, d=d, per_row=per_row)
            x = c.input("x", logits_of(batch, d) if fn == "top_k_mask" else probs_of(batch, d))
            if per_row:
                kk = c.input("top_k_arr", per_row_ks(batch, d))
                pp = c.input("top_p_arr", per_row_ps(batch))
            else:
                kk, pp = k, p
                c.meta["top_k"], c.meta["top_p"] = k, p
            if fn == "top_p_renorm":
                c.output("out", S.top_p_renorm_probs(x, pp), SOFT_RULE)
            elif fn == "top_k_renorm":
                c.output("out", S.top_k_renorm_probs(x, kk), SOFT_RULE)
            else:
                c.output("out", S.top_k_mask_logits(x, kk), "exact")
            yield c

    # softmax with a temperature (a number, none, or one per row).
    for batch, d, temp in ((1, 32000, 0.7), (19, 32000, 2.0), (5, 128256, None), (6, 32000, "arr")):
        tag = "arr" if temp == "arr" else ("none" if temp is None else f"t{temp}")
        c = Case(f"sampling_softmax_{batch}x{d}_{tag}", "sampling", fn="softmax", batch=batch, d=d,
                 per_row=temp == "arr")
        x = c.input("x", logits_of(batch, d, 3.0))
        if temp == "arr":
            temp = c.input("temperature", torch.tensor([0.3 + 0.5 * b for b in range(batch)], device=DEV))
        elif temp is None:
            c.meta["temperature"] = 1.0
        else:
            c.meta["temperature"] = temp
        c.output("out", S.softmax(x, temp), SOFT_RULE)
        yield c

    # top_k: values and indices, sorted by FlashInfer; the port's unsorted output is sorted in its runner.
    for batch, d, k in ((1, 32000, 50), (19, 32000, 1000), (3, 128256, 2048)):
        c = Case(f"sampling_topk_{batch}x{d}_k{k}", "sampling", fn="topk", batch=batch, d=d, k=k)
        x = c.input("x", logits_of(batch, d))
        values, indices = flashinfer.top_k(x, k, sorted=True)
        c.output("values", values, "exact")
        c.output("indices", indices, "exact")
        yield c


def spec_cases():
    for batch, n, d in ((5, 1, 32000), (19, 4, 32000), (3, 3, 128256)):
        c = Case(f"sampling_spec_{batch}x{n}x{d}", "sampling", fn="spec", batch=batch, n=n, d=d,
                 seed=77 + batch, offset=4)
        base = logits_of(batch * (n + 1), d).view(batch, n + 1, d)
        # The draft is a noisy copy of the target, so some draft tokens are accepted and some are not.
        draft_logits = base[:, :n] + torch.randn(batch, n, d, device=DEV) * 0.7
        target = torch.softmax(base, dim=-1).contiguous()
        draft = torch.softmax(draft_logits, dim=-1).contiguous()
        ids = torch.multinomial(draft.view(batch * n, d), 1).view(batch, n).int()
        c.input("draft_probs", draft)
        c.input("draft_ids", ids)
        c.input("target_probs", target)
        acc = c.input("accepted_init", torch.full((batch,), 3, dtype=torch.int32, device=DEV)).clone()
        emi = c.input("emitted_init", torch.full((batch,), 5, dtype=torch.int32, device=DEV)).clone()
        out, acc, emi = S.chain_speculative_sampling(draft, ids, target, acc, emi, seed=c.meta["seed"],
                                                     offset=c.meta["offset"])
        c.output("out", out, "exact")
        c.output("accepted", acc, "exact")
        c.output("emitted", emi, "exact")
        yield c


def transform_cases():
    g = torch.Generator(device=DEV)
    g.manual_seed(5)
    # (rows, stride, k, lengths, row_starts, page_size)
    shapes = (
        (1, 4096, 256, [4096], [0]),
        (4, 32000, 2048, [32000, 20000, 1500, 2048], [0, 100, 0, 0]),
        (20, 8192, 128, [8192 - 7 * i for i in range(20)], [i * 3 for i in range(20)]),
    )
    for kind in ("ragged", "page"):
        for page_size in ((1,) if kind == "ragged" else (1, 64)):
            for rows, stride, k, lens, starts in shapes:
                if kind == "page" and page_size == 1 and rows != 4:
                    continue
                if page_size > 1:
                    starts = [(s // page_size) * page_size for s in starts]
                name = f"sampling_transform_{kind}{page_size if kind == 'page' else ''}_{rows}x{stride}_k{k}"
                c = Case(name, "sampling", fn="transform_" + kind, rows=rows, stride=stride, k=k, page_size=page_size)
                scores = c.input("scores", logits_of(rows, stride))
                lengths = c.input("lengths", torch.tensor(lens, dtype=torch.int32, device=DEV))
                row_starts = c.input("row_starts", torch.tensor(starts, dtype=torch.int32, device=DEV))
                if kind == "ragged":
                    offsets = c.input("offsets", torch.arange(rows, dtype=torch.int32, device=DEV) * 1000 + 5)
                    out = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k, row_starts=row_starts)
                else:
                    batch = max(1, rows // 2)
                    width = stride // page_size + 2
                    table = c.input("table", torch.randint(0, 100000, (batch, width), dtype=torch.int32, device=DEV,
                                                           generator=g))
                    rtb = c.input("row_to_batch", (torch.arange(rows, dtype=torch.int32, device=DEV) % batch))
                    pts = c.input("pt_row_starts", row_starts // page_size)
                    out = flashinfer.top_k_page_table_transform(scores, table, lengths, k, row_to_batch=rtb,
                                                                row_starts=row_starts, page_table_row_starts=pts,
                                                                page_size=page_size)
                c.output("out", out.sort(dim=1).values, "exact")
                yield c


def cases():
    yield from sampler_cases()
    yield from op_cases()
    yield from spec_cases()
    yield from transform_cases()
