# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.gemm`: `SegmentGEMMWrapper.run` and `bmm_fp8`.

The segment GEMM runs in Float16 and BFloat16, with row- and column-major weights, with and
without `weight_indices`, and ragged segments (empty and one-row ones included); it accumulates
into `out`, so some cases start from a non-zero `out`. `bmm_fp8` takes E4M3 and E5M2 inputs
with per-tensor scales to BFloat16 and Float16 (E5M2 is refused by FlashInfer's cuBLASLt path on SM89). Weights are scaled by 1/sqrt(fan-in) so the
outputs are of order 1; sums of many terms use `close`.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
F16, BF16 = (torch.float16, "f16"), (torch.bfloat16, "bf16")
RULE = {"f16": "close:1e-3:2e-3", "bf16": "close:8e-3:1.6e-2"}

# (name, segment lengths, d_in, d_out, number of weights, weight_indices, start from a non-zero out)
SEGMENTS = (
    ("small", [5, 0, 3], 64, 128, 3, None, False),
    ("ragged", [37, 128, 1, 300], 512, 768, 6, [2, 2, 0, 5], True),
    ("one", [1], 256, 256, 1, None, True),
    ("empty_first", [0, 129, 64, 1], 384, 256, 5, [4, 1, 3, 0], False),
)
# (name, batch, m, n, k)
BMM = (
    ("b1m64", 1, 64, 512, 512),
    ("b4m16", 4, 16, 256, 1024),
    ("b8m4", 8, 4, 512, 2048),
    ("b2m100", 2, 100, 384, 640),
)


def segment_cases():
    ws = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=DEV)
    wrapper = flashinfer.gemm.SegmentGEMMWrapper(ws, backend="auto")
    for dt, tag in (F16, BF16):
        for col in (False, True):
            for name, lens, d_in, d_out, nw, idx, accumulate in SEGMENTS:
                rows = sum(lens)
                layout = "col" if col else "row"
                c = Case(
                    f"gemm_segment_{name}_{tag}_{layout}", "gemm", op="segment", dtype=tag, d_in=d_in, d_out=d_out,
                    col_major=col, seg_lens=lens, num_weights=nw, indexed=idx is not None,
                )
                x = torch.randn(rows, d_in, device=DEV).to(dt)
                shape = (nw, d_out, d_in) if col else (nw, d_in, d_out)
                w = (torch.randn(*shape, device=DEV) / d_in**0.5).to(dt)
                y = (torch.randn(rows, d_out, device=DEV) * 0.5).to(dt) if accumulate else torch.zeros(rows, d_out, dtype=dt, device=DEV)
                indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64, device=DEV)
                c.input("x", x)
                c.input("w", w)
                c.input("y0", y)
                c.input("indptr", indptr)
                kwargs = {}
                if idx is not None:
                    wi = torch.tensor(idx, dtype=torch.int64, device=DEV)
                    c.input("weight_indices", wi)
                    kwargs["weight_indices"] = wi
                wrapper.run(x, w, len(lens), col, out=y, seg_indptr=indptr, **kwargs)
                c.output("y", y, RULE[tag])
                yield c


def to_float8(x, dtype):
    # Scaled so the largest magnitude is the type's largest value; the scale is its inverse.
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    return (x * scale).clamp(finfo.min, finfo.max).to(dtype), scale.float().reciprocal()


def bmm_cases():
    # cuBLASLt refuses E5M2 for both operands on this GPU ("both A and B are e5m2"), and the port has one
    # FP8 type for both, so only E4M3 is compared.
    for in_dt, in_tag in ((torch.float8_e4m3fn, "e4m3"),):
        for out_dt, out_tag in (BF16, F16):
            for name, b, m, n, k in BMM:
                c = Case(f"gemm_bmm_fp8_{name}_{in_tag}_{out_tag}", "gemm", op="bmm_fp8", fp8=in_tag, dtype=out_tag, batch=b, m=m, n=n, k=k)
                a, a_scale = to_float8(torch.randn(b, m, k, device=DEV), in_dt)
                bm, b_scale = to_float8(torch.randn(b, n, k, device=DEV) / k**0.5, in_dt)
                c.input("a", a)
                # B is column-major: the [batch, n, k] tensor, viewed as [batch, k, n].
                c.input("b", bm)
                c.input("a_scale", a_scale.reshape(1))
                c.input("b_scale", b_scale.reshape(1))
                out = torch.empty(b, m, n, dtype=out_dt, device=DEV)
                flashinfer.bmm_fp8(a, bm.transpose(-2, -1), a_scale, b_scale, out_dt, out=out)
                c.output("out", out, RULE[out_tag])
                yield c


def cases():
    yield from segment_cases()
    yield from bmm_cases()
