# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.quantization`: `packbits`, `segment_packbits` and `per_token_group_quant_8bit`.

Packed bits are integers, and the port documents the group quantiser as bit-exact (every
step is a Float32 operation both sides do the same way), so everything here is `exact`.
FlashInfer's `cuda` backend of `per_token_group_quant_8bit` takes group size 128 only; the
default `cutile` backend needs the `tileiras` compiler, which was not installed for this run, so group
sizes 64 and 256 cannot be run.
"""
import torch

import flashinfer
from flashinfer.quantization import per_token_group_quant_8bit
from common import Case

DEV = "cuda"
PACK_LENS = (1, 7, 8, 100, 2047, 2048, 2049, 100003)
# Segment lengths: ragged, a segment of one bit, whole bytes, and starts off every 8-byte boundary.
SEGMENTS = ((5, 3, 8, 17, 1, 64, 1000), (2049, 4096, 13), (1, 1, 1, 1), (300, 2000, 7, 2048, 9))
QUANT_SHAPES = ((1, 128), (7, 1024), (300, 2048), (1024, 7168))
OUT_TYPES = (
    ("e4m3", torch.float8_e4m3fn),
    ("e5m2", torch.float8_e5m2),
    ("int8", torch.int8),
)


def cases():
    for order in ("big", "little"):
        for n in PACK_LENS:
            x = torch.rand(n, device=DEV) < 0.5
            c = Case(f"quantization_packbits_{order}_{n}", "quantization", op="packbits", order=order, n=n)
            c.input("x", x)
            c.output("out", flashinfer.packbits(x, order), "exact")
            yield c

        for k, lens in enumerate(SEGMENTS):
            indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=DEV)
            x = torch.rand(int(indptr[-1]), device=DEV) < 0.5
            c = Case(f"quantization_segment_packbits_{order}_{k}", "quantization", op="segment_packbits",
                     order=order, segments=len(lens))
            c.input("x", x)
            c.input("indptr", indptr)
            out, new_indptr = flashinfer.segment_packbits(x, indptr, order)
            c.output("out", out, "exact")
            c.output("new_indptr", new_indptr, "exact")
            yield c

    for dt, tag in ((torch.float16, "f16"), (torch.bfloat16, "bf16")):
        for tokens, hidden in QUANT_SHAPES:
            x = torch.randn(tokens, hidden, dtype=dt, device=DEV) * 3
            x.view(-1)[5::97] *= 40  # outliers set the scale of their groups
            if hidden >= 384:
                x[:, 256:384] = 0  # all-zero groups: the scale is eps / max
            for out_tag, out_dt in OUT_TYPES:
                for column_major, ue8m0 in ((False, False), (True, False), (True, True)):
                    if out_tag != "e4m3" and (ue8m0 or tokens in (1, 1024)):
                        continue  # the other output types need fewer shapes
                    if column_major and not ue8m0 and out_tag == "e5m2":
                        continue
                    eps = 1e-10
                    layout = ("col" if column_major else "row") + ("_ue8m0" if ue8m0 else "")
                    c = Case(f"quantization_group_quant_{tag}_{out_tag}_{layout}_{tokens}x{hidden}", "quantization",
                             op="group_quant", dtype=tag, out=out_tag, tokens=tokens, hidden=hidden, group_size=128,
                             column_major=column_major, ue8m0=ue8m0, eps=eps)
                    c.input("x", x)
                    q, s = per_token_group_quant_8bit(x, 128, eps, out_dt, column_major, False, ue8m0, backend="cuda")
                    c.output("q", q, "exact")
                    # Column-major scales are stored `[groups, tokens]`; save that memory order.
                    c.output("s", s.t() if column_major else s, "exact")
                    yield c

        # A larger eps than the default reaches the clamp of the scale on near-zero groups.
        x = torch.randn(33, 1024, dtype=dt, device=DEV) * 1e-3
        c = Case(f"quantization_group_quant_{tag}_e4m3_row_eps_33x1024", "quantization", op="group_quant", dtype=tag,
                 out="e4m3", tokens=33, hidden=1024, group_size=128, column_major=False, ue8m0=False, eps=1e-2)
        c.input("x", x)
        q, s = per_token_group_quant_8bit(x, 128, 1e-2, torch.float8_e4m3fn, False, False, False, backend="cuda")
        c.output("q", q, "exact")
        c.output("s", s, "exact")
        yield c
