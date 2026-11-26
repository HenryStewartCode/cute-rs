# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.cascade`: `merge_state`, `merge_state_in_place` (also masked) and `merge_states`.

A merge is a short fixed computation per element (two exponentials, a weighted sum, one
rounding), so both the values and the Float32 log-sum-exp are held to `ulp:1`.
"""
import torch

import flashinfer
from common import Case

DEV = "cuda"
DTYPES = ((torch.float16, "f16"), (torch.bfloat16, "bf16"))
# (seq, heads, head_dim): one position, ragged, and a serving-sized batch.
SHAPES = ((1, 8, 64), (37, 16, 128), (129, 4, 256), (512, 32, 128))
# (seq, sets, heads, head_dim); sets >= seq reaches the large-number-of-sets kernel.
STATES = ((16, 4, 8, 128), (3, 9, 16, 64), (1, 33, 8, 128), (100, 7, 4, 256), (64, 64, 8, 128))


def lse(*shape):
    return torch.randn(*shape, dtype=torch.float32, device=DEV) * 3


def cases():
    for dt, tag in DTYPES:
        for seq, heads, hd in SHAPES:
            shape = dict(dtype=tag, seq=seq, heads=heads, head_dim=hd)
            va, vb = (torch.randn(seq, heads, hd, dtype=dt, device=DEV) for _ in range(2))
            sa, sb = lse(seq, heads), lse(seq, heads)

            c = Case(f"cascade_merge_state_{tag}_{seq}x{heads}x{hd}", "cascade", op="merge_state", **shape)
            for n, t in (("v_a", va), ("s_a", sa), ("v_b", vb), ("s_b", sb)):
                c.input(n, t)
            v, s = flashinfer.merge_state(va, sa, vb, sb)
            c.output("v", v, "ulp:1")
            c.output("s", s, "ulp:1")
            yield c

            for masked in (False, True):
                name = "merge_state_in_place_mask" if masked else "merge_state_in_place"
                c = Case(f"cascade_{name}_{tag}_{seq}x{heads}x{hd}", "cascade", op=name, **shape)
                v, s = c.input("v", va).clone(), c.input("s", sa).clone()
                c.input("v_other", vb)
                c.input("s_other", sb)
                mask = None
                if masked:
                    mask = (torch.rand(seq, device=DEV) > 0.4).to(torch.uint8)
                    c.input("mask", mask)
                flashinfer.merge_state_in_place(v, s, vb, sb, mask)
                c.output("v", v, "ulp:1")
                c.output("s", s, "ulp:1")
                yield c

        # An empty state (log-sum-exp -inf) on one side of a pair weighs nothing.
        seq, heads, hd = 16, 8, 128
        va, vb = (torch.randn(seq, heads, hd, dtype=dt, device=DEV) for _ in range(2))
        sa, sb = lse(seq, heads), lse(seq, heads)
        sa[::3] = float("-inf")
        sb[1::4] = float("-inf")
        sb[::3] = 0.0  # keep one side finite wherever the other is empty
        sa[1::4] = 0.0
        c = Case(f"cascade_merge_state_empty_{tag}_{seq}x{heads}x{hd}", "cascade", op="merge_state",
                 dtype=tag, seq=seq, heads=heads, head_dim=hd)
        for n, t in (("v_a", va), ("s_a", sa), ("v_b", vb), ("s_b", sb)):
            c.input(n, t)
        v, s = flashinfer.merge_state(va, sa, vb, sb)
        c.output("v", v, "ulp:1")
        c.output("s", s, "ulp:1")
        yield c

        for seq, sets, heads, hd in STATES:
            v = torch.randn(seq, sets, heads, hd, dtype=dt, device=DEV)
            s = lse(seq, sets, heads)
            c = Case(f"cascade_merge_states_{tag}_{seq}x{sets}x{heads}x{hd}", "cascade", op="merge_states",
                     dtype=tag, seq=seq, sets=sets, heads=heads, head_dim=hd)
            c.input("v", v)
            c.input("s", s)
            vo, so = flashinfer.merge_states(v, s)
            c.output("v", vo, "ulp:1")
            c.output("s", so, "ulp:1")
            yield c
