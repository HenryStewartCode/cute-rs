# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.gdn_decode` and `flashinfer.gdn_prefill`: Gated Delta Net decode (state `[V, K]` and `[K, V]`),
multi-token decode (`gated_delta_rule_mtp`, and the BFloat16 state path of
`gated_delta_rule_decode_pretranspose` for T > 1) and chunked prefill.

Outputs and states come from dot products over the 128-wide state plus a recurrent update, so
they are matched under `close` rules of their element types. State pools start random and are
saved whole, so slots that must stay untouched are checked too.
"""
import zlib

import torch

import flashinfer
from flashinfer.gdn_decode import gated_delta_rule_decode, gated_delta_rule_decode_pretranspose, gated_delta_rule_mtp
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from common import Case, refused

DEV = "cuda"
K = 128
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
RULE = {"f16": "close:1e-3:2e-3", "bf16": "close:8e-3:1.6e-2", "f32": "close:1e-4:1e-4"}
SMALL = dict(H=4, HV=8)
REAL = dict(H=16, HV=32)


def _gen(name):
    g = torch.Generator(device=DEV)
    g.manual_seed(zlib.crc32(name.encode()))
    return g


def _decode(name, fn="pretranspose", inp="bf16", st="f32", H=4, HV=8, batch=4, T=1, l2norm=True, pool=False, out_idx=False,
            pad=False, cache=0, no_update=False, scatter=False):
    """A decode call. `fn`: `pretranspose` (state `[V, K]`; one token, or several with a BFloat16 state), `kv`
    (state `[K, V]`) or `mtp` (`gated_delta_rule_mtp`, state pool). `pool`: the state is a pool addressed by indices;
    `cache`: steps of the intermediate-state cache; `scatter`: per-token pool slots."""
    it, sdt = DTYPES[inp], DTYPES[st]
    g = _gen(name)

    def rnd(*s, scale=1.0, dtype=torch.float32):
        return (torch.randn(*s, device=DEV, generator=g) * scale).to(dtype)

    qs = 1.0 if l2norm else 0.1
    # Pool slots: the read ones, the written ones (all distinct), and for scatter the per-token ones.
    reads = batch + 3 if pool else batch
    slots = reads
    if out_idx:
        slots = 2 * reads
    if scatter:
        slots = reads + batch * T
    scale = 1.0
    c = Case(name, "gdn", fn=fn, inp=inp, st=st, batch=batch, hq=H, hv=HV, tokens=T, l2norm=l2norm, pool=pool,
             out_idx=out_idx, cache=cache, no_update=no_update, scatter=scatter, scale=scale)
    q = rnd(batch, T, H, K, scale=qs, dtype=it)
    k = rnd(batch, T, H, K, scale=qs, dtype=it)
    v = rnd(batch, T, HV, K, dtype=it)
    a = rnd(batch, T, HV, scale=0.5, dtype=it)
    b = rnd(batch, T, HV, scale=1.5, dtype=it)
    A_log = rnd(HV, scale=0.3)
    dt_bias = rnd(HV, scale=0.3)
    state = rnd(slots, HV, K, K, scale=0.5, dtype=sdt)
    for n, t in (("state", state), ("q", q), ("k", k), ("v", v), ("a", a), ("b", b), ("A_log", A_log),
                 ("dt_bias", dt_bias)):
        c.input(n, t)
    idx = oidx = sidx = None
    if pool or fn == "mtp":
        # Reading slots are the first `reads`; written slots above them.
        perm = torch.randperm(reads, device=DEV, generator=g).to(torch.int32)[:batch]
        if pad and batch > 1:
            perm[1] = -1
        idx = c.input("indices", perm)
        if out_idx:
            oidx = c.input("out_indices", torch.where(perm < 0, perm, perm + reads))
        if scatter:
            sidx = c.input("ssm_indices", (reads + torch.randperm(batch * T, device=DEV, generator=g)).to(
                torch.int32).view(batch, T))
    st_run = state.clone()
    out = torch.zeros(batch, T, HV, K, dtype=it, device=DEV)
    kw = {}
    ibuf = None
    if cache:
        ibuf = c.input("cache", rnd(batch, cache, HV, K, K, scale=0.5, dtype=sdt)).clone()
        kw["intermediate_states_buffer"] = ibuf
    if fn == "pretranspose":
        if no_update:
            kw["disable_state_update"] = True
        if pool:
            if out_idx:
                kw["output_state_indices"] = oidx
            gated_delta_rule_decode_pretranspose(q=q, k=k, v=v, state=None, A_log=A_log, a=a, dt_bias=dt_bias, b=b,
                                                 scale=scale, output=out, use_qk_l2norm=l2norm, initial_state=st_run,
                                                 initial_state_indices=idx, **kw)
        else:
            gated_delta_rule_decode_pretranspose(q=q, k=k, v=v, state=st_run, A_log=A_log, a=a, dt_bias=dt_bias, b=b,
                                                 scale=scale, output=out, use_qk_l2norm=l2norm, **kw)
    elif fn == "kv":
        gated_delta_rule_decode(q=q, k=k, v=v, state=st_run, A_log=A_log, a=a, dt_bias=dt_bias, b=b, scale=scale,
                                output=out, use_qk_l2norm=l2norm, backend="flashinfer")
    else:
        if oidx is not None:
            kw["output_state_indices"] = oidx
        if sidx is not None:
            kw["ssm_state_indices"] = sidx
        gated_delta_rule_mtp(q, k, v, st_run, idx, A_log, a, dt_bias, b, scale, out, disable_state_update=no_update,
                             use_qk_l2norm=l2norm, **kw)
    # FlashInfer's Float16-input decode paths cast q, k, v (and a, b) to BFloat16 before the kernel
    # (`as_bf16` in gdn_kernels/gdn_decode_pretranspose.py:1033, gdn_decode_bf16_state.py and gdn_decode_mtp.py:3068)
    # and write BFloat16 outputs, so those references are only BFloat16-accurate. The [K, V] path does not cast.
    lossy = inp == "f16" and fn != "kv"
    orule, srule = RULE["bf16" if lossy else inp], RULE["bf16" if lossy else st]
    c.output("out", out, orule)
    c.output("state", st_run, srule)
    if cache:
        c.output("cache", ibuf, srule)
    return c


def _prefill(name, seqlens, inp="bf16", st="f32", H=4, HV=8, l2norm=True, initial=True, pool=False, out_idx=False,
             pad=False):
    """`chunk_gated_delta_rule` over packed sequences: `pool` reads and writes rows of state pools through indices."""
    it, sdt = DTYPES[inp], DTYPES[st]
    g = _gen(name)

    def rnd(*s, lo=None, hi=None, scale=1.0, dtype=torch.float32):
        if lo is not None:
            return (torch.rand(*s, device=DEV, generator=g) * (hi - lo) + lo).to(dtype)
        return (torch.randn(*s, device=DEV, generator=g) * scale).to(dtype)

    # FlashInfer promises `-1` only with `output_state_indices` and an explicit `output_state` pool.
    assert not pad or out_idx
    n, total = len(seqlens), sum(seqlens)
    reads = n + 3 if pool else n
    slots = 2 * reads if out_idx else reads
    scale = 1.0 if l2norm else 0.1
    c = Case(name, "gdn", fn="prefill", inp=inp, st=st, hq=H, hv=HV, l2norm=l2norm, initial=initial, pool=pool,
             out_idx=out_idx, scale=scale)
    # Without the L2 norm, keys of about unit norm keep the recurrence stable.
    q = rnd(total, H, K, lo=-1.0, hi=1.0, dtype=it)
    k = rnd(total, H, K, lo=-1.0 if l2norm else -0.15, hi=1.0 if l2norm else 0.15, dtype=it)
    v = rnd(total, HV, K, lo=-1.0, hi=1.0, dtype=it)
    alpha = rnd(total, HV, lo=0.85, hi=1.0)
    beta = rnd(total, HV, lo=0.0, hi=1.0)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), dtype=torch.int32, device=DEV)
    for nm, t in (("q", q), ("k", k), ("v", v), ("g", alpha), ("beta", beta), ("cu_seqlens", cu)):
        c.input(nm, t)
    kw = {}
    if initial:
        s0 = c.input("state", rnd(slots, HV, K, K, lo=-0.5, hi=0.5, dtype=sdt))
    else:
        s0 = None
    if pool:
        perm = torch.randperm(reads, device=DEV, generator=g).to(torch.int32)[:n]
        if pad and n > 1:
            perm[1] = -1  # starts from the zero state
        kw["state_indices"] = c.input("indices", perm)
        if out_idx:
            oi = (reads + torch.randperm(reads, device=DEV, generator=g)[:n]).to(torch.int32)
            if pad and n > 2:
                oi[2] = -1  # skips the final write
            kw["output_state_indices"] = c.input("out_indices", oi)
    final = c.input("state_out", rnd(slots, HV, K, K, lo=-0.5, hi=0.5, dtype=sdt)).clone()
    out = torch.zeros(total, HV, K, dtype=it, device=DEV)
    chunk_gated_delta_rule(q, k, v, alpha, beta, scale, s0.clone() if s0 is not None else None, True, cu, l2norm,
                           output=out, output_state=final, use_cp=False, **kw)
    c.output("out", out, RULE[inp])
    # Both kernels multiply with 16-bit operands, so the final state is no closer than the input type allows:
    # the looser of the input's and the state's rule.
    order = ["f32", "f16", "bf16"]
    c.output("state_out", final, RULE[max(inp, st, key=order.index)])
    return c


def _safe(fn):
    def run(name, *args, **kw):
        try:
            return fn(name, *args, **kw)
        except Exception as e:  # noqa: BLE001
            refused(name, e)
            torch.cuda.synchronize()
            return None

    return run


decode = _safe(_decode)
prefill = _safe(_prefill)


def cases():
    for c in _cases():
        if c is not None:
            yield c


def _cases():
    # gated_delta_rule_decode_pretranspose, one token: Float32 state (the legacy kernel).
    yield decode("gdn_pre_t1_bf16_f32_real", **REAL, batch=8)
    yield decode("gdn_pre_t1_f16_f32", inp="f16", batch=3)
    yield decode("gdn_pre_t1_bf16_f32_nonorm", batch=3, l2norm=False)
    yield decode("gdn_pre_t1_bf16_f32_batch1", H=8, HV=16, batch=1)
    # BFloat16 state, in place and as a pool (read slots, written slots).
    yield decode("gdn_pre_t1_bf16_bf16_real", st="bf16", **REAL, batch=8)
    yield decode("gdn_pre_t1_f16_bf16", inp="f16", st="bf16", batch=3)
    yield decode("gdn_pre_t1_bf16_bf16_batch1", st="bf16", batch=1)
    yield decode("gdn_pre_t1_bf16_bf16_pool", st="bf16", batch=5, pool=True)
    yield decode("gdn_pre_t1_bf16_bf16_pool_out", st="bf16", batch=5, pool=True, out_idx=True)
    yield decode("gdn_pre_t1_f16_bf16_pool_nonorm", inp="f16", st="bf16", batch=4, pool=True, l2norm=False)
    # Float16 state, and a Float32 pool: probably outside what FlashInfer takes.
    yield decode("gdn_pre_t1_bf16_f16", st="f16", batch=3)
    yield decode("gdn_pre_t1_bf16_f32_pool", st="f32", batch=3, pool=True)
    # The non-transposed [K, V] state (Float32 only).
    yield decode("gdn_kv_t1_bf16_f32_real", fn="kv", **REAL, batch=4)
    yield decode("gdn_kv_t1_f16_f32", fn="kv", inp="f16", batch=3)
    yield decode("gdn_kv_t1_bf16_f32_nonorm", fn="kv", batch=3, l2norm=False)
    yield decode("gdn_kv_t1_bf16_f32_batch1", fn="kv", batch=1)
    # Several tokens through the BFloat16 state path of decode_pretranspose.
    yield decode("gdn_pre_t4_bf16_bf16", st="bf16", batch=3, T=4)
    yield decode("gdn_pre_t2_f16_bf16", inp="f16", st="bf16", batch=3, T=2)
    yield decode("gdn_pre_t4_bf16_bf16_pool", st="bf16", batch=5, T=4, pool=True)
    yield decode("gdn_pre_t4_bf16_bf16_noupdate", st="bf16", batch=3, T=4, no_update=True)
    yield decode("gdn_pre_t4_bf16_bf16_cache", st="bf16", batch=3, T=4, cache=4)
    # gated_delta_rule_mtp: a Float32 pool, T tokens.
    yield decode("gdn_mtp_t4_bf16_f32_real", fn="mtp", **REAL, batch=3, T=4, pool=True)
    yield decode("gdn_mtp_t2_f16_f32", fn="mtp", inp="f16", batch=4, T=2, pool=True)
    yield decode("gdn_mtp_t6_bf16_f32_nonorm", fn="mtp", batch=3, T=6, pool=True, l2norm=False)
    yield decode("gdn_mtp_t4_bf16_f32_pad", fn="mtp", batch=5, T=4, pool=True, pad=True)
    yield decode("gdn_mtp_t4_bf16_f32_out", fn="mtp", batch=5, T=4, pool=True, out_idx=True)
    yield decode("gdn_mtp_t4_bf16_f32_out_pad", fn="mtp", batch=5, T=4, pool=True, out_idx=True, pad=True)
    yield decode("gdn_mtp_t4_bf16_f32_batch1", fn="mtp", batch=1, T=4, pool=True)
    yield decode("gdn_mtp_t4_bf16_f32_noupdate", fn="mtp", batch=3, T=4, pool=True, no_update=True)
    # The intermediate-state cache: FlashInfer rejects one with exactly T steps (a stride check), so most of
    # these have two more steps than tokens; the T-step one is kept to show that.
    yield decode("gdn_mtp_t4_bf16_f32_cache_t_steps", fn="mtp", batch=3, T=4, pool=True, cache=4)
    yield decode("gdn_mtp_t4_bf16_f32_cache", fn="mtp", batch=3, T=4, pool=True, cache=6)
    yield decode("gdn_mtp_t4_f16_f32_cache", fn="mtp", inp="f16", batch=3, T=4, pool=True, cache=6)
    yield decode("gdn_mtp_t2_bf16_f32_cache", fn="mtp", batch=4, T=2, pool=True, cache=4)
    yield decode("gdn_mtp_t4_bf16_f32_cache_pad", fn="mtp", batch=4, T=4, pool=True, cache=6, pad=True)
    yield decode("gdn_mtp_t4_bf16_f32_cache_out", fn="mtp", batch=4, T=4, pool=True, cache=6, out_idx=True)
    # The per-token pool scatter.
    yield decode("gdn_mtp_t4_bf16_f32_scatter", fn="mtp", batch=4, T=4, pool=True, scatter=True)
    yield decode("gdn_mtp_t2_f16_f32_scatter", fn="mtp", inp="f16", batch=5, T=2, pool=True, scatter=True)
    yield decode("gdn_mtp_t4_bf16_f32_scatter_pad", fn="mtp", batch=4, T=4, pool=True, scatter=True, pad=True)
    # Chunked prefill: a few hundred to a few thousand tokens, ragged lengths, states in every type.
    yield prefill("gdn_prefill_bf16_f32_1k", [1000], **SMALL)
    yield prefill("gdn_prefill_bf16_f32_ragged", [300, 45, 1, 512, 130], **SMALL)
    yield prefill("gdn_prefill_bf16_f32_real", [1024, 600, 424], **REAL)
    yield prefill("gdn_prefill_f16_f32", [257, 64, 500], inp="f16", **SMALL)
    yield prefill("gdn_prefill_bf16_bf16", [300, 129, 77], st="bf16", **SMALL)
    yield prefill("gdn_prefill_bf16_f16", [300, 129, 77], st="f16", **SMALL)
    yield prefill("gdn_prefill_f16_f16", [200, 63], inp="f16", st="f16", **SMALL)
    yield prefill("gdn_prefill_bf16_f32_nonorm", [400, 100], l2norm=False, **SMALL)
    yield prefill("gdn_prefill_bf16_f32_nostate", [350, 65], initial=False, **SMALL)
    yield prefill("gdn_prefill_bf16_f32_short", [1, 3, 17], **SMALL)
    yield prefill("gdn_prefill_bf16_f32_pool", [200, 90, 130], pool=True, **SMALL)
    yield prefill("gdn_prefill_bf16_f32_pool_out", [200, 90, 130], pool=True, out_idx=True, **SMALL)
    yield prefill("gdn_prefill_bf16_f32_pool_pad", [200, 90, 130], pool=True, out_idx=True, pad=True, **SMALL)
    yield prefill("gdn_prefill_bf16_bf16_pool_out", [150, 70], st="bf16", pool=True, out_idx=True, **SMALL)
    # Last: the verify call with a cache, which the port's notes say fails on this GPU (a failure may leave
    # the context unusable, so nothing follows).
    yield decode("gdn_mtp_t4_bf16_f32_verify_cache", fn="mtp", batch=3, T=4, pool=True, cache=6, no_update=True)
    yield decode("gdn_mtp_t4_bf16_f32_verify_cache_t_steps", fn="mtp", batch=3, T=4, pool=True, cache=4, no_update=True)
