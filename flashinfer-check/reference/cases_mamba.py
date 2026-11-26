# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.mamba.selective_state_update`: single- and multi-token Mamba-2 state updates.

Each output is a dot product over `dstate` of a recurrently updated state, so the output and
the states are matched under `close` rules of their element types. The int16 state is
block-scaled: its integers may land one apart where a value sits on a rounding boundary.
Every state pool starts random, and the whole pool is saved after the call, so that slots
which must stay untouched are checked too.
"""
import zlib

import torch

import flashinfer
from flashinfer.mamba import selective_state_update
from common import Case, refused

DEV = "cuda"
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
OUT_RULE = {"f16": "close:1e-3:2e-3", "bf16": "close:8e-3:1.6e-2", "f32": "close:1e-4:1e-4"}
# FlashInfer compiles one kernel per (types, dim, dstate, tokens): dim and dstate stay fixed.
SMALL = dict(nheads=8, dim=64, dstate=128, ngroups=2)
REAL = dict(nheads=32, dim=64, dstate=128, ngroups=8)
MID = dict(nheads=16, dim=64, dstate=128, ngroups=4)


def i32(v):
    return torch.tensor(v, dtype=torch.int32, device=DEV)


def _make(name, inp="bf16", st="bf16", shape=SMALL, batch=3, T=1, z=False, D=True, bias=True, softplus=True,
         idx=None, dst=None, varlen=None, accepted=False, istate=None, int16=False, no_update=False):
    """One call. `idx`: None, `"pad"`; `dst`: None or `"pad"`; `varlen`: a list of sequence lengths;
    `istate`: None, `"idx"` (own indices) or `"slot"` (the state slot's)."""
    nh, dim, ds, ng = (shape[k] for k in ("nheads", "dim", "dstate", "ngroups"))
    it, sdt = DTYPES[inp], DTYPES[st]
    g = torch.Generator(device=DEV)
    g.manual_seed(zlib.crc32(name.encode()))

    def rnd(*s, scale=1.0, dtype=torch.float32):
        return (torch.randn(*s, device=DEV, generator=g) * scale).to(dtype)

    nt = max(varlen) if varlen else T
    total = sum(varlen) if varlen else batch * T
    multi = T > 1 or varlen is not None
    # State slots: one per sequence, or one per (sequence, token) when the accepted-token column is read.
    cols = T if accepted else 1
    sources = batch * cols
    dcols = T if (dst and multi) else 1
    if idx or dst or accepted:
        cache = sources + (batch * dcols if dst else 0) + 3
    else:
        cache = batch
    meta = dict(inp=inp, st=st, batch=batch, tokens=nt, nheads=nh, dim=dim, dstate=ds, ngroups=ng,
                z=z, d=D, bias=bias, softplus=softplus, idx=bool(idx), dst=bool(dst), varlen=varlen is not None,
                accepted=accepted, istate=istate or "", int16=int16, no_update=no_update, pad=-1)
    c = Case(name, "mamba", **meta)

    if int16:
        state = torch.randint(-20000, 20000, (cache, nh, dim, ds), device=DEV, generator=g, dtype=torch.int32).to(
            torch.int16)
        scale = (torch.rand(cache, nh, dim, device=DEV, generator=g) * 1.0e-5 + 2.0e-5).float()
    else:
        state = rnd(cache, nh, dim, ds, scale=0.5, dtype=sdt)
    lead = (total,) if varlen else ((batch, T) if multi else (batch,))
    x = rnd(*lead, nh, dim, dtype=it)
    # Without softplus, dt must itself be positive so that exp(A dt) decays.
    dt_raw = rnd(*lead, nh, scale=0.7) if softplus else torch.rand(*lead, nh, device=DEV, generator=g) * 0.5 + 0.05
    A = -(torch.rand(nh, device=DEV, generator=g) + 0.1)
    Bm = rnd(*lead, ng, ds, scale=0.7, dtype=it)
    Cm = rnd(*lead, ng, ds, scale=0.7, dtype=it)
    # FlashInfer needs a D; a zero one is the same as none (the port is built without the D term).
    Dv = rnd(nh) if D else torch.zeros(nh, device=DEV)
    dtb = rnd(nh, scale=0.5) if bias else None
    zv = rnd(*lead, nh, dim, dtype=it) if z else None
    kw = {}

    c.input("state", state)
    if int16:
        kw["state_scale"] = c.input("state_scale", scale).clone()
        scale_in = kw["state_scale"]
    c.input("x", x)
    c.input("dt", dt_raw)
    c.input("A", A)
    c.input("B", Bm)
    c.input("C", Cm)
    if D:
        c.input("D", Dv)
    if bias:
        c.input("dt_bias", dtb)
    if z:
        c.input("z", zv)

    def perm(n, lo=0):
        return (torch.randperm(n, device=DEV, generator=g) + lo).to(torch.int32)

    if idx:
        if accepted:
            sidx = perm(sources).view(batch, T)
            # One pad in a column that is read, one in a column that is not.
            acc = (torch.arange(batch, device=DEV) % T + 1).to(torch.int32)
            if batch > 2:
                sidx[1, int(acc[1]) - 1] = -1
                sidx[2, T - 1 if int(acc[2]) < T else 0] = -1
        else:
            sidx = perm(sources)
            if batch > 1:
                sidx[1] = -1
        kw["state_batch_indices"] = c.input("indices", sidx)
        if accepted:
            kw["num_accepted_tokens"] = c.input("accepted", acc)
    elif accepted:
        raise ValueError("accepted tokens need indices")
    if dst:
        d = perm(batch * dcols, sources).view(batch, dcols) if (multi) else perm(batch, sources)
        flat = d.view(-1)
        if multi:
            # FlashInfer documents `pad_slot_id` only for `state_batch_indices`; its single-token path writes a
            # padded destination (to the pool's last slot), so a pad in `dst` is used only with several tokens.
            flat[2 % flat.numel()] = -1
        if not multi and idx and batch > 2:
            flat[0] = kw["state_batch_indices"][0]  # in place
        kw["dst_state_batch_indices"] = c.input("dst", d)
    if varlen is not None:
        cu = i32([0] + list(torch.tensor(varlen).cumsum(0)))
        kw["cu_seqlens"] = c.input("cu_seqlens", cu)
    if istate:
        # FlashInfer wants one scale slot per sequence, so the int16 cache has exactly `batch` slots.
        icache = (batch if int16 else batch + 2) if istate == "idx" else cache
        if int16:
            ibuf = torch.randint(-20000, 20000, (icache, nt, nh, dim, ds), device=DEV, generator=g, dtype=torch.int32)
            ibuf = ibuf.to(torch.int16)
        else:
            ibuf = rnd(icache, nt, nh, dim, ds, scale=0.5, dtype=sdt)
        ibuf = c.input("istate", ibuf).clone()
        kw["intermediate_states_buffer"] = ibuf
        if istate == "idx":
            ii = perm(icache)[:batch]
            kw["intermediate_state_indices"] = c.input("istate_indices", ii)
        if int16:
            isc = c.input("istate_scale", torch.rand(icache, nt, nh, dim, device=DEV, generator=g) * 1e-5 + 2e-5)
            isc = isc.clone()
            kw["intermediate_state_scales"] = isc
    if no_update:
        kw["disable_state_update"] = True
    if multi:
        kw["cache_steps"] = nt

    # Real FlashInfer takes `dt`, `A`, `D` and `dt_bias` with zero strides over `dim` (and `dstate`).
    dt_e = dt_raw[..., None].expand(*lead, nh, dim)
    A_e = A[:, None, None].expand(nh, dim, ds)
    D_e = Dv[:, None].expand(nh, dim)
    dtb_e = dtb[:, None].expand(nh, dim) if bias else None
    st_run = state.clone()
    out = torch.zeros_like(x)
    selective_state_update(st_run, x, dt_e, A_e, Bm, Cm, D_e, z=zv, dt_bias=dtb_e, dt_softplus=softplus, out=out, **kw)
    c.output("out", out, OUT_RULE[inp])
    c.output("state", st_run, "int:1" if int16 else OUT_RULE[st])
    if int16:
        c.output("state_scale", scale_in, "close:1e-4:1e-4")
    if istate:
        c.output("istate", kw["intermediate_states_buffer"], "int:1" if int16 else OUT_RULE[st])
        if int16:
            c.output("istate_scale", kw["intermediate_state_scales"], "close:1e-4:1e-4")
    return c


def make(name, *args, **kw):
    """`_make`, or None (recorded as refused) when FlashInfer cannot run the call on this GPU."""
    try:
        return _make(name, *args, **kw)
    except Exception as e:  # noqa: BLE001
        refused(name, e)
        torch.cuda.synchronize()
        return None


def cases():
    for c in _cases():
        if c is not None:
            yield c


def _cases():
    # Single token, every input and state type, on a realistic shape.
    for inp, st in (("f16", "f16"), ("bf16", "bf16"), ("bf16", "f32"), ("f16", "f32"), ("bf16", "f16"), ("f16", "bf16")):
        yield make(f"mamba_t1_{inp}_{st}_real", inp, st, REAL, batch=4, z=True)
    yield make("mamba_t1_bf16_bf16_batch1", "bf16", "bf16", REAL, batch=1)
    yield make("mamba_t1_f16_f32_bare", "f16", "f32", SMALL, batch=5, D=False, bias=False, softplus=False)
    yield make("mamba_t1_bf16_f32_z_nod", "bf16", "f32", SMALL, batch=5, z=True, D=False)
    yield make("mamba_t1_f16_f16_nobias", "f16", "f16", SMALL, batch=2, bias=False)
    yield make("mamba_t1_bf16_bf16_nosoftplus", "bf16", "bf16", SMALL, batch=4, softplus=False)
    yield make("mamba_t1_bf16_f32_pad", "bf16", "f32", MID, batch=5, idx="pad")
    yield make("mamba_t1_f16_f16_pad_z", "f16", "f16", SMALL, batch=4, idx="pad", z=True)
    yield make("mamba_t1_bf16_bf16_dst", "bf16", "bf16", MID, batch=5, idx="pad", dst="pad")
    yield make("mamba_t1_f16_f32_dst_noidx", "f16", "f32", SMALL, batch=4, dst="pad")
    yield make("mamba_t1_bf16_f16_noupdate", "bf16", "f16", SMALL, batch=3, no_update=True)
    yield make("mamba_t1_bf16_int16", "bf16", "f16", MID, batch=4, int16=True)
    yield make("mamba_t1_f16_int16_pad", "f16", "f16", SMALL, batch=4, int16=True, idx="pad", z=True)

    # Several tokens (FlashInfer's `cache_steps`).
    for inp, st in (("bf16", "bf16"), ("f16", "f16"), ("bf16", "f32")):
        yield make(f"mamba_t6_{inp}_{st}_real", inp, st, REAL, batch=3, T=6, z=True)
    yield make("mamba_t6_bf16_bf16_batch1", "bf16", "bf16", REAL, batch=1, T=6)
    yield make("mamba_t4_f16_f32_bare", "f16", "f32", SMALL, batch=4, T=4, D=False, bias=False, softplus=False)
    yield make("mamba_t4_bf16_bf16_nosoftplus_z", "bf16", "bf16", SMALL, batch=3, T=4, softplus=False, z=True)
    yield make("mamba_t4_bf16_f32_pad", "bf16", "f32", MID, batch=5, T=4, idx="pad")
    yield make("mamba_t4_f16_f16_dst", "f16", "f16", MID, batch=5, T=4, idx="pad", dst="pad")
    yield make("mamba_t4_bf16_bf16_dst_noidx", "bf16", "bf16", SMALL, batch=4, T=4, dst="pad")
    yield make("mamba_t4_bf16_f32_noupdate", "bf16", "f32", SMALL, batch=3, T=4, no_update=True)
    yield make("mamba_t4_bf16_bf16_accepted", "bf16", "bf16", MID, batch=5, T=4, idx="pad", accepted=True)
    yield make("mamba_t4_f16_f32_accepted", "f16", "f32", SMALL, batch=4, T=4, idx="pad", accepted=True)
    yield make("mamba_t4_bf16_f16_accepted_dst", "bf16", "f16", SMALL, batch=4, T=4, idx="pad", accepted=True, dst="pad")
    yield make("mamba_varlen_bf16_bf16", "bf16", "bf16", MID, batch=5, varlen=[3, 1, 4, 2, 4])
    yield make("mamba_varlen_f16_f32_z", "f16", "f32", SMALL, batch=4, varlen=[4, 2, 1, 4], z=True)
    yield make("mamba_varlen_bf16_f32_pad", "bf16", "f32", SMALL, batch=4, varlen=[2, 3, 1, 4], idx="pad")
    yield make("mamba_varlen_f16_f16_empty", "f16", "f16", SMALL, batch=4, varlen=[3, 0, 2, 4])
    yield make("mamba_varlen_bf16_bf16_batch1", "bf16", "bf16", REAL, batch=1, varlen=[6])

    # The intermediate-state cache: each token's state, with its own slot indices or the state slot's.
    yield make("mamba_t4_bf16_bf16_istate_idx", "bf16", "bf16", MID, batch=3, T=4, istate="idx")
    yield make("mamba_t4_f16_f32_istate_idx", "f16", "f32", SMALL, batch=4, T=4, istate="idx", z=True)
    yield make("mamba_t4_bf16_f16_istate_idx", "bf16", "f16", SMALL, batch=4, T=4, istate="idx")
    yield make("mamba_t4_bf16_bf16_istate_slot", "bf16", "bf16", SMALL, batch=4, T=4, istate="slot", idx="pad")
    yield make("mamba_t4_f16_f16_istate_noidx", "f16", "f16", SMALL, batch=4, T=4, istate="slot")
    yield make("mamba_varlen_bf16_f32_istate", "bf16", "f32", SMALL, batch=4, varlen=[2, 4, 1, 3], istate="idx")
    yield make("mamba_t4_bf16_f32_istate_noupdate", "bf16", "f32", SMALL, batch=3, T=4, istate="idx", no_update=True)

    # int16 block-scaled state over several tokens.
    yield make("mamba_t4_bf16_int16", "bf16", "f16", SMALL, batch=4, T=4, int16=True)
    yield make("mamba_t4_f16_int16_pad", "f16", "f16", MID, batch=4, T=4, int16=True, idx="pad", z=True)
    yield make("mamba_t4_bf16_int16_istate", "bf16", "f16", SMALL, batch=3, T=4, int16=True, istate="idx")
    yield make("mamba_varlen_bf16_int16", "bf16", "f16", SMALL, batch=4, varlen=[3, 1, 4, 2], int16=True)
