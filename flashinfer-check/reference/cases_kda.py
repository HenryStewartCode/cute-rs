# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.recurrent_kda`: Kimi Delta Attention decode, one token or several.

The output and the state come from dot products over the 128-wide state plus a recurrent
update, so both are matched under `close` rules of their element types. Every state pool starts
random and is saved whole after the call, so slots that must stay untouched are checked too.
"""
import zlib

import torch
import torch.nn.functional as F

import flashinfer
from common import Case, refused

DEV = "cuda"
K = 128
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
RULE = {"f16": "close:1e-3:2e-3", "bf16": "close:8e-3:1.6e-2", "f32": "close:1e-4:1e-4"}


def _make(name, inp="bf16", st="bf16", H=4, HV=8, batch=4, T=1, gate="pre", bias=True, l2norm=True, logit=False,
          idx=False, pad=False, per_token=False, accepted=False, varlen=None):
    """One call. `gate`: `pre` (log-space input), `softplus` or `lower_bound` (in the kernel).
    `idx`: a state pool addressed by `ssm_state_indices` (`per_token`: `[batch, T]` checkpoint slots,
    else `[batch]`); `varlen`: sequence lengths of a packed call; `T` is then the longest."""
    it, sdt = DTYPES[inp], DTYPES[st]
    g = torch.Generator(device=DEV)
    g.manual_seed(zlib.crc32(name.encode()))

    def rnd(*s, scale=1.0, dtype=torch.float32):
        return (torch.randn(*s, device=DEV, generator=g) * scale).to(dtype)

    total = sum(varlen) if varlen else batch * T
    spec = T > 1
    # Pool slots: the read ones, and the written ones for checkpoints (all distinct).
    reads = batch * T if per_token else batch
    slots = reads + 3 if (idx or per_token) else (batch * T if spec else reads)
    scale = 1.0 if l2norm else 0.1
    meta = dict(inp=inp, st=st, batch=batch, hq=H, hv=HV, tokens=T, gate=gate, bias=bias, l2norm=l2norm, logit=logit,
                idx=idx, per_token=per_token, accepted=accepted, varlen=varlen is not None, scale=scale,
                lower_bound=-5.0)
    c = Case(name, "kda", **meta)

    lead = (1, total) if varlen else (batch, T)
    q = rnd(*lead, H, K, scale=1.0 if l2norm else 0.1, dtype=it)
    k = rnd(*lead, H, K, scale=1.0 if l2norm else 0.1, dtype=it)
    v = rnd(*lead, HV, K, dtype=it)
    if gate == "pre":
        gv = F.logsigmoid(rnd(*lead, HV, K)).to(it)
    else:
        gv = rnd(*lead, HV, K, dtype=it)
    beta_raw = rnd(*lead, HV)
    beta = (beta_raw if logit else beta_raw.sigmoid()).to(it)
    A_log = rnd(H, scale=0.3) if gate != "pre" else None
    dt_bias = rnd(H * K, scale=0.3) if (gate != "pre" and bias) else None
    state = rnd(slots, HV, K, K, scale=0.5, dtype=sdt)
    c.input("state", state)
    for n, t in (("q", q), ("k", k), ("v", v), ("g", gv), ("beta", beta)):
        c.input(n, t)
    if A_log is not None:
        c.input("A_log", A_log)
    if dt_bias is not None:
        c.input("dt_bias", dt_bias)

    kw = {}
    if gate != "pre":
        kw.update(use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias)
        if gate == "lower_bound":
            kw["lower_bound"] = -5.0
    if logit:
        kw["beta_is_logit"] = True
    if varlen is not None:
        kw["cu_seqlens"] = c.input("cu_seqlens", torch.tensor([0] + list(torch.tensor(varlen).cumsum(0)),
                                                              dtype=torch.int32, device=DEV))
    if idx or per_token:
        perm = torch.randperm(slots, device=DEV, generator=g)[:reads].to(torch.int32)
        sidx = perm.view(batch, T) if per_token else perm
        if pad and batch > 1:
            # A pad in a column that is read is undefined in FlashInfer (its documented promise covers a padded
            # row, not a padded read slot), so checkpoint pads sit only in the last column, after every column
            # that is read. (A pad before the read column makes FlashInfer's later tokens differ from the port's:
            # not a documented case either.)
            if per_token:
                sidx[1, T - 1] = -1
                if accepted and batch > 2:
                    sidx[2, T - 1] = -1
            else:
                sidx[1] = -1
        kw["ssm_state_indices"] = c.input("indices", sidx)
    if spec:
        kw["num_spec_tokens"] = T - 1
    if accepted:
        acc = torch.tensor([(i % (T + 1)) for i in range(batch)], dtype=torch.int32, device=DEV)
        acc[0] = T + 2  # above the last checkpoint: clamped
        kw["num_accepted_tokens"] = c.input("accepted", acc)

    st_run = state.clone()
    out = torch.zeros(*lead, HV, K, dtype=it, device=DEV)
    res, _ = flashinfer.recurrent_kda(q=q, k=k, v=v, g=gv, beta=beta, scale=scale, initial_state=st_run,
                                      use_qk_l2norm_in_kernel=l2norm, output=out, **kw)
    if varlen and res.data_ptr() != out.data_ptr():
        # With sequences of unequal length FlashInfer ignores `output` and returns its own `[1, batch * T, ...]`
        # tensor, whose first `total` rows are the packed outputs (the rest is unwritten).
        out = res[:, :total].contiguous()
    if idx and not per_token and pad and batch > 1:
        # The output of a padded row (state index -1) is not a defined value in FlashInfer: it documents zero on
        # one path and out-of-pool access on another, and here it computed from another slot. Assert the documented
        # behaviour (zero output, state untouched) for the port; the whole state pool is still compared.
        out.view(-1, HV, K)[kw["ssm_state_indices"] < 0] = 0
    c.output("out", out, RULE[inp])
    c.output("state", st_run, RULE[st])
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
    # One token: input and state types, on Kimi-Linear-like heads (grouped values).
    yield make("kda_t1_bf16_bf16_real", H=16, HV=32, batch=8)
    yield make("kda_t1_bf16_f32_real", st="f32", H=16, HV=32, batch=8)
    yield make("kda_t1_bf16_f16", st="f16", batch=4)
    yield make("kda_t1_f16_f16", inp="f16", st="f16", batch=4)
    yield make("kda_t1_f16_f32", inp="f16", st="f32", batch=4)
    yield make("kda_t1_f16_bf16", inp="f16", st="bf16", batch=4)
    yield make("kda_t1_bf16_bf16_equalheads", H=8, HV=8, batch=1)
    yield make("kda_t1_bf16_bf16_nonorm", batch=3, l2norm=False)
    yield make("kda_t1_bf16_f32_logit", st="f32", batch=3, logit=True)
    # In-kernel gates.
    yield make("kda_t1_bf16_bf16_softplus", gate="softplus", batch=4)
    yield make("kda_t1_bf16_f32_softplus_nobias", st="f32", gate="softplus", bias=False, batch=3)
    yield make("kda_t1_bf16_bf16_lowerbound", gate="lower_bound", batch=4)
    yield make("kda_t1_bf16_f32_lowerbound_nobias_nonorm", st="f32", gate="lower_bound", bias=False, l2norm=False,
               batch=3)
    yield make("kda_t1_f16_f16_softplus", inp="f16", st="f16", gate="softplus", batch=3)
    # A state pool with a pad slot.
    yield make("kda_t1_bf16_bf16_pool", batch=6, idx=True)
    yield make("kda_t1_bf16_f32_pool_pad", st="f32", H=16, HV=32, batch=6, idx=True, pad=True)
    yield make("kda_t1_bf16_bf16_pool_pad_softplus", gate="softplus", batch=5, idx=True, pad=True)
    # Several tokens (`num_spec_tokens = T - 1`) with the checkpoint slots implied (`B * T`), then given.
    yield make("kda_t4_bf16_bf16", batch=3, T=4)
    yield make("kda_t4_bf16_f32", st="f32", batch=3, T=4)
    yield make("kda_t6_bf16_bf16_real", H=8, HV=16, batch=4, T=6)
    yield make("kda_t4_bf16_bf16_checkpoints", batch=4, T=4, per_token=True)
    yield make("kda_t4_bf16_f32_checkpoints_pad", st="f32", batch=4, T=4, per_token=True, pad=True)
    yield make("kda_t4_bf16_bf16_softplus_checkpoints", gate="softplus", batch=3, T=4, per_token=True)
    yield make("kda_t4_bf16_bf16_lowerbound_nonorm", gate="lower_bound", l2norm=False, batch=3, T=4, per_token=True)
    yield make("kda_t4_bf16_bf16_batch1", batch=1, T=4, per_token=True)
    yield make("kda_t4_f16_f32_checkpoints", inp="f16", st="f32", batch=3, T=4, per_token=True)
    # Packed sequences, one token each and ragged, with and without accepted tokens.
    yield make("kda_varlen_t1_bf16_bf16", batch=6, varlen=[1] * 6)
    yield make("kda_varlen_t1_bf16_f32_pool_pad", st="f32", batch=6, varlen=[1] * 6, idx=True, pad=True)
    yield make("kda_varlen_t4_bf16_bf16_ragged", batch=4, T=4, varlen=[4, 2, 1, 3], per_token=True)
    yield make("kda_varlen_t4_bf16_f32_equal", st="f32", batch=3, T=4, varlen=[4, 4, 4], per_token=True)
    yield make("kda_accepted_t4_bf16_bf16", batch=5, T=4, per_token=True, accepted=True)
    yield make("kda_accepted_t4_bf16_f32_pad", st="f32", batch=5, T=4, per_token=True, accepted=True, pad=True)
    yield make("kda_accepted_varlen_t4_bf16_bf16", batch=5, T=4, varlen=[4] * 5, per_token=True, accepted=True)
    yield make("kda_accepted_varlen_t4_softplus", gate="softplus", batch=4, T=4, varlen=[4] * 4, per_token=True,
               accepted=True)
