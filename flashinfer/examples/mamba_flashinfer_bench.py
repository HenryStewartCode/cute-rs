# FlashInfer selective_state_update (single token, SM89 "simple" kernel) on the shapes of cute-rs's
# `mamba` example: nheads 128, dim 64, dstate 128, ngroups 8, bf16 inputs, f32 dt/A/D/dt_bias with
# dt_softplus, batch {1, 16, 128}. Timed from CUDA-graph replays (100 launches per graph).
import torch
import flashinfer
from flashinfer.mamba import selective_state_update

dev = "cuda"


def graph_bench(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(100):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = None
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        t = a.elapsed_time(b) * 1000 / 100
        best = t if best is None else min(best, t)
    return best


torch.manual_seed(0)
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
print("state  batch  time(us)    GB/s")
nheads, dim, dstate, ngroups = 128, 64, 128, 8
for sdt, name in ((torch.bfloat16, "bf16"), (torch.float16, "f16"), (torch.float32, "f32")):
    for batch in (1, 16, 128):
        state = torch.randn(batch, nheads, dim, dstate, device=dev).to(sdt)
        x = torch.randn(batch, nheads, dim, device=dev, dtype=torch.bfloat16)
        dt = torch.randn(batch, nheads, device=dev, dtype=torch.float32)[:, :, None].expand(batch, nheads, dim)
        A = (-torch.rand(nheads, device=dev) - 0.1)[:, None, None].expand(nheads, dim, dstate)
        B = torch.randn(batch, ngroups, dstate, device=dev, dtype=torch.bfloat16)
        C = torch.randn(batch, ngroups, dstate, device=dev, dtype=torch.bfloat16)
        D = torch.randn(nheads, device=dev)[:, None].expand(nheads, dim)
        dt_bias = torch.randn(nheads, device=dev)[:, None].expand(nheads, dim)
        out = torch.empty_like(x)
        fn = lambda: selective_state_update(state, x, dt, A, B, C, D, dt_bias=dt_bias, dt_softplus=True, out=out)
        t = graph_bench(fn)
        nbytes = 2 * state.numel() * state.element_size() + (2 * x.numel() + 2 * B.numel()) * 2
        print(f"{name:5}  {batch:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")

# Multi-token (cache_steps = 6, no intermediate states).
T = 6
for sdt, name in ((torch.bfloat16, "bf16"), (torch.float32, "f32")):
    for batch in (1, 16, 128):
        state = torch.randn(batch, nheads, dim, dstate, device=dev).to(sdt)
        x = torch.randn(batch, T, nheads, dim, device=dev, dtype=torch.bfloat16)
        dt = torch.randn(batch, T, nheads, device=dev, dtype=torch.float32)[..., None].expand(batch, T, nheads, dim)
        A = (-torch.rand(nheads, device=dev) - 0.1)[:, None, None].expand(nheads, dim, dstate)
        B = torch.randn(batch, T, ngroups, dstate, device=dev, dtype=torch.bfloat16)
        C = torch.randn(batch, T, ngroups, dstate, device=dev, dtype=torch.bfloat16)
        D = torch.randn(nheads, device=dev)[:, None].expand(nheads, dim)
        dt_bias = torch.randn(nheads, device=dev)[:, None].expand(nheads, dim)
        out = torch.empty_like(x)
        fn = lambda: selective_state_update(state, x, dt, A, B, C, D, dt_bias=dt_bias, dt_softplus=True, out=out, cache_steps=T)
        t = graph_bench(fn)
        nbytes = 2 * state.numel() * state.element_size() + (2 * x.numel() + 2 * B.numel()) * 2
        print(f"{name:5}  {batch:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}  (6 tokens)")

# The options beyond the plain update, one shape each (batch 16, 6 tokens where several, bf16
# state); matches the "option" table of the `mamba` example.
print("option           time(us)")
batch, T = 16, 6


def inputs(tokens, state_dtype=torch.bfloat16, cache=None, varlen_total=None):
    n = varlen_total if varlen_total is not None else batch * tokens
    shape = (n,) if varlen_total is not None else (batch, tokens)
    state = torch.randn(cache or batch, nheads, dim, dstate, device=dev).to(state_dtype)
    x = torch.randn(*shape, nheads, dim, device=dev, dtype=torch.bfloat16)
    dt = torch.randn(*shape, nheads, device=dev, dtype=torch.float32)[..., None].expand(*shape, nheads, dim)
    B = torch.randn(*shape, ngroups, dstate, device=dev, dtype=torch.bfloat16)
    C = torch.randn(*shape, ngroups, dstate, device=dev, dtype=torch.bfloat16)
    return state, x, dt, B, C


A = (-torch.rand(nheads, device=dev) - 0.1)[:, None, None].expand(nheads, dim, dstate)
D = torch.randn(nheads, device=dev)[:, None].expand(nheads, dim)
dt_bias = torch.randn(nheads, device=dev)[:, None].expand(nheads, dim)
common = dict(dt_softplus=True, dt_bias=dt_bias)
i32 = lambda v: torch.tensor(v, device=dev, dtype=torch.int32)

# dst, one token: read slot b, write slot batch + b.
state, x, dt, B, C = inputs(1, cache=2 * batch)
x, dt, B, C = x[:, 0], dt[:, 0], B[:, 0], C[:, 0]
out = torch.empty_like(x)
src, dst = torch.arange(batch, device=dev, dtype=torch.int32), torch.arange(batch, 2 * batch, device=dev, dtype=torch.int32)
t = graph_bench(lambda: selective_state_update(state, x, dt, A, B, C, D, state_batch_indices=src, dst_state_batch_indices=dst, out=out, **common))
print(f"{'dst (1 token)':15} {t:8.2f}")

# varlen: lengths cycling 1..6.
lens = [(b % T) + 1 for b in range(batch)]
cu = i32([0] + list(torch.tensor(lens).cumsum(0)))
state, x, dt, B, C = inputs(T, varlen_total=sum(lens))
out = torch.empty_like(x)
t = graph_bench(lambda: selective_state_update(state, x, dt, A, B, C, D, cu_seqlens=cu, cache_steps=T, out=out, **common))
print(f"{'varlen':15} {t:8.2f}")

# accepted tokens: 2D indices [batch, T], accepted cycling 1..6.
state, x, dt, B, C = inputs(T, cache=batch * T)
out = torch.empty_like(x)
idx = torch.arange(batch * T, device=dev, dtype=torch.int32).view(batch, T)
acc = i32([(b % T) + 1 for b in range(batch)])
t = graph_bench(lambda: selective_state_update(state, x, dt, A, B, C, D, state_batch_indices=idx, num_accepted_tokens=acc, cache_steps=T, out=out, **common))
print(f"{'accepted':15} {t:8.2f}")

# intermediate states: [cache, T, nheads, dim, dstate] with its own indices.
state, x, dt, B, C = inputs(T)
out = torch.empty_like(x)
istate = torch.zeros(batch, T, nheads, dim, dstate, device=dev, dtype=torch.bfloat16)
iidx = torch.arange(batch - 1, -1, -1, device=dev, dtype=torch.int32)
t = graph_bench(lambda: selective_state_update(state, x, dt, A, B, C, D, intermediate_states_buffer=istate, intermediate_state_indices=iidx, cache_steps=T, out=out, **common))
print(f"{'istate':15} {t:8.2f}")

# dst, 6 tokens: [batch, T] destinations after the sources.
state, x, dt, B, C = inputs(T, cache=batch + batch * T)
out = torch.empty_like(x)
src = torch.arange(batch, device=dev, dtype=torch.int32)
dst = torch.arange(batch, batch + batch * T, device=dev, dtype=torch.int32).view(batch, T)
t = graph_bench(lambda: selective_state_update(state, x, dt, A, B, C, D, state_batch_indices=src, dst_state_batch_indices=dst, cache_steps=T, out=out, **common))
print(f"{'dst (6 tokens)':15} {t:8.2f}")

# int16 state with per-row scales, one token.
state, x, dt, B, C = inputs(1)
x, dt, B, C = x[:, 0], dt[:, 0], B[:, 0], C[:, 0]
state = (state.float() * 1000).clamp(-32767, 32767).to(torch.int16)
scale = torch.full((batch, nheads, dim), 1e-3, device=dev)
out = torch.empty_like(x)
t = graph_bench(lambda: selective_state_update(state, x, dt, A, B, C, D, state_scale=scale, out=out, **common))
print(f"{'int16':15} {t:8.2f}")
