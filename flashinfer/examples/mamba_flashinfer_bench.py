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
