# FlashInfer recurrent_kda (Kimi Delta Attention decode) on the shapes of cute-rs's `kda` example
# and of FlashInfer's benchmarks/bench_recurrent_kda.py: 16 q/k heads and 32 v heads (and Kimi-Linear's
# 32/32), K = V = 128, bf16 q/k/v/g/beta, bf16 state, precomputed log-space gate, L2 norm in the
# kernel; T = 1 standard decode and T = 4 (num_spec_tokens = 3, per-token checkpoint slots).
# Timed from CUDA-graph replays (100 launches per graph). Compare with the Rust example's table.
import torch
import torch.nn.functional as F
import flashinfer
from flashinfer import recurrent_kda

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
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
K = 128
dt = torch.bfloat16
print("heads(q/v)  T  batch  time(us)    GB/s")
for H, HV in ((16, 32), (32, 32)):
    for T in (1, 4):
        for batch in (1, 16, 64, 128):
            q = torch.randn(batch, T, H, K, device=dev, dtype=dt)
            k = torch.randn(batch, T, H, K, device=dev, dtype=dt)
            v = torch.randn(batch, T, HV, K, device=dev, dtype=dt)
            g = F.logsigmoid(torch.randn(batch, T, HV, K, device=dev)).to(dt)
            beta = torch.randn(batch, T, HV, device=dev, dtype=dt).sigmoid()
            state = torch.randn(batch * T, HV, K, K, device=dev).to(dt)
            nsp = None if T == 1 else T - 1

            def fn():
                recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=state, scale=K ** -0.5,
                              use_qk_l2norm_in_kernel=True, num_spec_tokens=nsp)

            try:
                t = graph_bench(fn)
            except Exception as e:
                print(f"{H:3}/{HV:<3}   {T:3}  {batch:5}  failed: {type(e).__name__}: {str(e)[:200]}")
                continue
            sb = batch * HV * K * K * 2
            nbytes = sb * (1 + T) + batch * T * (2 * H * K + 3 * HV * K + HV) * 2
            print(f"{H:3}/{HV:<3}   {T:3}  {batch:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")

# ---- cu_seqlens packing and num_accepted_tokens (cute-rs's `varlen` rows) --------------------------
print("heads(q/v)  mode           batch  time(us)    GB/s")
H, HV = 16, 32
for mode, T in (("varlen T=1", 1), ("varlen T=4+acc", 4)):
    for batch in (1, 16, 64):
        q = torch.randn(1, batch * T, H, K, device=dev, dtype=dt)
        k = torch.randn(1, batch * T, H, K, device=dev, dtype=dt)
        v = torch.randn(1, batch * T, HV, K, device=dev, dtype=dt)
        g = F.logsigmoid(torch.randn(1, batch * T, HV, K, device=dev)).to(dt)
        beta = torch.randn(1, batch * T, HV, device=dev, dtype=dt).sigmoid()
        state = torch.randn(batch * T, HV, K, K, device=dev).to(dt)
        cu = torch.arange(0, batch * T + 1, T, dtype=torch.int32, device=dev)
        ssi = torch.arange(batch * T, dtype=torch.int32, device=dev).reshape(batch, T) if T > 1 else \
            torch.arange(batch, dtype=torch.int32, device=dev)
        acc = ((torch.arange(batch, device=dev) % T) + 1).to(torch.int32) if T > 1 else None
        nsp = None if T == 1 else T - 1

        def fn():
            recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=state, scale=K ** -0.5,
                          use_qk_l2norm_in_kernel=True, cu_seqlens=cu, ssm_state_indices=ssi,
                          num_spec_tokens=nsp, num_accepted_tokens=acc)

        try:
            t = graph_bench(fn)
        except Exception as e:
            print(f"{H:3}/{HV:<3}      {mode:14} {batch:5}  failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        sb = batch * HV * K * K * 2
        nbytes = sb * (1 + T) + batch * T * (2 * H * K + 3 * HV * K + HV) * 2
        print(f"{H:3}/{HV:<3}      {mode:14} {batch:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")
